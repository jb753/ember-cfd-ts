# Port TS4 (Turbostream 4) as `ember.plugins.ts.TS4Solver`

## Context

`ember-cfd-ts` already ports Turbostream 3 as `TS3Solver`, a from-scratch
reimplementation against `ember`'s own `Grid`/`Block`/`patch` API (there is no
dependency on the old `turbigen` package anywhere in this repo). Turbigen also
has a TS4 solver (`turbigen/src/turbigen/solvers/ts4.py`) written against its
*old*, pre-ember API. We want the same kind of port for TS4, reusing as much
of the already-ported TS3 machinery and existing core-`ember` capability as
possible rather than re-transcribing turbigen's code.

Two things investigated in this session change the shape of the port versus
turbigen's implementation:

1. **`Grid.set_primitive_cart_unstr` / `set_conserved_cart_unstr`** (core
   `ember/grid.py:518,635`) already do exactly what turbigen's hand-rolled
   `_read_flow` KDTree/rotation code did: align an unstructured Cartesian
   point cloud (auto-detecting axis permutation/sign) back onto the
   structured grid's own nodes and set the flow field. This removes the need
   for any bespoke remap code on read-back.
2. **The TS3->TS4 mesh conversion is not portable.** `convert_ts3_to_ts4_native.py`
   is glue code around Turbostream Ltd's own proprietary `ts.process`/VTK
   stack (`ts.process.ts3_reader`, `vtk_adaptor.create_ts_grid`, ...) -- it has
   no ember/turbigen API dependency at all, so it can be vendored into this
   plugin almost verbatim, but it must remain an external subprocess call
   (sourced via `environment_script`, same as running `turbostream4` itself).
   It cannot be reimplemented natively.

Per direction from the user, this port also **drops `nstep_ts3`**: turbigen's
TS4 solver could optionally run a short TS3 solve first as an initial guess.
`TS4Solver` won't do that -- like `TS3Solver`, it takes a `grid` that the
caller has already brought to a valid initial state (flow field, wdist, etc.)
and just writes it out as the TS3-format mesh/BC/flow file that the converter
consumes. No TS3 *execution* happens inside `TS4Solver`.

Real-gas tables, the `custom_pipeline` (paraview) conversion route, and
unsteady forcing are out of scope for this first port, matching the level
`TS3Solver` itself is currently scoped at (`PerfectFluid` only, no unsteady).
GPU/SLURM launch mechanics are kept, but simplified to the same "1 GPU,
serial" shape `TS3Solver._execute` uses today -- multi-GPU/SLURM can follow
later.

## Pipeline

```
Grid (caller-supplied, already has a valid flow field + wdist)
  |  _write_ts3_mesh: TS3Writer.get(grid, strict=True) / .write()  [reuse _io.py, no execution]
  |  _write_config_ofp: DEFAULT_CONFIG + typed fields + grid-derived gas properties
  v
TS3-format mesh/flow hdf5 + config.ofp
  |  vendored convert_ts3_to_ts4.py (subprocess, sources environment_script)
  v
TS4 input hdf5 (unstructured node arrays)
  |  _write_wall_distance  (new, no KDTree -- see "Core ember prerequisite" below)
  |  _write_throttle       (grid.patches.outlet -> OutletPatch.K_pid/mdot_target)
  |  _write_probes_ofp / point+logical probes
  v
run turbostream4 (subprocess, mpirun; TS4 aborts itself on NAN)
  |
  v
TS4 output hdf5 (unstructured node arrays: ro, rovx/y/z, roe, turb0)
  |  P from (rho, u) via the shared TS datum helper [see precursor step]
  |  grid.set_primitive_cart_unstr(xyz, primitive_cart)   [reuse core ember]
  |  block.set_mu_turb(turb0)
  v
Grid, updated in place
```

## Core `ember` prerequisite: wall-distance transfer

Turbigen's `_write_wall_distance` recomputes wall distance from scratch
against the TS4 unstructured cloud (a fresh KDTree of wall nodes, queried at
the TS4 node coordinates). That's not necessary here: the TS3->TS4 conversion
is a pure format/connectivity translation, not a remesh, so a freshly
converted TS4 *input* file's `node_array_x/y/z` sit at exactly the same
physical points as `block.x/y/z` -- the same exact-coincidence property that
already makes `set_primitive_cart_unstr` work on the *read* side (via
`Grid._align_cartesian`, `grid.py:330`, which builds the KDTree, permutation
search, and `distance_upper_bound=atol` exact match).

So instead of computing wall distance a second, independent way, `wdist` can
be **transplanted**: `block.wdist` is already correct (periodicity/pitch-tiling
included, via `Grid.calculate_wdist`) on the structured grid before TS4 ever
sees it. Run the same alignment used for read-back once, immediately after
conversion, and scatter `block.wdist` into `node_array_wdist` using the
resulting index correspondence.

### The node-count invariant

`Grid._align_cartesian` opens with `assert xyz.shape[0] == self.size`, and
`Grid.size` (`grid.py:1950`) is `sum(block.size for block in self)` -- it
counts nodes on *both* sides of every block-to-block interface. So the whole
design (read-back *and* wdist transfer) rests on the vendor converter emitting
one TS4 node per structured grid node, with interface duplicates preserved
rather than merged.

**Per direction from the user, that is the invariant we hold the vendor code
to.** A TS4 input file whose node count differs from `grid.size` is a vendor
bug, not a case to accommodate: there is no fallback path and no resampling
support. What the port must do is *fail loudly and legibly* rather than
propagate silent corruption:

- `_write_wall_distance` checks `len(node_array_x) == grid.size` up front and
  raises a plugin-specific error naming both counts and pointing at the
  converter, before any alignment work. The bare `assert` inside
  `_align_cartesian` is not sufficient: asserts vanish under `-O` and the
  message would blame ember rather than the converter.
### Gather, not scatter (correction found during implementation)

An earlier draft of this plan specified the transfer as a *scatter*: walk the
blocks and write each grid node's `wdist` into its TS4 index, using the
`block_indices` from `align_cart_unstr`. **That does not work, and the reason
is the very invariant above.** A matching block interface is represented on
both sides, so both the grid and the TS4 file hold two exactly coincident nodes
there. A nearest-neighbour query cannot distinguish them: both blocks' interface
nodes resolve to the same TS4 index, and the other TS4 node is never written.
Measured on a two-block grid with a shared face, 20 of 120 nodes were left
unwritten. Every real multi-block case is affected.

The transfer is therefore a **gather**: each TS4 node takes the `wdist` of the
grid node it sits on. This is well-defined for duplicated nodes precisely
because wall distance is a function of position alone, so coincident nodes must
receive the same value anyway. Only `perm`/`signs` are used from
`align_cart_unstr`; the `block_indices` it also returns are the scatter map.

```python
n_ts4 = f["node_array_x"].shape[0]
if n_ts4 != grid.size:
    raise TS4ConversionError(...)  # names both counts and the converter

perm, signs, _ = grid.align_cart_unstr(xyz_ts4_input)
xyz_aligned = xyz_ts4_input[:, list(perm)] * np.asarray(signs)

grid_xyz = np.concatenate([...b.x/b.y/b.z...])
wdist_grid = np.concatenate([b.wdist.ravel() for b in grid])

atol = 1e-6 * np.max(np.ptp(grid_xyz, axis=0))   # same rule as core ember
distance, indices = KDTree(grid_xyz).query(xyz_aligned)
if (distance > atol).any():
    raise TS4ConversionError(...)  # the mesh was resampled, not translated
f["node_array_wdist"][:] = wdist_grid[indices]
```

This costs a `scipy` dependency (already an ember dependency transitively, now
declared directly). Note the read direction is unaffected: `_read_flow` and
`set_primitive_cart_unstr` both *gather* into the grid, which duplicated nodes
never break.

A test with a genuinely matching interface guards this, since the disjoint
fixtures used elsewhere cannot reach the case.

This needs one small addition to core `ember` (sibling repo
`/home/jb753/python/ember`): a public wrapper around `Grid._align_cartesian`
(e.g. `Grid.align_cart_unstr(xyz)`, returning `(perm, signs, block_indices)`)
-- it's already computed internally by `set_primitive_cart_unstr`/
`set_conserved_cart_unstr`, just not exposed. This is a smaller, more general
addition than a bespoke wall-node accessor: one alignment mechanism now
serves both directions (flow fields in on read, `wdist` out on write) instead
of two independently-implemented geometric algorithms that would each have to
agree with `calculate_wdist`.

Caveat: this relies on exact node coincidence, which only holds for the
native conversion route (not `custom_pipeline`/paraview, which may resample
the mesh) -- fine here since `custom_pipeline` is already out of scope for
this port.

## Precursor step: unify Turbostream datum handling in `_io.py`

This lands **before** the TS4 read-back, as a self-contained change to the
existing TS3 plugin with no TS4 code involved. TS4 then consumes the result
rather than adding a fifth copy of it.

### The problem

Turbostream measures internal energy from `T = 0` (`u = cv*T`), whereas an
ember grid carries a general datum. `_io.py` converts between the two in
**four** places, each hand-rolled, each hardcoding perfect-gas relations, and
they do not all agree:

| site | direction | code |
|---|---|---|
| `read_conserved` (`_io.py:412`) | in | `T = u / block.cv`, `P = rho * block.Rgas * T` |
| `read_ts3` (`_io.py:559-562`) | in | `cv = cp/ga; Rgas = cp-cv; T = u/cv; P = rho*Rgas*T` |
| `TS3Writer.get_blocks` (`_io.py:934`) | out | `roe = rho * (block.cv*block.T + 0.5*block.V**2)` |
| probe/cut read (`_io.py:1421`) | in | `set_fluid(PerfectFluid(..., T_dtm=1.0))`, then plain `set_conserved` |

### Resolve the 0 K versus 1 K disagreement first

Three sites take the datum as `T = 0`; the fourth uses `T_dtm = 1.0` with the
comment "TS3 sets internal energy datum near 1 K". These are not the same
thing, and the difference is not cosmetic: interpreting a Turbostream `roe`
through a `T_dtm = 1.0` fluid recovers `T = u/cv + 1`, one kelvin high,
because the file's `u` was written as `cv*T` from zero. Against `T ~ 300 K`
that is a 0.33% bias in both `T` and `P` at fixed density, and it means the
two read paths disagree by 1 K on the same file.

**First task: write a test that reads one file through both `read_conserved`
and the probe/cut path and compares `T`.** If it shows the 1 K offset, the
probe path is wrong and the datum is `T = 0` everywhere. Fix it as part of
this step rather than preserving it.

The reason the stand-in exists at all is that **ember cannot represent a
`T_dtm = 0` fluid**: the datum sets `u = s = 0` simultaneously
(`fluid.py:34-38`), and entropy is `s = cp ln(T/T_dtm) - R ln(p/p_dtm)`
(`fluid.py:1012`), which is singular at `T_dtm = 0`. Internal energy is fine
there; entropy is not. So `change_datum(P_dtm, 0.0)` is not a legal call, and
any design that materialises a fluid on the Turbostream datum is ruled out --
including the otherwise-attractive "temporarily put the block on the foreign
datum via `set_fluid`" idiom. The conversion has to stay an explicit `u <-> T`
relation.

### The shape of the fix

One conversion, in one place, used by all four sites and by TS4:

- A single module constant naming the convention (`_TS_T_DTM = 0.0`) with a
  comment recording *why* it cannot be expressed as a `Fluid` datum.
- One inbound helper (`u` on the TS datum, plus `rho`, to dimensional `P`/`T`
  for `set_P_T`) and one outbound helper (block state to `roe` on the TS
  datum). Both take the fluid explicitly, since `read_ts3` has only the
  file's gas properties and no block fluid yet.
- An explicit `isinstance(fluid, PerfectFluid)` guard raising a clear "real
  gas not supported by this datum conversion" error, rather than silently
  returning `u/cv` for a fluid where that is meaningless.

### Why this is not made EOS-general

An earlier draft of this plan tried to express the conversion through the
`Fluid` ABC alone (`change_datum` + `get_P`/`get_T`) so that a future real
fluid would work unchanged. That is not achievable, and the reason is worth
recording so it is not attempted again:

- A `u = cv*T` datum measured from absolute zero is a perfect-gas concept. The
  general offset is `u` evaluated at `T = 0`, which a table-backed EOS cannot
  supply -- real-gas tables have a finite temperature range (see turbigen's
  real-gas notes on choosing `smin`/`smax`/`Tmax`).
- A real-gas Turbostream would not use that datum anyway. Its `roe` would sit
  on whatever datum the CoolProp-derived tables define, which *is* a normal
  `(P_dtm, T_dtm > 0)` pair and *would* go through `change_datum` /
  `Block.set_fluid` (`block.py:740-785`, which reads dimensional `P`/`T`/`V`
  through the old fluid and re-expresses through the new, handling reference
  scales and datum levels generically).

So the perfect-gas assumption is real, but it is confined to the
`fluid_model = 0` case that is all this port supports. Making it explicit at
one guarded site leaves the general path open for real gas later, instead of
being implicit at four sites today.

### Two sibling-repo docstring fixes, while here

- `Fluid.change_datum`'s ABC docstring (`fluid.py:209`) claims it returns
  `(fluid_new, rho_nd_new, u_nd_new)`; `PerfectFluid` returns only
  `fluid_new`. Stale, and it will mislead whoever implements the real-gas EOS.
- `set_conserved_cart_unstr` does not say which datum the incoming `roe` is
  assumed to be on. State that it is the grid's own, so the next importer does
  not walk into this. (This is why TS4 uses the primitive route below despite
  `[ro, rovx, rovy, rovz, roe]` matching the conserved signature exactly.)

## Read-back: TS4 consumes the precursor

With the above in place, `_read_flow` needs no thermodynamics of its own:
divide out density, convert `u = roe/ro - 0.5*Vsq` to `P` with the shared
inbound helper, and call
`grid.set_primitive_cart_unstr(xyz, [rho, Vx, Vy, Vz, P])`. Primitives are
datum-free, so once `P` is in hand the datum is out of the picture and the
state lands correctly on each block's own datum.

Turbulent viscosity is transferred too: turbigen assigns
`block.mu_turb = turb0[ind_pts]` (ts4.py:507) and dropping it would leave a
restart or any post-processing with stale/zero `mu_turb`. Use
`Block.set_mu_turb` (`block.py:926`), scattered through the same
`block_indices` correspondence, mirroring what `_io.read_mu_turb` does for
TS3.

## Inlet pitch-angle sign convention

Turbigen flips the inlet pitch angle before writing and restores it after
(`inpatch.Beta *= -1.0`, ts4.py:652 and 875). This must be resolved
explicitly, not silently dropped: if it is a genuine TS3->TS4 convention
difference, omitting it mis-sets the inlet flow angle with no error. **First
implementation task is to determine which it is** (compare what the TS3 writer
emits for `Beta` against what `ts3_reader`/`create_ts_grid` produce for the
TS4 inlet bcond), and record the finding in a comment either way.

If the flip is required, it must not be implemented turbigen's way: mutating
the caller's grid and restoring it later leaks the flipped sign on any
exception raised between the two statements. Apply it inside the write path
(so only the file content is affected, never the caller's grid), or failing
that in a `try/finally`.

## `config.ofp` is grid-coupled

`DEFAULT_CONFIG` and `_write_ofp` are pure dict/text formatting, but the
config file as actually written is not: `run()` injects gas properties from
the inlet patch state (ts4.py:657-661) and `to_ofp()` derives step counts and
ramp flags. The port needs an explicit `_config_from_grid(cfg, grid)` step
covering:

- `gamma`, `cp`, `viscosity`, `prandtl_laminar` from the inlet patch state,
  `prandtl_turbulent = Pr * 1.25`.
- `fluid_model = 0` unconditionally (only `PerfectFluid` is in scope; raise a
  clear "real gas out of scope for this port" error otherwise, rather than
  emitting `fluid_model = 1` with no tables).
- `istep_avg_start = nstep - nstep_avg`, with turbigen's validation that it is
  less than `nstep`.
- The ramp-flag derivations from `to_ofp()` (`cfl_ramp_en = cfl`,
  `pout_fac_ramp`, `precon_fac_ramp`, `rot_fac_ramp`, `mixing_rf_ramp` each
  set from their corresponding `*_nstep` field). The `if_dts` branch is
  dropped with unsteady.

## New files (mirroring the `_io.py`/`_convergence.py`/`_solver.py` split)

All added under `src/ember/plugins/ts/`, exported from `__init__.py` next to
the existing TS3 exports (package docstring updated to mention TS4 too).
Filenames are `_<role>_ts4.py` rather than `_<role>4.py`, so they read as "TS4"
and not "version 4 of `_io`":

- **`_scripts/convert_ts3_to_ts4.py`** -- vendored copy of turbigen's
  `convert_ts3_to_ts4_native.py`, trimmed to the steady case (drop the
  inlet/outlet "group" tagging used only for unsteady forcing, out of scope
  here). Pure glue against `ts.process`/VTK, no ember dependency -- invoked as
  a subprocess exactly like `turbostream4` itself.

- **`_io_ts4.py`** -- TS4 unstructured hdf5 I/O, built against `ember.grid.Grid`/
  `ember.block.Block` instead of turbigen's types:
  - `_write_wall_distance(grid, ts4_input_path)` -- no KDTree of its own;
    reads TS4's `node_array_x/y/z`, enforces the node-count invariant above,
    calls the new `grid.align_cart_unstr()` wrapper to get the
    struct/unstruct index correspondence, and scatters the already-computed
    `block.wdist` into `node_array_wdist` with the NaN guard.
  - `_write_throttle(ts4_conf, grid, ts4_input_path)` -- reads
    `grid.patches.outlet`, `patch.K_pid`, `patch.mdot_target`,
    `patch.block.Nb` (all present in core `ember`, confirmed). Two deviations
    from turbigen's version, both deliberate:
    - Turbigen builds `throttle_tag_list` from the bcell-name match but the
      target/kind/k0 lists per outlet patch (ts4.py:514-552), with nothing
      guaranteeing equal length or matching order. Build one entry per bcell
      tag from an explicit patch-to-bcell correspondence, and assert the list
      lengths agree before writing.
    - Turbigen's `if patch.Kpid is not None` does not translate: ember's
      `OutletPatch._K_pid` defaults to `(0.0, 0.0, 0.0)` and is never `None`
      (`outlet.py:224`), so that branch would always take the mdot path. The
      correct predicate against ember is `patch.mdot_target > 0`.
    - Keep turbigen's error listing the available `bcell_names` when
      `outlet_tag` does not match: TS3 hdf5 stores patch *kind* codes, not
      names (`_io.py:42-60`), so the `"Outlet"` string is invented by the
      vendor's `ts3_reader` and is a real coupling to the vendor stack.
  - `_read_flow(grid, output_path, output_avg_path)` -- replaces turbigen's
    whole hand-rolled KDTree/rotation block with the datum conversion above
    plus one call to `grid.set_primitive_cart_unstr(xyz, primitive_cart)` and
    the `mu_turb` transfer. Coordinates come from the instantaneous output,
    state from the averaged output; check the averaged file exists first and
    raise the same way `TS3Solver._read_hdf5` does (it may be absent with
    `nstep_avg = 0` or an early stop). Wrap the read in the same
    `ValueError -> DivergenceError` translation `_read_hdf5` uses.
  - `_write_ofp`, `_write_probes_ofp`, `_check_probes`, `_write_point_probe`,
    `_write_logical_probe`, `DEFAULT_CONFIG` -- ported near-verbatim (pure
    text/dict formatting, no grid coupling). **`_read_ofp` is not vendored**:
    it opens the file with mode `"w"`, truncating what it claims to read
    (ts4.py:366), and nothing calls it.

- **`_convergence_ts4.py`** -- `parse_ts4_log`, following the shape of the
  existing `parse_ts3_log` (`_convergence.py`): reconstructs a
  `ConvergenceHistory` via `ConvergenceHistory.from_grid` +
  `record_convergence`, but parses TS4's log format (`ROVX DELTA:`,
  `INLET:`/`OUTLET:` lines -- turbigen's `parse_log`/`re_nan`/`re_current_step`)
  instead of TS3's. Two robustness fixes over turbigen's `parse_log`:
  - Bound-check the step index. Turbigen preallocates `nstep`-length arrays
    and increments `istep` per `OUTLET:` line unchecked, so a log with more
    steps than `nstep` raises `IndexError` and one with fewer leaves a silent
    NaN tail. Raise a clear error on overrun and truncate on undershoot.
  - Do not swallow parse failures. Turbigen wraps the whole call in a bare
    `except: conv = None` (ts4.py:891). Since `INLET:`/`OUTLET:` values are
    per-passage and get scaled by `Nb`, a grid/log mismatch corrupts mass flow
    silently; follow `parse_ts3_log` and let failures raise.

- **`_solver_ts4.py`** -- `TS4Solver(BaseSolver)` dataclass, same shape as
  `TS3Solver` in `_solver.py`:
  - Typed config fields mirroring turbigen's `ts4` dataclass (`cfl`,
    `nstep`/`nstep_avg`, `implicit_scheme`, `viscous_model`, mixing-plane and
    ramp settings, `environment_script`), **minus** `nstep_ts3`,
    `body_force_template`/`body_force_params`, `tables_path`, `custom_pipeline`
    (all descoped).
  - `run(self, grid)`: write TS3 mesh (via `TS3Writer`, no execution) + write
    `config.ofp` -> convert (subprocess) -> write throttle/wdist/probes ->
    execute TS4 (subprocess) -> `_read_flow` back into `grid` ->
    `parse_ts4_log`.
  - Both subprocess calls use `check=True`. Turbigen reuses the `check` flag
    from the conversion branch for the TS4 run itself (ts4.py:856), so with a
    custom pipeline a TS4 crash silently goes undetected; with
    `custom_pipeline` descoped there is no reason to carry that wart.

### No live log polling for TS4

`TS3Solver._execute` polls `log.txt` while the solver runs and kills the
process group on NAN (`_solver.py:216-252`), because TS3 will happily march on
forever once the solution has gone bad. **TS4 has its own in-process NAN check
and aborts itself**, so that machinery is not needed here and must not be
copied over: `_execute` for TS4 is a plain launch-and-wait, and the log is
scanned once after the process exits to translate TS4's abort into a
`DivergenceError` (turbigen's `_check_nan`, ts4.py:313, with
`re_nan = ^detected NAN$` and `re_current_step = ^RK LOOP NO:\s*(\d*)$` to
recover the step number).

That removes most of the surface the two `_execute` functions would otherwise
share, so the poll loop, the `_LOG_TIMEOUT_S` wait-for-log, and the
process-group kill all stay TS3-only in `_solver.py`. What is still worth
sharing is the small, genuinely common piece:

- **`_log.py`** -- `_archive_log(workdir)` and the `_re_log_index` numbering
  (`_solver.py:163`), moved out of `_solver.py` unchanged and imported by both
  solvers, so successive TS4 runs in one workdir keep `log_000.txt`,
  `log_001.txt`, ... the same way TS3 runs do. Note TS4's live log is
  `log_ts4.txt`, not `log.txt`, so `_archive_log` gains a log-name argument
  (defaulted to `log.txt` to leave the TS3 call sites untouched).

The `KeyboardInterrupt` handling and the non-zero-return-code error formatting
are a dozen lines each and read differently once the poll loop is gone;
duplicate them in `_solver_ts4.py` rather than forcing a shared abstraction
over two now-dissimilar control flows.

## Testing

Follow `tests/test_ts3_solver.py`'s pattern exactly:

- **Unit**: `_write_throttle`/`_write_wall_distance`/`_read_flow`/probe
  writers against small `Grid` fixtures (same `_make_solved_grid`/
  `_make_grid_with_inlet` helpers, extended as needed), no subprocess
  involved. Include a **two-block** fixture, not just single-block: the node
  correspondence and the throttle patch-to-bcell mapping are both trivially
  satisfied on one block.
- **Unit**: `_write_wall_distance` against a deliberately wrong-sized TS4 node
  array, asserting the vendor-bug error is raised with both counts in the
  message. This is the guard the whole design leans on, so it gets a direct
  test rather than only being exercised implicitly.
Precursor step (lands first, TS3 only, no TS4 code involved):

- **Unit**: the 0 K versus 1 K tripwire -- read one TS3 file through both
  `read_conserved` and the probe/cut path and assert the recovered `T` agree.
  This is the test that decides whether `_io.py:1421`'s `T_dtm = 1.0` is a bug;
  write it before changing anything.
- **Unit**: datum round-trip through the unified helpers -- write a grid out
  and read it back, asserting `P`/`T` match the original to tolerance, on a
  grid whose fluid is *not* on the Turbostream datum (otherwise the test
  passes with the datum handling deleted entirely). Covers both directions,
  since the writer is one of the four sites being unified.
- **Unit**: the real-gas guard raises rather than silently applying `u/cv`.

TS4 port proper:

- **Unit**: `_read_flow` datum round-trip, same shape as the precursor's, to
  confirm TS4 picks up the shared helper rather than growing a fifth copy.
- **Unit**: `_convergence_ts4.parse_ts4_log` against a small literal TS4 log
  fixture string, plus the overrun and undershoot cases.
- **End-to-end**: `_write_mock_env`-style fake `environment_script` that puts
  fake `convert_ts3_to_ts4`, `mpirun`, and `turbostream4` scripts on `PATH`.
  Since the real converter's internals are opaque vendor code, the mock
  converter can't exercise real conversion logic -- it fabricates a plausible
  TS4 unstructured hdf5 directly from the TS3 mesh input's node coordinates
  (identity, or a small perturbation to prove the remap path actually ran,
  same trick the TS3 "happy" mock uses). Add "nan"/"diverged" modes to
  exercise `DivergenceError` -- these differ from the TS3 mocks in that the
  fake `turbostream4` writes `detected NAN` to the log and *exits*, mirroring
  TS4's self-abort, rather than running on until an external poll kills it --
  and a
  "merged-nodes" mode that drops the duplicate interface nodes on the
  two-block fixture, asserting the vendor-bug error surfaces through the full
  `run()` rather than as a bare assertion from inside core ember.

## Verification

- `uv run pytest tests/test_ts4_solver.py -v` (new file, mirroring
  `test_ts3_solver.py`'s structure) exercises config defaults, input-writing
  (config.ofp/throttle/wdist/probes), and the mocked end-to-end
  write->convert->execute->read path including divergence handling and the
  node-count invariant.
- `uv run pytest` (full suite) after the **precursor step alone**, before any
  TS4 work starts. That step rewrites all four datum sites in `_io.py`, so
  `test_ts3_io.py`, `test_ts3_readback.py`, `test_ts3_probe.py` and
  `test_read_probe_dat.py` are its regression guard and must pass unchanged --
  except where the 0 K/1 K resolution deliberately changes a value, which must
  be an explicit, reviewed diff rather than a silent adjustment.
- `uv run pytest` (full suite) again after the TS4 work. Note this is not
  purely additive either: `_archive_log` moves out of `_solver.py`, so
  `test_ts3_solver.py` guards that extraction. The TS3 poll loop itself is
  untouched.
- Core `ember` sibling repo: only one functional prerequisite,
  `Grid.align_cart_unstr`, which needs its own test there before this plugin
  work lands. The datum handling needs no core change -- only two docstring
  corrections (`Fluid.change_datum`'s stale return signature, and the
  grid-datum assumption in `set_conserved_cart_unstr`).
- No real Turbostream/TS4 install is available here, so actual execution
  against the vendor binaries is out of scope for verification in this
  environment -- the mock-env end-to-end test is the ceiling for this port.
  In particular, the node-count invariant and the inlet `Beta` sign convention
  are both stated against the *real* converter and can only be confirmed on a
  machine that has it; both are written to fail loudly rather than degrade.
