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
2. **The TS3→TS4 mesh conversion is not portable.** `convert_ts3_to_ts4_native.py`
   is glue code around Turbostream Ltd's own proprietary `ts.process`/VTK
   stack (`ts.process.ts3_reader`, `vtk_adaptor.create_ts_grid`, ...) — it has
   no ember/turbigen API dependency at all, so it can be vendored into this
   plugin almost verbatim, but it must remain an external subprocess call
   (sourced via `environment_script`, same as running `turbostream4` itself).
   It cannot be reimplemented natively.

Per direction from the user, this port also **drops `nstep_ts3`**: turbigen's
TS4 solver could optionally run a short TS3 solve first as an initial guess.
`TS4Solver` won't do that — like `TS3Solver`, it takes a `grid` that the
caller has already brought to a valid initial state (flow field, wdist, etc.)
and just writes it out as the TS3-format mesh/BC/flow file that the converter
consumes. No TS3 *execution* happens inside `TS4Solver`.

Real-gas tables, the `custom_pipeline` (paraview) conversion route, and
unsteady forcing are out of scope for this first port, matching the level
`TS3Solver` itself is currently scoped at (`PerfectFluid` only, no unsteady).
GPU/SLURM launch mechanics are kept, but simplified to the same "1 GPU,
serial" shape `TS3Solver._execute` uses today — multi-GPU/SLURM can follow
later.

## Pipeline

```
Grid (caller-supplied, already has a valid flow field + wdist)
   │  TS3Writer.get(grid, strict=True) / .write()        [reuse _io.py, no execution]
   ▼
TS3-format mesh/flow hdf5
   │  vendored convert_ts3_to_ts4.py (subprocess, sources environment_script)
   ▼
TS4 input hdf5 (unstructured node arrays)
   │  _write_wall_distance  (new, bespoke KDTree — see "Core ember prerequisite" below)
   │  _write_throttle       (grid.patches.outlet → OutletPatch.K_pid/mdot_target)
   │  _write_probes_ofp / point+logical probes
   ▼
run turbostream4 (subprocess, mpirun)
   │
   ▼
TS4 output hdf5 (unstructured node arrays: ro, rovx/y/z, roe, turb0)
   │  grid.set_primitive_cart_unstr(xyz, [rho, Vx, Vy, Vz, P])   [reuse core ember]
   ▼
Grid, updated in place
```

## Core `ember` prerequisite: wall-distance transfer

Turbigen's `_write_wall_distance` recomputes wall distance from scratch
against the TS4 unstructured cloud (a fresh KDTree of wall nodes, queried at
the TS4 node coordinates). That's not necessary here: the TS3→TS4 conversion
is a pure format/connectivity translation, not a remesh, so a freshly
converted TS4 *input* file's `node_array_x/y/z` sit at exactly the same
physical points as `block.x/y/z` — the same exact-coincidence property that
already makes `set_primitive_cart_unstr` work on the *read* side (via
`Grid._align_cartesian`, `grid.py:330`, which builds the KDTree, permutation
search, and `distance_upper_bound=atol` exact match).

So instead of computing wall distance a second, independent way, `wdist` can
be **transplanted**: `block.wdist` is already correct (periodicity/pitch-tiling
included, via `Grid.calculate_wdist`) on the structured grid before TS4 ever
sees it. Run the same alignment used for read-back once, immediately after
conversion, and scatter `block.wdist` into `node_array_wdist` using the
resulting index correspondence:

```python
perm, signs, block_indices = grid.align_cart_unstr(xyz_ts4_input)  # new public wrapper
wdist_unstr = np.empty(xyz_ts4_input.shape[0])
for ib, block in enumerate(grid):
    wdist_unstr[block_indices[ib].flatten()] = block.wdist.flatten()
f["node_array_wdist"][:] = wdist_unstr
```

This needs one small addition to core `ember` (sibling repo
`/home/jb753/python/ember`): a public wrapper around `Grid._align_cartesian`
(e.g. `Grid.align_cart_unstr(xyz)`, returning `(perm, signs, block_indices)`)
— it's already computed internally by `set_primitive_cart_unstr`/
`set_conserved_cart_unstr`, just not exposed. This is a smaller, more general
addition than a bespoke wall-node accessor: one alignment mechanism now
serves both directions (flow fields in on read, `wdist` out on write) instead
of two independently-implemented geometric algorithms that would each have to
agree with `calculate_wdist`.

Caveat: this relies on exact node coincidence, which only holds for the
native conversion route (not `custom_pipeline`/paraview, which may resample
the mesh) — fine here since `custom_pipeline` is already out of scope for
this port.

## New files (mirroring the `_io.py`/`_convergence.py`/`_solver.py` split)

All added under `src/ember/plugins/ts/`, exported from `__init__.py` next to
the existing TS3 exports (package docstring updated to mention TS4 too):

- **`_scripts/convert_ts3_to_ts4.py`** — vendored copy of turbigen's
  `convert_ts3_to_ts4_native.py`, trimmed to the steady case (drop the
  inlet/outlet "group" tagging used only for unsteady forcing, out of scope
  here). Pure glue against `ts.process`/VTK, no ember dependency — invoked as
  a subprocess exactly like `turbostream4` itself.

- **`_io4.py`** — TS4 unstructured hdf5 I/O, built against `ember.grid.Grid`/
  `ember.block.Block` instead of turbigen's types:
  - `_write_wall_distance(grid, ts4_input_path)` — no KDTree of its own;
    reads TS4's `node_array_x/y/z`, calls the new `grid.align_cart_unstr()`
    wrapper (above) to get the struct↔unstruct index correspondence, and
    scatters the already-computed `block.wdist` into `node_array_wdist`.
  - `_write_throttle(ts4_conf, grid, ts4_input_path)` — mirrors turbigen's
    version but reads `grid.patches.outlet`, `patch.K_pid`, `patch.mdot_target`,
    `patch.block.Nb` (all present in core `ember`, confirmed).
  - `_read_flow(grid, output_path, output_avg_path)` — replaces turbigen's
    whole hand-rolled KDTree/rotation block with one call to
    `grid.set_primitive_cart_unstr(xyz, primitive_cart)` after computing
    `rho`, `Vx/Vy/Vz`, `P` from the TS4 conserved arrays.
  - `_write_ofp` / `_read_ofp`, `_write_probes_ofp`, `_write_point_probe`,
    `_write_logical_probe`, `DEFAULT_CONFIG` — ported near-verbatim (pure
    text/dict formatting, no grid coupling).

- **`_convergence4.py`** — `parse_ts4_log`, following the shape of the
  existing `parse_ts3_log` (`_convergence.py`): reconstructs a
  `ConvergenceHistory` via `ConvergenceHistory.from_grid` +
  `record_convergence`, but parses TS4's log format (`ROVX DELTA:`,
  `INLET:`/`OUTLET:` lines — turbigen's `parse_log`/`re_nan`/`re_current_step`)
  instead of TS3's.

- **`_solver4.py`** — `TS4Solver(BaseSolver)` dataclass, same shape as
  `TS3Solver` in `_solver.py`:
  - Typed config fields mirroring turbigen's `ts4` dataclass (`cfl`,
    `nstep`/`nstep_avg`, `implicit_scheme`, `viscous_model`, mixing-plane and
    ramp settings, `environment_script`), **minus** `nstep_ts3`,
    `body_force_template`/`body_force_params`, `tables_path`, `custom_pipeline`
    (all descoped).
  - `run(self, grid)`: write TS3 mesh (via `TS3Writer`, no execution) →
    convert (subprocess) → write throttle/wdist/probes → execute TS4
    (subprocess) → `_read_flow` back into `grid` → `parse_ts4_log`. Reuses
    `TS3Solver`'s `_archive_log`/`_check_nan`/`DivergenceError` pattern rather
    than re-deriving it — factor the small amount of genuinely shared
    subprocess-polling logic (log-timeout wait loop, NaN poll, cleanup) out of
    `_solver.py` into a small shared helper if it turns out to be
    copy-pastable verbatim; otherwise duplicate it the same way turbigen did
    (ts3.py/ts4.py were already two independent copies there) — decide once
    the two `_execute` functions are written side by side.

## Testing

Follow `tests/test_ts3_solver.py`'s pattern exactly:

- **Unit**: `_write_throttle`/`_write_wall_distance`/`_read_flow`/probe
  writers against small single-block `Grid` fixtures (same
  `_make_solved_grid`/`_make_grid_with_inlet` helpers, extended as needed),
  no subprocess involved.
- **Unit**: `_convergence4.parse_ts4_log` against a small literal TS4 log
  fixture string.
- **End-to-end**: `_write_mock_env`-style fake `environment_script` that puts
  fake `convert_ts3_to_ts4`, `mpirun`, and `turbostream4` scripts on `PATH`.
  Since the real converter's internals are opaque vendor code, the mock
  converter can't exercise real conversion logic — it fabricates a plausible
  TS4 unstructured hdf5 directly from the TS3 mesh input's node coordinates
  (identity, or a small perturbation to prove the remap path actually ran,
  same trick the TS3 "happy" mock uses). Add "nan"/"diverged" modes
  mirroring the existing TS3 mock modes to exercise `DivergenceError`.

## Verification

- `uv run pytest tests/test_ts4_solver.py -v` (new file, mirroring
  `test_ts3_solver.py`'s structure) exercises config defaults, input-writing
  (throttle/wdist/probes), and the mocked end-to-end write→convert→execute→
  read path including divergence handling.
- `uv run pytest` (full suite) to confirm no regression in the existing TS3
  tests, since `_solver4.py`/`_io4.py` are additive and `TS3Writer` is reused
  read-only.
- No real Turbostream/TS4 install is available here, so actual execution
  against the vendor binaries is out of scope for verification in this
  environment — the mock-env end-to-end test is the ceiling for this port.
