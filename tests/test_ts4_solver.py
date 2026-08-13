"""Tests for the ``ember.plugins.ts.TS4Solver`` run driver and TS4 I/O.

These cover what is testable without a Turbostream install: the config
dataclass, the TS3-format input written for the converter, the fields added to
a converted TS4 file (wall distance, throttle, probes), the read-back onto the
grid, log parsing, and the whole write -> convert -> execute -> read path
against mock programs on ``PATH``.

The real converter is opaque vendor code, so the mock stands in for it by
fabricating a TS4 file from the TS3 input's own node coordinates. That makes
the exact node correspondence trivially true, which is why the invariant it
relies on is *also* tested directly, with a deliberately mismatched file.
"""

import os
import stat
import sys
import textwrap
from pathlib import Path

import h5py
import numpy as np
import pytest

import ember.util
from ember.block import Block
from ember.fluid import PerfectFluid
from ember.grid import Grid, DivergenceError
from ember.patch import InletPatch, OutletPatch

import ember.plugins.ts._io_ts4 as io4
import ember.plugins.ts._solver_ts4 as ts4_mod
from ember.plugins.ts import (
    TS4Solver,
    TS4ConversionError,
    parse_ts4_log,
    read_ts3,
)

CP, GAMMA, MU, PR = 1005.0, 1.4, 1.8e-5, 0.7
CV = CP / GAMMA
RGAS = CP - CV

# Blocks must span exactly one whole pitch: the inlet is non-reflecting, and its
# pitchwise node weights are checked to sum to 1.
NBLADE = 36
PITCH = 2.0 * np.pi / NBLADE


def _make_block(shape, xs, rs, ts_, T_dtm, P, T):
    block = Block(shape=shape)
    xrt = ember.util.linmesh3(xs, rs, ts_, shape)
    block.set_x(xrt[..., 0])
    block.set_r(xrt[..., 1])
    block.set_t(xrt[..., 2])
    block.set_fluid(
        PerfectFluid(cp=CP, gamma=GAMMA, mu=MU, Pr=PR, T_dtm=T_dtm)
    )
    block.set_rpm(0.0)
    block.set_Nb(NBLADE)
    block.set_P_T(np.full(shape, P), np.full(shape, T))
    block.set_Vxrt(
        np.stack(
            [np.full(shape, 120.0), np.full(shape, 5.0), np.full(shape, -20.0)],
            axis=-1,
        )
    )
    block.set_mu_turb(np.full(shape, 1e-3))
    # Wall distance varies with position, so a mistransferred value cannot pass
    # by happening to be the right constant everywhere. It is a function of the
    # *absolute* coordinate, not of the index, so two blocks meeting on a
    # shared face agree there -- as real Grid.calculate_wdist output does, and
    # as any position-indexed transfer requires.
    block.set_wdist((0.01 + 0.01 * block.r).astype(np.float32))
    return block


def _make_grid(T_dtm=300.0, n_block=2):
    """Grid with an inlet and an outlet patch, on a non-Turbostream datum.

    Two blocks by default: a single block would satisfy the node
    correspondence and the outlet-to-bcell mapping trivially.
    """
    blocks = [
        _make_block((3, 4, 5), [0.0, 1.0], [0.5, 1.5], [0.0, PITCH], T_dtm, 1.2e5, 320.0)
    ]
    if n_block > 1:
        # Geometrically disjoint from block 0, so no node is shared. A shared
        # face would mean coincident nodes holding this fixture's deliberately
        # *different* per-block flow values, which no position-indexed transfer
        # can reproduce and which a real matching interface never has. The
        # coincident-node case is covered on its own, where the transferred
        # quantity is consistent across the join.
        blocks.append(
            _make_block(
                (4, 3, 6), [1.5, 2.5], [0.5, 1.5], [0.0, PITCH], T_dtm, 1.1e5, 310.0
            )
        )

    inlet = InletPatch(i=0, j=(0, -1), k=(0, -1), label="inlet")
    blocks[0].patches.append(inlet)
    inlet.set_Po_To(Po=1.3e5, To=330.0)
    inlet.set_Alpha(10.0)
    # A non-zero pitch, so the TS4 sign flip is observable.
    inlet.set_Beta(7.5)
    outlet = OutletPatch(i=-1, j=(0, -1), k=(0, -1), label="outlet")
    blocks[-1].patches.append(outlet)
    outlet.set_P(1.0e5)
    return Grid(blocks)


def _grid_xyz(grid):
    """Cartesian node coordinates in block order, as the converter emits them."""
    return np.concatenate(
        [
            np.stack([b.x.flatten(), b.y.flatten(), b.z.flatten()], axis=-1)
            for b in grid
        ]
    )


def _read_ofp(path):
    """Parse an ``.ofp`` file back into a dict, for numeric comparison."""
    conf = {}
    for line in Path(path).read_text().splitlines():
        key, _, value = line.partition(" = ")
        if value.startswith("["):
            conf[key] = [float(v) for v in value.strip("[]").split(",") if v.strip()]
        else:
            conf[key] = value
    return conf


def _write_ts4_file(path, grid, *, n_node=None, avg=False, bcell_names=("Outlet",)):
    """Fabricate a TS4 hdf5 file from a grid, as the converter would produce.

    ``n_node`` overrides the node count to simulate a converter that failed to
    preserve the mesh.
    """
    xyz = _grid_xyz(grid)
    if n_node is not None:
        xyz = xyz[:n_node]
    n = xyz.shape[0]

    with h5py.File(path, "w") as f:
        f.create_dataset("node_array_x", data=xyz[:, 0])
        f.create_dataset("node_array_y", data=xyz[:, 1])
        f.create_dataset("node_array_z", data=xyz[:, 2])
        f.create_dataset("node_array_wdist", data=np.zeros(n))
        names = f.create_group("bcell_names")
        names.attrs["names"] = list(bcell_names)

        if avg:
            rho = np.concatenate([b.rho.flatten() for b in grid])
            Vx = np.concatenate([b.Vx.flatten() for b in grid])
            Vy = np.concatenate([b.Vy.flatten() for b in grid])
            Vz = np.concatenate([b.Vz.flatten() for b in grid])
            T = np.concatenate([b.T.flatten() for b in grid])
            mu_turb = np.concatenate([b.mu_turb.flatten() for b in grid])
            # Energy on Turbostream's datum, which is what TS4 writes.
            roe = rho * (CV * T + 0.5 * (Vx**2 + Vy**2 + Vz**2))
            f.create_dataset("node_array_ro", data=rho)
            f.create_dataset("node_array_rovx", data=rho * Vx)
            f.create_dataset("node_array_rovy", data=rho * Vy)
            f.create_dataset("node_array_rovz", data=rho * Vz)
            f.create_dataset("node_array_roe", data=roe)
            f.create_dataset("node_array_turb0", data=mu_turb)


# --- config -----------------------------------------------------------------


def test_module_imports():
    """TS4Solver is exported and is a BaseSolver implementing run(grid)."""
    from ember.solver import BaseSolver

    assert issubclass(TS4Solver, BaseSolver)
    assert hasattr(ts4_mod, "TS4Solver")


def test_config_instantiation_defaults():
    """The TS4Solver config dataclass instantiates with its documented defaults."""
    cfg = TS4Solver()
    assert cfg.cfl == 25.0
    assert cfg.nstep == 5000
    assert cfg.nstep_avg == 1000
    assert cfg.viscous_model == 2
    assert cfg.outlet_tag == "Outlet"


def test_config_from_grid_gas_properties():
    """Gas properties come from the grid's fluid, not from defaults."""
    cfg = TS4Solver(nstep=4000, nstep_avg=1000)
    conf = io4.config_from_grid(cfg, _make_grid())

    assert conf["cp"] == pytest.approx(CP)
    assert conf["gamma"] == pytest.approx(GAMMA)
    assert conf["viscosity"] == pytest.approx(MU)
    assert conf["prandtl_laminar"] == pytest.approx(PR)
    assert conf["prandtl_turbulent"] == pytest.approx(PR * 1.25)
    assert conf["fluid_model"] == 0

    # Averaging window counted back from the end of the run.
    assert conf["istep_avg_start"] == 3000
    # Typed fields override the baseline.
    assert conf["nstep"] == 4000


def test_config_from_grid_rejects_over_long_average():
    """Averaging for longer than the run is a configuration error."""
    cfg = TS4Solver(nstep=100, nstep_avg=500)
    with pytest.raises(ValueError, match="exceeds nstep"):
        io4.config_from_grid(cfg, _make_grid())


def test_config_ramp_flags_follow_step_counts():
    """Each ramp is enabled by its own step count, so the two cannot disagree."""
    grid = _make_grid()

    off = io4.config_from_grid(TS4Solver(), grid)
    assert off["rot_fac_ramp"] == 0
    assert off["pout_fac_ramp"] == 0

    on = io4.config_from_grid(
        TS4Solver(rot_fac_ramp_nstep=250, pout_fac_ramp_nstep=100), grid
    )
    assert on["rot_fac_ramp"] == 1
    assert on["pout_fac_ramp"] == 1
    # A CFL ramp always finishes at the requested CFL number.
    assert on["cfl_ramp_en"] == pytest.approx(on["cfl"])


def test_write_ofp_formats_lists(tmp_path):
    """Lists are written in the bracketed form TS4 expects."""
    path = tmp_path / "x.ofp"
    io4.write_ofp(path, {"a": 1, "b": 2.5, "c": [1, 2, 3]})
    assert path.read_text() == "a = 1\nb = 2.5\nc = [1, 2, 3]\n"


# --- TS3 input for the converter --------------------------------------------


def test_write_ts3_input_inverts_inlet_pitch(tmp_path):
    """Inlet pitch is written with TS4's opposite sign convention."""
    grid = _make_grid()
    fname = str(tmp_path / "input.hdf5")
    io4.write_ts3_input(grid, fname)

    read_back = read_ts3(fname)
    # The prescription survives as sinBeta, so invert it back to degrees.
    written = np.degrees(np.arcsin(read_back[0].patches.inlet[0].sinBeta))
    np.testing.assert_allclose(written, -7.5, rtol=1e-5)


def test_write_ts3_input_does_not_mutate_grid(tmp_path):
    """The sign flip touches the file only, never the caller's grid."""
    grid = _make_grid()
    before = grid[0].patches.inlet[0].sinBeta.copy()

    io4.write_ts3_input(grid, str(tmp_path / "input.hdf5"))

    np.testing.assert_allclose(grid[0].patches.inlet[0].sinBeta, before)


# --- wall distance ----------------------------------------------------------


def test_write_wall_distance_transplants_grid_values(tmp_path):
    """The grid's own wdist is scattered onto the matching TS4 nodes."""
    grid = _make_grid()
    fname = tmp_path / "input_ts4.hdf5"
    _write_ts4_file(fname, grid)

    io4.write_wall_distance(grid, str(fname))

    with h5py.File(fname, "r") as f:
        wdist = f["node_array_wdist"][:]

    expected = np.concatenate([b.wdist.flatten() for b in grid])
    np.testing.assert_allclose(wdist, expected, rtol=1e-6)


def test_write_wall_distance_across_matching_interface(tmp_path):
    """Duplicated nodes at a matching block interface are all filled.

    Two blocks meeting on a shared face hold exactly coincident nodes there,
    and so does the TS4 file. A nearest-neighbour query cannot tell those
    apart, so scattering each grid node's value into the TS4 array sends both
    blocks' interface nodes to the same index and leaves the other unwritten.
    This is the case that forces the transfer to be a gather.
    """
    shape = (3, 4, 5)
    blocks = [
        _make_block(shape, [0.0, 1.0], [0.5, 1.5], [0.0, PITCH], 300.0, 1.2e5, 320.0),
        # Identical j/k distribution on the shared x = 1 plane.
        _make_block(shape, [1.0, 2.0], [0.5, 1.5], [0.0, PITCH], 300.0, 1.1e5, 310.0),
    ]
    grid = Grid(blocks)
    xyz = _grid_xyz(grid)
    n_coincident = xyz.shape[0] - np.unique(np.round(xyz, 12), axis=0).shape[0]
    assert n_coincident > 0, "fixture must actually have duplicated nodes"

    fname = tmp_path / "input_ts4.hdf5"
    _write_ts4_file(fname, grid)

    io4.write_wall_distance(grid, str(fname))

    with h5py.File(fname, "r") as f:
        wdist = f["node_array_wdist"][:]

    assert not np.isnan(wdist).any()
    np.testing.assert_allclose(
        wdist, np.concatenate([b.wdist.flatten() for b in grid]), rtol=1e-6
    )


def test_write_wall_distance_rejects_node_count_mismatch(tmp_path):
    """A converter that dropped nodes is a hard error naming both counts."""
    grid = _make_grid()
    fname = tmp_path / "input_ts4.hdf5"
    _write_ts4_file(fname, grid, n_node=grid.size - 3)

    with pytest.raises(TS4ConversionError) as excinfo:
        io4.write_wall_distance(grid, str(fname))

    msg = str(excinfo.value)
    assert str(grid.size - 3) in msg and str(grid.size) in msg
    assert "vendor converter" in msg


def test_write_wall_distance_rejects_unmatched_nodes(tmp_path):
    """Equal counts do not by themselves prove a one-to-one correspondence."""
    grid = _make_grid()
    fname = tmp_path / "input_ts4.hdf5"
    _write_ts4_file(fname, grid)
    # Duplicate one node onto another: the counts still agree, but one grid
    # node now has no partner.
    with h5py.File(fname, "r+") as f:
        for axis in ("x", "y", "z"):
            arr = f[f"node_array_{axis}"][:]
            arr[1] = arr[0]
            f[f"node_array_{axis}"][:] = arr

    with pytest.raises((TS4ConversionError, ValueError)):
        io4.write_wall_distance(grid, str(fname))


# --- throttle ---------------------------------------------------------------


def test_write_throttle_pressure_outlet(tmp_path):
    """An outlet with no mass-flow target is written as a pressure boundary."""
    grid = _make_grid()
    fname = tmp_path / "input_ts4.hdf5"
    _write_ts4_file(fname, grid)

    mdot_flag = io4.write_throttle(TS4Solver(), grid, str(fname), tmp_path)

    assert mdot_flag is False
    text = (tmp_path / "throttle_config.ofp").read_text()
    assert "throttle_kind_list = [1]" in text
    assert "throttle_target_list = [0.0]" in text


def test_write_throttle_mass_flow_outlet(tmp_path):
    """A throttled outlet carries a per-passage target and the integral gain."""
    grid = _make_grid()
    grid.patches.outlet[0].set_throttle(mdot_target=7.2, Kp=1.0, Ki=0.25)
    fname = tmp_path / "input_ts4.hdf5"
    _write_ts4_file(fname, grid)

    mdot_flag = io4.write_throttle(TS4Solver(), grid, str(fname), tmp_path)

    assert mdot_flag is True
    conf = _read_ofp(tmp_path / "throttle_config.ofp")
    assert conf["throttle_kind_list"] == [0.0]
    # Annulus target divided by blade count, and only the I gain carries over,
    # halved for the implicit scheme. Compared numerically: mdot_target is
    # stored as float32, so the written decimal is not the float64 quotient.
    assert conf["throttle_target_list"][0] == pytest.approx(7.2 / 36.0, rel=1e-6)
    assert conf["throttle_k0_list"][0] == pytest.approx(0.25 * 0.5, rel=1e-6)


def test_write_throttle_unknown_tag(tmp_path):
    """A missing outlet tag names the tags the file actually has."""
    grid = _make_grid()
    fname = tmp_path / "input_ts4.hdf5"
    _write_ts4_file(fname, grid, bcell_names=("Inlet", "Wall"))

    with pytest.raises(TS4ConversionError, match="Inlet"):
        io4.write_throttle(TS4Solver(), grid, str(fname), tmp_path)


def test_write_throttle_rejects_tag_patch_mismatch(tmp_path):
    """More tagged outlets than outlet patches is ambiguous, not guesswork."""
    grid = _make_grid()
    fname = tmp_path / "input_ts4.hdf5"
    _write_ts4_file(fname, grid, bcell_names=("Outlet", "Outlet"))

    with pytest.raises(TS4ConversionError, match="ambiguous"):
        io4.write_throttle(TS4Solver(), grid, str(fname), tmp_path)


# --- probes -----------------------------------------------------------------


def test_probe_file_is_valid_python(tmp_path):
    """Generated probe definitions must compile: TS4 executes this file."""
    cfg = TS4Solver()
    io4.write_probes_ofp(tmp_path)
    io4.write_point_probe(
        cfg, tmp_path, np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]), 0, "pp"
    )
    io4.write_logical_probe(cfg, tmp_path, tag="Outlet", label="lp")

    io4.check_probes(tmp_path)

    text = (tmp_path / "probes.ofp").read_text()
    assert text.count("probe_list.append(p)") == 2


def test_check_probes_rejects_broken_file(tmp_path):
    """A malformed probe file is caught here, not after the job starts."""
    (tmp_path / "probes.ofp").write_text("p = (\n")
    with pytest.raises(ValueError, match="not valid Python"):
        io4.check_probes(tmp_path)


# --- read back --------------------------------------------------------------


def test_read_flow_datum_round_trip(tmp_path):
    """P, T and velocities survive the read back onto a non-Turbostream datum.

    The grid is on T_dtm=300, so a reader that treated the file's energy as
    being on the grid's own datum would recover a temperature 300 K adrift.
    """
    grid = _make_grid(T_dtm=300.0)
    expected = [(b.P.copy(), b.T.copy(), b.Vx.copy(), b.Vr.copy()) for b in grid]

    out = tmp_path / "output_ts4.hdf5"
    out_avg = tmp_path / "output_ts4_avg.hdf5"
    _write_ts4_file(out, grid)
    _write_ts4_file(out_avg, grid, avg=True)

    # Blank the flow field so the read has to restore it.
    target = _make_grid(T_dtm=300.0)
    for b in target:
        b.set_P_T(np.full(b.shape, 1e5), np.full(b.shape, 300.0))
        b.set_Vxrt(np.zeros((*b.shape, 3)))
        b.set_mu_turb(np.zeros(b.shape))

    io4.read_flow(target, str(out), str(out_avg))

    for i, (b, (P, T, Vx, Vr)) in enumerate(zip(target, expected)):
        np.testing.assert_allclose(b.P, P, rtol=1e-4, err_msg=f"block {i} P")
        np.testing.assert_allclose(b.T, T, rtol=1e-4, err_msg=f"block {i} T")
        np.testing.assert_allclose(b.Vx, Vx, rtol=1e-4, atol=1e-3)
        np.testing.assert_allclose(b.Vr, Vr, rtol=1e-4, atol=1e-3)


def test_read_flow_transfers_mu_turb(tmp_path):
    """Turbulent viscosity comes back too, so a restart is not left with zeros."""
    grid = _make_grid()
    for i, b in enumerate(grid):
        b.set_mu_turb(np.full(b.shape, 1e-3 * (i + 1)))

    out = tmp_path / "output_ts4.hdf5"
    out_avg = tmp_path / "output_ts4_avg.hdf5"
    _write_ts4_file(out, grid)
    _write_ts4_file(out_avg, grid, avg=True)

    target = _make_grid()
    for b in target:
        b.set_mu_turb(np.zeros(b.shape))

    io4.read_flow(target, str(out), str(out_avg))

    for i, b in enumerate(target):
        np.testing.assert_allclose(b.mu_turb, 1e-3 * (i + 1), rtol=1e-5)


def test_read_flow_requires_averaged_output(tmp_path):
    """A missing averaged file is reported plainly, not as a KeyError."""
    grid = _make_grid()
    out = tmp_path / "output_ts4.hdf5"
    _write_ts4_file(out, grid)

    with pytest.raises(FileNotFoundError, match="averaged"):
        io4.read_flow(grid, str(out), str(tmp_path / "missing.hdf5"))


def test_read_flow_diverged_density(tmp_path):
    """A non-physical solution is rejected rather than stored."""
    grid = _make_grid()
    out = tmp_path / "output_ts4.hdf5"
    out_avg = tmp_path / "output_ts4_avg.hdf5"
    _write_ts4_file(out, grid)
    _write_ts4_file(out_avg, grid, avg=True)
    with h5py.File(out_avg, "r+") as f:
        rho = f["node_array_ro"][:]
        rho[0] = -1.0
        f["node_array_ro"][:] = rho

    with pytest.raises(ValueError, match="density"):
        io4.read_flow(grid, str(out), str(out_avg))


# --- log parsing ------------------------------------------------------------

_LOG = """\
STARTING TS4
RK LOOP NO: 1
TOTAL TIME: 1.5e-1
ROVX DELTA:  1.0e-2
INLET:  0.5 3.3e5 1.30e5
OUTLET:  0.51 3.31e5 1.01e5
RK LOOP NO: 2
TOTAL TIME: 2.5e-1
ROVX DELTA:  1.0e-4
INLET:  0.52 3.3e5 1.30e5
OUTLET:  0.52 3.31e5 1.02e5
"""


def test_parse_ts4_log_reads_all_steps(tmp_path):
    """Both completed steps are recorded, with the log's residuals."""
    path = tmp_path / "log_ts4.txt"
    path.write_text(_LOG)

    hist = parse_ts4_log(str(path), _make_grid(n_block=1))

    assert hist.i_log + 1 == 2
    np.testing.assert_allclose(hist.residual[:2, 0], [1.0e-2, 1.0e-4], rtol=1e-6)


def test_parse_ts4_log_scales_mass_flow_by_blade_count(tmp_path):
    """The log's per-passage mass flow becomes a whole-annulus value."""
    path = tmp_path / "log_ts4.txt"
    path.write_text(_LOG)
    grid = _make_grid(n_block=1)

    hist = parse_ts4_log(str(path), grid)

    fluid = grid.patches.outlet[0].block_view.fluid
    mdot_ref = fluid.rhoV_ref * grid[0].L_ref**2
    np.testing.assert_allclose(
        hist.mdot_nd[0, 0], 0.5 * 36.0 / mdot_ref, rtol=1e-5
    )


def test_parse_ts4_log_reads_wall_time(tmp_path):
    """hist.time is the cumulative sum of each step's TOTAL TIME, not the
    time taken to parse the (already-finished) log."""
    path = tmp_path / "log_ts4.txt"
    path.write_text(_LOG)

    hist = parse_ts4_log(str(path), _make_grid(n_block=1))

    np.testing.assert_allclose(
        hist.time[:2] * hist._TIME_SCALE, [0.15, 0.4], rtol=1e-6
    )


def test_parse_ts4_log_rejects_truncated_records(tmp_path):
    """An internally inconsistent log raises rather than padding with NaN."""
    path = tmp_path / "log_ts4.txt"
    # Two OUTLET records but only one INLET.
    path.write_text(_LOG.replace("INLET:  0.52 3.3e5 1.30e5\n", ""))

    with pytest.raises(ValueError, match="inconsistent"):
        parse_ts4_log(str(path), _make_grid(n_block=1))


def test_parse_ts4_log_rejects_empty_log(tmp_path):
    """A log with no completed step is an error, not an empty history."""
    path = tmp_path / "log_ts4.txt"
    path.write_text("STARTING TS4\nRK LOOP NO: 1\n")

    with pytest.raises(ValueError, match="No completed steps"):
        parse_ts4_log(str(path), _make_grid(n_block=1))


# --- divergence detection ---------------------------------------------------


def test_check_nan_finds_step(tmp_path):
    """The reported step is the last RK loop banner before the NaN."""
    path = tmp_path / "log_ts4.txt"
    path.write_text("RK LOOP NO: 7\nsome output\ndetected NAN\n")
    assert ts4_mod._check_nan(str(path)) == 7


def test_check_nan_clean_log(tmp_path):
    """A clean log reports zero."""
    path = tmp_path / "log_ts4.txt"
    path.write_text("RK LOOP NO: 7\nall well\n")
    assert ts4_mod._check_nan(str(path)) == 0


def test_check_nan_before_any_step(tmp_path):
    """A NaN before the first step banner is still detected."""
    path = tmp_path / "log_ts4.txt"
    path.write_text("detected NAN\n")
    assert ts4_mod._check_nan(str(path)) == -1


# --- end to end -------------------------------------------------------------

# The mock converter reads the TS3 file back through the plugin's own reader
# rather than reimplementing the on-disk layout. It stands in for opaque vendor
# code, so what matters is that it emits the node cloud the real converter is
# required to emit, not how it gets there.
_CONVERT_BODY = {
    # Faithful mock: one TS4 node per TS3 node, in block order.
    "happy": """
    import sys
    import h5py
    import numpy as np
    from ember.plugins.ts import read_ts3

    src, stem = sys.argv[1], sys.argv[2]
    grid = read_ts3(src)
    x = np.concatenate([b.x.flatten() for b in grid])
    y = np.concatenate([b.y.flatten() for b in grid])
    z = np.concatenate([b.z.flatten() for b in grid])

    with h5py.File(f"{stem}.hdf5", "w") as f:
        f.create_dataset("node_array_x", data=x)
        f.create_dataset("node_array_y", data=y)
        f.create_dataset("node_array_z", data=z)
        f.create_dataset("node_array_wdist", data=np.zeros(x.size))
        g = f.create_group("bcell_names")
        g.attrs["names"] = ["Inlet", "Outlet"]
    """,
    # A converter that merged coincident interface nodes, breaking the
    # one-node-per-node invariant the wall-distance transfer relies on.
    "merged": """
    import sys
    import h5py
    import numpy as np
    from ember.plugins.ts import read_ts3

    src, stem = sys.argv[1], sys.argv[2]
    grid = read_ts3(src)
    n = grid.size - 4

    with h5py.File(f"{stem}.hdf5", "w") as f:
        f.create_dataset("node_array_x", data=np.zeros(n))
        f.create_dataset("node_array_y", data=np.zeros(n))
        f.create_dataset("node_array_z", data=np.zeros(n))
        f.create_dataset("node_array_wdist", data=np.zeros(n))
        g = f.create_group("bcell_names")
        g.attrs["names"] = ["Inlet", "Outlet"]
    """,
}

_TURBOSTREAM_BODY = {
    # Identity "solution": copy the input nodes into both output files.
    "happy": """
    import sys
    import h5py
    import numpy as np

    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    inp, stem = args[1], args[4]

    cp, ga = 1005.0, 1.4
    cv = cp / ga
    with h5py.File(inp, "r") as f:
        x = f["node_array_x"][:]
        y = f["node_array_y"][:]
        z = f["node_array_z"][:]

    rho = np.full(x.size, 1.3)
    Vx = np.full(x.size, 120.0)
    T = np.full(x.size, 320.0)
    roe = rho * (cv * T + 0.5 * Vx**2)

    for name in (f"{stem}.hdf5", f"{stem}_avg.hdf5"):
        with h5py.File(name, "w") as f:
            f.create_dataset("node_array_x", data=x)
            f.create_dataset("node_array_y", data=y)
            f.create_dataset("node_array_z", data=z)
            f.create_dataset("node_array_ro", data=rho)
            f.create_dataset("node_array_rovx", data=rho * Vx)
            f.create_dataset("node_array_rovy", data=np.zeros(x.size))
            f.create_dataset("node_array_rovz", data=np.zeros(x.size))
            f.create_dataset("node_array_roe", data=roe)
            f.create_dataset("node_array_turb0", data=np.full(x.size, 2e-3))

    print("RK LOOP NO: 1")
    print("TOTAL TIME: 1.0e-1")
    print("ROVX DELTA:  1.0e-3")
    print("INLET:  0.5 3.3e5 1.30e5")
    print("OUTLET:  0.5 3.3e5 1.01e5")
    """,
    # TS4 detects NaN in-process and stops of its own accord.
    "nan": """
    import sys
    print("RK LOOP NO: 42")
    print("detected NAN")
    """,
}


def _write_mock_env(tmp_path, convert_mode="happy", solver_mode="happy"):
    """Mock environment script putting fake converter/mpirun/solver on PATH."""
    bindir = tmp_path / "bin"
    bindir.mkdir(exist_ok=True)

    # mpirun: drop "-np N" style flags, then exec the rest.
    mpirun = bindir / "mpirun"
    mpirun.write_text(
        '#!/bin/sh\nwhile [ "${1#-}" != "$1" ]; do shift 2; done\nexec "$@"\n'
    )
    mpirun.chmod(mpirun.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

    convert = tmp_path / "_convert.py"
    convert.write_text(textwrap.dedent(_CONVERT_BODY[convert_mode]))

    solver = tmp_path / "_turbostream4.py"
    solver.write_text(textwrap.dedent(_TURBOSTREAM_BODY[solver_mode]))

    # The driver invokes the converter by absolute path and the solver through
    # $TSHOME/$TSDIR; point both at the mocks.
    tshome = tmp_path / "ts"
    (tshome / "dir" / "bin").mkdir(parents=True, exist_ok=True)
    (tshome / "dir" / "bin" / "turbostream.py").write_text(
        f'import runpy, sys\nrunpy.run_path("{solver}", run_name="__main__")\n'
    )

    env = tmp_path / "env.sh"
    env.write_text(
        f'export PATH="{bindir}:$PATH"\n'
        f'export TSHOME="{tshome}"\n'
        'export TSDIR="dir"\n'
    )
    return env, convert


def test_run_end_to_end_via_mock_env(tmp_path, monkeypatch):
    """run() drives write -> convert -> execute -> read against mocks."""
    grid = _make_grid()
    env, convert = _write_mock_env(tmp_path)
    monkeypatch.setattr(ts4_mod, "_CONVERT_SCRIPT", str(convert))

    cfg = TS4Solver(nstep=100, nstep_avg=20)
    cfg.environment_script = env
    cfg.workdir = tmp_path

    hist = cfg.run(grid)

    # Every stage left its artefact behind.
    assert (tmp_path / "input.hdf5").exists()
    assert (tmp_path / "config.ofp").exists()
    assert (tmp_path / "input_ts4.hdf5").exists()
    assert (tmp_path / "throttle_config.ofp").exists()
    assert (tmp_path / "probes.ofp").exists()
    assert (tmp_path / "log_ts4.txt").exists()

    # The mock's uniform "solution" came back onto the grid.
    for b in grid:
        np.testing.assert_allclose(b.T, 320.0, rtol=1e-3)
        np.testing.assert_allclose(b.Vx, 120.0, rtol=1e-3)
        np.testing.assert_allclose(b.mu_turb, 2e-3, rtol=1e-3)

    # The wall distance written into the converted file is the grid's own.
    with h5py.File(tmp_path / "input_ts4.hdf5", "r") as f:
        wdist = f["node_array_wdist"][:]
    np.testing.assert_allclose(
        wdist, np.concatenate([b.wdist.flatten() for b in grid]), rtol=1e-5
    )

    assert hist is not None
    assert hist.i_log + 1 == 1


def test_run_diverged_raises(tmp_path, monkeypatch):
    """TS4's own NaN report surfaces as DivergenceError, naming the step."""
    grid = _make_grid()
    env, convert = _write_mock_env(tmp_path, solver_mode="nan")
    monkeypatch.setattr(ts4_mod, "_CONVERT_SCRIPT", str(convert))

    cfg = TS4Solver(nstep=100, nstep_avg=20)
    cfg.environment_script = env
    cfg.workdir = tmp_path

    with pytest.raises(DivergenceError, match="step 42"):
        cfg.run(grid)


def test_run_rejects_converter_that_merged_nodes(tmp_path, monkeypatch):
    """A conversion that did not preserve nodes fails loudly through run()."""
    grid = _make_grid()
    env, convert = _write_mock_env(tmp_path, convert_mode="merged")
    monkeypatch.setattr(ts4_mod, "_CONVERT_SCRIPT", str(convert))

    cfg = TS4Solver(nstep=100, nstep_avg=20)
    cfg.environment_script = env
    cfg.workdir = tmp_path

    with pytest.raises(TS4ConversionError, match="one-for-one"):
        cfg.run(grid)


def test_run_requires_workdir():
    """run() without a workdir is a configuration error, not a crash."""
    with pytest.raises(ValueError, match="workdir"):
        TS4Solver().run(_make_grid())


def test_run_restores_cwd_on_failure(tmp_path, monkeypatch):
    """A failed run does not strand the process in the working directory."""
    grid = _make_grid()
    env, convert = _write_mock_env(tmp_path, solver_mode="nan")
    monkeypatch.setattr(ts4_mod, "_CONVERT_SCRIPT", str(convert))

    cfg = TS4Solver(nstep=100, nstep_avg=20)
    cfg.environment_script = env
    cfg.workdir = tmp_path

    cwd_before = os.getcwd()
    with pytest.raises(DivergenceError):
        cfg.run(grid)
    assert os.getcwd() == cwd_before
