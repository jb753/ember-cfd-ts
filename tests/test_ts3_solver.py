"""Tests for the ``ember.plugins.ts.TS3Solver`` run driver.

These cover the pieces that are testable without a Turbostream binary or the
HPC: the module imports, the config dataclass behaves, the input file is written
with the right variables, the output round-trips back into a grid, and the whole
write -> execute -> read path runs against a mock ``turbostream`` on ``PATH``.
"""

import os
import stat
import sys
import textwrap
from pathlib import Path

import h5py
import numpy as np
import pytest

from ember.convergence_history import ConvergenceHistory

import ember.util
from ember.block import Block
from ember.fluid import PerfectFluid
from ember.grid import Grid, DivergenceError
from ember.patch import InletPatch, OutletPatch

import ember.plugins.ts._solver as ts3_mod
import ember.plugins.ts._log as ts_log
from ember.plugins.ts import TS3Solver, parse_ts3_log, write_ts3

_DATA = Path(__file__).parent / "data"


def _make_solved_grid(T_dtm=300.0):
    """Single-block ember grid on a general datum with a known flow field."""
    # The block must span exactly one whole pitch: the inlet is non-reflecting,
    # and its pitchwise node weights are checked to sum to 1.
    nblade = 32
    shape = (3, 4, 5)
    block = Block(shape=shape)
    xrt = ember.util.linmesh3(
        [0.0, 1.0], [0.5, 1.5], [0.0, 2.0 * np.pi / nblade], shape
    )
    block.set_x(xrt[..., 0])
    block.set_r(xrt[..., 1])
    block.set_t(xrt[..., 2])
    block.set_fluid(PerfectFluid(cp=1005.0, gamma=1.4, mu=1.8e-5, Pr=0.7, T_dtm=T_dtm))
    block.set_rpm(0.0)
    block.set_Nb(nblade)
    block.set_P_T(np.full(shape, 1.2e5), np.full(shape, 320.0))
    block.set_Vxrt(
        np.stack(
            [120.0 * np.ones(shape), 5.0 * np.ones(shape), -20.0 * np.ones(shape)],
            axis=-1,
        )
    )
    block.set_mu_turb(np.full(shape, 1e-3))
    block.set_wdist(np.full(shape, 0.01, dtype=np.float32))
    return Grid([block])


def _make_grid_with_inlet():
    """Solved single-block grid with an inlet and outlet patch attached."""
    grid = _make_solved_grid()
    block = grid[0]
    inlet = InletPatch(i=0, j=(0, -1), k=(0, -1), label="inlet")
    outlet = OutletPatch(i=-1, j=(0, -1), k=(0, -1), label="outlet")
    block.patches.append(inlet)
    block.patches.append(outlet)
    inlet.set_Po_To(Po=1.2e5, To=320.0)
    inlet.set_Alpha(0.0)
    inlet.set_Beta(0.0)
    outlet.set_P(1.1e5)
    return grid


def _write_outputs(grid, workdir):
    """Write the grid as the two TS3 output files _read_hdf5 expects."""
    write_ts3(grid, str(workdir / "output_avg.hdf5"))
    write_ts3(grid, str(workdir / "output.hdf5"))


def test_module_imports():
    """TS3Solver is exported and is a BaseSolver implementing run(grid)."""
    from ember.solver import BaseSolver

    assert issubclass(TS3Solver, BaseSolver)
    assert hasattr(ts3_mod, "TS3Solver")


def test_config_instantiation_defaults():
    """The TS3Solver config dataclass instantiates with its documented defaults."""
    cfg = TS3Solver()
    assert cfg.cfl == 0.4
    assert cfg.ilos == 2
    assert cfg.n_step == 10000


def _av(f, name):
    """Read a scalar application variable from an open TS3 hdf5 file."""
    return float(np.squeeze(f[f"{name}_av"][...]))


def _bv(f, bid, name):
    """Read a scalar block variable from an open TS3 hdf5 file."""
    return float(np.squeeze(f[f"block{bid}"][f"{name}_bv"][...]))


def test_write_input_av_bv_roundtrip(tmp_path):
    """_write_input forwards typed and derived av/bv into input.hdf5."""
    cfg = TS3Solver(n_step=8000, n_step_avg=2000, cfl=0.35, fmgrid=0.15)
    ts3_mod._write_input(_make_grid_with_inlet(), cfg, tmp_path)

    with h5py.File(tmp_path / "input.hdf5", "r") as f:
        assert _av(f, "cfl") == pytest.approx(0.35)
        assert _av(f, "nstep") == pytest.approx(8000)
        assert _av(f, "nstep_save_start") == pytest.approx(6000)
        assert _av(f, "restart") == pytest.approx(1)
        assert _bv(f, 0, "fmgrid") == pytest.approx(0.15)


def test_write_input_rfin_pv(tmp_path):
    """rfin is applied to inlet patches as a patch variable."""
    cfg = TS3Solver(rfin=0.25)
    ts3_mod._write_input(_make_grid_with_inlet(), cfg, tmp_path)

    with h5py.File(tmp_path / "input.hdf5", "r") as f:
        # Inlet is patch0 (rotating patches excluded; none here).
        rfin = float(np.squeeze(f["block0"]["patch0"]["rfin_pv"][...]))
    assert rfin == pytest.approx(0.25)


def test_write_input_raw_av_override(tmp_path):
    """A raw av override with no typed field lands in the file."""
    cfg = TS3Solver(av={"sfin_sa": 0.1})
    ts3_mod._write_input(_make_grid_with_inlet(), cfg, tmp_path)

    with h5py.File(tmp_path / "input.hdf5", "r") as f:
        assert _av(f, "sfin_sa") == pytest.approx(0.1)


def test_write_input_raw_bv_override(tmp_path):
    """A raw bv override with no typed field lands in the file."""
    cfg = TS3Solver(bv={0: {"free_turb": 0.02}})
    ts3_mod._write_input(_make_grid_with_inlet(), cfg, tmp_path)

    with h5py.File(tmp_path / "input.hdf5", "r") as f:
        assert _bv(f, 0, "free_turb") == pytest.approx(0.02)


def test_write_input_av_overlap_errors(tmp_path):
    """A raw av override colliding with a non-default typed field errors."""
    cfg = TS3Solver(cfl=0.2, av={"cfl": 0.3})
    with pytest.raises(ValueError, match="cfl"):
        ts3_mod._write_input(_make_grid_with_inlet(), cfg, tmp_path)


def test_write_input_bv_overlap_errors(tmp_path):
    """A raw bv override colliding with a non-default typed field errors."""
    cfg = TS3Solver(fmgrid=0.1, bv={0: {"fmgrid": 0.0}})
    with pytest.raises(ValueError, match="fmgrid"):
        ts3_mod._write_input(_make_grid_with_inlet(), cfg, tmp_path)


def test_write_input_av_overlap_ok_when_default(tmp_path):
    """An av override matching a typed field at its default is allowed."""
    # cfl left at its default, so the av dict is the sole source.
    cfg = TS3Solver(av={"cfl": 0.3})
    ts3_mod._write_input(_make_grid_with_inlet(), cfg, tmp_path)
    with h5py.File(tmp_path / "input.hdf5", "r") as f:
        assert _av(f, "cfl") == pytest.approx(0.3)


def test_run_writes_input(tmp_path, monkeypatch):
    """_run writes input.hdf5 (no longer raising NotImplementedError)."""
    monkeypatch.setattr(ts3_mod, "_execute", lambda cfg, workdir: None)
    monkeypatch.setattr(ts3_mod, "_read_hdf5", lambda grid, cfg, workdir: None)
    cfg = TS3Solver()
    ts3_mod._run(_make_grid_with_inlet(), cfg, tmp_path)
    assert (tmp_path / "input.hdf5").exists()


def test_read_hdf5_roundtrip(tmp_path):
    """_read_hdf5 loads the TS3 output back onto an existing grid via ember."""
    # Reference solution and the dimensional state we expect to recover.
    ref = _make_solved_grid()
    P, T = ref[0].P.copy(), ref[0].T.copy()
    Vx, mu = ref[0].Vx.copy(), ref[0].mu_turb.copy()

    # Writing zeroes the datum in place, so use a separate grid for the files.
    _write_outputs(_make_solved_grid(), tmp_path)

    # A blank target grid on the original datum receives the solution.
    target = _make_solved_grid()
    target[0].set_P_T(np.full(target[0].shape, 1e5), np.full(target[0].shape, 300.0))

    cfg = TS3Solver()
    ts3_mod._read_hdf5(target, cfg, tmp_path)

    np.testing.assert_allclose(target[0].P, P, rtol=1e-4)
    np.testing.assert_allclose(target[0].T, T, rtol=1e-4)
    np.testing.assert_allclose(target[0].Vx, Vx, rtol=1e-4, atol=1e-3)
    np.testing.assert_allclose(target[0].mu_turb, mu, rtol=1e-5)


def test_read_hdf5_diverged_raises_convergence_error(tmp_path):
    """A diverged (negative-density) output surfaces as DivergenceError."""
    _write_outputs(_make_solved_grid(), tmp_path)
    with h5py.File(tmp_path / "output_avg.hdf5", "r+") as f:
        ro = f["block0"]["ro_bp"]
        ro[...] = -np.abs(ro[...]) - 1.0

    cfg = TS3Solver()
    with pytest.raises(DivergenceError):
        ts3_mod._read_hdf5(_make_solved_grid(), cfg, tmp_path)


def test_read_hdf5_missing_output_raises(tmp_path):
    """No output file present raises a clear error."""
    cfg = TS3Solver()
    with pytest.raises(Exception, match="No Turbostream output file"):
        ts3_mod._read_hdf5(_make_solved_grid(), cfg, tmp_path)


def test_convergence_history_from_log():
    """parse_ts3_log builds an ember ConvergenceHistory from the TS3 log + grid.

    Exercises the exact post-run call ``TS3Solver.run`` makes — log parsing lives
    in ember.plugins.ts; the grid supplies the reference scales the log lacks
    (V_ref, T_ref, areas) and is checked for gas-property consistency against the
    log header.
    """
    grid = _make_grid_with_inlet()
    # Match the log_duct.txt header so parse_ts3_log's consistency check passes.
    grid[0].set_fluid(PerfectFluid(cp=1005.0, gamma=1.4, mu=1.0e-3, Pr=0.72))
    conv = parse_ts3_log(_DATA / "log_duct.txt", grid)
    assert isinstance(conv, ConvergenceHistory)
    # All 399 step blocks parsed.
    assert conv.i_log + 1 == 399
    # Grid-derived metadata is stamped, and the per-step monitors the parser
    # fills (the DAVG residual in column 0, and station mass flows) are finite --
    # confirming the grid reference scales were applied.
    assert conv._get_metadata_by_key("n_node") == grid.size
    assert np.all(np.isfinite(np.asarray(conv.residual)[:, 0]))
    assert np.all(np.isfinite(np.asarray(conv.mdot_nd)))


# ----------------------------------------------------------------------------
# End-to-end tests through the real _execute subprocess, using a mock
# Turbostream environment in place of the binary. The only OS coupling in the
# TS3 path is the sourced command
# ". {env}; mpirun ... turbostream input.hdf5 output 1 > log.txt", so faking
# mpirun/turbostream on PATH exercises _write_input -> _execute -> _read_hdf5
# unchanged. See plan: quizzical-swimming-gem.
# ----------------------------------------------------------------------------

# Behaviour of the mock turbostream, switched per test. Runs under the test's
# own interpreter (sys.executable) so h5py is available for the diverged case.
# CWD is the work directory (ts3._execute chdir's there before launching).
_TURBOSTREAM_BODY = {
    # Pretend to converge: echo a normal log line and copy the input we were
    # handed straight to the two output files (an identity "solution").
    "happy": """
        print("OUTER STEP No. 1", flush=True)
        print("mock run complete", flush=True)
        shutil.copy(sys.argv[1], "output_avg.hdf5")
        shutil.copy(sys.argv[1], "output.hdf5")
    """,
    # Diverge mid-run: "NAN" must be the first line of the log for _check_nan's
    # anchored regex to match, and the step line must be visible in the same read
    # (write both in one flush to avoid a partial-read race), so the reported step
    # is deterministic. Stay alive so the parent's monitor catches it.
    "nan": """
        print("NAN\\nOUTER STEP No. 5", flush=True)
        time.sleep(30)
    """,
    # Exit cleanly but write a non-physical (negative-density) solution, which
    # the plugin's read_conserved rejects on the way back in.
    "diverged": """
        shutil.copy(sys.argv[1], "output_avg.hdf5")
        shutil.copy(sys.argv[1], "output.hdf5")
        with h5py.File("output_avg.hdf5", "r+") as f:
            ro = f["block0"]["ro_bp"]
            ro[...] = -np.abs(ro[...]) - 1.0
    """,
}


def _write_mock_env(tmp_path, mode):
    """Write a mock TS3 environment script and return its path.

    Sourcing the returned script prepends a ``bin/`` holding fake ``mpirun`` and
    ``turbostream`` programs to ``PATH``; ``mode`` selects the turbostream
    behaviour (``"happy"``, ``"nan"`` or ``"diverged"``).
    """
    bindir = tmp_path / "bin"
    bindir.mkdir()

    # mpirun: drop "-npernode N -np N" style flags, then exec the rest.
    mpirun = bindir / "mpirun"
    mpirun.write_text(
        '#!/bin/sh\nwhile [ "${1#-}" != "$1" ]; do shift 2; done\nexec "$@"\n'
    )

    # The real work lives in a Python script run under this interpreter.
    script = tmp_path / "_turbostream.py"
    script.write_text(
        "import shutil, sys, time\n"
        "import numpy as np\n"
        "import h5py\n" + textwrap.dedent(_TURBOSTREAM_BODY[mode])
    )
    turbostream = bindir / "turbostream"
    turbostream.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{script}" "$@"\n')

    for f in (mpirun, turbostream):
        f.chmod(f.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

    env = tmp_path / "env.sh"
    env.write_text(f'export PATH="{bindir}:$PATH"\n')
    return env


def test_run_end_to_end_via_mock_env(tmp_path):
    """_run drives write -> _execute(mock) -> read and round-trips the flow."""
    grid = _make_grid_with_inlet()
    P, T, Vx = grid[0].P.copy(), grid[0].T.copy(), grid[0].Vx.copy()

    cfg = TS3Solver(n_step=8000, n_step_avg=2000, environment_script=str(_write_mock_env(tmp_path, "happy")))

    ts3_mod._run(grid, cfg, tmp_path)

    # The real subprocess ran: it produced the output and _execute removed input.
    assert (tmp_path / "log.txt").exists()
    assert (tmp_path / "output_avg.hdf5").exists()
    assert not (tmp_path / "input.hdf5").exists()

    # Identity "solution" copied input to output, so the grid is unchanged.
    np.testing.assert_allclose(grid[0].P, P, rtol=1e-4)
    np.testing.assert_allclose(grid[0].T, T, rtol=1e-4)
    np.testing.assert_allclose(grid[0].Vx, Vx, rtol=1e-4, atol=1e-3)


def test_execute_nan_raises_convergence_error(tmp_path, monkeypatch):
    """A NAN in the log mid-run surfaces as DivergenceError, naming the step."""
    # The mock stays alive after printing NAN, so poll it fast instead of the 10s
    # production interval (the real loop, including proc.wait, is still exercised).
    monkeypatch.setattr(ts3_mod, "_POLL_INTERVAL_S", 0.02)

    cfg = TS3Solver(environment_script=str(_write_mock_env(tmp_path, "nan")))

    cwd_before = os.getcwd()
    with pytest.raises(DivergenceError, match="step 5"):
        ts3_mod._execute(cfg, tmp_path)
    # The try/finally in _execute restores the working directory on the error path.
    assert os.getcwd() == cwd_before


def test_run_diverged_solution_raises(tmp_path):
    """A clean exit with a non-physical output raises DivergenceError on read."""
    cfg = TS3Solver(environment_script=str(_write_mock_env(tmp_path, "diverged")))

    cwd_before = os.getcwd()
    with pytest.raises(DivergenceError):
        ts3_mod._run(_make_grid_with_inlet(), cfg, tmp_path)
    assert os.getcwd() == cwd_before


def test_archive_log_increments_suffix(tmp_path):
    """archive_log renames log.txt to the next free log_NNN.txt, keeping all."""
    log = tmp_path / "log.txt"

    # No log present: no-op.
    ts_log.archive_log(tmp_path)
    assert not list(tmp_path.glob("log_*.txt"))

    # First archive -> log_000.txt.
    log.write_text("run 0")
    ts_log.archive_log(tmp_path)
    assert not log.exists()
    assert (tmp_path / "log_000.txt").read_text() == "run 0"

    # Second archive advances past the existing index -> log_001.txt.
    log.write_text("run 1")
    ts_log.archive_log(tmp_path)
    assert (tmp_path / "log_001.txt").read_text() == "run 1"

    # A legacy log_old.txt is ignored by the numbering.
    (tmp_path / "log_old.txt").write_text("legacy")
    log.write_text("run 2")
    ts_log.archive_log(tmp_path)
    assert (tmp_path / "log_002.txt").read_text() == "run 2"
    assert (tmp_path / "log_old.txt").read_text() == "legacy"


def test_run_keeps_a_log_per_run(tmp_path):
    """run() archives the previous log, so two runs in one workdir keep both."""
    grid = _make_grid_with_inlet()
    cfg = TS3Solver(environment_script=str(_write_mock_env(tmp_path, "happy")))

    cfg.run(grid, tmp_path)  # writes log.txt
    cfg.run(grid, tmp_path)  # archives the previous log -> log_000.txt, writes a fresh log.txt

    assert (tmp_path / "log.txt").exists()
    assert (tmp_path / "log_000.txt").exists()


def test_soft_detunes_a_copy():
    solver = TS3Solver()
    soft = solver.soft()
    assert soft is not solver and solver.cfl == 0.4
    assert soft.cfl < solver.cfl and soft.fmgrid < solver.fmgrid
    assert soft.dampin < solver.dampin and soft.facsecin > solver.facsecin
    assert soft.n_step == solver.n_step


def test_restart_drops_only_the_start_up_ramp():
    solver = TS3Solver(nchange=2000)
    restart = solver.restart()
    assert restart is not solver and solver.nchange == 2000
    assert restart.nchange == 0
    assert TS3Solver(nchange=0) == restart


def test_soft_keeps_settings_already_more_robust():
    solver = TS3Solver(cfl=0.1, fmgrid=0.0)
    soft = solver.soft()
    assert soft.cfl == 0.1 and soft.fmgrid == 0.0
