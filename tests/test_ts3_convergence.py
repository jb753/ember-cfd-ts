"""Tests for parse_ts3_log: reconstructing a ConvergenceHistory from a TS3 log.

Moved out of core ember's test_convergence_history.py when the TS3 machinery
became the ember-cfd-ts plugin. The former ConvergenceHistory.from_ts3
classmethod is now ember.plugins.ts.parse_ts3_log(filename, grid).
"""

from pathlib import Path

import numpy as np
import pytest

from ember.fluid import PerfectFluid
from ember.plugins.ts import parse_ts3_log
from ember.plugins.ts._convergence import _half_last_digit

_DATA = Path(__file__).parent / "data"


def _make_duct_grid():
    """Single-block grid with inlet/outlet and a uniform flow field.

    Supplies from_ts3 with reference scales: a known mean axial velocity sets
    V_ref and a known outlet temperature sets T_ref. The geometry is otherwise
    arbitrary; the per-step history comes from the log, not the grid.
    """
    import ember.util
    from ember.block import Block
    from ember.grid import Grid
    from ember.patch import InletPatch, OutletPatch

    # Gas properties match the log_duct.txt header so from_ts3's grid/log
    # consistency check passes (cp=1005, ga=1.4, viscosity=0.001, prandtl=0.72).
    fluid = PerfectFluid(cp=1005.0, gamma=1.4, mu=1.0e-3, Pr=0.72)

    # The block must span exactly one whole pitch: the inlet is non-reflecting,
    # and its pitchwise node weights are checked to sum to 1.
    nblade = 32
    shape = (4, 4, 4)
    block = Block(shape=shape)
    xrt = ember.util.linmesh3(
        [0.0, 1.0], [0.5, 1.5], [0.0, 2.0 * np.pi / nblade], shape
    )
    block.set_x(xrt[..., 0])
    block.set_r(xrt[..., 1])
    block.set_t(xrt[..., 2])
    block.set_fluid(fluid)
    block.set_rpm(0.0)
    block.set_Nb(nblade)
    block.set_P_T(np.full(shape, 1.0e5), np.full(shape, 300.0))
    block.set_Vxrt(
        np.stack([np.full(shape, 100.0), np.zeros(shape), np.zeros(shape)], axis=-1)
    )
    inlet = InletPatch(i=0, j=(0, -1), k=(0, -1), label="inlet")
    outlet = OutletPatch(i=-1, j=(0, -1), k=(0, -1), label="outlet")
    block.patches.append(inlet)
    block.patches.append(outlet)
    inlet.set_Po_To(Po=1.0e5, To=300.0)
    inlet.set_Alpha(0.0)
    inlet.set_Beta(0.0)
    outlet.set_P(0.9e5)
    return Grid([block])


@pytest.fixture(scope="module")
def ts3_hist():
    """ConvergenceHistory parsed from the duct log file with a grid for scales."""
    return parse_ts3_log(_DATA / "log_duct.txt", _make_duct_grid())


def test_from_ts3_step_count(ts3_hist):
    """from_ts3 parses all 399 STEP blocks."""
    assert ts3_hist.i_log + 1 == 399


def test_from_ts3_steps_monotonic(ts3_hist):
    """Steps array starts at 50 and is strictly increasing."""
    n = ts3_hist.i_log + 1
    steps = ts3_hist.i_step[:n]
    assert steps[0] == 50
    assert np.all(np.diff(steps) > 0)


def test_from_ts3_metadata(ts3_hist):
    """Grid-derived metadata: single non-rotating row, no kinetic-energy scales."""
    assert ts3_hist._get_metadata_by_key("n_row") == 1
    assert "V_ref" not in ts3_hist._metadata
    assert "T_ref" not in ts3_hist._metadata


def test_from_ts3_mdot_finite(ts3_hist):
    """mdot_in and mdot_out are finite for all recorded steps."""
    n = ts3_hist.i_log + 1
    assert np.all(np.isfinite(ts3_hist.mdot_nd[:n, 0]))
    assert np.all(np.isfinite(ts3_hist.mdot_nd[:n, -1]))


def test_from_ts3_first_step_mdot(ts3_hist):
    """First-step mass flows match the log file values."""
    assert ts3_hist.mdot_nd[0, 0] == pytest.approx(1.38360, rel=1e-4)
    assert ts3_hist.mdot_nd[0, -1] == pytest.approx(1.38339, rel=1e-4)


def test_from_ts3_residual_drho_finite(ts3_hist):
    """drho residual is finite; other residual components are NaN (not in TS3 log)."""
    n = ts3_hist.i_log + 1
    assert np.all(np.isfinite(ts3_hist.residual[:n, 0]))


def test_from_ts3_ho_s_finite(ts3_hist):
    """Inlet and outlet stagnation enthalpy and entropy are finite for all steps."""
    n = ts3_hist.i_log + 1
    for name in ("ho_nd", "s_nd"):
        stations = getattr(ts3_hist, name)[:n]
        assert np.all(np.isfinite(stations[:, 0])), f"{name} inlet"
        assert np.all(np.isfinite(stations[:, -1])), f"{name} outlet"


def test_from_ts3_zeta_positive(ts3_hist):
    """Total pressure loss coefficient zeta is positive for all steps."""
    n = ts3_hist.i_log + 1
    assert np.all(ts3_hist.zeta[:n] > 0)


def test_from_ts3_fluid_mismatch_raises():
    """A grid whose fluid disagrees with the log header is rejected."""
    grid = _make_duct_grid()
    # Re-set the fluid to a different cp than the log header records.
    grid[0].set_fluid(PerfectFluid(cp=1100.0, gamma=1.4, mu=1.0e-3, Pr=0.72))
    with pytest.raises(ValueError, match="does not match grid"):
        parse_ts3_log(_DATA / "log_duct.txt", grid)


@pytest.mark.parametrize(
    "text, half",
    [("0.000130", 5e-7), ("1005.000000", 5e-7), ("1.5e-05", 5e-7), ("1.50e-05", 5e-8), ("2", 0.5)],
)
def test_half_last_digit_is_the_printed_rounding(text, half):
    assert _half_last_digit(text) == pytest.approx(half)


def test_from_ts3_fluid_check_allows_for_printed_rounding(tmp_path):
    """A small viscosity printed to fixed decimals keeps only a couple of
    significant figures, and must not be mistaken for the wrong grid."""
    log = (_DATA / "log_duct.txt").read_text()
    printed = "viscosity:                  0.000130"
    assert "viscosity:                  0.001000" in log
    path = tmp_path / "log.txt"
    path.write_text(log.replace("viscosity:                  0.001000", printed, 1))

    grid = _make_duct_grid()
    block = grid[0]
    # Rounds to the printed 0.000130, but is 0.15% away from it
    block.set_fluid(
        PerfectFluid(
            cp=float(block.cp.mean()),
            gamma=float(block.gamma.mean()),
            mu=0.000129813,
            Pr=float(block.Pr.mean()),
        )
    )
    parse_ts3_log(path, grid)
