"""Reconstruct an ember ConvergenceHistory from a Turbostream 3 text log.

Extracted from the former ``ConvergenceHistory.from_ts3`` classmethod when the
TS3 machinery moved out of core ember into this plugin. It is implemented as a
free function here rather than a method so core ember carries no TS3 knowledge.

Note: this function drives a few ``ConvergenceHistory`` internals
(``_set_grid_metadata``, ``_set_metadata_by_key``, ``_set_data_by_keys``,
``_TIME_SCALE``). That coupling to core internals is deliberate and accepted for
this first-party plugin; keep it in sync with core if those helpers change.
"""

import re
import time as _time

import numpy as np

from ember.convergence_history import ConvergenceHistory
from ember.fluid import PerfectFluid

f32 = np.float32


def parse_ts3_log(filename, grid):
    """Reconstruct a ConvergenceHistory from a TS3 text log file.

    The per-step history (residuals, mass flows, stagnation conditions) is
    parsed from the log; the reference scales needed to non-dimensionalize
    it (areas, V_ref, T_ref, node count) are derived from `grid`, which the
    log does not record. The fluid is taken from the log header, the
    authoritative record of what TS3 actually ran with.

    Parameters
    ----------
    filename : str
        Path to TS3 log file (e.g. log_duct.txt)
    grid : ember.grid.Grid
        The grid that was solved, for reference scales (V_ref, T_ref, areas).

    Returns
    -------
    ember.convergence_history.ConvergenceHistory
    """
    with open(filename, "r") as f:
        text = f.read()

    # --- Parse header (before main loop) ---
    header_match = re.search(
        r"APPLICATION VARIABLES:(.*?)STARTING THE MAIN TIME STEPPING LOOP",
        text,
        re.DOTALL,
    )
    if header_match is None:
        raise ValueError("Could not find APPLICATION VARIABLES header in log")
    header_text = header_match.group(1)

    def _get_var(name, txt):
        m = re.search(r"^\s+" + name + r":\s+([\d.Ee+-]+)", txt, re.MULTILINE)
        if m is None:
            raise ValueError(f"Could not find '{name}' in header")
        return float(m.group(1))

    cp = _get_var("cp", header_text)
    ga = _get_var("ga", header_text)
    mu = _get_var("viscosity", header_text)
    Pr = _get_var("prandtl", header_text)

    # Guard against a grid/log mismatch: the gas properties recorded in the
    # log header must agree with the solved grid's fluid (grid[0].cp etc.
    # are spatially constant for a perfect gas, so compare their means).
    block0 = grid[0]
    for name, log_val, grid_val in (
        ("cp", cp, block0.cp),
        ("ga", ga, block0.gamma),
        ("viscosity", mu, block0.mu),
        ("prandtl", Pr, block0.Pr),
    ):
        grid_val = float(np.mean(grid_val))
        if not np.isclose(log_val, grid_val, rtol=1e-3):
            raise ValueError(
                f"TS3 log {name}={log_val:g} does not match grid "
                f"{name}={grid_val:g}; wrong grid for this log?"
            )

    fluid = PerfectFluid(cp=cp, gamma=ga, mu=mu, Pr=Pr, T_dtm=1.0)

    # --- Split into per-step blocks (after main loop start) ---
    body = text[header_match.end() :]

    # Collect all timing values for mean dt estimation
    timing_vals = [
        float(v) for v in re.findall(r"TIME FOR \d+ STEPS = ([\d.]+)", body)
    ]
    mean_dt = np.mean(timing_vals) if timing_vals else 0.0

    # Parse step blocks: each starts with "STEP No. <n>"
    step_re = re.compile(r"STEP No\.\s+(\d+)")
    davg_re = re.compile(r"TOTAL DAVG\s+([\d.E+\-]+)")
    flows_re = re.compile(r"INLET FLOW =\s+([\d.]+)\s+OUTLET FLOW =\s+([\d.]+)")
    stagP_re = re.compile(
        r"AVG INLET STAG P =\s+([\d.]+)\s+AVG OUTLET STAG P =\s+([\d.]+)"
    )
    stagT_re = re.compile(
        r"AVG INLET STAG T =\s+([\d.]+)\s+AVG OUTLET STAG T =\s+([\d.]+)"
    )

    # Find positions of all "STEP No." matches
    step_starts = [m.start() for m in step_re.finditer(body)]

    step_blocks = []
    for idx, start in enumerate(step_starts):
        end = step_starts[idx + 1] if idx + 1 < len(step_starts) else len(body)
        chunk = body[start:end]

        i_step_m = step_re.match(chunk)
        davg_m = davg_re.search(chunk)
        flows_m = flows_re.search(chunk)
        stagP_m = stagP_re.search(chunk)
        stagT_m = stagT_re.search(chunk)

        if not all([i_step_m, davg_m, flows_m, stagP_m, stagT_m]):
            continue  # skip incomplete blocks

        step_blocks.append(
            {
                "i_step": int(i_step_m.group(1)),
                "davg": float(davg_m.group(1)),
                "mdot_in": float(flows_m.group(1)),
                "mdot_out": float(flows_m.group(2)),
                "Po_in": float(stagP_m.group(1)),
                "Po_out": float(stagP_m.group(2)),
                "To_in": float(stagT_m.group(1)),
                "To_out": float(stagT_m.group(2)),
            }
        )

    n_log = len(step_blocks)
    out = ConvergenceHistory(shape=(n_log,))

    # --- Set metadata ---
    # Reference scales and counts from the grid (the log does not record
    # them); fluid from the log header (what TS3 actually ran with).
    ConvergenceHistory._set_grid_metadata(out, grid)
    out._set_metadata_by_key("fluid", fluid)
    out._set_metadata_by_key(
        "_time_start", np.array(_time.perf_counter(), dtype=np.float64)
    )
    out._set_metadata_by_key("i_log", n_log - 1)

    # The log records only overall inlet/outlet flows, so map them to the
    # first and last through-flow stations; any interior stations stay NaN.
    n_row = len(grid.rows)
    out._set_metadata_by_key("n_row", n_row)
    st_in, st_out = 0, 2 * n_row - 1
    # mdot is non-dimensionalised by the fluid mass-flux scale, matching the
    # live grid path (Grid._station_stats); ho/s from get_h/get_s on nd
    # inputs are already non-dimensional.
    mdot_ref = fluid.rhoV_ref * grid[0].L_ref ** 2

    for i, blk in enumerate(step_blocks):
        # Navigate to this index via a temporary slice view
        view = out[i]

        view._set_data_by_keys(("i_step",), blk["i_step"])
        view._set_data_by_keys(("time",), f32(i * mean_dt / ConvergenceHistory._TIME_SCALE))

        # Convert dimensional Po, To → non-dimensional ho, s
        for side, st in (("in", st_in), ("out", st_out)):
            Po = blk[f"Po_{side}"]
            To = blk[f"To_{side}"]
            rho_nd, u_nd = fluid.set_P_T(Po / fluid.P_ref, To / fluid.T_ref)
            ho_nd = fluid.get_h(rho_nd, u_nd)
            s_nd = fluid.get_s(rho_nd, u_nd)
            view._set_data_by_keys((f"ho_st{st}",), f32(ho_nd))
            view._set_data_by_keys((f"s_st{st}",), f32(s_nd))
            view._set_data_by_keys(
                (f"mdot_st{st}",), f32(blk[f"mdot_{side}"] / mdot_ref)
            )

        # DAVG → drho; other residuals NaN
        view._set_data_by_keys(
            ("drho", "drhoVx", "drhoVr", "drhorVt", "drhoe"),
            np.array([blk["davg"], np.nan, np.nan, np.nan, np.nan], dtype=f32),
        )

    return out
