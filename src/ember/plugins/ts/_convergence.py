"""Reconstruct an ember ConvergenceHistory from a Turbostream 3 text log.

Extracted from the former ``ConvergenceHistory.from_ts3`` classmethod when the
TS3 machinery moved out of core ember into this plugin. It is implemented as a
free function here rather than a method so core ember carries no TS3 knowledge.

The history is assembled entirely through the public core API: it is sized and
seeded with :meth:`ConvergenceHistory.from_grid`, then filled one step at a time
with :meth:`ConvergenceHistory.record_convergence` (passing the log's own time
and NaN-filled monitors for quantities the log does not record).

The grid is the reference frame the history is projected onto -- it supplies the
fluid, length scale, and station/node counts that the dimensional, topology-free
log never records. The log header echoes the gas properties, which is used only
as a tripwire to confirm the caller passed the right grid for this log.
"""

import re

import numpy as np

from ember.convergence_history import ConvergenceHistory
from ember.grid import ConvergenceStep


def parse_ts3_log(filename, grid):
    """Reconstruct a ConvergenceHistory from a TS3 text log file.

    The per-step history (residuals, mass flows, stagnation conditions) is
    parsed from the log; the fluid and the reference scales needed to
    non-dimensionalize it (areas, V_ref, T_ref, node/station counts) are all
    derived from `grid`, which the log does not record. The gas properties in
    the log header are used only to check that the right grid was passed.

    Parameters
    ----------
    filename : str
        Path to TS3 log file (e.g. log_duct.txt)
    grid : ember.grid.Grid
        The grid that was solved, supplying the fluid and reference scales
        (V_ref, T_ref, areas) and validated against the log header.

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

    # The grid is the reference frame the history is projected onto: it supplies
    # the fluid, length scale, and station/node counts the log never records.
    # Use the same fluid `from_grid` stamps so metadata and monitors agree.
    outlet_block = grid.patches.outlet[0].block_view
    fluid = outlet_block.fluid

    # Tripwire (not a fluid source): the gas properties are the only field the
    # log echoes that also lives on the grid, so compare them to catch a
    # mismatched grid/log pair -- which would otherwise silently corrupt the
    # reference scaling. Block fluid is spatially constant, so compare means.
    for name, log_val, grid_val in (
        ("cp", cp, outlet_block.cp),
        ("ga", ga, outlet_block.gamma),
        ("viscosity", mu, outlet_block.mu),
        ("prandtl", Pr, outlet_block.Pr),
    ):
        grid_val = float(np.mean(grid_val))
        if not np.isclose(log_val, grid_val, rtol=1e-3):
            raise ValueError(
                f"TS3 log {name}={log_val:g} does not match grid "
                f"{name}={grid_val:g}; wrong grid for this log?"
            )

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

    # Size and seed the history through the public API; from_grid derives all
    # reference scales/counts (and the fluid) from the grid.
    n_log = len(step_blocks)
    out = ConvergenceHistory.from_grid(n_log, grid)

    # The log records only overall inlet/outlet flows, so map them to the first
    # and last through-flow stations; interior stations (n_row == 2) stay NaN.
    n_row = len(grid.rows)
    n_st = 2 * n_row
    st_in, st_out = 0, n_st - 1
    # mdot is non-dimensionalised by the fluid mass-flux scale, matching the
    # live grid path (Grid._station_stats); ho/s from get_h/get_s on nd
    # inputs are already non-dimensional.
    mdot_ref = fluid.rhoV_ref * grid[0].L_ref ** 2

    for i, blk in enumerate(step_blocks):
        mdot = np.full(n_st, np.nan)
        ho = np.full(n_st, np.nan)
        s = np.full(n_st, np.nan)
        # Convert dimensional Po, To → non-dimensional ho, s at each endpoint.
        for side, st in (("in", st_in), ("out", st_out)):
            rho_nd, u_nd = fluid.set_P_T(
                blk[f"Po_{side}"] / fluid.P_ref, blk[f"To_{side}"] / fluid.T_ref
            )
            ho[st] = fluid.get_h(rho_nd, u_nd)
            s[st] = fluid.get_s(rho_nd, u_nd)
            mdot[st] = blk[f"mdot_{side}"] / mdot_ref

        # DAVG → drho; the other four residuals and all throttle state are
        # absent from the log, so leave them NaN.
        conv = ConvergenceStep(
            residual=np.array([blk["davg"], np.nan, np.nan, np.nan, np.nan]),
            mdot=mdot,
            ho=ho,
            s=s,
            mdot_target=np.nan,
            mdot_throttle=np.nan,
            P_throttle=np.nan,
            dP_P=np.nan,
            dP_I=np.nan,
            dP_D=np.nan,
        )
        out.record_convergence(blk["i_step"], conv, time=i * mean_dt)

    return out
