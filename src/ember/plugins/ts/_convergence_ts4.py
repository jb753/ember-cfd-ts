"""Reconstruct an ember ConvergenceHistory from a Turbostream 4 text log.

Follows the same shape as :func:`~ember.plugins.ts._convergence.parse_ts3_log`:
the history is sized and seeded with
:meth:`~ember.convergence_history.ConvergenceHistory.from_grid` and filled a
step at a time through
:meth:`~ember.convergence_history.ConvergenceHistory.record_convergence`, with
the grid supplying the fluid and reference scales that the dimensional,
topology-free log never records. Only the log format differs: TS4 emits one
``ROVX DELTA:`` residual line and one ``INLET:``/``OUTLET:`` pair per step,
where TS3 writes a multi-line step block.
"""

import re

import numpy as np

from ember.convergence_history import ConvergenceHistory
from ember.grid import ConvergenceStep

# TS4 logs one residual and one line per through-flow boundary per step. The
# INLET/OUTLET lines carry mass flow, stagnation enthalpy and stagnation
# pressure, in that order. Each step block also logs its own solve time
# ("TOTAL TIME: <seconds>"), immediately before that step's ROVX DELTA line
# -- the cumulative sum of these is the real elapsed wall-clock time, unlike
# perf_counter() sampled while parsing an already-finished log.
_re_resid = re.compile(r"^ROVX DELTA:\s+(\S+)", re.MULTILINE)
_re_inlet = re.compile(r"^INLET:\s+(\S+)\s+(\S+)\s+(\S+)", re.MULTILINE)
_re_outlet = re.compile(r"^OUTLET:\s+(\S+)\s+(\S+)\s+(\S+)", re.MULTILINE)
_re_step_time = re.compile(r"^TOTAL TIME:\s+(\S+)", re.MULTILINE)


def parse_ts4_log(filename, grid):
    """Reconstruct a ConvergenceHistory from a TS4 text log file.

    Parameters
    ----------
    filename : str
        Path to the TS4 log (e.g. ``log_ts4.txt``).
    grid : ember.grid.Grid
        The grid that was solved, supplying the fluid and the reference scales
        needed to non-dimensionalise the log's dimensional values, plus the
        blade counts that convert per-passage mass flow to the whole annulus.

    Returns
    -------
    ember.convergence_history.ConvergenceHistory

    Raises
    ------
    ValueError
        If the log's per-step records are inconsistent with each other. Unlike
        turbigen's parser this does not silently truncate or pad: a log that
        does not parse is surfaced, since the mass flows are scaled by blade
        count and a mismatched grid/log pair would corrupt them quietly.
    """
    with open(filename, "r") as f:
        text = f.read()

    resid = [float(v) for v in _re_resid.findall(text)]
    inlet = [[float(v) for v in m] for m in _re_inlet.findall(text)]
    outlet = [[float(v) for v in m] for m in _re_outlet.findall(text)]
    step_time = [float(v) for v in _re_step_time.findall(text)]

    # A step is complete once its OUTLET line has been written, so that count
    # sets the history length and the other records are truncated to match a
    # run that was killed mid-step. Any record *shorter* than the outlet count
    # means the log is internally inconsistent, not merely incomplete.
    n_log = len(outlet)
    if n_log == 0:
        raise ValueError(f"No completed steps found in TS4 log {filename}")
    if len(inlet) < n_log or len(resid) < n_log or len(step_time) < n_log:
        raise ValueError(
            f"TS4 log {filename} has {n_log} OUTLET records but "
            f"{len(inlet)} INLET, {len(resid)} residual and "
            f"{len(step_time)} TOTAL TIME records; the log is internally "
            "inconsistent."
        )

    # Cumulative sum of each step's own TOTAL TIME line is the real elapsed
    # wall-clock time; record_convergence's default (perf_counter() sampled
    # while parsing an already-finished log) would otherwise report the
    # parsing loop's own near-instant runtime instead of the solve's.
    cum_time = np.cumsum(step_time[:n_log])

    out = ConvergenceHistory.from_grid(n_log, grid)

    inlet_patch = grid.patches.inlet[0]
    outlet_patch = grid.patches.outlet[0]
    fluid = outlet_patch.block_view.fluid

    # TS4 reports the mass flow through a single passage; the history records
    # the whole annulus, as the live grid path does.
    Nb_in = float(np.mean(inlet_patch.block.Nb))
    Nb_out = float(np.mean(outlet_patch.block.Nb))

    n_row = len(grid.rows)
    n_st = 2 * n_row
    st_in, st_out = 0, n_st - 1
    mdot_ref = fluid.rhoV_ref * grid[0].L_ref**2

    for i in range(n_log):
        mdot = np.full(n_st, np.nan)
        ho = np.full(n_st, np.nan)
        s = np.full(n_st, np.nan)

        for (mdot_i, ho_i, Po_i), Nb, st in (
            (inlet[i], Nb_in, st_in),
            (outlet[i], Nb_out, st_out),
        ):
            rho_nd, u_nd = fluid.set_P_h(Po_i / fluid.P_ref, ho_i / fluid.u_ref)
            ho[st] = fluid.get_h(rho_nd, u_nd)
            s[st] = fluid.get_s(rho_nd, u_nd)
            mdot[st] = mdot_i * Nb / mdot_ref

        # TS4 logs a single density-flux residual; the other four components
        # and all throttle state are absent from the log.
        conv = ConvergenceStep(
            residual=np.array([resid[i], np.nan, np.nan, np.nan, np.nan]),
            mdot=mdot,
            ho=ho,
            s=s,
            mdot_target=np.nan,
            mdot_throttle=np.nan,
            dP_throttle=np.nan,
            dP_P=np.nan,
            dP_I=np.nan,
            dP_D=np.nan,
        )
        out.record_convergence(i, conv, time=float(cum_time[i]))

    return out
