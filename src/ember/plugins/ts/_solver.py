import logging
from dataclasses import dataclass, fields
from timeit import default_timer as timer
from glob import glob
import subprocess
import os
from pathlib import Path
import signal
import sys
import re

from ember.grid import DivergenceError
from ember.solver import BaseSolver
from ember.plugins.ts._io import TS3Writer, read_conserved, read_mu_turb
from ember.plugins.ts._convergence import parse_ts3_log
from ember.plugins.ts._log import archive_log

logger = logging.getLogger("ember")

# TS3 application variables that TS3Solver exposes as a typed field of the same
# name. This is a deliberately hand-maintained *mirror* of a subset of the
# dataclass fields below: every entry must be (a) an actual TS3Solver field, so
# `getattr(cfg, name)` in _write_input resolves, and (b) a name the writer
# accepts as an application variable. It duplicates those field names on
# purpose -- keeping the forwarding surface greppable in one place rather than
# deriving it from `fields(TS3Solver)` via metadata -- so the standing cost is
# that adding a typed av field means adding its name here too (and a typo here
# would surface as a getattr/writer error). Consumed by _write_input (to forward
# the values via writer.set_av) and by _check_overlap (to reject raw av
# overrides that collide with a non-default typed field). _BV_FIELDS is the same
# idea for block variables.
_AV_FIELDS = frozenset(
    {
        "cfl",
        "dampin",
        "facsecin",
        "ilos",
        "nchange",
        "viscosity_law",
        "nstep",
        "rfmix",
        "sfin",
    }
)
# Block variables with a typed field (see the _AV_FIELDS note above); currently
# just the multigrid factor. Forwarded via writer.set_bv per block.
_BV_FIELDS = frozenset({"fmgrid"})


@dataclass
class TS3Solver(BaseSolver):
    """Run Turbostream 3 through the uniform :class:`ember.solver.BaseSolver` interface.

    Turbostream 3 is a multi-block structured, GPU-accelerated Reynolds-averaged
    Navier--Stokes code developed by Brandvik & Pullan (2011). Construct with the
    configuration below and call :meth:`run` to write the TS3 input, execute the
    solver, and read the solution back into the grid in place.
    """

    workdir: Path = None
    """Working directory to run the simulation in. Must be set before ``run``."""

    environment_script: Path = Path(
        "/usr/local/software/turbostream/ts3610_a100/bashrc_module_ts3610_a100"
    )
    """Setup environment shell script to be sourced before running."""

    cfl: float = 0.4
    """Courant--Friedrichs--Lewy number, reduce for more stability."""

    dampin: float = 25.0
    """Negative feedback factor, reduce for more stability."""

    facsecin: float = 0.005
    """Fourth-order smoothing factor, increase for more stability."""

    fmgrid: float = 0.2
    """Multigrid factor, reduce for more stability."""

    ilos: int = 2
    """Viscous model, 0 for inviscid, 1 for mixing-length, 2 for Spalart-Allmaras."""

    nchange: int = 2000
    """At start of simulation, ramp smoothing and damping over this many time steps."""

    viscosity_law: int = 0
    """Variation of viscosity with temperature, 0 for constant, 1 for 0.62 power law."""

    nstep: int = 10000
    """Number of time steps."""

    nstep_avg: int = 5000
    """Average over the last `nstep_avg` steps of the calculation."""

    rfmix: float = 0.0
    """Mixing plane relaxation factor."""

    sfin: float = 0.5
    """Proportion of second-order smoothing, increase for more stability."""

    rfin: float = 0.2
    """Inlet relaxation factor, reduce for low-Mach flows."""

    av: dict = None
    """Raw application-variable overrides, ``av[name] = value``, for any TS3
    variable without a typed field. Applied after the typed fields; setting a
    name that is also driven by a non-default typed field is an error."""

    bv: dict = None
    """Raw block-variable overrides, ``bv[bid][name] = value``, for any TS3
    variable without a typed field. Applied after the typed fields; setting a
    name that is also driven by a non-default typed field is an error."""

    def __post_init__(self):
        if isinstance(self.workdir, str):
            self.workdir = Path(self.workdir)
        if isinstance(self.environment_script, str):
            self.environment_script = Path(self.environment_script)

    def run(self, grid):
        """Write, run, and read TS3 for ``grid``, updating it in place.

        Advances ``grid`` in place with the solved flow field and returns the
        convergence history parsed from the TS3 log (``None`` if the log could
        not be parsed). ``workdir`` must be set on the config.

        Raises
        ------
        DivergenceError
            If the solver diverged (NaN in the log, or a non-physical solution
            on read-back).
        """
        if self.workdir is None:
            raise ValueError("TS3Solver.workdir must be set before run().")
        if not self.workdir.exists():
            self.workdir.mkdir(parents=True, exist_ok=True)

        # Archive any existing log so every run in this workdir keeps its own
        # (e.g. one per full-multigrid level), rather than clobbering it.
        archive_log(self.workdir)
        log_path = os.path.join(self.workdir, "log.txt")

        _run(grid, self)

        # Build the convergence history from the log. The reference scales
        # (V_ref, T_ref, areas) are derived from the solved grid, which the log
        # itself does not record.
        try:
            return parse_ts3_log(log_path, grid)
        except Exception as e:
            logger.warning(f"Failed to parse log file {log_path}")
            logger.warning(f"Exception: {e}")
            return None


# How long to wait for the solver to create its log file before giving up, and
# how often to poll the running solver for divergence, in seconds.
_LOG_TIMEOUT_S = 60
_POLL_INTERVAL_S = 10


def _execute(ts3_config):
    """Using a given configuration, execute TS3."""

    # Store old working directory and change to this config's. The finally below
    # restores it on every exit path (timeout, divergence, non-zero return code),
    # so a failed run does not leave the process stranded in the work directory.
    old_workdir = os.getcwd()
    os.chdir(ts3_config.workdir)

    try:
        if not os.path.exists(ts3_config.environment_script):
            raise Exception(
                f"""Could not locate the TS3 environment script:
    {ts3_config.environment_script}
Set `environment_script` to the shell script that sets up the Turbostream
runtime environment. On a cluster, check that you are on a node with the
solver installed and that you have access to the Turbostream installation.
"""
            )

        # Open a subshell, source the environment and run the solver (serial, 1 GPU)
        logger.info("Using 1 GPU.")
        cmd_str = (
            f". {ts3_config.environment_script};"
            f" mpirun -npernode 1 -np 1 turbostream"
            f" input.hdf5 output 1 > log.txt"
        )

        # Remove old probe data
        probe_dat = glob("output_probe_*.dat")
        for fname in probe_dat:
            os.remove(fname)

        # Start the Turbostream process
        with subprocess.Popen(
            cmd_str, shell=True, stderr=subprocess.PIPE, preexec_fn=os.setsid
        ) as proc:
            # Wait for the log to appear, then watch it for divergence until the
            # solver exits. proc.wait(timeout=...) returns the instant the solver
            # finishes, so a fast run is not held up by the polling interval, and
            # the divergence check happens before each wait rather than after a
            # blind sleep.
            try:
                start = timer()
                while proc.poll() is None:
                    have_log = os.path.isfile("log.txt")
                    if not have_log and (timer() - start) > _LOG_TIMEOUT_S:
                        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                        raise Exception(
                            f"Timed out after {_LOG_TIMEOUT_S}s waiting for TS3 "
                            "log file to appear"
                        )
                    if have_log and (istep_nan := _check_nan("log.txt")):
                        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                        raise DivergenceError(
                            f"TS3 diverged at step {istep_nan}"
                        ) from None
                    try:
                        proc.wait(timeout=_POLL_INTERVAL_S)
                    except subprocess.TimeoutExpired:
                        pass
            except KeyboardInterrupt:
                logger.warning("******")
                logger.warning("Caught interrupt, killing solver...")
                logger.warning("******")
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                proc.wait()
                logger.warning("Killed solver.")

            proc.wait()

            # If we have an error code, print debugging info
            if proc.returncode:
                raise Exception(
                    f"""TS3 failed, exit code {proc.returncode}
COMMAND: {cmd_str}
STDERR: {proc.stderr.read().decode(sys.getfilesystemencoding()).strip()}

Check that Turbostream is available and runnable on this node."""
                ) from None

        # Delete extraneous files
        for f in ("stopit", "output_avg.xdmf", "output.xdmf", "input.hdf5"):
            try:
                os.remove(f)
            except FileNotFoundError:
                pass

        # Remove empty hdf5 probes (we don't use them)
        probe_hdf5 = glob("output_probe_*.hdf5")
        for fname in probe_hdf5:
            os.remove(fname)
    finally:
        os.chdir(old_workdir)


def _read_hdf5(grid, ts3_config):
    """Using a given configuration, load flow solution and insert into grid.

    The averaged flow field is read from ``output_avg.hdf5`` and the turbulent
    viscosity from the instantaneous ``output.hdf5`` (Turbostream only writes
    ``trans_dyn_vis`` to the latter). A diverged solution surfaces as a
    :class:`~ember.grid.DivergenceError`.
    """

    output_file_path = os.path.join(ts3_config.workdir, "output_avg.hdf5")
    output_inst_file_path = os.path.join(ts3_config.workdir, "output.hdf5")
    if not os.path.exists(output_file_path):
        raise Exception(f"""No Turbostream output file found at: {output_file_path}""")

    # The plugin's _io owns the datum-correct read into the existing grid; a
    # diverged solution (negative/NaN density, etc.) is raised as ValueError
    # there and translated to DivergenceError.
    try:
        read_conserved(grid, output_file_path)
        read_mu_turb(grid, output_inst_file_path)
    except ValueError as e:
        raise DivergenceError(f"TS3 solution diverged: {e}") from e


def _non_default_fields(ts3_config, names):
    """Return the subset of `names` whose config value differs from the default.

    Used to decide which typed fields actively drive a TS3 variable, so the
    raw av/bv override dicts can be rejected when they would compete with one.
    """
    defaults = {f.name: f.default for f in fields(ts3_config)}
    return {name for name in names if getattr(ts3_config, name) != defaults[name]}


def _check_overlap(ts3_config):
    """Reject raw av/bv overrides that collide with a non-default typed field.

    A TS3 variable must have a single source. The typed fields and the raw
    override dicts can each set the same name; forbid that when the typed field
    has been moved off its default (so e.g. `robust()` setting cfl and an `av`
    dict also setting cfl will correctly error).
    """
    av = ts3_config.av or {}
    bv = ts3_config.bv or {}

    av_clash = _non_default_fields(ts3_config, _AV_FIELDS) & av.keys()
    if av_clash:
        raise ValueError(
            f"av override(s) {sorted(av_clash)} also set by a non-default typed "
            "field; remove one of the two competing sources."
        )

    bv_fields = _non_default_fields(ts3_config, _BV_FIELDS)
    for bid, block_bv in bv.items():
        bv_clash = bv_fields & block_bv.keys()
        if bv_clash:
            raise ValueError(
                f"bv override(s) {sorted(bv_clash)} for block {bid} also set by a "
                "non-default typed field; remove one of the two competing sources."
            )


def _write_input(grid, ts3_config):
    """Write the TS3 input file from an ember grid + config via the writer.

    Drives :class:`~ember.plugins.ts.TS3Writer` so the solver's typed fields and
    raw av/bv override dicts layer on top of the grid-derived defaults. Mirrors
    ``ember.plugins.ts.write_ts3`` but with the extra config surface.

    The grid must already carry ``wdist`` (it holds the baked-in mixing-length
    limit); ``get_blocks`` reads it and also writes total energy on
    Turbostream's zero datum internally, so no datum handling is needed here.
    """
    # Forbid two competing sources for one variable before writing anything.
    _check_overlap(ts3_config)

    # strict=True: the contract is to hand TS3 a complete, runnable grid, so a
    # missing flow field, fluid, or periodic connectivity is a bug here and must
    # fail at write time rather than silently omitting variables.
    writer = TS3Writer()
    writer.get(grid, strict=True)

    # Typed application variables (forwarded by matching name).
    writer.set_av(**{name: getattr(ts3_config, name) for name in _AV_FIELDS})

    # Derived application variables.
    writer.set_av(
        nstep_save_start=ts3_config.nstep - ts3_config.nstep_avg,
        restart=1,
    )

    # Typed block variables, per block. xllim is left at the writer's
    # non-clamping default since the limit is already baked into wdist.
    for bid in range(len(grid)):
        writer.set_bv(bid, fmgrid=ts3_config.fmgrid)

    # Inlet relaxation factor, applied to each inlet patch as a patch variable.
    # Iterate the writer's own pv (keyed by its renumbered pid, rotating patches
    # excluded); inlets are the patches carrying an rfin variable.
    for bid, block_pv in enumerate(writer.pv):
        for pid, pv in block_pv.items():
            if "rfin" in pv:
                writer.set_pv(bid, pid, rfin=ts3_config.rfin)

    # Raw overrides last (validated for overlap above). set_av/set_bv validate
    # unknown names, type-cast, and reject NaN.
    writer.set_av(**(ts3_config.av or {}))
    for bid, block_bv in (ts3_config.bv or {}).items():
        writer.set_bv(bid, **block_bv)

    writer.check()
    input_path = os.path.join(ts3_config.workdir, "input.hdf5")
    writer.write(input_path)
    writer.write_probe_meta(input_path)


def _run(grid, ts3_config):
    """Perform all steps on a grid and config."""

    _write_input(grid, ts3_config)
    _execute(ts3_config)
    _read_hdf5(grid, ts3_config)


# Live-divergence detection during a run reads only these two patterns;
# full log parsing for the convergence history lives in ember.plugins.ts.
re_nan = re.compile(r"NAN")
re_current_step = re.compile(r"^O?U?T?E?R? ?STEP No\.\s*(\d*)", flags=re.MULTILINE)


def _check_nan(fname):
    """Return step number of divergence from TS3 log, or zero if no NANs found.

    The whole log is read and searched, so detection does not depend on where in
    the file the NAN lands (the previous chunked, anchored scan missed NANs that
    were not at the start of a 2048-byte block or straddled a block boundary). The
    log is small relative to the polling interval, so reading it whole is cheap.
    When a NAN is present the most recently logged step number is returned, or -1
    if no step has been logged yet.
    """
    with open(fname, "r") as f:
        content = f.read()
    if not re_nan.search(content):
        return 0
    steps = re_current_step.findall(content)
    return int(steps[-1]) if steps else -1
