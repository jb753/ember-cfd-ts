"""Run Turbostream 4 through ember's uniform solver interface.

The driver differs from the TS3 one in two structural ways. First, the input
file is not written directly: a TS3-format file is written and handed to the
vendor's converter as a subprocess, then the fields the conversion cannot
supply (wall distance, throttle, probes) are added to its output. Second, there
is no live log polling -- TS4 checks for NaN in-process and aborts itself, so
the log is scanned once after the run to turn that abort into a
:class:`~ember.grid.DivergenceError`.
"""

import logging
import os
import re
import signal
import subprocess
import sys
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np

from ember.grid import DivergenceError
from ember.solver import BaseSolver
from ember.plugins.ts._convergence_ts4 import parse_ts4_log
from ember.plugins.ts._log import archive_log
from ember.plugins.ts import _io_ts4

logger = logging.getLogger("ember")

LOG_NAME = "log_ts4.txt"

# TS4 reports its own divergence check, then stops. The step number is
# recovered from the most recent RK loop banner before the failure.
_re_nan = re.compile(r"^detected NAN$", re.MULTILINE)
_re_current_step = re.compile(r"^RK LOOP NO:\s*(\d*)$", re.MULTILINE)

# Filename of the vendored TS3 to TS4 conversion script, resolved next to this
# module and invoked through the sourced Turbostream environment.
_CONVERT_SCRIPT = os.path.join(
    os.path.dirname(__file__), "_scripts", "convert_ts3_to_ts4.py"
)


@dataclass(frozen=True)
class TS4Solver(BaseSolver):
    """Run Turbostream 4 through the uniform :class:`ember.solver.BaseSolver` interface.

    Turbostream 4 is an unstructured, GPU-accelerated Reynolds-averaged
    Navier--Stokes code developed by Turbostream Ltd. Construct with the
    configuration below and call :meth:`run` to write the input, execute the
    solver, and read the solution back into the grid in place.

    The grid must already carry a valid initial state (flow field, wall
    distance, fluid); this solver writes it out and runs, it does not generate
    an initial guess.
    """

    environment_script: str = (
        "/usr/local/software/turbostream/ts42111/bashrc_module_ts42111_a100"
    )
    """Setup environment shell script to be sourced before running."""

    cfl: float = 25.0
    """Courant--Friedrichs--Lewy number setting the time step."""

    cfl_ramp_nstep: int = 500
    """Ramp the CFL number up over this many initial time steps."""

    cfl_ramp_st: float = 1.0
    """Starting value for the CFL ramp."""

    cfl_turb_fac: float = 0.5
    """CFL factor for the turbulence equations, relative to the mean flow."""

    implicit_scheme: int = 1
    """1 for implicit, 0 for explicit time marching."""

    viscous_model: int = 2
    """Turbulence model: 0 inviscid, 1 laminar, 2 Spalart-Allmaras, 3 k-omega."""

    nstep: int = 5000
    """Number of time steps for the calculation."""

    nstep_avg: int = 1000
    """Average over the last ``nstep_avg`` steps of the calculation."""

    area_avg_pout: bool = True
    """Allow a non-uniform outlet pressure, forcing its area average to target."""

    outlet_tag: str = "Outlet"
    """Name identifying the outlet boundary in the converted TS4 input file."""

    kappa2: float = 1.0
    """Second-order smoothing factor."""

    kappa4: float = 1.0 / 128.0
    """Fourth-order smoothing factor."""

    inlet_relax_fac: float = 0.5
    """Inlet boundary-condition relaxation factor."""

    mixing_bc_method: int = 2
    mixing_distribution_kind: int = 1
    mixing_method: int = 2
    mixing_rf_ramp_nstep: int = 0
    """Mixing-plane treatment and the ramp applied to its relaxation factor."""

    rot_fac_ramp_nstep: int = 0
    rot_fac_ramp_st: float = 0.1
    rot_fac_ramp_en: float = 1.0
    """Ramp applied to rotational speed over the start of the run."""

    pout_fac_ramp_nstep: int = 0
    """Ramp applied to outlet pressure over the start of the run."""

    precon: int = 0
    precon_fac_ramp_nstep: int = 100
    precon_fac_ramp_st: float = 0.1
    precon_fac_ramp_en: float = 1.0
    precon_sigma_pgr: float = 3.0
    """Low-Mach preconditioning and its ramp."""

    halo_implementation: int = 1
    """Halo exchange scheme for multi-GPU runs."""

    nstep_save_start_probe_1d: int = 0
    nstep_save_probe_1d: int = 100
    nstep_save_start_probe_2d: int = 0
    nstep_save_probe_2d: int = 100
    """Probe output cadence."""

    point_probe: list = None
    """Point probes, each ``{"xyz": ..., "domain": int, "label": str}``."""

    logical_probe: list = None
    """Logical probes, each ``{"tag": str, "label": str}``."""

    ofp: dict = field(default_factory=dict)
    """Raw ``.ofp`` overrides for any TS4 variable without a typed field.
    Applied last, after the typed fields and the grid-derived properties."""

    def soft(self):
        """Return a copy detuned for a robust start; see :meth:`BaseSolver.soft`.

        A smaller time step and more fourth-order smoothing, neither moved past
        where it already is, so a solver already configured more
        conservatively than this keeps its own value. Averaging is cut to the
        last step: a soft run's field is only ever an initial guess.
        """
        return replace(
            self,
            cfl=min(self.cfl, 5.0),
            kappa4=max(self.kappa4, 1.0 / 64.0),
            nstep_avg=1,
        )

    def run(self, grid, workdir=None):
        """Write, run, and read TS4 for ``grid`` in ``workdir``, updating it in
        place.

        Advances ``grid`` in place with the solved flow field and returns the
        convergence history parsed from the TS4 log (``None`` if the log could
        not be parsed). ``workdir`` is the directory this run writes its input,
        output and log in, made if it does not exist; it must be given.

        Raises
        ------
        DivergenceError
            If the solver diverged (NaN reported in the log, or a non-physical
            solution on read-back).
        """
        if workdir is None:
            raise ValueError("TS4Solver.run() needs a workdir to run in.")
        workdir = Path(workdir)
        workdir.mkdir(parents=True, exist_ok=True)

        # Archive any existing log so successive runs in one workdir each keep
        # their own, rather than clobbering it.
        archive_log(workdir, LOG_NAME)
        log_path = os.path.join(workdir, LOG_NAME)

        _write_input(grid, self, workdir)
        _convert(self, workdir)
        _write_converted_input(grid, self, workdir)
        _execute(self, workdir)
        _read_output(grid, self, workdir)

        try:
            return parse_ts4_log(log_path, grid)
        except Exception as e:
            logger.warning(f"Failed to parse log file {log_path}")
            logger.warning(f"Exception: {e}")
            return None


def _write_input(grid, config, workdir):
    """Write the TS3-format input and the ``.ofp`` configuration."""
    _io_ts4.write_ts3_input(grid, os.path.join(workdir, "input.hdf5"))

    conf = _io_ts4.config_from_grid(config, grid)
    conf.update(config.ofp or {})
    _io_ts4.write_ofp(os.path.join(workdir, "config.ofp"), conf)


def _write_converted_input(grid, config, workdir):
    """Add the fields the conversion cannot supply to the TS4 input file."""
    input_path = os.path.join(workdir, "input_ts4.hdf5")

    _io_ts4.write_throttle(config, grid, input_path, workdir)
    _io_ts4.write_wall_distance(grid, input_path)

    _io_ts4.write_probes_ofp(workdir)
    for probe in config.point_probe or []:
        xyz = np.asarray(probe["xyz"]).T
        if xyz.ndim != 2 or xyz.shape[0] != 3:
            raise ValueError(
                f"Point probe {probe.get('label')!r} needs xyz of shape (n, 3)"
            )
        _io_ts4.write_point_probe(
            config, workdir, xyz, int(probe["domain"]), probe["label"]
        )
    for probe in config.logical_probe or []:
        _io_ts4.write_logical_probe(config, workdir, **probe)
    _io_ts4.check_probes(workdir)


def _check_environment_script(config):
    if not os.path.exists(config.environment_script):
        raise Exception(
            f"""Could not locate the TS4 environment script:
    {config.environment_script}
Set `environment_script` to the shell script that sets up the Turbostream
runtime environment. On a cluster, check that you are on a node with the
solver installed and that you have access to the Turbostream installation.
"""
        )


def _run_in_workdir(config, workdir, cmd_str, what):
    """Run a command in the working directory with the environment sourced."""
    old_workdir = os.getcwd()
    os.chdir(workdir)
    try:
        proc = subprocess.run(
            cmd_str, shell=True, capture_output=True, check=False
        )
    finally:
        os.chdir(old_workdir)

    if proc.returncode:
        stdout = proc.stdout.decode(sys.getfilesystemencoding()).strip()
        stderr = proc.stderr.decode(sys.getfilesystemencoding()).strip()
        raise Exception(
            f"""{what} failed, exit code {proc.returncode}
COMMAND: {cmd_str}
STDOUT: {stdout}
STDERR: {stderr}
"""
        )


def _convert(config, workdir):
    """Convert the TS3-format input to a TS4 input file, as a subprocess.

    The conversion wraps Turbostream Ltd's own ``ts.process``/VTK stack, so it
    has to run inside the sourced Turbostream environment rather than in this
    interpreter.
    """
    _check_environment_script(config)
    logger.info("Converting TS3 to TS4...")
    _run_in_workdir(
        config,
        workdir,
        f". {config.environment_script} 2> /dev/null;"
        # Invoked via "python3 <script>" rather than executing the script
        # directly: a non-editable install (uv pip install, no -e) does not
        # preserve the source tree's executable bit on installed data files,
        # so mpirun -np 1 {_CONVERT_SCRIPT} would fail with "could not
        # access or execute" once ember-cfd-ts is installed that way. Use
        # whatever "python3" the sourced Turbostream environment provides
        # (its own ts.process/VTK stack), not this interpreter's own
        # sys.executable.
        f" mpirun -np 1 python3 {_CONVERT_SCRIPT} input.hdf5 input_ts4"
        f" > convert.log",
        "TS3 to TS4 conversion",
    )

    input_path = os.path.join(workdir, "input_ts4.hdf5")
    if not os.path.exists(input_path):
        raise _io_ts4.TS4ConversionError(
            f"Conversion reported success but produced no {input_path}; see "
            f"{os.path.join(workdir, 'convert.log')}."
        )


def _execute(config, workdir):
    """Run TS4 itself, as a subprocess.

    Unlike TS3, the log is not polled during the run: TS4 detects NaN
    in-process and stops, so a diverged run ends on its own and is diagnosed
    from the log afterwards.
    """
    _check_environment_script(config)
    logger.info("Using 1 GPU.")
    logger.info("Running TS4...")

    old_workdir = os.getcwd()
    os.chdir(workdir)
    try:
        cmd_str = (
            f". {config.environment_script} 2> /dev/null;"
            " mpirun -np 1 python $TSHOME/$TSDIR/bin/turbostream.py"
            " config.ofp input_ts4.hdf5 input_ts4.hdf5 input_ts4.hdf5 output_ts4"
            f" 1 --fname_out_procid=procids.hdf5 > {LOG_NAME} 2> err.txt"
        )
        with subprocess.Popen(
            cmd_str, shell=True, stderr=subprocess.PIPE, preexec_fn=os.setsid
        ) as proc:
            try:
                proc.wait()
            except KeyboardInterrupt:
                logger.warning("******")
                logger.warning("Caught interrupt, killing solver...")
                logger.warning("******")
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                proc.wait()
                logger.warning("Killed solver.")
                raise

            # Divergence first: TS4 stops itself on NaN, which may or may not
            # also be a non-zero exit, and "diverged" is the more useful
            # diagnosis of the two.
            if os.path.isfile(LOG_NAME) and (istep_nan := _check_nan(LOG_NAME)):
                raise DivergenceError(f"TS4 diverged at step {istep_nan}")

            if proc.returncode:
                stderr = proc.stderr.read().decode(sys.getfilesystemencoding())
                raise Exception(
                    f"""TS4 failed, exit code {proc.returncode}
COMMAND: {cmd_str}
STDERR: {stderr.strip()}

Check that Turbostream is available and runnable on this node."""
                )
    finally:
        os.chdir(old_workdir)


def _read_output(grid, config, workdir):
    """Load the TS4 solution back into the grid."""
    output_path = os.path.join(workdir, "output_ts4.hdf5")
    output_avg_path = os.path.join(workdir, "output_ts4_avg.hdf5")
    if not os.path.exists(output_path):
        raise Exception(f"No Turbostream 4 output file found at: {output_path}")

    try:
        _io_ts4.read_flow(grid, output_path, output_avg_path)
    except ValueError as e:
        raise DivergenceError(f"TS4 solution diverged: {e}") from e


def _check_nan(fname):
    """Return the step number TS4 reported NaN at, or zero if it did not.

    The whole log is read and searched, so detection does not depend on where
    in the file the report lands. When a NaN is present the most recently
    logged step number is returned, or -1 if no step has been logged yet.
    """
    with open(fname, "r") as f:
        content = f.read()
    if not _re_nan.search(content):
        return 0
    steps = _re_current_step.findall(content)
    return int(steps[-1]) if steps and steps[-1] else -1
