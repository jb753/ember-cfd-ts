"""Turbostream 3 (TS3) file-format support for EMBER CFD.

This package plugs into the PEP 420 namespace ``ember.plugins.ts`` and provides
read/write access to the HDF5-based TS3 solver format, probe output, and TS3
convergence logs, plus :class:`TS3Solver` to run Turbostream 3 through the
uniform :class:`ember.solver.BaseSolver` interface. Core ``ember`` carries no
TS3 knowledge; install this plugin (``ember-cfd-ts``) to use these entry points.

The implementation lives in the private submodules ``_io`` (grid/probe file
I/O, depends on ``Grid``/``Block``/``patch``), ``_convergence`` (log parsing,
depends on ``ConvergenceHistory``), and ``_solver`` (the run driver); this
module re-exports the supported API.
"""

from ember.plugins.ts._io import (
    read_ts3,
    write_ts3,
    read_conserved,
    read_mu_turb,
    TS3Writer,
    read_probe_metadata,
    read_probe_dat,
    read_probe_dat_dir,
    PATCH_BY_KIND,
    KIND_BY_PATCH,
)
from ember.plugins.ts._convergence import parse_ts3_log
from ember.plugins.ts._solver import TS3Solver

__all__ = [
    "read_ts3",
    "write_ts3",
    "read_conserved",
    "read_mu_turb",
    "TS3Writer",
    "read_probe_metadata",
    "read_probe_dat",
    "read_probe_dat_dir",
    "PATCH_BY_KIND",
    "KIND_BY_PATCH",
    "parse_ts3_log",
    "TS3Solver",
]

try:
    from ember.plugins.ts._version import __version__
except ImportError:  # pragma: no cover - bare, unbuilt source tree
    from importlib.metadata import version, PackageNotFoundError

    try:
        __version__ = version("ember-cfd-ts")
    except PackageNotFoundError:
        __version__ = "0.0.0+unknown"
