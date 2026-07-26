"""Turbostream (TS3 and TS4) support for EMBER CFD.

This package plugs into the PEP 420 namespace ``ember.plugins.ts`` and provides
read/write access to the HDF5-based TS3 solver format, probe output, and
Turbostream convergence logs, plus :class:`TS3Solver` and :class:`TS4Solver` to
run either solver through the uniform :class:`ember.solver.BaseSolver`
interface. Core ``ember`` carries no Turbostream knowledge; install this plugin
(``ember-cfd-ts``) to use these entry points.

The implementation lives in private submodules: ``_io`` (grid/probe file I/O,
depends on ``Grid``/``Block``/``patch``), ``_convergence`` (log parsing, depends
on ``ConvergenceHistory``), ``_solver`` (the run driver), their ``_ts4``
counterparts, and ``_log`` (shared log housekeeping). This module re-exports the
supported API.

TS4 has no writer of its own: its input file is produced by converting a
TS3-format file through Turbostream Ltd's own proprietary tooling, which runs as
a subprocess (``_scripts/convert_ts3_to_ts4.py``) and cannot be reimplemented
here. :class:`TS4Solver` therefore depends on the TS3 writer as well.
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
from ember.plugins.ts._convergence_ts4 import parse_ts4_log
from ember.plugins.ts._solver import TS3Solver
from ember.plugins.ts._solver_ts4 import TS4Solver
from ember.plugins.ts._io_ts4 import TS4ConversionError

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
    "parse_ts4_log",
    "TS3Solver",
    "TS4Solver",
    "TS4ConversionError",
]

try:
    from ember.plugins.ts._version import __version__
except ImportError:  # pragma: no cover - bare, unbuilt source tree
    from importlib.metadata import version, PackageNotFoundError

    try:
        __version__ = version("ember-cfd-ts")
    except PackageNotFoundError:
        __version__ = "0.0.0+unknown"
