# ember-cfd-ts

Turbostream 3 and 4 (TS3, TS4) plugin for [EMBER CFD](https://github.com/jb753/ember).

This package installs into the PEP 420 namespace `ember.plugins.ts`, adding
read/write support for the HDF5-based TS3 solver format, probe output, and
convergence logs, plus `TS3Solver` and `TS4Solver` to run either solver through
ember's uniform `BaseSolver` interface. Core `ember` carries no Turbostream
knowledge; install this plugin when you need it.

Turbostream itself is not included: running a solver needs a licensed
Turbostream installation. TS4 inputs are produced by converting a TS3-format
file with Turbostream Ltd's own tooling, which is run from that installation.

## Install

```bash
pip install ember-cfd        # core (required)
pip install ember-cfd-ts     # this plugin
```

## Usage

```python
import ember.plugins.ts as ts

grid = ts.read_ts3("input.hdf5")
ts.write_ts3(grid, "output.hdf5", strict=False)

# Reconstruct a convergence history from a TS3 text log:
hist = ts.parse_ts3_log("log_duct.txt", grid)

# Run TS3 on a grid in place, sourcing your Turbostream environment first:
solver = ts.TS3Solver(environment_script="/path/to/turbostream/bashrc")
solver.run(grid)
```

`TS4Solver` takes the same form; its `environment_script` must set up a TS4
installation.

## Public API

- `read_ts3(filename) -> Grid`, `write_ts3(grid, filename, strict=False)`
- `read_conserved`, `read_mu_turb`, `TS3Writer`
- `read_probe_metadata`, `read_probe_dat`, `read_probe_dat_dir`
- `parse_ts3_log(filename, grid)`, `parse_ts4_log(filename, grid) -> ConvergenceHistory`
- `TS3Solver`, `TS4Solver`, `TS4ConversionError`
- `PATCH_BY_KIND`, `KIND_BY_PATCH`

## Licence

GPL-3.0-or-later; see [LICENSE](LICENSE).
