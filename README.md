# ember-cfd-ts

Turbostream 3 (TS3) file-format plugin for [EMBER CFD](https://github.com/jb753/ember).

This package installs into the PEP 420 namespace `ember.plugins.ts`, adding
read/write support for the HDF5-based TS3 solver format on top of a core `ember`
install. Core `ember` carries no TS3 knowledge; install this plugin when you need
to round-trip grids, probes, and convergence logs against Turbostream 3.

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
```

## Public API

- `read_ts3(filename) -> Grid`, `write_ts3(grid, filename, strict=False)`
- `read_conserved`, `read_mu_turb`, `TS3Writer`
- `read_probe_metadata`, `read_probe_dat`, `read_probe_dat_dir`
- `parse_ts3_log(filename, grid) -> ConvergenceHistory`
- `PATCH_BY_KIND`, `KIND_BY_PATCH`
