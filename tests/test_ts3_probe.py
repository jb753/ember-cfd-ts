"""Tests for TS3 probe file I/O: metadata sidecar write/read.

Covers the ``probe_meta.yaml`` sidecar emitted alongside a TS3 file and the
loaders that consume it (``ember.plugins.ts``). The passive-overlay behaviour of
``ProbePatch`` itself is a core concern and is tested in core ember's
``test_probe.py``.
"""

import os

import h5py
import numpy as np
import pytest

from ember.block import Block
from ember.fluid import PerfectFluid
from ember.grid import Grid
from ember.patch import ProbePatch
import ember.plugins.ts as ts3
from ember.plugins.ts import _io
import ember.util_yaml as util_yaml


def _make_block(shape=(6, 5, 4), rpm=3600.0, Nb=24):
    """Single block with coordinates, fluid, and flow data set."""
    block = Block(shape=shape)
    xv, rv, tv = np.meshgrid(
        np.linspace(0.0, 1.0, shape[0]),
        np.linspace(0.8, 1.2, shape[1]),
        np.linspace(0.0, np.pi / 8, shape[2]),
        indexing="ij",
    )
    block.set_x(xv)
    block.set_r(rv)
    block.set_t(tv)
    block.set_fluid(PerfectFluid(cp=1005.0, gamma=1.4, mu=1.8e-5, Pr=0.7))
    block.set_rpm(rpm)
    block.set_Nb(Nb)

    rho = np.ones(shape, dtype=np.float32) * 1.2
    conserved = np.stack(
        [rho, 150.0 * rho, 0.0 * rho, 100.0 * rho, np.ones(shape) * 2.5e5], axis=-1
    )
    block.set_conserved(conserved)
    block.set_wdist(np.ones(shape, dtype=np.float32) * 1e-3)
    return block


# --- write side -----------------------------------------------------------


def test_write_emits_probe_metadata(tmp_path):
    block = _make_block()
    block.patches.append(ProbePatch(i=2, j=3, k=1, label="probe_mid"))
    grid = Grid([block])

    fname = tmp_path / "input.hdf5"
    ts3.write_ts3(grid, str(fname), strict=False)

    meta_path = tmp_path / "probe_meta.yaml"
    assert meta_path.exists()

    meta = util_yaml.read_yaml(str(meta_path))
    entry = meta[0][0]
    assert entry["shape"] == [1, 1, 1]
    assert entry["Nb"] == 24
    assert entry["label"] == "probe_mid"
    assert entry["Omega"] == pytest.approx(float(block.Omega), rel=1e-5)
    # Self-describing gas properties + sampling frequency
    assert entry["cp"] == pytest.approx(1005.0, rel=1e-5)
    assert entry["ga"] == pytest.approx(1.4, rel=1e-5)
    assert entry["mu"] == pytest.approx(1.8e-5, rel=1e-5)
    # nstep_save_probe defaults to 0, so fs is guarded to 0.0
    assert entry["fs"] == 0.0
    # Raw sampling vars for on-read time-dimension validation (all 0 at defaults)
    assert entry["ncycle"] == 0
    assert entry["nstep_cycle"] == 0
    assert entry["nstep_save_probe"] == 0
    assert entry["nstep_save_start_probe"] == 0


def test_write_sets_probe_kind_and_pv(tmp_path):
    block = _make_block()
    block.patches.append(ProbePatch(i=2, j=3, k=1, label="probe_mid"))
    grid = Grid([block])

    fname = tmp_path / "input.hdf5"
    ts3.write_ts3(grid, str(fname), strict=False)

    with h5py.File(fname, "r") as f:
        patch_group = f["block0/patch0"]
        assert int(patch_group.attrs["kind"]) == 8
        assert int(patch_group["probe_append_pv"][0]) == 1


def test_probe_round_trips_through_read_ts3(tmp_path):
    block = _make_block()
    block.patches.append(ProbePatch(i=2, j=3, k=1, label="probe_mid"))
    grid = Grid([block])

    fname = tmp_path / "input.hdf5"
    ts3.write_ts3(grid, str(fname), strict=False)

    grid2 = ts3.read_ts3(str(fname))
    assert len(grid2[0].patches.probe) == 1


def test_write_without_probes_makes_no_sidecar(tmp_path):
    grid = Grid([_make_block()])
    fname = tmp_path / "input.hdf5"
    ts3.write_ts3(grid, str(fname), strict=False)
    assert not (tmp_path / "probe_meta.yaml").exists()


# --- read side ------------------------------------------------------------


def _write_meta(dname, meta):
    util_yaml.write_yaml(meta, os.path.join(str(dname), "probe_meta.yaml"))


def test_read_probe_metadata(tmp_path):
    meta = {0: {0: {"shape": [2, 3], "Omega": 0.0, "Nb": 1, "label": "p"}}}
    _write_meta(tmp_path, meta)
    dat = tmp_path / "output_probe_p_0_0.dat"
    dat.touch()

    out = ts3.read_probe_metadata(tmp_path)
    assert out == {str(dat): meta[0][0]}


def test_read_probe_metadata_empty_dir(tmp_path):
    assert ts3.read_probe_metadata(tmp_path) == {}


def test_get_probe_metadata_from_yaml(tmp_path):
    meta = {0: {2: {"shape": [4, 5], "Omega": 12.5, "Nb": 7, "label": "q"}}}
    _write_meta(tmp_path, meta)
    dat = tmp_path / "output_probe_q_0_2.dat"
    dat.touch()

    shape, Omega = _io._get_probe_metadata(str(dat))
    assert shape == (4, 5, -1)
    assert Omega == pytest.approx(12.5)


def test_get_probe_metadata_missing_sidecar_raises(tmp_path):
    dat = tmp_path / "output_probe_x_0_0.dat"
    dat.touch()
    with pytest.raises(FileNotFoundError):
        _io._get_probe_metadata(str(dat))


def test_get_probe_metadata_missing_entry_raises(tmp_path):
    _write_meta(tmp_path, {0: {0: {"shape": [1], "Omega": 0.0, "Nb": 1, "label": "p"}}})
    dat = tmp_path / "output_probe_x_3_9.dat"  # bid=3, pid=9 absent
    dat.touch()
    with pytest.raises(KeyError):
        _io._get_probe_metadata(str(dat))
