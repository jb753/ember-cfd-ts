"""Tests for TS3 grid file I/O via the ember-cfd-ts plugin.

Round-trip verification and error handling for reading/writing grids in the
Turbostream 3 HDF5 format, plus the TS3Writer patch-variable API. Moved out of
core ember's test_io.py when the TS3 machinery became a plugin.
"""

import numpy as np
import pytest
from ember.grid import Grid
from ember.block import Block
from ember.patch import InletPatch, OutletPatch, PeriodicPatch
from ember.fluid import PerfectFluid
from ember import util
import ember.plugins.ts as ts3


def _inlet_Po_To(patch):
    """Dimensional stagnation pressure and temperature an InletPatch prescribes.

    ember's non-reflecting InletPatch stores its target as nondimensional
    stagnation enthalpy/entropy (``ho_nd``, ``s_nd``); it no longer exposes a
    dimensional ``Po``/``To`` getter, so this inverts through the fluid the
    same way ``ember.block.Block.Po``/``To`` do.
    """
    fluid = patch.block.fluid
    rhoo_nd, uo_nd = fluid.set_h_s(patch.ho_nd, patch.s_nd)
    return (
        fluid.get_P(rhoo_nd, uo_nd) * fluid.P_ref,
        fluid.get_T(rhoo_nd, uo_nd) * fluid.T_ref,
    )


def _outlet_P(patch):
    """Dimensional static pressure an OutletPatch prescribes."""
    return patch.P_nd * patch.block.fluid.P_ref


def test_write_ts3_multiblock_roundtrip(tmp_path):
    """Test Grid.write_ts3 with multiblock grid and verify roundtrip."""
    # Create multiblock grid with different shaped blocks
    block1 = Block(shape=(3, 4, 5))
    block2 = Block(shape=(4, 3, 6))
    block3 = Block(shape=(2, 5, 3))

    # Set coordinates for each block
    xrt1 = util.linmesh3([0.0, 1.0], [0.5, 1.5], [0.0, 0.2], (3, 4, 5))
    xrt2 = util.linmesh3([1.0, 2.0], [1.5, 2.5], [0.1, 0.3], (4, 3, 6))
    xrt3 = util.linmesh3([2.0, 2.5], [0.8, 1.8], [0.0, 0.15], (2, 5, 3))

    block1.set_x(xrt1[..., 0])
    block1.set_r(xrt1[..., 1])
    block1.set_t(xrt1[..., 2])
    block2.set_x(xrt2[..., 0])
    block2.set_r(xrt2[..., 1])
    block2.set_t(xrt2[..., 2])
    block3.set_x(xrt3[..., 0])
    block3.set_r(xrt3[..., 1])
    block3.set_t(xrt3[..., 2])

    # Create fluid and set flow data for TS3 requirements
    fluid = PerfectFluid(cp=1005.0, gamma=1.4, mu=1.8e-5, Pr=0.7)

    for block in [block1, block2, block3]:
        block.set_fluid(fluid)
        block.set_rpm(0.0)
        block.set_Nb(1)

        # A non-trivial, physically meaningful state: P/T vary along i and all
        # three velocity components are non-zero, so the flow-field roundtrip is
        # actually exercised (not just a uniform field).
        ramp = np.linspace(0.0, 1.0, block.shape[0]).reshape(-1, 1, 1)
        P = 1.0e5 + 2.0e4 * ramp * np.ones(block.shape)
        T = 300.0 + 50.0 * ramp * np.ones(block.shape)
        block.set_P_T(P, T)
        block.set_Vxrt(
            np.stack(
                [
                    120.0 * np.ones(block.shape),
                    10.0 * np.ones(block.shape),
                    -30.0 * np.ones(block.shape),
                ],
                axis=-1,
            )
        )
        block.set_wdist(np.ones(block.shape, dtype=np.float32) * 0.01)

    # Create original grid
    original_grid = Grid([block1, block2, block3])

    # Write to file using Grid.write_ts3 method
    output_file = tmp_path / "test_multiblock.ts3"
    ts3.write_ts3(original_grid, str(output_file))

    # Verify file was created
    assert output_file.exists()

    # Read back using ts3.read_ts3
    reconstructed_grid = ts3.read_ts3(str(output_file))

    # Verify grid structure
    assert len(reconstructed_grid) == 3, "Should have 3 blocks"

    # Compare each block's coordinates
    for i, (orig_block, recon_block) in enumerate(
        zip(original_grid, reconstructed_grid)
    ):
        assert orig_block.shape == recon_block.shape, f"Block {i + 1} shape mismatch"

        # Compare coordinates (relaxed tolerance for HDF5 round-off)
        np.testing.assert_allclose(
            recon_block.xrt,
            orig_block.xrt,
            rtol=1e-5,
            atol=1e-6,
            err_msg=f"Block {i + 1} coordinates don't match after roundtrip",
        )

        # Compare the dimensional flow field. Density and the three momenta are
        # datum-independent and must round-trip directly; P and T cover the
        # energy, which is written on TS3's u = cv * T datum and converted back
        # on read. f4 storage limits accuracy to ~1e-4 relative.
        for name in ("rho", "rhoVx", "rhoVr", "rhorVt"):
            np.testing.assert_allclose(
                getattr(recon_block, name),
                getattr(orig_block, name),
                rtol=1e-4,
                atol=1e-3,
                err_msg=f"Block {i + 1} {name} doesn't match after roundtrip",
            )
        np.testing.assert_allclose(
            recon_block.P,
            orig_block.P,
            rtol=1e-4,
            err_msg=f"Block {i + 1} P doesn't match after roundtrip",
        )
        np.testing.assert_allclose(
            recon_block.T,
            orig_block.T,
            rtol=1e-4,
            err_msg=f"Block {i + 1} T doesn't match after roundtrip",
        )


def test_write_ts3_empty_grid():
    """Test that empty grid raises appropriate error."""
    grid = Grid()
    with pytest.raises(
        ValueError, match="Cannot write TS3 file: grid contains no blocks"
    ):
        ts3.write_ts3(grid, "test.ts3")


def test_write_ts3_no_fluid_geometry_only_roundtrip(tmp_path):
    """A grid with no fluid round-trips as geometry only, fabricating no fluid.

    Gas properties default to NaN in DEFAULT_AV, so write omits them; their
    absence is the "no fluid" signal on read, which then leaves the block
    fluid-less rather than inventing default gas properties.
    """
    block = Block(shape=(3, 4, 5))
    xrt = util.linmesh3([0.0, 1.0], [0.5, 1.5], [0.0, 0.2], (3, 4, 5))
    block.set_x(xrt[..., 0])
    block.set_r(xrt[..., 1])
    block.set_t(xrt[..., 2])
    block.set_rpm(0.0)
    block.set_Nb(1)
    grid = Grid([block])

    output_file = tmp_path / "no_fluid.hdf5"
    # Non-strict (default) write must succeed without a fluid.
    ts3.write_ts3(grid, str(output_file))

    recon = ts3.read_ts3(str(output_file))
    assert len(recon) == 1

    # Coordinates survive.
    np.testing.assert_allclose(recon[0].xrt, block.xrt, rtol=1e-5, atol=1e-6)

    # No fluid was fabricated: accessing fluid properties raises.
    with pytest.raises(ValueError, match="Working fluid must be set"):
        recon[0].fluid


def test_write_ts3_no_fluid_strict_raises():
    """Strict write of a fluid-less grid still raises."""
    block = Block(shape=(3, 4, 5))
    xrt = util.linmesh3([0.0, 1.0], [0.5, 1.5], [0.0, 0.2], (3, 4, 5))
    block.set_x(xrt[..., 0])
    block.set_r(xrt[..., 1])
    block.set_t(xrt[..., 2])
    block.set_rpm(0.0)
    block.set_Nb(1)
    grid = Grid([block])

    with pytest.raises(ValueError, match="Requires working fluid set in strict mode"):
        ts3.write_ts3(grid, "unused.hdf5", strict=True)


def test_ts3_basic_roundtrip(tmp_path):
    """Test basic TS3 Grid methods roundtrip without patches."""
    # Create a simple single block grid
    block = Block(shape=(3, 4, 5))

    # Set coordinates
    xrt = util.linmesh3([0.0, 1.0], [0.5, 1.5], [0.0, 0.2], (3, 4, 5))
    block.set_x(xrt[..., 0])
    block.set_r(xrt[..., 1])
    block.set_t(xrt[..., 2])

    # Set up fluid and minimal flow data
    fluid = PerfectFluid(cp=1005.0, gamma=1.4, mu=1.8e-5, Pr=0.7)
    block.set_fluid(fluid)
    block.set_rpm(0.0)
    block.set_Nb(1)

    # Set simple conserved variables
    rho = np.ones(block.shape, dtype=np.float32)
    rhoVx = np.zeros(block.shape, dtype=np.float32)
    rhoVr = np.zeros(block.shape, dtype=np.float32)
    rhoVt = np.zeros(block.shape, dtype=np.float32)
    rhoe = np.ones(block.shape, dtype=np.float32) * 1e5

    conserved = np.stack([rho, rhoVx, rhoVr, rhoVt, rhoe], axis=-1)
    block.set_conserved(conserved)
    block.set_wdist(np.ones(block.shape, dtype=np.float32) * 0.01)

    # Create grid
    original_grid = Grid([block])

    # Write TS3 file using Grid method
    ts3_file = tmp_path / "test_basic.ts3"
    ts3.write_ts3(original_grid, str(ts3_file))

    # Read back using Grid method
    reconstructed_grid = ts3.read_ts3(str(ts3_file))

    # Verify structure
    assert len(reconstructed_grid) == 1, "Should have 1 block"

    # Verify coordinates match
    orig_block = original_grid[0]
    recon_block = reconstructed_grid[0]

    assert orig_block.shape == recon_block.shape, "Block shape mismatch"
    np.testing.assert_allclose(
        recon_block.xrt,
        orig_block.xrt,
        rtol=1e-5,
        atol=1e-6,
        err_msg="Block coordinates don't match",
    )


def test_ts3_geometry_only_non_strict(tmp_path):
    """Test TS3 Grid write in non-strict mode with geometry only (no flow data)."""
    # Create a simple block with only coordinates
    block = Block(shape=(3, 4, 5))
    xrt = util.linmesh3([0.0, 1.0], [0.5, 1.5], [0.0, 0.2], (3, 4, 5))
    block.set_x(xrt[..., 0])
    block.set_r(xrt[..., 1])
    block.set_t(xrt[..., 2])

    # Set minimal required properties for TS3
    block.set_rpm(0.0)
    block.set_Nb(1)

    # Create grid without setting any flow data
    grid = Grid([block])

    # Should work in non-strict mode
    ts3_file = tmp_path / "test_geometry_only.ts3"
    ts3.write_ts3(grid, str(ts3_file), strict=False)

    # Verify file was created
    assert ts3_file.exists()


def test_ts3_geometry_only_strict_mode(tmp_path):
    """Test that strict mode rejects geometry-only grids."""
    # Create a simple block with only coordinates
    block = Block(shape=(3, 4, 5))
    xrt = util.linmesh3([0.0, 1.0], [0.5, 1.5], [0.0, 0.2], (3, 4, 5))
    block.set_x(xrt[..., 0])
    block.set_r(xrt[..., 1])
    block.set_t(xrt[..., 2])

    # Set minimal required properties for TS3
    block.set_rpm(0.0)
    block.set_Nb(1)

    # Create grid without setting any flow data
    grid = Grid([block])

    # Should fail in strict mode
    ts3_file = tmp_path / "test_strict_fail.ts3"
    with pytest.raises(ValueError):
        ts3.write_ts3(grid, str(ts3_file), strict=True)


def test_ts3_multiblock_with_patches_roundtrip(tmp_path):
    """Test TS3 round-trip with multiblock grid including inlet, outlet, and periodic patches."""
    # Create a three-block turbomachinery configuration
    # Block 1: Inlet block with inlet patch
    # Block 2: Middle block with periodic patches
    # Block 3: Outlet block with outlet patch

    # Create blocks with different shapes
    block1 = Block(shape=(5, 6, 8))  # Inlet block
    block2 = Block(shape=(6, 6, 8))  # Middle block
    block3 = Block(shape=(5, 6, 8))  # Outlet block

    # Set coordinates for each block to represent a turbomachinery passage
    # Block 1: x=0 to 0.5
    x1 = np.linspace(0.0, 0.5, 5)
    r1 = np.linspace(0.8, 1.2, 6)
    t1 = np.linspace(0.0, 2 * np.pi / 24, 8)  # whole pitch for Nb=24
    xv1, rv1, tv1 = np.meshgrid(x1, r1, t1, indexing="ij")
    xrt1 = np.stack([xv1, rv1, tv1], axis=-1)

    # Block 2: x=0.4 to 0.9 (overlapping with block 1)
    x2 = np.linspace(0.4, 0.9, 6)
    r2 = np.linspace(0.8, 1.2, 6)
    t2 = np.linspace(0.0, 2 * np.pi / 24, 8)
    xv2, rv2, tv2 = np.meshgrid(x2, r2, t2, indexing="ij")
    xrt2 = np.stack([xv2, rv2, tv2], axis=-1)

    # Block 3: x=0.8 to 1.3 (overlapping with block 2)
    x3 = np.linspace(0.8, 1.3, 5)
    r3 = np.linspace(0.8, 1.2, 6)
    t3 = np.linspace(0.0, 2 * np.pi / 24, 8)
    xv3, rv3, tv3 = np.meshgrid(x3, r3, t3, indexing="ij")
    xrt3 = np.stack([xv3, rv3, tv3], axis=-1)

    # Set coordinates
    block1.set_x(xrt1[..., 0])
    block1.set_r(xrt1[..., 1])
    block1.set_t(xrt1[..., 2])
    block2.set_x(xrt2[..., 0])
    block2.set_r(xrt2[..., 1])
    block2.set_t(xrt2[..., 2])
    block3.set_x(xrt3[..., 0])
    block3.set_r(xrt3[..., 1])
    block3.set_t(xrt3[..., 2])

    # Create fluid and set flow properties
    fluid = PerfectFluid(cp=1005.0, gamma=1.4, mu=1.8e-5, Pr=0.7)

    # Set up all blocks with fluid and flow data
    for i, block in enumerate([block1, block2, block3]):
        block.set_fluid(fluid)
        block.set_rpm(3600.0)  # Rotating at 3600 RPM
        block.set_Nb(24)  # 24 blades

        # Set realistic conserved variables for compressor flow
        rho = np.ones(block.shape, dtype=np.float32) * 1.2  # kg/m³
        rhoVx = np.ones(block.shape, dtype=np.float32) * 150.0 * rho  # Axial velocity
        rhoVr = np.zeros(block.shape, dtype=np.float32)  # Radial velocity
        rhoVt = (
            np.ones(block.shape, dtype=np.float32) * 100.0 * rho
        )  # Tangential velocity
        # Internal energy for realistic temperature
        rhoe = np.ones(block.shape, dtype=np.float32) * 2.5e5

        conserved = np.stack([rho, rhoVx, rhoVr, rhoVt, rhoe], axis=-1)
        block.set_conserved(conserved)
        block.set_wdist(np.ones(block.shape, dtype=np.float32) * 0.001)

    # Add inlet patch to block 1 (i=0 face). The pitch (k) range must cover
    # the whole blade pitch, per NonReflectingPatch._calc_hilbert, so only the
    # span (j) range is a partial slice.
    inlet_patch = InletPatch(i=0, j=(1, 5), k=(0, 7), label="inlet")
    block1.patches.append(inlet_patch)
    inlet_patch.set_Po_To(Po=101325.0, To=288.15)
    inlet_patch.set_Alpha(0.0)  # Yaw angle
    inlet_patch.set_Beta(15.0)  # Pitch angle (degrees)

    # Add outlet patch to block 3 (i=-1 face)
    outlet_patch = OutletPatch(i=-1, j=(1, 5), k=(0, 7), label="outlet")
    block3.patches.append(outlet_patch)
    outlet_patch.set_P(120000.0)  # Static pressure (Pa)

    # Add periodic patches to all blocks (k=0 and k=-1 faces)
    for i, block in enumerate([block1, block2, block3]):
        # Lower periodic boundary (k=0)
        periodic_low = PeriodicPatch(
            i=(1, block.shape[0] - 1),
            j=(1, block.shape[1] - 1),
            k=0,
            label=f"periodic_low_block{i}",
        )
        block.patches.append(periodic_low)

        # Upper periodic boundary (k=-1)
        periodic_high = PeriodicPatch(
            i=(1, block.shape[0] - 1),
            j=(1, block.shape[1] - 1),
            k=-1,
            label=f"periodic_high_block{i}",
        )
        block.patches.append(periodic_high)

    # Create original grid
    original_grid = Grid([block1, block2, block3])

    # Write to TS3 file
    output_file = tmp_path / "test_multiblock_patches.ts3"
    ts3.write_ts3(original_grid, str(output_file))

    # Verify file was created
    assert output_file.exists()

    # Read back the grid
    reconstructed_grid = ts3.read_ts3(str(output_file))

    # Verify grid structure
    assert len(reconstructed_grid) == 3, "Should have 3 blocks"

    # Compare each block's basic properties
    for i, (orig_block, recon_block) in enumerate(
        zip(original_grid, reconstructed_grid)
    ):
        assert orig_block.shape == recon_block.shape, f"Block {i} shape mismatch"
        assert orig_block.rpm == recon_block.rpm, f"Block {i} rpm mismatch"
        assert orig_block.Nb == recon_block.Nb, f"Block {i} Nb mismatch"

        # Compare coordinates (relaxed tolerance for HDF5 round-off)
        np.testing.assert_allclose(
            recon_block.xrt,
            orig_block.xrt,
            rtol=1e-5,
            atol=1e-6,
            err_msg=f"Block {i} coordinates don't match after roundtrip",
        )

    # Verify patches were preserved
    # Block 1 should have inlet patch + 2 periodic patches
    assert len(reconstructed_grid[0].patches) == 3
    inlet_patches = [
        p for p in reconstructed_grid[0].patches if isinstance(p, InletPatch)
    ]
    periodic_patches_b1 = [
        p for p in reconstructed_grid[0].patches if isinstance(p, PeriodicPatch)
    ]
    assert len(inlet_patches) == 1, "Block 1 should have 1 inlet patch"
    assert len(periodic_patches_b1) == 2, "Block 1 should have 2 periodic patches"

    # Block 2 should have 2 periodic patches
    assert len(reconstructed_grid[1].patches) == 2
    periodic_patches_b2 = [
        p for p in reconstructed_grid[1].patches if isinstance(p, PeriodicPatch)
    ]
    assert len(periodic_patches_b2) == 2, "Block 2 should have 2 periodic patches"

    # Block 3 should have outlet patch + 2 periodic patches
    assert len(reconstructed_grid[2].patches) == 3
    outlet_patches = [
        p for p in reconstructed_grid[2].patches if isinstance(p, OutletPatch)
    ]
    periodic_patches_b3 = [
        p for p in reconstructed_grid[2].patches if isinstance(p, PeriodicPatch)
    ]
    assert len(outlet_patches) == 1, "Block 3 should have 1 outlet patch"
    assert len(periodic_patches_b3) == 2, "Block 3 should have 2 periodic patches"

    # Verify inlet patch properties were preserved
    recon_inlet = inlet_patches[0]
    recon_Po, recon_To = _inlet_Po_To(recon_inlet)
    orig_Po, orig_To = _inlet_Po_To(inlet_patch)
    np.testing.assert_allclose(
        recon_Po, orig_Po, rtol=1e-5, err_msg="Inlet total pressure not preserved"
    )
    np.testing.assert_allclose(
        recon_To, orig_To, rtol=1e-5, err_msg="Inlet total temperature not preserved"
    )
    np.testing.assert_allclose(
        np.degrees(np.arcsin(recon_inlet.sinBeta)),
        np.degrees(np.arcsin(inlet_patch.sinBeta)),
        rtol=1e-5,
        err_msg="Inlet pitch angle not preserved",
    )

    # Verify outlet patch properties were preserved
    recon_outlet = outlet_patches[0]
    np.testing.assert_allclose(
        _outlet_P(recon_outlet),
        _outlet_P(outlet_patch),
        rtol=1e-5,
        err_msg="Outlet pressure not preserved",
    )

    # Verify periodic patch connectivity is maintained
    original_connectivity = original_grid.connectivity.periodic.pair()
    reconstructed_connectivity = reconstructed_grid.connectivity.periodic.pair()

    # Should have same number of connections (both directions counted)
    assert len(original_connectivity) == len(reconstructed_connectivity), (
        "Periodic connectivity not preserved"
    )


def test_ts3_write_preserves_block_variables(tmp_path):
    """Regression: writing TS3 must preserve rpm/Nb block variables.

    Previously set_bv reset self.bv[bid] to defaults on every call (it used a
    dict-style membership guard against a list), so a post-get_blocks set_bv
    discarded the rpm/nblade/fblade/fracann set earlier by get_blocks, silently
    writing a rotating block as stationary with a single blade.
    """
    block = Block(shape=(3, 4, 5))
    xrt = util.linmesh3([0.0, 1.0], [0.5, 1.5], [0.0, 0.2], (3, 4, 5))
    block.set_x(xrt[..., 0])
    block.set_r(xrt[..., 1])
    block.set_t(xrt[..., 2])

    fluid = PerfectFluid(cp=1005.0, gamma=1.4, mu=1.8e-5, Pr=0.7)
    block.set_fluid(fluid)
    block.set_rpm(3600.0)
    block.set_Nb(24)

    rho = np.ones(block.shape, dtype=np.float32)
    rhoVx = np.zeros(block.shape, dtype=np.float32)
    rhoVr = np.zeros(block.shape, dtype=np.float32)
    rhoVt = np.zeros(block.shape, dtype=np.float32)
    rhoe = np.ones(block.shape, dtype=np.float32) * 1e5
    block.set_conserved(np.stack([rho, rhoVx, rhoVr, rhoVt, rhoe], axis=-1))
    block.set_wdist(np.ones(block.shape, dtype=np.float32) * 0.01)

    original_grid = Grid([block])

    ts3_file = tmp_path / "test_bv.ts3"
    ts3.write_ts3(original_grid, str(ts3_file))

    reconstructed_grid = ts3.read_ts3(str(ts3_file))

    recon_block = reconstructed_grid[0]
    assert recon_block.rpm == 3600.0, "rpm wiped by set_bv"
    assert recon_block.Nb == 24, "Nb wiped by set_bv"


def _make_inlet_writer():
    """A TS3Writer populated from a single-block grid with an inlet patch."""
    from ember.plugins.ts import TS3Writer

    block = Block(shape=(3, 4, 5))
    # Nb defaults to 1, so the whole pitch is the full circle; see
    # ember.nonreflecting.NonReflectingPatch._calc_hilbert.
    xrt = util.linmesh3([0.0, 1.0], [0.5, 1.5], [0.0, 2 * np.pi], (3, 4, 5))
    block.set_x(xrt[..., 0])
    block.set_r(xrt[..., 1])
    block.set_t(xrt[..., 2])
    block.set_fluid(PerfectFluid(cp=1005.0, gamma=1.4, mu=1.8e-5, Pr=0.7))
    inlet = InletPatch(i=0, j=(0, -1), k=(0, -1), label="inlet")
    block.patches.append(inlet)
    inlet.set_Po_To(Po=1.0e5, To=300.0)
    inlet.set_Alpha(0.0)
    inlet.set_Beta(0.0)
    writer = TS3Writer()
    writer.get(Grid([block]), strict=False)
    return writer


def test_set_pv_casts_existing_key():
    """set_pv updates an existing patch variable, casting to its type."""
    writer = _make_inlet_writer()
    writer.set_pv(0, 0, rfin=0.25)
    assert writer.pv[0][0]["rfin"] == pytest.approx(0.25)


def test_set_pv_rejects_unknown_name():
    """set_pv rejects a name not already present on the patch."""
    writer = _make_inlet_writer()
    with pytest.raises(Exception, match="Unknown patch variable"):
        writer.set_pv(0, 0, not_a_pv=1.0)


def test_set_pv_before_get_raises():
    """set_pv before get() raises a clear error."""
    from ember.plugins.ts import TS3Writer

    with pytest.raises(RuntimeError, match="before get"):
        TS3Writer().set_pv(0, 0, rfin=0.25)
