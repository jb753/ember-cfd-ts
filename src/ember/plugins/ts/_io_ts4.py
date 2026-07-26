"""Turbostream 4 (TS4) input/output, built on ember's ``Grid``/``Block`` API.

TS4 is an unstructured solver, but its input file is produced by converting a
TS3-format file rather than being written directly: the conversion is glue
around Turbostream Ltd's own proprietary ``ts.process``/VTK stack and cannot be
reimplemented here (see ``_scripts/convert_ts3_to_ts4.py``). This module
therefore covers the two ends of that pipeline:

* preparing the TS3-format file and the ``.ofp`` configuration the converter
  and solver consume, and post-processing the converted file to add the fields
  the conversion cannot supply (wall distance, throttle, probes);
* reading the unstructured solution back onto the structured grid.

The structured/unstructured correspondence is exact -- the conversion is a
format translation, not a remesh -- so both directions go through
:meth:`ember.grid.Grid.align_cart_unstr` rather than any interpolation. See
:func:`write_wall_distance` for what happens when that invariant is broken.
"""

import logging
import os

import h5py
import numpy as np
from scipy.spatial import KDTree

from ember.plugins.ts._io import (
    TS3Writer,
    _P_T_from_ts_datum,
    _check_perfect_fluid,
    _cv_Rgas_from_gas_properties,
)

logger = logging.getLogger("ember")


class TS4ConversionError(Exception):
    """The TS3 to TS4 conversion produced a file this plugin cannot use."""


# Baseline TS4 configuration. Values a TS4Solver field drives are overwritten in
# config_from_grid; the rest are solver defaults carried over verbatim.
DEFAULT_CONFIG = {
    "cfl": 50.0,
    "cfl_ramp": 1,
    "cfl_ramp_nstep": 500,
    "cfl_ramp_st": 1.0,
    "cfl_turb_fac": 0.25,
    "fluid_model": 0,
    "implicit_scheme": 1,
    "inlet_relax_fac": 0.5,
    "interpolation_extend_fac": 0.5,
    "interpolation_time_step_fac": 0.1,
    "istep_avg_start": 4000,
    "halo_implementation": 1,
    "kappa2": 1.0,
    "kappa4": 1.0 / 128.0,
    "mixing_alpha": 1.0,
    "mixing_bc_method": 2,
    "mixing_distribution_kind": 1,
    "mixing_method": 2,
    "mixing_distribution_spacing": 5e-4,
    "mixing_nstation": 200,
    "mixing_rf": 0.25,
    "mixing_rf_ramp_nstep": 500,
    "mixing_rf_ramp_st": 0.05,
    "mixing_rf_ramp_en": 0.25,
    "rot_fac_ramp": 0,
    "rot_fac_ramp_nstep": 0,
    "rot_fac_ramp_st": 0,
    "rot_fac_ramp_en": 1.0,
    "node_ordering_option": 2,
    "nstep": 5000,
    "prandtl_turbulent": 0.9,
    "sa_helicity_model": 0,
    "sa_lim_neg": 1,
    "time_step_mod_fac_nsmooth": 0,
    "time_step_mod_fac_smooth_frac": 0.5,
    "update_mixing_nstep": 5,
    "viscous_model": 2,
    "precon": 0,
    "precon_dissipation": 0,
    "precon_sigma_pgr": 3.0,
    "precon_beta": 3.0,
    "precon_mach_ref": 0.1,
    "precon_mach_lim": 0.1,
    "precon_fac_ramp": 0,
    "precon_fac_ramp_nstep": 100,
    "precon_fac_ramp_st": 0.1,
    "precon_fac_ramp_en": 1.0,
    "pout_fac_ramp_nstep": 0,
    "pout_fac_ramp": 0,
    "pout_fac_ramp_st": 0.8,
    "pout_fac_ramp_en": 1.0,
    "use_gpu_direct": 1,
    "if_dts": 0,
    "ncycle": 0,
    "nstep_cycle": 0,
    "frequency": 0.0,
    "nstep_save": 1e9,
}

# Ramp flags are each enabled by their own step-count field being non-zero, so
# a caller sets one number rather than a number and a flag that must agree.
_RAMP_FLAGS = {
    "pout_fac_ramp": "pout_fac_ramp_nstep",
    "rot_fac_ramp": "rot_fac_ramp_nstep",
    "mixing_rf_ramp": "mixing_rf_ramp_nstep",
}


def write_ofp(fname, var):
    """Write a dictionary of configuration variables in TS4's ``.ofp`` format."""
    with open(fname, "w") as f:
        for k, v in var.items():
            if isinstance(v, list):
                f.write(k + " = [" + ", ".join([str(vi) for vi in v]) + "]\n")
            else:
                f.write(k + " = " + str(v) + "\n")


def config_from_grid(ts4_config, grid):
    """Build the TS4 ``.ofp`` configuration dict for ``grid``.

    The configuration is not purely a dump of the solver's own fields: the gas
    properties come from the grid's fluid, and several ramp flags are derived
    from the step counts that drive them.

    Raises
    ------
    TypeError
        If the grid carries a fluid TS4 cannot be configured for here (only the
        perfect gas is supported; real gas needs property tables, which are out
        of scope for this port).
    """
    conf = DEFAULT_CONFIG.copy()

    # Typed fields of the same name override the defaults.
    for key in conf:
        if hasattr(ts4_config, key):
            conf[key] = getattr(ts4_config, key)

    # Gas properties from the grid's own fluid, sampled at a single node (a
    # block's fluid is spatially constant). Same source as the TS3 writer's av.
    node = grid[0][0, 0, 0]
    _check_perfect_fluid(grid[0].fluid)
    conf["fluid_model"] = 0
    conf["gamma"] = float(node.gamma)
    conf["cp"] = float(node.cp)
    conf["viscosity"] = float(node.mu)
    conf["prandtl_laminar"] = float(node.Pr)
    conf["prandtl_turbulent"] = float(node.Pr) * 1.25

    # Averaging window, counted back from the end of the run.
    nstep = conf["nstep"]
    istep_avg_start = nstep - ts4_config.nstep_avg
    if istep_avg_start < 0:
        raise ValueError(
            f"nstep_avg={ts4_config.nstep_avg} exceeds nstep={nstep}, so "
            "averaging would start before the run does."
        )
    conf["istep_avg_start"] = istep_avg_start

    # A CFL ramp always ends at the requested CFL number.
    if conf["cfl_ramp"]:
        conf["cfl_ramp_en"] = conf["cfl"]

    # Enable each ramp whose step count is set.
    for flag, nstep_key in _RAMP_FLAGS.items():
        if conf.get(nstep_key):
            conf[flag] = 1
    if conf["precon"]:
        conf["precon_fac_ramp"] = 1

    return conf


def write_ts3_input(grid, fname):
    """Write the TS3-format file that the TS3 to TS4 converter consumes.

    This is the same writer the TS3 solver uses, with one TS4-specific
    difference: the inlet pitch angle is written with its sign inverted, since
    TS4 takes the opposite sign convention to TS3 for that boundary condition.

    The flip is applied to the writer's extracted patch properties, not to the
    grid. Mutating the caller's grid and restoring it afterwards (as turbigen's
    TS4 solver did) leaves the sign inverted on the caller's own object if
    anything raises in between; here the grid is never touched at all.
    """
    writer = TS3Writer()
    # strict=True: the contract is a complete, runnable grid, so a missing flow
    # field or fluid is a bug to surface at write time.
    writer.get(grid, strict=True)

    # Inlet patches are the ones carrying stagnation conditions.
    n_flipped = 0
    for block_pp in writer.pp:
        for pp in block_pp.values():
            if "pitch" in pp:
                pp["pitch"] = -pp["pitch"]
                n_flipped += 1
    logger.debug(f"Inverted inlet pitch angle on {n_flipped} patches for TS4.")

    writer.check()
    writer.write(fname)


def write_wall_distance(grid, fname):
    """Transplant the grid's wall distance into a converted TS4 input file.

    The conversion is a format translation rather than a remesh, so every TS4
    node coincides with a grid node and the already-correct ``block.wdist`` can
    be scattered across directly. This is deliberately not a second,
    independent wall-distance calculation: ``block.wdist`` carries the
    periodicity and pitch-tiling handled by
    :meth:`~ember.grid.Grid.calculate_wdist`, and a recomputation here would
    have to be kept in agreement with it.

    The transfer is a *gather* -- each TS4 node takes the wall distance of the
    grid node it sits on -- rather than a scatter of each grid node's value
    into the TS4 array. That is not interchangeable here. A matching block
    interface is represented on both sides, so the grid and the TS4 file each
    hold two exactly coincident nodes there, and a nearest-neighbour query
    cannot tell them apart: scattering sends both blocks' interface nodes to
    the same TS4 index and leaves the other one unwritten. Gathering is
    well-defined for duplicates precisely because wall distance is a function
    of position alone, so coincident nodes must receive the same value anyway.

    Raises
    ------
    TS4ConversionError
        If the converted file does not have one node per grid node, or if any
        TS4 node does not lie on a grid node. Both indicate the conversion did
        not preserve the mesh, which this plugin cannot work around.
    """
    with h5py.File(fname, "r+") as f:
        n_ts4 = f["node_array_x"].shape[0]
        # Checked before alignment so the error names the converter rather than
        # surfacing as a bare assertion from inside core ember's aligner.
        if n_ts4 != grid.size:
            raise TS4ConversionError(
                f"TS4 input file {fname} has {n_ts4} nodes but the grid has "
                f"{grid.size}. The TS3 to TS4 conversion must preserve nodes "
                "one-for-one, including the duplicates at block interfaces; "
                "this indicates a bug in the vendor converter."
            )

        xyz = np.stack(
            [f["node_array_x"][:], f["node_array_y"][:], f["node_array_z"][:]],
            axis=-1,
        )

        # Reuse core ember's axis detection, then do the lookup in the grid's
        # own frame. Only perm/signs are needed: the block indices it also
        # returns are the scatter map, which is the direction that cannot
        # represent duplicated interface nodes.
        perm, signs, _ = grid.align_cart_unstr(xyz)
        xyz_aligned = xyz[:, list(perm)] * np.asarray(signs)

        grid_xyz = np.concatenate(
            [
                np.stack([b.x.ravel(), b.y.ravel(), b.z.ravel()], axis=-1)
                for b in grid
            ]
        )
        wdist_grid = np.concatenate([b.wdist.ravel() for b in grid])

        # Same tolerance rule core ember's aligner uses: a fraction of the
        # overall extent, so this is an exact-coincidence match rather than an
        # interpolation onto a resampled mesh.
        atol = 1e-6 * np.max(np.ptp(grid_xyz, axis=0))
        distance, indices = KDTree(grid_xyz).query(xyz_aligned)
        n_missed = int((distance > atol).sum())
        if n_missed:
            raise TS4ConversionError(
                f"{n_missed} of {n_ts4} TS4 nodes in {fname} do not lie on a "
                f"grid node (worst offset {distance.max():.3g}, tolerance "
                f"{atol:.3g}). The conversion must translate the mesh, not "
                "resample it."
            )

        f["node_array_wdist"][:] = wdist_grid[indices]


def write_throttle(ts4_config, grid, fname, workdir):
    """Write ``throttle_config.ofp`` for the grid's outlet patches.

    TS4 identifies outlet boundaries by tag string, so the grid's outlet
    patches must be matched up with the tags the converter emitted. Returns
    True if any outlet is under mass-flow control.
    """
    with h5py.File(fname, "r") as f:
        bcell_names = list(f["bcell_names"].attrs["names"])
    bcell_names = [
        n.decode() if isinstance(n, bytes) else str(n) for n in bcell_names
    ]
    tags = [i for i, b in enumerate(bcell_names) if ts4_config.outlet_tag == b]
    if not tags:
        raise TS4ConversionError(
            f"Could not find outlet tag {ts4_config.outlet_tag!r} in {fname}; "
            f"available bcell names are {bcell_names}."
        )

    outlets = grid.patches.outlet
    # One throttle entry per tag, so the parallel lists below cannot fall out of
    # step with each other. turbigen built the tag list and the value lists from
    # two different sources and relied on them happening to correspond.
    if len(tags) != len(outlets):
        raise TS4ConversionError(
            f"{fname} has {len(tags)} outlet bcells tagged "
            f"{ts4_config.outlet_tag!r} but the grid has {len(outlets)} outlet "
            "patches; the throttle configuration would be ambiguous."
        )

    mass_target = []
    throttle_kind = []
    throttle_k0 = []
    mdot_flag = False
    # TS4 converges in fewer steps than TS3, so the implicit scheme needs a
    # gentler gain than the TS3 value the patch carries.
    fac_adjust = 0.5 if ts4_config.implicit_scheme else 1.0

    for patch in outlets:
        # Note the predicate: ember's OutletPatch always has a K_pid tuple
        # (defaulting to zeros), so unlike turbigen's port there is no None to
        # test against -- an unthrottled outlet is one with no mass-flow target.
        if patch.mdot_target > 0:
            mdot_flag = True
            # TS4 targets the mass flow through one passage, not the annulus.
            mass_target.append(float(patch.mdot_target) / float(patch.block.Nb))
            throttle_kind.append(0)
            # TS4 offers integral control only, so only the I gain carries over.
            throttle_k0.append(float(patch.K_pid[1]) * fac_adjust)
        else:
            mass_target.append(0.0)
            throttle_kind.append(1)
            throttle_k0.append(0.0)

    throttle_path = os.path.join(workdir, "throttle_config.ofp")
    if os.path.exists(throttle_path):
        os.remove(throttle_path)

    if mdot_flag or ts4_config.area_avg_pout:
        write_ofp(
            throttle_path,
            {
                "throttle_tag_list": tags,
                "throttle_target_list": mass_target,
                "throttle_k0_list": throttle_k0,
                "throttle_correction_initial_list": [0.0 for _ in tags],
                "throttle_kind_list": throttle_kind,
            },
        )

    return mdot_flag


def write_probes_ofp(workdir):
    """Start a fresh ``probes.ofp``, holding an empty probe list."""
    with open(os.path.join(workdir, "probes.ofp"), "w") as f:
        f.write(
            """
import ts.process.probe.probe_definition
probe_list = []

"""
        )


def write_point_probe(ts4_config, workdir, xyz, domain, label):
    """Append a point probe definition to ``probes.ofp``."""
    with open(os.path.join(workdir, "probes.ofp"), "a") as f:
        f.write(
            f"""
p = ts.process.probe.probe_definition.ProbeDefinition()
p.kind = "point"
p.x = {xyz[0].tolist()}
p.y = {xyz[1].tolist()}
p.z = {xyz[2].tolist()}
p.idomain = {domain}
p.absolute_frame = True
p.fname_root = "point_probe_{label}"
p.write_2d = True
p.nstep_save_start_1d = {ts4_config.nstep_save_start_probe_1d}
p.nstep_save_1d = {ts4_config.nstep_save_probe_1d}
p.nstep_save_start_2d = {ts4_config.nstep_save_start_probe_1d}
p.nstep_save_2d = {ts4_config.nstep_save_probe_1d}
p.time_average = False
probe_list.append(p)

"""
        )


def write_logical_probe(ts4_config, workdir, tag, label):
    """Append a logical (named-boundary) probe definition to ``probes.ofp``."""
    istep_save_start = ts4_config.nstep - ts4_config.nstep_avg
    with open(os.path.join(workdir, "probes.ofp"), "a") as f:
        f.write(
            f"""
p = ts.process.probe.probe_definition.ProbeDefinition()
p.kind = "logical"
p.val = "{tag}"
p.fname_root = "logical_probe_{label}"
p.write_2d = False
p.nstep_save_start_1d = {istep_save_start}
p.nstep_save_1d = {ts4_config.nstep_save_probe_1d}
p.nstep_save_start_2d = 99999
p.nstep_save_2d = 0
probe_list.append(p)

"""
        )


def check_probes(workdir):
    """Reject a ``probes.ofp`` that TS4 would fail to execute.

    The file is Python that the solver runs, so a syntax error would otherwise
    only surface after the job has been queued and started.
    """
    path = os.path.join(workdir, "probes.ofp")
    with open(path) as f:
        text = f.read()
    try:
        compile(text, path, "exec")
    except SyntaxError as e:
        raise ValueError(f"Generated probe file {path} is not valid Python") from e


def read_flow(grid, fname, fname_avg):
    """Read a TS4 solution back onto ``grid``, in place.

    Coordinates come from the instantaneous output and the flow field from the
    averaged one, matching how TS4 writes them.

    The conserved variables are converted to primitives before being handed to
    ember, rather than going through
    :meth:`~ember.grid.Grid.set_conserved_cart_unstr` despite the array
    signatures matching: TS4's total energy sits on Turbostream's own
    thermodynamic datum, and the conserved route would store it on the grid's
    datum and silently corrupt temperature. Pressure and density are
    datum-free, so the primitive route needs no further handling.

    Raises
    ------
    FileNotFoundError
        If the averaged output is absent (TS4 omits it when the run stopped
        before averaging began).
    """
    if not os.path.exists(fname_avg):
        raise FileNotFoundError(
            f"No averaged TS4 output at {fname_avg}. TS4 writes it only once "
            "averaging has started, so the run may have stopped early or "
            "nstep_avg may be zero."
        )

    with h5py.File(fname, "r") as f:
        xyz = np.stack(
            [f["node_array_x"][:], f["node_array_y"][:], f["node_array_z"][:]],
            axis=-1,
        )
    with h5py.File(fname_avg, "r") as f:
        rho = f["node_array_ro"][:]
        rhoVx = f["node_array_rovx"][:]
        rhoVy = f["node_array_rovy"][:]
        rhoVz = f["node_array_rovz"][:]
        rhoe = f["node_array_roe"][:]
        mu_turb = f["node_array_turb0"][:]

    if np.any(rho <= 0.0) or not np.all(np.isfinite(rho)):
        raise ValueError("TS4 output has non-positive or non-finite density")

    Vx, Vy, Vz = rhoVx / rho, rhoVy / rho, rhoVz / rho
    u = rhoe / rho - 0.5 * (Vx**2 + Vy**2 + Vz**2)

    fluid = grid[0].fluid
    _check_perfect_fluid(fluid)
    node = grid[0][0, 0, 0]
    cv, Rgas = _cv_Rgas_from_gas_properties(float(node.cp), float(node.gamma))
    P, _ = _P_T_from_ts_datum(rho, u, cv, Rgas)

    primitive_cart = np.stack([rho, Vx, Vy, Vz, P], axis=-1)
    grid.set_primitive_cart_unstr(xyz, primitive_cart)

    # Turbulent viscosity is a plain nodal scalar, so it is scattered on the
    # index correspondence rather than going through the primitive vector. This
    # repeats the alignment that set_primitive_cart_unstr just did internally;
    # exposing the indices from that call would avoid it, but at the cost of
    # widening a core ember signature for one caller.
    _, _, block_indices = grid.align_cart_unstr(xyz)
    for block, indices in zip(grid, block_indices):
        block.set_mu_turb(mu_turb[indices.flatten()].reshape(block.shape))

    return grid


__all__ = [
    "DEFAULT_CONFIG",
    "TS4ConversionError",
    "check_probes",
    "config_from_grid",
    "read_flow",
    "write_logical_probe",
    "write_ofp",
    "write_point_probe",
    "write_probes_ofp",
    "write_throttle",
    "write_ts3_input",
    "write_wall_distance",
]
