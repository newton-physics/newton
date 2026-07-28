# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import time
import unittest
from enum import Enum

import numpy as np
import warp as wp

import newton
from newton.geometry import HydroelasticSDF
from newton.tests.unittest_utils import (
    add_function_test,
    get_selected_cuda_test_devices,
)

# --- Configuration ---


class ShapeType(Enum):
    PRIMITIVE = "primitive"
    MESH = "mesh"


# Scene parameters
CUBE_HALF_LARGE = 0.5  # 1m cube
CUBE_HALF_SMALL = 0.005  # 1cm cube
NUM_CUBES = 3

# Simulation parameters
SIM_SUBSTEPS = 10
SIM_DT = 1.0 / 60.0
SIM_TIME = 1.0
VIEWER_NUM_FRAMES = 300

# Test thresholds
POSITION_THRESHOLD_FACTOR = 0.20  # multiplied by cube_half
MAX_ROTATION_DEG = 10.0

# Devices and solvers
cuda_devices = get_selected_cuda_test_devices()

solvers = {
    "mujoco_warp": lambda model: newton.solvers.SolverMuJoCo(
        model,
        use_mujoco_cpu=False,
        use_mujoco_contacts=False,
        njmax=500,
        nconmax=200,
        solver="newton",
        ls_iterations=100,
    ),
    "xpbd": lambda model: newton.solvers.SolverXPBD(model, iterations=10),
}


# --- Helper functions ---


def simulate(solver, model, state_0, state_1, control, contacts, collision_pipeline, sim_dt, substeps):
    for _ in range(substeps):
        state_0.clear_forces()
        collision_pipeline.collide(state_0, contacts)
        solver.step(state_0, state_1, control, contacts, sim_dt / substeps)
        state_0, state_1 = state_1, state_0
    return state_0, state_1


def build_stacked_cubes_scene(
    device,
    solver_fn,
    shape_type: ShapeType,
    cube_half: float = CUBE_HALF_LARGE,
    reduce_contacts: bool = True,
    sdf_hydroelastic_config: HydroelasticSDF.Config | None = None,
):
    """Build the stacked cubes scene and return all components for simulation."""
    cube_mesh = None
    if shape_type == ShapeType.MESH:
        cube_mesh = newton.Mesh.create_box(
            cube_half,
            cube_half,
            cube_half,
            duplicate_vertices=False,
            compute_normals=False,
            compute_uvs=False,
            compute_inertia=False,
        )

    # Scale SDF parameters proportionally to cube size
    narrow_band = cube_half * 0.2
    contact_gap = cube_half * 0.2

    if cube_mesh is not None:
        cube_mesh.build_sdf(
            max_resolution=32,
            narrow_band_range=(-narrow_band, narrow_band),
            margin=contact_gap,
            device=device,
        )

    builder = newton.ModelBuilder()
    if shape_type == ShapeType.PRIMITIVE:
        builder.default_shape_cfg = newton.ModelBuilder.ShapeConfig(
            mu=0.5,
            sdf_max_resolution=32,
            is_hydroelastic=True,
            sdf_narrow_band_range=(-narrow_band, narrow_band),
            gap=contact_gap,
        )
    else:
        builder.default_shape_cfg = newton.ModelBuilder.ShapeConfig(
            mu=0.5,
            is_hydroelastic=True,
            gap=contact_gap,
        )

    builder.add_ground_plane()

    initial_positions = []
    for i in range(NUM_CUBES):
        z_pos = cube_half + i * cube_half * 2.0
        initial_positions.append(wp.vec3(0.0, 0.0, z_pos))
        body = builder.add_body(
            xform=wp.transform(initial_positions[-1], wp.quat_identity()),
            label=f"{shape_type.value}_cube_{i}",
        )

        if shape_type == ShapeType.PRIMITIVE:
            builder.add_shape_box(body=body, hx=cube_half, hy=cube_half, hz=cube_half)
        else:
            builder.add_shape_mesh(body=body, mesh=cube_mesh)

    model = builder.finalize(device=device)
    solver = solver_fn(model)

    state_0 = model.state()
    state_1 = model.state()
    control = model.control()

    newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)

    if sdf_hydroelastic_config is None:
        sdf_hydroelastic_config = HydroelasticSDF.Config(
            output_contact_surface=True,
            reduce_contacts=reduce_contacts,
            anchor_contact=True,
            buffer_fraction=1.0,
        )

    # Hydroelastic without contact reduction can generate many contacts
    rigid_contact_max = 6000 if not reduce_contacts else 100

    collision_pipeline = newton.CollisionPipeline(
        model,
        rigid_contact_max=rigid_contact_max,
        broad_phase="explicit",
        sdf_hydroelastic_config=sdf_hydroelastic_config,
    )

    return model, solver, state_0, state_1, control, collision_pipeline, initial_positions, cube_half


# --- Test functions ---


def run_stacked_cubes_hydroelastic_test(
    test,
    device,
    solver_fn,
    shape_type: ShapeType,
    cube_half: float = CUBE_HALF_LARGE,
    reduce_contacts: bool = True,
    config: HydroelasticSDF.Config | None = None,
    position_threshold_factor: float = POSITION_THRESHOLD_FACTOR,
    substeps: int | None = None,
):
    """Shared test for stacking 3 cubes using hydroelastic contacts."""
    model, solver, state_0, state_1, control, collision_pipeline, initial_positions, cube_half = (
        build_stacked_cubes_scene(device, solver_fn, shape_type, cube_half, reduce_contacts, config)
    )

    contacts = collision_pipeline.contacts()
    collision_pipeline.collide(state_0, contacts)

    sdf_sdf_count = collision_pipeline.narrow_phase.shape_pairs_sdf_sdf_count.numpy()[0]
    test.assertEqual(sdf_sdf_count, NUM_CUBES - 1, f"Expected {NUM_CUBES - 1} sdf_sdf collisions, got {sdf_sdf_count}")

    num_frames = int(SIM_TIME / SIM_DT)

    # Scale substeps for small objects - they need smaller time steps for stability
    if substeps is None:
        substeps = SIM_SUBSTEPS if cube_half >= CUBE_HALF_LARGE else 25

    for _ in range(num_frames):
        state_0, state_1 = simulate(
            solver, model, state_0, state_1, control, contacts, collision_pipeline, SIM_DT, substeps
        )

    body_q = state_0.body_q.numpy()

    position_threshold = position_threshold_factor * cube_half

    for i in range(NUM_CUBES):
        expected_z = initial_positions[i][2]
        actual_pos = body_q[i, :3]
        displacement = np.linalg.norm(actual_pos - np.array([0.0, 0.0, expected_z]))

        test.assertLess(
            displacement,
            position_threshold,
            f"{shape_type.value.capitalize()} cube {i} moved {displacement:.6f}, exceeding threshold {position_threshold:.6f}",
        )

        initial_quat = np.array([0.0, 0.0, 0.0, 1.0])
        final_quat = body_q[i, 3:]
        dot_product = np.abs(np.dot(initial_quat, final_quat))
        dot_product = np.clip(dot_product, 0.0, 1.0)
        rotation_angle = 2.0 * np.arccos(dot_product)

        test.assertLess(
            rotation_angle,
            np.radians(MAX_ROTATION_DEG),
            f"{shape_type.value.capitalize()} cube {i} rotated {np.degrees(rotation_angle):.2f} degrees, exceeding threshold {MAX_ROTATION_DEG} degrees",
        )


def test_stacked_mesh_cubes_hydroelastic(test, device, solver_fn):
    """Test 3 mesh cubes (1m) stacked on each other remain stable for 1 second using hydroelastic contacts."""
    run_stacked_cubes_hydroelastic_test(test, device, solver_fn, ShapeType.MESH, CUBE_HALF_LARGE)


def test_stacked_small_primitive_cubes_hydroelastic(test, device, solver_fn):
    """Test 3 small primitive cubes (1cm) stacked on each other remain stable for 1 second using hydroelastic contacts."""
    # This scene can exceed the default pre-pruned face-contact budget on CI GPUs,
    # which emits overflow warnings and can perturb stability assertions.
    # Keep defaults unchanged and increase capacity only for this stress test.
    config = HydroelasticSDF.Config(buffer_mult_contact=2)
    run_stacked_cubes_hydroelastic_test(test, device, solver_fn, ShapeType.PRIMITIVE, CUBE_HALF_SMALL, config=config)


def test_stacked_small_mesh_cubes_hydroelastic(test, device, solver_fn):
    """Test 3 small mesh cubes (1cm) stacked on each other remain stable for 1 second using hydroelastic contacts."""
    # This scene can exceed the default pre-pruned face-contact budget on CI GPUs,
    # which emits overflow warnings that fail check_output-enabled tests.
    # Keep defaults unchanged and increase capacity only for this stress test.
    config = HydroelasticSDF.Config(buffer_mult_contact=2)
    run_stacked_cubes_hydroelastic_test(test, device, solver_fn, ShapeType.MESH, CUBE_HALF_SMALL, config=config)


def test_stacked_primitive_cubes_hydroelastic_no_reduction(test, device, solver_fn):
    """Test 3 primitive cubes (1m) stacked without contact reduction using hydroelastic contacts."""
    run_stacked_cubes_hydroelastic_test(
        test,
        device,
        solver_fn,
        ShapeType.PRIMITIVE,
        CUBE_HALF_LARGE,
        False,
        position_threshold_factor=0.50,
        substeps=20,
    )


def test_buffer_fraction_no_crash(test, device):
    """Validate reduced buffer allocation still yields contacts.

    Args:
        test: Unittest-style assertion helper.
        device: Warp device under test.
    """
    cube_half = 0.5
    narrow_band = cube_half * 0.2
    contact_gap = cube_half * 0.2
    num_cubes = 3

    builder = newton.ModelBuilder()
    builder.default_shape_cfg = newton.ModelBuilder.ShapeConfig(
        sdf_max_resolution=32,
        is_hydroelastic=True,
        sdf_narrow_band_range=(-narrow_band, narrow_band),
        gap=contact_gap,
    )
    builder.add_ground_plane()

    for i in range(num_cubes):
        z_pos = cube_half + i * cube_half * 2.0
        body = builder.add_body(xform=wp.transform(p=wp.vec3(0.0, 0.0, z_pos), q=wp.quat_identity()))
        builder.add_shape_box(body=body, hx=cube_half, hy=cube_half, hz=cube_half)

    model = builder.finalize(device=device)
    state = model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state)

    # Reduced allocation with moderate headroom.
    config_reduced = HydroelasticSDF.Config(buffer_fraction=0.8)
    pipeline_reduced = newton.CollisionPipeline(
        model,
        broad_phase="explicit",
        sdf_hydroelastic_config=config_reduced,
    )

    contacts_reduced = pipeline_reduced.contacts()
    pipeline_reduced.collide(state, contacts_reduced)
    reduced_count = int(contacts_reduced.rigid_contact_count.numpy()[0])
    test.assertGreater(reduced_count, 0, "Expected non-zero contacts with reduced buffer_fraction")

    # Full allocation should not produce significantly fewer contacts.
    # Allow a small tolerance for non-deterministic contact counts.
    config_full = HydroelasticSDF.Config(buffer_fraction=1.0)
    pipeline_full = newton.CollisionPipeline(
        model,
        broad_phase="explicit",
        sdf_hydroelastic_config=config_full,
    )
    contacts_full = pipeline_full.contacts()
    pipeline_full.collide(state, contacts_full)
    full_count = int(contacts_full.rigid_contact_count.numpy()[0])

    tolerance = max(2, int(0.05 * reduced_count))
    test.assertGreaterEqual(
        full_count + tolerance,
        reduced_count,
        f"Full buffers ({full_count}) produced significantly fewer contacts than reduced buffers ({reduced_count})",
    )


def test_iso_scan_scratch_buffers_are_level_sized(test, device):
    """Validate iso-scan scratch buffers match each level input size.

    Args:
        test: Unittest-style assertion helper.
        device: Warp device under test.
    """
    # Small cubes generate many contacts; increase buffer to avoid overflow warnings
    model, _, state_0, _, _, pipeline, _, _ = build_stacked_cubes_scene(
        device=device,
        solver_fn=solvers["xpbd"],
        shape_type=ShapeType.PRIMITIVE,
        cube_half=CUBE_HALF_SMALL,
        reduce_contacts=True,
        sdf_hydroelastic_config=HydroelasticSDF.Config(buffer_mult_contact=2),
    )
    newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)
    contacts = pipeline.contacts()
    pipeline.collide(state_0, contacts)
    wp.synchronize()

    hydro = pipeline.hydroelastic_sdf
    test.assertIsNotNone(hydro)

    test.assertEqual(len(hydro.input_sizes), 4)
    test.assertEqual(len(hydro.iso_buffer_num_scratch), 4)
    test.assertEqual(len(hydro.iso_buffer_prefix_scratch), 4)
    test.assertEqual(len(hydro.iso_subblock_idx_scratch), 4)
    for i, level_input in enumerate(hydro.input_sizes):
        test.assertEqual(hydro.iso_buffer_num_scratch[i].shape[0], level_input)
        test.assertEqual(hydro.iso_buffer_prefix_scratch[i].shape[0], level_input)
        test.assertEqual(hydro.iso_subblock_idx_scratch[i].shape[0], level_input)


def test_reduce_contacts_with_pre_prune_disabled_no_crash(test, device):
    """Validate the reduce_contacts=True, pre_prune_contacts=False path."""
    config = HydroelasticSDF.Config(
        reduce_contacts=True,
        pre_prune_contacts=False,
        buffer_fraction=1.0,
        buffer_mult_contact=2,
    )
    model, _, state_0, _, _, pipeline, _, _ = build_stacked_cubes_scene(
        device=device,
        solver_fn=solvers["xpbd"],
        shape_type=ShapeType.MESH,
        cube_half=CUBE_HALF_SMALL,
        reduce_contacts=True,
        sdf_hydroelastic_config=config,
    )
    newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)
    contacts = pipeline.contacts()
    pipeline.collide(state_0, contacts)

    rigid_count = int(contacts.rigid_contact_count.numpy()[0])
    test.assertGreater(rigid_count, 0, "Expected non-zero contacts with pre_prune_contacts=False")


@wp.kernel
def _set_body_z_kernel(
    body_q: wp.array[wp.transform],
    body_idx: int,
    z: float,
):
    cur = body_q[body_idx]
    p = wp.transform_get_translation(cur)
    body_q[body_idx] = wp.transform(wp.vec3(p[0], p[1], z), wp.transform_get_rotation(cur))


def _extract_contact_forces(contacts, model, state, shape_pair=None):
    """Extract active contact force magnitudes, world-frame points, normals, and friction.

    Args:
        contacts: Contacts buffer.
        model: Newton model.
        state: Newton state.
        shape_pair: Optional (shape_a, shape_b) tuple to filter contacts to a specific pair.

    Returns (force_mag, p0w, p1w, normals, friction) arrays filtered to active contacts,
    or all-empty arrays when there are no active contacts.
    """
    n = int(contacts.rigid_contact_count.numpy()[0])
    empty = np.empty((0, 3)), np.empty((0, 3)), np.empty((0, 3)), np.empty(0), np.empty(0)
    if n == 0 or contacts.rigid_contact_stiffness is None:
        return empty

    normals = contacts.rigid_contact_normal.numpy()[:n]
    p0 = contacts.rigid_contact_point0.numpy()[:n]
    p1 = contacts.rigid_contact_point1.numpy()[:n]
    stiffness = contacts.rigid_contact_stiffness.numpy()[:n]
    shape0 = contacts.rigid_contact_shape0.numpy()[:n]
    shape1 = contacts.rigid_contact_shape1.numpy()[:n]
    shape_body = model.shape_body.numpy()
    body_q = state.body_q.numpy()

    b0 = shape_body[shape0]
    b1 = shape_body[shape1]
    # Translate contact points to world frame (body == -1 means world already)
    off0 = np.where((b0 != -1)[:, None], body_q[np.maximum(b0, 0), :3], 0.0)
    off1 = np.where((b1 != -1)[:, None], body_q[np.maximum(b1, 0), :3], 0.0)
    p0w = p0 + off0
    p1w = p1 + off1
    depth = np.einsum("ij,ij->i", p0w - p1w, -normals) / 2.0
    mask = (stiffness > 0) & (depth < 0)
    if shape_pair is not None:
        pair_mask = (shape0 == shape_pair[0]) & (shape1 == shape_pair[1])
        pair_mask |= (shape0 == shape_pair[1]) & (shape1 == shape_pair[0])
        mask = mask & pair_mask

    force_mag = stiffness[mask] * (-depth[mask])
    friction = contacts.rigid_contact_friction.numpy()[:n][mask]
    # friction == 0 means "unset" → default scale 1.0
    friction = np.where(friction > 0.0, friction, 1.0)
    return p0w[mask], p1w[mask], normals[mask], force_mag, friction


def _compute_net_force(contacts, model, state):
    """Compute net contact force from a contacts buffer."""
    _, _, normals, force_mag, _ = _extract_contact_forces(contacts, model, state)
    if len(force_mag) == 0:
        return np.zeros(3)
    return np.sum(force_mag[:, None] * (-normals), axis=0)


def _compute_force_weighted_anchor(contacts, model, state, shape_pair=None):
    """Return the force-weighted center of pressure for active contacts."""
    p0w, p1w, _, force_mag, _ = _extract_contact_forces(contacts, model, state, shape_pair=shape_pair)
    if len(force_mag) == 0:
        return np.zeros(3)
    contact_pos = (p0w + p1w) / 2.0
    return (force_mag[:, None] * contact_pos).sum(axis=0) / force_mag.sum()


def _compute_net_moment(contacts, model, state, anchor=None, shape_pair=None):
    """Compute net friction moment from a contacts buffer."""
    p0w, p1w, normals, force_mag, friction = _extract_contact_forces(contacts, model, state, shape_pair=shape_pair)
    if len(force_mag) == 0:
        return 0.0

    contact_pos = (p0w + p1w) / 2.0
    if anchor is None:
        total_weight = force_mag.sum()
        anchor = (force_mag[:, None] * contact_pos).sum(axis=0) / total_weight

    r = contact_pos - anchor
    neg_normals = -normals
    lever = np.linalg.norm(np.cross(r, neg_normals), axis=1)

    return float((friction * force_mag * lever).sum())


def _build_cube_sphere_scene(device, cube_half=0.1, sphere_radius=0.1):
    """Build a cube-on-ground + sphere-on-cube scene for contact comparison tests.

    Returns (model, state, sphere_body, rest_z).
    """
    shape_cfg = newton.ModelBuilder.ShapeConfig(
        sdf_max_resolution=128,
        is_hydroelastic=True,
        sdf_narrow_band_range=(-0.01, 0.01),
        gap=0.01,
        kh=1e9,
    )
    builder = newton.ModelBuilder()
    builder.default_shape_cfg = shape_cfg
    builder.add_ground_plane()

    cube_body = builder.add_body(
        xform=wp.transform(wp.vec3(0.0, 0.0, cube_half), wp.quat_identity()),
        label="cube",
    )
    builder.add_shape_box(body=cube_body, hx=cube_half, hy=cube_half, hz=cube_half)

    rest_z = 2 * cube_half + sphere_radius
    sphere_body = builder.add_body(
        xform=wp.transform(wp.vec3(0.0, 0.0, rest_z), wp.quat_identity()),
        label="sphere",
    )
    builder.add_shape_sphere(body=sphere_body, radius=sphere_radius)

    model = builder.finalize(device=device)
    state = model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state)

    return model, state, sphere_body, rest_z


def _make_pipelines(model, configs, rigid_contact_maxes=None):
    """Create collision pipelines and contacts for a list of HydroelasticSDF.Configs.

    Returns list of (pipeline, contacts) tuples.
    """
    if rigid_contact_maxes is None:
        rigid_contact_maxes = [500] * len(configs)
    result = []
    for cfg, rcm in zip(configs, rigid_contact_maxes, strict=True):
        pipe = newton.CollisionPipeline(model, rigid_contact_max=rcm, sdf_hydroelastic_config=cfg)
        result.append((pipe, pipe.contacts()))
    return result


def test_reduced_vs_unreduced_contact_forces(test, device, anchor_contact=False):
    """Reduced and unreduced hydroelastic forces must agree within 1%."""
    model, state, sphere_body, rest_z = _build_cube_sphere_scene(device)

    cfg_reduced = HydroelasticSDF.Config(
        output_contact_surface=True,
        reduce_contacts=True,
        anchor_contact=anchor_contact,
    )
    cfg_unreduced = HydroelasticSDF.Config(
        output_contact_surface=True,
        reduce_contacts=False,
        anchor_contact=False,
    )
    (pipe_red, contacts_red), (pipe_unr, contacts_unr) = _make_pipelines(
        model, [cfg_reduced, cfg_unreduced], [500, 20000]
    )

    anchor_label = "with anchor" if anchor_contact else "without anchor"

    for pen in [0.0, 1e-4, 1e-3, 1e-2]:
        sphere_z = rest_z - pen
        wp.launch(_set_body_z_kernel, dim=1, inputs=[state.body_q, sphere_body, sphere_z], device=device)

        pipe_red.collide(state, contacts_red)
        pipe_unr.collide(state, contacts_unr)

        f_red = _compute_net_force(contacts_red, model, state)
        f_unr = _compute_net_force(contacts_unr, model, state)

        if pen == 0.0:
            # No penetration — both forces should be near zero
            test.assertLess(np.linalg.norm(f_red), 1e-3, f"pen={pen} ({anchor_label}): reduced force should be ~0")
            test.assertLess(np.linalg.norm(f_unr), 1e-3, f"pen={pen} ({anchor_label}): unreduced force should be ~0")
            continue

        # z-component (normal force) — must be positive and match within 1%
        test.assertGreater(f_unr[2], 0.0, f"pen={pen} ({anchor_label}): unreduced Fz should be positive")
        rel_z = abs(f_red[2] - f_unr[2]) / abs(f_unr[2])
        test.assertLess(rel_z, 0.01, f"pen={pen} ({anchor_label}): Fz mismatch {rel_z * 100:.2f}%")

        # xy-components — should be small; match as fraction of Fz
        for axis, label in [(0, "Fx"), (1, "Fy")]:
            abs_diff = abs(f_red[axis] - f_unr[axis])
            test.assertLess(
                abs_diff / abs(f_unr[2]),
                0.01,
                f"pen={pen} ({anchor_label}): {label} diff {abs_diff:.4f} > 1% of Fz {f_unr[2]:.4f}",
            )


def test_reduced_vs_unreduced_contact_forces_with_anchor_contact(test, device):
    """Reduced hydroelastic forces must still match with anchor_contact enabled."""
    test_reduced_vs_unreduced_contact_forces(test, device, anchor_contact=True)


def test_reduced_vs_unreduced_contact_moments(test, device):
    """Reduced and unreduced hydroelastic moments must agree with moment_matching."""
    model, state, sphere_body, rest_z = _build_cube_sphere_scene(device)

    cfg_reduced = HydroelasticSDF.Config(
        output_contact_surface=True,
        reduce_contacts=True,
        anchor_contact=True,
        moment_matching=True,
    )
    cfg_unreduced = HydroelasticSDF.Config(
        output_contact_surface=True,
        reduce_contacts=False,
        anchor_contact=False,
    )
    (pipe_red, contacts_red), (pipe_unr, contacts_unr) = _make_pipelines(
        model, [cfg_reduced, cfg_unreduced], [500, 20000]
    )

    # Filter to the cube-sphere shape pair (shape 1=cube, shape 2=sphere).
    sp = (1, 2)

    for pen in [0.0, 1e-3, 1e-2]:
        sphere_z = rest_z - pen
        wp.launch(_set_body_z_kernel, dim=1, inputs=[state.body_q, sphere_body, sphere_z], device=device)

        pipe_red.collide(state, contacts_red)
        pipe_unr.collide(state, contacts_unr)

        anchor = _compute_force_weighted_anchor(contacts_unr, model, state, shape_pair=sp)

        m_red = _compute_net_moment(contacts_red, model, state, anchor=anchor, shape_pair=sp)
        m_unr = _compute_net_moment(contacts_unr, model, state, anchor=anchor, shape_pair=sp)

        if pen == 0.0:
            test.assertLess(abs(m_red), 1e-3, f"pen={pen}: reduced moment should be ~0")
            test.assertLess(abs(m_unr), 1e-3, f"pen={pen}: unreduced moment should be ~0")
            continue

        # Both moments should be non-negative
        test.assertGreaterEqual(m_unr, 0.0, f"pen={pen}: unreduced moment should be >= 0")

        if m_unr > 1e-6:
            rel = abs(m_red - m_unr) / m_unr
            test.assertLess(
                rel,
                0.4,
                f"pen={pen}: moment mismatch {rel * 100:.2f}% (reduced={m_red:.6f}, unreduced={m_unr:.6f})",
            )


def _compute_total_friction_capacity(contacts, model, state, shape_pair=None):
    """Compute total lateral friction capacity: sum(friction_scale * normal_force)."""
    _, _, _, force_mag, friction = _extract_contact_forces(contacts, model, state, shape_pair=shape_pair)
    if len(force_mag) == 0:
        return 0.0
    return float((friction * force_mag).sum())


def _build_cube_cube_scene(device, cube_half_lower=0.2, cube_half_upper=0.1, kh_lower=1e9, kh_upper=1e9):
    """Build a big-cube-on-ground + small-cube-on-top scene for contact comparison tests.

    Returns (model, state, upper_body, rest_z).
    """

    def shape_cfg(kh):
        return newton.ModelBuilder.ShapeConfig(
            sdf_max_resolution=128,
            is_hydroelastic=True,
            sdf_narrow_band_range=(-0.01, 0.01),
            gap=0.01,
            kh=kh,
        )

    builder = newton.ModelBuilder()
    builder.add_ground_plane()

    lower_body = builder.add_body(
        xform=wp.transform(wp.vec3(0.0, 0.0, cube_half_lower), wp.quat_identity()),
        label="lower_cube",
    )
    builder.add_shape_box(
        body=lower_body,
        hx=cube_half_lower,
        hy=cube_half_lower,
        hz=cube_half_lower,
        cfg=shape_cfg(kh_lower),
    )

    rest_z = 2 * cube_half_lower + cube_half_upper
    upper_body = builder.add_body(
        xform=wp.transform(wp.vec3(0.0, 0.0, rest_z), wp.quat_identity()),
        label="upper_cube",
    )
    builder.add_shape_box(
        body=upper_body,
        hx=cube_half_upper,
        hy=cube_half_upper,
        hz=cube_half_upper,
        cfg=shape_cfg(kh_upper),
    )

    model = builder.finalize(device=device)
    state = model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state)

    return model, state, upper_body, rest_z


def test_reduced_vs_unreduced_contact_forces_cube_on_cube(test, device):
    """Reduced and unreduced hydroelastic forces must agree within 1% for cube-on-cube."""
    model, state, upper_body, rest_z = _build_cube_cube_scene(device)

    cfg_reduced = HydroelasticSDF.Config(
        output_contact_surface=True,
        reduce_contacts=True,
        anchor_contact=False,
    )
    cfg_unreduced = HydroelasticSDF.Config(
        output_contact_surface=True,
        reduce_contacts=False,
        anchor_contact=False,
    )
    (pipe_red, contacts_red), (pipe_unr, contacts_unr) = _make_pipelines(
        model, [cfg_reduced, cfg_unreduced], [500, 50000]
    )

    for pen in [1e-4, 1e-3, 1e-2]:
        upper_z = rest_z - pen
        wp.launch(_set_body_z_kernel, dim=1, inputs=[state.body_q, upper_body, upper_z], device=device)

        pipe_red.collide(state, contacts_red)
        pipe_unr.collide(state, contacts_unr)

        f_red = _compute_net_force(contacts_red, model, state)
        f_unr = _compute_net_force(contacts_unr, model, state)

        # z-component (normal force) — must be nonzero and match within 1%
        test.assertGreater(abs(f_unr[2]), 0.0, f"pen={pen}: unreduced Fz should be nonzero")
        rel_z = abs(f_red[2] - f_unr[2]) / abs(f_unr[2])
        test.assertLess(rel_z, 0.01, f"pen={pen}: Fz mismatch {rel_z * 100:.2f}%")

        # xy-components — should be small; match as fraction of |Fz|
        for axis, label in [(0, "Fx"), (1, "Fy")]:
            abs_diff = abs(f_red[axis] - f_unr[axis])
            test.assertLess(
                abs_diff / abs(f_unr[2]),
                0.01,
                f"pen={pen}: {label} diff {abs_diff:.4f} > 1% of |Fz| {abs(f_unr[2]):.4f}",
            )


# User-defined pressure-callback equivalent to the built-in linear law
# ``pressure = -kh * signed_depth``. Defined here (not imported from
# ``newton._src``) to exercise the public callback API the same way user code
# would, mirroring ``newton/examples/contacts/example_nut_bolt_hydro.py``.
@wp.struct
class _LinearPressureData:
    shape_kh: wp.array[wp.float32]


@wp.func
def _linear_pressure(signed_depth: wp.float32, shape_idx: wp.int32, data: _LinearPressureData) -> wp.float32:
    return -data.shape_kh[shape_idx] * signed_depth


@wp.struct
class _PowerPressureData:
    shape_kh: wp.array[wp.float32]
    depth_ref_m: wp.float32
    exponent: wp.float32


@wp.func
def _power_pressure(signed_depth: wp.float32, shape_idx: wp.int32, data: _PowerPressureData) -> wp.float32:
    kh = data.shape_kh[shape_idx]
    if signed_depth >= 0.0:
        return -kh * signed_depth
    depth = -signed_depth
    return kh * data.depth_ref_m * wp.pow(depth / data.depth_ref_m, data.exponent)


def test_custom_pressure_func_matches_default_linear(test, device):
    """User-supplied linear ``pressure_func`` must match the built-in default within 1%."""
    model, state, upper_body, rest_z = _build_cube_cube_scene(device)

    pressure_data = _LinearPressureData()
    pressure_data.shape_kh = model.shape_material_kh

    cfg_default = HydroelasticSDF.Config(
        output_contact_surface=True,
        reduce_contacts=False,
        anchor_contact=False,
    )
    cfg_callback = HydroelasticSDF.Config(
        output_contact_surface=True,
        reduce_contacts=False,
        anchor_contact=False,
        pressure_func=_linear_pressure,
        pressure_data=pressure_data,
    )
    (pipe_default, contacts_default), (pipe_callback, contacts_callback) = _make_pipelines(
        model, [cfg_default, cfg_callback], [50000, 50000]
    )

    for pen in [1e-4, 1e-3, 1e-2]:
        upper_z = rest_z - pen
        wp.launch(_set_body_z_kernel, dim=1, inputs=[state.body_q, upper_body, upper_z], device=device)

        pipe_default.collide(state, contacts_default)
        pipe_callback.collide(state, contacts_callback)

        f_default = _compute_net_force(contacts_default, model, state)
        f_callback = _compute_net_force(contacts_callback, model, state)

        test.assertGreater(abs(f_default[2]), 0.0, f"pen={pen}: default Fz should be nonzero")
        rel_z = abs(f_callback[2] - f_default[2]) / abs(f_default[2])
        test.assertLess(
            rel_z,
            0.01,
            f"pen={pen}: Fz mismatch {rel_z * 100:.2f}% (callback={f_callback[2]:.4f}, default={f_default[2]:.4f})",
        )

        for axis, label in [(0, "Fx"), (1, "Fy")]:
            abs_diff = abs(f_callback[axis] - f_default[axis])
            test.assertLess(
                abs_diff / abs(f_default[2]),
                0.01,
                f"pen={pen}: {label} diff {abs_diff:.4f} > 1% of |Fz| {abs(f_default[2]):.4f}",
            )


def test_custom_pressure_func_matches_default_linear_with_stiffness_ratio(test, device):
    """Exponent-1 power pressure must match the default for unequal stiffnesses."""
    model, state, upper_body, rest_z = _build_cube_cube_scene(device, kh_lower=1e9, kh_upper=1e10)

    pressure_data = _PowerPressureData()
    pressure_data.shape_kh = model.shape_material_kh
    pressure_data.depth_ref_m = 1.0e-3
    pressure_data.exponent = 1.0

    cfg_default = HydroelasticSDF.Config(
        output_contact_surface=True,
        reduce_contacts=False,
        anchor_contact=False,
    )
    cfg_callback = HydroelasticSDF.Config(
        output_contact_surface=True,
        reduce_contacts=False,
        anchor_contact=False,
        pressure_func=_power_pressure,
        pressure_data=pressure_data,
    )
    (pipe_default, contacts_default), (pipe_callback, contacts_callback) = _make_pipelines(
        model, [cfg_default, cfg_callback], [50000, 50000]
    )

    for pen in [1e-4, 5e-4, 1e-3]:
        upper_z = rest_z - pen
        wp.launch(_set_body_z_kernel, dim=1, inputs=[state.body_q, upper_body, upper_z], device=device)

        pipe_default.collide(state, contacts_default)
        pipe_callback.collide(state, contacts_callback)

        f_default = _compute_net_force(contacts_default, model, state)
        f_callback = _compute_net_force(contacts_callback, model, state)

        test.assertGreater(abs(f_default[2]), 0.0, f"pen={pen}: default Fz should be nonzero")
        rel_z = abs(f_callback[2] - f_default[2]) / abs(f_default[2])
        test.assertLess(
            rel_z,
            0.01,
            f"pen={pen}: unequal-kh Fz mismatch {rel_z * 100:.2f}% "
            f"(callback={f_callback[2]:.4f}, default={f_default[2]:.4f})",
        )

        for axis, label in [(0, "Fx"), (1, "Fy")]:
            abs_diff = abs(f_callback[axis] - f_default[axis])
            test.assertLess(
                abs_diff / abs(f_default[2]),
                0.01,
                f"pen={pen}: unequal-kh {label} diff {abs_diff:.4f} > 1% of |Fz| {abs(f_default[2]):.4f}",
            )


# Cubic pressure law for non-linear regression tests:
# ``p = kh * (-d)^3``. Sign-preserving (cube of pen has same sign as pen) and
# monotone non-increasing in signed_depth, satisfying the iso-surface
# precondition. Per-face force becomes ``area * kh * (-d)^3``; for the cube-
# cube scene where contact area is approximately constant in depth, total Fz
# scales as ``|d|^3``.
@wp.struct
class _CubicPressureData:
    shape_kh: wp.array[wp.float32]


@wp.func
def _cubic_pressure(signed_depth: wp.float32, shape_idx: wp.int32, data: _CubicPressureData) -> wp.float32:
    pen = -signed_depth  # positive when penetrating
    return data.shape_kh[shape_idx] * pen * pen * pen


def test_custom_pressure_func_force_scales_with_pressure_law(test, device):
    """Cubic pressure law must produce a steeper Fz(depth) curve than linear.

    The contact area in a cube-on-cube scene is itself depth-dependent, so the
    absolute force-vs-depth exponent is geometry-coupled. To isolate the
    *pressure-law* contribution, this test compares the ratio ``F(2d)/F(d)``
    under linear and cubic laws on the same geometry: the area scaling cancels,
    leaving only the pressure-law factor (2x for linear, 8x for cubic). The
    ratio-of-ratios should equal 4 regardless of how area scales with depth.
    """
    model, state, upper_body, rest_z = _build_cube_cube_scene(device)

    cubic_data = _CubicPressureData()
    cubic_data.shape_kh = model.shape_material_kh
    linear_data = _LinearPressureData()
    linear_data.shape_kh = model.shape_material_kh

    cfg_cubic = HydroelasticSDF.Config(
        output_contact_surface=True,
        reduce_contacts=False,
        anchor_contact=False,
        pressure_func=_cubic_pressure,
        pressure_data=cubic_data,
    )
    cfg_linear = HydroelasticSDF.Config(
        output_contact_surface=True,
        reduce_contacts=False,
        anchor_contact=False,
        pressure_func=_linear_pressure,
        pressure_data=linear_data,
    )
    (pipe_c, contacts_c), (pipe_l, contacts_l) = _make_pipelines(model, [cfg_cubic, cfg_linear], [50000, 50000])

    def fz_at(pipe, contacts, pen):
        wp.launch(_set_body_z_kernel, dim=1, inputs=[state.body_q, upper_body, rest_z - pen], device=device)
        pipe.collide(state, contacts)
        return abs(_compute_net_force(contacts, model, state)[2])

    pen_d, pen_2d = 1e-3, 2e-3
    f_l_d = fz_at(pipe_l, contacts_l, pen_d)
    f_l_2d = fz_at(pipe_l, contacts_l, pen_2d)
    f_c_d = fz_at(pipe_c, contacts_c, pen_d)
    f_c_2d = fz_at(pipe_c, contacts_c, pen_2d)

    test.assertGreater(f_l_d, 0.0)
    test.assertGreater(f_c_d, 0.0)

    linear_ratio = f_l_2d / f_l_d
    cubic_ratio = f_c_2d / f_c_d

    # Linear law's F-doubling ratio should be near 2 (force grows roughly with
    # depth at constant patch area). Cubic pressure must produce a substantially
    # steeper curve — if pressure_func were ignored downstream we'd see the
    # same ratio as linear. Bounds are intentionally wide because MC vertex
    # interpolation under a non-linear law shifts vertex positions along
    # voxel edges, perturbing patch area in a depth-dependent way.
    test.assertGreater(linear_ratio, 1.5, f"linear F(2d)/F(d) = {linear_ratio:.2f}")
    test.assertLess(linear_ratio, 3.0, f"linear F(2d)/F(d) = {linear_ratio:.2f}")
    test.assertGreater(
        cubic_ratio,
        4.0 * linear_ratio,
        f"cubic ratio {cubic_ratio:.2f} vs linear {linear_ratio:.2f}: "
        f"pressure_func may not be applied to per-contact force",
    )


def test_custom_pressure_func_reduced_matches_unreduced_cubic(test, device):
    """Under a cubic pressure law, reduced and unreduced net force must still agree."""
    model, state, upper_body, rest_z = _build_cube_cube_scene(device)

    pressure_data = _CubicPressureData()
    pressure_data.shape_kh = model.shape_material_kh

    cfg_red = HydroelasticSDF.Config(
        output_contact_surface=True,
        reduce_contacts=True,
        anchor_contact=False,
        pressure_func=_cubic_pressure,
        pressure_data=pressure_data,
    )
    cfg_unr = HydroelasticSDF.Config(
        output_contact_surface=True,
        reduce_contacts=False,
        anchor_contact=False,
        pressure_func=_cubic_pressure,
        pressure_data=pressure_data,
    )
    (pipe_red, contacts_red), (pipe_unr, contacts_unr) = _make_pipelines(model, [cfg_red, cfg_unr], [500, 50000])

    for pen in [1e-3, 4e-3]:
        upper_z = rest_z - pen
        wp.launch(_set_body_z_kernel, dim=1, inputs=[state.body_q, upper_body, upper_z], device=device)
        pipe_red.collide(state, contacts_red)
        pipe_unr.collide(state, contacts_unr)

        f_red = _compute_net_force(contacts_red, model, state)
        f_unr = _compute_net_force(contacts_unr, model, state)
        test.assertGreater(abs(f_unr[2]), 0.0, f"pen={pen}: unreduced cubic Fz should be nonzero")
        rel_z = abs(f_red[2] - f_unr[2]) / abs(f_unr[2])
        test.assertLess(
            rel_z,
            0.02,
            f"pen={pen}: cubic reduced/unreduced Fz mismatch {rel_z * 100:.2f}% "
            f"(red={f_red[2]:.4f}, unr={f_unr[2]:.4f})",
        )


@wp.struct
class _DecoupledPressureData:
    coeff: wp.float32  # Pa/m, fixed — deliberately independent of shape_material_kh


@wp.func
def _decoupled_pressure(signed_depth: wp.float32, shape_idx: wp.int32, data: _DecoupledPressureData) -> wp.float32:
    # Linear in penetration but with a coefficient that does NOT read
    # shape_material_kh. Models the documented custom-pressure_func case where
    # the pressure magnitude is decoupled from the per-shape hydroelastic
    # stiffness. The direction-reliability gate must not assume otherwise.
    return -data.coeff * signed_depth


def _build_offset_cube_sphere_scene(device, kh, cube_half=0.1, sphere_radius=0.1, x_offset=0.05):
    """Cube-on-ground + sphere-on-cube offset laterally so the contact patch is
    off-center (non-trivial center of pressure and tilted normals) and the
    shape ``kh`` is configurable. Returns (model, state, sphere_body, rest_z)."""
    shape_cfg = newton.ModelBuilder.ShapeConfig(
        sdf_max_resolution=128,
        is_hydroelastic=True,
        sdf_narrow_band_range=(-0.01, 0.01),
        gap=0.01,
        kh=kh,
    )
    builder = newton.ModelBuilder()
    builder.default_shape_cfg = shape_cfg
    builder.add_ground_plane()

    cube_body = builder.add_body(
        xform=wp.transform(wp.vec3(0.0, 0.0, cube_half), wp.quat_identity()),
        label="cube",
    )
    builder.add_shape_box(body=cube_body, hx=cube_half, hy=cube_half, hz=cube_half)

    rest_z = 2 * cube_half + sphere_radius
    sphere_body = builder.add_body(
        xform=wp.transform(wp.vec3(x_offset, 0.0, rest_z), wp.quat_identity()),
        label="sphere",
    )
    builder.add_shape_sphere(body=sphere_body, radius=sphere_radius)

    model = builder.finalize(device=device)
    state = model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state)

    return model, state, sphere_body, rest_z


def test_reduction_preserves_force_at_high_kh_decoupled_pressure(test, device):
    """Reduction must preserve net force under a kh-decoupled pressure law at high kh.

    The direction-reliability gate uses a pressure-law-agnostic geometric
    depth-volume, so reduction must reproduce the unreduced aggregate force at
    any stiffness and for any pressure law. This guards against a regression to a
    pressure-scaled gate (e.g. dividing the aggregate force magnitude by
    ``shape_material_kh`` before the ``EPS_LARGE`` comparison): under a custom
    ``pressure_func`` whose magnitude does not scale with kh, a large kh would
    drive that scaled magnitude below ``EPS_LARGE`` and silently disable anchor /
    normal matching, so the reduced contacts would stop reproducing the unreduced
    force. The sphere-over-edge geometry spreads the contact normals so the
    resulting direction error is observable in the net force.
    """
    kh = 1.0e10
    model, state, sphere_body, rest_z = _build_offset_cube_sphere_scene(device, kh=kh, x_offset=0.1)
    pdata = _DecoupledPressureData()
    pdata.coeff = 1.0e6
    common = {"output_contact_surface": True, "pressure_func": _decoupled_pressure, "pressure_data": pdata}
    cfg_red = HydroelasticSDF.Config(
        reduce_contacts=True, anchor_contact=True, normal_matching=True, moment_matching=True, **common
    )
    cfg_unr = HydroelasticSDF.Config(reduce_contacts=False, anchor_contact=False, **common)
    (pipe_red, c_red), (pipe_unr, c_unr) = _make_pipelines(model, [cfg_red, cfg_unr], [500, 20000])

    for pen in (2e-3, 5e-3):
        wp.launch(_set_body_z_kernel, dim=1, inputs=[state.body_q, sphere_body, rest_z - pen], device=device)
        pipe_red.collide(state, c_red)
        pipe_unr.collide(state, c_unr)

        f_red = _compute_net_force(c_red, model, state)
        f_unr = _compute_net_force(c_unr, model, state)
        fz = abs(f_unr[2])
        test.assertGreater(fz, 0.0, f"pen={pen}: unreduced Fz should be nonzero")
        rel = np.linalg.norm(f_red - f_unr) / fz
        test.assertLess(
            rel,
            0.01,
            f"pen={pen}: reduced net force deviates {rel * 100:.2f}% from unreduced at kh={kh:.0e} "
            f"(red={f_red}, unr={f_unr})",
        )


def test_custom_pressure_func_requires_pressure_data(test, device):
    """Setting ``pressure_func`` without ``pressure_data`` must raise."""
    model, state, _, _ = _build_cube_cube_scene(device)
    del state

    cfg = HydroelasticSDF.Config(
        output_contact_surface=True,
        pressure_func=_linear_pressure,
        pressure_data=None,
    )
    with test.assertRaises(ValueError):
        newton.CollisionPipeline(model, sdf_hydroelastic_config=cfg)


def test_reduced_vs_unreduced_contact_moments_cube_on_cube(test, device):
    """Reduced and unreduced hydroelastic moments must agree for cube-on-cube with moment_matching."""
    model, state, upper_body, rest_z = _build_cube_cube_scene(device)

    cfg_reduced = HydroelasticSDF.Config(
        output_contact_surface=True,
        reduce_contacts=True,
        anchor_contact=True,
        moment_matching=True,
    )
    cfg_unreduced = HydroelasticSDF.Config(
        output_contact_surface=True,
        reduce_contacts=False,
        anchor_contact=False,
    )
    (pipe_red, contacts_red), (pipe_unr, contacts_unr) = _make_pipelines(
        model, [cfg_reduced, cfg_unreduced], [500, 50000]
    )

    # Filter to the lower-upper cube shape pair (shape 1=lower, shape 2=upper).
    sp = (1, 2)

    for pen in [1e-4, 1e-3, 1e-2]:
        upper_z = rest_z - pen
        wp.launch(_set_body_z_kernel, dim=1, inputs=[state.body_q, upper_body, upper_z], device=device)

        pipe_red.collide(state, contacts_red)
        pipe_unr.collide(state, contacts_unr)

        anchor = _compute_force_weighted_anchor(contacts_unr, model, state, shape_pair=sp)

        m_red = _compute_net_moment(contacts_red, model, state, anchor=anchor, shape_pair=sp)
        m_unr = _compute_net_moment(contacts_unr, model, state, anchor=anchor, shape_pair=sp)

        # Both moments should be non-negative
        test.assertGreaterEqual(m_unr, 0.0, f"pen={pen}: unreduced moment should be >= 0")

        # Moments should match within 5%
        if m_unr > 1e-6:
            rel = abs(m_red - m_unr) / m_unr
            test.assertLess(
                rel,
                0.05,
                f"pen={pen}: moment mismatch {rel * 100:.2f}% (reduced={m_red:.6f}, unreduced={m_unr:.6f})",
            )


def test_translational_friction_invariance(test, device):
    """Total lateral friction capacity must be preserved when moment_matching is enabled."""
    model, state, sphere_body, rest_z = _build_cube_sphere_scene(device)

    cfg_moment = HydroelasticSDF.Config(
        output_contact_surface=True,
        reduce_contacts=True,
        anchor_contact=True,
        moment_matching=True,
    )
    cfg_no_moment = HydroelasticSDF.Config(
        output_contact_surface=True,
        reduce_contacts=True,
        anchor_contact=True,
        moment_matching=False,
    )
    (pipe_moment, contacts_moment), (pipe_no_moment, contacts_no_moment) = _make_pipelines(
        model, [cfg_moment, cfg_no_moment]
    )

    for pen in [1e-4, 1e-3, 1e-2]:
        sphere_z = rest_z - pen
        wp.launch(_set_body_z_kernel, dim=1, inputs=[state.body_q, sphere_body, sphere_z], device=device)

        pipe_moment.collide(state, contacts_moment)
        pipe_no_moment.collide(state, contacts_no_moment)

        # Filter to cube-sphere pair (shape 1=cube, shape 2=sphere).
        sp = (1, 2)
        fc_moment = _compute_total_friction_capacity(contacts_moment, model, state, shape_pair=sp)
        fc_no_moment = _compute_total_friction_capacity(contacts_no_moment, model, state, shape_pair=sp)

        # Both should have nonzero friction capacity
        test.assertGreater(fc_no_moment, 0.0, f"pen={pen}: no-moment friction capacity should be > 0")

        # Friction capacity must match within 1%
        if fc_no_moment > 1e-6:
            rel = abs(fc_moment - fc_no_moment) / fc_no_moment
            test.assertLess(
                rel,
                0.01,
                f"pen={pen}: translational friction mismatch {rel * 100:.2f}% "
                f"(moment_matching={fc_moment:.6f}, no_moment={fc_no_moment:.6f})",
            )


def test_entry_k_eff_matches_shape_harmonic_mean(test, device):
    """Validate entry_k_eff uses the pairwise harmonic-mean stiffness formula."""
    expected_k_eff = 0.5 * 1.0e10  # k_a == k_b == default kh for these shapes
    config = HydroelasticSDF.Config(
        reduce_contacts=True,
        pre_prune_contacts=False,
        buffer_fraction=1.0,
        buffer_mult_contact=2,
    )
    model, _, state_0, _, _, pipeline, _, _ = build_stacked_cubes_scene(
        device=device,
        solver_fn=solvers["xpbd"],
        shape_type=ShapeType.MESH,
        cube_half=CUBE_HALF_SMALL,
        reduce_contacts=True,
        sdf_hydroelastic_config=config,
    )
    newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)
    contacts = pipeline.contacts()
    pipeline.collide(state_0, contacts)

    hydro = pipeline.hydroelastic_sdf
    reducer = hydro.contact_reduction.reducer
    active_slots = reducer.hashtable.active_slots.numpy()
    ht_capacity = reducer.hashtable.capacity
    active_count = int(active_slots[ht_capacity])
    test.assertGreater(active_count, 0, "Expected at least one active reduction hashtable entry")

    active_indices = active_slots[:active_count]
    entry_k_eff = reducer.entry_k_eff.numpy()[active_indices]
    nonzero_k_eff = entry_k_eff[entry_k_eff > 0.0]
    test.assertGreater(len(nonzero_k_eff), 0, "Expected non-zero entry_k_eff values")
    test.assertTrue(
        np.allclose(nonzero_k_eff, expected_k_eff, rtol=1.0e-4, atol=1.0e-3),
        f"Expected entry_k_eff to match harmonic mean ({expected_k_eff:.6e})",
    )


def test_mujoco_hydroelastic_penetration_depth(test, device):
    """Test that hydroelastic penetration depth matches expectation.

    Creates 4 box pairs with different kh and area combinations:
    - Case 0: k=1e8, area=0.01 (small stiffness, small area)
    - Case 1: k=1e9, area=0.01 (large stiffness, small area)
    - Case 2: k=1e8, area=0.0225 (small stiffness, large area)
    - Case 3: k=1e9, area=0.0225 (large stiffness, large area)
    """
    # Test parameters
    box_size_lower = 0.2
    box_half_lower = box_size_lower / 2.0
    mass_lower = 1.0
    mass_upper = 0.5
    gravity = 10.0
    external_force = 20.0

    # 4 test cases: (kh, upper_box_size)
    test_cases = [
        (1e8, 0.1),
        (1e9, 0.1),
        (1e8, 0.15),
        (1e9, 0.15),
    ]

    # Inertia for lower box
    inertia_lower = (1.0 / 6.0) * mass_lower * box_size_lower * box_size_lower
    I_m_lower = wp.mat33(inertia_lower, 0.0, 0.0, 0.0, inertia_lower, 0.0, 0.0, 0.0, inertia_lower)

    builder = newton.ModelBuilder(gravity=(0.0, 0.0, -gravity))

    lower_body_indices = []
    upper_body_indices = []
    lower_shape_indices = []
    upper_shape_indices = []
    initial_upper_positions = []
    areas = []
    kh_values = []

    spacing = 0.5

    for i, (kh_val, upper_size) in enumerate(test_cases):
        upper_half = upper_size / 2.0
        area = upper_size * upper_size
        areas.append(area)
        kh_values.append(0.5 * kh_val)  # effective stiffness for two equal k shapes

        # Inertia for this upper box
        inertia_upper = (1.0 / 6.0) * mass_upper * upper_size * upper_size
        I_m_upper = wp.mat33(inertia_upper, 0.0, 0.0, 0.0, inertia_upper, 0.0, 0.0, 0.0, inertia_upper)

        shape_cfg = newton.ModelBuilder.ShapeConfig(
            sdf_max_resolution=64,
            is_hydroelastic=True,
            sdf_narrow_band_range=(-0.1, 0.1),
            gap=0.01,
            kh=kh_val,
            density=0.0,
        )

        x_pos = (i - len(test_cases) / 2) * spacing

        # Lower box
        lower_pos = wp.vec3(x_pos, 0.0, box_half_lower)
        body_lower = builder.add_body(
            xform=wp.transform(p=lower_pos, q=wp.quat_identity()),
            label=f"lower_{i}",
            mass=mass_lower,
            inertia=I_m_lower,
        )
        shape_lower = builder.add_shape_box(
            body_lower, hx=box_half_lower, hy=box_half_lower, hz=box_half_lower, cfg=shape_cfg
        )
        lower_body_indices.append(body_lower)
        lower_shape_indices.append(shape_lower)

        # Upper box
        expected_dist = box_half_lower + upper_half
        upper_z = box_half_lower + expected_dist
        upper_pos = wp.vec3(x_pos, 0.0, upper_z)
        body_upper = builder.add_body(
            xform=wp.transform(p=upper_pos, q=wp.quat_identity()),
            label=f"upper_{i}",
            mass=mass_upper,
            inertia=I_m_upper,
        )
        shape_upper = builder.add_shape_box(body_upper, hx=upper_half, hy=upper_half, hz=upper_half, cfg=shape_cfg)
        upper_body_indices.append(body_upper)
        upper_shape_indices.append(shape_upper)
        initial_upper_positions.append(np.array([x_pos, 0.0, upper_z]))

    builder.add_ground_plane()
    model = builder.finalize(device=device)

    solver = newton.solvers.SolverMuJoCo(
        model,
        use_mujoco_contacts=False,
        solver="newton",
        integrator="implicitfast",
        cone="elliptic",
        njmax=2000,
        nconmax=2000,
        iterations=20,
        ls_iterations=100,
        impratio=1000.0,
    )

    state_0 = model.state()
    state_1 = model.state()
    control = model.control()

    newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)

    sdf_config = HydroelasticSDF.Config(output_contact_surface=True, buffer_fraction=1.0)
    collision_pipeline = newton.CollisionPipeline(
        model,
        broad_phase="explicit",
        sdf_hydroelastic_config=sdf_config,
    )
    contacts = collision_pipeline.contacts()

    # Simulate for 3 seconds to reach equilibrium
    sim_dt = 1.0 / 60.0
    substeps = 10
    sim_time = 3.0
    num_frames = int(sim_time / sim_dt)
    total_steps = num_frames * substeps

    # Pre-compute forces as a Warp array
    forces_np = np.zeros(model.body_count * 6, dtype=np.float32)
    for body_idx in upper_body_indices:
        forces_np[body_idx * 6 + 2] = -external_force
    precomputed_forces = wp.array(forces_np.reshape(model.body_count, 6), dtype=wp.spatial_vector, device=device)

    for _ in range(total_steps):
        wp.copy(state_0.body_f, precomputed_forces)
        collision_pipeline.collide(state_0, contacts)
        solver.step(state_0, state_1, control, contacts, sim_dt / substeps)
        state_0, state_1 = state_1, state_0

    # Check that upper cubes are near their original positions
    body_q = state_0.body_q.numpy()
    position_tolerance = 0.001

    for i in range(len(test_cases)):
        body_idx = upper_body_indices[i]
        final_pos = body_q[body_idx, :3]
        initial_pos = initial_upper_positions[i]
        displacement = np.linalg.norm(final_pos - initial_pos)

        test.assertLess(
            displacement,
            position_tolerance,
            f"Case {i}: Upper cube moved {displacement:.4f}m from initial position, exceeds {position_tolerance}m tolerance",
        )

    # Measure penetration from contact surface depth
    contact_surface_data = (
        collision_pipeline.hydroelastic_sdf.get_contact_surface()
        if collision_pipeline.hydroelastic_sdf is not None
        else None
    )
    test.assertIsNotNone(contact_surface_data, "Hydroelastic contact surface data should be available")

    num_faces = int(contact_surface_data.face_contact_count.numpy()[0])
    test.assertGreater(num_faces, 0, "Should have face contacts")

    depths = contact_surface_data.contact_surface_depth.numpy()[:num_faces]
    shape_pairs = contact_surface_data.contact_surface_shape_pair.numpy()[:num_faces]

    # Calculate expected and measured penetration for each case
    total_force = gravity * mass_upper + external_force
    effective_mass = (mass_lower * mass_upper) / (mass_lower + mass_upper)

    for i in range(len(test_cases)):
        lower_shape = lower_shape_indices[i]
        upper_shape = upper_shape_indices[i]
        kh_val = kh_values[i]
        area = areas[i]

        # Expected: depth = F / (k_eff * A_eff) / mujoco_scaling
        effective_area = area
        expected = total_force / (kh_val * effective_area)
        expected /= effective_mass

        # Filter depths for this shape pair
        mask = ((shape_pairs[:, 0] == lower_shape) & (shape_pairs[:, 1] == upper_shape)) | (
            (shape_pairs[:, 0] == upper_shape) & (shape_pairs[:, 1] == lower_shape)
        )
        instance_depths = depths[mask]
        # Standard convention: negative depth = penetrating
        instance_depths = instance_depths[instance_depths < 0]

        test.assertGreater(len(instance_depths), 0, f"Case {i} should have penetrating contacts (negative depth)")

        # x2 because depth is distance to isosurface; use |depth| for magnitude
        measured = 2.0 * np.mean(-instance_depths)
        ratio = measured / expected

        # We expect a ratio > 1 due to non-uniform pressure distribution.
        test.assertGreater(
            ratio, 1.0, f"Case {i}: ratio {ratio:.3f} too low (measured={measured:.6f}, expected={expected:.6f})"
        )
        test.assertLess(
            ratio, 1.2, f"Case {i}: ratio {ratio:.3f} too high (measured={measured:.6f}, expected={expected:.6f})"
        )


def test_convex_mesh_hydroelastic_contacts(test, device):
    """SDF-backed convex meshes should be valid hydroelastic shapes."""
    cube_mesh = newton.Mesh.create_box(
        0.5,
        0.5,
        0.5,
        duplicate_vertices=False,
        compute_normals=False,
        compute_uvs=False,
        compute_inertia=False,
    )
    cube_mesh.build_sdf(max_resolution=32, narrow_band_range=(-0.1, 0.1), margin=0.02, device=device)

    cfg = newton.ModelBuilder.ShapeConfig(is_hydroelastic=True, gap=0.02)
    builder = newton.ModelBuilder()
    body_a = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()))
    body_b = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.9), wp.quat_identity()))
    builder.add_shape_convex_hull(body=body_a, mesh=cube_mesh, cfg=cfg)
    builder.add_shape_convex_hull(body=body_b, mesh=cube_mesh, cfg=cfg)

    model = builder.finalize(device=device)
    collision_pipeline = newton.CollisionPipeline(
        model,
        broad_phase="sap",
        rigid_contact_max=256,
        sdf_hydroelastic_config=HydroelasticSDF.Config(buffer_mult_contact=2),
    )
    contacts = collision_pipeline.contacts()
    collision_pipeline.collide(model.state(), contacts)

    test.assertIsNotNone(collision_pipeline.hydroelastic_sdf)
    test.assertGreater(int(contacts.rigid_contact_count.numpy()[0]), 0)


# --- Test class ---


class TestHydroelastic(unittest.TestCase):
    def test_mc_edge_clamp_min_validation(self):
        """``HydroelasticSDF.Config.mc_edge_clamp_min`` validates its range at construction.

        The validator runs in ``Config.__post_init__`` and is host-side only,
        so this test is device-independent and runs even on CPU-only CI.
        """
        # In-range values, including the boundaries, must construct cleanly.
        for good_value in (0.0, 0.02, 0.5):
            HydroelasticSDF.Config(mc_edge_clamp_min=good_value)

        # Out-of-range values, including NaN, must raise ``ValueError``.
        for bad_value in (-0.1, 0.51, float("nan")):
            with self.assertRaises(ValueError, msg=f"Should reject mc_edge_clamp_min={bad_value}"):
                HydroelasticSDF.Config(mc_edge_clamp_min=bad_value)

    @unittest.skip("Visual debugging - run manually to view simulation")
    def test_view_stacked_primitive_cubes(self):
        """View stacked primitive cubes simulation with hydroelastic contacts."""
        self._run_viewer_test(ShapeType.PRIMITIVE)

    @unittest.skip("Visual debugging - run manually to view simulation")
    def test_view_stacked_mesh_cubes(self):
        """View stacked mesh cubes simulation with hydroelastic contacts."""
        self._run_viewer_test(ShapeType.MESH)

    def _run_viewer_test(self, shape_type: ShapeType, solver_name: str = "xpbd", cube_half: float = CUBE_HALF_LARGE):
        device = wp.get_device("cuda:0")
        solver_fn = solvers[solver_name]

        model, solver, state_0, state_1, control, collision_pipeline, _, _ = build_stacked_cubes_scene(
            device, solver_fn, shape_type, cube_half
        )

        try:
            viewer = newton.viewer.ViewerGL()
            viewer.set_model(model)
        except Exception as e:
            self.skipTest(f"ViewerGL not available: {e}")
            return

        sim_time = 0.0
        contacts = collision_pipeline.contacts()
        collision_pipeline.collide(state_0, contacts)

        print(
            f"\nRunning {shape_type.value} cubes simulation with {solver_name} solver for {VIEWER_NUM_FRAMES} frames..."
        )
        print("Close the viewer window to stop.")

        try:
            for _frame in range(VIEWER_NUM_FRAMES):
                viewer.begin_frame(sim_time)
                viewer.log_state(state_0)
                viewer.log_contacts(contacts, state_0)
                viewer.log_hydro_contact_surface(
                    (
                        collision_pipeline.hydroelastic_sdf.get_contact_surface()
                        if collision_pipeline.hydroelastic_sdf is not None
                        else None
                    ),
                    penetrating_only=False,
                )
                viewer.end_frame()

                state_0, state_1 = simulate(
                    solver, model, state_0, state_1, control, contacts, collision_pipeline, SIM_DT, SIM_SUBSTEPS
                )

                sim_time += SIM_DT
                time.sleep(0.016)

        except KeyboardInterrupt:
            print("\nSimulation stopped by user.")


# --- Register tests ---

add_function_test(
    TestHydroelastic,
    "test_stacked_small_primitive_cubes_hydroelastic_mujoco_warp",
    test_stacked_small_primitive_cubes_hydroelastic,
    devices=cuda_devices,
    solver_fn=solvers["mujoco_warp"],
)

add_function_test(
    TestHydroelastic,
    "test_stacked_small_mesh_cubes_hydroelastic_xpbd",
    test_stacked_small_mesh_cubes_hydroelastic,
    devices=cuda_devices,
    solver_fn=solvers["xpbd"],
)

add_function_test(
    TestHydroelastic,
    "test_stacked_primitive_cubes_hydroelastic_xpbd_no_reduction",
    test_stacked_primitive_cubes_hydroelastic_no_reduction,
    devices=cuda_devices,
    solver_fn=solvers["xpbd"],
)

# Penetration depth validation test
add_function_test(
    TestHydroelastic,
    "test_mujoco_hydroelastic_penetration_depth",
    test_mujoco_hydroelastic_penetration_depth,
    devices=cuda_devices,
)

add_function_test(
    TestHydroelastic,
    "test_convex_mesh_hydroelastic_contacts",
    test_convex_mesh_hydroelastic_contacts,
    devices=cuda_devices,
    check_output=False,
)

add_function_test(
    TestHydroelastic,
    "test_buffer_fraction_no_crash",
    test_buffer_fraction_no_crash,
    devices=cuda_devices,
    check_output=False,
)

add_function_test(
    TestHydroelastic,
    "test_iso_scan_scratch_buffers_are_level_sized",
    test_iso_scan_scratch_buffers_are_level_sized,
    devices=cuda_devices,
)

add_function_test(
    TestHydroelastic,
    "test_reduce_contacts_with_pre_prune_disabled_no_crash",
    test_reduce_contacts_with_pre_prune_disabled_no_crash,
    devices=cuda_devices,
    check_output=False,
)
add_function_test(
    TestHydroelastic,
    "test_entry_k_eff_matches_shape_harmonic_mean",
    test_entry_k_eff_matches_shape_harmonic_mean,
    devices=cuda_devices,
)

add_function_test(
    TestHydroelastic,
    "test_reduced_vs_unreduced_contact_forces",
    test_reduced_vs_unreduced_contact_forces,
    devices=cuda_devices,
    check_output=False,
)

add_function_test(
    TestHydroelastic,
    "test_reduced_vs_unreduced_contact_forces_with_anchor_contact",
    test_reduced_vs_unreduced_contact_forces_with_anchor_contact,
    devices=cuda_devices,
    check_output=False,
)

add_function_test(
    TestHydroelastic,
    "test_reduced_vs_unreduced_contact_moments",
    test_reduced_vs_unreduced_contact_moments,
    devices=cuda_devices,
    check_output=False,
)


add_function_test(
    TestHydroelastic,
    "test_translational_friction_invariance",
    test_translational_friction_invariance,
    devices=cuda_devices,
    check_output=False,
)

add_function_test(
    TestHydroelastic,
    "test_reduced_vs_unreduced_contact_forces_cube_on_cube",
    test_reduced_vs_unreduced_contact_forces_cube_on_cube,
    devices=cuda_devices,
    check_output=False,
)

add_function_test(
    TestHydroelastic,
    "test_reduced_vs_unreduced_contact_moments_cube_on_cube",
    test_reduced_vs_unreduced_contact_moments_cube_on_cube,
    devices=cuda_devices,
    check_output=False,
)

add_function_test(
    TestHydroelastic,
    "test_custom_pressure_func_matches_default_linear",
    test_custom_pressure_func_matches_default_linear,
    devices=cuda_devices,
    check_output=False,
)

add_function_test(
    TestHydroelastic,
    "test_custom_pressure_func_matches_default_linear_with_stiffness_ratio",
    test_custom_pressure_func_matches_default_linear_with_stiffness_ratio,
    devices=cuda_devices,
    check_output=False,
)

add_function_test(
    TestHydroelastic,
    "test_custom_pressure_func_force_scales_with_pressure_law",
    test_custom_pressure_func_force_scales_with_pressure_law,
    devices=cuda_devices,
    check_output=False,
)

add_function_test(
    TestHydroelastic,
    "test_custom_pressure_func_reduced_matches_unreduced_cubic",
    test_custom_pressure_func_reduced_matches_unreduced_cubic,
    devices=cuda_devices,
    check_output=False,
)

add_function_test(
    TestHydroelastic,
    "test_reduction_preserves_force_at_high_kh_decoupled_pressure",
    test_reduction_preserves_force_at_high_kh_decoupled_pressure,
    devices=cuda_devices,
    check_output=False,
)

add_function_test(
    TestHydroelastic,
    "test_custom_pressure_func_requires_pressure_data",
    test_custom_pressure_func_requires_pressure_data,
    devices=cuda_devices,
)


def test_no_degenerate_triangles_deep_penetration(test, device):
    """Verify marching cubes produces no zero-area triangles and fewer than 2% near-degenerate triangles under deep interpenetration.

    Two hydroelastic boxes with controlled overlap are tested at multiple
    penetration depths and stiffness ratios.  The isosurface should be free
    of degenerate (zero-area) triangles that arise from vertex collapse at
    SDF ridge boundaries.

    The edge-interpolation clamp
    (:attr:`HydroelasticSDF.Config.mc_edge_clamp_min`) is the mechanism that
    prevents these vertex collapses, so this test is only meaningful when
    ``mc_edge_clamp_min`` is non-zero.

    Args:
        test: Unittest-style assertion helper.
        device: Warp device under test.
    """
    box_half = 0.1  # 10 cm half-extent
    narrow_band = box_half * 0.2
    contact_gap = box_half * 0.2

    def make_cfg(kh):
        return newton.ModelBuilder.ShapeConfig(
            mu=0.5,
            kh=kh,
            sdf_max_resolution=64,
            is_hydroelastic=True,
            sdf_narrow_band_range=(-narrow_band, narrow_band),
            gap=contact_gap,
        )

    configs = [
        # (overlap, kh_a, kh_b, label)
        (0.05, 1e10, 1e10, "equal stiffness 25% overlap"),
        (0.10, 1e10, 1e10, "equal stiffness 50% overlap"),
        (0.15, 1e10, 1e10, "equal stiffness 75% overlap"),
        (0.19, 1e10, 1e10, "equal stiffness 95% overlap"),
        (0.10, 1e10, 1e8, "asymmetric stiffness 50% overlap"),
    ]

    for overlap, kh_a, kh_b, label in configs:
        builder = newton.ModelBuilder()
        body_a = builder.add_body(
            xform=wp.transform(wp.vec3(0.0, 0.0, box_half), wp.quat_identity()),
        )
        builder.add_shape_box(body=body_a, hx=box_half, hy=box_half, hz=box_half, cfg=make_cfg(kh_a))

        z_b = box_half + 2.0 * box_half - overlap
        body_b = builder.add_body(
            xform=wp.transform(wp.vec3(0.0, 0.0, z_b), wp.quat_identity()),
        )
        builder.add_shape_box(body=body_b, hx=box_half, hy=box_half, hz=box_half, cfg=make_cfg(kh_b))

        model = builder.finalize(device=device)
        state = model.state()
        newton.eval_fk(model, model.joint_q, model.joint_qd, state)

        hydro_config = HydroelasticSDF.Config(
            output_contact_surface=True,
            reduce_contacts=False,
            buffer_mult_iso=4,
            buffer_mult_contact=4,
            mc_edge_clamp_min=0.02,
        )
        collision_pipeline = newton.CollisionPipeline(
            model,
            rigid_contact_max=100000,
            broad_phase="explicit",
            sdf_hydroelastic_config=hydro_config,
        )
        contacts = collision_pipeline.contacts()
        collision_pipeline.collide(state, contacts)

        cs = collision_pipeline.hydroelastic_sdf.get_contact_surface()
        test.assertIsNotNone(cs, f"[{label}] Expected contact surface")

        num_faces = int(cs.face_contact_count.numpy()[0])
        test.assertGreater(num_faces, 0, f"[{label}] Expected non-zero face count")

        vertices = cs.contact_surface_point.numpy()
        v = vertices[: num_faces * 3].reshape(num_faces, 3, 3)
        e1 = v[:, 1] - v[:, 0]
        e2 = v[:, 2] - v[:, 0]
        areas = 0.5 * np.linalg.norm(np.cross(e1, e2), axis=1)

        num_zero = int((areas < 1e-20).sum())
        test.assertEqual(
            num_zero,
            0,
            f"[{label}] Found {num_zero}/{num_faces} zero-area triangles ({num_zero / num_faces * 100:.1f}%)",
        )

        median_area = np.median(areas)
        num_degen = int((areas < 0.01 * median_area).sum())
        degen_pct = num_degen / num_faces * 100
        test.assertLess(
            degen_pct,
            2.0,
            f"[{label}] {degen_pct:.1f}% degenerate triangles (< 1% median area); expected < 2%",
        )


add_function_test(
    TestHydroelastic,
    "test_no_degenerate_triangles_deep_penetration",
    test_no_degenerate_triangles_deep_penetration,
    devices=cuda_devices,
    check_output=False,
)


def _build_two_box_hydro_pipeline(device, mc_edge_clamp_min: float):
    """Build a deeply-overlapping two-box hydroelastic scene and return the live pipeline.

    The pipeline (and its model) are kept alive by the caller so that the
    Warp arrays referenced by the contact surface remain valid until
    ``.numpy()`` reads have completed.
    """
    box_half = 0.1
    narrow_band = box_half * 0.2
    contact_gap = box_half * 0.2

    cfg = newton.ModelBuilder.ShapeConfig(
        mu=0.5,
        kh=1e10,
        sdf_max_resolution=64,
        is_hydroelastic=True,
        sdf_narrow_band_range=(-narrow_band, narrow_band),
        gap=contact_gap,
    )
    builder = newton.ModelBuilder()
    body_a = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, box_half), wp.quat_identity()))
    builder.add_shape_box(body=body_a, hx=box_half, hy=box_half, hz=box_half, cfg=cfg)
    overlap = 0.10
    z_b = box_half + 2.0 * box_half - overlap
    body_b = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, z_b), wp.quat_identity()))
    builder.add_shape_box(body=body_b, hx=box_half, hy=box_half, hz=box_half, cfg=cfg)
    model = builder.finalize(device=device)
    state = model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state)

    hydro_config = HydroelasticSDF.Config(
        output_contact_surface=True,
        reduce_contacts=False,
        buffer_mult_iso=4,
        buffer_mult_contact=4,
        mc_edge_clamp_min=mc_edge_clamp_min,
    )
    pipeline = newton.CollisionPipeline(
        model,
        rigid_contact_max=100000,
        broad_phase="explicit",
        sdf_hydroelastic_config=hydro_config,
    )
    contacts = pipeline.contacts()
    pipeline.collide(state, contacts)
    return pipeline, model


def _contact_surface_aggregates(cs):
    """Return order-invariant scalar aggregates summarizing a contact surface.

    Returns ``(face_count, total_triangle_area, sum_of_vertex_norms)``.  All
    three are commutative reductions, so they are insensitive to the
    ``wp.atomic_add`` ordering the contact writer uses to assign output
    slots; comparing them across configs avoids false negatives from
    atomic-ordering noise without depending on bit-exact reproducibility.
    """
    n = int(cs.face_contact_count.numpy()[0])
    if n == 0:
        return 0, 0.0, 0.0
    verts = cs.contact_surface_point.numpy()[: n * 3].astype(np.float64).reshape(-1, 3)
    e1 = verts[1::3] - verts[0::3]
    e2 = verts[2::3] - verts[0::3]
    total_area = 0.5 * float(np.linalg.norm(np.cross(e1, e2), axis=1).sum())
    vertex_norm_sum = float(np.linalg.norm(verts, axis=1).sum())
    return n, total_area, vertex_norm_sum


def test_mc_edge_clamp_min_changes_contact_surface(test, device):
    """Verify ``mc_edge_clamp_min`` actually flows through to vertex placement.

    Builds the same two-box scene with ``mc_edge_clamp_min=0.02`` and with
    ``mc_edge_clamp_min=0.0`` and asserts that at least one of three
    order-invariant scalar aggregates (face count, total triangle area, sum
    of vertex norms) differs by more than a relative tolerance.  A kernel
    that ignored the parameter would produce identical aggregates and fail
    the test.
    """
    pipe_clamped, _model_clamped = _build_two_box_hydro_pipeline(device, mc_edge_clamp_min=0.02)
    pipe_unclamped, _model_unclamped = _build_two_box_hydro_pipeline(device, mc_edge_clamp_min=0.0)

    n_c, area_c, norm_c = _contact_surface_aggregates(pipe_clamped.hydroelastic_sdf.get_contact_surface())
    n_u, area_u, norm_u = _contact_surface_aggregates(pipe_unclamped.hydroelastic_sdf.get_contact_surface())

    test.assertGreater(n_c, 0, "Expected non-empty contact surface for the clamped build")
    test.assertGreater(n_u, 0, "Expected non-empty contact surface for the unclamped build")

    rel_tol = 1e-3
    differs = (
        n_c != n_u
        or abs(area_c - area_u) / max(area_c, area_u, 1e-12) > rel_tol
        or abs(norm_c - norm_u) / max(norm_c, norm_u, 1e-12) > rel_tol
    )
    test.assertTrue(
        differs,
        f"mc_edge_clamp_min did not change the contact surface: "
        f"n=({n_c},{n_u}) area=({area_c:.6f},{area_u:.6f}) "
        f"norm_sum=({norm_c:.6f},{norm_u:.6f})",
    )


add_function_test(
    TestHydroelastic,
    "test_mc_edge_clamp_min_changes_contact_surface",
    test_mc_edge_clamp_min_changes_contact_surface,
    devices=cuda_devices,
    check_output=False,
)


def test_hydroelastic_mesh_empty_sdf_raises_value_error(test, device):
    mesh = newton.Mesh.create_box(
        0.1,
        0.1,
        0.1,
        duplicate_vertices=False,
        compute_normals=False,
        compute_uvs=False,
        compute_inertia=False,
    )
    mesh.sdf = newton.SDF.create_from_data()

    cfg = newton.ModelBuilder.ShapeConfig(is_hydroelastic=True)
    builder = newton.ModelBuilder()
    body_a = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()))
    body_b = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.1), wp.quat_identity()))
    builder.add_shape_mesh(body=body_a, mesh=mesh, cfg=cfg)
    builder.add_shape_mesh(body=body_b, mesh=mesh, cfg=cfg)
    model = builder.finalize(device=device)

    with test.assertRaisesRegex(ValueError, "requires texture SDF data"):
        newton.CollisionPipeline(model, broad_phase="explicit")


add_function_test(
    TestHydroelastic,
    "test_hydroelastic_mesh_empty_sdf_raises_value_error",
    test_hydroelastic_mesh_empty_sdf_raises_value_error,
    devices=cuda_devices,
    check_output=False,
)


def test_deep_penetration_contact_surface_has_no_central_hole(test, device):
    """Regression test for newton-physics/newton#2611.

    Two hydroelastic boxes are overlapped by an amount that is much larger
    than the SDF narrow band.  Before the fix, the broadphase skipped any
    subgrid whose center fell deeper than the narrow band, so the
    contact surface formed a thin annulus around the box perimeter with
    no triangles in the central region (visible in the issue images as a
    "center hole" in the contact patch).  The fix visits every subgrid
    arithmetically; the central region of the patch must now be
    populated.

    The scene mirrors the minimal repro from the issue: two 20 cm boxes,
    10 cm overlap (5x the 20 mm narrow band), ``kh=1e10``,
    ``sdf_max_resolution=64``, ``reduce_contacts=False``.

    The assertion is targeted at the *symptom* described in the issue —
    the contact patch is annular, with no centroids near the center of
    the overlap region.  A simple total-area check is not enough: a
    thick perimeter ring could still pass an area threshold without
    filling the middle, which is exactly what the bug looked like.
    """
    box_half = 0.10  # 20 cm box -> 10 cm half-extent (issue #2611)
    narrow_band = 0.02  # 20 mm narrow band
    overlap = 0.10  # 10 cm overlap == 5x narrow band
    contact_gap = 0.02

    cfg = newton.ModelBuilder.ShapeConfig(
        mu=0.5,
        kh=1e10,
        sdf_max_resolution=64,
        is_hydroelastic=True,
        sdf_narrow_band_range=(-narrow_band, narrow_band),
        gap=contact_gap,
    )
    builder = newton.ModelBuilder()
    body_a = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, box_half), wp.quat_identity()))
    builder.add_shape_box(body=body_a, hx=box_half, hy=box_half, hz=box_half, cfg=cfg)
    z_b = box_half + 2.0 * box_half - overlap
    body_b = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, z_b), wp.quat_identity()))
    builder.add_shape_box(body=body_b, hx=box_half, hy=box_half, hz=box_half, cfg=cfg)

    model = builder.finalize(device=device)
    state = model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state)

    hydro_config = HydroelasticSDF.Config(
        output_contact_surface=True,
        reduce_contacts=False,
        buffer_mult_iso=4,
        buffer_mult_contact=4,
    )
    pipeline = newton.CollisionPipeline(
        model,
        rigid_contact_max=200000,
        broad_phase="explicit",
        sdf_hydroelastic_config=hydro_config,
    )
    contacts = pipeline.contacts()
    pipeline.collide(state, contacts)

    cs = pipeline.hydroelastic_sdf.get_contact_surface()
    test.assertIsNotNone(cs, "Expected a contact surface for deeply overlapping hydroelastic boxes")

    num_faces = int(cs.face_contact_count.numpy()[0])
    test.assertGreater(num_faces, 0, "Expected a non-empty contact surface")

    verts = cs.contact_surface_point.numpy()[: num_faces * 3].astype(np.float64).reshape(num_faces, 3, 3)
    centroids = verts.mean(axis=1)  # (num_faces, 3) world-space face centroids

    # The boxes are stacked on Z, so the *pressure-equilibrium* plane
    # (where the hydroelastic iso-surface should pass through the
    # center of the overlap volume) sits at z = mid-overlap.  Look for
    # face centroids in a thin slab around that mid plane, then require
    # that some of them fall in the *central XY quarter* of the face
    # (|x|,|y| <= box_half / 2).  The issue's debug sweep used exactly
    # this "centroid-in-central-region" coverage metric and reported it
    # as ``0.00`` for this config under the bug; with the fix the
    # central XY region of the mid-z slab must be populated.
    mid_z = 2.0 * box_half - 0.5 * overlap  # midpoint between the two box centers along Z
    mid_slab_half = 0.5 * narrow_band  # ~5 mm slab around the mid plane
    in_mid_slab = np.abs(centroids[:, 2] - mid_z) <= mid_slab_half
    in_central_xy = np.maximum(np.abs(centroids[:, 0]), np.abs(centroids[:, 1])) <= 0.5 * box_half
    central_count = int((in_mid_slab & in_central_xy).sum())
    slab_count = int(in_mid_slab.sum())

    test.assertGreater(
        slab_count,
        0,
        f"No contact-surface centroids in the mid-z slab around z={mid_z:.4f} "
        f"(num_faces={num_faces}); contact surface is not reaching the "
        f"pressure-equilibrium plane.",
    )
    central_frac_of_slab = central_count / slab_count
    test.assertGreater(
        central_frac_of_slab,
        0.05,
        f"Only {central_count}/{slab_count} = {100.0 * central_frac_of_slab:.2f}% "
        f"of contact-surface centroids in the mid-z slab fall inside the "
        f"central XY quarter; the contact patch is annular with a center "
        f"hole — see newton-physics/newton#2611.",
    )


add_function_test(
    TestHydroelastic,
    "test_deep_penetration_contact_surface_has_no_central_hole",
    test_deep_penetration_contact_surface_has_no_central_hole,
    devices=cuda_devices,
    check_output=False,
)


if __name__ == "__main__":
    unittest.main(verbosity=2, failfast=True)
