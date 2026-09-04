# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import time
import unittest
from enum import Enum

import numpy as np
import warp as wp

import newton
from newton._src.geometry.contact_reduction_hydroelastic import (
    FIXED_EXP_NONE,
    SPECULATIVE_BIN_OFFSET,
    _fixed_mantissa_bits,
    _from_fixed,
    _to_fixed,
)
from newton._src.geometry.sdf_hydroelastic import (
    _MAX_DETERMINISTIC_ISO_VOXELS,
    _extract_mc_corner_pair,
    _mc_corner_offset,
    classify_hydroelastic_contact,
    pack_hydro_voxel_record,
    unpack_hydro_voxel_coords,
    vec8f,
)
from newton._src.geometry.sdf_mc import get_triangle_fraction
from newton._src.geometry.utils import _scan_scratch_size, scan_with_total
from newton.geometry import HydroelasticSDF
from newton.tests.unittest_utils import (
    add_function_test,
    get_selected_cuda_test_devices,
    get_test_devices,
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
scan_devices = get_test_devices()

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


@wp.kernel
def _test_fixed_point_extreme_exponents(
    values: wp.array[wp.float32],
    exponents: wp.array[wp.int32],
    mantissa_bits: int,
    fixed_values: wp.array[wp.int64],
    roundtrip_values: wp.array[wp.float32],
):
    """Convert sentinel and high finite pressure contributions in fixed point."""
    tid = wp.tid()
    fixed_values[tid] = _to_fixed(values[tid], exponents[tid], mantissa_bits)
    roundtrip_values[tid] = _from_fixed(fixed_values[tid], exponents[tid], mantissa_bits)


@wp.kernel
def _test_mc_corner_offsets(offsets: wp.array[wp.vec3i]):
    """Store each canonical marching-cubes corner offset."""
    corner_idx = wp.tid()
    offsets[corner_idx] = _mc_corner_offset(corner_idx)


@wp.kernel
def _test_mc_corner_pair_selection(
    values: wp.array[wp.float32],
    depths: wp.array[wp.float32],
    selected: wp.array[wp.vec2f],
):
    """Select every marching-cubes value/depth pair from register vectors."""
    corner_idx = wp.tid()
    corner_vals = vec8f(values[0], values[1], values[2], values[3], values[4], values[5], values[6], values[7])
    corner_depths = vec8f(depths[0], depths[1], depths[2], depths[3], depths[4], depths[5], depths[6], depths[7])
    selected[corner_idx] = _extract_mc_corner_pair(corner_vals, corner_depths, corner_idx)


@wp.kernel
def _test_triangle_fraction_rotations(
    depths: wp.array[wp.vec3f],
    num_inside: wp.array[wp.int32],
    fractions: wp.array[wp.float32],
):
    """Evaluate each triangle-depth rotation used for partial coverage."""
    tid = wp.tid()
    fractions[tid] = get_triangle_fraction(depths[tid], num_inside[tid])


@wp.kernel
def _test_hydro_voxel_record_roundtrip(
    coords: wp.array[wp.vec3us],
    pairs: wp.array[wp.vec2i],
    records: wp.array[wp.vec3ui],
    decoded_coords: wp.array[wp.vec3us],
    decoded_pairs: wp.array[wp.vec2i],
):
    """Pack and unpack one hydroelastic octree record."""
    tid = wp.tid()
    record = pack_hydro_voxel_record(coords[tid], tid)
    records[tid] = record
    decoded_coords[tid] = unpack_hydro_voxel_coords(record)
    decoded_pairs[tid] = pairs[wp.int32(record[2])]


def test_triangle_fraction_rotations(test, device):
    """Verify triangle coverage is invariant to the distinct vertex rotation."""
    depths_np = np.array(
        [
            [1.0, 2.0, 3.0],
            [-1.0, -2.0, -3.0],
            [-1.0, 2.0, 3.0],
            [2.0, -1.0, 3.0],
            [2.0, 3.0, -1.0],
            [1.0, -2.0, -3.0],
            [-2.0, 1.0, -3.0],
            [-2.0, -3.0, 1.0],
        ],
        dtype=np.float32,
    )
    num_inside_np = np.array([0, 3, 1, 1, 1, 2, 2, 2], dtype=np.int32)
    expected = np.array(
        [
            0.0,
            1.0,
            1.0 / 12.0,
            1.0 / 12.0,
            1.0 / 12.0,
            11.0 / 12.0,
            11.0 / 12.0,
            11.0 / 12.0,
        ]
    )

    fractions = wp.empty(len(depths_np), dtype=wp.float32, device=device)
    wp.launch(
        _test_triangle_fraction_rotations,
        dim=len(depths_np),
        inputs=[
            wp.array(depths_np, dtype=wp.vec3f, device=device),
            wp.array(num_inside_np, dtype=wp.int32, device=device),
            fractions,
        ],
        device=device,
    )
    np.testing.assert_allclose(fractions.numpy(), expected, rtol=1.0e-6, atol=0.0)


def test_hydro_voxel_record_roundtrip(test, device):
    """Verify packed hydroelastic octree records preserve coordinates and pair indices."""
    coords_np = np.array([[0, 0, 0], [1, 255, 256], [65535, 32768, 17]], dtype=np.uint16)
    pairs_np = np.array([[0, 1], [12345, 67890], [2**27 - 1, 2**27]], dtype=np.int32)
    coords = wp.array(coords_np, dtype=wp.vec3us, device=device)
    pairs = wp.array(pairs_np, dtype=wp.vec2i, device=device)
    records = wp.empty(len(coords_np), dtype=wp.vec3ui, device=device)
    decoded_coords = wp.empty_like(coords)
    decoded_pairs = wp.empty_like(pairs)
    wp.launch(
        _test_hydro_voxel_record_roundtrip,
        dim=len(coords_np),
        inputs=[coords, pairs, records, decoded_coords, decoded_pairs],
        device=device,
    )
    np.testing.assert_array_equal(decoded_coords.numpy(), coords_np)
    np.testing.assert_array_equal(decoded_pairs.numpy(), pairs_np)


def test_mc_corner_offsets_match_canonical(test, device):
    """Verify canonical marching-cubes corner offsets."""
    offsets = wp.empty(8, dtype=wp.vec3i, device=device)
    wp.launch(_test_mc_corner_offsets, dim=8, inputs=[offsets], device=device)
    expected = np.asarray(wp.MarchingCubes.CUBE_CORNER_OFFSETS, dtype=np.int32)
    np.testing.assert_array_equal(offsets.numpy(), expected)


def test_mc_corner_pair_selection(test, device):
    """Preserve every marching-cubes corner value and depth during selection."""
    values_np = np.arange(8, dtype=np.float32) + 0.25
    depths_np = -np.arange(8, dtype=np.float32) - 0.5
    selected = wp.empty(8, dtype=wp.vec2f, device=device)
    wp.launch(
        _test_mc_corner_pair_selection,
        dim=8,
        inputs=[
            wp.array(values_np, dtype=wp.float32, device=device),
            wp.array(depths_np, dtype=wp.float32, device=device),
            selected,
        ],
        device=device,
    )
    np.testing.assert_array_equal(selected.numpy(), np.column_stack((values_np, depths_np)))


def test_scan_with_total_boundaries(test, device):
    """Match hydroelastic count scans at empty, partial, clamped, and multi-chunk sizes."""
    capacity = 4099
    counts_np = (np.arange(capacity, dtype=np.int32) % 7) + 1

    for case, active_count in (
        ("empty", 0),
        ("partial", 37),
        ("clamped", capacity + 19),
        ("multi_chunk", capacity),
    ):
        with test.subTest(case=case):
            counts = wp.array(counts_np, dtype=wp.int32, device=device)
            num_elements = wp.array([active_count], dtype=wp.int32, device=device)
            fallback_prefix = wp.full(capacity, -1, dtype=wp.int32, device=device)
            scratch_prefix = wp.full(capacity, -1, dtype=wp.int32, device=device)
            fallback_total = wp.full(1, -1, dtype=wp.int32, device=device)
            scratch_total = wp.full(1, -1, dtype=wp.int32, device=device)
            scratch = wp.zeros(_scan_scratch_size(capacity, device), dtype=wp.int32, device=device)

            scan_with_total(counts, fallback_prefix, num_elements, fallback_total)
            scan_with_total(counts, scratch_prefix, num_elements, scratch_total, scratch=scratch)

            count = min(max(active_count, 0), capacity)
            expected_prefix = np.zeros(count, dtype=np.int32)
            if count > 1:
                expected_prefix[1:] = np.cumsum(counts_np[: count - 1], dtype=np.int32)
            scratch_prefix_np = scratch_prefix.numpy()[:count]
            np.testing.assert_array_equal(scratch_prefix_np, expected_prefix)
            np.testing.assert_array_equal(scratch_prefix_np, fallback_prefix.numpy()[:count])
            test.assertEqual(int(scratch_total.numpy()[0]), int(counts_np[:count].sum()))
            test.assertEqual(int(scratch_total.numpy()[0]), int(fallback_total.numpy()[0]))


# --- Helper functions ---


@wp.kernel
def _classify_hydroelastic_contacts_kernel(
    pair_separations: wp.array[wp.float32],
    gap_sum: float,
    contact_bands: wp.array[wp.int32],
):
    tid = wp.tid()
    contact_bands[tid] = classify_hydroelastic_contact(pair_separations[tid], gap_sum)


def test_hydroelastic_contact_band_boundaries(test, device):
    """Classify exact hydroelastic margin and gap boundaries."""
    pair_separations = wp.array([-0.01, 0.0, 0.05, 0.1, 0.1001], dtype=wp.float32, device=device)
    contact_bands = wp.empty(5, dtype=wp.int32, device=device)

    wp.launch(
        kernel=_classify_hydroelastic_contacts_kernel,
        dim=5,
        inputs=[pair_separations, 0.1],
        outputs=[contact_bands],
        device=device,
    )

    np.testing.assert_array_equal(contact_bands.numpy(), np.array([-1, 0, 0, 0, 1], dtype=np.int32))


def test_hydroelastic_sdf_padding_covers_margin_and_gap(test, device):
    """Accept exact and reject insufficient hydroelastic SDF padding."""
    exact_builder = newton.ModelBuilder()
    exact_builder.default_shape_cfg = newton.ModelBuilder.ShapeConfig(
        is_hydroelastic=True,
        margin=0.2,
        gap=0.1,
        sdf_max_resolution=32,
        sdf_padding=0.3,
    )
    exact_builder.add_shape_box(body=-1, hx=0.5, hy=0.5, hz=0.5)

    test.assertTrue(exact_builder._validate_shapes())

    builder = newton.ModelBuilder()
    builder.default_shape_cfg = newton.ModelBuilder.ShapeConfig(
        is_hydroelastic=True,
        margin=0.2,
        gap=0.1,
        sdf_max_resolution=32,
        sdf_padding=0.25,
    )
    builder.add_shape_box(body=-1, hx=0.5, hy=0.5, hz=0.5)

    with test.assertRaisesRegex(ValueError, r"sdf_padding >= margin \+ gap"):
        builder.finalize(device=device)


def test_hydroelastic_sdf_padding_validation_can_be_skipped(test, device):
    """Honor flags that skip hydroelastic shape-padding validation."""
    skip_options = (
        {"skip_validation_shapes": True},
        {"skip_all_validations": True},
    )
    for skip_option in skip_options:
        with test.subTest(**skip_option):
            builder = newton.ModelBuilder()
            builder.default_shape_cfg = newton.ModelBuilder.ShapeConfig(
                is_hydroelastic=True,
                margin=0.2,
                gap=0.1,
                sdf_max_resolution=32,
                sdf_padding=0.25,
            )
            builder.add_shape_box(body=-1, hx=0.5, hy=0.5, hz=0.5)

            model = builder.finalize(device=device, **skip_option)

            test.assertEqual(model.shape_count, 1)


def test_particle_only_hydroelastic_shape_ignores_sdf_padding(test, device):
    """Allow unused SDF padding when hydroelastic shape collisions are disabled."""
    builder = newton.ModelBuilder()
    builder.add_shape_box(
        body=-1,
        hx=0.5,
        hy=0.5,
        hz=0.5,
        cfg=newton.ModelBuilder.ShapeConfig(
            is_hydroelastic=True,
            has_shape_collision=False,
            margin=0.2,
            gap=0.1,
            sdf_padding=0.0,
        ),
    )

    builder.finalize(device=device)


def test_hydroelastic_attached_sdf_requires_padding_metadata(test, device):
    """Reject hydroelastic texture SDF data with unknown construction padding."""
    mesh = newton.Mesh.create_box(
        0.5,
        0.5,
        0.5,
        duplicate_vertices=False,
        compute_normals=False,
        compute_uvs=False,
        compute_inertia=False,
    )
    mesh.build_sdf(device=device, max_resolution=32, margin=0.3)
    mesh.sdf = newton.SDF.create_from_data(texture_data=mesh.sdf.texture_data)

    builder = newton.ModelBuilder()
    builder.add_shape_mesh(
        body=-1,
        mesh=mesh,
        cfg=newton.ModelBuilder.ShapeConfig(
            is_hydroelastic=True,
            margin=0.2,
            gap=0.1,
        ),
    )

    with test.assertRaisesRegex(ValueError, "unknown construction padding"):
        builder.finalize(device=device)

    skip_options = (
        {"skip_validation_shapes": True},
        {"skip_all_validations": True},
    )
    for skip_option in skip_options:
        with test.subTest(**skip_option):
            builder = newton.ModelBuilder()
            builder.add_shape_mesh(
                body=-1,
                mesh=mesh,
                cfg=newton.ModelBuilder.ShapeConfig(
                    is_hydroelastic=True,
                    margin=0.2,
                    gap=0.1,
                ),
            )

            model = builder.finalize(device=device, **skip_option)

            test.assertEqual(model.shape_count, 1)


def test_hydroelastic_attached_sdf_uses_padding_metadata(test, device):
    """Accept sufficient and reject insufficient declared SDF construction padding."""
    mesh = newton.Mesh.create_box(
        0.5,
        0.5,
        0.5,
        duplicate_vertices=False,
        compute_normals=False,
        compute_uvs=False,
        compute_inertia=False,
    )
    mesh.build_sdf(device=device, max_resolution=32, margin=0.3)
    source_sdf = mesh.sdf
    texture_data = source_sdf.texture_data

    for construction_padding, succeeds in ((0.3, True), (0.29, False)):
        with test.subTest(construction_padding=construction_padding):
            mesh.sdf = newton.SDF.create_from_data(
                texture_data=texture_data,
                construction_padding=construction_padding,
            )
            builder = newton.ModelBuilder()
            body_a = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()))
            body_b = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 1.2), wp.quat_identity()))
            cfg = newton.ModelBuilder.ShapeConfig(
                is_hydroelastic=True,
                margin=0.2,
                gap=0.1,
            )
            builder.add_shape_mesh(
                body=body_a,
                mesh=mesh,
                cfg=cfg,
            )
            builder.add_shape_mesh(body=body_b, mesh=mesh, cfg=cfg)

            if succeeds:
                model = builder.finalize(device=device)
                state = model.state()
                newton.eval_fk(model, model.joint_q, model.joint_qd, state)
                pipeline = newton.CollisionPipeline(model, broad_phase="explicit")
                contacts = pipeline.contacts()

                pipeline.collide(state, contacts)

                test.assertGreater(int(contacts.rigid_contact_count.numpy()[0]), 0)
            else:
                with test.assertRaisesRegex(ValueError, r"construction padding >= margin \+ gap"):
                    builder.finalize(device=device)


def test_sdf_construction_padding_validation(test, device):
    """Reject negative and non-finite SDF construction padding."""
    del device
    for construction_padding in (-0.1, np.nan, np.inf):
        with test.subTest(construction_padding=construction_padding):
            with test.assertRaisesRegex(ValueError, "construction_padding must be finite and >= 0"):
                newton.SDF.create_from_data(construction_padding=construction_padding)


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
    deterministic: bool = False,
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
        deterministic=deterministic,
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


def test_deterministic_hydroelastic_contacts(test, device, moment_matching=False):
    """Produce bit-identical hydroelastic contacts across repeated collision calls."""
    model, _, state, _, _, pipeline, _, _ = build_stacked_cubes_scene(
        device=device,
        solver_fn=lambda model: None,
        shape_type=ShapeType.PRIMITIVE,
        deterministic=True,
        sdf_hydroelastic_config=HydroelasticSDF.Config(
            reduce_contacts=True,
            anchor_contact=True,
            moment_matching=moment_matching,
        ),
    )
    newton.eval_fk(model, model.joint_q, model.joint_qd, state)
    contacts = pipeline.contacts()
    hydro = pipeline.hydroelastic_sdf
    test.assertIsNotNone(hydro)
    test.assertTrue(hydro.config.reduce_contacts)
    test.assertTrue(hydro.contact_reduction.deterministic)
    snapshots = []
    contact_fields = (
        "rigid_contact_point_id",
        "rigid_contact_shape0",
        "rigid_contact_shape1",
        "rigid_contact_point0",
        "rigid_contact_point1",
        "rigid_contact_offset0",
        "rigid_contact_offset1",
        "rigid_contact_normal",
        "rigid_contact_margin0",
        "rigid_contact_margin1",
        "rigid_contact_tids",
        "rigid_contact_stiffness",
        "rigid_contact_damping",
        "rigid_contact_friction",
    )

    for _ in range(5):
        pipeline.collide(state, contacts)
        count = int(contacts.rigid_contact_count.numpy()[0])
        face_count = int(hydro.contact_reduction.contact_count.numpy()[0])
        insert_failures = int(hydro.contact_reduction.reducer.ht_insert_failures.numpy()[0])
        test.assertLess(face_count, hydro.max_num_face_contacts, "Hydroelastic face-contact buffer saturated")
        test.assertEqual(insert_failures, 0, "Hydroelastic reduction hashtable insertion failed")
        test.assertLess(count, contacts.rigid_contact_max, "Rigid-contact buffer saturated")

        sort_keys = pipeline._sort_key_array.numpy()[:count]
        test.assertEqual(len(np.unique(sort_keys)), count, "Hydroelastic contact sort keys must be unique")
        snapshots.append((count, tuple(getattr(contacts, name).numpy()[:count].copy() for name in contact_fields)))

    test.assertGreater(snapshots[0][0], 0)
    for count, fields in snapshots[1:]:
        test.assertEqual(count, snapshots[0][0])
        for name, expected, actual in zip(contact_fields, snapshots[0][1], fields, strict=True):
            np.testing.assert_array_equal(actual, expected, err_msg=name)


def test_deterministic_hydroelastic_contacts_moment_matching(test, device):
    """Keep hydroelastic contacts bit-identical when moment matching is enabled."""
    test_deterministic_hydroelastic_contacts(test, device, moment_matching=True)


def test_cached_shape_sdf_data_matches_fallback(test, device):
    """Keep cached and per-frame SDF descriptor mapping bit-identical."""
    model, _, state, _, _, pipeline, _, _ = build_stacked_cubes_scene(
        device=device,
        solver_fn=lambda model: None,
        shape_type=ShapeType.PRIMITIVE,
        deterministic=True,
    )
    newton.eval_fk(model, model.joint_q, model.joint_qd, state)
    contacts = pipeline.contacts()
    test.assertTrue(pipeline._hydro_shape_sdf_data_prepared)

    fields = (
        "rigid_contact_shape0",
        "rigid_contact_shape1",
        "rigid_contact_point0",
        "rigid_contact_point1",
        "rigid_contact_normal",
        "rigid_contact_stiffness",
    )
    pipeline.collide(state, contacts)
    cached_count = int(contacts.rigid_contact_count.numpy()[0])
    cached = tuple(getattr(contacts, name).numpy()[:cached_count].copy() for name in fields)

    pipeline._hydro_shape_sdf_data_prepared = False
    pipeline.collide(state, contacts)
    fallback_count = int(contacts.rigid_contact_count.numpy()[0])
    test.assertEqual(fallback_count, cached_count)
    for name, expected in zip(fields, cached, strict=True):
        np.testing.assert_array_equal(getattr(contacts, name).numpy()[:fallback_count], expected, err_msg=name)


def test_deterministic_hydroelastic_contacts_unreduced(test, device):
    """Produce bit-identical hydroelastic contacts with contact reduction disabled.

    The unreduced path exports straight from the contact buffer, so it has to
    sort on the geometric fingerprint rather than the atomically assigned buffer
    slot, which varies between runs.
    """
    model, _, state, _, _, pipeline, _, _ = build_stacked_cubes_scene(
        device=device,
        solver_fn=lambda model: None,
        shape_type=ShapeType.PRIMITIVE,
        deterministic=True,
        reduce_contacts=False,
        sdf_hydroelastic_config=HydroelasticSDF.Config(reduce_contacts=False),
    )
    newton.eval_fk(model, model.joint_q, model.joint_qd, state)
    contacts = pipeline.contacts()
    test.assertFalse(pipeline.hydroelastic_sdf.config.reduce_contacts)

    snapshots = []
    for _ in range(4):
        pipeline.collide(state, contacts)
        count = int(contacts.rigid_contact_count.numpy()[0])
        snapshots.append(
            (
                count,
                contacts.rigid_contact_point0.numpy()[:count].copy(),
                contacts.rigid_contact_normal.numpy()[:count].copy(),
            )
        )

    test.assertGreater(snapshots[0][0], 0)
    for count, point0, normal in snapshots[1:]:
        test.assertEqual(count, snapshots[0][0])
        np.testing.assert_array_equal(point0, snapshots[0][1], err_msg="rigid_contact_point0")
        np.testing.assert_array_equal(normal, snapshots[0][2], err_msg="rigid_contact_normal")


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
    depth = np.einsum("ij,ij->i", p0w - p1w, -normals)
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


def _make_pipelines(model, configs, rigid_contact_maxes=None, deterministic=False):
    """Create collision pipelines and contacts for a list of HydroelasticSDF.Configs.

    Returns list of (pipeline, contacts) tuples.
    """
    if rigid_contact_maxes is None:
        rigid_contact_maxes = [500] * len(configs)
    result = []
    for cfg, rcm in zip(configs, rigid_contact_maxes, strict=True):
        pipe = newton.CollisionPipeline(
            model, rigid_contact_max=rcm, sdf_hydroelastic_config=cfg, deterministic=deterministic
        )
        result.append((pipe, pipe.contacts()))
    return result


def _build_margin_gap_boxes(device, gaps=(0.03, 0.05)):
    """Build two hydroelastic boxes with asymmetric margins and gaps."""
    builder = newton.ModelBuilder()
    common = {
        "is_hydroelastic": True,
        "kh": 1.0e8,
        "sdf_max_resolution": 64,
        "sdf_narrow_band_range": (-0.25, 0.25),
    }
    cfg_a = newton.ModelBuilder.ShapeConfig(margin=0.05, gap=gaps[0], **common)
    cfg_b = newton.ModelBuilder.ShapeConfig(
        margin=0.07, gap=gaps[1], kh=2.0e8, **{k: v for k, v in common.items() if k != "kh"}
    )

    body_a = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()))
    body_b = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 1.16), wp.quat_identity()))
    builder.add_shape_box(body=body_a, hx=0.5, hy=0.5, hz=0.5, cfg=cfg_a)
    builder.add_shape_box(body=body_b, hx=0.5, hy=0.5, hz=0.5, cfg=cfg_b)

    model = builder.finalize(device=device)
    state = model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state)
    return model, state, body_b


def _get_contact_distances(contacts, model, state):
    """Return public contact distances reconstructed in world space."""
    count = int(contacts.rigid_contact_count.numpy()[0])
    if count == 0:
        return np.empty(0)

    shape0 = contacts.rigid_contact_shape0.numpy()[:count]
    shape1 = contacts.rigid_contact_shape1.numpy()[:count]
    shape_body = model.shape_body.numpy()
    body_q = state.body_q.numpy()
    point0 = contacts.rigid_contact_point0.numpy()[:count]
    point1 = contacts.rigid_contact_point1.numpy()[:count]
    normal = contacts.rigid_contact_normal.numpy()[:count]
    body0 = shape_body[shape0]
    body1 = shape_body[shape1]
    offset0 = np.where((body0 != -1)[:, None], body_q[np.maximum(body0, 0), :3], 0.0)
    offset1 = np.where((body1 != -1)[:, None], body_q[np.maximum(body1, 0), :3], 0.0)
    return np.einsum("ij,ij->i", point1 + offset1 - point0 - offset0, normal)


def test_hydroelastic_pre_prune_writes_contact_fingerprints(test, device):
    """Write stable fingerprints for every pre-pruned hydroelastic face."""
    model, state, _ = _build_margin_gap_boxes(device)
    pipeline = newton.CollisionPipeline(
        model,
        broad_phase="explicit",
        rigid_contact_max=20000,
        sdf_hydroelastic_config=HydroelasticSDF.Config(
            reduce_contacts=True,
            pre_prune_contacts=True,
            buffer_fraction=1.0,
        ),
    )
    contacts = pipeline.contacts()
    reducer = pipeline.hydroelastic_sdf.contact_reduction.reducer
    unwritten_fingerprint = -1
    reducer.contact_fingerprints.fill_(unwritten_fingerprint)

    pipeline.collide(state, contacts)

    face_count = int(reducer.contact_count.numpy()[0])
    test.assertGreater(face_count, 0)
    fingerprints = reducer.contact_fingerprints.numpy()[1 : face_count + 1]
    test.assertFalse(np.any(fingerprints == unwritten_fingerprint))
    test.assertEqual(len(np.unique(fingerprints)), face_count)


def test_deterministic_hydroelastic_speculative_contacts(test, device):
    """Keep speculative contacts deterministic in their separate key range."""
    model, state, _ = _build_margin_gap_boxes(device)
    pipeline = newton.CollisionPipeline(
        model,
        broad_phase="explicit",
        rigid_contact_max=20000,
        sdf_hydroelastic_config=HydroelasticSDF.Config(
            reduce_contacts=True,
            buffer_fraction=1.0,
        ),
        deterministic=True,
    )
    contacts = pipeline.contacts()
    reducer = pipeline.hydroelastic_sdf.contact_reduction.reducer
    contact_fields = (
        "rigid_contact_point_id",
        "rigid_contact_shape0",
        "rigid_contact_shape1",
        "rigid_contact_point0",
        "rigid_contact_point1",
        "rigid_contact_offset0",
        "rigid_contact_offset1",
        "rigid_contact_normal",
        "rigid_contact_margin0",
        "rigid_contact_margin1",
        "rigid_contact_tids",
        "rigid_contact_stiffness",
        "rigid_contact_damping",
        "rigid_contact_friction",
    )

    snapshots = []
    for _ in range(4):
        pipeline.collide(state, contacts)
        count = int(contacts.rigid_contact_count.numpy()[0])
        test.assertGreater(count, 0)
        test.assertTrue(np.all(_get_contact_distances(contacts, model, state) >= 0.0))
        test.assertTrue(np.all(contacts.rigid_contact_stiffness.numpy()[:count] > 0.0))
        test.assertEqual(int(reducer.ht_insert_failures.numpy()[0]), 0)

        active_slots = reducer.hashtable.active_slots.numpy()
        active_count = int(active_slots[reducer.hashtable.capacity])
        active_keys = reducer.hashtable.keys.numpy()[active_slots[:active_count]]
        bin_ids = (active_keys >> np.uint64(55)) & np.uint64(0xFF)
        test.assertTrue(np.all(bin_ids >= SPECULATIVE_BIN_OFFSET))

        snapshots.append((count, tuple(getattr(contacts, name).numpy()[:count].copy() for name in contact_fields)))

    for count, fields in snapshots[1:]:
        test.assertEqual(count, snapshots[0][0])
        for name, expected, actual in zip(contact_fields, snapshots[0][1], fields, strict=True):
            np.testing.assert_array_equal(actual, expected, err_msg=name)


def test_hydroelastic_margin_gap_bands(test, device, reduce_contacts):
    """Generate penetrating, speculative, and absent hydroelastic contacts."""
    model, state, body_b = _build_margin_gap_boxes(device)
    config = HydroelasticSDF.Config(
        reduce_contacts=reduce_contacts,
        pre_prune_contacts=reduce_contacts,
        output_contact_surface=True,
        buffer_fraction=1.0,
    )
    pipeline = newton.CollisionPipeline(
        model,
        broad_phase="explicit",
        rigid_contact_max=20000,
        sdf_hydroelastic_config=config,
    )
    contacts = pipeline.contacts()

    margin_sum = 0.12
    gap_sum = 0.08
    voxel_size = max(model._texture_sdf_data.numpy()["voxel_size"][0])
    tolerance = 2.0 * voxel_size

    cases = (
        (0.08, -0.04, True),
        (margin_sum, 0.0, True),
        (0.16, 0.04, True),
        (0.24, 0.12, False),
    )
    for real_surface_separation, expected_distance, expect_contacts in cases:
        wp.launch(
            kernel=_set_body_z_kernel,
            dim=1,
            inputs=[state.body_q, body_b, 1.0 + real_surface_separation],
            device=device,
        )
        pipeline.collide(state, contacts)
        distances = _get_contact_distances(contacts, model, state)

        if not expect_contacts:
            test.assertEqual(len(distances), 0)
            continue

        test.assertGreater(
            len(distances),
            0,
            f"Expected contacts at real surface separation {real_surface_separation}",
        )
        test.assertTrue(
            abs(distances.min() - expected_distance) <= tolerance,
            f"surface separation {real_surface_separation}: expected {expected_distance} +/- {tolerance}, "
            f"got [{distances.min()}, {distances.max()}]",
        )
        test.assertTrue(np.all(distances <= gap_sum + tolerance))
        stiffness = contacts.rigid_contact_stiffness.numpy()[: len(distances)]
        test.assertTrue(np.all(stiffness > 0.0))
        if expected_distance >= 0.0:
            test.assertTrue(np.all(distances >= -tolerance))
            if expected_distance > tolerance:
                test.assertTrue(
                    np.all(distances >= 0.0),
                    "Speculative contacts must not move into the penetrating margin region.",
                )
            if reduce_contacts and expected_distance > tolerance:
                reducer = pipeline.hydroelastic_sdf.contact_reduction.reducer
                active_slots = reducer.hashtable.active_slots.numpy()
                active_count = int(active_slots[reducer.hashtable.capacity])
                active_keys = reducer.hashtable.keys.numpy()[active_slots[:active_count]]
                bin_ids = (active_keys >> np.uint64(55)) & np.uint64(0xFF)
                test.assertTrue(np.all(bin_ids >= 128), "Speculative contacts must use disjoint reduction keys.")


def test_hydroelastic_zero_gap_omits_speculative_contacts(test, device, reduce_contacts):
    """Omit positive-separation hydroelastic contacts when both gaps are zero."""
    model, state, body_b = _build_margin_gap_boxes(device, gaps=(0.0, 0.0))
    pipeline = newton.CollisionPipeline(
        model,
        broad_phase="explicit",
        rigid_contact_max=20000,
        sdf_hydroelastic_config=HydroelasticSDF.Config(
            reduce_contacts=reduce_contacts,
            pre_prune_contacts=reduce_contacts,
            buffer_fraction=1.0,
        ),
    )
    contacts = pipeline.contacts()

    # The real surfaces are 0.16 m apart, while the margin-inflated surfaces
    # are 0.04 m apart. A nonzero gap would generate speculative contacts here.
    wp.launch(
        kernel=_set_body_z_kernel,
        dim=1,
        inputs=[state.body_q, body_b, 1.16],
        device=device,
    )
    pipeline.collide(state, contacts)
    test.assertEqual(int(contacts.rigid_contact_count.numpy()[0]), 0)

    # Zero gap must not disable normal penetrating hydroelastic contacts.
    wp.launch(
        kernel=_set_body_z_kernel,
        dim=1,
        inputs=[state.body_q, body_b, 1.08],
        device=device,
    )
    pipeline.collide(state, contacts)
    distances = _get_contact_distances(contacts, model, state)
    test.assertGreater(len(distances), 0)
    test.assertTrue(np.all(distances < 0.0))


def test_hydroelastic_margin_contact_area_is_deprecated(test, device, reduce_contacts):
    """Preserve and warn about a deprecated margin contact area override."""
    model, state, _ = _build_margin_gap_boxes(device)
    with test.assertWarnsRegex(DeprecationWarning, "margin_contact_area.*deprecated"):
        newton.CollisionPipeline(
            model,
            broad_phase="explicit",
            rigid_contact_max=20000,
            sdf_hydroelastic_config=HydroelasticSDF.Config(
                margin_contact_area=1.0e-2,
                reduce_contacts=reduce_contacts,
                pre_prune_contacts=reduce_contacts,
                buffer_fraction=1.0,
            ),
        )

    margin_contact_area = 0.02
    with test.assertWarnsRegex(DeprecationWarning, "margin_contact_area.*deprecated"):
        pipeline = newton.CollisionPipeline(
            model,
            broad_phase="explicit",
            rigid_contact_max=20000,
            sdf_hydroelastic_config=HydroelasticSDF.Config(
                margin_contact_area=margin_contact_area,
                reduce_contacts=reduce_contacts,
                pre_prune_contacts=reduce_contacts,
                buffer_fraction=1.0,
            ),
        )
    contacts = pipeline.contacts()
    pipeline.collide(state, contacts)
    count = int(contacts.rigid_contact_count.numpy()[0])
    test.assertGreater(count, 0)
    k_eff = 1.0e8 * 2.0e8 / (1.0e8 + 2.0e8)
    np.testing.assert_allclose(
        contacts.rigid_contact_stiffness.numpy()[:count],
        margin_contact_area * k_eff,
        rtol=1.0e-5,
    )


def test_mujoco_warp_hydroelastic_speculative_activation(test, device):
    """Keep speculative contacts inactive until they enter the margin band."""
    model, state_0, body_b = _build_margin_gap_boxes(device)
    state_1 = model.state()
    control = model.control()
    pipeline = newton.CollisionPipeline(
        model,
        broad_phase="explicit",
        rigid_contact_max=20000,
        sdf_hydroelastic_config=HydroelasticSDF.Config(
            reduce_contacts=True,
            buffer_fraction=1.0,
        ),
    )
    contacts = pipeline.contacts()
    solver = newton.solvers.SolverMuJoCo(
        model,
        use_mujoco_contacts=False,
        solver="newton",
        nconmax=20000,
        njmax=20000,
    )

    wp.launch(
        kernel=_set_body_z_kernel,
        dim=1,
        inputs=[state_0.body_q, body_b, 1.16],
        device=device,
    )
    pipeline.collide(state_0, contacts)
    test.assertTrue(np.all(_get_contact_distances(contacts, model, state_0) >= 0.0))
    solver.step(state_0, state_1, control, contacts, 1.0 / 600.0)

    nacon = int(solver.mjw_data.nacon.numpy()[0])
    test.assertGreater(nacon, 0)
    speculative_dist = solver.mjw_data.contact.dist.numpy().reshape(-1)[:nacon]
    speculative_efc = solver.mjw_data.contact.efc_address.numpy()[:nacon]
    test.assertTrue(np.all(speculative_dist >= 0.0))
    test.assertTrue(np.all(speculative_efc == -1))

    contact_count = int(contacts.rigid_contact_count.numpy()[0])
    stiffness = contacts.rigid_contact_stiffness.numpy()[:contact_count]
    damping = contacts.rigid_contact_damping.numpy()[:contact_count]
    tid_to_cid = solver._contact_tid_to_cid.numpy()[:contact_count]
    solref = solver.mjw_data.contact.solref.numpy().reshape(-1, 2)
    solimp = solver.mjw_data.contact.solimp.numpy().reshape(-1, 5)
    for tid, cid in enumerate(tid_to_cid):
        if cid < 0:
            continue
        contact_ke = stiffness[tid] * (1.0 - solimp[cid, 1])
        if damping[tid] > 0.0:
            expected_timeconst = 2.0 / damping[tid]
            expected_dampratio = np.sqrt(1.0 / (expected_timeconst**2 * contact_ke))
        else:
            expected_timeconst = np.sqrt(1.0 / contact_ke)
            expected_dampratio = 1.0
        np.testing.assert_allclose(
            solref[cid],
            (expected_timeconst, expected_dampratio),
            rtol=1.0e-5,
        )

    # Move the bodies while retaining the generated speculative contacts. The
    # MuJoCo Warp fast path must update their separation without regenerating
    # contact geometry or material data.
    voxel_size = max(model._texture_sdf_data.numpy()["voxel_size"][0])
    boundary_tolerance = 2.0 * voxel_size
    wp.launch(
        kernel=_set_body_z_kernel,
        dim=1,
        inputs=[state_0.body_q, body_b, 1.12 + boundary_tolerance],
        device=device,
    )
    cached_distances = _get_contact_distances(contacts, model, state_0)
    test.assertTrue(np.all(cached_distances >= 0.0))
    test.assertTrue(np.any(cached_distances <= 2.0 * boundary_tolerance))
    solver.step(state_0, state_1, control, contacts, 1.0 / 600.0)
    test.assertTrue(np.all(solver.mjw_data.contact.efc_address.numpy()[:nacon] == -1))

    wp.launch(
        kernel=_set_body_z_kernel,
        dim=1,
        inputs=[state_0.body_q, body_b, 1.08],
        device=device,
    )
    test.assertTrue(np.any(_get_contact_distances(contacts, model, state_0) < 0.0))
    solver.step(state_0, state_1, control, contacts, 1.0 / 600.0)

    nacon = int(solver.mjw_data.nacon.numpy()[0])
    active_efc = solver.mjw_data.contact.efc_address.numpy()[:nacon]
    active_solref = solver.mjw_data.contact.solref.numpy().reshape(-1, 2)[:nacon]
    test.assertTrue(np.any(active_efc >= 0))
    test.assertTrue(np.all(np.isfinite(active_solref)))
    test.assertTrue(np.all(active_solref[:, 0] > 0.0))


def test_reduced_vs_unreduced_contact_forces(test, device, anchor_contact=False, deterministic=False):
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
        model, [cfg_reduced, cfg_unreduced], [500, 20000], deterministic=deterministic
    )
    if deterministic:
        test.assertLessEqual(pipe_red.hydroelastic_sdf.max_num_iso_voxels, _MAX_DETERMINISTIC_ISO_VOXELS)
        test.assertLessEqual(pipe_unr.hydroelastic_sdf.max_num_iso_voxels, _MAX_DETERMINISTIC_ISO_VOXELS)

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


def test_reduced_vs_unreduced_contact_moments(test, device, deterministic=False):
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
        model, [cfg_reduced, cfg_unreduced], [500, 20000], deterministic=deterministic
    )
    if deterministic:
        test.assertLessEqual(pipe_red.hydroelastic_sdf.max_num_iso_voxels, _MAX_DETERMINISTIC_ISO_VOXELS)
        test.assertLessEqual(pipe_unr.hydroelastic_sdf.max_num_iso_voxels, _MAX_DETERMINISTIC_ISO_VOXELS)

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


def test_reduced_vs_unreduced_contact_forces_deterministic(test, device):
    """Reduced hydroelastic forces must still match when determinism is enabled.

    Deterministic mode accumulates the aggregates that drive contact stiffness in
    int64 fixed point, so this checks that path against the unreduced reference
    rather than only against itself.
    """
    test_reduced_vs_unreduced_contact_forces(test, device, anchor_contact=True, deterministic=True)


def test_reduced_vs_unreduced_contact_moments_deterministic(test, device):
    """Reduced hydroelastic moments must still match when determinism is enabled.

    Exercises the fixed-point unreduced/reduced friction-moment accumulators,
    which deterministic mode computes in separate kernels from the default path.
    """
    test_reduced_vs_unreduced_contact_moments(test, device, deterministic=True)


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


def test_exported_margin_stiffness_matches_shape_series_combination(test, device):
    """Verify exported margin stiffness uses the pairwise series combination."""
    margin_contact_area = 0.0125
    with test.assertWarnsRegex(DeprecationWarning, "margin_contact_area.*deprecated"):
        config = HydroelasticSDF.Config(
            reduce_contacts=True,
            pre_prune_contacts=False,
            anchor_contact=False,
            margin_contact_area=margin_contact_area,
            buffer_fraction=1.0,
            buffer_mult_contact=2,
        )
    model, _, state_0, _, _, pipeline, _, _ = build_stacked_cubes_scene(
        device=device,
        solver_fn=solvers["xpbd"],
        shape_type=ShapeType.PRIMITIVE,
        cube_half=CUBE_HALF_SMALL,
        reduce_contacts=True,
        sdf_hydroelastic_config=config,
    )
    newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)
    top_body = 2
    contacts = pipeline.contacts()
    pipeline.collide(state_0, contacts)

    count = int(contacts.rigid_contact_count.numpy()[0])
    test.assertGreater(count, 0, "Expected exported hydroelastic contacts")

    shape0 = contacts.rigid_contact_shape0.numpy()[:count]
    shape1 = contacts.rigid_contact_shape1.numpy()[:count]
    shape_body = model.shape_body.numpy()
    lower_shape = int(np.flatnonzero(shape_body == top_body - 1)[0])
    top_shape = int(np.flatnonzero(shape_body == top_body)[0])
    pair_mask = ((shape0 == lower_shape) & (shape1 == top_shape)) | ((shape0 == top_shape) & (shape1 == lower_shape))
    test.assertTrue(np.any(pair_mask), "Expected contacts for the touching top-cube pair")

    point0 = contacts.rigid_contact_point0.numpy()[:count]
    point1 = contacts.rigid_contact_point1.numpy()[:count]
    normal = contacts.rigid_contact_normal.numpy()[:count]
    body_q = state_0.body_q.numpy()
    body0 = shape_body[shape0]
    body1 = shape_body[shape1]
    point0_world = point0 + np.where((body0 != -1)[:, None], body_q[np.maximum(body0, 0), :3], 0.0)
    point1_world = point1 + np.where((body1 != -1)[:, None], body_q[np.maximum(body1, 0), :3], 0.0)
    contact_distance = np.einsum("ij,ij->i", point1_world - point0_world, normal)
    margin_mask = pair_mask & (contact_distance >= 0.0)
    test.assertTrue(np.any(margin_mask), "Expected nonpenetrating margin contacts for the touching pair")

    stiffness = contacts.rigid_contact_stiffness.numpy()[:count]
    shape_kh = model.shape_material_kh.numpy()
    k_a = shape_kh[shape0[margin_mask]]
    k_b = shape_kh[shape1[margin_mask]]
    expected_stiffness = margin_contact_area * (k_a * k_b) / (k_a + k_b)
    np.testing.assert_allclose(
        stiffness[margin_mask],
        expected_stiffness,
        rtol=1.0e-5,
        atol=1.0e-3,
        err_msg="Exported margin stiffness must use the pairwise series combination",
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

        measured = np.mean(-instance_depths)
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


def _canonicalize_contact_records(records):
    """Sort exported contacts independently by identity and rounded geometry."""
    point_id, shape0, shape1, point0, point1, normal, _, _ = records
    keys = np.column_stack(
        (
            point_id,
            shape0,
            shape1,
            np.round(point0, decimals=4),
            np.round(point1, decimals=4),
            np.round(normal, decimals=4),
        )
    )
    order = np.lexsort(tuple(keys[:, column] for column in reversed(range(keys.shape[1]))))
    return tuple(values[order] for values in records)


def test_scalar_sdf_texture_hydroelastic_contacts(test, device):
    """Preserve hydroelastic contacts across paired and scalar texture storage."""

    def collide(paired_samples):
        cube_mesh = newton.Mesh.create_box(
            0.5,
            0.5,
            0.5,
            duplicate_vertices=False,
            compute_normals=False,
            compute_uvs=False,
            compute_inertia=False,
        )
        cube_mesh.build_sdf(
            max_resolution=32,
            narrow_band_range=(-0.1, 0.1),
            margin=0.02,
            paired_samples=paired_samples,
            device=device,
        )

        cfg = newton.ModelBuilder.ShapeConfig(is_hydroelastic=True, gap=0.02)
        builder = newton.ModelBuilder(sdf_texture_paired_samples=paired_samples)
        body_a = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()))
        body_b = builder.add_body(
            xform=wp.transform(
                wp.vec3(0.07, -0.04, 0.9),
                wp.quat_from_axis_angle(wp.normalize(wp.vec3(0.3, 1.0, -0.2)), 0.08),
            )
        )
        builder.add_shape_convex_hull(body=body_a, mesh=cube_mesh, cfg=cfg)
        builder.add_shape_convex_hull(body=body_b, mesh=cube_mesh, cfg=cfg)

        model = builder.finalize(device=device)
        state = model.state()
        collision_pipeline = newton.CollisionPipeline(
            model,
            broad_phase="sap",
            rigid_contact_max=256,
            sdf_hydroelastic_config=HydroelasticSDF.Config(buffer_mult_contact=2),
        )
        contacts = collision_pipeline.contacts()
        collision_pipeline.collide(state, contacts)

        count = int(contacts.rigid_contact_count.numpy()[0])
        test.assertGreater(count, 0)
        point_id = contacts.rigid_contact_point_id.numpy()[:count]
        shape0 = contacts.rigid_contact_shape0.numpy()[:count]
        shape1 = contacts.rigid_contact_shape1.numpy()[:count]
        point0 = contacts.rigid_contact_point0.numpy()[:count]
        point1 = contacts.rigid_contact_point1.numpy()[:count]
        normal = contacts.rigid_contact_normal.numpy()[:count]
        stiffness = contacts.rigid_contact_stiffness.numpy()[:count]
        body_q = state.body_q.numpy()
        shape_body = model.shape_body.numpy()
        point0_world = point0 + body_q[shape_body[shape0], :3]
        point1_world = point1 + body_q[shape_body[shape1], :3]
        penetration = np.einsum("ij,ij->i", point1_world - point0_world, normal)
        return cube_mesh.sdf._coarse_texture.num_channels, tuple(
            values.copy() for values in (point_id, shape0, shape1, point0, point1, normal, penetration, stiffness)
        )

    paired_channels, paired = collide(True)
    scalar_channels, scalar = collide(False)
    test.assertEqual(paired_channels, 2)
    test.assertEqual(scalar_channels, 1)
    test.assertEqual(len(scalar[0]), len(paired[0]))
    paired = _canonicalize_contact_records(paired)
    scalar = _canonicalize_contact_records(scalar)
    for name, paired_values, scalar_values in zip(
        ("point_id", "shape0", "shape1", "point0", "point1", "normal", "penetration", "stiffness"),
        paired,
        scalar,
        strict=True,
    ):
        if name == "point_id" or name.startswith("shape"):
            np.testing.assert_array_equal(scalar_values, paired_values, err_msg=name)
        else:
            # Margin-relative pressure evaluation adds one float32 subtraction
            # before interpolation; normals amplify that small positional delta.
            tolerance = 5.0e-5 if name == "normal" else 2.0e-5
            np.testing.assert_allclose(
                scalar_values,
                paired_values,
                rtol=tolerance,
                atol=tolerance,
                err_msg=name,
            )


def test_fixed_point_extreme_exponents(test, device):
    """Handle sentinel and high finite pressure contributions without overflow."""
    mantissa_bits = _fixed_mantissa_bits(1024)
    high_value = np.finfo(np.float32).max
    values_np = np.array([0.0, high_value, -high_value], dtype=np.float32)
    exponents_np = np.array([int(FIXED_EXP_NONE), 127, 127], dtype=np.int32)
    values = wp.array(values_np, dtype=wp.float32, device=device)
    exponents = wp.array(exponents_np, dtype=wp.int32, device=device)
    fixed_values = wp.empty(len(values_np), dtype=wp.int64, device=device)
    roundtrip_values = wp.empty(len(values_np), dtype=wp.float32, device=device)

    wp.launch(
        _test_fixed_point_extreme_exponents,
        dim=len(values_np),
        inputs=[values, exponents, mantissa_bits, fixed_values, roundtrip_values],
        device=device,
    )

    fixed_np = fixed_values.numpy()
    roundtrip_np = roundtrip_values.numpy()
    test.assertEqual(fixed_np[0], 0)
    test.assertEqual(roundtrip_np[0], 0.0)
    test.assertTrue(np.all(np.abs(fixed_np[1:]) < np.iinfo(np.int64).max))
    np.testing.assert_array_equal(roundtrip_np[1:], values_np[1:])


# --- Test class ---


class TestHydroelastic(unittest.TestCase):
    def test_fixed_point_extreme_exponents(self):
        """Handle sentinel and high finite pressure contributions without overflow."""
        test_fixed_point_extreme_exponents(self, wp.get_device("cpu"))

    def test_fixed_point_accumulator_cannot_overflow(self):
        """``_fixed_mantissa_bits`` keeps deterministic fixed-point sums inside int64.

        A contribution equal to the entry maximum scales to just under
        ``2**(bits + 1)``, because ``|x| / 2**exponent`` lies in ``[1, 2)``.  The
        worst case is every term hitting that ceiling in the same entry, so the
        chosen width must keep ``max_terms * 2**(bits + 1)`` below ``2**63``.
        This bound is host-side only, so the test runs even on CPU-only CI.
        """
        int64_max = 2**63 - 1
        for max_terms in (1, 2, 64, 7168, 28672, 1 << 20, 1835008, (1 << 24) + 1):
            bits = _fixed_mantissa_bits(max_terms)
            worst_case_sum = max_terms * 2 ** (bits + 1)
            self.assertLessEqual(worst_case_sum, int64_max, msg=f"max_terms={max_terms}, bits={bits}")
            # Must still beat float32's 24-bit significand by a wide margin.
            self.assertGreater(bits, 24, msg=f"max_terms={max_terms}")

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
    "test_hydroelastic_contact_band_boundaries",
    test_hydroelastic_contact_band_boundaries,
    devices=["cpu"],
)

add_function_test(
    TestHydroelastic,
    "test_hydroelastic_sdf_padding_covers_margin_and_gap",
    test_hydroelastic_sdf_padding_covers_margin_and_gap,
    devices=["cpu"],
)

add_function_test(
    TestHydroelastic,
    "test_hydroelastic_sdf_padding_validation_can_be_skipped",
    test_hydroelastic_sdf_padding_validation_can_be_skipped,
    devices=cuda_devices,
)

add_function_test(
    TestHydroelastic,
    "test_particle_only_hydroelastic_shape_ignores_sdf_padding",
    test_particle_only_hydroelastic_shape_ignores_sdf_padding,
    devices=["cpu"],
)

add_function_test(
    TestHydroelastic,
    "test_hydroelastic_attached_sdf_requires_padding_metadata",
    test_hydroelastic_attached_sdf_requires_padding_metadata,
    devices=cuda_devices,
)

add_function_test(
    TestHydroelastic,
    "test_hydroelastic_attached_sdf_uses_padding_metadata",
    test_hydroelastic_attached_sdf_uses_padding_metadata,
    devices=cuda_devices,
)

add_function_test(
    TestHydroelastic,
    "test_sdf_construction_padding_validation",
    test_sdf_construction_padding_validation,
    devices=["cpu"],
)

add_function_test(
    TestHydroelastic,
    "test_hydroelastic_pre_prune_writes_contact_fingerprints",
    test_hydroelastic_pre_prune_writes_contact_fingerprints,
    devices=cuda_devices,
)

add_function_test(
    TestHydroelastic,
    "test_deterministic_hydroelastic_speculative_contacts",
    test_deterministic_hydroelastic_speculative_contacts,
    devices=cuda_devices,
)

add_function_test(
    TestHydroelastic,
    "test_hydroelastic_margin_gap_bands_reduced",
    test_hydroelastic_margin_gap_bands,
    devices=cuda_devices,
    reduce_contacts=True,
)

add_function_test(
    TestHydroelastic,
    "test_hydroelastic_margin_gap_bands_unreduced",
    test_hydroelastic_margin_gap_bands,
    devices=cuda_devices,
    reduce_contacts=False,
)

add_function_test(
    TestHydroelastic,
    "test_hydroelastic_zero_gap_omits_speculative_contacts_reduced",
    test_hydroelastic_zero_gap_omits_speculative_contacts,
    devices=cuda_devices,
    reduce_contacts=True,
)

add_function_test(
    TestHydroelastic,
    "test_hydroelastic_zero_gap_omits_speculative_contacts_unreduced",
    test_hydroelastic_zero_gap_omits_speculative_contacts,
    devices=cuda_devices,
    reduce_contacts=False,
)

add_function_test(
    TestHydroelastic,
    "test_hydroelastic_margin_contact_area_is_deprecated_reduced",
    test_hydroelastic_margin_contact_area_is_deprecated,
    devices=cuda_devices,
    reduce_contacts=True,
)

add_function_test(
    TestHydroelastic,
    "test_hydroelastic_margin_contact_area_is_deprecated_unreduced",
    test_hydroelastic_margin_contact_area_is_deprecated,
    devices=cuda_devices,
    reduce_contacts=False,
)

add_function_test(
    TestHydroelastic,
    "test_mujoco_warp_hydroelastic_speculative_activation",
    test_mujoco_warp_hydroelastic_speculative_activation,
    devices=cuda_devices,
)

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
    "test_scalar_sdf_texture_hydroelastic_contacts",
    test_scalar_sdf_texture_hydroelastic_contacts,
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
    "test_deterministic_hydroelastic_contacts",
    test_deterministic_hydroelastic_contacts,
    devices=cuda_devices,
    check_output=False,
)

add_function_test(
    TestHydroelastic,
    "test_cached_shape_sdf_data_matches_fallback",
    test_cached_shape_sdf_data_matches_fallback,
    devices=cuda_devices,
    check_output=False,
)

add_function_test(
    TestHydroelastic,
    "test_fixed_point_extreme_exponents_cuda",
    test_fixed_point_extreme_exponents,
    devices=cuda_devices,
    check_output=False,
)

add_function_test(
    TestHydroelastic,
    "test_triangle_fraction_rotations",
    test_triangle_fraction_rotations,
    devices=["cpu", *cuda_devices],
    check_output=False,
)

add_function_test(
    TestHydroelastic,
    "test_hydro_voxel_record_roundtrip",
    test_hydro_voxel_record_roundtrip,
    devices=cuda_devices,
    check_output=False,
)

add_function_test(
    TestHydroelastic,
    "test_deterministic_hydroelastic_contacts_moment_matching",
    test_deterministic_hydroelastic_contacts_moment_matching,
    devices=cuda_devices,
    check_output=False,
)

add_function_test(
    TestHydroelastic,
    "test_deterministic_hydroelastic_contacts_unreduced",
    test_deterministic_hydroelastic_contacts_unreduced,
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
    "test_exported_margin_stiffness_matches_shape_series_combination",
    test_exported_margin_stiffness_matches_shape_series_combination,
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
    "test_reduced_vs_unreduced_contact_forces_deterministic",
    test_reduced_vs_unreduced_contact_forces_deterministic,
    devices=cuda_devices,
    check_output=False,
)

add_function_test(
    TestHydroelastic,
    "test_mc_corner_offsets_match_canonical",
    test_mc_corner_offsets_match_canonical,
    devices=cuda_devices,
)

add_function_test(
    TestHydroelastic,
    "test_mc_corner_pair_selection",
    test_mc_corner_pair_selection,
    devices=cuda_devices,
)

add_function_test(
    TestHydroelastic,
    "test_scan_with_total_boundaries",
    test_scan_with_total_boundaries,
    devices=scan_devices,
)

add_function_test(
    TestHydroelastic,
    "test_reduced_vs_unreduced_contact_moments_deterministic",
    test_reduced_vs_unreduced_contact_moments_deterministic,
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
    """Verify deep-penetration contact surfaces have stable face counts and are non-degenerate.

    Two hydroelastic boxes with controlled overlap are tested at multiple
    penetration depths and stiffness ratios.  The isosurface should be free
    of degenerate (zero-area) triangles that arise from vertex collapse at
    SDF ridge boundaries. The deepest-penetration case is rebuilt repeatedly
    to verify primitive SDF construction produces a stable contact-surface
    face count.

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
        # (overlap, kh_a, kh_b, repeats, label)
        (0.05, 1e10, 1e10, 1, "equal stiffness 25% overlap"),
        (0.10, 1e10, 1e10, 1, "equal stiffness 50% overlap"),
        (0.15, 1e10, 1e10, 1, "equal stiffness 75% overlap"),
        (0.19, 1e10, 1e10, 3, "equal stiffness 95% overlap"),
        (0.10, 1e10, 1e8, 1, "asymmetric stiffness 50% overlap"),
    ]

    for overlap, kh_a, kh_b, repeats, label in configs:
        face_counts = []
        for repeat in range(repeats):
            run_label = f"{label}, run {repeat + 1}/{repeats}"
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
                rigid_contact_max=200000,
                broad_phase="explicit",
                sdf_hydroelastic_config=hydro_config,
            )
            contacts = collision_pipeline.contacts()
            collision_pipeline.collide(state, contacts)

            cs = collision_pipeline.hydroelastic_sdf.get_contact_surface()
            test.assertIsNotNone(cs, f"[{run_label}] Expected contact surface")

            num_faces = int(cs.face_contact_count.numpy()[0])
            test.assertGreater(num_faces, 0, f"[{run_label}] Expected non-zero face count")
            face_counts.append(num_faces)

            vertices = cs.contact_surface_point.numpy()
            v = vertices[: num_faces * 3].reshape(num_faces, 3, 3)
            e1 = v[:, 1] - v[:, 0]
            e2 = v[:, 2] - v[:, 0]
            areas = 0.5 * np.linalg.norm(np.cross(e1, e2), axis=1)

            num_zero = int((areas < 1e-20).sum())
            test.assertEqual(
                num_zero,
                0,
                (f"[{run_label}] Found {num_zero}/{num_faces} zero-area triangles ({num_zero / num_faces * 100:.1f}%)"),
            )

            median_area = np.median(areas)
            num_degen = int((areas < 0.01 * median_area).sum())
            degen_pct = num_degen / num_faces * 100
            test.assertLess(
                degen_pct,
                2.0,
                f"[{run_label}] {degen_pct:.1f}% degenerate triangles (< 1% median area); expected < 2%",
            )

        test.assertEqual(
            len(set(face_counts)),
            1,
            f"[{label}] Contact-surface face count changed across SDF rebuilds: {face_counts}",
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


def _sanding_contact_builder(resolution, *, delta=0.0, pad_first=False):
    """Build one round-pad and spherical-workpiece hydroelastic world."""
    pad_radius = 0.0675
    pad_half_height = 0.010
    sphere_radius = 0.9

    def make_cfg(kh):
        return newton.ModelBuilder.ShapeConfig(
            kh=kh,
            is_hydroelastic=True,
            gap=0.0,
            sdf_max_resolution=resolution,
            sdf_narrow_band_range=(-0.002, 0.002),
        )

    builder = newton.ModelBuilder()

    def add_sphere():
        builder.add_shape_sphere(
            body=-1,
            xform=wp.transform(wp.vec3(0.0, 0.0, -sphere_radius), wp.quat_identity()),
            radius=sphere_radius,
            cfg=make_cfg(1.0e9),
        )

    def add_pad():
        pad_body = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, pad_half_height - delta), wp.quat_identity()))
        builder.add_shape_cylinder(
            body=pad_body,
            radius=pad_radius,
            half_height=pad_half_height,
            cfg=make_cfg(5.3e6),
        )

    if pad_first:
        add_pad()
        add_sphere()
    else:
        add_sphere()
        add_pad()
    return builder


def test_hydroelastic_replica_buffers_scale_with_traversed_grids(test, device):
    """Size replica buffers from each pair's finer traversal grid."""
    world_count = 4
    one_world = _sanding_contact_builder(64)
    builder = newton.ModelBuilder()
    builder.replicate(one_world, world_count)
    model = builder.finalize(device=device)
    pipeline = newton.CollisionPipeline(model, broad_phase="explicit")
    hydro = pipeline.hydroelastic_sdf

    # Each world contributes one pad/sphere pair. The pad's SDF is the finer of
    # the two (a 0.135 m shape at resolution 64 against a 1.8 m sphere at the
    # same resolution), so the broadphase traverses the pad's 4x4x8 = 128
    # subgrids and ignores the sphere's 512. Sizing off the shapes rather than
    # the traversed grids inflated this to 2560.
    blocks_per_pair = 128
    expected_blocks = world_count * blocks_per_pair

    test.assertEqual(len(model.shape_contact_pairs.numpy().reshape(-1, 2)), world_count)
    test.assertEqual(hydro.total_num_tiles, expected_blocks)
    # Defaults are buffer_mult_broad=1 and buffer_fraction=1.0, so the
    # broadphase buffer is exactly the traversed block count.
    test.assertEqual(hydro.max_num_blocks_broad, expected_blocks)

    # Refinement work follows a two-dimensional contact surface, so size its
    # buffers from narrow-band subgrids instead of every dense-grid block.
    test.assertLess(hydro.total_num_active_tiles, hydro.total_num_tiles)
    expected_iso_dims = tuple(mult * hydro.total_num_active_tiles for mult in (8, 32, 128, 256))
    test.assertEqual(hydro.iso_max_dims, expected_iso_dims)

    state = model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state)
    pipeline.collide(state, pipeline.contacts())
    test.assertEqual(int(hydro.block_broad_collide_count.numpy()[0]), expected_blocks)


def test_hydroelastic_pair_buffer_grid_selection(test, device):
    """Match host buffer sizing to broadphase traversal-grid selection."""
    # Shape A is the finer pad. This exercises the swap branch opposite the
    # replicated test above, where shape B is finer.
    model = _sanding_contact_builder(64, pad_first=True).finalize(device=device)
    pipeline = newton.CollisionPipeline(model, broad_phase="explicit")
    hydro = pipeline.hydroelastic_sdf
    test.assertEqual(tuple(model.shape_contact_pairs.numpy()[0]), (0, 1))
    test.assertEqual(hydro.total_num_tiles, 128)

    state = model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state)
    pipeline.collide(state, pipeline.contacts())
    test.assertEqual(int(hydro.block_broad_collide_count.numpy()[0]), 128)

    # Give two differently sized grids exactly equal voxel radii. The kernel's
    # tie rule retains shape B, so both host sizing and device traversal must
    # use B's larger dense grid.
    cfg = newton.ModelBuilder.ShapeConfig(
        kh=1.0e6,
        is_hydroelastic=True,
        gap=0.0,
        sdf_max_resolution=64,
        sdf_narrow_band_range=(-0.002, 0.002),
    )
    builder = newton.ModelBuilder()
    body = builder.add_body()
    builder.add_shape_box(body=body, hx=0.9, hy=0.9, hz=0.1, cfg=cfg)
    builder.add_shape_box(body=-1, hx=0.9, hy=0.9, hz=0.3, cfg=cfg)
    model = builder.finalize(device=device)

    shape_sdf_index = model._shape_sdf_index.numpy()
    texture_sdf_data = model._texture_sdf_data.numpy()
    sdf_idx_a, sdf_idx_b = (int(shape_sdf_index[i]) for i in range(2))
    texture_sdf_data[sdf_idx_b]["voxel_radius"] = texture_sdf_data[sdf_idx_a]["voxel_radius"]
    model._texture_sdf_data.assign(texture_sdf_data)

    texture_a = model._texture_sdf_coarse_textures[sdf_idx_a]
    texture_b = model._texture_sdf_coarse_textures[sdf_idx_b]
    blocks_a = (texture_a.width - 1) * (texture_a.height - 1) * (texture_a.depth - 1)
    blocks_b = (texture_b.width - 1) * (texture_b.height - 1) * (texture_b.depth - 1)
    test.assertNotEqual(blocks_a, blocks_b)

    pipeline = newton.CollisionPipeline(model, broad_phase="explicit")
    hydro = pipeline.hydroelastic_sdf
    test.assertEqual(hydro.total_num_tiles, blocks_b)
    hydro._prepare_shape_sdf_data(model._texture_sdf_data, model._shape_sdf_index)
    hydro._broadphase_sdfs(
        hydro._shape_sdf_data,
        wp.array([wp.transform_identity(), wp.transform_identity()], dtype=wp.transform, device=device),
        wp.array([wp.vec2i(0, 1)], dtype=wp.vec2i, device=device),
        wp.array([1], dtype=wp.int32, device=device),
    )
    test.assertEqual(int(hydro.block_broad_collide_count.numpy()[0]), blocks_b)


add_function_test(
    TestHydroelastic,
    "test_hydroelastic_replica_buffers_scale_with_traversed_grids",
    test_hydroelastic_replica_buffers_scale_with_traversed_grids,
    devices=cuda_devices,
    check_output=False,
)

add_function_test(
    TestHydroelastic,
    "test_hydroelastic_pair_buffer_grid_selection",
    test_hydroelastic_pair_buffer_grid_selection,
    devices=cuda_devices,
    check_output=False,
)


def test_hydroelastic_traversal_buffers_have_headroom(test, device):
    """Keep every traversal stage within its buffer on a curved deep contact.

    Sizing the buffers from the traversed grid rather than from every
    hydroelastic shape makes them substantially smaller, so this guards that a
    demanding contact -- a round pad pressed into a large spherical workpiece
    until the patch spans most of the pad -- still fits at default settings.
    """
    kh_pad = 5.3e6
    kh_workpiece = 1.0e9
    sphere_radius = 0.9
    target_force = 20.0
    # Hydroelastic pressure is kh * depth, so a sphere indented by delta carries
    # kh * pi * R * delta**2. Invert that for the penetration hitting target_force.
    kh_effective = kh_pad * kh_workpiece / (kh_pad + kh_workpiece)
    delta = np.sqrt(target_force / (np.pi * kh_effective * sphere_radius))

    builder = _sanding_contact_builder(64, delta=delta)
    model = builder.finalize(device=device)
    state = model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state)
    pipeline = newton.CollisionPipeline(
        model,
        rigid_contact_max=100000,
        broad_phase="explicit",
        sdf_hydroelastic_config=HydroelasticSDF.Config(reduce_contacts=False),
    )
    contacts = pipeline.contacts()
    pipeline.collide(state, contacts)

    hydro = pipeline.hydroelastic_sdf
    # Pairs match verify_collision_step(): broadphase blocks, then the three
    # octree subblock levels, then iso voxels.
    stages = [
        ("broadphase blocks", hydro.block_broad_collide_count, hydro.max_num_blocks_broad),
        ("iso subblock L0", hydro.iso_buffer_counts[1], hydro.iso_max_dims[0]),
        ("iso subblock L1", hydro.iso_buffer_counts[2], hydro.iso_max_dims[1]),
        ("iso subblock L2", hydro.iso_buffer_counts[3], hydro.iso_max_dims[2]),
        ("iso voxels", hydro.iso_voxel_count, hydro.max_num_iso_voxels),
    ]
    for name, count_array, capacity in stages:
        count = int(count_array.numpy()[0])
        test.assertGreater(count, 0, f"{name} produced no work; the scene is not exercising the contact")
        test.assertLessEqual(count, capacity, f"{name} overflowed: {count} > {capacity}")

    face_count = int(hydro.contact_reduction.contact_count.numpy()[0])
    test.assertLessEqual(face_count, hydro.max_num_face_contacts)

    # Unreduced output uses the contact arrays directly. Hashtable values and
    # per-bin aggregates are dead storage in this mode and must stay empty.
    reducer = hydro.contact_reduction.reducer
    test.assertEqual(reducer.ht_values.shape[0], 0)
    test.assertEqual(reducer.agg_force.shape[0], 0)
    test.assertEqual(reducer.contact_nbin_entry.shape[0], 0)


add_function_test(
    TestHydroelastic,
    "test_hydroelastic_traversal_buffers_have_headroom",
    test_hydroelastic_traversal_buffers_have_headroom,
    devices=cuda_devices,
    check_output=False,
)


if __name__ == "__main__":
    unittest.main(verbosity=2, failfast=True)
