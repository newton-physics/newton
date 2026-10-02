# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Test hydroelastic pressure gradients without requiring texture construction."""

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.geometry.contact_data import ContactData
from newton._src.geometry.contact_reduction_global import GlobalContactReducerData, decode_oct
from newton._src.geometry.contact_reduction_hydroelastic import (
    HydroelasticContactReduction,
    HydroelasticReductionConfig,
    _linearize_contact,
    export_hydroelastic_contact_to_buffer,
)
from newton._src.geometry.sdf_hydroelastic import (
    _mc_pressure_gradient,
    _mc_trilinear_gradient,
    get_decode_contacts_kernel,
    mc_calc_face_texture,
    vec8f,
)
from newton._src.geometry.sdf_mc import get_mc_tables
from newton._src.geometry.sdf_texture import TextureSDFData
from newton.geometry import HydroelasticSDF
from newton.tests.unittest_utils import add_function_test, get_selected_cuda_test_devices, get_test_devices

_CORNERS = np.array(
    [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]],
    dtype=np.float64,
)


@wp.func
def _corner_values(values: wp.array2d[float], index: int) -> vec8f:
    return vec8f(
        values[index, 0],
        values[index, 1],
        values[index, 2],
        values[index, 3],
        values[index, 4],
        values[index, 5],
        values[index, 6],
        values[index, 7],
    )


@wp.kernel
def _evaluate_trilinear_gradient(
    values: wp.array2d[float],
    points: wp.array[wp.vec3],
    voxel_sizes: wp.array[wp.vec3],
    gradients: wp.array[wp.vec3],
):
    """Evaluate production corner interpolation at independent sample points."""
    index = wp.tid()
    gradients[index] = _mc_trilinear_gradient(_corner_values(values, index), points[index], voxel_sizes[index])


@wp.kernel
def _evaluate_pressure_gradient(
    corners_a: wp.array2d[float],
    corners_b: wp.array2d[float],
    points: wp.array[wp.vec3],
    voxel_sizes: wp.array[wp.vec3],
    normals: wp.array[wp.vec3],
    moduli: wp.array[wp.vec2],
    gradients: wp.array[float],
):
    """Evaluate both bodies' projected gradients in the production helper."""
    index = wp.tid()
    gradients[index] = _mc_pressure_gradient(
        _corner_values(corners_a, index),
        _corner_values(corners_b, index),
        points[index],
        voxel_sizes[index],
        normals[index],
        moduli[index][0],
        moduli[index][1],
    )


@wp.kernel
def _evaluate_linearization(values: wp.array[wp.vec4], result: wp.array[wp.vec2]):
    """Evaluate solver-pair conversion independently of contact reduction."""
    index = wp.tid()
    value = values[index]
    stiffness, distance = _linearize_contact(value[0], value[1], value[2], value[3])
    result[index] = wp.vec2(stiffness, distance)


@wp.kernel
def _extract_affine_contact_faces(
    corners_a: wp.array2d[float],
    corners_b: wp.array2d[float],
    moduli: wp.array[wp.vec2],
    voxel_size: wp.vec3,
    triangle_ranges: wp.array[int],
    edge_vertices: wp.array[wp.vec2ub],
    face_count: wp.array[int],
    geometry: wp.array2d[wp.vec4],
    centers: wp.array2d[wp.vec3],
    normals: wp.array2d[wp.vec3],
    springs: wp.array2d[wp.vec2],
):
    """Extract actual marching-cubes faces from two affine pressure fields."""
    index = wp.tid()
    values_a = _corner_values(corners_a, index)
    values_b = _corner_values(corners_b, index)
    kh_a = moduli[index][0]
    kh_b = moduli[index][1]
    difference = vec8f()
    cube = int(0)
    for corner in range(8):
        difference[corner] = -kh_a * values_a[corner] + kh_b * values_b[corner]
        if difference[corner] < 0.0:
            cube = cube | (1 << corner)
    start = triangle_ranges[cube]
    count = (triangle_ranges[cube + 1] - start) // 3
    face_count[index] = count
    descriptor = TextureSDFData()
    descriptor.sdf_box_lower = wp.vec3(0.0)
    descriptor.voxel_size = voxel_size
    for face in range(count):
        area, geometric_area, normal, center, sdf_b, separation, _vertices = mc_calc_face_texture(
            edge_vertices, start + 3 * face, difference, values_b, values_a, descriptor, 0, 0, 0, 0.02, 0.98
        )
        pressure = -kh_b * sdf_b
        gradient = _mc_pressure_gradient(
            values_a, values_b, wp.cw_div(center, voxel_size), voxel_size, normal, kh_a, kh_b
        )
        stiffness, distance = _linearize_contact(area * pressure / (-separation), separation, pressure, gradient)
        geometry[index, face] = wp.vec4(area, geometric_area, separation, pressure)
        centers[index, face] = center
        normals[index, face] = normal
        springs[index, face] = wp.vec2(stiffness, distance)


@wp.struct
class _CapturedContacts:
    contact_count: wp.array[int]
    contact_max: int
    contacts: wp.array[ContactData]


@wp.func
def _capture_contact(contact: ContactData, output: _CapturedContacts, index: int):
    output_index = index
    if output_index < 0:
        output_index = wp.atomic_add(output.contact_count, 0, 1)
    if output_index < output.contact_max:
        output.contacts[output_index] = contact


@wp.kernel
def _seed_faces(
    reducer: GlobalContactReducerData,
    positions: wp.array[wp.vec3],
    face_data: wp.array[wp.vec4],
):
    """Seed production face buffers with independently specified material data."""
    index = wp.tid()
    data = face_data[index]
    contact_id = export_hydroelastic_contact_to_buffer(
        0,
        1,
        positions[index],
        wp.vec3(0.0, 0.0, 1.0),
        data[0],
        data[1],
        data[2],
        index,
        reducer,
    )
    if contact_id >= 0:
        reducer.contact_pressure_gradient[contact_id] = data[3]


def _make_capture(device, capacity=64):
    output = _CapturedContacts()
    output.contact_count = wp.zeros(1, dtype=int, device=device)
    output.contact_max = capacity
    output.contacts = wp.zeros(capacity, dtype=ContactData, device=device)
    return output


def _read_contacts(output):
    count = int(output.contact_count.numpy()[0])
    return output.contacts.numpy()[: min(count, output.contact_max)]


def _pressure_after_displacement(displacement, pressure, modulus_a, modulus_b, gradient_a, gradient_b, normal):
    alpha = modulus_a * np.dot(gradient_a, normal)
    beta = modulus_b * np.dot(gradient_b, normal)
    # Solve for interface position and pressure, not for the series formula.
    return np.linalg.solve([[alpha, 1.0], [beta, 1.0]], [pressure, pressure + beta * displacement])[1]


def test_trilinear_gradient(test, device):
    """Differentiate a trilinear polynomial in physical anisotropic coordinates."""
    rng = np.random.default_rng(3503)
    coefficients = rng.uniform(-2.0, 2.0, size=(64, 8))
    points = rng.uniform(0.0, 1.0, size=(64, 3))
    points[:3] = [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [0.5, 0.5, 0.5]]
    voxel_sizes = rng.uniform(0.1, 1.5, size=(64, 3))
    values = []
    expected = []
    for coefficient, point, size in zip(coefficients, points, voxel_sizes, strict=True):
        a, b, c, d, e, f, g, h = coefficient
        x, y, z = (_CORNERS * size).T
        values.append(a + b * x + c * y + d * z + e * x * y + f * y * z + g * x * z + h * x * y * z)
        x, y, z = point * size
        expected.append([b + e * y + g * z + h * y * z, c + e * x + f * z + h * x * z, d + f * y + g * x + h * x * y])
    result = wp.empty(len(values), dtype=wp.vec3, device=device)
    wp.launch(
        _evaluate_trilinear_gradient,
        dim=len(values),
        inputs=[
            wp.array(values, dtype=float, device=device),
            wp.array(points, dtype=wp.vec3, device=device),
            wp.array(voxel_sizes, dtype=wp.vec3, device=device),
            result,
        ],
        device=device,
    )
    np.testing.assert_allclose(result.numpy(), expected, rtol=2.0e-5, atol=8.0e-6)


def test_pressure_gradient_matches_equilibrium(test, device):
    """Match pressure continuity under relative motion, rotation, and body swapping."""
    corner_a = []
    corner_b = []
    points = []
    voxel_sizes = []
    normals = []
    moduli = []
    expected = []
    pressure = 1.0e4
    rotation = np.asarray(
        wp.quat_to_matrix(wp.quat_from_axis_angle(wp.normalize(wp.vec3(1.0, 2.0, 3.0)), 0.73))
    ).reshape(3, 3)
    for ratio in (0.01, 0.1, 1.0, 10.0, 100.0):
        for tangent_fraction in (0.0, 0.25, 0.5, np.sqrt(3.0) / 2.0, 0.99):
            modulus_a, modulus_b = 1.0e6, 1.0e6 * ratio
            tangent = min(modulus_a, modulus_b) * tangent_fraction
            x_a, x_b = tangent / modulus_a, tangent / modulus_b
            original_a = np.array([x_a, 0.0, np.sqrt(1.0 - x_a * x_a)])
            original_b = np.array([x_b, 0.0, -np.sqrt(1.0 - x_b * x_b)])
            for frame in (np.eye(3), rotation):
                for swapped in (False, True):
                    gradient_a, gradient_b = frame @ original_a, frame @ original_b
                    normal = frame[:, 2]
                    k_a, k_b = modulus_a, modulus_b
                    if swapped:
                        gradient_a, gradient_b = gradient_b, gradient_a
                        k_a, k_b = k_b, k_a
                        normal = -normal
                    point = np.array([0.27, 0.43, 0.61])
                    size = np.array([0.11, 0.17, 0.23])
                    offsets = (_CORNERS - point) * size
                    corner_a.append(-pressure / k_a + offsets @ gradient_a)
                    corner_b.append(-pressure / k_b + offsets @ gradient_b)
                    points.append(point)
                    voxel_sizes.append(size)
                    normals.append(normal)
                    moduli.append([k_a, k_b])
                    derivatives = []
                    for fraction in (0.01, 0.001, 0.0001):
                        step = pressure * (1.0 / k_a + 1.0 / k_b) * fraction
                        args = (pressure, k_a, k_b, gradient_a, gradient_b, normal)
                        derivatives.append(
                            (_pressure_after_displacement(-step, *args) - _pressure_after_displacement(step, *args))
                            / (2.0 * step)
                        )
                    np.testing.assert_allclose(derivatives, derivatives[0], rtol=1.0e-8)
                    expected.append(derivatives[1])
    result = wp.empty(len(expected), dtype=float, device=device)
    wp.launch(
        _evaluate_pressure_gradient,
        dim=len(expected),
        inputs=[
            wp.array(corner_a, dtype=float, device=device),
            wp.array(corner_b, dtype=float, device=device),
            wp.array(points, dtype=wp.vec3, device=device),
            wp.array(voxel_sizes, dtype=wp.vec3, device=device),
            wp.array(normals, dtype=wp.vec3, device=device),
            wp.array(moduli, dtype=wp.vec2, device=device),
            result,
        ],
        device=device,
    )
    np.testing.assert_allclose(result.numpy(), expected, rtol=3.0e-5, atol=1.0e-2)


def test_linearization_preserves_force(test, device):
    """Preserve current force and retain speculative and derivative-free contacts."""
    values = np.array(
        [
            [100000.0, -0.02, 10000.0, 250000.0],
            [750.0, -0.03, 300.0, 20000.0],
            [100000.0, -0.02, 10000.0, 0.0],
            [100000.0, -0.02, 10000.0, -1.0],
            [100000.0, -0.02, 10000.0, np.nan],
            [100000.0, -0.02, 10000.0, np.inf],
            [5000.0, 0.0, 0.0, 250000.0],
            [5000.0, 0.005, 0.0, 250000.0],
        ],
        dtype=np.float32,
    )
    result = wp.empty(len(values), dtype=wp.vec2, device=device)
    wp.launch(
        _evaluate_linearization,
        dim=len(values),
        inputs=[wp.array(values, dtype=wp.vec4, device=device), result],
        device=device,
    )
    actual = result.numpy()
    np.testing.assert_allclose(actual[:2], [[50000.0, -0.04], [1500.0, -0.015]], rtol=2.0e-6)
    np.testing.assert_array_equal(actual[2:], values[2:, :2])
    np.testing.assert_allclose(actual[:, 0] * actual[:, 1], values[:, 0] * values[:, 1], rtol=2.0e-6)


def test_pressure_gradient_degenerate_fields(test, device):
    """Reject invalid projected gradients and avoid overflowing finite series inputs."""
    z = _CORNERS[:, 2] - 0.5
    corners_a = np.array([z, np.zeros(8), -z, z, z, z, np.full(8, np.nan)])
    corners_b = np.array([-z, -z, -z, z, -z, -z, -z])
    moduli = np.array(
        [
            [1.0e30, 1.0e30],
            [1.0e6, 1.0e6],
            [1.0e6, 1.0e6],
            [1.0e6, 1.0e6],
            [np.inf, 1.0e6],
            [np.nan, 1.0e6],
            [1.0e6, 1.0e6],
        ]
    )
    count = len(moduli)
    result = wp.empty(count, dtype=float, device=device)
    wp.launch(
        _evaluate_pressure_gradient,
        dim=count,
        inputs=[
            wp.array(corners_a, dtype=float, device=device),
            wp.array(corners_b, dtype=float, device=device),
            wp.full(count, wp.vec3(0.5), dtype=wp.vec3, device=device),
            wp.full(count, wp.vec3(1.0), dtype=wp.vec3, device=device),
            wp.full(count, wp.vec3(0.0, 0.0, 1.0), dtype=wp.vec3, device=device),
            wp.array(moduli, dtype=wp.vec2, device=device),
            result,
        ],
        device=device,
    )
    actual = result.numpy()
    np.testing.assert_allclose(actual[0], 5.0e29, rtol=2.0e-6)
    np.testing.assert_array_equal(actual[1:], np.zeros(count - 1))


def test_marching_cubes_pressure_linearization(test, device):
    """Linearize real marching-cubes faces without changing their geometry or pressure force."""
    size = np.array([0.002, 0.004, 0.006])
    interface = np.array([0.5, 0.5, 0.4]) * size
    p0 = 10000.0
    cases = ((1.0e6, 1.0e6, 0.0), (1.0e6, 1.0e6, np.sqrt(3.0) / 2.0), (1.0e6, 2.0e6, 0.8))
    corner_a, corner_b, field_gradients = [], [], []
    for kh_a, kh_b, tangent_fraction in cases:
        tangent = min(kh_a, kh_b) * tangent_fraction
        x_a, x_b = tangent / kh_a, tangent / kh_b
        gradient_a = np.array([x_a, 0.0, np.sqrt(1.0 - x_a * x_a)])
        gradient_b = np.array([x_b, 0.0, -np.sqrt(1.0 - x_b * x_b)])
        corner_a.append(-p0 / kh_a + (_CORNERS * size - interface) @ gradient_a)
        corner_b.append(-p0 / kh_b + (_CORNERS * size - interface) @ gradient_b)
        field_gradients.append((gradient_a, gradient_b))
    tables = get_mc_tables(device)
    count = wp.zeros(len(cases), dtype=int, device=device)
    geometry = wp.zeros((len(cases), 5), dtype=wp.vec4, device=device)
    centers = wp.zeros((len(cases), 5), dtype=wp.vec3, device=device)
    normals = wp.zeros((len(cases), 5), dtype=wp.vec3, device=device)
    springs = wp.zeros((len(cases), 5), dtype=wp.vec2, device=device)
    wp.launch(
        _extract_affine_contact_faces,
        dim=len(cases),
        inputs=[
            wp.array(corner_a, dtype=float, device=device),
            wp.array(corner_b, dtype=float, device=device),
            wp.array([[case[0], case[1]] for case in cases], dtype=wp.vec2, device=device),
            wp.vec3(size),
            tables[0],
            tables[4],
            count,
            geometry,
            centers,
            normals,
            springs,
        ],
        device=device,
    )
    np.testing.assert_array_equal(count.numpy(), np.full(len(cases), 2))
    geometry_np, centers_np, normals_np, springs_np = (
        geometry.numpy(),
        centers.numpy(),
        normals.numpy(),
        springs.numpy(),
    )
    for index, (kh_a, kh_b, _) in enumerate(cases):
        gradient_a, gradient_b = field_gradients[index]
        np.testing.assert_allclose(np.sum(geometry_np[index, :2, 0]), size[0] * size[1], rtol=3.0e-6)
        for face in range(2):
            center = centers_np[index, face]
            normal = np.array([0.0, 0.0, 1.0])
            np.testing.assert_allclose(normals_np[index, face], normal, atol=2.0e-6)
            np.testing.assert_allclose(center[2], interface[2], atol=2.0e-8)
            pressure = p0 - kh_b * np.dot(gradient_b, center - interface)
            step = 1.0e-6
            args = (pressure, kh_a, kh_b, gradient_a, gradient_b, normal)
            gradient = (_pressure_after_displacement(-step, *args) - _pressure_after_displacement(step, *args)) / (
                2.0 * step
            )
            expected_stiffness = geometry_np[index, face, 0] * gradient
            np.testing.assert_allclose(springs_np[index, face], [expected_stiffness, -pressure / gradient], rtol=3.0e-5)
            np.testing.assert_allclose(geometry_np[index, face, 2], -pressure / kh_a - pressure / kh_b, rtol=3.0e-6)
            np.testing.assert_allclose(
                -np.prod(springs_np[index, face]), geometry_np[index, face, 0] * pressure, rtol=3.0e-6
            )


def test_unreduced_contact_linearization(test, device):
    """Export oblique solver pairs while retaining geometric buffers and gap contacts."""
    reduction = HydroelasticContactReduction(16, device=device, writer_func=_capture_contact, enable_reduction=False)
    reducer = reduction.reducer
    face_data = np.array(
        [[-0.02, 0.2, 10000.0, 250000.0], [-0.02, 0.2, 10000.0, 0.0], [0.0, 0.2, 0.0, 0.0], [0.005, 0.2, 0.0, 0.0]],
        dtype=np.float32,
    )
    wp.launch(
        _seed_faces,
        dim=len(face_data),
        inputs=[
            reducer.get_data_struct(),
            wp.zeros(len(face_data), dtype=wp.vec3, device=device),
            wp.array(face_data, dtype=wp.vec4, device=device),
        ],
        device=device,
    )
    output = _make_capture(device)
    transforms = wp.array(
        [
            wp.transform_identity(),
            wp.transform(wp.vec3(1.0, 2.0, 3.0), wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), wp.pi / 2.0)),
        ],
        dtype=wp.transform,
        device=device,
    )
    wp.launch(
        get_decode_contacts_kernel(0.01, _capture_contact),
        dim=3,
        inputs=[
            3,
            reducer.contact_count,
            wp.array([1.0e6, 1.0e6], dtype=float, device=device),
            transforms,
            wp.array([0.01, 0.02], dtype=float, device=device),
            reducer.position_depth,
            reducer.normal,
            reducer.shape_pairs,
            reducer.contact_fingerprints,
            reducer.contact_area,
            reducer.contact_pressure,
            reducer.contact_pressure_gradient,
            reducer.capacity,
            output,
        ],
        device=device,
    )
    contacts = _read_contacts(output)
    test.assertEqual(len(contacts), len(face_data))
    contacts = contacts[np.argsort(contacts["sort_sub_key"])]
    np.testing.assert_allclose(contacts["contact_stiffness"], [50000.0, 100000.0, 5000.0, 5000.0], rtol=2.0e-6)
    np.testing.assert_allclose(contacts["contact_distance"], [-0.04, -0.02, 0.0, 0.005], rtol=2.0e-6)
    np.testing.assert_allclose(contacts["gap_sum"], 0.03, rtol=2.0e-6)
    np.testing.assert_allclose(contacts["contact_point_center"], np.tile([1.0, 2.0, 3.0], (len(contacts), 1)))
    np.testing.assert_allclose(
        contacts["contact_normal_a_to_b"], np.tile([1.0, 0.0, 0.0], (len(contacts), 1)), atol=2.0e-7
    )
    fingerprints = reducer.contact_fingerprints.numpy()[1 : len(face_data) + 1]
    np.testing.assert_array_equal(reducer.position_depth.numpy()[1 : len(face_data) + 1, 3], face_data[fingerprints, 0])


def test_reduced_contact_linearization(test, device, anchor_contact=False, moment_matching=False):
    """Preserve reduced force while applying each representative's pressure derivative."""
    config = HydroelasticReductionConfig(anchor_contact=anchor_contact, moment_matching=moment_matching)
    reduction = HydroelasticContactReduction(
        32, device=device, writer_func=_capture_contact, config=config, deterministic=True
    )
    positions = np.array([[-0.3, -0.2, 0.0], [-0.3, 0.2, 0.0], [0.3, -0.2, 0.0], [0.3, 0.2, 0.0]], dtype=np.float32)
    pressures = np.array([5000.0, 10000.0, 15000.0, 20000.0], dtype=np.float32)
    areas = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
    gradients = np.full(4, 250000.0, dtype=np.float32)
    face_data = np.column_stack((-pressures / 500000.0, areas, pressures, gradients))
    transforms = wp.array([wp.transform_identity(), wp.transform_identity()], dtype=wp.transform, device=device)
    kh = wp.array([1.0e6, 1.0e6], dtype=float, device=device)
    lower = wp.array([[-1.0, -1.0, -1.0], [-1.0, -1.0, -1.0]], dtype=wp.vec3, device=device)
    upper = wp.array([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]], dtype=wp.vec3, device=device)
    resolution = wp.array([[8, 8, 8], [8, 8, 8]], dtype=wp.vec3i, device=device)
    snapshots = []
    for _ in range(2):
        reduction.clear()
        wp.launch(
            _seed_faces,
            dim=len(positions),
            inputs=[
                reduction.get_data_struct(),
                wp.array(positions, dtype=wp.vec3, device=device),
                wp.array(face_data, dtype=wp.vec4, device=device),
            ],
            device=device,
        )
        reduction.reduce(kh, transforms, lower, upper, resolution, grid_size=8)
        output = _make_capture(device)
        reduction.export(kh, wp.zeros(2, dtype=float, device=device), transforms, output, grid_size=8)
        contacts = _read_contacts(output)
        test.assertGreater(len(contacts), 0)
        test.assertLessEqual(len(contacts), output.contact_max)
        test.assertEqual(int(reduction.reducer.ht_insert_failures.numpy()[0]), 0)
        contacts = contacts[np.argsort(contacts["sort_sub_key"])]
        force = -contacts["contact_stiffness"] * contacts["contact_distance"]
        np.testing.assert_allclose(np.sum(force), np.dot(areas, pressures), rtol=5.0e-6)
        test.assertTrue(np.all(np.isfinite(contacts["contact_stiffness"])))
        test.assertTrue(np.all(contacts["contact_stiffness"] > 0.0))
        for distance in contacts["contact_distance"]:
            test.assertLess(np.min(np.abs(distance + pressures / gradients)), 1.0e-6)
        snapshots.append(contacts)
    np.testing.assert_array_equal(snapshots[0], snapshots[1])


def test_rotated_box_generated_pressure_gradients(test, device):
    """Generate analytic oblique gradients through texture traversal and face pre-pruning."""
    rotation = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), 0.45)
    box_specs = (
        (wp.transform_identity(), 1.0e6),
        (wp.transform(wp.vec3(0.0, 0.0, 0.82), rotation), 2.0e6),
    )
    for swapped in (False, True):
        builder = newton.ModelBuilder()
        specs = box_specs[::-1] if swapped else box_specs
        frames = []
        stiffnesses = []
        for transform, kh in specs:
            body = builder.add_body(xform=transform)
            cfg = newton.ModelBuilder.ShapeConfig(
                is_hydroelastic=True,
                kh=kh,
                sdf_max_resolution=48,
                sdf_narrow_band_range=(-0.3, 0.1),
                margin=0.0,
                gap=0.0,
            )
            builder.add_shape_box(body=body, hx=0.5, hy=0.5, hz=0.5, cfg=cfg)
            frames.append(
                (
                    np.asarray(wp.transform_get_translation(transform)),
                    np.asarray(wp.quat_to_matrix(wp.transform_get_rotation(transform))).reshape(3, 3),
                )
            )
            stiffnesses.append(kh)
        model = builder.finalize(device=device)
        state = model.state()
        newton.eval_fk(model, model.joint_q, model.joint_qd, state)
        # Keep every interpolation stencil away from box SDF ridges.
        ridge_clearance = 4.0 * np.max(model._texture_sdf_data.numpy()["voxel_size"])
        for reduce_contacts in (False, True):
            with test.subTest(swapped=swapped, reduce_contacts=reduce_contacts):
                pipeline = newton.CollisionPipeline(
                    model,
                    broad_phase="explicit",
                    rigid_contact_max=10000,
                    sdf_hydroelastic_config=HydroelasticSDF.Config(
                        reduce_contacts=reduce_contacts,
                        pre_prune_contacts=reduce_contacts,
                        buffer_fraction=1.0,
                        # Compare against analytic planes without edge-clamp distortion.
                        mc_edge_clamp_min=0.0,
                    ),
                )
                contacts = pipeline.contacts()
                pipeline.collide(state, contacts)
                reducer = pipeline.hydroelastic_sdf.contact_reduction.reducer
                count = int(reducer.contact_count.numpy()[0])
                test.assertGreater(count, 0)
                test.assertLessEqual(count, reducer.capacity)
                positions = reducer.position_depth.numpy()[1 : count + 1]
                pairs = reducer.shape_pairs.numpy()[1 : count + 1]
                encoded_normals = reducer.normal.numpy()[1 : count + 1]
                pressures = reducer.contact_pressure.numpy()[1 : count + 1]
                actual_gradients = reducer.contact_pressure_gradient.numpy()[1 : count + 1]
                selected = 0
                for index, (shape_a, shape_b) in enumerate(pairs):
                    if positions[index, 3] >= 0.0 or pressures[index] <= 0.0:
                        continue
                    origin_b, frame_b = frames[shape_b]
                    point_world = origin_b + frame_b @ positions[index, :3]
                    field_gradients = []
                    for shape in (shape_a, shape_b):
                        origin, frame = frames[shape]
                        point_local = frame.T @ (point_world - origin)
                        distances = np.abs(point_local) - 0.5
                        order = np.argsort(distances)
                        if distances[order[-1]] >= 0.0 or distances[order[-1]] - distances[order[-2]] < ridge_clearance:
                            break
                        local_gradient = np.zeros(3)
                        local_gradient[order[-1]] = np.sign(point_local[order[-1]])
                        field_gradients.append(frame @ local_gradient)
                    if len(field_gradients) != 2:
                        continue
                    gradient_a, gradient_b = field_gradients
                    kh_a, kh_b = stiffnesses[shape_a], stiffnesses[shape_b]
                    expected_normal = kh_a * gradient_a - kh_b * gradient_b
                    expected_normal /= np.linalg.norm(expected_normal)
                    if np.dot(gradient_a, expected_normal) <= 0.0 or np.dot(gradient_b, expected_normal) >= 0.0:
                        continue
                    step = 1.0e-6
                    args = (float(pressures[index]), kh_a, kh_b, gradient_a, gradient_b, expected_normal)
                    expected_gradient = (
                        _pressure_after_displacement(-step, *args) - _pressure_after_displacement(step, *args)
                    ) / (2.0 * step)
                    test.assertGreater(expected_gradient, 0.0)
                    if expected_gradient >= 0.99 / (1.0 / kh_a + 1.0 / kh_b):
                        continue
                    actual_normal = frame_b @ np.asarray(decode_oct(wp.vec2(encoded_normals[index])))
                    np.testing.assert_allclose(actual_normal, expected_normal, atol=2.0e-3)
                    np.testing.assert_allclose(actual_gradients[index], expected_gradient, rtol=0.01)
                    selected += 1
                test.assertGreaterEqual(selected, 4, "The scene must retain an oblique planar contact patch")


class TestHydroelasticPressureGradient(unittest.TestCase):
    pass


devices = get_test_devices()
add_function_test(TestHydroelasticPressureGradient, "test_trilinear_gradient", test_trilinear_gradient, devices=devices)
add_function_test(
    TestHydroelasticPressureGradient,
    "test_pressure_gradient_matches_equilibrium",
    test_pressure_gradient_matches_equilibrium,
    devices=devices,
)
add_function_test(
    TestHydroelasticPressureGradient,
    "test_linearization_preserves_force",
    test_linearization_preserves_force,
    devices=devices,
)
add_function_test(
    TestHydroelasticPressureGradient,
    "test_unreduced_contact_linearization",
    test_unreduced_contact_linearization,
    devices=devices,
)
add_function_test(
    TestHydroelasticPressureGradient,
    "test_pressure_gradient_degenerate_fields",
    test_pressure_gradient_degenerate_fields,
    devices=devices,
)
add_function_test(
    TestHydroelasticPressureGradient,
    "test_marching_cubes_pressure_linearization",
    test_marching_cubes_pressure_linearization,
    devices=devices,
)
add_function_test(
    TestHydroelasticPressureGradient,
    "test_reduced_contact_linearization",
    test_reduced_contact_linearization,
    devices=devices,
)
add_function_test(
    TestHydroelasticPressureGradient,
    "test_reduced_contact_linearization_anchor",
    test_reduced_contact_linearization,
    devices=devices,
    anchor_contact=True,
)
add_function_test(
    TestHydroelasticPressureGradient,
    "test_reduced_contact_linearization_moments",
    test_reduced_contact_linearization,
    devices=devices,
    moment_matching=True,
)
add_function_test(
    TestHydroelasticPressureGradient,
    "test_rotated_box_generated_pressure_gradients",
    test_rotated_box_generated_pressure_gradients,
    devices=get_selected_cuda_test_devices(),
)


if __name__ == "__main__":
    unittest.main(verbosity=2)
