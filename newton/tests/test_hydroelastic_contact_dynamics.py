# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Exercise hydroelastic spring tangents in a preloaded, moving two-pad contact.

Affine pressure fields replace texture traversal, allowing the same actual
marching-cubes, contact export, and SemiImplicit solver path on CPU and CUDA.
The mirrored pads cancel their pressure-gradient moments. Their fixed areas
and persistent topology isolate the effect of reusing contacts across steps.
"""

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.geometry.contact_reduction_global import GlobalContactReducer, GlobalContactReducerData
from newton._src.geometry.contact_reduction_hydroelastic import export_hydroelastic_contact_to_buffer
from newton._src.geometry.sdf_hydroelastic import (
    _mc_pressure_gradient,
    get_decode_contacts_kernel,
    mc_calc_face_texture,
)
from newton._src.geometry.sdf_mc import get_mc_tables
from newton._src.geometry.sdf_texture import TextureSDFData
from newton._src.sim.collide import ContactWriterData, write_contact
from newton.solvers import SolverSemiImplicit
from newton.tests.unittest_utils import add_function_test, get_test_devices

_CORNERS = wp.constant(
    wp.types.matrix(shape=(8, 3), dtype=wp.float32)(
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        1.0,
        1.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        1.0,
        1.0,
        0.0,
        1.0,
        1.0,
        1.0,
        1.0,
        0.0,
        1.0,
        1.0,
    )
)
_VEC8 = wp.types.vector(length=8, dtype=wp.float32)


@wp.kernel
def _generate_pad_contacts(
    body_q: wp.array[wp.transform],
    kh: wp.vec2,
    tangent_fraction: float,
    pressure_at_rest: float,
    use_series: bool,
    triangle_ranges: wp.array[int],
    edge_vertices: wp.array[wp.vec2ub],
    reducer: GlobalContactReducerData,
    shape_transform: wp.array[wp.transform],
):
    """Extract mirrored affine patches at the current body translation."""
    transform = body_q[0]
    displacement = wp.transform_get_translation(transform)[2]
    shape_transform[0] = wp.transform_identity()
    shape_transform[1] = transform
    size = wp.vec3(0.01, 0.005, 0.01)
    tau = tangent_fraction * wp.min(kh[0], kh[1])
    for pad in range(2):
        direction = float(2 * pad - 1)
        center_x = direction * 0.012
        gradient_a = wp.vec3(direction * tau / kh[0], 0.0, wp.sqrt(1.0 - (tau / kh[0]) ** 2.0))
        gradient_b = wp.vec3(direction * tau / kh[1], 0.0, -wp.sqrt(1.0 - (tau / kh[1]) ** 2.0))
        a = _VEC8()
        b = _VEC8()
        difference = _VEC8()
        cube = int(0)
        for corner in range(8):
            point = wp.cw_mul(
                wp.vec3(_CORNERS[corner, 0], _CORNERS[corner, 1], _CORNERS[corner, 2]) - wp.vec3(0.5), size
            )
            a[corner] = -pressure_at_rest / kh[0] + wp.dot(gradient_a, point)
            b[corner] = -pressure_at_rest / kh[1] + wp.dot(gradient_b, point - wp.vec3(0.0, 0.0, displacement))
            difference[corner] = -kh[0] * a[corner] + kh[1] * b[corner]
            if difference[corner] < 0.0:
                cube = cube | (1 << corner)
        descriptor = TextureSDFData()
        descriptor.voxel_size = size
        descriptor.sdf_box_lower = wp.vec3(center_x, 0.0, 0.0) - 0.5 * size
        start = triangle_ranges[cube]
        count = (triangle_ranges[cube + 1] - start) // 3
        for face in range(count):
            area, _geometric_area, normal, center, sdf_b, separation, _vertices = mc_calc_face_texture(
                edge_vertices, start + 3 * face, difference, b, a, descriptor, 0, 0, 0, 0.0, 1.0
            )
            pressure = -kh[1] * sdf_b
            inverse_transform = wp.transform_inverse(transform)
            local_center = wp.transform_point(inverse_transform, center)
            local_normal = wp.transform_vector(inverse_transform, normal)
            contact_id = export_hydroelastic_contact_to_buffer(
                0, 1, local_center, local_normal, separation, area, pressure, 5 * pad + face, reducer
            )
            if contact_id >= 0 and use_series:
                point = wp.cw_div(center - descriptor.sdf_box_lower, size)
                reducer.contact_pressure_gradient[contact_id] = _mc_pressure_gradient(
                    a, b, point, size, normal, kh[0], kh[1]
                )


@wp.kernel
def _apply_preload(body_f: wp.array[wp.spatial_vector], load: float):
    """Balance the two pads' rest pressure with a constant external force."""
    body_f[0] = wp.spatial_vector(0.0, 0.0, -load, 0.0, 0.0, 0.0)


@wp.kernel
def _record_motion(
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    index: int,
    motion: wp.array[wp.vec2],
    transverse: wp.array[wp.spatial_vector],
):
    """Record translation and velocity after an actual solver step."""
    motion[index] = wp.vec2(wp.transform_get_translation(body_q[0])[2], body_qd[0][2])
    transverse[index] = body_qd[0]


def _writer_data(model, state, contacts):
    """Connect the decoder to Newton's standard rigid-contact writer."""
    writer = ContactWriterData()
    writer.contact_max = contacts.rigid_contact_max
    writer.body_q = state.body_q
    writer.shape_body = model.shape_body
    writer.shape_gap = model.shape_gap
    writer.contact_count = contacts.rigid_contact_count
    for output, attribute in (
        ("out_shape0", "rigid_contact_shape0"),
        ("out_shape1", "rigid_contact_shape1"),
        ("out_point0", "rigid_contact_point0"),
        ("out_point1", "rigid_contact_point1"),
        ("out_offset0", "rigid_contact_offset0"),
        ("out_offset1", "rigid_contact_offset1"),
        ("out_normal", "rigid_contact_normal"),
        ("out_margin0", "rigid_contact_margin0"),
        ("out_margin1", "rigid_contact_margin1"),
        ("out_tids", "rigid_contact_tids"),
        ("out_stiffness", "rigid_contact_stiffness"),
        ("out_damping", "rigid_contact_damping"),
        ("out_friction", "rigid_contact_friction"),
    ):
        setattr(writer, output, getattr(contacts, attribute))
    writer.out_sort_key = wp.zeros(0, dtype=wp.int64, device=model.device)
    return writer


def run_pad_motion(device, *, use_series, tangent_fraction, modulus_ratio=1.0, refresh_steps=16, steps=400, dt=0.0005):
    """Advance the actual solver while refreshing the pressure patch periodically."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    body = builder.add_body(mass=1.0, inertia=wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0), lock_inertia=True)
    cfg = newton.ModelBuilder.ShapeConfig(density=0.0, ke=123.0, kd=0.0, kf=0.0, mu=0.0, ka=0.0, margin=0.0, gap=0.0)
    builder.add_shape_box(body=-1, hx=0.03, hy=0.01, hz=0.02, cfg=cfg)
    builder.add_shape_box(body=body, hx=0.03, hy=0.01, hz=0.02, cfg=cfg)
    model = builder.finalize(device=device)
    state_in, state_out = model.state(), model.state()
    state_in.body_qd.assign(np.array([[0.0, 0.0, 0.01, 0.0, 0.0, 0.0]], dtype=np.float32))
    control = model.control()
    solver = SolverSemiImplicit(model, angular_damping=0.0, enable_tri_contact=False)
    contacts = newton.Contacts(16, 0, device=device, per_contact_shape_properties=True)
    writer = _writer_data(model, state_in, contacts)
    reducer = GlobalContactReducer(16, device=device, store_hydroelastic_data=True, enable_reduction=False)
    transforms = wp.empty(2, dtype=wp.transform, device=device)
    moduli = np.array([4.0e7, 4.0e7 * modulus_ratio])
    kh = wp.array(moduli, dtype=float, device=device)
    tables = get_mc_tables(device)
    decode = get_decode_contacts_kernel(0.01, write_contact)
    pressure_at_rest = 4.0e5
    total_area = 1.0e-4
    motion = wp.zeros(steps, dtype=wp.vec2, device=device)
    transverse = wp.zeros(steps, dtype=wp.spatial_vector, device=device)
    counts = []
    first_springs = None
    for step in range(steps):
        if step % refresh_steps == 0:
            writer.body_q = state_in.body_q
            reducer.contact_count.zero_()
            contacts.clear()
            wp.launch(
                _generate_pad_contacts,
                dim=1,
                inputs=[
                    state_in.body_q,
                    wp.vec2(moduli),
                    tangent_fraction,
                    pressure_at_rest,
                    use_series,
                    tables[0],
                    tables[4],
                    reducer.get_data_struct(),
                    transforms,
                ],
                device=device,
            )
            wp.launch(
                decode,
                dim=1,
                inputs=[
                    1,
                    reducer.contact_count,
                    kh,
                    transforms,
                    model.shape_gap,
                    reducer.position_depth,
                    reducer.normal,
                    reducer.shape_pairs,
                    reducer.contact_fingerprints,
                    reducer.contact_area,
                    reducer.contact_pressure,
                    reducer.contact_pressure_gradient,
                    reducer.capacity,
                    writer,
                ],
                device=device,
            )
            counts.append(int(contacts.rigid_contact_count.numpy()[0]))
            if first_springs is None:
                first_springs = contacts.rigid_contact_stiffness.numpy()[: counts[-1]]
        state_in.clear_forces()
        wp.launch(_apply_preload, dim=1, inputs=[state_in.body_f, pressure_at_rest * total_area], device=device)
        solver.step(state_in, state_out, control, contacts, dt)
        wp.launch(
            _record_motion, dim=1, inputs=[state_out.body_q, state_out.body_qd, step, motion, transverse], device=device
        )
        state_in, state_out = state_out, state_in

    # Independent pressure continuity: solve for interface location and pressure.
    tau = tangent_fraction * min(moduli)
    slope_a = moduli[0] * np.sqrt(1.0 - (tau / moduli[0]) ** 2)
    slope_b = -moduli[1] * np.sqrt(1.0 - (tau / moduli[1]) ** 2)
    displacement = 1.0e-5
    pressure_plus = np.linalg.solve(
        [[slope_a, 1.0], [slope_b, 1.0]], [pressure_at_rest, pressure_at_rest + slope_b * displacement]
    )[1]
    pressure_minus = np.linalg.solve(
        [[slope_a, 1.0], [slope_b, 1.0]], [pressure_at_rest, pressure_at_rest - slope_b * displacement]
    )[1]
    stiffness = total_area * (pressure_minus - pressure_plus) / (2.0 * displacement)
    q, velocity = 0.0, 0.01
    reference = []
    for _ in range(steps):
        velocity -= dt * stiffness * q
        q += dt * velocity
        reference.append([q, velocity])
    omega = np.sqrt(stiffness)
    time = dt * np.arange(1, steps + 1)
    continuous = np.column_stack((0.01 / omega * np.sin(omega * time), 0.01 * np.cos(omega * time)))
    return {
        "motion": motion.numpy(),
        "transverse": transverse.numpy(),
        "reference": np.asarray(reference),
        "continuous": continuous,
        "stiffness": stiffness,
        "counts": np.asarray(counts),
        "first_springs": first_springs,
    }


def test_contact_refresh_dynamics(test, device):
    """Preserve a preloaded pad's response when contacts span multiple solver steps."""
    for ratio in (1.0, 4.0):
        with test.subTest(modulus_ratio=ratio):
            options = {"tangent_fraction": np.sqrt(3.0) / 2.0, "modulus_ratio": ratio, "refresh_steps": 16}
            tangent = run_pad_motion(device, use_series=True, **options)
            secant = run_pad_motion(device, use_series=False, **options)
            np.testing.assert_array_equal(tangent["counts"], 4)
            np.testing.assert_array_equal(secant["counts"], 4)
            np.testing.assert_allclose(np.sum(tangent["first_springs"]), tangent["stiffness"], rtol=2.0e-6)
            np.testing.assert_allclose(tangent["motion"], tangent["reference"], rtol=0.0, atol=2.0e-6)
            np.testing.assert_allclose(tangent["transverse"][:, [0, 1, 3, 4, 5]], 0.0, atol=2.0e-6)
            amplitude = 0.01 / np.sqrt(tangent["stiffness"])
            tangent_error = np.sqrt(np.mean((tangent["motion"][:, 0] - tangent["reference"][:, 0]) ** 2)) / amplitude
            secant_error = np.sqrt(np.mean((secant["motion"][:, 0] - secant["reference"][:, 0]) ** 2)) / amplitude
            test.assertLess(tangent_error, 0.002)
            test.assertGreater(secant_error, 20.0 * max(tangent_error, 1.0e-6))
            test.assertGreater(secant_error, 0.03)


def test_contact_refresh_controls(test, device):
    """Recover equal responses for head-on fields and contacts refreshed every step."""
    for tangent_fraction, refresh_steps in ((0.0, 16), (np.sqrt(3.0) / 2.0, 1)):
        with test.subTest(tangent_fraction=tangent_fraction, refresh_steps=refresh_steps):
            options = {"tangent_fraction": tangent_fraction, "refresh_steps": refresh_steps, "steps": 120}
            tangent = run_pad_motion(device, use_series=True, **options)
            secant = run_pad_motion(device, use_series=False, **options)
            np.testing.assert_allclose(tangent["motion"], secant["motion"], rtol=0.0, atol=2.0e-6)
            np.testing.assert_allclose(tangent["motion"][:, 0], tangent["continuous"][:, 0], rtol=0.0, atol=4.0e-6)


class TestHydroelasticContactDynamics(unittest.TestCase):
    pass


devices = get_test_devices()
add_function_test(
    TestHydroelasticContactDynamics, "test_contact_refresh_dynamics", test_contact_refresh_dynamics, devices=devices
)
add_function_test(
    TestHydroelasticContactDynamics, "test_contact_refresh_controls", test_contact_refresh_controls, devices=devices
)


if __name__ == "__main__":
    unittest.main(verbosity=2)
