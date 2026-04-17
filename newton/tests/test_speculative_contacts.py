# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for speculative contact support in the collision pipeline."""

import unittest

import numpy as np
import warp as wp

import newton
from newton.tests.unittest_utils import add_function_test, get_cuda_test_devices


def _build_two_sphere_model(
    device,
    gap: float = 3.0,
    velocity: float = 0.0,
    speculative_config: newton.SpeculativeContactConfig | None = None,
):
    """Build a simple scene with two spheres separated along the X axis.

    Body A is at ``(-gap/2, 0, 0)`` moving with ``+velocity`` in X.
    Body B is at ``(+gap/2, 0, 0)`` and is static (body index -1 via ground plane trick
    is avoided; instead it's a dynamic body with zero velocity).

    Returns:
        model, state, pipeline, contacts
    """
    builder = newton.ModelBuilder(gravity=0.0)
    builder.rigid_gap = 0.0

    body_a = builder.add_body(xform=wp.transform(wp.vec3(-gap / 2.0, 0.0, 0.0)))
    builder.add_shape_sphere(body_a, radius=0.5)
    builder.body_qd[-1] = (velocity, 0.0, 0.0, 0.0, 0.0, 0.0)

    body_b = builder.add_body(xform=wp.transform(wp.vec3(gap / 2.0, 0.0, 0.0)))
    builder.add_shape_sphere(body_b, radius=0.5)

    model = builder.finalize(device=device)
    state = model.state()

    pipeline = newton.CollisionPipeline(
        model,
        broad_phase="nxn",
        speculative_config=speculative_config,
    )
    contacts = pipeline.contacts()
    return model, state, pipeline, contacts


def test_speculative_disabled_no_contacts(test, device):
    """Two spheres far apart with no speculative config produce zero contacts."""
    _, state, pipeline, contacts = _build_two_sphere_model(device, gap=3.0, velocity=100.0)
    pipeline.collide(state, contacts)
    count = contacts.rigid_contact_count.numpy()[0]
    test.assertEqual(count, 0, "Non-speculative pipeline should not detect contacts for separated spheres")


def test_speculative_enabled_catches_fast_object(test, device):
    """Two spheres far apart but approaching fast produce speculative contacts."""
    config = newton.SpeculativeContactConfig(
        max_speculative_extension=10.0,
        collision_update_dt=1.0 / 60.0,
    )
    _, state, pipeline, contacts = _build_two_sphere_model(
        device,
        gap=3.0,
        velocity=200.0,
        speculative_config=config,
    )
    pipeline.collide(state, contacts)
    count = contacts.rigid_contact_count.numpy()[0]
    test.assertGreater(count, 0, "Speculative pipeline should detect contacts for fast approaching spheres")


def test_speculative_diverging_no_contacts(test, device):
    """Two spheres moving apart should not produce speculative contacts.

    Body A moves in -X (away from B). The directed gap extension should be
    zero because the approach velocity is negative.
    """
    config = newton.SpeculativeContactConfig(
        max_speculative_extension=10.0,
        collision_update_dt=1.0 / 60.0,
    )
    _, state, pipeline, contacts = _build_two_sphere_model(
        device,
        gap=3.0,
        velocity=-200.0,
        speculative_config=config,
    )
    pipeline.collide(state, contacts)
    count = contacts.rigid_contact_count.numpy()[0]
    test.assertEqual(count, 0, "Speculative pipeline should not detect contacts for diverging spheres")


def test_speculative_max_extension_clamp(test, device):
    """Max speculative extension clamps the gap so very far objects are not detected."""
    config = newton.SpeculativeContactConfig(
        max_speculative_extension=0.01,
        collision_update_dt=1.0 / 60.0,
    )
    _, state, pipeline, contacts = _build_two_sphere_model(
        device,
        gap=3.0,
        velocity=200.0,
        speculative_config=config,
    )
    pipeline.collide(state, contacts)
    count = contacts.rigid_contact_count.numpy()[0]
    test.assertEqual(count, 0, "Max extension clamp should prevent detection of far-away objects")


def test_speculative_dt_override(test, device):
    """Passing dt to collide() overrides the config default."""
    config = newton.SpeculativeContactConfig(
        max_speculative_extension=10.0,
        collision_update_dt=1e-6,
    )
    _, state, pipeline, contacts = _build_two_sphere_model(
        device,
        gap=3.0,
        velocity=200.0,
        speculative_config=config,
    )
    # With the tiny default dt, no contacts
    pipeline.collide(state, contacts)
    count_tiny = contacts.rigid_contact_count.numpy()[0]
    test.assertEqual(count_tiny, 0, "Tiny default dt should not produce contacts")

    # Override with a large dt
    contacts.clear()
    pipeline.collide(state, contacts, dt=1.0 / 60.0)
    count_override = contacts.rigid_contact_count.numpy()[0]
    test.assertGreater(count_override, 0, "Overridden dt should produce contacts")


def test_speculative_angular_velocity(test, device):
    """Angular velocity contributes to speculative extension.

    A spinning body should produce speculative contacts even with zero linear
    velocity if the angular speed bound is large enough.
    """
    builder = newton.ModelBuilder(gravity=0.0)
    builder.rigid_gap = 0.0

    body_a = builder.add_body(xform=wp.transform(wp.vec3(-0.75, 0.0, 0.0)))
    builder.add_shape_box(body_a, hx=0.5, hy=0.5, hz=0.5)
    builder.body_qd[-1] = (0.0, 0.0, 0.0, 0.0, 0.0, 200.0)

    body_b = builder.add_body(xform=wp.transform(wp.vec3(0.75, 0.0, 0.0)))
    builder.add_shape_box(body_b, hx=0.5, hy=0.5, hz=0.5)

    model = builder.finalize(device=device)
    state = model.state()

    config = newton.SpeculativeContactConfig(
        max_speculative_extension=10.0,
        collision_update_dt=1.0 / 60.0,
    )
    pipeline = newton.CollisionPipeline(model, broad_phase="nxn", speculative_config=config)
    contacts = pipeline.contacts()
    pipeline.collide(state, contacts)
    count = contacts.rigid_contact_count.numpy()[0]
    test.assertGreater(count, 0, "Angular velocity should contribute to speculative extension")


def _run_sphere_vs_thin_box_sim(device, speculative_config, num_frames=5):
    """Simulate a fast sphere aimed at a thin static box.

    The sphere (radius 0.25) starts at x = -2 moving at +50 m/s.
    The thin box (half-thickness 0.02 in X) is centred at the origin.
    With dt = 1/60 the sphere travels ~0.83 m per frame -- much more than
    the box thickness (0.04 m), so without speculative contacts the sphere
    tunnels straight through.

    Returns the final X position of the sphere body.
    """
    builder = newton.ModelBuilder(gravity=0.0)
    builder.rigid_gap = 0.0

    sphere_body = builder.add_body(xform=wp.transform(wp.vec3(-2.0, 0.0, 0.0)))
    builder.add_shape_sphere(sphere_body, radius=0.25)
    builder.body_qd[-1] = (50.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    builder.add_shape_box(
        body=-1,
        xform=wp.transform_identity(),
        hx=0.02,
        hy=1.0,
        hz=1.0,
    )

    model = builder.finalize(device=device)

    pipeline = newton.CollisionPipeline(
        model,
        broad_phase="nxn",
        speculative_config=speculative_config,
    )
    contacts = pipeline.contacts()
    solver = newton.solvers.SolverXPBD(model)

    state_0 = model.state()
    state_1 = model.state()
    control = model.control()
    frame_dt = 1.0 / 60.0

    for _ in range(num_frames):
        pipeline.collide(state_0, contacts, dt=frame_dt)
        state_0.clear_forces()
        solver.step(state_0, state_1, control, contacts, frame_dt)
        state_0, state_1 = state_1, state_0

    body_q = state_0.body_q.numpy()
    sphere_x = float(body_q[0][0])
    return sphere_x


def test_speculative_tunneling_without(test, device):
    """Without speculative contacts the sphere tunnels through the thin box."""
    final_x = _run_sphere_vs_thin_box_sim(device, speculative_config=None)
    test.assertGreater(
        final_x,
        0.5,
        f"Sphere should tunnel through the thin box (final x={final_x:.3f})",
    )


def test_speculative_tunneling_with(test, device):
    """With speculative contacts the sphere is stopped by the thin box."""
    config = newton.SpeculativeContactConfig(
        max_speculative_extension=2.0,
        collision_update_dt=1.0 / 60.0,
    )
    final_x = _run_sphere_vs_thin_box_sim(device, speculative_config=config)
    test.assertLess(
        final_x,
        -0.2,
        f"Sphere should be stopped by the thin box (final x={final_x:.3f})",
    )


def _make_thin_wall_mesh(hx=0.02, hy=1.0, hz=1.0):
    """Return a :class:`newton.Mesh` representing a thin box (wall).

    The wall is centred at the origin with half-extents ``(hx, hy, hz)``.
    Winding is CCW when viewed from the +X side so that the face normals
    point outward.
    """
    verts = np.array(
        [
            [-hx, -hy, -hz],
            [hx, -hy, -hz],
            [hx, hy, -hz],
            [-hx, hy, -hz],
            [-hx, -hy, hz],
            [hx, -hy, hz],
            [hx, hy, hz],
            [-hx, hy, hz],
        ],
        dtype=np.float32,
    )
    tris = np.array(
        [
            # -X face
            0,
            3,
            7,
            0,
            7,
            4,
            # +X face
            1,
            5,
            6,
            1,
            6,
            2,
            # -Y face
            0,
            4,
            5,
            0,
            5,
            1,
            # +Y face
            3,
            2,
            6,
            3,
            6,
            7,
            # -Z face
            0,
            1,
            2,
            0,
            2,
            3,
            # +Z face
            4,
            7,
            6,
            4,
            6,
            5,
        ],
        dtype=np.int32,
    )
    return newton.Mesh(verts, tris, compute_inertia=False)


# -- Sphere vs mesh wall (mesh-triangle path) --------------------------------


def _run_sphere_vs_mesh_wall_sim(device, speculative_config, num_frames=5):
    """Simulate a fast sphere aimed at a thin *mesh* wall.

    Same geometry as ``_run_sphere_vs_thin_box_sim`` but the wall is a
    triangle mesh instead of a primitive box, exercising the mesh-triangle
    contact path.
    """
    builder = newton.ModelBuilder(gravity=0.0)
    builder.rigid_gap = 0.0

    sphere_body = builder.add_body(xform=wp.transform(wp.vec3(-2.0, 0.0, 0.0)))
    builder.add_shape_sphere(sphere_body, radius=0.25)
    builder.body_qd[-1] = (50.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    wall_mesh = _make_thin_wall_mesh()
    builder.add_shape_mesh(body=-1, mesh=wall_mesh, xform=wp.transform_identity())

    model = builder.finalize(device=device)

    pipeline = newton.CollisionPipeline(
        model,
        broad_phase="nxn",
        speculative_config=speculative_config,
    )
    contacts = pipeline.contacts()
    solver = newton.solvers.SolverXPBD(model)

    state_0 = model.state()
    state_1 = model.state()
    control = model.control()
    frame_dt = 1.0 / 60.0

    for _ in range(num_frames):
        pipeline.collide(state_0, contacts, dt=frame_dt)
        state_0.clear_forces()
        solver.step(state_0, state_1, control, contacts, frame_dt)
        state_0, state_1 = state_1, state_0

    body_q = state_0.body_q.numpy()
    return float(body_q[0][0])


def test_speculative_tunneling_mesh_wall_without(test, device):
    """Without speculative contacts the sphere tunnels through the mesh wall."""
    final_x = _run_sphere_vs_mesh_wall_sim(device, speculative_config=None)
    test.assertGreater(
        final_x,
        0.5,
        f"Sphere should tunnel through the mesh wall (final x={final_x:.3f})",
    )


def test_speculative_tunneling_mesh_wall_with(test, device):
    """With speculative contacts the sphere is stopped by the mesh wall."""
    config = newton.SpeculativeContactConfig(
        max_speculative_extension=2.0,
        collision_update_dt=1.0 / 60.0,
    )
    final_x = _run_sphere_vs_mesh_wall_sim(device, speculative_config=config)
    test.assertLess(
        final_x,
        -0.2,
        f"Sphere should be stopped by the mesh wall (final x={final_x:.3f})",
    )


# -- Mesh box vs mesh wall (mesh-mesh SDF path) ------------------------------


def _run_mesh_box_vs_mesh_wall_sim(device, speculative_config, num_frames=10):
    """Simulate a fast mesh box aimed at a thin mesh wall.

    Both the projectile and the wall are triangle meshes with SDFs,
    exercising the mesh-mesh SDF contact path.  Same geometry as the
    sphere-vs-primitive-box test (wall half-thickness 0.02 m).

    Uses ``max_resolution=256`` so the SDF has enough voxels across the
    0.04 m wall, and 8 XPBD iterations because mesh-mesh SDF contacts
    have zero effective radii (unlike sphere contacts) and need more
    solver work to fully resolve the large speculative gap.
    """
    builder = newton.ModelBuilder(gravity=0.0)
    builder.rigid_gap = 0.0

    box_mesh = newton.Mesh.create_box(0.25, compute_normals=False, compute_uvs=False)
    box_mesh.build_sdf(device=device, max_resolution=256)

    box_body = builder.add_body(xform=wp.transform(wp.vec3(-2.0, 0.0, 0.0)))
    builder.add_shape_mesh(box_body, mesh=box_mesh)
    builder.body_qd[-1] = (50.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    wall_mesh = _make_thin_wall_mesh()
    wall_mesh.build_sdf(device=device, max_resolution=256)
    builder.add_shape_mesh(body=-1, mesh=wall_mesh, xform=wp.transform_identity())

    model = builder.finalize(device=device)

    pipeline = newton.CollisionPipeline(
        model,
        broad_phase="nxn",
        speculative_config=speculative_config,
    )
    contacts = pipeline.contacts()
    solver = newton.solvers.SolverXPBD(model, iterations=8)

    state_0 = model.state()
    state_1 = model.state()
    control = model.control()
    frame_dt = 1.0 / 60.0

    for _ in range(num_frames):
        pipeline.collide(state_0, contacts, dt=frame_dt)
        state_0.clear_forces()
        solver.step(state_0, state_1, control, contacts, frame_dt)
        state_0, state_1 = state_1, state_0

    body_q = state_0.body_q.numpy()
    return float(body_q[0][0])


def test_speculative_tunneling_mesh_mesh_without(test, device):
    """Without speculative contacts the mesh box tunnels through the mesh wall."""
    final_x = _run_mesh_box_vs_mesh_wall_sim(device, speculative_config=None)
    test.assertGreater(
        final_x,
        0.5,
        f"Mesh box should tunnel through the mesh wall (final x={final_x:.3f})",
    )


def test_speculative_tunneling_mesh_mesh_with(test, device):
    """With speculative contacts the mesh box is stopped by the mesh wall."""
    config = newton.SpeculativeContactConfig(
        max_speculative_extension=2.0,
        collision_update_dt=1.0 / 60.0,
    )
    final_x = _run_mesh_box_vs_mesh_wall_sim(device, speculative_config=config)
    test.assertLess(
        final_x,
        -0.2,
        f"Mesh box should be stopped by the mesh wall (final x={final_x:.3f})",
    )


class TestSpeculativeContacts(unittest.TestCase):
    pass


devices = get_cuda_test_devices()
if not devices:
    devices = [wp.get_device("cpu")]

add_function_test(
    TestSpeculativeContacts,
    "test_speculative_disabled_no_contacts",
    test_speculative_disabled_no_contacts,
    devices=devices,
)
add_function_test(
    TestSpeculativeContacts,
    "test_speculative_enabled_catches_fast_object",
    test_speculative_enabled_catches_fast_object,
    devices=devices,
)
add_function_test(
    TestSpeculativeContacts,
    "test_speculative_diverging_no_contacts",
    test_speculative_diverging_no_contacts,
    devices=devices,
)
add_function_test(
    TestSpeculativeContacts,
    "test_speculative_max_extension_clamp",
    test_speculative_max_extension_clamp,
    devices=devices,
)
add_function_test(
    TestSpeculativeContacts, "test_speculative_dt_override", test_speculative_dt_override, devices=devices
)
add_function_test(
    TestSpeculativeContacts, "test_speculative_angular_velocity", test_speculative_angular_velocity, devices=devices
)
add_function_test(
    TestSpeculativeContacts, "test_speculative_tunneling_without", test_speculative_tunneling_without, devices=devices
)
add_function_test(
    TestSpeculativeContacts, "test_speculative_tunneling_with", test_speculative_tunneling_with, devices=devices
)
add_function_test(
    TestSpeculativeContacts,
    "test_speculative_tunneling_mesh_wall_without",
    test_speculative_tunneling_mesh_wall_without,
    devices=devices,
)
add_function_test(
    TestSpeculativeContacts,
    "test_speculative_tunneling_mesh_wall_with",
    test_speculative_tunneling_mesh_wall_with,
    devices=devices,
)
add_function_test(
    TestSpeculativeContacts,
    "test_speculative_tunneling_mesh_mesh_without",
    test_speculative_tunneling_mesh_mesh_without,
    devices=devices,
)
add_function_test(
    TestSpeculativeContacts,
    "test_speculative_tunneling_mesh_mesh_with",
    test_speculative_tunneling_mesh_mesh_with,
    devices=devices,
)


if __name__ == "__main__":
    wp.clear_kernel_cache()
    unittest.main(verbosity=2)
