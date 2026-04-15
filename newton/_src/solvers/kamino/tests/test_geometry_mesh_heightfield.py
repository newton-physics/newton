# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for mesh and heightfield collision support in Kamino.

Tests the unified collision pipeline, Newton-to-Kamino contact conversion,
and solver integration with mesh and heightfield shapes via the
``ModelKamino.from_newton()`` path.
"""

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.solvers.kamino._src.core.bodies import convert_body_com_to_origin
from newton._src.solvers.kamino._src.core.model import ModelKamino
from newton._src.solvers.kamino._src.geometry.contacts import ContactsKamino, convert_contacts_newton_to_kamino
from newton._src.solvers.kamino._src.geometry.unified import CollisionPipelineUnifiedKamino
from newton._src.solvers.kamino._src.solver_kamino_impl import SolverKaminoImpl
from newton._src.solvers.kamino._src.utils import logger as msg
from newton._src.solvers.kamino.tests import setup_tests, test_context

_cuda_available = wp.is_cuda_available()

###
# Scene Builders
###

SPHERE_RADIUS = 0.25
BOX_HALF = 0.5


def _build_sphere_on_heightfield(sphere_z=None) -> newton.ModelBuilder:
    """Sphere resting on a flat heightfield (elevation = 0 everywhere)."""
    builder = newton.ModelBuilder(up_axis=newton.Axis.Z)

    nrow, ncol = 8, 8
    elevation = np.zeros((nrow, ncol), dtype=np.float32)
    hfield = newton.Heightfield(data=elevation, nrow=nrow, ncol=ncol, hx=4.0, hy=4.0)

    cfg = newton.ModelBuilder.ShapeConfig()
    cfg.margin = 0.0
    cfg.gap = 1e-3

    builder.add_shape_heightfield(heightfield=hfield, cfg=cfg)

    z = sphere_z if sphere_z is not None else SPHERE_RADIUS
    body = builder.add_body(xform=wp.transform(p=(0.0, 0.0, z), q=wp.quat_identity()))
    builder.add_shape_sphere(body, radius=SPHERE_RADIUS, cfg=cfg)

    return builder


def _build_sphere_on_mesh_box(sphere_z=None) -> newton.ModelBuilder:
    """Sphere resting on a box-shaped triangle mesh."""
    builder = newton.ModelBuilder(up_axis=newton.Axis.Z)

    mesh = newton.Mesh.create_box(BOX_HALF, BOX_HALF, BOX_HALF)
    cfg = newton.ModelBuilder.ShapeConfig()
    cfg.margin = 0.0
    cfg.gap = 0.02

    # Static mesh box centered at origin (top face at z = 0.5)
    builder.add_shape_mesh(body=-1, mesh=mesh, cfg=cfg)

    # Sphere slightly penetrating the mesh box top face
    z = sphere_z if sphere_z is not None else (BOX_HALF + SPHERE_RADIUS - 0.005)
    body = builder.add_body(xform=wp.transform(p=(0.0, 0.0, z), q=wp.quat_identity()))
    builder.add_shape_sphere(body, radius=SPHERE_RADIUS, cfg=cfg)

    return builder


def _build_box_on_heightfield() -> newton.ModelBuilder:
    """Box resting on a flat heightfield."""
    builder = newton.ModelBuilder(up_axis=newton.Axis.Z)

    nrow, ncol = 8, 8
    elevation = np.zeros((nrow, ncol), dtype=np.float32)
    hfield = newton.Heightfield(data=elevation, nrow=nrow, ncol=ncol, hx=4.0, hy=4.0)

    cfg = newton.ModelBuilder.ShapeConfig()
    cfg.margin = 0.0
    cfg.gap = 1e-3

    builder.add_shape_heightfield(heightfield=hfield, cfg=cfg)

    body = builder.add_body(xform=wp.transform(p=(0.0, 0.0, BOX_HALF), q=wp.quat_identity()))
    builder.add_shape_box(body, hx=BOX_HALF, hy=BOX_HALF, hz=BOX_HALF, cfg=cfg)

    return builder


def _build_mixed_scene() -> newton.ModelBuilder:
    """Scene with both primitive shapes and a mesh — sphere on mesh box + sphere on ground plane."""
    builder = newton.ModelBuilder(up_axis=newton.Axis.Z)

    mesh = newton.Mesh.create_box(BOX_HALF, BOX_HALF, BOX_HALF)
    cfg = newton.ModelBuilder.ShapeConfig()
    cfg.margin = 0.0
    cfg.gap = 0.02

    # Static mesh box at x=2
    builder.add_shape_mesh(
        body=-1,
        mesh=mesh,
        xform=wp.transform(p=(2.0, 0.0, 0.0), q=wp.quat_identity()),
        cfg=cfg,
    )

    # Ground plane
    builder.add_ground_plane(cfg=cfg)

    # Sphere on ground plane at origin (touching)
    body_a = builder.add_body(xform=wp.transform(p=(0.0, 0.0, SPHERE_RADIUS), q=wp.quat_identity()))
    builder.add_shape_sphere(body_a, radius=SPHERE_RADIUS, cfg=cfg)

    # Sphere on mesh box at x=2 (slightly penetrating)
    body_b = builder.add_body(
        xform=wp.transform(p=(2.0, 0.0, BOX_HALF + SPHERE_RADIUS - 0.005), q=wp.quat_identity()),
    )
    builder.add_shape_sphere(body_b, radius=SPHERE_RADIUS, cfg=cfg)

    return builder


def _build_heightfield_terrain() -> newton.ModelBuilder:
    """Sphere on a non-flat heightfield with sine-wave terrain."""
    builder = newton.ModelBuilder(up_axis=newton.Axis.Z)

    nrow, ncol = 20, 20
    x = np.linspace(-2.0, 2.0, ncol)
    y = np.linspace(-2.0, 2.0, nrow)
    xx, yy = np.meshgrid(x, y)
    elevation = (0.1 * np.sin(xx) * np.cos(yy)).astype(np.float32)
    # Elevation at center (0,0) ≈ 0
    hfield = newton.Heightfield(data=elevation, nrow=nrow, ncol=ncol, hx=2.0, hy=2.0)

    cfg = newton.ModelBuilder.ShapeConfig()
    cfg.margin = 0.0
    cfg.gap = 1e-3

    builder.add_shape_heightfield(heightfield=hfield, cfg=cfg)

    # Sphere at the center, slightly above terrain surface (elevation ≈ 0)
    body = builder.add_body(xform=wp.transform(p=(0.0, 0.0, SPHERE_RADIUS), q=wp.quat_identity()))
    builder.add_shape_sphere(body, radius=SPHERE_RADIUS, cfg=cfg)

    return builder


def _build_multi_world_heightfield(num_worlds: int = 3) -> newton.ModelBuilder:
    """Multi-world scene, each with sphere-on-flat-heightfield."""
    single = newton.ModelBuilder(up_axis=newton.Axis.Z)

    nrow, ncol = 8, 8
    elevation = np.zeros((nrow, ncol), dtype=np.float32)
    hfield = newton.Heightfield(data=elevation, nrow=nrow, ncol=ncol, hx=4.0, hy=4.0)

    cfg = newton.ModelBuilder.ShapeConfig()
    cfg.margin = 0.0
    cfg.gap = 1e-3

    body = single.add_body(xform=wp.transform(p=(0.0, 0.0, SPHERE_RADIUS), q=wp.quat_identity()))
    single.add_shape_sphere(body, radius=SPHERE_RADIUS, cfg=cfg)

    multi = newton.ModelBuilder(up_axis=newton.Axis.Z)
    for _ in range(num_worlds):
        multi.add_world(single)

    multi.add_shape_heightfield(heightfield=hfield, cfg=cfg)

    return multi


###
# Helpers
###


def _finalize_and_get_kamino(builder, device):
    """Finalize Newton model, create Kamino model, data, and state."""
    newton_model = builder.finalize(device=device)
    model = ModelKamino.from_newton(newton_model)
    data = model.data(device=device)
    state = model.state(device=device)
    return newton_model, model, data, state


def _run_unified_pipeline(model, data, state, device):
    """Create unified pipeline, allocate contacts, run collision detection."""
    num_worlds = model.size.num_worlds
    pipeline = CollisionPipelineUnifiedKamino(
        model=model,
        broadphase="nxn",
        device=device,
    )
    contacts = ContactsKamino(capacity=[4096] * num_worlds, device=device)
    contacts.clear()
    pipeline.collide(data, state, contacts)
    return contacts


def _run_newton_cd_and_convert(newton_model, device):
    """Run Newton's CD pipeline and convert contacts to Kamino format."""
    # Normalize shape_world for single-world models
    if newton_model.world_count == 1:
        sw = newton_model.shape_world.numpy()
        if np.any(sw < 0):
            sw[sw < 0] = 0
            newton_model.shape_world.assign(sw)

    state = newton_model.state()
    newton.eval_fk(newton_model, newton_model.joint_q, newton_model.joint_qd, state)
    newton_contacts = newton_model.collide(state)

    nc = int(newton_contacts.rigid_contact_count.numpy()[0])
    kamino_contacts = ContactsKamino(capacity=[max(nc + 64, 256)], device=device)
    convert_contacts_newton_to_kamino(newton_model, state, newton_contacts, kamino_contacts)
    wp.synchronize()

    return kamino_contacts, nc


def _step_with_newton_cd(builder, device, num_steps=200, dt=0.005):
    """Build scene, step solver with Newton CD, return final body positions (COM frame)."""
    newton_model = builder.finalize(device=device)

    # Normalize shape_world
    if newton_model.world_count == 1:
        sw = newton_model.shape_world.numpy()
        if np.any(sw < 0):
            sw[sw < 0] = 0
            newton_model.shape_world.assign(sw)

    newton_model.set_gravity((0.0, 0.0, -9.81))

    model = ModelKamino.from_newton(newton_model)
    model.time.set_uniform_timestep(dt)

    state_p = model.state(device=device)
    state_n = model.state(device=device)
    control = model.control(device=device)

    per_world = max(1024, newton_model.rigid_contact_max // max(newton_model.world_count, 1))
    contacts = ContactsKamino(capacity=[per_world], device=device)

    solver = SolverKaminoImpl(model=model, contacts=contacts)
    solver.reset(state_out=state_n)
    state_p.copy_from(state_n)

    newton_state = newton_model.state()
    newton_contacts = newton_model.contacts()

    for _ in range(num_steps):
        state_p.copy_from(state_n)

        convert_body_com_to_origin(
            body_com=model.bodies.i_r_com_i,
            body_q_com=state_p.q_i,
            body_q=newton_state.body_q,
        )
        newton_model.collide(newton_state, newton_contacts)
        convert_contacts_newton_to_kamino(newton_model, newton_state, newton_contacts, contacts)

        solver.step(
            state_in=state_p,
            state_out=state_n,
            control=control,
            contacts=contacts,
            detector=None,
        )

    return state_n.q_i.numpy()


###
# Test Classes
###


class TestUnifiedPipelineMeshHeightfield(unittest.TestCase):
    """Tests Kamino unified collision pipeline with heightfield shapes via from_newton().

    The unified pipeline handles heightfield-vs-convex contacts directly.
    Each test verifies exact contact counts, contact positions, normals,
    and signed distances against analytically known values.
    """

    def setUp(self):
        if not test_context.setup_done:
            setup_tests(clear_cache=False)
        self.default_device = wp.get_device(test_context.device)
        self.verbose = test_context.verbose
        if self.verbose:
            msg.set_log_level(msg.LogLevel.DEBUG)
        else:
            msg.reset_log_level()

    def tearDown(self):
        self.default_device = None
        if self.verbose:
            msg.reset_log_level()

    def test_01_sphere_on_flat_heightfield(self):
        """Sphere touching flat heightfield: 2 contacts (one per cell triangle), normal=(0,0,1), position at z=0."""
        _, model, data, state = _finalize_and_get_kamino(_build_sphere_on_heightfield(), self.default_device)
        contacts = _run_unified_pipeline(model, data, state, self.default_device)

        nc = int(contacts.model_active_contacts.numpy()[0])
        # A sphere centered over a heightfield cell touches both triangles in that cell,
        # producing 2 contacts on a flat surface.
        self.assertEqual(nc, 2, f"Sphere-on-flat-heightfield should produce 2 contacts (1 per triangle), got {nc}")

        gapfunc = contacts.gapfunc.numpy()[:nc]
        pos_a = contacts.position_A.numpy()[:nc]

        for i in range(nc):
            normal = gapfunc[i, :3]
            signed_dist = gapfunc[i, 3]

            # Normal must be (0, 0, 1) for a flat heightfield
            np.testing.assert_allclose(normal, [0.0, 0.0, 1.0], atol=0.05, err_msg=f"Contact {i}: normal must point up")

            # Signed distance: sphere center is at z=R, surface at z=0, so distance ≈ 0
            self.assertLessEqual(
                signed_dist, 0.01, f"Contact {i}: signed distance should be near zero, got {signed_dist}"
            )
            self.assertGreaterEqual(
                signed_dist, -0.01, f"Contact {i}: signed distance should be near zero, got {signed_dist}"
            )

            # Contact point on the heightfield side (position_A) should be near (0, 0, 0)
            np.testing.assert_allclose(
                pos_a[i, 2],
                0.0,
                atol=0.02,
                err_msg=f"Contact {i}: heightfield contact z should be ≈ 0, got {pos_a[i, 2]}",
            )

    def test_02_box_on_flat_heightfield(self):
        """Box resting on flat heightfield: >=4 contacts, all at z ≈ 0."""
        _, model, data, state = _finalize_and_get_kamino(_build_box_on_heightfield(), self.default_device)
        contacts = _run_unified_pipeline(model, data, state, self.default_device)

        nc = int(contacts.model_active_contacts.numpy()[0])
        self.assertGreaterEqual(nc, 4, f"Box face on heightfield should produce >=4 contacts, got {nc}")

        gapfunc = contacts.gapfunc.numpy()[:nc]
        for i in range(nc):
            normal = gapfunc[i, :3]
            signed_dist = gapfunc[i, 3]

            # All normals should point up
            np.testing.assert_allclose(
                normal, [0.0, 0.0, 1.0], atol=0.1, err_msg=f"Contact {i}: normal should point up"
            )
            # Signed distance near zero (touching)
            self.assertLessEqual(
                abs(signed_dist), 0.02, f"Contact {i}: distance should be near zero, got {signed_dist}"
            )

        # All contact positions should be at z ≈ 0 and within the box footprint
        pos_a = contacts.position_A.numpy()[:nc]
        for i in range(nc):
            self.assertAlmostEqual(pos_a[i, 2], 0.0, places=1, msg=f"Contact {i}: z should be ≈ 0")
            self.assertLessEqual(abs(pos_a[i, 0]), BOX_HALF + 0.01, f"Contact {i}: x out of box bounds")
            self.assertLessEqual(abs(pos_a[i, 1]), BOX_HALF + 0.01, f"Contact {i}: y out of box bounds")

    def test_03_heightfield_terrain(self):
        """Sphere on sine-wave terrain: contacts generated, normal has upward component."""
        _, model, data, state = _finalize_and_get_kamino(_build_heightfield_terrain(), self.default_device)
        contacts = _run_unified_pipeline(model, data, state, self.default_device)

        nc = int(contacts.model_active_contacts.numpy()[0])
        self.assertGreaterEqual(nc, 1, "Sphere on terrain must produce contacts")

        gapfunc = contacts.gapfunc.numpy()[:nc]
        for i in range(nc):
            normal = gapfunc[i, :3]
            norm_len = np.linalg.norm(normal)
            self.assertTrue(np.isclose(norm_len, 1.0, atol=1e-4), f"Contact {i}: normal not unit")
            # On gently undulating terrain near the center, normal should still be mostly upward
            self.assertGreater(normal[2], 0.7, f"Contact {i}: normal z={normal[2]}, expected mostly up")

    def test_04_multi_world_heightfield(self):
        """Multi-world sphere-on-heightfield: each world gets exactly 2 contacts (1 per triangle)."""
        num_worlds = 3
        _, model, data, state = _finalize_and_get_kamino(
            _build_multi_world_heightfield(num_worlds), self.default_device
        )
        contacts = _run_unified_pipeline(model, data, state, self.default_device)

        nc = int(contacts.model_active_contacts.numpy()[0])
        expected_total = 2 * num_worlds  # 2 contacts per sphere (one per cell triangle)
        self.assertEqual(nc, expected_total, f"Expected {expected_total} contacts (2 per world), got {nc}")

        world_counts = contacts.world_active_contacts.numpy()[:num_worlds]
        for w in range(num_worlds):
            self.assertEqual(int(world_counts[w]), 2, f"World {w}: expected 2 contacts, got {world_counts[w]}")

    def test_05_no_contacts_when_separated(self):
        """Sphere far above heightfield must produce zero contacts."""
        _, model, data, state = _finalize_and_get_kamino(
            _build_sphere_on_heightfield(sphere_z=SPHERE_RADIUS + 1.0), self.default_device
        )
        contacts = _run_unified_pipeline(model, data, state, self.default_device)

        nc = int(contacts.model_active_contacts.numpy()[0])
        self.assertEqual(nc, 0, f"Separated sphere should produce 0 contacts, got {nc}")


class TestNewtonCollisionPathMeshHeightfield(unittest.TestCase):
    """Tests Newton model.collide() -> convert_contacts_newton_to_kamino() path.

    This is the primary path for mesh collisions in Kamino.  Tests verify
    that contact data survives conversion with correct body indices,
    physically plausible normals, positions, and signed distances.
    """

    def setUp(self):
        if not test_context.setup_done:
            setup_tests(clear_cache=False)
        self.default_device = wp.get_device(test_context.device)
        self.verbose = test_context.verbose
        if self.verbose:
            msg.set_log_level(msg.LogLevel.DEBUG)
        else:
            msg.reset_log_level()

    def tearDown(self):
        self.default_device = None
        if self.verbose:
            msg.reset_log_level()

    def test_01_newton_to_kamino_heightfield(self):
        """Heightfield contacts survive Newton->Kamino conversion with correct geometry."""
        builder = _build_sphere_on_heightfield()
        newton_model = builder.finalize(device=self.default_device)

        kamino_contacts, newton_count = _run_newton_cd_and_convert(newton_model, self.default_device)
        self.assertGreater(newton_count, 0, "Newton must produce contacts for sphere on heightfield")

        nc = int(kamino_contacts.model_active_contacts.numpy()[0])
        self.assertGreater(nc, 0, "Conversion must preserve contacts")

        gapfunc = kamino_contacts.gapfunc.numpy()[:nc]
        bid_AB = kamino_contacts.bid_AB.numpy()[:nc]
        pos_a = kamino_contacts.position_A.numpy()[:nc]

        for i in range(nc):
            # A/B convention: bid_B must be dynamic (>= 0)
            self.assertGreaterEqual(int(bid_AB[i, 1]), 0, f"Contact {i}: bid_B must be >= 0")

            # Normal must be approximately upward
            normal = gapfunc[i, :3]
            np.testing.assert_allclose(
                normal, [0.0, 0.0, 1.0], atol=0.1, err_msg=f"Contact {i}: normal should point up"
            )

            # Contact position on heightfield should be near z=0
            self.assertAlmostEqual(pos_a[i, 2], 0.0, places=1, msg=f"Contact {i}: position z should be ≈ 0")

            # Signed distance should be non-positive (penetrating or touching)
            self.assertLessEqual(gapfunc[i, 3], 0.01, f"Contact {i}: distance should be <= 0")

    @unittest.skipUnless(_cuda_available, "mesh collision requires CUDA")
    def test_02_newton_to_kamino_mesh(self):
        """Mesh box contacts survive Newton->Kamino conversion with correct geometry."""
        builder = _build_sphere_on_mesh_box()
        newton_model = builder.finalize(device=self.default_device)

        kamino_contacts, newton_count = _run_newton_cd_and_convert(newton_model, self.default_device)
        self.assertGreater(newton_count, 0, "Newton must produce contacts for sphere on mesh")

        nc = int(kamino_contacts.model_active_contacts.numpy()[0])
        self.assertGreater(nc, 0, "Conversion must preserve contacts")

        gapfunc = kamino_contacts.gapfunc.numpy()[:nc]
        bid_AB = kamino_contacts.bid_AB.numpy()[:nc]
        pos_a = kamino_contacts.position_A.numpy()[:nc]

        for i in range(nc):
            self.assertGreaterEqual(int(bid_AB[i, 1]), 0, f"Contact {i}: bid_B must be >= 0")

            normal = gapfunc[i, :3]
            # Normal should point mostly upward (sphere on top of mesh box)
            self.assertGreater(normal[2], 0.5, f"Contact {i}: normal z={normal[2]}, expected upward")

            # Contact on mesh top face should be near z = BOX_HALF = 0.5
            self.assertAlmostEqual(pos_a[i, 2], BOX_HALF, delta=0.1, msg=f"Contact {i}: z should be near mesh top face")

    @unittest.skipUnless(_cuda_available, "mesh collision requires CUDA")
    def test_03_newton_to_kamino_mixed(self):
        """Mixed scene: both primitive-plane and mesh contacts converted correctly."""
        builder = _build_mixed_scene()
        newton_model = builder.finalize(device=self.default_device)

        kamino_contacts, newton_count = _run_newton_cd_and_convert(newton_model, self.default_device)
        self.assertGreater(newton_count, 0, "Newton must produce contacts")

        nc = int(kamino_contacts.model_active_contacts.numpy()[0])
        # Must have contacts for both spheres (one on plane, one on mesh)
        self.assertGreaterEqual(nc, 2, f"Mixed scene should have >=2 contacts, got {nc}")

        # Verify we got contacts involving different bodies
        bid_AB = kamino_contacts.bid_AB.numpy()[:nc]
        dynamic_bodies = {int(bid_AB[i, 1]) for i in range(nc)}
        self.assertGreaterEqual(
            len(dynamic_bodies), 2, f"Should have contacts for >=2 different bodies, got {dynamic_bodies}"
        )

    @unittest.skipUnless(_cuda_available, "mesh collision requires CUDA")
    def test_04_no_contacts_when_separated(self):
        """Sphere far above mesh box must produce zero contacts after conversion."""
        builder = _build_sphere_on_mesh_box(sphere_z=BOX_HALF + SPHERE_RADIUS + 2.0)
        newton_model = builder.finalize(device=self.default_device)

        kamino_contacts, newton_count = _run_newton_cd_and_convert(newton_model, self.default_device)
        self.assertEqual(newton_count, 0, "Separated shapes should produce 0 Newton contacts")

        nc = int(kamino_contacts.model_active_contacts.numpy()[0])
        self.assertEqual(nc, 0, "Separated shapes should produce 0 Kamino contacts")


@unittest.skipUnless(_cuda_available, "Kamino solver requires CUDA")
class TestSolverWithMeshHeightfield(unittest.TestCase):
    """Tests Kamino solver produces physically correct behavior with mesh/heightfield contacts.

    Drops objects onto surfaces and verifies they come to rest at the
    analytically expected height rather than falling through or floating.
    """

    def setUp(self):
        if not test_context.setup_done:
            setup_tests(clear_cache=False)
        self.default_device = wp.get_device(test_context.device)
        self.verbose = test_context.verbose
        if self.verbose:
            msg.set_log_level(msg.LogLevel.DEBUG)
        else:
            msg.reset_log_level()

    def tearDown(self):
        self.default_device = None
        if self.verbose:
            msg.reset_log_level()

    def test_01_sphere_falls_onto_heightfield(self):
        """Sphere dropped from 0.5m above flat heightfield must rest at z ≈ SPHERE_RADIUS."""
        drop_height = SPHERE_RADIUS + 0.5
        builder = _build_sphere_on_heightfield(sphere_z=drop_height)

        q_i = _step_with_newton_cd(builder, self.default_device, num_steps=500, dt=0.005)

        # Sphere COM should be near SPHERE_RADIUS (resting on z=0 surface)
        sphere_z = float(q_i[0, 2])
        expected_z = SPHERE_RADIUS  # 0.25

        # Tight tolerance: must be within 5cm of expected rest height
        self.assertAlmostEqual(
            sphere_z,
            expected_z,
            delta=0.05,
            msg=f"Sphere should rest at z≈{expected_z}, got z={sphere_z:.4f}",
        )

    def test_02_sphere_falls_onto_mesh_box(self):
        """Sphere dropped from 0.5m above mesh box must rest at z ≈ BOX_HALF + SPHERE_RADIUS."""
        drop_height = BOX_HALF + SPHERE_RADIUS + 0.5
        builder = _build_sphere_on_mesh_box(sphere_z=drop_height)

        q_i = _step_with_newton_cd(builder, self.default_device, num_steps=500, dt=0.005)

        sphere_z = float(q_i[0, 2])
        expected_z = BOX_HALF + SPHERE_RADIUS  # 0.75

        self.assertAlmostEqual(
            sphere_z,
            expected_z,
            delta=0.05,
            msg=f"Sphere should rest at z≈{expected_z}, got z={sphere_z:.4f}",
        )

    def test_03_sphere_does_not_fall_through_heightfield(self):
        """Sphere placed at contact boundary must not fall below the surface after stepping."""
        builder = _build_sphere_on_heightfield(sphere_z=SPHERE_RADIUS)

        q_i = _step_with_newton_cd(builder, self.default_device, num_steps=200, dt=0.005)

        sphere_z = float(q_i[0, 2])
        # With gravity pulling down, if contacts aren't working the sphere goes negative.
        # Allow a small margin for numerical settling but it must stay above the surface.
        self.assertGreater(
            sphere_z,
            SPHERE_RADIUS - 0.02,
            f"Sphere fell through heightfield: z={sphere_z:.4f}, min allowed={SPHERE_RADIUS - 0.02}",
        )


if __name__ == "__main__":
    unittest.main()
