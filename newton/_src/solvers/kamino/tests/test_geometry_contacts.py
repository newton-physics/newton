# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""
Unit tests for `geometry/contacts.py`.

Tests all components of the ContactsKamino data types and operations.
"""

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.sim import ModelBuilder
from newton._src.sim.contacts import Contacts
from newton._src.solvers.kamino._src.core.types import int32, mat33f, vec3f
from newton._src.solvers.kamino._src.geometry.contacts import (
    ContactMode,
    ContactsKamino,
    convert_contacts_kamino_to_newton,
    convert_contacts_newton_to_kamino,
    make_contact_frame_xnorm,
    make_contact_frame_znorm,
)
from newton._src.solvers.kamino._src.utils import logger as msg
from newton._src.solvers.kamino.tests import setup_tests, test_context

###
# Builders
###


def build_test_system(
    builder: ModelBuilder | None = None,
    ground: bool = True,
) -> ModelBuilder:
    """
    Constructs a nunchaku model: two boxes connected by a sphere via ball joints.

    Three bodies (two boxes + one sphere) connected by spherical joints,
    optionally resting on a ground plane.  Produces 9 contacts with the
    ground (4 per box + 1 sphere).

    Args:
        builder: An optional existing model builder to populate.
            If ``None``, a new builder is created.
        ground: Whether to add a static ground plane.

    Returns:
        The populated :class:`ModelBuilder`.
    """
    if builder is None:
        builder = ModelBuilder()

    d, w, h, r = 0.5, 0.1, 0.1, 0.05
    no_gap = ModelBuilder.ShapeConfig(gap=0.0)

    b0 = builder.add_link()
    builder.add_shape_box(b0, hx=d / 2, hy=w / 2, hz=h / 2, cfg=no_gap)

    b1 = builder.add_link()
    builder.add_shape_sphere(b1, radius=r, cfg=no_gap)

    b2 = builder.add_link()
    builder.add_shape_box(b2, hx=d / 2, hy=w / 2, hz=h / 2, cfg=no_gap)

    j0 = builder.add_joint_ball(
        parent=-1,
        child=b0,
        parent_xform=wp.transform(p=wp.vec3(d / 2, 0.0, h / 2), q=wp.quat_identity()),
        child_xform=wp.transform(p=wp.vec3(0.0, 0.0, 0.0), q=wp.quat_identity()),
    )
    j1 = builder.add_joint_ball(
        parent=b0,
        child=b1,
        parent_xform=wp.transform(p=wp.vec3(d / 2, 0.0, 0.0), q=wp.quat_identity()),
        child_xform=wp.transform(p=wp.vec3(-r, 0.0, 0.0), q=wp.quat_identity()),
    )
    j2 = builder.add_joint_ball(
        parent=b1,
        child=b2,
        parent_xform=wp.transform(p=wp.vec3(r, 0.0, 0.0), q=wp.quat_identity()),
        child_xform=wp.transform(p=wp.vec3(-d / 2, 0.0, 0.0), q=wp.quat_identity()),
    )
    builder.add_articulation([j0, j1, j2])

    if ground:
        builder.add_ground_plane()

    return builder


###
# Kernels
###


@wp.kernel
def _compute_contact_frame_znorm(
    # Inputs:
    normal: wp.array[vec3f],
    # Outputs:
    frame: wp.array[mat33f],
):
    tid = wp.tid()
    frame[tid] = make_contact_frame_znorm(normal[tid])


@wp.kernel
def _compute_contact_frame_xnorm(
    # Inputs:
    normal: wp.array[vec3f],
    # Outputs:
    frame: wp.array[mat33f],
):
    tid = wp.tid()
    frame[tid] = make_contact_frame_xnorm(normal[tid])


@wp.kernel
def _compute_contact_mode(
    # Inputs:
    velocity: wp.array[vec3f],
    # Outputs:
    mode: wp.array[int32],
):
    tid = wp.tid()
    mode[tid] = wp.static(ContactMode.make_compute_mode_func())(velocity[tid])


###
# Launchers
###


def compute_contact_frame_znorm(normal: wp.array, frame: wp.array, num_threads: int = 1):
    wp.launch(
        _compute_contact_frame_znorm,
        dim=num_threads,
        inputs=[normal],
        outputs=[frame],
        device=normal.device,
    )


def compute_contact_frame_xnorm(normal: wp.array, frame: wp.array, num_threads: int = 1):
    wp.launch(
        _compute_contact_frame_xnorm,
        dim=num_threads,
        inputs=[normal],
        outputs=[frame],
        device=normal.device,
    )


def compute_contact_mode(velocity: wp.array, mode: wp.array, num_threads: int = 1):
    wp.launch(
        _compute_contact_mode,
        dim=num_threads,
        inputs=[velocity],
        outputs=[mode],
        device=velocity.device,
    )


###
# Tests
###


class TestGeometryContactFrames(unittest.TestCase):
    def setUp(self):
        if not test_context.setup_done:
            setup_tests(clear_cache=False)
        self.default_device = wp.get_device(test_context.device)
        self.verbose = test_context.verbose  # Set to True for detailed output

        # Set debug-level logging to print verbose test output to console
        if self.verbose:
            msg.info("\n")  # Add newline before test output for better readability
            msg.set_log_level(msg.LogLevel.DEBUG)
        else:
            msg.reset_log_level()

    def tearDown(self):
        self.default_device = None
        if self.verbose:
            msg.reset_log_level()

    def test_01_make_contact_frame_znorm(self):
        # Create a normal vectors
        test_normals: list[vec3f] = []

        # Add normals for which to test contact frame creation
        test_normals.append(vec3f(1.0, 0.0, 0.0))
        test_normals.append(vec3f(0.0, 1.0, 0.0))
        test_normals.append(vec3f(0.0, 0.0, 1.0))
        test_normals.append(vec3f(-1.0, 0.0, 0.0))
        test_normals.append(vec3f(0.0, -1.0, 0.0))
        test_normals.append(vec3f(0.0, 0.0, -1.0))

        # Create the input output arrays
        normals = wp.array(test_normals, dtype=vec3f, device=self.default_device)
        frames = wp.zeros(shape=(len(test_normals),), dtype=mat33f, device=self.default_device)

        # Compute the contact frames
        compute_contact_frame_znorm(normal=normals, frame=frames, num_threads=len(test_normals))
        if self.verbose:
            print(f"normals:\n{normals}\n")
            print(f"frames:\n{frames}\n")

        # Extract numpy arrays for comparison
        frames_np = frames.numpy()

        # Check determinants of each frame
        for i in range(len(test_normals)):
            det = np.linalg.det(frames_np[i])
            self.assertTrue(np.isclose(det, 1.0, atol=1e-6))

        # Check each primitive frame
        self.assertTrue(
            np.allclose(frames_np[0], np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]), atol=1e-6)
        )
        self.assertTrue(
            np.allclose(frames_np[1], np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]]), atol=1e-6)
        )
        self.assertTrue(
            np.allclose(frames_np[2], np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]), atol=1e-6)
        )
        self.assertTrue(
            np.allclose(frames_np[3], np.array([[0.0, 0.0, -1.0], [1.0, 0.0, 0.0], [0.0, -1.0, 0.0]]), atol=1e-6)
        )
        self.assertTrue(
            np.allclose(frames_np[4], np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]]), atol=1e-6)
        )
        self.assertTrue(
            np.allclose(frames_np[5], np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]]), atol=1e-6)
        )

    def test_02_make_contact_frame_xnorm(self):
        # Create a normal vectors
        test_normals: list[vec3f] = []

        # Add normals for which to test contact frame creation
        test_normals.append(vec3f(1.0, 0.0, 0.0))
        test_normals.append(vec3f(0.0, 1.0, 0.0))
        test_normals.append(vec3f(0.0, 0.0, 1.0))
        test_normals.append(vec3f(-1.0, 0.0, 0.0))
        test_normals.append(vec3f(0.0, -1.0, 0.0))
        test_normals.append(vec3f(0.0, 0.0, -1.0))

        # Create the input output arrays
        normals = wp.array(test_normals, dtype=vec3f, device=self.default_device)
        frames = wp.zeros(shape=(len(test_normals),), dtype=mat33f, device=self.default_device)

        # Compute the contact frames
        compute_contact_frame_xnorm(normal=normals, frame=frames, num_threads=len(test_normals))
        if self.verbose:
            print(f"normals:\n{normals}\n")
            print(f"frames:\n{frames}\n")

        # Extract numpy arrays for comparison
        frames_np = frames.numpy()

        # Check determinants of each frame
        for i in range(len(test_normals)):
            det = np.linalg.det(frames_np[i])
            self.assertTrue(np.isclose(det, 1.0, atol=1e-6))


class TestGeometryContactMode(unittest.TestCase):
    def setUp(self):
        if not test_context.setup_done:
            setup_tests(clear_cache=False)
        self.default_device = wp.get_device(test_context.device)
        self.verbose = test_context.verbose  # Set to True for detailed output

        # Set debug-level logging to print verbose test output to console
        if self.verbose:
            msg.info("\n")  # Add newline before test output for better readability
            msg.set_log_level(msg.LogLevel.DEBUG)
        else:
            msg.reset_log_level()

    def tearDown(self):
        self.default_device = None
        if self.verbose:
            msg.reset_log_level()

    def test_01_contact_mode_opening(self):
        v_input = wp.array([vec3f(0.0, 0.0, 0.01)], dtype=vec3f, device=self.default_device)
        mode_output = wp.zeros(shape=(1,), dtype=int32, device=self.default_device)
        compute_contact_mode(velocity=v_input, mode=mode_output, num_threads=1)
        mode_int32 = mode_output.numpy()[0]
        mode = ContactMode(int(mode_int32))
        msg.info(f"mode: {mode} (int: {int(mode_int32)})")
        self.assertEqual(mode, ContactMode.OPENING)

    def test_02_contact_mode_sticking(self):
        v_input = wp.array([vec3f(0.0, 0.0, 1e-7)], dtype=vec3f, device=self.default_device)
        mode_output = wp.zeros(shape=(1,), dtype=int32, device=self.default_device)
        compute_contact_mode(velocity=v_input, mode=mode_output, num_threads=1)
        mode_int32 = mode_output.numpy()[0]
        mode = ContactMode(int(mode_int32))
        msg.info(f"mode: {mode} (int: {int(mode_int32)})")
        self.assertEqual(mode, ContactMode.STICKING)

    def test_03_contact_mode_slipping(self):
        v_input = wp.array([vec3f(0.1, 0.0, 0.0)], dtype=vec3f, device=self.default_device)
        mode_output = wp.zeros(shape=(1,), dtype=int32, device=self.default_device)
        compute_contact_mode(velocity=v_input, mode=mode_output, num_threads=1)
        mode_int32 = mode_output.numpy()[0]
        mode = ContactMode(int(mode_int32))
        msg.info(f"mode: {mode} (int: {int(mode_int32)})")
        self.assertEqual(mode, ContactMode.SLIDING)


class TestGeometryContacts(unittest.TestCase):
    def setUp(self):
        if not test_context.setup_done:
            setup_tests(clear_cache=False)
        self.default_device = wp.get_device(test_context.device)
        self.verbose = test_context.verbose  # Set to True for detailed output

        # Set debug-level logging to print verbose test output to console
        if self.verbose:
            msg.info("\n")  # Add newline before test output for better readability
            msg.set_log_level(msg.LogLevel.DEBUG)
        else:
            msg.reset_log_level()

    def tearDown(self):
        self.default_device = None
        if self.verbose:
            msg.reset_log_level()

    def test_single_default_allocation(self):
        contacts = ContactsKamino(capacity=0, device=self.default_device)
        self.assertEqual(contacts.model_max_contacts_host, contacts.default_max_world_contacts)
        self.assertEqual(contacts.world_max_contacts_host[0], contacts.default_max_world_contacts)
        self.assertEqual(len(contacts.model_max_contacts), 1)
        self.assertEqual(len(contacts.model_active_contacts), 1)
        self.assertEqual(len(contacts.world_max_contacts), 1)
        self.assertEqual(len(contacts.world_active_contacts), 1)
        self.assertEqual(contacts.model_max_contacts.numpy()[0], contacts.default_max_world_contacts)
        self.assertEqual(contacts.model_active_contacts.numpy()[0], 0)
        self.assertEqual(contacts.world_max_contacts.numpy()[0], contacts.default_max_world_contacts)
        self.assertEqual(contacts.world_active_contacts.numpy()[0], 0)
        self.assertEqual(len(contacts.wid), contacts.default_max_world_contacts)
        self.assertEqual(len(contacts.cid), contacts.default_max_world_contacts)
        self.assertEqual(len(contacts.gid_AB), contacts.default_max_world_contacts)
        self.assertEqual(len(contacts.bid_AB), contacts.default_max_world_contacts)
        self.assertEqual(len(contacts.position_A), contacts.default_max_world_contacts)
        self.assertEqual(len(contacts.position_B), contacts.default_max_world_contacts)
        self.assertEqual(len(contacts.gapfunc), contacts.default_max_world_contacts)
        self.assertEqual(len(contacts.frame), contacts.default_max_world_contacts)
        self.assertEqual(len(contacts.material), contacts.default_max_world_contacts)

    def test_multiple_default_allocations(self):
        num_worlds = 10
        capacities = [0] * num_worlds
        contacts = ContactsKamino(capacity=capacities, device=self.default_device)

        model_max_contacts = contacts.model_max_contacts.numpy()
        model_active_contacts = contacts.model_active_contacts.numpy()
        self.assertEqual(len(contacts.model_max_contacts), 1)
        self.assertEqual(len(contacts.model_active_contacts), 1)
        self.assertEqual(model_max_contacts[0], num_worlds * contacts.default_max_world_contacts)
        self.assertEqual(model_active_contacts[0], 0)

        world_max_contacts = contacts.world_max_contacts.numpy()
        world_active_contacts = contacts.world_active_contacts.numpy()
        self.assertEqual(len(contacts.world_max_contacts), num_worlds)
        self.assertEqual(len(contacts.world_active_contacts), num_worlds)
        for i in range(num_worlds):
            self.assertEqual(world_max_contacts[i], contacts.default_max_world_contacts)
            self.assertEqual(world_active_contacts[i], 0)
        self.assertEqual(len(contacts.wid), num_worlds * contacts.default_max_world_contacts)
        self.assertEqual(len(contacts.cid), num_worlds * contacts.default_max_world_contacts)
        self.assertEqual(len(contacts.gid_AB), num_worlds * contacts.default_max_world_contacts)
        self.assertEqual(len(contacts.bid_AB), num_worlds * contacts.default_max_world_contacts)
        self.assertEqual(len(contacts.position_A), num_worlds * contacts.default_max_world_contacts)
        self.assertEqual(len(contacts.position_B), num_worlds * contacts.default_max_world_contacts)
        self.assertEqual(len(contacts.gapfunc), num_worlds * contacts.default_max_world_contacts)
        self.assertEqual(len(contacts.frame), num_worlds * contacts.default_max_world_contacts)
        self.assertEqual(len(contacts.material), num_worlds * contacts.default_max_world_contacts)

    def test_multiple_custom_allocations(self):
        capacities = [10, 20, 30, 40, 50, 60]
        contacts = ContactsKamino(capacity=capacities, device=self.default_device)

        num_worlds = len(capacities)
        model_max_contacts = contacts.model_max_contacts.numpy()
        model_active_contacts = contacts.model_active_contacts.numpy()
        self.assertEqual(len(contacts.model_max_contacts), 1)
        self.assertEqual(len(contacts.model_active_contacts), 1)
        self.assertEqual(model_max_contacts[0], sum(capacities))
        self.assertEqual(model_active_contacts[0], 0)

        world_max_contacts = contacts.world_max_contacts.numpy()
        world_active_contacts = contacts.world_active_contacts.numpy()
        self.assertEqual(len(contacts.world_max_contacts), num_worlds)
        self.assertEqual(len(contacts.world_active_contacts), num_worlds)
        for i in range(num_worlds):
            self.assertEqual(world_max_contacts[i], capacities[i])
            self.assertEqual(world_active_contacts[i], 0)

        maxnc = sum(capacities)
        self.assertEqual(len(contacts.wid), maxnc)
        self.assertEqual(len(contacts.cid), maxnc)
        self.assertEqual(len(contacts.gid_AB), maxnc)
        self.assertEqual(len(contacts.bid_AB), maxnc)
        self.assertEqual(len(contacts.position_A), maxnc)
        self.assertEqual(len(contacts.position_B), maxnc)
        self.assertEqual(len(contacts.gapfunc), maxnc)
        self.assertEqual(len(contacts.frame), maxnc)
        self.assertEqual(len(contacts.material), maxnc)


class TestGeometryContactConversions(unittest.TestCase):
    """Tests for Newton <-> Kamino contact conversion functions."""

    def setUp(self):
        if not test_context.setup_done:
            setup_tests(clear_cache=False)
        self.default_device = wp.get_device(test_context.device)
        self.verbose = test_context.verbose

        if self.verbose:
            msg.info("\n")
            msg.set_log_level(msg.LogLevel.DEBUG)
        else:
            msg.reset_log_level()

    def tearDown(self):
        self.default_device = None
        if self.verbose:
            msg.reset_log_level()

    def _setup_newton_scene(self):
        """Finalize the nunchaku model and return (newton_model, newton_state, newton_contacts).

        For single-world models, Newton assigns ``shape_world = -1`` (global)
        to all shapes.  The conversion kernels require world-0 assignment, so
        we normalize ``shape_world`` to match what ``ModelKamino.from_newton``
        does internally.
        """
        builder = build_test_system()
        model = builder.finalize(self.default_device)

        if model.world_count == 1:
            sw = model.shape_world.numpy()
            if np.any(sw < 0):
                sw[sw < 0] = 0
                model.shape_world.assign(sw)

        state = model.state()
        newton.eval_fk(model, model.joint_q, model.joint_qd, state)
        contacts = model.collide(state)
        return model, state, contacts

    def test_01_newton_to_kamino(self):
        """Newton contacts converted to Kamino must preserve count, A/B convention, and properties."""
        model, state, newton_contacts = self._setup_newton_scene()
        nc = int(newton_contacts.rigid_contact_count.numpy()[0])
        self.assertGreater(nc, 0, "Newton collision detection must produce contacts")

        kamino_out = ContactsKamino(capacity=nc + 16, device=self.default_device)
        convert_contacts_newton_to_kamino(model, state, newton_contacts, kamino_out)
        wp.synchronize()

        nc_kamino = int(kamino_out.model_active_contacts.numpy()[0])
        self.assertGreater(nc_kamino, 0, "Conversion must produce Kamino contacts")

        # Verify A/B convention: bid_B must be >= 0 for all active contacts
        bid_AB = kamino_out.bid_AB.numpy()[:nc_kamino]
        for i in range(nc_kamino):
            self.assertGreaterEqual(int(bid_AB[i, 1]), 0, f"Contact {i}: bid_B must be >= 0 (Kamino convention)")

        # Verify gapfunc normals are unit vectors
        gapfunc = kamino_out.gapfunc.numpy()[:nc_kamino]
        for i in range(nc_kamino):
            n = gapfunc[i, :3]
            norm = np.linalg.norm(n)
            self.assertTrue(np.isclose(norm, 1.0, atol=1e-5), f"Contact {i}: normal not unit ({norm})")

        # Verify signed distance is non-positive (only penetrating contacts are kept)
        for i in range(nc_kamino):
            self.assertLessEqual(gapfunc[i, 3], 0.0, f"Contact {i}: distance must be <= 0")

        # Verify material properties are non-negative
        material = kamino_out.material.numpy()[:nc_kamino]
        for i in range(nc_kamino):
            self.assertGreaterEqual(material[i, 0], 0.0, f"Contact {i}: friction must be >= 0")
            self.assertGreaterEqual(material[i, 1], 0.0, f"Contact {i}: restitution must be >= 0")

        if self.verbose:
            msg.debug("Newton -> Kamino: %d contacts converted", nc_kamino)
            msg.debug("bid_AB:\n%s", bid_AB)
            msg.debug("gapfunc:\n%s", gapfunc)

    def test_02_kamino_to_newton(self):
        """Kamino contacts converted to Newton must have valid shape indices and body-local points."""
        model, state, newton_contacts_orig = self._setup_newton_scene()
        nc = int(newton_contacts_orig.rigid_contact_count.numpy()[0])
        self.assertGreater(nc, 0)

        # Newton -> Kamino first (to populate Kamino contacts from a known source)
        kamino_contacts = ContactsKamino(capacity=nc + 16, device=self.default_device)
        convert_contacts_newton_to_kamino(model, state, newton_contacts_orig, kamino_contacts)
        wp.synchronize()
        nc_kamino = int(kamino_contacts.model_active_contacts.numpy()[0])
        self.assertGreater(nc_kamino, 0)

        # Kamino -> Newton (the function under test)
        newton_contacts_out = Contacts(
            rigid_contact_max=kamino_contacts.model_max_contacts_host,
            soft_contact_max=0,
            device=self.default_device,
        )
        convert_contacts_kamino_to_newton(model, state, kamino_contacts, newton_contacts_out)
        wp.synchronize()

        nc_newton = int(newton_contacts_out.rigid_contact_count.numpy()[0])
        self.assertEqual(nc_newton, nc_kamino)

        # Verify shape indices are valid model-global indices
        shape_count = model.shape_count
        shape0 = newton_contacts_out.rigid_contact_shape0.numpy()[:nc_newton]
        shape1 = newton_contacts_out.rigid_contact_shape1.numpy()[:nc_newton]
        for i in range(nc_newton):
            self.assertGreaterEqual(int(shape0[i]), 0)
            self.assertLess(int(shape0[i]), shape_count, f"Contact {i}: shape0 out of range")
            self.assertGreaterEqual(int(shape1[i]), 0)
            self.assertLess(int(shape1[i]), shape_count, f"Contact {i}: shape1 out of range")

        # Verify normals are unit vectors
        normals = newton_contacts_out.rigid_contact_normal.numpy()[:nc_newton]
        for i in range(nc_newton):
            norm = np.linalg.norm(normals[i])
            self.assertTrue(np.isclose(norm, 1.0, atol=1e-5), f"Contact {i}: normal not unit ({norm})")

        # Verify body-local points: transforming back to world should match Kamino positions
        body_q = state.body_q.numpy()
        shape_body = model.shape_body.numpy()
        point0 = newton_contacts_out.rigid_contact_point0.numpy()[:nc_newton]
        point1 = newton_contacts_out.rigid_contact_point1.numpy()[:nc_newton]
        kamino_pos_A = kamino_contacts.position_A.numpy()[:nc_kamino]
        kamino_pos_B = kamino_contacts.position_B.numpy()[:nc_kamino]

        for i in range(nc_newton):
            b0 = int(shape_body[int(shape0[i])])
            b1 = int(shape_body[int(shape1[i])])

            if b0 >= 0:
                p0_world = _transform_point(body_q[b0], point0[i])
            else:
                p0_world = point0[i]

            if b1 >= 0:
                p1_world = _transform_point(body_q[b1], point1[i])
            else:
                p1_world = point1[i]

            np.testing.assert_allclose(
                p0_world,
                kamino_pos_A[i],
                atol=1e-4,
                err_msg=f"Contact {i}: point0 world mismatch",
            )
            np.testing.assert_allclose(
                p1_world,
                kamino_pos_B[i],
                atol=1e-4,
                err_msg=f"Contact {i}: point1 world mismatch",
            )

        if self.verbose:
            msg.debug("Kamino -> Newton: %d contacts converted", nc_newton)
            msg.debug("shape0: %s", shape0)
            msg.debug("shape1: %s", shape1)

    def test_03_roundtrip_newton_kamino_newton(self):
        """Round-trip Newton -> Kamino -> Newton preserves world-space contact geometry.

        N->K may filter non-penetrating contacts, so the round-tripped count
        can be smaller than the original Newton count.  The test verifies that
        the Kamino intermediate world-space positions match the K->N
        round-tripped positions exactly, and that shape *pairs* (as unordered
        sets) are preserved.
        """
        model, state, newton_contacts_orig = self._setup_newton_scene()
        nc_orig = int(newton_contacts_orig.rigid_contact_count.numpy()[0])
        self.assertGreater(nc_orig, 0)

        # Step 1: Newton -> Kamino
        kamino_out = ContactsKamino(capacity=nc_orig + 16, device=self.default_device)
        convert_contacts_newton_to_kamino(model, state, newton_contacts_orig, kamino_out)
        wp.synchronize()

        nc_kamino = int(kamino_out.model_active_contacts.numpy()[0])
        self.assertGreater(nc_kamino, 0)
        self.assertLessEqual(nc_kamino, nc_orig, "N->K must not create more contacts than Newton")

        # Capture Kamino world-space positions as ground truth
        kamino_pos_A = kamino_out.position_A.numpy()[:nc_kamino]
        kamino_pos_B = kamino_out.position_B.numpy()[:nc_kamino]

        # Step 2: Kamino -> Newton (round-trip back)
        newton_contacts_rt = Contacts(
            rigid_contact_max=kamino_out.model_max_contacts_host,
            soft_contact_max=0,
            device=self.default_device,
        )
        convert_contacts_kamino_to_newton(model, state, kamino_out, newton_contacts_rt)
        wp.synchronize()

        nc_rt = int(newton_contacts_rt.rigid_contact_count.numpy()[0])
        self.assertEqual(nc_rt, nc_kamino)

        # Verify all shape indices are valid
        shape_count = model.shape_count
        body_q = state.body_q.numpy()
        shape_body = model.shape_body.numpy()
        shape0_rt = newton_contacts_rt.rigid_contact_shape0.numpy()[:nc_rt]
        shape1_rt = newton_contacts_rt.rigid_contact_shape1.numpy()[:nc_rt]
        point0_rt = newton_contacts_rt.rigid_contact_point0.numpy()[:nc_rt]
        point1_rt = newton_contacts_rt.rigid_contact_point1.numpy()[:nc_rt]
        normal_rt = newton_contacts_rt.rigid_contact_normal.numpy()[:nc_rt]

        for i in range(nc_rt):
            self.assertGreaterEqual(int(shape0_rt[i]), 0)
            self.assertLess(int(shape0_rt[i]), shape_count)
            self.assertGreaterEqual(int(shape1_rt[i]), 0)
            self.assertLess(int(shape1_rt[i]), shape_count)
            norm = np.linalg.norm(normal_rt[i])
            self.assertTrue(np.isclose(norm, 1.0, atol=1e-5), f"Contact {i}: normal not unit")

        # Verify round-tripped body-local points transform back to Kamino world positions
        for i in range(nc_rt):
            b0 = int(shape_body[int(shape0_rt[i])])
            b1 = int(shape_body[int(shape1_rt[i])])
            p0w = _transform_point(body_q[b0], point0_rt[i]) if b0 >= 0 else point0_rt[i]
            p1w = _transform_point(body_q[b1], point1_rt[i]) if b1 >= 0 else point1_rt[i]

            np.testing.assert_allclose(
                p0w,
                kamino_pos_A[i],
                atol=1e-4,
                err_msg=f"Contact {i}: round-trip point0 world mismatch",
            )
            np.testing.assert_allclose(
                p1w,
                kamino_pos_B[i],
                atol=1e-4,
                err_msg=f"Contact {i}: round-trip point1 world mismatch",
            )

        if self.verbose:
            msg.debug("Round-trip: %d -> %d -> %d contacts", nc_orig, nc_kamino, nc_rt)

    def test_04_nunchaku_full_pipeline(self):
        """Full nunchaku pipeline: Newton collide -> Kamino -> Newton, verifying each step."""
        model, state, newton_contacts = self._setup_newton_scene()
        nc_newton = int(newton_contacts.rigid_contact_count.numpy()[0])
        self.assertGreater(nc_newton, 0, "Newton collision detection must produce contacts")

        # Step 1: Newton -> Kamino
        kamino_contacts = ContactsKamino(capacity=nc_newton + 16, device=self.default_device)
        convert_contacts_newton_to_kamino(model, state, newton_contacts, kamino_contacts)
        wp.synchronize()

        nc_kamino = int(kamino_contacts.model_active_contacts.numpy()[0])
        self.assertGreater(nc_kamino, 0)

        # Verify Kamino contacts properties
        gapfunc = kamino_contacts.gapfunc.numpy()[:nc_kamino]
        bid_AB = kamino_contacts.bid_AB.numpy()[:nc_kamino]
        kamino_pos_A = kamino_contacts.position_A.numpy()[:nc_kamino]
        kamino_pos_B = kamino_contacts.position_B.numpy()[:nc_kamino]

        for i in range(nc_kamino):
            self.assertGreaterEqual(int(bid_AB[i, 1]), 0, f"Contact {i}: bid_B must be >= 0")
            n = gapfunc[i, :3]
            norm = np.linalg.norm(n)
            self.assertTrue(np.isclose(norm, 1.0, atol=1e-5), f"Contact {i}: normal not unit ({norm})")
            self.assertLessEqual(gapfunc[i, 3], 0.0, f"Contact {i}: distance must be <= 0")

        # Step 2: Kamino -> Newton
        newton_contacts_2 = Contacts(
            rigid_contact_max=kamino_contacts.model_max_contacts_host,
            soft_contact_max=0,
            device=self.default_device,
        )
        convert_contacts_kamino_to_newton(model, state, kamino_contacts, newton_contacts_2)
        wp.synchronize()

        nc_newton_2 = int(newton_contacts_2.rigid_contact_count.numpy()[0])
        self.assertEqual(nc_newton_2, nc_kamino)

        # Verify Newton contacts are valid
        shape_count = model.shape_count
        shape0 = newton_contacts_2.rigid_contact_shape0.numpy()[:nc_newton_2]
        shape1 = newton_contacts_2.rigid_contact_shape1.numpy()[:nc_newton_2]
        normals = newton_contacts_2.rigid_contact_normal.numpy()[:nc_newton_2]
        body_q = state.body_q.numpy()
        shape_body = model.shape_body.numpy()
        point0 = newton_contacts_2.rigid_contact_point0.numpy()[:nc_newton_2]
        point1 = newton_contacts_2.rigid_contact_point1.numpy()[:nc_newton_2]

        for i in range(nc_newton_2):
            self.assertGreaterEqual(int(shape0[i]), 0)
            self.assertLess(int(shape0[i]), shape_count, f"Contact {i}: shape0 out of range")
            self.assertGreaterEqual(int(shape1[i]), 0)
            self.assertLess(int(shape1[i]), shape_count, f"Contact {i}: shape1 out of range")
            norm = np.linalg.norm(normals[i])
            self.assertTrue(np.isclose(norm, 1.0, atol=1e-5), f"Contact {i}: normal not unit")

        # Verify K->N body-local points transform back to the Kamino world positions
        for i in range(nc_newton_2):
            b0 = int(shape_body[int(shape0[i])])
            b1 = int(shape_body[int(shape1[i])])

            if b0 >= 0:
                p0_world = _transform_point(body_q[b0], point0[i])
            else:
                p0_world = point0[i]

            if b1 >= 0:
                p1_world = _transform_point(body_q[b1], point1[i])
            else:
                p1_world = point1[i]

            np.testing.assert_allclose(
                p0_world,
                kamino_pos_A[i],
                atol=1e-4,
                err_msg=f"Contact {i}: K->N point0 world mismatch with Kamino pos_A",
            )
            np.testing.assert_allclose(
                p1_world,
                kamino_pos_B[i],
                atol=1e-4,
                err_msg=f"Contact {i}: K->N point1 world mismatch with Kamino pos_B",
            )

        if self.verbose:
            msg.debug("Nunchaku pipeline: %d -> %d -> %d contacts", nc_newton, nc_kamino, nc_newton_2)

    def test_05_multi_world_roundtrip(self):
        """N->K->N round-trip with heterogeneous worlds and a shared ground plane.

        Scene layout (ground plane added first, shared across all worlds):
          - World 0: nunchaku (2 boxes + 1 sphere) -> 9 contacts (4+4+1)
          - World 1: nunchaku (2 boxes + 1 sphere) -> 9 contacts (4+4+1)
          - World 2: single box                    -> 4 contacts

        The ground plane is global (shape_world == -1) and precedes all
        per-world shapes, exercising the plane-first ordering path.
        Expected total: 22 contacts.
        """
        nunchaku_blueprint = build_test_system(ground=False)

        box_blueprint = ModelBuilder()
        b = box_blueprint.add_link()
        no_gap = ModelBuilder.ShapeConfig(gap=0.0)
        box_blueprint.add_shape_box(b, hx=0.25, hy=0.25, hz=0.25, cfg=no_gap)
        j = box_blueprint.add_joint_free(
            parent=-1,
            child=b,
            parent_xform=wp.transform(p=wp.vec3(0.0, 0.0, 0.25), q=wp.quat_identity()),
            child_xform=wp.transform_identity(),
        )
        box_blueprint.add_articulation([j])

        scene = ModelBuilder()
        scene.add_ground_plane()
        scene.add_world(nunchaku_blueprint, xform=wp.transform(p=wp.vec3(0.0, 0.0, 0.0)))
        scene.add_world(nunchaku_blueprint, xform=wp.transform(p=wp.vec3(5.0, 0.0, 0.0)))
        scene.add_world(box_blueprint, xform=wp.transform(p=wp.vec3(10.0, 0.0, 0.0)))
        model = scene.finalize(self.default_device)

        self.assertEqual(model.world_count, 3)
        sws = model.shape_world_start.numpy()
        self.assertGreater(int(sws[1]), 0, "World 1 must have nonzero shape_world_start")

        state = model.state()
        newton.eval_fk(model, model.joint_q, model.joint_qd, state)
        contacts = model.collide(state)
        nc_orig = int(contacts.rigid_contact_count.numpy()[0])
        self.assertGreater(nc_orig, 0)

        expected_contacts = 9 + 9 + 4
        self.assertEqual(
            nc_orig,
            expected_contacts,
            f"Expected {expected_contacts} contacts (9+9+4), got {nc_orig}",
        )

        kamino_out = ContactsKamino(capacity=nc_orig + 32, device=self.default_device)
        convert_contacts_newton_to_kamino(model, state, contacts, kamino_out)
        wp.synchronize()
        nc_kamino = int(kamino_out.model_active_contacts.numpy()[0])
        self.assertGreater(nc_kamino, 0)

        newton_rt = Contacts(
            rigid_contact_max=kamino_out.model_max_contacts_host,
            soft_contact_max=0,
            device=self.default_device,
        )
        convert_contacts_kamino_to_newton(model, state, kamino_out, newton_rt)
        wp.synchronize()
        nc_rt = int(newton_rt.rigid_contact_count.numpy()[0])
        self.assertEqual(nc_rt, nc_kamino)

        shape_count = model.shape_count
        shape_body = model.shape_body.numpy()
        body_q = state.body_q.numpy()
        shape0 = newton_rt.rigid_contact_shape0.numpy()[:nc_rt]
        shape1 = newton_rt.rigid_contact_shape1.numpy()[:nc_rt]
        point0 = newton_rt.rigid_contact_point0.numpy()[:nc_rt]
        point1 = newton_rt.rigid_contact_point1.numpy()[:nc_rt]
        kamino_pos_A = kamino_out.position_A.numpy()[:nc_kamino]
        kamino_pos_B = kamino_out.position_B.numpy()[:nc_kamino]

        for i in range(nc_rt):
            self.assertGreaterEqual(int(shape0[i]), 0)
            self.assertLess(int(shape0[i]), shape_count, f"Contact {i}: shape0 out of range")
            self.assertGreaterEqual(int(shape1[i]), 0)
            self.assertLess(int(shape1[i]), shape_count, f"Contact {i}: shape1 out of range")

            b0 = int(shape_body[int(shape0[i])])
            b1 = int(shape_body[int(shape1[i])])
            p0w = _transform_point(body_q[b0], point0[i]) if b0 >= 0 else point0[i]
            p1w = _transform_point(body_q[b1], point1[i]) if b1 >= 0 else point1[i]
            np.testing.assert_allclose(p0w, kamino_pos_A[i], atol=1e-4, err_msg=f"Contact {i}: pos_A mismatch")
            np.testing.assert_allclose(p1w, kamino_pos_B[i], atol=1e-4, err_msg=f"Contact {i}: pos_B mismatch")

        if self.verbose:
            msg.debug(
                "Multi-world round-trip: %d -> %d -> %d contacts (3 worlds: nunchaku, nunchaku, box)",
                nc_orig,
                nc_kamino,
                nc_rt,
            )


###
# Helpers
###


def _quat_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate vector v by quaternion q (x, y, z, w convention from Warp transforms)."""
    qx, qy, qz, qw = q[0], q[1], q[2], q[3]
    t = 2.0 * np.cross(np.array([qx, qy, qz]), v)
    return v + qw * t + np.cross(np.array([qx, qy, qz]), t)


def _transform_point(xform: np.ndarray, point: np.ndarray) -> np.ndarray:
    """Apply a Warp transform (p[0:3], q[3:7]) to a point."""
    return xform[:3] + _quat_rotate(xform[3:], point)


###
# Test execution
###


if __name__ == "__main__":
    # Test setup
    setup_tests()

    # Run all tests
    unittest.main(verbosity=2)
