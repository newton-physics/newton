# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""
Unit tests for `geometry/contacts.py`.

Tests all components of the ContactsKamino data types and operations.
"""

import unittest
from collections.abc import Callable

import numpy as np
import warp as wp

import newton
from newton._src.sim import Model, ModelBuilder, State
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


def build_nunchaku_scene(
    builder: ModelBuilder | None = None,
    ground: bool = True,
) -> ModelBuilder:
    """
    Constructs a nunchaku model: two boxes connected by a sphere via ball joints.

    Three bodies (two boxes + one sphere) connected by spherical joints,
    optionally resting on a ground plane. With the ground plane present the
    scene produces 9 contacts (4 per box + 1 sphere), all between a dynamic
    body and the static ground -- i.e. every contact exercises the static-swap
    branch of N->K (Kamino ``bid_A == -1``).

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
    builder.add_shape_box(b0, hx=0.5 * d, hy=0.5 * w, hz=0.5 * h, cfg=no_gap)

    b1 = builder.add_link()
    builder.add_shape_sphere(b1, radius=r, cfg=no_gap)

    b2 = builder.add_link()
    builder.add_shape_box(b2, hx=0.5 * d, hy=0.5 * w, hz=0.5 * h, cfg=no_gap)

    j0 = builder.add_joint_ball(
        parent=-1,
        child=b0,
        parent_xform=wp.transform(p=wp.vec3(0.5 * d, 0.0, 0.5 * h), q=wp.quat_identity()),
        child_xform=wp.transform(p=wp.vec3(0.0, 0.0, 0.0), q=wp.quat_identity()),
    )
    j1 = builder.add_joint_ball(
        parent=b0,
        child=b1,
        parent_xform=wp.transform(p=wp.vec3(0.5 * d, 0.0, 0.0), q=wp.quat_identity()),
        child_xform=wp.transform(p=wp.vec3(-r, 0.0, 0.0), q=wp.quat_identity()),
    )
    j2 = builder.add_joint_ball(
        parent=b1,
        child=b2,
        parent_xform=wp.transform(p=wp.vec3(r, 0.0, 0.0), q=wp.quat_identity()),
        child_xform=wp.transform(p=wp.vec3(-0.5 * d, 0.0, 0.0), q=wp.quat_identity()),
    )
    builder.add_articulation([j0, j1, j2])

    if ground:
        builder.add_ground_plane()

    return builder


def build_two_box_stack_scene(
    builder: ModelBuilder | None = None,
    penetration: float = 0.02,
) -> ModelBuilder:
    """
    Constructs a two-free-body box stack with no ground plane.

    Both boxes are dynamic (free joints), so every contact between them
    exercises the *no-swap* branch of N->K (Kamino A = Newton shape0,
    Kamino B = Newton shape1, both with ``bid >= 0``).

    Args:
        builder: An optional existing model builder to populate.
            If ``None``, a new builder is created.
        penetration: How deeply the upper box is pushed into the lower box
            along the Z axis. Must be small enough that Newton's contact
            detector reports the resulting points as penetrating
            (``distance <= 0``).

    Returns:
        The populated :class:`ModelBuilder`.
    """
    if builder is None:
        builder = ModelBuilder()

    h = 0.25
    no_gap = ModelBuilder.ShapeConfig(gap=0.0)

    b0 = builder.add_link()
    builder.add_shape_box(b0, hx=h, hy=h, hz=h, cfg=no_gap)
    j0 = builder.add_joint_free(
        parent=-1,
        child=b0,
        parent_xform=wp.transform(p=wp.vec3(0.0, 0.0, h), q=wp.quat_identity()),
        child_xform=wp.transform_identity(),
    )
    builder.add_articulation([j0])

    b1 = builder.add_link()
    builder.add_shape_box(b1, hx=h, hy=h, hz=h, cfg=no_gap)
    j1 = builder.add_joint_free(
        parent=-1,
        child=b1,
        parent_xform=wp.transform(
            p=wp.vec3(0.0, 0.0, 3.0 * h - penetration),
            q=wp.quat_identity(),
        ),
        child_xform=wp.transform_identity(),
    )
    builder.add_articulation([j1])

    return builder


def build_dynamic_static_sphere_scene(
    builder: ModelBuilder | None = None,
    penetration: float = 0.02,
) -> ModelBuilder:
    """
    Constructs a dynamic sphere resting against a static (world-attached) sphere.

    The dynamic sphere is added first (shape 0, ``body == 0``) and a
    world-static sphere is added second (shape 1, ``body == -1``). Newton's
    collision detector preserves this ordering, so the resulting contact has
    ``shape0`` = dynamic and ``shape1`` = static. This is the only scene
    in this file that exercises the *swap* branch of the N->K kernel
    (``bid_1 < 0`` triggers Kamino A <- shape1, Kamino B <- shape0). The
    ``add_ground_plane`` builder, by contrast, gets re-ordered by Newton to
    appear as ``shape0`` and thus only exercises the no-swap branch.
    """
    if builder is None:
        builder = ModelBuilder()

    r = 0.1
    no_gap = ModelBuilder.ShapeConfig(gap=0.0)

    b = builder.add_link()
    builder.add_shape_sphere(b, radius=r, cfg=no_gap)
    j = builder.add_joint_free(
        parent=-1,
        child=b,
        parent_xform=wp.transform(
            p=wp.vec3(0.0, 0.0, 2.0 * r - penetration),
            q=wp.quat_identity(),
        ),
        child_xform=wp.transform_identity(),
    )
    builder.add_articulation([j])

    builder.add_shape_sphere(-1, radius=r, cfg=no_gap)
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
# Functions
###


def _quat_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate vector v by quaternion q (x, y, z, w convention from Warp transforms)."""
    qx, qy, qz, qw = q[0], q[1], q[2], q[3]
    t = 2.0 * np.cross(np.array([qx, qy, qz]), v)
    return v + qw * t + np.cross(np.array([qx, qy, qz]), t)


def _transform_point(xform: np.ndarray, point: np.ndarray) -> np.ndarray:
    """Apply a Warp transform (p[0:3], q[3:7]) to a point."""
    return xform[:3] + _quat_rotate(xform[3:], point)


def _make_contact_frame_znorm_np(n: np.ndarray) -> np.ndarray:
    """NumPy reference of :func:`make_contact_frame_znorm`.

    Returns the 3x3 rotation ``R = world <- contact`` whose columns are
    ``[t, o, n]`` (tangent, other, normal), built with the same tangent-axis
    selection rule as the Warp ``@wp.func`` (`UNIT_X` unless ``|n . X| >=
    cos(pi/6)``, in which case `UNIT_Y`).
    """
    cos_pi_6 = 0.8660254037844387
    unit_x = np.array([1.0, 0.0, 0.0])
    unit_y = np.array([0.0, 1.0, 0.0])

    n = np.asarray(n, dtype=np.float64)
    n = n / np.linalg.norm(n)
    e = unit_x if abs(np.dot(n, unit_x)) < cos_pi_6 else unit_y
    o = np.cross(n, e)
    o = o / np.linalg.norm(o)
    t = np.cross(o, n)
    t = t / np.linalg.norm(t)
    return np.column_stack([t, o, n])


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
        contacts = ContactsKamino(capacity=0, device=self.default_device, remappable=True)
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
        self.assertEqual(len(contacts.margins), contacts.default_max_world_contacts)
        self.assertEqual(len(contacts.key), contacts.default_max_world_contacts)
        self.assertEqual(len(contacts.reaction), contacts.default_max_world_contacts)
        self.assertEqual(len(contacts.velocity), contacts.default_max_world_contacts)
        self.assertEqual(len(contacts.mode), contacts.default_max_world_contacts)
        self.assertEqual(len(contacts.remap), contacts.default_max_world_contacts)

    def test_multiple_default_allocations(self):
        num_worlds = 10
        capacities = [0] * num_worlds
        contacts = ContactsKamino(capacity=capacities, device=self.default_device, remappable=True)

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
        self.assertEqual(len(contacts.margins), num_worlds * contacts.default_max_world_contacts)
        self.assertEqual(len(contacts.key), num_worlds * contacts.default_max_world_contacts)
        self.assertEqual(len(contacts.reaction), num_worlds * contacts.default_max_world_contacts)
        self.assertEqual(len(contacts.velocity), num_worlds * contacts.default_max_world_contacts)
        self.assertEqual(len(contacts.mode), num_worlds * contacts.default_max_world_contacts)
        self.assertEqual(len(contacts.remap), num_worlds * contacts.default_max_world_contacts)

    def test_multiple_custom_allocations(self):
        capacities = [10, 20, 30, 40, 50, 60]
        contacts = ContactsKamino(capacity=capacities, device=self.default_device, remappable=True)

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
        self.assertEqual(len(contacts.margins), maxnc)
        self.assertEqual(len(contacts.key), maxnc)
        self.assertEqual(len(contacts.reaction), maxnc)
        self.assertEqual(len(contacts.velocity), maxnc)
        self.assertEqual(len(contacts.mode), maxnc)
        self.assertEqual(len(contacts.remap), maxnc)


class TestGeometryContactConversions(unittest.TestCase):
    """Tests for Newton <-> Kamino contact conversion functions.

    Three reference scenes are used here:

    - ``build_nunchaku_scene`` (with ground) -- every contact pairs a dynamic
      body with the static ground plane. Newton's collision detector re-orders
      the ``add_ground_plane`` shape so it appears as ``shape0`` and the
      dynamic body as ``shape1``, so the N->K kernel takes the *no-swap*
      branch with Kamino A = Newton shape0 = ground (``bid_A == -1``).
    - ``build_two_box_stack_scene`` -- two free-body boxes stacked on top of
      each other with no ground plane. Newton's ordering produces ``shape0``
      = bottom box, ``shape1`` = top box, so the N->K kernel takes the
      *no-swap* branch but with both ``bid_A`` and ``bid_B`` dynamic.
    - ``build_dynamic_static_sphere_scene`` -- a dynamic sphere (added first)
      resting against a world-static sphere (added second). Newton preserves
      this insertion order, so ``shape0 = dynamic`` and ``shape1 = static``
      (``bid_1 < 0``) and the N->K kernel takes the *swap* branch.

    Together these scenes cover the three contact configurations:
    static-vs-dynamic with the static body as Newton ``shape0`` (no-swap),
    dynamic-vs-dynamic (no-swap), and static-vs-dynamic with the static
    body as Newton ``shape1`` (swap).
    """

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

    ###
    # Helpers
    ###

    def _setup_newton_scene(
        self,
        builder_fn: Callable[[], ModelBuilder] = build_nunchaku_scene,
        *,
        with_force: bool = False,
    ) -> tuple[Model, State, Contacts]:
        """Finalize a scene and return ``(model, state, contacts)``.

        For single-world models, Newton assigns ``shape_world = -1`` (global)
        to all shapes. The N->K conversion kernel requires non-negative world
        assignments, so we normalize ``shape_world`` to match what
        ``ModelKamino.from_newton`` does internally.

        Args:
            builder_fn: Scene builder function returning a populated
                :class:`ModelBuilder`.
            with_force: If ``True``, request the optional ``force`` contact
                attribute so that ``Contacts.force`` is allocated.

        Returns:
            Tuple ``(newton_model, newton_state, newton_contacts)``.
        """
        builder = builder_fn()
        if with_force:
            builder.request_contact_attributes("force")
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

    @staticmethod
    def _seed_constant_linear_force(contacts: Contacts, f_world: np.ndarray) -> None:
        """Fill the linear part of every active ``force`` slot with ``f_world``.

        Models a "pure force at the contact point" (zero torque) as the input
        to N->K. Note that Newton's wrench is stored at the CoM of body0, so a
        non-zero ``f_world`` with zero torque does not generally round-trip
        back to zero torque through the existing-contacts K->N path -- the
        reconstructed torque is ``(r_pt_on_body0 - r_com_body0) x f_world``.
        """
        nc = int(contacts.rigid_contact_count.numpy()[0])
        force_np = contacts.force.numpy()
        force_np[:nc, :3] = f_world.astype(force_np.dtype, copy=False)
        force_np[:nc, 3:] = 0.0
        contacts.force.assign(force_np)

    @staticmethod
    def _compute_expected_existing_torque(
        model,
        state,
        contacts: Contacts,
        f_world: np.ndarray,
    ) -> np.ndarray:
        """Compute the expected torque produced by the existing-contacts K->N path.

        For each contact ``cid``, the kernel reconstructs the wrench as
        ``tau = (r_pt_on_body0 - r_com_body0) x f_world`` (world coordinates).
        Contacts whose ``body0`` is static produce zero torque.
        """
        nc = int(contacts.rigid_contact_count.numpy()[0])
        body_q = state.body_q.numpy()
        body_com = model.body_com.numpy()
        shape_body = model.shape_body.numpy()
        shape0 = contacts.rigid_contact_shape0.numpy()[:nc]
        point0 = contacts.rigid_contact_point0.numpy()[:nc]
        out = np.zeros((nc, 3), dtype=np.float64)
        for i in range(nc):
            b0 = int(shape_body[int(shape0[i])])
            if b0 < 0:
                continue
            r_pt = _transform_point(body_q[b0], point0[i])
            r_com = _transform_point(body_q[b0], body_com[b0])
            out[i] = np.cross(r_pt - r_com, f_world)
        return out

    ###
    # N->K
    ###

    def test_01_newton_to_kamino(self):
        """N->K populates geometry + (with ``convert_forces=True``) reaction.

        Verifies for the nunchaku-ground scene:
          - Active count > 0; ``bid_AB[:, 1] >= 0`` for every active contact
            (Kamino's A/B convention).
          - ``bid_AB[:, 0] == -1`` because Newton orders the static ground as
            shape0 and the kernel takes the no-swap branch (Kamino A = Newton
            shape0 = ground).
          - Gap-function normal is a unit vector pointing roughly +Z and
            signed distance <= 0.
          - Material properties are non-negative.
          - With a seeded constant ``f_world`` and ``convert_forces=True``,
            every reaction equals ``-R(normal)^T @ f_world``. Newton stores
            ``force[cid] = wrench on body0 (=ground) by body1 (=dynamic)``,
            and Kamino's ``reaction`` is the force on B (=dynamic) by A
            (=ground), so the linear parts are negatives of each other in
            world coordinates and a frame rotation introduces the inverse
            rotation.
        """
        f_world = np.array([0.5, -0.7, 0.3], dtype=np.float64)
        model, state, newton_contacts = self._setup_newton_scene(with_force=True)
        nc = int(newton_contacts.rigid_contact_count.numpy()[0])
        self.assertGreater(nc, 0, "Newton collision detection must produce contacts")
        self._seed_constant_linear_force(newton_contacts, f_world)

        kamino_out = ContactsKamino(capacity=nc + 16, device=self.default_device)
        convert_contacts_newton_to_kamino(model, state, newton_contacts, kamino_out, convert_forces=True)

        nc_kamino = int(kamino_out.model_active_contacts.numpy()[0])
        self.assertGreater(nc_kamino, 0, "Conversion must produce Kamino contacts")

        # A/B convention: bid_B must always be >= 0.
        bid_AB = kamino_out.bid_AB.numpy()[:nc_kamino]
        for i in range(nc_kamino):
            self.assertGreaterEqual(int(bid_AB[i, 1]), 0, f"Contact {i}: bid_B must be >= 0")

        # Newton orders the static ground first (shape0), so the N->K kernel
        # takes the no-swap branch and Kamino A is the static ground.
        np.testing.assert_array_equal(
            bid_AB[:, 0],
            np.full(nc_kamino, -1, dtype=bid_AB.dtype),
            err_msg="Nunchaku-ground scene must keep the ground as Kamino A (no-swap branch)",
        )

        gapfunc = kamino_out.gapfunc.numpy()[:nc_kamino]
        for i in range(nc_kamino):
            n = gapfunc[i, :3]
            self.assertTrue(np.isclose(np.linalg.norm(n), 1.0, atol=1e-5), f"Contact {i}: normal not unit")
            self.assertLessEqual(gapfunc[i, 3], 0.0, f"Contact {i}: distance must be <= 0")
            self.assertGreater(n[2], 0.5, f"Contact {i}: ground normal expected near +Z")

        material = kamino_out.material.numpy()[:nc_kamino]
        self.assertTrue(np.all(material >= 0.0), "Material values must be non-negative")

        # Wrench: no-swap branch stores ``reaction = -R(normal)^T @ f_world``.
        reactions = kamino_out.reaction.numpy()[:nc_kamino]
        for i in range(nc_kamino):
            r_wc = _make_contact_frame_znorm_np(gapfunc[i, :3].astype(np.float64))
            np.testing.assert_allclose(
                reactions[i].astype(np.float64),
                -(r_wc.T @ f_world),
                atol=1e-5,
                err_msg=f"Contact {i}: reaction does not match -R(normal)^T @ f_world",
            )

        if self.verbose:
            msg.debug("N->K: %d contacts converted, max |reaction| = %g", nc_kamino, float(np.max(np.abs(reactions))))

    ###
    # K->N (active path)
    ###

    def test_02_kamino_to_newton_active(self):
        """K->N active path repopulates geometry and (with forces) writes f_world on body0.

        For the nunchaku-ground scene the active path writes Newton's
        ``shape0`` = Kamino A = ground (static) and ``shape1`` = Kamino B =
        dynamic body. body0 is the ground (``shape_body == -1``), so the
        active kernel writes a zero-torque wrench whose linear part is
        ``-quat_rotate(frame, reaction)``.
        """
        model, state, newton_contacts_orig = self._setup_newton_scene(with_force=True)
        nc = int(newton_contacts_orig.rigid_contact_count.numpy()[0])
        self.assertGreater(nc, 0)

        kamino = ContactsKamino(capacity=nc + 16, device=self.default_device)
        convert_contacts_newton_to_kamino(model, state, newton_contacts_orig, kamino)

        nc_kamino = int(kamino.model_active_contacts.numpy()[0])
        self.assertGreater(nc_kamino, 0)

        # Stuff a known local-frame reaction so we can verify K->N's rotation
        # independently of the inverse-rotation performed in N->K.
        reaction_local = np.tile(
            np.array([0.11, -0.22, 0.33], dtype=np.float32),
            (kamino.model_max_contacts_host, 1),
        )
        kamino.reaction.assign(reaction_local)

        newton_out = Contacts(
            rigid_contact_max=kamino.model_max_contacts_host,
            soft_contact_max=0,
            device=self.default_device,
            requested_attributes={"force"},
        )
        convert_contacts_kamino_to_newton(
            model,
            state,
            kamino,
            newton_out,
            clear_output=True,
            convert_forces=True,
        )

        nc_out = int(newton_out.rigid_contact_count.numpy()[0])
        self.assertEqual(nc_out, nc_kamino)

        # In nunchaku-ground every Kamino A is the ground, so Newton's shape0
        # must correspond to a shape with no body (shape_body == -1).
        shape_body = model.shape_body.numpy()
        shape0 = newton_out.rigid_contact_shape0.numpy()[:nc_out]
        for i in range(nc_out):
            self.assertEqual(
                int(shape_body[int(shape0[i])]),
                -1,
                f"Contact {i}: active path must write Newton shape0 = Kamino A (ground)",
            )

        # Verify body-local point0 transforms back to Kamino position_A.
        body_q = state.body_q.numpy()
        point0 = newton_out.rigid_contact_point0.numpy()[:nc_out]
        point1 = newton_out.rigid_contact_point1.numpy()[:nc_out]
        shape1 = newton_out.rigid_contact_shape1.numpy()[:nc_out]
        pos_A = kamino.position_A.numpy()[:nc_kamino]
        pos_B = kamino.position_B.numpy()[:nc_kamino]
        for i in range(nc_out):
            np.testing.assert_allclose(point0[i], pos_A[i], atol=1e-5)
            b1 = int(shape_body[int(shape1[i])])
            p1w = _transform_point(body_q[b1], point1[i]) if b1 >= 0 else point1[i]
            np.testing.assert_allclose(p1w, pos_B[i], atol=1e-4)

        # Wrench: linear part on body0 is ``-R(frame) @ reaction``, torque is
        # zero (body0 is the static ground -> no moment arm in the kernel).
        gapfunc = kamino.gapfunc.numpy()[:nc_kamino]
        force_out = newton_out.force.numpy()[:nc_out]
        for i in range(nc_out):
            r_wc = _make_contact_frame_znorm_np(gapfunc[i, :3].astype(np.float64))
            expected_linear = -(r_wc @ reaction_local[i].astype(np.float64))
            np.testing.assert_allclose(
                force_out[i, :3].astype(np.float64),
                expected_linear,
                atol=1e-5,
                err_msg=f"Contact {i}: linear force mismatch",
            )
            np.testing.assert_allclose(
                force_out[i, 3:].astype(np.float64),
                np.zeros(3),
                atol=1e-6,
                err_msg=f"Contact {i}: torque must be zero when body0 is static",
            )

    ###
    # K->N (existing path)
    ###

    def test_03_kamino_to_newton_existing(self):
        """K->N existing path writes Newton's wrench back at the original cid.

        Verifies that the existing-contacts path:
          - Preserves Newton's original ``rigid_contact_shape0/shape1`` and
            count (no clear).
          - Correctly identifies the static-swap and applies the sign such
            that ``force[cid_orig].linear`` recovers the input ``f_world``.
          - Reconstructs the torque as ``(r_pt - r_com) x f_world`` against
            Newton's preserved ``body0`` / ``point0``.
        """
        f_world = np.array([0.2, 0.4, -0.6], dtype=np.float64)
        model, state, newton_contacts = self._setup_newton_scene(with_force=True)
        nc = int(newton_contacts.rigid_contact_count.numpy()[0])
        self.assertGreater(nc, 0)
        self._seed_constant_linear_force(newton_contacts, f_world)

        # Snapshot the original geometry so we can verify it's preserved.
        shape0_orig = newton_contacts.rigid_contact_shape0.numpy()[:nc].copy()
        shape1_orig = newton_contacts.rigid_contact_shape1.numpy()[:nc].copy()
        expected_torque = self._compute_expected_existing_torque(model, state, newton_contacts, f_world)

        # N->K with remappable=True so the K->N existing path has remap data.
        kamino = ContactsKamino(capacity=nc + 16, device=self.default_device, remappable=True)
        convert_contacts_newton_to_kamino(model, state, newton_contacts, kamino, convert_forces=True)
        nc_kamino = int(kamino.model_active_contacts.numpy()[0])
        self.assertGreater(nc_kamino, 0)

        # Wipe the Newton-side ``force`` to make sure K->N actually rewrites it.
        newton_contacts.force.zero_()

        # K->N existing path: only writes ``force``, keeps geometry intact.
        convert_contacts_kamino_to_newton(
            model,
            state,
            kamino,
            newton_contacts,
            clear_output=False,
            convert_forces=True,
        )

        self.assertEqual(int(newton_contacts.rigid_contact_count.numpy()[0]), nc, "Count must be unchanged")
        np.testing.assert_array_equal(
            newton_contacts.rigid_contact_shape0.numpy()[:nc],
            shape0_orig,
            err_msg="shape0 must be preserved by the existing-contacts path",
        )
        np.testing.assert_array_equal(
            newton_contacts.rigid_contact_shape1.numpy()[:nc],
            shape1_orig,
            err_msg="shape1 must be preserved by the existing-contacts path",
        )

        # Only the contacts that were mapped through Kamino should have non-zero
        # forces. Build the set of written cids from the remap.
        remap = kamino.remap.numpy()[:nc_kamino]
        cids_out = remap[remap >= 0]
        self.assertGreater(len(cids_out), 0, "remap must produce at least one valid target index")

        force = newton_contacts.force.numpy()
        for cid_out in cids_out:
            np.testing.assert_allclose(
                force[cid_out, :3].astype(np.float64),
                f_world,
                atol=1e-5,
                err_msg=f"Contact cid={cid_out}: linear force must equal input f_world",
            )
            np.testing.assert_allclose(
                force[cid_out, 3:].astype(np.float64),
                expected_torque[cid_out],
                atol=1e-5,
                err_msg=f"Contact cid={cid_out}: torque must equal (r_pt - r_com) x f_world",
            )

    ###
    # Force round-trip across both K->N paths and both swap branches
    ###

    def _run_active_roundtrip(self, builder_fn, f_world: np.ndarray, expected_linear_sign: int = 1):
        """Helper: round-trip the linear force through the K->N active path.

        The active path writes ``f_linear = -R(frame) @ reaction`` on Newton's
        output ``body0``, which equals Kamino A. Whether this is also Newton's
        *original* body0 depends on whether N->K took the swap branch:

        - no-swap (``bid_1 >= 0``): Kamino A = Newton shape0, the output
          force is still "on body0_in", and the round-trip preserves
          ``+f_world`` (``expected_linear_sign = +1``).
        - swap (``bid_1 < 0``): Kamino A = Newton shape1, the output
          force is now "on body0_out = body1_in", which by Newton's third
          law is ``-f_world`` (``expected_linear_sign = -1``).
        """
        model, state, contacts = self._setup_newton_scene(builder_fn=builder_fn, with_force=True)
        nc = int(contacts.rigid_contact_count.numpy()[0])
        self.assertGreater(nc, 0)
        self._seed_constant_linear_force(contacts, f_world)

        kamino = ContactsKamino(capacity=nc + 16, device=self.default_device)
        convert_contacts_newton_to_kamino(model, state, contacts, kamino, convert_forces=True)
        nc_kamino = int(kamino.model_active_contacts.numpy()[0])
        self.assertGreater(nc_kamino, 0)

        newton_rt = Contacts(
            rigid_contact_max=kamino.model_max_contacts_host,
            soft_contact_max=0,
            device=self.default_device,
            requested_attributes={"force"},
        )
        convert_contacts_kamino_to_newton(
            model,
            state,
            kamino,
            newton_rt,
            clear_output=True,
            convert_forces=True,
        )

        nc_rt = int(newton_rt.rigid_contact_count.numpy()[0])
        self.assertEqual(nc_rt, nc_kamino)

        force_rt = newton_rt.force.numpy()[:nc_rt]
        for i in range(nc_rt):
            np.testing.assert_allclose(
                force_rt[i, :3].astype(np.float64),
                expected_linear_sign * f_world,
                atol=1e-5,
                err_msg=f"Active-path round-trip: contact {i} linear force mismatch",
            )

    def _run_existing_roundtrip(self, builder_fn, f_world: np.ndarray):
        """Helper: round-trip the linear force through the K->N existing path."""
        model, state, contacts = self._setup_newton_scene(builder_fn=builder_fn, with_force=True)
        nc = int(contacts.rigid_contact_count.numpy()[0])
        self.assertGreater(nc, 0)
        self._seed_constant_linear_force(contacts, f_world)
        expected_torque = self._compute_expected_existing_torque(model, state, contacts, f_world)

        kamino = ContactsKamino(capacity=nc + 16, device=self.default_device, remappable=True)
        convert_contacts_newton_to_kamino(model, state, contacts, kamino, convert_forces=True)
        nc_kamino = int(kamino.model_active_contacts.numpy()[0])
        self.assertGreater(nc_kamino, 0)

        contacts.force.zero_()
        convert_contacts_kamino_to_newton(
            model,
            state,
            kamino,
            contacts,
            clear_output=False,
            convert_forces=True,
        )

        remap = kamino.remap.numpy()[:nc_kamino]
        force = contacts.force.numpy()
        for cid_out in remap[remap >= 0]:
            np.testing.assert_allclose(
                force[cid_out, :3].astype(np.float64),
                f_world,
                atol=1e-5,
                err_msg=f"Existing-path round-trip: cid={cid_out} linear force mismatch",
            )
            np.testing.assert_allclose(
                force[cid_out, 3:].astype(np.float64),
                expected_torque[cid_out],
                atol=1e-5,
                err_msg=f"Existing-path round-trip: cid={cid_out} torque mismatch",
            )

    def test_04_roundtrip_force(self):
        """Round-trip forces through both K->N paths and both N->K branches.

        For each ``(scene, path)`` combination, seed Newton with a constant
        linear ``f_world`` and verify that the round-trip recovers the linear
        force on the correct Newton body0 and the torque expected from the
        reconstructed moment arm.

        Scenes used:

        - ``build_nunchaku_scene`` -- ground plane is re-ordered by Newton
          to ``shape0``, so N->K takes the no-swap branch.
        - ``build_two_box_stack_scene`` -- both bodies are dynamic and N->K
          again takes the no-swap branch.
        - ``build_dynamic_static_sphere_scene`` -- the static (world)
          sphere stays at ``shape1`` (Newton does not reorder non-ground
          static shapes), so N->K takes the swap branch (``bid_1 < 0``).

        Active path: writes the wrench on Newton's output body0, which is
        Kamino A. In the no-swap cases that equals Newton's original
        body0, so ``force_rt.linear = +f_world``. In the swap case the
        output body0 becomes the body originally identified by ``shape1``
        (Newton's third law gives ``force_rt.linear = -f_world``).

        Existing path: keeps Newton's original ``shape0``/``point0`` and
        detects the swap via the preserved ``shape0``, so it always
        recovers ``+f_world`` on the original body0.
        """
        f_world = np.array([0.25, -0.4, 0.65], dtype=np.float64)

        # Active path -- no-swap scenes preserve sign, swap scene flips it.
        self._run_active_roundtrip(build_nunchaku_scene, f_world, expected_linear_sign=+1)
        self._run_active_roundtrip(build_two_box_stack_scene, f_world, expected_linear_sign=+1)
        self._run_active_roundtrip(build_dynamic_static_sphere_scene, f_world, expected_linear_sign=-1)

        # Existing path -- always recovers +f_world on the original body0.
        self._run_existing_roundtrip(build_nunchaku_scene, f_world)
        self._run_existing_roundtrip(build_two_box_stack_scene, f_world)
        self._run_existing_roundtrip(build_dynamic_static_sphere_scene, f_world)

    ###
    # Multi-world
    ###

    def test_05_multi_world(self):
        """Multi-world N->K->N round-trip preserves geometry across heterogeneous worlds.

        Scene layout (ground plane added first, shared across all worlds):
          - World 0: nunchaku (2 boxes + 1 sphere)
          - World 1: nunchaku (2 boxes + 1 sphere)
          - World 2: single box
        Expected total: 22 contacts (9 + 9 + 4).

        Also runs a force-seeding sanity check to ensure the multi-world
        wrench round-trip preserves the linear force on the (swap-branch)
        ground contacts.
        """
        f_world = np.array([0.0, 0.0, 1.25], dtype=np.float64)

        nunchaku_blueprint = build_nunchaku_scene(ground=False)
        box_blueprint = build_two_box_stack_scene()  # two boxes, no ground

        scene = ModelBuilder()
        scene.request_contact_attributes("force")
        scene.add_ground_plane()
        scene.add_world(nunchaku_blueprint, xform=wp.transform(p=wp.vec3(0.0, 0.0, 0.0)))
        scene.add_world(nunchaku_blueprint, xform=wp.transform(p=wp.vec3(5.0, 0.0, 0.0)))
        # Single-box blueprint: just take a fresh nunchaku-less box scene with one body.
        single_box = ModelBuilder()
        single_box.request_contact_attributes("force")
        b = single_box.add_link()
        no_gap = ModelBuilder.ShapeConfig(gap=0.0)
        single_box.add_shape_box(b, hx=0.25, hy=0.25, hz=0.25, cfg=no_gap)
        j = single_box.add_joint_free(
            parent=-1,
            child=b,
            parent_xform=wp.transform(p=wp.vec3(0.0, 0.0, 0.25), q=wp.quat_identity()),
            child_xform=wp.transform_identity(),
        )
        single_box.add_articulation([j])
        scene.add_world(single_box, xform=wp.transform(p=wp.vec3(10.0, 0.0, 0.0)))
        # Suppress the unused-name warning from the box stack helper; we keep
        # the call above just to maintain symmetry with builder_fn use elsewhere.
        del box_blueprint

        model = scene.finalize(self.default_device)
        self.assertEqual(model.world_count, 3)

        state = model.state()
        newton.eval_fk(model, model.joint_q, model.joint_qd, state)
        contacts = model.collide(state)
        nc_orig = int(contacts.rigid_contact_count.numpy()[0])
        self.assertEqual(nc_orig, 9 + 9 + 4, f"Expected 22 contacts (9+9+4), got {nc_orig}")

        self._seed_constant_linear_force(contacts, f_world)

        kamino_out = ContactsKamino(capacity=nc_orig + 32, device=self.default_device)
        convert_contacts_newton_to_kamino(model, state, contacts, kamino_out, convert_forces=True)
        nc_kamino = int(kamino_out.model_active_contacts.numpy()[0])
        self.assertGreater(nc_kamino, 0)

        newton_rt = Contacts(
            rigid_contact_max=kamino_out.model_max_contacts_host,
            soft_contact_max=0,
            device=self.default_device,
            requested_attributes={"force"},
        )
        convert_contacts_kamino_to_newton(
            model,
            state,
            kamino_out,
            newton_rt,
            clear_output=True,
            convert_forces=True,
        )
        nc_rt = int(newton_rt.rigid_contact_count.numpy()[0])
        self.assertEqual(nc_rt, nc_kamino)

        # Geometry: round-tripped body-local points map back to Kamino world.
        shape_body = model.shape_body.numpy()
        body_q = state.body_q.numpy()
        shape0 = newton_rt.rigid_contact_shape0.numpy()[:nc_rt]
        shape1 = newton_rt.rigid_contact_shape1.numpy()[:nc_rt]
        point0 = newton_rt.rigid_contact_point0.numpy()[:nc_rt]
        point1 = newton_rt.rigid_contact_point1.numpy()[:nc_rt]
        pos_A = kamino_out.position_A.numpy()[:nc_kamino]
        pos_B = kamino_out.position_B.numpy()[:nc_kamino]
        for i in range(nc_rt):
            b0 = int(shape_body[int(shape0[i])])
            b1 = int(shape_body[int(shape1[i])])
            p0w = _transform_point(body_q[b0], point0[i]) if b0 >= 0 else point0[i]
            p1w = _transform_point(body_q[b1], point1[i]) if b1 >= 0 else point1[i]
            np.testing.assert_allclose(p0w, pos_A[i], atol=1e-4)
            np.testing.assert_allclose(p1w, pos_B[i], atol=1e-4)

        # Force sanity: every contact in this scene is dynamic-vs-ground, and
        # Newton orders the ground as shape0, so the N->K kernel hits the
        # no-swap branch and the active K->N round-trip writes ``+f_world``
        # on the ground (body0 == -1, torque zero).
        force_rt = newton_rt.force.numpy()[:nc_rt]
        for i in range(nc_rt):
            self.assertEqual(int(shape_body[int(shape0[i])]), -1, f"Contact {i}: shape0 must map to ground")
            np.testing.assert_allclose(
                force_rt[i, :3].astype(np.float64),
                f_world,
                atol=1e-5,
                err_msg=f"Contact {i}: linear force mismatch",
            )

    ###
    # Optional Contacts.force handling
    ###

    def test_06_force_optional(self):
        """Both conversion launchers tolerate a missing optional ``Contacts.force``.

        - N->K with ``convert_forces=False`` (the default) and no
          ``Contacts.force`` leaves Kamino's ``reaction`` at zero.
        - K->N active with ``convert_forces=False`` updates geometry only
          and does not require ``Contacts.force``.
        - K->N existing with ``convert_forces=False`` is a no-op.
        """
        model, state, contacts_no_force = self._setup_newton_scene(with_force=False)
        nc = int(contacts_no_force.rigid_contact_count.numpy()[0])
        self.assertGreater(nc, 0)
        self.assertIsNone(contacts_no_force.force, "Test precondition: Contacts.force must be unallocated")

        # N->K without forces -> reactions remain zero.
        kamino = ContactsKamino(capacity=nc + 16, device=self.default_device, remappable=True)
        convert_contacts_newton_to_kamino(model, state, contacts_no_force, kamino)
        nc_kamino = int(kamino.model_active_contacts.numpy()[0])
        self.assertGreater(nc_kamino, 0)
        reactions = kamino.reaction.numpy()[:nc_kamino]
        np.testing.assert_array_equal(
            reactions,
            np.zeros_like(reactions),
            err_msg="Reactions must be zero when Newton.force is unallocated",
        )

        # K->N active path: geometry updates, no force write.
        newton_out_no_force = Contacts(
            rigid_contact_max=kamino.model_max_contacts_host,
            soft_contact_max=0,
            device=self.default_device,
        )
        self.assertIsNone(newton_out_no_force.force)
        convert_contacts_kamino_to_newton(
            model,
            state,
            kamino,
            newton_out_no_force,
            clear_output=True,
            convert_forces=False,
        )
        self.assertEqual(int(newton_out_no_force.rigid_contact_count.numpy()[0]), nc_kamino)
        self.assertIsNone(newton_out_no_force.force)

        # K->N existing path with no force is a no-op (returns without launch).
        # We pre-populate Newton's contacts (geometry only) and check that
        # this call doesn't raise and doesn't allocate force.
        convert_contacts_kamino_to_newton(
            model,
            state,
            kamino,
            newton_out_no_force,
            clear_output=False,
            convert_forces=False,
        )
        self.assertIsNone(newton_out_no_force.force)

    ###
    # Edge cases
    ###

    def test_07_edge_cases(self):
        """Edge cases for the contact conversion pipeline.

        - No-collision scene: Newton produces zero contacts, N->K also
          produces zero Kamino contacts (kernel correctly handles empty
          input).
        - Capacity overflow: when Kamino's capacity is smaller than Newton's
          contact count, the active count saturates at the capacity (no
          out-of-bounds writes), and the atomic-add bookkeeping keeps the
          model count consistent with the capacity.
        - Missing ``remap`` array on the existing-contacts path raises a
          clear ``ValueError`` when ``convert_forces=True``.
        """
        # 1) No-collision scene -> 0 Kamino contacts.
        model_nc, state_nc, contacts_nc = self._setup_newton_scene(
            builder_fn=lambda: build_nunchaku_scene(ground=False),
            with_force=False,
        )
        nc_nc = int(contacts_nc.rigid_contact_count.numpy()[0])
        self.assertEqual(nc_nc, 0, "Nunchaku without ground must produce no Newton contacts")
        kamino_nc = ContactsKamino(capacity=8, device=self.default_device)
        convert_contacts_newton_to_kamino(model_nc, state_nc, contacts_nc, kamino_nc)
        self.assertEqual(int(kamino_nc.model_active_contacts.numpy()[0]), 0)
        del model_nc, state_nc, contacts_nc, kamino_nc

        # 2) Capacity overflow.
        model, state, contacts = self._setup_newton_scene(with_force=False)
        nc = int(contacts.rigid_contact_count.numpy()[0])
        self.assertGreater(nc, 2, "Need a few contacts to test capacity overflow")
        small_capacity = max(1, nc // 3)
        self.assertLess(small_capacity, nc)
        kamino_small = ContactsKamino(capacity=small_capacity, device=self.default_device)
        convert_contacts_newton_to_kamino(model, state, contacts, kamino_small)
        nc_small = int(kamino_small.model_active_contacts.numpy()[0])
        self.assertLessEqual(
            nc_small,
            small_capacity,
            f"Active count {nc_small} must not exceed Kamino capacity {small_capacity}",
        )
        # Geometry must be sound for whatever was kept.
        bid_AB = kamino_small.bid_AB.numpy()[:nc_small]
        for i in range(nc_small):
            self.assertGreaterEqual(int(bid_AB[i, 1]), 0)

        # 3) Missing remap on the existing-contacts path raises.
        kamino_no_remap = ContactsKamino(capacity=nc + 16, device=self.default_device, remappable=False)
        convert_contacts_newton_to_kamino(model, state, contacts, kamino_no_remap)
        contacts_with_force = Contacts(
            rigid_contact_max=nc + 16,
            soft_contact_max=0,
            device=self.default_device,
            requested_attributes={"force"},
        )
        with self.assertRaises(ValueError):
            convert_contacts_kamino_to_newton(
                model,
                state,
                kamino_no_remap,
                contacts_with_force,
                clear_output=False,
                convert_forces=True,
            )


###
# Test execution
###


if __name__ == "__main__":
    # Test setup
    setup_tests()

    # Run all tests
    unittest.main(verbosity=2)
