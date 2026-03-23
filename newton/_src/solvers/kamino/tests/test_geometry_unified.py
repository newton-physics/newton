# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""
Unit tests for `geometry/unified.py`

Tests the unified collision detection pipeline.
"""

import unittest

import numpy as np
import warp as wp

from newton._src.solvers.kamino._src.core.builder import ModelBuilderKamino
from newton._src.solvers.kamino._src.core.data import DataKamino
from newton._src.solvers.kamino._src.core.math import I_3
from newton._src.solvers.kamino._src.core.model import ModelKamino
from newton._src.solvers.kamino._src.core.shapes import SphereShape
from newton._src.solvers.kamino._src.core.state import StateKamino
from newton._src.solvers.kamino._src.core.types import transformf, vec6f
from newton._src.solvers.kamino._src.geometry.contacts import ContactsKamino
from newton._src.solvers.kamino._src.geometry.unified import CollisionPipelineUnifiedKamino
from newton._src.solvers.kamino._src.models.builders import basics, testing
from newton._src.solvers.kamino._src.utils import logger as msg
from newton._src.solvers.kamino.tests import setup_tests, test_context
from newton._src.solvers.kamino.tests.test_geometry_primitive import check_contacts

###
# Constants
###


nominal_expected_contacts_per_shape_pair = {
    ("sphere", "sphere"): 1,
    ("sphere", "cylinder"): 1,
    ("sphere", "cone"): 1,
    ("sphere", "capsule"): 1,
    ("sphere", "box"): 1,
    ("sphere", "ellipsoid"): 1,
    ("sphere", "plane"): 1,
    ("cylinder", "sphere"): 1,
    ("cylinder", "cylinder"): 4,
    ("cylinder", "cone"): 1,
    ("cylinder", "capsule"): 1,
    ("cylinder", "box"): 4,
    ("cylinder", "ellipsoid"): 1,
    ("cylinder", "plane"): 3,  # GJK manifold on the circular face yields 3 points
    ("cone", "sphere"): 1,
    ("cone", "cylinder"): 4,
    ("cone", "cone"): 1,
    ("cone", "capsule"): 1,
    ("cone", "box"): 4,
    ("cone", "ellipsoid"): 1,
    ("cone", "plane"): 4,
    ("capsule", "cone"): 1,
    ("capsule", "capsule"): 1,
    ("capsule", "box"): 1,
    ("capsule", "ellipsoid"): 1,
    ("capsule", "plane"): 1,
    ("box", "cone"): 1,
    ("box", "box"): 4,
    ("box", "ellipsoid"): 1,
    ("box", "plane"): 4,
    ("ellipsoid", "cone"): 1,
    ("ellipsoid", "ellipsoid"): 1,
    ("ellipsoid", "plane"): 1,
}
"""
Defines the expected number of contacts for each supported
shape combination under the following idealized conditions:
- all shapes are perfectly stacked along the vertical (z) axis
- all shapes are centered at the origin in the (x,y) plane
- the geoms are perfectly touching (i.e. penetration is exactly zero)
- all contact margins are set to zero
- all shapes are positioned and oriented in configurations
that would would generate a "nominal" number of contacts per shape combination

Notes:
- We refer to these "nominal" expected contacts as those that are neither the worst-case
(i.e. maximum possible contacts) nor the best-case (i.e. minimum possible contacts).
- An example of a "nominal" configuration is a box-on-box arrangement where two boxes are
perfectly aligned and touching face-to-face, generating 4 contact points. The worst-case
would be if the boxes were slightly offset, generating 8 contact points (i.e. full face-face
contact with 4 points on each face). The best-case would be if the boxes were touching at a
single edge or corner, generating only 1 contact point.
"""


###
# Testing Operations
###


def test_unified_pipeline(
    builder: ModelBuilderKamino,
    expected: dict,
    max_contacts_per_pair: int = 12,
    margin: float = 0.0,
    rtol: float = 1e-6,
    atol: float = 0.0,
    case: str = "",
    broadphase_modes: list[str] | None = None,
    device: wp.DeviceLike = None,
):
    """
    Runs the unified collision detection pipeline using all broad-phase backends
    on a system specified via a ModelBuilderKamino and checks the results.
    """
    # Run the narrow-phase test over each broad-phase backend
    if broadphase_modes is None:
        broadphase_modes = ["nxn", "sap", "explicit"]
    for bp_mode in broadphase_modes:
        msg.info("Testing unified CD on '%s' using '%s'", case, bp_mode)

        # Create a test model and data
        model: ModelKamino = builder.finalize(device)
        data: DataKamino = model.data()
        state: StateKamino = model.state()

        # Create a pipeline
        pipeline = CollisionPipelineUnifiedKamino(
            model=model,
            broadphase=bp_mode,
            default_gap=margin,
            device=device,
        )

        # Create a contacts container using the worst-case capacity of NxN over model-wise geom pairs
        # NOTE: This is required by the unified pipeline when using SAP and NXN broad-phases
        capacity = 2 * max_contacts_per_pair * ((builder.num_geoms * (builder.num_geoms - 1)) // 2)
        contacts = ContactsKamino(capacity=capacity, device=device)
        contacts.clear()

        # Execute the unified collision detection pipeline
        pipeline.collide(data, state, contacts)

        # Optional verbose output
        msg.debug("[%s][%s]: bodies.q_i:\n%s", case, bp_mode, data.bodies.q_i)
        msg.debug("[%s][%s]: contacts.model_active_contacts: %s", case, bp_mode, contacts.model_active_contacts)
        msg.debug("[%s][%s]: contacts.world_active_contacts: %s", case, bp_mode, contacts.world_active_contacts)
        msg.debug("[%s][%s]: contacts.wid: %s", case, bp_mode, contacts.wid)
        msg.debug("[%s][%s]: contacts.cid: %s", case, bp_mode, contacts.cid)
        msg.debug("[%s][%s]: contacts.gid_AB:\n%s", case, bp_mode, contacts.gid_AB)
        msg.debug("[%s][%s]: contacts.bid_AB:\n%s", case, bp_mode, contacts.bid_AB)
        msg.debug("[%s][%s]: contacts.position_A:\n%s", case, bp_mode, contacts.position_A)
        msg.debug("[%s][%s]: contacts.position_B:\n%s", case, bp_mode, contacts.position_B)
        msg.debug("[%s][%s]: contacts.gapfunc:\n%s", case, bp_mode, contacts.gapfunc)
        msg.debug("[%s][%s]: contacts.frame:\n%s", case, bp_mode, contacts.frame)
        msg.debug("[%s][%s]: contacts.material:\n%s", case, bp_mode, contacts.material)

        # Check results
        check_contacts(
            contacts,
            expected,
            rtol=rtol,
            atol=atol,
            case=f"{case} using {bp_mode}",
            header="unified CD pipeline",
        )


def test_unified_pipeline_on_shape_pair(
    testcase: unittest.TestCase,
    shape_pair: tuple[str, str],
    expected_contacts: int,
    distance: float = 0.0,
    margin: float = 0.0,
    builder_kwargs: dict | None = None,
):
    """
    Tests the unified collision detection pipeline on a single shape pair.

    Note:
        This test only checks the number of generated contacts.
    """
    # Set default builder kwargs if none provided
    if builder_kwargs is None:
        builder_kwargs = {}

    # Create a builder for the specified shape pair
    builder = testing.make_single_shape_pair_builder(shapes=shape_pair, distance=distance, **builder_kwargs)

    # Define expected contacts dictionary
    expected = {
        "model_active_contacts": expected_contacts,
        "world_active_contacts": [expected_contacts],
    }

    # Run the narrow-phase test
    test_unified_pipeline(
        builder=builder,
        expected=expected,
        margin=margin,
        case=f"shape_pair='{shape_pair}'",
        device=testcase.default_device,
    )


###
# Tests
###


class TestCollisionPipelineUnified(unittest.TestCase):
    def setUp(self):
        if not test_context.setup_done:
            setup_tests(clear_cache=False)
        self.default_device = wp.get_device(test_context.device)
        self.verbose = test_context.verbose  # Set to True for detailed output
        self.skip_buggy_tests = False  # Set to True to skip known-buggy tests

        # Set debug-level logging to print verbose test output to console
        if self.verbose:
            msg.set_log_level(msg.LogLevel.INFO)
        else:
            msg.reset_log_level()

        # Generate a list of supported shape pairs
        self.supported_shape_pairs = nominal_expected_contacts_per_shape_pair.keys()
        msg.debug("Supported shape pairs for unified pipeline tests:\n%s", self.supported_shape_pairs)

    def tearDown(self):
        self.default_device = None
        if self.verbose:
            msg.reset_log_level()

    def test_00_on_specific_primitive_shape_pair(self):
        """
        Tests the unified collision pipeline on a specific primitive shape pair.

        NOTE: This is mainly for debugging purposes, where we can easily test a specific case.
        """
        if self.skip_buggy_tests:
            self.skipTest("Skipping 'specific_primitive_shape_pair' test")

        # Define the specific shape pair to test
        shape_pair = ("cylinder", "ellipsoid")
        msg.info(f"Testing narrow-phase tests with exact boundaries on {shape_pair}")

        # Define any special kwargs for specific shape pairs
        kwargs = {
            "top_dims": (0.5, 1.0),  # radius, height of cylinder
            "bottom_dims": (1.0, 1.0, 0.5),  # radii(a,b,c) of ellipsoid
        }

        # Retrieve the nominal expected contacts for the shape pair
        # TODO: This should be =1, but generates =4
        expected_contacts = 1

        # Run the narrow-phase test on the shape pair
        test_unified_pipeline_on_shape_pair(
            self,
            shape_pair=shape_pair,
            expected_contacts=expected_contacts,
            margin=0.0,  # No contact margin
            distance=1.0e-8,  # Exactly touching
            builder_kwargs=kwargs,
        )

    def test_01_on_each_primitive_shape_pair_touching(self):
        """
        Tests the unified collision pipeline on each supported primitive
        shape pair when placed exactly at their contact boundaries.
        """
        msg.info("Testing unified pipeline tests with exact boundaries")
        # Each shape pair in its own world with
        for shape_pair in self.supported_shape_pairs:
            # Define any special kwargs for specific shape pairs
            kwargs = {}
            if shape_pair == ("box", "box"):
                # NOTE: To asses "nominal" contacts for box-box,
                # we need to specify larger box dimensions for
                # the bottom box to avoid contacts on edges
                kwargs["bottom_dims"] = (2.0, 2.0, 1.0)

            # Retrieve the nominal expected contacts for the shape pair
            expected_contacts = nominal_expected_contacts_per_shape_pair.get(shape_pair, 0)

            # Run the narrow-phase test on the shape pair
            test_unified_pipeline_on_shape_pair(
                self,
                shape_pair=shape_pair,
                expected_contacts=expected_contacts,
                margin=1.0e-5,  # Default contact margin
                distance=0.0,  # Exactly touching
                builder_kwargs=kwargs,
            )

    def test_02_on_each_primitive_shape_pair_apart(self):
        """
        Tests the unified collision pipeline on each
        supported primitive shape pair when placed apart.
        """
        msg.info("Testing unified pipeline tests with shapes apart")
        # Each shape pair in its own world with
        for shape_pair in self.supported_shape_pairs:
            test_unified_pipeline_on_shape_pair(
                self,
                shape_pair=shape_pair,
                expected_contacts=0,
                margin=0.0,  # No contact margin
                distance=1e-6,  # Shapes apart with epsilon distance
            )

    def test_03_on_each_primitive_shape_pair_apart_with_margin(self):
        """
        Tests the unified collision pipeline on each supported
        primitive shape pair when placed apart but with contact margin.
        """
        msg.info("Testing unified pipeline tests with shapes apart")
        # Each shape pair in its own world with
        # - zero distance: (i.e., exactly touching)
        # - zero margin: no preemption of collisions
        for shape_pair in self.supported_shape_pairs:
            # Define any special kwargs for specific shape pairs
            kwargs = {}
            if shape_pair == ("box", "box"):
                # NOTE: To asses "nominal" contacts for box-box,
                # we need to specify larger box dimensions for
                # the bottom box to avoid contacts on edges
                kwargs["bottom_dims"] = (2.0, 2.0, 1.0)

            # Retrieve the nominal expected contacts for the shape pair
            expected_contacts = nominal_expected_contacts_per_shape_pair.get(shape_pair, 0)

            # Run the narrow-phase test on the shape pair
            test_unified_pipeline_on_shape_pair(
                self,
                shape_pair=shape_pair,
                expected_contacts=expected_contacts,
                margin=1e-5,  # Contact margin
                distance=1e-6,  # Shapes apart
                builder_kwargs=kwargs,
            )

    ###
    # Tests for special cases of shape combinations/configurations
    ###

    def test_04_sphere_on_sphere_detailed(self):
        """
        Tests all unified pipeline output data for the case of two spheres
        stacked along the vertical (z) axis, centered at the origin
        in the (x,y) plane, and slightly penetrating each other.
        """
        if self.skip_buggy_tests:
            self.skipTest("Skipping `sphere_on_sphere_detailed` test")

        # NOTE: We set to negative value to move the geoms into each other,
        # i.e. move the bottom geom upwards and the top geom downwards.
        distance = 0.0

        # Define expected contact data
        expected = {
            "model_active_contacts": 1,
            "world_active_contacts": [1],
            "gid_AB": np.array([[0, 1]], dtype=np.int32),
            "bid_AB": np.array([[0, 1]], dtype=np.int32),
            "position_A": np.array([[0.0, 0.0, 0.5 * abs(distance)]], dtype=np.float32),
            "position_B": np.array([[0.0, 0.0, -0.5 * abs(distance)]], dtype=np.float32),
            "gapfunc": np.array([[0.0, 0.0, 1.0, distance]], dtype=np.float32),
            "frame": np.array([[0.0, 0.0, 0.0, 1.0]], dtype=np.float32),
        }

        # Create a builder for the specified shape pair
        builder = testing.make_single_shape_pair_builder(
            shapes=("sphere", "sphere"),
            distance=distance,
        )

        # Run the narrow-phase test on the shape pair
        test_unified_pipeline(
            builder=builder,
            expected=expected,
            case="sphere_on_sphere_detailed",
            device=self.default_device,
            # rtol=0.0,
            # atol=1e-5,
        )

    def test_05_box_on_box_simple(self):
        """
        Tests unified pipeline output contacts for the case of two boxes
        stacked along the vertical (z) axis, centered at the origin in
        the (x,y) plane, and slightly penetrating each other.

        This test makes the bottom box larger in the (x,y) dimensions
        to ensure that only four contact points are generated at the
        face of the top box.
        """
        # NOTE: We set to negative value to move the geoms into each other,
        # i.e. move the bottom geom upwards and the top geom downwards.
        distance = -0.01

        # Define expected contact data
        expected = {
            "model_active_contacts": 4,
            "world_active_contacts": [4],
            "gid_AB": np.tile(np.array([0, 1], dtype=np.int32), reps=(4, 1)),
            "bid_AB": np.tile(np.array([0, 1], dtype=np.int32), reps=(4, 1)),
            "position_A": np.array(
                [
                    [-0.5, -0.5, 0.5 * abs(distance)],
                    [0.5, -0.5, 0.5 * abs(distance)],
                    [0.5, 0.5, 0.5 * abs(distance)],
                    [-0.5, 0.5, 0.5 * abs(distance)],
                ],
                dtype=np.float32,
            ),
            "position_B": np.array(
                [
                    [-0.5, -0.5, -0.5 * abs(distance)],
                    [0.5, -0.5, -0.5 * abs(distance)],
                    [0.5, 0.5, -0.5 * abs(distance)],
                    [-0.5, 0.5, -0.5 * abs(distance)],
                ],
                dtype=np.float32,
            ),
            "gapfunc": np.tile(np.array([0.0, 0.0, 1.0, distance], dtype=np.float32), reps=(4, 1)),
            "frame": np.tile(np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32), reps=(4, 1)),
        }

        # Create a builder for the specified shape pair
        builder = testing.make_single_shape_pair_builder(
            shapes=("box", "box"),
            distance=distance,
            bottom_dims=(2.0, 2.0, 1.0),  # Larger bottom box
        )

        # Run the narrow-phase test on the shape pair
        test_unified_pipeline(
            builder=builder,
            expected=expected,
            case="box_on_box_simple",
            device=self.default_device,
            rtol=0.0,
            atol=1e-5,
        )

    def test_07_on_box_on_box_vertex_on_face(self):
        """
        Tests the unified pipeline on the special case of two boxes
        stacked along the vertical (z) axis, centered at the origin
        in the (x,y) plane, and the top box rotated so that one of
        its lowest vertex is touching the top face of the bottom box.
        """
        # NOTE: We set to negative value to move the geoms into each other,
        # i.e. move the bottom geom upwards and the top geom downwards.
        penetration = -0.01

        # Define expected contact data
        expected = {
            "num_contacts": 1,
            "gid_AB": np.tile(np.array([0, 1], dtype=np.int32), reps=(1, 1)),
            "bid_AB": np.tile(np.array([0, 1], dtype=np.int32), reps=(1, 1)),
            "position_A": np.array([[0.0, 0.0, 0.0]], dtype=np.float32),
            "position_B": np.array([[0.0, 0.0, -abs(penetration)]], dtype=np.float32),
            "gapfunc": np.tile(np.array([0.0, 0.0, 1.0, penetration], dtype=np.float32), reps=(1, 1)),
            "frame": np.tile(np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32), reps=(1, 1)),
        }

        # Create a builder for the specified shape pair
        builder = testing.make_single_shape_pair_builder(
            shapes=("box", "box"),
            top_xyz=[0.0, 0.0, 0.5 * np.sqrt(3) + 0.5],
            top_rpy=[np.pi / 4, -np.arctan(1.0 / np.sqrt(2)), 0.0],
        )

        # Run the narrow-phase test on the shape pair
        test_unified_pipeline(
            builder=builder,
            expected=expected,
            case="box_on_box_vertex_on_face",
            device=self.default_device,
            rtol=1e-5,
            atol=1e-6,
        )

    def test_08_on_boxes_nunchaku(self):
        """
        Tests the unified collision detection pipeline on the boxes_nunchaku model.
        """
        # Define expected contact data
        expected = {
            "model_active_contacts": 9,
            "world_active_contacts": [9],
        }

        # Create a builder for the specified shape pair
        builder = basics.build_boxes_nunchaku()

        # Run the narrow-phase test on the shape pair
        # Note: Use small margin to handle floating point precision for touching contacts
        test_unified_pipeline(
            builder=builder,
            expected=expected,
            case="boxes_nunchaku",
            broadphase_modes=["explicit"],
            margin=1e-6,
            device=self.default_device,
        )


class TestUnifiedWriterContactDataRegression(unittest.TestCase):
    """
    Regression tests for the unified writer's ContactData API usage.

    The writer previously referenced ``thickness_a/b`` fields on
    :class:`ContactData` which no longer exist; the correct fields are
    ``radius_eff_a/b`` and ``margin_a/b``.  It also used a max-based
    margin threshold instead of additive per-shape gap.  These tests
    verify the corrected behaviour end-to-end.
    """

    def setUp(self):
        if not test_context.setup_done:
            setup_tests(clear_cache=False)
        self.default_device = wp.get_device(test_context.device)

    def tearDown(self):
        self.default_device = None

    def _run_pipeline(self, builder: ModelBuilderKamino, default_gap=0.0):
        model = builder.finalize(self.default_device)
        data = model.data()
        state = model.state()
        pipeline = CollisionPipelineUnifiedKamino(
            model=model,
            broadphase="explicit",
            default_gap=default_gap,
            device=self.default_device,
        )
        n_geoms = builder.num_geoms
        capacity = 8 * ((n_geoms * (n_geoms - 1)) // 2)
        contacts = ContactsKamino(capacity=max(capacity, 8), device=self.default_device)
        contacts.clear()
        pipeline.collide(data, state, contacts)
        return contacts

    def test_00_touching_spheres_produces_contact(self):
        """Two touching spheres must generate exactly one contact with d ≈ 0."""
        builder = testing.make_single_shape_pair_builder(
            shapes=("sphere", "sphere"),
            distance=0.0,
        )
        contacts = self._run_pipeline(builder, default_gap=1e-5)
        active = contacts.model_active_contacts.numpy()[0]
        self.assertEqual(active, 1, "Touching spheres should produce one contact")

        gapfunc = contacts.gapfunc.numpy()[0]
        self.assertAlmostEqual(float(gapfunc[3]), 0.0, places=4, msg="gapfunc.w should be ≈ 0 for touching spheres")

    def test_01_penetrating_spheres_negative_distance(self):
        """Penetrating spheres must produce a negative gapfunc.w."""
        penetration = -0.02
        builder = testing.make_single_shape_pair_builder(
            shapes=("sphere", "sphere"),
            distance=penetration,
        )
        contacts = self._run_pipeline(builder, default_gap=1e-5)
        active = contacts.model_active_contacts.numpy()[0]
        self.assertEqual(active, 1)

        gapfunc = contacts.gapfunc.numpy()[0]
        self.assertLess(float(gapfunc[3]), 0.0, "gapfunc.w must be negative for penetrating spheres")

    def test_02_gap_retains_nearby_contact(self):
        """Contact within detection gap must be retained by the writer."""
        separation = 1e-6
        builder = testing.make_single_shape_pair_builder(
            shapes=("sphere", "sphere"),
            distance=separation,
        )
        for geom in builder.geoms:
            geom.gap = 0.01
        contacts = self._run_pipeline(builder)
        active = contacts.model_active_contacts.numpy()[0]
        self.assertEqual(active, 1, "Contact within gap must be retained")

    def test_03_gap_rejects_distant_contact(self):
        """ContactsKamino beyond the detection gap must be rejected."""
        separation = 0.1
        builder = testing.make_single_shape_pair_builder(
            shapes=("sphere", "sphere"),
            distance=separation,
        )
        for geom in builder.geoms:
            geom.gap = 0.001
        contacts = self._run_pipeline(builder)
        active = contacts.model_active_contacts.numpy()[0]
        self.assertEqual(active, 0, "Contact beyond gap must be rejected")


class TestUnifiedPipelineNxnBroadphase(unittest.TestCase):
    """Tests verifying NXN broadphase correctness with collision radii and filter pairs."""

    def setUp(self):
        if not test_context.setup_done:
            setup_tests(clear_cache=False)
        self.default_device = wp.get_device(test_context.device)

    def tearDown(self):
        self.default_device = None

    def _make_two_sphere_builder(self, group_a=1, collides_a=1, group_b=1, collides_b=1, same_body=False):
        """Helper: build a single-world scene with two spheres near each other."""
        builder = ModelBuilderKamino()
        builder.add_world()
        bid_a = builder.add_rigid_body(
            m_i=1.0,
            i_I_i=I_3,
            q_i_0=transformf(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
            u_i_0=vec6f(0.0),
        )
        if same_body:
            bid_b = bid_a
        else:
            bid_b = builder.add_rigid_body(
                m_i=1.0,
                i_I_i=I_3,
                q_i_0=transformf(0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
                u_i_0=vec6f(0.0),
            )
        builder.add_geometry(body=bid_a, shape=SphereShape(radius=0.5), group=group_a, collides=collides_a)
        builder.add_geometry(body=bid_b, shape=SphereShape(radius=0.5), group=group_b, collides=collides_b)
        return builder

    def test_00_nxn_sphere_on_plane_generates_contacts(self):
        """Sphere resting on a plane via NXN broadphase must produce contacts.

        Validates that shape_collision_radius is populated correctly for
        infinite planes (which need a large bounding-sphere radius for
        AABB-based broadphase modes to detect them).
        """
        builder = testing.make_single_shape_pair_builder(
            shapes=("sphere", "plane"),
            distance=0.0,
        )

        expected = {
            "model_active_contacts": 1,
            "world_active_contacts": [1],
        }

        test_unified_pipeline(
            builder=builder,
            expected=expected,
            margin=1e-5,
            case="nxn_sphere_on_plane",
            broadphase_modes=["nxn"],
            device=self.default_device,
        )

    def test_01_nxn_box_on_plane_generates_contacts(self):
        """Box on a plane via NXN broadphase must produce contacts."""
        builder = testing.make_single_shape_pair_builder(
            shapes=("box", "plane"),
            distance=0.0,
        )

        expected = {
            "model_active_contacts": 4,
            "world_active_contacts": [4],
        }

        test_unified_pipeline(
            builder=builder,
            expected=expected,
            margin=1e-5,
            case="nxn_box_on_plane",
            broadphase_modes=["nxn"],
            device=self.default_device,
        )

    def test_02_nxn_excludes_non_collidable_pairs(self):
        """NXN broadphase must exclude pairs whose group/collides bitmasks do not overlap.

        Creates two spheres in the same world but with non-overlapping
        collision groups so that they should never collide.
        """
        builder = self._make_two_sphere_builder(group_a=0b01, collides_a=0b01, group_b=0b10, collides_b=0b10)

        model = builder.finalize(self.default_device)
        data = model.data()
        state = model.state()

        pipeline = CollisionPipelineUnifiedKamino(
            model=model,
            broadphase="nxn",
            default_gap=1.0,
            device=self.default_device,
        )

        n_geoms = builder.num_geoms
        capacity = 12 * ((n_geoms * (n_geoms - 1)) // 2)
        contacts = ContactsKamino(capacity=max(capacity, 12), device=self.default_device)
        contacts.clear()

        pipeline.collide(data, state, contacts)

        active = contacts.model_active_contacts.numpy()[0]
        self.assertEqual(active, 0, "Non-collidable groups must produce zero contacts via NXN")

    def test_03_nxn_same_body_excluded(self):
        """NXN broadphase must exclude same-body shape pairs.

        Attaches two collision geometries to the same body and verifies
        that no self-collision contacts are produced.
        """
        builder = self._make_two_sphere_builder(same_body=True)

        model = builder.finalize(self.default_device)
        data = model.data()
        state = model.state()

        pipeline = CollisionPipelineUnifiedKamino(
            model=model,
            broadphase="nxn",
            default_gap=1.0,
            device=self.default_device,
        )

        n_geoms = builder.num_geoms
        capacity = 12 * ((n_geoms * (n_geoms - 1)) // 2)
        contacts = ContactsKamino(capacity=max(capacity, 12), device=self.default_device)
        contacts.clear()

        pipeline.collide(data, state, contacts)

        active = contacts.model_active_contacts.numpy()[0]
        self.assertEqual(active, 0, "Same-body shapes must not collide via NXN broadphase")


###
# Test execution
###


if __name__ == "__main__":
    # Test setup
    setup_tests()

    # Run all tests
    unittest.main(verbosity=2)
