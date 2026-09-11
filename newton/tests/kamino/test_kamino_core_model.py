# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""
Unit tests for the :class:`ModelKamino` class, model conversion, and related functionality.
"""

import unittest

import numpy as np
import warp as wp

import newton
import newton.tests.kamino.utils.checks as test_util_checks
from newton import Control, Model, ModelBuilder, State
from newton._src.solvers.kamino._src.core import ControlKamino, ModelKamino, StateKamino
from newton._src.solvers.kamino._src.core.bodies import convert_body_com_to_origin, convert_body_origin_to_com
from newton._src.solvers.kamino._src.core.conversions import (
    compute_material_first_shape,
    convert_model_materials,
    convert_target_coords_to_target_dofs,
)
from newton._src.solvers.kamino._src.core.joints import (
    JOINT_TAUMAX,
    DofActuationPath,
    JointActuationType,
)
from newton._src.solvers.kamino._src.core.model import ModelKamino
from newton._src.solvers.kamino._src.utils import logger as msg
from newton._src.solvers.kamino.solver_kamino import SolverKamino
from newton.tests import get_kamino_basics_asset
from newton.tests.kamino import setup_tests, test_context
from newton.tests.kamino.utils import print as print_utils
from newton.tests.utils.basics import (
    build_box_pendulum,
    build_boxes_fourbar,
    build_boxes_hinged,
    build_boxes_nunchaku,
    make_basics_heterogeneous_builder,
)

###
# Tests
###


class TestModel(unittest.TestCase):
    def setUp(self):
        if not test_context.setup_done:
            setup_tests(clear_cache=False)
        self.default_device = wp.get_device(test_context.device)
        self.verbose = test_context.verbose  # Set to True for verbose output

        # Set debug-level logging to print verbose test output to console
        if self.verbose:
            print("\n")  # Add newline before test output for better readability
            msg.set_log_level(msg.LogLevel.DEBUG)
        else:
            msg.reset_log_level()

    def tearDown(self):
        self.default_device = None
        if self.verbose:
            msg.reset_log_level()

    def test_01_single_model(self):
        # Create a model builder
        builder = build_boxes_hinged()

        # Finalize the model
        model: ModelKamino = ModelKamino.from_newton(builder.finalize(device=self.default_device))
        if self.verbose:
            print("")  # Add a newline for better readability
            print_utils.print_model_info(model)

        # Create a model state
        state = model.data()
        if self.verbose:
            print("")  # Add a newline for better readability
            print_utils.print_data_info(state)

        # Check the model info entries
        self.assertEqual(model.size.sum_of_num_bodies, builder.body_count)
        self.assertEqual(model.size.sum_of_num_joints, builder.joint_count)
        self.assertEqual(model.size.sum_of_num_geoms, builder.shape_count)
        self.assertEqual(model.device, self.default_device)

    def test_02_double_model(self):
        # Create a model builder
        builder1 = build_boxes_hinged()
        builder2 = build_boxes_nunchaku()

        # Compute the total number of elements from the two builders
        total_nb = builder1.body_count + builder2.body_count
        total_nj = builder1.joint_count + builder2.joint_count
        total_ng = builder1.shape_count + builder2.shape_count

        # Add the second builder to the first one
        builder1.add_builder(builder2)

        # Finalize the model
        model: ModelKamino = ModelKamino.from_newton(builder1.finalize(device=self.default_device))
        if self.verbose:
            print("")  # Add a newline for better readability
            print_utils.print_model_info(model)

        # Create a model state
        data = model.data()
        if self.verbose:
            print("")  # Add a newline for better readability
            print_utils.print_data_info(data)

        # Check the model info entries
        self.assertEqual(model.size.sum_of_num_bodies, total_nb)
        self.assertEqual(model.size.sum_of_num_joints, total_nj)
        self.assertEqual(model.size.sum_of_num_geoms, total_ng)

    def test_03_homogeneous_model(self):
        # Constants
        num_worlds = 4

        # Create a model builder
        builder = ModelBuilder()
        builder.replicate(builder=build_boxes_hinged(), world_count=num_worlds)

        # Finalize the model
        model: ModelKamino = ModelKamino.from_newton(builder.finalize(device=self.default_device))
        if self.verbose:
            print("")  # Add a newline for better readability
            print_utils.print_model_info(model)

        # Create a model state
        state = model.data()
        if self.verbose:
            print("")  # Add a newline for better readability
            print_utils.print_data_info(state)

        # Check the model info entries
        self.assertEqual(model.size.sum_of_num_bodies, num_worlds * 2)
        self.assertEqual(model.size.sum_of_num_joints, num_worlds * 2)  # Includes free joint
        self.assertEqual(model.size.sum_of_num_geoms, num_worlds * 3)
        self.assertEqual(model.device, self.default_device)

    def test_04_hetereogeneous_model(self):
        # Create a model builder
        builder = make_basics_heterogeneous_builder()
        num_worlds = builder.world_count

        # Finalize the model
        model: ModelKamino = ModelKamino.from_newton(builder.finalize(device=self.default_device))
        if self.verbose:
            print("")  # Add a newline for better readability
            print_utils.print_model_info(model)
            print("")  # Add a newline for better readability
            print_utils.print_model_bodies(model)
            print("")  # Add a newline for better readability
            print_utils.print_model_joints(model)

        # Create a model state
        state = model.data()
        if self.verbose:
            print("")  # Add a newline for better readability
            print_utils.print_data_info(state)

        # Check the model info entries
        self.assertEqual(model.info.num_worlds, num_worlds)


class TestModelConversions(unittest.TestCase):
    def setUp(self):
        if not test_context.setup_done:
            setup_tests(clear_cache=False)
        self.default_device = wp.get_device(test_context.device)
        self.verbose = test_context.verbose  # Set to True to enable verbose output

        # Set debug-level logging to print verbose test output to console
        if self.verbose:
            print("\n")  # Add newline before test output for better readability
            msg.set_log_level(msg.LogLevel.INFO)  # TODO @nvtw: set this to DEBUG when investigating noted issues
        else:
            msg.reset_log_level()

    def tearDown(self):
        self.default_device = None
        if self.verbose:
            msg.reset_log_level()

    def test_01_model_conversions_consistency_fourbar_from_builder(self):
        """
        Test that the Newton to Kamino model conversion of a fourbar model
        (single- and multi-world) is consistent.
        """
        builder_single: ModelBuilder = ModelBuilder()
        builder_single.default_shape_cfg.margin = 0.0
        builder_single.default_shape_cfg.gap = 0.0
        build_boxes_fourbar(
            builder=builder_single,
            z_offset=0.0,
            fixedbase=False,
            floatingbase=True,
            limits=True,
            ground=True,
            dynamic_joints=False,
            implicit_pd=False,
            new_world=True,
            actuator_ids=[1, 3],
        )

        builder_newton: ModelBuilder = ModelBuilder()
        SolverKamino.register_custom_attributes(builder_newton)
        builder_newton.replicate(builder=builder_single, world_count=2)

        model_newton: Model = builder_newton.finalize(skip_validation_joints=True, device=self.default_device)
        model_kamino: ModelKamino = ModelKamino.from_newton(model_newton)

        test_util_checks.assert_model_conversion_consistency(self, model_newton, model_kamino)
        test_util_checks.assert_model_info_size_consistency(self, model_kamino)

    def test_02_model_conversions_consistency_material_variation_from_builder(self):
        """
        Test that the Newton to Kamino model conversion of a fourbar model with
        distinct per-shape friction/restitution overrides is consistent,
        exercising the materials dedup path.
        """
        builder_newton: ModelBuilder = ModelBuilder()
        SolverKamino.register_custom_attributes(builder_newton)
        builder_newton.default_shape_cfg.margin = 0.0
        builder_newton.default_shape_cfg.gap = 0.0

        build_boxes_fourbar(
            builder=builder_newton,
            z_offset=0.0,
            fixedbase=False,
            floatingbase=True,
            limits=True,
            ground=True,
            dynamic_joints=False,
            implicit_pd=False,
            new_world=True,
            actuator_ids=[1, 3],
        )

        # Give each shape distinct material properties
        restitution = [0.1, 0.2, 0.3, 0.4, 0.5]
        mu = [0.5, 0.6, 0.7, 0.8, 0.9]
        self.assertEqual(len(builder_newton.shape_material_restitution), len(restitution))
        builder_newton.shape_material_restitution = list(restitution)
        builder_newton.shape_material_mu = list(mu)

        model_newton: Model = builder_newton.finalize(skip_validation_joints=True, device=self.default_device)
        model_kamino: ModelKamino = ModelKamino.from_newton(model_newton)

        test_util_checks.assert_model_conversion_consistency(self, model_newton, model_kamino)
        test_util_checks.assert_model_info_size_consistency(self, model_kamino)

    def test_03_model_conversions_consistency_fourbar_from_usd(self):
        """
        Test that the Newton to Kamino model conversion of a fourbar model
        loaded from USD is consistent.
        """
        asset_file = get_kamino_basics_asset("boxes_fourbar.usda")

        builder_newton: ModelBuilder = ModelBuilder()
        SolverKamino.register_custom_attributes(builder_newton)
        builder_newton.default_shape_cfg.margin = 0.0
        builder_newton.default_shape_cfg.gap = 0.0

        builder_newton.begin_world()
        builder_newton.add_usd(
            source=asset_file,
            joint_ordering=None,
            force_show_colliders=True,
            force_position_velocity_actuation=True,
        )
        builder_newton.end_world()

        model_newton: Model = builder_newton.finalize(skip_validation_joints=True, device=self.default_device)
        model_kamino: ModelKamino = ModelKamino.from_newton(model_newton)

        test_util_checks.assert_model_conversion_consistency(self, model_newton, model_kamino)
        test_util_checks.assert_model_info_size_consistency(self, model_kamino)

    def test_04_model_conversions_consistency_box_on_plane_materials_from_usd(self):
        """
        Test that the Newton to Kamino model conversion of a box-on-plane model
        with distinct per-shape materials loaded from USD is consistent,
        exercising both multi-world duplication and the materials dedup path.
        """
        asset_file = get_kamino_basics_asset("box_on_plane.usda")

        builder_single: ModelBuilder = ModelBuilder()
        builder_single.default_shape_cfg.margin = 0.0
        builder_single.default_shape_cfg.gap = 0.0
        builder_single.begin_world()
        builder_single.add_usd(
            source=asset_file,
            joint_ordering=None,
            force_show_colliders=True,
            force_position_velocity_actuation=True,
        )
        builder_single.end_world()

        builder_newton: ModelBuilder = ModelBuilder()
        SolverKamino.register_custom_attributes(builder_newton)
        builder_newton.replicate(builder=builder_single, world_count=2)

        model_newton: Model = builder_newton.finalize(skip_validation_joints=True, device=self.default_device)
        model_kamino: ModelKamino = ModelKamino.from_newton(model_newton)

        test_util_checks.assert_model_conversion_consistency(self, model_newton, model_kamino)
        test_util_checks.assert_model_info_size_consistency(self, model_kamino)

    def test_10_model_conversions_base_assignment_non_floating_root(self):
        """
        Test per-world base assignment when articulation roots are not unary free joints.

        A free-rooted articulation following a fixed-rooted one still provides the
        world's floating base; a world whose articulations are fixed-rooted or
        rooted by a free joint with a body parent gets no base.
        """

        def build_model(free_root: str | None) -> Model:
            builder: ModelBuilder = ModelBuilder()
            SolverKamino.register_custom_attributes(builder)
            body_fixed = builder.add_link()
            builder.add_shape_box(body_fixed)
            joint_fixed = builder.add_joint_fixed(parent=-1, child=body_fixed)
            builder.add_articulation([joint_fixed])
            if free_root is not None:
                body_free = builder.add_link(xform=wp.transform(wp.vec3(0.0, 0.0, 2.0), wp.quat_identity()))
                builder.add_shape_box(body_free)
                parent = -1 if free_root == "world" else body_fixed
                joint_free = builder.add_joint_free(child=body_free, parent=parent)
                builder.add_articulation([joint_free])
            return builder.finalize(device=self.default_device)

        model_kamino = ModelKamino.from_newton(build_model(free_root="world"))
        self.assertEqual(model_kamino.info.base_body_index.numpy().tolist(), [1])
        self.assertEqual(model_kamino.info.base_joint_index.numpy().tolist(), [1])
        self.assertFalse(model_kamino.info.has_world_without_base_body)

        for free_root in (None, "body"):
            model_kamino = ModelKamino.from_newton(build_model(free_root=free_root))
            self.assertEqual(model_kamino.info.base_body_index.numpy().tolist(), [-1])
            self.assertEqual(model_kamino.info.base_joint_index.numpy().tolist(), [-1])
            self.assertTrue(model_kamino.info.has_world_without_base_body)

    def test_11_model_conversions_arbitrary_axis(self):
        """
        Test that Newton→Kamino conversion succeeds for a revolute joint
        with an arbitrary (non-canonical) axis, e.g. ``(1, 1, 0)``.
        """
        builder_newton: ModelBuilder = ModelBuilder()
        SolverKamino.register_custom_attributes(builder_newton)
        builder_newton.default_shape_cfg.margin = 0.0
        builder_newton.default_shape_cfg.gap = 0.0

        builder_newton.begin_world()

        # Parent body at origin
        bid0 = builder_newton.add_link(
            label="base",
            mass=1.0,
            inertia=wp.mat33f(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
            xform=wp.transformf(wp.vec3f(0.0, 0.0, 0.0), wp.quat_identity(dtype=wp.float32)),
            lock_inertia=True,
        )
        builder_newton.add_shape_box(label="box_base", body=bid0, hx=0.05, hy=0.05, hz=0.05)

        # Child body offset along z
        bid1 = builder_newton.add_link(
            label="pendulum",
            mass=1.0,
            inertia=wp.mat33f(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
            xform=wp.transformf(wp.vec3f(0.0, 0.0, 0.5), wp.quat_identity(dtype=wp.float32)),
            lock_inertia=True,
        )
        builder_newton.add_shape_box(label="box_pend", body=bid1, hx=0.05, hy=0.05, hz=0.25)

        # Fix the base to the world
        builder_newton.add_joint_fixed(
            label="world_to_base",
            parent=-1,
            child=bid0,
            parent_xform=wp.transform_identity(dtype=wp.float32),
            child_xform=wp.transform_identity(dtype=wp.float32),
        )

        # Diagonal revolute axis (non-canonical)
        axis_vec = wp.vec3(1.0, 1.0, 0.0)
        builder_newton.add_joint_revolute(
            label="base_to_pendulum",
            parent=bid0,
            child=bid1,
            axis=axis_vec,
            parent_xform=wp.transformf(wp.vec3f(0.0, 0.0, 0.25), wp.quat_identity(dtype=wp.float32)),
            child_xform=wp.transformf(wp.vec3f(0.0, 0.0, -0.25), wp.quat_identity(dtype=wp.float32)),
        )

        builder_newton.end_world()

        model_newton: Model = builder_newton.finalize(skip_validation_joints=True, device=self.default_device)

        # Conversion must succeed (previously raised ValueError)
        model_kamino_converted: ModelKamino = ModelKamino.from_newton(model_newton)

        # Verify that X_Bj's and X_Fj's first column is aligned with the expected axis direction
        X_Bj = model_kamino_converted.joints.X_Bj.numpy()
        X_Fj = model_kamino_converted.joints.X_Fj.numpy()
        # X_Bj has shape (num_joints, 3, 3); the revolute joint is the second one (index 1)
        R_B = X_Bj[1]  # 3x3 rotation matrix
        R_F = X_Fj[1]
        ax_col_B = R_B[:, 0]  # first column = joint axis direction
        ax_col_F = R_F[:, 0]  # first column = joint axis direction
        expected_ax = np.array([1.0, 1.0, 0.0])
        expected_ax = expected_ax / np.linalg.norm(expected_ax)
        np.testing.assert_allclose(ax_col_B, expected_ax, atol=1e-6)
        np.testing.assert_allclose(ax_col_F, expected_ax, atol=1e-6)

    def test_12_model_conversions_q_i_0_com_frame(self):
        """
        Test that ``q_i_0`` stores COM world poses (not body-origin poses)
        after Newton→Kamino conversion for bodies with non-zero COM offsets.
        """
        builder_newton: ModelBuilder = ModelBuilder()
        SolverKamino.register_custom_attributes(builder_newton)
        builder_newton.default_shape_cfg.margin = 0.0
        builder_newton.default_shape_cfg.gap = 0.0

        builder_newton.begin_world()

        # Body 0: at origin, identity rotation, COM offset along x
        bid0 = builder_newton.add_link(
            label="body0",
            mass=1.0,
            inertia=wp.mat33f(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
            xform=wp.transformf(wp.vec3f(0.0, 0.0, 0.0), wp.quat_identity(dtype=wp.float32)),
            com=wp.vec3f(0.1, 0.0, 0.0),
            lock_inertia=True,
        )
        builder_newton.add_shape_box(label="box0", body=bid0, hx=0.05, hy=0.05, hz=0.05)

        # Body 1: at (0,0,1), rotated 90° about z-axis, single-axis COM offset
        rot_90z = wp.quat_from_axis_angle(wp.vec3f(0.0, 0.0, 1.0), np.pi / 2.0)
        bid1 = builder_newton.add_link(
            label="body1",
            mass=1.0,
            inertia=wp.mat33f(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
            xform=wp.transformf(wp.vec3f(0.0, 0.0, 1.0), rot_90z),
            com=wp.vec3f(0.1, 0.0, 0.0),
            lock_inertia=True,
        )
        builder_newton.add_shape_box(label="box1", body=bid1, hx=0.05, hy=0.05, hz=0.05)

        # Body 2: at (1,0,0), rotated 90° about x-axis, 3D COM offset
        rot_90x = wp.quat_from_axis_angle(wp.vec3f(1.0, 0.0, 0.0), np.pi / 2.0)
        bid2 = builder_newton.add_link(
            label="body2",
            mass=1.0,
            inertia=wp.mat33f(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
            xform=wp.transformf(wp.vec3f(1.0, 0.0, 0.0), rot_90x),
            com=wp.vec3f(0.1, 0.2, 0.3),
            lock_inertia=True,
        )
        builder_newton.add_shape_box(label="box2", body=bid2, hx=0.05, hy=0.05, hz=0.05)

        # Fix body 0 to world
        builder_newton.add_joint_fixed(
            label="world_to_body0",
            parent=-1,
            child=bid0,
            parent_xform=wp.transform_identity(dtype=wp.float32),
            child_xform=wp.transform_identity(dtype=wp.float32),
        )

        # Revolute joint: body 0 → body 1
        builder_newton.add_joint_revolute(
            label="body0_to_body1",
            parent=bid0,
            child=bid1,
            axis=wp.vec3(0.0, 1.0, 0.0),
            parent_xform=wp.transformf(wp.vec3f(0.0, 0.0, 0.5), wp.quat_identity(dtype=wp.float32)),
            child_xform=wp.transformf(wp.vec3f(0.0, 0.0, -0.5), wp.quat_identity(dtype=wp.float32)),
        )

        # Revolute joint: body 1 → body 2
        builder_newton.add_joint_revolute(
            label="body1_to_body2",
            parent=bid1,
            child=bid2,
            axis=wp.vec3(0.0, 1.0, 0.0),
            parent_xform=wp.transformf(wp.vec3f(0.5, 0.0, 0.0), wp.quat_identity(dtype=wp.float32)),
            child_xform=wp.transformf(wp.vec3f(-0.5, 0.0, 0.0), wp.quat_identity(dtype=wp.float32)),
        )

        builder_newton.end_world()

        model_newton: Model = builder_newton.finalize(skip_validation_joints=True, device=self.default_device)
        model_kamino_converted: ModelKamino = ModelKamino.from_newton(model_newton)

        q_i_0_np = model_kamino_converted.bodies.q_i_0.numpy()  # shape (N, 7)
        body_q_np = model_newton.body_q.numpy()

        # Body 0: identity rotation, origin (0,0,0), COM (0.1,0,0) → world (0.1, 0, 0)
        np.testing.assert_allclose(q_i_0_np[0, :3], [0.1, 0.0, 0.0], atol=1e-6)
        np.testing.assert_allclose(q_i_0_np[0, 3:7], body_q_np[0, 3:7], atol=1e-6)

        # Body 1: 90° z-rotation maps local (0.1,0,0) → world (0, 0.1, 0), plus origin (0,0,1)
        np.testing.assert_allclose(q_i_0_np[1, :3], [0.0, 0.1, 1.0], atol=1e-6)
        np.testing.assert_allclose(q_i_0_np[1, 3:7], body_q_np[1, 3:7], atol=1e-6)

        # Body 2: 90° x-rotation maps local (0.1, 0.2, 0.3) → world (0.1, -0.3, 0.2),
        # plus origin (1,0,0) → (1.1, -0.3, 0.2)
        np.testing.assert_allclose(q_i_0_np[2, :3], [1.1, -0.3, 0.2], atol=1e-6)
        np.testing.assert_allclose(q_i_0_np[2, 3:7], body_q_np[2, 3:7], atol=1e-6)

    def _build_single_floating_body_com_offset_model(self) -> Model:
        """Build a single floating body with a non-zero COM offset."""
        builder_newton: ModelBuilder = ModelBuilder()
        SolverKamino.register_custom_attributes(builder_newton)
        builder_newton.default_shape_cfg.margin = 0.0
        builder_newton.default_shape_cfg.gap = 0.0

        builder_newton.begin_world()

        bid = builder_newton.add_link(
            label="body0",
            mass=1.0,
            inertia=wp.mat33f(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
            xform=wp.transformf(wp.vec3f(0.0, 0.0, 1.0), wp.quat_identity(dtype=wp.float32)),
            com=wp.vec3f(0.1, 0.0, 0.0),
            lock_inertia=True,
        )
        builder_newton.add_shape_box(label="box0", body=bid, hx=0.05, hy=0.05, hz=0.05)
        builder_newton.add_joint_free(
            label="world_to_body0",
            parent=-1,
            child=bid,
            parent_xform=wp.transform_identity(dtype=wp.float32),
            child_xform=wp.transform_identity(dtype=wp.float32),
        )

        builder_newton.end_world()

        return builder_newton.finalize(skip_validation_joints=True, device=self.default_device)

    def _build_com_offset_model(self, with_base_joint: bool = True):
        """Build a 3-body chain with non-zero COM offsets for reset tests."""
        builder_newton: ModelBuilder = ModelBuilder()
        SolverKamino.register_custom_attributes(builder_newton)
        builder_newton.default_shape_cfg.margin = 0.0
        builder_newton.default_shape_cfg.gap = 0.0

        builder_newton.begin_world()

        # Body 0: at origin, identity rotation, COM offset along x
        bid0 = builder_newton.add_link(
            label="body0",
            mass=1.0,
            inertia=wp.mat33f(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
            xform=wp.transformf(wp.vec3f(0.0, 0.0, 0.0), wp.quat_identity(dtype=wp.float32)),
            com=wp.vec3f(0.1, 0.0, 0.0),
            lock_inertia=True,
        )
        builder_newton.add_shape_box(label="box0", body=bid0, hx=0.05, hy=0.05, hz=0.05)

        # Body 1: at (0,0,1), rotated 90° about z-axis, single-axis COM offset
        rot_90z = wp.quat_from_axis_angle(wp.vec3f(0.0, 0.0, 1.0), np.pi / 2.0)
        bid1 = builder_newton.add_link(
            label="body1",
            mass=1.0,
            inertia=wp.mat33f(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
            xform=wp.transformf(wp.vec3f(0.0, 0.0, 1.0), rot_90z),
            com=wp.vec3f(0.1, 0.0, 0.0),
            lock_inertia=True,
        )
        builder_newton.add_shape_box(label="box1", body=bid1, hx=0.05, hy=0.05, hz=0.05)

        # Body 2: at (1,0,0), rotated 90° about x-axis, 3D COM offset
        rot_90x = wp.quat_from_axis_angle(wp.vec3f(1.0, 0.0, 0.0), np.pi / 2.0)
        bid2 = builder_newton.add_link(
            label="body2",
            mass=1.0,
            inertia=wp.mat33f(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
            xform=wp.transformf(wp.vec3f(1.0, 0.0, 0.0), rot_90x),
            com=wp.vec3f(0.1, 0.2, 0.3),
            lock_inertia=True,
        )
        builder_newton.add_shape_box(label="box2", body=bid2, hx=0.05, hy=0.05, hz=0.05)

        # Add free joint to body 0
        if with_base_joint:
            builder_newton.add_joint_free(
                label="world_to_body0",
                parent=-1,
                child=bid0,
                parent_xform=wp.transform_identity(dtype=wp.float32),
                child_xform=wp.transform_identity(dtype=wp.float32),
            )

        # Revolute joint: body 0 -> body 1
        builder_newton.add_joint_revolute(
            label="body0_to_body1",
            parent=bid0,
            child=bid1,
            axis=wp.vec3(0.0, 1.0, 0.0),
            parent_xform=wp.transformf(wp.vec3f(0.0, 0.0, 0.5), wp.quat_identity(dtype=wp.float32)),
            child_xform=wp.transformf(wp.vec3f(0.0, 0.0, -0.5), wp.quat_identity(dtype=wp.float32)),
        )

        # Revolute joint: body 1 -> body 2
        builder_newton.add_joint_revolute(
            label="body1_to_body2",
            parent=bid1,
            child=bid2,
            axis=wp.vec3(0.0, 1.0, 0.0),
            parent_xform=wp.transformf(wp.vec3f(0.5, 0.0, 0.0), wp.quat_identity(dtype=wp.float32)),
            child_xform=wp.transformf(wp.vec3f(-0.5, 0.0, 0.0), wp.quat_identity(dtype=wp.float32)),
        )

        builder_newton.end_world()

        return builder_newton.finalize(skip_validation_joints=True, device=self.default_device)

    def test_13_reset_produces_body_origin_frame(self):
        """
        Test that ``SolverKamino.reset()`` writes body-origin frame poses
        into ``state.body_q``, not COM-frame poses, for bodies with non-zero
        COM offsets.
        """
        for with_base_joint in [False, True]:  # Test both the base joint and base body case
            model = self._build_com_offset_model(with_base_joint=with_base_joint)
            body_q_expected = model.body_q.numpy().copy()

            solver = SolverKamino(model)

            # Default reset (no args) should restore body-origin poses
            state_out: State = model.state()
            solver.reset(state=state_out)
            body_q_after = state_out.body_q.numpy()

            for i in range(model.body_count):
                np.testing.assert_allclose(
                    body_q_after[i],
                    body_q_expected[i],
                    atol=1e-6,
                    err_msg=f"Default reset: body {i} pose is not in body-origin frame",
                )

            # Velocities should be zero after default reset
            body_qd_after = state_out.body_qd.numpy()
            np.testing.assert_allclose(
                body_qd_after,
                0.0,
                atol=1e-6,
                err_msg="Default reset: body velocities should be zero",
            )

    def test_14_base_reset_produces_body_origin_frame(self):
        """
        Test that ``SolverKamino.reset(base_q=..., base_u=...)`` writes
        body-origin frame poses and velocities into ``state.body_q`` and
        ``state.body_qd`` for bodies with non-zero COM offsets.
        """
        for with_base_joint in [False, True]:  # Test both the base joint and base body case
            model = self._build_com_offset_model(with_base_joint=with_base_joint)
            body_q_expected = model.body_q.numpy().copy()

            solver = SolverKamino(model)

            # --- Base reset with identity base pose should restore body-origin poses ---
            state_out: State = model.state()
            base_q = wp.array(
                [wp.transformf(wp.vec3f(0.0, 0.0, 0.0), wp.quat_identity(dtype=wp.float32))],
                dtype=wp.transformf,
            )
            base_u = wp.zeros(1, dtype=wp.spatial_vectorf)
            reset_config = SolverKamino.ResetConfig(
                base_pose=SolverKamino.ResetConfig.FromBaseQ(base_q),
                base_velocity=SolverKamino.ResetConfig.FromBaseU(base_u),
            )
            solver.reset(state=state_out, config=reset_config)
            body_q_after = state_out.body_q.numpy()

            for i in range(model.body_count):
                np.testing.assert_allclose(
                    body_q_after[i],
                    body_q_expected[i],
                    atol=1e-6,
                    err_msg=f"Base reset (identity): body {i} pose is not in body-origin frame",
                )

            # Velocities should be zero with zero base twist
            body_qd_after = state_out.body_qd.numpy()
            np.testing.assert_allclose(
                body_qd_after,
                0.0,
                atol=1e-6,
                err_msg="Base reset (identity): body velocities should be zero",
            )

            # --- Base reset with a translated base pose ---
            offset = np.array([2.0, 3.0, 5.0])
            base_q_shifted = wp.array(
                [wp.transformf(wp.vec3f(*offset), wp.quat_identity(dtype=wp.float32))],
                dtype=wp.transformf,
            )
            reset_config = SolverKamino.ResetConfig(
                base_pose=SolverKamino.ResetConfig.FromBaseQ(base_q_shifted),
                base_velocity=SolverKamino.ResetConfig.FromBaseU(base_u),
            )
            solver.reset(state=state_out, config=reset_config)
            body_q_shifted = state_out.body_q.numpy()

            for i in range(model.body_count):
                np.testing.assert_allclose(
                    body_q_shifted[i, :3],
                    body_q_expected[i, :3] + offset,
                    atol=1e-6,
                    err_msg=f"Base reset (translated): body {i} position mismatch",
                )
                np.testing.assert_allclose(
                    body_q_shifted[i, 3:7],
                    body_q_expected[i, 3:7],
                    atol=1e-6,
                    err_msg=f"Base reset (translated): body {i} rotation mismatch",
                )

    def test_15_preserve_reset_keeps_joint_q_consistent_with_com_offset_body(self):
        """
        Verify preserve reset leaves body_q and joint_q unchanged for COM-offset bodies.

        For a single floating body with a non-zero center-of-mass offset, a preserve
        reset should not modify ``body_q`` or re-derived ``joint_q``.
        """
        model = self._build_single_floating_body_com_offset_model()
        solver = SolverKamino(model)

        state: State = model.state()
        solver.reset(state=state)

        body_q_before = state.body_q.numpy().copy()
        joint_q_before = state.joint_q.numpy().copy()

        solver.reset(state=state, config=SolverKamino.ResetConfig.preserve())

        np.testing.assert_allclose(
            state.body_q.numpy(),
            body_q_before,
            atol=1e-6,
            err_msg="Preserve reset should not modify body_q",
        )
        np.testing.assert_allclose(
            state.joint_q.numpy(),
            joint_q_before,
            atol=1e-6,
            err_msg="Preserve reset should not modify joint_q when body_q is unchanged",
        )

    def test_16_model_conversions_shape_offset_com_relative(self):
        """
        Test that ``geoms.offset`` stores COM-relative shape positions
        after Newton→Kamino conversion, while ground shapes are unchanged.
        """
        builder_newton: ModelBuilder = ModelBuilder()
        SolverKamino.register_custom_attributes(builder_newton)
        builder_newton.default_shape_cfg.margin = 0.0
        builder_newton.default_shape_cfg.gap = 0.0

        builder_newton.begin_world()

        # Body with COM=(0.1, 0.2, 0.0), shape at (0.5, 0.0, 0.0)
        bid = builder_newton.add_link(
            label="body0",
            mass=1.0,
            inertia=wp.mat33f(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
            xform=wp.transformf(wp.vec3f(0.0, 0.0, 0.0), wp.quat_identity(dtype=wp.float32)),
            com=wp.vec3f(0.1, 0.2, 0.0),
            lock_inertia=True,
        )
        builder_newton.add_shape_box(
            label="box0",
            body=bid,
            hx=0.05,
            hy=0.05,
            hz=0.05,
            xform=wp.transformf(wp.vec3f(0.5, 0.0, 0.0), wp.quat_identity(dtype=wp.float32)),
        )
        # Ground shape (bid=-1) — should be left unchanged
        builder_newton.add_shape_box(
            label="ground_box",
            body=-1,
            hx=1.0,
            hy=1.0,
            hz=0.01,
            xform=wp.transformf(wp.vec3f(1.0, 0.0, 0.0), wp.quat_identity(dtype=wp.float32)),
        )

        builder_newton.add_joint_fixed(
            label="fix",
            parent=-1,
            child=bid,
            parent_xform=wp.transform_identity(dtype=wp.float32),
            child_xform=wp.transform_identity(dtype=wp.float32),
        )
        builder_newton.end_world()

        model_newton: Model = builder_newton.finalize(skip_validation_joints=True, device=self.default_device)
        model_kamino_converted: ModelKamino = ModelKamino.from_newton(model_newton)
        offset_np = model_kamino_converted.geoms.offset.numpy()

        # Shape on body: pos should be (0.5-0.1, 0.0-0.2, 0.0) = (0.4, -0.2, 0.0)
        np.testing.assert_allclose(offset_np[0, :3], [0.4, -0.2, 0.0], atol=1e-6)
        # Ground shape: pos unchanged at (1.0, 0.0, 0.0)
        np.testing.assert_allclose(offset_np[1, :3], [1.0, 0.0, 0.0], atol=1e-6)

    def test_20_origin_com_roundtrip(self):
        """
        Test that origin→COM→origin is the identity on body_q.
        """
        model = self._build_com_offset_model()
        body_q = wp.clone(model.body_q)
        q_orig = body_q.numpy().copy()

        convert_body_origin_to_com(model.body_com, body_q, body_q)
        convert_body_com_to_origin(model.body_com, body_q, body_q)

        np.testing.assert_allclose(body_q.numpy(), q_orig, atol=1e-6, err_msg="body_q roundtrip failed")

    def test_30_model_conversions_per_dof_constraint_rows_and_actuation_paths(self):
        """Map dynamic, effort, and friction properties to their configured DoFs."""
        builder = ModelBuilder()
        SolverKamino.register_custom_attributes(builder)
        builder.begin_world()
        # Give the body real inertia so it is treated as dynamic by Kamino;
        # otherwise the world<->massless-body joint is culled at conversion.
        body = builder.add_link(mass=1.0, inertia=wp.mat33f(np.eye(3, dtype=np.float32)))
        joint = builder.add_joint_d6(
            -1,
            body,
            angular_axes=[
                ModelBuilder.JointDofConfig(axis=newton.Axis.X, armature=1.0),
                ModelBuilder.JointDofConfig(
                    axis=newton.Axis.Y,
                    effort_limit=1.0,
                    target_ke=100.0,
                    actuator_mode=newton.JointTargetMode.POSITION,
                ),
                ModelBuilder.JointDofConfig(axis=newton.Axis.Z, friction=0.5),
            ],
        )
        builder.add_articulation([joint])
        builder.end_world()

        kamino = ModelKamino.from_newton(builder.finalize())

        np.testing.assert_array_equal(kamino.joints.dynamic_cts_axis.numpy(), [0])
        np.testing.assert_array_equal(kamino.joints.effort_cts_axis.numpy(), [1])
        np.testing.assert_array_equal(kamino.joints.friction_cts_axis.numpy(), [2])
        np.testing.assert_array_equal(
            kamino.joints.dof_act_types.numpy(),
            [
                JointActuationType.PASSIVE,
                JointActuationType.POSITION,
                JointActuationType.PASSIVE,
            ],
        )
        np.testing.assert_array_equal(
            kamino.joints.dof_act_paths.numpy(),
            [
                DofActuationPath.DYNAMIC_CTS,
                DofActuationPath.EFFORT_CTS,
                DofActuationPath.BODY_WRENCHES,
            ],
        )
        self.assertEqual(kamino.size.sum_of_num_dynamic_joint_cts, 1)
        self.assertEqual(kamino.size.sum_of_num_effort_joint_cts, 1)
        self.assertEqual(kamino.size.sum_of_num_friction_joint_cts, 1)

    def test_31_model_conversions_effort_constraint_allocation(self):
        """Allocate bounded implicit-PD rows and share dynamic rows as required."""
        cases = (
            ("bounded_implicit_pd", True, 1.0, False, 1, 0),
            ("unbounded_implicit_pd", True, JOINT_TAUMAX, False, 0, 1),
            ("finite_without_implicit_pd", False, 1.0, False, 0, 0),
            ("bounded_implicit_pd_with_armature", True, 1.0, True, 1, 1),
        )
        for name, implicit_pd, effort_limit, dynamic_joints, expected_effort, expected_dynamic in cases:
            with self.subTest(name=name):
                builder = ModelBuilder()
                SolverKamino.register_custom_attributes(builder)
                build_box_pendulum(
                    builder=builder,
                    ground=False,
                    implicit_pd=implicit_pd,
                    dynamic_joints=dynamic_joints,
                )
                model = builder.finalize()
                model.joint_effort_limit.assign([effort_limit])

                kamino = ModelKamino.from_newton(model)

                self.assertEqual(kamino.size.sum_of_num_effort_joint_cts, expected_effort)
                self.assertEqual(kamino.size.sum_of_num_dynamic_joint_cts, expected_dynamic)
                np.testing.assert_array_equal(kamino.joints.effort_cts_axis.numpy(), [0] * expected_effort)
                np.testing.assert_array_equal(kamino.joints.dynamic_cts_axis.numpy(), [0] * expected_dynamic)

    def test_32_model_conversions_multiworld_effort_offsets_follow_global_row_order(self):
        """Prefix bounded rows globally while retaining per-world offsets."""
        for friction in (0.0, 1.0):
            with self.subTest(friction=friction):
                builder = ModelBuilder()
                SolverKamino.register_custom_attributes(builder)
                for _ in range(2):
                    build_box_pendulum(builder=builder, ground=False, implicit_pd=True)
                model = builder.finalize()
                model.joint_effort_limit.fill_(1.0)
                if friction > 0.0:
                    model.joint_friction.fill_(friction)

                kamino = ModelKamino.from_newton(model)

                num_friction = int(friction > 0.0)
                num_effort = 1
                num_bounded = num_friction + num_effort

                np.testing.assert_array_equal(kamino.info.num_joint_effort_cts.numpy(), [num_effort, num_effort])
                np.testing.assert_array_equal(kamino.info.joint_effort_cts_offset.numpy(), [0, num_effort])
                np.testing.assert_array_equal(kamino.info.num_joint_friction_cts.numpy(), [num_friction, num_friction])
                np.testing.assert_array_equal(kamino.info.joint_friction_cts_offset.numpy(), [0, num_friction])
                np.testing.assert_array_equal(
                    kamino.info.num_joint_bounded_cts.numpy(),
                    [num_bounded, num_bounded],
                )
                np.testing.assert_array_equal(
                    kamino.info.joint_bounded_cts_offset.numpy(),
                    [0, num_bounded],
                )

                self.assertEqual(kamino.size.sum_of_num_effort_joint_cts, 2 * num_effort)
                self.assertEqual(kamino.size.sum_of_num_friction_joint_cts, 2 * num_friction)
                self.assertEqual(kamino.size.sum_of_num_bounded_joint_cts, 2 * num_bounded)

                np.testing.assert_array_equal(
                    kamino.joints.effort_cts_offset.numpy(),
                    [0, num_effort, 2 * num_effort],
                )
                np.testing.assert_array_equal(
                    kamino.joints.friction_cts_offset.numpy(),
                    [0, num_friction, 2 * num_friction],
                )
                np.testing.assert_array_equal(
                    kamino.joints.bounded_cts_offset.numpy(),
                    [0, num_bounded, 2 * num_bounded],
                )
                np.testing.assert_array_equal(kamino.joints.effort_cts_axis.numpy(), [0, 0])
                if num_friction:
                    np.testing.assert_array_equal(kamino.joints.friction_cts_axis.numpy(), [0, 0])
                else:
                    np.testing.assert_array_equal(kamino.joints.friction_cts_axis.numpy(), [])

    def test_40_model_conversions_materials_dedup_from_shared_and_distinct_shape_properties(self):
        """
        Test that Newton->Kamino conversion deduplicates materials by
        (static_friction, restitution): shapes sharing identical values collapse
        onto the same Kamino material, while shapes with distinct values register
        separate materials.
        """
        builder_newton: ModelBuilder = ModelBuilder()
        SolverKamino.register_custom_attributes(builder_newton)

        builder_newton.begin_world()
        bid = builder_newton.add_link(
            label="body0",
            mass=1.0,
            inertia=wp.mat33f(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
            lock_inertia=True,
        )
        # Two shapes sharing the same material properties
        builder_newton.add_shape_box(
            label="shapeA0",
            body=bid,
            hx=0.05,
            hy=0.05,
            hz=0.05,
            cfg=newton.ModelBuilder.ShapeConfig(mu=0.3, restitution=0.1),
        )
        builder_newton.add_shape_box(
            label="shapeA1",
            body=bid,
            hx=0.05,
            hy=0.05,
            hz=0.05,
            xform=wp.transformf(wp.vec3f(0.2, 0.0, 0.0), wp.quat_identity(dtype=wp.float32)),
            cfg=newton.ModelBuilder.ShapeConfig(mu=0.3, restitution=0.1),
        )
        # A third shape with distinct material properties
        builder_newton.add_shape_box(
            label="shapeB",
            body=bid,
            hx=0.05,
            hy=0.05,
            hz=0.05,
            xform=wp.transformf(wp.vec3f(0.4, 0.0, 0.0), wp.quat_identity(dtype=wp.float32)),
            cfg=newton.ModelBuilder.ShapeConfig(mu=0.6, restitution=0.25),
        )
        builder_newton.add_joint_fixed(parent=-1, child=bid)
        builder_newton.end_world()

        model_newton: Model = builder_newton.finalize(skip_validation_joints=True, device=self.default_device)
        model_kamino: ModelKamino = ModelKamino.from_newton(model_newton)

        # Default material (index 0) plus 2 distinct registered materials
        self.assertEqual(model_kamino.size.sum_of_num_materials, 3)

        geom_material = model_kamino.geoms.material.numpy()
        # The two shapes sharing properties must resolve to the same material id
        self.assertEqual(geom_material[0], geom_material[1])
        # The shape with distinct properties must resolve to a different material id
        self.assertNotEqual(geom_material[0], geom_material[2])
        # Neither matches Kamino's own default material properties
        self.assertNotEqual(geom_material[0], 0)
        self.assertNotEqual(geom_material[2], 0)

        static_friction = model_kamino.materials.static_friction.numpy()
        restitution = model_kamino.materials.restitution.numpy()
        np.testing.assert_allclose(static_friction[geom_material[0]], 0.3, atol=1e-6)
        np.testing.assert_allclose(restitution[geom_material[0]], 0.1, atol=1e-6)
        np.testing.assert_allclose(static_friction[geom_material[2]], 0.6, atol=1e-6)
        np.testing.assert_allclose(restitution[geom_material[2]], 0.25, atol=1e-6)

    def test_41_model_conversions_noncollidable_shape_has_no_material(self):
        """
        Test that a shape without shape collision enabled converts to a Kamino
        geom with no material assigned (material index -1), even when it shares
        material properties with a collidable shape.
        """
        builder_newton: ModelBuilder = ModelBuilder()
        SolverKamino.register_custom_attributes(builder_newton)

        builder_newton.begin_world()
        bid = builder_newton.add_link(
            label="body0",
            mass=1.0,
            inertia=wp.mat33f(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
            lock_inertia=True,
        )
        builder_newton.add_shape_box(
            label="collidable",
            body=bid,
            hx=0.05,
            hy=0.05,
            hz=0.05,
            cfg=newton.ModelBuilder.ShapeConfig(mu=0.4, restitution=0.2, has_shape_collision=True),
        )
        builder_newton.add_shape_box(
            label="visual_only",
            body=bid,
            hx=0.05,
            hy=0.05,
            hz=0.05,
            xform=wp.transformf(wp.vec3f(0.2, 0.0, 0.0), wp.quat_identity(dtype=wp.float32)),
            cfg=newton.ModelBuilder.ShapeConfig(mu=0.4, restitution=0.2, has_shape_collision=False),
        )
        builder_newton.add_joint_fixed(parent=-1, child=bid)
        builder_newton.end_world()

        model_newton: Model = builder_newton.finalize(skip_validation_joints=True, device=self.default_device)
        model_kamino: ModelKamino = ModelKamino.from_newton(model_newton)

        geom_material = model_kamino.geoms.material.numpy()
        self.assertNotEqual(geom_material[0], -1)
        self.assertEqual(geom_material[1], -1)


class TestStateControlConversions(unittest.TestCase):
    def setUp(self):
        if not test_context.setup_done:
            setup_tests(clear_cache=False)
        self.default_device = wp.get_device(test_context.device)
        self.verbose = test_context.verbose  # Set to True to enable verbose output

        # Set debug-level logging to print verbose test output to console
        if self.verbose:
            print("\n")  # Add newline before test output for better readability
            msg.set_log_level(msg.LogLevel.INFO)  # TODO @nvtw: set this to DEBUG when investigating noted issues
        else:
            msg.reset_log_level()

    def tearDown(self):
        self.default_device = None
        if self.verbose:
            msg.reset_log_level()

    def test_01_state_conversions(self):
        """
        Test the conversion operations between newton.State and kamino.StateKamino.
        """
        # Create a fourbar
        builder_single = ModelBuilder()
        builder_single.default_shape_cfg.margin = 0.0
        builder_single.default_shape_cfg.gap = 0.0
        build_boxes_fourbar(
            builder=builder_single,
            z_offset=0.0,
            fixedbase=False,
            floatingbase=True,
            limits=True,
            ground=True,
            new_world=True,
            actuator_ids=[2, 4],
        )
        builder_newton: ModelBuilder = ModelBuilder()
        SolverKamino.register_custom_attributes(builder_newton)
        builder_newton.replicate(builder=builder_single, world_count=2)

        # Create models from the builders and conversion operations, and check for consistency
        model_newton: Model = builder_newton.finalize(skip_validation_joints=True, device=self.default_device)
        model_kamino: ModelKamino = ModelKamino.from_newton(model_newton)

        # Create a Newton state container
        state_newton: State = model_newton.state()
        self.assertIsInstance(state_newton.body_q, wp.array)
        self.assertEqual(state_newton.body_q.size, model_newton.body_count)
        self.assertIsNotNone(state_newton.joint_q_prev)
        self.assertEqual(state_newton.joint_q_prev.size, model_newton.joint_coord_count)
        self.assertIsNotNone(state_newton.joint_lambdas)
        self.assertEqual(state_newton.joint_lambdas.size, model_newton.joint_constraint_count)
        self.assertEqual(model_newton.joint_constraint_count, model_kamino.size.sum_of_num_kinematic_joint_cts)

        # Create a Kamino state container
        state_kamino: StateKamino = model_kamino.state()
        self.assertIsInstance(state_kamino.q_i, wp.array)
        self.assertEqual(state_kamino.q_i.size, model_kamino.size.sum_of_num_bodies)

        state_kamino: StateKamino = StateKamino.from_newton(model_kamino.size, model_newton, state_newton, True, False)
        self.assertIsInstance(state_kamino.q_i, wp.array)
        self.assertEqual(state_kamino.q_i.size, model_kamino.size.sum_of_num_bodies)
        # NOTE: Check ptr due to conversion from wp.spatial_vectorf
        self.assertIs(state_kamino.u_i.ptr, state_newton.body_qd.ptr)
        self.assertIs(state_kamino.w_i_e.ptr, state_newton.body_f.ptr)
        self.assertIs(state_kamino.w_i.ptr, state_newton.body_f_total.ptr)
        # NOTE: Check that arrays are the same because these should be pure references
        self.assertIs(state_kamino.q_i, state_newton.body_q)
        self.assertIs(state_kamino.q_j, state_newton.joint_q)
        self.assertIs(state_kamino.dq_j, state_newton.joint_qd)
        self.assertIs(state_kamino.q_j_p, state_newton.joint_q_prev)
        self.assertIs(state_kamino.lambda_kin_j, state_newton.joint_lambdas)
        self.assertIsNot(state_kamino.lambda_dyn_j, state_newton.joint_lambdas)

        state_newton_converted: State = StateKamino.to_newton(model_newton, state_kamino)
        self.assertIsInstance(state_newton_converted.body_q, wp.array)
        self.assertEqual(state_newton_converted.body_q.size, model_newton.body_count)
        # NOTE: Check ptr due to conversion from wp.spatial_vectorf
        self.assertIs(state_newton_converted.body_qd.ptr, state_kamino.u_i.ptr)
        self.assertIs(state_newton_converted.body_f.ptr, state_kamino.w_i_e.ptr)
        self.assertIs(state_newton_converted.body_f_total.ptr, state_kamino.w_i.ptr)
        # NOTE: Check that arrays are the same because these should be pure references
        self.assertIs(state_newton_converted.body_q, state_kamino.q_i)
        self.assertIs(state_newton_converted.joint_q, state_kamino.q_j)
        self.assertIs(state_newton_converted.joint_qd, state_kamino.dq_j)
        self.assertIs(state_newton_converted.joint_q_prev, state_kamino.q_j_p)
        self.assertIs(state_newton_converted.joint_lambdas, state_kamino.lambda_kin_j)
        self.assertIs(state_newton_converted.joint_lambdas_dyn, state_kamino.lambda_dyn_j)
        self.assertIs(state_newton_converted.joint_lambdas_f, state_kamino.lambda_f_j)
        self.assertIs(state_newton_converted.joint_lambdas_tau, state_kamino.lambda_tau_j)

    def test_10_control_conversions(self):
        """
        Test the conversions between newton.Control and kamino.ControlKamino.
        """
        # Create a fourbar
        builder_single = ModelBuilder()
        builder_single.default_shape_cfg.margin = 0.0
        builder_single.default_shape_cfg.gap = 0.0
        build_boxes_fourbar(
            builder=builder_single,
            z_offset=0.0,
            fixedbase=False,
            floatingbase=True,
            # dynamic_joints=True,
            # implicit_pd=True,
            limits=True,
            ground=True,
            new_world=True,
            actuator_ids=[1, 2, 3, 4],
        )
        builder_newton: ModelBuilder = ModelBuilder()
        SolverKamino.register_custom_attributes(builder_newton)
        builder_newton.replicate(builder=builder_single, world_count=2)

        # Create models from the builders and conversion operations, and check for consistency
        model_newton: Model = builder_newton.finalize(skip_validation_joints=True, device=self.default_device)
        model_kamino: ModelKamino = ModelKamino.from_newton(model_newton)

        # Create a Newton control container
        control_newton: Control = model_newton.control()
        if newton.use_coord_layout_targets:
            control_newton.joint_target_q = wp.clone(model_newton.joint_q, device=self.default_device)
        else:
            joint_target_q_dof_space = wp.zeros_like(control_newton.joint_target_q, device=self.default_device)
            convert_target_coords_to_target_dofs(model_newton.joint_q, joint_target_q_dof_space, model_kamino)
            control_newton.joint_target_q = joint_target_q_dof_space
        # TODO: remove above lines if joint_target_q in newton gets updated to take into account
        # initial pose like joint_q does (cf issue #3380 in newton)
        self.assertIsInstance(control_newton.joint_f, wp.array)
        self.assertEqual(control_newton.joint_f.size, model_newton.joint_dof_count)

        # Create a Kamino control container
        control_kamino: ControlKamino = model_kamino.control()
        self.assertIsInstance(control_kamino.tau_j, wp.array)
        self.assertEqual(control_kamino.tau_j.size, model_kamino.size.sum_of_num_joint_dofs)

        control_kamino_converted = ControlKamino()
        control_kamino_converted.finalize(model_kamino)
        control_kamino_converted.from_newton(control_newton, model_kamino)
        self.assertIsInstance(control_kamino_converted.tau_j, wp.array)
        self.assertIs(control_kamino_converted.tau_j, control_newton.joint_f)
        self.assertEqual(control_kamino_converted.tau_j.size, model_newton.joint_dof_count)
        self.assertIsNone(control_kamino_converted.tau_j_ref)
        test_util_checks.assert_control_equal(self, control_kamino_converted, control_kamino, excluded=["tau_j_ref"])

        # Convert back to a Newton control container.
        control_newton.joint_act.fill_(42.0)
        control_newton_converted: Control = model_newton.control()
        joint_act_before = control_newton_converted.joint_act.numpy().copy()
        control_kamino_converted.to_newton(control_newton_converted, model_kamino)
        self.assertIsInstance(control_newton_converted.joint_f, wp.array)
        self.assertIs(control_newton_converted.joint_f, control_kamino_converted.tau_j)
        self.assertEqual(control_newton_converted.joint_f.size, model_newton.joint_dof_count)
        np.testing.assert_array_equal(control_newton_converted.joint_act.numpy(), joint_act_before)
        test_util_checks.assert_array_attributes_equal(
            self,
            control_newton_converted,
            control_newton,
            attributes=["joint_f", "joint_target_q", "joint_target_qd"],
        )


class TestConvertModelMaterials(unittest.TestCase):
    def setUp(self):
        if not test_context.setup_done:
            setup_tests(clear_cache=False)
        self.default_device = wp.get_device(test_context.device)

    def tearDown(self):
        self.default_device = None

    def _build_two_material_model(self) -> tuple[Model, ModelKamino]:
        """Build a model with 2 shapes sharing a material and 1 shape with a distinct one."""
        builder = ModelBuilder()
        SolverKamino.register_custom_attributes(builder)

        builder.begin_world()
        bid = builder.add_link(
            label="body0",
            mass=1.0,
            inertia=wp.mat33f(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
            lock_inertia=True,
        )
        builder.add_shape_box(
            label="shapeA0",
            body=bid,
            hx=0.05,
            hy=0.05,
            hz=0.05,
            cfg=newton.ModelBuilder.ShapeConfig(mu=0.3, restitution=0.1),
        )
        builder.add_shape_box(
            label="shapeA1",
            body=bid,
            hx=0.05,
            hy=0.05,
            hz=0.05,
            xform=wp.transformf(wp.vec3f(0.2, 0.0, 0.0), wp.quat_identity(dtype=wp.float32)),
            cfg=newton.ModelBuilder.ShapeConfig(mu=0.3, restitution=0.1),
        )
        builder.add_shape_box(
            label="shapeB",
            body=bid,
            hx=0.05,
            hy=0.05,
            hz=0.05,
            xform=wp.transformf(wp.vec3f(0.4, 0.0, 0.0), wp.quat_identity(dtype=wp.float32)),
            cfg=newton.ModelBuilder.ShapeConfig(mu=0.6, restitution=0.25),
        )
        builder.add_joint_fixed(parent=-1, child=bid)
        builder.end_world()

        model = builder.finalize(skip_validation_joints=True, device=self.default_device)
        model_kamino = ModelKamino.from_newton(model)
        return model, model_kamino

    def test_00_updates_material_properties_in_place_from_first_shape(self):
        """Non-conflicting per-shape updates propagate to the material and pair tables."""
        model, model_kamino = self._build_two_material_model()
        geom_material = model_kamino.geoms.material.numpy()
        mat_a, mat_b = int(geom_material[0]), int(geom_material[2])
        self.assertEqual(int(geom_material[1]), mat_a)
        self.assertNotEqual(mat_a, mat_b)

        first_shape = compute_material_first_shape(model_kamino.geoms.material, model_kamino.materials.num_materials)
        conflict = wp.empty(1, dtype=wp.int32, device=self.default_device)

        # Consistently update both shapes sharing material A, and the shape with material B.
        mu_np = model.shape_material_mu.numpy()
        restitution_np = model.shape_material_restitution.numpy()
        mu_np[0] = mu_np[1] = 0.55
        restitution_np[0] = restitution_np[1] = 0.35
        mu_np[2] = 0.8
        restitution_np[2] = 0.9
        model.shape_material_mu.assign(mu_np)
        model.shape_material_restitution.assign(restitution_np)

        convert_model_materials(model, model_kamino, first_shape, conflict)

        static_friction = model_kamino.materials.static_friction.numpy()
        dynamic_friction = model_kamino.materials.dynamic_friction.numpy()
        restitution = model_kamino.materials.restitution.numpy()
        np.testing.assert_allclose(static_friction[mat_a], 0.55, atol=1e-6)
        np.testing.assert_allclose(dynamic_friction[mat_a], 0.55, atol=1e-6)
        np.testing.assert_allclose(restitution[mat_a], 0.35, atol=1e-6)
        np.testing.assert_allclose(static_friction[mat_b], 0.8, atol=1e-6)
        np.testing.assert_allclose(restitution[mat_b], 0.9, atol=1e-6)

    def test_01_conflicting_shape_update_raises(self):
        """Splitting a material across two shapes that no longer agree must raise."""
        model, model_kamino = self._build_two_material_model()

        first_shape = compute_material_first_shape(model_kamino.geoms.material, model_kamino.materials.num_materials)
        conflict = wp.empty(1, dtype=wp.int32, device=self.default_device)

        # Shapes 0 and 1 share a material; give them conflicting friction values.
        mu_np = model.shape_material_mu.numpy()
        mu_np[0] = 0.55
        mu_np[1] = 0.65
        model.shape_material_mu.assign(mu_np)

        with self.assertRaises(RuntimeError):
            convert_model_materials(model, model_kamino, first_shape, conflict)


###
# Test execution
###

if __name__ == "__main__":
    # Test setup
    setup_tests()

    # Run all tests
    unittest.main(verbosity=2)
