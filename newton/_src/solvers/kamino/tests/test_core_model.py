# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""
Unit tests for the :class:`ModelKamino` class and related functionality.
"""

import copy
import os
import unittest

import numpy as np
import warp as wp

import newton
import newton._src.solvers.kamino.tests.utils.checks as test_util_checks
from newton._src.sim import Control, Model, ModelBuilder, State
from newton._src.solvers.kamino._src.core.bodies import convert_body_com_to_origin, convert_body_origin_to_com
from newton._src.solvers.kamino._src.core.builder import ModelBuilderKamino
from newton._src.solvers.kamino._src.core.control import ControlKamino
from newton._src.solvers.kamino._src.core.model import MaterialDescriptor, ModelKamino
from newton._src.solvers.kamino._src.core.state import StateKamino
from newton._src.solvers.kamino._src.models import basics as basics_kamino
from newton._src.solvers.kamino._src.models import basics_newton, get_basics_usd_assets_path
from newton._src.solvers.kamino._src.models.builders import utils as model_utils
from newton._src.solvers.kamino._src.utils import logger as msg
from newton._src.solvers.kamino._src.utils.io.usd import USDImporter
from newton._src.solvers.kamino.solver_kamino import SolverKamino
from newton._src.solvers.kamino.tests import setup_tests, test_context
from newton._src.solvers.kamino.tests.utils import print as print_utils

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
        builder = basics_kamino.build_boxes_hinged()

        # Finalize the model
        model: ModelKamino = builder.finalize(self.default_device)
        if self.verbose:
            print("")  # Add a newline for better readability
            print_utils.print_model_info(model)

        # Create a model state
        state = model.data()
        if self.verbose:
            print("")  # Add a newline for better readability
            print_utils.print_data_info(state)

        # Check the model info entries
        self.assertEqual(model.size.sum_of_num_bodies, builder.num_bodies)
        self.assertEqual(model.size.sum_of_num_joints, builder.num_joints)
        self.assertEqual(model.size.sum_of_num_geoms, builder.num_geoms)
        self.assertEqual(model.device, self.default_device)

    def test_02_double_model(self):
        # Create a model builder
        builder1 = basics_kamino.build_boxes_hinged()
        builder2 = basics_kamino.build_boxes_nunchaku()

        # Compute the total number of elements from the two builders
        total_nb = builder1.num_bodies + builder2.num_bodies
        total_nj = builder1.num_joints + builder2.num_joints
        total_ng = builder1.num_geoms + builder2.num_geoms

        # Add the second builder to the first one
        builder1.add_builder(builder2)

        # Finalize the model
        model: ModelKamino = builder1.finalize(self.default_device)
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
        builder = model_utils.make_homogeneous_builder(num_worlds=num_worlds, build_fn=basics_kamino.build_boxes_hinged)

        # Finalize the model
        model: ModelKamino = builder.finalize(self.default_device)
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
        self.assertEqual(model.size.sum_of_num_joints, num_worlds * 1)
        self.assertEqual(model.size.sum_of_num_geoms, num_worlds * 3)
        self.assertEqual(model.device, self.default_device)

    def test_04_hetereogeneous_model(self):
        # Create a model builder
        builder = basics_kamino.make_basics_heterogeneous_builder()
        num_worlds = builder.num_worlds

        # Finalize the model
        model: ModelKamino = builder.finalize(self.default_device)
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

    def test_00_model_conversions_fourbar_from_builder(self):
        """
        Test the conversion operations between newton.Model and kamino.ModelKamino
        on a simple fourbar model created explicitly using the builder.
        """
        # Create a fourbar using Newton's ModelBuilder and
        # register Kamino-specific custom attributes
        builder_0: ModelBuilder = ModelBuilder()
        SolverKamino.register_custom_attributes(builder_0)
        builder_0.default_shape_cfg.margin = 0.0
        builder_0.default_shape_cfg.gap = 0.0

        # Create a fourbar using Newton's ModelBuilder
        builder_0: ModelBuilder = basics_newton.build_boxes_fourbar(
            builder=builder_0,
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

        # Overwriting mu = 0.7 to match Kamino's default material properties
        builder_0.shape_material_mu = [0.7] * len(builder_0.shape_material_mu)

        # Duplicate the world to test multi-world handling
        builder_0.begin_world()
        builder_0.add_builder(copy.deepcopy(builder_0))
        builder_0.end_world()

        # Create a fourbar using Kamino's ModelBuilderKamino
        builder_1: ModelBuilderKamino = basics_kamino.build_boxes_fourbar(
            builder=None,
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

        # Duplicate the world to test multi-world handling
        builder_1.add_builder(copy.deepcopy(builder_1))

        # Create models from the builders and conversion operations, and check for consistency
        model_0: Model = builder_0.finalize(skip_validation_joints=True)
        model_1: ModelKamino = builder_1.finalize()
        model_2: ModelKamino = ModelKamino.from_newton(model_0)
        test_util_checks.assert_model_equal(self, model_2, model_1)

    def test_01_model_conversions_fourbar_from_usd(self):
        """
        Test the conversion operations between newton.Model and kamino.ModelKamino
        on a simple fourbar model loaded from USD.
        """
        # Define the path to the USD file for the fourbar model
        asset_file = os.path.join(get_basics_usd_assets_path(), "boxes_fourbar.usda")

        # Create a fourbar using Newton's ModelBuilder and
        # register Kamino-specific custom attributes
        builder_0: ModelBuilder = ModelBuilder()
        SolverKamino.register_custom_attributes(builder_0)
        builder_0.default_shape_cfg.margin = 0.0
        builder_0.default_shape_cfg.gap = 0.0

        # Create a fourbar using Newton's ModelBuilder
        builder_0.begin_world()
        builder_0.add_usd(
            source=asset_file,
            joint_ordering=None,
            force_show_colliders=True,
            force_position_velocity_actuation=True,
        )
        builder_0.end_world()

        # Overwriting mu = 0.7 to match Kamino's default material properties
        builder_0.shape_material_mu = [0.7] * len(builder_0.shape_material_mu)

        # Import the same fourbar using Kamino's USDImporter and ModelBuilderKamino
        importer = USDImporter()
        builder_1: ModelBuilderKamino = importer.import_from(
            source=asset_file,
            load_drive_dynamics=True,
            load_static_geometry=True,
            force_show_colliders=True,
            use_prim_path_names=True,
            use_angular_drive_scaling=True,
        )

        # Create models from the builders and conversion operations, and check for consistency
        model_0: Model = builder_0.finalize(skip_validation_joints=True)
        model_1: ModelKamino = builder_1.finalize()
        model_2: ModelKamino = ModelKamino.from_newton(model_0)
        test_util_checks.assert_model_equal(self, model_2, model_1)

        # TODO: IMPLEMENT THIS CHECK: We wanna see if the both generate
        # the same data containers and unilateral constraint info
        # data_1: DataKamino = model_1.data()
        # data_2: DataKamino = model_2.data()
        # make_unilateral_constraints_info(model=model_1, data=data_1)
        # make_unilateral_constraints_info(model=model_2, data=data_2)
        # test_util_checks.assert_model_equal(self, model_2, model_1)
        # test_util_checks.assert_data_equal(self, data_2, data_1)

    def test_02_model_conversions_dr_testmech_from_usd(self):
        """
        Test the conversion operations between newton.Model and kamino.ModelKamino
        on the DR testmechanism model loaded from USD.
        """
        # Define the path to the USD file for the DR testmechanism model
        asset_path = newton.utils.download_asset("disneyresearch")
        asset_file = str(asset_path / "dr_testmech" / "usd" / "dr_testmech.usda")

        # Create a fourbar using Newton's ModelBuilder and
        # register Kamino-specific custom attributes
        builder_0: ModelBuilder = ModelBuilder()
        SolverKamino.register_custom_attributes(builder_0)
        builder_0.default_shape_cfg.margin = 0.0
        builder_0.default_shape_cfg.gap = 0.0

        # Create a fourbar using Newton's ModelBuilder
        builder_0.begin_world()
        builder_0.add_usd(
            source=asset_file,
            joint_ordering=None,
            force_show_colliders=True,
        )
        builder_0.end_world()

        # Overwriting mu = 0.7 to match Kamino's default material properties
        builder_0.shape_material_mu = [0.7] * len(builder_0.shape_material_mu)

        # Import the same fourbar using Kamino's USDImporter and ModelBuilderKamino
        importer = USDImporter()
        builder_1: ModelBuilderKamino = importer.import_from(
            source=asset_file,
            load_static_geometry=True,
            retain_joint_ordering=False,
            meshes_are_collidable=True,
            force_show_colliders=True,
            use_prim_path_names=True,
        )

        # Create models from the builders and conversion operations, and check for consistency
        model_0: Model = builder_0.finalize(skip_validation_joints=True)
        model_1: ModelKamino = builder_1.finalize()
        model_2: ModelKamino = ModelKamino.from_newton(model_0)
        # NOTE: We don't check:
        # - mesh geometry pointers since they have been loaded separately
        # Inversion of the inertia matrix amplifies small floating-point differences,
        # so inv_i_I_i needs a somewhat higher tolerance.
        rtol = {"inv_i_I_i": 1e-5}
        atol = {"inv_i_I_i": 1e-6}
        test_util_checks.assert_model_equal(self, model_2, model_1, excluded=["ptr"], rtol=rtol, atol=atol)

    def test_03_model_conversions_dr_legs_from_usd(self):
        """
        Test the conversion operations between newton.Model and kamino.ModelKamino
        on the DR legs model loaded from USD.
        """
        # Define the path to the USD file for the DR legs model
        asset_path = newton.utils.download_asset("disneyresearch")
        asset_file = str(asset_path / "dr_legs" / "usd" / "dr_legs_with_meshes_and_boxes.usda")

        # Create a fourbar using Newton's ModelBuilder and
        # register Kamino-specific custom attributes
        builder_0: ModelBuilder = ModelBuilder()
        SolverKamino.register_custom_attributes(builder_0)
        builder_0.default_shape_cfg.margin = 0.0
        builder_0.default_shape_cfg.gap = 0.0

        # Create a fourbar using Newton's ModelBuilder
        builder_0.begin_world()
        builder_0.add_usd(
            source=asset_file,
            joint_ordering=None,
            force_show_colliders=True,
            force_position_velocity_actuation=True,
        )
        builder_0.end_world()

        # Overwriting mu = 0.7 to match Kamino's default material properties
        builder_0.shape_material_mu = [0.7] * len(builder_0.shape_material_mu)

        # Import the same fourbar using Kamino's USDImporter and ModelBuilderKamino
        importer = USDImporter()
        builder_1: ModelBuilderKamino = importer.import_from(
            source=asset_file,
            load_drive_dynamics=True,
            force_show_colliders=True,
            use_prim_path_names=True,
            use_angular_drive_scaling=True,
        )

        # Create models from the builders and conversion operations, and check for consistency
        model_0: Model = builder_0.finalize(skip_validation_joints=True)
        model_1: ModelKamino = builder_1.finalize()
        model_2: ModelKamino = ModelKamino.from_newton(model_0)
        # NOTE: We don't check:
        # - mesh geometry pointers since they have been loaded separately
        # - the shape contact group (TODO @nvtw: investigate why) because newton.ModelBuilder
        #   sets it to `1` even for non-collidable visual shapes
        # - shape gap since newton.ModelBuilder sets it to `0.001` for all shapes even if
        #   the default shape config has gap=0.0
        # - excluded/filtered collision pairs since newton.ModelBuilder preemptively adds
        #   geom-pairs of joint neighbours to `shape_collision_filter_pairs` regardless of
        #   whether they are actually collidable or not, which leads to differences in the
        #   number of excluded pairs and their contents
        excluded = ["ptr", "group", "gap", "num_excluded_pairs", "excluded_pairs"]
        rtol = {"inv_i_I_i": 1e-5}
        atol = {"inv_i_I_i": 1e-6}
        test_util_checks.assert_model_equal(self, model_2, model_1, excluded=excluded, rtol=rtol, atol=atol)

    def test_04_model_conversions_anymal_d_from_usd(self):
        """
        Test the conversion operations between newton.Model and kamino.ModelKamino
        on the Anymal D model loaded from USD.
        """
        # Define the path to the USD file for the Anymal D model
        asset_path = newton.utils.download_asset("anybotics_anymal_d")
        asset_file = str(asset_path / "usd" / "anymal_d.usda")

        # Create a fourbar using Newton's ModelBuilder and
        # register Kamino-specific custom attributes
        builder_0: ModelBuilder = ModelBuilder()
        SolverKamino.register_custom_attributes(builder_0)
        builder_0.default_shape_cfg.margin = 0.0
        builder_0.default_shape_cfg.gap = 0.0

        # Create a fourbar using Newton's ModelBuilder
        builder_0.begin_world()
        builder_0.add_usd(
            source=asset_file,
            collapse_fixed_joints=False,
            enable_self_collisions=False,
            force_show_colliders=True,
        )
        builder_0.end_world()

        # Overwriting mu = 0.7 to match Kamino's default material properties
        builder_0.shape_material_mu = [0.7] * len(builder_0.shape_material_mu)

        # Import the same fourbar using Kamino's USDImporter and ModelBuilderKamino
        importer = USDImporter()
        builder_1: ModelBuilderKamino = importer.import_from(
            source=asset_file,
            load_static_geometry=True,
            retain_geom_ordering=False,
            use_articulation_root_name=False,
            force_show_colliders=True,
            use_prim_path_names=True,
            use_angular_drive_scaling=True,
        )

        # Create models from the builders and conversion operations, and check for consistency
        model_0: Model = builder_0.finalize()
        # TODO @nvtw: Why are shape_collision_group[i] values for
        # visual shapes set to `=1` since they are not collidable?
        msg.error(f"model_0.shape_collision_group:\n{model_0.shape_collision_group}\n")
        model_1: ModelKamino = builder_1.finalize()
        model_2: ModelKamino = ModelKamino.from_newton(model_0)
        # NOTE: We don't check mesh geometry pointers since they have been loaded separately
        excluded = [
            "i_r_com_i",  # TODO: Investigate if the difference is expected or not
            "i_I_i",  # TODO: Investigate if the difference is expected or not
            "inv_i_I_i",  # TODO: Investigate if the difference is expected or not
            "q_i_0",  # TODO: Investigate if the difference is expected or not
            "B_r_Bj",  # TODO: Investigate if the difference is expected or not
            "F_r_Fj",  # TODO: Investigate if the difference is expected or not
            "X_j",  # TODO: Investigate if the difference is expected or not
            "q_j_0",  # TODO: Investigate if the difference is expected or not
            "num_collidable_pairs",  # TODO: newton.ModelBuilder preemptively adding geom-pairs to shape_collision_filter_pairs
            "num_excluded_pairs",  # TODO: newton.ModelBuilder preemptively adding geom-pairs to shape_collision_filter_pairs
            "model_minimum_contacts",  # TODO: Investigate
            "world_minimum_contacts",  # TODO: Investigate
            "offset",  # TODO: Investigate if the difference is expected or not
            "group",  # TODO: newton.ModelBuilder setting shape_collision_group=1 for all shapes even non-collidable ones
            "gap",  # TODO: newton.ModelBuilder setting shape gap to 0.001 for all shapes even if default shape config has gap=0.0
            "ptr",  # Exclude geometry pointers since they have been loaded separately
            "collidable_pairs",  # TODO @nvtw: not sure why these are different
            "excluded_pairs",  # TODO: newton.ModelBuilder preemptively adding geom-pairs to shape_collision_filter_pairs
        ]
        test_util_checks.assert_model_equal(self, model_2, model_1, excluded=excluded)

    def test_05_model_conversions_arbitrary_axis(self):
        """
        Test that Newton→Kamino conversion succeeds for a revolute joint
        with an arbitrary (non-canonical) axis, e.g. ``(1, 1, 0)``.
        """
        builder: ModelBuilder = ModelBuilder()
        SolverKamino.register_custom_attributes(builder)
        builder.default_shape_cfg.margin = 0.0
        builder.default_shape_cfg.gap = 0.0

        builder.begin_world()

        # Parent body at origin
        bid0 = builder.add_link(
            label="base",
            mass=1.0,
            xform=wp.transformf(wp.vec3f(0.0, 0.0, 0.0), wp.quat_identity(dtype=wp.float32)),
            lock_inertia=True,
        )
        builder.add_shape_box(label="box_base", body=bid0, hx=0.05, hy=0.05, hz=0.05)

        # Child body offset along z
        bid1 = builder.add_link(
            label="pendulum",
            mass=1.0,
            xform=wp.transformf(wp.vec3f(0.0, 0.0, 0.5), wp.quat_identity(dtype=wp.float32)),
            lock_inertia=True,
        )
        builder.add_shape_box(label="box_pend", body=bid1, hx=0.05, hy=0.05, hz=0.25)

        # Fix the base to the world
        builder.add_joint_fixed(
            label="world_to_base",
            parent=-1,
            child=bid0,
            parent_xform=wp.transform_identity(dtype=wp.float32),
            child_xform=wp.transform_identity(dtype=wp.float32),
        )

        # Diagonal revolute axis (non-canonical)
        axis_vec = wp.vec3(1.0, 1.0, 0.0)
        builder.add_joint_revolute(
            label="base_to_pendulum",
            parent=bid0,
            child=bid1,
            axis=axis_vec,
            parent_xform=wp.transformf(wp.vec3f(0.0, 0.0, 0.25), wp.quat_identity(dtype=wp.float32)),
            child_xform=wp.transformf(wp.vec3f(0.0, 0.0, -0.25), wp.quat_identity(dtype=wp.float32)),
        )

        builder.end_world()

        model: Model = builder.finalize(skip_validation_joints=True)

        # Conversion must succeed (previously raised ValueError)
        kamino_model: ModelKamino = ModelKamino.from_newton(model)

        # Verify X_j first column is aligned with the expected axis direction
        X_j = kamino_model.joints.X_j.numpy()
        # X_j has shape (num_joints, 3, 3); the revolute joint is the second one (index 1)
        R = X_j[1]  # 3x3 rotation matrix
        ax_col = R[:, 0]  # first column = joint axis direction
        expected_ax = np.array([1.0, 1.0, 0.0])
        expected_ax = expected_ax / np.linalg.norm(expected_ax)
        np.testing.assert_allclose(ax_col, expected_ax, atol=1e-6)

    def test_06_model_conversions_q_i_0_com_frame(self):
        """
        Test that ``q_i_0`` stores COM world poses (not body-origin poses)
        after Newton→Kamino conversion for bodies with non-zero COM offsets.
        """
        builder: ModelBuilder = ModelBuilder()
        SolverKamino.register_custom_attributes(builder)
        builder.default_shape_cfg.margin = 0.0
        builder.default_shape_cfg.gap = 0.0

        builder.begin_world()

        # Body 0: at origin, identity rotation, COM offset along x
        bid0 = builder.add_link(
            label="body0",
            mass=1.0,
            xform=wp.transformf(wp.vec3f(0.0, 0.0, 0.0), wp.quat_identity(dtype=wp.float32)),
            com=wp.vec3f(0.1, 0.0, 0.0),
            lock_inertia=True,
        )
        builder.add_shape_box(label="box0", body=bid0, hx=0.05, hy=0.05, hz=0.05)

        # Body 1: at (0,0,1), rotated 90° about z-axis, single-axis COM offset
        rot_90z = wp.quat_from_axis_angle(wp.vec3f(0.0, 0.0, 1.0), np.pi / 2.0)
        bid1 = builder.add_link(
            label="body1",
            mass=1.0,
            xform=wp.transformf(wp.vec3f(0.0, 0.0, 1.0), rot_90z),
            com=wp.vec3f(0.1, 0.0, 0.0),
            lock_inertia=True,
        )
        builder.add_shape_box(label="box1", body=bid1, hx=0.05, hy=0.05, hz=0.05)

        # Body 2: at (1,0,0), rotated 90° about x-axis, 3D COM offset
        rot_90x = wp.quat_from_axis_angle(wp.vec3f(1.0, 0.0, 0.0), np.pi / 2.0)
        bid2 = builder.add_link(
            label="body2",
            mass=1.0,
            xform=wp.transformf(wp.vec3f(1.0, 0.0, 0.0), rot_90x),
            com=wp.vec3f(0.1, 0.2, 0.3),
            lock_inertia=True,
        )
        builder.add_shape_box(label="box2", body=bid2, hx=0.05, hy=0.05, hz=0.05)

        # Fix body 0 to world
        builder.add_joint_fixed(
            label="world_to_body0",
            parent=-1,
            child=bid0,
            parent_xform=wp.transform_identity(dtype=wp.float32),
            child_xform=wp.transform_identity(dtype=wp.float32),
        )

        # Revolute joint: body 0 → body 1
        builder.add_joint_revolute(
            label="body0_to_body1",
            parent=bid0,
            child=bid1,
            axis=wp.vec3(0.0, 1.0, 0.0),
            parent_xform=wp.transformf(wp.vec3f(0.0, 0.0, 0.5), wp.quat_identity(dtype=wp.float32)),
            child_xform=wp.transformf(wp.vec3f(0.0, 0.0, -0.5), wp.quat_identity(dtype=wp.float32)),
        )

        # Revolute joint: body 1 → body 2
        builder.add_joint_revolute(
            label="body1_to_body2",
            parent=bid1,
            child=bid2,
            axis=wp.vec3(0.0, 1.0, 0.0),
            parent_xform=wp.transformf(wp.vec3f(0.5, 0.0, 0.0), wp.quat_identity(dtype=wp.float32)),
            child_xform=wp.transformf(wp.vec3f(-0.5, 0.0, 0.0), wp.quat_identity(dtype=wp.float32)),
        )

        builder.end_world()

        model: Model = builder.finalize(skip_validation_joints=True)
        kamino_model: ModelKamino = ModelKamino.from_newton(model)

        q_i_0_np = kamino_model.bodies.q_i_0.numpy()  # shape (N, 7)
        body_q_np = model.body_q.numpy()

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

    def _build_com_offset_model(self):
        """Build a 3-body chain with non-zero COM offsets for reset tests."""
        builder: ModelBuilder = ModelBuilder()
        SolverKamino.register_custom_attributes(builder)
        builder.default_shape_cfg.margin = 0.0
        builder.default_shape_cfg.gap = 0.0

        builder.begin_world()

        # Body 0: at origin, identity rotation, COM offset along x
        bid0 = builder.add_link(
            label="body0",
            mass=1.0,
            xform=wp.transformf(wp.vec3f(0.0, 0.0, 0.0), wp.quat_identity(dtype=wp.float32)),
            com=wp.vec3f(0.1, 0.0, 0.0),
            lock_inertia=True,
        )
        builder.add_shape_box(label="box0", body=bid0, hx=0.05, hy=0.05, hz=0.05)

        # Body 1: at (0,0,1), rotated 90° about z-axis, single-axis COM offset
        rot_90z = wp.quat_from_axis_angle(wp.vec3f(0.0, 0.0, 1.0), np.pi / 2.0)
        bid1 = builder.add_link(
            label="body1",
            mass=1.0,
            xform=wp.transformf(wp.vec3f(0.0, 0.0, 1.0), rot_90z),
            com=wp.vec3f(0.1, 0.0, 0.0),
            lock_inertia=True,
        )
        builder.add_shape_box(label="box1", body=bid1, hx=0.05, hy=0.05, hz=0.05)

        # Body 2: at (1,0,0), rotated 90° about x-axis, 3D COM offset
        rot_90x = wp.quat_from_axis_angle(wp.vec3f(1.0, 0.0, 0.0), np.pi / 2.0)
        bid2 = builder.add_link(
            label="body2",
            mass=1.0,
            xform=wp.transformf(wp.vec3f(1.0, 0.0, 0.0), rot_90x),
            com=wp.vec3f(0.1, 0.2, 0.3),
            lock_inertia=True,
        )
        builder.add_shape_box(label="box2", body=bid2, hx=0.05, hy=0.05, hz=0.05)

        # Fix body 0 to world
        builder.add_joint_fixed(
            label="world_to_body0",
            parent=-1,
            child=bid0,
            parent_xform=wp.transform_identity(dtype=wp.float32),
            child_xform=wp.transform_identity(dtype=wp.float32),
        )

        # Revolute joint: body 0 -> body 1
        builder.add_joint_revolute(
            label="body0_to_body1",
            parent=bid0,
            child=bid1,
            axis=wp.vec3(0.0, 1.0, 0.0),
            parent_xform=wp.transformf(wp.vec3f(0.0, 0.0, 0.5), wp.quat_identity(dtype=wp.float32)),
            child_xform=wp.transformf(wp.vec3f(0.0, 0.0, -0.5), wp.quat_identity(dtype=wp.float32)),
        )

        # Revolute joint: body 1 -> body 2
        builder.add_joint_revolute(
            label="body1_to_body2",
            parent=bid1,
            child=bid2,
            axis=wp.vec3(0.0, 1.0, 0.0),
            parent_xform=wp.transformf(wp.vec3f(0.5, 0.0, 0.0), wp.quat_identity(dtype=wp.float32)),
            child_xform=wp.transformf(wp.vec3f(-0.5, 0.0, 0.0), wp.quat_identity(dtype=wp.float32)),
        )

        builder.end_world()

        return builder.finalize(skip_validation_joints=True)

    def test_07_reset_produces_body_origin_frame(self):
        """
        Test that ``SolverKamino.reset()`` writes body-origin frame poses
        into ``state.body_q``, not COM-frame poses, for bodies with non-zero
        COM offsets.
        """
        model = self._build_com_offset_model()
        body_q_expected = model.body_q.numpy().copy()

        solver = SolverKamino(model)

        # Default reset (no args) should restore body-origin poses
        state_out: State = model.state()
        solver.reset(state_out=state_out)
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

    def test_08_base_reset_produces_body_origin_frame(self):
        """
        Test that ``SolverKamino.reset(base_q=..., base_u=...)`` writes
        body-origin frame poses and velocities into ``state.body_q`` and
        ``state.body_qd`` for bodies with non-zero COM offsets.
        """
        model = self._build_com_offset_model()
        body_q_expected = model.body_q.numpy().copy()

        solver = SolverKamino(model)

        # --- Base reset with identity base pose should restore body-origin poses ---
        state_out: State = model.state()
        base_q = wp.array(
            [wp.transformf(wp.vec3f(0.0, 0.0, 0.0), wp.quat_identity(dtype=wp.float32))],
            dtype=wp.transformf,
        )
        base_u = wp.zeros(1, dtype=wp.spatial_vectorf)
        solver.reset(state_out=state_out, base_q=base_q, base_u=base_u)
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
        solver.reset(state_out=state_out, base_q=base_q_shifted, base_u=base_u)
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

    def test_09_model_conversions_shape_offset_com_relative(self):
        """
        Test that ``geoms.offset`` stores COM-relative shape positions
        after Newton→Kamino conversion, while ground shapes are unchanged.
        """
        builder: ModelBuilder = ModelBuilder()
        SolverKamino.register_custom_attributes(builder)
        builder.default_shape_cfg.margin = 0.0
        builder.default_shape_cfg.gap = 0.0

        builder.begin_world()

        # Body with COM=(0.1, 0.2, 0.0), shape at (0.5, 0.0, 0.0)
        bid = builder.add_link(
            label="body0",
            mass=1.0,
            xform=wp.transformf(wp.vec3f(0.0, 0.0, 0.0), wp.quat_identity(dtype=wp.float32)),
            com=wp.vec3f(0.1, 0.2, 0.0),
            lock_inertia=True,
        )
        builder.add_shape_box(
            label="box0",
            body=bid,
            hx=0.05,
            hy=0.05,
            hz=0.05,
            xform=wp.transformf(wp.vec3f(0.5, 0.0, 0.0), wp.quat_identity(dtype=wp.float32)),
        )
        # Ground shape (bid=-1) — should be left unchanged
        builder.add_shape_box(
            label="ground_box",
            body=-1,
            hx=1.0,
            hy=1.0,
            hz=0.01,
            xform=wp.transformf(wp.vec3f(1.0, 0.0, 0.0), wp.quat_identity(dtype=wp.float32)),
        )

        builder.add_joint_fixed(
            label="fix",
            parent=-1,
            child=bid,
            parent_xform=wp.transform_identity(dtype=wp.float32),
            child_xform=wp.transform_identity(dtype=wp.float32),
        )
        builder.end_world()

        model: Model = builder.finalize(skip_validation_joints=True)
        kamino_model: ModelKamino = ModelKamino.from_newton(model)
        offset_np = kamino_model.geoms.offset.numpy()

        # Shape on body: pos should be (0.5-0.1, 0.0-0.2, 0.0) = (0.4, -0.2, 0.0)
        np.testing.assert_allclose(offset_np[0, :3], [0.4, -0.2, 0.0], atol=1e-6)
        # Ground shape: pos unchanged at (1.0, 0.0, 0.0)
        np.testing.assert_allclose(offset_np[1, :3], [1.0, 0.0, 0.0], atol=1e-6)

    def test_10_origin_com_roundtrip(self):
        """
        Test that origin→COM→origin is the identity on body_q.
        """
        model = self._build_com_offset_model()
        body_q = wp.clone(model.body_q)
        q_orig = body_q.numpy().copy()

        convert_body_origin_to_com(model.body_com, body_q, body_q)
        convert_body_com_to_origin(model.body_com, body_q, body_q)

        np.testing.assert_allclose(body_q.numpy(), q_orig, atol=1e-6, err_msg="body_q roundtrip failed")

    def test_11_model_conversions_material_fourbar_from_builder(self):
        """
        Test the conversion operations between newton.Model and kamino.ModelKamino
        on a simple fourbar model with different materials, created explicitly using the builder.
        """
        # Create a fourbar using Newton's ModelBuilder and
        # register Kamino-specific custom attributes
        builder_0: ModelBuilder = ModelBuilder()
        SolverKamino.register_custom_attributes(builder_0)
        builder_0.default_shape_cfg.margin = 0.0
        builder_0.default_shape_cfg.gap = 0.0

        # Create a fourbar using Newton's ModelBuilder
        builder_0: ModelBuilder = basics_newton.build_boxes_fourbar(
            builder=builder_0,
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

        # Setting material properties
        restitution = [0.1, 0.2, 0.3, 0.4, 0.5]
        mu = [0.5, 0.6, 0.7, 0.8, 0.9]
        builder_0.shape_material_restitution = list(restitution)
        builder_0.shape_material_mu = list(mu)

        # Duplicate the world to test multi-world handling
        builder_0.begin_world()
        builder_0.add_builder(copy.deepcopy(builder_0))
        builder_0.end_world()

        # Create a fourbar using Kamino's ModelBuilderKamino
        builder_1: ModelBuilderKamino = basics_kamino.build_boxes_fourbar(
            builder=None,
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

        # Setting material properties
        for i in range(len(mu)):
            mid = builder_1.add_material(
                MaterialDescriptor(
                    name=f"mat{i}",
                    restitution=restitution[i],
                    static_friction=mu[i],
                    dynamic_friction=mu[i],
                )
            )
            builder_1.geoms[i].material = mid
            builder_1.geoms[i].mid = mid

        # Duplicate the world to test multi-world handling
        builder_1.add_builder(copy.deepcopy(builder_1))

        # Create models from the builders and conversion operations, and check for consistency
        model_0: Model = builder_0.finalize(skip_validation_joints=True)
        model_1: ModelKamino = builder_1.finalize()
        model_2: ModelKamino = ModelKamino.from_newton(model_0)
        test_util_checks.assert_model_equal(self, model_2, model_1)

    def test_12_model_conversions_material_box_on_plane_from_usd(self):
        """
        Test the conversion operations between newton.Model and kamino.ModelKamino
        on a simple box on plane model loaded from USD, containing different materials.
        """
        # Define the path to the USD file for the fourbar model
        asset_file = os.path.join(get_basics_usd_assets_path(), "box_on_plane.usda")

        # Create a fourbar using Newton's ModelBuilder and
        # register Kamino-specific custom attributes
        builder_0: ModelBuilder = ModelBuilder()
        SolverKamino.register_custom_attributes(builder_0)
        builder_0.default_shape_cfg.margin = 0.0
        builder_0.default_shape_cfg.gap = 0.0

        # Create a fourbar using Newton's ModelBuilder
        builder_0.begin_world()
        builder_0.add_usd(
            source=asset_file,
            joint_ordering=None,
            force_show_colliders=True,
            force_position_velocity_actuation=True,
        )
        builder_0.end_world()

        # Duplicate the world to test multi-world handling
        builder_0.begin_world()
        builder_0.add_builder(copy.deepcopy(builder_0))
        builder_0.end_world()

        # Import the same fourbar using Kamino's USDImporter and ModelBuilderKamino
        importer = USDImporter()
        builder_1: ModelBuilderKamino = importer.import_from(
            source=asset_file,
            load_drive_dynamics=True,
            load_static_geometry=True,
            force_show_colliders=True,
            use_prim_path_names=True,
        )

        # Resetting default material parameters, since the Newton USD importer does not import a
        # default material and therefore does not have a non-standard default material
        builder_1.materials[0].dynamic_friction = 0.7

        # Overwriting dynamic friction with static friction, since the Newton USD importer only
        # imports static friction and the Kamino conversion uses this to initialize both parameters
        for mat in builder_1.materials:
            mat.static_friction = mat.dynamic_friction

        # Duplicate the world to test multi-world handling
        builder_1.add_builder(copy.deepcopy(builder_1))

        # Create models from the builders and conversion operations, and check for consistency
        model_0: Model = builder_0.finalize(skip_validation_joints=True)
        model_1: ModelKamino = builder_1.finalize()
        model_2: ModelKamino = ModelKamino.from_newton(model_0)

        msg.warning(f"{model_1.materials.restitution}")
        msg.warning(f"{model_2.materials.restitution}")
        msg.warning(f"{model_1.material_pairs.restitution}")
        msg.warning(f"{model_2.material_pairs.restitution}")

        test_util_checks.assert_model_geoms_equal(self, model_2.geoms, model_1.geoms)
        test_util_checks.assert_model_materials_equal(self, model_2.materials, model_1.materials)
        # TODO: Material pairs are currently not checked. The Kamino USD importer will set material
        #       pair properties based on the list of materials, using the average of the material
        #       properties. The Newton-to-Kamino conversion will leave the material pair properties
        #       uninitialized, leaving the choice of how to combine materials for a pair to the
        #       runtime material resolution system (see :class:`MaterialMuxMode`).
        # test_util_checks.assert_model_material_pairs_equal(self, model_2.material_pairs, model_1.material_pairs)

    def test_20_state_conversions(self):
        """
        Test the conversion operations between newton.State and kamino.StateKamino.
        """
        # Create a fourbar using Newton's ModelBuilder and
        # register Kamino-specific custom attributes
        builder_0: ModelBuilder = ModelBuilder()
        SolverKamino.register_custom_attributes(builder_0)
        builder_0.default_shape_cfg.margin = 0.0
        builder_0.default_shape_cfg.gap = 0.0

        # Create a fourbar using Newton's ModelBuilder
        builder_0: ModelBuilder = basics_newton.build_boxes_fourbar(
            builder=builder_0,
            z_offset=0.0,
            fixedbase=False,
            floatingbase=True,
            limits=True,
            ground=True,
            new_world=True,
            actuator_ids=[2, 4],
        )

        # Overwriting mu = 0.7 to match Kamino's default material properties
        builder_0.shape_material_mu = [0.7] * len(builder_0.shape_material_mu)

        # Duplicate the world to test multi-world handling
        builder_0.begin_world()
        builder_0.add_builder(copy.deepcopy(builder_0))
        builder_0.end_world()

        # Create a fourbar using Kamino's ModelBuilderKamino
        builder_1: ModelBuilderKamino = basics_kamino.build_boxes_fourbar(
            builder=None,
            z_offset=0.0,
            fixedbase=False,
            floatingbase=True,
            limits=True,
            ground=True,
            new_world=True,
            actuator_ids=[2, 4],
        )

        # Duplicate the world to test multi-world handling
        builder_1.add_builder(copy.deepcopy(builder_1))

        # Create models from the builders and conversion operations, and check for consistency
        model_0: Model = builder_0.finalize(skip_validation_joints=True)
        model_1: ModelKamino = builder_1.finalize()
        model_2: ModelKamino = ModelKamino.from_newton(model_0)
        test_util_checks.assert_model_equal(self, model_1, model_2)

        # Create a Newton state container
        state_0: State = model_0.state()
        self.assertIsInstance(state_0.body_q, wp.array)
        self.assertEqual(state_0.body_q.size, model_0.body_count)
        self.assertIsNotNone(state_0.joint_q_prev)
        self.assertEqual(state_0.joint_q_prev.size, model_0.joint_coord_count)
        self.assertIsNotNone(state_0.joint_lambdas)
        self.assertEqual(state_0.joint_lambdas.size, model_0.joint_constraint_count)

        # Create a Kamino state container
        state_1: StateKamino = model_1.state()
        self.assertIsInstance(state_1.q_i, wp.array)
        self.assertEqual(state_1.q_i.size, model_1.size.sum_of_num_bodies)

        state_2: StateKamino = StateKamino.from_newton(model_2.size, model_0, state_0, True, False)
        self.assertIsInstance(state_2.q_i, wp.array)
        self.assertEqual(state_2.q_i.size, model_1.size.sum_of_num_bodies)
        # NOTE: Check ptr due to conversion from wp.spatial_vectorf
        self.assertIs(state_2.u_i.ptr, state_0.body_qd.ptr)
        self.assertIs(state_2.w_i_e.ptr, state_0.body_f.ptr)
        self.assertIs(state_2.w_i.ptr, state_0.body_f_total.ptr)
        # NOTE: Check that the same arrays because these should be pure references
        self.assertIs(state_2.q_i, state_0.body_q)
        self.assertIs(state_2.q_j, state_0.joint_q)
        self.assertIs(state_2.dq_j, state_0.joint_qd)
        self.assertIs(state_2.q_j_p, state_0.joint_q_prev)
        self.assertIs(state_2.lambda_j, state_0.joint_lambdas)
        test_util_checks.assert_state_equal(self, state_2, state_1)

        state_3: State = StateKamino.to_newton(model_0, state_2)
        self.assertIsInstance(state_3.body_q, wp.array)
        self.assertEqual(state_3.body_q.size, model_0.body_count)
        # NOTE: Check ptr due to conversion from vec6f
        self.assertIs(state_3.body_qd.ptr, state_2.u_i.ptr)
        self.assertIs(state_3.body_f.ptr, state_2.w_i_e.ptr)
        self.assertIs(state_3.body_f_total.ptr, state_2.w_i.ptr)
        # NOTE: Check that the same arrays because these should be pure references
        self.assertIs(state_3.body_q, state_2.q_i)
        self.assertIs(state_3.joint_q, state_2.q_j)
        self.assertIs(state_3.joint_qd, state_2.dq_j)
        self.assertIs(state_3.joint_q_prev, state_2.q_j_p)
        self.assertIs(state_3.joint_lambdas, state_2.lambda_j)

    def test_30_control_conversions(self):
        """
        Test the conversions between newton.Control and kamino.ControlKamino.
        """
        # Create a fourbar using Newton's ModelBuilder and
        # register Kamino-specific custom attributes
        builder_0: ModelBuilder = ModelBuilder()
        SolverKamino.register_custom_attributes(builder_0)
        builder_0.default_shape_cfg.margin = 0.0
        builder_0.default_shape_cfg.gap = 0.0

        # Create a fourbar using Newton's ModelBuilder
        builder_0: ModelBuilder = basics_newton.build_boxes_fourbar(
            builder=builder_0,
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

        # Overwriting mu = 0.7 to match Kamino's default material properties
        builder_0.shape_material_mu = [0.7] * len(builder_0.shape_material_mu)

        # Duplicate the world to test multi-world handling
        builder_0.begin_world()
        builder_0.add_builder(copy.deepcopy(builder_0))
        builder_0.end_world()

        # Create a fourbar using Kamino's ModelBuilderKamino
        builder_1: ModelBuilderKamino = basics_kamino.build_boxes_fourbar(
            builder=None,
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

        # Duplicate the world to test multi-world handling
        builder_1.add_builder(copy.deepcopy(builder_1))

        # Create models from the builders and conversion operations, and check for consistency
        model_0: Model = builder_0.finalize(skip_validation_joints=True)
        model_1: ModelKamino = builder_1.finalize()
        model_2: ModelKamino = ModelKamino.from_newton(model_0)
        test_util_checks.assert_model_equal(self, model_2, model_1)

        # Create a Newton control container
        control_0: Control = model_0.control()
        self.assertIsInstance(control_0.joint_f, wp.array)
        self.assertEqual(control_0.joint_f.size, model_0.joint_dof_count)

        # Create a Kamino control container
        control_1: ControlKamino = model_1.control()
        self.assertIsInstance(control_1.tau_j, wp.array)
        self.assertEqual(control_1.tau_j.size, model_1.size.sum_of_num_joint_dofs)

        # Create a Kamino control container
        control_2: ControlKamino = ControlKamino.from_newton(control_0)
        self.assertIsInstance(control_2.tau_j, wp.array)
        self.assertIs(control_2.tau_j, control_0.joint_f)
        self.assertEqual(control_2.tau_j.size, model_0.joint_dof_count)
        test_util_checks.assert_control_equal(self, control_2, control_1)

        # Convert back to a Newton control container
        control_3: Control = ControlKamino.to_newton(control_2)
        self.assertIsInstance(control_3.joint_f, wp.array)
        self.assertIs(control_3.joint_f, control_2.tau_j)
        self.assertEqual(control_3.joint_f.size, model_0.joint_dof_count)


###
# Test execution
###

if __name__ == "__main__":
    # Test setup
    setup_tests()

    # Run all tests
    unittest.main(verbosity=2)
