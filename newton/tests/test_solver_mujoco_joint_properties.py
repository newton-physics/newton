# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import unittest
import warnings

import numpy as np
import warp as wp

import newton
from newton import JointType, ModelFlags
from newton._src.core.types import vec5
from newton.solvers import SolverMuJoCo
from newton.tests.utils.mujoco import _expected_positive_limit_solref, _MuJoCoSolverPropertiesFixture


class TestMuJoCoSolverJointProperties(unittest.TestCase):
    def test_joint_properties_conversion_and_updates(self):
        """Convert joint attributes and update drive properties, limit gains, and ranges."""
        _MuJoCoSolverPropertiesFixture.setUp(self)
        if self.model.joint_dof_count == 0:
            self.skipTest("No joints in model, skipping joint attributes test")
        dofs_per_world = self.model.joint_dof_count // self.model.world_count
        joints_per_world = self.model.joint_count // self.model.world_count
        initial_effort_limits = np.zeros(self.model.joint_dof_count)
        initial_velocity_limits = np.zeros(self.model.joint_dof_count)
        initial_friction = np.zeros(self.model.joint_dof_count)
        initial_armature = np.zeros(self.model.joint_dof_count)
        joint_qd_start = self.model.joint_qd_start.numpy()
        joint_dof_dim = self.model.joint_dof_dim.numpy()
        joint_type = self.model.joint_type.numpy()
        for world_idx in range(self.model.world_count):
            world_joint_offset = world_idx * joints_per_world

            for joint_idx in range(joints_per_world):
                global_joint_idx = world_joint_offset + joint_idx

                # Skip free joints
                if joint_type[global_joint_idx] == JointType.FREE:
                    continue

                # Get DOF start and count for this joint
                dof_start = joint_qd_start[global_joint_idx]
                dof_count = joint_dof_dim[global_joint_idx].sum()

                # Set values for each DOF in this joint
                for dof_offset in range(dof_count):
                    global_dof_idx = dof_start + dof_offset

                    # Effort limit: 50 + dof_offset * 10 + joint_idx * 5 + world_idx * 100
                    initial_effort_limits[global_dof_idx] = (
                        50.0 + dof_offset * 10.0 + joint_idx * 5.0 + world_idx * 100.0
                    )
                    # Velocity limit: 10 + dof_offset * 2 + joint_idx * 1 + world_idx * 20
                    initial_velocity_limits[global_dof_idx] = (
                        10.0 + dof_offset * 2.0 + joint_idx * 1.0 + world_idx * 20.0
                    )
                    # Friction: 0.5 + dof_offset * 0.1 + joint_idx * 0.05 + world_idx * 0.5
                    initial_friction[global_dof_idx] = 0.5 + dof_offset * 0.1 + joint_idx * 0.05 + world_idx * 0.5
                    # Armature: 0.01 + dof_offset * 0.005 + joint_idx * 0.002 + world_idx * 0.05
                    initial_armature[global_dof_idx] = 0.01 + dof_offset * 0.005 + joint_idx * 0.002 + world_idx * 0.05
        self.model.joint_effort_limit.assign(initial_effort_limits)
        self.model.joint_velocity_limit.assign(initial_velocity_limits)
        self.model.joint_friction.assign(initial_friction)
        self.model.joint_armature.assign(initial_armature)
        initial_limit_ke = np.zeros(self.model.joint_dof_count)
        initial_limit_kd = np.zeros(self.model.joint_dof_count)
        for world_idx in range(self.model.world_count):
            world_dof_offset = world_idx * dofs_per_world

            for dof_idx in range(dofs_per_world):
                global_dof_idx = world_dof_offset + dof_idx
                # Stiffness: 1000 + dof_idx * 100 + world_idx * 1000
                initial_limit_ke[global_dof_idx] = 1000.0 + dof_idx * 100.0 + world_idx * 1000.0
                # Damping: 10 + dof_idx * 1 + world_idx * 10
                initial_limit_kd[global_dof_idx] = 10.0 + dof_idx * 1.0 + world_idx * 10.0
        self.model.joint_limit_ke.assign(initial_limit_ke)
        self.model.joint_limit_kd.assign(initial_limit_kd)
        initial_limit_lower = np.zeros(self.model.joint_dof_count)
        initial_limit_upper = np.zeros(self.model.joint_dof_count)
        for world_idx in range(self.model.world_count):
            world_dof_offset = world_idx * dofs_per_world

            for dof_idx in range(dofs_per_world):
                global_dof_idx = world_dof_offset + dof_idx
                # Lower limit: -2.0 - dof_idx * 0.1 - world_idx * 0.5
                initial_limit_lower[global_dof_idx] = -2.0 - dof_idx * 0.1 - world_idx * 0.5
                # Upper limit: 2.0 + dof_idx * 0.1 + world_idx * 0.5
                initial_limit_upper[global_dof_idx] = 2.0 + dof_idx * 0.1 + world_idx * 0.5
        self.model.joint_limit_lower.assign(initial_limit_lower)
        self.model.joint_limit_upper.assign(initial_limit_upper)
        solver = SolverMuJoCo(self.model, iterations=1, disable_contacts=True)
        with self.subTest(property="joint_attributes_registration_and_updates", phase="conversion"):
            for world_idx in range(self.model.world_count):
                for dof_idx in range(min(dofs_per_world, solver.mjw_model.dof_armature.shape[1])):
                    global_dof_idx = world_idx * dofs_per_world + dof_idx
                    expected_armature = initial_armature[global_dof_idx]
                    actual_armature = solver.mjw_model.dof_armature.numpy()[world_idx, dof_idx]
                    self.assertAlmostEqual(
                        actual_armature,
                        expected_armature,
                        places=3,
                        msg=f"MuJoCo DOF {dof_idx} in world {world_idx} armature should match Newton value",
                    )
            for world_idx in range(self.model.world_count):
                for dof_idx in range(min(dofs_per_world, solver.mjw_model.dof_frictionloss.shape[1])):
                    global_dof_idx = world_idx * dofs_per_world + dof_idx
                    expected_friction = initial_friction[global_dof_idx]
                    actual_friction = solver.mjw_model.dof_frictionloss.numpy()[world_idx, dof_idx]
                    self.assertAlmostEqual(
                        actual_friction,
                        expected_friction,
                        places=4,
                        msg=f"MuJoCo DOF {dof_idx} in world {world_idx} friction should match Newton value",
                    )

        with self.subTest(property="joint_limit_solref_conversion", phase="conversion"):
            mjc_revolute_indices = [2, 3]  # MuJoCo joint indices for revolute joints
            newton_revolute_dof_indices = [12, 13]  # Newton DOF indices for revolute joints
            jnt_dofadr = solver.mjw_model.jnt_dofadr.numpy()

            def expected_scaled_solref(world_idx, mjc_idx, ke, kd):
                dof_adr = int(jnt_dofadr[mjc_idx])
                invw = float(solver.mjw_model.dof_invweight0.numpy()[world_idx, dof_adr])
                dmax = float(solver.mjw_model.jnt_solimp.numpy()[world_idx, mjc_idx][1])
                factor = invw * (1.0 - dmax) if invw > 0.0 and dmax < 1.0 else 1.0
                return _expected_positive_limit_solref(ke, kd, factor)

            for world_idx in range(self.model.world_count):
                for _i, (mjc_idx, newton_dof_idx) in enumerate(
                    zip(mjc_revolute_indices, newton_revolute_dof_indices, strict=False)
                ):
                    global_dof_idx = world_idx * dofs_per_world + newton_dof_idx
                    expected_solref = expected_scaled_solref(
                        world_idx, mjc_idx, initial_limit_ke[global_dof_idx], initial_limit_kd[global_dof_idx]
                    )

                    # Get actual values from MuJoCo's jnt_solref array. Solref is
                    # stored in float32 while ``expected_*`` is computed in float64,
                    # so use a relative tolerance instead of ``places`` (absolute).
                    rel_tol = 1e-4
                    actual_solref = solver.mjw_model.jnt_solref.numpy()[world_idx, mjc_idx]
                    self.assertAlmostEqual(
                        float(actual_solref[0]),
                        expected_solref[0],
                        delta=abs(expected_solref[0]) * rel_tol,
                        msg=f"Initial solref time constant for MuJoCo joint {mjc_idx} (Newton DOF {newton_dof_idx}) in world {world_idx}",
                    )
                    self.assertAlmostEqual(
                        float(actual_solref[1]),
                        expected_solref[1],
                        delta=abs(expected_solref[1]) * rel_tol,
                        msg=f"Initial solref damping ratio for MuJoCo joint {mjc_idx} (Newton DOF {newton_dof_idx}) in world {world_idx}",
                    )

        with self.subTest(property="joint_limit_range_conversion", phase="conversion"):
            mjc_revolute_indices = [2, 3]  # MuJoCo joint indices for revolute joints
            newton_revolute_dof_indices = [12, 13]  # Newton DOF indices for revolute joints
            for world_idx in range(self.model.world_count):
                for _i, (mjc_idx, newton_dof_idx) in enumerate(
                    zip(mjc_revolute_indices, newton_revolute_dof_indices, strict=False)
                ):
                    global_dof_idx = world_idx * dofs_per_world + newton_dof_idx
                    expected_lower = initial_limit_lower[global_dof_idx]
                    expected_upper = initial_limit_upper[global_dof_idx]

                    # Get actual values from MuJoCo's jnt_range array
                    actual_range = solver.mjw_model.jnt_range.numpy()[world_idx, mjc_idx]
                    self.assertAlmostEqual(
                        actual_range[0],
                        expected_lower,
                        places=5,
                        msg=f"Initial range lower for MuJoCo joint {mjc_idx} (Newton DOF {newton_dof_idx}) in world {world_idx}",
                    )
                    self.assertAlmostEqual(
                        actual_range[1],
                        expected_upper,
                        places=5,
                        msg=f"Initial range upper for MuJoCo joint {mjc_idx} (Newton DOF {newton_dof_idx}) in world {world_idx}",
                    )

        with self.subTest(property="joint_attributes_registration_and_updates", phase="update"):
            updated_effort_limits = np.zeros(self.model.joint_dof_count)
            updated_velocity_limits = np.zeros(self.model.joint_dof_count)
            updated_friction = np.zeros(self.model.joint_dof_count)
            updated_armature = np.zeros(self.model.joint_dof_count)
            for world_idx in range(self.model.world_count):
                world_joint_offset = world_idx * joints_per_world

                for joint_idx in range(joints_per_world):
                    global_joint_idx = world_joint_offset + joint_idx

                    # Skip free joints
                    if joint_type[global_joint_idx] == JointType.FREE:
                        continue

                    # Get DOF start and count for this joint
                    dof_start = joint_qd_start[global_joint_idx]
                    dof_count = joint_dof_dim[global_joint_idx].sum()

                    # Set updated values for each DOF in this joint
                    for dof_offset in range(dof_count):
                        global_dof_idx = dof_start + dof_offset

                        # Updated effort limit: 100 + dof_offset * 15 + joint_idx * 8 + world_idx * 150
                        updated_effort_limits[global_dof_idx] = (
                            100.0 + dof_offset * 15.0 + joint_idx * 8.0 + world_idx * 150.0
                        )
                        # Updated velocity limit: 20 + dof_offset * 3 + joint_idx * 2 + world_idx * 30
                        updated_velocity_limits[global_dof_idx] = (
                            20.0 + dof_offset * 3.0 + joint_idx * 2.0 + world_idx * 30.0
                        )
                        # Updated friction: 1.0 + dof_offset * 0.2 + joint_idx * 0.1 + world_idx * 1.0
                        updated_friction[global_dof_idx] = 1.0 + dof_offset * 0.2 + joint_idx * 0.1 + world_idx * 1.0
                        # Updated armature: 0.05 + dof_offset * 0.01 + joint_idx * 0.005 + world_idx * 0.1
                        updated_armature[global_dof_idx] = (
                            0.05 + dof_offset * 0.01 + joint_idx * 0.005 + world_idx * 0.1
                        )
            self.model.joint_effort_limit.assign(updated_effort_limits)
            self.model.joint_velocity_limit.assign(updated_velocity_limits)
            self.model.joint_friction.assign(updated_friction)
            self.model.joint_armature.assign(updated_armature)
            solver.notify_model_changed(ModelFlags.JOINT_DOF_PROPERTIES)
            for world_idx in range(self.model.world_count):
                for dof_idx in range(min(dofs_per_world, solver.mjw_model.dof_armature.shape[1])):
                    global_dof_idx = world_idx * dofs_per_world + dof_idx
                    expected_armature = updated_armature[global_dof_idx]
                    actual_armature = solver.mjw_model.dof_armature.numpy()[world_idx, dof_idx]
                    self.assertAlmostEqual(
                        actual_armature,
                        expected_armature,
                        places=4,
                        msg=f"Updated MuJoCo DOF {dof_idx} in world {world_idx} armature should match Newton value",
                    )
            for world_idx in range(self.model.world_count):
                for dof_idx in range(min(dofs_per_world, solver.mjw_model.dof_frictionloss.shape[1])):
                    global_dof_idx = world_idx * dofs_per_world + dof_idx
                    expected_friction = updated_friction[global_dof_idx]
                    actual_friction = solver.mjw_model.dof_frictionloss.numpy()[world_idx, dof_idx]
                    self.assertAlmostEqual(
                        actual_friction,
                        expected_friction,
                        places=4,
                        msg=f"Updated MuJoCo DOF {dof_idx} in world {world_idx} friction should match Newton value",
                    )

        with self.subTest(property="joint_limit_solref_conversion", phase="update"):
            updated_limit_ke = initial_limit_ke * 2.0
            updated_limit_kd = initial_limit_kd * 2.0
            self.model.joint_limit_ke.assign(updated_limit_ke)
            self.model.joint_limit_kd.assign(updated_limit_kd)
            solver.notify_model_changed(ModelFlags.JOINT_DOF_PROPERTIES)
            for world_idx in range(self.model.world_count):
                for _i, (mjc_idx, newton_dof_idx) in enumerate(
                    zip(mjc_revolute_indices, newton_revolute_dof_indices, strict=False)
                ):
                    global_dof_idx = world_idx * dofs_per_world + newton_dof_idx
                    expected_solref = expected_scaled_solref(
                        world_idx, mjc_idx, updated_limit_ke[global_dof_idx], updated_limit_kd[global_dof_idx]
                    )

                    # Get actual values from MuJoCo's jnt_solref array.
                    rel_tol = 1e-4
                    actual_solref = solver.mjw_model.jnt_solref.numpy()[world_idx, mjc_idx]
                    self.assertAlmostEqual(
                        float(actual_solref[0]),
                        expected_solref[0],
                        delta=abs(expected_solref[0]) * rel_tol,
                        msg=f"Updated solref time constant for MuJoCo joint {mjc_idx} (Newton DOF {newton_dof_idx}) in world {world_idx}",
                    )
                    self.assertAlmostEqual(
                        float(actual_solref[1]),
                        expected_solref[1],
                        delta=abs(expected_solref[1]) * rel_tol,
                        msg=f"Updated solref damping ratio for MuJoCo joint {mjc_idx} (Newton DOF {newton_dof_idx}) in world {world_idx}",
                    )

        with self.subTest(property="joint_limit_range_conversion", phase="update"):
            updated_limit_lower = np.zeros(self.model.joint_dof_count)
            updated_limit_upper = np.zeros(self.model.joint_dof_count)
            for world_idx in range(self.model.world_count):
                world_dof_offset = world_idx * dofs_per_world

                for dof_idx in range(dofs_per_world):
                    global_dof_idx = world_dof_offset + dof_idx
                    # Different values per world to verify per-world updates
                    # Lower limit: -1.5 - dof_idx * 0.2 - world_idx * 1.0
                    updated_limit_lower[global_dof_idx] = -1.5 - dof_idx * 0.2 - world_idx * 1.0
                    # Upper limit: 1.5 + dof_idx * 0.2 + world_idx * 1.0
                    updated_limit_upper[global_dof_idx] = 1.5 + dof_idx * 0.2 + world_idx * 1.0
            self.model.joint_limit_lower.assign(updated_limit_lower)
            self.model.joint_limit_upper.assign(updated_limit_upper)
            solver.notify_model_changed(ModelFlags.JOINT_DOF_PROPERTIES)
            for world_idx in range(self.model.world_count):
                for _i, (mjc_idx, newton_dof_idx) in enumerate(
                    zip(mjc_revolute_indices, newton_revolute_dof_indices, strict=False)
                ):
                    global_dof_idx = world_idx * dofs_per_world + newton_dof_idx
                    expected_lower = updated_limit_lower[global_dof_idx]
                    expected_upper = updated_limit_upper[global_dof_idx]

                    # Get actual values from MuJoCo's jnt_range array
                    actual_range = solver.mjw_model.jnt_range.numpy()[world_idx, mjc_idx]
                    self.assertAlmostEqual(
                        actual_range[0],
                        expected_lower,
                        places=5,
                        msg=f"Updated range lower for MuJoCo joint {mjc_idx} (Newton DOF {newton_dof_idx}) in world {world_idx}",
                    )
                    self.assertAlmostEqual(
                        actual_range[1],
                        expected_upper,
                        places=5,
                        msg=f"Updated range upper for MuJoCo joint {mjc_idx} (Newton DOF {newton_dof_idx}) in world {world_idx}",
                    )
            for world_idx in range(self.model.world_count):
                for _i, (_mjc_idx, newton_dof_idx) in enumerate(
                    zip(mjc_revolute_indices, newton_revolute_dof_indices, strict=False)
                ):
                    global_dof_idx = world_idx * dofs_per_world + newton_dof_idx
                    initial_lower = initial_limit_lower[global_dof_idx]
                    initial_upper = initial_limit_upper[global_dof_idx]
                    updated_lower = updated_limit_lower[global_dof_idx]
                    updated_upper = updated_limit_upper[global_dof_idx]

                    # Verify values actually changed
                    self.assertNotAlmostEqual(
                        initial_lower,
                        updated_lower,
                        places=5,
                        msg=f"Range lower should have changed for Newton DOF {newton_dof_idx} in world {world_idx}",
                    )
                    self.assertNotAlmostEqual(
                        initial_upper,
                        updated_upper,
                        places=5,
                        msg=f"Range upper should have changed for Newton DOF {newton_dof_idx} in world {world_idx}",
                    )

    def test_jnt_solimp_conversion_and_updates(self):
        """
        Verify that custom solimplimit attribute:
        1. Is properly registered in Newton Model
        2. Is properly converted to MuJoCo jnt_solimp
        3. Can be changed during simulation via notify_model_changed()
        4. Is properly expanded for multi-world models

        Uses different values for each joint DOF and world to catch indexing bugs.
        """
        # Skip if no joints

        # Step 1: Create a template builder and register SolverMuJoCo custom attributes
        template_builder = newton.ModelBuilder()
        SolverMuJoCo.register_custom_attributes(template_builder)
        shape_cfg = newton.ModelBuilder.ShapeConfig(density=1000.0)

        # Free-floating body
        free_body_initial_pos = wp.transform((0.5, 0.5, 0.0), wp.quat_identity())
        free_body_idx = template_builder.add_body(mass=0.2, xform=free_body_initial_pos)
        template_builder.add_shape_box(body=free_body_idx, xform=wp.transform(), hx=0.1, hy=0.1, hz=0.1, cfg=shape_cfg)

        # Articulated tree
        link_radius = 0.05
        link_half_length = 0.15
        tree_root_initial_pos_y = link_half_length * 2.0
        tree_root_initial_transform = wp.transform((0.0, tree_root_initial_pos_y, 0.0), wp.quat_identity())

        body1_idx = template_builder.add_link(mass=0.1)
        joint1_idx = template_builder.add_joint_free(child=body1_idx, parent_xform=tree_root_initial_transform)
        template_builder.add_shape_capsule(
            body=body1_idx, xform=wp.transform(), radius=link_radius, half_height=link_half_length, cfg=shape_cfg
        )

        body2_idx = template_builder.add_link(mass=0.1)
        template_builder.add_shape_capsule(
            body=body2_idx, xform=wp.transform(), radius=link_radius, half_height=link_half_length, cfg=shape_cfg
        )
        joint2_idx = template_builder.add_joint_revolute(
            parent=body1_idx,
            child=body2_idx,
            axis=(1.0, 0.0, 0.0),
            parent_xform=wp.transform((0.0, link_half_length, 0.0), wp.quat_identity()),
            child_xform=wp.transform((0.0, -link_half_length, 0.0), wp.quat_identity()),
            limit_lower=-np.pi / 2,
            limit_upper=np.pi / 2,
        )

        body3_idx = template_builder.add_link(mass=0.1)
        template_builder.add_shape_capsule(
            body=body3_idx, xform=wp.transform(), radius=link_radius, half_height=link_half_length, cfg=shape_cfg
        )
        joint3_idx = template_builder.add_joint_revolute(
            parent=body2_idx,
            child=body3_idx,
            axis=(0.0, 1.0, 0.0),
            parent_xform=wp.transform((0.0, link_half_length, 0.0), wp.quat_identity()),
            child_xform=wp.transform((0.0, -link_half_length, 0.0), wp.quat_identity()),
            limit_lower=-np.pi / 3,
            limit_upper=np.pi / 3,
        )

        template_builder.add_articulation([joint1_idx, joint2_idx, joint3_idx])

        # Replicate to create multiple worlds
        world_count = 2
        builder = newton.ModelBuilder()
        builder.replicate(template_builder, world_count)
        model = builder.finalize()

        # Step 2: Set initial solimplimit values
        joints_per_world = model.joint_count // model.world_count

        # Create initial solimplimit array
        initial_solimplimit = np.zeros((model.joint_dof_count, 5), dtype=np.float32)

        # Iterate over joints and set values for each DOF (skip free joints)
        joint_qd_start = model.joint_qd_start.numpy()
        joint_dof_dim = model.joint_dof_dim.numpy()
        joint_type = model.joint_type.numpy()

        for world_idx in range(model.world_count):
            world_joint_offset = world_idx * joints_per_world

            for joint_idx in range(joints_per_world):
                global_joint_idx = world_joint_offset + joint_idx

                # Skip free joints
                if joint_type[global_joint_idx] == JointType.FREE:
                    continue

                # Get DOF start and count for this joint
                dof_start = joint_qd_start[global_joint_idx]
                dof_count = joint_dof_dim[global_joint_idx].sum()

                # Set values for each DOF in this joint
                for dof_offset in range(dof_count):
                    global_dof_idx = dof_start + dof_offset

                    # Pattern: base values + dof_offset * 0.01 + joint_idx * 0.005 + world_idx * 0.1
                    val0 = 0.89 + dof_offset * 0.01 + joint_idx * 0.005 + world_idx * 0.1
                    val1 = 0.90 + dof_offset * 0.01 + joint_idx * 0.005 + world_idx * 0.1
                    val2 = 0.01 + dof_offset * 0.001 + joint_idx * 0.0005 + world_idx * 0.01
                    val3 = 2.0 + dof_offset * 0.1 + joint_idx * 0.05 + world_idx * 0.5
                    val4 = 1.8 + dof_offset * 0.1 + joint_idx * 0.05 + world_idx * 0.5
                    initial_solimplimit[global_dof_idx] = [val0, val1, val2, val3, val4]

        # Assign to model
        model.mujoco.solimplimit.assign(wp.array(initial_solimplimit, dtype=vec5, device=model.device))

        # Step 3: Create solver (it will read the updated values now)
        solver = SolverMuJoCo(model, iterations=1, disable_contacts=True)

        # Step 4: Verify jnt_solimp is properly expanded for multi-world
        jnt_solimp = solver.mjw_model.jnt_solimp.numpy()
        self.assertEqual(jnt_solimp.shape[0], model.world_count, "jnt_solimp should have one entry per world")

        # Step 5: Verify initial values were converted correctly
        # Iterate over MuJoCo joints and verify values match Newton's
        mjc_jnt_to_newton_dof = solver.mjc_jnt_to_newton_dof.numpy()
        nworld_mjc = mjc_jnt_to_newton_dof.shape[0]
        njnt_mjc = mjc_jnt_to_newton_dof.shape[1]

        for world_idx in range(nworld_mjc):
            for mjc_jnt in range(njnt_mjc):
                newton_dof = mjc_jnt_to_newton_dof[world_idx, mjc_jnt]
                if newton_dof < 0:
                    continue  # Skip unmapped joints

                # Get expected solimplimit from Newton model
                expected_solimp = model.mujoco.solimplimit.numpy()[newton_dof, :]

                # Get actual jnt_solimp from MuJoCo
                actual_solimp = jnt_solimp[world_idx, mjc_jnt, :]

                # Verify they match
                np.testing.assert_allclose(
                    actual_solimp,
                    expected_solimp,
                    rtol=1e-5,
                    atol=1e-6,
                    err_msg=f"Initial jnt_solimp[{world_idx}, {mjc_jnt}] doesn't match "
                    f"Newton solimplimit[{newton_dof}]",
                )

        # Step 6: Update solimplimit values with different patterns
        updated_solimplimit = np.zeros((model.joint_dof_count, 5), dtype=np.float32)

        # Iterate over joints and set updated values for each DOF (skip free joints)
        for world_idx in range(model.world_count):
            world_joint_offset = world_idx * joints_per_world

            for joint_idx in range(joints_per_world):
                global_joint_idx = world_joint_offset + joint_idx

                # Skip free joints
                if joint_type[global_joint_idx] == JointType.FREE:
                    continue

                # Get DOF start and count for this joint
                dof_start = joint_qd_start[global_joint_idx]
                dof_count = joint_dof_dim[global_joint_idx].sum()

                # Set updated values for each DOF in this joint
                for dof_offset in range(dof_count):
                    global_dof_idx = dof_start + dof_offset

                    # Updated pattern: different from initial
                    val0 = 0.85 + dof_offset * 0.02 + joint_idx * 0.01 + world_idx * 0.15
                    val1 = 0.88 + dof_offset * 0.02 + joint_idx * 0.01 + world_idx * 0.15
                    val2 = 0.005 + dof_offset * 0.0005 + joint_idx * 0.00025 + world_idx * 0.005
                    val3 = 1.5 + dof_offset * 0.15 + joint_idx * 0.08 + world_idx * 0.6
                    val4 = 2.2 + dof_offset * 0.15 + joint_idx * 0.08 + world_idx * 0.6
                    updated_solimplimit[global_dof_idx] = [val0, val1, val2, val3, val4]

        model.mujoco.solimplimit.assign(wp.array(updated_solimplimit, dtype=vec5, device=model.device))

        # Step 7: Notify solver of changes
        solver.notify_model_changed(ModelFlags.JOINT_DOF_PROPERTIES)

        # Step 8: Verify updated values were converted correctly
        updated_jnt_solimp = solver.mjw_model.jnt_solimp.numpy()

        for world_idx in range(nworld_mjc):
            for mjc_jnt in range(njnt_mjc):
                newton_dof = mjc_jnt_to_newton_dof[world_idx, mjc_jnt]
                if newton_dof < 0:
                    continue  # Skip unmapped joints

                # Get expected solimplimit from updated Newton model
                expected_solimp = model.mujoco.solimplimit.numpy()[newton_dof, :]

                # Get actual jnt_solimp from MuJoCo
                actual_solimp = updated_jnt_solimp[world_idx, mjc_jnt, :]

                # Verify they match
                np.testing.assert_allclose(
                    actual_solimp,
                    expected_solimp,
                    rtol=1e-5,
                    atol=1e-6,
                    err_msg=f"Updated jnt_solimp[{world_idx}, {mjc_jnt}] doesn't match "
                    f"Newton solimplimit[{newton_dof}]",
                )

    def test_limit_margin_runtime_update(self):
        """Test multi-world expansion and runtime updates of limit_margin."""
        # Step 1: Create a template builder and register SolverMuJoCo custom attributes
        template_builder = newton.ModelBuilder()
        newton.solvers.SolverMuJoCo.register_custom_attributes(template_builder)
        shape_cfg = newton.ModelBuilder.ShapeConfig(density=1000.0)

        # Free-floating body
        free_body_initial_pos = wp.transform((0.5, 0.5, 0.0), wp.quat_identity())
        free_body_idx = template_builder.add_body(mass=0.2, xform=free_body_initial_pos)
        template_builder.add_shape_box(body=free_body_idx, xform=wp.transform(), hx=0.1, hy=0.1, hz=0.1, cfg=shape_cfg)

        # Articulated tree
        link_radius = 0.05
        link_half_length = 0.15
        tree_root_initial_pos_y = link_half_length * 2.0
        tree_root_initial_transform = wp.transform((0.0, tree_root_initial_pos_y, 0.0), wp.quat_identity())

        link1_idx = template_builder.add_link(mass=0.1)
        template_builder.add_shape_capsule(
            body=link1_idx, xform=wp.transform(), radius=link_radius, half_height=link_half_length, cfg=shape_cfg
        )
        joint1_idx = template_builder.add_joint_free(child=link1_idx, parent_xform=tree_root_initial_transform)

        link2_idx = template_builder.add_link(mass=0.1)
        template_builder.add_shape_capsule(
            body=link2_idx, xform=wp.transform(), radius=link_radius, half_height=link_half_length, cfg=shape_cfg
        )
        joint2_idx = template_builder.add_joint_revolute(
            parent=link1_idx,
            child=link2_idx,
            axis=(1.0, 0.0, 0.0),
            parent_xform=wp.transform((0.0, link_half_length, 0.0), wp.quat_identity()),
            child_xform=wp.transform((0.0, -link_half_length, 0.0), wp.quat_identity()),
            limit_lower=-np.pi / 2,
            limit_upper=np.pi / 2,
            custom_attributes={"mujoco:limit_margin": [0.01]},
        )

        link3_idx = template_builder.add_link(mass=0.1)
        template_builder.add_shape_capsule(
            body=link3_idx, xform=wp.transform(), radius=link_radius, half_height=link_half_length, cfg=shape_cfg
        )
        joint3_idx = template_builder.add_joint_revolute(
            parent=link2_idx,
            child=link3_idx,
            axis=(0.0, 1.0, 0.0),
            parent_xform=wp.transform((0.0, link_half_length, 0.0), wp.quat_identity()),
            child_xform=wp.transform((0.0, -link_half_length, 0.0), wp.quat_identity()),
            limit_lower=-np.pi / 3,
            limit_upper=np.pi / 3,
            custom_attributes={"mujoco:limit_margin": [0.02]},
        )

        template_builder.add_articulation([joint1_idx, joint2_idx, joint3_idx])

        # Step 2: Replicate to multiple worlds
        world_count = 3
        builder = newton.ModelBuilder()
        builder.replicate(template_builder, world_count)
        model = builder.finalize()

        # Step 3: Initialize solver
        solver = SolverMuJoCo(model, separate_worlds=True, iterations=1, disable_contacts=True)

        # Check solver attribute (jnt_margin)
        jnt_margin = solver.mjw_model.jnt_margin.numpy()

        # Retrieve model info
        joint_qd_start = model.joint_qd_start.numpy()
        joint_dof_dim = model.joint_dof_dim.numpy()
        joint_type = model.joint_type.numpy()

        joints_per_world = model.joint_count // model.world_count

        # Step 4: Verify initial values - iterate over MuJoCo joints
        limit_margin = model.mujoco.limit_margin.numpy()
        mjc_jnt_to_newton_dof = solver.mjc_jnt_to_newton_dof.numpy()
        nworld_mjc = mjc_jnt_to_newton_dof.shape[0]
        njnt_mjc = mjc_jnt_to_newton_dof.shape[1]

        for world_idx in range(nworld_mjc):
            for mjc_jnt in range(njnt_mjc):
                newton_dof = mjc_jnt_to_newton_dof[world_idx, mjc_jnt]
                if newton_dof < 0:
                    continue

                expected_val = limit_margin[newton_dof]
                actual_val = jnt_margin[world_idx, mjc_jnt]
                self.assertAlmostEqual(actual_val, expected_val, places=6)

        # Step 5: Update limit_margin values at runtime
        new_margins = np.zeros_like(limit_margin)

        for world_idx in range(model.world_count):
            world_joint_offset = world_idx * joints_per_world
            for joint_idx in range(joints_per_world):
                global_joint_idx = world_joint_offset + joint_idx

                if joint_type[global_joint_idx] == JointType.FREE:
                    continue

                newton_dof_start = joint_qd_start[global_joint_idx]
                dof_count = int(joint_dof_dim[global_joint_idx].sum())

                for dof_offset in range(dof_count):
                    newton_dof_idx = newton_dof_start + dof_offset
                    val = 0.1 + world_idx * 0.1 + joint_idx * 0.01
                    new_margins[newton_dof_idx] = val

        model.mujoco.limit_margin.assign(new_margins)

        # Step 6: Notify solver
        solver.notify_model_changed(ModelFlags.JOINT_DOF_PROPERTIES)

        # Step 7: Verify updates - iterate over MuJoCo joints
        updated_jnt_margin = solver.mjw_model.jnt_margin.numpy()

        for world_idx in range(nworld_mjc):
            for mjc_jnt in range(njnt_mjc):
                newton_dof = mjc_jnt_to_newton_dof[world_idx, mjc_jnt]
                if newton_dof < 0:
                    continue

                expected_val = new_margins[newton_dof]
                actual_val = updated_jnt_margin[world_idx, mjc_jnt]
                self.assertAlmostEqual(actual_val, expected_val, places=6)

    def test_dof_passive_stiffness_damping_multiworld(self):
        """
        Verify that dof_passive_stiffness and joint_damping propagate correctly:
        1. Different per-world values survive conversion to MuJoCo.
        2. notify_model_changed updates all worlds consistently.
        """

        template_builder = newton.ModelBuilder()
        SolverMuJoCo.register_custom_attributes(template_builder)

        pendulum = template_builder.add_link(
            mass=1.0,
            com=wp.vec3(0.0, 0.0, 0.0),
            inertia=wp.mat33(np.eye(3)),
        )
        template_builder.add_shape_box(
            body=pendulum,
            xform=wp.transform(),
            hx=0.05,
            hy=0.05,
            hz=0.05,
        )
        joint = template_builder.add_joint_revolute(
            parent=-1,
            child=pendulum,
            axis=(0.0, 0.0, 1.0),
            parent_xform=wp.transform(),
            child_xform=wp.transform(),
        )
        template_builder.add_articulation([joint])

        world_count = 3
        builder = newton.ModelBuilder()
        builder.replicate(template_builder, world_count)
        model = builder.finalize()

        dofs_per_world = model.joint_dof_count // model.world_count

        initial_stiffness = np.zeros(model.joint_dof_count, dtype=np.float32)
        initial_damping = np.zeros(model.joint_dof_count, dtype=np.float32)

        for world_idx in range(model.world_count):
            world_dof_offset = world_idx * dofs_per_world
            for dof_idx in range(dofs_per_world):
                global_idx = world_dof_offset + dof_idx
                initial_stiffness[global_idx] = 0.05 + 0.01 * dof_idx + 0.25 * world_idx
                initial_damping[global_idx] = 0.4 + 0.02 * dof_idx + 0.3 * world_idx

        model.mujoco.dof_passive_stiffness.assign(initial_stiffness)
        model.joint_damping.assign(initial_damping)

        solver = SolverMuJoCo(model, iterations=1, disable_contacts=True)

        # Get mappings
        mjc_dof_to_newton_dof = solver.mjc_dof_to_newton_dof.numpy()
        mjc_jnt_to_newton_dof = solver.mjc_jnt_to_newton_dof.numpy()
        nworld_mjc = mjc_dof_to_newton_dof.shape[0]
        nv_mjc = mjc_dof_to_newton_dof.shape[1]
        njnt_mjc = mjc_jnt_to_newton_dof.shape[1]

        def assert_passive_values(expected_stiffness: np.ndarray, expected_damping: np.ndarray):
            dof_damping = solver.mjw_model.dof_damping.numpy()
            jnt_stiffness = solver.mjw_model.jnt_stiffness.numpy()

            # Check DOF damping - iterate over MuJoCo DOFs
            for world_idx in range(nworld_mjc):
                for mjc_dof in range(nv_mjc):
                    newton_dof = mjc_dof_to_newton_dof[world_idx, mjc_dof]
                    if newton_dof < 0:
                        continue
                    self.assertAlmostEqual(
                        dof_damping[world_idx, mjc_dof],
                        expected_damping[newton_dof],
                        places=6,
                        msg=f"dof_damping mismatch for world={world_idx}, mjc_dof={mjc_dof}, newton_dof={newton_dof}",
                    )

            # Check joint stiffness - iterate over MuJoCo joints
            for world_idx in range(nworld_mjc):
                for mjc_jnt in range(njnt_mjc):
                    newton_dof = mjc_jnt_to_newton_dof[world_idx, mjc_jnt]
                    if newton_dof < 0:
                        continue
                    self.assertAlmostEqual(
                        jnt_stiffness[world_idx, mjc_jnt],
                        expected_stiffness[newton_dof],
                        places=6,
                        msg=f"jnt_stiffness mismatch for world={world_idx}, mjc_jnt={mjc_jnt}, newton_dof={newton_dof}",
                    )

        assert_passive_values(initial_stiffness, initial_damping)

        updated_stiffness = initial_stiffness + 0.5 + 0.05 * np.arange(model.joint_dof_count, dtype=np.float32)
        updated_damping = initial_damping + 0.3 + 0.03 * np.arange(model.joint_dof_count, dtype=np.float32)

        model.mujoco.dof_passive_stiffness.assign(updated_stiffness)
        model.joint_damping.assign(updated_damping)
        solver.notify_model_changed(ModelFlags.JOINT_DOF_PROPERTIES)

        assert_passive_values(updated_stiffness, updated_damping)

    def test_joint_friction_parameters_conversion_and_update(self):
        """Convert actuator gravity compensation and update friction solver parameters."""
        template_builder = newton.ModelBuilder()
        SolverMuJoCo.register_custom_attributes(template_builder)
        b1 = template_builder.add_link()
        j1 = template_builder.add_joint_revolute(
            -1, b1, axis=(0, 0, 1), custom_attributes={"mujoco:jnt_actgravcomp": True}
        )
        template_builder.add_shape_box(body=b1, hx=0.1, hy=0.1, hz=0.1)
        b2 = template_builder.add_link()
        j2 = template_builder.add_joint_revolute(
            b1, b2, axis=(1, 0, 0), custom_attributes={"mujoco:jnt_actgravcomp": False}
        )
        template_builder.add_shape_box(body=b2, hx=0.1, hy=0.1, hz=0.1)
        template_builder.add_articulation([j1, j2])
        world_count = 2
        builder = newton.ModelBuilder()
        SolverMuJoCo.register_custom_attributes(builder)
        builder.replicate(template_builder, world_count)
        model = builder.finalize()
        self.assertTrue(hasattr(model, "mujoco"))
        self.assertTrue(hasattr(model.mujoco, "solimpfriction"))
        total_dofs = model.joint_dof_count
        initial_values_solimp = np.zeros((total_dofs, 5), dtype=np.float32)
        for i in range(total_dofs):
            # Unique pattern: [i, i*2, i*3, i*4, i*5] normalized roughly
            initial_values_solimp[i] = [
                0.1 + (i * 0.01) % 0.8,
                0.1 + (i * 0.02) % 0.8,
                0.001 + (i * 0.001) % 0.1,
                0.5 + (i * 0.1) % 0.5,
                1.0 + (i * 0.1) % 2.0,
            ]
        model.mujoco.solimpfriction.assign(wp.array(initial_values_solimp, dtype=vec5, device=model.device))
        self.assertTrue(hasattr(model, "mujoco"))
        self.assertTrue(hasattr(model.mujoco, "solreffriction"))
        total_dofs = model.joint_dof_count
        initial_values_solref = np.zeros((total_dofs, 2), dtype=np.float32)
        for i in range(total_dofs):
            # Unique pattern for 2-element solref
            initial_values_solref[i] = [
                0.01 + (i * 0.005) % 0.05,  # timeconst
                0.5 + (i * 0.1) % 1.5,  # dampratio
            ]
        model.mujoco.solreffriction.assign(initial_values_solref)
        solver = SolverMuJoCo(model)
        np.testing.assert_array_equal(model.mujoco.jnt_actgravcomp.numpy(), [True, False, True, False])
        np.testing.assert_array_equal(solver.mj_model.jnt_actgravcomp, [1, 0])

        with self.subTest(property="solimpfriction", phase="conversion"):
            mjc_dof_to_newton_dof = solver.mjc_dof_to_newton_dof.numpy()
            mjw_dof_solimp = solver.mjw_model.dof_solimp.numpy()
            nv = solver.mj_model.nv  # Number of MuJoCo DOFs

            def check_values_solimp(expected_values, actual_mjw_values, msg_prefix):
                for w in range(world_count):
                    for mjc_dof in range(nv):
                        newton_dof = mjc_dof_to_newton_dof[w, mjc_dof]
                        if newton_dof < 0:
                            continue

                        expected = expected_values[newton_dof]
                        actual = actual_mjw_values[w, mjc_dof]

                        np.testing.assert_allclose(
                            actual,
                            expected,
                            rtol=1e-5,
                            err_msg=f"{msg_prefix} mismatch at World {w}, MuJoCo DOF {mjc_dof}, Newton DOF {newton_dof}",
                        )

            check_values_solimp(initial_values_solimp, mjw_dof_solimp, "Initial conversion")

        with self.subTest(property="solreffriction", phase="conversion"):
            mjc_dof_to_newton_dof = solver.mjc_dof_to_newton_dof.numpy()
            mjw_dof_solref = solver.mjw_model.dof_solref.numpy()
            nv = mjc_dof_to_newton_dof.shape[1]  # Number of MuJoCo DOFs

            def check_values_solref(expected_values, actual_mjw_values, msg_prefix):
                for w in range(world_count):
                    for mjc_dof in range(nv):
                        newton_dof = mjc_dof_to_newton_dof[w, mjc_dof]
                        if newton_dof < 0:
                            continue

                        expected = expected_values[newton_dof]
                        actual = actual_mjw_values[w, mjc_dof]

                        np.testing.assert_allclose(
                            actual,
                            expected,
                            rtol=1e-5,
                            err_msg=f"{msg_prefix} mismatch at World {w}, MuJoCo DOF {mjc_dof}, Newton DOF {newton_dof}",
                        )

            check_values_solref(initial_values_solref, mjw_dof_solref, "Initial conversion")

        with self.subTest(property="solimpfriction", phase="update"):
            updated_values_solimp = np.zeros((total_dofs, 5), dtype=np.float32)
            for i in range(total_dofs):
                updated_values_solimp[i] = [
                    0.8 - (i * 0.01) % 0.8,
                    0.8 - (i * 0.02) % 0.8,
                    0.1 - (i * 0.001) % 0.05,
                    0.9 - (i * 0.1) % 0.5,
                    2.5 - (i * 0.1) % 1.0,
                ]
            model.mujoco.solimpfriction.assign(wp.array(updated_values_solimp, dtype=vec5, device=model.device))
            solver.notify_model_changed(ModelFlags.JOINT_DOF_PROPERTIES)
            mjw_dof_solimp_updated = solver.mjw_model.dof_solimp.numpy()
            check_values_solimp(updated_values_solimp, mjw_dof_solimp_updated, "Runtime update")
            self.assertFalse(
                np.allclose(mjw_dof_solimp_updated[0, 0], initial_values_solimp[0]),
                "Value did not change from initial!",
            )

        with self.subTest(property="solreffriction", phase="update"):
            updated_values_solref = np.zeros((total_dofs, 2), dtype=np.float32)
            for i in range(total_dofs):
                updated_values_solref[i] = [
                    0.05 - (i * 0.005) % 0.04,  # timeconst
                    2.0 - (i * 0.1) % 1.0,  # dampratio
                ]
            model.mujoco.solreffriction.assign(updated_values_solref)
            solver.notify_model_changed(ModelFlags.JOINT_DOF_PROPERTIES)
            mjw_dof_solref_updated = solver.mjw_model.dof_solref.numpy()
            check_values_solref(updated_values_solref, mjw_dof_solref_updated, "Runtime update")
            self.assertFalse(
                np.allclose(mjw_dof_solref_updated[0, 0], initial_values_solref[0]),
                "Value did not change from initial!",
            )


class TestMuJoCoSolverFreeJointBodyPos(unittest.TestCase):
    """Verify free joint bodies preserve their initial position in qpos0."""

    def test_free_joint_body_pos(self):
        """Verify free joint qpos0 contains the MJCF body position.

        A free joint body placed at pos="0 0 1.5" should produce
        ``solver.mj_model.qpos0`` whose first three elements match
        the body's initial z-offset ``[0, 0, 1.5]``.
        """
        mjcf = """
        <mujoco>
            <worldbody>
                <body name="robot" pos="0 0 1.5">
                    <freejoint name="root"/>
                    <geom type="sphere" size="0.1" mass="1.0"/>
                    <inertial pos="0 0 0" mass="1.0" diaginertia="0.01 0.01 0.01"/>
                </body>
            </worldbody>
        </mujoco>
        """
        builder = newton.ModelBuilder()
        builder.add_mjcf(mjcf)
        model = builder.finalize()
        solver = SolverMuJoCo(model)

        # qpos0 should have the body's initial z position
        qpos0 = np.array(solver.mj_model.qpos0)
        np.testing.assert_allclose(
            qpos0[:3],
            [0.0, 0.0, 1.5],
            atol=1e-6,
            err_msg="Free joint qpos0 position should match body pos from MJCF",
        )


class TestMuJoCoSolverFreeJointNonWorldParent(unittest.TestCase):
    """SolverMuJoCo warns when a FREE joint has a non-world parent."""

    def _make_world_parent_free_model(self):
        builder = newton.ModelBuilder()
        body = builder.add_link(mass=1.0, com=wp.vec3(0.0), inertia=wp.mat33(np.eye(3)))
        j = builder.add_joint_free(child=body)
        builder.add_articulation([j])
        return builder.finalize()

    def _make_non_world_parent_free_model(self, label="floater"):
        builder = newton.ModelBuilder()
        parent = builder.add_link(mass=1.0, com=wp.vec3(0.0), inertia=wp.mat33(np.eye(3)))
        child = builder.add_link(mass=1.0, com=wp.vec3(0.0), inertia=wp.mat33(np.eye(3)))
        j_root = builder.add_joint_fixed(parent=-1, child=parent)
        j_free = builder.add_joint_free(parent=parent, child=child, label=label)
        builder.add_articulation([j_root, j_free])
        return builder.finalize()

    def test_free_joint_world_parent_no_warning(self):
        """SolverMuJoCo must not warn when a FREE joint attaches directly to the world."""
        try:
            model = self._make_world_parent_free_model()
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                SolverMuJoCo(model)
        except ImportError as e:
            self.skipTest(f"MuJoCo or deps not installed: {e}")
        self.assertFalse(
            any("SolverMuJoCo" in str(w.message) for w in caught),
            f"Unexpected SolverMuJoCo warning for world-parent free joint: {[str(w.message) for w in caught]}",
        )

    def test_free_joint_non_world_parent_warns(self):
        """SolverMuJoCo must emit a UserWarning when a FREE joint has a non-world parent.

        MuJoCo only allows free joints directly under the worldbody; a free joint
        parented to another body will not simulate correctly. MuJoCo also raises a
        ValueError on compile, but the warning must be emitted before that.
        """
        try:
            model = self._make_non_world_parent_free_model(label="floater")
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                try:
                    SolverMuJoCo(model)
                except ValueError:
                    pass  # MuJoCo rejects free joints not at the top level after warning
        except ImportError as e:
            self.skipTest(f"MuJoCo or deps not installed: {e}")
        mjc_warnings = [w for w in caught if "SolverMuJoCo" in str(w.message)]
        self.assertEqual(len(mjc_warnings), 1)
        self.assertEqual(mjc_warnings[0].category, UserWarning)
        self.assertIn("floater", str(mjc_warnings[0].message))
