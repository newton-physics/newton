# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""
Defines the Forward Kinematics solver class.

See the :mod:`newton._src.solvers.kamino._src.solvers.fk` module for a detailed description.
"""

from __future__ import annotations

import sys

import numpy as np
import warp as wp

from ....config import ForwardKinematicsSolverConfig
from ...core.joints import JointActuationType, JointDoFType
from ...core.model import ModelKamino
from ...core.types import vec6f
from ...linalg.blas import (
    block_sparse_ATA_blockwise_3_4_inv_diagonal_2d,
    block_sparse_ATA_inv_diagonal_2d,
    get_blockwise_diag_3_4_gemv_2d,
)
from ...linalg.conjugate import BatchedLinearOperator, CGSolver
from ...linalg.factorize.llt_blocked_semi_sparse import SemiSparseBlockCholeskySolverBatched
from ...linalg.sparse_matrix import BlockDType, BlockSparseMatrices
from ...linalg.sparse_operator import BlockSparseLinearOperators
from .kernels import (
    _apply_line_search_step,
    _eval_body_velocities,
    _eval_fk_actuated_dofs_or_coords,
    _eval_linear_combination,
    _eval_position_control_transformations,
    _eval_rhs,
    _eval_stepped_state,
    _eval_target_constraint_velocities,
    _eval_unit_quaternion_constraints,
    _eval_unit_quaternion_constraints_jacobian,
    _eval_unit_quaternion_constraints_sparse_jacobian,
    _line_search_check,
    _newton_check,
    _reset_state,
    _reset_state_base_q,
    _update_cg_tolerance_kernel,
    create_eval_joint_constraints_jacobian_kernel,
    create_eval_joint_constraints_kernel,
    create_eval_joint_constraints_sparse_jacobian_kernel,
    create_tile_based_kernels,
)
from .types import FKJointDoFType, ForwardKinematicsPreconditionerType, ForwardKinematicsStatus

###
# Module interface
###

__all__ = ["ForwardKinematicsSolver"]


###
# Module configs
###

wp.set_module_options({"enable_backward": False})


###
# Interfaces
###


class ForwardKinematicsSolver:
    """
    Forward Kinematics solver class
    """

    PreconditionerType = ForwardKinematicsPreconditionerType
    """Type alias of the FK solver preconditioning options."""

    Config = ForwardKinematicsSolverConfig
    """
    Defines a type alias of the FK solver configurations container, including convergence
    criteria, maximum iterations, and options for the linear solver and preconditioning.

    See :class:`ForwardKinematicsSolverConfig` for the full
    list of configuration options and their descriptions.
    """

    Status = ForwardKinematicsStatus
    """Type alias of the FK solver status."""

    def __init__(self, model: ModelKamino | None = None, config: ForwardKinematicsSolver.Config | None = None):
        """
        Initializes the solver to solve forward kinematics for a given model.

        Parameters
        ----------
        model : ModelKamino, optional
            ModelKamino for which to solve forward kinematics. If not provided, the finalize() method
            must be called at a later time for deferred initialization (default: None).
        config : ForwardKinematicsSolver.Config, optional
            Solver config. If not provided, the default config will be used (default: None).
        """

        self.model: ModelKamino | None = None
        """Underlying model"""

        self.device: wp.DeviceLike = None
        """Device for data allocations"""

        self.config: ForwardKinematicsSolver.Config = ForwardKinematicsSolver.Config()
        """Solver config"""

        self.graph: wp.Graph | None = None
        """Cuda graph for the convenience function with verbosity options"""

        # Note: there are many other internal data members below, which are not documented here

        # Set model and config, and finalize if model was provided
        self.model = model
        if config is not None:
            self.config = config
        if model is not None:
            self.finalize()

    def finalize(self, model: ModelKamino | None = None, config: ForwardKinematicsSolver.Config | None = None):
        """
        Finishes the solver initialization, performing necessary allocations and precomputations.
        This method only needs to be called manually if a model was not provided in the constructor,
        or to reset the solver for a new model.

        Parameters
        ----------
        model : ModelKamino, optional
            ModelKamino for which to solve forward kinematics. If not provided, the model given to the
            constructor will be used. Must be provided if not given to the constructor (default: None).
        config : ForwardKinematicsSolver.Config, optional
            Solver config. If not provided, the config given to the constructor, or if not, the
            default config will be used (default: None).
        """

        # Initialize the model and config if provided
        if model is not None:
            self.model = model
        if config is not None:
            self.config = config
        if self.model is None:
            raise ValueError("ForwardKinematicsSolver: error, provided model is None.")

        # Initialize device
        self.device = self.model.device

        # Retrieve / compute dimensions - Worlds
        self.num_worlds = self.model.size.num_worlds  # For convenience

        # Convert preconditioner type
        self._preconditioner_type = ForwardKinematicsSolver.PreconditionerType.from_string(self.config.preconditioner)

        # Retrieve / compute dimensions - Bodies
        num_bodies = self.model.info.num_bodies.numpy()  # Number of bodies per world
        first_body_id = np.concatenate(([0], num_bodies.cumsum()))  # Index of first body per world
        self.num_bodies_max = self.model.size.max_of_num_bodies  # Max number of bodies across worlds

        # Retrieve / compute dimensions - States (i.e., body poses)
        num_states = 7 * num_bodies  # Number of state dimensions per world
        self.num_states_tot = 7 * self.model.size.sum_of_num_bodies  # State dimensions for the whole model
        self.num_states_max = 7 * self.num_bodies_max  # Max state dimension across worlds

        # Retrieve / compute dimensions - Joints (main model)
        num_joints_prev = self.model.info.num_joints.numpy().copy()  # Number of joints per world
        first_joint_id_prev = np.concatenate(([0], num_joints_prev.cumsum()))  # Index of first joint per world

        # Retrieve / compute dimensions - Actuated coordinates/dofs (main model)
        actuated_coord_offsets_prev = self.model.joints.actuated_coords_offset.numpy().copy()
        actuated_dof_offsets_prev = self.model.joints.actuated_dofs_offset.numpy().copy()

        # Create a copy of the model's joints with added joints as needed:
        # - actuated free joints to reset the base position/orientation
        # - axis joints to factor out superfluous DoFs at tie rods
        joints_dof_type_prev = self.model.joints.dof_type.numpy().copy()
        joints_act_type_prev = self.model.joints.act_type.numpy().copy()
        joints_bid_B_prev = self.model.joints.bid_B.numpy().copy()
        joints_bid_F_prev = self.model.joints.bid_F.numpy().copy()
        joints_B_r_Bj_prev = self.model.joints.B_r_Bj.numpy().copy()
        joints_F_r_Fj_prev = self.model.joints.F_r_Fj.numpy().copy()
        joints_X_j_prev = self.model.joints.X_j.numpy().copy()
        joints_num_coords_prev = self.model.joints.num_coords.numpy().copy()
        joints_num_dofs_prev = self.model.joints.num_dofs.numpy().copy()
        joints_dof_type = []
        joints_act_type = []
        joints_bid_B = []
        joints_bid_F = []
        joints_B_r_Bj = []
        joints_F_r_Fj = []
        joints_X_j = []
        joints_num_actuated_coords = []  # Number of actuated coordinates per joint (0 for passive joints)
        joints_num_actuated_dofs = []  # Number of actuated dofs per joint (0 for passive joints)
        num_joints = np.zeros(self.num_worlds, dtype=np.int32)  # Number of joints per world
        self.num_joints_tot = 0  # Number of joints for all worlds
        actuated_coords_map = []  # Map of new actuated coordinates to these in the model or to the base coordinates
        actuated_dofs_map = []  # Map of new actuated dofs to these in the model or to the base dofs
        base_q_default = np.zeros(7 * self.num_worlds, dtype=np.float32)  # Default base pose
        bodies_q_0 = self.model.bodies.q_i_0.numpy()
        base_joint_ids = self.num_worlds * [-1]  # Base joint id per world
        base_joint_ids_input = self.model.info.base_joint_index.numpy().tolist()
        base_body_ids_input = self.model.info.base_body_index.numpy().tolist()
        for wd_id in range(self.num_worlds):
            # Retrieve base joint id
            base_joint_id = base_joint_ids_input[wd_id]

            # Copy data for all kept joints
            world_joint_ids = [
                i for i in range(first_joint_id_prev[wd_id], first_joint_id_prev[wd_id + 1]) if i != base_joint_id
            ]
            for jt_id_prev in world_joint_ids:
                # Note: we use the fact that integer values of the FK vs Kamino dof type enums
                # are matched for all joints that are not FK-specific
                joints_dof_type.append(joints_dof_type_prev[jt_id_prev])
                joints_act_type.append(joints_act_type_prev[jt_id_prev])
                joints_bid_B.append(joints_bid_B_prev[jt_id_prev])
                joints_bid_F.append(joints_bid_F_prev[jt_id_prev])
                joints_B_r_Bj.append(joints_B_r_Bj_prev[jt_id_prev])
                joints_F_r_Fj.append(joints_F_r_Fj_prev[jt_id_prev])
                joints_X_j.append(joints_X_j_prev[jt_id_prev])
                if joints_act_type[-1] != JointActuationType.PASSIVE:
                    num_coords_jt = joints_num_coords_prev[jt_id_prev]
                    joints_num_actuated_coords.append(num_coords_jt)
                    coord_offset = actuated_coord_offsets_prev[jt_id_prev]
                    actuated_coords_map.extend(range(coord_offset, coord_offset + num_coords_jt))

                    num_dofs_jt = joints_num_dofs_prev[jt_id_prev]
                    joints_num_actuated_dofs.append(num_dofs_jt)
                    dof_offset = actuated_dof_offsets_prev[jt_id_prev]
                    actuated_dofs_map.extend(range(dof_offset, dof_offset + num_dofs_jt))
                else:
                    joints_num_actuated_coords.append(0)
                    joints_num_actuated_dofs.append(0)

            # Add axis joints as needed
            if self.config.add_axis_joints:
                # Find all bodies incident to two spherical joints (and nothing more)
                num_joints_per_body = np.zeros(dtype=np.int32, shape=num_bodies[wd_id])
                spherical_joints_per_body = [[] for i in range(num_bodies[wd_id])]
                for jt_id_prev in world_joint_ids:
                    is_spherical = joints_dof_type_prev[jt_id_prev] == JointDoFType.SPHERICAL
                    bid_B = joints_bid_B_prev[jt_id_prev]
                    if bid_B >= 0:
                        bid_B -= first_body_id[wd_id]
                        num_joints_per_body[bid_B] += 1
                        if is_spherical:
                            spherical_joints_per_body[bid_B].append(jt_id_prev)
                    bid_F = joints_bid_F_prev[jt_id_prev] - first_body_id[wd_id]
                    num_joints_per_body[bid_F] += 1
                    if is_spherical:
                        spherical_joints_per_body[bid_F].append(jt_id_prev)

                # Add an axis joint for each such body
                for rb_id in range(num_bodies[wd_id]):
                    if num_joints_per_body[rb_id] != 2 or len(spherical_joints_per_body[rb_id]) != 2:
                        continue
                    rb_id_tot = first_body_id[wd_id] + rb_id
                    joints_dof_type.append(FKJointDoFType.AXIS)
                    joints_act_type.append(JointActuationType.PASSIVE)
                    joints_bid_B.append(-1)
                    joints_bid_F.append(rb_id_tot)
                    joints_B_r_Bj.append(np.zeros(dtype=np.float32, shape=3))
                    joints_F_r_Fj.append(np.zeros(dtype=np.float32, shape=3))
                    joints_num_actuated_coords.append(0)
                    joints_num_actuated_dofs.append(0)

                    # Compute position of both spherical joints on initial pose
                    def eval_joint_pos_init(jt_id_prev):
                        bid_B = joints_bid_B_prev[jt_id_prev]
                        bid_F = joints_bid_F_prev[jt_id_prev]
                        if bid_B == rb_id_tot:  # Body is the joint's base  # noqa: B023
                            q_B = bodies_q_0[bid_B]
                            B_r_B = joints_B_r_Bj_prev[jt_id_prev]
                            return q_B[:3] + np.array(wp.quat_rotate(wp.quat(q_B[3:]), wp.vec3f(B_r_B)))
                        else:  # Body is the joint's follower
                            assert bid_F == rb_id_tot  # noqa: B023
                            q_F = bodies_q_0[bid_F]
                            F_r_F = joints_F_r_Fj_prev[jt_id_prev]
                            return q_F[:3] + np.array(wp.quat_rotate(wp.quat(q_F[3:]), wp.vec3f(F_r_F)))

                    pos_0 = eval_joint_pos_init(spherical_joints_per_body[rb_id][0])
                    pos_1 = eval_joint_pos_init(spherical_joints_per_body[rb_id][1])

                    # Joint frame: set X axis that connects both spherical joints (= tie rod axis)
                    a_x = pos_1 - pos_0
                    a_x /= np.linalg.norm(a_x)
                    if np.abs(a_x[2]) < 0.99:
                        a_y = np.cross(np.array([0.0, 0.0, 1.0]), a_x)
                    else:
                        a_y = np.cross(np.array([0.0, 1.0, 0.0]), a_x)
                    a_y /= np.linalg.norm(a_y)
                    a_z = np.cross(a_x, a_y)
                    a_z /= np.linalg.norm(a_z)
                    joints_X_j.append(np.stack((a_x, a_y, a_z), axis=1))

            # Add joint for base joint / base body
            if base_joint_id >= 0:  # Replace base joint with an actuated free joint
                joints_dof_type.append(JointDoFType.FREE)
                joints_act_type.append(JointActuationType.FORCE)
                joints_bid_B.append(-1)
                joints_bid_F.append(joints_bid_F_prev[base_joint_id])
                joints_B_r_Bj.append(joints_B_r_Bj_prev[base_joint_id])
                joints_F_r_Fj.append(joints_F_r_Fj_prev[base_joint_id])
                joints_X_j.append(joints_X_j_prev[base_joint_id])
                joints_num_actuated_coords.append(7)
                coord_offset = -7 * wd_id - 1  # We encode offsets in base_q negatively with i -> -i - 1
                actuated_coords_map.extend(range(coord_offset, coord_offset - 7, -1))
                base_q_default[7 * wd_id : 7 * wd_id + 7] = [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                ]  # Default to zero of free joint
                joints_num_actuated_dofs.append(6)
                dof_offset = -6 * wd_id - 1  # We encode offsets in base_u negatively with i -> -i - 1
                actuated_dofs_map.extend(range(dof_offset, dof_offset - 6, -1))
                base_joint_ids[wd_id] = len(joints_dof_type) - 1
            elif base_body_ids_input[wd_id] >= 0:  # Add an actuated free joint to the base body
                base_body_id = base_body_ids_input[wd_id]
                joints_dof_type.append(JointDoFType.FREE)
                joints_act_type.append(JointActuationType.FORCE)
                joints_bid_B.append(-1)
                joints_bid_F.append(base_body_id)
                joints_B_r_Bj.append(np.zeros(3, dtype=np.float32))
                joints_F_r_Fj.append(np.zeros(3, dtype=np.float32))
                joints_X_j.append(np.eye(3, 3, dtype=np.float32))
                joints_num_actuated_coords.append(7)
                # Note: we rely on the initial body orientations being identity
                # Only then will the corresponding joint coordinates be interpretable as
                # specifying the absolute base position and orientation
                coord_offset = -7 * wd_id - 1  # We encode offsets in base_q negatively with i -> -i - 1
                actuated_coords_map.extend(range(coord_offset, coord_offset - 7, -1))
                base_q_default[7 * wd_id : 7 * wd_id + 7] = bodies_q_0[base_body_id]  # Default to initial body pose
                joints_num_actuated_dofs.append(6)
                dof_offset = -6 * wd_id - 1  # We encode offsets in base_u negatively with i -> -i - 1
                actuated_dofs_map.extend(range(dof_offset, dof_offset - 6, -1))
                base_joint_ids[wd_id] = len(joints_dof_type) - 1

            # Record number of joints
            num_joints_world = len(joints_dof_type) - self.num_joints_tot
            self.num_joints_tot += num_joints_world
            num_joints[wd_id] = num_joints_world

        # Retrieve / compute dimensions - Joints (FK model)
        first_joint_id = np.concatenate(([0], num_joints.cumsum()))  # Index of first joint per world
        self.num_joints_max = max(num_joints)  # Max number of joints across worlds

        # Retrieve / compute dimensions - Actuated coordinates (FK model)
        joints_num_actuated_coords = np.array(joints_num_actuated_coords)  # Number of actuated coordinates per joint
        actuated_coord_offsets = np.concatenate(
            ([0], joints_num_actuated_coords.cumsum())
        )  # First actuated coordinate offset per joint, among all actuated coordinates
        self.num_actuated_coords = actuated_coord_offsets[-1]

        # Retrieve / compute dimensions - Actuated dofs (FK model)
        joints_num_actuated_dofs = np.array(joints_num_actuated_dofs)  # Number of actuated dofs per joint
        actuated_dof_offsets = np.concatenate(
            ([0], joints_num_actuated_dofs.cumsum())
        )  # First actuated dof offset per joint, among all actuated dofs
        self.num_actuated_dofs = actuated_dof_offsets[-1]

        # Retrieve / compute dimensions - Constraints
        num_constraints = num_bodies.copy()  # Number of kinematic constraints per world (unit quat. + joints)
        has_universal_joints = False  # Whether the model has a least one passive universal joint
        constraint_full_to_red_map = np.full(6 * self.num_joints_tot, -1, dtype=np.int32)
        for wd_id in range(self.num_worlds):
            ct_count = num_constraints[wd_id]
            for jt_id_loc in range(num_joints[wd_id]):
                jt_id_tot = first_joint_id[wd_id] + jt_id_loc  # Joint id among all joints
                act_type = joints_act_type[jt_id_tot]
                if act_type != JointActuationType.PASSIVE:  # Actuator: select all six constraints
                    for i in range(6):
                        constraint_full_to_red_map[6 * jt_id_tot + i] = ct_count + i
                    ct_count += 6
                else:
                    dof_type = joints_dof_type[jt_id_tot]
                    if dof_type == FKJointDoFType.AXIS:
                        constraint_full_to_red_map[6 * jt_id_tot + 3] = ct_count
                        ct_count += 1
                    elif dof_type == FKJointDoFType.CARTESIAN:
                        for i in range(3):
                            constraint_full_to_red_map[6 * jt_id_tot + 3 + i] = ct_count + i
                        ct_count += 3
                    elif dof_type == FKJointDoFType.CYLINDRICAL:
                        constraint_full_to_red_map[6 * jt_id_tot + 1] = ct_count
                        constraint_full_to_red_map[6 * jt_id_tot + 2] = ct_count + 1
                        constraint_full_to_red_map[6 * jt_id_tot + 4] = ct_count + 2
                        constraint_full_to_red_map[6 * jt_id_tot + 5] = ct_count + 3
                        ct_count += 4
                    elif dof_type == FKJointDoFType.FIXED:
                        for i in range(6):
                            constraint_full_to_red_map[6 * jt_id_tot + i] = ct_count + i
                        ct_count += 6
                    elif dof_type == FKJointDoFType.FREE:
                        pass
                    elif dof_type == FKJointDoFType.PRISMATIC:
                        constraint_full_to_red_map[6 * jt_id_tot + 1] = ct_count
                        constraint_full_to_red_map[6 * jt_id_tot + 2] = ct_count + 1
                        for i in range(3):
                            constraint_full_to_red_map[6 * jt_id_tot + 3 + i] = ct_count + 2 + i
                        ct_count += 5
                    elif dof_type == FKJointDoFType.REVOLUTE:
                        for i in range(3):
                            constraint_full_to_red_map[6 * jt_id_tot + i] = ct_count + i
                        constraint_full_to_red_map[6 * jt_id_tot + 4] = ct_count + 3
                        constraint_full_to_red_map[6 * jt_id_tot + 5] = ct_count + 4
                        ct_count += 5
                    elif dof_type == FKJointDoFType.SPHERICAL:
                        for i in range(3):
                            constraint_full_to_red_map[6 * jt_id_tot + i] = ct_count + i
                        ct_count += 3
                    elif dof_type == FKJointDoFType.UNIVERSAL:
                        for i in range(3):
                            constraint_full_to_red_map[6 * jt_id_tot + i] = ct_count + i
                        constraint_full_to_red_map[6 * jt_id_tot + 5] = ct_count + 3
                        ct_count += 4
                        has_universal_joints = True
                    else:
                        raise RuntimeError("Unknown joint dof type")
            num_constraints[wd_id] = ct_count
        self.num_constraints_max = np.max(num_constraints)

        # Retrieve / compute dimensions - Number of tiles (for kernels using Tile API)
        self.num_tiles_constraints = (
            self.num_constraints_max + self.config.TILE_SIZE_CTS - 1
        ) // self.config.TILE_SIZE_CTS
        self.num_tiles_states = (self.num_states_max + self.config.TILE_SIZE_VRS - 1) // self.config.TILE_SIZE_VRS

        # Data allocation or transfer from numpy to warp
        with wp.ScopedDevice(self.device):
            # Dimensions
            self.first_body_id = wp.from_numpy(first_body_id, dtype=wp.int32)
            self.num_joints = wp.from_numpy(num_joints, dtype=wp.int32)
            self.first_joint_id = wp.from_numpy(first_joint_id, dtype=wp.int32)
            self.actuated_coord_offsets = wp.from_numpy(actuated_coord_offsets, dtype=wp.int32)
            self.actuated_coords_map = wp.from_numpy(np.array(actuated_coords_map), dtype=wp.int32)
            self.actuated_dof_offsets = wp.from_numpy(actuated_dof_offsets, dtype=wp.int32)
            self.actuated_dofs_map = wp.from_numpy(np.array(actuated_dofs_map), dtype=wp.int32)
            self.num_states = wp.from_numpy(num_states, dtype=wp.int32)
            self.num_constraints = wp.from_numpy(num_constraints, dtype=wp.int32)
            self.constraint_full_to_red_map = wp.from_numpy(constraint_full_to_red_map, dtype=wp.int32)

            # Modified joints
            self.joints_dof_type = wp.from_numpy(joints_dof_type, dtype=wp.int32)
            self.joints_act_type = wp.from_numpy(joints_act_type, dtype=wp.int32)
            self.joints_bid_B = wp.from_numpy(joints_bid_B, dtype=wp.int32)
            self.joints_bid_F = wp.from_numpy(joints_bid_F, dtype=wp.int32)
            self.joints_B_r_Bj = wp.from_numpy(joints_B_r_Bj, dtype=wp.vec3f)
            self.joints_F_r_Fj = wp.from_numpy(joints_F_r_Fj, dtype=wp.vec3f)
            self.joints_X_j = wp.from_numpy(joints_X_j, dtype=wp.mat33f)
            self.base_joint_id = wp.from_numpy(base_joint_ids, dtype=wp.int32)

            # Default base state
            self.base_q_default = wp.from_numpy(base_q_default, dtype=wp.transformf)
            self.base_u_default = wp.zeros(shape=(self.num_worlds,), dtype=vec6f)

            # Line search
            self.max_line_search_iterations = wp.array(dtype=wp.int32, shape=(1,))  # Max iterations
            self.max_line_search_iterations.fill_(self.config.max_line_search_iterations)
            self.line_search_iteration = wp.array(dtype=wp.int32, shape=(self.num_worlds,))  # Iteration count
            self.line_search_loop_condition = wp.array(dtype=wp.int32, shape=(1,))  # Loop condition
            self.line_search_success = wp.array(dtype=wp.int32, shape=(self.num_worlds,))  # Convergence, per world
            self.line_search_mask = wp.array(
                dtype=wp.int32, shape=(self.num_worlds,)
            )  # Flag to keep iterating per world
            self.val_0 = wp.array(dtype=wp.float32, shape=(self.num_worlds,))  # Merit function value at 0, per world
            self.grad_0 = wp.array(
                dtype=wp.float32, shape=(self.num_worlds,)
            )  # Merit function gradient at 0, per world
            self.alpha = wp.array(dtype=wp.float32, shape=(self.num_worlds,))  # Step size, per world
            self.bodies_q_alpha = wp.array(dtype=wp.transformf, shape=(self.model.size.sum_of_num_bodies,))  # New state
            self.val_alpha = wp.array(dtype=wp.float32, shape=(self.num_worlds,))  # New merit function value, per world

            # Gauss-Newton
            self.max_newton_iterations = wp.array(dtype=wp.int32, shape=(1,))  # Max iterations
            self.max_newton_iterations.fill_(self.config.max_newton_iterations)
            self.newton_iteration = wp.array(dtype=wp.int32, shape=(self.num_worlds,))  # Iteration count
            self.newton_loop_condition = wp.array(dtype=wp.int32, shape=(1,))  # Loop condition
            self.newton_success = wp.array(dtype=wp.int32, shape=(self.num_worlds,))  # Convergence per world
            self.newton_mask = wp.array(dtype=wp.int32, shape=(self.num_worlds,))  # Flag to keep iterating per world
            self.tolerance = wp.array(dtype=wp.float32, shape=(1,))  # Tolerance on max constraint
            self.tolerance.fill_(self.config.tolerance)
            self.actuators_q = wp.array(dtype=wp.float32, shape=(self.num_actuated_coords,))  # Actuated coordinates
            self.pos_control_transforms = wp.array(
                dtype=wp.transformf, shape=(self.num_joints_tot,)
            )  # Position-control transformations at joints
            self.constraints = wp.zeros(
                dtype=wp.float32,
                shape=(
                    self.num_worlds,
                    self.num_constraints_max,
                ),
            )  # Constraints vector per world
            self.max_constraint = wp.array(dtype=wp.float32, shape=(self.num_worlds,))  # Maximal constraint per world
            self.jacobian = wp.zeros(
                dtype=wp.float32, shape=(self.num_worlds, self.num_constraints_max, self.num_states_max)
            )  # Constraints Jacobian per world
            if not self.config.use_sparsity:
                self.lhs = wp.zeros(
                    dtype=wp.float32, shape=(self.num_worlds, self.num_states_max, self.num_states_max)
                )  # Gauss-Newton left-hand side per world
            self.grad = wp.zeros(
                dtype=wp.float32, shape=(self.num_worlds, self.num_states_max)
            )  # Merit function gradient w.r.t. state per world
            self.rhs = wp.zeros(
                dtype=wp.float32, shape=(self.num_worlds, self.num_states_max)
            )  # Gauss-Newton right-hand side per world (=-grad)
            self.step = wp.zeros(
                dtype=wp.float32, shape=(self.num_worlds, self.num_states_max)
            )  # Step in state variables per world
            self.jacobian_times_vector = wp.zeros(
                dtype=wp.float32, shape=(self.num_worlds, self.num_constraints_max)
            )  # Intermediary vector when computing J^T * (J * x)
            self.lhs_times_vector = wp.zeros(
                dtype=wp.float32, shape=(self.num_worlds, self.num_states_max)
            )  # Intermediary vector when computing J^T * (J * x)

            # Velocity solver
            self.actuators_u = wp.array(
                dtype=wp.float32, shape=(self.num_actuated_dofs,)
            )  # Velocities for actuated dofs of fk model
            self.target_cts_u = wp.zeros(
                dtype=wp.float32,
                shape=(
                    self.num_worlds,
                    self.num_constraints_max,
                ),
            )  # Target velocity per constraint
            self.bodies_q_dot = wp.array(
                ptr=self.step.ptr, dtype=wp.float32, shape=(self.num_worlds, self.num_states_max), copy=False
            )  # Time derivative of body poses (alias of self.step for data re-use)
            # Note: we also re-use self.jacobian, self.lhs and self.rhs for the velocity solver

        # Initialize kernels that depend on static values
        self._eval_joint_constraints_kernel = create_eval_joint_constraints_kernel(has_universal_joints)
        self._eval_joint_constraints_jacobian_kernel = create_eval_joint_constraints_jacobian_kernel(
            has_universal_joints
        )
        (
            self._eval_pattern_T_pattern_kernel,
            self._eval_max_constraint_kernel,
            self._eval_jacobian_T_jacobian_kernel,
            self._eval_jacobian_T_constraints_kernel,
            self._eval_merit_function_kernel,
            self._eval_merit_function_gradient_kernel,
        ) = create_tile_based_kernels(self.config.TILE_SIZE_CTS, self.config.TILE_SIZE_VRS)

        # Compute sparsity pattern and initialize linear solver for dense (semi-sparse) case
        if not self.config.use_sparsity:
            # Jacobian sparsity pattern
            sparsity_pattern = np.zeros((self.num_worlds, self.num_constraints_max, self.num_states_max), dtype=int)
            for wd_id in range(self.num_worlds):
                for rb_id_loc in range(num_bodies[wd_id]):
                    sparsity_pattern[wd_id, rb_id_loc, 7 * rb_id_loc + 3 : 7 * rb_id_loc + 7] = 1
                for jt_id_loc in range(num_joints[wd_id]):
                    jt_id_tot = first_joint_id[wd_id] + jt_id_loc
                    base_id_tot = joints_bid_B[jt_id_tot]
                    follower_id_tot = joints_bid_F[jt_id_tot]
                    rb_ids_tot = [base_id_tot, follower_id_tot] if base_id_tot >= 0 else [follower_id_tot]
                    for rb_id_tot in rb_ids_tot:
                        rb_id_loc = rb_id_tot - first_body_id[wd_id]
                        state_offset = 7 * rb_id_loc
                        for i in range(3):
                            ct_offset = constraint_full_to_red_map[6 * jt_id_tot + i]  # ith translation constraint
                            if ct_offset >= 0:
                                sparsity_pattern[wd_id, ct_offset, state_offset : state_offset + 7] = 1
                            ct_offset = constraint_full_to_red_map[6 * jt_id_tot + 3 + i]  # ith rotation constraint
                            if ct_offset >= 0:
                                sparsity_pattern[wd_id, ct_offset, state_offset + 3 : state_offset + 7] = 1

            # Jacobian^T * Jacobian sparsity pattern
            sparsity_pattern_wp = wp.from_numpy(sparsity_pattern, dtype=wp.float32, device=self.device)
            sparsity_pattern_lhs_wp = wp.zeros(
                dtype=wp.float32, shape=(self.num_worlds, self.num_states_max, self.num_states_max), device=self.device
            )
            wp.launch_tiled(
                self._eval_pattern_T_pattern_kernel,
                dim=(self.num_worlds, self.num_tiles_states, self.num_tiles_states),
                inputs=[sparsity_pattern_wp, sparsity_pattern_lhs_wp],
                block_dim=64,
                device=self.device,
            )
            sparsity_pattern_lhs = sparsity_pattern_lhs_wp.numpy().astype("int32")

            # Initialize linear solver (semi-sparse LLT)
            self.linear_solver_llt = SemiSparseBlockCholeskySolverBatched(
                self.num_worlds,
                self.num_states_max,
                block_size=16,  # TODO: optimize this (e.g. 14 ?)
                device=self.device,
                enable_reordering=True,
            )
            self.linear_solver_llt.capture_sparsity_pattern(sparsity_pattern_lhs, num_states)

        # Compute sparsity pattern and initialize linear solver for sparse case
        if self.config.use_sparsity:
            self.sparse_jacobian = BlockSparseMatrices(
                device=self.device, nzb_dtype=BlockDType(dtype=wp.float32, shape=(7,)), num_matrices=self.num_worlds
            )
            jacobian_dims = list(zip(num_constraints.tolist(), (7 * num_bodies).tolist(), strict=True))

            # Determine number of nzb, per world and in total
            num_nzb = num_bodies.copy()  # nzb due to rigid body unit quaternion constraints
            jt_num_constraints = (constraint_full_to_red_map.reshape((-1, 6)) >= 0).sum(axis=1)
            jt_num_bodies = np.array([1 if joints_bid_B[i] < 0 else 2 for i in range(self.num_joints_tot)])
            for wd_id in range(self.num_worlds):  # nzb due to joint constraints
                start = first_joint_id[wd_id]
                end = start + num_joints[wd_id]
                num_nzb[wd_id] += (jt_num_constraints[start:end] * jt_num_bodies[start:end]).sum()
            first_nzb = np.concatenate(([0], num_nzb.cumsum()))
            num_nzb_tot = num_nzb.sum()

            # Symbolic assembly
            nzb_row = np.empty(num_nzb_tot, dtype=np.int32)
            nzb_col = np.empty(num_nzb_tot, dtype=np.int32)
            rb_nzb_id = np.empty(self.model.size.sum_of_num_bodies, dtype=np.int32)
            ct_nzb_id_base = np.full(6 * self.num_joints_tot, -1, dtype=np.int32)
            ct_nzb_id_follower = np.full(6 * self.num_joints_tot, -1, dtype=np.int32)
            for wd_id in range(self.num_worlds):
                start_nzb = first_nzb[wd_id]

                # Compute index, row and column of rigid body nzb
                start_rb = first_body_id[wd_id]
                size_rb = num_bodies[wd_id]
                rb_ids = np.arange(size_rb)
                rb_nzb_id[start_rb : start_rb + size_rb] = start_nzb + rb_ids
                nzb_row[start_nzb : start_nzb + size_rb] = rb_ids
                nzb_col[start_nzb : start_nzb + size_rb] = 7 * rb_ids

                # Compute index, row and column of constraint nzb
                start_nzb += size_rb
                for jt_id_loc in range(num_joints[wd_id]):
                    jt_id_tot = jt_id_loc + first_joint_id[wd_id]
                    has_base = joints_bid_B[jt_id_tot] >= 0
                    row_ids_full = constraint_full_to_red_map[6 * jt_id_tot : 6 * jt_id_tot + 6]
                    row_ids_red = [i for i in row_ids_full if i >= 0]
                    num_cts = len(row_ids_red)
                    if has_base:
                        nzb_id_base = ct_nzb_id_base[6 * jt_id_tot : 6 * jt_id_tot + 6]
                        nzb_id_base[row_ids_full >= 0] = np.arange(start_nzb, start_nzb + num_cts)
                        nzb_row[start_nzb : start_nzb + num_cts] = row_ids_red
                        base_id_loc = joints_bid_B[jt_id_tot] - first_body_id[wd_id]
                        nzb_col[start_nzb : start_nzb + num_cts] = 7 * base_id_loc
                        start_nzb += num_cts
                    nzb_id_follower = ct_nzb_id_follower[6 * jt_id_tot : 6 * jt_id_tot + 6]
                    nzb_id_follower[row_ids_full >= 0] = np.arange(start_nzb, start_nzb + num_cts)
                    nzb_row[start_nzb : start_nzb + num_cts] = row_ids_red
                    follower_id_loc = joints_bid_F[jt_id_tot] - first_body_id[wd_id]
                    nzb_col[start_nzb : start_nzb + num_cts] = 7 * follower_id_loc
                    start_nzb += num_cts

            # Transfer data to GPU
            self.sparse_jacobian.finalize(jacobian_dims, num_nzb.tolist())
            self.sparse_jacobian.dims.assign(jacobian_dims)
            self.sparse_jacobian.num_nzb.assign(num_nzb)
            self.sparse_jacobian.nzb_coords.assign(np.stack((nzb_row, nzb_col)).T.flatten())
            with wp.ScopedDevice(self.device):
                self.rb_nzb_id = wp.from_numpy(rb_nzb_id, dtype=wp.int32)
                self.ct_nzb_id_base = wp.from_numpy(ct_nzb_id_base, dtype=wp.int32)
                self.ct_nzb_id_follower = wp.from_numpy(ct_nzb_id_follower, dtype=wp.int32)

            # Initialize Jacobian assembly kernel
            self._eval_joint_constraints_sparse_jacobian_kernel = create_eval_joint_constraints_sparse_jacobian_kernel(
                has_universal_joints
            )

            # Initialize Jacobian linear operator
            self.sparse_jacobian_op = BlockSparseLinearOperators(self.sparse_jacobian)

            # Compute flat-array offsets for the CG solver (uniform world dimensions)
            cg_vio = wp.from_numpy(np.arange(self.num_worlds, dtype=np.int32) * self.num_states_max, device=self.device)
            cg_total_vec_size = self.num_worlds * self.num_states_max

            # Initialize preconditioner
            if self._preconditioner_type == ForwardKinematicsSolver.PreconditionerType.JACOBI_DIAGONAL:
                self.jacobian_diag_inv = wp.array(
                    dtype=wp.float32, device=self.device, shape=(self.num_worlds, self.num_states_max)
                )
                preconditioner_op = BatchedLinearOperator.from_diagonal(
                    self.jacobian_diag_inv.reshape((cg_total_vec_size,)),
                    self.num_states,
                    cg_vio,
                    self.num_states_max,
                )
            elif self._preconditioner_type == ForwardKinematicsSolver.PreconditionerType.JACOBI_BLOCK_DIAGONAL:
                self.inv_blocks_3 = wp.array(
                    dtype=wp.mat33f, shape=(self.num_worlds, self.num_bodies_max), device=self.device
                )
                self.inv_blocks_4 = wp.array(
                    dtype=wp.mat44f, shape=(self.num_worlds, self.num_bodies_max), device=self.device
                )
                blockwise_gemv_2d = get_blockwise_diag_3_4_gemv_2d(
                    self.inv_blocks_3, self.inv_blocks_4, self.num_states
                )
                n_wd, n_st = self.num_worlds, self.num_states_max

                def _blockwise_gemv_flat(x, y, world_active, alpha, beta):
                    blockwise_gemv_2d(x.reshape((n_wd, n_st)), y.reshape((n_wd, n_st)), world_active, alpha, beta)

                preconditioner_op = BatchedLinearOperator(
                    gemv_fn=_blockwise_gemv_flat,
                    n_worlds=self.num_worlds,
                    max_dim=self.num_states_max,
                    active_dims=self.num_states,
                    device=self.device,
                    dtype=wp.float32,
                    vio=cg_vio,
                    total_vec_size=cg_total_vec_size,
                )
            else:
                preconditioner_op = None

            # Initialize CG solver — wrap 2D gemv for flat 1D arrays
            n_wd, n_st = self.num_worlds, self.num_states_max

            def _cg_gemv_flat(x, y, world_active, alpha, beta):
                self._eval_lhs_gemv(x.reshape((n_wd, n_st)), y.reshape((n_wd, n_st)), world_active, alpha, beta)

            cg_op = BatchedLinearOperator(
                n_worlds=self.num_worlds,
                max_dim=self.num_states_max,
                active_dims=self.num_states,
                dtype=wp.float32,
                device=self.device,
                gemv_fn=_cg_gemv_flat,
                vio=cg_vio,
                total_vec_size=cg_total_vec_size,
            )
            self.cg_atol = wp.array(dtype=wp.float32, shape=self.num_worlds, device=self.device)
            self.cg_rtol = wp.array(dtype=wp.float32, shape=self.num_worlds, device=self.device)
            self.cg_max_iter = wp.from_numpy(2 * self.num_states.numpy(), dtype=wp.int32, device=self.device)
            self.linear_solver_cg = CGSolver(
                A=cg_op,
                active_dims=self.num_states,
                Mi=preconditioner_op,
                atol=self.cg_atol,
                rtol=self.cg_rtol,
                maxiter=self.cg_max_iter,
            )

    ###
    # Internal evaluators (graph-capturable functions working on pre-allocated data)
    ###

    def _reset_state(
        self,
        bodies_q: wp.array[wp.transformf],
        world_mask: wp.array[wp.int32],
    ):
        """
        Internal function resetting the bodies state to the reference state stored in the model.
        """
        wp.launch(
            _reset_state,
            dim=(self.num_worlds, self.num_states_max),
            inputs=[
                self.model.info.num_bodies,
                self.first_body_id,
                wp.array(
                    ptr=self.model.bodies.q_i_0.ptr,
                    dtype=wp.float32,
                    shape=(self.num_states_tot,),
                    device=self.device,
                    copy=False,
                ),
                world_mask,
                wp.array(
                    ptr=bodies_q.ptr, dtype=wp.float32, shape=(self.num_states_tot,), device=self.device, copy=False
                ),
            ],
            device=self.device,
        )

    def _reset_state_base_q(
        self,
        bodies_q: wp.array[wp.transformf],
        base_q: wp.array[wp.transformf],
        world_mask: wp.array[wp.int32],
    ):
        """
        Internal function resetting the bodies state to a rigid transformation of the reference state,
        computed so that the base body is aligned on its prescribed pose.
        """
        wp.launch(
            _reset_state_base_q,
            dim=(self.num_worlds, self.num_bodies_max),
            inputs=[
                self.base_joint_id,
                base_q,
                self.joints_X_j,
                self.joints_B_r_Bj,
                self.model.info.num_bodies,
                self.first_body_id,
                self.model.bodies.q_i_0,
                world_mask,
                bodies_q,
            ],
            device=self.device,
        )

    def _eval_position_control_transformations(
        self,
        base_q: wp.array[wp.transformf],
        actuators_q: wp.array[wp.float32],
        pos_control_transforms: wp.array[wp.transformf],
    ):
        """
        Internal evaluator for position control transformations, from actuated & base coordinates of the main model.
        """
        # Compute actuators_q of fk model with modified joints
        wp.launch(
            _eval_fk_actuated_dofs_or_coords,
            dim=(self.num_actuated_coords,),
            inputs=[
                wp.array(
                    ptr=base_q.ptr, dtype=wp.float32, shape=(7 * self.num_worlds,), device=self.device, copy=False
                ),
                actuators_q,
                self.actuated_coords_map,
                self.actuators_q,
            ],
            device=self.device,
        )

        # Compute position control transformations
        wp.launch(
            _eval_position_control_transformations,
            dim=(self.num_joints_tot,),
            inputs=[
                self.joints_dof_type,
                self.joints_act_type,
                self.actuated_coord_offsets,
                self.joints_X_j,
                self.actuators_q,
                pos_control_transforms,
            ],
            device=self.device,
        )

    def _eval_kinematic_constraints(
        self,
        bodies_q: wp.array[wp.transformf],
        pos_control_transforms: wp.array[wp.transformf],
        world_mask: wp.array[wp.int32],
        constraints: wp.array2d[wp.float32],
    ):
        """
        Internal evaluator for the kinematic constraints vector, from body poses and position-control transformations
        """

        # Evaluate unit norm quaternion constraints
        wp.launch(
            _eval_unit_quaternion_constraints,
            dim=(self.num_worlds, self.num_bodies_max),
            inputs=[self.model.info.num_bodies, self.first_body_id, bodies_q, world_mask, constraints],
            device=self.device,
        )
        # Evaluate joint constraints
        wp.launch(
            self._eval_joint_constraints_kernel,
            dim=(self.num_worlds, self.num_joints_max),
            inputs=[
                self.num_joints,
                self.first_joint_id,
                self.joints_dof_type,
                self.joints_act_type,
                self.joints_bid_B,
                self.joints_bid_F,
                self.joints_X_j,
                self.joints_B_r_Bj,
                self.joints_F_r_Fj,
                bodies_q,
                pos_control_transforms,
                self.constraint_full_to_red_map,
                world_mask,
                constraints,
            ],
            device=self.device,
        )

    def _eval_max_constraint(self, constraints: wp.array2d[wp.float32], max_constraint: wp.array[wp.float32]):
        """
        Internal evaluator for the maximal absolute constraint, from the constraints vector, in each world
        """
        max_constraint.zero_()
        wp.launch_tiled(
            self._eval_max_constraint_kernel,
            dim=(self.num_worlds, self.num_tiles_constraints),
            inputs=[constraints, max_constraint],
            block_dim=64,
            device=self.device,
        )

    def _eval_kinematic_constraints_jacobian(
        self,
        bodies_q: wp.array[wp.transformf],
        pos_control_transforms: wp.array[wp.transformf],
        world_mask: wp.array[wp.int32],
        constraints_jacobian: wp.array3d[wp.float32],
    ):
        """
        Internal evaluator for the kinematic constraints Jacobian with respect to body poses, from body poses
        and position-control transformations
        """

        # Evaluate unit norm quaternion constraints Jacobian
        wp.launch(
            _eval_unit_quaternion_constraints_jacobian,
            dim=(self.num_worlds, self.num_bodies_max),
            inputs=[self.model.info.num_bodies, self.first_body_id, bodies_q, world_mask, constraints_jacobian],
            device=self.device,
        )

        # Evaluate joint constraints Jacobian
        wp.launch(
            self._eval_joint_constraints_jacobian_kernel,
            dim=(self.num_worlds, self.num_joints_max),
            inputs=[
                self.num_joints,
                self.first_joint_id,
                self.first_body_id,
                self.joints_dof_type,
                self.joints_act_type,
                self.joints_bid_B,
                self.joints_bid_F,
                self.joints_X_j,
                self.joints_B_r_Bj,
                self.joints_F_r_Fj,
                bodies_q,
                pos_control_transforms,
                self.constraint_full_to_red_map,
                world_mask,
                constraints_jacobian,
            ],
            device=self.device,
        )

    def _assemble_sparse_jacobian(
        self,
        bodies_q: wp.array[wp.transformf],
        pos_control_transforms: wp.array[wp.transformf],
        world_mask: wp.array[wp.int32],
    ):
        """
        Internal evaluator for the sparse kinematic constraints Jacobian with respect to body poses, from body poses
        and position-control transformations
        """

        self.sparse_jacobian.zero()

        # Evaluate unit norm quaternion constraints Jacobian
        wp.launch(
            _eval_unit_quaternion_constraints_sparse_jacobian,
            dim=(self.num_worlds, self.num_bodies_max),
            inputs=[
                self.model.info.num_bodies,
                self.first_body_id,
                bodies_q,
                self.rb_nzb_id,
                world_mask,
                self.sparse_jacobian.nzb_values,
            ],
            device=self.device,
        )

        # Evaluate joint constraints Jacobian
        wp.launch(
            self._eval_joint_constraints_sparse_jacobian_kernel,
            dim=(self.num_worlds, self.num_joints_max),
            inputs=[
                self.num_joints,
                self.first_joint_id,
                self.first_body_id,
                self.joints_dof_type,
                self.joints_act_type,
                self.joints_bid_B,
                self.joints_bid_F,
                self.joints_X_j,
                self.joints_B_r_Bj,
                self.joints_F_r_Fj,
                bodies_q,
                pos_control_transforms,
                self.ct_nzb_id_base,
                self.ct_nzb_id_follower,
                world_mask,
                self.sparse_jacobian.nzb_values,
            ],
            device=self.device,
        )

    def _eval_lhs_gemv(
        self,
        x: wp.array2d[wp.float32],
        y: wp.array2d[wp.float32],
        world_mask: wp.array[wp.int32],
        alpha: wp.float32,
        beta: wp.float32,
    ):
        """
        Internal evaluator for y = alpha * J^T * J * x + beta * y, using the assembled sparse Jacobian J
        """
        self.sparse_jacobian_op.matvec(x, self.jacobian_times_vector, world_mask)
        self.sparse_jacobian_op.matvec_transpose(self.jacobian_times_vector, self.lhs_times_vector, world_mask)
        wp.launch(
            _eval_linear_combination,
            dim=(self.num_worlds, self.num_states_max),
            inputs=[alpha, self.lhs_times_vector, beta, y, self.num_constraints, world_mask, y],
            device=self.device,
        )

    def _eval_merit_function(self, constraints: wp.array2d[wp.float32], error: wp.array[wp.float32]):
        """
        Internal evaluator for the line search merit function, i.e. the least-squares error 1/2 * ||C||^2,
        from the constraints vector C, in each world
        """
        error.zero_()
        wp.launch_tiled(
            self._eval_merit_function_kernel,
            dim=(self.num_worlds, self.num_tiles_constraints),
            inputs=[constraints, error],
            block_dim=64,
            device=self.device,
        )

    def _eval_merit_function_gradient(
        self,
        step: wp.array2d[wp.float32],
        grad: wp.array2d[wp.float32],
        error_grad: wp.array[wp.float32],
    ):
        """
        Internal evaluator for the merit function gradient w.r.t. line search step size, from the step direction
        and the gradient in state space (= dC_ds^T * C). This is simply the dot product between these two vectors.
        """
        error_grad.zero_()
        wp.launch_tiled(
            self._eval_merit_function_gradient_kernel,
            dim=(self.num_worlds, self.num_tiles_states),
            inputs=[step, grad, error_grad],
            block_dim=64,
            device=self.device,
        )

    def _run_line_search_iteration(self, bodies_q: wp.array[wp.transformf]):
        """
        Internal function running one iteration of line search, checking the Armijo sufficient descent condition
        """
        # Eval stepped state
        wp.launch(
            _eval_stepped_state,
            dim=(self.num_worlds, self.num_states_max),
            inputs=[
                self.model.info.num_bodies,
                self.first_body_id,
                wp.array(
                    ptr=bodies_q.ptr, dtype=wp.float32, shape=(self.num_states_tot,), device=self.device, copy=False
                ),
                self.alpha,
                self.step,
                self.line_search_mask,
                wp.array(
                    ptr=self.bodies_q_alpha.ptr,
                    dtype=wp.float32,
                    shape=(self.num_states_tot,),
                    device=self.device,
                    copy=False,
                ),
            ],
            device=self.device,
        )

        # Evaluate new constraints and merit function (least squares norm of constraints)
        self._eval_kinematic_constraints(
            self.bodies_q_alpha, self.pos_control_transforms, self.line_search_mask, self.constraints
        )
        self._eval_merit_function(self.constraints, self.val_alpha)

        # Check decrease and update step
        self.line_search_loop_condition.zero_()
        wp.launch(
            _line_search_check,
            dim=(self.num_worlds,),
            inputs=[
                self.val_0,
                self.grad_0,
                self.alpha,
                self.val_alpha,
                self.line_search_iteration,
                self.max_line_search_iterations,
                self.line_search_success,
                self.line_search_mask,
                self.line_search_loop_condition,
            ],
            device=self.device,
        )

    def _update_cg_tolerance(
        self,
        residual_norm: wp.array[wp.float32],
        world_mask: wp.array[wp.int32],
    ):
        """
        Internal function heuristically adapting the CG tolerance based on the current constraint residual
        (starting with a loose tolerance, and tightening it as we converge)
        Note: needs to be refined, until then we are still using a fixed tolerance
        """
        wp.launch(
            _update_cg_tolerance_kernel,
            dim=(self.num_worlds,),
            inputs=[residual_norm, world_mask, self.cg_atol, self.cg_rtol],
            device=self.device,
        )

    def _run_newton_iteration(self, bodies_q: wp.array[wp.transformf]):
        """
        Internal function running one iteration of Gauss-Newton. Assumes the constraints vector to be already
        up-to-date (because we will already have checked convergence before the first loop iteration)
        """
        # Evaluate constraints Jacobian
        if self.config.use_sparsity:
            self._assemble_sparse_jacobian(bodies_q, self.pos_control_transforms, self.newton_mask)
        else:
            self._eval_kinematic_constraints_jacobian(
                bodies_q, self.pos_control_transforms, self.newton_mask, self.jacobian
            )

        # Evaluate Gauss-Newton left-hand side (J^T * J) if needed, and right-hand side (-J^T * C)
        if self.config.use_sparsity:
            self.sparse_jacobian_op.matvec_transpose(self.constraints, self.grad, self.newton_mask)
        else:
            wp.launch_tiled(
                self._eval_jacobian_T_jacobian_kernel,
                dim=(self.num_worlds, self.num_tiles_states, self.num_tiles_states),
                inputs=[self.jacobian, self.newton_mask, self.lhs],
                block_dim=64,
                device=self.device,
            )
            wp.launch_tiled(
                self._eval_jacobian_T_constraints_kernel,
                dim=(self.num_worlds, self.num_tiles_states),
                inputs=[self.jacobian, self.constraints, self.newton_mask, self.grad],
                block_dim=64,
                device=self.device,
            )
        wp.launch(
            _eval_rhs,
            dim=(self.num_worlds, self.num_states_max),
            inputs=[self.grad, self.rhs],
            device=self.device,
        )

        # Compute step (system solve)
        if self.config.use_sparsity:
            if self._preconditioner_type == ForwardKinematicsSolver.PreconditionerType.JACOBI_DIAGONAL:
                block_sparse_ATA_inv_diagonal_2d(self.sparse_jacobian, self.jacobian_diag_inv, self.newton_mask)
            elif self._preconditioner_type == ForwardKinematicsSolver.PreconditionerType.JACOBI_BLOCK_DIAGONAL:
                block_sparse_ATA_blockwise_3_4_inv_diagonal_2d(
                    self.sparse_jacobian, self.inv_blocks_3, self.inv_blocks_4, self.newton_mask
                )
            self.step.zero_()
            if self.config.use_adaptive_cg_tolerance:
                self._update_cg_tolerance(self.max_constraint, self.newton_mask)
            else:
                self.cg_atol.fill_(1e-8)
                self.cg_rtol.fill_(1e-8)
            self.linear_solver_cg.solve(
                self.rhs.reshape((-1,)), self.step.reshape((-1,)), world_active=self.newton_mask
            )
        else:
            self.linear_solver_llt.factorize(self.lhs, self.num_states, self.newton_mask)
            self.linear_solver_llt.solve(
                self.rhs.reshape((self.num_worlds, self.num_states_max, 1)),
                self.step.reshape((self.num_worlds, self.num_states_max, 1)),
                self.newton_mask,
            )

        # Line search
        self.line_search_iteration.zero_()
        self.line_search_success.zero_()
        wp.copy(self.line_search_mask, self.newton_mask)
        self.line_search_loop_condition.fill_(1)
        self._eval_merit_function(self.constraints, self.val_0)
        self._eval_merit_function_gradient(self.step, self.grad, self.grad_0)
        self.alpha.fill_(1.0)
        wp.capture_while(self.line_search_loop_condition, lambda: self._run_line_search_iteration(bodies_q))

        # Apply line search step and update max constraint
        wp.launch(
            _apply_line_search_step,
            dim=(self.num_worlds, self.num_bodies_max),
            inputs=[
                self.model.info.num_bodies,
                self.first_body_id,
                self.bodies_q_alpha,
                self.line_search_success,
                bodies_q,
            ],
            device=self.device,
        )
        self._eval_max_constraint(self.constraints, self.max_constraint)

        # Check convergence
        self.newton_loop_condition.zero_()
        wp.launch(
            _newton_check,
            dim=(self.num_worlds,),
            inputs=[
                self.max_constraint,
                self.tolerance,
                self.newton_iteration,
                self.max_newton_iterations,
                self.line_search_success,
                self.newton_success,
                self.newton_mask,
                self.newton_loop_condition,
            ],
            device=self.device,
        )

    def _solve_for_body_velocities(
        self,
        pos_control_transforms: wp.array[wp.transformf],
        base_u: wp.array[vec6f],
        actuators_u: wp.array[wp.float32],
        bodies_q: wp.array[wp.transformf],
        bodies_u: wp.array[vec6f],
        world_mask: wp.array[wp.int32],
    ):
        """
        Internal function solving for body velocities, so that constraint velocities are zero,
        except at actuated dofs and at the base joint, where they must match prescribed velocities.
        """
        # Compute actuators_u of fk model with modified joints
        wp.launch(
            _eval_fk_actuated_dofs_or_coords,
            dim=(self.num_actuated_dofs,),
            inputs=[
                wp.array(
                    ptr=base_u.ptr, dtype=wp.float32, shape=(6 * self.num_worlds,), device=self.device, copy=False
                ),
                actuators_u,
                self.actuated_dofs_map,
                self.actuators_u,
            ],
            device=self.device,
        )

        # Compute target constraint velocities (prescribed for actuated dofs, zero for passive constraints)
        self.target_cts_u.zero_()
        wp.launch(
            _eval_target_constraint_velocities,
            dim=(
                self.num_worlds,
                self.num_joints_max,
            ),
            inputs=[
                self.num_joints,
                self.first_joint_id,
                self.joints_dof_type,
                self.joints_act_type,
                self.actuated_dof_offsets,
                self.constraint_full_to_red_map,
                self.actuators_u,
                world_mask,
                self.target_cts_u,
            ],
            device=self.device,
        )

        # Update constraints Jacobian
        if self.config.use_sparsity:
            self._assemble_sparse_jacobian(bodies_q, pos_control_transforms, world_mask)
        else:
            self._eval_kinematic_constraints_jacobian(bodies_q, pos_control_transforms, world_mask, self.jacobian)

        # Evaluate system left-hand side (J^T * J) if needed, and right-hand side (J^T * targets_cts_u)
        if self.config.use_sparsity:
            self.sparse_jacobian_op.matvec_transpose(self.target_cts_u, self.rhs, world_mask)
        else:
            wp.launch_tiled(
                self._eval_jacobian_T_jacobian_kernel,
                dim=(self.num_worlds, self.num_tiles_states, self.num_tiles_states),
                inputs=[self.jacobian, world_mask, self.lhs],
                block_dim=64,
                device=self.device,
            )
            wp.launch_tiled(
                self._eval_jacobian_T_constraints_kernel,
                dim=(self.num_worlds, self.num_tiles_states),
                inputs=[self.jacobian, self.target_cts_u, world_mask, self.rhs],
                block_dim=64,
                device=self.device,
            )

        # Compute body velocities (system solve)
        if self.config.use_sparsity:
            if self._preconditioner_type == ForwardKinematicsSolver.PreconditionerType.JACOBI_DIAGONAL:
                block_sparse_ATA_inv_diagonal_2d(self.sparse_jacobian, self.jacobian_diag_inv, world_mask)
            elif self._preconditioner_type == ForwardKinematicsSolver.PreconditionerType.JACOBI_BLOCK_DIAGONAL:
                block_sparse_ATA_blockwise_3_4_inv_diagonal_2d(
                    self.sparse_jacobian, self.inv_blocks_3, self.inv_blocks_4, world_mask
                )
            self.bodies_q_dot.zero_()
            self.cg_atol.fill_(1e-8)
            self.cg_rtol.fill_(1e-8)
            self.linear_solver_cg.solve(
                self.rhs.reshape((-1,)), self.bodies_q_dot.reshape((-1,)), world_active=world_mask
            )
        else:
            self.linear_solver_llt.factorize(self.lhs, self.num_states, world_mask)
            self.linear_solver_llt.solve(
                self.rhs.reshape((self.num_worlds, self.num_states_max, 1)),
                self.bodies_q_dot.reshape((self.num_worlds, self.num_states_max, 1)),
                world_mask,
            )
        wp.launch(
            _eval_body_velocities,
            dim=(self.num_worlds, self.num_bodies_max),
            inputs=[self.model.info.num_bodies, self.first_body_id, bodies_q, self.bodies_q_dot, world_mask, bodies_u],
            device=self.device,
        )

    ###
    # Exposed functions (overall solve_fk() function + constraints (Jacobian) evaluators for debugging)
    ###

    def eval_position_control_transformations(
        self, actuators_q: wp.array[wp.float32], base_q: wp.array[wp.transformf] | None = None
    ):
        """
        Evaluates and returns position control transformations (an intermediary quantity needed for the
        kinematic constraints/Jacobian evaluation) for a model given actuated coordinates, and optionally
        the base pose (the default base pose is used if not provided).
        """
        assert base_q is None or base_q.device == self.device
        assert actuators_q.device == self.device

        if base_q is None:
            base_q = self.base_q_default

        pos_control_transforms = wp.array(dtype=wp.transformf, shape=(self.num_joints_tot,), device=self.device)
        self._eval_position_control_transformations(base_q, actuators_q, pos_control_transforms)
        return pos_control_transforms

    def eval_kinematic_constraints(
        self, bodies_q: wp.array[wp.transformf], pos_control_transforms: wp.array[wp.transformf]
    ):
        """
        Evaluates and returns the kinematic constraints vector given the body poses and the position
        control transformations.
        """
        assert bodies_q.device == self.device
        assert pos_control_transforms.device == self.device

        constraints = wp.zeros(
            dtype=wp.float32,
            shape=(
                self.num_worlds,
                self.num_constraints_max,
            ),
            device=self.device,
        )
        world_mask = wp.ones(dtype=wp.int32, shape=(self.num_worlds,), device=self.device)
        self._eval_kinematic_constraints(bodies_q, pos_control_transforms, world_mask, constraints)
        return constraints

    def eval_kinematic_constraints_jacobian(
        self, bodies_q: wp.array[wp.transformf], pos_control_transforms: wp.array[wp.transformf]
    ):
        """
        Evaluates and returns the kinematic constraints Jacobian (w.r.t. body poses) given the body poses
        and the position control transformations.
        """
        assert bodies_q.device == self.device
        assert pos_control_transforms.device == self.device

        constraints_jacobian = wp.zeros(
            dtype=wp.float32, shape=(self.num_worlds, self.num_constraints_max, self.num_states_max), device=self.device
        )
        world_mask = wp.ones(dtype=wp.int32, shape=(self.num_worlds,), device=self.device)
        self._eval_kinematic_constraints_jacobian(bodies_q, pos_control_transforms, world_mask, constraints_jacobian)
        return constraints_jacobian

    def assemble_sparse_jacobian(
        self, bodies_q: wp.array[wp.transformf], pos_control_transforms: wp.array[wp.transformf]
    ):
        """
        Assembles the sparse Jacobian (under self.sparse_jacobian) given input body poses and control transforms.
        Note: only safe to call if this object was finalized with sparsity enabled in the config.
        """
        assert bodies_q.device == self.device
        assert pos_control_transforms.device == self.device

        world_mask = wp.ones(dtype=wp.int32, shape=(self.num_worlds,), device=self.device)
        self._assemble_sparse_jacobian(bodies_q, pos_control_transforms, world_mask)

    def solve_for_body_velocities(
        self,
        pos_control_transforms: wp.array[wp.transformf],
        actuators_u: wp.array[wp.float32],
        bodies_q: wp.array[wp.transformf],
        bodies_u: wp.array[vec6f],
        base_u: wp.array[vec6f] | None = None,
        world_mask: wp.array[wp.int32] | None = None,
    ):
        """
        Graph-capturable function solving for body velocities as a post-processing to the FK solve.
        More specifically, solves for body twists yielding zero constraint velocities, except at
        actuated dofs and at the base joint, where velocities must match prescribed velocities.

        Parameters
        ----------
        pos_control_transforms : wp.array
            Array of position-control transforms, encoding actuated coordinates and base pose.
            Expects shape of ``(num_fk_joints,)`` and type :class:`transform`
        actuators_u : wp.array
            Array of actuated joint velocities.
            Expects shape of ``(sum_of_num_actuated_joint_dofs,)`` and type :class:`float`.
        bodies_q : wp.array
            Array of rigid body poses. Must be the solution of FK given the position-control transforms.
            Expects shape of ``(num_bodies,)`` and type :class:`transform`.
        bodies_u : wp.array
            Array of rigid body velocities (twists), written out by the solver.
            Expects shape of ``(num_bodies,)`` and type :class:`vec6`.
        base_u : wp.array, optional
            Velocity (twist) of the base body for each world, in the frame of the base joint if it was set, or
            absolute otherwise.
            If not provided, will default to zero. Ignored if no base body or joint was set for this model.
            If this function is captured in a graph, must be either always or never provided.
            Expects shape of ``(num_worlds,)`` and type :class:`vec6`.
        world_mask : wp.array, optional
            Array of per-world flags that indicate which worlds should be processed (0 = leave that world unchanged).
            If not provided, all worlds will be processed.
            If this function is captured in a graph, must be either always or never provided.
        """
        assert pos_control_transforms.device == self.device
        assert actuators_u.device == self.device
        assert bodies_q.device == self.device
        assert bodies_u.device == self.device
        assert base_u is None or base_u.device == self.device
        assert world_mask is None or world_mask.device == self.device

        # Use default base velocity if not provided
        if base_u is None:
            base_u = self.base_u_default

        # Compute velocities
        self._solve_for_body_velocities(pos_control_transforms, base_u, actuators_u, bodies_q, bodies_u, world_mask)

    def run_fk_solve(
        self,
        actuators_q: wp.array[wp.float32],
        bodies_q: wp.array[wp.transformf],
        base_q: wp.array[wp.transformf] | None = None,
        actuators_u: wp.array[wp.float32] | None = None,
        base_u: wp.array[vec6f] | None = None,
        bodies_u: wp.array[vec6f] | None = None,
        world_mask: wp.array[wp.int32] | None = None,
    ):
        """
        Graph-capturable function solving forward kinematics with Gauss-Newton.

        More specifically, solves for the rigid body poses satisfying
        kinematic constraints, given actuated joint coordinates and
        base pose. Optionally also solves for rigid body velocities
        given actuator and base body velocities.

        Parameters
        ----------
        actuators_q : wp.array
            Array of actuated joint coordinates.
            Expects shape of ``(sum_of_num_actuated_joint_coords,)`` and type :class:`float`.
        bodies_q : wp.array
            Array of rigid body poses, written out by the solver and read in as initial guess if the reset_state
            solver setting is False.
            Expects shape of ``(num_bodies,)`` and type :class:`transform`.
        base_q : wp.array, optional
            Pose of the base body for each world, in the frame of the base joint if it was set, or absolute otherwise.
            If not provided, will default to zero coordinates of the base joint, or the initial pose of the base body.
            If no base body or joint was set for this model, will be ignored.
            If this function is captured in a graph, must be either always or never provided.
            Expects shape of ``(num_worlds,)`` and type :class:`transform`.
        actuators_u : wp.array, optional
            Array of actuated joint velocities.
            Must be provided when solving for body velocities, i.e. if bodies_u is provided.
            If this function is captured in a graph, must be either always or never provided.
            Expects shape of ``(sum_of_num_actuated_joint_dofs,)`` and type :class:`float`.
        base_u : wp.array, optional
            Velocity (twist) of the base body for each world, in the frame of the base joint if it was set, or
            absolute otherwise.
            If not provided, will default to zero. Ignored if no base body or joint was set for this model.
            If this function is captured in a graph, must be either always or never provided.
            Expects shape of ``(num_worlds,)`` and type :class:`vec6`.
        bodies_u : wp.array, optional
            Array of rigid body velocities (twists), written out by the solver if provided.
            If this function is captured in a graph, must be either always or never provided.
            Expects shape of ``(num_bodies,)`` and type :class:`vec6`.
        world_mask : wp.array, optional
            Array of per-world flags that indicate which worlds should be processed (0 = leave that world unchanged).
            If not provided, all worlds will be processed.
            If this function is captured in a graph, must be either always or never provided.
        """
        # Check that actuators_u are provided if we need to solve for bodies_u
        if bodies_u is not None and actuators_u is None:
            raise ValueError(
                "run_fk_solve: actuators_u must be provided to solve for velocities (i.e. if bodies_u is provided)."
            )

        # Use default base state if not provided
        if base_q is None:
            base_q = self.base_q_default
        if bodies_u is not None and base_u is None:
            base_u = self.base_u_default

        # Compute position control transforms (independent of body poses, depends on controls only)
        self._eval_position_control_transformations(base_q, actuators_q, self.pos_control_transforms)

        # Reset iteration count and success/continuation flags
        self.newton_iteration.fill_(-1)  # The initial loop condition check will increment this to zero
        self.newton_success.zero_()
        if world_mask is not None:
            self.newton_mask.assign(world_mask)
        else:
            self.newton_mask.fill_(1)

        # Optionally reset state
        if self.config.reset_state:
            if base_q is None:
                self._reset_state(bodies_q, self.newton_mask)
            else:
                self._reset_state_base_q(bodies_q, base_q, self.newton_mask)

        # Evaluate constraints, and initialize loop condition (might not even need to loop)
        self._eval_kinematic_constraints(bodies_q, self.pos_control_transforms, self.newton_mask, self.constraints)
        self._eval_max_constraint(self.constraints, self.max_constraint)
        self.newton_loop_condition.zero_()
        wp.copy(self.line_search_success, self.newton_mask)  # Newton check will abort in case of line search failure
        wp.launch(
            _newton_check,
            dim=(self.num_worlds,),
            inputs=[
                self.max_constraint,
                self.tolerance,
                self.newton_iteration,
                self.max_newton_iterations,
                self.line_search_success,
                self.newton_success,
                self.newton_mask,
                self.newton_loop_condition,
            ],
            device=self.device,
        )

        # Main loop
        wp.capture_while(self.newton_loop_condition, lambda: self._run_newton_iteration(bodies_q))

        # Velocity solve, for worlds where FK ran and was successful
        if bodies_u is not None:
            self._solve_for_body_velocities(
                self.pos_control_transforms, base_u, actuators_u, bodies_q, bodies_u, self.newton_success
            )

    def solve_fk(
        self,
        actuators_q: wp.array[wp.float32],
        bodies_q: wp.array[wp.transformf],
        base_q: wp.array[wp.transformf] | None = None,
        actuators_u: wp.array[wp.float32] | None = None,
        base_u: wp.array[vec6f] | None = None,
        bodies_u: wp.array[vec6f] | None = None,
        world_mask: wp.array[wp.int32] | None = None,
        verbose: bool = False,
        return_status: bool = False,
        use_graph: bool = True,
    ):
        """
        Convenience function with verbosity options (non graph-capturable), solving
        forward kinematics with Gauss-Newton. More specifically, it solves for the
        rigid body poses satisfying kinematic constraints, given actuated joint
        coordinates and base pose. Optionally also solves for rigid body velocities
        given actuator and base body velocities.

        Parameters
        ----------
        actuators_q : wp.array
            Array of actuated joint coordinates.
            Expects shape of ``(sum_of_num_actuated_joint_coords,)`` and type :class:`float`.
        bodies_q : wp.array
            Array of rigid body poses, written out by the solver and read in as initial guess if the reset_state
            solver setting is False.
            Expects shape of ``(num_bodies,)`` and type :class:`transform`.
        base_q : wp.array, optional
            Pose of the base body for each world, in the frame of the base joint if it was set, or absolute otherwise.
            If not provided, will default to zero coordinates of the base joint, or the initial pose of the base body.
            If no base body or joint was set for this model, will be ignored.
            Expects shape of ``(num_worlds,)`` and type :class:`transform`.
        actuators_u : wp.array, optional
            Array of actuated joint velocities.
            Must be provided when solving for body velocities, i.e. if bodies_u is provided.
            Expects shape of ``(sum_of_num_actuated_joint_dofs,)`` and type :class:`float`.
        base_u : wp.array, optional
            Velocity (twist) of the base body for each world, in the frame of the base joint if it was set, or
            absolute otherwise.
            If not provided, will default to zero. Ignored if no base body or joint was set for this model.
            Expects shape of ``(num_worlds,)`` and type :class:`vec6`.
        bodies_u : wp.array, optional
            Array of rigid body velocities (twists), written out by the solver if provided.
            Expects shape of ``(num_bodies,)`` and type :class:`vec6`.
        world_mask : wp.array, optional
            Array of per-world flags that indicate which worlds should be processed (0 = leave that world unchanged).
            If not provided, all worlds will be processed.
        verbose : bool, optional
            whether to write a status message at the end (default: False)
        return_status : bool, optional
            whether to return the detailed solver status (default: False)
        use_graph : bool, optional
            whether to use graph capture internally to accelerate multiple calls to this function. Can be turned
            off for profiling individual kernels (default: True)

        Returns
        -------
        solver_status : ForwardKinematicsSolverStatus, optional
            the detailed solver status with success flag, number of iterations and constraint residual per world
        """
        assert base_q is None or base_q.device == self.device
        assert actuators_q.device == self.device
        assert bodies_q.device == self.device
        assert base_u is None or base_u.device == self.device
        assert actuators_u is None or actuators_u.device == self.device
        assert bodies_u is None or bodies_u.device == self.device

        # Run solve (with or without graph)
        if use_graph:
            if self.graph is None:
                wp.capture_begin(self.device)
                self.run_fk_solve(actuators_q, bodies_q, base_q, actuators_u, base_u, bodies_u, world_mask)
                self.graph = wp.capture_end()
            wp.capture_launch(self.graph)
        else:
            self.run_fk_solve(actuators_q, bodies_q, base_q, actuators_u, base_u, bodies_u, world_mask)

        # Status message
        if verbose or return_status:
            success = self.newton_success.numpy().copy()
            iterations = self.newton_iteration.numpy().copy()
            max_constraints = self.max_constraint.numpy().copy()
            num_active_worlds = self.num_worlds if world_mask is None else world_mask.numpy().sum()
            if verbose:
                sys.__stdout__.write(f"Newton success for {success.sum()}/{num_active_worlds} worlds; ")
                sys.__stdout__.write(f"num iterations={iterations.max()}; ")
                sys.__stdout__.write(f"max constraint={max_constraints.max()}\n")

        # Return solver status
        if return_status:
            return ForwardKinematicsSolver.Status(
                iterations=iterations, max_constraints=max_constraints, success=success
            )
