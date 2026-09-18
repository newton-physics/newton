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

from ......sim import ModelFlags
from ....config import ForwardKinematicsSolverConfig
from ...core.joints import JointActuationType, JointDoFType
from ...core.model import ModelKamino
from ...core.types import assign_to_warp_int32_array, to_warp_int32_array, vec7f
from ...kinematics.resets import get_base_q_from_joint_q_and_body_q
from ...linalg.blas import (
    block_sparse_ATA_blockwise_3_4_inv_diagonal_2d,
    block_sparse_ATA_inv_diagonal_2d,
    get_blockwise_diag_3_4_gemv_2d,
)
from ...linalg.conjugate import BatchedLinearOperator, CGSolver
from ...linalg.factorize.llt_blocked_semi_sparse import SemiSparseBlockCholeskySolverBatched
from ...linalg.sparse_matrix import BlockDType, BlockSparseMatrices
from ...linalg.sparse_operator import BlockSparseLinearOperators
from ...utils import logger as msg
from ...utils.tile import get_block_dim, get_num_tiles, get_tile_size
from ...utils.world_equivalence import DiscreteSignature, compute_equivalence_classes
from .kernels import (
    _add_regularizer_to_diagonal,
    _apply_line_search_step,
    _compute_fk_axis_joint_frames,
    _compute_fk_joint_frames,
    _correct_actuator_coords,
    _correct_universal_constraint_velocities,
    _eval_actuator_coords,
    _eval_body_velocities,
    _eval_fk_actuated_dofs_or_coords,
    _eval_incremental_target_actuator_coords,
    _eval_linear_combination,
    _eval_regularizer_gradient,
    _eval_rhs,
    _eval_stepped_state,
    _eval_target_constraint_velocities,
    _eval_target_relative_transformations,
    _eval_unit_quaternion_constraints,
    _eval_unit_quaternion_constraints_jacobian,
    _eval_unit_quaternion_constraints_sparse_jacobian,
    _initialize_jacobian_update_masks,
    _line_search_check,
    _newton_check,
    _reset_state,
    _reset_state_base_q,
    _resolve_fk_actuation_types,
    _update_cg_tolerance_kernel,
    make_1d_tile_based_kernels,
    make_2d_tile_based_kernels,
    make_eval_joint_constraints_jacobian_kernel,
    make_eval_joint_constraints_kernel,
    make_eval_joint_constraints_sparse_jacobian_kernel,
    make_eval_min_num_iterations_kernel,
    validate_fk_actuation_updates,
)
from .types import (
    FKData,
    FKJointDoFType,
    FKVelocitySolveData,
    ForwardKinematicsPreconditionerType,
    ForwardKinematicsStatus,
)

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
    Forward Kinematics solver class.

    FK requires every joint to be either fully passive or fully actuated.
    Dynamics supports per-DoF actuation, but mixed passive and actuated DoFs
    within one joint are not yet supported by FK.
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

        Args:
            model: Model for which to solve forward kinematics. If not provided, the finalize() method
                must be called at a later time for deferred initialization.
            config: Solver config. If not provided, the default config will be used.
        """

        self.model: ModelKamino | None = None
        """Underlying model"""

        self.device: wp.DeviceLike = None
        """Device for data allocations"""

        self.config: ForwardKinematicsSolver.Config = ForwardKinematicsSolver.Config()
        """Solver config

        Unless documented explicitly for individual settings, configuration changes must occur
        before ``finalize()`` is called (else they will have no effect or lead to undefined behavior).
        """

        self.data: FKData | None = None
        """ Internal solver data, allocated by finalize() """

        self.graph: wp.Graph | None = None
        """Cuda graph for the convenience function with verbosity options"""

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

        Args:
            model: Model for which to solve forward kinematics. If not provided, the model given to the
                constructor will be used. Must be provided if not given to the constructor.
            config: Solver config. If not provided, the config given to the constructor, or if not, the
                default config will be used.
        """

        # Initialize the model and config if provided
        if model is not None:
            self.model = model
        if config is not None:
            self.config = config
        if self.model is None:
            raise ValueError("ForwardKinematicsSolver: error, provided model is None.")

        # Validate config
        try:
            self.config.validate()
        except Exception as e:
            raise RuntimeError("Solver configuration is invalid.") from e

        # Initialize device
        self.device = self.model.device

        # Initialize internal data
        self.data = FKData()
        self.data.linear_system.preconditioner_type = ForwardKinematicsSolver.PreconditionerType.from_string(
            self.config.preconditioner
        )

        # Retrieve / compute dimensions - Bodies
        num_bodies = self.model.info.num_bodies.numpy()  # Number of bodies per world
        bodies_offset = self.model.info.bodies_offset.numpy()  # Index of first body per world

        # Retrieve / compute dimensions - States (i.e., body poses)
        num_states = 7 * num_bodies  # Number of state dimensions per world
        self.data.dimensions.num_states_tot = (
            7 * self.model.size.sum_of_num_bodies
        )  # State dimensions for the whole model
        self.data.dimensions.num_states_max = 7 * self.model.size.max_of_num_bodies  # Max state dimension across worlds

        # Retrieve / compute dimensions - Joints (main model)
        num_joints_prev = self.model.info.num_joints.numpy().copy()  # Number of joints per world
        joints_offset_prev = np.concatenate(([0], num_joints_prev.cumsum()))  # Index of first joint per world

        # Resolve custom actuation types
        if self.model.joints.fk_act_flag is not None:
            fk_act_flag = self.model.joints.fk_act_flag.numpy()
            invalid = np.flatnonzero((fk_act_flag < -1) | (fk_act_flag > 1))
            if invalid.size > 0:
                joint = int(invalid[0])
                raise ValueError(f"Invalid FK actuation flag for joint {joint}: expected -1, 0, or 1")
        resolved_act_type = wp.empty(
            shape=self.model.size.sum_of_num_joints,
            dtype=wp.int32,
            device=self.device,
        )
        if self.model.size.sum_of_num_joints > 0:
            wp.launch(
                _resolve_fk_actuation_types,
                dim=self.model.size.sum_of_num_joints,
                inputs=[
                    self.model.joints.act_type,
                    self.model.joints.fk_act_flag,
                    resolved_act_type,
                ],
                device=self.device,
            )
        joint_act_type_prev = resolved_act_type.numpy()
        joint_dof_start = self.model.joints.dofs_offset.numpy()
        joint_dof_act_types = self.model.joints.dof_act_types.numpy()
        base_joint_ids_input = self.model.info.base_joint_index.numpy().tolist()
        base_joint_id_set = {joint_id for joint_id in base_joint_ids_input if joint_id >= 0}
        for joint in range(self.model.size.sum_of_num_joints):
            if joint in base_joint_id_set:
                continue
            dof_actuation = joint_dof_act_types[joint_dof_start[joint] : joint_dof_start[joint + 1]]
            passive = dof_actuation == JointActuationType.PASSIVE
            if np.any(passive) and not np.all(passive):
                raise ValueError(
                    f"Invalid FK actuation for joint {joint}: all DoFs must be passive or all must be actuated."
                )
        # Indexed by model joint: 0 is passive, 1 is actuated, and -1 skips
        # validation for an explicit base joint that FK replaces.
        built_fk_actuated = (joint_act_type_prev != JointActuationType.PASSIVE).astype(np.int32)

        # Retrieve / compute dimensions - Actuated coordinates/dofs (main model)
        if self.model.joints.fk_act_flag is None:
            actuated_coords_offset_prev = self.model.joints.actuated_coords_offset.numpy().copy()
            actuated_dofs_offset_prev = self.model.joints.actuated_dofs_offset.numpy().copy()
        else:
            num_act_coords = self.model.joints.num_coords.numpy()
            num_act_dofs = self.model.joints.num_dofs.numpy()
            passive_mask = joint_act_type_prev == JointActuationType.PASSIVE
            num_act_coords[passive_mask] = 0
            num_act_dofs[passive_mask] = 0
            actuated_coords_offset_prev = np.concatenate(([0], num_act_coords.cumsum()))
            actuated_dofs_offset_prev = np.concatenate(([0], num_act_dofs.cumsum()))

        # Determine which worlds are equivalent for FK (at least discrete data)
        classes = compute_fk_equivalence_classes(self.model)
        num_classes = len(classes)

        # Create a copy of the model's joints with added joints as needed:
        # - actuated free joints to reset the base position/orientation
        # - axis joints to factor out superfluous DoFs at tie rods
        # We resolve discrete joint data (e.g. types and indices) first, then
        # copy or compute continuous joint data (e.g. frames).
        joint_dof_type_prev = self.model.joints.dof_type.numpy().copy()
        joint_bid_B_prev = self.model.joints.bid_B.numpy().copy()
        joint_bid_F_prev = self.model.joints.bid_F.numpy().copy()
        joints_num_coords_prev = self.model.joints.num_coords.numpy().copy()
        joints_num_dofs_prev = self.model.joints.num_dofs.numpy().copy()
        joint_dof_type = []
        joint_act_type = []
        joint_bid_B = []
        joint_bid_F = []
        joints_num_actuated_coords = []  # Number of actuated coordinates per joint (0 for passive joints)
        joints_num_actuated_dofs = []  # Number of actuated dofs per joint (0 for passive joints)
        joints_source_id = []  # Source joint in the main model, or -1 for synthetic joints
        joint_world_id = []  # World index per joint
        axis_joint_d = []  # FK index of each synthetic axis joint
        axis_body_id = []  # Body defining each synthetic axis joint
        axis_source_joint_0 = []  # First source joint defining each synthetic axis joint
        axis_source_joint_1 = []  # Second source joint defining each synthetic axis joint
        num_joints = np.zeros(self.model.size.num_worlds, dtype=np.int32)  # Number of joints per world
        self.data.dimensions.num_joints_tot = 0  # Number of joints for all worlds
        actuated_coords_map = []  # Map of new actuated coordinates to these in the model or to the base coordinates
        actuated_dofs_map = []  # Map of new actuated dofs to these in the model or to the base dofs
        base_joint_ids = self.model.size.num_worlds * [-1]  # Base joint id per world
        base_body_ids_input = self.model.info.base_body_index.numpy().tolist()
        for base_joint_id in base_joint_ids_input:
            if base_joint_id >= 0:
                # FK always replaces an explicit base joint with an actuated free joint.
                built_fk_actuated[base_joint_id] = -1
        for wd_id in range(self.model.size.num_worlds):
            # Retrieve base joint id
            base_joint_id = base_joint_ids_input[wd_id]

            # Copy data for all kept joints
            world_joint_ids = [
                i for i in range(joints_offset_prev[wd_id], joints_offset_prev[wd_id + 1]) if i != base_joint_id
            ]
            for jt_id_prev in world_joint_ids:
                # Note: we use the fact that integer values of the FK vs Kamino dof type enums
                # are matched for all joints that are not FK-specific
                joint_dof_type.append(joint_dof_type_prev[jt_id_prev])
                joint_act_type.append(joint_act_type_prev[jt_id_prev])
                joint_bid_B.append(joint_bid_B_prev[jt_id_prev])
                joint_bid_F.append(joint_bid_F_prev[jt_id_prev])
                joints_source_id.append(jt_id_prev)
                joint_world_id.append(wd_id)
                if joint_act_type[-1] != JointActuationType.PASSIVE:
                    num_coords_jt = joints_num_coords_prev[jt_id_prev]
                    joints_num_actuated_coords.append(num_coords_jt)
                    coord_offset = actuated_coords_offset_prev[jt_id_prev]
                    actuated_coords_map.extend(range(coord_offset, coord_offset + num_coords_jt))

                    num_dofs_jt = joints_num_dofs_prev[jt_id_prev]
                    joints_num_actuated_dofs.append(num_dofs_jt)
                    dof_offset = actuated_dofs_offset_prev[jt_id_prev]
                    actuated_dofs_map.extend(range(dof_offset, dof_offset + num_dofs_jt))
                else:
                    joints_num_actuated_coords.append(0)
                    joints_num_actuated_dofs.append(0)

            # Add axis joints as needed
            if self.config.add_axis_joints:
                # Find all bodies incident to two pure 3-DoF rotation joints.
                num_joints_per_body = np.zeros(dtype=np.int32, shape=num_bodies[wd_id])
                rotation_joints_per_body = [[] for _ in range(num_bodies[wd_id])]
                for jt_id_prev in world_joint_ids:
                    is_rotation = JointDoFType(joint_dof_type_prev[jt_id_prev]).is_pure_three_dof_rotation
                    bid_B = joint_bid_B_prev[jt_id_prev]
                    if bid_B >= 0:
                        bid_B -= bodies_offset[wd_id]
                        num_joints_per_body[bid_B] += 1
                        if is_rotation:
                            rotation_joints_per_body[bid_B].append(jt_id_prev)
                    bid_F = joint_bid_F_prev[jt_id_prev] - bodies_offset[wd_id]
                    num_joints_per_body[bid_F] += 1
                    if is_rotation:
                        rotation_joints_per_body[bid_F].append(jt_id_prev)

                # Add an axis joint for each such body
                for rb_id in range(num_bodies[wd_id]):
                    if num_joints_per_body[rb_id] != 2 or len(rotation_joints_per_body[rb_id]) != 2:
                        continue
                    rb_id_tot = bodies_offset[wd_id] + rb_id
                    joint_dof_type.append(FKJointDoFType.AXIS)
                    joint_act_type.append(JointActuationType.PASSIVE)
                    joint_bid_B.append(-1)
                    joint_bid_F.append(rb_id_tot)
                    joints_source_id.append(-1)
                    joint_world_id.append(wd_id)
                    axis_joint_d.append(len(joint_dof_type) - 1)
                    axis_body_id.append(rb_id_tot)
                    axis_source_joint_0.append(rotation_joints_per_body[rb_id][0])
                    axis_source_joint_1.append(rotation_joints_per_body[rb_id][1])
                    joints_num_actuated_coords.append(0)
                    joints_num_actuated_dofs.append(0)

            # Add joint for base joint / base body
            if base_joint_id >= 0:  # Replace base joint with an actuated free joint
                joint_dof_type.append(JointDoFType.FREE)
                joint_act_type.append(JointActuationType.FORCE)
                joint_bid_B.append(-1)
                joint_bid_F.append(joint_bid_F_prev[base_joint_id])
                joints_source_id.append(base_joint_id)
                joint_world_id.append(wd_id)
                joints_num_actuated_coords.append(7)
                coord_offset = -7 * wd_id - 1  # We encode offsets in base_q negatively with i -> -i - 1
                actuated_coords_map.extend(range(coord_offset, coord_offset - 7, -1))
                joints_num_actuated_dofs.append(6)
                dof_offset = -6 * wd_id - 1  # We encode offsets in base_u negatively with i -> -i - 1
                actuated_dofs_map.extend(range(dof_offset, dof_offset - 6, -1))
                base_joint_ids[wd_id] = len(joint_dof_type) - 1
            elif base_body_ids_input[wd_id] >= 0:  # Add an actuated free joint to the base body
                base_body_id = base_body_ids_input[wd_id]
                joint_dof_type.append(JointDoFType.FREE)
                joint_act_type.append(JointActuationType.FORCE)
                joint_bid_B.append(-1)
                joint_bid_F.append(base_body_id)
                joints_source_id.append(-1)
                joint_world_id.append(wd_id)
                joints_num_actuated_coords.append(7)
                coord_offset = -7 * wd_id - 1  # We encode offsets in base_q negatively with i -> -i - 1
                actuated_coords_map.extend(range(coord_offset, coord_offset - 7, -1))
                joints_num_actuated_dofs.append(6)
                dof_offset = -6 * wd_id - 1  # We encode offsets in base_u negatively with i -> -i - 1
                actuated_dofs_map.extend(range(dof_offset, dof_offset - 6, -1))
                base_joint_ids[wd_id] = len(joint_dof_type) - 1

            # Record number of joints
            num_joints_world = len(joint_dof_type) - self.data.dimensions.num_joints_tot
            self.data.dimensions.num_joints_tot += num_joints_world
            num_joints[wd_id] = num_joints_world

        # Retrieve / compute dimensions - Joints (FK model)
        joints_offset = np.concatenate(([0], num_joints.cumsum()))  # Index of first joint per world
        self.data.dimensions.num_joints_max = max(num_joints)  # Max number of joints across worlds

        # Retrieve / compute dimensions - Actuated coordinates (FK model)
        joints_num_actuated_coords = np.array(joints_num_actuated_coords)  # Number of actuated coordinates per joint
        actuated_coords_offset = np.concatenate(
            ([0], joints_num_actuated_coords.cumsum())
        )  # First actuated coordinate offset per joint, among all actuated coordinates
        self.data.dimensions.num_actuated_coords = actuated_coords_offset[-1]
        world_num_actuated_coords = np.array(
            [
                joints_num_actuated_coords[joints_offset[wd_id] : joints_offset[wd_id + 1]].sum()
                for wd_id in range(self.model.size.num_worlds)
            ]
        )
        world_actuated_coords_offset = np.concatenate(([0], world_num_actuated_coords.cumsum()))
        self.data.dimensions.num_actuated_coords_max = np.max(world_num_actuated_coords)

        # Retrieve / compute dimensions - Actuated dofs (FK model)
        joints_num_actuated_dofs = np.array(joints_num_actuated_dofs)  # Number of actuated dofs per joint
        actuated_dofs_offset = np.concatenate(
            ([0], joints_num_actuated_dofs.cumsum())
        )  # First actuated dof offset per joint, among all actuated dofs
        self.data.dimensions.num_actuated_dofs = actuated_dofs_offset[-1]

        # Retrieve / compute dimensions - Constraints
        num_constraints = num_bodies.copy()  # Number of kinematic constraints per world (unit quat. + joints)
        has_universal_joints = False  # Whether the model has at least one passive universal joint
        self.data.joints.has_universal_actuators = False  # Whether the model has at least one actuated universal joint
        constraint_full_to_red_map = np.full(6 * self.data.dimensions.num_joints_tot, -1, dtype=np.int32)
        for eq_class in classes:
            # Count constraints for first world in equivalence class
            wd_id = eq_class[0]
            ct_count = num_constraints[wd_id]
            for jt_id in range(joints_offset[wd_id], joints_offset[wd_id + 1]):
                act_type = joint_act_type[jt_id]
                dof_type = joint_dof_type[jt_id]
                if act_type != JointActuationType.PASSIVE:  # Actuator: select all six constraints
                    for i in range(6):
                        constraint_full_to_red_map[6 * jt_id + i] = ct_count + i
                    ct_count += 6
                    if dof_type == FKJointDoFType.UNIVERSAL:
                        self.data.joints.has_universal_actuators = True
                else:
                    if dof_type == FKJointDoFType.AXIS:
                        constraint_full_to_red_map[6 * jt_id + 3] = ct_count
                        ct_count += 1
                    elif dof_type == FKJointDoFType.CARTESIAN:
                        for i in range(3):
                            constraint_full_to_red_map[6 * jt_id + 3 + i] = ct_count + i
                        ct_count += 3
                    elif dof_type == FKJointDoFType.CYLINDRICAL:
                        constraint_full_to_red_map[6 * jt_id + 1] = ct_count
                        constraint_full_to_red_map[6 * jt_id + 2] = ct_count + 1
                        constraint_full_to_red_map[6 * jt_id + 4] = ct_count + 2
                        constraint_full_to_red_map[6 * jt_id + 5] = ct_count + 3
                        ct_count += 4
                    elif dof_type == FKJointDoFType.FIXED:
                        for i in range(6):
                            constraint_full_to_red_map[6 * jt_id + i] = ct_count + i
                        ct_count += 6
                    elif dof_type == FKJointDoFType.FREE:
                        pass
                    elif dof_type == FKJointDoFType.PRISMATIC:
                        constraint_full_to_red_map[6 * jt_id + 1] = ct_count
                        constraint_full_to_red_map[6 * jt_id + 2] = ct_count + 1
                        for i in range(3):
                            constraint_full_to_red_map[6 * jt_id + 3 + i] = ct_count + 2 + i
                        ct_count += 5
                    elif dof_type == FKJointDoFType.REVOLUTE:
                        for i in range(3):
                            constraint_full_to_red_map[6 * jt_id + i] = ct_count + i
                        constraint_full_to_red_map[6 * jt_id + 4] = ct_count + 3
                        constraint_full_to_red_map[6 * jt_id + 5] = ct_count + 4
                        ct_count += 5
                    elif dof_type == FKJointDoFType.SPHERICAL:
                        for i in range(3):
                            constraint_full_to_red_map[6 * jt_id + i] = ct_count + i
                        ct_count += 3
                    elif dof_type == FKJointDoFType.GIMBAL or dof_type == FKJointDoFType.GIMBAL_LEFT_HANDED:
                        for i in range(3):
                            constraint_full_to_red_map[6 * jt_id + i] = ct_count + i
                        ct_count += 3
                    elif dof_type == FKJointDoFType.UNIVERSAL:
                        for i in range(3):
                            constraint_full_to_red_map[6 * jt_id + i] = ct_count + i
                        constraint_full_to_red_map[6 * jt_id + 5] = ct_count + 3
                        ct_count += 4
                        has_universal_joints = True
                    else:
                        raise RuntimeError("Unknown joint dof type")
            num_constraints[wd_id] = ct_count

            # Copy constraints counts/map data for other worlds in equivalence class
            for wd_id_1 in eq_class[1:]:
                constraint_full_to_red_map[6 * joints_offset[wd_id_1] : 6 * joints_offset[wd_id_1 + 1]] = (
                    constraint_full_to_red_map[6 * joints_offset[wd_id] : 6 * joints_offset[wd_id + 1]]
                )
                num_constraints[wd_id_1] = num_constraints[wd_id]
        self.data.dimensions.num_constraints_max = np.max(num_constraints)

        # Initialize maximal step size per iteration in actuated coordinates
        if self.config.use_incremental_solve:
            delta_q_max = np.zeros(self.data.dimensions.num_actuated_coords, dtype=np.float32)
            max_step_linear = self.config.max_linear_incremental_step
            max_step_angular = self.config.max_angular_incremental_step
            half_angle = 0.5 * min(max_step_angular, np.pi)
            max_step_quat = max(np.sin(half_angle), 1.0 - np.cos(half_angle))
            for eq_class in classes:
                # Initialize delta_q_max for first world in equivalence class
                wd_id = eq_class[0]
                for jt_id in range(joints_offset[wd_id], joints_offset[wd_id + 1]):
                    if joints_num_actuated_coords[jt_id] == 0:
                        continue
                    dof_type = joint_dof_type[jt_id]
                    coord_id = actuated_coords_offset[jt_id]
                    if dof_type == FKJointDoFType.CARTESIAN:
                        for i in range(3):
                            delta_q_max[coord_id + i] = max_step_linear
                    elif dof_type == FKJointDoFType.CYLINDRICAL:
                        delta_q_max[coord_id] = max_step_linear
                        delta_q_max[coord_id + 1] = max_step_angular
                    elif dof_type == FKJointDoFType.FIXED:
                        pass
                    elif dof_type == FKJointDoFType.FREE:
                        for i in range(3):
                            delta_q_max[coord_id + i] = max_step_linear
                        for i in range(4):
                            delta_q_max[coord_id + 3 + i] = max_step_quat
                    elif dof_type == FKJointDoFType.PRISMATIC:
                        delta_q_max[coord_id] = max_step_linear
                    elif dof_type == FKJointDoFType.REVOLUTE:
                        delta_q_max[coord_id] = max_step_angular
                    elif dof_type == FKJointDoFType.SPHERICAL:
                        for i in range(4):
                            delta_q_max[coord_id + i] = max_step_quat
                    elif dof_type == FKJointDoFType.GIMBAL or dof_type == FKJointDoFType.GIMBAL_LEFT_HANDED:
                        for i in range(3):
                            delta_q_max[coord_id + i] = max_step_angular
                    elif dof_type == FKJointDoFType.UNIVERSAL:
                        delta_q_max[coord_id] = max_step_angular
                        delta_q_max[coord_id + 1] = max_step_angular
                    else:
                        raise RuntimeError("Invalid joint dof type for an actuator")

                # Copy delta_q_max for other worlds in equivalence class
                for wd_id_1 in eq_class[1:]:
                    delta_q_max[world_actuated_coords_offset[wd_id_1] : world_actuated_coords_offset[wd_id_1 + 1]] = (
                        delta_q_max[world_actuated_coords_offset[wd_id] : world_actuated_coords_offset[wd_id + 1]]
                    )

        # Retrieve / compute dimensions - Number of tiles (for kernels using Tile API)
        # For 1d reduction kernels, large tiles yield the best performance
        self.data.dimensions.tile_size_cts_1d = get_tile_size(self.data.dimensions.num_constraints_max)
        self.data.dimensions.num_tiles_cts_1d = get_num_tiles(
            self.data.dimensions.num_constraints_max, self.data.dimensions.tile_size_cts_1d
        )
        self.data.dimensions.tile_size_vrs_1d = get_tile_size(self.data.dimensions.num_states_max)
        self.data.dimensions.num_tiles_vrs_1d = get_num_tiles(
            self.data.dimensions.num_states_max, self.data.dimensions.tile_size_vrs_1d
        )
        # For 2d matrix product kernels, smaller 16x16 tiles give the best tradeoff (also for using sparsity)
        self.data.dimensions.tile_size_cts_2d = 16
        self.data.dimensions.num_tiles_cts_2d = get_num_tiles(
            self.data.dimensions.num_constraints_max, self.data.dimensions.tile_size_cts_2d
        )
        self.data.dimensions.tile_size_vrs_2d = 16
        self.data.dimensions.num_tiles_vrs_2d = get_num_tiles(
            self.data.dimensions.num_states_max, self.data.dimensions.tile_size_vrs_2d
        )
        # For optional 1d reduction kernel over actuated coordinates
        if self.config.use_incremental_solve:
            self.data.dimensions.tile_size_coords = get_tile_size(self.data.dimensions.num_actuated_coords_max)
            self.data.dimensions.num_tiles_coords = get_num_tiles(
                self.data.dimensions.num_actuated_coords_max, self.data.dimensions.tile_size_coords
            )

        # Data allocation or transfer from numpy to warp
        with wp.ScopedDevice(self.device):
            # Dimensions
            self.data.dimensions.num_joints = to_warp_int32_array(num_joints)
            self.data.dimensions.joints_offset = to_warp_int32_array(joints_offset)
            self.data.dimensions.actuated_coords_offset = to_warp_int32_array(actuated_coords_offset)
            self.data.dimensions.actuated_coords_map = to_warp_int32_array(np.array(actuated_coords_map))
            self.data.dimensions.world_actuated_coords_offset = to_warp_int32_array(world_actuated_coords_offset)
            self.data.dimensions.actuated_dofs_offset = to_warp_int32_array(actuated_dofs_offset)
            self.data.dimensions.actuated_dofs_map = to_warp_int32_array(np.array(actuated_dofs_map))
            self.data.dimensions.num_states = to_warp_int32_array(num_states)
            self.data.dimensions.num_constraints = to_warp_int32_array(num_constraints)
            self.data.dimensions.constraint_full_to_red_map = to_warp_int32_array(constraint_full_to_red_map)
            self.data.dimensions.num_axis_joints = len(axis_joint_d)
            self.data.dimensions.all_worlds_mask = wp.full(
                shape=(self.model.size.num_worlds,), value=True, dtype=wp.bool
            )

            # Joints — helper data for model updates validation
            self.data.joints._built_actuated = to_warp_int32_array(built_fk_actuated)
            self.data.joints._actuation_violations = wp.empty(2, dtype=wp.int32)

            # Joints — FK joints model
            self.data.joints.dof_type = to_warp_int32_array(joint_dof_type)
            self.data.joints.act_type = to_warp_int32_array(joint_act_type)
            self.data.joints.bid_B = to_warp_int32_array(joint_bid_B)
            self.data.joints.bid_F = to_warp_int32_array(joint_bid_F)
            self.data.joints.B_r_Bj = wp.empty(self.data.dimensions.num_joints_tot, dtype=wp.vec3f)
            self.data.joints.F_r_Fj = wp.empty(self.data.dimensions.num_joints_tot, dtype=wp.vec3f)
            self.data.joints.X_Bj = wp.empty(self.data.dimensions.num_joints_tot, dtype=wp.mat33f)
            self.data.joints.X_Fj = wp.empty(self.data.dimensions.num_joints_tot, dtype=wp.mat33f)
            self.data.joints.source_id = to_warp_int32_array(joints_source_id)
            self.data.joints.world_id = to_warp_int32_array(joint_world_id)
            self.data.joints.axis_joint_id = to_warp_int32_array(axis_joint_d)
            self.data.joints.axis_body_id = to_warp_int32_array(axis_body_id)
            self.data.joints.axis_source_joint_0 = to_warp_int32_array(axis_source_joint_0)
            self.data.joints.axis_source_joint_1 = to_warp_int32_array(axis_source_joint_1)
            self.data.joints.base_joint_id = to_warp_int32_array(base_joint_ids)

            # Problem data
            self.data.problem.actuator_q_next = wp.array(
                dtype=wp.float32, shape=(self.data.dimensions.num_actuated_coords,)
            )
            if self.config.use_incremental_solve:
                self.data.problem.actuator_q_prev = wp.array(
                    dtype=wp.float32, shape=(self.data.dimensions.num_actuated_coords,)
                )
                self.data.problem.actuator_q_curr = wp.array(
                    dtype=wp.float32, shape=(self.data.dimensions.num_actuated_coords,)
                )
            self.data.problem.target_rel_transforms = wp.array(
                dtype=wp.transformf, shape=(self.data.dimensions.num_joints_tot,)
            )
            if self.config.use_regularization:
                self.data.problem.body_q_ref = wp.array(dtype=wp.transformf, shape=(self.model.size.sum_of_num_bodies,))
            self.data.problem.base_q_default = wp.zeros(shape=self.model.size.num_worlds, dtype=wp.transformf)
            self.data.problem.base_u_default = wp.zeros(shape=(self.model.size.num_worlds,), dtype=wp.spatial_vectorf)
            self.data.problem.constraints = wp.zeros(
                dtype=wp.float32,
                shape=(
                    self.model.size.num_worlds,
                    self.data.dimensions.num_constraints_max,
                ),
            )
            self.data.problem.jacobian = wp.zeros(
                dtype=wp.float32,
                shape=(
                    self.model.size.num_worlds,
                    self.data.dimensions.num_constraints_max,
                    self.data.dimensions.num_states_max,
                ),
            )

            # Line search
            self.data.line_search.iteration = wp.array(dtype=wp.int32, shape=(self.model.size.num_worlds,))
            self.data.line_search.loop_condition = wp.array(dtype=wp.int32, shape=(1,))
            self.data.line_search.success = wp.array(dtype=wp.bool, shape=(self.model.size.num_worlds,))
            self.data.line_search.mask = wp.array(dtype=wp.bool, shape=(self.model.size.num_worlds,))
            self.data.line_search.val_0 = wp.array(dtype=wp.float32, shape=(self.model.size.num_worlds,))
            self.data.line_search.grad_0 = wp.array(dtype=wp.float32, shape=(self.model.size.num_worlds,))
            self.data.line_search.alpha = wp.array(dtype=wp.float32, shape=(self.model.size.num_worlds,))
            self.data.line_search.body_q_alpha = wp.array(
                dtype=wp.transformf, shape=(self.model.size.sum_of_num_bodies,)
            )
            self.data.line_search.val_alpha = wp.array(dtype=wp.float32, shape=(self.model.size.num_worlds,))

            # Gauss-Newton
            self.data.gauss_newton.min_iterations = wp.zeros(dtype=wp.int32, shape=(self.model.size.num_worlds,))
            self.data.gauss_newton.iteration = wp.array(dtype=wp.int32, shape=(self.model.size.num_worlds,))
            self.data.gauss_newton.loop_condition = wp.array(dtype=wp.int32, shape=(1,))
            self.data.gauss_newton.success = wp.array(dtype=wp.bool, shape=(self.model.size.num_worlds,))
            self.data.gauss_newton.mask = wp.array(dtype=wp.bool, shape=(self.model.size.num_worlds,))
            if self.config.use_regularization and self.config.use_incremental_solve:
                self.data.gauss_newton.jacobian_early_update_mask = wp.array(
                    dtype=wp.bool, shape=(self.model.size.num_worlds,)
                )
                self.data.gauss_newton.jacobian_late_update_mask = wp.array(
                    dtype=wp.bool, shape=(self.model.size.num_worlds,)
                )
            else:
                self.data.gauss_newton.jacobian_early_update_mask = None
                self.data.gauss_newton.jacobian_late_update_mask = None
            if self.config.use_incremental_solve:
                self.data.gauss_newton.delta_q_max = wp.from_numpy(delta_q_max, dtype=wp.float32)
            self.data.gauss_newton.grad = wp.zeros(
                dtype=wp.float32, shape=(self.model.size.num_worlds, self.data.dimensions.num_states_max)
            )
            self.data.gauss_newton.step = wp.zeros(
                dtype=wp.float32, shape=(self.model.size.num_worlds, self.data.dimensions.num_states_max)
            )
            self.data.gauss_newton.max_residual = wp.array(dtype=wp.float32, shape=(self.model.size.num_worlds,))

            # Linear system
            if not self.config.use_sparsity:
                self.data.linear_system.lhs = wp.zeros(
                    dtype=wp.float32,
                    shape=(
                        self.model.size.num_worlds,
                        self.data.dimensions.num_states_max,
                        self.data.dimensions.num_states_max,
                    ),
                )
            self.data.linear_system.rhs = wp.zeros(
                dtype=wp.float32, shape=(self.model.size.num_worlds, self.data.dimensions.num_states_max)
            )
            self.data.linear_system.jacobian_times_vector = wp.zeros(
                dtype=wp.float32, shape=(self.model.size.num_worlds, self.data.dimensions.num_constraints_max)
            )
            self.data.linear_system.lhs_times_vector = wp.zeros(
                dtype=wp.float32, shape=(self.model.size.num_worlds, self.data.dimensions.num_states_max)
            )

            # Velocity solve — preallocate batch_size 1; the RHS and body time derivative arrays
            # alias the Gauss-Newton RHS/step to reuse memory.
            self.data.velocity_solve[1] = FKVelocitySolveData(
                fk_actuator_u=wp.zeros(dtype=wp.float32, shape=(1, self.data.dimensions.num_actuated_dofs)),
                target_cts_u=wp.zeros(
                    dtype=wp.float32,
                    shape=(self.model.size.num_worlds, self.data.dimensions.num_constraints_max, 1),
                ),
                rhs=self.data.linear_system.rhs.reshape(
                    (self.model.size.num_worlds, self.data.dimensions.num_states_max, 1)
                ),
                body_q_dot=self.data.gauss_newton.step.reshape(
                    (self.model.size.num_worlds, self.data.dimensions.num_states_max, 1)
                ),
            )

        # Initialize kernels that depend on static values
        self._eval_joint_constraints_kernel = make_eval_joint_constraints_kernel(has_universal_joints)
        self._eval_joint_constraints_jacobian_kernel = make_eval_joint_constraints_jacobian_kernel(has_universal_joints)
        (
            self._eval_pattern_T_pattern_kernel,
            self._eval_jacobian_T_jacobian_kernel,
            self._eval_jacobian_T_constraints_kernel,
        ) = make_2d_tile_based_kernels(self.data.dimensions.tile_size_cts_2d, self.data.dimensions.tile_size_vrs_2d)
        (
            self._eval_max_residual_kernel,
            self._eval_merit_function_kernel,
            self._eval_regularizer_kernel,
            self._eval_merit_function_gradient_kernel,
        ) = make_1d_tile_based_kernels(
            self.data.dimensions.tile_size_cts_1d,
            self.data.dimensions.tile_size_vrs_1d,
            self.config.use_regularization,
        )
        if self.config.use_incremental_solve:
            self._eval_min_num_iterations_kernel = make_eval_min_num_iterations_kernel(
                self.data.dimensions.tile_size_coords
            )

        # Compute sparsity pattern and initialize linear solver for dense (semi-sparse) case
        if not self.config.use_sparsity:
            # Jacobian sparsity pattern
            sparsity_pattern = np.zeros(
                (num_classes, self.data.dimensions.num_constraints_max, self.data.dimensions.num_states_max),
                dtype=int,
            )
            for class_id in range(num_classes):
                wd_id = classes[class_id][0]  # Compute sparsity pattern for first world in equivalence class
                for rb_id_loc in range(num_bodies[wd_id]):
                    sparsity_pattern[class_id, rb_id_loc, 7 * rb_id_loc + 3 : 7 * rb_id_loc + 7] = 1
                for jt_id_loc in range(num_joints[wd_id]):
                    jt_id_tot = joints_offset[wd_id] + jt_id_loc
                    base_id_tot = joint_bid_B[jt_id_tot]
                    follower_id_tot = joint_bid_F[jt_id_tot]
                    rb_ids_tot = [base_id_tot, follower_id_tot] if base_id_tot >= 0 else [follower_id_tot]
                    for rb_id_tot in rb_ids_tot:
                        rb_id_loc = rb_id_tot - bodies_offset[wd_id]
                        state_offset = 7 * rb_id_loc
                        for i in range(3):
                            ct_offset = constraint_full_to_red_map[6 * jt_id_tot + i]  # ith translation constraint
                            if ct_offset >= 0:
                                sparsity_pattern[class_id, ct_offset, state_offset : state_offset + 7] = 1
                            ct_offset = constraint_full_to_red_map[6 * jt_id_tot + 3 + i]  # ith rotation constraint
                            if ct_offset >= 0:
                                sparsity_pattern[class_id, ct_offset, state_offset + 3 : state_offset + 7] = 1

            # Jacobian^T * Jacobian sparsity pattern
            sparsity_pattern_wp = wp.from_numpy(sparsity_pattern, dtype=wp.float32, device=self.device)
            sparsity_pattern_lhs_wp = wp.zeros(
                dtype=wp.float32,
                shape=(num_classes, self.data.dimensions.num_states_max, self.data.dimensions.num_states_max),
                device=self.device,
            )
            wp.launch_tiled(
                self._eval_pattern_T_pattern_kernel,
                dim=(num_classes, self.data.dimensions.num_tiles_vrs_2d, self.data.dimensions.num_tiles_vrs_2d),
                inputs=[sparsity_pattern_wp, sparsity_pattern_lhs_wp],
                block_dim=32,
                device=self.device,
            )
            sparsity_pattern_lhs = sparsity_pattern_lhs_wp.numpy().astype("int32")
            if self.config.use_regularization:  # Account for diagonal perturbation in sparsity pattern
                for class_id in range(num_classes):
                    wd_id = classes[class_id][0]
                    N = num_states[wd_id]
                    np.fill_diagonal(sparsity_pattern_lhs[class_id, :N, :N], 1)

            # Initialize linear solver (semi-sparse LLT)
            self.linear_solver_llt = SemiSparseBlockCholeskySolverBatched(
                self.model.size.num_worlds,
                self.data.dimensions.num_states_max,
                block_size=16,  # TODO: optimize this (e.g. 14 ?)
                device=self.device,
                enable_reordering=True,
            )
            num_states_per_class = np.array([num_states[eq_class[0]] for eq_class in classes])
            self.linear_solver_llt.capture_sparsity_pattern(sparsity_pattern_lhs, num_states_per_class, classes)

            # Compute tile-level Jacobian sparsity pattern, to skip zero tiles in tile-based matrix products
            tile_sparsity_pattern_np = np.zeros(
                (
                    self.model.size.num_worlds,
                    self.data.dimensions.num_tiles_cts_2d,
                    self.data.dimensions.num_tiles_vrs_2d,
                ),
                dtype=np.int32,
            )
            for class_id, eq_class in enumerate(classes):
                pattern = np.zeros(
                    (self.data.dimensions.num_tiles_cts_2d, self.data.dimensions.num_tiles_vrs_2d), dtype=np.int32
                )
                for i in range(self.data.dimensions.num_constraints_max):
                    for j in range(self.data.dimensions.num_states_max):
                        if sparsity_pattern[class_id, i, j] != 0:
                            tile_row = i // self.data.dimensions.tile_size_cts_2d
                            tile_col = j // self.data.dimensions.tile_size_vrs_2d
                            pattern[tile_row, tile_col] = 1
                for wd_id in eq_class:
                    tile_sparsity_pattern_np[wd_id] = pattern
            self.data.dimensions.tile_sparsity_pattern = to_warp_int32_array(
                tile_sparsity_pattern_np, device=self.device
            )

        # Compute sparsity pattern and initialize linear solver for sparse case
        if self.config.use_sparsity:
            self.data.problem.sparse_jacobian: BlockSparseMatrices[wp.float32, wp.int32, vec7f] = BlockSparseMatrices(
                device=self.device,
                nzb_dtype=BlockDType[wp.float32](dtype=wp.float32, shape=(7,)),
                num_matrices=self.model.size.num_worlds,
            )
            jacobian_dims = list(zip(num_constraints.tolist(), (7 * num_bodies).tolist(), strict=True))

            # Determine number of nzb, per world and in total
            num_nzb = num_bodies.copy()  # nzb due to rigid body unit quaternion constraints
            jt_num_constraints = (constraint_full_to_red_map.reshape((-1, 6)) >= 0).sum(axis=1)
            jt_num_bodies = np.array(
                [1 if joint_bid_B[i] < 0 else 2 for i in range(self.data.dimensions.num_joints_tot)]
            )
            for wd_id in range(self.model.size.num_worlds):  # nzb due to joint constraints
                start = joints_offset[wd_id]
                end = start + num_joints[wd_id]
                num_nzb[wd_id] += (jt_num_constraints[start:end] * jt_num_bodies[start:end]).sum()
            first_nzb = np.concatenate(([0], num_nzb.cumsum()))
            num_nzb_tot = num_nzb.sum()

            # Symbolic assembly
            nzb_row = np.empty(num_nzb_tot, dtype=np.int32)
            nzb_col = np.empty(num_nzb_tot, dtype=np.int32)
            rb_nzb_id = np.empty(self.model.size.sum_of_num_bodies, dtype=np.int32)
            ct_nzb_id_base = np.full(6 * self.data.dimensions.num_joints_tot, -1, dtype=np.int32)
            ct_nzb_id_follower = np.full(6 * self.data.dimensions.num_joints_tot, -1, dtype=np.int32)
            for wd_id in range(self.model.size.num_worlds):
                start_nzb = first_nzb[wd_id]

                # Compute index, row and column of rigid body nzb
                start_rb = bodies_offset[wd_id]
                size_rb = num_bodies[wd_id]
                rb_ids = np.arange(size_rb)
                rb_nzb_id[start_rb : start_rb + size_rb] = start_nzb + rb_ids
                nzb_row[start_nzb : start_nzb + size_rb] = rb_ids
                nzb_col[start_nzb : start_nzb + size_rb] = 7 * rb_ids

                # Compute index, row and column of constraint nzb
                start_nzb += size_rb
                for jt_id_loc in range(num_joints[wd_id]):
                    jt_id_tot = jt_id_loc + joints_offset[wd_id]
                    has_base = joint_bid_B[jt_id_tot] >= 0
                    row_ids_full = constraint_full_to_red_map[6 * jt_id_tot : 6 * jt_id_tot + 6]
                    row_ids_red = [i for i in row_ids_full if i >= 0]
                    num_cts = len(row_ids_red)
                    if has_base:
                        nzb_id_base = ct_nzb_id_base[6 * jt_id_tot : 6 * jt_id_tot + 6]
                        nzb_id_base[row_ids_full >= 0] = np.arange(start_nzb, start_nzb + num_cts)
                        nzb_row[start_nzb : start_nzb + num_cts] = row_ids_red
                        base_id_loc = joint_bid_B[jt_id_tot] - bodies_offset[wd_id]
                        nzb_col[start_nzb : start_nzb + num_cts] = 7 * base_id_loc
                        start_nzb += num_cts
                    nzb_id_follower = ct_nzb_id_follower[6 * jt_id_tot : 6 * jt_id_tot + 6]
                    nzb_id_follower[row_ids_full >= 0] = np.arange(start_nzb, start_nzb + num_cts)
                    nzb_row[start_nzb : start_nzb + num_cts] = row_ids_red
                    follower_id_loc = joint_bid_F[jt_id_tot] - bodies_offset[wd_id]
                    nzb_col[start_nzb : start_nzb + num_cts] = 7 * follower_id_loc
                    start_nzb += num_cts

            # Transfer data to GPU
            self.data.problem.sparse_jacobian.finalize(jacobian_dims, num_nzb.tolist())
            self.data.problem.sparse_jacobian.dims.assign(jacobian_dims)
            assign_to_warp_int32_array(self.data.problem.sparse_jacobian.num_nzb, num_nzb)
            assign_to_warp_int32_array(
                self.data.problem.sparse_jacobian.nzb_coords, np.stack((nzb_row, nzb_col)).T.flatten()
            )
            with wp.ScopedDevice(self.device):
                self.data.dimensions.rb_nzb_id = to_warp_int32_array(rb_nzb_id)
                self.data.dimensions.ct_nzb_id_base = to_warp_int32_array(ct_nzb_id_base)
                self.data.dimensions.ct_nzb_id_follower = to_warp_int32_array(ct_nzb_id_follower)

            # Initialize Jacobian assembly kernel
            self._eval_joint_constraints_sparse_jacobian_kernel = make_eval_joint_constraints_sparse_jacobian_kernel(
                has_universal_joints
            )

            # Initialize Jacobian linear operator
            self.data.problem.sparse_jacobian_op = BlockSparseLinearOperators[wp.float32, wp.int32](
                self.data.problem.sparse_jacobian
            )

            # Compute flat-array offsets for the CG solver (uniform world dimensions)
            cg_vio = wp.from_numpy(
                np.arange(self.model.size.num_worlds, dtype=np.int32) * self.data.dimensions.num_states_max,
                device=self.device,
            )
            cg_total_vec_size = self.model.size.num_worlds * self.data.dimensions.num_states_max

            # Initialize preconditioner
            if (
                self.data.linear_system.preconditioner_type
                == ForwardKinematicsSolver.PreconditionerType.JACOBI_DIAGONAL
            ):
                self.data.linear_system.jacobian_diag_inv = wp.array(
                    dtype=wp.float32,
                    device=self.device,
                    shape=(self.model.size.num_worlds, self.data.dimensions.num_states_max),
                )
                preconditioner_op = BatchedLinearOperator.from_diagonal(
                    self.data.linear_system.jacobian_diag_inv.reshape((cg_total_vec_size,)),
                    self.data.dimensions.num_states,
                    cg_vio,
                    self.data.dimensions.num_states_max,
                )
            elif (
                self.data.linear_system.preconditioner_type
                == ForwardKinematicsSolver.PreconditionerType.JACOBI_BLOCK_DIAGONAL
            ):
                self.data.linear_system.inv_blocks_3 = wp.array(
                    dtype=wp.mat33f,
                    shape=(self.model.size.num_worlds, self.model.size.max_of_num_bodies),
                    device=self.device,
                )
                self.data.linear_system.inv_blocks_4 = wp.array(
                    dtype=wp.mat44f,
                    shape=(self.model.size.num_worlds, self.model.size.max_of_num_bodies),
                    device=self.device,
                )
                blockwise_gemv_2d = get_blockwise_diag_3_4_gemv_2d(
                    self.data.linear_system.inv_blocks_3,
                    self.data.linear_system.inv_blocks_4,
                    self.data.dimensions.num_states,
                )
                n_wd, n_st = self.model.size.num_worlds, self.data.dimensions.num_states_max

                def _blockwise_gemv_flat(x, y, world_active, alpha, beta):
                    blockwise_gemv_2d(x.reshape((n_wd, n_st)), y.reshape((n_wd, n_st)), world_active, alpha, beta)

                preconditioner_op = BatchedLinearOperator(
                    gemv_fn=_blockwise_gemv_flat,
                    n_worlds=self.model.size.num_worlds,
                    max_dim=self.data.dimensions.num_states_max,
                    active_dims=self.data.dimensions.num_states,
                    device=self.device,
                    dtype=wp.float32,
                    vio=cg_vio,
                    total_vec_size=cg_total_vec_size,
                )
            else:
                preconditioner_op = None

            # Initialize CG solver — wrap 2D gemv for flat 1D arrays
            n_wd, n_st = self.model.size.num_worlds, self.data.dimensions.num_states_max

            def _cg_gemv_flat(x, y, world_active, alpha, beta):
                self._eval_lhs_gemv(x.reshape((n_wd, n_st)), y.reshape((n_wd, n_st)), world_active, alpha, beta)

            def _cg_matvec_flat(x, y, world_active):
                self._eval_lhs_matvec(x.reshape((n_wd, n_st)), y.reshape((n_wd, n_st)), world_active)

            cg_op = BatchedLinearOperator(
                n_worlds=self.model.size.num_worlds,
                max_dim=self.data.dimensions.num_states_max,
                active_dims=self.data.dimensions.num_states,
                dtype=wp.float32,
                device=self.device,
                gemv_fn=_cg_gemv_flat,
                matvec_fn=_cg_matvec_flat,
                vio=cg_vio,
                total_vec_size=cg_total_vec_size,
            )
            self.data.linear_system.cg_atol = wp.array(
                dtype=wp.float32, shape=self.model.size.num_worlds, device=self.device
            )
            self.data.linear_system.cg_rtol = wp.array(
                dtype=wp.float32, shape=self.model.size.num_worlds, device=self.device
            )
            self.data.linear_system.cg_max_iter = wp.from_numpy(
                2 * self.data.dimensions.num_states.numpy(), dtype=wp.int32, device=self.device
            )
            self.linear_solver_cg = CGSolver(
                A=cg_op,
                active_dims=self.data.dimensions.num_states,
                Mi=preconditioner_op,
                atol=self.data.linear_system.cg_atol,
                rtol=self.data.linear_system.cg_rtol,
                maxiter=self.data.linear_system.cg_max_iter,
            )

        # Initialize continuous joint data (e.g. joint frames)
        self._update_joint_frames()
        self._update_axis_joint_frames()
        self._update_base_q_default()

    def validate_model_changed(self, flags: ModelFlags | int) -> None:
        """Validate FK structural invariants before model values are updated.

        Args:
            flags: Bitmask indicating which model properties changed.

        Raises:
            RuntimeError: If the effective set of joints that are actuated for FK changed.
            ValueError: If an FK actuation override is invalid.
        """
        if not flags & (ModelFlags.JOINT_DOF_PROPERTIES | ModelFlags.ACTUATOR_PROPERTIES):
            return
        joint_count = self.model.size.sum_of_num_joints
        if joint_count == 0:
            return

        self.data.joints._actuation_violations.fill_(joint_count)
        wp.launch(
            validate_fk_actuation_updates,
            dim=joint_count,
            inputs=[
                self.model.joints.act_type,
                self.model.joints.fk_act_flag,
                self.data.joints._built_actuated,
                self.data.joints._actuation_violations,
            ],
            device=self.device,
        )
        changed_joint, invalid_joint = self.data.joints._actuation_violations.numpy()
        if invalid_joint != joint_count:
            raise ValueError(f"Invalid FK actuation flag for joint {int(invalid_joint)}: expected -1, 0, or 1")
        if changed_joint != joint_count:
            raise RuntimeError(
                f"Changing the actuated vs passive status of joint {int(changed_joint)} for FK is not supported; "
                "recreate SolverKamino to apply the change."
            )

    def notify_model_changed(self, flags: ModelFlags | int) -> None:
        """Refresh FK-owned values after an in-place model update.

        Structural changes must be rejected by the owning solver before this
        method is called. Updates here preserve allocations and pointers.

        Args:
            flags: Bitmask indicating which model properties changed.
        """
        if flags & (ModelFlags.JOINT_PROPERTIES | ModelFlags.BODY_INERTIAL_PROPERTIES):
            self._update_joint_frames()

        if flags & (ModelFlags.JOINT_PROPERTIES | ModelFlags.BODY_PROPERTIES | ModelFlags.BODY_INERTIAL_PROPERTIES):
            self._update_axis_joint_frames()

        if flags & (ModelFlags.JOINT_PROPERTIES | ModelFlags.BODY_PROPERTIES | ModelFlags.BODY_INERTIAL_PROPERTIES):
            self._update_base_q_default()

    def _update_joint_frames(self) -> None:
        """Compute FK joint frames from the current Kamino model."""
        if self.data.dimensions.num_joints_tot == 0:
            return
        wp.launch(
            _compute_fk_joint_frames,
            dim=self.data.dimensions.num_joints_tot,
            inputs=[
                self.data.joints.source_id,
                self.model.joints.B_r_Bj,
                self.model.joints.F_r_Fj,
                self.model.joints.X_Bj,
                self.model.joints.X_Fj,
                self.data.joints.B_r_Bj,
                self.data.joints.F_r_Fj,
                self.data.joints.X_Bj,
                self.data.joints.X_Fj,
            ],
            device=self.device,
        )

    def _update_axis_joint_frames(self) -> None:
        """Compute synthetic axis-joint frames from the current model."""
        if self.data.dimensions.num_axis_joints == 0:
            return
        wp.launch(
            _compute_fk_axis_joint_frames,
            dim=self.data.dimensions.num_axis_joints,
            inputs=[
                self.data.joints.axis_joint_id,
                self.data.joints.axis_body_id,
                self.data.joints.axis_source_joint_0,
                self.data.joints.axis_source_joint_1,
                self.model.joints.bid_B,
                self.model.joints.B_r_Bj,
                self.model.joints.F_r_Fj,
                self.model.bodies.q_i_0,
                self.data.joints.X_Bj,
                self.data.joints.X_Fj,
            ],
            device=self.device,
        )

    def _update_base_q_default(self) -> None:
        """Compute default FK base poses from the current reference pose."""
        if self.model.size.num_worlds == 0:
            return
        get_base_q_from_joint_q_and_body_q(
            model=self.model,
            joint_q=self.model.joints.q_j_0,
            body_q=self.model.bodies.q_i_0,
            base_q=self.data.problem.base_q_default,
            world_mask=self.data.dimensions.all_worlds_mask,
        )

    ###
    # Internal evaluators (graph-capturable functions working on pre-allocated data)
    ###

    def _reset_state(
        self,
        body_q: wp.array[wp.transformf],
        world_mask: wp.array[wp.bool],
    ):
        """
        Internal function resetting the bodies state to the reference state stored in the model.
        """
        wp.launch(
            _reset_state,
            dim=(self.model.size.num_worlds, self.data.dimensions.num_states_max),
            inputs=[
                self.model.info.num_bodies,
                self.model.info.bodies_offset,
                self.model.bodies.q_i_0.view(wp.float32).flatten(),
                world_mask,
                body_q.view(wp.float32).flatten(),
            ],
            device=self.device,
        )

    def _reset_state_base_q(
        self,
        body_q: wp.array[wp.transformf],
        base_q: wp.array[wp.transformf],
        world_mask: wp.array[wp.bool],
    ):
        """
        Internal function resetting the bodies state to a rigid transformation of the reference state,
        computed so that the base body is aligned on its prescribed pose.
        """
        wp.launch(
            _reset_state_base_q,
            dim=(self.model.size.num_worlds, self.model.size.max_of_num_bodies),
            inputs=[
                self.data.joints.base_joint_id,
                base_q,
                self.data.joints.bid_F,
                self.data.joints.X_Bj,
                self.data.joints.X_Fj,
                self.data.joints.B_r_Bj,
                self.data.joints.F_r_Fj,
                self.model.info.num_bodies,
                self.model.info.bodies_offset,
                self.model.bodies.q_i_0,
                world_mask,
                body_q,
            ],
            device=self.device,
        )

    def _eval_actuator_coords(
        self,
        body_q: wp.array[wp.transformf],
        actuator_q: wp.array[wp.float32],
        actuator_q_ref: wp.array[wp.float32] | None = None,
    ):
        """
        Internal evaluator evaluating effective actuator coordinates based on body poses,
        with 2 Pi / quaternion sign correction w.r.t. reference coordinates if provided.
        """
        # Extract current actuator coordinates
        wp.launch(
            _eval_actuator_coords,
            dim=(self.model.size.num_worlds, self.data.dimensions.num_joints_max),
            inputs=[
                self.data.dimensions.num_joints,
                self.data.dimensions.joints_offset,
                self.data.joints.dof_type,
                self.data.joints.bid_B,
                self.data.joints.bid_F,
                self.data.joints.X_Bj,
                self.data.joints.X_Fj,
                self.data.joints.B_r_Bj,
                self.data.joints.F_r_Fj,
                body_q,
                self.data.dimensions.actuated_coords_offset,
                actuator_q,
            ],
            device=self.device,
        )
        # Correct w.r.t. reference coordinates
        if actuator_q_ref is not None:
            wp.launch(
                _correct_actuator_coords,
                dim=(self.data.dimensions.num_joints_tot,),
                inputs=[
                    self.data.dimensions.actuated_coords_offset,
                    self.data.joints.dof_type,
                    actuator_q_ref,
                    actuator_q,
                ],
                device=self.device,
            )

    def _initialize_incremental_solve(self, body_q: wp.array[wp.transformf]):
        """
        Internal function running all necessary precomputations for the incremental solve.
        Assumes without check that data related to incremental solve is allocated.
        """
        # Extract current actuator coordinates, and correct w.r.t. target coordinates
        self._eval_actuator_coords(body_q, self.data.problem.actuator_q_prev, self.data.problem.actuator_q_next)
        # Compute necessary number of Newton steps, before the incremental target matches the true target
        self.data.gauss_newton.min_iterations.zero_()
        wp.launch_tiled(
            self._eval_min_num_iterations_kernel,
            dim=(self.model.size.num_worlds, self.data.dimensions.num_tiles_coords),
            block_dim=get_block_dim(self.data.dimensions.tile_size_coords),
            inputs=[
                self.data.dimensions.world_actuated_coords_offset,
                self.data.problem.actuator_q_prev,
                self.data.problem.actuator_q_next,
                self.data.gauss_newton.delta_q_max,
                self.data.gauss_newton.min_iterations,
            ],
            device=self.device,
        )

        # Initialize Jacobian update masks
        if self.config.use_regularization:
            wp.launch(
                _initialize_jacobian_update_masks,
                dim=(self.model.size.num_worlds,),
                inputs=[
                    self.data.gauss_newton.mask,
                    self.data.gauss_newton.min_iterations,
                    self.data.gauss_newton.jacobian_early_update_mask,
                    self.data.gauss_newton.jacobian_late_update_mask,
                ],
                device=self.device,
            )

    def _eval_target_actuator_q(
        self,
        base_q_model: wp.array[wp.transformf],
        actuator_q_model: wp.array[wp.float32],
        actuator_q_next: wp.array[wp.float32],
    ):
        """
        Internal evaluator, converting actuator and base coordinates of the main model, to actuator
        coordinates of the FK model.
        """
        wp.launch(
            _eval_fk_actuated_dofs_or_coords,
            dim=(1, self.data.dimensions.num_actuated_coords),
            inputs=[
                base_q_model.view(wp.float32).reshape((1, 7 * self.model.size.num_worlds)),
                actuator_q_model.reshape((1, actuator_q_model.shape[0])),
                self.data.dimensions.actuated_coords_map,
                actuator_q_next.reshape((1, actuator_q_next.shape[0])),
            ],
            device=self.device,
        )

    def _update_incremental_target_actuator_q(
        self,
        iteration: wp.array[wp.int32],
        world_mask: wp.array[wp.bool],
    ):
        """
        Internal evaluator, updating the incremental target for actuator coordinates by interpolating
        between previous and next actuator coordinates, based on the current Newton iteration.
        """
        wp.launch(
            _eval_incremental_target_actuator_coords,
            dim=(self.model.size.num_worlds, self.data.dimensions.num_actuated_coords_max),
            inputs=[
                self.data.dimensions.world_actuated_coords_offset,
                self.data.problem.actuator_q_prev,
                self.data.problem.actuator_q_next,
                self.data.gauss_newton.delta_q_max,
                iteration,
                world_mask,
                self.data.problem.actuator_q_curr,
            ],
            device=self.device,
        )

    def _eval_target_relative_transformations(
        self,
        actuator_q: wp.array[wp.float32],
        target_rel_transforms: wp.array[wp.transformf],
        world_mask: wp.array[wp.bool],
    ):
        """
        Internal evaluator for target relative transformations, from actuated coordinates of the FK model.
        """
        wp.launch(
            _eval_target_relative_transformations,
            dim=(self.data.dimensions.num_joints_tot,),
            inputs=[
                self.data.joints.dof_type,
                self.data.joints.act_type,
                self.data.dimensions.actuated_coords_offset,
                self.data.joints.X_Bj,
                self.data.joints.X_Fj,
                actuator_q,
                self.config.use_incremental_solve,  # Incremental solve may result in non-unit quaternions
                self.data.joints.world_id,
                world_mask,
                target_rel_transforms,
            ],
            device=self.device,
        )

    def _eval_kinematic_constraints(
        self,
        body_q: wp.array[wp.transformf],
        target_rel_transforms: wp.array[wp.transformf],
        world_mask: wp.array[wp.bool],
        constraints: wp.array2d[wp.float32],
    ):
        """
        Internal evaluator for the kinematic constraints vector, from body poses and target relative transforms
        """

        # Evaluate unit norm quaternion constraints
        wp.launch(
            _eval_unit_quaternion_constraints,
            dim=(self.model.size.num_worlds, self.model.size.max_of_num_bodies),
            inputs=[self.model.info.num_bodies, self.model.info.bodies_offset, body_q, world_mask, constraints],
            device=self.device,
        )
        # Evaluate joint constraints
        wp.launch(
            self._eval_joint_constraints_kernel,
            dim=(self.model.size.num_worlds, self.data.dimensions.num_joints_max),
            inputs=[
                self.data.dimensions.num_joints,
                self.data.dimensions.joints_offset,
                self.data.joints.dof_type,
                self.data.joints.act_type,
                self.data.joints.bid_B,
                self.data.joints.bid_F,
                self.data.joints.X_Bj,
                self.data.joints.X_Fj,
                self.data.joints.B_r_Bj,
                self.data.joints.F_r_Fj,
                body_q,
                target_rel_transforms,
                self.data.dimensions.constraint_full_to_red_map,
                world_mask,
                constraints,
            ],
            device=self.device,
        )

    def _eval_max_residual(
        self,
        constraints: wp.array2d[wp.float32],
        gradient: wp.array2d[wp.float32],
        max_residual: wp.array[wp.float32],
    ):
        """
        Internal evaluator for the maximal absolute residual in each world, from either the constraints
        vector (by default) or the gradient vector (if regularization is enabled).

        Indeed, if a regularizer is added to the constraints squared norm objective, we cannot expect
        Gauss-Newton to converge to zero constraints anymore.
        """
        max_residual.zero_()
        if self.config.use_regularization:
            wp.launch_tiled(
                self._eval_max_residual_kernel,
                dim=(self.model.size.num_worlds, self.data.dimensions.num_tiles_vrs_1d),
                inputs=[gradient, max_residual],
                block_dim=get_block_dim(self.data.dimensions.tile_size_vrs_1d),
                device=self.device,
            )
        else:
            wp.launch_tiled(
                self._eval_max_residual_kernel,
                dim=(self.model.size.num_worlds, self.data.dimensions.num_tiles_cts_1d),
                inputs=[constraints, max_residual],
                block_dim=get_block_dim(self.data.dimensions.tile_size_cts_1d),
                device=self.device,
            )

    def _eval_kinematic_constraints_jacobian(
        self,
        body_q: wp.array[wp.transformf],
        target_rel_transforms: wp.array[wp.transformf],
        world_mask: wp.array[wp.bool],
        constraints_jacobian: wp.array3d[wp.float32],
    ):
        """
        Internal evaluator for the kinematic constraints Jacobian with respect to body poses, from body poses
        and target relative transforms
        """

        # Evaluate unit norm quaternion constraints Jacobian
        wp.launch(
            _eval_unit_quaternion_constraints_jacobian,
            dim=(self.model.size.num_worlds, self.model.size.max_of_num_bodies),
            inputs=[
                self.model.info.num_bodies,
                self.model.info.bodies_offset,
                body_q,
                world_mask,
                constraints_jacobian,
            ],
            device=self.device,
        )

        # Evaluate joint constraints Jacobian
        wp.launch(
            self._eval_joint_constraints_jacobian_kernel,
            dim=(self.model.size.num_worlds, self.data.dimensions.num_joints_max),
            inputs=[
                self.data.dimensions.num_joints,
                self.data.dimensions.joints_offset,
                self.model.info.bodies_offset,
                self.data.joints.dof_type,
                self.data.joints.act_type,
                self.data.joints.bid_B,
                self.data.joints.bid_F,
                self.data.joints.X_Bj,
                self.data.joints.X_Fj,
                self.data.joints.B_r_Bj,
                self.data.joints.F_r_Fj,
                body_q,
                target_rel_transforms,
                self.data.dimensions.constraint_full_to_red_map,
                world_mask,
                constraints_jacobian,
            ],
            device=self.device,
        )

    def _assemble_sparse_jacobian(
        self,
        body_q: wp.array[wp.transformf],
        target_rel_transforms: wp.array[wp.transformf],
        world_mask: wp.array[wp.bool],
    ):
        """
        Internal evaluator for the sparse kinematic constraints Jacobian with respect to body poses, from body poses
        and target relative transforms
        """

        self.data.problem.sparse_jacobian.zero(world_mask)

        # Evaluate unit norm quaternion constraints Jacobian
        wp.launch(
            _eval_unit_quaternion_constraints_sparse_jacobian,
            dim=(self.model.size.num_worlds, self.model.size.max_of_num_bodies),
            inputs=[
                self.model.info.num_bodies,
                self.model.info.bodies_offset,
                body_q,
                self.data.dimensions.rb_nzb_id,
                world_mask,
                self.data.problem.sparse_jacobian.nzb_values,
            ],
            device=self.device,
        )

        # Evaluate joint constraints Jacobian
        wp.launch(
            self._eval_joint_constraints_sparse_jacobian_kernel,
            dim=(self.model.size.num_worlds, self.data.dimensions.num_joints_max),
            inputs=[
                self.data.dimensions.num_joints,
                self.data.dimensions.joints_offset,
                self.model.info.bodies_offset,
                self.data.joints.dof_type,
                self.data.joints.act_type,
                self.data.joints.bid_B,
                self.data.joints.bid_F,
                self.data.joints.X_Bj,
                self.data.joints.X_Fj,
                self.data.joints.B_r_Bj,
                self.data.joints.F_r_Fj,
                body_q,
                target_rel_transforms,
                self.data.dimensions.ct_nzb_id_base,
                self.data.dimensions.ct_nzb_id_follower,
                world_mask,
                self.data.problem.sparse_jacobian.nzb_values,
            ],
            device=self.device,
        )

    def _update_jacobian(
        self,
        body_q: wp.array[wp.transformf],
        target_rel_transforms: wp.array[wp.transformf],
        world_mask: wp.array[wp.bool],
    ):
        """
        Convenience function updating the constraints Jacobian, given body poses and target relative
        transforms
        Solver configuration (sparsity, regularization) are taken into account.
        """
        if self.config.use_sparsity:
            self._assemble_sparse_jacobian(body_q, target_rel_transforms, world_mask)
        else:
            self._eval_kinematic_constraints_jacobian(
                body_q, target_rel_transforms, world_mask, self.data.problem.jacobian
            )

    def _update_lhs(
        self,
        world_mask: wp.array[wp.bool],
    ):
        """
        Convenience function updating the system left-hand side (J^T * J + regularization, optionally),
        using the lastly assembled Jacobian
        Solver configuration (sparsity, regularization) are taken into account.
        """
        if self.config.use_sparsity:
            return  # No lhs to assemble for the sparse case (represented implicitly as an operator)

        wp.launch_tiled(
            self._eval_jacobian_T_jacobian_kernel,
            dim=(
                self.model.size.num_worlds,
                self.data.dimensions.num_tiles_vrs_2d,
                self.data.dimensions.num_tiles_vrs_2d,
            ),
            inputs=[
                self.data.problem.jacobian,
                self.data.dimensions.tile_sparsity_pattern,
                world_mask,
                self.data.linear_system.lhs,
            ],
            block_dim=32,
            device=self.device,
        )
        if self.config.use_regularization:
            wp.launch(
                _add_regularizer_to_diagonal,
                dim=(self.model.size.num_worlds, self.data.dimensions.num_states_max),
                inputs=[
                    self.config.regularization_weight,
                    self.data.dimensions.num_states,
                    world_mask,
                    self.data.linear_system.lhs,
                ],
                device=self.device,
            )

    def _update_gradient(
        self,
        body_q: wp.array[wp.transformf],
        world_mask: wp.array[wp.bool],
    ):
        """
        Convenience function updating the objective gradient (J^T * constraints + regularization, optionally),
        given body poses and using the lastly assembled Jacobian and constraints.
        Solver configuration (sparsity, regularization) are taken into account.
        """
        if self.config.use_sparsity:
            self.data.problem.sparse_jacobian_op.matvec_transpose(
                self.data.problem.constraints, self.data.gauss_newton.grad, world_mask
            )
        else:
            wp.launch_tiled(
                self._eval_jacobian_T_constraints_kernel,
                dim=(self.model.size.num_worlds, self.data.dimensions.num_tiles_vrs_2d, 1),
                inputs=[
                    self.data.problem.jacobian,
                    self.data.problem.constraints.reshape(
                        (self.model.size.num_worlds, self.data.dimensions.num_constraints_max, 1)
                    ),
                    self.data.dimensions.tile_sparsity_pattern,
                    world_mask,
                    self.data.gauss_newton.grad.reshape(
                        (self.model.size.num_worlds, self.data.dimensions.num_states_max, 1)
                    ),
                ],
                block_dim=32,
                device=self.device,
            )

        if self.config.use_regularization:
            wp.launch(
                _eval_regularizer_gradient,
                dim=(self.model.size.num_worlds, self.data.dimensions.num_states_max),
                inputs=[
                    self.model.info.num_bodies,
                    self.model.info.bodies_offset,
                    self.config.regularization_weight,
                    body_q.view(wp.float32).flatten(),
                    self.data.problem.body_q_ref.view(wp.float32).flatten(),
                    world_mask,
                    self.data.gauss_newton.grad,
                ],
                device=self.device,
            )

    def _eval_lhs_gemv(
        self,
        x: wp.array2d[wp.float32],
        y: wp.array2d[wp.float32],
        world_mask: wp.array[wp.bool],
        alpha: wp.float32,
        beta: wp.float32,
    ):
        """
        Internal evaluator for y = alpha * lhs * x + beta * y, using the assembled sparse Jacobian J,
        and with lhs = J^T * J (plus optionally the regularizer Hessian reg_weight * I)
        """
        self.data.problem.sparse_jacobian_op.matvec(x, self.data.linear_system.jacobian_times_vector, world_mask)
        self.data.problem.sparse_jacobian_op.matvec_transpose(
            self.data.linear_system.jacobian_times_vector, self.data.linear_system.lhs_times_vector, world_mask
        )
        if self.config.use_regularization:
            wp.launch(
                _eval_linear_combination,
                dim=(self.model.size.num_worlds, self.data.dimensions.num_states_max),
                inputs=[
                    1.0,
                    self.data.linear_system.lhs_times_vector,
                    self.config.regularization_weight,
                    x,
                    self.data.dimensions.num_constraints,
                    world_mask,
                    self.data.linear_system.lhs_times_vector,
                ],
                device=self.device,
            )
        wp.launch(
            _eval_linear_combination,
            dim=(self.model.size.num_worlds, self.data.dimensions.num_states_max),
            inputs=[
                alpha,
                self.data.linear_system.lhs_times_vector,
                beta,
                y,
                self.data.dimensions.num_constraints,
                world_mask,
                y,
            ],
            device=self.device,
        )

    def _eval_lhs_matvec(
        self,
        x: wp.array2d[wp.float32],
        y: wp.array2d[wp.float32],
        world_mask: wp.array[wp.bool],
    ):
        """
        Internal evaluator for y = lhs * x, using the assembled sparse Jacobian J,
        and with lhs = J^T * J (plus optionally the regularizer Hessian reg_weight * I)
        """
        self.data.problem.sparse_jacobian_op.matvec(x, self.data.linear_system.jacobian_times_vector, world_mask)
        self.data.problem.sparse_jacobian_op.matvec_transpose(
            self.data.linear_system.jacobian_times_vector, y, world_mask
        )
        if self.config.use_regularization:
            wp.launch(
                _eval_linear_combination,
                dim=(self.model.size.num_worlds, self.data.dimensions.num_states_max),
                inputs=[
                    1.0,
                    y,
                    self.config.regularization_weight,
                    x,
                    self.data.dimensions.num_constraints,
                    world_mask,
                    y,
                ],
                device=self.device,
            )

    def _eval_merit_function(
        self,
        constraints: wp.array2d[wp.float32],
        merit_function: wp.array[wp.float32],
        body_q: wp.array[wp.transformf] | None = None,
    ):
        """
        Internal evaluator for the line search merit function, i.e. the least-squares error
        1/2 * ||C||^2, plus optionally the regularizer 1/2 * reg_weight * ||s - s_ref||^2,
        from the constraints vector C, in each world
        """
        merit_function.zero_()
        wp.launch_tiled(
            self._eval_merit_function_kernel,
            dim=(self.model.size.num_worlds, self.data.dimensions.num_tiles_cts_1d),
            inputs=[constraints, merit_function],
            block_dim=get_block_dim(self.data.dimensions.tile_size_cts_1d),
            device=self.device,
        )
        if self.config.use_regularization and body_q is not None:
            wp.launch_tiled(
                self._eval_regularizer_kernel,
                dim=(self.model.size.num_worlds, self.data.dimensions.num_tiles_vrs_1d),
                inputs=[
                    self.model.info.bodies_offset,
                    self.config.regularization_weight,
                    body_q.view(wp.float32).flatten(),
                    self.data.problem.body_q_ref.view(wp.float32).flatten(),
                    merit_function,
                ],
                block_dim=get_block_dim(self.data.dimensions.tile_size_vrs_1d),
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
        and the gradient in state space (= dC_ds^T * C, plus optionally reg_weight * (s - s_ref)).
        This is simply the dot product between these two vectors.
        """
        error_grad.zero_()
        wp.launch_tiled(
            self._eval_merit_function_gradient_kernel,
            dim=(self.model.size.num_worlds, self.data.dimensions.num_tiles_vrs_1d),
            inputs=[step, grad, error_grad],
            block_dim=get_block_dim(self.data.dimensions.tile_size_vrs_1d),
            device=self.device,
        )

    def _run_line_search_iteration(self, body_q: wp.array[wp.transformf]):
        """
        Internal function running one iteration of line search, checking the Armijo sufficient descent condition
        """
        # Eval stepped state
        wp.launch(
            _eval_stepped_state,
            dim=(self.model.size.num_worlds, self.data.dimensions.num_states_max),
            inputs=[
                self.model.info.num_bodies,
                self.model.info.bodies_offset,
                body_q.view(wp.float32).flatten(),
                self.data.line_search.alpha,
                self.data.gauss_newton.step,
                self.data.line_search.mask,
                self.data.line_search.body_q_alpha.view(wp.float32).flatten(),
            ],
            device=self.device,
        )

        # Evaluate new constraints and merit function (least squares norm of constraints)
        self._eval_kinematic_constraints(
            self.data.line_search.body_q_alpha,
            self.data.problem.target_rel_transforms,
            self.data.line_search.mask,
            self.data.problem.constraints,
        )
        self._eval_merit_function(
            self.data.problem.constraints, self.data.line_search.val_alpha, self.data.line_search.body_q_alpha
        )

        # Check decrease and update step
        self.data.line_search.loop_condition.zero_()
        wp.launch(
            _line_search_check,
            dim=(self.model.size.num_worlds,),
            inputs=[
                self.data.line_search.val_0,
                self.data.line_search.grad_0,
                self.data.line_search.alpha,
                self.data.line_search.val_alpha,
                self.data.line_search.iteration,
                self.config.max_line_search_iterations,
                self.data.line_search.success,
                self.data.line_search.mask,
                self.data.line_search.loop_condition,
            ],
            device=self.device,
        )

    def _update_cg_tolerance(
        self,
        residual_norm: wp.array[wp.float32],
        world_mask: wp.array[wp.bool],
    ):
        """
        Internal function heuristically adapting the CG tolerance based on the current constraint/gradient residual
        (starting with a loose tolerance, and tightening it as we converge)
        """
        wp.launch(
            _update_cg_tolerance_kernel,
            dim=(self.model.size.num_worlds,),
            inputs=[residual_norm, world_mask, self.data.linear_system.cg_atol, self.data.linear_system.cg_rtol],
            device=self.device,
        )

    def _run_newton_iteration(self, body_q: wp.array[wp.transformf]):
        """
        Internal function running one iteration of Gauss-Newton. Assumes the constraints vector to be already
        up-to-date (because we will already have checked convergence before the first loop iteration)
        """
        # Update actuator_q and kinematic constraints, for incremental solve
        if self.config.use_incremental_solve:
            self._update_incremental_target_actuator_q(self.data.gauss_newton.iteration, self.data.gauss_newton.mask)
            self._eval_target_relative_transformations(
                self.data.problem.actuator_q_curr,
                self.data.problem.target_rel_transforms,
                self.data.gauss_newton.mask,
            )
            self._eval_kinematic_constraints(
                body_q,
                self.data.problem.target_rel_transforms,
                self.data.gauss_newton.mask,
                self.data.problem.constraints,
            )

        # Evaluate constraints Jacobian if needed
        if not self.config.use_regularization:
            self._update_jacobian(body_q, self.data.problem.target_rel_transforms, self.data.gauss_newton.mask)
        elif self.config.use_incremental_solve:
            self._update_jacobian(
                body_q, self.data.problem.target_rel_transforms, self.data.gauss_newton.jacobian_late_update_mask
            )

        # Evaluate Gauss-Newton left-hand side (J^T * J) if needed, and right-hand side (-J^T * C)
        self._update_lhs(self.data.gauss_newton.mask)
        if not self.config.use_regularization:
            self._update_gradient(body_q, self.data.gauss_newton.mask)
        elif self.config.use_incremental_solve:
            self._update_gradient(body_q, self.data.gauss_newton.jacobian_late_update_mask)
        wp.launch(
            _eval_rhs,
            dim=(self.model.size.num_worlds, self.data.dimensions.num_states_max),
            inputs=[self.data.gauss_newton.grad, self.data.linear_system.rhs],
            device=self.device,
        )

        # Compute step (system solve)
        if self.config.use_sparsity:
            offset = self.config.regularization_weight if self.config.use_regularization else 0.0
            if (
                self.data.linear_system.preconditioner_type
                == ForwardKinematicsSolver.PreconditionerType.JACOBI_DIAGONAL
            ):
                block_sparse_ATA_inv_diagonal_2d(
                    self.data.problem.sparse_jacobian,
                    self.data.linear_system.jacobian_diag_inv,
                    self.data.gauss_newton.mask,
                    diag_offset=offset,
                )
            elif (
                self.data.linear_system.preconditioner_type
                == ForwardKinematicsSolver.PreconditionerType.JACOBI_BLOCK_DIAGONAL
            ):
                block_sparse_ATA_blockwise_3_4_inv_diagonal_2d(
                    self.data.problem.sparse_jacobian,
                    self.data.linear_system.inv_blocks_3,
                    self.data.linear_system.inv_blocks_4,
                    self.data.gauss_newton.mask,
                    diag_offset=offset,
                )

            self.data.gauss_newton.step.zero_()
            if self.config.use_adaptive_cg_tolerance:
                self._update_cg_tolerance(self.data.gauss_newton.max_residual, self.data.gauss_newton.mask)
            else:
                self.data.linear_system.cg_atol.fill_(1e-8)
                self.data.linear_system.cg_rtol.fill_(1e-8)
            self.linear_solver_cg.solve(
                self.data.linear_system.rhs.reshape((-1,)),
                self.data.gauss_newton.step.reshape((-1,)),
                world_active=self.data.gauss_newton.mask,
            )
        else:
            self.linear_solver_llt.factorize(
                self.data.linear_system.lhs, self.data.dimensions.num_states, self.data.gauss_newton.mask
            )
            self.linear_solver_llt.solve(
                self.data.linear_system.rhs.reshape(
                    (self.model.size.num_worlds, self.data.dimensions.num_states_max, 1)
                ),
                self.data.gauss_newton.step.reshape(
                    (self.model.size.num_worlds, self.data.dimensions.num_states_max, 1)
                ),
                self.data.gauss_newton.mask,
            )

        # Line search
        self.data.line_search.iteration.zero_()
        self.data.line_search.success.zero_()
        wp.copy(self.data.line_search.mask, self.data.gauss_newton.mask)
        self.data.line_search.loop_condition.fill_(1)
        self._eval_merit_function(self.data.problem.constraints, self.data.line_search.val_0, body_q)
        self._eval_merit_function_gradient(
            self.data.gauss_newton.step, self.data.gauss_newton.grad, self.data.line_search.grad_0
        )
        self.data.line_search.alpha.fill_(1.0)
        wp.capture_while(self.data.line_search.loop_condition, lambda: self._run_line_search_iteration(body_q))

        # Apply line search step and update max constraint
        wp.launch(
            _apply_line_search_step,
            dim=(self.model.size.num_worlds, self.model.size.max_of_num_bodies),
            inputs=[
                self.model.info.num_bodies,
                self.model.info.bodies_offset,
                self.data.line_search.body_q_alpha,
                self.data.line_search.success,
                body_q,
            ],
            device=self.device,
        )
        if self.config.use_regularization:
            mask = (
                self.data.gauss_newton.jacobian_early_update_mask
                if self.config.use_incremental_solve
                else self.data.gauss_newton.mask
            )
            self._update_jacobian(body_q, self.data.problem.target_rel_transforms, mask)
            self._update_gradient(body_q, mask)
        self._eval_max_residual(
            self.data.problem.constraints, self.data.gauss_newton.grad, self.data.gauss_newton.max_residual
        )

        # Check convergence
        self.data.gauss_newton.loop_condition.zero_()
        wp.launch(
            _newton_check,
            dim=(self.model.size.num_worlds,),
            inputs=[
                self.data.gauss_newton.max_residual,
                self.config.tolerance,
                self.data.gauss_newton.iteration,
                self.data.gauss_newton.min_iterations,
                self.config.max_newton_iterations,
                self.data.line_search.success,
                self.data.gauss_newton.success,
                self.data.gauss_newton.mask,
                self.data.gauss_newton.loop_condition,
                self.data.gauss_newton.jacobian_early_update_mask,
                self.data.gauss_newton.jacobian_late_update_mask,
            ],
            device=self.device,
        )

    def _solve_for_body_velocities(
        self,
        target_rel_transforms: wp.array[wp.transformf],
        base_u: wp.array2d[wp.spatial_vectorf],
        actuator_u: wp.array2d[wp.float32],
        body_q: wp.array[wp.transformf],
        body_u: wp.array2d[wp.spatial_vectorf],
        world_mask: wp.array[wp.bool],
    ):
        """
        Internal function solving for body velocities, so that constraint velocities are zero,
        except at actuated dofs and at the base joint, where they must match prescribed velocities.

        Processes a batch of batch_size velocity vectors in parallel.
        """
        # Retrieve temporary buffers
        batch_size = actuator_u.shape[0]
        velocity_data = self.data.velocity_solve.get(batch_size)
        if velocity_data is None:
            raise ValueError(
                f"Velocity buffers for batch_size={batch_size} are not allocated; "
                "call request_velocity_solve_batch_size() before solving."
            )
        fk_actuator_u = velocity_data.fk_actuator_u
        target_cts_u = velocity_data.target_cts_u
        rhs = velocity_data.rhs
        body_q_dot = velocity_data.body_q_dot

        # Compute actuator_u of fk model with modified joints
        wp.launch(
            _eval_fk_actuated_dofs_or_coords,
            dim=(batch_size, self.data.dimensions.num_actuated_dofs),
            inputs=[
                base_u.view(wp.float32).reshape((base_u.shape[0], 6 * self.model.size.num_worlds)),
                actuator_u,
                self.data.dimensions.actuated_dofs_map,
                fk_actuator_u,
            ],
            device=self.device,
        )

        # Compute target constraint velocities (prescribed for actuated dofs, zero for passive constraints)
        self._eval_actuator_coords(body_q, self.data.problem.actuator_q_next)
        target_cts_u.zero_()
        wp.launch(
            _eval_target_constraint_velocities,
            dim=(batch_size, self.model.size.num_worlds, self.data.dimensions.num_joints_max),
            inputs=[
                self.data.dimensions.num_joints,
                self.data.dimensions.joints_offset,
                self.data.joints.dof_type,
                self.data.joints.act_type,
                self.data.dimensions.actuated_coords_offset,
                self.data.dimensions.actuated_dofs_offset,
                self.data.dimensions.constraint_full_to_red_map,
                self.data.problem.actuator_q_next,
                fk_actuator_u,
                world_mask,
                target_cts_u,
            ],
            device=self.device,
        )
        if self.data.joints.has_universal_actuators:
            wp.launch(
                _correct_universal_constraint_velocities,
                dim=(batch_size, self.model.size.num_worlds, self.data.dimensions.num_joints_max),
                inputs=[
                    self.data.dimensions.num_joints,
                    self.data.dimensions.joints_offset,
                    self.data.joints.dof_type,
                    self.data.joints.act_type,
                    self.data.joints.bid_B,
                    self.data.joints.bid_F,
                    self.data.joints.X_Bj,
                    self.data.joints.X_Fj,
                    self.data.dimensions.constraint_full_to_red_map,
                    body_q,
                    world_mask,
                    target_cts_u,
                ],
                device=self.device,
            )

        # Update constraints Jacobian
        self._update_jacobian(body_q, target_rel_transforms, world_mask)

        # Evaluate system left-hand side (for the dense solver) and right-hand side
        # These are J^T * J (+ regularizer Hessian), and J^T * targets_cts_u
        self._update_lhs(world_mask)
        if self.config.use_sparsity:
            self.data.problem.sparse_jacobian_op.matvec_transpose(
                target_cts_u.reshape((self.model.size.num_worlds, self.data.dimensions.num_constraints_max)),
                rhs.reshape((self.model.size.num_worlds, self.data.dimensions.num_states_max)),
                world_mask,
            )
        else:
            wp.launch_tiled(
                self._eval_jacobian_T_constraints_kernel,
                dim=(self.model.size.num_worlds, self.data.dimensions.num_tiles_vrs_2d, batch_size),
                inputs=[
                    self.data.problem.jacobian,
                    target_cts_u,
                    self.data.dimensions.tile_sparsity_pattern,
                    world_mask,
                    rhs,
                ],
                block_dim=32,
                device=self.device,
            )

        # Compute body velocities (system solve)
        if self.config.use_sparsity:
            offset = self.config.regularization_weight if self.config.use_regularization else 0.0
            if (
                self.data.linear_system.preconditioner_type
                == ForwardKinematicsSolver.PreconditionerType.JACOBI_DIAGONAL
            ):
                block_sparse_ATA_inv_diagonal_2d(
                    self.data.problem.sparse_jacobian,
                    self.data.linear_system.jacobian_diag_inv,
                    world_mask,
                    diag_offset=offset,
                )
            elif (
                self.data.linear_system.preconditioner_type
                == ForwardKinematicsSolver.PreconditionerType.JACOBI_BLOCK_DIAGONAL
            ):
                block_sparse_ATA_blockwise_3_4_inv_diagonal_2d(
                    self.data.problem.sparse_jacobian,
                    self.data.linear_system.inv_blocks_3,
                    self.data.linear_system.inv_blocks_4,
                    world_mask,
                    diag_offset=offset,
                )
            body_q_dot.zero_()
            self.data.linear_system.cg_atol.fill_(1e-8)
            self.data.linear_system.cg_rtol.fill_(1e-8)
            self.linear_solver_cg.solve(rhs.reshape((-1,)), body_q_dot.reshape((-1,)), world_active=world_mask)
        else:
            self.linear_solver_llt.factorize(self.data.linear_system.lhs, self.data.dimensions.num_states, world_mask)
            self.linear_solver_llt.solve(rhs, body_q_dot, world_mask)
        wp.launch(
            _eval_body_velocities,
            dim=(batch_size, self.model.size.num_worlds, self.model.size.max_of_num_bodies),
            inputs=[self.model.info.num_bodies, self.model.info.bodies_offset, body_q, body_q_dot, world_mask, body_u],
            device=self.device,
        )

    ###
    # Exposed functions (overall solve_fk() function + constraints (Jacobian) evaluators for debugging)
    ###

    def eval_target_relative_transforms(
        self, actuator_q: wp.array[wp.float32], base_q: wp.array[wp.transformf] | None = None
    ) -> wp.array[wp.transformf]:
        """
        Evaluates and returns the target relative transforms (an intermediary quantity needed for the
        kinematic constraints/Jacobian evaluation) for a model given actuated coordinates, and optionally
        the base pose (the default base pose is used if not provided).

        Args:
            actuator_q: Actuated joint coordinates, with shape ``(num_fk_actuated_coords,)``.
            base_q: Base pose per world, with shape ``(num_worlds,)``. Defaults to the model's base pose.

        Raises:
            ValueError: If any input array is not on this solver's device.

        Returns:
            Target relative transforms, with shape ``(num_fk_joints)``.
        """
        if actuator_q.device != self.device:
            raise ValueError("actuator_q must be on the solver's device")
        if base_q is not None and base_q.device != self.device:
            raise ValueError("base_q must be on the solver's device")

        if base_q is None:
            base_q = self.data.problem.base_q_default

        # Convert base_q, actuator_q from the main model to actuator_q for the FK model
        actuator_q_fk = wp.array(
            dtype=wp.float32, shape=(self.data.dimensions.num_actuated_coords,), device=self.device
        )
        self._eval_target_actuator_q(base_q, actuator_q, actuator_q_fk)

        # Evaluate target relative transformations
        target_rel_transforms = wp.array(
            dtype=wp.transformf, shape=(self.data.dimensions.num_joints_tot,), device=self.device
        )
        self._eval_target_relative_transformations(
            actuator_q_fk, target_rel_transforms, self.data.dimensions.all_worlds_mask
        )

        return target_rel_transforms

    def eval_kinematic_constraints(
        self, body_q: wp.array[wp.transformf], target_rel_transforms: wp.array[wp.transformf]
    ) -> wp.array2d[wp.float32]:
        """
        Evaluates and returns the kinematic constraints vector given the body poses and the position
        control transformations.

        Args:
            body_q: Body poses, with shape ``(num_bodies,)``.
            target_rel_transforms: Target relative transforms per joint, with shape ``(num_fk_joints,)``.

        Raises:
            ValueError: If any input array is not on this solver's device.

        Returns:
            Evaluated constraints, with shape ``(num_worlds, num_constraints_max)``.
        """
        if body_q.device != self.device:
            raise ValueError("body_q must be on the solver's device")
        if target_rel_transforms.device != self.device:
            raise ValueError("target_rel_transforms must be on the solver's device")

        constraints = wp.zeros(
            dtype=wp.float32,
            shape=(
                self.model.size.num_worlds,
                self.data.dimensions.num_constraints_max,
            ),
            device=self.device,
        )
        world_mask = wp.ones(dtype=wp.bool, shape=(self.model.size.num_worlds,), device=self.device)
        self._eval_kinematic_constraints(body_q, target_rel_transforms, world_mask, constraints)
        return constraints

    def eval_kinematic_constraints_jacobian(
        self, body_q: wp.array[wp.transformf], target_rel_transforms: wp.array[wp.transformf]
    ) -> wp.array3d[wp.float32]:
        """
        Evaluates and returns the kinematic constraints Jacobian (w.r.t. body poses) given the body poses
        and the target relative transforms.

        Args:
            body_q: Body poses, with shape ``(num_bodies,)``.
            target_rel_transforms: Target relative transforms per joint, with shape ``(num_fk_joints,)``.

        Raises:
            ValueError: If any input array is not on this solver's device.

        Returns:
            Evaluated dense Jacobian, with shape ``(num_worlds, num_constraints_max, num_states_max)``.
        """
        if body_q.device != self.device:
            raise ValueError("body_q must be on the solver's device")
        if target_rel_transforms.device != self.device:
            raise ValueError("target_rel_transforms must be on the solver's device")

        constraints_jacobian = wp.zeros(
            dtype=wp.float32,
            shape=(
                self.model.size.num_worlds,
                self.data.dimensions.num_constraints_max,
                self.data.dimensions.num_states_max,
            ),
            device=self.device,
        )
        world_mask = wp.ones(dtype=wp.bool, shape=(self.model.size.num_worlds,), device=self.device)
        self._eval_kinematic_constraints_jacobian(body_q, target_rel_transforms, world_mask, constraints_jacobian)
        return constraints_jacobian

    def assemble_sparse_jacobian(self, body_q: wp.array[wp.transformf], target_rel_transforms: wp.array[wp.transformf]):
        """
        Assembles the sparse Jacobian (under self.data.problem.sparse_jacobian) given input body poses and control transforms.

        Note: only safe to call if this object was finalized with sparsity enabled in the config.

        Args:
            body_q: Body poses, with shape ``(num_bodies,)``.
            target_rel_transforms: Target relative transforms per joint, with shape ``(num_fk_joints,)``.

        Raises:
            ValueError: If any input array is not on this solver's device.
        """
        if body_q.device != self.device:
            raise ValueError("body_q must be on the solver's device")
        if target_rel_transforms.device != self.device:
            raise ValueError("target_rel_transforms must be on the solver's device")

        world_mask = wp.ones(dtype=wp.bool, shape=(self.model.size.num_worlds,), device=self.device)
        self._assemble_sparse_jacobian(body_q, target_rel_transforms, world_mask)

    def request_velocity_solve_batch_size(self, batch_size: int) -> None:
        """
        Preallocate the necessary internal buffers for a velocity solve with specified batch size.

        Must be called before the first call to :meth:`solve_for_body_velocities()` with this specific
        batch size.

        Args:
            batch_size: Number of velocity right-hand sides solved together.
        """
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if self.config.use_sparsity and batch_size != 1:
            raise ValueError("Multiple velocity right-hand sides require the dense FK solver")
        if batch_size in self.data.velocity_solve:
            return

        self.data.velocity_solve[batch_size] = FKVelocitySolveData(
            fk_actuator_u=wp.zeros(
                (batch_size, self.data.dimensions.num_actuated_dofs), dtype=wp.float32, device=self.device
            ),
            target_cts_u=wp.zeros(
                (self.model.size.num_worlds, self.data.dimensions.num_constraints_max, batch_size),
                dtype=wp.float32,
                device=self.device,
            ),
            rhs=wp.zeros(
                (self.model.size.num_worlds, self.data.dimensions.num_states_max, batch_size),
                dtype=wp.float32,
                device=self.device,
            ),
            body_q_dot=wp.zeros(
                (self.model.size.num_worlds, self.data.dimensions.num_states_max, batch_size),
                dtype=wp.float32,
                device=self.device,
            ),
        )
        self.linear_solver_llt.request_rhs_size(batch_size)

    def solve_for_body_velocities(
        self,
        actuator_u: wp.array[wp.float32] | wp.array2d[wp.float32],
        body_q: wp.array[wp.transformf],
        body_u: wp.array[wp.spatial_vectorf] | wp.array2d[wp.spatial_vectorf],
        base_u: wp.array[wp.spatial_vectorf] | wp.array2d[wp.spatial_vectorf] | None = None,
        target_rel_transforms: wp.array[wp.transformf] | None = None,
        world_mask: wp.array[wp.bool] | None = None,
    ):
        """
        Graph-capturable function solving for body velocities as a post-processing to the FK solve.
        More specifically, solves for body twists yielding zero constraint velocities, except at
        actuated dofs and at the base joint, where velocities must match prescribed velocities.

        Processes ``batch_size`` velocity vectors in parallel. For ``batch_size > 1``,
        :meth:`request_velocity_solve_batch_size()` must be called beforehand.

        Args:
            actuator_u: Actuated joint velocities, with shape ``(batch_size, num_fk_actuated_dofs)``,
                or a 1D array if batch_size = 1.
            body_q: Body poses (FK solution given the target relative transforms), with shape ``(num_bodies,)``.
            body_u: Body velocities (twists) written out by the solver, with shape ``(batch_size, num_bodies)``,
                or a 1D array if batch_size = 1.
            base_u: Base body twist per world, in the base joint frame if applicable, absolute otherwise.
                Shape of ``(batch_size, num_worlds)``, or a 1D array if batch_size = 1.
                Defaults to zero; ignored when no base body/joint is set.
                A single row is broadcast across the batch.
            target_rel_transforms: Target relative transforms per joint, encoding actuated coordinates and
                base pose. Shape of ``(num_fk_joints,)``. If not provided, inferred from ``body_q``.
            world_mask: Per-world boolean flags selecting which worlds to process, shape ``(num_worlds,)``.
                Defaults to all worlds.

        Raises:
            ValueError: If ``batch_size`` was not requested beforehand, if any input array is not on this
                solver's device, or if batched inputs have inconsistent shapes.
        """
        for name, arr in (
            ("actuator_u", actuator_u),
            ("body_q", body_q),
            ("body_u", body_u),
            ("base_u", base_u),
            ("target_rel_transforms", target_rel_transforms),
            ("world_mask", world_mask),
        ):
            if arr is not None and arr.device != self.device:
                raise ValueError(f"{name} must be on the solver's device")

        # Resolve batch size (= number of right-hand sides)
        if len(actuator_u.shape) > 1:
            batch_size = actuator_u.shape[0]
            if len(body_u.shape) <= 1 or body_u.shape[0] != batch_size:
                raise ValueError(f"body_u must have batch dimension of size {batch_size}")
            if base_u is not None and (len(base_u.shape) <= 1 or base_u.shape[0] not in (1, batch_size)):
                raise ValueError(f"base_u first dimension must be 1 or {batch_size}")
            if self.config.use_sparsity:
                raise ValueError("Multi-RHS velocity FK currently requires the dense FK solver")
        else:
            actuator_u = actuator_u.reshape((1, actuator_u.shape[0]))
            body_u = body_u.reshape((1, body_u.shape[0]))
            if base_u is not None:
                base_u = base_u.reshape((1, base_u.shape[0]))

        # Use default base velocity if not provided
        if base_u is None:
            base_u = self.data.problem.base_u_default.reshape((1, self.model.size.num_worlds))

        # Use default mask with all worlds if not provided
        world_mask = self.data.dimensions.all_worlds_mask if world_mask is None else world_mask

        # Extract target relative transformations from state if not provided
        if target_rel_transforms is None:
            self._eval_actuator_coords(body_q, self.data.problem.actuator_q_next)
            self._eval_target_relative_transformations(
                self.data.problem.actuator_q_next, self.data.problem.target_rel_transforms, world_mask
            )
            target_rel_transforms = self.data.problem.target_rel_transforms

        # Compute velocities
        self._solve_for_body_velocities(target_rel_transforms, base_u, actuator_u, body_q, body_u, world_mask)

    def run_fk_solve(
        self,
        actuator_q: wp.array[wp.float32],
        body_q: wp.array[wp.transformf],
        base_q: wp.array[wp.transformf] | None = None,
        actuator_u: wp.array[wp.float32] | None = None,
        base_u: wp.array[wp.spatial_vectorf] | None = None,
        body_u: wp.array[wp.spatial_vectorf] | None = None,
        world_mask: wp.array[wp.bool] | None = None,
    ):
        """
        Graph-capturable function solving forward kinematics with Gauss-Newton.

        More specifically, solves for the rigid body poses satisfying kinematic constraints,
        given actuated joint coordinates and base pose. Optionally also solves for rigid body velocities
        given actuator and base body velocities.

        Args:
            actuator_q: Actuated joint coordinates, shape ``(num_fk_actuated_coords,)``.
            body_q: Body poses, shape ``(num_bodies,)``. Written out by the solver, and used as an initial
                guess when ``reset_state`` is False.
            base_q: Base pose per world, in the base joint frame if applicable, absolute otherwise.
                Shape of ``(num_worlds,)``. Defaults to zero coordinates of the base joint, or the initial
                base body pose; ignored when no base body/joint is set.
            actuator_u: Actuated joint velocities, shape ``(num_fk_actuated_dofs,)``. Required
                when solving for velocities (``body_u`` provided).
            base_u: Base body twist per world, in the base joint frame if applicable, absolute otherwise.
                Shape of ``(num_worlds,)``. Defaults to zero; ignored when no base body/joint is set.
            body_u: Body velocities (twists) written out by the solver if provided, shape ``(num_bodies,)``.
            world_mask: Per-world boolean flags selecting which worlds to process, shape ``(num_worlds,)``.
                Defaults to all worlds.

        Raises:
            ValueError: If ``body_u`` is provided without ``actuator_u``.
        """
        # Check that actuator_u are provided if we need to solve for body_u
        if body_u is not None and actuator_u is None:
            raise ValueError(
                "run_fk_solve: actuator_u must be provided to solve for velocities (i.e. if body_u is provided)."
            )

        # Reset iteration count and success/continuation flags
        self.data.gauss_newton.iteration.fill_(-1)  # The initial Newton convergence check will increment this to zero
        self.data.gauss_newton.success.zero_()
        if world_mask is not None:
            self.data.gauss_newton.mask.assign(world_mask)
        else:
            wp.copy(self.data.gauss_newton.mask, self.data.dimensions.all_worlds_mask)
        self.data.gauss_newton.min_iterations.fill_(-1)  # To disregard min iterations in initial Newton check

        # Optionally reset state
        if self.config.reset_state:
            if base_q is None:
                self._reset_state(body_q, self.data.gauss_newton.mask)
            else:
                self._reset_state_base_q(body_q, base_q, self.data.gauss_newton.mask)

        # Optionally initialize the reference pose for the regularizer
        if self.config.use_regularization:
            wp.copy(self.data.problem.body_q_ref, body_q)

        # Use default base state if not provided
        if base_q is None:
            base_q = self.data.problem.base_q_default
        if body_u is not None and base_u is None:
            base_u = self.data.problem.base_u_default

        # Compute target actuator coordinates and corresponding transforms
        self._eval_target_actuator_q(base_q, actuator_q, self.data.problem.actuator_q_next)
        self._eval_target_relative_transformations(
            self.data.problem.actuator_q_next, self.data.problem.target_rel_transforms, self.data.gauss_newton.mask
        )

        # Evaluate constraints, and initialize loop condition (might not even need to loop)
        self._eval_kinematic_constraints(
            body_q,
            self.data.problem.target_rel_transforms,
            self.data.gauss_newton.mask,
            self.data.problem.constraints,
        )
        if self.config.use_regularization:  # Update Jacobian and gradient for stopping criterion
            self._update_jacobian(body_q, self.data.problem.target_rel_transforms, self.data.gauss_newton.mask)
            self._update_gradient(body_q, self.data.gauss_newton.mask)
        self._eval_max_residual(
            self.data.problem.constraints, self.data.gauss_newton.grad, self.data.gauss_newton.max_residual
        )
        self.data.gauss_newton.loop_condition.zero_()
        wp.copy(
            self.data.line_search.success, self.data.gauss_newton.mask
        )  # To disregard line search success in initial Newton check
        wp.launch(
            _newton_check,
            dim=(self.model.size.num_worlds,),
            inputs=[
                self.data.gauss_newton.max_residual,
                self.config.tolerance,
                self.data.gauss_newton.iteration,
                self.data.gauss_newton.min_iterations,
                self.config.max_newton_iterations,
                self.data.line_search.success,
                self.data.gauss_newton.success,
                self.data.gauss_newton.mask,
                self.data.gauss_newton.loop_condition,
                self.data.gauss_newton.jacobian_early_update_mask,
                self.data.gauss_newton.jacobian_late_update_mask,
            ],
            device=self.device,
        )

        # Initialize incremental solve
        if self.config.use_incremental_solve:
            wp.capture_if(self.data.gauss_newton.loop_condition, lambda: self._initialize_incremental_solve(body_q))

        # Main loop
        wp.capture_while(self.data.gauss_newton.loop_condition, lambda: self._run_newton_iteration(body_q))

        # Velocity solve, for worlds where FK ran and was successful
        if body_u is not None:
            self._solve_for_body_velocities(
                self.data.problem.target_rel_transforms,
                base_u.reshape((1, base_u.shape[0])),
                actuator_u.reshape((1, actuator_u.shape[0])),
                body_q,
                body_u.reshape((1, body_u.shape[0])),
                self.data.gauss_newton.success,
            )

    def solve_fk(
        self,
        actuator_q: wp.array[wp.float32],
        body_q: wp.array[wp.transformf],
        base_q: wp.array[wp.transformf] | None = None,
        actuator_u: wp.array[wp.float32] | None = None,
        base_u: wp.array[wp.spatial_vectorf] | None = None,
        body_u: wp.array[wp.spatial_vectorf] | None = None,
        world_mask: wp.array[wp.bool] | None = None,
        verbose: bool = False,
        return_status: bool = False,
        use_graph: bool = True,
    ):
        """
        Non-graph-capturable convenience wrapper around :meth:`run_fk_solve` adding verbosity and optional
        internal graph capture across calls.

        Args:
            actuator_q: Actuated joint coordinates, shape ``(num_fk_actuated_coords,)``.
            body_q: Body poses, shape ``(num_bodies,)``. Written out by the solver, and used as an initial
                guess when ``reset_state`` is False.
            base_q: Base pose per world, in the base joint frame if applicable, absolute otherwise.
                Shape of ``(num_worlds,)``. Defaults to zero coordinates of the base joint, or the initial
                base body pose; ignored when no base body/joint is set.
            actuator_u: Actuated joint velocities, shape ``(num_fk_actuated_dofs,)``. Required
                when solving for velocities (``body_u`` provided).
            base_u: Base body twist per world, in the base joint frame if applicable, absolute otherwise.
                Shape of ``(num_worlds,)``. Defaults to zero; ignored when no base body/joint is set.
            body_u: Body velocities (twists) written out by the solver if provided, shape ``(num_bodies,)``.
            world_mask: Per-world boolean flags selecting which worlds to process, shape ``(num_worlds,)``.
                Defaults to all worlds.
            verbose: Whether to write a status message at the end.
            return_status: Whether to return the detailed solver status.
            use_graph: Whether to use graph capture internally to accelerate repeated calls (turn off to
                profile individual kernels). If enabled, optional arrays must be either always or never
                provided across calls.

        Returns:
            The detailed solver status (success flag, iterations, residual per world) if
            ``return_status`` is True, else nothing.

        Raises:
            ValueError: If any input array is not on this solver's device.
        """
        for name, arr in (
            ("actuator_q", actuator_q),
            ("body_q", body_q),
            ("base_q", base_q),
            ("actuator_u", actuator_u),
            ("base_u", base_u),
            ("body_u", body_u),
        ):
            if arr is not None and arr.device != self.device:
                raise ValueError(f"{name} must be on the solver's device")

        # Warn if any world does not have an assigned base body when base attributes are provided.
        if (base_q is not None or base_u is not None) and self.model.info.has_world_without_base_body:
            msg.warning(
                "Some worlds have no free-floating base body assigned, possibly due to a non-free articulation root (fixed-base system). "
                "Base pose/velocity updates for forward kinematics will have no effect for those worlds."
            )

        # Run solve (with or without graph)
        if use_graph:
            if self.graph is None:
                wp.capture_begin(self.device)
                self.run_fk_solve(actuator_q, body_q, base_q, actuator_u, base_u, body_u, world_mask)
                self.graph = wp.capture_end()
            wp.capture_launch(self.graph)
        else:
            self.run_fk_solve(actuator_q, body_q, base_q, actuator_u, base_u, body_u, world_mask)

        # Status message
        if verbose or return_status:
            success = self.data.gauss_newton.success.numpy().copy()
            iterations = self.data.gauss_newton.iteration.numpy().copy()
            max_residual = self.data.gauss_newton.max_residual.numpy().copy()
            num_active_worlds = self.model.size.num_worlds if world_mask is None else world_mask.numpy().sum()
            if verbose:
                sys.__stdout__.write(f"Newton success for {success.sum()}/{num_active_worlds} worlds; ")
                sys.__stdout__.write(f"num iterations={iterations.max()}; ")
                sys.__stdout__.write(f"max residual={max_residual.max()}\n")

        # Return solver status
        if return_status:
            return ForwardKinematicsSolver.Status(iterations=iterations, max_residual=max_residual, success=success)


###
# Functions
###


def compute_fk_equivalence_classes(model: ModelKamino) -> list[list[int]]:
    """Groups world that are equivalent for FK discrete information"""
    sig_num_bodies = DiscreteSignature(num_worlds=model.size.num_worlds, data=model.info.num_bodies)
    sig_joint_act_type = DiscreteSignature(
        num_worlds=model.size.num_worlds,
        data=model.joints.act_type,
        world_offset=model.info.joints_offset,
        world_size=model.info.num_joints,
    )
    sig_joint_dof_type = DiscreteSignature(
        num_worlds=model.size.num_worlds,
        data=model.joints.dof_type,
        world_offset=model.info.joints_offset,
        world_size=model.info.num_joints,
    )
    sig_joint_bid_B = DiscreteSignature(
        num_worlds=model.size.num_worlds,
        data=model.joints.bid_B,
        world_offset=model.info.joints_offset,
        world_size=model.info.num_joints,
        world_delta=model.info.bodies_offset,
        ignore_negative=True,
    )
    sig_joint_bid_F = DiscreteSignature(
        num_worlds=model.size.num_worlds,
        data=model.joints.bid_F,
        world_offset=model.info.joints_offset,
        world_size=model.info.num_joints,
        world_delta=model.info.bodies_offset,
    )
    sig_base_body = DiscreteSignature(
        num_worlds=model.size.num_worlds,
        data=model.info.base_body_index,
        world_delta=model.info.bodies_offset,
        ignore_negative=True,
    )
    sig_base_joint = DiscreteSignature(
        num_worlds=model.size.num_worlds,
        data=model.info.base_joint_index,
        world_delta=model.info.joints_offset,
        ignore_negative=True,
    )
    signatures = [
        sig_num_bodies,
        sig_joint_act_type,
        sig_joint_dof_type,
        sig_joint_bid_B,
        sig_joint_bid_F,
        sig_base_body,
        sig_base_joint,
    ]

    if model.joints.fk_act_flag is not None:
        sig_joint_fk_act_flag = DiscreteSignature(
            num_worlds=model.size.num_worlds,
            data=model.joints.fk_act_flag,
            world_offset=model.info.joints_offset,
            world_size=model.info.num_joints,
        )
        signatures.append(sig_joint_fk_act_flag)

    return compute_equivalence_classes(signatures)
