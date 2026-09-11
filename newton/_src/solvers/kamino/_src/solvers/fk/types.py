# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Defines data types used by the Forward Kinematics solver."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import TYPE_CHECKING

import numpy as np
import warp as wp

if TYPE_CHECKING:
    from newton._src.solvers.kamino._src.core.types import vec7f
    from newton._src.solvers.kamino._src.linalg.sparse_matrix import BlockSparseMatrices
    from newton._src.solvers.kamino._src.linalg.sparse_operator import BlockSparseLinearOperators

###
# Module interface
###

__all__ = [
    "FKData",
    "FKDimensions",
    "FKGaussNewtonData",
    "FKJointDoFType",
    "FKJointsModel",
    "FKLineSearchData",
    "FKLinearSystemData",
    "FKProblemData",
    "FKVelocitySolveData",
    "ForwardKinematicsPreconditionerType",
    "ForwardKinematicsStatus",
]


###
# Module configs
###

wp.set_module_options({"enable_backward": False})


###
# Types
###


class FKJointDoFType(IntEnum):
    """
    Joint dof types for the FK solver, which currently differs from Kamino's main joint types
    by the addition of the axis joint, used to regularize tie rods between two spherical joints
    (taking out the rotation dof about their own axis).

    Importantly, the integer value is the same for all joints not specific to FK, allowing seamless
    conversions.
    """

    FREE = 0
    REVOLUTE = 1
    PRISMATIC = 2
    CYLINDRICAL = 3
    UNIVERSAL = 4
    SPHERICAL = 5
    CARTESIAN = 6
    FIXED = 7
    GIMBAL = 8
    GIMBAL_LEFT_HANDED = 9
    AXIS = 10


class ForwardKinematicsPreconditionerType(IntEnum):
    """Conjugate gradient preconditioning options of the FK solver, if sparsity is enabled."""

    NONE = 0
    """No preconditioning"""

    JACOBI_DIAGONAL = 1
    """Diagonal Jacobi preconditioner"""

    JACOBI_BLOCK_DIAGONAL = 2
    """Blockwise-diagonal Jacobi preconditioner, alternating blocks of size 3 and 4 along the diagonal,
    corresponding to the position and orientation (quaternion) of individual rigid bodies."""

    @classmethod
    def from_string(cls, s: str) -> ForwardKinematicsPreconditionerType:
        """Converts a string to a ForwardKinematicsPreconditionerType enum value."""
        try:
            return cls[s.upper()]
        except KeyError as e:
            raise ValueError(
                f"Invalid ForwardKinematicsPreconditionerType: {s}. Valid options are: {[e.name for e in cls]}"
            ) from e


@dataclass
class ForwardKinematicsStatus:
    """
    Container holding detailed information on the success/failure status of a forward kinematics solve.
    """

    success: np.ndarray(dtype=np.int32)
    """
    Solver success flag per world, as an integer array (0 = failure, 1 = success).
    Shape of `(num_worlds,)`.

    Note that in some cases the solver may fail to converge within the maximum number
    of iterations, but still produce a solution with a reasonable residual.
    In such cases, the success flag will be set to 0, but the `max_residual` field
    can be inspected to check the actual residuals and determine if the solution is acceptable
    for the intended application.
    """

    iterations: np.ndarray(dtype=np.int32)
    """
    Number of Gauss-Newton iterations executed per world.
    Shape of `(num_worlds,)`.
    """

    max_residual: np.ndarray(dtype=np.float32)
    """
    Maximal absolute residual at the final solution, per world. In the general case, the residual vector
    is the kinematic constraints vector; if regularization is enabled, it is the penalty gradient.

    Shape of `(num_worlds,)`.
    """


###
# Internal solver data (FKData and sub-containers)
###


@dataclass
class FKDimensions:
    """Sizes, index maps and world/tile bookkeeping used across the FK solver."""

    ###
    # Scalar counts
    ###

    num_states_tot: int = 0
    """FK state dims (7 * num bodies) for the whole model."""

    num_states_max: int = 0
    """Max FK state dim across worlds."""

    num_joints_tot: int = 0
    """Total joints (FK model, including axis joints)."""

    num_joints_max: int = 0
    """Max number of FK joints across worlds."""

    num_axis_joints: int = 0
    """Number of FK-specific axis joints across worlds."""

    num_actuated_coords: int = 0
    """Total actuated coords across worlds."""

    num_actuated_coords_max: int = 0
    """Max actuated coords across worlds."""

    num_actuated_dofs: int = 0
    """Total actuated dofs across worlds."""

    num_constraints_max: int = 0
    """Max kinematic constraints across worlds."""

    ###
    # Tile sizes for tile-based kernels
    ###

    tile_size_cts_1d: int = 0
    """Tile size for 1d tile-based kernels, along the constraints axis."""

    num_tiles_cts_1d: int = 0
    """Tile count for 1d tile-based kernels, along the constraints axis."""

    tile_size_vrs_1d: int = 0
    """Tile size for 1d tile-based kernels, along the states axis."""

    num_tiles_vrs_1d: int = 0
    """Tile count for 1d tile-based kernels, along the states axis."""

    tile_size_cts_2d: int = 0
    """Tile size for 2d tile-based kernels, along the constraints axis."""

    num_tiles_cts_2d: int = 0
    """Tile count for 2d tile-based kernels, along the constraints axis."""

    tile_size_vrs_2d: int = 0
    """Tile size for 2d tile-based kernels, along the states axis."""

    num_tiles_vrs_2d: int = 0
    """Tile count for 2d tile-based kernels, along the states axis."""

    tile_size_coords: int = 0
    """Tile size for 1d tile-based kernels, along the actuated coords axis."""

    num_tiles_coords: int = 0
    """Tile count for 1d tile-based kernels, along the actuated coords axis."""

    ###
    # Per-world counts and offsets
    ###

    num_joints: wp.array[wp.int32] | None = None
    """
    Number of FK joints per world.
    Shape of ``(num_worlds,)``.
    """

    joints_offset: wp.array[wp.int32] | None = None
    """
    FK joint index offset per world, w.r.t. FK joints of all worlds.
    Shape of ``(num_worlds + 1,)``.
    """

    num_states: wp.array[wp.int32] | None = None
    """
    Number of FK state dimensions per world.
    Shape of ``(num_worlds,)``.
    """

    num_constraints: wp.array[wp.int32] | None = None
    """
    Number of FK constraint dimensions per world.
    Shape of ``(num_worlds,)``.
    """

    actuated_coords_offset: wp.array[wp.int32] | None = None
    """
    Per-joint actuated coord offset.
    Shape of ``(num_joints_tot + 1,)``.
    """

    world_actuated_coords_offset: wp.array[wp.int32] | None = None
    """
    Per-world actuated coord offset.
    Shape of ``(num_worlds + 1,)``.
    """

    actuated_coords_map: wp.array[wp.int32] | None = None
    """
    Map of FK actuated coords to model actuated or base coords.
    Shape of ``(num_actuated_coords,)``.
    """

    actuated_dofs_offset: wp.array[wp.int32] | None = None
    """
    Per-joint actuated dof offset.
    Shape of ``(num_joints_tot + 1,)``.
    """

    actuated_dofs_map: wp.array[wp.int32] | None = None
    """
    Map of FK actuated dofs to model actuated or base dofs.
    Shape of ``(num_actuated_dofs,)``.
    """

    constraint_full_to_red_map: wp.array[wp.int32] | None = None
    """
    Full to reduced constraint index map, from 6 maximal constraints per joint to actual constraints.
    Shape of ``(6 * num_joints_tot,)``.
    """

    ###
    # Sparse Jacobian tile pattern
    ###

    tile_sparsity_pattern: wp.array3d[wp.int32] | None = None
    """
    0-1 flag per Jacobian tile, with 0 for tiles that are structurally fully zero.
    Shape of ``(num_worlds, num_tiles_cts_2d, num_tiles_vrs_2d)``.
    """

    rb_nzb_id: wp.array[wp.int32] | None = None
    """
    Non-zero block id for the quaternion norm constraint row associated to a rigid body.
    Shape of ``(num_bodies_tot,)``.
    """

    ct_nzb_id_base: wp.array[wp.int32] | None = None
    """
    Map from full constraint id to non-zero block id, for the base body constraint block.
    Shape of ``(6 * num_joints_tot,)``.
    """

    ct_nzb_id_follower: wp.array[wp.int32] | None = None
    """
    Map from full constraint id to non-zero block id, for the follower body constraint block.
    Shape of ``(6 * num_joints_tot,)``.
    """

    ###
    # Convenience mask covering all worlds
    ###

    all_worlds_mask: wp.array[wp.bool] | None = None
    """
    True for all worlds.
    Shape of ``(num_worlds,)``.
    """


@dataclass
class FKJointsModel:
    """FK-local joints model (may differ from the main model's joints; adds axis joints and base joint)."""

    ###
    # FK joints definition
    ###

    dof_type: wp.array[wp.int32] | None = None
    """
    Joint dof type per joint.
    Shape of ``(num_joints_tot,)``.
    """

    act_type: wp.array[wp.int32] | None = None
    """
    Joint actuation type per joint.
    Shape of ``(num_joints_tot,)``.
    """

    bid_B: wp.array[wp.int32] | None = None
    """
    Base body id per joint.
    Shape of ``(num_joints_tot,)``.
    """

    bid_F: wp.array[wp.int32] | None = None
    """
    Follower body id per joint.
    Shape of ``(num_joints_tot,)``.
    """

    B_r_Bj: wp.array[wp.vec3f] | None = None
    """
    Joint local origin on base body.
    Shape of ``(num_joints_tot,)``.
    """

    F_r_Fj: wp.array[wp.vec3f] | None = None
    """
    Joint local origin on follower body.
    Shape of ``(num_joints_tot,)``.
    """

    X_Bj: wp.array[wp.mat33f] | None = None
    """
    Joint local frame on base body.
    Shape of ``(num_joints_tot,)``.
    """

    X_Fj: wp.array[wp.mat33f] | None = None
    """
    Joint local frame on follower body.
    Shape of ``(num_joints_tot,)``.
    """

    source_id: wp.array[wp.int32] | None = None
    """
    Corresponding joint id in the main model, -1 for FK-only joints.
    Shape of ``(num_joints_tot,)``.
    """

    world_id: wp.array[wp.int32] | None = None
    """
    World id per joint.
    Shape of ``(num_joints_tot,)``.
    """

    ###
    # Axis joints data
    ###

    axis_joint_id: wp.array[wp.int32] | None = None
    """
    Indices of FK joints that are axis joints, across worlds.
    Shape of ``(num_axis_joints,)``.
    """

    axis_body_id: wp.array[wp.int32] | None = None
    """
    Body id targeted by each axis joint (i.e. tie rod body).
    Shape of ``(num_axis_joints,)``.
    """

    axis_source_joint_0: wp.array[wp.int32] | None = None
    """
    First source joint id per axis joint (i.e. first spherical joint).
    Shape of ``(num_axis_joints,)``.
    """

    axis_source_joint_1: wp.array[wp.int32] | None = None
    """
    Second source joint id per axis joint (i.e. second spherical joint).
    Shape of ``(num_axis_joints,)``.
    """

    ###
    # Global data related to joints
    ###

    base_joint_id: wp.array[wp.int32] | None = None
    """
    Base joint id per world (-1 = none).
    Shape of ``(num_worlds,)``.
    """

    has_universal_actuators: bool = False
    """True iff the model has at least one actuated universal joint."""

    _built_actuated: wp.array[wp.int32] | None = None
    """
    Built-time actuated flags per joint.
    Shape of ``(num_joints_tot,)``.
    """

    _actuation_violations: wp.array[wp.int32] | None = None
    """
    Violations counter for model-change validation.
    Shape of ``(2,)``.
    """


@dataclass
class FKProblemData:
    """Problem inputs, current estimates, and constraint/Jacobian storage for the FK Gauss-Newton solve."""

    ###
    # Actuator targets
    ###

    actuator_q_next: wp.array[wp.float32] | None = None
    """
    Target actuated coords for the FK solve.
    Shape of ``(num_actuated_coords,)``.
    """

    actuator_q_prev: wp.array[wp.float32] | None = None
    """
    Actuated coords before the FK solve.
    Shape of ``(num_actuated_coords,)``.
    """

    actuator_q_curr: wp.array[wp.float32] | None = None
    """
    Current incremental target for actuated coords.
    Shape of ``(num_actuated_coords,)``.
    """

    target_rel_transforms: wp.array[wp.transformf] | None = None
    """
    Target relative transforms per joint.
    Shape of ``(num_joints_tot,)``.
    """

    ###
    # Reference state
    ###

    body_q_ref: wp.array[wp.transformf] | None = None
    """
    Reference body poses for the regularizer.
    Shape of ``(num_bodies_tot,)``.
    """

    base_q_default: wp.array[wp.transformf] | None = None
    """
    Default base pose per world.
    Shape of ``(num_worlds,)``.
    """

    base_u_default: wp.array[wp.spatial_vectorf] | None = None
    """
    Default base twist per world.
    Shape of ``(num_worlds,)``.
    """

    ###
    # Constraints vector and Jacobian
    ###

    constraints: wp.array2d[wp.float32] | None = None
    """
    Kinematic constraints vector.
    Shape of ``(num_worlds, num_constraints_max)``.
    """

    jacobian: wp.array3d[wp.float32] | None = None
    """
    Dense constraints Jacobian.
    Shape of ``(num_worlds, num_constraints_max, num_states_max)``.
    """

    sparse_jacobian: BlockSparseMatrices[wp.float32, wp.int32, vec7f] | None = None
    """Sparse constraints Jacobian."""

    sparse_jacobian_op: BlockSparseLinearOperators[wp.float32, wp.int32] | None = None
    """Sparse Jacobian operator."""


@dataclass
class FKGaussNewtonData:
    """State and companion arrays driving the outer Gauss-Newton loop."""

    ###
    # Loop status and termination data
    ###

    max_residual: wp.array[wp.float32] | None = None
    """
    Max constraint / gradient residual per world.
    Shape of ``(num_worlds,)``.
    """

    iteration: wp.array[wp.int32] | None = None
    """
    Iteration count per world.
    Shape of ``(num_worlds,)``.
    """

    success: wp.array[wp.bool] | None = None
    """
    Convergence flag per world.
    Shape of ``(num_worlds,)``.
    """

    mask: wp.array[wp.bool] | None = None
    """
    Continue-iterating flag per world.
    Shape of ``(num_worlds,)``.
    """

    loop_condition: wp.array[wp.int32] | None = None
    """
    Global loop condition.
    Shape of ``(1,)``.
    """

    ###
    # Gauss-Newton gradient and step
    ###

    grad: wp.array2d[wp.float32] | None = None
    """
    Merit function gradient w.r.t. state.
    Shape of ``(num_worlds, num_states_max)``.
    """

    step: wp.array2d[wp.float32] | None = None
    """
    State step per world.
    Shape of ``(num_worlds, num_states_max)``.
    """

    ###
    # Incremental solve data
    ###

    delta_q_max: wp.array[wp.float32] | None = None
    """
    Max step size per actuated coord.
    Shape of ``(num_actuated_coords,)``.
    """

    min_iterations: wp.array[wp.int32] | None = None
    """
    Min iterations per world.
    Shape of ``(num_worlds,)``.
    """

    jacobian_early_update_mask: wp.array[wp.bool] | None = None
    """
    Per-world flag, True if early Jacobian update needed.
    Shape of ``(num_worlds,)``.
    """

    jacobian_late_update_mask: wp.array[wp.bool] | None = None
    """
    Per-world flag, True if late Jacobian update needed.
    Shape of ``(num_worlds,)``.
    """


@dataclass
class FKLineSearchData:
    """State and companion arrays driving the inner backtracking line search."""

    ###
    # Loop status and termination data
    ###

    iteration: wp.array[wp.int32] | None = None
    """
    Iteration count per world.
    Shape of ``(num_worlds,)``.
    """

    success: wp.array[wp.bool] | None = None
    """
    Line-search success flag per world.
    Shape of ``(num_worlds,)``.
    """

    mask: wp.array[wp.bool] | None = None
    """
    Continue-iterating flag per world.
    Shape of ``(num_worlds,)``.
    """

    loop_condition: wp.array[wp.int32] | None = None
    """
    Global loop condition.
    Shape of ``(1,)``.
    """

    ###
    # Candidate step data
    ###

    alpha: wp.array[wp.float32] | None = None
    """
    Current step size per world.
    Shape of ``(num_worlds,)``.
    """

    body_q_alpha: wp.array[wp.transformf] | None = None
    """
    Trial body poses at alpha.
    Shape of ``(num_bodies_tot,)``.
    """

    val_0: wp.array[wp.float32] | None = None
    """
    Merit function value at 0.
    Shape of ``(num_worlds,)``.
    """

    val_alpha: wp.array[wp.float32] | None = None
    """
    Merit function value at alpha.
    Shape of ``(num_worlds,)``.
    """

    grad_0: wp.array[wp.float32] | None = None
    """
    Merit function directional gradient at 0.
    Shape of ``(num_worlds,)``.
    """


@dataclass
class FKLinearSystemData:
    """Left/right-hand sides, temporaries and preconditioner for the linear system in each Newton step."""

    ###
    # System left/right-hand sides
    ###

    lhs: wp.array3d[wp.float32] | None = None
    """
    Dense Gauss-Newton LHS per world.
    Shape of ``(num_worlds, num_states_max, num_states_max)``.
    """

    rhs: wp.array2d[wp.float32] | None = None
    """
    Gauss-Newton RHS per world (= -grad).
    Shape of ``(num_worlds, num_states_max)``.
    """

    jacobian_times_vector: wp.array2d[wp.float32] | None = None
    """
    J * x intermediary.
    Shape of ``(num_worlds, num_constraints_max)``.
    """

    lhs_times_vector: wp.array2d[wp.float32] | None = None
    """
    (J^T J + regI) * x intermediary.
    Shape of ``(num_worlds, num_states_max)``.
    """

    ###
    # Preconditioner data for the sparse CG solve
    ###

    preconditioner_type: ForwardKinematicsPreconditionerType = ForwardKinematicsPreconditionerType.NONE
    """Preconditioner type used by the CG solver."""

    jacobian_diag_inv: wp.array2d[wp.float32] | None = None
    """
    Diagonal Jacobi preconditioner (CG).
    Shape of ``(num_worlds, num_states_max)``.
    """

    inv_blocks_3: wp.array2d[wp.mat33f] | None = None
    """
    Block-diagonal preconditioner 3x3 blocks.
    Shape of ``(num_worlds, num_bodies_max)``.
    """

    inv_blocks_4: wp.array2d[wp.mat44f] | None = None
    """
    Block-diagonal preconditioner 4x4 blocks.
    Shape of ``(num_worlds, num_bodies_max)``.
    """

    ###
    # CG termination data
    ###

    cg_atol: wp.array[wp.float32] | None = None
    """
    Absolute tolerance per world (CG).
    Shape of ``(num_worlds,)``.
    """

    cg_rtol: wp.array[wp.float32] | None = None
    """
    Relative tolerance per world (CG).
    Shape of ``(num_worlds,)``.
    """

    cg_max_iter: wp.array[wp.int32] | None = None
    """
    Max iterations per world (CG).
    Shape of ``(num_worlds,)``.
    """


@dataclass
class FKVelocitySolveData:
    """Preallocated buffers for one batch size of the post-FK velocity solve."""

    fk_actuator_u: wp.array2d[wp.float32] | None = None
    """
    Actuated dof velocities (FK model).
    Shape of ``(batch_size, num_actuated_dofs)``.
    """

    target_cts_u: wp.array3d[wp.float32] | None = None
    """
    Target constraint velocities.
    Shape of ``(num_worlds, num_constraints_max, batch_size)``.
    """

    rhs: wp.array3d[wp.float32] | None = None
    """
    Body-space RHS.
    Shape of ``(num_worlds, num_states_max, batch_size)``.
    """

    body_q_dot: wp.array3d[wp.float32] | None = None
    """
    Body pose time derivatives.
    Shape of ``(num_worlds, num_states_max, batch_size)``.
    """


@dataclass
class FKData:
    """Aggregated internal data of the FK solver, split into coherent sub-containers."""

    dimensions: FKDimensions = field(default_factory=FKDimensions)
    """Problem dimensions."""

    joints: FKJointsModel = field(default_factory=FKJointsModel)
    """FK joints model."""

    problem: FKProblemData = field(default_factory=FKProblemData)
    """FK problem data."""

    gauss_newton: FKGaussNewtonData = field(default_factory=FKGaussNewtonData)
    """Gauss-Newton data."""

    line_search: FKLineSearchData = field(default_factory=FKLineSearchData)
    """Line-search data."""

    linear_system: FKLinearSystemData = field(default_factory=FKLinearSystemData)
    """Linear solve data."""

    velocity_solve: dict[int, FKVelocitySolveData] = field(default_factory=dict)
    """Velocity solve data, stored by batch size."""
