# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Containers of the LOX constraint rows, splitting state, and solver status."""

from __future__ import annotations

from collections.abc import Sequence
from enum import IntEnum
from typing import TYPE_CHECKING

import numpy as np
import warp as wp

from ......core.types import override
from ...core.types import mat36f, mat66f, to_warp_int32_array, vec6f
from .contact import mat55f, vec5f
from .llt.info import DenseBlockInfo
from .spatial_warmstart import WarmstarterAngularContacts

if TYPE_CHECKING:
    from .metrics import DualSolution

###
# Module interface
###

__all__ = [
    "APGDData",
    "AndersonData",
    "AndersonRowsData",
    "BoxRows",
    "ColoredRowsData",
    "ColoringData",
    "ContactRows",
    "DynamicRows",
    "EffortRows",
    "LOXData",
    "LOXProblemData",
    "LOXResiduals",
    "LOXStatus",
    "LOXSystemData",
    "ProjectionAcceleration",
    "ProjectionData",
    "SpatialContactBlocks",
    "SpatialContactRows",
    "SplittingState",
    "StructuralRows",
    "capacity_offsets",
    "make_status",
    "segment_local_indices",
]


###
# Enumerations
###


class ProjectionAcceleration(IntEnum):
    """Acceleration of the unilateral projection sweeps."""

    NONE = 0
    """Plain projection sweeps."""

    APGD = 1
    """Restarted Nesterov extrapolation; requires the Jacobi schedule."""

    ANDERSON = 2
    """One safeguarded Anderson direction per projection sweep."""

    @classmethod
    def from_string(cls, s: str) -> ProjectionAcceleration:
        """Converts a string to a ProjectionAcceleration enum value."""
        try:
            return cls[s.upper()]
        except KeyError as e:
            raise ValueError(f"Invalid ProjectionAcceleration: {s}. Valid options are: {[e.name for e in cls]}") from e

    @override
    def __str__(self):
        """Returns a string representation of the ProjectionAcceleration."""
        return f"ProjectionAcceleration.{self.name} ({self.value})"

    @override
    def __repr__(self):
        """Returns a string representation of the ProjectionAcceleration."""
        return self.__str__()


###
# Structs
###


@wp.struct
class LOXStatus:
    """Per-world LOX terminal status."""

    converged: wp.int32
    """Whether the world met every configured LOX stopping tolerance."""
    iterations: wp.int32
    """Number of splitting iterations performed for the world."""
    r_p: wp.float32
    """Maximum splitting consensus gap ``|W (v - p)|`` [N·s or N·m·s], or NaN before solving or on failure."""
    r_d: wp.float32
    """Maximum last change of the smooth solution ``|v - v_prev|`` [m/s or rad/s], or NaN before solving or on failure."""
    r_c: wp.float32
    """Squared maximum metric-scaled natural-map residual [J] of the unilateral rows, or NaN before solving or on failure."""
    failed: wp.int32
    """Whether the solve of the world failed: an invalid local block or a non-finite update."""


###
# Functions
###


@wp.func
def make_status(converged: wp.bool, iterations: wp.int32, failed: wp.bool) -> LOXStatus:
    """Return a world status whose residuals are not yet available."""
    status = LOXStatus()
    status.converged = wp.int32(converged)
    status.iterations = iterations
    status.r_p = wp.nan
    status.r_d = wp.nan
    status.r_c = wp.nan
    status.failed = wp.int32(failed)
    return status


def capacity_offsets(capacities: Sequence[int] | np.ndarray) -> np.ndarray:
    """Return exclusive prefix sums of ``capacities`` with the total appended."""
    capacities_np = np.asarray(capacities, dtype=np.int32)
    offsets = np.empty(capacities_np.size + 1, dtype=np.int32)
    offsets[0] = 0
    np.cumsum(capacities_np, out=offsets[1:])
    return offsets


def segment_local_indices(counts: np.ndarray, offsets: np.ndarray | None = None) -> np.ndarray:
    """Return packed indices relative to the start of each variable-length segment."""
    if offsets is None:
        offsets = capacity_offsets(counts)
    return np.arange(int(offsets[-1]), dtype=np.int32) - np.repeat(offsets[:-1], counts)


###
# Containers
###


class BoxRows:
    """Joint friction and joint limit rows with ``lower <= lambda <= upper``.

    The rows of a world are its joint friction rows, which are always active,
    followed by the capacity for its detected joint limits. Rows past
    :attr:`world_count` of their world are inactive.

    Args:
        world_friction_count: Number of friction rows of each world.
        world_limit_capacity: Maximum number of detected limits of each world.
        friction_world: World of each friction row, sorted by world.
        friction_body_a: Body A of each friction row.
        friction_body_b: Body B of each friction row.
        device: Device of the arrays.
    """

    def __init__(
        self,
        world_friction_count: np.ndarray,
        world_limit_capacity: np.ndarray,
        friction_world: np.ndarray,
        friction_body_a: np.ndarray,
        friction_body_b: np.ndarray,
        device: wp.DeviceLike,
    ):
        world_count = len(world_friction_count)
        world_capacity = world_friction_count + world_limit_capacity
        offsets = capacity_offsets(world_capacity)
        self.capacity = int(offsets[-1])
        """Total number of rows."""
        self.friction_capacity = int(world_friction_count.sum())
        """Number of friction rows."""
        self.limit_capacity = int(world_limit_capacity.sum())
        """Number of rows reserved for detected limits."""
        self.max_world_capacity = int(np.max(world_capacity, initial=0))
        """Largest number of rows of one world."""

        self.world_offset = to_warp_int32_array(offsets[:-1], device)
        """First row of each world."""
        self.world_count = wp.zeros(world_count, dtype=wp.int32, device=device)
        """Number of active rows of each world."""
        self.world_friction_count = to_warp_int32_array(world_friction_count, device)
        """Number of friction rows of each world."""
        self.world_limit_capacity = to_warp_int32_array(world_limit_capacity, device)
        """Maximum number of detected limits of each world."""
        self.world_limit_offset = to_warp_int32_array(offsets[:-1] + world_friction_count, device)
        """First limit row of each world."""
        self.world = to_warp_int32_array(np.repeat(np.arange(world_count, dtype=np.int32), world_capacity), device)
        """World of each row."""
        self.local = to_warp_int32_array(segment_local_indices(world_capacity, offsets), device)
        """Index of each row within its world."""

        friction_row = offsets[friction_world] + segment_local_indices(world_friction_count)
        self.friction_row = to_warp_int32_array(friction_row, device)
        """Row of each friction constraint."""
        body_a = np.full(self.capacity, -1, dtype=np.int32)
        body_b = np.full(self.capacity, -1, dtype=np.int32)
        body_a[friction_row] = friction_body_a
        body_b[friction_row] = friction_body_b
        upper = np.full(self.capacity, np.inf, dtype=np.float32)
        upper[friction_row] = 0.0
        self.body_a = to_warp_int32_array(body_a, device)
        """Body A of each row, or ``-1``."""
        self.body_b = to_warp_int32_array(body_b, device)
        """Body B of each row, or ``-1``."""
        self.jacobian_a = wp.zeros(self.capacity, dtype=vec6f, device=device)
        """Jacobian of each row with respect to the twist of body A."""
        self.jacobian_b = wp.zeros(self.capacity, dtype=vec6f, device=device)
        """Jacobian of each row with respect to the twist of body B."""
        self.bias = wp.zeros(self.capacity, dtype=wp.float32, device=device)
        """Velocity bias of each row (limit stabilization, zero for friction)."""
        self.lower = wp.zeros(self.capacity, dtype=wp.float32, device=device)
        """Lower impulse bound of each row."""
        self.upper = wp.array(upper, dtype=wp.float32, device=device)
        """Upper impulse bound of each row."""
        self.reaction = wp.zeros(self.capacity, dtype=wp.float32, device=device)
        """Impulse of each row."""
        self.velocity = wp.zeros(self.capacity, dtype=wp.float32, device=device)
        """Constraint velocity ``J v`` of each row, without bias."""
        self.physical_delassus = wp.zeros(self.capacity, dtype=wp.float32, device=device)
        """Unsplit Delassus coefficient ``J W^-1 J^T`` of each row, which scales its residual."""
        self.world_residual_max = wp.zeros(world_count, dtype=wp.float32, device=device)
        """Largest row residual of each world."""


class ContactRows:
    """Normal-last Coulomb contact rows.

    Detected contacts are compacted per world into the first
    :attr:`world_count` rows of the world's capacity.

    Args:
        world_capacity: Maximum number of contacts of each world.
        device: Device of the arrays.
        compliant: Whether to allocate the normal compliance of each contact.
        restitution: Whether to allocate the in-kernel restitution inputs of each contact.
        delassus_type: Type of the unsplit Delassus block of each contact.
    """

    def __init__(
        self,
        world_capacity: np.ndarray,
        device: wp.DeviceLike,
        compliant: bool = False,
        restitution: bool = False,
        *,
        delassus_type: type = wp.mat33f,
    ):
        world_count = len(world_capacity)
        offsets = capacity_offsets(world_capacity)
        self.capacity = int(offsets[-1])
        """Total number of rows."""
        self.world_capacity = to_warp_int32_array(world_capacity, device)
        """Maximum number of contacts of each world."""
        self.max_world_capacity = int(np.max(world_capacity, initial=0))
        """Largest number of contacts of one world."""
        self.world_offset = to_warp_int32_array(offsets[:-1], device)
        """First row of each world."""
        self.world_count = wp.zeros(world_count, dtype=wp.int32, device=device)
        """Number of active contacts of each world."""
        self.world = to_warp_int32_array(np.repeat(np.arange(world_count, dtype=np.int32), world_capacity), device)
        """World of each row."""
        self.local = to_warp_int32_array(segment_local_indices(world_capacity, offsets), device)
        """Index of each row within its world."""
        self.source_to_internal = wp.full(self.capacity, -1, dtype=wp.int32, device=device)
        """Row of each detected contact, or ``-1`` for a contact without dynamic bodies."""
        self.body_a = wp.full(self.capacity, -1, dtype=wp.int32, device=device)
        """Body A of each contact, or ``-1``."""
        self.body_b = wp.full(self.capacity, -1, dtype=wp.int32, device=device)
        """Body B of each contact, or ``-1``."""
        self.jacobian_a = wp.zeros(self.capacity, dtype=mat36f, device=device)
        """Normal-last Jacobian of each contact with respect to the twist of body A."""
        self.jacobian_b = wp.zeros(self.capacity, dtype=mat36f, device=device)
        """Normal-last Jacobian of each contact with respect to the twist of body B."""
        self.bias = wp.zeros(self.capacity, dtype=wp.vec3f, device=device)
        """Velocity bias of each contact (penetration and restitution targets).

        Excludes the trial-dependent in-kernel restitution target.
        """
        self.normal_compliance = wp.zeros(self.capacity, dtype=wp.float32, device=device) if compliant else None
        """Impulse-space normal compliance ``compliance / dt**2`` [1/kg] of each contact, if compliant.

        Added to the normal Delassus coefficient of the local contact solve.
        """
        self.restitution = wp.zeros(self.capacity, dtype=wp.vec4f, device=device) if restitution else None
        """In-kernel restitution inputs ``(gap, begin-step normal velocity, coefficient, dt)`` of each contact."""
        self.frame: wp.array[wp.mat33f] | None = None
        """Contact frame ``(tangent_0, tangent_1, normal)`` as matrix columns, for spatial contacts."""
        self.angular_friction: wp.array[wp.vec2f] | None = None
        """Spinning and rolling friction coefficients [m] of each contact, for spatial contacts.

        The sliding coefficient is :attr:`friction`.
        """
        self.angular_reaction: wp.array[wp.vec3f] | None = None
        """Spin and rolling impulses of each contact, for spatial contacts."""
        self.physical_delassus = wp.zeros(self.capacity, dtype=delassus_type, device=device)
        """Unsplit Delassus block ``J W^-1 J^T`` of each contact, including its normal compliance, which
        scales its residual: normal-last ``3 x 3``, or normal-first ``6 x 6`` for spatial contacts."""
        self.angular_warmstarter: WarmstarterAngularContacts | None = None
        """Warm starter of the angular impulses across time steps, for spatial contacts."""
        self.friction = wp.zeros(self.capacity, dtype=wp.float32, device=device)
        """Coulomb friction coefficient of each contact."""
        self.reaction = wp.zeros(self.capacity, dtype=wp.vec3f, device=device)
        """Normal-last impulse of each contact."""
        self.velocity = wp.zeros(self.capacity, dtype=wp.vec3f, device=device)
        """Contact velocity ``J v`` of each contact, without bias."""
        self.world_residual_max = wp.zeros(world_count, dtype=wp.float32, device=device)
        """Largest contact residual of each world."""


class SpatialContactRows(ContactRows):
    """Contact rows with torsional and rolling friction.

    Each contact gains three angular reaction rows (spin about the normal, then
    rolling about the two tangents) and is solved as one 6D cone in the
    normal-first order of :mod:`.contact`. The linear rows keep the
    normal-last storage of :class:`ContactRows`; the angular rows reuse the
    contact frame axes.

    Args:
        world_capacity: Maximum number of contacts of each world.
        device: Device of the arrays.
        source_capacity: Number of detected contacts cached for the angular warm start.
        compliant: Whether to allocate the normal compliance of each contact.
        restitution: Whether to allocate the in-kernel restitution inputs of each contact.
    """

    def __init__(
        self,
        world_capacity: np.ndarray,
        device: wp.DeviceLike,
        source_capacity: int,
        compliant: bool = False,
        restitution: bool = False,
    ):
        super().__init__(world_capacity, device, compliant=compliant, restitution=restitution, delassus_type=mat66f)
        self.frame = wp.zeros(self.capacity, dtype=wp.mat33f, device=device)
        self.angular_friction = wp.zeros(self.capacity, dtype=wp.vec2f, device=device)
        self.angular_reaction = wp.zeros(self.capacity, dtype=wp.vec3f, device=device)
        self.angular_warmstarter = WarmstarterAngularContacts(source_capacity, device)


class DynamicRows:
    """Joint rows with implicit drives, damping, and armature (Kamino's dynamic constraints).

    Each row is a smooth penalty ``0.5 m (J v - v_free)^2`` assembled into the
    body system, with effective inertia ``m`` and free velocity ``v_free``.

    Args:
        world: World of each row.
        joint: Joint of each row.
        body_a: Body A of each row, or ``-1``.
        body_b: Body B of each row, or ``-1``.
        value_index: Kamino dynamic-constraint index of each row, or ``-1``.
        dof_index: Joint DOF of each row.
        multiplier_index: Kamino dynamic multiplier of each row, or ``-1``.
        sparse_a_index: Sparse Jacobian block of body A, or ``-1``.
        sparse_b_index: Sparse Jacobian block of body B.
        uses_dof_jacobian: Whether the row reads the DOF Jacobian instead of the constraint Jacobian.
        joint_offset: First row of each joint.
        joint_count: Number of rows of each joint.
        device: Device of the arrays.
    """

    def __init__(
        self,
        world: np.ndarray,
        joint: np.ndarray,
        body_a: np.ndarray,
        body_b: np.ndarray,
        value_index: np.ndarray,
        dof_index: np.ndarray,
        multiplier_index: np.ndarray,
        sparse_a_index: np.ndarray,
        sparse_b_index: np.ndarray,
        uses_dof_jacobian: np.ndarray,
        joint_offset: np.ndarray,
        joint_count: np.ndarray,
        device: wp.DeviceLike,
    ):
        self.count = len(world)
        """Total number of rows."""
        self.world = to_warp_int32_array(world, device)
        """World of each row."""
        self.joint = to_warp_int32_array(joint, device)
        """Joint of each row."""
        self.body_a = to_warp_int32_array(body_a, device)
        """Body A of each row, or ``-1``."""
        self.body_b = to_warp_int32_array(body_b, device)
        """Body B of each row, or ``-1``."""
        self.value_index = to_warp_int32_array(value_index, device)
        """Kamino dynamic-constraint index of each row, or ``-1``."""
        self.dof_index = to_warp_int32_array(dof_index, device)
        """Joint DOF of each row."""
        self.multiplier_index = to_warp_int32_array(multiplier_index, device)
        """Kamino dynamic multiplier of each row, or ``-1``."""
        self.sparse_a_index = to_warp_int32_array(sparse_a_index, device)
        """Sparse Jacobian block of body A, or ``-1``."""
        self.sparse_b_index = to_warp_int32_array(sparse_b_index, device)
        """Sparse Jacobian block of body B."""
        self.uses_dof_jacobian = wp.array(uses_dof_jacobian, dtype=wp.bool, device=device)
        """Whether the row reads the DOF Jacobian instead of the constraint Jacobian."""
        self.joint_offset = to_warp_int32_array(joint_offset, device)
        """First row of each joint."""
        self.joint_count = to_warp_int32_array(joint_count, device)
        """Number of rows of each joint."""
        self.effort_index = wp.full(self.count, -1, dtype=wp.int32, device=device)
        """Bounded-effort correction of each row, or ``-1``."""
        self.jacobian_a = wp.zeros(self.count, dtype=vec6f, device=device)
        """Jacobian of each row with respect to the twist of body A."""
        self.jacobian_b = wp.zeros(self.count, dtype=vec6f, device=device)
        """Jacobian of each row with respect to the twist of body B."""
        self.effective_inertia = wp.zeros(self.count, dtype=wp.float32, device=device)
        """Effective inertia ``m`` of each row."""
        self.free_velocity = wp.zeros(self.count, dtype=wp.float32, device=device)
        """Free velocity ``v_free`` of each row."""


class StructuralRows:
    """Joint rows with kinematic constraints, enforced by an augmented Lagrangian penalty.

    The residuals and reactions alias Kamino's joint data, which already
    stores kinematic constraints in this row order.

    Args:
        world: World of each row.
        body_a: Body A of each row, or ``-1``.
        body_b: Body B of each row, or ``-1``.
        sparse_a_index: Sparse Jacobian block of body A, or ``-1``.
        sparse_b_index: Sparse Jacobian block of body B.
        num_worlds: Number of worlds.
        residual: Kamino kinematic-constraint residual of each row.
        reaction: Kamino kinematic-constraint multiplier of each row.
        num_bodies: Number of bodies.
        proximal: Whether to allocate the candidate-pose residuals of the joint proximal relaxation.
        device: Device of the arrays.
    """

    def __init__(
        self,
        world: np.ndarray,
        body_a: np.ndarray,
        body_b: np.ndarray,
        sparse_a_index: np.ndarray,
        sparse_b_index: np.ndarray,
        num_worlds: int,
        residual: wp.array[wp.float32],
        reaction: wp.array[wp.float32],
        num_bodies: int,
        proximal: bool,
        device: wp.DeviceLike,
    ):
        self.count = len(world)
        """Total number of rows."""
        self.world = to_warp_int32_array(world, device)
        """World of each row."""
        self.body_a = to_warp_int32_array(body_a, device)
        """Body A of each row, or ``-1``."""
        self.body_b = to_warp_int32_array(body_b, device)
        """Body B of each row, or ``-1``."""
        self.sparse_a_index = to_warp_int32_array(sparse_a_index, device)
        """Sparse Jacobian block of body A, or ``-1``."""
        self.sparse_b_index = to_warp_int32_array(sparse_b_index, device)
        """Sparse Jacobian block of body B."""
        self.jacobian_a = wp.zeros(self.count, dtype=vec6f, device=device)
        """Jacobian of each row with respect to the twist of body A."""
        self.jacobian_b = wp.zeros(self.count, dtype=vec6f, device=device)
        """Jacobian of each row with respect to the twist of body B."""
        self.residual = residual
        """Constraint residual of each row (Kamino's ``r_j``)."""
        self.reaction = reaction
        """Multiplier of each row (Kamino's ``lambda_kin_j``)."""
        self.effective_mass = wp.zeros(self.count, dtype=wp.float32, device=device)
        """Effective mass of each row, including the dynamic-row compliance."""
        self.penalty = wp.zeros(self.count, dtype=wp.float32, device=device)
        """Augmented Lagrangian penalty of each row."""
        self.body_impulse = wp.zeros(num_bodies, dtype=vec6f, device=device)
        """Multiplier impulse ``h J^T lambda`` of the penalized rows on each body, updated with the multipliers."""
        self.candidate_residual = wp.zeros(self.count, dtype=wp.float32, device=device) if proximal else None
        """Residual of each row at the candidate pose, with joint proximal relaxation."""
        self.proximal_defect = wp.zeros(self.count, dtype=wp.float32, device=device) if proximal else None
        """Proximal defect of each row, with joint proximal relaxation."""
        self.residual_velocity_scratch = wp.zeros(self.count, dtype=wp.float32, device=device) if proximal else None
        """Scratch residual velocity of each row, with joint proximal relaxation."""
        self.world_residual = wp.zeros(num_worlds, dtype=wp.float32, device=device)
        """Largest scaled residual of each world."""


class EffortRows:
    """Bounded actuator-effort corrections of dynamic rows.

    Args:
        world: World of each correction.
        dynamic_row_index: Dynamic row of each correction.
        value_index: Kamino effort index of each correction.
        body_offset: First correction of each body in :attr:`body_index`.
        body_index: Corrections touching each body.
        body_side: Side (0 for A, 1 for B) of each body correction.
        world_count: Number of worlds.
        device: Device of the arrays.
    """

    def __init__(
        self,
        world: np.ndarray,
        dynamic_row_index: np.ndarray,
        value_index: np.ndarray,
        body_offset: np.ndarray,
        body_index: np.ndarray,
        body_side: np.ndarray,
        world_count: int,
        device: wp.DeviceLike,
    ):
        self.capacity = len(world)
        """Number of corrections."""
        self.world = to_warp_int32_array(world, device)
        """World of each correction."""
        self.dynamic_row_index = to_warp_int32_array(dynamic_row_index, device)
        """Dynamic row of each correction."""
        self.value_index = to_warp_int32_array(value_index, device)
        """Kamino effort index of each correction."""
        self.body_offset = to_warp_int32_array(body_offset, device)
        """First correction of each body in :attr:`body_index`."""
        self.body_index = to_warp_int32_array(body_index, device)
        """Corrections touching each body."""
        self.body_side = to_warp_int32_array(body_side, device)
        """Side (0 for A, 1 for B) of each body correction."""
        self.intercept = wp.zeros(self.capacity, dtype=wp.float32, device=device)
        """Affine effort intercept of each correction."""
        self.slope = wp.zeros(self.capacity, dtype=wp.float32, device=device)
        """Affine effort slope of each correction."""
        self.impulse_bound = wp.zeros(self.capacity, dtype=wp.float32, device=device)
        """Effort impulse bound of each correction."""
        self.counter = wp.zeros(self.capacity, dtype=wp.float32, device=device)
        """Counter-impulse that clamps each drive to its bound at the latest projected twist.

        It enters the right-hand side of the next candidate solve."""
        self.net_applied = wp.zeros(self.capacity, dtype=wp.float32, device=device)
        """Clamped drive impulse of each correction at the latest projected twist."""
        self.world_residual_max = wp.zeros(world_count, dtype=wp.float32, device=device)
        """Largest effort residual of each world."""


class SplittingState:
    """Body and world state of the LOX splitting iterations.

    Args:
        body_counts: Number of bodies of each world.
        device: Device of the arrays.
    """

    def __init__(self, body_counts: Sequence[int], device: wp.DeviceLike):
        num_worlds = len(body_counts)
        num_bodies = sum(body_counts)
        self.num_worlds = num_worlds
        """Number of worlds."""
        self.num_bodies = num_bodies
        """Number of bodies."""
        self.projected_twist = wp.zeros(num_bodies, dtype=vec6f, device=device)
        """Projected twist ``p`` of each body."""
        self.projected_twist_previous = wp.zeros(num_bodies, dtype=vec6f, device=device)
        """Projected twist of each body at the previous iteration."""
        self.global_twist = wp.zeros(num_bodies, dtype=vec6f, device=device)
        """Candidate twist ``v`` of each body from the smooth solve."""
        self.global_twist_previous = wp.zeros(num_bodies, dtype=vec6f, device=device)
        """Candidate twist of each body at the previous iteration."""
        self.splitting_dual = wp.zeros(num_bodies, dtype=vec6f, device=device)
        """Scaled splitting dual ``u`` of each body."""
        self.world_active = wp.ones(num_worlds, dtype=wp.bool, device=device)
        """Whether each world is still iterating."""
        self.world_converged = wp.zeros(num_worlds, dtype=wp.bool, device=device)
        """Whether each world met the convergence tolerances."""
        self.iteration_count = wp.zeros(num_worlds, dtype=wp.int32, device=device)
        """Number of splitting iterations of each world."""
        self.world_failed = wp.zeros(num_worlds, dtype=wp.bool, device=device)
        """Whether the solve of each world failed: an invalid local block or a non-finite update."""


class LOXProblemData:
    """Constraint rows and per-step body data of a LOX problem.

    The rows and the friction maps are allocated by
    :meth:`LOXProblem.rebuild_dynamic_body_topology`; the per-step body data is
    refreshed at the beginning of each time step.

    Args:
        num_worlds: Number of worlds.
        num_bodies: Total number of bodies.
        world_body_offset: First body of each world (Kamino's ``bodies_offset``).
        world_body_count: Number of bodies of each world (Kamino's ``num_bodies``).
        device: Device of the arrays.
    """

    def __init__(
        self,
        num_worlds: int,
        num_bodies: int,
        world_body_offset: wp.array[wp.int32],
        world_body_count: wp.array[wp.int32],
        device: wp.DeviceLike,
    ):
        self.num_worlds = num_worlds
        """Number of worlds."""
        self.num_bodies = num_bodies
        """Total number of bodies."""
        self.device = wp.get_device(device)
        """Device of the arrays."""
        self.world_body_offset = world_body_offset
        """First body of each world."""
        self.world_body_count = world_body_count
        """Number of bodies of each world."""
        self.dynamic_rows: DynamicRows | None = None
        """Implicit joint-dynamics rows."""
        self.structural_rows: StructuralRows | None = None
        """Joint kinematic rows enforced by an augmented Lagrangian penalty."""
        self.effort_rows: EffortRows | None = None
        """Bounded actuator-effort corrections of the dynamic rows."""
        self.box_rows: BoxRows | None = None
        """Joint friction and joint limit rows."""
        self.contact_rows: ContactRows | None = None
        """Contact rows."""
        self.joint_friction_force: wp.array[wp.float32] | None = None
        """Friction force bound of each joint DOF [N or N m]."""
        self.friction_dof_index: wp.array[wp.int32] | None = None
        """Joint DOF of each friction row."""
        self.friction_multiplier_index: wp.array[wp.int32] | None = None
        """Kamino joint-friction multiplier of each friction row."""
        self.friction_sparse_a_index: wp.array[wp.int32] | None = None
        """Sparse DOF-Jacobian block of body A of each friction row, or ``-1``."""
        self.friction_sparse_b_index: wp.array[wp.int32] | None = None
        """Sparse DOF-Jacobian block of body B of each friction row."""
        self.joint_coordinate_scratch: wp.array[wp.float32] | None = None
        """Scratch joint coordinates of the candidate-pose joint residuals, with joint proximal relaxation."""
        self.joint_velocity_scratch: wp.array[wp.float32] | None = None
        """Scratch joint velocities of the candidate-pose joint residuals, with joint proximal relaxation."""
        self.body_incidence: wp.array2d[wp.int32] | None = None
        """Number of active unilateral rows touching each body."""
        self.body_unilateral_count: wp.array[wp.int32] | None = None
        """Number of active unilateral rows touching each body, a one-dimensional view of
        :attr:`body_incidence`; nonzero for the bodies that have unilateral rows."""
        self.body_weight_enabled: wp.array[wp.int32] | None = None
        """Whether each body receives a splitting weight, if nonzero; aliases :attr:`body_unilateral_count`
        with selective weights."""
        self.block_has_unilateral: wp.array[wp.int32] | None = None
        """Whether unilateral rows touch each factor block, without selective weights."""


class SpatialContactBlocks:
    """Prepared spatial contact metrics of the projection sweeps.

    Args:
        capacity: Number of contacts.
        device: Device of the arrays.
    """

    def __init__(self, capacity: int, device: wp.DeviceLike):
        self.delassus = wp.zeros(capacity, dtype=mat66f, device=device)
        """Mass-split, compliant normal-first metric of each contact."""
        self.eigenvectors = wp.zeros(capacity, dtype=mat55f, device=device)
        """Eigenvectors (columns) of the scaled friction Schur complement."""
        self.eigenvalues = wp.zeros(capacity, dtype=vec5f, device=device)
        """Eigenvalues of the scaled friction Schur complement."""


class ColoredRowsData:
    """Colors of one row family and its active rows ordered by color.

    Args:
        capacity: Number of rows.
        color_count: Number of colors.
        device: Device of the arrays.
    """

    def __init__(self, capacity: int, color_count: int, device: wp.DeviceLike):
        self.capacity = capacity
        """Number of rows."""
        self.color = wp.full(capacity, -1, dtype=wp.int32, device=device)
        """Color of each row, or ``-1`` for an inactive row."""
        self.proposal = wp.full(capacity, -1, dtype=wp.int32, device=device)
        """Repair color proposal of each row, or ``-1``."""
        self.color_count = wp.zeros(color_count, dtype=wp.int32, device=device)
        """Number of active rows of each color."""
        self.color_offset = wp.zeros(color_count, dtype=wp.int32, device=device)
        """Start of each color in :attr:`order`."""
        self.cursor = wp.zeros(color_count, dtype=wp.int32, device=device)
        """Scatter cursor of each color while sorting the rows."""
        self.order = wp.full(capacity, -1, dtype=wp.int32, device=device)
        """Active rows sorted by color."""


class ColoringData:
    """Gauss--Seidel coloring of the box rows and contacts.

    The color count is bounded by the total row capacity so that every color can be occupied.

    Args:
        problem_data: Rows and body data of the LOX problem.
        color_count: Maximum number of colors.
    """

    def __init__(self, problem_data: LOXProblemData, color_count: int):
        device = problem_data.device
        box_capacity = problem_data.box_rows.capacity
        contact_capacity = problem_data.contact_rows.capacity
        self.color_count = max(1, min(color_count, box_capacity + contact_capacity))
        """Number of colors."""
        self.occupancy = wp.zeros((problem_data.num_bodies, self.color_count), dtype=wp.int32, device=device)
        """Number of active rows of each color touching each body."""
        self.world_color_count = wp.zeros((problem_data.num_worlds, self.color_count), dtype=wp.int32, device=device)
        """Number of active rows of each color in each world, or ``None`` unless the sweeps run per world."""
        self.world_color_cursor = wp.zeros_like(self.world_color_count)
        """Next row of each color of each world in :attr:`world_order`, or ``None`` unless the sweeps run per world."""
        self.world_order = wp.zeros(box_capacity + contact_capacity, dtype=wp.int32, device=device)
        """Active rows of each world sorted by color, or ``None`` unless the sweeps run per world.

        The rows of a world start at the sum of its box and contact offsets; box rows keep their index and
        contacts are offset by the box capacity.
        """
        self.body_lock = wp.zeros(problem_data.num_bodies, dtype=wp.int32, device=device)
        """Repair lock of each body, reset before every repair pass."""
        self.box = ColoredRowsData(box_capacity, self.color_count, device)
        """Colors of the box rows."""
        self.contact = ColoredRowsData(contact_capacity, self.color_count, device)
        """Colors of the contacts."""


class AndersonRowsData:
    """The five vectors of a recyclable Anderson(1) secant over one row family.

    Args:
        capacity: Number of rows.
        dtype: Reaction type of the rows.
        device: Device of the arrays.
    """

    def __init__(self, capacity: int, dtype, device: wp.DeviceLike):
        self.capacity = capacity
        """Number of rows."""
        self.map_input = wp.zeros(capacity, dtype=dtype, device=device)
        """Reaction input of the current projection map."""
        self.sample_residual = wp.zeros_like(self.map_input)
        """Map residual of the stored sample."""
        self.sample_image = wp.zeros_like(self.map_input)
        """Map image of the stored sample."""
        self.direction_residual = wp.zeros_like(self.map_input)
        """Residual difference of the stored secant direction."""
        self.direction_image = wp.zeros_like(self.map_input)
        """Image difference of the stored secant direction."""


class AndersonData:
    """Samples, secant direction, and per-world coefficients of Anderson-accelerated projection.

    Args:
        problem_data: Rows and body data of the LOX problem.
    """

    def __init__(self, problem_data: LOXProblemData):
        device = problem_data.device
        world_count = problem_data.num_worlds
        rows = problem_data.contact_rows
        self.box = AndersonRowsData(problem_data.box_rows.capacity, wp.float32, device)
        """Secant vectors of the box rows."""
        self.contact = AndersonRowsData(rows.capacity, wp.vec3f, device)
        """Secant vectors of the contacts."""
        self.angular = AndersonRowsData(rows.capacity, wp.vec3f, device) if rows.angular_reaction is not None else None
        """Secant vectors of the angular contact reactions, with spatial friction."""
        self.capacity = self.box.capacity + self.contact.capacity
        """Number of rows."""
        self.global_reduction = world_count == 1 and self.capacity > 0 and wp.get_device(device).is_cuda
        """Whether one thread block reduces the secant terms of the single world; atomics reduce them otherwise."""
        self.terms = wp.zeros(self.capacity, dtype=wp.vec3f, device=device) if self.global_reduction else None
        """Per-row terms of the secant reductions, with the single-world block reduction."""
        self.gram = wp.zeros(world_count, dtype=wp.float32, device=device)
        """Squared norm of the residual difference of each world."""
        self.rhs = wp.zeros(world_count, dtype=wp.float32, device=device)
        """Projection of the residual on the residual difference of each world."""
        self.residual_norm = wp.zeros(world_count, dtype=wp.float32, device=device)
        """Map residual norm of each world."""
        self.coefficient = wp.zeros(world_count, dtype=wp.float32, device=device)
        """Mixing coefficient of each world."""
        self.previous_residual_norm = wp.zeros(world_count, dtype=wp.float32, device=device)
        """Map residual norm of each world at the previous map."""
        self.sample_valid = wp.zeros(world_count, dtype=wp.int32, device=device)
        """Whether each world has a raw sample."""
        self.direction_valid = wp.zeros(world_count, dtype=wp.int32, device=device)
        """Whether each world has a recyclable direction."""
        self.guard_pending = wp.zeros(world_count, dtype=wp.int32, device=device)
        """Whether the safeguard of each world must check the next map."""


class APGDData:
    """Momentum state of APGD-accelerated Jacobi projection.

    Args:
        problem_data: Rows and body data of the LOX problem.
    """

    def __init__(self, problem_data: LOXProblemData):
        device = problem_data.device
        world_count = problem_data.num_worlds
        box_capacity = problem_data.box_rows.capacity
        rows = problem_data.contact_rows
        spatial = rows.angular_reaction is not None
        self.theta = wp.ones(world_count, dtype=wp.float32, device=device)
        """Momentum parameter of each world."""
        self.beta = wp.zeros(world_count, dtype=wp.float32, device=device)
        """Extrapolation factor of each world."""
        self.restart_dot = wp.zeros(world_count, dtype=wp.float32, device=device)
        """Restart criterion of each world."""
        self.box_trial = wp.zeros(box_capacity, dtype=wp.float32, device=device)
        """Extrapolated trial reaction of each box row."""
        self.box_previous = wp.zeros(box_capacity, dtype=wp.float32, device=device)
        """Reaction of each box row at the previous sweep."""
        self.contact_trial = wp.zeros(rows.capacity, dtype=wp.vec3f, device=device)
        """Extrapolated trial reaction of each contact."""
        self.contact_previous = wp.zeros(rows.capacity, dtype=wp.vec3f, device=device)
        """Reaction of each contact at the previous sweep."""
        self.angular_trial = wp.zeros(rows.capacity, dtype=wp.vec3f, device=device) if spatial else None
        """Extrapolated trial angular reaction of each contact, with spatial friction."""
        self.angular_previous = wp.zeros(rows.capacity, dtype=wp.vec3f, device=device) if spatial else None
        """Angular reaction of each contact at the previous sweep, with spatial friction."""


class ProjectionData:
    """Local blocks, sweep buffer, and coloring or acceleration state of the projection schedule.

    Args:
        problem_data: Rows and body data of the LOX problem.
        color_count: Maximum number of Gauss--Seidel colors; one selects Jacobi.
        acceleration: Acceleration of the projection sweeps.
    """

    def __init__(self, problem_data: LOXProblemData, color_count: int, acceleration: ProjectionAcceleration):
        device = problem_data.device
        box_capacity = problem_data.box_rows.capacity
        contact_capacity = problem_data.contact_rows.capacity
        self.coloring = ColoringData(problem_data, color_count) if color_count > 1 else None
        """Gauss--Seidel coloring, or ``None`` for Jacobi."""
        colored = self.coloring is not None
        spatial = problem_data.contact_rows.frame is not None
        self.box_delassus = wp.zeros(box_capacity, dtype=wp.float32, device=device)
        """Mass-split local Delassus coefficient of each box row."""
        self.contact_delassus = None if spatial else wp.zeros(contact_capacity, dtype=wp.mat33f, device=device)
        """Mass-split, regularized local Delassus block of each contact, including its normal compliance.

        ``None`` for spatial contacts, whose metrics are :attr:`spatial_blocks`.
        """
        self.box_smoothing_delassus = wp.zeros(box_capacity, dtype=wp.float32, device=device) if colored else None
        """Jacobi Delassus coefficient of each box row for the final smoothing sweep."""
        self.contact_smoothing_delassus = (
            wp.zeros(contact_capacity, dtype=wp.mat33f, device=device) if colored and not spatial else None
        )
        """Jacobi Delassus block of each contact for the final smoothing sweep, or ``None`` for spatial contacts."""
        self.spatial_blocks = SpatialContactBlocks(contact_capacity, device) if spatial else None
        """Spatial contact metrics of the sweeps, with angular friction."""
        self.spatial_smoothing_blocks = SpatialContactBlocks(contact_capacity, device) if spatial and colored else None
        """Spatial contact metrics of the final smoothing sweep, with angular friction."""
        self.twist_delta = wp.zeros(problem_data.num_bodies, dtype=vec6f, device=device)
        """Per-body correction accumulated by one sweep, zero between sweeps."""
        self.anderson = AndersonData(problem_data) if acceleration == ProjectionAcceleration.ANDERSON else None
        """Anderson acceleration state, if selected."""
        self.apgd = APGDData(problem_data) if acceleration == ProjectionAcceleration.APGD else None
        """APGD acceleration state, if selected."""


class LOXSystemData:
    """Arrays of the batched smooth body system.

    Bodies stay in model order, while the dense matrices and vectors are packed by
    factor block.

    Args:
        body_block: Factor block of each body, or ``-1`` for a prescribed body.
        body_local: Index of each body within its factor block, or ``-1``.
        body_vector_index: First entry of each body in the packed system vectors, or ``-1``.
        info: Dense layout of the factor blocks.
        device: Device of the arrays.
    """

    def __init__(
        self,
        body_block: np.ndarray,
        body_local: np.ndarray,
        body_vector_index: np.ndarray,
        info: DenseBlockInfo,
        device: wp.DeviceLike,
    ):
        num_bodies = len(body_block)
        self.body_block = to_warp_int32_array(body_block, device)
        """Factor block of each body, or ``-1`` for a prescribed body."""
        self.body_local = to_warp_int32_array(body_local, device)
        """Index of each body within its factor block, or ``-1``."""
        self.body_vector_index = to_warp_int32_array(body_vector_index, device)
        """First entry of each body in the packed system vectors, or ``-1``."""
        self.info = info
        """Dense layout of the factor blocks."""
        self.matrix = wp.zeros(info.total_mat_size, dtype=wp.float32, device=device)
        """Dense matrix of each factor block: the smooth matrix ``A`` once assembled, then ``A + W``
        once the splitting weights are added."""
        self.right_hand_side = wp.zeros(info.total_vec_size, dtype=wp.float32, device=device)
        """Smooth right-hand side ``f``, frozen during the iterations."""
        self.candidate_right_hand_side = wp.zeros(info.total_vec_size, dtype=wp.float32, device=device)
        """Right-hand side of the candidate solve of the current splitting iteration."""
        self.packed_solution = wp.zeros(info.total_vec_size, dtype=wp.float32, device=device)
        """Packed solution of the candidate solve."""
        self.weight = wp.zeros(num_bodies, dtype=mat66f, device=device)
        """Splitting weight ``W`` of each body."""
        self.inverse_weight = wp.zeros(num_bodies, dtype=mat66f, device=device)
        """Inverse splitting weight of each body."""

    def zero(self) -> None:
        """Clear the matrix before an assembly.

        The joint rows accumulate into the matrix. The body terms overwrite the diagonal blocks and
        the right-hand side, and the other arrays are overwritten before they are read.
        """
        self.matrix.zero_()


class LOXResiduals:
    """Per-world residuals of the LOX splitting iterations.

    The scaled residuals are relative to their tolerances and drive the convergence test;
    the physical residuals are reported in the terminal status.

    Args:
        num_worlds: Number of worlds.
        device: Device of the arrays.
    """

    def __init__(self, num_worlds: int, device: wp.DeviceLike):
        self.r_change = wp.zeros(num_worlds, dtype=wp.float32, device=device)
        """Scaled iterate change of each world."""
        self.r_split = wp.zeros(num_worlds, dtype=wp.float32, device=device)
        """Scaled splitting residual ``|v - p|`` of each world."""
        self.r_cross_iterate = wp.zeros(num_worlds, dtype=wp.float32, device=device)
        """Scaled cross-iterate residual of each world."""
        self.r_unilateral = wp.zeros(num_worlds, dtype=wp.float32, device=device)
        """Velocity-scaled natural-map residual of the unilateral rows of each world, relative to the tolerance."""
        self.r_lagged = wp.zeros(num_worlds, dtype=wp.float32, device=device)
        """Largest change ``|J (v - p_prev)|`` of the unilateral row velocities of each world over the last
        iteration, relative to the velocity tolerance.

        The splitting consensus ``v = p`` and the unilateral laws can hold at every iterate while
        the iterates still move; this velocity-level dual residual detects that. The body-space
        change residuals are position-level and ``dt`` times looser. Joint rows are excluded: their
        bodies are unweighted unless incident to a unilateral row, so ``p = v`` there and the check
        would only tighten the structural solve to velocity level.
        """
        self.r_primal = wp.zeros(num_worlds, dtype=wp.float32, device=device)
        """Largest consensus gap ``|W (v - p)|`` [N·s or N·m·s] of each world at its last iteration."""
        self.r_dual = wp.zeros(num_worlds, dtype=wp.float32, device=device)
        """Largest change ``|v - v_prev|`` [m/s or rad/s] of the smooth solution of each world at its last iteration."""


class LOXData:
    """Solver state, parameters, and outputs of the LOX solver.

    Args:
        body_counts: Number of bodies of each world.
        solution: Dual solution container for the solution metrics, or ``None`` without them.
        joint_penalty_scale: Initial dimensionless structural ALM penalty scale of every world.
        device: Device of the arrays.
    """

    def __init__(
        self,
        body_counts: Sequence[int],
        solution: DualSolution | None,
        joint_penalty_scale: float,
        device: wp.DeviceLike,
    ):
        num_worlds = len(body_counts)
        self.state = SplittingState(body_counts, device)
        """Body and world state of the splitting iterations."""
        self.residuals = LOXResiduals(num_worlds, device)
        """Per-world residuals of the splitting iterations."""
        self.status = wp.empty(num_worlds, dtype=LOXStatus, device=device)
        """Terminal status of each world."""
        self.joint_penalty_scale = wp.full(num_worlds, joint_penalty_scale, dtype=wp.float32, device=device)
        """Dimensionless structural ALM penalty scale of each world."""
        self.dual_wrench = wp.zeros(sum(body_counts), dtype=vec6f, device=device)
        """Splitting dual of each body as the generalized wrench ``W u / dt`` [N, N·m], kept across time steps.

        The next time step converts it back to its splitting dual with its own body weights; failed worlds
        store zero. Like the contact and limit warm starts, it belongs to the solver.
        """
        self.solution = solution
        """Dual solution of the last solve in Kamino's constraint layout.

        LOX solves for body twists and row reactions; this translation is only filled by
        :meth:`LOXSolver.build_dual_solution` to evaluate the solution metrics, and is ``None``
        when the solver was built without them.
        """
