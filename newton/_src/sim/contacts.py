# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import warp as wp
from warp import DeviceLike as Devicelike

from ..utils.deprecation import deprecate_nonkeyword_arguments

GENERATION_SENTINEL = -1
"""Value reserved as an impossible generation; the increment kernel skips it."""


@wp.kernel(enable_backward=False)
def _increment_contact_generation(generation: wp.array[wp.int32]):
    g = generation[0]
    if g == 2147483647:
        g = 0
    else:
        g = g + 1
    generation[0] = g


@wp.kernel(enable_backward=False)
def _clear_counters_and_bump_generation(
    counters: wp.array[wp.int32],
    generation: wp.array[wp.int32],
    num_counters: int,
    bump_generation: int,
):
    """Zero counter array and optionally increment generation in one kernel launch."""
    tid = wp.tid()
    if tid < num_counters:
        counters[tid] = 0
    if tid == 0 and bump_generation != 0:
        g = generation[0]
        if g == 2147483647:
            g = 0
        else:
            g = g + 1
        generation[0] = g


@wp.func
def contact_surface_separation(
    point0_world: wp.vec3,
    point1_world: wp.vec3,
    normal: wp.vec3,
    margin0: float,
    margin1: float,
) -> float:
    """Signed separation between the two effective contact surfaces along the normal.

    Positive values are a gap; negative values are penetration.

    Args:
        point0_world: Support-shape contact point on shape 0 [m], world space.
        point1_world: Support-shape contact point on shape 1 [m], world space.
        normal: Unit contact normal pointing from shape 0 toward shape 1.
        margin0: Effective surface thickness of shape 0 [m] (effective radius + margin).
        margin1: Effective surface thickness of shape 1 [m].

    Returns:
        Separation between the effective surfaces [m].
    """
    return wp.dot(normal, point1_world - point0_world) - (margin0 + margin1)


@wp.func
def contact_surface_point(
    X_wb: wp.transform,
    point_local: wp.vec3,
    offset_local: wp.vec3,
) -> wp.vec3:
    """World-space effective-surface contact point for one shape.

    Shifts the body-frame support point by the body-frame surface offset and maps the result
    to world space. Because the offset is expressed in the body frame, a persisted/reused
    contact tracks the material point under rotation.

    Args:
        X_wb: Body-to-world transform of the shape's body.
        point_local: Support-shape contact point in the body frame [m].
        offset_local: Surface offset in the body frame [m] (effective thickness along the normal).

    Returns:
        Effective-surface contact point [m], world space.
    """
    return wp.transform_point(X_wb, point_local + offset_local)


class Contacts:
    """
    Stores contact information for rigid and soft body collisions, to be consumed by a solver.

    This class manages buffers for contact data such as positions, normals, margins, and shape indices
    for both rigid-rigid and soft-rigid contacts. The buffers are allocated on the specified device and can
    optionally require gradients for differentiable simulation.

    .. experimental::

        This class is a temporary solution and its interface may change without
        prior notice.
    """

    EXTENDED_ATTRIBUTES: frozenset[str] = frozenset(("force",))
    """
    Names of optional extended contact attributes that are not allocated by default.

    These can be requested via :meth:`newton.ModelBuilder.request_contact_attributes` or
    :meth:`newton.Model.request_contact_attributes` before calling
    :meth:`newton.CollisionPipeline.contacts`. When constructing :class:`newton.Contacts` directly,
    pass the names via ``requested_attributes``.

    See :ref:`extended_contact_attributes` for details and usage.
    """

    @classmethod
    def validate_extended_attributes(cls, attributes: tuple[str, ...]) -> None:
        """Validate names passed to request_contact_attributes().

        Only extended contact attributes listed in :attr:`EXTENDED_ATTRIBUTES` are accepted.

        Args:
            attributes: Tuple of attribute names to validate.

        Raises:
            ValueError: If any attribute name is not in :attr:`EXTENDED_ATTRIBUTES`.
        """
        if not attributes:
            return

        invalid = sorted(set(attributes).difference(cls.EXTENDED_ATTRIBUTES))
        if invalid:
            allowed = ", ".join(sorted(cls.EXTENDED_ATTRIBUTES))
            bad = ", ".join(invalid)
            raise ValueError(f"Unknown extended contact attribute(s): {bad}. Allowed: {allowed}.")

    @deprecate_nonkeyword_arguments
    def __init__(
        self,
        rigid_contact_max: int,
        soft_contact_max: int,
        *,
        soft_contact_tids_size: int | None = None,
        requires_grad: bool = False,
        device: Devicelike = None,
        per_contact_shape_properties: bool = False,
        clear_buffers: bool = False,
        requested_attributes: set[str] | None = None,
        contact_matching: bool = False,
        contact_report: bool = False,
    ):
        """
        Initialize Contacts storage.

        Args:
            rigid_contact_max: Maximum number of rigid contacts
            soft_contact_max: Maximum number of soft contacts
            soft_contact_tids_size: Length of the internal per-thread replay-index array
                (``soft_contact_tids``) used for differentiable backward. Defaults to
                ``soft_contact_max``; the collision pipeline sets it to the full
                particle + edge + face candidate-pair count so a custom (smaller)
                ``soft_contact_max`` cannot drop a launch thread's replay slot.
            requires_grad: Whether contact arrays require gradients for differentiable
                simulation.  When ``True``, soft contact arrays (body_pos, body_vel, normal)
                are allocated with gradients so that gradient-based optimization can flow
                through particle-shape contacts, **and** additional differentiable rigid
                contact arrays are allocated (``rigid_contact_diff_*``) that provide
                first-order gradients of contact distance and world-space points with
                respect to body poses.
            device: Device to allocate buffers on
            per_contact_shape_properties: Enable per-contact stiffness/damping/friction arrays
            clear_buffers: If True, clear() will zero all contact buffers (slower but conservative).
                If False (default), clear() only resets counts in a single fused kernel launch,
                relying on collision detection to overwrite active contacts. This is much faster
                than the conservative path and safe since solvers only read up to contact_count.
            requested_attributes: Set of extended contact attribute names to allocate.
                See :attr:`EXTENDED_ATTRIBUTES` for available options.
            contact_matching: Allocate a per-contact match index array
                (:attr:`rigid_contact_match_index`) that stores frame-to-frame
                contact correspondences filled by the collision pipeline.
            contact_report: Allocate compact index lists of new and broken
                contacts (:attr:`rigid_contact_new_indices`,
                :attr:`rigid_contact_new_count`,
                :attr:`rigid_contact_broken_indices`,
                :attr:`rigid_contact_broken_count`) populated each frame by
                the collision pipeline.  Requires ``contact_matching=True``.

        .. experimental::

            The ``rigid_contact_diff_*`` arrays allocated when
            ``requires_grad=True`` may change without prior notice; see
            :meth:`newton.CollisionPipeline.collide`.
        """
        if contact_report and not contact_matching:
            raise ValueError("contact_report=True requires contact_matching=True")
        self.per_contact_shape_properties = per_contact_shape_properties
        self.clear_buffers = clear_buffers
        with wp.ScopedDevice(device):
            # One int32[2] array holding two independent contact counts: [0] rigid, [1] soft.
            # rigid_contact_count (the [0:1] view) and soft_contact_count (the [1:2] view) index
            # into this same array, so each remains a separate count; they share one array only so
            # a single kernel can reset both to zero in one launch instead of two. The reset
            # happens at the start of every collision pass -- folded into the first kernel that
            # runs, compute_shape_aabbs -- and clear() resets them as well.
            self.contact_counters = wp.zeros(2, dtype=wp.int32)
            # Sliced view for the rigid counter (no additional allocation)
            self.rigid_contact_count = self.contact_counters[0:1]

            self.contact_generation = wp.zeros(1, dtype=wp.int32)
            """Device-side generation counter, incremented each time :meth:`clear` is called.

            Solvers can compare this against a cached value to detect whether the
            contact set changed since the last conversion pass."""

            # rigid contacts — never requires_grad (narrow phase has enable_backward=False)
            self.rigid_contact_point_id = wp.zeros(rigid_contact_max, dtype=wp.int32)
            self.rigid_contact_shape0 = wp.full(rigid_contact_max, -1, dtype=wp.int32)
            self.rigid_contact_shape1 = wp.full(rigid_contact_max, -1, dtype=wp.int32)
            self.rigid_contact_point0 = wp.zeros(rigid_contact_max, dtype=wp.vec3)
            """Body-frame contact point on shape 0 [m], shape (rigid_contact_max,), dtype :class:`vec3`."""
            self.rigid_contact_point1 = wp.zeros(rigid_contact_max, dtype=wp.vec3)
            """Body-frame contact point on shape 1 [m], shape (rigid_contact_max,), dtype :class:`vec3`."""
            self.rigid_contact_offset0 = wp.zeros(rigid_contact_max, dtype=wp.vec3)
            """Body-frame friction anchor offset for shape 0 [m], shape (rigid_contact_max,), dtype :class:`vec3`.

            Equal to the contact normal scaled by ``effective_radius + margin`` and
            expressed in shape 0's body frame. Combined with
            ``rigid_contact_point0`` to form a shifted friction anchor that accounts
            for rotational effects of finite contact thickness in tangential friction
            calculations."""
            self.rigid_contact_offset1 = wp.zeros(rigid_contact_max, dtype=wp.vec3)
            """Body-frame friction anchor offset for shape 1 [m], shape (rigid_contact_max,), dtype :class:`vec3`.

            Equal to the contact normal scaled by ``effective_radius + margin`` and
            expressed in shape 1's body frame. Combined with
            ``rigid_contact_point1`` to form a shifted friction anchor that accounts
            for rotational effects of finite contact thickness in tangential friction
            calculations."""
            self.rigid_contact_normal = wp.zeros(rigid_contact_max, dtype=wp.vec3)
            """Contact normal pointing from shape 0 toward shape 1 (A-to-B) [unitless], shape (rigid_contact_max,), dtype :class:`vec3`."""
            self.rigid_contact_margin0 = wp.zeros(rigid_contact_max, dtype=wp.float32)
            """Surface thickness for shape 0: effective radius + margin [m], shape (rigid_contact_max,), dtype float."""
            self.rigid_contact_margin1 = wp.zeros(rigid_contact_max, dtype=wp.float32)
            """Surface thickness for shape 1: effective radius + margin [m], shape (rigid_contact_max,), dtype float."""
            self.rigid_contact_tids = wp.full(rigid_contact_max, -1, dtype=wp.int32)
            # to be filled by the solver (currently unused)
            self.rigid_contact_force = wp.zeros(rigid_contact_max, dtype=wp.vec3)
            """Contact force [N], shape (rigid_contact_max,), dtype :class:`vec3`."""

            # Differentiable rigid contact arrays -- only allocated when requires_grad
            # is True.  Populated by the post-processing kernel in
            # :mod:`newton._src.geometry.differentiable_contacts`.
            if requires_grad:
                self.rigid_contact_diff_distance = wp.zeros(rigid_contact_max, dtype=wp.float32, requires_grad=True)
                """Differentiable signed distance [m], shape (rigid_contact_max,), dtype float."""
                self.rigid_contact_diff_normal = wp.zeros(rigid_contact_max, dtype=wp.vec3, requires_grad=False)
                """Contact normal (A-to-B, world frame) [unitless], shape (rigid_contact_max,), dtype :class:`vec3`."""
                self.rigid_contact_diff_point0_world = wp.zeros(rigid_contact_max, dtype=wp.vec3, requires_grad=True)
                """World-space contact point on shape 0 [m], shape (rigid_contact_max,), dtype :class:`vec3`."""
                self.rigid_contact_diff_point1_world = wp.zeros(rigid_contact_max, dtype=wp.vec3, requires_grad=True)
                """World-space contact point on shape 1 [m], shape (rigid_contact_max,), dtype :class:`vec3`."""
            else:
                self.rigid_contact_diff_distance = None
                """Differentiable signed distance [m], shape (rigid_contact_max,), dtype float."""
                self.rigid_contact_diff_normal = None
                """Contact normal (A-to-B, world frame) [unitless], shape (rigid_contact_max,), dtype :class:`vec3`."""
                self.rigid_contact_diff_point0_world = None
                """World-space contact point on shape 0 [m], shape (rigid_contact_max,), dtype :class:`vec3`."""
                self.rigid_contact_diff_point1_world = None
                """World-space contact point on shape 1 [m], shape (rigid_contact_max,), dtype :class:`vec3`."""

            # contact stiffness/damping/friction (only allocated if per_contact_shape_properties is enabled)
            if self.per_contact_shape_properties:
                self.rigid_contact_stiffness = wp.zeros(rigid_contact_max, dtype=wp.float32)
                """Per-contact stiffness [N/m], shape (rigid_contact_max,), dtype float."""
                self.rigid_contact_damping = wp.zeros(rigid_contact_max, dtype=wp.float32)
                """Per-contact damping [N·s/m], shape (rigid_contact_max,), dtype float."""
                self.rigid_contact_friction = wp.zeros(rigid_contact_max, dtype=wp.float32)
                """Per-contact friction coefficient [dimensionless], shape (rigid_contact_max,), dtype float."""
            else:
                self.rigid_contact_stiffness = None
                """Per-contact stiffness [N/m], shape (rigid_contact_max,), dtype float."""
                self.rigid_contact_damping = None
                """Per-contact damping [N·s/m], shape (rigid_contact_max,), dtype float."""
                self.rigid_contact_friction = None
                """Per-contact friction coefficient [dimensionless], shape (rigid_contact_max,), dtype float."""

            # Contact matching index — filled by the collision pipeline when
            # contact_matching is enabled.
            self.contact_matching = contact_matching
            self.contact_report = contact_report
            if contact_matching:
                self.rigid_contact_match_index = wp.full(rigid_contact_max, -1, dtype=wp.int32)
                """Per-contact match index from frame-to-frame matching.

                Non-negative elements index matching contacts in the previous sorted contact buffer.
                Negative elements indicate new or broken contacts.
                Shape (rigid_contact_max,), dtype int32."""
            else:
                self.rigid_contact_match_index = None

            if contact_report:
                self.rigid_contact_new_indices = wp.zeros(rigid_contact_max, dtype=wp.int32)
                """Indices of new contacts in the current sorted buffer.

                Valid after the collision pipeline runs.
                Shape (rigid_contact_max,), dtype int32."""
                self.rigid_contact_new_count = wp.zeros(1, dtype=wp.int32)
                """Device-side count of new contacts (single-element int32)."""
                self.rigid_contact_broken_indices = wp.zeros(rigid_contact_max, dtype=wp.int32)
                """Indices of broken contacts in the previous frame's sorted buffer.

                Valid after the collision pipeline runs.
                Shape (rigid_contact_max,), dtype int32."""
                self.rigid_contact_broken_count = wp.zeros(1, dtype=wp.int32)
                """Device-side count of broken contacts (single-element int32)."""
            else:
                self.rigid_contact_new_indices = None
                self.rigid_contact_new_count = None
                self.rigid_contact_broken_indices = None
                self.rigid_contact_broken_count = None

            # requires_grad flows through the soft-contact arrays below for differentiable simulation.
            # soft_contact_count is the [1:2] view of contact_counters above -- the total number of
            # soft (particle + edge + face) contacts. With the full-surface flag off, only the
            # particle pass emits records, so this equals the particle-contact count and is
            # bit-identical in shape and meaning to a build without the feature.
            self.soft_contact_count = self.contact_counters[1:2]
            # The soft-contact data arrays below are all length soft_contact_max and share one index
            # space. Each record self-describes its feature kind through soft_contact_indices -- the
            # soft-side particle ids, -1 padded -- paired with soft_contact_barycentric:
            #   particle contact:  indices = (p,  -1, -1),  barycentric = (1, 0, 0)
            #   edge contact:      indices = (v0, v1, -1),  barycentric = (u, 1 - u, 0)
            #   face contact:      indices = (v0, v1, v2),  barycentric = (w0, w1, w2)
            # so the number of non-negative index slots gives the kind; there is no per-record kind
            # flag and no per-kind count ranges. The contact point is
            # sum_i barycentric[i] * particle_q[indices[i]] over the non-negative slots.
            self.soft_contact_indices = wp.full(soft_contact_max, wp.vec3i(-1, -1, -1), dtype=wp.vec3i)
            """Soft-side particle ids per contact, -1 padded [dimensionless], shape (soft_contact_max,), dtype :class:`vec3i`.

            Particle contact ``(p, -1, -1)``, edge contact ``(v0, v1, -1)``, face contact
            ``(v0, v1, v2)``. Pair with :attr:`soft_contact_barycentric` to recover the contact
            point over the non-negative slots."""
            # Particle-only view kept for solvers that consume particle contacts exclusively (XPBD,
            # semi-implicit, Style3D). Holds the particle id for particle contacts; -1 for edge/face.
            self.soft_contact_particle = wp.full(soft_contact_max, -1, dtype=int)
            """Particle id per particle contact, -1 for edge/face records [dimensionless], shape (soft_contact_max,), dtype int.

            The particle-only view of :attr:`soft_contact_indices`; use ``soft_contact_indices`` for
            full-surface (edge/face) contacts."""
            self.soft_contact_barycentric = wp.zeros(soft_contact_max, dtype=wp.vec3, requires_grad=requires_grad)
            """Barycentric weights of the contact point on the soft feature's particles [unitless], shape (soft_contact_max,), dtype :class:`vec3`."""
            self.soft_contact_shape = wp.full(soft_contact_max, -1, dtype=int)
            self.soft_contact_body_pos = wp.zeros(soft_contact_max, dtype=wp.vec3, requires_grad=requires_grad)
            """Contact position on body [m], shape (soft_contact_max,), dtype :class:`vec3`.

            Point on the raw (un-inflated) shape surface; per-shape ``shape_margin`` is
            applied analytically by consumers rather than baked into this point."""
            self.soft_contact_body_vel = wp.zeros(soft_contact_max, dtype=wp.vec3, requires_grad=requires_grad)
            """Contact velocity on body [m/s], shape (soft_contact_max,), dtype :class:`vec3`."""
            self.soft_contact_normal = wp.zeros(soft_contact_max, dtype=wp.vec3, requires_grad=requires_grad)
            """Contact normal direction [unitless], shape (soft_contact_max,), dtype :class:`vec3`."""
            # Replay index array for differentiable backward: recorded per launch *thread*, not per
            # contact, so it must span the full particle+edge+face candidate-pair space -- which can
            # exceed soft_contact_max when the caller overrides that capacity. Sized independently so a
            # smaller soft_contact_max never drops a thread's replay slot. Defaults to soft_contact_max.
            _tids_size = soft_contact_tids_size if soft_contact_tids_size is not None else soft_contact_max
            self.soft_contact_tids = wp.full(_tids_size, -1, dtype=int)

            # Private capability flag: set by the collision pipeline when full-surface (edge/face)
            # soft contacts are enabled, so soft_contact_indices may hold edge/face records. Solvers
            # that only consume particle contacts (everything but VBD) raise on this rather than
            # silently misreading edge/face records -- the pipeline is solver-agnostic, so the check
            # lives at the consuming solver. Kept private to avoid a public API/deprecation surface.
            self._enable_rigid_soft_full_surface_contact = False

            # Extended contact attributes (optional, allocated on demand)
            self.force: wp.array | None = None
            """Contact forces (spatial) [N, N·m], shape (rigid_contact_max + soft_contact_max,), dtype :class:`spatial_vector`.
            Force and torque exerted on body0 by body1, referenced to the center of mass (COM) of body0, and in world frame, where body0 and body1 are the bodies of shape0 and shape1.
            First three entries: linear force [N]; last three entries: torque (moment) [N·m].
            When both rigid and soft contacts are present, soft contact forces follow rigid contact forces.

            This is an extended contact attribute; see :ref:`extended_contact_attributes` for more information.
            """
            if requested_attributes and "force" in requested_attributes:
                total_contacts = rigid_contact_max + soft_contact_max
                self.force = wp.zeros(total_contacts, dtype=wp.spatial_vector, requires_grad=requires_grad)

        self.requires_grad = requires_grad

        self.rigid_contact_max = rigid_contact_max
        self.soft_contact_max = soft_contact_max

    def clear(self, bump_generation: bool = True):
        """
        Clear contact data, resetting counts and optionally clearing all buffers.

        By default (clear_buffers=False), only resets contact counts. This is highly optimized,
        requiring just a single fused kernel launch that zeroes all counters and bumps the
        generation counter. Collision detection overwrites all data up to the new
        contact_count, and solvers only read up to count, so clearing stale data is unnecessary.

        If clear_buffers=True (conservative mode), performs full buffer clearing with sentinel
        values and zeros. This requires several additional kernel launches but may be useful for debugging.

        Args:
            bump_generation: If True (default), increment ``contact_generation`` to invalidate
                previously-observed contact data. Callers that will immediately re-bump the
                generation via another fused kernel (e.g. :func:`compute_shape_aabbs`) can pass
                ``False`` to avoid an unnecessary double-bump per collision pass.
        """
        # Clear all counters and (optionally) bump generation in a single kernel launch.
        num_counters = self.contact_counters.shape[0]
        wp.launch(
            _clear_counters_and_bump_generation,
            dim=max(num_counters, 1),
            inputs=[self.contact_counters, self.contact_generation, num_counters, int(bump_generation)],
            device=self.contact_generation.device,
            record_tape=False,
        )

        if self.clear_buffers:
            # Conservative path: clear all buffers with sentinel values and zeros.
            # Slower than the fast path but may be useful for debugging or special cases.
            self.rigid_contact_shape0.fill_(-1)
            self.rigid_contact_shape1.fill_(-1)
            self.rigid_contact_tids.fill_(-1)
            self.rigid_contact_force.zero_()

            if self.force is not None:
                self.force.zero_()

            if self.rigid_contact_diff_distance is not None:
                self.rigid_contact_diff_distance.zero_()
                self.rigid_contact_diff_normal.zero_()
                self.rigid_contact_diff_point0_world.zero_()
                self.rigid_contact_diff_point1_world.zero_()

            if self.per_contact_shape_properties:
                self.rigid_contact_stiffness.zero_()
                self.rigid_contact_damping.zero_()
                self.rigid_contact_friction.zero_()

            if self.rigid_contact_match_index is not None:
                self.rigid_contact_match_index.fill_(-1)

            self.soft_contact_indices.fill_(wp.vec3i(-1, -1, -1))
            self.soft_contact_particle.fill_(-1)
            self.soft_contact_shape.fill_(-1)
            self.soft_contact_tids.fill_(-1)
        # else: Optimized path (default) - only counter clear needed
        #   Collision detection overwrites all active contacts [0, contact_count)
        #   Solvers only read [0, contact_count), so stale data is never accessed

    @property
    def device(self):
        """
        Returns the device on which the contact buffers are allocated.
        """
        return self.rigid_contact_count.device

    def _assert_particle_only_soft_contacts(self, solver_name: str):
        """Raise if these contacts include full-surface (edge/face) soft records.

        Solvers that only consume particle soft contacts call this before reading the soft-contact
        buffer, so enabling ``enable_rigid_soft_full_surface_contact`` with an unsupported solver
        fails loudly instead of silently misreading edge/face records as particle contacts.

        Args:
            solver_name: Name of the calling solver, used in the error message.
        """
        if self._enable_rigid_soft_full_surface_contact:
            raise NotImplementedError(
                f"{solver_name} does not support full-surface soft contacts "
                "(CollisionPipeline was built with enable_rigid_soft_full_surface_contact=True); "
                "only SolverVBD consumes edge/face soft contacts. Disable the flag or use SolverVBD."
            )
