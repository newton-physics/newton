# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import warp as wp
from warp import DeviceLike as Devicelike

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


class Contacts:
    """
    Stores contact information for rigid and soft body collisions, to be consumed by a solver.

    This class manages buffers for contact data such as positions, normals, margins, and shape indices
    for both rigid-rigid and soft-rigid contacts. The buffers are allocated on the specified device and can
    optionally require gradients for differentiable simulation.

    .. note::
        This class is a temporary solution and its interface may change in the future.
    """

    EXTENDED_ATTRIBUTES: frozenset[str] = frozenset(("force",))
    """
    Names of optional extended contact attributes that are not allocated by default.

    These can be requested via :meth:`newton.ModelBuilder.request_contact_attributes` or
    :meth:`newton.Model.request_contact_attributes` before calling :meth:`newton.Model.contacts` or
    :meth:`newton.CollisionPipeline.contacts`.

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

    def __init__(
        self,
        rigid_contact_max: int,
        soft_contact_max: int,
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

        .. note::
            The ``rigid_contact_diff_*`` arrays allocated when ``requires_grad=True`` are
            **experimental**; see :meth:`newton.CollisionPipeline.collide`.
        """
        if contact_report and not contact_matching:
            raise ValueError("contact_report=True requires contact_matching=True")
        self.per_contact_shape_properties = per_contact_shape_properties
        self.clear_buffers = clear_buffers
        with wp.ScopedDevice(device):
            # Packed counter array [rigid_contact_count, soft_contact_count] so
            # all counts can be zeroed together in one fused kernel launch.
            # Every entry must be safe to reset to zero at the start of a
            # collision pass.
            self.contact_counters = wp.zeros(2, dtype=wp.int32)
            # Create sliced views for individual counters (no additional allocation)
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

                Values: ``>= 0`` matched old contact index;
                :data:`newton.geometry.MATCH_NOT_FOUND` (``-1``) new contact;
                :data:`newton.geometry.MATCH_BROKEN` (``-2``) key matched but
                position/normal thresholds exceeded.
                Shape (rigid_contact_max,), dtype int32."""
            else:
                self.rigid_contact_match_index = None

            if contact_report:
                self.rigid_contact_new_indices = wp.zeros(rigid_contact_max, dtype=wp.int32)
                """Indices of new contacts in the current sorted buffer (where ``match_index < 0``).

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

            # soft contacts — requires_grad flows through here for differentiable simulation
            self.soft_contact_count = self.contact_counters[1:2]
            self.soft_contact_particle = wp.full(soft_contact_max, -1, dtype=int)
            self.soft_contact_shape = wp.full(soft_contact_max, -1, dtype=int)
            self.soft_contact_body_pos = wp.zeros(soft_contact_max, dtype=wp.vec3, requires_grad=requires_grad)
            """Contact position on body [m], shape (soft_contact_max,), dtype :class:`vec3`."""
            self.soft_contact_body_vel = wp.zeros(soft_contact_max, dtype=wp.vec3, requires_grad=requires_grad)
            """Contact velocity on body [m/s], shape (soft_contact_max,), dtype :class:`vec3`."""
            self.soft_contact_normal = wp.zeros(soft_contact_max, dtype=wp.vec3, requires_grad=requires_grad)
            """Contact normal direction [unitless], shape (soft_contact_max,), dtype :class:`vec3`."""
            self.soft_contact_tids = wp.full(soft_contact_max, -1, dtype=int)

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
