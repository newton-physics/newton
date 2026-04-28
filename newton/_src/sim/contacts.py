# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import warnings

import warp as wp
from warp import DeviceLike as Devicelike


def _deprecated_alias(fq_old: str, new: str, extra: str = "") -> property:
    """Build a read-only ``@property`` that warns and forwards ``getattr(self, new)``.

    Args:
        fq_old: Fully-qualified old name used in the ``DeprecationWarning`` message
            (e.g. ``"Contacts.rigid_contact_max"``).
        new: Target attribute name on the instance.
        extra: Optional extra sentence appended to the warning (e.g. semantic changes).
    """
    msg = f"{fq_old} is deprecated; use {new}."
    if extra:
        msg = f"{msg} {extra}"

    def getter(self):
        warnings.warn(msg, DeprecationWarning, stacklevel=2)
        return getattr(self, new)

    getter.__doc__ = f"Deprecated; use :attr:`{new}`."
    return property(getter)


GENERATION_SENTINEL = -1
"""Value reserved as an impossible generation; the increment kernel skips it."""


@wp.kernel(enable_backward=False)
def _increment_generation(generation: wp.array[wp.int32]):
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
    """Stores contact information for rigid and soft body collisions, to be consumed by a solver.

    This class manages buffers for contact data such as positions, normals, margins, and shape indices
    for both rigid-rigid and soft-rigid contacts. The buffers are allocated on the specified device and can
    optionally require gradients for differentiable simulation.

    """

    EXTENDED_ATTRIBUTES: frozenset[str] = frozenset(("rigid_force", "rigid_velocity", "rigid_key"))
    """Names of optional extended contact attributes that are not allocated by default.

    These can be requested via :meth:`newton.ModelBuilder.request_contact_attributes` or
    :meth:`newton.Model.request_contact_attributes` before calling :meth:`newton.Model.contacts` or
    :meth:`newton.CollisionPipeline.contacts`.

    See :ref:`extended_contact_attributes` for details and usage.
    """

    _LEGACY_EXTENDED_ATTRIBUTES: dict[str, str] = {
        "force": "rigid_force",
        "velocity": "rigid_velocity",
        "key": "rigid_key",
    }
    """Deprecated request-name aliases. Callers passing ``"force"``/``"velocity"``/``"key"`` to
    ``request_contact_attributes`` are translated to the new ``rigid_*`` names with a
    ``DeprecationWarning``."""

    # Configuration
    rigid_count_max: int
    """Maximum permissible number of rigid contacts."""

    soft_count_max: int
    """Maximum permissible number of soft contacts."""

    per_contact_shape_properties: bool
    """Whether per-contact stiffness/damping/friction arrays are allocated."""

    clear_buffers: bool
    """Whether :meth:`clear` performs a full buffer wipe (slower, debugging aid)."""

    requires_grad: bool
    """Whether differentiable contact arrays are allocated."""

    # Device-side counters
    contact_counters: wp.array[wp.int32]
    """Packed contact counts ``[rigid_count_active, soft_count_active]``, shape (2,).

    The counts share one allocation so producers can reset them together. Use the sliced views
    :attr:`rigid_count_active` and :attr:`soft_count_active` for access.
    """

    rigid_count_active: wp.array[wp.int32]
    """Single-element device view into :attr:`contact_counters` holding the active rigid contact count, shape (1,)."""

    soft_count_active: wp.array[wp.int32]
    """Single-element device view into :attr:`contact_counters` holding the active soft contact count, shape (1,)."""

    # Rigid contacts (core)
    rigid_distance: wp.array[wp.float32]
    """Signed contact distance, raw surface distance minus per-shape margins [m], shape (rigid_count_max,).

    Negative when penetrating.
    """

    rigid_normal: wp.array[wp.vec3]
    """Contact normal pointing from shape 0 toward shape 1 (A-to-B) [unitless], shape (rigid_count_max,)."""

    rigid_point_0: wp.array[wp.vec3]
    """Contact point in body-0 frame [m], shape (rigid_count_max,)."""

    rigid_point_1: wp.array[wp.vec3]
    """Contact point in body-1 frame [m], shape (rigid_count_max,)."""

    rigid_shapes: wp.array[wp.vec2i]
    """Shape-pair global indices ``(shape_0, shape_1)``, shape (rigid_count_max,).

    Stores both shape indices per contact as a single packed pair; unused slots are filled with ``(-1, -1)``.
    """

    rigid_margins: wp.array[wp.vec2f]
    """Surface thickness ``(margin_0, margin_1)`` [m], shape (rigid_count_max,).

    Per-shape effective radius plus margin, stored as a packed pair.
    """

    # storage for deprecated offset arrays
    _deprecated_rigid_offset_0: wp.array[wp.vec3]
    _deprecated_rigid_offset_1: wp.array[wp.vec3]

    rigid_tids: wp.array[wp.int32]
    """Narrow-phase thread index that produced each contact [dimensionless], shape (rigid_count_max,).

    Used for gradient routing through the differentiable collision path.
    """

    # Rigid contacts (gated on ``per_contact_shape_properties``)
    rigid_stiffness: wp.array[wp.float32] | None
    """Per-contact stiffness [N/m], shape (rigid_count_max,).

    ``None`` unless ``per_contact_shape_properties=True`` was passed to the constructor.
    """

    rigid_damping: wp.array[wp.float32] | None
    """Per-contact damping [N·s/m], shape (rigid_count_max,).

    ``None`` unless ``per_contact_shape_properties=True`` was passed to the constructor.
    """

    rigid_friction: wp.array[wp.float32] | None
    """Per-contact friction coefficient [dimensionless], shape (rigid_count_max,).

    ``None`` unless ``per_contact_shape_properties=True`` was passed to the constructor.
    """

    # Rigid contacts (gated on ``requires_grad``)
    rigid_diff_distance: wp.array[wp.float32] | None
    """Differentiable signed distance [m], shape (rigid_count_max,).

    ``None`` unless ``requires_grad=True``. Populated by :meth:`newton.CollisionPipeline.collide`
    when ``requires_grad=True``. **Experimental.**
    """

    rigid_diff_normal: wp.array[wp.vec3] | None
    """Contact normal (A-to-B, world frame) [unitless], shape (rigid_count_max,).

    ``None`` unless ``requires_grad=True``. **Experimental.**
    """

    rigid_diff_point_0_world: wp.array[wp.vec3] | None
    """World-space contact point on shape 0 [m], shape (rigid_count_max,).

    ``None`` unless ``requires_grad=True``. **Experimental.**
    """

    rigid_diff_point_1_world: wp.array[wp.vec3] | None
    """World-space contact point on shape 1 [m], shape (rigid_count_max,).

    ``None`` unless ``requires_grad=True``. **Experimental.**
    """

    # Soft contacts
    soft_particle: wp.array[wp.int32]
    """Particle index per contact [dimensionless], shape (soft_count_max,)."""

    soft_shape: wp.array[wp.int32]
    """Shape index per contact [dimensionless], shape (soft_count_max,)."""

    soft_body_pos: wp.array[wp.vec3]
    """Contact position on body [m], shape (soft_count_max,)."""

    soft_body_vel: wp.array[wp.vec3]
    """Contact velocity on body [m/s], shape (soft_count_max,)."""

    soft_normal: wp.array[wp.vec3]
    """Contact normal direction [unitless], shape (soft_count_max,)."""

    soft_tids: wp.array[wp.int32]
    """Narrow-phase thread index that produced each contact [dimensionless], shape (soft_count_max,).

    Used for gradient routing through the differentiable collision path.
    """

    # Extended rigid attributes (request-gated)
    rigid_force: wp.array[wp.spatial_vector] | None
    """Rigid contact spatial force [N, N·m], shape (rigid_count_max,).
    First three entries: linear force [N]; last three: torque [N·m]. Solvers may populate the wrench
    partially (e.g., linear only). ``None`` unless requested.

    This is an extended contact attribute; see :ref:`extended_contact_attributes` for more information.

    .. note::
        Wrench exerted **by body 0 on body 1** in world frame; the linear part is broadly collinear
        with :attr:`rigid_normal`.
    """

    rigid_velocity: wp.array[wp.spatial_vector] | None
    """Rigid contact spatial velocity (solver output) [m/s, rad/s], shape (rigid_count_max,).

    ``None`` unless requested.

    This is an extended contact attribute; see :ref:`extended_contact_attributes` for more information.
    """

    rigid_key: wp.array[wp.uint64] | None
    """Cross-step rigid contact identifier for warmstart/matching [dimensionless], shape (rigid_count_max,).

    Solver-populated.  ``None`` unless requested.

    This is an extended contact attribute; see :ref:`extended_contact_attributes` for more information.
    """

    @classmethod
    def validate_extended_attributes(cls, attributes: tuple[str, ...]) -> None:
        """Validate names passed to request_contact_attributes().

        Only extended contact attributes listed in :attr:`EXTENDED_ATTRIBUTES` are accepted.
        Legacy aliases in :attr:`_LEGACY_EXTENDED_ATTRIBUTES` are accepted for backward
        compatibility (translated to their new names in :meth:`__init__` with a deprecation
        warning).

        Args:
            attributes: Tuple of attribute names to validate.

        Raises:
            ValueError: If any attribute name is not recognised.
        """
        if not attributes:
            return

        accepted = cls.EXTENDED_ATTRIBUTES | cls._LEGACY_EXTENDED_ATTRIBUTES.keys()
        invalid = sorted(set(attributes).difference(accepted))
        if invalid:
            allowed = ", ".join(sorted(cls.EXTENDED_ATTRIBUTES))
            bad = ", ".join(invalid)
            raise ValueError(f"Unknown extended contact attribute(s): {bad}. Allowed: {allowed}.")

    def __init__(
        self,
        rigid_count_max: int | None = None,
        soft_count_max: int | None = None,
        requires_grad: bool = False,
        device: Devicelike = None,
        per_contact_shape_properties: bool = False,
        clear_buffers: bool = False,
        requested_attributes: set[str] | None = None,
        contact_matching: bool = False,
        contact_report: bool = False,
        rigid_contact_max: int | None = None,
        soft_contact_max: int | None = None,
    ):
        """Initialize Contacts storage.

        Args:
            rigid_count_max: Maximum number of rigid contacts. Required.
            soft_count_max: Maximum number of soft contacts. Required.
            requires_grad: Whether contact arrays require gradients for differentiable simulation.
                When ``True``, soft contact arrays (body_pos, body_vel, normal) are allocated with
                gradients so that gradient-based optimization can flow through particle-shape contacts,
                **and** additional differentiable rigid contact arrays are allocated
                (``rigid_diff_*``) that provide first-order gradients of contact distance and
                world-space points with respect to body poses.
            device: Device to allocate buffers on.
            per_contact_shape_properties: Enable per-contact stiffness/damping/friction arrays.
            clear_buffers: If ``True``, :meth:`clear` will zero all contact buffers (slower but
                conservative). If ``False`` (default), :meth:`clear` only resets counts in a single
                fused kernel launch, relying on collision detection to overwrite active contacts.
                Safe since solvers only read up to ``rigid_count_active``.
            requested_attributes: Set of extended contact attribute names to allocate. See
                :attr:`EXTENDED_ATTRIBUTES` for available options.
            contact_matching: Allocate a per-contact match index array
                (:attr:`rigid_match_index`) that stores frame-to-frame contact
                correspondences filled by the collision pipeline.
            contact_report: Allocate compact index lists of new and broken contacts
                (:attr:`rigid_new_indices`, :attr:`rigid_new_count`,
                :attr:`rigid_broken_indices`, :attr:`rigid_broken_count`)
                populated each frame by the collision pipeline. Requires ``contact_matching=True``.
            rigid_contact_max: Deprecated alias for ``rigid_count_max``.
            soft_contact_max: Deprecated alias for ``soft_count_max``.

        .. note::
            The ``rigid_diff_*`` arrays allocated when ``requires_grad=True`` are **experimental**;
            see :meth:`newton.CollisionPipeline.collide`.
        """
        if contact_report and not contact_matching:
            raise ValueError("contact_report=True requires contact_matching=True")
        if rigid_contact_max is not None:
            warnings.warn(
                "Contacts(rigid_contact_max=...) is deprecated; use rigid_count_max.",
                DeprecationWarning,
                stacklevel=2,
            )
            if rigid_count_max is not None:
                raise TypeError("Pass either rigid_count_max or the deprecated rigid_contact_max, not both.")
            rigid_count_max = rigid_contact_max
        if soft_contact_max is not None:
            warnings.warn(
                "Contacts(soft_contact_max=...) is deprecated; use soft_count_max.",
                DeprecationWarning,
                stacklevel=2,
            )
            if soft_count_max is not None:
                raise TypeError("Pass either soft_count_max or the deprecated soft_contact_max, not both.")
            soft_count_max = soft_contact_max
        if rigid_count_max is None or soft_count_max is None:
            raise TypeError("Contacts requires rigid_count_max and soft_count_max.")

        self.rigid_count_max = rigid_count_max
        self.soft_count_max = soft_count_max
        self.per_contact_shape_properties = per_contact_shape_properties
        self.clear_buffers = clear_buffers
        self.contact_matching = contact_matching
        self.contact_report = contact_report
        self.requires_grad = requires_grad

        with wp.ScopedDevice(device):
            self.contact_counters = wp.zeros(2, dtype=wp.int32)
            self.rigid_count_active = self.contact_counters[0:1]
            self.soft_count_active = self.contact_counters[1:2]
            self.generation = wp.zeros(1, dtype=wp.int32)
            """Device-side generation counter, incremented each time :meth:`clear` is called.

            Solvers can compare this against a cached value to detect whether the
            contact set changed since the last conversion pass."""

            # Rigid core
            self.rigid_distance = wp.zeros(rigid_count_max, dtype=wp.float32)
            self.rigid_normal = wp.zeros(rigid_count_max, dtype=wp.vec3)
            self.rigid_point_0 = wp.zeros(rigid_count_max, dtype=wp.vec3)
            self.rigid_point_1 = wp.zeros(rigid_count_max, dtype=wp.vec3)
            self.rigid_shapes = wp.full(rigid_count_max, wp.vec2i(-1, -1), dtype=wp.vec2i)
            self.rigid_margins = wp.zeros(rigid_count_max, dtype=wp.vec2f)
            self._deprecated_rigid_offset_0 = wp.zeros(rigid_count_max, dtype=wp.vec3)
            self._deprecated_rigid_offset_1 = wp.zeros(rigid_count_max, dtype=wp.vec3)
            self.rigid_tids = wp.full(rigid_count_max, -1, dtype=wp.int32)

            # Rigid differentiable arrays (gated on requires_grad)
            if requires_grad:
                self.rigid_diff_distance = wp.zeros(rigid_count_max, dtype=wp.float32, requires_grad=True)
                self.rigid_diff_normal = wp.zeros(rigid_count_max, dtype=wp.vec3, requires_grad=False)
                self.rigid_diff_point_0_world = wp.zeros(rigid_count_max, dtype=wp.vec3, requires_grad=True)
                self.rigid_diff_point_1_world = wp.zeros(rigid_count_max, dtype=wp.vec3, requires_grad=True)
            else:
                self.rigid_diff_distance = None
                self.rigid_diff_normal = None
                self.rigid_diff_point_0_world = None
                self.rigid_diff_point_1_world = None

            # Rigid per-shape-properties (gated)
            if per_contact_shape_properties:
                self.rigid_stiffness = wp.zeros(rigid_count_max, dtype=wp.float32)
                self.rigid_damping = wp.zeros(rigid_count_max, dtype=wp.float32)
                self.rigid_friction = wp.zeros(rigid_count_max, dtype=wp.float32)
            else:
                self.rigid_stiffness = None
                self.rigid_damping = None
                self.rigid_friction = None

            # Contact matching / reporting (gated)
            # Filled by the collision pipeline when contact_matching is enabled.
            if contact_matching:
                self.rigid_match_index = wp.full(rigid_count_max, -1, dtype=wp.int32)
            else:
                self.rigid_match_index = None

            if contact_report:
                self.rigid_new_indices = wp.zeros(rigid_count_max, dtype=wp.int32)
                self.rigid_new_count = wp.zeros(1, dtype=wp.int32)
                self.rigid_broken_indices = wp.zeros(rigid_count_max, dtype=wp.int32)
                self.rigid_broken_count = wp.zeros(1, dtype=wp.int32)
            else:
                self.rigid_new_indices = None
                self.rigid_new_count = None
                self.rigid_broken_indices = None
                self.rigid_broken_count = None

            # Soft contacts
            # requires_grad flows through here for differentiable simulation.
            self.soft_particle = wp.full(soft_count_max, -1, dtype=wp.int32)
            self.soft_shape = wp.full(soft_count_max, -1, dtype=wp.int32)
            self.soft_body_pos = wp.zeros(soft_count_max, dtype=wp.vec3, requires_grad=requires_grad)
            self.soft_body_vel = wp.zeros(soft_count_max, dtype=wp.vec3, requires_grad=requires_grad)
            self.soft_normal = wp.zeros(soft_count_max, dtype=wp.vec3, requires_grad=requires_grad)
            self.soft_tids = wp.full(soft_count_max, -1, dtype=wp.int32)

            # Extended rigid attributes (request-gated)
            self.rigid_force = None
            self.rigid_velocity = None
            self.rigid_key = None
            if requested_attributes:
                self.validate_extended_attributes(tuple(requested_attributes))
                normalized: set[str] = set()
                for attr in requested_attributes:
                    new_name = self._LEGACY_EXTENDED_ATTRIBUTES.get(attr)
                    if new_name is not None:
                        warnings.warn(
                            f"Contacts extended attribute '{attr}' is deprecated; use '{new_name}'.",
                            DeprecationWarning,
                            stacklevel=2,
                        )
                        normalized.add(new_name)
                    else:
                        normalized.add(attr)

                if "rigid_force" in normalized:
                    self.rigid_force = wp.zeros(rigid_count_max, dtype=wp.spatial_vector, requires_grad=requires_grad)
                if "rigid_velocity" in normalized:
                    self.rigid_velocity = wp.zeros(
                        rigid_count_max, dtype=wp.spatial_vector, requires_grad=requires_grad
                    )
                if "rigid_key" in normalized:
                    self.rigid_key = wp.zeros(rigid_count_max, dtype=wp.uint64)

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
            bump_generation: If True (default), increment ``generation`` to invalidate
                previously-observed contact data. Callers that will immediately re-bump the
                generation via another fused kernel (e.g. :func:`compute_shape_aabbs`) can pass
                ``False`` to avoid an unnecessary double-bump per collision pass.
        """
        # Clear all counters and (optionally) bump generation in a single kernel launch.
        num_counters = self.contact_counters.shape[0]
        wp.launch(
            _clear_counters_and_bump_generation,
            dim=max(num_counters, 1),
            inputs=[self.contact_counters, self.generation, num_counters, int(bump_generation)],
            device=self.generation.device,
            record_tape=False,
        )

        if self.clear_buffers:
            # Conservative path: clear all buffers with sentinel values and zeros.
            # Slower than the fast path but may be useful for debugging or special cases.
            self.rigid_shapes.fill_(wp.vec2i(-1, -1))
            self.rigid_tids.fill_(-1)
            self.rigid_distance.zero_()
            self.rigid_normal.zero_()
            self.rigid_point_0.zero_()
            self.rigid_point_1.zero_()
            self.rigid_margins.zero_()
            self._deprecated_rigid_offset_0.zero_()
            self._deprecated_rigid_offset_1.zero_()

            if self.rigid_force is not None:
                self.rigid_force.zero_()
            if self.rigid_velocity is not None:
                self.rigid_velocity.zero_()
            if self.rigid_key is not None:
                self.rigid_key.zero_()

            if self.rigid_diff_distance is not None:
                self.rigid_diff_distance.zero_()
                self.rigid_diff_normal.zero_()
                self.rigid_diff_point_0_world.zero_()
                self.rigid_diff_point_1_world.zero_()

            if self.per_contact_shape_properties:
                self.rigid_stiffness.zero_()
                self.rigid_damping.zero_()
                self.rigid_friction.zero_()

            if self.rigid_match_index is not None:
                self.rigid_match_index.fill_(-1)

            self.soft_particle.fill_(-1)
            self.soft_shape.fill_(-1)
            self.soft_tids.fill_(-1)
            self.soft_body_pos.zero_()
            self.soft_body_vel.zero_()
            self.soft_normal.zero_()
        # else: Optimized path (default) - only counter clear needed
        #   Collision detection overwrites all active contacts [0, contact_count)
        #   Solvers only read [0, contact_count), so stale data is never accessed

    @property
    def device(self):
        """Returns the device on which the contact buffers are allocated."""
        return self.rigid_count_active.device

    rigid_contact_max = _deprecated_alias("Contacts.rigid_contact_max", "rigid_count_max")
    soft_contact_max = _deprecated_alias("Contacts.soft_contact_max", "soft_count_max")
    rigid_contact_count = _deprecated_alias("Contacts.rigid_contact_count", "rigid_count_active")
    soft_contact_count = _deprecated_alias("Contacts.soft_contact_count", "soft_count_active")
    rigid_contact_normal = _deprecated_alias("Contacts.rigid_contact_normal", "rigid_normal")
    rigid_contact_point0 = _deprecated_alias("Contacts.rigid_contact_point0", "rigid_point_0")
    rigid_contact_point1 = _deprecated_alias("Contacts.rigid_contact_point1", "rigid_point_1")
    rigid_contact_tids = _deprecated_alias("Contacts.rigid_contact_tids", "rigid_tids")

    @property
    def rigid_contact_offset0(self) -> wp.array:
        """Deprecated."""
        warnings.warn("Contacts.rigid_contact_offset0 is deprecated.", DeprecationWarning, stacklevel=2)
        return self._deprecated_rigid_offset_0

    @property
    def rigid_contact_offset1(self) -> wp.array:
        """Deprecated."""
        warnings.warn("Contacts.rigid_contact_offset1 is deprecated.", DeprecationWarning, stacklevel=2)
        return self._deprecated_rigid_offset_1

    rigid_contact_stiffness = _deprecated_alias("Contacts.rigid_contact_stiffness", "rigid_stiffness")
    rigid_contact_damping = _deprecated_alias("Contacts.rigid_contact_damping", "rigid_damping")
    rigid_contact_friction = _deprecated_alias("Contacts.rigid_contact_friction", "rigid_friction")
    rigid_contact_diff_distance = _deprecated_alias("Contacts.rigid_contact_diff_distance", "rigid_diff_distance")
    rigid_contact_diff_normal = _deprecated_alias("Contacts.rigid_contact_diff_normal", "rigid_diff_normal")
    rigid_contact_diff_point0_world = _deprecated_alias(
        "Contacts.rigid_contact_diff_point0_world", "rigid_diff_point_0_world"
    )
    rigid_contact_diff_point1_world = _deprecated_alias(
        "Contacts.rigid_contact_diff_point1_world", "rigid_diff_point_1_world"
    )
    soft_contact_particle = _deprecated_alias("Contacts.soft_contact_particle", "soft_particle")
    soft_contact_shape = _deprecated_alias("Contacts.soft_contact_shape", "soft_shape")
    soft_contact_body_pos = _deprecated_alias("Contacts.soft_contact_body_pos", "soft_body_pos")
    soft_contact_body_vel = _deprecated_alias("Contacts.soft_contact_body_vel", "soft_body_vel")
    soft_contact_normal = _deprecated_alias("Contacts.soft_contact_normal", "soft_normal")
    soft_contact_tids = _deprecated_alias("Contacts.soft_contact_tids", "soft_tids")
    force = _deprecated_alias(
        "Contacts.force",
        "rigid_force",
        extra="Sign convention also changed: rigid_force is the force BY body 0 ON body 1 (negate the legacy values).",
    )
    velocity = _deprecated_alias("Contacts.velocity", "rigid_velocity")
    key = _deprecated_alias("Contacts.key", "rigid_key")

    # Variants with extra logic

    @property
    def rigid_contact_shape0(self) -> wp.array:
        """Deprecated; use :attr:`rigid_shapes` (packed ``vec2i``)."""
        warnings.warn(
            "Contacts.rigid_contact_shape0 is deprecated; use rigid_shapes (packed vec2i).",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.rigid_shapes.view(wp.int32)[:, 0]

    @property
    def rigid_contact_shape1(self) -> wp.array:
        """Deprecated; use :attr:`rigid_shapes` (packed ``vec2i``)."""
        warnings.warn(
            "Contacts.rigid_contact_shape1 is deprecated; use rigid_shapes (packed vec2i).",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.rigid_shapes.view(wp.int32)[:, 1]

    @property
    def rigid_contact_margin0(self) -> wp.array:
        """Deprecated; use :attr:`rigid_margins` (packed ``vec2f``)."""
        warnings.warn(
            "Contacts.rigid_contact_margin0 is deprecated; use rigid_margins (packed vec2f).",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.rigid_margins.view(wp.float32)[:, 0]

    @property
    def rigid_contact_margin1(self) -> wp.array:
        """Deprecated; use :attr:`rigid_margins` (packed ``vec2f``)."""
        warnings.warn(
            "Contacts.rigid_contact_margin1 is deprecated; use rigid_margins (packed vec2f).",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.rigid_margins.view(wp.float32)[:, 1]

    @property
    def rigid_contact_point_id(self) -> None:
        """Deprecated; attribute removed (had no consumers)."""
        warnings.warn(
            "Contacts.rigid_contact_point_id is deprecated and no longer allocated.",
            DeprecationWarning,
            stacklevel=2,
        )
        return None

    @property
    def rigid_contact_force(self) -> wp.array | None:
        """Deprecated; use extended attribute :attr:`rigid_force` (``spatial_vector``).

        Returns a strided ``vec3`` view over the linear part of :attr:`rigid_force`, or ``None`` if
        ``rigid_force`` was not requested. Solvers that write through this shim must ensure their
        model has requested ``"rigid_force"`` first.
        """
        warnings.warn(
            "Contacts.rigid_contact_force is deprecated; use the extended attribute 'rigid_force' "
            "(spatial_vector, request via request_contact_attributes).",
            DeprecationWarning,
            stacklevel=2,
        )
        if self.rigid_force is None:
            return None
        return self.rigid_force.view(wp.float32)[:, :3].view(wp.vec3)
