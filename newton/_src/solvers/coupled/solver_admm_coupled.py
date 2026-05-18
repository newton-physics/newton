# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""ADMM-style coupled multi-solver simulations."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import warp as wp

from ...geometry.flags import ShapeFlags
from ...sim import JointType
from ..flags import SolverNotifyFlags
from .admm_contact_stream import (
    AdmmContactStream,
    AdmmContactType,
    admm_contact_stream_reset_count_kernel,
    admm_contact_stream_update_normal_force_kernel,
)
from .admm_utils import (
    attach_rp_accumulate_forces_kernel,
    attach_rp_compute_Jv_kernel,
    attach_rp_compute_u_target_kernel,
    attach_rr_angular_accumulate_forces_kernel,
    attach_rr_angular_compute_Jv_kernel,
    attach_rr_angular_compute_u_target_kernel,
    attach_rr_angular_local_accumulate_forces_kernel,
    attach_rr_angular_local_compute_Jv_kernel,
    attach_rr_compute_u_target_kernel,
    attach_rr_revolute_angular_local_accumulate_forces_kernel,
    attach_rr_revolute_angular_local_compute_Jv_kernel,
    attach_rr_revolute_angular_local_compute_u_target_kernel,
    contact_lambda_update_kernel,
    contact_pp_accumulate_forces_kernel,
    contact_pp_compute_Jv_kernel,
    contact_pp_compute_u_min_kernel,
    contact_pp_fill_from_particle_contacts_kernel,
    contact_pp_reset_kernel,
    contact_pp_snapshot_kernel,
    contact_rp_accumulate_forces_kernel,
    contact_rp_compute_Jv_kernel,
    contact_rp_compute_u_min_kernel,
    contact_rp_fill_from_soft_contacts_kernel,
    contact_rp_reset_kernel,
    contact_rp_snapshot_kernel,
    contact_rr_accumulate_forces_kernel,
    contact_rr_compute_Jv_kernel,
    contact_rr_compute_u_min_kernel,
    contact_rr_fill_from_rigid_contacts_kernel,
    contact_rr_reset_kernel,
    contact_rr_snapshot_kernel,
    contact_u_update_kernel,
    joint_box_friction_u_update_kernel,
    lambda_update_kernel,
    particle_particle_contacts_hashgrid_kernel,
    u_update_quadratic_kernel,
    velocity_proximal_shift_body_kernel,
    velocity_proximal_shift_joint_kernel,
    velocity_proximal_shift_particle_kernel,
)
from .interface import (
    CouplingEndpointKind,
    CouplingHook,
    CouplingInputStateFlags,
)
from .model_view import ModelView
from .solver_coupled import SolverCoupled, SolverEntry, _copy_prefix

if TYPE_CHECKING:
    from ...sim import Contacts, Control, Model, ModelBuilder, State


_DEFAULT_DETECTION_MARGIN = 0.01
"""Default ADMM contact detection margin [m] when a ContactPair leaves it unset."""


@dataclass
class _AdmmBuffers:
    """Per-entry per-step working buffers used by ADMM iterations."""

    body_q_n: wp.array = field(default=None)
    body_qd_n: wp.array = field(default=None)
    body_qd_k: wp.array = field(default=None)
    particle_q_n: wp.array = field(default=None)
    particle_qd_n: wp.array = field(default=None)
    particle_qd_k: wp.array = field(default=None)
    joint_q_n: wp.array = field(default=None)
    joint_qd_n: wp.array = field(default=None)
    joint_qd_k: wp.array = field(default=None)
    body_f: wp.array = field(default=None)
    particle_f: wp.array = field(default=None)
    body_effective_mass: wp.array = field(default=None)
    particle_effective_mass: wp.array = field(default=None)


@dataclass
class _AdmmRigidRigidAttachmentGroup:
    """Rigid-body to rigid-body ADMM point attachment group for one owner pair."""

    body_entry_name_a: str
    body_entry_name_b: str
    body_ids_a: wp.array
    point_a: wp.array
    body_ids_b: wp.array
    point_b: wp.array
    kappa: wp.array
    damping: wp.array
    W: wp.array
    u: wp.array
    lambda_: wp.array
    Jv: wp.array
    u_target: wp.array

    @property
    def count(self) -> int:
        return self.body_ids_a.shape[0]


@dataclass
class _AdmmRigidRigidAngularAttachmentGroup:
    """Rigid-body to rigid-body ADMM angular attachment group for one owner pair."""

    body_entry_name_a: str
    body_entry_name_b: str
    body_ids_a: wp.array
    frame_a: wp.array
    body_ids_b: wp.array
    frame_b: wp.array
    kappa: wp.array
    damping: wp.array
    W: wp.array
    u: wp.array
    lambda_: wp.array
    Jv: wp.array
    u_target: wp.array

    @property
    def count(self) -> int:
        return self.body_ids_a.shape[0]


@dataclass
class _AdmmRigidRigidAngularFrictionGroup:
    """Rigid-body to rigid-body ADMM angular box-friction group for one owner pair."""

    body_entry_name_a: str
    body_entry_name_b: str
    body_ids_a: wp.array
    frame_a: wp.array
    body_ids_b: wp.array
    friction: wp.array
    W: wp.array
    u: wp.array
    lambda_: wp.array
    Jv: wp.array

    @property
    def count(self) -> int:
        return self.body_ids_a.shape[0]


@dataclass
class _AdmmRigidParticleAttachmentGroup:
    """Rigid-body to particle ADMM attachment group for one owner pair."""

    body_entry_name: str
    particle_entry_name: str
    body_ids: wp.array
    point_body: wp.array
    particle_ids: wp.array
    kappa: wp.array
    damping: wp.array
    W: wp.array
    u: wp.array
    lambda_: wp.array
    Jv: wp.array
    u_target: wp.array

    @property
    def count(self) -> int:
        return self.body_ids.shape[0]


@dataclass
class _AdmmRigidRigidContactGroup:
    """Rigid-body to rigid-body ADMM contact group for one owner pair."""

    body_entry_name_a: str
    body_entry_name_b: str
    body_ids_a: wp.array
    point_a: wp.array
    body_ids_b: wp.array
    point_b: wp.array
    normal: wp.array
    contact_distance: wp.array
    W: wp.array
    friction: wp.array
    u: wp.array
    lambda_: wp.array
    Jv: wp.array
    u_min: wp.array
    capacity: int | None = None
    active_count: wp.array | None = None
    active_count_max: wp.array | None = None
    active: wp.array | None = None
    shape_ids_a: wp.array | None = None
    shape_ids_b: wp.array | None = None
    point_ids: wp.array | None = None
    body_mask_a: wp.array | None = None
    body_mask_b: wp.array | None = None
    shape_mask_a: wp.array | None = None
    shape_mask_b: wp.array | None = None
    contact_distance_value: float = 0.0
    use_contact_margins: bool = False
    prev_body_ids_a: wp.array | None = None
    prev_body_ids_b: wp.array | None = None
    prev_shape_ids_a: wp.array | None = None
    prev_shape_ids_b: wp.array | None = None
    prev_point_ids: wp.array | None = None
    prev_active: wp.array | None = None
    prev_u: wp.array | None = None
    prev_lambda: wp.array | None = None

    @property
    def count(self) -> int:
        return self.capacity if self.capacity is not None else self.body_ids_a.shape[0]


@dataclass
class _AdmmRigidParticleContactGroup:
    """Rigid-body to particle ADMM contact group for one owner pair."""

    body_entry_name: str
    particle_entry_name: str
    body_ids: wp.array
    point_body: wp.array
    particle_ids: wp.array
    normal: wp.array
    body_sign: wp.array
    contact_distance: wp.array
    W: wp.array
    friction: wp.array
    u: wp.array
    lambda_: wp.array
    Jv: wp.array
    u_min: wp.array
    capacity: int | None = None
    active_count: wp.array | None = None
    active_count_max: wp.array | None = None
    active: wp.array | None = None
    shape_ids: wp.array | None = None
    particle_mask: wp.array | None = None
    body_mask: wp.array | None = None
    shape_mask: wp.array | None = None
    contact_distance_value: float = 0.0
    use_particle_radius: bool = False
    prev_body_ids: wp.array | None = None
    prev_particle_ids: wp.array | None = None
    prev_shape_ids: wp.array | None = None
    prev_active: wp.array | None = None
    prev_u: wp.array | None = None
    prev_lambda: wp.array | None = None

    @property
    def count(self) -> int:
        return self.capacity if self.capacity is not None else self.body_ids.shape[0]


@dataclass
class _AdmmParticleParticleContactGroup:
    """Particle-to-particle ADMM contact group for one owner pair."""

    particle_entry_name_a: str
    particle_entry_name_b: str
    particle_ids_a: wp.array
    particle_ids_b: wp.array
    normal: wp.array
    contact_distance: wp.array
    W: wp.array
    friction: wp.array
    u: wp.array
    lambda_: wp.array
    Jv: wp.array
    u_min: wp.array
    capacity: int | None = None
    active_count: wp.array | None = None
    active_count_max: wp.array | None = None
    active: wp.array | None = None
    contact_stream: AdmmContactStream | None = None
    particle_mask_a: wp.array | None = None
    particle_mask_b: wp.array | None = None
    contact_distance_value: float = 0.0
    use_radius_sum: bool = False
    detection_margin: float = 0.0
    query_radius: float = 0.0
    prev_particle_ids_a: wp.array | None = None
    prev_particle_ids_b: wp.array | None = None
    prev_active: wp.array | None = None
    prev_u: wp.array | None = None
    prev_lambda: wp.array | None = None

    @property
    def count(self) -> int:
        return self.capacity if self.capacity is not None else self.particle_ids_a.shape[0]


@dataclass(frozen=True)
class _AdmmRigidParticleContactSpec:
    """Internal particle-shape contact source derived from model ownership."""

    particle_owner: str
    body_owner: str
    shapes: tuple[int, ...] | None = None
    contact_distance: float | None = None
    detection_margin: float | None = None


@dataclass(frozen=True)
class _AdmmRigidRigidContactSpec:
    """Internal rigid-rigid contact source derived from model ownership."""

    owner_a: str
    owner_b: str
    shapes_a: tuple[int, ...] | None = None
    shapes_b: tuple[int, ...] | None = None
    contact_distance: float | None = None


@dataclass(frozen=True)
class _AdmmParticleParticleContactSpec:
    """Internal particle-particle contact source derived from model ownership."""

    owner_a: str
    owner_b: str
    particles_a: tuple[int, ...] | None = None
    particles_b: tuple[int, ...] | None = None
    contact_distance: float | None = None
    detection_margin: float | None = None


class SolverAdmmCoupled(SolverCoupled):
    """Couple multiple solvers with linearized ADMM over model-derived constraints."""

    BODY_PARTICLE_ATTACHMENT_FREQUENCY = "coupling:body_particle_attachment"
    BODY_PARTICLE_ATTACHMENT_BODY_ATTR = "coupling:body_particle_attachment_body"
    BODY_PARTICLE_ATTACHMENT_PARTICLE_ATTR = "coupling:body_particle_attachment_particle"
    BODY_PARTICLE_ATTACHMENT_BODY_POINT_ATTR = "coupling:body_particle_attachment_body_point"
    BODY_PARTICLE_ATTACHMENT_STIFFNESS_ATTR = "coupling:body_particle_attachment_stiffness"
    BODY_PARTICLE_ATTACHMENT_DAMPING_ATTR = "coupling:body_particle_attachment_damping"
    BODY_PARTICLE_ATTACHMENT_ENABLED_ATTR = "coupling:body_particle_attachment_enabled"

    @classmethod
    def register_custom_attributes(cls, builder: ModelBuilder) -> None:
        """Register ADMM coupling custom attributes on a model builder.

        The registered ``coupling:body_particle_attachment`` custom frequency
        stores model-level rigid-body-to-particle attachment annotations. During
        construction, :class:`SolverAdmmCoupled` converts rows whose body and
        particle endpoints are owned by different solver entries into ADMM
        attachment constraints.

        Args:
            builder: Model builder receiving the custom frequency and attributes.
        """
        from ...sim import Model, ModelBuilder  # noqa: PLC0415

        builder.add_custom_frequency(
            ModelBuilder.CustomFrequency(name="body_particle_attachment", namespace="coupling")
        )
        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="body_particle_attachment_body",
                frequency=cls.BODY_PARTICLE_ATTACHMENT_FREQUENCY,
                assignment=Model.AttributeAssignment.MODEL,
                dtype=wp.int32,
                default=-1,
                namespace="coupling",
                references="body",
            )
        )
        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="body_particle_attachment_particle",
                frequency=cls.BODY_PARTICLE_ATTACHMENT_FREQUENCY,
                assignment=Model.AttributeAssignment.MODEL,
                dtype=wp.int32,
                default=-1,
                namespace="coupling",
                references="particle",
            )
        )
        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="body_particle_attachment_body_point",
                frequency=cls.BODY_PARTICLE_ATTACHMENT_FREQUENCY,
                assignment=Model.AttributeAssignment.MODEL,
                dtype=wp.vec3,
                default=wp.vec3(0.0, 0.0, 0.0),
                namespace="coupling",
            )
        )
        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="body_particle_attachment_stiffness",
                frequency=cls.BODY_PARTICLE_ATTACHMENT_FREQUENCY,
                assignment=Model.AttributeAssignment.MODEL,
                dtype=wp.float32,
                default=1.0e4,
                namespace="coupling",
            )
        )
        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="body_particle_attachment_damping",
                frequency=cls.BODY_PARTICLE_ATTACHMENT_FREQUENCY,
                assignment=Model.AttributeAssignment.MODEL,
                dtype=wp.float32,
                default=0.0,
                namespace="coupling",
            )
        )
        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="body_particle_attachment_enabled",
                frequency=cls.BODY_PARTICLE_ATTACHMENT_FREQUENCY,
                assignment=Model.AttributeAssignment.MODEL,
                dtype=wp.bool,
                default=True,
                namespace="coupling",
            )
        )

    @classmethod
    def add_body_particle_attachment(
        cls,
        builder: ModelBuilder,
        body: int,
        particle: int,
        *,
        body_point: tuple[float, float, float] | wp.vec3 = (0.0, 0.0, 0.0),
        stiffness: float = 1.0e4,
        damping: float = 0.0,
        enabled: bool = True,
    ) -> int:
        """Add a model-level rigid-body-to-particle ADMM attachment.

        Args:
            builder: Model builder that owns the body and particle.
            body: Body index for the rigid endpoint.
            particle: Particle index for the deformable endpoint.
            body_point: Body-local attachment point [m].
            stiffness: Quadratic ADMM attachment stiffness [N/m].
            damping: Quadratic ADMM attachment damping [N*s/m].
            enabled: Whether the attachment row is active.

        Returns:
            The custom-frequency row index for the attachment.
        """
        cls.register_custom_attributes(builder)
        point = wp.vec3(float(body_point[0]), float(body_point[1]), float(body_point[2]))
        indices = builder.add_custom_values(
            **{
                cls.BODY_PARTICLE_ATTACHMENT_BODY_ATTR: int(body),
                cls.BODY_PARTICLE_ATTACHMENT_PARTICLE_ATTR: int(particle),
                cls.BODY_PARTICLE_ATTACHMENT_BODY_POINT_ATTR: point,
                cls.BODY_PARTICLE_ATTACHMENT_STIFFNESS_ATTR: float(stiffness),
                cls.BODY_PARTICLE_ATTACHMENT_DAMPING_ATTR: float(damping),
                cls.BODY_PARTICLE_ATTACHMENT_ENABLED_ATTR: bool(enabled),
            }
        )
        return indices[cls.BODY_PARTICLE_ATTACHMENT_BODY_ATTR]

    @dataclass(frozen=True)
    class ContactPair:
        """One cross-solver contact interface for ADMM coupling.

        A ``ContactPair`` activates ADMM contacts between two solver entries.
        The coupler inspects ownership for ``source`` and ``destination`` and
        emits the applicable subset of {rigid-rigid, rigid-particle,
        particle-particle} ADMM contact rows. If neither entry owns shapes or
        particles, no contacts are emitted.

        Friction is derived from shape and particle material properties
        (``shape_material_mu`` and ``Model.particle_mu``), so it is not a
        ContactPair field — set those on the model to control friction.

        Args:
            source: Name of one solver entry.
            destination: Name of the other solver entry. Must differ from
                ``source``.
            contact_distance: Optional minimum contact gap [m]. ``None`` uses
                collision margins (rigid-rigid), particle radii (rigid-particle),
                or radius sums (particle-particle).
            detection_margin: Optional detection margin [m]. For
                rigid-particle pairs, this sets the soft-contact margin on the
                shared collision pipeline. For particle-particle pairs, this
                sets the hash-grid detection margin. ``None`` keeps the
                solver's default.
        """

        source: str
        destination: str
        contact_distance: float | None = None
        detection_margin: float | None = None

    @dataclass(frozen=True)
    class Config:
        """Linearized ADMM coupling configuration.

        Args:
            iterations: Number of ADMM iterations per solver step.
            rho: ADMM penalty parameter.
            gamma: Proximal mass scaling parameter.
            baumgarte: Position error correction fraction.
            joint_stiffness: Quadratic stiffness for translational ADMM
                attachments derived from cross-solver model joints [N/m].
            joint_damping: Quadratic damping for translational ADMM
                attachments derived from cross-solver model joints [N*s/m].
            joint_angular_stiffness: Quadratic stiffness for angular ADMM
                attachments derived from cross-solver fixed and revolute
                joints [N*m/rad].
            joint_angular_damping: Quadratic damping for angular ADMM
                attachments derived from cross-solver fixed and revolute
                joints [N*m*s/rad].
            contact_pairs: Per-interface contact pairs to enable. Empty list
                disables ADMM-managed contacts. Use
                :meth:`SolverAdmmCoupled.auto_detect_contact_pairs` to build the
                old auto-discovery list.
        """

        iterations: int = 5
        rho: float = 1.0
        gamma: float = 0.0
        baumgarte: float = 0.0
        joint_stiffness: float = 1.0e4
        joint_damping: float = 0.0
        joint_angular_stiffness: float = 1.0e4
        joint_angular_damping: float = 0.0
        contact_pairs: Sequence[SolverAdmmCoupled.ContactPair] = ()

    def __init__(
        self,
        model: Model,
        entries: Sequence[SolverCoupled.Entry],
        coupling: SolverAdmmCoupled.Config,
    ) -> None:
        self._admm_buffers: dict[str, _AdmmBuffers] = {}
        self._admm_rr_groups: list[_AdmmRigidRigidAttachmentGroup] = []
        self._admm_rr_angular_groups: list[_AdmmRigidRigidAngularAttachmentGroup] = []
        self._admm_rr_revolute_angular_groups: list[_AdmmRigidRigidAngularAttachmentGroup] = []
        self._admm_rr_angular_friction_groups: list[_AdmmRigidRigidAngularFrictionGroup] = []
        self._admm_rp_groups: list[_AdmmRigidParticleAttachmentGroup] = []
        self._admm_rr_contact_groups: list[_AdmmRigidRigidContactGroup] = []
        self._admm_dynamic_rr_contact_groups: list[_AdmmRigidRigidContactGroup] = []
        self._admm_rp_contact_groups: list[_AdmmRigidParticleContactGroup] = []
        self._admm_dynamic_rp_contact_groups: list[_AdmmRigidParticleContactGroup] = []
        self._admm_dynamic_pp_contact_groups: list[_AdmmParticleParticleContactGroup] = []
        self._admm_pp_contact_groups: list[_AdmmParticleParticleContactGroup] = []
        self._admm_rigid_particle_contact_specs: list[_AdmmRigidParticleContactSpec] = []
        self._admm_rigid_rigid_contact_specs: list[_AdmmRigidRigidContactSpec] = []
        self._admm_particle_particle_contact_specs: list[_AdmmParticleParticleContactSpec] = []
        self._admm_collision_pipeline = None
        self._admm_internal_contacts = None
        self._admm_particle_contact_grid = None
        self._admm_particle_contact_query_radius = 0.0
        self._entry_body_sets: dict[str, set[int]] = {}
        self._entry_particle_sets: dict[str, set[int]] = {}
        self._admm_rigid_particle_shape_filters: dict[int, set[int] | None] = {}
        self._admm_effective_mass_unsupported: set[tuple[str, int]] = set()

        super().__init__(
            model=model,
            entries=entries,
            coupling=coupling,
        )

        self._setup_admm(coupling)

    def _sum_active_count(self, attr: str) -> int:
        """Sum a per-group active-count array across all dynamic contact groups.

        Each `.numpy()` call is a device-to-host sync — paid once per group per call.
        """
        total = 0
        for groups in (
            self._admm_dynamic_rr_contact_groups,
            self._admm_dynamic_rp_contact_groups,
            self._admm_dynamic_pp_contact_groups,
        ):
            for group in groups:
                counter = getattr(group, attr)
                if counter is not None:
                    total += min(int(counter.numpy()[0]), group.count)
        return total

    @property
    def collision_contact_count(self) -> int:
        """Number of collision-detected ADMM contacts active in the last step."""
        return self._sum_active_count("active_count")

    @property
    def collision_contact_count_max(self) -> int:
        """Maximum collision-detected ADMM contact count observed so far."""
        return self._sum_active_count("active_count_max")

    def _customize_view(self, name: str, view: ModelView, body_indices: wp.array) -> None:
        """Apply ADMM proximal mass scaling before sub-solver construction."""
        del name
        gamma = float(self._coupling.gamma)
        if gamma <= 0.0:
            return
        scale = 1.0 + gamma
        if body_indices.shape[0] > 0:
            view.scale_body_mass(body_indices, scale)
        view.scale_particle_mass(scale)

    def _setup_admm(self, coupling: SolverAdmmCoupled.Config) -> None:
        for entry in self._entries.values():
            buf = _AdmmBuffers()
            s0 = entry.state_0
            if s0.body_q is not None:
                buf.body_q_n = wp.empty_like(s0.body_q)
                buf.body_qd_n = wp.empty_like(s0.body_qd)
                buf.body_qd_k = wp.empty_like(s0.body_qd)
            if s0.body_f is not None:
                buf.body_f = wp.empty_like(s0.body_f)
            if s0.particle_q is not None:
                buf.particle_q_n = wp.empty_like(s0.particle_q)
                buf.particle_qd_n = wp.empty_like(s0.particle_qd)
                buf.particle_qd_k = wp.empty_like(s0.particle_qd)
            if s0.particle_f is not None:
                buf.particle_f = wp.empty_like(s0.particle_f)
            if s0.joint_q is not None:
                buf.joint_q_n = wp.empty_like(s0.joint_q)
                buf.joint_qd_n = wp.empty_like(s0.joint_qd)
                buf.joint_qd_k = wp.empty_like(s0.joint_qd)
            self._admm_buffers[entry.name] = buf

        if coupling.gamma > 0.0:
            for entry in self._entries.values():
                entry.solver.notify_model_changed(SolverNotifyFlags.BODY_INERTIAL_PROPERTIES)

        self._entry_body_sets = {
            name: {int(i) for i in entry.body_indices.numpy()} for name, entry in self._entries.items()
        }
        self._entry_particle_sets = {
            name: {int(i) for i in entry.particle_indices.numpy()} for name, entry in self._entries.items()
        }

        for entry in self._entries.values():
            self._setup_admm_effective_mass_buffers(entry, self._admm_buffers[entry.name])

        self._build_admm_joint_groups(coupling)
        self._build_admm_body_particle_attachment_groups()
        self._setup_admm_contact_specs(coupling)

        if self._admm_rigid_particle_contact_specs or self._admm_rigid_rigid_contact_specs:
            if self._admm_rigid_particle_contact_specs:
                self._validate_rigid_particle_contact_specs()
            if self._admm_rigid_rigid_contact_specs:
                self._validate_rigid_rigid_contact_specs()
            self._admm_rigid_particle_shape_filters = {
                spec_idx: None if spec.shapes is None else {int(shape) for shape in spec.shapes}
                for spec_idx, spec in enumerate(self._admm_rigid_particle_contact_specs)
            }
            from ...sim import CollisionPipeline  # noqa: PLC0415

            self._admm_collision_pipeline = CollisionPipeline(
                self.model,
                broad_phase="explicit",
                soft_contact_margin=float(self._rigid_particle_detection_margin()),
            )
            if self._admm_rigid_particle_contact_specs:
                self._admm_dynamic_rp_contact_groups = self._build_collision_rigid_particle_contact_groups()
            if self._admm_rigid_rigid_contact_specs:
                self._admm_dynamic_rr_contact_groups = self._build_collision_rigid_rigid_contact_groups()

        if self._admm_particle_particle_contact_specs:
            self._validate_particle_particle_contact_specs()
            self._admm_dynamic_pp_contact_groups = self._build_collision_particle_particle_contact_groups()
            if self._admm_dynamic_pp_contact_groups:
                self._admm_particle_contact_query_radius = max(
                    group.query_radius for group in self._admm_dynamic_pp_contact_groups
                )
                with wp.ScopedDevice(self.model.device):
                    self._admm_particle_contact_grid = wp.HashGrid(128, 128, 128)
                    self._admm_particle_contact_grid.reserve(self.model.particle_count)

        self._admm_rr_contact_groups = list(self._admm_dynamic_rr_contact_groups)
        self._admm_rp_contact_groups = list(self._admm_dynamic_rp_contact_groups)
        self._admm_pp_contact_groups = list(self._admm_dynamic_pp_contact_groups)

        # Eagerly allocate the internal contact buffer so it exists before any
        # CUDA graph capture. Lazy allocation during capture leaves a bogus
        # pointer in the captured graph.
        if (
            self._admm_dynamic_rr_contact_groups or self._admm_dynamic_rp_contact_groups
        ) and self._admm_internal_contacts is None:
            self._admm_internal_contacts = self._admm_collision_pipeline.contacts()

    def _setup_admm_effective_mass_buffers(self, entry: SolverEntry, buf: _AdmmBuffers) -> None:
        device = self.model.device
        if self.model.body_mass is not None:
            body_mass = self.model.body_mass.numpy().copy()
            self._apply_custom_effective_mass(
                entry,
                CouplingEndpointKind.BODY,
                entry.body_indices,
                body_mass,
            )
            buf.body_effective_mass = wp.array(body_mass, dtype=float, device=device)
        if self.model.particle_mass is not None:
            particle_mass = self.model.particle_mass.numpy().copy()
            self._apply_custom_effective_mass(
                entry,
                CouplingEndpointKind.PARTICLE,
                entry.particle_indices,
                particle_mass,
            )
            buf.particle_effective_mass = wp.array(particle_mass, dtype=float, device=device)

    def _apply_custom_effective_mass(
        self,
        entry: SolverEntry,
        endpoint_kind: CouplingEndpointKind,
        endpoint_indices: wp.array,
        mass_values,
    ) -> None:
        masses = self._eval_effective_masses(
            entry,
            endpoint_kind,
            endpoint_indices,
            raise_on_unsupported=False,
        )
        if masses is None:
            self._admm_effective_mass_unsupported.add((entry.name, int(endpoint_kind)))
            return
        indices = [int(i) for i in endpoint_indices.numpy()]
        for index, mass in zip(indices, masses, strict=True):
            mass_values[index] = float(mass)

    def _body_effective_mass_np(self, entry_name: str):
        buf = self._admm_buffers[entry_name]
        return buf.body_effective_mass.numpy() if buf.body_effective_mass is not None else []

    def _particle_effective_mass_np(self, entry_name: str):
        buf = self._admm_buffers[entry_name]
        return buf.particle_effective_mass.numpy() if buf.particle_effective_mass is not None else []

    def _require_effective_mass(self, entry_name: str, endpoint_kind: CouplingEndpointKind) -> None:
        if (entry_name, int(endpoint_kind)) not in self._admm_effective_mass_unsupported:
            return
        solver = self._entries[entry_name].solver
        raise NotImplementedError(
            f"{solver.__class__.__name__} does not support coupling hook {CouplingHook.EFFECTIVE_MASS_DIAGONAL.name}"
        )

    @staticmethod
    def _interface_weight(m_a: float, m_b: float) -> float:
        if m_a > 0.0 and m_b > 0.0:
            return ((m_a * m_b) / (m_a + m_b)) ** 0.5
        return 1.0

    def _setup_admm_contact_specs(self, coupling: SolverAdmmCoupled.Config) -> None:
        """Populate dynamic ADMM contact specs from configured contact pairs."""
        if not coupling.contact_pairs:
            return

        # Discover all candidate specs from model state (one rigid-rigid/rigid-particle
        # /particle-particle entry per cross-owner combination), then keep only those
        # whose owner pair appears in the user's ContactPair list. ContactPair fields
        # override per-pair friction/contact_distance.
        pair_by_owners: dict[frozenset[str], SolverAdmmCoupled.ContactPair] = {}
        for pair in coupling.contact_pairs:
            if pair.source == pair.destination:
                raise ValueError(f"ADMM ContactPair requires distinct source and destination, got {pair.source!r}")
            if pair.source not in self._entries:
                raise ValueError(f"Unknown ADMM ContactPair source {pair.source!r}")
            if pair.destination not in self._entries:
                raise ValueError(f"Unknown ADMM ContactPair destination {pair.destination!r}")
            if pair.contact_distance is not None and pair.contact_distance < 0.0:
                raise ValueError("ADMM ContactPair contact_distance must be non-negative")
            if pair.detection_margin is not None and pair.detection_margin < 0.0:
                raise ValueError("ADMM ContactPair detection_margin must be non-negative")
            key = frozenset({pair.source, pair.destination})
            if key in pair_by_owners:
                raise ValueError(f"Duplicate ADMM ContactPair for entries {pair.source!r} and {pair.destination!r}")
            pair_by_owners[key] = pair

        rp_specs = self._discover_rigid_particle_contact_specs()
        rr_specs = self._discover_rigid_rigid_contact_specs()
        pp_specs = self._discover_particle_particle_contact_specs()

        def matching_pair(owner_a: str, owner_b: str):
            return pair_by_owners.get(frozenset({owner_a, owner_b}))

        def override(spec, pair):
            return spec.__class__(
                **{
                    **{k: getattr(spec, k) for k in spec.__dataclass_fields__},
                    "contact_distance": pair.contact_distance,
                    **(
                        {"detection_margin": pair.detection_margin}
                        if "detection_margin" in spec.__dataclass_fields__
                        else {}
                    ),
                }
            )

        self._admm_rigid_particle_contact_specs = [
            override(spec, matching_pair(spec.body_owner, spec.particle_owner))
            for spec in rp_specs
            if matching_pair(spec.body_owner, spec.particle_owner) is not None
        ]
        self._admm_rigid_rigid_contact_specs = [
            override(spec, matching_pair(spec.owner_a, spec.owner_b))
            for spec in rr_specs
            if matching_pair(spec.owner_a, spec.owner_b) is not None
        ]
        self._admm_particle_particle_contact_specs = [
            override(spec, matching_pair(spec.owner_a, spec.owner_b))
            for spec in pp_specs
            if matching_pair(spec.owner_a, spec.owner_b) is not None
        ]

    @classmethod
    def auto_detect_contact_pairs(
        cls,
        entries: Sequence[SolverCoupled.Entry],
        *,
        contact_distance: float | None = None,
        detection_margin: float | None = None,
    ) -> list[SolverAdmmCoupled.ContactPair]:
        """Return ContactPair entries for every cross-owner interface.

        Mirrors the prior auto-detection behavior: a pair is emitted for every
        distinct combination of entries. Friction is read from
        ``shape_material_mu`` and ``Model.particle_mu`` at contact-fill time;
        only ``contact_distance`` and ``detection_margin`` need to be supplied
        here, and they default to the solver's own defaults.

        Args:
            entries: Sub-solver entries that will be passed to
                :class:`SolverAdmmCoupled`.
            contact_distance: Default minimum contact gap [m].
            detection_margin: Default detection margin [m].
        """
        names = [e.name for e in entries]
        pairs: list[SolverAdmmCoupled.ContactPair] = []
        for i, a in enumerate(names):
            for b in names[i + 1 :]:
                pairs.append(
                    cls.ContactPair(
                        source=a,
                        destination=b,
                        contact_distance=contact_distance,
                        detection_margin=detection_margin,
                    )
                )
        return pairs

    def _shape_flagged(self, shape_flags, shape: int, flag: ShapeFlags) -> bool:
        if shape_flags is None:
            return False
        return bool(int(shape_flags[shape]) & int(flag))

    def _rigid_particle_detection_margin(self) -> float:
        """Return the soft-contact margin to apply on the shared collision pipeline.

        The collision pipeline is shared across all rigid-particle ADMM pairs,
        so it must use a single margin. We pick the max of all
        rigid-particle ``ContactPair.detection_margin`` values (defaulting to
        :data:`_DEFAULT_DETECTION_MARGIN` when unset).
        """
        margin = _DEFAULT_DETECTION_MARGIN
        for spec in self._admm_rigid_particle_contact_specs:
            value = _DEFAULT_DETECTION_MARGIN if spec.detection_margin is None else float(spec.detection_margin)
            margin = max(margin, value)
        return margin

    def _discover_rigid_particle_contact_specs(self) -> list[_AdmmRigidParticleContactSpec]:
        shape_body = self.model.shape_body.numpy() if self.model.shape_body is not None else []
        shape_flags = self.model.shape_flags.numpy() if getattr(self.model, "shape_flags", None) is not None else None
        shapes_by_owner: dict[str, list[int]] = {}
        for shape in range(self.model.shape_count):
            if not self._shape_flagged(shape_flags, shape, ShapeFlags.COLLIDE_PARTICLES):
                continue
            body = int(shape_body[shape])
            owner = self._entry_name_for_body(body)
            if owner is None:
                continue
            shapes_by_owner.setdefault(owner, []).append(shape)

        specs: list[_AdmmRigidParticleContactSpec] = []
        for particle_owner, particles in self._entry_particle_sets.items():
            if not particles:
                continue
            for body_owner, shapes in shapes_by_owner.items():
                if body_owner == particle_owner or not shapes:
                    continue
                specs.append(
                    _AdmmRigidParticleContactSpec(
                        particle_owner=particle_owner,
                        body_owner=body_owner,
                        shapes=tuple(shapes),
                    )
                )
        return specs

    def _discover_rigid_rigid_contact_specs(self) -> list[_AdmmRigidRigidContactSpec]:
        if getattr(self.model, "shape_contact_pairs", None) is None:
            return []
        shape_body = self.model.shape_body.numpy() if self.model.shape_body is not None else []
        entry_order = {name: i for i, name in enumerate(self._entries)}
        grouped: dict[tuple[str, str], tuple[set[int], set[int]]] = {}
        for pair in self.model.shape_contact_pairs.numpy():
            shape_a = int(pair[0])
            shape_b = int(pair[1])
            body_a = int(shape_body[shape_a])
            body_b = int(shape_body[shape_b])
            owner_a = self._entry_name_for_body(body_a)
            owner_b = self._entry_name_for_body(body_b)
            if owner_a is None or owner_b is None or owner_a == owner_b:
                continue
            if entry_order[owner_b] < entry_order[owner_a]:
                owner_a, owner_b = owner_b, owner_a
                shape_a, shape_b = shape_b, shape_a
            shapes_a, shapes_b = grouped.setdefault((owner_a, owner_b), (set(), set()))
            shapes_a.add(shape_a)
            shapes_b.add(shape_b)

        return [
            _AdmmRigidRigidContactSpec(
                owner_a=owner_a,
                owner_b=owner_b,
                shapes_a=tuple(sorted(shapes_a)),
                shapes_b=tuple(sorted(shapes_b)),
            )
            for (owner_a, owner_b), (shapes_a, shapes_b) in grouped.items()
        ]

    def _discover_particle_particle_contact_specs(self) -> list[_AdmmParticleParticleContactSpec]:
        entries = [(name, particles) for name, particles in self._entry_particle_sets.items() if particles]
        specs: list[_AdmmParticleParticleContactSpec] = []
        for i, (owner_a, particles_a) in enumerate(entries):
            for owner_b, particles_b in entries[i + 1 :]:
                if owner_a == owner_b:
                    continue
                specs.append(
                    _AdmmParticleParticleContactSpec(
                        owner_a=owner_a,
                        owner_b=owner_b,
                        particles_a=tuple(sorted(particles_a)),
                        particles_b=tuple(sorted(particles_b)),
                    )
                )
        return specs

    def _validate_rigid_particle_contact_specs(self) -> None:
        for spec in self._admm_rigid_particle_contact_specs:
            if spec.particle_owner not in self._entries:
                raise ValueError(f"Unknown ADMM rigid-particle contact particle owner '{spec.particle_owner}'")
            if spec.body_owner not in self._entries:
                raise ValueError(f"Unknown ADMM rigid-particle contact body owner '{spec.body_owner}'")
            if not self._entry_particle_sets.get(spec.particle_owner):
                raise ValueError(
                    f"ADMM rigid-particle contact particle owner '{spec.particle_owner}' does not own any particles"
                )
            if not self._entry_body_sets.get(spec.body_owner):
                raise ValueError(f"ADMM rigid-particle contact body owner '{spec.body_owner}' does not own any bodies")
            if spec.contact_distance is not None and spec.contact_distance < 0.0:
                raise ValueError("ADMM rigid-particle contact distances must be non-negative")
            if spec.shapes is None:
                continue
            for shape in spec.shapes:
                shape_index = int(shape)
                if shape_index < 0 or shape_index >= self.model.shape_count:
                    raise IndexError(f"ADMM rigid-particle contact shape index {shape_index} out of range")

    def _validate_rigid_rigid_contact_specs(self) -> None:
        for spec in self._admm_rigid_rigid_contact_specs:
            if spec.owner_a not in self._entries:
                raise ValueError(f"Unknown ADMM rigid-rigid contact owner '{spec.owner_a}'")
            if spec.owner_b not in self._entries:
                raise ValueError(f"Unknown ADMM rigid-rigid contact owner '{spec.owner_b}'")
            if spec.owner_a == spec.owner_b:
                raise ValueError("ADMM rigid-rigid contacts require distinct solver owners")
            if not self._entry_body_sets.get(spec.owner_a):
                raise ValueError(f"ADMM rigid-rigid contact owner '{spec.owner_a}' does not own any bodies")
            if not self._entry_body_sets.get(spec.owner_b):
                raise ValueError(f"ADMM rigid-rigid contact owner '{spec.owner_b}' does not own any bodies")
            if spec.contact_distance is not None and spec.contact_distance < 0.0:
                raise ValueError("ADMM rigid-rigid contact distances must be non-negative")
            self._validate_shape_contact_subset(spec.owner_a, spec.shapes_a)
            self._validate_shape_contact_subset(spec.owner_b, spec.shapes_b)

    def _validate_particle_particle_contact_specs(self) -> None:
        for spec in self._admm_particle_particle_contact_specs:
            if spec.owner_a not in self._entries:
                raise ValueError(f"Unknown ADMM particle-particle contact owner '{spec.owner_a}'")
            if spec.owner_b not in self._entries:
                raise ValueError(f"Unknown ADMM particle-particle contact owner '{spec.owner_b}'")
            if spec.owner_a == spec.owner_b:
                raise ValueError("ADMM particle-particle contacts require distinct solver owners")
            if not self._entry_particle_sets.get(spec.owner_a):
                raise ValueError(f"ADMM particle-particle contact owner '{spec.owner_a}' does not own any particles")
            if not self._entry_particle_sets.get(spec.owner_b):
                raise ValueError(f"ADMM particle-particle contact owner '{spec.owner_b}' does not own any particles")
            if spec.contact_distance is not None and spec.contact_distance < 0.0:
                raise ValueError("ADMM particle-particle contact distances must be non-negative")
            if spec.detection_margin is not None and spec.detection_margin < 0.0:
                raise ValueError("ADMM particle-particle contact margins must be non-negative")
            self._validate_particle_contact_subset(spec.owner_a, spec.particles_a)
            self._validate_particle_contact_subset(spec.owner_b, spec.particles_b)

    def _validate_particle_contact_subset(self, owner: str, particles: Sequence[int] | None) -> None:
        if particles is None:
            return
        owner_particles = self._entry_particle_sets[owner]
        for particle in particles:
            particle_index = int(particle)
            if particle_index < 0 or particle_index >= self.model.particle_count:
                raise IndexError(f"ADMM particle-particle contact particle index {particle_index} out of range")
            if particle_index not in owner_particles:
                raise ValueError(f"ADMM particle-particle contact particle {particle_index} is not owned by '{owner}'")

    def _validate_shape_contact_subset(self, owner: str, shapes: Sequence[int] | None) -> None:
        if shapes is None:
            return
        owner_bodies = self._entry_body_sets[owner]
        shape_body = self.model.shape_body.numpy() if self.model.shape_body is not None else []
        for shape in shapes:
            shape_index = int(shape)
            if shape_index < 0 or shape_index >= self.model.shape_count:
                raise IndexError(f"ADMM rigid-rigid contact shape index {shape_index} out of range")
            if int(shape_body[shape_index]) not in owner_bodies:
                raise ValueError(f"ADMM rigid-rigid contact shape {shape_index} is not owned by '{owner}'")

    @staticmethod
    def _transform_from_row(row) -> wp.transform:
        return wp.transform(
            wp.vec3(float(row[0]), float(row[1]), float(row[2])),
            wp.quat(float(row[3]), float(row[4]), float(row[5]), float(row[6])),
        )

    @staticmethod
    def _transform_translation_from_row(row) -> tuple[float, float, float]:
        return float(row[0]), float(row[1]), float(row[2])

    @staticmethod
    def _quat_multiply_np(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        av = a[:3]
        aw = float(a[3])
        bv = b[:3]
        bw = float(b[3])
        out = np.empty(4, dtype=np.float32)
        out[:3] = aw * bv + bw * av + np.cross(av, bv)
        out[3] = aw * bw - float(np.dot(av, bv))
        return out

    @staticmethod
    def _quat_inverse_np(q: np.ndarray) -> np.ndarray:
        inv = np.array([-q[0], -q[1], -q[2], q[3]], dtype=np.float32)
        norm_sq = float(np.dot(q, q))
        if norm_sq > 0.0:
            inv /= norm_sq
        return inv

    @staticmethod
    def _quat_rotate_np(q: np.ndarray, v: np.ndarray) -> np.ndarray:
        qv = q[:3]
        qw = float(q[3])
        return v + 2.0 * np.cross(qv, np.cross(qv, v) + qw * v)

    @staticmethod
    def _quat_from_x_axis_np(direction: np.ndarray) -> wp.quat:
        direction = np.asarray(direction, dtype=np.float32)
        length = float(np.linalg.norm(direction))
        if length <= 0.0:
            raise ValueError("Cannot build a revolute ADMM frame from a zero axis")
        direction = direction / length
        source = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        axis = np.cross(source, direction)
        w = 1.0 + float(np.dot(source, direction))
        if w < 1.0e-6:
            axis = np.array([0.0, 1.0, 0.0], dtype=np.float32)
            w = 0.0
        quat = np.array([axis[0], axis[1], axis[2], w], dtype=np.float32)
        quat /= np.linalg.norm(quat)
        return wp.quat(float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3]))

    @classmethod
    def _revolute_axis_frames_from_rows(
        cls,
        joint_X_p_row,
        joint_X_c_row,
        axis_parent: np.ndarray,
    ) -> tuple[wp.transform, wp.transform]:
        q_parent = np.asarray(joint_X_p_row[3:7], dtype=np.float32)
        q_child = np.asarray(joint_X_c_row[3:7], dtype=np.float32)
        parent_to_child = cls._quat_multiply_np(cls._quat_inverse_np(q_child), q_parent)
        axis_child = cls._quat_rotate_np(parent_to_child, np.asarray(axis_parent, dtype=np.float32))
        frame_child = wp.transform(
            wp.vec3(float(joint_X_c_row[0]), float(joint_X_c_row[1]), float(joint_X_c_row[2])),
            cls._quat_from_x_axis_np(axis_child),
        )
        frame_parent = wp.transform(
            wp.vec3(float(joint_X_p_row[0]), float(joint_X_p_row[1]), float(joint_X_p_row[2])),
            cls._quat_from_x_axis_np(axis_parent),
        )
        return frame_child, frame_parent

    @staticmethod
    def _inertia_scalar(inertia) -> float:
        mat = np.asarray(inertia, dtype=np.float32)
        if mat.shape == (3, 3):
            value = float(np.trace(mat) / 3.0)
        else:
            value = float(np.mean(mat))
        return max(value, 0.0)

    def _entry_name_for_body(self, body: int) -> str | None:
        if body < 0 or body >= len(self._body_owner):
            return None
        owner = self._body_owner[body]
        if owner < 0:
            return None
        return self._entry_configs[owner].name

    def _entry_name_for_particle(self, particle: int) -> str | None:
        if particle < 0 or particle >= len(self._particle_owner):
            return None
        owner = self._particle_owner[particle]
        if owner < 0:
            return None
        return self._entry_configs[owner].name

    def _body_local_id(self, entry_name: str, body: int) -> int:
        mapping = self._entries[entry_name].body_global_to_local.numpy()
        local = int(mapping[body]) if 0 <= body < len(mapping) else -1
        if local < 0:
            raise ValueError(f"Body {body} is not visible in coupled solver entry {entry_name!r}")
        return local

    def _cross_solver_joint_entries(self, joint: int, parent: int, child: int) -> tuple[str, str] | None:
        parent_entry = self._entry_name_for_body(parent)
        child_entry = self._entry_name_for_body(child)
        if parent_entry is None or child_entry is None or parent_entry == child_entry:
            return None
        if self._joint_owner[joint] >= 0:
            raise ValueError(
                f"ADMM cross-solver joint {joint} must not be owned by a sub-solver entry; "
                "leave it to SolverAdmmCoupled so the constraint is not applied twice"
            )
        return child_entry, parent_entry

    def _angular_effective_weight(self, entry_name_a: str, body_a: int, entry_name_b: str, body_b: int) -> float:
        ids_a = wp.array([body_a], dtype=int, device=self.model.device)
        ids_b = wp.array([body_b], dtype=int, device=self.model.device)
        props_a = self._eval_effective_body_inertial_properties(
            self._entries[entry_name_a], ids_a, raise_on_unsupported=False
        )
        props_b = self._eval_effective_body_inertial_properties(
            self._entries[entry_name_b], ids_b, raise_on_unsupported=False
        )
        if props_a is None or props_b is None:
            return 1.0
        inertia_a = self._inertia_scalar(props_a[1][0])
        inertia_b = self._inertia_scalar(props_b[1][0])
        return self._interface_weight(inertia_a, inertia_b)

    def _build_admm_joint_groups(self, coupling: SolverAdmmCoupled.Config) -> None:
        """Build quadratic ADMM attachments from cross-solver model joints."""
        if (
            coupling.joint_stiffness < 0.0
            or coupling.joint_damping < 0.0
            or coupling.joint_angular_stiffness < 0.0
            or coupling.joint_angular_damping < 0.0
        ):
            raise ValueError("ADMM joint attachment stiffness and damping values must be non-negative")
        if self.model.joint_count == 0:
            return

        joint_type = self.model.joint_type.numpy()
        joint_parent = self.model.joint_parent.numpy()
        joint_child = self.model.joint_child.numpy()
        joint_enabled = self.model.joint_enabled.numpy()
        joint_X_p = self.model.joint_X_p.numpy()
        joint_X_c = self.model.joint_X_c.numpy()
        joint_qd_start = self.model.joint_qd_start.numpy()
        joint_friction = self.model.joint_friction.numpy()
        joint_axis = self.model.joint_axis.numpy()

        point_items: dict[
            tuple[str, str],
            list[tuple[int, tuple[float, float, float], int, tuple[float, float, float], float, float]],
        ] = {}
        angular_items: dict[tuple[str, str], list[tuple[int, wp.transform, int, wp.transform, float, float]]] = {}
        revolute_angular_items: dict[
            tuple[str, str], list[tuple[int, wp.transform, int, wp.transform, float, float]]
        ] = {}
        angular_friction_items: dict[
            tuple[str, str], list[tuple[int, wp.transform, int, tuple[float, float, float]]]
        ] = {}

        for joint in range(self.model.joint_count):
            if not bool(joint_enabled[joint]):
                continue
            parent = int(joint_parent[joint])
            child = int(joint_child[joint])
            owner_pair = self._cross_solver_joint_entries(joint, parent, child)
            if owner_pair is None:
                continue

            child_entry, parent_entry = owner_pair
            jtype = int(joint_type[joint])
            if jtype == int(JointType.BALL):
                point_items.setdefault((child_entry, parent_entry), []).append(
                    (
                        child,
                        self._transform_translation_from_row(joint_X_c[joint]),
                        parent,
                        self._transform_translation_from_row(joint_X_p[joint]),
                        float(coupling.joint_stiffness),
                        float(coupling.joint_damping),
                    )
                )
                qd_start = int(joint_qd_start[joint])
                friction = (
                    float(joint_friction[qd_start + 0]),
                    float(joint_friction[qd_start + 1]),
                    float(joint_friction[qd_start + 2]),
                )
                if friction[0] < 0.0 or friction[1] < 0.0 or friction[2] < 0.0:
                    raise ValueError(f"ADMM cross-solver ball joint {joint} has negative friction")
                if friction[0] > 0.0 or friction[1] > 0.0 or friction[2] > 0.0:
                    angular_friction_items.setdefault((child_entry, parent_entry), []).append(
                        (
                            child,
                            self._transform_from_row(joint_X_c[joint]),
                            parent,
                            friction,
                        )
                    )
            elif jtype == int(JointType.REVOLUTE):
                point_items.setdefault((child_entry, parent_entry), []).append(
                    (
                        child,
                        self._transform_translation_from_row(joint_X_c[joint]),
                        parent,
                        self._transform_translation_from_row(joint_X_p[joint]),
                        float(coupling.joint_stiffness),
                        float(coupling.joint_damping),
                    )
                )
                qd_start = int(joint_qd_start[joint])
                axis_parent = np.asarray(joint_axis[qd_start], dtype=np.float32)
                frame_child, frame_parent = self._revolute_axis_frames_from_rows(
                    joint_X_p[joint],
                    joint_X_c[joint],
                    axis_parent,
                )
                revolute_angular_items.setdefault((child_entry, parent_entry), []).append(
                    (
                        child,
                        frame_child,
                        parent,
                        frame_parent,
                        float(coupling.joint_angular_stiffness),
                        float(coupling.joint_angular_damping),
                    )
                )
                friction_value = float(joint_friction[qd_start])
                if friction_value < 0.0:
                    raise ValueError(f"ADMM cross-solver revolute joint {joint} has negative friction")
                if friction_value > 0.0:
                    angular_friction_items.setdefault((child_entry, parent_entry), []).append(
                        (
                            child,
                            frame_child,
                            parent,
                            (friction_value, 0.0, 0.0),
                        )
                    )
            elif jtype == int(JointType.FIXED):
                point_items.setdefault((child_entry, parent_entry), []).append(
                    (
                        child,
                        self._transform_translation_from_row(joint_X_c[joint]),
                        parent,
                        self._transform_translation_from_row(joint_X_p[joint]),
                        float(coupling.joint_stiffness),
                        float(coupling.joint_damping),
                    )
                )
                angular_items.setdefault((child_entry, parent_entry), []).append(
                    (
                        child,
                        self._transform_from_row(joint_X_c[joint]),
                        parent,
                        self._transform_from_row(joint_X_p[joint]),
                        float(coupling.joint_angular_stiffness),
                        float(coupling.joint_angular_damping),
                    )
                )
            elif jtype in (int(JointType.FREE), int(JointType.DISTANCE)):
                continue
            else:
                name = JointType(jtype).name if jtype in [int(t) for t in JointType] else str(jtype)
                raise NotImplementedError(
                    f"ADMM cross-solver model joint {joint} has unsupported type {name}; "
                    "only BALL, REVOLUTE, and FIXED joints are currently mapped to ADMM attachments"
                )

        device = self.model.device
        for (entry_name_a, entry_name_b), items in point_items.items():
            self._require_effective_mass(entry_name_a, CouplingEndpointKind.BODY)
            self._require_effective_mass(entry_name_b, CouplingEndpointKind.BODY)
            body_mass_np_a = self._body_effective_mass_np(entry_name_a)
            body_mass_np_b = self._body_effective_mass_np(entry_name_b)
            body_ids_a = [self._body_local_id(entry_name_a, item[0]) for item in items]
            points_a = [wp.vec3(*item[1]) for item in items]
            body_ids_b = [self._body_local_id(entry_name_b, item[2]) for item in items]
            points_b = [wp.vec3(*item[3]) for item in items]
            kappa = [item[4] for item in items]
            damping = [item[5] for item in items]
            W = []
            for body_a, _, body_b, _, _, _ in items:
                m_a = float(body_mass_np_a[body_a]) if len(body_mass_np_a) > body_a else 0.0
                m_b = float(body_mass_np_b[body_b]) if len(body_mass_np_b) > body_b else 0.0
                W.append(self._interface_weight(m_a, m_b))

            n = len(items)
            self._admm_rr_groups.append(
                _AdmmRigidRigidAttachmentGroup(
                    body_entry_name_a=entry_name_a,
                    body_entry_name_b=entry_name_b,
                    body_ids_a=wp.array(body_ids_a, dtype=int, device=device),
                    point_a=wp.array(points_a, dtype=wp.vec3, device=device),
                    body_ids_b=wp.array(body_ids_b, dtype=int, device=device),
                    point_b=wp.array(points_b, dtype=wp.vec3, device=device),
                    kappa=wp.array(kappa, dtype=float, device=device),
                    damping=wp.array(damping, dtype=float, device=device),
                    W=wp.array(W, dtype=float, device=device),
                    u=wp.zeros(n, dtype=wp.vec3, device=device),
                    lambda_=wp.zeros(n, dtype=wp.vec3, device=device),
                    Jv=wp.zeros(n, dtype=wp.vec3, device=device),
                    u_target=wp.zeros(n, dtype=wp.vec3, device=device),
                )
            )

        for (entry_name_a, entry_name_b), items in angular_items.items():
            body_ids_a = [self._body_local_id(entry_name_a, item[0]) for item in items]
            frames_a = [item[1] for item in items]
            body_ids_b = [self._body_local_id(entry_name_b, item[2]) for item in items]
            frames_b = [item[3] for item in items]
            kappa = [item[4] for item in items]
            damping = [item[5] for item in items]
            W = [
                self._angular_effective_weight(entry_name_a, body_a, entry_name_b, body_b)
                for body_a, _, body_b, _, _, _ in items
            ]
            n = len(items)
            self._admm_rr_angular_groups.append(
                _AdmmRigidRigidAngularAttachmentGroup(
                    body_entry_name_a=entry_name_a,
                    body_entry_name_b=entry_name_b,
                    body_ids_a=wp.array(body_ids_a, dtype=int, device=device),
                    frame_a=wp.array(frames_a, dtype=wp.transform, device=device),
                    body_ids_b=wp.array(body_ids_b, dtype=int, device=device),
                    frame_b=wp.array(frames_b, dtype=wp.transform, device=device),
                    kappa=wp.array(kappa, dtype=float, device=device),
                    damping=wp.array(damping, dtype=float, device=device),
                    W=wp.array(W, dtype=float, device=device),
                    u=wp.zeros(n, dtype=wp.vec3, device=device),
                    lambda_=wp.zeros(n, dtype=wp.vec3, device=device),
                    Jv=wp.zeros(n, dtype=wp.vec3, device=device),
                    u_target=wp.zeros(n, dtype=wp.vec3, device=device),
                )
            )

        for (entry_name_a, entry_name_b), items in revolute_angular_items.items():
            body_ids_a = [self._body_local_id(entry_name_a, item[0]) for item in items]
            frames_a = [item[1] for item in items]
            body_ids_b = [self._body_local_id(entry_name_b, item[2]) for item in items]
            frames_b = [item[3] for item in items]
            kappa = [item[4] for item in items]
            damping = [item[5] for item in items]
            W = [
                self._angular_effective_weight(entry_name_a, body_a, entry_name_b, body_b)
                for body_a, _, body_b, _, _, _ in items
            ]
            n = len(items)
            self._admm_rr_revolute_angular_groups.append(
                _AdmmRigidRigidAngularAttachmentGroup(
                    body_entry_name_a=entry_name_a,
                    body_entry_name_b=entry_name_b,
                    body_ids_a=wp.array(body_ids_a, dtype=int, device=device),
                    frame_a=wp.array(frames_a, dtype=wp.transform, device=device),
                    body_ids_b=wp.array(body_ids_b, dtype=int, device=device),
                    frame_b=wp.array(frames_b, dtype=wp.transform, device=device),
                    kappa=wp.array(kappa, dtype=float, device=device),
                    damping=wp.array(damping, dtype=float, device=device),
                    W=wp.array(W, dtype=float, device=device),
                    u=wp.zeros(n, dtype=wp.vec3, device=device),
                    lambda_=wp.zeros(n, dtype=wp.vec3, device=device),
                    Jv=wp.zeros(n, dtype=wp.vec3, device=device),
                    u_target=wp.zeros(n, dtype=wp.vec3, device=device),
                )
            )

        for (entry_name_a, entry_name_b), items in angular_friction_items.items():
            body_ids_a = [self._body_local_id(entry_name_a, item[0]) for item in items]
            frames_a = [item[1] for item in items]
            body_ids_b = [self._body_local_id(entry_name_b, item[2]) for item in items]
            friction = [wp.vec3(*item[3]) for item in items]
            W = [
                self._angular_effective_weight(entry_name_a, body_a, entry_name_b, body_b)
                for body_a, _, body_b, _ in items
            ]
            n = len(items)
            self._admm_rr_angular_friction_groups.append(
                _AdmmRigidRigidAngularFrictionGroup(
                    body_entry_name_a=entry_name_a,
                    body_entry_name_b=entry_name_b,
                    body_ids_a=wp.array(body_ids_a, dtype=int, device=device),
                    frame_a=wp.array(frames_a, dtype=wp.transform, device=device),
                    body_ids_b=wp.array(body_ids_b, dtype=int, device=device),
                    friction=wp.array(friction, dtype=wp.vec3, device=device),
                    W=wp.array(W, dtype=float, device=device),
                    u=wp.zeros(n, dtype=wp.vec3, device=device),
                    lambda_=wp.zeros(n, dtype=wp.vec3, device=device),
                    Jv=wp.zeros(n, dtype=wp.vec3, device=device),
                )
            )

    def _build_admm_body_particle_attachment_groups(self) -> None:
        """Build quadratic ADMM attachments from model custom attributes."""
        count = int(self.model.custom_frequency_counts.get(self.BODY_PARTICLE_ATTACHMENT_FREQUENCY, 0))
        if count == 0:
            return

        coupling_ns = getattr(self.model, "coupling", None)
        required_attrs = (
            "body_particle_attachment_body",
            "body_particle_attachment_particle",
            "body_particle_attachment_body_point",
            "body_particle_attachment_stiffness",
            "body_particle_attachment_damping",
            "body_particle_attachment_enabled",
        )
        if coupling_ns is None or any(not hasattr(coupling_ns, attr) for attr in required_attrs):
            raise ValueError(
                "ADMM body-particle attachments require SolverAdmmCoupled.register_custom_attributes(builder) "
                "before finalizing the model"
            )

        body_np = coupling_ns.body_particle_attachment_body.numpy()
        particle_np = coupling_ns.body_particle_attachment_particle.numpy()
        point_np = coupling_ns.body_particle_attachment_body_point.numpy()
        stiffness_np = coupling_ns.body_particle_attachment_stiffness.numpy()
        damping_np = coupling_ns.body_particle_attachment_damping.numpy()
        enabled_np = coupling_ns.body_particle_attachment_enabled.numpy()

        grouped: dict[tuple[str, str], list[tuple[int, tuple[float, float, float], int, float, float]]] = {}
        for row in range(count):
            if not bool(enabled_np[row]):
                continue
            body = int(body_np[row])
            particle = int(particle_np[row])
            if body < 0 or body >= self.model.body_count:
                raise IndexError(f"ADMM body-particle attachment row {row} has body index {body} out of range")
            if particle < 0 or particle >= self.model.particle_count:
                raise IndexError(f"ADMM body-particle attachment row {row} has particle index {particle} out of range")
            stiffness = float(stiffness_np[row])
            if stiffness < 0.0:
                raise ValueError(f"ADMM body-particle attachment row {row} has negative stiffness")
            damping = float(damping_np[row])
            if damping < 0.0:
                raise ValueError(f"ADMM body-particle attachment row {row} has negative damping")

            body_entry = self._entry_name_for_body(body)
            particle_entry = self._entry_name_for_particle(particle)
            if body_entry is None or particle_entry is None or body_entry == particle_entry:
                continue

            point = (float(point_np[row][0]), float(point_np[row][1]), float(point_np[row][2]))
            grouped.setdefault((body_entry, particle_entry), []).append((body, point, particle, stiffness, damping))

        device = self.model.device
        for (body_entry, particle_entry), items in grouped.items():
            self._require_effective_mass(body_entry, CouplingEndpointKind.BODY)
            self._require_effective_mass(particle_entry, CouplingEndpointKind.PARTICLE)
            body_mass_np = self._body_effective_mass_np(body_entry)
            particle_mass_np = self._particle_effective_mass_np(particle_entry)
            body_ids = [self._body_local_id(body_entry, item[0]) for item in items]
            points = [wp.vec3(*item[1]) for item in items]
            particle_ids = [item[2] for item in items]
            kappa = [item[3] for item in items]
            damping = [item[4] for item in items]
            W = []
            for body, _, particle, _, _ in items:
                m_body = float(body_mass_np[body]) if len(body_mass_np) > body else 0.0
                m_particle = float(particle_mass_np[particle]) if len(particle_mass_np) > particle else 0.0
                W.append(self._interface_weight(m_body, m_particle))

            n = len(items)
            self._admm_rp_groups.append(
                _AdmmRigidParticleAttachmentGroup(
                    body_entry_name=body_entry,
                    particle_entry_name=particle_entry,
                    body_ids=wp.array(body_ids, dtype=int, device=device),
                    point_body=wp.array(points, dtype=wp.vec3, device=device),
                    particle_ids=wp.array(particle_ids, dtype=int, device=device),
                    kappa=wp.array(kappa, dtype=float, device=device),
                    damping=wp.array(damping, dtype=float, device=device),
                    W=wp.array(W, dtype=float, device=device),
                    u=wp.zeros(n, dtype=wp.vec3, device=device),
                    lambda_=wp.zeros(n, dtype=wp.vec3, device=device),
                    Jv=wp.zeros(n, dtype=wp.vec3, device=device),
                    u_target=wp.zeros(n, dtype=wp.vec3, device=device),
                )
            )

    def _step_coupled(
        self,
        state_in: State,
        state_out: State,
        control: Control | None,
        contacts: Contacts | None,
        dt: float,
    ) -> None:
        """Run ADMM iterations over all sub-solvers."""
        del state_out
        coupling = self._coupling
        iters = max(1, int(coupling.iterations))
        self._refresh_collision_contact_groups(state_in)

        for name, entry in self._entries.items():
            buf = self._admm_buffers[name]
            if buf.body_q_n is not None:
                wp.copy(buf.body_q_n, entry.state_0.body_q)
                wp.copy(buf.body_qd_n, entry.state_0.body_qd)
                wp.copy(buf.body_qd_k, entry.state_0.body_qd)
            if buf.particle_q_n is not None:
                wp.copy(buf.particle_q_n, entry.state_0.particle_q)
                wp.copy(buf.particle_qd_n, entry.state_0.particle_qd)
                wp.copy(buf.particle_qd_k, entry.state_0.particle_qd)
            if buf.joint_q_n is not None:
                wp.copy(buf.joint_q_n, entry.state_0.joint_q)
                wp.copy(buf.joint_qd_n, entry.state_0.joint_qd)
                wp.copy(buf.joint_qd_k, entry.state_0.joint_qd)

        self._admm_begin_step(dt)

        for k in range(iters):
            for name, entry in self._entries.items():
                self._prepare_admm_iteration_state(
                    entry,
                    self._admm_buffers[name],
                    state_in,
                    dt,
                    iteration_restart=k > 0,
                )

            self._accumulate_admm_forces(k, dt)

            for name, entry in self._entries.items():
                self._apply_admm_force_inputs(entry, self._admm_buffers[name], dt)

            for entry in self._entries.values():
                self._step_entry(entry, control, contacts, dt)

            for name, entry in self._entries.items():
                buf = self._admm_buffers[name]
                if buf.body_qd_k is not None:
                    wp.copy(buf.body_qd_k, entry.state_1.body_qd)
                if buf.particle_qd_k is not None:
                    wp.copy(buf.particle_qd_k, entry.state_1.particle_qd)
                if buf.joint_qd_k is not None:
                    wp.copy(buf.joint_qd_k, entry.state_1.joint_qd)

            self._update_admm_dual(k, dt)

    def _refresh_collision_contact_groups(self, state_in: State) -> None:
        if (
            not self._admm_dynamic_rr_contact_groups
            and not self._admm_dynamic_rp_contact_groups
            and not self._admm_dynamic_pp_contact_groups
        ):
            return

        if self._admm_dynamic_rr_contact_groups or self._admm_dynamic_rp_contact_groups:
            self._admm_collision_pipeline.collide(state_in, self._admm_internal_contacts)

        for group in self._admm_dynamic_rr_contact_groups:
            if group.count == 0:
                continue
            entry_a = self._entries[group.body_entry_name_a]
            entry_b = self._entries[group.body_entry_name_b]
            buf_a = self._admm_buffers[group.body_entry_name_a]
            buf_b = self._admm_buffers[group.body_entry_name_b]
            wp.launch(
                contact_rr_snapshot_kernel,
                dim=group.count,
                inputs=[
                    group.body_ids_a,
                    group.body_ids_b,
                    group.shape_ids_a,
                    group.shape_ids_b,
                    group.point_ids,
                    group.active,
                    group.u,
                    group.lambda_,
                ],
                outputs=[
                    group.prev_body_ids_a,
                    group.prev_body_ids_b,
                    group.prev_shape_ids_a,
                    group.prev_shape_ids_b,
                    group.prev_point_ids,
                    group.prev_active,
                    group.prev_u,
                    group.prev_lambda,
                ],
                device=self.model.device,
            )
            wp.launch(
                contact_rr_reset_kernel,
                dim=group.count,
                inputs=[
                    group.active_count,
                    group.body_ids_a,
                    group.point_a,
                    group.body_ids_b,
                    group.point_b,
                    group.shape_ids_a,
                    group.shape_ids_b,
                    group.point_ids,
                    group.active,
                    group.normal,
                    group.contact_distance,
                    group.W,
                    group.friction,
                    group.u,
                    group.lambda_,
                    group.Jv,
                    group.u_min,
                ],
                device=self.model.device,
            )
            wp.launch(
                contact_rr_fill_from_rigid_contacts_kernel,
                dim=self._admm_internal_contacts.rigid_contact_max,
                inputs=[
                    self._admm_internal_contacts.rigid_contact_count,
                    self._admm_internal_contacts.rigid_contact_shape0,
                    self._admm_internal_contacts.rigid_contact_shape1,
                    self._admm_internal_contacts.rigid_contact_point0,
                    self._admm_internal_contacts.rigid_contact_point1,
                    self._admm_internal_contacts.rigid_contact_normal,
                    self._admm_internal_contacts.rigid_contact_margin0,
                    self._admm_internal_contacts.rigid_contact_margin1,
                    self._admm_internal_contacts.rigid_contact_point_id,
                    self.model.shape_body,
                    group.body_mask_a,
                    group.body_mask_b,
                    group.shape_mask_a,
                    group.shape_mask_b,
                    entry_a.body_global_to_local,
                    entry_b.body_global_to_local,
                    buf_a.body_effective_mass,
                    buf_b.body_effective_mass,
                    self.model.shape_material_mu,
                    float(group.contact_distance_value),
                    1 if group.use_contact_margins else 0,
                    int(group.count),
                    group.active_count,
                    group.active_count_max,
                    group.prev_shape_ids_a,
                    group.prev_shape_ids_b,
                    group.prev_point_ids,
                    group.prev_active,
                    group.prev_u,
                    group.prev_lambda,
                ],
                outputs=[
                    group.body_ids_a,
                    group.point_a,
                    group.body_ids_b,
                    group.point_b,
                    group.shape_ids_a,
                    group.shape_ids_b,
                    group.point_ids,
                    group.active,
                    group.normal,
                    group.contact_distance,
                    group.W,
                    group.friction,
                    group.u,
                    group.lambda_,
                ],
                device=self.model.device,
            )

        if self._admm_dynamic_rp_contact_groups:
            for group in self._admm_dynamic_rp_contact_groups:
                if group.count == 0:
                    continue
                body_entry = self._entries[group.body_entry_name]
                body_buf = self._admm_buffers[group.body_entry_name]
                particle_buf = self._admm_buffers[group.particle_entry_name]
                wp.launch(
                    contact_rp_snapshot_kernel,
                    dim=group.count,
                    inputs=[
                        group.body_ids,
                        group.particle_ids,
                        group.shape_ids,
                        group.active,
                        group.u,
                        group.lambda_,
                    ],
                    outputs=[
                        group.prev_body_ids,
                        group.prev_particle_ids,
                        group.prev_shape_ids,
                        group.prev_active,
                        group.prev_u,
                        group.prev_lambda,
                    ],
                    device=self.model.device,
                )
                wp.launch(
                    contact_rp_reset_kernel,
                    dim=group.count,
                    inputs=[
                        group.active_count,
                        group.body_ids,
                        group.point_body,
                        group.particle_ids,
                        group.shape_ids,
                        group.active,
                        group.normal,
                        group.body_sign,
                        group.contact_distance,
                        group.W,
                        group.friction,
                        group.u,
                        group.lambda_,
                        group.Jv,
                        group.u_min,
                    ],
                    device=self.model.device,
                )
                wp.launch(
                    contact_rp_fill_from_soft_contacts_kernel,
                    dim=self._admm_internal_contacts.soft_contact_max,
                    inputs=[
                        self._admm_internal_contacts.soft_contact_count,
                        self._admm_internal_contacts.soft_contact_particle,
                        self._admm_internal_contacts.soft_contact_shape,
                        self._admm_internal_contacts.soft_contact_body_pos,
                        self._admm_internal_contacts.soft_contact_normal,
                        self.model.shape_body,
                        group.particle_mask,
                        group.body_mask,
                        group.shape_mask,
                        body_entry.body_global_to_local,
                        self.model.particle_radius,
                        body_buf.body_effective_mass,
                        particle_buf.particle_effective_mass,
                        self.model.shape_material_mu,
                        float(self.model.particle_mu),
                        float(group.contact_distance_value),
                        1 if group.use_particle_radius else 0,
                        int(group.count),
                        group.active_count,
                        group.active_count_max,
                        group.prev_particle_ids,
                        group.prev_shape_ids,
                        group.prev_active,
                        group.prev_u,
                        group.prev_lambda,
                    ],
                    outputs=[
                        group.body_ids,
                        group.point_body,
                        group.particle_ids,
                        group.shape_ids,
                        group.active,
                        group.normal,
                        group.body_sign,
                        group.contact_distance,
                        group.W,
                        group.friction,
                        group.u,
                        group.lambda_,
                    ],
                    device=self.model.device,
                )

        if self._admm_dynamic_pp_contact_groups:
            self._admm_particle_contact_grid.build(
                state_in.particle_q,
                radius=self._admm_particle_contact_query_radius,
            )

        for group in self._admm_dynamic_pp_contact_groups:
            if group.count == 0:
                continue
            contact_stream = group.contact_stream
            wp.launch(
                contact_pp_snapshot_kernel,
                dim=group.count,
                inputs=[
                    group.particle_ids_a,
                    group.particle_ids_b,
                    group.active,
                    group.u,
                    group.lambda_,
                ],
                outputs=[
                    group.prev_particle_ids_a,
                    group.prev_particle_ids_b,
                    group.prev_active,
                    group.prev_u,
                    group.prev_lambda,
                ],
                device=self.model.device,
            )
            wp.launch(
                contact_pp_reset_kernel,
                dim=group.count,
                inputs=[
                    group.active_count,
                    group.particle_ids_a,
                    group.particle_ids_b,
                    group.active,
                    group.normal,
                    group.contact_distance,
                    group.W,
                    group.friction,
                    group.u,
                    group.lambda_,
                    group.Jv,
                    group.u_min,
                ],
                device=self.model.device,
            )
            wp.launch(
                admm_contact_stream_reset_count_kernel,
                dim=1,
                inputs=[contact_stream.count],
                device=self.model.device,
            )
            wp.launch(
                particle_particle_contacts_hashgrid_kernel,
                dim=self.model.particle_count,
                inputs=[
                    self._admm_particle_contact_grid.id,
                    state_in.particle_q,
                    self.model.particle_radius,
                    self.model.particle_flags,
                    self.model.particle_world,
                    group.particle_mask_a,
                    group.particle_mask_b,
                    float(group.contact_distance_value),
                    1 if group.use_radius_sum else 0,
                    float(group.detection_margin),
                    float(group.query_radius),
                    int(contact_stream.capacity),
                    contact_stream.count,
                    contact_stream.count_max,
                ],
                outputs=[
                    contact_stream.particle_a,
                    contact_stream.particle_b,
                    contact_stream.normal,
                    contact_stream.distance,
                    contact_stream.source_id,
                ],
                device=self.model.device,
            )
            wp.launch(
                contact_pp_fill_from_particle_contacts_kernel,
                dim=contact_stream.capacity,
                inputs=[
                    contact_stream.count,
                    contact_stream.particle_a,
                    contact_stream.particle_b,
                    contact_stream.normal,
                    contact_stream.distance,
                    self._admm_buffers[group.particle_entry_name_a].particle_effective_mass,
                    self._admm_buffers[group.particle_entry_name_b].particle_effective_mass,
                    float(self.model.particle_mu),
                    int(group.count),
                    group.active_count,
                    group.active_count_max,
                    group.prev_particle_ids_a,
                    group.prev_particle_ids_b,
                    group.prev_active,
                    group.prev_u,
                    group.prev_lambda,
                ],
                outputs=[
                    group.particle_ids_a,
                    group.particle_ids_b,
                    group.active,
                    group.normal,
                    group.contact_distance,
                    group.W,
                    group.friction,
                    group.u,
                    group.lambda_,
                ],
                device=self.model.device,
            )

    def _make_int_mask_array(self, count: int, indices: set[int]) -> wp.array:
        device = self.model.device
        return wp.array([1 if i in indices else 0 for i in range(count)], dtype=int, device=device)

    def _particle_contact_candidates(self, owner: str, particles: Sequence[int] | None) -> list[int]:
        owner_particles = self._entry_particle_sets[owner]
        if particles is None:
            return sorted(owner_particles)
        return list(dict.fromkeys(int(particle) for particle in particles))

    def _shape_contact_candidates(self, owner: str, shapes: Sequence[int] | None) -> list[int]:
        owner_bodies = self._entry_body_sets[owner]
        shape_body = self.model.shape_body.numpy() if self.model.shape_body is not None else []
        if shapes is None:
            return [shape for shape in range(self.model.shape_count) if int(shape_body[shape]) in owner_bodies]
        return list(dict.fromkeys(int(shape) for shape in shapes))

    def _build_collision_rigid_rigid_contact_groups(self) -> list[_AdmmRigidRigidContactGroup]:
        device = self.model.device
        groups = []

        for spec in self._admm_rigid_rigid_contact_specs:
            shapes_a = self._shape_contact_candidates(spec.owner_a, spec.shapes_a)
            shapes_b = self._shape_contact_candidates(spec.owner_b, spec.shapes_b)
            # Primitive pairs may emit a small manifold rather than one row.
            capacity = 8 * len(shapes_a) * len(shapes_b)
            if capacity == 0:
                continue
            self._require_effective_mass(spec.owner_a, CouplingEndpointKind.BODY)
            self._require_effective_mass(spec.owner_b, CouplingEndpointKind.BODY)

            groups.append(
                _AdmmRigidRigidContactGroup(
                    body_entry_name_a=spec.owner_a,
                    body_entry_name_b=spec.owner_b,
                    body_ids_a=wp.zeros(capacity, dtype=int, device=device),
                    point_a=wp.zeros(capacity, dtype=wp.vec3, device=device),
                    body_ids_b=wp.zeros(capacity, dtype=int, device=device),
                    point_b=wp.zeros(capacity, dtype=wp.vec3, device=device),
                    normal=wp.zeros(capacity, dtype=wp.vec3, device=device),
                    contact_distance=wp.zeros(capacity, dtype=float, device=device),
                    W=wp.zeros(capacity, dtype=float, device=device),
                    friction=wp.zeros(capacity, dtype=float, device=device),
                    u=wp.zeros(capacity, dtype=wp.vec3, device=device),
                    lambda_=wp.zeros(capacity, dtype=wp.vec3, device=device),
                    Jv=wp.zeros(capacity, dtype=wp.vec3, device=device),
                    u_min=wp.zeros(capacity, dtype=float, device=device),
                    capacity=capacity,
                    active_count=wp.zeros(1, dtype=int, device=device),
                    active_count_max=wp.zeros(1, dtype=int, device=device),
                    active=wp.zeros(capacity, dtype=int, device=device),
                    shape_ids_a=wp.full(capacity, -1, dtype=int, device=device),
                    shape_ids_b=wp.full(capacity, -1, dtype=int, device=device),
                    point_ids=wp.full(capacity, -1, dtype=int, device=device),
                    body_mask_a=self._make_int_mask_array(self.model.body_count, self._entry_body_sets[spec.owner_a]),
                    body_mask_b=self._make_int_mask_array(self.model.body_count, self._entry_body_sets[spec.owner_b]),
                    shape_mask_a=self._make_int_mask_array(self.model.shape_count, set(shapes_a)),
                    shape_mask_b=self._make_int_mask_array(self.model.shape_count, set(shapes_b)),
                    contact_distance_value=0.0 if spec.contact_distance is None else float(spec.contact_distance),
                    use_contact_margins=spec.contact_distance is None,
                    prev_body_ids_a=wp.zeros(capacity, dtype=int, device=device),
                    prev_body_ids_b=wp.zeros(capacity, dtype=int, device=device),
                    prev_shape_ids_a=wp.full(capacity, -1, dtype=int, device=device),
                    prev_shape_ids_b=wp.full(capacity, -1, dtype=int, device=device),
                    prev_point_ids=wp.full(capacity, -1, dtype=int, device=device),
                    prev_active=wp.zeros(capacity, dtype=int, device=device),
                    prev_u=wp.zeros(capacity, dtype=wp.vec3, device=device),
                    prev_lambda=wp.zeros(capacity, dtype=wp.vec3, device=device),
                )
            )

        return groups

    def _build_collision_rigid_particle_contact_groups(self) -> list[_AdmmRigidParticleContactGroup]:
        device = self.model.device
        groups = []
        shape_body = self.model.shape_body.numpy() if self.model.shape_body is not None else []

        for spec_idx, spec in enumerate(self._admm_rigid_particle_contact_specs):
            particle_candidates = sorted(self._entry_particle_sets[spec.particle_owner])
            body_candidates = set(self._entry_body_sets[spec.body_owner])
            shape_filter = self._admm_rigid_particle_shape_filters.get(spec_idx)
            if shape_filter is None:
                shape_candidates = [
                    shape for shape in range(self.model.shape_count) if int(shape_body[shape]) in body_candidates
                ]
            else:
                shape_candidates = sorted(shape_filter)

            capacity = len(particle_candidates) * len(shape_candidates)
            if capacity == 0:
                continue
            self._require_effective_mass(spec.body_owner, CouplingEndpointKind.BODY)
            self._require_effective_mass(spec.particle_owner, CouplingEndpointKind.PARTICLE)

            groups.append(
                _AdmmRigidParticleContactGroup(
                    body_entry_name=spec.body_owner,
                    particle_entry_name=spec.particle_owner,
                    body_ids=wp.zeros(capacity, dtype=int, device=device),
                    point_body=wp.zeros(capacity, dtype=wp.vec3, device=device),
                    particle_ids=wp.zeros(capacity, dtype=int, device=device),
                    normal=wp.zeros(capacity, dtype=wp.vec3, device=device),
                    body_sign=wp.full(capacity, -1, dtype=int, device=device),
                    contact_distance=wp.zeros(capacity, dtype=float, device=device),
                    W=wp.zeros(capacity, dtype=float, device=device),
                    friction=wp.zeros(capacity, dtype=float, device=device),
                    u=wp.zeros(capacity, dtype=wp.vec3, device=device),
                    lambda_=wp.zeros(capacity, dtype=wp.vec3, device=device),
                    Jv=wp.zeros(capacity, dtype=wp.vec3, device=device),
                    u_min=wp.zeros(capacity, dtype=float, device=device),
                    capacity=capacity,
                    active_count=wp.zeros(1, dtype=int, device=device),
                    active_count_max=wp.zeros(1, dtype=int, device=device),
                    active=wp.zeros(capacity, dtype=int, device=device),
                    shape_ids=wp.full(capacity, -1, dtype=int, device=device),
                    particle_mask=self._make_int_mask_array(self.model.particle_count, set(particle_candidates)),
                    body_mask=self._make_int_mask_array(self.model.body_count, body_candidates),
                    shape_mask=self._make_int_mask_array(self.model.shape_count, set(shape_candidates)),
                    contact_distance_value=0.0 if spec.contact_distance is None else float(spec.contact_distance),
                    use_particle_radius=spec.contact_distance is None,
                    prev_body_ids=wp.zeros(capacity, dtype=int, device=device),
                    prev_particle_ids=wp.zeros(capacity, dtype=int, device=device),
                    prev_shape_ids=wp.full(capacity, -1, dtype=int, device=device),
                    prev_active=wp.zeros(capacity, dtype=int, device=device),
                    prev_u=wp.zeros(capacity, dtype=wp.vec3, device=device),
                    prev_lambda=wp.zeros(capacity, dtype=wp.vec3, device=device),
                )
            )

        return groups

    def _build_collision_particle_particle_contact_groups(self) -> list[_AdmmParticleParticleContactGroup]:
        device = self.model.device
        groups = []

        for spec in self._admm_particle_particle_contact_specs:
            particles_a = self._particle_contact_candidates(spec.owner_a, spec.particles_a)
            particles_b = self._particle_contact_candidates(spec.owner_b, spec.particles_b)
            capacity = len(particles_a) * len(particles_b)
            if capacity == 0:
                continue
            self._require_effective_mass(spec.owner_a, CouplingEndpointKind.PARTICLE)
            self._require_effective_mass(spec.owner_b, CouplingEndpointKind.PARTICLE)

            use_radius_sum = spec.contact_distance is None
            detection_margin = (
                _DEFAULT_DETECTION_MARGIN if spec.detection_margin is None else float(spec.detection_margin)
            )
            query_radius = (
                2.0 * float(self.model.particle_max_radius) + detection_margin
                if use_radius_sum
                else float(spec.contact_distance) + detection_margin
            )
            contact_stream = AdmmContactStream.allocate(
                capacity=capacity,
                device=device,
                contact_type=AdmmContactType.PARTICLE_PARTICLE,
            )

            groups.append(
                _AdmmParticleParticleContactGroup(
                    particle_entry_name_a=spec.owner_a,
                    particle_entry_name_b=spec.owner_b,
                    particle_ids_a=wp.zeros(capacity, dtype=int, device=device),
                    particle_ids_b=wp.zeros(capacity, dtype=int, device=device),
                    normal=wp.zeros(capacity, dtype=wp.vec3, device=device),
                    contact_distance=wp.zeros(capacity, dtype=float, device=device),
                    W=wp.zeros(capacity, dtype=float, device=device),
                    friction=wp.zeros(capacity, dtype=float, device=device),
                    u=wp.zeros(capacity, dtype=wp.vec3, device=device),
                    lambda_=wp.zeros(capacity, dtype=wp.vec3, device=device),
                    Jv=wp.zeros(capacity, dtype=wp.vec3, device=device),
                    u_min=wp.zeros(capacity, dtype=float, device=device),
                    capacity=capacity,
                    active_count=wp.zeros(1, dtype=int, device=device),
                    active_count_max=wp.zeros(1, dtype=int, device=device),
                    active=wp.zeros(capacity, dtype=int, device=device),
                    contact_stream=contact_stream,
                    particle_mask_a=self._make_int_mask_array(self.model.particle_count, set(particles_a)),
                    particle_mask_b=self._make_int_mask_array(self.model.particle_count, set(particles_b)),
                    contact_distance_value=0.0 if spec.contact_distance is None else float(spec.contact_distance),
                    use_radius_sum=use_radius_sum,
                    detection_margin=detection_margin,
                    query_radius=query_radius,
                    prev_particle_ids_a=wp.zeros(capacity, dtype=int, device=device),
                    prev_particle_ids_b=wp.zeros(capacity, dtype=int, device=device),
                    prev_active=wp.zeros(capacity, dtype=int, device=device),
                    prev_u=wp.zeros(capacity, dtype=wp.vec3, device=device),
                    prev_lambda=wp.zeros(capacity, dtype=wp.vec3, device=device),
                )
            )

        return groups

    def _admm_begin_step(self, dt: float) -> None:
        coupling = self._coupling
        for group in self._admm_rr_groups:
            if group.count == 0:
                continue
            if coupling.baumgarte <= 0.0:
                group.u_target.zero_()
                continue
            entry_a = self._entries[group.body_entry_name_a]
            entry_b = self._entries[group.body_entry_name_b]
            wp.launch(
                attach_rr_compute_u_target_kernel,
                dim=group.count,
                inputs=[
                    group.body_ids_a,
                    group.point_a,
                    group.body_ids_b,
                    group.point_b,
                    entry_a.state_0.body_q,
                    entry_b.state_0.body_q,
                    float(coupling.baumgarte),
                    float(dt),
                ],
                outputs=[group.u_target],
                device=self.model.device,
            )
        for group in self._admm_rr_angular_groups:
            if group.count == 0:
                continue
            if coupling.baumgarte <= 0.0:
                group.u_target.zero_()
                continue
            entry_a = self._entries[group.body_entry_name_a]
            entry_b = self._entries[group.body_entry_name_b]
            wp.launch(
                attach_rr_angular_compute_u_target_kernel,
                dim=group.count,
                inputs=[
                    group.body_ids_a,
                    group.frame_a,
                    group.body_ids_b,
                    group.frame_b,
                    entry_a.state_0.body_q,
                    entry_b.state_0.body_q,
                    float(coupling.baumgarte),
                    float(dt),
                ],
                outputs=[group.u_target],
                device=self.model.device,
            )
        for group in self._admm_rr_revolute_angular_groups:
            if group.count == 0:
                continue
            if coupling.baumgarte <= 0.0:
                group.u_target.zero_()
                continue
            entry_a = self._entries[group.body_entry_name_a]
            entry_b = self._entries[group.body_entry_name_b]
            wp.launch(
                attach_rr_revolute_angular_local_compute_u_target_kernel,
                dim=group.count,
                inputs=[
                    group.body_ids_a,
                    group.frame_a,
                    group.body_ids_b,
                    group.frame_b,
                    entry_a.state_0.body_q,
                    entry_b.state_0.body_q,
                    float(coupling.baumgarte),
                    float(dt),
                ],
                outputs=[group.u_target],
                device=self.model.device,
            )
        for group in self._admm_rp_groups:
            if group.count == 0:
                continue
            if coupling.baumgarte <= 0.0:
                group.u_target.zero_()
                continue
            body_entry = self._entries[group.body_entry_name]
            particle_entry = self._entries[group.particle_entry_name]
            wp.launch(
                attach_rp_compute_u_target_kernel,
                dim=group.count,
                inputs=[
                    group.body_ids,
                    group.point_body,
                    group.particle_ids,
                    body_entry.state_0.body_q,
                    particle_entry.state_0.particle_q,
                    float(coupling.baumgarte),
                    float(dt),
                ],
                outputs=[group.u_target],
                device=self.model.device,
            )
        for group in self._admm_rr_contact_groups:
            if group.count == 0:
                continue
            entry_a = self._entries[group.body_entry_name_a]
            entry_b = self._entries[group.body_entry_name_b]
            wp.launch(
                contact_rr_compute_u_min_kernel,
                dim=group.count,
                inputs=[
                    group.body_ids_a,
                    group.point_a,
                    group.body_ids_b,
                    group.point_b,
                    group.normal,
                    group.contact_distance,
                    entry_a.state_0.body_q,
                    entry_b.state_0.body_q,
                    float(coupling.baumgarte),
                    float(dt),
                ],
                outputs=[group.u_min],
                device=self.model.device,
            )
        for group in self._admm_rp_contact_groups:
            if group.count == 0:
                continue
            body_entry = self._entries[group.body_entry_name]
            particle_entry = self._entries[group.particle_entry_name]
            wp.launch(
                contact_rp_compute_u_min_kernel,
                dim=group.count,
                inputs=[
                    group.body_ids,
                    group.point_body,
                    group.particle_ids,
                    group.normal,
                    group.body_sign,
                    group.contact_distance,
                    body_entry.state_0.body_q,
                    particle_entry.state_0.particle_q,
                    float(coupling.baumgarte),
                    float(dt),
                ],
                outputs=[group.u_min],
                device=self.model.device,
            )
        for group in self._admm_pp_contact_groups:
            if group.count == 0:
                continue
            entry_a = self._entries[group.particle_entry_name_a]
            entry_b = self._entries[group.particle_entry_name_b]
            wp.launch(
                contact_pp_compute_u_min_kernel,
                dim=group.count,
                inputs=[
                    group.particle_ids_a,
                    group.particle_ids_b,
                    group.normal,
                    group.contact_distance,
                    entry_a.state_0.particle_q,
                    entry_b.state_0.particle_q,
                    float(coupling.baumgarte),
                    float(dt),
                ],
                outputs=[group.u_min],
                device=self.model.device,
            )

    def _apply_admm_velocity_proximal_shift(
        self,
        entry: SolverEntry,
        buf: _AdmmBuffers,
        gamma: float,
        dt: float,
    ) -> None:
        device = self.model.device
        flags = CouplingInputStateFlags(0)
        if buf.body_qd_n is not None:
            wp.launch(
                velocity_proximal_shift_body_kernel,
                dim=buf.body_qd_n.shape[0],
                inputs=[buf.body_qd_n, buf.body_qd_k, gamma, entry.state_0.body_qd],
                device=device,
            )
            flags |= CouplingInputStateFlags.BODY_QD
        if buf.particle_qd_n is not None:
            wp.launch(
                velocity_proximal_shift_particle_kernel,
                dim=buf.particle_qd_n.shape[0],
                inputs=[buf.particle_qd_n, buf.particle_qd_k, gamma, entry.state_0.particle_qd],
                device=device,
            )
            flags |= CouplingInputStateFlags.PARTICLE_QD
        if buf.joint_qd_n is not None and buf.joint_qd_n.shape[0] > 0:
            wp.launch(
                velocity_proximal_shift_joint_kernel,
                dim=buf.joint_qd_n.shape[0],
                inputs=[buf.joint_qd_n, buf.joint_qd_k, gamma, entry.state_0.joint_qd],
                device=device,
            )
            flags |= CouplingInputStateFlags.JOINT_QD
        self._notify_input_state_update(entry, flags, dt=dt)

    def _prepare_admm_iteration_state(
        self,
        entry: SolverEntry,
        buf: _AdmmBuffers,
        state_in: State,
        dt: float,
        *,
        iteration_restart: bool = False,
    ) -> None:
        gamma = float(self._coupling.gamma)
        apply_proximal = gamma > 0.0
        flags = CouplingInputStateFlags(0)

        if buf.body_q_n is not None:
            wp.copy(entry.state_0.body_q, buf.body_q_n)
            wp.copy(entry.state_0.body_qd, buf.body_qd_n)
            flags |= CouplingInputStateFlags.BODY

        if buf.particle_q_n is not None:
            wp.copy(entry.state_0.particle_q, buf.particle_q_n)
            wp.copy(entry.state_0.particle_qd, buf.particle_qd_n)
            flags |= CouplingInputStateFlags.PARTICLE

        if buf.joint_q_n is not None:
            wp.copy(entry.state_0.joint_q, buf.joint_q_n)
            wp.copy(entry.state_0.joint_qd, buf.joint_qd_n)
            flags |= CouplingInputStateFlags.JOINT

        self._notify_input_state_update(entry, flags, dt=dt, restart=bool(iteration_restart) and bool(flags))

        if apply_proximal:
            self._apply_admm_velocity_proximal_shift(entry, buf, gamma, dt)

        if buf.body_f is not None:
            if state_in.body_f is not None:
                _copy_prefix(buf.body_f, state_in.body_f, "body_f")
            else:
                buf.body_f.zero_()
        if buf.particle_f is not None:
            if state_in.particle_f is not None:
                _copy_prefix(buf.particle_f, state_in.particle_f, "particle_f")
            else:
                buf.particle_f.zero_()

    def _apply_admm_force_inputs(self, entry: SolverEntry, buf: _AdmmBuffers, dt: float) -> None:
        if entry.body_indices.shape[0] > 0:
            self._set_body_force_input(entry, buf.body_f, dt=dt)
        if entry.particle_indices.shape[0] > 0:
            self._set_particle_force_input(entry, buf.particle_f, dt=dt)

    def _accumulate_admm_forces(self, iteration_k: int, dt: float) -> None:
        del iteration_k
        coupling = self._coupling
        for group in self._admm_rr_groups:
            if group.count == 0:
                continue
            entry_a = self._entries[group.body_entry_name_a]
            entry_b = self._entries[group.body_entry_name_b]
            buf_a = self._admm_buffers[group.body_entry_name_a]
            buf_b = self._admm_buffers[group.body_entry_name_b]
            wp.launch(
                contact_rr_compute_Jv_kernel,
                dim=group.count,
                inputs=[
                    group.body_ids_a,
                    group.point_a,
                    group.body_ids_b,
                    group.point_b,
                    entry_a.state_0.body_q,
                    entry_a.view.body_com,
                    buf_a.body_qd_k,
                    entry_b.state_0.body_q,
                    entry_b.view.body_com,
                    buf_b.body_qd_k,
                ],
                outputs=[group.Jv],
                device=self.model.device,
            )
            wp.launch(
                contact_rr_accumulate_forces_kernel,
                dim=group.count,
                inputs=[
                    group.body_ids_a,
                    group.point_a,
                    group.body_ids_b,
                    group.point_b,
                    entry_a.state_0.body_q,
                    entry_a.view.body_com,
                    entry_b.state_0.body_q,
                    entry_b.view.body_com,
                    float(coupling.rho),
                    group.W,
                    group.lambda_,
                    group.u,
                    group.Jv,
                ],
                outputs=[buf_a.body_f, buf_b.body_f],
                device=self.model.device,
            )
        for group in self._admm_rr_angular_groups:
            if group.count == 0:
                continue
            buf_a = self._admm_buffers[group.body_entry_name_a]
            buf_b = self._admm_buffers[group.body_entry_name_b]
            wp.launch(
                attach_rr_angular_compute_Jv_kernel,
                dim=group.count,
                inputs=[
                    group.body_ids_a,
                    group.body_ids_b,
                    buf_a.body_qd_k,
                    buf_b.body_qd_k,
                ],
                outputs=[group.Jv],
                device=self.model.device,
            )
            wp.launch(
                attach_rr_angular_accumulate_forces_kernel,
                dim=group.count,
                inputs=[
                    group.body_ids_a,
                    group.body_ids_b,
                    float(coupling.rho),
                    group.W,
                    group.lambda_,
                    group.u,
                    group.Jv,
                ],
                outputs=[buf_a.body_f, buf_b.body_f],
                device=self.model.device,
            )
        for group in self._admm_rr_revolute_angular_groups:
            if group.count == 0:
                continue
            entry_a = self._entries[group.body_entry_name_a]
            buf_a = self._admm_buffers[group.body_entry_name_a]
            buf_b = self._admm_buffers[group.body_entry_name_b]
            wp.launch(
                attach_rr_revolute_angular_local_compute_Jv_kernel,
                dim=group.count,
                inputs=[
                    group.body_ids_a,
                    group.frame_a,
                    group.body_ids_b,
                    entry_a.state_0.body_q,
                    buf_a.body_qd_k,
                    buf_b.body_qd_k,
                ],
                outputs=[group.Jv],
                device=self.model.device,
            )
            wp.launch(
                attach_rr_revolute_angular_local_accumulate_forces_kernel,
                dim=group.count,
                inputs=[
                    group.body_ids_a,
                    group.frame_a,
                    group.body_ids_b,
                    entry_a.state_0.body_q,
                    float(coupling.rho),
                    group.W,
                    group.lambda_,
                    group.u,
                    group.Jv,
                ],
                outputs=[buf_a.body_f, buf_b.body_f],
                device=self.model.device,
            )
        for group in self._admm_rr_angular_friction_groups:
            if group.count == 0:
                continue
            entry_a = self._entries[group.body_entry_name_a]
            buf_a = self._admm_buffers[group.body_entry_name_a]
            buf_b = self._admm_buffers[group.body_entry_name_b]
            wp.launch(
                attach_rr_angular_local_compute_Jv_kernel,
                dim=group.count,
                inputs=[
                    group.body_ids_a,
                    group.frame_a,
                    group.body_ids_b,
                    entry_a.state_0.body_q,
                    buf_a.body_qd_k,
                    buf_b.body_qd_k,
                ],
                outputs=[group.Jv],
                device=self.model.device,
            )
            wp.launch(
                attach_rr_angular_local_accumulate_forces_kernel,
                dim=group.count,
                inputs=[
                    group.body_ids_a,
                    group.frame_a,
                    group.body_ids_b,
                    entry_a.state_0.body_q,
                    float(coupling.rho),
                    group.W,
                    group.lambda_,
                    group.u,
                    group.Jv,
                ],
                outputs=[buf_a.body_f, buf_b.body_f],
                device=self.model.device,
            )
        for group in self._admm_rp_groups:
            if group.count == 0:
                continue
            body_entry = self._entries[group.body_entry_name]
            body_buf = self._admm_buffers[group.body_entry_name]
            particle_buf = self._admm_buffers[group.particle_entry_name]
            wp.launch(
                attach_rp_compute_Jv_kernel,
                dim=group.count,
                inputs=[
                    group.body_ids,
                    group.point_body,
                    group.particle_ids,
                    body_entry.state_0.body_q,
                    body_entry.view.body_com,
                    body_buf.body_qd_k,
                    particle_buf.particle_qd_k,
                ],
                outputs=[group.Jv],
                device=self.model.device,
            )
            wp.launch(
                attach_rp_accumulate_forces_kernel,
                dim=group.count,
                inputs=[
                    group.body_ids,
                    group.point_body,
                    group.particle_ids,
                    body_entry.state_0.body_q,
                    body_entry.view.body_com,
                    float(coupling.rho),
                    group.W,
                    group.lambda_,
                    group.u,
                    group.Jv,
                ],
                outputs=[body_buf.body_f, particle_buf.particle_f],
                device=self.model.device,
            )
        for group in self._admm_rr_contact_groups:
            if group.count == 0:
                continue
            entry_a = self._entries[group.body_entry_name_a]
            entry_b = self._entries[group.body_entry_name_b]
            buf_a = self._admm_buffers[group.body_entry_name_a]
            buf_b = self._admm_buffers[group.body_entry_name_b]
            wp.launch(
                contact_rr_compute_Jv_kernel,
                dim=group.count,
                inputs=[
                    group.body_ids_a,
                    group.point_a,
                    group.body_ids_b,
                    group.point_b,
                    entry_a.state_0.body_q,
                    entry_a.view.body_com,
                    buf_a.body_qd_k,
                    entry_b.state_0.body_q,
                    entry_b.view.body_com,
                    buf_b.body_qd_k,
                ],
                outputs=[group.Jv],
                device=self.model.device,
            )
            wp.launch(
                contact_rr_accumulate_forces_kernel,
                dim=group.count,
                inputs=[
                    group.body_ids_a,
                    group.point_a,
                    group.body_ids_b,
                    group.point_b,
                    entry_a.state_0.body_q,
                    entry_a.view.body_com,
                    entry_b.state_0.body_q,
                    entry_b.view.body_com,
                    float(coupling.rho),
                    group.W,
                    group.lambda_,
                    group.u,
                    group.Jv,
                ],
                outputs=[buf_a.body_f, buf_b.body_f],
                device=self.model.device,
            )
        for group in self._admm_rp_contact_groups:
            if group.count == 0:
                continue
            body_entry = self._entries[group.body_entry_name]
            body_buf = self._admm_buffers[group.body_entry_name]
            particle_buf = self._admm_buffers[group.particle_entry_name]
            wp.launch(
                contact_rp_compute_Jv_kernel,
                dim=group.count,
                inputs=[
                    group.body_ids,
                    group.point_body,
                    group.particle_ids,
                    group.body_sign,
                    body_entry.state_0.body_q,
                    body_entry.view.body_com,
                    body_buf.body_qd_k,
                    particle_buf.particle_qd_k,
                ],
                outputs=[group.Jv],
                device=self.model.device,
            )
            wp.launch(
                contact_rp_accumulate_forces_kernel,
                dim=group.count,
                inputs=[
                    group.body_ids,
                    group.point_body,
                    group.particle_ids,
                    group.body_sign,
                    body_entry.state_0.body_q,
                    body_entry.view.body_com,
                    float(coupling.rho),
                    group.W,
                    group.lambda_,
                    group.u,
                    group.Jv,
                ],
                outputs=[body_buf.body_f, particle_buf.particle_f],
                device=self.model.device,
            )
        for group in self._admm_pp_contact_groups:
            if group.count == 0:
                continue
            buf_a = self._admm_buffers[group.particle_entry_name_a]
            buf_b = self._admm_buffers[group.particle_entry_name_b]
            wp.launch(
                contact_pp_compute_Jv_kernel,
                dim=group.count,
                inputs=[
                    group.particle_ids_a,
                    group.particle_ids_b,
                    buf_a.particle_qd_k,
                    buf_b.particle_qd_k,
                ],
                outputs=[group.Jv],
                device=self.model.device,
            )
            wp.launch(
                contact_pp_accumulate_forces_kernel,
                dim=group.count,
                inputs=[
                    group.particle_ids_a,
                    group.particle_ids_b,
                    float(coupling.rho),
                    group.W,
                    group.lambda_,
                    group.u,
                    group.Jv,
                ],
                outputs=[buf_a.particle_f, buf_b.particle_f],
                device=self.model.device,
            )
            if group.contact_stream is not None:
                wp.launch(
                    admm_contact_stream_update_normal_force_kernel,
                    dim=group.count,
                    inputs=[
                        group.active_count,
                        float(dt),
                        float(coupling.rho),
                        group.W,
                        group.normal,
                        group.lambda_,
                        group.u,
                        group.Jv,
                    ],
                    outputs=[group.contact_stream.normal_force, group.contact_stream.normal_impulse],
                    device=self.model.device,
                )

    def _update_admm_dual(self, iteration_k: int, dt: float) -> None:
        del iteration_k, dt
        coupling = self._coupling
        for group in self._admm_rr_groups:
            if group.count == 0:
                continue
            entry_a = self._entries[group.body_entry_name_a]
            entry_b = self._entries[group.body_entry_name_b]
            wp.launch(
                contact_rr_compute_Jv_kernel,
                dim=group.count,
                inputs=[
                    group.body_ids_a,
                    group.point_a,
                    group.body_ids_b,
                    group.point_b,
                    entry_a.state_1.body_q,
                    entry_a.view.body_com,
                    entry_a.state_1.body_qd,
                    entry_b.state_1.body_q,
                    entry_b.view.body_com,
                    entry_b.state_1.body_qd,
                ],
                outputs=[group.Jv],
                device=self.model.device,
            )
            wp.launch(
                u_update_quadratic_kernel,
                dim=group.count,
                inputs=[
                    group.kappa,
                    group.damping,
                    group.W,
                    float(coupling.rho),
                    group.lambda_,
                    group.Jv,
                    group.u_target,
                ],
                outputs=[group.u],
                device=self.model.device,
            )
            wp.launch(
                lambda_update_kernel,
                dim=group.count,
                inputs=[float(coupling.rho), group.W, group.u, group.Jv],
                outputs=[group.lambda_],
                device=self.model.device,
            )
        for group in self._admm_rr_angular_groups:
            if group.count == 0:
                continue
            entry_a = self._entries[group.body_entry_name_a]
            entry_b = self._entries[group.body_entry_name_b]
            wp.launch(
                attach_rr_angular_compute_Jv_kernel,
                dim=group.count,
                inputs=[
                    group.body_ids_a,
                    group.body_ids_b,
                    entry_a.state_1.body_qd,
                    entry_b.state_1.body_qd,
                ],
                outputs=[group.Jv],
                device=self.model.device,
            )
            wp.launch(
                u_update_quadratic_kernel,
                dim=group.count,
                inputs=[
                    group.kappa,
                    group.damping,
                    group.W,
                    float(coupling.rho),
                    group.lambda_,
                    group.Jv,
                    group.u_target,
                ],
                outputs=[group.u],
                device=self.model.device,
            )
            wp.launch(
                lambda_update_kernel,
                dim=group.count,
                inputs=[float(coupling.rho), group.W, group.u, group.Jv],
                outputs=[group.lambda_],
                device=self.model.device,
            )
        for group in self._admm_rr_revolute_angular_groups:
            if group.count == 0:
                continue
            entry_a = self._entries[group.body_entry_name_a]
            entry_b = self._entries[group.body_entry_name_b]
            wp.launch(
                attach_rr_revolute_angular_local_compute_Jv_kernel,
                dim=group.count,
                inputs=[
                    group.body_ids_a,
                    group.frame_a,
                    group.body_ids_b,
                    entry_a.state_1.body_q,
                    entry_a.state_1.body_qd,
                    entry_b.state_1.body_qd,
                ],
                outputs=[group.Jv],
                device=self.model.device,
            )
            wp.launch(
                u_update_quadratic_kernel,
                dim=group.count,
                inputs=[
                    group.kappa,
                    group.damping,
                    group.W,
                    float(coupling.rho),
                    group.lambda_,
                    group.Jv,
                    group.u_target,
                ],
                outputs=[group.u],
                device=self.model.device,
            )
            wp.launch(
                lambda_update_kernel,
                dim=group.count,
                inputs=[float(coupling.rho), group.W, group.u, group.Jv],
                outputs=[group.lambda_],
                device=self.model.device,
            )
        for group in self._admm_rr_angular_friction_groups:
            if group.count == 0:
                continue
            entry_a = self._entries[group.body_entry_name_a]
            entry_b = self._entries[group.body_entry_name_b]
            wp.launch(
                attach_rr_angular_local_compute_Jv_kernel,
                dim=group.count,
                inputs=[
                    group.body_ids_a,
                    group.frame_a,
                    group.body_ids_b,
                    entry_a.state_1.body_q,
                    entry_a.state_1.body_qd,
                    entry_b.state_1.body_qd,
                ],
                outputs=[group.Jv],
                device=self.model.device,
            )
            wp.launch(
                joint_box_friction_u_update_kernel,
                dim=group.count,
                inputs=[group.friction, group.W, float(coupling.rho), group.lambda_, group.Jv],
                outputs=[group.u],
                device=self.model.device,
            )
            wp.launch(
                lambda_update_kernel,
                dim=group.count,
                inputs=[float(coupling.rho), group.W, group.u, group.Jv],
                outputs=[group.lambda_],
                device=self.model.device,
            )
        for group in self._admm_rp_groups:
            if group.count == 0:
                continue
            body_entry = self._entries[group.body_entry_name]
            particle_entry = self._entries[group.particle_entry_name]
            wp.launch(
                attach_rp_compute_Jv_kernel,
                dim=group.count,
                inputs=[
                    group.body_ids,
                    group.point_body,
                    group.particle_ids,
                    body_entry.state_1.body_q,
                    body_entry.view.body_com,
                    body_entry.state_1.body_qd,
                    particle_entry.state_1.particle_qd,
                ],
                outputs=[group.Jv],
                device=self.model.device,
            )
            wp.launch(
                u_update_quadratic_kernel,
                dim=group.count,
                inputs=[
                    group.kappa,
                    group.damping,
                    group.W,
                    float(coupling.rho),
                    group.lambda_,
                    group.Jv,
                    group.u_target,
                ],
                outputs=[group.u],
                device=self.model.device,
            )
            wp.launch(
                lambda_update_kernel,
                dim=group.count,
                inputs=[float(coupling.rho), group.W, group.u, group.Jv],
                outputs=[group.lambda_],
                device=self.model.device,
            )
        for group in self._admm_rr_contact_groups:
            if group.count == 0:
                continue
            entry_a = self._entries[group.body_entry_name_a]
            entry_b = self._entries[group.body_entry_name_b]
            wp.launch(
                contact_rr_compute_Jv_kernel,
                dim=group.count,
                inputs=[
                    group.body_ids_a,
                    group.point_a,
                    group.body_ids_b,
                    group.point_b,
                    entry_a.state_1.body_q,
                    entry_a.view.body_com,
                    entry_a.state_1.body_qd,
                    entry_b.state_1.body_q,
                    entry_b.view.body_com,
                    entry_b.state_1.body_qd,
                ],
                outputs=[group.Jv],
                device=self.model.device,
            )
            wp.launch(
                contact_u_update_kernel,
                dim=group.count,
                inputs=[
                    group.u_min,
                    group.W,
                    float(coupling.rho),
                    group.friction,
                    group.normal,
                    group.lambda_,
                    group.Jv,
                ],
                outputs=[group.u],
                device=self.model.device,
            )
            wp.launch(
                contact_lambda_update_kernel,
                dim=group.count,
                inputs=[float(coupling.rho), group.W, group.u, group.Jv],
                outputs=[group.lambda_],
                device=self.model.device,
            )
        for group in self._admm_rp_contact_groups:
            if group.count == 0:
                continue
            body_entry = self._entries[group.body_entry_name]
            particle_entry = self._entries[group.particle_entry_name]
            wp.launch(
                contact_rp_compute_Jv_kernel,
                dim=group.count,
                inputs=[
                    group.body_ids,
                    group.point_body,
                    group.particle_ids,
                    group.body_sign,
                    body_entry.state_1.body_q,
                    body_entry.view.body_com,
                    body_entry.state_1.body_qd,
                    particle_entry.state_1.particle_qd,
                ],
                outputs=[group.Jv],
                device=self.model.device,
            )
            wp.launch(
                contact_u_update_kernel,
                dim=group.count,
                inputs=[
                    group.u_min,
                    group.W,
                    float(coupling.rho),
                    group.friction,
                    group.normal,
                    group.lambda_,
                    group.Jv,
                ],
                outputs=[group.u],
                device=self.model.device,
            )
            wp.launch(
                contact_lambda_update_kernel,
                dim=group.count,
                inputs=[float(coupling.rho), group.W, group.u, group.Jv],
                outputs=[group.lambda_],
                device=self.model.device,
            )
        for group in self._admm_pp_contact_groups:
            if group.count == 0:
                continue
            entry_a = self._entries[group.particle_entry_name_a]
            entry_b = self._entries[group.particle_entry_name_b]
            wp.launch(
                contact_pp_compute_Jv_kernel,
                dim=group.count,
                inputs=[
                    group.particle_ids_a,
                    group.particle_ids_b,
                    entry_a.state_1.particle_qd,
                    entry_b.state_1.particle_qd,
                ],
                outputs=[group.Jv],
                device=self.model.device,
            )
            wp.launch(
                contact_u_update_kernel,
                dim=group.count,
                inputs=[
                    group.u_min,
                    group.W,
                    float(coupling.rho),
                    group.friction,
                    group.normal,
                    group.lambda_,
                    group.Jv,
                ],
                outputs=[group.u],
                device=self.model.device,
            )
            wp.launch(
                contact_lambda_update_kernel,
                dim=group.count,
                inputs=[float(coupling.rho), group.W, group.u, group.Jv],
                outputs=[group.lambda_],
                device=self.model.device,
            )
