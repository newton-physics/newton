# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""ADMM-style coupled multi-solver simulations."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import numpy as np
import warp as wp

from ...geometry.flags import ShapeFlags
from ...math import quat_between_vectors_robust
from ...sim import BodyFlags, JointType, ModelFlags, StateFlags
from .admm_contact_stream import (
    AdmmContactStream,
    AdmmContactType,
    admm_contact_stream_reset_count_kernel,
    admm_contact_stream_update_normal_force_kernel,
)
from .admm_utils import (
    accumulate_active_body_contact_proximal_lump_kernel,
    accumulate_active_body_point_proximal_lump_kernel,
    accumulate_active_global_indices_proximal_lump_kernel,
    accumulate_active_indices_proximal_lump_kernel,
    accumulate_body_angular_proximal_lump_kernel,
    accumulate_body_point_proximal_lump_kernel,
    accumulate_global_indices_proximal_lump_kernel,
    accumulate_indices_proximal_lump_kernel,
    accumulate_joint_qd_factor_from_body_proximal_lump_kernel,
    attach_rp_accumulate_forces_kernel,
    attach_rp_compute_Jv_kernel,
    attach_rp_compute_u_target_kernel,
    attach_rr_accumulate_forces_kernel,
    attach_rr_angular_accumulate_forces_kernel,
    attach_rr_angular_compute_Jv_kernel,
    attach_rr_angular_compute_u_target_kernel,
    attach_rr_angular_local_accumulate_forces_kernel,
    attach_rr_angular_local_compute_Jv_kernel,
    attach_rr_compute_Jv_kernel,
    attach_rr_compute_u_target_kernel,
    attach_rr_revolute_angular_local_accumulate_forces_kernel,
    attach_rr_revolute_angular_local_compute_Jv_kernel,
    attach_rr_revolute_angular_local_compute_u_target_kernel,
    body_gravity_compensation_lumped_kernel,
    compute_interface_weights_kernel,
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
    contact_rr_clear_contact_snapshot_kernel,
    contact_rr_compute_Jv_kernel,
    contact_rr_compute_u_min_kernel,
    contact_rr_fill_from_rigid_contacts_kernel,
    contact_rr_reset_kernel,
    contact_rr_snapshot_by_contact_kernel,
    contact_u_update_kernel,
    joint_box_friction_u_update_kernel,
    lambda_update_kernel,
    mark_active_global_indices_mask_kernel,
    mark_active_indices_mask_kernel,
    mark_active_pair_indices_mask_kernel,
    mark_global_indices_mask_kernel,
    mark_indices_mask_kernel,
    mark_local_indices_from_global_mask_kernel,
    particle_gravity_compensation_lumped_kernel,
    particle_particle_contacts_hashgrid_kernel,
    scatter_body_effective_mass_block_kernel,
    scatter_effective_mass_kernel,
    u_update_quadratic_kernel,
    velocity_proximal_shift_body_lumped_kernel,
    velocity_proximal_shift_joint_lumped_kernel,
    velocity_proximal_shift_particle_lumped_kernel,
)
from .interface import (
    CouplingEndpointKind,
)
from .model_view import ModelView
from .solver_coupled import (
    SolverCoupled,
    SolverEntry,
    _copy_mapped_spatial_vector,
    _copy_mapped_vec3,
)

if TYPE_CHECKING:
    from ...sim import Contacts, Control, Model, ModelBuilder, State


@wp.kernel(enable_backward=False)
def _disable_proxy_shape_collisions_kernel(
    shape_body: wp.array[int],
    body_flags: wp.array[int],
    proxy_flag: int,
    collision_mask: int,
    shape_flags: wp.array[int],
):
    shape = wp.tid()
    body = shape_body[shape]
    if body >= 0 and body_flags[body] & proxy_flag:
        shape_flags[shape] = shape_flags[shape] & ~collision_mask


@wp.kernel(enable_backward=False)
def _compute_body_inertia_scalar_kernel(
    body_inertia: wp.array[wp.mat33],
    body_inertia_scalar: wp.array[float],
):
    body = wp.tid()
    inertia = body_inertia[body]
    body_inertia_scalar[body] = wp.max((inertia[0, 0] + inertia[1, 1] + inertia[2, 2]) / 3.0, 0.0)


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
    body_effective_inertia_scalar: wp.array = field(default=None)
    particle_effective_mass: wp.array = field(default=None)
    body_proximal_mask: wp.array = field(default=None)
    body_proximal_mass: wp.array = field(default=None)
    body_proximal_inertia: wp.array = field(default=None)
    particle_proximal_mask: wp.array = field(default=None)
    particle_proximal_mass: wp.array = field(default=None)
    joint_qd_proximal_mask: wp.array = field(default=None)
    joint_qd_proximal_factor: wp.array = field(default=None)
    supports_dynamic_inertial_refresh: bool = False
    body_joint_qd_start: wp.array = field(default=None)
    body_joint_qd_count: wp.array = field(default=None)
    body_joint_qd_indices: wp.array = field(default=None)
    body_endpoint_kind: wp.array = field(default=None)
    body_endpoint_index: wp.array = field(default=None)
    body_endpoint_local_pos: wp.array = field(default=None)
    body_effective_mass_local: wp.array = field(default=None)
    body_effective_inertia_local: wp.array = field(default=None)
    particle_endpoint_kind: wp.array = field(default=None)
    particle_endpoint_index: wp.array = field(default=None)
    particle_endpoint_local_pos: wp.array = field(default=None)
    particle_effective_mass_local: wp.array = field(default=None)


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
    offset_a: wp.array
    body_ids_b: wp.array
    point_b: wp.array
    offset_b: wp.array
    contact_ids: wp.array
    normal: wp.array
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
    prev_contact_active: wp.array | None = None
    prev_contact_lambda: wp.array | None = None
    prev_contact_W: wp.array | None = None
    body_mask_a: wp.array | None = None
    body_mask_b: wp.array | None = None
    shape_mask_a: wp.array | None = None
    shape_mask_b: wp.array | None = None
    candidate_body_ids_a: wp.array | None = None
    candidate_body_ids_b: wp.array | None = None
    candidate_W: wp.array | None = None

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
    prev_body_ids: wp.array | None = None
    prev_particle_ids: wp.array | None = None
    prev_shape_ids: wp.array | None = None
    prev_active: wp.array | None = None
    prev_W: wp.array | None = None
    prev_lambda: wp.array | None = None
    candidate_body_ids: wp.array | None = None
    candidate_particle_ids: wp.array | None = None
    candidate_W: wp.array | None = None

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
    query_radius: float = 0.0
    prev_particle_ids_a: wp.array | None = None
    prev_particle_ids_b: wp.array | None = None
    prev_active: wp.array | None = None
    prev_W: wp.array | None = None
    prev_lambda: wp.array | None = None
    candidate_particle_ids_a: wp.array | None = None
    candidate_particle_ids_b: wp.array | None = None
    candidate_W: wp.array | None = None

    @property
    def count(self) -> int:
        return self.capacity if self.capacity is not None else self.particle_ids_a.shape[0]


@dataclass(frozen=True)
class _AdmmRigidParticleContactSpec:
    """Internal particle-shape contact source derived from model ownership."""

    particle_owner: str
    body_owner: str
    shapes: tuple[int, ...] | None = None


@dataclass(frozen=True)
class _AdmmRigidRigidContactSpec:
    """Internal rigid-rigid contact source derived from model ownership."""

    owner_a: str
    owner_b: str
    shapes_a: tuple[int, ...] | None = None
    shapes_b: tuple[int, ...] | None = None
    shape_pairs: tuple[tuple[int, int], ...] | None = None


@dataclass(frozen=True)
class _AdmmParticleParticleContactSpec:
    """Internal particle-particle contact source derived from model ownership."""

    owner_a: str
    owner_b: str
    particles_a: tuple[int, ...] | None = None
    particles_b: tuple[int, ...] | None = None


@dataclass
class _AdmmJointProxyMapping:
    """Cross-solver joint neighbor bodies kept dynamic in one entry view."""

    src_name: str
    dst_name: str
    body_ids_global: wp.array | None = None
    body_ids_local: wp.array | None = None
    proxy_mass: wp.array | None = None
    proxy_inertia: wp.array | None = None


_AdmmQuadraticGroup = (
    _AdmmRigidRigidAttachmentGroup | _AdmmRigidRigidAngularAttachmentGroup | _AdmmRigidParticleAttachmentGroup
)
_AdmmContactGroup = _AdmmRigidRigidContactGroup | _AdmmRigidParticleContactGroup | _AdmmParticleParticleContactGroup


class SolverCoupledADMM(SolverCoupled):
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
        construction, :class:`SolverCoupledADMM` converts rows whose body and
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
        """

        source: str
        destination: str

    @dataclass(frozen=True)
    class Config:
        """Linearized ADMM coupling configuration.

        Args:
            iterations: Positive number of ADMM iterations per solver step.
            rho: Positive ADMM penalty parameter.
            gamma: Nonnegative proximal mass scaling parameter.
            baumgarte: Nonnegative position error correction fraction.
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
            joint_proximal_bodies: Keep cross-solver joint neighbor bodies
                dynamic in each subsolver view as local inertial proxies.
            joint_proximal_destination_entries: Optional entry names that
                receive cross-solver joint proximal proxy bodies. ``None``
                keeps the default symmetric visibility.
            joint_proximal_mass_scale: Multiplier applied to source effective
                masses before installing cross-solver joint proxy inertias.
            rigid_contact_matching: Frame-to-frame contact matching mode for
                collision-detected rigid-rigid ADMM contacts. Use
                ``"disabled"`` to reset dynamic rigid contact state every
                refresh, ``"latest"`` to warm start matched contacts from the
                previous refresh, or ``"sticky"`` to also replay matched contact
                geometry. Matched contacts reuse only ADMM dual warm-start
                state; primal contact state is reset on every refresh.
            contact_matching_pos_threshold: World-space distance threshold [m]
                between previous and current rigid contact midpoints for
                non-disabled ``rigid_contact_matching`` modes. ``None`` uses
                the :class:`CollisionPipeline` default.
            contact_matching_normal_dot_threshold: Minimum dot product between
                previous and current rigid contact normals for non-disabled
                ``rigid_contact_matching`` modes. ``None`` uses the
                :class:`CollisionPipeline` default.
            contact_matching_force_scale: Multiplier applied to the rescaled
                previous-refresh ADMM contact dual when a rigid-rigid contact
                matches. ``0`` disables dual warm-start while preserving
                contact matching.
            contact_pairs: Per-interface contact pairs to enable. Empty list
                disables ADMM-managed contacts. Use
                :meth:`SolverCoupledADMM.auto_detect_contact_pairs` to build the
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
        joint_proximal_bodies: bool = True
        joint_proximal_destination_entries: Sequence[str] | None = None
        joint_proximal_mass_scale: float = 1.0
        rigid_contact_matching: Literal["disabled", "latest", "sticky"] = "disabled"
        contact_matching_pos_threshold: float | None = None
        contact_matching_normal_dot_threshold: float | None = None
        contact_matching_force_scale: float = 0.9
        contact_pairs: Sequence[SolverCoupledADMM.ContactPair] = ()

    def __init__(
        self,
        model: Model,
        entries: Sequence[SolverCoupled.Entry],
        coupling: SolverCoupledADMM.Config,
    ) -> None:
        self._admm_buffers: dict[str, _AdmmBuffers] = {}
        self._admm_rr_groups: list[_AdmmRigidRigidAttachmentGroup] = []
        self._admm_rr_angular_groups: list[_AdmmRigidRigidAngularAttachmentGroup] = []
        self._admm_rr_revolute_angular_groups: list[_AdmmRigidRigidAngularAttachmentGroup] = []
        self._admm_rr_angular_friction_groups: list[_AdmmRigidRigidAngularFrictionGroup] = []
        self._admm_rp_groups: list[_AdmmRigidParticleAttachmentGroup] = []
        self._admm_dynamic_rr_contact_groups: list[_AdmmRigidRigidContactGroup] = []
        self._admm_dynamic_rp_contact_groups: list[_AdmmRigidParticleContactGroup] = []
        self._admm_dynamic_pp_contact_groups: list[_AdmmParticleParticleContactGroup] = []
        self._admm_rigid_particle_contact_specs: list[_AdmmRigidParticleContactSpec] = []
        self._admm_rigid_rigid_contact_specs: list[_AdmmRigidRigidContactSpec] = []
        self._admm_particle_particle_contact_specs: list[_AdmmParticleParticleContactSpec] = []
        self._admm_collision_pipeline = None
        self._admm_internal_contacts = None
        self._admm_particle_contact_query_radius = 0.0
        self._entry_body_sets: dict[str, set[int]] = {}
        self._entry_particle_sets: dict[str, set[int]] = {}
        self._admm_rigid_particle_shape_filters: dict[int, set[int] | None] = {}
        self._admm_effective_mass_unsupported: set[tuple[str, int]] = set()
        self._admm_joint_proxy_body_keep: dict[str, set[int]] = {}
        self._admm_joint_proxy_joint_keep: dict[str, set[int]] = {}
        self._admm_joint_proxy_mappings: list[_AdmmJointProxyMapping] = []

        self._validate_config(coupling)
        if coupling.joint_proximal_bodies:
            self._init_admm_joint_proxy_visibility(model, entries, coupling.joint_proximal_destination_entries)

        super().__init__(
            model=model,
            entries=entries,
            coupling=coupling,
        )

        self._setup_admm(coupling)
        self._apply_cached_admm_joint_proxy_effective_masses()

    @classmethod
    def _validate_config(cls, coupling: SolverCoupledADMM.Config) -> None:
        cls._positive_integer(coupling.iterations, "ADMM iterations")
        cls._finite_scalar(coupling.rho, "ADMM rho", lower_bound=0.0, lower_inclusive=False)
        cls._finite_scalar(coupling.gamma, "ADMM gamma", lower_bound=0.0)
        cls._finite_scalar(coupling.baumgarte, "ADMM baumgarte", lower_bound=0.0)
        cls._finite_scalar(coupling.joint_stiffness, "ADMM joint_stiffness", lower_bound=0.0)
        cls._finite_scalar(coupling.joint_damping, "ADMM joint_damping", lower_bound=0.0)
        cls._finite_scalar(coupling.joint_angular_stiffness, "ADMM joint_angular_stiffness", lower_bound=0.0)
        cls._finite_scalar(coupling.joint_angular_damping, "ADMM joint_angular_damping", lower_bound=0.0)
        cls._finite_scalar(
            coupling.joint_proximal_mass_scale,
            "ADMM joint_proximal_mass_scale",
            lower_bound=0.0,
            lower_inclusive=False,
        )
        cls._finite_scalar(
            coupling.contact_matching_force_scale,
            "ADMM contact_matching_force_scale",
            lower_bound=0.0,
        )

        if coupling.rigid_contact_matching not in ("disabled", "latest", "sticky"):
            raise ValueError(
                "ADMM rigid_contact_matching must be 'disabled', 'latest', or 'sticky', "
                f"got {coupling.rigid_contact_matching!r}"
            )
        if coupling.contact_matching_pos_threshold is not None:
            cls._finite_scalar(
                coupling.contact_matching_pos_threshold,
                "ADMM contact_matching_pos_threshold",
                lower_bound=0.0,
            )
        if coupling.contact_matching_normal_dot_threshold is not None:
            cls._finite_scalar(
                coupling.contact_matching_normal_dot_threshold,
                "ADMM contact_matching_normal_dot_threshold",
                lower_bound=-1.0,
                upper_bound=1.0,
            )

    @staticmethod
    def _finite_scalar(
        value: float,
        label: str,
        *,
        lower_bound: float | None = None,
        upper_bound: float | None = None,
        lower_inclusive: bool = True,
    ) -> float:
        converted = float(value)
        if not np.isfinite(converted):
            raise ValueError(f"{label} must be finite, got {value!r}")
        if lower_bound is not None:
            below = converted < lower_bound if lower_inclusive else converted <= lower_bound
            if below:
                relation = ">=" if lower_inclusive else ">"
                raise ValueError(f"{label} must be {relation} {lower_bound}, got {value!r}")
        if upper_bound is not None and converted > upper_bound:
            raise ValueError(f"{label} must be <= {upper_bound}, got {value!r}")
        return converted

    def _init_admm_joint_proxy_visibility(
        self,
        model: Model,
        entries: Sequence[SolverCoupled.Entry],
        destination_entries: Sequence[str] | None,
    ) -> None:
        """Expose cross-solver joint neighbors as local inertial proxies."""
        entry_names = [entry.name for entry in entries]
        destination_names = {str(name) for name in destination_entries} if destination_entries is not None else None
        if destination_names is not None:
            unknown = destination_names - set(entry_names)
            if unknown:
                raise ValueError(f"Unknown ADMM joint proximal destination entries: {sorted(unknown)}")
        body_owner = self._build_owner_map(model.body_count, [entry.bodies for entry in entries])
        joint_owner = self._build_owner_map(model.joint_count, [entry.joints for entry in entries])
        owned_bodies = {entry.name: {int(body) for body in entry.bodies} for entry in entries}
        body_keep = {entry.name: set() for entry in entries}
        joint_keep = {entry.name: set() for entry in entries}
        pair_bodies: dict[tuple[str, str], set[int]] = {}

        if model.joint_count == 0 or model.body_count == 0:
            self._admm_joint_proxy_body_keep = body_keep
            self._admm_joint_proxy_joint_keep = joint_keep
            return

        joint_type = model.joint_type.numpy()
        joint_parent = model.joint_parent.numpy()
        joint_child = model.joint_child.numpy()
        joint_enabled = model.joint_enabled.numpy()
        body_world = model.body_world.numpy() if model.body_world is not None else []
        supported_joint_types = (int(JointType.BALL), int(JointType.REVOLUTE), int(JointType.FIXED))

        def add_proxy_body(dst_name: str, src_name: str | None, body: int) -> None:
            if destination_names is not None and dst_name not in destination_names:
                return
            body_keep[dst_name].add(body)
            if src_name is not None and src_name != dst_name:
                pair_bodies.setdefault((src_name, dst_name), set()).add(body)

        for joint in range(model.joint_count):
            if not bool(joint_enabled[joint]) or joint_owner[joint] >= 0:
                continue
            if int(joint_type[joint]) not in supported_joint_types:
                continue

            parent = int(joint_parent[joint])
            child = int(joint_child[joint])
            if parent < 0 or child < 0:
                continue
            parent_owner = body_owner[parent] if parent < len(body_owner) else -1
            child_owner = body_owner[child] if child < len(body_owner) else -1
            if parent_owner < 0 or child_owner < 0 or parent_owner == child_owner:
                continue

            world_parent = int(body_world[parent]) if len(body_world) > parent else -1
            world_child = int(body_world[child]) if len(body_world) > child else -1
            if world_parent != world_child:
                raise ValueError(
                    "ADMM cross-solver joint proximal bodies require source and destination bodies "
                    f"to live in the same world; joint {joint} references bodies {parent} in world {world_parent} "
                    f"and {child} in world {world_child}."
                )

            parent_name = entry_names[parent_owner]
            child_name = entry_names[child_owner]
            if destination_names is None or parent_name in destination_names:
                add_proxy_body(parent_name, child_name, child)
                joint_keep[parent_name].add(joint)
            if destination_names is None or child_name in destination_names:
                add_proxy_body(child_name, parent_name, parent)
                joint_keep[child_name].add(joint)

        self._add_admm_joint_proxy_topology_paths(
            model,
            entry_names,
            body_owner,
            owned_bodies,
            body_keep,
            joint_keep,
            pair_bodies,
        )

        self._admm_joint_proxy_body_keep = body_keep
        self._admm_joint_proxy_joint_keep = joint_keep
        self._admm_joint_proxy_mappings = [
            _AdmmJointProxyMapping(
                src_name=src_name,
                dst_name=dst_name,
                body_ids_global=wp.array(sorted(bodies), dtype=int, device=model.device),
            )
            for (src_name, dst_name), bodies in sorted(pair_bodies.items())
            if bodies
        ]

    def _add_admm_joint_proxy_topology_paths(
        self,
        model: Model,
        entry_names: Sequence[str],
        body_owner: Sequence[int],
        owned_bodies: dict[str, set[int]],
        body_keep: dict[str, set[int]],
        joint_keep: dict[str, set[int]],
        pair_bodies: dict[tuple[str, str], set[int]],
    ) -> None:
        """Add minimal incoming tree-joint paths needed to instantiate proxy bodies."""
        if model.joint_count == 0 or model.joint_articulation is None:
            return

        joint_child = model.joint_child.numpy()
        joint_parent = model.joint_parent.numpy()
        joint_articulation = model.joint_articulation.numpy()
        incoming_tree_joint: dict[int, int] = {}
        for joint, child in enumerate(joint_child):
            if int(joint_articulation[joint]) >= 0:
                incoming_tree_joint.setdefault(int(child), joint)

        def add_path_body(dst_name: str, body: int) -> None:
            if body < 0 or body >= len(body_owner):
                return
            body_keep[dst_name].add(body)
            owner = int(body_owner[body])
            if owner >= 0:
                src_name = entry_names[owner]
                if src_name != dst_name:
                    pair_bodies.setdefault((src_name, dst_name), set()).add(body)

        for dst_name in list(body_keep):
            visible_bodies = set(owned_bodies[dst_name]) | set(body_keep[dst_name])
            queue = list(body_keep[dst_name])
            while queue:
                body = queue.pop()
                if body in owned_bodies[dst_name]:
                    continue
                joint = incoming_tree_joint.get(body)
                if joint is None:
                    continue
                joint_keep[dst_name].add(joint)
                parent = int(joint_parent[joint])
                if parent < 0 or parent in visible_bodies:
                    continue
                visible_bodies.add(parent)
                add_path_body(dst_name, parent)
                queue.append(parent)

    def _entry_proxy_body_keep_indices(self, name: str) -> set[int]:
        return set(self._admm_joint_proxy_body_keep.get(name, ()))

    def _entry_proxy_joint_keep_indices(self, name: str) -> set[int]:
        return set(self._admm_joint_proxy_joint_keep.get(name, ()))

    def _after_entries_constructed(self) -> None:
        self._refresh_admm_joint_proxy_view_maps()
        self._cache_admm_joint_proxy_effective_masses()

    def _entry_needs_gravity_acceleration(self, entry: SolverEntry) -> bool:
        del entry
        return True

    def _refresh_admm_joint_proxy_view_maps(self) -> None:
        for mapping in self._admm_joint_proxy_mappings:
            if mapping.body_ids_global is None:
                continue
            mapping.body_ids_local = wp.array(
                [self._body_local_id(mapping.dst_name, int(body)) for body in mapping.body_ids_global.numpy()],
                dtype=int,
                device=self.model.device,
            )

    def _cache_admm_joint_proxy_effective_masses(self) -> None:
        mass_scale = float(self._coupling.joint_proximal_mass_scale)
        for mapping in self._admm_joint_proxy_mappings:
            if (
                mapping.body_ids_global is None
                or mapping.body_ids_local is None
                or mapping.body_ids_global.shape[0] == 0
            ):
                continue
            src = self._entries[mapping.src_name]
            inertial_properties = self._eval_effective_body_inertial_properties(
                src,
                mapping.body_ids_global,
                raise_on_unsupported=False,
            )
            if inertial_properties is None:
                continue
            masses, inertias = inertial_properties
            proxy_masses = wp.array(
                [mass_scale * float(mass) for mass in masses], dtype=float, device=self.model.device
            )
            proxy_inertias = wp.array(
                [wp.mat33(np.asarray(inertia, dtype=np.float32) * mass_scale) for inertia in inertias],
                dtype=wp.mat33,
                device=self.model.device,
            )
            mapping.proxy_mass = proxy_masses
            mapping.proxy_inertia = proxy_inertias

    def _apply_cached_admm_joint_proxy_effective_masses(self) -> None:
        for mapping in self._admm_joint_proxy_mappings:
            if mapping.body_ids_local is None or mapping.proxy_mass is None or mapping.proxy_inertia is None:
                continue
            dst = self._entries[mapping.dst_name]
            self._apply_body_inertia_override(dst, mapping.body_ids_local, mapping.proxy_mass, mapping.proxy_inertia)

    def _refresh_model_view_overrides(self, flags: int) -> None:
        super()._refresh_model_view_overrides(flags)

    def notify_model_changed(self, flags: int) -> None:
        super().notify_model_changed(flags)
        if int(flags) & int(ModelFlags.BODY_INERTIAL_PROPERTIES):
            self._refresh_admm_body_effective_mass_buffers()
            self._cache_admm_joint_proxy_effective_masses()
            self._apply_cached_admm_joint_proxy_effective_masses()

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

    def _customize_compact_view(self, view: ModelView) -> None:
        """Apply ADMM customizations to the compact entry view."""
        self._disable_admm_joint_proxy_shape_collisions(view)

    def _disable_admm_joint_proxy_shape_collisions(self, view: ModelView) -> None:
        if view.shape_count == 0 or view.shape_body is None or view.shape_flags is None or view.body_flags is None:
            return

        proxy_flag = int(BodyFlags.PROXY)
        collision_mask = int(ShapeFlags.COLLIDE_SHAPES | ShapeFlags.COLLIDE_PARTICLES | ShapeFlags.HYDROELASTIC)
        wp.launch(
            _disable_proxy_shape_collisions_kernel,
            dim=view.shape_count,
            inputs=[view.shape_body, view.body_flags, proxy_flag, collision_mask],
            outputs=[view._cow_array("shape_flags")],
            device=self.model.device,
        )

    def _refresh_body_inertial_view_overrides(self, entry: SolverEntry) -> None:
        gamma = float(self._coupling.gamma)
        if gamma <= 0.0:
            super()._refresh_body_inertial_view_overrides(entry)
            return

        entry.view._refresh_body_inertial_properties(entry.body_local_to_global)
        buf = self._admm_buffers.get(entry.name)
        if buf is not None and buf.body_proximal_mass is not None and buf.body_proximal_inertia is not None:
            entry.view.add_body_lumped_inertia(buf.body_proximal_mass, buf.body_proximal_inertia)
        if entry.body_dynamics_disabled_local_indices.shape[0] > 0:
            entry.view.disable_body_dynamics(entry.body_dynamics_disabled_local_indices)

    def _setup_admm(self, coupling: SolverCoupledADMM.Config) -> None:
        for entry in self._entries.values():
            buf = _AdmmBuffers()
            buf.supports_dynamic_inertial_refresh = bool(entry.solver.coupling_supports_inertial_property_refresh())
            s0 = entry.state_0
            if s0.body_q is not None:
                buf.body_q_n = wp.empty_like(s0.body_q)
                buf.body_qd_n = wp.empty_like(s0.body_qd)
                buf.body_qd_k = wp.empty_like(s0.body_qd)
                buf.body_proximal_mask = wp.zeros(s0.body_qd.shape[0], dtype=int, device=self.model.device)
                buf.body_proximal_mass = wp.zeros(s0.body_qd.shape[0], dtype=float, device=self.model.device)
                buf.body_proximal_inertia = wp.zeros(s0.body_qd.shape[0], dtype=float, device=self.model.device)
            if s0.body_f is not None:
                buf.body_f = wp.empty_like(s0.body_f)
            if s0.particle_q is not None:
                buf.particle_q_n = wp.empty_like(s0.particle_q)
                buf.particle_qd_n = wp.empty_like(s0.particle_qd)
                buf.particle_qd_k = wp.empty_like(s0.particle_qd)
                buf.particle_proximal_mask = wp.zeros(s0.particle_qd.shape[0], dtype=int, device=self.model.device)
                buf.particle_proximal_mass = wp.zeros(s0.particle_qd.shape[0], dtype=float, device=self.model.device)
            if s0.particle_f is not None:
                buf.particle_f = wp.empty_like(s0.particle_f)
            if s0.joint_q is not None:
                buf.joint_q_n = wp.empty_like(s0.joint_q)
                buf.joint_qd_n = wp.empty_like(s0.joint_qd)
                buf.joint_qd_k = wp.empty_like(s0.joint_qd)
                buf.joint_qd_proximal_mask = wp.zeros(s0.joint_qd.shape[0], dtype=int, device=self.model.device)
                buf.joint_qd_proximal_factor = wp.zeros(s0.joint_qd.shape[0], dtype=float, device=self.model.device)
            self._admm_buffers[entry.name] = buf

        self._entry_body_sets = {
            name: {int(i) for i in entry.body_indices.numpy()} for name, entry in self._entries.items()
        }
        self._entry_particle_sets = {
            name: {int(i) for i in entry.particle_indices.numpy()} for name, entry in self._entries.items()
        }

        for entry in self._entries.values():
            self._setup_admm_effective_mass_buffers(entry, self._admm_buffers[entry.name])
            self._setup_admm_body_joint_qd_proximal_map(entry, self._admm_buffers[entry.name])

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
            admm_shape_pairs = (
                self._build_admm_rigid_shape_pair_array() if self._admm_rigid_rigid_contact_specs else None
            )
            rigid_contact_max = self._admm_rigid_contact_capacity() if self._admm_rigid_rigid_contact_specs else None
            if self._admm_rigid_rigid_contact_specs:
                self._admm_dynamic_rr_contact_groups = self._build_collision_rigid_rigid_contact_groups()
            from ...sim import CollisionPipeline  # noqa: PLC0415

            matching_kwargs = {}
            if coupling.contact_matching_pos_threshold is not None:
                matching_kwargs["contact_matching_pos_threshold"] = float(coupling.contact_matching_pos_threshold)
            if coupling.contact_matching_normal_dot_threshold is not None:
                matching_kwargs["contact_matching_normal_dot_threshold"] = float(
                    coupling.contact_matching_normal_dot_threshold
                )

            self._admm_collision_pipeline = CollisionPipeline(
                self.model,
                broad_phase="explicit",
                shape_pairs_filtered=admm_shape_pairs,
                rigid_contact_max=rigid_contact_max,
                soft_contact_max=None if self._admm_rigid_particle_contact_specs else 0,
                soft_contact_margin=0.0,
                contact_matching=(
                    coupling.rigid_contact_matching if self._admm_rigid_rigid_contact_specs else "disabled"
                ),
                **matching_kwargs,
            )
            if self._admm_rigid_particle_contact_specs:
                self._admm_dynamic_rp_contact_groups = self._build_collision_rigid_particle_contact_groups()

        if self._admm_particle_particle_contact_specs:
            self._validate_particle_particle_contact_specs()
            if self.model.particle_grid is not None:
                self._admm_dynamic_pp_contact_groups = self._build_collision_particle_particle_contact_groups()
                if self._admm_dynamic_pp_contact_groups:
                    self._admm_particle_contact_query_radius = max(
                        group.query_radius for group in self._admm_dynamic_pp_contact_groups
                    )
                    with wp.ScopedDevice(self.model.device):
                        self.model.particle_grid.reserve(self.model.particle_count)

        # Eagerly allocate the internal contact buffer so it exists before any
        # CUDA graph capture. Lazy allocation during capture leaves a bogus
        # pointer in the captured graph.
        if (
            self._admm_dynamic_rr_contact_groups or self._admm_dynamic_rp_contact_groups
        ) and self._admm_internal_contacts is None:
            self._admm_internal_contacts = self._admm_collision_pipeline.contacts()

        if coupling.gamma > 0.0:
            self._refresh_admm_proximal_masks()
            self._refresh_admm_proximal_view_overrides(
                refresh_supported_solvers=True,
                notify_unsupported_solvers=True,
            )

    def _setup_admm_body_joint_qd_proximal_map(self, entry: SolverEntry, buf: _AdmmBuffers) -> None:
        if (
            buf.body_proximal_mass is None
            or buf.joint_qd_proximal_factor is None
            or buf.body_proximal_mass.shape[0] == 0
            or buf.joint_qd_proximal_factor.shape[0] == 0
            or self.model.joint_count == 0
        ):
            return

        joint_child = self.model.joint_child.numpy() if self.model.joint_child is not None else []
        joint_articulation = self.model.joint_articulation.numpy() if self.model.joint_articulation is not None else []
        joint_qd_start = self.model.joint_qd_start.numpy()
        articulation_start = self.model.articulation_start.numpy() if self.model.articulation_start is not None else []
        articulation_end = self.model.articulation_end.numpy() if self.model.articulation_end is not None else []
        joint_dof_global_to_local = entry.joint_dof_global_to_local.numpy()

        incoming_joint_by_body: dict[int, int] = {}
        for joint, child in enumerate(joint_child):
            incoming_joint_by_body.setdefault(int(child), joint)

        starts: list[int] = []
        counts: list[int] = []
        indices: list[int] = []
        for global_body in entry.body_local_to_global.numpy():
            starts.append(len(indices))
            joint = incoming_joint_by_body.get(int(global_body))
            if joint is None:
                counts.append(0)
                continue

            joint_start = joint
            joint_end = joint + 1
            if len(joint_articulation) > joint:
                articulation = int(joint_articulation[joint])
                if articulation >= 0 and len(articulation_start) > articulation:
                    joint_start = int(articulation_start[articulation])
                    if len(articulation_end) > articulation:
                        joint_end = int(articulation_end[articulation])
                    elif len(articulation_start) > articulation + 1:
                        joint_end = int(articulation_start[articulation + 1])

            local_dofs: list[int] = []
            for joint_id in range(joint_start, joint_end):
                if joint_id < 0 or joint_id + 1 >= len(joint_qd_start):
                    continue
                for global_dof in range(int(joint_qd_start[joint_id]), int(joint_qd_start[joint_id + 1])):
                    if global_dof < 0 or global_dof >= len(joint_dof_global_to_local):
                        continue
                    local_dof = int(joint_dof_global_to_local[global_dof])
                    if local_dof >= 0:
                        local_dofs.append(local_dof)

            unique_dofs = sorted(set(local_dofs))
            indices.extend(unique_dofs)
            counts.append(len(unique_dofs))

        device = self.model.device
        buf.body_joint_qd_start = wp.array(starts, dtype=int, device=device)
        buf.body_joint_qd_count = wp.array(counts, dtype=int, device=device)
        buf.body_joint_qd_indices = wp.array(indices, dtype=int, device=device)

    def _refresh_admm_proximal_masks(self) -> None:
        for buf in self._admm_buffers.values():
            self._zero_array(buf.body_proximal_mask)
            self._zero_array(buf.body_proximal_mass)
            self._zero_array(buf.body_proximal_inertia)
            self._zero_array(buf.particle_proximal_mask)
            self._zero_array(buf.particle_proximal_mass)
            self._zero_array(buf.joint_qd_proximal_mask)
            self._zero_array(buf.joint_qd_proximal_factor)

        self._mark_static_admm_proximal_masks()
        self._mark_dynamic_contact_admm_proximal_masks()
        self._mark_joint_qd_proximal_masks_from_bodies()

    def _proximal_gamma_rho(self) -> float:
        return float(self._coupling.gamma) * float(self._coupling.rho)

    def _mark_indices_for_proximal_mask(self, mask: wp.array | None, indices: wp.array | None) -> None:
        if mask is None or indices is None or indices.shape[0] == 0:
            return
        wp.launch(
            mark_indices_mask_kernel,
            dim=indices.shape[0],
            inputs=[indices, mask],
            device=self.model.device,
        )

    def _mark_active_indices_for_proximal_mask(
        self,
        active_count: wp.array | None,
        indices: wp.array | None,
        mask: wp.array | None,
    ) -> None:
        if active_count is None or indices is None or mask is None or indices.shape[0] == 0:
            return
        wp.launch(
            mark_active_indices_mask_kernel,
            dim=indices.shape[0],
            inputs=[active_count, indices, mask],
            device=self.model.device,
        )

    def _mark_global_indices_for_proximal_mask(
        self,
        indices: wp.array | None,
        global_to_local: wp.array | None,
        mask: wp.array | None,
    ) -> None:
        if indices is None or global_to_local is None or mask is None or indices.shape[0] == 0:
            return
        wp.launch(
            mark_global_indices_mask_kernel,
            dim=indices.shape[0],
            inputs=[indices, global_to_local, mask],
            device=self.model.device,
        )

    def _mark_active_global_indices_for_proximal_mask(
        self,
        active_count: wp.array | None,
        indices: wp.array | None,
        global_to_local: wp.array | None,
        mask: wp.array | None,
    ) -> None:
        if active_count is None or indices is None or global_to_local is None or mask is None or indices.shape[0] == 0:
            return
        wp.launch(
            mark_active_global_indices_mask_kernel,
            dim=indices.shape[0],
            inputs=[active_count, indices, global_to_local, mask],
            device=self.model.device,
        )

    def _mark_active_pair_for_proximal_mask(
        self,
        active_count: wp.array | None,
        indices_a: wp.array | None,
        mask_a: wp.array | None,
        indices_b: wp.array | None,
        mask_b: wp.array | None,
    ) -> None:
        if (
            active_count is None
            or indices_a is None
            or mask_a is None
            or indices_b is None
            or mask_b is None
            or indices_a.shape[0] == 0
        ):
            return
        wp.launch(
            mark_active_pair_indices_mask_kernel,
            dim=indices_a.shape[0],
            inputs=[active_count, indices_a, mask_a, indices_b, mask_b],
            device=self.model.device,
        )

    def _mark_global_candidates_for_proximal_mask(
        self,
        global_mask: wp.array | None,
        local_to_global: wp.array | None,
        local_mask: wp.array | None,
    ) -> None:
        if global_mask is None or local_to_global is None or local_mask is None or local_to_global.shape[0] == 0:
            return
        wp.launch(
            mark_local_indices_from_global_mask_kernel,
            dim=local_to_global.shape[0],
            inputs=[local_to_global, global_mask, local_mask],
            device=self.model.device,
        )

    def _accumulate_body_point_proximal_lump(
        self,
        entry: SolverEntry,
        buf: _AdmmBuffers,
        body_ids: wp.array | None,
        point_local: wp.array | None,
        W: wp.array | None,
    ) -> None:
        if (
            body_ids is None
            or point_local is None
            or W is None
            or buf.body_proximal_mass is None
            or buf.body_proximal_inertia is None
            or buf.body_proximal_mask is None
            or body_ids.shape[0] == 0
        ):
            return
        wp.launch(
            accumulate_body_point_proximal_lump_kernel,
            dim=body_ids.shape[0],
            inputs=[
                body_ids,
                point_local,
                entry.state_0.body_q,
                entry.view.body_com,
                W,
                self._proximal_gamma_rho(),
                buf.body_proximal_mass,
                buf.body_proximal_inertia,
                buf.body_proximal_mask,
            ],
            device=self.model.device,
        )

    def _accumulate_active_body_point_proximal_lump(
        self,
        entry: SolverEntry,
        buf: _AdmmBuffers,
        active_count: wp.array | None,
        body_ids: wp.array | None,
        point_local: wp.array | None,
        W: wp.array | None,
    ) -> None:
        if (
            active_count is None
            or body_ids is None
            or point_local is None
            or W is None
            or buf.body_proximal_mass is None
            or buf.body_proximal_inertia is None
            or buf.body_proximal_mask is None
            or body_ids.shape[0] == 0
        ):
            return
        wp.launch(
            accumulate_active_body_point_proximal_lump_kernel,
            dim=body_ids.shape[0],
            inputs=[
                active_count,
                body_ids,
                point_local,
                entry.state_0.body_q,
                entry.view.body_com,
                W,
                self._proximal_gamma_rho(),
                buf.body_proximal_mass,
                buf.body_proximal_inertia,
                buf.body_proximal_mask,
            ],
            device=self.model.device,
        )

    def _accumulate_active_body_contact_proximal_lump(
        self,
        entry: SolverEntry,
        buf: _AdmmBuffers,
        active_count: wp.array | None,
        body_ids: wp.array | None,
        point_local: wp.array | None,
        point_offset_local: wp.array | None,
        W: wp.array | None,
    ) -> None:
        if (
            active_count is None
            or body_ids is None
            or point_local is None
            or point_offset_local is None
            or W is None
            or buf.body_proximal_mass is None
            or buf.body_proximal_inertia is None
            or buf.body_proximal_mask is None
            or body_ids.shape[0] == 0
        ):
            return
        wp.launch(
            accumulate_active_body_contact_proximal_lump_kernel,
            dim=body_ids.shape[0],
            inputs=[
                active_count,
                body_ids,
                point_local,
                point_offset_local,
                entry.state_0.body_q,
                entry.view.body_com,
                W,
                self._proximal_gamma_rho(),
                buf.body_proximal_mass,
                buf.body_proximal_inertia,
                buf.body_proximal_mask,
            ],
            device=self.model.device,
        )

    def _accumulate_body_angular_proximal_lump(
        self,
        buf: _AdmmBuffers,
        body_ids: wp.array | None,
        W: wp.array | None,
        component_lump: float,
    ) -> None:
        if (
            body_ids is None
            or W is None
            or buf.body_proximal_inertia is None
            or buf.body_proximal_mask is None
            or body_ids.shape[0] == 0
        ):
            return
        wp.launch(
            accumulate_body_angular_proximal_lump_kernel,
            dim=body_ids.shape[0],
            inputs=[
                body_ids,
                W,
                self._proximal_gamma_rho(),
                float(component_lump),
                buf.body_proximal_inertia,
                buf.body_proximal_mask,
            ],
            device=self.model.device,
        )

    def _accumulate_indices_proximal_lump(
        self,
        indices: wp.array | None,
        W: wp.array | None,
        lump: wp.array | None,
        mask: wp.array | None,
    ) -> None:
        if indices is None or W is None or lump is None or mask is None or indices.shape[0] == 0:
            return
        wp.launch(
            accumulate_indices_proximal_lump_kernel,
            dim=indices.shape[0],
            inputs=[indices, W, self._proximal_gamma_rho(), lump, mask],
            device=self.model.device,
        )

    def _accumulate_active_indices_proximal_lump(
        self,
        active_count: wp.array | None,
        indices: wp.array | None,
        W: wp.array | None,
        lump: wp.array | None,
        mask: wp.array | None,
    ) -> None:
        if (
            active_count is None
            or indices is None
            or W is None
            or lump is None
            or mask is None
            or indices.shape[0] == 0
        ):
            return
        wp.launch(
            accumulate_active_indices_proximal_lump_kernel,
            dim=indices.shape[0],
            inputs=[active_count, indices, W, self._proximal_gamma_rho(), lump, mask],
            device=self.model.device,
        )

    def _accumulate_global_indices_proximal_lump(
        self,
        indices: wp.array | None,
        global_to_local: wp.array | None,
        W: wp.array | None,
        lump: wp.array | None,
        mask: wp.array | None,
    ) -> None:
        if (
            indices is None
            or global_to_local is None
            or W is None
            or lump is None
            or mask is None
            or indices.shape[0] == 0
        ):
            return
        wp.launch(
            accumulate_global_indices_proximal_lump_kernel,
            dim=indices.shape[0],
            inputs=[indices, global_to_local, W, self._proximal_gamma_rho(), lump, mask],
            device=self.model.device,
        )

    def _accumulate_active_global_indices_proximal_lump(
        self,
        active_count: wp.array | None,
        indices: wp.array | None,
        global_to_local: wp.array | None,
        W: wp.array | None,
        lump: wp.array | None,
        mask: wp.array | None,
    ) -> None:
        if (
            active_count is None
            or indices is None
            or global_to_local is None
            or W is None
            or lump is None
            or mask is None
            or indices.shape[0] == 0
        ):
            return
        wp.launch(
            accumulate_active_global_indices_proximal_lump_kernel,
            dim=indices.shape[0],
            inputs=[active_count, indices, global_to_local, W, self._proximal_gamma_rho(), lump, mask],
            device=self.model.device,
        )

    def _mark_static_admm_proximal_masks(self) -> None:
        for group in self._admm_rr_groups:
            entry_a = self._entries[group.body_entry_name_a]
            entry_b = self._entries[group.body_entry_name_b]
            buf_a = self._admm_buffers[group.body_entry_name_a]
            buf_b = self._admm_buffers[group.body_entry_name_b]
            self._accumulate_body_point_proximal_lump(entry_a, buf_a, group.body_ids_a, group.point_a, group.W)
            self._accumulate_body_point_proximal_lump(entry_b, buf_b, group.body_ids_b, group.point_b, group.W)

        for group in self._admm_rr_angular_groups:
            buf_a = self._admm_buffers[group.body_entry_name_a]
            buf_b = self._admm_buffers[group.body_entry_name_b]
            self._accumulate_body_angular_proximal_lump(buf_a, group.body_ids_a, group.W, 1.0)
            self._accumulate_body_angular_proximal_lump(buf_b, group.body_ids_b, group.W, 1.0)

        for group in self._admm_rr_revolute_angular_groups:
            buf_a = self._admm_buffers[group.body_entry_name_a]
            buf_b = self._admm_buffers[group.body_entry_name_b]
            self._accumulate_body_angular_proximal_lump(buf_a, group.body_ids_a, group.W, 2.0 / 3.0)
            self._accumulate_body_angular_proximal_lump(buf_b, group.body_ids_b, group.W, 2.0 / 3.0)

        for group in self._admm_rr_angular_friction_groups:
            buf_a = self._admm_buffers[group.body_entry_name_a]
            buf_b = self._admm_buffers[group.body_entry_name_b]
            self._accumulate_body_angular_proximal_lump(buf_a, group.body_ids_a, group.W, 1.0)
            self._accumulate_body_angular_proximal_lump(buf_b, group.body_ids_b, group.W, 1.0)

        for group in self._admm_rp_groups:
            body_entry = self._entries[group.body_entry_name]
            particle_entry = self._entries[group.particle_entry_name]
            body_buf = self._admm_buffers[group.body_entry_name]
            particle_buf = self._admm_buffers[group.particle_entry_name]
            self._accumulate_body_point_proximal_lump(
                body_entry,
                body_buf,
                group.body_ids,
                group.point_body,
                group.W,
            )
            self._accumulate_global_indices_proximal_lump(
                group.particle_ids,
                particle_entry.particle_global_to_local,
                group.W,
                particle_buf.particle_proximal_mass,
                particle_buf.particle_proximal_mask,
            )

    def _mark_dynamic_contact_admm_proximal_masks(self) -> None:
        for group in self._admm_dynamic_rr_contact_groups:
            entry_a = self._entries[group.body_entry_name_a]
            entry_b = self._entries[group.body_entry_name_b]
            buf_a = self._admm_buffers[group.body_entry_name_a]
            buf_b = self._admm_buffers[group.body_entry_name_b]
            if buf_a.supports_dynamic_inertial_refresh:
                self._accumulate_active_body_contact_proximal_lump(
                    entry_a,
                    buf_a,
                    group.active_count,
                    group.body_ids_a,
                    group.point_a,
                    group.offset_a,
                    group.W,
                )
            else:
                self._accumulate_indices_proximal_lump(
                    group.candidate_body_ids_a,
                    group.candidate_W,
                    buf_a.body_proximal_mass,
                    buf_a.body_proximal_mask,
                )
            if buf_b.supports_dynamic_inertial_refresh:
                self._accumulate_active_body_contact_proximal_lump(
                    entry_b,
                    buf_b,
                    group.active_count,
                    group.body_ids_b,
                    group.point_b,
                    group.offset_b,
                    group.W,
                )
            else:
                self._accumulate_indices_proximal_lump(
                    group.candidate_body_ids_b,
                    group.candidate_W,
                    buf_b.body_proximal_mass,
                    buf_b.body_proximal_mask,
                )

        for group in self._admm_dynamic_rp_contact_groups:
            body_entry = self._entries[group.body_entry_name]
            particle_entry = self._entries[group.particle_entry_name]
            body_buf = self._admm_buffers[group.body_entry_name]
            particle_buf = self._admm_buffers[group.particle_entry_name]
            if body_buf.supports_dynamic_inertial_refresh:
                self._accumulate_active_body_point_proximal_lump(
                    body_entry,
                    body_buf,
                    group.active_count,
                    group.body_ids,
                    group.point_body,
                    group.W,
                )
            else:
                self._accumulate_indices_proximal_lump(
                    group.candidate_body_ids,
                    group.candidate_W,
                    body_buf.body_proximal_mass,
                    body_buf.body_proximal_mask,
                )
            if particle_buf.supports_dynamic_inertial_refresh:
                self._accumulate_active_global_indices_proximal_lump(
                    group.active_count,
                    group.particle_ids,
                    particle_entry.particle_global_to_local,
                    group.W,
                    particle_buf.particle_proximal_mass,
                    particle_buf.particle_proximal_mask,
                )
            else:
                self._accumulate_global_indices_proximal_lump(
                    group.candidate_particle_ids,
                    particle_entry.particle_global_to_local,
                    group.candidate_W,
                    particle_buf.particle_proximal_mass,
                    particle_buf.particle_proximal_mask,
                )

        for group in self._admm_dynamic_pp_contact_groups:
            entry_a = self._entries[group.particle_entry_name_a]
            entry_b = self._entries[group.particle_entry_name_b]
            buf_a = self._admm_buffers[group.particle_entry_name_a]
            buf_b = self._admm_buffers[group.particle_entry_name_b]
            if buf_a.supports_dynamic_inertial_refresh:
                self._accumulate_active_global_indices_proximal_lump(
                    group.active_count,
                    group.particle_ids_a,
                    entry_a.particle_global_to_local,
                    group.W,
                    buf_a.particle_proximal_mass,
                    buf_a.particle_proximal_mask,
                )
            else:
                self._accumulate_global_indices_proximal_lump(
                    group.candidate_particle_ids_a,
                    entry_a.particle_global_to_local,
                    group.candidate_W,
                    buf_a.particle_proximal_mass,
                    buf_a.particle_proximal_mask,
                )
            if buf_b.supports_dynamic_inertial_refresh:
                self._accumulate_active_global_indices_proximal_lump(
                    group.active_count,
                    group.particle_ids_b,
                    entry_b.particle_global_to_local,
                    group.W,
                    buf_b.particle_proximal_mass,
                    buf_b.particle_proximal_mask,
                )
            else:
                self._accumulate_global_indices_proximal_lump(
                    group.candidate_particle_ids_b,
                    entry_b.particle_global_to_local,
                    group.candidate_W,
                    buf_b.particle_proximal_mass,
                    buf_b.particle_proximal_mask,
                )

    def _mark_joint_qd_proximal_masks_from_bodies(self) -> None:
        for entry_name, buf in self._admm_buffers.items():
            if (
                buf.body_proximal_mass is None
                or buf.body_proximal_inertia is None
                or buf.body_effective_mass is None
                or buf.body_effective_inertia_scalar is None
                or buf.joint_qd_proximal_mask is None
                or buf.joint_qd_proximal_factor is None
                or buf.body_joint_qd_start is None
                or buf.body_joint_qd_count is None
                or buf.body_joint_qd_indices is None
                or buf.body_proximal_mass.shape[0] == 0
                or buf.joint_qd_proximal_mask.shape[0] == 0
            ):
                continue
            entry = self._entries[entry_name]
            wp.launch(
                accumulate_joint_qd_factor_from_body_proximal_lump_kernel,
                dim=buf.body_proximal_mass.shape[0],
                inputs=[
                    buf.body_proximal_mass,
                    buf.body_proximal_inertia,
                    entry.body_local_to_global,
                    buf.body_effective_mass,
                    buf.body_effective_inertia_scalar,
                    buf.body_joint_qd_start,
                    buf.body_joint_qd_count,
                    buf.body_joint_qd_indices,
                    buf.joint_qd_proximal_factor,
                    buf.joint_qd_proximal_mask,
                ],
                device=self.model.device,
            )

    def _refresh_admm_proximal_view_overrides(
        self,
        *,
        refresh_supported_solvers: bool,
        notify_unsupported_solvers: bool = False,
    ) -> None:
        gamma = float(self._coupling.gamma)
        if gamma <= 0.0:
            return

        for entry in self._entries.values():
            buf = self._admm_buffers[entry.name]
            if buf.body_proximal_mass is not None and buf.body_proximal_inertia is not None:
                entry.view._refresh_body_inertial_properties(entry.body_local_to_global)
                entry.view.add_body_lumped_inertia(buf.body_proximal_mass, buf.body_proximal_inertia)
                if entry.body_dynamics_disabled_local_indices.shape[0] > 0:
                    entry.view.disable_body_dynamics(entry.body_dynamics_disabled_local_indices)
            if buf.particle_proximal_mass is not None:
                entry.view._refresh_particle_mass_properties(entry.particle_local_to_global)
                if entry.particle_dynamics_disabled_local_indices.shape[0] > 0:
                    entry.view.zero_particle_mass(entry.particle_dynamics_disabled_local_indices)
                entry.view.add_particle_lumped_mass(buf.particle_proximal_mass)
            if refresh_supported_solvers and buf.supports_dynamic_inertial_refresh:
                entry.solver.notify_model_changed(ModelFlags.BODY_INERTIAL_PROPERTIES)
            elif notify_unsupported_solvers and not buf.supports_dynamic_inertial_refresh:
                entry.solver.notify_model_changed(ModelFlags.BODY_INERTIAL_PROPERTIES)

    def _reset_coupling_state(
        self,
        state: State,
        *,
        world_mask: wp.array | None = None,
        flags: StateFlags | int | None = None,
    ) -> None:
        """Clear ADMM warm-start and internal contact buffers after reset."""
        super()._reset_coupling_state(state, world_mask=world_mask, flags=flags)
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
            self._zero_array(buf.body_f)
            self._zero_array(buf.particle_f)

        for group in (
            *self._admm_rr_groups,
            *self._admm_rr_angular_groups,
            *self._admm_rr_revolute_angular_groups,
            *self._admm_rr_angular_friction_groups,
            *self._admm_rp_groups,
            *self._admm_dynamic_rr_contact_groups,
            *self._admm_dynamic_rp_contact_groups,
            *self._admm_dynamic_pp_contact_groups,
        ):
            self._zero_group_reset_arrays(group)

        if self._admm_internal_contacts is not None:
            self._admm_internal_contacts.clear(bump_generation=True)
        if float(self._coupling.gamma) > 0.0:
            self._refresh_admm_proximal_masks()
            self._refresh_admm_proximal_view_overrides(refresh_supported_solvers=True)

    @staticmethod
    def _zero_array(array) -> None:
        if array is not None:
            array.zero_()

    @classmethod
    def _zero_group_reset_arrays(cls, group) -> None:
        for attr in (
            "u",
            "lambda_",
            "Jv",
            "u_target",
            "u_min",
            "active",
            "active_count",
            "active_count_max",
            "prev_active",
            "prev_W",
            "prev_lambda",
            "prev_contact_active",
            "prev_contact_lambda",
            "prev_contact_W",
        ):
            cls._zero_array(getattr(group, attr, None))
        contact_stream = getattr(group, "contact_stream", None)
        if contact_stream is None:
            return
        for attr in ("count", "count_max", "normal_force", "normal_impulse"):
            cls._zero_array(getattr(contact_stream, attr, None))

    def _setup_admm_effective_mass_buffers(self, entry: SolverEntry, buf: _AdmmBuffers) -> None:
        device = self.model.device
        if self.model.body_mass is not None:
            buf.body_effective_mass = wp.clone(self.model.body_mass, requires_grad=False)
            if self.model.body_inertia is not None:
                buf.body_effective_inertia_scalar = wp.empty(
                    self.model.body_inertia.shape[0],
                    dtype=wp.float32,
                    device=device,
                )
                wp.launch(
                    _compute_body_inertia_scalar_kernel,
                    dim=self.model.body_inertia.shape[0],
                    inputs=[self.model.body_inertia],
                    outputs=[buf.body_effective_inertia_scalar],
                    device=device,
                )
            (
                buf.body_endpoint_kind,
                buf.body_endpoint_index,
                buf.body_endpoint_local_pos,
                buf.body_effective_mass_local,
            ) = self._setup_admm_effective_mass_endpoint_buffers(
                entry,
                CouplingEndpointKind.BODY,
                entry.body_indices,
            )
            buf.body_effective_inertia_local = wp.empty(entry.body_indices.shape[0], dtype=wp.mat33, device=device)
            self._populate_admm_body_effective_mass_buffer(entry, buf, raise_on_unsupported=False)
        if self.model.particle_mass is not None:
            buf.particle_effective_mass = wp.clone(self.model.particle_mass, requires_grad=False)
            (
                buf.particle_endpoint_kind,
                buf.particle_endpoint_index,
                buf.particle_endpoint_local_pos,
                buf.particle_effective_mass_local,
            ) = self._setup_admm_effective_mass_endpoint_buffers(
                entry,
                CouplingEndpointKind.PARTICLE,
                entry.particle_indices,
            )
            self._populate_admm_particle_effective_mass_buffer(entry, buf, raise_on_unsupported=False)

    def _refresh_admm_body_effective_mass_buffers(self) -> None:
        if self.model.body_mass is None:
            return
        for entry in self._entries.values():
            buf = self._admm_buffers.get(entry.name)
            if buf is None or buf.body_effective_mass is None:
                continue
            wp.copy(buf.body_effective_mass, self.model.body_mass)
            if buf.body_effective_inertia_scalar is not None and self.model.body_inertia is not None:
                wp.launch(
                    _compute_body_inertia_scalar_kernel,
                    dim=self.model.body_inertia.shape[0],
                    inputs=[self.model.body_inertia],
                    outputs=[buf.body_effective_inertia_scalar],
                    device=self.model.device,
                )
            self._populate_admm_body_effective_mass_buffer(entry, buf, raise_on_unsupported=False)

    def _setup_admm_effective_mass_endpoint_buffers(
        self,
        entry: SolverEntry,
        endpoint_kind: CouplingEndpointKind,
        endpoint_indices: wp.array,
    ) -> tuple[wp.array, wp.array, wp.array, wp.array]:
        device = self.model.device
        indices = [int(i) for i in endpoint_indices.numpy()]
        local_indices = self._endpoint_indices_to_local(entry, endpoint_kind, indices)
        count = len(indices)
        return (
            wp.array([int(endpoint_kind)] * count, dtype=int, device=device),
            wp.array(local_indices, dtype=int, device=device),
            wp.zeros(count, dtype=wp.vec3, device=device),
            wp.empty(count, dtype=float, device=device),
        )

    def _mark_effective_mass_unsupported(
        self,
        entry: SolverEntry,
        endpoint_kind: CouplingEndpointKind,
    ) -> None:
        self._admm_effective_mass_unsupported.add((entry.name, int(endpoint_kind)))

    def _is_effective_mass_unsupported(self, entry_name: str, endpoint_kind: CouplingEndpointKind) -> bool:
        return (entry_name, int(endpoint_kind)) in self._admm_effective_mass_unsupported

    def _require_effective_mass(self, entry_name: str, endpoint_kind: CouplingEndpointKind) -> None:
        if not self._is_effective_mass_unsupported(entry_name, endpoint_kind):
            return
        solver = self._entries[entry_name].solver
        raise NotImplementedError(f"{solver.__class__.__name__} does not support coupling_eval_effective_mass()")

    def _compute_interface_weights(
        self,
        indices_a: Sequence[int],
        masses_a: wp.array[float] | None,
        indices_b: Sequence[int],
        masses_b: wp.array[float] | None,
    ) -> wp.array[float]:
        """Compute indexed endpoint-pair weights on the model device."""
        if len(indices_a) != len(indices_b):
            raise ValueError("ADMM interface weight index arrays must have the same length")
        count = len(indices_a)
        if masses_a is None or masses_b is None:
            return wp.ones(count, dtype=float, device=self.model.device)

        weights = wp.empty(count, dtype=float, device=self.model.device)
        wp.launch(
            compute_interface_weights_kernel,
            dim=count,
            inputs=[
                wp.array(indices_a, dtype=int, device=self.model.device),
                masses_a,
                wp.array(indices_b, dtype=int, device=self.model.device),
                masses_b,
            ],
            outputs=[weights],
            device=self.model.device,
        )
        return weights

    def _populate_admm_body_effective_mass_buffer(
        self,
        entry: SolverEntry,
        buf: _AdmmBuffers,
        *,
        raise_on_unsupported: bool = True,
    ) -> None:
        if buf.body_effective_mass is None or buf.body_endpoint_index is None or buf.body_endpoint_index.shape[0] == 0:
            return

        if buf.body_effective_inertia_scalar is not None:
            try:
                entry.solver.coupling_eval_effective_mass_block(
                    buf.body_endpoint_kind,
                    buf.body_endpoint_index,
                    buf.body_endpoint_local_pos,
                    buf.body_effective_mass_local,
                    buf.body_effective_inertia_local,
                )
            except NotImplementedError:
                if raise_on_unsupported:
                    raise
                self._mark_effective_mass_unsupported(entry, CouplingEndpointKind.BODY)
                return
            wp.launch(
                scatter_body_effective_mass_block_kernel,
                dim=entry.body_indices.shape[0],
                inputs=[
                    entry.body_indices,
                    buf.body_effective_mass_local,
                    buf.body_effective_inertia_local,
                    buf.body_effective_mass,
                    buf.body_effective_inertia_scalar,
                ],
                device=self.model.device,
            )
            return

        try:
            entry.solver.coupling_eval_effective_mass(
                buf.body_endpoint_kind,
                buf.body_endpoint_index,
                buf.body_endpoint_local_pos,
                buf.body_effective_mass_local,
            )
        except NotImplementedError:
            if raise_on_unsupported:
                raise
            self._mark_effective_mass_unsupported(entry, CouplingEndpointKind.BODY)
            return
        wp.launch(
            scatter_effective_mass_kernel,
            dim=entry.body_indices.shape[0],
            inputs=[entry.body_indices, buf.body_effective_mass_local, buf.body_effective_mass],
            device=self.model.device,
        )

    def _populate_admm_particle_effective_mass_buffer(
        self,
        entry: SolverEntry,
        buf: _AdmmBuffers,
        *,
        raise_on_unsupported: bool = True,
    ) -> None:
        if (
            buf.particle_effective_mass is None
            or buf.particle_endpoint_index is None
            or buf.particle_endpoint_index.shape[0] == 0
        ):
            return

        try:
            entry.solver.coupling_eval_effective_mass(
                buf.particle_endpoint_kind,
                buf.particle_endpoint_index,
                buf.particle_endpoint_local_pos,
                buf.particle_effective_mass_local,
            )
        except NotImplementedError:
            if raise_on_unsupported:
                raise
            self._mark_effective_mass_unsupported(entry, CouplingEndpointKind.PARTICLE)
            return

        wp.launch(
            scatter_effective_mass_kernel,
            dim=entry.particle_indices.shape[0],
            inputs=[entry.particle_indices, buf.particle_effective_mass_local, buf.particle_effective_mass],
            device=self.model.device,
        )

    def _setup_admm_contact_specs(self, coupling: SolverCoupledADMM.Config) -> None:
        """Populate dynamic ADMM contact specs from configured contact pairs."""
        if not coupling.contact_pairs:
            return

        # Discover all candidate specs from model state (one rigid-rigid/rigid-particle
        # /particle-particle entry per cross-owner combination), then keep only those
        # whose owner pair appears in the user's ContactPair list.
        pair_by_owners: dict[frozenset[str], SolverCoupledADMM.ContactPair] = {}
        for pair in coupling.contact_pairs:
            if pair.source == pair.destination:
                raise ValueError(f"ADMM ContactPair requires distinct source and destination, got {pair.source!r}")
            if pair.source not in self._entries:
                raise ValueError(f"Unknown ADMM ContactPair source {pair.source!r}")
            if pair.destination not in self._entries:
                raise ValueError(f"Unknown ADMM ContactPair destination {pair.destination!r}")
            key = frozenset({pair.source, pair.destination})
            if key in pair_by_owners:
                raise ValueError(f"Duplicate ADMM ContactPair for entries {pair.source!r} and {pair.destination!r}")
            pair_by_owners[key] = pair

        rp_specs = self._discover_rigid_particle_contact_specs()
        rr_specs = self._discover_rigid_rigid_contact_specs()
        pp_specs = self._discover_particle_particle_contact_specs()

        def matching_pair(owner_a: str, owner_b: str):
            return pair_by_owners.get(frozenset({owner_a, owner_b}))

        self._admm_rigid_particle_contact_specs = [
            spec for spec in rp_specs if matching_pair(spec.body_owner, spec.particle_owner) is not None
        ]
        self._admm_rigid_rigid_contact_specs = [
            spec for spec in rr_specs if matching_pair(spec.owner_a, spec.owner_b) is not None
        ]
        self._admm_particle_particle_contact_specs = [
            spec for spec in pp_specs if matching_pair(spec.owner_a, spec.owner_b) is not None
        ]

    @classmethod
    def auto_detect_contact_pairs(
        cls,
        entries: Sequence[SolverCoupled.Entry],
    ) -> list[SolverCoupledADMM.ContactPair]:
        """Return ContactPair entries for every cross-owner interface.

        Mirrors the prior auto-detection behavior: a pair is emitted for every
        distinct combination of entries. Friction is read from
        ``shape_material_mu`` and ``Model.particle_mu`` at contact-fill time.

        Args:
            entries: Sub-solver entries that will be passed to
                :class:`SolverCoupledADMM`.
        """
        names = [e.name for e in entries]
        pairs: list[SolverCoupledADMM.ContactPair] = []
        for i, a in enumerate(names):
            for b in names[i + 1 :]:
                pairs.append(
                    cls.ContactPair(
                        source=a,
                        destination=b,
                    )
                )
        return pairs

    def _shape_flagged(self, shape_flags, shape: int, flag: ShapeFlags) -> bool:
        if shape_flags is None:
            return False
        return bool(int(shape_flags[shape]) & int(flag))

    def _rigid_rigid_spec_shape_pairs(self, spec: _AdmmRigidRigidContactSpec) -> list[tuple[int, int]]:
        if spec.shape_pairs is not None:
            return [(int(a), int(b)) for a, b in spec.shape_pairs]
        shapes_a = [] if spec.shapes_a is None else [int(shape) for shape in spec.shapes_a]
        shapes_b = [] if spec.shapes_b is None else [int(shape) for shape in spec.shapes_b]
        return [(shape_a, shape_b) for shape_a in shapes_a for shape_b in shapes_b]

    def _admm_rigid_contact_capacity(self) -> int:
        """Return rigid ADMM contact row capacity for exact shape pairs."""
        return sum(8 * len(self._rigid_rigid_spec_shape_pairs(spec)) for spec in self._admm_rigid_rigid_contact_specs)

    def _build_admm_rigid_shape_pair_array(self) -> wp.array:
        """Build the exact rigid shape pairs needed by ADMM contacts."""
        pairs: list[tuple[int, int]] = []
        seen: set[tuple[int, int]] = set()
        for spec in self._admm_rigid_rigid_contact_specs:
            for shape_a, shape_b in self._rigid_rigid_spec_shape_pairs(spec):
                pair = (int(shape_a), int(shape_b))
                if pair in seen:
                    continue
                seen.add(pair)
                pairs.append(pair)

        if not pairs:
            return wp.zeros(0, dtype=wp.vec2i, device=self.model.device)
        return wp.array(np.asarray(pairs, dtype=np.int32), dtype=wp.vec2i, device=self.model.device)

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
        body_world = self.model.body_world.numpy() if self.model.body_world is not None else []
        entry_order = {name: i for i, name in enumerate(self._entries)}
        grouped: dict[tuple[str, str], dict[str, object]] = {}
        for pair in self.model.shape_contact_pairs.numpy():
            shape_a = int(pair[0])
            shape_b = int(pair[1])
            body_a = int(shape_body[shape_a])
            body_b = int(shape_body[shape_b])
            owner_a = self._entry_name_for_body(body_a)
            owner_b = self._entry_name_for_body(body_b)
            if owner_a is None or owner_b is None or owner_a == owner_b:
                continue
            world_a = int(body_world[body_a]) if body_a >= 0 and len(body_world) > body_a else -1
            world_b = int(body_world[body_b]) if body_b >= 0 and len(body_world) > body_b else -1
            if world_a != world_b:
                raise ValueError(
                    "ADMM rigid ContactPair requires source and destination bodies to live in the same world; "
                    f"shape pair ({shape_a}, {shape_b}) references bodies {body_a} in world {world_a} "
                    f"and {body_b} in world {world_b}."
                )
            if entry_order[owner_b] < entry_order[owner_a]:
                owner_a, owner_b = owner_b, owner_a
                shape_a, shape_b = shape_b, shape_a
            bucket = grouped.setdefault((owner_a, owner_b), {"shapes_a": set(), "shapes_b": set(), "pairs": []})
            shapes_a = bucket["shapes_a"]
            shapes_b = bucket["shapes_b"]
            pairs = bucket["pairs"]
            shapes_a.add(shape_a)
            shapes_b.add(shape_b)
            pairs.append((shape_a, shape_b))

        return [
            _AdmmRigidRigidContactSpec(
                owner_a=owner_a,
                owner_b=owner_b,
                shapes_a=tuple(sorted(bucket["shapes_a"])),
                shapes_b=tuple(sorted(bucket["shapes_b"])),
                shape_pairs=tuple(bucket["pairs"]),
            )
            for (owner_a, owner_b), bucket in grouped.items()
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
    def _quat_from_x_axis(direction) -> wp.quat:
        direction = wp.vec3(float(direction[0]), float(direction[1]), float(direction[2]))
        length = float(wp.length(direction))
        if length <= 0.0:
            raise ValueError("Cannot build a revolute ADMM frame from a zero axis")
        return quat_between_vectors_robust(wp.vec3(1.0, 0.0, 0.0), direction / length)

    @classmethod
    def _revolute_axis_frames_from_rows(
        cls,
        joint_X_p_row,
        joint_X_c_row,
        axis_parent: np.ndarray,
    ) -> tuple[wp.transform, wp.transform]:
        q_parent = wp.quat(*(float(value) for value in joint_X_p_row[3:7]))
        q_child = wp.quat(*(float(value) for value in joint_X_c_row[3:7]))
        parent_to_child = wp.mul(wp.quat_inverse(q_child), q_parent)
        axis_child = wp.quat_rotate(
            parent_to_child,
            wp.vec3(float(axis_parent[0]), float(axis_parent[1]), float(axis_parent[2])),
        )
        frame_child = wp.transform(
            wp.vec3(float(joint_X_c_row[0]), float(joint_X_c_row[1]), float(joint_X_c_row[2])),
            cls._quat_from_x_axis(axis_child),
        )
        frame_parent = wp.transform(
            wp.vec3(float(joint_X_p_row[0]), float(joint_X_p_row[1]), float(joint_X_p_row[2])),
            cls._quat_from_x_axis(axis_parent),
        )
        return frame_child, frame_parent

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
                "leave it to SolverCoupledADMM so the constraint is not applied twice"
            )
        return child_entry, parent_entry

    def _build_admm_joint_groups(self, coupling: SolverCoupledADMM.Config) -> None:
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
            body_global_ids_a = [item[0] for item in items]
            body_global_ids_b = [item[2] for item in items]
            body_ids_a = [self._body_local_id(entry_name_a, body) for body in body_global_ids_a]
            points_a = [wp.vec3(*item[1]) for item in items]
            body_ids_b = [self._body_local_id(entry_name_b, body) for body in body_global_ids_b]
            points_b = [wp.vec3(*item[3]) for item in items]
            kappa = [item[4] for item in items]
            damping = [item[5] for item in items]
            W = self._compute_interface_weights(
                body_global_ids_a,
                self._admm_buffers[entry_name_a].body_effective_mass,
                body_global_ids_b,
                self._admm_buffers[entry_name_b].body_effective_mass,
            )

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
                    W=W,
                    u=wp.zeros(n, dtype=wp.vec3, device=device),
                    lambda_=wp.zeros(n, dtype=wp.vec3, device=device),
                    Jv=wp.zeros(n, dtype=wp.vec3, device=device),
                    u_target=wp.zeros(n, dtype=wp.vec3, device=device),
                )
            )

        for (entry_name_a, entry_name_b), items in angular_items.items():
            body_global_ids_a = [item[0] for item in items]
            body_global_ids_b = [item[2] for item in items]
            body_ids_a = [self._body_local_id(entry_name_a, body) for body in body_global_ids_a]
            frames_a = [item[1] for item in items]
            body_ids_b = [self._body_local_id(entry_name_b, body) for body in body_global_ids_b]
            frames_b = [item[3] for item in items]
            kappa = [item[4] for item in items]
            damping = [item[5] for item in items]
            W = self._compute_interface_weights(
                body_global_ids_a,
                self._admm_buffers[entry_name_a].body_effective_inertia_scalar,
                body_global_ids_b,
                self._admm_buffers[entry_name_b].body_effective_inertia_scalar,
            )
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
                    W=W,
                    u=wp.zeros(n, dtype=wp.vec3, device=device),
                    lambda_=wp.zeros(n, dtype=wp.vec3, device=device),
                    Jv=wp.zeros(n, dtype=wp.vec3, device=device),
                    u_target=wp.zeros(n, dtype=wp.vec3, device=device),
                )
            )

        for (entry_name_a, entry_name_b), items in revolute_angular_items.items():
            body_global_ids_a = [item[0] for item in items]
            body_global_ids_b = [item[2] for item in items]
            body_ids_a = [self._body_local_id(entry_name_a, body) for body in body_global_ids_a]
            frames_a = [item[1] for item in items]
            body_ids_b = [self._body_local_id(entry_name_b, body) for body in body_global_ids_b]
            frames_b = [item[3] for item in items]
            kappa = [item[4] for item in items]
            damping = [item[5] for item in items]
            W = self._compute_interface_weights(
                body_global_ids_a,
                self._admm_buffers[entry_name_a].body_effective_inertia_scalar,
                body_global_ids_b,
                self._admm_buffers[entry_name_b].body_effective_inertia_scalar,
            )
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
                    W=W,
                    u=wp.zeros(n, dtype=wp.vec3, device=device),
                    lambda_=wp.zeros(n, dtype=wp.vec3, device=device),
                    Jv=wp.zeros(n, dtype=wp.vec3, device=device),
                    u_target=wp.zeros(n, dtype=wp.vec3, device=device),
                )
            )

        for (entry_name_a, entry_name_b), items in angular_friction_items.items():
            body_global_ids_a = [item[0] for item in items]
            body_global_ids_b = [item[2] for item in items]
            body_ids_a = [self._body_local_id(entry_name_a, body) for body in body_global_ids_a]
            frames_a = [item[1] for item in items]
            body_ids_b = [self._body_local_id(entry_name_b, body) for body in body_global_ids_b]
            friction = [wp.vec3(*item[3]) for item in items]
            W = self._compute_interface_weights(
                body_global_ids_a,
                self._admm_buffers[entry_name_a].body_effective_inertia_scalar,
                body_global_ids_b,
                self._admm_buffers[entry_name_b].body_effective_inertia_scalar,
            )
            n = len(items)
            self._admm_rr_angular_friction_groups.append(
                _AdmmRigidRigidAngularFrictionGroup(
                    body_entry_name_a=entry_name_a,
                    body_entry_name_b=entry_name_b,
                    body_ids_a=wp.array(body_ids_a, dtype=int, device=device),
                    frame_a=wp.array(frames_a, dtype=wp.transform, device=device),
                    body_ids_b=wp.array(body_ids_b, dtype=int, device=device),
                    friction=wp.array(friction, dtype=wp.vec3, device=device),
                    W=W,
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
                "ADMM body-particle attachments require SolverCoupledADMM.register_custom_attributes(builder) "
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
            body_global_ids = [item[0] for item in items]
            particle_ids = [item[2] for item in items]
            body_ids = [self._body_local_id(body_entry, body) for body in body_global_ids]
            points = [wp.vec3(*item[1]) for item in items]
            kappa = [item[3] for item in items]
            damping = [item[4] for item in items]
            W = self._compute_interface_weights(
                body_global_ids,
                self._admm_buffers[body_entry].body_effective_mass,
                particle_ids,
                self._admm_buffers[particle_entry].particle_effective_mass,
            )

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
                    W=W,
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
        iters = int(coupling.iterations)
        self._refresh_collision_contact_groups(state_in)
        if float(coupling.gamma) > 0.0:
            self._refresh_admm_proximal_masks()
            self._refresh_admm_proximal_view_overrides(refresh_supported_solvers=True)

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

            self._accumulate_admm_forces(k, dt, refresh_jv=k == 0, initialize_contact_u=k == 0)

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
            rigid_contact_match_index = self._admm_internal_contacts.rigid_contact_match_index
            use_contact_matching = rigid_contact_match_index is not None
            if rigid_contact_match_index is None:
                rigid_contact_match_index = group.prev_contact_active
            wp.launch(
                contact_rr_clear_contact_snapshot_kernel,
                dim=group.prev_contact_active.shape[0],
                inputs=[
                    group.prev_contact_active,
                    group.prev_contact_lambda,
                    group.prev_contact_W,
                ],
                device=self.model.device,
            )
            wp.launch(
                contact_rr_snapshot_by_contact_kernel,
                dim=group.count,
                inputs=[
                    group.active_count,
                    group.contact_ids,
                    group.active,
                    group.W,
                    group.lambda_,
                    group.prev_contact_active,
                    group.prev_contact_lambda,
                    group.prev_contact_W,
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
                    group.offset_a,
                    group.body_ids_b,
                    group.point_b,
                    group.offset_b,
                    group.contact_ids,
                    group.shape_ids_a,
                    group.shape_ids_b,
                    group.point_ids,
                    group.active,
                    group.normal,
                    group.W,
                    group.friction,
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
                    self._admm_internal_contacts.rigid_contact_offset0,
                    self._admm_internal_contacts.rigid_contact_offset1,
                    self._admm_internal_contacts.rigid_contact_normal,
                    self._admm_internal_contacts.rigid_contact_point_id,
                    rigid_contact_match_index,
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
                    1 if use_contact_matching else 0,
                    float(self._coupling.contact_matching_force_scale),
                    int(group.count),
                    group.active_count,
                    group.active_count_max,
                    group.prev_contact_active,
                    group.prev_contact_lambda,
                    group.prev_contact_W,
                ],
                outputs=[
                    group.body_ids_a,
                    group.point_a,
                    group.offset_a,
                    group.body_ids_b,
                    group.point_b,
                    group.offset_b,
                    group.contact_ids,
                    group.shape_ids_a,
                    group.shape_ids_b,
                    group.point_ids,
                    group.active,
                    group.normal,
                    group.W,
                    group.friction,
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
                        group.W,
                        group.lambda_,
                    ],
                    outputs=[
                        group.prev_body_ids,
                        group.prev_particle_ids,
                        group.prev_shape_ids,
                        group.prev_active,
                        group.prev_W,
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
                        group.W,
                        group.friction,
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
                        body_buf.body_effective_mass,
                        particle_buf.particle_effective_mass,
                        self.model.shape_material_mu,
                        float(self.model.particle_mu),
                        int(group.count),
                        group.active_count,
                        group.active_count_max,
                        group.prev_particle_ids,
                        group.prev_shape_ids,
                        group.prev_active,
                        group.prev_W,
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
                        group.W,
                        group.friction,
                        group.lambda_,
                    ],
                    device=self.model.device,
                )

        if self._admm_dynamic_pp_contact_groups:
            with wp.ScopedDevice(self.model.device):
                self.model.particle_grid.build(
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
                    group.W,
                    group.lambda_,
                ],
                outputs=[
                    group.prev_particle_ids_a,
                    group.prev_particle_ids_b,
                    group.prev_active,
                    group.prev_W,
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
                    group.W,
                    group.friction,
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
                    self.model.particle_grid.id,
                    state_in.particle_q,
                    self.model.particle_radius,
                    self.model.particle_flags,
                    self.model.particle_world,
                    group.particle_mask_a,
                    group.particle_mask_b,
                    float(group.query_radius),
                    int(contact_stream.capacity),
                    contact_stream.count,
                    contact_stream.count_max,
                ],
                outputs=[
                    contact_stream.particle_a,
                    contact_stream.particle_b,
                    contact_stream.normal,
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
                    self._admm_buffers[group.particle_entry_name_a].particle_effective_mass,
                    self._admm_buffers[group.particle_entry_name_b].particle_effective_mass,
                    float(self.model.particle_mu),
                    int(group.count),
                    group.active_count,
                    group.active_count_max,
                    group.prev_particle_ids_a,
                    group.prev_particle_ids_b,
                    group.prev_active,
                    group.prev_W,
                    group.prev_lambda,
                ],
                outputs=[
                    group.particle_ids_a,
                    group.particle_ids_b,
                    group.active,
                    group.normal,
                    group.W,
                    group.friction,
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
        shape_body = self.model.shape_body.numpy() if self.model.shape_body is not None else []

        for spec in self._admm_rigid_rigid_contact_specs:
            shapes_a = self._shape_contact_candidates(spec.owner_a, spec.shapes_a)
            shapes_b = self._shape_contact_candidates(spec.owner_b, spec.shapes_b)
            body_candidates_a = {int(shape_body[shape]) for shape in shapes_a if 0 <= shape < len(shape_body)}
            body_candidates_b = {int(shape_body[shape]) for shape in shapes_b if 0 <= shape < len(shape_body)}
            # Primitive pairs may emit a small manifold rather than one row.
            capacity = 8 * len(self._rigid_rigid_spec_shape_pairs(spec))
            contact_capacity = self._admm_rigid_contact_capacity()
            if capacity == 0:
                continue
            self._require_effective_mass(spec.owner_a, CouplingEndpointKind.BODY)
            self._require_effective_mass(spec.owner_b, CouplingEndpointKind.BODY)
            body_global_ids_a = []
            body_global_ids_b = []
            candidate_body_ids_a = []
            candidate_body_ids_b = []
            for shape_a, shape_b in self._rigid_rigid_spec_shape_pairs(spec):
                body_a = int(shape_body[shape_a])
                body_b = int(shape_body[shape_b])
                body_global_ids_a.append(body_a)
                body_global_ids_b.append(body_b)
                candidate_body_ids_a.append(self._body_local_id(spec.owner_a, body_a))
                candidate_body_ids_b.append(self._body_local_id(spec.owner_b, body_b))
            candidate_W = self._compute_interface_weights(
                body_global_ids_a,
                self._admm_buffers[spec.owner_a].body_effective_mass,
                body_global_ids_b,
                self._admm_buffers[spec.owner_b].body_effective_mass,
            )

            groups.append(
                _AdmmRigidRigidContactGroup(
                    body_entry_name_a=spec.owner_a,
                    body_entry_name_b=spec.owner_b,
                    body_ids_a=wp.zeros(capacity, dtype=int, device=device),
                    point_a=wp.zeros(capacity, dtype=wp.vec3, device=device),
                    offset_a=wp.zeros(capacity, dtype=wp.vec3, device=device),
                    body_ids_b=wp.zeros(capacity, dtype=int, device=device),
                    point_b=wp.zeros(capacity, dtype=wp.vec3, device=device),
                    offset_b=wp.zeros(capacity, dtype=wp.vec3, device=device),
                    contact_ids=wp.full(capacity, -1, dtype=int, device=device),
                    normal=wp.zeros(capacity, dtype=wp.vec3, device=device),
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
                    prev_contact_active=wp.zeros(contact_capacity, dtype=int, device=device),
                    prev_contact_lambda=wp.zeros(contact_capacity, dtype=wp.vec3, device=device),
                    prev_contact_W=wp.zeros(contact_capacity, dtype=float, device=device),
                    body_mask_a=self._make_int_mask_array(self.model.body_count, body_candidates_a),
                    body_mask_b=self._make_int_mask_array(self.model.body_count, body_candidates_b),
                    shape_mask_a=self._make_int_mask_array(self.model.shape_count, set(shapes_a)),
                    shape_mask_b=self._make_int_mask_array(self.model.shape_count, set(shapes_b)),
                    candidate_body_ids_a=wp.array(candidate_body_ids_a, dtype=int, device=device),
                    candidate_body_ids_b=wp.array(candidate_body_ids_b, dtype=int, device=device),
                    candidate_W=candidate_W,
                )
            )

        return groups

    def _build_collision_rigid_particle_contact_groups(self) -> list[_AdmmRigidParticleContactGroup]:
        device = self.model.device
        groups = []
        shape_body = self.model.shape_body.numpy() if self.model.shape_body is not None else []

        for spec_idx, spec in enumerate(self._admm_rigid_particle_contact_specs):
            particle_candidates = sorted(self._entry_particle_sets[spec.particle_owner])
            owner_body_candidates = set(self._entry_body_sets[spec.body_owner])
            shape_filter = self._admm_rigid_particle_shape_filters.get(spec_idx)
            if shape_filter is None:
                shape_candidates = [
                    shape for shape in range(self.model.shape_count) if int(shape_body[shape]) in owner_body_candidates
                ]
            else:
                shape_candidates = sorted(shape_filter)
            body_candidates = {int(shape_body[shape]) for shape in shape_candidates if 0 <= shape < len(shape_body)}

            capacity = len(particle_candidates) * len(shape_candidates)
            if capacity == 0:
                continue
            self._require_effective_mass(spec.body_owner, CouplingEndpointKind.BODY)
            self._require_effective_mass(spec.particle_owner, CouplingEndpointKind.PARTICLE)
            body_global_ids = []
            candidate_body_ids = []
            candidate_particle_ids = []
            for particle in particle_candidates:
                for shape in shape_candidates:
                    body = int(shape_body[shape])
                    body_global_ids.append(body)
                    candidate_body_ids.append(self._body_local_id(spec.body_owner, body))
                    candidate_particle_ids.append(int(particle))
            candidate_W = self._compute_interface_weights(
                body_global_ids,
                self._admm_buffers[spec.body_owner].body_effective_mass,
                candidate_particle_ids,
                self._admm_buffers[spec.particle_owner].particle_effective_mass,
            )

            groups.append(
                _AdmmRigidParticleContactGroup(
                    body_entry_name=spec.body_owner,
                    particle_entry_name=spec.particle_owner,
                    body_ids=wp.zeros(capacity, dtype=int, device=device),
                    point_body=wp.zeros(capacity, dtype=wp.vec3, device=device),
                    particle_ids=wp.zeros(capacity, dtype=int, device=device),
                    normal=wp.zeros(capacity, dtype=wp.vec3, device=device),
                    body_sign=wp.full(capacity, -1, dtype=int, device=device),
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
                    prev_body_ids=wp.zeros(capacity, dtype=int, device=device),
                    prev_particle_ids=wp.zeros(capacity, dtype=int, device=device),
                    prev_shape_ids=wp.full(capacity, -1, dtype=int, device=device),
                    prev_active=wp.zeros(capacity, dtype=int, device=device),
                    prev_W=wp.zeros(capacity, dtype=float, device=device),
                    prev_lambda=wp.zeros(capacity, dtype=wp.vec3, device=device),
                    candidate_body_ids=wp.array(candidate_body_ids, dtype=int, device=device),
                    candidate_particle_ids=wp.array(candidate_particle_ids, dtype=int, device=device),
                    candidate_W=candidate_W,
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
            candidate_particle_ids_a = []
            candidate_particle_ids_b = []
            for particle_a in particles_a:
                for particle_b in particles_b:
                    candidate_particle_ids_a.append(int(particle_a))
                    candidate_particle_ids_b.append(int(particle_b))
            candidate_W = self._compute_interface_weights(
                candidate_particle_ids_a,
                self._admm_buffers[spec.owner_a].particle_effective_mass,
                candidate_particle_ids_b,
                self._admm_buffers[spec.owner_b].particle_effective_mass,
            )

            query_radius = 2.0 * float(self.model.particle_max_radius)
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
                    query_radius=query_radius,
                    prev_particle_ids_a=wp.zeros(capacity, dtype=int, device=device),
                    prev_particle_ids_b=wp.zeros(capacity, dtype=int, device=device),
                    prev_active=wp.zeros(capacity, dtype=int, device=device),
                    prev_W=wp.zeros(capacity, dtype=float, device=device),
                    prev_lambda=wp.zeros(capacity, dtype=wp.vec3, device=device),
                    candidate_particle_ids_a=wp.array(candidate_particle_ids_a, dtype=int, device=device),
                    candidate_particle_ids_b=wp.array(candidate_particle_ids_b, dtype=int, device=device),
                    candidate_W=candidate_W,
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
        for group in self._admm_dynamic_rr_contact_groups:
            if group.count == 0:
                continue
            entry_a = self._entries[group.body_entry_name_a]
            entry_b = self._entries[group.body_entry_name_b]
            wp.launch(
                contact_rr_compute_u_min_kernel,
                dim=group.count,
                inputs=[
                    group.active_count,
                    group.body_ids_a,
                    group.point_a,
                    group.offset_a,
                    group.body_ids_b,
                    group.point_b,
                    group.offset_b,
                    group.normal,
                    entry_a.state_0.body_q,
                    entry_b.state_0.body_q,
                    float(coupling.baumgarte),
                    float(dt),
                ],
                outputs=[group.u_min],
                device=self.model.device,
            )
        for group in self._admm_dynamic_rp_contact_groups:
            if group.count == 0:
                continue
            body_entry = self._entries[group.body_entry_name]
            particle_entry = self._entries[group.particle_entry_name]
            wp.launch(
                contact_rp_compute_u_min_kernel,
                dim=group.count,
                inputs=[
                    group.active_count,
                    group.body_ids,
                    group.point_body,
                    group.particle_ids,
                    group.normal,
                    group.body_sign,
                    body_entry.state_0.body_q,
                    particle_entry.state_0.particle_q,
                    self.model.particle_radius,
                    float(coupling.baumgarte),
                    float(dt),
                ],
                outputs=[group.u_min],
                device=self.model.device,
            )
        for group in self._admm_dynamic_pp_contact_groups:
            if group.count == 0:
                continue
            entry_a = self._entries[group.particle_entry_name_a]
            entry_b = self._entries[group.particle_entry_name_b]
            wp.launch(
                contact_pp_compute_u_min_kernel,
                dim=group.count,
                inputs=[
                    group.active_count,
                    group.particle_ids_a,
                    group.particle_ids_b,
                    group.normal,
                    entry_a.state_0.particle_q,
                    entry_b.state_0.particle_q,
                    self.model.particle_radius,
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
        dt: float,
    ) -> None:
        device = self.model.device
        flags = int(StateFlags.NONE)
        if buf.body_qd_n is not None and buf.body_proximal_mass is not None and buf.body_proximal_inertia is not None:
            wp.launch(
                velocity_proximal_shift_body_lumped_kernel,
                dim=buf.body_qd_n.shape[0],
                inputs=[
                    buf.body_qd_n,
                    buf.body_qd_k,
                    buf.body_proximal_mass,
                    buf.body_proximal_inertia,
                    entry.view.body_mass,
                    entry.view.body_inertia,
                    entry.state_0.body_qd,
                ],
                device=device,
            )
            flags |= StateFlags.BODY_QD
        if buf.particle_qd_n is not None and buf.particle_proximal_mass is not None:
            wp.launch(
                velocity_proximal_shift_particle_lumped_kernel,
                dim=buf.particle_qd_n.shape[0],
                inputs=[
                    buf.particle_qd_n,
                    buf.particle_qd_k,
                    buf.particle_proximal_mass,
                    entry.view.particle_mass,
                    entry.state_0.particle_qd,
                ],
                device=device,
            )
            flags |= StateFlags.PARTICLE_QD
        if buf.joint_qd_n is not None and buf.joint_qd_n.shape[0] > 0 and buf.joint_qd_proximal_factor is not None:
            wp.launch(
                velocity_proximal_shift_joint_lumped_kernel,
                dim=buf.joint_qd_n.shape[0],
                inputs=[
                    buf.joint_qd_n,
                    buf.joint_qd_k,
                    buf.joint_qd_proximal_factor,
                    entry.state_0.joint_qd,
                ],
                device=device,
            )
            flags |= StateFlags.JOINT_QD
        if flags:
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
        flags = int(StateFlags.NONE)

        if buf.body_q_n is not None:
            wp.copy(entry.state_0.body_q, buf.body_q_n)
            wp.copy(entry.state_0.body_qd, buf.body_qd_n)
            flags |= StateFlags.BODY

        if buf.particle_q_n is not None:
            wp.copy(entry.state_0.particle_q, buf.particle_q_n)
            wp.copy(entry.state_0.particle_qd, buf.particle_qd_n)
            flags |= StateFlags.PARTICLE

        if buf.joint_q_n is not None:
            wp.copy(entry.state_0.joint_q, buf.joint_q_n)
            wp.copy(entry.state_0.joint_qd, buf.joint_qd_n)
            flags |= StateFlags.JOINT

        self._notify_input_state_update(entry, flags, dt=dt, iteration_restart=bool(iteration_restart) and bool(flags))

        if apply_proximal:
            self._apply_admm_velocity_proximal_shift(entry, buf, dt)

        if buf.body_f is not None:
            if state_in.body_f is not None:
                wp.launch(
                    _copy_mapped_spatial_vector,
                    dim=entry.body_local_to_global.shape[0],
                    inputs=[entry.body_local_to_global, state_in.body_f, buf.body_f],
                    device=self.model.device,
                )
            else:
                buf.body_f.zero_()
            if apply_proximal and buf.body_proximal_mass is not None and entry.view.body_count > 0:
                wp.launch(
                    body_gravity_compensation_lumped_kernel,
                    dim=entry.view.body_count,
                    inputs=[
                        buf.body_proximal_mass,
                        entry.view.body_inv_mass,
                        entry.body_gravity_acceleration,
                    ],
                    outputs=[buf.body_f],
                    device=self.model.device,
                )
        if buf.particle_f is not None:
            if state_in.particle_f is not None:
                wp.launch(
                    _copy_mapped_vec3,
                    dim=entry.particle_local_to_global.shape[0],
                    inputs=[entry.particle_local_to_global, state_in.particle_f, buf.particle_f],
                    device=self.model.device,
                )
            else:
                buf.particle_f.zero_()
            if apply_proximal and buf.particle_proximal_mass is not None and entry.view.particle_count > 0:
                wp.launch(
                    particle_gravity_compensation_lumped_kernel,
                    dim=entry.view.particle_count,
                    inputs=[
                        buf.particle_proximal_mass,
                        entry.view.particle_inv_mass,
                        entry.view.particle_flags,
                        entry.particle_gravity_acceleration,
                    ],
                    outputs=[buf.particle_f],
                    device=self.model.device,
                )

    def _apply_admm_force_inputs(self, entry: SolverEntry, buf: _AdmmBuffers, dt: float) -> None:
        if entry.body_indices.shape[0] > 0:
            self._set_local_body_force_input(entry, buf.body_f, dt=dt)
        if entry.particle_indices.shape[0] > 0:
            self._set_local_particle_force_input(entry, buf.particle_f, dt=dt)

    def _update_admm_contact_u(self, group: _AdmmContactGroup) -> None:
        wp.launch(
            contact_u_update_kernel,
            dim=group.count,
            inputs=[
                group.active_count,
                group.u_min,
                group.W,
                float(self._coupling.rho),
                group.friction,
                group.normal,
                group.lambda_,
                group.Jv,
            ],
            outputs=[group.u],
            device=self.model.device,
        )

    def _update_admm_quadratic_dual(self, group: _AdmmQuadraticGroup) -> None:
        wp.launch(
            u_update_quadratic_kernel,
            dim=group.count,
            inputs=[
                group.kappa,
                group.damping,
                group.W,
                float(self._coupling.rho),
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
            inputs=[float(self._coupling.rho), group.W, group.u, group.Jv],
            outputs=[group.lambda_],
            device=self.model.device,
        )

    def _update_admm_contact_dual(self, group: _AdmmContactGroup) -> None:
        self._update_admm_contact_u(group)
        wp.launch(
            contact_lambda_update_kernel,
            dim=group.count,
            inputs=[group.active_count, float(self._coupling.rho), group.W, group.u, group.Jv],
            outputs=[group.lambda_],
            device=self.model.device,
        )

    def _accumulate_admm_forces(
        self,
        iteration_k: int,
        dt: float,
        *,
        refresh_jv: bool,
        initialize_contact_u: bool,
    ) -> None:
        del iteration_k
        coupling = self._coupling
        for group in self._admm_rr_groups:
            if group.count == 0:
                continue
            entry_a = self._entries[group.body_entry_name_a]
            entry_b = self._entries[group.body_entry_name_b]
            buf_a = self._admm_buffers[group.body_entry_name_a]
            buf_b = self._admm_buffers[group.body_entry_name_b]
            if refresh_jv:
                wp.launch(
                    attach_rr_compute_Jv_kernel,
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
                attach_rr_accumulate_forces_kernel,
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
            if refresh_jv:
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
            if refresh_jv:
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
            if refresh_jv:
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
            if refresh_jv:
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
        for group in self._admm_dynamic_rr_contact_groups:
            if group.count == 0:
                continue
            entry_a = self._entries[group.body_entry_name_a]
            entry_b = self._entries[group.body_entry_name_b]
            buf_a = self._admm_buffers[group.body_entry_name_a]
            buf_b = self._admm_buffers[group.body_entry_name_b]
            if refresh_jv:
                wp.launch(
                    contact_rr_compute_Jv_kernel,
                    dim=group.count,
                    inputs=[
                        group.active_count,
                        group.body_ids_a,
                        group.point_a,
                        group.offset_a,
                        group.body_ids_b,
                        group.point_b,
                        group.offset_b,
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
            if initialize_contact_u:
                self._update_admm_contact_u(group)
            wp.launch(
                contact_rr_accumulate_forces_kernel,
                dim=group.count,
                inputs=[
                    group.active_count,
                    group.body_ids_a,
                    group.point_a,
                    group.offset_a,
                    group.body_ids_b,
                    group.point_b,
                    group.offset_b,
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
        for group in self._admm_dynamic_rp_contact_groups:
            if group.count == 0:
                continue
            body_entry = self._entries[group.body_entry_name]
            body_buf = self._admm_buffers[group.body_entry_name]
            particle_buf = self._admm_buffers[group.particle_entry_name]
            if refresh_jv:
                wp.launch(
                    contact_rp_compute_Jv_kernel,
                    dim=group.count,
                    inputs=[
                        group.active_count,
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
            if initialize_contact_u:
                self._update_admm_contact_u(group)
            wp.launch(
                contact_rp_accumulate_forces_kernel,
                dim=group.count,
                inputs=[
                    group.active_count,
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
        for group in self._admm_dynamic_pp_contact_groups:
            if group.count == 0:
                continue
            buf_a = self._admm_buffers[group.particle_entry_name_a]
            buf_b = self._admm_buffers[group.particle_entry_name_b]
            if refresh_jv:
                wp.launch(
                    contact_pp_compute_Jv_kernel,
                    dim=group.count,
                    inputs=[
                        group.active_count,
                        group.particle_ids_a,
                        group.particle_ids_b,
                        buf_a.particle_qd_k,
                        buf_b.particle_qd_k,
                    ],
                    outputs=[group.Jv],
                    device=self.model.device,
                )
            if initialize_contact_u:
                self._update_admm_contact_u(group)
            wp.launch(
                contact_pp_accumulate_forces_kernel,
                dim=group.count,
                inputs=[
                    group.active_count,
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
                attach_rr_compute_Jv_kernel,
                dim=group.count,
                inputs=[
                    group.body_ids_a,
                    group.point_a,
                    group.body_ids_b,
                    group.point_b,
                    entry_a.state_0.body_q,
                    entry_a.view.body_com,
                    entry_a.state_1.body_qd,
                    entry_b.state_0.body_q,
                    entry_b.view.body_com,
                    entry_b.state_1.body_qd,
                ],
                outputs=[group.Jv],
                device=self.model.device,
            )
            self._update_admm_quadratic_dual(group)
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
            self._update_admm_quadratic_dual(group)
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
                    entry_a.state_0.body_q,
                    entry_a.state_1.body_qd,
                    entry_b.state_1.body_qd,
                ],
                outputs=[group.Jv],
                device=self.model.device,
            )
            self._update_admm_quadratic_dual(group)
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
                    entry_a.state_0.body_q,
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
                    body_entry.state_0.body_q,
                    body_entry.view.body_com,
                    body_entry.state_1.body_qd,
                    particle_entry.state_1.particle_qd,
                ],
                outputs=[group.Jv],
                device=self.model.device,
            )
            self._update_admm_quadratic_dual(group)
        for group in self._admm_dynamic_rr_contact_groups:
            if group.count == 0:
                continue
            entry_a = self._entries[group.body_entry_name_a]
            entry_b = self._entries[group.body_entry_name_b]
            wp.launch(
                contact_rr_compute_Jv_kernel,
                dim=group.count,
                inputs=[
                    group.active_count,
                    group.body_ids_a,
                    group.point_a,
                    group.offset_a,
                    group.body_ids_b,
                    group.point_b,
                    group.offset_b,
                    entry_a.state_0.body_q,
                    entry_a.view.body_com,
                    entry_a.state_1.body_qd,
                    entry_b.state_0.body_q,
                    entry_b.view.body_com,
                    entry_b.state_1.body_qd,
                ],
                outputs=[group.Jv],
                device=self.model.device,
            )
            self._update_admm_contact_dual(group)
        for group in self._admm_dynamic_rp_contact_groups:
            if group.count == 0:
                continue
            body_entry = self._entries[group.body_entry_name]
            particle_entry = self._entries[group.particle_entry_name]
            wp.launch(
                contact_rp_compute_Jv_kernel,
                dim=group.count,
                inputs=[
                    group.active_count,
                    group.body_ids,
                    group.point_body,
                    group.particle_ids,
                    group.body_sign,
                    body_entry.state_0.body_q,
                    body_entry.view.body_com,
                    body_entry.state_1.body_qd,
                    particle_entry.state_1.particle_qd,
                ],
                outputs=[group.Jv],
                device=self.model.device,
            )
            self._update_admm_contact_dual(group)
        for group in self._admm_dynamic_pp_contact_groups:
            if group.count == 0:
                continue
            entry_a = self._entries[group.particle_entry_name_a]
            entry_b = self._entries[group.particle_entry_name_b]
            wp.launch(
                contact_pp_compute_Jv_kernel,
                dim=group.count,
                inputs=[
                    group.active_count,
                    group.particle_ids_a,
                    group.particle_ids_b,
                    entry_a.state_1.particle_qd,
                    entry_b.state_1.particle_qd,
                ],
                outputs=[group.Jv],
                device=self.model.device,
            )
            self._update_admm_contact_dual(group)
