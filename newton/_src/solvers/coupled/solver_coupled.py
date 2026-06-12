# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Unified coupled multi-solver API."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np
import warp as wp

from ...geometry import ParticleFlags, ShapeFlags
from ...sim import ModelFlags, StateFlags
from ..solver import SolverBase
from .interface import (
    CouplingEndpointKind,
    CouplingInterface,
)
from .model_view import ModelView

if TYPE_CHECKING:
    from ...sim import Contacts, Control, Model, State


def _identity_index_map(count: int, device) -> wp.array:
    """Return a dense local-to-global identity map."""
    return wp.array(list(range(count)), dtype=int, device=device)


def _inverse_index_map(local_to_global: wp.array, global_count: int, device) -> wp.array:
    """Return a dense global-to-local map with -1 for hidden ids."""
    mapping = [-1] * int(global_count)
    for local_id, raw_global_id in enumerate(local_to_global.numpy()):
        global_id = int(raw_global_id)
        if 0 <= global_id < global_count:
            mapping[global_id] = local_id
    return wp.array(mapping, dtype=int, device=device)


@dataclass(frozen=True)
class _EntryIndexMaps:
    body_local_to_global: wp.array
    body_global_to_local: wp.array
    particle_local_to_global: wp.array
    particle_global_to_local: wp.array
    joint_coord_local_to_global: wp.array
    joint_coord_global_to_local: wp.array
    joint_dof_local_to_global: wp.array
    joint_dof_global_to_local: wp.array


@dataclass(frozen=True)
class _CompactIndexLists:
    body_local_to_global: list[int]
    joint_local_to_global: list[int]
    joint_coord_local_to_global: list[int]
    joint_dof_local_to_global: list[int]
    shape_local_to_global: list[int]
    articulation_local_to_global: list[int]
    equality_constraint_local_to_global: list[int]
    constraint_mimic_local_to_global: list[int]


def _coupling_endpoint_arrays(
    endpoint_kind: CouplingEndpointKind,
    endpoint_indices: wp.array,
    device,
) -> tuple[wp.array, wp.array, wp.array]:
    """Return SoA endpoint arrays for a batch of same-kind endpoints."""
    count = endpoint_indices.shape[0]
    return (
        wp.array([int(endpoint_kind)] * count, dtype=int, device=device),
        endpoint_indices,
        wp.zeros(count, dtype=wp.vec3, device=device),
    )


def _require_supports_coupling(solver: SolverBase) -> None:
    if not isinstance(solver, CouplingInterface):
        raise TypeError(
            f"{type(solver).__name__} cannot participate in a coupled simulation; "
            "inherit CouplingInterface and override the hook methods it needs."
        )


@dataclass
class SolverEntry:
    """Runtime state for one sub-solver entry inside ``SolverCoupled``."""

    name: str
    solver: SolverBase
    substeps: int
    view: ModelView
    body_indices: wp.array
    particle_indices: wp.array
    joint_indices: wp.array
    body_dynamics_disabled_indices: wp.array
    joint_q_indices: wp.array
    joint_qd_indices: wp.array
    shape_indices: wp.array
    body_local_to_global: wp.array
    body_global_to_local: wp.array
    particle_local_to_global: wp.array
    particle_global_to_local: wp.array
    joint_coord_local_to_global: wp.array
    joint_coord_global_to_local: wp.array
    joint_dof_local_to_global: wp.array
    joint_dof_global_to_local: wp.array
    preserve_shape_ids: bool
    in_place: bool
    state_0: State | None = None
    state_1: State | None = None
    state_tmp: State | None = None
    control: Control | None = None
    has_body_force_input: bool = False
    has_particle_force_input: bool = False
    body_gravity_acceleration: wp.array[wp.vec3] | None = None
    particle_gravity_acceleration: wp.array[wp.vec3] | None = None


class SolverCoupled(SolverBase, CouplingInterface):
    """Couple multiple solvers through explicit ownership and coupling config.

    ``SolverCoupled`` owns generic mechanics that can be derived from
    ``Model``, ``ModelView`` and ``State``: per-solver views, ownership masks,
    state distribution/reconciliation, per-entry substeps, and shared coupling
    hook dispatch helpers. Algorithm-specific couplers such as
    :class:`~newton.solvers.experimental.coupled.SolverCoupledADMM` and
    :class:`~newton.solvers.experimental.coupled.SolverCoupledProxy` derive from this base class.

    Args:
        model: Shared model.
        entries: Sub-solver entries with explicit ownership.
        coupling: Optional algorithm configuration reserved for derived
            couplers. The base class steps entries independently and
            reconciles owned state.
    """

    @dataclass(frozen=True)
    class Entry:
        """Public configuration for one sub-solver.

        Each entry names a solver factory, the global model ids owned by that
        solver, an optional :class:`ModelView` customization callback, and
        stepping policy. The factory is called as ``solver(view)`` with the
        per-entry :class:`ModelView` and must return a configured
        :class:`SolverBase`. Bind any extra constructor arguments in the
        factory itself (e.g. ``lambda v: SolverVBD(model=v, iterations=10)``).
        Entry names must be unique. In-place stepping is only valid for solvers
        that explicitly support it. Shape ids remain in the parent model
        namespace by default; set
        ``preserve_shape_ids=False`` to expose a compact entry-local shape
        namespace instead.

        Args:
            name: Unique entry name used by coupling configuration.
            solver: Factory called with this entry's :class:`ModelView`.
            bodies: Global body ids owned by this entry.
            particles: Global particle ids owned by this entry.
            joints: Global joint ids owned by this entry.
            shapes: Global shape ids owned by this entry.
            configure_view: Optional callback for entry-local view overrides.
            substeps: Number of substeps to run per coupled step.
            in_place: Whether the sub-solver may step in-place.
            preserve_shape_ids: Whether shape ids remain in the parent model
                namespace instead of being compacted.
        """

        name: str
        solver: Callable[[ModelView], SolverBase]
        bodies: Sequence[int] = ()
        particles: Sequence[int] = ()
        joints: Sequence[int] = ()
        shapes: Sequence[int] = ()
        configure_view: Callable[[ModelView], None] | None = None
        substeps: int = 1
        in_place: bool = False
        preserve_shape_ids: bool = True

    def __init__(
        self,
        model: Model,
        entries: Sequence[SolverCoupled.Entry],
        coupling: object | None = None,
    ) -> None:
        super().__init__(model)

        self._entry_configs = list(entries)
        self._coupling = coupling
        self._entries: dict[str, SolverEntry] = {}
        self._solver_order: list[str] = []
        self._entry_contact_buffers: dict[str, Contacts] = {}
        self._entry_contact_sources: dict[str, Contacts] = {}
        self._entry_rigid_contact_generation: dict[str, wp.array] = {}
        self._entry_soft_contact_generation: dict[str, wp.array] = {}
        self._entry_rigid_contact_update: dict[str, wp.array] = {}
        self._entry_soft_contact_update: dict[str, wp.array] = {}
        self._entry_rigid_contact_src_to_dst: dict[str, wp.array] = {}
        self._entry_soft_contact_src_to_dst: dict[str, wp.array] = {}
        self._entry_output_state_valid = False

        self._validate_entry_names()
        self._body_owner = self._build_owner_map(model.body_count, [e.bodies for e in self._entry_configs])
        self._particle_owner = self._build_owner_map(model.particle_count, [e.particles for e in self._entry_configs])
        self._joint_owner = self._build_owner_map(model.joint_count, [e.joints for e in self._entry_configs])
        self._shape_owner = self._build_owner_map(model.shape_count, [e.shapes for e in self._entry_configs])

        self._build_entries()

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    def _validate_entry_names(self) -> None:
        names: set[str] = set()
        for cfg in self._entry_configs:
            if cfg.name in names:
                raise ValueError(f"Duplicate coupled solver entry name {cfg.name!r}")
            names.add(cfg.name)

    def _build_owner_map(self, count: int, owned_by_entry: Sequence[Sequence[int]]) -> list[int]:
        owner = [-1] * count
        for entry_idx, indices in enumerate(owned_by_entry):
            for raw_index in indices:
                index = int(raw_index)
                if index < 0 or index >= count:
                    raise IndexError(f"Ownership index {index} out of range for count {count}")
                if owner[index] != -1:
                    raise ValueError(f"Index {index} is owned by more than one coupled solver entry")
                owner[index] = entry_idx
        return owner

    def _build_entries(self) -> None:
        model = self.model
        device = model.device
        any_body_owner = any(owner >= 0 for owner in self._body_owner)
        any_particle_owner = any(owner >= 0 for owner in self._particle_owner)
        any_joint_owner = any(owner >= 0 for owner in self._joint_owner)

        for idx, cfg in enumerate(self._entry_configs):
            self._solver_order.append(cfg.name)
            substeps = int(cfg.substeps)
            if substeps < 1:
                raise ValueError(f"SolverCoupled.Entry {cfg.name!r} substeps must be >= 1")

            body_indices = wp.array([int(i) for i in cfg.bodies], dtype=int, device=device)
            particle_indices = wp.array([int(i) for i in cfg.particles], dtype=int, device=device)
            joint_indices = wp.array([int(i) for i in cfg.joints], dtype=int, device=device)
            shape_indices = wp.array([int(i) for i in cfg.shapes], dtype=int, device=device)
            joint_q_indices, joint_qd_indices = self._joint_state_indices(cfg.joints)
            view = ModelView(model, cfg.name)
            proxy_body_keep: set[int] = set()
            proxy_particle_keep: set[int] = set()
            proxy_joint_keep: set[int] = set()
            body_dynamics_disabled_indices = wp.zeros(0, dtype=int, device=device)

            if any_body_owner:
                proxy_body_keep = self._entry_proxy_body_keep_indices(cfg.name)
                to_zero = [i for i, owner in enumerate(self._body_owner) if owner != idx and i not in proxy_body_keep]
                if to_zero:
                    body_dynamics_disabled_indices = wp.array(to_zero, dtype=int, device=device)
                    view.disable_body_dynamics(body_dynamics_disabled_indices)
                if proxy_body_keep:
                    view.mark_proxy_bodies(wp.array(sorted(proxy_body_keep), dtype=int, device=device))

            if any_particle_owner:
                proxy_particle_keep = self._entry_proxy_particle_keep_indices(cfg.name)
                to_zero = [
                    i for i, owner in enumerate(self._particle_owner) if owner != idx and i not in proxy_particle_keep
                ]
                if to_zero:
                    to_zero_array = wp.array(to_zero, dtype=int, device=device)
                    view.zero_particle_mass(to_zero_array)
                    view.disable_particles(to_zero_array)
                if proxy_particle_keep:
                    view.mark_proxy_particles(wp.array(sorted(proxy_particle_keep), dtype=int, device=device))

            if any_joint_owner:
                proxy_joint_keep = self._entry_proxy_joint_keep_indices(cfg.name)
                to_disable = [
                    i for i, owner in enumerate(self._joint_owner) if owner != idx and i not in proxy_joint_keep
                ]
                if to_disable:
                    view.disable_joints(wp.array(to_disable, dtype=int, device=device))

            self._apply_entry_shape_visibility(view, cfg, proxy_body_keep)
            self._customize_view(cfg.name, view, body_indices)
            if cfg.configure_view is not None:
                cfg.configure_view(view)
            index_lists = self._compact_entry_view_if_needed(
                view, cfg, proxy_body_keep, proxy_particle_keep, proxy_joint_keep
            )
            if index_lists is None:
                self._apply_entry_prefix_limits(
                    view,
                    cfg,
                    proxy_body_keep,
                    proxy_particle_keep,
                    proxy_joint_keep,
                    preserve_shape_ids=bool(cfg.preserve_shape_ids),
                )
            self._filter_shape_contact_pairs(view)

            index_maps = self._build_entry_index_maps(view, index_lists)

            solver = cfg.solver(view)
            self._entries[cfg.name] = SolverEntry(
                name=cfg.name,
                solver=solver,
                substeps=substeps,
                view=view,
                body_indices=body_indices,
                particle_indices=particle_indices,
                joint_indices=joint_indices,
                body_dynamics_disabled_indices=body_dynamics_disabled_indices,
                joint_q_indices=joint_q_indices,
                joint_qd_indices=joint_qd_indices,
                shape_indices=shape_indices,
                body_local_to_global=index_maps.body_local_to_global,
                body_global_to_local=index_maps.body_global_to_local,
                particle_local_to_global=index_maps.particle_local_to_global,
                particle_global_to_local=index_maps.particle_global_to_local,
                joint_coord_local_to_global=index_maps.joint_coord_local_to_global,
                joint_coord_global_to_local=index_maps.joint_coord_global_to_local,
                joint_dof_local_to_global=index_maps.joint_dof_local_to_global,
                joint_dof_global_to_local=index_maps.joint_dof_global_to_local,
                preserve_shape_ids=bool(cfg.preserve_shape_ids),
                in_place=bool(cfg.in_place),
            )

        self._after_entries_constructed()

        for entry in self._entries.values():
            entry.state_0 = entry.view.state()
            entry.state_1 = entry.state_0 if entry.in_place else entry.view.state()
            entry.control = _entry_control(entry.view)
            entry.has_body_force_input = entry.state_0.body_f is not None
            entry.has_particle_force_input = entry.state_0.particle_f is not None
            if entry.substeps > 1 and not entry.in_place:
                entry.state_tmp = entry.view.state()

        self._after_entry_states_created()

    def _joint_state_indices(self, joints: Sequence[int]) -> tuple[wp.array, wp.array]:
        model = self.model
        device = model.device
        if not joints or model.joint_count == 0:
            return wp.zeros(0, dtype=int, device=device), wp.zeros(0, dtype=int, device=device)

        q_start = model.joint_q_start.numpy()
        qd_start = model.joint_qd_start.numpy()
        q_indices: list[int] = []
        qd_indices: list[int] = []
        for raw_joint in joints:
            joint = int(raw_joint)
            q_indices.extend(range(int(q_start[joint]), int(q_start[joint + 1])))
            qd_indices.extend(range(int(qd_start[joint]), int(qd_start[joint + 1])))
        return wp.array(q_indices, dtype=int, device=device), wp.array(qd_indices, dtype=int, device=device)

    def _entry_proxy_body_keep_indices(self, name: str) -> set[int]:
        """Return body indices that should remain dynamic as proxies in one view."""
        del name
        return set()

    def _entry_proxy_particle_keep_indices(self, name: str) -> set[int]:
        """Return particle indices that should remain dynamic as proxies in one view."""
        del name
        return set()

    def _entry_proxy_joint_keep_indices(self, name: str) -> set[int]:
        """Return joint indices that should remain enabled as proxies in one view."""
        del name
        return set()

    def _customize_view(self, name: str, view: ModelView, body_indices: wp.array) -> None:
        """Hook for subclasses that still need construction-time view edits."""
        del name, view, body_indices

    def _apply_entry_shape_visibility(
        self,
        view: ModelView,
        cfg: SolverCoupled.Entry,
        proxy_body_keep: set[int],
    ) -> None:
        """Restrict shape collisions to entry-owned and proxy-visible shapes."""
        model = self.model
        if model.shape_count == 0 or model.shape_flags is None:
            return

        visible = {int(i) for i in cfg.shapes}
        for shape_id in visible:
            if shape_id < 0 or shape_id >= model.shape_count:
                raise IndexError(f"Shape ownership index {shape_id} out of range for count {model.shape_count}")

        visible_bodies = set(proxy_body_keep)
        if not cfg.shapes:
            visible_bodies.update(int(i) for i in cfg.bodies)
        visible.update(self._entry_visible_shapes(cfg, visible_bodies))
        if not visible:
            return

        collision_mask = int(ShapeFlags.COLLIDE_SHAPES | ShapeFlags.COLLIDE_PARTICLES | ShapeFlags.HYDROELASTIC)
        shape_flags = view.shape_flags.numpy().copy()
        hidden = np.ones(model.shape_count, dtype=bool)
        if visible:
            hidden[np.fromiter(visible, dtype=np.int32)] = False
        shape_flags[hidden] &= ~collision_mask
        view.shape_flags = wp.array(shape_flags, dtype=wp.int32, device=model.device)

    def _filter_shape_contact_pairs(self, view: ModelView) -> None:
        """Filter explicit contact pairs against a solver view's shape mask."""
        pairs = getattr(view, "shape_contact_pairs", None)
        if pairs is None:
            return

        shape_count = int(getattr(view, "shape_count", self.model.shape_count))
        flags = getattr(view, "shape_flags", None)
        flags_np = flags.numpy() if flags is not None else None
        collide_mask = int(ShapeFlags.COLLIDE_SHAPES)

        keep: list[tuple[int, int]] = []
        for pair in pairs.numpy():
            shape_a = int(pair[0])
            shape_b = int(pair[1])
            if shape_a < 0 or shape_b < 0 or shape_a >= shape_count or shape_b >= shape_count:
                continue
            if flags_np is not None and ((int(flags_np[shape_a]) & collide_mask) == 0):
                continue
            if flags_np is not None and ((int(flags_np[shape_b]) & collide_mask) == 0):
                continue
            keep.append((shape_a, shape_b))

        filtered = np.asarray(keep, dtype=np.int32).reshape((-1, 2))
        view.shape_contact_pairs = wp.array(filtered, dtype=wp.vec2i, device=self.model.device)
        view.shape_contact_pair_count = len(keep)

    @staticmethod
    def _prefix_length(indices: set[int]) -> int | None:
        """Return the dense prefix length represented by *indices*, or None."""
        if not indices:
            return 0
        prefix_len = max(indices) + 1
        return prefix_len if indices == set(range(prefix_len)) else None

    def _apply_entry_prefix_limits(
        self,
        view: ModelView,
        cfg: SolverCoupled.Entry,
        proxy_body_keep: set[int],
        proxy_particle_keep: set[int],
        proxy_joint_keep: set[int],
        *,
        preserve_shape_ids: bool = False,
    ) -> None:
        """Expose compact body/joint view counts when visible entities form a prefix."""
        visible_bodies = {int(i) for i in cfg.bodies} | {int(i) for i in proxy_body_keep}
        visible_particles = {int(i) for i in cfg.particles} | {int(i) for i in proxy_particle_keep}
        visible_joints = {int(i) for i in cfg.joints} | {int(i) for i in proxy_joint_keep}

        self._apply_body_prefix_limit(view, visible_bodies)
        self._apply_particle_prefix_limit(view, visible_particles)
        self._apply_joint_prefix_limit(view, visible_joints)
        if preserve_shape_ids:
            view.shape_count = self.model.shape_count
            self._apply_preserved_shape_metadata_without_compaction(view, cfg, visible_bodies)
        else:
            self._apply_shape_prefix_limit(view, cfg, visible_bodies)

    def _apply_body_prefix_limit(self, view: ModelView, visible_bodies: set[int]) -> None:
        body_prefix = self._prefix_length(visible_bodies)
        if body_prefix is not None and body_prefix < self.model.body_count:
            view.body_count = body_prefix

    def _apply_particle_prefix_limit(self, view: ModelView, visible_particles: set[int]) -> None:
        # Keep particle arrays globally indexed for now. Particle contacts and
        # deformable element connectivity still commonly carry global particle
        # ids. The pure-rigid case is the exception: expose zero particles and
        # zero deformable element counts so rigid-only solvers never see stale
        # particle metadata.
        if visible_particles or not self.model.particle_count:
            return
        view.particle_count = 0
        view.spring_count = 0
        view.tri_count = 0
        view.edge_count = 0
        view.tet_count = 0
        view.muscle_count = 0

    def _apply_joint_prefix_limit(self, view: ModelView, visible_joints: set[int]) -> None:
        joint_prefix = self._prefix_length(visible_joints)
        if joint_prefix is None or joint_prefix >= self.model.joint_count:
            return
        articulation_prefix = self._articulation_prefix_count(joint_prefix)
        if articulation_prefix is None:
            return
        view.joint_count = joint_prefix
        view.joint_coord_count = int(self.model.joint_q_start.numpy()[joint_prefix])
        view.joint_dof_count = int(self.model.joint_qd_start.numpy()[joint_prefix])
        view.articulation_count = articulation_prefix

    def _apply_shape_prefix_limit(self, view: ModelView, cfg: SolverCoupled.Entry, visible_bodies: set[int]) -> None:
        visible_shapes = self._entry_visible_shapes(cfg, visible_bodies)
        if not visible_shapes:
            return
        shape_prefix = self._prefix_length(visible_shapes)
        if shape_prefix is not None and shape_prefix < self.model.shape_count:
            view.shape_count = shape_prefix

    def _apply_preserved_shape_metadata_without_compaction(
        self,
        view: ModelView,
        cfg: SolverCoupled.Entry,
        visible_bodies: set[int],
    ) -> None:
        model = self.model
        visible_shapes = self._entry_visible_shapes(cfg, visible_bodies)
        body_count = int(getattr(view, "body_count", model.body_count))

        shape_body = getattr(view, "shape_body", None)
        if shape_body is not None:
            shape_body_np = shape_body.numpy().copy()
            for shape_id, body_id in enumerate(shape_body_np):
                if shape_id not in visible_shapes or int(body_id) >= body_count:
                    shape_body_np[shape_id] = -1
            view.shape_body = wp.array(shape_body_np, dtype=wp.int32, device=model.device)

        body_global_to_local = {body_id: body_id for body_id in range(body_count)}
        view.body_shapes = self._global_shape_body_shapes(model.body_shapes, body_global_to_local, visible_shapes)
        view.shape_collision_filter_pairs = set(model.shape_collision_filter_pairs)

    def _build_entry_index_maps(self, view: ModelView, index_lists: _CompactIndexLists | None) -> _EntryIndexMaps:
        """Build local/global id maps for a completed entry view."""
        model = self.model
        device = model.device
        if index_lists is None:
            body_local_to_global = _identity_index_map(int(view.body_count), device)
            particle_local_to_global = _identity_index_map(int(view.particle_count), device)
            joint_coord_local_to_global = _identity_index_map(int(view.joint_coord_count), device)
            joint_dof_local_to_global = _identity_index_map(int(view.joint_dof_count), device)
        else:
            body_local_to_global = wp.array(index_lists.body_local_to_global, dtype=int, device=device)
            particle_local_to_global = _identity_index_map(int(view.particle_count), device)
            joint_coord_local_to_global = wp.array(index_lists.joint_coord_local_to_global, dtype=int, device=device)
            joint_dof_local_to_global = wp.array(index_lists.joint_dof_local_to_global, dtype=int, device=device)

        return _EntryIndexMaps(
            body_local_to_global=body_local_to_global,
            body_global_to_local=_inverse_index_map(body_local_to_global, model.body_count, device),
            particle_local_to_global=particle_local_to_global,
            particle_global_to_local=_inverse_index_map(particle_local_to_global, model.particle_count, device),
            joint_coord_local_to_global=joint_coord_local_to_global,
            joint_coord_global_to_local=_inverse_index_map(
                joint_coord_local_to_global, model.joint_coord_count, device
            ),
            joint_dof_local_to_global=joint_dof_local_to_global,
            joint_dof_global_to_local=_inverse_index_map(joint_dof_local_to_global, model.joint_dof_count, device),
        )

    def _compact_entry_view_if_needed(
        self,
        view: ModelView,
        cfg: SolverCoupled.Entry,
        proxy_body_keep: set[int],
        proxy_particle_keep: set[int],
        proxy_joint_keep: set[int],
    ) -> _CompactIndexLists | None:
        """Compact rigid entry views while preserving global mappings."""
        model = self.model
        visible_bodies = {int(i) for i in cfg.bodies} | {int(i) for i in proxy_body_keep}
        visible_particles = {int(i) for i in cfg.particles} | {int(i) for i in proxy_particle_keep}
        visible_joints = {int(i) for i in cfg.joints} | {int(i) for i in proxy_joint_keep}
        visible_shapes = self._entry_visible_shapes(cfg, visible_bodies)

        if visible_particles:
            return None
        if not visible_bodies and not visible_joints and not visible_shapes:
            return None

        body_order = self._ordered_world_subset(
            visible_bodies,
            model.body_world,
            getattr(model, "body_world_start", None),
            model.body_count,
            "bodies",
        )
        joint_order = self._ordered_world_subset(
            visible_joints,
            model.joint_world,
            getattr(model, "joint_world_start", None),
            model.joint_count,
            "joints",
        )
        shape_order = self._ordered_world_subset(
            visible_shapes,
            model.shape_world,
            getattr(model, "shape_world_start", None),
            model.shape_count,
            "shapes",
            allow_global=True,
        )
        if body_order is None or joint_order is None or shape_order is None:
            return None

        compact = self._compact_index_lists(view, body_order, joint_order, shape_order)
        if compact is None:
            return None

        self._apply_compact_entry_view(view, compact, preserve_shape_ids=bool(cfg.preserve_shape_ids))
        return compact

    def _entry_visible_shapes(self, cfg: SolverCoupled.Entry, visible_bodies: set[int]) -> set[int]:
        model = self.model
        visible_shapes = {int(i) for i in cfg.shapes}
        include_default_static_shapes = not cfg.shapes
        if model.shape_count and model.shape_body is not None and (visible_bodies or include_default_static_shapes):
            shape_body = model.shape_body.numpy()
            visible_shapes.update(
                int(shape_id)
                for shape_id, body_id in enumerate(shape_body)
                if (include_default_static_shapes and int(body_id) < 0) or int(body_id) in visible_bodies
            )
        return visible_shapes

    def _ordered_world_subset(
        self,
        indices: set[int],
        world_array: wp.array | None,
        world_start_array: wp.array | None,
        total_count: int,
        entity_name: str,
        *,
        allow_global: bool = False,
    ) -> list[int] | None:
        """Return indices ordered by compact world layout, or None if not homogeneous."""
        if not indices:
            return []

        world_count = int(self.model.world_count)
        if world_count <= 1:
            return sorted(indices)
        if total_count % world_count != 0 and world_start_array is None:
            return None

        worlds = world_array.numpy() if world_array is not None else np.zeros(total_count, dtype=np.int32)
        starts = world_start_array.numpy() if world_start_array is not None else None
        buckets: list[list[int]] = [[] for _ in range(world_count)]
        global_front: list[int] = []

        for index in sorted(indices):
            if index < 0 or index >= total_count:
                raise IndexError(f"{entity_name} index {index} out of range for count {total_count}")
            world = int(worlds[index])
            if world < 0:
                if not allow_global:
                    return None
                global_front.append(index)
                continue
            if world >= world_count:
                return None
            start = self._world_start_for_index(starts, total_count, world_count, world)
            end = self._world_start_for_index(starts, total_count, world_count, world + 1)
            if index < start or index >= end:
                return None
            buckets[world].append(index - start)

        template = sorted(buckets[0])
        for bucket in buckets[1:]:
            if sorted(bucket) != template:
                return None

        ordered = sorted(global_front)
        for world in range(world_count):
            start = self._world_start_for_index(starts, total_count, world_count, world)
            ordered.extend(start + offset for offset in template)
        return ordered

    @staticmethod
    def _world_start_for_index(
        starts: np.ndarray | None,
        total_count: int,
        world_count: int,
        world_or_sentinel: int,
    ) -> int:
        if starts is not None:
            return int(starts[world_or_sentinel])
        per_world = total_count // world_count
        return int(world_or_sentinel) * per_world

    def _compact_index_lists(
        self,
        view: ModelView,
        body_order: list[int],
        joint_order: list[int],
        shape_order: list[int],
    ) -> _CompactIndexLists | None:
        model = self.model
        body_set = set(body_order)
        joint_set = set(joint_order)
        body_global_to_local = {global_id: local_id for local_id, global_id in enumerate(body_order)}
        joint_global_to_local = {global_id: local_id for local_id, global_id in enumerate(joint_order)}

        joint_parent = model.joint_parent.numpy() if model.joint_count else np.empty(0, dtype=np.int32)
        joint_child = model.joint_child.numpy() if model.joint_count else np.empty(0, dtype=np.int32)
        for joint in joint_order:
            parent = int(joint_parent[joint])
            child = int(joint_child[joint])
            if child not in body_set or (parent >= 0 and parent not in body_set):
                return None

        shape_body = model.shape_body.numpy() if model.shape_count and model.shape_body is not None else []
        for shape in shape_order:
            body = int(shape_body[shape]) if len(shape_body) else -1
            if body >= 0 and body not in body_set:
                return None

        articulation_order = self._compact_articulation_order(view, joint_order, joint_set)
        if articulation_order is None:
            return None

        equality_order = self._compact_equality_constraint_order(
            body_global_to_local,
            joint_global_to_local,
        )
        if equality_order is None:
            return None
        mimic_order = self._compact_mimic_constraint_order(joint_global_to_local)
        if mimic_order is None:
            return None

        joint_q_start = model.joint_q_start.numpy()
        joint_qd_start = model.joint_qd_start.numpy()
        joint_coord_order: list[int] = []
        joint_dof_order: list[int] = []
        for joint in joint_order:
            joint_coord_order.extend(range(int(joint_q_start[joint]), int(joint_q_start[joint + 1])))
            joint_dof_order.extend(range(int(joint_qd_start[joint]), int(joint_qd_start[joint + 1])))

        return _CompactIndexLists(
            body_local_to_global=body_order,
            joint_local_to_global=joint_order,
            joint_coord_local_to_global=joint_coord_order,
            joint_dof_local_to_global=joint_dof_order,
            shape_local_to_global=shape_order,
            articulation_local_to_global=articulation_order,
            equality_constraint_local_to_global=equality_order,
            constraint_mimic_local_to_global=mimic_order,
        )

    def _compact_articulation_order(
        self,
        view: ModelView,
        joint_order: list[int],
        joint_set: set[int],
    ) -> list[int] | None:
        model = self.model
        if model.articulation_count == 0:
            return []
        joint_articulation = model.joint_articulation.numpy()
        articulation_start = model.articulation_start.numpy()
        joint_enabled = view.joint_enabled.numpy() if view.joint_enabled is not None else model.joint_enabled.numpy()
        selected = {int(joint_articulation[joint]) for joint in joint_order if int(joint_articulation[joint]) >= 0}
        for articulation in selected:
            start = int(articulation_start[articulation])
            end = int(articulation_start[articulation + 1])
            articulation_joints = set(range(start, end))
            omitted_enabled = [
                joint
                for joint in articulation_joints.difference(joint_set)
                if joint < len(joint_enabled) and bool(joint_enabled[joint])
            ]
            if omitted_enabled:
                return None
        return self._ordered_world_subset(
            selected,
            model.articulation_world,
            getattr(model, "articulation_world_start", None),
            model.articulation_count,
            "articulations",
        )

    def _compact_equality_constraint_order(
        self,
        body_global_to_local: dict[int, int],
        joint_global_to_local: dict[int, int],
    ) -> list[int] | None:
        count = 0
        body1 = body2 = joint1 = joint2 = None
        world = world_start = None

        for full_name, frequency in self.model.attribute_frequency.items():
            if ":" not in full_name or not isinstance(frequency, str):
                continue
            if frequency.rsplit(":", 1)[-1] != "equality_constraint":
                continue

            count = int(self.model.custom_frequency_counts[frequency])
            namespace_name, attr_name = full_name.split(":", 1)
            namespace = getattr(self.model, namespace_name, None)
            if namespace is None:
                continue
            try:
                value = object.__getattribute__(namespace, attr_name)
            except AttributeError:
                continue
            if not isinstance(value, wp.array):
                continue

            if attr_name == "equality_constraint_body1":
                body1 = value.numpy()
            elif attr_name == "equality_constraint_body2":
                body2 = value.numpy()
            elif attr_name == "equality_constraint_joint1":
                joint1 = value.numpy()
            elif attr_name == "equality_constraint_joint2":
                joint2 = value.numpy()
            elif attr_name == "equality_constraint_world":
                world = value
                world_start = self._world_start_array(value, count)

        if count == 0:
            return []
        if body1 is None or body2 is None or joint1 is None or joint2 is None:
            return []

        selected: set[int] = set()
        for constraint in range(count):
            if not self._constraint_ref_visible(int(body1[constraint]), body_global_to_local):
                continue
            if not self._constraint_ref_visible(int(body2[constraint]), body_global_to_local):
                continue
            if not self._constraint_ref_visible(int(joint1[constraint]), joint_global_to_local):
                continue
            if not self._constraint_ref_visible(int(joint2[constraint]), joint_global_to_local):
                continue
            selected.add(constraint)
        return self._ordered_world_subset(
            selected,
            world,
            world_start,
            count,
            "equality constraints",
        )

    def _compact_mimic_constraint_order(self, joint_global_to_local: dict[int, int]) -> list[int] | None:
        model = self.model
        if model.constraint_mimic_count == 0:
            return []
        joint0 = model.constraint_mimic_joint0.numpy()
        joint1 = model.constraint_mimic_joint1.numpy()
        selected = {
            constraint
            for constraint in range(model.constraint_mimic_count)
            if int(joint0[constraint]) in joint_global_to_local and int(joint1[constraint]) in joint_global_to_local
        }
        return self._ordered_world_subset(
            selected,
            model.constraint_mimic_world,
            None,
            model.constraint_mimic_count,
            "mimic constraints",
        )

    @staticmethod
    def _constraint_ref_visible(index: int, global_to_local: dict[int, int]) -> bool:
        return index < 0 or index in global_to_local

    def _apply_compact_entry_view(
        self,
        view: ModelView,
        compact: _CompactIndexLists,
        *,
        preserve_shape_ids: bool = False,
    ) -> None:
        """Install compact topology arrays on an entry view."""
        model = self.model
        device = model.device
        body_order = compact.body_local_to_global
        joint_order = compact.joint_local_to_global
        coord_order = compact.joint_coord_local_to_global
        dof_order = compact.joint_dof_local_to_global
        visible_shape_order = compact.shape_local_to_global
        shape_order = list(range(model.shape_count)) if preserve_shape_ids else visible_shape_order
        articulation_order = compact.articulation_local_to_global
        equality_order = compact.equality_constraint_local_to_global
        mimic_order = compact.constraint_mimic_local_to_global

        body_global_to_local = {global_id: local_id for local_id, global_id in enumerate(body_order)}
        joint_global_to_local = {global_id: local_id for local_id, global_id in enumerate(joint_order)}
        shape_global_to_local = {global_id: local_id for local_id, global_id in enumerate(shape_order)}
        articulation_global_to_local = {global_id: local_id for local_id, global_id in enumerate(articulation_order)}

        view.body_count = len(body_order)
        view.joint_count = len(joint_order)
        view.joint_coord_count = len(coord_order)
        view.joint_dof_count = len(dof_order)
        view.shape_count = model.shape_count if preserve_shape_ids else len(shape_order)
        view.articulation_count = len(articulation_order)
        view.constraint_mimic_count = len(mimic_order)
        view.particle_count = 0
        view.spring_count = 0
        view.tri_count = 0
        view.edge_count = 0
        view.tet_count = 0
        view.muscle_count = 0

        index_orders_by_frequency = self._compact_index_orders_by_frequency(
            body_order,
            joint_order,
            coord_order,
            dof_order,
            shape_order,
            articulation_order,
            equality_order,
            mimic_order,
        )
        self._set_compact_custom_frequency_counts(view, index_orders_by_frequency)
        remapped_or_derived_attrs = {
            "joint_parent",
            "joint_child",
            "joint_ancestor",
            "joint_articulation",
            "shape_body",
            "constraint_mimic_joint0",
            "constraint_mimic_joint1",
        }
        renamed_attrs = {"joint_target_pos", "joint_target_vel"}
        handled_attrs = self._select_compact_attributes_by_frequency(
            view,
            index_orders_by_frequency,
            exclude=remapped_or_derived_attrs | renamed_attrs,
        )
        self._select_compact_prefix_attributes(
            view,
            {
                "body_": (body_order, model.body_count),
                "joint_": (joint_order, model.joint_count),
                "shape_": (shape_order, model.shape_count),
                "_shape_": (shape_order, model.shape_count),
                "articulation_": (articulation_order, model.articulation_count),
                "constraint_mimic_": (mimic_order, model.constraint_mimic_count),
            },
            exclude=handled_attrs | remapped_or_derived_attrs | renamed_attrs,
        )

        joint_parent = self._select_numpy_array(view, "joint_parent", joint_order)
        joint_child = self._select_numpy_array(view, "joint_child", joint_order)
        if joint_parent is not None:
            view.joint_parent = wp.array(
                [self._remap_optional_index(int(parent), body_global_to_local) for parent in joint_parent],
                dtype=wp.int32,
                device=device,
            )
        if joint_child is not None:
            view.joint_child = wp.array(
                [body_global_to_local[int(child)] for child in joint_child],
                dtype=wp.int32,
                device=device,
            )

        joint_articulation = self._select_numpy_array(view, "joint_articulation", joint_order)
        if joint_articulation is not None:
            view.joint_articulation = wp.array(
                [
                    self._remap_optional_index(int(articulation), articulation_global_to_local)
                    for articulation in joint_articulation
                ],
                dtype=wp.int32,
                device=device,
            )
        joint_parent_local = view.joint_parent.numpy() if len(joint_order) else np.empty(0, dtype=np.int32)
        joint_child_local = view.joint_child.numpy() if len(joint_order) else np.empty(0, dtype=np.int32)
        child_to_joint = {int(child): joint for joint, child in enumerate(joint_child_local)}
        view.joint_ancestor = wp.array(
            [child_to_joint.get(int(parent), -1) for parent in joint_parent_local],
            dtype=wp.int32,
            device=device,
        )

        view.joint_q_start = wp.array(
            self._rebased_joint_starts(model.joint_q_start.numpy(), joint_order),
            dtype=wp.int32,
            device=device,
        )
        view.joint_qd_start = wp.array(
            self._rebased_joint_starts(model.joint_qd_start.numpy(), joint_order),
            dtype=wp.int32,
            device=device,
        )

        shape_body = self._select_numpy_array(view, "shape_body", shape_order)
        if shape_body is not None:
            visible_shapes = set(visible_shape_order)
            view.shape_body = wp.array(
                [
                    self._remap_shape_body(
                        global_shape=shape,
                        body=int(body),
                        visible_shapes=visible_shapes,
                        body_global_to_local=body_global_to_local,
                        preserve_shape_ids=preserve_shape_ids,
                    )
                    for shape, body in zip(shape_order, shape_body, strict=True)
                ],
                dtype=wp.int32,
                device=device,
            )
        if preserve_shape_ids:
            view.body_shapes = self._global_shape_body_shapes(
                model.body_shapes,
                body_global_to_local,
                set(visible_shape_order),
            )
            view.shape_collision_filter_pairs = set(model.shape_collision_filter_pairs)
        else:
            view.body_shapes = self._compact_body_shapes(model.body_shapes, body_global_to_local, shape_global_to_local)
            view.shape_collision_filter_pairs = self._compact_shape_pair_set(
                model.shape_collision_filter_pairs,
                shape_global_to_local,
            )
            self._compact_shape_contact_pairs(view, shape_global_to_local)

        articulation_starts = self._compact_articulation_starts(joint_order, articulation_order)
        view.articulation_start = wp.array(articulation_starts, dtype=wp.int32, device=device)
        view.articulation_end = wp.array(articulation_starts[1:], dtype=wp.int32, device=device)
        self._set_compact_articulation_extents(view, articulation_order)

        self._compact_mimic_constraints(view, mimic_order, joint_global_to_local)

        # For VBD solver we require color groups to be compacted too.
        self._compact_color_groups(view, body_global_to_local)

        self._set_world_start_arrays(view)
        self._compact_custom_attribute_namespaces(
            view,
            body_order,
            joint_order,
            coord_order,
            dof_order,
            shape_order,
            articulation_order,
            equality_order,
            mimic_order,
        )

    def _select_numpy_array(self, view: ModelView, name: str, indices: Sequence[int]) -> np.ndarray | None:
        value = self._raw_view_value(view, name)
        if value is None or not isinstance(value, wp.array):
            return None
        if not indices:
            return np.asarray([], dtype=value.numpy().dtype)
        return value.numpy()[np.asarray(indices, dtype=np.int64)]

    def _select_model_value(self, view: ModelView, name: str, indices: Sequence[int]) -> bool:
        value = self._raw_view_value(view, name)
        selected = self._select_attribute_value(value, indices)
        if selected is None:
            return False
        setattr(view, name, selected)
        return True

    def _compact_index_orders_by_frequency(
        self,
        body_order: Sequence[int],
        joint_order: Sequence[int],
        coord_order: Sequence[int],
        dof_order: Sequence[int],
        shape_order: Sequence[int],
        articulation_order: Sequence[int],
        equality_order: Sequence[int],
        mimic_order: Sequence[int],
    ) -> dict[Model.AttributeFrequency | str, tuple[Sequence[int], int]]:
        freq = self.model.AttributeFrequency
        model = self.model
        index_orders_by_frequency = {
            freq.BODY: (body_order, model.body_count),
            freq.JOINT: (joint_order, model.joint_count),
            freq.JOINT_COORD: (coord_order, model.joint_coord_count),
            freq.JOINT_DOF: (dof_order, model.joint_dof_count),
            freq.SHAPE: (shape_order, model.shape_count),
            freq.ARTICULATION: (articulation_order, model.articulation_count),
            freq.CONSTRAINT_MIMIC: (mimic_order, model.constraint_mimic_count),
        }
        index_orders_by_name = {
            "body": (body_order, model.body_count),
            "joint": (joint_order, model.joint_count),
            "joint_coord": (coord_order, model.joint_coord_count),
            "joint_dof": (dof_order, model.joint_dof_count),
            "shape": (shape_order, model.shape_count),
            "articulation": (articulation_order, model.articulation_count),
            "equality_constraint": (equality_order, 0),
            "constraint_mimic": (mimic_order, model.constraint_mimic_count),
        }
        for frequency in model.custom_frequency_counts:
            frequency_name = frequency.rsplit(":", 1)[-1]
            index_order = index_orders_by_name.get(frequency_name)
            if index_order is not None:
                indices, _source_count = index_order
                index_orders_by_frequency[frequency] = (indices, int(model.custom_frequency_counts[frequency]))
        return index_orders_by_frequency

    def _set_compact_custom_frequency_counts(
        self,
        view: ModelView,
        index_orders_by_frequency: dict[Model.AttributeFrequency | str, tuple[Sequence[int], int]],
    ) -> None:
        custom_frequency_counts = dict(self.model.custom_frequency_counts)
        for frequency, (indices, _source_count) in index_orders_by_frequency.items():
            if isinstance(frequency, str):
                custom_frequency_counts[frequency] = len(indices)
        if custom_frequency_counts != self.model.custom_frequency_counts:
            view.custom_frequency_counts = custom_frequency_counts

    def _select_compact_attributes_by_frequency(
        self,
        view: ModelView,
        indices_by_frequency: dict[Model.AttributeFrequency | str, tuple[Sequence[int], int]],
        *,
        exclude: set[str],
    ) -> set[str]:
        handled: set[str] = set()
        model_assignment = self.model.AttributeAssignment.MODEL
        for full_name, frequency in self.model.attribute_frequency.items():
            if ":" in full_name or full_name in exclude:
                continue
            if self.model.attribute_assignment.get(full_name, model_assignment) != model_assignment:
                continue
            indexed_projection = indices_by_frequency.get(frequency)
            if indexed_projection is None:
                continue
            indices, source_count = indexed_projection
            value = self._raw_view_value(view, full_name)
            if not self._is_indexed_compact_value(value, source_count):
                continue
            selected = self._select_attribute_value(value, indices)
            if selected is None:
                continue
            setattr(view, full_name, selected)
            handled.add(full_name)
        return handled

    def _select_compact_prefix_attributes(
        self,
        view: ModelView,
        prefix_indices: dict[str, tuple[Sequence[int], int]],
        *,
        exclude: set[str],
    ) -> set[str]:
        handled: set[str] = set()
        parent = object.__getattribute__(view, "_parent")
        overrides = object.__getattribute__(view, "_overrides")
        names = set(overrides)
        for name in dir(parent):
            if name.startswith("_") or hasattr(type(parent), name):
                continue
            names.add(name)
        names = sorted(names)
        for name in names:
            if name in exclude or self._is_compact_projection_private_name(name):
                continue
            for prefix, (indices, source_count) in prefix_indices.items():
                if not name.startswith(prefix):
                    continue
                value = self._raw_view_value(view, name)
                if not self._is_indexed_compact_value(value, source_count):
                    break
                if self._select_model_value(view, name, indices):
                    handled.add(name)
                break
        return handled

    @staticmethod
    def _is_compact_projection_private_name(name: str) -> bool:
        return (
            name.endswith("_start")
            or name.endswith("_count")
            or name.endswith("_color_groups")
            or name.endswith("_pairs")
        )

    @staticmethod
    def _is_indexed_compact_value(value, source_count: int) -> bool:
        if isinstance(value, wp.array):
            return value.shape[0] == source_count
        if isinstance(value, np.ndarray):
            return value.shape[0] == source_count
        if isinstance(value, list):
            return len(value) == source_count
        return False

    @staticmethod
    def _raw_view_value(view: ModelView, name: str):
        overrides = object.__getattribute__(view, "_overrides")
        if name in overrides:
            return overrides[name]
        parent = object.__getattribute__(view, "_parent")
        return getattr(parent, name, None)

    @staticmethod
    def _remap_optional_index(index: int, global_to_local: dict[int, int]) -> int:
        return -1 if index < 0 else global_to_local[index]

    @staticmethod
    def _remap_shape_body(
        *,
        global_shape: int,
        body: int,
        visible_shapes: set[int],
        body_global_to_local: dict[int, int],
        preserve_shape_ids: bool,
    ) -> int:
        if preserve_shape_ids and int(global_shape) not in visible_shapes:
            return -1
        return -1 if body < 0 else body_global_to_local[body]

    @staticmethod
    def _rebased_joint_starts(starts: np.ndarray, joint_order: Sequence[int]) -> list[int]:
        rebased: list[int] = []
        cursor = 0
        for joint in joint_order:
            rebased.append(cursor)
            cursor += int(starts[joint + 1]) - int(starts[joint])
        rebased.append(cursor)
        return rebased

    @staticmethod
    def _compact_body_shapes(
        body_shapes: dict[int, list[int]],
        body_global_to_local: dict[int, int],
        shape_global_to_local: dict[int, int],
    ) -> dict[int, list[int]]:
        compact: dict[int, list[int]] = {-1: []}
        for global_body, local_body in body_global_to_local.items():
            compact[local_body] = []
            for shape in body_shapes.get(global_body, []):
                local_shape = shape_global_to_local.get(int(shape))
                if local_shape is not None:
                    compact[local_body].append(local_shape)
        for shape in body_shapes.get(-1, []):
            local_shape = shape_global_to_local.get(int(shape))
            if local_shape is not None:
                compact[-1].append(local_shape)
        return compact

    @staticmethod
    def _global_shape_body_shapes(
        body_shapes: dict[int, list[int]],
        body_global_to_local: dict[int, int],
        visible_shapes: set[int],
    ) -> dict[int, list[int]]:
        compact: dict[int, list[int]] = {-1: []}
        for local_body in body_global_to_local.values():
            compact[local_body] = []
        for global_body, shapes in body_shapes.items():
            if global_body < 0:
                local_body = -1
            else:
                local_body = body_global_to_local.get(int(global_body))
                if local_body is None:
                    continue
            for shape in shapes:
                if int(shape) in visible_shapes:
                    compact.setdefault(local_body, []).append(int(shape))
        return compact

    @staticmethod
    def _compact_shape_pair_set(
        pairs: set[tuple[int, int]],
        shape_global_to_local: dict[int, int],
    ) -> set[tuple[int, int]]:
        compact: set[tuple[int, int]] = set()
        for shape_a, shape_b in pairs:
            local_a = shape_global_to_local.get(int(shape_a))
            local_b = shape_global_to_local.get(int(shape_b))
            if local_a is None or local_b is None:
                continue
            compact.add((min(local_a, local_b), max(local_a, local_b)))
        return compact

    def _compact_shape_contact_pairs(self, view: ModelView, shape_global_to_local: dict[int, int]) -> None:
        pairs = getattr(view, "shape_contact_pairs", None)
        if pairs is None:
            return
        compact: list[tuple[int, int]] = []
        for shape_a, shape_b in pairs.numpy():
            local_a = shape_global_to_local.get(int(shape_a))
            local_b = shape_global_to_local.get(int(shape_b))
            if local_a is None or local_b is None:
                continue
            compact.append((local_a, local_b))
        array = np.asarray(compact, dtype=np.int32).reshape((-1, 2))
        view.shape_contact_pairs = wp.array(array, dtype=wp.vec2i, device=self.model.device)
        view.shape_contact_pair_count = len(compact)

    def _compact_articulation_starts(
        self,
        joint_order: Sequence[int],
        articulation_order: Sequence[int],
    ) -> list[int]:
        joint_global_to_local = {global_id: local_id for local_id, global_id in enumerate(joint_order)}
        joint_articulation = self.model.joint_articulation.numpy() if self.model.joint_count else []
        starts: list[int] = []
        for articulation in articulation_order:
            local_joints = [
                local
                for global_joint, local in joint_global_to_local.items()
                if int(joint_articulation[global_joint]) == int(articulation)
            ]
            starts.append(min(local_joints) if local_joints else len(joint_order))
        starts.append(len(joint_order))
        return starts

    def _set_compact_articulation_extents(
        self,
        view: ModelView,
        articulation_order: Sequence[int],
    ) -> None:
        if not articulation_order:
            view.max_joints_per_articulation = 0
            view.max_dofs_per_articulation = 0
            return
        starts = view.articulation_start.numpy()
        qd_starts = view.joint_qd_start.numpy()
        max_joints = 0
        max_dofs = 0
        for art_id in range(len(articulation_order)):
            joint_start = int(starts[art_id])
            joint_end = int(starts[art_id + 1])
            max_joints = max(max_joints, joint_end - joint_start)
            max_dofs = max(max_dofs, int(qd_starts[joint_end]) - int(qd_starts[joint_start]))
        view.max_joints_per_articulation = max_joints
        view.max_dofs_per_articulation = max_dofs

    def _compact_mimic_constraints(
        self,
        view: ModelView,
        mimic_order: Sequence[int],
        joint_global_to_local: dict[int, int],
    ) -> None:
        joint0 = self._select_numpy_array(view, "constraint_mimic_joint0", mimic_order)
        joint1 = self._select_numpy_array(view, "constraint_mimic_joint1", mimic_order)
        if joint0 is not None:
            view.constraint_mimic_joint0 = wp.array(
                [joint_global_to_local[int(index)] for index in joint0],
                dtype=wp.int32,
                device=self.model.device,
            )
        if joint1 is not None:
            view.constraint_mimic_joint1 = wp.array(
                [joint_global_to_local[int(index)] for index in joint1],
                dtype=wp.int32,
                device=self.model.device,
            )

    def _compact_color_groups(
        self,
        view: ModelView,
        body_global_to_local: dict[int, int],
    ) -> None:
        value = self._raw_view_value(view, "body_color_groups")
        # List (length number of color groups) of body ids.
        if not isinstance(value, list):
            return
        remapped: list = []
        for group in value:
            if not isinstance(group, wp.array):
                remapped.append(group)
                continue
            local = [body_global_to_local[g] for g in (int(x) for x in group.numpy()) if g in body_global_to_local]
            remapped.append(wp.array(local, dtype=wp.int32, device=self.model.device))
        view.body_color_groups = remapped

    def _set_world_start_arrays(self, view: ModelView) -> None:
        view.body_world_start = self._world_start_array(view.body_world, int(view.body_count))
        view.shape_world_start = self._world_start_array(view.shape_world, int(view.shape_count))
        view.joint_world_start = self._world_start_array(view.joint_world, int(view.joint_count))
        view.articulation_world_start = self._world_start_array(
            view.articulation_world,
            int(view.articulation_count),
        )
        view.joint_coord_world_start = self._joint_space_world_start_array(
            view.joint_world,
            view.joint_q_start,
            int(view.joint_coord_count),
        )
        view.joint_dof_world_start = self._joint_space_world_start_array(
            view.joint_world,
            view.joint_qd_start,
            int(view.joint_dof_count),
        )
        view.joint_constraint_world_start = wp.zeros(
            int(self.model.world_count) + 2, dtype=wp.int32, device=self.model.device
        )

    def _world_start_array(self, world_array: wp.array, count: int) -> wp.array:
        world_count = int(self.model.world_count)
        starts = [0] * (world_count + 2)
        worlds = world_array.numpy() if count else np.empty(0, dtype=np.int32)
        front_global = 0
        for world in worlds:
            if int(world) == -1:
                front_global += 1
            else:
                break
        starts[0] = front_global
        counts = np.bincount(worlds[worlds >= 0].astype(np.int64), minlength=world_count) if count else []
        for world in range(world_count):
            starts[world + 1] = starts[world] + (int(counts[world]) if len(counts) else 0)
        starts[-1] = count
        return wp.array(starts, dtype=wp.int32, device=self.model.device)

    def _joint_space_world_start_array(self, joint_world: wp.array, joint_starts: wp.array, count: int) -> wp.array:
        world_count = int(self.model.world_count)
        starts = [0] * (world_count + 2)
        worlds = joint_world.numpy()
        joint_starts_np = joint_starts.numpy()
        front_global = 0
        for joint, world in enumerate(worlds):
            width = int(joint_starts_np[joint + 1]) - int(joint_starts_np[joint])
            if int(world) == -1:
                front_global += width
            else:
                break
        starts[0] = front_global
        counts = [0] * world_count
        for joint, world_id in enumerate(worlds):
            world = int(world_id)
            if world < 0:
                continue
            counts[world] += int(joint_starts_np[joint + 1]) - int(joint_starts_np[joint])
        for world in range(world_count):
            starts[world + 1] = starts[world] + counts[world]
        starts[-1] = count
        return wp.array(starts, dtype=wp.int32, device=self.model.device)

    def _compact_custom_attribute_namespaces(
        self,
        view: ModelView,
        body_order: Sequence[int],
        joint_order: Sequence[int],
        coord_order: Sequence[int],
        dof_order: Sequence[int],
        shape_order: Sequence[int],
        articulation_order: Sequence[int],
        equality_order: Sequence[int],
        mimic_order: Sequence[int],
    ) -> None:
        freq = self.model.AttributeFrequency
        index_orders_by_frequency = self._compact_index_orders_by_frequency(
            body_order,
            joint_order,
            coord_order,
            dof_order,
            shape_order,
            articulation_order,
            equality_order,
            mimic_order,
        )
        indices_by_frequency = {
            frequency: indices for frequency, (indices, _source_count) in index_orders_by_frequency.items()
        }
        body_global_to_local = {global_id: local_id for local_id, global_id in enumerate(body_order)}
        joint_global_to_local = {global_id: local_id for local_id, global_id in enumerate(joint_order)}

        for full_name, frequency in self.model.attribute_frequency.items():
            if ":" not in full_name:
                continue
            namespace_name, attr_name = full_name.split(":", 1)
            parent_ns = getattr(self.model, namespace_name, None)
            if parent_ns is None:
                continue
            try:
                value = object.__getattribute__(parent_ns, attr_name)
            except AttributeError:
                continue
            if value is None:
                continue
            overrides = object.__getattribute__(view, "_overrides")
            namespace = overrides.get(namespace_name)
            if namespace is None:
                namespace = type(parent_ns)(namespace_name)
                for name in dir(parent_ns):
                    if name.startswith("_") or hasattr(type(parent_ns), name):
                        continue
                    try:
                        setattr(namespace, name, object.__getattribute__(parent_ns, name))
                    except AttributeError:
                        continue
                setattr(view, namespace_name, namespace)
            if frequency in (freq.ONCE, freq.WORLD):
                setattr(namespace, attr_name, value)
                continue
            indices = indices_by_frequency.get(frequency)
            if indices is None:
                continue
            selected = self._select_attribute_value(value, indices)
            if selected is not None:
                reference_map = None
                if isinstance(frequency, str) and frequency.rsplit(":", 1)[-1] == "equality_constraint":
                    if attr_name in ("equality_constraint_body1", "equality_constraint_body2"):
                        reference_map = body_global_to_local
                    elif attr_name in ("equality_constraint_joint1", "equality_constraint_joint2"):
                        reference_map = joint_global_to_local
                if reference_map is not None:
                    selected = self._remap_compact_reference_value(selected, reference_map)
                setattr(namespace, attr_name, selected)
        self._sync_custom_frequency_namespace_metadata(view)

    def _remap_compact_reference_value(self, value, global_to_local: dict[int, int]):
        if isinstance(value, wp.array):
            remapped = [self._remap_optional_index(int(index), global_to_local) for index in value.numpy()]
            return wp.array(remapped, dtype=value.dtype, device=self.model.device)
        if isinstance(value, np.ndarray):
            return np.asarray([self._remap_optional_index(int(index), global_to_local) for index in value])
        if isinstance(value, list):
            return [self._remap_optional_index(int(index), global_to_local) for index in value]
        return value

    def _sync_custom_frequency_namespace_metadata(self, view: ModelView) -> None:
        overrides = object.__getattribute__(view, "_overrides")
        for frequency, count in view.custom_frequency_counts.items():
            if ":" not in frequency:
                continue
            namespace_name, frequency_name = frequency.split(":", 1)
            namespace = overrides.get(namespace_name)
            if namespace is None:
                continue
            setattr(namespace, f"{frequency_name}_count", int(count))
            world = getattr(namespace, f"{frequency_name}_world", None)
            if isinstance(world, wp.array):
                setattr(namespace, f"{frequency_name}_world_start", self._world_start_array(world, int(count)))

    def _select_attribute_value(self, value, indices: Sequence[int]):
        if isinstance(value, wp.array):
            host = value.numpy()
            if not indices:
                selected = host[:0]
            elif host.shape[0] <= max(indices):
                return None
            else:
                selected = host[np.asarray(indices, dtype=np.int64)]
            return wp.array(selected, dtype=value.dtype, device=self.model.device)
        if isinstance(value, list):
            if not indices:
                return []
            if len(value) <= max(indices):
                return None
            return [value[int(index)] for index in indices]
        if isinstance(value, np.ndarray):
            if not indices:
                return value[:0]
            if value.shape[0] <= max(indices):
                return None
            return value[np.asarray(indices, dtype=np.int64)]
        return None

    def _articulation_prefix_count(self, joint_prefix: int) -> int | None:
        """Return articulation count for a joint prefix, if it ends on a boundary."""
        if joint_prefix == 0:
            return 0
        starts = self.model.articulation_start.numpy() if self.model.articulation_start is not None else []
        for index, start in enumerate(starts):
            if int(start) == joint_prefix:
                return index
            if int(start) > joint_prefix:
                return None
        return None

    def _after_entries_constructed(self) -> None:
        """Hook called after sub-solvers are constructed and before state allocation."""

    def _after_entry_states_created(self) -> None:
        """Hook called after per-entry states and scratch buffers are allocated."""
        self._refresh_gravity_accelerations()

    def _entry_needs_gravity_acceleration(self, entry: SolverEntry) -> bool:
        del entry
        return False

    def _gravity_acceleration_refresh_flags(self) -> int:
        return int(ModelFlags.MODEL_PROPERTIES | ModelFlags.BODY_INERTIAL_PROPERTIES)

    def _refresh_gravity_accelerations(self) -> None:
        for entry in self._entries.values():
            self._refresh_entry_gravity_acceleration(entry)

    def _refresh_entry_gravity_acceleration(self, entry: SolverEntry) -> None:
        if not self._entry_needs_gravity_acceleration(entry):
            entry.body_gravity_acceleration = None
            entry.particle_gravity_acceleration = None
            return

        body_count = int(entry.view.body_count)
        particle_count = int(entry.view.particle_count)
        device = self.model.device

        if body_count > 0:
            if entry.body_gravity_acceleration is None or entry.body_gravity_acceleration.shape[0] != body_count:
                entry.body_gravity_acceleration = wp.empty(body_count, dtype=wp.vec3, device=device)
        else:
            entry.body_gravity_acceleration = None

        if particle_count > 0:
            if (
                entry.particle_gravity_acceleration is None
                or entry.particle_gravity_acceleration.shape[0] != particle_count
            ):
                entry.particle_gravity_acceleration = wp.empty(particle_count, dtype=wp.vec3, device=device)
        else:
            entry.particle_gravity_acceleration = None

        entry.solver.coupling_eval_gravity_acceleration(
            entry.body_gravity_acceleration,
            entry.particle_gravity_acceleration,
        )

    def _eval_effective_masses(
        self,
        entry: SolverEntry,
        endpoint_kind: CouplingEndpointKind,
        endpoint_indices: wp.array,
        *,
        raise_on_unsupported: bool = True,
    ) -> list[float] | None:
        """Return scalar effective masses for one entry's endpoints."""
        if endpoint_indices.shape[0] == 0:
            return []

        indices = [int(i) for i in endpoint_indices.numpy()]
        local_indices = self._endpoint_indices_to_local(entry, endpoint_kind, indices)
        local_indices_array = wp.array(local_indices, dtype=int, device=self.model.device)
        endpoint_kind_array, endpoint_index, endpoint_local_pos = _coupling_endpoint_arrays(
            endpoint_kind,
            local_indices_array,
            self.model.device,
        )
        out = wp.empty(endpoint_indices.shape[0], dtype=float, device=self.model.device)

        _require_supports_coupling(entry.solver)
        try:
            entry.solver.coupling_eval_effective_mass(
                endpoint_kind_array,
                endpoint_index,
                endpoint_local_pos,
                out,
            )
        except NotImplementedError:
            if raise_on_unsupported:
                raise
            return None
        return [float(value) for value in out.numpy()]

    def _eval_effective_body_inertial_properties(
        self,
        entry: SolverEntry,
        body_indices: wp.array,
        *,
        raise_on_unsupported: bool = True,
    ) -> tuple[list[float], list[wp.mat33]] | None:
        """Return scalar mass and full inertia tensors for body endpoints."""
        if body_indices.shape[0] == 0:
            return [], []

        indices = [int(i) for i in body_indices.numpy()]
        local_indices = self._endpoint_indices_to_local(entry, CouplingEndpointKind.BODY, indices)
        local_indices_array = wp.array(local_indices, dtype=int, device=self.model.device)
        endpoint_kind_array, endpoint_index, endpoint_local_pos = _coupling_endpoint_arrays(
            CouplingEndpointKind.BODY,
            local_indices_array,
            self.model.device,
        )
        out_mass = wp.empty(body_indices.shape[0], dtype=float, device=self.model.device)
        out_inertia = wp.empty(body_indices.shape[0], dtype=wp.mat33, device=self.model.device)

        _require_supports_coupling(entry.solver)
        try:
            entry.solver.coupling_eval_effective_mass_block(
                endpoint_kind_array,
                endpoint_index,
                endpoint_local_pos,
                out_mass,
                out_inertia,
            )
        except NotImplementedError:
            if raise_on_unsupported:
                raise
            return None
        masses = [float(value) for value in out_mass.numpy()]
        inertias = [wp.mat33(np.asarray(value, dtype=np.float32)) for value in out_inertia.numpy()]
        return masses, inertias

    def _endpoint_indices_to_local(
        self,
        entry: SolverEntry,
        endpoint_kind: CouplingEndpointKind,
        indices: Sequence[int],
    ) -> list[int]:
        if int(endpoint_kind) == int(CouplingEndpointKind.BODY):
            mapping = entry.body_global_to_local.numpy()
            count = self.model.body_count
            label = "Body"
        elif int(endpoint_kind) == int(CouplingEndpointKind.PARTICLE):
            mapping = entry.particle_global_to_local.numpy()
            count = self.model.particle_count
            label = "Particle"
        else:
            raise ValueError(f"Unknown coupling endpoint kind {endpoint_kind}")
        local_indices: list[int] = []
        for index in indices:
            local = int(mapping[index]) if 0 <= index < count else -1
            if local < 0:
                raise ValueError(f"{label} {index} is not visible in coupled solver entry {entry.name!r}")
            local_indices.append(local)
        return local_indices

    def _body_indices_to_local_array(self, entry: SolverEntry, body_indices: wp.array) -> wp.array:
        if body_indices.shape[0] == 0:
            return wp.zeros(0, dtype=int, device=self.model.device)
        mapping = entry.body_global_to_local.numpy()
        local = []
        for index in body_indices.numpy():
            global_id = int(index)
            if global_id < 0 or global_id >= len(mapping):
                continue
            local_id = int(mapping[global_id])
            if local_id >= 0:
                local.append(local_id)
        return wp.array(local, dtype=int, device=self.model.device)

    def _apply_body_inertia_override(
        self,
        entry: SolverEntry,
        body_indices: wp.array,
        body_mass: wp.array,
        body_inertia: wp.array,
    ) -> None:
        """Apply body mass/inertia to the destination model view."""
        entry.view.set_body_inertial_properties(body_indices, body_mass, body_inertia)
        entry.solver.notify_model_changed(ModelFlags.BODY_INERTIAL_PROPERTIES)

    def _apply_particle_mass_override(
        self,
        entry: SolverEntry,
        particle_indices: wp.array,
        particle_mass: wp.array,
    ) -> None:
        """Apply particle mass to the destination model view."""
        entry.view.set_particle_mass(particle_indices, particle_mass)
        entry.solver.notify_model_changed(ModelFlags.MODEL_PROPERTIES)

    # ------------------------------------------------------------------
    # Sub-solver access
    # ------------------------------------------------------------------

    def solver(self, name: str) -> SolverBase:
        """Return the sub-solver registered under *name*."""
        return self._entries[name].solver

    def view(self, name: str) -> ModelView:
        """Return the :class:`ModelView` for the sub-solver *name*."""
        return self._entries[name].view

    def entry_names(self) -> tuple[str, ...]:
        """Return coupled sub-solver entry names in stepping order."""
        return tuple(self._solver_order)

    def entry_view(self, name: str) -> ModelView:
        """Return the :class:`ModelView` for coupled sub-solver *name*."""
        return self.view(name)

    def entry_state(self, name: str, phase: Literal["current", "input", "output"] = "current") -> State:
        """Return an entry-local state suitable for visualization.

        Args:
            name: Coupled sub-solver entry name.
            phase: Which state phase to return. ``"input"`` returns the
                distributed input state, ``"output"`` returns the last
                sub-solver output state, and ``"current"`` returns output
                after the coupled solver has stepped at least once or input
                before the first step.

        Returns:
            Entry-local state whose arrays match :meth:`entry_view`.
        """
        entry = self._entries[name]
        if phase == "input":
            return entry.state_0
        if phase == "output":
            return entry.state_1
        if phase != "current":
            raise ValueError(f"Unsupported coupled entry state phase {phase!r}")
        if self._entry_output_state_valid:
            return entry.state_1
        return entry.state_0

    def entry_contacts(self, name: str, contacts: Contacts | None) -> Contacts | None:
        """Return contacts filtered for a coupled entry's view, when possible.

        Args:
            name: Coupled sub-solver entry name.
            contacts: Parent-model contacts to filter.

        Returns:
            Entry-local contacts, or ``None`` when no compatible contact buffer
            is available.
        """
        if contacts is None:
            return None
        entry = self._entries[name]
        if not entry.preserve_shape_ids:
            return None
        return self._contacts_for_entry(entry, contacts)

    def entry_output_state_valid(self) -> bool:
        """Return whether entry output states reflect the last coupled step."""
        return self._entry_output_state_valid

    def sync_entry_states(self, state_in: State, dt: float = 0.0) -> None:
        """Synchronize entry input states from a parent-model state.

        This is primarily useful for visualization before the first coupled
        step has produced entry output states.

        Args:
            state_in: Parent-model state to distribute into entry-local states.
            dt: Time step metadata forwarded to coupling input-state hooks.
        """
        self._distribute_state(state_in, dt=dt)
        self._entry_output_state_valid = False

    def reset(
        self,
        state: State,
        world_mask: wp.array | None = None,
        flags: StateFlags | int | None = None,
    ) -> None:
        """Reset coupled sub-solvers and clear coupled-solver transient state.

        Args:
            state: Parent-model simulation state to reset (modified in place).
            world_mask: Optional boolean mask of shape ``(world_count,)``
                selecting which worlds to reset. If ``None``, all worlds are
                reset.
            flags: Optional :class:`~newton.StateFlags` bitmask controlling
                which state quantities sub-solvers should reset. If ``None``,
                all state quantities are reset.
        """
        if state is None:
            raise ValueError("'state' argument is required.")

        self._distribute_state(state, iteration_restart=True)
        for entry in self._entries.values():
            entry.solver.reset(entry.state_0, world_mask=world_mask, flags=flags)
            self._sync_entry_reset_state(entry)

        self._reconcile_state(state)
        self._reset_coupling_state(state, world_mask=world_mask, flags=flags)
        self._clear_entry_contact_buffers()
        self._rebuild_entry_solver_state_caches()
        self._entry_output_state_valid = False

    # ------------------------------------------------------------------
    # SolverBase interface
    # ------------------------------------------------------------------

    def step(
        self,
        state_in: State,
        state_out: State,
        control: Control | None,
        contacts: Contacts | None,
        dt: float,
    ) -> None:
        """Step all coupled sub-solvers for one time step.

        ``contacts`` is forwarded to sub-solvers that accept it; ``SolverCoupled``
        itself does not maintain a global contact buffer. Coupling schemes that
        need a private contact pipeline (e.g. proxy collisions, ADMM internal
        contacts) own their own buffers internally.
        """
        self._distribute_state(state_in, dt=dt)
        self._step_coupled(state_in, state_out, control, contacts, dt)
        _copy_state(state_in, state_out)
        self._reconcile_state(state_out)
        self._entry_output_state_valid = True

    def prepare_contacts(self, contacts: Contacts | None) -> None:
        """Preallocate entry-local filtered contact buffers for graph capture."""
        if contacts is None:
            return
        for entry in self._entries.values():
            if entry.preserve_shape_ids:
                self._ensure_entry_contact_buffer(entry, contacts)

    def _sync_entry_reset_state(self, entry: SolverEntry) -> None:
        """Mirror a reset entry input state to persistent entry buffers."""
        if entry.state_1 is not None and entry.state_1 is not entry.state_0:
            _copy_state(entry.state_0, entry.state_1)

        for entry_state in (entry.state_0, entry.state_1):
            if entry_state is not None:
                _clear_transient_state_buffers(entry_state)

    def _reset_coupling_state(
        self,
        state: State,
        *,
        world_mask: wp.array | None = None,
        flags: StateFlags | int | None = None,
    ) -> None:
        """Hook for subclasses to clear algorithm-specific reset state."""
        del state, world_mask, flags

    def _clear_entry_contact_buffers(self) -> None:
        """Invalidate cached entry-local contact buffers after a reset."""
        for contacts in self._entry_contact_buffers.values():
            contacts.clear(bump_generation=True)
        self._entry_contact_sources.clear()

    def _rebuild_entry_solver_state_caches(self) -> None:
        """Refresh optional sub-solver spatial caches from reset entry states."""
        for entry in self._entries.values():
            rebuild_bvh = getattr(entry.solver, "rebuild_bvh", None)
            if callable(rebuild_bvh):
                rebuild_bvh(entry.state_0)

    def _step_coupled(
        self,
        state_in: State,
        state_out: State,
        control: Control | None,
        contacts: Contacts | None,
        dt: float,
    ) -> None:
        """Template method for coupling algorithms."""
        del state_out
        for name in self._solver_order:
            entry = self._entries[name]
            self._step_entry(entry, control, contacts, dt)

    # ------------------------------------------------------------------
    # State distribution and reconciliation
    # ------------------------------------------------------------------

    def _distribute_state(
        self,
        state_in: State,
        *,
        dt: float = 0.0,
        iteration_restart: bool = False,
    ) -> None:
        """Copy ``state_in`` into each sub-solver's ``state_0``."""
        for entry in self._entries.values():
            flags = self._input_state_copy_flags(state_in, entry.state_0)
            _copy_state_to_entry(state_in, entry.state_0, entry)
            self._notify_input_state_update(entry, flags, dt=dt, iteration_restart=iteration_restart)

    def _reconcile_state(self, state_out: State) -> None:
        """Merge owned sub-solver state into ``state_out``."""
        for entry in self._entries.values():
            if entry.state_1 is None:
                continue
            if entry.body_indices.shape[0] > 0 and entry.state_1.body_q is not None and state_out.body_q is not None:
                wp.launch(
                    _scatter_body_state_mapped,
                    dim=entry.body_indices.shape[0],
                    inputs=[
                        entry.body_indices,
                        entry.body_global_to_local,
                        entry.state_1.body_q,
                        entry.state_1.body_qd,
                        state_out.body_q,
                        state_out.body_qd,
                    ],
                    device=self.model.device,
                )
            if (
                entry.particle_indices.shape[0] > 0
                and entry.state_1.particle_q is not None
                and state_out.particle_q is not None
            ):
                wp.launch(
                    _scatter_particle_state_mapped,
                    dim=entry.particle_indices.shape[0],
                    inputs=[
                        entry.particle_indices,
                        entry.particle_global_to_local,
                        entry.state_1.particle_q,
                        entry.state_1.particle_qd,
                        state_out.particle_q,
                        state_out.particle_qd,
                    ],
                    device=self.model.device,
                )
            if (
                entry.joint_q_indices.shape[0] > 0
                and entry.state_1.joint_q is not None
                and state_out.joint_q is not None
            ):
                wp.launch(
                    _scatter_scalar_state_mapped,
                    dim=entry.joint_q_indices.shape[0],
                    inputs=[
                        entry.joint_q_indices,
                        entry.joint_coord_global_to_local,
                        entry.state_1.joint_q,
                        state_out.joint_q,
                    ],
                    device=self.model.device,
                )
            if (
                entry.joint_qd_indices.shape[0] > 0
                and entry.state_1.joint_qd is not None
                and state_out.joint_qd is not None
            ):
                wp.launch(
                    _scatter_scalar_state_mapped,
                    dim=entry.joint_qd_indices.shape[0],
                    inputs=[
                        entry.joint_qd_indices,
                        entry.joint_dof_global_to_local,
                        entry.state_1.joint_qd,
                        state_out.joint_qd,
                    ],
                    device=self.model.device,
                )

    # ------------------------------------------------------------------
    # Generic proxy implementation
    # ------------------------------------------------------------------

    def _clear_body_force_input(self, entry: SolverEntry) -> None:
        """Clear an entry's public body force input before mapped additions."""
        if entry.state_0.body_f is not None:
            entry.state_0.body_f.zero_()

    def _add_body_force_input(
        self,
        entry: SolverEntry,
        body_local_to_global: wp.array,
        body_f: wp.array | None,
    ) -> None:
        """Add mapped body forces to an entry's public force buffer."""
        if body_f is None or entry.state_0.body_f is None:
            return

        wp.launch(
            _add_mapped_body_forces_kernel,
            dim=body_local_to_global.shape[0],
            inputs=[body_local_to_global, body_f, entry.state_0.body_f],
            device=self.model.device,
        )

    def _set_body_force_input(
        self,
        entry: SolverEntry,
        body_f: wp.array | None,
        body_local_to_global: wp.array | None = None,
        dt: float = 0.0,
    ) -> None:
        """Replace an entry's body force input through mapped force addition."""
        self._clear_body_force_input(entry)
        if body_local_to_global is None:
            body_local_to_global = entry.body_local_to_global
        self._add_body_force_input(entry, body_local_to_global, body_f)
        if entry.state_0.body_f is not None:
            self._notify_input_state_update(entry, StateFlags.BODY_F, dt=dt)

    def _set_local_body_force_input(
        self,
        entry: SolverEntry,
        body_f: wp.array | None,
        dt: float = 0.0,
    ) -> None:
        """Replace an entry's body force input from an entry-local buffer."""
        if entry.state_0.body_f is None:
            return
        if body_f is None:
            entry.state_0.body_f.zero_()
        else:
            _copy_prefix(entry.state_0.body_f, body_f, "body_f")
        self._notify_input_state_update(entry, StateFlags.BODY_F, dt=dt)

    def _clear_particle_force_input(self, entry: SolverEntry) -> None:
        """Clear an entry's public particle force input before mapped additions."""
        if entry.state_0.particle_f is not None:
            entry.state_0.particle_f.zero_()

    def _add_particle_force_input(
        self,
        entry: SolverEntry,
        particle_local_to_global: wp.array,
        particle_f: wp.array | None,
    ) -> None:
        """Add mapped particle forces to an entry's public force buffer."""
        if particle_f is None or entry.state_0.particle_f is None:
            return

        wp.launch(
            _add_mapped_particle_forces_kernel,
            dim=particle_local_to_global.shape[0],
            inputs=[particle_local_to_global, particle_f, entry.state_0.particle_f],
            device=self.model.device,
        )

    def _set_particle_force_input(
        self,
        entry: SolverEntry,
        particle_f: wp.array | None,
        particle_local_to_global: wp.array | None = None,
        dt: float = 0.0,
    ) -> None:
        """Replace an entry's particle force input through mapped force addition."""
        self._clear_particle_force_input(entry)
        if particle_local_to_global is None:
            particle_local_to_global = entry.particle_local_to_global
        self._add_particle_force_input(entry, particle_local_to_global, particle_f)
        if entry.state_0.particle_f is not None:
            self._notify_input_state_update(entry, StateFlags.PARTICLE_F, dt=dt)

    def _set_local_particle_force_input(
        self,
        entry: SolverEntry,
        particle_f: wp.array | None,
        dt: float = 0.0,
    ) -> None:
        """Replace an entry's particle force input from an entry-local buffer."""
        if entry.state_0.particle_f is None:
            return
        if particle_f is None:
            entry.state_0.particle_f.zero_()
        else:
            _copy_prefix(entry.state_0.particle_f, particle_f, "particle_f")
        self._notify_input_state_update(entry, StateFlags.PARTICLE_F, dt=dt)

    @staticmethod
    def _input_state_copy_flags(src: State, dst: State) -> StateFlags | int:
        """Return state-array flags that ``_copy_state`` will update."""
        flags = StateFlags.NONE
        if src.body_q is not None and dst.body_q is not None:
            flags |= StateFlags.BODY_Q
            if src.body_qd is not None and dst.body_qd is not None:
                flags |= StateFlags.BODY_QD
        if dst.body_f is not None:
            flags |= StateFlags.BODY_F
        if src.particle_q is not None and dst.particle_q is not None:
            flags |= StateFlags.PARTICLE_Q
            if src.particle_qd is not None and dst.particle_qd is not None:
                flags |= StateFlags.PARTICLE_QD
        if dst.particle_f is not None:
            flags |= StateFlags.PARTICLE_F
        if src.joint_q is not None and dst.joint_q is not None:
            flags |= StateFlags.JOINT_Q
            if src.joint_qd is not None and dst.joint_qd is not None:
                flags |= StateFlags.JOINT_QD
        return flags

    def _notify_input_state_update(
        self,
        entry: SolverEntry,
        flags: StateFlags | int,
        *,
        dt: float = 0.0,
        iteration_restart: bool = False,
    ) -> None:
        """Notify custom solvers after coupler-produced input state updates."""
        flags = StateFlags(flags)
        if flags == StateFlags.NONE and not iteration_restart:
            return
        _require_supports_coupling(entry.solver)
        entry.solver.coupling_notify_input_state_update(
            entry.state_0, flags, iteration_restart=iteration_restart, dt=dt
        )

    def _step_entry(
        self,
        entry: SolverEntry,
        control: Control | None,
        contacts: Contacts | None,
        dt: float,
        *,
        filter_contacts: bool = True,
    ) -> Contacts | None:
        """Step one sub-solver entry, honoring its local substep count."""
        if filter_contacts:
            contacts = self._contacts_for_entry(entry, contacts)
        control = _copy_control_to_entry(control, entry)
        if entry.in_place:
            substep_dt = dt / float(entry.substeps)
            for _ in range(entry.substeps):
                entry.solver.step(entry.state_0, entry.state_0, control, contacts, substep_dt)
            return contacts

        if entry.substeps == 1:
            entry.solver.step(entry.state_0, entry.state_1, control, contacts, dt)
            return contacts

        substep_dt = dt / float(entry.substeps)
        if entry.state_tmp is None:
            raise RuntimeError(f"SolverCoupled.Entry {entry.name!r} is missing a substep scratch state")
        _copy_state(entry.state_0, entry.state_1)
        state_in = entry.state_1
        state_out = entry.state_tmp
        for substep in range(entry.substeps):
            if substep > 0:
                _copy_forces(entry.state_0, state_in)
            entry.solver.step(state_in, state_out, control, contacts, substep_dt)
            state_in, state_out = state_out, state_in
        if state_in is entry.state_tmp:
            _copy_state(entry.state_tmp, entry.state_1)
        return contacts

    def _contacts_for_entry(self, entry: SolverEntry, contacts: Contacts | None) -> Contacts | None:
        if contacts is None or not entry.preserve_shape_ids:
            return contacts
        if contacts is self._entry_contact_buffers.get(entry.name):
            return contacts

        filtered = self._ensure_entry_contact_buffer(entry, contacts)
        force_contact_update = int(self._entry_contact_sources.get(entry.name) is not contacts)
        if force_contact_update:
            self._entry_contact_sources[entry.name] = contacts

        if filtered.rigid_contact_force is not None:
            filtered.rigid_contact_force.zero_()
        if filtered.force is not None:
            filtered.force.zero_()

        contact_update_dim = max(
            contacts.rigid_contact_max, contacts.soft_contact_max, filtered.contact_counters.shape[0]
        )
        if contact_update_dim > 0:
            wp.launch(
                _prepare_filtered_contact_update_kernel,
                dim=contact_update_dim,
                inputs=[
                    contacts.contact_generation,
                    self._entry_rigid_contact_generation[entry.name],
                    self._entry_soft_contact_generation[entry.name],
                    self._entry_rigid_contact_update[entry.name],
                    self._entry_soft_contact_update[entry.name],
                    filtered.contact_counters,
                    filtered.contact_counters.shape[0],
                    self._entry_rigid_contact_src_to_dst[entry.name],
                    contacts.rigid_contact_max,
                    self._entry_soft_contact_src_to_dst[entry.name],
                    contacts.soft_contact_max,
                    force_contact_update,
                ],
                device=self.model.device,
            )

        if contacts.rigid_contact_max > 0:
            rigid_src_to_dst = self._entry_rigid_contact_src_to_dst[entry.name]
            wp.launch(
                _filter_rigid_contacts_global_shape_ids_kernel,
                dim=contacts.rigid_contact_max,
                inputs=[
                    self._entry_rigid_contact_update[entry.name],
                    contacts.rigid_contact_count,
                    contacts.rigid_contact_shape0,
                    contacts.rigid_contact_shape1,
                    contacts.rigid_contact_point_id,
                    contacts.rigid_contact_point0,
                    contacts.rigid_contact_point1,
                    contacts.rigid_contact_offset0,
                    contacts.rigid_contact_offset1,
                    contacts.rigid_contact_normal,
                    contacts.rigid_contact_margin0,
                    contacts.rigid_contact_margin1,
                    contacts.rigid_contact_tids,
                    entry.view.shape_flags,
                    int(ShapeFlags.COLLIDE_SHAPES),
                    filtered.rigid_contact_count,
                    filtered.rigid_contact_shape0,
                    filtered.rigid_contact_shape1,
                    filtered.rigid_contact_point_id,
                    filtered.rigid_contact_point0,
                    filtered.rigid_contact_point1,
                    filtered.rigid_contact_offset0,
                    filtered.rigid_contact_offset1,
                    filtered.rigid_contact_normal,
                    filtered.rigid_contact_margin0,
                    filtered.rigid_contact_margin1,
                    filtered.rigid_contact_tids,
                    rigid_src_to_dst,
                ],
                device=self.model.device,
            )
            if contacts.rigid_contact_stiffness is not None and filtered.rigid_contact_stiffness is not None:
                wp.launch(
                    _copy_filtered_rigid_contact_properties_kernel,
                    dim=contacts.rigid_contact_max,
                    inputs=[
                        self._entry_rigid_contact_update[entry.name],
                        rigid_src_to_dst,
                        contacts.rigid_contact_stiffness,
                        contacts.rigid_contact_damping,
                        contacts.rigid_contact_friction,
                        filtered.rigid_contact_stiffness,
                        filtered.rigid_contact_damping,
                        filtered.rigid_contact_friction,
                    ],
                    device=self.model.device,
                )
            if contacts.rigid_contact_match_index is not None and filtered.rigid_contact_match_index is not None:
                wp.launch(
                    _copy_filtered_rigid_contact_match_index_kernel,
                    dim=contacts.rigid_contact_max,
                    inputs=[
                        self._entry_rigid_contact_update[entry.name],
                        rigid_src_to_dst,
                        contacts.rigid_contact_match_index,
                        filtered.rigid_contact_match_index,
                    ],
                    device=self.model.device,
                )
            if contacts.rigid_contact_diff_distance is not None and filtered.rigid_contact_diff_distance is not None:
                wp.launch(
                    _copy_filtered_rigid_contact_diff_kernel,
                    dim=contacts.rigid_contact_max,
                    inputs=[
                        self._entry_rigid_contact_update[entry.name],
                        rigid_src_to_dst,
                        contacts.rigid_contact_diff_distance,
                        contacts.rigid_contact_diff_normal,
                        contacts.rigid_contact_diff_point0_world,
                        contacts.rigid_contact_diff_point1_world,
                        filtered.rigid_contact_diff_distance,
                        filtered.rigid_contact_diff_normal,
                        filtered.rigid_contact_diff_point0_world,
                        filtered.rigid_contact_diff_point1_world,
                    ],
                    device=self.model.device,
                )

        if contacts.soft_contact_max > 0 and int(entry.view.particle_count) > 0:
            soft_src_to_dst = self._entry_soft_contact_src_to_dst[entry.name]
            wp.launch(
                _filter_soft_contacts_global_shape_ids_kernel,
                dim=contacts.soft_contact_max,
                inputs=[
                    self._entry_soft_contact_update[entry.name],
                    contacts.soft_contact_count,
                    contacts.soft_contact_particle,
                    contacts.soft_contact_shape,
                    contacts.soft_contact_body_pos,
                    contacts.soft_contact_body_vel,
                    contacts.soft_contact_normal,
                    contacts.soft_contact_tids,
                    entry.view.shape_flags,
                    entry.view.particle_flags,
                    int(ShapeFlags.COLLIDE_PARTICLES),
                    int(ParticleFlags.ACTIVE),
                    filtered.soft_contact_count,
                    filtered.soft_contact_particle,
                    filtered.soft_contact_shape,
                    filtered.soft_contact_body_pos,
                    filtered.soft_contact_body_vel,
                    filtered.soft_contact_normal,
                    filtered.soft_contact_tids,
                    soft_src_to_dst,
                ],
                device=self.model.device,
            )

        return filtered

    def _ensure_entry_contact_buffer(self, entry: SolverEntry, contacts: Contacts) -> Contacts:
        filtered = self._entry_contact_buffers.get(entry.name)
        if filtered is None or not self._entry_contact_buffer_matches(filtered, contacts):
            from ...sim import Contacts  # noqa: PLC0415

            requested = {"force"} if contacts.force is not None else None
            filtered = Contacts(
                contacts.rigid_contact_max,
                contacts.soft_contact_max,
                requires_grad=contacts.requires_grad,
                device=contacts.device,
                per_contact_shape_properties=contacts.per_contact_shape_properties,
                requested_attributes=requested,
                contact_matching=contacts.rigid_contact_match_index is not None,
            )
            self._entry_contact_buffers[entry.name] = filtered
            self._entry_contact_sources[entry.name] = contacts
            self._entry_rigid_contact_generation[entry.name] = wp.full(
                1,
                -1,
                dtype=wp.int32,
                device=contacts.device,
            )
            self._entry_soft_contact_generation[entry.name] = wp.full(
                1,
                -1,
                dtype=wp.int32,
                device=contacts.device,
            )
            self._entry_rigid_contact_update[entry.name] = wp.zeros(
                1,
                dtype=wp.int32,
                device=contacts.device,
            )
            self._entry_soft_contact_update[entry.name] = wp.zeros(
                1,
                dtype=wp.int32,
                device=contacts.device,
            )
            self._entry_rigid_contact_src_to_dst[entry.name] = wp.full(
                contacts.rigid_contact_max,
                -1,
                dtype=wp.int32,
                device=contacts.device,
            )
            self._entry_soft_contact_src_to_dst[entry.name] = wp.full(
                contacts.soft_contact_max,
                -1,
                dtype=wp.int32,
                device=contacts.device,
            )
        return filtered

    @staticmethod
    def _entry_contact_buffer_matches(filtered: Contacts, contacts: Contacts) -> bool:
        return (
            filtered.rigid_contact_max == contacts.rigid_contact_max
            and filtered.soft_contact_max == contacts.soft_contact_max
            and filtered.requires_grad == contacts.requires_grad
            and filtered.per_contact_shape_properties == contacts.per_contact_shape_properties
            and (filtered.force is not None) == (contacts.force is not None)
            and (filtered.rigid_contact_match_index is not None) == (contacts.rigid_contact_match_index is not None)
        )

    def _refresh_model_view_overrides(self, flags: int) -> None:
        """Refresh parent-derived view overrides before solver notification."""
        if not int(flags) & int(ModelFlags.BODY_INERTIAL_PROPERTIES):
            return
        for entry in self._entries.values():
            self._refresh_body_inertial_view_overrides(entry)

    def _refresh_body_inertial_view_overrides(self, entry: SolverEntry) -> None:
        """Refresh base ownership masks derived from parent body inertia."""
        if entry.body_local_to_global.shape[0] > 0:
            entry.view._refresh_body_inertial_properties(entry.body_local_to_global)
        if entry.body_dynamics_disabled_indices.shape[0] > 0:
            entry.view.disable_body_dynamics(
                self._body_indices_to_local_array(entry, entry.body_dynamics_disabled_indices)
            )

    def notify_model_changed(self, flags: int) -> None:
        """Forward model change notifications to all sub-solvers."""
        self._refresh_model_view_overrides(flags)
        for entry in self._entries.values():
            entry.solver.notify_model_changed(flags)
        if int(flags) & self._gravity_acceleration_refresh_flags():
            self._refresh_gravity_accelerations()


def _entry_control(view: ModelView) -> Control:
    """Allocate standard entry-local control arrays for a model view."""
    from ...sim import Control  # noqa: PLC0415

    control = Control()
    use_coord_layout_targets = bool(getattr(view.parent, "use_coord_layout_targets", False))
    control._use_coord_layout_targets = use_coord_layout_targets
    target_q_count = int(view.joint_coord_count if use_coord_layout_targets else view.joint_dof_count)
    dof_count = int(view.joint_dof_count)
    requires_grad = bool(getattr(view.parent, "requires_grad", False))
    device = view.parent.device
    if target_q_count or dof_count:
        if target_q_count:
            control.joint_target_q = wp.zeros(
                target_q_count,
                dtype=float,
                device=device,
                requires_grad=requires_grad,
            )
        if dof_count:
            control.joint_target_qd = wp.zeros(dof_count, dtype=float, device=device, requires_grad=requires_grad)
        control.joint_act = wp.zeros(dof_count, dtype=float, device=device, requires_grad=requires_grad)
        control.joint_f = wp.zeros(dof_count, dtype=float, device=device, requires_grad=requires_grad)
    if int(view.tri_count):
        control.tri_activations = wp.clone(view.tri_activations, requires_grad=requires_grad)
    if int(view.tet_count):
        control.tet_activations = wp.clone(view.tet_activations, requires_grad=requires_grad)
    if int(view.muscle_count):
        control.muscle_activations = wp.clone(view.muscle_activations, requires_grad=requires_grad)
    return control


def _copy_control_to_entry(src: Control | None, entry: SolverEntry) -> Control | None:
    """Copy full-model controls into an entry-local control object."""
    if src is None:
        return None
    dst = entry.control
    if dst is None:
        return src

    device = entry.view.parent.device
    dof_map = entry.joint_dof_local_to_global
    use_coord_layout_targets = bool(getattr(entry.view.parent, "use_coord_layout_targets", False))
    target_q_map = entry.joint_coord_local_to_global if use_coord_layout_targets else dof_map
    for name, local_to_global in (
        ("joint_f", dof_map),
        ("joint_target_q", target_q_map),
        ("joint_target_qd", dof_map),
        ("joint_act", dof_map),
    ):
        _copy_control_float_array(src, dst, name, local_to_global, device)
    for name in ("tri_activations", "tet_activations", "muscle_activations"):
        _copy_control_prefix_float_array(src, dst, name)
    return dst


def _copy_control_float_array(
    src_control: Control,
    dst_control: Control,
    name: str,
    local_to_global: wp.array,
    device,
) -> None:
    src = getattr(src_control, name, None)
    dst = getattr(dst_control, name, None)
    if dst is None:
        return
    if src is None:
        dst.zero_()
        return
    if int(src.shape[0]) == int(dst.shape[0]):
        wp.copy(dst, src)
        return
    wp.launch(
        _copy_mapped_float,
        dim=local_to_global.shape[0],
        inputs=[local_to_global, src, dst],
        device=device,
    )


def _copy_control_prefix_float_array(src_control: Control, dst_control: Control, name: str) -> None:
    dst = getattr(dst_control, name, None)
    if dst is None:
        return
    src = getattr(src_control, name, None)
    if src is None:
        dst.zero_()
        return
    _copy_prefix(dst, src, name)


def _copy_state_to_entry(src: State, dst: State, entry: SolverEntry) -> None:
    """Copy global state arrays into an entry state through local/global maps."""
    if src is dst:
        return
    device = entry.view.parent.device
    if src.body_q is not None and dst.body_q is not None:
        wp.launch(
            _copy_mapped_body_state,
            dim=entry.body_local_to_global.shape[0],
            inputs=[entry.body_local_to_global, src.body_q, src.body_qd, dst.body_q, dst.body_qd],
            device=device,
        )
    if dst.body_f is not None:
        if src.body_f is None:
            dst.body_f.zero_()
        else:
            wp.launch(
                _copy_mapped_spatial_vector,
                dim=entry.body_local_to_global.shape[0],
                inputs=[entry.body_local_to_global, src.body_f, dst.body_f],
                device=device,
            )
    if src.particle_q is not None and dst.particle_q is not None:
        wp.launch(
            _copy_mapped_particle_state,
            dim=entry.particle_local_to_global.shape[0],
            inputs=[
                entry.particle_local_to_global,
                src.particle_q,
                src.particle_qd,
                dst.particle_q,
                dst.particle_qd,
            ],
            device=device,
        )
    if dst.particle_f is not None:
        if src.particle_f is None:
            dst.particle_f.zero_()
        else:
            wp.launch(
                _copy_mapped_vec3,
                dim=entry.particle_local_to_global.shape[0],
                inputs=[entry.particle_local_to_global, src.particle_f, dst.particle_f],
                device=device,
            )
    if src.joint_q is not None and dst.joint_q is not None:
        wp.launch(
            _copy_mapped_float,
            dim=entry.joint_coord_local_to_global.shape[0],
            inputs=[entry.joint_coord_local_to_global, src.joint_q, dst.joint_q],
            device=device,
        )
        wp.launch(
            _copy_mapped_float,
            dim=entry.joint_dof_local_to_global.shape[0],
            inputs=[entry.joint_dof_local_to_global, src.joint_qd, dst.joint_qd],
            device=device,
        )


def _copy_state(src: State, dst: State) -> None:
    """Copy all matching state arrays from *src* to *dst*."""
    if src is dst:
        return
    if src.body_q is not None and dst.body_q is not None:
        _copy_prefix(dst.body_q, src.body_q, "body_q")
        _copy_prefix(dst.body_qd, src.body_qd, "body_qd")
    if dst.body_f is not None:
        if src.body_f is not None:
            _copy_prefix(dst.body_f, src.body_f, "body_f")
        else:
            dst.body_f.zero_()
    if src.particle_q is not None and dst.particle_q is not None:
        _copy_prefix(dst.particle_q, src.particle_q, "particle_q")
        _copy_prefix(dst.particle_qd, src.particle_qd, "particle_qd")
    if dst.particle_f is not None:
        if src.particle_f is not None:
            _copy_prefix(dst.particle_f, src.particle_f, "particle_f")
        else:
            dst.particle_f.zero_()
    if src.joint_q is not None and dst.joint_q is not None:
        _copy_prefix(dst.joint_q, src.joint_q, "joint_q")
        _copy_prefix(dst.joint_qd, src.joint_qd, "joint_qd")


def _copy_forces(src: State, dst: State) -> None:
    """Copy force buffers without disturbing positions or velocities."""
    if dst.body_f is not None:
        if src.body_f is not None:
            _copy_prefix(dst.body_f, src.body_f, "body_f")
        else:
            dst.body_f.zero_()
    if dst.particle_f is not None:
        if src.particle_f is not None:
            _copy_prefix(dst.particle_f, src.particle_f, "particle_f")
        else:
            dst.particle_f.zero_()


def _clear_transient_state_buffers(state: State) -> None:
    """Clear force and acceleration buffers that should not survive reset."""
    for name in ("body_f", "particle_f", "body_qdd", "body_parent_f"):
        array = getattr(state, name, None)
        if array is not None:
            array.zero_()


def _copy_prefix(dst: wp.array, src: wp.array, name: str) -> None:
    """Copy *src* into *dst*, allowing destination prefix views."""
    dst_len = int(dst.shape[0])
    src_len = int(src.shape[0])
    if dst_len == src_len:
        wp.copy(dst, src)
    elif dst_len < src_len:
        wp.copy(dst, src, count=dst_len)
    else:
        raise RuntimeError(f"Cannot copy {name}: source length {src_len} is smaller than destination length {dst_len}")


@wp.kernel(enable_backward=False)
def _copy_mapped_body_state(
    local_to_global: wp.array[int],
    src_body_q: wp.array[wp.transform],
    src_body_qd: wp.array[wp.spatial_vector],
    dst_body_q: wp.array[wp.transform],
    dst_body_qd: wp.array[wp.spatial_vector],
):
    local_id = wp.tid()
    global_id = local_to_global[local_id]
    if global_id < 0:
        return
    dst_body_q[local_id] = src_body_q[global_id]
    dst_body_qd[local_id] = src_body_qd[global_id]


@wp.kernel(enable_backward=False)
def _copy_mapped_particle_state(
    local_to_global: wp.array[int],
    src_particle_q: wp.array[wp.vec3],
    src_particle_qd: wp.array[wp.vec3],
    dst_particle_q: wp.array[wp.vec3],
    dst_particle_qd: wp.array[wp.vec3],
):
    local_id = wp.tid()
    global_id = local_to_global[local_id]
    if global_id < 0:
        return
    dst_particle_q[local_id] = src_particle_q[global_id]
    dst_particle_qd[local_id] = src_particle_qd[global_id]


@wp.kernel(enable_backward=False)
def _copy_mapped_spatial_vector(
    local_to_global: wp.array[int],
    src: wp.array[wp.spatial_vector],
    dst: wp.array[wp.spatial_vector],
):
    local_id = wp.tid()
    global_id = local_to_global[local_id]
    if global_id < 0:
        return
    dst[local_id] = src[global_id]


@wp.kernel(enable_backward=False)
def _copy_mapped_vec3(
    local_to_global: wp.array[int],
    src: wp.array[wp.vec3],
    dst: wp.array[wp.vec3],
):
    local_id = wp.tid()
    global_id = local_to_global[local_id]
    if global_id < 0:
        return
    dst[local_id] = src[global_id]


@wp.kernel(enable_backward=False)
def _copy_mapped_float(
    local_to_global: wp.array[int],
    src: wp.array[float],
    dst: wp.array[float],
):
    local_id = wp.tid()
    global_id = local_to_global[local_id]
    if global_id < 0:
        return
    dst[local_id] = src[global_id]


@wp.kernel(enable_backward=False)
def _scatter_body_state(
    indices: wp.array[int],
    src_body_q: wp.array[wp.transform],
    src_body_qd: wp.array[wp.spatial_vector],
    dst_body_q: wp.array[wp.transform],
    dst_body_qd: wp.array[wp.spatial_vector],
):
    i = wp.tid()
    idx = indices[i]
    dst_body_q[idx] = src_body_q[idx]
    dst_body_qd[idx] = src_body_qd[idx]


@wp.kernel(enable_backward=False)
def _scatter_particle_state(
    indices: wp.array[int],
    src_particle_q: wp.array[wp.vec3],
    src_particle_qd: wp.array[wp.vec3],
    dst_particle_q: wp.array[wp.vec3],
    dst_particle_qd: wp.array[wp.vec3],
):
    i = wp.tid()
    idx = indices[i]
    dst_particle_q[idx] = src_particle_q[idx]
    dst_particle_qd[idx] = src_particle_qd[idx]


@wp.kernel(enable_backward=False)
def _scatter_scalar_state(
    indices: wp.array[int],
    src: wp.array[float],
    dst: wp.array[float],
):
    i = wp.tid()
    idx = indices[i]
    dst[idx] = src[idx]


@wp.kernel(enable_backward=False)
def _scatter_body_state_mapped(
    indices: wp.array[int],
    global_to_local: wp.array[int],
    src_body_q: wp.array[wp.transform],
    src_body_qd: wp.array[wp.spatial_vector],
    dst_body_q: wp.array[wp.transform],
    dst_body_qd: wp.array[wp.spatial_vector],
):
    i = wp.tid()
    global_id = indices[i]
    local_id = global_to_local[global_id]
    if local_id < 0:
        return
    dst_body_q[global_id] = src_body_q[local_id]
    dst_body_qd[global_id] = src_body_qd[local_id]


@wp.kernel(enable_backward=False)
def _scatter_particle_state_mapped(
    indices: wp.array[int],
    global_to_local: wp.array[int],
    src_particle_q: wp.array[wp.vec3],
    src_particle_qd: wp.array[wp.vec3],
    dst_particle_q: wp.array[wp.vec3],
    dst_particle_qd: wp.array[wp.vec3],
):
    i = wp.tid()
    global_id = indices[i]
    local_id = global_to_local[global_id]
    if local_id < 0:
        return
    dst_particle_q[global_id] = src_particle_q[local_id]
    dst_particle_qd[global_id] = src_particle_qd[local_id]


@wp.kernel(enable_backward=False)
def _scatter_scalar_state_mapped(
    indices: wp.array[int],
    global_to_local: wp.array[int],
    src: wp.array[float],
    dst: wp.array[float],
):
    i = wp.tid()
    global_id = indices[i]
    local_id = global_to_local[global_id]
    if local_id < 0:
        return
    dst[global_id] = src[local_id]


@wp.kernel(enable_backward=False)
def _add_mapped_body_forces_kernel(
    body_local_to_global: wp.array[int],
    src_f: wp.array[wp.spatial_vector],
    dst_f: wp.array[wp.spatial_vector],
):
    local_id = wp.tid()
    global_id = body_local_to_global[local_id]
    if global_id < 0:
        return

    dst_f[local_id] = dst_f[local_id] + src_f[global_id]


@wp.kernel(enable_backward=False)
def _add_mapped_particle_forces_kernel(
    particle_local_to_global: wp.array[int],
    src_f: wp.array[wp.vec3],
    dst_f: wp.array[wp.vec3],
):
    local_id = wp.tid()
    global_id = particle_local_to_global[local_id]
    if global_id < 0:
        return

    dst_f[local_id] = dst_f[local_id] + src_f[global_id]


@wp.kernel(enable_backward=False)
def _prepare_filtered_contact_update_kernel(
    src_generation: wp.array[wp.int32],
    rigid_generation: wp.array[wp.int32],
    soft_generation: wp.array[wp.int32],
    rigid_update_out: wp.array[wp.int32],
    soft_update_out: wp.array[wp.int32],
    dst_counters: wp.array[wp.int32],
    counter_count: int,
    rigid_src_to_dst: wp.array[wp.int32],
    rigid_contact_max: int,
    soft_src_to_dst: wp.array[wp.int32],
    soft_contact_max: int,
    force_update: int,
):
    tid = wp.tid()
    rigid_update = wp.int32(0)
    if force_update != 0 or src_generation[0] != rigid_generation[0]:
        rigid_update = wp.int32(1)
    soft_update = wp.int32(0)
    if force_update != 0 or src_generation[0] != soft_generation[0]:
        soft_update = wp.int32(1)
    if tid == 0:
        rigid_update_out[0] = rigid_update
        soft_update_out[0] = soft_update
        if rigid_update != 0:
            rigid_generation[0] = src_generation[0]
        if soft_update != 0:
            soft_generation[0] = src_generation[0]
        if rigid_update != 0 and counter_count > 0:
            dst_counters[0] = 0
        if soft_update != 0 and counter_count > 1:
            dst_counters[1] = 0
    if rigid_update != 0 and tid < rigid_contact_max:
        rigid_src_to_dst[tid] = -1
    if soft_update != 0 and tid < soft_contact_max:
        soft_src_to_dst[tid] = -1


@wp.kernel(enable_backward=False)
def _filter_rigid_contacts_global_shape_ids_kernel(
    update_filter: wp.array[wp.int32],
    src_count: wp.array[wp.int32],
    src_shape0: wp.array[wp.int32],
    src_shape1: wp.array[wp.int32],
    src_point_id: wp.array[wp.int32],
    src_point0: wp.array[wp.vec3],
    src_point1: wp.array[wp.vec3],
    src_offset0: wp.array[wp.vec3],
    src_offset1: wp.array[wp.vec3],
    src_normal: wp.array[wp.vec3],
    src_margin0: wp.array[wp.float32],
    src_margin1: wp.array[wp.float32],
    src_tids: wp.array[wp.int32],
    shape_flags: wp.array[wp.int32],
    collide_mask: int,
    dst_count: wp.array[wp.int32],
    dst_shape0: wp.array[wp.int32],
    dst_shape1: wp.array[wp.int32],
    dst_point_id: wp.array[wp.int32],
    dst_point0: wp.array[wp.vec3],
    dst_point1: wp.array[wp.vec3],
    dst_offset0: wp.array[wp.vec3],
    dst_offset1: wp.array[wp.vec3],
    dst_normal: wp.array[wp.vec3],
    dst_margin0: wp.array[wp.float32],
    dst_margin1: wp.array[wp.float32],
    dst_tids: wp.array[wp.int32],
    src_to_dst: wp.array[wp.int32],
):
    if update_filter[0] == 0:
        return

    contact_id = wp.tid()
    if contact_id >= src_count[0]:
        return

    shape0 = src_shape0[contact_id]
    shape1 = src_shape1[contact_id]
    if shape0 < 0 or shape1 < 0:
        return
    if shape0 >= shape_flags.shape[0] or shape1 >= shape_flags.shape[0]:
        return
    if (shape_flags[shape0] & collide_mask) == 0 or (shape_flags[shape1] & collide_mask) == 0:
        return

    dst_id = wp.atomic_add(dst_count, 0, wp.int32(1))
    src_to_dst[contact_id] = dst_id

    dst_shape0[dst_id] = shape0
    dst_shape1[dst_id] = shape1
    dst_point_id[dst_id] = src_point_id[contact_id]
    dst_point0[dst_id] = src_point0[contact_id]
    dst_point1[dst_id] = src_point1[contact_id]
    dst_offset0[dst_id] = src_offset0[contact_id]
    dst_offset1[dst_id] = src_offset1[contact_id]
    dst_normal[dst_id] = src_normal[contact_id]
    dst_margin0[dst_id] = src_margin0[contact_id]
    dst_margin1[dst_id] = src_margin1[contact_id]
    dst_tids[dst_id] = src_tids[contact_id]


@wp.kernel(enable_backward=False)
def _copy_filtered_rigid_contact_properties_kernel(
    update_filter: wp.array[wp.int32],
    src_to_dst: wp.array[wp.int32],
    src_stiffness: wp.array[wp.float32],
    src_damping: wp.array[wp.float32],
    src_friction: wp.array[wp.float32],
    dst_stiffness: wp.array[wp.float32],
    dst_damping: wp.array[wp.float32],
    dst_friction: wp.array[wp.float32],
):
    if update_filter[0] == 0:
        return

    src_id = wp.tid()
    dst_id = src_to_dst[src_id]
    if dst_id < 0:
        return

    dst_stiffness[dst_id] = src_stiffness[src_id]
    dst_damping[dst_id] = src_damping[src_id]
    dst_friction[dst_id] = src_friction[src_id]


@wp.kernel(enable_backward=False)
def _copy_filtered_rigid_contact_match_index_kernel(
    update_filter: wp.array[wp.int32],
    src_to_dst: wp.array[wp.int32],
    src_match_index: wp.array[wp.int32],
    dst_match_index: wp.array[wp.int32],
):
    if update_filter[0] == 0:
        return

    src_id = wp.tid()
    dst_id = src_to_dst[src_id]
    if dst_id < 0:
        return

    match_id = src_match_index[src_id]
    if match_id < 0:
        dst_match_index[dst_id] = match_id
    else:
        dst_match_index[dst_id] = -1


@wp.kernel(enable_backward=False)
def _copy_filtered_rigid_contact_diff_kernel(
    update_filter: wp.array[wp.int32],
    src_to_dst: wp.array[wp.int32],
    src_distance: wp.array[wp.float32],
    src_normal: wp.array[wp.vec3],
    src_point0_world: wp.array[wp.vec3],
    src_point1_world: wp.array[wp.vec3],
    dst_distance: wp.array[wp.float32],
    dst_normal: wp.array[wp.vec3],
    dst_point0_world: wp.array[wp.vec3],
    dst_point1_world: wp.array[wp.vec3],
):
    if update_filter[0] == 0:
        return

    src_id = wp.tid()
    dst_id = src_to_dst[src_id]
    if dst_id < 0:
        return

    dst_distance[dst_id] = src_distance[src_id]
    dst_normal[dst_id] = src_normal[src_id]
    dst_point0_world[dst_id] = src_point0_world[src_id]
    dst_point1_world[dst_id] = src_point1_world[src_id]


@wp.kernel(enable_backward=False)
def _filter_soft_contacts_global_shape_ids_kernel(
    update_filter: wp.array[wp.int32],
    src_count: wp.array[wp.int32],
    src_particle: wp.array[int],
    src_shape: wp.array[int],
    src_body_pos: wp.array[wp.vec3],
    src_body_vel: wp.array[wp.vec3],
    src_normal: wp.array[wp.vec3],
    src_tids: wp.array[int],
    shape_flags: wp.array[wp.int32],
    particle_flags: wp.array[wp.int32],
    collide_particles_mask: int,
    active_particle_mask: int,
    dst_count: wp.array[wp.int32],
    dst_particle: wp.array[int],
    dst_shape: wp.array[int],
    dst_body_pos: wp.array[wp.vec3],
    dst_body_vel: wp.array[wp.vec3],
    dst_normal: wp.array[wp.vec3],
    dst_tids: wp.array[int],
    src_to_dst: wp.array[wp.int32],
):
    if update_filter[0] == 0:
        return

    contact_id = wp.tid()
    if contact_id >= src_count[0]:
        return

    particle = src_particle[contact_id]
    shape = src_shape[contact_id]
    if particle < 0 or shape < 0:
        return
    if particle >= particle_flags.shape[0] or shape >= shape_flags.shape[0]:
        return
    if (shape_flags[shape] & collide_particles_mask) == 0:
        return
    if (particle_flags[particle] & active_particle_mask) == 0:
        return

    dst_id = wp.atomic_add(dst_count, 0, wp.int32(1))
    src_to_dst[contact_id] = dst_id

    dst_particle[dst_id] = particle
    dst_shape[dst_id] = shape
    dst_body_pos[dst_id] = src_body_pos[contact_id]
    dst_body_vel[dst_id] = src_body_vel[contact_id]
    dst_normal[dst_id] = src_normal[contact_id]
    dst_tids[dst_id] = src_tids[contact_id]
