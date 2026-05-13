# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Unified coupled multi-solver API."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import warp as wp

from ...geometry import ShapeFlags
from ..coupling import (
    COUPLING_HOOK_METHOD,
    CouplingEndpointKind,
    CouplingHook,
    CouplingInputStateFlags,
    CouplingInterface,
)
from ..flags import SolverNotifyFlags
from ..solver import SolverBase
from .model_view import ModelView

if TYPE_CHECKING:
    from ...sim import Contacts, Control, Model, State


def _identity_index_map(count: int, device) -> wp.array:
    """Return a dense local-to-global identity map."""
    return wp.array(list(range(count)), dtype=int, device=device)


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


_UNSUPPORTED = object()
"""Sentinel returned by :func:`_coupling_method_or_fallback` when the solver
declared the hook unsupported and ``raise_on_unsupported`` is False."""


def _require_supports_coupling(solver: SolverBase) -> None:
    if not isinstance(solver, CouplingInterface):
        raise TypeError(
            f"{type(solver).__name__} cannot participate in a coupled simulation; "
            "inherit CouplingInterface and override the hook methods it needs."
        )


def _coupling_method_or_fallback(
    solver: SolverBase,
    hook: CouplingHook,
    *,
    raise_on_unsupported: bool = True,
):
    """Look up a solver's hook method.

    Returns the bound method to call if the solver overrides the hook, ``None``
    if the solver supports the hook but expects the wrapper fallback, or
    :data:`_UNSUPPORTED` if the solver declared the hook unsupported and
    ``raise_on_unsupported`` is False. Raises :class:`NotImplementedError`
    when the hook is unsupported and ``raise_on_unsupported`` is True, or
    :class:`TypeError` when the solver does not inherit :class:`CouplingInterface`.
    """
    _require_supports_coupling(solver)
    if hook in solver.coupling_unsupported:
        if raise_on_unsupported:
            raise NotImplementedError(f"{type(solver).__name__} declared coupling hook {hook.name} as unsupported")
        return _UNSUPPORTED
    method = getattr(solver, COUPLING_HOOK_METHOD[hook], None)
    return method if callable(method) else None


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
    joint_q_indices: wp.array
    joint_qd_indices: wp.array
    shape_indices: wp.array
    body_local_to_global: wp.array
    particle_local_to_global: wp.array
    joint_dof_local_to_global: wp.array
    in_place: bool
    state_0: State | None = None
    state_1: State | None = None
    state_tmp: State | None = None
    state_tmp_1: State | None = None
    body_force_input: wp.array = field(default=None)
    particle_force_input: wp.array = field(default=None)


class SolverCoupled(SolverBase, CouplingInterface):
    """Couple multiple solvers through explicit ownership and coupling config.

    ``SolverCoupled`` owns generic mechanics that can be derived from
    ``Model``, ``ModelView`` and ``State``: per-solver views, ownership masks,
    state distribution/reconciliation, per-entry substeps, and shared coupling
    hook dispatch helpers. Algorithm-specific couplers such as
    :class:`~newton.solvers.SolverAdmmCoupled` and
    :class:`~newton.solvers.SolverProxyCoupled` derive from this base class.

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
        that explicitly support it and currently requires ``substeps=1``.
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

        self._validate_entry_names()
        self._body_owner = self._build_owner_map(model.body_count, [e.bodies for e in self._entry_configs])
        self._particle_owner = self._build_owner_map(model.particle_count, [e.particles for e in self._entry_configs])
        self._joint_owner = self._build_owner_map(model.joint_count, [e.joints for e in self._entry_configs])

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
            if cfg.in_place and substeps != 1:
                raise ValueError(f"SolverCoupled.Entry {cfg.name!r} in_place requires substeps=1")

            body_indices = wp.array([int(i) for i in cfg.bodies], dtype=int, device=device)
            particle_indices = wp.array([int(i) for i in cfg.particles], dtype=int, device=device)
            joint_indices = wp.array([int(i) for i in cfg.joints], dtype=int, device=device)
            shape_indices = wp.array([int(i) for i in cfg.shapes], dtype=int, device=device)
            joint_q_indices, joint_qd_indices = self._joint_state_indices(cfg.joints)
            body_local_to_global = _identity_index_map(model.body_count, device)
            particle_local_to_global = _identity_index_map(model.particle_count, device)
            joint_dof_local_to_global = _identity_index_map(model.joint_dof_count, device)

            view = ModelView(model, cfg.name)
            proxy_body_keep: set[int] = set()

            if any_body_owner:
                proxy_body_keep = self._entry_proxy_body_keep_indices(cfg.name)
                to_zero = [i for i, owner in enumerate(self._body_owner) if owner != idx and i not in proxy_body_keep]
                if to_zero:
                    view.zero_body_mass(wp.array(to_zero, dtype=int, device=device))
                if proxy_body_keep:
                    view.mark_proxy_bodies(wp.array(sorted(proxy_body_keep), dtype=int, device=device))

            if any_particle_owner:
                proxy_keep = self._entry_proxy_particle_keep_indices(cfg.name)
                to_zero = [i for i, owner in enumerate(self._particle_owner) if owner != idx and i not in proxy_keep]
                if to_zero:
                    to_zero_array = wp.array(to_zero, dtype=int, device=device)
                    view.zero_particle_mass(to_zero_array)
                    view.deactivate_particles(to_zero_array)
                if proxy_keep:
                    view.mark_proxy_particles(wp.array(sorted(proxy_keep), dtype=int, device=device))

            if any_joint_owner:
                to_disable = [i for i, owner in enumerate(self._joint_owner) if owner != idx]
                if to_disable:
                    view.disable_joints(wp.array(to_disable, dtype=int, device=device))

            self._apply_entry_shape_visibility(view, cfg, proxy_body_keep)
            self._customize_view(cfg.name, view, body_indices)
            if cfg.configure_view is not None:
                cfg.configure_view(view)
            self._filter_shape_contact_pairs(view)

            solver = cfg.solver(view)
            self._entries[cfg.name] = SolverEntry(
                name=cfg.name,
                solver=solver,
                substeps=substeps,
                view=view,
                body_indices=body_indices,
                particle_indices=particle_indices,
                joint_indices=joint_indices,
                joint_q_indices=joint_q_indices,
                joint_qd_indices=joint_qd_indices,
                shape_indices=shape_indices,
                body_local_to_global=body_local_to_global,
                particle_local_to_global=particle_local_to_global,
                joint_dof_local_to_global=joint_dof_local_to_global,
                in_place=bool(cfg.in_place),
            )

        self._after_entries_constructed()

        for entry in self._entries.values():
            entry.state_0 = entry.view.state()
            entry.state_1 = entry.state_0 if entry.in_place else entry.view.state()
            if model.body_count:
                entry.body_force_input = wp.zeros(model.body_count, dtype=wp.spatial_vector, device=device)
            if model.particle_count:
                entry.particle_force_input = wp.zeros(model.particle_count, dtype=wp.vec3, device=device)
            if entry.substeps > 1:
                entry.state_tmp = entry.view.state()
                entry.state_tmp_1 = entry.view.state()

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
        if not cfg.shapes:
            return

        model = self.model
        if model.shape_count == 0 or model.shape_flags is None:
            return

        visible = {int(i) for i in cfg.shapes}
        for shape_id in visible:
            if shape_id < 0 or shape_id >= model.shape_count:
                raise IndexError(f"Shape ownership index {shape_id} out of range for count {model.shape_count}")

        shape_body = getattr(model, "shape_body", None)
        if proxy_body_keep and shape_body is not None:
            shape_body_np = shape_body.numpy()
            for shape_id, body_id in enumerate(shape_body_np):
                if int(body_id) in proxy_body_keep:
                    visible.add(shape_id)

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

    def _after_entries_constructed(self) -> None:
        """Hook called after sub-solvers are constructed and before state allocation."""

    def _after_entry_states_created(self) -> None:
        """Hook called after per-entry states and scratch buffers are allocated."""

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

        method = _coupling_method_or_fallback(
            entry.solver,
            CouplingHook.EFFECTIVE_MASS_DIAGONAL,
            raise_on_unsupported=raise_on_unsupported,
        )
        if method is _UNSUPPORTED:
            return None

        indices = [int(i) for i in endpoint_indices.numpy()]
        if method is not None:
            endpoint_kind_array, endpoint_index, endpoint_local_pos = _coupling_endpoint_arrays(
                endpoint_kind,
                endpoint_indices,
                self.model.device,
            )
            out = wp.empty(endpoint_indices.shape[0], dtype=float, device=self.model.device)
            method(
                endpoint_kind_array,
                endpoint_index,
                endpoint_local_pos,
                out,
            )
            return [float(value) for value in out.numpy()]

        if int(endpoint_kind) == int(CouplingEndpointKind.BODY):
            mass = entry.view.body_mass.numpy() if entry.view.body_mass is not None else []
        elif int(endpoint_kind) == int(CouplingEndpointKind.PARTICLE):
            mass = entry.view.particle_mass.numpy() if entry.view.particle_mass is not None else []
        else:
            raise ValueError(f"Unknown coupling endpoint kind {endpoint_kind}")
        return [float(mass[index]) if len(mass) > index else 0.0 for index in indices]

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
        block_method = _coupling_method_or_fallback(
            entry.solver,
            CouplingHook.EFFECTIVE_MASS_BLOCK,
            raise_on_unsupported=raise_on_unsupported,
        )
        if block_method is _UNSUPPORTED:
            return None

        if block_method is not None:
            endpoint_kind_array, endpoint_index, endpoint_local_pos = _coupling_endpoint_arrays(
                CouplingEndpointKind.BODY,
                body_indices,
                self.model.device,
            )
            out_mass = wp.empty(body_indices.shape[0], dtype=float, device=self.model.device)
            out_inertia = wp.empty(body_indices.shape[0], dtype=wp.mat33, device=self.model.device)
            block_method(
                endpoint_kind_array,
                endpoint_index,
                endpoint_local_pos,
                out_mass,
                out_inertia,
            )
            masses = [float(value) for value in out_mass.numpy()]
            inertias = [wp.mat33(np.asarray(value, dtype=np.float32)) for value in out_inertia.numpy()]
            return masses, inertias

        masses = self._eval_effective_masses(
            entry,
            CouplingEndpointKind.BODY,
            body_indices,
            raise_on_unsupported=raise_on_unsupported,
        )
        if masses is None:
            return None

        model_mass = entry.view.body_mass.numpy() if entry.view.body_mass is not None else []
        model_inertia = entry.view.body_inertia.numpy() if entry.view.body_inertia is not None else []
        inertias = []
        for index, mass in zip(indices, masses, strict=True):
            if len(model_inertia) <= index:
                inertias.append(wp.mat33(0.0))
                continue
            inertia = np.asarray(model_inertia[index], dtype=np.float32)
            base_mass = float(model_mass[index]) if len(model_mass) > index else 0.0
            if base_mass > 0.0:
                inertia = inertia * (float(mass) / base_mass)
            inertias.append(wp.mat33(inertia))
        return masses, inertias

    def _apply_body_inertia_override(
        self,
        entry: SolverEntry,
        body_indices: wp.array,
        body_mass: wp.array,
        body_inertia: wp.array,
    ) -> None:
        """Apply body mass/inertia to the destination model view."""
        entry.view.set_body_inertial_properties(body_indices, body_mass, body_inertia)
        entry.solver.notify_model_changed(SolverNotifyFlags.BODY_INERTIAL_PROPERTIES)

    def _apply_particle_mass_override(
        self,
        entry: SolverEntry,
        particle_indices: wp.array,
        particle_mass: wp.array,
    ) -> None:
        """Apply particle mass to the destination model view."""
        entry.view.set_particle_mass(particle_indices, particle_mass)
        entry.solver.notify_model_changed(SolverNotifyFlags.MODEL_PROPERTIES)

    # ------------------------------------------------------------------
    # Sub-solver access
    # ------------------------------------------------------------------

    def solver(self, name: str) -> SolverBase:
        """Return the sub-solver registered under *name*."""
        return self._entries[name].solver

    def view(self, name: str) -> ModelView:
        """Return the :class:`ModelView` for the sub-solver *name*."""
        return self._entries[name].view

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
        restart: bool = False,
    ) -> None:
        """Copy ``state_in`` into each sub-solver's ``state_0``."""
        for entry in self._entries.values():
            flags = self._input_state_copy_flags(state_in, entry.state_0)
            _copy_state(state_in, entry.state_0)
            self._notify_input_state_update(entry, flags, dt=dt, restart=restart)

    def _reconcile_state(self, state_out: State) -> None:
        """Merge owned sub-solver state into ``state_out``."""
        for entry in self._entries.values():
            if entry.state_1 is None:
                continue
            if entry.body_indices.shape[0] > 0 and entry.state_1.body_q is not None and state_out.body_q is not None:
                wp.launch(
                    _scatter_body_state,
                    dim=entry.body_indices.shape[0],
                    inputs=[
                        entry.body_indices,
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
                    _scatter_particle_state,
                    dim=entry.particle_indices.shape[0],
                    inputs=[
                        entry.particle_indices,
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
                    _scatter_scalar_state,
                    dim=entry.joint_q_indices.shape[0],
                    inputs=[entry.joint_q_indices, entry.state_1.joint_q, state_out.joint_q],
                    device=self.model.device,
                )
            if (
                entry.joint_qd_indices.shape[0] > 0
                and entry.state_1.joint_qd is not None
                and state_out.joint_qd is not None
            ):
                wp.launch(
                    _scatter_scalar_state,
                    dim=entry.joint_qd_indices.shape[0],
                    inputs=[entry.joint_qd_indices, entry.state_1.joint_qd, state_out.joint_qd],
                    device=self.model.device,
                )

    # ------------------------------------------------------------------
    # Generic proxy implementation
    # ------------------------------------------------------------------

    @staticmethod
    def _uses_custom_coupling_hook(solver: SolverBase, hook: CouplingHook) -> bool:
        _require_supports_coupling(solver)
        if hook in solver.coupling_unsupported:
            raise NotImplementedError(f"{type(solver).__name__} declared coupling hook {hook.name} as unsupported")
        return callable(getattr(solver, COUPLING_HOOK_METHOD[hook], None))

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
            self._notify_input_state_update(entry, CouplingInputStateFlags.BODY_F, dt=dt)

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
            self._notify_input_state_update(entry, CouplingInputStateFlags.PARTICLE_F, dt=dt)

    @staticmethod
    def _input_state_copy_flags(src: State, dst: State) -> CouplingInputStateFlags:
        """Return state-array flags that ``_copy_state`` will update."""
        flags = CouplingInputStateFlags(0)
        if src.body_q is not None and dst.body_q is not None:
            flags |= CouplingInputStateFlags.BODY_Q
            if src.body_qd is not None and dst.body_qd is not None:
                flags |= CouplingInputStateFlags.BODY_QD
        if dst.body_f is not None:
            flags |= CouplingInputStateFlags.BODY_F
        if src.particle_q is not None and dst.particle_q is not None:
            flags |= CouplingInputStateFlags.PARTICLE_Q
            if src.particle_qd is not None and dst.particle_qd is not None:
                flags |= CouplingInputStateFlags.PARTICLE_QD
        if dst.particle_f is not None:
            flags |= CouplingInputStateFlags.PARTICLE_F
        if src.joint_q is not None and dst.joint_q is not None:
            flags |= CouplingInputStateFlags.JOINT_Q
            if src.joint_qd is not None and dst.joint_qd is not None:
                flags |= CouplingInputStateFlags.JOINT_QD
        return flags

    def _notify_input_state_update(
        self,
        entry: SolverEntry,
        flags: CouplingInputStateFlags | int,
        *,
        dt: float = 0.0,
        restart: bool = False,
    ) -> None:
        """Notify custom solvers after coupler-produced input state updates."""
        flags = CouplingInputStateFlags(flags)
        if flags == 0 and not restart:
            return
        if self._uses_custom_coupling_hook(entry.solver, CouplingHook.NOTIFY_INPUT_STATE_UPDATE):
            entry.solver.coupling_notify_input_state_update(entry.state_0, flags, restart=restart, dt=dt)

    def _step_entry(
        self,
        entry: SolverEntry,
        control: Control | None,
        contacts: Contacts | None,
        dt: float,
    ) -> None:
        """Step one sub-solver entry, honoring its local substep count."""
        if entry.in_place:
            entry.solver.step(entry.state_0, entry.state_0, control, contacts, dt)
            return

        if entry.substeps == 1:
            entry.solver.step(entry.state_0, entry.state_1, control, contacts, dt)
            return

        substep_dt = dt / float(entry.substeps)
        if entry.state_tmp is None or entry.state_tmp_1 is None:
            raise RuntimeError(f"SolverCoupled.Entry {entry.name!r} is missing substep scratch states")
        _copy_state(entry.state_0, entry.state_tmp)
        state_in = entry.state_tmp
        for substep in range(entry.substeps):
            remaining = entry.substeps - substep - 1
            if remaining == 0:
                state_out = entry.state_1
            elif state_in is entry.state_tmp:
                state_out = entry.state_tmp_1
            else:
                state_out = entry.state_tmp
            if substep > 0:
                _copy_forces(entry.state_0, state_in)
            entry.solver.step(state_in, state_out, control, contacts, substep_dt)
            state_in = state_out

    def notify_model_changed(self, flags: int) -> None:
        """Forward model change notifications to all sub-solvers."""
        for entry in self._entries.values():
            entry.solver.notify_model_changed(flags)


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
