# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Lagged-impulse proxy coupled multi-solver simulations."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from enum import IntEnum
from typing import TYPE_CHECKING, Any

import numpy as np
import warp as wp

from ...sim import JointType, ModelFlags, StateFlags
from .interface import (
    CouplingEndpointKind,
)
from .proxy_utils import (
    blend_proxy_forces_kernel,
    restore_filtered_proxy_rigid_contacts_kernel,
    stash_proxy_forces_kernel,
    sync_proxy_particles_kernel,
    sync_proxy_states_kernel,
)
from .solver_coupled import SolverCoupled

if TYPE_CHECKING:
    from ...sim import Contacts, Control, Model, State
    from .model_view import ModelView


@dataclass
class _ProxyEntityMapping:
    """Runtime mapping from source entities to destination proxy entities.

    ``coupling_forces`` stores feedback at global proxy ids. Dense maps route
    those values to destination proxy-local ids or back to source-local ids.
    Body and particle paths share this indexing and relaxation state while
    retaining their type-specific force and state kernels.
    """

    src_name: str
    dst_name: str
    src_ids: wp.array = field(default=None)
    proxy_ids_global: wp.array = field(default=None)
    proxy_ids_local: wp.array = field(default=None)
    source_local_to_proxy_local: wp.array = field(default=None)
    source_local_to_proxy_global: wp.array = field(default=None)
    destination_local_to_proxy_global: wp.array = field(default=None)
    coupling_forces: wp.array = field(default=None)
    coupling_forces_previous: wp.array = field(default=None)
    proxy_qd_before: wp.array = field(default=None)
    mass_scale: float = 1.0
    mode: int = 0
    proxy_relaxation: float = 1.0
    proxy_relaxation_mode: int = 0
    proxy_relaxation_min: float = 0.1
    proxy_relaxation_max: float = 1.0
    aitken_residual_previous: wp.array = field(default=None)
    aitken_stats: wp.array = field(default=None)
    aitken_relaxation: wp.array = field(default=None)
    aitken_has_previous: wp.array = field(default=None)


@dataclass
class _ProxyJointMapping:
    """Runtime mapping from source joints to destination proxy joints."""

    src_name: str
    dst_name: str
    src_joint_ids: wp.array = field(default=None)
    proxy_joint_ids_global: wp.array = field(default=None)
    source_target_q_indices_global: wp.array = field(default=None)
    source_target_qd_indices_global: wp.array = field(default=None)
    destination_target_q_indices_local: wp.array = field(default=None)
    destination_target_qd_indices_local: wp.array = field(default=None)


@dataclass
class _ProxyCollisionConfig:
    """Runtime collision pipeline for one proxy source/destination solve."""

    src_name: str
    dst_name: str
    factory: Callable[[ModelView], object | None]
    collide_interval: int
    pipeline: object | None = None
    contacts: Contacts | None = None
    collide_counter: int = 0


class _ProxyMode(IntEnum):
    """Internal numeric tag for proxy state transfer modes."""

    LAGGED = 0
    STAGGERED = 1


class _ProxyRelaxationMode(IntEnum):
    """Internal numeric tag for proxy feedback relaxation modes."""

    FIXED = 0
    AITKEN = 1


_PROXY_MODE_BY_NAME = {"lagged": _ProxyMode.LAGGED, "staggered": _ProxyMode.STAGGERED}
_PROXY_RELAXATION_MODE_BY_NAME = {
    "fixed": _ProxyRelaxationMode.FIXED,
    "aitken": _ProxyRelaxationMode.AITKEN,
}


@wp.kernel(enable_backward=False)
def _copy_indexed_float_kernel(
    src_indices: wp.array[int],
    dst_indices: wp.array[int],
    src: wp.array[float],
    dst: wp.array[float],
):
    i = wp.tid()
    src_index = src_indices[i]
    dst_index = dst_indices[i]
    if src_index >= 0 and dst_index >= 0:
        dst[dst_index] = src[src_index]


@wp.kernel(enable_backward=False)
def _reset_aitken_state_kernel(
    initial_relaxation: float,
    relaxation: wp.array[float],
    has_previous: wp.array[int],
):
    relaxation[0] = initial_relaxation
    has_previous[0] = 0


@wp.kernel(enable_backward=False)
def _accumulate_aitken_stats_kernel(
    proxy_ids: wp.array[int],
    force_previous: wp.array[Any],
    force_raw: wp.array[Any],
    residual_previous: wp.array[Any],
    has_previous: wp.array[int],
    stats: wp.array[float],
):
    i = wp.tid()
    proxy_id = proxy_ids[i]
    residual = force_raw[proxy_id] - force_previous[i]
    if has_previous[0] != 0:
        delta = residual - residual_previous[i]
        wp.atomic_add(stats, 0, wp.dot(residual_previous[i], delta))
        wp.atomic_add(stats, 1, wp.dot(delta, delta))


@wp.kernel(enable_backward=False)
def _update_aitken_relaxation_kernel(
    relaxation_min: float,
    relaxation_max: float,
    stats: wp.array[float],
    relaxation: wp.array[float],
    has_previous: wp.array[int],
):
    if has_previous[0] != 0:
        denominator = stats[1]
        if denominator > 1.0e-20:
            candidate = -relaxation[0] * stats[0] / denominator
            relaxation[0] = wp.clamp(candidate, relaxation_min, relaxation_max)
    has_previous[0] = 1


@wp.kernel(enable_backward=False)
def _blend_aitken_forces_kernel(
    proxy_ids: wp.array[int],
    force_previous: wp.array[Any],
    residual_previous: wp.array[Any],
    relaxation: wp.array[float],
    force: wp.array[Any],
):
    i = wp.tid()
    proxy_id = proxy_ids[i]
    residual = force[proxy_id] - force_previous[i]
    force[proxy_id] = force_previous[i] + relaxation[0] * residual
    residual_previous[i] = residual


class SolverCoupledProxy(SolverCoupled):
    """Couple two solvers with lagged-impulse virtual proxy bodies or particles."""

    @dataclass(frozen=True)
    class Proxy:
        """Proxy mapping for virtual-inertia coupling.

        Args:
            source: Name of the source solver that owns ``bodies`` and/or
                ``particles``.
            destination: Name of the destination solver that receives proxies.
            bodies: Source body ids to map into destination proxies.
            proxy_bodies: Optional destination body ids. Defaults to
                ``bodies``.
            joints: Source joint ids to keep enabled in the destination
                proxy view. One-DoF drive targets are copied from the source
                control before the destination solve.
            proxy_joints: Optional destination joint ids. Defaults to
                ``joints``.
            mass_scale: Scale factor applied to source effective body
                mass/inertia and particle mass when assigning destination
                proxy properties. This does not modify the source modeled
                free-body mass/inertia or particle mass. Must be finite and
                positive.
            mode: Proxy transfer mode, ``"lagged"`` or ``"staggered"``.
                ``"lagged"`` syncs source begin poses and end velocities, then
                prepares proxies to avoid double-counting lagged feedback.
                ``"staggered"`` syncs source end poses and end velocities
                directly.
            proxy_relaxation: Nonnegative relaxation factor used when
                updating lagged proxy feedback after the destination solve:
                ``proxy_relaxation * coupling_forces_new + (1 - proxy_relaxation) * coupling_forces_old``.
                Values below ``1`` underrelax the update, ``1`` keeps the
                harvested force unchanged, and values above ``1`` overrelax it.
            proxy_relaxation_mode: Feedback relaxation mode. ``"fixed"`` uses
                ``proxy_relaxation`` directly. ``"aitken"`` updates it from
                consecutive feedback residuals within one solver step.
            proxy_relaxation_min: Minimum Aitken relaxation factor.
            proxy_relaxation_max: Maximum Aitken relaxation factor.
            particles: Source particle ids to map into destination proxies.
            proxy_particles: Optional destination particle ids. Defaults to
                ``particles``.
            collision_pipeline: Optional factory called as
                ``collision_pipeline(destination_model_view)``. When supplied,
                ``SolverCoupledProxy`` uses the returned pipeline to detect
                destination proxy contacts before each destination solve. If
                the factory returns ``None``, the destination solve receives
                the outer-level contacts passed to :meth:`step`.
            collide_interval: Collision-detection refresh interval for
                ``collision_pipeline``. ``None`` means every proxy pass when a
                custom pipeline is supplied. Explicit values must be positive
                integers.
        """

        source: str
        destination: str
        bodies: Sequence[int] = ()
        proxy_bodies: Sequence[int] | None = None
        joints: Sequence[int] = ()
        proxy_joints: Sequence[int] | None = None
        mass_scale: float = 1.0
        mode: str = "lagged"
        proxy_relaxation: float = 1.0
        proxy_relaxation_mode: str = "fixed"
        proxy_relaxation_min: float = 0.1
        proxy_relaxation_max: float = 1.0
        particles: Sequence[int] = ()
        proxy_particles: Sequence[int] | None = None
        collision_pipeline: Callable[[ModelView], object | None] | None = None
        collide_interval: int | None = None

    @dataclass(frozen=True)
    class Config:
        """Lagged-impulse proxy coupling configuration.

        Args:
            proxies: Directed proxy mappings between solver entries.
            iterations: Positive number of proxy relaxation passes per step.
        """

        proxies: Sequence[SolverCoupledProxy.Proxy]
        iterations: int = 1

    def __init__(
        self,
        model: Model,
        entries: Sequence[SolverCoupled.Entry],
        coupling: SolverCoupledProxy.Config,
    ) -> None:
        if len(entries) > 2:
            raise ValueError("Proxy coupling currently supports at most two solver entries")

        self._validate_config(entries, coupling)

        entry_body_sets = {entry.name: {int(i) for i in entry.bodies} for entry in entries}
        entry_particle_sets = {entry.name: {int(i) for i in entry.particles} for entry in entries}
        entry_joint_sets = {entry.name: {int(i) for i in entry.joints} for entry in entries}

        self._proxy_mappings = self._build_proxy_mappings(model, coupling, entry_body_sets)
        self._proxy_particle_mappings = self._build_proxy_particle_mappings(
            model,
            coupling,
            entry_particle_sets,
        )
        self._proxy_joint_mappings = self._build_proxy_joint_mappings(
            model,
            coupling,
            entry_body_sets,
            entry_joint_sets,
            self._proxy_body_sets_by_destination(),
        )
        self._proxy_collision_configs = self._build_proxy_collision_configs(coupling)
        self._proxy_groups = self._build_proxy_groups()

        super().__init__(
            model=model,
            entries=entries,
            coupling=coupling,
        )

    @classmethod
    def _validate_config(
        cls,
        entries: Sequence[SolverCoupled.Entry],
        coupling: SolverCoupledProxy.Config,
    ) -> None:
        cls._positive_integer(coupling.iterations, "Proxy coupling iterations")
        entry_names = {entry.name for entry in entries}
        for proxy in coupling.proxies:
            if proxy.source not in entry_names:
                raise ValueError(f"Unknown proxy source entry {proxy.source!r}")
            if proxy.destination not in entry_names:
                raise ValueError(f"Unknown proxy destination entry {proxy.destination!r}")
            if proxy.source == proxy.destination:
                raise ValueError("Proxy source and destination entries must differ")
            if not proxy.bodies and not proxy.particles and not proxy.joints:
                raise ValueError("Proxy mapping must contain at least one body, particle, or joint")

            mass_scale = float(proxy.mass_scale)
            if not np.isfinite(mass_scale) or mass_scale <= 0.0:
                raise ValueError(f"Proxy mass_scale must be finite and > 0, got {proxy.mass_scale!r}")

            cls._proxy_mode_value(proxy.mode)
            relaxation = cls._proxy_relaxation_value(proxy.proxy_relaxation)
            relaxation_mode = cls._proxy_relaxation_mode_value(proxy.proxy_relaxation_mode)
            cls._proxy_relaxation_bounds(proxy, relaxation, relaxation_mode)

            if proxy.collide_interval is not None:
                cls._positive_integer(proxy.collide_interval, "Proxy collide_interval")

    @staticmethod
    def _proxy_mode_value(mode: str) -> int:
        try:
            return int(_PROXY_MODE_BY_NAME[mode.lower()])
        except (AttributeError, KeyError) as err:
            raise ValueError(f"Unknown proxy coupling mode {mode!r}; expected 'lagged' or 'staggered'") from err

    @staticmethod
    def _proxy_relaxation_value(proxy_relaxation: float) -> float:
        relaxation = float(proxy_relaxation)
        if not np.isfinite(relaxation) or relaxation < 0.0:
            raise ValueError(f"Proxy proxy_relaxation must be finite and >= 0, got {proxy_relaxation!r}")
        return relaxation

    @staticmethod
    def _proxy_relaxation_mode_value(mode: str) -> int:
        try:
            return int(_PROXY_RELAXATION_MODE_BY_NAME[mode.lower()])
        except (AttributeError, KeyError) as err:
            raise ValueError(f"Unknown proxy relaxation mode {mode!r}; expected 'fixed' or 'aitken'") from err

    @staticmethod
    def _proxy_relaxation_bounds(
        proxy: SolverCoupledProxy.Proxy,
        relaxation: float,
        relaxation_mode: int,
    ) -> tuple[float, float]:
        relaxation_min = float(proxy.proxy_relaxation_min)
        relaxation_max = float(proxy.proxy_relaxation_max)
        if not np.isfinite(relaxation_min) or relaxation_min < 0.0:
            raise ValueError(f"Proxy proxy_relaxation_min must be finite and >= 0, got {proxy.proxy_relaxation_min!r}")
        if not np.isfinite(relaxation_max) or relaxation_max < relaxation_min:
            raise ValueError(
                "Proxy proxy_relaxation_max must be finite and >= proxy_relaxation_min, "
                f"got {proxy.proxy_relaxation_max!r}"
            )
        if int(relaxation_mode) == int(_ProxyRelaxationMode.AITKEN) and (
            relaxation < relaxation_min or relaxation > relaxation_max
        ):
            raise ValueError(
                f"Proxy proxy_relaxation must be within [{relaxation_min}, {relaxation_max}] "
                f"for Aitken relaxation, got {relaxation}"
            )
        return relaxation_min, relaxation_max

    @staticmethod
    def _validate_proxy_ids(label: str, ids: Sequence[int], count: int) -> None:
        for raw_id in ids:
            id_ = int(raw_id)
            if id_ < 0 or id_ >= count:
                raise ValueError(f"{label} id {id_} out of range [0, {count})")

    @staticmethod
    def _validate_unique_proxy_ids(label: str, ids: Sequence[int]) -> None:
        if len(ids) != len(set(ids)):
            raise ValueError(f"Duplicate {label} ids in proxy mapping")

    @staticmethod
    def _validate_proxy_destination_ids_not_owned(
        label: str,
        proxy_ids: Sequence[int],
        destination: str,
        destination_owned_ids: set[int] | None,
    ) -> None:
        if destination_owned_ids is None:
            raise ValueError(f"Unknown proxy destination entry {destination!r}")
        overlap = sorted({int(i) for i in proxy_ids} & destination_owned_ids)
        if overlap:
            raise ValueError(
                f"Proxy destination {label} ids must not be owned by destination entry {destination!r}: {overlap}"
            )

    @staticmethod
    def _validate_proxy_body_worlds(model: Model, source_ids: Sequence[int], proxy_ids: Sequence[int]) -> None:
        if model.body_world is None:
            return

        body_world = model.body_world.numpy()
        for source_id, proxy_id in zip(source_ids, proxy_ids, strict=True):
            source_world = int(body_world[source_id])
            proxy_world = int(body_world[proxy_id])
            if source_world != proxy_world:
                raise ValueError(
                    "Proxy source body and destination proxy body must live in the same world: "
                    f"source body {source_id} is in world {source_world}, "
                    f"proxy body {proxy_id} is in world {proxy_world}"
                )

    def _proxy_body_sets_by_destination(self) -> dict[str, set[int]]:
        proxy_bodies: dict[str, set[int]] = {}
        for mapping in self._proxy_mappings:
            proxy_bodies.setdefault(mapping.dst_name, set()).update(int(i) for i in mapping.proxy_ids_global.numpy())
        return proxy_bodies

    @staticmethod
    def _validate_proxy_source_ids_owned(
        label: str,
        source_ids: Sequence[int],
        source: str,
        source_owned_ids: set[int] | None,
    ) -> None:
        if source_owned_ids is None:
            raise ValueError(f"Unknown proxy source entry {source!r}")
        missing = sorted({int(i) for i in source_ids} - source_owned_ids)
        if missing:
            raise ValueError(f"Proxy source {label} ids must be owned by source entry {source!r}: {missing}")

    @staticmethod
    def _validate_proxy_joint_body_visibility(
        model: Model,
        joint_ids: Sequence[int],
        visible_bodies: set[int],
        entry_name: str,
        label: str,
    ) -> None:
        joint_parent = model.joint_parent.numpy()
        joint_child = model.joint_child.numpy()
        for joint_id in joint_ids:
            parent = int(joint_parent[joint_id])
            child = int(joint_child[joint_id])
            missing = []
            if parent >= 0 and parent not in visible_bodies:
                missing.append(parent)
            if child not in visible_bodies:
                missing.append(child)
            if missing:
                raise ValueError(
                    f"{label.capitalize()} joint {joint_id} references bodies not visible in "
                    f"coupled solver entry {entry_name!r}: {missing}"
                )

    def _build_proxy_joint_mappings(
        self,
        model: Model,
        coupling: SolverCoupledProxy.Config,
        entry_body_sets: dict[str, set[int]],
        entry_joint_sets: dict[str, set[int]],
        proxy_body_sets_by_destination: dict[str, set[int]],
    ) -> list[_ProxyJointMapping]:
        mappings = []
        device = model.device
        joint_type = model.joint_type.numpy() if model.joint_count else np.empty(0, dtype=np.int32)
        supported_types = (int(JointType.FIXED), int(JointType.PRISMATIC), int(JointType.REVOLUTE))

        for proxy in coupling.proxies:
            src_ids = [int(i) for i in proxy.joints]
            if not src_ids:
                continue
            proxy_local_ids = [int(i) for i in (proxy.proxy_joints if proxy.proxy_joints is not None else proxy.joints)]
            if len(src_ids) != len(proxy_local_ids):
                raise ValueError("Proxy source joints and proxy_joints must have the same length")
            self._validate_proxy_ids("Proxy source joint", src_ids, model.joint_count)
            self._validate_proxy_ids("Proxy destination joint", proxy_local_ids, model.joint_count)
            self._validate_unique_proxy_ids("source joint", src_ids)
            self._validate_unique_proxy_ids("proxy joint", proxy_local_ids)
            self._validate_proxy_source_ids_owned(
                "joint",
                src_ids,
                proxy.source,
                entry_joint_sets.get(proxy.source),
            )
            self._validate_proxy_destination_ids_not_owned(
                "joint",
                proxy_local_ids,
                proxy.destination,
                entry_joint_sets.get(proxy.destination),
            )

            source_visible_bodies = entry_body_sets.get(proxy.source)
            if source_visible_bodies is None:
                raise ValueError(f"Unknown proxy source entry {proxy.source!r}")
            destination_visible_bodies = set(entry_body_sets.get(proxy.destination, set()))
            destination_visible_bodies.update(proxy_body_sets_by_destination.get(proxy.destination, set()))
            self._validate_proxy_joint_body_visibility(
                model,
                src_ids,
                source_visible_bodies,
                proxy.source,
                "source",
            )
            self._validate_proxy_joint_body_visibility(
                model,
                proxy_local_ids,
                destination_visible_bodies,
                proxy.destination,
                "proxy",
            )

            for source_joint, proxy_joint in zip(src_ids, proxy_local_ids, strict=True):
                source_type = int(joint_type[source_joint])
                proxy_type = int(joint_type[proxy_joint])
                if source_type not in supported_types:
                    raise ValueError(
                        f"Unsupported proxy source joint type {JointType(source_type).name} for joint {source_joint}; "
                        "expected FIXED, PRISMATIC, or REVOLUTE"
                    )
                if proxy_type not in supported_types:
                    raise ValueError(
                        f"Unsupported proxy destination joint type {JointType(proxy_type).name} for joint "
                        f"{proxy_joint}; expected FIXED, PRISMATIC, or REVOLUTE"
                    )
                if source_type != proxy_type:
                    raise ValueError(
                        f"Proxy source joint {source_joint} and destination joint {proxy_joint} must have "
                        "matching joint types"
                    )

            mappings.append(
                _ProxyJointMapping(
                    src_name=proxy.source,
                    dst_name=proxy.destination,
                    src_joint_ids=wp.array(src_ids, dtype=int, device=device),
                    proxy_joint_ids_global=wp.array(proxy_local_ids, dtype=int, device=device),
                )
            )

        return mappings

    def _build_proxy_mappings(
        self,
        model: Model,
        coupling: SolverCoupledProxy.Config,
        entry_body_sets: dict[str, set[int]],
    ) -> list[_ProxyEntityMapping]:
        return self._build_proxy_entity_mappings(
            model,
            coupling,
            entry_body_sets,
            endpoint_kind=CouplingEndpointKind.BODY,
        )

    def _build_proxy_entity_mappings(
        self,
        model: Model,
        coupling: SolverCoupledProxy.Config,
        entry_entity_sets: dict[str, set[int]],
        *,
        endpoint_kind: CouplingEndpointKind,
    ) -> list[_ProxyEntityMapping]:
        """Build the shared indexing and relaxation state for body or particle proxies."""
        mappings = []
        device = model.device
        is_body = endpoint_kind == CouplingEndpointKind.BODY
        entity_name = "body" if is_body else "particle"
        entity_plural = "bodies" if is_body else "particles"
        entity_count = model.body_count if is_body else model.particle_count

        for proxy in coupling.proxies:
            proxy_relaxation = self._proxy_relaxation_value(proxy.proxy_relaxation)
            proxy_relaxation_mode = self._proxy_relaxation_mode_value(proxy.proxy_relaxation_mode)
            proxy_relaxation_min, proxy_relaxation_max = self._proxy_relaxation_bounds(
                proxy,
                proxy_relaxation,
                proxy_relaxation_mode,
            )
            source_values = proxy.bodies if is_body else proxy.particles
            destination_values = proxy.proxy_bodies if is_body else proxy.proxy_particles
            src_ids = [int(i) for i in source_values]
            if not src_ids:
                continue
            proxy_local_ids = [int(i) for i in (source_values if destination_values is None else destination_values)]
            if len(src_ids) != len(proxy_local_ids):
                raise ValueError(f"Proxy source {entity_plural} and proxy_{entity_plural} must have the same length")
            self._validate_proxy_ids(f"Proxy source {entity_name}", src_ids, entity_count)
            self._validate_proxy_ids(f"Proxy destination {entity_name}", proxy_local_ids, entity_count)
            self._validate_unique_proxy_ids(f"source {entity_name}", src_ids)
            self._validate_unique_proxy_ids(f"proxy {entity_name}", proxy_local_ids)
            if is_body:
                self._validate_proxy_body_worlds(model, src_ids, proxy_local_ids)
            self._validate_proxy_source_ids_owned(
                entity_name,
                src_ids,
                proxy.source,
                entry_entity_sets.get(proxy.source),
            )
            self._validate_proxy_destination_ids_not_owned(
                entity_name,
                proxy_local_ids,
                proxy.destination,
                entry_entity_sets.get(proxy.destination),
            )
            proxy_global_ids = proxy_local_ids

            source_local_to_proxy_local = [-1] * entity_count
            source_local_to_proxy_global = [-1] * entity_count
            destination_local_to_proxy_global = [-1] * entity_count
            for source_local, proxy_local, proxy_global in zip(src_ids, proxy_local_ids, proxy_global_ids, strict=True):
                source_local_to_proxy_local[source_local] = proxy_local
                source_local_to_proxy_global[source_local] = proxy_global
                destination_local_to_proxy_global[proxy_local] = proxy_global

            mappings.append(
                _ProxyEntityMapping(
                    src_name=proxy.source,
                    dst_name=proxy.destination,
                    src_ids=wp.array(src_ids, dtype=int, device=device),
                    proxy_ids_global=wp.array(proxy_global_ids, dtype=int, device=device),
                    proxy_ids_local=wp.array(proxy_local_ids, dtype=int, device=device),
                    source_local_to_proxy_local=wp.array(source_local_to_proxy_local, dtype=int, device=device),
                    source_local_to_proxy_global=wp.array(source_local_to_proxy_global, dtype=int, device=device),
                    destination_local_to_proxy_global=wp.array(
                        destination_local_to_proxy_global, dtype=int, device=device
                    ),
                    mass_scale=float(proxy.mass_scale),
                    mode=self._proxy_mode_value(proxy.mode),
                    proxy_relaxation=proxy_relaxation,
                    proxy_relaxation_mode=proxy_relaxation_mode,
                    proxy_relaxation_min=proxy_relaxation_min,
                    proxy_relaxation_max=proxy_relaxation_max,
                )
            )
        return mappings

    def _build_proxy_collision_configs(
        self,
        coupling: SolverCoupledProxy.Config,
    ) -> dict[tuple[str, str], _ProxyCollisionConfig]:
        configs: dict[tuple[str, str], _ProxyCollisionConfig] = {}
        for proxy in coupling.proxies:
            if proxy.collision_pipeline is None:
                if proxy.collide_interval is not None:
                    raise ValueError("Proxy collide_interval requires a collision_pipeline factory")
                continue
            if not callable(proxy.collision_pipeline):
                raise TypeError("Proxy collision_pipeline must be callable")

            key = (proxy.source, proxy.destination)
            collide_interval = 1 if proxy.collide_interval is None else int(proxy.collide_interval)
            existing = configs.get(key)
            if existing is not None:
                if existing.factory is not proxy.collision_pipeline or existing.collide_interval != collide_interval:
                    raise ValueError(
                        "Proxy collision_pipeline and collide_interval must match for all proxies "
                        f"from {proxy.source!r} to {proxy.destination!r}"
                    )
                continue

            configs[key] = _ProxyCollisionConfig(
                src_name=proxy.source,
                dst_name=proxy.destination,
                factory=proxy.collision_pipeline,
                collide_interval=collide_interval,
            )
        return configs

    def _build_proxy_particle_mappings(
        self,
        model: Model,
        coupling: SolverCoupledProxy.Config,
        entry_particle_sets: dict[str, set[int]],
    ) -> list[_ProxyEntityMapping]:
        return self._build_proxy_entity_mappings(
            model,
            coupling,
            entry_particle_sets,
            endpoint_kind=CouplingEndpointKind.PARTICLE,
        )

    def _entry_proxy_body_keep_indices(self, name: str) -> set[int]:
        proxy_keep: set[int] = set()
        for mapping in self._proxy_mappings:
            if mapping.dst_name == name and mapping.proxy_ids_global is not None:
                proxy_keep.update(int(i) for i in mapping.proxy_ids_global.numpy())
        return proxy_keep

    def _entry_proxy_particle_keep_indices(self, name: str) -> set[int]:
        proxy_keep: set[int] = set()
        for mapping in self._proxy_particle_mappings:
            if mapping.dst_name == name and mapping.proxy_ids_global is not None:
                proxy_keep.update(int(i) for i in mapping.proxy_ids_global.numpy())
        return proxy_keep

    def _entry_proxy_joint_keep_indices(self, name: str) -> set[int]:
        proxy_keep: set[int] = set()
        for mapping in self._proxy_joint_mappings:
            if mapping.dst_name == name and mapping.proxy_joint_ids_global is not None:
                proxy_keep.update(int(i) for i in mapping.proxy_joint_ids_global.numpy())
        return proxy_keep

    def _after_entries_constructed(self) -> None:
        self._refresh_proxy_view_maps()
        self._validate_in_place_proxy_entries()
        self._apply_proxy_effective_masses()
        self._init_proxy_collision_pipelines()

    def _refresh_proxy_view_maps(self) -> None:
        """Remap dense proxy maps to the source/destination view layouts."""
        device = self.model.device
        for mapping in self._proxy_mappings:
            src = self._entries[mapping.src_name]
            dst = self._entries[mapping.dst_name]
            self._refresh_proxy_entity_view_map(
                mapping,
                src.body_global_to_local,
                dst.body_global_to_local,
                int(src.view.body_count),
                int(dst.view.body_count),
                "body",
            )

        for mapping in self._proxy_particle_mappings:
            src = self._entries[mapping.src_name]
            dst = self._entries[mapping.dst_name]
            self._refresh_proxy_entity_view_map(
                mapping,
                src.particle_global_to_local,
                dst.particle_global_to_local,
                int(src.view.particle_count),
                int(dst.view.particle_count),
                "particle",
            )

        for mapping in self._proxy_joint_mappings:
            dst = self._entries[mapping.dst_name]
            src_joint_globals = [int(i) for i in mapping.src_joint_ids.numpy()]
            proxy_joint_globals = [int(i) for i in mapping.proxy_joint_ids_global.numpy()]
            (
                source_target_q_indices_global,
                source_target_qd_indices_global,
                destination_target_q_indices_local,
                destination_target_qd_indices_local,
            ) = self._proxy_joint_control_index_maps(dst, src_joint_globals, proxy_joint_globals)

            mapping.source_target_q_indices_global = wp.array(
                source_target_q_indices_global,
                dtype=int,
                device=device,
            )
            mapping.source_target_qd_indices_global = wp.array(
                source_target_qd_indices_global,
                dtype=int,
                device=device,
            )
            mapping.destination_target_q_indices_local = wp.array(
                destination_target_q_indices_local,
                dtype=int,
                device=device,
            )
            mapping.destination_target_qd_indices_local = wp.array(
                destination_target_qd_indices_local,
                dtype=int,
                device=device,
            )

    def _refresh_proxy_entity_view_map(
        self,
        mapping: _ProxyEntityMapping,
        source_global_to_local: wp.array,
        destination_global_to_local: wp.array,
        source_count: int,
        destination_count: int,
        entity_name: str,
    ) -> None:
        """Remap one proxy mapping from global model ids to compact view ids."""
        proxy_name = f"proxy {entity_name}"
        source_globals = [int(i) for i in mapping.src_ids.numpy()]
        proxy_globals = [int(i) for i in mapping.proxy_ids_global.numpy()]
        source_locals = self._local_ids_from_global(
            source_global_to_local,
            source_globals,
            mapping.src_name,
            entity_name,
        )
        proxy_locals = self._local_ids_from_global(
            destination_global_to_local,
            proxy_globals,
            mapping.dst_name,
            proxy_name,
        )

        source_local_to_proxy_local = [-1] * source_count
        source_local_to_proxy_global = [-1] * source_count
        destination_local_to_proxy_global = [-1] * destination_count
        for source_local, proxy_local, proxy_global in zip(
            source_locals,
            proxy_locals,
            proxy_globals,
            strict=True,
        ):
            source_local_to_proxy_local[source_local] = proxy_local
            source_local_to_proxy_global[source_local] = proxy_global
            destination_local_to_proxy_global[proxy_local] = proxy_global

        device = self.model.device
        mapping.proxy_ids_local = wp.array(proxy_locals, dtype=int, device=device)
        mapping.source_local_to_proxy_local = wp.array(source_local_to_proxy_local, dtype=int, device=device)
        mapping.source_local_to_proxy_global = wp.array(source_local_to_proxy_global, dtype=int, device=device)
        mapping.destination_local_to_proxy_global = wp.array(
            destination_local_to_proxy_global,
            dtype=int,
            device=device,
        )

    @staticmethod
    def _local_ids_from_global(
        global_to_local: wp.array,
        global_ids: Sequence[int],
        entry_name: str,
        label: str,
    ) -> list[int]:
        mapping = global_to_local.numpy()
        locals_: list[int] = []
        for global_id in global_ids:
            local_id = int(mapping[global_id]) if 0 <= global_id < len(mapping) else -1
            if local_id < 0:
                raise ValueError(
                    f"{label.capitalize()} {global_id} is not visible in coupled solver entry {entry_name!r}"
                )
            locals_.append(local_id)
        return locals_

    @staticmethod
    def _local_scalar_id_from_global(
        mapping: np.ndarray,
        global_id: int,
        entry_name: str,
        label: str,
    ) -> int:
        local_id = int(mapping[global_id]) if 0 <= global_id < len(mapping) else -1
        if local_id < 0:
            raise ValueError(f"{label.capitalize()} {global_id} is not visible in coupled solver entry {entry_name!r}")
        return local_id

    def _proxy_joint_control_index_maps(
        self,
        dst,
        src_joint_globals: Sequence[int],
        proxy_joint_globals: Sequence[int],
    ) -> tuple[list[int], list[int], list[int], list[int]]:
        model = self.model
        joint_type = model.joint_type.numpy()
        joint_qd_start = model.joint_qd_start.numpy()
        joint_target_q_start = model.joint_target_q_start.numpy()
        target_q_global_to_local = (
            dst.joint_coord_global_to_local.numpy()
            if bool(model.use_coord_layout_targets)
            else dst.joint_dof_global_to_local.numpy()
        )
        dst_qd_global_to_local = dst.joint_dof_global_to_local.numpy()

        source_target_q_indices_global: list[int] = []
        source_target_qd_indices_global: list[int] = []
        destination_target_q_indices_local: list[int] = []
        destination_target_qd_indices_local: list[int] = []

        for source_joint, proxy_joint in zip(src_joint_globals, proxy_joint_globals, strict=True):
            if int(joint_type[source_joint]) == int(JointType.FIXED):
                continue

            source_target_q_count = int(joint_target_q_start[source_joint + 1] - joint_target_q_start[source_joint])
            source_qd_count = int(joint_qd_start[source_joint + 1] - joint_qd_start[source_joint])
            proxy_target_q_count = int(joint_target_q_start[proxy_joint + 1] - joint_target_q_start[proxy_joint])
            proxy_qd_count = int(joint_qd_start[proxy_joint + 1] - joint_qd_start[proxy_joint])
            if source_target_q_count != 1 or source_qd_count != 1 or proxy_target_q_count != 1 or proxy_qd_count != 1:
                raise ValueError(
                    "Proxy joint target synchronization only supports 1-DoF joints; "
                    f"source joint {source_joint} has ({source_target_q_count}, {source_qd_count}) "
                    f"target coordinates/DOFs and destination joint {proxy_joint} has "
                    f"({proxy_target_q_count}, {proxy_qd_count}) target coordinates/DOFs"
                )

            source_target_q_global = int(joint_target_q_start[source_joint])
            source_target_qd_global = int(joint_qd_start[source_joint])
            proxy_target_q_global = int(joint_target_q_start[proxy_joint])
            proxy_qd_global = int(joint_qd_start[proxy_joint])
            source_target_q_indices_global.append(source_target_q_global)
            source_target_qd_indices_global.append(source_target_qd_global)
            destination_target_q_indices_local.append(
                self._local_scalar_id_from_global(
                    target_q_global_to_local,
                    proxy_target_q_global,
                    dst.name,
                    "destination joint target coordinate",
                )
            )
            destination_target_qd_indices_local.append(
                self._local_scalar_id_from_global(
                    dst_qd_global_to_local,
                    proxy_qd_global,
                    dst.name,
                    "destination joint target DOF",
                )
            )

        return (
            source_target_q_indices_global,
            source_target_qd_indices_global,
            destination_target_q_indices_local,
            destination_target_qd_indices_local,
        )

    def _validate_in_place_proxy_entries(self) -> None:
        for proxy in [*self._proxy_mappings, *self._proxy_particle_mappings]:
            if int(proxy.mode) != int(_ProxyMode.LAGGED):
                continue
            if self._entries[proxy.src_name].in_place:
                raise ValueError(
                    f"Proxy source entry {proxy.src_name!r} cannot use in_place=True with lagged proxy mode"
                )

    def _init_proxy_collision_pipelines(self) -> None:
        disabled_configs: list[tuple[str, str]] = []
        for key, config in self._proxy_collision_configs.items():
            dst = self._entries[config.dst_name]
            pipeline = config.factory(dst.view)
            if pipeline is None:
                # Keep the default proxy path: pass the outer-level contacts
                # through to the destination solve instead of creating a
                # proxy-local contact buffer.
                disabled_configs.append(key)
                continue
            if not callable(getattr(pipeline, "contacts", None)) or not callable(getattr(pipeline, "collide", None)):
                raise TypeError("Proxy collision_pipeline factory must return an object with contacts() and collide()")
            config.pipeline = pipeline
            config.contacts = pipeline.contacts()
        for key in disabled_configs:
            del self._proxy_collision_configs[key]

    def _proxy_collision_contacts(
        self,
        config: _ProxyCollisionConfig,
        state: State,
        *,
        iteration_restart: bool = False,
    ) -> tuple[Contacts, bool]:
        if config.pipeline is None or config.contacts is None:
            raise RuntimeError("Proxy collision pipeline was not initialized")

        # Inner proxy iterations reuse the contacts detected at iteration 0.
        # Detection margin/gap handles small proxy motion between relaxation
        # passes, so the collision cadence remains an outer-step policy.
        if iteration_restart:
            return config.contacts, False

        contacts_freshly_detected = config.collide_counter % config.collide_interval == 0
        if contacts_freshly_detected:
            config.pipeline.collide(state, config.contacts)
        config.collide_counter += 1
        return config.contacts, contacts_freshly_detected

    def get_proxy_contacts(self, source: str, destination: str) -> Contacts | None:
        """Return the internally detected contacts for one proxy direction."""
        config = self._proxy_collision_configs.get((source, destination))
        return None if config is None else config.contacts

    def get_proxy_collision_state(self) -> dict[tuple[str, str], int]:
        """Return host-side proxy collision cadence state for later restore."""
        return {key: config.collide_counter for key, config in self._proxy_collision_configs.items()}

    def restore_proxy_collision_state(self, state: dict[tuple[str, str], int]) -> None:
        """Restore host-side proxy collision cadence state."""
        for key, collide_counter in state.items():
            config = self._proxy_collision_configs.get(key)
            if config is not None:
                config.collide_counter = int(collide_counter)

    def _after_entry_states_created(self) -> None:
        super()._after_entry_states_created()
        model = self.model
        for mapping in self._proxy_mappings:
            self._initialize_proxy_feedback_state(mapping, model.body_count, wp.spatial_vector)
        for mapping in self._proxy_particle_mappings:
            self._initialize_proxy_feedback_state(mapping, model.particle_count, wp.vec3)

    def _initialize_proxy_feedback_state(
        self,
        mapping: _ProxyEntityMapping,
        entity_count: int,
        force_dtype,
    ) -> None:
        """Allocate feedback and relaxation buffers shared by body and particle proxies."""
        device = self.model.device
        proxy_count = mapping.proxy_ids_global.shape[0]
        mapping.coupling_forces = wp.zeros(entity_count, dtype=force_dtype, device=device)
        if mapping.proxy_relaxation != 1.0 or int(mapping.proxy_relaxation_mode) == int(_ProxyRelaxationMode.AITKEN):
            mapping.coupling_forces_previous = wp.zeros(proxy_count, dtype=force_dtype, device=device)
        if int(mapping.proxy_relaxation_mode) == int(_ProxyRelaxationMode.AITKEN):
            mapping.aitken_residual_previous = wp.zeros(proxy_count, dtype=force_dtype, device=device)
            mapping.aitken_stats = wp.zeros(2, dtype=float, device=device)
            mapping.aitken_relaxation = wp.array([mapping.proxy_relaxation], dtype=float, device=device)
            mapping.aitken_has_previous = wp.zeros(1, dtype=int, device=device)
        mapping.proxy_qd_before = wp.zeros(entity_count, dtype=force_dtype, device=device)

    def _entry_needs_gravity_acceleration(self, entry) -> bool:
        return any(mapping.dst_name == entry.name for mapping in self._proxy_mappings) or any(
            mapping.dst_name == entry.name for mapping in self._proxy_particle_mappings
        )

    def _reset_coupling_state(
        self,
        state: State,
        *,
        world_mask: wp.array | None = None,
        flags: StateFlags | int | None = None,
    ) -> None:
        """Clear lagged proxy feedback and collision caches after reset."""
        super()._reset_coupling_state(state, world_mask=world_mask, flags=flags)
        for mapping in [*self._proxy_mappings, *self._proxy_particle_mappings]:
            if mapping.coupling_forces is not None:
                mapping.coupling_forces.zero_()
            if mapping.coupling_forces_previous is not None:
                mapping.coupling_forces_previous.zero_()
            if mapping.aitken_residual_previous is not None:
                mapping.aitken_residual_previous.zero_()
            if mapping.aitken_stats is not None:
                mapping.aitken_stats.zero_()
            if mapping.aitken_relaxation is not None:
                mapping.aitken_relaxation.fill_(mapping.proxy_relaxation)
            if mapping.aitken_has_previous is not None:
                mapping.aitken_has_previous.zero_()
            if mapping.proxy_qd_before is not None:
                mapping.proxy_qd_before.zero_()
        for config in self._proxy_collision_configs.values():
            config.collide_counter = 0
            if config.contacts is not None:
                config.contacts.clear(bump_generation=True)

    def _stash_proxy_feedback(self, proxy: _ProxyEntityMapping) -> None:
        if proxy.coupling_forces_previous is None:
            return
        wp.launch(
            stash_proxy_forces_kernel,
            dim=proxy.proxy_ids_global.shape[0],
            inputs=[
                proxy.proxy_ids_global,
                proxy.coupling_forces,
                proxy.coupling_forces_previous,
            ],
            device=self.model.device,
        )

    def _blend_proxy_feedback(self, proxy: _ProxyEntityMapping) -> None:
        """Blend raw feedback using the shared fixed or Aitken relaxation path."""
        if proxy.coupling_forces_previous is None:
            return
        if int(proxy.proxy_relaxation_mode) == int(_ProxyRelaxationMode.AITKEN):
            proxy.aitken_stats.zero_()
            wp.launch(
                _accumulate_aitken_stats_kernel,
                dim=proxy.proxy_ids_global.shape[0],
                inputs=[
                    proxy.proxy_ids_global,
                    proxy.coupling_forces_previous,
                    proxy.coupling_forces,
                    proxy.aitken_residual_previous,
                    proxy.aitken_has_previous,
                    proxy.aitken_stats,
                ],
                device=self.model.device,
            )
            wp.launch(
                _update_aitken_relaxation_kernel,
                dim=1,
                inputs=[
                    float(proxy.proxy_relaxation_min),
                    float(proxy.proxy_relaxation_max),
                    proxy.aitken_stats,
                    proxy.aitken_relaxation,
                    proxy.aitken_has_previous,
                ],
                device=self.model.device,
            )
            wp.launch(
                _blend_aitken_forces_kernel,
                dim=proxy.proxy_ids_global.shape[0],
                inputs=[
                    proxy.proxy_ids_global,
                    proxy.coupling_forces_previous,
                    proxy.aitken_residual_previous,
                    proxy.aitken_relaxation,
                    proxy.coupling_forces,
                ],
                device=self.model.device,
            )
            return
        wp.launch(
            blend_proxy_forces_kernel,
            dim=proxy.proxy_ids_global.shape[0],
            inputs=[
                float(proxy.proxy_relaxation),
                proxy.proxy_ids_global,
                proxy.coupling_forces_previous,
                proxy.coupling_forces,
            ],
            device=self.model.device,
        )

    def _sync_proxy_joint_targets(
        self,
        joint_proxies: Sequence[_ProxyJointMapping],
        source_control: Control | None,
        dst_control: Control | None,
    ) -> None:
        if source_control is None or dst_control is None:
            return

        for proxy in joint_proxies:
            if (
                proxy.source_target_q_indices_global is not None
                and proxy.destination_target_q_indices_local is not None
                and proxy.source_target_q_indices_global.shape[0] > 0
                and source_control.joint_target_q is not None
                and dst_control.joint_target_q is not None
            ):
                wp.launch(
                    _copy_indexed_float_kernel,
                    dim=proxy.source_target_q_indices_global.shape[0],
                    inputs=[
                        proxy.source_target_q_indices_global,
                        proxy.destination_target_q_indices_local,
                        source_control.joint_target_q,
                        dst_control.joint_target_q,
                    ],
                    device=self.model.device,
                )

            if (
                proxy.source_target_qd_indices_global is not None
                and proxy.destination_target_qd_indices_local is not None
                and proxy.source_target_qd_indices_global.shape[0] > 0
                and source_control.joint_target_qd is not None
                and dst_control.joint_target_qd is not None
            ):
                wp.launch(
                    _copy_indexed_float_kernel,
                    dim=proxy.source_target_qd_indices_global.shape[0],
                    inputs=[
                        proxy.source_target_qd_indices_global,
                        proxy.destination_target_qd_indices_local,
                        source_control.joint_target_qd,
                        dst_control.joint_target_qd,
                    ],
                    device=self.model.device,
                )

    def _entry_has_body_proxy_overrides(self, name: str) -> bool:
        for proxy in self._proxy_mappings:
            if proxy.dst_name == name and proxy.proxy_ids_local is not None and proxy.proxy_ids_local.shape[0] > 0:
                return True
        return False

    def _refresh_body_inertial_view_overrides(self, entry) -> None:
        if not self._entry_has_body_proxy_overrides(entry.name):
            super()._refresh_body_inertial_view_overrides(entry)
            return

        entry.view._refresh_body_inertial_properties(entry.body_local_to_global)
        if entry.body_dynamics_disabled_indices.shape[0] > 0:
            entry.view.disable_body_dynamics(
                self._body_indices_to_local_array(entry, entry.body_dynamics_disabled_indices)
            )

    def _apply_proxy_effective_masses(self) -> None:
        """Install virtual proxy masses from source solver effective masses."""
        self._apply_proxy_body_effective_masses()
        self._apply_proxy_particle_effective_masses()

    def _apply_proxy_body_effective_masses(self) -> None:
        """Install virtual proxy body inertia from source solver effective masses."""
        device = self.model.device

        for proxy in self._proxy_mappings:
            if proxy.src_ids is None or proxy.src_ids.shape[0] == 0:
                continue
            src = self._entries[proxy.src_name]
            dst = self._entries[proxy.dst_name]
            inertial_properties = self._eval_effective_body_inertial_properties(src, proxy.src_ids)
            if inertial_properties is None:
                continue
            masses, inertias = inertial_properties
            proxy_masses = wp.array(
                [float(proxy.mass_scale) * mass for mass in masses],
                dtype=float,
                device=device,
            )
            proxy_inertias = wp.array(
                [wp.mat33(np.asarray(inertia, dtype=np.float32) * float(proxy.mass_scale)) for inertia in inertias],
                dtype=wp.mat33,
                device=device,
            )
            self._apply_body_inertia_override(dst, proxy.proxy_ids_local, proxy_masses, proxy_inertias)

    def _apply_proxy_particle_effective_masses(self) -> None:
        """Install virtual proxy particle masses from source solver effective masses."""
        device = self.model.device

        for proxy in self._proxy_particle_mappings:
            if proxy.src_ids is None or proxy.src_ids.shape[0] == 0:
                continue
            src = self._entries[proxy.src_name]
            dst = self._entries[proxy.dst_name]
            masses = self._eval_effective_masses(
                src,
                CouplingEndpointKind.PARTICLE,
                proxy.src_ids,
            )
            if masses is None:
                continue
            proxy_masses = wp.array(
                [float(proxy.mass_scale) * mass for mass in masses],
                dtype=float,
                device=device,
            )
            self._apply_particle_mass_override(dst, proxy.proxy_ids_local, proxy_masses)

    def notify_model_changed(self, flags: int) -> None:
        """Refresh proxy inertia after source solvers consume model updates."""
        super().notify_model_changed(flags)
        if int(flags) & int(ModelFlags.BODY_INERTIAL_PROPERTIES):
            self._apply_proxy_body_effective_masses()

    def _step_coupled(
        self,
        state_in: State,
        state_out: State,
        control: Control | None,
        contacts: Contacts | None,
        dt: float,
    ) -> None:
        """Run lagged-impulse proxy iterations for one coupled step."""
        del state_out
        self._reset_aitken_iteration_state()
        iterations = int(self._coupling.iterations)
        for k in range(iterations):
            # Some solvers use state_in arrays as temporary buffers during a
            # step. Proxy iterations are repeated solves over the same top-level
            # interval, so relaxation restarts copy the original distributed
            # input state and only carry harvested feedback buffers forward.
            if k > 0:
                self._distribute_state(state_in, dt=dt, iteration_restart=True)
            self._step_proxy(state_in, control, contacts, dt, iteration_restart=k > 0)

    def _reset_aitken_iteration_state(self) -> None:
        for proxy in [*self._proxy_mappings, *self._proxy_particle_mappings]:
            if int(proxy.proxy_relaxation_mode) != int(_ProxyRelaxationMode.AITKEN):
                continue
            wp.launch(
                _reset_aitken_state_kernel,
                dim=1,
                inputs=[
                    float(proxy.proxy_relaxation),
                    proxy.aitken_relaxation,
                    proxy.aitken_has_previous,
                ],
                device=self.model.device,
            )

    def _build_proxy_groups(self) -> dict[tuple[str, str], dict[str, list]]:
        """Bucket proxy mappings by (src, dst) once at construction."""
        groups: dict[tuple[str, str], dict[str, list]] = {}
        for proxy in self._proxy_mappings:
            groups.setdefault((proxy.src_name, proxy.dst_name), {"bodies": [], "particles": [], "joints": []})[
                "bodies"
            ].append(proxy)
        for proxy in self._proxy_particle_mappings:
            groups.setdefault((proxy.src_name, proxy.dst_name), {"bodies": [], "particles": [], "joints": []})[
                "particles"
            ].append(proxy)
        for proxy in self._proxy_joint_mappings:
            groups.setdefault((proxy.src_name, proxy.dst_name), {"bodies": [], "particles": [], "joints": []})[
                "joints"
            ].append(proxy)
        return groups

    def _step_proxy(
        self,
        state_in: State,
        control: Control | None,
        contacts: Contacts | None,
        dt: float,
        iteration_restart: bool = False,
    ) -> None:
        """Run one lagged-impulse proxy coupling pass."""
        for (src_name, dst_name), group in self._proxy_groups.items():
            body_proxies = group["bodies"]
            particle_proxies = group["particles"]
            joint_proxies = group["joints"]
            src = self._entries[src_name]
            dst = self._entries[dst_name]

            for proxy in body_proxies:
                self._stash_proxy_feedback(proxy)
            for proxy in particle_proxies:
                self._stash_proxy_feedback(proxy)

            if src.has_body_force_input and (src.body_indices.shape[0] > 0 or body_proxies):
                self._clear_body_force_input(src)
                self._add_body_force_input(src, src.body_local_to_global, state_in.body_f)
                for proxy in body_proxies:
                    self._add_body_force_input(
                        src,
                        proxy.source_local_to_proxy_global,
                        proxy.coupling_forces,
                    )
                self._notify_input_state_update(src, StateFlags.BODY_F, dt=dt)

            if src.has_particle_force_input and (src.particle_indices.shape[0] > 0 or particle_proxies):
                self._clear_particle_force_input(src)
                self._add_particle_force_input(src, src.particle_local_to_global, state_in.particle_f)
                for proxy in particle_proxies:
                    self._add_particle_force_input(
                        src,
                        proxy.source_local_to_proxy_global,
                        proxy.coupling_forces,
                    )
                self._notify_input_state_update(src, StateFlags.PARTICLE_F, dt=dt)

            self._step_entry(src, control, contacts, dt)

            for proxy in body_proxies:
                is_staggered = int(proxy.mode) == int(_ProxyMode.STAGGERED)
                sync_body_q = src.state_1.body_q if is_staggered else src.state_0.body_q

                wp.launch(
                    sync_proxy_states_kernel,
                    dim=proxy.source_local_to_proxy_local.shape[0],
                    inputs=[
                        sync_body_q,
                        src.state_1.body_qd,
                        proxy.source_local_to_proxy_local,
                        dst.state_0.body_q,
                        dst.state_0.body_qd,
                    ],
                    device=self.model.device,
                )

                self._notify_input_state_update(
                    dst,
                    StateFlags.BODY_Q | StateFlags.BODY_QD,
                    dt=dt,
                )

                wp.copy(proxy.proxy_qd_before, dst.state_0.body_qd)

                if is_staggered:
                    proxy.coupling_forces.zero_()

                dst.solver.coupling_rewind_proxy_body(
                    proxy.destination_local_to_proxy_global,
                    dst.state_0,
                    proxy.coupling_forces,
                    dst.body_gravity_acceleration,
                    dt,
                )
                self._notify_input_state_update(dst, StateFlags.BODY_QD | StateFlags.BODY_F, dt=dt)

            for proxy in particle_proxies:
                is_staggered = int(proxy.mode) == int(_ProxyMode.STAGGERED)
                sync_particle_q = src.state_1.particle_q if is_staggered else src.state_0.particle_q

                wp.launch(
                    sync_proxy_particles_kernel,
                    dim=proxy.source_local_to_proxy_local.shape[0],
                    inputs=[
                        sync_particle_q,
                        src.state_1.particle_qd,
                        proxy.source_local_to_proxy_local,
                        dst.state_0.particle_q,
                        dst.state_0.particle_qd,
                    ],
                    device=self.model.device,
                )

                self._notify_input_state_update(
                    dst,
                    StateFlags.PARTICLE_Q | StateFlags.PARTICLE_QD,
                    dt=dt,
                )

                wp.copy(proxy.proxy_qd_before, dst.state_0.particle_qd)

                if is_staggered:
                    continue

                dst.solver.coupling_rewind_proxy_particle(
                    proxy.destination_local_to_proxy_global,
                    dst.state_0,
                    proxy.coupling_forces,
                    dst.particle_gravity_acceleration,
                    dt,
                )
                self._notify_input_state_update(dst, StateFlags.PARTICLE_QD, dt=dt)

            dst_contacts = contacts
            # Without a proxy-local collision pipeline, the caller-provided
            # contact set is the fresh outer-step result. Inner proxy
            # iterations reuse it.
            contacts_freshly_detected = not iteration_restart
            filter_dst_contacts = True
            collision_config = self._proxy_collision_configs.get((src_name, dst_name))
            if collision_config is not None:
                dst_contacts, contacts_freshly_detected = self._proxy_collision_contacts(
                    collision_config, dst.state_0, iteration_restart=iteration_restart
                )
                filter_dst_contacts = False

            restore_external_contacts = None
            dst_contacts_used = dst_contacts

            if body_proxies:
                contacts_before_prepare = dst_contacts
                dst_contacts = dst.solver.coupling_prepare_proxy_contacts(
                    dst.state_0,
                    dst_contacts,
                    contacts_freshly_detected=contacts_freshly_detected,
                )
                if (
                    collision_config is None
                    and contacts_before_prepare is contacts
                    and contacts_before_prepare is not None
                    and contacts_before_prepare.rigid_contact_count is not None
                ):
                    restore_external_contacts = contacts_before_prepare

            control_callback = None
            if joint_proxies:

                def control_callback(dst_control, joint_proxies=joint_proxies, source_control=control):
                    self._sync_proxy_joint_targets(joint_proxies, source_control, dst_control)

            try:
                dst_contacts_used = self._step_entry(
                    dst,
                    control,
                    dst_contacts,
                    dt,
                    filter_contacts=filter_dst_contacts,
                    control_callback=control_callback,
                )
            finally:
                if restore_external_contacts is not None:
                    wp.launch(
                        restore_filtered_proxy_rigid_contacts_kernel,
                        dim=restore_external_contacts.rigid_contact_shape0.shape[0],
                        inputs=[
                            restore_external_contacts.rigid_contact_count,
                            restore_external_contacts.rigid_contact_shape0,
                            restore_external_contacts.rigid_contact_shape1,
                        ],
                        device=self.model.device,
                    )

            for proxy in body_proxies:
                dst.solver.coupling_harvest_proxy_wrenches(
                    proxy.destination_local_to_proxy_global,
                    proxy.coupling_forces,
                    body_qd_before=proxy.proxy_qd_before,
                    state=dst.state_0,
                    state_out=dst.state_1,
                    contacts=dst_contacts_used,
                    dt=dt,
                )
                self._blend_proxy_feedback(proxy)

            for proxy in particle_proxies:
                dst.solver.coupling_harvest_proxy_particle_forces(
                    proxy.destination_local_to_proxy_global,
                    proxy.coupling_forces,
                    particle_qd_before=proxy.proxy_qd_before,
                    state=dst.state_0,
                    state_out=dst.state_1,
                    contacts=dst_contacts_used,
                    dt=dt,
                )
                self._blend_proxy_feedback(proxy)
