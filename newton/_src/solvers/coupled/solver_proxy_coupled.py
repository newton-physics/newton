# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Lagged-impulse proxy coupled multi-solver simulations."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from enum import IntEnum
from typing import TYPE_CHECKING

import numpy as np
import warp as wp

from ...sim import BodyFlags
from ..coupling import (
    CouplingEndpointKind,
    CouplingHook,
    CouplingInputStateFlags,
)
from .proxy_utils import (
    subtract_proxy_forces_kernel,
    subtract_proxy_particle_forces_kernel,
    sync_proxy_particles_kernel,
    sync_proxy_states_kernel,
)
from .solver_coupled import SolverCoupled, _coupling_method_or_fallback

if TYPE_CHECKING:
    from ...sim import Contacts, Control, Model, State
    from .model_view import ModelView


@dataclass
class _ProxyBodyMapping:
    """Runtime mapping from source bodies to destination proxy bodies.

    ``coupling_forces`` stores feedback at global proxy body ids. Dense
    local-to-global maps route those values either to destination proxy-local
    ids for rewind/harvest or back to source-local ids for source feedback.
    """

    src_name: str
    dst_name: str
    src_body_ids: wp.array = field(default=None)
    proxy_body_ids_local: wp.array = field(default=None)
    source_local_to_proxy_local: wp.array = field(default=None)
    source_local_to_proxy_global: wp.array = field(default=None)
    destination_local_to_proxy_global: wp.array = field(default=None)
    coupling_forces: wp.array = field(default=None)
    proxy_qd_before: wp.array = field(default=None)
    mass_scale: float = 1.0
    mode: int = 0


@dataclass
class _ProxyParticleMapping:
    """Runtime mapping from source particles to destination proxy particles.

    ``coupling_forces`` stores feedback at global proxy particle ids. Dense
    local-to-global maps route those values either to destination proxy-local
    ids for rewind/harvest or back to source-local ids for source feedback.
    """

    src_name: str
    dst_name: str
    src_particle_ids: wp.array = field(default=None)
    proxy_particle_ids_local: wp.array = field(default=None)
    source_local_to_proxy_local: wp.array = field(default=None)
    source_local_to_proxy_global: wp.array = field(default=None)
    destination_local_to_proxy_global: wp.array = field(default=None)
    coupling_forces: wp.array = field(default=None)
    proxy_qd_before: wp.array = field(default=None)
    mass_scale: float = 1.0
    mode: int = 0


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


_PROXY_MODE_BY_NAME = {"lagged": _ProxyMode.LAGGED, "staggered": _ProxyMode.STAGGERED}


class SolverProxyCoupled(SolverCoupled):
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
            mass_scale: Destination proxy body mass/inertia and particle mass
                scale factor.
            mode: Proxy transfer mode, ``"lagged"`` or ``"staggered"``.
                ``"lagged"`` syncs source begin poses and end velocities, then
                prepares proxies to avoid double-counting lagged feedback.
                ``"staggered"`` syncs source end poses and end velocities
                directly.
            particles: Source particle ids to map into destination proxies.
            proxy_particles: Optional destination particle ids. Defaults to
                ``particles``.
            collision_pipeline: Optional factory called as
                ``collision_pipeline(destination_model_view)``. When supplied,
                ``SolverProxyCoupled`` uses the returned pipeline to detect
                destination proxy contacts before each destination solve. If
                the factory returns ``None``, the destination solve receives
                the outer-level contacts passed to :meth:`step`.
            collide_interval: Collision-detection refresh interval for
                ``collision_pipeline``. ``None`` means every proxy pass when a
                custom pipeline is supplied.
        """

        source: str
        destination: str
        bodies: Sequence[int] = ()
        proxy_bodies: Sequence[int] | None = None
        mass_scale: float = 1.0
        mode: str = "lagged"
        particles: Sequence[int] = ()
        proxy_particles: Sequence[int] | None = None
        collision_pipeline: Callable[[ModelView], object | None] | None = None
        collide_interval: int | None = None

    @dataclass(frozen=True)
    class Config:
        """Lagged-impulse proxy coupling configuration."""

        proxies: Sequence[SolverProxyCoupled.Proxy]
        iterations: int = 1

    def __init__(
        self,
        model: Model,
        entries: Sequence[SolverCoupled.Entry],
        coupling: SolverProxyCoupled.Config,
    ) -> None:
        if len(entries) > 2:
            raise ValueError("Proxy coupling currently supports at most two solver entries")

        self._proxy_coupling = coupling
        self._proxy_mappings = self._build_proxy_mappings(model, coupling)
        self._proxy_particle_mappings = self._build_proxy_particle_mappings(model, coupling)
        self._proxy_collision_configs = self._build_proxy_collision_configs(coupling)

        super().__init__(
            model=model,
            entries=entries,
            coupling=coupling,
        )

    @staticmethod
    def _proxy_mode_value(mode: str) -> int:
        try:
            return int(_PROXY_MODE_BY_NAME[mode.lower()])
        except (AttributeError, KeyError) as err:
            raise ValueError(f"Unknown proxy coupling mode {mode!r}; expected 'lagged' or 'staggered'") from err

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

    def _build_proxy_mappings(
        self,
        model: Model,
        coupling: SolverProxyCoupled.Config,
    ) -> list[_ProxyBodyMapping]:
        mappings = []
        device = model.device
        for proxy in coupling.proxies:
            src_ids = [int(i) for i in proxy.bodies]
            if not src_ids:
                continue
            proxy_local_ids = [int(i) for i in (proxy.proxy_bodies if proxy.proxy_bodies is not None else proxy.bodies)]
            if len(src_ids) != len(proxy_local_ids):
                raise ValueError("Proxy source bodies and proxy_bodies must have the same length")
            self._validate_proxy_ids("Proxy source body", src_ids, model.body_count)
            self._validate_proxy_ids("Proxy destination body", proxy_local_ids, model.body_count)
            self._validate_unique_proxy_ids("source body", src_ids)
            self._validate_unique_proxy_ids("proxy body", proxy_local_ids)
            # ModelView currently preserves parent-model indexing, so the
            # configured proxy ids are both local ids in the destination view
            # and global ids in the parent model. Keep both names explicit so
            # hook indexing remains unambiguous if views become compact later.
            proxy_global_ids = proxy_local_ids

            source_local_to_proxy_local = [-1] * model.body_count
            source_local_to_proxy_global = [-1] * model.body_count
            destination_local_to_proxy_global = [-1] * model.body_count
            for source_local, proxy_local, proxy_global in zip(src_ids, proxy_local_ids, proxy_global_ids, strict=True):
                source_local_to_proxy_local[source_local] = proxy_local
                source_local_to_proxy_global[source_local] = proxy_global
                destination_local_to_proxy_global[proxy_local] = proxy_global

            mappings.append(
                _ProxyBodyMapping(
                    src_name=proxy.source,
                    dst_name=proxy.destination,
                    src_body_ids=wp.array(src_ids, dtype=int, device=device),
                    proxy_body_ids_local=wp.array(proxy_local_ids, dtype=int, device=device),
                    source_local_to_proxy_local=wp.array(source_local_to_proxy_local, dtype=int, device=device),
                    source_local_to_proxy_global=wp.array(source_local_to_proxy_global, dtype=int, device=device),
                    destination_local_to_proxy_global=wp.array(
                        destination_local_to_proxy_global, dtype=int, device=device
                    ),
                    mass_scale=float(proxy.mass_scale),
                    mode=self._proxy_mode_value(proxy.mode),
                )
            )
        return mappings

    def _build_proxy_collision_configs(
        self,
        coupling: SolverProxyCoupled.Config,
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
            collide_interval = 1 if proxy.collide_interval is None else max(1, int(proxy.collide_interval))
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
        coupling: SolverProxyCoupled.Config,
    ) -> list[_ProxyParticleMapping]:
        mappings = []
        device = model.device
        for proxy in coupling.proxies:
            src_ids = [int(i) for i in proxy.particles]
            if not src_ids:
                continue
            proxy_local_ids = [
                int(i) for i in (proxy.proxy_particles if proxy.proxy_particles is not None else proxy.particles)
            ]
            if len(src_ids) != len(proxy_local_ids):
                raise ValueError("Proxy source particles and proxy_particles must have the same length")
            self._validate_proxy_ids("Proxy source particle", src_ids, model.particle_count)
            self._validate_proxy_ids("Proxy destination particle", proxy_local_ids, model.particle_count)
            self._validate_unique_proxy_ids("source particle", src_ids)
            self._validate_unique_proxy_ids("proxy particle", proxy_local_ids)
            # ModelView currently preserves parent-model indexing; see the
            # body-proxy construction above for the local/global convention.
            proxy_global_ids = proxy_local_ids

            source_local_to_proxy_local = [-1] * model.particle_count
            source_local_to_proxy_global = [-1] * model.particle_count
            destination_local_to_proxy_global = [-1] * model.particle_count
            for source_local, proxy_local, proxy_global in zip(src_ids, proxy_local_ids, proxy_global_ids, strict=True):
                source_local_to_proxy_local[source_local] = proxy_local
                source_local_to_proxy_global[source_local] = proxy_global
                destination_local_to_proxy_global[proxy_local] = proxy_global

            mappings.append(
                _ProxyParticleMapping(
                    src_name=proxy.source,
                    dst_name=proxy.destination,
                    src_particle_ids=wp.array(src_ids, dtype=int, device=device),
                    proxy_particle_ids_local=wp.array(proxy_local_ids, dtype=int, device=device),
                    source_local_to_proxy_local=wp.array(source_local_to_proxy_local, dtype=int, device=device),
                    source_local_to_proxy_global=wp.array(source_local_to_proxy_global, dtype=int, device=device),
                    destination_local_to_proxy_global=wp.array(
                        destination_local_to_proxy_global, dtype=int, device=device
                    ),
                    mass_scale=float(proxy.mass_scale),
                    mode=self._proxy_mode_value(proxy.mode),
                )
            )
        return mappings

    def _entry_proxy_body_keep_indices(self, name: str) -> set[int]:
        proxy_keep: set[int] = set()
        for mapping in self._proxy_mappings:
            if mapping.dst_name == name and mapping.proxy_body_ids_local is not None:
                proxy_keep.update(int(i) for i in mapping.proxy_body_ids_local.numpy())
        return proxy_keep

    def _entry_proxy_particle_keep_indices(self, name: str) -> set[int]:
        proxy_keep: set[int] = set()
        for mapping in self._proxy_particle_mappings:
            if mapping.dst_name == name and mapping.proxy_particle_ids_local is not None:
                proxy_keep.update(int(i) for i in mapping.proxy_particle_ids_local.numpy())
        return proxy_keep

    def _after_entries_constructed(self) -> None:
        self._refresh_proxy_view_maps()
        self._validate_in_place_proxy_entries()
        self._apply_proxy_effective_masses()
        self._init_proxy_collision_pipelines()

    def _refresh_proxy_view_maps(self) -> None:
        """Resize dense proxy maps to the source/destination view counts."""
        device = self.model.device
        for mapping in self._proxy_mappings:
            src_count = int(self._entries[mapping.src_name].view.body_count)
            dst_count = int(self._entries[mapping.dst_name].view.body_count)
            mapping.source_local_to_proxy_local = wp.array(
                mapping.source_local_to_proxy_local.numpy()[:src_count],
                dtype=int,
                device=device,
            )
            mapping.source_local_to_proxy_global = wp.array(
                mapping.source_local_to_proxy_global.numpy()[:src_count],
                dtype=int,
                device=device,
            )
            mapping.destination_local_to_proxy_global = wp.array(
                mapping.destination_local_to_proxy_global.numpy()[:dst_count],
                dtype=int,
                device=device,
            )

        for mapping in self._proxy_particle_mappings:
            src_count = int(self._entries[mapping.src_name].view.particle_count)
            dst_count = int(self._entries[mapping.dst_name].view.particle_count)
            mapping.source_local_to_proxy_local = wp.array(
                mapping.source_local_to_proxy_local.numpy()[:src_count],
                dtype=int,
                device=device,
            )
            mapping.source_local_to_proxy_global = wp.array(
                mapping.source_local_to_proxy_global.numpy()[:src_count],
                dtype=int,
                device=device,
            )
            mapping.destination_local_to_proxy_global = wp.array(
                mapping.destination_local_to_proxy_global.numpy()[:dst_count],
                dtype=int,
                device=device,
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
    ) -> tuple[Contacts, bool]:
        if config.pipeline is None or config.contacts is None:
            raise RuntimeError("Proxy collision pipeline was not initialized")

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
        model = self.model
        device = model.device
        for mapping in self._proxy_mappings:
            mapping.coupling_forces = wp.zeros(model.body_count, dtype=wp.spatial_vector, device=device)
            mapping.proxy_qd_before = wp.zeros(model.body_count, dtype=wp.spatial_vector, device=device)
        for mapping in self._proxy_particle_mappings:
            mapping.coupling_forces = wp.zeros(model.particle_count, dtype=wp.vec3, device=device)
            mapping.proxy_qd_before = wp.zeros(model.particle_count, dtype=wp.vec3, device=device)

    def _apply_proxy_effective_masses(self) -> None:
        """Install virtual proxy masses from source solver effective masses."""
        device = self.model.device

        for proxy in self._proxy_mappings:
            if proxy.src_body_ids is None or proxy.src_body_ids.shape[0] == 0:
                continue
            src = self._entries[proxy.src_name]
            dst = self._entries[proxy.dst_name]
            inertial_properties = self._eval_effective_body_inertial_properties(src, proxy.src_body_ids)
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
            self._apply_body_inertia_override(dst, proxy.proxy_body_ids_local, proxy_masses, proxy_inertias)

        for proxy in self._proxy_particle_mappings:
            if proxy.src_particle_ids is None or proxy.src_particle_ids.shape[0] == 0:
                continue
            src = self._entries[proxy.src_name]
            dst = self._entries[proxy.dst_name]
            masses = self._eval_effective_masses(
                src,
                CouplingEndpointKind.PARTICLE,
                proxy.src_particle_ids,
            )
            if masses is None:
                continue
            proxy_masses = wp.array(
                [float(proxy.mass_scale) * mass for mass in masses],
                dtype=float,
                device=device,
            )
            self._apply_particle_mass_override(dst, proxy.proxy_particle_ids_local, proxy_masses)

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
        iterations = max(1, int(self._proxy_coupling.iterations))
        for k in range(iterations):
            # Some solvers use state_in arrays as temporary buffers during a
            # step. Proxy iterations are repeated solves over the same top-level
            # interval, so each relaxation pass restarts from the original
            # distributed input state and only carries harvested feedback
            # buffers forward.
            self._distribute_state(state_in, dt=dt, restart=k > 0)
            self._step_proxy(state_in, control, contacts, dt)

    def _step_proxy(
        self,
        state_in: State,
        control: Control | None,
        contacts: Contacts | None,
        dt: float,
    ) -> None:
        """Run one lagged-impulse proxy coupling pass."""
        groups: dict[tuple[str, str], dict[str, list]] = {}

        def group_for(src_name: str, dst_name: str) -> dict[str, list]:
            key = (src_name, dst_name)
            if key not in groups:
                groups[key] = {"bodies": [], "particles": []}
            return groups[key]

        for proxy in self._proxy_mappings:
            group_for(proxy.src_name, proxy.dst_name)["bodies"].append(proxy)
        for proxy in self._proxy_particle_mappings:
            group_for(proxy.src_name, proxy.dst_name)["particles"].append(proxy)

        for (src_name, dst_name), group in groups.items():
            body_proxies = group["bodies"]
            particle_proxies = group["particles"]
            src = self._entries[src_name]
            dst = self._entries[dst_name]

            if src.body_force_input is not None and (src.body_indices.shape[0] > 0 or body_proxies):
                self._clear_body_force_input(src)
                self._add_body_force_input(src, src.body_local_to_global, state_in.body_f)
                for proxy in body_proxies:
                    self._add_body_force_input(
                        src,
                        proxy.source_local_to_proxy_global,
                        proxy.coupling_forces,
                    )
                self._notify_input_state_update(src, CouplingInputStateFlags.BODY_F, dt=dt)

            if src.particle_force_input is not None and (src.particle_indices.shape[0] > 0 or particle_proxies):
                self._clear_particle_force_input(src)
                self._add_particle_force_input(src, src.particle_local_to_global, state_in.particle_f)
                for proxy in particle_proxies:
                    self._add_particle_force_input(
                        src,
                        proxy.source_local_to_proxy_global,
                        proxy.coupling_forces,
                    )
                self._notify_input_state_update(src, CouplingInputStateFlags.PARTICLE_F, dt=dt)

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
                    CouplingInputStateFlags.BODY_Q | CouplingInputStateFlags.BODY_QD,
                    dt=dt,
                )

                body_qd_rewound = False
                if not is_staggered:
                    rewind_method = _coupling_method_or_fallback(dst.solver, CouplingHook.BODY_PROXY_REWIND_VELOCITY)
                else:
                    rewind_method = None
                if rewind_method is not None and not is_staggered:
                    rewind_method(
                        proxy.destination_local_to_proxy_global,
                        dst.state_0,
                        proxy.coupling_forces,
                        dt,
                    )
                    body_qd_rewound = True
                elif not is_staggered:
                    wp.launch(
                        subtract_proxy_forces_kernel,
                        dim=proxy.destination_local_to_proxy_global.shape[0],
                        inputs=[
                            dt,
                            self.model.gravity,
                            self.model.body_world,
                            dst.state_0.body_q,
                            dst.state_0.body_f,
                            proxy.coupling_forces,
                            proxy.destination_local_to_proxy_global,
                            dst.view.body_inv_mass,
                            dst.view.body_inv_inertia,
                            dst.state_0.body_qd,
                        ],
                        device=self.model.device,
                    )
                    body_qd_rewound = True
                if body_qd_rewound:
                    self._notify_input_state_update(dst, CouplingInputStateFlags.BODY_QD, dt=dt)

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
                    CouplingInputStateFlags.PARTICLE_Q | CouplingInputStateFlags.PARTICLE_QD,
                    dt=dt,
                )

                particle_qd_rewound = False
                if not is_staggered:
                    rewind_method = _coupling_method_or_fallback(
                        dst.solver, CouplingHook.PARTICLE_PROXY_REWIND_VELOCITY
                    )
                else:
                    rewind_method = None
                if rewind_method is not None and not is_staggered:
                    rewind_method(
                        proxy.destination_local_to_proxy_global,
                        dst.state_0,
                        proxy.coupling_forces,
                        dt,
                    )
                    particle_qd_rewound = True
                elif not is_staggered:
                    wp.launch(
                        subtract_proxy_particle_forces_kernel,
                        dim=proxy.destination_local_to_proxy_global.shape[0],
                        inputs=[
                            dt,
                            self.model.gravity,
                            self.model.particle_world,
                            dst.state_0.particle_f,
                            proxy.coupling_forces,
                            proxy.destination_local_to_proxy_global,
                            dst.view.particle_inv_mass,
                            dst.state_0.particle_qd,
                        ],
                        device=self.model.device,
                    )
                    particle_qd_rewound = True
                if particle_qd_rewound:
                    self._notify_input_state_update(dst, CouplingInputStateFlags.PARTICLE_QD, dt=dt)

            dst_contacts = contacts
            contacts_freshly_detected = False
            collision_config = self._proxy_collision_configs.get((src_name, dst_name))
            if collision_config is not None:
                dst_contacts, contacts_freshly_detected = self._proxy_collision_contacts(collision_config, dst.state_0)

            restore_filtered_contacts = False
            try:
                if body_proxies:
                    if self._uses_custom_coupling_hook(dst.solver, CouplingHook.PROXY_CONTACT_PREPARE):
                        dst_contacts = dst.solver.coupling_prepare_proxy_contacts(
                            dst.state_0,
                            dst_contacts,
                            contacts_freshly_detected=contacts_freshly_detected,
                        )

                    if dst_contacts is contacts and contacts is not None and contacts.rigid_contact_count is not None:
                        wp.launch(
                            _filter_proxy_rigid_contacts_kernel,
                            dim=contacts.rigid_contact_shape0.shape[0],
                            inputs=[
                                contacts.rigid_contact_count,
                                contacts.rigid_contact_shape0,
                                contacts.rigid_contact_shape1,
                                self.model.shape_body,
                                dst.view.body_flags,
                                dst.view.body_inv_mass,
                                int(BodyFlags.PROXY),
                            ],
                            device=self.model.device,
                        )
                        restore_filtered_contacts = True

                for proxy in body_proxies:
                    wp.copy(proxy.proxy_qd_before, dst.state_0.body_qd)
                for proxy in particle_proxies:
                    wp.copy(proxy.proxy_qd_before, dst.state_0.particle_qd)

                self._step_entry(dst, control, dst_contacts, dt)
            finally:
                if restore_filtered_contacts:
                    wp.launch(
                        _restore_filtered_proxy_rigid_contacts_kernel,
                        dim=contacts.rigid_contact_shape0.shape[0],
                        inputs=[
                            contacts.rigid_contact_count,
                            contacts.rigid_contact_shape0,
                            contacts.rigid_contact_shape1,
                        ],
                        device=self.model.device,
                    )

            for proxy in body_proxies:
                proxy.coupling_forces.zero_()
                if self._uses_custom_coupling_hook(dst.solver, CouplingHook.BODY_PROXY_HARVEST):
                    dst.solver.coupling_harvest_proxy_wrenches(
                        proxy.destination_local_to_proxy_global,
                        proxy.coupling_forces,
                        state=dst.state_0,
                        state_out=dst.state_1,
                        contacts=dst_contacts,
                        dt=dt,
                    )
                else:
                    wp.launch(
                        _harvest_proxy_momentum_forces_kernel,
                        dim=proxy.destination_local_to_proxy_global.shape[0],
                        inputs=[
                            dt,
                            proxy.destination_local_to_proxy_global,
                            proxy.proxy_qd_before,
                            dst.state_1.body_qd,
                            dst.view.body_mass,
                            dst.view.body_inertia,
                            dst.state_1.body_q,
                            self.model.gravity,
                            self.model.body_world,
                            proxy.coupling_forces,
                        ],
                        device=self.model.device,
                    )

            for proxy in particle_proxies:
                proxy.coupling_forces.zero_()
                if self._uses_custom_coupling_hook(dst.solver, CouplingHook.PARTICLE_PROXY_HARVEST):
                    dst.solver.coupling_harvest_proxy_particle_forces(
                        proxy.destination_local_to_proxy_global,
                        proxy.coupling_forces,
                        state=dst.state_0,
                        state_out=dst.state_1,
                        contacts=dst_contacts,
                        dt=dt,
                    )
                else:
                    wp.launch(
                        _harvest_proxy_particle_momentum_forces_kernel,
                        dim=proxy.destination_local_to_proxy_global.shape[0],
                        inputs=[
                            dt,
                            proxy.destination_local_to_proxy_global,
                            proxy.proxy_qd_before,
                            dst.state_1.particle_qd,
                            dst.view.particle_mass,
                            self.model.gravity,
                            self.model.particle_world,
                            proxy.coupling_forces,
                        ],
                        device=self.model.device,
                    )


@wp.kernel(enable_backward=False)
def _filter_proxy_rigid_contacts_kernel(
    rigid_contact_count: wp.array[int],
    rigid_contact_shape0: wp.array[wp.int32],
    rigid_contact_shape1: wp.array[wp.int32],
    shape_body: wp.array[wp.int32],
    body_flags: wp.array[wp.int32],
    body_inv_mass: wp.array[float],
    proxy_flag: int,
):
    """Invalidate proxy-vs-static and proxy-vs-proxy rigid contacts."""
    contact_id = wp.tid()
    if contact_id >= rigid_contact_count[0]:
        return

    s0 = rigid_contact_shape0[contact_id]
    s1 = rigid_contact_shape1[contact_id]
    body0 = shape_body[s0] if s0 >= 0 and s0 < shape_body.shape[0] else -1
    body1 = shape_body[s1] if s1 >= 0 and s1 < shape_body.shape[0] else -1

    is_proxy0 = 0
    if body0 >= 0 and body0 < body_flags.shape[0]:
        if (body_flags[body0] & proxy_flag) != 0:
            is_proxy0 = 1
    is_proxy1 = 0
    if body1 >= 0 and body1 < body_flags.shape[0]:
        if (body_flags[body1] & proxy_flag) != 0:
            is_proxy1 = 1

    is_static0 = 0
    if body0 < 0:
        is_static0 = 1
    elif body0 < body_inv_mass.shape[0] and body_inv_mass[body0] == 0.0:
        is_static0 = 1

    is_static1 = 0
    if body1 < 0:
        is_static1 = 1
    elif body1 < body_inv_mass.shape[0] and body_inv_mass[body1] == 0.0:
        is_static1 = 1

    discard = 0
    if is_proxy0 == 1 and is_proxy1 == 1:
        discard = 1
    if is_proxy0 == 1 and is_static1 == 1:
        discard = 1
    if is_proxy1 == 1 and is_static0 == 1:
        discard = 1

    if discard == 1:
        if s0 >= 0:
            rigid_contact_shape0[contact_id] = -s0 - 2
        if s1 >= 0:
            rigid_contact_shape1[contact_id] = -s1 - 2


@wp.kernel(enable_backward=False)
def _restore_filtered_proxy_rigid_contacts_kernel(
    rigid_contact_count: wp.array[int],
    rigid_contact_shape0: wp.array[wp.int32],
    rigid_contact_shape1: wp.array[wp.int32],
):
    """Restore contacts temporarily encoded by proxy contact filtering."""
    contact_id = wp.tid()
    if contact_id >= rigid_contact_count[0]:
        return

    s0 = rigid_contact_shape0[contact_id]
    s1 = rigid_contact_shape1[contact_id]
    if s0 < -1:
        rigid_contact_shape0[contact_id] = -s0 - 2
    if s1 < -1:
        rigid_contact_shape1[contact_id] = -s1 - 2


@wp.kernel(enable_backward=False)
def _harvest_proxy_momentum_forces_kernel(
    dt: float,
    body_local_to_proxy_global: wp.array[int],
    qd_before: wp.array[wp.spatial_vector],
    qd_after: wp.array[wp.spatial_vector],
    body_mass: wp.array[float],
    body_inertia: wp.array[wp.mat33],
    body_q: wp.array[wp.transform],
    gravity: wp.array[wp.vec3],
    body_world: wp.array[wp.int32],
    out_coupling_forces: wp.array[wp.spatial_vector],
):
    """Estimate proxy feedback force from destination velocity change."""
    local_id = wp.tid()
    global_id = body_local_to_proxy_global[local_id]
    if global_id < 0:
        return

    dv = wp.spatial_top(qd_after[local_id]) - wp.spatial_top(qd_before[local_id])
    dw = wp.spatial_bottom(qd_after[local_id]) - wp.spatial_bottom(qd_before[local_id])

    m = body_mass[local_id]
    I_body = body_inertia[local_id]
    r = wp.transform_get_rotation(body_q[local_id])

    world_idx = body_world[local_id]
    g = gravity[wp.max(world_idx, 0)]

    f = m * dv / dt - m * g
    tau = wp.quat_rotate(r, I_body * wp.quat_rotate_inv(r, dw)) / dt

    wp.atomic_add(out_coupling_forces, global_id, wp.spatial_vector(f, tau))


@wp.kernel(enable_backward=False)
def _harvest_proxy_particle_momentum_forces_kernel(
    dt: float,
    particle_local_to_proxy_global: wp.array[int],
    qd_before: wp.array[wp.vec3],
    qd_after: wp.array[wp.vec3],
    particle_mass: wp.array[float],
    gravity: wp.array[wp.vec3],
    particle_world: wp.array[wp.int32],
    out_coupling_forces: wp.array[wp.vec3],
):
    """Estimate proxy particle feedback force from destination velocity change."""
    local_id = wp.tid()
    global_id = particle_local_to_proxy_global[local_id]
    if global_id < 0:
        return

    dv = qd_after[local_id] - qd_before[local_id]
    m = particle_mass[local_id]

    world_idx = particle_world[local_id]
    g = gravity[wp.max(world_idx, 0)]

    wp.atomic_add(out_coupling_forces, global_id, m * dv / dt - m * g)
