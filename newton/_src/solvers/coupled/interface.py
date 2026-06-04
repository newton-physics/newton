# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Interface contract for multi-solver coupling.

Solvers that participate in coupled simulations inherit
:class:`CouplingInterface` and override hook methods only when they need
solver-specific behavior. The mixin methods provide generic defaults derived
from the solver's model and the hook arguments.

Hook method contract
--------------------

Hooks are instance methods with default implementations. A solver that cannot
support a hook should override that method and raise
:class:`NotImplementedError`.

Endpoint arrays use structure-of-arrays indexing. ``endpoint_kind`` contains
``CouplingInterface.EndpointKind`` values, ``endpoint_index`` contains local body
or particle ids in the solver's model view, and ``endpoint_local_pos`` stores
the body-frame point for body endpoints [m] or zero for particles.

Proxy maps are dense local-to-global arrays. ``body_local_to_proxy_global`` and
``particle_local_to_proxy_global`` are indexed by local ids in the destination
solver's model view; proxy entries contain the corresponding global proxy id in
the shared model, while non-proxy entries contain ``-1``. Output force buffers
passed to harvest hooks are indexed by those global proxy ids.

Supported hook signatures are:

.. code-block:: python

    def coupling_eval_effective_mass(endpoint_kind, endpoint_index, endpoint_local_pos, out) -> None: ...


    def coupling_eval_effective_mass_block(
        endpoint_kind, endpoint_index, endpoint_local_pos, out_mass, out_inertia=None
    ) -> None: ...


    def coupling_notify_input_state_update(state, flags, *, restart=False, dt=0.0) -> None: ...


    def coupling_rewind_proxy_body_velocity(body_local_to_proxy_global, state, coupling_forces, dt) -> None: ...


    def coupling_rewind_proxy_particle_velocity(particle_local_to_proxy_global, state, coupling_forces, dt) -> None: ...


    def coupling_harvest_proxy_wrenches(
        body_local_to_proxy_global,
        out_body_f,
        *,
        body_qd_before=None,
        state=None,
        state_out=None,
        contacts=None,
        dt=0.0,
    ) -> None: ...


    def coupling_harvest_proxy_particle_forces(
        particle_local_to_proxy_global,
        out_particle_f,
        *,
        particle_qd_before=None,
        state=None,
        state_out=None,
        contacts=None,
        dt=0.0,
    ) -> None: ...


    def coupling_prepare_proxy_contacts(state, contacts, *, contacts_freshly_detected=False): ...
"""

from __future__ import annotations

from enum import IntEnum, IntFlag
from typing import TYPE_CHECKING

import numpy as np
import warp as wp

from .proxy_utils import (
    harvest_proxy_momentum_forces_kernel,
    harvest_proxy_particle_momentum_forces_kernel,
    subtract_proxy_forces_kernel,
    subtract_proxy_particle_forces_kernel,
)

if TYPE_CHECKING:
    from ...sim import Contacts, State

__all__ = ["CouplingInterface"]


class CouplingInterface:
    """Marker mixin for solvers that participate in coupled simulations.

    Inheriting buys into the coupling contract:

    - Override hook methods on the solver class to provide custom behavior.
      Otherwise, the mixin's generic defaults are used.
    - Override a hook and raise :class:`NotImplementedError` when no generic
      default can produce a meaningful result for the solver.

    The nested ``InputStateFlags`` and ``EndpointKind`` enums keep the public
    coupling namespace compact.
    """

    class EndpointKind(IntEnum):
        """Kinds of model endpoints addressed by coupling hooks."""

        BODY = 0
        PARTICLE = 1

    class InputStateFlags(IntFlag):
        """Input state arrays the coupler updated on a sub-solver's state."""

        BODY_Q = 1 << 0
        BODY_QD = 1 << 1
        PARTICLE_Q = 1 << 2
        PARTICLE_QD = 1 << 3
        JOINT_Q = 1 << 4
        JOINT_QD = 1 << 5
        BODY_F = 1 << 6
        PARTICLE_F = 1 << 7
        JOINT_F = 1 << 8

        BODY = BODY_Q | BODY_QD
        PARTICLE = PARTICLE_Q | PARTICLE_QD
        JOINT = JOINT_Q | JOINT_QD
        FORCE = BODY_F | PARTICLE_F | JOINT_F
        ALL = BODY | PARTICLE | JOINT | FORCE

    def coupling_eval_effective_mass(
        self,
        endpoint_kind: wp.array[int],
        endpoint_index: wp.array[int],
        endpoint_local_pos: wp.array[wp.vec3],
        out: wp.array[float],
    ) -> None:
        """Evaluate scalar effective masses for coupling endpoints.

        Args:
            endpoint_kind: Endpoint kinds.
            endpoint_index: Endpoint-local body or particle ids.
            endpoint_local_pos: Body-frame endpoint positions [m].
            out: Output effective masses [kg].
        """
        del endpoint_local_pos
        if out.shape[0] == 0:
            return

        model = self.model
        body_inv_mass = model.body_inv_mass.numpy() if getattr(model, "body_inv_mass", None) is not None else []
        particle_inv_mass = (
            model.particle_inv_mass.numpy() if getattr(model, "particle_inv_mass", None) is not None else []
        )

        values: list[float] = []
        for raw_kind, raw_index in zip(endpoint_kind.numpy(), endpoint_index.numpy(), strict=True):
            kind = int(raw_kind)
            index = int(raw_index)
            if kind == int(CouplingInterface.EndpointKind.BODY):
                inv_mass = float(body_inv_mass[index]) if 0 <= index < len(body_inv_mass) else 0.0
            elif kind == int(CouplingInterface.EndpointKind.PARTICLE):
                inv_mass = float(particle_inv_mass[index]) if 0 <= index < len(particle_inv_mass) else 0.0
            else:
                raise ValueError(f"Unknown coupling endpoint kind {kind}")
            values.append(0.0 if inv_mass == 0.0 else 1.0 / inv_mass)

        wp.copy(out, wp.array(values, dtype=float, device=model.device))

    def coupling_eval_effective_mass_block(
        self,
        endpoint_kind: wp.array[int],
        endpoint_index: wp.array[int],
        endpoint_local_pos: wp.array[wp.vec3],
        out_mass: wp.array[float],
        out_inertia: wp.array[wp.mat33] | None = None,
    ) -> None:
        """Evaluate effective mass and inertia blocks for coupling endpoints.

        Args:
            endpoint_kind: Endpoint kinds.
            endpoint_index: Endpoint-local body or particle ids.
            endpoint_local_pos: Body-frame endpoint positions [m].
            out_mass: Output effective masses [kg].
            out_inertia: Optional output body inertia tensors [kg m^2].
        """
        self.coupling_eval_effective_mass(endpoint_kind, endpoint_index, endpoint_local_pos, out_mass)
        if out_inertia is None or out_inertia.shape[0] == 0:
            return

        model = self.model
        masses = out_mass.numpy()
        body_mass = model.body_mass.numpy() if getattr(model, "body_mass", None) is not None else []
        body_inertia = model.body_inertia.numpy() if getattr(model, "body_inertia", None) is not None else []

        inertias: list[wp.mat33] = []
        for raw_kind, raw_index, mass in zip(endpoint_kind.numpy(), endpoint_index.numpy(), masses, strict=True):
            kind = int(raw_kind)
            index = int(raw_index)
            if kind != int(CouplingInterface.EndpointKind.BODY) or index < 0 or index >= len(body_inertia):
                inertias.append(wp.mat33(0.0))
                continue

            inertia = np.asarray(body_inertia[index], dtype=np.float32)
            base_mass = float(body_mass[index]) if index < len(body_mass) else 0.0
            if base_mass > 0.0:
                inertia = inertia * (float(mass) / base_mass)
            inertias.append(wp.mat33(inertia))

        wp.copy(out_inertia, wp.array(inertias, dtype=wp.mat33, device=model.device))

    def coupling_notify_input_state_update(
        self,
        state: State,
        flags: InputStateFlags | int,
        *,
        restart: bool = False,
        dt: float = 0.0,
    ) -> None:
        """React to coupler-produced input state updates."""
        del state, flags, restart, dt

    def coupling_rewind_proxy_body_velocity(
        self,
        body_local_to_proxy_global: wp.array[int],
        state: State,
        coupling_forces: wp.array[wp.spatial_vector],
        dt: float,
    ) -> None:
        """Remove lagged proxy feedback, public forces, and gravity from body velocities."""
        if body_local_to_proxy_global.shape[0] == 0 or state.body_qd is None:
            return

        model = self.model
        wp.launch(
            subtract_proxy_forces_kernel,
            dim=body_local_to_proxy_global.shape[0],
            inputs=[
                float(dt),
                model.gravity,
                model.body_world,
                state.body_q,
                state.body_f,
                coupling_forces,
                body_local_to_proxy_global,
                model.body_inv_mass,
                model.body_inv_inertia,
                state.body_qd,
            ],
            device=model.device,
        )

    def coupling_rewind_proxy_particle_velocity(
        self,
        particle_local_to_proxy_global: wp.array[int],
        state: State,
        coupling_forces: wp.array[wp.vec3],
        dt: float,
    ) -> None:
        """Remove lagged proxy feedback, public forces, and gravity from particle velocities."""
        if particle_local_to_proxy_global.shape[0] == 0 or state.particle_qd is None:
            return

        model = self.model
        wp.launch(
            subtract_proxy_particle_forces_kernel,
            dim=particle_local_to_proxy_global.shape[0],
            inputs=[
                float(dt),
                model.gravity,
                model.particle_world,
                state.particle_f,
                coupling_forces,
                particle_local_to_proxy_global,
                model.particle_inv_mass,
                state.particle_qd,
            ],
            device=model.device,
        )

    def coupling_harvest_proxy_wrenches(
        self,
        body_local_to_proxy_global: wp.array[int],
        out_body_f: wp.array[wp.spatial_vector],
        *,
        body_qd_before: wp.array[wp.spatial_vector] | None = None,
        state: State | None = None,
        state_out: State | None = None,
        contacts: Contacts | None = None,
        dt: float = 0.0,
    ) -> None:
        """Estimate proxy-body feedback from destination momentum change."""
        del state, contacts
        if body_local_to_proxy_global.shape[0] == 0:
            return
        if body_qd_before is None or state_out is None or state_out.body_qd is None:
            raise ValueError("Default body proxy harvest requires body_qd_before and state_out.body_qd")
        if dt <= 0.0:
            raise ValueError("Default body proxy harvest requires dt > 0")

        model = self.model
        wp.launch(
            harvest_proxy_momentum_forces_kernel,
            dim=body_local_to_proxy_global.shape[0],
            inputs=[
                float(dt),
                body_local_to_proxy_global,
                body_qd_before,
                state_out.body_qd,
                model.body_mass,
                model.body_inertia,
                state_out.body_q,
                model.gravity,
                model.body_world,
                out_body_f,
            ],
            device=model.device,
        )

    def coupling_harvest_proxy_particle_forces(
        self,
        particle_local_to_proxy_global: wp.array[int],
        out_particle_f: wp.array[wp.vec3],
        *,
        particle_qd_before: wp.array[wp.vec3] | None = None,
        state: State | None = None,
        state_out: State | None = None,
        contacts: Contacts | None = None,
        dt: float = 0.0,
    ) -> None:
        """Estimate proxy-particle feedback from destination momentum change."""
        del state, contacts
        if particle_local_to_proxy_global.shape[0] == 0:
            return
        if particle_qd_before is None or state_out is None or state_out.particle_qd is None:
            raise ValueError("Default particle proxy harvest requires particle_qd_before and state_out.particle_qd")
        if dt <= 0.0:
            raise ValueError("Default particle proxy harvest requires dt > 0")

        model = self.model
        wp.launch(
            harvest_proxy_particle_momentum_forces_kernel,
            dim=particle_local_to_proxy_global.shape[0],
            inputs=[
                float(dt),
                particle_local_to_proxy_global,
                particle_qd_before,
                state_out.particle_qd,
                model.particle_mass,
                model.gravity,
                model.particle_world,
                out_particle_f,
            ],
            device=model.device,
        )

    def coupling_prepare_proxy_contacts(
        self,
        state: State,
        contacts: Contacts | None,
        *,
        contacts_freshly_detected: bool = False,
    ) -> Contacts | None:
        """Prepare contacts for a proxy destination solve."""
        del state, contacts_freshly_detected
        return contacts


CouplingEndpointKind = CouplingInterface.EndpointKind
CouplingInputStateFlags = CouplingInterface.InputStateFlags
