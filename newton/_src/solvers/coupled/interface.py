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


    def coupling_notify_input_state_update(state, flags, *, iteration_restart=False, dt=0.0) -> None: ...


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

import warp as wp

from ...sim import BodyFlags
from .proxy_utils import (
    filter_proxy_rigid_contacts_kernel,
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
        body_inv_mass = getattr(model, "body_inv_mass", None)
        particle_inv_mass = getattr(model, "particle_inv_mass", None)
        if body_inv_mass is not None and particle_inv_mass is not None:
            wp.launch(
                _coupling_eval_effective_mass_kernel,
                dim=out.shape[0],
                inputs=[
                    endpoint_kind,
                    endpoint_index,
                    body_inv_mass,
                    particle_inv_mass,
                    out,
                ],
                device=model.device,
            )
        elif body_inv_mass is not None:
            wp.launch(
                _coupling_eval_effective_mass_body_kernel,
                dim=out.shape[0],
                inputs=[endpoint_kind, endpoint_index, body_inv_mass, out],
                device=model.device,
            )
        elif particle_inv_mass is not None:
            wp.launch(
                _coupling_eval_effective_mass_particle_kernel,
                dim=out.shape[0],
                inputs=[endpoint_kind, endpoint_index, particle_inv_mass, out],
                device=model.device,
            )
        else:
            wp.launch(_coupling_zero_mass_kernel, dim=out.shape[0], inputs=[out], device=model.device)

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
        body_mass = getattr(model, "body_mass", None)
        body_inertia = getattr(model, "body_inertia", None)
        if body_mass is None or body_inertia is None:
            wp.launch(
                _coupling_zero_inertia_kernel,
                dim=out_inertia.shape[0],
                inputs=[out_inertia],
                device=model.device,
            )
            return

        wp.launch(
            _coupling_eval_effective_inertia_kernel,
            dim=out_inertia.shape[0],
            inputs=[
                endpoint_kind,
                endpoint_index,
                body_mass,
                body_inertia,
                out_mass,
                out_inertia,
            ],
            device=model.device,
        )

    def coupling_notify_input_state_update(
        self,
        state: State,
        flags: InputStateFlags | int,
        *,
        iteration_restart: bool = False,
        dt: float = 0.0,
    ) -> None:
        """React to coupler-produced input state updates."""
        del state, flags, iteration_restart, dt

    def coupling_rewind_proxy_body_velocity(
        self,
        body_local_to_proxy_global: wp.array[int],
        state: State,
        coupling_forces: wp.array[wp.spatial_vector],
        dt: float,
    ) -> None:
        """Remove velocity-level feedback, public forces, and gravity from proxy velocities.

        The default proxy feedback is harvested from destination momentum
        change, so ``coupling_forces`` are treated as lagged velocity-level
        response and rewound before the destination solve. Solvers whose
        feedback is position-dependent, such as barrier-style contact, should
        override this hook and leave ``coupling_forces`` in the synced velocity.
        """
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
        """Remove velocity-level feedback, public forces, and gravity from proxy velocities.

        The default proxy feedback is harvested from destination momentum
        change, so ``coupling_forces`` are treated as lagged velocity-level
        response and rewound before the destination solve. Solvers whose
        feedback is position-dependent, such as barrier-style contact, should
        override this hook and leave ``coupling_forces`` in the synced velocity.
        """
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
        del contacts
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
                state.body_f if state is not None else None,
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
        del contacts
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
                state.particle_f if state is not None else None,
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
        """Prepare contacts for a proxy destination solve.

        The generic momentum harvest treats proxy feedback as a destination
        momentum change. Proxy-static and proxy-proxy rigid contacts therefore
        must not be passed through as solver contacts because they would feed
        constraints between virtual objects back to the source.
        """
        del state, contacts_freshly_detected
        if contacts is None or contacts.rigid_contact_count is None or contacts.rigid_contact_max == 0:
            return contacts

        model = self.model
        wp.launch(
            filter_proxy_rigid_contacts_kernel,
            dim=contacts.rigid_contact_shape0.shape[0],
            inputs=[
                contacts.rigid_contact_count,
                contacts.rigid_contact_shape0,
                contacts.rigid_contact_shape1,
                model.shape_body,
                model.body_flags,
                model.body_inv_mass,
                int(BodyFlags.PROXY),
            ],
            device=model.device,
        )
        return contacts


@wp.func
def _mass_from_inverse(inv_mass: float) -> float:
    if inv_mass == 0.0:
        return 0.0
    return 1.0 / inv_mass


@wp.kernel(enable_backward=False)
def _coupling_eval_effective_mass_kernel(
    endpoint_kind: wp.array[int],
    endpoint_index: wp.array[int],
    body_inv_mass: wp.array[float],
    particle_inv_mass: wp.array[float],
    out: wp.array[float],
):
    i = wp.tid()
    kind = endpoint_kind[i]
    index = endpoint_index[i]
    inv_mass = 0.0

    if kind == wp.static(int(CouplingInterface.EndpointKind.BODY)):
        if index >= 0 and index < body_inv_mass.shape[0]:
            inv_mass = body_inv_mass[index]
    elif kind == wp.static(int(CouplingInterface.EndpointKind.PARTICLE)):
        if index >= 0 and index < particle_inv_mass.shape[0]:
            inv_mass = particle_inv_mass[index]

    out[i] = _mass_from_inverse(inv_mass)


@wp.kernel(enable_backward=False)
def _coupling_eval_effective_mass_body_kernel(
    endpoint_kind: wp.array[int],
    endpoint_index: wp.array[int],
    inv_mass: wp.array[float],
    out: wp.array[float],
):
    i = wp.tid()
    mass = 0.0
    index = endpoint_index[i]
    if endpoint_kind[i] == wp.static(int(CouplingInterface.EndpointKind.BODY)) and index >= 0:
        if index < inv_mass.shape[0]:
            mass = _mass_from_inverse(inv_mass[index])
    out[i] = mass


@wp.kernel(enable_backward=False)
def _coupling_eval_effective_mass_particle_kernel(
    endpoint_kind: wp.array[int],
    endpoint_index: wp.array[int],
    inv_mass: wp.array[float],
    out: wp.array[float],
):
    i = wp.tid()
    mass = 0.0
    index = endpoint_index[i]
    if endpoint_kind[i] == wp.static(int(CouplingInterface.EndpointKind.PARTICLE)) and index >= 0:
        if index < inv_mass.shape[0]:
            mass = _mass_from_inverse(inv_mass[index])
    out[i] = mass


@wp.kernel(enable_backward=False)
def _coupling_zero_mass_kernel(out: wp.array[float]):
    out[wp.tid()] = 0.0


@wp.kernel(enable_backward=False)
def _coupling_eval_effective_inertia_kernel(
    endpoint_kind: wp.array[int],
    endpoint_index: wp.array[int],
    body_mass: wp.array[float],
    body_inertia: wp.array[wp.mat33],
    out_mass: wp.array[float],
    out_inertia: wp.array[wp.mat33],
):
    i = wp.tid()
    index = endpoint_index[i]
    inertia = wp.mat33(0.0)

    if endpoint_kind[i] == wp.static(int(CouplingInterface.EndpointKind.BODY)) and index >= 0:
        if index < body_inertia.shape[0]:
            inertia = body_inertia[index]
            if index < body_mass.shape[0]:
                mass = body_mass[index]
                if mass > 0.0:
                    inertia = inertia * (out_mass[i] / mass)

    out_inertia[i] = inertia


@wp.kernel(enable_backward=False)
def _coupling_zero_inertia_kernel(out_inertia: wp.array[wp.mat33]):
    out_inertia[wp.tid()] = wp.mat33(0.0)


CouplingEndpointKind = CouplingInterface.EndpointKind
CouplingInputStateFlags = CouplingInterface.InputStateFlags
