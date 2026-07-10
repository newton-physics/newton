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


    def coupling_supports_inertial_property_refresh() -> bool: ...


    def coupling_rewind_proxy_body(
        body_local_to_proxy_global, state, coupling_forces, body_gravity_acceleration, dt
    ) -> None: ...


    def coupling_rewind_proxy_particle(
        particle_local_to_proxy_global, state, coupling_forces, particle_gravity_acceleration, dt
    ) -> None: ...


    def coupling_harvest_proxy_wrenches(
        body_local_to_proxy_global,
        out_body_f,
        *,
        body_qd_before,
        state,
        state_out,
        contacts,
        dt,
    ) -> None: ...


    def coupling_harvest_proxy_particle_forces(
        particle_local_to_proxy_global,
        out_particle_f,
        *,
        particle_qd_before,
        state,
        state_out,
        contacts,
        dt,
    ) -> None: ...


    def coupling_prepare_proxy_contacts(state, contacts, *, contacts_freshly_detected=False): ...
"""

from __future__ import annotations

from enum import IntEnum
from typing import TYPE_CHECKING

import warp as wp

from ...geometry import ParticleFlags
from ...sim import BodyFlags, StateFlags
from .proxy_utils import (
    filter_proxy_rigid_contacts_kernel,
    harvest_proxy_momentum_forces_kernel,
    harvest_proxy_particle_momentum_forces_kernel,
    subtract_proxy_body_forces_kernel,
    subtract_proxy_particle_forces_kernel,
)

if TYPE_CHECKING:
    from ...sim import Contacts, State

__all__ = ["CouplingInterface"]


class CouplingInterface:
    """Marker mixin for solvers that participate in coupled simulations.

    .. experimental::

    Inheriting buys into the coupling contract:

    - Override hook methods on the solver class to provide custom behavior.
      Otherwise, the mixin's generic defaults are used.
    - Override a hook and raise :class:`NotImplementedError` when no generic
      default can produce a meaningful result for the solver.

    ``EndpointKind`` stays nested because it is coupling-specific. Input update
    notifications reuse :class:`newton.StateFlags`.
    """

    class EndpointKind(IntEnum):
        """Kinds of model endpoints addressed by coupling hooks."""

        BODY = 0
        PARTICLE = 1

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
            out_inertia: Optional output body inertia tensors [kg m^2]. Body
                effective inertia must not be smaller than modeled inertia
                around any axis.
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
        flags: StateFlags | int,
        *,
        iteration_restart: bool = False,
        dt: float = 0.0,
    ) -> None:
        """React to coupler-produced public input updates.

        ``flags`` uses :class:`~newton.StateFlags` bits for both kinematic
        state arrays and public force-input buffers.
        """
        del state, flags, iteration_restart, dt

    def coupling_supports_inertial_property_refresh(self) -> bool:
        """Return whether inertial property refresh is safe during graph capture.

        Solvers that read mass and inertia arrays directly, or can refresh
        their derived inertial buffers with device work only, should override
        this to return ``True`` and provide a graph-capturable implementation
        of :meth:`notify_model_changed` for BODY_INERTIAL_PROPERTIES.
        """
        return False

    def coupling_eval_gravity_acceleration(
        self,
        out_body_acceleration: wp.array[wp.vec3] | None,
        out_particle_acceleration: wp.array[wp.vec3] | None,
    ) -> None:
        """Evaluate solver-applied gravity-like acceleration for all local entities.

        The coupled solvers cache these arrays at initialization and refresh
        them on relevant model changes. Solvers that apply scaled or compensated
        gravity should override this hook so proxy and ADMM coupling can remove
        exactly the acceleration the sub-solver will apply internally.

        Args:
            out_body_acceleration: Optional output per local body [m/s^2].
            out_particle_acceleration: Optional output per local particle [m/s^2].
        """
        model = self.model
        if out_body_acceleration is not None and out_body_acceleration.shape[0] > 0:
            wp.launch(
                _coupling_eval_body_gravity_acceleration_kernel,
                dim=out_body_acceleration.shape[0],
                inputs=[model.gravity, model.body_world],
                outputs=[out_body_acceleration],
                device=model.device,
            )
        if out_particle_acceleration is not None and out_particle_acceleration.shape[0] > 0:
            wp.launch(
                _coupling_eval_particle_gravity_acceleration_kernel,
                dim=out_particle_acceleration.shape[0],
                inputs=[model.gravity, model.particle_world],
                outputs=[out_particle_acceleration],
                device=model.device,
            )

    def coupling_rewind_proxy_body(
        self,
        body_local_to_proxy_global: wp.array[int],
        state: State,
        coupling_forces: wp.array[wp.spatial_vector],
        body_gravity_acceleration: wp.array[wp.vec3],
        dt: float,
    ) -> None:
        """Rewind lagged proxy-body feedback, gravity acceleration and external forces
        before the destination solve, so those are not double-counted.

        Implementations may update either ``state.body_qd`` or ``state.body_f``.
        """
        del dt
        if body_local_to_proxy_global.shape[0] == 0 or state.body_f is None:
            return

        model = self.model
        wp.launch(
            subtract_proxy_body_forces_kernel,
            dim=body_local_to_proxy_global.shape[0],
            inputs=[
                body_gravity_acceleration,
                state.body_f,
                coupling_forces,
                body_local_to_proxy_global,
                model.body_mass,
                model.body_inv_mass,
            ],
            device=model.device,
        )

    def coupling_rewind_proxy_particle(
        self,
        particle_local_to_proxy_global: wp.array[int],
        state: State,
        coupling_forces: wp.array[wp.vec3],
        particle_gravity_acceleration: wp.array[wp.vec3],
        dt: float,
    ) -> None:
        """Rewind lagged proxy-body feedback, gravity acceleration and external forces
        before the destination solve, so those are not double-counted.

        Implementations may update either ``state.particle_qd`` or ``state.particle_f``.
        """
        if particle_local_to_proxy_global.shape[0] == 0 or state.particle_qd is None:
            return

        model = self.model
        wp.launch(
            subtract_proxy_particle_forces_kernel,
            dim=particle_local_to_proxy_global.shape[0],
            inputs=[
                float(dt),
                particle_gravity_acceleration,
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
        body_qd_before: wp.array[wp.spatial_vector],
        state: State,
        state_out: State,
        contacts: Contacts | None,
        dt: float,
    ) -> None:
        """Accumulate proxy-body feedback from destination momentum change."""
        del state, contacts
        if body_local_to_proxy_global.shape[0] == 0:
            return
        if state_out.body_qd is None:
            raise ValueError("Default body proxy harvest requires state_out.body_qd")
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
                out_body_f,
            ],
            device=model.device,
        )

    def coupling_harvest_proxy_particle_forces(
        self,
        particle_local_to_proxy_global: wp.array[int],
        out_particle_f: wp.array[wp.vec3],
        *,
        particle_qd_before: wp.array[wp.vec3],
        state: State,
        state_out: State,
        contacts: Contacts | None,
        dt: float,
    ) -> None:
        """Accumulate proxy-particle feedback from destination momentum change."""
        del state, contacts
        if particle_local_to_proxy_global.shape[0] == 0:
            return
        if state_out.particle_qd is None:
            raise ValueError("Default particle proxy harvest requires state_out.particle_qd")
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
                model.particle_flags,
                int(ParticleFlags.ACTIVE),
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


@wp.kernel(enable_backward=False)
def _coupling_eval_body_gravity_acceleration_kernel(
    gravity: wp.array[wp.vec3],
    body_world: wp.array[wp.int32],
    out: wp.array[wp.vec3],
):
    i = wp.tid()
    out[i] = gravity[wp.max(body_world[i], 0)]


@wp.kernel(enable_backward=False)
def _coupling_eval_particle_gravity_acceleration_kernel(
    gravity: wp.array[wp.vec3],
    particle_world: wp.array[wp.int32],
    out: wp.array[wp.vec3],
):
    i = wp.tid()
    out[i] = gravity[wp.max(particle_world[i], 0)]


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
                    inertia = inertia * wp.max(out_mass[i] / mass, 1.0)

    out_inertia[i] = inertia


@wp.kernel(enable_backward=False)
def _coupling_zero_inertia_kernel(out_inertia: wp.array[wp.mat33]):
    out_inertia[wp.tid()] = wp.mat33(0.0)


CouplingEndpointKind = CouplingInterface.EndpointKind
