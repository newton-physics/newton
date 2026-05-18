# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Interface contract for multi-solver coupling.

Solvers that participate in coupled simulations inherit
:class:`CouplingInterface` and override the hook methods they want to provide a
custom implementation for. Hooks that are fundamentally incompatible with the
solver can be declared in :attr:`CouplingInterface.coupling_unsupported`.

The coupled wrapper detects custom hook implementations by name lookup on the
solver instance: if the solver defines the hook method, the wrapper calls it;
otherwise it uses a generic fallback derived from the shared model/state.

Hook method contract
--------------------

Hooks are optional instance methods. A solver should define only the methods
for which it needs custom behavior; missing methods use wrapper fallbacks.
Instance attributes set to ``None`` are also treated as missing, which lets a
solver disable a hook when construction-time state is unavailable.

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
        body_local_to_proxy_global, out_body_f, *, state=None, state_out=None, contacts=None, dt=0.0
    ) -> None: ...


    def coupling_harvest_proxy_particle_forces(
        particle_local_to_proxy_global, out_particle_f, *, state=None, state_out=None, contacts=None, dt=0.0
    ) -> None: ...


    def coupling_prepare_proxy_contacts(state, contacts, *, contacts_freshly_detected=False): ...
"""

from __future__ import annotations

from enum import IntEnum, IntFlag
from typing import ClassVar

__all__ = ["CouplingInterface"]


class CouplingInterface:
    """Marker mixin for solvers that participate in coupled simulations.

    Inheriting buys into the coupling contract:

    - Hook overrides are detected by method name lookup. Override the hook
      method on the solver class to provide a custom implementation; the
      coupled wrapper will call it. If the method is not defined, the wrapper
      falls back to its own generic logic derived from the shared model/state.
    - List a ``CouplingInterface.Hook`` value in
      :attr:`coupling_unsupported` to declare that no fallback can produce a
      meaningful result for this solver. The wrapper raises
      :class:`NotImplementedError` rather than silently using a fallback.

    The nested ``Hook``, ``InputStateFlags``, and ``EndpointKind`` enums keep
    the public coupling namespace compact.
    """

    class Hook(IntFlag):
        """Coupling dispatch points exposed by coupled solvers."""

        BODY_PROXY_REWIND_VELOCITY = 1 << 0
        PARTICLE_PROXY_REWIND_VELOCITY = 1 << 1
        BODY_PROXY_HARVEST = 1 << 2
        PARTICLE_PROXY_HARVEST = 1 << 3
        EFFECTIVE_MASS_DIAGONAL = 1 << 4
        EFFECTIVE_MASS_BLOCK = 1 << 5
        NOTIFY_INPUT_STATE_UPDATE = 1 << 6
        PROXY_CONTACT_PREPARE = 1 << 7

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

    coupling_unsupported: ClassVar[frozenset[Hook]] = frozenset()


CouplingHook = CouplingInterface.Hook
CouplingEndpointKind = CouplingInterface.EndpointKind
CouplingInputStateFlags = CouplingInterface.InputStateFlags


COUPLING_HOOK_METHOD: dict[CouplingHook, str] = {
    CouplingHook.BODY_PROXY_REWIND_VELOCITY: "coupling_rewind_proxy_body_velocity",
    CouplingHook.PARTICLE_PROXY_REWIND_VELOCITY: "coupling_rewind_proxy_particle_velocity",
    CouplingHook.BODY_PROXY_HARVEST: "coupling_harvest_proxy_wrenches",
    CouplingHook.PARTICLE_PROXY_HARVEST: "coupling_harvest_proxy_particle_forces",
    CouplingHook.EFFECTIVE_MASS_DIAGONAL: "coupling_eval_effective_mass",
    CouplingHook.EFFECTIVE_MASS_BLOCK: "coupling_eval_effective_mass_block",
    CouplingHook.NOTIFY_INPUT_STATE_UPDATE: "coupling_notify_input_state_update",
    CouplingHook.PROXY_CONTACT_PREPARE: "coupling_prepare_proxy_contacts",
}
