# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import warp as wp

from ..geometry.differentiable_contacts import _launch_rigid_contact_kinematics
from .contacts import Contacts
from .model import Model
from .state import State

__all__ = ["eval_rigid_contact_kinematics"]


def _validate_output(
    name: str,
    output: wp.array | None,
    dtype: type,
    contact_max: int,
    device,
) -> None:
    if output is None:
        return
    if not isinstance(output, wp.array):
        raise TypeError(f"{name} must be a Warp array or None, got {type(output).__name__}.")
    if output.dtype != dtype:
        raise TypeError(f"{name} has dtype {output.dtype}, expected {dtype}.")
    if output.shape != (contact_max,):
        raise ValueError(f"{name} has shape {output.shape}, expected ({contact_max},).")
    if output.device != device:
        raise ValueError(f"{name} is on device {output.device}, expected {device}.")


def eval_rigid_contact_kinematics(
    model: Model,
    state: State,
    contacts: Contacts,
    *,
    out_distance: wp.array[float] | None = None,
    out_point0_world: wp.array[wp.vec3] | None = None,
    out_point1_world: wp.array[wp.vec3] | None = None,
) -> None:
    """Evaluate selected rigid-contact kinematic quantities in caller-provided arrays.

    Reconstructs world-space contact points from the body-local points produced
    by collision detection and computes signed contact distance from those
    points, the frozen world-space contact normal, and the contact margins. The
    launch is recorded by :class:`wp.Tape`, so gradients can flow through
    ``state.body_q`` when the provided outputs require gradients. Gradients do
    not flow through the contact normal or the discrete contact set.

    Pass ``None`` for quantities that are not needed. The world-space contact
    normal is already available as :attr:`newton.Contacts.rigid_contact_normal`
    and does not need a derived output.

    Args:
        model: Model providing the shape-to-body mapping.
        state: State providing body transforms.
        contacts: Populated contacts whose rigid-contact geometry is evaluated.
        out_distance: Optional signed contact distance [m], shape
            ``(contacts.rigid_contact_max,)``, dtype float. Positive values are
            gaps and negative values are penetrations.
        out_point0_world: Optional world-space support point on shape 0 [m],
            shape ``(contacts.rigid_contact_max,)``, dtype :class:`wp.vec3`.
        out_point1_world: Optional world-space support point on shape 1 [m],
            shape ``(contacts.rigid_contact_max,)``, dtype :class:`wp.vec3`.

    Raises:
        ValueError: If no output is provided or an output has an unexpected
            shape or device.
        TypeError: If an output is not a Warp array or has an unexpected dtype.

    .. experimental::

        The contact set and normal are frozen outputs of non-differentiable
        collision detection. Resulting gradients are a first-order tangent
        approximation and may change without prior notice.
    """
    if out_distance is None and out_point0_world is None and out_point1_world is None:
        raise ValueError("At least one output must be provided.")

    if contacts.device != model.device:
        raise ValueError(f"contacts are on device {contacts.device}, expected {model.device}.")
    if state.body_q.device != model.device:
        raise ValueError(f"state.body_q is on device {state.body_q.device}, expected {model.device}.")

    _validate_output("out_distance", out_distance, wp.float32, contacts.rigid_contact_max, model.device)
    _validate_output("out_point0_world", out_point0_world, wp.vec3, contacts.rigid_contact_max, model.device)
    _validate_output("out_point1_world", out_point1_world, wp.vec3, contacts.rigid_contact_max, model.device)

    _launch_rigid_contact_kinematics(
        contacts,
        state.body_q,
        model.shape_body,
        out_distance=out_distance,
        out_normal=None,
        out_point0_world=out_point0_world,
        out_point1_world=out_point1_world,
        device=model.device,
    )
