# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Compute selected rigid-contact quantities from frozen contact geometry.

The narrow-phase collision kernels use ``enable_backward=False`` so they are
never recorded on a :class:`wp.Tape`.  This module provides lightweight kernels
that re-read the frozen contact geometry (body-local points, world normal,
margins) produced by the narrow phase and reconstruct world-space quantities
through the *differentiable* body transforms ``body_q``.

Caller-provided outputs can participate in autodiff, giving first-order
(tangent-plane) gradients of contact distance and world-space contact points
with respect to body poses. The frozen world-space normal passes through
unchanged — gradients flow through the contact *points* and *distance* but
**not** through the normal direction.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import warp as wp

if TYPE_CHECKING:
    from ..sim.contacts import Contacts
    from ..sim.model import Model
    from ..sim.state import State


@wp.kernel
def rigid_contact_kinematics_kernel(
    body_q: wp.array[wp.transform],
    shape_body: wp.array[int],
    contact_count: wp.array[int],
    contact_shape0: wp.array[int],
    contact_shape1: wp.array[int],
    contact_point0: wp.array[wp.vec3],
    contact_point1: wp.array[wp.vec3],
    contact_normal: wp.array[wp.vec3],
    contact_margin0: wp.array[float],
    contact_margin1: wp.array[float],
    # outputs
    out_distance: wp.array[float],
    out_normal: wp.array[wp.vec3],
    out_point0_world: wp.array[wp.vec3],
    out_point1_world: wp.array[wp.vec3],
):
    """Differentiable contact augmentation.

    Transforms body-local contact points into world space through the
    differentiable ``body_q`` and computes the signed contact distance.
    The world-space normal is passed through from the narrow phase as-is
    (frozen, no orientation gradients).

    Outputs (per contact):

    * ``out_distance`` — signed gap ``dot(n, p_b - p_a) - thickness`` [m].
    * ``out_normal`` — world-space contact normal (frozen, equals input).
    * ``out_point0_world`` — contact point on shape A in world space [m].
    * ``out_point1_world`` — contact point on shape B in world space [m].
    """
    tid = wp.tid()
    count = contact_count[0]
    if tid >= count:
        return

    shape_a = contact_shape0[tid]
    shape_b = contact_shape1[tid]

    body_a = -1
    if shape_a >= 0:
        body_a = shape_body[shape_a]
    body_b = -1
    if shape_b >= 0:
        body_b = shape_body[shape_b]

    X_wb_a = wp.transform_identity()
    X_wb_b = wp.transform_identity()
    if body_a >= 0:
        X_wb_a = body_q[body_a]
    if body_b >= 0:
        X_wb_b = body_q[body_b]

    bx_a = wp.transform_point(X_wb_a, contact_point0[tid])
    bx_b = wp.transform_point(X_wb_b, contact_point1[tid])

    if out_distance.shape[0] > 0:
        n = contact_normal[tid]
        thickness = contact_margin0[tid] + contact_margin1[tid]
        out_distance[tid] = wp.dot(n, bx_b - bx_a) - thickness
    if out_normal.shape[0] > 0:
        out_normal[tid] = contact_normal[tid]
    if out_point0_world.shape[0] > 0:
        out_point0_world[tid] = bx_a
    if out_point1_world.shape[0] > 0:
        out_point1_world[tid] = bx_b


def _launch_rigid_contact_kinematics(
    contacts: Contacts,
    body_q: wp.array[wp.transform],
    shape_body: wp.array[int],
    *,
    out_distance: wp.array[float] | None,
    out_normal: wp.array[wp.vec3] | None,
    out_point0_world: wp.array[wp.vec3] | None,
    out_point1_world: wp.array[wp.vec3] | None,
    device=None,
) -> None:
    wp.launch(
        kernel=rigid_contact_kinematics_kernel,
        dim=contacts.rigid_contact_max,
        inputs=[
            body_q,
            shape_body,
            contacts.rigid_contact_count,
            contacts.rigid_contact_shape0,
            contacts.rigid_contact_shape1,
            contacts.rigid_contact_point0,
            contacts.rigid_contact_point1,
            contacts.rigid_contact_normal,
            contacts.rigid_contact_margin0,
            contacts.rigid_contact_margin1,
        ],
        outputs=[out_distance, out_normal, out_point0_world, out_point1_world],
        device=device,
    )


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


def compute_rigid_contact_kinematics(
    model: Model,
    state: State,
    contacts: Contacts,
    *,
    out_distance: wp.array[float] | None = None,
    out_point0_world: wp.array[wp.vec3] | None = None,
    out_point1_world: wp.array[wp.vec3] | None = None,
) -> None:
    """Compute selected rigid-contact quantities in caller-provided arrays.

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


def launch_differentiable_contact_augment(
    contacts: Contacts,
    body_q: wp.array[wp.transform],
    shape_body: wp.array[int],
    device=None,
) -> None:
    """Launch the differentiable contact augmentation kernel.

    Gradients flow through the contact points and distance but the normal
    direction is frozen (constant).

    Args:
        contacts: :class:`~newton.Contacts` instance with differentiable arrays allocated.
        body_q: Body transforms, shape ``(body_count,)``, dtype :class:`wp.transform`.
        shape_body: Per-shape body index, shape ``(shape_count,)``, dtype ``int``.
        device: Warp device.
    """
    _launch_rigid_contact_kinematics(
        contacts,
        body_q,
        shape_body,
        out_distance=contacts._rigid_contact_diff_distance,
        out_normal=contacts._rigid_contact_diff_normal_override,
        out_point0_world=contacts._rigid_contact_diff_point0_world,
        out_point1_world=contacts._rigid_contact_diff_point1_world,
        device=device,
    )
