# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Post-processing kernels that augment rigid contacts with differentiable data.

The narrow-phase collision kernels use ``enable_backward=False`` so they are
never recorded on a :class:`wp.Tape`.  This module provides lightweight kernels
that re-read the frozen contact geometry (body-local points, world normal,
margins) produced by the narrow phase and reconstruct world-space quantities
through the *differentiable* body transforms ``body_q``.

The resulting arrays carry ``requires_grad=True`` and participate in autodiff,
giving first-order (tangent-plane) gradients of contact distance and world-space
contact points with respect to body poses.

Two variants are provided:

* **Standard** (:func:`differentiable_contact_augment_kernel`) — the frozen
  world-space normal passes through unchanged.  Gradients flow through the
  contact *points* and *distance* but **not** through the normal direction.
  This is cheaper and sufficient when the optimisation objective depends only on
  penetration depth.

* **Rotation-invariant** (:func:`differentiable_contact_augment_rotation_invariant_kernel`)
  — the frozen normal is first expressed in the slerp-midpoint frame of the two
  body orientations (off-tape), then rotated back to world space through the
  differentiable ``body_q``.  This makes the output normal differentiable with
  respect to body orientations, which is important for objectives that depend on
  the contact normal direction (e.g. grasp quality, friction-cone alignment).
  See :func:`launch_differentiable_contact_augment_rotation_invariant` for why
  a two-kernel split is required to break the gradient symmetry.
"""

from __future__ import annotations

import warp as wp


@wp.func
def _slerp_midpoint(q_a: wp.quat, q_b: wp.quat) -> wp.quat:
    """Geodesic midpoint of two quaternions (specialised ``slerp`` at ``t = 0.5``).

    For the general ``t`` the slerp formula
    ``sin((1-t)θ) / sin(θ) · q_a + sin(tθ) / sin(θ) · q_b`` has a ``0/0``
    singularity at ``θ = 0`` (identical orientations) whose automatic adjoint
    in Warp produces zero gradients.

    At ``t = 0.5`` the two coefficients are equal,
    ``sin(θ/2) / sin(θ) = 1 / (2 cos(θ/2))``, so the result simplifies to
    ``normalize((q_a + q_b) / (2 cos(θ/2)))``.  Since normalization removes
    positive scalars this is **algebraically identical** to
    ``normalize(q_a + q_b)`` — no approximation involved.  The sum
    ``q_a + q_b`` has magnitude ``2 cos(θ/2)`` which is ``2`` when ``θ = 0``
    (not zero), so the gradient of ``normalize`` is well-defined everywhere
    except when the quaternions are antipodal (``θ = π``), which the
    hemisphere-flip below prevents.
    """
    if wp.dot(q_a, q_b) < 0.0:
        q_b = -q_b
    return wp.normalize(q_a + q_b)


@wp.kernel
def differentiable_contact_augment_kernel(
    body_q: wp.array(dtype=wp.transform),
    shape_body: wp.array(dtype=int),
    contact_count: wp.array(dtype=int),
    contact_shape0: wp.array(dtype=int),
    contact_shape1: wp.array(dtype=int),
    contact_point0: wp.array(dtype=wp.vec3),
    contact_point1: wp.array(dtype=wp.vec3),
    contact_normal: wp.array(dtype=wp.vec3),
    contact_margin0: wp.array(dtype=float),
    contact_margin1: wp.array(dtype=float),
    # outputs
    out_distance: wp.array(dtype=float),
    out_normal: wp.array(dtype=wp.vec3),
    out_point0_world: wp.array(dtype=wp.vec3),
    out_point1_world: wp.array(dtype=wp.vec3),
):
    """Standard differentiable contact augmentation.

    Transforms body-local contact points into world space through the
    differentiable ``body_q`` and computes the signed contact distance.
    The world-space normal is passed through from the narrow phase as-is
    (frozen, no orientation gradients).

    **Design note:** the normal is *not* rotated through ``body_q`` here.
    For the original kernel this is intentional — the narrow-phase normal is
    already in world space, so re-rotating it through the same differentiable
    orientation would compose ``R(q) R(q)^{-1} n = n``, yielding a constant
    with zero total derivative regardless of how tape-based AD traces through
    it.  See :func:`differentiable_contact_augment_rotation_invariant_kernel`
    for a variant that breaks this symmetry with a two-kernel split to obtain
    non-trivial normal-direction gradients.

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

    n = contact_normal[tid]
    thickness = contact_margin0[tid] + contact_margin1[tid]
    d = wp.dot(n, bx_b - bx_a) - thickness

    out_distance[tid] = d
    out_normal[tid] = n
    out_point0_world[tid] = bx_a
    out_point1_world[tid] = bx_b


@wp.kernel
def _contact_normal_to_avg_frame_kernel(
    body_q: wp.array(dtype=wp.transform),
    shape_body: wp.array(dtype=int),
    contact_count: wp.array(dtype=int),
    contact_shape0: wp.array(dtype=int),
    contact_shape1: wp.array(dtype=int),
    contact_normal: wp.array(dtype=wp.vec3),
    out_normal_local: wp.array(dtype=wp.vec3),
):
    """Project world-space normals into the slerp-average body frame.

    For each contact, computes ``n_local = R(q_avg)^{-1} n`` where ``q_avg``
    is ``slerp(q_a, q_b, 0.5)`` — the geodesic midpoint of the two body
    orientations.  The midpoint is chosen because it is symmetric: swapping
    the two bodies does not change the reference frame.

    **Must** be launched with ``record_tape=False`` so that
    ``out_normal_local`` acts as a frozen (non-differentiable) constant for
    the subsequent rotation-invariant kernel.  This is the mechanism that
    breaks the ``R(q) R(q)^{-1}`` identity and allows the forward rotation
    in the second kernel to produce non-trivial orientation gradients.
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

    q_a = wp.quat_identity()
    q_b = wp.quat_identity()
    if body_a >= 0:
        q_a = wp.transform_get_rotation(body_q[body_a])
    if body_b >= 0:
        q_b = wp.transform_get_rotation(body_q[body_b])

    q_avg = _slerp_midpoint(q_a, q_b)
    q_avg_inv = wp.quat_inverse(q_avg)

    out_normal_local[tid] = wp.quat_rotate(q_avg_inv, contact_normal[tid])


@wp.kernel
def differentiable_contact_augment_rotation_invariant_kernel(
    body_q: wp.array(dtype=wp.transform),
    shape_body: wp.array(dtype=int),
    contact_count: wp.array(dtype=int),
    contact_shape0: wp.array(dtype=int),
    contact_shape1: wp.array(dtype=int),
    contact_point0: wp.array(dtype=wp.vec3),
    contact_point1: wp.array(dtype=wp.vec3),
    contact_normal_local: wp.array(dtype=wp.vec3),
    contact_margin0: wp.array(dtype=float),
    contact_margin1: wp.array(dtype=float),
    # outputs
    out_distance: wp.array(dtype=float),
    out_normal: wp.array(dtype=wp.vec3),
    out_point0_world: wp.array(dtype=wp.vec3),
    out_point1_world: wp.array(dtype=wp.vec3),
):
    """Rotation-invariant variant of :func:`differentiable_contact_augment_kernel`.

    The contact normal co-rotates with the bodies: it is stored in the
    slerp-midpoint frame of the two body orientations and rotated back to world
    space through the differentiable ``body_q``.  This provides first-order
    gradients of both contact distance *and* normal direction with respect to
    body orientations.

    **Why a separate prep kernel is needed:** within a single autodiff kernel,
    ``n_world = R(q_avg) · R(q_avg)^{-1} · n`` is algebraically the identity
    and tape-based AD correctly computes ``dn_world/dq = 0``.  By computing
    ``n_local = R(q_avg)^{-1} · n`` in a prior kernel launched with
    ``record_tape=False``, the tape only sees the forward rotation
    ``n_world = R(q_avg) · n_local`` where ``n_local`` is a frozen constant.
    The Jacobian ``dn_world/dq_avg`` is then the skew-symmetric cross-product
    matrix of the rotated normal, giving meaningful first-order gradients.

    The forward-pass result is exact: at the current body poses the two
    rotations still compose to the identity, so ``n_world`` equals the
    original narrow-phase normal.

    Outputs (per contact):

    * ``out_distance`` — signed gap ``dot(n, p_b - p_a) - thickness`` [m].
    * ``out_normal`` — world-space contact normal (differentiable w.r.t.
      body orientations via the slerp-midpoint rotation).
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

    q_a = wp.transform_get_rotation(X_wb_a)
    q_b = wp.transform_get_rotation(X_wb_b)
    q_avg = _slerp_midpoint(q_a, q_b)
    n = wp.quat_rotate(q_avg, contact_normal_local[tid])

    thickness = contact_margin0[tid] + contact_margin1[tid]
    d = wp.dot(n, bx_b - bx_a) - thickness

    out_distance[tid] = d
    out_normal[tid] = n
    out_point0_world[tid] = bx_a
    out_point1_world[tid] = bx_b


def launch_differentiable_contact_augment(
    contacts,
    body_q: wp.array,
    shape_body: wp.array,
    device=None,
):
    """Launch the standard differentiable contact augmentation kernel.

    This is the cheaper variant: gradients flow through the contact points
    and distance but the normal direction is frozen.  Use
    :func:`launch_differentiable_contact_augment_rotation_invariant` when
    the objective also depends on the normal direction.

    Args:
        contacts: :class:`~newton.Contacts` instance with differentiable arrays allocated.
        body_q: Body transforms, shape ``(body_count,)``, dtype :class:`wp.transform`.
        shape_body: Per-shape body index, shape ``(shape_count,)``, dtype ``int``.
        device: Warp device.
    """
    wp.launch(
        kernel=differentiable_contact_augment_kernel,
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
        outputs=[
            contacts.rigid_contact_diff_distance,
            contacts.rigid_contact_diff_normal,
            contacts.rigid_contact_diff_point0_world,
            contacts.rigid_contact_diff_point1_world,
        ],
        device=device,
    )


def launch_differentiable_contact_augment_rotation_invariant(
    contacts,
    body_q: wp.array,
    shape_body: wp.array,
    device=None,
):
    """Launch the rotation-invariant differentiable contact augmentation.

    Unlike :func:`launch_differentiable_contact_augment`, this variant makes
    the output normal differentiable with respect to body orientations by
    expressing the narrow-phase normal in the slerp-midpoint body frame and
    rotating it back through the differentiable ``body_q``.

    Internally runs two kernels:

    1. **Prep** (not recorded on tape) — projects the frozen world-space
       normal into the average body frame.
    2. **Main** (recorded on tape) — reconstructs world-space quantities
       from ``body_q``, using the average-frame normal so that
       ``out_normal`` carries orientation gradients.

    Args:
        contacts: :class:`~newton.Contacts` instance with differentiable arrays allocated.
        body_q: Body transforms, shape ``(body_count,)``, dtype :class:`wp.transform`.
        shape_body: Per-shape body index, shape ``(shape_count,)``, dtype ``int``.
        device: Warp device.
    """
    n_max = contacts.rigid_contact_max
    if device is None:
        device = body_q.device

    if not hasattr(contacts, "_normal_avg_frame") or contacts._normal_avg_frame.shape[0] != n_max:
        contacts._normal_avg_frame = wp.zeros(n_max, dtype=wp.vec3, device=device, requires_grad=False)

    wp.launch(
        kernel=_contact_normal_to_avg_frame_kernel,
        dim=n_max,
        inputs=[
            body_q,
            shape_body,
            contacts.rigid_contact_count,
            contacts.rigid_contact_shape0,
            contacts.rigid_contact_shape1,
            contacts.rigid_contact_normal,
        ],
        outputs=[
            contacts._normal_avg_frame,
        ],
        device=device,
        record_tape=False,
    )

    wp.launch(
        kernel=differentiable_contact_augment_rotation_invariant_kernel,
        dim=n_max,
        inputs=[
            body_q,
            shape_body,
            contacts.rigid_contact_count,
            contacts.rigid_contact_shape0,
            contacts.rigid_contact_shape1,
            contacts.rigid_contact_point0,
            contacts.rigid_contact_point1,
            contacts._normal_avg_frame,
            contacts.rigid_contact_margin0,
            contacts.rigid_contact_margin1,
        ],
        outputs=[
            contacts.rigid_contact_diff_distance,
            contacts.rigid_contact_diff_normal,
            contacts.rigid_contact_diff_point0_world,
            contacts.rigid_contact_diff_point1_world,
        ],
        device=device,
    )
