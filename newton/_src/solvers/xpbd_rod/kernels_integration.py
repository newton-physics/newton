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

"""Warp kernels for position and rotation integration.

This module contains kernels for XPBD-style position and rotation
prediction and integration.
"""

from __future__ import annotations

import warp as wp

from .kernels_math import _warp_quat_conjugate, _warp_quat_mul, _warp_quat_normalize


@wp.kernel
def _warp_predict_positions(
    positions: wp.array(dtype=wp.vec3),
    velocities: wp.array(dtype=wp.vec3),
    forces: wp.array(dtype=wp.vec3),
    inv_masses: wp.array(dtype=wp.float32),
    gravity: wp.vec3,
    dt: float,
    damping: float,
    predicted: wp.array(dtype=wp.vec3),
):
    """Predict particle positions for constraint projection.

    Updates velocities with forces and gravity, applies damping,
    then computes predicted positions.

    Args:
        positions: Current particle positions.
        velocities: Current particle velocities (updated in-place).
        forces: External forces on particles.
        inv_masses: Inverse masses (0 for fixed particles).
        gravity: Gravity vector.
        dt: Time step size.
        damping: Linear velocity damping factor.
        predicted: Output predicted positions.
    """
    i = wp.tid()
    inv_mass = inv_masses[i]
    if inv_mass > 0.0:
        v = velocities[i] + (forces[i] * inv_mass + gravity) * dt
        v = v * (1.0 - damping)
        velocities[i] = v
        predicted[i] = positions[i] + v * dt
    else:
        velocities[i] = wp.vec3(0.0, 0.0, 0.0)
        predicted[i] = positions[i]


@wp.kernel
def _warp_integrate_positions(
    positions: wp.array(dtype=wp.vec3),
    predicted: wp.array(dtype=wp.vec3),
    velocities: wp.array(dtype=wp.vec3),
    inv_masses: wp.array(dtype=wp.float32),
    dt: float,
):
    """Integrate positions after constraint projection.

    Computes new velocities from position change and updates positions.

    Args:
        positions: Current positions (updated in-place).
        predicted: Predicted/corrected positions.
        velocities: Velocities (updated in-place).
        inv_masses: Inverse masses (0 for fixed particles).
        dt: Time step size.
    """
    i = wp.tid()
    inv_mass = inv_masses[i]
    if inv_mass > 0.0:
        v = (predicted[i] - positions[i]) * (1.0 / dt)
        velocities[i] = v
        positions[i] = predicted[i]


@wp.kernel
def _warp_predict_rotations(
    orientations: wp.array(dtype=wp.quat),
    angular_velocities: wp.array(dtype=wp.vec3),
    torques: wp.array(dtype=wp.vec3),
    quat_inv_masses: wp.array(dtype=wp.float32),
    dt: float,
    damping: float,
    predicted: wp.array(dtype=wp.quat),
):
    """Predict particle orientations for constraint projection.

    Updates angular velocities with torques, applies damping,
    then computes predicted orientations using quaternion integration.

    Args:
        orientations: Current particle orientations.
        angular_velocities: Current angular velocities (updated in-place).
        torques: External torques on particles.
        quat_inv_masses: Inverse rotational masses (0 for locked particles).
        dt: Time step size.
        damping: Angular velocity damping factor.
        predicted: Output predicted orientations.
    """
    i = wp.tid()
    inv_mass = quat_inv_masses[i]
    if inv_mass > 0.0:
        half_dt = 0.5 * dt
        w = angular_velocities[i] + torques[i] * inv_mass * dt
        w = w * (1.0 - damping)
        angular_velocities[i] = w
        q = orientations[i]
        omega_q = wp.quat(w.x, w.y, w.z, 0.0)
        qdot = _warp_quat_mul(omega_q, q)
        q_pred = wp.quat(
            q.x + qdot.x * half_dt,
            q.y + qdot.y * half_dt,
            q.z + qdot.z * half_dt,
            q.w + qdot.w * half_dt,
        )
        predicted[i] = _warp_quat_normalize(q_pred)
    else:
        angular_velocities[i] = wp.vec3(0.0, 0.0, 0.0)
        predicted[i] = orientations[i]


@wp.kernel
def _warp_integrate_rotations(
    orientations: wp.array(dtype=wp.quat),
    predicted: wp.array(dtype=wp.quat),
    prev_orientations: wp.array(dtype=wp.quat),
    angular_velocities: wp.array(dtype=wp.vec3),
    quat_inv_masses: wp.array(dtype=wp.float32),
    dt: float,
):
    """Integrate orientations after constraint projection.

    Computes new angular velocities from orientation change and updates orientations.

    Args:
        orientations: Current orientations (updated in-place).
        predicted: Predicted/corrected orientations.
        prev_orientations: Previous orientations (updated in-place).
        angular_velocities: Angular velocities (updated in-place).
        quat_inv_masses: Inverse rotational masses (0 for locked particles).
        dt: Time step size.
    """
    i = wp.tid()
    if quat_inv_masses[i] > 0.0:
        q = orientations[i]
        rel = _warp_quat_mul(predicted[i], _warp_quat_conjugate(q))
        angular_velocities[i] = wp.vec3(rel.x, rel.y, rel.z) * (2.0 / dt)
        prev_orientations[i] = q
        orientations[i] = predicted[i]


@wp.kernel
def _warp_predict_positions_batched(
    positions: wp.array(dtype=wp.vec3),
    velocities: wp.array(dtype=wp.vec3),
    forces: wp.array(dtype=wp.vec3),
    inv_masses: wp.array(dtype=wp.float32),
    gravity: wp.array(dtype=wp.vec3),
    particle_rod_id: wp.array(dtype=wp.int32),
    dt: float,
    damping: float,
    predicted: wp.array(dtype=wp.vec3),
):
    """Predict particle positions for multiple rods in a single launch.

    This batched version processes all particles across all rods. Each particle
    looks up its rod ID to get the appropriate gravity vector.

    Args:
        positions: Concatenated current positions for all rods.
        velocities: Concatenated velocities (updated in-place).
        forces: Concatenated external forces.
        inv_masses: Concatenated inverse masses (0 for fixed particles).
        gravity: Per-rod gravity vectors [n_rods].
        particle_rod_id: Rod index for each particle.
        dt: Time step size.
        damping: Linear velocity damping factor.
        predicted: Output predicted positions.
    """
    i = wp.tid()
    inv_mass = inv_masses[i]
    if inv_mass > 0.0:
        rod_id = particle_rod_id[i]
        grav = gravity[rod_id]
        v = velocities[i] + (forces[i] * inv_mass + grav) * dt
        v = v * (1.0 - damping)
        velocities[i] = v
        predicted[i] = positions[i] + v * dt
    else:
        velocities[i] = wp.vec3(0.0, 0.0, 0.0)
        predicted[i] = positions[i]


@wp.kernel
def _warp_integrate_positions_batched(
    positions: wp.array(dtype=wp.vec3),
    predicted: wp.array(dtype=wp.vec3),
    velocities: wp.array(dtype=wp.vec3),
    inv_masses: wp.array(dtype=wp.float32),
    dt: float,
):
    """Integrate positions for multiple rods in a single launch.

    This batched version is identical to the single-rod version since the
    operation doesn't need rod-specific parameters.

    Args:
        positions: Concatenated positions (updated in-place).
        predicted: Concatenated predicted/corrected positions.
        velocities: Concatenated velocities (updated in-place).
        inv_masses: Concatenated inverse masses (0 for fixed particles).
        dt: Time step size.
    """
    i = wp.tid()
    inv_mass = inv_masses[i]
    if inv_mass > 0.0:
        v = (predicted[i] - positions[i]) * (1.0 / dt)
        velocities[i] = v
        positions[i] = predicted[i]


@wp.kernel
def _warp_predict_rotations_batched(
    orientations: wp.array(dtype=wp.quat),
    angular_velocities: wp.array(dtype=wp.vec3),
    torques: wp.array(dtype=wp.vec3),
    quat_inv_masses: wp.array(dtype=wp.float32),
    dt: float,
    damping: float,
    predicted: wp.array(dtype=wp.quat),
):
    """Predict rotations for multiple rods in a single launch.

    This batched version is identical to the single-rod version since the
    operation doesn't need rod-specific parameters.

    Args:
        orientations: Concatenated orientations.
        angular_velocities: Concatenated angular velocities (updated in-place).
        torques: Concatenated external torques.
        quat_inv_masses: Concatenated inverse rotational masses (0 for locked).
        dt: Time step size.
        damping: Angular velocity damping factor.
        predicted: Output predicted orientations.
    """
    i = wp.tid()
    inv_mass = quat_inv_masses[i]
    if inv_mass > 0.0:
        half_dt = 0.5 * dt
        w = angular_velocities[i] + torques[i] * inv_mass * dt
        w = w * (1.0 - damping)
        angular_velocities[i] = w
        q = orientations[i]
        omega_q = wp.quat(w.x, w.y, w.z, 0.0)
        qdot = _warp_quat_mul(omega_q, q)
        q_pred = wp.quat(
            q.x + qdot.x * half_dt,
            q.y + qdot.y * half_dt,
            q.z + qdot.z * half_dt,
            q.w + qdot.w * half_dt,
        )
        predicted[i] = _warp_quat_normalize(q_pred)
    else:
        angular_velocities[i] = wp.vec3(0.0, 0.0, 0.0)
        predicted[i] = orientations[i]


@wp.kernel
def _warp_integrate_rotations_batched(
    orientations: wp.array(dtype=wp.quat),
    predicted: wp.array(dtype=wp.quat),
    prev_orientations: wp.array(dtype=wp.quat),
    angular_velocities: wp.array(dtype=wp.vec3),
    quat_inv_masses: wp.array(dtype=wp.float32),
    dt: float,
):
    """Integrate rotations for multiple rods in a single launch.

    This batched version is identical to the single-rod version since the
    operation doesn't need rod-specific parameters.

    Args:
        orientations: Concatenated orientations (updated in-place).
        predicted: Concatenated predicted/corrected orientations.
        prev_orientations: Concatenated previous orientations (updated in-place).
        angular_velocities: Concatenated angular velocities (updated in-place).
        quat_inv_masses: Concatenated inverse rotational masses (0 for locked).
        dt: Time step size.
    """
    i = wp.tid()
    if quat_inv_masses[i] > 0.0:
        q = orientations[i]
        rel = _warp_quat_mul(predicted[i], _warp_quat_conjugate(q))
        angular_velocities[i] = wp.vec3(rel.x, rel.y, rel.z) * (2.0 / dt)
        prev_orientations[i] = q
        orientations[i] = predicted[i]


__all__ = [
    "_warp_integrate_positions",
    "_warp_integrate_positions_batched",
    "_warp_integrate_rotations",
    "_warp_integrate_rotations_batched",
    "_warp_predict_positions",
    "_warp_predict_positions_batched",
    "_warp_predict_rotations",
    "_warp_predict_rotations_batched",
]
