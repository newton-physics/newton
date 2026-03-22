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

"""Builder helper for adding Cosserat elastic rods to a Newton model."""

from __future__ import annotations

import math

import numpy as np

from ...sim import ModelBuilder


def _quat_from_axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    """Create a quaternion [x, y, z, w] from an axis-angle."""
    axis = np.asarray(axis, dtype=np.float32)
    norm = np.linalg.norm(axis)
    if norm < 1.0e-8:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    axis = axis / norm
    half = angle * 0.5
    s = math.sin(half)
    return np.array([axis[0] * s, axis[1] * s, axis[2] * s, math.cos(half)], dtype=np.float32)


def add_elastic_rod(
    builder: ModelBuilder,
    positions: np.ndarray,
    radius: float = 0.01,
    particle_mass: float = 0.1,
    bend_stiffness: float = 1.0,
    twist_stiffness: float = 1.0,
    young_modulus: float = 1.0e6,
    torsion_modulus: float = 1.0e6,
    lock_root: bool = True,
    lock_root_rotation: bool = True,
) -> list[int]:
    """Add a Cosserat elastic rod to the model.

    Adds particles and stores rod metadata in the ``xpbd_rod`` custom
    namespace. The solver reads this data at construction time to build
    internal GPU arrays.

    Args:
        builder: Model builder to add the rod to.
        positions: Initial positions as ``(N, 3)`` array [m].
        radius: Rod cross-section radius [m].
        particle_mass: Mass of each particle [kg].
        bend_stiffness: Bending stiffness coefficient.
        twist_stiffness: Twist stiffness coefficient.
        young_modulus: Young's modulus [Pa].
        torsion_modulus: Torsion modulus [Pa].
        lock_root: Whether the first particle is position-locked.
        lock_root_rotation: Whether the first particle is rotation-locked.

    Returns:
        List of particle indices.
    """
    positions = np.asarray(positions, dtype=np.float32)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("positions must be (N, 3)")

    num_points = positions.shape[0]
    if num_points < 2:
        raise ValueError("Rod requires at least 2 points")

    num_edges = num_points - 1

    # Compute rest lengths from positions
    rest_lengths = np.linalg.norm(np.diff(positions, axis=0), axis=1).astype(np.float32)

    # Compute initial orientations: align local Z-axis along each segment
    # Default orientation: Z-axis = (0,0,1). Rotate to align with segment direction.
    q_align = _quat_from_axis_angle(np.array([0.0, 1.0, 0.0]), math.pi * 0.5)
    orientations = np.tile(q_align, (num_points, 1)).astype(np.float32)

    # Compute initial rest Darboux vectors (zero for straight rods)
    rest_darboux = np.zeros((num_edges, 3), dtype=np.float32)

    # Compute edge bend/twist stiffness vectors
    bend_stiffness_vec = np.zeros((num_edges, 3), dtype=np.float32)
    bend_stiffness_vec[:, 0] = bend_stiffness
    bend_stiffness_vec[:, 1] = bend_stiffness
    bend_stiffness_vec[:, 2] = twist_stiffness

    # Inverse masses
    inv_mass = 0.0 if particle_mass == 0.0 else 1.0 / particle_mass
    inv_masses = np.full(num_points, inv_mass, dtype=np.float32)
    if lock_root:
        inv_masses[0] = 0.0

    # Quaternion inverse masses (rotation lock)
    quat_inv_masses = np.ones(num_points, dtype=np.float32)
    if lock_root:
        quat_inv_masses[0] = 0.0
    if lock_root_rotation:
        quat_inv_masses[0] = 0.0

    # Add particles
    particle_indices = []
    for i in range(num_points):
        mass = 0.0 if inv_masses[i] == 0.0 else particle_mass
        idx = builder.add_particle(
            pos=tuple(positions[i]),
            vel=(0.0, 0.0, 0.0),
            mass=mass,
            radius=radius,
        )
        particle_indices.append(idx)

    # Store rod data in xpbd_rod namespace lists
    ns = builder._xpbd_rod_data

    ns["rod_num_points"].append(num_points)
    ns["rod_particle_start"].append(particle_indices[0])
    ns["rod_young_modulus"].append(young_modulus)
    ns["rod_torsion_modulus"].append(torsion_modulus)

    ns["orientations"].extend(orientations.tolist())
    ns["quat_inv_masses"].extend(quat_inv_masses.tolist())
    ns["rest_lengths"].extend(rest_lengths.tolist())
    ns["rest_darboux"].extend(rest_darboux.tolist())
    ns["bend_stiffness"].extend(bend_stiffness_vec.tolist())

    return particle_indices
