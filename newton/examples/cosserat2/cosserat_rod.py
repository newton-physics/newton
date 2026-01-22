# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Cosserat rod data structure for XPBD simulation.

Encapsulates rod-specific state separate from Newton's Model/State.
"""

from dataclasses import dataclass
from typing import Optional

import warp as wp


@dataclass
class FrictionState:
    """State for internal friction models.

    Used by strain-rate damping and Dahl hysteresis friction.
    """

    kappa_prev: wp.array  # Previous curvature values [num_bend]
    kappa_current: wp.array  # Current curvature values [num_bend]
    sigma_prev: wp.array  # Previous friction stress (Dahl) [num_bend]
    sigma_current: wp.array  # Current friction stress (Dahl) [num_bend]
    dkappa_prev: wp.array  # Previous curvature change direction (Dahl) [num_bend]
    dkappa_current: wp.array  # Current curvature change direction (Dahl) [num_bend]


class CosseratRod:
    """Cosserat rod data structure for XPBD simulation.

    Encapsulates rod-specific state including:
    - Geometry (inverse masses, rest lengths)
    - Orientation state (edge quaternions, rest Darboux vectors)
    - Correction buffers for Jacobi-style iteration
    - Friction state for internal friction models

    Args:
        num_particles: Number of particles in the rod.
        particle_inv_mass: Per-particle inverse mass (0 = kinematic).
        edge_inv_mass: Per-edge inverse mass (0 = kinematic).
        rest_length: Rest length per edge.
        edge_q_init: Initial edge quaternions.
        rest_darboux_init: Initial rest Darboux vectors.
        device: Warp device to use.
    """

    def __init__(
        self,
        num_particles: int,
        particle_inv_mass: wp.array,
        edge_inv_mass: wp.array,
        rest_length: wp.array,
        edge_q_init: wp.array,
        rest_darboux_init: wp.array,
        device: str = "cuda:0",
    ):
        self.num_particles = num_particles
        self.num_stretch = num_particles - 1
        self.num_bend = num_particles - 2
        self.device = device

        # Geometry (read-only after initialization)
        self.particle_inv_mass = particle_inv_mass
        self.edge_inv_mass = edge_inv_mass
        self.rest_length = rest_length

        # Orientation state
        self.edge_q = edge_q_init
        self.edge_q_new = wp.zeros_like(edge_q_init)
        self.rest_darboux = rest_darboux_init

        # Correction buffers for Jacobi-style iteration
        self.particle_delta = wp.zeros(num_particles, dtype=wp.vec3, device=device)
        self.edge_q_delta = wp.zeros(self.num_stretch, dtype=wp.quat, device=device)

        # Friction state
        self.friction_state = FrictionState(
            kappa_prev=wp.zeros(max(1, self.num_bend), dtype=wp.vec3, device=device),
            kappa_current=wp.zeros(max(1, self.num_bend), dtype=wp.vec3, device=device),
            sigma_prev=wp.zeros(max(1, self.num_bend), dtype=wp.vec3, device=device),
            sigma_current=wp.zeros(max(1, self.num_bend), dtype=wp.vec3, device=device),
            dkappa_prev=wp.zeros(max(1, self.num_bend), dtype=wp.vec3, device=device),
            dkappa_current=wp.zeros(max(1, self.num_bend), dtype=wp.vec3, device=device),
        )

    def swap_edge_q(self):
        """Swap edge_q and edge_q_new buffers."""
        self.edge_q, self.edge_q_new = self.edge_q_new, self.edge_q

    def update_friction_state_strain_rate(self):
        """Update friction state for strain-rate damping."""
        wp.copy(self.friction_state.kappa_prev, self.friction_state.kappa_current)

    def update_friction_state_dahl(self):
        """Update friction state for Dahl hysteresis."""
        wp.copy(self.friction_state.kappa_prev, self.friction_state.kappa_current)
        wp.copy(self.friction_state.sigma_prev, self.friction_state.sigma_current)
        wp.copy(self.friction_state.dkappa_prev, self.friction_state.dkappa_current)
