# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Base class and enums for constraint solvers."""

from abc import ABC, abstractmethod
from enum import IntEnum, auto
from typing import TYPE_CHECKING

import warp as wp

if TYPE_CHECKING:
    from newton.examples.cosserat2.cosserat_rod import CosseratRod


class ConstraintSolverType(IntEnum):
    """Enumeration of available constraint solver types."""

    JACOBI = 0  # Iterative Jacobi (from 02, 04, 05, 10_sim_aorta)
    THOMAS = auto()  # Thomas algorithm O(n) (from 01_thomas)
    CHOLESKY_SINGLE = auto()  # Tiled Cholesky 32x32 (from 07)
    CHOLESKY_MULTI = auto()  # Multi-tile Cholesky (from 08_multitile)
    LOCAL = auto()  # Local iterative solver with velocity update (from 02_local_cosserat_rod)


class FrictionMethod(IntEnum):
    """Enumeration of internal friction methods."""

    NONE = 0  # No internal friction
    VELOCITY_DAMPING = 1  # v *= coeff (simplest)
    STRAIN_RATE_DAMPING = 2  # f_damp = damping * k * d(kappa)/dt
    DAHL_HYSTERESIS = 3  # Path-dependent friction with memory


class ConstraintSolverBase(ABC):
    """Abstract base class for Cosserat rod constraint solvers.

    All constraint solvers implement the same interface for solving
    stretch/shear and bend/twist constraints. This allows the main
    solver to swap constraint solving methods at runtime.

    Args:
        rod: The CosseratRod data structure.
        device: Warp device to use.
    """

    def __init__(self, rod: "CosseratRod", device: str = "cuda:0"):
        self.rod = rod
        self.device = device
        self.num_particles = rod.num_particles
        self.num_stretch = rod.num_stretch
        self.num_bend = rod.num_bend

    @property
    @abstractmethod
    def solver_type(self) -> ConstraintSolverType:
        """Return the solver type enum value."""
        pass

    @property
    def name(self) -> str:
        """Return human-readable solver name."""
        names = {
            ConstraintSolverType.JACOBI: "Jacobi Iteration",
            ConstraintSolverType.THOMAS: "Thomas Algorithm",
            ConstraintSolverType.CHOLESKY_SINGLE: "Cholesky (Single Tile)",
            ConstraintSolverType.CHOLESKY_MULTI: "Cholesky (Multi-Tile)",
            ConstraintSolverType.LOCAL: "Local Iterative",
        }
        return names.get(self.solver_type, "Unknown")

    @abstractmethod
    def solve_stretch_shear(
        self,
        particle_q: wp.array,
        particle_q_out: wp.array,
        stretch_shear_stiffness: wp.vec3,
    ):
        """Solve stretch/shear constraints.

        Args:
            particle_q: Current particle positions [num_particles].
            particle_q_out: Output corrected positions [num_particles].
            stretch_shear_stiffness: Stiffness vector (shear_d1, shear_d2, stretch_d3).
        """
        pass

    @abstractmethod
    def solve_bend_twist(
        self,
        bend_twist_stiffness: wp.vec3,
        friction_method: FrictionMethod,
        friction_params: dict,
        dt: float,
    ):
        """Solve bend/twist constraints.

        Args:
            bend_twist_stiffness: Stiffness vector (bend_d1, twist, bend_d2).
            friction_method: Which friction model to use.
            friction_params: Parameters for the friction model.
            dt: Time step for friction calculations.
        """
        pass
