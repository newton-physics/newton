# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Cosserat rod solver with pluggable backends.

This module provides the main CosseratSolver class that orchestrates the
simulation loop, delegating actual computation to pluggable backend
implementations.
"""

from enum import IntEnum, auto
from typing import TYPE_CHECKING, Optional

import numpy as np

if TYPE_CHECKING:
    from .backends.base import BackendBase
    from .model import CosseratRodModel


class BackendType(IntEnum):
    """Enumeration of available solver backends.

    Each backend implements the same simulation algorithm with different
    underlying technologies:

    REFERENCE: C/C++ DLL backend (DefKit). Uses native compiled code for
               maximum performance. Requires the DLL files to be available.

    NUMPY: Pure Python/NumPy implementation. Good for debugging and
           understanding the algorithm. Not optimized for performance.

    WARP_CPU: Warp kernels running on CPU. Useful for debugging Warp code
              without GPU overhead. Uses numpy-backed arrays.

    WARP_GPU: Warp kernels running on GPU. Full GPU acceleration with optional
              CUDA graph capture for maximum throughput.
    """

    REFERENCE = 0  # C/C++ DLL (DefKit)
    NUMPY = auto()  # Pure NumPy
    WARP_CPU = auto()  # Warp on CPU (debugging)
    WARP_GPU = auto()  # Warp on GPU with CUDA graph


class CosseratSolver:
    """Main solver for Cosserat rod simulation.

    This class provides a unified interface to simulate Cosserat rods using
    different computational backends. The simulation loop is:

    1. Predict positions (semi-implicit Euler)
    2. Predict rotations (quaternion integration)
    3. Prepare constraints (compute compliance)
    4. Update constraint values (compute violations)
    5. Compute Jacobians
    6. Assemble JMJT matrix
    7. Solve linear system and apply corrections
    8. Integrate positions (derive velocities)
    9. Integrate rotations (derive angular velocities)

    Example:
        model = create_straight_rod(n_particles=20, ...)
        solver = CosseratSolver(model, backend=BackendType.WARP_GPU)
        for _ in range(100):
            solver.step(dt=1/240)
            positions = model.get_positions_3d()
    """

    def __init__(
        self,
        model: "CosseratRodModel",
        backend: BackendType = BackendType.NUMPY,
        device: str = "cuda:0",
        use_cuda_graph: bool = False,
        dll_path: str = "unity_ref",
    ):
        """Initialize the solver.

        Args:
            model: The rod model to simulate.
            backend: Which backend implementation to use.
            device: Device string for Warp backends (e.g., "cuda:0", "cpu").
            use_cuda_graph: Whether to use CUDA graph capture (WARP_GPU only).
            dll_path: Path to DefKit DLLs (REFERENCE backend only).
        """
        self.model = model
        self.backend_type = backend
        self.device = device
        self.use_cuda_graph = use_cuda_graph
        self.dll_path = dll_path

        # Create the backend instance
        self._backend: Optional["BackendBase"] = None
        self._create_backend()

        # Statistics
        self.step_count = 0
        self.last_step_time = 0.0

    def _create_backend(self):
        """Create the backend instance based on backend_type."""
        from .backends import create_backend

        self._backend = create_backend(
            backend_type=self.backend_type,
            model=self.model,
            device=self.device,
            use_cuda_graph=self.use_cuda_graph,
            dll_path=self.dll_path,
        )

    @property
    def backend(self) -> "BackendBase":
        """Get the current backend instance."""
        if self._backend is None:
            raise RuntimeError("Backend not initialized")
        return self._backend

    def set_backend(self, backend: BackendType):
        """Switch to a different backend at runtime.

        This will recreate the backend instance. Current simulation state
        is preserved in the model.

        Args:
            backend: New backend type to use.
        """
        if backend == self.backend_type:
            return

        self.backend_type = backend
        self._create_backend()

    def step(self, dt: float):
        """Advance simulation by one timestep.

        Args:
            dt: Time step size in seconds.
        """
        import time

        start = time.perf_counter()

        # Delegate to backend
        self.backend.step(dt)

        self.last_step_time = time.perf_counter() - start
        self.step_count += 1

    def step_substeps(self, dt: float, substeps: int = 4):
        """Advance simulation with multiple substeps.

        Args:
            dt: Total time step size.
            substeps: Number of substeps to take.
        """
        sub_dt = dt / substeps
        for _ in range(substeps):
            self.step(sub_dt)

    # =========================================================================
    # Configuration
    # =========================================================================

    def set_gravity(self, gx: float, gy: float, gz: float):
        """Set gravity vector.

        Args:
            gx, gy, gz: Gravity components.
        """
        self.model.config.gravity[:] = [gx, gy, gz]
        self.backend.on_gravity_changed()

    def set_damping(self, position_damping: float, rotation_damping: float):
        """Set velocity damping factors.

        Args:
            position_damping: Linear velocity damping (0-1).
            rotation_damping: Angular velocity damping (0-1).
        """
        self.model.config.position_damping = position_damping
        self.model.config.rotation_damping = rotation_damping

    def set_material(
        self,
        young_modulus: Optional[float] = None,
        torsion_modulus: Optional[float] = None,
        radius: Optional[float] = None,
    ):
        """Update material properties.

        Args:
            young_modulus: Young's modulus (Pa).
            torsion_modulus: Torsion modulus (Pa).
            radius: Cross-section radius (m).
        """
        if young_modulus is not None:
            self.model.material.young_modulus = young_modulus
        if torsion_modulus is not None:
            self.model.material.torsion_modulus = torsion_modulus
        if radius is not None:
            self.model.material.radius = radius
        self.backend.on_material_changed()

    def set_rest_curvature(self, kappa1: float, kappa2: float, tau: float):
        """Set uniform rest curvature for the rod.

        Args:
            kappa1: First bending curvature.
            kappa2: Second bending curvature.
            tau: Twist.
        """
        self.model.set_rest_curvature(kappa1, kappa2, tau)
        self.backend.on_rest_shape_changed()

    def set_bend_stiffness(self, k1: float, k2: float, k_tau: float):
        """Set uniform bending/twist stiffness.

        Args:
            k1: First bending stiffness.
            k2: Second bending stiffness.
            k_tau: Twist stiffness.
        """
        self.model.set_bend_stiffness_uniform(k1, k2, k_tau)
        self.backend.on_stiffness_changed()

    # =========================================================================
    # State access
    # =========================================================================

    def get_positions(self) -> np.ndarray:
        """Get current positions as (n, 3) array."""
        return self.model.get_positions_3d()

    def get_tip_position(self) -> np.ndarray:
        """Get tip (last particle) position."""
        return self.model.get_tip_position()

    def reset(self):
        """Reset simulation to initial state (zero velocities)."""
        self.model.reset_to_initial()
        self.backend.on_reset()
        self.step_count = 0

    # =========================================================================
    # Diagnostics
    # =========================================================================

    def get_backend_name(self) -> str:
        """Get human-readable name of current backend."""
        return self.backend.name

    def get_stats(self) -> dict:
        """Get solver statistics.

        Returns:
            Dictionary with step count, timing info, etc.
        """
        return {
            "backend": self.get_backend_name(),
            "step_count": self.step_count,
            "last_step_time_ms": self.last_step_time * 1000,
            "n_particles": self.model.n_particles,
            "n_edges": self.model.n_edges,
        }
