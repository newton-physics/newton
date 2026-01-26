# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Reference C/C++ DLL backend using DefKit.

This backend uses the native DefKit library for maximum performance and
serves as the reference implementation for validation.
"""

from typing import TYPE_CHECKING

import numpy as np

from .base import BackendBase

if TYPE_CHECKING:
    from ..model import CosseratRodModel


class ReferenceBackend(BackendBase):
    """C/C++ DLL backend using DefKit library.

    This backend wraps the DefKit and DefKitAdv DLLs to provide native
    compiled performance. It serves as the reference implementation that
    other backends validate against.

    The DLL uses the direct solver approach with banded Cholesky decomposition.
    """

    def __init__(self, model: "CosseratRodModel", dll_path: str = "unity_ref"):
        """Initialize DLL backend.

        Args:
            model: The rod model to operate on.
            dll_path: Path to directory containing DefKit DLLs.

        Raises:
            FileNotFoundError: If DLL files are not found.
        """
        super().__init__(model)

        # Import wrapper lazily to avoid dependency if not using this backend
        from newton.examples.cosserat_dll.defkit_wrapper import DefKitWrapper

        self.dll = DefKitWrapper(dll_path)

        # DLL requires 4-component arrays with padding for SIMD alignment
        # Model already stores data in this format

        # Rest Darboux in 4-component format for DLL
        self._rest_darboux_4 = np.zeros((model.n_edges, 4), dtype=np.float32)
        self._bend_stiffness_4 = np.ones((model.n_edges, 4), dtype=np.float32)
        self._gravity_4 = np.zeros(4, dtype=np.float32)

        # Sync from model
        self._sync_parameters()

        # Initialize the direct solver
        self._rod_ptr = self.dll.init_direct_elastic_rod(
            model.positions,
            model.orientations,
            model.material.radius,
            model.rest_lengths,
            model.material.young_modulus,
            model.material.torsion_modulus,
        )

    def __del__(self):
        """Clean up native resources."""
        if hasattr(self, "_rod_ptr") and self._rod_ptr is not None:
            self.dll.destroy_direct_elastic_rod(self._rod_ptr)
            self._rod_ptr = None

    @property
    def name(self) -> str:
        return "Reference (C++ DLL)"

    def _sync_parameters(self):
        """Sync parameters from model to DLL-compatible format."""
        m = self.model

        # Convert rest Darboux (n_edges, 3) to (n_edges, 4)
        self._rest_darboux_4[:, :3] = m.rest_darboux
        self._rest_darboux_4[:, 3] = 0.0

        # Convert bend stiffness (n_edges, 3) to (n_edges, 4)
        self._bend_stiffness_4[:, :3] = m.bend_stiffness
        self._bend_stiffness_4[:, 3] = 0.0

        # Gravity vector
        self._gravity_4[:3] = m.config.gravity
        self._gravity_4[3] = 0.0

    def step(self, dt: float):
        """Advance simulation by one timestep using DLL."""
        m = self.model
        n_constraints = m.n_edges

        # 1. Predict positions
        self.dll.predict_positions(
            dt,
            m.config.position_damping,
            m.positions,
            m.predicted_positions,
            m.velocities,
            m.forces,
            m.inv_masses,
            self._gravity_4,
        )

        # 2. Predict rotations
        self.dll.predict_rotations(
            dt,
            m.config.rotation_damping,
            m.orientations,
            m.predicted_orientations,
            m.angular_velocities,
            m.torques,
            m.quat_inv_masses,
        )

        # 3. Prepare constraints
        self.dll.prepare_direct_elastic_rod_constraints(
            self._rod_ptr,
            n_constraints,
            dt,
            self._bend_stiffness_4,
            self._rest_darboux_4,
            m.rest_lengths,
            m.material.young_modulus,
            m.material.torsion_modulus,
        )

        # 4. Update constraint state
        self.dll.update_direct_constraints(
            self._rod_ptr,
            m.predicted_positions,
            m.predicted_orientations,
            m.inv_masses,
        )

        # 5. Compute Jacobians
        self.dll.compute_jacobians_direct(
            self._rod_ptr,
            0,
            n_constraints,
            m.predicted_positions,
            m.predicted_orientations,
            m.inv_masses,
        )

        # 6. Assemble JMJT
        self.dll.assemble_jmjt_direct(
            self._rod_ptr,
            0,
            n_constraints,
            m.predicted_positions,
            m.predicted_orientations,
            m.inv_masses,
        )

        # 7. Solve and apply corrections
        self.dll.solve_direct_constraints(
            self._rod_ptr,
            m.predicted_positions,
            m.predicted_orientations,
            m.inv_masses,
        )

        # 8. Integrate positions
        self.dll.integrate_positions(
            dt,
            m.positions,
            m.predicted_positions,
            m.velocities,
            m.inv_masses,
        )

        # 9. Integrate rotations
        self.dll.integrate_rotations(
            dt,
            m.orientations,
            m.predicted_orientations,
            m.prev_orientations,
            m.angular_velocities,
            m.quat_inv_masses,
        )

        # 10. Clear forces
        m.clear_forces()

    # =========================================================================
    # State change notifications
    # =========================================================================

    def on_gravity_changed(self):
        self._gravity_4[:3] = self.model.config.gravity

    def on_material_changed(self):
        # DLL doesn't need notification - uses model values directly
        pass

    def on_rest_shape_changed(self):
        self._rest_darboux_4[:, :3] = self.model.rest_darboux

    def on_stiffness_changed(self):
        self._bend_stiffness_4[:, :3] = self.model.bend_stiffness
