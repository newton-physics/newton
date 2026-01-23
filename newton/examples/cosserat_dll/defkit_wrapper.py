# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Python wrapper for DefKit/DefKitAdv DLL functions."""

import ctypes
import os
from pathlib import Path


class DefKitWrapper:
    """Wrapper for DefKit and DefKitAdv DLL functions.

    This wrapper provides Python bindings to the native C++ Cosserat rod
    simulation functions from the DefKit library.
    """

    def __init__(self, dll_path: str = "unity_ref"):
        """Load the DLL libraries.

        Args:
            dll_path: Path to directory containing DefKit.dll and DefKitAdv.dll.
                     Defaults to "unity_ref" (pre-compiled DLLs).
        """
        dll_dir = Path(dll_path).resolve()

        # Add DLL directory to PATH for MKL dependencies
        if dll_dir.exists():
            os.add_dll_directory(str(dll_dir))
            # Also add to PATH for older Python versions
            os.environ["PATH"] = str(dll_dir) + os.pathsep + os.environ.get("PATH", "")

        # Load DLLs
        defkit_path = dll_dir / "DefKit.dll"
        defkit_adv_path = dll_dir / "DefKitAdv.dll"

        if not defkit_path.exists():
            raise FileNotFoundError(f"DefKit.dll not found at {defkit_path}")
        if not defkit_adv_path.exists():
            raise FileNotFoundError(f"DefKitAdv.dll not found at {defkit_adv_path}")

        self.defkit = ctypes.CDLL(str(defkit_path))
        self.defkit_adv = ctypes.CDLL(str(defkit_adv_path))

        # Define function signatures
        self._setup_function_signatures()

    def _setup_function_signatures(self):
        """Set up ctypes function argument and return types."""

        # PredictPositions_native
        # void PredictPositions_native(float dt, float damping, int pointsCount,
        #     btVector3* positions, btVector3* predictedPositions,
        #     btVector3* velocities, btVector3* forces, float* invMasses, btVector3* gravity)
        self.defkit.PredictPositions_native.argtypes = [
            ctypes.c_float,  # dt
            ctypes.c_float,  # damping
            ctypes.c_int,  # pointsCount
            ctypes.POINTER(ctypes.c_float),  # positions (btVector3* = 4 floats each)
            ctypes.POINTER(ctypes.c_float),  # predictedPositions
            ctypes.POINTER(ctypes.c_float),  # velocities
            ctypes.POINTER(ctypes.c_float),  # forces
            ctypes.POINTER(ctypes.c_float),  # invMasses
            ctypes.POINTER(ctypes.c_float),  # gravity (btVector3*)
        ]
        self.defkit.PredictPositions_native.restype = None

        # PredictRotationsPBD
        # void PredictRotationsPBD(float dt, float damping, int pointsCount,
        #     btQuaternion* orientations, btQuaternion* predictedOrientations,
        #     btVector3* angVelocities, btVector3* torques, float* quatInvMass)
        self.defkit.PredictRotationsPBD.argtypes = [
            ctypes.c_float,  # dt
            ctypes.c_float,  # damping
            ctypes.c_int,  # pointsCount
            ctypes.POINTER(ctypes.c_float),  # orientations (btQuaternion* = 4 floats each)
            ctypes.POINTER(ctypes.c_float),  # predictedOrientations
            ctypes.POINTER(ctypes.c_float),  # angVelocities (btVector3* = 4 floats each)
            ctypes.POINTER(ctypes.c_float),  # torques
            ctypes.POINTER(ctypes.c_float),  # quatInvMass
        ]
        self.defkit.PredictRotationsPBD.restype = None

        # Integrate_native
        # void Integrate_native(float dt, int pointsCount,
        #     btVector3* positions, btVector3* predictedPositions,
        #     btVector3* velocities, float* invMasses)
        self.defkit.Integrate_native.argtypes = [
            ctypes.c_float,  # dt
            ctypes.c_int,  # pointsCount
            ctypes.POINTER(ctypes.c_float),  # positions
            ctypes.POINTER(ctypes.c_float),  # predictedPositions
            ctypes.POINTER(ctypes.c_float),  # velocities
            ctypes.POINTER(ctypes.c_float),  # invMasses
        ]
        self.defkit.Integrate_native.restype = None

        # IntegrateRotationsPBD
        # void IntegrateRotationsPBD(float dt, int pointsCount,
        #     btQuaternion* orientations, btQuaternion* predictedOrientations,
        #     btQuaternion* prevOrientations, btVector3* angVelocities, float* quatInvMass)
        self.defkit.IntegrateRotationsPBD.argtypes = [
            ctypes.c_float,  # dt
            ctypes.c_int,  # pointsCount
            ctypes.POINTER(ctypes.c_float),  # orientations
            ctypes.POINTER(ctypes.c_float),  # predictedOrientations
            ctypes.POINTER(ctypes.c_float),  # prevOrientations
            ctypes.POINTER(ctypes.c_float),  # angVelocities
            ctypes.POINTER(ctypes.c_float),  # quatInvMass
        ]
        self.defkit.IntegrateRotationsPBD.restype = None

        # ProjectElasticRodConstraints (from DefKitAdv)
        # void ProjectElasticRodConstraints(int pointsCount,
        #     btVector3* positions, btQuaternion* orientations,
        #     float* invMasses, float* quatInvMasses,
        #     btQuaternion* restDarboux, btVector3* bendAndTwistKs,
        #     float* restLength, float stretchKs, float shearKs)
        self.defkit_adv.ProjectElasticRodConstraints.argtypes = [
            ctypes.c_int,  # pointsCount
            ctypes.POINTER(ctypes.c_float),  # positions (btVector3*)
            ctypes.POINTER(ctypes.c_float),  # orientations (btQuaternion*)
            ctypes.POINTER(ctypes.c_float),  # invMasses
            ctypes.POINTER(ctypes.c_float),  # quatInvMasses
            ctypes.POINTER(ctypes.c_float),  # restDarboux (btQuaternion*)
            ctypes.POINTER(ctypes.c_float),  # bendAndTwistKs (btVector3*)
            ctypes.POINTER(ctypes.c_float),  # restLength
            ctypes.c_float,  # stretchKs
            ctypes.c_float,  # shearKs (actually bendAndTwistKs multiplier)
        ]
        self.defkit_adv.ProjectElasticRodConstraints.restype = None

        # ===== Direct Solver Functions =====

        # InitDirectElasticRod
        # DirectPositionBasedSolverForStiffRodsConstraint* InitDirectElasticRod(
        #     int pointsCount, btVector3* positions, btQuaternion* orientations,
        #     float radius, float* restLengths, float youngModulus, float torsionModulus)
        self.defkit_adv.InitDirectElasticRod.argtypes = [
            ctypes.c_int,  # pointsCount
            ctypes.POINTER(ctypes.c_float),  # positions (btVector3*)
            ctypes.POINTER(ctypes.c_float),  # orientations (btQuaternion*)
            ctypes.c_float,  # radius
            ctypes.POINTER(ctypes.c_float),  # restLengths
            ctypes.c_float,  # youngModulus
            ctypes.c_float,  # torsionModulus
        ]
        self.defkit_adv.InitDirectElasticRod.restype = ctypes.c_void_p

        # PrepareDirectElasticRodConstraints
        # void PrepareDirectElasticRodConstraints(void* rod, int pointsCount, float dt,
        #     btVector3* bendStiffness, btVector3* restDarboux, float* restLengths,
        #     float youngModulusMult, float torsionModulusMult)
        self.defkit_adv.PrepareDirectElasticRodConstraints.argtypes = [
            ctypes.c_void_p,  # rod
            ctypes.c_int,  # pointsCount (number of constraints = n_particles - 1)
            ctypes.c_float,  # dt
            ctypes.POINTER(ctypes.c_float),  # bendStiffness (btVector3*)
            ctypes.POINTER(ctypes.c_float),  # restDarboux (btVector3*)
            ctypes.POINTER(ctypes.c_float),  # restLengths
            ctypes.c_float,  # youngModulusMult
            ctypes.c_float,  # torsionModulusMult
        ]
        self.defkit_adv.PrepareDirectElasticRodConstraints.restype = None

        # UpdateConstraints_DirectElasticRodConstraintsBanded
        # void UpdateConstraints_DirectElasticRodConstraintsBanded(void* rod, int pointsCount,
        #     btVector3* positions, btQuaternion* orientations, float* invMasses)
        self.defkit_adv.UpdateConstraints_DirectElasticRodConstraintsBanded.argtypes = [
            ctypes.c_void_p,  # rod
            ctypes.c_int,  # pointsCount
            ctypes.POINTER(ctypes.c_float),  # positions (btVector3*)
            ctypes.POINTER(ctypes.c_float),  # orientations (btQuaternion*)
            ctypes.POINTER(ctypes.c_float),  # invMasses
        ]
        self.defkit_adv.UpdateConstraints_DirectElasticRodConstraintsBanded.restype = None

        # ComputeJacobians_DirectElasticRodConstraintsBanded
        # void ComputeJacobians_DirectElasticRodConstraintsBanded(void* rod, int startId, int count,
        #     btVector3* positions, btQuaternion* orientations, float* invMasses)
        self.defkit_adv.ComputeJacobians_DirectElasticRodConstraintsBanded.argtypes = [
            ctypes.c_void_p,  # rod
            ctypes.c_int,  # startId
            ctypes.c_int,  # count
            ctypes.POINTER(ctypes.c_float),  # positions
            ctypes.POINTER(ctypes.c_float),  # orientations
            ctypes.POINTER(ctypes.c_float),  # invMasses
        ]
        self.defkit_adv.ComputeJacobians_DirectElasticRodConstraintsBanded.restype = None

        # AssembleJMJT_DirectElasticRodConstraintsBanded
        # void AssembleJMJT_DirectElasticRodConstraintsBanded(void* rod, int startId, int count,
        #     btVector3* positions, btQuaternion* orientations, float* invMasses)
        self.defkit_adv.AssembleJMJT_DirectElasticRodConstraintsBanded.argtypes = [
            ctypes.c_void_p,  # rod
            ctypes.c_int,  # startId
            ctypes.c_int,  # count
            ctypes.POINTER(ctypes.c_float),  # positions
            ctypes.POINTER(ctypes.c_float),  # orientations
            ctypes.POINTER(ctypes.c_float),  # invMasses
        ]
        self.defkit_adv.AssembleJMJT_DirectElasticRodConstraintsBanded.restype = None

        # ProjectJMJT_DirectElasticRodConstraintsBanded
        # void ProjectJMJT_DirectElasticRodConstraintsBanded(void* rod, int pointsCount,
        #     btVector3* positions, btQuaternion* orientations, float* invMasses,
        #     btVector3* posCorr, btQuaternion* rotCorr)
        self.defkit_adv.ProjectJMJT_DirectElasticRodConstraintsBanded.argtypes = [
            ctypes.c_void_p,  # rod
            ctypes.c_int,  # pointsCount
            ctypes.POINTER(ctypes.c_float),  # positions
            ctypes.POINTER(ctypes.c_float),  # orientations
            ctypes.POINTER(ctypes.c_float),  # invMasses
            ctypes.POINTER(ctypes.c_float),  # posCorr (output)
            ctypes.POINTER(ctypes.c_float),  # rotCorr (output)
        ]
        self.defkit_adv.ProjectJMJT_DirectElasticRodConstraintsBanded.restype = None

        # DestroyDirectElasticRod
        # void DestroyDirectElasticRod(void* rod)
        self.defkit_adv.DestroyDirectElasticRod.argtypes = [ctypes.c_void_p]
        self.defkit_adv.DestroyDirectElasticRod.restype = None

    def predict_positions(self, dt, damping, positions, predicted_positions, velocities, forces, inv_masses, gravity):
        """Predict particle positions using semi-implicit Euler integration.

        Args:
            dt: Time step size
            damping: Velocity damping factor (0-1)
            positions: Current positions, shape (n, 4), float32
            predicted_positions: Output predicted positions, shape (n, 4), float32
            velocities: Particle velocities, shape (n, 4), float32
            forces: External forces, shape (n, 4), float32
            inv_masses: Inverse masses, shape (n,), float32
            gravity: Gravity vector, shape (4,), float32
        """
        n = len(positions)
        self.defkit.PredictPositions_native(
            ctypes.c_float(dt),
            ctypes.c_float(damping),
            ctypes.c_int(n),
            positions.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            predicted_positions.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            velocities.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            forces.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            inv_masses.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            gravity.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        )

    def predict_rotations(self, dt, damping, orientations, predicted_orientations, ang_velocities, torques, quat_inv_masses):
        """Predict quaternion orientations using angular velocity integration.

        Args:
            dt: Time step size
            damping: Angular velocity damping factor (0-1)
            orientations: Current orientations (x,y,z,w), shape (n, 4), float32
            predicted_orientations: Output predicted orientations, shape (n, 4), float32
            ang_velocities: Angular velocities, shape (n, 4), float32
            torques: External torques, shape (n, 4), float32
            quat_inv_masses: Inverse rotational masses, shape (n,), float32
        """
        n = len(orientations)
        self.defkit.PredictRotationsPBD(
            ctypes.c_float(dt),
            ctypes.c_float(damping),
            ctypes.c_int(n),
            orientations.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            predicted_orientations.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ang_velocities.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            torques.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            quat_inv_masses.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        )

    def integrate_positions(self, dt, positions, predicted_positions, velocities, inv_masses):
        """Integrate positions and update velocities after constraint projection.

        Args:
            dt: Time step size
            positions: Positions to update in-place, shape (n, 4), float32
            predicted_positions: Predicted (corrected) positions, shape (n, 4), float32
            velocities: Velocities to update in-place, shape (n, 4), float32
            inv_masses: Inverse masses, shape (n,), float32
        """
        n = len(positions)
        self.defkit.Integrate_native(
            ctypes.c_float(dt),
            ctypes.c_int(n),
            positions.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            predicted_positions.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            velocities.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            inv_masses.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        )

    def integrate_rotations(self, dt, orientations, predicted_orientations, prev_orientations, ang_velocities, quat_inv_masses):
        """Integrate orientations and update angular velocities after constraint projection.

        Args:
            dt: Time step size
            orientations: Orientations to update in-place, shape (n, 4), float32
            predicted_orientations: Predicted (corrected) orientations, shape (n, 4), float32
            prev_orientations: Previous orientations (updated in-place), shape (n, 4), float32
            ang_velocities: Angular velocities to update in-place, shape (n, 4), float32
            quat_inv_masses: Inverse rotational masses, shape (n,), float32
        """
        n = len(orientations)
        self.defkit.IntegrateRotationsPBD(
            ctypes.c_float(dt),
            ctypes.c_int(n),
            orientations.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            predicted_orientations.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            prev_orientations.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ang_velocities.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            quat_inv_masses.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        )

    def project_elastic_rod_constraints(
        self,
        positions,
        orientations,
        inv_masses,
        quat_inv_masses,
        rest_darboux,
        bend_twist_ks,
        rest_lengths,
        stretch_ks,
        bend_twist_ks_mult,
    ):
        """Project stretch-shear and bend-twist constraints for an elastic rod.

        This implements the iterative Position and Orientation Based Cosserat Rods
        constraint projection from Kugelstadt et al.

        Args:
            positions: Particle positions (predicted), shape (n, 4), float32
            orientations: Quaternion orientations (predicted), shape (n, 4), float32
            inv_masses: Inverse masses, shape (n,), float32
            quat_inv_masses: Inverse rotational masses, shape (n,), float32
            rest_darboux: Rest Darboux vectors as quaternions, shape (n-1, 4), float32
            bend_twist_ks: Bending/twisting stiffness, shape (n-1, 4), float32
            rest_lengths: Rest lengths per edge, shape (n-1,), float32
            stretch_ks: Stretch stiffness coefficient (scalar)
            bend_twist_ks_mult: Bend/twist stiffness multiplier (scalar)
        """
        n = len(positions)
        self.defkit_adv.ProjectElasticRodConstraints(
            ctypes.c_int(n),
            positions.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            orientations.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            inv_masses.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            quat_inv_masses.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            rest_darboux.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            bend_twist_ks.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            rest_lengths.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.c_float(stretch_ks),
            ctypes.c_float(bend_twist_ks_mult),
        )

    # ===== Direct Solver Methods =====

    def init_direct_elastic_rod(
        self,
        positions,
        orientations,
        radius,
        rest_lengths,
        young_modulus,
        torsion_modulus,
    ):
        """Initialize the direct elastic rod solver.

        This creates internal data structures for the direct (global) solver
        including the banded matrix system.

        Args:
            positions: Initial positions, shape (n, 4), float32
            orientations: Initial orientations (x,y,z,w), shape (n, 4), float32
            radius: Rod cross-section radius
            rest_lengths: Rest lengths per edge, shape (n-1,), float32
            young_modulus: Young's modulus (bending stiffness)
            torsion_modulus: Torsion modulus (twist stiffness)

        Returns:
            Opaque pointer to the rod constraint object (store this for later calls)
        """
        n = len(positions)
        rod_ptr = self.defkit_adv.InitDirectElasticRod(
            ctypes.c_int(n),
            positions.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            orientations.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.c_float(radius),
            rest_lengths.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.c_float(young_modulus),
            ctypes.c_float(torsion_modulus),
        )
        return rod_ptr

    def prepare_direct_elastic_rod_constraints(
        self,
        rod_ptr,
        n_constraints,
        dt,
        bend_stiffness,
        rest_darboux,
        rest_lengths,
        young_modulus_mult,
        torsion_modulus_mult,
    ):
        """Prepare direct solver for a new timestep.

        Resets lambda accumulators and updates stiffness/rest shape parameters.

        Args:
            rod_ptr: Pointer from init_direct_elastic_rod
            n_constraints: Number of constraints (n_particles - 1)
            dt: Time step size
            bend_stiffness: Bending stiffness per constraint, shape (n_constraints, 4), float32
                           (x=kappa1_stiffness, y=kappa2_stiffness, z=tau_stiffness)
            rest_darboux: Rest Darboux vector per constraint, shape (n_constraints, 4), float32
                         (x=kappa1, y=kappa2, z=tau) - NOTE: Vector3, not quaternion!
            rest_lengths: Rest lengths per constraint, shape (n_constraints,), float32
            young_modulus_mult: Multiplier for Young's modulus
            torsion_modulus_mult: Multiplier for torsion modulus
        """
        self.defkit_adv.PrepareDirectElasticRodConstraints(
            ctypes.c_void_p(rod_ptr),
            ctypes.c_int(n_constraints),
            ctypes.c_float(dt),
            bend_stiffness.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            rest_darboux.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            rest_lengths.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.c_float(young_modulus_mult),
            ctypes.c_float(torsion_modulus_mult),
        )

    def update_direct_constraints(self, rod_ptr, positions, orientations, inv_masses):
        """Update constraint state with current positions and orientations.

        Args:
            rod_ptr: Pointer from init_direct_elastic_rod
            positions: Current (predicted) positions, shape (n, 4), float32
            orientations: Current (predicted) orientations, shape (n, 4), float32
            inv_masses: Inverse masses, shape (n,), float32
        """
        n = len(positions)
        self.defkit_adv.UpdateConstraints_DirectElasticRodConstraintsBanded(
            ctypes.c_void_p(rod_ptr),
            ctypes.c_int(n),
            positions.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            orientations.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            inv_masses.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        )

    def compute_jacobians_direct(self, rod_ptr, start_id, count, positions, orientations, inv_masses):
        """Compute Jacobian matrices for a range of constraints.

        Args:
            rod_ptr: Pointer from init_direct_elastic_rod
            start_id: Starting constraint index
            count: Number of constraints to process
            positions: Current positions, shape (n, 4), float32
            orientations: Current orientations, shape (n, 4), float32
            inv_masses: Inverse masses, shape (n,), float32
        """
        self.defkit_adv.ComputeJacobians_DirectElasticRodConstraintsBanded(
            ctypes.c_void_p(rod_ptr),
            ctypes.c_int(start_id),
            ctypes.c_int(count),
            positions.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            orientations.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            inv_masses.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        )

    def assemble_jmjt_direct(self, rod_ptr, start_id, count, positions, orientations, inv_masses):
        """Assemble the JMJT banded matrix for a range of constraints.

        Args:
            rod_ptr: Pointer from init_direct_elastic_rod
            start_id: Starting constraint index
            count: Number of constraints to process
            positions: Current positions, shape (n, 4), float32
            orientations: Current orientations, shape (n, 4), float32
            inv_masses: Inverse masses, shape (n,), float32
        """
        self.defkit_adv.AssembleJMJT_DirectElasticRodConstraintsBanded(
            ctypes.c_void_p(rod_ptr),
            ctypes.c_int(start_id),
            ctypes.c_int(count),
            positions.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            orientations.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            inv_masses.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        )

    def solve_direct_constraints(self, rod_ptr, positions, orientations, inv_masses):
        """Solve the constraint system and apply corrections.

        This solves the banded matrix system and directly updates positions
        and orientations in-place.

        Args:
            rod_ptr: Pointer from init_direct_elastic_rod
            positions: Positions to update in-place, shape (n, 4), float32
            orientations: Orientations to update in-place, shape (n, 4), float32
            inv_masses: Inverse masses, shape (n,), float32
        """
        import numpy as np

        n = len(positions)
        # Allocate output arrays for corrections (not used but required by API)
        pos_corr = np.zeros((n, 4), dtype=np.float32)
        rot_corr = np.zeros((n, 4), dtype=np.float32)

        self.defkit_adv.ProjectJMJT_DirectElasticRodConstraintsBanded(
            ctypes.c_void_p(rod_ptr),
            ctypes.c_int(n),
            positions.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            orientations.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            inv_masses.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            pos_corr.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            rot_corr.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        )

    def destroy_direct_elastic_rod(self, rod_ptr):
        """Destroy the direct elastic rod solver and free memory.

        Args:
            rod_ptr: Pointer from init_direct_elastic_rod
        """
        if rod_ptr is not None:
            self.defkit_adv.DestroyDirectElasticRod(ctypes.c_void_p(rod_ptr))
