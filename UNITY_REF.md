# Unity Reference Implementation Integration Plan

This document outlines the plan to create a Python application using Newton as a viewer that invokes `DefKitAdv.dll` to simulate Cosserat rods based on the Unity reference implementation.

## 1. Overview

### Goal
Create a Python application that:
1. Uses the `DefKitAdv.dll` native library to simulate Cosserat rods
2. Uses Newton's viewer infrastructure for visualization
3. Starts with the **iterative Position and Orientation Based Cosserat Rods** solver

### Reference Files
- **C++ Implementation**: `unity_ref/Native/ElasticRod.cpp` - iterative solver
- **C# Wrapper**: `unity_ref/DefKitElasticRodSystem.cs` - simulation orchestration
- **Data Structures**: `unity_ref/Body.cs`, `unity_ref/ElasticRod.cs`
- **Newton Example Pattern**: `newton/examples/cosserat2/example_simple_direct_rod.py`

---

## 2. DLL Interface Analysis

### 2.1 Required DLL Exports

From `DefKit.dll`:
```c
// Position prediction and integration
void PredictPositions_native(float dt, float damping, int pointsCount,
    btVector3* positions, btVector3* predictedPositions,
    btVector3* velocities, btVector3* forces, float* invMasses, btVector3* gravity);

void Integrate_native(float dt, int pointsCount,
    btVector3* positions, btVector3* predictedPositions,
    btVector3* velocities, float* invMasses);

// Rotation prediction and integration
void PredictRotationsPBD(float dt, float damping, int pointsCount,
    btQuaternion* orientations, btQuaternion* predictedOrientations,
    btVector3* angVelocities, btVector3* torques, float* quatInvMass);

void IntegrateRotationsPBD(float dt, int pointsCount,
    btQuaternion* orientations, btQuaternion* predictedOrientations,
    btQuaternion* prevOrientations, btVector3* angVelocities, float* quatInvMass);
```

From `DefKitAdv.dll`:
```c
// Iterative rod constraint projection
void ProjectElasticRodConstraints(int pointsCount,
    btVector3* positions, btQuaternion* orientations,
    float* invMasses, float* quatInvMasses,
    btQuaternion* restDarboux, btVector3* bendAndTwistKs,
    float* restLength, float stretchKs, float shearKs);
```

### 2.2 Data Types Mapping

| C++ Type | Size | Python ctypes | NumPy dtype |
|----------|------|---------------|-------------|
| `btVector3` | 16 bytes (4 floats, SIMD aligned) | `c_float * 4` | `np.float32` (4,) |
| `btQuaternion` | 16 bytes (x, y, z, w) | `c_float * 4` | `np.float32` (4,) |
| `float` | 4 bytes | `c_float` | `np.float32` |
| `int` | 4 bytes | `c_int` | `np.int32` |

**Note**: `btVector3` uses 4 floats for SIMD alignment (x, y, z, w=0), same as Unity's `Vector4`.

---

## 3. Simulation Loop

Based on `DefKitElasticRodSystem.cs`, the simulation loop is:

```
Per substep:
    1. OnSubStepStart:
       - PredictPositions_native(dt, damping, positions, predictedPositions, velocities, forces, invMasses, gravity)
       - PredictRotationsPBD(dt, rotDamping, orientations, predictedOrientations, angVelocities, torques, quatInvMasses)

    2. OnConstraintsIterationStart (for each iteration):
       - ProjectElasticRodConstraints(pointsCount, predictedPositions, predictedOrientations,
           invMasses, quatInvMasses, restDarboux, bendAndTwistKs, restLengths, stretchKs, shearKs)

    3. OnSubStepEnd:
       - Integrate_native(dt, positions, predictedPositions, velocities, invMasses)
       - IntegrateRotationsPBD(dt, orientations, predictedOrientations, prevOrientations, angVelocities, quatInvMasses)
```

---

## 4. Implementation Plan

### Phase 1: DLL Wrapper Module

Create `newton/examples/cosserat_dll/defkit_wrapper.py`:

```python
"""Python wrapper for DefKit/DefKitAdv DLL functions."""

import ctypes
import numpy as np
from pathlib import Path


class DefKitWrapper:
    """Wrapper for DefKit and DefKitAdv DLL functions."""

    def __init__(self, dll_path: str = "unity_ref"):
        """Load the DLL libraries.

        Args:
            dll_path: Path to directory containing DefKit.dll and DefKitAdv.dll.
                     Defaults to "unity_ref" (pre-compiled DLLs).
        """
        # Load DLLs
        self.defkit = ctypes.CDLL(str(Path(dll_path) / "DefKit.dll"))
        self.defkit_adv = ctypes.CDLL(str(Path(dll_path) / "DefKitAdv.dll"))

        # Define function signatures
        self._setup_function_signatures()

    def _setup_function_signatures(self):
        """Set up ctypes function argument and return types."""

        # PredictPositions_native
        self.defkit.PredictPositions_native.argtypes = [
            ctypes.c_float,  # dt
            ctypes.c_float,  # damping
            ctypes.c_int,    # pointsCount
            ctypes.POINTER(ctypes.c_float),  # positions (btVector3*)
            ctypes.POINTER(ctypes.c_float),  # predictedPositions
            ctypes.POINTER(ctypes.c_float),  # velocities
            ctypes.POINTER(ctypes.c_float),  # forces
            ctypes.POINTER(ctypes.c_float),  # invMasses
            ctypes.POINTER(ctypes.c_float),  # gravity (btVector3*)
        ]
        self.defkit.PredictPositions_native.restype = None

        # PredictRotationsPBD
        self.defkit.PredictRotationsPBD.argtypes = [
            ctypes.c_float,  # dt
            ctypes.c_float,  # damping
            ctypes.c_int,    # pointsCount
            ctypes.POINTER(ctypes.c_float),  # orientations (btQuaternion*)
            ctypes.POINTER(ctypes.c_float),  # predictedOrientations
            ctypes.POINTER(ctypes.c_float),  # angVelocities (btVector3*)
            ctypes.POINTER(ctypes.c_float),  # torques
            ctypes.POINTER(ctypes.c_float),  # quatInvMass
        ]
        self.defkit.PredictRotationsPBD.restype = None

        # Integrate_native
        self.defkit.Integrate_native.argtypes = [
            ctypes.c_float,  # dt
            ctypes.c_int,    # pointsCount
            ctypes.POINTER(ctypes.c_float),  # positions
            ctypes.POINTER(ctypes.c_float),  # predictedPositions
            ctypes.POINTER(ctypes.c_float),  # velocities
            ctypes.POINTER(ctypes.c_float),  # invMasses
        ]
        self.defkit.Integrate_native.restype = None

        # IntegrateRotationsPBD
        self.defkit.IntegrateRotationsPBD.argtypes = [
            ctypes.c_float,  # dt
            ctypes.c_int,    # pointsCount
            ctypes.POINTER(ctypes.c_float),  # orientations
            ctypes.POINTER(ctypes.c_float),  # predictedOrientations
            ctypes.POINTER(ctypes.c_float),  # prevOrientations
            ctypes.POINTER(ctypes.c_float),  # angVelocities
            ctypes.POINTER(ctypes.c_float),  # quatInvMass
        ]
        self.defkit.IntegrateRotationsPBD.restype = None

        # ProjectElasticRodConstraints (from DefKitAdv)
        self.defkit_adv.ProjectElasticRodConstraints.argtypes = [
            ctypes.c_int,    # pointsCount
            ctypes.POINTER(ctypes.c_float),  # positions (btVector3*)
            ctypes.POINTER(ctypes.c_float),  # orientations (btQuaternion*)
            ctypes.POINTER(ctypes.c_float),  # invMasses
            ctypes.POINTER(ctypes.c_float),  # quatInvMasses
            ctypes.POINTER(ctypes.c_float),  # restDarboux (btQuaternion*)
            ctypes.POINTER(ctypes.c_float),  # bendAndTwistKs (btVector3*)
            ctypes.POINTER(ctypes.c_float),  # restLength
            ctypes.c_float,  # stretchKs
            ctypes.c_float,  # shearKs
        ]
        self.defkit_adv.ProjectElasticRodConstraints.restype = None
```

### Phase 2: Rod State Manager

Create `newton/examples/cosserat_dll/rod_state.py`:

```python
"""State management for Cosserat rod simulation using DefKit DLL."""

import numpy as np


class RodState:
    """Manages all state arrays for a Cosserat rod."""

    def __init__(self, n_particles: int):
        self.n_particles = n_particles
        self.n_edges = n_particles - 1

        # Position state (btVector3 = 4 floats for SIMD alignment)
        self.positions = np.zeros((n_particles, 4), dtype=np.float32)
        self.predicted_positions = np.zeros((n_particles, 4), dtype=np.float32)
        self.velocities = np.zeros((n_particles, 4), dtype=np.float32)
        self.forces = np.zeros((n_particles, 4), dtype=np.float32)
        self.inv_masses = np.ones(n_particles, dtype=np.float32)

        # Orientation state (btQuaternion = 4 floats: x, y, z, w)
        self.orientations = np.zeros((n_particles, 4), dtype=np.float32)
        self.orientations[:, 3] = 1.0  # Identity quaternion (w=1)
        self.predicted_orientations = np.zeros((n_particles, 4), dtype=np.float32)
        self.predicted_orientations[:, 3] = 1.0
        self.prev_orientations = np.zeros((n_particles, 4), dtype=np.float32)
        self.prev_orientations[:, 3] = 1.0
        self.angular_velocities = np.zeros((n_particles, 4), dtype=np.float32)
        self.torques = np.zeros((n_particles, 4), dtype=np.float32)
        self.quat_inv_masses = np.ones(n_particles, dtype=np.float32)

        # Rod properties
        self.rest_lengths = np.zeros(n_edges, dtype=np.float32)
        self.rest_darboux = np.zeros((n_edges, 4), dtype=np.float32)  # Quaternion
        self.rest_darboux[:, 3] = 1.0  # Identity
        self.bend_twist_ks = np.ones((n_edges, 4), dtype=np.float32)  # btVector3 (4 floats)

        # Stiffness parameters
        self.stretch_ks = 1.0
        self.shear_ks = 1.0
```

### Phase 3: Simulation Driver

Create `newton/examples/cosserat_dll/simulation.py`:

```python
"""Cosserat rod simulation using DefKit DLL."""

import numpy as np
from .defkit_wrapper import DefKitWrapper
from .rod_state import RodState


class CosseratRodSimulation:
    """Cosserat rod simulation using DefKit DLL backend."""

    def __init__(self, dll_path: str, n_particles: int):
        self.dll = DefKitWrapper(dll_path)
        self.state = RodState(n_particles)

        # Simulation parameters
        self.position_damping = 0.001
        self.rotation_damping = 0.001
        self.gravity = np.array([0.0, 0.0, -9.81, 0.0], dtype=np.float32)
        self.constraint_iterations = 4

    def step(self, dt: float):
        """Advance simulation by one timestep."""
        s = self.state
        n = s.n_particles

        # Get ctypes pointers
        pos_ptr = s.positions.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        pred_pos_ptr = s.predicted_positions.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        vel_ptr = s.velocities.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        force_ptr = s.forces.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        inv_mass_ptr = s.inv_masses.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        gravity_ptr = self.gravity.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        orient_ptr = s.orientations.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        pred_orient_ptr = s.predicted_orientations.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        prev_orient_ptr = s.prev_orientations.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        ang_vel_ptr = s.angular_velocities.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        torque_ptr = s.torques.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        quat_inv_mass_ptr = s.quat_inv_masses.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        rest_darboux_ptr = s.rest_darboux.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        bend_twist_ptr = s.bend_twist_ks.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        rest_length_ptr = s.rest_lengths.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        # 1. Predict positions and rotations
        self.dll.defkit.PredictPositions_native(
            dt, self.position_damping, n,
            pos_ptr, pred_pos_ptr, vel_ptr, force_ptr, inv_mass_ptr, gravity_ptr
        )

        self.dll.defkit.PredictRotationsPBD(
            dt, self.rotation_damping, n,
            orient_ptr, pred_orient_ptr, ang_vel_ptr, torque_ptr, quat_inv_mass_ptr
        )

        # 2. Project constraints (iterative)
        for _ in range(self.constraint_iterations):
            self.dll.defkit_adv.ProjectElasticRodConstraints(
                n, pred_pos_ptr, pred_orient_ptr,
                inv_mass_ptr, quat_inv_mass_ptr,
                rest_darboux_ptr, bend_twist_ptr, rest_length_ptr,
                s.stretch_ks, s.shear_ks
            )

        # 3. Integrate
        self.dll.defkit.Integrate_native(
            dt, n, pos_ptr, pred_pos_ptr, vel_ptr, inv_mass_ptr
        )

        self.dll.defkit.IntegrateRotationsPBD(
            dt, n, orient_ptr, pred_orient_ptr, prev_orient_ptr, ang_vel_ptr, quat_inv_mass_ptr
        )

        # Clear forces for next step
        s.forces.fill(0)
        s.torques.fill(0)
```

### Phase 4: Newton Example Integration

Create `newton/examples/cosserat_dll/example_dll_cosserat_rod.py`:

```python
"""Cosserat rod simulation using DefKit DLL with Newton viewer.

Command: uv run python newton/examples/cosserat_dll/example_dll_cosserat_rod.py
"""

import numpy as np
import warp as wp
import argparse

import newton
import newton.examples
from .simulation import CosseratRodSimulation


class Example:
    """Demo of Cosserat rod using DefKit DLL backend."""

    def __init__(self, viewer, args=None):
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.substeps = 4

        self.viewer = viewer
        self.args = args

        # Get DLL path from args (default: unity_ref/)
        dll_path = args.dll_path if args and hasattr(args, 'dll_path') else "unity_ref"

        # Rod parameters
        self.n_particles = 10
        self.segment_length = 0.1
        self.particle_radius = 0.02

        # Create simulation
        self.sim = CosseratRodSimulation(dll_path, self.n_particles)

        # Initialize rod as horizontal cantilever along X
        for i in range(self.n_particles):
            self.sim.state.positions[i] = [i * self.segment_length, 0.0, 1.0, 0.0]
            self.sim.state.predicted_positions[i] = self.sim.state.positions[i].copy()

        # Set rest lengths
        self.sim.state.rest_lengths[:] = self.segment_length

        # Fix first particle
        self.sim.state.inv_masses[0] = 0.0
        self.sim.state.quat_inv_masses[0] = 0.0

        # Initialize orientations (align with rod direction)
        # Quaternion rotating Z to X: 90 degrees around Y
        q_y90 = np.array([0, np.sin(np.pi/4), 0, np.cos(np.pi/4)], dtype=np.float32)
        for i in range(self.n_particles):
            self.sim.state.orientations[i] = q_y90
            self.sim.state.predicted_orientations[i] = q_y90
            self.sim.state.prev_orientations[i] = q_y90

        # Build Newton model for visualization
        builder = newton.ModelBuilder()
        builder.add_ground_plane()

        for i in range(self.n_particles):
            mass = 0.0 if i == 0 else 1.0
            pos = tuple(self.sim.state.positions[i][:3])
            builder.add_particle(pos=pos, vel=(0, 0, 0), mass=mass, radius=self.particle_radius)

        self.model = builder.finalize()
        self.state = self.model.state()

        self._sync_state()

        self.viewer.set_model(self.model)
        self.viewer.show_particles = True

    def _sync_state(self):
        """Sync DLL state to Newton state for visualization."""
        positions_3d = self.sim.state.positions[:, :3].astype(np.float32)
        positions_wp = wp.array(positions_3d, dtype=wp.vec3, device=self.model.device)
        self.state.particle_q.assign(positions_wp)

    def step(self):
        sub_dt = self.frame_dt / self.substeps

        for _ in range(self.substeps):
            self.sim.step(sub_dt)

        self._sync_state()
        self.sim_time += self.frame_dt

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state)

        # Draw rod segments as lines
        positions_3d = self.sim.state.positions[:, :3].astype(np.float32)
        starts = wp.array(positions_3d[:-1], dtype=wp.vec3, device=self.model.device)
        ends = wp.array(positions_3d[1:], dtype=wp.vec3, device=self.model.device)
        colors = wp.array([[0.2, 0.6, 1.0]] * (self.n_particles - 1), dtype=wp.vec3, device=self.model.device)
        self.viewer.log_lines("/rod", starts, ends, colors)

        self.viewer.end_frame()

    def gui(self, ui):
        ui.text("DefKit DLL Cosserat Rod")
        ui.text(f"Particles: {self.n_particles}")
        ui.separator()

        _, self.substeps = ui.slider_int("Substeps", self.substeps, 1, 16)
        _, self.sim.constraint_iterations = ui.slider_int("Iterations", self.sim.constraint_iterations, 1, 16)

        ui.separator()
        _, self.sim.state.stretch_ks = ui.slider_float("Stretch Ks", self.sim.state.stretch_ks, 0.0, 2.0)
        _, self.sim.state.shear_ks = ui.slider_float("Shear Ks", self.sim.state.shear_ks, 0.0, 2.0)

    def test_final(self):
        """Validation after simulation."""
        tip_z = self.sim.state.positions[-1, 2]
        assert tip_z < 0.9, f"Tip should drop below 0.9, got {tip_z}"


def add_args(parser: argparse.ArgumentParser):
    parser.add_argument("--dll-path", type=str, default="unity_ref",
                       help="Path to directory containing DefKit.dll and DefKitAdv.dll")


if __name__ == "__main__":
    viewer, args = newton.examples.init(add_args)
    example = Example(viewer, args)
    newton.examples.run(example, args)
```

---

## 5. File Structure

```
newton/examples/cosserat_dll/
    __init__.py
    defkit_wrapper.py      # DLL ctypes wrapper
    rod_state.py           # State management
    simulation.py          # Simulation driver
    example_dll_cosserat_rod.py  # Newton example
```

---

## 6. Pre-compiled DLLs

The DLLs are already compiled and available in `unity_ref/`:

```
unity_ref/
    DefKit.dll          # Core position/rotation integration
    DefKitAdv.dll       # Elastic rod constraint solvers
    mkl_core.2.dll      # Intel MKL dependencies (for direct solver)
    mkl_def.2.dll
    mkl_sequential.2.dll
    mkl_msg.dll
```

The Python wrapper will load these directly from `unity_ref/`.

---

## 7. Key Implementation Notes

### 7.1 Data Alignment
- `btVector3` uses 4 floats (16 bytes) for SIMD alignment, not 3
- This matches Unity's `Vector4` usage in the C# code
- NumPy arrays must use shape `(n, 4)` for vectors

### 7.2 Quaternion Convention
- Bullet uses (x, y, z, w) order
- Ensure consistent quaternion ordering throughout

### 7.3 Constraint Projection
The iterative solver (`ProjectElasticRodConstraints`) projects constraints from both ends toward the middle (left-right sweep) for better convergence.

### 7.4 Rest Darboux Vector
The rest Darboux vector (`intrinsicBend` in C#) encodes the intrinsic curvature and twist. For a straight rod, this is the identity quaternion (0, 0, 0, 1).

---

## 8. Testing Plan

1. **Unit Tests**: Test DLL wrapper with simple data
2. **Integration Test**: Single particle with gravity (no constraints)
3. **Rod Test**: Cantilever bending under gravity
4. **Comparison**: Compare results with pure Python/NumPy implementation in `cosserat2/reference/`

---

## 9. Future Extensions

### Phase 2: Direct Solver
After the iterative solver works:
1. Add `InitDirectElasticRod` wrapper
2. Add `PrepareDirectElasticRodConstraints`
3. Add `ProjectJMJT_DirectElasticRodConstraintsBanded`
4. Add `DestroyDirectElasticRod`

### Phase 3: Collisions
- Add SDF collision detection integration
- Use existing Newton collision infrastructure

---

## 10. Development Steps

1. [x] **Step 1**: Create folder structure and `__init__.py`
2. [x] **Step 2**: Implement `defkit_wrapper.py` with ctypes bindings
3. [x] **Step 3**: Implement `rod_state.py` for state management
4. [x] **Step 4**: Implement `simulation.py` with simulation loop
5. [x] **Step 5**: Implement `example_dll_cosserat_rod.py` Newton example
6. [x] **Step 6**: Test with pre-compiled DLLs from `unity_ref/`
7. [x] **Step 7**: Debug and validate against C# behavior
8. [x] **Step 8**: Add GUI controls and parameter tuning
9. [ ] **Step 9**: Document and add to README

---

## Appendix A: Function Signatures Reference

### DefKit.dll

```c
// Position integration
void PredictPositions_native(float dt, float damping, int pointsCount,
    btVector3* positions, btVector3* predictedPositions,
    btVector3* velocities, btVector3* forces, float* invMasses, btVector3* gravity);

void Integrate_native(float dt, int pointsCount,
    btVector3* positions, btVector3* predictedPositions,
    btVector3* velocities, float* invMasses);

// Rotation integration
void PredictRotationsPBD(float dt, float damping, int pointsCount,
    btQuaternion* orientations, btQuaternion* predictedOrientations,
    btVector3* angVelocities, btVector3* torques, float* quatInvMass);

void IntegrateRotationsPBD(float dt, int pointsCount,
    btQuaternion* orientations, btQuaternion* predictedOrientations,
    btQuaternion* prevOrientations, btVector3* angVelocities, float* quatInvMass);
```

### DefKitAdv.dll

```c
// Iterative constraint projection
void ProjectElasticRodConstraints(int pointsCount,
    btVector3* positions, btQuaternion* orientations,
    float* invMasses, float* quatInvMasses,
    btQuaternion* restDarboux, btVector3* bendAndTwistKs,
    float* restLength, float stretchKs, float shearKs);

// Direct solver (Phase 2)
void* InitDirectElasticRod(int pointsCount, btVector3* positions,
    btQuaternion* orientations, float radius, float* restLengths,
    float youngModulus, float torsionModulus);

void PrepareDirectElasticRodConstraints(void* rod, int pointsCount, float dt,
    btVector3* bendStiffness, btVector3* restDarboux, float* restLengths,
    float youngModulusMult, float torsionModulusMult);

void UpdateConstraints_DirectElasticRodConstraintsBanded(void* rod, int pointsCount,
    btVector3* positions, btQuaternion* orientations, float* invMasses);

void ComputeJacobians_DirectElasticRodConstraintsBanded(void* rod, int startId, int pointsCount,
    btVector3* positions, btQuaternion* orientations, float* invMasses);

void AssembleJMJT_DirectElasticRodConstraintsBanded(void* rod, int startId, int pointsCount,
    btVector3* positions, btQuaternion* orientations, float* invMasses);

void ProjectJMJT_DirectElasticRodConstraintsBanded(void* rod, int pointsCount,
    btVector3* positions, btQuaternion* orientations, float* invMasses,
    btVector3* posCorr, btQuaternion* rotCorr);

void DestroyDirectElasticRod(void* rod);
```
