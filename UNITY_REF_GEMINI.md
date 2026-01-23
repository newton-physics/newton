# Unity Reference Cosserat Rod Simulation Plan

This plan outlines the steps to create a Python application using Newton as a viewer that interoperates with a native C++ DLL (`DefKitAdv.dll`) to simulate a Cosserat rod model, based on the reference implementation in `unity_ref`.

## 1. Build `DefKitAdv.dll`

We need to compile the C++ source files into a dynamic link library that exports the necessary functions for the simulation.

### Source Files
- `unity_ref/Native/DefKit.cpp`: Exports `PredictRotationsPBD`, `IntegrateRotationsPBD`.
- `unity_ref/Native/ElasticRod.cpp`: Exports `ProjectElasticRodConstraints`.
- `unity_ref/Native/PositionBasedDynamics/*.cpp`: PBD library implementation (needed for compilation).
- `unity_ref/Native/LinearMath/*.cpp`: Bullet Math implementation.

### CMake Configuration
Create `unity_ref/Native/CMakeLists.txt` to define the `DefKitAdv` library.

**Key Requirements:**
- Include directories: `.` (current), `PositionBasedDynamics`, `LinearMath`.
- Define `EXPORT_API` (handled in source).
- Target: `SHARED` library named `DefKitAdv`.

### Build Process
1.  Configure with CMake.
2.  Build using the generated project (e.g., Visual Studio or Ninja).

## 2. Python Implementation (`newton/examples/cosserat3/cosserat_gemini.py`)

Create a Newton example script that acts as the simulation controller and viewer.

### Architecture
- **Viewer**: Use `newton.viewer` (like in `example_pypbd_cosserat_rod.py`).
- **Simulation**:
    - Load `DefKitAdv.dll` using `ctypes`.
    - Define `ctypes` Structures for `btVector3` (4 floats) and `btQuaternion` (4 floats).
    - Manage memory for particle positions, orientations, velocities, etc., using `ctypes` arrays or `numpy` arrays (passed as pointers).
    - Implement the simulation loop in Python, calling C++ functions:
        1.  **Predict Rotations**: Call `PredictRotationsPBD`.
        2.  **Constraint Projection**: Call `ProjectElasticRodConstraints` (iteratively).
        3.  **Integration**: Call `IntegrateRotationsPBD`.
        4.  (Also need to handle position integration if not covered by the above - the C# ref seems to separate rotation and position PBD, but `DefKitElasticRodSystem` focuses on rotation. I should check if positions are also simulated or if it's just a rod with fixed positions for testing, or if I need to implement `PredictPositions_native` and `Integrate_native` calls as well. The C# code has `body.predictedPositionsNativePtr`, so positions are likely simulated too. I will add position simulation steps using `PredictPositions_native` and `Integrate_native` which are in `DefKit.cpp`).

### Data Structures
- `positions`: Array of `btVector3`
- `velocities`: Array of `btVector3`
- `orientations`: Array of `btQuaternion`
- `angularVelocities`: Array of `btVector3` (or Vector4 in C#?)
- Mass/InverseMass arrays.

### Simulation Loop (per substep)
1.  `PredictPositions_native(...)`
2.  `PredictRotationsPBD(...)`
3.  Loop `constraintIterations`:
    - `ProjectElasticRodConstraints(...)`
4.  `Integrate_native(...)`
5.  `IntegrateRotationsPBD(...)`

## 3. Verification
- Run `uv run -m newton.examples.cosserat3.cosserat_gemini`
- Visualize the rod in the viewer.
- Check for stability and correct behavior (bending, twisting).

## 4. Tasks
- [ ] Create `unity_ref/Native/CMakeLists.txt`
- [ ] Build `DefKitAdv.dll`
- [ ] Create `newton/examples/cosserat3/__init__.py`
- [ ] Create `newton/examples/cosserat3/cosserat_gemini.py`
- [ ] Implement simulation loop and visualization.
