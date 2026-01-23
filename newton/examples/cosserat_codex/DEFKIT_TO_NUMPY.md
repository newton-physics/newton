# DefKit to NumPy Port Log

This file tracks the incremental port of the Direct Position Based Solver for
Stiff Rods from `DefKitAdv.dll` to NumPy in `numpy_cosserat_codex.py`.

## Current Setup
- Reference rod: native DLL pipeline (banded or non-banded).
- NumPy rod: hybrid pipeline with per-step NumPy overrides.

## Method Status
| Step | C/C++ Function | NumPy Status | Notes |
| --- | --- | --- | --- |
| Predict positions | `PredictPositions_native` | NumPy implemented | Vectorized translation of `DefKit.cpp` |
| Integrate positions | `Integrate_native` | NumPy implemented | Vectorized translation of `DefKit.cpp` |
| Predict rotations | `PredictRotationsPBD` | NumPy implemented | Vectorized translation of `DefKit.cpp` |
| Integrate rotations | `IntegrateRotationsPBD` | NumPy implemented | Vectorized translation of `DefKit.cpp` |
| Prepare constraints | `PrepareDirectElasticRodConstraints` | NumPy implemented | Compliance/lambda prep |
| Darboux vector | `DirectPositionBasedSolverForStiffRods::computeDarbouxVector` | NumPy implemented | Matches `q0.conjugate() * q1` vector part |
| Bending/torsion Jacobians | `DirectPositionBasedSolverForStiffRods::computeBendingAndTorsionJacobians` | NumPy implemented | Uses exact `jOmega * G` (no length scaling) |
| Quaternion G matrix | `DirectPositionBasedSolverForStiffRods::computeMatrixG` | NumPy implemented | Full 4x3 correction (incl. w) |
| Update constraints (banded) | `UpdateConstraints_DirectElasticRodConstraintsBanded` | NumPy implemented | Constraint error eval |
| Compute Jacobians (banded) | `ComputeJacobians_DirectElasticRodConstraintsBanded` | NumPy implemented | Linearized jacobians |
| Assemble JMJT (banded) | `AssembleJMJT_DirectElasticRodConstraintsBanded` | NumPy implemented | Banded JMJT |
| Project JMJT (banded) | `ProjectJMJT_DirectElasticRodConstraintsBanded` | NumPy implemented | Banded solve + apply |
| Project direct (non-banded) | `ProjectDirectElasticRodConstraints` | NumPy implemented | Dense solve + coupling + lambdaSum/quatG |
