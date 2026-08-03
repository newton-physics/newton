# LIMX Cloth Self-Collision Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add frictionless VF/EE cloth self-collision plus matrix-free EF untangling to LIMX on CUDA.

**Architecture:** A public `ConstraintSelfCollision` owns fixed-capacity contact buffers and triangle/edge BVHs. At each current-position Newton linearization it freezes VF, EE, and EF contacts. The dynamic operator then adds balanced forces, complete rank-one PSD Hessian-vector products, and exact block-Jacobi diagonal blocks without inserting collision topology into block-CSR.

**Tech Stack:** Python, Warp CUDA kernels/BVH, NumPy topology preprocessing, Newton `SolverBase`, `unittest`.

## Global Constraints

- Follow `docs/superpowers/specs/2026-08-03-limx-cloth-self-collision-design.md`.
- Keep static elasticity in 3-by-3 block-CSR and every collision contribution matrix-free.
- Detect at the current Newton iterate and freeze one contact snapshot for force, diagonal, and all PCG HVPs.
- Use full rank-one VF/EE/EF Hessians, not diagonal-only collision Hessians.
- Use correct EE weights `(1-s, s, -(1-t), -t)`.
- Add no friction, CCD, rigid-body collision, CPU simulation path, or new dependency.
- Preserve the current example timestep, Newton/PCG budgets, no-damping behavior, and warm start.

---

### Task 1: Rank-One Dynamic Contact Algebra

**Files:**
- Create: `newton/_src/solvers/limx/constraints/self_collision.py`
- Modify: `newton/tests/test_solver_limx.py`

- [ ] Add CUDA tests with literal four- and five-particle contacts for balanced force, dense-reference HVP, exact diagonal blocks, PSD quadratic form, and inactive-count behavior.
- [ ] Run the focused test class and verify RED because the dynamic contact implementation is absent.
- [ ] Implement fixed-buffer force, HVP, and diagonal kernels using one scalar reduction per contact and atomic particle accumulation.
- [ ] Run focused tests and verify GREEN.
- [ ] Add validation tests for capacity, finite positive parameters, array devices, and bounded overflow behavior; implement only enough host API to satisfy them.

### Task 2: GPU VF and EE Detection

**Files:**
- Modify: `newton/_src/solvers/limx/constraints/self_collision.py`
- Modify: `newton/tests/test_solver_limx.py`

- [ ] Add CUDA VF tests for one known point/triangle contact, barycentric weights and depth, membership rejection, world rejection, and degenerate geometry.
- [ ] Run the focused VF tests and verify RED.
- [ ] Build fixed triangle topology/AABBs/BVH and implement direct point-AABB query plus VF narrow phase and atomic append.
- [ ] Run VF tests and verify GREEN.
- [ ] Add CUDA EE tests with asymmetric interior parameters `s != t`, plus duplicate, shared-endpoint, world, parallel, and degenerate rejection.
- [ ] Run EE tests and verify RED.
- [ ] Derive all unique edges through `MeshAdjacency`, build/refit the edge BVH, and implement EE query/narrow phase with correct `t` weights.
- [ ] Run VF/EE tests and verify GREEN.

### Task 3: GPU EF Untangling

**Files:**
- Modify: `newton/_src/solvers/limx/constraints/self_collision.py`
- Modify: `newton/tests/test_solver_limx.py`

- [ ] Add a literal edge-through-triangle CUDA fixture and assert one EF contact, five signed weights summing to zero, finite unit ICM direction, and depth `2*thickness`.
- [ ] Assert the five-particle HVP contains full off-diagonal coupling and matches the independent dense rank-one reference.
- [ ] Run EF tests and verify RED.
- [ ] Implement edge/triangle crossing narrow phase and ICM recovery direction from the edge's two adjacent triangles.
- [ ] Run all self-collision unit tests and verify GREEN.

### Task 4: Newton Lifecycle and Public API

**Files:**
- Modify: `newton/_src/solvers/limx/operator.py`
- Modify: `newton/_src/solvers/limx/solver_newton.py`
- Modify: `newton/_src/solvers/limx/constraints/__init__.py`
- Modify: `newton/_src/solvers/limx/__init__.py`
- Modify: `newton/_src/solvers/__init__.py`
- Modify: `newton/tests/test_solver_limx.py`
- Modify: `CHANGELOG.md`
- Regenerate: `docs/api/newton_solvers.rst`

- [ ] Add a recording dynamic operator test requiring one `prepare(current_iterate)` before force/diagonal/HVP in every Newton iteration and no detection inside PCG.
- [ ] Add a public-export test for `newton.solvers.ConstraintSelfCollision` and verify RED.
- [ ] Extend the dynamic protocol with `prepare`; call it at the correct solver lifecycle point and preserve backward compatibility for operators with no preparation work.
- [ ] Export the self-collision class, add an `[Unreleased] / Added` entry, and run `docs/generate_api.py`.
- [ ] Run the lifecycle/public tests and verify GREEN.

### Task 5: CUDA Example and Integration Verification

**Files:**
- Modify: `newton/examples/cloth/example_cloth_limx.py`
- Modify: `newton/tests/test_solver_limx.py`

- [ ] Add a failing example test requiring the self-collision dynamic operator without changing the existing physics/render parameters.
- [ ] Add a small CUDA integration fixture with initially penetrating cloth pieces and assert finite state plus increasing separation after a solve.
- [ ] Enable self-collision in `cloth_limx` with particle-diameter thickness and tuned penalty stiffness; verify CUDA graph capture and launch.
- [ ] Run all focused self-collision, LIMX public/lifecycle, and example CUDA tests.
- [ ] Run the headless CUDA `cloth_limx` example for 100 frames.
- [ ] Run `uvx pre-commit run -a`, `git diff --check`, and inspect the final diff/status without staging or reverting `lessons.md` or `solver_convergence.png`.
