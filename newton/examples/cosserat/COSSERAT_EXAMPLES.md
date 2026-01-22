# Cosserat Rod Examples

This document describes the Cosserat rod examples, their implementation approaches, and status.

## Terminology

- **Jacobi iteration**: All constraints compute corrections in parallel, accumulate via atomics, then apply together. This is NOT Gauss-Seidel.
- **Global solve**: Assembles and solves a system matrix A = J M^{-1} J^T directly using Cholesky or Thomas algorithm.
- **Block-Jacobi**: Partitions constraints into tiles, solves each tile independently, coupling happens through outer iterations.

## Summary Table

| Example | Solver Type | Status | Unique Feature |
|---------|-------------|--------|----------------|
| 00 | Cholesky (single tile) | ✅ | Distance constraints only |
| 01_multitile | Block-Jacobi Cholesky | ✅ | Multi-tile distance constraints |
| 01_thomas | Thomas algorithm | ✅ | O(n) direct solve |
| 02 | Jacobi iteration | ✅ | Basic Cosserat rod |
| 03_broken | Block-Thomas | ❌ | Stretch-only, visualization issue |
| 03_full_broken | Block-Thomas | ❌ | Full 6-DOF, visualization issue |
| 04_multitile | Jacobi iteration | ✅ | Extended chain (129 particles) |
| 05 | Jacobi + friction | ✅ | 3 internal friction models |
| 06 | Per-joint Cholesky | ✅ | 6×6 per-joint solve |
| 07 | Global Cholesky | ✅ | Scalar reduction |
| 08_multitile | Block-Jacobi Cholesky | ✅ | Scalar reduction, multi-tile |
| 08_cholesky | Global Cholesky | ✅ | 3-DOF tangent space |
| 09 | Block-Jacobi Cholesky | ⚠️ | 3-DOF tangent, multi-tile |
| 09_combined | Global Cholesky | ? | Combined stretch+bend system |

---

## Foundation Examples (Distance Constraints Only)

### 00_global_pbd_chain.py - ✅ WORKING

Distance constraints only (no Cosserat rod physics), uses single 32×32 tile and Cholesky to solve the tridiagonal system.

**Linear system solver**: Tile Cholesky (dense, single tile)
**Constraint coupling**: Tridiagonal structure from distance constraints

---

### 01_global_pbd_chain_multitile.py - ✅ WORKING

Distance constraints only, uses multiple tiles with Block-Jacobi Cholesky. Each tile is solved independently in parallel. Boundary particles receive corrections from both adjacent tiles using atomic operations. Coupling between tiles happens through outer constraint iterations.

**Linear system solver**: Tile Cholesky (Block-Jacobi, 4 tiles)
**Constraint coupling**: Tridiagonal within each tile, inter-tile via iterations

---

### 01_global_pbd_chain_thomas.py - ✅ WORKING

Distance constraints only. Demonstrates a global matrix-based PBD approach using the Thomas algorithm (TDMA - TriDiagonal Matrix Algorithm) for solving the tridiagonal system. Unlike tile-based Cholesky which is limited by shared memory, Thomas algorithm can handle arbitrarily long chains.

**Linear system solver**: Thomas algorithm (O(n) direct solve)
**Constraint coupling**: Full tridiagonal coupling

---

## Jacobi-Style Iterative Examples

### 02_local_cosserat_rod.py - ✅ WORKING

Demonstrates Position And Orientation Based Cosserat Rods using iterative Jacobi-style constraint projection. Implements the two core constraint solvers from the paper:
- Stretch/Shear constraint: enforces edge length and alignment with d3 director
- Bend/Twist constraint: enforces relative rotation via Darboux vector

**Linear system solver**: None (iterative Jacobi projection)
**Constraint coupling**: Stretch and bend solved sequentially per iteration, corrections accumulated via atomics

Reference: "Position And Orientation Based Cosserat Rods" by Tassilo Kugelstadt, RWTH Aachen University
https://animation.rwth-aachen.de/publication/0550/

---

### 04_global_cosserat_rod_multitile.py - ✅ WORKING

Extended version of 02 for longer rods (129 particles). Uses iterative Jacobi-style projection with atomic accumulation.

**Linear system solver**: None (iterative Jacobi projection)
**Constraint coupling**: Same as 02, just more particles
**Note**: Redundant with 05 - consider using 05 instead which adds friction models

---

### 05_global_cosserat_rod_multitile_friction.py - ✅ WORKING

Extends example 04 with three internal friction models:
1. **Velocity Damping**: v_new = v * damping_coeff (simplest, not physically accurate)
2. **Strain-Rate Damping**: Damping proportional to curvature change rate
3. **Dahl Hysteresis**: Path-dependent friction with hysteresis loops

**Linear system solver**: None (iterative Jacobi projection)
**Constraint coupling**: Same as 02/04
**Recommended**: Use this instead of 04 for Jacobi-style Cosserat simulation

---

## Block-Tridiagonal Examples (NOT WORKING)

### 03_block_tridiagonal_cosserat_broken.py - ❌ NOT WORKING

Demonstrates a block-tridiagonal solver for Cosserat rod constraints with 3×3 blocks (stretch only). Uses block Thomas algorithm for O(n) direct solve.

**Linear system solver**: Block Thomas algorithm (3×3 blocks)
**Constraint coupling**: Block-tridiagonal structure
**Known issue**: Rod not visible after start - likely initialization or rendering bug

---

### 03_block_tridiagonal_cosserat_full_broken.py - ❌ NOT WORKING

Attempts full Cosserat Rod with 6×6 block-tridiagonal global solver combining stretch/shear (3 DOFs) and bend/twist (3 DOFs) per edge.

**Linear system solver**: Block Thomas algorithm (6×6 blocks)
**Constraint coupling**: Block-tridiagonal, but indexing may be incorrect
**Known issue**: Rod not visible - may have block indexing issues in addition to visualization

---

## Per-Joint Direct Solver

### 06_direct_elastic_rod.py - ✅ WORKING

Per-joint direct solver using 6×6 SPD system embedded in 8×8 tile for Warp's Cholesky. Combines stretch and bend at each joint.

**Linear system solver**: Per-joint 6×6 Cholesky (independent solves)
**Constraint coupling**: No inter-joint coupling in matrix (resolved via iterations)

---

## Global Cholesky Examples

### 07_global_cosserat_rod_cholesky.py - ✅ WORKING

TRUE global Cholesky solve for Cosserat rods. Assembles A = J M^{-1} J^T and solves directly. 

**Formulation**: Scalar reduction - 3D constraints reduced to scalar magnitude
**Linear system solver**: Tile Cholesky (32×32)
**Constraint coupling**: Tridiagonal within each system (stretch or bend solved separately)

---

### 08_global_cosserat_rod_cholesky_multitile.py - ✅ WORKING

Block-Jacobi version of 07 for longer rods. Assembles block systems per tile and solves via Cholesky in parallel.

**Formulation**: Scalar reduction (same as 07)
**Linear system solver**: Tile Cholesky (Block-Jacobi, 4 tiles)
**Constraint coupling**: Tridiagonal within tiles, inter-tile via iterations and atomics at boundaries

---

### 08_cholesky_cosserat_rod.py - ✅ WORKING

Global Cholesky solver with tangent-space (3-DOF) quaternion parameterization. More accurate than scalar reduction.

**Formulation**: Full 3-DOF tangent-space
- 16 stretch × 3 = 48 scalar constraints in 64×64 tile
- 15 bend × 3 = 45 scalar constraints in separate 64×64 tile

**Linear system solver**: Tile Cholesky (separate stretch/bend systems)
**Constraint coupling**: Full coupling within each 3D constraint system

---

### 09_cholesky_cosserat_rod_multitile.py - ⚠️ PARTIALLY WORKING

Block-Jacobi Cholesky for longer rods with 3-DOF tangent-space formulation. 32 constraints × 3 = 96 scalar DOFs per tile.

**Formulation**: Full 3-DOF tangent-space (same as 08_cholesky)
**Linear system solver**: Tile Cholesky (Block-Jacobi, 96×96 tiles)
**Constraint coupling**: Full 3D coupling within tiles
**Known issue**: Explodes when changing rest shape via UI

---

### 09_cholesky_cosserat_rod_multitile_combined.py - STATUS UNKNOWN

Combined stretch+bend in same 6×6 block per joint. This is the most physically accurate formulation as it captures coupling between stretch and bend constraints.

**Formulation**: Combined 6-DOF per joint
**Linear system solver**: Tile Cholesky
**Note**: Status needs verification

---

## Application Examples

### 10_sim_aorta.py - STATUS UNKNOWN

Aorta simulation example. Status needs verification.

---

### 11_cable_aorta.py - STATUS UNKNOWN

Cable in aorta simulation. Status needs verification.

---

### 12_cable_sim.py - STATUS UNKNOWN

Cable simulation example. Status needs verification.

---

## Recommendations

### For learning Cosserat rod simulation:
1. Start with **02** to understand basic Jacobi-style PBD for Cosserat rods
2. Move to **05** to see friction models
3. Study **07** or **08_cholesky** for global Cholesky approaches

### For production use:
- **Short rods (< 32 particles)**: Use **07** (scalar) or **08_cholesky** (3-DOF)
- **Long rods with friction**: Use **05**
- **Long rods without friction**: Use **04_multitile** or the multi-tile Cholesky variants

### Redundancy notes:
- **04_multitile** is redundant with **05** (05 has all features of 04 plus friction)
- **08_multitile** is the multi-tile version of **07** (same formulation, longer rods)
- **09** is the multi-tile version of **08_cholesky** (same formulation, longer rods)
