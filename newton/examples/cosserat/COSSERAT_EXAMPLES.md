# Cosserat Rod Examples

This document describes the Cosserat rod examples, their implementation approaches, and status.

## Terminology

- **Jacobi iteration**: All constraints compute corrections in parallel, accumulate via atomics, then apply together. This is NOT Gauss-Seidel.
- **Global solve**: Assembles and solves a system matrix A = J M^{-1} J^T directly using Cholesky or Thomas algorithm.
- **Block-Jacobi**: Partitions constraints into tiles, solves each tile independently, coupling happens through outer iterations.

---

## 00_global_pbd_chain.py - ✅ WORKING

Distance constraints only (no Cosserat rod physics), uses single 32×32 tile and Cholesky to solve the tridiagonal system.

**Linear system solver**: Tile Cholesky (dense, single tile)
**Constraint coupling**: Tridiagonal structure from distance constraints

---

## 01_global_pbd_chain_multitile.py - ✅ WORKING

Distance constraints only, uses multiple tiles with Block-Jacobi Cholesky. Each tile is solved independently in parallel. Boundary particles receive corrections from both adjacent tiles using atomic operations. Coupling between tiles happens through outer constraint iterations.

**Linear system solver**: Tile Cholesky (Block-Jacobi, 4 tiles)
**Constraint coupling**: Tridiagonal within each tile, inter-tile via iterations

---

## 01_global_pbd_chain_thomas.py - ✅ WORKING

Distance constraints only. Demonstrates a global matrix-based PBD approach using the Thomas algorithm (TDMA - TriDiagonal Matrix Algorithm) for solving the tridiagonal system. Unlike tile-based Cholesky which is limited by shared memory, Thomas algorithm can handle arbitrarily long chains.

**Linear system solver**: Thomas algorithm (O(n) direct solve)
**Constraint coupling**: Full tridiagonal coupling

---

## 02_local_cosserat_rod.py - ✅ WORKING

Demonstrates Position And Orientation Based Cosserat Rods using iterative Jacobi-style constraint projection. Implements the two core constraint solvers from the paper:
- Stretch/Shear constraint: enforces edge length and alignment with d3 director
- Bend/Twist constraint: enforces relative rotation via Darboux vector

**Linear system solver**: None (iterative Jacobi projection)
**Constraint coupling**: Stretch and bend solved sequentially per iteration, corrections accumulated via atomics

Reference: "Position And Orientation Based Cosserat Rods" by Tassilo Kugelstadt, RWTH Aachen University
https://animation.rwth-aachen.de/publication/0550/

---

## 03_block_tridiagonal_cosserat.py - ❌ NOT WORKING (visualization issue)

Demonstrates a block-tridiagonal solver for Cosserat rod constraints with 3×3 blocks (stretch only). Uses block Thomas algorithm for O(n) direct solve.

**Linear system solver**: Block Thomas algorithm (3×3 blocks)
**Constraint coupling**: Block-tridiagonal structure
**Known issue**: Rod not visible after start - likely initialization or rendering bug

---

## 03_block_tridiagonal_cosserat_full.py - ❌ NOT WORKING (visualization issue)

Attempts full Cosserat Rod with 6×6 block-tridiagonal global solver combining stretch/shear (3 DOFs) and bend/twist (3 DOFs) per edge.

**Linear system solver**: Block Thomas algorithm (6×6 blocks)
**Constraint coupling**: Block-tridiagonal, but indexing may be incorrect (bend constraint k spans edges k and k+1, not contained within edge k)
**Known issue**: Rod not visible - may have block indexing issues in addition to visualization

---

## 04_global_cosserat_rod.py - ✅ WORKING

Full Cosserat rod with iterative Jacobi-style constraint projection (NOT global matrix solve despite filename). Implements stretch/shear and bend/twist constraints solved sequentially per iteration.

**Linear system solver**: None (iterative Jacobi projection)
**Constraint coupling**: Stretch and bend solved sequentially, corrections accumulated via atomics

---

## 04_global_cosserat_rod_multitile.py - ✅ WORKING

Extended version of 04 for longer rods (129 particles). Uses iterative Jacobi-style projection with atomic accumulation - NOT tiled Cholesky despite the name suggesting "multi-tile".

**Linear system solver**: None (iterative Jacobi projection)
**Constraint coupling**: Same as 04, just more particles
**Note**: Name is misleading - this is NOT Block-Jacobi Cholesky, just Jacobi iteration on a longer chain

---

## 05_global_cosserat_rod_multitile_friction.py - ✅ WORKING

Extends example 04 with three internal friction models:
1. Velocity Damping: v_new = v * damping_coeff (simplest, not physically accurate)
2. Strain-Rate Damping: Damping proportional to curvature change rate
3. Dahl Hysteresis: Path-dependent friction with hysteresis loops

**Linear system solver**: None (iterative Jacobi projection)
**Constraint coupling**: Same as 04/04_multitile

---

## 06_direct_elastic_rod.py - ✅ WORKING

Per-joint direct solver using 6×6 SPD system embedded in 8×8 tile for Warp's Cholesky. Combines stretch and bend at each joint.

**Linear system solver**: Per-joint 6×6 Cholesky (independent solves)
**Constraint coupling**: No inter-joint coupling in matrix (resolved via iterations)

---

## 07_global_cosserat_rod_cholesky.py - ✅ WORKING

TRUE global Cholesky solve for Cosserat rods. Assembles A = J M^{-1} J^T and solves directly. Stretch and bend are solved as separate systems.

**Linear system solver**: Tile Cholesky (scalar reduction of 3D constraints)
**Constraint coupling**: Tridiagonal within each system (stretch or bend)

---

## 08_global_cosserat_rod_cholesky_multitile.py - ✅ WORKING

Block-Jacobi with TRUE tiled Cholesky for longer rods. Assembles block systems per tile and solves via Cholesky in parallel.

**Linear system solver**: Tile Cholesky (Block-Jacobi, 4 tiles)
**Constraint coupling**: Tridiagonal within tiles, inter-tile via iterations and atomics at boundaries

---

## 08_cholesky_cosserat_rod.py - ✅ WORKING

Global Cholesky solver with tangent-space (3-DOF) quaternion parameterization. 16 stretch × 3 = 48 scalar constraints in 64×64 tile; 15 bend × 3 = 45 scalar constraints in separate 64×64 tile.

**Linear system solver**: Tile Cholesky (separate stretch/bend systems)
**Constraint coupling**: Full coupling within each 3D constraint system

---

## 09_cholesky_cosserat_rod_multitile.py - ⚠️ PARTIALLY WORKING

Block-Jacobi Cholesky for longer rods with 3-DOF tangent-space formulation. 32 constraints × 3 = 96 scalar DOFs per tile.

**Linear system solver**: Tile Cholesky (Block-Jacobi, 96×96 tiles)
**Constraint coupling**: Full 3D coupling within tiles
**Known issue**: Explodes when changing rest shape via UI

---

## 09_cholesky_cosserat_rod_multitile_combined.py - STATUS UNKNOWN

Similar to 09 but with combined stretch+bend in same solve. Status needs verification.

---

## 10_sim_aorta.py - STATUS UNKNOWN

Aorta simulation example. Status needs verification.

---

## 11_cable_aorta.py - STATUS UNKNOWN

Cable in aorta simulation. Status needs verification.

---

## 12_cable_sim.py - STATUS UNKNOWN

Cable simulation example. Status needs verification.
