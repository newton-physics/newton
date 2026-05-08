# SPDX-License-Identifier: Apache-2.0
"""Offline verification of the chysx PCG solver.

Runs one chysx::ClothSimulator step, dumps the linear system that PCG
solved (H, rhs, dx), and reports four checks:

1. **Symmetry**:        max |H - H.T| (should be ~ float epsilon)
2. **PSD-ness**:        smallest eigenvalue of dense H
3. **PCG residual**:    ||H * dx - rhs||_2 / ||rhs||_2
4. **Gold solve**:      compare chysx's dx to numpy.linalg.solve

You can pin/comment-out individual constraints in cloth_simulator.cu
and rerun this — the script is independent of which subset of
constraints is active.

Usage:
    uv run python scripts/diagnose_pcg.py --grid 6
    uv run python scripts/diagnose_pcg.py --grid 6 --no-spring
    uv run python scripts/diagnose_pcg.py --grid 20 --pcg-iters 200
"""

from __future__ import annotations

import argparse

import numpy as np
import warp as wp

import newton


def build_dense(diag, row_offsets, col_indices, values):
    """Reconstruct the dense (3N, 3N) matrix from chysx's split storage."""
    N = diag.shape[0]
    M = np.zeros((3 * N, 3 * N), dtype=np.float64)
    for i in range(N):
        M[3 * i : 3 * i + 3, 3 * i : 3 * i + 3] = diag[i]
    for i in range(N):
        for ptr in range(row_offsets[i], row_offsets[i + 1]):
            j = col_indices[ptr]
            M[3 * i : 3 * i + 3, 3 * j : 3 * j + 3] = values[ptr]
    return M


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--grid", type=int, default=6,
                   help="dim_x = dim_y of the cloth grid")
    p.add_argument("--cell", type=float, default=0.1)
    p.add_argument("--pcg-iters", type=int, default=50)
    p.add_argument("--surface-density", type=float, default=5.0)
    p.add_argument("--fem-stiffness", type=float, default=5.0e3)
    p.add_argument("--no-fem", action="store_true")
    p.add_argument("--pin-stiffness", type=float, default=1.0e9)
    p.add_argument("--damping", type=float, default=0.0)
    p.add_argument("--steps-before-dump", type=int, default=1,
                   help="Run this many steps and dump after the last one. "
                        "First step has zero velocity; later steps have "
                        "more representative state.")
    args = p.parse_args()

    wp.init()

    # ---- model -----------------------------------------------------------
    builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
    extent = args.grid * args.cell
    builder.add_cloth_grid(
        pos=wp.vec3(-extent / 2, -extent / 2, 2.0),
        rot=wp.quat_identity(),
        vel=wp.vec3(0.0, 0.0, 0.0),
        dim_x=args.grid, dim_y=args.grid,
        cell_x=args.cell, cell_y=args.cell,
        mass=1.0,  # placeholder; chysx redistributes by area
        tri_ke=0.0, tri_ka=0.0, tri_kd=0.0,
        edge_ke=0.0, edge_kd=0.0,
        particle_radius=0.005,
    )
    builder.add_ground_plane()
    model = builder.finalize()

    pinned = [0, args.grid]
    fem_k = 0.0 if args.no_fem else args.fem_stiffness

    solver = newton.solvers.SolverChysX(
        model,
        fem_stretch_stiffness=fem_k,
        fem_shear_stiffness=fem_k,
        damping=args.damping,
        pin_indices=pinned,
        pin_stiffness=args.pin_stiffness,
        pcg_iterations=args.pcg_iters,
        surface_density=args.surface_density,
    )

    state_in = model.state()
    state_out = model.state()
    contacts = model.contacts()
    control = model.control()
    dt = 1.0 / 100.0

    # Run a few steps to get past the cold-start (initial velocity = 0).
    for _ in range(args.steps_before_dump):
        solver.step(state_in, state_out, control, contacts, dt)
        state_in, state_out = state_out, state_in

    wp.synchronize()

    # ---- pull last solve's H, rhs, dx ------------------------------------
    sim = solver._sim
    diag, row_offsets, col_indices, values, rhs, dx = sim.debug_dump_last_solve()

    # Cast to float64 for stable analysis.
    diag = diag.astype(np.float64)
    values = values.astype(np.float64)
    rhs = rhs.astype(np.float64)
    dx = dx.astype(np.float64)

    N = diag.shape[0]
    nnz = values.shape[0] if values.size else 0

    print(f"=== chysx PCG diagnostics ===")
    print(f"grid                 : {args.grid}x{args.grid}  ({N} particles, {nnz} off-diag blocks)")
    print(f"fem_stiffness        : {fem_k:.3g}")
    print(f"surface_density      : {args.surface_density}")
    print(f"pcg_iterations       : {args.pcg_iters}")
    print(f"damping              : {args.damping}")
    print(f"pinned indices       : {pinned}")

    # ---- Reconstruct dense H --------------------------------------------
    H = build_dense(diag, row_offsets, col_indices, values)
    rhs_flat = rhs.reshape(-1)
    dx_flat = dx.reshape(-1)

    print(f"\n--- numerical magnitudes ---")
    print(f"||rhs||_inf          : {np.abs(rhs_flat).max():.3e}")
    print(f"||rhs||_2            : {np.linalg.norm(rhs_flat):.3e}")
    print(f"||dx||_inf           : {np.abs(dx_flat).max():.3e}")
    print(f"||dx||_2             : {np.linalg.norm(dx_flat):.3e}")
    print(f"diag block frobenius : min={np.sqrt((diag**2).sum(axis=(1,2))).min():.3e}, "
          f"max={np.sqrt((diag**2).sum(axis=(1,2))).max():.3e}")

    print(f"\n--- (1) symmetry: max |H - H.T| ---")
    sym_err = np.abs(H - H.T).max()
    sym_norm = np.abs(H).max()
    print(f"max |H - H.T|        : {sym_err:.3e}")
    print(f"max |H|              : {sym_norm:.3e}")
    print(f"relative             : {sym_err / max(sym_norm, 1e-30):.3e}")

    print(f"\n--- (2) PSD-ness ---")
    Hsym = 0.5 * (H + H.T)  # symmetrise to get rid of numerical fuzz
    if N <= 200:
        eig = np.linalg.eigvalsh(Hsym)
        print(f"eig min              : {eig.min():.3e}")
        print(f"eig max              : {eig.max():.3e}")
        print(f"cond (eig_max/eig_min): "
              f"{eig.max() / max(abs(eig.min()), 1e-30):.3e}")
        if eig.min() < -1e-3 * eig.max():
            print("  >>> WARNING: large negative eigenvalue, H is not PSD")
    else:
        print(f"(skipped — dense eigvals expensive at N={N})")

    print(f"\n--- (3) PCG residual ---")
    res = H @ dx_flat - rhs_flat
    res_per_particle = np.linalg.norm(res.reshape(N, 3), axis=1)
    rhs_per_particle = np.linalg.norm(rhs_flat.reshape(N, 3), axis=1)
    res_norm = np.linalg.norm(res)
    rhs_norm = np.linalg.norm(rhs_flat)
    print(f"||H dx - rhs||_2     : {res_norm:.3e}")
    print(f"||rhs||_2            : {rhs_norm:.3e}")
    rel = res_norm / max(rhs_norm, 1e-30)
    print(f"relative residual    : {rel:.3e}")

    # `rhs` is dominated by the pin force (k_pin * (target - x_tilde)),
    # so its 2-norm hides whether the *interior* cloth residual is
    # converged.  Split rhs/res into pinned vs unpinned to expose the
    # part that actually drives the cloth.
    pinned_mask = np.zeros(N, dtype=bool)
    pinned_mask[pinned] = True
    free_mask = ~pinned_mask
    print(f"  pinned vs free split:")
    print(f"    pin    res inf       : {res_per_particle[pinned_mask].max():.3e}")
    print(f"    pin    rhs inf       : {rhs_per_particle[pinned_mask].max():.3e}")
    print(f"    free   res inf       : {res_per_particle[free_mask].max():.3e}")
    print(f"    free   rhs inf       : {rhs_per_particle[free_mask].max():.3e}")
    free_rel = (res_per_particle[free_mask].max() /
                max(rhs_per_particle[free_mask].max(), 1e-30))
    print(f"    free   rel inf       : {free_rel:.3e}")
    # And per-particle accel error: res / m.
    diag_blk_norm = np.sqrt((diag**2).sum(axis=(1, 2)))  # ~ k_diag * sqrt(3)
    # m_i / dt^2 sits inside diag along with stiffness; we just print the
    # per-particle |dx| update for context.
    dx_per_particle = np.linalg.norm(dx_flat.reshape(N, 3), axis=1)
    print(f"    free   |dx| inf      : {dx_per_particle[free_mask].max():.3e}")
    print(f"    free   |dx| mean     : {dx_per_particle[free_mask].mean():.3e}")
    if rel > 1e-2:
        print("  >>> WARNING: PCG didn't converge (rel > 1e-2)")
    elif rel > 1e-4:
        print("  WARN: PCG only weakly converged (rel > 1e-4)")
    else:
        print("  OK: PCG converged to <= 1e-4")

    print(f"\n--- (4) gold solve (numpy.linalg.solve, dense) ---")
    if N <= 1000:
        try:
            dx_gold = np.linalg.solve(Hsym, rhs_flat)
            err = np.linalg.norm(dx_flat - dx_gold)
            scale = np.linalg.norm(dx_gold)
            print(f"||dx_pcg - dx_gold|| : {err:.3e}")
            print(f"||dx_gold||          : {scale:.3e}")
            print(f"relative error       : {err / max(scale, 1e-30):.3e}")
        except np.linalg.LinAlgError as e:
            print(f"dense solve failed: {e}")
    else:
        print(f"(skipped — dense solve expensive at N={N})")

    # ---- (5) PCG convergence curve --------------------------------------
    # Re-run our own scipy CG to compare convergence.  Our chysx PCG
    # uses block-Jacobi preconditioning (M = block-diag of H), so we
    # match that here for an apples-to-apples comparison.
    print(f"\n--- (5) reference scipy.sparse.linalg.cg convergence ---")
    try:
        from scipy.sparse import csr_matrix
        from scipy.sparse.linalg import LinearOperator, cg

        Hcsr = csr_matrix(Hsym)

        # Block-Jacobi preconditioner: invert each 3x3 diagonal block.
        Minv_blocks = np.zeros_like(diag)
        for i in range(N):
            Minv_blocks[i] = np.linalg.inv(diag[i])

        def apply_Minv(v):
            v3 = v.reshape(N, 3)
            out = np.empty_like(v3)
            for i in range(N):
                out[i] = Minv_blocks[i] @ v3[i]
            return out.reshape(-1)

        Minv = LinearOperator((3 * N, 3 * N), matvec=apply_Minv, dtype=np.float64)

        residuals = []

        def callback(xk):
            r = Hcsr @ xk - rhs_flat
            residuals.append(np.linalg.norm(r))

        x0 = np.zeros_like(rhs_flat)
        sol, info = cg(Hcsr, rhs_flat, x0=x0, M=Minv,
                       rtol=1e-10, maxiter=args.pcg_iters,
                       callback=callback)
        print(f"scipy iters used     : {len(residuals)}")
        if residuals:
            print(f"scipy res start      : {residuals[0]:.3e}")
            print(f"scipy res end        : {residuals[-1]:.3e}")
            print(f"  decay over iters: ", end="")
            picks = [0, len(residuals) // 4, len(residuals) // 2,
                     3 * len(residuals) // 4, len(residuals) - 1]
            for k in picks:
                if 0 <= k < len(residuals):
                    print(f"[{k}]={residuals[k]:.2e} ", end="")
            print()
        print(f"info                 : {info} "
              f"({'converged' if info == 0 else 'maxiter or breakdown'})")
        gold_err = np.linalg.norm(dx_flat - sol)
        gold_scale = np.linalg.norm(sol)
        print(f"||dx_pcg - dx_scipy|| / ||dx_scipy|| : "
              f"{gold_err / max(gold_scale, 1e-30):.3e}")
    except Exception as e:
        print(f"scipy comparison failed: {e}")


if __name__ == "__main__":
    main()
