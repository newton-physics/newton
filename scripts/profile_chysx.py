# SPDX-License-Identifier: Apache-2.0
"""Headless workload for Nsight Systems / nsys profiling of chysx.

Runs the same physics setup as `example_chysx_hanging_cloth` but with
no viewer, no rendering, and a tight stepping loop, so the captured
timeline is dominated by chysx's actual GPU work and the NVTX ranges
in `cloth_simulator.cu` line up directly with kernel activity.

Typical usage:

    nsys profile -t cuda,nvtx,osrt ^
        --output=chysx_profile --force-overwrite=true ^
        uv run python scripts/profile_chysx.py --steps 300

Open the resulting `chysx_profile.nsys-rep` in the Nsight Systems UI
(`nsys-ui`) and look for the blue "chysx::cloth::step" range; inside
it you'll see the coloured sub-ranges (`step::predictor`,
`step::gradient`, `step::topology_rebuild`, `step::hessian`,
`step::pcg`, `step::finalize`) interleaved with the actual CUDA
kernel launches on the GPU row.
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import warp as wp

import newton


def build_model(grid_dim: int = 10, cell: float = 0.1) -> newton.Model:
    """Build the same hanging-cloth model the example uses."""
    builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
    z0 = 2.0
    builder.add_cloth_grid(
        pos=wp.vec3(-0.5, -0.5, z0),
        rot=wp.quat_identity(),
        vel=wp.vec3(0.0, 0.0, 0.0),
        dim_x=grid_dim,
        dim_y=grid_dim,
        cell_x=cell,
        cell_y=cell,
        mass=0.05,
        tri_ke=0.0,
        tri_ka=0.0,
        tri_kd=0.0,
        edge_ke=0.0,
        edge_kd=0.0,
        particle_radius=0.02,
    )
    builder.add_ground_plane()
    return builder.finalize()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=300,
                        help="Number of physics steps to run inside the timed region.")
    parser.add_argument("--warmup", type=int, default=30,
                        help="Steps to run before the timed region (kernel JIT, alloc, etc.).")
    parser.add_argument("--grid", type=int, default=10,
                        help="Cloth grid dimension (grid+1 particles per side).")
    parser.add_argument("--cell", type=float, default=0.1,
                        help="Cloth cell size in metres.")
    parser.add_argument("--pcg-iters", type=int, default=50,
                        help="Max PCG iterations per step.")
    args = parser.parse_args()

    model = build_model(grid_dim=args.grid, cell=args.cell)

    solver = newton.solvers.SolverChysX(
        model,
        spring_stiffness=5.0e2,
        fem_stretch_stiffness=5.0e2,
        damping=1.0,
        pin_indices=[0],
        pin_stiffness=1.0e6,
        pcg_iterations=args.pcg_iters,
    )

    state_0 = model.state()
    state_1 = model.state()
    control = model.control()
    contacts = model.contacts()

    dt = 1.0 / 60.0

    # Warmup — primes Warp's kernel cache, allocates work buffers, and
    # triggers the one-time `ensure_hessian_topology()` rebuild.  We
    # leave it outside the timed region so the steady-state cost shows
    # up cleanly in the timeline.
    for _ in range(args.warmup):
        state_0.clear_forces()
        solver.step(state_0, state_1, control, contacts, dt)
        state_0, state_1 = state_1, state_0

    wp.synchronize()
    t0 = time.perf_counter()
    for _ in range(args.steps):
        state_0.clear_forces()
        solver.step(state_0, state_1, control, contacts, dt)
        state_0, state_1 = state_1, state_0
    wp.synchronize()
    elapsed = time.perf_counter() - t0

    print(
        f"steps={args.steps}  elapsed={elapsed*1000:.1f} ms  "
        f"=> {args.steps/elapsed:.1f} fps  ({1000*elapsed/args.steps:.3f} ms/step)"
    )

    q = state_0.particle_q.numpy().reshape(-1, 3)
    qd = state_0.particle_qd.numpy().reshape(-1, 3)
    if not (np.isfinite(q).all() and np.isfinite(qd).all()):
        raise SystemExit("non-finite values in particle state")
    print(
        f"final mean z={q[:,2].mean():.3f}  far_corner_z={q[-1,2]:.3f}  "
        f"max|v|={np.linalg.norm(qd, axis=1).max():.3f}"
    )


if __name__ == "__main__":
    main()
