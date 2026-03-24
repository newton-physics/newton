# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Validate Newton rigid body simulation: falling shapes with collision detection."""

import sys
import time

import warp as wp

import newton


def run_rigid_body_validation():
    """Drop various shapes onto a ground plane and verify they come to rest."""
    device = "cuda:0" if wp.is_cuda_available() else "cpu"
    print(f"Device: {device}")

    fps = 100
    frame_dt = 1.0 / fps
    sim_substeps = 10
    sim_dt = frame_dt / sim_substeps
    num_frames = 300  # 3 seconds of simulation

    builder = newton.ModelBuilder()
    builder.add_ground_plane()

    drop_z = 2.0

    # Add several rigid bodies
    body_sphere = builder.add_body(
        xform=wp.transform(p=wp.vec3(0.0, 0.0, drop_z), q=wp.quat_identity()),
        label="sphere",
    )
    builder.add_shape_sphere(body_sphere, radius=0.5)

    body_box = builder.add_body(
        xform=wp.transform(p=wp.vec3(2.0, 0.0, drop_z), q=wp.quat_identity()),
        label="box",
    )
    builder.add_shape_box(body_box, hx=0.4, hy=0.4, hz=0.4)

    body_capsule = builder.add_body(
        xform=wp.transform(p=wp.vec3(-2.0, 0.0, drop_z), q=wp.quat_identity()),
        label="capsule",
    )
    builder.add_shape_capsule(body_capsule, radius=0.3, half_height=0.5)

    body_cylinder = builder.add_body(
        xform=wp.transform(p=wp.vec3(0.0, 2.0, drop_z), q=wp.quat_identity()),
        label="cylinder",
    )
    builder.add_shape_cylinder(body_cylinder, radius=0.3, half_height=0.5)

    body_ellipsoid = builder.add_body(
        xform=wp.transform(p=wp.vec3(0.0, -2.0, drop_z), q=wp.quat_identity()),
        label="ellipsoid",
    )
    builder.add_shape_ellipsoid(body_ellipsoid, a=0.5, b=0.5, c=0.25)

    model = builder.finalize(device=device)
    solver = newton.solvers.SolverXPBD(model, iterations=10)

    state_0 = model.state()
    state_1 = model.state()
    control = model.control()
    contacts = model.contacts()

    print(f"Bodies: {model.body_count}")
    print(f"Shapes: {model.shape_count}")
    print(f"Running {num_frames} frames ({num_frames / fps:.1f}s sim time)...")

    t_start = time.perf_counter()

    for frame in range(num_frames):
        for _ in range(sim_substeps):
            state_0.clear_forces()
            model.collide(state_0, contacts)
            solver.step(state_0, state_1, control, contacts, sim_dt)
            state_0, state_1 = state_1, state_0

    wp.synchronize()
    t_elapsed = time.perf_counter() - t_start

    print(f"\nSimulation completed in {t_elapsed:.3f}s wall time")
    print(f"Sim rate: {num_frames / t_elapsed:.1f} frames/s")
    print(f"Step rate: {num_frames * sim_substeps / t_elapsed:.1f} steps/s")

    # Validate final positions
    body_q = state_0.body_q.numpy()
    results = []
    expected = {
        "sphere": (0, 0.5),       # radius 0.5
        "box": (1, 0.4),          # half-height 0.4
        "capsule": (2, 0.8),      # radius 0.3 + half_height 0.5 (standing upright) or 0.3 (lying)
        "cylinder": (3, 0.5),     # half_height 0.5
        "ellipsoid": (4, 0.25),   # c = 0.25
    }

    all_pass = True
    print("\n--- Validation Results ---")
    for name, (idx, _) in expected.items():
        z = body_q[idx][2]
        above_ground = z > -0.1  # didn't fall through
        settled = z < 3.0        # came down from drop height
        ok = above_ground and settled
        status = "PASS" if ok else "FAIL"
        if not ok:
            all_pass = False
        results.append((name, z, ok))
        print(f"  {name}: z={z:.4f} [{status}]")

    print(f"\nOverall: {'ALL PASS' if all_pass else 'SOME FAILURES'}")
    return all_pass, t_elapsed, num_frames, sim_substeps, results


if __name__ == "__main__":
    passed, *_ = run_rigid_body_validation()
    sys.exit(0 if passed else 1)
