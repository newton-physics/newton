# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Benchmark Newton simulation step throughput on GPU."""

import json
import time

import warp as wp

import newton


def build_scene(num_bodies: int, device: str):
    """Build a scene with num_bodies spheres dropped onto a ground plane."""
    builder = newton.ModelBuilder()
    builder.add_ground_plane()

    spacing = 1.5
    cols = int(num_bodies**0.5) + 1
    for i in range(num_bodies):
        row = i // cols
        col = i % cols
        pos = wp.vec3(col * spacing, row * spacing, 2.0 + (i % 5) * 0.5)
        body = builder.add_body(
            xform=wp.transform(p=pos, q=wp.quat_identity()),
        )
        builder.add_shape_sphere(body, radius=0.3)

    model = builder.finalize(device=device)
    return model


def benchmark(num_bodies: int, num_steps: int, device: str):
    """Run benchmark and return timing results."""
    model = build_scene(num_bodies, device)
    solver = newton.solvers.SolverXPBD(model, iterations=10)

    state_0 = model.state()
    state_1 = model.state()
    control = model.control()
    contacts = model.contacts()

    dt = 1.0 / 1000.0  # 1ms timestep

    # Warmup
    for _ in range(10):
        state_0.clear_forces()
        model.collide(state_0, contacts)
        solver.step(state_0, state_1, control, contacts, dt)
        state_0, state_1 = state_1, state_0

    wp.synchronize()

    # Timed run
    t_start = time.perf_counter()
    for _ in range(num_steps):
        state_0.clear_forces()
        model.collide(state_0, contacts)
        solver.step(state_0, state_1, control, contacts, dt)
        state_0, state_1 = state_1, state_0

    wp.synchronize()
    t_elapsed = time.perf_counter() - t_start

    steps_per_sec = num_steps / t_elapsed
    us_per_step = (t_elapsed / num_steps) * 1e6

    return {
        "num_bodies": num_bodies,
        "num_steps": num_steps,
        "total_time_s": round(t_elapsed, 4),
        "steps_per_sec": round(steps_per_sec, 1),
        "us_per_step": round(us_per_step, 1),
        "device": device,
    }


def main():
    device = "cuda:0" if wp.is_cuda_available() else "cpu"
    print(f"Device: {device}")
    print(f"Newton version: {getattr(newton, '__version__', 'unknown')}")
    print()

    configs = [
        (1, 1000),
        (10, 1000),
        (50, 500),
        (100, 500),
        (200, 300),
        (500, 200),
    ]

    results = []
    print(f"{'Bodies':>8} {'Steps':>8} {'Time(s)':>10} {'Steps/s':>12} {'us/step':>10}")
    print("-" * 58)

    for num_bodies, num_steps in configs:
        r = benchmark(num_bodies, num_steps, device)
        results.append(r)
        print(
            f"{r['num_bodies']:>8} {r['num_steps']:>8} {r['total_time_s']:>10.4f} "
            f"{r['steps_per_sec']:>12.1f} {r['us_per_step']:>10.1f}"
        )

    # Also benchmark with MuJoCo Warp solver if on CUDA
    mujoco_results = []
    if device == "cuda:0":
        print("\n--- MuJoCo Warp Solver ---")
        print(f"{'Bodies':>8} {'Steps':>8} {'Time(s)':>10} {'Steps/s':>12} {'us/step':>10}")
        print("-" * 58)

        for num_bodies, num_steps in [(1, 1000), (10, 1000), (50, 500), (100, 500)]:
            model = build_scene(num_bodies, device)
            try:
                solver = newton.solvers.SolverMuJoCo(model)
            except Exception as e:
                print(f"  MuJoCo Warp solver not available for {num_bodies} bodies: {e}")
                continue

            state_0 = model.state()
            state_1 = model.state()
            control = model.control()
            contacts = model.contacts()
            dt = 1.0 / 1000.0

            # Warmup
            for _ in range(10):
                state_0.clear_forces()
                model.collide(state_0, contacts)
                solver.step(state_0, state_1, control, contacts, dt)
                state_0, state_1 = state_1, state_0
            wp.synchronize()

            t_start = time.perf_counter()
            for _ in range(num_steps):
                state_0.clear_forces()
                model.collide(state_0, contacts)
                solver.step(state_0, state_1, control, contacts, dt)
                state_0, state_1 = state_1, state_0
            wp.synchronize()
            t_elapsed = time.perf_counter() - t_start

            r = {
                "num_bodies": num_bodies,
                "num_steps": num_steps,
                "total_time_s": round(t_elapsed, 4),
                "steps_per_sec": round(num_steps / t_elapsed, 1),
                "us_per_step": round((t_elapsed / num_steps) * 1e6, 1),
                "device": device,
                "solver": "mujoco_warp",
            }
            mujoco_results.append(r)
            print(
                f"{r['num_bodies']:>8} {r['num_steps']:>8} {r['total_time_s']:>10.4f} "
                f"{r['steps_per_sec']:>12.1f} {r['us_per_step']:>10.1f}"
            )

    all_results = {"xpbd": results, "mujoco_warp": mujoco_results}
    with open("autopilot-reports/newton-test/benchmark_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print("\nResults saved to autopilot-reports/newton-test/benchmark_results.json")
    return all_results


if __name__ == "__main__":
    main()
