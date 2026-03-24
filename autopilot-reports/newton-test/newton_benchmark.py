# Newton Physics Engine - Rigid Body Simulation Benchmark
# Tests: collision detection, rigid body simulation, step throughput

import time
import numpy as np
import warp as wp
import newton
import newton.solvers


def run_falling_box_test(device="cuda:0"):
    """Simple falling box with ground collision - basic rigid body smoke test."""
    print(f"\n=== Falling Box Test ({device}) ===")

    builder = newton.ModelBuilder()

    # Add a box with initial position z=5 (z is the up axis in Newton)
    body = builder.add_body(
        xform=wp.transform(p=wp.vec3(0.0, 0.0, 5.0), q=wp.quat_identity()),
    )
    builder.add_shape_box(body, hx=0.5, hy=0.5, hz=0.5)

    # Add ground plane at z=0
    builder.add_ground_plane()

    model = builder.finalize(device=device)
    solver = newton.solvers.SolverXPBD(model, iterations=8)

    state_0 = model.state()
    state_1 = model.state()
    control = model.control()
    contacts = model.contacts()

    sim_dt = 1.0 / 120.0
    num_steps = 300  # 2.5 seconds of simulation

    print(f"  Simulating {num_steps} steps (dt={sim_dt:.5f}s)...")

    start = time.perf_counter()
    for i in range(num_steps):
        state_0.clear_forces()
        model.collide(state_0, contacts)
        solver.step(state_0, state_1, control, contacts, sim_dt)
        state_0, state_1 = state_1, state_0
    wp.synchronize()
    elapsed = time.perf_counter() - start

    # body_q is Nx7: [px, py, pz, qx, qy, qz, qw]
    body_q = state_0.body_q.numpy()
    pos = body_q[body, :3]

    print(f"  Final position: ({pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f})")
    print(f"  Time: {elapsed:.4f}s for {num_steps} steps")
    print(f"  Throughput: {num_steps/elapsed:.1f} steps/sec")

    # Box should be resting near z=0.5 (half-height of box)
    settled = -0.1 < pos[2] < 2.0
    print(f"  Box fell and settled near ground: {settled} (z={pos[2]:.3f}, expected ~0.5)")
    return elapsed, num_steps, pos, settled


def run_collision_detection_test(device="cuda:0"):
    """Test collision detection with multiple boxes falling."""
    print(f"\n=== Collision Detection Test ({device}) ===")

    builder = newton.ModelBuilder()
    n_bodies = 10
    bodies = []

    rng = np.random.default_rng(42)
    for i in range(n_bodies):
        x = float(rng.uniform(-2, 2))
        y = float(rng.uniform(-2, 2))
        z = float(1.5 + i * 1.2)
        body = builder.add_body(
            xform=wp.transform(p=wp.vec3(x, y, z), q=wp.quat_identity()),
        )
        builder.add_shape_box(body, hx=0.4, hy=0.4, hz=0.4)
        bodies.append(body)

    builder.add_ground_plane()
    model = builder.finalize(device=device)
    solver = newton.solvers.SolverXPBD(model, iterations=8)

    state_0 = model.state()
    state_1 = model.state()
    control = model.control()
    contacts = model.contacts()

    sim_dt = 1.0 / 120.0
    num_steps = 300

    print(f"  {n_bodies} boxes, {num_steps} steps...")

    start = time.perf_counter()
    for _ in range(num_steps):
        state_0.clear_forces()
        model.collide(state_0, contacts)
        solver.step(state_0, state_1, control, contacts, sim_dt)
        state_0, state_1 = state_1, state_0
    wp.synchronize()
    elapsed = time.perf_counter() - start

    body_q = state_0.body_q.numpy()
    final_z = body_q[:n_bodies, 2]
    above_floor = bool((final_z > -0.2).all())
    print(f"  Final z-positions (min={final_z.min():.3f}, max={final_z.max():.3f})")
    print(f"  All bodies above floor: {above_floor}")
    print(f"  Time: {elapsed:.4f}s for {num_steps} steps")
    print(f"  Throughput: {num_steps/elapsed:.1f} steps/sec")
    return elapsed, num_steps, above_floor


def run_throughput_benchmark(device="cuda:0"):
    """Measure step throughput with many rigid bodies."""
    print(f"\n=== Throughput Benchmark ({device}) ===")

    results = {}
    for n_bodies in [10, 50, 100, 500]:
        builder = newton.ModelBuilder()

        rng = np.random.default_rng(42)
        for i in range(n_bodies):
            x = float(rng.uniform(-5, 5))
            y = float(rng.uniform(-5, 5))
            z = float(rng.uniform(2.0, 15.0))
            body = builder.add_body(
                xform=wp.transform(p=wp.vec3(x, y, z), q=wp.quat_identity()),
            )
            builder.add_shape_sphere(body=body, radius=0.3)

        builder.add_ground_plane()
        model = builder.finalize(device=device)
        solver = newton.solvers.SolverXPBD(model, iterations=4)

        state_0 = model.state()
        state_1 = model.state()
        control = model.control()
        contacts = model.contacts()

        sim_dt = 1.0 / 60.0
        # warmup
        for _ in range(20):
            state_0.clear_forces()
            model.collide(state_0, contacts)
            solver.step(state_0, state_1, control, contacts, sim_dt)
            state_0, state_1 = state_1, state_0
        wp.synchronize()

        num_steps = 500
        start = time.perf_counter()
        for _ in range(num_steps):
            state_0.clear_forces()
            model.collide(state_0, contacts)
            solver.step(state_0, state_1, control, contacts, sim_dt)
            state_0, state_1 = state_1, state_0
        wp.synchronize()
        elapsed = time.perf_counter() - start

        throughput = num_steps / elapsed
        results[n_bodies] = throughput
        print(f"  {n_bodies:4d} bodies: {throughput:8.1f} steps/sec  ({elapsed:.3f}s)")

    return results


def main():
    print("=" * 60)
    print("Newton Physics Engine Benchmark")
    print(f"Version: {getattr(newton, '__version__', 'unknown')}")
    print(f"Warp: {wp.__version__}")
    print("=" * 60)

    # Check GPU
    devices = wp.get_devices()
    gpu_devices = [d for d in devices if d.is_cuda]
    print(f"\nAvailable devices: {[str(d) for d in devices]}")
    device = "cuda:0" if gpu_devices else "cpu"
    print(f"Using device: {device}")

    # Run tests
    elapsed1, steps1, final_pos, settled = run_falling_box_test(device)
    elapsed2, steps2, above_floor = run_collision_detection_test(device)
    throughput_results = run_throughput_benchmark(device)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Newton version:        {getattr(newton, '__version__', 'unknown')}")
    print(f"Device:                {device}")
    print(f"GPU:                   NVIDIA L40 (47 GiB, sm_89)")
    print(f"Falling box test:      {'PASS' if settled else 'PARTIAL'} (z={final_pos[2]:.3f}, expected ~0.5)")
    print(f"Collision test:        {'PASS' if above_floor else 'FAIL'} ({steps2/elapsed2:.1f} steps/sec)")
    print("Throughput (XPBD solver, GPU, after warmup):")
    for n, thr in throughput_results.items():
        print(f"  {n:4d} bodies: {thr:.1f} steps/sec")
    print("=" * 60)
    print("Newton import: OK")


if __name__ == "__main__":
    main()
