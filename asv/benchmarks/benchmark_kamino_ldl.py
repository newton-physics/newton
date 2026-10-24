# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Compare Kamino DVI joint backends on existing robot examples.

Run from the repository root with ``uv run --no-sync python
asv/benchmarks/benchmark_kamino_ldl.py --output asv/results/kamino_ldl``.
Each case runs in a fresh process, sequentially, to avoid GPU contention.
Internal linalg APIs are used only to measure and validate isolated factors.
"""

import argparse
import hashlib
import importlib
import itertools
import json
import os
import platform
import random
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import warp as wp

from newton._src.solvers.kamino._src.linalg import (
    DenseLinearOperatorData,
    DenseSquareMultiLinearInfo,
    LDLBlockSparseSolver,
    LLTBlockedRCMSolver,
    LLTBlockedSolver,
)
from newton._src.solvers.kamino._src.solvers.dvi.kernels import BILATERAL_FAILED_PIVOT_SHIFT
from newton._src.solvers.kamino._src.solvers.dvi.sparse import _assemble_sparse_bilateral_block
from newton._src.solvers.kamino._src.solvers.dvi.sparse_factor import joint_block_pattern
from newton._src.utils.download_assets import NEWTON_ASSETS_REF
from newton.viewer import ViewerNull

BACKENDS = ("LDL-MECA", "LDL-RCM", "LLTB", "LLTBRCM")
ROBOTS = {
    "h1": "robot.example_robot_h1",
    "g1": "robot.example_robot_g1",
    "dr_legs": "kamino.example_kamino_robot_dr_legs",
}


def make_example(robot, worlds, backend):
    cls = importlib.import_module(f"newton.examples.{ROBOTS[robot]}").Example
    flags = ["--world-count", str(worlds), "--bilateral-solver-type", backend.split("-")[0]]
    flags += ["--dynamics-solver", "dvi"] if robot == "dr_legs" else ["--solver", "kamino"]
    if backend.startswith("LDL"):
        flags += ["--ldl-ordering", backend.split("-")[1].lower()]
    example = cls(ViewerNull(), cls.create_parser().parse_args(flags))
    if example.graph is None:
        raise RuntimeError("This benchmark requires CUDA graph capture")
    actual = example.solver._solver_kamino.solver_fd._bilateral_solver
    if backend.startswith("LDL"):
        assert isinstance(actual, LDLBlockSparseSolver)
        assert actual.ordering == backend.split("-")[1].lower()
    else:
        assert isinstance(actual, LLTBlockedSolver if backend == "LLTB" else LLTBlockedRCMSolver)
    return example


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def make_fixture(robot, output):
    """Freeze one physical, normalized joint matrix for all backend microbenchmarks."""
    example = make_example(robot, 1, "LLTB")
    example.step()
    impl = example.solver._solver_kamino
    path = impl.solver_fd._sparse_path
    operator = path.data.bilateral_operator
    matrix = wp.zeros(operator.info.total_mat_size, dtype=wp.float32)
    _assemble_sparse_bilateral_block(path, impl.problem_fd, matrix)
    n = int(operator.info.dim.numpy()[0])
    rows, edges = joint_block_pattern(impl._model, [n])
    a = matrix.numpy().reshape(n, n)
    np.testing.assert_allclose(a, a.T, atol=1e-7)
    np.linalg.cholesky(a.astype(np.float64))
    np.save(output / f"{robot}_matrix.npy", a)
    write_json(
        output / f"{robot}_fixture.json",
        {
            "dimension": n,
            "rows": rows[0],
            "edges": edges[0],
            "condition_number": float(np.linalg.cond(a.astype(np.float64))),
        },
    )


def measure(action, *, repeats, samples, validate=None):
    """Record synchronized wall and CUDA-event times per graph launch."""
    start, end = wp.Event(enable_timing=True), wp.Event(enable_timing=True)
    wall, gpu = [], []
    for _ in range(samples):
        wp.synchronize_device()
        before = time.perf_counter()
        wp.record_event(start)
        for _ in range(repeats):
            action()
        wp.record_event(end)
        wp.synchronize_device()
        wall.append((time.perf_counter() - before) * 1000 / repeats)
        gpu.append(wp.get_event_elapsed_time(start, end) / repeats)
        if validate:
            validate()
    return {"wall_ms": wall, "gpu_ms": gpu, "median_wall_ms": float(np.median(wall))}


def capture(action):
    action()  # Compile before capture and timing.
    with wp.ScopedCapture() as captured:
        action()
    for _ in range(10):
        wp.capture_launch(captured.graph)
    wp.synchronize_device()
    return lambda: wp.capture_launch(captured.graph)


def microbenchmark(robot, worlds, backend, output, samples):
    """Use identical frozen A and b, including native sparse input for LDL."""
    fixture = json.loads((output / f"{robot}_fixture.json").read_text())
    a = np.load(output / f"{robot}_matrix.npy")
    n = len(a)
    capacity = (n + 31) // 32 * 32 if backend == "LLTBRCM" else n
    info = DenseSquareMultiLinearInfo()
    info.finalize([capacity] * worlds, dtype=wp.float32, device="cuda:0")
    info.dim = wp.full(worlds, n, dtype=wp.int32)
    padded = np.zeros((capacity, capacity), dtype=np.float32)
    padded[:n, :n] = a
    matrix = wp.array(np.tile(padded.ravel(), worlds), dtype=wp.float32)
    operator = DenseLinearOperatorData(info=info, mat=matrix)
    kwargs = {"failed_pivot_shift": BILATERAL_FAILED_PIVOT_SHIFT, "device": wp.get_device("cuda:0")}
    before = time.perf_counter()
    if backend.startswith("LDL"):
        solver = LDLBlockSparseSolver(
            operator=operator,
            block_rows=[fixture["rows"]] * worlds,
            block_edges=[fixture["edges"]] * worlds,
            ordering=backend.split("-")[1].lower(),
            **kwargs,
        )
    elif backend == "LLTB":
        solver = LLTBlockedSolver(operator=operator, factorize_block_size=64, solve_block_dim=256, **kwargs)
    else:
        solver = LLTBlockedRCMSolver(
            operator=operator, capacity_stride=True, parallel_factorization=robot == "dr_legs", **kwargs
        )
    wp.synchronize_device()
    allocation_ms = (time.perf_counter() - before) * 1000
    # Gather once into native sparse input; exclude assembly for every backend.
    solver.compute(matrix)
    factor = solver.factorize_values if backend.startswith("LDL") else lambda: solver.compute(matrix)
    b0 = np.random.default_rng(20261024).normal(size=n).astype(np.float32)
    rhs = np.zeros((worlds, capacity), dtype=np.float32)
    rhs[:, :n] = b0
    b, x = wp.array(rhs.ravel()), wp.zeros(worlds * capacity, dtype=wp.float32)
    factor_times = measure(capture(factor), repeats=100, samples=samples)
    solve_times = measure(capture(lambda: solver.solve(b, x)), repeats=100, samples=samples)
    solutions = x.numpy().reshape(worlds, capacity)[:, :n].astype(np.float64)
    reference = np.linalg.solve(a.astype(np.float64), b0.astype(np.float64))
    residuals = solutions @ a.astype(np.float64).T - b0
    relative_residual = float(np.max(np.linalg.norm(residuals, axis=1)) / np.linalg.norm(b0))
    relative_error = float(np.max(np.linalg.norm(solutions - reference, axis=1)) / np.linalg.norm(reference))
    backward_error = float(
        np.max(
            np.max(np.abs(residuals), axis=1)
            / (
                np.linalg.norm(a.astype(np.float64), ord=np.inf) * np.max(np.abs(solutions), axis=1)
                + np.linalg.norm(b0, ord=np.inf)
            )
        )
    )
    # Closed loops can be severely ill conditioned. Gate normwise backward
    # stability, and retain forward error and relative residual as diagnostics.
    if not np.isfinite(solutions).all() or backward_error > 1e-6:
        raise AssertionError(f"Linear solve failed: backward error={backward_error}")
    if backend.startswith("LDL"):
        factor_bytes = solver.factors.size * 4 + solver.diagonal.size * 4
        blocks = solver.factors.shape[0] // worlds
        fill = blocks - len(fixture["rows"]) - len(fixture["edges"])
        min_d = float(solver.diagonal.numpy().min())
        if min_d <= 0:
            raise AssertionError("LDL has a non-positive diagonal")
        retry_worlds = int(np.count_nonzero(solver.failed.numpy()))
    else:
        factor_bytes, blocks, fill, min_d = solver.L.size * 4, None, None, None
        retry_worlds = int(np.count_nonzero(solver._retry_dim.numpy()))
    return {
        "factor": factor_times,
        "solve": solve_times,
        "allocation_ms": allocation_ms,
        "dimension": n,
        "capacity": capacity,
        "factor_bytes": factor_bytes,
        "blocks_per_world": blocks,
        "fill_blocks_per_world": fill,
        "minimum_d": min_d,
        "retry_worlds": retry_worlds,
        "relative_residual": relative_residual,
        "relative_solution_error": relative_error,
        "backward_error_inf": backward_error,
    }


def run_case(args):
    robot, worlds, backend = args.robots[0], args.worlds[0], args.backends[0]
    example = make_example(robot, worlds, backend)
    initial = example.state_0.body_q.numpy()
    for _ in range(args.warmup):
        example.step()
    wp.synchronize_device()

    def validate():
        example.test_final()
        if hasattr(example, "test_post_step"):
            example.test_post_step()
        if not np.isfinite(example.state_0.body_q.numpy()).all():
            raise AssertionError("Non-finite body pose")
        if not np.isfinite(example.state_0.body_qd.numpy()).all():
            raise AssertionError("Non-finite body velocity")
        if robot in ("h1", "g1"):
            if np.abs(example.state_0.body_qd.numpy()).max() >= 0.015:
                raise AssertionError("Robot velocity exceeds the settled-state bound")
            if example.state_0.body_q.numpy()[:, 2].min() <= 0.0:
                raise AssertionError("A robot body is below the ground")

    validate()
    frame = measure(example.step, repeats=args.frames, samples=args.samples, validate=validate)
    positions = example.state_0.body_q.numpy()
    velocities = example.state_0.body_qd.numpy()
    bodies_per_world = example.model.body_count // worlds
    data = {
        "robot": robot,
        "worlds": worlds,
        "backend": backend,
        "bodies_per_world": bodies_per_world,
        "substeps": example.sim_substeps,
        "frame_dt": example.frame_dt,
        "frame": frame,
        "max_abs_velocity": float(np.abs(velocities).max()),
        "min_body_height": float(positions[:, 2].min()),
        "max_translation_from_initial": float(np.linalg.norm(positions[:, :3] - initial[:, :3], axis=1).max()),
        "validation": "passed",
        "first_world_body_q": positions[:bodies_per_world].tolist(),
        "first_world_body_qd": velocities[:bodies_per_world].tolist(),
        "micro": microbenchmark(robot, worlds, backend, args.output, args.samples),
    }
    write_json(args.output / f"{robot}_{worlds}_{backend}.json", data)
    print(f"{robot} worlds={worlds} {backend}: {frame['median_wall_ms']:.3f} ms/frame", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("asv/results/kamino_ldl"))
    parser.add_argument("--robots", nargs="+", choices=ROBOTS, default=list(ROBOTS))
    parser.add_argument("--worlds", nargs="+", type=int, default=[1, 32, 256])
    parser.add_argument("--backends", nargs="+", choices=BACKENDS, default=list(BACKENDS))
    parser.add_argument("--warmup", type=int, default=500)
    parser.add_argument("--frames", type=int, default=100)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--worker", choices=("fixture", "case"), help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    wp.config.enable_backward = False
    wp.config.log_level = wp.LOG_WARNING
    if args.worker:
        with wp.ScopedDevice("cuda:0"):
            if args.worker == "fixture":
                make_fixture(args.robots[0], args.output)
            else:
                run_case(args)
        return

    def command(*flags):
        return [sys.executable, str(Path(__file__).resolve()), "--output", str(args.output), *flags]

    def git(*flags):
        return subprocess.check_output(["git", *flags], text=True).strip()

    metadata = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "commit": git("rev-parse", "HEAD"),
        "branch": git("branch", "--show-current"),
        "tracked_diff_sha256": hashlib.sha256(git("diff", "HEAD").encode()).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "assets_ref": NEWTON_ASSETS_REF,
        "python": sys.version,
        "platform": platform.platform(),
        "warp": wp.__version__,
        "numpy": np.__version__,
        "cpu": subprocess.check_output(["lscpu"], text=True),
        "gpu": subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total,power.limit", "--format=csv"], text=True
        ),
        "parameters": {k: v for k, v in vars(args).items() if k not in ("output", "worker")},
        "thread_environment": {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS")},
    }
    write_json(args.output / "metadata.json", metadata)
    for robot in args.robots:
        with (args.output / f"{robot}_fixture.log").open("w") as log:
            subprocess.run(command("--worker", "fixture", "--robots", robot), stdout=log, stderr=log, check=True)
    cases = list(itertools.product(args.robots, args.worlds, args.backends))
    random.Random(20261024).shuffle(cases)
    failures = []
    for index, (robot, worlds, backend) in enumerate(cases, 1):
        name = f"{robot}_{worlds}_{backend}"
        print(f"[{index}/{len(cases)}] {name}", flush=True)
        with (args.output / f"{name}.log").open("w") as log:
            result = subprocess.run(
                command(
                    "--worker",
                    "case",
                    "--robots",
                    robot,
                    "--worlds",
                    str(worlds),
                    "--backends",
                    backend,
                    "--warmup",
                    str(args.warmup),
                    "--frames",
                    str(args.frames),
                    "--samples",
                    str(args.samples),
                ),
                stdout=log,
                stderr=log,
                check=False,
            )
        if result.returncode:
            failures.append({"case": name, "returncode": result.returncode})
            print(f"FAILED: {name}; see {name}.log", flush=True)
    write_json(args.output / "failures.json", failures)
    if failures:
        raise SystemExit(f"{len(failures)} benchmark cases failed")


if __name__ == "__main__":
    main()
