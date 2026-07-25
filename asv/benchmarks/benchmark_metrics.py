# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import math
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import warp as wp
from asv_runner.benchmarks.mark import skip_benchmark_if


@dataclass(frozen=True)
class SimulationMetrics:
    """Metrics collected from one simulation benchmark configuration."""

    mean_world_step_time_ms: float
    world_steps_per_second: float
    real_time_factor: float
    p95_frame_time_ms: float
    gpu_memory_mib: float
    sim_dt: float
    sim_substeps: int
    solver_niter_mean: float | None = None
    solver_niter_max: float | None = None


class _SimulationMetricTracks:
    """ASV track methods backed by cached simulation metrics."""

    @skip_benchmark_if(wp.get_cuda_device_count() == 0)
    def track_simulate(self, metrics, world_count):
        return metrics[world_count].mean_world_step_time_ms

    track_simulate.unit = "ms/world-step"

    @skip_benchmark_if(wp.get_cuda_device_count() == 0)
    def track_simulation_steps_per_second(self, metrics, world_count):
        return metrics[world_count].world_steps_per_second

    track_simulation_steps_per_second.unit = "world-steps/s"

    @skip_benchmark_if(wp.get_cuda_device_count() == 0)
    def track_real_time_factor(self, metrics, world_count):
        return metrics[world_count].real_time_factor

    track_real_time_factor.unit = "x"

    @skip_benchmark_if(wp.get_cuda_device_count() == 0)
    def track_p95_step_time(self, metrics, world_count):
        return metrics[world_count].p95_frame_time_ms

    track_p95_step_time.unit = "ms/frame"

    @skip_benchmark_if(wp.get_cuda_device_count() == 0)
    def track_steady_state_gpu_memory(self, metrics, world_count):
        return metrics[world_count].gpu_memory_mib

    track_steady_state_gpu_memory.unit = "MiB"

    @skip_benchmark_if(wp.get_cuda_device_count() == 0)
    def track_sim_dt(self, metrics, world_count):
        return metrics[world_count].sim_dt

    track_sim_dt.unit = "s"

    @skip_benchmark_if(wp.get_cuda_device_count() == 0)
    def track_sim_substeps(self, metrics, world_count):
        return metrics[world_count].sim_substeps

    track_sim_substeps.unit = "simulation-steps/frame"


class _SimulationMetricTracksUnparameterized:
    """ASV track methods backed by one cached simulation configuration."""

    @skip_benchmark_if(wp.get_cuda_device_count() == 0)
    def track_mean_world_step_time(self, metrics):
        return metrics.mean_world_step_time_ms

    track_mean_world_step_time.unit = "ms/world-step"

    @skip_benchmark_if(wp.get_cuda_device_count() == 0)
    def track_simulation_steps_per_second(self, metrics):
        return metrics.world_steps_per_second

    track_simulation_steps_per_second.unit = "world-steps/s"

    @skip_benchmark_if(wp.get_cuda_device_count() == 0)
    def track_real_time_factor(self, metrics):
        return metrics.real_time_factor

    track_real_time_factor.unit = "x"

    @skip_benchmark_if(wp.get_cuda_device_count() == 0)
    def track_p95_step_time(self, metrics):
        return metrics.p95_frame_time_ms

    track_p95_step_time.unit = "ms/frame"

    @skip_benchmark_if(wp.get_cuda_device_count() == 0)
    def track_steady_state_gpu_memory(self, metrics):
        return metrics.gpu_memory_mib

    track_steady_state_gpu_memory.unit = "MiB"

    @skip_benchmark_if(wp.get_cuda_device_count() == 0)
    def track_sim_dt(self, metrics):
        return metrics.sim_dt

    track_sim_dt.unit = "s"

    @skip_benchmark_if(wp.get_cuda_device_count() == 0)
    def track_sim_substeps(self, metrics):
        return metrics.sim_substeps

    track_sim_substeps.unit = "simulation-steps/frame"


def compute_simulation_metrics(
    frame_times: Sequence[float],
    sim_dt: float,
    sim_substeps: int,
    world_count: int,
    gpu_memory_bytes: int,
    experience_frame_times: Sequence[float] | None = None,
) -> SimulationMetrics:
    """Compute comparable simulation metrics from synchronized frame times."""
    if not frame_times or any(not math.isfinite(value) or value <= 0.0 for value in frame_times):
        raise ValueError("frame_times must contain positive finite values")
    if experience_frame_times is None:
        experience_frame_times = frame_times
    if len(experience_frame_times) != len(frame_times) or any(
        not math.isfinite(value) or value <= 0.0 for value in experience_frame_times
    ):
        raise ValueError("experience_frame_times must contain one positive finite value per frame")
    if not math.isfinite(sim_dt) or sim_dt <= 0.0:
        raise ValueError("sim_dt must be positive and finite")
    if sim_substeps <= 0 or world_count <= 0:
        raise ValueError("sim_substeps and world_count must be positive")
    if gpu_memory_bytes < 0:
        raise ValueError("gpu_memory_bytes must be non-negative")

    total_time = sum(frame_times)
    experience_total_time = sum(experience_frame_times)
    world_steps = len(frame_times) * sim_substeps * world_count
    return SimulationMetrics(
        mean_world_step_time_ms=total_time * 1000.0 / world_steps,
        world_steps_per_second=world_steps / experience_total_time,
        real_time_factor=world_steps * sim_dt / experience_total_time,
        p95_frame_time_ms=float(np.percentile(experience_frame_times, 95.0)) * 1000.0,
        gpu_memory_mib=gpu_memory_bytes / 1024**2,
        sim_dt=sim_dt,
        sim_substeps=sim_substeps,
    )


def validate_simulation_state(
    state: Any,
    max_linear_speed: float,
    max_angular_speed: float,
    quaternion_tolerance: float = 1.0e-3,
):
    """Validate finite rigid-body state, normalized rotations, and bounded speeds."""
    state_values = {}
    for name in ("joint_q", "joint_qd", "body_q", "body_qd"):
        values = getattr(state, name).numpy()
        if not np.isfinite(values).all():
            raise RuntimeError(f"Simulation produced non-finite values in state.{name}")
        state_values[name] = values

    body_q = state_values["body_q"].reshape(-1, 7)
    quaternion_norms = np.linalg.norm(body_q[:, 3:7], axis=-1)
    if not np.allclose(quaternion_norms, 1.0, atol=quaternion_tolerance, rtol=0.0):
        max_error = np.abs(quaternion_norms - 1.0).max()
        raise RuntimeError(f"Maximum body quaternion norm error is {max_error:.3g}")

    body_qd = state_values["body_qd"].reshape(-1, 6)
    max_measured_linear_speed = np.linalg.norm(body_qd[:, :3], axis=-1).max()
    max_measured_angular_speed = np.linalg.norm(body_qd[:, 3:], axis=-1).max()
    if max_measured_linear_speed > max_linear_speed:
        raise RuntimeError(
            f"Maximum body linear speed is {max_measured_linear_speed:.3f} m/s, exceeding {max_linear_speed:.1f} m/s"
        )
    if max_measured_angular_speed > max_angular_speed:
        raise RuntimeError(
            f"Maximum body angular speed is {max_measured_angular_speed:.3f} rad/s, "
            f"exceeding {max_angular_speed:.1f} rad/s"
        )


def collect_simulation_metrics(
    create_workload: Callable[[], Any],
    world_count: int,
    num_frames: int,
    samples: int,
    synchronize: Callable[[], None] | None = None,
    validate: Callable[[Any], None] | None = None,
    timer: Callable[[], float] = time.perf_counter,
) -> SimulationMetrics:
    """Collect simulation metrics using internal or synchronized wall timing."""
    frame_times = []
    experience_frame_times = []
    gpu_memory_bytes = None
    sim_dt = None
    sim_substeps = None

    wp.synchronize_device()
    device = wp.get_device()
    free_memory_before = device.free_memory

    for sample_index in range(samples):
        workload = create_workload()
        if sim_dt is None:
            sim_dt = workload.sim_dt
            sim_substeps = workload.sim_substeps
        elif workload.sim_dt != sim_dt or workload.sim_substeps != sim_substeps:
            raise ValueError("simulation parameters changed between samples")

        if synchronize is not None:
            synchronize()
        for _ in range(num_frames):
            experience_start_time = timer()
            benchmark_start_time = workload.benchmark_time if synchronize is None else None
            workload.step()
            if synchronize is not None:
                synchronize()
            experience_frame_time = timer() - experience_start_time
            experience_frame_times.append(experience_frame_time)
            frame_times.append(
                experience_frame_time
                if benchmark_start_time is None
                else workload.benchmark_time - benchmark_start_time
            )

        if sample_index == 0:
            wp.synchronize_device()
            gpu_memory_bytes = free_memory_before - device.free_memory
            if gpu_memory_bytes < 0:
                raise RuntimeError("GPU free memory increased after workload initialization")
        if validate is not None:
            validate(workload)

    return compute_simulation_metrics(
        frame_times=frame_times,
        sim_dt=sim_dt,
        sim_substeps=sim_substeps,
        world_count=world_count,
        gpu_memory_bytes=gpu_memory_bytes,
        experience_frame_times=experience_frame_times,
    )
