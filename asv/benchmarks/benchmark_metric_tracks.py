# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import warp as wp
from asv_runner.benchmarks.mark import skip_benchmark_if


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
