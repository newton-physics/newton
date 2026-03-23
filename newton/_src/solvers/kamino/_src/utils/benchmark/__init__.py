# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Utilities for solver configuration benchmarking."""

from .configs import make_benchmark_configs, make_solver_config_default
from .metrics import (
    BenchmarkMetrics,
    SolverMetrics,
    StatsBinary,
    StatsFloat,
    StatsInteger,
)
from .problems import (
    BenchmarkProblemNameToConfigFn,
    CameraConfig,
    ControlConfig,
    ProblemConfig,
    ProblemSet,
    make_benchmark_problems,
)
from .render import (
    render_solver_configs_table,
    render_subcolumn_metrics_table,
)
from .runner import (
    BenchmarkSim,
    run_single_benchmark,
    run_single_benchmark_silent,
    run_single_benchmark_with_progress,
    run_single_benchmark_with_step_metrics,
    run_single_benchmark_with_viewer,
)

###
# Module interface
###

__all__ = [
    "BenchmarkMetrics",
    "BenchmarkProblemNameToConfigFn",
    "BenchmarkSim",
    "CameraConfig",
    "ControlConfig",
    "ProblemConfig",
    "ProblemSet",
    "SolverMetrics",
    "StatsBinary",
    "StatsFloat",
    "StatsInteger",
    "make_benchmark_configs",
    "make_benchmark_problems",
    "make_solver_config_default",
    "render_solver_configs_table",
    "render_subcolumn_metrics_table",
    "run_single_benchmark",
    "run_single_benchmark_silent",
    "run_single_benchmark_with_progress",
    "run_single_benchmark_with_step_metrics",
    "run_single_benchmark_with_viewer",
]
