# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import argparse
import datetime
import os

import numpy as np
import warp as wp

from newton._src.solvers.kamino._src.core.builder import ModelBuilderKamino
from newton._src.solvers.kamino._src.utils import logger as msg
from newton._src.solvers.kamino._src.utils.benchmark.configs import make_benchmark_configs
from newton._src.solvers.kamino._src.utils.benchmark.metrics import BenchmarkMetrics, CodeInfo
from newton._src.solvers.kamino._src.utils.benchmark.problems import (
    BenchmarkProblemNameToConfigFn,
    make_benchmark_problems,
)
from newton._src.solvers.kamino._src.utils.benchmark.render import (
    render_problem_dimensions_table,
    render_solver_configs_table,
)
from newton._src.solvers.kamino._src.utils.benchmark.runner import run_single_benchmark
from newton._src.solvers.kamino._src.utils.device import get_device_spec_info
from newton._src.solvers.kamino._src.utils.sim import Simulator

###
# Constants
###

SUPPORTED_BENCHMARK_RUN_MODES = ["total", "stepstats", "convergence", "accuracy", "import"]
"""
A list of supported benchmark run modes that determine the level of metrics collected during execution.

Each mode includes the metrics of the previous modes, with increasing levels of detail.

The supported modes are as follows:

- "total":
    Only collects total runtime and final memory usage of each solver configuration and problem.\n
    This mode is intended to be used for a high-level comparison of the overall throughput of
    different solver configurations across problems, without detailed step-by-step metrics.

- "stepstats":
    Collects detailed timings of each simulation step to compute throughput statistics.\n
    This mode lightly impacts overall throughput as it requires synchronizing the device at
    each  step to measure accurate timings. It is intended to be used for analyzing the step
    time distribution and variability across different solver configurations.

- "convergence":
    Collects solver performance metrics such as PADMM iterations and residuals.\n
    This mode moderately impacts overall throughput as it requires additional computation to
    collect and store solver metrics at each step. It is intended to be used for analyzing
    solver convergence behavior and its relationship to step time.

- "accuracy":
    Collects solver performance metrics that evaluate the physical accuracy of the simulation.
    This mode significantly impacts overall throughput as it requires additional computation to
    evaluate the physical accuracy metrics at each step. This is intended to be used for in-depth
    analysis and to evaluate the trade-off between fast convergence and physical correctness.

- "import":
    Generates plots for the collected metrics given an HDF5 file containing benchmark results.\n
    NOTE: This mode does not execute any benchmarks and only produces plots from existing data.
"""

SUPPORTED_BENCHMARK_OUTPUT_MODES = ["console", "full"]  # TODO: add more modes
"""
A list of supported benchmark outputs that determine the format and detail level of the benchmark results.

- "console": Only prints benchmark results to the console as formatted tables.
- "full": TODO.
"""

###
# Functions
###


def parse_benchmark_arguments():
    parser = argparse.ArgumentParser(description="Solver performance benchmark")

    # Warp runtime arguments
    parser.add_argument("--device", type=str, help="Define the Warp device to operate on, e.g. 'cuda:0' or 'cpu'.")
    parser.add_argument(
        "--cuda-graph",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Set to `True` to enable CUDA graph capture (only available on CUDA devices). Defaults to `True`.",
    )
    parser.add_argument(
        "--clear-cache",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Set to `True` to clear Warp's kernel and LTO caches before execution. Defaults to `False`.",
    )

    # World configuration arguments
    parser.add_argument(
        "--num-worlds",
        type=int,
        default=1,
        help="Sets the number of parallel simulation worlds to run. Defaults to `1`.",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=100,
        help="Sets the number of simulation steps to execute. Defaults to `100`.",
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=0.001,
        help="Sets the simulation time step. Defaults to `0.001`.",
    )
    parser.add_argument(
        "--gravity",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enables/disables gravity in the simulation. Defaults to `True`.",
    )
    parser.add_argument(
        "--ground",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enables/disables ground geometry in the simulation. Defaults to `True`.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Sets the random seed for the simulation. Defaults to `0`.",
    )

    # Benchmark execution arguments
    parser.add_argument(
        "--mode",
        type=str,
        choices=SUPPORTED_BENCHMARK_RUN_MODES,
        default="accuracy",
        help=f"Defines the benchmark mode to run. Defaults to 'accuracy'.\n{SUPPORTED_BENCHMARK_RUN_MODES}",
    )
    parser.add_argument(
        "--problem",
        type=str,
        choices=BenchmarkProblemNameToConfigFn.keys(),
        default="dr_legs",
        help="Defines a single problem to benchmark. Defaults to 'dr_legs'. Ignored if '--problem-set' is provided.",
    )
    parser.add_argument(
        "--problem-set",
        nargs="+",
        default=list(BenchmarkProblemNameToConfigFn.keys()),
        help="Defines the benchmark problem(s) to run. If unspecified, all available problems will be used.",
    )
    parser.add_argument(
        "--output",
        type=str,
        choices=SUPPORTED_BENCHMARK_OUTPUT_MODES,
        default="full",
        help=f"Defines the benchmark output mode. Defaults to 'full'.\n{SUPPORTED_BENCHMARK_OUTPUT_MODES}",
    )
    parser.add_argument(
        "--import-path",
        type=str,
        default=None,
        help="Defines the path to the HDF5 benchmark data to import in 'import' mode. Defaults to `None`.",
    )
    parser.add_argument(
        "--viewer",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Set to `True` to run with the simulation viewer. Defaults to `False`.",
    )
    parser.add_argument(
        "--test",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Set to `True` to run `newton.example.run` tests. Defaults to `False`.",
    )

    return parser.parse_args()


def benchmark_run(args: argparse.Namespace):
    """
    Executes the benchmark data generation with the provided arguments.

    This function performs the following steps:
    1. Parses the benchmark arguments to determine the configuration of the run.
    2. Sets the Warp device and determines if CUDA graphs can be used.
    3. Prints device specification info to the console for reference.
    4. Determines the level of metrics to collect based on the specified benchmark mode.
    5. Generates the problem set based on the provided problem names and arguments.
    6. Constructs the `BenchmarkMetrics` object to store collected data.
    7. Iterates over all problem names and settings, executing the benchmark for each combination.
    8. Computes final statistics for the collected benchmark results.
    9. Saves the collected benchmark data to an HDF5 file for later analysis and plotting.
    10. Optionally generates plots from the collected benchmark data.

    Args:
        args: An `argparse.Namespace` object containing the parsed benchmark arguments.

    Returns:
        output_path: The path to the hdf5 file created, that contains all collected data.
    """

    # First print the benchmark configuration to the console for reference
    msg.notif(f"Running benchmark in mode: {args.mode}")

    # Print the git commit hash and repository info to the
    # console for traceability and reproducibility of benchmark runs
    codeinfo = CodeInfo()
    msg.notif(f"Benchmark will run with the following repository:\n{codeinfo}\n")

    # Set device if specified, otherwise use Warp's default
    if args.device:
        device = wp.get_device(args.device)
        wp.set_device(device)
    else:
        device = wp.get_preferred_device()

    # Print device specification info to console for reference
    spec_info = get_device_spec_info(device)
    msg.notif("[Device]: %s", spec_info)

    # Determine if CUDA graphs should be used for execution
    can_use_cuda_graph = device.is_cuda and wp.is_mempool_enabled(device)
    use_cuda_graph = can_use_cuda_graph and args.cuda_graph
    msg.info(f"can_use_cuda_graph: {can_use_cuda_graph}")
    msg.info(f"using_cuda_graph: {use_cuda_graph}")

    # Determine the metrics to collect based on the benchmark mode
    if args.mode == "total":
        collect_step_metrics = False
        collect_solver_metrics = False
        collect_physics_metrics = False
    elif args.mode == "stepstats":
        collect_step_metrics = True
        collect_solver_metrics = False
        collect_physics_metrics = False
    elif args.mode == "convergence":
        collect_step_metrics = True
        collect_solver_metrics = True
        collect_physics_metrics = False
    elif args.mode == "accuracy":
        collect_step_metrics = True
        collect_solver_metrics = True
        collect_physics_metrics = True
    else:
        raise ValueError(f"Unsupported benchmark mode '{args.mode}'. Supported modes: {SUPPORTED_BENCHMARK_RUN_MODES}")
    msg.info(f"collect_step_metrics: {collect_step_metrics}")
    msg.info(f"collect_solver_metrics: {collect_solver_metrics}")
    msg.info(f"collect_physics_metrics: {collect_physics_metrics}")

    # Determine the problem set from
    # the single and list arguments
    if len(args.problem_set) == 0:
        problem_names = [args.problem]
    else:
        problem_names = args.problem_set
    msg.notif(f"problem_names: {problem_names}")

    # Define and create the output directory for the benchmark results
    RUN_OUTPUT_PATH = None
    if args.output == "full":
        DATA_DIR_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "./data"))
        RUN_OUTPUT_NAME = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        RUN_OUTPUT_PATH = f"{DATA_DIR_PATH}/{RUN_OUTPUT_NAME}"
        os.makedirs(RUN_OUTPUT_PATH, exist_ok=True)

    # Generate a set of solver configurations to benchmark over
    configs_set = make_benchmark_configs(include_default=False)
    msg.notif(f"config_names: {list(configs_set.keys())}")
    render_solver_configs_table(configs=configs_set, groups=["sparse", "linear", "padmm"], to_console=True)
    if args.output == "full":
        render_solver_configs_table(
            configs=configs_set,
            path=os.path.join(RUN_OUTPUT_PATH, "solver_configs.txt"),
            groups=["cts", "sparse", "linear", "padmm", "warmstart"],
            to_console=False,
        )

    # Generate the problem set based on the
    # provided problem names and arguments
    problem_set = make_benchmark_problems(
        names=problem_names,
        num_worlds=args.num_worlds,
        gravity=args.gravity,
        ground=args.ground,
    )

    # Construct and initialize the metrics
    # object to store benchmark data
    metrics = BenchmarkMetrics(
        problems=problem_names,
        configs=configs_set,
        num_steps=args.num_steps,
        step_metrics=collect_step_metrics,
        solver_metrics=collect_solver_metrics,
        physics_metrics=collect_physics_metrics,
    )

    # Iterator over all problem names and settings and run benchmarks for each
    for problem_name, problem_config in problem_set.items():
        # Unpack problem configurations
        builder, control, camera = problem_config
        if not isinstance(builder, ModelBuilderKamino):
            builder = builder()

        for config_name, configs in configs_set.items():
            msg.notif("Running benchmark for problem '%s' with simulation configs '%s'", problem_name, config_name)

            # Retrieve problem and config indices
            problem_idx = metrics._problem_names.index(problem_name)
            config_idx = metrics._config_names.index(config_name)

            # Construct simulator configurations based on the solver
            # configurations for the current benchmark configuration
            sim_configs = Simulator.Config(dt=args.dt, solver=configs)
            sim_configs.solver.use_fk_solver = False

            # Execute the benchmark for the current problem and settings
            run_single_benchmark(
                problem_idx=problem_idx,
                config_idx=config_idx,
                metrics=metrics,
                args=args,
                builder=builder,
                configs=sim_configs,
                control=control,
                camera=camera,
                device=device,
                use_cuda_graph=use_cuda_graph,
                print_device_info=True,
            )

    # Print table with problem dimensions
    render_problem_dimensions_table(metrics._problem_dims, to_console=True)
    if args.output == "full":
        render_problem_dimensions_table(
            metrics._problem_dims,
            path=os.path.join(RUN_OUTPUT_PATH, "problem_dimensions.txt"),
            to_console=False,
        )

    # Compute final statistics for the benchmark results
    metrics.compute_stats()

    # Export the collected benchmark data to an HDF5 file for later analysis and plotting
    if args.output == "full":
        msg.info("Saving benchmark data to HDF5...")
        RUN_HDF5_OUTPUT_PATH = f"{RUN_OUTPUT_PATH}/metrics.hdf5"
        metrics.save_to_hdf5(path=RUN_HDF5_OUTPUT_PATH)
        msg.info("Done.")

    # Return collected metrics and path to export folder (None if not exported)
    return metrics, RUN_OUTPUT_PATH


def load_metrics(data_import_path: str | None):
    # If the import path is not specified load the latest created HDF5 file in the output directory
    if data_import_path is None:
        DATA_DIR_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "./data"))
        all_runs = next(os.walk(DATA_DIR_PATH))[1]
        all_runs = sorted(all_runs, key=lambda x: os.stat(os.path.join(DATA_DIR_PATH, x)).st_mtime)
        if len(all_runs) == 0:
            raise FileNotFoundError(f"No benchmark runs found in output directory '{DATA_DIR_PATH}'.")
        latest_run = all_runs[-1]
        data_import_path = os.path.join(DATA_DIR_PATH, latest_run, "metrics.hdf5")
        msg.notif(f"No import path specified. Loading latest benchmark data from '{data_import_path}'.")

    # Ensure that the specified import path exists and is a valid HDF5 file
    if not os.path.exists(data_import_path):
        raise FileNotFoundError(f"The specified import path '{data_import_path}' does not exist.")
    elif not os.path.isfile(data_import_path):
        raise ValueError(f"The specified import path '{data_import_path}' is not a file.")
    elif not data_import_path.endswith(".hdf5"):
        raise ValueError(f"The specified import path '{data_import_path}' is not an HDF5 file.")

    # Retrieve the parent directory of the import path to use as the base output path for any generated plots
    import_parent_dir = os.path.dirname(data_import_path)
    msg.notif(f"Output will be generated in directory '{import_parent_dir}'.")

    # Load the benchmark data from the specified HDF5 file into
    # a `BenchmarkMetrics` object for analysis and plotting
    metrics = BenchmarkMetrics(path=data_import_path)

    # Return loaded metrics and the path of the containing folder
    return metrics, import_parent_dir


def benchmark_output(metrics: BenchmarkMetrics, export_dir: str | None):
    # Compute statistics for the collected benchmark
    # data to prepare for plotting and analysis
    metrics.compute_stats()

    # Print the total performance summary as a formatted table to the console:
    # - The columns span the problems, with a sub-column for each
    #   metric (e.g. total time, total FPS, memory used)
    # - The rows span the solver configurations
    total_metrics_table_path = None
    if export_dir is not None:
        total_metrics_table_path = os.path.join(export_dir, "total_metrics.txt")
    metrics.render_total_metrics_table(path=total_metrics_table_path)

    # For each problem, export a table summarizing the step-time for each solver configuration:
    # - A sub-column for each statistic (mean, std, min, max)
    # - The rows span the solver configurations
    if metrics.step_time is not None:
        step_time_summary_path = None
        if export_dir is not None:
            step_time_summary_path = os.path.join(export_dir, "step_time")
        metrics.render_step_time_table(path=step_time_summary_path)

    # For each problem, export a table summarizing the PADMM metrics for each solver configuration:
    # - The columns span the metrics (e.g. step time, padmm.*, physics.*),
    #   with a sub-column for each statistic (mean, std, min, max)
    # - The rows span the solver configurations
    if metrics.solver_metrics is not None:
        padmm_metrics_summary_path = None
        padmm_metrics_plots_path = None
        if export_dir is not None:
            padmm_metrics_summary_path = os.path.join(export_dir, "padmm_metrics")
            padmm_metrics_plots_path = os.path.join(export_dir, "padmm_metrics")
        metrics.render_padmm_metrics_table(path=padmm_metrics_summary_path)
        metrics.render_padmm_metrics_plots(path=padmm_metrics_plots_path)

    # For each problem, export a table summarizing the PADMM metrics for each solver configuration:
    # - The columns span the metrics (e.g. step time, padmm.*, physics.*),
    #   with a sub-column for each statistic (mean, std, min, max)
    # - The rows span the solver configurations
    if metrics.physics_metrics is not None:
        physics_metrics_summary_path = None
        physics_metrics_plots_path = None
        if export_dir is not None:
            physics_metrics_summary_path = os.path.join(export_dir, "physics_metrics")
            physics_metrics_plots_path = os.path.join(export_dir, "physics_metrics")
        metrics.render_physics_metrics_table(path=physics_metrics_summary_path)
        metrics.render_physics_metrics_plots(path=physics_metrics_plots_path)


###
# Main function
###

if __name__ == "__main__":
    # Load benchmark-specific program arguments
    args = parse_benchmark_arguments()

    # Set global numpy configurations
    np.set_printoptions(linewidth=20000, precision=6, threshold=10000, suppress=True)  # Suppress scientific notation

    # Clear warp cache if requested
    if args.clear_cache:
        wp.clear_kernel_cache()
        wp.clear_lto_cache()

    # TODO: Make optional
    # Set the verbosity of the global message logger
    msg.set_log_level(msg.LogLevel.INFO)

    # If the benchmark mode is not "import", first execute the
    # benchmark and then produce output from the collected data
    if args.mode != "import":
        metrics, export_dir = benchmark_run(args)
        benchmark_output(metrics=metrics, export_dir=export_dir)
    else:
        if args.import_path is not None:
            msg.notif(f"Loading benchmark data from specified import path '{args.import_path}'.")
        metrics, export_dir = load_metrics(args.import_path)
        benchmark_output(metrics=metrics, export_dir=export_dir)
