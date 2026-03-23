# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import os
from typing import Any, Literal

import numpy as np

from ......core.types import override
from ...solver_kamino_impl import SolverKaminoImpl
from .configs import load_solver_configs_to_hdf5, save_solver_configs_to_hdf5
from .problems import ProblemDimensions, save_problem_dimensions_to_hdf5
from .render import (
    ColumnGroup,
    render_subcolumn_metrics_table,
    render_subcolumn_table,
)

###
# Module interface
###

__all__ = [
    "BenchmarkMetrics",
    "SolverMetrics",
    "StatsBinary",
    "StatsFloat",
    "StatsInteger",
]

###
# Types - Meta-Data
###


class CodeInfo:
    """
    A utility container to encapsulate information about the code
    repository, such as the remote URL, branch, and commit hash.
    """

    def __init__(self, path: str | None = None, empty: bool = False):
        """
        Initialize a CodeInfo object.

        Args:
            path (str | None):
                The path to the git repository. If None, the current working directory is used.

        Raises:
            RuntimeError: If there is an error retrieving git repository info from the specified path.
        """
        # TODO: Consider using a silent warning and allowing the CodeInfo
        # to be initialized with None values instead of raising an error
        # Attempt to import git first, and warn user
        # if the necessary package is not installed
        try:
            import git
        except ImportError as e:
            raise ImportError(
                "The GitPython package is required for downloading git folders. Install it with: pip install gitpython"
            ) from e

        # Declare git repository info attributes
        self.repo: git.Repo | None = None
        self.path: str | None = None
        self.remote: str | None = None
        self.branch: str | None = None
        self.commit: str | None = None
        self.diff: str | None = None

        # If a path is provided, attempt to retrieve git repository info from
        # that, otherwise attempt to retrieve from the current working directory
        if path is not None:
            _path = path
        elif not empty:
            _path = str(os.path.dirname(__file__))
        else:
            # If empty is True, skip retrieving git repository info and leave all attributes as None
            return

        # Attempt to retrieve git repository info from the specified path;
        # if any error occurs, raise a RuntimeError with the error message
        try:
            self.repo = git.Repo(path=_path, search_parent_directories=True)
            self.path = self.repo.working_tree_dir
            self.remote = self.repo.remote().url if self.repo.remotes else None
            self.branch = str(self.repo.active_branch)
            self.commit = str(self.repo.head.object.hexsha)
            self.diff = str(self.repo.git.diff())
        except Exception as e:
            raise RuntimeError(f"Error retrieving git repository info: {e}") from e

    def __repr__(self):
        """Returns a human-readable string representation of the CodeInfo."""
        return (
            f"CodeInfo(\n"
            f"  path: {self.path}\n"
            f"  remote: {self.remote}\n"
            f"  branch: {self.branch}\n"
            f"  commit: {self.commit}\n"
            f")"
        )

    def __str__(self):
        """Returns a human-readable string representation of the CodeInfo (same as __repr__)."""
        return self.__repr__()

    def as_dict(self) -> dict:
        """Returns a dictionary representation of the CodeInfo."""
        return {
            "path": self.path,
            "remote": self.remote,
            "branch": self.branch,
            "commit": self.commit,
        }


###
# Types - Statistics
###


class StatsFloat:
    """A utility class to compute statistics for floating-point data arrays, such as step times and residuals."""

    def __init__(self, data: np.ndarray, name: str | None = None):
        """
        Initialize a StatsFloat object.

        Args:
            data (np.ndarray): A floating-point data array.
            name (str | None): An optional name for the statistics object.

        Raises:
            ValueError: If the data array is not of a floating-point type.
        """
        if not np.issubdtype(data.dtype, np.floating):
            raise ValueError("StatsFloat requires a floating-point data array.")
        self.name: str | None = name

        # Declare statistics arrays
        self.median: np.ndarray = np.zeros((data.shape[0], data.shape[1]), dtype=np.float32)
        self.mean: np.ndarray = np.zeros((data.shape[0], data.shape[1]), dtype=np.float32)
        self.std: np.ndarray = np.zeros((data.shape[0], data.shape[1]), dtype=np.float32)
        self.min: np.ndarray = np.zeros((data.shape[0], data.shape[1]), dtype=np.float32)
        self.max: np.ndarray = np.zeros((data.shape[0], data.shape[1]), dtype=np.float32)

        # Compute float stats of each problem (i.e. along axis=2)
        self.median[:, :] = np.median(data, axis=2)
        self.mean[:, :] = np.mean(data, axis=2)
        self.std[:, :] = np.std(data, axis=2)
        self.min[:, :] = np.min(data, axis=2)
        self.max[:, :] = np.max(data, axis=2)

    @override
    def __repr__(self):
        """Returns a human-readable string representation of the StatsFloat."""
        return (
            f"StatsFloat[{self.name or '-'}](\n"
            f"median:\n{self.median},\n"
            f"mean:\n{self.mean},\n"
            f"std:\n{self.std},\n"
            f"min:\n{self.min},\n"
            f"max:\n{self.max}\n"
        )


class StatsInteger:
    """A utility class to compute statistics for integer data arrays, such as counts and distributions."""

    def __init__(self, data: np.ndarray, num_bins: int = 20, name: str | None = None):
        """
        Initialize a StatsInteger object.

        Args:
            data (np.ndarray): An integer data array.
            num_bins (int): Number of bins for histogram (default: 20).
            name (str | None): An optional name for the statistics object.

        Raises:
            ValueError: If the data array is not of an integer type.
        """
        if not np.issubdtype(data.dtype, np.integer):
            raise ValueError("StatsInteger requires an integer data array.")
        self.name: str | None = name

        # Declare statistics arrays
        self.median: np.ndarray = np.zeros((data.shape[0], data.shape[1]), dtype=np.float32)
        self.mean: np.ndarray = np.zeros((data.shape[0], data.shape[1]), dtype=np.float32)
        self.std: np.ndarray = np.zeros((data.shape[0], data.shape[1]), dtype=np.float32)
        self.min: np.ndarray = np.zeros((data.shape[0], data.shape[1]), dtype=np.float32)
        self.max: np.ndarray = np.zeros((data.shape[0], data.shape[1]), dtype=np.float32)

        # Compute integer stats of each problem (i.e. along axis=2)
        self.median[:, :] = np.median(data.astype(np.float32), axis=2)
        self.mean[:, :] = np.mean(data.astype(np.float32), axis=2)
        self.std[:, :] = np.std(data.astype(np.float32), axis=2)
        self.min[:, :] = np.min(data.astype(np.float32), axis=2)
        self.max[:, :] = np.max(data.astype(np.float32), axis=2)

    @override
    def __repr__(self):
        """Returns a human-readable string representation of the StatsInteger."""
        return (
            f"StatsInteger[{self.name or '-'}](\n"
            f"median:\n{self.median},\n"
            f"mean:\n{self.mean},\n"
            f"std:\n{self.std},\n"
            f"min:\n{self.min},\n"
            f"max:\n{self.max}\n"
        )


class StatsBinary:
    """A utility class to compute statistics for binary (boolean) data arrays, such as counts of zeros and ones."""

    def __init__(self, data: np.ndarray, name: str | None = None):
        """
        Initialize a StatsBinary object.

        Args:
            data (np.ndarray): A binary (boolean) data array.
            name (str | None): An optional name for the statistics object.

        Raises:
            ValueError: If the data array is not of a binary (boolean) type.
        """
        if not np.issubdtype(data.dtype, np.integer) or not np.array_equal(data, data.astype(bool)):
            raise ValueError("StatsBinary requires a binary (boolean) data array.")
        self.name: str | None = name

        # Declare Binary statistics arrays
        self.count_zeros: np.ndarray = np.zeros((data.shape[0], data.shape[1]), dtype=np.float32)
        self.count_ones: np.ndarray = np.zeros((data.shape[0], data.shape[1]), dtype=np.float32)

        # Compute binary stats of each problem (i.e. along axis=2)
        self.count_zeros[:, :] = np.sum(data == 0, axis=2)
        self.count_ones[:, :] = np.sum(data == 1, axis=2)

    @override
    def __repr__(self):
        """Returns a human-readable string representation of the StatsBinary."""
        return f"StatsBinary[{self.name or '-'}](\ncount_zeros:\n{self.count_zeros},\ncount_ones:\n{self.count_ones}\n"


###
# Types - Metrics
###


class SolverMetrics:
    def __init__(self, num_problems: int, num_configs: int, num_steps: int):
        # Solver-specific metrics
        self.padmm_converged: np.ndarray = np.zeros((num_problems, num_configs, num_steps), dtype=np.int32)
        self.padmm_iters: np.ndarray = np.zeros((num_problems, num_configs, num_steps), dtype=np.int32)
        self.padmm_r_p: np.ndarray = np.zeros((num_problems, num_configs, num_steps), dtype=np.float32)
        self.padmm_r_d: np.ndarray = np.zeros((num_problems, num_configs, num_steps), dtype=np.float32)
        self.padmm_r_c: np.ndarray = np.zeros((num_problems, num_configs, num_steps), dtype=np.float32)

        # Linear solver metrics (placeholders for now)
        self.linear_solver_iters: np.ndarray = np.zeros((num_problems, num_configs, num_steps), dtype=np.float32)
        self.linear_solver_r_error: np.ndarray = np.zeros((num_problems, num_configs, num_steps), dtype=np.float32)

        # Stats (computed after data collection)
        self.padmm_success_stats: StatsBinary | None = None
        self.padmm_iters_stats: StatsInteger | None = None
        self.padmm_r_p_stats: StatsFloat | None = None
        self.padmm_r_d_stats: StatsFloat | None = None
        self.padmm_r_c_stats: StatsFloat | None = None
        self.linear_solver_iters_stats: StatsInteger | None = None
        self.linear_solver_r_error_stats: StatsFloat | None = None

    def compute_stats(self):
        self.padmm_success_stats = StatsBinary(self.padmm_converged, name="padmm_converged")
        self.padmm_iters_stats = StatsInteger(self.padmm_iters, name="padmm_iters")
        self.padmm_r_p_stats = StatsFloat(self.padmm_r_p, name="padmm_r_p")
        self.padmm_r_d_stats = StatsFloat(self.padmm_r_d, name="padmm_r_d")
        self.padmm_r_c_stats = StatsFloat(self.padmm_r_c, name="padmm_r_c")
        # TODO: self.linear_solver_iters_stats = StatsInteger(self.linear_solver_iters, name="linear_solver_iters")
        # TODO: self.linear_solver_r_error_stats = StatsFloat(self.linear_solver_r_error, name="linear_solver_r_error")


class PhysicsMetrics:
    def __init__(self, num_problems: int, num_configs: int, num_steps: int):
        # Physics-specific metrics
        self.r_eom: np.ndarray = np.zeros((num_problems, num_configs, num_steps), dtype=np.float32)
        self.r_kinematics: np.ndarray = np.zeros((num_problems, num_configs, num_steps), dtype=np.float32)
        self.r_cts_joints: np.ndarray = np.zeros((num_problems, num_configs, num_steps), dtype=np.float32)
        self.r_cts_limits: np.ndarray = np.zeros((num_problems, num_configs, num_steps), dtype=np.float32)
        self.r_cts_contacts: np.ndarray = np.zeros((num_problems, num_configs, num_steps), dtype=np.float32)
        self.r_v_plus: np.ndarray = np.zeros((num_problems, num_configs, num_steps), dtype=np.float32)
        self.r_ncp_primal: np.ndarray = np.zeros((num_problems, num_configs, num_steps), dtype=np.float32)
        self.r_ncp_dual: np.ndarray = np.zeros((num_problems, num_configs, num_steps), dtype=np.float32)
        self.r_ncp_compl: np.ndarray = np.zeros((num_problems, num_configs, num_steps), dtype=np.float32)
        self.r_vi_natmap: np.ndarray = np.zeros((num_problems, num_configs, num_steps), dtype=np.float32)
        self.f_ncp: np.ndarray = np.zeros((num_problems, num_configs, num_steps), dtype=np.float32)
        self.f_ccp: np.ndarray = np.zeros((num_problems, num_configs, num_steps), dtype=np.float32)

        # Stats (computed after data collection)
        self.r_eom_stats: StatsFloat | None = None
        self.r_kinematics_stats: StatsFloat | None = None
        self.r_cts_joints_stats: StatsFloat | None = None
        self.r_cts_limits_stats: StatsFloat | None = None
        self.r_cts_contacts_stats: StatsFloat | None = None
        self.r_v_plus_stats: StatsFloat | None = None
        self.r_ncp_primal_stats: StatsFloat | None = None
        self.r_ncp_dual_stats: StatsFloat | None = None
        self.r_ncp_compl_stats: StatsFloat | None = None
        self.r_vi_natmap_stats: StatsFloat | None = None
        self.f_ncp_stats: StatsFloat | None = None
        self.f_ccp_stats: StatsFloat | None = None

    def compute_stats(self):
        self.r_eom_stats = StatsFloat(self.r_eom, name="r_eom")
        self.r_kinematics_stats = StatsFloat(self.r_kinematics, name="r_kinematics")
        self.r_cts_joints_stats = StatsFloat(self.r_cts_joints, name="r_cts_joints")
        self.r_cts_limits_stats = StatsFloat(self.r_cts_limits, name="r_cts_limits")
        self.r_cts_contacts_stats = StatsFloat(self.r_cts_contacts, name="r_cts_contacts")
        self.r_v_plus_stats = StatsFloat(self.r_v_plus, name="r_v_plus")
        self.r_ncp_primal_stats = StatsFloat(self.r_ncp_primal, name="r_ncp_primal")
        self.r_ncp_dual_stats = StatsFloat(self.r_ncp_dual, name="r_ncp_dual")
        self.r_ncp_compl_stats = StatsFloat(self.r_ncp_compl, name="r_ncp_compl")
        self.r_vi_natmap_stats = StatsFloat(self.r_vi_natmap, name="r_vi_natmap")
        self.f_ncp_stats = StatsFloat(self.f_ncp, name="f_ncp")
        self.f_ccp_stats = StatsFloat(self.f_ccp, name="f_ccp")


class BenchmarkMetrics:
    def __init__(
        self,
        problems: list[str] | None = None,
        configs: dict[str, SolverKaminoImpl.Config] | None = None,
        num_steps: int | None = None,
        step_metrics: bool = False,
        solver_metrics: bool = False,
        physics_metrics: bool = False,
        path: str | None = None,
    ):
        # Declare data-set dimensions
        self._problem_names: list[str] | None = None
        self._config_names: list[str] | None = None
        self._num_steps: int | None = None

        # Declare problem dimensions
        self._problem_dims: dict[str, ProblemDimensions] = {}

        # Declare cache of the solver configurations used in the
        # benchmark for easy reference when analyzing results
        self._configs: dict[str, SolverKaminoImpl.Config] | None = None

        # One-time metrics
        self.memory_used: np.ndarray | None = None
        self.total_time: np.ndarray | None = None
        self.total_fps: np.ndarray | None = None

        # Per-step metrics
        self.step_time: np.ndarray | None = None
        self.step_time_stats: StatsFloat | None = None

        # Optional solver-specific metrics
        self.solver_metrics: SolverMetrics | None = None

        # Optional physics-specific metrics
        self.physics_metrics: PhysicsMetrics | None = None

        # Meta-data about the code repository at the time of the
        # benchmark run for traceability and reproducibility
        self.codeinfo: CodeInfo | None = None

        # Initialize metrics data structures if problem
        # names, config names, and num_steps are provided,
        # otherwise load from HDF5 if path is provided
        if problems is not None and configs is not None:
            self.finalize(
                problems=problems,
                configs=configs,
                num_steps=num_steps,
                step_metrics=step_metrics,
                solver_metrics=solver_metrics,
                physics_metrics=physics_metrics,
            )
        elif path is not None:
            self.load_from_hdf5(path=path)

    @property
    def num_problems(self) -> int:
        if self._problem_names is None:
            raise ValueError("BenchmarkMetrics: problem names not set. Call finalize() first.")
        return len(self._problem_names)

    @property
    def num_configs(self) -> int:
        if self._config_names is None:
            raise ValueError("BenchmarkMetrics: config names not set. Call finalize() first.")
        return len(self._config_names)

    @property
    def num_steps(self) -> int:
        if self._num_steps is None:
            raise ValueError("BenchmarkMetrics: num_steps not set. Call finalize() first.")
        return self._num_steps

    def finalize(
        self,
        problems: list[str],
        configs: dict[str, SolverKaminoImpl.Config],
        num_steps: int | None = None,
        step_metrics: bool = False,
        solver_metrics: bool = False,
        physics_metrics: bool = False,
    ):
        # Cache run problem and config names as well as total step counts
        self._problem_names = problems
        self._config_names = list(configs.keys())
        self._configs = configs
        self._num_steps = num_steps if num_steps is not None else 1

        # Allocate arrays for one-time total run metrics
        self.memory_used = np.zeros((self.num_problems, self.num_configs), dtype=np.float32)
        self.total_time = np.zeros((self.num_problems, self.num_configs), dtype=np.float32)
        self.total_fps = np.zeros((self.num_problems, self.num_configs), dtype=np.float32)

        # Allocate per-step metrics arrays if enabled
        if step_metrics:
            self.step_time = np.zeros((self.num_problems, self.num_configs, self._num_steps), dtype=np.float32)
        if solver_metrics:
            self.solver_metrics = SolverMetrics(self.num_problems, self.num_configs, self._num_steps)
        if physics_metrics:
            self.physics_metrics = PhysicsMetrics(self.num_problems, self.num_configs, self._num_steps)

        # Generate meta-data to record git repository info
        self.codeinfo = CodeInfo()

    def record_step(
        self,
        problem_idx: int,
        config_idx: int,
        step_idx: int,
        step_time: float,
        solver: SolverKaminoImpl | None = None,
    ):
        if self.step_time is None:
            raise ValueError(
                "BenchmarkMetrics: step_time array not initialized. Call finalize() with step_metrics=True first."
            )
        self.step_time[problem_idx, config_idx, step_idx] = step_time
        if self.solver_metrics is not None and solver is not None:
            # Extract PADMM solver status info - this is multiworld
            solver_status_np = solver._solver_fd.data.status.numpy()
            solver_status_np = {name: solver_status_np[name].max() for name in solver_status_np.dtype.names}
            self.solver_metrics.padmm_converged[problem_idx, config_idx, step_idx] = solver_status_np["converged"]
            self.solver_metrics.padmm_iters[problem_idx, config_idx, step_idx] = solver_status_np["iterations"]
            self.solver_metrics.padmm_r_p[problem_idx, config_idx, step_idx] = solver_status_np["r_p"]
            self.solver_metrics.padmm_r_d[problem_idx, config_idx, step_idx] = solver_status_np["r_d"]
            self.solver_metrics.padmm_r_c[problem_idx, config_idx, step_idx] = solver_status_np["r_c"]
        if self.physics_metrics is not None and solver is not None and solver.metrics is not None:
            r_eom_np = solver.metrics.data.r_eom.numpy().max(axis=0)
            r_kinematics_np = solver.metrics.data.r_kinematics.numpy().max(axis=0)
            r_cts_joints_np = solver.metrics.data.r_cts_joints.numpy().max(axis=0)
            r_cts_limits_np = solver.metrics.data.r_cts_limits.numpy().max(axis=0)
            r_cts_contacts_np = solver.metrics.data.r_cts_contacts.numpy().max(axis=0)
            r_v_plus_np = solver.metrics.data.r_v_plus.numpy().max(axis=0)
            r_ncp_primal_np = solver.metrics.data.r_ncp_primal.numpy().max(axis=0)
            r_ncp_dual_np = solver.metrics.data.r_ncp_dual.numpy().max(axis=0)
            r_ncp_compl_np = solver.metrics.data.r_ncp_compl.numpy().max(axis=0)
            r_vi_natmap_np = solver.metrics.data.r_vi_natmap.numpy().max(axis=0)
            f_ncp_np = solver.metrics.data.f_ncp.numpy().max(axis=0)
            f_ccp_np = solver.metrics.data.f_ccp.numpy().max(axis=0)
            self.physics_metrics.r_eom[problem_idx, config_idx, step_idx] = r_eom_np
            self.physics_metrics.r_kinematics[problem_idx, config_idx, step_idx] = r_kinematics_np
            self.physics_metrics.r_cts_joints[problem_idx, config_idx, step_idx] = r_cts_joints_np
            self.physics_metrics.r_cts_limits[problem_idx, config_idx, step_idx] = r_cts_limits_np
            self.physics_metrics.r_cts_contacts[problem_idx, config_idx, step_idx] = r_cts_contacts_np
            self.physics_metrics.r_v_plus[problem_idx, config_idx, step_idx] = r_v_plus_np
            self.physics_metrics.r_ncp_primal[problem_idx, config_idx, step_idx] = r_ncp_primal_np
            self.physics_metrics.r_ncp_dual[problem_idx, config_idx, step_idx] = r_ncp_dual_np
            self.physics_metrics.r_ncp_compl[problem_idx, config_idx, step_idx] = r_ncp_compl_np
            self.physics_metrics.r_vi_natmap[problem_idx, config_idx, step_idx] = r_vi_natmap_np
            self.physics_metrics.f_ncp[problem_idx, config_idx, step_idx] = f_ncp_np
            self.physics_metrics.f_ccp[problem_idx, config_idx, step_idx] = f_ccp_np

    def record_total(
        self,
        problem_idx: int,
        config_idx: int,
        total_steps: int,
        total_time: float,
        memory_used: float,
    ):
        self.memory_used[problem_idx, config_idx] = memory_used
        self.total_time[problem_idx, config_idx] = total_time
        self.total_fps[problem_idx, config_idx] = float(total_steps) / total_time if total_time > 0.0 else 0.0

    def compute_stats(self):
        if self.step_time is not None:
            self.step_time_stats = StatsFloat(self.step_time, name="step_time")
        if self.solver_metrics is not None:
            self.solver_metrics.compute_stats()
        if self.physics_metrics is not None:
            self.physics_metrics.compute_stats()

    def save_to_hdf5(self, path: str):
        # Attempt to import h5py first, and warn user
        # if the necessary package is not installed
        try:
            import h5py  # noqa: PLC0415
        except ImportError as e:
            raise ImportError(
                "The `h5py` package is required for saving to HDF5. Install it with: pip install h5py"
            ) from e

        # Ensure that there is in fact data to save before attempting to write to HDF5
        if self._problem_names is None or self._config_names is None or self._num_steps is None:
            raise ValueError("BenchmarkMetrics: problem names, config names, and num_steps must be set before saving.")

        # Open an HDF5 file for writing and save all data arrays along with meta-data about
        # the benchmark run and code repository info for traceability and reproducibility
        with h5py.File(path, "w") as datafile:
            # Info about the project at the time of the benchmark run
            datafile["Info/code/path"] = self.codeinfo.path
            datafile["Info/code/remote"] = self.codeinfo.remote
            datafile["Info/code/branch"] = self.codeinfo.branch
            datafile["Info/code/commit"] = self.codeinfo.commit
            datafile["Info/code/diff"] = self.codeinfo.diff

            # Problem dimensions
            save_problem_dimensions_to_hdf5(self._problem_dims, datafile)

            # Save solver configuration parameters
            save_solver_configs_to_hdf5(self._configs, datafile)

            # Info about the benchmark data
            datafile["Info/problem_names"] = self._problem_names
            datafile["Info/config_names"] = self._config_names
            datafile["Info/num_steps"] = self._num_steps
            datafile["Info/has_step_metrics"] = self.step_time is not None
            datafile["Info/has_solver_metrics"] = self.solver_metrics is not None
            datafile["Info/has_physics_metrics"] = self.physics_metrics is not None

            # Basic run metrics
            datafile["Data/total/memory_used"] = self.memory_used
            datafile["Data/total/total_time"] = self.total_time
            datafile["Data/total/total_fps"] = self.total_fps
            if self.step_time is not None:
                datafile["Data/perstep/step_time"] = self.step_time
            if self.solver_metrics is not None:
                datafile["Data/perstep/padmm/converged"] = self.solver_metrics.padmm_converged
                datafile["Data/perstep/padmm/iterations"] = self.solver_metrics.padmm_iters
                datafile["Data/perstep/padmm/r_p"] = self.solver_metrics.padmm_r_p
                datafile["Data/perstep/padmm/r_d"] = self.solver_metrics.padmm_r_d
                datafile["Data/perstep/padmm/r_c"] = self.solver_metrics.padmm_r_c
            if self.physics_metrics is not None:
                datafile["Data/perstep/physics/r_eom"] = self.physics_metrics.r_eom
                datafile["Data/perstep/physics/r_kinematics"] = self.physics_metrics.r_kinematics
                datafile["Data/perstep/physics/r_cts_joints"] = self.physics_metrics.r_cts_joints
                datafile["Data/perstep/physics/r_cts_limits"] = self.physics_metrics.r_cts_limits
                datafile["Data/perstep/physics/r_cts_contacts"] = self.physics_metrics.r_cts_contacts
                datafile["Data/perstep/physics/r_v_plus"] = self.physics_metrics.r_v_plus
                datafile["Data/perstep/physics/r_ncp_primal"] = self.physics_metrics.r_ncp_primal
                datafile["Data/perstep/physics/r_ncp_dual"] = self.physics_metrics.r_ncp_dual
                datafile["Data/perstep/physics/r_ncp_compl"] = self.physics_metrics.r_ncp_compl
                datafile["Data/perstep/physics/r_vi_natmap"] = self.physics_metrics.r_vi_natmap
                datafile["Data/perstep/physics/f_ncp"] = self.physics_metrics.f_ncp
                datafile["Data/perstep/physics/f_ccp"] = self.physics_metrics.f_ccp

    def load_from_hdf5(self, path: str):
        # Attempt to import h5py first, and warn user
        # if the necessary package is not installed
        try:
            import h5py  # noqa: PLC0415
        except ImportError as e:
            raise ImportError(
                "The `h5py` package is required for saving to HDF5. Install it with: pip install h5py"
            ) from e

        """Load raw data arrays from the HDF5 file into the BenchmarkMetrics instance"""
        with h5py.File(path, "r") as datafile:
            # First load the info group to get the dimensions and initialize the data arrays
            self._problem_names = datafile["Info/problem_names"][:].astype(str).tolist()
            self._config_names = datafile["Info/config_names"][:].astype(str).tolist()
            self._num_steps = int(datafile["Info/num_steps"][()])
            has_step_metrics = bool(datafile["Info/has_step_metrics"][()])
            has_solver_metrics = bool(datafile["Info/has_solver_metrics"][()])
            has_physics_metrics = bool(datafile["Info/has_physics_metrics"][()])

            # Load code state info for traceability and reproducibility
            self.codeinfo = CodeInfo(empty=True)
            self.codeinfo.path = datafile["Info/code/path"][()]
            self.codeinfo.remote = datafile["Info/code/remote"][()]
            self.codeinfo.branch = datafile["Info/code/branch"][()]
            self.codeinfo.commit = datafile["Info/code/commit"][()]
            self.codeinfo.diff = datafile["Info/code/diff"][()]

            # Load solver configurations into the cache for reference
            self._configs = load_solver_configs_to_hdf5(datafile)

            # Load raw data directly into the corresponding array attributes
            self.memory_used = datafile["Data/total/memory_used"][:, :].astype(np.int32)
            self.total_time = datafile["Data/total/total_time"][:, :].astype(np.float32)
            self.total_fps = datafile["Data/total/total_fps"][:, :].astype(np.float32)
            if has_step_metrics:
                self.step_time = datafile["Data/perstep/step_time"][:, :, :].astype(np.float32)
            if has_solver_metrics:
                solv_ns = "Data/perstep/padmm/"
                self.solver_metrics = SolverMetrics(1, 1, 1)  # Placeholder initialization to create the object
                self.solver_metrics.padmm_converged = datafile[f"{solv_ns}converged"][:, :, :].astype(np.int32)
                self.solver_metrics.padmm_iters = datafile[f"{solv_ns}iterations"][:, :, :].astype(np.int32)
                self.solver_metrics.padmm_r_p = datafile[f"{solv_ns}r_p"][:, :, :].astype(np.float32)
                self.solver_metrics.padmm_r_d = datafile[f"{solv_ns}r_d"][:, :, :].astype(np.float32)
                self.solver_metrics.padmm_r_c = datafile[f"{solv_ns}r_c"][:, :, :].astype(np.float32)
            if has_physics_metrics:
                phys_ns = "Data/perstep/physics/"
                self.physics_metrics = PhysicsMetrics(1, 1, 1)  # Placeholder initialization to create the object
                self.physics_metrics.r_eom = datafile[f"{phys_ns}r_eom"][:, :, :].astype(np.float32)
                self.physics_metrics.r_kinematics = datafile[f"{phys_ns}r_kinematics"][:, :, :].astype(np.float32)
                self.physics_metrics.r_cts_joints = datafile[f"{phys_ns}r_cts_joints"][:, :, :].astype(np.float32)
                self.physics_metrics.r_cts_limits = datafile[f"{phys_ns}r_cts_limits"][:, :, :].astype(np.float32)
                self.physics_metrics.r_cts_contacts = datafile[f"{phys_ns}r_cts_contacts"][:, :, :].astype(np.float32)
                self.physics_metrics.r_v_plus = datafile[f"{phys_ns}r_v_plus"][:, :, :].astype(np.float32)
                self.physics_metrics.r_ncp_primal = datafile[f"{phys_ns}r_ncp_primal"][:, :, :].astype(np.float32)
                self.physics_metrics.r_ncp_dual = datafile[f"{phys_ns}r_ncp_dual"][:, :, :].astype(np.float32)
                self.physics_metrics.r_ncp_compl = datafile[f"{phys_ns}r_ncp_compl"][:, :, :].astype(np.float32)
                self.physics_metrics.r_vi_natmap = datafile[f"{phys_ns}r_vi_natmap"][:, :, :].astype(np.float32)
                self.physics_metrics.f_ncp = datafile[f"{phys_ns}f_ncp"][:, :, :].astype(np.float32)
                self.physics_metrics.f_ccp = datafile[f"{phys_ns}f_ccp"][:, :, :].astype(np.float32)

    def render_total_metrics_table(self, path: str | None = None):
        """
        Outputs a formatted table summarizing the total metrics
        (memory used, total time, total FPS) for each solver
        configuration and problem, and optionally saves the
        table to a text file at the specified path.

        Args:
            path (`str`):
                File path to save the table as a text file.

        Raises:
            ValueError: If the total metrics (memory used, total time, total FPS) are not available.
        """
        # Generate the table string for the total metrics summary and print it to the console;
        total_metric_data = [self.memory_used, self.total_time, self.total_fps]
        total_metric_names = ["Memory (MB)", "Total Time (s)", "Total FPS (Hz)"]
        total_metric_formats = [lambda x: f"{x / (1024 * 1024):.2f}", ".2f", ".2f"]
        render_subcolumn_metrics_table(
            title="Solver Benchmark: Total Metrics",
            row_header="Solver Configuration",
            row_titles=self._config_names,
            col_titles=self._problem_names,
            subcol_titles=total_metric_names,
            subcol_data=total_metric_data,
            subcol_formats=total_metric_formats,
            path=path,
            to_console=True,
        )

    def render_step_time_table(self, path: str | None = None, units: Literal["s", "ms", "us"] = "ms"):
        """
        Outputs a formatted table for each problem summarizing the per-step time
        metrics for each solver configuration and problem, and optionally saves
        the table to a text file at the specified path.

        Args:
            path (`str`):
                File path to save the table as a text file.

        Raises:
            ValueError: If the step time metrics are not available.
        """
        if self.step_time_stats is None:
            raise ValueError("Step time metrics are not available in this BenchmarkMetrics instance.")

        # For each problem, generate the table string for the step time metrics summary and print it to the console;
        units_scaling = {"s": 1.0, "ms": 1e3, "us": 1e6}[units]
        for prob_idx, prob_name in enumerate(self._problem_names):
            problem_table_path = f"{path}_{prob_name}.txt" if path is not None else None

            cols: list[ColumnGroup] = []
            cols.append(
                ColumnGroup(
                    header="Solver Configuration",
                    subheaders=["Name"],
                    justify="left",
                    color="white",
                )
            )
            cols.append(
                ColumnGroup(
                    header=f"Step Time ({units})",
                    subheaders=["median", "mean", "max", "min"],
                    subfmt=[".3f", ".3f", ".3f", ".3f"],
                    justify="left",
                    color="cyan",
                )
            )
            rows: list[list[Any]] = []
            for config_idx, config_name in enumerate(self._config_names):
                rows.append(
                    [
                        [config_name],
                        [
                            self.step_time_stats.median[prob_idx, config_idx] * units_scaling,
                            self.step_time_stats.mean[prob_idx, config_idx] * units_scaling,
                            self.step_time_stats.max[prob_idx, config_idx] * units_scaling,
                            self.step_time_stats.min[prob_idx, config_idx] * units_scaling,
                        ],
                    ]
                )
            render_subcolumn_table(
                title=f"Solver Benchmark: Step Time - {prob_name}",
                cols=cols,
                rows=rows,
                max_width=300,
                path=problem_table_path,
                to_console=True,
            )

    def render_padmm_metrics_table(self, path: str | None = None):
        """
        Outputs a formatted table for each problem summarizing the PADMM
        solver metrics (convergence, iterations, residuals) for each solver
        configuration and problem, and optionally saves the table to a text
        file at the specified path.

        Args:
            path (`str`):
                File path to save the table as a text file.

        Raises:
            ValueError: If the PADMM solver metrics are not available.
        """
        if self.solver_metrics is None:
            raise ValueError("PADMM solver metrics are not available in this BenchmarkMetrics instance.")

        # For each problem, generate the table string for the PADMM solver metrics summary and print it to the console;
        for prob_idx, prob_name in enumerate(self._problem_names):
            problem_table_path = f"{path}_{prob_name}.txt" if path is not None else None

            cols: list[ColumnGroup] = []
            cols.append(
                ColumnGroup(
                    header="Solver Configuration",
                    subheaders=["Name"],
                    justify="left",
                    color="white",
                )
            )
            cols.append(
                ColumnGroup(
                    header="Converged",
                    subheaders=["Count", "Rate"],
                    subfmt=["d", ".2%"],
                    justify="left",
                    color="magenta",
                )
            )
            cols.append(
                ColumnGroup(
                    header="Iterations",
                    subheaders=["median", "mean", "max", "min"],
                    subfmt=[".0f", ".0f", ".0f", ".0f"],
                    justify="left",
                    color="cyan",
                )
            )
            cols.append(
                ColumnGroup(
                    header="Primal Residual (r_p)",
                    subheaders=["median", "mean", "max", "min"],
                    subfmt=[".3e", ".3e", ".3e", ".3e"],
                    justify="left",
                    color="red",
                )
            )
            cols.append(
                ColumnGroup(
                    header="Dual Residual (r_d)",
                    subheaders=["median", "mean", "max", "min"],
                    subfmt=[".3e", ".3e", ".3e", ".3e"],
                    justify="left",
                    color="blue",
                )
            )
            cols.append(
                ColumnGroup(
                    header="Complementarity Residual (r_c)",
                    subheaders=["median", "mean", "max", "min"],
                    subfmt=[".3e", ".3e", ".3e", ".3e"],
                    justify="left",
                    color="green",
                )
            )
            rows: list[list[Any]] = []
            for config_idx, config_name in enumerate(self._config_names):
                success_count = self.solver_metrics.padmm_success_stats.count_ones[prob_idx, config_idx]
                fail_count = self.solver_metrics.padmm_success_stats.count_zeros[prob_idx, config_idx]
                total_count = success_count + fail_count
                success_rate = success_count / total_count if total_count > 0 else 0.0
                rows.append(
                    [
                        [config_name],
                        [success_count, success_rate],
                        [
                            self.solver_metrics.padmm_iters_stats.median[prob_idx, config_idx],
                            self.solver_metrics.padmm_iters_stats.mean[prob_idx, config_idx],
                            self.solver_metrics.padmm_iters_stats.max[prob_idx, config_idx],
                            self.solver_metrics.padmm_iters_stats.min[prob_idx, config_idx],
                        ],
                        [
                            self.solver_metrics.padmm_r_p_stats.median[prob_idx, config_idx],
                            self.solver_metrics.padmm_r_p_stats.mean[prob_idx, config_idx],
                            self.solver_metrics.padmm_r_p_stats.max[prob_idx, config_idx],
                            self.solver_metrics.padmm_r_p_stats.min[prob_idx, config_idx],
                        ],
                        [
                            self.solver_metrics.padmm_r_d_stats.median[prob_idx, config_idx],
                            self.solver_metrics.padmm_r_d_stats.mean[prob_idx, config_idx],
                            self.solver_metrics.padmm_r_d_stats.max[prob_idx, config_idx],
                            self.solver_metrics.padmm_r_d_stats.min[prob_idx, config_idx],
                        ],
                        [
                            self.solver_metrics.padmm_r_c_stats.median[prob_idx, config_idx],
                            self.solver_metrics.padmm_r_c_stats.mean[prob_idx, config_idx],
                            self.solver_metrics.padmm_r_c_stats.max[prob_idx, config_idx],
                            self.solver_metrics.padmm_r_c_stats.min[prob_idx, config_idx],
                        ],
                    ],
                )
            render_subcolumn_table(
                title=f"Solver Benchmark: PADMM Convergence - {prob_name}",
                cols=cols,
                rows=rows,
                max_width=300,
                path=problem_table_path,
            )

    def render_physics_metrics_table(self, path: str | None = None):
        """
        Outputs a formatted table for each problem summarizing the physics
        metrics for each solver configuration and problem, and optionally
        saves the table to a text file at the specified path.

        Args:
            path (`str`):
                File path to save the table as a text file.

        Raises:
            ValueError: If the physics metrics are not available.
        """
        if self.physics_metrics is None:
            raise ValueError("Physics metrics are not available in this BenchmarkMetrics instance.")

        # For each problem, generate the table string for the physics metrics summary and print it to the console;
        for prob_idx, prob_name in enumerate(self._problem_names):
            problem_table_path = f"{path}_{prob_name}.txt" if path is not None else None
            cols: list[ColumnGroup] = []
            cols.append(
                ColumnGroup(
                    header="Solver Configuration",
                    subheaders=["Name"],
                    justify="left",
                    color="white",
                )
            )
            cols.append(
                ColumnGroup(
                    header="r_eom",
                    subheaders=["median", "mean", "max", "min"],
                    subfmt=[".3e", ".3e", ".3e", ".3e"],
                    justify="left",
                    color="cyan",
                )
            )
            cols.append(
                ColumnGroup(
                    header="r_kinematics",
                    subheaders=["median", "mean", "max", "min"],
                    subfmt=[".3e", ".3e", ".3e", ".3e"],
                    justify="left",
                    color="cyan",
                )
            )
            cols.append(
                ColumnGroup(
                    header="r_cts_joints",
                    subheaders=["median", "mean", "max", "min"],
                    subfmt=[".3e", ".3e", ".3e", ".3e"],
                    justify="left",
                    color="cyan",
                )
            )
            cols.append(
                ColumnGroup(
                    header="r_cts_limits",
                    subheaders=["median", "mean", "max", "min"],
                    subfmt=[".3e", ".3e", ".3e", ".3e"],
                    justify="left",
                    color="cyan",
                )
            )
            cols.append(
                ColumnGroup(
                    header="r_cts_contacts",
                    subheaders=["median", "mean", "max", "min"],
                    subfmt=[".3e", ".3e", ".3e", ".3e"],
                    justify="left",
                    color="cyan",
                )
            )
            cols.append(
                ColumnGroup(
                    header="r_v_plus",
                    subheaders=["median", "mean", "max", "min"],
                    subfmt=[".3e", ".3e", ".3e", ".3e"],
                    justify="left",
                    color="cyan",
                )
            )
            cols.append(
                ColumnGroup(
                    header="r_ncp_primal",
                    subheaders=["median", "mean", "max", "min"],
                    subfmt=[".3e", ".3e", ".3e", ".3e"],
                    justify="left",
                    color="cyan",
                )
            )
            cols.append(
                ColumnGroup(
                    header="r_ncp_dual",
                    subheaders=["median", "mean", "max", "min"],
                    subfmt=[".3e", ".3e", ".3e", ".3e"],
                    justify="left",
                    color="cyan",
                )
            )
            cols.append(
                ColumnGroup(
                    header="r_ncp_compl",
                    subheaders=["median", "mean", "max", "min"],
                    subfmt=[".3e", ".3e", ".3e", ".3e"],
                    justify="left",
                    color="cyan",
                )
            )
            cols.append(
                ColumnGroup(
                    header="r_vi_natmap",
                    subheaders=["median", "mean", "max", "min"],
                    subfmt=[".3e", ".3e", ".3e", ".3e"],
                    justify="left",
                    color="cyan",
                )
            )
            cols.append(
                ColumnGroup(
                    header="f_ncp",
                    subheaders=["median", "mean", "max", "min"],
                    subfmt=[".3e", ".3e", ".3e", ".3e"],
                    justify="left",
                    color="cyan",
                )
            )
            cols.append(
                ColumnGroup(
                    header="f_ccp",
                    subheaders=["median", "mean", "max", "min"],
                    subfmt=[".3e", ".3e", ".3e", ".3e"],
                    justify="left",
                    color="cyan",
                )
            )
            rows: list[list[Any]] = []
            for config_idx, config_name in enumerate(self._config_names):
                rows.append(
                    [
                        [config_name],
                        [
                            self.physics_metrics.r_eom_stats.median[prob_idx, config_idx],
                            self.physics_metrics.r_eom_stats.mean[prob_idx, config_idx],
                            self.physics_metrics.r_eom_stats.max[prob_idx, config_idx],
                            self.physics_metrics.r_eom_stats.min[prob_idx, config_idx],
                        ],
                        [
                            self.physics_metrics.r_kinematics_stats.median[prob_idx, config_idx],
                            self.physics_metrics.r_kinematics_stats.mean[prob_idx, config_idx],
                            self.physics_metrics.r_kinematics_stats.max[prob_idx, config_idx],
                            self.physics_metrics.r_kinematics_stats.min[prob_idx, config_idx],
                        ],
                        [
                            self.physics_metrics.r_cts_joints_stats.median[prob_idx, config_idx],
                            self.physics_metrics.r_cts_joints_stats.mean[prob_idx, config_idx],
                            self.physics_metrics.r_cts_joints_stats.max[prob_idx, config_idx],
                            self.physics_metrics.r_cts_joints_stats.min[prob_idx, config_idx],
                        ],
                        [
                            self.physics_metrics.r_cts_limits_stats.median[prob_idx, config_idx],
                            self.physics_metrics.r_cts_limits_stats.mean[prob_idx, config_idx],
                            self.physics_metrics.r_cts_limits_stats.max[prob_idx, config_idx],
                            self.physics_metrics.r_cts_limits_stats.min[prob_idx, config_idx],
                        ],
                        [
                            self.physics_metrics.r_cts_contacts_stats.median[prob_idx, config_idx],
                            self.physics_metrics.r_cts_contacts_stats.mean[prob_idx, config_idx],
                            self.physics_metrics.r_cts_contacts_stats.max[prob_idx, config_idx],
                            self.physics_metrics.r_cts_contacts_stats.min[prob_idx, config_idx],
                        ],
                        [
                            self.physics_metrics.r_v_plus_stats.median[prob_idx, config_idx],
                            self.physics_metrics.r_v_plus_stats.mean[prob_idx, config_idx],
                            self.physics_metrics.r_v_plus_stats.max[prob_idx, config_idx],
                            self.physics_metrics.r_v_plus_stats.min[prob_idx, config_idx],
                        ],
                        [
                            self.physics_metrics.r_ncp_primal_stats.median[prob_idx, config_idx],
                            self.physics_metrics.r_ncp_primal_stats.mean[prob_idx, config_idx],
                            self.physics_metrics.r_ncp_primal_stats.max[prob_idx, config_idx],
                            self.physics_metrics.r_ncp_primal_stats.min[prob_idx, config_idx],
                        ],
                        [
                            self.physics_metrics.r_ncp_dual_stats.median[prob_idx, config_idx],
                            self.physics_metrics.r_ncp_dual_stats.mean[prob_idx, config_idx],
                            self.physics_metrics.r_ncp_dual_stats.max[prob_idx, config_idx],
                            self.physics_metrics.r_ncp_dual_stats.min[prob_idx, config_idx],
                        ],
                        [
                            self.physics_metrics.r_ncp_compl_stats.median[prob_idx, config_idx],
                            self.physics_metrics.r_ncp_compl_stats.mean[prob_idx, config_idx],
                            self.physics_metrics.r_ncp_compl_stats.max[prob_idx, config_idx],
                            self.physics_metrics.r_ncp_compl_stats.min[prob_idx, config_idx],
                        ],
                        [
                            self.physics_metrics.r_vi_natmap_stats.median[prob_idx, config_idx],
                            self.physics_metrics.r_vi_natmap_stats.mean[prob_idx, config_idx],
                            self.physics_metrics.r_vi_natmap_stats.max[prob_idx, config_idx],
                            self.physics_metrics.r_vi_natmap_stats.min[prob_idx, config_idx],
                        ],
                        [
                            self.physics_metrics.f_ncp_stats.median[prob_idx, config_idx],
                            self.physics_metrics.f_ncp_stats.mean[prob_idx, config_idx],
                            self.physics_metrics.f_ncp_stats.max[prob_idx, config_idx],
                            self.physics_metrics.f_ncp_stats.min[prob_idx, config_idx],
                        ],
                        [
                            self.physics_metrics.f_ccp_stats.median[prob_idx, config_idx],
                            self.physics_metrics.f_ccp_stats.mean[prob_idx, config_idx],
                            self.physics_metrics.f_ccp_stats.max[prob_idx, config_idx],
                            self.physics_metrics.f_ccp_stats.min[prob_idx, config_idx],
                        ],
                    ]
                )
            render_subcolumn_table(
                title=f"Solver Benchmark: Physics Metrics - {prob_name}",
                cols=cols,
                rows=rows,
                max_width=650,
                path=problem_table_path,
            )

    def render_padmm_metrics_plots(self, path: str):
        """
        Generates time-series plots of the PADMM solver metrics
        (convergence, iterations, residuals) across the simulation
        steps for each solver configuration and problem, and
        optionally saves the plots to a file at the specified path.

        Args:
            path (`str`):
                Target file path of the generated plot image.\n

        Raises:
            ValueError: If the PADMM solver metrics are not available.
        """
        # Ensure that the PADMM solver metrics are available before attempting to render the plots
        if self.solver_metrics is None:
            raise ValueError("PADMM solver metrics are not available in this BenchmarkMetrics instance.")

        # Attempt to import matplotlib for plotting, and raise an informative error if it's not installed
        try:
            import matplotlib.pyplot as plt
        except Exception as e:
            raise ImportError(
                "matplotlib is required to render PADMM metrics plots. Please install matplotlib and try again."
            ) from e

        # Generate time-series plots of the PADMM solver metrics across the simulation steps of each problem:
        # - For each problem we create a figure
        # - Each figure has a subplot for each PADMM metric in (iterations, r_p, r_d, r_c)
        # - Within each subplot we plot a metric curve for each solver configuration
        for prob_idx, prob_name in enumerate(self._problem_names):
            fig, axs = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle(f"PADMM Metrics vs Simulation Steps - {prob_name}", fontsize=16)
            titles = [
                "PADMM Iterations",
                "PADMM Primal Residual",
                "PADMM Dual Residual",
                "PADMM Complementary Residual",
            ]
            names = ["iterations", "r_p", "r_d", "r_c"]
            data = [
                self.solver_metrics.padmm_iters[prob_idx, :, :],
                self.solver_metrics.padmm_r_p[prob_idx, :, :],
                self.solver_metrics.padmm_r_d[prob_idx, :, :],
                self.solver_metrics.padmm_r_c[prob_idx, :, :],
            ]
            for metric_idx, (title, name, array) in enumerate(zip(titles, names, data, strict=True)):
                ax = axs[metric_idx // 2, metric_idx % 2]
                for config_idx, config_name in enumerate(self._config_names):
                    ax.plot(
                        np.arange(self.num_steps),
                        array[config_idx, :],
                        label=config_name,
                        marker="o",
                        markersize=4,
                    )
                ax.set_title(title)
                ax.set_xlabel("Simulation Step")
                ax.set_ylabel(name)
                ax.grid()
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])

            # Get handles/labels from any one subplot (since they are identical)
            handles, labels = axs.flat[0].get_legend_handles_labels()

            # Add ONE legend for the whole figure, centered at the bottom
            fig.legend(
                handles,
                labels,
                loc="lower center",
                ncol=len(labels),  # put entries in one row
                bbox_to_anchor=(0.5, 0.02),  # (x, y) in figure coordinates
                frameon=False,
            )

            # Make room at the bottom for the legend
            fig.subplots_adjust(bottom=0.15)

            # If a path is provided, also save the plot to an image file at the specified path
            if path is not None:
                # Check if the directory for the specified path exists, and if not, create it
                path_dir = os.path.dirname(path)
                if path_dir and not os.path.exists(path_dir):
                    raise ValueError(
                        f"Directory for path '{path}' does not exist. "
                        "Please create the directory before saving the plot."
                    )
                fig_path = path + f"_{prob_name}.pdf"
                plt.savefig(fig_path, format="pdf", dpi=300, bbox_inches="tight")

            # Close the figure to free up memory after saving
            # (or if not saving) before the next iteration
            plt.close(fig)

    def render_physics_metrics_plots(self, path: str):
        """
        Generates time-series plots of the physics metrics (e.g.,
        constraint violation etc) across the simulation steps for
        each solver configuration and problem, and optionally
        saves the plots to a file at the specified path.

        Args:
            path (`str`):
                Target file path of the generated plot image.\n

        Raises:
            ValueError: If the physics metrics are not available.
        """
        # Ensure that the physics metrics are available before attempting to render the plots
        if self.physics_metrics is None:
            raise ValueError("Physics metrics are not available in this BenchmarkMetrics instance.")

        # Attempt to import matplotlib for plotting, and raise an informative error if it's not installed
        try:
            import matplotlib.pyplot as plt
        except Exception as e:
            raise ImportError(
                "matplotlib is required to render physics metrics plots. Please install matplotlib and try again."
            ) from e

        # Generate time-series plots of the physics solver metrics across the simulation steps of each problem:
        # - For each problem we create a figure
        # - Each figure has a subplot for each physics metric in (constraint violation, energy, etc.)
        # - Within each subplot we plot a metric curve for each solver configuration
        for prob_idx, prob_name in enumerate(self._problem_names):
            fig, axs = plt.subplots(3, 4, figsize=(20, 15))
            fig.suptitle(f"Physics Metrics vs Simulation Steps - {prob_name}", fontsize=16)
            equations = [
                "$\\Vert \\, M \\, (u^+ - u^-) - dt \\, (h + J_a^T \\, \\tau) - J^T \\, \\lambda \\, \\Vert_\\infty $",
                "$\\Vert \\, J_j \\cdot u^+ \\, \\Vert_\\infty $",
                "$\\Vert \\, f_j(q) \\, \\Vert_\\infty $",
                "$\\Vert \\, f_l(q) \\, \\Vert_\\infty $",
                "$\\Vert \\, f_{c,N}(q) \\, \\Vert_\\infty $",
                "$\\Vert \\, v^+ - D \\cdot \\lambda - v_f \\, \\Vert_\\infty $",
                "$\\Vert \\, \\lambda - P_K(\\lambda) \\, \\Vert_\\infty $",
                "$\\Vert \\, v_a^+ - P_{K^*}(v_a^+) \\, \\Vert_\\infty $",
                "$\\Vert \\, \\lambda^T \\, v_a^+ \\, \\Vert_\\infty $",
                "$\\Vert \\, \\lambda - P_{K^*}(\\lambda - v_a^+(\\lambda)) \\, \\Vert_\\infty $",
                "$ 0.5 \\, \\lambda^T \\, D \\, \\lambda + \\lambda^T \\, (v_f + s) $",
                "$ 0.5 \\, \\lambda^T \\, D \\, \\lambda + v_f^T \\, \\lambda $",
            ]
            titles = [
                f"Equations-of-Motion Residual \n ({equations[0]})",
                f"Joint Kinematics Constraint Residual \n ({equations[1]})",
                f"Joints Constraint Residual \n ({equations[2]})",
                f"Limits Constraint Residual \n ({equations[3]})",
                f"Contacts Constraint Residual \n ({equations[4]})",
                f"Post-Event Constraint Velocity Residual \n ({equations[5]})",
                f"NCP Primal Residual \n ({equations[6]})",
                f"NCP Dual Residual\n ({equations[7]})",
                f"NCP Complementary Residual\n ({equations[8]})",
                f"VI Natural-Map Residual\n ({equations[9]})",
                f"NCP Objective \n ({equations[10]})",
                f"CCP Objective \n ({equations[11]})",
            ]
            names = [
                "r_eom",
                "r_kinematics",
                "r_cts_joints",
                "r_cts_limits",
                "r_cts_contacts",
                "r_v_plus",
                "r_ncp_primal",
                "r_ncp_dual",
                "r_ncp_compl",
                "r_vi_natmap",
                "f_ncp",
                "f_ccp",
            ]
            data = [
                self.physics_metrics.r_eom[prob_idx, :, :],
                self.physics_metrics.r_kinematics[prob_idx, :, :],
                self.physics_metrics.r_cts_joints[prob_idx, :, :],
                self.physics_metrics.r_cts_limits[prob_idx, :, :],
                self.physics_metrics.r_cts_contacts[prob_idx, :, :],
                self.physics_metrics.r_v_plus[prob_idx, :, :],
                self.physics_metrics.r_ncp_primal[prob_idx, :, :],
                self.physics_metrics.r_ncp_dual[prob_idx, :, :],
                self.physics_metrics.r_ncp_compl[prob_idx, :, :],
                self.physics_metrics.r_vi_natmap[prob_idx, :, :],
                self.physics_metrics.f_ncp[prob_idx, :, :],
                self.physics_metrics.f_ccp[prob_idx, :, :],
            ]
            for metric_idx, (title, name, array) in enumerate(zip(titles, names, data, strict=True)):
                ax = axs[metric_idx // 4, metric_idx % 4]
                for config_idx, config_name in enumerate(self._config_names):
                    ax.plot(
                        np.arange(self.num_steps),
                        array[config_idx, :],
                        label=config_name,
                        marker="o",
                        markersize=4,
                    )
                ax.set_title(title)
                ax.set_xlabel("Simulation Step")
                ax.set_ylabel(name)
                ax.grid()
            plt.tight_layout(rect=[0, 0.03, 1, 0.95], h_pad=3.0, w_pad=2.0)

            # Get handles/labels from any one subplot (since they are identical)
            handles, labels = axs.flat[0].get_legend_handles_labels()

            # Add ONE legend for the whole figure, centered at the bottom
            fig.legend(
                handles,
                labels,
                loc="lower center",
                ncol=len(labels),  # put entries in one row
                bbox_to_anchor=(0.5, 0.02),  # (x, y) in figure coordinates
                frameon=False,
            )

            # Make room at the bottom for the legend
            fig.subplots_adjust(bottom=0.15)

            # If a path is provided, also save the plot to an image file at the specified path
            if path is not None:
                # Check if the directory for the specified path exists, and if not, create it
                path_dir = os.path.dirname(path)
                if path_dir and not os.path.exists(path_dir):
                    raise ValueError(
                        f"Directory for path '{path}' does not exist. "
                        "Please create the directory before saving the plot."
                    )
                fig_path = path + f"_{prob_name}.pdf"
                plt.savefig(fig_path, format="pdf", dpi=300, bbox_inches="tight")

            # Close the figure to free up memory after saving
            # (or if not saving) before the next iteration
            plt.close(fig)
