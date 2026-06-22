# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import warp as wp
from asv_runner.benchmarks.mark import SkipNotImplemented, skip_benchmark_if

wp.config.enable_backward = False
wp.config.log_level = wp.LOG_WARNING

import importlib

import newton.examples
from newton.viewer import ViewerNull

ISAACGYM_ENVS_REPO_URL = "https://github.com/isaac-sim/IsaacGymEnvs.git"
ISAACGYM_NUT_BOLT_FOLDER = "assets/factory/mesh/factory_nut_bolt"

try:
    from newton.examples import download_external_git_folder as _download_external_git_folder
except ImportError:
    from newton._src.utils.download_assets import download_git_folder as _download_external_git_folder


def _import_example_class(module_names: list[str]):
    """Import and return the ``Example`` class from candidate modules.

    Args:
        module_names: Ordered module names to try importing.

    Returns:
        The first successfully imported module's ``Example`` class.

    Raises:
        SkipNotImplemented: If none of the module names can be imported.
    """
    for module_name in module_names:
        try:
            module = importlib.import_module(module_name)
        except ModuleNotFoundError:
            continue
        return module.Example

    raise SkipNotImplemented


class FastExampleContactSdfDefaults:
    """Benchmark the SDF nut-bolt example default configuration."""

    repeat = 2
    number = 1

    def setup_cache(self):
        _download_external_git_folder(ISAACGYM_ENVS_REPO_URL, ISAACGYM_NUT_BOLT_FOLDER)

    def setup(self):
        example_cls = _import_example_class(
            [
                "newton.examples.contacts.example_nut_bolt_sdf",
            ]
        )
        self.num_frames = 20
        if hasattr(newton.examples, "default_args") and hasattr(example_cls, "create_parser"):
            args = newton.examples.default_args(example_cls.create_parser())
            self.example = example_cls(ViewerNull(num_frames=self.num_frames), args)
        else:
            self.example = example_cls(
                viewer=ViewerNull(num_frames=self.num_frames),
                world_count=100,
                num_per_world=1,
                scene="nut_bolt",
                solver="mujoco",
                test_mode=False,
            )

    @skip_benchmark_if(wp.get_cuda_device_count() == 0)
    def time_simulate(self):
        for _ in range(self.num_frames):
            self.example.step()
        wp.synchronize_device()


class FastExampleContactSdfTunnelingSweep:
    """Track SDF nut-bolt tunneling sensitivity across contact settings."""

    _CONFIGS = {
        "baseline_substeps4": {
            "sim_substeps": 4,
        },
        "baseline_substeps8": {
            "sim_substeps": 8,
        },
        "baseline_substeps12": {
            "sim_substeps": 12,
        },
        "collide_substeps4": {
            "sim_substeps": 4,
            "collide_every_substep": True,
        },
        "sticky_substeps4": {
            "sim_substeps": 4,
            "contact_matching": "sticky",
        },
        "large_hash_substeps4": {
            "sim_substeps": 4,
            "contact_reduction_hashtable_size_factor": 1.0,
        },
        "no_reduction_substeps4": {
            "sim_substeps": 4,
            "reduce_contacts": False,
        },
        "gap_10mm_substeps4": {
            "sim_substeps": 4,
            "shape_gap": 0.01,
        },
    }

    params = (list(_CONFIGS),)
    param_names = ["config"]
    repeat = 1
    number = 1
    num_frames = 120
    world_count = 4

    def setup_cache(self):
        _download_external_git_folder(ISAACGYM_ENVS_REPO_URL, ISAACGYM_NUT_BOLT_FOLDER)

    def setup(self, config):
        example_cls = _import_example_class(
            [
                "newton.examples.contacts.example_nut_bolt_sdf",
            ]
        )
        args = newton.examples.default_args(example_cls.create_parser())
        cfg = self._CONFIGS[config]
        args.world_count = self.world_count
        args.num_per_world = 1
        args.sim_substeps = cfg["sim_substeps"]
        args.collide_every_substep = cfg.get("collide_every_substep", False)
        args.track_contact_metrics = True
        args.reduce_contacts = cfg.get("reduce_contacts", True)
        args.contact_matching = cfg.get("contact_matching", "disabled")
        args.contact_reduction_hashtable_size_factor = cfg.get("contact_reduction_hashtable_size_factor", 0.25)
        args.shape_gap = cfg.get("shape_gap", 0.005)
        args.rigid_contact_max = 500 * self.world_count
        args.max_triangle_pairs = 1_000_000
        args.broad_phase = "sap"
        self.example = example_cls(ViewerNull(num_frames=self.num_frames), args)
        self._metrics = None

    def _run(self):
        if self._metrics is not None:
            return self._metrics

        for _ in range(self.num_frames):
            self.example.step()
        wp.synchronize_device()

        metrics = self.example.get_metrics()
        if not metrics:
            raise RuntimeError("SDF nut-bolt metrics were not collected")
        self._metrics = metrics
        return metrics

    @skip_benchmark_if(wp.get_cuda_device_count() == 0)
    def time_simulate(self, config):
        self._run()

    @skip_benchmark_if(wp.get_cuda_device_count() == 0)
    def track_tunneling_depth(self, config):
        metrics = self._run()
        return max(0.0, -metrics["min_nut_center_z_relative_to_bolt"])

    track_tunneling_depth.unit = "m"

    @skip_benchmark_if(wp.get_cuda_device_count() == 0)
    def track_max_nut_drop(self, config):
        return self._run()["max_nut_drop"]

    track_max_nut_drop.unit = "m"

    @skip_benchmark_if(wp.get_cuda_device_count() == 0)
    def track_max_bolt_displacement(self, config):
        return self._run()["max_bolt_displacement"]

    track_max_bolt_displacement.unit = "m"

    @skip_benchmark_if(wp.get_cuda_device_count() == 0)
    def track_max_contact_count(self, config):
        return self._run()["max_contact_count"]

    track_max_contact_count.unit = "contacts"


class FastExampleContactHydroWorkingDefaults:
    """Benchmark the hydroelastic nut-bolt example default configuration."""

    repeat = 2
    number = 1

    def setup_cache(self):
        _download_external_git_folder(ISAACGYM_ENVS_REPO_URL, ISAACGYM_NUT_BOLT_FOLDER)

    def setup(self):
        example_cls = _import_example_class(
            [
                "newton.examples.contacts.example_nut_bolt_hydro",
            ]
        )
        self.num_frames = 20
        if hasattr(newton.examples, "default_args") and hasattr(example_cls, "create_parser"):
            args = newton.examples.default_args(example_cls.create_parser())
            self.example = example_cls(ViewerNull(num_frames=self.num_frames), args)
        else:
            self.example = example_cls(
                viewer=ViewerNull(num_frames=self.num_frames),
                world_count=20,
                num_per_world=1,
                scene="nut_bolt",
                solver="mujoco",
                test_mode=False,
            )

    @skip_benchmark_if(wp.get_cuda_device_count() == 0)
    def time_simulate(self):
        for _ in range(self.num_frames):
            self.example.step()
        wp.synchronize_device()


class FastExampleContactPyramidDefaults:
    """Benchmark the box pyramid example with default configuration."""

    repeat = 2
    number = 1

    def setup(self):
        example_cls = _import_example_class(
            [
                "newton.examples.contacts.example_pyramid",
            ]
        )
        self.num_frames = 20
        if hasattr(newton.examples, "default_args") and hasattr(example_cls, "create_parser"):
            args = newton.examples.default_args(example_cls.create_parser())
            self.example = example_cls(ViewerNull(num_frames=self.num_frames), args)
        else:
            self.example = example_cls(
                viewer=ViewerNull(num_frames=self.num_frames),
                solver="xpbd",
                test_mode=False,
            )

    @skip_benchmark_if(wp.get_cuda_device_count() == 0)
    def time_simulate(self):
        for _ in range(self.num_frames):
            self.example.step()
        wp.synchronize_device()


if __name__ == "__main__":
    import argparse

    from newton.utils import run_benchmark

    benchmark_list = {
        "FastExampleContactSdfDefaults": FastExampleContactSdfDefaults,
        "FastExampleContactSdfTunnelingSweep": FastExampleContactSdfTunnelingSweep,
        "FastExampleContactHydroWorkingDefaults": FastExampleContactHydroWorkingDefaults,
        "FastExampleContactPyramidDefaults": FastExampleContactPyramidDefaults,
    }

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "-b",
        "--bench",
        default=None,
        action="append",
        choices=benchmark_list.keys(),
        help="Run a specific benchmark; may be repeated to run multiple (e.g., --bench A --bench B).",
    )
    args = parser.parse_known_args()[0]

    if args.bench is None:
        benchmarks = benchmark_list.keys()
    else:
        benchmarks = args.bench

    for key in benchmarks:
        benchmark = benchmark_list[key]
        run_benchmark(benchmark)
