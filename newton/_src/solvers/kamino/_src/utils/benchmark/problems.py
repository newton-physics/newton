# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections.abc import Callable
from dataclasses import dataclass

import warp as wp

import newton.utils

from ...core.builder import ModelBuilderKamino
from ...models import basics
from ...models.builders.utils import (
    add_ground_box,
    make_homogeneous_builder,
    set_uniform_body_pose_offset,
)
from ...utils import logger as msg
from ..io.usd import USDImporter

###
# Module interface
###

__all__ = [
    "BenchmarkProblemNameToConfigFn",
    "CameraConfig",
    "ControlConfig",
    "ProblemConfig",
    "ProblemDimensions",
    "ProblemSet",
    "make_benchmark_problems",
    "save_problem_dimensions_to_hdf5",
]

###
# Types
###


@dataclass
class ProblemDimensions:
    num_body_dofs: int = -1
    num_joint_dofs: int = -1
    min_delassus_dim: int = -1
    max_delassus_dim: int = -1


@dataclass
class ControlConfig:
    disable_controller: bool = False
    decimation: int | list[int] | None = None
    scale: float | list[float] | None = None


# TODO: Use set_camera_lookat params instead
@dataclass
class CameraConfig:
    position: tuple[float, float, float]
    pitch: float
    yaw: float


ProblemConfig = tuple[ModelBuilderKamino | Callable, ControlConfig | None, CameraConfig | None]
"""
Defines the configurations for a single benchmark problem.

This contains:
- A model builder that constructs the simulation worlds for the benchmark problem, or a callable
  taking no arguments returning such a builder (for deferred loading of the problem assets).
- Optional control configurations for perturbing the benchmark problem.
- Optional camera configurations for visualizing the benchmark problem.
"""


ProblemSet = dict[str, ProblemConfig]
"""
Defines a set of benchmark problems, indexed by a string name.

Each entry contains the configurations for a single
benchmark problem, including the model builder and
optional camera configurations for visualization.
"""


###
# Problem Definitions
###


def make_benchmark_problem_fourbar(
    num_worlds: int = 1,
    gravity: bool = True,
    ground: bool = True,
) -> ProblemConfig:
    def builder_fn():
        builder = make_homogeneous_builder(
            num_worlds=num_worlds,
            build_fn=basics.build_boxes_fourbar,
            ground=ground,
        )
        for w in range(num_worlds):
            builder.gravity[w].enabled = gravity
        return builder

    control = ControlConfig(decimation=20, scale=10.0)
    camera = CameraConfig(
        position=(-0.2, -0.5, 0.1),
        pitch=-5.0,
        yaw=70.0,
    )
    return builder_fn, control, camera


def make_benchmark_problem_dr_legs(
    num_worlds: int = 1,
    gravity: bool = True,
    ground: bool = True,
) -> ProblemConfig:
    # Set the path to the external USD assets
    asset_path = newton.utils.download_asset("disneyresearch")
    asset_file = str(asset_path / "dr_legs/usd" / "dr_legs_with_meshes_and_boxes.usda")

    def builder_fn():
        # Create a model builder from the imported USD
        msg.notif("Constructing builder from imported USD ...")
        importer = USDImporter()
        builder: ModelBuilderKamino = make_homogeneous_builder(
            num_worlds=num_worlds,
            build_fn=importer.import_from,
            load_static_geometry=True,
            source=asset_file,
            load_drive_dynamics=True,
            use_angular_drive_scaling=True,
        )
        # Offset the model to place it above the ground
        # NOTE: The USD model is centered at the origin
        offset = wp.transformf(0.0, 0.0, 0.265, 0.0, 0.0, 0.0, 1.0)
        set_uniform_body_pose_offset(builder=builder, offset=offset)
        # Add a static collision geometry for the plane
        if ground:
            for w in range(num_worlds):
                add_ground_box(builder, world_index=w)
        # Set gravity
        for w in range(builder.num_worlds):
            builder.gravity[w].enabled = gravity
        return builder

    # Set control configurations
    control = ControlConfig(decimation=20, scale=5.0)
    # Set the camera configuration for better visualization of the system
    camera = CameraConfig(
        position=(0.6, 0.6, 0.3),
        pitch=-10.0,
        yaw=225.0,
    )
    return builder_fn, control, camera


###
# Problem Set Generator
###

BenchmarkProblemNameToConfigFn: dict[str, Callable[..., ProblemConfig]] = {
    "fourbar": make_benchmark_problem_fourbar,
    "dr_legs": make_benchmark_problem_dr_legs,
}
"""
Defines a mapping from benchmark problem names to their
corresponding problem configuration generator functions.
"""


def make_benchmark_problems(
    names: list[str],
    num_worlds: int = 1,
    gravity: bool = True,
    ground: bool = True,
) -> ProblemSet:
    # Ensure that problem names are provided and valid
    if names is None:
        raise ValueError("Problem names must be provided as a list of strings.")

    # Define common generator kwargs for all problems to avoid repetition
    generator_kwargs = {"num_worlds": num_worlds, "gravity": gravity, "ground": ground}

    # Generate the problem configurations for each specified problem name
    problems: ProblemSet = {}
    for name in names:
        if name not in BenchmarkProblemNameToConfigFn.keys():
            raise ValueError(
                f"Unsupported problem name: {name}.\nSupported names are: {list(BenchmarkProblemNameToConfigFn.keys())}"
            )

        problems[name] = BenchmarkProblemNameToConfigFn[name](**generator_kwargs)
    return problems


def save_problem_dimensions_to_hdf5(problem_dims: dict[str, ProblemDimensions], datafile):
    for problem_name, dims in problem_dims.items():
        scope = f"Problems/{problem_name}"
        datafile[f"{scope}/num_body_dofs"] = dims.num_body_dofs
        datafile[f"{scope}/num_joint_dofs"] = dims.num_joint_dofs
        datafile[f"{scope}/min_delassus_dim"] = dims.min_delassus_dim
        datafile[f"{scope}/max_delassus_dim"] = dims.max_delassus_dim
