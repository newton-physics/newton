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

import os

import numpy as np
import warp as wp

import newton


def get_source_directory() -> str:
    return os.path.realpath(os.path.dirname(__file__))


def get_asset_directory() -> str:
    return os.path.join(get_source_directory(), "assets")


def get_asset(filename: str) -> str:
    return os.path.join(get_asset_directory(), filename)


def run(example):
    if hasattr(example, "gui") and hasattr(example.viewer, "register_ui_callback"):
        example.viewer.register_ui_callback(lambda ui: example.gui(ui), position="side")

    while example.viewer.is_running():
        with wp.ScopedTimer("step", active=False):
            example.step()

        with wp.ScopedTimer("render", active=False):
            example.render()

    example.viewer.close()


def compute_env_offsets(
    num_envs: int, env_offset: tuple[float, float, float] = (5.0, 5.0, 0.0), up_axis: newton.AxisType = newton.Axis.Z
):
    # raise deprecation warning
    import warnings  # noqa: PLC0415

    warnings.warn(
        (
            "compute_env_offsets is deprecated and will be removed in a future version. "
            "Use the builder.replicate() function instead."
        ),
        stacklevel=2,
    )

    # compute positional offsets per environment
    env_offset = np.array(env_offset)
    nonzeros = np.nonzero(env_offset)[0]
    num_dim = nonzeros.shape[0]
    if num_dim > 0:
        side_length = int(np.ceil(num_envs ** (1.0 / num_dim)))
        env_offsets = []
        if num_dim == 1:
            for i in range(num_envs):
                env_offsets.append(i * env_offset)
        elif num_dim == 2:
            for i in range(num_envs):
                d0 = i // side_length
                d1 = i % side_length
                offset = np.zeros(3)
                offset[nonzeros[0]] = d0 * env_offset[nonzeros[0]]
                offset[nonzeros[1]] = d1 * env_offset[nonzeros[1]]
                env_offsets.append(offset)
        elif num_dim == 3:
            for i in range(num_envs):
                d0 = i // (side_length * side_length)
                d1 = (i // side_length) % side_length
                d2 = i % side_length
                offset = np.zeros(3)
                offset[0] = d0 * env_offset[0]
                offset[1] = d1 * env_offset[1]
                offset[2] = d2 * env_offset[2]
                env_offsets.append(offset)
        env_offsets = np.array(env_offsets)
    else:
        env_offsets = np.zeros((num_envs, 3))
    min_offsets = np.min(env_offsets, axis=0)
    correction = min_offsets + (np.max(env_offsets, axis=0) - min_offsets) / 2.0
    # ensure the envs are not shifted below the ground plane
    correction[newton.Axis.from_any(up_axis)] = 0.0
    env_offsets -= correction
    return env_offsets


def create_parser():
    """Create a base argument parser with common parameters for Newton examples.

    Individual examples can use this as a parent parser and add their own
    specific arguments.

    Returns:
        argparse.ArgumentParser: Base parser with common arguments
    """
    import argparse  # noqa: PLC0415

    # add_help=False since this is a parent parser
    parser = argparse.ArgumentParser(add_help=False, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--device", type=str, default=None, help="Override the default Warp device.")
    parser.add_argument(
        "--viewer",
        type=str,
        default="gl",
        choices=["gl", "usd", "rerun", "null"],
        help="Viewer to use (gl, usd, rerun, or null).",
    )
    parser.add_argument(
        "--output-path", type=str, default=None, help="Path to the output USD file (required for usd viewer)."
    )
    parser.add_argument("--num-frames", type=int, default=100, help="Total number of frames.")
    parser.add_argument(
        "--headless",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Whether to initialize the viewer headless (for OpenGL viewer only).",
    )

    return parser


def init(parser=None):
    """Initialize Newton example components from parsed arguments.

    Args:
        parser: Parsed arguments from argparse (should include arguments from
              create_parser())

    Returns:
        tuple: (viewer, args) where viewer is configured based on args.viewer

    Raises:
        ValueError: If invalid viewer type or missing required arguments
    """
    import warp as wp  # noqa: PLC0415

    import newton.viewer  # noqa: PLC0415

    # parse args
    if parser is None:
        parser = create_parser()

    args = parser.parse_known_args()[0]

    # Set device if specified
    if args.device:
        wp.set_device(args.device)

    # Create viewer based on type
    if args.viewer == "gl":
        viewer = newton.viewer.ViewerGL(headless=args.headless)
    elif args.viewer == "usd":
        if args.output_path is None:
            raise ValueError("--output-path is required when using usd viewer")
        viewer = newton.viewer.ViewerUSD(output_path=args.output_path, num_frames=args.num_frames)
    elif args.viewer == "rerun":
        viewer = newton.viewer.ViewerRerun()
    elif args.viewer == "null":
        viewer = newton.viewer.ViewerNull(num_frames=args.num_frames)
    else:
        raise ValueError(f"Invalid viewer: {args.viewer}")

    return viewer, args


def main():
    """Main entry point for running examples via 'python -m newton.examples <example_name>'."""
    import runpy  # noqa: PLC0415
    import sys  # noqa: PLC0415

    # Map short names to full module paths
    example_map = {
        "basic_pendulum": "newton.examples.basic.example_basic_pendulum",
        "basic_urdf": "newton.examples.basic.example_basic_urdf",
        "basic_viewer": "newton.examples.basic.example_basic_viewer",
        "basic_shapes": "newton.examples.basic.example_basic_shapes",
        "basic_joints": "newton.examples.basic.example_basic_joints",
        "cloth_bending": "newton.examples.cloth.example_cloth_bending",
        "cloth_franka": "newton.examples.cloth.example_cloth_franka",
        "cloth_hanging": "newton.examples.cloth.example_cloth_hanging",
        "cloth_style3d": "newton.examples.cloth.example_cloth_style3d",
        "ik_benchmark": "newton.examples.ik.example_ik_benchmark",
        "ik_franka": "newton.examples.ik.example_ik_franka",
        "ik_h1": "newton.examples.ik.example_ik_h1",
        "cloth_twist": "newton.examples.cloth.example_cloth_twist",
        "cloth_example": "newton.examples.cloth.example_cloth_example",
        "mpm_granular": "newton.examples.mpm.example_mpm_granular",
        "mpm_anymal": "newton.examples.mpm.example_mpm_anymal",
        "robot_anymal_c_walk": "newton.examples.robot.example_robot_anymal_c_walk",
        "robot_anymal_d": "newton.examples.robot.example_robot_anymal_d",
        "robot_cartpole": "newton.examples.robot.example_robot_cartpole",
        "robot_g1": "newton.examples.robot.example_robot_g1",
        "robot_h1": "newton.examples.robot.example_robot_h1",
        "robot_humanoid": "newton.examples.robot.example_robot_humanoid",
        "robot_policy": "newton.examples.robot.example_robot_policy",
        "robot_ur10": "newton.examples.robot.example_robot_ur10",
        "selection_articulations": "newton.examples.selection.example_selection_articulations",
        "selection_cartpole": "newton.examples.selection.example_selection_cartpole",
        "selection_materials": "newton.examples.selection.example_selection_materials",
        "diffsim_ball": "newton.examples.diffsim.example_diffsim_ball",
        "diffsim_cloth": "newton.examples.diffsim.example_diffsim_cloth",
        "diffsim_drone": "newton.examples.diffsim.example_diffsim_drone",
        "diffsim_spring_cage": "newton.examples.diffsim.example_diffsim_spring_cage",
        "diffsim_soft_body": "newton.examples.diffsim.example_diffsim_soft_body",
    }

    if len(sys.argv) < 2:
        print("Usage: python -m newton.examples <example_name>")
        print("\nAvailable examples:")
        for name in example_map.keys():
            print(f"  {name}")
        sys.exit(1)

    example_name = sys.argv[1]

    if example_name not in example_map:
        print(f"Error: Unknown example '{example_name}'")
        print("\nAvailable examples:")
        for name in example_map.keys():
            print(f"  {name}")
        sys.exit(1)

    # Set up sys.argv for the target script
    target_module = example_map[example_name]
    # Keep the module name as argv[0] and pass remaining args
    sys.argv = [target_module, *sys.argv[2:]]

    # Run the target example module
    runpy.run_module(target_module, run_name="__main__")


if __name__ == "__main__":
    main()


__all__ = ["create_parser", "init", "run"]
