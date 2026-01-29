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

"""CLI utilities for the Cosserat rod simulation."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

import newton.examples

from .rod import RodConfig


class SolverType(Enum):
    """Solver types for rod simulation."""

    NUMPY = "numpy"
    WARP = "warp"
    DLL = "dll"


@dataclass
class SolverConfig:
    """Configuration for the XPBD solver."""

    substeps: int
    linear_damping: float
    angular_damping: float
    use_banded: bool
    use_cuda_graph: bool


def create_base_parser():
    """Create the base argument parser with rod simulation options."""
    import argparse  # noqa: PLC0415

    parser = newton.examples.create_parser()
    parser.add_argument(
        "--dll-path",
        type=str,
        default=None,
        help="Path to DefKitAdv.dll. If omitted, attempts to load from PATH.",
    )
    parser.add_argument(
        "--calling-convention",
        type=str,
        choices=["cdecl", "stdcall"],
        default="cdecl",
        help="Calling convention used by the DLL (cdecl or stdcall).",
    )
    parser.add_argument("--num-points", type=int, default=64, help="Number of rod points.")
    parser.add_argument("--segment-length", type=float, default=0.025, help="Rest length per segment.")
    parser.add_argument("--particle-mass", type=float, default=1.0, help="Mass per particle (root fixed).")
    parser.add_argument("--particle-radius", type=float, default=0.02, help="Particle visualization radius.")
    parser.add_argument("--particle-height", type=float, default=1.0, help="Initial rod height (z).")
    parser.add_argument(
        "--rod-radius",
        type=float,
        default=None,
        help="Physical rod radius for direct solver (defaults to particle-radius).",
    )
    parser.add_argument(
        "--compare-offset",
        type=float,
        default=0.0,
        help="Y-offset separating reference and NumPy rods.",
    )
    parser.add_argument("--substeps", type=int, default=4, help="Integration substeps per frame.")
    parser.add_argument("--bend-stiffness", type=float, default=1.0, help="Per-edge bend stiffness.")
    parser.add_argument("--twist-stiffness", type=float, default=1.0, help="Per-edge twist stiffness.")
    parser.add_argument("--rest-bend-d1", type=float, default=0.0, help="Rest bend around d1 axis (rad/segment).")
    parser.add_argument("--rest-bend-d2", type=float, default=0.0, help="Rest bend around d2 axis (rad/segment).")
    parser.add_argument("--rest-twist", type=float, default=0.0, help="Rest twist around d3 axis (rad/segment).")
    parser.add_argument("--young-modulus", type=float, default=1.0e6, help="Young's modulus multiplier.")
    parser.add_argument("--torsion-modulus", type=float, default=1.0e6, help="Torsion modulus multiplier.")
    parser.add_argument(
        "--use-banded",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use banded direct solver (disable to use non-banded if available).",
    )
    parser.add_argument("--linear-damping", type=float, default=0.001, help="Linear damping coefficient.")
    parser.add_argument("--angular-damping", type=float, default=0.001, help="Angular damping coefficient.")
    parser.add_argument(
        "--gravity",
        type=float,
        nargs=3,
        default=[0.0, 0.0, -9.81],
        help="Gravity vector (x y z).",
    )
    parser.add_argument(
        "--lock-root-rotation",
        action="store_true",
        default=False,
        help="Lock root rotation by zeroing quaternion inverse mass.",
    )
    return parser


def create_parser():
    """Create argument parser with GPU-specific options."""
    import argparse  # noqa: PLC0415

    parser = create_base_parser()
    parser.add_argument(
        "--use-cuda-graph",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable CUDA graph capture for the GPU rods.",
    )
    parser.add_argument(
        "--rod-count",
        type=int,
        default=1,
        help="Number of GPU rods to simulate in parallel.",
    )
    parser.add_argument(
        "--rod-spacing",
        type=float,
        default=0.0,
        help="Spacing between GPU rods along the X axis. Default 0 for concentric rods.",
    )
    parser.add_argument(
        "--rod-solvers",
        type=str,
        default="numpy,warp",
        help=(
            "Comma-separated list of solver types for each rod. "
            "Available types: numpy, warp, dll. "
            "Example: --rod-solvers numpy,warp,dll creates 3 rods with different solvers. "
            "If fewer solvers than --rod-count, the last solver type is repeated."
        ),
    )
    return parser


def parse_solver_types(solver_str: str, rod_count: int) -> list[SolverType]:
    """Parse solver types string into a list of SolverType enums.

    Args:
        solver_str: Comma-separated string of solver types (e.g., "numpy,warp,dll").
        rod_count: Total number of rods to create.

    Returns:
        List of SolverType enums, one per rod.
    """
    solver_names = [s.strip().lower() for s in solver_str.split(",")]
    solver_types = []

    for name in solver_names:
        if name == "numpy":
            solver_types.append(SolverType.NUMPY)
        elif name == "warp":
            solver_types.append(SolverType.WARP)
        elif name == "dll":
            solver_types.append(SolverType.DLL)
        else:
            raise ValueError(f"Unknown solver type: {name}. Must be one of: numpy, warp, dll")

    # Extend to match rod_count if needed
    while len(solver_types) < rod_count:
        solver_types.append(solver_types[-1] if solver_types else SolverType.WARP)

    # Truncate if more solvers than rods
    return solver_types[:rod_count]


def build_rod_configs(args) -> list[RodConfig]:
    """Build rod configurations from parsed arguments."""
    rod_radius = args.rod_radius if args.rod_radius is not None else args.particle_radius
    gravity = np.array(args.gravity, dtype=np.float32)
    rod_count = max(int(args.rod_count), 1)
    configs = []
    for _ in range(rod_count):
        configs.append(
            RodConfig(
                num_points=args.num_points,
                segment_length=args.segment_length,
                particle_mass=args.particle_mass,
                particle_height=args.particle_height,
                rod_radius=rod_radius,
                bend_stiffness=args.bend_stiffness,
                twist_stiffness=args.twist_stiffness,
                rest_bend_d1=args.rest_bend_d1,
                rest_bend_d2=args.rest_bend_d2,
                rest_twist=args.rest_twist,
                young_modulus=args.young_modulus,
                torsion_modulus=args.torsion_modulus,
                gravity=gravity,
                lock_root_rotation=args.lock_root_rotation,
            )
        )
    return configs


def build_solver_config(args) -> SolverConfig:
    """Build solver configuration from parsed arguments."""
    return SolverConfig(
        substeps=args.substeps,
        linear_damping=args.linear_damping,
        angular_damping=args.angular_damping,
        use_banded=args.use_banded,
        use_cuda_graph=args.use_cuda_graph,
    )


__all__ = [
    "SolverConfig",
    "SolverType",
    "build_rod_configs",
    "build_solver_config",
    "create_base_parser",
    "create_parser",
    "parse_solver_types",
]
