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

"""CLI utilities for the GPU Cosserat example."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from newton.examples.cosserat_codex import warp_cosserat_codex as base

from .model import RodConfig


@dataclass
class SolverConfig:
    substeps: int
    linear_damping: float
    angular_damping: float
    use_banded: bool
    use_cuda_graph: bool


def create_parser():
    import argparse  # noqa: PLC0415

    parser = base.create_parser()
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
        default=0.5,
        help="Spacing between GPU rods along the X axis.",
    )
    return parser


def build_rod_configs(args) -> List[RodConfig]:
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
    return SolverConfig(
        substeps=args.substeps,
        linear_damping=args.linear_damping,
        angular_damping=args.angular_damping,
        use_banded=args.use_banded,
        use_cuda_graph=args.use_cuda_graph,
    )


__all__ = ["SolverConfig", "build_rod_configs", "build_solver_config", "create_parser"]
