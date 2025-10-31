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

"""
KAMINO: MODELS: MODEL BUILDER UTILITIES
"""

from ..core import ModelBuilder
from ..models.builders import (
    build_box_on_plane,
    build_box_pendulum,
    build_boxes_fourbar,
    build_boxes_hinged,
    build_boxes_nunchaku,
)

###
# Module interface
###

__all__ = [
    "make_heterogeneous_builder",
    "make_homogeneous_builder",
    "make_single_builder",
]


###
# Builder utilities
###


def make_single_builder(build_func=build_boxes_nunchaku, **kwargs) -> tuple[ModelBuilder, list[int], list[int]]:
    num_jcts = []
    num_bodies = []
    builder = ModelBuilder()
    build_func(builder, **kwargs)
    num_jcts.append(builder.world.num_joint_cts)
    num_bodies.append(builder.world.num_bodies)
    return builder, num_bodies, num_jcts


def make_homogeneous_builder(num_worlds: int, build_func=build_boxes_nunchaku) -> ModelBuilder:
    builder = ModelBuilder()
    build_func(builder)
    for _ in range(num_worlds - 1):
        other = ModelBuilder()
        build_func(other)
        builder.add_builder(other)
    return builder


def make_heterogeneous_builder() -> tuple[ModelBuilder, list[int], list[int]]:
    num_jcts = []
    num_bodies = []
    builder = ModelBuilder()
    build_boxes_fourbar(builder)
    num_jcts.append(builder.world.num_joint_cts)
    num_bodies.append(builder.world.num_bodies)

    other_boxes_nunchaku = ModelBuilder()
    build_boxes_nunchaku(other_boxes_nunchaku)
    num_jcts.append(other_boxes_nunchaku.world.num_joint_cts)
    num_bodies.append(other_boxes_nunchaku.world.num_bodies)
    builder.add_builder(other_boxes_nunchaku)

    other_boxes_hinged = ModelBuilder()
    build_boxes_hinged(other_boxes_hinged)
    num_jcts.append(other_boxes_hinged.world.num_joint_cts)
    num_bodies.append(other_boxes_hinged.world.num_bodies)
    builder.add_builder(other_boxes_hinged)

    other_box_pendulum = ModelBuilder()
    build_box_pendulum(other_box_pendulum)
    num_jcts.append(other_box_pendulum.world.num_joint_cts)
    num_bodies.append(other_box_pendulum.world.num_bodies)
    builder.add_builder(other_box_pendulum)

    builder_box_on_floor = ModelBuilder()
    build_box_on_plane(builder_box_on_floor)
    num_jcts.append(builder_box_on_floor.world.num_joint_cts)
    num_bodies.append(builder_box_on_floor.world.num_bodies)
    builder.add_builder(builder_box_on_floor)

    return builder, num_bodies, num_jcts
