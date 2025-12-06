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

"""Utilities for building test models"""

from ..core import ModelBuilder
from ..models.builders import (
    build_box_on_plane,
    build_box_pendulum,
    build_boxes_fourbar,
    build_boxes_hinged,
    build_boxes_nunchaku,
    build_cartpole,
)

###
# Module interface
###

__all__ = [
    "make_heterogeneous_builder",
    "make_homogeneous_builder",
]


###
# Builder utilities
###


def make_homogeneous_builder(num_worlds: int, build_fn=build_boxes_nunchaku, **kwargs) -> ModelBuilder:
    """
    Utility factory function to create a multi-world builder with identical worlds replicated across the model.

    Args:
        num_worlds (int): The number of worlds to create.
        build_fn (callable): The model builder function to use.
        **kwargs: Additional keyword arguments to pass to the builder function.

    Returns:
        ModelBuilder: The constructed model builder.
    """
    # First build a single world
    # NOTE: We want to do this first to avoid re-constructing the same model multiple
    # times especially if the construction is expensive such as importing from USD.
    single = build_fn(**kwargs)

    # Then replicate it across the specified number of worlds
    builder = ModelBuilder(default_world=False)
    for _ in range(num_worlds):
        builder.add_builder(single)
    return builder


def make_heterogeneous_builder(ground: bool = True) -> ModelBuilder:
    """
    Utility factory function to create a multi-world builder with different worlds in each model.

    This function constructs a model builder containing all test models defined in Kamino.

    Returns:
        ModelBuilder: The constructed model builder.
    """
    builder = ModelBuilder(default_world=False)
    builder.add_builder(build_boxes_fourbar(ground=ground))
    builder.add_builder(build_boxes_nunchaku(ground=ground))
    builder.add_builder(build_boxes_hinged(ground=ground))
    builder.add_builder(build_box_pendulum(ground=ground))
    builder.add_builder(build_box_on_plane(ground=ground))
    builder.add_builder(build_cartpole(z_offset=0.5, ground=ground))
    return builder
