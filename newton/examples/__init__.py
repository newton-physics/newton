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


def run(viewer, example):
    while viewer.is_running():
        with wp.ScopedTimer("step"):
            example.step()

        with wp.ScopedTimer("render"):
            example.render()

    viewer.close()


def compute_env_offsets(
    num_envs: int, env_offset: tuple[float, float, float] = (5.0, 5.0, 0.0), up_axis: newton.AxisType = newton.Axis.Z
):
    # raise deprecation warning
    import warnings  # noqa: PLC0415

    warnings.warn(
        "compute_env_offsets is deprecated and will be removed in a future version. Use the builder.replicate() function instead.",
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


from .basic import example_basic_pendulum as basic_pendulum
from .basic import example_basic_urdf as basic_urdf
from .basic import example_basic_viewer as basic_viewer

__all__ = ["basic_pendulum", "basic_urdf", "basic_viewer"]
