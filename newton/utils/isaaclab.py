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

from typing import Any

import warp as wp

import newton
from newton.examples import compute_env_offsets


def replicate_environment(
    source,
    prototype_path: str,
    path_pattern: str,
    num_envs: int,
    env_spacing: tuple[float],
    up_axis: newton.AxisType = "Z",
    **usd_kwargs,
) -> tuple[newton.ModelBuilder, dict[str:Any]]:
    """
    Replicates a prototype USD environment in Newton.

    Args:
        source (str | pxr.UsdStage): The file path to the USD file, or an existing USD stage instance.
        prototype_path (str): The USD path where the prototype env is defined, e.g., "/World/envs/env_0".
        path_pattern (str): The USD path pattern for replicated envs, e.g., "/World/envs/env_{}".
        num_envs (int): Number of replicas to create.
        env_spacing (tuple[float]): Environment spacing vector.
        up_axis (AxisType): The desired up-vector (should match the USD stage).
        **usd_kwargs: Keyword arguments to pass to the USD importer (see `newton.utils.parse_usd()`).

    Returns:
        (ModelBuilder, dict): The resulting ModelBuilder containing all replicated environments and a dictionary with USD stage information.
    """

    builder = newton.ModelBuilder(up_axis=up_axis)

    # first, load everything except the prototype env
    stage_info = newton.utils.parse_usd(
        source,
        builder,
        ignore_paths=[prototype_path],
        **usd_kwargs,
    )

    # up_axis sanity check
    stage_up_axis = stage_info.get("up_axis")
    if isinstance(stage_up_axis, str) and stage_up_axis.upper() != up_axis.upper():
        print(f"WARNING: up_axis '{up_axis}' does not match USD stage up_axis '{stage_up_axis}'")

    # load just the prototype env
    prototype_builder = newton.ModelBuilder(up_axis=up_axis)
    newton.utils.parse_usd(
        source,
        prototype_builder,
        root_path=prototype_path,
        **usd_kwargs,
    )

    env_offsets = compute_env_offsets(num_envs, env_offset=env_spacing, up_axis=up_axis)

    # clone the prototype env with updated paths
    for i in range(num_envs):
        body_start = builder.body_count
        shape_start = builder.shape_count
        joint_start = builder.joint_count
        articulation_start = builder.articulation_count

        builder.add_builder(prototype_builder, xform=wp.transform(env_offsets[i], wp.quat_identity()))

        if i > 0:
            update_paths(
                builder,
                prototype_path,
                path_pattern.format(i),
                body_start=body_start,
                shape_start=shape_start,
                joint_start=joint_start,
                articulation_start=articulation_start,
            )

    return builder, stage_info


def update_paths(
    builder, old_root, new_root, body_start=None, shape_start=None, joint_start=None, articulation_start=None
):
    old_len = len(old_root)
    if body_start is not None:
        for i in range(body_start, builder.body_count):
            builder.body_key[i] = f"{new_root}{builder.body_key[i][old_len:]}"
    if shape_start is not None:
        for i in range(shape_start, builder.shape_count):
            builder.shape_key[i] = f"{new_root}{builder.shape_key[i][old_len:]}"
    if joint_start is not None:
        for i in range(joint_start, builder.joint_count):
            builder.joint_key[i] = f"{new_root}{builder.joint_key[i][old_len:]}"
    if articulation_start is not None:
        for i in range(articulation_start, builder.articulation_count):
            builder.articulation_key[i] = f"{new_root}{builder.articulation_key[i][old_len:]}"
