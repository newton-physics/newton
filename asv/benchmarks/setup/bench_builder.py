# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Benchmark builder composition with built-in and custom references."""

import warp as wp

import newton
from newton.solvers import SolverMuJoCo, mujoco


class BuilderReplicate:
    params = (["plain", "custom", "mujoco", "mujoco_rows"], [128, 1024])
    param_names = ["attributes", "world_count"]

    def setup(self, attributes, world_count):
        self.builder = builder = newton.ModelBuilder()
        self.prefixes = [f"world_{i}" for i in range(world_count)]
        if attributes.startswith("mujoco"):
            SolverMuJoCo.register_custom_attributes(builder)
        elif attributes == "custom":
            builder.add_custom_attribute(
                newton.ModelBuilder.CustomAttribute(
                    name="peer",
                    dtype=wp.int32,
                    frequency=newton.Model.AttributeFrequency.BODY,
                    references="body",
                    default=-1,
                )
            )

        for i in range(32):
            body = builder.add_body(custom_attributes={"peer": i} if attributes == "custom" else None)
            builder.add_shape_sphere(
                body,
                radius=0.1,
                custom_attributes={"mujoco:collision_mask_domain": i // 8} if attributes.startswith("mujoco") else None,
            )
            if attributes == "mujoco_rows":
                mujoco.add_equality_connect(builder, body, label=f"constraint_{i}")
        if attributes == "mujoco_rows":
            builder.custom_attributes["mujoco:equality_constraint_target_kind"].values = [1] * 32
            builder.custom_attributes["mujoco:equality_constraint_target"].values = list(range(32))

    def time_replicate(self, attributes, world_count):
        destination = newton.ModelBuilder()
        destination.replicate(self.builder, world_count, label_prefixes=self.prefixes)
