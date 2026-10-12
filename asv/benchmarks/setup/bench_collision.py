# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import warp as wp

import newton


class CollisionPipelineSetup:
    """Measure collision setup for replicated rigid worlds."""

    params = ([1, 1024, 4096], ["sap", "nxn", "explicit"])
    param_names = ["world_count", "broad_phase"]
    number = 1
    repeat = 3

    def setup(self, world_count, broad_phase):
        world = newton.ModelBuilder()
        for i in range(16):
            body = world.add_body(xform=wp.transform((i * 0.4, 0.0, 0.5), wp.quat_identity()))
            if i % 2:
                world.add_shape_cylinder(body, radius=0.15, half_height=0.3)
            else:
                world.add_shape_box(body, hx=0.15, hy=0.15, hz=0.3)
        builder = newton.ModelBuilder()
        builder.replicate(world, world_count)
        builder.add_ground_plane()
        self.model = builder.finalize()
        self.pipeline = newton.CollisionPipeline(self.model, broad_phase=broad_phase)
        wp.synchronize_device(self.model.device)

    def time_construct(self, world_count, broad_phase):
        self.pipeline = newton.CollisionPipeline(self.model, broad_phase=broad_phase)
        wp.synchronize_device(self.model.device)
