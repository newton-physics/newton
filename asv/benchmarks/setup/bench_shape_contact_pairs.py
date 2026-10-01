# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Track model setup with dense, sparse and replicated collision topology."""

import warp as wp

import newton


def _make_builder(scene, shape_count):
    builder = newton.ModelBuilder()
    if scene == "replicated":
        world = newton.ModelBuilder()
        for _ in range(16):
            world.add_shape_sphere(world.add_body())
        builder.add_ground_plane()
        builder.replicate(world, max(1, shape_count // 16))
        return builder

    if scene == "ragged":
        remaining = shape_count
        world = 0
        while remaining:
            size = min(remaining, 8 + world % 17)
            builder.begin_world()
            for i in range(size):
                builder.add_shape_sphere(builder.add_body())
                if i == world % size:
                    builder.add_shape_sphere(builder.shape_body[-1])
            builder.end_world()
            remaining -= size
            world += 1
        builder.add_ground_plane()
        return builder

    shared_body = builder.add_body() if scene == "shared_body" else None
    grouped_bodies = [builder.add_body(), builder.add_body()] if scene == "body_groups" else None
    for i in range(shape_count):
        group = 1
        if scene == "sparse_groups":
            group = 1 + i // 2
        elif scene == "negative_group":
            group = -1 if i < shape_count - 1 else -2
        elif scene == "body_groups":
            group = 1 + i // ((shape_count + 1) // 2)
        if scene == "body_groups" and i < shape_count - 1:
            body = grouped_bodies[group - 1]
        elif scene == "shared_body" and i < shape_count - 1:
            body = shared_body
        else:
            body = builder.add_body()
        cfg = newton.ModelBuilder.ShapeConfig(collision_group=group)
        builder.add_shape_sphere(body, cfg=cfg)
        if scene == "filtered" and i:
            builder.add_shape_collision_filter_pair(i - 1, i)
    return builder


class ShapeContactPairSetup:
    params = (
        ["dense", "filtered", "sparse_groups", "shared_body", "negative_group", "body_groups", "replicated", "ragged"],
        [100, 1000, 5000],
        ["cpu", "cuda:0"],
    )
    param_names = ["scene", "shape_count", "device"]
    rounds = 1
    repeat = 3
    number = 1
    min_run_count = 1
    timeout = 120

    def setup(self, scene, shape_count, device):
        if device != "cpu" and not wp.is_cuda_available():
            raise NotImplementedError("CUDA is unavailable")
        wp.init()
        warmup = _make_builder("dense", 2).finalize(device=device)
        _ = warmup.shape_contact_pairs
        wp.synchronize_device(device)
        self.builder = _make_builder(scene, shape_count)

    def time_finalize(self, scene, shape_count, device):
        _model = self.builder.finalize(device=device)
        wp.synchronize_device(device)

    def time_finalize_explicit(self, scene, shape_count, device):
        model = self.builder.finalize(device=device)
        _ = model.shape_contact_pairs
        wp.synchronize_device(device)

    def peakmem_finalize(self, scene, shape_count, device):
        self.time_finalize(scene, shape_count, device)

    def peakmem_finalize_explicit(self, scene, shape_count, device):
        self.time_finalize_explicit(scene, shape_count, device)
