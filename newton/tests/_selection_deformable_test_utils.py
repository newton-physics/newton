# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Shared native builders for deformable selection tests."""

import warp as wp

import newton

_CABLE_PTS = [(0.0, 0.0, 1.0), (0.1, 0.0, 1.0), (0.2, 0.0, 1.0), (0.3, 0.0, 1.0)]


def _replicated_model(world_count=3, device=None):
    """One cloth + one cable per world, replicated."""
    sub = newton.ModelBuilder()
    _add_test_cloth(sub, label="/World/Cloth")
    _add_test_cable(sub, label="/World/Cable")
    scene = newton.ModelBuilder()
    scene.replicate(sub, world_count)
    return scene.finalize() if device is None else scene.finalize(device=device)


def _irregular_model(device=None):
    """Three equal cloths and cables separated by unequal unrelated data."""
    builder = newton.ModelBuilder()
    _add_test_cloth(builder, label="selected_cloth_0")
    builder.add_particle(wp.vec3(0.0), wp.vec3(0.0), 1.0)
    _add_test_cloth(builder, label="selected_cloth_1")
    builder.add_particle(wp.vec3(0.0), wp.vec3(0.0), 1.0)
    builder.add_particle(wp.vec3(0.0), wp.vec3(0.0), 1.0)
    _add_test_cloth(builder, label="selected_cloth_2")

    _add_test_cable(builder, label="selected_cable_0")
    builder.add_link()
    _add_test_cable(builder, label="selected_cable_1")
    builder.add_link()
    builder.add_link()
    _add_test_cable(builder, label="selected_cable_2")
    return builder.finalize() if device is None else builder.finalize(device=device)


def _add_test_articulation(builder):
    root = builder.add_link(label="robot/root")
    root_joint = builder.add_joint_free(child=root, label="robot/root_joint")
    builder.add_articulation([root_joint], label="robot")


def _add_test_cable(builder, label="cable"):
    builder.add_rod(
        rod=newton.Rod(_CABLE_PTS, radius=0.02),
        label=label,
        body_frame_origin="com",
    )


def _add_test_cloth(builder, label="cloth", vertices=None, indices=None):
    if vertices is None:
        vertices = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0)]
    if indices is None:
        indices = [0, 1, 2, 0, 2, 3]
    builder.add_cloth_mesh(
        pos=wp.vec3(0.0, 0.0, 2.0),
        rot=wp.quat_identity(),
        scale=1.0,
        vel=wp.vec3(0.0),
        vertices=vertices,
        indices=indices,
        density=1.0,
        label=label,
    )


def _add_test_soft_body(builder, label="soft"):
    builder.add_soft_mesh(
        pos=wp.vec3(0.0, 0.0, 3.0),
        rot=wp.quat_identity(),
        scale=1.0,
        vel=wp.vec3(0.0),
        vertices=[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)],
        indices=[0, 1, 2, 3],
        density=1.0,
        k_mu=100.0,
        k_lambda=100.0,
        k_damp=0.0,
        label=label,
    )


def _add_test_anchored_cable(builder, label="anchored_curve"):
    """Add a two-segment cable fixed to the world for collapse tests."""
    bodies, joints = builder.add_rod(
        rod=newton.Rod([(0.0, 0.0, 1.0), (0.1, 0.0, 1.0), (0.2, 0.0, 1.0)], radius=0.02),
        label=label,
        wrap_in_articulation=False,
        body_frame_origin="com",
    )
    anchor = builder.add_joint_fixed(-1, bodies[0], label="anchor")
    builder.add_articulation([*joints, anchor])


def _add_test_soft_grid(builder, *, pos, label=None):
    """Add a one-cell volume with eight particles and five tetrahedra."""
    builder.add_soft_grid(
        pos=pos,
        rot=wp.quat_identity(),
        vel=wp.vec3(0.0),
        dim_x=1,
        dim_y=1,
        dim_z=1,
        cell_x=1.0,
        cell_y=1.0,
        cell_z=1.0,
        density=1.0,
        k_mu=1.0,
        k_lambda=1.0,
        k_damp=0.0,
        label=label,
    )
