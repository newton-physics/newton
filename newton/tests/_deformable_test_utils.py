# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Shared native deformable builders for recording, selection, and coupled tests."""

import warp as wp

import newton


def _add_curve(builder, label=None, topology="chain"):
    points = [(0.0, 0.0, 1.0), (0.1, 0.0, 1.0), (0.2, 0.0, 1.0), (0.1, 0.1, 1.0)]
    if topology == "closed":
        points = [points[0], points[1], points[3], points[0]]
    edges = [(0, 1), (1, 2), (1, 3)] if topology == "graph" else None
    return builder.add_rod(
        rod=newton.Rod(points, edges=edges, closed=topology == "closed", radius=0.02),
        label=label,
        body_frame_origin="com",
    )


def _add_surface(builder, label=None, grid=False):
    common = {"pos": wp.vec3(0.0, 0.0, 2.0), "rot": wp.quat_identity(), "vel": wp.vec3(0.0), "label": label}
    if grid:
        builder.add_cloth_grid(**common, dim_x=1, dim_y=1, cell_x=1.0, cell_y=1.0, mass=1.0)
    else:
        builder.add_cloth_mesh(
            **common,
            scale=1.0,
            vertices=[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0)],
            indices=[0, 1, 2, 0, 2, 3],
            density=1.0,
        )


def _add_volume(builder, label=None, grid=False):
    common = {
        "pos": wp.vec3(0.0, 0.0, 3.0),
        "rot": wp.quat_identity(),
        "vel": wp.vec3(0.0),
        "density": 1.0,
        "k_mu": 1.0,
        "k_lambda": 1.0,
        "k_damp": 0.0,
        "label": label,
    }
    if grid:
        builder.add_soft_grid(**common, dim_x=1, dim_y=1, dim_z=1, cell_x=1.0, cell_y=1.0, cell_z=1.0)
    else:
        builder.add_soft_mesh(
            **common,
            scale=1.0,
            vertices=[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)],
            indices=[0, 1, 2, 3],
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
