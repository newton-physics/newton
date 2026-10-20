# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Steel Determinism Pile
#
# Drops two humanoids next to a stack of eight boxes and simulates the
# scene four times in one process from the same initial state: two copies
# with Steel's default settings (left) and two with
# deterministic=wp.DeterministicMode.RUN_TO_RUN and
# CollisionPipeline(deterministic=True) (right). Each side overlays its two
# copies at one place, one magenta and one green, both half transparent.
# The default copies split within a few steps; in the GL viewer they
# come apart into two ghosts, while the deterministic copies stay bitwise
# identical and blend into one gray scene.
#
# Command: python -m newton.examples steel_determinism_pile
#
###########################################################################

import numpy as np
import warp as wp

import newton
import newton.examples
from newton.examples.steel._determinism import DeterminismExample

BOX_COUNT = 8
BOX_HALF_EXTENTS = (0.06, 0.05, 0.06)


def build_scene() -> newton.ModelBuilder:
    """Build two floating humanoids and a stack of boxes."""
    humanoid = newton.ModelBuilder()
    humanoid.rigid_gap = 0.01
    humanoid.add_mjcf(newton.examples.get_asset("nv_humanoid.xml"), floating=True, ignore_names=["floor"])
    # Steel ignores passive joint springs (newton-physics/newton#4516) and warns when they are set.
    humanoid.custom_attributes["mujoco:dof_passive_stiffness"].values.clear()

    scene = newton.ModelBuilder()
    scene.rigid_gap = humanoid.rigid_gap
    # One humanoid lies on the ground and the other drops from 1.4 m beside it.
    scene.add_builder(
        humanoid,
        xform=wp.transform((0.0, 0.0, 0.35), wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), 1.5)),
    )
    scene.add_builder(humanoid, xform=wp.transform((1.2, 0.0, 1.4), wp.quat_identity()))

    rng = np.random.default_rng(0)
    box_cfg = newton.ModelBuilder.ShapeConfig(density=300.0, mu=0.7)
    for box_id in range(BOX_COUNT):
        position = (-1.2 + 0.05 * rng.standard_normal(), 0.05 * rng.standard_normal(), 0.06 + 0.13 * box_id)
        yaw = float(rng.uniform(0.0, np.pi))
        body = scene.add_body(xform=wp.transform(position, wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), yaw)))
        scene.add_shape_box(body, hx=BOX_HALF_EXTENTS[0], hy=BOX_HALF_EXTENTS[1], hz=BOX_HALF_EXTENTS[2], cfg=box_cfg)
    return scene


class Example(DeterminismExample):
    def __init__(self, viewer, args):
        """Build the pile and its default and deterministic copies."""
        super().__init__(
            viewer,
            build_scene(),
            solver_options={
                "enable_joint_limits": True,
                "articulation_constraint_max": 384,
                "free_body_constraint_max": 128,
            },
            side_offset=(0.0, -1.3, 0.0),
            camera={"pos": wp.vec3(-4.6, 0.0, 2.4), "pitch": -24.0, "yaw": 0.0},
        )


if __name__ == "__main__":
    viewer, args = newton.examples.init()
    newton.examples.run(Example(viewer, args), args)
