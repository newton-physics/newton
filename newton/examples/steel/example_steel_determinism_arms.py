# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Steel Determinism Arms
#
# Two fixed-base Franka arms follow the same scripted sweep through six
# boxes. The scene is simulated four times in one process from the same
# initial state and with the same joint targets: two copies with Steel's
# default settings (left) and two with
# deterministic=wp.DeterministicMode.RUN_TO_RUN and
# CollisionPipeline(deterministic=True) (right). Each side overlays its two
# copies at one place, one magenta and one green, both half transparent.
# The default copies push the boxes to different places; in the GL
# viewer they come apart into two ghosts, while the deterministic copies
# stay bitwise identical and blend into one gray scene. The plots show,
# for each side, the largest distance in mm between a body in the magenta
# copy and the same body in the green copy.
#
# Command: python -m newton.examples steel_determinism_arms
#
###########################################################################

import math

import numpy as np
import warp as wp

import newton
import newton.examples
from newton.examples.steel._determinism import DeterminismExample

ARM_DOF_COUNT = wp.constant(9)
ARM_REACH_POSE = (0.0, 0.4, 0.0, -2.2, 0.0, 2.6, 0.785, 0.04, 0.04)
# Base yaw [rad] and period [s], then shoulder pitch amplitude, period and offset of the sweep.
YAW_AMPLITUDE = wp.constant(0.55)
YAW_PERIOD = wp.constant(3.0)
SHOULDER_AMPLITUDE = wp.constant(0.25)
SHOULDER_PERIOD = wp.constant(1.5)
SHOULDER_OFFSET = wp.constant(-0.05)
BOX_COUNT = 6
BOX_HALF_EXTENTS = (0.06, 0.05, 0.06)


@wp.kernel(enable_backward=False)
def sweep_arm_targets(
    step_index: wp.array[wp.int32],
    dt: float,
    reach_q: wp.array[wp.float32],
    joint_target_q: wp.array[wp.float32],
):
    """Sweep each arm's base yaw and shoulder joints, the second arm half a period behind the first."""
    arm = wp.tid()
    t = float(step_index[0]) * dt
    phase = float(arm) * wp.pi
    offset = arm * ARM_DOF_COUNT
    joint_target_q[offset] = reach_q[0] + YAW_AMPLITUDE * wp.sin(2.0 * wp.pi * t / YAW_PERIOD + phase)
    joint_target_q[offset + 1] = (
        reach_q[1] + SHOULDER_AMPLITUDE * wp.sin(2.0 * wp.pi * t / SHOULDER_PERIOD + phase) + SHOULDER_OFFSET
    )


def build_scene() -> newton.ModelBuilder:
    """Build two facing Franka arms and a 3x2 grid of boxes between them."""
    scene = newton.ModelBuilder()
    scene.rigid_gap = 0.01
    urdf = newton.utils.download_asset("franka_emika_panda") / "urdf/fr3_franka_hand.urdf"
    for x, y, yaw in ((-0.25, -0.55, 0.5 * math.pi), (0.25, 0.55, -0.5 * math.pi)):
        scene.add_urdf(
            urdf,
            xform=wp.transform((x, y, 0.0), wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), yaw)),
            floating=False,
            enable_self_collisions=False,
            joint_damping_mode="passive",
        )
        dofs = slice(len(scene.joint_q) - ARM_DOF_COUNT, len(scene.joint_q))
        scene.joint_q[dofs] = ARM_REACH_POSE
        scene.joint_target_q[dofs] = ARM_REACH_POSE
        scene.joint_target_ke[dofs] = [4000.0] * 7 + [100.0] * 2
        scene.joint_target_kd[dofs] = [400.0] * 7 + [10.0] * 2
        scene.joint_target_mode[dofs] = [int(newton.JointTargetMode.POSITION)] * ARM_DOF_COUNT

    rng = np.random.default_rng(0)
    box_cfg = newton.ModelBuilder.ShapeConfig(density=300.0, mu=0.7)
    for box_id in range(BOX_COUNT):
        position = (-0.25 + 0.25 * (box_id // 2) + 0.01 * rng.standard_normal(), -0.07 + 0.14 * (box_id % 2), 0.065)
        yaw = float(rng.uniform(0.0, np.pi))
        body = scene.add_body(xform=wp.transform(position, wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), yaw)))
        scene.add_shape_box(body, hx=BOX_HALF_EXTENTS[0], hy=BOX_HALF_EXTENTS[1], hz=BOX_HALF_EXTENTS[2], cfg=box_cfg)
    return scene


class Example(DeterminismExample):
    def __init__(self, viewer, args):
        """Build the arms, their sweep and the default and deterministic copies."""
        # Check the device before build_scene() downloads and parses the Franka asset.
        if not wp.get_device().is_cuda:
            raise RuntimeError("The Steel determinism example requires a CUDA device.")
        self.reach_q = wp.array(ARM_REACH_POSE, dtype=wp.float32)
        super().__init__(
            viewer,
            build_scene(),
            solver_options={
                "enable_joint_limits": True,
                "articulation_constraint_max": 384,
                "free_body_constraint_max": 128,
            },
            side_offset=(0.0, 1.0, 0.0),
            camera={"pos": wp.vec3(2.7, 0.0, 1.3), "pitch": -20.0, "yaw": 180.0},
        )

    def apply_control(self, step_index):
        """Set the sweep targets of both arms for the next substep."""
        wp.launch(
            sweep_arm_targets,
            dim=2,
            inputs=[step_index, self.sim_dt, self.reach_q],
            outputs=[self.control.joint_target_q],
        )


if __name__ == "__main__":
    viewer, args = newton.examples.init()
    newton.examples.run(Example(viewer, args), args)
