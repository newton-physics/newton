# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Robot Allegro Hand
#
# Shows how to set up a simulation of a Allegro hand articulation
# from a USD file using newton.ModelBuilder.add_usd().
# We also apply a sinusoidal trajectory to the joint targets and
# apply a continuous rotation to the fixed root joint in the form
# of the joint parent transform. The MuJoCo solver is updated
# about this change in the joint parent transform by calling
# self.solver.notify_model_changed(SolverNotifyFlags.JOINT_PROPERTIES).
#
# Command: python -m newton.examples robot_allegro_hand --world-count 16
#
###########################################################################


import numpy as np
import warp as wp

import newton
import newton.examples
from newton import JointTargetMode
from newton.solvers import SolverNotifyFlags


@wp.kernel
def move_hand(
    joint_qd_start: wp.array(dtype=wp.int32),
    joint_limit_lower: wp.array(dtype=wp.float32),
    joint_limit_upper: wp.array(dtype=wp.float32),
    sim_time: wp.array(dtype=wp.float32),
    sim_dt: float,
    hand_rotation: wp.quat,
    # outputs
    joint_target_pos: wp.array(dtype=wp.float32),
    joint_parent_xform: wp.array(dtype=wp.transform),
):
    world_id = wp.tid()
    root_joint_id = world_id * 22
    t = sim_time[world_id]

    root_dof_start = joint_qd_start[root_joint_id]

    # animate the finger joints
    for i in range(20):
        di = root_dof_start + i
        target = wp.sin(t + float(i * 6) * 0.1) * 0.1 + 0.3
        joint_target_pos[di] = wp.clamp(target, joint_limit_lower[di], joint_limit_upper[di])

    # animate the root joint transform
    q = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), wp.sin(t) * 0.1)
    root_xform = joint_parent_xform[root_joint_id]
    joint_parent_xform[root_joint_id] = wp.transform(root_xform.p, q * hand_rotation)

    # update the sim time
    sim_time[world_id] += sim_dt


class Example:
    def __init__(self, viewer, args):
        self.fps = 50
        self.frame_dt = 1.0 / self.fps

        self.sim_time = 0.0
        self.sim_substeps = 8
        self.sim_dt = self.frame_dt / self.sim_substeps

        self.world_count = args.world_count

        self.viewer = viewer

        self.device = wp.get_device()

        self.hand_rotation = wp.normalize(wp.quat(0.21643, 0.706218, -0.648166, 0.185191))
        max_contacts_per_world = 300

        allegro_hand = newton.ModelBuilder()
        newton.solvers.SolverMuJoCo.register_custom_attributes(allegro_hand)
        allegro_hand.default_shape_cfg.ke = 1.0e3
        allegro_hand.default_shape_cfg.kd = 1.0e2
        allegro_hand.default_shape_cfg.margin = 0.005
        allegro_hand.default_shape_cfg.gap = 0.015

        asset_path = newton.utils.download_asset("wonik_allegro")
        asset_file = str(asset_path / "usd" / "allegro_left_hand_with_cube.usda")
        allegro_hand.add_usd(
            asset_file,
            xform=wp.transform(wp.vec3(0, 0, 0.5)),
            enable_self_collisions=False,
            ignore_paths=[".*Dummy", ".*CollisionPlane"],
            hide_collision_shapes=True,
        )

        # set joint targets and joint drive gains (only on hand, not the floating-body cube)
        for i in range(allegro_hand.joint_dof_count - 6):
            allegro_hand.joint_target_ke[i] = 150
            allegro_hand.joint_target_kd[i] = 5
            allegro_hand.joint_q[i] = 0.3
            allegro_hand.joint_target_pos[i] = 0.3
            if allegro_hand.joint_label[i][-2:] == "_0":
                allegro_hand.joint_q[i] = 0.6
                allegro_hand.joint_target_pos[i] = 0.6
            allegro_hand.joint_target_mode[i] = int(JointTargetMode.POSITION)
            if allegro_hand.joint_type[i] == newton.JointType.REVOLUTE:
                allegro_hand.joint_armature[i] = 1e-2

        # Update root pose of the cube (free joint)
        q = np.array(allegro_hand.joint_q)
        q[-7:-4] += np.array([0.0, 0.0, 0.05])
        q[-4:] = wp.quat_rpy(0.3, 0.5, 0.1)
        allegro_hand.joint_q = q.tolist()

        builder = newton.ModelBuilder()
        builder.replicate(allegro_hand, self.world_count)

        builder.default_shape_cfg.ke = 1.0e3
        builder.default_shape_cfg.kd = 1.0e2
        builder.add_ground_plane()

        self.model = builder.finalize()

        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.model)
        self.initial_world_positions = self.model.body_q.numpy()[:: allegro_hand.body_count, :3].copy()

        # Find the cube body index (it's the last body in each world)
        self.cube_body_offset = allegro_hand.body_count - 1

        self.world_time = wp.zeros(self.world_count, dtype=wp.float32)

        self.solver = newton.solvers.SolverMuJoCo(
            self.model,
            solver="newton",
            integrator="implicitfast",
            njmax=200,
            nconmax=max_contacts_per_world,
            impratio=10.0,
            cone="elliptic",
            iterations=100,
            ls_iterations=50,
            use_mujoco_contacts=False,
        )

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.model.contacts()

        self.viewer.set_model(self.model)

        self.capture()

    def capture(self):
        self.graph = None
        if wp.get_device().is_cuda:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph

    def simulate(self):
        self.model.collide(self.state_0, self.contacts)
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()

            # apply forces to the model for picking, wind, etc
            self.viewer.apply_forces(self.state_0)

            wp.launch(
                move_hand,
                dim=self.world_count,
                inputs=[
                    self.model.joint_qd_start,
                    self.model.joint_limit_lower,
                    self.model.joint_limit_upper,
                    self.world_time,
                    self.sim_dt,
                    self.hand_rotation,
                ],
                outputs=[self.control.joint_target_pos, self.model.joint_X_p],
            )

            # # update the solver since we have updated the joint parent transforms
            self.solver.notify_model_changed(SolverNotifyFlags.JOINT_PROPERTIES)

            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)

            # swap states
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()

        self.sim_time += self.frame_dt

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()

    def test_final(self):
        num_bodies_per_world = self.model.body_count // self.world_count
        for i in range(self.world_count):
            world_offset = i * num_bodies_per_world
            world_pos = wp.vec3(*self.initial_world_positions[i])

            # Test hand bodies (all except the cube) - keep original tight bounds
            hand_lower = world_pos - wp.vec3(0.5, 0.5, 0.5)
            hand_upper = world_pos + wp.vec3(0.5, 0.5, 0.5)
            hand_body_indices = np.arange(num_bodies_per_world - 1, dtype=np.int32) + world_offset
            newton.examples.test_body_state(
                self.model,
                self.state_0,
                f"hand bodies from world {i} are close to the initial position",
                lambda q, qd: newton.math.vec_inside_limits(q.p, hand_lower, hand_upper),  # noqa: B023
                indices=hand_body_indices,
            )

            # Test cube body - allow it to fall to ground plane
            # Keep X/Y bounds tight, but allow Z from ground (0.0) to initial position + 0.5
            cube_body_idx = world_offset + self.cube_body_offset
            cube_lower = wp.vec3(world_pos.x - 0.5, world_pos.y - 0.5, 0.0)
            cube_upper = world_pos + wp.vec3(0.5, 0.5, 0.5)
            newton.examples.test_body_state(
                self.model,
                self.state_0,
                f"cube from world {i} is within bounds and above ground",
                lambda q, _qd, lower=cube_lower, upper=cube_upper: newton.math.vec_inside_limits(q.p, lower, upper)
                and q.p[2] > 0.0,
                indices=np.array([cube_body_idx], dtype=np.int32),
            )

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        newton.examples.add_world_count_arg(parser)
        parser.set_defaults(world_count=100)
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)

    example = Example(viewer, args)

    newton.examples.run(example, args)
