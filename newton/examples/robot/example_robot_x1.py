# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Coordinate the X1 mobile base, lift, and dual arms in a pre-grasp task.

The current X1 asset has no grippers, so this example deliberately ends at a
pre-grasp pose instead of claiming to grasp or transport the object.

Command: python -m newton.examples robot_x1
"""

from pathlib import Path

import numpy as np
import warp as wp

import newton
import newton.examples
import newton.utils
from newton import JointTargetMode

DRIVE_DURATION = 2.0
ARM_MOTION_DURATION = 4.0
PREGRASP_TIME = DRIVE_DURATION + ARM_MOTION_DURATION
WHEEL_SPEED = 2.0


@wp.kernel
def update_x1_position_targets(
    motion_time: wp.array[wp.float32],
    home_q: wp.array[wp.float32],
    pregrasp_q: wp.array[wp.float32],
    target_q: wp.array[wp.float32],
):
    """Blend the lift and dual-arm position targets toward pre-grasp."""
    coordinate = wp.tid()
    time = motion_time[0]

    blend = wp.clamp((time - DRIVE_DURATION) / ARM_MOTION_DURATION, 0.0, 1.0)
    blend = blend * blend * (3.0 - 2.0 * blend)
    target_q[coordinate] = home_q[coordinate] + blend * (pregrasp_q[coordinate] - home_q[coordinate])


@wp.kernel
def update_x1_velocity_targets(
    motion_time: wp.array[wp.float32],
    wheel_dofs: wp.array[wp.int32],
    target_qd: wp.array[wp.float32],
):
    """Drive the two wheel DOFs forward, accounting for mirrored wheel axes."""
    wheel_index = wp.tid()
    dof = wheel_dofs[wheel_index]

    target_qd[dof] = 0.0
    if motion_time[0] < DRIVE_DURATION:
        if wheel_index == 0:
            target_qd[dof] = WHEEL_SPEED
        else:
            target_qd[dof] = -WHEEL_SPEED


@wp.kernel
def advance_x1_motion_time(motion_time: wp.array[wp.float32], dt: wp.float32):
    """Advance the device-side task clock."""
    motion_time[0] = motion_time[0] + dt


class Example:
    """Run a staged mobile-and-bimanual X1 coordination task."""

    def __init__(self, viewer, args):
        """Load X1 and configure velocity-controlled wheels and position-controlled joints."""
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = 4
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0
        self.viewer = viewer
        self.device = wp.get_device()

        asset_root = newton.utils.download_asset("onerobotics_x1")
        builder = newton.ModelBuilder()
        newton.solvers.SolverMuJoCo.register_custom_attributes(builder)
        builder.default_joint_cfg = newton.ModelBuilder.JointDofConfig(limit_ke=1.0e3, limit_kd=1.0e1)
        builder.add_urdf(
            str(asset_root / Path("urdf/x1.urdf")),
            floating=True,
            collapse_fixed_joints=False,
            enable_self_collisions=False,
        )
        builder.add_ground_plane()

        joint_indices = {}
        for index, label in enumerate(builder.joint_label):
            if sum(builder.joint_dof_dim[index]) == 0:
                continue
            short_label = label.rsplit("/", 1)[-1]
            if short_label in joint_indices:
                raise ValueError(f"X1 joint label is not unique: {short_label}")
            joint_indices[short_label] = index
        joint_dofs = {label: builder.joint_qd_start[index] for label, index in joint_indices.items()}
        joint_coordinates = {label: builder.joint_q_start[index] for label, index in joint_indices.items()}
        wheel_dofs = [joint_dofs["lun_r"], joint_dofs["lun_l"]]

        for dof in range(builder.joint_dof_count):
            builder.joint_target_mode[dof] = int(JointTargetMode.NONE)
        for dof in wheel_dofs:
            builder.joint_target_mode[dof] = int(JointTargetMode.VELOCITY)
            builder.joint_target_kd[dof] = 20.0

        controlled_joints = ("t01", "t02", "sj", *(f"r{i}" for i in range(1, 8)), *(f"l{i}" for i in range(1, 8)))
        for label in controlled_joints:
            dof = joint_dofs[label]
            builder.joint_target_mode[dof] = int(JointTargetMode.POSITION)
            builder.joint_target_ke[dof] = 500.0
            builder.joint_target_kd[dof] = 25.0
            coordinate = joint_coordinates[label]
            builder.joint_target_q[coordinate] = builder.joint_q[coordinate]
        builder.joint_target_ke[joint_dofs["sj"]] = 2.0e3
        builder.joint_target_kd[joint_dofs["sj"]] = 50.0

        home_q = np.asarray(builder.joint_target_q, dtype=np.float32)
        pregrasp_q = home_q.copy()
        pose = {
            "sj": -0.25,
            "r1": 0.20,
            "r2": -1.00,
            "r3": 0.00,
            "r4": 0.80,
            "r5": 0.00,
            "r6": 0.50,
            "r7": 0.00,
            "l1": -0.20,
            "l2": 1.00,
            "l3": 0.00,
            "l4": -0.80,
            "l5": 0.00,
            "l6": -0.50,
            "l7": 0.00,
        }
        for label, target in pose.items():
            pregrasp_q[joint_coordinates[label]] = target

        self.model = builder.finalize()
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.home_q = wp.array(home_q, dtype=wp.float32, device=self.device)
        self.pregrasp_q = wp.array(pregrasp_q, dtype=wp.float32, device=self.device)
        self.wheel_dofs = wp.array(wheel_dofs, dtype=wp.int32, device=self.device)
        self.motion_time = wp.zeros(1, dtype=wp.float32, device=self.device)
        self.controlled_coordinates = np.asarray([joint_coordinates[label] for label in pose], dtype=np.int32)
        self.base_body = next(
            (index for index, label in enumerate(builder.body_label) if label.rsplit("/", 1)[-1] == "base_link"),
            None,
        )
        if self.base_body is None:
            raise ValueError("X1 base body is missing: base_link")

        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)
        self.initial_base_transform = self.state_0.body_q.numpy()[self.base_body].copy()
        qx, qy, qz, qw = self.initial_base_transform[3:]
        self.initial_heading = np.array([1.0 - 2.0 * (qy * qy + qz * qz), 2.0 * (qx * qy + qw * qz)], dtype=np.float32)
        self.solver = newton.solvers.SolverMuJoCo(self.model)

        self.viewer.set_model(self.model)
        self.viewer.set_camera(pos=wp.vec3(3.0, -4.0, 2.0), pitch=-15.0, yaw=140.0)
        self.capture()

    def capture(self):
        """Capture a simulation frame on CUDA and use direct execution otherwise."""
        if wp.get_device().is_cuda and not wp.config.verify_cuda:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph
        else:
            self.graph = None

    def simulate(self):
        """Advance the staged X1 task by one rendered frame."""
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            wp.launch(
                update_x1_position_targets,
                dim=self.model.joint_coord_count,
                inputs=[self.motion_time, self.home_q, self.pregrasp_q],
                outputs=[self.control.joint_target_q],
                device=self.device,
            )
            wp.launch(
                update_x1_velocity_targets,
                dim=len(self.wheel_dofs),
                inputs=[self.motion_time, self.wheel_dofs],
                outputs=[self.control.joint_target_qd],
                device=self.device,
            )
            wp.launch(advance_x1_motion_time, dim=1, inputs=[self.motion_time, self.sim_dt], device=self.device)
            self.viewer.apply_forces(self.state_0)
            self.solver.step(self.state_0, self.state_1, self.control, None, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        """Advance the simulation and elapsed presentation time."""
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()
        self.sim_time += self.frame_dt

    def render(self):
        """Render the current X1 state."""
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()

    def test_final(self):
        """Verify base motion, stability, and pre-grasp convergence after the task."""
        motion_time = self.motion_time.numpy()[0]
        if motion_time <= 0.0:
            raise ValueError("X1 task clock did not advance")
        joint_q = self.state_0.joint_q.numpy()
        if not np.isfinite(joint_q).all() or not np.isfinite(self.state_0.joint_qd.numpy()).all():
            raise ValueError("X1 joint state contains non-finite values")
        if motion_time < PREGRASP_TIME:
            # The CPU test uses 10 frames as an intentional finite-value smoke test.
            return

        base_transform = self.state_0.body_q.numpy()[self.base_body]
        base_displacement = base_transform[:2] - self.initial_base_transform[:2]
        forward_displacement = np.dot(base_displacement, self.initial_heading)
        if forward_displacement < 0.05:
            raise ValueError(f"X1 base moved only {forward_displacement:.3f} m along its initial heading")

        qx, qy = base_transform[3], base_transform[4]
        upright_alignment = 1.0 - 2.0 * (qx * qx + qy * qy)
        if upright_alignment < 0.95:
            raise ValueError(f"X1 base tipped over: upright alignment is {upright_alignment:.3f}")

        pregrasp_q = self.pregrasp_q.numpy()
        max_position_error = np.max(
            np.abs(joint_q[self.controlled_coordinates] - pregrasp_q[self.controlled_coordinates])
        )
        if max_position_error > 0.05:
            raise ValueError(f"X1 controlled joints missed pre-grasp by {max_position_error:.3f} rad or m")

    @staticmethod
    def create_parser():
        """Create the common Newton example parser."""
        return newton.examples.create_parser()


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
