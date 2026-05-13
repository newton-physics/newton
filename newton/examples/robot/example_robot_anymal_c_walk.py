# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Robot ANYmal C Walk
#
# Shows how to simulate ANYmal C using SolverMuJoCo and control it with a
# policy trained in PhysX, exported to ONNX and run via Newton's
# Warp-backed :class:`~newton.utils.OnnxRuntime`.
#
# Command: python -m newton.examples robot_anymal_c_walk
#
###########################################################################

import numpy as np
import warp as wp

import newton
import newton.examples
import newton.utils

lab_to_mujoco = [0, 6, 3, 9, 1, 7, 4, 10, 2, 8, 5, 11]
mujoco_to_lab = [0, 4, 8, 2, 6, 10, 1, 5, 9, 3, 7, 11]


@wp.kernel
def _build_joint_target_pos_kernel(
    act: wp.array2d[float],  # (1, num_dofs) on device
    joint_pos_initial: wp.array[float],  # (num_dofs,) on device
    reorder: wp.array[int],  # (num_dofs,) mjc_to_physx mapping
    action_scale: float,
    num_prefix_zeros: int,
    out: wp.array[float],  # (num_prefix_zeros + num_dofs,)
):
    """Reorder, scale, and prepend ``num_prefix_zeros`` zeros into ``out``.

    Avoids the device->host->device round-trip the NumPy version did each
    policy step and keeps the value on-device for graph capture.
    """
    i = wp.tid()
    if i < num_prefix_zeros:
        out[i] = 0.0
    else:
        j = i - num_prefix_zeros
        idx = reorder[j]
        out[i] = joint_pos_initial[j] + action_scale * act[0, idx]


@wp.kernel
def _compute_obs_kernel(
    joint_q: wp.array[float],  # full joint_q, [px,py,pz, qx,qy,qz,qw, q0..q11]
    joint_qd: wp.array[float],  # full joint_qd, [vx,vy,vz, wx,wy,wz, qd0..qd11]
    joint_pos_initial: wp.array[float],  # (12,) reference posture (lab order)
    lab_to_mujoco_idx: wp.array[int],  # (12,) reordering for joint pos/vel
    gravity_w: wp.vec3,  # gravity unit vector in world frame
    command: wp.vec3,  # (fwd, lat, yaw) command
    prev_act: wp.array2d[float],  # (1, 12) previous policy action
    obs: wp.array2d[float],  # (1, 48) destination
):
    """Build the 48-element ANYmal observation entirely on-device.

    Layout:
      [0:3]   body-frame linear velocity
      [3:6]   body-frame angular velocity
      [6:9]   body-frame projected gravity
      [9:12]  command (fwd, lat, yaw)
      [12:24] joint position error, reordered to mujoco order
      [24:36] joint velocity, reordered to mujoco order
      [36:48] previous action
    """
    # qx,qy,qz,qw is the XYZW layout used by both the URDF importer and
    # warp.quat (warp expects XYZW too); reading 4 floats this way matches
    # the previous NumPy code exactly.
    q = wp.quat(joint_q[3], joint_q[4], joint_q[5], joint_q[6])

    lin_w = wp.vec3(joint_qd[0], joint_qd[1], joint_qd[2])
    ang_w = wp.vec3(joint_qd[3], joint_qd[4], joint_qd[5])

    vel_b = wp.quat_rotate_inv(q, lin_w)
    avel_b = wp.quat_rotate_inv(q, ang_w)
    grav_b = wp.quat_rotate_inv(q, gravity_w)

    obs[0, 0] = vel_b[0]
    obs[0, 1] = vel_b[1]
    obs[0, 2] = vel_b[2]
    obs[0, 3] = avel_b[0]
    obs[0, 4] = avel_b[1]
    obs[0, 5] = avel_b[2]
    obs[0, 6] = grav_b[0]
    obs[0, 7] = grav_b[1]
    obs[0, 8] = grav_b[2]
    obs[0, 9] = command[0]
    obs[0, 10] = command[1]
    obs[0, 11] = command[2]

    # Reorder joint pos error / vel from lab order into mujoco order.
    for k in range(12):
        idx = lab_to_mujoco_idx[k]
        obs[0, 12 + k] = joint_q[7 + idx] - joint_pos_initial[idx]
        obs[0, 24 + k] = joint_qd[6 + idx]
        obs[0, 36 + k] = prev_act[0, k]


class Example:
    def __init__(self, viewer, args):
        self.viewer = viewer
        self.device = wp.get_device()

        builder = newton.ModelBuilder()
        newton.solvers.SolverMuJoCo.register_custom_attributes(builder)
        builder.default_joint_cfg = newton.ModelBuilder.JointDofConfig(
            armature=0.06,
            limit_ke=1.0e3,
            limit_kd=1.0e1,
        )
        builder.default_shape_cfg.ke = 5.0e4
        builder.default_shape_cfg.kd = 5.0e2
        builder.default_shape_cfg.kf = 1.0e3
        builder.default_shape_cfg.mu = 0.75

        asset_path = newton.utils.download_asset("anybotics_anymal_c")
        stage_path = str(asset_path / "urdf" / "anymal.urdf")
        builder.add_urdf(
            stage_path,
            xform=wp.transform(wp.vec3(0.0, 0.0, 0.62), wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), wp.pi * 0.5)),
            floating=True,
            enable_self_collisions=False,
            collapse_fixed_joints=True,
            ignore_inertial_definitions=False,
        )

        builder.add_ground_plane()

        self.sim_time = 0.0
        self.sim_step = 0
        fps = 50
        self.frame_dt = 1.0 / fps

        self.sim_substeps = 4
        self.sim_dt = self.frame_dt / self.sim_substeps

        initial_q = {
            "RH_HAA": 0.0,
            "RH_HFE": -0.4,
            "RH_KFE": 0.8,
            "LH_HAA": 0.0,
            "LH_HFE": -0.4,
            "LH_KFE": 0.8,
            "RF_HAA": 0.0,
            "RF_HFE": 0.4,
            "RF_KFE": -0.8,
            "LF_HAA": 0.0,
            "LF_HFE": 0.4,
            "LF_KFE": -0.8,
        }
        for name, value in initial_q.items():
            idx = next(
                (i for i, lbl in enumerate(builder.joint_label) if lbl.endswith(f"/{name}")),
                None,
            )
            if idx is None:
                raise ValueError(f"Joint '{name}' not found in builder.joint_label")
            builder.joint_q[idx + 6] = value

        for i in range(len(builder.joint_target_ke)):
            builder.joint_target_ke[i] = 150
            builder.joint_target_kd[i] = 5

        self.model = builder.finalize()
        use_mujoco_contacts = getattr(args, "use_mujoco_contacts", False)

        self.solver = newton.solvers.SolverMuJoCo(
            self.model,
            use_mujoco_contacts=use_mujoco_contacts,
            solver="newton",
            ls_parallel=False,
            ls_iterations=50,
            njmax=50,
            nconmax=100,
        )

        self.viewer.set_model(self.model)

        self.follow_cam = True

        if isinstance(self.viewer, newton.viewer.ViewerGL):

            def toggle_follow_cam(imgui):
                changed, follow_cam = imgui.checkbox("Follow Camera", self.follow_cam)
                if changed:
                    self.follow_cam = follow_cam

            self.viewer.register_ui_callback(toggle_follow_cam, position="side")

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()

        newton.eval_fk(self.model, self.state_0.joint_q, self.state_0.joint_qd, self.state_0)

        if use_mujoco_contacts:
            self.contacts = None
        else:
            self.contacts = self.model.contacts()

        # Load ONNX policy bundled with Newton (Warp-backed runtime, no torch dependency).
        policy_path = str(asset_path / "rl_policies" / "anymal_walking_policy_physx.onnx")

        self.policy = newton.utils.OnnxRuntime(policy_path, device=str(self.device))
        self._policy_input_name = self.policy.input_names[0]
        self._policy_output_name = self.policy.output_names[0]

        # Everything the policy step touches is on-device.  The host never
        # syncs joint state, builds the obs vector, or copies the action back.
        # Reference posture is sliced on-device to avoid a startup host sync.
        self._joint_pos_initial_wp = wp.clone(self.state_0.joint_q[7:])
        self._lab_to_mujoco_wp = wp.array(np.asarray(lab_to_mujoco, dtype=np.int32), dtype=wp.int32, device=self.device)
        self._mujoco_to_lab_wp = wp.array(np.asarray(mujoco_to_lab, dtype=np.int32), dtype=wp.int32, device=self.device)
        self._gravity_w = wp.vec3(0.0, 0.0, -1.0)
        # Command is mutable host-side; the kernel reads it as a wp.vec3 value
        # passed by argument, so it never lives in a device buffer.
        self._command = wp.vec3(1.0, 0.0, 0.0)

        self._obs_wp = wp.zeros((1, 48), dtype=wp.float32, device=self.device)
        # Keeps the previous policy action on device so it can be fed back
        # into the next obs without a host round-trip.
        self._prev_act_wp = wp.zeros((1, 12), dtype=wp.float32, device=self.device)
        self._action_scale = 0.5
        self._num_prefix_zeros = 6
        self._num_dofs = 12

        self.capture()

    def capture(self):
        self.graph = None
        self.use_cuda_graph = False
        # Mirror the gating used by example_robot_policy: graph capture only
        # works when both CUDA and the Warp memory pool are available.
        if self.device.is_cuda and wp.is_mempool_enabled(self.device):
            self.use_cuda_graph = True
            self.control.joint_target_pos = wp.zeros(18, dtype=wp.float32, device=self.device)
            # Capture the full step (obs build + policy + joint_target_pos +
            # substeps) so the host issues a single ``wp.capture_launch`` per
            # frame and the GPU never waits on Python.
            with wp.ScopedCapture() as capture:
                self._policy_step()
                self.simulate()
            self.graph = capture.graph

    def simulate(self):
        # When the substep count is odd, swapping the python references at
        # capture time leaves the captured graph reading from the *wrong*
        # buffer on every replay.  Mirror the pattern used in
        # example_robot_policy: copy state_1 into state_0 in-place on the
        # last odd substep, swap python refs otherwise.
        need_state_copy = self.use_cuda_graph and self.sim_substeps % 2 == 1

        for i in range(self.sim_substeps):
            self.state_0.clear_forces()

            self.viewer.apply_forces(self.state_0)

            if self.contacts is not None:
                self.model.collide(self.state_0, self.contacts)

            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)

            if need_state_copy and i == self.sim_substeps - 1:
                self.state_0.assign(self.state_1)
            else:
                self.state_0, self.state_1 = self.state_1, self.state_0

    def _policy_step(self):
        """Build obs -> run policy -> write joint_target_pos.  All on-device."""
        wp.launch(
            _compute_obs_kernel,
            dim=1,
            inputs=[
                self.state_0.joint_q,
                self.state_0.joint_qd,
                self._joint_pos_initial_wp,
                self._lab_to_mujoco_wp,
                self._gravity_w,
                self._command,
                self._prev_act_wp,
                self._obs_wp,
            ],
            device=self.device,
        )
        out = self.policy({self._policy_input_name: self._obs_wp})
        act_wp = out[self._policy_output_name]

        # Build joint_target_pos directly on device: reorder, scale,
        # prepend the six zeros for the floating base in a single launch.
        wp.launch(
            _build_joint_target_pos_kernel,
            dim=self._num_prefix_zeros + self._num_dofs,
            inputs=[
                act_wp,
                self._joint_pos_initial_wp,
                self._mujoco_to_lab_wp,
                self._action_scale,
                self._num_prefix_zeros,
                self.control.joint_target_pos,
            ],
            device=self.device,
        )

        # Stash the action for the *next* obs build, on device (no sync).
        wp.copy(self._prev_act_wp, act_wp)

    def step(self):
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self._policy_step()
            self.simulate()
        self.sim_time += self.frame_dt

    def render(self):
        if self.follow_cam:
            self.viewer.set_camera(
                pos=wp.vec3(*self.state_0.joint_q.numpy()[:3]) + wp.vec3(10.0, 0.0, 2.0), pitch=0.0, yaw=-180.0
            )

        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()

    def test_final(self):
        body_names = [lbl.split("/")[-1] for lbl in self.model.body_label]
        assert body_names == [
            "base",
            "LF_HIP",
            "LF_THIGH",
            "LF_SHANK",
            "RF_HIP",
            "RF_THIGH",
            "RF_SHANK",
            "LH_HIP",
            "LH_THIGH",
            "LH_SHANK",
            "RH_HIP",
            "RH_THIGH",
            "RH_SHANK",
        ]
        joint_names = [lbl.split("/")[-1] for lbl in self.model.joint_label]
        assert joint_names == [
            "floating_base",
            "LF_HAA",
            "LF_HFE",
            "LF_KFE",
            "RF_HAA",
            "RF_HFE",
            "RF_KFE",
            "LH_HAA",
            "LH_HFE",
            "LH_KFE",
            "RH_HAA",
            "RH_HFE",
            "RH_KFE",
        ]

        newton.examples.test_body_state(
            self.model,
            self.state_0,
            "all bodies are above the ground",
            lambda q, qd: q[2] > 0.1,
        )

        newton.examples.test_body_state(
            self.model,
            self.state_0,
            "the robot went in the right direction",
            lambda q, qd: q[1] > 9.0,
        )

        forward_vel_min = wp.spatial_vector(-0.5, 0.9, -0.2, -0.8, -1.5, -0.5)
        forward_vel_max = wp.spatial_vector(0.5, 1.1, 0.2, 0.8, 1.5, 0.5)
        newton.examples.test_body_state(
            self.model,
            self.state_0,
            "the robot is moving forward and not falling",
            lambda q, qd: newton.math.vec_inside_limits(qd, forward_vel_min, forward_vel_max),
            indices=[0],
        )

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        newton.examples.add_mujoco_contacts_arg(parser)
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)

    newton.examples.run(Example(viewer, args), args)
