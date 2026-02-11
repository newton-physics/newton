# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest

import numpy as np
import warp as wp

import newton
import newton.utils
from newton import ActuatorMode
from newton._src.utils.download_assets import download_git_folder
from newton.solvers import SolverMuJoCo
from newton.tests.unittest_utils import add_function_test, find_nan_members, get_cuda_test_devices


class RobotComposerSim:
    """Test ``base_joint`` and ``parent_body`` importer functionality.

    Builds four composed-robot scenarios that exercise hierarchical
    composition across URDF, MJCF, and USD importers:

    1. UR5e (MJCF) + Robotiq 2F-85 gripper (MJCF) with a planar D6 base joint.
       The gripper is actuated via ``joint_target_pos`` on the driver joints
       (``right_driver_joint``, ``left_driver_joint``) instead of the default
       MuJoCo actuator, which is disabled to avoid instability in MJWarp.
    2. UR5e (MJCF) + LEAP hand left (MJCF) with a planar D6 base joint.
    3. Franka FR3 (URDF) + Allegro hand (MJCF) with a planar D6 base joint.
    4. UR10 (USD) with a planar D6 base joint (no end effector).

    Each scenario uses ``parent_body`` to attach the end effector to the
    arm's wrist link and ``base_joint`` to override the default fixed-base
    behaviour with a planar (2-linear + 1-angular) D6 joint.
    """

    def __init__(self, device, do_rendering=False, num_frames=50, num_worlds=2):
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = 10
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.num_worlds = num_worlds
        self.num_frames = num_frames
        self.do_rendering = do_rendering
        self.device = device

        self.collide_substeps = False
        self.gripper_target_pos = 0.0

        # Download required assets
        self._download_assets()

        # Build the scene
        builder = newton.ModelBuilder()
        self._build_scene(builder)

        # Replicate for parallel simulation
        scene = newton.ModelBuilder()
        scene.replicate(builder, self.num_worlds)
        scene.add_ground_plane()

        self.model = scene.finalize(device=device)

        # Initialize states and control
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)

        # Create solver
        self.solver = newton.solvers.SolverMuJoCo(
            self.model,
            use_mujoco_contacts=True,
            solver="newton",
            integrator="implicitfast",
            cone="elliptic",
            njmax=500,
            nconmax=500,
            iterations=15,
            ls_iterations=100,
            ls_parallel=False,
            impratio=1.0,
        )

        # MuJoCo solver handles contacts internally (use_mujoco_contacts=True),
        # so no Newton collision pipeline is needed.
        self.contacts = newton.Contacts(0, 0)

        # Create viewer
        if self.do_rendering:
            self.viewer = newton.viewer.ViewerGL()
        else:
            self.viewer = newton.viewer.ViewerNull()
        self.viewer.set_model(self.model)
        if hasattr(self.viewer, "renderer"):
            self.viewer.set_world_offsets(wp.vec3(4.0, 4.0, 0.0))

        # Initialize joint target positions
        self.joint_target_pos = wp.zeros_like(self.control.joint_target_pos)
        wp.copy(self.joint_target_pos, self.control.joint_target_pos)

        self.capture()

        # Store initial joint positions for pose verification test
        self.initial_joint_q = self.state_0.joint_q.numpy().copy()

    def _download_assets(self):
        """Download required assets from repositories."""
        print("Downloading assets...")

        # Download Franka from newton assets
        try:
            franka_asset = newton.utils.download_asset("franka_emika_panda")
            self.franka_urdf = franka_asset / "urdf" / "fr3.urdf"
            print(f"  Franka arm: {self.franka_urdf.exists()}")
        except Exception as e:
            print(f"  Could not download Franka: {e}")
            self.franka_urdf = None

        # Download from MuJoCo Menagerie
        try:
            ur5e_folder = download_git_folder(
                git_url="https://github.com/google-deepmind/mujoco_menagerie.git",
                folder_path="universal_robots_ur5e",
            )
            self.ur5e_path = ur5e_folder / "ur5e.xml"
            print(f"  UR5e: {self.ur5e_path.exists()}")
        except Exception as e:
            print(f"  Could not download UR5e: {e}")
            self.ur5e_path = None

        try:
            leap_folder = download_git_folder(
                git_url="https://github.com/google-deepmind/mujoco_menagerie.git",
                folder_path="leap_hand",
            )
            self.leap_path = leap_folder / "left_hand.xml"
            print(f"  LEAP hand left: {self.leap_path.exists()}")
        except Exception as e:
            print(f"  Could not download LEAP hand: {e}")
            self.leap_path = None

        try:
            allegro_folder = download_git_folder(
                git_url="https://github.com/google-deepmind/mujoco_menagerie.git",
                folder_path="wonik_allegro",
            )
            self.allegro_path = allegro_folder / "left_hand.xml"
            print(f"  Allegro hand: {self.allegro_path.exists()}")
        except Exception as e:
            print(f"  Could not download Allegro hand: {e}")
            self.allegro_path = None

        # Download UR10 from Newton assets
        try:
            ur10_asset = newton.utils.download_asset("universal_robots_ur10")
            self.ur10_usd = ur10_asset / "usd" / "ur10_instanceable.usda"
            print(f"  UR10: {self.ur10_usd.exists()}")
        except Exception as e:
            print(f"  Could not download UR10: {e}")
            self.ur10_usd = None

        # Download Robotiq 2F85 gripper
        try:
            robotiq_2f85_folder = download_git_folder(
                git_url="https://github.com/google-deepmind/mujoco_menagerie.git",
                folder_path="robotiq_2f85",
            )
            self.robotiq_2f85_path = robotiq_2f85_folder / "2f85.xml"
            print(f"  Robotiq 2F85 gripper: {self.robotiq_2f85_path.exists()}")
        except Exception as e:
            print(f"  Could not download Robotiq 2F85 gripper: {e}")
            self.robotiq_2f85_path = None

    def _build_scene(self, builder):
        # Small vertical offset to avoid collision with the ground plane
        z_offset = 0.05

        self._build_ur5e_mjcf_with_base_joint_and_robotiq_gripper_mjcf(builder, pos=wp.vec3(0.0, -2.0, z_offset))

        self._build_ur5e_mjcf_with_base_joint_and_leap_hand_mjcf(builder, pos=wp.vec3(0.0, -1.0, z_offset))

        self._build_franka_urdf_with_base_joint_and_allegro_hand_mjcf(builder, pos=wp.vec3(0.0, 0.0, z_offset))

        self._build_ur10_usd_with_base_joint(builder, pos=wp.vec3(0.0, 1.0, z_offset))

    def _build_ur5e_mjcf_with_base_joint_and_robotiq_gripper_mjcf(self, builder, pos):
        ur5e_with_robotiq_gripper = newton.ModelBuilder()

        # Load UR5e with fixed base
        ur5e_quat_base = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), wp.pi)
        ur5e_with_robotiq_gripper.add_mjcf(
            str(self.ur5e_path),
            xform=wp.transform(pos, ur5e_quat_base),
            base_joint={
                "joint_type": newton.JointType.D6,
                "linear_axes": [
                    newton.ModelBuilder.JointDofConfig(axis=[1.0, 0.0, 0.0]),
                    newton.ModelBuilder.JointDofConfig(axis=[0.0, 1.0, 0.0]),
                ],
                "angular_axes": [newton.ModelBuilder.JointDofConfig(axis=[0.0, 0.0, 1.0])],
            },
        )

        self.robotiq_gripper_dof_offset = ur5e_with_robotiq_gripper.joint_dof_count

        # Base joints
        ur5e_with_robotiq_gripper.joint_target_pos[:3] = [0.0, 0.0, 0.0]
        ur5e_with_robotiq_gripper.joint_target_ke[:3] = [500.0] * 3
        ur5e_with_robotiq_gripper.joint_target_kd[:3] = [50.0] * 3
        ur5e_with_robotiq_gripper.joint_act_mode[:3] = [int(ActuatorMode.POSITION)] * 3

        init_q = [0, -wp.half_pi, wp.half_pi, -wp.half_pi, -wp.half_pi, 0]
        ur5e_with_robotiq_gripper.joint_q[-6:] = init_q[:6]
        ur5e_with_robotiq_gripper.joint_target_pos[-6:] = init_q[:6]
        ur5e_with_robotiq_gripper.joint_target_ke[-6:] = [4500.0] * 6
        ur5e_with_robotiq_gripper.joint_target_kd[-6:] = [450.0] * 6
        ur5e_with_robotiq_gripper.joint_effort_limit[-6:] = [100.0] * 6
        ur5e_with_robotiq_gripper.joint_armature[-6:] = [0.2] * 6
        ur5e_with_robotiq_gripper.joint_act_mode[-6:] = [int(ActuatorMode.POSITION)] * 6

        # Find end effector body by searching body names
        ee_name = "wrist_3_link"
        ee_body_idx = ur5e_with_robotiq_gripper.body_key.index(ee_name)

        # Attach Robotiq 2F85 gripper to end effector
        gripper_quat = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), -wp.pi / 2)
        ee_xform = wp.transform((0.00, 0.1, 0.0), gripper_quat)
        ur5e_with_robotiq_gripper.add_mjcf(
            str(self.robotiq_2f85_path),
            xform=ee_xform,
            parent_body=ee_body_idx,
        )

        # Set MuJoCo control source for all actuators (6 UR5e + 1 gripper) to JOINT_TARGET.
        # Setting the gripper's ctrl_source to JOINT_TARGET disables the MuJoCo actuator that causes instability.
        # See discussion in https://github.com/google-deepmind/mujoco_warp/discussions/1112
        ctrl_source = [SolverMuJoCo.CtrlSource.JOINT_TARGET] * 7
        ur5e_with_robotiq_gripper.custom_attributes["mujoco:ctrl_source"].values[:7] = ctrl_source

        # Instead, we can actuate the gripper with joint targets for the driver joints.
        # Gripper actuated joints: right_driver_joint and left_driver_joint (dof indexes 0 and 4 within gripper)
        self.robotiq_gripper_dofs = [0, 4]

        # Set gripper joint gains
        for i in self.robotiq_gripper_dofs:
            idx = self.robotiq_gripper_dof_offset + i
            ur5e_with_robotiq_gripper.joint_target_ke[idx] = 20.0
            ur5e_with_robotiq_gripper.joint_target_kd[idx] = 1.0
            ur5e_with_robotiq_gripper.joint_target_pos[idx] = self.gripper_target_pos
            ur5e_with_robotiq_gripper.joint_act_mode[idx] = int(ActuatorMode.POSITION)

        builder.add_builder(ur5e_with_robotiq_gripper)

    def _build_ur5e_mjcf_with_base_joint_and_leap_hand_mjcf(self, builder, pos):
        ur5e_with_hand = newton.ModelBuilder()

        # Load UR5e with fixed base
        ur5e_quat_base = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), wp.pi)
        ur5e_with_hand.add_mjcf(
            str(self.ur5e_path),
            xform=wp.transform(pos, ur5e_quat_base),
            base_joint={
                "joint_type": newton.JointType.D6,
                "linear_axes": [
                    newton.ModelBuilder.JointDofConfig(axis=[1.0, 0.0, 0.0]),
                    newton.ModelBuilder.JointDofConfig(axis=[0.0, 1.0, 0.0]),
                ],
                "angular_axes": [newton.ModelBuilder.JointDofConfig(axis=[0.0, 0.0, 1.0])],
            },
        )

        # Base joints
        ur5e_with_hand.joint_target_pos[:3] = [0.0, 0.0, 0.0]
        ur5e_with_hand.joint_target_ke[:3] = [500.0] * 3
        ur5e_with_hand.joint_target_kd[:3] = [50.0] * 3
        ur5e_with_hand.joint_act_mode[:3] = [int(ActuatorMode.POSITION)] * 3

        init_q = [0, -wp.half_pi, wp.half_pi, -wp.half_pi, -wp.half_pi, 0]
        ur5e_with_hand.joint_q[-6:] = init_q[:6]
        ur5e_with_hand.joint_target_pos[-6:] = init_q[:6]
        ur5e_with_hand.joint_target_ke[-6:] = [4500.0] * 6
        ur5e_with_hand.joint_target_kd[-6:] = [450.0] * 6
        ur5e_with_hand.joint_effort_limit[-6:] = [100.0] * 6
        ur5e_with_hand.joint_armature[-6:] = [0.2] * 6
        ur5e_with_hand.joint_act_mode[-6:] = [int(ActuatorMode.POSITION)] * 6

        # Find end effector body by searching body names
        ee_name = "wrist_3_link"
        ee_body_idx = ur5e_with_hand.body_key.index(ee_name)

        # Attach LEAP hand left to end effector
        quat_z = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), wp.pi / 2)
        quat_y = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), wp.pi)
        hand_quat = quat_y * quat_z
        ee_xform = wp.transform((-0.065, 0.28, 0.10), hand_quat)
        ur5e_with_hand.add_mjcf(
            str(self.leap_path),
            xform=ee_xform,
            parent_body=ee_body_idx,
        )

        # Set ctrl_source of all Mujoco actuators to be JOINT_TARGET
        num_mujoco_actuators = len(ur5e_with_hand.custom_attributes["mujoco:ctrl_source"].values)
        ctrl_source = [SolverMuJoCo.CtrlSource.JOINT_TARGET] * num_mujoco_actuators
        ur5e_with_hand.custom_attributes["mujoco:ctrl_source"].values = ctrl_source

        builder.add_builder(ur5e_with_hand)

    def _build_franka_urdf_with_base_joint_and_allegro_hand_mjcf(self, builder, pos):
        franka_with_hand = newton.ModelBuilder()

        # Load Franka arm with base joint
        franka_with_hand.add_urdf(
            str(self.franka_urdf),
            xform=wp.transform(pos),
            base_joint={
                "joint_type": newton.JointType.D6,
                "linear_axes": [
                    newton.ModelBuilder.JointDofConfig(axis=[1.0, 0.0, 0.0]),
                    newton.ModelBuilder.JointDofConfig(axis=[0.0, 1.0, 0.0]),
                ],
                "angular_axes": [newton.ModelBuilder.JointDofConfig(axis=[0.0, 0.0, 1.0])],
            },
        )

        # Base joints
        franka_with_hand.joint_target_pos[:3] = [0.0, 0.0, 0.0]
        franka_with_hand.joint_target_ke[:3] = [500.0] * 3
        franka_with_hand.joint_target_kd[:3] = [50.0] * 3
        franka_with_hand.joint_act_mode[:3] = [int(ActuatorMode.POSITION)] * 3

        # Set panda joint positions and joint targets
        init_q = [
            -3.6802115e-03,
            2.3901723e-02,
            3.6804110e-03,
            -2.3683236e00,
            -1.2918962e-04,
            2.3922248e00,
            7.8549200e-01,
        ]

        franka_with_hand.joint_q[-7:] = init_q[:7]
        franka_with_hand.joint_target_pos[-7:] = init_q[:7]
        franka_with_hand.joint_target_ke[-7:] = [4500, 4500, 3500, 3500, 2000, 2000, 2000]
        franka_with_hand.joint_target_kd[-7:] = [450, 450, 350, 350, 200, 200, 200]
        franka_with_hand.joint_effort_limit[-7:] = [87, 87, 87, 87, 12, 12, 12]
        franka_with_hand.joint_armature[-7:] = [0.195] * 4 + [0.074] * 3
        franka_with_hand.joint_act_mode[-7:] = [int(ActuatorMode.POSITION)] * 7

        # Find end effector body by searching body names
        franka_ee_name = "fr3_link8"
        franka_ee_idx = franka_with_hand.body_key.index(franka_ee_name)

        # Attach Allegro hand with custom base joint
        quat_z = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), -init_q[-1])
        quat_y = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), -wp.pi / 2)
        hand_quat = quat_z * quat_y
        ee_xform = wp.transform((0.0, 0.0, 0.1), hand_quat)

        franka_with_hand.add_mjcf(
            str(self.allegro_path),
            xform=ee_xform,
            parent_body=franka_ee_idx,
        )

        allegro_dof_count = franka_with_hand.joint_dof_count - 7 - 3
        franka_with_hand.joint_target_pos[-allegro_dof_count:] = franka_with_hand.joint_q[-allegro_dof_count:]

        num_mujoco_actuators = len(franka_with_hand.custom_attributes["mujoco:ctrl_source"].values)
        ctrl_source = [SolverMuJoCo.CtrlSource.JOINT_TARGET] * num_mujoco_actuators
        franka_with_hand.custom_attributes["mujoco:ctrl_source"].values = ctrl_source

        builder.add_builder(franka_with_hand)

    def _build_ur10_usd_with_base_joint(self, builder, pos):
        ur10_builder = newton.ModelBuilder()

        # Load UR10 from USD with planar base joint (like UR5e)
        ur10_builder.add_usd(
            str(self.ur10_usd),
            xform=wp.transform(pos),
            enable_self_collisions=False,
            hide_collision_shapes=True,
            base_joint={
                "joint_type": newton.JointType.D6,
                "linear_axes": [
                    newton.ModelBuilder.JointDofConfig(axis=[1.0, 0.0, 0.0]),
                    newton.ModelBuilder.JointDofConfig(axis=[0.0, 1.0, 0.0]),
                ],
                "angular_axes": [newton.ModelBuilder.JointDofConfig(axis=[0.0, 0.0, 1.0])],
            },
        )

        # Set gains for base joint DOFs (first 3 DOFs)
        ur10_builder.joint_target_pos[:3] = [0.0, 0.0, 0.0]
        ur10_builder.joint_target_ke[:3] = [500.0] * 3
        ur10_builder.joint_target_kd[:3] = [50.0] * 3
        ur10_builder.joint_act_mode[:3] = [int(ActuatorMode.POSITION)] * 3

        # Initialize arm joints to elbow down configuration (same as UR5e)
        init_q = [0, -wp.half_pi, wp.half_pi, -wp.half_pi, -wp.half_pi, 0]
        ur10_builder.joint_q[-6:] = init_q[:6]
        ur10_builder.joint_target_pos[-6:] = init_q[:6]

        # Set joint targets and gains for arm joints
        ur10_builder.joint_target_ke[-6:] = [4500.0] * 6
        ur10_builder.joint_target_kd[-6:] = [450.0] * 6
        ur10_builder.joint_effort_limit[-6:] = [100.0] * 6
        ur10_builder.joint_armature[-6:] = [0.2] * 6
        ur10_builder.joint_act_mode[-6:] = [int(ActuatorMode.POSITION)] * 6

        builder.add_builder(ur10_builder)

    def capture(self):
        """Capture simulation graph for efficient execution."""
        self.graph = None
        if wp.get_device().is_cuda:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph

    def simulate(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            # apply forces to the model for picking, wind, etc
            self.viewer.apply_forces(self.state_0)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        wp.copy(self.control.joint_target_pos, self.joint_target_pos)

        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()

        self.sim_time += self.frame_dt

    def render(self):
        """Render the current state."""
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()

    def gui(self, imgui):
        imgui.text("Robotiq 2F85 gripper target")

        def update_gripper_target_pos(value):
            self.gripper_target_pos = value
            # The actuated joints are right_driver_joint and left_driver_joint (dof indexes 0 and 4 within gripper).
            # robotiq_gripper_dof_offset accounts for base_joint(3) + arm(6) DOFs.
            joint_target_pos = self.joint_target_pos.reshape((self.num_worlds, -1)).numpy()
            for i in self.robotiq_gripper_dofs:
                joint_target_pos[:, self.robotiq_gripper_dof_offset + i] = value
            wp.copy(self.joint_target_pos, wp.array(joint_target_pos.flatten(), dtype=wp.float32))

        changed, value = imgui.slider_float(
            "gripper_target_pos_slider", self.gripper_target_pos, 0.0, 0.8, format="%.3f"
        )
        if changed:
            update_gripper_target_pos(value)

        changed, value = imgui.input_float("gripper_target_pos", self.gripper_target_pos, format="%.3f")
        if changed:
            value = min(max(value, 0.0), 0.8)
            update_gripper_target_pos(value)

    def run(self):
        if self.do_rendering:
            if hasattr(self.viewer, "register_ui_callback"):
                self.viewer.register_ui_callback(lambda ui: self.gui(ui), position="side")
            while self.viewer.is_running():
                if not self.viewer.is_paused():
                    self.step()
                self.render()
        else:
            for _ in range(self.num_frames):
                self.step()


def test_robot_composer(test, device):
    """Test that composed robots build correctly, simulate stably, and move."""
    sim = RobotComposerSim(device, num_frames=50, num_worlds=2)

    # Model structure: at least 4 articulations (UR5e+Robotiq, UR5e+LEAP, Franka+Allegro, UR10)
    test.assertGreaterEqual(sim.model.articulation_count, 4)
    test.assertGreater(sim.model.body_count, 20)
    test.assertGreater(sim.model.joint_count, 20)
    test.assertGreater(sim.state_0.joint_q.shape[0], 0)

    sim.run()

    # Stability: no NaN or non-finite values
    nan_members_0 = find_nan_members(sim.state_0)
    nan_members_1 = find_nan_members(sim.state_1)
    test.assertEqual(nan_members_0, [], f"NaN found in state_0: {nan_members_0}")
    test.assertEqual(nan_members_1, [], f"NaN found in state_1: {nan_members_1}")

    joint_q = sim.state_0.joint_q.numpy()
    joint_qd = sim.state_0.joint_qd.numpy()
    test.assertTrue(np.isfinite(joint_q).all(), "Non-finite values in joint_q")
    test.assertTrue(np.isfinite(joint_qd).all(), "Non-finite values in joint_qd")

    # Movement: at least some joints should have changed
    test.assertTrue(
        np.any(np.abs(sim.initial_joint_q - joint_q) > 1e-6),
        "No joints moved during simulation",
    )


devices = get_cuda_test_devices(mode="basic")


class TestRobotComposer(unittest.TestCase):
    pass


add_function_test(
    TestRobotComposer,
    "test_robot_composer",
    test_robot_composer,
    devices=devices,
    check_output=False,
)


if __name__ == "__main__":
    unittest.main(verbosity=2, failfast=True)
