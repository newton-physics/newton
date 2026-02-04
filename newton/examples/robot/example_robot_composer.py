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

###########################################################################
# Example Robot Composer
#
# Shows how to compose robots by attaching end effectors using parent_body.
# Uses multiple importers (URDF, MJCF) and demonstrates different base joint configurations (floating, planar).
#
# This example showcases:
# - Composing robots by attaching end effectors using parent_body
# - Using multiple importers (URDF, MJCF)
# - Overriding floating/base_joint behavior during import
# - Stress testing articulation bookkeeping with multiple compositions
#
# We create several scenarios demonstrating hierarchical composition:
# 1. UR5e + LEAP hand left (MJCF + MJCF composition)
# 2. Franka arm + Allegro hand (URDF + MJCF composition)
# 3. Robots with different base joint configurations (floating, planar)
#
#
# Command: python -m newton.examples robot_composer --num-worlds 1
#
###########################################################################

import warp as wp

import newton
import newton.examples
import newton.utils
from newton import ActuatorMode
from newton._src.utils.download_assets import download_git_folder
from newton.solvers import SolverMuJoCo


class Example:
    def __init__(self, viewer, num_worlds=1, args=None):
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = 10
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.num_worlds = num_worlds
        self.viewer = viewer

        self.viewer._paused = True

        self.collide_substeps = False

        # Download required assets
        self._download_assets()

        # Build the scene
        builder = newton.ModelBuilder()
        self._build_scene(builder)

        # Replicate for parallel simulation
        scene = newton.ModelBuilder()
        scene.replicate(builder, self.num_worlds)
        scene.add_ground_plane()

        self.model = scene.finalize()

        # Initialize states and control
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)

        # Create solver
        self.use_mujoco_contacts = args.use_mujoco_contacts if args is not None else False
        self.solver = newton.solvers.SolverMuJoCo(
            self.model,
            use_mujoco_contacts=self.use_mujoco_contacts,
            solver="newton",
            integrator="implicitfast",
            cone="elliptic",
            njmax=10000,
            nconmax=10000,
            iterations=15,
            ls_iterations=100,
            ls_parallel=True,
            impratio=1000.0,
        )

        # Create collision pipeline from command-line args (default: CollisionPipelineUnified with EXPLICIT)
        self.collision_pipeline = newton.examples.create_collision_pipeline(self.model, args)
        self.contacts = self.model.collide(self.state_0, collision_pipeline=self.collision_pipeline)

        # Setup viewer
        self.viewer.set_model(self.model)
        if hasattr(self.viewer, "renderer"):
            self.viewer.set_world_offsets(wp.vec3(4.0, 4.0, 0.0))

        # Initialize joint target positions.
        self.direct_control = wp.zeros_like(self.control.mujoco.ctrl)
        self.gripper_target_pos = 0.0

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
        z_offset = 0.01

        self._build_ur5e_with_robotiq_gripper(builder, pos=wp.vec3(0.0, -2.0, z_offset))

        self._build_ur5e_mjcf_with_base_joint_and_leap_hand_mjcf(builder, pos=wp.vec3(0.0, -1.0, z_offset))

        self._build_franka_urdf_with_allegro_hand_mjcf(builder, pos=wp.vec3(0.0, 0.0, z_offset))

        self._build_ur10_usd_with_base_joint(builder, pos=wp.vec3(0.0, 1.0, z_offset))

    def _build_ur5e_with_robotiq_gripper(self, builder, pos):
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
        # Rotate the gripper to align with the arm
        gripper_quat = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), -wp.pi / 2)
        # ee_xform is set for illustrative purposes.
        ee_xform = wp.transform((0.00, 0.1, 0.0), gripper_quat)
        ur5e_with_robotiq_gripper.add_mjcf(
            str(self.robotiq_2f85_path),
            xform=ee_xform,
            parent_body=ee_body_idx,
        )

        # Set MuJoCo control source for the ur5e (first 6 actuators)
        ur5e_ctrl_source = [SolverMuJoCo.CtrlSource.JOINT_TARGET] * 6
        ur5e_with_robotiq_gripper.custom_attributes["mujoco:ctrl_source"].values[:6] = ur5e_ctrl_source

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
        # Rotate the hand to align with the arm
        quat_z = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), wp.pi / 2)
        quat_y = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), wp.pi)
        hand_quat = quat_y * quat_z
        # ee_xform is set for illustrative purposes.
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

    def _build_franka_urdf_with_allegro_hand_mjcf(self, builder, pos):
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
        # Rotate the hand around the y axis by -90 degrees
        quat_z = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), -init_q[-1])
        quat_y = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), -wp.pi / 2)
        hand_quat = quat_z * quat_y
        # ee_xform is set for illustrative purposes.
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
            },  # Planar mobile base
        )

        # Set gains for base joint DOFs (first 3 DOFs)
        ur10_builder.joint_target_pos[:3] = [0.0, 0.0, 0.0]
        ur10_builder.joint_target_ke[:3] = [500.0] * 3
        ur10_builder.joint_target_kd[:3] = [50.0] * 3
        ur10_builder.joint_act_mode[:3] = [int(ActuatorMode.POSITION)] * 3

        # Initialize arm joints to elbow down configuration (same as UR5e)
        # Arm joints are the last 6
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
        if not self.collide_substeps and not self.use_mujoco_contacts:
            self.contacts = self.model.collide(self.state_0, collision_pipeline=self.collision_pipeline)

        for _ in range(self.sim_substeps):
            if self.collide_substeps and not self.use_mujoco_contacts:
                self.contacts = self.model.collide(self.state_0, collision_pipeline=self.collision_pipeline)

            self.state_0.clear_forces()

            # Apply forces for interactive picking
            self.viewer.apply_forces(self.state_0)

            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)

            # Swap states
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        # Set direct control from GUI
        wp.copy(self.control.mujoco.ctrl, self.direct_control)

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
        imgui.text("Gripper target")
        # The first robot to be built is the UR5e with Robotiq 2F85 gripper.
        # The gripper motor is the 7th actuator. So, the actuator index is 6.
        actuator_idx = 6
        changed, value = imgui.slider_float("gripper_target_pos", self.gripper_target_pos, 0.0, 255, format="%.3f")
        if changed:
            self.gripper_target_pos = value
            direct_control = self.direct_control.reshape((self.num_worlds, -1)).numpy()
            direct_control[:, actuator_idx] = value
            wp.copy(self.direct_control, wp.array(direct_control.flatten(), dtype=wp.float32))

    def test_final(self):
        """Test that the composed model is valid and simulates correctly."""
        pass


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    parser.add_argument("--num-worlds", type=int, default=4, help="Total number of simulated worlds.")

    viewer, args = newton.examples.init(parser)

    args.use_mujoco_contacts = True

    example = Example(viewer, num_worlds=args.num_worlds, args=args)

    newton.examples.run(example, args)
