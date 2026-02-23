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
# Example Robot Menagerie USD
#
# Parameterized example for visualizing and testing MuJoCo Menagerie
# robot assets that have been converted to USD format.
#
# Usage:
#   uv run -m newton.examples robot_menagerie_usd --robot h1
#   uv run -m newton.examples robot_menagerie_usd --robot apptronik_apollo
#   uv run -m newton.examples robot_menagerie_usd --robot shadow_hand
#
# Available robots:
#   apptronik_apollo, g1_with_hands, h1,
#   robotiq_2f85_v4, shadow_hand, wonik_allegro
#
###########################################################################

from pathlib import Path

import numpy as np
import warp as wp

import newton
import newton.examples
from newton import ActuatorMode

# Menagerie USD asset configurations.
# TODO: Migrate to newton-assets repo. When available, replace local paths with
# download_asset("menagerie/<robot>") from newton._src.utils.download_assets.
MENAGERIE_USD_ROBOTS = {
    "apptronik_apollo": {
        "path": "newton/tests/assets/menagerie/apptronik_apollo/apptronik_apollo scene.usda",
        "initial_height": 1.0,
        "is_floating": True,
    },
    "g1_with_hands": {
        "path": "newton/tests/assets/menagerie/g1_with_hands/g1_29dof_with_hand_rev_1_0 scene.usda",
        "initial_height": 0.8,
        "is_floating": True,
    },
    "h1": {
        "path": "newton/tests/assets/menagerie/h1/h1 scene.usda",
        "initial_height": 1.0,
        "is_floating": True,
    },
    "robotiq_2f85_v4": {
        "path": "newton/tests/assets/menagerie/robotiq_2f85_v4/2f85 scene.usda",
        "initial_height": 0.3,
        "is_floating": False,
    },
    "shadow_hand": {
        "path": "newton/tests/assets/menagerie/shadow_hand/right_shadow_hand scene.usda",
        "initial_height": 0.5,
        "is_floating": False,
    },
    "wonik_allegro": {
        "path": "newton/tests/assets/menagerie/wonik_allegro/allegro_right.usda",
        "initial_height": 0.5,
        "is_floating": False,
    },
}


def get_repo_root() -> Path:
    """Get the newton repository root directory."""
    return Path(__file__).parent.parent.parent.parent


class Example:
    def __init__(self, viewer, robot_name: str, num_worlds: int = 4, args=None):
        self.fps = 50
        self.frame_dt = 1.0 / self.fps

        self.sim_time = 0.0
        self.sim_substeps = 4
        self.sim_dt = self.frame_dt / self.sim_substeps

        self.num_worlds = num_worlds
        self.robot_name = robot_name

        self.viewer = viewer
        self.device = wp.get_device()

        # Get robot config
        if robot_name not in MENAGERIE_USD_ROBOTS:
            raise ValueError(f"Unknown robot: {robot_name}. Available: {list(MENAGERIE_USD_ROBOTS.keys())}")

        robot_config = MENAGERIE_USD_ROBOTS[robot_name]
        asset_path = get_repo_root() / robot_config["path"]

        if not asset_path.exists():
            raise FileNotFoundError(f"Asset not found: {asset_path}")

        # Build robot template
        robot_builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        newton.solvers.SolverMuJoCo.register_custom_attributes(robot_builder)
        robot_builder.default_joint_cfg = newton.ModelBuilder.JointDofConfig(
            limit_ke=1.0e3,
            limit_kd=1.0e1,
            friction=1e-5,
        )
        robot_builder.default_shape_cfg.ke = 2.0e3
        robot_builder.default_shape_cfg.kd = 1.0e2
        robot_builder.default_shape_cfg.kf = 1.0e3
        robot_builder.default_shape_cfg.mu = 0.75

        initial_height = robot_config["initial_height"]
        robot_builder.add_usd(
            str(asset_path),
            xform=wp.transform(wp.vec3(0, 0, initial_height)),
            collapse_fixed_joints=False,
            enable_self_collisions=False,
            hide_collision_shapes=True,
        )

        # Approximate meshes for faster collision detection
        robot_builder.approximate_meshes("bounding_box")

        # Set joint actuator properties
        for i in range(robot_builder.joint_dof_count):
            robot_builder.joint_target_ke[i] = 150.0
            robot_builder.joint_target_kd[i] = 5.0
            robot_builder.joint_act_mode[i] = int(ActuatorMode.POSITION)

        # Build main model with replication
        builder = newton.ModelBuilder()
        builder.replicate(robot_builder, self.num_worlds)

        builder.default_shape_cfg.ke = 1.0e3
        builder.default_shape_cfg.kd = 1.0e2
        builder.add_ground_plane()

        self.model = builder.finalize()

        # Create solver
        self.solver = newton.solvers.SolverMuJoCo(
            self.model,
            iterations=100,
            ls_iterations=50,
            njmax=300,
            nconmax=150,
            use_mujoco_contacts=args.use_mujoco_contacts if args else False,
        )

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()

        # Initialize FK
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)

        # Create collision pipeline
        self.collision_pipeline = newton.examples.create_collision_pipeline(self.model, args)
        self.contacts = self.collision_pipeline.contacts()
        self.collision_pipeline.collide(self.state_0, self.contacts)

        self.viewer.set_model(self.model)
        self.viewer.set_world_offsets((3.0, 3.0, 0.0))

        self.capture()

    def capture(self):
        self.graph = None
        if wp.get_device().is_cuda:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph

    def simulate(self):
        self.collision_pipeline.collide(self.state_0, self.contacts)
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()

            # Apply forces for picking, wind, etc.
            self.viewer.apply_forces(self.state_0)

            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)

            # Swap states
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
        # Check that all bodies are above the ground
        newton.examples.test_body_state(
            self.model,
            self.state_0,
            "all bodies are above the ground",
            lambda q, qd: q[2] > -0.1,
        )

        # Check for NaN values in state
        body_q = self.state_0.body_q.numpy()
        body_qd = self.state_0.body_qd.numpy()

        if np.any(np.isnan(body_q)):
            raise AssertionError(f"{self.robot_name}: NaN detected in body_q")
        if np.any(np.isnan(body_qd)):
            raise AssertionError(f"{self.robot_name}: NaN detected in body_qd")

        # Check that velocities are reasonable
        max_velocity = np.max(np.abs(body_qd))
        if max_velocity > 100.0:
            raise AssertionError(f"{self.robot_name}: Velocity too high: {max_velocity}")


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    parser.add_argument(
        "--robot",
        type=str,
        default="h1",
        choices=list(MENAGERIE_USD_ROBOTS.keys()),
        help="Menagerie USD robot to load",
    )
    parser.add_argument(
        "--num-worlds",
        type=int,
        default=4,
        help="Number of simulated worlds",
    )

    viewer, args = newton.examples.init(parser)

    print(f"[INFO] Loading menagerie USD robot: {args.robot}")
    example = Example(viewer, args.robot, args.num_worlds, args)

    newton.examples.run(example, args)
