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
# Parameterized example for visualizing MuJoCo Menagerie robot assets
# that have been converted to USD format, simulated with SolverMuJoCo.
#
# Usage:
#   uv run -m newton.examples robot_menagerie_usd --robot h1
#   uv run -m newton.examples robot_menagerie_usd --robot apptronik_apollo
#   uv run -m newton.examples robot_menagerie_usd --robot shadow_hand
#
# Available robots:
#   apptronik_apollo, booster_t1, g1_with_hands, h1,
#   robotiq_2f85_v4, shadow_hand, wonik_allegro
#
# Asset resolution:
#   Assets are downloaded automatically from the newton-assets GitHub repo.
#
###########################################################################

import numpy as np
import warp as wp

import newton
import newton.examples
import newton.utils
from newton._src.usd.schemas import SchemaResolverMjc, SchemaResolverNewton

# ---------------------------------------------------------------------------
# Robot configurations
# ---------------------------------------------------------------------------
# Each entry maps a robot key to its newton-assets folder + scene file,
# plus visualization parameters. Assets are downloaded via download_asset().

MENAGERIE_USD_ROBOTS = {
    "apptronik_apollo": {"asset_folder": "apptronik_apollo", "scene_file": "usd_structured/apptronik_apollo.usda", "initial_height": 1.0},
    "booster_t1": {"asset_folder": "booster_t1", "scene_file": "usd_structured/T1.usda", "initial_height": 1.0},
    "g1_with_hands": {"asset_folder": "unitree_g1", "scene_file": "usd_structured/g1_29dof_with_hand_rev_1_0.usda", "initial_height": 0.8},
    "h1": {"asset_folder": "unitree_h1", "scene_file": "usd_structured/h1.usda", "initial_height": 1.0},
    "robotiq_2f85_v4": {"asset_folder": "robotiq_2f85", "scene_file": "usd_structured/robotiq_2f85.usda", "initial_height": 0.3},
    "shadow_hand": {"asset_folder": "shadow_hand", "scene_file": "usd_structured/left_shadow_hand.usda", "initial_height": 0.5},
    "wonik_allegro": {"asset_folder": "wonik_allegro", "scene_file": "usd_structured/allegro_left.usda", "initial_height": 0.5},
}


class Example:
    """Menagerie USD simulation example."""

    def __init__(self, viewer, robot_name: str, num_worlds: int = 4, args=None):
        """Initialize the example.

        Args:
            viewer: Viewer instance for rendering.
            robot_name: Menagerie key to load.
            num_worlds: Number of simulated worlds [count].
            args: Parsed CLI args.
        """
        self.fps = 50  # [Hz]
        self.frame_dt = 1.0 / self.fps  # [s]

        self.sim_time = 0.0  # [s]
        self.sim_substeps = 4
        self.sim_dt = self.frame_dt / self.sim_substeps  # [s]

        self.num_worlds = num_worlds
        self.robot_name = robot_name

        self.viewer = viewer

        if robot_name not in MENAGERIE_USD_ROBOTS:
            raise ValueError(f"Unknown robot: {robot_name}. Available: {list(MENAGERIE_USD_ROBOTS.keys())}")

        robot_config = MENAGERIE_USD_ROBOTS[robot_name]
        asset_root = newton.utils.download_asset(robot_config["asset_folder"])
        asset_path = asset_root / robot_config["scene_file"]

        robot_builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        newton.solvers.SolverMuJoCo.register_custom_attributes(robot_builder)

        initial_height = robot_config["initial_height"]  # [m]
        robot_builder.add_usd(
            str(asset_path),
            xform=wp.transform(wp.vec3(0, 0, initial_height)),
            collapse_fixed_joints=False,
            enable_self_collisions=False,
            hide_collision_shapes=True,
            schema_resolvers=[SchemaResolverMjc(), SchemaResolverNewton()],
        )

        robot_builder.approximate_meshes("bounding_box")

        builder = newton.ModelBuilder()
        newton.solvers.SolverMuJoCo.register_custom_attributes(builder)
        builder.replicate(robot_builder, self.num_worlds)

        builder.add_ground_plane()

        self.model = builder.finalize()

        self.solver = newton.solvers.SolverMuJoCo(
            self.model,
            iterations=100,
            ls_iterations=50,
            njmax=300,
            nconmax=150,
            use_mujoco_contacts=True,
        )

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()

        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)

        self.viewer.set_model(self.model)
        self.viewer.set_world_offsets((3.0, 3.0, 0.0))

        self.capture()

    def capture(self):
        """Capture the simulation loop into a CUDA graph if running on GPU."""
        self.graph = None
        if wp.get_device().is_cuda:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph

    def simulate(self):
        """Run one frame of simulation (substeps with MuJoCo contacts)."""
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            self.solver.step(self.state_0, self.state_1, self.control, None, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        """Advance the simulation by one frame [s]."""
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()

        self.sim_time += self.frame_dt

    def render(self):
        """Render the current simulation state to the viewer."""
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()

    def test_final(self):
        """Validate simulation output after the run completes."""
        body_q = self.state_0.body_q.numpy()
        body_qd = self.state_0.body_qd.numpy()

        newton.examples.test_body_state(
            self.model,
            self.state_0,
            "all bodies are above the ground",
            lambda q, qd: q[2] > -0.1,
        )

        if np.any(np.isnan(body_q)):
            raise AssertionError(f"{self.robot_name}: NaN detected in body_q")
        if np.any(np.isnan(body_qd)):
            raise AssertionError(f"{self.robot_name}: NaN detected in body_qd")

        max_velocity = np.max(np.abs(body_qd))
        if max_velocity > 100.0:  # [m/s]
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
