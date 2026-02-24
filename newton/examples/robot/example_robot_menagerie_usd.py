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
#   Set NEWTON_ASSETS_PATH to the root of the newton-assets repo.
#   Falls back to newton/tests/assets/menagerie/ for local assets.
#
###########################################################################

import os
from pathlib import Path

import numpy as np
import warp as wp

import newton
import newton.examples

# ---------------------------------------------------------------------------
# Asset resolution
# ---------------------------------------------------------------------------

NEWTON_ASSETS_PATH_ENV = "NEWTON_ASSETS_PATH"
LOCAL_TEST_ASSETS = Path(__file__).parent.parent.parent / "tests" / "assets" / "menagerie"


def _assets_root() -> Path:
    """Return the menagerie USD assets root directory.

    Checks NEWTON_ASSETS_PATH env var first, then falls back to the local
    test assets bundled in the Newton repo.
    """
    env = os.environ.get(NEWTON_ASSETS_PATH_ENV)
    if env:
        return Path(env)
    return LOCAL_TEST_ASSETS


# ---------------------------------------------------------------------------
# Robot configurations
# ---------------------------------------------------------------------------
# Each entry maps a robot key to its USD scene file path (relative to
# the assets root), plus visualization parameters.
# The ``usd_scene`` paths follow the layout in newton/tests/assets/menagerie/.

MENAGERIE_USD_ROBOTS = {
    "apptronik_apollo": {"usd_scene": "apptronik_apollo/apptronik_apollo scene.usda", "initial_height": 1.0},
    "booster_t1": {"usd_scene": "booster_t1/t1 scene.usda", "initial_height": 1.0},
    "g1_with_hands": {"usd_scene": "g1_with_hands/g1_29dof_with_hand_rev_1_0 scene.usda", "initial_height": 0.8},
    "h1": {"usd_scene": "h1/h1 scene.usda", "initial_height": 1.0},
    "robotiq_2f85_v4": {"usd_scene": "robotiq_2f85_v4/2f85 scene.usda", "initial_height": 0.3},
    "shadow_hand": {"usd_scene": "shadow_hand/right_shadow_hand scene.usda", "initial_height": 0.5},
    "wonik_allegro": {"usd_scene": "wonik_allegro/allegro_right.usda", "initial_height": 0.5},
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
        asset_path = _assets_root() / robot_config["usd_scene"]

        if not asset_path.exists():
            raise FileNotFoundError(
                f"Asset not found: {asset_path}\n"
                f"Set {NEWTON_ASSETS_PATH_ENV} to the newton-assets repo root, or ensure "
                f"local test assets exist at {LOCAL_TEST_ASSETS}."
            )

        robot_builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        newton.solvers.SolverMuJoCo.register_custom_attributes(robot_builder)

        robot_builder.default_shape_cfg.mu = 1.0
        robot_builder.default_shape_cfg.mu_torsional = 0.005
        robot_builder.default_shape_cfg.mu_rolling = 0.0001

        initial_height = robot_config["initial_height"]  # [m]
        robot_builder.add_usd(
            str(asset_path),
            xform=wp.transform(wp.vec3(0, 0, initial_height)),
            collapse_fixed_joints=False,
            enable_self_collisions=False,
            hide_collision_shapes=True,
        )

        robot_builder.approximate_meshes("bounding_box")

        builder = newton.ModelBuilder()
        newton.solvers.SolverMuJoCo.register_custom_attributes(builder)
        builder.replicate(robot_builder, self.num_worlds)

        builder.default_shape_cfg.ke = 1.0e3
        builder.default_shape_cfg.kd = 1.0e2
        builder.add_ground_plane()

        self.model = builder.finalize()

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

        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)

        self.collision_pipeline = newton.examples.create_collision_pipeline(self.model, args)
        self.contacts = self.collision_pipeline.contacts()
        self.collision_pipeline.collide(self.state_0, self.contacts)

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
        """Run one frame of simulation (collision + substeps)."""
        self.collision_pipeline.collide(self.state_0, self.contacts)
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
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
        self.viewer.log_contacts(self.contacts, self.state_0)
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
