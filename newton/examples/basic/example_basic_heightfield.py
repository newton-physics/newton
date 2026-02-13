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
# Example Basic Heightfield
#
# Demonstrates heightfield terrain with objects dropped onto it.
# Supports both Newton's native CollisionPipeline and MuJoCo solver.
#
# Command: uv run -m newton.examples basic_heightfield
# MuJoCo: uv run -m newton.examples basic_heightfield --solver mujoco
#
###########################################################################

import numpy as np
import warp as wp

import newton
import newton.examples


class Example:
    def __init__(self, viewer, args):
        self.fps = 100
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = 10
        self.sim_dt = self.frame_dt / self.sim_substeps

        self.viewer = viewer
        self.solver_type = args.solver if hasattr(args, "solver") and args.solver else "xpbd"

        builder = newton.ModelBuilder()

        # Create a wave-like heightfield terrain
        nrow, ncol = 50, 50
        hx, hy = 5.0, 5.0
        x = np.linspace(-hx, hx, ncol)
        y = np.linspace(-hy, hy, nrow)
        xx, yy = np.meshgrid(x, y)
        elevation = np.sin(xx * 1.0) * np.cos(yy * 1.0) * 0.5

        hfield = newton.Heightfield(
            data=elevation,
            nrow=nrow,
            ncol=ncol,
            hx=hx,
            hy=hy,
        )
        builder.add_shape_heightfield(heightfield=hfield)

        # Drop several spheres onto the terrain
        drop_z = 3.0
        self.sphere_bodies = []
        positions = [
            (-2.0, -2.0),
            (0.0, 0.0),
            (2.0, 2.0),
            (-1.0, 1.5),
            (1.5, -1.0),
        ]
        for x_pos, y_pos in positions:
            body = builder.add_body(
                xform=wp.transform(p=wp.vec3(x_pos, y_pos, drop_z), q=wp.quat_identity()),
            )
            builder.add_shape_sphere(body=body, radius=0.3)
            self.sphere_bodies.append(body)

        self.model = builder.finalize()

        if self.solver_type == "mujoco":
            self.solver = newton.solvers.SolverMuJoCo(self.model)
            self.collision_pipeline = None
        else:
            self.solver = newton.solvers.SolverXPBD(self.model, iterations=10)
            self.collision_pipeline = newton.examples.create_collision_pipeline(self.model, args)

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()

        if self.collision_pipeline is not None:
            self.contacts = self.model.collide(self.state_0, collision_pipeline=self.collision_pipeline)
        else:
            self.contacts = None

        self.viewer.set_model(self.model)

        self.capture()

    def capture(self):
        if wp.get_device().is_cuda:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph
        else:
            self.graph = None

    def simulate(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            if self.collision_pipeline is not None:
                self.contacts = self.model.collide(self.state_0, collision_pipeline=self.collision_pipeline)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
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
        self.viewer.end_frame()

    def test_final(self):
        """Verify all spheres are resting on the heightfield (not fallen through)."""
        body_q = self.state_0.body_q.numpy()
        for body_idx in self.sphere_bodies:
            z = float(body_q[body_idx, 2])
            assert z > -1.0, f"Sphere body {body_idx} fell through heightfield: z={z:.4f}"


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    parser.add_argument(
        "--solver",
        type=str,
        default="xpbd",
        choices=["xpbd", "mujoco"],
        help="Solver type: xpbd (default, native collision) or mujoco",
    )
    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args)
    newton.examples.run(example, args)
