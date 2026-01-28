# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
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
# Example Softbody Dropping to Cloth
#
# This simulation demonstrates a volumetric soft body (pyramid tet mesh)
# dropping onto a cloth sheet. The mesh data mirrors `newton.tests.test_softbody.py`
# to keep the example self-contained.
#
# Command: python -m newton.examples.mutlphysics.example_softbody_dropping_to_cloth
#
###########################################################################

import numpy as np
import warp as wp

import newton
import newton.examples

# Pyramid tet mesh data (2x2x1 grid of particles forming 20 tets)
PYRAMID_TET_INDICES = np.array(
    [
        [0, 1, 3, 9],
        [1, 4, 3, 13],
        [1, 3, 9, 13],
        [3, 9, 13, 12],
        [1, 9, 10, 13],
        [1, 2, 4, 10],
        [2, 5, 4, 14],
        [2, 4, 10, 14],
        [4, 10, 14, 13],
        [2, 10, 11, 14],
        [3, 4, 6, 12],
        [4, 7, 6, 16],
        [4, 6, 12, 16],
        [6, 12, 16, 15],
        [4, 12, 13, 16],
        [4, 5, 7, 13],
        [5, 8, 7, 17],
        [5, 7, 13, 17],
        [7, 13, 17, 16],
        [5, 13, 14, 17],
    ],
    dtype=np.int32,
)

PYRAMID_PARTICLES = [
    (0.0, 0.0, 0.0),  # 0
    (1.0, 0.0, 0.0),  # 1
    (2.0, 0.0, 0.0),  # 2
    (0.0, 1.0, 0.0),  # 3
    (1.0, 1.0, 0.0),  # 4
    (2.0, 1.0, 0.0),  # 5
    (0.0, 2.0, 0.0),  # 6
    (1.0, 2.0, 0.0),  # 7
    (2.0, 2.0, 0.0),  # 8
    (0.0, 0.0, 1.0),  # 9
    (1.0, 0.0, 1.0),  # 10
    (2.0, 0.0, 1.0),  # 11
    (0.0, 1.0, 1.0),  # 12
    (1.0, 1.0, 1.0),  # 13
    (2.0, 1.0, 1.0),  # 14
    (0.0, 2.0, 1.0),  # 15
    (1.0, 2.0, 1.0),  # 16
    (2.0, 2.0, 1.0),  # 17
]


class Example:
    def __init__(self, viewer, args=None, solver_type: str = "vbd"):
        self.viewer = viewer
        self.solver_type = solver_type
        self.sim_time = 0.0
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = 10
        self.iterations = 10
        self.sim_dt = self.frame_dt / self.sim_substeps

        if self.solver_type != "vbd":
            raise ValueError("The softbody dropping to cloth example only supports the VBD solver.")

        builder = newton.ModelBuilder()
        builder.add_ground_plane()

        # Add soft body (pyramid tet mesh) at elevated position
        builder.add_soft_mesh(
            pos=(0.0, 0.0, 2.0),
            rot=wp.quat_identity(),
            scale=0.2,
            vel=(0.0, 0.0, 0.0),
            vertices=PYRAMID_PARTICLES,
            indices=PYRAMID_TET_INDICES.flatten().tolist(),
            density=1.0e3,
            k_mu=1.0e5,
            k_lambda=1.0e5,
            k_damp=1e-5,
        )

        # Add cloth grid below the soft body
        builder.add_cloth_grid(
            pos=(-1.0, -1.0, 1.0),
            rot=wp.quat_identity(),
            vel=(0.0, 0.0, 0.0),
            fix_left=True,
            fix_right=True,
            dim_x=40,
            dim_y=40,
            cell_x=0.05,
            cell_y=0.05,
            mass=0.0005,
            tri_ke=1e5,
            tri_ka=1e5,
            tri_kd=1e-5,
            edge_ke=0.01,
            edge_kd=1e-2,
            particle_radius=0.05,
        )

        # Color the mesh for VBD solver
        builder.color()

        self.model = builder.finalize()

        # Contact parameters
        self.model.soft_contact_ke = 1.0e5
        self.model.soft_contact_kd = 1e-5
        self.model.soft_contact_mu = 1.0

        self.solver = newton.solvers.SolverVBD(
            model=self.model,
            iterations=self.iterations,
            particle_enable_self_contact=True,
            particle_self_contact_radius=0.01,
            particle_self_contact_margin=0.02,
            particle_enable_tile_solve=True,
        )

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()

        # Create collision pipeline (default: unified)
        self.collision_pipeline = newton.examples.create_collision_pipeline(self.model, args)
        self.contacts = self.model.collide(self.state_0, collision_pipeline=self.collision_pipeline)

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

            # apply forces to the model
            self.viewer.apply_forces(self.state_0)

            self.contacts = self.model.collide(self.state_0, collision_pipeline=self.collision_pipeline)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)

            # swap states
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()

        self.sim_time += self.frame_dt

    def test_final(self):
        # Test that particles are in a reasonable range
        # Soft body should have dropped and settled on the cloth
        p_lower = wp.vec3(-3.0, -3.0, -0.5)
        p_upper = wp.vec3(3.0, 3.0, 3.0)
        newton.examples.test_particle_state(
            self.state_0,
            "particles are within a reasonable volume",
            lambda q, qd: newton.utils.vec_inside_limits(q, p_lower, p_upper),
        )

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()


if __name__ == "__main__":
    # Create parser with base arguments
    parser = newton.examples.create_parser()

    # Add solver-specific arguments
    parser.add_argument(
        "--solver",
        help="Type of solver (only 'vbd' supports volumetric soft bodies in this example)",
        type=str,
        choices=["vbd"],
        default="vbd",
    )

    # Parse arguments and initialize viewer
    viewer, args = newton.examples.init(parser)

    # Create example and run
    example = Example(
        viewer=viewer,
        args=args,
        solver_type=args.solver,
    )

    newton.examples.run(example, args)
