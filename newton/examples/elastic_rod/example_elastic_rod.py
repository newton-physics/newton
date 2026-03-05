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
# Example Elastic Rod
#
# A horizontal Cosserat elastic rod fixed at one end, bending under gravity.
# Uses the XPBD rod solver with a direct block-tridiagonal solve.
#
# Command: uv run -m newton.examples elastic_rod
#
###########################################################################

import numpy as np

import newton
import newton.examples
import newton.solvers
from newton.solvers import xpbd_rod


class Example:
    def __init__(self, viewer, args=None):
        self.viewer = viewer
        self.sim_time = 0.0
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = 20
        self.sim_dt = self.frame_dt / self.sim_substeps

        num_points = 40
        spacing = 0.05

        builder = newton.ModelBuilder()

        # Register rod solver attributes before adding rods
        newton.solvers.SolverXPBDRod.register_custom_attributes(builder)

        # Create rod positions along X-axis at height z=1.0
        positions = np.zeros((num_points, 3), dtype=np.float32)
        for i in range(num_points):
            positions[i, 0] = i * spacing
            positions[i, 2] = 1.0

        # Add the elastic rod with moderate stiffness for visible bending
        xpbd_rod.add_elastic_rod(
            builder,
            positions=positions,
            radius=0.005,
            particle_mass=0.05,
            bend_stiffness=0.1,
            twist_stiffness=0.1,
            young_modulus=1.0e4,
            torsion_modulus=1.0e4,
            lock_root=True,
            lock_root_rotation=True,
        )

        self.model = builder.finalize()

        self.solver = newton.solvers.SolverXPBDRod(
            model=self.model,
            linear_damping=0.01,
            angular_damping=0.01,
            solver_backend="block_thomas",
            floor_z=0.0,
        )

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.model.contacts()

        self.viewer.set_model(self.model)

    def simulate(self):
        for _ in range(self.sim_substeps):
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        self.simulate()
        self.sim_time += self.frame_dt

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()

    def test_final(self):
        # Verify the rod tip has descended below its initial height (gravity works)
        # and no NaN values are present
        particle_q = self.state_0.particle_q.numpy()

        # Check for NaN
        assert not np.any(np.isnan(particle_q)), "Particle positions contain NaN"

        # The tip (last particle) should have descended from z=1.0
        tip_z = particle_q[-1, 2]
        assert tip_z < 1.0, f"Rod tip should bend under gravity, but tip z={tip_z:.3f}"

        # The root (first particle) should remain fixed near z=1.0
        root_z = particle_q[0, 2]
        assert abs(root_z - 1.0) < 0.01, f"Root should be fixed at z=1.0, but z={root_z:.3f}"


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    viewer, args = newton.examples.init(parser)
    example = Example(viewer=viewer, args=args)
    newton.examples.run(example, args)
