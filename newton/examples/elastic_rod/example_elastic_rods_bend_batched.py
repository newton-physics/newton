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
# Example Elastic Rods Bend (Batched)
#
# Demonstrates elastic rod bending with batched multi-rod XPBD solver.
# Same 5-rod setup as example_elastic_rods_bend.py but uses the batched
# kernel path for reduced launch overhead and inter-rod parallelism.
#
# Command: uv run -m newton.examples elastic_rods_bend_batched
#
###########################################################################

import numpy as np
import warp as wp

import newton
import newton.examples
import newton.solvers
from newton.solvers import xpbd_rod

from newton.examples.elastic_rod.rod_mesher import BatchedRodMesher


class Example:
    def __init__(self, viewer, args=None):
        self.viewer = viewer
        self.sim_time = 0.0
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = 8
        self.sim_dt = self.frame_dt / self.sim_substeps

        num_points = 64
        spacing = 0.05
        y_separation = 0.3

        self.num_rods = args.num_rods if args is not None and hasattr(args, "num_rods") else 5
        bend_stiffness_values = np.logspace(-3, 1, self.num_rods).tolist()

        builder = newton.ModelBuilder()
        newton.solvers.SolverXPBDRod.register_custom_attributes(builder)

        self.rod_particle_indices: list[list[int]] = []

        for i, bend_stiffness in enumerate(bend_stiffness_values):
            y_pos = (i - (self.num_rods - 1) / 2.0) * y_separation

            positions = np.zeros((num_points, 3), dtype=np.float32)
            for j in range(num_points):
                positions[j, 0] = j * spacing
                positions[j, 1] = y_pos
                positions[j, 2] = 1.0

            particle_indices = xpbd_rod.add_elastic_rod(
                builder,
                positions=positions,
                radius=0.005,
                particle_mass=0.05,
                bend_stiffness=bend_stiffness,
                twist_stiffness=0.1,
                young_modulus=1.0e6,
                torsion_modulus=1.0e6,
                lock_root=True,
                lock_root_rotation=True,
            )

            self.rod_particle_indices.append(particle_indices)

        builder.add_ground_plane()

        self.model = builder.finalize()

        self.solver = newton.solvers.SolverXPBDRod(
            model=self.model,
            linear_damping=0.001,
            angular_damping=0.001,
            solver_backend="block_thomas",
            floor_z=0.0,
        )

        batched = self.solver._batched_ws is not None
        print(f"Batched path active: {batched}")

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.model.contacts()

        # Batched rod mesher — single kernel launch for all rods
        self._mesher = BatchedRodMesher(
            num_rods=self.num_rods,
            num_points=num_points,
            radius=0.01,
            resolution=8,
            smoothing=3,
            particle_offsets=self.solver._rod_particle_starts,
            device=self.model.device,
        )

        # Capture CUDA graph for the substep loop
        self.graph = None
        device = self.model.device
        if device.is_cuda and wp.is_mempool_enabled(device):
            with wp.ScopedCapture(device=device) as capture:
                self._simulate_substeps()
            self.graph = capture.graph
            print("CUDA graph captured for simulation substeps")
        else:
            print("CUDA graph not available, using standard kernel launches")

        self.viewer.set_model(self.model)
        self.viewer.show_particles = True

    def _simulate_substeps(self):
        for _ in range(self.sim_substeps):
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def simulate(self):
        if self.graph is not None:
            wp.capture_launch(self.graph)
        else:
            self._simulate_substeps()

    def step(self):
        self.simulate()
        self.sim_time += self.frame_dt

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)

        self._mesher.update(self.state_0.particle_q)
        for idx in range(self.num_rods):
            self.viewer.log_mesh(
                f"/rod_mesh_{idx}",
                self._mesher.rod_vertices(idx),
                self._mesher.rod_indices(),
                self._mesher.rod_normals(idx),
                self._mesher.rod_uvs(),
            )

        self.viewer.end_frame()

    def test_final(self):
        """Test elastic rod bending with batched solver."""
        particle_q = self.state_0.particle_q.numpy()

        # Check for NaN
        assert not np.any(np.isnan(particle_q)), "Particle positions contain NaN"
        assert np.all(np.isfinite(particle_q)), "Non-finite values in particle positions"

        # All roots should remain fixed near z=1.0
        for i, indices in enumerate(self.rod_particle_indices):
            root_z = particle_q[indices[0], 2]
            assert abs(root_z - 1.0) < 0.01, f"Rod {i} root should be fixed at z=1.0, but z={root_z:.3f}"

        # All tips should descend below initial height
        for i, indices in enumerate(self.rod_particle_indices):
            tip_z = particle_q[indices[-1], 2]
            assert tip_z < 1.0, f"Rod {i} tip should bend under gravity, but tip z={tip_z:.3f}"

        # Stiffer rods should have higher tip z (less droop)
        tip_heights = []
        for indices in self.rod_particle_indices:
            tip_heights.append(particle_q[indices[-1], 2])

        for i in range(len(tip_heights) - 1):
            assert tip_heights[i] <= tip_heights[i + 1] + 0.05, (
                f"Rod {i} tip (z={tip_heights[i]:.3f}) should be lower than or equal to "
                f"rod {i + 1} tip (z={tip_heights[i + 1]:.3f}) since it has lower bend stiffness"
            )


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    parser.add_argument("--num-rods", type=int, default=5, help="Number of rods to simulate.")
    viewer, args = newton.examples.init(parser)
    example = Example(viewer=viewer, args=args)
    newton.examples.run(example, args)
