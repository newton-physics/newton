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

import numpy as np
import warp as wp
from asv_runner.benchmarks.mark import skip_benchmark_if

import newton

wp.config.quiet = True


def create_flat_terrain_mesh(grid_resolution=100, size=20.0):
    """Create a flat terrain mesh with specified grid resolution."""
    # Create grid of vertices
    x = np.linspace(-size / 2, size / 2, grid_resolution)
    y = np.linspace(-size / 2, size / 2, grid_resolution)
    xx, yy = np.meshgrid(x, y)

    # Flat terrain at z=0
    vertices = np.stack([xx.flatten(), yy.flatten(), np.zeros_like(xx.flatten())], axis=1).astype(np.float32)

    # Create triangle indices
    indices = []
    for i in range(grid_resolution - 1):
        for j in range(grid_resolution - 1):
            # Two triangles per grid cell
            v0 = i * grid_resolution + j
            v1 = v0 + 1
            v2 = v0 + grid_resolution
            v3 = v2 + 1

            indices.extend([v0, v2, v1, v1, v2, v3])

    indices = np.array(indices, dtype=np.int32)
    return newton.Mesh(vertices, indices)


class UnifiedCollisionPipelineBenchmark:
    """Benchmark for unified collision pipeline with falling shapes on terrain."""

    repeat = 5
    number = 1
    params = [[100]]  # Grid resolution parameter
    param_names = ["grid_resolution"]

    def setup(self, grid_resolution):
        self.num_frames = 50
        self.fps = 100
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = 10
        self.sim_dt = self.frame_dt / self.sim_substeps

        # Build model
        builder = newton.ModelBuilder()

        # Add flat terrain mesh
        terrain_mesh = create_flat_terrain_mesh(grid_resolution=grid_resolution, size=20.0)
        builder.add_shape_mesh(
            body=-1,  # Static body
            mesh=terrain_mesh,
            xform=wp.transform_identity(),
        )

        # Add pyramid of boxes (base_size x base_size, decreasing to 1x1 at top)
        base_size = 10
        box_size = 0.5
        spacing = box_size * 1.01  # Slight gap between boxes

        for layer in range(base_size):
            layer_size = base_size - layer
            layer_height = layer * box_size

            # Center offset for this layer
            offset_x = -layer_size * spacing / 2.0 + spacing / 2.0
            offset_y = -layer_size * spacing / 2.0 + spacing / 2.0

            for ix in range(layer_size):
                for it in range(layer_size):
                    pos = wp.vec3(
                        offset_x + ix * spacing,
                        offset_y + it * spacing,
                        layer_height + box_size / 2.0,
                    )

                    body = builder.add_body(xform=wp.transform(p=pos, q=wp.quat_identity()))
                    builder.add_shape_box(body, hx=box_size / 2, hy=box_size / 2, hz=box_size / 2)
                    builder.add_joint_free(body)

        self.model = builder.finalize()

        # Create collision pipeline with SAP broad phase
        self.collision_pipeline = newton.CollisionPipelineUnified.from_model(
            self.model,
            rigid_contact_max_per_pair=10,
            rigid_contact_margin=0.01,
            broad_phase_mode=newton.BroadPhaseMode.SAP,
        )

        # Create XPBD solver
        self.solver = newton.solvers.SolverXPBD(
            self.model,
            iterations=2,
            rigid_contact_relaxation=0.8,
            angular_damping=0.0,
        )

        # Initialize states
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()

        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)

        # Capture one frame of simulation into a graph
        wp.capture_begin()

        # Compute contacts once per timestep
        contacts = self.model.collide(self.state_0, collision_pipeline=self.collision_pipeline)

        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.solver.step(self.state_0, self.state_1, self.control, contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

        self.graph = wp.capture_end()

    @skip_benchmark_if(wp.get_cuda_device_count() == 0)
    def time_simulate(self, grid_resolution):
        """Benchmark simulation with collision detection using graph capture."""
        for _ in range(self.num_frames):
            wp.capture_launch(self.graph)

        wp.synchronize_device()


if __name__ == "__main__":
    import argparse

    from newton.utils import run_benchmark

    benchmark_list = {
        "UnifiedCollisionPipelineBenchmark": UnifiedCollisionPipelineBenchmark,
    }

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "-b", "--bench", default=None, action="append", choices=benchmark_list.keys(), help="Run a single benchmark."
    )
    args = parser.parse_known_args()[0]

    if args.bench is None:
        benchmarks = benchmark_list.keys()
    else:
        benchmarks = args.bench

    for key in benchmarks:
        benchmark = benchmark_list[key]
        run_benchmark(benchmark)
