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

import subprocess

import warp as wp
from asv_runner.benchmarks.mark import skip_benchmark_if

import newton
import newton.examples
import newton.utils
from newton.examples.example_cartpole import Example


class CartpoleMemory:
    params = [128, 256]

    def setup(self, num_envs):
        wp.init()

    def peakmem_initialize_model(self, num_envs):
        with wp.ScopedDevice("cpu"):
            _example = Example(stage_path=None, num_envs=num_envs)


class CartpoleModel:
    params = [64, 128]

    number = 10

    def setup(self, num_envs):
        wp.init()

    def time_initialize_model(self, num_envs):
        with wp.ScopedDevice("cpu"):
            _example = Example(stage_path=None, num_envs=num_envs)


class CartpoleExampleLoad:
    warmup_time = 0
    repeat = 2
    number = 1
    timeout = 600

    def setup(self):
        wp.build.clear_lto_cache()
        wp.build.clear_kernel_cache()

    @skip_benchmark_if(wp.get_cuda_device_count() == 0)
    def time_load(self):
        """Time the amount of time it takes to load and run one frame of the Cartpole example."""

        command = [
            sys.executable,
            "-m",
            "newton.examples.example_cartpole",
            "--stage_path",
            "None",
            "--num_frames",
            "1",
            "--use_cuda_graph",
            "False",
        ]

        # Run the script as a subprocess
        result = subprocess.run(command, capture_output=True, text=True, check=True)

        print(f"Output:\n{result.stdout}\n{result.stderr}")


class CartpoleMuJoCoSolver:
    repeat = 10
    number = 1

    def setup(self):
        self.num_frames = 200

        num_envs = 8
        articulation_builder = newton.ModelBuilder()
        articulation_builder.default_shape_cfg.density = 100.0
        articulation_builder.default_joint_cfg.armature = 0.1
        articulation_builder.default_body_armature = 0.1

        newton.utils.parse_urdf(
            newton.examples.get_asset("cartpole.urdf"),
            articulation_builder,
            floating=False,
            enable_self_collisions=False,
            collapse_fixed_joints=True,
        )

        builder = newton.ModelBuilder()

        fps = 60
        self.frame_dt = 1.0 / fps
        self.sim_substeps = 10
        self.sim_dt = self.frame_dt / self.sim_substeps

        positions = newton.examples.compute_env_offsets(num_envs, env_offset=(1.0, 2.0, 0.0))

        for i in range(num_envs):
            builder.add_builder(articulation_builder, xform=wp.transform(positions[i], wp.quat_identity()))

            # joint initial positions
            builder.joint_q[-3:] = [0.0, 0.3, 0.0]

        # finalize model
        self.model = builder.finalize()

        self.solver = newton.solvers.MuJoCoSolver(self.model)

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()

        newton.sim.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)

        if wp.get_device().is_cuda:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph
        else:
            self.graph = None

    def simulate(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.solver.step(self.state_0, self.state_1, self.control, None, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    @skip_benchmark_if(wp.get_cuda_device_count() == 0 or wp.context.runtime.driver_version < (12, 3))
    def time_simulate(self):
        for _ in range(self.num_frames):
            wp.capture_launch(self.graph)
        wp.synchronize_device()
