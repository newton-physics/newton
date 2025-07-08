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
import sys

import warp as wp
from asv_runner.benchmarks.mark import skip_benchmark_if

import newton
import newton.solvers.euler.kernels  # For graph capture on CUDA <12.3
import newton.utils
from newton.examples.example_anymal_c_walk import AnymalController


class AnymalExampleLoad:
    warmup_time = 0
    repeat = 2
    number = 1
    timeout = 600

    def setup(self):
        wp.build.clear_lto_cache()
        wp.build.clear_kernel_cache()

    @skip_benchmark_if(wp.get_cuda_device_count() == 0)
    def time_load(self):
        command = [
            sys.executable,
            "-m",
            "newton.examples.example_anymal_c_walk",
            "--stage_path",
            "None",
            "--num_frames",
            "1",
            "--headless",
        ]

        # Run the script as a subprocess
        result = subprocess.run(command, capture_output=True, text=True, check=True)

        print(f"Output:\n{result.stdout}\n{result.stderr}")


class AnymalPretrainedSimulate:
    repeat = 3
    number = 1

    def setup(self):
        self.num_frames = 50

        builder = newton.ModelBuilder(up_axis=newton.Axis.Y)
        builder.default_joint_cfg = newton.ModelBuilder.JointDofConfig(armature=0.06, limit_ke=1.0e3, limit_kd=1.0e1)
        builder.default_shape_cfg.ke = 5.0e4
        builder.default_shape_cfg.kd = 5.0e2
        builder.default_shape_cfg.kf = 1.0e3
        builder.default_shape_cfg.mu = 0.75

        asset_path = newton.utils.download_asset("anymal_c_simple_description")

        newton.utils.parse_urdf(
            str(asset_path / "urdf" / "anymal.urdf"),
            builder,
            floating=True,
            enable_self_collisions=False,
            collapse_fixed_joints=True,
            ignore_inertial_definitions=False,
        )
        builder.add_ground_plane()

        fps = 60
        self.frame_dt = 1.0e0 / fps

        self.sim_substeps = 10
        self.sim_dt = self.frame_dt / self.sim_substeps

        builder.joint_q[:3] = [0.0, 0.7, 0.0]

        builder.joint_q[7:] = [
            0.03,  # LF_HAA
            0.4,  # LF_HFE
            -0.8,  # LF_KFE
            -0.03,  # RF_HAA
            0.4,  # RF_HFE
            -0.8,  # RF_KFE
            0.03,  # LH_HAA
            -0.4,  # LH_HFE
            0.8,  # LH_KFE
            -0.03,  # RH_HAA
            -0.4,  # RH_HFE
            0.8,  # RH_KFE
        ]

        # finalize model
        self.model = builder.finalize()

        # the policy was trained with the following inertia tensors
        # fmt: off
        self.model.body_inertia = wp.array(
            [
                [[1.30548920,  0.00067627, 0.05068519], [ 0.000676270, 2.74363500,  0.00123380], [0.05068519,  0.00123380, 2.82926230]],
                [[0.01809368,  0.00826303, 0.00475366], [ 0.008263030, 0.01629626, -0.00638789], [0.00475366, -0.00638789, 0.02370901]],
                [[0.18137439, -0.00109795, 0.05645556], [-0.001097950, 0.20255709, -0.00183889], [0.05645556, -0.00183889, 0.02763401]],
                [[0.03070243,  0.00022458, 0.00102368], [ 0.000224580, 0.02828139, -0.00652076], [0.00102368, -0.00652076, 0.00269065]],
                [[0.01809368, -0.00825236, 0.00474725], [-0.008252360, 0.01629626,  0.00638789], [0.00474725,  0.00638789, 0.02370901]],
                [[0.18137439,  0.00111040, 0.05645556], [ 0.001110400, 0.20255709,  0.00183910], [0.05645556,  0.00183910, 0.02763401]],
                [[0.03070243, -0.00022458, 0.00102368], [-0.000224580, 0.02828139,  0.00652076], [0.00102368,  0.00652076, 0.00269065]],
                [[0.01809368, -0.00825236, 0.00474726], [-0.008252360, 0.01629626,  0.00638789], [0.00474726,  0.00638789, 0.02370901]],
                [[0.18137439,  0.00111041, 0.05645556], [ 0.001110410, 0.20255709,  0.00183909], [0.05645556,  0.00183909, 0.02763401]],
                [[0.03070243, -0.00022458, 0.00102368], [-0.000224580, 0.02828139,  0.00652076], [0.00102368,  0.00652076, 0.00269065]],
                [[0.01809368,  0.00826303, 0.00475366], [ 0.008263030, 0.01629626, -0.00638789], [0.00475366, -0.00638789, 0.02370901]],
                [[0.18137439, -0.00109796, 0.05645556], [-0.001097960, 0.20255709, -0.00183888], [0.05645556, -0.00183888, 0.02763401]],
                [[0.03070243,  0.00022458, 0.00102368], [ 0.000224580, 0.02828139, -0.00652076], [0.00102368, -0.00652076, 0.00269065]]
            ],
            dtype=wp.mat33f,
        )
        self.model.body_mass = wp.array([27.99286, 2.51203, 3.27327, 0.55505, 2.51203, 3.27327, 0.55505, 2.51203, 3.27327, 0.55505, 2.51203, 3.27327, 0.55505], dtype=wp.float32,)
        # fmt: on

        self.solver = newton.solvers.FeatherstoneSolver(self.model)

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.model.collide(self.state_0, rigid_contact_margin=0.1)
        newton.sim.eval_fk(self.model, self.state_0.joint_q, self.state_0.joint_qd, self.state_0)

        self.controller = AnymalController(self.model, wp.get_device())
        self.controller.get_control(self.state_0)

        # Initial graph launch, load modules (necessary for drivers prior to CUDA 12.3)
        wp.load_module(newton.solvers.euler.kernels, device=wp.get_device())

        if wp.get_device().is_cuda:
            with wp.ScopedCapture() as capture:
                self.contacts = self.model.collide(self.state_0, rigid_contact_margin=0.1)
                for _ in range(self.sim_substeps):
                    self.state_0.clear_forces()
                    self.controller.assign_control(self.control, self.state_0)
                    self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
                    self.state_0, self.state_1 = self.state_1, self.state_0
            self.graph = capture.graph
        else:
            self.graph = None

    @skip_benchmark_if(wp.get_cuda_device_count() == 0)
    def time_simulate(self):
        for _ in range(self.num_frames):
            wp.capture_launch(self.graph)
            self.controller.get_control(self.state_0)
        wp.synchronize_device()
