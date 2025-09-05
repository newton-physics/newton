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
# Example Robot Allegro Hand
#
# Shows how to set up a simulation of a Allegro Hand articulation
# from a Mujoco file using newton.ModelBuilder.add_mjcf().
#
# Command: python -m newton.examples robot_allegro_hand --num-envs 16
#
###########################################################################

import warp as wp
import newton

import newton.examples

class Example:
    def __init__(self, viewer, num_envs=4):
        self.fps = 50
        self.frame_dt = 1.0 / self.fps

        self.sim_time = 0.0
        self.sim_substeps = 4
        self.sim_dt = self.frame_dt / self.sim_substeps

        self.num_envs = num_envs

        self.viewer = viewer

        self.device = wp.get_device()

        env = newton.ModelBuilder()
        env.default_joint_cfg = newton.ModelBuilder.JointDofConfig(limit_ke=1.0e3, limit_kd=1.0e1, friction=1e-5)
        env.default_shape_cfg.ke = 5.0e4
        env.default_shape_cfg.kd = 5.0e2
        env.default_shape_cfg.kf = 1.0e3
        env.default_shape_cfg.mu = 0.75

        fixed_box = env.add_body()
        env.add_shape_box(fixed_box, hx=0.5, hy=0.35, hz=0.25)
        env.add_joint_fixed(-1, fixed_box, parent_xform=wp.transform(p=wp.vec3(0.0, 2.0, 0), q=wp.quat_identity()))

        falling_box = env.add_body(xform=wp.transform(p=wp.vec3(0.0, 2.0, 1), q=wp.quat_identity()))
        env.add_shape_box(falling_box, hx=0.5, hy=0.35, hz=0.25)
        env.add_joint_free(falling_box)

        world = newton.ModelBuilder()
        world.replicate(env, self.num_envs, spacing=(3, 3, 0))

        self.model = world.finalize()

        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.model)



        self.solver = newton.solvers.SolverMuJoCo(
            self.model,
            use_mujoco_cpu=False,
            solver="newton",
            integrator="euler",
            # njmax=200,
            ncon_per_env=150,
            cone="elliptic",
            impratio=100,
            iterations=100,
            ls_iterations=50,
            save_to_mjcf="converted.xml",
            separate_envs_to_worlds=False,
        )
        # self.solver = newton.solvers.SolverXPBD(
        #     self.model,
        # )
        # with open(asset_file, "r") as f:
        #     spec = mujoco.MjSpec.from_string(f.read())
            
        # compare spec with solver.spec
        # print(spec.option.solver)
        # print(self.solver.spec.option.solver)
        # print(spec.option.integrator)
        # print(self.solver.spec.option.integrator)
        # print(spec.option.iterations)
        # print(self.solver.spec.option.iterations)
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.model.collide(self.state_0)

        self.viewer.set_model(self.model)

        self.capture()

    def capture(self):
        self.graph = None
        # if wp.get_device().is_cuda:
        #     with wp.ScopedCapture() as capture:
        #         self.simulate()
        #     self.graph = capture.graph

    def simulate(self):
        self.contacts = self.model.collide(self.state_0)
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()

            # apply forces to the model for picking, wind, etc
            self.viewer.apply_forces(self.state_0)

            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)

            # swap states
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

        self.solver.render_mujoco_viewer()

    def test(self):
        pass


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    parser.add_argument("--num-envs", type=int, default=2, help="Total number of simulated environments.")

    viewer, args = newton.examples.init(parser)

    viewer._paused = True
    example = Example(viewer, args.num_envs)

    newton.examples.run(example)
