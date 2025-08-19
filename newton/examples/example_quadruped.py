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
# Example Sim Quadruped
#
# Shows how to set up a simulation of a rigid-body quadruped articulation
# from a URDF using the newton.ModelBuilder().
# Note this example does not include a trained policy.
#
# Users can pick bodies by right-clicking and dragging with the mouse.
#
###########################################################################

import numpy as np
import warp as wp

wp.config.enable_backward = False

import newton
import newton.examples
import newton.utils
from newton.utils import BasicRecorder
from newton.viewer import RecorderImGuiManager


class Example:
    def __init__(self, stage_path="example_quadruped.usd", num_envs=8):
        articulation_builder = newton.ModelBuilder()
        articulation_builder.default_body_armature = 0.01
        articulation_builder.default_joint_cfg.armature = 0.01
        articulation_builder.default_joint_cfg.mode = newton.JointMode.TARGET_POSITION
        articulation_builder.default_joint_cfg.target_ke = 2000.0
        articulation_builder.default_joint_cfg.target_kd = 1.0
        articulation_builder.default_shape_cfg.ke = 1.0e4
        articulation_builder.default_shape_cfg.kd = 1.0e2
        articulation_builder.default_shape_cfg.kf = 1.0e2
        articulation_builder.default_shape_cfg.mu = 1.0
        newton.utils.parse_urdf(
            newton.examples.get_asset("quadruped.urdf"),
            articulation_builder,
            xform=wp.transform([0.0, 0.0, 0.7], wp.quat_identity()),
            floating=True,
            enable_self_collisions=False,
        )
        articulation_builder.joint_q[-12:] = [0.2, 0.4, -0.6, -0.2, -0.4, 0.6, -0.2, 0.4, -0.6, 0.2, -0.4, 0.6]
        articulation_builder.joint_target[-12:] = articulation_builder.joint_q[-12:]

        builder = newton.ModelBuilder()

        self.sim_time = 0.0
        fps = 100
        self.frame_dt = 1.0 / fps

        self.sim_substeps = 10
        self.sim_dt = self.frame_dt / self.sim_substeps

        self.num_envs = num_envs

        offsets = newton.examples.compute_env_offsets(self.num_envs)
        for i in range(self.num_envs):
            builder.add_builder(articulation_builder, xform=wp.transform(offsets[i], wp.quat_identity()))

        builder.add_ground_plane()

        np.set_printoptions(suppress=True)
        # finalize model
        self.model = builder.finalize()

        self.solver = newton.solvers.SolverXPBD(self.model)
        # self.solver = newton.solvers.SolverFeatherstone(self.model)
        # self.solver = newton.solvers.SolverSemiImplicit(self.model)
        # self.solver = newton.solvers.SolverMuJoCo(self.model)

        if stage_path:
            self.renderer = newton.viewer.RendererOpenGL(self.model, path=stage_path)
            self.recorder = BasicRecorder()
            self.gui = RecorderImGuiManager(self.renderer, self.recorder, self)
            self.renderer.render_2d_callbacks.append(self.gui.render_frame)
        else:
            self.renderer = None
            self.recorder = None
            self.gui = None

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.model.collide(self.state_0)

        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)

        # simulate() allocates memory via a clone, so we can't use graph capture if the device does not support mempools
        self.use_cuda_graph = wp.get_device().is_cuda and wp.is_mempool_enabled(wp.get_device())
        if self.use_cuda_graph:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph
        else:
            self.graph = None

    @property
    def paused(self):
        if self.renderer:
            return self.renderer.paused
        return False

    @paused.setter
    def paused(self, value):
        if self.renderer:
            if self.renderer.paused == value:
                return
            self.renderer.paused = value
            if self.gui:
                self.gui._clear_contact_points()

    def simulate(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            if self.renderer and hasattr(self.renderer, "apply_picking_force"):
                self.renderer.apply_picking_force(self.state_0)
            self.contacts = self.model.collide(self.state_0)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        if self.paused:
            return

        with wp.ScopedTimer("step"):
            if self.use_cuda_graph:
                wp.capture_launch(self.graph)
            else:
                self.simulate()
        self.sim_time += self.frame_dt

        if self.recorder:
            if self.renderer:
                self.renderer.compute_contact_rendering_points(self.state_0.body_q, self.contacts)
                contact_points = [self.renderer.contact_points0, self.renderer.contact_points1]
                self.recorder.record(self.state_0.body_q, contact_points)
            else:
                self.recorder.record(self.state_0.body_q)

    def render(self):
        if self.renderer is None:
            return

        with wp.ScopedTimer("render"):
            self.renderer.begin_frame(self.sim_time)
            if not self.paused:
                self.renderer.render(self.state_0)
                self.renderer.render_computed_contacts(contact_point_radius=1e-2)
            else:
                # in paused mode, the GUI will handle rendering from the recorder
                pass
            self.renderer.end_frame()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--device", type=str, default=None, help="Override the default Warp device.")
    parser.add_argument(
        "--stage-path",
        type=lambda x: None if x == "None" else str(x),
        default="example_quadruped.usd",
        help="Path to the output USD file.",
    )
    parser.add_argument("--num-frames", type=int, default=30000, help="Total number of frames.")
    parser.add_argument("--num-envs", type=int, default=100, help="Total number of simulated environments.")

    args = parser.parse_known_args()[0]

    with wp.ScopedDevice(args.device):
        example = Example(stage_path=args.stage_path, num_envs=args.num_envs)

        if example.renderer:
            while example.renderer.is_running():
                example.step()
                example.render()
        else:
            for _ in range(args.num_frames):
                example.step()
                example.render()

        # if example.renderer:
        #     example.renderer.save()
