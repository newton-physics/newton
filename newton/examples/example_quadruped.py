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
###########################################################################


import numpy as np
import warp as wp
from warp.render.imgui_manager import ImGuiManager

import newton
import newton.examples
import newton.sim
import newton.utils
from newton.utils.recorder import BodyTransformRecorder


class RecorderImGuiManager(ImGuiManager):
    """An ImGui manager for controlling simulation playback with a recorder."""

    def __init__(self, renderer, recorder, example, window_pos=(10, 10), window_size=(300, 120)):
        super().__init__(renderer)
        if not self.is_available:
            return

        self.window_pos = window_pos
        self.window_size = window_size
        self.recorder = recorder
        self.example = example
        self.selected_frame = 0

    def _update_frame(self, frame_id):
        """Update the selected frame and renderer transforms if paused."""
        self.selected_frame = frame_id
        if self.example.paused:
            transforms = self.recorder.playback(self.selected_frame)
            if transforms:
                self.renderer.update_body_transforms(transforms)

    def draw_ui(self):
        self.imgui.set_next_window_size(self.window_size[0], self.window_size[1], self.imgui.ONCE)
        self.imgui.set_next_window_position(self.window_pos[0], self.window_pos[1], self.imgui.ONCE)

        self.imgui.begin("Recorder Controls")

        # Start/Stop button
        if self.example.paused:
            if self.imgui.button("Resume"):
                self.example.paused = False
        else:
            if self.imgui.button("Pause"):
                self.example.paused = True

        self.imgui.same_line()
        # total frames
        total_frames = len(self.recorder.transforms_history)
        frame_time = self.selected_frame * self.example.frame_dt
        self.imgui.text(
            f"Frame: {self.selected_frame}/{total_frames - 1 if total_frames > 0 else 0} ({frame_time:.2f}s)"
        )

        # Frame slider
        if total_frames > 0:
            changed, self.selected_frame = self.imgui.slider_int("Timeline", self.selected_frame, 0, total_frames - 1)
            if changed and self.example.paused:
                self._update_frame(self.selected_frame)

            # Back/Forward buttons
            if self.imgui.button(" < "):
                self._update_frame(max(0, self.selected_frame - 1))

            self.imgui.same_line()

            if self.imgui.button(" > "):
                self._update_frame(min(total_frames - 1, self.selected_frame + 1))

        self.imgui.separator()

        if self.imgui.button("Save"):
            file_path = self.open_save_file_dialog(
                defaultextension=".npz",
                filetypes=[("Numpy Archives", "*.npz"), ("All files", "*.*")],
                title="Save Recording",
            )
            if file_path:
                self.recorder.save_to_file(file_path)

        self.imgui.same_line()

        if self.imgui.button("Load"):
            file_path = self.open_load_file_dialog(
                filetypes=[("Numpy Archives", "*.npz"), ("All files", "*.*")],
                title="Load Recording",
            )
            if file_path:
                self.recorder.load_from_file(file_path, device=wp.get_device())
                # When loading, pause the simulation and go to the first frame
                self.example.paused = True
                self.selected_frame = 0
                if len(self.recorder.transforms_history) > 0:
                    transforms = self.recorder.playback(self.selected_frame)
                    if transforms:
                        self.renderer.update_body_transforms(transforms)

        self.imgui.end()


class Example:
    def __init__(self, stage_path="example_quadruped.usd", num_envs=8):
        articulation_builder = newton.ModelBuilder()
        articulation_builder.default_body_armature = 0.01
        articulation_builder.default_joint_cfg.armature = 0.01
        articulation_builder.default_joint_cfg.mode = newton.JOINT_MODE_TARGET_POSITION
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

        self.solver = newton.solvers.XPBDSolver(self.model)
        # self.solver = newton.solvers.FeatherstoneSolver(self.model)
        # self.solver = newton.solvers.SemiImplicitSolver(self.model)
        # self.solver = newton.solvers.MuJoCoSolver(self.model)

        if stage_path:
            self.renderer = newton.utils.SimRendererOpenGL(path=stage_path)
            body_names = self.renderer.populate_bodies(self.model.body_key)

            geo_shape = {}
            self.renderer.populate_shapes(
                body_names,
                geo_shape,
                self.model.shape_body.numpy(),
                self.model.shape_geo_src,
                self.model.shape_geo.type.numpy(),
                self.model.shape_geo.scale.numpy(),
                self.model.shape_geo.thickness.numpy(),
                self.model.shape_geo.is_solid.numpy(),
                self.model.shape_transform.numpy(),
                self.model.shape_flags.numpy(),
                self.model.shape_key,
            )

            self.recorder = BodyTransformRecorder()
            self.gui = RecorderImGuiManager(self.renderer, self.recorder, self)
            self.renderer.render_2d_callbacks.append(self.gui.render_frame)
            self.paused = False
        else:
            self.renderer = None
            self.recorder = None
            self.gui = None
            self.paused = False

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.model.collide(self.state_0)

        newton.sim.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)

        # simulate() allocates memory via a clone, so we can't use graph capture if the device does not support mempools
        self.use_cuda_graph = wp.get_device().is_cuda and wp.is_mempool_enabled(wp.get_device())
        if self.use_cuda_graph:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph
        else:
            self.graph = None

    def simulate(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
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
            self.recorder.record(self.state_0.body_q)

    def render(self):
        if self.renderer is None:
            return

        with wp.ScopedTimer("render"):
            self.renderer.begin_frame(self.sim_time)
            if not self.paused:
                self.renderer.update_body_transforms(self.state_0.body_q)
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
