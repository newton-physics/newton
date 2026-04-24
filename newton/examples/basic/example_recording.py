# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Recording
#
# Shows how to record a simulation using ViewerFile for automatic recording
# of model structure and state data during simulation.
#
# Recording happens automatically - ViewerFile captures all logged states
# and saves them when the viewer is closed.
#
# Command: python -m newton.examples recording
#
###########################################################################

import numpy as np
import warp as wp

import newton
import newton.examples


class Example:
    def __init__(self, viewer, _args):
        # Setup simulation parameters
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = 10
        self.sim_dt = self.frame_dt / self.sim_substeps

        self.viewer = viewer
        self.world_count = 100

        # Set numpy random seed for reproducibility
        self.seed = 123
        self.rng = np.random.default_rng(self.seed)

        start_rot = wp.quat_from_axis_angle(wp.normalize(wp.vec3(*self.rng.uniform(-1.0, 1.0, size=3))), -wp.pi * 0.5)

        mjcf_filename = newton.examples.get_asset("nv_humanoid.xml")

        articulation_builder = newton.ModelBuilder()
        articulation_builder.add_mjcf(
            mjcf_filename,
            ignore_names=["floor", "ground"],
            up_axis="Z",
        )

        root_pose = wp.transform((0.0, 0.0, 1.5), start_rot)
        articulation_builder.joint_X_p[0] = root_pose
        articulation_builder.body_q[0] = root_pose

        builder = newton.ModelBuilder()
        for _i in range(self.world_count):
            articulation_builder.joint_q[7:] = self.rng.uniform(
                -1.0, 1.0, size=(len(articulation_builder.joint_q) - 7,)
            ).tolist()
            builder.add_world(articulation_builder)
        builder.add_ground_plane()

        # Finalize model
        self.model = builder.finalize()
        self.control = self.model.control()

        self.solver = newton.solvers.SolverMuJoCo(
            self.model,
            use_mujoco_cpu=False,
            solver="newton",
            integrator="euler",
            iterations=10,
            ls_iterations=5,
            njmax=100,
        )

        self.state_0, self.state_1 = self.model.state(), self.model.state()

        self.use_cuda_graph = wp.get_device().is_cuda

        if self.use_cuda_graph:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph

        # Set model in viewer (ViewerFile will automatically record it)
        self.viewer.set_model(self.model)

    def simulate(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.solver.step(self.state_0, self.state_1, self.control, None, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        with wp.ScopedTimer("step", active=False):
            if self.use_cuda_graph:
                wp.capture_launch(self.graph)
            else:
                self.simulate()
        self.sim_time += self.frame_dt

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)  # ViewerFile automatically records this
        self.viewer.end_frame()

    def test_final(self):
        pass


if __name__ == "__main__":
    # Create ViewerFile for automatic recording
    import sys

    recording_file = "humanoid_recording.bin"

    # Check if user wants a different filename from command line
    if len(sys.argv) > 1 and not sys.argv[1].startswith("-"):
        recording_file = sys.argv[1]

    print(f"Recording simulation to: {recording_file}")
    print("ViewerFile will automatically save when the simulation ends.")

    parser = newton.examples.create_parser()
    args = parser.parse_args()

    # Create ViewerFile with auto_save=False to only save at the end
    viewer = newton.viewer.ViewerFile(recording_file, auto_save=False)

    # Create example
    example = Example(viewer, args)

    # Run for a reasonable number of frames
    max_frames = 1000
    frame_count = 0

    while frame_count < max_frames:
        example.step()
        example.render()
        frame_count += 1

        if frame_count % 100 == 0:
            print(f"Frame {frame_count}/{max_frames}")

    # Close viewer (automatically saves recording)
    viewer.close()
    print(f"Recording completed: {recording_file}")
