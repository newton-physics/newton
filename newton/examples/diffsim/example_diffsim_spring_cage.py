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
# Example Diff Sim Spring Cage
#
# A single particle is attached with springs to each point of a cage.
# The objective is to optimize the rest length of the springs in order
# for the particle to be pulled towards a target position.
#
###########################################################################

import warp as wp

import newton


@wp.kernel
def compute_loss_kernel(
    pos: wp.array(dtype=wp.vec3),
    target_pos: wp.vec3,
    loss: wp.array(dtype=float),
):
    loss[0] = wp.length_sq(pos[0] - target_pos)


@wp.kernel(enable_backward=False)
def apply_gradient_kernel(
    spring_rest_lengths_grad: wp.array(dtype=float),
    train_rate: float,
    spring_rest_lengths: wp.array(dtype=float),
):
    tid = wp.tid()

    spring_rest_lengths[tid] -= spring_rest_lengths_grad[tid] * train_rate


class Example:
    def __init__(self, viewer_type: str, stage_path="example_spring_cage.usd", num_frames=30, train_iters=25):
        # Number of frames per second.
        self.fps = 30

        # Duration of a single simulation iteration in number of frames.
        self.num_frames = num_frames

        # Number of simulation steps to take per frame.
        self.sim_substep_count = 1

        # Delta time between each simulation substep.
        self.sim_dt = 1.0 / (self.fps * self.sim_substep_count)

        # Target position that we want the main particle to reach by optimising
        # the rest lengths of the springs.
        self.target_pos = (0.375, 0.125, 0.25)

        # Number of training iterations.
        self.train_iters = train_iters

        # Factor by which the rest lengths of the springs are adjusted after each
        # iteration, relatively to the corresponding gradients. Lower values
        # converge more slowly but have less chances to miss the local minimum.
        self.train_rate = 0.5

        # Initialize the helper to build a physics scene.
        builder = newton.ModelBuilder(gravity=0.0)

        # Define the main particle at the origin.
        particle_mass = 1.0
        builder.add_particle((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), particle_mass)

        # Define the cage made of points that will be pulling our main particle
        # using springs.
        # fmt: off
        builder.add_particle(( 0.2, -0.7,  0.8), (0.0, 0.0, 0.0), 0.0)
        builder.add_particle(( 1.1,  0.0,  0.2), (0.0, 0.0, 0.0), 0.0)
        builder.add_particle((-1.2,  0.1,  0.1), (0.0, 0.0, 0.0), 0.0)
        builder.add_particle(( 0.4,  0.6,  0.4), (0.0, 0.0, 0.0), 0.0)
        builder.add_particle((-0.2,  0.7, -0.9), (0.0, 0.0, 0.0), 0.0)
        builder.add_particle(( 0.1, -0.8, -0.8), (0.0, 0.0, 0.0), 0.0)
        builder.add_particle((-0.8, -0.9,  0.2), (0.0, 0.0, 0.0), 0.0)
        builder.add_particle((-0.1,  1.0,  0.4), (0.0, 0.0, 0.0), 0.0)
        # fmt: on

        # Define the spring constraints between the main particle and the cage points.
        spring_elastic_stiffness = 100.0
        spring_elastic_damping = 10.0
        for i in range(1, builder.particle_count):
            builder.add_spring(0, i, spring_elastic_stiffness, spring_elastic_damping, 0)

        # Build the model and set-up its properties.
        self.model = builder.finalize(requires_grad=True)

        # Use the Euler integrator for stepping through the simulation.
        self.solver = newton.solvers.SolverSemiImplicit(self.model)

        # Initialize a state for each simulation step.
        self.states = tuple(self.model.state() for _ in range(self.num_frames * self.sim_substep_count + 1))

        # Initialize a loss value that will represent the distance of the main
        # particle to the target position. It needs to be defined as an array
        # so that it can be written out by a kernel.
        self.loss = wp.zeros(1, dtype=float, requires_grad=True)

        if viewer_type == "usd":
            from newton.viewer import ViewerUSD  # noqa: PLC0415

            self.viewer = ViewerUSD(self.model, output_path=stage_path)
        elif viewer_type == "rerun":
            from newton.viewer import ViewerRerun  # noqa: PLC0415

            self.viewer = ViewerRerun(self.model, server=True, launch_viewer=True)
        else:
            from newton.viewer import ViewerGL  # noqa: PLC0415

            self.viewer = ViewerGL(self.model)

        # Allows rendering one simulation every N training iterations.
        self.render_iteration_steps = 2

        # Frame number used to render the simulation iterations.
        self.render_frame = 0

        self.use_cuda_graph = wp.get_device().is_cuda
        if self.use_cuda_graph:
            # Capture all the kernel launches into a CUDA graph so that they can
            # all be run in a single graph launch, which helps with performance.
            with wp.ScopedCapture() as capture:
                self.tape = wp.Tape()
                with self.tape:
                    self.forward()
                self.tape.backward(loss=self.loss)
            self.graph = capture.graph

    def forward(self):
        for i in range(1, len(self.states)):
            prev = self.states[i - 1]
            curr = self.states[i]
            prev.clear_forces()
            self.solver.step(
                prev,
                curr,
                control=None,
                contacts=None,
                dt=self.sim_dt,
            )

        last_state = self.states[-1]
        wp.launch(
            compute_loss_kernel,
            dim=1,
            inputs=(
                last_state.particle_q,
                self.target_pos,
            ),
            outputs=(self.loss,),
        )

    def step(self):
        with wp.ScopedTimer("step"):
            if self.use_cuda_graph:
                wp.capture_launch(self.graph)
            else:
                self.tape = wp.Tape()
                with self.tape:
                    self.forward()
                self.tape.backward(loss=self.loss)

            wp.launch(
                apply_gradient_kernel,
                dim=self.model.spring_count,
                inputs=(
                    self.model.spring_rest_length.grad,
                    self.train_rate,
                ),
                outputs=(self.model.spring_rest_length,),
            )

            self.tape.zero()

    def render(self):
        if self.viewer is None:
            return

        with wp.ScopedTimer("render"):
            self.viewer.begin_frame(0.0)
            self.viewer.log_shapes(
                "/target",
                newton.GeoType.BOX,
                (0.1, 0.1, 0.1),
                wp.array([wp.transform(self.target_pos, wp.quat_identity())], dtype=wp.transform),
                wp.array([wp.vec3(1.0, 0.0, 0.0)], dtype=wp.vec3),
            )
            self.viewer.end_frame()

            for frame in range(self.num_frames):
                self.viewer.begin_frame(self.render_frame / self.fps)
                # TODO: Make sure that log state, logs springs.
                self.viewer.log_state(self.states[frame * self.sim_substep_count])
                self.viewer.end_frame()

                self.render_frame += 1


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--viewer", choices=["gl", "usd", "rerun"], default="gl", help="Viewer backend to use.")
    parser.add_argument("--device", type=str, default=None, help="Override the default Warp device.")
    parser.add_argument(
        "--stage_path",
        type=lambda x: None if x == "None" else str(x),
        default="example_spring_cage.usd",
        help="Path to the output USD file.",
    )
    parser.add_argument("--num_frames", type=int, default=30, help="Total number of frames per training iteration.")
    parser.add_argument("--train_iters", type=int, default=25, help="Total number of training iterations.")
    parser.add_argument("--verbose", action="store_true", help="Print out additional status messages during execution.")

    args = parser.parse_known_args()[0]

    with wp.ScopedDevice(args.device):
        example = Example(
            viewer_type=args.viewer,
            stage_path=args.stage_path,
            num_frames=args.num_frames,
            train_iters=args.train_iters,
        )

        for iteration in range(args.train_iters):
            example.step()

            loss = example.loss.numpy()[0]

            if args.verbose:
                print(f"[{iteration:3d}] loss={loss:.8f}")

            render_iteration = iteration == example.train_iters - 1 or iteration % example.render_iteration_steps == 0
            if example.viewer and render_iteration:
                example.render()

        if example.viewer:
            example.viewer.close()
