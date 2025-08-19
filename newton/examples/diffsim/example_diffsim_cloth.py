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
# Example Sim Grad Cloth
#
# Shows how to use Warp to optimize the initial velocities of a piece of
# cloth such that its center of mass hits a target after a specified time.
#
# This example uses the built-in wp.Tape() object to compute gradients of
# the distance to target (loss) w.r.t the initial velocity, followed by
# a simple gradient-descent optimization step.
#
###########################################################################

import warp as wp

import newton


@wp.kernel
def com_kernel(positions: wp.array(dtype=wp.vec3), n: int, com: wp.array(dtype=wp.vec3)):
    tid = wp.tid()

    # compute center of mass
    wp.atomic_add(com, 0, positions[tid] / float(n))


@wp.kernel
def loss_kernel(com: wp.array(dtype=wp.vec3), target: wp.vec3, loss: wp.array(dtype=float)):
    # sq. distance to target
    delta = com[0] - target

    loss[0] = wp.dot(delta, delta)


@wp.kernel
def step_kernel(x: wp.array(dtype=wp.vec3), grad: wp.array(dtype=wp.vec3), alpha: float):
    tid = wp.tid()

    # gradient descent step
    x[tid] = x[tid] - grad[tid] * alpha


class Example:
    def __init__(self, viewer_type: str, stage_path="example_cloth_throw.usd", verbose=False):
        self.verbose = verbose

        # seconds
        sim_duration = 2.0

        # control frequency
        fps = 60
        self.frame_dt = 1.0 / fps
        frame_steps = int(sim_duration / self.frame_dt)

        # sim frequency
        self.sim_substeps = 16
        self.sim_steps = frame_steps * self.sim_substeps
        self.sim_dt = self.frame_dt / self.sim_substeps

        self.iter = 0
        self.render_time = 0.0

        self.train_rate = 5.0

        builder = newton.ModelBuilder()
        builder.default_particle_radius = 0.01

        dim_x = 16
        dim_y = 16

        builder.add_cloth_grid(
            pos=wp.vec3(0.0, 0.0, 0.0),
            vel=wp.vec3(0.0, 0.1, 0.1),
            rot=wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), -wp.pi * 0.75),
            dim_x=dim_x,
            dim_y=dim_y,
            cell_x=1.0 / dim_x,
            cell_y=1.0 / dim_y,
            mass=1.0,
            tri_ke=10000.0,
            tri_ka=10000.0,
            tri_kd=100.0,
            tri_lift=10.0,
            tri_drag=5.0,
        )

        self.model = builder.finalize()

        self.solver = newton.solvers.SolverSemiImplicit(self.model)

        self.target = (0.0, 8.0, 0.0)
        self.com = wp.zeros(1, dtype=wp.vec3, requires_grad=True)
        self.loss = wp.zeros(1, dtype=wp.float32, requires_grad=True)

        # allocate sim states for trajectory
        self.states = []
        for _i in range(self.sim_steps + 1):
            self.states.append(self.model.state(requires_grad=True))

        if viewer_type == "usd":
            from newton.viewer import ViewerUSD  # noqa: PLC0415

            self.viewer = ViewerUSD(self.model, output_path=stage_path)
        elif viewer_type == "rerun":
            from newton.viewer import ViewerRerun  # noqa: PLC0415

            self.viewer = ViewerRerun(self.model, server=True, launch_viewer=True)
        else:
            from newton.viewer import ViewerGL  # noqa: PLC0415

            self.viewer = ViewerGL(self.model)

        # capture forward/backward passes
        self.use_cuda_graph = wp.get_device().is_cuda
        if self.use_cuda_graph:
            with wp.ScopedCapture() as capture:
                self.tape = wp.Tape()
                with self.tape:
                    self.forward()
                self.tape.backward(self.loss)
            self.graph = capture.graph

    def forward(self):
        # run control loop
        for i in range(self.sim_steps):
            self.states[i].clear_forces()

            self.solver.step(self.states[i], self.states[i + 1], None, None, self.sim_dt)

        # compute loss on final state
        self.com.zero_()
        wp.launch(
            com_kernel,
            dim=self.model.particle_count,
            inputs=[self.states[-1].particle_q, self.model.particle_count, self.com],
        )
        wp.launch(loss_kernel, dim=1, inputs=[self.com, self.target, self.loss])

    def step(self):
        with wp.ScopedTimer("step"):
            if self.use_cuda_graph:
                wp.capture_launch(self.graph)
            else:
                self.tape = wp.Tape()
                with self.tape:
                    self.forward()
                self.tape.backward(self.loss)

            # gradient descent step
            x = self.states[0].particle_qd

            if self.verbose:
                print(f"Iter: {self.iter} Loss: {self.loss}")

            wp.launch(step_kernel, dim=len(x), inputs=[x, x.grad, self.train_rate])

            # clear grads for next iteration
            self.tape.zero()

            self.iter = self.iter + 1

    def render(self):
        if self.viewer is None:
            return

        with wp.ScopedTimer("render"):
            # draw trajectory
            traj_verts = [self.states[0].particle_q.numpy().mean(axis=0)]

            for i in range(0, self.sim_steps, self.sim_substeps):
                traj_verts.append(self.states[i].particle_q.numpy().mean(axis=0))

                self.viewer.begin_frame(self.render_time)
                self.viewer.log_state(self.states[i])
                self.viewer.log_shapes(
                    "/target",
                    newton.GeoType.BOX,
                    (0.1, 0.1, 0.1),
                    wp.array([wp.transform(self.target, wp.quat_identity())], dtype=wp.transform),
                    wp.array([wp.vec3(1.0, 0.0, 0.0)], dtype=wp.vec3),
                )
                self.viewer.log_lines(
                    f"/traj_{self.iter - 1}",
                    wp.array(traj_verts[0:-1], dtype=wp.vec3),
                    wp.array(traj_verts[1:], dtype=wp.vec3),
                    wp.render.bourke_color_map(0.0, 269.0, self.loss.numpy()[0]),
                )
                self.viewer.end_frame()

                self.render_time += self.frame_dt


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--viewer", choices=["gl", "usd", "rerun"], default="gl", help="Viewer backend to use.")
    parser.add_argument("--device", type=str, default=None, help="Override the default Warp device.")
    parser.add_argument(
        "--stage_path",
        type=lambda x: None if x == "None" else str(x),
        default="example_cloth_throw.usd",
        help="Path to the output USD file.",
    )
    parser.add_argument("--train_iters", type=int, default=64, help="Total number of training iterations.")
    parser.add_argument("--verbose", action="store_true", help="Print out additional status messages during execution.")

    args = parser.parse_known_args()[0]

    with wp.ScopedDevice(args.device):
        example = Example(viewer_type=args.viewer, stage_path=args.stage_path, verbose=args.verbose)

        # replay and optimize
        for i in range(args.train_iters):
            example.step()
            if i % 4 == 0:
                example.render()

        if example.viewer:
            example.viewer.close()
