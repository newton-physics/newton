# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Example Diffsim Ball

Optimize independent launch velocities in four parallel worlds so each ball
bounces off the wall and floor to hit its target. Use ``--world-count`` to
change the batch size.

The default solver uses a particle. ``--solver mujoco`` uses a rigid sphere
with Newton's SolverMuJoCo and requires the MuJoCo Warp adjoint branch.

This example uses the built-in wp.Tape() object to compute gradients of
the distance to target (loss) w.r.t the initial velocity, followed by
a simple gradient-descent optimization step.

Success criteria:
- A ball bounces off the ground and wall toward a floating target, with previous trajectories
  retained in the view.
- Training improves target accuracy and the loss plot shows the optimization trend.
- When the gradient check runs, numeric and analytic gradients agree within the example check
  tolerance.

Runs:
- `python -m newton.examples diffsim_ball`
- `python -m newton.examples diffsim_ball --solver mujoco`
"""

import numpy as np
import warp as wp

import newton
import newton.examples
from newton.tests.unittest_utils import assert_np_equal
from newton.utils import bourke_color_map


@wp.kernel
def loss_kernel(pos: wp.array[wp.vec3], target: wp.vec3, losses: wp.array[float], loss: wp.array[float]):
    world = wp.tid()
    # distance to target
    delta = pos[world] - target
    losses[world] = wp.dot(delta, delta)
    wp.atomic_add(loss, 0, losses[world])


@wp.kernel
def rigid_loss_kernel(pos: wp.array[wp.transform], target: wp.vec3, losses: wp.array[float], loss: wp.array[float]):
    world = wp.tid()
    delta = wp.transform_get_translation(pos[world]) - target
    losses[world] = wp.dot(delta, delta)
    wp.atomic_add(loss, 0, losses[world])


@wp.kernel
def set_rigid_velocity(velocity: wp.array[wp.vec3], joint_qd: wp.array[float]):
    world = wp.tid()
    for i in range(3):
        joint_qd[6 * world + i] = velocity[world][i]
        joint_qd[6 * world + i + 3] = 0.0


@wp.kernel
def step_kernel(x: wp.array[wp.vec3], grad: wp.array[wp.vec3], alpha: float):
    tid = wp.tid()

    # gradient descent step
    x[tid] = x[tid] - grad[tid] * alpha


class Example:
    def __init__(self, viewer, args):
        self.use_mujoco = args.solver == "mujoco"
        self.world_count = args.world_count
        if self.world_count < 1:
            raise ValueError("--world-count must be positive")
        # setup simulation parameters first
        self.fps = 60
        self.frame = 0
        self.frame_dt = 1.0 / self.fps
        self.sim_steps = 36
        self.sim_substeps = 8
        self.sim_dt = self.frame_dt / self.sim_substeps

        self.verbose = args.verbose

        self.train_iter = 0
        self.train_rate = 0.02
        self.target = (0.0, -2.0, 1.5)
        self.loss = wp.zeros(1, dtype=wp.float32, requires_grad=True)
        self.losses = wp.zeros(self.world_count, dtype=float, requires_grad=True)
        self.loss_history = []

        self.viewer = viewer
        self.viewer.show_particles = True

        # setup simulation scene
        scene = newton.ModelBuilder(up_axis=newton.Axis.Z)

        # add wall and ground plane
        ke = 1.0e4
        kf = 0.0
        kd = 1.0e1
        mu = 0.2

        if self.use_mujoco:
            ball = scene.add_body(xform=wp.transform(wp.vec3(0.0, -0.5, 1.0)))
            radius = 0.1
            scene.add_shape_sphere(
                ball,
                radius=radius,
                cfg=newton.ModelBuilder.ShapeConfig(
                    density=1.0 / (4.0 / 3.0 * np.pi * radius**3),
                    ke=ke,
                    kd=kd,
                    mu=mu,
                ),
            )
        else:
            scene.add_particle(pos=wp.vec3(0.0, -0.5, 1.0), vel=wp.vec3(0.0, 5.0, -5.0), mass=1.0)

        scene.add_shape_box(
            body=-1,
            xform=wp.transform(wp.vec3(0.0, 2.0, 1.0), wp.quat_identity()),
            hx=1.0,
            hy=0.25,
            hz=1.0,
            cfg=newton.ModelBuilder.ShapeConfig(ke=ke, kf=kf, kd=kd, mu=mu),
        )

        builder = newton.ModelBuilder()
        builder.replicate(scene, self.world_count)
        builder.add_ground_plane(cfg=newton.ModelBuilder.ShapeConfig(ke=ke, kf=kf, kd=kd, mu=mu))

        # finalize model
        # use `requires_grad=True` to create a model for differentiable simulation
        self.model = builder.finalize(requires_grad=True)

        self.model.soft_contact_ke = ke
        self.model.soft_contact_kf = kf
        self.model.soft_contact_kd = kd
        self.model.soft_contact_mu = mu
        self.model.soft_contact_restitution = 1.0
        # Each world contains one particle, so no particle-particle contacts are needed.
        self.model.particle_grid = None

        self.solver = (
            newton.solvers.SolverMuJoCo(self.model, integrator="euler", njmax=16, nconmax=4)
            if self.use_mujoco
            else newton.solvers.SolverSemiImplicit(self.model)
        )

        # allocate sim states, initialize control and one-shot contacts (valid for simple collisions against constant plane)
        self.states = [self.model.state() for _ in range(self.sim_steps * self.sim_substeps + 1)]
        self.control = self.model.control()
        self.initial_velocity = (
            wp.empty(self.world_count, dtype=wp.vec3, requires_grad=True)
            if self.use_mujoco
            else self.states[0].particle_qd
        )
        velocities = np.tile((0.0, 5.0, -5.0), (self.world_count, 1)).astype(np.float32)
        if self.world_count > 1:
            velocities[:, 0] += np.linspace(-0.25, 0.25, self.world_count)
        self.initial_velocity.assign(velocities)
        self.starting_velocities = velocities

        # Create collision pipeline (requires_grad for differentiable simulation)
        self.contacts = None
        if not self.use_mujoco:
            self.collision_pipeline = newton.CollisionPipeline(
                self.model,
                broad_phase="explicit",
                soft_contact_gap=10.0,
                requires_grad=True,
            )
            self.contacts = self.collision_pipeline.contacts()
            self.collision_pipeline.collide(self.states[0], self.contacts)

        self.viewer.set_model(self.model)
        self.viewer.set_world_offsets((2.5, 5.0, 0.0))
        self.world_offsets = self.viewer.world_offsets.numpy()
        self.target_xforms = wp.array(
            [wp.transform(np.array(self.target) + offset, wp.quat_identity()) for offset in self.world_offsets],
            dtype=wp.transform,
        )

        # capture forward/backward passes
        self.capture()

    def capture(self):
        # MuJoCo allocates per-step adjoint storage on the first rollout.
        self.forward_backward()
        self.tape.zero()
        self.graph = None
        if self.model.device.is_cuda:
            with wp.ScopedCapture() as capture:
                self.forward_backward()
            self.graph = capture.graph

    def forward_backward(self):
        self.tape = wp.Tape()
        with self.tape:
            self.forward()
        self.tape.backward(self.loss)

    def forward(self):
        self.loss.zero_()
        if self.use_mujoco:
            wp.launch(
                set_rigid_velocity,
                dim=self.world_count,
                inputs=[self.initial_velocity],
                outputs=[self.states[0].joint_qd],
            )
        # run simulation loop
        for sim_step in range(self.sim_steps):
            self.simulate(sim_step)

        # compute loss on final state
        wp.launch(
            rigid_loss_kernel if self.use_mujoco else loss_kernel,
            dim=self.world_count,
            inputs=[self.states[-1].body_q if self.use_mujoco else self.states[-1].particle_q, self.target],
            outputs=[self.losses, self.loss],
        )

        return self.loss

    def simulate(self, sim_step):
        for i in range(self.sim_substeps):
            t = sim_step * self.sim_substeps + i
            self.states[t].clear_forces()
            self.solver.step(self.states[t], self.states[t + 1], self.control, self.contacts, self.sim_dt)

    def step(self):
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.forward_backward()

        x = self.initial_velocity

        if self.verbose:
            print(f"Train iter: {self.train_iter} Loss: {self.loss}")
            print(f"    x: {x} g: {x.grad}")

        # gradient descent step
        wp.launch(step_kernel, dim=len(x), inputs=[x, x.grad, self.train_rate])

        # clear grads for next iteration
        self.tape.zero()

        self.train_iter += 1
        self.loss_history.append(self.losses.numpy().copy())

    def test_final(self):
        """Check launch-velocity gradients and require a reduction in target error."""
        # Check a fixed initial point: optimizer iterates can land on contact
        # changes where a central difference spans two different derivatives.
        optimized = self.initial_velocity.numpy().copy()
        self.initial_velocity.assign(self.starting_velocities)
        x_grad_numeric, x_grad_analytic = self.check_grad()
        self.initial_velocity.assign(optimized)
        assert np.all(np.isfinite(x_grad_numeric)) and np.all(np.isfinite(x_grad_analytic))
        assert_np_equal(x_grad_numeric, x_grad_analytic, tol=5e-2)
        assert np.all(np.isfinite(self.loss_history))
        assert np.all(self.loss_history[-1] < 0.75 * self.loss_history[0]), "Training did not improve every world"

    def render(self):
        if self.viewer.is_paused():
            self.viewer.begin_frame(self.viewer.time)
            self.viewer.end_frame()
            return

        if self.frame > 0 and self.train_iter % 16 != 0:
            return

        # draw trajectory
        traj_verts = [self.ball_positions(self.states[0])]

        for i in range(self.sim_steps + 1):
            state = self.states[i * self.sim_substeps]
            traj_verts.append(self.ball_positions(state))

            self.viewer.begin_frame(self.frame * self.frame_dt)
            self.viewer.log_scalar("/loss", self.loss.numpy()[0] / self.world_count)
            self.viewer.log_state(state)
            if self.contacts is not None:
                self.viewer.log_contacts(self.contacts, state)
            self.viewer.log_shapes(
                "/target",
                newton.GeoType.BOX,
                (0.1, 0.1, 0.1),
                self.target_xforms,
                wp.array([wp.vec3(0.5, 0.0, 0.5)], dtype=wp.vec3),
            )
            self.viewer.log_lines(
                f"/traj_{self.train_iter - 1}",
                wp.array(np.array(traj_verts[:-1]).reshape(-1, 3), dtype=wp.vec3),
                wp.array(np.array(traj_verts[1:]).reshape(-1, 3), dtype=wp.vec3),
                bourke_color_map(0.0, 7.0, self.loss.numpy()[0] / self.world_count),
            )
            self.viewer.end_frame()

            self.frame += 1

    def ball_positions(self, state):
        positions = state.body_q.numpy()[:, :3] if self.use_mujoco else state.particle_q.numpy()
        return positions + self.world_offsets

    def check_grad(self):
        param = self.initial_velocity

        # initial value
        x_c = param.numpy().copy()

        # compute numeric gradient
        x_grad_numeric = np.zeros_like(x_c)

        # Worlds are independent, so perturb the same component in every world
        # and differentiate their individual losses in just six rollouts.
        for i in range(3):
            # Keep the loss difference above float32 rollout noise.
            eps = 2.0e-3

            step = np.zeros_like(x_c)
            step[:, i] = eps

            x_1 = x_c + step
            x_0 = x_c - step

            param.assign(x_1)
            self.forward()
            l_1 = self.losses.numpy().copy()

            param.assign(x_0)
            self.forward()
            l_0 = self.losses.numpy().copy()

            dldx = (l_1 - l_0) / (eps * 2.0)

            x_grad_numeric[:, i] = dldx

        # reset initial state
        param.assign(x_c)

        # compute analytic gradient
        tape = wp.Tape()
        with tape:
            l = self.forward()

        tape.backward(l)

        x_grad_numeric = x_grad_numeric.flatten()
        x_grad_analytic = param.grad.numpy().flatten().copy()

        print(f"numeric grad: {x_grad_numeric.tolist()}")
        print(f"analytic grad: {x_grad_analytic.tolist()}")

        tape.zero()

        return x_grad_numeric, x_grad_analytic

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.add_argument("--solver", choices=("semi-implicit", "mujoco"), default="semi-implicit")
        newton.examples.add_world_count_arg(parser)
        parser.set_defaults(world_count=4, num_frames=1024)
        parser.add_argument(
            "--verbose", action="store_true", help="Print out additional status messages during execution."
        )
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)

    newton.examples.run(Example(viewer, args), args)
