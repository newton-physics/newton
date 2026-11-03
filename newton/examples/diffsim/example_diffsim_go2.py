# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Optimize a Unitree Go2 handstand through Newton's SolverMuJoCo.

Learn an open-loop sequence of joint position targets for the Newton-assets Go2.
The objective rotates the trunk onto its front legs while keeping its head off
the ground and limiting angular velocity. This follows the handstand task in
the MuJoCo Warp diffsim demos, using Newton states, controls, and assets.

Requires the experimental MuJoCo Warp adjoint branch (``enable_grad`` API).

Success criteria:
- Training reduces the handstand objective with finite gradients.
- The optimized motion lifts the rear legs and pitches the trunk toward vertical.
- The head stays above the floor during the handstand.

Runs:
- ``python -m newton.examples diffsim_go2``
- ``python -m newton.examples diffsim_go2 --device cpu --world-count 1``
"""

import numpy as np
import warp as wp
from warp.optim import Adam

import newton
import newton.examples
import newton.utils


@wp.kernel
def set_targets(parameter: wp.array2d[float], indices: wp.array[int], step: int, target: wp.array[float]):
    i = wp.tid()
    target[indices[i]] = parameter[step, i]


@wp.kernel
def handstand_loss(
    q: wp.array[float],
    qd: wp.array[float],
    progress: float,
    scale: float,
    loss: wp.array[float],
):
    world = wp.tid()
    qi = world * 19
    vi = world * 18
    rotation = wp.quat(q[qi + 3], q[qi + 4], q[qi + 5], q[qi + 6])
    forward = wp.quat_rotate(rotation, wp.vec3(1.0, 0.0, 0.0))
    alignment = 0.70710678 * (rotation[3] + rotation[1])
    head_below = wp.max(0.25 - q[qi + 2] - 0.3 * forward[2], 0.0)
    omega = wp.vec3(qd[vi + 3], qd[vi + 4], qd[vi + 5])
    error = 2.0 * (1.0 - alignment * alignment) + 8.0 * head_below * head_below + 0.02 * wp.dot(omega, omega)
    wp.atomic_add(loss, 0, scale * progress * error)


@wp.kernel
def control_loss(parameter: wp.array2d[float], home: wp.array[float], step: int, scale: float, loss: wp.array[float]):
    i = wp.tid()
    delta = parameter[step, i] - home[i]
    wp.atomic_add(loss, 0, scale * 0.002 * delta * delta)


@wp.kernel
def front_foot_loss(
    body_q: wp.array[wp.transform],
    foot_shapes: wp.array[int],
    shape_body: wp.array[int],
    shape_transform: wp.array[wp.transform],
    shape_scale: wp.array[wp.vec3],
    scale: float,
    loss: wp.array[float],
):
    shape = foot_shapes[wp.tid()]
    foot = wp.transform_point(body_q[shape_body[shape]], wp.transform_get_translation(shape_transform[shape]))
    height = foot[2] - shape_scale[shape][0]
    wp.atomic_add(loss, 0, scale * 40.0 * height * height)


@wp.kernel
def project_controls(parameter: wp.array2d[float], lower: wp.array[float], upper: wp.array[float]):
    step, i = wp.tid()
    parameter[step, i] = wp.clamp(parameter[step, i], lower[i], upper[i])


class Example:
    def __init__(self, viewer, args):
        self.viewer = viewer
        self.world_count = args.world_count
        self.sim_steps = args.sim_steps
        if self.world_count < 1 or self.sim_steps < 2:
            raise ValueError("--world-count must be positive and --sim-steps must be at least 2")
        self.sim_dt = 0.005
        self.frame_dt = self.sim_dt
        self.sim_time = 0.0
        self.verbose = args.verbose
        self.loss_history = []

        robot = newton.ModelBuilder()
        newton.solvers.SolverMuJoCo.register_custom_attributes(robot)
        asset = newton.utils.download_asset("unitree_go2")
        robot.add_usd(
            str(asset / "usd" / "go2.usda"),
            collapse_fixed_joints=True,
            enable_self_collisions=False,
            hide_collision_shapes=True,
            joint_ordering="dfs",
        )
        robot.approximate_meshes("convex_hull")
        robot.joint_q[:7] = [0.0, 0.0, 0.35, 0.0, 0.0, 0.0, 1.0]
        target_indices, home_targets, lower, upper = [], [], [], []
        # Standing pose and drive gains match the Go2 asset's Newton policy setup.
        for joint, label in enumerate(robot.joint_label):
            if robot.joint_type[joint] != newton.JointType.REVOLUTE:
                continue
            name = label.rsplit("/", 1)[-1]
            if "hip" in name:
                home = 0.1 if name.startswith(("FL", "RL")) else -0.1
            elif "thigh" in name:
                home = 0.8 if name.startswith(("FL", "FR")) else 1.0
            else:
                home = -1.5
            dof = robot.joint_qd_start[joint]
            coord = robot.joint_q_start[joint]
            robot.joint_q[coord] = home
            robot.joint_target_q[coord] = home
            robot.joint_target_ke[dof] = 50.0
            robot.joint_target_kd[dof] = 1.0
            robot.joint_armature[dof] = 0.1
            robot.joint_target_mode[dof] = int(newton.JointTargetMode.POSITION)
            target_indices.append(coord)
            home_targets.append(home)
            lower.append(max(home - 1.2, robot.joint_limit_lower[dof]))
            upper.append(min(home + 1.2, robot.joint_limit_upper[dof]))
        if robot.joint_coord_count != 19 or robot.joint_dof_count != 18:
            raise ValueError("The Go2 handstand requires a free base and twelve revolute joints")
        builder = newton.ModelBuilder()
        builder.replicate(robot, self.world_count)
        builder.add_ground_plane()
        self.model = builder.finalize(requires_grad=True)
        front_shapes = [
            next(i for i, label in enumerate(robot.shape_label) if f"/{leg}_foot/collisions/" in label)
            for leg in ("FL", "FR")
        ]
        self.front_foot_shapes = wp.array(
            [world * robot.shape_count + shape for world in range(self.world_count) for shape in front_shapes],
            dtype=int,
        )
        self.solver = newton.solvers.SolverMuJoCo(
            self.model, integrator="euler", cone="pyramidal", impratio=1.0, njmax=512, nconmax=128
        )
        self.states = [self.model.state() for _ in range(self.sim_steps + 1)]
        self.controls = [self.model.control() for _ in range(self.sim_steps)]
        self.home = wp.array(np.tile(home_targets, self.world_count), dtype=float)
        self.target_indices = wp.array(
            [world * 19 + coord for world in range(self.world_count) for coord in target_indices], dtype=int
        )

        # Settle under the standing targets, then fix that initial condition for training.
        current, output = self.model.state(), self.model.state()
        for _ in range(round(0.8 / self.sim_dt)):
            self.solver.step(current, output, self.controls[0], None, self.sim_dt)
            current, output = output, current
        wp.copy(self.states[0].joint_q, current.joint_q)
        self.states[0].joint_qd.zero_()
        newton.eval_fk(self.model, self.states[0].joint_q, self.states[0].joint_qd, self.states[0])

        # Hold each target for two physics steps (100 Hz control).
        control_steps = (self.sim_steps + 1) // 2
        home = self.home.numpy()
        offset = np.random.default_rng(args.seed).uniform(-0.12, 0.12, (self.world_count, 12)).astype(np.float32)
        initial = np.tile(home + offset.ravel(), (control_steps, 1))
        self.parameter = wp.array(initial, dtype=float, requires_grad=True)
        self.lower = wp.array(np.tile(lower, self.world_count), dtype=float)
        self.upper = wp.array(np.tile(upper, self.world_count), dtype=float)
        self.loss = wp.zeros(1, requires_grad=True)
        self.optimizer = Adam([self.parameter.flatten()], lr=args.learning_rate, betas=(0.7, 0.95))

        self.forward_backward()
        self.initial_loss = float(self.loss.numpy()[0])
        self.tape.zero()
        self.graph = None
        if self.model.device.is_cuda:
            with wp.ScopedCapture() as capture:
                self.forward_backward()
            self.graph = capture.graph
        self.viewer.set_model(self.model)
        self.viewer.set_world_offsets((1.5, 1.5, 0.0))
        self.viewer.set_camera(wp.vec3(1.6, -2.0, 1.0), pitch=-15.0, yaw=130.0)

    def forward(self):
        self.loss.zero_()
        scale = 1.0 / (self.sim_steps * self.world_count)
        for i in range(self.sim_steps):
            target = self.controls[i].joint_target_q
            wp.launch(
                set_targets, self.home.size, inputs=[self.parameter, self.target_indices, i // 2], outputs=[target]
            )
            self.solver.step(self.states[i], self.states[i + 1], self.controls[i], None, self.sim_dt)
            wp.launch(
                handstand_loss,
                self.world_count,
                inputs=[self.states[i + 1].joint_q, self.states[i + 1].joint_qd, (i + 1) / self.sim_steps, scale],
                outputs=[self.loss],
            )
            wp.launch(
                control_loss, self.home.size, inputs=[self.parameter, self.home, i // 2, scale], outputs=[self.loss]
            )
            wp.launch(
                front_foot_loss,
                self.front_foot_shapes.size,
                inputs=[
                    self.states[i + 1].body_q,
                    self.front_foot_shapes,
                    self.model.shape_body,
                    self.model.shape_transform,
                    self.model.shape_scale,
                    scale * (i + 1) / self.sim_steps,
                ],
                outputs=[self.loss],
            )

    def forward_backward(self):
        self.tape = wp.Tape()
        with self.tape:
            self.forward()
        self.tape.backward(self.loss)

    def step(self):
        if self.graph is not None:
            wp.capture_launch(self.graph)
        else:
            self.forward_backward()
        loss = float(self.loss.numpy()[0])
        if not np.isfinite(loss) or not np.all(np.isfinite(self.parameter.grad.numpy())):
            raise RuntimeError("Non-finite Go2 loss or gradient")
        self.loss_history.append(loss)
        if self.verbose:
            print(f"Iteration {len(self.loss_history)}: loss={loss:.6g}, alignment={self.alignment().max():.4f}")
        self.optimizer.step([self.parameter.grad.flatten()])
        wp.launch(project_controls, self.parameter.shape, inputs=[self.parameter, self.lower, self.upper])
        self.tape.zero()

    def alignment(self):
        q = self.states[-1].joint_q.numpy().reshape(self.world_count, 19)
        return 0.5 * (q[:, 6] + q[:, 4]) ** 2

    def render(self):
        for state in self.states[1::4]:
            self.viewer.begin_frame(self.sim_time)
            self.viewer.log_state(state)
            self.viewer.log_scalar("loss", self.loss_history[-1] if self.loss_history else self.initial_loss)
            self.viewer.end_frame()
            self.sim_time += 4.0 * self.sim_dt

    def test_final(self):
        """Require finite trajectories and improvement in the handstand objective."""
        assert len(self.loss_history) >= 2, "Run at least two optimization iterations"
        assert np.all(np.isfinite(self.loss_history))
        assert min(self.loss_history[1:]) < self.initial_loss, "Training did not reduce the handstand objective"
        assert np.all(np.isfinite(self.states[-1].body_q.numpy()))
        if self.sim_steps >= 200 and len(self.loss_history) >= 80:
            q = self.states[-1].joint_q.numpy().reshape(self.world_count, 19)
            head_height = q[:, 2] + 0.6 * (q[:, 3] * q[:, 5] - q[:, 6] * q[:, 4])
            assert np.any((self.alignment() > 0.9) & (head_height > 0.1)), "No world reached a handstand"

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        newton.examples.add_world_count_arg(parser)
        parser.set_defaults(num_frames=8000, world_count=4)
        parser.add_argument("--sim-steps", type=int, default=200, help="Physics steps in each optimization rollout.")
        parser.add_argument("--learning-rate", type=float, default=0.03)
        parser.add_argument("--seed", type=int, default=0)
        parser.add_argument(
            "--verbose", action="store_true", help="Print loss and orientation alignment during training."
        )
        return parser


if __name__ == "__main__":
    viewer, args = newton.examples.init(Example.create_parser())
    example = Example(viewer, args)
    newton.examples.run(example, args)
