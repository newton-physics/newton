# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Identify Unitree G1 armature from the recorded shuffle-dance motion.

Reproduce the ``diffsim/sysid_g1`` task through Newton's SolverMuJoCo: replay
recorded position targets, fit an armature scale to the first 50 motion frames,
and evaluate on the following 50 frames without fitting them. Four independent
worlds start with different armature scales. Blue translucent robots show the
recording. The robot is loaded from the 29-DOF Unitree G1 MJCF in newton-assets,
with the position controller and foot contacts used to collect the recording.
Recorded position targets are applied through Newton's joint controls.

Physics runs at 200 Hz, controls and observations at 50 Hz. The fit includes
the recorded root pose and differentiates through the full fitting horizon:
joint-only fitting with short truncated gradients can leave the floating base
drifting despite small joint errors. ``--tbptt 40`` enables the reference demo's
truncated backward windows to reduce memory use. The viewer shows
the first rollout and every 50th optimization iteration, including the held-out
continuation. Use the standard ``--viewer usd --output-path`` options to export.

Requires the experimental MuJoCo Warp adjoint branch (``enable_grad`` API).

Success criteria:
- Training reduces the pose error against the recorded motion.
- Losses, parameter gradients, and the held-out continuation remain finite.
- The simulated robots approach the blue reference ghosts as training progresses.
- At the nominal armature, the model reproduces the recorded joint and root poses.
- After 400 iterations, the fitted pelvis stays within 1 cm of the recording.

Runs:
- ``python -m newton.examples diffsim_g1``
- ``python -m newton.examples diffsim_g1 --viewer usd --output-path g1_sysid.usd``
"""

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import warp as wp
from warp.optim import Adam

import newton
import newton.examples
import newton.utils


@wp.kernel
def set_armature(base: wp.array[float], scale: wp.array[float], armature: wp.array[float]):
    i = wp.tid()
    armature[i] = base[i] * scale[i // 35]


@wp.kernel
def trajectory_loss(q: wp.array[float], reference: wp.array2d[float], frame: int, scale: float, loss: wp.array[float]):
    world, coord = wp.tid()
    # Include the floating base: matching joint angles alone can hide root drift.
    error = q[world * 36 + coord] - reference[frame, coord]
    wp.atomic_add(loss, 0, scale * error * error)


@wp.kernel
def accumulate_gradient(gradient: wp.array[float], total: wp.array[float]):
    i = wp.tid()
    total[i] += gradient[i]


@wp.kernel
def project_parameter(parameter: wp.array[float]):
    i = wp.tid()
    parameter[i] = wp.clamp(parameter[i], 0.1, 10.0)


class Example:
    def __init__(self, viewer, args):
        self.viewer = viewer
        self.world_count = args.world_count
        self.sim_steps = args.sim_steps
        self.eval_steps = args.eval_steps
        self.window = self.sim_steps if args.tbptt is None else min(args.tbptt, self.sim_steps)
        self.render_every = args.render_every
        self.sim_dt = 0.005
        self.frame_dt = 0.02
        self.repeat = 4
        self.sim_time = 0.0
        self.verbose = args.verbose
        self.loss_history = []
        self.eval_loss_history = []
        self.render_rollout = True
        if min(self.world_count, self.sim_steps, self.window, self.render_every) < 1 or self.eval_steps < 0:
            raise ValueError("World count, fit steps, TBPTT window, and render interval must be positive")
        if self.sim_steps % self.repeat or self.eval_steps % self.repeat or self.window % self.repeat:
            raise ValueError("Fit steps, evaluation steps, and TBPTT window must be multiples of 4 motion substeps")

        data_path = Path(newton.examples.get_asset("g1_sysid"))
        asset = newton.utils.download_asset("unitree_g1")
        with np.load(data_path / "shuffle_dance.npz") as motion:
            qpos = motion["qpos"].copy()
            qvel0 = motion["qvel"][0].copy()
            targets = motion["ctrl"]
            joint_names = motion["joint_names"]
            control_joints = motion["control_joints"]
            robot = self.create_robot(asset, motion)
        total_frames = (self.sim_steps + self.eval_steps) // self.repeat
        if total_frames > len(targets):
            raise ValueError(f"The recording provides at most {len(targets) * self.repeat} physics steps")
        qpos = qpos[: total_frames + 1]
        # MuJoCo stores wxyz quaternions; Newton stores xyzw.
        qpos[:, 3:7] = qpos[:, [4, 5, 6, 3]]

        assert robot.joint_dof_count - 6 == 29, "Expected the G1 with 29 actuated joints and a floating base"
        assert robot.joint_coord_count == qpos.shape[1] == 36
        assert len(joint_names) == len(control_joints) == 29
        joints = {name.rsplit("/", 1)[-1]: i for i, name in enumerate(robot.joint_label)}
        coords = np.array([robot.joint_q_start[joints[name]] for name in joint_names])
        dofs = np.array([robot.joint_qd_start[joints[name]] for name in joint_names])
        # Both poses and controls are name-mapped; their recorded orders differ.
        qpos[:, coords] = qpos[:, 7:].copy()
        qvel0[dofs] = qvel0[6:].copy()
        target_coords = [robot.joint_q_start[joints[name]] for name in control_joints]
        robot.joint_q[:] = qpos[0].tolist()
        # MuJoCo's free velocity is at the body origin with body-frame angular
        # velocity. Newton's is at the COM, both components in the parent frame
        # (world for this model's identity root joint anchors).
        rotation = wp.quat(qpos[0, 3:7])
        angular = np.array(wp.quat_rotate(rotation, wp.vec3(qvel0[3:6])))
        com = np.array(wp.quat_rotate(rotation, wp.vec3(robot.body_com[0])))
        qvel0[:3] += np.cross(angular, com)
        qvel0[3:6] = angular
        robot.joint_qd[:] = qvel0.tolist()

        builder = newton.ModelBuilder()
        builder.default_shape_cfg.gap = 0.0
        builder.replicate(robot, self.world_count)
        self.ghost_model = builder.finalize()
        self.ghost_model.shape_color.fill_(wp.vec3(0.2, 0.65, 1.0))
        self.ghost_model.shape_opacity.fill_(0.25)
        builder.add_ground_plane()
        self.model = builder.finalize(requires_grad=True)
        # Four friction rows for each of up to 28 sole contacts, plus joint limits.
        self.solver = newton.solvers.SolverMuJoCo(
            self.model, integrator="implicitfast", cone="pyramidal", iterations=10, ls_iterations=20, njmax=160
        )
        self.states = [self.model.state() for _ in range(self.window + 1)]
        self.initial_q = wp.clone(self.model.joint_q, requires_grad=False)
        self.initial_qd = wp.clone(self.model.joint_qd, requires_grad=False)
        self.controls = []
        for target in targets[:total_frames]:
            control = self.model.control(requires_grad=False)
            joint_target = np.zeros(robot.joint_coord_count, dtype=np.float32)
            joint_target[target_coords] = target
            control.joint_target_q.assign(np.tile(joint_target, self.world_count))
            self.controls.append(control)

        self.reference = wp.array(qpos, dtype=float)
        self.history = [self.model.state(requires_grad=False) for _ in range(total_frames + 1)]
        self.ghosts = [self.ghost_model.state() for _ in range(total_frames + 1)]
        for frame, ghost in enumerate(self.ghosts):
            ghost.joint_q.assign(np.tile(qpos[frame], self.world_count))
            newton.eval_fk(self.ghost_model, ghost.joint_q, ghost.joint_qd, ghost)
        newton.eval_fk(self.model, self.initial_q, self.initial_qd, self.history[0])

        self.base_armature = wp.clone(self.model.joint_armature, requires_grad=False)
        scales = np.linspace(0.5, 3.5, self.world_count) if self.world_count > 1 else np.array([2.0])
        self.parameter = wp.array(scales, dtype=float, requires_grad=True)
        self.gradient = wp.zeros_like(self.parameter, requires_grad=False)
        self.loss = wp.zeros(1, requires_grad=True)
        self.eval_loss = wp.zeros(1)
        self.optimizer = Adam([self.parameter], lr=args.learning_rate)
        self.project()

        # Warm up all per-step storage before capture. Each truncated window
        # finishes backward before reusing its states, bounding adjoint memory.
        self.forward_backward()
        self.initial_loss = float(self.loss.numpy()[0])
        self.evaluate()
        self.initial_eval_loss = float(self.eval_loss.numpy()[0])
        self.graph = None
        if self.model.device.is_cuda:
            with wp.ScopedCapture() as capture:
                self.forward_backward()
            self.graph = capture.graph

        self.viewer.activate("reference")
        self.viewer.set_model(self.ghost_model)
        self.viewer.set_world_offsets((2.7, 2.7, 0.0))
        self.viewer.activate("simulation")
        self.viewer.set_model(self.model)
        self.viewer.set_world_offsets((2.7, 2.7, 0.0))
        self.viewer.set_camera(wp.vec3(3.0, -4.0, 2.0), pitch=-18.0, yaw=130.0)

    @staticmethod
    def create_robot(asset, motion):
        """Configure the asset with the controller and foot contacts of the recording."""
        xml = ET.parse(asset / "mjcf" / "g1_29dof_rev_1_0.xml").getroot()
        xml.find("compiler").set("meshdir", str(asset / "meshes"))
        for name, armature in zip(motion["joint_names"], motion["armature"], strict=True):
            xml.find(f".//joint[@name='{name}']").set("armature", str(armature))

        # The asset supplies torque motors. This experiment recorded position
        # targets, with these gains and limits, rather than motor torques.
        actuators = xml.find("actuator")
        actuators.clear()
        for name, kp, kd, ctrlrange, forcerange in zip(
            motion["control_joints"], motion["kp"], motion["kd"], motion["ctrlrange"], motion["forcerange"], strict=True
        ):
            ET.SubElement(
                actuators,
                "position",
                joint=str(name),
                kp=str(kp),
                kv=str(kd),
                ctrlrange=" ".join(map(str, ctrlrange)),
                forcerange=" ".join(map(str, forcerange)),
            )

        # The recording uses capsule soles, while the stock asset has four
        # contact spheres per foot. Armature fitting cannot correct that change
        # in contact geometry. Keep the asset's links, inertias, and meshes.
        sole_segments = (
            "0.1 -0.026 -0.025 0.05 -0.026 -0.025",
            "-0.044 -0.018 -0.025 0.123 -0.018 -0.025",
            "-0.052 -0.01 -0.025 0.13 -0.01 -0.025",
            "-0.054 0 -0.025 0.132 0 -0.025",
            "-0.052 0.01 -0.025 0.13 0.01 -0.025",
            "-0.044 0.018 -0.025 0.123 0.018 -0.025",
            "0.1 0.026 -0.025 0.05 0.026 -0.025",
        )
        for side in ("left", "right"):
            foot = xml.find(f".//body[@name='{side}_ankle_roll_link']")
            for geom in foot.findall("geom"):
                if geom.get("type", "sphere") == "sphere":
                    foot.remove(geom)
            for i, segment in enumerate(sole_segments):
                fromto = "0.1 -0.026 -0.025 0.05 -0.027 -0.025" if side == "left" and i == 0 else segment
                ET.SubElement(
                    foot, "geom", type="capsule", size="0.01", fromto=fromto, friction="0.6", priority="1", group="3"
                )

        robot = newton.ModelBuilder()
        robot.default_shape_cfg.gap = 0.0
        robot.add_mjcf(ET.tostring(xml, encoding="unicode"), ignore_names=("floor",), enable_self_collisions=False)
        return robot

    def set_armature(self):
        wp.launch(
            set_armature,
            self.model.joint_dof_count,
            inputs=[self.base_armature, self.parameter],
            outputs=[self.model.joint_armature],
        )

    def project(self):
        wp.launch(project_parameter, self.parameter.size, inputs=[self.parameter])
        self.set_armature()
        # Reference-pose inverse weights are refreshed between optimizer steps,
        # outside the tape, following MuJoCo Warp's parameter-gradient convention.
        self.solver.notify_model_changed(newton.ModelFlags.JOINT_DOF_INERTIAL_PROPERTIES)

    def forward_backward(self):
        self.loss.zero_()
        self.gradient.zero_()
        wp.copy(self.states[0].joint_q, self.initial_q)
        wp.copy(self.states[0].joint_qd, self.initial_qd)
        for start in range(0, self.sim_steps, self.window):
            count = min(self.window, self.sim_steps - start)
            tape = wp.Tape()
            with tape:
                self.set_armature()
                for i in range(count):
                    step = start + i
                    self.solver.step(
                        self.states[i], self.states[i + 1], self.controls[step // self.repeat], None, self.sim_dt
                    )
                    if (step + 1) % self.repeat == 0:
                        frame = (step + 1) // self.repeat
                        wp.launch(
                            trajectory_loss,
                            (self.world_count, 36),
                            inputs=[self.states[i + 1].joint_q, self.reference, frame, 1.0 / self.world_count],
                            outputs=[self.loss],
                        )
                        wp.copy(self.history[frame].body_q, self.states[i + 1].body_q)
            tape.backward(self.loss)
            wp.launch(accumulate_gradient, self.world_count, inputs=[self.parameter.grad], outputs=[self.gradient])
            tape.zero()
            # Carry values into the next window after backward, detaching history.
            wp.copy(self.states[0].joint_q, self.states[count].joint_q)
            wp.copy(self.states[0].joint_qd, self.states[count].joint_qd)

    def evaluate(self):
        """Continue from the fitted horizon without adding gradients or resetting state."""
        self.eval_loss.zero_()
        for step in range(self.sim_steps, self.sim_steps + self.eval_steps):
            self.solver.step(self.states[0], self.states[1], self.controls[step // self.repeat], None, self.sim_dt)
            if (step + 1) % self.repeat == 0:
                frame = (step + 1) // self.repeat
                wp.launch(
                    trajectory_loss,
                    (self.world_count, 36),
                    inputs=[self.states[1].joint_q, self.reference, frame, 1.0 / self.world_count],
                    outputs=[self.eval_loss],
                )
                wp.copy(self.history[frame].body_q, self.states[1].body_q)
            wp.copy(self.states[0].joint_q, self.states[1].joint_q)
            wp.copy(self.states[0].joint_qd, self.states[1].joint_qd)

    def step(self):
        if self.graph is not None:
            wp.capture_launch(self.graph)
        else:
            self.forward_backward()
        loss = float(self.loss.numpy()[0])
        if not np.isfinite(loss) or not np.all(np.isfinite(self.gradient.numpy())):
            raise RuntimeError("Non-finite G1 loss or gradient")
        self.loss_history.append(loss)
        self.render_rollout = len(self.loss_history) == 1 or len(self.loss_history) % self.render_every == 0
        if self.render_rollout:
            self.evaluate()
            self.eval_loss_history.append(float(self.eval_loss.numpy()[0]))
            if self.verbose:
                print(
                    f"Iteration {len(self.loss_history)}: fit={loss:.6g}, "
                    f"held-out={self.eval_loss_history[-1]:.6g}, scales={self.parameter.numpy()}"
                )
        self.optimizer.step([self.gradient])
        self.project()

    def render(self):
        if not self.render_rollout:
            return
        for state, ghost in zip(self.history, self.ghosts, strict=True):
            self.viewer.begin_frame(self.sim_time)
            self.viewer.activate("reference")
            self.viewer.log_state(ghost)
            self.viewer.activate("simulation")
            self.viewer.log_state(state)
            self.viewer.log_scalar("fit_loss", self.loss_history[-1] if self.loss_history else self.initial_loss)
            self.viewer.log_scalar(
                "held_out_loss", self.eval_loss_history[-1] if self.eval_loss_history else self.initial_eval_loss
            )
            self.viewer.end_frame()
            self.sim_time += self.frame_dt

    def test_final(self):
        """Check learning and require the calibrated model to reproduce the recording."""
        assert len(self.loss_history) >= 2, "Run at least two optimization iterations"
        assert np.all(np.isfinite(self.loss_history))
        assert np.all(np.isfinite(self.eval_loss_history))
        assert min(self.loss_history[1:]) < self.initial_loss, "Training did not reduce the recorded-motion error"
        assert np.all(np.isfinite(self.history[-1].body_q.numpy()))
        # Short CI runs check learning and nominal replay below. A complete run
        # must also match the recording with the parameters actually learned.
        if len(self.loss_history) >= 400:
            for state, ghost in zip(self.history, self.ghosts, strict=True):
                actual = state.body_q.numpy().reshape(self.world_count, -1, 7)[:, 0, :3]
                reference = ghost.body_q.numpy().reshape(self.world_count, -1, 7)[:, 0, :3]
                assert np.max(np.linalg.norm(actual - reference, axis=1)) < 0.01, (
                    "The fitted G1 pelvis drifts over 1 cm"
                )

        # A decreasing loss alone can hide a model mismatch that armature cannot
        # correct. The recording's nominal armature must reproduce its motion,
        # including the floating base and the unfitted continuation.
        fitted_parameter = self.parameter.numpy()
        self.parameter.fill_(1.0)
        self.project()
        wp.copy(self.states[0].joint_q, self.initial_q)
        wp.copy(self.states[0].joint_qd, self.initial_qd)
        reference = self.reference.numpy()
        try:
            for step in range(self.sim_steps + self.eval_steps):
                self.solver.step(self.states[0], self.states[1], self.controls[step // self.repeat], None, self.sim_dt)
                if (step + 1) % self.repeat == 0:
                    frame = (step + 1) // self.repeat
                    actual = self.states[1].joint_q.numpy().reshape(self.world_count, 36)
                    np.testing.assert_allclose(
                        actual,
                        np.broadcast_to(reference[frame], actual.shape),
                        atol=2.0e-3,
                        rtol=0.0,
                        err_msg=f"The calibrated G1 model does not reproduce recording frame {frame}",
                    )
                wp.copy(self.states[0].joint_q, self.states[1].joint_q)
                wp.copy(self.states[0].joint_qd, self.states[1].joint_qd)
        finally:
            self.parameter.assign(fitted_parameter)
            self.project()

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        newton.examples.add_world_count_arg(parser)
        parser.set_defaults(num_frames=900, world_count=4)
        parser.add_argument("--sim-steps", type=int, default=200, help="Physics steps used for fitting.")
        parser.add_argument("--eval-steps", type=int, default=200, help="Held-out continuation steps.")
        parser.add_argument(
            "--tbptt", type=int, help="Backward window in physics steps; defaults to the full fit horizon."
        )
        parser.add_argument(
            "--render-every", type=int, default=50, help="Optimizer iterations between rendered rollouts."
        )
        parser.add_argument("--learning-rate", type=float, default=0.03)
        parser.add_argument("--verbose", action="store_true", help="Print losses and fitted armature scales.")
        return parser


if __name__ == "__main__":
    viewer, args = newton.examples.init(Example.create_parser())
    example = Example(viewer, args)
    newton.examples.run(example, args)
