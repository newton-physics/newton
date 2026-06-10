# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Controllers — PID swinging a double pendulum to a setpoint.
#
# A ControllerPID drives both joints of a planar 2-link pendulum to fixed
# target angles by writing joint efforts into control.joint_f, which the
# XPBD solver applies each substep.
#
# Command: python -m newton.examples pid_pendulum
###########################################################################

import warp as wp

import newton
import newton.examples
from newton.controllers import ControllerPID


class Example:
    def __init__(self, viewer, args):
        self.fps = 200
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = 10
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0
        self.viewer = viewer
        self.device = wp.get_device()

        # Two-link planar pendulum, one revolute joint per link.
        builder = newton.ModelBuilder()
        link_0 = builder.add_link()
        builder.add_shape_box(link_0, hx=0.5, hy=0.05, hz=0.05)
        link_1 = builder.add_link()
        builder.add_shape_box(link_1, hx=0.5, hy=0.05, hz=0.05)

        j0 = builder.add_joint_revolute(
            parent=-1,
            child=link_0,
            axis=wp.vec3(0.0, 1.0, 0.0),
            parent_xform=wp.transform(p=wp.vec3(0.0, 0.0, 3.0)),
            child_xform=wp.transform(p=wp.vec3(-0.5, 0.0, 0.0)),
        )
        j1 = builder.add_joint_revolute(
            parent=link_0,
            child=link_1,
            axis=wp.vec3(0.0, 1.0, 0.0),
            parent_xform=wp.transform(p=wp.vec3(0.5, 0.0, 0.0)),
            child_xform=wp.transform(p=wp.vec3(-0.5, 0.0, 0.0)),
        )
        builder.add_articulation([j0, j1], label="pendulum")
        self.model = builder.finalize()

        self.solver = newton.solvers.SolverMuJoCo(self.model)
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = None
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)

        # PID with all defaults: reads joint_q / joint_qd / joint_target_q /
        # joint_target_qd from the input struct and writes joint_f to the
        # output struct. Gains baked in as wp.arrays.
        default_dof_indices = wp.array([0, 1], dtype=wp.uint32, device=self.device)
        self.controller = ControllerPID(
            kp=wp.array([900.0, 300.0], dtype=wp.float32, device=self.device),
            kd=wp.array([80.0, 80.0], dtype=wp.float32, device=self.device),
            ki=wp.zeros(2, dtype=wp.float32, device=self.device),
            integral_max=wp.full(2, float("inf"), dtype=wp.float32, device=self.device),
            default_dof_indices=default_dof_indices,
            device=self.device,
        )

        # The input struct's joint_q / joint_qd / joint_target_q / joint_target_qd
        # fields default to wp.zeros; we rebind joint_q/qd each frame to the
        # current sim state, and seed the setpoint once.
        self._input = self.controller.input_struct()
        self._input.joint_target_q.assign([-wp.pi/2, 0.0])  # target angles [rad]

        # The output struct's joint_f field is what the solver consumes — wire it
        # straight into control.joint_f so the PID writes directly into the
        # arrays the solver reads.
        self._output = self.controller.output_struct()
        self._output.joint_f = self.control.joint_f

        # Double-buffered controller state — PID's integral state lives here.
        self._cs0 = self.controller.state()
        self._cs1 = self.controller.state()

        self.viewer.set_model(self.model)
        self.graph = None
        if self.device.is_cuda:
            with wp.ScopedCapture() as capture:
                self._simulate()
            self.graph = capture.graph

    def _simulate(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            # Rebind to whichever buffer the swap last left at state_0; SimpleNamespace
            # attribute assignment is a Python ref so this is free.
            self._input.joint_q = self.state_0.joint_q
            self._input.joint_qd = self.state_0.joint_qd
            self.controller.compute(self._input, self._output, self._cs0, self._cs1, time_step=self.sim_dt)
            self._cs0, self._cs1 = self._cs1, self._cs0
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self._simulate()
        self.sim_time += self.frame_dt

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()

    def test_final(self):
        # After enough sim time the PID should hold the joints near the targets.
        joint_q = self.state_0.joint_q.numpy()
        assert abs(joint_q[0] - 0.6) < 0.1, f"joint 0 = {joint_q[0]:.3f}, expected ~0.6"
        assert abs(joint_q[1] - -1.2) < 0.1, f"joint 1 = {joint_q[1]:.3f}, expected ~-1.2"


if __name__ == "__main__":
    viewer, args = newton.examples.init()
    newton.examples.run(Example(viewer, args), args)
