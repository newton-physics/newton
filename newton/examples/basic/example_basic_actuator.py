# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Basic Actuator
#
# Demonstrates two actuator compositions:
# - A refrigerator door with position-dependent magnetic closing effort.
# - A three-detent switch with stable positions at -15, 0, and 15 degrees.
#
# Right-drag the refrigerator door to pull it. Use Left/Right to push the
# switch, or press 1/2/3 to select a detent through applied torque.
#
# Command: python -m newton.examples basic_actuator
#
###########################################################################

import math

import numpy as np
import warp as wp

import newton
import newton.examples
from newton.actuators import ClampingPositionBased, ControllerDetent, ControllerPD


@wp.kernel
def apply_external_efforts(
    joint_f: wp.array[float],
    dof_indices: wp.array[wp.uint32],
    external_efforts: wp.array[float],
):
    actuator_index = wp.tid()
    dof_index = dof_indices[actuator_index]
    joint_f[dof_index] = joint_f[dof_index] + external_efforts[actuator_index]


class Example:
    def __init__(self, viewer, args):
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = 8
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0
        self.frame = 0

        self.viewer = viewer
        self.args = args
        self.test_mode = args.test
        self.switch_detents = tuple(math.radians(angle) for angle in (-30.0, 0.0, 30.0))
        self.selected_detent: int | None = None
        self.visited_detents = [False, False, False]
        self.max_door_angle = 0.0

        builder = newton.ModelBuilder()
        static_cfg = builder.default_shape_cfg.copy()
        static_cfg.density = 0.0
        static_cfg.has_shape_collision = False

        self._build_refrigerator(builder, static_cfg)
        self._build_detent_switch(builder, static_cfg)

        self.model = builder.finalize()
        self.model.set_gravity((0.0, 0.0, 0.0))
        self.solver = newton.solvers.SolverMuJoCo(self.model, disable_contacts=True)
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.external_efforts = wp.zeros(2, dtype=wp.float32, device=self.model.device)
        self.external_dof_indices = wp.array(
            [self.door_dof_index, self.switch_dof_index], dtype=wp.uint32, device=self.model.device
        )

        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)
        self.state_1.assign(self.state_0)

        self.viewer.set_model(self.model)
        self.viewer.set_camera(wp.vec3(0.0, -5.2, 2.2), -7.0, 90.0)
        self.capture()

    def _build_refrigerator(self, builder, static_cfg):
        cabinet_center = wp.vec3(-1.1, 0.0, 1.25)
        builder.add_shape_box(
            body=-1,
            xform=wp.transform(p=cabinet_center, q=wp.quat_identity()),
            hx=0.52,
            hy=0.42,
            hz=0.78,
            cfg=static_cfg,
            color=wp.vec3(0.18, 0.20, 0.23),
        )

        door = builder.add_link(label="refrigerator_door")
        door_cfg = builder.default_shape_cfg.copy()
        door_cfg.density = 80.0
        door_cfg.has_shape_collision = False
        builder.add_shape_box(
            body=door,
            hx=0.45,
            hy=0.045,
            hz=0.72,
            cfg=door_cfg,
            color=wp.vec3(0.72, 0.77, 0.82),
        )
        builder.add_shape_capsule(
            body=door,
            xform=wp.transform(p=wp.vec3(0.30, -0.10, 0.0), q=wp.quat_identity()),
            radius=0.035,
            half_height=0.28,
            cfg=door_cfg,
            color=wp.vec3(0.12, 0.14, 0.16),
        )

        hinge = builder.add_joint_revolute(
            parent=-1,
            child=door,
            axis=wp.vec3(0.0, 0.0, -1.0),
            parent_xform=wp.transform(p=wp.vec3(-1.55, -0.48, 1.25), q=wp.quat_identity()),
            child_xform=wp.transform(p=wp.vec3(-0.45, 0.0, 0.0), q=wp.quat_identity()),
            limit_lower=0.0,
            limit_upper=math.radians(110.0),
            limit_ke=2.0e4,
            limit_kd=80.0,
            damping=2.0,
            armature=0.03,
            label="refrigerator_hinge",
        )
        builder.add_articulation([hinge], label="refrigerator")
        self.door_dof_index = builder.joint_qd_start[hinge]
        self.door_position_index = builder.joint_q_start[hinge]
        magnetic_range = tuple(math.radians(angle) for angle in (0.0, 3.0, 8.0, 15.0, 30.0, 110.0))
        builder.add_actuator(
            ControllerPD,
            index=self.door_dof_index,
            pos_index=self.door_position_index,
            kp=25.0,
            kd=3.0,
            clamping=[
                (
                    ClampingPositionBased,
                    {
                        "lookup_positions": magnetic_range,
                        "lookup_efforts": (4.0, 3.5, 2.2, 0.8, 0.0, 0.0),
                    },
                )
            ],
        )

    def _build_detent_switch(self, builder, static_cfg):
        pivot = wp.vec3(1.15, 0.05, 1.25)
        builder.add_shape_box(
            body=-1,
            xform=wp.transform(p=wp.vec3(1.15, 0.13, 1.25), q=wp.quat_identity()),
            hx=0.68,
            hy=0.08,
            hz=0.92,
            cfg=static_cfg,
            color=wp.vec3(0.16, 0.18, 0.20),
        )
        for angle in self.switch_detents:
            marker_position = pivot + wp.vec3(0.58 * math.sin(angle), -0.08, 0.58 * math.cos(angle))
            builder.add_shape_sphere(
                body=-1,
                xform=wp.transform(p=marker_position, q=wp.quat_identity()),
                radius=0.05,
                cfg=static_cfg,
                color=wp.vec3(0.86, 0.24, 0.17),
            )

        lever = builder.add_link(label="three_detent_lever")
        lever_cfg = builder.default_shape_cfg.copy()
        lever_cfg.density = 90.0
        lever_cfg.has_shape_collision = False
        builder.add_shape_capsule(
            body=lever,
            xform=wp.transform(p=wp.vec3(0.0, 0.0, 0.34), q=wp.quat_identity()),
            radius=0.045,
            half_height=0.34,
            cfg=lever_cfg,
            color=wp.vec3(0.88, 0.90, 0.92),
        )
        builder.add_shape_sphere(
            body=lever,
            xform=wp.transform(p=wp.vec3(0.0, 0.0, 0.72), q=wp.quat_identity()),
            radius=0.075,
            cfg=lever_cfg,
            color=wp.vec3(0.18, 0.48, 0.78),
        )

        hinge = builder.add_joint_revolute(
            parent=-1,
            child=lever,
            axis=newton.Axis.Y,
            parent_xform=wp.transform(p=pivot, q=wp.quat_identity()),
            child_xform=wp.transform_identity(),
            limit_lower=math.radians(-36.0),
            limit_upper=math.radians(36.0),
            limit_ke=2.0e4,
            limit_kd=50.0,
            damping=0.0,
            armature=0.02,
            label="three_detent_hinge",
        )
        builder.add_articulation([hinge], label="three_detent_switch")
        self.switch_dof_index = builder.joint_qd_start[hinge]
        self.switch_position_index = builder.joint_q_start[hinge]
        builder.joint_q[self.switch_position_index] = self.switch_detents[0]
        builder.add_actuator(
            ControllerDetent,
            index=self.switch_dof_index,
            pos_index=self.switch_position_index,
            detent_positions=self.switch_detents,
            holding_efforts=(2.0, 2.0, 2.0),
            breakaway_efforts=(3.5, 3.5),
            transition_width=math.radians(2.5),
            damping=4.0,
        )

    def capture(self):
        self.graph = None
        with wp.ScopedCapture() as capture:
            self.simulate()
        self.graph = capture.graph

    def simulate(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            self.control.joint_f.zero_()
            for actuator in self.model.actuators:
                actuator.step(self.state_0, self.control, dt=self.sim_dt)
            wp.launch(
                apply_external_efforts,
                dim=2,
                inputs=[self.control.joint_f, self.external_dof_indices, self.external_efforts],
                device=self.model.device,
            )
            self.solver.step(self.state_0, self.state_1, self.control, None, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def _test_external_efforts(self) -> tuple[float, float]:
        door_effort = 22.0 if 5 <= self.frame < 24 else 0.0
        if 30 <= self.frame < 90:
            door_position = float(self.state_0.joint_q.numpy()[self.door_position_index])
            door_velocity = float(self.state_0.joint_qd.numpy()[self.door_dof_index])
            staging_position = math.radians(8.0)
            if door_position > staging_position:
                door_effort = float(
                    np.clip(80.0 * (staging_position - door_position) - 20.0 * door_velocity, -50.0, 50.0)
                )
        switch_effort = 10.0 if 8 <= self.frame < 17 or 55 <= self.frame < 64 else 0.0
        return door_effort, switch_effort

    def _interactive_switch_effort(self) -> float:
        right = self.viewer.is_key_down("right") if hasattr(self.viewer, "is_key_down") else False
        left = self.viewer.is_key_down("left") if hasattr(self.viewer, "is_key_down") else False
        if right or left:
            self.selected_detent = None
            return 6.0 * (float(right) - float(left))

        for key, detent_index in (("1", 0), ("2", 1), ("3", 2)):
            if hasattr(self.viewer, "is_key_down") and self.viewer.is_key_down(key):
                self.selected_detent = detent_index

        if self.selected_detent is None:
            return 0.0
        position = float(self.state_0.joint_q.numpy()[self.switch_position_index])
        error = self.switch_detents[self.selected_detent] - position
        if abs(error) < math.radians(1.0):
            return 0.0
        return math.copysign(6.0, error)

    def step(self):
        if self.test_mode:
            door_effort, switch_effort = self._test_external_efforts()
        else:
            door_effort = 0.0
            switch_effort = self._interactive_switch_effort()
        self.external_efforts.assign([door_effort, switch_effort])

        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()
        self.frame += 1
        self.sim_time += self.frame_dt

    def test_post_step(self):
        """Track finite door motion and each stable switch detent."""
        joint_q = self.state_0.joint_q.numpy()
        joint_qd = self.state_0.joint_qd.numpy()
        door_position = float(joint_q[self.door_position_index])
        switch_position = float(joint_q[self.switch_position_index])
        switch_velocity = float(joint_qd[self.switch_dof_index])
        if not np.isfinite((door_position, switch_position, switch_velocity)).all():
            raise ValueError("Actuator example state became non-finite")
        self.max_door_angle = max(self.max_door_angle, door_position)
        for index, detent in enumerate(self.switch_detents):
            if abs(switch_position - detent) < math.radians(2.0) and abs(switch_velocity) < 0.2:
                self.visited_detents[index] = True

    def test_final(self):
        """Verify magnetic closure and all three stable switch positions."""
        final_door_position = float(self.state_0.joint_q.numpy()[self.door_position_index])
        if self.max_door_angle < math.radians(35.0):
            raise ValueError(f"Refrigerator door did not open far enough: {math.degrees(self.max_door_angle):.2f} deg")
        if final_door_position >= math.radians(3.0):
            raise ValueError(f"Refrigerator door did not magnetically close: {math.degrees(final_door_position):.2f} deg")
        if not all(self.visited_detents):
            raise ValueError(f"Switch did not settle at all detents: {self.visited_detents}")

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()


if __name__ == "__main__":
    viewer, args = newton.examples.init()
    newton.examples.run(Example(viewer, args), args)
