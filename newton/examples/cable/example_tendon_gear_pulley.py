# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Tendon Gear Pulley
#
# Five-pulley block-and-tackle mechanism based on the one-to-four gear
# system in Muller et al.'s Cable Joints paper.  The route is a pure
# rolling-contact cable: fixed dead-end -> moving block pulley -> fixed
# block pulley -> moving block pulley -> fixed block pulley -> fixed
# redirect pulley -> free counterweight.
#
# Command: python -m newton.examples tendon_gear_pulley
#
###########################################################################

import numpy as np
import warp as wp

import newton
import newton.examples
from newton._src.sim.builder import Axis
from newton._src.sim.tendon import TendonLinkType
from newton.examples.cable.cable import assert_tendon_total_length, get_tendon_cable_lines


class Example:
    def __init__(self, viewer, args):
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = 16
        self.sim_dt = self.frame_dt / self.sim_substeps

        self.viewer = viewer
        self.args = args

        builder = newton.ModelBuilder(up_axis=Axis.Z, gravity=-9.81)

        self.r_large = 0.22
        self.r_small = 0.16
        self.r_redirect = 0.18
        pulley_half_height = 0.045
        pulley_mass = 0.50
        self.lower_block_mass = 2.80
        self.right_mass = 1.0
        self.tendon_compliance = 2.0e-6
        self.tendon_damping = 0.25

        pulley_shape_cfg = newton.ModelBuilder.ShapeConfig(
            density=0.0,
            has_shape_collision=False,
            collision_group=1,
        )
        frame_shape_cfg = newton.ModelBuilder.ShapeConfig(
            density=0.0,
            has_shape_collision=False,
        )
        weight_shape_cfg = newton.ModelBuilder.ShapeConfig(
            density=0.0,
            has_shape_collision=True,
            collision_group=2,
            mu=0.7,
            margin=0.01,
            gap=0.02,
        )

        def cylinder_inertia(mass, radius, half_height):
            inertia_y = 0.5 * mass * radius * radius
            inertia_xz = (1.0 / 12.0) * mass * (3.0 * radius * radius + (2.0 * half_height) ** 2)
            return wp.mat33(
                inertia_xz,
                0.0,
                0.0,
                0.0,
                inertia_y,
                0.0,
                0.0,
                0.0,
                inertia_xz,
            )

        def box_inertia(mass, hx, hy, hz):
            sx = 2.0 * hx
            sy = 2.0 * hy
            sz = 2.0 * hz
            return wp.mat33(
                (1.0 / 12.0) * mass * (sy * sy + sz * sz),
                0.0,
                0.0,
                0.0,
                (1.0 / 12.0) * mass * (sx * sx + sz * sz),
                0.0,
                0.0,
                0.0,
                (1.0 / 12.0) * mass * (sx * sx + sy * sy),
            )

        q_cyl = wp.quat(np.sin(np.pi / 4.0), 0.0, 0.0, np.cos(np.pi / 4.0))

        def add_pulley_body(pos, radius, label):
            body = builder.add_body(
                xform=wp.transform(p=pos, q=wp.quat_identity()),
                mass=pulley_mass,
                inertia=cylinder_inertia(pulley_mass, radius, pulley_half_height),
                lock_inertia=True,
                label=label,
            )
            builder.add_shape_cylinder(
                body,
                xform=wp.transform(q=q_cyl),
                radius=radius,
                half_height=pulley_half_height,
                cfg=pulley_shape_cfg,
                color=(0.58, 0.58, 0.56),
            )
            return body

        self.fixed_anchor_pos = wp.vec3(-0.70, 0.0, 4.36)
        self.upper_large_pos = wp.vec3(-0.66, 0.0, 4.18)
        self.upper_small_pos = wp.vec3(-0.48, 0.0, 3.64)
        self.redirect_pos = wp.vec3(0.92, 0.0, 4.02)
        self.lower_block_pos = wp.vec3(-0.58, 0.0, 1.46)
        self.moving_large_local = wp.vec3(-0.08, 0.0, 0.40)
        self.moving_small_local = wp.vec3(0.10, 0.0, 1.04)
        self.right_initial_pos = wp.vec3(1.10, 0.0, 2.82)

        anchor = builder.add_body(
            xform=wp.transform(p=self.fixed_anchor_pos, q=wp.quat_identity()),
            mass=0.0,
            is_kinematic=True,
            label="dead_end_anchor",
        )
        builder.add_shape_box(
            anchor,
            xform=wp.transform(p=wp.vec3(0.13, 0.12, -0.04)),
            hx=0.04,
            hy=0.03,
            hz=0.55,
            cfg=frame_shape_cfg,
            color=(0.55, 0.55, 0.55),
        )
        self.anchor_idx = anchor

        self.upper_large_idx = upper_large = add_pulley_body(self.upper_large_pos, self.r_large, "upper_large")
        self.upper_small_idx = upper_small = add_pulley_body(self.upper_small_pos, self.r_small, "upper_small")
        self.redirect_idx = redirect = add_pulley_body(self.redirect_pos, self.r_redirect, "redirect")

        self.lower_idx = lower = builder.add_link(
            xform=wp.transform(p=self.lower_block_pos, q=wp.quat_identity()),
            mass=self.lower_block_mass,
            inertia=box_inertia(self.lower_block_mass, 0.30, 0.12, 0.30),
            lock_inertia=True,
            label="lower_block_weight",
        )
        builder.add_shape_box(
            lower,
            xform=wp.transform(p=wp.vec3(0.0, 0.0, -0.52)),
            hx=0.30,
            hy=0.12,
            hz=0.24,
            cfg=weight_shape_cfg,
            color=(0.82, 0.82, 0.78),
        )
        builder.add_shape_box(
            lower,
            xform=wp.transform(p=wp.vec3(0.0, 0.0, 0.44)),
            hx=0.045,
            hy=0.05,
            hz=0.72,
            cfg=frame_shape_cfg,
            color=(0.55, 0.55, 0.55),
        )

        moving_large_pos = self.lower_block_pos + self.moving_large_local
        moving_small_pos = self.lower_block_pos + self.moving_small_local
        self.moving_large_idx = moving_large = add_pulley_body(moving_large_pos, self.r_large, "moving_large")
        self.moving_small_idx = moving_small = add_pulley_body(moving_small_pos, self.r_small, "moving_small")

        self.right_idx = right = builder.add_body(
            xform=wp.transform(p=self.right_initial_pos, q=wp.quat_identity()),
            mass=self.right_mass,
            inertia=box_inertia(self.right_mass, 0.14, 0.10, 0.12),
            lock_inertia=True,
            label="free_counterweight",
        )
        builder.add_shape_box(
            right,
            hx=0.14,
            hy=0.10,
            hz=0.12,
            cfg=weight_shape_cfg,
            color=(0.78, 0.78, 0.74),
        )

        Dof = newton.ModelBuilder.JointDofConfig
        planar_lin = [Dof(axis=Axis.X), Dof(axis=Axis.Z)]
        planar_ang = []

        j_lower = builder.add_joint_d6(
            parent=-1,
            child=lower,
            linear_axes=planar_lin,
            angular_axes=planar_ang,
            parent_xform=wp.transform(p=self.lower_block_pos, q=wp.quat_identity()),
            child_xform=wp.transform(),
            label="lower_block_planar",
        )
        j_moving_large = builder.add_joint_revolute(
            parent=lower,
            child=moving_large,
            axis=Axis.Y,
            parent_xform=wp.transform(p=self.moving_large_local),
            child_xform=wp.transform(),
            label="moving_large_y",
        )
        j_moving_small = builder.add_joint_revolute(
            parent=lower,
            child=moving_small,
            axis=Axis.Y,
            parent_xform=wp.transform(p=self.moving_small_local),
            child_xform=wp.transform(),
            label="moving_small_y",
        )
        j_upper_large = builder.add_joint_revolute(
            parent=-1,
            child=upper_large,
            axis=Axis.Y,
            parent_xform=wp.transform(p=self.upper_large_pos),
            child_xform=wp.transform(),
            label="upper_large_y",
        )
        j_upper_small = builder.add_joint_revolute(
            parent=-1,
            child=upper_small,
            axis=Axis.Y,
            parent_xform=wp.transform(p=self.upper_small_pos),
            child_xform=wp.transform(),
            label="upper_small_y",
        )
        j_redirect = builder.add_joint_revolute(
            parent=-1,
            child=redirect,
            axis=Axis.Y,
            parent_xform=wp.transform(p=self.redirect_pos),
            child_xform=wp.transform(),
            label="redirect_y",
        )

        builder.add_articulation([j_lower, j_moving_large, j_moving_small])
        builder.add_articulation([j_upper_large])
        builder.add_articulation([j_upper_small])
        builder.add_articulation([j_redirect])

        axis = (0.0, 1.0, 0.0)
        drive_mu = 1000.0
        builder.add_tendon()
        for body, link_type, radius, orientation, offset in [
            (anchor, TendonLinkType.ATTACHMENT, 0.0, 1, (0.0, 0.0, 0.0)),
            (moving_large, TendonLinkType.ROLLING, self.r_large, 1, (0.0, 0.0, 0.0)),
            (upper_large, TendonLinkType.ROLLING, self.r_large, 1, (0.0, 0.0, 0.0)),
            (moving_small, TendonLinkType.ROLLING, self.r_small, 1, (0.0, 0.0, 0.0)),
            (upper_small, TendonLinkType.ROLLING, self.r_small, 1, (0.0, 0.0, 0.0)),
            (redirect, TendonLinkType.ROLLING, self.r_redirect, 1, (0.0, 0.0, 0.0)),
            (right, TendonLinkType.ATTACHMENT, 0.0, 1, (0.0, 0.0, 0.12)),
        ]:
            builder.add_tendon_link(
                body=body,
                link_type=int(link_type),
                radius=radius,
                orientation=orientation,
                mu=drive_mu,
                offset=offset,
                axis=axis,
                compliance=self.tendon_compliance,
                damping=self.tendon_damping,
                rest_length=-1.0,
            )

        builder.add_ground_plane()
        self.model = builder.finalize()

        self.solver = newton.solvers.SolverXPBD(
            self.model,
            iterations=40,
            joint_linear_relaxation=0.60,
            rigid_contact_relaxation=0.30,
        )

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.model.contacts()

        body_q = self.state_0.body_q.numpy()
        self._initial_lower_z = float(body_q[self.lower_idx][2])
        self._initial_right_z = float(body_q[self.right_idx][2])
        self._initial_lower_x = float(body_q[self.lower_idx][0])
        self._initial_right_x = float(body_q[self.right_idx][0])
        self._pulley_indices = [
            self.moving_large_idx,
            self.upper_large_idx,
            self.moving_small_idx,
            self.upper_small_idx,
            self.redirect_idx,
        ]
        self._pulley_names = ["moving_large", "upper_large", "moving_small", "upper_small", "redirect"]
        self._pulley_radii = np.array([self.r_large, self.r_large, self.r_small, self.r_small, self.r_redirect])
        self._pulley_theta = np.zeros(len(self._pulley_indices), dtype=float)
        self._last_pulley_angles = None
        self._pulley_rotation_history = []
        self._lower_history = []
        self._right_history = []
        self._moving_mount_offsets = [
            np.array([self.moving_large_local[0], self.moving_large_local[1], self.moving_large_local[2]], dtype=float),
            np.array([self.moving_small_local[0], self.moving_small_local[1], self.moving_small_local[2]], dtype=float),
        ]
        self._fixed_pulley_positions = np.array(
            [body_q[self.upper_large_idx][:3], body_q[self.upper_small_idx][:3], body_q[self.redirect_idx][:3]],
            dtype=float,
        )
        self._axis_error_history = []
        self._direction_validation_frames = 150

        if self.viewer is not None:
            self.viewer.set_model(self.model)
            self.viewer.set_camera(pos=wp.vec3(0.0, -8.2, 3.0), pitch=2.0, yaw=90.0)
            if hasattr(self.viewer, "renderer"):
                self.viewer.renderer.show_wireframe_overlay = True

    def simulate(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            if self.viewer is not None:
                self.viewer.apply_forces(self.state_0)
            self.model.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        self.simulate()
        self.sim_time += self.frame_dt
        self._record_motion_sample()

    @staticmethod
    def _hinge_y_angle(q):
        return float(2.0 * np.arctan2(q[4], q[6]))

    @staticmethod
    def _angle_delta(prev_angle, angle):
        return float((angle - prev_angle + np.pi) % (2.0 * np.pi) - np.pi)

    def _record_motion_sample(self):
        body_q = self.state_0.body_q.numpy()
        angles = np.array([self._hinge_y_angle(body_q[idx]) for idx in self._pulley_indices], dtype=float)
        if self._last_pulley_angles is not None:
            for i, angle in enumerate(angles):
                self._pulley_theta[i] += self._angle_delta(self._last_pulley_angles[i], angle)
        self._last_pulley_angles = angles
        self._pulley_rotation_history.append(np.array(self._pulley_theta, copy=True))
        self._lower_history.append(np.array(body_q[self.lower_idx][:3], dtype=float))
        self._right_history.append(np.array(body_q[self.right_idx][:3], dtype=float))

        lower_pos = np.array(body_q[self.lower_idx][:3], dtype=float)
        moving_positions = np.array(
            [body_q[self.moving_large_idx][:3], body_q[self.moving_small_idx][:3]],
            dtype=float,
        )
        moving_expected = np.array([lower_pos + offset for offset in self._moving_mount_offsets], dtype=float)
        fixed_positions = np.array(
            [body_q[self.upper_large_idx][:3], body_q[self.upper_small_idx][:3], body_q[self.redirect_idx][:3]],
            dtype=float,
        )
        moving_error = np.linalg.norm(moving_positions - moving_expected, axis=1)
        fixed_error = np.linalg.norm(fixed_positions - self._fixed_pulley_positions, axis=1)
        self._axis_error_history.append(np.concatenate([moving_error, fixed_error]))

    def test_post_step(self):
        body_q = self.state_0.body_q.numpy()
        assert np.isfinite(body_q).all(), "Gear pulley produced non-finite body state"
        assert float(np.max(np.abs(body_q[:, :3]))) < 20.0, "Gear pulley body state became unbounded"

        if len(self._pulley_rotation_history) <= self._direction_validation_frames:
            assert_tendon_total_length(self, rel_tol=0.35)

        if self._axis_error_history:
            axis_error = self._axis_error_history[-1]
            assert np.isfinite(axis_error).all(), "Gear pulley axis/mount drift became non-finite"
            assert float(axis_error.max()) < 0.06, (
                f"Gear pulleys should stay on their revolute mounts: max_error={float(axis_error.max()):.4f}"
            )

        lower = body_q[self.lower_idx]
        right = body_q[self.right_idx]
        assert abs(float(lower[1])) < 1.0e-4 and abs(float(right[1])) < 1.0e-4, (
            f"Gear weights should remain in the cable plane: lower_y={float(lower[1]):.5f}, "
            f"right_y={float(right[1]):.5f}"
        )

    def test_final(self):
        if not self._pulley_rotation_history:
            self._record_motion_sample()

        sample = min(len(self._pulley_rotation_history) - 1, self._direction_validation_frames - 1)
        lower_positions = np.array(self._lower_history[: sample + 1])
        right_positions = np.array(self._right_history[: sample + 1])
        rotations = np.array(self._pulley_rotation_history[: sample + 1])
        axis_errors = np.array(self._axis_error_history[: sample + 1])
        assert np.isfinite(lower_positions).all(), "Non-finite lower-block motion in gear pulley"
        assert np.isfinite(right_positions).all(), "Non-finite counterweight motion in gear pulley"
        assert np.isfinite(rotations).all(), "Non-finite pulley rotation in gear pulley"
        assert np.isfinite(axis_errors).all(), "Non-finite pulley mount error in gear pulley"

        lower_rise = float(lower_positions[sample, 2]) - self._initial_lower_z
        right_drop = self._initial_right_z - float(right_positions[sample, 2])
        right_travel = float(np.linalg.norm(right_positions[sample] - right_positions[0]))
        lower_x_drift = abs(float(lower_positions[sample, 0]) - self._initial_lower_x)
        right_x_drift = abs(float(right_positions[sample, 0]) - self._initial_right_x)
        assert lower_rise > 0.05, f"Moving block should rise under the 4:1 gear balance: dz={lower_rise:.4f}"
        assert right_drop > 0.20, f"Light free counterweight should fall in the gear example: dz={right_drop:.4f}"

        travel_ratio = right_travel / max(lower_rise, 1.0e-6)
        assert 2.5 < travel_ratio < 5.5, (
            f"Free counterweight travel should be amplified by the moving block route: "
            f"lower_rise={lower_rise:.4f}, right_travel={right_travel:.4f}, ratio={travel_ratio:.3f}"
        )
        assert lower_x_drift < 0.15 and right_x_drift < 1.5, (
            f"Gear weights should remain bounded while the free load swings: "
            f"lower_x={lower_x_drift:.4f}, right_x={right_x_drift:.4f}"
        )
        right_inv_inertia = np.array(self.model.body_inv_inertia.numpy()[self.right_idx], dtype=float)
        right_inv_mass = float(self.model.body_inv_mass.numpy()[self.right_idx])
        assert right_inv_mass > 0.0 and float(np.linalg.norm(right_inv_inertia)) > 0.0, (
            f"Free counterweight should remain a free dynamic body, not a locked/guided link: "
            f"inv_mass={right_inv_mass:.4f}, inv_inertia_norm={float(np.linalg.norm(right_inv_inertia)):.4f}"
        )

        early_sample = min(len(rotations) - 1, 45)
        early_rim_travel = np.abs(rotations[early_sample] * self._pulley_radii)
        final_rim_travel = np.abs(rotations[sample] * self._pulley_radii)
        max_frame_rotation = float(np.max(np.abs(np.diff(rotations, axis=0))))
        inactive_early = [
            self._pulley_names[i] for i, travel in enumerate(early_rim_travel) if float(travel) <= 0.003
        ]
        inactive_final = [
            self._pulley_names[i] for i, travel in enumerate(final_rim_travel) if float(travel) <= 0.025
        ]
        assert not inactive_early, (
            "All five gear pulleys should be coupled by the same cable early in the motion; "
            f"inactive={inactive_early}, rim_travel={early_rim_travel}"
        )
        assert not inactive_final, (
            "All five gear pulleys should rotate over the validated prefix; "
            f"inactive={inactive_final}, rim_travel={final_rim_travel}"
        )
        assert max_frame_rotation < 0.40, (
            f"Gear pulleys should not spin up with unbounded per-frame rotation: "
            f"max_frame_rotation={max_frame_rotation:.4f} rad/frame"
        )

        max_step = max(
            float(np.max(np.linalg.norm(np.diff(lower_positions, axis=0), axis=1))),
            float(np.max(np.linalg.norm(np.diff(right_positions, axis=0), axis=1))),
        )
        assert max_step < 0.25, f"Gear pulley weights should not jump per frame: max_step={max_step:.4f}"
        assert float(axis_errors.max()) < 0.06, (
            f"Gear pulley revolute mount errors should stay bounded: max={float(axis_errors.max()):.4f}"
        )

    def render(self):
        if self.viewer is not None:
            self.viewer.begin_frame(self.sim_time)
            self.viewer.log_state(self.state_0)
            starts, ends = get_tendon_cable_lines(self.solver, self.model, self.state_0)
            self.viewer.log_lines("cable", starts, ends, colors=(0.95, 0.08, 0.06), width=0.008)
            self.viewer.end_frame()


if __name__ == "__main__":
    viewer, args = newton.examples.init()
    example = Example(viewer, args)
    newton.examples.run(example, args)
