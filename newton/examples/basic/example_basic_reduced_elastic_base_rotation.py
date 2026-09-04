# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Basic Reduced Elastic Base Rotation
#
# Demonstrates the floating-frame rotational (Euler) acceleration coupling
# for reduced elastic bodies, and that it is invariant to the choice of
# reference-frame origin. Three identical cantilevers occupying the same
# world span are clamped to a kinematic base that oscillates in rotation
# about the y axis. They share the same modal mass, stiffness, and mode
# shape; only the placement of the body frame (and the inertia coupling
# integrals computed about it) differs.
#
#   - "coupled_clamp": the body frame sits at the clamp, on the rotation
#     axis, so its origin only rotates and never translates. The
#     translational coupling -S_lin . a stays near zero and the
#     angular-acceleration term -S_ang . alpha alone bends the beam.
#   - "coupled_mid": the body frame sits at the beam center, off the axis,
#     so its origin swings. Here -S_lin . a and -S_ang . alpha both
#     contribute -- but their sum reproduces the exact same physical
#     deformation as "coupled_clamp", because the per-point acceleration is
#     independent of how the frame is split. The two coupled beams track
#     each other to within a few percent.
#   - "uncoupled": no sample mass (all coupling integrals are zero), so it
#     rides the base rigidly with no modal deformation.
#
# As with the base-excitation example the clamp uses the cantilever tip
# mode (phi = 0 at the clamp), so the joint reaction projects to zero on the
# mode; the floating-frame inertia coupling is the only thing that bends the
# beam.
#
# Command: python -m newton.examples basic_reduced_elastic_base_rotation
#
###########################################################################

import math

import numpy as np
import warp as wp

import newton
import newton.examples
from newton.examples.basic._reduced_elastic import (
    FRAME_AXES,
    beam_render_sample_points,
    cantilever_tip_mode,
    find_free_joint_q_start,
    quat_rotate,
    set_camera_from_bounds,
)
from newton.examples.basic._reduced_elastic_contact import apply_kinematic_targets


class Example:
    def __init__(self, viewer, args):
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = 4
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0

        self.viewer = viewer
        self.args = args

        self.length = 0.8
        self.hy = 0.04
        self.hz = 0.03
        self.base_height = 0.8
        self.y_offset = 0.22
        self.base_amplitude = 0.25
        self.base_frequency = 1.5
        self.beam_mass = 1.0

        centered_points = beam_render_sample_points(
            self.length,
            self.hy,
            self.hz,
            extra_points=((-0.5 * self.length, 0.0, 0.0), (0.5 * self.length, 0.0, 0.0)),
        )
        phi = cantilever_tip_mode(centered_points, self.length)
        clamp_points = centered_points + np.array([0.5 * self.length, 0.0, 0.0], dtype=np.float32)

        sample_mass = np.full(centered_points.shape[0], self.beam_mass / centered_points.shape[0], dtype=np.float32)
        modal_mass = float(np.sum(sample_mass * np.sum(phi * phi, axis=1)))
        mode_stiffness = 60.0
        mode_damping = 2.0 * 0.15 * math.sqrt(mode_stiffness * modal_mass)

        phi_3d = phi.reshape((-1, 1, 3))
        coupled_clamp_basis = newton.ModalBasis(
            sample_points=clamp_points,
            sample_phi=phi_3d,
            sample_mass=sample_mass,
            mode_stiffness=[mode_stiffness],
            mode_damping=[mode_damping],
            label="base_rotation_coupled_clamp_basis",
        )
        coupled_mid_basis = newton.ModalBasis(
            sample_points=centered_points,
            sample_phi=phi_3d,
            sample_mass=sample_mass,
            mode_stiffness=[mode_stiffness],
            mode_damping=[mode_damping],
            label="base_rotation_coupled_mid_basis",
        )
        uncoupled_basis = newton.ModalBasis(
            sample_points=clamp_points,
            sample_phi=phi_3d,
            mode_mass=[modal_mass],
            mode_stiffness=[mode_stiffness],
            mode_damping=[mode_damping],
            label="base_rotation_uncoupled_basis",
        )

        builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        builder.add_ground_plane()

        inertia = wp.mat33(0.01, 0.0, 0.0, 0.0, 0.01, 0.0, 0.0, 0.0, 0.01)
        shape_cfg = newton.ModelBuilder.ShapeConfig()
        shape_cfg.density = 0.0
        shape_cfg.has_shape_collision = False
        shape_cfg.has_particle_collision = False

        self.base = builder.add_body(
            xform=wp.transform(wp.vec3(0.0, 0.0, self.base_height), wp.quat_identity()),
            mass=2.0,
            inertia=wp.mat33(0.05, 0.0, 0.0, 0.0, 0.05, 0.0, 0.0, 0.0, 0.05),
            is_kinematic=True,
            label="rocking_base",
        )
        builder.add_shape_box(self.base, hx=0.05, hy=0.5 * self.length, hz=0.05, cfg=shape_cfg)

        half = 0.5 * self.length
        self.beams = {}
        for name, basis, y, origin_x in (
            ("coupled_clamp", coupled_clamp_basis, -self.y_offset, 0.0),
            ("coupled_mid", coupled_mid_basis, 0.0, half),
            ("uncoupled", uncoupled_basis, self.y_offset, 0.0),
        ):
            beam = builder.add_body_elastic(
                xform=wp.transform(wp.vec3(origin_x, y, self.base_height), wp.quat_identity()),
                com=wp.vec3(half - origin_x, 0.0, 0.0),
                mass=0.2,
                inertia=inertia,
                mode_q=[0.0],
                modal_basis=basis,
                label=f"base_rotation_{name}_beam",
            )
            builder.add_shape_box(
                beam,
                xform=wp.transform(wp.vec3(half - origin_x, 0.0, 0.0), wp.quat_identity()),
                hx=half,
                hy=self.hy,
                hz=self.hz,
                cfg=shape_cfg,
            )
            builder.add_joint_fixed(
                parent=self.base,
                child=beam,
                parent_xform=wp.transform(wp.vec3(0.0, y, 0.0), wp.quat_identity()),
                child_xform=wp.transform(wp.vec3(-origin_x, 0.0, 0.0), wp.quat_identity()),
                label=f"clamp_{name}",
            )
            self.beams[name] = beam

        builder.color()

        self.model = builder.finalize()
        self.device = self.model.device
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = None

        base_q_start, base_qd_start = find_free_joint_q_start(self.model, self.base)
        self._base_q_starts = {self.base: base_q_start}
        self._base_qd_starts = {self.base: base_qd_start}

        elastic_joint = self.model.elastic_joint.numpy()
        body_elastic_index = self.model.body_elastic_index.numpy()
        joint_q_start = self.model.joint_q_start.numpy()
        self._q_index = {}
        for name, beam in self.beams.items():
            owner = int(elastic_joint[int(body_elastic_index[beam])])
            self._q_index[name] = int(joint_q_start[owner]) + 7

        self.max_abs_q = dict.fromkeys(self.beams, 0.0)
        self.max_abs_invariance_error = 0.0

        self.solver = newton.solvers.SolverVBD(self.model, iterations=32)

        self.viewer.set_model(self.model)
        self.viewer.show_elastic_strain = True
        self.viewer.elastic_strain_color_max = 0.02
        swing = self.length * math.sin(self.base_amplitude)
        bounds_min = np.array([-0.15, -self.y_offset - 0.1, self.base_height - swing - 0.15])
        bounds_max = np.array([self.length + 0.15, self.y_offset + 0.1, self.base_height + swing + 0.15])
        set_camera_from_bounds(self.viewer, bounds_min, bounds_max, np.array([-0.4, -1.0, 0.45]))

    def _base_angle(self, t: float) -> float:
        return self.base_amplitude * math.sin(2.0 * math.pi * self.base_frequency * t)

    def _base_rate(self, t: float) -> float:
        return (
            self.base_amplitude
            * 2.0
            * math.pi
            * self.base_frequency
            * math.cos(2.0 * math.pi * self.base_frequency * t)
        )

    def _drive_targets(self, t: float):
        quat = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), self._base_angle(t))
        return {self.base: (wp.vec3(0.0, 0.0, self.base_height), quat)}

    def _drive_velocities(self, t: float):
        return {self.base: (wp.vec3(0.0, 0.0, 0.0), wp.vec3(0.0, self._base_rate(t), 0.0))}

    def _mode_value(self, name: str) -> float:
        return float(self.state_0.joint_q.numpy()[self._q_index[name]])

    def simulate(self):
        for substep in range(self.sim_substeps):
            t = self.sim_time + substep * self.sim_dt
            targets = self._drive_targets(t)
            velocities = self._drive_velocities(t)
            apply_kinematic_targets(self.state_0, self._base_q_starts, targets, velocities, self._base_qd_starts)
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        self.simulate()
        self.sim_time += self.frame_dt
        q = {name: self._mode_value(name) for name in self.beams}
        for name, value in q.items():
            self.max_abs_q[name] = max(self.max_abs_q[name], abs(value))
        self.max_abs_invariance_error = max(self.max_abs_invariance_error, abs(q["coupled_clamp"] - q["coupled_mid"]))

    def _log_frames(self):
        axis_len = 0.2
        starts = []
        ends = []
        colors = []
        for beam in self.beams.values():
            bq = self.state_0.body_q.numpy()[beam]
            origin = np.array(bq[:3], dtype=np.float32)
            for axis, color in FRAME_AXES:
                starts.append(origin)
                ends.append((origin + quat_rotate(bq[3:7], axis) * axis_len).astype(np.float32))
                colors.append(color)
        self.viewer.log_lines(
            "frames",
            wp.array(np.array(starts, dtype=np.float32), dtype=wp.vec3),
            wp.array(np.array(ends, dtype=np.float32), dtype=wp.vec3),
            wp.array(np.array(colors, dtype=np.float32), dtype=wp.vec3),
            width=0.012,
        )

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self._log_frames()
        self.viewer.end_frame()

    def test_final(self):
        if not np.isfinite(self.state_0.joint_q.numpy()).all():
            raise AssertionError("joint coordinates contain non-finite values")
        for name in ("coupled_clamp", "coupled_mid"):
            if self.max_abs_q[name] < 2.0e-2:
                raise AssertionError(
                    f"{name} beam was not excited by base rotation: max |q| = {self.max_abs_q[name]:.3e}"
                )
        if self.max_abs_q["uncoupled"] > 2.0e-3:
            raise AssertionError(f"uncoupled beam unexpectedly deflected: max |q| = {self.max_abs_q['uncoupled']:.3e}")
        reference = max(self.max_abs_q["coupled_clamp"], self.max_abs_q["coupled_mid"])
        if self.max_abs_invariance_error > 0.05 * reference:
            raise AssertionError(
                f"deformation depends on frame placement: max |q_clamp - q_mid| = "
                f"{self.max_abs_invariance_error:.3e} vs max |q| = {reference:.3e}"
            )


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args)
    newton.examples.run(example, args)
