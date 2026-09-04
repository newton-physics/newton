# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Basic Reduced Elastic Centrifugal
#
# Demonstrates the floating-frame centrifugal coupling for reduced elastic
# bodies: two identical beams are clamped to a kinematic hub that spins at a
# constant rate about the vertical z axis, each carrying a single axial
# (stretch) mode. They share the same modal mass, stiffness, and mode shape;
# only the inertia coupling integrals differ.
#
#   - The "coupled" beam's basis carries per-sample masses, so the steady
#     rotation drives the mode through the centrifugal force -floor(omega)^2 . M
#     and the beam stretches radially outward until stiffness balances it.
#   - The "uncoupled" beam's basis has no sample mass (all coupling integrals
#     are zero), so it spins rigidly with no stretch.
#
# Each beam's frame sits at its clamp on the spin axis, so the frame origin
# only rotates (no translational acceleration), the spin is steady (no Euler
# term), and a single mode has no Coriolis coupling -- leaving the centrifugal
# term as the only thing that deforms the beam.
#
# Command: python -m newton.examples basic_reduced_elastic_centrifugal
#
###########################################################################

import math

import numpy as np
import warp as wp

import newton
import newton.examples
from newton.examples.basic._reduced_elastic import (
    beam_render_sample_points,
    find_free_joint_q_start,
    poisson_axial_mode,
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
        self.z_gap = 0.18
        self.spin_rate = 5.0
        self.ramp_time = 1.0
        self.beam_mass = 1.0
        self.poisson = 0.3

        centered_points = beam_render_sample_points(
            self.length,
            self.hy,
            self.hz,
            extra_points=((-0.5 * self.length, 0.0, 0.0), (0.5 * self.length, 0.0, 0.0)),
        )
        clamp_points = centered_points + np.array([0.5 * self.length, 0.0, 0.0], dtype=np.float32)
        phi = poisson_axial_mode(clamp_points, self.length, self.poisson)

        sample_mass = np.full(clamp_points.shape[0], self.beam_mass / clamp_points.shape[0], dtype=np.float32)
        modal_mass = float(np.sum(sample_mass * np.sum(phi * phi, axis=1)))
        mode_stiffness = 60.0
        mode_damping = 2.0 * 0.2 * math.sqrt(mode_stiffness * modal_mass)

        phi_3d = phi.reshape((-1, 1, 3))
        coupled_basis = newton.ModalBasis(
            sample_points=clamp_points,
            sample_phi=phi_3d,
            sample_mass=sample_mass,
            mode_stiffness=[mode_stiffness],
            mode_damping=[mode_damping],
            label="centrifugal_coupled_basis",
        )
        uncoupled_basis = newton.ModalBasis(
            sample_points=clamp_points,
            sample_phi=phi_3d,
            mode_mass=[modal_mass],
            mode_stiffness=[mode_stiffness],
            mode_damping=[mode_damping],
            label="centrifugal_uncoupled_basis",
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
            label="spinning_base",
        )
        builder.add_shape_box(
            self.base,
            xform=wp.transform(wp.vec3(0.0, 0.0, 0.5 * self.z_gap), wp.quat_identity()),
            hx=0.06,
            hy=0.06,
            hz=0.5 * self.z_gap + 0.04,
            cfg=shape_cfg,
        )

        half = 0.5 * self.length
        self.beams = {}
        for name, basis, z in (
            ("coupled", coupled_basis, 0.0),
            ("uncoupled", uncoupled_basis, self.z_gap),
        ):
            beam = builder.add_body_elastic(
                xform=wp.transform(wp.vec3(0.0, 0.0, self.base_height + z), wp.quat_identity()),
                com=wp.vec3(half, 0.0, 0.0),
                mass=0.2,
                inertia=inertia,
                mode_q=[0.0],
                modal_basis=basis,
                label=f"centrifugal_{name}_beam",
            )
            builder.add_shape_box(
                beam,
                xform=wp.transform(wp.vec3(half, 0.0, 0.0), wp.quat_identity()),
                hx=half,
                hy=self.hy,
                hz=self.hz,
                cfg=shape_cfg,
            )
            builder.add_joint_fixed(
                parent=self.base,
                child=beam,
                parent_xform=wp.transform(wp.vec3(0.0, 0.0, z), wp.quat_identity()),
                child_xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
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

        self.solver = newton.solvers.SolverVBD(self.model, iterations=32)

        self.viewer.set_model(self.model)
        self.viewer.show_elastic_strain = True
        self.viewer.elastic_strain_color_max = 0.18
        reach = self.length + 0.05
        bounds_min = np.array([-reach, -reach, self.base_height - 0.2])
        bounds_max = np.array([reach, reach, self.base_height + self.z_gap + 0.2])
        set_camera_from_bounds(self.viewer, bounds_min, bounds_max, np.array([-0.4, -1.0, 0.45]))

    def _spin_rate_at(self, t: float) -> float:
        return self.spin_rate * min(t / self.ramp_time, 1.0)

    def _spin_angle(self, t: float) -> float:
        if t < self.ramp_time:
            return self.spin_rate * t * t / (2.0 * self.ramp_time)
        return self.spin_rate * (t - 0.5 * self.ramp_time)

    def _drive_targets(self, t: float):
        quat = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), self._spin_angle(t))
        return {self.base: (wp.vec3(0.0, 0.0, self.base_height), quat)}

    def _drive_velocities(self, t: float):
        return {self.base: (wp.vec3(0.0, 0.0, 0.0), wp.vec3(0.0, 0.0, self._spin_rate_at(t)))}

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
        for name in self.beams:
            self.max_abs_q[name] = max(self.max_abs_q[name], abs(self._mode_value(name)))

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()

    def test_final(self):
        if not np.isfinite(self.state_0.joint_q.numpy()).all():
            raise AssertionError("joint coordinates contain non-finite values")
        if self.max_abs_q["coupled"] < 2.0e-2:
            raise AssertionError(f"coupled beam did not stretch under spin: max |q| = {self.max_abs_q['coupled']:.3e}")
        if self._mode_value("coupled") <= 0.0:
            raise AssertionError("coupled beam should stretch outward (positive q) under centrifugal load")
        if self.max_abs_q["uncoupled"] > 2.0e-3:
            raise AssertionError(f"uncoupled beam unexpectedly deformed: max |q| = {self.max_abs_q['uncoupled']:.3e}")


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args)
    newton.examples.run(example, args)
