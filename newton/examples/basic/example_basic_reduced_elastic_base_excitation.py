# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Basic Reduced Elastic Base Excitation
#
# Demonstrates the floating-frame acceleration coupling for reduced elastic
# bodies: two identical cantilevers are clamped to a kinematic base that
# oscillates vertically. They share the same modal mass, stiffness, and mode
# shape; only the inertia coupling integral S_i differs.
#
#   - The "coupled" beam's basis carries per-sample masses, so the base
#     acceleration excites the mode through the modal force -S_i . a and the
#     beam bends as the base shakes.
#   - The "uncoupled" beam's basis has no sample mass (S_i = 0), so it rides the
#     base rigidly with no modal deformation.
#
# The clamp uses the cantilever tip mode (phi = 0 at the clamp), so the joint
# reaction projects to zero on the mode; the acceleration coupling is the only
# thing that bends the beam.
#
# Command: python -m newton.examples basic_reduced_elastic_base_excitation
#
###########################################################################

import math

import numpy as np
import warp as wp

import newton
import newton.examples
from newton.examples.basic._reduced_elastic import (
    beam_render_sample_points,
    cantilever_tip_mode,
    find_free_joint_q_start,
    set_camera_from_bounds,
)
from newton.examples.basic._reduced_elastic_contact import (
    apply_kinematic_targets,
    finite_difference_target_velocities,
)


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
        self.y_offset = 0.18
        self.base_amplitude = 0.08
        self.base_frequency = 1.5
        self.beam_mass = 1.0

        sample_points = beam_render_sample_points(
            self.length,
            self.hy,
            self.hz,
            extra_points=((-0.5 * self.length, 0.0, 0.0), (0.5 * self.length, 0.0, 0.0)),
        )
        phi = cantilever_tip_mode(sample_points, self.length)

        sample_mass = np.full(sample_points.shape[0], self.beam_mass / sample_points.shape[0], dtype=np.float32)
        modal_mass = float(np.sum(sample_mass * np.sum(phi * phi, axis=1)))
        mode_stiffness = 60.0
        mode_damping = 2.0 * 0.15 * math.sqrt(mode_stiffness * modal_mass)

        phi_3d = phi.reshape((-1, 1, 3))
        coupled_basis = newton.ModalBasis(
            sample_points=sample_points,
            sample_phi=phi_3d,
            sample_mass=sample_mass,
            mode_stiffness=[mode_stiffness],
            mode_damping=[mode_damping],
            label="base_excitation_coupled_basis",
        )
        uncoupled_basis = newton.ModalBasis(
            sample_points=sample_points,
            sample_phi=phi_3d,
            mode_mass=[modal_mass],
            mode_stiffness=[mode_stiffness],
            mode_damping=[mode_damping],
            label="base_excitation_uncoupled_basis",
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
            label="moving_base",
        )
        builder.add_shape_box(self.base, hx=0.05, hy=0.5 * self.length, hz=0.04, cfg=shape_cfg)

        self.beams = {}
        for name, basis, y in (
            ("coupled", coupled_basis, -self.y_offset),
            ("uncoupled", uncoupled_basis, self.y_offset),
        ):
            beam = builder.add_body_elastic(
                xform=wp.transform(wp.vec3(0.5 * self.length, y, self.base_height), wp.quat_identity()),
                mass=0.2,
                inertia=inertia,
                mode_q=[0.0],
                modal_basis=basis,
                label=f"base_excitation_{name}_beam",
            )
            builder.add_shape_box(beam, hx=0.5 * self.length, hy=self.hy, hz=self.hz, cfg=shape_cfg)
            builder.add_joint_fixed(
                parent=self.base,
                child=beam,
                parent_xform=wp.transform(wp.vec3(0.0, y, 0.0), wp.quat_identity()),
                child_xform=wp.transform(wp.vec3(-0.5 * self.length, 0.0, 0.0), wp.quat_identity()),
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

        self.max_abs_coupled_q = 0.0
        self.max_abs_uncoupled_q = 0.0

        self.solver = newton.solvers.SolverVBD(self.model, iterations=32)

        self.viewer.set_model(self.model)
        self.viewer.show_elastic_strain = True
        self.viewer.elastic_strain_color_max = 0.02
        bounds_min = np.array([-0.12, -self.y_offset - 0.1, self.base_height - self.base_amplitude - 0.15])
        bounds_max = np.array(
            [0.5 * self.length + 0.1, self.y_offset + 0.1, self.base_height + self.base_amplitude + 0.1]
        )
        set_camera_from_bounds(self.viewer, bounds_min, bounds_max, np.array([-0.4, -1.0, 0.45]))

    def _base_z(self, t: float) -> float:
        return self.base_height + self.base_amplitude * math.sin(2.0 * math.pi * self.base_frequency * t)

    def _drive_targets(self, t: float):
        return {self.base: (wp.vec3(0.0, 0.0, self._base_z(t)), wp.quat_identity())}

    def _mode_value(self, name: str) -> float:
        return float(self.state_0.joint_q.numpy()[self._q_index[name]])

    def simulate(self):
        for substep in range(self.sim_substeps):
            t = self.sim_time + substep * self.sim_dt
            targets = self._drive_targets(t)
            previous_targets = self._drive_targets(max(t - self.sim_dt, 0.0))
            velocities = finite_difference_target_velocities(targets, previous_targets, self.sim_dt)
            apply_kinematic_targets(self.state_0, self._base_q_starts, targets, velocities, self._base_qd_starts)
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        self.simulate()
        self.sim_time += self.frame_dt
        self.max_abs_coupled_q = max(self.max_abs_coupled_q, abs(self._mode_value("coupled")))
        self.max_abs_uncoupled_q = max(self.max_abs_uncoupled_q, abs(self._mode_value("uncoupled")))

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()

    def test_final(self):
        if not np.isfinite(self.state_0.joint_q.numpy()).all():
            raise AssertionError("joint coordinates contain non-finite values")
        if self.max_abs_coupled_q < 2.0e-2:
            raise AssertionError(f"coupled beam was not excited by base motion: max |q| = {self.max_abs_coupled_q:.3e}")
        if self.max_abs_uncoupled_q > 2.0e-3:
            raise AssertionError(f"uncoupled beam unexpectedly deflected: max |q| = {self.max_abs_uncoupled_q:.3e}")


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args)
    newton.examples.run(example, args)
