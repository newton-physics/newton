# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Basic Reduced Elastic Gravity Coupling
#
# Demonstrates the floating-frame gravity coupling for reduced elastic bodies:
# two identical horizontal cantilevers are clamped to the world and released
# under gravity with no tip load. They share the same modal mass, stiffness,
# and mode shape; the only difference is the inertia coupling integral S_i.
#
#   - The "coupled" beam's basis carries per-sample masses, so S_i is non-zero
#     and the modal gravity force S_i . g sags it under its own weight.
#   - The "uncoupled" beam's basis has no sample mass (S_i = 0), so gravity acts
#     only on its rigid floating frame (held by the clamp) and the mode stays
#     undeflected.
#
# The clamp uses the cantilever tip mode (phi = 0 at the clamp), so the joint
# reaction projects to zero on the mode: the only thing that can bend the beam
# is the gravity coupling. The two beams therefore isolate its effect.
#
# Command: python -m newton.examples basic_reduced_elastic_gravity_coupling
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
    set_camera_from_bounds,
)


class Example:
    def __init__(self, viewer, args):
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = 3
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0

        self.viewer = viewer
        self.args = args

        self.length = 0.8
        self.hy = 0.04
        self.hz = 0.03
        self.z = 0.6
        self.y_offset = 0.18
        self.gravity = 9.81
        self.beam_mass = 1.0
        self.expected_tip_deflection = 0.12
        self.expected_q = -self.expected_tip_deflection

        sample_points = beam_render_sample_points(
            self.length,
            self.hy,
            self.hz,
            extra_points=((-0.5 * self.length, 0.0, 0.0), (0.5 * self.length, 0.0, 0.0)),
        )
        phi = cantilever_tip_mode(sample_points, self.length)

        sample_mass = np.full(sample_points.shape[0], self.beam_mass / sample_points.shape[0], dtype=np.float32)
        coupling_z = float(np.sum(sample_mass * phi[:, 2]))
        modal_mass = float(np.sum(sample_mass * np.sum(phi * phi, axis=1)))
        mode_stiffness = coupling_z * self.gravity / self.expected_tip_deflection
        mode_damping = 2.0 * 0.7 * math.sqrt(mode_stiffness * modal_mass)

        phi_3d = phi.reshape((-1, 1, 3))
        coupled_basis = newton.ModalBasis(
            sample_points=sample_points,
            sample_phi=phi_3d,
            sample_mass=sample_mass,
            mode_stiffness=[mode_stiffness],
            mode_damping=[mode_damping],
            label="gravity_coupled_basis",
        )
        uncoupled_basis = newton.ModalBasis(
            sample_points=sample_points,
            sample_phi=phi_3d,
            mode_mass=[modal_mass],
            mode_stiffness=[mode_stiffness],
            mode_damping=[mode_damping],
            label="gravity_uncoupled_basis",
        )

        builder = newton.ModelBuilder(gravity=(0.0, 0.0, -self.gravity))
        builder.add_ground_plane()

        inertia = wp.mat33(0.01, 0.0, 0.0, 0.0, 0.01, 0.0, 0.0, 0.0, 0.01)
        shape_cfg = newton.ModelBuilder.ShapeConfig()
        shape_cfg.density = 0.0
        shape_cfg.has_shape_collision = False
        shape_cfg.has_particle_collision = False

        builder.add_shape_box(
            -1,
            xform=wp.transform(wp.vec3(-0.5 * self.length, 0.0, self.z), wp.quat_identity()),
            hx=0.05,
            hy=0.5 * self.length,
            hz=0.04,
            cfg=shape_cfg,
        )

        self.beams = {}
        for name, basis, y in (
            ("coupled", coupled_basis, -self.y_offset),
            ("uncoupled", uncoupled_basis, self.y_offset),
        ):
            beam = builder.add_body_elastic(
                xform=wp.transform(wp.vec3(0.0, y, self.z), wp.quat_identity()),
                mass=0.2,
                inertia=inertia,
                mode_q=[0.0],
                modal_basis=basis,
                label=f"gravity_{name}_beam",
            )
            builder.add_shape_box(beam, hx=0.5 * self.length, hy=self.hy, hz=self.hz, cfg=shape_cfg)
            builder.add_joint_fixed(
                parent=-1,
                child=beam,
                parent_xform=wp.transform(wp.vec3(-0.5 * self.length, y, self.z), wp.quat_identity()),
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

        elastic_joint = self.model.elastic_joint.numpy()
        body_elastic_index = self.model.body_elastic_index.numpy()
        joint_q_start = self.model.joint_q_start.numpy()
        self._q_index = {}
        for name, beam in self.beams.items():
            owner = int(elastic_joint[int(body_elastic_index[beam])])
            self._q_index[name] = int(joint_q_start[owner]) + 7

        self.min_coupled_q = 0.0
        self.max_abs_uncoupled_q = 0.0

        self.solver = newton.solvers.SolverVBD(
            self.model,
            iterations=24,
            rigid_joint_linear_k_start=1.0e5,
            rigid_joint_angular_k_start=1.0e5,
            rigid_joint_linear_ke=2.0e6,
            rigid_joint_angular_ke=2.0e6,
        )

        self.viewer.set_model(self.model)
        self.viewer.show_elastic_strain = True
        self.viewer.elastic_strain_color_max = 0.02
        bounds_min = np.array([-0.5 * self.length - 0.06, -self.y_offset - 0.1, self.z + 1.4 * self.expected_q - 0.08])
        bounds_max = np.array([0.5 * self.length + 0.1, self.y_offset + 0.1, self.z + 0.1])
        set_camera_from_bounds(self.viewer, bounds_min, bounds_max, np.array([-0.4, -1.0, 0.45]))

    def _mode_value(self, name: str) -> float:
        return float(self.state_0.joint_q.numpy()[self._q_index[name]])

    def simulate(self):
        for _substep in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        self.simulate()
        self.sim_time += self.frame_dt
        self.min_coupled_q = min(self.min_coupled_q, self._mode_value("coupled"))
        self.max_abs_uncoupled_q = max(self.max_abs_uncoupled_q, abs(self._mode_value("uncoupled")))

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()

    def test_final(self):
        if not np.isfinite(self.state_0.joint_q.numpy()).all():
            raise AssertionError("joint coordinates contain non-finite values")
        if abs(self._mode_value("coupled") - self.expected_q) > 1.5e-2:
            raise AssertionError(
                f"coupled beam deflection {self._mode_value('coupled'):.4f} differs from {self.expected_q:.4f}"
            )
        if self.max_abs_uncoupled_q > 2.0e-3:
            raise AssertionError(f"uncoupled beam unexpectedly deflected: max |q| = {self.max_abs_uncoupled_q:.3e}")


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args)
    newton.examples.run(example, args)
