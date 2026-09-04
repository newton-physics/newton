# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Basic Reduced Elastic Clamp Moment
#
# Demonstrates the joint rotational (clamp moment) coupling for reduced elastic
# bodies: a fixed attachment transmits the parent's rotation into modal
# deformation through the per-endpoint angular mode shape psi, and the modal
# deformation reacts the matching moment back on the parent.
#
# Two beams are clamped at their base to a fixed support and at their tip to a
# drum that spins about the beam axis. Each beam carries a single torsion mode
# and no sample mass, so the coupling is isolated from every other effect:
#
#   - Both clamp points lie on the beam centerline, where the translational
#     mode shape phi is zero, so the linear part of each clamp only pins the
#     floating frame and cannot drive the mode. The base clamp also sits at the
#     torsion node (psi = 0) and fully fixes the frame; the tip clamp sits where
#     psi is non-zero, so only its clamp moment can twist the beam.
#   - With no sample mass the floating-frame inertia coupling integrals are all
#     zero, so spinning the drum cannot twist the beam through inertia. Gravity
#     is off.
#
#   - "coupled": the basis carries psi, so the modal twist gives the frame an
#     extra degree of freedom that satisfies both clamps exactly: the base end
#     stays aligned with the support while the tip twists to follow the drum.
#   - "uncoupled": the same basis with psi zeroed, so the clamp moment cannot
#     reach the mode. With no twist available the beam can only rotate as a
#     rigid body, and the two clamps are soft penalties that pull its frame
#     toward the support and toward the drum respectively. They balance at the
#     midpoint, so the beam rigidly rotates to half the drum angle and stays
#     visibly misaligned with both ends instead of twisting.
#
# Command: python -m newton.examples basic_reduced_elastic_clamp_moment
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
        self.z = 0.6
        self.y_offset = 0.22
        self.drum_amplitude = 0.6
        self.drum_frequency = 0.3

        sample_points = beam_render_sample_points(
            self.length,
            self.hy,
            self.hz,
            extra_points=((-0.5 * self.length, 0.0, 0.0), (0.5 * self.length, 0.0, 0.0)),
        )
        generator = newton.ModalGeneratorBeam(
            self.length,
            self.hy,
            self.hz,
            mode_specs=[{"type": "torsion", "boundary": "linear", "order": 1}],
            young_modulus=1.0e6,
            density=1000.0,
            damping_ratio=0.1,
            label="clamp_moment_coupled_basis",
        )
        coupled_basis = generator.build(sample_points)
        uncoupled_basis = newton.ModalBasis(
            sample_points=coupled_basis.sample_points,
            sample_phi=coupled_basis.sample_phi,
            mode_mass=coupled_basis.mode_mass,
            mode_stiffness=coupled_basis.mode_stiffness,
            mode_damping=coupled_basis.mode_damping,
            label="clamp_moment_uncoupled_basis",
        )

        builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        builder.add_ground_plane()

        inertia = wp.mat33(0.01, 0.0, 0.0, 0.0, 0.01, 0.0, 0.0, 0.0, 0.01)
        shape_cfg = newton.ModelBuilder.ShapeConfig()
        shape_cfg.density = 0.0
        shape_cfg.has_shape_collision = False
        shape_cfg.has_particle_collision = False

        builder.add_shape_box(
            -1,
            xform=wp.transform(wp.vec3(-0.5 * self.length - 0.05, 0.0, self.z), wp.quat_identity()),
            hx=0.05,
            hy=0.5 * self.length,
            hz=0.05,
            cfg=shape_cfg,
        )

        self.drums = {}
        self.beams = {}
        for name, basis, y in (
            ("coupled", coupled_basis, -self.y_offset),
            ("uncoupled", uncoupled_basis, self.y_offset),
        ):
            drum = builder.add_body(
                xform=wp.transform(wp.vec3(0.5 * self.length + 0.05, y, self.z), wp.quat_identity()),
                mass=2.0,
                inertia=wp.mat33(0.05, 0.0, 0.0, 0.0, 0.05, 0.0, 0.0, 0.0, 0.05),
                is_kinematic=True,
                label=f"drum_{name}",
            )
            builder.add_shape_box(drum, hx=0.05, hy=0.05, hz=0.05, cfg=shape_cfg)

            beam = builder.add_body_elastic(
                xform=wp.transform(wp.vec3(0.0, y, self.z), wp.quat_identity()),
                mass=1.0,
                inertia=inertia,
                mode_q=[0.0],
                modal_basis=basis,
                label=f"clamp_moment_{name}_beam",
            )
            builder.add_shape_box(beam, hx=0.5 * self.length, hy=self.hy, hz=self.hz, cfg=shape_cfg)
            builder.add_joint_fixed(
                parent=-1,
                child=beam,
                parent_xform=wp.transform(wp.vec3(-0.5 * self.length, y, self.z), wp.quat_identity()),
                child_xform=wp.transform(wp.vec3(-0.5 * self.length, 0.0, 0.0), wp.quat_identity()),
                label=f"clamp_wall_{name}",
            )
            builder.add_joint_fixed(
                parent=drum,
                child=beam,
                parent_xform=wp.transform(wp.vec3(-0.05, 0.0, 0.0), wp.quat_identity()),
                child_xform=wp.transform(wp.vec3(0.5 * self.length, 0.0, 0.0), wp.quat_identity()),
                label=f"clamp_drum_{name}",
            )
            self.drums[name] = drum
            self.beams[name] = beam

        builder.color()

        self.model = builder.finalize()
        self.device = self.model.device
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = None

        self._drum_q_starts = {}
        self._drum_qd_starts = {}
        for drum in self.drums.values():
            q_start, qd_start = find_free_joint_q_start(self.model, drum)
            self._drum_q_starts[drum] = q_start
            self._drum_qd_starts[drum] = qd_start

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
        self.viewer.elastic_strain_color_max = 0.02
        bounds_min = np.array([-0.5 * self.length - 0.12, -self.y_offset - 0.14, self.z - 0.16])
        bounds_max = np.array([0.5 * self.length + 0.12, self.y_offset + 0.14, self.z + 0.16])
        set_camera_from_bounds(self.viewer, bounds_min, bounds_max, np.array([-0.5, -1.0, 0.4]))

    def _drum_angle(self, t: float) -> float:
        return self.drum_amplitude * math.sin(2.0 * math.pi * self.drum_frequency * t)

    def _drum_rate(self, t: float) -> float:
        return (
            self.drum_amplitude
            * 2.0
            * math.pi
            * self.drum_frequency
            * math.cos(2.0 * math.pi * self.drum_frequency * t)
        )

    def _drive_targets(self, t: float):
        quat = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), self._drum_angle(t))
        targets = {}
        for name, drum in self.drums.items():
            y = -self.y_offset if name == "coupled" else self.y_offset
            targets[drum] = (wp.vec3(0.5 * self.length + 0.05, y, self.z), quat)
        return targets

    def _drive_velocities(self, t: float):
        ang = wp.vec3(self._drum_rate(t), 0.0, 0.0)
        return {drum: (wp.vec3(0.0, 0.0, 0.0), ang) for drum in self.drums.values()}

    def _mode_value(self, name: str) -> float:
        return float(self.state_0.joint_q.numpy()[self._q_index[name]])

    def simulate(self):
        for substep in range(self.sim_substeps):
            t = self.sim_time + substep * self.sim_dt
            targets = self._drive_targets(t)
            velocities = self._drive_velocities(t)
            apply_kinematic_targets(self.state_0, self._drum_q_starts, targets, velocities, self._drum_qd_starts)
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
        if self.max_abs_q["coupled"] < 5.0e-2:
            raise AssertionError(
                f"coupled beam was not twisted by the clamp moment: max |q| = {self.max_abs_q['coupled']:.3e}"
            )
        if self.max_abs_q["uncoupled"] > 5.0e-3:
            raise AssertionError(f"uncoupled beam unexpectedly twisted: max |q| = {self.max_abs_q['uncoupled']:.3e}")


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args)
    newton.examples.run(example, args)
