# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Basic Reduced Elastic Frame Coupling
#
# Demonstrates the floating-frame modal-to-frame inertia coupling (the
# back-reaction) for reduced elastic bodies: two identical beams float freely
# in zero gravity with nothing holding or driving them, each carrying a single
# transverse bump mode 1 - (2x/L)^2 that is deflected at t = 0 and released.
#
#   - The "coupled" beam's basis carries per-sample masses, so the modal
#     vibration pushes back on its frame through -A . S q_ddot. The frame
#     recoils against the bulge and the body center of mass stays pinned where
#     it started, conserving linear momentum.
#   - The "uncoupled" beam's basis has no sample mass (all coupling integrals
#     are zero), so its frame stays put while the bulge oscillates and the
#     center of mass sloshes back and forth, silently violating momentum.
#
# The bump mode is symmetric on a uniform bar, so it has net linear coupling
# but no net angular coupling: the recoil is a pure translation with no induced
# rotation, exactly the term the current back-coupling implements.
#
# Command: python -m newton.examples basic_reduced_elastic_frame_coupling
#
###########################################################################


import numpy as np
import warp as wp

import newton
import newton.examples
from newton.examples.basic._reduced_elastic import (
    FRAME_AXES,
    beam_render_sample_points,
    quat_rotate,
    set_camera_from_bounds,
)


class Example:
    def __init__(self, viewer, args):
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = 8
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0

        self.viewer = viewer
        self.args = args

        self.length = 1.0
        self.hy = 0.05
        self.hz = 0.04
        self.height = 1.0
        self.y_gap = 0.6
        self.beam_mass = 1.0
        self.deflection = 0.12
        mode_stiffness = 10.0

        centered_points = beam_render_sample_points(
            self.length,
            self.hy,
            self.hz,
            extra_points=((-0.5 * self.length, 0.0, 0.0), (0.5 * self.length, 0.0, 0.0)),
        )
        xs = centered_points[:, 0]
        phi = np.zeros_like(centered_points, dtype=np.float32)
        phi[:, 2] = 1.0 - (2.0 * xs / self.length) ** 2

        sample_mass = np.full(centered_points.shape[0], self.beam_mass / centered_points.shape[0], dtype=np.float32)
        modal_mass = float(np.sum(sample_mass * np.sum(phi * phi, axis=1)))

        phi_3d = phi.reshape((-1, 1, 3))
        coupled_basis = newton.ModalBasis(
            sample_points=centered_points,
            sample_phi=phi_3d,
            sample_mass=sample_mass,
            mode_stiffness=[mode_stiffness],
            mode_damping=[0.0],
            label="frame_coupling_coupled_basis",
        )
        uncoupled_basis = newton.ModalBasis(
            sample_points=centered_points,
            sample_phi=phi_3d,
            mode_mass=[modal_mass],
            mode_stiffness=[mode_stiffness],
            mode_damping=[0.0],
            label="frame_coupling_uncoupled_basis",
        )

        self.com_factor = float(coupled_basis.mode_coupling_linear[0][2]) / self.beam_mass

        builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        builder.add_ground_plane()

        inertia = wp.mat33(0.02, 0.0, 0.0, 0.0, 0.02, 0.0, 0.0, 0.0, 0.02)
        shape_cfg = newton.ModelBuilder.ShapeConfig()
        shape_cfg.density = 0.0
        shape_cfg.has_shape_collision = False
        shape_cfg.has_particle_collision = False

        half = 0.5 * self.length
        self.beams = {}
        for name, basis, y in (
            ("coupled", coupled_basis, -0.5 * self.y_gap),
            ("uncoupled", uncoupled_basis, 0.5 * self.y_gap),
        ):
            beam = builder.add_body_elastic(
                xform=wp.transform(wp.vec3(0.0, y, self.height), wp.quat_identity()),
                com=wp.vec3(0.0, 0.0, 0.0),
                mass=self.beam_mass,
                inertia=inertia,
                mode_q=[self.deflection],
                modal_basis=basis,
                label=f"frame_coupling_{name}_beam",
            )
            builder.add_shape_box(
                beam,
                xform=wp.transform_identity(),
                hx=half,
                hy=self.hy,
                hz=self.hz,
                cfg=shape_cfg,
            )
            self.beams[name] = beam

        builder.color()

        self.model = builder.finalize()
        self.device = self.model.device
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = None

        joint_q_start = self.model.joint_q_start.numpy()
        elastic_joint = self.model.elastic_joint.numpy()
        body_elastic_index = self.model.body_elastic_index.numpy()
        self._frame_z_index = {}
        self._q_index = {}
        for name, beam in self.beams.items():
            owner = int(elastic_joint[int(body_elastic_index[beam])])
            self._frame_z_index[name] = int(joint_q_start[owner]) + 2
            self._q_index[name] = int(joint_q_start[owner]) + 7

        self._frame_z0 = {name: self._frame_z(name) for name in self.beams}
        self._com0 = {name: self._com(name) for name in self.beams}
        self.max_frame_move = dict.fromkeys(self.beams, 0.0)
        self.max_com_drift = dict.fromkeys(self.beams, 0.0)

        self.solver = newton.solvers.SolverVBD(self.model, iterations=24)

        self.viewer.set_model(self.model)
        self.viewer.show_elastic_strain = True
        self.viewer.elastic_strain_color_max = 0.12
        half = 0.5 * self.length
        y_half = 0.5 * self.y_gap + self.hy
        bounds_min = np.array([-half - 0.1, -y_half - 0.1, self.height - 0.3])
        bounds_max = np.array([half + 0.1, y_half + 0.1, self.height + 0.3])
        set_camera_from_bounds(self.viewer, bounds_min, bounds_max, np.array([-0.4, -1.0, 0.45]))

    def _frame_z(self, name: str) -> float:
        return float(self.state_0.joint_q.numpy()[self._frame_z_index[name]])

    def _mode_value(self, name: str) -> float:
        return float(self.state_0.joint_q.numpy()[self._q_index[name]])

    def _com(self, name: str) -> float:
        return self._frame_z(name) + self.com_factor * self._mode_value(name)

    def simulate(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        self.simulate()
        self.sim_time += self.frame_dt
        for name in self.beams:
            self.max_frame_move[name] = max(self.max_frame_move[name], abs(self._frame_z(name) - self._frame_z0[name]))
            self.max_com_drift[name] = max(self.max_com_drift[name], abs(self._com(name) - self._com0[name]))

    def _log_frames(self):
        axis_len = 0.225
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
            width=0.015,
        )

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self._log_frames()
        self.viewer.end_frame()

    def test_final(self):
        if not np.isfinite(self.state_0.joint_q.numpy()).all():
            raise AssertionError("joint coordinates contain non-finite values")
        if self.max_frame_move["coupled"] < 0.02:
            raise AssertionError(f"coupled frame did not recoil: max move = {self.max_frame_move['coupled']:.3e}")
        if self.max_com_drift["coupled"] > 0.1 * self.max_frame_move["coupled"]:
            raise AssertionError(
                f"coupled center of mass drifted: {self.max_com_drift['coupled']:.3e} "
                f"vs frame move {self.max_frame_move['coupled']:.3e}"
            )
        if self.max_frame_move["uncoupled"] > 1.0e-3:
            raise AssertionError(
                f"uncoupled frame unexpectedly moved: max move = {self.max_frame_move['uncoupled']:.3e}"
            )
        if self.max_com_drift["uncoupled"] < 0.02:
            raise AssertionError(
                f"uncoupled center of mass should slosh: max drift = {self.max_com_drift['uncoupled']:.3e}"
            )


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args)
    newton.examples.run(example, args)
