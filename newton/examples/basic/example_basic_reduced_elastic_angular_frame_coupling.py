# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Basic Reduced Elastic Angular Frame Coupling
#
# Demonstrates the floating-frame modal-to-frame angular coupling for reduced
# elastic bodies: the rotational twin of the frame coupling example. Two
# identical beams float freely in zero gravity with nothing holding or driving
# them, each carrying a single antisymmetric bending mode (2x/L)(1 - (2x/L)^2)
# that deflects in y. The mode shape is odd, so its linear coupling S vanishes
# exactly -- there is no linear recoil -- but it carries angular momentum G qdot
# about z. Both modes are set oscillating from the rest shape by an initial
# modal velocity, with the body's total angular momentum kept at zero.
#
#   - The "coupled" beam's basis carries per-sample masses, so its moving mode
#     carries angular momentum; its frame is given the matching counter-rotation
#     so the total stays zero. As the mode oscillates the frame rocks back and
#     forth about z (the moment A (G + floor(c) S) qddot) and returns to rest --
#     the mode visibly turning the frame.
#   - The "uncoupled" beam's basis has no sample mass (all coupling integrals
#     are zero), so its mode carries no momentum and its frame needs no
#     counter-rotation: the mode wiggles while the frame stays put.
#
# This is the angular counterpart of the frame coupling example, where a
# symmetric mode makes the frame translate; here an antisymmetric mode makes the
# frame rotate.
#
# Command: python -m newton.examples basic_reduced_elastic_angular_frame_coupling
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
    quat_rotate,
    set_camera_from_bounds,
)


class Example:
    def __init__(self, viewer, args):
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = 16
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0

        self.viewer = viewer
        self.args = args

        self.length = 1.0
        self.hx = 0.5 * self.length
        self.hy = 0.04
        self.hz = 0.03
        self.height = 1.0
        self.z_gap = 0.5
        self.beam_mass = 1.0
        self.mode_amplitude = 0.25
        self.inertia_zz = 0.09
        mode_stiffness = 5.0

        centered_points = beam_render_sample_points(
            self.length,
            self.hy,
            self.hz,
            extra_points=((-0.5 * self.length, 0.0, 0.0), (0.5 * self.length, 0.0, 0.0)),
        )
        xi = 2.0 * centered_points[:, 0] / self.length
        phi = np.zeros_like(centered_points, dtype=np.float32)
        phi[:, 1] = xi - xi**3

        sample_mass = np.full(centered_points.shape[0], self.beam_mass / centered_points.shape[0], dtype=np.float32)
        modal_mass = float(np.sum(sample_mass * np.sum(phi * phi, axis=1)))

        phi_3d = phi.reshape((-1, 1, 3))
        coupled_basis = newton.ModalBasis(
            sample_points=centered_points,
            sample_phi=phi_3d,
            sample_mass=sample_mass,
            mode_stiffness=[mode_stiffness],
            mode_damping=[0.0],
            label="angular_frame_coupling_coupled_basis",
        )
        uncoupled_basis = newton.ModalBasis(
            sample_points=centered_points,
            sample_phi=phi_3d,
            mode_mass=[modal_mass],
            mode_stiffness=[mode_stiffness],
            mode_damping=[0.0],
            label="angular_frame_coupling_uncoupled_basis",
        )

        coupling_z = float(coupled_basis.mode_coupling_angular[0][2])
        coupled_modal_inertia = modal_mass - coupling_z**2 / self.inertia_zz
        self.mode_kick = {
            "uncoupled": self.mode_amplitude * math.sqrt(mode_stiffness / modal_mass),
            "coupled": self.mode_amplitude * math.sqrt(mode_stiffness / coupled_modal_inertia),
        }
        self.frame_counter_spin = coupling_z / self.inertia_zz * self.mode_kick["coupled"]

        builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        builder.add_ground_plane()

        inertia = wp.mat33(0.1, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, self.inertia_zz)
        shape_cfg = newton.ModelBuilder.ShapeConfig()
        shape_cfg.density = 0.0
        shape_cfg.has_shape_collision = False
        shape_cfg.has_particle_collision = False

        self.beams = {}
        for name, basis, z in (
            ("uncoupled", uncoupled_basis, self.height),
            ("coupled", coupled_basis, self.height + self.z_gap),
        ):
            beam = builder.add_body_elastic(
                xform=wp.transform(wp.vec3(0.0, 0.0, z), wp.quat_identity()),
                com=wp.vec3(0.0, 0.0, 0.0),
                mass=self.beam_mass,
                inertia=inertia,
                mode_q=[0.0],
                modal_basis=basis,
                label=f"angular_frame_coupling_{name}_beam",
            )
            builder.add_shape_box(
                beam,
                xform=wp.transform_identity(),
                hx=self.hx,
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
        joint_qd_start = self.model.joint_qd_start.numpy()
        elastic_joint = self.model.elastic_joint.numpy()
        body_elastic_index = self.model.body_elastic_index.numpy()
        self._q_index = {}
        qd0 = self.state_0.joint_qd.numpy()
        for name, beam in self.beams.items():
            owner = int(elastic_joint[int(body_elastic_index[beam])])
            self._q_index[name] = int(joint_q_start[owner]) + 7
            qd0[int(joint_qd_start[owner]) + 6] = self.mode_kick[name]
            if name == "coupled":
                qd0[int(joint_qd_start[owner]) + 5] = self.frame_counter_spin
        self.state_0.joint_qd.assign(qd0)

        self._origin0 = {name: self._origin(name) for name in self.beams}
        self.max_yaw = dict.fromkeys(self.beams, 0.0)
        self.max_origin_move = dict.fromkeys(self.beams, 0.0)
        self.max_mode_excursion = dict.fromkeys(self.beams, 0.0)

        self.solver = newton.solvers.SolverVBD(self.model, iterations=24)

        self.viewer.set_model(self.model)
        self.viewer.show_elastic_strain = True
        self.viewer.elastic_strain_color_max = 0.18
        reach = 0.5 * self.length + 0.2
        bounds_min = np.array([-reach, -reach, self.height - 0.2])
        bounds_max = np.array([reach, reach, self.height + self.z_gap + 0.2])
        set_camera_from_bounds(self.viewer, bounds_min, bounds_max, np.array([-0.4, -1.0, 0.45]))

    def _origin(self, name: str) -> np.ndarray:
        return np.array(self.state_0.body_q.numpy()[self.beams[name]][:3], dtype=np.float64)

    def _mode_value(self, name: str) -> float:
        return float(self.state_0.joint_q.numpy()[self._q_index[name]])

    def _yaw(self, name: str) -> float:
        bq = self.state_0.body_q.numpy()[self.beams[name]]
        return 2.0 * math.atan2(abs(float(bq[5])), abs(float(bq[6])))

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
            self.max_yaw[name] = max(self.max_yaw[name], self._yaw(name))
            self.max_origin_move[name] = max(
                self.max_origin_move[name], float(np.linalg.norm(self._origin(name) - self._origin0[name]))
            )
            self.max_mode_excursion[name] = max(self.max_mode_excursion[name], abs(self._mode_value(name)))

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
        if self.max_mode_excursion["coupled"] < 0.5 * self.mode_amplitude:
            raise AssertionError(f"coupled mode did not vibrate: max |q| = {self.max_mode_excursion['coupled']:.3e}")
        if self.max_yaw["coupled"] < 0.1:
            raise AssertionError(f"coupled frame did not rotate: max yaw = {self.max_yaw['coupled']:.3e}")
        if self.max_yaw["coupled"] > 0.4:
            raise AssertionError(f"coupled frame ran away instead of rocking: max yaw = {self.max_yaw['coupled']:.3e}")
        if self.max_yaw["uncoupled"] > 5.0e-3:
            raise AssertionError(f"uncoupled frame unexpectedly rotated: max yaw = {self.max_yaw['uncoupled']:.3e}")
        if self.max_origin_move["coupled"] > 5.0e-3:
            raise AssertionError(
                f"coupled frame unexpectedly recoiled: max move = {self.max_origin_move['coupled']:.3e}"
            )


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args)
    newton.examples.run(example, args)
