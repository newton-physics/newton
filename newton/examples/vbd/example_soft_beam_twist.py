# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Soft Beam Twist — 180° Twist Stability
#
# A vertical tetrahedral beam with its bottom face pinned and top face
# twisted 180° around the Z-axis over a linear ramp. The stable
# Neo-Hookean material handles this extreme rotation without inversion
# (unlike StVK). The test validates volume preservation and stability.
#
# Command: python -m newton.examples vbd.example_soft_beam_twist
#
###########################################################################

import numpy as np
import warp as wp

import newton
import newton.examples
from newton import ParticleFlags


def _compute_tet_volume(q: np.ndarray, tet_indices: np.ndarray) -> float:
    v0 = q[tet_indices[:, 0]]
    v1 = q[tet_indices[:, 1]]
    v2 = q[tet_indices[:, 2]]
    v3 = q[tet_indices[:, 3]]
    d1 = v1 - v0
    d2 = v2 - v0
    d3 = v3 - v0
    volumes = np.einsum("ij,ij->i", d1, np.cross(d2, d3)) / 6.0
    return float(np.sum(np.abs(volumes)))


class Example:
    DIM_X = 3
    DIM_Y = 3
    DIM_Z = 16
    CELL = 0.05
    DENSITY = 1000.0
    K_MU = 1.0e4
    K_LAMBDA = 1.0e4
    K_DAMP = 1.0e-3
    TWIST_ANGLE = np.pi  # 180°
    RAMP_FRAMES = 200

    def __init__(self, viewer, args=None):
        self.viewer = viewer
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = 5
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.iterations = 20
        self.sim_time = 0.0

        builder = newton.ModelBuilder()

        self.beam_height = self.DIM_Z * self.CELL
        self.beam_cx = self.DIM_X * self.CELL / 2.0
        self.beam_cy = self.DIM_Y * self.CELL / 2.0

        builder.add_soft_grid(
            pos=wp.vec3(0.0, 0.0, 1.0),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0, 0.0, 0.0),
            dim_x=self.DIM_X,
            dim_y=self.DIM_Y,
            dim_z=self.DIM_Z,
            cell_x=self.CELL,
            cell_y=self.CELL,
            cell_z=self.CELL,
            density=self.DENSITY,
            k_mu=self.K_MU,
            k_lambda=self.K_LAMBDA,
            k_damp=self.K_DAMP,
            fix_bottom=False,
        )

        builder.color()
        self.model = builder.finalize()
        self.model.set_gravity((0.0, 0.0, 0.0))

        q_np = self.model.particle_q.numpy()

        # Pin bottom-Z layer
        bot_z = 1.0
        self.bot_mask = np.abs(q_np[:, 2] - bot_z) < 1e-6
        # Pin top-Z layer (will be twisted)
        top_z = 1.0 + self.beam_height
        self.top_mask = np.abs(q_np[:, 2] - top_z) < 1e-6
        self.top_indices = np.where(self.top_mask)[0]

        flags = self.model.particle_flags.numpy()
        for i in np.where(self.bot_mask | self.top_mask)[0]:
            flags[i] = flags[i] & ~int(ParticleFlags.ACTIVE)
        self.model.particle_flags = wp.array(flags)

        # Store rest positions of top face (relative to beam center for rotation)
        self.top_rest = q_np[self.top_indices].copy()
        self.top_center_xy = np.array([self.beam_cx, self.beam_cy])

        self.solver = newton.solvers.SolverVBD(
            model=self.model,
            iterations=self.iterations,
            particle_enable_self_contact=False,
        )

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.model.contacts()

        self.tet_indices = self.model.tet_indices.numpy()
        self.rest_volume = _compute_tet_volume(q_np, self.tet_indices)

        self._frame_index = 0

        self.viewer.set_model(self.model)
        self.capture()

    def capture(self):
        if wp.get_device().is_cuda:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph
        else:
            self.graph = None

    def simulate(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            self.model.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def _apply_twist_ramp(self):
        if self._frame_index >= self.RAMP_FRAMES:
            angle = self.TWIST_ANGLE
        else:
            t = self._frame_index / self.RAMP_FRAMES
            angle = t * self.TWIST_ANGLE

        cos_a = np.cos(angle)
        sin_a = np.sin(angle)

        q = self.state_0.particle_q.numpy()
        for i, idx in enumerate(self.top_indices):
            rx = self.top_rest[i, 0] - self.top_center_xy[0]
            ry = self.top_rest[i, 1] - self.top_center_xy[1]
            q[idx, 0] = self.top_center_xy[0] + cos_a * rx - sin_a * ry
            q[idx, 1] = self.top_center_xy[1] + sin_a * rx + cos_a * ry
        self.state_0.particle_q.assign(q)

    def step(self):
        self._apply_twist_ramp()
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()
        self.sim_time += self.frame_dt
        self._frame_index += 1

    def test_final(self):
        newton.examples.test_particle_state(
            self.state_0,
            "particle velocities do not explode",
            lambda q, qd: wp.length(qd) < 50.0,
        )

        q = self.state_0.particle_q.numpy()

        # No NaN
        if np.any(np.isnan(q)):
            raise ValueError("NaN detected in particle positions")

        # Volume preservation
        final_volume = _compute_tet_volume(q, self.tet_indices)
        volume_ratio = final_volume / self.rest_volume
        if abs(volume_ratio - 1.0) > 0.10:
            raise ValueError(
                f"Volume not preserved under twist: ratio {volume_ratio:.3f} "
                f"(rest={self.rest_volume:.6f}, final={final_volume:.6f})"
            )

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    viewer, args = newton.examples.init(parser)
    example = Example(viewer=viewer, args=args)
    newton.examples.run(example, args)
