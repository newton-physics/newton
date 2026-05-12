# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Soft Cube Compression — Extreme Compression + Recovery
#
# A tetrahedral cube with its bottom face pinned. The top face is driven
# downward to 10% of the rest height over 150 frames, then released.
# The stable Neo-Hookean material should survive the near-planar
# compression without NaN or inversion, and recover toward the rest
# height after release.
#
# Command: python -m newton.examples vbd.example_soft_cube_compression
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
    DIM = 6
    CELL = 0.05
    DENSITY = 1000.0
    K_MU = 1.0e4
    K_LAMBDA = 1.0e4
    K_DAMP = 1.0e-2
    COMPRESS_RATIO = 0.10
    COMPRESS_FRAMES = 150
    RELEASE_FRAME = 150

    def __init__(self, viewer, args=None):
        self.viewer = viewer
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = 5
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.iterations = 30
        self.sim_time = 0.0

        builder = newton.ModelBuilder()

        self.cube_size = self.DIM * self.CELL
        self.base_z = 1.0

        builder.add_soft_grid(
            pos=wp.vec3(0.0, 0.0, self.base_z),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0, 0.0, 0.0),
            dim_x=self.DIM,
            dim_y=self.DIM,
            dim_z=self.DIM,
            cell_x=self.CELL,
            cell_y=self.CELL,
            cell_z=self.CELL,
            density=self.DENSITY,
            k_mu=self.K_MU,
            k_lambda=self.K_LAMBDA,
            k_damp=self.K_DAMP,
        )

        builder.color()
        self.model = builder.finalize()
        self.model.set_gravity((0.0, 0.0, 0.0))

        q_np = self.model.particle_q.numpy()

        # Pin bottom-Z layer
        bot_mask = np.abs(q_np[:, 2] - self.base_z) < 1e-6
        # Top-Z layer (driven then released)
        top_z = self.base_z + self.cube_size
        self.top_mask = np.abs(q_np[:, 2] - top_z) < 1e-6
        self.top_indices = np.where(self.top_mask)[0]
        self.top_rest_z = top_z

        flags = self.model.particle_flags.numpy()
        for i in np.where(bot_mask | self.top_mask)[0]:
            flags[i] = flags[i] & ~int(ParticleFlags.ACTIVE)
        self.model.particle_flags = wp.array(flags)

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
        self._released = False

        self.viewer.set_model(self.model)

    def simulate(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            self.model.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def _apply_compression(self):
        if self._frame_index >= self.RELEASE_FRAME:
            if not self._released:
                flags = self.model.particle_flags.numpy()
                for i in self.top_indices:
                    flags[i] = flags[i] | int(ParticleFlags.ACTIVE)
                self.model.particle_flags = wp.array(flags)
                self._released = True
            return

        t = self._frame_index / self.COMPRESS_FRAMES
        target_ratio = 1.0 - t * (1.0 - self.COMPRESS_RATIO)
        target_z = self.base_z + target_ratio * self.cube_size

        q = self.state_0.particle_q.numpy()
        q[self.top_indices, 2] = target_z
        self.state_0.particle_q.assign(q)

    def step(self):
        self._apply_compression()
        self.simulate()
        self.sim_time += self.frame_dt
        self._frame_index += 1

    def test_final(self):
        q = self.state_0.particle_q.numpy()

        if np.any(np.isnan(q)):
            raise ValueError("NaN detected in particle positions")

        newton.examples.test_particle_state(
            self.state_0,
            "particle velocities do not explode",
            lambda q, qd: wp.length(qd) < 100.0,
        )

        # Height recovery: top particles should recover to >50% of rest height
        top_z_final = float(np.mean(q[self.top_indices, 2]))
        recovered_height = top_z_final - self.base_z
        recovery_ratio = recovered_height / self.cube_size
        if recovery_ratio < 0.50:
            raise ValueError(
                f"Insufficient height recovery: {recovery_ratio:.1%} of rest "
                f"(top_z={top_z_final:.4f}, base={self.base_z}, rest_height={self.cube_size})"
            )

        # Volume should partially recover
        final_volume = _compute_tet_volume(q, self.tet_indices)
        volume_ratio = final_volume / self.rest_volume
        if volume_ratio < 0.50:
            raise ValueError(f"Volume collapsed: ratio {volume_ratio:.3f}")

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
