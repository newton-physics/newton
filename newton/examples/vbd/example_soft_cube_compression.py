# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Soft Cube Compression — Extreme Compression + Recovery
#
# A tetrahedral cube with its bottom face pinned. The top face is driven
# downward to 50% of the rest height over 150 frames, then released.
# The stable Neo-Hookean material should survive the compression
# without NaN or inversion, and recover toward the rest height after
# release.
#
# The test additionally drives a near-flat (90%) compression that inverts
# elements, then checks the cube springs back to its rest height and volume
# once released, exercising inversion recovery directly.
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


def _run_compression(
    compress_ratio: float,
    n_compress: int = 150,
    n_settle: int = 250,
    dim: int = 6,
    cell: float = 0.05,
    density: float = 1000.0,
    k_mu: float = 1.0e4,
    k_lambda: float = 1.0e4,
    k_damp: float = 1.0e-2,
    iterations: int = 30,
    substeps: int = 5,
) -> tuple[float, float, bool]:
    """Headless compress-then-release run used to probe extreme-deformation recovery.

    Drives the pinned cube's top face down to ``compress_ratio`` of the rest height,
    releases it, and lets the cube settle under zero gravity. Returns the recovered
    height ratio, the recovered volume ratio, and whether any NaN appeared.
    """
    builder = newton.ModelBuilder()
    cube_size = dim * cell
    base_z = 1.0
    builder.add_soft_grid(
        pos=wp.vec3(0.0, 0.0, base_z),
        rot=wp.quat_identity(),
        vel=wp.vec3(0.0, 0.0, 0.0),
        dim_x=dim,
        dim_y=dim,
        dim_z=dim,
        cell_x=cell,
        cell_y=cell,
        cell_z=cell,
        density=density,
        k_mu=k_mu,
        k_lambda=k_lambda,
        k_damp=k_damp,
    )
    builder.color()
    model = builder.finalize()
    model.set_gravity((0.0, 0.0, 0.0))

    q0 = model.particle_q.numpy()
    bot_mask = np.abs(q0[:, 2] - base_z) < 1e-6
    top_mask = np.abs(q0[:, 2] - (base_z + cube_size)) < 1e-6
    top_indices = np.where(top_mask)[0]
    flags = model.particle_flags.numpy()
    for i in np.where(bot_mask | top_mask)[0]:
        flags[i] = flags[i] & ~int(ParticleFlags.ACTIVE)
    model.particle_flags = wp.array(flags)

    solver = newton.solvers.SolverVBD(model=model, iterations=iterations, particle_enable_self_contact=False)
    s0 = model.state()
    s1 = model.state()
    ctrl = model.control()
    contacts = model.contacts()
    dt = 1.0 / 60 / substeps

    tet_indices = model.tet_indices.numpy()
    rest_volume = _compute_tet_volume(q0, tet_indices)

    # Drive the top face down to compress_ratio of the rest height.
    for f in range(n_compress):
        t = min(f / max(n_compress - 1, 1), 1.0)
        target_z = base_z + (1.0 - t * (1.0 - compress_ratio)) * cube_size
        q = s0.particle_q.numpy()
        q[top_indices, 2] = target_z
        s0.particle_q.assign(q)
        model.collide(s0, contacts)
        for _ in range(substeps):
            s0.clear_forces()
            solver.step(s0, s1, ctrl, contacts, dt)
            s0, s1 = s1, s0

    # Release the top face and let the cube spring back.
    flags = model.particle_flags.numpy()
    for i in top_indices:
        flags[i] = flags[i] | int(ParticleFlags.ACTIVE)
    model.particle_flags = wp.array(flags)
    for _ in range(n_settle):
        model.collide(s0, contacts)
        for _ in range(substeps):
            s0.clear_forces()
            solver.step(s0, s1, ctrl, contacts, dt)
            s0, s1 = s1, s0

    q = s0.particle_q.numpy()
    has_nan = bool(np.any(np.isnan(q)))
    recovered_height = (float(np.mean(q[top_indices, 2])) - base_z) / cube_size
    recovered_volume = _compute_tet_volume(q, tet_indices) / rest_volume
    return recovered_height, recovered_volume, has_nan


class Example:
    DIM = 6
    CELL = 0.05
    DENSITY = 1000.0
    K_MU = 1.0e4
    K_LAMBDA = 1.0e4
    K_DAMP = 1.0e-2
    COMPRESS_RATIO = 0.50
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
        self.model.collide(self.state_0, self.contacts)
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
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

        denom = max(self.COMPRESS_FRAMES - 1, 1)
        t = min(self._frame_index / denom, 1.0)
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

        # Extreme robustness: a near-flat (90%) compression inverts elements, and the
        # stable Neo-Hookean material should un-invert them so the cube springs back to
        # its rest height and volume once released.
        rec, vol, has_nan = _run_compression(compress_ratio=0.10)
        if has_nan:
            raise ValueError("NaN during near-flat (90%) compression recovery")
        if rec < 0.90:
            raise ValueError(f"Near-flat cube failed to recover height: {rec:.1%} of rest")
        if vol < 0.90:
            raise ValueError(f"Near-flat cube failed to recover volume: ratio {vol:.3f}")

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
