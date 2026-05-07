# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example ChysX Hanging Cloth
#
# A flat cloth patch is hung from one corner and let drape under gravity,
# integrated by the toy CUDA backend in `ChysX/` (the same engine that
# powers `chysx_freefall`).  This example exercises the full
# implicit-Euler / PCG path:
#
#   * `PinConstraint`  pins particle 0 (a grid corner) with a large
#                      penalty stiffness; the resulting `k I` diagonal
#                      block dominates that particle's row so the PCG
#                      solve effectively clamps it in place every step.
#   * `SpringConstraint`  one Hookean spring per unique mesh edge,
#                         supplying stretch resistance along edges.
#   * `TriangleStretchConstraint`  one Baraff-Witkin (1998) membrane
#                                  element per triangle, supplying
#                                  in-plane stretch resistance over
#                                  the whole face (PSD-projected
#                                  Hessian, same trick as cuda-cloth).
#
# All three contribute gradient + 3x3 Hessian blocks into the global
# block-CSR matrix that `chysx::solver::PCGSolver` then solves each
# step.  Newton owns particle storage; chysx writes the implicit-Euler
# update directly into Newton's particle buffers (zero-copy via
# `wp.array.ptr`).
#
# Visually you see the corner stay fixed while the rest of the sheet
# drapes down and slowly settles into a hanging configuration.
#
# Command: python -m newton.examples chysx_hanging_cloth
#
###########################################################################

from __future__ import annotations

import numpy as np
import warp as wp

import newton
import newton.examples


class Example:
    def __init__(self, viewer, args):
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = 1
        self.sim_dt = self.frame_dt / self.sim_substeps

        self.viewer = viewer
        self.args = args

        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)

        z0 = 2.0
        self._initial_z = z0

        # 100x100 cloth patch (1 m x 1 m, 101x101 = 10201 particles,
        # ~20000 triangles).  Newton's `add_cloth_grid` lays particles
        # out in row-major order: idx = j*(dim_x+1) + i.  We leave
        # Newton's own spring / bending stiffnesses at zero — every
        # elastic contribution comes from chysx's own constraints.
        #
        # Per-particle mass is scaled down by 100x relative to the
        # 10x10 demo so that surface density (and thus the static
        # tension a stretched spring has to carry per metre of cloth)
        # stays the same.  Spring / FEM stiffness is bumped 10x to
        # keep `cell * k` in the same neighbourhood — the natural
        # scaling for an explicit / implicit Euler cloth.
        self._dim_x = 100
        self._dim_y = 100
        cell = 0.01
        builder.add_cloth_grid(
            pos=wp.vec3(-0.5, -0.5, z0),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0, 0.0, 0.0),
            dim_x=self._dim_x,
            dim_y=self._dim_y,
            cell_x=cell,
            cell_y=cell,
            mass=5.0e-4,
            tri_ke=0.0,
            tri_ka=0.0,
            tri_kd=0.0,
            edge_ke=0.0,
            edge_kd=0.0,
            particle_radius=0.005,
        )

        builder.add_ground_plane()

        self.model = builder.finalize()

        # Pin two corners along the y=-0.5 edge so the cloth hangs
        # like a flag stretched between two posts.  Newton's grid
        # indexing is row-major: idx = j*(dim_x+1) + i, so the two
        # bottom-edge corners are i=0 and i=dim_x at j=0.
        self._pinned_indices = [0, self._dim_x]
        self.solver = newton.solvers.SolverChysX(
            self.model,
            spring_stiffness=5.0e3,
            fem_stretch_stiffness=5.0e3,
            damping=2.0,
            pin_indices=self._pinned_indices,
            pin_stiffness=1.0e7,
        )

        # Snapshot the pinned corners' initial positions so test_final
        # can verify each pin held.
        q0 = self.model.particle_q.numpy().reshape(-1, 3)
        self._pinned_targets = q0[self._pinned_indices].copy()

        # The point furthest from the pinned edge is the midpoint of
        # the opposite edge (j = dim_y, i = dim_x // 2): this is the
        # particle that should drape down the most.
        self._far_idx = self._dim_y * (self._dim_x + 1) + (self._dim_x // 2)

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.model.contacts()

        self.viewer.set_model(self.model)
        self.viewer.set_camera(
            pos=wp.vec3(2.5, 2.5, 1.5),
            pitch=-15.0,
            yaw=-135.0,
        )

    def step(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

        self.sim_time += self.frame_dt

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()

    def test_final(self):
        # Validate that the implicit-Euler PCG step produced a stable
        # hanging configuration:
        #
        #   * the pinned corner has not drifted;
        #   * no NaN/Inf in positions or velocities;
        #   * every particle has at least started to fall (mean z below
        #     the initial plane);
        #   * the far corner has actually dropped a noticeable amount
        #     (the cloth is hanging, not stuck rigidly in mid-air);
        #   * velocities are bounded (no numerical explosion).
        q = self.state_0.particle_q.numpy().reshape(-1, 3)
        qd = self.state_0.particle_qd.numpy().reshape(-1, 3)

        if not (np.isfinite(q).all() and np.isfinite(qd).all()):
            raise ValueError("non-finite values in particle state")

        for k, idx in enumerate(self._pinned_indices):
            target = self._pinned_targets[k]
            drift = float(np.linalg.norm(q[idx] - target))
            if drift > 1.0e-3:
                raise ValueError(
                    f"pinned particle {idx} drifted by {drift:.4g} m "
                    f"(target={target}, got={q[idx]})"
                )

        mean_z = float(q[:, 2].mean())
        if mean_z >= self._initial_z:
            raise ValueError(
                f"cloth did not fall: mean z = {mean_z:.3f} (initial {self._initial_z:.3f})"
            )

        # Far point: midpoint of the edge opposite the pinned edge —
        # the particle that should drape down the most.  After ~2
        # seconds of simulation it ought to have moved well below
        # its starting height.
        far_corner_drop = self._initial_z - float(q[self._far_idx, 2])
        if far_corner_drop < 0.5:
            raise ValueError(
                f"far point barely moved (drop = {far_corner_drop:.3f} m); "
                "expected the cloth to drape visibly"
            )

        max_speed = float(np.linalg.norm(qd, axis=1).max())
        if max_speed > 20.0:
            raise ValueError(f"particle speed exploded: max |v| = {max_speed:.3f}")


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    parser.set_defaults(num_frames=240)

    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
