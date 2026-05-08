# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Soft Beam Stretch — Volume Preservation
#
# A horizontal tetrahedral beam with its left face pinned and right face
# stretched to 2x its rest length over a linear ramp. The stable
# Neo-Hookean material should preserve volume: with high bulk modulus
# (k_lambda >> k_mu) the volume ratio at 2x stretch should stay close
# to 1.0. The test measures the total tet volume at equilibrium and
# compares it to the rest volume.
#
# Command: python -m newton.examples vbd.example_soft_beam_stretch
#
###########################################################################

import numpy as np
import warp as wp

import newton
import newton.examples
from newton import ParticleFlags


def _compute_tet_volume(q: np.ndarray, tet_indices: np.ndarray) -> float:
    """Sum of signed tet volumes from current positions."""
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
    DIM_X = 10
    DIM_Y = 3
    DIM_Z = 3
    CELL = 0.05
    DENSITY = 1000.0
    K_MU = 1.0e4
    K_LAMBDA = 1.0e5
    K_DAMP = 1.0e-3
    STRETCH = 2.0
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

        self.rest_x = self.DIM_X * self.CELL

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
            fix_left=True,
        )

        builder.color()
        self.model = builder.finalize()
        self.model.set_gravity((0.0, 0.0, 0.0))

        # Pin right face manually (fix_right sets mass=0 at build time,
        # but we also need to track which particles to drive)
        q_np = self.model.particle_q.numpy()
        right_x = self.rest_x
        self.right_mask = np.abs(q_np[:, 0] - right_x) < 1e-6
        self.right_indices = np.where(self.right_mask)[0]
        self.right_rest_x = right_x

        # Set right face as kinematic
        flags = self.model.particle_flags.numpy()
        for i in self.right_indices:
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

        # Compute rest volume
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

    def _apply_stretch_ramp(self):
        if self._frame_index >= self.RAMP_FRAMES:
            target_stretch = self.STRETCH
        else:
            t = self._frame_index / self.RAMP_FRAMES
            target_stretch = 1.0 + t * (self.STRETCH - 1.0)
        target_x = target_stretch * self.right_rest_x
        q = self.state_0.particle_q.numpy()
        q[self.right_indices, 0] = target_x
        self.state_0.particle_q.assign(q)

    def step(self):
        self._apply_stretch_ramp()
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
            lambda q, qd: wp.length(qd) < 20.0,
        )

        q = self.state_0.particle_q.numpy()
        final_volume = _compute_tet_volume(q, self.tet_indices)
        volume_ratio = final_volume / self.rest_volume

        if abs(volume_ratio - 1.0) > 0.10:
            raise ValueError(
                f"Volume not preserved: ratio {volume_ratio:.3f} "
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
