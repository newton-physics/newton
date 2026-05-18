# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Soft Convergence Under Refinement
#
# Runs the same gravity-extension scenario at three mesh resolutions
# (coarse, medium, fine) to validate that the Neo-Hookean volumetric
# solver produces physically reasonable results regardless of mesh
# density. All three meshes should produce a positive downward
# displacement that stays within the same order of magnitude.
#
# Command: python -m newton.examples vbd.example_soft_convergence_refinement
#
###########################################################################

import numpy as np
import warp as wp

import newton
import newton.examples
from newton import ParticleFlags


def _run_extension(dim_xy: int, dim_z: int, cell: float, n_frames: int) -> float:
    """Run gravity extension and return bottom-layer displacement."""
    builder = newton.ModelBuilder()
    builder.add_ground_plane()
    builder.add_soft_grid(
        pos=wp.vec3(0.0, 0.0, 2.0),
        rot=wp.quat_identity(),
        vel=wp.vec3(0.0, 0.0, 0.0),
        dim_x=dim_xy,
        dim_y=dim_xy,
        dim_z=dim_z,
        cell_x=cell,
        cell_y=cell,
        cell_z=cell,
        density=1000.0,
        k_mu=5.0e4,
        k_lambda=5.0e4,
        k_damp=0.1,
    )
    builder.color()
    model = builder.finalize()
    model.soft_contact_ke = 1.0e2
    model.soft_contact_kd = 1.0e0
    model.soft_contact_mu = 1.0

    q_np = model.particle_q.numpy()
    beam_height = dim_z * cell
    top_z = 2.0 + beam_height
    top_mask = np.abs(q_np[:, 2] - top_z) < 1e-6
    flags = model.particle_flags.numpy()
    for i in np.where(top_mask)[0]:
        flags[i] = flags[i] & ~int(ParticleFlags.ACTIVE)
    model.particle_flags = wp.array(flags)

    solver = newton.solvers.SolverVBD(model=model, iterations=20, particle_enable_self_contact=False)
    s0 = model.state()
    s1 = model.state()
    ctrl = model.control()
    contacts = model.contacts()
    dt = 1.0 / 60 / 5

    if wp.get_device().is_cuda:
        with wp.ScopedCapture() as capture:
            for _ in range(5):
                s0.clear_forces()
                model.collide(s0, contacts)
                solver.step(s0, s1, ctrl, contacts, dt)
                s0, s1 = s1, s0
        graph = capture.graph
        for _ in range(n_frames):
            wp.capture_launch(graph)
    else:
        for _ in range(n_frames):
            for _ in range(5):
                s0.clear_forces()
                model.collide(s0, contacts)
                solver.step(s0, s1, ctrl, contacts, dt)
                s0, s1 = s1, s0

    q2 = s0.particle_q.numpy()
    bot_mask = np.abs(q_np[:, 2] - 2.0) < 1e-6
    return float(2.0 - np.mean(q2[bot_mask, 2]))


class Example:
    def __init__(self, viewer, args=None):
        self.viewer = viewer
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = 5
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.iterations = 20

        configs = [
            {"name": "coarse", "dim_xy": 2, "dim_z": 10, "cell": 0.10},
            {"name": "medium", "dim_xy": 4, "dim_z": 20, "cell": 0.05},
            {"name": "fine", "dim_xy": 8, "dim_z": 40, "cell": 0.025},
        ]

        self.deltas: list[float] = []
        self.config_names: list[str] = []
        for cfg in configs:
            delta = _run_extension(cfg["dim_xy"], cfg["dim_z"], cfg["cell"], n_frames=300)
            self.deltas.append(delta)
            self.config_names.append(cfg["name"])

        # Visual: medium mesh
        builder = newton.ModelBuilder()
        builder.add_ground_plane()
        builder.add_soft_grid(
            pos=wp.vec3(0.0, 0.0, 2.0),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0, 0.0, 0.0),
            dim_x=4,
            dim_y=4,
            dim_z=20,
            cell_x=0.05,
            cell_y=0.05,
            cell_z=0.05,
            density=1000.0,
            k_mu=5e4,
            k_lambda=5e4,
            k_damp=0.1,
        )
        builder.color()
        self.model = builder.finalize()

        q_np = self.model.particle_q.numpy()
        top_mask = np.abs(q_np[:, 2] - 3.0) < 1e-6
        flags = self.model.particle_flags.numpy()
        for i in np.where(top_mask)[0]:
            flags[i] = flags[i] & ~int(ParticleFlags.ACTIVE)
        self.model.particle_flags = wp.array(flags)

        self.solver = newton.solvers.SolverVBD(model=self.model, iterations=20, particle_enable_self_contact=False)
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.model.contacts()
        self.model.soft_contact_ke = 1e2
        self.model.soft_contact_kd = 1e0
        self.model.soft_contact_mu = 1.0

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

    def step(self):
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()
        self.sim_time += self.frame_dt

    def test_final(self):
        for name, delta in zip(self.config_names, self.deltas, strict=True):
            if delta <= 0:
                raise ValueError(f"{name}: non-positive displacement {delta:.4f}")
            if delta > 1.0:
                raise ValueError(f"{name}: excessive displacement {delta:.4f}")

        # All should be same order of magnitude (within 3x)
        d_min = min(self.deltas)
        d_max = max(self.deltas)
        if d_max / d_min > 3.0:
            raise ValueError(
                "Displacement spread too large: "
                + ", ".join(f"{n}={d:.4f}" for n, d in zip(self.config_names, self.deltas, strict=True))
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
