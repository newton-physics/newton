# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example VBD DAT — Rigid-Soft Trampoline (penetration-free truncation)
#
# Heavy, fast, spinning rigid bodies (a sphere and two capsules) are shot
# onto a pinned cloth sheet. Contact stiffness alone cannot stop bodies
# this heavy within a step: without penetration-free truncation the bodies
# drive the cloth through their surfaces and tunnel out.
#
# With ``rigid_soft_enable_dat=True`` the solver truncates both the
# cloth displacements and the rigid pose updates against per-contact
# division planes (Divide and Truncate), so the sheet always stays outside
# the bodies while it catches them. Set ``"enable_dat": False`` in PARAMS
# to see the bodies punch through.
#
# Command: python -m newton.examples vbd_dat_rigid_soft
###########################################################################

from __future__ import annotations

import numpy as np
import warp as wp

import newton
import newton.examples

PARAMS = {
    # simulation
    "fps": 60,
    "sim_substeps": 4,
    "solver_iterations": 8,
    "num_frames": 300,
    # rigid DAT penetration-free truncation (the feature under demonstration)
    "enable_dat": True,
    # contact detection gap: rigid DAT derives its per-detection motion budget
    # (~0.5 * relaxation * gap) from this automatically; it must exceed the fastest
    # per-detection-interval motion or DAT throttles the body
    "soft_contact_gap": 0.06,
    # cloth sheet (pinned at all four edges)
    "cloth_size": 1.6,
    "cloth_res": 40,
    "cloth_mass": 0.6,
    "cloth_z": 1.0,
    "cloth_tri_ke": 4.0e3,
    "cloth_tri_kd": 2.0e-1,
    "cloth_edge_ke": 1.0e-2,
    "particle_radius": 6.0e-3,
    # projectiles: (kind, xy offset, start z, velocity, angular velocity, size, mass)
    "bodies": [
        ("sphere", (0.00, 0.00), 1.5, (0.0, 0.0, -3.0), (0.0, 8.0, 0.0), 0.18, 10.0),
        ("capsule", (-0.35, 0.30), 1.9, (0.5, -0.4, -3.0), (6.0, 0.0, 4.0), 0.10, 6.0),
        ("capsule", (0.35, -0.30), 2.3, (-0.5, 0.4, -2.5), (0.0, 6.0, 6.0), 0.08, 4.0),
    ],
    "capsule_half_height": 0.16,
    # soft-rigid contact material
    "soft_contact_ke": 2.0e4,
    "soft_contact_kd": 1.0e-4,
    "soft_contact_mu": 0.6,
    # collision
    "collision_broad_phase": "nxn",
    "rigid_body_particle_contact_buffer_size": 4096,
    # camera (fixed side view)
    "camera_pos": (2.4, -2.4, 1.6),
    "camera_pitch": -11.0,
    "camera_yaw": 135.0,
    "camera_fov": 45.0,
}


class Example:
    def __init__(self, viewer, args):
        self.viewer = viewer
        self.params = PARAMS
        self.frame_dt = 1.0 / self.params["fps"]
        self.sim_substeps = self.params["sim_substeps"]
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0
        self.frame = 0

        builder = newton.ModelBuilder()  # Z up, gravity -Z
        self._build_cloth(builder)
        self._build_bodies(builder)

        builder.color()
        self.model = builder.finalize()

        self.model.soft_contact_ke = self.params["soft_contact_ke"]
        self.model.soft_contact_kd = self.params["soft_contact_kd"]
        self.model.soft_contact_mu = self.params["soft_contact_mu"]

        self.collision_pipeline = newton.CollisionPipeline(
            self.model,
            broad_phase=self.params["collision_broad_phase"],
            soft_contact_gap=self.params["soft_contact_gap"],
            enable_rigid_soft_full_surface_contact=True,
        )
        # The solver owns the pipeline and drives detection itself; rigid DAT derives
        # its motion budgets from the pipeline's detection distances.
        self.solver = newton.solvers.SolverVBD(
            self.model,
            iterations=self.params["solver_iterations"],
            rigid_soft_enable_dat=self.params["enable_dat"],
            rigid_body_particle_contact_buffer_size=self.params["rigid_body_particle_contact_buffer_size"],
            collision_pipeline=self.collision_pipeline,
        )

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()

        # launch velocities
        qd = self.state_0.body_qd.numpy()
        for body_index, spec in zip(self._bodies, self.params["bodies"], strict=True):
            qd[body_index][:3] = spec[3]
            qd[body_index][3:] = spec[4]
        self.state_0.body_qd.assign(qd)
        wp.copy(self.state_1.body_qd, self.state_0.body_qd)

        self.max_penetration = 0.0

        self.viewer.set_model(self.model)
        if hasattr(self.viewer, "set_camera"):
            self.viewer.set_camera(
                wp.vec3(*self.params["camera_pos"]),
                self.params["camera_pitch"],
                self.params["camera_yaw"],
            )
        if hasattr(self.viewer, "camera") and hasattr(self.viewer.camera, "fov"):
            self.viewer.camera.fov = self.params["camera_fov"]

    # ── model construction ──────────────────────────────────────────────

    def _build_cloth(self, builder):
        p = self.params
        res = p["cloth_res"]
        size = p["cloth_size"]
        builder.add_cloth_grid(
            pos=wp.vec3(-0.5 * size, -0.5 * size, p["cloth_z"]),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0),
            dim_x=res,
            dim_y=res,
            cell_x=size / res,
            cell_y=size / res,
            mass=p["cloth_mass"] / ((res + 1) * (res + 1)),
            fix_left=True,
            fix_right=True,
            fix_top=True,
            fix_bottom=True,
            tri_ke=p["cloth_tri_ke"],
            tri_ka=p["cloth_tri_ke"],
            tri_kd=p["cloth_tri_kd"],
            edge_ke=p["cloth_edge_ke"],
            particle_radius=p["particle_radius"],
        )

    def _build_bodies(self, builder):
        p = self.params
        self._bodies = []
        palette = [(0.85, 0.3, 0.25), (0.25, 0.5, 0.85), (0.95, 0.75, 0.2)]
        for i, (kind, offset, start_z, _vel, _omega, size, mass) in enumerate(p["bodies"]):
            if kind == "sphere":
                inertia_val = 0.4 * mass * size * size
            else:
                # conservative sphere-like lumped inertia for the capsule
                reach = size + p["capsule_half_height"]
                inertia_val = 0.4 * mass * reach * reach
            inertia = wp.mat33(inertia_val, 0.0, 0.0, 0.0, inertia_val, 0.0, 0.0, 0.0, inertia_val)
            body = builder.add_body(
                xform=wp.transform(wp.vec3(offset[0], offset[1], start_z), wp.quat_identity()),
                mass=mass,
                inertia=inertia,
                lock_inertia=True,
            )
            if kind == "sphere":
                builder.add_shape_sphere(body=body, radius=size, color=palette[i % len(palette)])
            else:
                builder.add_shape_capsule(
                    body=body,
                    radius=size,
                    half_height=p["capsule_half_height"],
                    color=palette[i % len(palette)],
                )
            self._bodies.append(body)

    # ── simulation loop ─────────────────────────────────────────────────

    def simulate(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            self.solver.step(self.state_0, self.state_1, self.control, None, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        self.frame += 1
        self.simulate()
        self.sim_time += self.frame_dt

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()

    # ── validation ──────────────────────────────────────────────────────

    def _cloth_penetration(self) -> float:
        """Deepest penetration [m] of any cloth vertex into any projectile (0 if none)."""
        p = self.params
        q = self.state_0.particle_q.numpy()
        body_q = self.state_0.body_q.numpy()
        deepest = 0.0
        for body_index, spec in zip(self._bodies, p["bodies"], strict=True):
            kind, size = spec[0], spec[5]
            bq = body_q[body_index]
            pos, rot = bq[:3], bq[3:]
            local = q - pos[None, :]
            if kind == "sphere":
                sdf = np.linalg.norm(local, axis=1) - size
            else:
                # capsule along body-frame Z: rotate world offsets into the body frame
                x, y, z, w = rot
                rot_mat = np.array(
                    [
                        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
                    ]
                )
                local = local @ rot_mat  # world->body: R^T applied to rows
                seg_z = np.clip(local[:, 2], -p["capsule_half_height"], p["capsule_half_height"])
                closest = np.stack([np.zeros_like(seg_z), np.zeros_like(seg_z), seg_z], axis=1)
                sdf = np.linalg.norm(local - closest, axis=1) - size
            deepest = max(deepest, -float(sdf.min()))
        return deepest

    def test_post_step(self):
        pen = self._cloth_penetration()
        self.max_penetration = max(self.max_penetration, pen)

    def test_final(self):
        q = self.state_0.particle_q.numpy()
        body_q = self.state_0.body_q.numpy()
        if not (np.isfinite(q).all() and np.isfinite(body_q).all()):
            raise AssertionError("simulation produced non-finite state")
        if self.max_penetration > 1.0e-4:
            raise AssertionError(f"cloth penetrated a rigid body by {self.max_penetration:.6f} m")
        for body_index in self._bodies:
            z = float(body_q[body_index][2])
            if z < 0.0:
                raise AssertionError(f"body {body_index} tunneled through the sheet (z={z:.3f})")

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.set_defaults(num_frames=PARAMS["num_frames"])
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args)
    newton.examples.run(example, args)
