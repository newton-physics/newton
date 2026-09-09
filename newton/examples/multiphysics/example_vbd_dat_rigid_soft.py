# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example VBD DAT — Rigid-Soft Trampoline (penetration-free truncation)
#
# Three small bodies (a box, a sphere and a lying capsule) rest on a sheet
# pinned at its four edges, a smaller free sheet lies over them, and heavy,
# fast, spinning projectiles (a sphere and two capsules) are shot onto the
# stack from above.
# Contact stiffness alone cannot stop bodies this heavy within a step:
# without penetration-free truncation they drive the cloth through their
# surfaces and tunnel out.
#
# With ``rigid_soft_enable_dat=True`` the solver truncates both the cloth
# displacements and the rigid pose updates against per-contact division
# planes (Divide and Truncate), so the sheets always stay outside the bodies
# while they catch them; the same truncation keeps the two sheets from
# passing through each other (particle self-contact). Body-body contacts use
# compliant ALM with a stiff authored material. Set ``"enable_dat": False``
# in PARAMS to see the projectiles punch through.
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
    # body-body contact: compliant ALM with a stiff authored material. The default
    # ShapeConfig stiffness (2.5e3 N/m) lets these 4-10 kg projectiles interpenetrate
    # by several centimeters on impact; 3e6 N/m keeps overlap below 0.5 mm, while 1e7
    # destabilizes the resting box at this substep and iteration budget.
    "rigid_compliant_alm": True,
    "shape_ke": 3.0e6,
    "shape_kd": 1.0e3,
    # bottom cloth sheet (pinned at all four edges)
    "cloth_size": 1.6,
    "cloth_res": 40,
    "cloth_mass": 0.6,
    "cloth_z": 1.0,
    "cloth_tri_ke": 4.0e3,
    "cloth_tri_kd": 2.0e-1,
    "cloth_edge_ke": 1.0,
    "particle_radius": 6.0e-3,
    # top cloth sheet (free): drapes over the resting bodies and is struck by the projectiles
    "top_cloth_size": 1.5,
    "top_cloth_res": 25,
    "top_cloth_mass": 0.25,
    # height of the free sheet above the pinned sheet: clears the tallest resting body
    # (sphere, top at ~0.22 m) by more than the detection gap
    "top_cloth_height": 0.30,
    # cloth-cloth contact (self-contact DAT): interaction distance + detection reach.
    # The per-detection motion budget is ~0.5 * relaxation * (margin + gap), so the
    # reach must cover the fastest cloth motion per substep.
    "self_contact_margin": 1.2e-2,
    "self_contact_gap": 4.0e-2,
    # resting bodies start this far above the pinned sheet: beyond the rigid-soft detection
    # gap plus the particle radius, so nothing is in contact at frame 0 and the bodies settle
    "rest_clearance": 0.08,
    # rigid bodies: (kind, xy offset, start height, velocity, angular velocity, size, mass, rotation);
    # start height is measured above the pinned sheet, None rests the body on it; size is a
    # radius for spheres and capsules and (hx, hy, hz) half-extents for boxes; rotation is
    # None or (axis, angle).
    # The first three are projectiles above the top sheet; the last three sit below it.
    "bodies": [
        ("sphere", (0.00, 0.00), 0.6, (0.0, 0.0, -3.0), (0.0, 8.0, 0.0), 0.18, 10.0, None),
        ("capsule", (-0.35, 0.30), 0.9, (0.5, -0.4, -3.0), (6.0, 0.0, 4.0), 0.10, 6.0, None),
        ("capsule", (0.35, -0.30), 1.3, (-0.5, 0.4, -2.5), (0.0, 6.0, 6.0), 0.08, 4.0, None),
        ("box", (0.32, 0.32), None, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.07, 0.05, 0.04), 1.2, None),
        ("sphere", (-0.32, -0.32), None, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), 0.07, 1.0, None),
        ("capsule", (0.00, -0.38), None, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), 0.05, 0.8, ((0.0, 1.0, 0.0), 0.5 * np.pi)),
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
    "camera_offset": (2.4, -2.4, 0.6),  # relative to the pinned sheet center
    "camera_pitch": -11.0,
    "camera_yaw": 135.0,
    "camera_fov": 45.0,
}


class Example:
    def __init__(self, viewer, args):
        """Build the pinned cloth, the projectiles, the owned collision pipeline, and the DAT-enabled solver."""
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
            rigid_compliant_alm=self.params["rigid_compliant_alm"],
            rigid_soft_enable_dat=self.params["enable_dat"],
            particle_enable_self_contact=True,
            particle_self_contact_margin=self.params["self_contact_margin"],
            particle_self_contact_gap=self.params["self_contact_gap"],
            particle_vertex_contact_buffer_size=64,
            particle_edge_contact_buffer_size=128,
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
                wp.vec3(
                    self.params["camera_offset"][0],
                    self.params["camera_offset"][1],
                    self.params["cloth_z"] + self.params["camera_offset"][2],
                ),
                self.params["camera_pitch"],
                self.params["camera_yaw"],
            )
        if hasattr(self.viewer, "camera") and hasattr(self.viewer.camera, "fov"):
            self.viewer.camera.fov = self.params["camera_fov"]

    # ── model construction ──────────────────────────────────────────────

    def _build_cloth(self, builder):
        """Add the four-edge-pinned bottom sheet and the free top sheet to ``builder``."""
        p = self.params
        self._add_sheet(builder, p["cloth_size"], p["cloth_res"], p["cloth_mass"], p["cloth_z"], pinned=True)
        self._add_sheet(
            builder,
            p["top_cloth_size"],
            p["top_cloth_res"],
            p["top_cloth_mass"],
            p["cloth_z"] + p["top_cloth_height"],
            pinned=False,
        )

    def _add_sheet(self, builder, size, res, mass, z, *, pinned):
        """Add one square cloth sheet centered on the z axis, optionally pinned on all four edges."""
        p = self.params
        builder.add_cloth_grid(
            pos=wp.vec3(-0.5 * size, -0.5 * size, z),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0),
            dim_x=res,
            dim_y=res,
            cell_x=size / res,
            cell_y=size / res,
            mass=mass / ((res + 1) * (res + 1)),
            fix_left=pinned,
            fix_right=pinned,
            fix_top=pinned,
            fix_bottom=pinned,
            tri_ke=p["cloth_tri_ke"],
            tri_ka=p["cloth_tri_ke"],
            tri_kd=p["cloth_tri_kd"],
            edge_ke=p["cloth_edge_ke"],
            particle_radius=p["particle_radius"],
        )

    def _build_bodies(self, builder):
        """Add the rigid bodies (projectiles and resting shapes) to ``builder``."""
        p = self.params
        self._bodies = []
        palette = [
            (0.85, 0.3, 0.25),
            (0.25, 0.5, 0.85),
            (0.95, 0.75, 0.2),
            (0.3, 0.7, 0.4),
            (0.7, 0.4, 0.8),
            (0.9, 0.5, 0.3),
        ]
        cfg = newton.ModelBuilder.ShapeConfig(ke=p["shape_ke"], kd=p["shape_kd"])
        for i, (kind, offset, spec_z, _vel, _omega, size, mass, rotation) in enumerate(p["bodies"]):
            if kind == "sphere":
                inertia_val = 0.4 * mass * size * size
                inertia = wp.mat33(inertia_val, 0.0, 0.0, 0.0, inertia_val, 0.0, 0.0, 0.0, inertia_val)
            elif kind == "box":
                hx, hy, hz = size
                inertia = wp.mat33(
                    mass / 3.0 * (hy * hy + hz * hz),
                    0.0,
                    0.0,
                    0.0,
                    mass / 3.0 * (hx * hx + hz * hz),
                    0.0,
                    0.0,
                    0.0,
                    mass / 3.0 * (hx * hx + hy * hy),
                )
            else:
                # conservative sphere-like lumped inertia for the capsule
                reach = size + p["capsule_half_height"]
                inertia_val = 0.4 * mass * reach * reach
                inertia = wp.mat33(inertia_val, 0.0, 0.0, 0.0, inertia_val, 0.0, 0.0, 0.0, inertia_val)
            rot = (
                wp.quat_identity() if rotation is None else wp.quat_from_axis_angle(wp.vec3(*rotation[0]), rotation[1])
            )
            if spec_z is None:
                start_z = p["cloth_z"] + self._extent_below_center(kind, size, rot) + p["rest_clearance"]
            else:
                start_z = p["cloth_z"] + spec_z
            body = builder.add_body(
                xform=wp.transform(wp.vec3(offset[0], offset[1], start_z), rot),
                mass=mass,
                inertia=inertia,
                lock_inertia=True,
            )
            color = palette[i % len(palette)]
            if kind == "sphere":
                builder.add_shape_sphere(body=body, radius=size, cfg=cfg, color=color)
            elif kind == "box":
                builder.add_shape_box(body=body, hx=size[0], hy=size[1], hz=size[2], cfg=cfg, color=color)
            else:
                builder.add_shape_capsule(
                    body=body,
                    radius=size,
                    half_height=p["capsule_half_height"],
                    cfg=cfg,
                    color=color,
                )
            self._bodies.append(body)

    def _extent_below_center(self, kind, size, rot):
        """Return how far a shape reaches below its body origin along world z for orientation ``rot``."""
        if kind == "sphere":
            return size
        x, y, z, w = (float(rot[0]), float(rot[1]), float(rot[2]), float(rot[3]))
        # third row of the body-to-world rotation: |R_z,i| scales the body-axis half-extents
        row_z = np.abs([2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)])
        if kind == "box":
            return float(row_z @ np.asarray(size))
        return float(row_z[2] * self.params["capsule_half_height"] + size)

    # ── simulation loop ─────────────────────────────────────────────────

    def simulate(self):
        """Advance the solver by ``sim_substeps`` sub-steps."""
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            self.solver.step(self.state_0, self.state_1, self.control, None, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        """Advance one frame."""
        self.frame += 1
        self.simulate()
        self.sim_time += self.frame_dt

    def render(self):
        """Log the current state to the viewer."""
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()

    # ── validation ──────────────────────────────────────────────────────

    def _cloth_penetration(self) -> float:
        """Deepest penetration [m] of any cloth vertex into any rigid body (0 if none)."""
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
                # capsule (along body-frame Z) or box: rotate world offsets into the body frame
                x, y, z, w = rot
                rot_mat = np.array(
                    [
                        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
                    ]
                )
                local = local @ rot_mat  # world->body: R^T applied to rows
                if kind == "box":
                    excess = np.abs(local) - np.asarray(size)[None, :]
                    outside = np.linalg.norm(np.maximum(excess, 0.0), axis=1)
                    inside = np.minimum(excess.max(axis=1), 0.0)
                    sdf = outside + inside
                else:
                    seg_z = np.clip(local[:, 2], -p["capsule_half_height"], p["capsule_half_height"])
                    closest = np.stack([np.zeros_like(seg_z), np.zeros_like(seg_z), seg_z], axis=1)
                    sdf = np.linalg.norm(local - closest, axis=1) - size
            deepest = max(deepest, -float(sdf.min()))
        return deepest

    def test_post_step(self):
        """Track the deepest cloth-projectile penetration observed so far."""
        pen = self._cloth_penetration()
        self.max_penetration = max(self.max_penetration, pen)

    def test_final(self):
        """Assert finite state, the penetration bound, and that every body is still held by the pinned sheet."""
        q = self.state_0.particle_q.numpy()
        body_q = self.state_0.body_q.numpy()
        if not (np.isfinite(q).all() and np.isfinite(body_q).all()):
            raise AssertionError("simulation produced non-finite state")
        if self.max_penetration > 1.0e-4:
            raise AssertionError(f"cloth penetrated a rigid body by {self.max_penetration:.6f} m")
        for body_index in self._bodies:
            z = float(body_q[body_index][2])
            if z < self.params["cloth_z"] - 1.0:
                raise AssertionError(f"body {body_index} fell more than 1 m below the pinned sheet (z={z:.3f})")

    @staticmethod
    def create_parser():
        """Create the example's argument parser with the scene's default frame count."""
        parser = newton.examples.create_parser()
        parser.set_defaults(num_frames=PARAMS["num_frames"])
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args)
    newton.examples.run(example, args)
