# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example ADMM Contact Coupled Solver
#
# An XPBD particle pad falls under gravity into a ball-jointed rigid drawer.
# SolverCoupledADMM runs particle-shape collision detection internally and
# supplies the drawer response through frictionless ADMM contacts.
#
# Pass ``--solver free`` to disable the ADMM contacts and compare against the
# uncoupled baseline.
#
# Command: python -m newton.examples admm_contact_solver
#          python -m newton.examples admm_contact_solver --solver free
#
###########################################################################

from __future__ import annotations

import numpy as np
import warp as wp
from newton.solvers.experimental.coupled import SolverCoupled, SolverCoupledADMM

import newton
import newton.examples
from newton.solvers import SolverSemiImplicit, SolverXPBD


@wp.kernel(enable_backward=False)
def _gather_particles(
    particle_ids: wp.array[int],
    particle_q: wp.array[wp.vec3],
    points: wp.array[wp.vec3],
):
    i = wp.tid()
    points[i] = particle_q[particle_ids[i]]


class Example:
    def __init__(self, viewer, args):
        self.viewer = viewer
        self.sim_time = 0.0
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = 2
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.solver_type = args.solver
        self.track_gap = args.test
        self.particle_radius = 0.025

        builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
        (
            self.falling_particles_a,
            self.falling_particles_b,
            self.tray_body,
            self.tray_joint,
        ) = self._emit_drop_scene(builder, args)
        self.falling_particles = self.falling_particles_a + self.falling_particles_b

        builder.set_coloring([self.falling_particles])
        self.model = builder.finalize()
        # Each XPBD particle owner handles same-color particle-particle
        # contacts in its own solver view; ADMM handles particle-shape tray
        # response and cross-owner particle-particle contacts.
        self.model.soft_contact_ke = 0.0
        self.model.soft_contact_kd = 0.0
        self.model.soft_contact_kf = 0.0
        self.model.soft_contact_mu = 0.0

        self.solver = SolverCoupledADMM(
            model=self.model,
            entries=[
                SolverCoupled.Entry(
                    name="drop_a",
                    solver=lambda v: SolverXPBD(model=v, iterations=args.xpbd_iterations),
                    particles=self.falling_particles_a,
                ),
                SolverCoupled.Entry(
                    name="drop_b",
                    solver=lambda v: SolverXPBD(model=v, iterations=args.xpbd_iterations),
                    particles=self.falling_particles_b,
                ),
                SolverCoupled.Entry(
                    name="tray",
                    solver=lambda v: SolverSemiImplicit(
                        model=v,
                        **{"enable_tri_contact": False, "joint_attach_ke": 2.5e4, "joint_attach_kd": 4.0e2},
                    ),
                    bodies=[self.tray_body],
                    joints=[self.tray_joint],
                ),
            ],
            coupling=SolverCoupledADMM.Config(
                iterations=args.admm_iterations,
                rho=args.rho,
                gamma=args.gamma,
                baumgarte=args.baumgarte,
                contact_pairs=(
                    [
                        SolverCoupledADMM.ContactPair(
                            source="drop_a",
                            destination="tray",
                        ),
                        SolverCoupledADMM.ContactPair(
                            source="drop_b",
                            destination="tray",
                        ),
                        SolverCoupledADMM.ContactPair(
                            source="drop_a",
                            destination="drop_b",
                        ),
                    ]
                    if self.solver_type == "admm"
                    else []
                ),
            ),
        )

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_1)
        self.control = self.model.control()
        self.collision_pipeline = newton.CollisionPipeline(self.model)
        self.contacts = self.collision_pipeline.contacts()

        self.falling_ids = wp.array(self.falling_particles, dtype=int, device=self.model.device)
        self.falling_points = wp.empty(len(self.falling_particles), dtype=wp.vec3, device=self.model.device)
        self.falling_radii = wp.full(
            len(self.falling_particles),
            0.025,
            dtype=wp.float32,
            device=self.model.device,
        )
        self.falling_colors = wp.array(
            [wp.vec3(0.12, 0.38, 0.92)] * len(self.falling_particles_a)
            + [wp.vec3(0.92, 0.42, 0.12)] * len(self.falling_particles_b),
            dtype=wp.vec3,
            device=self.model.device,
        )

        self.initial_gap = self._min_contact_gap(self.model.particle_q.numpy())
        self.min_observed_gap = self.initial_gap
        self.max_collision_contacts = 0
        self.initial_tray_origin = self._tray_origin(self.model.body_q.numpy())
        self.max_tray_origin_error = 0.0

        newton.examples.configure_coupled_view(self, args)
        if hasattr(self.viewer, "show_particles"):
            self.viewer.show_particles = False
        camera_target = np.array([0.0, 0.0, 0.06], dtype=np.float32)
        camera_offset = np.array([0.72, -0.9, 0.56], dtype=np.float32)
        camera_offset /= np.linalg.norm(camera_offset)
        camera_pos_np = camera_target + camera_offset
        view_dir = camera_target - camera_pos_np
        pitch = float(np.rad2deg(np.arcsin(view_dir[2] / np.linalg.norm(view_dir))))
        yaw = float(np.rad2deg(np.arctan2(view_dir[1], view_dir[0])))
        camera_pos = wp.vec3(float(camera_pos_np[0]), float(camera_pos_np[1]), float(camera_pos_np[2]))
        camera_target_wp = wp.vec3(float(camera_target[0]), float(camera_target[1]), float(camera_target[2]))
        self.viewer.set_camera(pos=camera_pos, pitch=pitch, yaw=yaw)
        if hasattr(self.viewer, "camera") and hasattr(self.viewer.camera, "look_at"):
            self.viewer.camera.look_at(camera_target_wp)

        self.capture()

    def capture(self):
        if not self.model.device.is_cuda:
            self.graph = None
            return

        with wp.ScopedDevice(self.model.device):
            with wp.ScopedCapture() as capture:
                self.simulate()
        self.graph = capture.graph
        if self.graph is None:
            raise RuntimeError(f"CUDA graph capture failed on device {self.model.device}")

    def simulate(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            newton.examples.apply_coupled_viewer_forces(self, self.state_0)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        if self.graph:
            with wp.ScopedDevice(self.model.device):
                wp.capture_launch(self.graph)
        else:
            self.simulate()
        if self.track_gap:
            self.min_observed_gap = min(self.min_observed_gap, self._min_contact_gap(self.state_0.particle_q.numpy()))
            self.max_collision_contacts = max(self.max_collision_contacts, self.solver.collision_contact_count_max)
            origin_error = np.linalg.norm(self._tray_origin(self.state_0.body_q.numpy()) - self.initial_tray_origin)
            self.max_tray_origin_error = max(self.max_tray_origin_error, float(origin_error))
        self.sim_time += self.frame_dt

    def test_final(self):
        particle_q = self.state_0.particle_q.numpy()
        assert np.isfinite(particle_q).all(), "Particle positions contain NaN or inf values"

        max_extent = np.linalg.norm(np.max(particle_q, axis=0) - np.min(particle_q, axis=0))
        assert max_extent < 2.0, f"Particle pads escaped: bbox={max_extent:.3f}"

        if self.solver_type == "admm":
            min_gap = self._min_contact_gap(particle_q)
            assert self.max_collision_contacts > 0, "Collision detection did not produce ADMM contact candidates"
            assert self.min_observed_gap <= 0.01, (
                f"ADMM contact did not reach the contact surface: min_observed_gap={self.min_observed_gap:.4f}"
            )
            assert self.min_observed_gap > -0.02, f"ADMM contact penetrated too deeply: {self.min_observed_gap:.4f}"
            assert min_gap > -0.01, f"ADMM contact did not keep the pad above the tray surface: final_gap={min_gap:.4f}"
            assert self.max_tray_origin_error < 0.05, (
                f"ball-jointed tray origin drifted too far: max_error={self.max_tray_origin_error:.4f}"
            )

    def render(self):
        render_state = newton.examples.get_coupled_view_state(self)
        wp.launch(
            _gather_particles,
            dim=len(self.falling_particles),
            inputs=[self.falling_ids, render_state.particle_q],
            outputs=[self.falling_points],
            device=self.model.device,
        )

        self.viewer.begin_frame(self.sim_time)
        newton.examples.log_coupled_view(self, log_contacts=False)
        self.viewer.log_points(
            "/admm_contact/falling_particles",
            self.falling_points,
            radii=self.falling_radii,
            colors=self.falling_colors,
        )
        self.viewer.end_frame()

    def _emit_drop_scene(
        self,
        builder: newton.ModelBuilder,
        args,
    ) -> tuple[list[int], list[int], int, int]:
        tray_body = builder.add_body(
            xform=wp.transform(p=wp.vec3(0.0, 0.0, 0.0), q=wp.quat_identity()),
            mass=args.tray_mass,
            inertia=wp.mat33(np.eye(3) * args.tray_inertia),
            label="admm_contact_tray",
        )
        tray_joint = builder.add_joint_ball(
            parent=-1,
            child=tray_body,
            parent_xform=wp.transform(p=wp.vec3(0.0, 0.0, 0.0), q=wp.quat_identity()),
            child_xform=wp.transform(p=wp.vec3(0.0, 0.0, 0.0), q=wp.quat_identity()),
            label="admm_contact_tray_origin",
        )
        builder.add_articulation([tray_joint], label="admm_contact_tray_articulation")

        tray_cfg = newton.ModelBuilder.ShapeConfig()
        tray_cfg.has_shape_collision = False
        tray_cfg.has_particle_collision = True
        tray_cfg.is_visible = True
        builder.add_shape_box(
            tray_body,
            xform=wp.transform(p=wp.vec3(0.0, 0.0, -0.025), q=wp.quat_identity()),
            hx=0.34,
            hy=0.34,
            hz=0.025,
            cfg=tray_cfg,
            color=(0.54, 0.56, 0.60),
        )

        wall_h = 0.075
        wall_t = 0.025
        wall_z = wall_h
        wall_extent = 0.34
        builder.add_shape_box(
            tray_body,
            xform=wp.transform(p=wp.vec3(wall_extent, 0.0, wall_z), q=wp.quat_identity()),
            hx=wall_t,
            hy=wall_extent,
            hz=wall_h,
            cfg=tray_cfg,
            color=(0.43, 0.46, 0.52),
        )
        builder.add_shape_box(
            tray_body,
            xform=wp.transform(p=wp.vec3(-wall_extent, 0.0, wall_z), q=wp.quat_identity()),
            hx=wall_t,
            hy=wall_extent,
            hz=wall_h,
            cfg=tray_cfg,
            color=(0.43, 0.46, 0.52),
        )
        builder.add_shape_box(
            tray_body,
            xform=wp.transform(p=wp.vec3(0.0, wall_extent, wall_z), q=wp.quat_identity()),
            hx=wall_extent,
            hy=wall_t,
            hz=wall_h,
            cfg=tray_cfg,
            color=(0.43, 0.46, 0.52),
        )
        builder.add_shape_box(
            tray_body,
            xform=wp.transform(p=wp.vec3(0.0, -wall_extent, wall_z), q=wp.quat_identity()),
            hx=wall_extent,
            hy=wall_t,
            hz=wall_h,
            cfg=tray_cfg,
            color=(0.43, 0.46, 0.52),
        )

        falling_particles_a: list[int] = []
        falling_particles_b: list[int] = []
        spacing = 0.055
        dim_x = 7
        dim_y = 7

        for ix in range(dim_x):
            x = (ix - 0.5 * (dim_x - 1)) * spacing
            for iy in range(dim_y):
                y = (iy - 0.5 * (dim_y - 1)) * spacing
                is_group_a = ix < dim_x // 2
                vx = args.particle_contact_push if is_group_a else -args.particle_contact_push
                particle = builder.add_particle(
                    pos=(x, y, args.drop_height),
                    vel=(vx, 0.0, 0.0),
                    mass=0.025,
                    radius=self.particle_radius,
                )
                if is_group_a:
                    falling_particles_a.append(particle)
                else:
                    falling_particles_b.append(particle)

        return falling_particles_a, falling_particles_b, tray_body, tray_joint

    def _min_contact_gap(self, particle_q: np.ndarray) -> float:
        tray_z = self._tray_origin(self.state_0.body_q.numpy())[2] if hasattr(self, "state_0") else 0.0
        return min(
            float(particle_q[particle, 2] - tray_z - self.particle_radius) for particle in self.falling_particles
        )

    def _tray_origin(self, body_q: np.ndarray) -> np.ndarray:
        return np.asarray(body_q[self.tray_body, :3], dtype=np.float32)

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        newton.examples.add_coupled_view_args(parser)
        parser.add_argument(
            "--solver",
            "-s",
            help="'admm' for ADMM contacts, or 'free' for the uncoupled baseline",
            type=str,
            choices=["admm", "free"],
            default="admm",
        )
        parser.add_argument(
            "--admm-iterations",
            help="ADMM iterations per frame",
            type=int,
            default=12,
        )
        parser.add_argument(
            "--xpbd-iterations",
            help="XPBD iterations per particle-owner solve",
            type=int,
            default=6,
        )
        parser.add_argument(
            "--rho",
            help="ADMM penalty parameter",
            type=float,
            default=45.0,
        )
        parser.add_argument(
            "--gamma",
            help="ADMM proximal metric scale",
            type=float,
            default=0.001,
        )
        parser.add_argument(
            "--baumgarte",
            help="Fraction of contact penetration corrected per frame",
            type=float,
            default=0.1,
        )
        parser.add_argument(
            "--particle-contact-push",
            help="Initial opposing horizontal speed for the two particle owners",
            type=float,
            default=0.0,
        )
        parser.add_argument(
            "--drop-height",
            help="Initial height of the falling particle pad above the tray",
            type=float,
            default=0.65,
        )
        parser.add_argument(
            "--tray-mass",
            help="Dynamic tray mass",
            type=float,
            default=2.0,
        )
        parser.add_argument(
            "--tray-inertia",
            help="Diagonal tray inertia",
            type=float,
            default=0.02,
        )
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args)
    newton.examples.run(example, args)
