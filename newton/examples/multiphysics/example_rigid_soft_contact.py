# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Rigid Soft Contact
#
# Shows how to set up a rigid sphere colliding with a soft FEM beam.
#
# Command: uv run -m newton.examples rigid_soft_contact
#
###########################################################################

import numpy as np
import warp as wp

import newton
import newton.examples

GRID_DIM_X = 20
GRID_DIM_Y = 10
GRID_DIM_Z = 10
GRID_CELL_SIZE = 0.1
SPHERE_RADIUS = 0.75
SPHERE_INITIAL_Z = 2.5

SOFT_GRID_DENSITY = 100.0
SOFT_GRID_K_MU = 1.0e4
SOFT_GRID_K_LAMBDA = 5.0e4
SOFT_GRID_K_DAMP = 1.0

RIGID_SOFT_SPHERE_DENSITY = 13.5
RIGID_SOFT_CONTACT_KE = 75.0
RIGID_SOFT_CONTACT_KD = 1.0
RIGID_SOFT_CONTACT_KF = 1.0e3
RIGID_SOFT_CONTACT_MU = 1.0
GROUND_CONTACT_KE = 2.0e5


def _grid_index(x, y, z):
    return (GRID_DIM_X + 1) * (GRID_DIM_Y + 1) * z + (GRID_DIM_X + 1) * y + x


GRID_CORNER_INDICES = np.array(
    [_grid_index(x, y, z) for x in (0, GRID_DIM_X) for y in (0, GRID_DIM_Y) for z in (0, GRID_DIM_Z)],
    dtype=np.int32,
)


def _tet_volumes(particle_q, tet_indices):
    x0 = particle_q[tet_indices[:, 0]]
    x1 = particle_q[tet_indices[:, 1]]
    x2 = particle_q[tet_indices[:, 2]]
    x3 = particle_q[tet_indices[:, 3]]
    return np.linalg.det(np.stack((x1 - x0, x2 - x0, x3 - x0), axis=-1)) / 6.0


class Example:
    def __init__(self, viewer, args):
        self.viewer = viewer
        self.solver_type = args.solver
        self.sim_time = 0.0
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = 32
        self.sim_dt = self.frame_dt / self.sim_substeps

        if self.solver_type not in {"semi_implicit", "xpbd", "vbd"}:
            raise ValueError("The rigid soft contact example supports the semi_implicit, xpbd, and vbd solvers.")

        if self.solver_type in {"semi_implicit", "xpbd"}:
            # Share the same scene material for force-based and XPBD solves.
            # XPBD rigid-soft contact is a positional projection and ignores
            # the normal contact stiffness, but SemiImplicit uses it as a
            # penalty stiffness, so keep the value low enough for visible
            # penetration while the tet material carries the shape recovery.
            sphere_contact_cfg = newton.ModelBuilder.ShapeConfig(
                density=RIGID_SOFT_SPHERE_DENSITY,
                ke=RIGID_SOFT_CONTACT_KE,
                kd=RIGID_SOFT_CONTACT_KD,
                kf=RIGID_SOFT_CONTACT_KF,
                mu=RIGID_SOFT_CONTACT_MU,
            )
            ground_contact_cfg = sphere_contact_cfg.copy()
            ground_contact_cfg.ke = GROUND_CONTACT_KE
        else:
            sphere_contact_cfg = newton.ModelBuilder.ShapeConfig(
                density=RIGID_SOFT_SPHERE_DENSITY,
                ke=1.0e5,
                kd=1.0e-4,
                kf=1.0e3,
                mu=0.3,
            )
            ground_contact_cfg = sphere_contact_cfg.copy()
            ground_contact_cfg.ke = 1.0e5
            ground_contact_cfg.mu = 0.5

        builder = newton.ModelBuilder()
        builder.default_particle_radius = 0.01
        builder.particle_max_velocity = 50.0
        builder.add_ground_plane(cfg=ground_contact_cfg)

        builder.add_soft_grid(
            pos=wp.vec3(0.0, 0.0, 0.0),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0, 0.0, 0.0),
            dim_x=GRID_DIM_X,
            dim_y=GRID_DIM_Y,
            dim_z=GRID_DIM_Z,
            cell_x=GRID_CELL_SIZE,
            cell_y=GRID_CELL_SIZE,
            cell_z=GRID_CELL_SIZE,
            density=SOFT_GRID_DENSITY,
            k_mu=SOFT_GRID_K_MU,
            k_lambda=SOFT_GRID_K_LAMBDA,
            k_damp=SOFT_GRID_K_DAMP,
        )

        # Warp's original example is y-up; Newton examples are z-up.
        sphere_body = builder.add_body(
            xform=wp.transform(wp.vec3(0.5, 0.5, SPHERE_INITIAL_Z), wp.quat_identity()),
            label="sphere",
        )
        builder.add_shape_sphere(
            sphere_body,
            radius=SPHERE_RADIUS,
            cfg=sphere_contact_cfg,
            color=wp.vec3(0.95, 0.43, 0.18),
            label="rigid_sphere",
        )

        if self.solver_type == "vbd":
            builder.color()

        self.model = builder.finalize()
        if self.solver_type in {"semi_implicit", "xpbd"}:
            self.model.soft_contact_ke = RIGID_SOFT_CONTACT_KE
            self.model.soft_contact_kd = RIGID_SOFT_CONTACT_KD
            self.model.soft_contact_kf = RIGID_SOFT_CONTACT_KF
            self.model.soft_contact_mu = RIGID_SOFT_CONTACT_MU
        elif self.solver_type == "vbd":
            self.model.soft_contact_ke = 1.0e5
            self.model.soft_contact_kd = 1.0e-4
            self.model.soft_contact_kf = 1.0e3
            self.model.soft_contact_mu = 0.3

        if self.solver_type == "semi_implicit":
            self.solver = newton.solvers.SolverSemiImplicit(model=self.model)
        elif self.solver_type == "xpbd":
            self.solver = newton.solvers.SolverXPBD(
                model=self.model,
                iterations=20,
                soft_body_relaxation=0.55,
                soft_contact_relaxation=0.5,
            )
        elif self.solver_type == "vbd":
            self.solver = newton.solvers.SolverVBD(
                model=self.model,
                iterations=10,
                particle_enable_self_contact=False,
                particle_enable_tile_solve=False,
                rigid_contact_hard=False,
                rigid_body_particle_contact_buffer_size=512,
            )

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.model.contacts()
        self.rest_particle_q = np.array(self.state_0.particle_q.numpy(), copy=True)
        self.tet_indices = np.array(self.model.tet_indices.numpy(), copy=True)
        self.rest_tet_volumes = _tet_volumes(self.rest_particle_q, self.tet_indices)

        self.viewer.set_model(self.model)
        self.viewer.set_camera(
            pos=wp.vec3(4.0, -5.0, 3.0),
            pitch=-25.0,
            yaw=135.0,
        )
        if hasattr(self.viewer, "camera") and hasattr(self.viewer.camera, "fov"):
            self.viewer.camera.fov = 45.0

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
        particle_q = self.state_0.particle_q.numpy()
        body_q = self.state_0.body_q.numpy()

        min_pos = np.min(particle_q, axis=0)
        max_pos = np.max(particle_q, axis=0)
        bbox_extent = max_pos - min_pos
        bbox_size = np.linalg.norm(max_pos - min_pos)
        sphere_z = body_q[0, 2]

        assert bbox_size < 6.0, f"Soft body exploded: bbox_size={bbox_size:.2f}"
        assert min_pos[2] > -0.1, f"Soft body penetrated the ground: z_min={min_pos[2]:.4f}"
        assert 0.5 < sphere_z < 2.6, f"Sphere left expected vertical range: z={sphere_z:.4f}"

        # Regression check for an XPBD tuning failure where the off-center drop
        # permanently folded the soft-grid corners inward after impact.
        horizontal_translation = np.mean(particle_q[:, :2], axis=0) - np.mean(self.rest_particle_q[:, :2], axis=0)
        recovered_corner_xy = particle_q[GRID_CORNER_INDICES, :2] - horizontal_translation
        corner_xy_drift = np.linalg.norm(
            recovered_corner_xy - self.rest_particle_q[GRID_CORNER_INDICES, :2],
            axis=1,
        )
        max_corner_xy_drift = np.max(corner_xy_drift)
        assert max_corner_xy_drift < 0.25, f"Soft grid corners did not recover: drift={max_corner_xy_drift:.4f}"

        rest_extent = np.max(self.rest_particle_q, axis=0) - np.min(self.rest_particle_q, axis=0)
        assert bbox_extent[0] < rest_extent[0] + 0.35, f"Soft grid stretched too far in x: extent={bbox_extent[0]:.4f}"
        assert bbox_extent[1] < rest_extent[1] + 0.30, f"Soft grid stretched too far in y: extent={bbox_extent[1]:.4f}"
        assert bbox_extent[2] < rest_extent[2] + 0.30, f"Soft grid stretched too far in z: extent={bbox_extent[2]:.4f}"

        tet_volumes = _tet_volumes(particle_q, self.tet_indices)
        assert np.min(tet_volumes) > 0.0, "Soft grid contains inverted tetrahedra"
        volume_ratio = tet_volumes / self.rest_tet_volumes
        assert np.min(volume_ratio) > 0.2, f"Soft grid has collapsed tets: ratio={np.min(volume_ratio):.4f}"
        assert np.max(volume_ratio) < 1.25, f"Soft grid has over-expanded tets: ratio={np.max(volume_ratio):.4f}"

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.add_argument(
            "--solver",
            help="Type of solver",
            type=str,
            choices=["semi_implicit", "xpbd", "vbd"],
            default="xpbd",
        )
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args)
    newton.examples.run(example, args)
