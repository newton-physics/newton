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
from newton.solvers.experimental.coupled import SolverCoupledProxy

import newton
import newton.examples
from newton.solvers import SolverKamino, SolverMuJoCo, SolverSemiImplicit, SolverVBD, SolverXPBD
from newton.viewer import ViewerBase

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


def _normalized_rigid_solver_name(rigid_solver: str) -> str:
    if rigid_solver == "mujoco":
        return "mjc"
    return rigid_solver


def _register_rigid_solver_custom_attributes(builder: newton.ModelBuilder, rigid_solver: str) -> None:
    if _normalized_rigid_solver_name(rigid_solver) == "kamino":
        SolverKamino.register_custom_attributes(builder)


def _make_kamino_config() -> SolverKamino.Config:
    config = SolverKamino.Config()
    config.use_collision_detector = False
    config.use_fk_solver = False
    config.dynamics.preconditioning = True
    config.padmm.max_iterations = 80
    config.padmm.primal_tolerance = 1.0e-5
    config.padmm.dual_tolerance = 1.0e-5
    config.padmm.compl_tolerance = 1.0e-5
    config.padmm.rho_0 = 0.1
    config.padmm.use_acceleration = True
    config.padmm.warmstart_mode = "containers"
    return config


def _rigid_solver_entry_args(rigid_solver: str):
    rigid_solver = _normalized_rigid_solver_name(rigid_solver)
    if rigid_solver == "vbd":
        return "avbd", SolverVBD, {"iterations": 10}
    if rigid_solver == "kamino":
        return "kamino", SolverKamino, {"config": _make_kamino_config()}
    if rigid_solver == "mjc":
        return "mjc", SolverMuJoCo, {"use_mujoco_contacts": False, "njmax": 64}
    raise ValueError(f"Unsupported rigid solver {rigid_solver!r}")


def _soft_solver_entry_args(soft_solver: str, args):
    if soft_solver == "semi_implicit":
        return "semi_implicit", SolverSemiImplicit, {}
    if soft_solver == "xpbd":
        return "xpbd", SolverXPBD, {"iterations": args.xpbd_iterations}
    if soft_solver == "vbd":
        return (
            "vbd",
            SolverVBD,
            {
                "iterations": args.vbd_iterations,
                "particle_enable_self_contact": False,
                "particle_enable_tile_solve": False,
                "rigid_contact_hard": False,
                "rigid_body_particle_contact_buffer_size": 512,
            },
        )
    raise ValueError(f"Unsupported soft solver {soft_solver!r}")


class Example:
    def __init__(self, viewer: ViewerBase, args):
        self.viewer = viewer
        self.solver_type = args.solver
        self.rigid_solver = _normalized_rigid_solver_name(args.rigid_solver)
        self.soft_solver = args.soft_solver if self.solver_type == "coupled" else self.solver_type
        self.sim_time = 0.0
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = 32
        self.sim_dt = self.frame_dt / self.sim_substeps

        if self.solver_type not in {"semi_implicit", "xpbd", "vbd", "coupled"}:
            raise ValueError(
                "The rigid soft contact example supports the semi_implicit, xpbd, vbd, and coupled solvers."
            )

        if self.soft_solver in {"semi_implicit", "xpbd"}:
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
        if self.solver_type == "coupled":
            _register_rigid_solver_custom_attributes(builder, self.rigid_solver)
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
        sphere_xform = wp.transform(wp.vec3(0.2, 0.5, SPHERE_INITIAL_Z), wp.quat_identity())
        if self.solver_type == "coupled":
            sphere_body = builder.add_link(xform=sphere_xform, label="sphere")
            sphere_joint = builder.add_joint_free(child=sphere_body, label="sphere_free")
            builder.add_articulation([sphere_joint], label="sphere")
        else:
            sphere_body = builder.add_body(xform=sphere_xform, label="sphere")
            sphere_joint = None
        builder.add_shape_sphere(
            sphere_body,
            radius=SPHERE_RADIUS,
            cfg=sphere_contact_cfg,
            color=wp.vec3(0.95, 0.43, 0.18),
            label="rigid_sphere",
        )

        if "vbd" in (self.soft_solver, self.rigid_solver):
            builder.color()

        self.model = builder.finalize()
        if self.soft_solver in {"semi_implicit", "xpbd"}:
            self.model.soft_contact_ke = RIGID_SOFT_CONTACT_KE
            self.model.soft_contact_kd = RIGID_SOFT_CONTACT_KD
            self.model.soft_contact_kf = RIGID_SOFT_CONTACT_KF
            self.model.soft_contact_mu = RIGID_SOFT_CONTACT_MU
        elif self.soft_solver == "vbd":
            self.model.soft_contact_ke = 1.0e5
            self.model.soft_contact_kd = 1.0e-4
            self.model.soft_contact_kf = 1.0e3
            self.model.soft_contact_mu = 0.3

        if self.solver_type == "semi_implicit":
            self.solver = SolverSemiImplicit(model=self.model)
        elif self.solver_type == "xpbd":
            self.solver = SolverXPBD(
                model=self.model,
                iterations=10,
            )
        elif self.solver_type == "vbd":
            self.solver = SolverVBD(
                model=self.model,
                iterations=10,
                particle_enable_self_contact=False,
                particle_enable_tile_solve=False,
                rigid_contact_hard=False,
                rigid_body_particle_contact_buffer_size=512,
            )
        elif self.solver_type == "coupled":
            rigid_name, rigid_solver, rigid_kwargs = _rigid_solver_entry_args(self.rigid_solver)
            soft_name, soft_solver, soft_kwargs = _soft_solver_entry_args(self.soft_solver, args)
            particle_indices = list(range(self.model.particle_count))
            self.solver = SolverCoupledProxy(
                model=self.model,
                entries=[
                    SolverCoupledProxy.Entry(
                        name=rigid_name,
                        solver=lambda v: rigid_solver(model=v, **rigid_kwargs),
                        bodies=[sphere_body],
                        joints=[sphere_joint],
                    ),
                    SolverCoupledProxy.Entry(
                        name=soft_name,
                        solver=lambda v: soft_solver(model=v, **soft_kwargs),
                        particles=particle_indices,
                    ),
                ],
                coupling=SolverCoupledProxy.Config(
                    proxies=[
                        SolverCoupledProxy.Proxy(
                            source=rigid_name,
                            destination=soft_name,
                            bodies=[sphere_body],
                            mass_scale=args.mass_scale,
                            mode=args.coupling_mode,
                            collision_pipeline=lambda model: newton.examples.create_collision_pipeline(model, args),
                            collide_interval=1,
                        )
                    ],
                    iterations=args.proxy_iterations,
                ),
            )

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.collision_pipeline = newton.CollisionPipeline(self.model)
        self.contacts = self.collision_pipeline.contacts()

        newton.examples.configure_coupled_view(self, args)
        self.viewer.set_camera(
            pos=wp.vec3(1.0, -6.4, 3.0),
            pitch=-14.0,
            yaw=96.0,
        )

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

            newton.examples.apply_coupled_viewer_forces(self, self.state_0)
            self.collision_pipeline.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)

            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()

        self.sim_time += self.frame_dt

    def test_final(self):
        def _grid_index(x, y, z):
            return (GRID_DIM_X + 1) * (GRID_DIM_Y + 1) * z + (GRID_DIM_X + 1) * y + x

        def _tet_volumes(particle_q, tet_indices):
            x0 = particle_q[tet_indices[:, 0]]
            x1 = particle_q[tet_indices[:, 1]]
            x2 = particle_q[tet_indices[:, 2]]
            x3 = particle_q[tet_indices[:, 3]]
            return np.linalg.det(np.stack((x1 - x0, x2 - x0, x3 - x0), axis=-1)) / 6.0

        grid_corner_indices = np.array(
            [_grid_index(x, y, z) for x in (0, GRID_DIM_X) for y in (0, GRID_DIM_Y) for z in (0, GRID_DIM_Z)],
            dtype=np.int32,
        )

        particle_q = self.state_0.particle_q.numpy()
        body_q = self.state_0.body_q.numpy()

        rest_particle_q = np.array(self.model.particle_q.numpy())
        tet_indices = np.array(self.model.tet_indices.numpy())

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
        horizontal_translation = np.mean(particle_q[:, :2], axis=0) - np.mean(rest_particle_q[:, :2], axis=0)
        recovered_corner_xy = particle_q[grid_corner_indices, :2] - horizontal_translation
        corner_xy_drift = np.linalg.norm(
            recovered_corner_xy - rest_particle_q[grid_corner_indices, :2],
            axis=1,
        )
        max_corner_xy_drift = np.max(corner_xy_drift)
        assert max_corner_xy_drift < 0.25, f"Soft grid corners did not recover: drift={max_corner_xy_drift:.4f}"

        rest_extent = np.max(rest_particle_q, axis=0) - np.min(rest_particle_q, axis=0)
        assert bbox_extent[0] < rest_extent[0] + 0.35, f"Soft grid stretched too far in x: extent={bbox_extent[0]:.4f}"
        assert bbox_extent[1] < rest_extent[1] + 0.30, f"Soft grid stretched too far in y: extent={bbox_extent[1]:.4f}"
        assert bbox_extent[2] < rest_extent[2] + 0.30, f"Soft grid stretched too far in z: extent={bbox_extent[2]:.4f}"

        tet_volumes = _tet_volumes(particle_q, tet_indices)
        rest_tet_volumes = _tet_volumes(rest_particle_q, tet_indices)
        assert np.min(tet_volumes) > 0.0, "Soft grid contains inverted tetrahedra"
        volume_ratio = tet_volumes / rest_tet_volumes
        assert np.min(volume_ratio) > 0.2, f"Soft grid has collapsed tets: ratio={np.min(volume_ratio):.4f}"
        assert np.max(volume_ratio) < 1.25, f"Soft grid has over-expanded tets: ratio={np.max(volume_ratio):.4f}"

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        newton.examples.log_coupled_view(self, self.contacts)
        self.viewer.end_frame()

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        newton.examples.add_coupled_view_args(parser)
        parser.add_argument(
            "--solver",
            help="Type of solver",
            type=str,
            choices=["semi_implicit", "xpbd", "vbd", "coupled"],
            default="xpbd",
        )
        parser.add_argument(
            "--rigid-solver",
            help="Rigid solver used by --solver coupled",
            type=str,
            choices=["mjc", "mujoco", "kamino", "vbd"],
            default="mjc",
        )
        parser.add_argument(
            "--soft-solver",
            help="Soft-body solver used by --solver coupled",
            type=str,
            choices=["semi_implicit", "xpbd", "vbd"],
            default="vbd",
        )
        parser.add_argument(
            "--coupling-mode",
            help="Proxy state transfer mode",
            type=str,
            choices=["lagged", "staggered"],
            default="lagged",
        )
        parser.add_argument(
            "--mass-scale",
            "-pmr",
            help="Scale factor for rigid effective mass/inertia used by the soft-solver proxy",
            type=float,
            default=1.0,
        )
        parser.add_argument(
            "--proxy-iterations",
            help="Number of proxy relaxation passes per substep",
            type=int,
            default=1,
        )
        parser.add_argument(
            "--xpbd-iterations",
            help="XPBD solver iterations per substep in coupled mode",
            type=int,
            default=10,
        )
        parser.add_argument(
            "--vbd-iterations",
            help="VBD solver iterations per substep in coupled mode",
            type=int,
            default=10,
        )
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args)
    newton.examples.run(example, args)
