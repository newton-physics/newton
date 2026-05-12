# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Rigid-MPM Coupled Solver
#
# Rigid boxes driven by MuJoCo or Kamino fall into a granular MPM bed.  The rigid bodies
# are exposed to MPM as proxy colliders in the same shared model, and MPM
# grid impulses are harvested back into rigid-body wrenches.
#
# Command: python -m newton.examples mujoco_mpm_coupled_solver
#
###########################################################################

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import warp as wp

import newton
import newton.examples
from newton.solvers import ModelView, SolverImplicitMPM, SolverKamino, SolverMuJoCo, SolverProxyCoupled


def _add_rigid_solver_arg(parser) -> None:
    parser.add_argument(
        "--rigid-solver",
        help="Rigid-body solver used by the coupled path.",
        type=str,
        choices=["mujoco", "kamino"],
        default="mujoco",
    )


def _configure_kamino_rigid_view(view: ModelView) -> None:
    view.particle_count = 0
    view.spring_count = 0
    view.tri_count = 0
    view.edge_count = 0
    view.tet_count = 0
    view.muscle_count = 0
    view.equality_constraint_count = 0


def _register_rigid_solver_custom_attributes(builder: newton.ModelBuilder, rigid_solver: str) -> None:
    if rigid_solver == "kamino":
        SolverKamino.register_custom_attributes(builder)


def _make_kamino_config() -> SolverKamino.Config:
    config = SolverKamino.Config()
    config.use_collision_detector = False
    config.use_fk_solver = False
    config.dynamics.preconditioning = True
    config.padmm.max_iterations = 120
    config.padmm.primal_tolerance = 1.0e-5
    config.padmm.dual_tolerance = 1.0e-5
    config.padmm.compl_tolerance = 1.0e-5
    config.padmm.rho_0 = 0.1
    config.padmm.use_acceleration = True
    config.padmm.warmstart_mode = "containers"
    return config


def _rigid_solver_entry_args(
    rigid_solver: str,
    *,
    mujoco_kwargs: dict[str, object] | None = None,
):
    if rigid_solver == "kamino":
        return "kamino", SolverKamino, {"config": _make_kamino_config()}, _configure_kamino_rigid_view
    if rigid_solver == "mujoco":
        return "mjc", SolverMuJoCo, dict(mujoco_kwargs or {}), None
    raise ValueError(f"Unsupported rigid solver '{rigid_solver}'")


def _capture_frame_graph(model: newton.Model, simulate: Callable[[], None], *, enabled: bool = True):
    if not enabled or not model.device.is_cuda:
        return None

    with wp.ScopedDevice(model.device):
        with wp.ScopedCapture() as capture:
            simulate()

    if capture.graph is None:
        raise RuntimeError(f"CUDA graph capture failed on device {model.device}")
    return capture.graph


def _launch_frame_graph(model: newton.Model, graph) -> bool:
    if graph is None:
        return False

    with wp.ScopedDevice(model.device):
        wp.capture_launch(graph)
    return True


class Example:
    def __init__(self, viewer, args):
        self.viewer = viewer
        self.sim_time = 0.0
        self.fps = 100
        self.frame_dt = 1.0 / self.fps
        self.rigid_solver = getattr(args, "rigid_solver", "mujoco")

        builder = newton.ModelBuilder()
        builder.default_shape_cfg.mu = 0.5
        _register_rigid_solver_custom_attributes(builder, self.rigid_solver)
        SolverImplicitMPM.register_custom_attributes(builder)

        rigid_body_start = builder.body_count
        self._emit_rigid_bodies(builder)
        rigid_body_end = builder.body_count
        builder.add_ground_plane()

        voxel_size = 0.05
        self._emit_particles(builder, voxel_size)

        self.model = builder.finalize()

        mpm_config = SolverImplicitMPM.Config()
        mpm_config.voxel_size = voxel_size
        mpm_config.grid_type = "fixed"
        mpm_config.grid_padding = 50
        mpm_config.max_active_cell_count = 1 << 15
        mpm_config.strain_basis = "P0"
        mpm_config.max_iterations = 50
        mpm_config.critical_fraction = 0.0

        rigid_name, rigid_solver, rigid_kwargs, rigid_configure_view = _rigid_solver_entry_args(
            self.rigid_solver,
            mujoco_kwargs={"use_mujoco_contacts": False, "njmax": 100},
        )
        rigid_body_indices = wp.array(
            list(range(rigid_body_start, rigid_body_end)),
            dtype=int,
            device=self.model.device,
        )
        self.solver = SolverProxyCoupled(
            model=self.model,
            entries=[
                SolverProxyCoupled.Entry(
                    name=rigid_name,
                    solver=rigid_solver,
                    bodies=[int(i) for i in rigid_body_indices.numpy()],
                    joints=list(range(self.model.joint_count)),
                    solver_kwargs=rigid_kwargs,
                    configure_view=rigid_configure_view,
                    substeps=args.rigid_substeps,
                ),
                SolverProxyCoupled.Entry(
                    name="mpm",
                    solver=SolverImplicitMPM,
                    particles=list(range(self.model.particle_count)),
                    solver_kwargs={"config": mpm_config},
                    in_place=True,
                ),
            ],
            coupling=SolverProxyCoupled.CouplingProxy(
                proxies=[
                    SolverProxyCoupled.Proxy(
                        source=rigid_name,
                        destination="mpm",
                        bodies=[int(i) for i in rigid_body_indices.numpy()],
                        mass_scale=getattr(args, "proxy_mass_relaxation", 1.0),
                        mode=SolverProxyCoupled.ProxyMode.LAGGED,
                        # MPM handles collider contact internally; no proxy
                        # collision pipeline should generate Contacts here.
                        collision_pipeline=lambda _model: None,
                    )
                ],
                iterations=args.proxy_iterations,
            ),
        )
        self.mpm_solver = self.solver.get_solver("mpm")

        self.state_0 = self.model.state()
        self.rigid_collision_pipeline = newton.CollisionPipeline(self.model, soft_contact_max=0)
        self.contacts = self.rigid_collision_pipeline.contacts()
        self.control = self.model.control()

        self.viewer.set_model(self.model)
        if isinstance(self.viewer, newton.viewer.ViewerGL):
            self.viewer.register_ui_callback(self.render_ui, position="side")
        self.viewer.show_particles = True
        self.show_impulses = False

        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)

        self.particle_render_colors = wp.full(
            self.model.particle_count,
            value=wp.vec3(0.7, 0.6, 0.4),
            dtype=wp.vec3,
            device=self.model.device,
        )

        self.graph = None
        self.capture()

    def capture(self):
        self.graph = _capture_frame_graph(self.model, self.simulate)

    def simulate(self):
        self.state_0.clear_forces()
        self.viewer.apply_forces(self.state_0)
        self.rigid_collision_pipeline.collide(self.state_0, self.contacts)
        self.solver.step(self.state_0, self.state_0, self.control, self.contacts, self.frame_dt)

    def step(self):
        if not _launch_frame_graph(self.model, self.graph):
            self.simulate()
        self.sim_time += self.frame_dt

    def test_final(self):
        newton.examples.test_body_state(
            self.model,
            self.state_0,
            "all bodies are above the sand",
            lambda q, qd: q[2] > 0.45,
        )
        voxel_size = self.mpm_solver.voxel_size
        newton.examples.test_particle_state(
            self.state_0,
            "all particles are above the ground",
            lambda q, qd: q[2] > -voxel_size,
        )

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)

        self.viewer.log_points(
            "/sand",
            points=self.state_0.particle_q,
            radii=self.model.particle_radius,
            colors=self.particle_render_colors,
            hidden=not self.viewer.show_particles,
        )

        if self.show_impulses:
            impulses, pos, _cid = self.mpm_solver.collect_collider_impulses(self.state_0)
            self.viewer.log_lines(
                "/impulses",
                starts=pos,
                ends=pos + impulses,
                colors=wp.full(pos.shape[0], value=wp.vec3(1.0, 0.0, 0.0), dtype=wp.vec3),
            )
        else:
            self.viewer.log_lines("/impulses", None, None, None)

        self.viewer.end_frame()

    def render_ui(self, imgui):
        _changed, self.show_impulses = imgui.checkbox("Show Impulses", self.show_impulses)

    def _emit_rigid_bodies(self, builder: newton.ModelBuilder):
        drop_z = 2.0
        offsets_xy = [
            (0.00, 0.00),
            (0.10, 0.00),
            (-0.10, 0.00),
            (0.00, 0.10),
            (0.00, -0.10),
            (0.10, 0.10),
            (-0.10, 0.10),
            (0.10, -0.10),
            (-0.10, -0.10),
            (0.15, 0.00),
            (-0.15, 0.00),
            (0.00, 0.15),
        ]
        boxes = [
            (0.25, 0.35, 0.25),
            (0.25, 0.25, 0.25),
            (0.3, 0.2, 0.2),
            (0.25, 0.35, 0.25),
            (0.25, 0.25, 0.25),
            (0.3, 0.2, 0.2),
        ]

        for box_index, (hx, hy, hz) in enumerate(boxes):
            ox, oy = offsets_xy[box_index % len(offsets_xy)]
            pz = drop_z + float(box_index) * 0.6
            body = builder.add_body(
                xform=wp.transform(p=wp.vec3(float(ox), float(oy), pz), q=wp.quat_identity()),
                mass=75.0,
            )
            builder.add_shape_box(body, hx=float(hx), hy=float(hy), hz=float(hz))

    def _emit_particles(self, builder: newton.ModelBuilder, voxel_size: float):
        particles_per_cell = 3.0
        density = 2500.0

        bed_lo = np.array([-1.0, -1.0, 0.0])
        bed_hi = np.array([1.0, 1.0, 0.5])
        bed_res = np.array(np.ceil(particles_per_cell * (bed_hi - bed_lo) / voxel_size), dtype=int)

        cell_size = (bed_hi - bed_lo) / bed_res
        cell_volume = np.prod(cell_size)
        radius = float(np.max(cell_size) * 0.5)
        mass = float(np.prod(cell_volume) * density)

        builder.add_particle_grid(
            pos=wp.vec3(bed_lo),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0),
            dim_x=bed_res[0] + 1,
            dim_y=bed_res[1] + 1,
            dim_z=bed_res[2] + 1,
            cell_x=cell_size[0],
            cell_y=cell_size[1],
            cell_z=cell_size[2],
            mass=mass,
            jitter=2.0 * radius,
            radius_mean=radius,
            custom_attributes={"mpm:friction": 0.75},
        )

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        _add_rigid_solver_arg(parser)
        parser.add_argument(
            "--proxy-mass-relaxation",
            "-pmr",
            help="Scale factor for proxy collider mass in MPM (< 1 = softer coupling)",
            type=float,
            default=1.0,
        )
        parser.add_argument(
            "--proxy-iterations",
            help="Number of proxy relaxation passes per coupled step",
            type=int,
            default=1,
        )
        parser.add_argument(
            "--rigid-substeps",
            "--mujoco-substeps",
            dest="rigid_substeps",
            help="Number of rigid-solver substeps per coupled step",
            type=int,
            default=4,
        )
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args)
    newton.examples.run(example, args)
