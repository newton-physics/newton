# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Rigid-VBD Coupled Solver
#
# Rigid boxes and an articulated pendulum chain (driven by MuJoCo or Kamino)
# interact with a cloth sheet and several soft bodies (simulated by
# VBD).  Contact forces from the deformables are fed back to the
# rigid bodies, causing them to bounce and settle realistically.
#
# This example builds rigid/VBD proxy coupling directly through
# ``SolverProxyCoupled`` so the generic proxy path exercises the default API.
#
# Pass ``--solver vbd`` to run the same scene with a single VBD solver
# (no coupling) as a reference baseline.
#
# Command: python -m newton.examples mujoco_vbd_coupled_solver
#          python -m newton.examples mujoco_vbd_coupled_solver --solver vbd
#
###########################################################################

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import warp as wp

import newton
import newton.examples
from newton.solvers import ModelView, SolverKamino, SolverMuJoCo, SolverProxyCoupled, SolverVBD


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
        self.args = args
        self.viewer = viewer
        self.sim_time = 0.0
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = 8
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.use_coupled = getattr(args, "solver", "coupled") == "coupled"

        self.rigid_solver = getattr(args, "rigid_solver", "mujoco")

        builder = newton.ModelBuilder()
        _register_rigid_solver_custom_attributes(builder, self.rigid_solver)
        builder.add_ground_plane()

        # ---- Rigid bodies (free-floating + articulated) ----
        rigid_body_start = builder.body_count
        self._emit_rigid_bodies(builder)
        self._emit_articulated_chain(builder)
        rigid_body_end = builder.body_count

        # ---- Cloth ----
        self._emit_cloth(builder)

        # ---- Soft bodies (tetrahedral, owned by VBD) ----
        self._emit_soft_bodies(builder)

        # Color the mesh for VBD solver
        builder.color()

        self.model = builder.finalize()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.model)

        # Contact parameters
        self.model.soft_contact_ke = 1.0e5
        self.model.soft_contact_kd = 0.0
        self.model.soft_contact_mu = 0.5
        # VBD now damps body-particle contacts from full relative body/particle
        # motion. Keep material damping disabled here so proxy impact forces stay
        # close to the pre-rebase penalty-contact behavior.
        self.model.shape_material_kd.fill_(0.0)

        vbd_kwargs = {
            "iterations": 10,
            "friction_epsilon": 0.01,
            "particle_enable_self_contact": True,
            "particle_self_contact_radius": 0.01,
            "particle_self_contact_margin": 0.01,
            # Match the pre-AVBD-hard-constraint defaults used by this scene by
            # default, while allowing hard-contact experiments from the CLI.
            "rigid_contact_hard": args.rigid_contact_hard,
            "rigid_avbd_beta": 1.0e5,
            "rigid_avbd_gamma": 0.99,
            "rigid_contact_k_start": 1.0e2,
            "rigid_joint_linear_k_start": 1.0e4,
            "rigid_joint_angular_k_start": 1.0e1,
            "rigid_joint_linear_ke": 1.0e9,
            "rigid_joint_angular_ke": 1.0e9,
            "rigid_joint_linear_kd": 1.0e-2,
            "rigid_joint_angular_kd": 0.0,
        }

        if self.use_coupled:
            # ---------- Coupled path: rigid solver + VBD ----------
            rigid_name, rigid_solver, rigid_kwargs, rigid_configure_view = _rigid_solver_entry_args(
                self.rigid_solver,
                mujoco_kwargs={"use_mujoco_contacts": False, "njmax": 200},
            )
            rigid_body_indices = wp.array(list(range(rigid_body_start, rigid_body_end)), dtype=int)
            vbd_body_indices = wp.array(
                [i for i in range(self.model.body_count) if i < rigid_body_start or i >= rigid_body_end],
                dtype=int,
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
                    ),
                    SolverProxyCoupled.Entry(
                        name="vbd",
                        solver=SolverVBD,
                        bodies=[int(i) for i in vbd_body_indices.numpy()],
                        particles=list(range(self.model.particle_count)),
                        solver_kwargs=vbd_kwargs,
                    ),
                ],
                coupling=SolverProxyCoupled.CouplingProxy(
                    proxies=[
                        SolverProxyCoupled.Proxy(
                            source=rigid_name,
                            destination="vbd",
                            bodies=[int(i) for i in rigid_body_indices.numpy()],
                            mass_scale=args.proxy_mass_relaxation,
                            mode=args.coupling_mode,
                            collision_pipeline=lambda model: newton.examples.create_collision_pipeline(
                                model, self.args
                            ),
                            collide_interval=1,
                        )
                    ],
                    iterations=args.proxy_iterations,
                ),
            )

        else:
            # ---------- Pure-VBD path (reference baseline) ----------
            self.solver = SolverVBD(model=self.model, **vbd_kwargs)

        # Simulation state
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.contacts = self.model.contacts()
        self.control = self.model.control()

        self.viewer.set_model(self.model)

        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)

        self.capture()

    def capture(self):
        self.graph = _capture_frame_graph(self.model, self.simulate)

    def simulate(self):
        self.model.collide(self.state_0, self.contacts)
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        if not _launch_frame_graph(self.model, self.graph):
            self.simulate()
        self.sim_time += self.frame_dt

    def test_final(self):
        # Rigid bodies should have settled onto the cloth above the ground
        newton.examples.test_body_state(
            self.model,
            self.state_0,
            "all rigid bodies are above the ground",
            lambda q, qd: q[2] > -0.1,
        )
        # Particles (cloth) should not have exploded
        particle_q = self.state_0.particle_q.numpy()
        min_pos = np.min(particle_q, axis=0)
        max_pos = np.max(particle_q, axis=0)
        bbox_size = np.linalg.norm(max_pos - min_pos)
        assert bbox_size < 20.0, f"Bounding box exploded: size={bbox_size:.2f}"
        assert min_pos[2] > -0.5, f"Excessive penetration: z_min={min_pos[2]:.4f}"

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()

    def _emit_rigid_bodies(self, builder: newton.ModelBuilder):
        """Add a few rigid boxes above the cloth."""
        boxes = [
            # (position, half-extents, mass)
            (wp.vec3(0.0, 0.0, 2.0), (0.15, 0.15, 0.15), 10.0),
            (wp.vec3(0.3, 0.1, 2.5), (0.10, 0.20, 0.10), 5.0),
            (wp.vec3(-0.2, -0.1, 3.0), (0.12, 0.12, 0.12), 8.0),
        ]
        for pos, (hx, hy, hz), mass in boxes:
            body = builder.add_body(
                xform=wp.transform(p=pos, q=wp.quat_identity()),
                mass=mass,
            )
            builder.add_shape_box(body, hx=hx, hy=hy, hz=hz)

    def _emit_cloth(self, builder: newton.ModelBuilder):
        """Add a cloth sheet at z=1.0, fixed at left and right edges."""
        builder.add_cloth_grid(
            pos=wp.vec3(-0.5, -0.5, 1.0),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0),
            fix_left=True,
            fix_right=True,
            dim_x=30,
            dim_y=30,
            cell_x=1.0 / 30.0,
            cell_y=1.0 / 30.0,
            mass=0.1,
            tri_ke=1e5,
            tri_ka=1e5,
            tri_kd=1e-2,
            edge_ke=0.01,
            edge_kd=1e-2,
            particle_radius=0.01,
        )

    def _emit_articulated_chain(self, builder: newton.ModelBuilder):
        """Add a 3-link pendulum chain anchored to the world.

        The rigid solver owns the joints; VBD sees the links as disjoint proxy bodies.
        """
        hx, hy, hz = 0.21, 0.05, 0.05
        damping = 5.0
        anchor = wp.vec3(0.6, 0.0, 2.25)

        link_0 = builder.add_link()
        builder.add_shape_box(link_0, hx=hx, hy=hy, hz=hz)
        j0 = builder.add_joint_revolute(
            parent=-1,
            child=link_0,
            axis=wp.vec3(0.0, 1.0, 0.0),
            target_kd=damping,
            parent_xform=wp.transform(p=anchor, q=wp.quat_identity()),
            child_xform=wp.transform(p=wp.vec3(-hx, 0.0, 0.0), q=wp.quat_identity()),
        )

        link_1 = builder.add_link()
        builder.add_shape_box(link_1, hx=hx, hy=hy, hz=hz)
        j1 = builder.add_joint_revolute(
            parent=link_0,
            child=link_1,
            axis=wp.vec3(0.0, 1.0, 0.0),
            target_kd=damping,
            parent_xform=wp.transform(p=wp.vec3(hx, 0.0, 0.0), q=wp.quat_identity()),
            child_xform=wp.transform(p=wp.vec3(-hx, 0.0, 0.0), q=wp.quat_identity()),
        )

        link_2 = builder.add_link()
        builder.add_shape_box(link_2, hx=hx, hy=hy, hz=hz)
        j2 = builder.add_joint_revolute(
            parent=link_1,
            child=link_2,
            axis=wp.vec3(0.0, 1.0, 0.0),
            target_kd=damping,
            parent_xform=wp.transform(p=wp.vec3(hx, 0.0, 0.0), q=wp.quat_identity()),
            child_xform=wp.transform(p=wp.vec3(-hx, 0.0, 0.0), q=wp.quat_identity()),
        )

        builder.add_articulation([j0, j1, j2], label="pendulum")

    def _emit_soft_bodies(self, builder: newton.ModelBuilder):
        """Add several volumetric soft bodies above the cloth."""
        grids = [
            # (position, dims, cell_size, density, stiffness)
            (wp.vec3(-0.15, -0.15, 1.3), (3, 3, 3), 0.07, 1.0e3, 1.0e6),
            (wp.vec3(0.25, 0.20, 1.5), (2, 2, 4), 0.07, 1.0e3, 1.0e6),
            (wp.vec3(-0.30, 0.25, 1.8), (2, 4, 2), 0.07, 1.0e3, 1.0e6),
        ]
        for pos, (dx, dy, dz), cell, density, stiffness in grids:
            builder.add_soft_grid(
                pos=pos,
                rot=wp.quat_identity(),
                vel=wp.vec3(0.0, 0.0, 0.0),
                dim_x=dx,
                dim_y=dy,
                dim_z=dz,
                cell_x=cell,
                cell_y=cell,
                cell_z=cell,
                density=density,
                k_mu=stiffness,
                k_lambda=stiffness,
                k_damp=1e-3,
                particle_radius=0.025,
            )

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.add_argument(
            "--solver",
            "-s",
            help="'coupled' for rigid+VBD coupling, 'vbd' for pure-VBD baseline",
            type=str,
            choices=["coupled", "vbd"],
            default="coupled",
        )
        _add_rigid_solver_arg(parser)
        parser.add_argument(
            "--proxy-mass-relaxation",
            "-pmr",
            help="Scale factor for proxy body mass in VBD (< 1 = softer coupling)",
            type=float,
            default=1.0,
        )
        parser.add_argument(
            "--coupling-mode",
            help="'lagged' (default) or 'staggered' (direct end-of-step sync)",
            type=str,
            choices=["lagged", "staggered"],
            default="lagged",
        )
        parser.add_argument(
            "--proxy-iterations",
            help="Number of proxy relaxation passes per substep",
            type=int,
            default=1,
        )
        parser.add_argument(
            "--rigid-contact-hard",
            help="Use VBD hard augmented-Lagrangian rigid contacts instead of soft penalty contacts",
            action="store_true",
        )
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args)
    newton.examples.run(example, args)
