# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Rigid-XPBD Coupled Solver
#
# Rigid boxes and an articulated pendulum chain (driven by MuJoCo or Kamino)
# interact with a cloth sheet (simulated by XPBD). Contact forces from the
# deformable side are fed back to the rigid bodies through SolverProxyCoupled's
# proxy-body coupling path with momentum-based force harvesting.
#
# Pass ``--solver xpbd`` to run the same scene with a single XPBD solver
# (no coupling) as a reference baseline.
#
# Command: python -m newton.examples mujoco_xpbd_coupled_solver
#          python -m newton.examples mujoco_xpbd_coupled_solver --solver xpbd
#
###########################################################################

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import warp as wp
from newton.solvers.coupled_experimental import SolverProxyCoupled

import newton
import newton.examples
from newton.solvers import SolverKamino, SolverMuJoCo, SolverXPBD


def _add_rigid_solver_arg(parser) -> None:
    parser.add_argument(
        "--rigid-solver",
        help="Rigid-body solver used by the coupled path.",
        type=str,
        choices=["mujoco", "kamino"],
        default="mujoco",
    )


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
        return "kamino", SolverKamino, {"config": _make_kamino_config()}, None
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
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = 16
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

        self.model = builder.finalize()

        # Contact parameters
        # self.model.soft_contact_ke = 1.0e6
        # self.model.soft_contact_kd = 1.03
        # self.model.soft_contact_mu = 0.5

        xpbd_kwargs = {
            "iterations": 40,
        }

        if self.use_coupled:
            # ---------- Coupled path: rigid solver + XPBD ----------
            rigid_name, rigid_solver, rigid_kwargs, rigid_configure_view = _rigid_solver_entry_args(
                self.rigid_solver,
                mujoco_kwargs={"use_mujoco_contacts": False, "njmax": 200, "nconmax": 200},
            )
            rigid_body_indices = wp.array(list(range(rigid_body_start, rigid_body_end)), dtype=int)
            xpbd_body_indices = wp.array(
                [i for i in range(self.model.body_count) if i < rigid_body_start or i >= rigid_body_end],
                dtype=int,
            )

            self.solver = SolverProxyCoupled(
                model=self.model,
                entries=[
                    SolverProxyCoupled.Entry(
                        name=rigid_name,
                        solver=lambda v: rigid_solver(model=v, **rigid_kwargs),
                        bodies=[int(i) for i in rigid_body_indices.numpy()],
                        joints=list(range(self.model.joint_count)),
                        configure_view=rigid_configure_view,
                    ),
                    SolverProxyCoupled.Entry(
                        name="xpbd",
                        solver=lambda v: SolverXPBD(model=v, **xpbd_kwargs),
                        bodies=[int(i) for i in xpbd_body_indices.numpy()],
                        particles=list(range(self.model.particle_count)),
                    ),
                ],
                coupling=SolverProxyCoupled.Config(
                    proxies=[
                        SolverProxyCoupled.Proxy(
                            source=rigid_name,
                            destination="xpbd",
                            bodies=[int(i) for i in rigid_body_indices.numpy()],
                            mass_scale=args.mass_scale,
                        )
                    ],
                    iterations=args.proxy_iterations,
                ),
            )
        else:
            # ---------- Pure-XPBD path (reference baseline) ----------
            self.solver = newton.solvers.SolverXPBD(model=self.model, **xpbd_kwargs)

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
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            self.model.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        if not _launch_frame_graph(self.model, self.graph):
            self.simulate()
        self.sim_time += self.frame_dt

    def test_final(self):
        newton.examples.test_body_state(
            self.model,
            self.state_0,
            "all rigid bodies are above the ground",
            lambda q, qd: q[2] > -0.1,
        )
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
        saved_density = builder.default_shape_cfg.density
        builder.default_shape_cfg.density = 100.0  # light boxes
        boxes = [
            (wp.vec3(0.0, 0.0, 2.0), (0.15, 0.15, 0.15)),
            (wp.vec3(0.3, 0.1, 2.5), (0.10, 0.20, 0.10)),
            (wp.vec3(-0.2, -0.1, 3.0), (0.12, 0.12, 0.12)),
        ]
        for pos, (hx, hy, hz) in boxes:
            body = builder.add_body(
                xform=wp.transform(p=pos, q=wp.quat_identity()),
            )
            builder.add_shape_box(body, hx=hx, hy=hy, hz=hz)
        builder.default_shape_cfg.density = saved_density

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
            add_springs=True,
            spring_ke=1.0e4,
            spring_kd=1.0,
            tri_ke=1.0e4,
            tri_ka=1.0e4,
            tri_kd=0.0,
            edge_ke=1.0e2,
            edge_kd=0.0,
            particle_radius=0.01,
        )

    def _emit_articulated_chain(self, builder: newton.ModelBuilder):
        """Add a free-falling 8-link chain."""
        n_links = 8
        hx, hy, hz = 0.06, 0.03, 0.03
        damping = 0.5
        start = wp.vec3(-0.3, 0.0, 3.5)

        saved_density = builder.default_shape_cfg.density
        builder.default_shape_cfg.density = 100.0

        prev_link = builder.add_link(
            xform=wp.transform(p=start, q=wp.quat_identity()),
        )
        builder.add_shape_box(prev_link, hx=hx, hy=hy, hz=hz)
        joints = [builder.add_joint_free(child=prev_link)]

        for _ in range(n_links - 1):
            link = builder.add_link()
            builder.add_shape_box(link, hx=hx, hy=hy, hz=hz)
            joints.append(
                builder.add_joint_revolute(
                    parent=prev_link,
                    child=link,
                    axis=wp.vec3(0.0, 1.0, 0.0),
                    target_kd=damping,
                    parent_xform=wp.transform(p=wp.vec3(hx, 0.0, 0.0), q=wp.quat_identity()),
                    child_xform=wp.transform(p=wp.vec3(-hx, 0.0, 0.0), q=wp.quat_identity()),
                )
            )
            prev_link = link

        builder.add_articulation(joints, label="chain")
        builder.default_shape_cfg.density = saved_density

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.add_argument(
            "--solver",
            "-s",
            help="'coupled' for rigid+XPBD coupling, 'xpbd' for pure-XPBD baseline",
            type=str,
            choices=["coupled", "xpbd"],
            default="coupled",
        )
        _add_rigid_solver_arg(parser)
        parser.add_argument(
            "--mass-scale",
            "-pmr",
            help="Scale factor for proxy body mass in XPBD (< 1 = softer coupling)",
            type=float,
            default=1.0,
        )
        parser.add_argument(
            "--proxy-iterations",
            help="Number of proxy relaxation passes per substep",
            type=int,
            default=1,
        )
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args)
    newton.examples.run(example, args)
