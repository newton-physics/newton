# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Rigid-VBD ADMM Coupled Solver
#
# A rigid ball is attached to the centre of a pinned VBD cloth sheet
# through a model-level body-particle attachment annotation, while a separate
# rigid pendulum link carries a VBD rigid payload through a normal model ball
# joint. SolverCoupledADMM converts both cross-solver couplings into ADMM
# attachment constraints.
#
# This example demonstrates ``SolverCoupledADMM``, an
# alternative to proxy-body coupling based on linearised ADMM over model
# joints, model attachment annotations, and collision-detected contacts rather
# than proxy bodies. See ``docs/plans/2026-04-23-admm-coupling.tex`` for the
# algorithm.
#
# Command: python -m newton.examples mujoco_vbd_admm_solver
#
###########################################################################

from __future__ import annotations

import argparse
from collections.abc import Callable

import numpy as np
import warp as wp
from newton.solvers.experimental.coupled import SolverCoupled, SolverCoupledADMM

import newton
import newton.examples
from newton.solvers import SolverKamino, SolverMuJoCo, SolverVBD


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
        return "kamino", SolverKamino, {"config": _make_kamino_config()}
    if rigid_solver == "mujoco":
        return "mjc", SolverMuJoCo, dict(mujoco_kwargs or {})
    raise ValueError(f"Unsupported rigid solver '{rigid_solver}'")


def _capture_frame_graph(model: newton.Model, simulate: Callable[[], None], *, enabled: bool = True):
    if not enabled:
        return None

    with wp.ScopedDevice(model.device):
        with wp.ScopedCapture() as capture:
            simulate()

    if capture.graph is None:
        raise RuntimeError(f"Graph capture failed on device {model.device}")
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
        self.sim_substeps = 8
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.rigid_solver = getattr(args, "rigid_solver", "mujoco")

        builder = newton.ModelBuilder()
        _register_rigid_solver_custom_attributes(builder, self.rigid_solver)
        builder.add_ground_plane()

        dim = 11
        cloth_z = 2.0
        particle_start = builder.particle_count
        cloth_tri_ke = 1.0e3
        cloth_edge_ke = 0.01
        builder.add_cloth_grid(
            pos=wp.vec3(-0.5, -0.5, cloth_z),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0),
            fix_left=True,
            fix_right=True,
            dim_x=dim,
            dim_y=dim,
            cell_x=0.1,
            cell_y=0.1,
            mass=0.05,
            tri_ke=cloth_tri_ke,
            tri_ka=cloth_tri_ke,
            tri_kd=1.0e-2 * cloth_tri_ke,
            edge_ke=cloth_edge_ke,
            edge_kd=1.0e-2 * cloth_edge_ke,
            particle_radius=0.01,
        )

        center = dim // 2
        self.center_particle = particle_start + center * (dim + 1) + center

        ball_radius = 0.08
        self.ball_body = builder.add_body(
            xform=wp.transform(p=wp.vec3(0.0, 0.0, cloth_z - ball_radius), q=wp.quat_identity()),
            mass=0.5,
            inertia=wp.mat33(np.eye(3) * 5.0e-3),
        )
        self.ball_joint = builder.joint_count - 1
        builder.add_shape_sphere(self.ball_body, radius=ball_radius)
        SolverCoupledADMM.add_body_particle_attachment(
            builder,
            self.ball_body,
            self.center_particle,
            body_point=wp.vec3(0.0, 0.0, ball_radius),
            stiffness=1.0e3,
        )

        link_hx = 0.28
        payload_hx = 0.12
        anchor = wp.vec3(1.4, 0.0, 2.2)

        self.pendulum_body = builder.add_link(
            xform=wp.transform(p=anchor + wp.vec3(link_hx, 0.0, 0.0), q=wp.quat_identity()),
            mass=0.6,
            inertia=wp.mat33(np.eye(3) * 1.0e-2),
        )
        builder.add_shape_box(self.pendulum_body, hx=link_hx, hy=0.045, hz=0.045)
        self.pendulum_joint = builder.add_joint_revolute(
            parent=-1,
            child=self.pendulum_body,
            axis=wp.vec3(0.0, 1.0, 0.0),
            target_kd=0.5,
            parent_xform=wp.transform(p=anchor, q=wp.quat_identity()),
            child_xform=wp.transform(p=wp.vec3(-link_hx, 0.0, 0.0), q=wp.quat_identity()),
        )
        builder.add_articulation([self.pendulum_joint], label="pendulum")

        self.payload_body = builder.add_body(
            xform=wp.transform(p=anchor + wp.vec3(2.0 * link_hx + payload_hx, 0.0, 0.0), q=wp.quat_identity()),
            mass=0.35,
            inertia=wp.mat33(np.eye(3) * 6.0e-3),
        )
        self.payload_free_joint = builder.joint_count - 1
        builder.add_shape_box(self.payload_body, hx=payload_hx, hy=0.09, hz=0.09)
        builder.add_joint_ball(
            parent=self.pendulum_body,
            child=self.payload_body,
            friction=1.0,
            parent_xform=wp.transform(p=wp.vec3(link_hx, 0.0, 0.0), q=wp.quat_identity()),
            child_xform=wp.transform(p=wp.vec3(-payload_hx, 0.0, 0.0), q=wp.quat_identity()),
            collision_filter_parent=True,
        )

        builder.color()
        self.model = builder.finalize()
        # The ADMM attachment binds the ball to the cloth; ordinary soft contact
        # on the same pair would push against the attachment.
        self.model.soft_contact_ke = 0.0
        self.model.soft_contact_kd = 0.0

        rigid_name, rigid_solver, rigid_kwargs = _rigid_solver_entry_args(
            self.rigid_solver,
            mujoco_kwargs={"use_mujoco_contacts": False, "njmax": 32},
        )
        self.solver = SolverCoupledADMM(
            model=self.model,
            entries=[
                SolverCoupled.Entry(
                    name=rigid_name,
                    solver=lambda v: rigid_solver(model=v, **rigid_kwargs),
                    bodies=[self.ball_body, self.pendulum_body],
                    joints=[self.ball_joint, self.pendulum_joint],
                ),
                SolverCoupled.Entry(
                    name="vbd",
                    solver=lambda v: SolverVBD(model=v, iterations=8),
                    bodies=[self.payload_body],
                    joints=[self.payload_free_joint],
                    particles=list(range(self.model.particle_count)),
                ),
            ],
            coupling=SolverCoupledADMM.Config(
                iterations=2,
                rho=50,
                gamma=0.1,
                baumgarte=0.01,
                joint_proximal_bodies=args.joint_proximal_bodies,
                joint_proximal_destination_entries=(rigid_name,),
            ),
        )

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.contacts = self.model.contacts()
        self.control = self.model.control()

        newton.examples.configure_coupled_view(self, args)
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)

        self.capture()

    def capture(self):
        """Graph-capture the per-frame simulate loop on CUDA devices.

        The ADMM loop has no Python-side branching on runtime data — the
        attachment count and all conditionals are construction-time
        constants — so the launch sequence is identical every frame.
        """
        self.graph = _capture_frame_graph(self.model, self.simulate)

    def simulate(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            newton.examples.apply_coupled_viewer_forces(self, self.state_0)
            # ADMM builds this example's coupling from joints and
            # body-particle attachments, so keep state_0/contacts empty here
            # rather than asking collide() to add redundant constraints.
            # self.model.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            newton.eval_ik(self.model, self.state_1, self.state_1.joint_q, self.state_1.joint_qd)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        if not _launch_frame_graph(self.model, self.graph):
            self.simulate()
        self.sim_time += self.frame_dt

    def test_final(self):
        body_q = self.state_0.body_q.numpy()
        particle_q = self.state_0.particle_q.numpy()
        z = body_q[self.ball_body, 2]
        cloth_gap = np.linalg.norm(body_q[self.ball_body, :3] - particle_q[self.center_particle])
        pendulum_gap = np.linalg.norm(body_q[self.payload_body, :3] - body_q[self.pendulum_body, :3])
        assert np.all(np.isfinite(body_q))
        assert np.all(np.isfinite(particle_q))
        assert 1.0 < z < 2.0, f"rigid ball z={z:.3f}; expected a hanging motion below the cloth plane"
        assert cloth_gap < 0.5, f"body-particle attachment drifted too far: gap={cloth_gap:.3f}"
        assert pendulum_gap < 1.0, f"cross-solver pendulum joint drifted too far: gap={pendulum_gap:.3f}"

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        newton.examples.log_coupled_view(self, self.contacts)
        self.viewer.end_frame()

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        newton.examples.add_coupled_view_args(parser)
        _add_rigid_solver_arg(parser)
        parser.add_argument(
            "--joint-proximal-bodies",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="Keep cross-solver joint neighbor bodies as inertial ADMM proxies.",
        )
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args)
    newton.examples.run(example, args)
