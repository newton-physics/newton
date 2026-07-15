# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Kamino-MuJoCo ADMM Coupled Solver
#
# A closed four-bar linkage is split across two rigid solvers: two links are
# owned by Kamino and two links are owned by MuJoCo. SolverCoupledADMM detects
# the model revolute joints that connect bodies owned by different solvers and
# turns them into ADMM attachments.
#
# Command: python -m newton.examples kamino_mujoco_admm_solver
#
###########################################################################

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import warp as wp
from newton.solvers.experimental.coupled import ModelView, SolverCoupled, SolverCoupledADMM

import newton
import newton.examples
from newton.solvers import SolverKamino, SolverMuJoCo


def _configure_kamino_rigid_view(view: ModelView) -> None:
    # Kamino normalizes world ids and re-expresses body-local frames during
    # conversion. Keep those writes local to this mixed Kamino/MuJoCo view.
    for name in (
        "body_world",
        "body_world_start",
        "joint_world",
        "joint_world_start",
        "shape_world",
        "shape_world_start",
        "body_com",
        "body_inertia",
        "shape_transform",
        "joint_X_p",
        "joint_X_c",
    ):
        setattr(view, name, wp.clone(getattr(view, name)))


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


def _quat_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    qv = q[:3]
    qw = q[3]
    return v + 2.0 * np.cross(qv, np.cross(qv, v) + qw * v)


def _transform_point(body_q: np.ndarray, body: int, point: np.ndarray) -> np.ndarray:
    return body_q[body, :3] + _quat_rotate(body_q[body, 3:7], point)


def _quat_from_x_axis(direction: np.ndarray) -> wp.quat:
    direction = np.asarray(direction, dtype=np.float32)
    direction = direction / np.linalg.norm(direction)
    source = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    axis = np.cross(source, direction)
    w = 1.0 + float(np.dot(source, direction))
    if w < 1.0e-6:
        axis = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        w = 0.0
    quat = np.array([axis[0], axis[1], axis[2], w], dtype=np.float32)
    quat /= np.linalg.norm(quat)
    return wp.quat(float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3]))


class Example:
    def __init__(self, viewer, args):
        self.viewer = viewer
        self.sim_time = 0.0
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = max(1, int(args.substeps))
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.use_graph = bool(args.graph_capture)
        self.world_count = max(1, int(args.world_count))

        template = newton.ModelBuilder(gravity=-9.81)
        SolverKamino.register_custom_attributes(template)
        self._joint_checks: list[tuple[int, np.ndarray, int, np.ndarray]] = []
        self._emit_four_bar(template)

        bodies_per_world = template.body_count
        joints_per_world = template.joint_count

        builder = newton.ModelBuilder(gravity=-9.81)
        builder.replicate(template, world_count=self.world_count)
        builder.add_ground_plane()
        self._expand_world_indices(bodies_per_world, joints_per_world)

        builder.color()
        self.model = builder.finalize()
        self.model.rigid_contact_max = 96

        self.device = self.model.device

        kamino_config = _make_kamino_config()
        kamino_config.padmm.max_iterations = args.kamino_iterations

        self.solver = SolverCoupledADMM(
            model=self.model,
            entries=[
                SolverCoupled.Entry(
                    name="kamino",
                    solver=lambda v: SolverKamino(model=v, config=kamino_config),
                    bodies=self.kamino_bodies,
                    joints=self.kamino_joints,
                    configure_view=_configure_kamino_rigid_view,
                ),
                SolverCoupled.Entry(
                    name="mjc",
                    solver=lambda v: SolverMuJoCo(
                        model=v,
                        **{"use_mujoco_contacts": False, "njmax": 64, "nconmax": 64},
                    ),
                    bodies=self.mujoco_bodies,
                    joints=self.mujoco_joints,
                ),
            ],
            coupling=SolverCoupledADMM.Config(
                iterations=args.admm_iterations,
                rho=args.rho,
                gamma=args.gamma,
                baumgarte=args.baumgarte,
                joint_stiffness=args.joint_stiffness,
                joint_angular_stiffness=args.joint_stiffness,
                joint_damping=args.joint_damping,
                joint_angular_damping=args.joint_damping,
            ),
        )

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.contacts = self.model.contacts()
        self.control = self.model.control()

        newton.examples.configure_coupled_view(self, args)
        self.viewer.set_world_offsets((1.35, 1.35, 0.0))
        if isinstance(self.viewer, newton.viewer.ViewerGL):
            scale = max(1.0, float(np.sqrt(self.world_count)))
            self.viewer.set_camera(pos=wp.vec3(0.55 * scale, -2.4 * scale, 1.45 * scale), pitch=-14.0, yaw=105.0)
            if hasattr(self.viewer.camera, "look_at"):
                self.viewer.camera.look_at(wp.vec3(0.05, 0.0, 1.25))

        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_1)

        self.capture()

    def _emit_four_bar(self, builder: newton.ModelBuilder) -> None:
        left_base = np.array([-0.48, 0.0, 1.05], dtype=np.float32)
        right_base = np.array([0.48, 0.0, 1.05], dtype=np.float32)
        left_top = np.array([-0.30, 0.0, 1.55], dtype=np.float32)
        right_top = np.array([0.34, 0.0, 1.46], dtype=np.float32)
        ground_center = 0.5 * (left_base + right_base)
        ground_quat = _quat_from_x_axis(right_base - left_base)

        self.mujoco_ground, ground_left, ground_right = self._add_bar(
            builder,
            left_base,
            right_base,
            label="mujoco_ground_link",
            color=(0.45, 0.48, 0.54),
            mass=0.0,
        )
        self.mujoco_ground_joint = builder.add_joint_fixed(
            parent=-1,
            child=self.mujoco_ground,
            parent_xform=wp.transform(p=wp.vec3(*ground_center), q=ground_quat),
            label="mujoco_ground_fixed",
        )
        builder.add_articulation([self.mujoco_ground_joint], label="mujoco_ground")

        self.kamino_crank, crank_base, crank_top = self._add_bar(
            builder,
            left_base,
            left_top,
            label="kamino_crank",
            color=(0.16, 0.43, 0.86),
            mass=0.35,
        )
        self.kamino_crank_joint = builder.add_joint_free(child=self.kamino_crank, label="kamino_crank_free")
        builder.add_articulation([self.kamino_crank_joint], label="kamino_crank_articulation")

        self.mujoco_coupler, coupler_left, coupler_right = self._add_bar(
            builder,
            left_top,
            right_top,
            label="mujoco_coupler",
            color=(0.91, 0.47, 0.20),
            mass=0.42,
        )
        self.mujoco_coupler_joint = builder.add_joint_free(child=self.mujoco_coupler, label="mujoco_coupler_free")
        builder.add_articulation([self.mujoco_coupler_joint], label="mujoco_coupler_articulation")

        self.kamino_rocker, rocker_base, rocker_top = self._add_bar(
            builder,
            right_base,
            right_top,
            label="kamino_rocker",
            color=(0.08, 0.58, 0.72),
            mass=0.36,
        )
        self.kamino_rocker_joint = builder.add_joint_free(child=self.kamino_rocker, label="kamino_rocker_free")
        builder.add_articulation([self.kamino_rocker_joint], label="kamino_rocker_articulation")

        self._add_cross_joint(builder, self.mujoco_ground, ground_left, self.kamino_crank, crank_base, "left_base")
        self._add_cross_joint(builder, self.kamino_crank, crank_top, self.mujoco_coupler, coupler_left, "left_top")
        self._add_cross_joint(builder, self.mujoco_coupler, coupler_right, self.kamino_rocker, rocker_top, "right_top")
        self._add_cross_joint(builder, self.kamino_rocker, rocker_base, self.mujoco_ground, ground_right, "right_base")

        self.kamino_bodies = [self.kamino_crank, self.kamino_rocker]
        self.kamino_joints = [self.kamino_crank_joint, self.kamino_rocker_joint]
        self.mujoco_bodies = [self.mujoco_ground, self.mujoco_coupler]
        self.mujoco_joints = [self.mujoco_ground_joint, self.mujoco_coupler_joint]

    def _expand_world_indices(self, bodies_per_world: int, joints_per_world: int) -> None:
        """Expand one-world body/joint ids to all replicated worlds."""

        def expand(ids: list[int], stride: int) -> list[int]:
            return [world * stride + id_ for world in range(self.world_count) for id_ in ids]

        self.kamino_bodies = expand(self.kamino_bodies, bodies_per_world)
        self.kamino_joints = expand(self.kamino_joints, joints_per_world)
        self.mujoco_bodies = expand(self.mujoco_bodies, bodies_per_world)
        self.mujoco_joints = expand(self.mujoco_joints, joints_per_world)
        self._joint_checks = [
            (world * bodies_per_world + parent, parent_point, world * bodies_per_world + child, child_point)
            for world in range(self.world_count)
            for parent, parent_point, child, child_point in self._joint_checks
        ]

    def _add_bar(
        self,
        builder: newton.ModelBuilder,
        point_a: np.ndarray,
        point_b: np.ndarray,
        *,
        label: str,
        color: tuple[float, float, float],
        mass: float,
    ) -> tuple[int, np.ndarray, np.ndarray]:
        axis = np.asarray(point_b - point_a, dtype=np.float32)
        length = float(np.linalg.norm(axis))
        half_length = 0.5 * length
        center = 0.5 * (point_a + point_b)
        inertia_scale = max(mass, 1.0e-3) * length * length / 12.0
        inertia = wp.mat33(np.eye(3, dtype=np.float32) * inertia_scale)
        body = builder.add_link(
            xform=wp.transform(p=wp.vec3(*center), q=_quat_from_x_axis(axis)),
            mass=mass,
            inertia=inertia,
            label=label,
        )
        builder.add_shape_box(body, hx=half_length, hy=0.035, hz=0.035, color=color)
        return (
            body,
            np.array([-half_length, 0.0, 0.0], dtype=np.float32),
            np.array([half_length, 0.0, 0.0], dtype=np.float32),
        )

    def _add_cross_joint(
        self,
        builder: newton.ModelBuilder,
        parent: int,
        parent_point: np.ndarray,
        child: int,
        child_point: np.ndarray,
        label: str,
    ) -> None:
        builder.add_joint_revolute(
            parent=parent,
            child=child,
            parent_xform=wp.transform(p=wp.vec3(*parent_point), q=wp.quat_identity()),
            child_xform=wp.transform(p=wp.vec3(*child_point), q=wp.quat_identity()),
            axis=wp.vec3(0.0, 1.0, 0.0),
            friction=0.001,
            label=f"kamino_mujoco_four_bar_{label}",
            collision_filter_parent=True,
        )
        self._joint_checks.append((parent, parent_point, child, child_point))

    def capture(self):
        self.graph = _capture_frame_graph(self.model, self.simulate, enabled=self.use_graph)

    def simulate(self):
        need_state_copy = self.use_graph and self.sim_substeps % 2 == 1

        for i in range(self.sim_substeps):
            self.state_0.clear_forces()
            newton.examples.apply_coupled_viewer_forces(self, self.state_0)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            if need_state_copy and i == self.sim_substeps - 1:
                self.state_0.assign(self.state_1)
            else:
                self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        if not _launch_frame_graph(self.model, self.graph):
            self.simulate()
        self.sim_time += self.frame_dt

    def test_final(self):
        body_q = self.state_0.body_q.numpy()
        body_qd = self.state_0.body_qd.numpy()
        assert np.all(np.isfinite(body_q)), "Body positions contain NaN or inf values"
        assert np.all(np.isfinite(body_qd)), "Body velocities contain NaN or inf values"

        max_gap = 0.0
        for parent_body, parent_point, child_body, child_point in self._joint_checks:
            parent = _transform_point(body_q, parent_body, parent_point)
            child = _transform_point(body_q, child_body, child_point)
            max_gap = max(max_gap, float(np.linalg.norm(parent - child)))
        assert max_gap < 0.12, f"Kamino-MuJoCo four-bar ADMM joints drifted too far: gap={max_gap:.3f}"
        if self.use_graph:
            assert self.graph is not None, "Graph capture was requested but no graph was captured"

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        newton.examples.log_coupled_view(self, self.contacts)
        self.viewer.end_frame()

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        newton.examples.add_coupled_view_args(parser)
        newton.examples.add_world_count_arg(parser)
        parser.set_defaults(world_count=4)
        parser.add_argument("--substeps", type=int, default=3, help="Coupled substeps per rendered frame.")
        parser.add_argument("--admm-iterations", type=int, default=2, help="ADMM iterations per coupled substep.")
        parser.add_argument("--rho", type=float, default=50.0, help="ADMM penalty parameter.")
        parser.add_argument("--gamma", type=float, default=0.1, help="ADMM proximal mass scaling.")
        parser.add_argument("--baumgarte", type=float, default=0.02, help="Position error correction fraction.")
        parser.add_argument("--joint-stiffness", type=float, default=2.0e4, help="Cross-solver joint stiffness [N/m].")
        parser.add_argument("--joint-damping", type=float, default=1.0, help="Cross-solver joint damping [N*s/m].")
        parser.add_argument("--kamino-iterations", type=int, default=40, help="Kamino PADMM iterations per substep.")
        parser.add_argument(
            "--no-graph-capture",
            action="store_false",
            dest="graph_capture",
            default=True,
            help="Disable graph capture.",
        )
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args)
    newton.examples.run(example, args)
