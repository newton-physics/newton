# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example ADMM Rigid Contact Coupled Solver
#
# A dynamic rigid box rests on a kinematic rigid plane whose angle progressively
# increases. The two bodies are owned by separate solvers and interact through
# collision-detected rigid-rigid Coulomb contacts in SolverAdmmCoupled.
#
# Pass ``--solver free`` to disable the ADMM contacts and compare against the
# uncoupled baseline.
#
# Command: python -m newton.examples admm_rigid_contact_solver
#          python -m newton.examples admm_rigid_contact_solver --solver free
#
###########################################################################

from __future__ import annotations

import math

import numpy as np
import warp as wp
from newton.solvers.coupled_experimental import CouplingInterface, SolverAdmmCoupled, SolverCoupled

import newton
import newton.examples
from newton.solvers import SolverBase, SolverSemiImplicit


@wp.kernel(enable_backward=False)
def _set_kinematic_plane_kernel(
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    body_id: int,
    angle: wp.array[float],
):
    theta = angle[0]
    body_q[body_id] = wp.transform(
        wp.vec3(0.0, 0.0, 0.0),
        wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), theta),
    )
    body_qd[body_id] = wp.spatial_vector(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


class _KinematicPlaneSolver(SolverBase, CouplingInterface):
    """Solver that prescribes a fixed tilted plane pose."""

    def __init__(self, model, body: int, angle: wp.array[float]):
        super().__init__(model)
        self.body = int(body)
        self.angle = angle

    def step(self, state_in, state_out, control, contacts, dt):
        del control, contacts, dt
        if state_in.body_q is not None and state_out.body_q is not None:
            wp.copy(state_out.body_q, state_in.body_q)
            wp.copy(state_out.body_qd, state_in.body_qd)
            wp.launch(
                _set_kinematic_plane_kernel,
                dim=1,
                inputs=[state_out.body_q, state_out.body_qd, self.body, self.angle],
                device=self.model.device,
            )


class Example:
    def __init__(self, viewer, args):
        self.viewer = viewer
        self.sim_time = 0.0
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = 4
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.solver_type = args.solver
        self.start_angle = math.radians(args.start_angle)
        self.end_angle = math.radians(args.end_angle)
        self.current_angle = self.start_angle
        self.tilt_time = args.tilt_time
        self.critical_angle = math.atan(args.friction)
        self.box_half_extents = np.array([args.box_hx, args.box_hy, args.box_hz], dtype=np.float32)
        self.box_start_x = args.start_x
        self.penetration = args.initial_penetration

        builder = newton.ModelBuilder(gravity=-9.81)
        self.plane_body, self.box_body = self._emit_scene(builder, args)
        self.model = builder.finalize()
        self.angle_buffer = wp.array([self.current_angle], dtype=float, device=self.model.device)

        self.solver = SolverAdmmCoupled(
            model=self.model,
            entries=[
                SolverCoupled.Entry(
                    name="plane",
                    solver=lambda v: _KinematicPlaneSolver(model=v, body=self.plane_body, angle=self.angle_buffer),
                    bodies=[self.plane_body],
                ),
                SolverCoupled.Entry(
                    name="box",
                    solver=lambda v: SolverSemiImplicit(model=v, enable_tri_contact=False),
                    bodies=[self.box_body],
                ),
            ],
            coupling=SolverAdmmCoupled.Config(
                iterations=args.admm_iterations,
                rho=args.rho,
                gamma=args.gamma,
                baumgarte=args.baumgarte,
                contact_pairs=(
                    [SolverAdmmCoupled.ContactPair(source="plane", destination="box")]
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
        self.contacts = self.model.contacts()

        self.normal = self._rotate_y(np.array([0.0, 0.0, 1.0], dtype=np.float32), self.current_angle)
        self.tangent = self._rotate_y(np.array([1.0, 0.0, 0.0], dtype=np.float32), self.current_angle)
        self.initial_box_pos = self._body_position(self.state_0.body_q.numpy(), self.box_body)
        self.min_gap = self._min_contact_gap()
        self.max_precritical_slip = 0.0
        self.final_plane_local_x = self.box_start_x

        self.viewer.set_model(self.model)
        camera_target = wp.vec3(0.05, 0.0, 0.08)
        self.viewer.set_camera(pos=wp.vec3(1.05, -1.2, 0.72), pitch=-24.0, yaw=140.0)
        if hasattr(self.viewer, "camera") and hasattr(self.viewer.camera, "look_at"):
            self.viewer.camera.look_at(camera_target)

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
            self._sync_kinematic_plane(self.state_0)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        self._update_tilt_angle()
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()
        self.min_gap = min(self.min_gap, self._min_contact_gap())
        self.final_plane_local_x = self._box_plane_local_x()
        if self.current_angle < self.critical_angle:
            self.max_precritical_slip = max(
                self.max_precritical_slip,
                abs(self.final_plane_local_x - self.box_start_x),
            )
        self.sim_time += self.frame_dt

    def test_final(self):
        body_q = self.state_0.body_q.numpy()
        body_qd = self.state_0.body_qd.numpy()
        assert np.isfinite(body_q).all(), "Body positions contain NaN or inf values"
        assert np.isfinite(body_qd).all(), "Body velocities contain NaN or inf values"

        if self.solver_type == "admm":
            sliding_distance = self.final_plane_local_x - self.box_start_x
            assert self.max_precritical_slip < 0.05, (
                f"Box slipped before the critical angle: slip={self.max_precritical_slip:.4f}, "
                f"critical_angle={math.degrees(self.critical_angle):.2f} deg"
            )
            assert sliding_distance > 0.03, f"Box did not slide downhill: displacement={sliding_distance:.4f}"
            assert self.min_gap > -0.04, f"Rigid-rigid ADMM contact penetrated too deeply: gap={self.min_gap:.4f}"

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()

    def _emit_scene(self, builder: newton.ModelBuilder, args) -> tuple[int, int]:
        plane_q = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), self.current_angle)
        plane_body = builder.add_body(
            xform=wp.transform(p=wp.vec3(0.0, 0.0, 0.0), q=plane_q),
            mass=0.0,
            inertia=wp.mat33(),
            is_kinematic=True,
            label="admm_rigid_contact_plane",
        )

        visible_cfg = newton.ModelBuilder.ShapeConfig()
        visible_cfg.density = 0.0
        visible_cfg.has_shape_collision = True
        visible_cfg.has_particle_collision = False
        visible_cfg.mu = float(args.friction)
        builder.add_shape_box(
            plane_body,
            xform=wp.transform(p=wp.vec3(1.0, 0.0, -0.025), q=wp.quat_identity()),
            hx=3.0,
            hy=0.42,
            hz=0.025,
            cfg=visible_cfg,
            color=(0.42, 0.45, 0.50),
        )

        local_center = np.array(
            [self.box_start_x, 0.0, args.box_hz - self.penetration],
            dtype=np.float32,
        )
        box_pos = self._rotate_y(local_center, self.current_angle)
        box_body = builder.add_body(
            xform=wp.transform(
                p=wp.vec3(float(box_pos[0]), float(box_pos[1]), float(box_pos[2])),
                q=plane_q,
            ),
            mass=args.box_mass,
            inertia=wp.mat33(np.diag([args.box_inertia, args.box_inertia, args.box_inertia])),
            label="admm_rigid_contact_box",
        )
        builder.add_shape_box(
            box_body,
            hx=args.box_hx,
            hy=args.box_hy,
            hz=args.box_hz,
            cfg=visible_cfg,
            color=(0.15, 0.42, 0.88),
        )
        builder.color()
        return plane_body, box_body

    def _update_tilt_angle(self):
        if self.tilt_time <= 0.0:
            alpha = 1.0
        else:
            alpha = min(max(self.sim_time / self.tilt_time, 0.0), 1.0)
        alpha = alpha * alpha * (3.0 - 2.0 * alpha)
        self.current_angle = self.start_angle + alpha * (self.end_angle - self.start_angle)
        self.normal = self._rotate_y(np.array([0.0, 0.0, 1.0], dtype=np.float32), self.current_angle)
        self.tangent = self._rotate_y(np.array([1.0, 0.0, 0.0], dtype=np.float32), self.current_angle)
        self.angle_buffer.fill_(self.current_angle)

    def _sync_kinematic_plane(self, state):
        wp.launch(
            _set_kinematic_plane_kernel,
            dim=1,
            inputs=[state.body_q, state.body_qd, self.plane_body, self.angle_buffer],
            device=self.model.device,
        )

    def _min_contact_gap(self) -> float:
        body_q = self.state_0.body_q.numpy()
        point_b = np.array([self.box_start_x, 0.0, 0.0], dtype=np.float32)
        world_b = self._transform_point(body_q[self.plane_body], point_b)
        center_gap = float(np.dot(self.normal, self._body_position(body_q, self.box_body) - world_b))
        return center_gap - float(self.box_half_extents[2])

    def _box_plane_local_x(self) -> float:
        box_pos = self._body_position(self.state_0.body_q.numpy(), self.box_body)
        return float(self._rotate_y(box_pos, -self.current_angle)[0])

    @staticmethod
    def _body_position(body_q: np.ndarray, body: int) -> np.ndarray:
        return np.asarray(body_q[body, :3], dtype=np.float32)

    @staticmethod
    def _rotate_y(v: np.ndarray, angle: float) -> np.ndarray:
        c = math.cos(angle)
        s = math.sin(angle)
        return np.array([c * v[0] + s * v[2], v[1], -s * v[0] + c * v[2]], dtype=np.float32)

    @staticmethod
    def _quat_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
        qv = np.asarray(q[3:6], dtype=np.float32)
        qw = float(q[6])
        return v + 2.0 * np.cross(qv, np.cross(qv, v) + qw * v)

    def _transform_point(self, transform: np.ndarray, point: np.ndarray) -> np.ndarray:
        return np.asarray(transform[:3], dtype=np.float32) + self._quat_rotate(transform, point)

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.add_argument(
            "--solver",
            "-s",
            help="'admm' for collision-detected rigid-rigid ADMM contacts, or 'free' for the uncoupled baseline",
            type=str,
            choices=["admm", "free"],
            default="admm",
        )
        parser.add_argument("--start-angle", help="Initial plane inclination angle [deg]", type=float, default=2.0)
        parser.add_argument("--end-angle", help="Final plane inclination angle [deg]", type=float, default=34.0)
        parser.add_argument(
            "--tilt-time", help="Time taken to ramp from start to final angle [s]", type=float, default=2.0
        )
        parser.add_argument("--friction", help="Rigid-rigid Coulomb friction coefficient", type=float, default=0.45)
        parser.add_argument("--admm-iterations", help="ADMM iterations per substep", type=int, default=30)
        parser.add_argument("--rho", help="ADMM penalty parameter", type=float, default=5.0)
        parser.add_argument("--gamma", help="ADMM proximal mass rescaling", type=float, default=0.2)
        parser.add_argument(
            "--baumgarte", help="Fraction of contact penetration corrected per substep", type=float, default=0.03
        )
        parser.add_argument("--box-mass", help="Sliding box mass [kg]", type=float, default=1.0)
        parser.add_argument("--box-inertia", help="Diagonal box inertia [kg m^2]", type=float, default=0.01)
        parser.add_argument("--box-hx", help="Box half-width along plane tangent [m]", type=float, default=0.08)
        parser.add_argument("--box-hy", help="Box half-width across plane [m]", type=float, default=0.08)
        parser.add_argument("--box-hz", help="Box half-height normal to plane [m]", type=float, default=0.08)
        parser.add_argument(
            "--start-x", help="Initial box center coordinate in plane-local x [m]", type=float, default=-0.25
        )
        parser.add_argument(
            "--initial-penetration",
            help="Initial normal overlap used to activate the detected ADMM contacts [m]",
            type=float,
            default=0.004,
        )
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args)
    newton.examples.run(example, args)
