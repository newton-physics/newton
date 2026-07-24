# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Proxy Joint Gripper
#
# A MuJoCo-driven palm and two prismatic cube fingers grip a VBD soft grid.
# The VBD entry receives the gripper bodies as body proxies and keeps the
# fixed/prismatic joints enabled as proxy joints whose targets track the
# MuJoCo source joint configuration.
#
# Command: python -m newton.examples proxy_joint_gripper
#
###########################################################################

from __future__ import annotations

import argparse

import numpy as np
import warp as wp
from newton.solvers.experimental.coupled import SolverCoupledProxy

import newton
import newton.examples
from newton.solvers import SolverMuJoCo, SolverVBD


@wp.kernel
def _set_finger_targets(
    joint_target_q: wp.array[float],
    left_target_index: int,
    right_target_index: int,
    target: float,
):
    joint_target_q[left_target_index] = target
    joint_target_q[right_target_index] = target


class Example:
    def __init__(self, viewer, args):
        self.viewer = viewer
        self.args = args
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = int(args.substeps)
        self.use_graph = bool(args.graph_capture)
        if self.use_graph and self.sim_substeps % 2 != 0:
            raise ValueError("Graph capture requires an even number of simulation substeps")
        self.sim_dt = self.frame_dt / float(self.sim_substeps)
        self.sim_time = 0.0
        self.scenario = args.scenario
        self.close_distance = (
            float(args.close_distance)
            if args.close_distance is not None
            else (0.052 if self.scenario == "harsh" else 0.04)
        )
        self.close_time = (
            float(args.close_time) if args.close_time is not None else (0.3 if self.scenario == "harsh" else 0.8)
        )

        if hasattr(self.viewer, "set_camera"):
            self.viewer.set_camera(pos=wp.vec3(0.32, -0.42, 0.34), pitch=-24.0, yaw=136.0)

        builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        SolverMuJoCo.register_custom_attributes(builder)
        SolverVBD.register_custom_attributes(builder, dahl_defaults_enabled=False)
        builder.default_particle_radius = 0.01

        self.soft_particle_start = builder.particle_count
        self._emit_soft_object(builder)
        self.soft_particle_end = builder.particle_count

        self.gripper_bodies, self.gripper_joints = self._emit_gripper(builder)
        self.fixed_joint, self.left_joint, self.right_joint = self.gripper_joints

        builder.color()
        self.model = builder.finalize()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.model)

        self.model.soft_contact_ke = 1.5e5 if self.scenario == "harsh" else 5.0e4
        self.model.soft_contact_kd = 1.0e-4
        self.model.soft_contact_kf = 1.0e3
        self.model.soft_contact_mu = 2.0

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.collision_pipeline = newton.CollisionPipeline(self.model)
        self.contacts = self.collision_pipeline.contacts()
        self.control = self.model.control()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_1)

        self.left_target_index = int(self.model.joint_target_q_start.numpy()[self.left_joint])
        self.right_target_index = int(self.model.joint_target_q_start.numpy()[self.right_joint])
        self.left_q_index = int(self.model.joint_q_start.numpy()[self.left_joint])
        self.right_q_index = int(self.model.joint_q_start.numpy()[self.right_joint])
        self.initial_particle_bounds = self._particle_bounds()

        vbd_kwargs = {
            "iterations": int(args.vbd_iterations),
            "particle_enable_self_contact": False,
            "particle_enable_tile_solve": False,
            "rigid_contact_hard": False,
            "rigid_body_particle_contact_buffer_size": 1024 if self.scenario == "harsh" else 512,
            "rigid_joint_linear_ke": 5.0e5 if self.scenario == "harsh" else 2.0e7,
            "rigid_joint_angular_ke": 5.0e5 if self.scenario == "harsh" else 2.0e6,
        }

        self.solver = SolverCoupledProxy(
            model=self.model,
            entries=[
                SolverCoupledProxy.Entry(
                    name="mjc",
                    solver=lambda v: SolverMuJoCo(
                        model=v,
                        iterations=int(args.mujoco_iterations),
                        disable_contacts=True,
                        use_mujoco_contacts=False,
                    ),
                    bodies=self.gripper_bodies,
                    joints=self.gripper_joints,
                ),
                SolverCoupledProxy.Entry(
                    name="vbd",
                    solver=lambda v: SolverVBD(model=v, **vbd_kwargs),
                    particles=list(range(self.soft_particle_start, self.soft_particle_end)),
                ),
            ],
            coupling=SolverCoupledProxy.Config(
                proxies=[
                    SolverCoupledProxy.Proxy(
                        source="mjc",
                        destination="vbd",
                        bodies=self.gripper_bodies,
                        joints=self.gripper_joints if args.proxy_joints else (),
                        mass_scale=float(args.mass_scale),
                        mode=args.coupling_mode,
                        proxy_relaxation=float(args.proxy_relaxation),
                        proxy_relaxation_mode=args.proxy_relaxation_mode,
                        proxy_relaxation_min=float(args.proxy_relaxation_min),
                        proxy_relaxation_max=float(args.proxy_relaxation_max),
                        collision_pipeline=lambda model: newton.examples.create_collision_pipeline(
                            model,
                            broad_phase="explicit",
                        ),
                        collide_interval=1,
                    )
                ],
                iterations=int(args.proxy_iterations),
            ),
        )
        newton.examples.configure_coupled_view(self, args)
        self.capture()

    def capture(self) -> None:
        self.graph = None
        if not self.use_graph or not self.model.device.is_cuda:
            return

        with wp.ScopedDevice(self.model.device), wp.ScopedCapture() as capture:
            self.simulate()
        self.graph = capture.graph
        if self.graph is None:
            raise RuntimeError(f"CUDA graph capture failed on device {self.model.device}")

    def _emit_soft_object(self, builder: newton.ModelBuilder) -> None:
        size = 0.1 if self.scenario == "harsh" else 0.09
        cell = size / 3.0
        y_offset = 0.03 if self.scenario == "harsh" else 0.0
        builder.add_soft_grid(
            pos=wp.vec3(0.0, y_offset - 0.5 * size, 0.12 - 0.5 * size),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0, 0.0, 0.0),
            dim_x=1,
            dim_y=3,
            dim_z=3,
            cell_x=0.045,
            cell_y=cell,
            cell_z=cell,
            density=450.0 if self.scenario == "harsh" else 180.0,
            k_mu=1.2e4 if self.scenario == "harsh" else 2.0e3,
            k_lambda=3.6e4 if self.scenario == "harsh" else 6.0e3,
            k_damp=0.25 if self.scenario == "harsh" else 0.1,
            particle_radius=0.014 if self.scenario == "harsh" else 0.012,
            label="soft_grip_block",
        )

    def _emit_gripper(self, builder: newton.ModelBuilder) -> tuple[list[int], list[int]]:
        contact_cfg = newton.ModelBuilder.ShapeConfig(
            density=800.0,
            ke=1.5e5 if self.scenario == "harsh" else 8.0e4,
            kd=1.0e-4,
            kf=1.0e3,
            mu=0.7,
        )
        palm_pos = wp.vec3(-0.07, 0.0, 0.12)
        finger_y = 0.105
        palm = builder.add_link(xform=wp.transform(p=palm_pos, q=wp.quat_identity()), label="hand_palm")
        left = builder.add_link(
            xform=wp.transform(p=wp.vec3(0.0, finger_y, 0.12), q=wp.quat_identity()),
            label="left_finger",
        )
        right = builder.add_link(
            xform=wp.transform(p=wp.vec3(0.0, -finger_y, 0.12), q=wp.quat_identity()),
            label="right_finger",
        )

        builder.add_shape_box(
            palm,
            hx=0.025,
            hy=0.16,
            hz=0.055,
            cfg=contact_cfg,
            color=wp.vec3(0.22, 0.28, 0.34),
            label="hand_rectangle",
        )
        builder.add_shape_box(
            left,
            hx=0.045,
            hy=0.035,
            hz=0.045,
            cfg=contact_cfg,
            color=wp.vec3(0.88, 0.48, 0.22),
            label="left_cube_finger",
        )
        builder.add_shape_box(
            right,
            hx=0.045,
            hy=0.035,
            hz=0.045,
            cfg=contact_cfg,
            color=wp.vec3(0.88, 0.48, 0.22),
            label="right_cube_finger",
        )

        fixed_joint = builder.add_joint_fixed(
            parent=-1,
            child=palm,
            parent_xform=wp.transform(p=palm_pos, q=wp.quat_identity()),
            child_xform=wp.transform(p=wp.vec3(0.0, 0.0, 0.0), q=wp.quat_identity()),
            label="hand_world_fixed",
        )
        left_joint = builder.add_joint_prismatic(
            parent=palm,
            child=left,
            axis=wp.vec3(0.0, -1.0, 0.0),
            parent_xform=wp.transform(p=wp.vec3(0.07, finger_y, 0.0), q=wp.quat_identity()),
            child_xform=wp.transform(p=wp.vec3(0.0, 0.0, 0.0), q=wp.quat_identity()),
            target_pos=0.0,
            target_vel=0.0,
            target_ke=5.0e4 if self.scenario == "harsh" else 2.0e4,
            target_kd=500.0 if self.scenario == "harsh" else 250.0,
            limit_lower=0.0,
            limit_upper=0.055,
            limit_ke=5.0e4,
            limit_kd=100.0,
            effort_limit=500.0 if self.scenario == "harsh" else 250.0,
            label="left_finger_slide",
        )
        right_joint = builder.add_joint_prismatic(
            parent=palm,
            child=right,
            axis=wp.vec3(0.0, 1.0, 0.0),
            parent_xform=wp.transform(p=wp.vec3(0.07, -finger_y, 0.0), q=wp.quat_identity()),
            child_xform=wp.transform(p=wp.vec3(0.0, 0.0, 0.0), q=wp.quat_identity()),
            target_pos=0.0,
            target_vel=0.0,
            target_ke=5.0e4 if self.scenario == "harsh" else 2.0e4,
            target_kd=500.0 if self.scenario == "harsh" else 250.0,
            limit_lower=0.0,
            limit_upper=0.055,
            limit_ke=5.0e4,
            limit_kd=100.0,
            effort_limit=500.0 if self.scenario == "harsh" else 250.0,
            label="right_finger_slide",
        )
        builder.add_articulation([fixed_joint, left_joint, right_joint], label="proxy_joint_gripper")
        return [palm, left, right], [fixed_joint, left_joint, right_joint]

    def _update_gripper_target(self) -> None:
        if self.control.joint_target_q is None:
            return
        alpha = min(1.0, self.sim_time / self.close_time)
        target = self.close_distance * alpha
        wp.launch(
            _set_finger_targets,
            dim=1,
            inputs=[
                self.control.joint_target_q,
                self.left_target_index,
                self.right_target_index,
                target,
            ],
            device=self.model.device,
        )

    def _particle_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        particle_q = self.state_0.particle_q.numpy()[self.soft_particle_start : self.soft_particle_end]
        return np.min(particle_q, axis=0), np.max(particle_q, axis=0)

    def simulate(self) -> None:
        self.collision_pipeline.collide(self.state_0, self.contacts)
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self) -> None:
        self._update_gripper_target()
        if self.graph is not None:
            with wp.ScopedDevice(self.model.device):
                wp.capture_launch(self.graph)
        else:
            self.simulate()
        self.sim_time += self.frame_dt

    def test_final(self) -> None:
        joint_q = self.state_0.joint_q.numpy()
        left_q = float(joint_q[self.left_q_index])
        right_q = float(joint_q[self.right_q_index])
        assert left_q > 0.015, f"Left finger did not close enough: q={left_q:.5f}"
        assert right_q > 0.015, f"Right finger did not close enough: q={right_q:.5f}"

        particle_q = self.state_0.particle_q.numpy()[self.soft_particle_start : self.soft_particle_end]
        assert np.all(np.isfinite(particle_q)), "Soft object particle positions contain non-finite values"
        min_pos = np.min(particle_q, axis=0)
        max_pos = np.max(particle_q, axis=0)
        initial_min, initial_max = self.initial_particle_bounds
        bbox_limit = 0.8 if self.scenario == "harsh" else 0.35
        bbox_size = float(np.linalg.norm(max_pos - min_pos))
        assert bbox_size < bbox_limit, f"Soft object expanded beyond expected bounds: size={bbox_size:.5f}"

        center_z = float(np.mean(particle_q[:, 2]))
        initial_center_z = float(0.5 * (initial_min[2] + initial_max[2]))
        center_drift = abs(center_z - initial_center_z)
        center_drift_limit = 0.16 if self.scenario == "harsh" else 0.06
        assert center_drift < center_drift_limit, (
            f"Soft object drifted unexpectedly from the gripper: center_drift={center_drift:.5f}"
        )

    def render(self) -> None:
        self.viewer.begin_frame(self.sim_time)
        newton.examples.log_coupled_view(self, self.contacts)
        self.viewer.end_frame()

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.set_defaults(num_frames=120)
        parser.add_argument(
            "--scenario",
            type=str,
            choices=["default", "harsh"],
            default="default",
            help="Scene configuration to run.",
        )
        parser.add_argument("--substeps", type=int, default=6, help="Simulation substeps per rendered frame.")
        newton.examples.add_coupled_view_args(parser)
        parser.add_argument("--vbd-iterations", type=int, default=30, help="VBD iterations per substep.")
        parser.add_argument("--mujoco-iterations", type=int, default=20, help="MuJoCo solver iterations.")
        parser.add_argument("--proxy-iterations", type=int, default=1, help="Proxy coupling iterations per substep.")
        parser.add_argument(
            "--mass-scale",
            type=float,
            default=1.0,
            help="Scale factor for MuJoCo effective mass/inertia used by VBD proxy bodies.",
        )
        parser.add_argument(
            "--proxy-relaxation",
            type=float,
            default=1.0,
            help="Relaxation factor for proxy feedback forces.",
        )
        parser.add_argument(
            "--proxy-relaxation-mode",
            choices=["fixed", "aitken"],
            default="fixed",
            help="Proxy feedback relaxation mode.",
        )
        parser.add_argument("--proxy-relaxation-min", type=float, default=0.1, help="Minimum Aitken relaxation.")
        parser.add_argument("--proxy-relaxation-max", type=float, default=1.0, help="Maximum Aitken relaxation.")
        parser.add_argument(
            "--proxy-joints",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="Keep source-owned gripper joints enabled in the VBD proxy view.",
        )
        parser.add_argument(
            "--coupling-mode",
            type=str,
            choices=["lagged", "staggered"],
            default="lagged",
            help="Proxy state transfer mode.",
        )
        parser.add_argument("--close-distance", type=float, default=None, help="Final inward finger target [m].")
        parser.add_argument("--close-time", type=float, default=None, help="Finger closing ramp duration [s].")
        parser.add_argument(
            "--no-graph-capture",
            action="store_false",
            dest="graph_capture",
            default=True,
            help="Disable CUDA graph capture.",
        )
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args)
    newton.examples.run(example, args)
