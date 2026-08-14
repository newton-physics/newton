# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Franka Cloth IK Lift
#
# Lift a cloth patch with Franka IK and lagged proxy coupling.
#
# Command: python -m newton.examples franka_cloth_ik_lift
###########################################################################

from __future__ import annotations

import numpy as np
import warp as wp
from newton.solvers.experimental.coupled import SolverCoupled, SolverCoupledProxy

import newton
import newton.examples
import newton.ik as ik
import newton.utils
from newton.solvers import SolverMuJoCo, SolverVBD

FRANKA_Q = [0.0, -0.569, 0.0, -2.810, 0.0, 3.037, 0.741, 0.04, 0.04]
GRIP_OPEN = 0.04
GRIP_CLOSED = 0.0
CONTACT_KE = 8.0e3


@wp.kernel
def set_gripper_target(joint_q: wp.array2d[float], finger_pos: wp.array[float], idx0: int, idx1: int):
    joint_q[0, idx0] = finger_pos[0]
    joint_q[0, idx1] = finger_pos[0]


@wp.kernel
def set_task_target(
    target_position: wp.array[wp.vec3],
    target_rotation: wp.array[wp.vec4],
    finger_pos: wp.array[float],
    pos: wp.vec3,
    rot: wp.vec4,
    grip: float,
):
    target_position[0] = pos
    target_rotation[0] = rot
    finger_pos[0] = grip


class Example:
    def __init__(self, viewer, args):
        self.viewer = viewer
        self.frame_dt = 1.0 / 30.0
        self.sim_substeps = 8
        self.sim_dt = 1.0 / 240.0
        self.sim_time = 0.0
        newton.use_coord_layout_targets = True
        self._build_scene()
        self.control = self.model.control()
        self._build_solver()
        self._build_ik()

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_1)

        self.viewer.set_model(self.model)
        if isinstance(viewer, newton.viewer.ViewerGL):
            viewer.set_camera(pos=wp.vec3(0.9, -1.4, 0.9), pitch=-22.0, yaw=120.0)
            if hasattr(viewer.camera, "look_at"):
                viewer.camera.look_at(wp.vec3(0.4, 0.0, 0.2))
        self.capture()

    @staticmethod
    def _add_franka(builder):
        builder.add_urdf(
            newton.utils.download_asset("franka_emika_panda") / "urdf/fr3_franka_hand.urdf",
            floating=False,
            enable_self_collisions=False,
            parse_visuals_as_colliders=False,
            force_show_colliders=False,
        )
        builder.joint_q[:9] = FRANKA_Q
        builder.joint_target_q[:9] = FRANKA_Q

    def _build_scene(self):
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
        builder.rigid_gap = 0.01
        SolverMuJoCo.register_custom_attributes(builder)
        SolverVBD.register_custom_attributes(builder)

        body_start = builder.body_count
        joint_start = builder.joint_count
        self._add_franka(builder)
        self.franka_bodies = list(range(body_start, builder.body_count))
        self.franka_joints = list(range(joint_start, builder.joint_count))

        builder.joint_target_ke[:9] = [600.0, 600.0, 600.0, 600.0, 250.0, 150.0, 50.0, 350.0, 0.0]
        builder.joint_target_kd[:9] = [50.0, 50.0, 50.0, 50.0, 30.0, 25.0, 15.0, 175.0, 0.0]
        builder.joint_effort_limit[:9] = [87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0, 70.0, 1.0]
        builder.joint_velocity_limit[:9] = [2.175, 2.175, 2.175, 2.175, 2.61, 2.61, 2.61, 2.0, 2.0]
        builder.joint_armature[:9] = [0.6057, 0.6057, 0.4625, 0.4625, 0.2055, 0.2055, 0.2055, 0.1, 0.1]

        gravcomp = builder.custom_attributes["mujoco:gravcomp"]
        if gravcomp.values is None:
            gravcomp.values = {}
        for body in self.franka_bodies:
            gravcomp.values[body] = 1.0

        support_body_start = builder.body_count
        support_joint_start = builder.joint_count
        support_inertia = wp.mat33(
            0.0019083333,
            0.0,
            0.0,
            0.0,
            0.0027083333,
            0.0,
            0.0,
            0.0,
            0.0008666667,
        )
        support_cfg = newton.ModelBuilder.ShapeConfig(density=0.0, mu=0.01)
        for index, y in enumerate((0.02, -0.02)):
            body = builder.add_body(
                xform=wp.transform(wp.vec3(0.4, y, 0.075), wp.quat_identity()),
                mass=1.0,
                inertia=support_inertia,
                lock_inertia=True,
                is_kinematic=True,
                label=f"cloth_support_{index}",
            )
            builder.add_shape_box(body, hx=0.05, hy=0.01, hz=0.075, cfg=support_cfg)
        self.support_bodies = list(range(support_body_start, builder.body_count))
        self.support_joints = list(range(support_joint_start, builder.joint_count))

        self.particle_start = builder.particle_count
        resolution = 8
        vertices = [
            wp.vec3(-0.1 + 0.025 * x, -0.1 + 0.025 * y, 0.0)
            for y in range(resolution + 1)
            for x in range(resolution + 1)
        ]
        indices = []
        stride = resolution + 1
        for y in range(resolution):
            for x in range(resolution):
                v0 = y * stride + x
                v1, v2, v3 = v0 + 1, v0 + stride, v0 + stride + 1
                if (x % 2 == 0) != (y % 2 == 0):
                    indices.extend((v0, v1, v2, v1, v3, v2))
                else:
                    indices.extend((v0, v1, v3, v0, v3, v2))
        builder.add_cloth_mesh(
            pos=wp.vec3(0.4, 0.0, 0.17),
            rot=wp.quat(0.70710678, 0.0, 0.0, 0.70710678),
            scale=1.0,
            vel=wp.vec3(0.0, 0.0, 0.0),
            vertices=vertices,
            indices=indices,
            density=1.0,
            particle_radius=0.002,
            tri_ke=500.0,
            tri_ka=500.0,
            tri_kd=0.001,
            edge_ke=0.5,
            edge_kd=0.001,
            label="cloth",
        )
        self.cloth_particles = list(range(self.particle_start, builder.particle_count))

        self.gripper_bodies = [
            body
            for body in self.franka_bodies
            if "hand" in builder.body_label[body] or "finger" in builder.body_label[body]
        ]
        if not self.gripper_bodies:
            raise RuntimeError("Could not locate Franka gripper bodies")
        for shape in range(builder.shape_count):
            if builder.shape_body[shape] in self.gripper_bodies:
                builder.shape_force_sdf[shape] = True

        builder.add_shape_box(
            -1,
            xform=wp.transform(wp.vec3(0.5, 0.0, -0.525), wp.quat_identity()),
            hx=0.65,
            hy=0.45,
            hz=0.525,
            label="table",
        )
        builder.add_ground_plane(height=-1.05, label="ground")
        builder.color()
        self.model = builder.finalize()
        self.device = self.model.device
        self.model.soft_contact_ke = CONTACT_KE
        self.model.soft_contact_kd = 1.0e-2
        self.model.soft_contact_mu = 1000.0
        self.initial_cloth_height = float(np.mean(self.model.particle_q.numpy()[self.cloth_particles, 2]))

        self._build_waypoints()

    def _build_solver(self):
        source_bodies = self.franka_bodies + self.support_bodies
        source_joints = self.franka_joints + self.support_joints
        proxy_bodies = self.gripper_bodies + self.support_bodies
        self.solver = SolverCoupledProxy(
            model=self.model,
            entries=[
                SolverCoupled.Entry(
                    name="mjc",
                    solver=lambda view: SolverMuJoCo(
                        model=view,
                        solver="newton",
                        integrator="implicitfast",
                        cone="elliptic",
                        iterations=100,
                        ls_iterations=20,
                        use_mujoco_contacts=True,
                    ),
                    bodies=source_bodies,
                    joints=source_joints,
                ),
                SolverCoupled.Entry(
                    name="vbd",
                    solver=lambda view: SolverVBD(
                        model=view,
                        iterations=10,
                        particle_enable_self_contact=False,
                        rigid_body_particle_contact_buffer_size=1024,
                        rigid_contact_history=False,
                    ),
                    particles=self.cloth_particles,
                ),
            ],
            coupling=SolverCoupledProxy.Config(
                proxies=[
                    SolverCoupledProxy.Proxy(
                        source="mjc",
                        destination="vbd",
                        bodies=proxy_bodies,
                        mass_scale=1.0,
                        mode="lagged",
                        collision_pipeline=lambda model: newton.examples.create_collision_pipeline(
                            model,
                            broad_phase="explicit",
                            enable_rigid_soft_full_surface_contact=True,
                        ),
                        collide_interval=1,
                    )
                ],
                iterations=1,
            ),
        )

    def _build_ik(self):
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
        self._add_franka(builder)
        self.ik_model = builder.finalize(device=self.device)
        self.n_coords = self.ik_model.joint_coord_count
        self.ik_joint_q = wp.clone(self.model.joint_q[: self.n_coords].reshape((1, self.n_coords)))
        self.control_target_q = self.control.joint_target_q.reshape((1, -1))
        self.finger_pos = wp.array([GRIP_OPEN], dtype=float, device=self.device)

        hand_body = next(i for i, label in enumerate(self.ik_model.body_label) if label.endswith("fr3_hand"))
        first = self.targets[0]
        self.target_position = wp.array([wp.vec3(*first[:3])], dtype=wp.vec3, device=self.device)
        self.target_rotation = wp.array([wp.vec4(*first[3:7])], dtype=wp.vec4, device=self.device)
        position = ik.IKObjectivePosition(
            link_index=hand_body,
            link_offset=wp.vec3(0.0, 0.0, 0.107),
            target_positions=self.target_position,
        )
        rotation = ik.IKObjectiveRotation(
            link_index=hand_body,
            link_offset_rotation=wp.quat_identity(),
            target_rotations=self.target_rotation,
        )
        limits = ik.IKObjectiveJointLimit(
            joint_limit_lower=self.ik_model.joint_limit_lower,
            joint_limit_upper=self.ik_model.joint_limit_upper,
            weight=10.0,
        )
        self.ik_solver = ik.IKSolver(
            model=self.ik_model,
            n_problems=1,
            objectives=[position, rotation, limits],
            lambda_initial=0.6,
            jacobian_mode=ik.IKJacobianType.ANALYTIC,
        )

    def _build_waypoints(self):
        grasp_z = 0.16
        poses = np.array(
            [
                [1.5, 0.4, 0.0, grasp_z + 0.1, 1.0, 0.0, 0.0, 0.0, GRIP_OPEN],
                [1.5, 0.4, 0.0, grasp_z, 1.0, 0.0, 0.0, 0.0, GRIP_OPEN],
                [0.1, 0.4, 0.0, grasp_z, 1.0, 0.0, 0.0, 0.0, GRIP_CLOSED],
                [1.5, 0.4, 0.0, grasp_z, 1.0, 0.0, 0.0, 0.0, GRIP_CLOSED],
                [1.5, 0.4, 0.0, 0.375, 1.0, 0.0, 0.0, 0.0, GRIP_CLOSED],
                [2.0, 0.4, 0.0, 0.375, 1.0, 0.0, 0.0, 0.0, GRIP_CLOSED],
            ],
            dtype=np.float32,
        )
        self.targets = poses[:, 1:]
        self.key_times = np.cumsum(poses[:, 0])

    def _update_target(self):
        t = min(self.sim_time, float(self.key_times[-1]) - 1.0e-6)
        interval = int(np.searchsorted(self.key_times, t))
        start = self.key_times[interval - 1] if interval else 0.0
        alpha = float(np.clip((t - start) / (self.key_times[interval] - start), 0.0, 1.0))
        current = self.targets[interval]
        previous = self.targets[interval - 1] if interval else current
        target = (1.0 - alpha) * previous + alpha * current
        wp.launch(
            set_task_target,
            dim=1,
            inputs=[
                self.target_position,
                self.target_rotation,
                self.finger_pos,
                wp.vec3(*target[:3]),
                wp.vec4(*target[3:7]),
                float(target[-1]),
            ],
            device=self.device,
        )

    def capture(self):
        self.graph = None
        if self.device.is_cuda:
            with wp.ScopedDevice(self.device), wp.ScopedCapture() as capture:
                self.simulate()
            if capture.graph is None:
                raise RuntimeError(f"Graph capture failed on device {self.device}")
            self.graph = capture.graph

    def simulate(self):
        self.ik_solver.step(self.ik_joint_q, self.ik_joint_q, iterations=24)
        wp.launch(
            set_gripper_target,
            dim=1,
            inputs=[self.ik_joint_q, self.finger_pos, self.n_coords - 2, self.n_coords - 1],
            device=self.device,
        )
        wp.copy(self.control_target_q[:, : self.n_coords], self.ik_joint_q)
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            self.solver.step(self.state_0, self.state_1, self.control, None, self.sim_dt)
            newton.eval_ik(self.model, self.state_1, self.state_1.joint_q, self.state_1.joint_qd)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        self._update_target()
        if self.graph is not None:
            with wp.ScopedDevice(self.device):
                wp.capture_launch(self.graph)
        else:
            self.simulate()
        self.sim_time += self.frame_dt

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()

    def test_final(self):
        """Verify that the cloth remains finite and is lifted above its supports."""
        particle_q = self.state_0.particle_q.numpy()[self.particle_start :]
        assert np.all(np.isfinite(particle_q)), "Cloth positions contain non-finite values"
        center_z = float(np.mean(particle_q[:, 2]))
        assert center_z > self.initial_cloth_height + 0.02, f"Cloth was not lifted: COM height {center_z:.3f} m"


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    parser.set_defaults(num_frames=240)
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
