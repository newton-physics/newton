# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Franka Soft IK Lift
#
# Lift a soft beam with Franka IK and lagged proxy coupling.
#
# Command: python -m newton.examples franka_soft_ik_lift
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
BEAM_CENTER = (0.5, 0.0, 0.05)
GRIPPER_DOWN = (1.0, 0.0, 0.0, 0.0)
GRIP_OPEN = 0.04
GRIP_CLOSE = 0.0  # This URDF needs full closure to pinch the beam.


@wp.kernel
def set_gripper_q(joint_q: wp.array2d[float], finger_pos: wp.array[float], idx0: int, idx1: int):
    joint_q[0, idx0] = finger_pos[0]
    joint_q[0, idx1] = finger_pos[0]


@wp.kernel
def set_task_targets(
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
        self.sim_time = 0.0
        self.frame_dt = 1.0 / 30.0
        self.sim_substeps = 8
        self.sim_dt = 1.0 / 240.0

        self._build_scene()
        self.control = self.model.control()
        self._build_solver()
        self._build_keyframes()
        self._build_ik()

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.viewer.set_model(self.model)
        if isinstance(self.viewer, newton.viewer.ViewerGL):
            self.viewer.set_camera(pos=wp.vec3(1.3, -0.6, 0.5), pitch=-10.0, yaw=125.0)
            if hasattr(self.viewer.camera, "look_at"):
                self.viewer.camera.look_at(wp.vec3(*BEAM_CENTER))

        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_1)
        self.graph = None
        if self.model.device.is_cuda:
            with wp.ScopedDevice(self.model.device), wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph

    @staticmethod
    def _add_franka(builder):
        builder.add_urdf(
            newton.utils.download_asset("franka_emika_panda") / "urdf/fr3_franka_hand.urdf",
            floating=False,
            enable_self_collisions=False,
            parse_visuals_as_colliders=False,
            force_show_colliders=False,
        )
        builder.joint_q[: len(FRANKA_Q)] = FRANKA_Q
        builder.joint_target_q[: len(FRANKA_Q)] = FRANKA_Q

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
        particle_start = builder.particle_count
        youngs_modulus = 2.0e5
        poissons_ratio = 0.3
        builder.add_soft_grid(
            pos=wp.vec3(0.35, -0.02, 0.03),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0, 0.0, 0.0),
            dim_x=15,
            dim_y=2,
            dim_z=2,
            cell_x=0.02,
            cell_y=0.02,
            cell_z=0.02,
            density=1000.0,
            k_mu=youngs_modulus / (2.0 * (1.0 + poissons_ratio)),
            k_lambda=youngs_modulus * poissons_ratio / ((1.0 + poissons_ratio) * (1.0 - 2.0 * poissons_ratio)),
            k_damp=0.0,
            particle_radius=0.0025,
            label="soft_beam",
        )
        self.soft_particles = list(range(particle_start, builder.particle_count))

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
        self.model.soft_contact_ke = 8.0e3
        self.model.soft_contact_kd = 1.0e-2
        self.model.soft_contact_mu = 10.0
        self.initial_beam_height = float(np.mean(self.model.particle_q.numpy()[self.soft_particles, 2]))

    def _build_solver(self):
        self.solver = SolverCoupledProxy(
            model=self.model,
            entries=[
                SolverCoupled.Entry(
                    name="rigid",
                    solver=lambda model: SolverMuJoCo(
                        model=model,
                        solver="newton",
                        cone="elliptic",
                        integrator="implicitfast",
                        iterations=100,
                        ls_iterations=50,
                    ),
                    bodies=self.franka_bodies,
                    joints=self.franka_joints,
                ),
                SolverCoupled.Entry(
                    name="soft",
                    solver=lambda model: SolverVBD(
                        model=model,
                        iterations=10,
                        rigid_body_particle_contact_buffer_size=256,
                        rigid_compliant_alm=True,
                    ),
                    particles=self.soft_particles,
                ),
            ],
            coupling=SolverCoupledProxy.Config(
                proxies=[
                    SolverCoupledProxy.Proxy(
                        source="rigid",
                        destination="soft",
                        bodies=self.gripper_bodies,
                        mode="lagged",
                        collision_pipeline=lambda model: newton.CollisionPipeline(
                            model,
                            enable_rigid_soft_full_surface_contact=True,
                        ),
                        collide_interval=1,
                    )
                ],
                iterations=1,
            ),
        )

    def _build_keyframes(self):
        x, y, _ = BEAM_CENTER
        qx, qy, qz, qw = GRIPPER_DOWN
        grasp_z = 0.012
        poses = np.array(
            [
                [1.5, x, y, grasp_z + 0.1, qx, qy, qz, qw, GRIP_OPEN],
                [1.5, x, y, grasp_z, qx, qy, qz, qw, GRIP_OPEN],
                [0.1, x, y, grasp_z, qx, qy, qz, qw, GRIP_CLOSE],
                [1.5, x, y, grasp_z, qx, qy, qz, qw, GRIP_CLOSE],
                [1.5, x, y, 0.35, qx, qy, qz, qw, GRIP_CLOSE],
                [2.0, x, y, 0.35, qx, qy, qz, qw, GRIP_CLOSE],
            ],
            dtype=np.float32,
        )
        self.targets = poses[:, 1:]
        self.key_times = np.cumsum(poses[:, 0])

    def _build_ik(self):
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
        self._add_franka(builder)
        self.ik_model = builder.finalize(device=self.model.device)
        self.n_coords = self.ik_model.joint_coord_count
        self.ik_joint_q = wp.array(self.model.joint_q, shape=(1, self.n_coords))
        self.control_joint_target_q = self.control.joint_target_q.reshape((1, self.n_coords))
        self.finger_pos = wp.full(1, GRIP_OPEN, dtype=float, device=self.model.device)

        target_pos = wp.vec3(*self.targets[0, :3].tolist())
        target_rot = wp.vec4(*self.targets[0, 3:7].tolist())
        self.target_position = wp.array([target_pos], dtype=wp.vec3, device=self.model.device)
        self.target_rotation = wp.array([target_rot], dtype=wp.vec4, device=self.model.device)
        hand_body = next(i for i, label in enumerate(self.ik_model.body_label) if label.endswith("fr3_hand"))
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

    def update_ik_targets(self):
        t = min(self.sim_time, float(self.key_times[-1]) - 1.0e-6)
        interval = int(np.searchsorted(self.key_times, t))
        t_start = self.key_times[interval - 1] if interval else 0.0
        alpha = float(np.clip((t - t_start) / (self.key_times[interval] - t_start), 0.0, 1.0))
        current = self.targets[interval]
        previous = self.targets[interval - 1] if interval else current
        target = (1.0 - alpha) * previous + alpha * current
        wp.launch(
            set_task_targets,
            dim=1,
            inputs=[
                self.target_position,
                self.target_rotation,
                self.finger_pos,
                wp.vec3(*target[:3].tolist()),
                wp.vec4(*target[3:7].tolist()),
                float(target[-1]),
            ],
            device=self.model.device,
        )

    def simulate(self):
        self.ik_solver.step(self.ik_joint_q, self.ik_joint_q, iterations=24)
        wp.launch(
            set_gripper_q,
            dim=1,
            inputs=[self.ik_joint_q, self.finger_pos, self.n_coords - 2, self.n_coords - 1],
            device=self.model.device,
        )
        wp.copy(self.control_joint_target_q, self.ik_joint_q)
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            self.solver.step(self.state_0, self.state_1, self.control, None, self.sim_dt)
            newton.eval_ik(self.model, self.state_1, self.state_1.joint_q, self.state_1.joint_qd)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        self.update_ik_targets()
        if self.graph is None:
            self.simulate()
        else:
            wp.capture_launch(self.graph)
        self.sim_time += self.frame_dt

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()

    def test_final(self):
        """Verify the soft beam remains finite and is lifted above the table."""
        particle_q = self.state_0.particle_q.numpy()[self.soft_particles]
        particle_qd = self.state_0.particle_qd.numpy()[self.soft_particles]
        assert np.all(np.isfinite(particle_q)), "Soft beam positions contain NaN or inf values"
        assert np.all(np.isfinite(particle_qd)), "Soft beam velocities contain NaN or inf values"
        height = float(np.mean(particle_q[:, 2]))
        assert height > self.initial_beam_height + 0.02, f"Soft beam was not lifted: COM height {height:.3f} m"


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    parser.set_defaults(num_frames=240)
    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args)
    newton.examples.run(example, args)
