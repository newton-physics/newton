# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example MuJoCo Franka + Rigid Chain ADMM Pick-and-Place
#
# A fixed-base Franka arm tracks a pick-and-place IK sequence through MuJoCo
# position targets while a short rigid payload chain is simulated by XPBD by
# default. The original VBD cable payload is kept as an alternate mode for A/B
# testing. SolverCoupledADMM detects rigid-rigid contacts between the robot and
# the payload from the model collision pairs, and the same template is
# replicated across many worlds to exercise ADMM contact scaling.
#
# Command: python -m newton.examples mujoco_franka_vbd_cable_admm_solver
#
###########################################################################

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import warp as wp
from newton.solvers.experimental.coupled import SolverCoupled, SolverCoupledADMM

import newton
import newton.examples
import newton.ik as ik
import newton.utils
from newton.solvers import SolverMuJoCo, SolverVBD, SolverXPBD

PAYLOAD_CENTER = wp.vec3(0.5, 0.0, 0.256)
PAYLOAD_LENGTH = 0.42

# Top-down gripper orientation: 180 deg about world x flips the hand z-axis to -z.
GRIPPER_DOWN = (1.0, 0.0, 0.0, 0.0)  # (qx, qy, qz, qw)

GRIP_OPEN = 0.04
GRIP_CLOSE = 0.0
GRIP_HOLD_FACTOR = 0.0
GRIP_FORCE = 1000.0
GRIP_STIFFNESS = 1000.0

# Raised-arm, open-gripper starting configuration.
FRANKA_Q = [
    0.0,
    -0.569,
    0.0,
    -2.810,
    0.0,
    3.037,
    0.741,
    GRIP_OPEN,
    GRIP_OPEN,
]


@wp.kernel
def set_gripper_q(joint_q: wp.array2d[float], finger_pos: wp.array[float], idx0: int, idx1: int):
    world_idx = wp.tid()
    joint_q[world_idx, idx0] = finger_pos[world_idx]
    joint_q[world_idx, idx1] = finger_pos[world_idx]


@wp.kernel
def set_task_targets(
    target_positions: wp.array[wp.vec3],
    target_rotations: wp.array[wp.vec4],
    finger_pos: wp.array[float],
    pos: wp.vec3,
    rot: wp.vec4,
    grip_width: float,
):
    world_idx = wp.tid()
    target_positions[world_idx] = pos
    target_rotations[world_idx] = rot
    finger_pos[world_idx] = grip_width


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


def _find_label_index(labels: list[str], suffix: str) -> int:
    for index, label in enumerate(labels):
        if label.endswith(suffix):
            return index
    raise ValueError(f"Could not find label ending in {suffix!r}")


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
        self.payload_kind = str(args.payload_kind)
        self.payload_segments = max(2, int(args.payload_segments))
        self.payload_radius = float(args.payload_radius)
        self.surface_z = float(PAYLOAD_CENTER[2]) - self.payload_radius
        self.grip_hold = min(GRIP_OPEN, max(GRIP_CLOSE, GRIP_HOLD_FACTOR * self.payload_radius))

        template = newton.ModelBuilder(gravity=-9.81)
        template.rigid_gap = 0.005
        SolverMuJoCo.register_custom_attributes(template)
        if self.payload_kind == "vbd-cable":
            SolverVBD.register_custom_attributes(template, dahl_defaults_enabled=False)
        self._emit_template(template)

        bodies_per_world = template.body_count
        joints_per_world = template.joint_count
        shapes_per_world = template.shape_count

        builder = newton.ModelBuilder(gravity=-9.81)
        builder.replicate(template, world_count=self.world_count)
        self._expand_world_indices(bodies_per_world, joints_per_world, shapes_per_world)
        self.ground_shapes = [self._emit_ground_plane(builder)]

        builder.color()
        self.model = builder.finalize()
        self.device = self.model.device
        self.use_graph = self.use_graph and self.device.is_cuda
        self._count_admm_shape_pairs_per_world()

        mujoco_contact_budget = max(64, 16 * self.world_count)
        payload_name = "vbd" if self.payload_kind == "vbd-cable" else "xpbd"
        payload_solver = self._make_payload_solver(args)
        self.solver = SolverCoupledADMM(
            model=self.model,
            entries=[
                SolverCoupled.Entry(
                    name="mjc",
                    solver=lambda v: SolverMuJoCo(
                        model=v,
                        solver="newton",
                        integrator="implicitfast",
                        iterations=int(args.mujoco_iterations),
                        ls_iterations=int(args.mujoco_ls_iterations),
                        use_mujoco_contacts=False,
                        njmax=max(256, 64 * self.world_count),
                        nconmax=mujoco_contact_budget,
                    ),
                    bodies=self.franka_bodies,
                    joints=self.franka_joints,
                ),
                SolverCoupled.Entry(
                    name=payload_name,
                    solver=payload_solver,
                    bodies=self.payload_bodies,
                    joints=self.payload_joints,
                ),
            ],
            coupling=SolverCoupledADMM.Config(
                iterations=int(args.admm_iterations),
                rho=float(args.rho),
                gamma=float(args.gamma),
                baumgarte=float(args.baumgarte),
                rigid_contact_matching=str(args.rigid_contact_matching),
                contact_matching_pos_threshold=args.contact_matching_pos_threshold,
                contact_matching_normal_dot_threshold=args.contact_matching_normal_dot_threshold,
                contact_matching_force_scale=args.contact_matching_force_scale,
                contact_pairs=[
                    SolverCoupledADMM.ContactPair(
                        source="mjc",
                        destination=payload_name,
                    ),
                ],
            ),
        )

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.collision_pipeline = newton.CollisionPipeline(
            self.model,
        )
        self.contacts = self.collision_pipeline.contacts()
        self.solver.prepare_contacts(self.contacts)
        self.control = self.model.control()
        self._build_keyframes()
        self._build_ik()

        newton.examples.configure_coupled_view(self, args)
        self.viewer.set_world_offsets((1.1, 1.1, 0.0))
        if isinstance(self.viewer, newton.viewer.ViewerGL):
            scale = max(1.0, float(np.sqrt(self.world_count)))
            self.viewer.set_camera(pos=wp.vec3(0.9 * scale, -1.7 * scale, 0.95 * scale), pitch=-18.0, yaw=120.0)
            if hasattr(self.viewer.camera, "look_at"):
                self.viewer.camera.look_at(wp.vec3(0.45, 0.0, 0.28))

        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_1)

        self.capture()

    def _make_payload_solver(self, args):
        if self.payload_kind == "vbd-cable":
            vbd_iterations = int(args.vbd_iterations)
            return lambda v: SolverVBD(
                model=v,
                iterations=vbd_iterations,
                rigid_contact_history=False,
            )
        if self.payload_kind == "xpbd-chain":
            xpbd_iterations = int(args.xpbd_iterations)
            joint_linear_relaxation = float(args.xpbd_joint_linear_relaxation)
            joint_angular_relaxation = float(args.xpbd_joint_angular_relaxation)
            return lambda v: SolverXPBD(
                model=v,
                iterations=xpbd_iterations,
                joint_linear_relaxation=joint_linear_relaxation,
                joint_angular_relaxation=joint_angular_relaxation,
                angular_damping=0.02,
            )
        raise ValueError(f"Unsupported payload kind {self.payload_kind!r}")

    def _emit_template(self, builder: newton.ModelBuilder) -> None:
        franka_body_start = builder.body_count
        franka_joint_start = builder.joint_count
        franka_shape_start = builder.shape_count

        self._add_franka(builder, self.surface_z)
        builder.joint_target_ke[:7] = [900.0] * 7
        builder.joint_target_kd[:7] = [90.0] * 7
        builder.joint_target_ke[7:9] = [GRIP_STIFFNESS, GRIP_STIFFNESS]
        builder.joint_target_kd[7:9] = [100.0, 100.0]
        builder.joint_effort_limit[:7] = [80.0] * 7
        builder.joint_effort_limit[7:9] = [GRIP_FORCE, GRIP_FORCE]
        builder.joint_armature[:7] = [0.05] * 7
        builder.joint_armature[7:9] = [0.0, 0.0]

        franka_body_end = builder.body_count
        franka_joint_end = builder.joint_count
        franka_shape_end = builder.shape_count
        franka_bodies = list(range(franka_body_start, franka_body_end))

        gravcomp = builder.custom_attributes["mujoco:gravcomp"]
        if gravcomp.values is None:
            gravcomp.values = {}
        for body in franka_bodies:
            gravcomp.values[body] = 1.0

        payload_shape_start = builder.shape_count
        if self.payload_kind == "vbd-cable":
            payload_bodies, payload_joints = self._emit_vbd_cable(builder)
        else:
            payload_bodies, payload_joints = self._emit_xpbd_chain(builder)

        self.franka_bodies = franka_bodies
        self.franka_joints = list(range(franka_joint_start, franka_joint_end))
        self.franka_shapes = list(range(franka_shape_start, franka_shape_end))
        self.payload_bodies = payload_bodies
        self.payload_joints = payload_joints
        self.payload_shapes = list(range(payload_shape_start, builder.shape_count))
        self.payload_body_count_per_world = len(payload_bodies)
        self.payload_mid_body_offset = self.payload_body_count_per_world // 2

    @staticmethod
    def _add_franka(builder: newton.ModelBuilder, base_z: float) -> None:
        builder.add_urdf(
            newton.utils.download_asset("franka_emika_panda") / "urdf/fr3_franka_hand.urdf",
            xform=wp.transform(wp.vec3(0.0, 0.0, base_z), wp.quat_identity()),
            floating=False,
            enable_self_collisions=False,
            parse_visuals_as_colliders=False,
            force_show_colliders=False,
        )
        builder.joint_q[: len(FRANKA_Q)] = FRANKA_Q
        builder.joint_target_q[: len(FRANKA_Q)] = FRANKA_Q

    def _emit_ground_plane(self, builder: newton.ModelBuilder) -> int:
        plane_cfg = newton.ModelBuilder.ShapeConfig(ke=8.0e4, kd=2.0e1, mu=0.8, margin=0.001, gap=0.002)
        return builder.add_ground_plane(
            height=self.surface_z,
            cfg=plane_cfg,
            label="payload_ground_plane",
        )

    def _emit_vbd_cable(self, builder: newton.ModelBuilder) -> tuple[list[int], list[int]]:
        stretch_stiffness = 2.0e5
        bend_stiffness = 0.08
        cable_cfg = newton.ModelBuilder.ShapeConfig(
            density=1400.0,
            ke=5.0e4,
            kd=1.0e1,
            mu=0.9,
            margin=0.001,
            gap=0.002,
        )
        points, quats = newton.utils.create_straight_cable_points_and_quaternions(
            start=PAYLOAD_CENTER - wp.vec3(0.5 * PAYLOAD_LENGTH, 0.0, 0.0),
            direction=wp.vec3(1.0, 0.0, 0.0),
            length=PAYLOAD_LENGTH,
            num_segments=self.payload_segments,
            twist_total=0.0,
        )
        return builder.add_rod(
            positions=points,
            quaternions=quats,
            radius=self.payload_radius,
            body_frame_origin="start",
            cfg=cable_cfg,
            stretch_stiffness=stretch_stiffness,
            stretch_damping=2.0e-2,
            bend_stiffness=bend_stiffness,
            bend_damping=2.0e-2 * bend_stiffness,
            label="vbd_cable",
        )

    def _emit_xpbd_chain(self, builder: newton.ModelBuilder) -> tuple[list[int], list[int]]:
        chain_length = PAYLOAD_LENGTH
        segment_length = chain_length / float(self.payload_segments)
        segment_half_length = 0.5 * segment_length
        capsule_half_height = max(0.25 * self.payload_radius, segment_half_length - self.payload_radius)
        start = PAYLOAD_CENTER - wp.vec3(0.5 * PAYLOAD_LENGTH, 0.0, -0.002)
        direction = wp.vec3(1.0, 0.0, 0.0)
        capsule_rot = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), 0.5 * wp.pi)
        shape_xform = wp.transform(p=wp.vec3(0.0), q=capsule_rot)
        shape_cfg = newton.ModelBuilder.ShapeConfig(
            density=900.0,
            ke=6.0e4,
            kd=1.5e1,
            mu=0.9,
            margin=0.001,
            gap=0.002,
        )

        bodies = []
        joints = []
        for segment in range(self.payload_segments):
            center = start + direction * ((float(segment) + 0.5) * segment_length)
            body = builder.add_link(
                xform=wp.transform(p=center, q=wp.quat_identity()),
                label=f"xpbd_chain_link_{segment}",
            )
            builder.add_shape_capsule(
                body,
                xform=shape_xform,
                radius=self.payload_radius,
                half_height=capsule_half_height,
                cfg=shape_cfg,
                label=f"xpbd_chain_capsule_{segment}",
            )
            bodies.append(body)
            if segment == 0:
                joints.append(builder.add_joint_free(child=body, label="xpbd_chain_root"))
                continue

            joints.append(
                builder.add_joint_ball(
                    parent=bodies[segment - 1],
                    child=body,
                    friction=0.02,
                    parent_xform=wp.transform(p=wp.vec3(segment_half_length, 0.0, 0.0), q=wp.quat_identity()),
                    child_xform=wp.transform(p=wp.vec3(-segment_half_length, 0.0, 0.0), q=wp.quat_identity()),
                    collision_filter_parent=True,
                    label=f"xpbd_chain_joint_{segment - 1}_{segment}",
                )
            )

        builder.add_articulation(joints, label="xpbd_chain")
        return bodies, joints

    def _expand_world_indices(self, bodies_per_world: int, joints_per_world: int, shapes_per_world: int) -> None:
        def expand(ids: list[int], stride: int) -> list[int]:
            return [world * stride + id_ for world in range(self.world_count) for id_ in ids]

        self.franka_bodies = expand(self.franka_bodies, bodies_per_world)
        self.franka_joints = expand(self.franka_joints, joints_per_world)
        self.franka_shapes = expand(self.franka_shapes, shapes_per_world)
        self.payload_bodies = expand(self.payload_bodies, bodies_per_world)
        self.payload_joints = expand(self.payload_joints, joints_per_world)
        self.payload_shapes = expand(self.payload_shapes, shapes_per_world)

    def _count_admm_shape_pairs_per_world(self) -> None:
        shape_body = self.model.shape_body.numpy()
        shape_world = self.model.shape_world.numpy()
        franka_bodies = set(self.franka_bodies)
        payload_bodies = set(self.payload_bodies)
        counts = np.zeros(self.world_count, dtype=np.int32)

        for pair in self.model.shape_contact_pairs.numpy():
            shape_a = int(pair[0])
            shape_b = int(pair[1])
            body_a = int(shape_body[shape_a])
            body_b = int(shape_body[shape_b])
            owner_a = self._body_owner(body_a, franka_bodies, payload_bodies)
            owner_b = self._body_owner(body_b, franka_bodies, payload_bodies)
            if {owner_a, owner_b} != {"mjc", "payload"}:
                continue
            world_a = int(shape_world[shape_a])
            world_b = int(shape_world[shape_b])
            if world_a != world_b:
                raise RuntimeError("Cross-world Franka-payload contact pair was generated")
            if 0 <= world_a < self.world_count:
                counts[world_a] += 1

        self.admm_shape_pairs_per_world = counts

    @staticmethod
    def _body_owner(body: int, franka_bodies: set[int], payload_bodies: set[int]) -> str | None:
        if body in franka_bodies:
            return "mjc"
        if body in payload_bodies:
            return "payload"
        return None

    def _payload_ground_shape_pairs(self) -> wp.array:
        payload_shapes = set(self.payload_shapes)
        ground_shapes = set(self.ground_shapes)
        pairs = [
            (shape_a, shape_b)
            for shape_a, shape_b in self.model.shape_contact_pairs.numpy()
            if ({int(shape_a), int(shape_b)} & payload_shapes) and ({int(shape_a), int(shape_b)} & ground_shapes)
        ]
        if not pairs:
            raise RuntimeError("No payload-ground contact pairs were generated")
        return wp.array(np.asarray(pairs, dtype=np.int32), dtype=wp.vec2i, device=self.model.device)

    def _build_ik(self) -> None:
        # IK runs on a Franka-only model so payload coordinates do not enter the solve.
        ik_builder = newton.ModelBuilder(gravity=-9.81)
        self._add_franka(ik_builder, self.surface_z)
        self.ik_model = ik_builder.finalize(device=self.device)

        self.n_coords = self.ik_model.joint_coord_count
        self.ik_joint_q = wp.clone(self.model.joint_q.reshape((self.world_count, -1))[:, : self.n_coords])
        self.control_joint_target_q = self.control.joint_target_q.reshape((self.world_count, -1))
        self.finger_idx0 = self.n_coords - 2
        self.finger_idx1 = self.n_coords - 1
        self.finger_pos_buf = wp.full(self.world_count, GRIP_OPEN, dtype=float, device=self.device)
        hand_body = _find_label_index(self.ik_model.body_label, "fr3_hand")

        target_pos = wp.vec3(*self.targets[0][:3].tolist())
        target_rot = wp.vec4(*self.targets[0][3:7].tolist())
        self.ik_target_positions = wp.array([target_pos] * self.world_count, dtype=wp.vec3, device=self.device)
        self.ik_target_rotations = wp.array([target_rot] * self.world_count, dtype=wp.vec4, device=self.device)

        self.pos_obj = ik.IKObjectivePosition(
            link_index=hand_body,
            link_offset=wp.vec3(0.0, 0.0, 0.107),
            target_positions=self.ik_target_positions,
        )
        self.rot_obj = ik.IKObjectiveRotation(
            link_index=hand_body,
            link_offset_rotation=wp.quat_identity(),
            target_rotations=self.ik_target_rotations,
        )
        joint_limit_lower = wp.clone(self.model.joint_limit_lower.reshape((self.world_count, -1))[:, : self.n_coords])
        joint_limit_upper = wp.clone(self.model.joint_limit_upper.reshape((self.world_count, -1))[:, : self.n_coords])
        self.joint_limits_obj = ik.IKObjectiveJointLimit(
            joint_limit_lower=joint_limit_lower.flatten(),
            joint_limit_upper=joint_limit_upper.flatten(),
            weight=10.0,
        )
        self.ik_solver = ik.IKSolver(
            model=self.ik_model,
            n_problems=self.world_count,
            objectives=[self.pos_obj, self.rot_obj, self.joint_limits_obj],
            lambda_initial=0.05,
            jacobian_mode=ik.IKJacobianType.ANALYTIC,
        )
        self.ik_iters = 24

    def _build_keyframes(self) -> None:
        cx, cy, cz = PAYLOAD_CENTER[0], PAYLOAD_CENTER[1], PAYLOAD_CENTER[2]
        approach_z = cz + 0.20
        grasp_z = cz
        tx, ty = 0.4, 0.25

        start_pos, start_rot = self._initial_tcp_pose()
        qx, qy, qz, qw = GRIPPER_DOWN
        self.place_target_xy = (tx, ty)
        poses = np.array(
            [
                [0.25, *start_pos.tolist(), *start_rot.tolist(), GRIP_OPEN],
                [0.5, cx, cy, approach_z, qx, qy, qz, qw, GRIP_OPEN],
                [0.5, cx, cy, grasp_z, qx, qy, qz, qw, GRIP_OPEN],
                [1.0, cx, cy, grasp_z, qx, qy, qz, qw, self.grip_hold],
                [1.0, cx, cy, approach_z, qx, qy, qz, qw, self.grip_hold],
                [1.0, tx, ty, approach_z, qx, qy, qz, qw, self.grip_hold],
                [0.5, tx, ty, grasp_z, qx, qy, qz, qw, self.grip_hold],
                [0.5, tx, ty, grasp_z, qx, qy, qz, qw, self.grip_hold],
                [1.0, tx, ty, grasp_z, qx, qy, qz, qw, GRIP_OPEN],
                [0.5, tx, ty, approach_z, qx, qy, qz, qw, GRIP_OPEN],
            ],
            dtype=np.float32,
        )
        self.targets = poses[:, 1:]
        self.key_times = np.cumsum(poses[:, 0])

    def _initial_tcp_pose(self) -> tuple[np.ndarray, np.ndarray]:
        state = self.model.state()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, state)
        hand_body = _find_label_index(self.model.body_label, "fr3_hand")
        hand_q = state.body_q.numpy()[hand_body]

        pos = wp.vec3(float(hand_q[0]), float(hand_q[1]), float(hand_q[2]))
        rot = wp.quat(float(hand_q[3]), float(hand_q[4]), float(hand_q[5]), float(hand_q[6]))
        tcp_pos = pos + wp.quat_rotate(rot, wp.vec3(0.0, 0.0, 0.107))

        return (
            np.array([float(tcp_pos[0]), float(tcp_pos[1]), float(tcp_pos[2])], dtype=np.float32),
            np.array([float(rot[0]), float(rot[1]), float(rot[2]), float(rot[3])], dtype=np.float32),
        )

    def update_ik_targets(self) -> None:
        """Interpolate keyframes and update IK target arrays before graph launch."""
        t = min(self.sim_time, float(self.key_times[-1]) - 1.0e-6)
        interval = int(np.searchsorted(self.key_times, t))
        t_start = self.key_times[interval - 1] if interval > 0 else 0.0
        t_end = self.key_times[interval]
        alpha = float(np.clip((t - t_start) / max(t_end - t_start, 1.0e-6), 0.0, 1.0))

        cur = self.targets[interval]
        prev = self.targets[interval - 1] if interval > 0 else cur
        interp = (1.0 - alpha) * prev + alpha * cur

        wp.launch(
            set_task_targets,
            dim=self.world_count,
            inputs=[
                self.ik_target_positions,
                self.ik_target_rotations,
                self.finger_pos_buf,
                wp.vec3(*interp[:3].tolist()),
                wp.vec4(*interp[3:7].tolist()),
                float(interp[-1]),
            ],
            device=self.device,
        )

    def capture(self):
        self.graph = _capture_frame_graph(self.model, self.simulate, enabled=self.use_graph)

    def simulate(self):
        self.ik_solver.step(self.ik_joint_q, self.ik_joint_q, iterations=self.ik_iters)
        wp.launch(
            set_gripper_q,
            dim=self.world_count,
            inputs=[self.ik_joint_q, self.finger_pos_buf, self.finger_idx0, self.finger_idx1],
            device=self.device,
        )
        wp.copy(dest=self.control_joint_target_q[:, : self.n_coords], src=self.ik_joint_q)

        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            newton.examples.apply_coupled_viewer_forces(self, self.state_0)
            self.model.collide(self.state_0, self.contacts, collision_pipeline=self.collision_pipeline)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            newton.eval_ik(self.model, self.state_1, self.state_1.joint_q, self.state_1.joint_qd)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        self.update_ik_targets()
        if not _launch_frame_graph(self.model, self.graph):
            self.simulate()
        self.sim_time += self.frame_dt

    def test_final(self):
        body_q = self.state_0.body_q.numpy()
        body_qd = self.state_0.body_qd.numpy()
        assert np.all(np.isfinite(body_q)), "Body positions contain NaN or inf values"
        assert np.all(np.isfinite(body_qd)), "Body velocities contain NaN or inf values"
        assert np.all(self.admm_shape_pairs_per_world > 0), "Each world should have Franka-payload ADMM contact pairs"
        assert np.all(self.admm_shape_pairs_per_world == self.admm_shape_pairs_per_world[0]), (
            "Franka-payload ADMM contact pair counts should be identical across replicated worlds"
        )
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
        parser.set_defaults(world_count=8)
        parser.add_argument("--substeps", type=int, default=16, help="Coupled substeps per rendered frame.")
        parser.add_argument("--admm-iterations", type=int, default=5, help="ADMM iterations per coupled substep.")
        parser.add_argument("--rho", type=float, default=200.0, help="ADMM penalty parameter.")
        parser.add_argument("--gamma", type=float, default=0.001, help="ADMM proximal metric scale.")
        parser.add_argument("--baumgarte", type=float, default=0.5, help="Position error correction fraction.")
        parser.add_argument(
            "--rigid-contact-matching",
            choices=["disabled", "latest", "sticky"],
            default="latest",
            help="ADMM Franka-payload rigid contact matching mode.",
        )
        parser.add_argument(
            "--contact-matching-pos-threshold",
            type=float,
            default=None,
            help="ADMM rigid contact matching midpoint distance threshold [m]; omitted uses CollisionPipeline default.",
        )
        parser.add_argument(
            "--contact-matching-normal-dot-threshold",
            type=float,
            default=None,
            help="ADMM rigid contact matching normal dot-product threshold; omitted uses CollisionPipeline default.",
        )
        parser.add_argument(
            "--contact-matching-force-scale",
            type=float,
            default=0.9,
            help="Multiplier for matched previous-step ADMM rigid contact lambda warm-starts.",
        )
        parser.add_argument(
            "--payload-kind",
            choices=["xpbd-chain", "vbd-cable"],
            default="xpbd-chain",
            help="Payload simulated by the non-MuJoCo solver.",
        )
        parser.add_argument("--payload-segments", type=int, default=11, help="Number of payload rigid/cable segments.")
        parser.add_argument("--payload-radius", type=float, default=0.012, help="Payload capsule/cable radius [m].")
        parser.add_argument("--xpbd-iterations", type=int, default=16, help="XPBD iterations per coupled substep.")
        parser.add_argument(
            "--xpbd-joint-linear-relaxation",
            type=float,
            default=0.9,
            help="XPBD joint linear relaxation for the rigid-chain payload.",
        )
        parser.add_argument(
            "--xpbd-joint-angular-relaxation",
            type=float,
            default=0.5,
            help="XPBD joint angular relaxation for the rigid-chain payload.",
        )
        parser.add_argument("--vbd-iterations", type=int, default=8, help="VBD iterations per coupled substep.")
        parser.add_argument("--mujoco-iterations", type=int, default=12, help="MuJoCo solver iterations.")
        parser.add_argument("--mujoco-ls-iterations", type=int, default=25, help="MuJoCo line-search iterations.")
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
