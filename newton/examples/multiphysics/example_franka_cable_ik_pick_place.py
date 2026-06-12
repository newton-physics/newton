# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Franka Cable IK Pick-and-Place (proxy coupling)
#
# A fixed-base Franka FR3 arm uses Newton's GPU IK solver to pick up a
# deformable cable and move it to a target location. The arm is simulated
# by MuJoCo (PD position targets driven from the IK result), the cable is
# a VBD rod, and SolverCoupledProxy couples the two: the gripper bodies are
# exposed to VBD as virtual proxies so the cable detects them as contacts.
#
# Cable geometry/material, Franka joint gains, contact parameters and the
# proxy coupling solver settings mirror IsaacLab's franka_cable_env_cfg.
#
# Standalone: depends only on newton + warp (no IsaacLab).
#
# Command: python -m newton.examples franka_cable_ik_pick_place_proxy
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

# Initial Franka joint configuration (7 arm + 2 finger).
FRANKA_Q = [
    -3.6802115e-03,
    2.3901723e-02,
    3.6804110e-03,
    -2.3683236e00,
    -1.2918962e-04,
    2.3922248e00,
    7.8549200e-01,
    0.04,
    0.04,
]

# Cable: 19 capsule segments spaced 0.02 m (20 nodes), radius 0.005 m, per IsaacLab.
CABLE_CENTER = wp.vec3(0.5, 0.0, 0.256)
CABLE_LENGTH = 0.38

# Top-down gripper orientation: 180 deg about world x flips the hand z-axis to -z.
GRIPPER_DOWN = (1.0, 0.0, 0.0, 0.0)  # (qx, qy, qz, qw)

# Finger joint targets [m] (per finger). The fingers solve in MuJoCo where the cable is
# absent, so a full close ratchets shut through the lagged proxy contact and extrudes the
# thin cable. Instead the grasp closes to GRIP_CLOSE to capture, then holds GRIP_HOLD (just
# under the cable radius) so the fingers pinch firmly without ratcheting past it.
GRIP_OPEN = 0.04
GRIP_CLOSE = 0.0
GRIP_HOLD = 0.004  # held finger position once grasped; tuned to the default cable radius
GRIP_FORCE = 1500.0  # gripper effort limit [N] (IsaacLab panda_hand)
GRIP_STIFFNESS = 1000.0  # finger PD stiffness [N/m] (IsaacLab panda_hand)


@wp.kernel
def set_gripper_q(joint_q: wp.array2d[float], finger_pos: wp.array[float], idx0: int, idx1: int):
    joint_q[0, idx0] = finger_pos[0]
    joint_q[0, idx1] = finger_pos[0]


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
        self.payload_segments = max(2, int(args.payload_segments))
        self.payload_radius = float(args.payload_radius)
        # Working-surface height: the cable rests on it and the arm is mounted on it.
        self.surface_z = float(CABLE_CENTER[2]) - self.payload_radius

        self._build_scene()
        self._build_solvers(args)
        self._build_ik()

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.collision_pipeline = newton.CollisionPipeline(
            self.model,
            broad_phase="explicit",
            shape_pairs_filtered=self._payload_ground_shape_pairs(),
        )
        self.contacts = self.collision_pipeline.contacts()
        self.solver.prepare_contacts(self.contacts)
        self.control = self.model.control()

        self.viewer.set_model(self.model)
        if isinstance(self.viewer, newton.viewer.ViewerGL):
            self.viewer.set_camera(pos=wp.vec3(0.9, -1.4, 0.9), pitch=-22.0, yaw=120.0)
            if hasattr(self.viewer.camera, "look_at"):
                self.viewer.camera.look_at(wp.vec3(0.45, 0.0, 0.28))

        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_1)

        self.capture()

    # ------------------------------------------------------------------
    # Scene
    # ------------------------------------------------------------------
    @staticmethod
    def _add_franka(builder, base_z):
        """Add the Franka FR3 arm URDF mounted on the working surface and seed its config."""
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

    def _build_scene(self):
        builder = newton.ModelBuilder(gravity=-9.81)
        builder.rigid_gap = 0.01
        SolverMuJoCo.register_custom_attributes(builder)
        SolverVBD.register_custom_attributes(builder, dahl_defaults_enabled=False)

        franka_body_start = builder.body_count
        franka_joint_start = builder.joint_count
        franka_shape_start = builder.shape_count

        # Franka arm, mounted on the working surface.
        self._add_franka(builder, self.surface_z)
        # IsaacLab FRANKA_PANDA_HIGH_PD_CFG gains (arm joints 1-4 / 5-7) plus stiff gripper.
        builder.joint_target_ke[:7] = [400.0] * 7
        builder.joint_target_kd[:7] = [80.0] * 7
        builder.joint_target_ke[7:9] = [GRIP_STIFFNESS, GRIP_STIFFNESS]
        builder.joint_target_kd[7:9] = [100.0, 100.0]
        builder.joint_effort_limit[:4] = [87.0] * 4
        builder.joint_effort_limit[4:7] = [12.0] * 3
        builder.joint_effort_limit[7:9] = [GRIP_FORCE, GRIP_FORCE]
        builder.joint_armature[:7] = [1.0e-3] * 7
        builder.joint_armature[7:9] = [0.0, 0.0]

        self.franka_bodies = list(range(franka_body_start, builder.body_count))
        self.franka_joints = list(range(franka_joint_start, builder.joint_count))
        self.franka_shapes = list(range(franka_shape_start, builder.shape_count))

        # Full per-body gravity compensation on the arm (IsaacLab gravcomp=1.0), so the
        # low-PD actuators do not fight gravity sag during task-space tracking.
        gravcomp = builder.custom_attributes["mujoco:gravcomp"]
        if gravcomp.values is None:
            gravcomp.values = {}
        for body in self.franka_bodies:
            gravcomp.values[body] = 1.0

        # VBD cable.
        payload_body_start = builder.body_count
        payload_joint_start = builder.joint_count
        payload_shape_start = builder.shape_count
        cable_cfg = newton.ModelBuilder.ShapeConfig(
            density=100.0,
            ke=1.0e4,
            kd=1.0e-5,
            mu=1.0,
            margin=0.0,
            gap=0.01,
        )
        points, quats = newton.utils.create_straight_cable_points_and_quaternions(
            start=CABLE_CENTER - wp.vec3(0.5 * CABLE_LENGTH, 0.0, 0.0),
            direction=wp.vec3(1.0, 0.0, 0.0),
            length=CABLE_LENGTH,
            num_segments=self.payload_segments,
            twist_total=0.0,
        )
        builder.add_rod(
            positions=points,
            quaternions=quats,
            radius=self.payload_radius,
            cfg=cable_cfg,
            stretch_stiffness=1.0e6,
            stretch_damping=1.0e-1,
            bend_stiffness=5.0e-3,
            bend_damping=2.0e-3,
            label="vbd_cable",
        )
        self.payload_bodies = list(range(payload_body_start, builder.body_count))
        self.payload_joints = list(range(payload_joint_start, builder.joint_count))
        self.payload_shapes = list(range(payload_shape_start, builder.shape_count))

        # Working surface (cable rests on it).
        plane_cfg = newton.ModelBuilder.ShapeConfig(ke=1.0e4, kd=1.0e-5, mu=1.0, margin=0.0, gap=0.01)
        self.ground_shapes = [
            builder.add_ground_plane(
                height=self.surface_z,
                cfg=plane_cfg,
                label="cable_ground_plane",
            )
        ]

        builder.color()
        self.model = builder.finalize()
        self.device = self.model.device

        # Uniform rigid contact material across all shapes (IsaacLab NewtonModelCfg).
        self.model.shape_material_ke.fill_(1.0e4)
        self.model.shape_material_kd.fill_(1.0e-5)
        self.model.shape_material_mu.fill_(1.0)

        # Gripper bodies exposed to VBD as proxies for cable contact.
        self.gripper_bodies = [
            b for b in self.franka_bodies if "hand" in self.model.body_label[b] or "finger" in self.model.body_label[b]
        ]
        if not self.gripper_bodies:
            raise RuntimeError("Could not locate Franka gripper bodies for proxy coupling")

        # Build keyframe sequence now that the cable pose is known.
        self._build_keyframes()

    # ------------------------------------------------------------------
    # Solvers
    # ------------------------------------------------------------------
    def _build_solvers(self, args):
        self.solver = SolverCoupledProxy(
            model=self.model,
            entries=[
                SolverCoupled.Entry(
                    name="mjc",
                    solver=lambda v: SolverMuJoCo(
                        model=v,
                        solver="newton",
                        integrator="implicitfast",
                        cone="elliptic",
                        iterations=int(args.mujoco_iterations),
                        ls_iterations=int(args.mujoco_ls_iterations),
                        use_mujoco_contacts=False,
                        njmax=256,
                        nconmax=64,
                    ),
                    bodies=self.franka_bodies,
                    joints=self.franka_joints,
                    shapes=self.franka_shapes,
                ),
                SolverCoupled.Entry(
                    name="vbd",
                    solver=lambda v: SolverVBD(
                        model=v,
                        iterations=int(args.vbd_iterations),
                        rigid_avbd_beta=float(args.vbd_rigid_avbd_beta),
                        rigid_contact_k_start=float(args.vbd_rigid_contact_k_start),
                        rigid_contact_history=False,
                    ),
                    bodies=self.payload_bodies,
                    joints=self.payload_joints,
                    shapes=self.payload_shapes + self.ground_shapes,
                ),
            ],
            coupling=SolverCoupledProxy.Config(
                proxies=[
                    SolverCoupledProxy.Proxy(
                        source="mjc",
                        destination="vbd",
                        bodies=self.gripper_bodies,
                        mass_scale=float(args.mass_scale),
                        mode=args.coupling_mode,
                        collision_pipeline=lambda model: newton.examples.create_collision_pipeline(
                            model, broad_phase="explicit"
                        ),
                        collide_interval=1,
                    ),
                ],
                iterations=int(args.proxy_iterations),
            ),
        )

    def _payload_ground_shape_pairs(self) -> wp.array:
        payload_shapes = set(self.payload_shapes)
        ground_shapes = set(self.ground_shapes)
        pairs = [
            (int(a), int(b))
            for a, b in self.model.shape_contact_pairs.numpy()
            if ({int(a), int(b)} & payload_shapes) and ({int(a), int(b)} & ground_shapes)
        ]
        if not pairs:
            raise RuntimeError("No cable-ground contact pairs were generated")
        return wp.array(np.asarray(pairs, dtype=np.int32), dtype=wp.vec2i, device=self.model.device)

    # ------------------------------------------------------------------
    # IK
    # ------------------------------------------------------------------
    def _build_ik(self):
        # IK runs on a standalone Franka-only model so the solver does not see the
        # cable's articulated bodies. The Franka is added first in the coupled model,
        # so its coords (0 .. n_coords) line up with this model's coords.
        ik_builder = newton.ModelBuilder(gravity=-9.81)
        self._add_franka(ik_builder, self.surface_z)
        self.ik_model = ik_builder.finalize()

        self.n_coords = self.ik_model.joint_coord_count
        self.ik_joint_q = wp.array(self.ik_model.joint_q, shape=(1, self.n_coords))
        self.finger_idx0 = self.n_coords - 2
        self.finger_idx1 = self.n_coords - 1
        self.finger_pos_buf = wp.full(1, GRIP_OPEN, dtype=float)
        hand_body = _find_label_index(self.ik_model.body_label, "fr3_hand")

        target_pos = wp.vec3(*self.targets[0][:3].tolist())
        target_rot = wp.vec4(*self.targets[0][3:7].tolist())

        # Tool offset reaches the TCP between the fingers (IsaacLab body_offset 0.107).
        self.pos_obj = ik.IKObjectivePosition(
            link_index=hand_body,
            link_offset=wp.vec3(0.0, 0.0, 0.107),
            target_positions=wp.array([target_pos], dtype=wp.vec3),
        )
        self.rot_obj = ik.IKObjectiveRotation(
            link_index=hand_body,
            link_offset_rotation=wp.quat_identity(),
            target_rotations=wp.array([target_rot], dtype=wp.vec4),
        )
        self.joint_limits_obj = ik.IKObjectiveJointLimit(
            joint_limit_lower=self.ik_model.joint_limit_lower,
            joint_limit_upper=self.ik_model.joint_limit_upper,
            weight=10.0,
        )
        self.ik_solver = ik.IKSolver(
            model=self.ik_model,
            n_problems=1,
            objectives=[self.pos_obj, self.rot_obj, self.joint_limits_obj],
            lambda_initial=0.05,
            jacobian_mode=ik.IKJacobianType.ANALYTIC,
        )
        self.ik_iters = 24

    # ------------------------------------------------------------------
    # Keyframes
    # ------------------------------------------------------------------
    def _build_keyframes(self):
        cx, cy, cz = CABLE_CENTER[0], CABLE_CENTER[1], CABLE_CENTER[2]
        approach_z = cz + 0.20
        # TCP target at the cable centerline so the fingers close around the thin
        # (radius 5 mm) cable rather than above it.
        grasp_z = cz
        tx, ty = 0.4, 0.25  # place target (x, y)

        qx, qy, qz, qw = GRIPPER_DOWN
        self.place_target_xy = (tx, ty)
        # [duration, px, py, pz, qx, qy, qz, qw, finger_width]
        poses = np.array(
            [
                [1.0, cx, cy, approach_z, qx, qy, qz, qw, GRIP_OPEN],  # approach above cable
                [0.5, cx, cy, grasp_z, qx, qy, qz, qw, GRIP_OPEN],  # descend to cable
                [1.0, cx, cy, grasp_z, qx, qy, qz, qw, GRIP_CLOSE],  # grasp (capture)
                [1.0, cx, cy, approach_z, qx, qy, qz, qw, GRIP_HOLD],  # lift (hold pinch)
                [1.0, tx, ty, approach_z, qx, qy, qz, qw, GRIP_HOLD],  # move to target
                [0.5, tx, ty, grasp_z, qx, qy, qz, qw, GRIP_HOLD],  # lower
                [0.5, tx, ty, grasp_z, qx, qy, qz, qw, GRIP_HOLD],  # settle (cable comes to rest at target)
                [1.0, tx, ty, grasp_z, qx, qy, qz, qw, GRIP_OPEN],  # release
                [0.5, tx, ty, approach_z, qx, qy, qz, qw, GRIP_OPEN],  # retract
            ],
            dtype=np.float32,
        )
        self.targets = poses[:, 1:]
        self.key_times = np.cumsum(poses[:, 0])

    def update_ik_targets(self):
        """Interpolate keyframes and update IK target arrays (CPU, before graph launch)."""
        t = min(self.sim_time, float(self.key_times[-1]) - 1e-6)
        interval = int(np.searchsorted(self.key_times, t))
        t_start = self.key_times[interval - 1] if interval > 0 else 0.0
        t_end = self.key_times[interval]
        alpha = float(np.clip((t - t_start) / max(t_end - t_start, 1e-6), 0.0, 1.0))

        cur = self.targets[interval]
        prev = self.targets[interval - 1] if interval > 0 else cur
        interp = (1.0 - alpha) * prev + alpha * cur

        self.pos_obj.set_target_position(0, wp.vec3(*interp[:3].tolist()))
        self.rot_obj.set_target_rotation(0, wp.vec4(*interp[3:7].tolist()))
        self.finger_pos_buf.fill_(float(interp[-1]))

    # ------------------------------------------------------------------
    # Simulation
    # ------------------------------------------------------------------
    def capture(self):
        self.graph = None
        if self.use_graph and self.device.is_cuda:
            with wp.ScopedDevice(self.device), wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph

    def simulate(self):
        # GPU IK solve, then write solved arm joints + gripper width to PD targets.
        self.ik_solver.step(self.ik_joint_q, self.ik_joint_q, iterations=self.ik_iters)
        wp.launch(
            set_gripper_q,
            dim=1,
            inputs=[self.ik_joint_q, self.finger_pos_buf, self.finger_idx0, self.finger_idx1],
        )
        wp.copy(self.control.joint_target_q, self.ik_joint_q, count=self.n_coords)

        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            newton.examples.apply_coupled_viewer_forces(self, self.state_0)
            self.model.collide(self.state_0, self.contacts, collision_pipeline=self.collision_pipeline)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            newton.eval_ik(self.model, self.state_1, self.state_1.joint_q, self.state_1.joint_qd)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        self.update_ik_targets()
        if self.graph is not None:
            with wp.ScopedDevice(self.device):
                wp.capture_launch(self.graph)
        else:
            self.simulate()
        self.sim_time += self.frame_dt

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        newton.examples.log_coupled_view(self, self.contacts)
        self.viewer.end_frame()

    def test_final(self):
        body_q = self.state_0.body_q.numpy()
        body_qd = self.state_0.body_qd.numpy()
        assert np.all(np.isfinite(body_q)), "Body positions contain NaN or inf values"
        assert np.all(np.isfinite(body_qd)), "Body velocities contain NaN or inf values"

        # The cable is grasped at its midpoint and carried to the place target; verify that
        # grasped segment was placed within 1 cm of the target (x, y).
        mid_body = self.payload_bodies[len(self.payload_bodies) // 2]
        dist = float(np.linalg.norm(body_q[mid_body, :2] - np.asarray(self.place_target_xy, dtype=np.float32)))
        assert dist < 0.01, f"Cable placed {dist * 100:.1f} cm from target (expected < 1 cm)"

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        newton.examples.add_coupled_view_args(parser)
        parser.add_argument("--substeps", type=int, default=10, help="Coupled substeps per rendered frame.")
        parser.add_argument("--proxy-iterations", type=int, default=4, help="Proxy relaxation passes per substep.")
        parser.add_argument("--mass-scale", type=float, default=1.0, help="Proxy body mass scale in VBD.")
        parser.add_argument(
            "--coupling-mode",
            type=str,
            choices=["lagged", "staggered"],
            default="lagged",
            help="Proxy transfer mode.",
        )
        parser.add_argument("--payload-segments", type=int, default=19, help="Number of cable segments.")
        parser.add_argument("--payload-radius", type=float, default=0.005, help="Cable radius [m].")
        parser.add_argument("--vbd-iterations", type=int, default=20, help="VBD iterations per coupled substep.")
        parser.add_argument(
            "--vbd-rigid-avbd-beta",
            type=float,
            default=1.0e2,
            help="VBD AVBD penalty ramp rate per iteration (0 disables ramping).",
        )
        parser.add_argument(
            "--vbd-rigid-contact-k-start",
            type=float,
            default=1.0e2,
            help="VBD body-particle contact penalty seed when AVBD ramping is enabled.",
        )
        parser.add_argument("--mujoco-iterations", type=int, default=100, help="MuJoCo solver iterations.")
        parser.add_argument("--mujoco-ls-iterations", type=int, default=20, help="MuJoCo line-search iterations.")
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
    parser.set_defaults(num_frames=400)
    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args)
    newton.examples.run(example, args)
