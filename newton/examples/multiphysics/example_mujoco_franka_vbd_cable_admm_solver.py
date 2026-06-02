# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example MuJoCo Franka + Rigid Chain ADMM Coupled Solver
#
# A fixed-base Franka arm is simulated by MuJoCo while a short rigid payload
# chain is simulated by XPBD by default. The original VBD cable payload is kept
# as an alternate mode for A/B testing. SolverCoupledADMM detects rigid-rigid
# contacts between the robot and the payload from the model collision pairs,
# and the same template is replicated across many worlds to exercise ADMM
# contact scaling.
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
import newton.utils
from newton.solvers import SolverMuJoCo, SolverVBD, SolverXPBD

FRANKA_Q = [
    -3.6802115e-03,
    2.3901723e-02,
    3.6804110e-03,
    -2.3683236e00,
    -1.2918962e-04,
    2.3922248e00,
    7.8549200e-01,
    0.03,
    0.03,
]

PAYLOAD_CENTER = wp.vec3(0.5, 0.0, 0.256)
PAYLOAD_LENGTH = 0.32


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

        template = newton.ModelBuilder(gravity=-9.81)
        template.rigid_gap = 0.002
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
                    shapes=self.franka_shapes,
                ),
                SolverCoupled.Entry(
                    name=payload_name,
                    solver=payload_solver,
                    bodies=self.payload_bodies,
                    joints=self.payload_joints,
                    shapes=self.payload_shapes + self.ground_shapes,
                ),
            ],
            coupling=SolverCoupledADMM.Config(
                iterations=int(args.admm_iterations),
                rho=float(args.rho),
                gamma=float(args.gamma),
                baumgarte=float(args.baumgarte),
                rigid_contact_matching=str(args.rigid_contact_matching),
                contact_pairs=[
                    SolverCoupledADMM.ContactPair(
                        source="mjc",
                        destination=payload_name,
                        contact_distance=None,
                    ),
                ],
            ),
        )

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

        builder.add_urdf(
            newton.utils.download_asset("franka_emika_panda") / "urdf/fr3_franka_hand.urdf",
            floating=False,
            enable_self_collisions=False,
            parse_visuals_as_colliders=False,
            force_show_colliders=False,
        )
        builder.joint_q[: len(FRANKA_Q)] = FRANKA_Q
        builder.joint_target_q[: len(FRANKA_Q)] = FRANKA_Q
        builder.joint_target_ke[:7] = [900.0] * 7
        builder.joint_target_kd[:7] = [90.0] * 7
        builder.joint_target_ke[7:9] = [1500.0, 1500.0]
        builder.joint_target_kd[7:9] = [30.0, 30.0]
        builder.joint_effort_limit[:7] = [80.0] * 7
        builder.joint_effort_limit[7:9] = [40.0, 40.0]
        builder.joint_armature[:7] = [0.05] * 7
        builder.joint_armature[7:9] = [0.02, 0.02]

        pad_cfg = newton.ModelBuilder.ShapeConfig(ke=8.0e4, kd=2.0e1, mu=1.2, margin=0.001, gap=0.002)
        for suffix, color in (
            ("fr3_leftfinger", (0.92, 0.48, 0.18)),
            ("fr3_rightfinger", (0.92, 0.48, 0.18)),
        ):
            finger = _find_label_index(builder.body_label, suffix)
            builder.add_shape_sphere(
                finger,
                radius=0.022,
                cfg=pad_cfg,
                color=color,
                label=f"{suffix}_admm_contact_pad",
            )

        franka_body_end = builder.body_count
        franka_joint_end = builder.joint_count
        franka_shape_end = builder.shape_count

        payload_shape_start = builder.shape_count
        if self.payload_kind == "vbd-cable":
            payload_bodies, payload_joints = self._emit_vbd_cable(builder)
        else:
            payload_bodies, payload_joints = self._emit_xpbd_chain(builder)

        self.franka_bodies = list(range(franka_body_start, franka_body_end))
        self.franka_joints = list(range(franka_joint_start, franka_joint_end))
        self.franka_shapes = list(range(franka_shape_start, franka_shape_end))
        self.payload_bodies = payload_bodies
        self.payload_joints = payload_joints
        self.payload_shapes = list(range(payload_shape_start, builder.shape_count))

    def _emit_ground_plane(self, builder: newton.ModelBuilder) -> int:
        top_z = 0.256 - self.payload_radius
        plane_cfg = newton.ModelBuilder.ShapeConfig(ke=8.0e4, kd=2.0e1, mu=0.8, margin=0.001, gap=0.002)
        return builder.add_ground_plane(
            height=top_z,
            cfg=plane_cfg,
            label="payload_ground_plane",
        )

    def _emit_vbd_cable(self, builder: newton.ModelBuilder) -> tuple[list[int], list[int]]:
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
            cfg=cable_cfg,
            stretch_stiffness=2.0e5,
            stretch_damping=2.0e-2,
            bend_stiffness=0.08,
            bend_damping=2.0e-2,
            label="vbd_cable",
        )

    def _emit_xpbd_chain(self, builder: newton.ModelBuilder) -> tuple[list[int], list[int]]:
        chain_length = PAYLOAD_LENGTH
        segment_length = chain_length / float(self.payload_segments)
        segment_half_length = 0.5 * segment_length
        capsule_half_height = max(0.25 * self.payload_radius, segment_half_length - self.payload_radius)
        start = PAYLOAD_CENTER - wp.vec3(0.5 * PAYLOAD_LENGTH, 0.0, 0.0)
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

    def capture(self):
        self.graph = _capture_frame_graph(self.model, self.simulate, enabled=self.use_graph)

    def simulate(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            newton.examples.apply_coupled_viewer_forces(self, self.state_0)
            self.model.collide(self.state_0, self.contacts, collision_pipeline=self.collision_pipeline)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            newton.eval_ik(self.model, self.state_1, self.state_1.joint_q, self.state_1.joint_qd)
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
        assert np.all(self.admm_shape_pairs_per_world > 0), "Each world should have Franka-payload ADMM contact pairs"
        assert np.all(self.admm_shape_pairs_per_world == self.admm_shape_pairs_per_world[0]), (
            "Franka-payload ADMM contact pair counts should be identical across replicated worlds"
        )
        if self.use_graph and self.device.is_cuda:
            assert self.graph is not None, "CUDA graph capture was requested but no graph was captured"

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
        parser.add_argument("--gamma", type=float, default=0.1, help="ADMM proximal mass scaling.")
        parser.add_argument("--baumgarte", type=float, default=0.02, help="Position error correction fraction.")
        parser.add_argument(
            "--rigid-contact-matching",
            choices=["disabled", "latest", "sticky"],
            default="disabled",
            help="ADMM Franka-payload rigid contact matching mode.",
        )
        parser.add_argument(
            "--payload-kind",
            choices=["xpbd-chain", "vbd-cable"],
            default="xpbd-chain",
            help="Payload simulated by the non-MuJoCo solver.",
        )
        parser.add_argument("--payload-segments", type=int, default=8, help="Number of payload rigid/cable segments.")
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
            help="Disable CUDA graph capture.",
        )
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args)
    newton.examples.run(example, args)
