# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Gear SDF Contact
#
# Demonstrates mesh-mesh gear contact using SDF (Signed Distance Field).
#
# Command: python -m newton.examples gear_sdf
#
###########################################################################

import argparse
import tempfile
from dataclasses import replace
from pathlib import Path

import numpy as np
import trimesh
import warp as wp

import newton
import newton.examples
from newton.geometry import HydroelasticSDF

ISAACGYM_ENVS_REPO_URL = "https://github.com/isaac-sim/IsaacGymEnvs.git"
ISAACGYM_GEARS_FOLDER = "assets/factory/mesh/factory_gears"

GEAR_FILES = [
    ("factory_gear_base_loose_space_5e-4_subdiv_4x.obj", "gear_base"),
    ("factory_gear_large_space_5e-4.obj", "gear_large"),
    ("factory_gear_medium_space_5e-4.obj", "gear_medium"),
    ("factory_gear_small_space_5e-4.obj", "gear_small"),
]

SHAPE_CFG = newton.ModelBuilder.ShapeConfig(
    margin=0.0,
    mu=0.01,
    ke=1e7,
    kd=1e4,
    gap=0.005,
    density=8000.0,
    mu_torsional=0.0,
    mu_rolling=0.0,
    is_hydroelastic=False,
)
MESH_SDF_NARROW_BAND_RANGE = (-0.005, 0.005)
MESH_SDF_CACHE_DIR = Path(tempfile.gettempdir()) / "newton_sdf_cache"


def load_mesh_with_sdf(
    mesh_file: str,
    shape_cfg: newton.ModelBuilder.ShapeConfig | None = None,
    center_origin: bool = True,
    sdf_resolution: int = 512,
    contact_model: str = "sdf",
    scale: float = 1.0,
) -> tuple[newton.Mesh, wp.vec3]:
    mesh_data = trimesh.load(mesh_file, force="mesh")
    vertices = np.array(mesh_data.vertices, dtype=np.float32)
    indices = np.array(mesh_data.faces.flatten(), dtype=np.int32)
    center_vec = wp.vec3(0.0, 0.0, 0.0)

    if center_origin:
        min_extent = vertices.min(axis=0)
        max_extent = vertices.max(axis=0)
        center = (min_extent + max_extent) / 2.0
        vertices = vertices - center
        center_vec = wp.vec3(center)

    mesh = newton.Mesh(vertices, indices)
    sdf_kwargs = {}
    if contact_model == "hydro":
        sdf_kwargs["scale"] = (scale, scale, scale)

    mesh.build_sdf(
        max_resolution=sdf_resolution,
        narrow_band_range=MESH_SDF_NARROW_BAND_RANGE,
        margin=shape_cfg.gap if shape_cfg and shape_cfg.gap is not None else 0.05,
        cache_dir=MESH_SDF_CACHE_DIR,
        **sdf_kwargs,
    )
    return mesh, center_vec


def add_gear_mesh(
    builder: newton.ModelBuilder,
    mesh: newton.Mesh,
    transform: wp.transform,
    shape_cfg: newton.ModelBuilder.ShapeConfig,
    label: str,
    center_vec: wp.vec3,
    scale: float,
) -> int:
    center_world = wp.quat_rotate(transform.q, center_vec * scale)
    transform = wp.transform(transform.p + center_world, transform.q)

    if label == "gear_base":
        builder.add_shape_mesh(
            -1,
            mesh=mesh,
            scale=(scale, scale, scale),
            xform=transform,
            cfg=shape_cfg,
            label=label,
        )
        return -1

    body = builder.add_body(label=label, xform=transform)
    builder.add_shape_mesh(body, mesh=mesh, scale=(scale, scale, scale), cfg=shape_cfg)
    return body


class Example:
    def __init__(self, viewer, args):
        self.fps = 120
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = args.substeps
        self.sim_dt = self.frame_dt / self.sim_substeps

        self.viewer = viewer
        self.world_count = args.world_count
        self.solver_type = args.solver
        self.scene_scale = args.scene_scale
        self.drive_torque = args.drive_torque
        self.collide_every_substep = args.collide_every_substep
        self.test_mode = args.test
        self.contact_model = args.contact_model

        self.rigid_contact_max = args.contact_capacity * self.world_count
        self.broad_phase = args.broad_phase
        self.xpbd_contact_relaxation = args.xpbd_contact_relaxation
        self.shape_cfg = replace(
            SHAPE_CFG,
            margin=args.margin,
            gap=args.gap,
            is_hydroelastic=args.contact_model == "hydro",
        )

        world_builder = self._build_gears_scene(args.sdf_resolution)

        main_scene = newton.ModelBuilder()
        main_scene.default_shape_cfg.gap = args.gap
        main_scene.add_shape_plane(
            plane=(0.0, 0.0, 1.0, 0.01 * self.scene_scale),
            width=0.0,
            length=0.0,
            label="ground_plane",
        )
        main_scene.replicate(world_builder, world_count=self.world_count)

        self.model = main_scene.finalize()
        self.model.rigid_contact_max = self.rigid_contact_max

        hydroelastic_config = None
        if self.contact_model == "hydro":
            hydroelastic_config = HydroelasticSDF.Config(
                reduce_contacts=args.hydro_reduce_contacts,
                buffer_fraction=1.0,
                contact_buffer_fraction=args.hydro_contact_buffer_fraction,
                mc_edge_clamp_min=args.hydro_edge_clamp,
            )

        self.collision_pipeline = newton.CollisionPipeline(
            self.model,
            reduce_contacts=args.reduce_contacts,
            rigid_contact_max=self.rigid_contact_max,
            broad_phase=self.broad_phase,
            sdf_hydroelastic_config=hydroelastic_config,
        )

        if self.solver_type == "xpbd":
            self.solver = newton.solvers.SolverXPBD(
                self.model,
                iterations=args.xpbd_iterations,
                rigid_contact_relaxation=self.xpbd_contact_relaxation,
            )
        elif self.solver_type == "mujoco":
            num_per_world = self.collision_pipeline.rigid_contact_max // self.world_count
            self.solver = newton.solvers.SolverMuJoCo(
                self.model,
                use_mujoco_contacts=False,
                solver="newton",
                integrator="implicitfast",
                cone="elliptic",
                njmax=num_per_world,
                nconmax=num_per_world,
                iterations=args.mujoco_iterations,
                ls_iterations=args.mujoco_ls_iterations,
                impratio=1.0,
            )
        else:
            raise ValueError(f"Unknown solver type: {self.solver_type}. Choose from 'xpbd' or 'mujoco'.")

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()

        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)
        self._apply_drive_torque()

        self.contacts = self.collision_pipeline.contacts()

        self.viewer.set_model(self.model)
        offset = 0.25 * self.scene_scale
        self.viewer.set_world_offsets((offset, offset, 0.0))
        self.viewer.set_camera(pos=wp.vec3(0.0, -0.34, 0.65), pitch=-57.0, yaw=98.0)

        self._init_test_tracking()
        self.capture()

    def _build_gears_scene(self, sdf_resolution: int) -> newton.ModelBuilder:
        print("Downloading gear assets...")
        asset_path = newton.examples.download_external_git_folder(ISAACGYM_ENVS_REPO_URL, ISAACGYM_GEARS_FOLDER)
        print(f"Assets downloaded to: {asset_path}")

        world_builder = newton.ModelBuilder()
        world_builder.default_shape_cfg.gap = self.shape_cfg.gap

        gear_xform = wp.transform(wp.vec3(0.0, 0.0, 0.01) * self.scene_scale, wp.quat_identity())
        for gear_filename, gear_label in GEAR_FILES:
            gear_file = str(asset_path / gear_filename)
            gear_mesh, gear_center = load_mesh_with_sdf(
                gear_file,
                shape_cfg=self.shape_cfg,
                center_origin=True,
                sdf_resolution=sdf_resolution,
                contact_model=self.contact_model,
                scale=self.scene_scale,
            )
            add_gear_mesh(
                world_builder,
                gear_mesh,
                gear_xform,
                self.shape_cfg,
                label=gear_label,
                center_vec=gear_center,
                scale=self.scene_scale,
            )

        return world_builder

    def _find_body(self, label: str) -> int | None:
        for body_idx, body_label in enumerate(self.model.body_label):
            if body_label == label or body_label.endswith(f"/{label}"):
                return body_idx
        return None

    def _apply_drive_torque(self):
        drive_body = self._find_body("gear_large")
        if drive_body is None:
            return

        joint_child = self.model.joint_child.numpy()
        joint_qd_start = self.model.joint_qd_start.numpy()
        joint_f = self.control.joint_f.numpy()
        for joint_idx in range(self.model.joint_count):
            if joint_child[joint_idx] == drive_body:
                qd_start = int(joint_qd_start[joint_idx])
                joint_f[qd_start + 5] = self.drive_torque
                break
        self.control.joint_f.assign(joint_f)

    def capture(self):
        if wp.get_device().is_cuda:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph
        else:
            self.graph = None

    def simulate(self):
        if not self.collide_every_substep:
            self.collision_pipeline.collide(self.state_0, self.contacts)

        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            if self.collide_every_substep:
                self.collision_pipeline.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()

        self.sim_time += self.frame_dt
        self._track_test_data()

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()

    def _init_test_tracking(self):
        if not self.test_mode:
            self.drive_body = None
            return

        self.drive_body = self._find_body("gear_large")
        if self.drive_body is None:
            raise ValueError("Unable to find gear_large body for test tracking")

        self.dynamic_bodies = [
            body_idx
            for body_idx, label in enumerate(self.model.body_label)
            if any(label == gear_label or label.endswith(f"/{gear_label}") for _, gear_label in GEAR_FILES[1:])
        ]
        body_q = self.state_0.body_q.numpy()
        self.initial_body_q = {body_idx: body_q[body_idx].copy() for body_idx in self.dynamic_bodies}
        self.drive_max_rotation = 0.0
        self.max_displacement = 0.0

    def _track_test_data(self):
        if not self.test_mode:
            return

        body_q = self.state_0.body_q.numpy()
        initial_q = self.initial_body_q[self.drive_body]
        q_current = body_q[self.drive_body][3:7]
        q_initial = initial_q[3:7]
        dot = min(abs(float(np.dot(q_current, q_initial))), 1.0)
        self.drive_max_rotation = max(self.drive_max_rotation, 2.0 * np.arccos(dot))

        for body_idx in self.dynamic_bodies:
            displacement = np.linalg.norm(body_q[body_idx][:3] - self.initial_body_q[body_idx][:3])
            self.max_displacement = max(self.max_displacement, float(displacement))

    def test_final(self):
        if not self.test_mode:
            return

        assert self.drive_max_rotation > 1.0e-4, "Driven gear did not rotate"
        max_allowed_displacement = 5.0 * self.scene_scale
        assert self.max_displacement < max_allowed_displacement, (
            f"Gear scene became unstable. Max displacement={self.max_displacement:.4f}, "
            f"max allowed={max_allowed_displacement:.4f}"
        )

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        newton.examples.add_world_count_arg(parser)
        newton.examples.add_broad_phase_arg(parser)
        parser.set_defaults(world_count=1, broad_phase="sap")
        parser.add_argument(
            "--solver",
            type=str,
            choices=["xpbd", "mujoco"],
            default="xpbd",
            help="Solver to use.",
        )
        parser.add_argument(
            "--contact-model",
            type=str,
            choices=["sdf", "hydro"],
            default="sdf",
            help="Contact generation model to use.",
        )
        parser.add_argument(
            "--substeps",
            type=int,
            default=10,
            help="Simulation substeps per rendered frame.",
        )
        parser.add_argument(
            "--scene-scale",
            type=float,
            default=5.0,
            help="Uniform gear scene scale.",
        )
        parser.add_argument(
            "--drive-torque",
            type=float,
            default=2.0,
            help="Torque [N*m] applied to the large gear's free joint z-axis.",
        )
        parser.add_argument(
            "--contact-capacity",
            type=int,
            default=100000,
            help="Rigid contact capacity per world.",
        )
        parser.add_argument(
            "--gap",
            type=float,
            default=SHAPE_CFG.gap,
            help="Per-shape contact gap [m].",
        )
        parser.add_argument(
            "--margin",
            type=float,
            default=SHAPE_CFG.margin,
            help="Per-shape contact margin [m].",
        )
        parser.add_argument(
            "--reduce-contacts",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="Reduce mesh contact manifolds before solving.",
        )
        parser.add_argument(
            "--sdf-resolution",
            type=int,
            default=512,
            help="Maximum SDF resolution for each gear mesh.",
        )
        parser.add_argument(
            "--collide-every-substep",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="Refresh SDF contacts every substep.",
        )
        parser.add_argument(
            "--xpbd-iterations",
            type=int,
            default=10,
            help="XPBD solver iterations.",
        )
        parser.add_argument(
            "--xpbd-contact-relaxation",
            type=float,
            default=0.8,
            help="XPBD rigid contact relaxation.",
        )
        parser.add_argument(
            "--mujoco-iterations",
            type=int,
            default=15,
            help="MuJoCo-Warp solver iterations.",
        )
        parser.add_argument(
            "--mujoco-ls-iterations",
            type=int,
            default=100,
            help="MuJoCo-Warp line-search iterations.",
        )
        parser.add_argument(
            "--hydro-reduce-contacts",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="Reduce hydroelastic contact patches before solving.",
        )
        parser.add_argument(
            "--hydro-contact-buffer-fraction",
            type=float,
            default=1.0,
            help="Hydroelastic face-contact buffer fraction.",
        )
        parser.add_argument(
            "--hydro-edge-clamp",
            type=float,
            default=0.0,
            help="Hydroelastic marching-cubes edge clamp minimum.",
        )
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()

    viewer, args = newton.examples.init(parser)

    newton.examples.run(Example(viewer, args), args)
