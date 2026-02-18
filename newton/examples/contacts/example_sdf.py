# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

###########################################################################
# Example SDF Mesh Collision
#
# Demonstrates mesh-mesh collision using SDF (Signed Distance Field).
# Supports two scenes "nut_bolt" and "gears":
#
# Command: python -m newton.examples sdf --scene nut_bolt
#          python -m newton.examples sdf --scene gears
#
###########################################################################

import numpy as np
import trimesh
import warp as wp

import newton
import newton.examples
from newton._src.utils.download_assets import download_git_folder

# Assembly type for the nut and bolt
ASSEMBLY_STR = "m20_loose"

# Gear mesh files available (filename -> key)
GEAR_FILES = [
    ("factory_gear_base_loose_space_5e-4_subdiv_4x.obj", "gear_base"),
    ("factory_gear_large_space_5e-4.obj", "gear_large"),
    ("factory_gear_medium_space_5e-4.obj", "gear_medium"),
    ("factory_gear_small_space_5e-4.obj", "gear_small"),
]

SHAPE_CFG = newton.ModelBuilder.ShapeConfig(
    thickness=0.0,
    mu=0.01,
    ke=1e7,  # Contact stiffness for MuJoCo solver
    kd=1e4,  # Contact damping
    sdf_max_resolution=512,
    sdf_narrow_band_range=(-0.005, 0.005),
    contact_margin=0.005,
    density=8000.0,
    mu_torsional=0.0,
    mu_rolling=0.0,
    is_hydroelastic=False,
)


def add_mesh_object(
    builder: newton.ModelBuilder,
    mesh_file: str,
    transform: wp.transform,
    shape_cfg: newton.ModelBuilder.ShapeConfig | None = None,
    label: str | None = None,
    center_origin: bool = True,
    scale: float = 1.0,
) -> int:
    mesh_data = trimesh.load(mesh_file, force="mesh")
    vertices = np.array(mesh_data.vertices, dtype=np.float32)
    indices = np.array(mesh_data.faces.flatten(), dtype=np.int32)

    if center_origin:
        min_extent = vertices.min(axis=0)
        max_extent = vertices.max(axis=0)
        center = (min_extent + max_extent) / 2
        vertices = vertices - center
        center_vec = wp.vec3(center) * float(scale)
        center_world = wp.quat_rotate(transform.q, center_vec)
        transform = wp.transform(transform.p + center_world, transform.q)

    mesh = newton.Mesh(vertices, indices)

    if label == "gear_base":
        body = -1
        builder.add_shape_mesh(
            body, mesh=mesh, scale=(scale, scale, scale), xform=transform, cfg=shape_cfg, label=label
        )
    else:
        body = builder.add_body(label=label, xform=transform)
        builder.add_shape_mesh(body, mesh=mesh, scale=(scale, scale, scale), cfg=shape_cfg)
    return body


class Example:
    def __init__(self, viewer, world_count=1, num_per_world=1, scene="nut_bolt", solver="xpbd", test_mode=False):
        self.fps = 120
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        # Use more substeps for gears scene to improve stability
        self.sim_substeps = 50 if scene == "gears" else 5
        self.sim_dt = self.frame_dt / self.sim_substeps

        self.world_count = world_count
        self.viewer = viewer
        self.scene = scene
        self.solver_type = solver
        self.test_mode = test_mode

        # XPBD contact correction (0.0 = no correction, 1.0 = full correction)
        self.xpbd_contact_relaxation = 0.8

        # Scene scaling factor (1.0 = original size)
        self.scene_scale = 5.0

        # Ground plane offset (negative = below origin)
        self.ground_plane_offset = -0.01

        # Grid dimensions for nut/bolt scene (number of assemblies in X and Y)
        self.num_per_world = num_per_world
        self.grid_x = int(np.ceil(np.sqrt(num_per_world)))
        self.grid_y = int(np.ceil(num_per_world / self.grid_x))

        # Maximum number of rigid contacts to allocate (limits memory usage)
        # None = auto-calculate (can be very large), or set explicit limit (e.g., 1_000_000)
        self.rigid_contact_max = 100000

        # Broad phase mode: NXN (O(N²)), SAP (O(N log N)), EXPLICIT (precomputed pairs)
        self.broad_phase = "sap"

        if scene == "nut_bolt":
            world_builder = self._build_nut_bolt_scene()
        elif scene == "gears":
            world_builder = self._build_gears_scene()
        else:
            raise ValueError(f"Unknown scene: {scene}")

        main_scene = newton.ModelBuilder()
        main_scene.default_shape_cfg.contact_margin = 0.01
        # Add ground plane with offset (plane equation: z = offset)
        # For plane equation n·x + d = 0, with n=(0,0,1): z + d = 0, so z = -d
        # Therefore, to get plane at z = offset, we need d = -offset
        main_scene.add_shape_plane(
            plane=(0.0, 0.0, 1.0, -self.ground_plane_offset),
            width=0.0,
            length=0.0,
            label="ground_plane",
        )
        main_scene.replicate(world_builder, world_count=self.world_count)

        self.model = main_scene.finalize()

        # Override rigid_contact_max BEFORE creating collision pipeline to limit memory allocation
        self.model.rigid_contact_max = self.rigid_contact_max

        self.collision_pipeline = newton.CollisionPipeline(
            self.model,
            reduce_contacts=True,
            broad_phase=self.broad_phase,
        )

        # Create solver based on user choice
        if self.solver_type == "xpbd":
            self.solver = newton.solvers.SolverXPBD(
                self.model,
                iterations=10,
                rigid_contact_relaxation=self.xpbd_contact_relaxation,
            )
        elif self.solver_type == "mujoco":
            num_per_world = self.rigid_contact_max // self.world_count
            self.solver = newton.solvers.SolverMuJoCo(
                self.model,
                use_mujoco_contacts=False,
                solver="newton",
                integrator="implicitfast",
                cone="elliptic",
                njmax=num_per_world,
                nconmax=num_per_world,
                iterations=15,
                ls_iterations=100,
                impratio=1.0,
            )
        else:
            raise ValueError(f"Unknown solver type: {self.solver_type}. Choose from 'xpbd' or 'mujoco'.")

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()

        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)

        if self.scene == "gears":
            joint_child = self.model.joint_child.numpy()
            joint_qd_start = self.model.joint_qd_start.numpy()
            joint_f = self.control.joint_f.numpy()
            for body_idx, lbl in enumerate(self.model.body_label):
                if lbl.endswith("/gear_large") or lbl == "gear_large":
                    for j in range(self.model.joint_count):
                        if joint_child[j] == body_idx:
                            qd_start = int(joint_qd_start[j])
                            joint_f[qd_start + 5] = 2.0  # z-axis torque (N·m)
                            break
            self.control.joint_f.assign(joint_f)

        self.contacts = self.collision_pipeline.contacts()

        self.viewer.set_model(self.model)

        if scene == "nut_bolt":
            offset = 0.15 * self.scene_scale
            self.viewer.set_world_offsets((offset, offset, 0.0))
            self.viewer.set_camera(pos=wp.vec3(offset, -offset, 0.12 * self.scene_scale), pitch=-15.0, yaw=135.0)
        else:  # gears
            offset = 0.25 * self.scene_scale
            self.viewer.set_world_offsets((offset, offset, 0.0))
            self.viewer.set_camera(pos=wp.vec3(offset, -offset, 0.2 * self.scene_scale), pitch=-25.0, yaw=135.0)

        # Initialize test tracking data (only in test mode for nut_bolt scene)
        self._init_test_tracking()

        self.capture()

    def _build_nut_bolt_scene(self) -> newton.ModelBuilder:
        repo_url = "https://github.com/isaac-sim/IsaacGymEnvs.git"
        print(f"Downloading nut/bolt assets from {repo_url}...")
        asset_path = download_git_folder(repo_url, "assets/factory/mesh/factory_nut_bolt")
        print(f"Assets downloaded to: {asset_path}")

        world_builder = newton.ModelBuilder()
        world_builder.default_shape_cfg.contact_margin = 0.01 * self.scene_scale

        bolt_file = str(asset_path / f"factory_bolt_{ASSEMBLY_STR}.obj")
        nut_file = str(asset_path / f"factory_nut_{ASSEMBLY_STR}_subdiv_3x.obj")

        # Spacing between assemblies in the grid
        spacing = 0.1 * self.scene_scale

        # Create grid of nut/bolt assemblies
        count = 0
        for i in range(self.grid_x):
            if count >= self.num_per_world:
                break
            for j in range(self.grid_y):
                if count >= self.num_per_world:
                    break
                # Center the grid around origin
                x_offset = (i - (self.grid_x - 1) / 2.0) * spacing
                y_offset = (j - (self.grid_y - 1) / 2.0) * spacing

                # Add bolt at grid position
                bolt_xform = wp.transform(wp.vec3(x_offset, y_offset, 0.0 * self.scene_scale), wp.quat_identity())
                add_mesh_object(
                    world_builder,
                    bolt_file,
                    bolt_xform,
                    SHAPE_CFG,
                    label=f"bolt_{i}_{j}",
                    center_origin=True,
                    scale=self.scene_scale,
                )

                # Add nut above bolt at grid position
                nut_xform = wp.transform(
                    wp.vec3(x_offset, y_offset, 0.041 * self.scene_scale),
                    wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), np.pi / 8),
                )
                add_mesh_object(
                    world_builder,
                    nut_file,
                    nut_xform,
                    SHAPE_CFG,
                    label=f"nut_{i}_{j}",
                    center_origin=True,
                    scale=self.scene_scale,
                )
                count += 1

        return world_builder

    def _build_gears_scene(self) -> newton.ModelBuilder:
        repo_url = "https://github.com/isaac-sim/IsaacGymEnvs.git"
        print(f"Downloading gear assets from {repo_url}...")
        asset_path = download_git_folder(repo_url, "assets/factory/mesh/factory_gears")
        print(f"Assets downloaded to: {asset_path}")

        world_builder = newton.ModelBuilder()
        world_builder.default_shape_cfg.contact_margin = 0.003 * self.scene_scale

        for _, (gear_filename, gear_key) in enumerate(GEAR_FILES):
            gear_file = str(asset_path / gear_filename)
            gear_xform = wp.transform(wp.vec3(0.0, 0.0, 0.01) * self.scene_scale, wp.quat_identity())
            add_mesh_object(
                world_builder,
                gear_file,
                gear_xform,
                SHAPE_CFG,
                label=gear_key,
                center_origin=True,
                scale=self.scene_scale,
            )

        return world_builder

    def capture(self):
        if wp.get_device().is_cuda:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph
        else:
            self.graph = None

    def simulate(self):
        self.collision_pipeline.collide(self.state_0, self.contacts)
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()

            self.viewer.apply_forces(self.state_0)
            # self.collision_pipeline.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)

            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()

        self.sim_time += self.frame_dt

        # Track transforms for test validation
        self._track_test_data()

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()

    def _init_test_tracking(self):
        """Initialize tracking data for test validation (nut_bolt scene only)."""
        if not self.test_mode or self.scene != "nut_bolt":
            self.bolt_body_indices = None
            self.nut_body_indices = None
            return

        # Find bolt and nut body indices by key
        self.bolt_body_indices = []
        self.nut_body_indices = []

        for i in range(self.grid_x):
            for j in range(self.grid_y):
                bolt_key = f"bolt_{i}_{j}"
                nut_key = f"nut_{i}_{j}"

                if bolt_key in self.model.body_label:
                    self.bolt_body_indices.append(self.model.body_label.index(bolt_key))
                if nut_key in self.model.body_label:
                    self.nut_body_indices.append(self.model.body_label.index(nut_key))

        # Store initial transforms
        body_q = self.state_0.body_q.numpy()
        self.bolt_initial_transforms = [body_q[idx].copy() for idx in self.bolt_body_indices]
        self.nut_initial_transforms = [body_q[idx].copy() for idx in self.nut_body_indices]

        # Track maximum rotation change and z displacement for nuts
        self.nut_max_rotation_change = [0.0] * len(self.nut_body_indices)
        self.nut_min_z = [body_q[idx][2] for idx in self.nut_body_indices]

    def _track_test_data(self):
        """Track transforms for test validation (called each step in test mode)."""
        if not self.test_mode or self.scene != "nut_bolt":
            return

        body_q = self.state_0.body_q.numpy()

        # Track nut rotation and z position
        for i, nut_idx in enumerate(self.nut_body_indices):
            current_q = body_q[nut_idx]
            initial_q = self.nut_initial_transforms[i]

            # Compute rotation change using quaternion dot product
            # |q1 · q2| = cos(theta/2), where theta is the angle between orientations
            q_current = current_q[3:7]  # quaternion part (x, y, z, w)
            q_initial = initial_q[3:7]
            dot = abs(np.dot(q_current, q_initial))
            dot = min(dot, 1.0)  # Clamp for numerical stability
            rotation_angle = 2.0 * np.arccos(dot)
            self.nut_max_rotation_change[i] = max(self.nut_max_rotation_change[i], rotation_angle)

            # Track minimum z (nuts should move down)
            self.nut_min_z[i] = min(self.nut_min_z[i], current_q[2])

    def test_final(self):
        """Verify simulation state after example completes.

        For nut_bolt scene:
        - Bolts should stay approximately in place (limited displacement)
        - Nuts should rotate (thread engagement) and move slightly downward
        """
        if self.scene != "nut_bolt":
            # For gears scene, just verify simulation ran without error
            return

        body_q = self.state_0.body_q.numpy()

        # Check bolts stayed in place
        max_bolt_displacement = 0.01 * self.scene_scale  # 1cm scaled
        for i, bolt_idx in enumerate(self.bolt_body_indices):
            current_pos = body_q[bolt_idx][:3]
            initial_pos = self.bolt_initial_transforms[i][:3]
            displacement = np.linalg.norm(current_pos - initial_pos)
            assert displacement < max_bolt_displacement, (
                f"Bolt {i}: displaced too much. "
                f"Displacement={displacement:.4f} (max allowed={max_bolt_displacement:.4f})"
            )

        # Check nuts rotated and moved down
        min_rotation_threshold = 0.1  # At least ~5.7 degrees of rotation
        for i in range(len(self.nut_body_indices)):
            # Check rotation occurred
            max_rotation = self.nut_max_rotation_change[i]
            assert max_rotation > min_rotation_threshold, (
                f"Nut {i}: did not rotate enough. "
                f"Max rotation={np.degrees(max_rotation):.2f} degrees "
                f"(expected > {np.degrees(min_rotation_threshold):.2f} degrees)"
            )

            # Check nut moved downward (min_z should be less than initial z)
            initial_z = self.nut_initial_transforms[i][2]
            min_z = self.nut_min_z[i]
            assert min_z < initial_z, (
                f"Nut {i}: did not move downward. Initial z={initial_z:.4f}, min z reached={min_z:.4f}"
            )


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    parser.add_argument(
        "--world-count",
        type=int,
        default=100,
        help="Total number of simulated worlds.",
    )
    parser.add_argument(
        "--scene",
        type=str,
        choices=["nut_bolt", "gears"],
        default="nut_bolt",
        help="Scene to run: 'nut_bolt' or 'gears'.",
    )
    parser.add_argument(
        "--solver",
        type=str,
        choices=["xpbd", "mujoco"],
        default="mujoco",
        help="Solver to use: 'xpbd' (Extended Position-Based Dynamics) or 'mujoco' (MuJoCo constraint solver).",
    )

    viewer, args = newton.examples.init(parser)

    example = Example(
        viewer,
        world_count=args.world_count,
        scene=args.scene,
        solver=args.solver,
        test_mode=args.test,
    )

    newton.examples.run(example, args)
