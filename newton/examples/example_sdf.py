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
    sdf_max_dims=512,
    sdf_narrow_band_range=(-0.005, 0.005),
    contact_margin=0.005,
    density=8000.0,
    torsional_friction=0.0,
    rolling_friction=0.0,
    is_hydroelastic=False,
)


def add_mesh_object(
    builder: newton.ModelBuilder,
    mesh_file: str,
    transform: wp.transform,
    shape_cfg: newton.ModelBuilder.ShapeConfig | None = None,
    key: str | None = None,
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

    # Apply scale to the mesh shape
    body = builder.add_body(key=key, xform=transform)
    builder.add_shape_mesh(body, mesh=mesh, scale=(scale, scale, scale), cfg=shape_cfg)
    return body


class Example:
    def __init__(self, viewer, num_worlds=1, num_per_world=1, scene="nut_bolt", solver="xpbd"):
        self.fps = 120
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = 5
        self.sim_dt = self.frame_dt / self.sim_substeps

        self.num_worlds = num_worlds
        self.viewer = viewer
        self.scene = scene
        self.solver_type = solver

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
        self.broad_phase_mode = newton.BroadPhaseMode.SAP

        if scene == "nut_bolt":
            world_builder = self._build_nut_bolt_scene()
        elif scene == "gears":
            world_builder = self._build_gears_scene()
        else:
            raise ValueError(f"Unknown scene: {scene}")

        main_scene = newton.ModelBuilder()
        # Add ground plane with offset (plane equation: z = offset)
        main_scene.add_shape_plane(
            plane=(0.0, 0.0, 1.0, self.ground_plane_offset),
            width=0.0,
            length=0.0,
            key="ground_plane",
        )
        main_scene.replicate(world_builder, num_worlds=self.num_worlds)

        self.model = main_scene.finalize()

        # Override rigid_contact_max BEFORE creating collision pipeline to limit memory allocation
        self.model.rigid_contact_max = self.rigid_contact_max

        self.collision_pipeline = newton.CollisionPipelineUnified.from_model(
            self.model,
            reduce_contacts=True,
            broad_phase_mode=self.broad_phase_mode,
        )

        # Create solver based on user choice
        if self.solver_type == "xpbd":
            self.solver = newton.solvers.SolverXPBD(
                self.model, iterations=10, rigid_contact_relaxation=self.xpbd_contact_relaxation
            )
        elif self.solver_type == "mujoco":
            num_per_world = self.rigid_contact_max // self.num_worlds
            self.solver = newton.solvers.SolverMuJoCo(
                self.model,
                ls_parallel=True,
                solver="newton",
                integrator="implicitfast",
                njmax=num_per_world,
                nconmax=num_per_world,
                ls_iterations=100,
                use_mujoco_contacts=False,
            )
        else:
            raise ValueError(f"Unknown solver type: {self.solver_type}. Choose from 'xpbd' or 'mujoco'.")

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()

        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)

        self.contacts = self.model.collide(self.state_0, collision_pipeline=self.collision_pipeline)

        self.viewer.set_model(self.model)

        if scene == "nut_bolt":
            offset = 0.15 * self.scene_scale
            self.viewer.set_world_offsets((offset, offset, 0.0))
            self.viewer.set_camera(pos=wp.vec3(offset, -offset, 0.12 * self.scene_scale), pitch=-15.0, yaw=135.0)
        else:  # gears
            offset = 0.25 * self.scene_scale
            self.viewer.set_world_offsets((offset, offset, 0.0))
            self.viewer.set_camera(pos=wp.vec3(offset, -offset, 0.2 * self.scene_scale), pitch=-25.0, yaw=135.0)

        self.capture()

    def _build_nut_bolt_scene(self) -> newton.ModelBuilder:
        repo_url = "https://github.com/isaac-sim/IsaacGymEnvs.git"
        print(f"Downloading nut/bolt assets from {repo_url}...")
        asset_path = download_git_folder(repo_url, "assets/factory/mesh/factory_nut_bolt")
        print(f"Assets downloaded to: {asset_path}")

        world_builder = newton.ModelBuilder()
        world_builder.rigid_contact_margin = 0.001 * self.scene_scale

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
                    key=f"bolt_{i}_{j}",
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
                    key=f"nut_{i}_{j}",
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
        world_builder.rigid_contact_margin = 0.001 * self.scene_scale

        for _, (gear_filename, gear_key) in enumerate(GEAR_FILES):
            gear_file = str(asset_path / gear_filename)
            gear_xform = wp.transform(wp.vec3(0.0, 0.0, 0.01) * self.scene_scale, wp.quat_identity())
            add_mesh_object(
                world_builder,
                gear_file,
                gear_xform,
                SHAPE_CFG,
                key=gear_key,
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
        self.contacts = self.model.collide(self.state_0, collision_pipeline=self.collision_pipeline)
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()

            self.viewer.apply_forces(self.state_0)
            # self.contacts = self.model.collide(self.state_0, collision_pipeline=self.collision_pipeline)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)

            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()

        self.sim_time += self.frame_dt

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    parser.add_argument(
        "--num-worlds",
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
    parser.add_argument(
        "--num-per-world",
        type=int,
        default=1,
        help="Number of assemblies per world.",
    )

    viewer, args = newton.examples.init(parser)

    example = Example(
        viewer, num_worlds=args.num_worlds, num_per_world=args.num_per_world, scene=args.scene, solver=args.solver
    )

    newton.examples.run(example, args)
