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
# Example Panda Hydro
#
# Shows how to set up a pick-and-place manipulation simulation using a
# Franka Panda arm with SDF hydroelastic contacts and inverse kinematics.
# Supports different scene configurations: pen or cube.
#
# Command: python -m newton.examples panda_hydro --scene pen --num-worlds 1
#
###########################################################################

import copy
from enum import Enum

import numpy as np
import warp as wp
from pxr import Usd

import newton
import newton.examples
import newton.ik as ik
import newton.usd
import newton.utils
from newton.geometry import SDFHydroelasticConfig, create_box_mesh


class SceneType(Enum):
    PEN = "pen"
    CUBE = "cube"


def quat_to_vec4(q: wp.quat) -> wp.vec4:
    """Convert a quaternion to a vec4."""
    return wp.vec4(q[0], q[1], q[2], q[3])


@wp.kernel
def broadcast_ik_solution_kernel(
    ik_solution: wp.array2d(dtype=wp.float32),
    joint_targets: wp.array2d(dtype=wp.float32),
    gripper_value: float,
):
    world_idx = wp.tid()
    for j in range(7):
        joint_targets[world_idx, j] = ik_solution[0, j]
    joint_targets[world_idx, 7] = gripper_value
    joint_targets[world_idx, 8] = gripper_value


class Example:
    def __init__(self, viewer, scene=SceneType.PEN, num_worlds=1, test_mode=False):
        self.scene = SceneType(scene)
        self.test_mode = test_mode
        self.show_isosurface = False  # Disabled by default for performance
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = 10
        self.collide_substeps = 2  # run collision detection every X simulation steps
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.num_worlds = num_worlds
        self.viewer = viewer

        shape_cfg = newton.ModelBuilder.ShapeConfig(
            k_hydro=1e11,
            sdf_max_resolution=64,
            is_hydroelastic=True,
            sdf_narrow_band_range=(-0.01, 0.01),
            contact_margin=0.01,
            torsional_friction=0.0,
            rolling_friction=0.0,
        )

        builder = newton.ModelBuilder()
        builder.default_shape_cfg = shape_cfg

        builder.add_urdf(
            newton.utils.download_asset("franka_emika_panda") / "urdf/fr3_franka_hand.urdf",
            xform=wp.transform((-0.5, -0.5, 0.05), wp.quat_identity()),
            enable_self_collisions=False,
            parse_visuals_as_colliders=True,
        )

        # Disable SDF collisions on all panda links except the fingers and hand
        finger_body_indices = {
            builder.body_key.index("fr3_leftfinger"),
            builder.body_key.index("fr3_rightfinger"),
            builder.body_key.index("fr3_hand"),
        }
        non_finger_shape_indices = []
        for shape_idx, body_idx in enumerate(builder.shape_body):
            if body_idx not in finger_body_indices:
                builder.shape_flags[shape_idx] &= ~newton.ShapeFlags.HYDROELASTIC
                non_finger_shape_indices.append(shape_idx)

        # Convert non-finger shapes to convex hulls
        builder.approximate_meshes(
            method="convex_hull", shape_indices=non_finger_shape_indices, keep_visual_shapes=True
        )

        init_q = [
            -3.6802115e-03,
            2.3901723e-02,
            3.6804110e-03,
            -2.3683236e00,
            -1.2918962e-04,
            2.3922248e00,
            7.8549200e-01,
        ]
        builder.joint_q[:9] = [*init_q, 0.05, 0.05]
        builder.joint_target_pos[:9] = [*init_q, 1.0, 1.0]

        builder.joint_target_ke[:9] = [500.0] * 9
        builder.joint_target_kd[:9] = [100.0] * 9
        builder.joint_effort_limit[:7] = [80.0] * 7
        builder.joint_effort_limit[7:9] = [20.0] * 2
        builder.joint_armature[:7] = [0.1] * 7
        builder.joint_armature[7:9] = [0.5] * 2

        # Add gripper pads
        if self.scene in [SceneType.PEN, SceneType.CUBE]:
            left_finger_idx = builder.body_key.index("fr3_leftfinger")
            right_finger_idx = builder.body_key.index("fr3_rightfinger")

            pad_asset_path = newton.utils.download_asset("manipulation_objects/pad")
            pad_stage = Usd.Stage.Open(str(pad_asset_path / "model.usda"))
            pad_mesh = newton.usd.get_mesh(
                pad_stage.GetPrimAtPath("/root/Model/Model"),
                load_normals=True,
                face_varying_normal_conversion="vertex_splitting",
            )
            pad_scale = newton.usd.get_scale(pad_stage.GetPrimAtPath("/root/Model"))
            pad_xform = wp.transform(
                wp.vec3(0.0, 0.005, 0.045),
                wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), -np.pi),
            )
            builder.add_shape_mesh(body=left_finger_idx, mesh=pad_mesh, xform=pad_xform, scale=pad_scale, cfg=shape_cfg)
            builder.add_shape_mesh(
                body=right_finger_idx, mesh=pad_mesh, xform=pad_xform, scale=pad_scale, cfg=shape_cfg
            )

        # Table
        box_size = 0.05
        table_half_extents = (box_size * 2, box_size * 2, box_size)  # half-extents
        table_vertices, table_indices = create_box_mesh(table_half_extents, duplicate_vertices=True)
        table_mesh = newton.Mesh(table_vertices, table_indices)
        builder.add_shape_mesh(
            body=-1,
            mesh=table_mesh,
            xform=wp.transform(wp.vec3(0.08, -0.5, box_size), wp.quat_identity()),
            cfg=shape_cfg,
        )

        # Object to manipulate
        self.put_in_cup = True

        if self.scene == SceneType.PEN:
            radius = 0.005
            length = 0.14
            self.object_pos = [0.0, -0.5, 2 * box_size + radius + 0.001]
            object_xform = wp.transform(
                wp.vec3(self.object_pos),
                wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), np.pi / 2),
            )
            pen_cfg = copy.deepcopy(shape_cfg)
            self.object_body_local = builder.add_body(xform=object_xform, key="object")
            builder.add_shape_capsule(body=self.object_body_local, radius=radius, half_height=length / 2, cfg=pen_cfg)
            self.grasping_offset = [-0.03, 0.0, 0.13]
            self.place_offset = -0.015  # Gripper reaches 1.5cm further into cup

        elif self.scene == SceneType.CUBE:
            size = 0.04
            self.object_pos = [0.0, -0.5, 2 * box_size + 0.5 * size]
            object_xform = wp.transform(wp.vec3(self.object_pos), wp.quat_identity())
            self.object_body_local = builder.add_body(xform=object_xform, key="object")
            builder.add_shape_box(body=self.object_body_local, hx=size / 2, hy=size / 2, hz=size / 2)
            self.grasping_offset = [0.03, 0.0, 0.14]
            self.place_offset = 0.02

        if self.put_in_cup:
            self.cup_pos = [0.13, -0.5, box_size + 0.1]

            cup_asset_path = newton.utils.download_asset("manipulation_objects/cup")
            cup_stage = Usd.Stage.Open(str(cup_asset_path / "model.usda"))
            prim = cup_stage.GetPrimAtPath("/root/Model/Model")
            cup_mesh = newton.usd.get_mesh(prim, load_normals=True, face_varying_normal_conversion="vertex_splitting")
            cup_scale = newton.usd.get_scale(cup_stage.GetPrimAtPath("/root/Model"))
            cup_xform = wp.transform(
                wp.vec3(self.cup_pos),
                wp.quat_identity(),
            )
            cup_body = builder.add_body(key="cup", xform=cup_xform)
            builder.add_shape_mesh(body=cup_body, mesh=cup_mesh, scale=cup_scale, cfg=shape_cfg)

        # build model for IK
        self.model_single = copy.deepcopy(builder).finalize()

        # Store bodies per world before replication
        self.bodies_per_world = builder.body_count

        scene = newton.ModelBuilder()
        scene.replicate(builder, self.num_worlds)
        scene.add_ground_plane()

        self.model = scene.finalize()

        # num_hydroelastic = (self.model.shape_flags.numpy() & newton.ShapeFlags.HYDROELASTIC).astype(bool).sum()
        # print(f"Number of hydroelastic shapes: {num_hydroelastic} / {self.model.shape_count}")

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)

        # Create collision pipeline with SDF hydroelastic config
        # Enable output_contact_surface so the kernel code is compiled (allows runtime toggle)
        # The actual writing is controlled by set_output_contact_surface() at runtime
        sdf_hydroelastic_config = SDFHydroelasticConfig(
            output_contact_surface=hasattr(viewer, "renderer"),  # Compile in if viewer supports it
        )
        self.collision_pipeline = newton.CollisionPipeline.from_model(
            self.model,
            reduce_contacts=True,
            broad_phase_mode=newton.BroadPhaseMode.EXPLICIT,
            sdf_hydroelastic_config=sdf_hydroelastic_config,
        )
        self.contacts = self.model.collide(self.state_0, collision_pipeline=self.collision_pipeline)

        # Create MuJoCo solver with Newton contacts
        self.solver = newton.solvers.SolverMuJoCo(
            self.model,
            use_mujoco_contacts=False,
            solver="newton",
            integrator="implicitfast",
            cone="elliptic",
            njmax=500,
            nconmax=500,
            iterations=15,
            ls_iterations=100,
            impratio=1000.0,
        )

        self.viewer.set_model(self.model)
        self.viewer.picking_enabled = False  # Disable interactive picking for this example
        if hasattr(self.viewer, "renderer"):
            self.viewer.set_camera(wp.vec3(0.5, 0.0, 0.5), -15, -140)
            self.viewer.set_world_offsets(wp.vec3(1.0, 1.0, 0.0))
            self.viewer.show_hydro_contact_surface = self.show_isosurface
            self.viewer.register_ui_callback(self.render_ui, position="side")

        # Initialize state for IK setup
        self.state = self.model_single.state()
        newton.eval_fk(self.model_single, self.model.joint_q, self.model.joint_qd, self.state)

        self.setup_ik()
        self.control = self.model.control()
        self.joint_target_shape = self.control.joint_target_pos.reshape((self.num_worlds, -1)).shape
        self.joint_targets_2d = wp.zeros(self.joint_target_shape, dtype=wp.float32)
        wp.copy(self.control.joint_target_pos[:9], self.model.joint_q[:9])

        # Track maximum object height for testing (only in test mode)
        self.object_max_z = [self.object_pos[2]] * self.num_worlds if self.test_mode else None

        self.capture()
        self.capture_ik()

    def set_joint_targets(self):
        self.time_in_waypoint += self.frame_dt

        # interpolate between waypoints
        t = self.time_in_waypoint / self.waypoints[self.current_waypoint][1]
        next_waypoint = (self.current_waypoint + 1) % len(self.waypoints)
        target_position = self.waypoints[self.current_waypoint][0] * (1 - t) + self.waypoints[next_waypoint][0] * t

        target_angle_z = self.waypoints[self.current_waypoint][-1] * (1 - t) + self.waypoints[next_waypoint][-1] * t
        target_rotation = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), np.pi)
        target_rotation = wp.mul(target_rotation, wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), target_angle_z))

        self.pos_obj.set_target_positions(wp.array([target_position], dtype=wp.vec3))
        self.rot_obj.set_target_rotations(wp.array([quat_to_vec4(target_rotation)], dtype=wp.vec4))

        if self.graph_ik is not None:
            wp.capture_launch(self.graph_ik)
        else:
            self.ik_solver.step(self.joint_q_ik, self.joint_q_ik, iterations=self.ik_iters)

        # Broadcast single IK solution to all worlds
        t_gripper = self.waypoints[self.current_waypoint][2] * (1 - t) + self.waypoints[next_waypoint][2] * t
        gripper_value = 0.06 * (1 - t_gripper)
        wp.launch(
            broadcast_ik_solution_kernel,
            dim=self.num_worlds,
            inputs=[self.joint_q_ik, self.joint_targets_2d, gripper_value],
        )
        wp.copy(self.control.joint_target_pos, self.joint_targets_2d.flatten())

        if self.time_in_waypoint >= self.waypoints[self.current_waypoint][1]:
            self.current_waypoint = (self.current_waypoint + 1) % len(self.waypoints)
            self.time_in_waypoint = 0.0

    def capture_ik(self):
        self.graph_ik = None
        with wp.ScopedCapture() as capture:
            self.ik_solver.step(self.joint_q_ik, self.joint_q_ik, iterations=self.ik_iters)
        self.graph_ik = capture.graph

    def capture(self):
        self.graph = None
        if wp.get_device().is_cuda:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph

    def simulate(self):
        self.state_0.clear_forces()
        self.state_1.clear_forces()

        for i in range(self.sim_substeps):
            if i % self.collide_substeps == 0:
                self.contacts = self.model.collide(self.state_0, collision_pipeline=self.collision_pipeline)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        self.set_joint_targets()
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()

        self.sim_time += self.frame_dt

        # Track maximum object height for testing (only in test mode)
        if self.test_mode:
            body_q = self.state_0.body_q.numpy()
            for world_idx in range(self.num_worlds):
                object_body_idx = world_idx * self.bodies_per_world + self.object_body_local
                z_pos = float(body_q[object_body_idx][2])
                self.object_max_z[world_idx] = max(self.object_max_z[world_idx], z_pos)

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
        # Always call log_hydro_contact_surface - it handles show_hydro_contact_surface internally
        # and will clear the lines when disabled
        self.viewer.log_hydro_contact_surface(
            self.collision_pipeline.get_hydro_contact_surface(), penetrating_only=True
        )
        self.viewer.end_frame()

    def render_ui(self, imgui):
        changed, self.show_isosurface = imgui.checkbox("Show Isosurface", self.show_isosurface)
        if changed:
            self.viewer.show_hydro_contact_surface = self.show_isosurface
            # Toggle whether to compute/write the isosurface data in the collision pipeline
            self.collision_pipeline.set_output_contact_surface(self.show_isosurface)

    def test_final(self):
        # Verify that the object was picked up by checking the maximum height reached
        initial_z = self.object_pos[2]
        min_lift_height = 0.25  # Object should be lifted at least 25cm above initial position

        for world_idx in range(self.num_worlds):
            max_z = self.object_max_z[world_idx]
            max_lift = max_z - initial_z

            assert max_lift > min_lift_height, (
                f"World {world_idx}: Object was not picked up high enough. "
                f"Initial z={initial_z:.3f}, max z reached={max_z:.3f}, "
                f"max lift={max_lift:.3f} (expected > {min_lift_height})"
            )

    def setup_ik(self):
        self.ee_index = 10
        body_q_np = self.state.body_q.numpy()
        self.ee_tf = wp.transform(*body_q_np[self.ee_index])

        # Position objective (single IK problem)
        self.pos_obj = ik.IKObjectivePosition(
            link_index=self.ee_index,
            link_offset=wp.vec3(0.0, 0.0, 0.0),
            target_positions=wp.array([wp.transform_get_translation(self.ee_tf)], dtype=wp.vec3),
        )

        # Rotation objective (single IK problem)
        self.rot_obj = ik.IKObjectiveRotation(
            link_index=self.ee_index,
            link_offset_rotation=wp.quat_identity(),
            target_rotations=wp.array([quat_to_vec4(wp.transform_get_rotation(self.ee_tf))], dtype=wp.vec4),
        )

        # Joint limit objective
        self.obj_joint_limits = ik.IKObjectiveJointLimit(
            joint_limit_lower=self.model_single.joint_limit_lower,
            joint_limit_upper=self.model_single.joint_limit_upper,
        )

        # Variables the solver will update (single IK problem)
        self.joint_q_ik = wp.array(self.model_single.joint_q, shape=(1, self.model_single.joint_coord_count))

        self.ik_iters = 24
        self.ik_solver = ik.IKSolver(
            model=self.model_single,
            n_problems=1,
            objectives=[self.pos_obj, self.rot_obj, self.obj_joint_limits],
            lambda_initial=0.1,
            jacobian_mode=ik.IKJacobianType.ANALYTIC,
        )
        self.time_in_waypoint = 0.0
        self.current_waypoint = 0
        self.z_rest = 0.5
        grasping_pos = wp.vec3(self.object_pos) + wp.vec3(self.grasping_offset)
        resting_pos = wp.vec3([grasping_pos[0], grasping_pos[1], self.z_rest])
        grasp_pos = 1.0
        no_grasp_pos = 0.0
        rot_hand = 0.0
        self.waypoints = [
            [resting_pos, 1.0, no_grasp_pos, rot_hand],
            [grasping_pos, 1.0, no_grasp_pos, rot_hand],
            [grasping_pos, 1.0, grasp_pos, rot_hand],
            [resting_pos, 1.0, grasp_pos, rot_hand],
        ]

        if self.put_in_cup:
            loose_pos = 0.74
            wps = []
            cup_pos_higher = wp.vec3([self.cup_pos[0] + self.place_offset, self.cup_pos[1], self.z_rest])
            cup_pos_lower = wp.vec3([self.cup_pos[0] + self.place_offset, self.cup_pos[1], self.z_rest - 0.1])
            wps.extend(
                [
                    [cup_pos_higher, 2.0, grasp_pos, rot_hand],
                    [cup_pos_higher, 2.0, loose_pos, rot_hand],
                    [cup_pos_higher, 1.0, loose_pos, rot_hand],
                    [cup_pos_lower, 1.0, loose_pos, rot_hand],
                    [cup_pos_lower, 1.0, 0.0, rot_hand],
                ]
            )
            self.waypoints.extend(wps)


if __name__ == "__main__":
    # Parse arguments and initialize viewer
    parser = newton.examples.create_parser()
    parser.set_defaults(num_frames=600)
    parser.add_argument(
        "--scene",
        type=str,
        choices=[scene.value for scene in SceneType],
        default=SceneType.PEN.value,
        help="Scene type to load (pen, cube)",
    )
    parser.add_argument(
        "--num-worlds",
        type=int,
        default=1,
        help="Number of parallel worlds to simulate",
    )

    args = parser.parse_known_args()[0]

    viewer, args = newton.examples.init(parser)

    example = Example(viewer, scene=args.scene, num_worlds=args.num_worlds, test_mode=args.test)

    newton.examples.run(example, args)
