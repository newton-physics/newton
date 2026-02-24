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
# Example cloth H1 (cloth-robot interaction)
#
# The H1 robot in a jacket waves hello to us, powered by the Style3D solver
# for cloth and driven by an IkSolver for robot kinematics.
#
# Demonstrates how to leverage interpolated robot kinematics within the
# collision processing pipeline and feed the results to the cloth solver.
#
# Command: python -m newton.examples cloth_h1
#
###########################################################################

import numpy as np
import warp as wp
from pxr import Usd, UsdGeom

import newton
import newton.examples
import newton.ik as ik
import newton.usd
import newton.utils
from newton.solvers import style3d


class Example:
    def __init__(self, viewer, args=None):
        # frame timing
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0

        # must be an even number when using CUDA Graph
        self.sim_substeps = 8
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.iterations = 10

        self.viewer = viewer
        self.frame_index = 0

        # ------------------------------------------------------------------
        # Build a single H1 (fixed base for stability) + ground
        # ------------------------------------------------------------------
        h1 = newton.ModelBuilder()
        h1.add_mjcf(
            newton.utils.download_asset("unitree_h1") / "mjcf/h1_with_hand.xml",
            floating=False,
            enable_self_collisions=False,
        )
        h1.add_ground_plane()

        # ------------------------------------------------------------------
        # Build a cloth
        # ------------------------------------------------------------------
        garment_usd_name = "h1_jacket"
        # garment_usd_name = "h1_cake_skirt"
        cloth_builder = newton.ModelBuilder()
        newton.solvers.SolverStyle3D.register_custom_attributes(cloth_builder)
        asset_path = newton.utils.download_asset("style3d")
        usd_stage = Usd.Stage.Open(f"{asset_path}/garments/{garment_usd_name}.usd")
        usd_prim_garment = usd_stage.GetPrimAtPath(f"/Root/{garment_usd_name}/Root_Garment")

        garment_mesh = newton.usd.get_mesh(usd_prim_garment, load_uvs=False)
        self.garment_mesh_indices = garment_mesh.indices
        self.garment_mesh_points = garment_mesh.vertices[:, [2, 0, 1]]  # y-up to z-up

        # Load raw UV values and indices directly from the primvar
        # (get_mesh expands indexed UVs, but style3d needs raw values + indices)
        uv_primvar = UsdGeom.PrimvarsAPI(usd_prim_garment).GetPrimvar("st")
        self.garment_mesh_uv = np.array(uv_primvar.Get()) * 1e-3
        self.garment_mesh_uv_indices = np.array(uv_primvar.GetIndices())

        style3d.add_cloth_mesh(
            cloth_builder,
            pos=wp.vec3(0, 0, 0),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0, 0.0, 0.0),
            panel_verts=self.garment_mesh_uv.tolist(),
            panel_indices=self.garment_mesh_uv_indices.tolist(),
            vertices=self.garment_mesh_points.tolist(),
            indices=self.garment_mesh_indices.tolist(),
            density=0.5,
            scale=1.0,
            particle_radius=3.0e-3,
            tri_aniso_ke=wp.vec3(1.0e2, 1.0e2, 1.0e2) * 10.0,
            edge_aniso_ke=wp.vec3(1.0e-6, 1.0e-6, 1.0e-6) * 40.0,
        )
        h1.add_world(cloth_builder)

        self.graph = None
        self.model = h1.finalize()
        self.model.soft_contact_ke = 5e3
        # no friction
        self.model.soft_contact_mu = 0.0
        self.model.shape_material_mu.fill_(0.0)
        self.viewer.set_model(self.model)
        self.viewer.set_camera(wp.vec3(2.5, 0.0, 1.5), 0.0, 180.0)

        # states
        self.state = self.model.state()
        self.state1 = self.model.state()
        self.body_q_0 = wp.clone(self.state.body_q, device=self.model.device)  # last state
        self.body_q_1 = wp.clone(self.state.body_q, device=self.model.device)  # current state
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state)
        # ------------------------------------------------------------------
        # End effectors
        # ------------------------------------------------------------------
        self.ee = [
            ("left_hand", 16),
            ("right_hand", 33),
            ("left_foot", 5),
            ("right_foot", 10),
        ]

        # ------------------------------------------------------------------
        # Persistent gizmo transforms (pass-by-ref objects mutated by viewer)
        # ------------------------------------------------------------------
        body_q_np = self.state.body_q.numpy()
        self.ee_tfs = [wp.transform(*body_q_np[link_idx]) for _, link_idx in self.ee]

        # ------------------------------------------------------------------
        # IK setup (single problem)
        # ------------------------------------------------------------------
        def _q2v4(q):
            return wp.vec4(q[0], q[1], q[2], q[3])

        # Position & rotation objectives
        self.pos_objs = []
        self.rot_objs = []
        for ee_i, (_, link_idx) in enumerate(self.ee):
            tf = self.ee_tfs[ee_i]

            self.pos_objs.append(
                ik.IKObjectivePosition(
                    link_index=link_idx,
                    link_offset=wp.vec3(0.0, 0.0, 0.0),
                    target_positions=wp.array([wp.transform_get_translation(tf)], dtype=wp.vec3),
                )
            )

            self.rot_objs.append(
                ik.IKObjectiveRotation(
                    link_index=link_idx,
                    link_offset_rotation=wp.quat_identity(),
                    target_rotations=wp.array([_q2v4(wp.transform_get_rotation(tf))], dtype=wp.vec4),
                )
            )

        # Joint limit objective
        self.obj_joint_limits = ik.IKObjectiveJointLimit(
            joint_limit_lower=self.model.joint_limit_lower,
            joint_limit_upper=self.model.joint_limit_upper,
            weight=10.0,
        )

        # Variables the solver will update
        self.joint_q = wp.array(self.model.joint_q, shape=(1, self.model.joint_coord_count))

        self.ik_iters = 24
        self.ik_solver = ik.IKSolver(
            model=self.model,
            n_problems=1,
            objectives=[*self.pos_objs, *self.rot_objs, self.obj_joint_limits],
            lambda_initial=0.1,
            jacobian_mode=ik.IKJacobianType.ANALYTIC,
        )
        self.ik_solver.step(self.joint_q, self.joint_q, iterations=self.ik_iters)
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state)

        # ------------------------------------------------------------------
        # Cloth solver
        # ------------------------------------------------------------------
        self.cloth_solver = newton.solvers.SolverStyle3D(
            model=self.model,
            iterations=self.iterations,
        )
        self.cloth_solver.precompute(h1)
        self.cloth_solver.collision.radius = 3.5e-3
        self.control = self.model.control()

        self.contacts = self.model.contacts()
        self.shape_flags = self.model.shape_flags.numpy()

    # ----------------------------------------------------------------------
    # Helpers
    # ----------------------------------------------------------------------
    def capture(self):
        self.graph = None
        if wp.get_device().is_cuda:
            with wp.ScopedCapture() as cap:
                self.simulate()
            self.graph = cap.graph

    @wp.kernel
    def transform_interpolate(
        ratio: float,
        transform0: wp.array(dtype=wp.transform),
        transform1: wp.array(dtype=wp.transform),
        # outputs
        new_transform: wp.array(dtype=wp.transform),
    ):
        tid = wp.tid()
        tf0 = transform0[tid]
        tf1 = transform1[tid]
        rot0 = wp.transform_get_rotation(tf0)
        rot1 = wp.transform_get_rotation(tf1)
        pos0 = wp.transform_get_translation(tf0)
        pos1 = wp.transform_get_translation(tf1)
        new_pos = wp.lerp(pos0, pos1, ratio)
        new_rot = wp.quat_slerp(rot0, rot1, ratio)
        new_transform[tid] = wp.transformation(new_pos, new_rot, dtype=float)

    def simulate(self):
        self.ik_solver.step(self.joint_q, self.joint_q, iterations=self.ik_iters)
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state)

        self.body_q_1.assign(self.state.body_q)
        self.cloth_solver.rebuild_bvh(self.state)
        for ii in range(self.sim_substeps):
            wp.launch(
                self.transform_interpolate,
                inputs=[ii / (self.sim_substeps - 1.0), self.body_q_0, self.body_q_1],
                outputs=[self.state1.body_q],
                dim=self.model.body_count,
                device=self.model.device,
            )
            self.state.body_q.assign(self.state1.body_q)
            self.model.collide(self.state, self.contacts)
            self.cloth_solver.step(self.state, self.state1, self.control, self.contacts, self.sim_dt)
            (self.state, self.state1) = (self.state1, self.state)

        self.body_q_0.assign(self.body_q_1)

    def _push_targets_from_gizmos(self):
        """Read gizmo-updated transforms and push into IK objectives."""
        for i, tf in enumerate(self.ee_tfs):
            self.pos_objs[i].set_target_position(0, wp.transform_get_translation(tf))
            q = wp.transform_get_rotation(tf)
            self.rot_objs[i].set_target_rotation(0, wp.vec4(q[0], q[1], q[2], q[3]))

    def _force_update_targets(self):
        # key infos
        key_time = [2.0, 6.0, 10.0]  # second
        target_pos = [
            [wp.vec3(0.16, 0.65, 1.71), wp.vec3(0.28, -0.50, 1.19)],  # key 0
            [wp.vec3(0.12, 0.34, 0.99), wp.vec3(0.14, -0.35, 0.97)],  # key 1
        ]
        target_rot = [
            [wp.quat(0.58, -0.35, 0.29, 0.68), wp.quat(0.00, 0.00, 0.00, 0.00)],  # key 0
            [wp.quat(-0.09, 0.46, 0.03, 0.88), wp.quat(-0.09, 0.48, 0.01, 0.87)],  # key 1
        ]
        if self.sim_time < key_time[0]:
            """Raise hands"""
            rot_lerp_ratio = wp.clamp(0.3 * self.sim_time / key_time[0], 0.0, 1.0)
            pos_lerp_ratio = wp.clamp(0.1 * self.sim_time / key_time[0], 0.0, 1.0)
            for i in range(len(target_pos)):
                tf = self.ee_tfs[i]
                rot = wp.transform_get_rotation(tf)
                pos = wp.transform_get_translation(tf)
                wp.transform_set_translation(tf, wp.lerp(pos, target_pos[0][i], pos_lerp_ratio))
                wp.transform_set_rotation(tf, wp.quat_slerp(rot, target_rot[0][i], rot_lerp_ratio))
        elif self.sim_time < key_time[1]:
            """Wave hands"""
            time_budget = key_time[1] - key_time[0]
            rot_angle = wp.sin((self.sim_time - key_time[0]) * 7.5 * wp.pi / time_budget) * 0.3
            rot = wp.quat_from_axis_angle(axis=wp.vec3(1, 0, 0), angle=rot_angle) * target_rot[0][0]
            pos0 = target_pos[0][0] + wp.vec3(
                wp.sin((self.sim_time - key_time[0]) * 7.5 * wp.pi / time_budget) * 0.1, 0.0, 0.0
            )
            pos1 = target_pos[0][1] + wp.vec3(
                0.0, wp.sin((self.sim_time - key_time[0]) * 2.5 * wp.pi / time_budget) * 0.05, 0.0
            )
            wp.transform_set_rotation(self.ee_tfs[0], wp.quat(rot))
            wp.transform_set_translation(self.ee_tfs[0], pos0)
            wp.transform_set_translation(self.ee_tfs[1], pos1)
        elif self.sim_time < key_time[2]:
            """Drop hands"""
            pos_lerp_ratio = wp.clamp((self.sim_time - key_time[1]) / (key_time[2] - key_time[1]), 0.0, 1.0)
            rot_lerp_ratio = wp.clamp((self.sim_time - key_time[1]) / (key_time[2] - key_time[1]), 0.0, 1.0)
            for i in range(len(target_pos)):
                tf = self.ee_tfs[i]
                rot = wp.transform_get_rotation(tf)
                pos = wp.transform_get_translation(tf)
                wp.transform_set_translation(tf, wp.lerp(pos, target_pos[1][i], wp.pow(pos_lerp_ratio, 2.0)))
                wp.transform_set_rotation(tf, wp.quat_slerp(rot, target_rot[1][i], wp.pow(rot_lerp_ratio, 3.0)))

    # ----------------------------------------------------------------------
    # Template API
    # ----------------------------------------------------------------------
    def step(self):
        if self.frame_index > 0:
            self._force_update_targets()
            self._push_targets_from_gizmos()
            if self.graph:
                wp.capture_launch(self.graph)
            elif wp.get_device().is_cuda:
                self.capture()
            else:
                self.simulate()
            self.sim_time += self.frame_dt
        self.frame_index += 1

    def test_final(self):
        p_lower = wp.vec3(-0.3, -0.8, 0.8)
        p_upper = wp.vec3(0.5, 0.8, 1.8)
        newton.examples.test_particle_state(
            self.state,
            "particles are within a reasonable volume",
            lambda q, qd: newton.math.vec_inside_limits(q, p_lower, p_upper),
        )

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state)
        self.viewer.end_frame()
        wp.synchronize()


if __name__ == "__main__":
    # Parse arguments and initialize viewer
    parser = newton.examples.create_parser()
    parser.set_defaults(num_frames=601)
    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args)
    newton.examples.run(example, args)
