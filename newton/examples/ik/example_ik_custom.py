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
# Example IK Custom (custom collision objective + sphere gizmo)
#
# Inverse kinematics on a Franka arm, with collision
# avoidance against an interactive sphere obstacle.
#
# - Adds a custom CollisionSphereAvoidObjective (softplus penalty) for the EE
# - Adds gizmos for the end-effector target and the obstacle sphere
# - Re-solves IK every frame from the latest gizmo transforms
#
# Command: python -m newton.examples ik_custom
###########################################################################

import warp as wp

import newton
import newton.examples
import newton.ik as ik
import newton.utils


# -------------------------------------------------------------------------
# Custom collision residuals
# -------------------------------------------------------------------------
@wp.kernel
def _collision_residuals(
    body_q: wp.array2d(dtype=wp.transform),  # (batch_rows, n_bodies)
    obstacle_centers: wp.array1d(dtype=wp.vec3),  # (n_problems,)
    obstacle_radii: wp.array1d(dtype=wp.float32),  # (n_problems,)
    link_index: int,
    link_offset: wp.vec3,
    link_radius: float,
    start_idx: int,  # start row in the global residual vector
    weight: float,
    problem_idx: wp.array1d(dtype=wp.int32),  # (batch_rows,)
    # outputs
    residuals: wp.array2d(dtype=wp.float32),  # (batch_rows, total_residuals)
):
    row_idx = wp.tid()
    base = problem_idx[row_idx]

    # EE sphere centre in world space
    tf = body_q[row_idx, link_index]
    ee_pos = wp.transform_point(tf, link_offset)

    # Obstacle sphere (in world space)
    c = obstacle_centers[base]
    r_obs = obstacle_radii[base]

    # Softplus of penetration depth to keep it smooth
    dist = wp.length(ee_pos - c)
    delta = (link_radius + r_obs) - dist
    margin = 0.05
    pen = wp.log(1.0 + wp.exp(delta / margin)) * margin

    residuals[row_idx, start_idx] = weight * pen


@wp.kernel
def _update_center_target(
    problem_idx: int,
    new_center: wp.vec3,
    # outputs
    centers: wp.array1d(dtype=wp.vec3),  # (n_problems,)
):
    centers[problem_idx] = new_center


class CollisionSphereAvoidObjective(ik.IKObjective):
    """
    Sphere-sphere collision avoidance objective.
    Produces a single residual per problem (softplus of penetration depth).
    """

    def __init__(
        self,
        link_index,
        link_offset,
        link_radius,
        obstacle_centers,
        obstacle_radii,
        weight: float = 1.0,
    ):
        super().__init__()
        self.link_index = link_index
        self.link_offset = link_offset
        self.link_radius = link_radius
        self.obstacle_centers = obstacle_centers
        self.obstacle_radii = obstacle_radii
        self.weight = weight
        self.device = None

    def residual_dim(self):
        return 1

    def compute_residuals(self, body_q, joint_q, model, residuals, start_idx, problem_idx):
        count = body_q.shape[0]
        wp.launch(
            _collision_residuals,
            dim=count,
            inputs=[
                body_q,
                self.obstacle_centers,
                self.obstacle_radii,
                self.link_index,
                self.link_offset,
                self.link_radius,
                start_idx,
                self.weight,
                problem_idx,
            ],
            outputs=[residuals],
            device=self.device,
        )

    # --- API to move the obstacle center ---
    def set_obstacle_center(self, problem_idx: int, new_center: wp.vec3):
        if self.device is None:
            raise RuntimeError("CollisionSphereAvoidObjective is not bound to a device")
        wp.launch(
            _update_center_target,
            dim=1,
            inputs=[problem_idx, new_center],
            outputs=[self.obstacle_centers],
            device=self.device,
        )


class Example:
    def __init__(self, viewer):
        # frame timing
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0

        self.viewer = viewer
        self.viewer.show_particles = True

        # ------------------------------------------------------------------
        # Build a single FR3 (fixed base) + ground
        # ------------------------------------------------------------------
        builder = newton.ModelBuilder()
        builder.add_urdf(
            newton.utils.download_asset("franka_emika_panda") / "urdf/fr3_franka_hand.urdf",
            floating=False,
        )
        builder.add_ground_plane()

        # --- Particle for the obstacle sphere ----------
        self.obstacle_center = wp.vec3(0.5, 0.0, 0.5)
        self.obstacle_radius = 0.08
        self._obstacle_shape_index = None
        builder.add_particle(pos=self.obstacle_center, vel=wp.vec3(), mass=1.0, radius=self.obstacle_radius)

        self.graph = None
        self.model = builder.finalize()
        self.viewer.set_model(self.model)

        self.state = self.model.state()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state)

        # ------------------------------------------------------------------
        # Links and end effector indices
        # ------------------------------------------------------------------
        self.ee_index = 10  # fr3_hand_tcp
        self.links_to_check_collision = [
            ("fr3_link5", 7, 0.06),  # (name, index, radius)
            ("fr3_link7", 9, 0.06),
            ("fr3_hand_tcp", 10, 0.05),
            ("fr3_link3", 5, 0.08),  # elbow block: frequent contact risk during reach-backs and around-table moves
            ("fr3_link4", 6, 0.07),  # proximal forearm: fills gap between elbow and your existing link5 sphere
            ("fr3_link6", 8, 0.06),  # wrist housing: catches close passes near fixtures when orienting the tool
            # Optional but helpful if space is tight around the shoulder/torso:
            ("fr3_link2", 4, 0.075),  # upper arm near shoulder: guards early sweep during large reorientations
        ]

        # ------------------------------------------------------------------
        # Persistent gizmo transforms (pass-by-ref objects mutated by viewer)
        # ------------------------------------------------------------------
        body_q_np = self.state.body_q.numpy()
        self.ee_tf = wp.transform(*body_q_np[self.ee_index])
        self.sphere_tf = wp.transform(self.obstacle_center, wp.quat_identity())

        # ------------------------------------------------------------------
        # IK setup
        # ------------------------------------------------------------------
        def _q2v4(q):
            return wp.vec4(q[0], q[1], q[2], q[3])

        # Position & rotation objectives ------------------------------------
        self.pos_obj = ik.IKPositionObjective(
            link_index=self.ee_index,
            link_offset=wp.vec3(0.0, 0.0, 0.0),
            target_positions=wp.array([wp.transform_get_translation(self.ee_tf)], dtype=wp.vec3),
        )

        self.rot_obj = ik.IKRotationObjective(
            link_index=self.ee_index,
            link_offset_rotation=wp.quat_identity(),
            target_rotations=wp.array([_q2v4(wp.transform_get_rotation(self.ee_tf))], dtype=wp.vec4),
        )

        # Collision objectives ----------------------------------------------
        # Share the same arrays across all objectives so one update suffices.
        self.obstacle_centers = wp.array([self.obstacle_center], dtype=wp.vec3)
        self.obstacle_radii = wp.array([self.obstacle_radius], dtype=wp.float32)

        self.collision_objs = []
        for _, (_, link_idx, link_radius) in enumerate(self.links_to_check_collision):
            self.collision_objs.append(
                CollisionSphereAvoidObjective(
                    link_index=link_idx,
                    link_offset=wp.vec3(0.0, 0.0, 0.0),
                    link_radius=link_radius,
                    obstacle_centers=self.obstacle_centers,
                    obstacle_radii=self.obstacle_radii,
                    weight=10.0,
                )
            )

        # Joint limit objective (starts after collisions)
        self.obj_joint_limits = ik.IKJointLimitObjective(
            joint_limit_lower=self.model.joint_limit_lower,
            joint_limit_upper=self.model.joint_limit_upper,
            weight=10.0,
        )

        # Variables the solver will update
        self.joint_q = wp.array(self.model.joint_q, shape=(1, self.model.joint_coord_count))

        self.ik_iters = 10
        self.ik_solver = ik.IKSolver(
            model=self.model,
            n_problems=1,
            objectives=[self.pos_obj, self.rot_obj, *self.collision_objs, self.obj_joint_limits],
            optimizer=ik.IKOptimizer.LBFGS,
            h0_scale=1.0,
            line_search_alphas=[0.01, 0.1, 0.5, 0.75, 1.0],
            history_len=12,
            jacobian_mode=ik.IKJacobianMode.MIXED,
        )

        self.capture()

    # ----------------------------------------------------------------------
    # Helpers
    # ----------------------------------------------------------------------
    def capture(self):
        self.graph = None
        if wp.get_device().is_cuda:
            with wp.ScopedCapture() as cap:
                self.simulate()
            self.graph = cap.graph

    def simulate(self):
        self.ik_solver.reset()
        self.ik_solver.step(self.joint_q, self.joint_q, iterations=self.ik_iters)

    def _push_targets_from_gizmos(self):
        """Read gizmo-updated transforms and push into IK objectives."""
        # Update EE target
        self.pos_obj.set_target_position(0, wp.transform_get_translation(self.ee_tf))
        q = wp.transform_get_rotation(self.ee_tf)
        self.rot_obj.set_target_rotation(0, wp.vec4(q[0], q[1], q[2], q[3]))

        # Update obstacle center from its gizmo
        c = wp.transform_get_translation(self.sphere_tf)
        self.collision_objs[0].set_obstacle_center(0, c)  # all objectives share the same array
        self._sync_obstacle_visual(c)

    def _sync_obstacle_visual(self, center: wp.vec3):
        """Move the world-attached sphere's transform to match the gizmo."""
        self.state.particle_q.fill_(center)

    # ----------------------------------------------------------------------
    # Template API
    # ----------------------------------------------------------------------
    def step(self):
        self._push_targets_from_gizmos()
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()
        self.sim_time += self.frame_dt

    def test_final(self):
        pass

    def render(self):
        self.viewer.begin_frame(self.sim_time)

        # Register EE and obstacle gizmos
        self.viewer.log_gizmo("target_tcp", self.ee_tf)
        self.viewer.log_gizmo("obstacle_sphere", self.sphere_tf)

        # Visualize the current articulated state + world-attached shapes
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state)

        self.viewer.log_state(self.state)

        self.viewer.end_frame()
        wp.synchronize()


if __name__ == "__main__":
    # Parse arguments and initialize viewer
    viewer, args = newton.examples.init()
    example = Example(viewer)
    newton.examples.run(example, args)
