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
# Example Basic Shapes
#
# Shows how to programmatically create a variety of
# collision shapes using the newton.ModelBuilder() API.
# Supports XPBD (default) and VBD solvers.
#
# Command: python -m newton.examples basic_shapes
# With VBD: python -m newton.examples basic_shapes --solver vbd
#
#
###########################################################################

import warp as wp
from pxr import Usd

import newton
import newton.examples
import newton.usd


class Example:
    def __init__(self, viewer, args):
        # setup simulation parameters first
        self.fps = 100
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = 10
        self.sim_dt = self.frame_dt / self.sim_substeps

        self.viewer = viewer
        self.solver_type = args.solver if hasattr(args, "solver") and args.solver else "xpbd"

        builder = newton.ModelBuilder()

        if self.solver_type == "vbd":
            # VBD: Higher stiffness for stable rigid body contacts
            builder.default_shape_cfg.ke = 1.0e6  # Contact stiffness
            builder.default_shape_cfg.kd = 1.0e1  # Contact damping
            builder.default_shape_cfg.mu = 0.5  # Friction coefficient

        # add ground plane
        builder.add_ground_plane()

        # z height to drop shapes from
        drop_z = 2.0

        # SPHERE
        self.sphere_pos = wp.vec3(0.0, -2.0, drop_z)
        body_sphere = builder.add_body(xform=wp.transform(p=self.sphere_pos, q=wp.quat_identity()), key="sphere")
        builder.add_shape_sphere(body_sphere, radius=0.5)

        # ELLIPSOID (flat disk shape: a=b > c for stability when resting on ground)
        self.ellipsoid_pos = wp.vec3(0.0, -6.0, drop_z)
        body_ellipsoid = builder.add_body(
            xform=wp.transform(p=self.ellipsoid_pos, q=wp.quat_identity()), key="ellipsoid"
        )
        builder.add_shape_ellipsoid(body_ellipsoid, a=0.5, b=0.5, c=0.25)

        # CAPSULE
        self.capsule_pos = wp.vec3(0.0, 0.0, drop_z)
        body_capsule = builder.add_body(xform=wp.transform(p=self.capsule_pos, q=wp.quat_identity()), key="capsule")
        builder.add_shape_capsule(body_capsule, radius=0.3, half_height=0.7)

        # CYLINDER
        self.cylinder_pos = wp.vec3(0.0, -4.0, drop_z)
        body_cylinder = builder.add_body(xform=wp.transform(p=self.cylinder_pos, q=wp.quat_identity()), key="cylinder")
        builder.add_shape_cylinder(body_cylinder, radius=0.4, half_height=0.6)

        # BOX
        self.box_pos = wp.vec3(0.0, 2.0, drop_z)
        body_box = builder.add_body(xform=wp.transform(p=self.box_pos, q=wp.quat_identity()), key="box")
        builder.add_shape_box(body_box, hx=0.5, hy=0.35, hz=0.25)

        # MESH (bunny)
        usd_stage = Usd.Stage.Open(newton.examples.get_asset("bunny.usd"))
        demo_mesh = newton.usd.get_mesh(usd_stage.GetPrimAtPath("/root/bunny"))

        self.mesh_pos = wp.vec3(0.0, 4.0, drop_z - 0.5)
        body_mesh = builder.add_body(xform=wp.transform(p=self.mesh_pos, q=wp.quat(0.5, 0.5, 0.5, 0.5)), key="mesh")
        builder.add_shape_mesh(body_mesh, mesh=demo_mesh)

        # CONE (no collision support in the standard collision pipeline)
        self.cone_pos = wp.vec3(0.0, 6.0, drop_z)
        body_cone = builder.add_body(xform=wp.transform(p=self.cone_pos, q=wp.quat_identity()), key="cone")
        builder.add_shape_cone(body_cone, radius=0.45, half_height=0.6)

        # Color rigid bodies for VBD solver
        if self.solver_type == "vbd":
            builder.color()

        # finalize model
        self.model = builder.finalize()

        # Create solver based on type
        if self.solver_type == "vbd":
            self.solver = newton.solvers.SolverVBD(
                self.model,
                iterations=10,
            )
        else:
            self.solver = newton.solvers.SolverXPBD(self.model, iterations=10)

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()

        # Create collision pipeline from command-line args (default: CollisionPipelineUnified with EXPLICIT)
        # Override rigid_contact_max_per_pair because mesh vs plane creates a lot of contacts
        self.collision_pipeline = newton.examples.create_collision_pipeline(
            self.model,
            args,
            rigid_contact_max_per_pair=100,
        )
        self.contacts = self.model.collide(self.state_0, collision_pipeline=self.collision_pipeline)

        self.viewer.set_model(self.model)

        self.capture()

    def capture(self):
        if wp.get_device().is_cuda:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph
        else:
            self.graph = None

    def simulate(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()

            # apply forces to the model
            self.viewer.apply_forces(self.state_0)

            self.contacts = self.model.collide(self.state_0, collision_pipeline=self.collision_pipeline)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)

            # swap states
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()

        self.sim_time += self.frame_dt

    def test_final(self):
        self.sphere_pos[2] = 0.5
        sphere_q = wp.transform(self.sphere_pos, wp.quat_identity())
        newton.examples.test_body_state(
            self.model,
            self.state_0,
            "sphere at rest pose",
            lambda q, qd: newton.utils.vec_allclose(q, sphere_q, atol=2e-4),
            [0],
        )
        # Ellipsoid with a=b=0.5, c=0.25 is stable (flat disk), rests at z=0.25
        self.ellipsoid_pos[2] = 0.25
        ellipsoid_q = wp.transform(self.ellipsoid_pos, wp.quat_identity())
        newton.examples.test_body_state(
            self.model,
            self.state_0,
            "ellipsoid at rest pose",
            lambda q, qd: newton.utils.vec_allclose(q, ellipsoid_q, atol=2e-2),
            [1],
        )
        self.capsule_pos[2] = 1.0
        capsule_q = wp.transform(self.capsule_pos, wp.quat_identity())
        newton.examples.test_body_state(
            self.model,
            self.state_0,
            "capsule at rest pose",
            lambda q, qd: newton.utils.vec_allclose(q, capsule_q, atol=2e-4),
            [2],
        )
        # Custom test for cylinder: allow 0.01 error for X and Y, strict for Z and rotation
        self.cylinder_pos[2] = 0.6
        cylinder_q = wp.transform(self.cylinder_pos, wp.quat_identity())
        newton.examples.test_body_state(
            self.model,
            self.state_0,
            "cylinder at rest pose",
            lambda q, qd: abs(q[0] - cylinder_q[0]) < 0.01
            and abs(q[1] - cylinder_q[1]) < 0.01
            and abs(q[2] - cylinder_q[2]) < 1e-4
            and abs(q[3] - cylinder_q[3]) < 1e-4
            and abs(q[4] - cylinder_q[4]) < 1e-4
            and abs(q[5] - cylinder_q[5]) < 1e-4
            and abs(q[6] - cylinder_q[6]) < 1e-4,
            [3],
        )
        self.box_pos[2] = 0.25
        box_q = wp.transform(self.box_pos, wp.quat_identity())
        newton.examples.test_body_state(
            self.model,
            self.state_0,
            "box at rest pose",
            lambda q, qd: newton.utils.vec_allclose(q, box_q, atol=0.1),
            [4],
        )
        # we only test that the bunny didn't fall through the ground and didn't slide too far
        newton.examples.test_body_state(
            self.model,
            self.state_0,
            "bunny at rest pose",
            lambda q, qd: q[2] > 0.01 and abs(q[0]) < 0.1 and abs(q[1] - 4.0) < 0.1,
            [5],
        )

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()


if __name__ == "__main__":
    # Extend the shared examples parser with a solver choice
    parser = newton.examples.create_parser()
    parser.add_argument(
        "--solver",
        type=str,
        default="xpbd",
        choices=["vbd", "xpbd"],
        help="Solver type: xpbd (default) or vbd",
    )

    viewer, args = newton.examples.init(parser)

    example = Example(viewer, args)

    newton.examples.run(example, args)
