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

import unittest
from enum import IntFlag, auto

import warp as wp
import warp.examples

import newton
from newton import GeoType
from newton.examples import test_body_state
from newton.tests.unittest_utils import add_function_test, get_cuda_test_devices


class TestLevel(IntFlag):
    VELOCITY_X = auto()
    VELOCITY_YZ = auto()
    VELOCITY_LINEAR = VELOCITY_X | VELOCITY_YZ
    VELOCITY_ANGULAR = auto()
    STRICT = VELOCITY_LINEAR | VELOCITY_ANGULAR


def type_to_str(shape_type: GeoType):
    if shape_type == GeoType.SPHERE:
        return "sphere"
    elif shape_type == GeoType.BOX:
        return "box"
    elif shape_type == GeoType.CAPSULE:
        return "capsule"
    elif shape_type == GeoType.CYLINDER:
        return "cylinder"
    elif shape_type == GeoType.MESH:
        return "mesh"
    elif shape_type == GeoType.CONVEX_MESH:
        return "convex_hull"
    else:
        return "unknown"


class CollisionSetup:
    def __init__(
        self,
        viewer,
        device,
        shape_type_a,
        shape_type_b,
        solver_fn,
        sim_substeps,
        use_unified_pipeline=False,
    ):
        self.sim_substeps = sim_substeps
        self.frame_dt = 1 / 60
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0

        self.shape_type_a = shape_type_a
        self.shape_type_b = shape_type_b
        self.use_unified_pipeline = use_unified_pipeline

        self.builder = newton.ModelBuilder(gravity=0.0)
        self.builder.add_articulation()
        body_a = self.builder.add_body(xform=wp.transform(wp.vec3(-1.0, 0.0, 0.0)))
        self.add_shape(shape_type_a, body_a)
        self.builder.add_joint_free(body_a)

        self.init_velocity = 5.0
        self.builder.joint_qd[0] = self.builder.body_qd[-1][0] = self.init_velocity

        self.builder.add_articulation()
        body_b = self.builder.add_body(xform=wp.transform(wp.vec3(1.0, 0.0, 0.0)))
        self.add_shape(shape_type_b, body_b)
        self.builder.add_joint_free(body_b)

        self.model = self.builder.finalize(device=device)
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()

        # Initialize collision pipeline
        if use_unified_pipeline:
            self.collision_pipeline = newton.CollisionPipelineUnified.from_model(
                self.model,
                rigid_contact_max_per_pair=20,
                rigid_contact_margin=0.01,
                broad_phase_mode=newton.BroadPhaseMode.EXPLICIT,
            )
            self.contacts = self.model.collide(self.state_0, collision_pipeline=self.collision_pipeline)
        else:
            self.collision_pipeline = None
            self.contacts = self.model.collide(self.state_0)

        self.solver = solver_fn(self.model)

        self.viewer = viewer
        self.viewer.set_model(self.model)

        self.graph = None
        if wp.get_device(device).is_cuda:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph

    def add_shape(self, shape_type: GeoType, body: int):
        if shape_type == GeoType.BOX:
            self.builder.add_shape_box(body, key=type_to_str(shape_type))
        elif shape_type == GeoType.SPHERE:
            self.builder.add_shape_sphere(body, radius=0.5, key=type_to_str(shape_type))
        elif shape_type == GeoType.CAPSULE:
            self.builder.add_shape_capsule(body, radius=0.25, half_height=0.3, key=type_to_str(shape_type))
        elif shape_type == GeoType.CYLINDER:
            self.builder.add_shape_cylinder(body, radius=0.25, half_height=0.4, key=type_to_str(shape_type))
        elif shape_type == GeoType.MESH:
            # Use box mesh for unified pipeline (works correctly), sphere mesh for legacy pipeline (box mesh has issues)
            if self.use_unified_pipeline:
                vertices, indices = newton.utils.create_box_mesh(extents=(0.5, 0.5, 0.5))
            else:
                vertices, indices = newton.utils.create_sphere_mesh(radius=0.5)
            self.builder.add_shape_mesh(body, mesh=newton.Mesh(vertices[:, :3], indices), key=type_to_str(shape_type))
        elif shape_type == GeoType.CONVEX_MESH:
            # Use a sphere mesh as it's already convex
            vertices, indices = newton.utils.create_sphere_mesh(radius=0.5)
            mesh = newton.Mesh(vertices[:, :3], indices)
            self.builder.add_shape_convex_hull(body, mesh=mesh, key=type_to_str(shape_type))
        else:
            raise NotImplementedError(f"Shape type {shape_type} not implemented")

    def capture(self):
        if wp.get_device().is_cuda:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph
        else:
            self.graph = None

    def simulate(self):
        if self.use_unified_pipeline:
            self.contacts = self.model.collide(self.state_0, collision_pipeline=self.collision_pipeline)
        else:
            self.contacts = self.model.collide(self.state_0)

        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()

            # apply forces to the model
            self.viewer.apply_forces(self.state_0)

            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)

            # swap states
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

    def test(self, test_level: TestLevel, body: int, tolerance: float = 3e-3):
        body_name = f"body {body} ({self.model.shape_key[body]})"
        if test_level & TestLevel.VELOCITY_X:
            test_body_state(
                self.model,
                self.state_0,
                f"{body_name} is moving forward",
                lambda _q, qd: qd[0] > 0.03 and qd[0] <= wp.static(self.init_velocity),
                indices=[body],
                show_body_qd=True,
            )
        if test_level & TestLevel.VELOCITY_YZ:
            test_body_state(
                self.model,
                self.state_0,
                f"{body_name} has correct linear velocity",
                lambda _q, qd: abs(qd[1]) < tolerance and abs(qd[2]) < tolerance,
                indices=[body],
                show_body_qd=True,
            )
        if test_level & TestLevel.VELOCITY_ANGULAR:
            test_body_state(
                self.model,
                self.state_0,
                f"{body_name} has correct angular velocity",
                lambda _q, qd: abs(qd[3]) < tolerance and abs(qd[4]) < tolerance and abs(qd[5]) < tolerance,
                indices=[body],
                show_body_qd=True,
            )


devices = get_cuda_test_devices(mode="basic")


class TestCollisionPipeline(unittest.TestCase):
    pass


# Note that body A does sometimes bounce off body B or continue moving forward
# due to inertia differences, so we only test linear velocity along the Y and Z directions.
# Some collisions also cause unwanted angular velocity, so we only test linear velocity
# for those cases.
contact_tests = [
    (GeoType.SPHERE, GeoType.SPHERE, TestLevel.VELOCITY_YZ, TestLevel.STRICT),
    (GeoType.SPHERE, GeoType.BOX, TestLevel.VELOCITY_YZ, TestLevel.STRICT),
    (GeoType.SPHERE, GeoType.CAPSULE, TestLevel.VELOCITY_YZ, TestLevel.STRICT),
    (GeoType.SPHERE, GeoType.MESH, TestLevel.VELOCITY_YZ, TestLevel.STRICT),
    (GeoType.BOX, GeoType.BOX, TestLevel.VELOCITY_YZ, TestLevel.VELOCITY_LINEAR),
    (GeoType.BOX, GeoType.MESH, TestLevel.VELOCITY_YZ, TestLevel.STRICT),
    (GeoType.CAPSULE, GeoType.CAPSULE, TestLevel.VELOCITY_YZ, TestLevel.VELOCITY_LINEAR),
    (GeoType.CAPSULE, GeoType.MESH, TestLevel.VELOCITY_YZ, TestLevel.STRICT),
    (GeoType.MESH, GeoType.MESH, TestLevel.VELOCITY_YZ, TestLevel.STRICT),
]


def test_collision_pipeline(
    _test, device, shape_type_a: GeoType, shape_type_b: GeoType, test_level_a: TestLevel, test_level_b: TestLevel
):
    viewer = newton.viewer.ViewerNull()
    setup = CollisionSetup(
        viewer=viewer,
        device=device,
        solver_fn=newton.solvers.SolverXPBD,
        sim_substeps=10,
        shape_type_a=shape_type_a,
        shape_type_b=shape_type_b,
    )
    for _ in range(200):
        setup.step()
        setup.render()
    setup.test(test_level_a, 0)
    setup.test(test_level_b, 1)


for shape_type_a, shape_type_b, test_level_a, test_level_b in contact_tests:
    add_function_test(
        TestCollisionPipeline,
        f"test_{type_to_str(shape_type_a)}_{type_to_str(shape_type_b)}",
        test_collision_pipeline,
        devices=devices,
        shape_type_a=shape_type_a,
        shape_type_b=shape_type_b,
        test_level_a=test_level_a,
        test_level_b=test_level_b,
    )


class TestUnifiedCollisionPipeline(unittest.TestCase):
    pass


# Unified collision pipeline tests - now supports both MESH and CONVEX_MESH
# Note: MESH vs MESH is not yet supported
unified_contact_tests = [
    (GeoType.SPHERE, GeoType.SPHERE, TestLevel.VELOCITY_YZ, TestLevel.STRICT),
    (GeoType.SPHERE, GeoType.BOX, TestLevel.VELOCITY_YZ, TestLevel.STRICT),
    (GeoType.SPHERE, GeoType.CAPSULE, TestLevel.VELOCITY_YZ, TestLevel.STRICT),
    (GeoType.SPHERE, GeoType.MESH, TestLevel.VELOCITY_YZ, TestLevel.STRICT),
    (GeoType.SPHERE, GeoType.CONVEX_MESH, TestLevel.VELOCITY_YZ, TestLevel.STRICT),
    (GeoType.BOX, GeoType.BOX, TestLevel.VELOCITY_YZ, TestLevel.VELOCITY_LINEAR),
    (GeoType.BOX, GeoType.MESH, TestLevel.VELOCITY_YZ, TestLevel.STRICT),
    (GeoType.BOX, GeoType.CONVEX_MESH, TestLevel.VELOCITY_YZ, TestLevel.STRICT),
    (GeoType.CAPSULE, GeoType.CAPSULE, TestLevel.VELOCITY_YZ, TestLevel.VELOCITY_LINEAR),
    (GeoType.CAPSULE, GeoType.MESH, TestLevel.VELOCITY_YZ, TestLevel.STRICT),
    (GeoType.CAPSULE, GeoType.CONVEX_MESH, TestLevel.VELOCITY_YZ, TestLevel.STRICT),
    # (GeoType.MESH, GeoType.MESH, TestLevel.VELOCITY_YZ, TestLevel.STRICT),  # Not yet supported
    (GeoType.MESH, GeoType.CONVEX_MESH, TestLevel.VELOCITY_YZ, TestLevel.STRICT),
    (GeoType.CONVEX_MESH, GeoType.CONVEX_MESH, TestLevel.VELOCITY_YZ, TestLevel.STRICT),
]


def test_unified_collision_pipeline(
    _test, device, shape_type_a: GeoType, shape_type_b: GeoType, test_level_a: TestLevel, test_level_b: TestLevel
):
    viewer = newton.viewer.ViewerNull()
    setup = CollisionSetup(
        viewer=viewer,
        device=device,
        solver_fn=newton.solvers.SolverXPBD,
        sim_substeps=10,
        shape_type_a=shape_type_a,
        shape_type_b=shape_type_b,
        use_unified_pipeline=True,
    )
    for _ in range(200):
        setup.step()
        setup.render()
    setup.test(test_level_a, 0)
    setup.test(test_level_b, 1)


for shape_type_a, shape_type_b, test_level_a, test_level_b in unified_contact_tests:
    add_function_test(
        TestUnifiedCollisionPipeline,
        f"test_{type_to_str(shape_type_a)}_{type_to_str(shape_type_b)}",
        test_unified_collision_pipeline,
        devices=devices,
        shape_type_a=shape_type_a,
        shape_type_b=shape_type_b,
        test_level_a=test_level_a,
        test_level_b=test_level_b,
    )

if __name__ == "__main__":
    # wp.clear_kernel_cache()
    unittest.main(verbosity=2, failfast=False)
