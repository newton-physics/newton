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

import time
import unittest
from enum import Enum

import numpy as np
import warp as wp

import newton
from newton._src.geometry.sdf_hydroelastic import SDFHydroelasticConfig
from newton._src.geometry.utils import create_box_mesh
from newton.tests.unittest_utils import (
    add_function_test,
    get_selected_cuda_test_devices,
)

# --- Configuration ---


class ShapeType(Enum):
    PRIMITIVE = "primitive"
    MESH = "mesh"


# Scene parameters
CUBE_HALF_LARGE = 0.5  # 1m cube
CUBE_HALF_SMALL = 0.005  # 1cm cube
NUM_CUBES = 3

# Simulation parameters
SIM_SUBSTEPS = 10
SIM_DT = 1.0 / 60.0
SIM_TIME = 1.0
VIEWER_NUM_FRAMES = 300

# Test thresholds
POSITION_THRESHOLD_FACTOR = 0.1  # multiplied by cube_half
MAX_ROTATION_DEG = 10.0

# Devices and solvers
cuda_devices = get_selected_cuda_test_devices()

solvers = {
    "mujoco_warp": lambda model: newton.solvers.SolverMuJoCo(
        model,
        use_mujoco_cpu=False,
        use_mujoco_contacts=False,
        njmax=500,
        nconmax=200,
        solver="newton",
        ls_parallel=True,
        ls_iterations=100,
    ),
    "xpbd": lambda model: newton.solvers.SolverXPBD(model, iterations=10),
}


# --- Helper functions ---


def simulate(solver, model, state_0, state_1, control, contacts, collision_pipeline, sim_dt, substeps):
    for _ in range(substeps):
        state_0.clear_forces()
        contacts = model.collide(state_0, collision_pipeline=collision_pipeline)
        solver.step(state_0, state_1, control, contacts, sim_dt / substeps)
        state_0, state_1 = state_1, state_0
    return state_0, state_1


def build_stacked_cubes_scene(device, solver_fn, shape_type: ShapeType, cube_half: float = CUBE_HALF_LARGE):
    """Build the stacked cubes scene and return all components for simulation."""
    cube_mesh = None
    if shape_type == ShapeType.MESH:
        vertices, indices = create_box_mesh((cube_half, cube_half, cube_half))
        cube_mesh = newton.Mesh(vertices, indices)

    # Scale SDF parameters proportionally to cube size
    narrow_band = cube_half * 0.2
    contact_margin = cube_half * 0.2

    builder = newton.ModelBuilder()
    builder.default_shape_cfg = newton.ModelBuilder.ShapeConfig(
        sdf_max_resolution=32,
        is_hydroelastic=True,
        sdf_narrow_band_range=(-narrow_band, narrow_band),
        contact_margin=contact_margin,
    )

    builder.add_ground_plane()

    initial_positions = []
    for i in range(NUM_CUBES):
        z_pos = cube_half + i * cube_half * 2.0
        initial_positions.append(wp.vec3(0.0, 0.0, z_pos))
        body = builder.add_body(
            xform=wp.transform(initial_positions[-1], wp.quat_identity()),
            key=f"{shape_type.value}_cube_{i}",
        )

        if shape_type == ShapeType.PRIMITIVE:
            builder.add_shape_box(body=body, hx=cube_half, hy=cube_half, hz=cube_half)
        else:
            builder.add_shape_mesh(body=body, mesh=cube_mesh)

    model = builder.finalize(device=device)
    solver = solver_fn(model)

    state_0 = model.state()
    state_1 = model.state()
    control = model.control()

    newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)

    sdf_hydroelastic_config = SDFHydroelasticConfig(output_iso_vertices=True)

    collision_pipeline = newton.CollisionPipelineUnified.from_model(
        model,
        rigid_contact_max_per_pair=100,
        broad_phase_mode=newton.BroadPhaseMode.EXPLICIT,
        sdf_hydroelastic_config=sdf_hydroelastic_config,
    )

    return model, solver, state_0, state_1, control, collision_pipeline, initial_positions, cube_half


# --- Test functions ---


def run_stacked_cubes_hydroelastic_test(
    test, device, solver_fn, shape_type: ShapeType, cube_half: float = CUBE_HALF_LARGE
):
    """Shared test for stacking 3 cubes using hydroelastic contacts."""
    model, solver, state_0, state_1, control, collision_pipeline, initial_positions, cube_half = (
        build_stacked_cubes_scene(device, solver_fn, shape_type, cube_half)
    )

    contacts = model.collide(state_0, collision_pipeline=collision_pipeline)

    sdf_sdf_count = collision_pipeline.narrow_phase.shape_pairs_sdf_sdf_count.numpy()[0]
    test.assertEqual(sdf_sdf_count, NUM_CUBES - 1, f"Expected {NUM_CUBES - 1} sdf_sdf collisions, got {sdf_sdf_count}")

    num_frames = int(SIM_TIME / SIM_DT)

    for _ in range(num_frames):
        state_0, state_1 = simulate(
            solver, model, state_0, state_1, control, contacts, collision_pipeline, SIM_DT, SIM_SUBSTEPS
        )

    body_q = state_0.body_q.numpy()

    position_threshold = POSITION_THRESHOLD_FACTOR * cube_half

    for i in range(NUM_CUBES):
        expected_z = initial_positions[i][2]
        actual_pos = body_q[i, :3]
        displacement = np.linalg.norm(actual_pos - np.array([0.0, 0.0, expected_z]))

        test.assertLess(
            displacement,
            position_threshold,
            f"{shape_type.value.capitalize()} cube {i} moved {displacement:.6f}, exceeding threshold {position_threshold:.6f}",
        )

        initial_quat = np.array([0.0, 0.0, 0.0, 1.0])
        final_quat = body_q[i, 3:]
        dot_product = np.abs(np.dot(initial_quat, final_quat))
        dot_product = np.clip(dot_product, 0.0, 1.0)
        rotation_angle = 2.0 * np.arccos(dot_product)

        test.assertLess(
            rotation_angle,
            np.radians(MAX_ROTATION_DEG),
            f"{shape_type.value.capitalize()} cube {i} rotated {np.degrees(rotation_angle):.2f} degrees, exceeding threshold {MAX_ROTATION_DEG} degrees",
        )


def test_stacked_primitive_cubes_hydroelastic(test, device, solver_fn):
    """Test 3 primitive cubes (1m) stacked on each other remain stable for 1 second using hydroelastic contacts."""
    run_stacked_cubes_hydroelastic_test(test, device, solver_fn, ShapeType.PRIMITIVE, CUBE_HALF_LARGE)


def test_stacked_mesh_cubes_hydroelastic(test, device, solver_fn):
    """Test 3 mesh cubes (1m) stacked on each other remain stable for 1 second using hydroelastic contacts."""
    run_stacked_cubes_hydroelastic_test(test, device, solver_fn, ShapeType.MESH, CUBE_HALF_LARGE)


def test_stacked_small_primitive_cubes_hydroelastic(test, device, solver_fn):
    """Test 3 small primitive cubes (1cm) stacked on each other remain stable for 1 second using hydroelastic contacts."""
    run_stacked_cubes_hydroelastic_test(test, device, solver_fn, ShapeType.PRIMITIVE, CUBE_HALF_SMALL)


def test_stacked_small_mesh_cubes_hydroelastic(test, device, solver_fn):
    """Test 3 small mesh cubes (1cm) stacked on each other remain stable for 1 second using hydroelastic contacts."""
    run_stacked_cubes_hydroelastic_test(test, device, solver_fn, ShapeType.MESH, CUBE_HALF_SMALL)


# --- Test class ---


class TestHydroelastic(unittest.TestCase):
    @unittest.skip("Visual debugging - run manually to view simulation")
    def test_view_stacked_primitive_cubes(self):
        """View stacked primitive cubes simulation with hydroelastic contacts."""
        self._run_viewer_test(ShapeType.PRIMITIVE)

    @unittest.skip("Visual debugging - run manually to view simulation")
    def test_view_stacked_mesh_cubes(self):
        """View stacked mesh cubes simulation with hydroelastic contacts."""
        self._run_viewer_test(ShapeType.MESH)

    def _run_viewer_test(self, shape_type: ShapeType, solver_name: str = "xpbd", cube_half: float = CUBE_HALF_LARGE):
        device = wp.get_device("cuda:0")
        solver_fn = solvers[solver_name]

        model, solver, state_0, state_1, control, collision_pipeline, _, _ = build_stacked_cubes_scene(
            device, solver_fn, shape_type, cube_half
        )

        try:
            viewer = newton.viewer.ViewerGL()
            viewer.set_model(model)
        except Exception as e:
            self.skipTest(f"ViewerGL not available: {e}")
            return

        sim_time = 0.0
        contacts = model.collide(state_0, collision_pipeline=collision_pipeline)

        print(
            f"\nRunning {shape_type.value} cubes simulation with {solver_name} solver for {VIEWER_NUM_FRAMES} frames..."
        )
        print("Close the viewer window to stop.")

        try:
            for _frame in range(VIEWER_NUM_FRAMES):
                viewer.begin_frame(sim_time)
                viewer.log_state(state_0)
                viewer.log_contacts(contacts, state_0)
                viewer.log_isosurface(collision_pipeline.get_isosurface_data(), penetrating_only=False)
                viewer.end_frame()

                state_0, state_1 = simulate(
                    solver, model, state_0, state_1, control, contacts, collision_pipeline, SIM_DT, SIM_SUBSTEPS
                )

                sim_time += SIM_DT
                time.sleep(0.016)

        except KeyboardInterrupt:
            print("\nSimulation stopped by user.")


# --- Register tests ---

for solver_name, solver_fn in solvers.items():
    # Large cubes (1m)
    add_function_test(
        TestHydroelastic,
        f"test_stacked_primitive_cubes_hydroelastic_{solver_name}",
        test_stacked_primitive_cubes_hydroelastic,
        devices=cuda_devices,
        solver_fn=solver_fn,
    )

    add_function_test(
        TestHydroelastic,
        f"test_stacked_mesh_cubes_hydroelastic_{solver_name}",
        test_stacked_mesh_cubes_hydroelastic,
        devices=cuda_devices,
        solver_fn=solver_fn,
    )

    # Small cubes (1cm)
    add_function_test(
        TestHydroelastic,
        f"test_stacked_small_primitive_cubes_hydroelastic_{solver_name}",
        test_stacked_small_primitive_cubes_hydroelastic,
        devices=cuda_devices,
        solver_fn=solver_fn,
    )

    add_function_test(
        TestHydroelastic,
        f"test_stacked_small_mesh_cubes_hydroelastic_{solver_name}",
        test_stacked_small_mesh_cubes_hydroelastic,
        devices=cuda_devices,
        solver_fn=solver_fn,
    )


if __name__ == "__main__":
    unittest.main(verbosity=2, failfast=True)
