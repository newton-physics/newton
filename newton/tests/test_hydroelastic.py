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
from newton.geometry import HydroelasticSDF
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
        ls_iterations=100,
    ),
    "xpbd": lambda model: newton.solvers.SolverXPBD(model, iterations=10),
}


# --- Helper functions ---


def simulate(solver, model, state_0, state_1, control, contacts, collision_pipeline, sim_dt, substeps):
    for _ in range(substeps):
        state_0.clear_forces()
        collision_pipeline.collide(state_0, contacts)
        solver.step(state_0, state_1, control, contacts, sim_dt / substeps)
        state_0, state_1 = state_1, state_0
    return state_0, state_1


def build_stacked_cubes_scene(
    device, solver_fn, shape_type: ShapeType, cube_half: float = CUBE_HALF_LARGE, reduce_contacts: bool = True
):
    """Build the stacked cubes scene and return all components for simulation."""
    cube_mesh = None
    if shape_type == ShapeType.MESH:
        cube_mesh = newton.Mesh.create_box(
            cube_half,
            cube_half,
            cube_half,
            duplicate_vertices=False,
            compute_normals=False,
            compute_uvs=False,
            compute_inertia=False,
        )

    # Scale SDF parameters proportionally to cube size
    narrow_band = cube_half * 0.2
    contact_margin = cube_half * 0.2

    if cube_mesh is not None:
        cube_mesh.build_sdf(
            max_resolution=32,
            narrow_band_range=(-narrow_band, narrow_band),
            margin=contact_margin,
        )

    builder = newton.ModelBuilder()
    if shape_type == ShapeType.PRIMITIVE:
        builder.default_shape_cfg = newton.ModelBuilder.ShapeConfig(
            thickness=1e-5,
            mu=0.5,
            sdf_max_resolution=32,
            is_hydroelastic=True,
            sdf_narrow_band_range=(-narrow_band, narrow_band),
            contact_margin=contact_margin,
        )
    else:
        builder.default_shape_cfg = newton.ModelBuilder.ShapeConfig(
            thickness=1e-5,
            mu=0.5,
            is_hydroelastic=True,
            contact_margin=contact_margin,
        )

    builder.add_ground_plane()

    initial_positions = []
    for i in range(NUM_CUBES):
        z_pos = cube_half + i * cube_half * 2.0
        initial_positions.append(wp.vec3(0.0, 0.0, z_pos))
        body = builder.add_body(
            xform=wp.transform(initial_positions[-1], wp.quat_identity()),
            label=f"{shape_type.value}_cube_{i}",
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

    sdf_hydroelastic_config = HydroelasticSDF.Config(
        output_contact_surface=True, reduce_contacts=reduce_contacts, anchor_contact=True
    )

    # Hydroelastic without contact reduction can generate many contacts
    rigid_contact_max = 6000 if not reduce_contacts else 100

    collision_pipeline = newton.CollisionPipeline(
        model,
        rigid_contact_max=rigid_contact_max,
        broad_phase="explicit",
        sdf_hydroelastic_config=sdf_hydroelastic_config,
    )

    return model, solver, state_0, state_1, control, collision_pipeline, initial_positions, cube_half


# --- Test functions ---


def run_stacked_cubes_hydroelastic_test(
    test, device, solver_fn, shape_type: ShapeType, cube_half: float = CUBE_HALF_LARGE, reduce_contacts: bool = True
):
    """Shared test for stacking 3 cubes using hydroelastic contacts."""
    model, solver, state_0, state_1, control, collision_pipeline, initial_positions, cube_half = (
        build_stacked_cubes_scene(device, solver_fn, shape_type, cube_half, reduce_contacts)
    )

    contacts = collision_pipeline.contacts()
    collision_pipeline.collide(state_0, contacts)

    sdf_sdf_count = collision_pipeline.narrow_phase.shape_pairs_sdf_sdf_count.numpy()[0]
    test.assertEqual(sdf_sdf_count, NUM_CUBES - 1, f"Expected {NUM_CUBES - 1} sdf_sdf collisions, got {sdf_sdf_count}")

    num_frames = int(SIM_TIME / SIM_DT)

    # Scale substeps for small objects - they need smaller time steps for stability
    substeps = SIM_SUBSTEPS if cube_half >= CUBE_HALF_LARGE else 20

    for _ in range(num_frames):
        state_0, state_1 = simulate(
            solver, model, state_0, state_1, control, contacts, collision_pipeline, SIM_DT, substeps
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


def test_stacked_mesh_cubes_hydroelastic(test, device, solver_fn):
    """Test 3 mesh cubes (1m) stacked on each other remain stable for 1 second using hydroelastic contacts."""
    run_stacked_cubes_hydroelastic_test(test, device, solver_fn, ShapeType.MESH, CUBE_HALF_LARGE)


def test_stacked_small_primitive_cubes_hydroelastic(test, device, solver_fn):
    """Test 3 small primitive cubes (1cm) stacked on each other remain stable for 1 second using hydroelastic contacts."""
    run_stacked_cubes_hydroelastic_test(test, device, solver_fn, ShapeType.PRIMITIVE, CUBE_HALF_SMALL)


def test_stacked_small_mesh_cubes_hydroelastic(test, device, solver_fn):
    """Test 3 small mesh cubes (1cm) stacked on each other remain stable for 1 second using hydroelastic contacts."""
    run_stacked_cubes_hydroelastic_test(test, device, solver_fn, ShapeType.MESH, CUBE_HALF_SMALL)


def test_stacked_primitive_cubes_hydroelastic_no_reduction(test, device, solver_fn):
    """Test 3 primitive cubes (1m) stacked without contact reduction using hydroelastic contacts."""
    run_stacked_cubes_hydroelastic_test(test, device, solver_fn, ShapeType.PRIMITIVE, CUBE_HALF_LARGE, False)


def test_mujoco_hydroelastic_penetration_depth(test, device):
    """Test that hydroelastic penetration depth matches expectation.

    Creates 4 box pairs with different kh and area combinations:
    - Case 0: k=1e8, area=0.01 (small stiffness, small area)
    - Case 1: k=1e9, area=0.01 (large stiffness, small area)
    - Case 2: k=1e8, area=0.0225 (small stiffness, large area)
    - Case 3: k=1e9, area=0.0225 (large stiffness, large area)
    """
    # Test parameters
    box_size_lower = 0.2
    box_half_lower = box_size_lower / 2.0
    mass_lower = 1.0
    mass_upper = 0.5
    gravity = 10.0
    external_force = 20.0

    # 4 test cases: (kh, upper_box_size)
    test_cases = [
        (1e8, 0.1),
        (1e9, 0.1),
        (1e8, 0.15),
        (1e9, 0.15),
    ]

    # Inertia for lower box
    inertia_lower = (1.0 / 6.0) * mass_lower * box_size_lower * box_size_lower
    I_m_lower = wp.mat33(inertia_lower, 0.0, 0.0, 0.0, inertia_lower, 0.0, 0.0, 0.0, inertia_lower)

    builder = newton.ModelBuilder(gravity=-gravity)

    lower_body_indices = []
    upper_body_indices = []
    lower_shape_indices = []
    upper_shape_indices = []
    initial_upper_positions = []
    areas = []
    kh_values = []

    spacing = 0.5

    for i, (kh_val, upper_size) in enumerate(test_cases):
        upper_half = upper_size / 2.0
        area = upper_size * upper_size
        areas.append(area)
        kh_values.append(0.5 * kh_val)  # effective stiffness for two equal k shapes

        # Inertia for this upper box
        inertia_upper = (1.0 / 6.0) * mass_upper * upper_size * upper_size
        I_m_upper = wp.mat33(inertia_upper, 0.0, 0.0, 0.0, inertia_upper, 0.0, 0.0, 0.0, inertia_upper)

        shape_cfg = newton.ModelBuilder.ShapeConfig(
            thickness=1e-5,
            sdf_max_resolution=64,
            is_hydroelastic=True,
            sdf_narrow_band_range=(-0.1, 0.1),
            contact_margin=0.01,
            kh=kh_val,
            density=0.0,
        )

        x_pos = (i - len(test_cases) / 2) * spacing

        # Lower box
        lower_pos = wp.vec3(x_pos, 0.0, box_half_lower)
        body_lower = builder.add_body(
            xform=wp.transform(p=lower_pos, q=wp.quat_identity()),
            label=f"lower_{i}",
            mass=mass_lower,
            inertia=I_m_lower,
        )
        shape_lower = builder.add_shape_box(
            body_lower, hx=box_half_lower, hy=box_half_lower, hz=box_half_lower, cfg=shape_cfg
        )
        lower_body_indices.append(body_lower)
        lower_shape_indices.append(shape_lower)

        # Upper box
        expected_dist = box_half_lower + upper_half
        upper_z = box_half_lower + expected_dist
        upper_pos = wp.vec3(x_pos, 0.0, upper_z)
        body_upper = builder.add_body(
            xform=wp.transform(p=upper_pos, q=wp.quat_identity()),
            label=f"upper_{i}",
            mass=mass_upper,
            inertia=I_m_upper,
        )
        shape_upper = builder.add_shape_box(body_upper, hx=upper_half, hy=upper_half, hz=upper_half, cfg=shape_cfg)
        upper_body_indices.append(body_upper)
        upper_shape_indices.append(shape_upper)
        initial_upper_positions.append(np.array([x_pos, 0.0, upper_z]))

    builder.add_ground_plane()
    model = builder.finalize(device=device)

    solver = newton.solvers.SolverMuJoCo(
        model,
        use_mujoco_contacts=False,
        solver="newton",
        integrator="implicitfast",
        cone="elliptic",
        njmax=2000,
        nconmax=2000,
        iterations=20,
        ls_iterations=100,
        impratio=1000.0,
    )

    state_0 = model.state()
    state_1 = model.state()
    control = model.control()

    newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)

    sdf_config = HydroelasticSDF.Config(output_contact_surface=True)
    collision_pipeline = newton.CollisionPipeline(
        model,
        broad_phase="explicit",
        sdf_hydroelastic_config=sdf_config,
    )
    # Enable contact surface output for this test (validates penetration depth)
    collision_pipeline.set_output_contact_surface(True)
    contacts = collision_pipeline.contacts()

    # Simulate for 3 seconds to reach equilibrium
    sim_dt = 1.0 / 60.0
    substeps = 10
    sim_time = 3.0
    num_frames = int(sim_time / sim_dt)
    total_steps = num_frames * substeps

    # Pre-compute forces as a Warp array
    forces_np = np.zeros(model.body_count * 6, dtype=np.float32)
    for body_idx in upper_body_indices:
        forces_np[body_idx * 6 + 2] = -external_force
    precomputed_forces = wp.array(forces_np.reshape(model.body_count, 6), dtype=wp.spatial_vector, device=device)

    for _ in range(total_steps):
        wp.copy(state_0.body_f, precomputed_forces)
        collision_pipeline.collide(state_0, contacts)
        solver.step(state_0, state_1, control, contacts, sim_dt / substeps)
        state_0, state_1 = state_1, state_0

    # Check that upper cubes are near their original positions
    body_q = state_0.body_q.numpy()
    position_tolerance = 0.001

    for i in range(len(test_cases)):
        body_idx = upper_body_indices[i]
        final_pos = body_q[body_idx, :3]
        initial_pos = initial_upper_positions[i]
        displacement = np.linalg.norm(final_pos - initial_pos)

        test.assertLess(
            displacement,
            position_tolerance,
            f"Case {i}: Upper cube moved {displacement:.4f}m from initial position, exceeds {position_tolerance}m tolerance",
        )

    # Measure penetration from contact surface depth
    surface_data = collision_pipeline.get_hydro_contact_surface()
    test.assertIsNotNone(surface_data, "Hydroelastic contact surface data should be available")

    num_faces = int(surface_data.face_contact_count.numpy()[0])
    test.assertGreater(num_faces, 0, "Should have face contacts")

    depths = surface_data.contact_surface_depth.numpy()[:num_faces]
    shape_pairs = surface_data.contact_surface_shape_pair.numpy()[:num_faces]

    # Calculate expected and measured penetration for each case
    total_force = gravity * mass_upper + external_force
    effective_mass = (mass_lower * mass_upper) / (mass_lower + mass_upper)

    for i in range(len(test_cases)):
        lower_shape = lower_shape_indices[i]
        upper_shape = upper_shape_indices[i]
        kh_val = kh_values[i]
        area = areas[i]

        # Filter depths for this shape pair
        mask = ((shape_pairs[:, 0] == lower_shape) & (shape_pairs[:, 1] == upper_shape)) | (
            (shape_pairs[:, 0] == upper_shape) & (shape_pairs[:, 1] == lower_shape)
        )
        instance_depths = depths[mask]
        # Standard convention: negative depth = penetrating
        instance_depths = instance_depths[instance_depths < 0]

        test.assertGreater(len(instance_depths), 0, f"Case {i} should have penetrating contacts (negative depth)")

        # x2 because depth is distance to isosurface; use |depth| for magnitude
        measured = 2.0 * np.mean(-instance_depths)

        # Expected: depth = F / (k_eff * A_eff) / mujoco_scaling
        effective_area = area
        expected = total_force / (kh_val * effective_area)
        expected /= effective_mass
        ratio = measured / expected

        test.assertGreater(
            ratio, 0.9, f"Case {i}: ratio {ratio:.3f} too low (measured={measured:.6f}, expected={expected:.6f})"
        )
        test.assertLess(
            ratio, 1.1, f"Case {i}: ratio {ratio:.3f} too high (measured={measured:.6f}, expected={expected:.6f})"
        )


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
        contacts = collision_pipeline.contacts()
        collision_pipeline.collide(state_0, contacts)

        print(
            f"\nRunning {shape_type.value} cubes simulation with {solver_name} solver for {VIEWER_NUM_FRAMES} frames..."
        )
        print("Close the viewer window to stop.")

        try:
            for _frame in range(VIEWER_NUM_FRAMES):
                viewer.begin_frame(sim_time)
                viewer.log_state(state_0)
                viewer.log_contacts(contacts, state_0)
                viewer.log_hydro_contact_surface(collision_pipeline.get_hydro_contact_surface(), penetrating_only=False)
                viewer.end_frame()

                state_0, state_1 = simulate(
                    solver, model, state_0, state_1, control, contacts, collision_pipeline, SIM_DT, SIM_SUBSTEPS
                )

                sim_time += SIM_DT
                time.sleep(0.016)

        except KeyboardInterrupt:
            print("\nSimulation stopped by user.")


# --- Register tests ---

add_function_test(
    TestHydroelastic,
    "test_stacked_small_primitive_cubes_hydroelastic_mujoco_warp",
    test_stacked_small_primitive_cubes_hydroelastic,
    devices=cuda_devices,
    solver_fn=solvers["mujoco_warp"],
)

add_function_test(
    TestHydroelastic,
    "test_stacked_small_mesh_cubes_hydroelastic_xpbd",
    test_stacked_small_mesh_cubes_hydroelastic,
    devices=cuda_devices,
    solver_fn=solvers["xpbd"],
)

add_function_test(
    TestHydroelastic,
    "test_stacked_primitive_cubes_hydroelastic_xpbd_no_reduction",
    test_stacked_primitive_cubes_hydroelastic_no_reduction,
    devices=cuda_devices,
    solver_fn=solvers["xpbd"],
)

# Penetration depth validation test
add_function_test(
    TestHydroelastic,
    "test_mujoco_hydroelastic_penetration_depth",
    test_mujoco_hydroelastic_penetration_depth,
    devices=cuda_devices,
)


if __name__ == "__main__":
    unittest.main(verbosity=2, failfast=True)
