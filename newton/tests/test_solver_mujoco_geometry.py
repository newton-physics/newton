# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import unittest
import warnings

import numpy as np
import warp as wp

import newton
from newton import Mesh, ModelFlags
from newton._src.core.types import vec5
from newton.solvers import SolverMuJoCo
from newton.tests.unittest_utils import USD_AVAILABLE
from newton.tests.utils.mujoco import _MuJoCoSolverPropertiesFixture


class TestMuJoCoSolverCollisionMasks(unittest.TestCase):
    def test_large_graph_skips_before_pair_enumeration(self):
        """Skip large mask graphs before enumerating every shape pair."""

        class ShapeGroups:
            def numpy(self):
                return np.ones(257, dtype=np.int32)

        class ModelStub:
            shape_collision_group = ShapeGroups()

            def shape_collision_filter_mask(self, _pairs):
                raise AssertionError("large graphs must skip before querying candidate pairs")

        result = SolverMuJoCo._compile_newton_collision_masks(
            ModelStub(),
            np.arange(257, dtype=np.int32),
        )

        self.assertTrue(result.skipped)

    def test_sparse_filters_query_only_selected_pairs(self):
        """Query filters only for pairs in the selected collision graph."""

        class ShapeGroups:
            def numpy(self):
                return np.ones(5, dtype=np.int32)

        class ModelStub:
            shape_collision_group = ShapeGroups()

            def shape_collision_filter_pairs_array(self):
                raise AssertionError("selected graphs must not materialize every sparse filter")

            def shape_collision_filter_mask(self, pairs):
                np.testing.assert_array_equal(
                    pairs,
                    np.array([[1, 3], [1, 4], [3, 4]], dtype=np.int32),
                )
                return np.array([True, False, False])

        result = SolverMuJoCo._compile_newton_collision_masks(
            ModelStub(),
            np.array([1, 3, 4], dtype=np.int32),
        )

        actual = ((result.collision_type[:, None] & result.collision_affinity[None, :]) != 0) | (
            (result.collision_type[None, :] & result.collision_affinity[:, None]) != 0
        )
        expected = np.array(
            [
                [False, False, True],
                [False, False, True],
                [True, True, False],
            ]
        )
        np.testing.assert_array_equal(actual, expected)

    def test_native_newton_graph_compiles_exact_masks(self):
        """Compile native signed groups and pair filters into exact MuJoCo masks."""
        builder = newton.ModelBuilder()
        groups = [0, 1, 1, 2, 2, -3, -3, -7]
        shapes = []
        for index, group in enumerate(groups):
            body = builder.add_link(label=f"body_{index}")
            joint = builder.add_joint_free(parent=-1, child=body, label=f"joint_{index}")
            builder.add_articulation([joint])
            cfg = newton.ModelBuilder.ShapeConfig(collision_group=group)
            shapes.append(builder.add_shape_sphere(body, radius=0.1, cfg=cfg, label=f"shape_{index}"))
        builder.add_shape_collision_filter_pair(shapes[1], shapes[2])
        builder.add_shape_collision_filter_pair(shapes[3], shapes[5])

        model = builder.finalize(device="cpu")
        solver = SolverMuJoCo(model, use_mujoco_contacts=True)
        mapping = solver.mjc_geom_to_newton_shape.numpy()[0]
        contype = solver.mj_model.geom_contype
        conaffinity = solver.mj_model.geom_conaffinity

        for geom_a in range(solver.mj_model.ngeom - 1):
            shape_a = int(mapping[geom_a])
            for geom_b in range(geom_a + 1, solver.mj_model.ngeom):
                shape_b = int(mapping[geom_b])
                group_a = groups[shape_a]
                group_b = groups[shape_b]
                if group_a == 0 or group_b == 0:
                    expected = False
                elif group_a > 0:
                    expected = group_a == group_b or group_b < 0
                else:
                    expected = group_a != group_b
                expected = expected and not model.shape_collision_filter_contains(shape_a, shape_b)
                actual = bool(
                    (int(contype[geom_a]) & int(conaffinity[geom_b]))
                    or (int(contype[geom_b]) & int(conaffinity[geom_a]))
                )
                self.assertEqual(actual, expected, f"MuJoCo pair {(shape_a, shape_b)}")

    def test_falls_back_when_exact_masks_exceed_32_bits(self):
        """Fall back for a graph that provably needs 33 mask bits."""
        builder = newton.ModelBuilder()
        for index, group in enumerate(np.repeat(np.arange(1, 34), 2)):
            body = builder.add_link(label=f"body_{index}")
            joint = builder.add_joint_free(parent=-1, child=body, label=f"joint_{index}")
            builder.add_articulation([joint])
            cfg = newton.ModelBuilder.ShapeConfig(collision_group=int(group))
            builder.add_shape_sphere(body, radius=0.1, cfg=cfg, label=f"shape_{index}")

        with warnings.catch_warnings():
            warnings.filterwarnings("error", message=r"The selected Newton collision graph.*")
            solver = SolverMuJoCo(builder.finalize(device="cpu"), use_mujoco_contacts=True)
        self.assertEqual(solver.mj_model.ngeom, 66)


class TestMuJoCoSolverGeomProperties(unittest.TestCase):
    def test_geom_properties_conversion_and_update(self):
        """Verify initial and updated geometry properties on the same model."""
        _MuJoCoSolverPropertiesFixture.setUp(self)
        solver = SolverMuJoCo(self.model, iterations=1, disable_contacts=True)
        self.assertTrue(hasattr(solver, "mjc_geom_to_newton_shape"))
        mjc_geom_to_newton_shape = solver.mjc_geom_to_newton_shape.numpy()
        shape_types = self.model.shape_type.numpy()
        num_geoms = solver.mj_model.ngeom
        shape_mu = self.model.shape_material_mu.numpy()
        shape_ke = self.model.shape_material_ke.numpy()
        shape_kd = self.model.shape_material_kd.numpy()
        shape_sizes = self.model.shape_scale.numpy()
        shape_transforms = self.model.shape_transform.numpy()
        geom_friction = solver.mjw_model.geom_friction.numpy()
        geom_solref = solver.mjw_model.geom_solref.numpy()
        geom_size = solver.mjw_model.geom_size.numpy()
        geom_pos = solver.mjw_model.geom_pos.numpy()
        geom_quat = solver.mjw_model.geom_quat.numpy()
        tested_count = 0
        for world_idx in range(self.model.world_count):
            for geom_idx in range(num_geoms):
                shape_idx = mjc_geom_to_newton_shape[world_idx, geom_idx]
                if shape_idx < 0:  # No mapping for this geom
                    continue

                tested_count += 1
                shape_type = shape_types[shape_idx]

                # Test 1: Friction conversion
                expected_mu = shape_mu[shape_idx]
                actual_friction = geom_friction[world_idx, geom_idx]

                # Slide friction should match exactly
                self.assertAlmostEqual(
                    float(actual_friction[0]),
                    expected_mu,
                    places=5,
                    msg=f"Slide friction mismatch for shape {shape_idx} (type {shape_type}) in world {world_idx}, geom {geom_idx}",
                )

                # Torsional and rolling friction should be absolute values (not scaled by mu)
                expected_torsional = self.model.shape_material_mu_torsional.numpy()[shape_idx]
                expected_rolling = self.model.shape_material_mu_rolling.numpy()[shape_idx]

                self.assertAlmostEqual(
                    float(actual_friction[1]),
                    expected_torsional,
                    places=5,
                    msg=f"Torsional friction mismatch for shape {shape_idx} in world {world_idx}, geom {geom_idx}",
                )

                self.assertAlmostEqual(
                    float(actual_friction[2]),
                    expected_rolling,
                    places=5,
                    msg=f"Rolling friction mismatch for shape {shape_idx} in world {world_idx}, geom {geom_idx}",
                )

                # Test 2: Contact parameters (solref)
                actual_solref = geom_solref[world_idx, geom_idx]

                # Compute expected solref based on Newton's conversion logic
                ke = shape_ke[shape_idx]
                kd = shape_kd[shape_idx]

                if ke > 0.0 and kd > 0.0:
                    timeconst = 2.0 / kd
                    dampratio = np.sqrt(1.0 / (timeconst * timeconst * ke))
                    expected_solref = (timeconst, dampratio)
                else:
                    expected_solref = (0.02, 1.0)

                self.assertAlmostEqual(
                    float(actual_solref[0]),
                    expected_solref[0],
                    places=5,
                    msg=f"Solref[0] mismatch for shape {shape_idx} in world {world_idx}, geom {geom_idx}",
                )

                self.assertAlmostEqual(
                    float(actual_solref[1]),
                    expected_solref[1],
                    places=5,
                    msg=f"Solref[1] mismatch for shape {shape_idx} in world {world_idx}, geom {geom_idx}",
                )

                # Test 3: Size
                actual_size = geom_size[world_idx, geom_idx]
                expected_size = shape_sizes[shape_idx]
                for dim in range(3):
                    if expected_size[dim] > 0:  # Only check non-zero dimensions
                        self.assertAlmostEqual(
                            float(actual_size[dim]),
                            float(expected_size[dim]),
                            places=5,
                            msg=f"Size mismatch for shape {shape_idx} in world {world_idx}, geom {geom_idx}, dimension {dim}",
                        )

                # Test 4: Position and orientation (body-local coordinates)
                actual_pos = geom_pos[world_idx, geom_idx]
                actual_quat = geom_quat[world_idx, geom_idx]

                # Get expected transform from Newton (body-local coordinates)
                shape_transform = wp.transform(*shape_transforms[shape_idx])
                expected_pos = wp.vec3(*shape_transform.p)
                expected_quat = wp.quat(*shape_transform.q)

                # Convert expected quaternion to MuJoCo format (wxyz)
                expected_quat_mjc = np.array([expected_quat.w, expected_quat.x, expected_quat.y, expected_quat.z])

                # Test position
                for dim in range(3):
                    self.assertAlmostEqual(
                        float(actual_pos[dim]),
                        float(expected_pos[dim]),
                        places=5,
                        msg=f"Position mismatch for shape {shape_idx} in world {world_idx}, geom {geom_idx}, dimension {dim}",
                    )

                # Test quaternion
                for dim in range(4):
                    self.assertAlmostEqual(
                        float(actual_quat[dim]),
                        float(expected_quat_mjc[dim]),
                        places=5,
                        msg=f"Quaternion mismatch for shape {shape_idx} in world {world_idx}, geom {geom_idx}, component {dim}",
                    )
        self.assertGreater(tested_count, 0, "Should have tested at least one shape")
        with self.subTest(property="runtime shape update"):
            mjc_geom_to_newton_shape = solver.mjc_geom_to_newton_shape.numpy()
            num_geoms = solver.mj_model.ngeom
            solver.step(self.state_in, self.state_out, self.control, self.contacts, 0.01)
            self.state_in, self.state_out = self.state_out, self.state_in
            initial_friction = solver.mjw_model.geom_friction.numpy().copy()
            initial_solref = solver.mjw_model.geom_solref.numpy().copy()
            initial_size = solver.mjw_model.geom_size.numpy().copy()
            initial_pos = solver.mjw_model.geom_pos.numpy().copy()
            initial_quat = solver.mjw_model.geom_quat.numpy().copy()
            shape_count = self.model.shape_count
            new_mu = np.zeros(shape_count)
            new_torsional = np.zeros(shape_count)
            new_rolling = np.zeros(shape_count)
            for i in range(shape_count):
                new_mu[i] = 1.0 + (i + 1) * 0.05  # Pattern: 1.05, 1.10, ...
                new_torsional[i] = 0.6 + (i + 1) * 0.02  # Pattern: 0.62, 0.64, ...
                new_rolling[i] = 0.002 + (i + 1) * 0.0001  # Pattern: 0.0021, 0.0022, ...
            self.model.shape_material_mu.assign(new_mu)
            self.model.shape_material_mu_torsional.assign(new_torsional)
            self.model.shape_material_mu_rolling.assign(new_rolling)
            new_ke = np.ones(shape_count) * 1000.0  # High stiffness
            new_kd = np.ones(shape_count) * 10.0  # Some damping
            self.model.shape_material_ke.assign(new_ke)
            self.model.shape_material_kd.assign(new_kd)
            new_sizes = []
            for i in range(shape_count):
                old_size = self.model.shape_scale.numpy()[i]
                new_size = wp.vec3(old_size[0] * 1.2, old_size[1] * 1.2, old_size[2] * 1.2)
                new_sizes.append(new_size)
            self.model.shape_scale.assign(wp.array(new_sizes, dtype=wp.vec3, device=self.model.device))
            new_transforms = []
            for i in range(shape_count):
                # New position with offset
                new_pos = wp.vec3(0.5 + i * 0.1, 1.0 + i * 0.1, 1.5 + i * 0.1)
                # New orientation (small rotation)
                angle = 0.1 + i * 0.05
                new_quat = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), angle)
                new_transform = wp.transform(new_pos, new_quat)
                new_transforms.append(new_transform)
            self.model.shape_transform.assign(wp.array(new_transforms, dtype=wp.transform, device=self.model.device))
            solver.notify_model_changed(ModelFlags.SHAPE_PROPERTIES)
            updated_friction = solver.mjw_model.geom_friction.numpy()
            updated_solref = solver.mjw_model.geom_solref.numpy()
            updated_size = solver.mjw_model.geom_size.numpy()
            updated_pos = solver.mjw_model.geom_pos.numpy()
            updated_quat = solver.mjw_model.geom_quat.numpy()
            tested_count = 0
            for world_idx in range(self.model.world_count):
                for geom_idx in range(num_geoms):
                    shape_idx = mjc_geom_to_newton_shape[world_idx, geom_idx]
                    if shape_idx < 0:  # No mapping
                        continue

                    tested_count += 1

                    # Verify 1: Friction updated (slide, torsional, and rolling)
                    expected_mu = new_mu[shape_idx]
                    expected_torsional = new_torsional[shape_idx]
                    expected_rolling = new_rolling[shape_idx]

                    # Verify slide friction
                    self.assertAlmostEqual(
                        float(updated_friction[world_idx, geom_idx][0]),
                        expected_mu,
                        places=5,
                        msg=f"Updated slide friction should match new value for shape {shape_idx}",
                    )
                    # Verify torsional friction
                    self.assertAlmostEqual(
                        float(updated_friction[world_idx, geom_idx][1]),
                        expected_torsional,
                        places=5,
                        msg=f"Updated torsional friction should match new value for shape {shape_idx}",
                    )
                    # Verify rolling friction
                    self.assertAlmostEqual(
                        float(updated_friction[world_idx, geom_idx][2]),
                        expected_rolling,
                        places=5,
                        msg=f"Updated rolling friction should match new value for shape {shape_idx}",
                    )

                    # Verify all friction components changed from initial
                    self.assertNotAlmostEqual(
                        float(updated_friction[world_idx, geom_idx][0]),
                        float(initial_friction[world_idx, geom_idx][0]),
                        places=5,
                        msg=f"Slide friction should have changed for shape {shape_idx}",
                    )
                    self.assertNotAlmostEqual(
                        float(updated_friction[world_idx, geom_idx][1]),
                        float(initial_friction[world_idx, geom_idx][1]),
                        places=5,
                        msg=f"Torsional friction should have changed for shape {shape_idx}",
                    )
                    self.assertNotAlmostEqual(
                        float(updated_friction[world_idx, geom_idx][2]),
                        float(initial_friction[world_idx, geom_idx][2]),
                        places=5,
                        msg=f"Rolling friction should have changed for shape {shape_idx}",
                    )

                    # Verify 2: Contact parameters updated (solref)
                    # Compute expected values based on new ke/kd using timeconst/dampratio conversion
                    ke = new_ke[shape_idx]
                    kd = new_kd[shape_idx]

                    if ke > 0.0 and kd > 0.0:
                        timeconst = 2.0 / kd
                        dampratio = np.sqrt(1.0 / (timeconst * timeconst * ke))
                        expected_solref = (timeconst, dampratio)
                    else:
                        expected_solref = (0.02, 1.0)

                    self.assertAlmostEqual(
                        float(updated_solref[world_idx, geom_idx][0]),
                        expected_solref[0],
                        places=5,
                        msg=f"Updated solref[0] should match expected for shape {shape_idx}",
                    )

                    self.assertAlmostEqual(
                        float(updated_solref[world_idx, geom_idx][1]),
                        expected_solref[1],
                        places=5,
                        msg=f"Updated solref[1] should match expected for shape {shape_idx}",
                    )

                    # Also verify it changed from initial
                    self.assertFalse(
                        np.allclose(updated_solref[world_idx, geom_idx], initial_solref[world_idx, geom_idx]),
                        f"Contact parameters should have changed for shape {shape_idx}",
                    )

                    # Verify 3: Size updated
                    # Verify the size matches the expected new size
                    expected_size = new_sizes[shape_idx]
                    for dim in range(3):
                        self.assertAlmostEqual(
                            float(updated_size[world_idx, geom_idx][dim]),
                            float(expected_size[dim]),
                            places=5,
                            msg=f"Updated size mismatch for shape {shape_idx} in world {world_idx}, geom {geom_idx}, dimension {dim}",
                        )

                    # Also verify at least one dimension changed
                    size_changed = False
                    for dim in range(3):
                        if not np.isclose(
                            updated_size[world_idx, geom_idx][dim], initial_size[world_idx, geom_idx][dim]
                        ):
                            size_changed = True
                            break
                    self.assertTrue(size_changed, f"Size should have changed for shape {shape_idx}")

                    # Verify 4: Position and orientation updated (body-local coordinates)
                    # Compute expected values based on new transforms
                    new_transform = wp.transform(*new_transforms[shape_idx])
                    expected_pos = new_transform.p
                    expected_quat = new_transform.q

                    # Convert expected quaternion to MuJoCo format (wxyz)
                    expected_quat_mjc = np.array([expected_quat.w, expected_quat.x, expected_quat.y, expected_quat.z])

                    # Test position updated correctly
                    for dim in range(3):
                        self.assertAlmostEqual(
                            float(updated_pos[world_idx, geom_idx][dim]),
                            float(expected_pos[dim]),
                            places=5,
                            msg=f"Updated position mismatch for shape {shape_idx} in world {world_idx}, geom {geom_idx}, dimension {dim}",
                        )

                    # Test quaternion updated correctly
                    for dim in range(4):
                        self.assertAlmostEqual(
                            float(updated_quat[world_idx, geom_idx][dim]),
                            float(expected_quat_mjc[dim]),
                            places=5,
                            msg=f"Updated quaternion mismatch for shape {shape_idx} in world {world_idx}, geom {geom_idx}, component {dim}",
                        )

                    # Also verify they changed from initial values
                    self.assertFalse(
                        np.allclose(updated_pos[world_idx, geom_idx], initial_pos[world_idx, geom_idx]),
                        f"Position should have changed for shape {shape_idx}",
                    )
                    self.assertFalse(
                        np.allclose(updated_quat[world_idx, geom_idx], initial_quat[world_idx, geom_idx]),
                        f"Orientation should have changed for shape {shape_idx}",
                    )
            self.assertGreater(tested_count, 0, "Should have tested at least one shape")
            solver.step(self.state_in, self.state_out, self.control, self.contacts, 0.01)

    def test_mesh_maxhullvert_attribute(self):
        """Test that Mesh objects can store maxhullvert attribute"""

        vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
        indices = np.array([0, 1, 2, 0, 1, 3, 0, 2, 3, 1, 2, 3], dtype=np.int32)

        # Test default maxhullvert
        mesh1 = Mesh(vertices, indices)
        self.assertEqual(mesh1.maxhullvert, 64)

        # Test custom maxhullvert
        mesh2 = Mesh(vertices, indices, maxhullvert=128)
        self.assertEqual(mesh2.maxhullvert, 128)

    def test_mujoco_solver_uses_mesh_maxhullvert(self):
        """Test that MuJoCo solver uses per-mesh maxhullvert values"""

        builder = newton.ModelBuilder()

        # Create meshes with different maxhullvert values
        vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
        indices = np.array([0, 1, 2, 0, 1, 3, 0, 2, 3, 1, 2, 3], dtype=np.int32)

        mesh1 = Mesh(vertices, indices, maxhullvert=32)
        mesh2 = Mesh(vertices, indices, maxhullvert=128)

        # Add bodies and shapes with these meshes
        body1 = builder.add_body(mass=1.0)
        builder.add_shape_mesh(body=body1, mesh=mesh1)

        body2 = builder.add_body(mass=1.0)
        builder.add_shape_mesh(body=body2, mesh=mesh2)

        model = builder.finalize()

        # Create MuJoCo solver
        solver = SolverMuJoCo(model)

        # The solver should have used the per-mesh maxhullvert values
        # We can't directly verify this without inspecting MuJoCo internals,
        # but we can at least verify the solver was created successfully
        self.assertIsNotNone(solver)

        # Verify that the meshes retained their maxhullvert values
        self.assertEqual(model.shape_source[0].maxhullvert, 32)
        self.assertEqual(model.shape_source[1].maxhullvert, 128)

    def test_heterogeneous_per_shape_friction(self):
        """Test per-shape friction conversion to MuJoCo and dynamic updates across multiple worlds."""
        _MuJoCoSolverPropertiesFixture.setUp(self)
        # Use per-world iteration to handle potential global shapes correctly
        shape_world = self.model.shape_world.numpy()
        initial_mu = np.zeros(self.model.shape_count)
        initial_torsional = np.zeros(self.model.shape_count)
        initial_rolling = np.zeros(self.model.shape_count)

        # Set unique friction values per shape and world
        for world_idx in range(self.model.world_count):
            world_shape_indices = np.where(shape_world == world_idx)[0]
            for local_idx, shape_idx in enumerate(world_shape_indices):
                initial_mu[shape_idx] = 0.5 + local_idx * 0.1 + world_idx * 0.3
                initial_torsional[shape_idx] = 0.3 + local_idx * 0.05 + world_idx * 0.2
                initial_rolling[shape_idx] = 0.001 + local_idx * 0.0005 + world_idx * 0.002

        self.model.shape_material_mu.assign(initial_mu)
        self.model.shape_material_mu_torsional.assign(initial_torsional)
        self.model.shape_material_mu_rolling.assign(initial_rolling)

        solver = SolverMuJoCo(self.model, iterations=1, disable_contacts=True)
        mjc_geom_to_newton_shape = solver.mjc_geom_to_newton_shape.numpy()
        num_geoms = solver.mj_model.ngeom

        # Verify initial conversion
        geom_friction = solver.mjw_model.geom_friction.numpy()
        tested_count = 0
        for world_idx in range(self.model.world_count):
            for geom_idx in range(num_geoms):
                shape_idx = mjc_geom_to_newton_shape[world_idx, geom_idx]
                if shape_idx < 0:
                    continue

                tested_count += 1
                expected_mu = initial_mu[shape_idx]
                expected_torsional_abs = initial_torsional[shape_idx]
                expected_rolling_abs = initial_rolling[shape_idx]

                actual_friction = geom_friction[world_idx, geom_idx]

                self.assertAlmostEqual(
                    float(actual_friction[0]),
                    expected_mu,
                    places=5,
                    msg=f"Initial slide friction mismatch for shape {shape_idx} in world {world_idx}, geom {geom_idx}",
                )

                self.assertAlmostEqual(
                    float(actual_friction[1]),
                    expected_torsional_abs,
                    places=5,
                    msg=f"Initial torsional friction mismatch for shape {shape_idx} in world {world_idx}, geom {geom_idx}",
                )

                self.assertAlmostEqual(
                    float(actual_friction[2]),
                    expected_rolling_abs,
                    places=5,
                    msg=f"Initial rolling friction mismatch for shape {shape_idx} in world {world_idx}, geom {geom_idx}",
                )

        self.assertGreater(tested_count, 0, "Should have tested at least one shape")

        # Update with different values
        updated_mu = np.zeros(self.model.shape_count)
        updated_torsional = np.zeros(self.model.shape_count)
        updated_rolling = np.zeros(self.model.shape_count)

        for world_idx in range(self.model.world_count):
            world_shape_indices = np.where(shape_world == world_idx)[0]
            for local_idx, shape_idx in enumerate(world_shape_indices):
                updated_mu[shape_idx] = 1.0 + local_idx * 0.15 + world_idx * 0.4
                updated_torsional[shape_idx] = 0.6 + local_idx * 0.08 + world_idx * 0.25
                updated_rolling[shape_idx] = 0.005 + local_idx * 0.001 + world_idx * 0.003

        self.model.shape_material_mu.assign(updated_mu)
        self.model.shape_material_mu_torsional.assign(updated_torsional)
        self.model.shape_material_mu_rolling.assign(updated_rolling)

        solver.notify_model_changed(ModelFlags.SHAPE_PROPERTIES)

        # Verify updates
        updated_geom_friction = solver.mjw_model.geom_friction.numpy()

        for world_idx in range(self.model.world_count):
            for geom_idx in range(num_geoms):
                shape_idx = mjc_geom_to_newton_shape[world_idx, geom_idx]
                if shape_idx < 0:
                    continue

                expected_mu = updated_mu[shape_idx]
                expected_torsional_abs = updated_torsional[shape_idx]
                expected_rolling_abs = updated_rolling[shape_idx]

                actual_friction = updated_geom_friction[world_idx, geom_idx]

                self.assertAlmostEqual(
                    float(actual_friction[0]),
                    expected_mu,
                    places=5,
                    msg=f"Updated slide friction mismatch for shape {shape_idx} in world {world_idx}",
                )

                self.assertAlmostEqual(
                    float(actual_friction[1]),
                    expected_torsional_abs,
                    places=5,
                    msg=f"Updated torsional friction mismatch for shape {shape_idx} in world {world_idx}",
                )

                self.assertAlmostEqual(
                    float(actual_friction[2]),
                    expected_rolling_abs,
                    places=5,
                    msg=f"Updated rolling friction mismatch for shape {shape_idx} in world {world_idx}",
                )

    def test_geom_custom_properties_conversion_and_update(self):
        """Convert geom metadata and update per-world contact parameters."""
        world_count = 2
        template_builder = newton.ModelBuilder()
        SolverMuJoCo.register_custom_attributes(template_builder)
        shape_cfg = newton.ModelBuilder.ShapeConfig(density=1000.0)
        body1 = template_builder.add_link(mass=0.1)
        template_builder.add_shape_box(
            body=body1,
            hx=0.1,
            hy=0.1,
            hz=0.1,
            cfg=shape_cfg,
            custom_attributes={"mujoco:geom_group": 1, "mujoco:geom_priority": 1},
        )
        joint1 = template_builder.add_joint_free(child=body1)
        body2 = template_builder.add_link(mass=0.1)
        template_builder.add_shape_sphere(
            body=body2, radius=0.1, cfg=shape_cfg, custom_attributes={"mujoco:geom_group": 3, "mujoco:geom_priority": 0}
        )
        joint2 = template_builder.add_joint_revolute(parent=body1, child=body2, axis=(0.0, 0.0, 1.0))
        template_builder.add_articulation([joint1, joint2])
        builder = newton.ModelBuilder()
        SolverMuJoCo.register_custom_attributes(builder)
        builder.replicate(template_builder, world_count)
        model = builder.finalize()
        self.assertTrue(hasattr(model, "mujoco"), "Model should have mujoco namespace")
        self.assertTrue(hasattr(model.mujoco, "geom_solimp"), "Model should have geom_solimp attribute")
        shape_world = model.shape_world.numpy()
        initial_solimp = np.zeros((model.shape_count, 5), dtype=np.float32)
        for world_idx in range(model.world_count):
            world_shape_indices = np.where(shape_world == world_idx)[0]
            for local_idx, shape_idx in enumerate(world_shape_indices):
                initial_solimp[shape_idx] = [
                    0.8 + local_idx * 0.02 + world_idx * 0.05,  # dmin
                    0.9 + local_idx * 0.01 + world_idx * 0.02,  # dmax
                    0.001 + local_idx * 0.0005 + world_idx * 0.001,  # width
                    0.4 + local_idx * 0.05 + world_idx * 0.1,  # midpoint
                    2.0 + local_idx * 0.2 + world_idx * 0.5,  # power
                ]
        model.mujoco.geom_solimp.assign(wp.array(initial_solimp, dtype=vec5, device=model.device))
        self.assertTrue(hasattr(model, "mujoco"), "Model should have mujoco namespace")
        self.assertTrue(hasattr(model.mujoco, "geom_solmix"), "Model should have geom_solmix attribute")
        shape_world = model.shape_world.numpy()
        initial_solmix = np.zeros(model.shape_count, dtype=np.float32)
        for world_idx in range(model.world_count):
            world_shape_indices = np.where(shape_world == world_idx)[0]
            for local_idx, shape_idx in enumerate(world_shape_indices):
                initial_solmix[shape_idx] = 0.4 + local_idx * 0.2 + world_idx * 0.05
        model.mujoco.geom_solmix.assign(wp.array(initial_solmix, dtype=wp.float32, device=model.device))
        solver = SolverMuJoCo(model, iterations=1, disable_contacts=True)
        np.testing.assert_array_equal(model.mujoco.geom_group.numpy(), [1, 3, 1, 3])
        np.testing.assert_array_equal(model.mujoco.geom_priority.numpy(), [1, 0, 1, 0])
        np.testing.assert_array_equal(solver.mjw_model.geom_group.numpy(), [1, 3])
        np.testing.assert_array_equal(solver.mjw_model.geom_priority.numpy(), [1, 0])

        with self.subTest(property="solimp", phase="conversion"):
            mjc_geom_to_newton_shape = solver.mjc_geom_to_newton_shape.numpy()
            num_geoms = solver.mj_model.ngeom
            geom_solimp = solver.mjw_model.geom_solimp.numpy()
            tested_count = 0
            for world_idx in range(model.world_count):
                for geom_idx in range(num_geoms):
                    shape_idx = mjc_geom_to_newton_shape[world_idx, geom_idx]
                    if shape_idx < 0:
                        continue

                    tested_count += 1
                    expected_solimp = initial_solimp[shape_idx]
                    actual_solimp = geom_solimp[world_idx, geom_idx]

                    for i in range(5):
                        self.assertAlmostEqual(
                            float(actual_solimp[i]),
                            expected_solimp[i],
                            places=5,
                            msg=f"Initial geom_solimp[{i}] mismatch for shape {shape_idx} in world {world_idx}, geom {geom_idx}",
                        )
            self.assertGreater(tested_count, 0, "Should have tested at least one shape")

        with self.subTest(property="solmix", phase="conversion"):
            to_newton_shape_index = solver.mjc_geom_to_newton_shape.numpy()
            num_geoms = solver.mj_model.ngeom
            geom_solmix = solver.mjw_model.geom_solmix.numpy()
            tested_count = 0
            for world_idx in range(model.world_count):
                for geom_idx in range(num_geoms):
                    shape_idx = to_newton_shape_index[world_idx, geom_idx]
                    if shape_idx < 0:
                        continue

                    tested_count += 1
                    expected_solmix = initial_solmix[shape_idx]
                    actual_solmix = geom_solmix[world_idx, geom_idx]

                    self.assertAlmostEqual(
                        float(actual_solmix),
                        expected_solmix,
                        places=5,
                        msg=f"Initial geom_solmix mismatch for shape {shape_idx} in world {world_idx}, geom {geom_idx}",
                    )
            self.assertGreater(tested_count, 0, "Should have tested at least one shape")

        with self.subTest(property="solimp", phase="update"):
            updated_solimp = np.zeros((model.shape_count, 5), dtype=np.float32)
            for world_idx in range(model.world_count):
                world_shape_indices = np.where(shape_world == world_idx)[0]
                for local_idx, shape_idx in enumerate(world_shape_indices):
                    updated_solimp[shape_idx] = [
                        0.7 + local_idx * 0.03 + world_idx * 0.06,
                        0.85 + local_idx * 0.02 + world_idx * 0.03,
                        0.002 + local_idx * 0.0003 + world_idx * 0.0005,
                        0.5 + local_idx * 0.06 + world_idx * 0.08,
                        2.5 + local_idx * 0.3 + world_idx * 0.4,
                    ]
            model.mujoco.geom_solimp.assign(wp.array(updated_solimp, dtype=vec5, device=model.device))
            solver.notify_model_changed(ModelFlags.SHAPE_PROPERTIES)
            updated_geom_solimp = solver.mjw_model.geom_solimp.numpy()
            for world_idx in range(model.world_count):
                for geom_idx in range(num_geoms):
                    shape_idx = mjc_geom_to_newton_shape[world_idx, geom_idx]
                    if shape_idx < 0:
                        continue

                    expected_solimp = updated_solimp[shape_idx]
                    actual_solimp = updated_geom_solimp[world_idx, geom_idx]

                    for i in range(5):
                        self.assertAlmostEqual(
                            float(actual_solimp[i]),
                            expected_solimp[i],
                            places=5,
                            msg=f"Updated geom_solimp[{i}] mismatch for shape {shape_idx} in world {world_idx}",
                        )

        with self.subTest(property="solmix", phase="update"):
            updated_solmix = np.zeros(model.shape_count, dtype=np.float32)
            for world_idx in range(model.world_count):
                world_shape_indices = np.where(shape_world == world_idx)[0]
                for local_idx, shape_idx in enumerate(world_shape_indices):
                    updated_solmix[shape_idx] = 0.7 + local_idx * 0.03 + world_idx * 0.06
            model.mujoco.geom_solmix.assign(wp.array(updated_solmix, dtype=wp.float32, device=model.device))
            solver.notify_model_changed(ModelFlags.SHAPE_PROPERTIES)
            updated_geom_solmix = solver.mjw_model.geom_solmix.numpy()
            for world_idx in range(model.world_count):
                for geom_idx in range(num_geoms):
                    shape_idx = to_newton_shape_index[world_idx, geom_idx]
                    if shape_idx < 0:
                        continue

                    expected_solmix = updated_solmix[shape_idx]
                    actual_solmix = updated_geom_solmix[world_idx, geom_idx]

                    self.assertAlmostEqual(
                        float(actual_solmix),
                        expected_solmix,
                        places=5,
                        msg=f"Updated geom_solmix mismatch for shape {shape_idx} in world {world_idx}",
                    )

    def test_geom_margin_and_gap_from_shape_properties(self):
        """Verify shape margin and gap conversion and runtime updates.

        Confirms that shape properties are propagated during solver initialization
        and after runtime updates across multiple worlds.

        Uses use_mujoco_contacts=False because geom_margin is kept at zero
        when MuJoCo handles collisions (NATIVECCD compatibility, #2106).
        """
        num_worlds = 2
        template_builder = newton.ModelBuilder()
        shape_cfg = newton.ModelBuilder.ShapeConfig(density=1000.0, margin=0.005)

        body1 = template_builder.add_link(mass=0.1)
        template_builder.add_shape_box(body=body1, hx=0.1, hy=0.1, hz=0.1, cfg=shape_cfg)
        joint1 = template_builder.add_joint_free(child=body1)

        body2 = template_builder.add_link(mass=0.1)
        shape_cfg2 = newton.ModelBuilder.ShapeConfig(density=1000.0, margin=0.01)
        template_builder.add_shape_sphere(body=body2, radius=0.1, cfg=shape_cfg2)
        joint2 = template_builder.add_joint_revolute(parent=body1, child=body2, axis=(0.0, 0.0, 1.0))
        template_builder.add_articulation([joint1, joint2])

        builder = newton.ModelBuilder()
        builder.replicate(template_builder, num_worlds)
        model = builder.finalize()

        # Seed non-zero shape_gap to verify it does not affect geom_margin
        non_zero_gap = np.array([0.03 + i * 0.01 for i in range(model.shape_count)], dtype=np.float32)
        model.shape_gap.assign(wp.array(non_zero_gap, dtype=wp.float32, device=model.device))

        solver = SolverMuJoCo(model, iterations=1, disable_contacts=True, use_mujoco_contacts=False)
        to_newton = solver.mjc_geom_to_newton_shape.numpy()
        num_geoms = solver.mj_model.ngeom

        # Verify initial conversion: geom_margin should match shape_margin (not margin + gap)
        shape_margin = model.shape_margin.numpy()
        geom_margin = solver.mjw_model.geom_margin.numpy()
        geom_gap = solver.mjw_model.geom_gap.numpy()
        tested_count = 0
        for world_idx in range(model.world_count):
            for geom_idx in range(num_geoms):
                shape_idx = to_newton[world_idx, geom_idx]
                if shape_idx < 0:
                    continue
                tested_count += 1
                self.assertAlmostEqual(
                    float(geom_margin[world_idx, geom_idx]),
                    float(shape_margin[shape_idx]),
                    places=5,
                    msg=f"Initial geom_margin mismatch for shape {shape_idx} in world {world_idx}",
                )
                self.assertAlmostEqual(
                    float(geom_gap[world_idx, geom_idx]),
                    float(non_zero_gap[shape_idx]),
                    places=5,
                    msg=f"Initial geom_gap mismatch for shape {shape_idx} in world {world_idx}",
                )
        self.assertGreater(tested_count, 0)

        new_margin = np.array([0.02 + i * 0.005 for i in range(model.shape_count)], dtype=np.float32)
        new_gap = non_zero_gap * 2.0
        model.shape_margin.assign(wp.array(new_margin, dtype=wp.float32, device=model.device))
        model.shape_gap.assign(wp.array(new_gap, dtype=wp.float32, device=model.device))
        solver.notify_model_changed(ModelFlags.SHAPE_PROPERTIES)

        # Verify runtime update
        updated_margin = solver.mjw_model.geom_margin.numpy()
        updated_gap = solver.mjw_model.geom_gap.numpy()
        for world_idx in range(model.world_count):
            for geom_idx in range(num_geoms):
                shape_idx = to_newton[world_idx, geom_idx]
                if shape_idx < 0:
                    continue
                self.assertAlmostEqual(
                    float(updated_margin[world_idx, geom_idx]),
                    float(new_margin[shape_idx]),
                    places=5,
                    msg=f"Updated geom_margin mismatch for shape {shape_idx} in world {world_idx}",
                )
                self.assertAlmostEqual(
                    float(updated_gap[world_idx, geom_idx]),
                    float(new_gap[shape_idx]),
                    places=5,
                    msg=f"Updated geom_gap mismatch for shape {shape_idx} in world {world_idx}",
                )


class TestMuJoCoSolverPairProperties(unittest.TestCase):
    """Test contact pair property conversion and runtime updates across multiple worlds."""

    def test_pair_properties_conversion_and_update(self):
        """
        Test validation of contact pair custom attributes:
        1. Initial conversion from Model to MuJoCo (multi-world)
        2. Runtime updates (multi-world)

        Tests: pair_solref, pair_solreffriction, pair_solimp, pair_margin, pair_gap, pair_friction
        """
        world_count = 3
        pairs_per_world = 2

        # Create a simple model with geoms that we can create pairs between
        template_builder = newton.ModelBuilder()

        # Add a body with three shapes for creating pairs
        body_idx = template_builder.add_body()
        shape1_idx = template_builder.add_shape_box(
            body=body_idx,
            xform=wp.transform(wp.vec3(-0.5, 0.0, 0.5), wp.quat_identity()),
            hx=0.1,
            hy=0.1,
            hz=0.1,
        )
        shape2_idx = template_builder.add_shape_sphere(
            body=body_idx,
            xform=wp.transform(wp.vec3(0.5, 0.0, 0.5), wp.quat_identity()),
            radius=0.1,
        )
        shape3_idx = template_builder.add_shape_sphere(
            body=body_idx,
            xform=wp.transform(wp.vec3(0.0, 0.5, 0.5), wp.quat_identity()),
            radius=0.1,
        )

        # Build multi-world model
        builder = newton.ModelBuilder()
        builder.add_shape_plane()

        # Register MuJoCo custom attributes (including pair attributes)
        SolverMuJoCo.register_custom_attributes(builder)

        # Replicate template across worlds
        for i in range(world_count):
            world_transform = wp.transform((i * 2.0, 0.0, 0.0), wp.quat_identity())
            builder.add_world(template_builder, xform=world_transform)

        # Add contact pairs for each world
        # Each world gets pairs_per_world pairs
        total_pairs = world_count * pairs_per_world
        shapes_per_world = template_builder.shape_count

        for w in range(world_count):
            world_shape_offset = w * shapes_per_world + 1  # +1 for ground plane

            # Pair 1: shape1 <-> shape2
            builder.add_custom_values(
                **{
                    "mujoco:pair_world": w,
                    "mujoco:pair_geom1": world_shape_offset + shape1_idx,
                    "mujoco:pair_geom2": world_shape_offset + shape2_idx,
                    "mujoco:pair_condim": 3,
                    "mujoco:pair_solref": wp.vec2(0.02 + w * 0.01, 1.0 + w * 0.1),
                    "mujoco:pair_solreffriction": wp.vec2(0.03 + w * 0.01, 1.1 + w * 0.1),
                    "mujoco:pair_solimp": vec5(0.9 - w * 0.01, 0.95, 0.001, 0.5, 2.0),
                    "mujoco:pair_margin": 0.01 + w * 0.005,
                    "mujoco:pair_gap": 0.002 + w * 0.001,
                    "mujoco:pair_friction": vec5(1.0 + w * 0.1, 1.0, 0.005, 0.0001, 0.0001),
                }
            )

            # Pair 2: shape2 <-> shape3
            builder.add_custom_values(
                **{
                    "mujoco:pair_world": w,
                    "mujoco:pair_geom1": world_shape_offset + shape2_idx,
                    "mujoco:pair_geom2": world_shape_offset + shape3_idx,
                    "mujoco:pair_condim": 3,
                    "mujoco:pair_solref": wp.vec2(0.025 + w * 0.01, 1.2 + w * 0.1),
                    "mujoco:pair_solreffriction": wp.vec2(0.035 + w * 0.01, 1.3 + w * 0.1),
                    "mujoco:pair_solimp": vec5(0.85 - w * 0.01, 0.92, 0.002, 0.6, 2.5),
                    "mujoco:pair_margin": 0.015 + w * 0.005,
                    "mujoco:pair_gap": 0.003 + w * 0.001,
                    "mujoco:pair_friction": vec5(1.1 + w * 0.1, 1.1, 0.006, 0.0002, 0.0002),
                }
            )

        model = builder.finalize()

        # Verify custom attribute counts
        self.assertEqual(model.custom_frequency_counts.get("mujoco:pair", 0), total_pairs)

        # use_mujoco_contacts=False so pair margin/gap are restored from
        # custom attributes (with True, they stay zero for NATIVECCD compat #2106).
        solver = SolverMuJoCo(model, separate_worlds=True, iterations=1, use_mujoco_contacts=False)

        # Verify MuJoCo has the pairs (only from template world, which is world 0)
        npair = solver.mj_model.npair
        self.assertEqual(npair, pairs_per_world)

        # --- Step 1: Verify initial conversion ---
        # Use .copy() to ensure we capture the values, not a view (important for CPU mode)
        mjw_pair_solref = solver.mjw_model.pair_solref.numpy().copy()
        mjw_pair_solreffriction = solver.mjw_model.pair_solreffriction.numpy().copy()
        mjw_pair_solimp = solver.mjw_model.pair_solimp.numpy().copy()
        mjw_pair_margin = solver.mjw_model.pair_margin.numpy().copy()
        mjw_pair_gap = solver.mjw_model.pair_gap.numpy().copy()
        mjw_pair_friction = solver.mjw_model.pair_friction.numpy().copy()

        # Get expected values from Newton custom attributes (outside loop for performance)
        expected_solref_all = model.mujoco.pair_solref.numpy()
        expected_solreffriction_all = model.mujoco.pair_solreffriction.numpy()
        expected_solimp_all = model.mujoco.pair_solimp.numpy()
        expected_margin_all = model.mujoco.pair_margin.numpy()
        expected_gap_all = model.mujoco.pair_gap.numpy()
        expected_friction_all = model.mujoco.pair_friction.numpy()

        # Check values for each world and pair
        for w in range(world_count):
            newton_pair_base = w * pairs_per_world
            for p in range(pairs_per_world):
                newton_pair = newton_pair_base + p

                np.testing.assert_allclose(
                    mjw_pair_solref[w, p],
                    expected_solref_all[newton_pair],
                    rtol=1e-5,
                    err_msg=f"pair_solref mismatch at world {w}, pair {p}",
                )
                np.testing.assert_allclose(
                    mjw_pair_solreffriction[w, p],
                    expected_solreffriction_all[newton_pair],
                    rtol=1e-5,
                    err_msg=f"pair_solreffriction mismatch at world {w}, pair {p}",
                )
                np.testing.assert_allclose(
                    mjw_pair_solimp[w, p],
                    expected_solimp_all[newton_pair],
                    rtol=1e-5,
                    err_msg=f"pair_solimp mismatch at world {w}, pair {p}",
                )
                np.testing.assert_allclose(
                    mjw_pair_margin[w, p],
                    expected_margin_all[newton_pair],
                    rtol=1e-5,
                    err_msg=f"pair_margin mismatch at world {w}, pair {p}",
                )
                np.testing.assert_allclose(
                    mjw_pair_gap[w, p],
                    expected_gap_all[newton_pair],
                    rtol=1e-5,
                    err_msg=f"pair_gap mismatch at world {w}, pair {p}",
                )
                np.testing.assert_allclose(
                    mjw_pair_friction[w, p],
                    expected_friction_all[newton_pair],
                    rtol=1e-5,
                    err_msg=f"pair_friction mismatch at world {w}, pair {p}",
                )

        # --- Step 2: Runtime Update ---
        # Generate new values (different pattern)
        new_solref = np.zeros((total_pairs, 2), dtype=np.float32)
        new_solreffriction = np.zeros((total_pairs, 2), dtype=np.float32)
        new_solimp = np.zeros((total_pairs, 5), dtype=np.float32)
        new_margin = np.zeros(total_pairs, dtype=np.float32)
        new_gap = np.zeros(total_pairs, dtype=np.float32)
        new_friction = np.zeros((total_pairs, 5), dtype=np.float32)

        for i in range(total_pairs):
            new_solref[i] = [0.05 - i * 0.002, 2.0 - i * 0.1]
            new_solreffriction[i] = [0.06 - i * 0.002, 2.1 - i * 0.1]
            new_solimp[i] = [0.8 + i * 0.01, 0.9, 0.003, 0.4, 1.5]
            new_margin[i] = 0.02 + i * 0.003
            new_gap[i] = 0.005 + i * 0.001
            new_friction[i] = [1.5 + i * 0.05, 1.2, 0.007, 0.0003, 0.0003]

        # Update Newton model attributes
        model.mujoco.pair_solref.assign(wp.array(new_solref, dtype=wp.vec2, device=model.device))
        model.mujoco.pair_solreffriction.assign(wp.array(new_solreffriction, dtype=wp.vec2, device=model.device))
        model.mujoco.pair_solimp.assign(wp.array(new_solimp, dtype=vec5, device=model.device))
        model.mujoco.pair_margin.assign(wp.array(new_margin, dtype=wp.float32, device=model.device))
        model.mujoco.pair_gap.assign(wp.array(new_gap, dtype=wp.float32, device=model.device))
        model.mujoco.pair_friction.assign(wp.array(new_friction, dtype=vec5, device=model.device))

        # Notify solver of property change (pair properties are under SHAPE_PROPERTIES)
        solver.notify_model_changed(ModelFlags.SHAPE_PROPERTIES)

        # Verify updates
        mjw_pair_solref_updated = solver.mjw_model.pair_solref.numpy()
        mjw_pair_solreffriction_updated = solver.mjw_model.pair_solreffriction.numpy()
        mjw_pair_solimp_updated = solver.mjw_model.pair_solimp.numpy()
        mjw_pair_margin_updated = solver.mjw_model.pair_margin.numpy()
        mjw_pair_gap_updated = solver.mjw_model.pair_gap.numpy()
        mjw_pair_friction_updated = solver.mjw_model.pair_friction.numpy()

        for w in range(world_count):
            for p in range(pairs_per_world):
                newton_pair = w * pairs_per_world + p

                np.testing.assert_allclose(
                    mjw_pair_solref_updated[w, p],
                    new_solref[newton_pair],
                    rtol=1e-5,
                    err_msg=f"Updated pair_solref mismatch at world {w}, pair {p}",
                )
                np.testing.assert_allclose(
                    mjw_pair_solreffriction_updated[w, p],
                    new_solreffriction[newton_pair],
                    rtol=1e-5,
                    err_msg=f"Updated pair_solreffriction mismatch at world {w}, pair {p}",
                )
                np.testing.assert_allclose(
                    mjw_pair_solimp_updated[w, p],
                    new_solimp[newton_pair],
                    rtol=1e-5,
                    err_msg=f"Updated pair_solimp mismatch at world {w}, pair {p}",
                )
                np.testing.assert_allclose(
                    mjw_pair_margin_updated[w, p],
                    new_margin[newton_pair],
                    rtol=1e-5,
                    err_msg=f"Updated pair_margin mismatch at world {w}, pair {p}",
                )
                np.testing.assert_allclose(
                    mjw_pair_gap_updated[w, p],
                    new_gap[newton_pair],
                    rtol=1e-5,
                    err_msg=f"Updated pair_gap mismatch at world {w}, pair {p}",
                )
                np.testing.assert_allclose(
                    mjw_pair_friction_updated[w, p],
                    new_friction[newton_pair],
                    rtol=1e-5,
                    err_msg=f"Updated pair_friction mismatch at world {w}, pair {p}",
                )

        # Sanity check: values actually changed
        self.assertFalse(
            np.allclose(mjw_pair_solref_updated[0, 0], mjw_pair_solref[0, 0]),
            "pair_solref should have changed after update!",
        )

        with self.assertWarnsRegex(UserWarning, r"zeroed for NATIVECCD/MULTICCD"):
            mujoco_contacts_solver = SolverMuJoCo(
                model,
                separate_worlds=True,
                iterations=1,
                use_mujoco_contacts=True,
            )
        np.testing.assert_allclose(
            mujoco_contacts_solver.mjw_model.pair_gap.numpy(),
            new_gap.reshape(world_count, pairs_per_world),
        )
        np.testing.assert_array_equal(
            mujoco_contacts_solver.mjw_model.pair_margin.numpy(),
            np.zeros_like(mujoco_contacts_solver.mjw_model.pair_margin.numpy()),
        )

        final_gap = new_gap + 0.01
        model.mujoco.pair_gap.assign(wp.array(final_gap, dtype=wp.float32, device=model.device))
        mujoco_contacts_solver.notify_model_changed(ModelFlags.SHAPE_PROPERTIES)
        np.testing.assert_allclose(
            mujoco_contacts_solver.mjw_model.pair_gap.numpy(),
            final_gap.reshape(world_count, pairs_per_world),
        )
        np.testing.assert_array_equal(
            mujoco_contacts_solver.mjw_model.pair_margin.numpy(),
            np.zeros_like(mujoco_contacts_solver.mjw_model.pair_margin.numpy()),
        )

    def test_global_pair_exported_to_spec(self):
        """Pairs with pair_world=-1 (global) should be included in the MuJoCo spec.

        Regression test: previously global pairs were skipped because -1 != template_world.
        """
        mjcf = """<mujoco>
            <worldbody>
                <geom name="floor" type="plane" size="5 5 0.1"/>
                <body name="ball" pos="0 0 0.05">
                    <freejoint/>
                    <inertial pos="0 0 0" mass="1" diaginertia="0.01 0.01 0.01"/>
                    <geom name="ball_geom" type="sphere" size="0.1"/>
                </body>
            </worldbody>
            <contact>
                <pair geom1="floor" geom2="ball_geom" condim="3"
                      friction="2 2 0.01 0.0001 0.0001"/>
            </contact>
        </mujoco>"""
        builder = newton.ModelBuilder()
        builder.add_mjcf(mjcf)
        model = builder.finalize()
        solver = SolverMuJoCo(model)

        self.assertEqual(solver.mj_model.npair, 1, "Global pair should be exported to MuJoCo spec")

    @unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
    def test_joint_dof_label_resolution_all_joint_types(self):
        """Test that mujoco:joint_dof_label resolves correctly for fixed, revolute, spherical, and D6 joints."""
        from pxr import Usd

        usd_content = """#usda 1.0
(
    metersPerUnit = 1
    upAxis = "Z"
)
def PhysicsScene "physicsScene" {}
def Xform "R" (prepend apiSchemas = ["PhysicsArticulationRootAPI"])
{
    def Xform "Base" (prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI"])
    {
        float physics:mass = 1000
    }
    def Xform "B1" (prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI"])
    {
        float physics:mass = 1
    }
    def Xform "B2" (prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI"])
    {
        float physics:mass = 1
    }
    def Xform "B3" (prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI"])
    {
        float physics:mass = 1
    }
    def Xform "B4" (prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI"])
    {
        float physics:mass = 1
    }

    def PhysicsFixedJoint "FixRoot"
    {
        rel physics:body0 = None
        rel physics:body1 = </R/Base>
    }

    def PhysicsFixedJoint "Fixed"
    {
        rel physics:body0 = </R/Base>
        rel physics:body1 = </R/B1>
    }

    def PhysicsRevoluteJoint "Rev"
    {
        uniform token physics:axis = "X"
        rel physics:body0 = </R/Base>
        rel physics:body1 = </R/B2>
        float physics:lowerLimit = -90
        float physics:upperLimit = 90
    }

    def PhysicsSphericalJoint "Sph"
    {
        rel physics:body0 = </R/Base>
        rel physics:body1 = </R/B3>
    }

    def PhysicsJoint "D6" (
        prepend apiSchemas = ["PhysicsLimitAPI:rotX", "PhysicsLimitAPI:rotY", "PhysicsLimitAPI:rotZ",
                              "PhysicsLimitAPI:transX", "PhysicsLimitAPI:transY", "PhysicsLimitAPI:transZ"])
    {
        rel physics:body0 = </R/Base>
        rel physics:body1 = </R/B4>
        float limit:transX:physics:low = -1
        float limit:transX:physics:high = 1
        float limit:transY:physics:low = 1
        float limit:transY:physics:high = -1
        float limit:transZ:physics:low = 1
        float limit:transZ:physics:high = -1
        float limit:rotX:physics:low = -45
        float limit:rotX:physics:high = 45
        float limit:rotY:physics:low = -30
        float limit:rotY:physics:high = 30
        float limit:rotZ:physics:low = 1
        float limit:rotZ:physics:high = -1
    }
}
"""
        stage = Usd.Stage.CreateInMemory()
        stage.GetRootLayer().ImportFromString(usd_content)

        builder = newton.ModelBuilder()
        SolverMuJoCo.register_custom_attributes(builder)
        builder.add_usd(stage)

        # fixed=0 + fixed=0 + revolute=1 + spherical=3 + D6(transX,rotX,rotY)=3 → 7 DOFs
        self.assertEqual(builder.joint_dof_count, 7)

        dof_names = set(builder.custom_attributes["mujoco:joint_dof_label"].values.values())
        self.assertEqual(len(dof_names), 7)
        for expected in [
            "/R/Rev",
            "/R/Sph:rotX",
            "/R/Sph:rotY",
            "/R/Sph:rotZ",
            "/R/D6:transX",
            "/R/D6:rotX",
            "/R/D6:rotY",
        ]:
            self.assertIn(expected, dof_names)

        # Fixed joint with 0 DOFs: JOINT_DOF attribute on it should be silently skipped
        builder2 = newton.ModelBuilder()
        SolverMuJoCo.register_custom_attributes(builder2)
        body = builder2.add_link(mass=1.0, com=wp.vec3(0, 0, 0), inertia=wp.mat33(np.eye(3)))
        body2 = builder2.add_link(mass=1.0, com=wp.vec3(0, 0, 0), inertia=wp.mat33(np.eye(3)))
        builder2.add_joint_fixed(parent=-1, child=body)
        builder2.add_joint_fixed(parent=body, child=body2, custom_attributes={"mujoco:joint_dof_label": "ignored"})
        self.assertEqual(len(builder2.custom_attributes["mujoco:joint_dof_label"].values), 0)


class TestMuJoCoSolverFixedTendonProperties(unittest.TestCase):
    """Test fixed tendon property replication and runtime updates across multiple worlds."""

    def test_tendon_properties_conversion_and_update(self):
        """
        Test validation of fixed tendon custom attributes:
        1. Initial conversion from Model to MuJoCo (multi-world)
        2. Runtime updates via notify_model_changed (multi-world)
        """
        # Create template with tendons using MJCF
        mjcf = """<?xml version="1.0" ?>
<mujoco>
    <worldbody>
        <body name="root" pos="0 0 0">
            <geom type="box" size="0.1 0.1 0.1"/>
            <body name="link1" pos="0.0 -0.5 0">
                <joint name="joint1" type="slide" axis="1 0 0" range="-50.5 50.5"/>
                <geom type="cylinder" size="0.05 0.025"/>
                <inertial pos="0 0 0" mass="1" diaginertia="0.01 0.01 0.01"/>
            </body>
            <body name="link2" pos="-0.0 -0.7 0">
                <joint name="joint2" type="slide" axis="1 0 0" range="-50.5 50.5"/>
                <geom type="cylinder" size="0.05 0.025"/>
                <inertial pos="0 0 0" mass="1" diaginertia="0.01 0.01 0.01"/>
            </body>
        </body>
    </worldbody>
    <tendon>
        <fixed name="coupling_tendon" stiffness="1.0" damping="2.0" frictionloss="0.5">
            <joint joint="joint1" coef="1"/>
            <joint joint="joint2" coef="-1"/>
        </fixed>
    </tendon>
</mujoco>
"""

        template_builder = newton.ModelBuilder()
        template_builder.add_mjcf(mjcf)

        # Create main builder with multiple worlds
        world_count = 3
        builder = newton.ModelBuilder()
        SolverMuJoCo.register_custom_attributes(builder)
        builder.replicate(template_builder, world_count)
        model = builder.finalize()

        # Verify we have the custom attributes
        self.assertTrue(hasattr(model, "mujoco"))
        self.assertTrue(hasattr(model.mujoco, "tendon_stiffness"))

        # Get the total number of tendons (1 per world)
        tendon_count = len(model.mujoco.tendon_stiffness)
        self.assertEqual(tendon_count, world_count)  # 1 tendon per world

        # --- Step 1: Set initial values and verify conversion ---

        # Set different values for each world's tendon
        initial_stiffness = np.array([1.0 + i * 0.5 for i in range(world_count)], dtype=np.float32)
        initial_damping = np.array([2.0 + i * 0.3 for i in range(world_count)], dtype=np.float32)
        initial_frictionloss = np.array([0.5 + i * 0.1 for i in range(world_count)], dtype=np.float32)

        model.mujoco.tendon_stiffness.assign(initial_stiffness)
        model.mujoco.tendon_damping.assign(initial_damping)
        model.mujoco.tendon_frictionloss.assign(initial_frictionloss)

        solver = SolverMuJoCo(model, iterations=1, disable_contacts=True)

        # Check mapping exists
        self.assertIsNotNone(solver.mjc_tendon_to_newton_tendon)

        # Get MuJoCo tendon values
        mjc_tendon_to_newton = solver.mjc_tendon_to_newton_tendon.numpy()
        mjw_stiffness = solver.mjw_model.tendon_stiffness.numpy()
        mjw_damping = solver.mjw_model.tendon_damping.numpy()
        mjw_frictionloss = solver.mjw_model.tendon_frictionloss.numpy()

        ntendon = mjc_tendon_to_newton.shape[1]  # Number of MuJoCo tendons per world

        def check_values(
            expected_stiff, expected_damp, expected_friction, actual_stiff, actual_damp, actual_friction, msg_prefix
        ):
            for w in range(world_count):
                for mjc_tendon in range(ntendon):
                    newton_tendon = mjc_tendon_to_newton[w, mjc_tendon]
                    if newton_tendon < 0:
                        continue

                    self.assertAlmostEqual(
                        float(actual_stiff[w, mjc_tendon]),
                        float(expected_stiff[newton_tendon]),
                        places=4,
                        msg=f"{msg_prefix} stiffness mismatch at World {w}, tendon {mjc_tendon}",
                    )
                    self.assertAlmostEqual(
                        float(actual_damp[w, mjc_tendon]),
                        float(expected_damp[newton_tendon]),
                        places=4,
                        msg=f"{msg_prefix} damping mismatch at World {w}, tendon {mjc_tendon}",
                    )
                    self.assertAlmostEqual(
                        float(actual_friction[w, mjc_tendon]),
                        float(expected_friction[newton_tendon]),
                        places=4,
                        msg=f"{msg_prefix} frictionloss mismatch at World {w}, tendon {mjc_tendon}",
                    )

        check_values(
            initial_stiffness,
            initial_damping,
            initial_frictionloss,
            mjw_stiffness,
            mjw_damping,
            mjw_frictionloss,
            "Initial conversion",
        )

        # --- Step 2: Runtime Update ---

        # Generate new unique values
        updated_stiffness = np.array([10.0 + i * 2.0 for i in range(world_count)], dtype=np.float32)
        updated_damping = np.array([5.0 + i * 1.0 for i in range(world_count)], dtype=np.float32)
        updated_frictionloss = np.array([1.0 + i * 0.2 for i in range(world_count)], dtype=np.float32)

        # Update model attributes
        model.mujoco.tendon_stiffness.assign(updated_stiffness)
        model.mujoco.tendon_damping.assign(updated_damping)
        model.mujoco.tendon_frictionloss.assign(updated_frictionloss)

        # Notify solver
        solver.notify_model_changed(ModelFlags.TENDON_PROPERTIES)

        # Verify updates
        mjw_stiffness_updated = solver.mjw_model.tendon_stiffness.numpy()
        mjw_damping_updated = solver.mjw_model.tendon_damping.numpy()
        mjw_frictionloss_updated = solver.mjw_model.tendon_frictionloss.numpy()

        check_values(
            updated_stiffness,
            updated_damping,
            updated_frictionloss,
            mjw_stiffness_updated,
            mjw_damping_updated,
            mjw_frictionloss_updated,
            "Runtime update",
        )

        # Check that values actually changed (sanity check)
        self.assertFalse(
            np.allclose(mjw_stiffness_updated[0, 0], initial_stiffness[0]),
            "Stiffness value did not change from initial!",
        )
