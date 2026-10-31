# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import unittest
from dataclasses import replace
from unittest import mock

import numpy as np
import warp as wp

import newton
from newton._src.solvers.style3d.affine_body import AffineBody
from newton._src.solvers.style3d.builder import PDMatrixBuilder, _compute_max_neighbors
from newton.exceptions import NewtonDeprecationWarning
from newton.tests.unittest_utils import add_function_test, get_test_devices


def _assert_pd_matrix(test, model, solver):
    """Compare the stored matrix with independently assembled dense element contributions."""
    expected = np.zeros((model.particle_count, model.particle_count), dtype=np.float64)
    for face, pose, area, stiffness in zip(
        model.tri_indices.numpy(),
        model.tri_poses.numpy(),
        model.tri_areas.numpy(),
        model.style3d.tri_aniso_ke.numpy(),
        strict=True,
    ):
        gradients = np.vstack((-pose.sum(axis=0), pose)).astype(np.float64)
        weights = stiffness[:2] + stiffness[2]
        expected[np.ix_(face, face)] += area * (gradients * weights) @ gradients.T
    for edge, prop, area, cot in zip(
        model.edge_indices.numpy(),
        model.edge_bending_properties.numpy(),
        model.style3d.edge_rest_area.numpy(),
        model.style3d.edge_bending_cot.numpy(),
        strict=True,
    ):
        if edge[0] < 0 or edge[1] < 0:
            continue
        weights = np.array([-cot[0] - cot[2], -cot[1] - cot[3], cot[2] + cot[3], cot[0] + cot[1]])
        expected[np.ix_(edge, edge)] += prop[0] * (3.0 / area) * np.outer(weights, weights)
    actual = np.diag(solver._pd_diag.numpy())
    counts = solver._pd_off_diag.num_nz.numpy()
    entries = solver._pd_off_diag.nz_ell.numpy()
    for row, count in enumerate(counts):
        columns = entries["column_index"][:count, row]
        test.assertEqual(len(set(columns.tolist())), count)
        actual[row, columns] += entries["value"][:count, row]
    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-4)
    np.testing.assert_allclose(actual, actual.T, rtol=1e-6, atol=1e-5)
    np.testing.assert_allclose(actual.sum(axis=1), 0.0, atol=1e-3)


def test_constructor_precomputes_fixed_pd_matrix(test, device):
    """Preserve the ordinary grid's assembled PD stiffness matrix."""
    builder = newton.ModelBuilder()
    newton.solvers.SolverStyle3D.register_custom_attributes(builder)
    newton.solvers.style3d.add_cloth_grid(
        builder,
        pos=wp.vec3(0.0, 0.0, 1.0),
        rot=wp.quat_identity(),
        vel=wp.vec3(0.0, 0.0, 0.0),
        dim_x=2,
        dim_y=2,
        cell_x=0.1,
        cell_y=0.1,
        mass=0.1,
        tri_aniso_ke=wp.vec3(1.0e2, 1.0e2, 1.0e1),
        edge_aniso_ke=wp.vec3(2.0e-4, 1.0e-4, 5.0e-5),
    )
    model = builder.finalize(device=device)

    solver = newton.solvers.SolverStyle3D(model)
    config = newton.solvers.SolverStyle3D.Config()

    test.assertEqual(solver.config, config)

    test.assertGreater(float(solver._pd_diag.numpy().sum()), 0.0)
    test.assertGreater(int(solver._pd_off_diag.num_nz.numpy().sum()), 0)
    _assert_pd_matrix(test, model, solver)


def test_config_constructor_options(test, device):
    """Match Config and legacy cloth solves, including runtime updates and conflicts."""
    builder = newton.ModelBuilder()
    newton.solvers.SolverStyle3D.register_custom_attributes(builder)
    newton.solvers.style3d.add_cloth_grid(
        builder,
        pos=(0.0, 0.0, 1.0),
        rot=wp.quat_identity(),
        vel=(0.0, 0.0, 0.0),
        dim_x=1,
        dim_y=1,
        cell_x=0.1,
        cell_y=0.1,
        mass=0.1,
    )
    model = builder.finalize(device=device)
    config = newton.solvers.SolverStyle3D.Config(
        iterations=3,
        linear_iterations=5,
        drag_spring_stiff=42.0,
        enable_mouse_dragging=True,
    )

    solver = newton.solvers.SolverStyle3D(model, config=config)
    test.assertIs(solver.config, config)
    with test.assertWarnsRegex(NewtonDeprecationWarning, "config=SolverStyle3D.Config") as warning:
        legacy_solver = newton.solvers.SolverStyle3D(
            model,
            iterations=config.iterations,
            linear_iterations=config.linear_iterations,
            drag_spring_stiff=config.drag_spring_stiff,
            enable_mouse_dragging=config.enable_mouse_dragging,
        )
    test.assertEqual(warning.filename, __file__)
    test.assertIsNot(legacy_solver.config, config)
    test.assertEqual(legacy_solver.config, config)

    contacts = newton.CollisionPipeline(model).contacts()
    control = model.control()

    def solve(current_solver):
        state_in, state_out = model.state(), model.state()
        current_solver._update_drag_info(0, wp.vec3(0.0, 0.0, 1.1), wp.vec3(1.0, 0.0, 0.0))
        current_solver.step(state_in, state_out, control, contacts, 0.01)
        return state_out.particle_q.numpy(), state_out.particle_qd.numpy()

    outputs = [solve(current_solver) for current_solver in (solver, legacy_solver)]
    np.testing.assert_allclose(outputs[0][0], outputs[1][0], atol=1e-6)
    np.testing.assert_allclose(outputs[0][1], outputs[1][1], atol=1e-6)
    test.assertTrue(np.isfinite(outputs[0][0]).all())
    test.assertGreater(np.linalg.norm(outputs[0][1]), 0.0)

    # Reuse a solver after each individual edit and compare with a fresh solver.
    previous = outputs[0]
    for name, value in (
        ("iterations", 1),
        ("linear_iterations", 1),
        ("drag_spring_stiff", 420.0),
        ("enable_mouse_dragging", False),
    ):
        with test.subTest(runtime_option=name):
            setattr(config, name, value)
            actual = solve(solver)
            expected = solve(newton.solvers.SolverStyle3D(model, config=replace(config)))
            np.testing.assert_allclose(actual[0], expected[0], atol=1e-6)
            np.testing.assert_allclose(actual[1], expected[1], atol=1e-6)
            test.assertGreater(np.max(np.abs(actual[1] - previous[1])), 1e-6)
            legacy_name = "nonlinear_iterations" if name == "iterations" else name
            with test.assertWarnsRegex(NewtonDeprecationWarning, f"solver.config.{name}") as warning:
                test.assertEqual(getattr(solver, legacy_name), value)
            test.assertEqual(warning.filename, __file__)
            previous = actual

    # Replacing the public configuration also updates the next solve.
    solver.config = replace(legacy_solver.config)
    test.assertIsNot(solver.config, config)
    actual = solve(solver)
    np.testing.assert_allclose(actual[0], outputs[0][0], atol=1e-6)
    np.testing.assert_allclose(actual[1], outputs[0][1], atol=1e-6)

    # Legacy setters update the same configuration used by the solver.
    for name in ("iterations", "linear_iterations", "drag_spring_stiff", "enable_mouse_dragging"):
        legacy_name = "nonlinear_iterations" if name == "iterations" else name
        with test.assertWarnsRegex(NewtonDeprecationWarning, f"solver.config.{name}") as warning:
            setattr(solver, legacy_name, getattr(config, name))
        test.assertEqual(warning.filename, __file__)
    test.assertEqual(solver.config, config)
    actual = solve(solver)
    np.testing.assert_allclose(actual[0], previous[0], atol=1e-6)
    np.testing.assert_allclose(actual[1], previous[1], atol=1e-6)

    default_config = newton.solvers.SolverStyle3D.Config()
    with test.assertWarnsRegex(NewtonDeprecationWarning, "config=SolverStyle3D.Config"):
        partial_solver = newton.solvers.SolverStyle3D(model, config=None, iterations=2, drag_spring_stiff=0.0)
    test.assertEqual(partial_solver.config, replace(default_config, iterations=2, drag_spring_stiff=0.0))

    for name, value in (
        ("iterations", default_config.iterations),
        ("linear_iterations", default_config.linear_iterations),
        ("drag_spring_stiff", default_config.drag_spring_stiff),
        ("enable_mouse_dragging", default_config.enable_mouse_dragging),
    ):
        with test.subTest(option=name):
            with test.assertRaisesRegex(TypeError, "either 'config' or legacy solver options"):
                newton.solvers.SolverStyle3D(model, config=config, **{name: value})
    with test.assertRaisesRegex(TypeError, "Expected config to be SolverStyle3D.Config"):
        newton.solvers.SolverStyle3D(model, config={})

    # Invalid runtime edits must fail before either state is modified.
    valid_config = replace(solver.config)
    state_in, state_out = model.state(), model.state()
    state_in.particle_qd.fill_(wp.vec3(1.0, 2.0, 3.0))
    state_out.particle_q.fill_(wp.vec3(4.0, 5.0, 6.0))
    state_out.particle_qd.fill_(wp.vec3(7.0, 8.0, 9.0))
    arrays = (state_in.particle_q, state_in.particle_qd, state_out.particle_q, state_out.particle_qd)
    before = [array.numpy().copy() for array in arrays]
    for iterations in (0, -1):
        with test.subTest(invalid_iterations=iterations):
            if iterations == 0:
                solver.config.iterations = iterations
            else:
                solver.config = replace(valid_config, iterations=iterations)
            with test.assertRaisesRegex(ValueError, "Config.iterations must be at least 1"):
                solver.step(state_in, state_out, control, contacts, 0.01)
            for array, expected in zip(arrays, before, strict=True):
                np.testing.assert_array_equal(array.numpy(), expected)

    solver.config = valid_config
    actual = solve(solver)
    np.testing.assert_allclose(actual[0], previous[0], atol=1e-6)
    np.testing.assert_allclose(actual[1], previous[1], atol=1e-6)


def test_high_valence_pd_matrix(test, device):
    """Preserve stretch and bending entries across the old capacity boundary."""
    for n, outer_ring in ((32, False), (33, False), (20, True)):
        with test.subTest(neighbors=n, outer_ring=outer_ring):
            angle = np.linspace(0, 2 * np.pi, n, endpoint=False)
            vertices = [[0.0, 0.0, 0.0], *np.column_stack((np.cos(angle), np.sin(angle), np.zeros(n))).tolist()]
            triangles = [[0, 1 + i, 1 + (i + 1) % n] for i in range(n)]
            if outer_ring:
                outer_angle = angle + np.pi / n
                vertices += np.column_stack((2 * np.cos(outer_angle), 2 * np.sin(outer_angle), np.zeros(n))).tolist()
                triangles += [[1 + i, n + 1 + i, 1 + (i + 1) % n] for i in range(n)]
            builder = newton.ModelBuilder()
            newton.solvers.SolverStyle3D.register_custom_attributes(builder)
            newton.solvers.style3d.add_cloth_mesh(
                builder,
                pos=(0.0, 0.0, 0.0),
                rot=wp.quat_identity(),
                vel=(0.0, 0.0, 0.0),
                vertices=vertices,
                indices=np.array(triangles).flatten().tolist(),
                density=1.0,
                tri_aniso_ke=wp.vec3(100.0, 100.0, 10.0),
                edge_aniso_ke=wp.vec3(2.0e-4, 1.0e-4, 5.0e-5),
            )
            model = builder.finalize(device=device)
            solver = newton.solvers.SolverStyle3D(
                model, config=newton.solvers.SolverStyle3D.Config(iterations=1, linear_iterations=1)
            )
            test.assertEqual(int(solver._pd_off_diag.num_nz.numpy()[0]), n * (2 if outer_ring else 1))
            _assert_pd_matrix(test, model, solver)


class TestSolverStyle3D(unittest.TestCase):
    def test_capacity_matches_repeated_vertex_assembly(self):
        """Count self-target slots from repeated stencil positions as the kernels do."""
        self.assertEqual(_compute_max_neighbors(0, [], []), 0)
        self.assertEqual(_compute_max_neighbors(4, [], [[-1, 0, 1, 2]]), 0)
        for kind in ("stretch", "bend"):
            with self.subTest(kind=kind):
                faces = [[0, i, i + 1] for i in range(1, 32)] + [[0, 0, 1]]
                edges = [[0, i, i + 1, i + 2] for i in range(1, 31)] + [[0, 0, 1, 2]]
                width = _compute_max_neighbors(33, faces if kind == "stretch" else [], edges if kind == "bend" else [])
                self.assertEqual(width, 33)
                matrix = PDMatrixBuilder(33, max_neighbor=width)
                if kind == "stretch":
                    matrix.add_stretch_constraints(
                        faces, [[[1.0, 0.0], [0.0, 1.0]]] * len(faces), [[1.0] * 3] * len(faces), [0.5] * len(faces)
                    )
                else:
                    matrix.add_bend_constraints(
                        edges, [[1.0, 0.0]] * len(edges), [1.0] * len(edges), [[1.0] * 4] * len(edges)
                    )
                diag, counts, entries = matrix.finalize("cpu")
                self.assertEqual(int(counts.numpy()[0]), 33)
                self.assertEqual(set(entries.numpy()["column_index"][:, 0]), set(range(33)))
                self.assertTrue(np.isfinite(diag.numpy()).all())

    def test_overflow_rejects_incomplete_matrix(self):
        """Raise for undersized stretch and bending assembly, including later finalization."""
        for kind in ("stretch", "bend"):
            with self.subTest(kind=kind):
                builder = PDMatrixBuilder(4, max_neighbor=1)
                with self.assertRaisesRegex(ValueError, "PD matrix capacity 1 exceeded at vertex"):
                    if kind == "stretch":
                        builder.add_stretch_constraints(
                            [[0, 1, 2]], [[[1.0, 0.0], [0.0, 1.0]]], [[1.0, 1.0, 1.0]], [0.5]
                        )
                    else:
                        builder.add_bend_constraints([[0, 1, 2, 3]], [[1.0, 0.0]], [1.0], [[1.0, 1.0, 1.0, 1.0]])
                with self.assertRaisesRegex(ValueError, "PD matrix capacity"):
                    builder.finalize("cpu")


def test_invalid_topology_is_rejected_before_precompute(test, device):
    """Reject malformed cloth topology before Style3D precomputation."""
    builder = newton.ModelBuilder()
    newton.solvers.SolverStyle3D.register_custom_attributes(builder)
    newton.solvers.style3d.add_cloth_grid(
        builder,
        pos=wp.vec3(0.0, 0.0, 1.0),
        rot=wp.quat_identity(),
        vel=wp.vec3(0.0, 0.0, 0.0),
        dim_x=1,
        dim_y=1,
        cell_x=0.1,
        cell_y=0.1,
        mass=0.1,
    )
    i, j, _ = builder.tri_indices[0]
    builder.tri_indices[0] = (i, j, builder.particle_count)

    with mock.patch.object(newton.solvers.SolverStyle3D, "_precompute") as precompute:
        with test.assertRaisesRegex(ValueError, "tri_indices.*particle count"):
            model = builder.finalize(device=device)
            newton.solvers.SolverStyle3D(
                model, config=newton.solvers.SolverStyle3D.Config(iterations=1, linear_iterations=1)
            )

    precompute.assert_not_called()


def test_zero_mass_isolated_particle_remains_finite(test, device):
    """Keep an isolated zero-mass particle fixed with finite state."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    newton.solvers.SolverStyle3D.register_custom_attributes(builder)
    newton.solvers.style3d.add_cloth_mesh(
        builder,
        pos=(0.0, 0.0, 0.0),
        rot=wp.quat_identity(),
        vel=(1.0, 2.0, 3.0),
        vertices=((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (2.0, 2.0, 0.0)),
        indices=(0, 1, 2),
        density=1.0,
    )
    model = builder.finalize(device=device)
    test.assertEqual(float(model.particle_mass.numpy()[3]), 0.0)
    test.assertTrue(int(model.particle_flags.numpy()[3]) & int(newton.ParticleFlags.ACTIVE))

    solver = newton.solvers.SolverStyle3D(
        model, config=newton.solvers.SolverStyle3D.Config(iterations=1, linear_iterations=1)
    )
    state_0 = model.state()
    state_1 = model.state()
    collision_pipeline = newton.CollisionPipeline(model)
    contacts = collision_pipeline.contacts()
    initial_position = state_0.particle_q.numpy()[3].copy()
    solver.step(state_0, state_1, model.control(), contacts, 0.01)

    positions = state_1.particle_q.numpy()
    velocities = state_1.particle_qd.numpy()
    test.assertTrue(np.isfinite(positions).all())
    test.assertTrue(np.isfinite(velocities).all())
    np.testing.assert_allclose(positions[3], initial_position)
    np.testing.assert_allclose(velocities[3], 0.0)


def test_solver_flags_deactivate_zero_mass_without_mutating_model(test, device):
    """Deactivate zero-mass solver particles without changing shared model flags."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    newton.solvers.SolverStyle3D.register_custom_attributes(builder)
    newton.solvers.style3d.add_cloth_mesh(
        builder,
        pos=(0.0, 0.0, 0.0),
        rot=wp.quat_identity(),
        vel=(0.0, 0.0, 0.0),
        vertices=((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (2.0, 2.0, 0.0)),
        indices=(0, 1, 2),
        density=1.0,
    )
    builder.particle_flags[3] |= newton.ParticleFlags.PROXY
    model = builder.finalize(device=device)
    model_flags = model.particle_flags.numpy().copy()

    solver = newton.solvers.SolverStyle3D(
        model, config=newton.solvers.SolverStyle3D.Config(iterations=1, linear_iterations=1)
    )
    state_0 = model.state()
    state_1 = model.state()
    contacts = newton.CollisionPipeline(model).contacts()
    solver.step(state_0, state_1, model.control(), contacts, 0.01)

    np.testing.assert_array_equal(model.particle_flags.numpy(), model_flags)
    solver_flags = solver._particle_flags.numpy()
    active = int(newton.ParticleFlags.ACTIVE)
    proxy = int(newton.ParticleFlags.PROXY)
    test.assertTrue(np.all((solver_flags[:3] & active) != 0))
    test.assertEqual(int(solver_flags[3]) & active, 0)
    test.assertNotEqual(int(solver_flags[3]) & proxy, 0)


def test_solver_flags_track_runtime_model_changes(test, device):
    """Refresh solver activity after model masses and flags change."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    newton.solvers.SolverStyle3D.register_custom_attributes(builder)
    newton.solvers.style3d.add_cloth_grid(
        builder,
        pos=(0.0, 0.0, 0.0),
        rot=wp.quat_identity(),
        vel=(0.0, 0.0, 0.0),
        dim_x=1,
        dim_y=1,
        cell_x=0.1,
        cell_y=0.1,
        mass=0.1,
    )
    model = builder.finalize(device=device)
    solver = newton.solvers.SolverStyle3D(
        model, config=newton.solvers.SolverStyle3D.Config(iterations=1, linear_iterations=1)
    )

    masses = model.particle_mass.numpy().copy()
    masses[0] = 0.0
    model.particle_mass.assign(masses)
    model_flags = model.particle_flags.numpy().copy()
    model_flags[1] = 0
    model.particle_flags.assign(model_flags)

    state_0 = model.state()
    state_1 = model.state()
    collision_pipeline = newton.CollisionPipeline(model)
    contacts = collision_pipeline.contacts()
    solver.step(state_0, state_1, model.control(), contacts, 0.01)

    solver_flags = solver._particle_flags.numpy()
    active = int(newton.ParticleFlags.ACTIVE)
    test.assertEqual(int(solver_flags[0]), 0)
    test.assertEqual(int(solver_flags[1]), 0)
    test.assertNotEqual(int(solver_flags[2]) & active, 0)
    np.testing.assert_array_equal(model.particle_flags.numpy(), model_flags)


def test_global_particles_use_global_gravity(test, device):
    """Apply dedicated global gravity to Style3D particles."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, -2.0))
    newton.solvers.SolverStyle3D.register_custom_attributes(builder)
    newton.solvers.style3d.add_cloth_grid(
        builder,
        pos=(0.0, 0.0, 0.0),
        rot=wp.quat_identity(),
        vel=(0.0, 0.0, 0.0),
        dim_x=1,
        dim_y=1,
        cell_x=0.1,
        cell_y=0.1,
        mass=0.1,
    )
    builder.begin_world(gravity=(0.0, 0.0, 0.0))
    builder.end_world()
    model = builder.finalize(device=device)
    solver = newton.solvers.SolverStyle3D(
        model, config=newton.solvers.SolverStyle3D.Config(iterations=1, linear_iterations=1)
    )
    state_0, state_1 = model.state(), model.state()
    contacts = newton.CollisionPipeline(model).contacts()

    solver.step(state_0, state_1, model.control(), contacts, 0.1)

    test.assertTrue(np.all(state_1.particle_qd.numpy()[:, 2] < -1.0e-6))


@wp.kernel
def _advance_affine_coordinates(x: wp.array[wp.vec3], x_inertia: wp.array[wp.vec3], v: wp.array[wp.vec3], dt: float):
    coordinate = wp.tid()
    v[coordinate] = (x_inertia[coordinate] - x[coordinate]) / dt
    x[coordinate] = x_inertia[coordinate]


def test_affine_body_state_conversion(test, device):
    """Recover integrated rigid states and fit deformed affine configurations."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    moments = (
        np.array([[0.12, 0.02, -0.01], [0.02, 0.3, 0.025], [-0.01, 0.025, 0.6]]),
        np.eye(3) * 0.25,
        np.zeros((3, 3)),
        np.eye(3) * 0.25,
    )
    for i, moment in enumerate(moments):
        if i == 1:
            builder.begin_world(gravity=(1.0, -2.0, 0.5))
        builder.add_link(
            xform=wp.transform(
                wp.vec3(0.4 * i, -0.2, 1.0),
                wp.quat_from_axis_angle(wp.normalize(wp.vec3(1.0, 2.0, -1.0)), 0.7 + i),
            ),
            com=wp.vec3(0.2, -0.4, 0.1),
            mass=0.0 if i == 2 else 2.0,
            inertia=wp.mat33(np.trace(moment) * np.eye(3) - moment),
        )
        if i == 1:
            builder.end_world()
        builder.body_qd[i] = wp.spatial_vector(0.3, -0.1, 0.2, 0.4, -0.7, 1.2)
    builder.body_qd[1] = wp.spatial_vector(-0.4, 0.2, 0.7, 8.0, -10.0, 12.0)
    builder.body_qd[3] = wp.spatial_vector(0.0)
    model = builder.finalize(device=device)
    affine_body = AffineBody(model)
    rigid_solver = newton.solvers.SolverBase(model)
    state_in, state_ref, state_out = model.state(), model.state(), model.state()
    forces = np.zeros((model.body_count, 6), dtype=np.float32)
    forces[0] = (0.7, -0.3, 0.5, -0.2, 0.4, 0.3)
    state_in.body_f.assign(forces)
    poses = state_in.body_q.numpy().copy()
    velocities = state_in.body_qd.numpy().copy()
    centers = model.body_com.numpy()
    inertias = model.body_inertia.numpy()

    # Views leave particle entries and trailing storage outside the component.
    storage_x = wp.full(4 * model.body_count + 5, wp.vec3(17.0), dtype=wp.vec3, device=device)
    storage_x_inertia = wp.clone(storage_x)
    storage_v = wp.clone(storage_x)
    x, x_inertia, v = storage_x[2:-3], storage_x_inertia[2:-3], storage_v[2:-3]
    for dt in (0.02, 0.15):
        with test.subTest(dt=dt):
            rigid_solver.integrate_bodies(model, state_in, state_ref, dt)
            affine_body.to_internal_state(state_in, x, x_inertia, dt)
            coordinates = x.numpy().reshape(-1, 4, 3)
            inertial_coordinates = x_inertia.numpy().reshape(-1, 4, 3)
            for i, (pose, next_pose) in enumerate(zip(poses, state_ref.body_q.numpy(), strict=True)):
                rotation = np.array(wp.quat_to_matrix(wp.quat(*pose[3:]))).reshape(3, 3)
                next_rotation = np.array(wp.quat_to_matrix(wp.quat(*next_pose[3:]))).reshape(3, 3)
                expected_coordinates = np.vstack((pose[:3] + rotation @ centers[i], rotation.T))
                next_coordinates = np.vstack((next_pose[:3] + next_rotation @ centers[i], next_rotation.T))
                np.testing.assert_allclose(coordinates[i], expected_coordinates, atol=2e-6)
                np.testing.assert_allclose(inertial_coordinates[i], next_coordinates, atol=2e-6)

            wp.launch(_advance_affine_coordinates, dim=len(x), inputs=[x, x_inertia, v, dt], device=device)
            affine_body.from_internal_state(state_out, x, v, dt)
            if wp.get_device(device).is_cuda:
                with wp.ScopedCapture(device=device) as capture:
                    affine_body.to_internal_state(state_in, x, x_inertia, dt)
                    wp.launch(_advance_affine_coordinates, dim=len(x), inputs=[x, x_inertia, v, dt], device=device)
                    affine_body.from_internal_state(state_out, x, v, dt)
                state_out.body_q.fill_(wp.transform_identity())
                state_out.body_qd.fill_(wp.spatial_vector(0.0))
                wp.capture_launch(capture.graph)

            for expected, actual in zip(state_ref.body_q.numpy(), state_out.body_q.numpy(), strict=True):
                np.testing.assert_allclose(actual[:3], expected[:3], atol=2e-5)
                np.testing.assert_allclose(
                    wp.quat_to_matrix(wp.quat(*actual[3:])), wp.quat_to_matrix(wp.quat(*expected[3:])), atol=2e-5
                )
            np.testing.assert_allclose(state_out.body_qd.numpy(), state_ref.body_qd.numpy(), atol=1e-4, rtol=2e-5)

    # Stretch, shear, and reflection require fitting beyond a rigid round trip.
    deformations = (
        np.array([[1.1, 0.15, 0.0], [-0.05, 0.9, 0.12], [0.03, 0.0, 1.2]]),
        np.diag([1.2, 0.8, -0.6]),
        np.diag([1.1, 0.9, 1.2]),
        np.eye(3),
    )
    velocity_gradient = np.array([[0.1, -0.6, 0.3], [0.8, -0.2, 0.1], [-0.4, 0.5, 0.2]])
    coordinate_velocities = v.numpy().reshape(-1, 4, 3)
    expected_poses, expected_velocities = [], []
    for i, deformation in enumerate(deformations):
        rotation = np.array(wp.quat_to_matrix(wp.quat(*poses[i, 3:]))).reshape(3, 3)
        affine = rotation @ deformation
        center = coordinates[i, 0].copy()
        coordinates[i, 1:] = affine.T
        coordinate_velocities[i, 0] = velocities[i, :3]
        coordinate_velocities[i, 1:] = velocity_gradient.T
        moment = 0.5 * np.trace(inertias[i]) * np.eye(3) - inertias[i]
        if not np.any(moment):
            moment = np.eye(3)

        # Six mass samples reproduce the second moment for independent
        # NumPy Procrustes fits of the current and previous configurations.
        factor = np.linalg.cholesky(moment)
        samples = np.vstack((factor.T, -factor.T))
        fits = []
        for configuration in (affine, affine - dt * velocity_gradient):
            target_points = samples @ configuration.T
            u, _, vt = np.linalg.svd(target_points.T @ samples)
            fits.append(u @ np.diag([1.0, 1.0, np.linalg.det(u @ vt)]) @ vt)
        fit, previous_fit = fits
        expected_poses.append((center - fit @ centers[i], fit))
        # The inverse Cayley transform independently recovers the angular velocity.
        delta = fit @ previous_fit.T
        skew = (2.0 / dt) * np.linalg.solve((delta + np.eye(3)).T, (delta - np.eye(3)).T).T
        omega = np.array([skew[2, 1], skew[0, 2], skew[1, 0]])
        expected_velocities.append(np.concatenate((velocities[i, :3], omega)))
    x.assign(coordinates.reshape(-1, 3))
    v.assign(coordinate_velocities.reshape(-1, 3))
    affine_body.from_internal_state(state_out, x, v, dt)
    for actual, (position, rotation) in zip(state_out.body_q.numpy(), expected_poses, strict=True):
        np.testing.assert_allclose(actual[:3], position, atol=2e-5)
        actual_rotation = np.array(wp.quat_to_matrix(wp.quat(*actual[3:]))).reshape(3, 3)
        np.testing.assert_allclose(actual_rotation, rotation, atol=2e-5)
        test.assertAlmostEqual(float(np.linalg.det(actual_rotation)), 1.0, places=5)
    np.testing.assert_allclose(state_out.body_qd.numpy(), expected_velocities, atol=2e-5)
    np.testing.assert_array_equal(state_in.body_q.numpy(), poses)
    np.testing.assert_array_equal(state_in.body_qd.numpy(), velocities)
    np.testing.assert_array_equal(state_in.body_f.numpy(), forces)
    for storage in (storage_x, storage_x_inertia, storage_v):
        np.testing.assert_array_equal(storage.numpy()[[0, 1, -3, -2, -1]], np.full((5, 3), 17.0))

    empty_model = newton.ModelBuilder().finalize(device=device)
    empty_affine_body = AffineBody(empty_model)
    empty_affine_body.to_internal_state(empty_model.state(), x[:0], x_inertia[:0], dt)
    empty_affine_body.from_internal_state(empty_model.state(), x[:0], v[:0], dt)


devices = get_test_devices()


add_function_test(
    TestSolverStyle3D,
    "test_affine_body_state_conversion",
    test_affine_body_state_conversion,
    devices=devices,
    check_output=False,
)


add_function_test(
    TestSolverStyle3D,
    "test_high_valence_pd_matrix",
    test_high_valence_pd_matrix,
    devices=devices,
    check_output=False,
)

add_function_test(
    TestSolverStyle3D,
    "test_constructor_precomputes_fixed_pd_matrix",
    test_constructor_precomputes_fixed_pd_matrix,
    devices=devices,
    check_output=False,
)

add_function_test(
    TestSolverStyle3D,
    "test_config_constructor_options",
    test_config_constructor_options,
    devices=devices,
    check_output=False,
)

add_function_test(
    TestSolverStyle3D,
    "test_invalid_topology_is_rejected_before_precompute",
    test_invalid_topology_is_rejected_before_precompute,
    devices=devices,
    check_output=False,
)

add_function_test(
    TestSolverStyle3D,
    "test_zero_mass_isolated_particle_remains_finite",
    test_zero_mass_isolated_particle_remains_finite,
    devices=devices,
    check_output=False,
)

add_function_test(
    TestSolverStyle3D,
    "test_solver_flags_deactivate_zero_mass_without_mutating_model",
    test_solver_flags_deactivate_zero_mass_without_mutating_model,
    devices=devices,
    check_output=False,
)

add_function_test(
    TestSolverStyle3D,
    "test_solver_flags_track_runtime_model_changes",
    test_solver_flags_track_runtime_model_changes,
    devices=devices,
    check_output=False,
)

add_function_test(
    TestSolverStyle3D,
    "test_global_particles_use_global_gravity",
    test_global_particles_use_global_gravity,
    devices=devices,
    check_output=False,
)

if __name__ == "__main__":
    unittest.main(verbosity=2, failfast=True)
