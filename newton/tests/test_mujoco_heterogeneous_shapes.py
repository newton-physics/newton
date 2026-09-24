# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Exercise ragged convex colliders through Newton contacts and MuJoCo Warp."""

import unittest

import numpy as np
import warp as wp

import newton
from newton.selection import ArticulationView
from newton.solvers import SolverMuJoCo


def build_model(variants, device):
    """Build overlapping independent worlds with different convex decompositions."""
    builder = newton.ModelBuilder()
    builder.add_ground_plane()
    for count, half_height in variants:
        template = newton.ModelBuilder()
        inertia = np.diag([(0.2**2 + half_height**2) / 3, (0.24**2 + half_height**2) / 3, (0.24**2 + 0.2**2) / 3])
        body = template.add_link(
            label="object/body",
            xform=wp.transform((0.0, 0.0, 1.0), wp.quat_identity()),
            mass=1.0,
            inertia=wp.mat33(inertia),
        )
        joint = template.add_joint_free(body, label="object/free")
        template.add_articulation([joint], label="object")
        cfg = newton.ModelBuilder.ShapeConfig(density=0.0, ke=1.0e5, kd=1.0e3, mu=0.5)
        for part in range(count):
            half_width = 0.24 / count
            mesh = newton.Mesh.create_box(half_width, 0.2, half_height, compute_inertia=False)
            template.add_shape_convex_hull(
                body,
                mesh=mesh,
                xform=wp.transform((-0.24 + (2 * part + 1) * half_width, 0.0, 0.0), wp.quat_identity()),
                cfg=cfg,
                label=f"object/hull_{part}",
            )
        builder.add_world(template)
    return builder.finalize(device=device)


def create_simulation(model, *, heterogeneous):
    """Create independent contact and dynamics buffers for a model."""
    options = {"allow_heterogeneous_shapes": True} if heterogeneous else {}
    solver = SolverMuJoCo(
        model,
        use_mujoco_contacts=False,
        iterations=50,
        ls_iterations=20,
        nconmax=128,
        njmax=512,
        **options,
    )
    pipeline = newton.CollisionPipeline(model, rigid_contact_max=max(256, model.shape_count * 8))
    state = model.state()
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)
    return solver, pipeline, state, model.state(), model.control(), pipeline.contacts()


def advance(simulation, steps):
    """Advance dynamics using contacts against the original Newton meshes."""
    solver, pipeline, state, next_state, control, contacts = simulation
    for _ in range(steps):
        state.clear_forces()
        pipeline.collide(state, contacts)
        solver.step(state, next_state, control, contacts, 1.0 / 240.0)
        state, next_state = next_state, state
    return solver, pipeline, state, next_state, control, contacts


class TestMuJoCoHeterogeneousShapes(unittest.TestCase):
    def test_default_rejects_different_hull_counts(self):
        """Preserve the existing strict solver contract without the opt-in."""
        model = build_model([(1, 0.1), (3, 0.2), (2, 0.3)], "cpu")
        with self.assertRaisesRegex(ValueError, "homogeneous worlds"):
            create_simulation(model, heterogeneous=False)

    def test_native_contacts_reject_heterogeneous_mode(self):
        """Reject native collision detection instead of silently reusing template geometry."""
        model = build_model([(1, 0.1), (3, 0.2)], "cpu")
        with self.assertRaisesRegex(ValueError, "use_mujoco_contacts=False"):
            SolverMuJoCo(model, allow_heterogeneous_shapes=True)

    def test_empty_world_has_no_phantom_collider(self):
        """Let a shapeless body fall through the floor while other worlds settle."""
        model = build_model([(0, 0.1), (3, 0.2), (1, 0.3)], "cpu")
        simulation = advance(create_simulation(model, heterogeneous=True), 240)
        positions = simulation[2].body_q.numpy()[:, 2]
        self.assertLess(positions[0], -3.0)
        np.testing.assert_allclose(positions[1:], [0.2, 0.3], atol=0.015)

    @unittest.skipUnless(wp.is_cuda_available(), "CUDA graph replay requires a CUDA device")
    def test_sixteen_variants_with_cuda_graph(self):
        """Replay GPU dynamics for sixteen distinct convex-decomposed objects."""
        variants = [(1 + (world * 3) % 5, 0.08 + world * 0.01) for world in range(16)]
        with wp.ScopedDevice("cuda:0"):
            model = build_model(variants, "cuda:0")
            simulation = advance(create_simulation(model, heterogeneous=True), 2)
            with wp.ScopedCapture() as capture:
                # Two steps return to the same input/output buffers on each replay.
                advance(simulation, 2)
            for _ in range(180):
                wp.capture_launch(capture.graph)
            positions = simulation[2].body_q.numpy()[:, 2]
            np.testing.assert_allclose(positions, [height for _, height in variants], atol=0.015)
            self.assertTrue(np.isfinite(simulation[2].body_qd.numpy()).all())
            self.assertEqual(model.shape_count, 1 + sum(count for count, _ in variants))

    def test_batched_contacts_match_isolated_worlds(self):
        """Preserve each world's geometry and trajectory under ragged batching."""
        devices = ["cpu"]
        if wp.is_cuda_available():
            devices.append("cuda:0")
        variants = [(1, 0.1), (3, 0.2), (2, 0.3)]
        for device in devices:
            with self.subTest(device=device), wp.ScopedDevice(device):
                model = build_model(variants, device)
                shape_count = model.shape_count
                mass = model.body_mass.numpy().copy()
                inertia = model.body_inertia.numpy().copy()
                mesh_vertices = [shape.vertices.copy() for shape in model.shape_source if shape is not None]
                simulation = create_simulation(model, heterogeneous=True)
                view = ArticulationView(model, "*", include_shapes=False)
                isolated = [
                    create_simulation(build_model([variant], device), heterogeneous=False) for variant in variants
                ]
                for steps in (60, 60, 60, 180):
                    simulation = advance(simulation, steps)
                    isolated = [advance(reference, steps) for reference in isolated]
                    for world, reference in enumerate(isolated):
                        np.testing.assert_allclose(
                            simulation[2].body_q.numpy()[world], reference[2].body_q.numpy()[0], atol=2.0e-3
                        )
                        np.testing.assert_allclose(
                            simulation[2].body_qd.numpy()[world], reference[2].body_qd.numpy()[0], atol=2.0e-2
                        )
                state = simulation[2]
                batched_q = state.body_q.numpy()
                self.assertTrue(np.isfinite(batched_q).all())
                np.testing.assert_allclose(batched_q[:, 2], [0.1, 0.2, 0.3], atol=0.015)

                # Every original hull must contribute contacts, including slots absent in world zero.
                contacts = simulation[5]
                contact_count = int(contacts.rigid_contact_count.numpy()[0])
                touching = set(contacts.rigid_contact_shape0.numpy()[:contact_count])
                touching.update(contacts.rigid_contact_shape1.numpy()[:contact_count])
                self.assertEqual(touching, set(range(shape_count)))

                self.assertEqual(model.shape_count, shape_count)
                np.testing.assert_array_equal(model.body_mass.numpy(), mass)
                np.testing.assert_array_equal(model.body_inertia.numpy(), inertia)
                for expected, source in zip(
                    mesh_vertices, [shape for shape in model.shape_source if shape is not None], strict=True
                ):
                    np.testing.assert_array_equal(source.vertices, expected)

                # Teleport only the middle world, as an RL environment reset would.
                poses = view.get_root_transforms(state).numpy().copy()
                poses[1, 0, 2] = 1.0
                view.set_root_transforms(
                    state,
                    wp.array(poses, dtype=wp.transform, device=device),
                    mask=wp.array([False, True, False], dtype=bool, device=device),
                )
                view.set_root_velocities(
                    state,
                    wp.zeros((3, 1), dtype=wp.spatial_vector, device=device),
                    mask=wp.array([False, True, False], dtype=bool, device=device),
                )
                newton.eval_fk(model, state.joint_q, state.joint_qd, state)
                simulation[0].reset(state, world_mask=wp.array([False, True, False, False], dtype=bool, device=device))
                simulation = advance(simulation, 12)
                reset_positions = simulation[2].body_q.numpy()[:, 2]
                self.assertGreater(reset_positions[1], 0.9)
                np.testing.assert_allclose(reset_positions[[0, 2]], batched_q[[0, 2], 2], atol=2.0e-3)


if __name__ == "__main__":
    unittest.main()
