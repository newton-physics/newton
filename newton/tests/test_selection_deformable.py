# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for DeformableView batched selection over finalized deformable groups."""

import re
import unittest

import numpy as np
import warp as wp

import newton
from newton.selection import ArticulationView, DeformableView

_CABLE_PTS = [(0.0, 0.0, 1.0), (0.1, 0.0, 1.0), (0.2, 0.0, 1.0), (0.3, 0.0, 1.0)]


def _replicated_model(world_count=3, device=None):
    """One cloth + one cable per world, replicated."""
    sub = newton.ModelBuilder()
    _add_test_cloth(sub, label="/World/Cloth")
    _add_test_cable(sub, label="/World/Cable")
    scene = newton.ModelBuilder()
    scene.replicate(sub, world_count)
    return scene.finalize() if device is None else scene.finalize(device=device)


def _add_test_articulation(builder):
    root = builder.add_link(label="robot/root")
    root_joint = builder.add_joint_free(child=root, label="robot/root_joint")
    builder.add_articulation([root_joint], label="robot")


def _add_test_cable(builder, label="cable"):
    builder.add_rod(
        positions=_CABLE_PTS,
        radius=0.02,
        label=label,
        body_frame_origin="com",
    )


def _add_test_cloth(builder, label="cloth", vertices=None, indices=None):
    if vertices is None:
        vertices = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0)]
    if indices is None:
        indices = [0, 1, 2, 0, 2, 3]
    builder.add_cloth_mesh(
        pos=wp.vec3(0.0, 0.0, 2.0),
        rot=wp.quat_identity(),
        scale=1.0,
        vel=wp.vec3(0.0),
        vertices=vertices,
        indices=indices,
        density=1.0,
        label=label,
    )


def _add_test_soft_body(builder, label="soft"):
    builder.add_soft_mesh(
        pos=wp.vec3(0.0, 0.0, 3.0),
        rot=wp.quat_identity(),
        scale=1.0,
        vel=wp.vec3(0.0),
        vertices=[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)],
        indices=[0, 1, 2, 3],
        density=1.0,
        k_mu=100.0,
        k_lambda=100.0,
        k_damp=0.0,
        label=label,
    )


class TestDeformableView(unittest.TestCase):
    """Label-pattern selection and batched state access over deformable groups."""

    def test_cloth_view_selects_and_batches_across_worlds(self):
        """A replicated cloth selects one group per world with batched particle state."""
        model = _replicated_model(3)
        state = model.state()

        view = DeformableView(model, "/World/Cloth", family="surface")
        self.assertEqual((view.count, view.group_count, view.world_count, view.count_per_world), (3, 3, 3, 1))
        self.assertEqual(view.worlds, [0, 1, 2])
        self.assertEqual(view.particles_per_group, 4)

        positions = view.get_particle_positions(state)
        self.assertEqual(positions.shape, (3, 4))

        # Round-trip: lift each world's cloth by its world index and read it back.
        lifted = positions.numpy()
        for g in range(3):
            lifted[g, :, 2] += float(g + 1)
        view.set_particle_positions(state, wp.array(lifted, dtype=wp.vec3))
        np.testing.assert_allclose(view.get_particle_positions(state).numpy(), lifted, atol=1e-6)

        # Velocities go through the same path.
        velocities = np.full((3, 4, 3), 2.5, dtype=np.float32)
        view.set_particle_velocities(state, wp.array(velocities, dtype=wp.vec3))
        np.testing.assert_allclose(view.get_particle_velocities(state).numpy(), velocities, atol=1e-6)

    def test_cable_view_batches_body_transforms(self):
        """A replicated cable selects per world and round-trips its segment transforms."""
        model = _replicated_model(2)
        state = model.state()

        view = DeformableView(model, "/World/Cable", family="curve")
        self.assertEqual((view.count, view.bodies_per_group), (2, 3))

        transforms = view.get_body_transforms(state)
        self.assertEqual(transforms.shape, (2, 3))
        shifted = transforms.numpy()
        shifted[:, :, 1] += 5.0  # translate all segments in y
        view.set_body_transforms(state, wp.array(shifted, dtype=wp.transform))
        np.testing.assert_allclose(view.get_body_transforms(state).numpy(), shifted, atol=1e-6)

        velocities = view.get_body_velocities(state)
        self.assertEqual(velocities.shape, (2, 3))

    def test_soft_view_over_global_groups(self):
        """Groups outside any world (world -1) are selectable as a single-world view."""
        builder = newton.ModelBuilder()
        _add_test_soft_body(builder, label="/World/Soft")
        model = builder.finalize()

        view = DeformableView(model, "/World/Soft", family="volume")
        self.assertEqual((view.count, view.world_count), (1, 1))
        self.assertEqual(view.get_particle_positions(model.state()).shape, (1, 4))

    def test_pattern_matches_multiple_groups_per_world(self):
        """A wildcard pattern selects several groups per world when counts stay equal."""
        sub = newton.ModelBuilder()
        _add_test_cloth(sub, label="/World/ClothA")
        _add_test_cloth(sub, label="/World/ClothB")
        scene = newton.ModelBuilder()
        scene.replicate(sub, 2)
        model = scene.finalize()

        view = DeformableView(model, "/World/Cloth*", family="surface")
        self.assertEqual((view.count, view.count_per_world), (4, 2))
        self.assertEqual(view.labels, ["/World/ClothA", "/World/ClothB"] * 2)
        self.assertEqual(view.model_group_ids, [0, 1, 2, 3])
        self.assertEqual(view.ranges("triangle"), [(0, 2), (2, 4), (4, 6), (6, 8)])

    def test_varying_group_counts_across_worlds_remain_selectable(self):
        """Worlds may contribute different group counts while retaining stable order."""
        first = newton.ModelBuilder()
        _add_test_cloth(first, label="/World/ClothA")
        second = newton.ModelBuilder()
        _add_test_cloth(second, label="/World/ClothA")
        _add_test_cloth(second, label="/World/ClothB")
        scene = newton.ModelBuilder()
        scene.add_world(first)
        scene.add_world(second)
        scene.add_world(newton.ModelBuilder())

        view = DeformableView(scene.finalize(), "/World/Cloth*", family="surface")

        self.assertEqual((view.count, view.world_count, view.count_per_world), (3, 3, None))
        self.assertEqual(view.worlds, [0, 1, 1])
        self.assertEqual(view.world_ranges(), [(0, 1), (1, 3), (3, 3)])
        np.testing.assert_array_equal(view.world_starts.numpy(), [0, 1, 3, 3])
        np.testing.assert_array_equal(view.world_ids.numpy(), [0, 1, 1])
        self.assertEqual(view.labels, ["/World/ClothA", "/World/ClothA", "/World/ClothB"])
        self.assertEqual(view.get_particle_positions(view.model.state()).shape, (3, 4))
        with self.assertRaisesRegex(ValueError, "exactly one selected group"):
            view.set_particle_positions(view.model.state(), wp.zeros((1, 4), dtype=wp.vec3), world_indices=[0])

    def test_compiled_regex_pattern_selects_by_fullmatch(self):
        """A compiled regular expression selects groups by fullmatch: alternation picks
        two labels per replicated world, and a partial match selects nothing."""
        sub = newton.ModelBuilder()
        _add_test_cloth(sub, label="/World/ClothA")
        _add_test_cloth(sub, label="/World/ClothB")
        scene = newton.ModelBuilder()
        scene.replicate(sub, 2)
        model = scene.finalize()

        view = DeformableView(model, re.compile(r"/World/Cloth(A|B)"), family="surface")
        self.assertEqual((view.count, view.count_per_world), (4, 2))
        self.assertEqual(view.labels, ["/World/ClothA", "/World/ClothB"] * 2)

        # fullmatch: a prefix of the label is not a match.
        with self.assertRaises(KeyError):
            DeformableView(model, re.compile(r"/World/Cloth"), family="surface")

    def test_view_round_trip_on_cpu(self):
        """The gather/scatter path works on a CPU-finalized model, not just CUDA."""
        model = _replicated_model(2, device="cpu")
        state = model.state()

        view = DeformableView(model, "/World/Cloth", family="surface")
        positions = view.get_particle_positions(state)
        self.assertTrue(positions.device.is_cpu)
        lifted = positions.numpy()
        lifted[..., 2] += 1.0
        view.set_particle_positions(state, wp.array(lifted, dtype=wp.vec3, device="cpu"))
        np.testing.assert_allclose(view.get_particle_positions(state).numpy(), lifted, atol=1e-6)

    def test_selection_errors(self):
        """No match raises KeyError; ragged element counts and bad families raise ValueError."""
        builder = newton.ModelBuilder()
        _add_test_cloth(builder, label="/World/ClothA")  # 4 particles
        _add_test_cloth(
            builder,
            label="/World/ClothB",
            vertices=[
                (0.0, 0.0, 0.0),
                (1.0, 0.0, 0.0),
                (1.0, 1.0, 0.0),
                (0.0, 1.0, 0.0),
                (2.0, 0.0, 0.0),
            ],
            indices=[0, 1, 2, 0, 2, 3, 1, 4, 2],
        )
        model = builder.finalize()

        with self.assertRaises(KeyError):
            DeformableView(model, "/World/DoesNotExist", family="surface")
        view = DeformableView(model, "/World/Cloth*", family="surface")
        self.assertEqual([end - start for start, end in view.ranges("particle")], [4, 5])
        np.testing.assert_array_equal(view.starts("particle").numpy(), [0, 4])
        with self.assertRaisesRegex(ValueError, "Varying particle counts.*ranges"):
            view.elements_per_group("particle")
        with self.assertRaisesRegex(ValueError, "Varying particle counts.*ranges"):
            view.get_particle_positions(model.state())
        with self.assertRaisesRegex(ValueError, "Varying particle counts.*ranges"):
            view.set_particle_positions(model.state(), wp.zeros((2, 4), dtype=wp.vec3))
        with self.assertRaisesRegex(ValueError, "Unknown deformable family"):
            DeformableView(model, "/World/ClothA", family="ropes")
        view = DeformableView(model, "/World/ClothA", family="surface")
        with self.assertRaisesRegex(AttributeError, "no body elements"):
            view.get_body_transforms(model.state())
        with self.assertRaises(ValueError):
            view.set_particle_positions(model.state(), wp.zeros((2, 4), dtype=wp.vec3))

    def test_indexed_partial_writes_touch_only_selected_groups(self):
        """group_indices= scatters into selected groups only, from host and device index
        forms, and cable body velocities round-trip through an indexed write."""
        model = _replicated_model(3)
        state = model.state()

        cloth = DeformableView(model, "/World/Cloth", family="surface")
        before = cloth.get_particle_positions(state).numpy()
        moved = before[[1]].copy()
        moved[..., 2] += 5.0
        cloth.set_particle_positions(state, wp.array(moved, dtype=wp.vec3), group_indices=[1])
        after = cloth.get_particle_positions(state).numpy()
        np.testing.assert_array_equal(after[[0, 2]], before[[0, 2]])
        np.testing.assert_allclose(after[1], moved[0], atol=1e-6)

        cable = DeformableView(model, "/World/Cable", family="curve")
        velocities = np.zeros((1, cable.bodies_per_group, 6), dtype=np.float32)
        velocities[..., 3] = 2.0
        device_indices = wp.array([2], dtype=wp.int32, device=model.device)
        cable.set_body_velocities(state, wp.array(velocities, dtype=wp.spatial_vector), group_indices=device_indices)
        out = cable.get_body_velocities(state).numpy()
        np.testing.assert_allclose(out[2], velocities[0], atol=1e-6)
        np.testing.assert_array_equal(out[:2], np.zeros_like(out[:2]))

        with self.assertRaisesRegex(ValueError, "must be in"):
            cloth.set_particle_positions(state, wp.array(moved, dtype=wp.vec3), group_indices=[3])
        with self.assertRaisesRegex(ValueError, "duplicate"):
            cloth.set_particle_positions(
                state,
                wp.array(np.repeat(moved, 2, axis=0), dtype=wp.vec3),
                group_indices=[1, 1],
            )

    def test_host_indices_require_integral_values(self):
        """Host selectors reject lossy coercions before they can write another group."""
        model = _replicated_model(3, device="cpu")
        cloth = DeformableView(model, "/World/Cloth", family="surface")
        values = wp.full((1, cloth.particles_per_group), 9.0, dtype=wp.vec3, device="cpu")

        for index_argument in ("group_indices", "world_indices"):
            for invalid_indices in ([-0.2], [1.9], ["1"], [False], [True]):
                with self.subTest(index_argument=index_argument, invalid_indices=invalid_indices):
                    state = model.state()
                    before = cloth.get_particle_positions(state).numpy()
                    with self.assertRaisesRegex(TypeError, index_argument):
                        cloth.set_particle_positions(state, values, **{index_argument: invalid_indices})
                    np.testing.assert_array_equal(cloth.get_particle_positions(state).numpy(), before)

            with self.subTest(index_argument=index_argument, valid_indices="numpy integer"):
                state = model.state()
                before = cloth.get_particle_positions(state).numpy()
                cloth.set_particle_positions(state, values, **{index_argument: [np.int64(1)]})
                after = cloth.get_particle_positions(state).numpy()
                np.testing.assert_array_equal(after[[0, 2]], before[[0, 2]])
                np.testing.assert_array_equal(after[1], np.full_like(after[1], 9.0))

            with self.subTest(index_argument=index_argument, valid_indices="empty"):
                state = model.state()
                before = cloth.get_particle_positions(state).numpy()
                empty_values = wp.empty((0, cloth.particles_per_group), dtype=wp.vec3, device="cpu")
                cloth.set_particle_positions(state, empty_values, **{index_argument: []})
                np.testing.assert_array_equal(cloth.get_particle_positions(state).numpy(), before)

    def test_invalid_device_group_index_does_not_write(self):
        """An out-of-range device group index cannot address another group's state."""
        model = _replicated_model(3)
        state = model.state()
        cloth = DeformableView(model, "/World/Cloth", family="surface")
        before = cloth.get_particle_positions(state).numpy()
        values = np.full((1, cloth.particles_per_group, 3), 17.0, dtype=np.float32)

        cloth.set_particle_positions(
            state,
            wp.array(values, dtype=wp.vec3, device=model.device),
            group_indices=wp.array([cloth.count], dtype=wp.int32, device=model.device),
        )

        np.testing.assert_array_equal(cloth.get_particle_positions(state).numpy(), before)

    def test_device_group_indices_must_be_one_dimensional(self):
        """Device index arrays fail clearly before reaching a one-dimensional kernel input."""
        model = _replicated_model(3)
        state = model.state()
        cloth = DeformableView(model, "/World/Cloth", family="surface")
        values = wp.zeros((1, cloth.particles_per_group), dtype=wp.vec3, device=model.device)
        indices = wp.zeros((1, 2), dtype=wp.int32, device=model.device)

        with self.assertRaisesRegex(ValueError, "one-dimensional"):
            cloth.set_particle_positions(state, values, group_indices=indices)

    @unittest.skipUnless(wp.is_cuda_available(), "Requires CUDA graph capture")
    def test_device_index_writes_capture_and_replay(self):
        """Captured indexed setters reuse changing device values without unsafe writes."""
        device = wp.get_device("cuda:0")
        model = _replicated_model(3, device=device)
        cloth = DeformableView(model, "/World/Cloth", family="surface")

        for index_argument in ("group_indices", "world_indices"):
            with self.subTest(index_argument=index_argument):
                state = model.state()
                initial = cloth.get_particle_positions(state).numpy().copy()
                values_np = np.zeros((2, cloth.particles_per_group, 3), dtype=np.float32)
                values = wp.array(values_np, dtype=wp.vec3, device=device)
                indices = wp.array([cloth.count, cloth.count], dtype=wp.int32, device=device)

                # Compile every captured operation without changing state.
                cloth.set_particle_positions(state, values, **{index_argument: indices})

                indices.assign(np.array([1, 1], dtype=np.int32))
                values_np[0].fill(3.0)
                values_np[1].fill(8.0)
                values.assign(values_np)
                with wp.ScopedCapture(device) as capture:
                    cloth.set_particle_positions(state, values, **{index_argument: indices})

                wp.capture_launch(capture.graph)
                expected = initial.copy()
                expected[1].fill(8.0)
                np.testing.assert_allclose(cloth.get_particle_positions(state).numpy(), expected, atol=1e-6)

                indices.assign(np.array([0, 2], dtype=np.int32))
                values_np[0].fill(4.0)
                values_np[1].fill(9.0)
                values.assign(values_np)
                wp.capture_launch(capture.graph)
                expected[0].fill(4.0)
                expected[2].fill(9.0)
                np.testing.assert_allclose(cloth.get_particle_positions(state).numpy(), expected, atol=1e-6)

                before_invalid = cloth.get_particle_positions(state).numpy().copy()
                indices.assign(np.array([-1, cloth.count], dtype=np.int32))
                values_np.fill(12.0)
                values.assign(values_np)
                wp.capture_launch(capture.graph)
                np.testing.assert_array_equal(cloth.get_particle_positions(state).numpy(), before_invalid)

    def test_duplicate_device_group_index_uses_last_value(self):
        """Device duplicate indices are deterministic: the last row wins."""
        model = _replicated_model(3)
        state = model.state()
        cloth = DeformableView(model, "/World/Cloth", family="surface")
        before = cloth.get_particle_positions(state).numpy()
        values = np.empty((2, cloth.particles_per_group, 3), dtype=np.float32)
        values[0].fill(3.0)
        values[1].fill(8.0)

        cloth.set_particle_positions(
            state,
            wp.array(values, dtype=wp.vec3, device=model.device),
            group_indices=wp.array([1, 1], dtype=wp.int32, device=model.device),
        )

        after = cloth.get_particle_positions(state).numpy()
        np.testing.assert_array_equal(after[[0, 2]], before[[0, 2]])
        np.testing.assert_array_equal(after[1], values[1])

    def test_world_indices_write_one_group_per_world(self):
        """World indices address actual model worlds when each world has one match."""
        model = _replicated_model(3)
        state = model.state()
        cloth = DeformableView(model, "/World/Cloth", family="surface")
        before = cloth.get_particle_positions(state).numpy()
        moved = before[[2]].copy()
        moved[..., 2] += 4.0

        cloth.set_particle_positions(
            state,
            wp.array(moved, dtype=wp.vec3, device=model.device),
            world_indices=wp.array([2], dtype=wp.int32, device=model.device),
        )

        after = cloth.get_particle_positions(state).numpy()
        np.testing.assert_array_equal(after[:2], before[:2])
        np.testing.assert_allclose(after[2], moved[0], atol=1e-6)


class TestDeformableAndArticulationViews(unittest.TestCase):
    """Rigid and deformable selections sharing finalized mixed models."""

    def test_rigid_cable_cloth_and_volume_survive_unrelated_collapse(self):
        """All selection families coexist after an unrelated fixed joint collapses."""
        builder = newton.ModelBuilder()
        collapsed_body = builder.add_link(label="collapsed_body")
        builder.add_joint_fixed(parent=-1, child=collapsed_body, label="collapse_me")
        _add_test_articulation(builder)
        cable_bodies, _cable_joints = builder.add_rod(
            positions=[(0.0, 2.0, 1.0), (0.1, 2.0, 1.0), (0.2, 2.0, 1.0)],
            radius=0.02,
            label="cable",
            body_frame_origin="com",
        )
        _add_test_cloth(builder)
        _add_test_soft_body(builder)

        body_count = builder.body_count
        joint_count = builder.joint_count
        builder.collapse_fixed_joints()
        self.assertEqual(builder.body_count, body_count - 1)
        self.assertEqual(builder.joint_count, joint_count - 1)

        model = builder.finalize()
        state = model.state()
        rigid = ArticulationView(model, "robot")
        cable = DeformableView(model, "cable", family="curve")
        cloth = DeformableView(model, "cloth", family="surface")
        soft = DeformableView(model, "soft", family="volume")

        self.assertEqual(rigid.get_root_transforms(state).shape, (1, 1))
        self.assertEqual(cable.get_body_transforms(state).shape, (1, 2))
        self.assertEqual(cloth.get_particle_positions(state).shape, (1, 4))
        self.assertEqual(soft.get_particle_positions(state).shape, (1, 4))
        self.assertEqual(cable.ranges("body"), [(cable_bodies[0] - 1, cable_bodies[-1])])

        rigid_values = rigid.get_root_transforms(state).numpy()
        cable_values = cable.get_body_transforms(state).numpy()
        cloth_values = cloth.get_particle_positions(state).numpy()
        soft_values = soft.get_particle_positions(state).numpy()

        moved_rigid = rigid_values.copy()
        moved_rigid[..., 0] += 1.0
        rigid.set_root_transforms(state, wp.array(moved_rigid, dtype=wp.transform, device=model.device))
        np.testing.assert_array_equal(cable.get_body_transforms(state).numpy(), cable_values)
        np.testing.assert_array_equal(cloth.get_particle_positions(state).numpy(), cloth_values)
        np.testing.assert_array_equal(soft.get_particle_positions(state).numpy(), soft_values)

        moved_cable = cable_values.copy()
        moved_cable[..., 1] += 1.0
        cable.set_body_transforms(state, wp.array(moved_cable, dtype=wp.transform, device=model.device))
        np.testing.assert_allclose(rigid.get_root_transforms(state).numpy(), moved_rigid, atol=1e-6)
        np.testing.assert_array_equal(cloth.get_particle_positions(state).numpy(), cloth_values)
        np.testing.assert_array_equal(soft.get_particle_positions(state).numpy(), soft_values)

        moved_cloth = cloth_values.copy()
        moved_cloth[..., 2] += 1.0
        cloth.set_particle_positions(state, wp.array(moved_cloth, dtype=wp.vec3, device=model.device))
        np.testing.assert_allclose(cable.get_body_transforms(state).numpy(), moved_cable, atol=1e-6)
        np.testing.assert_array_equal(soft.get_particle_positions(state).numpy(), soft_values)

        moved_soft = soft_values.copy()
        moved_soft[..., 0] += 1.0
        soft.set_particle_positions(state, wp.array(moved_soft, dtype=wp.vec3, device=model.device))
        np.testing.assert_allclose(rigid.get_root_transforms(state).numpy(), moved_rigid, atol=1e-6)
        np.testing.assert_allclose(cable.get_body_transforms(state).numpy(), moved_cable, atol=1e-6)
        np.testing.assert_allclose(cloth.get_particle_positions(state).numpy(), moved_cloth, atol=1e-6)
        np.testing.assert_allclose(soft.get_particle_positions(state).numpy(), moved_soft, atol=1e-6)

        state_out = model.state()
        newton.solvers.SolverXPBD(model, iterations=1).step(
            state,
            state_out,
            control=None,
            contacts=None,
            dt=1.0 / 60.0,
        )
        self.assertTrue(np.isfinite(state_out.body_q.numpy()).all())
        self.assertTrue(np.isfinite(state_out.particle_q.numpy()).all())

    def test_views_update_disjoint_state_and_simulate(self):
        """Rigid, cloth, and volume views coexist across replicated worlds."""
        prototype = newton.ModelBuilder()
        _add_test_articulation(prototype)
        _add_test_cloth(prototype)
        _add_test_soft_body(prototype)
        scene = newton.ModelBuilder()
        scene.replicate(prototype, 2)
        model = scene.finalize()
        state = model.state()

        rigid = ArticulationView(model, "robot")
        cloth = DeformableView(model, "cloth", family="surface")
        soft = DeformableView(model, "soft", family="volume")
        self.assertEqual(rigid.get_root_transforms(state).shape, (2, 1))
        self.assertEqual(cloth.get_particle_positions(state).shape, (2, 4))
        self.assertEqual(soft.get_particle_positions(state).shape, (2, 4))

        rigid_before = rigid.get_root_transforms(state).numpy().copy()
        soft_before = soft.get_particle_positions(state).numpy().copy()
        cloth_values = cloth.get_particle_positions(state).numpy()
        cloth_values[1, :, 2] += 2.0
        cloth.set_particle_positions(state, wp.array(cloth_values, dtype=wp.vec3, device=model.device))
        np.testing.assert_array_equal(rigid.get_root_transforms(state).numpy(), rigid_before)
        np.testing.assert_array_equal(soft.get_particle_positions(state).numpy(), soft_before)

        rigid_values = rigid_before.copy()
        rigid_values[0, 0, 0] += 1.0
        rigid.set_root_transforms(state, wp.array(rigid_values, dtype=wp.transform, device=model.device))
        np.testing.assert_allclose(rigid.get_root_transforms(state).numpy(), rigid_values, atol=1e-6)
        np.testing.assert_array_equal(cloth.get_particle_positions(state).numpy(), cloth_values)

        state_out = model.state()
        newton.solvers.SolverXPBD(model, iterations=1).step(
            state,
            state_out,
            control=None,
            contacts=None,
            dt=1.0 / 60.0,
        )
        self.assertTrue(np.isfinite(state_out.body_q.numpy()).all())
        self.assertTrue(np.isfinite(state_out.particle_q.numpy()).all())

    def test_heterogeneous_deformables_keep_rigid_selection_uniform(self):
        """Family views retain empty worlds beside one common rigid articulation."""
        world_0 = newton.ModelBuilder()
        _add_test_articulation(world_0)
        _add_test_cloth(world_0)
        world_1 = newton.ModelBuilder()
        _add_test_articulation(world_1)
        _add_test_soft_body(world_1)
        world_2 = newton.ModelBuilder()
        _add_test_articulation(world_2)
        _add_test_cloth(world_2)
        world_2.add_rod(
            positions=[(0.0, 2.0, 1.0), (0.1, 2.0, 1.0), (0.2, 2.0, 1.0)],
            radius=0.02,
            label="cable",
            body_frame_origin="com",
        )
        scene = newton.ModelBuilder()
        scene.add_world(world_0)
        scene.add_world(world_1)
        scene.add_world(world_2)
        model = scene.finalize()

        rigid = ArticulationView(model, "robot")
        cloth = DeformableView(model, "cloth", family="surface")
        soft = DeformableView(model, "soft", family="volume")
        cable = DeformableView(model, "cable", family="curve")
        self.assertEqual((rigid.count, rigid.world_count, rigid.count_per_world), (3, 3, 1))
        self.assertEqual(rigid.get_root_transforms(model).shape, (3, 1))
        np.testing.assert_array_equal(cloth.world_starts.numpy(), [0, 1, 1, 2])
        np.testing.assert_array_equal(soft.world_starts.numpy(), [0, 0, 1, 1])
        np.testing.assert_array_equal(cable.world_starts.numpy(), [0, 0, 0, 1])

    def test_rigid_view_survives_dropped_curve_group(self):
        """Dropping an incomplete curve leaves an unrelated rigid view valid."""
        builder = newton.ModelBuilder()
        _add_test_articulation(builder)
        bodies, _joints = builder.add_rod(
            positions=[(0.0, 0.0, 1.0), (0.1, 0.0, 1.0), (0.2, 0.0, 1.0)],
            radius=0.02,
            label="anchored_curve",
            body_frame_origin="com",
        )
        builder.add_joint_fixed(-1, bodies[0], label="anchor")

        with self.assertWarnsRegex(UserWarning, "anchored_curve.*joints_to_keep"):
            builder.collapse_fixed_joints()
        model = builder.finalize()

        rigid = ArticulationView(model, "robot")
        self.assertEqual(rigid.get_root_transforms(model).shape, (1, 1))
        with self.assertRaisesRegex(KeyError, "anchored_curve"):
            DeformableView(model, "anchored_curve", family="curve")


class TestDeformableViewBuilderGroups(unittest.TestCase):
    """Groups recorded by labeled builder calls (no USD) are selectable through the view."""

    def test_labeled_curve_builders_record_one_complete_group(self):
        """Public rod builders hide their nested construction from group selection."""
        closed_builder = newton.ModelBuilder()
        closed_builder.add_rod(
            positions=[(0.0, 0.0, 1.0), (0.1, 0.0, 1.0), (0.1, 0.1, 1.0), (0.0, 0.0, 1.0)],
            radius=0.02,
            closed=True,
            label="closed",
            body_frame_origin="com",
        )
        closed = DeformableView(closed_builder.finalize(), "closed", family="curve")
        self.assertEqual((closed.count, closed.elements_per_group("body")), (1, 3))
        self.assertEqual(closed.elements_per_group("joint"), 3)

        graph_builder = newton.ModelBuilder()
        graph_builder.add_rod_graph(
            node_positions=[(0.0, 0.0, 1.0), (0.1, 0.0, 1.0), (0.2, 0.0, 1.0), (0.1, 0.1, 1.0)],
            edges=[(0, 1), (1, 2), (1, 3)],
            radius=0.02,
            label="graph",
            body_frame_origin="com",
        )
        graph = DeformableView(graph_builder.finalize(), "graph", family="curve")
        self.assertEqual((graph.count, graph.elements_per_group("body")), (1, 3))
        self.assertEqual(graph.elements_per_group("joint"), 2)

    def test_builder_built_prototype_clones_select_per_world(self):
        """A labeled soft body and rod built in a prototype and cloned per world with
        add_world stay selectable, with correctly offset ranges (the Isaac Lab pattern
        of building deformables in per-world builder hooks)."""
        proto = newton.ModelBuilder()
        proto.add_soft_mesh(
            pos=wp.vec3(0.0, 0.0, 1.0),
            rot=wp.quat_identity(),
            scale=1.0,
            vel=wp.vec3(0.0, 0.0, 0.0),
            vertices=[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)],
            indices=[0, 1, 2, 3],
            density=100.0,
            k_mu=1.0e4,
            k_lambda=1.0e4,
            k_damp=0.0,
            label="soft_proto",
        )
        proto.add_rod(
            positions=[(0.0, 2.0, 1.0), (0.1, 2.0, 1.0), (0.2, 2.0, 1.0)],
            radius=0.02,
            label="cable_proto",
            wrap_in_articulation=True,
            body_frame_origin="com",
        )

        scene = newton.ModelBuilder()
        scene.add_world(proto)
        scene.add_world(proto)
        model = scene.finalize()
        state = model.state()

        soft = DeformableView(model, "soft_proto", family="volume")
        self.assertEqual((soft.count, soft.worlds, soft.particles_per_group), (2, [0, 1], 4))
        (r0, r1) = soft.ranges("particle")
        self.assertEqual(r1[0] - r0[0], 4)
        self.assertNotEqual(r0, r1)

        cable = DeformableView(model, "cable_proto", family="curve")
        self.assertEqual((cable.count, cable.worlds, cable.bodies_per_group), (2, [0, 1], 2))
        self.assertEqual(cable.elements_per_group("joint"), 1)

        # State access round-trips through the offset ranges.
        positions = soft.get_particle_positions(state)
        lifted = positions.numpy()
        lifted[1, :, 2] += 3.0
        soft.set_particle_positions(state, wp.array(lifted, dtype=wp.vec3))
        np.testing.assert_allclose(soft.get_particle_positions(state).numpy(), lifted, atol=1e-6)

    def test_unlabeled_builder_deformables_record_no_group(self):
        """Deformables built without a label stay undiscoverable (no accidental groups)."""
        builder = newton.ModelBuilder()
        builder.add_cloth_mesh(
            pos=wp.vec3(0.0, 0.0, 1.0),
            rot=wp.quat_identity(),
            scale=1.0,
            vel=wp.vec3(0.0, 0.0, 0.0),
            vertices=[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0)],
            indices=[0, 1, 2, 0, 2, 3],
            density=0.1,
        )
        model = builder.finalize()
        with self.assertRaises(KeyError):
            DeformableView(model, "*", family="surface")

    def test_fixed_joint_collapse_drops_incomplete_curve_group(self):
        """A label does not prevent collapse; an incomplete curve is not selectable."""
        builder = newton.ModelBuilder()
        bodies, _joints = builder.add_rod(
            positions=[(0.0, 0.0, 1.0), (0.1, 0.0, 1.0), (0.2, 0.0, 1.0)],
            radius=0.02,
            label="anchored_curve",
            body_frame_origin="com",
        )
        builder.add_joint_fixed(-1, bodies[0], label="anchor")

        with self.assertWarnsRegex(UserWarning, "anchored_curve.*joints_to_keep"):
            builder.collapse_fixed_joints()

        self.assertEqual((builder.body_count, builder.joint_count), (1, 1))
        model = builder.finalize()

        with self.assertRaisesRegex(KeyError, "anchored_curve"):
            DeformableView(model, "anchored_curve", family="curve")

    def test_fixed_joint_collapse_is_label_neutral(self):
        """Otherwise identical labeled and unlabeled rods collapse identically."""

        def anchored_rod(label):
            builder = newton.ModelBuilder()
            bodies, _joints = builder.add_rod(
                positions=[(0.0, 0.0, 1.0), (0.1, 0.0, 1.0), (0.2, 0.0, 1.0)],
                radius=0.02,
                label=label,
                body_frame_origin="com",
            )
            builder.add_joint_fixed(-1, bodies[0], label="anchor")
            return builder

        unlabeled = anchored_rod(None)
        unlabeled.collapse_fixed_joints()
        labeled = anchored_rod("anchored_curve")
        with self.assertWarnsRegex(UserWarning, "anchored_curve.*joints_to_keep"):
            labeled.collapse_fixed_joints()

        self.assertEqual((labeled.body_count, labeled.joint_count), (unlabeled.body_count, unlabeled.joint_count))
        self.assertEqual(labeled.joint_type, unlabeled.joint_type)

    def test_fixed_joint_collapse_preserves_explicitly_kept_curve(self):
        """joints_to_keep retains a complete curve group when requested."""
        builder = newton.ModelBuilder()
        bodies, _joints = builder.add_rod(
            positions=[(0.0, 0.0, 1.0), (0.1, 0.0, 1.0), (0.2, 0.0, 1.0)],
            radius=0.02,
            label="anchored_curve",
            body_frame_origin="com",
        )
        builder.add_joint_fixed(-1, bodies[0], label="anchor")

        builder.collapse_fixed_joints(joints_to_keep=["anchor"])
        model = builder.finalize()
        view = DeformableView(model, "anchored_curve", family="curve")

        self.assertEqual((model.body_count, model.joint_count), (2, 2))
        self.assertEqual(view.ranges("body"), [(0, 2)])
        self.assertEqual(view.get_body_transforms(model.state()).shape, (1, 2))

    def test_labeled_soft_grid_is_selectable(self):
        """A labeled soft grid records one selectable volume group."""
        builder = newton.ModelBuilder()
        builder.add_soft_grid(
            pos=wp.vec3(0.0, 0.0, 0.0),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0, 0.0, 0.0),
            dim_x=1,
            dim_y=1,
            dim_z=1,
            cell_x=1.0,
            cell_y=1.0,
            cell_z=1.0,
            density=1.0,
            k_mu=1.0,
            k_lambda=1.0,
            k_damp=0.0,
            label="soft_grid",
        )

        model = builder.finalize()
        view = DeformableView(model, "soft_grid", family="volume")

        self.assertEqual(view.ranges("particle"), [(0, 8)])
        self.assertEqual(view.ranges("tetrahedron"), [(0, 5)])


if __name__ == "__main__":
    unittest.main(verbosity=2)
