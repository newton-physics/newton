# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import gc
import itertools
import unittest
import weakref
from unittest import mock

import numpy as np
import warp as wp

import newton
from newton._src.sim.collide import _estimate_rigid_contact_max_per_world
from newton._src.sim.model import _pack_shape_pair_codes
from newton._src.sim.shape_contact_pairs import (
    _shape_contact_pair_count_for_mask,
    _shape_contact_pairs_for_mask,
    _ShapeContactPairs,
)
from newton._src.viewer.viewer_file import depointer_as_key, pointer_as_key, transfer_to_model
from newton.tests.unittest_utils import add_function_test, get_selected_cuda_test_devices, get_test_devices


def _make_builder():
    builder = newton.ModelBuilder()
    builder.add_ground_plane()
    for world_size in (3, 5):
        builder.begin_world()
        shapes = []
        for i in range(world_size):
            body = builder.add_body(xform=wp.transform(wp.vec3(0.1 * i, 0.0, 0.2)))
            cfg = newton.ModelBuilder.ShapeConfig(collision_group=(1, -1, -2, 0, 2)[i])
            shapes.append(builder.add_shape_sphere(body, radius=0.3, cfg=cfg))
        builder.add_shape_sphere(body, radius=0.2)
        builder.add_shape_sphere(body, radius=0.4, cfg=newton.ModelBuilder.ShapeConfig(has_shape_collision=False))
        builder.add_shape_collision_filter_pair(shapes[0], shapes[1])
        builder.end_world()
    builder.add_shape_sphere(builder.add_body(), radius=0.2)
    return builder


def _reference_pairs(bodies, worlds, groups, flags, filters):
    filters = {tuple(sorted(pair)) for pair in filters}
    pairs = []
    for a, b in itertools.combinations(range(len(bodies)), 2):
        if not (flags[a] & newton.ShapeFlags.COLLIDE_SHAPES and flags[b] & newton.ShapeFlags.COLLIDE_SHAPES):
            continue
        if bodies[a] == bodies[b] or (bodies[a] < 0 and bodies[b] < 0):
            continue
        if worlds[a] != worlds[b] and worlds[a] != -1 and worlds[b] != -1:
            continue
        ga, gb = groups[a], groups[b]
        if ga == 0 or gb == 0:
            continue
        if ga > 0:
            compatible = ga == gb or gb < 0
        else:
            compatible = ga != gb
        if compatible and (a, b) not in filters:
            pairs.append((a, b))
    return np.asarray(pairs, dtype=np.int32).reshape((-1, 2))


def _sorted_pairs(pairs):
    return pairs[np.lexsort((pairs[:, 1], pairs[:, 0]))]


class TestShapeContactPairs(unittest.TestCase):
    def test_finalize_defers_pair_table(self):
        """Keep finalization compact while preserving the exact pair count."""
        builder = newton.ModelBuilder()
        for _ in range(4):
            builder.add_shape_sphere(builder.add_body())
        model = builder.finalize(device="cpu")
        self.assertEqual(model.shape_contact_pair_count, 6)
        self.assertIsNone(vars(model).get("shape_contact_pairs"))

    def test_randomized_topology(self):
        """Match an exhaustive oracle across groups, bodies, worlds, masks and filters."""
        rng = np.random.default_rng(4394)
        for case in range(250):
            with self.subTest(case=case):
                size = int(rng.integers(0, 90))
                bodies = rng.integers(-3, 16, size=size)
                worlds = rng.integers(-1, 6, size=size)
                groups = rng.choice([-2147483648, -4, -3, -2, -1, 0, 1, 2, 3, 4, 2147483647], size=size)
                flags = rng.choice([0, int(newton.ShapeFlags.COLLIDE_SHAPES)], size=size, p=[0.1, 0.9])
                filters = rng.integers(0, size, size=(size * 2, 2)) if size else np.empty((0, 2), dtype=int)
                packed = np.unique(_pack_shape_pair_codes(filters[:, 0], filters[:, 1]))
                data = _ShapeContactPairs(bodies, worlds, groups, flags, packed, 6)
                expected = _reference_pairs(bodies, worlds, groups, flags, filters)
                np.testing.assert_array_equal(
                    data.counts, np.bincount(np.max(worlds[expected], axis=1) + 1, minlength=7)
                )
                with mock.patch("newton._src.sim.shape_contact_pairs._PAIR_CHUNK_SIZE", 23):
                    np.testing.assert_array_equal(_sorted_pairs(data.build_pairs()), expected)
                    mask = rng.random(size) < 0.5
                    subset = expected[mask[expected].all(axis=1)]
                    np.testing.assert_array_equal(_sorted_pairs(data.build_pairs(mask)), subset)
                    self.assertEqual(int(data.count_pairs(mask).sum()), len(subset))

    def test_replicated_templates(self):
        """Replay heterogeneous templates without confusing global and local filters."""
        builder = newton.ModelBuilder()
        global_body = builder.add_body()
        builder.add_shape_sphere(global_body)
        for world in range(12):
            builder.begin_world()
            first = builder.shape_count
            for i in range(5 + world % 2):
                body = global_body if i == 1 and world % 3 == 0 else builder.add_body()
                builder.add_shape_sphere(body)
            # Global index zero and local offset zero must remain distinct in
            # template keys, even though both normalize to zero without tagging.
            builder.add_shape_collision_filter_pair(0 if world % 4 == 0 else first, first + 2)
            builder.end_world()
        builder.add_ground_plane()
        expected = _reference_pairs(
            builder.shape_body,
            builder.shape_world,
            builder.shape_collision_group,
            builder.shape_flags,
            builder.shape_collision_filter_pairs,
        )
        model = builder.finalize(device="cpu")
        with mock.patch("newton._src.sim.shape_contact_pairs._PAIR_CHUNK_SIZE", 31):
            np.testing.assert_array_equal(_sorted_pairs(model.shape_contact_pairs.numpy()), expected)

    def test_uniform_group_counts(self):
        """Match the vectorized histogram path with globals and repeated bodies."""
        rng = np.random.default_rng(4395)
        for _ in range(8):
            size = 300
            bodies = rng.choice([-3, -1, 0, 1, 2, 2147483647], size=size)
            worlds = rng.integers(-1, 4, size=size)
            groups = np.full(size, 2147483647)
            flags = np.full(size, newton.ShapeFlags.COLLIDE_SHAPES)
            filters = rng.integers(0, size, size=(400, 2))
            packed = np.unique(_pack_shape_pair_codes(filters[:, 0], filters[:, 1]))
            data = _ShapeContactPairs(bodies, worlds, groups, flags, packed, 4)
            expected = _reference_pairs(bodies, worlds, groups, flags, filters)
            np.testing.assert_array_equal(data.counts, np.bincount(np.max(worlds[expected], axis=1) + 1, minlength=5))
            np.testing.assert_array_equal(_sorted_pairs(data.build_pairs()), expected)

    def test_many_worlds_with_compound_bodies(self):
        """Keep mixed-group counting and replay sparse across thousands of worlds."""
        world_count = 10000
        bodies = np.append(np.repeat(np.arange(world_count * 2), 2), -1)
        worlds = np.append(np.repeat(np.arange(world_count), 4), -1)
        groups = np.append(np.tile([1, -1, 1, -1], world_count), 1)
        data = _ShapeContactPairs(
            bodies, worlds, groups, np.full(len(bodies), newton.ShapeFlags.COLLIDE_SHAPES), [], world_count
        )
        # Three intra-world pairs plus four pairs with the trailing global plane.
        np.testing.assert_array_equal(data.counts, np.append(0, np.full(world_count, 7)))
        pairs = data.build_pairs()
        self.assertEqual(len(pairs), world_count * 7)
        np.testing.assert_array_equal(
            np.bincount(np.max(worlds[pairs], axis=1) + 1, minlength=world_count + 1), data.counts
        )
        self.assertTrue((bodies[pairs[:, 0]] != bodies[pairs[:, 1]]).all())

    def test_explicit_override_defers_default_table(self):
        """Use caller-supplied explicit pairs without constructing the default table."""
        model = _make_builder().finalize(device="cpu")
        pairs = wp.array([[0, 1]], dtype=wp.vec2i, device="cpu")
        with mock.patch.object(_ShapeContactPairs, "build_pairs", side_effect=AssertionError("Enumerated pairs")):
            pipeline = newton.CollisionPipeline(model, shape_pairs_filtered=pairs)
        self.assertIs(pipeline.shape_pairs_filtered, pairs)
        self.assertIsNotNone(model._shape_contact_pair_data)

    def test_builder_snapshot_lifetime(self):
        """Keep finalized topology independent of later edits and builder lifetime."""
        builder = _make_builder()
        expected = _reference_pairs(
            builder.shape_body,
            builder.shape_world,
            builder.shape_collision_group,
            builder.shape_flags,
            builder.shape_collision_filter_pairs,
        )
        model = builder.finalize(device="cpu")
        builder.shape_body[1] = -1
        builder.shape_collision_group[1] = 0
        builder.shape_collision_filter_pairs.append((0, 1))
        model.shape_collision_group.zero_()
        model.shape_flags.zero_()
        reference = weakref.ref(builder)
        del builder
        gc.collect()
        self.assertIsNone(reference())
        self.assertEqual(model.shape_contact_pair_count, len(expected))
        np.testing.assert_array_equal(_sorted_pairs(model.shape_contact_pairs.numpy()), expected)

    def test_pair_cache_and_overrides(self):
        """Build once and preserve direct pair and count overrides."""
        model = _make_builder().finalize(device="cpu")
        count = model.shape_contact_pair_count
        data = model._shape_contact_pair_data
        model.shape_contact_pair_count = 0
        with mock.patch.object(data, "build_pairs", wraps=data.build_pairs) as build:
            pairs = model.shape_contact_pairs
            self.assertIs(model.shape_contact_pairs, pairs)
            build.assert_called_once_with()
        self.assertEqual(len(pairs), count)
        self.assertEqual(model.shape_contact_pair_count, 0)
        self.assertIsNone(model._shape_contact_pair_data)

        for replacement in (None, wp.array([[0, 1]], dtype=wp.vec2i, device="cpu")):
            with self.subTest(replacement=replacement):
                model = _make_builder().finalize(device="cpu")
                model.shape_contact_pairs = replacement
                self.assertIs(model.shape_contact_pairs, replacement)
                self.assertIsNone(model._shape_contact_pair_data)
                self.assertEqual(model.shape_contact_pair_count, count)

    def test_filter_validation(self):
        """Validate collision filters during finalization before any pair request."""
        builder = _make_builder()
        builder.shape_collision_filter_pairs.append((0, builder.shape_count))
        with self.assertRaisesRegex(ValueError, "contains invalid pair"):
            builder.finalize(device="cpu")

    def test_empty_models_and_filtered_scenes(self):
        """Avoid candidate enumeration for empty and fully excluded scenes."""
        empty = newton.ModelBuilder().finalize(device="cpu")
        self.assertEqual(empty.shape_contact_pair_count, 0)
        self.assertEqual(empty.shape_contact_pairs.shape, (0,))
        size = 20000
        for bodies, groups in (
            (np.zeros(size), np.ones(size)),
            (np.arange(size), -np.ones(size)),
            (np.arange(size), np.arange(1, size + 1)),
            (np.arange(size), np.zeros(size)),
        ):
            data = _ShapeContactPairs(
                bodies, -np.ones(size), groups, np.full(size, newton.ShapeFlags.COLLIDE_SHAPES), [], 0
            )
            with mock.patch.object(data, "_write_pairs", side_effect=AssertionError("Enumerated excluded pairs")):
                self.assertEqual(int(data.counts.sum()), 0)
                self.assertEqual(data.build_pairs().shape, (0, 2))

    def test_subset_and_per_world_counts(self):
        """Match subset and weighted per-world counts without building all pairs."""
        model = _make_builder().finalize(device="cpu")
        mask = np.arange(model.shape_count) % 3 != 0
        types = model.shape_type.numpy()
        types[::3] = int(newton.GeoType.MESH)
        model.shape_type.assign(types)
        with mock.patch.object(_ShapeContactPairs, "build_pairs", side_effect=AssertionError("Enumerated pairs")):
            estimate = _estimate_rigid_contact_max_per_world(model, 10000)
            subset_count = _shape_contact_pair_count_for_mask(model, mask)
        subset = _shape_contact_pairs_for_mask(model, mask)
        self.assertIsNone(vars(model)["shape_contact_pairs"])
        pairs = model.shape_contact_pairs.numpy()
        np.testing.assert_array_equal(_sorted_pairs(subset), _sorted_pairs(pairs[mask[pairs].all(axis=1)]))
        self.assertEqual(subset_count, len(subset))
        self.assertEqual(_estimate_rigid_contact_max_per_world(model, 10000), estimate)
        self.assertEqual(_estimate_rigid_contact_max_per_world(model, 1), 1)
        self.assertEqual(_estimate_rigid_contact_max_per_world(model, 0), 0)

    def test_sparse_group_body_combinations(self):
        """Keep enumeration proportional to output for correlated groups and bodies."""
        size = 4000
        for worlds in (np.full(size, -1), np.arange(size) % 2 - 1):
            for groups in (np.ones(size, dtype=int), np.arange(size) // (size // 2) + 1):
                bodies = groups.copy()
                bodies[-1] = 3
                data = _ShapeContactPairs(
                    bodies, worlds, groups, np.full(size, newton.ShapeFlags.COLLIDE_SHAPES), [], 1
                )
                with mock.patch.object(data, "_store_pairs", wraps=data._store_pairs) as store:
                    pairs = data.build_pairs()
                    candidates = sum(len(call.args[1]) for call in store.call_args_list)
                self.assertEqual(len(pairs), int(data.counts.sum()))
                self.assertLess(candidates, size * 2)
                self.assertTrue((pairs[:, 1] == size - 1).all())

    def test_recording_roundtrip(self):
        """Restore pending and materialized tables through both recording formats."""
        for format_type, materialize, finalized_target in itertools.product(
            ("json", "cbor2"), (False, True), (False, True)
        ):
            with self.subTest(format_type=format_type, materialize=materialize, finalized_target=finalized_target):
                model = _make_builder().finalize(device="cpu")
                expected = model._shape_contact_pair_data.build_pairs()
                if materialize:
                    _ = model.shape_contact_pairs
                with mock.patch.object(
                    _ShapeContactPairs, "build_pairs", side_effect=AssertionError("Enumerated pairs")
                ):
                    encoded = pointer_as_key(model, format_type=format_type)
                    decoded = depointer_as_key(encoded, format_type=format_type)
                    restored = (
                        _make_builder().finalize(device="cpu") if finalized_target else newton.Model(device="cpu")
                    )
                    transfer_to_model(decoded, restored)
                    self.assertEqual(restored.shape_contact_pair_count, len(expected))
                    self.assertEqual(restored._shape_contact_pair_data is None, materialize)
                np.testing.assert_array_equal(restored.shape_contact_pairs.numpy(), expected)


def test_lazy_collision(test, device):
    """Match explicit contacts on dynamic broadphases without constructing the table."""
    builder = _make_builder()
    reference = builder.finalize(device=device)
    pipeline = newton.CollisionPipeline(reference)
    test.assertEqual(reference.shape_contact_pairs.device, wp.get_device(device))
    contacts = pipeline.contacts()
    pipeline.collide(reference.state(), contacts)

    def contact_pairs(contacts):
        count = int(contacts.rigid_contact_count.numpy()[0])
        return {
            tuple(sorted((int(a), int(b))))
            for a, b in zip(
                contacts.rigid_contact_shape0.numpy()[:count],
                contacts.rigid_contact_shape1.numpy()[:count],
                strict=True,
            )
        }

    expected = contact_pairs(contacts)
    test.assertGreater(len(expected), 0)
    for broad_phase, speculative_gap in itertools.product(("sap", "nxn"), (None, 0.1)):
        with test.subTest(broad_phase=broad_phase, speculative_gap=speculative_gap):
            with mock.patch.object(_ShapeContactPairs, "build_pairs", side_effect=AssertionError("Enumerated pairs")):
                model = builder.finalize(device=device)
                pipeline = newton.CollisionPipeline(
                    model, broad_phase=broad_phase, shape_pairs_max=128, speculative_contact_gap_max=speculative_gap
                )
                contacts = pipeline.contacts()
                pipeline.collide(model.state(), contacts, dt=1.0 / 60.0)
                test.assertEqual(contact_pairs(contacts), expected)
                test.assertIsNone(vars(model)["shape_contact_pairs"])


def test_lazy_hydroelastic(test, device):
    """Build hydroelastic buffers from their subset without materializing other pairs."""
    builder = newton.ModelBuilder()
    cfg = newton.ModelBuilder.ShapeConfig(is_hydroelastic=True, sdf_max_resolution=32)
    for i in range(3):
        body = builder.add_body(xform=wp.transform(wp.vec3(0.2 * i, 0.0, 0.0)))
        builder.add_shape_sphere(body, cfg=cfg)
    for _ in range(20):
        builder.add_shape_sphere(builder.add_body())
    model = builder.finalize(device=device)
    data = model._shape_contact_pair_data
    with mock.patch.object(data, "build_pairs", wraps=data.build_pairs) as build:
        pipeline = newton.CollisionPipeline(model, broad_phase="sap", shape_pairs_max=512)
        test.assertIsNotNone(pipeline.hydroelastic_sdf)
        test.assertIsNone(vars(model)["shape_contact_pairs"])
        build.assert_called_once()
        np.testing.assert_array_equal(build.call_args.args[0], np.arange(model.shape_count) < 3)
    with mock.patch.object(data, "build_pairs", side_effect=AssertionError("Enumerated pairs")):
        explicit_pairs = wp.array([[0, 1]], dtype=wp.vec2i, device=device)
        explicit = newton.CollisionPipeline(model, shape_pairs_filtered=explicit_pairs)
        test.assertEqual(explicit.hydroelastic_sdf.max_num_shape_pairs, 1)
        contacts = explicit.contacts()
        explicit.collide(model.state(), contacts)
        count = int(contacts.rigid_contact_count.numpy()[0])
        test.assertGreater(count, 0)
        np.testing.assert_array_equal(contacts.rigid_contact_shape0.numpy()[:count], 0)
        np.testing.assert_array_equal(contacts.rigid_contact_shape1.numpy()[:count], 1)
        regular_pairs = wp.array([[0, 3]], dtype=wp.vec2i, device=device)
        regular = newton.CollisionPipeline(model, shape_pairs_filtered=regular_pairs, speculative_contact_gap_max=0.1)
        test.assertIsNone(regular.hydroelastic_sdf)
        with test.assertRaisesRegex(NotImplementedError, "hydroelastic"):
            newton.CollisionPipeline(model, broad_phase="nxn", shape_pairs_max=512, speculative_contact_gap_max=0.1)


def test_pair_cache_graph_capture(test, device):
    """Reject unsafe first use during capture and replay an initialized cache safely."""
    model = _make_builder().finalize(device=device)
    with wp.ScopedCapture(device=device):
        with test.assertRaisesRegex(RuntimeError, "before CUDA graph capture"):
            _ = model.shape_contact_pairs
    test.assertIsNotNone(model._shape_contact_pair_data)
    test.assertIsNone(vars(model)["shape_contact_pairs"])
    pairs = model.shape_contact_pairs
    expected = pairs.numpy().copy()
    output = wp.empty_like(pairs)
    with wp.ScopedCapture(device=device) as capture:
        wp.copy(output, model.shape_contact_pairs)
    gc.collect()
    for _ in range(3):
        output.zero_()
        wp.capture_launch(capture.graph)
        np.testing.assert_array_equal(output.numpy(), expected)

    empty = newton.ModelBuilder().finalize(device=device)
    with wp.ScopedCapture(device=device):
        test.assertEqual(empty.shape_contact_pairs.shape, (0,))


def test_hydroelastic_subset_grid_order(test, device):
    """Match runtime traversal-grid selection after narrow-phase type sorting."""
    builder = newton.ModelBuilder()
    cfg = newton.ModelBuilder.ShapeConfig(
        is_hydroelastic=True, sdf_max_resolution=64, sdf_narrow_band_range=(-0.01, 0.01), sdf_padding=0.02, gap=0.01
    )
    builder.add_shape_box(builder.add_body(), hx=0.5, hy=0.5, hz=0.5, cfg=cfg)
    body = builder.add_body(xform=wp.transform(wp.vec3(0.4, 0.0, 0.0)))
    builder.add_shape_sphere(body, radius=0.5, cfg=cfg)
    model = builder.finalize(device=device)
    indices = model._shape_sdf_index.numpy()
    sdf_data = model._texture_sdf_data.numpy()
    # Exercise the tie rule independently of primitive SDF padding/resolution
    # rounding: type sorting routes this pair as (sphere, box), retaining box B.
    sdf_data["voxel_radius"] = sdf_data["voxel_radius"].max()
    model._texture_sdf_data.assign(sdf_data)
    expected_active = int(sdf_data[indices[0]]["num_subgrids"])
    tex = model._texture_sdf_coarse_textures[indices[0]]
    expected_tiles = (tex.width - 1) * (tex.height - 1) * (tex.depth - 1)
    for broad_phase in ("sap", "explicit"):
        with test.subTest(broad_phase=broad_phase):
            pairs = wp.array([[0, 1]], dtype=wp.vec2i, device=device) if broad_phase == "explicit" else None
            pipeline = newton.CollisionPipeline(model, broad_phase=broad_phase, shape_pairs_filtered=pairs)
            hydro = pipeline.hydroelastic_sdf
            test.assertEqual(hydro.total_num_tiles, expected_tiles)
            test.assertEqual(hydro.total_num_active_tiles, expected_active)
            pipeline.collide(model.state(), pipeline.contacts())
            test.assertEqual(int(hydro.normalized_shape_pairs.numpy()[0, 1]), 0)
            test.assertIsNone(vars(model)["shape_contact_pairs"])


add_function_test(TestShapeContactPairs, "test_lazy_collision", test_lazy_collision, devices=get_test_devices())
add_function_test(
    TestShapeContactPairs, "test_lazy_hydroelastic", test_lazy_hydroelastic, devices=get_selected_cuda_test_devices()
)
add_function_test(
    TestShapeContactPairs,
    "test_pair_cache_graph_capture",
    test_pair_cache_graph_capture,
    devices=get_selected_cuda_test_devices(),
)
add_function_test(
    TestShapeContactPairs,
    "test_hydroelastic_subset_grid_order",
    test_hydroelastic_subset_grid_order,
    devices=get_selected_cuda_test_devices(),
)


if __name__ == "__main__":
    unittest.main(verbosity=2)
