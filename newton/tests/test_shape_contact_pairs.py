# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import gc
import itertools
import unittest
from unittest import mock

import numpy as np
import warp as wp

import newton
from newton._src.sim.collide import _estimate_rigid_contact_max_per_world
from newton._src.sim.model import _pack_shape_pair_codes, _ShapeCollisionFilterPairs
from newton._src.sim.shape_contact_pairs import (
    _shape_contact_pair_count_for_mask,
    _shape_contact_pair_counts,
    _shape_contact_pairs_for_mask,
    _ShapeContactPairs,
)
from newton._src.solvers.coupled.model_view import ModelView
from newton._src.solvers.kamino._src.core.conversions import compute_required_contact_capacity
from newton._src.solvers.kamino._src.core.model import ModelKamino
from newton._src.solvers.kamino._src.core.shapes import max_contacts_for_shape_pair
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


def _reference_builder_pairs(builder):
    return _reference_pairs(
        builder.shape_body,
        builder.shape_world,
        builder.shape_collision_group,
        builder.shape_flags,
        builder.shape_collision_filter_pairs,
    )


class TestShapeContactPairs(unittest.TestCase):
    def test_finalize_defers_pair_table(self):
        """Keep finalization compact while preserving the exact pair count."""
        builder = newton.ModelBuilder()
        for _ in range(4):
            builder.add_shape_sphere(builder.add_body())
        with mock.patch.object(_ShapeContactPairs, "build_pairs", side_effect=AssertionError("Enumerated pairs")):
            model = builder.finalize(device="cpu")
        self.assertEqual(model.shape_contact_pair_count, 6)
        self.assertIsNone(model._shape_contact_pairs)
        flags = model.shape_flags.numpy()
        flags[0] = 0
        model.shape_flags.assign(flags)
        expected = np.array(list(itertools.combinations(range(4), 2)), dtype=np.int32)
        np.testing.assert_array_equal(_sorted_pairs(model.shape_contact_pairs.numpy()), expected)
        self.assertEqual(model.shape_contact_pair_count, 6)
        np.testing.assert_array_equal(_shape_contact_pair_counts(model, shape_mask=np.ones(4, dtype=bool)), [6, 0])
        view = ModelView(model, "normalized")
        view.shape_world = wp.zeros(4, dtype=wp.int32, device="cpu")
        np.testing.assert_array_equal(_shape_contact_pair_counts(view), [6, 0])

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
        for template in range(12):
            for _ in range(6):
                builder.begin_world()
                first = builder.shape_count
                for i in range(5 + template % 2):
                    body = global_body if i == 1 and template % 3 == 0 else builder.add_body()
                    builder.add_shape_sphere(body)
                # Global index zero and local offset zero must remain distinct in
                # template keys, even though both normalize to zero without tagging.
                builder.add_shape_collision_filter_pair(0 if template % 4 == 0 else first, first + 2)
                builder.end_world()
        builder.add_ground_plane()
        expected = _reference_builder_pairs(builder)
        model = builder.finalize(device="cpu")
        # Repeated runs exceed a replay chunk and leave a partial final chunk.
        with mock.patch("newton._src.sim.shape_contact_pairs._PAIR_CHUNK_SIZE", 61):
            np.testing.assert_array_equal(_sorted_pairs(model.shape_contact_pairs.numpy()), expected)

    def test_uniform_group_counts(self):
        """Match the vectorized histogram path with globals and repeated bodies."""
        rng = np.random.default_rng(4395)
        for case in range(8):
            with self.subTest(case=case):
                size = 300
                bodies = rng.choice([-3, -1, 0, 1, 2, 2147483647], size=size)
                worlds = rng.integers(-1, 4, size=size)
                groups = np.full(size, 2147483647)
                flags = np.full(size, newton.ShapeFlags.COLLIDE_SHAPES)
                filters = rng.integers(0, size, size=(400, 2))
                packed = np.unique(_pack_shape_pair_codes(filters[:, 0], filters[:, 1]))
                data = _ShapeContactPairs(bodies, worlds, groups, flags, packed, 4)
                expected = _reference_pairs(bodies, worlds, groups, flags, filters)
                np.testing.assert_array_equal(
                    data.counts, np.bincount(np.max(worlds[expected], axis=1) + 1, minlength=5)
                )
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
        local_template = np.array([[0, 2], [0, 3], [1, 2]], dtype=np.int32)
        offsets = np.arange(world_count, dtype=np.int32) * 4
        local_pairs = local_template[None, :, :] + offsets[:, None, None]
        global_pairs = np.column_stack(
            (np.arange(world_count * 4, dtype=np.int32), np.full(world_count * 4, world_count * 4, dtype=np.int32))
        )
        expected = np.concatenate((local_pairs.reshape((-1, 2)), global_pairs))
        np.testing.assert_array_equal(_sorted_pairs(pairs), _sorted_pairs(expected))

    def test_explicit_override_defers_default_table(self):
        """Use caller-supplied explicit pairs without constructing the default table."""
        model = _make_builder().finalize(device="cpu")
        pairs = wp.array([[0, 1]], dtype=wp.vec2i, device="cpu")
        with mock.patch.object(_ShapeContactPairs, "build_pairs", side_effect=AssertionError("Enumerated pairs")):
            pipeline = newton.CollisionPipeline(model, shape_pairs_filtered=pairs)
        self.assertIs(pipeline.shape_pairs_filtered, pairs)
        self.assertIsNone(model._shape_contact_pairs)

    def test_pair_cache_lifecycle(self):
        """Cache lazy summaries, honor supplied prefixes and disable removed tables."""
        builder = newton.ModelBuilder()
        builder.add_shape_sphere(builder.add_body())
        for size in (2, 1):
            builder.begin_world()
            for _ in range(size):
                builder.add_shape_sphere(builder.add_body())
            builder.end_world()
        model = builder.finalize(device="cpu")
        mask = np.array([True, True, True, False])
        mask.setflags(write=False)
        with mock.patch.object(_ShapeContactPairs, "build_pairs", side_effect=AssertionError("Enumerated pairs")):
            counts = _shape_contact_pair_counts(model)
            masked_counts = _shape_contact_pair_counts(model, shape_mask=mask)
            np.testing.assert_array_equal(counts, [0, 3, 1])
            np.testing.assert_array_equal(masked_counts, [0, 3, 0])
            self.assertIs(_shape_contact_pair_counts(model, shape_mask=mask.copy()), masked_counts)
            for summary in (counts, masked_counts):
                with self.assertRaises(ValueError):
                    summary[0] = 1
        self.assertIsNone(model._shape_contact_pairs)
        model.shape_contact_pair_count = 0
        with mock.patch.object(
            _ShapeContactPairs, "build_pairs", autospec=True, side_effect=_ShapeContactPairs.build_pairs
        ) as build:
            pairs = model.shape_contact_pairs
            self.assertIs(model.shape_contact_pairs, pairs)
            build.assert_called_once()
        self.assertEqual(len(pairs), 4)
        self.assertEqual(model.shape_contact_pair_count, 0)
        np.testing.assert_array_equal(_shape_contact_pair_counts(model), [0, 0, 0])
        model.shape_contact_pair_count = len(pairs)
        self.assertIs(_shape_contact_pair_counts(model), counts)
        with mock.patch.object(pairs, "numpy", side_effect=AssertionError("Scanned default pairs")):
            masked_counts = _shape_contact_pair_counts(model, shape_mask=np.ones(model.shape_count, dtype=bool))
        np.testing.assert_array_equal(masked_counts, counts)
        pairs = wp.array([[0, 3], [1, 2]], dtype=wp.vec2i, device="cpu")
        model.shape_contact_pairs = pairs
        self.assertIs(model.shape_contact_pairs, pairs)
        self.assertEqual(model.shape_contact_pair_count, len(pairs))
        for count, expected in ((2, [0, 1, 1]), (1, [0, 0, 1]), (2, [0, 1, 1])):
            model.shape_contact_pair_count = count
            np.testing.assert_array_equal(_shape_contact_pair_counts(model), expected)
            self.assertEqual(int(_shape_contact_pair_counts(model, shape_mask=mask).sum()), count - 1)
        model.shape_contact_pairs = None
        self.assertIsNone(model.shape_contact_pairs)
        self.assertEqual(model.shape_contact_pair_count, 0)
        for shape_mask in (None, mask):
            np.testing.assert_array_equal(_shape_contact_pair_counts(model, shape_mask=shape_mask), [0, 0, 0])
        with self.assertRaisesRegex(ValueError, "shape_pairs_filtered must be provided"):
            newton.CollisionPipeline(model)

    def test_view_and_supplied_pair_count_sources(self):
        """Count view-local prefixes and supplied tables without generating base pairs."""
        model = _make_builder().finalize(device="cpu")
        mask = np.zeros(model.shape_count, dtype=bool)
        mask[[1, 2]] = True
        pairs = wp.array([[1, 2], [2, 1], [1, 2], [0, 1]], dtype=wp.vec2i, device="cpu")
        with (
            mock.patch.object(_ShapeContactPairs, "build_pairs", side_effect=AssertionError("Enumerated pairs")),
            mock.patch(
                "newton._src.sim.shape_contact_pairs._shape_contact_pairs_for_mask",
                side_effect=AssertionError("Materialized subset pairs"),
            ),
        ):
            view = ModelView(model, "pairs")
            invalid = [
                ("numpy", np.array([[1, 2]], dtype=np.int32)),
                ("dtype", wp.zeros(1, dtype=wp.int32, device="cpu")),
                ("dimension", wp.array(np.zeros((1, 1, 2), dtype=np.int32), dtype=wp.vec2i, device="cpu")),
            ]
            invalid.extend(
                (str(device), wp.array([[1, 2]], dtype=wp.vec2i, device=device))
                for device in get_selected_cuda_test_devices()
            )
            for case, value in invalid:
                with self.subTest(case=case), self.assertRaises(TypeError):
                    view.shape_contact_pairs = value
            view.shape_contact_pairs = pairs
            view.shape_contact_pair_count = 2
            self.assertEqual(_shape_contact_pair_count_for_mask(view, mask), 2)
            nested = ModelView(view, "nested")
            self.assertEqual(_shape_contact_pair_count_for_mask(nested, mask), 2)
            self.assertEqual(_shape_contact_pair_count_for_mask(model, mask, shape_pairs=pairs), 3)
            view.shape_contact_pair_count = 1
            self.assertEqual(_shape_contact_pair_count_for_mask(nested, mask), 1)
            view.shape_contact_pairs = None
            self.assertEqual(int(_shape_contact_pair_counts(view).sum()), 0)
            world_mask = np.zeros(model.shape_count, dtype=bool)
            world_mask[[1, 3]] = True
            np.testing.assert_array_equal(_shape_contact_pair_counts(model, shape_mask=world_mask), [0, 1, 0])
            world_view = ModelView(model, "worlds")
            world_view.shape_world = wp.full(model.shape_count, 1, dtype=wp.int32, device="cpu")
            world_view.shape_contact_pairs = wp.array([[1, 3]], dtype=wp.vec2i, device="cpu")
            world_view.shape_contact_pair_count = 1
            np.testing.assert_array_equal(_shape_contact_pair_counts(world_view, shape_mask=world_mask), [0, 0, 1])
        self.assertIsNone(model._shape_contact_pairs)

    def test_view_inherits_pair_topology(self):
        """Keep inherited pairs, totals and summaries consistent across view overrides."""
        model = _make_builder().finalize(device="cpu")
        counts = _shape_contact_pair_counts(model)
        capacity = compute_required_contact_capacity(model)
        views = []
        with mock.patch.object(_ShapeContactPairs, "build_pairs", side_effect=AssertionError("Enumerated pairs")):
            for name, value in (
                ("shape_count", 2),
                ("shape_body", wp.full(model.shape_count, -1, dtype=wp.int32, device="cpu")),
                ("shape_world", wp.full(model.shape_count, 1, dtype=wp.int32, device="cpu")),
                ("shape_collision_group", wp.zeros(model.shape_count, dtype=wp.int32, device="cpu")),
                ("shape_flags", wp.zeros(model.shape_count, dtype=wp.int32, device="cpu")),
                ("shape_collision_filter_pairs", _ShapeCollisionFilterPairs(np.empty(0, dtype=np.int64))),
            ):
                view = ModelView(model, name)
                setattr(view, name, value)
                for candidate in (view, ModelView(view, "nested")):
                    with self.subTest(override=name, view=candidate.name):
                        np.testing.assert_array_equal(_shape_contact_pair_counts(candidate), counts)
                        self.assertEqual(candidate.shape_contact_pair_count, int(counts.sum()))
                        if name != "shape_count":
                            self.assertEqual(compute_required_contact_capacity(candidate), capacity)
                        views.append(candidate)
        self.assertIsNone(model._shape_contact_pairs)
        for view in views:
            self.assertIs(view.shape_contact_pairs, model.shape_contact_pairs)
            if view.shape_count == model.shape_count:
                self.assertEqual(compute_required_contact_capacity(view, include_shape_contact_pairs=True), capacity)

    def test_supplied_pair_array_updates(self):
        """Refresh per-world and masked capacity estimates after supplied pairs change in place."""
        builder = newton.ModelBuilder()
        for shape_count in (3, 2):
            builder.begin_world()
            for _ in range(shape_count):
                builder.add_shape_sphere(builder.add_body())
            builder.end_world()
        initial = [[0, 1], [3, 4], [0, 2]]
        updated = [[0, 1], [0, 2], [3, 4]]
        mask = np.array([True, True, True, False, False])
        for device in get_test_devices():
            model = builder.finalize(device=device)
            pairs = wp.array(initial, dtype=wp.vec2i, device=device)
            model.shape_contact_pairs = pairs
            model.shape_contact_pair_count = 2
            for rows, counts, masked_counts, estimate in (
                (initial, [0, 1, 1], [0, 1, 0], 5),
                (updated, [0, 2, 0], [0, 2, 0], 10),
                (initial, [0, 1, 1], [0, 1, 0], 5),
            ):
                with self.subTest(device=device, pairs=rows):
                    pairs.assign(np.asarray(rows, dtype=np.int32))
                    np.testing.assert_array_equal(_shape_contact_pair_counts(model), counts)
                    np.testing.assert_array_equal(_shape_contact_pair_counts(model, shape_mask=mask), masked_counts)
                    self.assertEqual(_estimate_rigid_contact_max_per_world(model, 100), estimate)
                    self.assertEqual(compute_required_contact_capacity(model), (2, counts[1:]))
                    self.assertEqual(model.shape_contact_pair_count, 2)

    def test_empty_models_and_filtered_scenes(self):
        """Return empty tables for empty and fully excluded scenes."""
        empty = newton.ModelBuilder().finalize(device="cpu")
        self.assertEqual(empty.shape_contact_pair_count, 0)
        self.assertEqual(empty.shape_contact_pairs.shape, (0,))
        size = 20000
        for scene, bodies, groups in (
            ("same_body", np.zeros(size), np.ones(size)),
            ("same_negative_group", np.arange(size), -np.ones(size)),
            ("distinct_positive_groups", np.arange(size), np.arange(1, size + 1)),
            ("disabled_group", np.arange(size), np.zeros(size)),
        ):
            with self.subTest(scene=scene):
                data = _ShapeContactPairs(
                    bodies, -np.ones(size), groups, np.full(size, newton.ShapeFlags.COLLIDE_SHAPES), [], 0
                )
                self.assertEqual(int(data.counts.sum()), 0)
                self.assertEqual(data.build_pairs().shape, (0, 2))

    def test_subset_and_per_world_counts(self):
        """Match subset and weighted per-world counts without building all pairs."""
        builder = _make_builder()
        expected = _reference_builder_pairs(builder)
        worlds = np.asarray(builder.shape_world)
        model = builder.finalize(device="cpu")
        mask = np.arange(model.shape_count) % 3 != 0
        types = model.shape_type.numpy()
        types[::3] = int(newton.GeoType.MESH)
        model.shape_type.assign(types)
        is_mesh = np.isin(types, (newton.GeoType.MESH, newton.GeoType.HFIELD))
        weighted_counts = np.zeros(model.world_count + 1, dtype=int)
        for a, b in expected:
            weighted_counts[max(worlds[a], worlds[b]) + 1] += 40 if is_mesh[a] or is_mesh[b] else 5
        expected_estimate = int(weighted_counts[0] + weighted_counts[1:].max())
        for shape_mask in (None, ~is_mesh, mask):
            _shape_contact_pair_counts(model, shape_mask=shape_mask)
        with (
            mock.patch.object(_ShapeContactPairs, "build_pairs", side_effect=AssertionError("Enumerated pairs")),
            mock.patch.object(_ShapeContactPairs, "count_pairs", side_effect=AssertionError("Recounted cached pairs")),
        ):
            for cap in (0, 1, 10000):
                with self.subTest(cap=cap):
                    self.assertEqual(_estimate_rigid_contact_max_per_world(model, cap), min(cap, expected_estimate))
            subset_count = _shape_contact_pair_count_for_mask(model, mask)
        subset = _shape_contact_pairs_for_mask(model, mask)
        self.assertIsNone(model._shape_contact_pairs)
        np.testing.assert_array_equal(_sorted_pairs(subset), expected[mask[expected].all(axis=1)])
        self.assertEqual(subset_count, len(subset))
        _ = model.shape_contact_pairs
        self.assertEqual(_estimate_rigid_contact_max_per_world(model, 10000), expected_estimate)

    def test_kamino_single_world_normalization(self):
        """Match lazy and exact conversion capacities when globals join the only world."""
        for device, global_count in itertools.product(get_test_devices(), (0, 2, 4)):
            with self.subTest(device=device, global_count=global_count):
                builder = newton.ModelBuilder()
                shapes = (
                    builder.add_shape_box,
                    builder.add_shape_sphere,
                    builder.add_shape_capsule,
                    builder.add_shape_box,
                )
                for add_shape in shapes[:global_count]:
                    add_shape(builder.add_body())
                builder.begin_world()
                for add_shape in shapes[global_count:]:
                    add_shape(builder.add_body())
                builder.end_world()
                model = builder.finalize(device=device)
                types = model.shape_type.numpy()
                expected = sum(
                    sum(max_contacts_for_shape_pair(int(types[a]), int(types[b])))
                    for a, b in itertools.combinations(range(4), 2)
                )
                with mock.patch.object(
                    _ShapeContactPairs, "build_pairs", side_effect=AssertionError("Enumerated pairs")
                ):
                    lazy = ModelKamino.from_newton(model, include_shape_contact_pairs=False)
                self.assertEqual(lazy.geoms.model_minimum_contacts, expected)
                self.assertEqual(lazy.geoms.world_minimum_contacts, [expected])
                self.assertIsNone(model._shape_contact_pairs)
                exact = ModelKamino.from_newton(model)
                self.assertEqual(exact.geoms.model_minimum_contacts, expected)
                self.assertEqual(exact.geoms.world_minimum_contacts, [expected])

    def test_kamino_default_pair_prefix(self):
        """Honor active prefixes of default pairs before their first materialization."""
        builder = newton.ModelBuilder()
        builder.begin_world()
        for _ in range(4):
            builder.add_shape_sphere(builder.add_body())
        builder.add_shape_collision_filter_pair(2, 3)
        builder.end_world()
        builder.begin_world()
        for _ in range(2):
            builder.add_shape_box(builder.add_body())
        builder.end_world()
        for device, scope in itertools.product(get_test_devices(), ("model", "view", "nested")):
            with self.subTest(device=device, scope=scope):
                model = builder.finalize(device=device)
                view = ModelView(model, "prefix")
                candidate = {"model": model, "view": view, "nested": ModelView(view, "nested")}[scope]
                self.assertEqual(compute_required_contact_capacity(candidate), (13, [5, 8]))
                np.testing.assert_array_equal(_shape_contact_pair_counts(candidate), [0, 5, 1])
                self.assertIsNone(model._shape_contact_pairs)
                candidate.shape_contact_pair_count = 0
                with mock.patch.object(
                    _ShapeContactPairs, "build_pairs", side_effect=AssertionError("Enumerated pairs")
                ):
                    self.assertEqual(compute_required_contact_capacity(candidate), (0, [0, 0]))
                    np.testing.assert_array_equal(_shape_contact_pair_counts(candidate), [0, 0, 0])
                self.assertIsNone(model._shape_contact_pairs)
                candidate.shape_contact_pair_count = 1
                self.assertEqual(compute_required_contact_capacity(candidate), (1, [1, 0]))
                np.testing.assert_array_equal(_shape_contact_pair_counts(candidate), [0, 1, 0])
                mask = np.ones(model.shape_count, dtype=bool)
                np.testing.assert_array_equal(_shape_contact_pairs_for_mask(candidate, mask), [[0, 1]])
                for pair_count, pair_cap, world_cap, capacity in (
                    (2, None, None, (2, [2, 0])),
                    (5, None, None, (5, [5, 0])),
                    (6, None, None, (13, [5, 8])),
                    (6, 1, None, (6, [5, 1])),
                    (6, None, 3, (6, [3, 3])),
                ):
                    candidate.shape_contact_pair_count = pair_count
                    self.assertEqual(compute_required_contact_capacity(candidate, pair_cap, world_cap), capacity)

    def test_kamino_weighted_pair_counts(self):
        """Size contacts from pair weights, preserving caps, globals and explicit prefixes."""
        builder = _make_builder()
        expected = _reference_builder_pairs(builder)
        global_builder = newton.ModelBuilder()
        for _ in range(3):
            global_builder.add_shape_box(global_builder.add_body())
        global_pairs = np.array([[0, 1], [0, 2], [1, 2]], dtype=np.int32)
        for device in get_test_devices():
            model = builder.finalize(device=device)
            model.shape_type.assign(
                np.resize([newton.GeoType.SPHERE, newton.GeoType.BOX, newton.GeoType.CAPSULE], model.shape_count)
            )
            supplied = ModelView(model, "supplied")
            supplied.shape_contact_pairs = wp.array([[1, 2], [2, 1], [1, 2], [0, 6]], dtype=wp.vec2i, device=device)
            supplied.shape_contact_pair_count = 3
            typed = ModelView(model, "types")
            typed.shape_type = wp.array(np.roll(model.shape_type.numpy(), 1), dtype=wp.int32, device=device)
            attached = ModelView(model, "attached")
            bodies = model.shape_body.numpy()
            bodies[2] = bodies[3]
            attached.shape_body = wp.array(bodies, dtype=wp.int32, device=device)
            attached_pairs = expected[bodies[expected[:, 0]] != bodies[expected[:, 1]]]
            attached.shape_contact_pairs = wp.array(attached_pairs, dtype=wp.vec2i, device=device)
            attached.shape_contact_pair_count = len(attached_pairs)
            global_model = global_builder.finalize(device=device)
            normalized = ModelView(global_model, "single_world")
            normalized.shape_world = wp.zeros(global_model.shape_count, dtype=wp.int32, device=device)
            cases = []
            for source, candidate, pairs in (
                ("multiworld", model, expected),
                ("supplied_prefix", supplied, supplied.shape_contact_pairs.numpy()[:3]),
                ("type_override", typed, expected),
                ("body_override", attached, attached_pairs),
                ("global_only", global_model, global_pairs),
                ("normalized_globals", normalized, global_pairs),
            ):
                worlds, types = candidate.shape_world.numpy(), candidate.shape_type.numpy()
                for pair_cap, world_cap in ((None, None), (-1, None), (0, None), (1, None), (5, 7), (None, 0)):
                    with self.subTest(device=device, source=source, pair_cap=pair_cap, world_cap=world_cap):
                        required = np.zeros(candidate.world_count, dtype=int)
                        for a, b in pairs:
                            world = max(worlds[a], worlds[b])
                            if world < 0:
                                continue
                            weight = sum(max_contacts_for_shape_pair(int(types[a]), int(types[b])))
                            if pair_cap is not None and pair_cap >= 0:
                                weight = min(weight, pair_cap)
                            required[world] += weight
                        if world_cap is not None:
                            required = np.minimum(required, world_cap)
                        capacity = (int(required.sum()), required.tolist())
                        cases.append((source, candidate, pair_cap, world_cap, capacity))
                        with (
                            mock.patch.object(
                                _ShapeContactPairs, "build_pairs", side_effect=AssertionError("Enumerated pairs")
                            ),
                            mock.patch(
                                "newton._src.sim.shape_contact_pairs._shape_contact_pair_counts",
                                side_effect=AssertionError("Repeated masked counting"),
                            ),
                        ):
                            self.assertEqual(
                                compute_required_contact_capacity(candidate, pair_cap, world_cap), capacity
                            )
            self.assertIsNone(model._shape_contact_pairs)
            self.assertIsNone(global_model._shape_contact_pairs)
            with mock.patch.object(_ShapeContactPairs, "count_pairs", side_effect=AssertionError("Repeated counting")):
                for source, candidate, pair_cap, world_cap, capacity in cases:
                    with self.subTest(
                        device=device, source=source, pair_cap=pair_cap, world_cap=world_cap, explicit=True
                    ):
                        self.assertEqual(
                            compute_required_contact_capacity(
                                candidate, pair_cap, world_cap, include_shape_contact_pairs=True
                            ),
                            capacity,
                        )
            model.shape_contact_pair_count = 0
            with mock.patch.object(
                _ShapeContactPairs, "count_pairs", side_effect=AssertionError("Counted disabled pairs")
            ):
                self.assertEqual(compute_required_contact_capacity(model), (0, [0] * model.world_count))

    def test_sparse_group_body_combinations(self):
        """Skip dense same-body blocks for correlated collision groups."""
        size = 4000
        original_nonzero, original_repeat = np.nonzero, np.repeat
        for world_layout, worlds in (("global", np.full(size, -1)), ("global_and_local", np.arange(size) % 2 - 1)):
            for group_layout, groups in (
                ("uniform", np.ones(size, dtype=int)),
                ("body_partitioned", np.arange(size) // (size // 2) + 1),
            ):
                with self.subTest(world_layout=world_layout, group_layout=group_layout):
                    bodies = groups.copy()
                    bodies[-1] = 3
                    data = _ShapeContactPairs(
                        bodies, worlds, groups, np.full(size, newton.ShapeFlags.COLLIDE_SHAPES), [], 1
                    )
                    candidates = 0

                    def count_dense_candidates(predicate):
                        nonlocal candidates
                        if predicate.ndim == 2:
                            candidates += predicate.size
                        return original_nonzero(predicate)

                    def count_range_candidates(values, repeats, axis=None):
                        nonlocal candidates
                        rows = original_repeat(values, repeats, axis=axis)
                        candidates += rows.size
                        return rows

                    # Count predicate cells and ragged rows before same-body rejection.
                    with (
                        mock.patch("newton._src.sim.shape_contact_pairs.np.nonzero", new=count_dense_candidates),
                        mock.patch("newton._src.sim.shape_contact_pairs.np.repeat", new=count_range_candidates),
                    ):
                        pairs = data.build_pairs()
                    first = np.flatnonzero(groups[:-1] == groups[-1])
                    expected = np.column_stack((first, np.full(len(first), size - 1)))
                    np.testing.assert_array_equal(_sorted_pairs(pairs), expected)
                    self.assertEqual(int(data.counts.sum()), len(expected))
                    self.assertLess(candidates, size * 2)

    def test_recording_roundtrip(self):
        """Restore pending and materialized tables through both recording formats."""
        for format_type, materialize in itertools.product(("json", "cbor2"), (False, True)):
            with self.subTest(format_type=format_type, materialize=materialize):
                builder = _make_builder()
                expected = _reference_builder_pairs(builder)
                model = builder.finalize(device="cpu")
                if materialize:
                    _ = model.shape_contact_pairs
                with mock.patch.object(
                    _ShapeContactPairs, "build_pairs", side_effect=AssertionError("Enumerated pairs")
                ):
                    encoded = pointer_as_key(model, format_type=format_type)
                    decoded = depointer_as_key(encoded, format_type=format_type)
                    restored = _make_builder().finalize(device="cpu")
                    transfer_to_model(decoded, restored)
                    self.assertEqual(restored.shape_contact_pair_count, len(expected))
                    self.assertEqual(restored._shape_contact_pairs is not None, materialize)
                np.testing.assert_array_equal(_sorted_pairs(restored.shape_contact_pairs.numpy()), expected)


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
                test.assertIsNone(model._shape_contact_pairs)


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
    with mock.patch.object(
        _ShapeContactPairs, "build_pairs", autospec=True, side_effect=_ShapeContactPairs.build_pairs
    ) as build:
        pipeline = newton.CollisionPipeline(model, broad_phase="sap", shape_pairs_max=512)
        test.assertIsNotNone(pipeline.hydroelastic_sdf)
        test.assertIsNone(model._shape_contact_pairs)
        build.assert_called_once()
        np.testing.assert_array_equal(build.call_args.args[1], np.arange(model.shape_count) < 3)
    with mock.patch.object(_ShapeContactPairs, "build_pairs", side_effect=AssertionError("Enumerated pairs")):
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
    test.assertIsNone(model._shape_contact_pairs)
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
if __name__ == "__main__":
    unittest.main(verbosity=2)
