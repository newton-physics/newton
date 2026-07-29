# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import itertools
import unittest

import numpy as np

import newton
from newton._src.solvers.mujoco.collision_masks import (
    compile_collision_masks,
    compile_newton_collision_graph,
    verify_collision_mask_compilation,
    verify_newton_collision_graph_compilation,
)


class TestCollisionMaskCompiler(unittest.TestCase):
    @staticmethod
    def group_pair_allowed(group_a, group_b):
        """Evaluate Newton's signed collision-group predicate."""
        if group_a == 0 or group_b == 0:
            return False
        if group_a > 0:
            return group_a == group_b or group_b < 0
        return group_a != group_b

    def assert_compiles_exactly(self, collision_type, collision_affinity, **kwargs):
        """Compile masks and verify every resulting shape pair."""
        result = compile_collision_masks(collision_type, collision_affinity, **kwargs)
        verify_collision_mask_compilation(
            collision_type,
            collision_affinity,
            result,
            prefiltered_pairs=kwargs.get("prefiltered_pairs", ()),
        )
        return result

    def test_default_and_disabled_masks(self):
        """Map default colliders together and disable zero masks."""
        result = self.assert_compiles_exactly([0, 0, 1, 1], [0, 0, 1, 1])

        self.assertTrue(np.array_equal(result.groups[:2], np.zeros(2, dtype=np.int32)))
        self.assertEqual(result.groups[2], result.groups[3])
        self.assertGreater(result.groups[2], 0)
        self.assertEqual(result.excluded_pairs.shape, (0, 2))

    def test_cassie_mask_classes(self):
        """Represent Cassie's asymmetric left-right masks exactly."""
        collision_type = [0] + [1] * 5 + [2] * 7 + [4] * 7
        collision_affinity = [15] + [0] * 5 + [4] * 7 + [2] * 7
        result = self.assert_compiles_exactly(collision_type, collision_affinity)

        self.assertTrue(result.optimal)
        self.assertEqual(result.group_count, 3)
        self.assertEqual(result.excluded_pairs.shape[0], 35)

    def test_softfoot_mask_classes(self):
        """Collapse SoftFoot's dense incompatible class into one negative group."""
        collision_type = [1] * 51
        collision_affinity = [2] * 49 + [1, 15]
        result = self.assert_compiles_exactly(collision_type, collision_affinity)

        self.assertTrue(result.optimal)
        self.assertEqual(result.group_count, 2)
        self.assertEqual(result.excluded_pairs.shape[0], 0)
        self.assertLess(result.groups[0], 0)
        self.assertGreater(result.groups[-1], 0)

    def test_talos_mask_classes(self):
        """Use a negative group for Talos's mutually incompatible feet."""
        collision_type = [1] * 50 + [6, 6]
        collision_affinity = [1] * 52
        result = self.assert_compiles_exactly(collision_type, collision_affinity)

        self.assertTrue(result.optimal)
        self.assertEqual(result.group_count, 2)
        self.assertEqual(result.excluded_pairs.shape[0], 0)
        self.assertEqual(result.groups[-1], result.groups[-2])
        self.assertLess(result.groups[-1], 0)

    def test_model_builder_contact_pairs_match_masks(self):
        """Generate exactly the MuJoCo-compatible pairs in ModelBuilder."""
        collision_type = np.asarray([0] + [1] * 5 + [2] * 7 + [4] * 7, dtype=np.uint32)
        collision_affinity = np.asarray([15] + [0] * 5 + [4] * 7 + [2] * 7, dtype=np.uint32)
        result = self.assert_compiles_exactly(collision_type, collision_affinity)

        builder = newton.ModelBuilder()
        for group in result.groups:
            cfg = newton.ModelBuilder.ShapeConfig(collision_group=int(group), density=0.0)
            builder.add_shape_sphere(-1, radius=0.1, cfg=cfg)
        builder.shape_collision_filter_pairs = result.excluded_pairs.tolist()
        model = builder.finalize(device="cpu")

        actual = set(map(tuple, model.shape_contact_pairs.numpy().tolist()))
        expected = set()
        for shape_a in range(collision_type.shape[0] - 1):
            for shape_b in range(shape_a + 1, collision_type.shape[0]):
                if (collision_type[shape_a] & collision_affinity[shape_b]) or (
                    collision_type[shape_b] & collision_affinity[shape_a]
                ):
                    expected.add((shape_a, shape_b))
        self.assertEqual(actual, expected)

    def test_prefiltered_pairs_reduce_generated_exclusions(self):
        """Avoid regenerating exclusions already handled by structural filtering."""
        collision_type = [0] + [1] * 2 + [2] * 2 + [4] * 2
        collision_affinity = [7] + [0] * 2 + [4] * 2 + [2] * 2
        unfiltered = self.assert_compiles_exactly(collision_type, collision_affinity)
        prefiltered_pairs = [(shape_a, shape_b) for shape_a in (1, 2) for shape_b in (5, 6)]
        filtered = self.assert_compiles_exactly(
            collision_type,
            collision_affinity,
            prefiltered_pairs=prefiltered_pairs,
        )

        self.assertEqual(unfiltered.excluded_pairs.shape[0], 3)
        self.assertEqual(filtered.excluded_pairs.shape[0], 0)

    def test_random_masks(self):
        """Preserve the MuJoCo predicate across random 32-bit masks."""
        rng = np.random.default_rng(1234)
        for shape_count in (1, 2, 5, 20, 100):
            collision_type = rng.integers(0, 16, shape_count, dtype=np.uint32)
            collision_affinity = rng.integers(0, 16, shape_count, dtype=np.uint32)
            self.assert_compiles_exactly(
                collision_type,
                collision_affinity,
                max_search_nodes=50_000,
            )

    def test_many_mask_classes_use_fast_fallback(self):
        """Skip exponential search while preserving exactness for many classes."""
        collision_type = np.arange(1, 101, dtype=np.uint32)
        collision_affinity = np.arange(101, 201, dtype=np.uint32)
        result = self.assert_compiles_exactly(collision_type, collision_affinity)

        self.assertFalse(result.optimal)
        self.assertEqual(result.search_nodes, 0)

    def test_import_constraints_keep_shapes_in_default_positive_group(self):
        """Keep isolated shapes active and limit imported masks to one positive group."""
        collision_type = [1, 1, 2, 2]
        collision_affinity = [1, 1, 2, 2]
        result = self.assert_compiles_exactly(
            collision_type,
            collision_affinity,
            force_nonzero=True,
            max_positive_groups=1,
        )

        self.assertEqual(set(result.groups[result.groups > 0]), {1})
        self.assertTrue(np.all(result.groups != 0))
        self.assertEqual(result.excluded_pairs.shape[0], 4)

        isolated = self.assert_compiles_exactly(
            [1],
            [0],
            force_nonzero=True,
            max_positive_groups=1,
        )
        self.assertNotEqual(isolated.groups[0], 0)

    def test_small_class_search_matches_brute_force(self):
        """Find the globally best class-homogeneous plan on small random graphs."""
        rng = np.random.default_rng(9921)
        for _ in range(10):
            while True:
                class_masks = np.column_stack(
                    (
                        rng.integers(0, 8, 4, dtype=np.uint32),
                        rng.integers(0, 8, 4, dtype=np.uint32),
                    )
                )
                if np.unique(class_masks, axis=0).shape[0] == 4:
                    break
            counts = rng.integers(1, 4, 4)
            class_index = np.repeat(np.arange(4), counts)
            collision_type = class_masks[class_index, 0]
            collision_affinity = class_masks[class_index, 1]
            result = self.assert_compiles_exactly(collision_type, collision_affinity)

            labels = (*range(-4, 0), 0, *range(1, 5))
            best = None
            for assignment in itertools.product(labels, repeat=4):
                cost = 0
                feasible = True
                for shape_a in range(class_index.shape[0] - 1):
                    for shape_b in range(shape_a + 1, class_index.shape[0]):
                        expected = bool(
                            (collision_type[shape_a] & collision_affinity[shape_b])
                            or (collision_type[shape_b] & collision_affinity[shape_a])
                        )
                        actual = self.group_pair_allowed(
                            assignment[class_index[shape_a]],
                            assignment[class_index[shape_b]],
                        )
                        if expected and not actual:
                            feasible = False
                            break
                        cost += actual and not expected
                    if not feasible:
                        break
                if feasible:
                    key = (cost, len({group for group in assignment if group != 0}))
                    best = key if best is None else min(best, key)

            self.assertEqual((result.excluded_pairs.shape[0], result.group_count), best)

    def test_empty_masks(self):
        """Compile an empty shape collection."""
        result = self.assert_compiles_exactly([], [])

        self.assertEqual(result.groups.shape, (0,))
        self.assertEqual(result.excluded_pairs.shape, (0, 2))

    def test_full_uint32_masks(self):
        """Preserve high collision-mask bits and signed representations."""
        collision_type = [-2147483648, 0x80000000, -1, 0]
        collision_affinity = [0x80000000, -2147483648, 1, 0]
        self.assert_compiles_exactly(collision_type, collision_affinity)

    def test_bounded_search_remains_exact(self):
        """Return an exact fallback when optimization exhausts its node budget."""
        collision_type = np.arange(1, 17, dtype=np.uint32)
        collision_affinity = np.arange(16, 0, -1, dtype=np.uint32)
        result = self.assert_compiles_exactly(
            collision_type,
            collision_affinity,
            max_search_nodes=1,
        )

        self.assertFalse(result.optimal)

    def test_rejects_invalid_inputs(self):
        """Reject mismatched masks, invalid pairs, and invalid search budgets."""
        with self.assertRaises(ValueError):
            compile_collision_masks([1], [1, 2])
        with self.assertRaises(TypeError):
            compile_collision_masks([1.0], [1])
        with self.assertRaises(ValueError):
            compile_collision_masks([1], [1], prefiltered_pairs=[(0, 1)])
        with self.assertRaises(ValueError):
            compile_collision_masks([1], [1], max_search_nodes=0)
        with self.assertRaises(ValueError):
            compile_collision_masks([1], [1], max_exact_classes=0)
        with self.assertRaises(ValueError):
            compile_collision_masks([1], [1], max_positive_groups=0)


class TestNewtonCollisionGraphCompiler(unittest.TestCase):
    def assert_compiles_exactly(self, collision_groups, **kwargs):
        """Compile a Newton collision graph and verify every resulting pair."""
        result = compile_newton_collision_graph(collision_groups, **kwargs)
        self.assertTrue(result.exact)
        verify_newton_collision_graph_compilation(
            collision_groups,
            result,
            excluded_pairs=kwargs.get("excluded_pairs", ()),
        )
        return result

    def test_signed_groups_and_sparse_exclusions(self):
        """Represent arbitrary signed Newton groups and sparse exclusions."""
        groups = [0, 1, 1, 2, 2, -3, -3, -7, -7]
        exclusions = [(1, 2), (3, 5), (6, 8)]
        result = self.assert_compiles_exactly(groups, excluded_pairs=exclusions)

        self.assertLessEqual(result.bit_count, 8)

    def test_random_small_graphs(self):
        """Represent random small Newton graphs exactly within the star bound."""
        rng = np.random.default_rng(712)
        for shape_count in (2, 8, 20, 33):
            groups = rng.integers(-4, 5, shape_count, dtype=np.int32)
            exclusions = []
            for shape_a in range(shape_count - 1):
                for shape_b in range(shape_a + 1, shape_count):
                    if rng.random() < 0.15:
                        exclusions.append((shape_a, shape_b))
            result = self.assert_compiles_exactly(groups, excluded_pairs=exclusions)
            self.assertLessEqual(result.bit_count, 32)

    def test_many_negative_groups_use_binary_codes(self):
        """Encode a large complete multipartite graph with logarithmic bits."""
        groups = -np.arange(1, 101, dtype=np.int32)
        result = self.assert_compiles_exactly(groups)

        self.assertEqual(result.bit_count, 7)

    def test_disconnected_edges_exceed_mask_capacity(self):
        """Reject a graph that provably requires more than 32 mask bits."""
        groups = np.repeat(np.arange(1, 34, dtype=np.int32), 2)
        result = compile_newton_collision_graph(groups)

        self.assertFalse(result.exact)
        self.assertEqual(result.bit_count, 33)
        self.assertGreater(result.uncovered_pair_count, 0)

    def test_large_graph_skips_reverse_compilation(self):
        """Skip reverse compilation before allocating a large pair matrix."""
        groups = np.ones(257, dtype=np.int32)
        result = compile_newton_collision_graph(groups, max_shape_count=256)

        self.assertFalse(result.exact)
        self.assertTrue(result.skipped)
        self.assertIsNone(result.uncovered_pair_count)

    def test_heavily_filtered_graph_skips_reverse_compilation(self):
        """Skip reverse compilation when sparse filters exceed its work budget."""
        groups = np.ones(47, dtype=np.int32)
        exclusions = list(itertools.combinations(range(47), 2))[:1025]
        result = compile_newton_collision_graph(groups, excluded_pairs=exclusions)

        self.assertFalse(result.exact)
        self.assertTrue(result.skipped)
        self.assertIsNone(result.uncovered_pair_count)

    def test_rejects_invalid_reverse_inputs(self):
        """Reject malformed Newton graphs and invalid mask capacities."""
        with self.assertRaises(TypeError):
            compile_newton_collision_graph([1.0])
        with self.assertRaises(ValueError):
            compile_newton_collision_graph([[1]])
        for kwargs in (
            {"max_bits": 1.5},
            {"max_bits": np.nan},
            {"max_bits": True},
            {"max_shape_count": 1.5},
            {"max_shape_count": np.nan},
            {"max_shape_count": True},
            {"max_excluded_pair_count": 1.5},
            {"max_excluded_pair_count": np.nan},
            {"max_excluded_pair_count": True},
        ):
            with self.subTest(**kwargs), self.assertRaises(TypeError):
                compile_newton_collision_graph([1], **kwargs)
        with self.assertRaises(ValueError):
            compile_newton_collision_graph([1], max_bits=33)
        with self.assertRaises(ValueError):
            compile_newton_collision_graph([1], excluded_pairs=[(0, 1)])
        with self.assertRaises(ValueError):
            compile_newton_collision_graph([1], max_shape_count=0)
        with self.assertRaises(ValueError):
            compile_newton_collision_graph([1], max_excluded_pair_count=-1)


if __name__ == "__main__":
    unittest.main()
