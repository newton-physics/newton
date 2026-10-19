# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np
import warp as wp

from newton._src.core.host_transform import (
    join_warp_values,
    pack_warp_values,
    quat_rotate_rows,
    store_transform_rows,
    transform_add_translation,
    transform_assign,
    transform_assign_matrix,
    transform_assign_position_wxyz,
    transform_from_array,
    transform_inverse,
    transform_inverse_rows,
    transform_multiply,
    transform_multiply_rows,
    transform_point,
    transform_rows,
    transform_to_matrix,
    transform_to_position_wxyz,
    transform_vector,
    warp_array_from_values,
)


def random_transform_rows(rng: np.random.Generator, count: int) -> np.ndarray:
    q = rng.normal(size=(count, 4)).astype(np.float32)
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    p = rng.uniform(-10.0, 10.0, size=(count, 3)).astype(np.float32)
    return np.concatenate((p, q), axis=1)


class TestHostTransform(unittest.TestCase):
    def setUp(self):
        self.transform = wp.transform(
            wp.vec3(1.0, 2.0, 3.0),
            wp.quat_from_axis_angle(wp.normalize(wp.vec3(1.0, 2.0, 3.0)), 0.7),
        )

    def test_conversion_round_trip(self):
        """Round-trip Viser and matrix representations in place."""
        position, wxyz = transform_to_position_wxyz(self.transform)
        from_viser = wp.transform_identity()
        transform_assign_position_wxyz(from_viser, position, wxyz)
        np.testing.assert_allclose(from_viser, self.transform)

        from_matrix = wp.transform_identity()
        transform_assign_matrix(from_matrix, transform_to_matrix(self.transform))
        np.testing.assert_allclose(from_matrix, self.transform, rtol=1.0e-6, atol=1.0e-6)

    def test_transform_operations_match_warp(self):
        """Match Warp's transform operations without public host calls."""
        other = wp.transform(wp.vec3(-2.0, 0.5, 4.0), wp.quat_rpy(0.2, -0.4, 0.1))
        point = wp.vec3(2.0, -1.0, 0.5)

        np.testing.assert_allclose(transform_inverse(self.transform), wp.transform_inverse(self.transform))
        np.testing.assert_allclose(
            transform_multiply(self.transform, other), wp.transform_multiply(self.transform, other)
        )
        np.testing.assert_allclose(transform_point(self.transform, point), wp.transform_point(self.transform, point))
        np.testing.assert_allclose(transform_vector(self.transform, point), wp.transform_vector(self.transform, point))

    def test_array_copy_and_translation_preserve_inputs(self):
        """Create fast views and copies without replacing caller-owned values."""
        source = np.asarray((1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0), dtype=np.float32)
        from_array = transform_from_array(source)
        target = wp.transform_identity()
        transform_assign(target, from_array)
        translated = transform_add_translation(target, (4.0, 5.0, 6.0))

        np.testing.assert_allclose(target, source)
        np.testing.assert_allclose(translated, (5.0, 7.0, 9.0, 0.0, 0.0, 0.0, 1.0))

    def test_batched_transforms_match_scalar_builtins(self):
        """Match Warp's scalar builtins on both sides of the native-call batch threshold."""
        rng = np.random.default_rng(7)
        # Batched NumPy math matches the builtins bit for bit only without fused
        # multiply-add contraction, so compare within float32 round-off.
        tolerance = {"rtol": 1.0e-6, "atol": 1.0e-5}
        for count in (1, 16, 17, 64):
            with self.subTest(count=count):
                a = random_transform_rows(rng, count)
                b = random_transform_rows(rng, count)
                expected_product = [
                    transform_multiply(wp.transform(*x), wp.transform(*y)) for x, y in zip(a, b, strict=True)
                ]
                expected_inverse = [transform_inverse(wp.transform(*x)) for x in a]
                expected_rotated = [
                    transform_vector(wp.transform(*x), wp.vec3(*y[:3])) for x, y in zip(a, b, strict=True)
                ]
                np.testing.assert_allclose(transform_multiply_rows(a, b), expected_product, **tolerance)
                np.testing.assert_allclose(transform_inverse_rows(a), expected_inverse, **tolerance)
                np.testing.assert_allclose(quat_rotate_rows(a[:, 3:], b[:, :3]), expected_rotated, **tolerance)

        # Broadcasting composes every left transform with every right transform.
        a = random_transform_rows(rng, 5)
        b = random_transform_rows(rng, 7)
        product = transform_multiply_rows(a[:, None], b[None])
        self.assertEqual(product.shape, (5, 7, 7))
        np.testing.assert_allclose(
            product[3, 4], transform_multiply(wp.transform(*a[3]), wp.transform(*b[4])), **tolerance
        )

    def test_pack_warp_values(self):
        """Pack homogeneous Warp value lists in one pass and fall back for other inputs."""
        vectors = [wp.vec3(1.0, 2.0, 3.0), wp.vec3(4.0, 5.0, 6.0)]
        joined = join_warp_values(vectors, wp.vec3)
        np.testing.assert_array_equal(joined, [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        self.assertIsNone(join_warp_values([*vectors, (7.0, 8.0, 9.0)], wp.vec3))
        self.assertIsNone(join_warp_values([], wp.vec3))
        np.testing.assert_array_equal(
            pack_warp_values([*vectors, (7.0, 8.0, 9.0)], wp.vec3), [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]
        )

        matrices = [wp.mat33(*range(9)), wp.mat33(*range(9, 18))]
        array = warp_array_from_values(matrices, wp.mat33, device="cpu")
        np.testing.assert_array_equal(array.numpy(), wp.array(matrices, dtype=wp.mat33, device="cpu").numpy())

        pair = (wp.vec3(1.0, 2.0, 3.0), wp.quat_identity())
        np.testing.assert_array_equal(transform_rows([self.transform, pair]), [self.transform, wp.transform(*pair)])

    def test_store_transform_rows(self):
        """Write transform rows into list and array storage."""
        rows = np.asarray(((1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0), (4.0, 5.0, 6.0, 0.0, 0.0, 1.0, 0.0)), dtype=np.float32)
        targets = np.asarray((2, 0))

        as_list = [wp.transform_identity() for _ in range(3)]
        store_transform_rows(as_list, targets, rows)
        np.testing.assert_array_equal(np.asarray(as_list)[targets], rows)
        self.assertIsInstance(as_list[2], wp.transformf)

        as_array = np.zeros((3, 7), dtype=np.float32)
        store_transform_rows(as_array, targets, rows)
        np.testing.assert_array_equal(as_array[targets], rows)


if __name__ == "__main__":
    unittest.main()
