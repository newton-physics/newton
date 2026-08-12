# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np
import warp as wp

from newton._src.solvers.limx import AffineBodyModel
from newton._src.solvers.limx.constraints.affine_body_contact import ConstraintAffineBodyContact


def _unit_tetrahedron() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    vertices = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    tetrahedra = np.asarray([[0, 1, 2, 3]], dtype=np.int32)
    surface_triangles = np.asarray(
        [[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]],
        dtype=np.int32,
    )
    return vertices, tetrahedra, surface_triangles


def _two_body_model(translation: tuple[float, float, float], device: str = "cpu") -> AffineBodyModel:
    vertices, tetrahedra, surface_triangles = _unit_tetrahedron()
    return AffineBodyModel.from_instances(
        vertices,
        tetrahedra,
        surface_triangles,
        density=6.0,
        rigidity=0.0,
        initial_transforms=[
            wp.transform_identity(),
            wp.transform(wp.vec3(*translation), wp.quat_identity()),
        ],
        device=device,
    )


def _make_contact(model: AffineBodyModel, **overrides) -> ConstraintAffineBodyContact:
    parameters = {
        "body_model": model,
        "thickness": 0.01,
        "stiffness": 10.0,
        "normal_damping": 0.5,
        "friction": 0.2,
        "friction_epsilon": 1.0e-4,
        "max_contacts": 256,
    }
    parameters.update(overrides)
    return ConstraintAffineBodyContact(**parameters)


def _active_rows(buffer) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    count = min(int(buffer.count.numpy()[0]), buffer.capacity)
    return (
        buffer.ids.numpy()[:count],
        buffer.weights.numpy()[:count],
        buffer.depths.numpy()[:count],
    )


def _find_row(ids: np.ndarray, first: int, others: tuple[int, ...]) -> int:
    target = tuple(sorted(others))
    for row, contact_ids in enumerate(ids):
        if int(contact_ids[0]) == first and tuple(sorted(int(value) for value in contact_ids[1:])) == target:
            return row
    return -1


def _find_edge_row(ids: np.ndarray, edge_0: tuple[int, int], edge_1: tuple[int, int]) -> int:
    target_0 = tuple(sorted(edge_0))
    target_1 = tuple(sorted(edge_1))
    for row, contact_ids in enumerate(ids):
        actual_0 = tuple(sorted(int(value) for value in contact_ids[:2]))
        actual_1 = tuple(sorted(int(value) for value in contact_ids[2:]))
        if (actual_0, actual_1) == (target_0, target_1) or (actual_0, actual_1) == (target_1, target_0):
            return row
    return -1


class TestConstraintAffineBodyContactDetection(unittest.TestCase):
    def test_detects_vf_interior_edge_and_vertex_regions(self):
        """Retain triangle interior, edge, and vertex closest-point VF contacts."""
        placements = (
            ((0.33, 0.33, 0.342), 0),
            ((0.5, 0.5, -0.002), 1),
            ((1.002, -0.002, -0.002), 2),
        )

        for translation, expected_zero_weights in placements:
            with self.subTest(translation=translation):
                model = _two_body_model(translation)
                contact = _make_contact(model)
                contact.begin_step(model.q, model.qd, 0.01)
                contact.prepare(model.q)
                ids, weights, depths = _active_rows(contact.vertex_face_contacts)
                row = _find_row(ids, 4, (1, 2, 3))

                self.assertGreaterEqual(row, 0)
                self.assertGreater(depths[row], 0.0)
                zero_weights = int(np.count_nonzero(np.abs(weights[row, 1:]) < 1.0e-5))
                self.assertEqual(zero_weights, expected_zero_weights)

    def test_detects_only_cross_body_stencils(self):
        """Reject every VF and EE stencil whose features share one affine body."""
        model = _two_body_model((0.002, 0.0, 0.0))
        contact = _make_contact(model)
        contact.begin_step(model.q, model.qd, 0.01)
        contact.prepare(model.q)
        ownership = model.surface_ownership.numpy()
        vf_ids, _vf_weights, _vf_depths = _active_rows(contact.vertex_face_contacts)
        ee_ids, _ee_weights, _ee_depths = _active_rows(contact.edge_edge_contacts)

        self.assertGreater(len(vf_ids) + len(ee_ids), 0)
        self.assertTrue(np.all(ownership[vf_ids[:, 0]] != ownership[vf_ids[:, 1]]))
        self.assertTrue(np.all(ownership[ee_ids[:, 0]] != ownership[ee_ids[:, 2]]))

    def test_accepts_strict_interior_ee_and_rejects_endpoint_ee(self):
        """Accept EE only when both closest parameters lie strictly inside."""
        interior_model = _two_body_model((0.25, -0.5, 0.002))
        interior_contact = _make_contact(interior_model)
        interior_contact.begin_step(interior_model.q, interior_model.qd, 0.01)
        interior_contact.prepare(interior_model.q)
        interior_ids, interior_weights, _depths = _active_rows(interior_contact.edge_edge_contacts)
        row = _find_edge_row(interior_ids, (0, 1), (4, 6))

        self.assertGreaterEqual(row, 0)
        self.assertGreater(interior_weights[row, 1], 0.0)
        self.assertLess(interior_weights[row, 1], 1.0)
        self.assertGreater(-interior_weights[row, 3], 0.0)
        self.assertLess(-interior_weights[row, 3], 1.0)

        endpoint_model = _two_body_model((0.0, -0.5, 0.002))
        endpoint_contact = _make_contact(endpoint_model)
        endpoint_contact.begin_step(endpoint_model.q, endpoint_model.qd, 0.01)
        endpoint_contact.prepare(endpoint_model.q)
        endpoint_ids, _weights, _depths = _active_rows(endpoint_contact.edge_edge_contacts)

        self.assertEqual(_find_edge_row(endpoint_ids, (0, 1), (4, 6)), -1)

    def test_counts_contact_buffer_overflow(self):
        """Count excess contacts without writing beyond fixed buffer capacity."""
        model = _two_body_model((0.002, 0.0, 0.0))
        contact = _make_contact(model, max_contacts=1)
        contact.begin_step(model.q, model.qd, 0.01)
        contact.prepare(model.q)

        overflow = int(contact.vertex_face_contacts.overflow_count.numpy()[0])
        overflow += int(contact.edge_edge_contacts.overflow_count.numpy()[0])
        self.assertGreater(overflow, 0)

    def test_rejects_invalid_contact_configuration(self):
        """Reject invalid affine contact models, coefficients, and capacities."""
        model = _two_body_model((2.0, 0.0, 0.0))
        one_body_vertices, one_body_tets, one_body_surface = _unit_tetrahedron()
        one_body_model = AffineBodyModel(
            one_body_vertices,
            one_body_tets,
            one_body_surface,
            density=1.0,
            rigidity=0.0,
            initial_transform=wp.transform_identity(),
            device="cpu",
        )
        cases = [
            ({"body_model": object()}, "body_model"),
            ({"body_model": one_body_model}, "two affine bodies"),
            ({"thickness": 0.0}, "thickness"),
            ({"stiffness": np.nan}, "stiffness"),
            ({"normal_damping": -1.0}, "normal_damping"),
            ({"friction": -1.0}, "friction"),
            ({"friction_epsilon": 0.0}, "friction_epsilon"),
            ({"max_contacts": 0}, "max_contacts"),
        ]

        for overrides, message in cases:
            with self.subTest(overrides=overrides), self.assertRaisesRegex((TypeError, ValueError), message):
                _make_contact(model, **overrides)


if __name__ == "__main__":
    unittest.main()
