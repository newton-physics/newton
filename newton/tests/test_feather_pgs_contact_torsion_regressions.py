# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Exercise contact-torsion lifecycle and row-storage safety across steps."""

import unittest

import numpy as np
import warp as wp

from newton import GeoType
from newton._src.solvers.feather_pgs.contact_torsion import _contact_groups
from newton.solvers import SolverFeatherPGS
from newton.tests.test_feather_pgs_contact_torsion import fixture


@unittest.skipUnless(wp.is_cuda_available(), "Contact torsion requires CUDA")
class TestContactTorsionRegressions(unittest.TestCase):
    """Reject invalid storage/lifecycle paths rather than silently welding spin."""

    def test_radius_is_construction_only(self):
        """Reject radius mutation for both compiled feature variants."""
        for radius in (0.0, 0.01):
            _, solver, *_ = fixture(radius, center_only=True)
            for replacement in (0.0, 0.02, -1.0, float("nan")):
                with self.subTest(radius=radius, replacement=replacement), self.assertRaises(AttributeError):
                    solver.contact_torsion_radius = replacement

    def test_previous_friction_rows_are_not_current_rows(self):
        """Reject stale tangent metadata when a later step admits only a normal."""
        _, solver, model, initial, contacts = fixture(0.01, center_only=True)
        self.assertEqual(solver._torsion_stats["rows"], 1)
        solver.contact_friction_gap_threshold = -1.0
        solver.step(initial, model.state(), model.control(), contacts, 0.0025)
        self.assertEqual(solver._torsion_stats["rows"], 0)

    def test_both_tangent_rows_must_belong_to_current_normal(self):
        """Reject a missing or mismatched second tangent, not just the first."""
        for invalid in ("count", "type", "parent"):
            with self.subTest(invalid=invalid):
                _, solver, _model, initial, contacts = fixture(0.01, center_only=True)
                row = solver._torsion_stats["groups"][0]["normal_rows"][0]
                if invalid == "count":
                    counts = solver.constraint_count.numpy()
                    counts[0] = row + 2
                    solver.constraint_count.assign(counts)
                else:
                    array = solver.row_type if invalid == "type" else solver.row_parent
                    values = array.numpy()
                    values[0, row + 2] = -1
                    array.assign(values)
                self.assertEqual(_contact_groups(solver, initial, contacts), [])

    def test_selectors_are_construction_only(self):
        """Reject selector mutation rather than leaving a stale resolved selection."""
        _, solver, *_ = fixture(0.01, center_only=True)
        for name in ("contact_torsion_shape_indices", "contact_torsion_shape_patterns"):
            with self.subTest(name=name), self.assertRaises(AttributeError):
                setattr(solver, name, ())

    def test_normal_parent_metadata_is_preserved(self):
        """Keep CONTACT parents available for pooled friction-patch load rings."""
        result, _solver, *_ = fixture(0.01, center_only=True)
        normal = result["row_type"][0, : result["count"][0]] == 0
        self.assertGreater(np.count_nonzero(normal), 0)
        np.testing.assert_array_equal(result["row_parent"][0, : result["count"][0]][normal], -1)

    def test_torsion_uses_cfm_floor(self):
        """Apply the same configured diagonal floor as other dense rows."""
        result, solver, *_ = fixture(0.01, center_only=True, pgs_cfm=0.0007)
        active = result["row_type"] == 7
        self.assertGreater(np.count_nonzero(active), 0)
        np.testing.assert_allclose(solver.row_cfm.numpy()[active], 0.0007)

    def test_torsion_respects_relaxation(self):
        """Apply half the unconstrained spin correction at omega one half."""
        result, solver, *_ = fixture(100.0, center_only=True, pgs_iterations=1, pgs_omega=0.5)
        self.assertEqual(solver._torsion_stats["rows"], 1)
        spin = np.abs(result["v_out"].reshape(2, 6)[:, 5])
        np.testing.assert_allclose(spin, 0.5, atol=1e-4)

    def test_contact_row_loss_fails_without_diagnostics(self):
        """Detect rolled-back dropped contacts even with optional telemetry off."""
        baseline, *_ = fixture(0.0, center_only=True)
        limit = int(baseline["count"][0]) - 1
        with self.assertRaisesRegex(RuntimeError, "[Oo]verflow|[Dd]ropped|capacity"):
            fixture(0.01, center_only=True, row_limit=limit, row_watermark=False, warn_constraint_overflow=False)

    def test_selected_unsupported_shape_fails_at_construction(self):
        """Reject explicitly selected unsupported geometry instead of ignoring it."""
        _, _solver, model, *_ = fixture(0.0, center_only=True)
        types = model.shape_type.numpy()
        types[0] = int(GeoType.MESH)
        model.shape_type.assign(types)
        with self.assertRaisesRegex(ValueError, "[Uu]nsupported.*shape|shape.*[Uu]nsupported"):
            SolverFeatherPGS(
                model, pgs_mode="matrix_free", contact_torsion_radius=0.01, contact_torsion_shape_indices=(0,)
            )

    def test_hydro_contact_input_is_rejected(self):
        """Reject actual positive hydro contact stiffness, not a synthetic solver flag."""
        _, solver, model, initial, contacts = fixture(0.01, center_only=True)
        contacts.rigid_contact_stiffness = wp.ones(contacts.rigid_contact_max, device=model.device)
        with self.assertRaisesRegex(ValueError, "hydroelastic"):
            solver.step(initial, model.state(), model.control(), contacts, 0.0025)


if __name__ == "__main__":
    unittest.main()
