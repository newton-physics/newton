# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""
KAMINO: Tests for unit test utilities
"""

import unittest

import numpy as np
import warp as wp

from newton._src.solvers.kamino._src.core.builder import ModelBuilderKamino
from newton._src.solvers.kamino._src.core.joints import JointActuationType, JointDoFType
from newton._src.solvers.kamino._src.core.model import ModelKamino
from newton._src.solvers.kamino._src.core.shapes import SphereShape
from newton._src.solvers.kamino.tests import setup_tests, test_context
from newton.tests.kamino.utils.checks import assert_model_equal

###
# Tests
###


class TestModelComparisonReordering(unittest.TestCase):
    """Exercises the permutation/remap logic in ``assert_model_equal(allow_reordering=True)``."""

    def setUp(self):
        if not test_context.setup_done:
            setup_tests(clear_cache=False)
        self.default_device = wp.get_device(test_context.device)

    def _build_three_body_model(self, order: tuple[str, str, str]) -> ModelKamino:
        """Builds a 3-body/1-joint/3-geom world with bodies and geoms added in ``order``.

        Bodies "a" and "b" are connected by a revolute joint, which excludes their geoms from
        collision, leaving geom pairs (a, c) and (b, c) collidable: enough to exercise
        `collidable_pairs`/`excluded_pairs` remapping when geometries are reordered.
        """
        builder = ModelBuilderKamino(default_world=False)
        world_index = builder.add_world(name="w")
        identity = wp.mat33f(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
        inertia = wp.mat33f(0.01, 0.0, 0.0, 0.0, 0.01, 0.0, 0.0, 0.0, 0.01)
        # Pose is keyed by name, not insertion index, so each physical body has the same pose
        # regardless of `order`: only row order should differ between the two built models.
        x_by_name = {"a": 0.0, "b": 1.0, "c": 2.0}
        bid = {}
        for name in order:
            bid[name] = builder.add_rigid_body(
                name=name,
                m_i=1.0,
                i_I_i=inertia,
                q_i_0=wp.transformf(x_by_name[name], 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
                world_index=world_index,
            )
        for name in order:
            builder.add_geometry(
                name=f"{name}/geom",
                body=bid[name],
                shape=SphereShape(0.1),
                world_index=world_index,
            )
        builder.add_joint(
            name="hinge",
            dof_type=JointDoFType.REVOLUTE,
            act_type=JointActuationType.FORCE,
            bid_B=bid["a"],
            bid_F=bid["b"],
            B_r_Bj=wp.vec3f(0.0, 0.0, 0.0),
            F_r_Fj=wp.vec3f(0.0, 0.0, 0.0),
            X_Bj=identity,
            tau_j_max=10.0,
            world_index=world_index,
        )
        # Fix the base body to a label rather than leaving it to the builder's row-0 default,
        # which would otherwise point at a different physical body in each `order` and make
        # `base_body_index` a spurious mismatch unrelated to the reordering under test.
        builder.set_base_body("a", world_index=world_index)
        return builder.finalize(self.default_device)

    def test_reordered_model_compares_equal(self):
        """assert_model_equal succeeds when bodies/joints/geoms are permuted but label-matched."""
        model_forward = self._build_three_body_model(("a", "b", "c"))
        model_reversed = self._build_three_body_model(("c", "b", "a"))
        assert_model_equal(self, model_forward, model_reversed, allow_reordering=True)

    def test_reordered_model_rejects_strict_comparison(self):
        """The same reordered pair fails a strict, order-sensitive (allow_reordering=False) check."""
        model_forward = self._build_three_body_model(("a", "b", "c"))
        model_reversed = self._build_three_body_model(("c", "b", "a"))
        with self.assertRaises(AssertionError):
            assert_model_equal(self, model_forward, model_reversed, allow_reordering=False)

    def test_reordered_model_detects_corrupted_local_ids(self):
        """A corrupted world-local id (bid/jid/gid) is still caught after reordering."""
        model_forward = self._build_three_body_model(("a", "b", "c"))
        model_reversed = self._build_three_body_model(("c", "b", "a"))
        corrupted = np.full(model_reversed.bodies.bid.shape, 12345, dtype=np.int32)
        model_reversed.bodies.bid.assign(corrupted)
        with self.assertRaises(AssertionError):
            assert_model_equal(self, model_forward, model_reversed, allow_reordering=True)

    def test_reordered_model_detects_unremapped_collidable_pairs(self):
        """A corrupted ``collidable_pairs`` entry is still caught after geometry reordering."""
        model_forward = self._build_three_body_model(("a", "b", "c"))
        model_reversed = self._build_three_body_model(("c", "b", "a"))
        self.assertGreater(model_reversed.geoms.collidable_pairs.shape[0], 0)
        corrupted = model_reversed.geoms.collidable_pairs.numpy().copy()
        # A same-geom "pair" can never occur in a legitimate `collidable_pairs` list (self-pairs
        # are always excluded), so this is guaranteed to be a corruption regardless of remapping.
        corrupted[0] = (corrupted[0][0], corrupted[0][0])
        model_reversed.geoms.collidable_pairs.assign(corrupted)
        with self.assertRaises(AssertionError):
            assert_model_equal(self, model_forward, model_reversed, allow_reordering=True)


###
# Test execution
###

if __name__ == "__main__":
    # Test setup
    setup_tests()

    # Run all tests
    unittest.main(verbosity=2)
