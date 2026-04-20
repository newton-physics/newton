# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for Model.reset_state()."""

import unittest

import numpy as np
import warp as wp

import newton


class TestResetState(unittest.TestCase):
    """Tests that Model.reset_state() restores state arrays in-place."""

    def _build_body_model(self):
        """Build a model with one free body and a sphere shape."""
        builder = newton.ModelBuilder()
        body = builder.add_body(mass=1.0)
        builder.add_shape_sphere(body, radius=0.1)
        return builder.finalize()

    def _build_particle_model(self):
        """Build a model with 2 particles."""
        builder = newton.ModelBuilder()
        builder.add_particle(pos=(1.0, 2.0, 3.0), vel=(0.1, 0.2, 0.3), mass=1.0)
        builder.add_particle(pos=(4.0, 5.0, 6.0), vel=(0.4, 0.5, 0.6), mass=1.0)
        return builder.finalize()

    def _build_articulation_model(self):
        """Build a model with a revolute joint articulation.

        Sets non-zero joint offsets so that FK-computed body transforms
        differ from the raw model.body_q defaults.
        """
        builder = newton.ModelBuilder()
        link0 = builder.add_link(mass=1.0)
        builder.add_shape_sphere(link0, radius=0.1)
        link1 = builder.add_link(mass=1.0)
        builder.add_shape_sphere(link1, radius=0.1)
        j0 = builder.add_joint_revolute(
            parent=-1,
            child=link0,
            parent_xform=wp.transform((0.0, 1.0, 0.0), wp.quat_identity()),
        )
        j1 = builder.add_joint_revolute(
            parent=link0,
            child=link1,
            parent_xform=wp.transform((0.0, 1.0, 0.0), wp.quat_identity()),
        )
        builder.add_articulation([j0, j1])
        model = builder.finalize()

        # Set raw model.body_q to zeros so it differs from FK-computed values
        model.body_q.zero_()
        model.body_qd.zero_()

        return model

    def test_reset_restores_body_state(self):
        model = self._build_body_model()
        state = model.state()

        # Save initial values
        initial_body_q = state.body_q.numpy().copy()
        initial_body_qd = state.body_qd.numpy().copy()

        # Mutate body arrays with 999.0
        junk_q = wp.array(np.full_like(initial_body_q, 999.0), dtype=state.body_q.dtype)
        junk_qd = wp.array(np.full_like(initial_body_qd, 999.0), dtype=state.body_qd.dtype)
        wp.copy(state.body_q, junk_q)
        wp.copy(state.body_qd, junk_qd)

        # Seed body_f with non-zero to verify zeroing is meaningful
        junk_f = wp.array(np.full_like(state.body_f.numpy(), 42.0), dtype=state.body_f.dtype)
        wp.copy(state.body_f, junk_f)
        self.assertTrue(np.any(state.body_f.numpy() != 0.0))

        # Verify mutation took effect
        np.testing.assert_array_equal(state.body_q.numpy(), junk_q.numpy())

        # Reset
        model.reset_state(state)

        # Verify body_q and body_qd restored
        np.testing.assert_array_equal(state.body_q.numpy(), initial_body_q)
        np.testing.assert_array_equal(state.body_qd.numpy(), initial_body_qd)

        # Verify body_f is zeroed
        np.testing.assert_array_equal(state.body_f.numpy(), np.zeros_like(state.body_f.numpy()))

    def test_reset_restores_particle_state(self):
        model = self._build_particle_model()
        state = model.state()

        # Save initial values
        initial_particle_q = state.particle_q.numpy().copy()
        initial_particle_qd = state.particle_qd.numpy().copy()

        # Mutate both position and velocity
        junk_q = wp.array(np.full_like(initial_particle_q, 999.0), dtype=state.particle_q.dtype)
        junk_qd = wp.array(np.full_like(initial_particle_qd, 999.0), dtype=state.particle_qd.dtype)
        wp.copy(state.particle_q, junk_q)
        wp.copy(state.particle_qd, junk_qd)

        # Seed particle_f with non-zero to verify zeroing is meaningful
        junk_f = wp.array(np.full_like(state.particle_f.numpy(), 42.0), dtype=state.particle_f.dtype)
        wp.copy(state.particle_f, junk_f)
        self.assertTrue(np.any(state.particle_f.numpy() != 0.0))

        # Reset
        model.reset_state(state)

        # Verify particle arrays restored
        np.testing.assert_array_equal(state.particle_q.numpy(), initial_particle_q)
        np.testing.assert_array_equal(state.particle_qd.numpy(), initial_particle_qd)

        # Verify particle_f is zeroed
        np.testing.assert_array_equal(state.particle_f.numpy(), np.zeros_like(state.particle_f.numpy()))

    def test_reset_restores_joint_state(self):
        model = self._build_articulation_model()
        state = model.state()

        # Save initial joint values
        initial_joint_q = state.joint_q.numpy().copy()
        initial_joint_qd = state.joint_qd.numpy().copy()

        # Mutate both joint_q and joint_qd
        junk_q = wp.array(np.full_like(initial_joint_q, 999.0), dtype=state.joint_q.dtype)
        junk_qd = wp.array(np.full_like(initial_joint_qd, 999.0), dtype=state.joint_qd.dtype)
        wp.copy(state.joint_q, junk_q)
        wp.copy(state.joint_qd, junk_qd)

        # Reset
        model.reset_state(state)

        # Verify joint arrays restored
        np.testing.assert_array_equal(state.joint_q.numpy(), initial_joint_q)
        np.testing.assert_array_equal(state.joint_qd.numpy(), initial_joint_qd)

    def test_reset_with_eval_fk(self):
        model = self._build_articulation_model()
        state = model.state()

        # Compute FK to get expected body transforms (differs from raw model.body_q)
        newton.eval_fk(model, state.joint_q, state.joint_qd, state)
        expected_body_q = state.body_q.numpy().copy()
        expected_body_qd = state.body_qd.numpy().copy()

        # Verify FK result differs from raw model values
        self.assertFalse(np.array_equal(expected_body_q, model.body_q.numpy()))

        # Mutate body_q
        junk = wp.array(np.full_like(expected_body_q, 999.0), dtype=state.body_q.dtype)
        wp.copy(state.body_q, junk)

        # Reset with eval_fk=True (the default)
        model.reset_state(state, eval_fk=True)

        # Verify body_q matches FK-computed values, not raw model values
        np.testing.assert_allclose(state.body_q.numpy(), expected_body_q, atol=1e-5)
        np.testing.assert_allclose(state.body_qd.numpy(), expected_body_qd, atol=1e-5)

    def test_reset_without_eval_fk(self):
        model = self._build_articulation_model()
        state = model.state()

        # Compute FK first so we know what the FK result would be
        newton.eval_fk(model, state.joint_q, state.joint_qd, state)
        fk_body_q = state.body_q.numpy().copy()

        # Raw model values should differ from FK
        raw_body_q = model.body_q.numpy().copy()
        raw_body_qd = model.body_qd.numpy().copy()
        self.assertFalse(np.array_equal(raw_body_q, fk_body_q))

        # Mutate body_q
        junk = wp.array(np.full_like(raw_body_q, 999.0), dtype=state.body_q.dtype)
        wp.copy(state.body_q, junk)

        # Reset with eval_fk=False
        model.reset_state(state, eval_fk=False)

        # Verify body_q matches raw model values, NOT FK-computed
        np.testing.assert_array_equal(state.body_q.numpy(), raw_body_q)
        np.testing.assert_array_equal(state.body_qd.numpy(), raw_body_qd)

    def test_reset_does_not_reallocate(self):
        model = self._build_body_model()
        state = model.state()

        # Record pointer
        ptr_before = state.body_q.ptr

        # Reset
        model.reset_state(state)

        # Verify pointer unchanged (no reallocation)
        self.assertEqual(state.body_q.ptr, ptr_before)

    def test_reset_zeroes_extended_body_buffers(self):
        """Extended state attributes (body_qdd, body_parent_f) are zeroed."""
        builder = newton.ModelBuilder()
        builder.add_body(mass=1.0)
        builder.add_shape_sphere(body=0, radius=0.1)
        builder.request_state_attributes("body_qdd", "body_parent_f")
        model = builder.finalize()
        state = model.state()

        # Seed extended buffers with non-zero
        junk = wp.array(np.full_like(state.body_qdd.numpy(), 42.0), dtype=state.body_qdd.dtype)
        wp.copy(state.body_qdd, junk)
        wp.copy(state.body_parent_f, junk)
        self.assertTrue(np.any(state.body_qdd.numpy() != 0.0))

        model.reset_state(state)

        np.testing.assert_array_equal(state.body_qdd.numpy(), np.zeros_like(state.body_qdd.numpy()))
        np.testing.assert_array_equal(state.body_parent_f.numpy(), np.zeros_like(state.body_parent_f.numpy()))


if __name__ == "__main__":
    unittest.main(verbosity=2)
