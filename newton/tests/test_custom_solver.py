# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for extending solver change/reset flags with custom integer bits."""

import unittest

import numpy as np
import warp as wp

import newton


class DummySolver(newton.solvers.SolverBase):
    """Minimal solver that consumes extension flags and custom attributes."""

    # These bits intentionally live outside Newton's built-in flag range.
    MODEL_ATTRIBUTE_CHANGED = 1 << 20
    STATE_ATTRIBUTE_RESET = 1 << 21

    def __init__(self, model: newton.Model):
        """Initialize bookkeeping used by the tests."""
        super().__init__(model)
        self.notify_flags: int | None = None
        self.reset_flags: int | None = None
        self.saw_body_properties = False
        self.saw_body_q = False
        self.model_epoch: int | None = None
        self.reset_epoch: int | None = None

    def notify_model_changed(self, flags: newton.ModelFlags | int) -> None:
        """Consume both built-in model flags and a custom solver flag."""
        self.notify_flags = flags
        self.saw_body_properties = bool(flags & newton.ModelFlags.BODY_PROPERTIES)
        if flags & self.MODEL_ATTRIBUTE_CHANGED:
            self.model_epoch = int(self.model.custom_solver.model_epoch.numpy()[0])

    def reset(
        self,
        state: newton.State,
        world_mask: wp.array | None = None,
        flags: newton.StateFlags | int | None = None,
    ) -> None:
        """Consume both built-in state flags and a custom solver reset flag."""
        del world_mask
        reset_flags = int(newton.StateFlags.ALL if flags is None else flags)
        self.reset_flags = reset_flags
        self.saw_body_q = bool(reset_flags & newton.StateFlags.BODY_Q)
        if reset_flags & self.STATE_ATTRIBUTE_RESET:
            self.reset_epoch = int(state.custom_solver.reset_epoch.numpy()[0])
            state.custom_solver.reset_epoch.assign(np.array([self.reset_epoch + 1], dtype=np.int32))

    @staticmethod
    def register_custom_attributes(builder: newton.ModelBuilder) -> None:
        """Register custom buffers that the dummy solver owns."""
        builder.add_custom_attribute(
            newton.ModelBuilder.CustomAttribute(
                name="model_epoch",
                dtype=wp.int32,
                frequency=newton.Model.AttributeFrequency.BODY,
                assignment=newton.Model.AttributeAssignment.MODEL,
                namespace="custom_solver",
                default=-1,
            )
        )
        builder.add_custom_attribute(
            newton.ModelBuilder.CustomAttribute(
                name="reset_epoch",
                dtype=wp.int32,
                frequency=newton.Model.AttributeFrequency.BODY,
                assignment=newton.Model.AttributeAssignment.STATE,
                namespace="custom_solver",
                default=-1,
            )
        )


class TestCustomSolver(unittest.TestCase):
    """Verify custom solver flags can be regular Python integer bitmasks."""

    def _build_model(self) -> newton.Model:
        """Build a one-body model with custom solver-owned attributes."""
        builder = newton.ModelBuilder()
        DummySolver.register_custom_attributes(builder)
        builder.add_body(
            mass=0.0,
            custom_attributes={
                "custom_solver:model_epoch": 7,
                "custom_solver:reset_epoch": 11,
            },
        )
        return builder.finalize()

    def test_notify_model_changed_accepts_custom_int_flag(self):
        """Model-change notifications preserve custom integer bits."""
        model = self._build_model()
        solver = DummySolver(model)
        flags = newton.ModelFlags.BODY_PROPERTIES | DummySolver.MODEL_ATTRIBUTE_CHANGED

        # IntEnum combinations with unknown bits become plain ints, which is
        # what lets downstream solvers define their own extension flags.
        self.assertIs(type(flags), int)

        solver.notify_model_changed(flags)

        self.assertEqual(solver.notify_flags, flags)
        self.assertTrue(solver.saw_body_properties)
        self.assertEqual(solver.model_epoch, 7)

    def test_reset_accepts_custom_int_flag(self):
        """State resets preserve custom integer bits."""
        model = self._build_model()
        state = model.state()
        solver = DummySolver(model)
        flags = newton.StateFlags.BODY_Q | DummySolver.STATE_ATTRIBUTE_RESET

        # Keep this assertion explicit so a future enum implementation cannot
        # accidentally reject extension bits by coercing them back to StateFlags.
        self.assertIs(type(flags), int)

        solver.reset(state, flags=flags)

        self.assertEqual(solver.reset_flags, flags)
        self.assertTrue(solver.saw_body_q)
        self.assertEqual(solver.reset_epoch, 11)
        self.assertEqual(int(state.custom_solver.reset_epoch.numpy()[0]), 12)


if __name__ == "__main__":
    unittest.main()
