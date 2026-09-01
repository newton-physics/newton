# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for extending solver change/reset flags with custom integer bits."""

from __future__ import annotations

import unittest
from enum import Enum, IntEnum

import numpy as np
import warp as wp

import newton


class DummyOutputFlags(Enum):
    """Solver-specific outputs used to exercise extension behavior."""

    BODY_TEMPERATURE = "body_temperature"


class IntegerOutputFlags(IntEnum):
    """Invalid value-like enum used to verify collision prevention."""

    BODY_TEMPERATURE = 0


class DummySolverOutputs(newton.solvers.SolverOutputs):
    """Extend the standard output container with a custom body array."""

    def __init__(self, flags=()):
        """Initialize the inherited and custom output fields."""
        super().__init__(flags)
        self.body_temperature: wp.array[wp.float32] | None = None


class DummySolver(newton.solvers.SolverBase):
    """Minimal solver that consumes extension flags and custom attributes."""

    # These bits intentionally live outside Newton's built-in flag range.
    MODEL_ATTRIBUTE_CHANGED = 1 << 20
    STATE_ATTRIBUTE_RESET = 1 << 21
    OUTPUTS_TYPE = DummySolverOutputs
    SUPPORTED_OUTPUT_FLAGS = frozenset(
        {
            newton.solvers.SolverOutputFlags.BODY_QDD,
            DummyOutputFlags.BODY_TEMPERATURE,
        }
    )

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

    def _allocate_outputs(self, outputs: DummySolverOutputs, *, requires_grad: bool) -> None:
        """Allocate inherited outputs before solver-specific arrays."""
        super()._allocate_outputs(outputs, requires_grad=requires_grad)
        if DummyOutputFlags.BODY_TEMPERATURE in outputs:
            outputs.body_temperature = wp.zeros(
                self.model.body_count,
                dtype=wp.float32,
                device=self.model.device,
                requires_grad=requires_grad,
            )

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

    def test_base_reset_validates_global_world_mask_slot(self):
        """Require a final global slot while deprecating local-only masks."""
        builder = newton.ModelBuilder()
        builder.begin_world()
        builder.end_world()
        builder.begin_world()
        builder.end_world()
        model = builder.finalize()
        solver = newton.solvers.SolverBase(model)
        state = model.state()

        solver.reset(state, world_mask=wp.array((True, False, True), dtype=wp.bool, device=model.device))

        with self.assertWarnsRegex(DeprecationWarning, "world_count \\+ 1"):
            solver.reset(state, world_mask=wp.array((True, False), dtype=wp.bool, device=model.device))

        with self.assertRaisesRegex(ValueError, "expected 2 or 3"):
            solver.reset(state, world_mask=wp.array((True,), dtype=wp.bool, device=model.device))

    def test_outputs_compose_standard_and_custom_flags(self):
        """Allocate inherited and solver-specific outputs from one set."""
        model = self._build_model()
        solver = DummySolver(model)
        requested = {
            newton.solvers.SolverOutputFlags.BODY_QDD,
            DummyOutputFlags.BODY_TEMPERATURE,
        }

        outputs = solver.outputs(requested)

        self.assertIsInstance(outputs, DummySolverOutputs)
        self.assertEqual(outputs.flags, frozenset(requested))
        self.assertEqual(outputs.body_qdd.shape, (model.body_count,))
        self.assertEqual(outputs.body_temperature.shape, (model.body_count,))
        self.assertIsNone(outputs.body_parent_f)

    def test_outputs_reject_unsupported_flags(self):
        """Reject standard outputs not implemented by a solver."""
        model = self._build_model()
        solver = DummySolver(model)

        with self.assertRaisesRegex(ValueError, "BODY_PARENT_F"):
            solver.outputs({newton.solvers.SolverOutputFlags.BODY_PARENT_F})

    def test_outputs_reject_value_like_flags(self):
        """Reject string and integer enum keys that can collide across extensions."""
        model = self._build_model()
        solver = DummySolver(model)

        with self.assertRaisesRegex(TypeError, "plain enum"):
            solver.outputs({"body_qdd"})
        with self.assertRaisesRegex(TypeError, "IntEnum"):
            solver.outputs({IntegerOutputFlags.BODY_TEMPERATURE})

    def test_extended_attribute_requests_are_deprecated(self):
        """Keep legacy allocation requests while directing callers to solver outputs."""
        builder = newton.ModelBuilder()
        with self.assertWarnsRegex(DeprecationWarning, "SolverOutputs"):
            builder.request_state_attributes("body_qdd")
        with self.assertWarnsRegex(DeprecationWarning, "SolverOutputs"):
            builder.request_contact_attributes("force")

        model = builder.finalize()
        with self.assertWarnsRegex(DeprecationWarning, "SolverOutputs"):
            model.request_state_attributes("body_parent_f")
        with self.assertWarnsRegex(DeprecationWarning, "SolverOutputs"):
            model.request_contact_attributes("force")


if __name__ == "__main__":
    unittest.main()
