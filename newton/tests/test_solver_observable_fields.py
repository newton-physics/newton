# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Test declarative solver observable allocation and factory overrides."""

import unittest
from dataclasses import dataclass
from enum import Enum, IntEnum
from unittest.mock import patch

import numpy as np
import warp as wp

import newton
from newton.solvers import SolverBase, SolverObservableFlags, SolverObservables
from newton.solvers.experimental.coupled import SolverCoupled


class CustomFlags(Enum):
    """Use opaque identities unrelated to the Python field names."""

    TEMPERATURE = 0
    PRESSURE = "not_the_field_name"


@wp.kernel
def double_values(source: wp.array[float], destination: wp.array[float]):
    i = wp.tid()
    destination[i] = 2.0 * source[i]


class TestSolverObservableFields(unittest.TestCase):
    """Exercise inherited declarations without custom array initialization."""

    def setUp(self):
        """Build a body-indexed model without collision initialization."""
        builder = newton.ModelBuilder()
        builder.add_body(mass=0.0)
        self.model = builder.finalize(device="cpu")

    @staticmethod
    def make_solver(model):
        """Declare custom fields using only the public extension API."""

        @dataclass(eq=False)
        class ThermalObservables(SolverObservables):
            temperature: wp.array[float] | None = SolverObservables.field(
                flag=CustomFlags.TEMPERATURE, dtype=float, frequency=newton.Model.AttributeFrequency.BODY
            )

        @dataclass(eq=False)
        class ContactObservables(ThermalObservables):
            pressure: wp.array[float] | None = SolverObservables.field(
                flag=CustomFlags.PRESSURE, dtype=wp.float32, frequency=newton.Model.AttributeFrequency.CONTACT_RIGID
            )

        class Solver(SolverBase):
            OBSERVABLES_TYPE = ContactObservables
            SUPPORTED_OBSERVABLE_FLAGS = frozenset({*CustomFlags, SolverObservableFlags.BODY_QDD})

        return Solver(model)

    def test_inherited_fields_and_opaque_flags(self):
        """Allocate inherited scalar and spatial fields without name-based flags."""
        solver = self.make_solver(self.model)
        flags = {CustomFlags.TEMPERATURE, SolverObservableFlags.BODY_QDD}
        observables = solver.observables(flags, requires_grad=True)
        self.assertEqual(observables.temperature.shape, (1,))
        self.assertIs(observables.temperature.dtype, wp.float32)
        self.assertIs(observables.body_qdd.dtype, wp.spatial_vector)
        self.assertEqual(observables.get_attribute_frequency("temperature"), newton.Model.AttributeFrequency.BODY)
        self.assertIsNone(observables.pressure)
        self.assertIsNone(observables.body_parent_f)
        self.assertIsNone(self.model.rigid_contact_max)
        self.assertIsNotNone(observables.temperature.grad)
        self.assertIsNotNone(observables.body_qdd.grad)
        self.assertEqual(observables.flags, flags)
        self.assertEqual({observables: "identity"}[observables], "identity")
        self.assertNotEqual(observables, solver.observables(flags))

        with patch("newton._src.solvers.solver.wp.zeros", side_effect=AssertionError("unexpected allocation")):
            selected = observables.select({CustomFlags.TEMPERATURE})
            body_only = observables.select({SolverObservableFlags.BODY_QDD})
        self.assertIs(selected.temperature, observables.temperature)
        self.assertIs(selected.temperature.grad, observables.temperature.grad)
        self.assertIsNone(selected.body_qdd)
        self.assertIsNone(body_only.temperature)
        self.assertTrue(selected.is_requested(CustomFlags.TEMPERATURE))
        self.assertIsNone(selected.select(set()).temperature)

    def test_custom_contact_field_binds_storage(self):
        """Apply contact rules using the declaration rather than the flag value."""
        solver = self.make_solver(self.model)
        with self.assertRaisesRegex(RuntimeError, "CollisionPipeline"):
            solver.observables({CustomFlags.PRESSURE})
        pipeline = newton.CollisionPipeline(self.model, rigid_contact_max=0, soft_contact_max=0)
        observables = solver.observables({CustomFlags.PRESSURE})
        selected = observables.select({CustomFlags.PRESSURE})
        self.assertEqual(selected.pressure.shape, (0,))
        self.assertTrue(selected.is_requested(CustomFlags.PRESSURE))
        contacts = pipeline.contacts()
        solver.validate_observables(selected, contacts)
        self.assertIs(observables.contacts, contacts)
        self.assertIsNone(observables.select(set()).contacts)

    def test_redeclarations_and_enum_namespaces_are_independent(self):
        """Keep cached base declarations intact when a child overrides a field."""
        solver = self.make_solver(self.model)
        original = solver.observables({CustomFlags.TEMPERATURE})

        class OtherFlags(Enum):
            TENSOR = 0  # Same value as TEMPERATURE, but a different flag.

        @dataclass(eq=False)
        class DerivedObservables(solver.OBSERVABLES_TYPE):
            temperature: wp.array[wp.vec3] | None = SolverObservables.field(
                flag=CustomFlags.TEMPERATURE, dtype=wp.vec3, frequency=newton.Model.AttributeFrequency.PARTICLE
            )
            tensor: wp.array[wp.mat33] | None = SolverObservables.field(
                flag=OtherFlags.TENSOR, dtype=wp.mat33, frequency=newton.Model.AttributeFrequency.ONCE
            )

        class DerivedSolver(type(solver)):
            OBSERVABLES_TYPE = DerivedObservables
            SUPPORTED_OBSERVABLE_FLAGS = solver.SUPPORTED_OBSERVABLE_FLAGS | {OtherFlags.TENSOR}

        derived = DerivedSolver(self.model).observables({CustomFlags.TEMPERATURE, OtherFlags.TENSOR})
        self.assertEqual(derived.temperature.shape, (0,))
        self.assertIs(derived.temperature.dtype, wp.vec3)
        self.assertEqual(derived.tensor.shape, (1,))
        self.assertIs(derived.tensor.dtype, wp.mat33)
        self.assertIsNone(derived.select({CustomFlags.TEMPERATURE}).tensor)
        self.assertEqual(original.get_attribute_frequency("temperature"), newton.Model.AttributeFrequency.BODY)
        self.assertIs(solver.observables({CustomFlags.TEMPERATURE}).temperature.dtype, wp.float32)

    def test_custom_frequency_allocation(self):
        """Use registered model counts for custom string row domains."""
        builder = newton.ModelBuilder()
        builder.add_custom_frequency(newton.ModelBuilder.CustomFrequency(name="sample"))
        builder.add_custom_attribute(
            newton.ModelBuilder.CustomAttribute(name="sample_id", frequency="sample", dtype=int, default=0)
        )
        for sample_id in range(3):
            builder.add_custom_values(sample_id=sample_id)
        model = builder.finalize(device="cpu")

        @dataclass(eq=False)
        class SampleObservables(SolverObservables):
            temperature: wp.array[float] | None = SolverObservables.field(
                flag=CustomFlags.TEMPERATURE, dtype=float, frequency="sample"
            )

        class SampleSolver(SolverBase):
            OBSERVABLES_TYPE = SampleObservables
            SUPPORTED_OBSERVABLE_FLAGS = frozenset({CustomFlags.TEMPERATURE})

        observables = SampleSolver(model).observables({CustomFlags.TEMPERATURE})
        self.assertEqual(observables.temperature.shape, (3,))
        self.assertEqual(observables.get_attribute_frequency("temperature"), "sample")
        self.assertIsNone(model.rigid_contact_max)

    def test_declarations_do_not_imply_support(self):
        """Reject inherited fields the solver cannot compute before allocation."""
        solver = self.make_solver(self.model)
        with (
            patch("newton._src.solvers.solver.wp.zeros", side_effect=AssertionError("unexpected allocation")),
            self.assertRaisesRegex(ValueError, "does not support"),
        ):
            solver.observables({SolverObservableFlags.BODY_PARENT_F})

    def test_duplicate_flags_are_rejected_before_allocation(self):
        """Reject two fields declaring the same flag, including inherited fields."""
        solver = self.make_solver(self.model)

        @dataclass(eq=False)
        class DuplicateObservables(solver.OBSERVABLES_TYPE):
            duplicate: wp.array[float] | None = SolverObservables.field(
                flag=CustomFlags.TEMPERATURE, dtype=float, frequency=newton.Model.AttributeFrequency.BODY
            )

        solver.OBSERVABLES_TYPE = DuplicateObservables
        with (
            patch("newton._src.solvers.solver.wp.zeros", side_effect=AssertionError("unexpected allocation")),
            self.assertRaisesRegex(ValueError, "Duplicate.*TEMPERATURE"),
        ):
            solver.observables({CustomFlags.TEMPERATURE})

    def test_require_identity_dataclasses(self):
        """Reject value equality that would make observable sources unhashable."""
        solver = self.make_solver(self.model)

        @dataclass
        class ValueObservables(solver.OBSERVABLES_TYPE):
            pass

        solver.OBSERVABLES_TYPE = ValueObservables
        with self.assertRaisesRegex(TypeError, "eq=False"):
            solver.observables({CustomFlags.TEMPERATURE})

    def test_missing_dataclass_decorator(self):
        """Reject new field declarations that dataclasses have not processed."""
        solver = self.make_solver(self.model)

        class UndecoratedObservables(SolverObservables):
            temperature: wp.array[float] | None = SolverObservables.field(
                flag=CustomFlags.TEMPERATURE, dtype=float, frequency=newton.Model.AttributeFrequency.BODY
            )

        solver.OBSERVABLES_TYPE = UndecoratedObservables
        with self.assertRaisesRegex(TypeError, "dataclass"):
            solver.observables({CustomFlags.TEMPERATURE})

    def test_reject_value_like_flags(self):
        """Reject integer-like flags even when declarations use opaque values."""

        class IntegerFlags(IntEnum):
            VALUE = 0

        for flag in (0, "temperature", IntegerFlags.VALUE):
            with self.subTest(flag=flag), self.assertRaisesRegex(TypeError, "plain enum"):
                SolverObservables.field(flag=flag, dtype=float, frequency=newton.Model.AttributeFrequency.BODY)

    def test_public_factory_override(self):
        """Customize allocation through the public factory without additional hooks."""
        solver_type = type(self.make_solver(self.model))
        calls = []

        class CustomSolver(solver_type):
            def observables(self, flags, *, requires_grad=None):
                result = super().observables(flags, requires_grad=requires_grad)
                calls.append(result.flags)
                if result.is_requested(CustomFlags.TEMPERATURE):
                    result.temperature.fill_(3.0)
                self.created = result
                return result

        solver = CustomSolver(self.model)
        observables = solver.observables(iter([CustomFlags.TEMPERATURE]), requires_grad=True)
        self.assertEqual(calls, [frozenset({CustomFlags.TEMPERATURE})])
        self.assertIs(solver.created, observables)
        self.assertIs(observables.model, self.model)
        self.assertTrue(observables.temperature.requires_grad)
        np.testing.assert_array_equal(observables.temperature.numpy(), [3.0])
        observables.select(set())
        self.assertEqual(calls, [frozenset({CustomFlags.TEMPERATURE})])

    def test_factory_and_validation_are_the_only_public_lifecycle_methods(self):
        """Keep allocation and preparation details out of the public solver API."""
        for solver_type in (SolverBase, newton.solvers.SolverMuJoCo, newton.solvers.SolverKamino, SolverCoupled):
            with self.subTest(solver=solver_type.__name__):
                self.assertTrue(callable(solver_type.observables))
                self.assertTrue(callable(solver_type.validate_observables))
                self.assertFalse(hasattr(solver_type, "allocate_observable"))
                self.assertFalse(hasattr(solver_type, "prepare_observables"))

    def test_kamino_factory_failure_can_be_retried(self):
        """Keep capacity and scratch storage uncommitted when backend setup fails."""
        newton.CollisionPipeline(self.model, rigid_contact_max=1, soft_contact_max=0)
        solver = object.__new__(newton.solvers.SolverKamino)
        SolverBase.__init__(solver, self.model)
        solver._collision_detector_kamino = None
        solver._contact_observable_state = None
        flags = {SolverObservableFlags.CONTACT_F}
        with (
            patch("newton._src.solvers.kamino.solver_kamino.wp.empty", side_effect=MemoryError("scratch allocation")),
            self.assertRaisesRegex(MemoryError, "scratch allocation"),
        ):
            solver.observables(flags)
        self.assertIsNone(solver._contact_observable_state)
        newton.CollisionPipeline(self.model, rigid_contact_max=2, soft_contact_max=0)
        observables = solver.observables(flags)
        self.assertEqual(observables.contact_f.shape, (2,))
        self.assertEqual(solver._contact_observable_state.body_q.shape, (self.model.body_count,))

    def test_allocation_failure_does_not_freeze_capacity(self):
        """Leave contact capacities mutable if array allocation fails."""
        solver = self.make_solver(self.model)
        newton.CollisionPipeline(self.model, rigid_contact_max=1, soft_contact_max=0)
        with (
            patch("newton._src.solvers.solver.wp.zeros", side_effect=MemoryError("array allocation")),
            self.assertRaisesRegex(MemoryError, "array allocation"),
        ):
            solver.observables({CustomFlags.PRESSURE})
        newton.CollisionPipeline(self.model, rigid_contact_max=2, soft_contact_max=0)
        self.assertEqual(solver.observables({CustomFlags.PRESSURE}).pressure.shape, (2,))

    def test_empty_request_does_not_allocate(self):
        """Return an owned empty container without allocating arrays or requiring contacts."""
        solver = self.make_solver(self.model)
        with patch("newton._src.solvers.solver.wp.zeros", side_effect=AssertionError("unexpected allocation")):
            observables = solver.observables(set())
        self.assertIs(observables.model, self.model)
        self.assertEqual(observables.flags, frozenset())
        self.assertIsNone(observables.temperature)
        self.assertIsNone(observables.pressure)
        self.assertIsNone(observables.contact_f)
        solver.validate_observables(observables)

    def test_custom_field_gradients_and_graph_reuse(self):
        """Differentiate writes and replay CUDA graphs using selected custom arrays."""
        devices = ["cpu"] + (["cuda:0"] if wp.is_cuda_available() else [])
        for device in devices:
            with self.subTest(device=device):
                builder = newton.ModelBuilder()
                builder.add_body(mass=0.0)
                model = builder.finalize(device=device, requires_grad=True)
                solver = self.make_solver(model)
                observables = solver.observables({CustomFlags.TEMPERATURE, SolverObservableFlags.BODY_QDD})
                selected = observables.select({CustomFlags.TEMPERATURE})
                source = wp.ones(1, device=device, requires_grad=True)
                with wp.Tape() as tape:
                    wp.launch(double_values, dim=1, inputs=[source], outputs=[selected.temperature], device=device)
                tape.backward(grads={selected.temperature: wp.ones(1, device=device)})
                np.testing.assert_array_equal(source.grad.numpy(), [2.0])
                self.assertIs(selected.temperature.grad, observables.temperature.grad)
                if model.device.is_cuda:
                    pointer = selected.temperature.ptr
                    with wp.ScopedCapture(device=device) as capture:
                        wp.launch(double_values, dim=1, inputs=[source], outputs=[selected.temperature], device=device)
                    source.fill_(5.0)
                    wp.capture_launch(capture.graph)
                    np.testing.assert_array_equal(observables.temperature.numpy(), [10.0])
                    self.assertEqual(selected.temperature.ptr, pointer)


if __name__ == "__main__":
    unittest.main()
