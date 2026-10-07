# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Test declarative solver observable allocation and extension hooks."""

import unittest
from dataclasses import dataclass
from enum import Enum, IntEnum
from unittest.mock import patch

import numpy as np
import warp as wp

import newton
from newton.solvers import SolverBase, SolverObservableFlags, SolverObservables


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

        with patch.object(solver, "allocate_observable", side_effect=AssertionError("unexpected allocation")):
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
            patch.object(solver, "allocate_observable", side_effect=AssertionError("unexpected allocation")),
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
            patch.object(solver, "allocate_observable", side_effect=AssertionError("unexpected allocation")),
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

    def test_public_allocation_and_preparation_hooks(self):
        """Call preparation once after allocating only the requested fields."""
        solver_type = type(self.make_solver(self.model))
        calls = []

        class CustomSolver(solver_type):
            def allocate_observable(self, flag, *, requires_grad):
                calls.append(flag)
                array = super().allocate_observable(flag, requires_grad=requires_grad)
                array.fill_(3.0)
                return array

            def prepare_observables(self, observables, *, requires_grad):
                calls.append("prepare")
                self.prepared = observables
                self.prepared_gradient = requires_grad
                self.prepared_array = observables.temperature

        solver = CustomSolver(self.model)
        observables = solver.observables({CustomFlags.TEMPERATURE}, requires_grad=True)
        self.assertEqual(calls, [CustomFlags.TEMPERATURE, "prepare"])
        self.assertIs(solver.prepared, observables)
        self.assertIs(solver.prepared_array, observables.temperature)
        self.assertIs(observables.model, self.model)
        self.assertTrue(solver.prepared_gradient)
        np.testing.assert_array_equal(observables.temperature.numpy(), [3.0])
        observables.select(set())
        self.assertEqual(calls, [CustomFlags.TEMPERATURE, "prepare"])

    def test_allocation_failure_does_not_freeze_capacity(self):
        """Leave contact capacities mutable if preparation fails."""
        solver = self.make_solver(self.model)
        newton.CollisionPipeline(self.model, rigid_contact_max=1, soft_contact_max=0)
        with (
            patch.object(solver, "prepare_observables", side_effect=ValueError("preparation failed")),
            self.assertRaisesRegex(ValueError, "preparation failed"),
        ):
            solver.observables({CustomFlags.PRESSURE})
        newton.CollisionPipeline(self.model, rigid_contact_max=2, soft_contact_max=0)
        self.assertEqual(solver.observables({CustomFlags.PRESSURE}).pressure.shape, (2,))

    def test_invalid_allocation_hook_results(self):
        """Reject absent, wrong-dtype, and wrong-device arrays before preparation."""
        solver = self.make_solver(self.model)
        invalid = [(None, TypeError, "Warp array"), (wp.zeros(1, dtype=int, device="cpu"), TypeError, "dtype")]
        if wp.is_cuda_available():
            invalid.append((wp.zeros(1, dtype=float, device="cuda:0"), ValueError, "solver device"))
        for array, error, message in invalid:
            with (
                self.subTest(array=array),
                patch.object(solver, "allocate_observable", return_value=array),
                patch.object(solver, "prepare_observables", side_effect=AssertionError("unexpected preparation")),
                self.assertRaisesRegex(error, message),
            ):
                solver.observables({CustomFlags.TEMPERATURE})

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
