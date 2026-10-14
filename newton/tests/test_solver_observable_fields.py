# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Test declarative solver observable allocation and factory overrides."""

from __future__ import annotations

import unittest
from dataclasses import dataclass
from unittest.mock import patch

import numpy as np
import warp as wp

import newton
from newton.solvers import SolverBase


class CustomKind:
    """Exercise explicit kind names independently of Python field names."""

    TEMPERATURE = "temperature"
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

    def test_string_constants_and_inferred_fields(self):
        """Validate inherited string constants and allocate their fields by name."""

        class TemperatureSolver(SolverBase):
            class ObservableKind(SolverBase.ObservableKind):
                BODY_TEMPERATURE = "body_temperature"

            @dataclass(eq=False)
            class Observables(SolverBase.Observables):
                body_temperature: wp.array[float] | None = SolverBase.Observables.field(
                    dtype=float, frequency=newton.Model.AttributeFrequency.BODY
                )

            SUPPORTED_OBSERVABLES = frozenset({ObservableKind.BODY_QDD, ObservableKind.BODY_TEMPERATURE})

        solver = TemperatureSolver(self.model)
        kinds = solver.ObservableKind
        self.assertEqual(kinds.BODY_QDD, "body_qdd")

        class PressureKind(SolverBase.ObservableKind):
            BODY_PRESSURE = "body_pressure"

        class CombinedKind(kinds, PressureKind):
            BODY_TEMPERATURE = "body_temperature"
            _metadata = None

        self.assertEqual(CombinedKind.BODY_QDD, "body_qdd")
        self.assertEqual(CombinedKind.BODY_TEMPERATURE, "body_temperature")
        self.assertEqual(CombinedKind.BODY_PRESSURE, "body_pressure")
        self.assertFalse(hasattr(kinds, "BODY_PRESSURE"))
        self.assertFalse(hasattr(SolverBase.ObservableKind, "BODY_TEMPERATURE"))

        invalid_declarations = [
            ({"BODY_QDD": "other_acceleration"}, ValueError, "Conflicting.*BODY_QDD.*body_qdd.*other_acceleration"),
            (
                {"BODY_TEMPERATURE": "other_temperature"},
                ValueError,
                "Conflicting.*BODY_TEMPERATURE.*body_temperature.*other_temperature",
            ),
            ({"ACCELERATION": "body_qdd"}, ValueError, "Duplicate.*body_qdd.*BODY_QDD.*ACCELERATION"),
            (
                {"TEMPERATURE": "body_temperature"},
                ValueError,
                "Duplicate.*body_temperature.*BODY_TEMPERATURE.*TEMPERATURE",
            ),
            (
                {"BODY_PRESSURE": "body_pressure", "PRESSURE": "body_pressure"},
                ValueError,
                "Duplicate.*body_pressure.*BODY_PRESSURE.*PRESSURE",
            ),
        ]
        for declarations, error, message in invalid_declarations:
            with self.subTest(declarations=declarations), self.assertRaisesRegex(error, message):
                type("InvalidKind", (kinds,), declarations)
        for value in ("", None, 1, []):
            with self.subTest(value=value), self.assertRaisesRegex(TypeError, "BODY_PRESSURE.*nonempty string"):
                type("InvalidKind", (kinds,), {"BODY_PRESSURE": value})
        for value in ("", 0):
            with self.subTest(field_kind=value), self.assertRaisesRegex(TypeError, "nonempty strings"):
                SolverBase.Observables.field(kind=value, dtype=float, frequency=newton.Model.AttributeFrequency.BODY)

        class DifferentTemperatureKind(SolverBase.ObservableKind):
            BODY_TEMPERATURE = "other_temperature"

        class AliasedTemperatureKind(SolverBase.ObservableKind):
            TEMPERATURE = "body_temperature"

        for other, message in (
            (DifferentTemperatureKind, "Conflicting.*BODY_TEMPERATURE"),
            (AliasedTemperatureKind, "Duplicate.*body_temperature"),
        ):
            for bases in ((kinds, other), (other, kinds)):
                with self.subTest(bases=bases), self.assertRaisesRegex(ValueError, message):
                    type("ConflictingKind", bases, {"BODY_TEMPERATURE": "body_temperature"})

        for request in ({}, {"kinds": None}):
            with self.subTest(request=request), self.assertRaises(TypeError):
                solver.observables(**request)
        observables = solver.observables(solver.supported_observables, requires_grad=True)
        self.assertEqual(observables.kinds, {"body_qdd", "body_temperature"})
        self.assertEqual(observables.body_qdd.shape, (self.model.body_count,))
        self.assertEqual(observables.body_temperature.shape, (self.model.body_count,))
        self.assertIsNotNone(observables.body_temperature.grad)
        self.assertIsNone(observables.body_parent_f)
        self.assertIsNone(observables.contact_f)
        literal = solver.observables(kinds={"body_temperature"})
        self.assertIs(type(literal), TemperatureSolver.Observables)
        selected = observables.select({"body_temperature"})
        self.assertIs(type(selected), TemperatureSolver.Observables)
        self.assertIs(selected.body_temperature, observables.body_temperature)
        self.assertIsNone(selected.body_qdd)
        with self.assertRaisesRegex(TypeError, "collection of strings"):
            observables.select("body_temperature")
        with self.assertRaisesRegex(ValueError, "unknown_temperature"):
            solver.observables(kinds={"unknown_temperature"})

        @dataclass(eq=False)
        class DuplicateFields(TemperatureSolver.Observables):
            alias: wp.array[float] | None = SolverBase.Observables.field(
                kind="body_temperature", dtype=float, frequency=newton.Model.AttributeFrequency.BODY
            )

        solver.Observables = DuplicateFields
        with (
            patch("newton._src.solvers.solver.wp.zeros", side_effect=AssertionError("unexpected allocation")),
            self.assertRaisesRegex(ValueError, "Duplicate.*body_temperature"),
        ):
            solver.observables(kinds={"body_temperature"})

    @staticmethod
    def make_solver(model):
        """Declare custom fields using only the public extension API."""

        @dataclass(eq=False)
        class ThermalObservables(SolverBase.Observables):
            temperature: wp.array[float] | None = SolverBase.Observables.field(
                kind=CustomKind.TEMPERATURE, dtype=float, frequency=newton.Model.AttributeFrequency.BODY
            )

        @dataclass(eq=False)
        class ContactObservables(ThermalObservables):
            pressure: wp.array[float] | None = SolverBase.Observables.field(
                kind=CustomKind.PRESSURE, dtype=wp.float32, frequency=newton.Model.AttributeFrequency.CONTACT_RIGID
            )

        class Solver(SolverBase):
            Observables = ContactObservables
            SUPPORTED_OBSERVABLES = frozenset(
                {CustomKind.TEMPERATURE, CustomKind.PRESSURE, SolverBase.ObservableKind.BODY_QDD}
            )

        return Solver(model)

    def test_inherited_fields_and_explicit_kinds(self):
        """Allocate inherited scalar and spatial fields without name-based kinds."""
        solver = self.make_solver(self.model)
        kinds = {CustomKind.TEMPERATURE, SolverBase.ObservableKind.BODY_QDD}
        observables = solver.observables(kinds=kinds, requires_grad=True)
        self.assertEqual(observables.temperature.shape, (1,))
        self.assertIs(observables.temperature.dtype, wp.float32)
        self.assertIs(observables.body_qdd.dtype, wp.spatial_vector)
        self.assertEqual(observables.get_attribute_frequency("temperature"), newton.Model.AttributeFrequency.BODY)
        self.assertIsNone(observables.pressure)
        self.assertIsNone(observables.body_parent_f)
        self.assertIsNone(self.model.rigid_contact_max)
        self.assertIsNotNone(observables.temperature.grad)
        self.assertIsNotNone(observables.body_qdd.grad)
        self.assertEqual(observables.kinds, kinds)
        self.assertEqual({observables: "identity"}[observables], "identity")
        self.assertNotEqual(observables, solver.observables(kinds=kinds))

        with patch("newton._src.solvers.solver.wp.zeros", side_effect=AssertionError("unexpected allocation")):
            selected = observables.select({CustomKind.TEMPERATURE})
            body_only = observables.select({SolverBase.ObservableKind.BODY_QDD})
        self.assertIs(selected.temperature, observables.temperature)
        self.assertIs(selected.temperature.grad, observables.temperature.grad)
        self.assertIsNone(selected.body_qdd)
        self.assertIsNone(body_only.temperature)
        self.assertTrue(selected.is_requested(CustomKind.TEMPERATURE))
        self.assertIsNone(selected.select(set()).temperature)

    def test_custom_contact_field_binds_storage(self):
        """Apply contact rules using the declaration rather than the kind value."""
        solver = self.make_solver(self.model)
        with self.assertRaisesRegex(RuntimeError, "CollisionPipeline"):
            solver.observables(kinds={CustomKind.PRESSURE})
        pipeline = newton.CollisionPipeline(self.model, rigid_contact_max=0, soft_contact_max=0)
        contacts = pipeline.contacts()
        observables = solver.observables(contacts=contacts, kinds={CustomKind.PRESSURE})
        selected = observables.select({CustomKind.PRESSURE})
        self.assertEqual(selected.pressure.shape, (0,))
        self.assertTrue(selected.is_requested(CustomKind.PRESSURE))
        solver.validate_observables(selected, contacts)
        self.assertIs(observables.contacts, contacts)
        self.assertIsNone(observables.select(set()).contacts)

    def test_redeclarations_and_namespaces_are_independent(self):
        """Keep cached base declarations intact when a child overrides a field."""
        solver = self.make_solver(self.model)
        original = solver.observables(kinds={CustomKind.TEMPERATURE})

        class OtherKind:
            TENSOR = "tensor"

        @dataclass(eq=False)
        class DerivedObservables(solver.Observables):
            temperature: wp.array[wp.vec3] | None = SolverBase.Observables.field(
                kind=CustomKind.TEMPERATURE, dtype=wp.vec3, frequency=newton.Model.AttributeFrequency.PARTICLE
            )
            tensor: wp.array[wp.mat33] | None = SolverBase.Observables.field(
                kind=OtherKind.TENSOR, dtype=wp.mat33, frequency=newton.Model.AttributeFrequency.ONCE
            )

        class DerivedSolver(type(solver)):
            Observables = DerivedObservables
            SUPPORTED_OBSERVABLES = solver.SUPPORTED_OBSERVABLES | {OtherKind.TENSOR}

        derived = DerivedSolver(self.model).observables(kinds={CustomKind.TEMPERATURE, OtherKind.TENSOR})
        self.assertEqual(derived.temperature.shape, (0,))
        self.assertIs(derived.temperature.dtype, wp.vec3)
        self.assertEqual(derived.tensor.shape, (1,))
        self.assertIs(derived.tensor.dtype, wp.mat33)
        self.assertIsNone(derived.select({CustomKind.TEMPERATURE}).tensor)
        self.assertEqual(original.get_attribute_frequency("temperature"), newton.Model.AttributeFrequency.BODY)
        self.assertIs(solver.observables(kinds={CustomKind.TEMPERATURE}).temperature.dtype, wp.float32)

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
        class SampleObservables(SolverBase.Observables):
            temperature: wp.array[float] | None = SolverBase.Observables.field(
                kind=CustomKind.TEMPERATURE, dtype=float, frequency="sample"
            )

        class SampleSolver(SolverBase):
            Observables = SampleObservables
            SUPPORTED_OBSERVABLES = frozenset({CustomKind.TEMPERATURE})

        observables = SampleSolver(model).observables(kinds={CustomKind.TEMPERATURE})
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
            solver.observables(kinds={SolverBase.ObservableKind.BODY_PARENT_F})

    def test_require_identity_dataclasses(self):
        """Reject value equality that would make observable sources unhashable."""
        solver = self.make_solver(self.model)

        @dataclass
        class ValueObservables(solver.Observables):
            pass

        solver.Observables = ValueObservables
        with self.assertRaisesRegex(TypeError, "eq=False"):
            solver.observables(kinds={CustomKind.TEMPERATURE})

    def test_missing_dataclass_decorator(self):
        """Reject new field declarations that dataclasses have not processed."""
        solver = self.make_solver(self.model)

        class UndecoratedObservables(SolverBase.Observables):
            temperature: wp.array[float] | None = SolverBase.Observables.field(
                kind=CustomKind.TEMPERATURE, dtype=float, frequency=newton.Model.AttributeFrequency.BODY
            )

        solver.Observables = UndecoratedObservables
        with self.assertRaisesRegex(TypeError, "dataclass"):
            solver.observables(kinds={CustomKind.TEMPERATURE})

    def test_public_factory_override(self):
        """Customize allocation through the public factory without additional hooks."""
        solver_type = type(self.make_solver(self.model))
        calls = []

        class CustomSolver(solver_type):
            def observables(self, kinds, *, contacts=None, requires_grad=None):
                result = super().observables(kinds, contacts=contacts, requires_grad=requires_grad)
                calls.append(result.kinds)
                if result.is_requested(CustomKind.TEMPERATURE):
                    result.temperature.fill_(3.0)
                self.created = result
                return result

        solver = CustomSolver(self.model)
        observables = solver.observables(iter([CustomKind.TEMPERATURE]), requires_grad=True)
        self.assertEqual(calls, [frozenset({CustomKind.TEMPERATURE})])
        self.assertIs(solver.created, observables)
        self.assertIs(observables.model, self.model)
        self.assertTrue(observables.temperature.requires_grad)
        np.testing.assert_array_equal(observables.temperature.numpy(), [3.0])
        observables.select(set())
        self.assertEqual(calls, [frozenset({CustomKind.TEMPERATURE})])
        pipeline = newton.CollisionPipeline(self.model, rigid_contact_max=1, soft_contact_max=0)
        contacts = pipeline.contacts()
        all_observables = solver.observables(kinds=solver.supported_observables, contacts=contacts)
        self.assertEqual(all_observables.kinds, solver.supported_observables)
        self.assertEqual(calls[-1], solver.supported_observables)
        np.testing.assert_array_equal(all_observables.temperature.numpy(), [3.0])

    def test_kamino_factory_failure_can_be_retried(self):
        """Keep capacity and scratch storage uncommitted when backend setup fails."""
        pipeline = newton.CollisionPipeline(self.model, rigid_contact_max=1, soft_contact_max=0)
        contacts = pipeline.contacts()
        solver = object.__new__(newton.solvers.SolverKamino)
        SolverBase.__init__(solver, self.model)
        solver._collision_detector_kamino = None
        solver._contact_observable_state = None
        kinds = {SolverBase.ObservableKind.CONTACT_F}
        with (
            patch("newton._src.solvers.kamino.solver_kamino.wp.empty", side_effect=MemoryError("scratch allocation")),
            self.assertRaisesRegex(MemoryError, "scratch allocation"),
        ):
            solver.observables(contacts=contacts, kinds=kinds)
        self.assertIsNone(solver._contact_observable_state)
        pipeline = newton.CollisionPipeline(self.model, rigid_contact_max=2, soft_contact_max=0)
        contacts = pipeline.contacts()
        observables = solver.observables(contacts=contacts, kinds=kinds)
        self.assertEqual(observables.contact_f.shape, (2,))
        self.assertEqual(solver._contact_observable_state.body_q.shape, (self.model.body_count,))

    def test_allocation_failure_does_not_freeze_capacity(self):
        """Leave contact capacities mutable if array allocation fails."""
        solver = self.make_solver(self.model)
        pipeline = newton.CollisionPipeline(self.model, rigid_contact_max=1, soft_contact_max=0)
        contacts = pipeline.contacts()
        with (
            patch("newton._src.solvers.solver.wp.zeros", side_effect=MemoryError("array allocation")),
            self.assertRaisesRegex(MemoryError, "array allocation"),
        ):
            solver.observables(contacts=contacts, kinds={CustomKind.PRESSURE})
        pipeline = newton.CollisionPipeline(self.model, rigid_contact_max=2, soft_contact_max=0)
        contacts = pipeline.contacts()
        self.assertEqual(solver.observables(contacts=contacts, kinds={CustomKind.PRESSURE}).pressure.shape, (2,))

    def test_empty_request_does_not_allocate(self):
        """Return an owned empty container without allocating arrays or requiring contacts."""
        solver = self.make_solver(self.model)
        with patch("newton._src.solvers.solver.wp.zeros", side_effect=AssertionError("unexpected allocation")):
            observables = solver.observables(kinds=set())
            unsupported = SolverBase(self.model).observables(kinds=set())
        self.assertEqual(unsupported.kinds, frozenset())
        self.assertIs(unsupported.model, self.model)
        self.assertIs(observables.model, self.model)
        self.assertEqual(observables.kinds, frozenset())
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
                observables = solver.observables(kinds={CustomKind.TEMPERATURE, SolverBase.ObservableKind.BODY_QDD})
                selected = observables.select({CustomKind.TEMPERATURE})
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
