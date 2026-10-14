# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Test eager allocation, naming, and ownership of solver observables."""

from __future__ import annotations

import unittest
from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import warp as wp

import newton
from newton.selection import ArticulationView
from newton.sensors import SensorContact
from newton.solvers.experimental.coupled import SolverCoupled, SolverCoupledADMM
from newton.tests.unittest_utils import add_function_test, get_test_devices
from newton.viewer import ViewerNull


class ContactSolver(newton.solvers.SolverBase):
    """Exercise the base allocation contract without a numerical backend."""

    SUPPORTED_OBSERVABLES = frozenset(
        {
            newton.solvers.SolverBase.ObservableKind.BODY_QDD,
            newton.solvers.SolverBase.ObservableKind.BODY_PARENT_F,
            newton.solvers.SolverBase.ObservableKind.CONTACT_F,
        }
    )


class CustomContactKind:
    """Define a contact-indexed diagnostic independently of CONTACT_F."""

    PRESSURE = "pressure"


@dataclass(eq=False)
class CustomContactObservables(newton.solvers.SolverBase.Observables):
    """Keep custom contact pressure separate from the standard force array."""

    pressure: wp.array[float] | None = newton.solvers.SolverBase.Observables.field(
        kind=CustomContactKind.PRESSURE, dtype=float, frequency=newton.Model.AttributeFrequency.CONTACT_RIGID
    )


class CustomContactSolver(ContactSolver):
    """Exercise eager allocation of solver-specific contact fields."""

    Observables = CustomContactObservables
    SUPPORTED_OBSERVABLES = ContactSolver.SUPPORTED_OBSERVABLES | {CustomContactKind.PRESSURE}


class TestSolverObservables(unittest.TestCase):
    """Verify contact capacities are resolved before allocating observables."""

    def setUp(self):
        """Build an independent CPU model for each capacity test."""
        builder = newton.ModelBuilder()
        builder.add_body(mass=0.0)
        self.model = builder.finalize(device="cpu")
        self.solver = ContactSolver(self.model)
        self.kinds = {newton.solvers.SolverBase.ObservableKind.CONTACT_F}

    def test_contact_observables_size_from_contacts(self):
        """Size contact arrays from the supplied Contacts without a published model capacity."""
        self.assertIsNone(self.model.rigid_contact_max)
        self.assertIsNone(self.model.soft_contact_max)
        with self.assertRaisesRegex(ValueError, "Contacts"):
            self.solver.observables(kinds=self.kinds)
        with self.assertRaisesRegex(ValueError, "Contacts"):
            self.solver.observables(kinds=self.solver.supported_observables)
        contacts = newton.Contacts(5, 3, device="cpu")
        observables = self.solver.observables(kinds=self.kinds, contacts=contacts)
        self.assertEqual(observables.contact_f.shape, (8,))
        self.assertIs(observables.contacts, contacts)
        self.assertIsNone(self.model.rigid_contact_max)

    def test_body_only_observables_need_no_pipeline(self):
        """Allocate body observables without constructing a collision pipeline."""
        observables = self.solver.observables(kinds={newton.solvers.SolverBase.ObservableKind.BODY_QDD})
        self.assertEqual(observables.body_qdd.shape, (self.model.body_count,))
        self.assertIsNone(observables.contact_f)
        self.assertIsNone(observables.contacts)
        contacts = newton.Contacts(0, 0, device="cpu")
        for kinds in ({newton.solvers.SolverBase.ObservableKind.BODY_QDD}, set()):
            with self.subTest(kinds=kinds):
                allocated = self.solver.observables(kinds=kinds, contacts=contacts)
                self.assertIsNone(allocated.contacts)
                self.solver.validate_observables(allocated)

    def test_select_shares_arrays_without_allocating(self):
        """Select existing buffers without allocating, copying, or mutating the source."""
        kinds = newton.solvers.SolverBase.ObservableKind
        observables = self.solver.observables(kinds={kinds.BODY_QDD, kinds.BODY_PARENT_F}, requires_grad=True)
        with patch("newton._src.solvers.solver.wp.zeros", side_effect=AssertionError("allocated")):
            selected = observables.select({kinds.BODY_QDD})
        self.assertIsNot(selected, observables)
        self.assertIs(type(selected), type(observables))
        self.assertIs(selected.body_qdd, observables.body_qdd)
        self.assertIs(selected.body_qdd.grad, observables.body_qdd.grad)
        self.assertIsNone(selected.body_parent_f)
        self.assertIsNotNone(observables.body_parent_f)
        self.assertEqual(observables.kinds, {kinds.BODY_QDD, kinds.BODY_PARENT_F})
        self.assertEqual(selected.kinds, {kinds.BODY_QDD})
        self.assertTrue(selected.is_requested(kinds.BODY_QDD))
        self.assertFalse(selected.is_requested(kinds.BODY_PARENT_F))
        self.assertIs(selected.model, self.model)
        selected.body_qdd.fill_(wp.spatial_vector(7.0))
        np.testing.assert_array_equal(observables.body_qdd.numpy(), np.full((1, 6), 7.0))
        self.solver.validate_observables(selected)
        with self.assertRaisesRegex(ValueError, "solver instance"):
            ContactSolver(self.model).validate_observables(selected)

    def test_select_empty_and_nested_subsets(self):
        """Narrow selections without re-enabling fields excluded by their parent."""
        kinds = newton.solvers.SolverBase.ObservableKind
        observables = self.solver.observables(kinds={kinds.BODY_QDD, kinds.BODY_PARENT_F})
        selected = observables.select({kinds.BODY_QDD})
        self.assertIs(selected.select(selected.kinds).body_qdd, observables.body_qdd)
        empty = selected.select(set())
        self.assertEqual(empty.kinds, frozenset())
        self.assertIsNone(empty.body_qdd)
        self.assertIsNone(empty.body_parent_f)
        self.solver.validate_observables(empty)
        for source, requested in ((observables, {kinds.CONTACT_F}), (selected, {kinds.BODY_PARENT_F})):
            with self.subTest(requested=requested), self.assertRaisesRegex(ValueError, "not requested"):
                source.select(requested)
        with self.assertRaises(AttributeError):
            selected.kinds = frozenset()
        with self.assertRaisesRegex(ValueError, "allocated"):
            newton.solvers.SolverBase.Observables().select(set())

    def test_select_custom_fields_and_shared_contact_binding(self):
        """Share binding across custom contact subsets created before the first step."""
        kinds = newton.solvers.SolverBase.ObservableKind
        pipeline = newton.CollisionPipeline(self.model, rigid_contact_max=3, soft_contact_max=0)
        contacts = pipeline.contacts()
        solver = CustomContactSolver(self.model)
        observables = solver.observables(
            contacts=contacts, kinds={kinds.BODY_QDD, kinds.CONTACT_F, CustomContactKind.PRESSURE}
        )
        pressure = observables.select({CustomContactKind.PRESSURE})
        force = observables.select({kinds.CONTACT_F})
        body = observables.select({kinds.BODY_QDD})
        self.assertIs(type(pressure), CustomContactObservables)
        self.assertIs(pressure.pressure, observables.pressure)
        self.assertIsNone(pressure.contact_f)
        self.assertIsNone(force.pressure)
        self.assertEqual(pressure.get_attribute_frequency("pressure"), newton.Model.AttributeFrequency.CONTACT_RIGID)
        solver.validate_observables(body)
        solver.validate_observables(observables.select(set()))
        with self.assertRaisesRegex(ValueError, "Contacts"):
            solver.validate_observables(pressure)
        solver.validate_observables(pressure, contacts)
        for selection in (observables, force, force.select(force.kinds)):
            self.assertIs(selection.contacts, contacts)
        for selection in (observables, pressure):
            self.assertIs(selection.contacts_for_kind(CustomContactKind.PRESSURE), contacts)
        self.assertIsNone(pressure.contacts)
        self.assertIsNone(body.contacts)
        self.assertIsNone(body.contacts_for_kind(CustomContactKind.PRESSURE))
        self.assertIsNone(observables.contacts_for_kind(kinds.BODY_QDD))
        with self.assertRaisesRegex(KeyError, "No observable declared"):
            observables.contacts_for_kind("pressur")

    def test_selection_rebinds_only_written_arrays(self):
        """Track the contacts of each array so a subset step cannot relabel skipped arrays."""
        kinds = newton.solvers.SolverBase.ObservableKind
        pipeline = newton.CollisionPipeline(self.model, rigid_contact_max=3, soft_contact_max=0)
        contacts_a, contacts_b = pipeline.contacts(), pipeline.contacts()
        solver = CustomContactSolver(self.model)
        observables = solver.observables(contacts=contacts_a, kinds={kinds.CONTACT_F, CustomContactKind.PRESSURE})
        force = observables.select({kinds.CONTACT_F})
        pressure = observables.select({CustomContactKind.PRESSURE})
        solver.validate_observables(force, contacts_b)
        for selection in (observables, force):
            self.assertIs(selection.contacts, contacts_b)
        for selection in (observables, pressure):
            self.assertIs(selection.contacts_for_kind(CustomContactKind.PRESSURE), contacts_a)
        solver.validate_observables(observables, contacts_b)
        for selection in (observables, force):
            self.assertIs(selection.contacts, contacts_b)
        for selection in (observables, pressure):
            self.assertIs(selection.contacts_for_kind(CustomContactKind.PRESSURE), contacts_b)

    def test_select_zero_capacity_remains_requested(self):
        """Treat selected zero-length contact buffers as requested."""
        pipeline = newton.CollisionPipeline(self.model, rigid_contact_max=0, soft_contact_max=0)
        contacts = pipeline.contacts()
        with self.assertRaisesRegex(ValueError, "Contacts"):
            self.solver.observables(kinds=self.kinds)
        observables = self.solver.observables(contacts=contacts, kinds=self.kinds)
        selected = observables.select(self.kinds)
        self.assertTrue(selected.is_requested(newton.solvers.SolverBase.ObservableKind.CONTACT_F))
        self.assertIs(selected.contact_f, observables.contact_f)
        self.assertEqual(selected.contact_f.shape, (0,))

    def test_select_substep_schedule(self):
        """Update selected custom fields per substep and preserve skipped values."""
        kinds = newton.solvers.SolverBase.ObservableKind

        class SamplingSolver(CustomContactSolver):
            def step(self, state_in, state_out, control, contacts, dt, *, observables=None):
                self.validate_observables(observables, contacts)
                if observables is not None:
                    for kind in (kinds.BODY_QDD, CustomContactKind.PRESSURE):
                        if observables.is_requested(kind):
                            array = getattr(observables, kind)
                            array.fill_(dt if kind is CustomContactKind.PRESSURE else wp.spatial_vector(dt))

        pipeline = newton.CollisionPipeline(self.model, rigid_contact_max=2, soft_contact_max=0)
        contacts = pipeline.contacts()
        solver = SamplingSolver(self.model)
        observables = solver.observables(contacts=contacts, kinds={kinds.BODY_QDD, CustomContactKind.PRESSURE})
        every_substep = observables.select({kinds.BODY_QDD})
        observables.pressure.fill_(-1.0)
        for step in range(3):
            solver.step(None, None, None, contacts, step + 1, observables=every_substep)
            np.testing.assert_array_equal(observables.body_qdd.numpy(), np.full((1, 6), step + 1))
            np.testing.assert_array_equal(observables.pressure.numpy(), [-1.0, -1.0])
        solver.step(None, None, None, contacts, 4, observables=observables)
        np.testing.assert_array_equal(observables.pressure.numpy(), [4.0, 4.0])
        solver.step(None, None, None, contacts, 5, observables=observables.select(set()))
        solver.step(None, None, None, contacts, 6)
        np.testing.assert_array_equal(observables.body_qdd.numpy(), np.full((1, 6), 4.0))
        np.testing.assert_array_equal(observables.pressure.numpy(), [4.0, 4.0])

    def test_select_graph_substeps(self):
        """Capture a fixed schedule of shared subsets without changing skipped arrays."""
        if not wp.is_cuda_available():
            self.skipTest("CUDA graph capture requires a CUDA device")
        builder = newton.ModelBuilder()
        builder.add_body(mass=0.0)
        model = builder.finalize(device="cuda:0")
        solver = ContactSolver(model)
        kinds = newton.solvers.SolverBase.ObservableKind
        observables = solver.observables(kinds={kinds.BODY_QDD, kinds.BODY_PARENT_F})
        early = observables.select({kinds.BODY_QDD})
        last = observables.select({kinds.BODY_PARENT_F})
        pointer = observables.body_qdd.ptr
        with wp.ScopedCapture(device=model.device) as capture:
            for step in range(3):
                selected = last if step == 2 else early
                solver.validate_observables(selected)
                for kind in solver.supported_observables:
                    if selected.is_requested(kind):
                        getattr(selected, kind).fill_(wp.spatial_vector(step + 1.0))
        for _ in range(2):
            wp.capture_launch(capture.graph)
            np.testing.assert_array_equal(observables.body_qdd.numpy(), np.full((1, 6), 2.0))
            np.testing.assert_array_equal(observables.body_parent_f.numpy(), np.full((1, 6), 3.0))
        self.assertEqual(observables.body_qdd.ptr, pointer)

    def test_select_conditional_graph(self):
        """Preserve skipped custom fields and share contact binding through a conditional graph."""
        if not wp.is_cuda_available() or not wp.is_conditional_graph_supported():
            self.skipTest("Conditional CUDA graphs are unavailable")
        model = newton.ModelBuilder().finalize(device="cuda:0")
        pipeline = newton.CollisionPipeline(model, rigid_contact_max=2, soft_contact_max=0)
        contacts = pipeline.contacts()
        solver = CustomContactSolver(model)
        kinds = newton.solvers.SolverBase.ObservableKind
        observables = solver.observables(contacts=contacts, kinds={kinds.CONTACT_F, CustomContactKind.PRESSURE})
        selected = observables.select({CustomContactKind.PRESSURE})
        condition = wp.ones(1, dtype=wp.int32, device=model.device)
        observables.contact_f.fill_(wp.spatial_vector(-1.0))

        def body():
            solver.validate_observables(selected, contacts)
            if selected.is_requested(CustomContactKind.PRESSURE):
                selected.pressure.fill_(3.0)

        with wp.ScopedCapture(device=model.device) as capture:
            wp.capture_if(condition, body)
        for enabled in (0, 1, 0):
            condition.fill_(enabled)
            observables.pressure.zero_()
            wp.capture_launch(capture.graph)
            np.testing.assert_array_equal(observables.pressure.numpy(), np.full(2, 3.0 * enabled))
            np.testing.assert_array_equal(observables.contact_f.numpy(), np.full((2, 6), -1.0))
        self.assertIs(observables.contacts, contacts)

    def test_observable_frequency_metadata(self):
        """Inherit standard row frequencies without another capability kind set."""
        observables = CustomContactObservables()
        frequency = newton.Model.AttributeFrequency
        self.assertEqual(observables.get_attribute_frequency("body_qdd"), frequency.BODY)
        self.assertEqual(observables.get_attribute_frequency("pressure"), frequency.CONTACT_RIGID)
        self.assertEqual(observables.get_attribute_frequency("contact_f"), frequency.CONTACT)
        self.assertFalse(hasattr(ContactSolver, "CONTACT_OBSERVABLE_FLAGS"))

    def test_missing_observable_declaration(self):
        """Reject custom requests without a field declaration before allocating."""
        solver = CustomContactSolver(self.model)
        solver.Observables = newton.solvers.SolverBase.Observables
        with self.assertRaisesRegex(ValueError, "field.*pressure"):
            solver.observables(kinds={CustomContactKind.PRESSURE})

    def test_invalid_frequency_declaration(self):
        """Reject integer frequency values rather than guessing their domain."""

        with self.assertRaisesRegex(TypeError, "Invalid observable frequency"):
            newton.solvers.SolverBase.Observables.field(kind=CustomContactKind.PRESSURE, dtype=float, frequency=5)

    def test_contact_frequency_counts(self):
        """Resolve each contact domain from capacity rather than live counts."""
        frequency = newton.Model.AttributeFrequency
        for domain in (frequency.CONTACT, frequency.CONTACT_RIGID, frequency.CONTACT_SOFT):
            with self.subTest(domain=domain):
                count_attr = self.model._ATTRIBUTE_FREQUENCY_COUNT_ATTRS[domain]
                self.assertIsNone(getattr(self.model, count_attr))
                with self.assertRaisesRegex(RuntimeError, "CollisionPipeline"):
                    self.model._attribute_frequency_count(domain)

        for rigid_max, soft_max in ((5, 3), (0, 3), (5, 0), (0, 0)):
            newton.CollisionPipeline(self.model, rigid_contact_max=rigid_max, soft_contact_max=soft_max)
            for domain, count in (
                (frequency.CONTACT_RIGID, rigid_max),
                (frequency.CONTACT_SOFT, soft_max),
                (frequency.CONTACT, rigid_max + soft_max),
            ):
                with self.subTest(domain=domain, rigid_max=rigid_max, soft_max=soft_max):
                    count_attr = self.model._ATTRIBUTE_FREQUENCY_COUNT_ATTRS[domain]
                    self.assertEqual(getattr(self.model, count_attr), count)
                    self.assertEqual(self.model._attribute_frequency_count(domain), count)

    def test_custom_soft_contact_frequency(self):
        """Require contact binding for a soft-only diagnostic without standard forces."""

        @dataclass(eq=False)
        class SoftObservables(CustomContactObservables):
            pressure: wp.array[float] | None = newton.solvers.SolverBase.Observables.field(
                kind=CustomContactKind.PRESSURE, dtype=float, frequency=newton.Model.AttributeFrequency.CONTACT_SOFT
            )

        class SoftSolver(CustomContactSolver):
            Observables = SoftObservables

        solver = SoftSolver(self.model)
        with self.assertRaisesRegex(ValueError, "Contacts"):
            solver.observables(kinds={CustomContactKind.PRESSURE})
        pipeline = newton.CollisionPipeline(self.model, rigid_contact_max=5, soft_contact_max=3)
        contacts = pipeline.contacts()
        observables = solver.observables(contacts=contacts, kinds={CustomContactKind.PRESSURE})
        self.assertEqual(observables.pressure.shape, (3,))
        self.assertIsNone(observables.contact_f)
        with self.assertRaisesRegex(ValueError, "Contacts"):
            solver.validate_observables(observables)
        solver.validate_observables(observables, contacts)

    def test_selection_uses_observable_frequencies(self):
        """Select custom body and DOF arrays without registering fields on the model."""
        builder = newton.ModelBuilder()
        for index in range(3):
            body = builder.add_link(label=f"robot_{index}/body")
            joint = builder.add_joint_free(child=body, label=f"robot_{index}/joint")
            builder.add_articulation([joint], label=f"robot_{index}")
        model = builder.finalize(device="cpu")
        view = ArticulationView(model, [0, 2])
        for frequency, width in (
            (newton.Model.AttributeFrequency.BODY, 1),
            (newton.Model.AttributeFrequency.JOINT_DOF, 6),
        ):
            with self.subTest(frequency=frequency):

                class Solver(CustomContactSolver):
                    @dataclass(eq=False)
                    class Observables(CustomContactObservables):
                        pressure: wp.array[float] | None = newton.solvers.SolverBase.Observables.field(
                            kind=CustomContactKind.PRESSURE, dtype=float, frequency=frequency
                        )

                observables = Solver(model).observables(kinds={CustomContactKind.PRESSURE})
                values = np.arange(3 * width, dtype=np.float32)
                observables.pressure.assign(values)
                with self.assertRaises(KeyError):
                    model.get_attribute_frequency("pressure")
                np.testing.assert_array_equal(
                    view.get_attribute("pressure", observables).numpy(), values.reshape(1, 3, width)[:, [0, 2]]
                )
                with self.assertRaisesRegex(ValueError, "not requested"):
                    view.get_attribute("body_qdd", observables)
                with self.assertRaisesRegex(ValueError, "same model"):
                    view.get_attribute(
                        "body_qdd", self.solver.observables(kinds={newton.solvers.SolverBase.ObservableKind.BODY_QDD})
                    )

        pipeline = newton.CollisionPipeline(model, rigid_contact_max=2, soft_contact_max=1)
        contacts = pipeline.contacts()
        observables = ContactSolver(model).observables(contacts=contacts, kinds=self.kinds)
        with self.assertRaisesRegex(AttributeError, "dynamic contact"):
            view.get_attribute("contact_f", observables)
        self.assertEqual(observables.contact_f.shape, (pipeline.rigid_contact_max + pipeline.soft_contact_max,))

    def test_eager_capacity_allocation(self):
        """Allocate full rigid and soft capacity and bind Contacts immediately."""
        pipeline = newton.CollisionPipeline(self.model, rigid_contact_max=5, soft_contact_max=3)
        contacts = pipeline.contacts()
        observables = self.solver.observables(contacts=contacts, kinds=self.kinds, requires_grad=True)
        self.assertEqual((self.model.rigid_contact_max, self.model.soft_contact_max), (5, 3))
        self.assertEqual(observables.contact_f.shape, (8,))
        self.assertTrue(observables.contact_f.requires_grad)
        self.assertIs(observables.contacts, contacts)
        pointer = observables.contact_f.ptr
        self.solver.validate_observables(observables, contacts)
        self.assertIs(observables.contacts, contacts)
        self.assertEqual(observables.contact_f.ptr, pointer)

    def test_zero_capacity_is_requested(self):
        """Keep a requested empty array distinct from an unrequested field."""
        pipeline = newton.CollisionPipeline(self.model, rigid_contact_max=0, soft_contact_max=0)
        contacts = pipeline.contacts()
        observables = self.solver.observables(contacts=contacts, kinds=self.kinds)
        self.assertIsNotNone(observables.contact_f)
        self.assertEqual(observables.contact_f.shape, (0,))
        self.assertIsNone(self.solver.observables(kinds=set()).contact_f)

    def test_validate_contact_type_device_and_capacity(self):
        """Reject unusable contacts and accept any buffer that fits the allocated rows."""
        with self.assertRaisesRegex(ValueError, "Contacts"):
            self.solver.observables(kinds=self.kinds)
        with self.assertRaisesRegex(TypeError, "Contacts"):
            self.solver.observables(kinds=self.kinds, contacts=object())
        if wp.is_cuda_available():
            with self.assertRaisesRegex(ValueError, "device"):
                self.solver.observables(kinds=self.kinds, contacts=newton.Contacts(5, 3, device="cuda:0"))
        contacts = newton.Contacts(5, 3, device="cpu")
        observables = self.solver.observables(kinds=self.kinds, contacts=contacts)
        pointer = observables.contact_f.ptr
        with self.assertRaisesRegex(ValueError, "Contacts"):
            self.solver.validate_observables(observables)
        with self.assertRaisesRegex(TypeError, "Contacts"):
            self.solver.validate_observables(observables, object())
        if wp.is_cuda_available():
            with self.assertRaisesRegex(ValueError, "device"):
                self.solver.validate_observables(observables, newton.Contacts(5, 3, device="cuda:0"))
        for rigid_max, soft_max in ((9, 0), (5, 4), (0, 9)):
            with self.subTest(rigid_max=rigid_max, soft_max=soft_max), self.assertRaisesRegex(ValueError, "capacit"):
                self.solver.validate_observables(observables, newton.Contacts(rigid_max, soft_max, device="cpu"))
            self.assertIs(observables.contacts, contacts)
        # Larger arrays are accepted; rows follow the layout of the validated contacts.
        for rigid_max, soft_max in ((5, 3), (8, 0), (2, 1), (0, 0)):
            with self.subTest(rigid_max=rigid_max, soft_max=soft_max):
                smaller = newton.Contacts(rigid_max, soft_max, device="cpu")
                self.solver.validate_observables(observables, smaller)
                self.assertIs(observables.contacts, smaller)
                self.assertEqual(observables.contact_f.ptr, pointer)

    def test_validate_capacity_per_row_domain(self):
        """Check each requested array against the row domain it indexes."""
        solver = CustomContactSolver(self.model)
        kinds = {newton.solvers.SolverBase.ObservableKind.CONTACT_F, CustomContactKind.PRESSURE}
        observables = solver.observables(kinds=kinds, contacts=newton.Contacts(5, 3, device="cpu"))
        self.assertEqual(observables.pressure.shape, (5,))
        # The packed force array has room for 6 rigid rows, but the rigid pressure array does not.
        with self.assertRaisesRegex(ValueError, "pressure"):
            solver.validate_observables(observables, newton.Contacts(6, 0, device="cpu"))
        force = observables.select({newton.solvers.SolverBase.ObservableKind.CONTACT_F})
        solver.validate_observables(force, newton.Contacts(6, 0, device="cpu"))

    def test_pipeline_capacity_changes_keep_observables(self):
        """Let other pipelines publish different capacities without touching allocated arrays."""
        pipeline = newton.CollisionPipeline(self.model, rigid_contact_max=5, soft_contact_max=3)
        contacts = pipeline.contacts()
        observables = self.solver.observables(contacts=contacts, kinds=self.kinds)
        pointer = observables.contact_f.ptr
        newton.CollisionPipeline(self.model, rigid_contact_max=6, soft_contact_max=3)
        self.assertEqual((self.model.rigid_contact_max, self.model.soft_contact_max), (6, 3))
        self.model.rigid_contact_max = 1
        self.model.soft_contact_max = 0
        self.solver.validate_observables(observables, contacts)
        self.assertEqual(observables.contact_f.shape, (8,))
        self.assertEqual(observables.contact_f.ptr, pointer)
        self.assertIs(observables.contacts, contacts)

    def test_failed_pipeline_does_not_publish_capacity(self):
        """Leave capacities uninitialized when pipeline construction fails."""
        with self.assertRaises(ValueError):
            newton.CollisionPipeline(self.model, rigid_contact_max=5, max_triangle_pairs=0)
        self.assertIsNone(self.model.rigid_contact_max)
        self.assertIsNone(self.model.soft_contact_max)

    def test_late_pipeline_failure_preserves_published_capacity(self):
        """Publish both capacities only after every pipeline allocation succeeds."""
        pipeline = newton.CollisionPipeline(self.model, rigid_contact_max=5, soft_contact_max=3)
        contacts = pipeline.contacts()
        with patch("newton._src.sim.collide.ContactSorter", side_effect=RuntimeError("sorter allocation failed")):
            with self.assertRaisesRegex(RuntimeError, "sorter allocation failed"):
                newton.CollisionPipeline(self.model, rigid_contact_max=6, soft_contact_max=4, deterministic=True)
        self.assertEqual((self.model.rigid_contact_max, self.model.soft_contact_max), (5, 3))
        self.assertEqual(self.solver.observables(contacts=contacts, kinds=self.kinds).contact_f.shape, (8,))

    def test_private_pipeline_does_not_publish_capacity(self):
        """Keep the model defaults unchanged when a pipeline opts out of publishing."""
        private = newton.CollisionPipeline(self.model, rigid_contact_max=4, soft_contact_max=2, publish_capacity=False)
        self.assertIsNone(self.model.rigid_contact_max)
        self.assertIsNone(self.model.soft_contact_max)
        newton.CollisionPipeline(self.model, rigid_contact_max=7, soft_contact_max=1)
        newton.CollisionPipeline(self.model, rigid_contact_max=3, soft_contact_max=0, publish_capacity=False)
        self.assertEqual((self.model.rigid_contact_max, self.model.soft_contact_max), (7, 1))
        observables = self.solver.observables(kinds=self.kinds, contacts=private.contacts())
        self.assertEqual(observables.contact_f.shape, (6,))

    def test_alternate_compatible_contacts(self):
        """Alternate buffers from independent pipelines with one observable container."""
        pipeline = newton.CollisionPipeline(self.model, rigid_contact_max=5, soft_contact_max=3)
        contacts = pipeline.contacts()
        observables = self.solver.observables(contacts=contacts, kinds=self.kinds)
        matching = newton.CollisionPipeline(self.model, rigid_contact_max=5, soft_contact_max=3).contacts()
        for buffer in (contacts, matching, contacts):
            self.solver.validate_observables(observables, buffer)
            self.assertIs(observables.contacts, buffer)
        self.assertEqual(observables.contact_f.shape, (8,))

    def test_contact_count_does_not_resize_output(self):
        """Keep allocation stable as collision changes the active contact count."""
        builder = newton.ModelBuilder()
        builder.add_ground_plane()
        builder.add_particle(pos=(0, 0, 1), vel=(0, 0, 0), mass=1.0, radius=0.1)
        model = builder.finalize(device="cpu")
        pipeline = newton.CollisionPipeline(model, rigid_contact_max=0)
        contacts = pipeline.contacts()
        observables = ContactSolver(model).observables(contacts=contacts, kinds=self.kinds)
        pointer = observables.contact_f.ptr
        state = model.state()
        pipeline.collide(state, contacts)
        self.assertEqual(contacts.soft_contact_count.numpy()[0], 0)
        state.particle_q.assign([[0, 0, 0.05]])
        pipeline.collide(state, contacts)
        self.assertEqual(contacts.soft_contact_count.numpy()[0], 1)
        self.assertEqual(observables.contact_f.shape, (pipeline.soft_contact_max,))
        self.assertEqual(observables.contact_f.ptr, pointer)

    def test_model_capacity_does_not_size_observables(self):
        """Ignore model-published capacities when sizing contact arrays."""
        self.model.rigid_contact_max = 5
        self.model.soft_contact_max = 3
        with self.assertRaisesRegex(ValueError, "Contacts"):
            self.solver.observables(kinds=self.kinds)
        observables = self.solver.observables(kinds=self.kinds, contacts=newton.Contacts(2, 1, device="cpu"))
        self.assertEqual(observables.contact_f.shape, (3,))

    def test_owned_pipeline_contacts_size_observables(self):
        """Default to the solver-owned contacts when no buffer is supplied."""

        class OwningSolver(ContactSolver):
            supports_collision_pipeline = True

        pipeline = newton.CollisionPipeline(self.model, rigid_contact_max=4, soft_contact_max=2)
        solver = OwningSolver(self.model, collision_pipeline=pipeline)
        observables = solver.observables(kinds=self.kinds)
        self.assertIs(observables.contacts, solver.contacts)
        self.assertEqual(observables.contact_f.shape, (6,))
        solver.validate_observables(observables)
        external = newton.Contacts(1, 1, device="cpu")
        solver.validate_observables(observables, external)
        self.assertIs(observables.contacts, external)
        solver.validate_observables(observables)
        self.assertIs(observables.contacts, solver.contacts)

    def test_custom_contact_output_without_contact_force(self):
        """Size and bind custom contact kinds independently of the standard force array."""
        solver = CustomContactSolver(self.model)
        kinds = {CustomContactKind.PRESSURE}
        with self.assertRaisesRegex(ValueError, "Contacts"):
            solver.observables(kinds=kinds)
        pipeline = newton.CollisionPipeline(self.model, rigid_contact_max=4, soft_contact_max=0)
        contacts = pipeline.contacts()
        observables = solver.observables(contacts=contacts, kinds=kinds)
        self.assertEqual(observables.pressure.shape, (4,))
        self.assertIsNone(observables.contact_f)
        solver.validate_observables(observables, contacts)
        # Soft capacity is irrelevant to a rigid-only array.
        solver.validate_observables(observables, newton.Contacts(4, 100, device="cpu"))

    def test_native_backend_rejects_insufficient_capacity(self):
        """Reject native budgets that exceed the supplied contacts' rigid capacity."""
        pipeline = newton.CollisionPipeline(self.model, rigid_contact_max=2, soft_contact_max=0)
        contacts = pipeline.contacts()
        mujoco = object.__new__(newton.solvers.SolverMuJoCo)
        newton.solvers.SolverBase.__init__(mujoco, self.model)
        mujoco.use_mujoco_cpu = False
        mujoco.mjw_model = SimpleNamespace(opt=SimpleNamespace(run_collision_detection=True))
        mujoco.mjw_data = SimpleNamespace(naconmax=3)
        kamino = object.__new__(newton.solvers.SolverKamino)
        newton.solvers.SolverBase.__init__(kamino, self.model)
        kamino._contact_observable_state = None
        kamino._collision_detector_kamino = object()
        kamino._contacts_kamino = SimpleNamespace(model_max_contacts_host=3)
        for solver in (mujoco, kamino):
            with self.subTest(solver=type(solver).__name__):
                with self.assertRaisesRegex(ValueError, "exceeds Contacts capacity"):
                    solver.observables(contacts=contacts, kinds=self.kinds)
        # A larger buffer from another pipeline satisfies the budget.
        pipeline = newton.CollisionPipeline(self.model, rigid_contact_max=3, soft_contact_max=0)
        large = pipeline.contacts()
        for solver in (mujoco, kamino):
            with self.subTest(solver=type(solver).__name__):
                observables = solver.observables(contacts=large, kinds=self.kinds)
                self.assertEqual(observables.contact_f.shape, (3,))
                # Steps reject a smaller buffer before launching work or recording it.
                with self.assertRaisesRegex(ValueError, "exceeds Contacts capacity"):
                    solver.validate_observables(observables, contacts)
                self.assertIs(observables.contacts, large)
                solver.validate_observables(observables.select(set()), contacts)

    def test_graph_reuses_preallocated_output(self):
        """Reuse a preallocated output across CUDA graph replays."""
        if not wp.is_cuda_available():
            self.skipTest("CUDA graph capture requires a CUDA device")
        model = newton.ModelBuilder().finalize(device="cuda:0")
        pipeline = newton.CollisionPipeline(model, rigid_contact_max=3, soft_contact_max=0)
        contacts = pipeline.contacts()
        solver = ContactSolver(model)
        observables = solver.observables(contacts=contacts, kinds=self.kinds)
        pointer = observables.contact_f.ptr
        with wp.ScopedCapture(device=model.device) as capture:
            solver.validate_observables(observables, contacts)
            observables.contact_f.zero_()
        for _ in range(2):
            wp.capture_launch(capture.graph)
        self.assertEqual(observables.contact_f.ptr, pointer)
        self.assertEqual(observables.contact_f.numpy().sum(), 0.0)

    def test_conditional_graph_uses_preallocated_output(self):
        """Use contact observables in a conditional graph without allocating there."""
        if not wp.is_cuda_available() or not wp.is_conditional_graph_supported():
            self.skipTest("Conditional CUDA graphs are unavailable")
        model = newton.ModelBuilder().finalize(device="cuda:0")
        pipeline = newton.CollisionPipeline(model, rigid_contact_max=3, soft_contact_max=0)
        contacts = pipeline.contacts()
        solver = ContactSolver(model)
        observables = solver.observables(contacts=contacts, kinds=self.kinds)
        condition = wp.ones(1, dtype=wp.int32, device=model.device)

        def body():
            solver.validate_observables(observables, contacts)
            observables.contact_f.zero_()

        pointer = observables.contact_f.ptr
        with wp.ScopedCapture(device=model.device) as capture:
            wp.capture_if(condition, body)
        for _ in range(2):
            observables.contact_f.fill_(wp.spatial_vector(1.0))
            wp.capture_launch(capture.graph)
            self.assertEqual(observables.contact_f.numpy().sum(), 0.0)
        self.assertEqual(observables.contact_f.ptr, pointer)


def _make_admm_scene(device):
    """Build two touching bodies for an ADMM private contact pipeline."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    a = builder.add_body()
    builder.add_shape_box(a, hx=0.1, hy=0.1, hz=0.1)
    b = builder.add_body(xform=wp.transform(wp.vec3(0.15, 0.0, 0.0), wp.quat_identity()))
    builder.add_shape_sphere(b, radius=0.1)
    return builder.finalize(device=device), a, b


def _make_admm(model, a, b):
    """Couple two XPBD entries through ADMM rigid-rigid contacts."""
    return SolverCoupledADMM(
        model,
        [
            SolverCoupled.Entry(name="a", solver=newton.solvers.SolverXPBD, bodies=[a]),
            SolverCoupled.Entry(name="b", solver=newton.solvers.SolverXPBD, bodies=[b]),
        ],
        SolverCoupledADMM.Config(contact_pairs=[SolverCoupledADMM.ContactPair(source="a", destination="b")]),
    )


def test_external_and_private_admm_pipelines(test, device):
    """Keep an external pipeline's observables valid around ADMM's private pipeline in either order."""
    kinds = {newton.solvers.SolverBase.ObservableKind.CONTACT_F}
    for observables_first in (True, False):
        with test.subTest(observables_first=observables_first):
            model, a, b = _make_admm_scene(device)
            pipeline = newton.CollisionPipeline(model, rigid_contact_max=1000, soft_contact_max=0)
            contacts = pipeline.contacts()
            solver = newton.solvers.SolverXPBD(model)
            if observables_first:
                observables = solver.observables(kinds, contacts=contacts)
            admm = _make_admm(model, a, b)
            if not observables_first:
                observables = solver.observables(kinds, contacts=contacts)
            test.assertEqual(admm._admm_collision_pipeline.rigid_contact_max, 8)
            test.assertEqual((model.rigid_contact_max, model.soft_contact_max), (1000, 0))
            test.assertEqual(observables.contact_f.shape, (1000,))
            state_0, state_1 = model.state(), model.state()
            pipeline.collide(state_0, contacts)
            solver.step(state_0, state_1, model.control(), contacts, 0.001, observables=observables)
            test.assertIs(observables.contacts, contacts)
            admm.step(state_0, state_1, model.control(), None, 0.001)

    model, a, b = _make_admm_scene(device)
    _make_admm(model, a, b)
    test.assertIsNone(model.rigid_contact_max)
    test.assertGreater(newton.CollisionPipeline(model).rigid_contact_max, 8)


def _make_resting_box(device):
    """Build a box resting on the ground so XPBD produces rigid contact forces."""
    builder = newton.ModelBuilder()
    builder.add_ground_plane()
    body = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.099), wp.quat_identity()))
    builder.add_shape_box(body, hx=0.1, hy=0.1, hz=0.1, label="box")
    return builder.finalize(device=device)


def test_alternating_contacts_production_and_consumption(test, device):
    """Produce and consume forces through selections while alternating compatible buffers."""
    kinds = newton.solvers.SolverBase.ObservableKind
    model = _make_resting_box(device)
    large_pipeline = newton.CollisionPipeline(model, rigid_contact_max=64, soft_contact_max=0)
    small_pipeline = newton.CollisionPipeline(model, rigid_contact_max=16, soft_contact_max=0)
    contacts_large = large_pipeline.contacts()
    contacts_small = small_pipeline.contacts()
    solver = newton.solvers.SolverXPBD(model)
    observables = solver.observables({kinds.CONTACT_F, kinds.BODY_PARENT_F}, contacts=contacts_large)
    force = observables.select({kinds.CONTACT_F})
    body = observables.select({kinds.BODY_PARENT_F})
    sensor = SensorContact(model, sensing_shapes="box", request_contact_attributes=False)
    viewer = ViewerNull()
    viewer.set_model(model)
    viewer.show_contacts = True
    state_0, state_1 = model.state(), model.state()
    schedule = (
        (large_pipeline, contacts_large, contacts_small),
        (small_pipeline, contacts_small, contacts_large),
        (large_pipeline, contacts_large, contacts_small),
    )
    for pipeline, contacts, other in schedule:
        with test.subTest(rigid_contact_max=contacts.rigid_contact_max):
            pipeline.collide(state_0, contacts)
            solver.step(state_0, state_1, None, contacts, 1.0 / 240.0, observables=force)
            for selection in (observables, force):
                test.assertIs(selection.contacts, contacts)
            test.assertIsNone(body.contacts)
            count = int(contacts.rigid_contact_count.numpy()[0])
            test.assertGreater(count, 0)
            rows = observables.contact_f.numpy()
            np.testing.assert_array_equal(rows[count:], 0.0)
            sensor.update(state_0, contacts, observables=force)
            np.testing.assert_allclose(
                sensor.total_force.numpy()[0], -rows[:count, :3].sum(axis=0), rtol=1.0e-5, atol=1.0e-5
            )
            test.assertGreater(abs(sensor.total_force.numpy()[0, 2]), 0.0)
            viewer.log_contacts(contacts, state_0, observables=observables)
            with test.assertRaisesRegex(ValueError, "different Contacts"):
                sensor.update(state_0, other, observables=observables)
            with test.assertRaisesRegex(ValueError, "different Contacts"):
                viewer.log_contacts(other, state_0, observables=force)
            state_0, state_1 = state_1, state_0


def test_oversized_observables_fill_legacy_force(test, device):
    """Copy only the step contacts' rows into the deprecated force array."""
    kinds = newton.solvers.SolverBase.ObservableKind
    model = _make_resting_box(device)
    contacts_large = newton.CollisionPipeline(model, rigid_contact_max=64, soft_contact_max=0).contacts()
    with test.assertWarns(DeprecationWarning):
        model.request_contact_attributes("force")
    pipeline = newton.CollisionPipeline(model, rigid_contact_max=16, soft_contact_max=0)
    contacts = pipeline.contacts()
    test.assertEqual(contacts.force.shape, (16,))
    solver = newton.solvers.SolverXPBD(model)
    observables = solver.observables({kinds.CONTACT_F}, contacts=contacts_large)
    state_0, state_1 = model.state(), model.state()
    pipeline.collide(state_0, contacts)
    solver.step(state_0, state_1, None, contacts, 1.0 / 240.0, observables=observables)
    rows = observables.contact_f.numpy()
    test.assertGreater(np.abs(rows).sum(), 0.0)
    np.testing.assert_array_equal(contacts.force.numpy(), rows[:16])
    np.testing.assert_array_equal(rows[16:], 0.0)


def test_owned_and_external_pipelines(test, device):
    """Size each container from its own buffer when a solver-owned pipeline shares the model."""
    kinds = {newton.solvers.SolverBase.ObservableKind.CONTACT_F}
    for owned_first in (True, False):
        with test.subTest(owned_first=owned_first):
            builder = newton.ModelBuilder()
            builder.add_ground_plane()
            builder.add_particle(pos=wp.vec3(0.0, 0.0, 0.046), vel=wp.vec3(0.0), mass=1.0, radius=0.05)
            builder.color()
            model = builder.finalize(device=device)

            def make_owned(model=model):
                owned = newton.solvers.SolverVBD(
                    model,
                    iterations=2,
                    collision_pipeline=newton.CollisionPipeline(model, rigid_contact_max=8, soft_contact_max=8),
                )
                return owned, owned.observables(kinds)

            if owned_first:
                owned, owned_observables = make_owned()
            pipeline = newton.CollisionPipeline(model, rigid_contact_max=1000, soft_contact_max=64)
            contacts = pipeline.contacts()
            external = newton.solvers.SolverVBD(model, iterations=2)
            external_observables = external.observables(kinds, contacts=contacts)
            if not owned_first:
                owned, owned_observables = make_owned()

            test.assertEqual(owned_observables.contact_f.shape, (16,))
            test.assertEqual(external_observables.contact_f.shape, (1064,))
            owned.step(model.state(), model.state(), None, None, 1.0 / 60.0, observables=owned_observables)
            state_in = model.state()
            pipeline.collide(state_in, contacts)
            external.step(state_in, model.state(), None, contacts, 1.0 / 60.0, observables=external_observables)
            test.assertIs(owned_observables.contacts, owned.contacts)
            test.assertIs(external_observables.contacts, contacts)
            for observables, soft_start in ((owned_observables, 8), (external_observables, 1000)):
                rows = observables.contact_f.numpy()
                test.assertLess(rows[soft_start, 2], 0.0)
                np.testing.assert_array_equal(rows[:soft_start], 0.0)
                np.testing.assert_array_equal(rows[soft_start + 1 :], 0.0)


class TestContactObservableOwnership(unittest.TestCase):
    """Verify contact observables are owned by their Contacts, not by model-wide capacity."""


devices = get_test_devices()
for _name, _func in (
    ("test_external_and_private_admm_pipelines", test_external_and_private_admm_pipelines),
    ("test_alternating_contacts_production_and_consumption", test_alternating_contacts_production_and_consumption),
    ("test_oversized_observables_fill_legacy_force", test_oversized_observables_fill_legacy_force),
    ("test_owned_and_external_pipelines", test_owned_and_external_pipelines),
):
    add_function_test(TestContactObservableOwnership, _name, _func, devices=devices)


if __name__ == "__main__":
    unittest.main()
