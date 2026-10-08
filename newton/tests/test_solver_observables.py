# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Test eager allocation, naming, and ownership of solver observables."""

import inspect
import unittest
from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import warp as wp

import newton
from newton.selection import ArticulationView
from newton.solvers.experimental.coupled import SolverCoupled


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

    def test_uninitialized_capacities(self):
        """Distinguish uninitialized capacities from valid zero capacities."""
        self.assertIsNone(self.model.rigid_contact_max)
        self.assertIsNone(self.model.soft_contact_max)
        with self.assertRaisesRegex(RuntimeError, "CollisionPipeline"):
            self.solver.observables(kinds=self.kinds)
        with self.assertRaisesRegex(RuntimeError, "CollisionPipeline"):
            self.solver.observables()

    def test_observables_api_names(self):
        """Expose consistent observable names at producer and consumer boundaries."""
        self.assertIn("SolverBase", newton.solvers.__all__)
        self.assertNotIn("SolverObservableKind", newton.solvers.__all__)
        self.assertFalse(hasattr(newton.solvers, "SolverObservableKind"))
        self.assertEqual(newton.solvers.SolverBase.ObservableKind.__qualname__, "SolverBase.ObservableKind")
        self.assertNotIn("SolverObservables", newton.solvers.__all__)
        self.assertNotIn("SolverObservableFlags", newton.solvers.__all__)
        self.assertFalse(hasattr(newton.solvers, "SolverObservables"))
        self.assertFalse(hasattr(newton.solvers, "SolverObservableFlags"))
        self.assertEqual(newton.solvers.SolverBase.Observables.__qualname__, "SolverBase.Observables")
        self.assertIn("observables", inspect.signature(newton.solvers.SolverBase.step).parameters)
        for solver_type in (
            newton.solvers.SolverBase,
            newton.solvers.SolverMuJoCo,
            newton.solvers.SolverKamino,
            SolverCoupled,
        ):
            with self.subTest(solver=solver_type.__name__):
                kinds = inspect.signature(solver_type.observables).parameters["kinds"]
                self.assertEqual(kinds.kind, inspect.Parameter.KEYWORD_ONLY)
                self.assertIsNone(kinds.default)
        for consumer in (
            newton.sensors.SensorIMU.update,
            newton.sensors.SensorContact.update,
            newton.viewer.ViewerBase.log_contacts,
        ):
            self.assertIn("observables", inspect.signature(consumer).parameters)
            self.assertNotIn("outputs", inspect.signature(consumer).parameters)
            self.assertNotIn("solver_results", inspect.signature(consumer).parameters)
        self.assertTrue(callable(newton.solvers.SolverBase.observables))
        self.assertFalse(hasattr(newton.solvers, "SolverOutputs"))
        self.assertFalse(hasattr(newton.solvers, "SolverOutputFlags"))
        self.assertFalse(hasattr(newton.solvers.SolverBase, "outputs"))
        self.assertFalse(hasattr(newton.solvers.SolverBase, "results"))
        self.assertFalse(hasattr(newton.solvers, "SolverResults"))
        self.assertFalse(hasattr(newton.solvers, "SolverResultFlags"))
        self.assertFalse(hasattr(newton.solvers.SolverBase, "OBSERVABLES_TYPE"))
        self.assertFalse(hasattr(newton.solvers.SolverMuJoCo, "OBSERVABLES_TYPE"))
        self.assertIs(ContactSolver.Observables, newton.solvers.SolverBase.Observables)
        self.assertTrue(issubclass(newton.solvers.SolverMuJoCo.Observables, newton.solvers.SolverBase.Observables))
        self.assertTrue(
            issubclass(newton.solvers.SolverMuJoCo.ObservableKind, newton.solvers.SolverBase.ObservableKind)
        )
        self.assertIs(
            newton.solvers.SolverMuJoCo.ObservableKind.BODY_QDD,
            newton.solvers.SolverBase.ObservableKind.BODY_QDD,
        )

    def test_body_only_observables_need_no_pipeline(self):
        """Allocate body observables without constructing a collision pipeline."""
        observables = self.solver.observables(kinds={newton.solvers.SolverBase.ObservableKind.BODY_QDD})
        self.assertEqual(observables.body_qdd.shape, (self.model.body_count,))
        self.assertIsNone(observables.contact_f)

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
        solver = CustomContactSolver(self.model)
        observables = solver.observables(kinds={kinds.BODY_QDD, kinds.CONTACT_F, CustomContactKind.PRESSURE})
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
        contacts = pipeline.contacts()
        solver.validate_observables(pressure, contacts)
        for selection in (observables, pressure, force, force.select(force.kinds)):
            self.assertIs(selection.contacts, contacts)
            with self.assertRaisesRegex(ValueError, "Contacts instance"):
                solver.validate_observables(selection, pipeline.contacts())
        self.assertIsNone(body.contacts)

    def test_select_zero_capacity_remains_requested(self):
        """Treat selected zero-length contact buffers as requested."""
        newton.CollisionPipeline(self.model, rigid_contact_max=0, soft_contact_max=0)
        observables = self.solver.observables(kinds=self.kinds)
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
        observables = solver.observables(kinds={kinds.BODY_QDD, CustomContactKind.PRESSURE})
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
        solver = CustomContactSolver(model)
        kinds = newton.solvers.SolverBase.ObservableKind
        observables = solver.observables(kinds={kinds.CONTACT_F, CustomContactKind.PRESSURE})
        selected = observables.select({CustomContactKind.PRESSURE})
        contacts = pipeline.contacts()
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
        with self.assertRaisesRegex(RuntimeError, "CollisionPipeline"):
            solver.observables(kinds={CustomContactKind.PRESSURE})
        pipeline = newton.CollisionPipeline(self.model, rigid_contact_max=5, soft_contact_max=3)
        observables = solver.observables(kinds={CustomContactKind.PRESSURE})
        self.assertEqual(observables.pressure.shape, (3,))
        self.assertIsNone(observables.contact_f)
        with self.assertRaisesRegex(ValueError, "Contacts"):
            solver.validate_observables(observables)
        solver.validate_observables(observables, pipeline.contacts())

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
        observables = ContactSolver(model).observables(kinds=self.kinds)
        with self.assertRaisesRegex(AttributeError, "dynamic contact"):
            view.get_attribute("contact_f", observables)
        self.assertEqual(observables.contact_f.shape, (pipeline.rigid_contact_max + pipeline.soft_contact_max,))

    def test_eager_capacity_allocation(self):
        """Allocate full rigid and soft capacity before creating Contacts."""
        pipeline = newton.CollisionPipeline(self.model, rigid_contact_max=5, soft_contact_max=3)
        observables = self.solver.observables(kinds=self.kinds, requires_grad=True)
        self.assertEqual((self.model.rigid_contact_max, self.model.soft_contact_max), (5, 3))
        self.assertEqual(observables.contact_f.shape, (8,))
        self.assertTrue(observables.contact_f.requires_grad)
        self.assertIsNone(observables.contacts)
        contacts = pipeline.contacts()
        pointer = observables.contact_f.ptr
        self.solver.validate_observables(observables, contacts)
        self.assertIs(observables.contacts, contacts)
        self.assertEqual(observables.contact_f.ptr, pointer)

    def test_zero_capacity_is_requested(self):
        """Keep a requested empty array distinct from an unrequested field."""
        newton.CollisionPipeline(self.model, rigid_contact_max=0, soft_contact_max=0)
        observables = self.solver.observables(kinds=self.kinds)
        self.assertIsNotNone(observables.contact_f)
        self.assertEqual(observables.contact_f.shape, (0,))
        self.assertIsNone(self.solver.observables(kinds=set()).contact_f)

    def test_validate_contact_layout_and_identity(self):
        """Reject mismatched layouts and bind storage on the first step."""
        pipeline = newton.CollisionPipeline(self.model, rigid_contact_max=5, soft_contact_max=3)
        observables = self.solver.observables(kinds=self.kinds)
        with self.assertRaisesRegex(ValueError, "Contacts"):
            self.solver.validate_observables(observables)
        with self.assertRaisesRegex(ValueError, "capacit"):
            self.solver.validate_observables(observables, newton.Contacts(4, 4, device="cpu"))
        self.assertIsNone(observables.contacts)
        contacts = pipeline.contacts()
        self.solver.validate_observables(observables, contacts)
        self.solver.validate_observables(observables, contacts)
        with self.assertRaisesRegex(ValueError, "Contacts instance"):
            self.solver.validate_observables(observables, pipeline.contacts())

    def test_freeze_capacity_after_output_allocation(self):
        """Reject capacity changes that would invalidate allocated observables."""
        newton.CollisionPipeline(self.model, rigid_contact_max=5, soft_contact_max=3)
        self.solver.observables(kinds=self.kinds)
        with self.assertRaisesRegex(ValueError, "capacit"):
            newton.CollisionPipeline(self.model, rigid_contact_max=6, soft_contact_max=3)
        with self.assertRaisesRegex(ValueError, "capacit"):
            self.model.rigid_contact_max = 6
        with self.assertRaisesRegex(ValueError, "capacit"):
            self.model.soft_contact_max = 4
        self.assertEqual((self.model.rigid_contact_max, self.model.soft_contact_max), (5, 3))

    def test_failed_pipeline_does_not_publish_capacity(self):
        """Leave capacities uninitialized when pipeline construction fails."""
        with self.assertRaises(ValueError):
            newton.CollisionPipeline(self.model, rigid_contact_max=5, max_triangle_pairs=0)
        self.assertIsNone(self.model.rigid_contact_max)
        self.assertIsNone(self.model.soft_contact_max)

    def test_late_pipeline_failure_preserves_published_capacity(self):
        """Publish both capacities only after every pipeline allocation succeeds."""
        newton.CollisionPipeline(self.model, rigid_contact_max=5, soft_contact_max=3)
        with patch("newton._src.sim.collide.ContactSorter", side_effect=RuntimeError("sorter allocation failed")):
            with self.assertRaisesRegex(RuntimeError, "sorter allocation failed"):
                newton.CollisionPipeline(self.model, rigid_contact_max=6, soft_contact_max=4, deterministic=True)
        self.assertEqual((self.model.rigid_contact_max, self.model.soft_contact_max), (5, 3))
        self.assertEqual(self.solver.observables(kinds=self.kinds).contact_f.shape, (8,))

    def test_matching_pipeline_preserves_frozen_capacity(self):
        """Allow another pipeline only if its capacities preserve live observables."""
        newton.CollisionPipeline(self.model, rigid_contact_max=5, soft_contact_max=3)
        observables = self.solver.observables(kinds=self.kinds)
        matching = newton.CollisionPipeline(self.model, rigid_contact_max=5, soft_contact_max=3)
        self.solver.validate_observables(observables, matching.contacts())
        self.assertEqual(observables.contact_f.shape, (8,))

    def test_contact_count_does_not_resize_output(self):
        """Keep allocation stable as collision changes the active contact count."""
        builder = newton.ModelBuilder()
        builder.add_ground_plane()
        builder.add_particle(pos=(0, 0, 1), vel=(0, 0, 0), mass=1.0, radius=0.1)
        model = builder.finalize(device="cpu")
        pipeline = newton.CollisionPipeline(model, rigid_contact_max=0)
        observables = ContactSolver(model).observables(kinds=self.kinds)
        pointer = observables.contact_f.ptr
        contacts = pipeline.contacts()
        state = model.state()
        pipeline.collide(state, contacts)
        self.assertEqual(contacts.soft_contact_count.numpy()[0], 0)
        state.particle_q.assign([[0, 0, 0.05]])
        pipeline.collide(state, contacts)
        self.assertEqual(contacts.soft_contact_count.numpy()[0], 1)
        self.assertEqual(observables.contact_f.shape, (pipeline.soft_contact_max,))
        self.assertEqual(observables.contact_f.ptr, pointer)

    def test_manual_counts_do_not_replace_pipeline_setup(self):
        """Require pipeline initialization even when capacities are assigned."""
        self.model.rigid_contact_max = 5
        self.model.soft_contact_max = 3
        with self.assertRaisesRegex(RuntimeError, "CollisionPipeline"):
            self.solver.observables(kinds=self.kinds)

    def test_custom_contact_output_without_contact_force(self):
        """Resolve and freeze capacities for custom contact kinds independently."""
        solver = CustomContactSolver(self.model)
        kinds = {CustomContactKind.PRESSURE}
        with self.assertRaisesRegex(RuntimeError, "CollisionPipeline"):
            solver.observables(kinds=kinds)
        pipeline = newton.CollisionPipeline(self.model, rigid_contact_max=4, soft_contact_max=0)
        observables = solver.observables(kinds=kinds)
        self.assertEqual(observables.pressure.shape, (4,))
        self.assertIsNone(observables.contact_f)
        solver.validate_observables(observables, pipeline.contacts())
        with self.assertRaisesRegex(ValueError, "capacit"):
            self.model.rigid_contact_max = 5

    def test_native_backend_rejects_insufficient_capacity(self):
        """Reject incompatible native budgets without freezing the model's capacity."""
        newton.CollisionPipeline(self.model, rigid_contact_max=2, soft_contact_max=0)
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
                with self.assertRaisesRegex(ValueError, "exceeds CollisionPipeline capacity"):
                    solver.observables(kinds=self.kinds)
        # A failed request must not freeze setup; users can correct the budget.
        newton.CollisionPipeline(self.model, rigid_contact_max=3, soft_contact_max=0)
        for solver in (mujoco, kamino):
            self.assertEqual(solver.observables(kinds=self.kinds).contact_f.shape, (3,))

    def test_graph_reuses_preallocated_output(self):
        """Reuse a preallocated output across CUDA graph replays."""
        if not wp.is_cuda_available():
            self.skipTest("CUDA graph capture requires a CUDA device")
        model = newton.ModelBuilder().finalize(device="cuda:0")
        pipeline = newton.CollisionPipeline(model, rigid_contact_max=3, soft_contact_max=0)
        solver = ContactSolver(model)
        observables = solver.observables(kinds=self.kinds)
        contacts = pipeline.contacts()
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
        solver = ContactSolver(model)
        observables = solver.observables(kinds=self.kinds)
        contacts = pipeline.contacts()
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


if __name__ == "__main__":
    unittest.main()
