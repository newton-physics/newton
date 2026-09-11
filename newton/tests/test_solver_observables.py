# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Test eager allocation, naming, and ownership of solver observables."""

import inspect
import unittest
from enum import Enum
from types import SimpleNamespace
from typing import ClassVar
from unittest.mock import patch

import numpy as np
import warp as wp

import newton
from newton.selection import ArticulationView


class ContactSolver(newton.solvers.SolverBase):
    """Exercise the base allocation contract without a numerical backend."""

    SUPPORTED_OBSERVABLE_FLAGS = frozenset(newton.solvers.SolverObservableFlags)


class CustomContactFlags(Enum):
    """Define a contact-indexed diagnostic independently of CONTACT_F."""

    PRESSURE = "pressure"


class CustomContactObservables(newton.solvers.SolverObservables):
    """Keep custom contact pressure separate from the standard force array."""

    ATTRIBUTE_FREQUENCIES: ClassVar = {"pressure": newton.Model.AttributeFrequency.CONTACT_RIGID}

    def __init__(self, flags=()):
        super().__init__(flags)
        self.pressure: wp.array[float] | None = None


class CustomContactSolver(ContactSolver):
    """Exercise eager allocation of solver-specific contact fields."""

    OBSERVABLES_TYPE = CustomContactObservables
    SUPPORTED_OBSERVABLE_FLAGS = ContactSolver.SUPPORTED_OBSERVABLE_FLAGS | {CustomContactFlags.PRESSURE}

    def _allocate_observables(self, observables, *, requires_grad):
        super()._allocate_observables(observables, requires_grad=requires_grad)
        if observables.is_requested(CustomContactFlags.PRESSURE):
            observables.pressure = wp.zeros(
                self.model.rigid_contact_max, dtype=float, device=self.model.device, requires_grad=requires_grad
            )


class TestSolverObservables(unittest.TestCase):
    """Verify contact capacities are resolved before allocating results."""

    def setUp(self):
        """Build an independent CPU model for each capacity test."""
        builder = newton.ModelBuilder()
        builder.add_body(mass=0.0)
        self.model = builder.finalize(device="cpu")
        self.solver = ContactSolver(self.model)
        self.flags = {newton.solvers.SolverObservableFlags.CONTACT_F}

    def test_uninitialized_capacities(self):
        """Distinguish uninitialized capacities from valid zero capacities."""
        self.assertIsNone(self.model.rigid_contact_max)
        self.assertIsNone(self.model.soft_contact_max)
        with self.assertRaisesRegex(RuntimeError, "CollisionPipeline"):
            self.solver.observables(self.flags)

    def test_observables_api_names(self):
        """Expose consistent observable names at producer and consumer boundaries."""
        self.assertIn("SolverObservables", newton.solvers.__all__)
        self.assertIn("SolverObservableFlags", newton.solvers.__all__)
        self.assertIn("observables", inspect.signature(newton.solvers.SolverBase.step).parameters)
        for consumer in (
            newton.sensors.SensorIMU.update,
            newton.sensors.SensorContact.update,
            newton.viewer.ViewerBase.log_contacts,
        ):
            self.assertIn("solver_observables", inspect.signature(consumer).parameters)
            self.assertNotIn("outputs", inspect.signature(consumer).parameters)
            self.assertNotIn("solver_results", inspect.signature(consumer).parameters)
        self.assertTrue(callable(newton.solvers.SolverBase.observables))
        self.assertFalse(hasattr(newton.solvers, "SolverOutputs"))
        self.assertFalse(hasattr(newton.solvers, "SolverOutputFlags"))
        self.assertFalse(hasattr(newton.solvers.SolverBase, "outputs"))
        self.assertFalse(hasattr(newton.solvers.SolverBase, "results"))
        self.assertFalse(hasattr(newton.solvers, "SolverResults"))
        self.assertFalse(hasattr(newton.solvers, "SolverResultFlags"))
        self.assertIs(newton.solvers.SolverMuJoCo.OBSERVABLES_TYPE, newton.solvers.SolverMuJoCo.Observables)
        self.assertTrue(issubclass(newton.solvers.SolverMuJoCo.Observables, newton.solvers.SolverObservables))

    def test_body_only_observables_need_no_pipeline(self):
        """Allocate body observables without constructing a collision pipeline."""
        observables = self.solver.observables({newton.solvers.SolverObservableFlags.BODY_QDD})
        self.assertEqual(observables.body_qdd.shape, (self.model.body_count,))
        self.assertIsNone(observables.contact_f)

    def test_select_shares_arrays_without_allocating(self):
        """Select existing buffers without allocating, copying, or mutating the source."""
        flags = newton.solvers.SolverObservableFlags
        observables = self.solver.observables({flags.BODY_QDD, flags.BODY_PARENT_F}, requires_grad=True)
        with patch.object(self.solver, "_allocate_observables", side_effect=AssertionError("allocated")):
            selected = observables.select({flags.BODY_QDD})
        self.assertIsNot(selected, observables)
        self.assertIs(type(selected), type(observables))
        self.assertIs(selected.body_qdd, observables.body_qdd)
        self.assertIs(selected.body_qdd.grad, observables.body_qdd.grad)
        self.assertIsNone(selected.body_parent_f)
        self.assertIsNotNone(observables.body_parent_f)
        self.assertEqual(observables.flags, {flags.BODY_QDD, flags.BODY_PARENT_F})
        self.assertEqual(selected.flags, {flags.BODY_QDD})
        self.assertTrue(selected.is_requested(flags.BODY_QDD))
        self.assertFalse(selected.is_requested(flags.BODY_PARENT_F))
        self.assertIn(flags.BODY_QDD, selected)
        self.assertNotIn(flags.BODY_PARENT_F, selected)
        self.assertIs(selected.model, self.model)
        selected.body_qdd.fill_(wp.spatial_vector(7.0))
        np.testing.assert_array_equal(observables.body_qdd.numpy(), np.full((1, 6), 7.0))
        self.solver._validate_observables(selected)
        with self.assertRaisesRegex(ValueError, "solver instance"):
            ContactSolver(self.model)._validate_observables(selected)

    def test_select_empty_and_nested_subsets(self):
        """Narrow selections without re-enabling fields excluded by their parent."""
        flags = newton.solvers.SolverObservableFlags
        observables = self.solver.observables({flags.BODY_QDD, flags.BODY_PARENT_F})
        selected = observables.select({flags.BODY_QDD})
        self.assertIs(selected.select(selected.flags).body_qdd, observables.body_qdd)
        empty = selected.select(set())
        self.assertEqual(empty.flags, frozenset())
        self.assertIsNone(empty.body_qdd)
        self.assertIsNone(empty.body_parent_f)
        self.solver._validate_observables(empty)
        for source, requested in ((observables, {flags.CONTACT_F}), (selected, {flags.BODY_PARENT_F})):
            with self.subTest(requested=requested), self.assertRaisesRegex(ValueError, "not requested"):
                source.select(requested)
        with self.assertRaises(AttributeError):
            selected.flags = frozenset()
        with self.assertRaisesRegex(ValueError, "allocated"):
            newton.solvers.SolverObservables().select(set())

    def test_select_custom_fields_and_shared_contact_binding(self):
        """Share binding across custom contact subsets created before the first step."""
        flags = newton.solvers.SolverObservableFlags
        pipeline = newton.CollisionPipeline(self.model, rigid_contact_max=3, soft_contact_max=0)
        solver = CustomContactSolver(self.model)
        observables = solver.observables({flags.BODY_QDD, flags.CONTACT_F, CustomContactFlags.PRESSURE})
        pressure = observables.select({CustomContactFlags.PRESSURE})
        force = observables.select({flags.CONTACT_F})
        body = observables.select({flags.BODY_QDD})
        self.assertIs(type(pressure), CustomContactObservables)
        self.assertIs(pressure.pressure, observables.pressure)
        self.assertIsNone(pressure.contact_f)
        self.assertIsNone(force.pressure)
        self.assertEqual(pressure.get_attribute_frequency("pressure"), newton.Model.AttributeFrequency.CONTACT_RIGID)
        solver._validate_observables(body)
        solver._validate_observables(observables.select(set()))
        with self.assertRaisesRegex(ValueError, "Contacts"):
            solver._validate_observables(pressure)
        contacts = pipeline.contacts()
        solver._validate_observables(pressure, contacts)
        for selection in (observables, pressure, force, force.select(force.flags)):
            self.assertIs(selection.contacts, contacts)
            with self.assertRaisesRegex(ValueError, "Contacts instance"):
                solver._validate_observables(selection, pipeline.contacts())
        self.assertIsNone(body.contacts)

    def test_select_zero_capacity_remains_requested(self):
        """Treat selected zero-length contact buffers as requested."""
        newton.CollisionPipeline(self.model, rigid_contact_max=0, soft_contact_max=0)
        observables = self.solver.observables(self.flags)
        selected = observables.select(self.flags)
        self.assertTrue(selected.is_requested(newton.solvers.SolverObservableFlags.CONTACT_F))
        self.assertIs(selected.contact_f, observables.contact_f)
        self.assertEqual(selected.contact_f.shape, (0,))

    def test_select_substep_schedule(self):
        """Update selected custom fields per substep and preserve skipped values."""
        flags = newton.solvers.SolverObservableFlags

        class SamplingSolver(CustomContactSolver):
            def step(self, state_in, state_out, control, contacts, dt, *, observables=None):
                self._validate_observables(observables, contacts)
                if observables is not None:
                    for flag in (flags.BODY_QDD, CustomContactFlags.PRESSURE):
                        if observables.is_requested(flag):
                            array = getattr(observables, flag.value)
                            array.fill_(dt if flag is CustomContactFlags.PRESSURE else wp.spatial_vector(dt))

        pipeline = newton.CollisionPipeline(self.model, rigid_contact_max=2, soft_contact_max=0)
        contacts = pipeline.contacts()
        solver = SamplingSolver(self.model)
        observables = solver.observables({flags.BODY_QDD, CustomContactFlags.PRESSURE})
        every_substep = observables.select({flags.BODY_QDD})
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
        flags = newton.solvers.SolverObservableFlags
        observables = solver.observables({flags.BODY_QDD, flags.BODY_PARENT_F})
        early = observables.select({flags.BODY_QDD})
        last = observables.select({flags.BODY_PARENT_F})
        pointer = observables.body_qdd.ptr
        with wp.ScopedCapture(device=model.device) as capture:
            for step in range(3):
                selected = last if step == 2 else early
                solver._validate_observables(selected)
                for flag in flags:
                    if selected.is_requested(flag):
                        getattr(selected, flag.value).fill_(wp.spatial_vector(step + 1.0))
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
        flags = newton.solvers.SolverObservableFlags
        observables = solver.observables({flags.CONTACT_F, CustomContactFlags.PRESSURE})
        selected = observables.select({CustomContactFlags.PRESSURE})
        contacts = pipeline.contacts()
        condition = wp.ones(1, dtype=wp.int32, device=model.device)
        observables.contact_f.fill_(wp.spatial_vector(-1.0))

        def body():
            solver._validate_observables(selected, contacts)
            if selected.is_requested(CustomContactFlags.PRESSURE):
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
        """Inherit standard row frequencies without another capability flag set."""
        observables = CustomContactObservables()
        frequency = newton.Model.AttributeFrequency
        self.assertEqual(observables.get_attribute_frequency("body_qdd"), frequency.BODY)
        self.assertEqual(observables.get_attribute_frequency("pressure"), frequency.CONTACT_RIGID)
        self.assertEqual(observables.get_attribute_frequency("contact_f"), frequency.CONTACT)
        self.assertFalse(hasattr(ContactSolver, "CONTACT_OBSERVABLE_FLAGS"))

    def test_missing_observable_frequency(self):
        """Reject custom requests without a declared row frequency before allocating."""
        solver = CustomContactSolver(self.model)
        solver.OBSERVABLES_TYPE = newton.solvers.SolverObservables
        with self.assertRaisesRegex(ValueError, "frequency.*pressure"):
            solver.observables({CustomContactFlags.PRESSURE})

    def test_invalid_frequency_declaration(self):
        """Reject integer frequency values rather than guessing their domain."""

        class InvalidObservables(CustomContactObservables):
            ATTRIBUTE_FREQUENCIES: ClassVar = {"pressure": 5}

        solver = CustomContactSolver(self.model)
        solver.OBSERVABLES_TYPE = InvalidObservables
        with self.assertRaisesRegex(TypeError, "Invalid observable frequency.*pressure"):
            solver.observables({CustomContactFlags.PRESSURE})

    def test_flag_values_name_array_fields(self):
        """Reject plain enum members whose values cannot identify an array field."""

        class InvalidFlags(Enum):
            NUMBER = 0
            EXPRESSION = "pressure[0]"

        for flag in InvalidFlags:
            with self.subTest(flag=flag), self.assertRaisesRegex(TypeError, "string naming its array field"):
                self.solver.observables({flag})

    def test_contact_frequency_counts(self):
        """Resolve each contact domain from capacity rather than live counts."""
        frequency = newton.Model.AttributeFrequency
        with self.assertRaisesRegex(RuntimeError, "CollisionPipeline"):
            self.model._attribute_frequency_count(frequency.CONTACT_SOFT)
        newton.CollisionPipeline(self.model, rigid_contact_max=5, soft_contact_max=3)
        for domain, count in ((frequency.CONTACT_RIGID, 5), (frequency.CONTACT_SOFT, 3), (frequency.CONTACT, 8)):
            with self.subTest(domain=domain):
                self.assertEqual(self.model._attribute_frequency_count(domain), count)

    def test_custom_soft_contact_frequency(self):
        """Require contact binding for a soft-only diagnostic without standard forces."""

        class SoftObservables(CustomContactObservables):
            ATTRIBUTE_FREQUENCIES: ClassVar = {"pressure": newton.Model.AttributeFrequency.CONTACT_SOFT}

        class SoftSolver(CustomContactSolver):
            OBSERVABLES_TYPE = SoftObservables

            def _allocate_observables(self, observables, *, requires_grad):
                observables.pressure = wp.zeros(self.model.soft_contact_max, device=self.model.device)

        solver = SoftSolver(self.model)
        with self.assertRaisesRegex(RuntimeError, "CollisionPipeline"):
            solver.observables({CustomContactFlags.PRESSURE})
        pipeline = newton.CollisionPipeline(self.model, rigid_contact_max=5, soft_contact_max=3)
        observables = solver.observables({CustomContactFlags.PRESSURE})
        self.assertEqual(observables.pressure.shape, (3,))
        self.assertIsNone(observables.contact_f)
        with self.assertRaisesRegex(ValueError, "Contacts"):
            solver._validate_observables(observables)
        solver._validate_observables(observables, pipeline.contacts())

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

                class Observables(CustomContactObservables):
                    ATTRIBUTE_FREQUENCIES: ClassVar = {"pressure": frequency}

                class Solver(CustomContactSolver):
                    OBSERVABLES_TYPE = Observables

                    def _allocate_observables(self, observables, *, requires_grad):
                        count = self.model._attribute_frequency_count(observables.get_attribute_frequency("pressure"))
                        observables.pressure = wp.zeros(count, device=self.model.device)

                observables = Solver(model).observables({CustomContactFlags.PRESSURE})
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
                        "body_qdd", self.solver.observables({newton.solvers.SolverObservableFlags.BODY_QDD})
                    )

        pipeline = newton.CollisionPipeline(model, rigid_contact_max=2, soft_contact_max=1)
        observables = ContactSolver(model).observables(self.flags)
        with self.assertRaisesRegex(AttributeError, "dynamic contact"):
            view.get_attribute("contact_f", observables)
        self.assertEqual(observables.contact_f.shape, (pipeline.rigid_contact_max + pipeline.soft_contact_max,))

    def test_eager_capacity_allocation(self):
        """Allocate full rigid and soft capacity before creating Contacts."""
        pipeline = newton.CollisionPipeline(self.model, rigid_contact_max=5, soft_contact_max=3)
        observables = self.solver.observables(self.flags, requires_grad=True)
        self.assertEqual((self.model.rigid_contact_max, self.model.soft_contact_max), (5, 3))
        self.assertEqual(observables.contact_f.shape, (8,))
        self.assertTrue(observables.contact_f.requires_grad)
        self.assertIsNone(observables.contacts)
        contacts = pipeline.contacts()
        pointer = observables.contact_f.ptr
        self.solver._validate_observables(observables, contacts)
        self.assertIs(observables.contacts, contacts)
        self.assertEqual(observables.contact_f.ptr, pointer)

    def test_zero_capacity_is_requested(self):
        """Keep a requested empty array distinct from an unrequested field."""
        newton.CollisionPipeline(self.model, rigid_contact_max=0, soft_contact_max=0)
        observables = self.solver.observables(self.flags)
        self.assertIsNotNone(observables.contact_f)
        self.assertEqual(observables.contact_f.shape, (0,))
        self.assertIsNone(self.solver.observables(set()).contact_f)

    def test_validate_contact_layout_and_identity(self):
        """Reject mismatched layouts and bind storage on the first step."""
        pipeline = newton.CollisionPipeline(self.model, rigid_contact_max=5, soft_contact_max=3)
        observables = self.solver.observables(self.flags)
        with self.assertRaisesRegex(ValueError, "Contacts"):
            self.solver._validate_observables(observables)
        with self.assertRaisesRegex(ValueError, "capacit"):
            self.solver._validate_observables(observables, newton.Contacts(4, 4, device="cpu"))
        self.assertIsNone(observables.contacts)
        contacts = pipeline.contacts()
        self.solver._validate_observables(observables, contacts)
        self.solver._validate_observables(observables, contacts)
        with self.assertRaisesRegex(ValueError, "Contacts instance"):
            self.solver._validate_observables(observables, pipeline.contacts())

    def test_freeze_capacity_after_output_allocation(self):
        """Reject capacity changes that would invalidate allocated observables."""
        newton.CollisionPipeline(self.model, rigid_contact_max=5, soft_contact_max=3)
        self.solver.observables(self.flags)
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
        self.assertEqual(self.solver.observables(self.flags).contact_f.shape, (8,))

    def test_matching_pipeline_preserves_frozen_capacity(self):
        """Allow another pipeline only if its capacities preserve live observables."""
        newton.CollisionPipeline(self.model, rigid_contact_max=5, soft_contact_max=3)
        observables = self.solver.observables(self.flags)
        matching = newton.CollisionPipeline(self.model, rigid_contact_max=5, soft_contact_max=3)
        self.solver._validate_observables(observables, matching.contacts())
        self.assertEqual(observables.contact_f.shape, (8,))

    def test_contact_count_does_not_resize_output(self):
        """Keep allocation stable as collision changes the active contact count."""
        builder = newton.ModelBuilder()
        builder.add_ground_plane()
        builder.add_particle(pos=(0, 0, 1), vel=(0, 0, 0), mass=1.0, radius=0.1)
        model = builder.finalize(device="cpu")
        pipeline = newton.CollisionPipeline(model, rigid_contact_max=0)
        observables = ContactSolver(model).observables(self.flags)
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
            self.solver.observables(self.flags)

    def test_custom_contact_output_without_contact_force(self):
        """Resolve and freeze capacities for custom contact flags independently."""
        solver = CustomContactSolver(self.model)
        flags = {CustomContactFlags.PRESSURE}
        with self.assertRaisesRegex(RuntimeError, "CollisionPipeline"):
            solver.observables(flags)
        pipeline = newton.CollisionPipeline(self.model, rigid_contact_max=4, soft_contact_max=0)
        observables = solver.observables(flags)
        self.assertEqual(observables.pressure.shape, (4,))
        self.assertIsNone(observables.contact_f)
        solver._validate_observables(observables, pipeline.contacts())
        with self.assertRaisesRegex(ValueError, "capacit"):
            self.model.rigid_contact_max = 5

    def test_native_backend_rejects_insufficient_capacity(self):
        """Reject native backend budgets before allocating any output arrays."""
        newton.CollisionPipeline(self.model, rigid_contact_max=2, soft_contact_max=0)
        mujoco = object.__new__(newton.solvers.SolverMuJoCo)
        newton.solvers.SolverBase.__init__(mujoco, self.model)
        mujoco.use_mujoco_cpu = False
        mujoco.mjw_data = SimpleNamespace(naconmax=3)
        kamino = object.__new__(newton.solvers.SolverKamino)
        newton.solvers.SolverBase.__init__(kamino, self.model)
        kamino._collision_detector_kamino = object()
        kamino._contacts_kamino = SimpleNamespace(model_max_contacts_host=3)
        for solver in (mujoco, kamino):
            with self.subTest(solver=type(solver).__name__):
                with self.assertRaisesRegex(ValueError, "exceeds CollisionPipeline capacity"):
                    solver.observables(self.flags)
        # A failed request must not freeze setup; users can correct the budget.
        newton.CollisionPipeline(self.model, rigid_contact_max=3, soft_contact_max=0)
        for solver in (mujoco, kamino):
            self.assertEqual(solver.observables(self.flags).contact_f.shape, (3,))

    def test_graph_reuses_preallocated_output(self):
        """Reuse a preallocated output across CUDA graph replays."""
        if not wp.is_cuda_available():
            self.skipTest("CUDA graph capture requires a CUDA device")
        model = newton.ModelBuilder().finalize(device="cuda:0")
        pipeline = newton.CollisionPipeline(model, rigid_contact_max=3, soft_contact_max=0)
        solver = ContactSolver(model)
        observables = solver.observables(self.flags)
        contacts = pipeline.contacts()
        pointer = observables.contact_f.ptr
        with wp.ScopedCapture(device=model.device) as capture:
            solver._validate_observables(observables, contacts)
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
        observables = solver.observables(self.flags)
        contacts = pipeline.contacts()
        condition = wp.ones(1, dtype=wp.int32, device=model.device)

        def body():
            solver._validate_observables(observables, contacts)
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
