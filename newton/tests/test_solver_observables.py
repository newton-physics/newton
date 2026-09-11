# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Test eager allocation, naming, and ownership of solver observables."""

import inspect
import unittest
from enum import Enum
from types import SimpleNamespace
from unittest.mock import patch

import warp as wp

import newton


class ContactSolver(newton.solvers.SolverBase):
    """Exercise the base allocation contract without a numerical backend."""

    SUPPORTED_OBSERVABLE_FLAGS = frozenset(newton.solvers.SolverObservableFlags)


class CustomContactFlags(Enum):
    """Define a contact-indexed diagnostic independently of CONTACT_F."""

    PRESSURE = "pressure"


class CustomContactObservables(newton.solvers.SolverObservables):
    """Keep custom contact pressure separate from the standard force array."""

    def __init__(self, flags=()):
        super().__init__(flags)
        self.pressure: wp.array[float] | None = None


class CustomContactSolver(ContactSolver):
    """Exercise eager allocation of solver-specific contact fields."""

    OBSERVABLES_TYPE = CustomContactObservables
    SUPPORTED_OBSERVABLE_FLAGS = ContactSolver.SUPPORTED_OBSERVABLE_FLAGS | {CustomContactFlags.PRESSURE}
    CONTACT_OBSERVABLE_FLAGS = ContactSolver.CONTACT_OBSERVABLE_FLAGS | {CustomContactFlags.PRESSURE}

    def _allocate_observables(self, observables, *, requires_grad):
        super()._allocate_observables(observables, requires_grad=requires_grad)
        if CustomContactFlags.PRESSURE in observables:
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
