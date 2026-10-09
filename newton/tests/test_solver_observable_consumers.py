# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Validate observable consumers before the first simulation step."""

import importlib
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import newton
from newton.solvers import SolverObservableFlags, SolverXPBD
from newton.viewer import ViewerNull


class TestSolverObservableConsumers(unittest.TestCase):
    """Check contact allocation boundaries without requiring a solver step."""

    def test_robot_examples_export_only_on_final_substep(self):
        """Run real example loops with lightweight state and solver stand-ins."""
        for name in ("g1", "h1", "anymal_d", "policy"):
            with self.subTest(example=name):
                try:
                    module = importlib.import_module(f"newton.examples.robot.example_robot_{name}")
                except ModuleNotFoundError as error:
                    if name == "policy" and error.name == "warp_nn":
                        self.skipTest("Policy example requires the optional warp-nn dependency")
                    raise
                for substeps in (1, 3, 4):
                    example = module.Example.__new__(module.Example)
                    example.use_mujoco_contacts = True
                    example.use_graph = True
                    example.sim_substeps = substeps
                    example.sim_dt = 0.001
                    example.state_0 = SimpleNamespace(clear_forces=Mock(), assign=Mock())
                    example.state_1 = SimpleNamespace(clear_forces=Mock(), assign=Mock())
                    example.viewer = SimpleNamespace(apply_forces=Mock())
                    example.solver = SimpleNamespace(step=Mock())
                    example.control = object()
                    example.contacts = object()
                    example.solver_observables = object()
                    example.simulate()
                    requests = [call.kwargs["observables"] for call in example.solver.step.call_args_list]
                    self.assertEqual(requests, [None] * (substeps - 1) + [example.solver_observables])

    def test_builder_rejects_contact_frequencies(self):
        """Reject unsupported builder row domains before registering a custom attribute."""
        builder = newton.ModelBuilder()
        frequency = newton.Model.AttributeFrequency
        for domain in (frequency.CONTACT, frequency.CONTACT_RIGID, frequency.CONTACT_SOFT):
            with self.subTest(frequency=domain):
                attribute = newton.ModelBuilder.CustomAttribute(name="pressure", frequency=domain, dtype=float)
                with self.assertRaisesRegex(ValueError, "SolverObservables"):
                    builder.add_custom_attribute(attribute)
                self.assertNotIn("pressure", builder.custom_attributes)

    def test_viewer_initial_contact_read(self):
        """Render pre-step zero forces and validate storage only when contacts are shown."""
        builder = newton.ModelBuilder()
        builder.add_ground_plane()
        model = builder.finalize(device="cpu")
        pipeline = newton.CollisionPipeline(model, rigid_contact_max=2, soft_contact_max=0)
        contacts = pipeline.contacts()
        observables = SolverXPBD(model).observables({SolverObservableFlags.CONTACT_F})
        viewer = ViewerNull()
        viewer.set_model(model)
        viewer.show_contacts = False
        viewer.log_contacts(contacts, model.state(), observables=observables)
        self.assertIsNone(observables.contacts)
        viewer.show_contacts = True
        viewer.log_contacts(contacts, model.state(), observables=observables)
        self.assertIs(observables.contacts, contacts)
        with self.assertRaisesRegex(ValueError, "Contacts instance"):
            viewer.log_contacts(pipeline.contacts(), model.state(), observables=observables)

    def test_kamino_does_not_require_export_capacity_without_request(self):
        """Allow a native solver with a smaller unrelated pipeline until forces are requested."""
        builder = newton.ModelBuilder()
        builder.begin_world()
        body = builder.add_body(mass=1.0)
        builder.add_shape_sphere(body, radius=0.1)
        builder.add_ground_plane()
        builder.end_world()
        model = builder.finalize(device="cpu")
        newton.CollisionPipeline(model, rigid_contact_max=0, soft_contact_max=0)
        config = newton.solvers.SolverKamino.Config(use_collision_detector=True)
        solver = newton.solvers.SolverKamino(model, config=config)
        solver.step(model.state(), model.state(), model.control(), None, 0.001)
        with self.assertRaisesRegex(ValueError, "exceeds CollisionPipeline capacity"):
            solver.observables({SolverObservableFlags.CONTACT_F})


if __name__ == "__main__":
    unittest.main()
