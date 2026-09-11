# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Verify per-interface ADMM parameters through solver motion."""

import unittest

import numpy as np
import warp as wp

import newton
from newton.solvers import SolverSemiImplicit, SolverVBD, SolverXPBD
from newton.solvers.experimental.coupled import SolverCoupled, SolverCoupledADMM
from newton.tests.unittest_utils import get_cuda_test_devices


class TestAdmmInterfaceParameters(unittest.TestCase):
    """Exercise attachment and contact groups with independent interface settings."""

    device = "cpu"
    kinds = ("rr", "rp", "pp", "ball", "fixed", "revolute", "attachment", "mixed")

    def _scene(self, kind, params=None):
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        endpoints = []
        joint = kind in ("ball", "fixed", "revolute")
        endpoint_kinds = "rr" if joint else "rp" if kind in ("attachment", "mixed") else kind
        for index, endpoint in enumerate(endpoint_kinds):
            pos = (0.08 * index, 0.0, 0.0)
            if endpoint == "r":
                rotation = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), 0.2 * index) if joint else wp.quat_identity()
                body = builder.add_body(xform=wp.transform(pos, rotation))
                builder.add_shape_sphere(body, radius=0.05, cfg=newton.ModelBuilder.ShapeConfig(density=1000.0, mu=0.0))
                endpoints.append({"bodies": [body]})
            else:
                particle = builder.add_particle(pos=pos, vel=wp.vec3(), mass=1.0, radius=0.05)
                endpoints.append({"particles": [particle]})
        if joint:
            if params is not None:
                SolverCoupledADMM.register_custom_attributes(builder)
            kwargs = {"friction": 0.1} if kind in ("ball", "revolute") else {}
            if params is not None:
                kwargs["custom_attributes"] = {
                    "coupling:joint_" + name: value for name, value in params.items() if value is not None
                }
            getattr(builder, "add_joint_" + kind)(parent=0, child=1, collision_filter_parent=False, **kwargs)
            # Exercise both hinge friction and constrained angular directions.
            builder.body_qd[1] = wp.spatial_vector(0.0, 0.0, 0.0, 1.0, 0.5, 0.0)
        elif kind in ("attachment", "mixed"):
            SolverCoupledADMM.add_body_particle_attachment(
                builder, 0, 0, stiffness=5000.0, damping=2.0, **(params or {})
            )
        model = builder.finalize(device=self.device)
        return model, endpoints

    def _solver(self, kind, global_params, pair_params, reverse=False):
        model, endpoints = self._scene(kind, pair_params)
        names = ("a", "b")
        entries = [
            SolverCoupled.Entry(name, lambda view: SolverSemiImplicit(view, enable_tri_contact=False), **owned)
            for name, owned in zip(names, endpoints, strict=True)
        ]
        pair = SolverCoupledADMM.ContactPair(*(reversed(names) if reverse else names), **(pair_params or {}))
        solver = SolverCoupledADMM(
            model,
            entries,
            SolverCoupledADMM.Config(
                iterations=3,
                contact_pairs=[pair] if kind in ("rr", "rp", "pp", "mixed") else [],
                joint_proximal_bodies=False,
                **global_params,
            ),
        )
        return model, solver

    def _run(self, model, solver, *, capture=False):
        state, other = model.state(), model.state()
        if capture:
            solver.step(state, other, model.control(), None, 1.0 / 240.0)
            state, other = model.state(), model.state()
            solver.reset(state, flags=0)
            with wp.ScopedDevice(model.device), wp.ScopedCapture() as captured:
                solver.step(state, other, model.control(), None, 1.0 / 240.0)
            wp.capture_launch(captured.graph)
            state = other
        else:
            for _ in range(3):
                state.clear_forces()
                solver.step(state, other, model.control(), None, 1.0 / 240.0)
                state, other = other, state
        arrays = [
            getattr(state, name).numpy()
            for name in ("body_q", "body_qd", "particle_q", "particle_qd")
            if getattr(state, name) is not None
        ]
        return np.concatenate([array.flatten() for array in arrays])

    def test_explicit_inheritance_preserves_motion(self):
        """Match omitted, inherited, partial, and explicit interface parameters."""
        effective = {"rho": 0.4, "gamma": 0.3, "baumgarte": 0.1}
        for kind in self.kinds:
            with self.subTest(kind=kind):
                model, solver = self._solver(kind, effective, None)
                reference = self._run(model, solver)
                for params in (
                    {},
                    {"rho": None, "gamma": None, "baumgarte": None},
                    {"rho": effective["rho"]},
                    {"gamma": effective["gamma"]},
                    {"baumgarte": effective["baumgarte"]},
                    effective,
                ):
                    other_model, other_solver = self._solver(kind, effective, params)
                    np.testing.assert_allclose(self._run(other_model, other_solver), reference, atol=1e-7, rtol=1e-6)

    def test_overrides_replace_globals_for_all_interface_kinds(self):
        """Use pair values in correction, projection, force, dual, and proximal paths."""
        effective = {"rho": 0.4, "gamma": 0.3, "baumgarte": 0.1}
        for kind in self.kinds:
            with self.subTest(kind=kind):
                model, solver = self._solver(kind, effective, {})
                reference = self._run(model, solver)
                override_model, override_solver = self._solver(
                    kind,
                    {"rho": 0.1, "gamma": 0.0, "baumgarte": 0.0},
                    effective,
                    reverse=True,
                )
                np.testing.assert_allclose(self._run(override_model, override_solver), reference, atol=1e-7, rtol=1e-6)
                override_solver.reset(override_model.state(), flags=0)
                np.testing.assert_allclose(self._run(override_model, override_solver), reference, atol=1e-7, rtol=1e-6)

    def test_zero_gamma_disables_interface_proximal_terms(self):
        """Honor explicit zero gamma despite positive global gamma."""
        for kind in self.kinds:
            with self.subTest(kind=kind):
                model, solver = self._solver(kind, {"rho": 0.4, "gamma": 0.0, "baumgarte": 0.1}, {})
                reference = self._run(model, solver)
                model, solver = self._solver(
                    kind,
                    {"rho": 0.4, "gamma": 0.5, "baumgarte": 0.1},
                    {"gamma": 0.0},
                )
                np.testing.assert_allclose(self._run(model, solver), reference, atol=1e-7, rtol=1e-6)

    def test_two_interfaces_add_independent_proximal_terms(self):
        """Sum distinct interface gamma-rho contributions on their shared endpoint."""
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        for x in (-0.08, 0.0, 0.08):
            builder.add_particle(pos=(x, 0.0, 0.0), vel=wp.vec3(), mass=1.0, radius=0.05)
        model = builder.finalize(device=self.device)
        entries = [
            SolverCoupled.Entry(name, lambda view: SolverSemiImplicit(view, enable_tri_contact=False), particles=[i])
            for i, name in enumerate(("a", "b", "c"))
        ]
        solver = SolverCoupledADMM(
            model,
            entries,
            SolverCoupledADMM.Config(
                gamma=0.0,
                contact_pairs=[
                    SolverCoupledADMM.ContactPair("a", "b", rho=0.2, gamma=0.2, baumgarte=0.1),
                    SolverCoupledADMM.ContactPair("c", "b", rho=0.8, gamma=0.3, baumgarte=0.2),
                ],
            ),
        )
        solver._refresh_collision_contact_groups(model.state())
        solver._refresh_admm_proximal_masks()
        masses = {
            name: buf.particle_proximal_mass.numpy()[index]
            for index, (name, buf) in enumerate(solver._admm_buffers.items())
        }
        self.assertGreater(masses["a"], 0.0)
        self.assertAlmostEqual(masses["c"] / masses["a"], 6.0, places=5)
        self.assertAlmostEqual(masses["b"], masses["a"] + masses["c"], places=5)
        solver.sync_entry_states(model.state())
        solver._admm_begin_step(1.0 / 240.0)
        targets = [group.u_min.numpy()[0] for group in solver._admm_dynamic_pp_contact_groups]
        self.assertAlmostEqual(targets[1] / targets[0], 2.0, places=5)

    def test_invalid_overrides_fail_validation(self):
        """Reject nonfinite, negative, and zero-penalty overrides before solver setup."""
        for field, values in (
            ("rho", (0.0, -1.0, float("nan"), float("inf"))),
            ("gamma", (-1.0, float("nan"), float("inf"))),
            ("baumgarte", (-1.0, float("nan"), float("inf"))),
        ):
            for value in values:
                with (
                    self.subTest(field=field, value=value),
                    self.assertRaisesRegex(ValueError, "ContactPair " + field),
                ):
                    SolverCoupledADMM._validate_config(
                        SolverCoupledADMM.Config(
                            contact_pairs=[SolverCoupledADMM.ContactPair("a", "b", **{field: value})]
                        )
                    )

    def test_invalid_interface_entries_fail_validation(self):
        """Reject unknown, identical, duplicate, and invalid interface definitions."""
        params = SolverCoupledADMM.ContactPair
        cases = (
            ([params("a", "a")], "distinct"),
            ([params("missing", "b")], "Unknown"),
            ([params("a", "missing")], "Unknown"),
            ([params("a", "b"), params("b", "a")], "Duplicate"),
            ([{"source": "a", "destination": "b"}], "ContactPair instances"),
        )
        for overrides, message in cases:
            with self.subTest(overrides=overrides), self.assertRaisesRegex(ValueError, message):
                model, endpoints = self._scene("pp")
                entries = [
                    SolverCoupled.Entry(name, SolverSemiImplicit, **owned)
                    for name, owned in zip(("a", "b"), endpoints, strict=True)
                ]
                SolverCoupledADMM(model, entries, SolverCoupledADMM.Config(contact_pairs=overrides))

    def test_attachment_overrides_do_not_enable_contacts(self):
        """Keep overlapping endpoints contact-free when only parameters are configured."""
        for kind in ("ball", "fixed", "revolute", "attachment"):
            with self.subTest(kind=kind):
                model, solver = self._solver(kind, {"gamma": 0.0}, {"rho": 0.4, "gamma": 0.3, "baumgarte": 0.1})
                self.assertTrue(np.all(np.isfinite(self._run(model, solver))))
                self.assertEqual(solver.collision_contact_count_max, 0)

    def test_override_preserves_other_interface_motion(self):
        """Keep a second interface on the global parameters when overriding the first."""
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        for x in (0.0, 0.08, 1.0, 1.08):
            builder.add_particle(pos=(x, 0.0, 0.0), vel=wp.vec3(), mass=1.0, radius=0.05)
        model = builder.finalize(device=self.device)
        entries = [
            SolverCoupled.Entry(name, SolverSemiImplicit, particles=[i]) for i, name in enumerate(("a", "b", "c", "d"))
        ]
        global_params = {"rho": 0.4, "gamma": 0.3, "baumgarte": 0.1}
        solver = SolverCoupledADMM(
            model,
            entries,
            SolverCoupledADMM.Config(
                iterations=3,
                **global_params,
                contact_pairs=[
                    SolverCoupledADMM.ContactPair("a", "b", gamma=0.0, baumgarte=0.0),
                    SolverCoupledADMM.ContactPair("c", "d"),
                ],
            ),
        )
        actual = self._run(model, solver).reshape(2, 4, 3)
        np.testing.assert_allclose(actual[0, :2], ((0.0, 0.0, 0.0), (0.08, 0.0, 0.0)), atol=1e-7)
        np.testing.assert_allclose(actual[1, :2], 0.0, atol=1e-7)
        reference_model, reference_solver = self._solver("pp", global_params, None)
        reference = self._run(reference_model, reference_solver).reshape(2, 2, 3)
        actual[0, 2:, 0] -= 1.0
        np.testing.assert_allclose(actual[:, 2:], reference, atol=1e-6, rtol=1e-5)

    def test_contact_overrides_do_not_change_joint_or_attachment_motion(self):
        """Keep contact tuning separate from constraints sharing the same entries."""
        effective = {"rho": 0.4, "gamma": 0.3, "baumgarte": 0.1}
        for kind in ("ball", "fixed", "revolute", "attachment"):
            results = []
            for contact_params in ({}, {"rho": 0.8, "gamma": 0.0, "baumgarte": 0.0}):
                model, endpoints = self._scene(kind)
                # Disable collision geometry so only the joint or attachment drives motion.
                model.shape_flags.zero_()
                model.shape_contact_pairs = wp.zeros(0, dtype=wp.vec2i, device=model.device)
                entries = [
                    SolverCoupled.Entry(name, SolverSemiImplicit, **owned)
                    for name, owned in zip(("a", "b"), endpoints, strict=True)
                ]
                solver = SolverCoupledADMM(
                    model,
                    entries,
                    SolverCoupledADMM.Config(
                        iterations=3,
                        joint_proximal_bodies=False,
                        **effective,
                        contact_pairs=[SolverCoupledADMM.ContactPair("a", "b", **contact_params)],
                    ),
                )
                results.append(self._run(model, solver))
            with self.subTest(kind=kind):
                np.testing.assert_allclose(*results, atol=1e-7, rtol=1e-6)

    def test_same_owner_pair_preserves_individual_constraint_parameters(self):
        """Match separately tuned constraints to independent simulations with their global settings."""
        settings = [{"rho": 0.2, "gamma": 0.0, "baumgarte": 0.1}, {"rho": 0.8, "gamma": 0.3, "baumgarte": 0.2}]

        def simulate(kind, parameters, defaults):
            builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
            SolverCoupledADMM.register_custom_attributes(builder)
            bodies_a, bodies_b, particles = [], [], []
            for index, params in enumerate(parameters):
                body = builder.add_body(xform=wp.transform((0.0, float(index), 0.0), wp.quat_identity()))
                builder.add_shape_sphere(body, radius=0.05)
                bodies_a.append(body)
                if kind == "attachment":
                    particle = builder.add_particle(pos=(0.08, float(index), 0.0), vel=wp.vec3(), mass=1.0)
                    particles.append(particle)
                    SolverCoupledADMM.add_body_particle_attachment(builder, body, particle, **params)
                else:
                    child = builder.add_body(xform=wp.transform((0.08, float(index), 0.0), wp.quat_identity()))
                    builder.add_shape_sphere(child, radius=0.05)
                    bodies_b.append(child)
                    kwargs = {"friction": 0.1} if kind in ("ball", "revolute") else {}
                    getattr(builder, "add_joint_" + kind)(
                        body,
                        child,
                        custom_attributes={"coupling:joint_" + name: value for name, value in params.items()},
                        **kwargs,
                    )
                    builder.body_qd[child] = wp.spatial_vector(0.0, 0.0, 0.0, 1.0, 0.5, 0.0)
            model = builder.finalize(device=self.device)
            solver = SolverCoupledADMM(
                model,
                [
                    SolverCoupled.Entry("a", SolverSemiImplicit, bodies=bodies_a),
                    SolverCoupled.Entry("b", SolverSemiImplicit, bodies=bodies_b, particles=particles),
                ],
                SolverCoupledADMM.Config(iterations=3, joint_proximal_bodies=False, **defaults),
            )
            result = self._run(model, solver)
            n = model.body_count
            result[: n * 7].reshape(n, 7)[:, 1] -= np.repeat(
                np.arange(len(parameters)), 1 if kind == "attachment" else 2
            )
            if model.particle_count:
                result[n * 13 : n * 13 + model.particle_count * 3].reshape(-1, 3)[:, 1] -= np.arange(len(parameters))
            return np.split(result, [n * 7, n * 13, n * 13 + model.particle_count * 3])

        for kind in ("ball", "fixed", "revolute", "attachment"):
            with self.subTest(kind=kind):
                actual = simulate(kind, settings, {"rho": 0.1, "gamma": 0.0, "baumgarte": 0.0})
                references = [simulate(kind, [{}], params) for params in settings]
                for field, values in enumerate(actual):
                    np.testing.assert_allclose(
                        values, np.concatenate([ref[field] for ref in references]), atol=1e-7, rtol=1e-6
                    )

    def test_invalid_authored_parameters_fail_validation(self):
        """Validate custom-attribute overrides as well as attachment helper arguments."""
        for kind in ("ball", "attachment"):
            for name, value in (
                ("rho", 0.0),
                ("rho", -2.0),
                ("gamma", -2.0),
                ("baumgarte", -2.0),
                ("rho", float("inf")),
                ("gamma", float("nan")),
                ("baumgarte", float("inf")),
            ):
                with self.subTest(kind=kind, name=name, value=value), self.assertRaisesRegex(ValueError, name):
                    model, endpoints = self._scene(kind, {})
                    getattr(model.coupling, ("joint_" if kind == "ball" else "body_particle_attachment_") + name).fill_(
                        value
                    )
                    entries = [
                        SolverCoupled.Entry(entry, SolverSemiImplicit, **owned)
                        for entry, owned in zip(("a", "b"), endpoints, strict=True)
                    ]
                    SolverCoupledADMM(model, entries, SolverCoupledADMM.Config())
        for name in ("rho", "gamma", "baumgarte"):
            with self.subTest(helper=name), self.assertRaisesRegex(ValueError, name):
                self._scene("attachment", {name: -1.0})

    def test_cuda_inertial_refresh_with_authored_gamma(self):
        """Refresh XPBD and VBD inertias when only a contact or joint enables gamma."""
        devices = get_cuda_test_devices()
        if not devices:
            self.skipTest("CUDA device required")
        for device in devices:
            for kind in ("contact", "joint"):
                with self.subTest(device=device, kind=kind):
                    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
                    for x in (0.0, 0.08):
                        body = builder.add_body(xform=wp.transform((x, 0.0, 0.0), wp.quat_identity()))
                        builder.add_shape_box(body, hx=0.05, hy=0.05, hz=0.05)
                    if kind == "joint":
                        SolverCoupledADMM.register_custom_attributes(builder)
                        builder.add_joint_fixed(0, 1, custom_attributes={"coupling:joint_gamma": 0.3})
                    builder.color()
                    model = builder.finalize(device=device)
                    solver = SolverCoupledADMM(
                        model,
                        [
                            SolverCoupled.Entry("a", lambda view: SolverXPBD(view, iterations=1), bodies=[0]),
                            SolverCoupled.Entry(
                                "b", lambda view: SolverVBD(view, iterations=1, rigid_compliant_alm=True), bodies=[1]
                            ),
                        ],
                        SolverCoupledADMM.Config(
                            gamma=0.0,
                            joint_proximal_bodies=False,
                            contact_pairs=[SolverCoupledADMM.ContactPair("a", "b", gamma=0.3)]
                            if kind == "contact"
                            else [],
                        ),
                    )
                    self.assertTrue(np.all(np.isfinite(self._run(model, solver, capture=True))))
                    self.assertTrue(
                        any(np.any(buf.body_proximal_mass.numpy() > 0.0) for buf in solver._admm_buffers.values())
                    )

    def test_cuda_capture_with_interface_only_gamma(self):
        """Capture overrides when only the interface enables proximal terms."""
        devices = get_cuda_test_devices()
        if not devices:
            self.skipTest("CUDA device required")
        for device in devices:
            self.device = device
            for kind in self.kinds:
                with self.subTest(device=device, kind=kind):
                    model, solver = self._solver(
                        kind,
                        {"rho": 0.1, "gamma": 0.0, "baumgarte": 0.0},
                        {"rho": 0.4, "gamma": 0.3, "baumgarte": 0.1},
                    )
                    self.assertTrue(np.all(np.isfinite(self._run(model, solver, capture=True))))


if __name__ == "__main__":
    unittest.main()
