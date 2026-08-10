# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Gradient tests for the differentiable elastic-foundation midsole."""

import json
import unittest
from pathlib import Path

import numpy as np
import warp as wp

import newton
from projects.digital_instron_v2 import core, dynamics, inverse_id, scenarios_diff, workflow
from projects.digital_instron_v2.dynamics import FoundationConfig
from projects.digital_instron_v2.dynamics_diff import DifferentiableMidsoleFoundation
from projects.digital_instron_v2.geometry import build_column_grid, load_mesh

MANIFEST = "DigitalInstron/manifest_v2.json"


@wp.kernel
def _final_height(body_q: wp.array[wp.transform], out: wp.array[wp.float32]):
    out[0] = wp.transform_get_translation(body_q[0])[2]


@wp.kernel
def _tangential_x(body_f: wp.array[wp.spatial_vector], out: wp.array[wp.float32]):
    out[0] = wp.spatial_top(body_f[0])[0]


class _Drop:
    """Differentiable drop of a unit carrier body onto the calibrated foam bed.

    Wraps the whole ``num_substeps`` substep loop in a single :class:`warp.Tape`
    so an objective on the final state can be differentiated w.r.t. the initial
    pose and the foam ``material_params``. The body starts already compressed
    (``z0 < 0``) so the rollout stays on the continuous branch of the contact,
    where the analytic gradient matches a finite difference; a drop from above
    the foam would cross the touchdown make/break event where the (correct)
    gradient is a subgradient that a finite difference will not match.
    """

    def __init__(self, device, num_substeps=60, config=None):
        geo = dynamics.build_foundation_geometry(MANIFEST)
        material = dynamics.load_fitted_material(MANIFEST)
        self.device = device
        self.nsub = num_substeps
        self.dt = 1.0 / 60.0 / 32.0

        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        builder.add_ground_plane()
        self.carrier = builder.add_body(mass=1.0, com=wp.vec3(0.0, 0.0, 0.0), inertia=wp.mat33(np.eye(3)))
        self.model = builder.finalize(requires_grad=True, device=device)
        self.solver = newton.solvers.SolverSemiImplicit(self.model, enable_tri_contact=False)
        self.states = [self.model.state() for _ in range(num_substeps + 1)]

        anchor = np.column_stack([geo.uv_m[:, 0], geo.uv_m[:, 1], geo.slack_m])
        self.foundation = DifferentiableMidsoleFoundation(
            anchor_local=anchor,
            z_free=geo.slack_m,
            rest_len=geo.slack_m,
            area=np.full(len(geo.slack_m), geo.area_m2),
            neighbors=geo.neighbors,
            spacing_m=geo.spacing_m,
            material=material,
            carrier_body=self.carrier,
            body_com=self.model.body_com,
            num_substeps=num_substeps,
            config=config or FoundationConfig(normal_damping=5.0),
            device=device,
        )
        self.loss = wp.zeros(1, dtype=wp.float32, device=device, requires_grad=True)

    def set_initial(self, z0, vx=0.0):
        q = np.zeros((1, 7), np.float32)
        q[0, 2] = z0
        q[0, 6] = 1.0
        self.states[0].body_q.assign(q)
        qd = np.zeros((1, 6), np.float32)
        qd[0, 0] = vx  # spatial_top (linear) x-velocity
        self.states[0].body_qd.assign(qd)

    def forward(self):
        for t in range(self.nsub):
            self.states[t].body_f.zero_()
            self.foundation.apply(self.states[t], t, self.dt)
            self.solver.step(self.states[t], self.states[t + 1], None, None, self.dt)
        wp.launch(_final_height, dim=1, inputs=[self.states[self.nsub].body_q, self.loss], device=self.device)
        return self.loss


class TestDifferentiableFoundationGradients(unittest.TestCase):
    def test_initial_height_gradient_matches_finite_difference(self):
        """Differentiate the final drop height w.r.t. the initial height and match a central difference."""
        device = wp.get_preferred_device()
        drop = _Drop(device, num_substeps=60)
        z0 = -0.003

        drop.set_initial(z0)
        tape = wp.Tape()
        with tape:
            loss = drop.forward()
        tape.backward(loss)
        analytic = float(drop.states[0].body_q.grad.numpy()[0][2])

        tape.zero()
        drop.foundation.zero_grad()
        eps = 2.0e-5
        drop.set_initial(z0 + eps)
        plus = float(drop.forward().numpy()[0])
        drop.set_initial(z0 - eps)
        minus = float(drop.forward().numpy()[0])
        numeric = (plus - minus) / (2.0 * eps)

        self.assertLess(abs(analytic - numeric) / (abs(numeric) + 1.0e-9), 1.0e-2)

    def test_material_parameter_gradients_match_finite_difference(self):
        """Differentiate the final drop height w.r.t. each foam material parameter and match central differences.

        Covers the full differentiable constitutive vector -- equilibrium shear
        modulus, Hyperfoam exponent, Maxwell overstress ratio, and Pasternak
        coupling -- with the viscoelastic overstress branch active.
        """
        device = wp.get_preferred_device()
        drop = _Drop(device, num_substeps=60)
        z0 = -0.003

        drop.set_initial(z0)
        tape = wp.Tape()
        with tape:
            loss = drop.forward()
        tape.backward(loss)
        analytic = drop.foundation.material_params.grad.numpy().copy()

        tape.zero()
        drop.foundation.zero_grad()
        base = drop.foundation.material_params.numpy().copy()
        numeric = np.empty_like(base)
        for k in range(len(base)):
            h = max(abs(base[k]) * 1.0e-3, 1.0e-6)
            perturbed = base.copy()
            perturbed[k] += h
            drop.foundation.material_params.assign(perturbed)
            drop.set_initial(z0)
            plus = float(drop.forward().numpy()[0])
            perturbed = base.copy()
            perturbed[k] -= h
            drop.foundation.material_params.assign(perturbed)
            drop.set_initial(z0)
            minus = float(drop.forward().numpy()[0])
            numeric[k] = (plus - minus) / (2.0 * h)
            drop.foundation.material_params.assign(base)

        # Vector-norm relative error: robust when one component's gradient is much
        # smaller than the others (the Hyperfoam exponent here), whose per-component
        # finite difference sits near the deterministic floor of the short rollout.
        rel = np.linalg.norm(analytic - numeric) / (np.linalg.norm(numeric) + 1.0e-30)
        self.assertLess(rel, 1.0e-2)

    def test_smooth_friction_respects_cone_and_is_differentiable(self):
        """Verify the smooth Coulomb friction stays inside the cone and has a finite-difference-matching gradient.

        A single substep with a planted, laterally sliding contact patch must
        produce a tangential force that opposes motion, is bounded by ``mu * fn``,
        and whose sensitivity to the slip velocity matches a central difference.
        """
        device = wp.get_preferred_device()
        config = FoundationConfig(normal_damping=5.0, mu=0.6)
        drop = _Drop(device, num_substeps=1, config=config)

        out = wp.zeros(1, dtype=wp.float32, device=device, requires_grad=True)
        vx0 = 0.03

        def tangential(vx):
            drop.set_initial(-0.004, vx=vx)
            drop.states[0].body_f.zero_()
            drop.foundation.apply(drop.states[0], 0, drop.dt)
            wp.launch(_tangential_x, dim=1, inputs=[drop.states[0].body_f, out], device=device)
            return float(out.numpy()[0])

        drop.set_initial(-0.004, vx=vx0)
        drop.states[0].body_f.zero_()
        tape = wp.Tape()
        with tape:
            drop.foundation.apply(drop.states[0], 0, drop.dt)
            wp.launch(_tangential_x, dim=1, inputs=[drop.states[0].body_f, out], device=device)
        tape.backward(out)
        analytic = float(drop.states[0].body_qd.grad.numpy()[0][0])

        ft_x = tangential(vx0)
        fn = drop.foundation.diagnostics()["normal_force_n"]
        self.assertGreater(fn, 0.0)
        self.assertLess(ft_x, 0.0)  # friction opposes +x motion
        self.assertLessEqual(abs(ft_x), config.mu * fn)  # inside the cone

        eps = 1.0e-3
        numeric = (tangential(vx0 + eps) - tangential(vx0 - eps)) / (2.0 * eps)
        self.assertLess(abs(analytic - numeric) / (abs(numeric) + 1.0e-9), 5.0e-2)

    def test_inverse_identification_recovers_stiffness(self):
        """Recover a perturbed foam stiffness from a target drop height using the analytic gradient.

        Gauss-Newton on the residual final_height(s) - target, stepping with the
        exact tape gradient d(final_height)/ds, must recover the true stiffness
        scale from a 30% error in a few iterations -- the differentiable-contact
        inverse-identification payoff.
        """
        device = wp.get_preferred_device()
        drop = _Drop(device, num_substeps=60)
        z0 = -0.003
        base = drop.foundation.material_params.numpy().copy()
        g_eq_ref = float(base[0])

        drop.set_initial(z0)
        target = float(drop.forward().numpy()[0])

        base[0] = 0.7 * g_eq_ref  # 30% stiffness error
        drop.foundation.material_params.assign(base)
        for _ in range(6):
            drop.set_initial(z0)
            tape = wp.Tape()
            with tape:
                z = drop.forward()
            tape.backward(z)
            drds = float(drop.foundation.material_params.grad.numpy()[0])
            residual = float(z.numpy()[0]) - target
            tape.zero()
            drop.foundation.zero_grad()
            if abs(residual) < 1.0e-7 or abs(drds) < 1.0e-15:
                break
            cur = drop.foundation.material_params.numpy().copy()
            cur[0] -= residual / drds
            drop.foundation.material_params.assign(cur)

        recovered = float(drop.foundation.material_params.numpy()[0])
        self.assertLess(abs(recovered - g_eq_ref) / g_eq_ref, 1.0e-2)


@wp.kernel
def _sum_force(force: wp.array[wp.float32], out: wp.array[wp.float32]):
    wp.atomic_add(out, 0, force[wp.tid()])


class TestForceMatchingInverseIdentification(unittest.TestCase):
    def _replay(self, device, samples=48):
        geometry = dynamics.build_foundation_geometry(MANIFEST)
        material = dynamics.load_fitted_material(MANIFEST)
        displacement = inverse_id._triangular_cycle(peak_m=0.006, samples=samples)
        replay = inverse_id.InstronReplay(displacement, 1.0 / 200.0, material, geometry, device=device)
        return replay, displacement, material, geometry

    def test_instron_force_gradient_matches_finite_difference(self):
        """Differentiate the total Instron reaction impulse w.r.t. the foam material and match central differences.

        Exercises the prescribed-displacement force readout
        (:func:`~projects.digital_instron_v2.inverse_id._accumulate_normal_force`)
        with the Maxwell recurrence active.
        """
        device = wp.get_preferred_device()
        replay, _, _, _ = self._replay(device)
        out = wp.zeros(1, dtype=wp.float32, device=device, requires_grad=True)

        replay.zero_grad()
        out.zero_()
        tape = wp.Tape()
        with tape:
            force = replay.forward()
            wp.launch(_sum_force, dim=replay.nsteps, inputs=[force, out], device=device)
        tape.backward(out)
        analytic = replay.material_params.grad.numpy().copy()

        tape.zero()
        replay.zero_grad()
        base = replay.material_params.numpy().copy()
        numeric = np.empty_like(base)
        for k in range(len(base)):
            h = max(abs(base[k]) * 1.0e-3, 1.0e-6)
            perturbed = base.copy()
            perturbed[k] += h
            replay.set_material(perturbed)
            plus = float(replay.forward().numpy().sum())
            perturbed = base.copy()
            perturbed[k] -= h
            replay.set_material(perturbed)
            minus = float(replay.forward().numpy().sum())
            numeric[k] = (plus - minus) / (2.0 * h)
            replay.set_material(base)

        rel = np.linalg.norm(analytic - numeric) / (np.linalg.norm(numeric) + 1.0e-30)
        self.assertLess(rel, 1.0e-2)

    def test_force_matching_recovers_elastic_material(self):
        """Recover perturbed elastic foam parameters from a synthetic force curve with the overstress pinned.

        A single-rate load/unload cycle constrains stiffness and Maxwell overstress
        along a strongly correlated direction, so the overstress ratio (obtained
        separately from a relaxation test) is held fixed while the equilibrium
        modulus, Hyperfoam exponent, and Pasternak coupling are fit to the curve.
        """
        device = wp.get_preferred_device()
        replay, displacement, material, geometry = self._replay(device)
        target = replay.forward().numpy().copy()

        result = inverse_id.fit_material_to_force_curve(
            target,
            displacement,
            1.0 / 200.0,
            material,
            geometry,
            scale0=np.array([1.3, 0.85, 1.0, 0.7], np.float32),
            fit_mask=np.array([1.0, 1.0, 0.0, 1.0], np.float32),
            iterations=400,
            device=device,
        )
        self.assertLess(abs(result.scale[0] - 1.0), 2.0e-2)  # g_eq
        self.assertLess(abs(result.scale[3] - 1.0), 2.0e-2)  # pasternak
        residual = float(np.sqrt(np.mean((result.force - target) ** 2)))
        self.assertLess(residual / target.max(), 1.0e-3)


def _synthetic_trial(frames: int = 80, columns: int = 24, seed: int = 0):
    """Build a core.Trial with a shaped (per-column) triangular compression cycle."""
    rng = np.random.default_rng(seed)
    slack = np.full(columns, 0.03, np.float32)  # 30 mm foam columns
    peak_compression = (0.004 + 0.006 * rng.random(columns)).astype(np.float32)  # 4-10 mm, varies per column
    half = frames // 2
    ramp = np.concatenate([np.linspace(0.0, 1.0, half, endpoint=False), np.linspace(1.0, 0.0, frames - half)]).astype(
        np.float32
    )
    compression = ramp[:, None] * peak_compression[None, :]
    lengths = slack[None, :] - compression
    dt = np.full(frames, 1.0 / 200.0, np.float64)
    # A smooth, both-signed Pasternak Laplacian field so the coupling and pressure floor are exercised.
    laplacian = (200.0 * (compression - compression.mean(axis=1, keepdims=True))).astype(np.float32)
    displacement = (ramp * float(peak_compression.max())).astype(np.float32)
    force = np.zeros(frames, np.float32)
    return core.Trial(
        "synthetic", slack, float(np.pi * 0.011**2 / columns), lengths, dt, force, displacement, laplacian
    )


class TestMeasuredTrialForceMatching(unittest.TestCase):
    @staticmethod
    def _load_measured_trials():
        base = Path("DigitalInstron")
        config = json.loads((base / "manifest_v2.json").read_text())
        midsole = load_mesh(str(base / config["midsole_mesh"]), 0.001)
        grid = build_column_grid(midsole, config["grid"]["coarse_spacing_m"])
        trials, _, _ = workflow.prepare_trials(base, config, grid, midsole)
        material = dynamics.load_fitted_material(MANIFEST)
        return trials, material

    def test_predictor_reproduces_core_predict(self):
        """Match the quasi-static core.predict force curve on a shaped-indenter trial to machine precision.

        The differentiable predictor evaluates the periodic generalized-Maxwell
        overstress fixed point with an explicit transient-then-cycle pass, so it
        must reproduce the closed-form periodic branch of core.predict.
        """
        device = wp.get_preferred_device()
        material = dynamics.load_fitted_material(MANIFEST)
        trial = _synthetic_trial()
        predicted = inverse_id.DifferentiableTrial(trial, material, device=device).forward().numpy()
        reference = core.predict(trial, material)
        peak = max(float(np.max(np.abs(reference))), 1.0e-9)
        self.assertLess(float(np.max(np.abs(predicted - reference))) / peak, 1.0e-3)

    def test_material_gradient_matches_finite_difference(self):
        """Differentiate a shaped-indenter reaction impulse w.r.t. the foam material and match central differences.

        Covers the full ``[g_eq, alpha, overstress, pasternak]`` vector through the
        two-pass periodic recurrence with the viscoelastic branch active.
        """
        device = wp.get_preferred_device()
        material = dynamics.load_fitted_material(MANIFEST)
        trial = _synthetic_trial()
        predictor = inverse_id.DifferentiableTrial(trial, material, device=device)
        target = wp.zeros(predictor.frame_count, dtype=wp.float32, device=device)  # loss = sum(force^2)
        loss = wp.zeros(1, dtype=wp.float32, device=device, requires_grad=True)

        predictor.zero_grad()
        loss.zero_()
        tape = wp.Tape()
        with tape:
            force = predictor.forward()
            wp.launch(
                inverse_id._weighted_force_mse,
                dim=predictor.frame_count,
                inputs=[force, target, 1.0, loss],
                device=device,
            )
        tape.backward(loss)
        analytic = predictor.material_params.grad.numpy().copy()

        tape.zero()
        predictor.zero_grad()
        base = predictor.material_params.numpy().copy()
        numeric = np.empty_like(base)
        for k in range(len(base)):
            h = max(abs(base[k]) * 1.0e-3, 1.0e-6)
            perturbed = base.copy()
            perturbed[k] += h
            predictor.set_material(perturbed)
            plus = float(np.sum(predictor.forward().numpy().astype(np.float64) ** 2))
            perturbed = base.copy()
            perturbed[k] -= h
            predictor.set_material(perturbed)
            minus = float(np.sum(predictor.forward().numpy().astype(np.float64) ** 2))
            numeric[k] = (plus - minus) / (2.0 * h)
            predictor.set_material(base)

        rel = np.linalg.norm(analytic - numeric) / (np.linalg.norm(numeric) + 1.0e-30)
        self.assertLess(rel, 1.0e-2)

    @unittest.skipUnless(Path(MANIFEST).exists(), "requires the DigitalInstron dataset")
    def test_reproduces_core_predict_on_measured_trials(self):
        """Reproduce core.predict on both measured fixtures (spherical punch and shoe last)."""
        device = wp.get_preferred_device()
        trials, material = self._load_measured_trials()
        for trial in trials:
            predicted = inverse_id.DifferentiableTrial(trial, material, device=device).forward().numpy()
            reference = core.predict(trial, material)
            peak = max(float(np.max(np.abs(reference))), 1.0e-9)
            self.assertLess(float(np.max(np.abs(predicted - reference))) / peak, 1.0e-3)

    @unittest.skipUnless(Path(MANIFEST).exists(), "requires the DigitalInstron dataset")
    def test_joint_fit_reproduces_shipped_calibration(self):
        """Recover the shipped calibration by peak-normalized joint fitting to the measured trials.

        Starting from the shipped material (scale = 1), the differentiable joint
        fit must stay at that optimum: the fitted scale barely moves and the final
        per-trial force RMS reproduces core.metrics for the shipped material,
        cross-validating the exact gradients against the production scipy fit.
        """
        device = wp.get_preferred_device()
        trials, material = self._load_measured_trials()
        result = inverse_id.fit_material_to_trials(trials, material, iterations=250, learning_rate=0.01, device=device)
        self.assertTrue(np.all(np.abs(result.scale - 1.0) < 0.05))
        for trial in trials:
            expected = core.metrics(trial.force_n, core.predict(trial, material), trial.displacement_m)[
                "force_rmse_relative"
            ]
            self.assertAlmostEqual(result.rms_relative[trial.name], expected, delta=2.0e-3)


class TestDifferentiableGaitScenarios(unittest.TestCase):
    @staticmethod
    def _press_driver(device, nsteps=160):
        geo = dynamics.build_foundation_geometry(MANIFEST)
        material = dynamics.load_fitted_material(MANIFEST)
        center = geo.uv_m.mean(axis=0)
        press_z = float(geo.z_free_m.mean()) - 0.02  # penetrate the ground plane for continuous contact
        pose = np.array([center[0], center[1], press_z, 0.0, 0.0, 0.0, 1.0], np.float32)
        targets = np.tile(pose, (nsteps, 1))
        velocities = np.zeros((nsteps, 6), np.float32)
        dt = (1.0 / 60.0) / 128.0
        return scenarios_diff.DifferentiableAttached(geo, material, targets, velocities, dt, device=device)

    def test_stride_reproduces_shipped_forward_model(self):
        """Reproduce the shipped MidsoleFoundation GRF over a kinematic heel-to-toe stride to sub-milli-newton."""
        device = wp.get_preferred_device()
        geo = dynamics.build_foundation_geometry(MANIFEST)
        material = dynamics.load_fitted_material(MANIFEST)
        stride = scenarios_diff.DifferentiableStride(geo, material, device=device)
        diff = stride.forward().numpy()
        reference = stride.reference_grf()
        self.assertGreater(float(np.max(reference)), 100.0)  # the stride actually loads the bed
        self.assertLess(float(np.max(np.abs(diff - reference))), 1.0e-2)

    def test_stride_impulse_gradient_matches_finite_difference(self):
        """Differentiate the kinematic stride GRF impulse w.r.t. the equilibrium modulus and match a central difference."""
        device = wp.get_preferred_device()
        geo = dynamics.build_foundation_geometry(MANIFEST)
        material = dynamics.load_fitted_material(MANIFEST)
        stride = scenarios_diff.DifferentiableStride(geo, material, device=device)
        loss = wp.zeros(1, dtype=wp.float32, device=device, requires_grad=True)

        stride.zero_grad()
        loss.zero_()
        tape = wp.Tape()
        with tape:
            grf = stride.forward()
            wp.launch(scenarios_diff._reduce_sum, dim=stride.substep_count, inputs=[grf, loss], device=device)
        tape.backward(loss)
        analytic = float(stride.material_params.grad.numpy()[0])

        tape.zero()
        stride.zero_grad()
        base = stride.material_params.numpy().copy()
        h = base[0] * 1.0e-3

        def impulse(g_eq):
            perturbed = base.copy()
            perturbed[0] = g_eq
            stride.material_params.assign(perturbed.astype(np.float32))
            return float(stride.forward().numpy().astype(np.float64).sum())

        numeric = (impulse(base[0] + h) - impulse(base[0] - h)) / (2.0 * h)
        stride.material_params.assign(base.astype(np.float32))
        self.assertLess(abs(analytic - numeric) / (abs(numeric) + 1.0e-30), 1.0e-2)

    def test_attached_press_gradient_matches_finite_difference(self):
        """Differentiate the fully dynamic press GRF impulse w.r.t. the equilibrium modulus under continuous contact.

        Drives the shoe with a constant target pose so the contact patch stays
        engaged for the whole rollout; the gradient through the PD upper, the
        semi-implicit solver, and the smooth-friction foundation then matches a
        central difference (a stride roll would cross contact make/break events
        where the correct subgradient differs from a finite difference).
        """
        device = wp.get_preferred_device()
        driver = self._press_driver(device, nsteps=160)
        loss = wp.zeros(1, dtype=wp.float32, device=device, requires_grad=True)

        driver.zero_grad()
        loss.zero_()
        tape = wp.Tape()
        with tape:
            grf = driver.forward()
            wp.launch(scenarios_diff._reduce_sum, dim=driver.substep_count, inputs=[grf, loss], device=device)
        tape.backward(loss)
        analytic = float(driver.material_params.grad.numpy()[0])

        tape.zero()
        driver.zero_grad()
        base = driver.material_params.numpy().copy()
        h = base[0] * 1.0e-3

        def impulse(g_eq):
            perturbed = base.copy()
            perturbed[0] = g_eq
            driver.material_params.assign(perturbed.astype(np.float32))
            return float(driver.forward().numpy().astype(np.float64).sum())

        numeric = (impulse(base[0] + h) - impulse(base[0] - h)) / (2.0 * h)
        driver.material_params.assign(base.astype(np.float32))
        self.assertLess(abs(analytic - numeric) / (abs(numeric) + 1.0e-30), 2.0e-2)

    def test_attached_forward_produces_valid_grf(self):
        """A fully dynamic attached press yields a finite, non-negative GRF with active contact."""
        device = wp.get_preferred_device()
        driver = self._press_driver(device, nsteps=96)
        grf = driver.forward().numpy()
        self.assertTrue(np.all(np.isfinite(grf)))
        self.assertGreaterEqual(float(np.min(grf)), 0.0)
        self.assertGreater(float(np.max(grf)), 100.0)


class TestDifferentiableFriction(unittest.TestCase):
    def test_slide_friction_gradient_matches_finite_difference(self):
        """Differentiate the lateral drag impulse of a constant slide w.r.t. the friction coefficient.

        Drives the foam bed at a fixed penetration and constant lateral speed so
        the contact patch stays engaged and the tangential velocity never crosses
        zero; the gradient of the accumulated drag with respect to ``mu`` then
        matches a central difference.
        """
        device = wp.get_preferred_device()
        geo = dynamics.build_foundation_geometry(MANIFEST)
        material = dynamics.load_fitted_material(MANIFEST)
        slide = scenarios_diff.DifferentiableSlide(
            geo, material, depth_m=0.012, slide_speed_m_s=0.25, substeps=32, device=device
        )
        loss = wp.zeros(1, dtype=wp.float32, device=device, requires_grad=True)

        slide.zero_grad()
        loss.zero_()
        tape = wp.Tape()
        with tape:
            shear = slide.forward()
            wp.launch(scenarios_diff._drag_impulse, dim=slide.substep_count, inputs=[shear, loss], device=device)
        tape.backward(loss)
        analytic = float(slide.friction_params.grad.numpy()[0])

        tape.zero()
        slide.zero_grad()
        base = float(slide.friction_params.numpy()[0])
        h = base * 1.0e-3

        def drag(mu):
            slide.friction_params.assign(np.array([mu], np.float32))
            return -float(slide.forward().numpy()[:, 0].astype(np.float64).sum())

        numeric = (drag(base + h) - drag(base - h)) / (2.0 * h)
        slide.friction_params.assign(np.array([base], np.float32))
        self.assertGreater(analytic, 0.0)  # more friction => more drag
        self.assertLess(abs(analytic - numeric) / (abs(numeric) + 1.0e-30), 5.0e-3)

    def test_slide_recovers_friction_coefficient(self):
        """Recover the Coulomb friction coefficient from a lateral-force target using the analytic gradient.

        The smooth-friction drag is exactly linear in ``mu`` at fixed kinematics, so
        a single gradient-informed step from a wrong guess recovers the coefficient
        that reproduces a measured drag impulse.
        """
        device = wp.get_preferred_device()
        geo = dynamics.build_foundation_geometry(MANIFEST)
        material = dynamics.load_fitted_material(MANIFEST)
        slide = scenarios_diff.DifferentiableSlide(
            geo, material, depth_m=0.012, slide_speed_m_s=0.25, substeps=32, device=device
        )
        mu_ref = float(slide.friction_params.numpy()[0])
        target = -float(slide.forward().numpy()[:, 0].astype(np.float64).sum())

        slide.friction_params.assign(np.array([mu_ref * 0.4], np.float32))
        loss = wp.zeros(1, dtype=wp.float32, device=device, requires_grad=True)
        slide.zero_grad()
        loss.zero_()
        tape = wp.Tape()
        with tape:
            shear = slide.forward()
            wp.launch(scenarios_diff._drag_impulse, dim=slide.substep_count, inputs=[shear, loss], device=device)
        tape.backward(loss)
        sensitivity = float(slide.friction_params.grad.numpy()[0])  # d(drag)/d(mu), constant
        drag_guess = float(loss.numpy()[0])
        recovered = float(slide.friction_params.numpy()[0]) + (target - drag_guess) / sensitivity
        self.assertLess(abs(recovered - mu_ref) / mu_ref, 5.0e-3)

    def test_attached_records_lateral_shear(self):
        """The dynamic attached rollout records a finite, non-trivial lateral shear on the tape."""
        device = wp.get_preferred_device()
        geo = dynamics.build_foundation_geometry(MANIFEST)
        material = dynamics.load_fitted_material(MANIFEST)
        com_z = float(geo.z_free_m.mean())
        poses, velocities, dt = scenarios_diff.stride_trajectory(
            geo,
            com_z,
            period_s=0.15,
            peak_depth_m=0.03,
            pitch_deg=8.0,
            roll_fraction=0.15,
            frame_dt=1.0 / 60.0,
            substeps=64,
            with_velocity=True,
        )
        attached = scenarios_diff.DifferentiableAttached(geo, material, poses, velocities, dt, device=device)
        attached.forward()
        shear = attached.shear.numpy()
        self.assertEqual(shear.shape, (attached.substep_count, 2))
        self.assertTrue(np.all(np.isfinite(shear)))
        self.assertGreater(float(np.max(np.abs(shear))), 1.0)
        self.assertEqual(float(attached.friction_params.numpy()[0]), attached.config.mu)


if __name__ == "__main__":
    unittest.main()
