# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Numerical checks for the coupled spatial LOX contact law."""

import unittest

import numpy as np
import warp as wp

from newton._src.solvers.kamino._src.core.types import mat36f, mat66f, vec6f
from newton._src.solvers.kamino._src.solvers.lox.contact import (
    SpatialContactStatus,
    compute_spatial_contact_residual,
    prepare_spatial_contact,
    solve_spatial_contact,
    spatial_contact_delassus,
)
from newton.tests.kamino import setup_tests, test_context


@wp.kernel
def _solve(
    matrices: wp.array[mat66f],
    velocities: wp.array[vec6f],
    frictions: wp.array[wp.vec3f],
    reactions: wp.array[vec6f],
    residuals: wp.array[vec6f],
    statuses: wp.array[wp.int32],
):
    i = wp.tid()
    eigenvectors, eigenvalues, valid = prepare_spatial_contact(matrices[i], frictions[i])
    reaction = vec6f(0.0)
    status = wp.int32(SpatialContactStatus.INVALID)
    if valid:
        reaction, status = solve_spatial_contact(matrices[i], eigenvectors, eigenvalues, frictions[i], velocities[i])
    reactions[i] = reaction
    statuses[i] = status
    residuals[i] = compute_spatial_contact_residual(
        matrices[i], frictions[i], reaction, matrices[i] * reaction + velocities[i]
    )


@wp.kernel
def _prepare_assembled(
    jacobians: wp.array[mat36f],
    frames: wp.array[wp.mat33f],
    inverse_weights: wp.array[mat66f],
    friction: wp.vec3f,
    metrics: wp.array[mat66f],
    valid: wp.array[wp.int32],
):
    i = wp.tid()
    metric = spatial_contact_delassus(i, -1, 0, jacobians, jacobians, frames, mat66f(0.0), inverse_weights[i])
    _eigenvectors, _eigenvalues, prepared = prepare_spatial_contact(metric, friction)
    metrics[i] = metric
    valid[i] = wp.int32(prepared)


class TestLOXSpatialContact(unittest.TestCase):
    def setUp(self):
        if not test_context.setup_done:
            setup_tests(clear_cache=False)
        self.default_device = wp.get_device(test_context.device)

    def _run(self, matrices, velocities, frictions, expected=None, *, status=0):
        device = self.default_device
        count = len(matrices)
        reactions = wp.empty(count, dtype=vec6f, device=device)
        residuals = wp.empty(count, dtype=vec6f, device=device)
        statuses = wp.empty(count, dtype=wp.int32, device=device)
        wp.launch(
            _solve,
            dim=count,
            inputs=[
                wp.array(matrices, dtype=mat66f, device=device),
                wp.array(velocities, dtype=vec6f, device=device),
                wp.array(frictions, dtype=wp.vec3f, device=device),
                reactions,
                residuals,
                statuses,
            ],
            device=device,
        )
        np.testing.assert_array_equal(statuses.numpy(), status)
        if expected is not None:
            np.testing.assert_allclose(reactions.numpy(), expected, atol=3e-5, rtol=3e-5)
            metric_scales = np.sqrt(np.max(np.abs(matrices), axis=(1, 2)))
            np.testing.assert_allclose(residuals.numpy() / metric_scales[:, None], 0.0, atol=3e-5)
        return residuals.numpy()

    def test_coupled_sticking_and_sliding(self):
        """Recover constructed contact solutions with full normal and angular coupling."""
        rng = np.random.default_rng(894)
        matrices, velocities, frictions, expected = [], [], [], []
        for sliding in (False, True):
            for _ in range(24):
                factor = rng.normal(size=(6, 6))
                canonical = factor @ factor.T + 4.0 * np.eye(6)
                friction = np.array([0.7, 0.08, 0.12])
                scaling = np.array([1.0, 0.7, 0.7, 0.08, 0.12, 0.12])
                rho = rng.normal(size=6)
                rho[0] = 2.0
                rho[1:] *= (2.0 if sliding else 0.5) / np.linalg.norm(rho[1:])
                gamma = np.zeros(6)
                if sliding:
                    gamma[1:] = -3.0 * rho[1:]
                matrix = canonical / np.outer(scaling, scaling)
                reaction = scaling * rho
                free = gamma / scaling - matrix @ reaction
                if free[0] >= 0.0:
                    continue
                matrices.append(matrix)
                velocities.append(free)
                frictions.append(friction)
                expected.append(reaction)
        self._run(matrices, velocities, frictions, expected)

    def test_shared_friction_budget(self):
        """Make simultaneous sliding, spinning and rolling share one cone boundary."""
        self._run(
            [np.eye(6)],
            [[-1.0, 2.0, 0.0, 2.0, 2.0, 0.0]],
            [[1.0, 1.0, 1.0]],
            [[1.0, -1.0 / np.sqrt(3), 0.0, -1.0 / np.sqrt(3), -1.0 / np.sqrt(3), 0.0]],
        )

    def test_zero_coefficients(self):
        """Omit disabled rows including singular unused angular blocks."""
        matrices, velocities, frictions, expected = [], [], [], []
        for friction in ([0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [0.0, 0.2, 0.0], [0.0, 0.0, 0.3]):
            scaling = np.array([1.0, friction[0], friction[0], friction[1], friction[2], friction[2]])
            active = scaling > 0
            matrix = np.diag(active.astype(float))
            free = np.array([-1.0, 1.0, 0.0, 1.0, 1.0, 0.0])
            reaction = np.zeros(6)
            reaction[0] = 1.0
            if friction[0]:
                reaction[1] = -friction[0]
            elif friction[1]:
                reaction[3] = -friction[1]
            elif friction[2]:
                reaction[4] = -friction[2]
            matrices.append(matrix)
            velocities.append(free)
            frictions.append(friction)
            expected.append(reaction)
        self._run(matrices, velocities, frictions, expected)

    def test_separation_and_metric_scaling(self):
        """Preserve the law under scaling and return zero for separating contact."""
        matrices, velocities, frictions, expected = [], [], [], []
        for scale in (1e-8, 1.0, 1e8):
            for normal in (-1.0, 0.1):
                matrices.append(scale * np.eye(6))
                velocities.append(scale * np.array([normal, 2.0, 0.0, 0.0, 0.0, 0.0]))
                frictions.append([0.5, 0.1, 0.2])
                expected.append([1.0, -0.5, 0.0, 0.0, 0.0, 0.0] if normal < 0 else np.zeros(6))
        self._run(matrices, velocities, frictions, expected)

    def test_mixed_active_rows(self):
        """Retain coupling when only some friction families are enabled."""
        rng = np.random.default_rng(401)
        matrices, velocities, frictions, expected = [], [], [], []
        for friction in ([0.0, 0.2, 0.3], [0.7, 0.0, 0.3], [0.7, 0.2, 0.0], [0.7, 0.001, 0.002]):
            for _ in range(8):
                factor = rng.normal(size=(6, 6))
                matrix = factor @ factor.T + 5.0 * np.eye(6)
                scaling = np.array([1.0, friction[0], friction[0], friction[1], friction[2], friction[2]])
                active = scaling > 0.0
                rho = rng.normal(size=6)
                rho[~active] = 0.0
                rho[0] = 2.0
                rho[1:] *= rho[0] / np.linalg.norm(rho[1:])
                velocity = np.zeros(6)
                velocity[1:][active[1:]] = -2.0 * rho[1:][active[1:]] / scaling[1:][active[1:]]
                reaction = scaling * rho
                matrices.append(matrix)
                velocities.append(velocity - matrix @ reaction)
                frictions.append(friction)
                expected.append(reaction)
        self._run(matrices, velocities, frictions, expected)

    def test_nonmonotone_root(self):
        """Solve a strongly coupled case whose scalar root is not monotone."""
        schur = np.array([0.1, 10.0, 2.0, 2.0, 2.0])
        coupling = np.array([0.0, 2.0, 0.0, 0.0, 0.0])
        rhs = np.array([1.0, 1.0, 0.0, 0.0, 0.0])
        normal_rhs = -0.01
        matrix = np.zeros((6, 6))
        matrix[0, 0] = 1.0
        matrix[0, 1:] = coupling
        matrix[1:, 0] = coupling
        matrix[1:, 1:] = np.diag(schur) + np.outer(coupling, coupling)
        lower, upper = 0.0, 100.0
        for _ in range(90):
            alpha = 0.5 * (lower + upper)
            solution = rhs / (schur + alpha)
            value = np.linalg.norm(solution) - coupling @ solution + normal_rhs
            if value > 0.0:
                lower = alpha
            else:
                upper = alpha
        reaction = np.r_[coupling @ solution - normal_rhs, -solution]
        free = np.r_[normal_rhs, rhs + normal_rhs * coupling]
        self._run([matrix], [free], [[1.0, 1.0, 1.0]], [reaction])

    def test_root_resolved_to_working_precision(self):
        """Accept a sliding root bracketed by adjacent floats where rounding keeps phi above the tolerance."""
        # A sliding and spinning sphere captured from a box and sphere pile
        matrix = np.array(
            [
                [4.5364478e-01, -4.9251363e-02, 1.2013567e-01, 1.2791183e-07, 1.2013613e00, 4.9251497e-01],
                [-4.9251363e-02, 1.2250419e00, 3.1556573e-02, -1.2013613e00, 1.3654721e-08, 4.093334e00],
                [1.2013567e-01, 3.1556573e-02, 1.1610042e00, -4.925149e-01, -4.0933337e00, -1.4145377e-07],
                [1.2792172e-07, -1.2013612e00, -4.9251488e-01, 7.8433426e01, 2.390207e-06, -9.3596725e-07],
                [1.2013613e00, 1.3611128e-08, -4.093334e00, 2.3898185e-06, 7.843347e01, -1.6652045e-06],
                [4.9251494e-01, 4.093334e00, -1.4153427e-07, -9.4009386e-07, -1.6652043e-06, 7.843341e01],
            ],
            dtype=np.float32,
        )
        free = np.array(
            [-4.8010784e-01, 2.1781872e-03, 3.1286456e-02, -6.6916915e-03, -7.083571e-02, -1.0622519e-01],
            dtype=np.float32,
        )
        residuals = self._run([matrix], [free], [[0.5, 0.01, 0.005]])
        np.testing.assert_allclose(residuals / np.sqrt(np.max(np.abs(matrix))), 0.0, atol=3e-5)

    def test_unbracketed_root_status(self):
        """Report an exhausted root bracket instead of silently accepting an impulse."""
        self._run([np.eye(6)], [[-1e-30, 1.0, 0.0, 0.0, 0.0, 0.0]], [[1.0, 1.0, 1.0]], status=2)

    def test_fast_path_metric_and_coefficient_scaling(self):
        """Avoid determinant underflow in the three-dimensional fast path."""
        matrices, velocities, frictions, expected = [], [], [], []
        for scale in (1e-16, 1.0, 1e16):
            matrices.append(scale * np.eye(6))
            velocities.append(scale * np.array([-1.0, 2.0, 0.0, 0.0, 0.0, 0.0]))
            frictions.append([0.5, 0.0, 0.0])
            expected.append([1.0, -0.5, 0.0, 0.0, 0.0, 0.0])
        matrices.append(np.eye(6))
        velocities.append([-1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        frictions.append([1e-12, 0.0, 0.0])
        expected.append([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self._run(matrices, velocities, frictions, expected)

    def test_assembled_metric_is_symmetric(self):
        """Assemble an exactly symmetric metric for spheres with nearly isotropic inertia."""
        # The spin and rolling cross terms of these metrics vanish up to rounding, which an
        # unsymmetrized triple product can leave asymmetric beyond the preparation tolerance.
        rng = np.random.default_rng(3)
        count, radius, mass = 512, 0.1, 4.19
        frames = np.linalg.qr(rng.normal(size=(count, 3, 3)))[0]
        body_axes = np.linalg.qr(rng.normal(size=(count, 3, 3)))[0]
        directions = np.transpose(frames, (0, 2, 1))
        offsets = -radius * frames[:, None, :, 2]
        jacobians = np.concatenate([directions, np.cross(offsets, directions)], axis=2)
        principal = 0.4 * mass * radius**2 * rng.uniform(0.9, 1.1, size=(count, 3))
        inverse_weights = np.zeros((count, 6, 6))
        inverse_weights[:, :3, :3] = np.eye(3) / mass
        inverse_weights[:, 3:, 3:] = np.einsum("nij,nj,nkj->nik", body_axes, 1.0 / principal, body_axes)
        device = self.default_device
        metrics = wp.empty(count, dtype=mat66f, device=device)
        valid = wp.empty(count, dtype=wp.int32, device=device)
        wp.launch(
            _prepare_assembled,
            dim=count,
            inputs=[
                wp.array(jacobians, dtype=mat36f, device=device),
                wp.array(frames, dtype=wp.mat33f, device=device),
                wp.array(inverse_weights, dtype=mat66f, device=device),
                wp.vec3f(0.5, 0.01, 0.005),
                metrics,
                valid,
            ],
            device=device,
        )
        metric = metrics.numpy()
        np.testing.assert_array_equal(metric, np.transpose(metric, (0, 2, 1)))
        np.testing.assert_array_equal(valid.numpy(), 1)

    def test_invalid_preparation(self):
        """Report invalid active metrics and coefficients explicitly."""
        invalid = np.eye(6)
        invalid[3, 3] = -1.0
        self._run(
            [invalid, np.eye(6), np.eye(6)],
            [[-1.0, 0.0, 0.0, 0.0, 0.0, 0.0]] * 3,
            [[0.5, 0.1, 0.1], [-1.0, 0.1, 0.1], [0.5, np.nan, 0.1]],
            status=1,
        )


if __name__ == "__main__":
    # Test setup
    setup_tests()

    # Run all tests
    unittest.main(verbosity=2)
