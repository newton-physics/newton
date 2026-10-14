# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.solvers.implicit_mpm.implicit_mpm_solver_kernels import (
    compute_nodal_diagonalization,
    normalize_strain_yield_parameters,
)
from newton._src.solvers.implicit_mpm.rheology_solver_kernels import YieldParamVec
from newton.solvers import SolverImplicitMPM
from newton.tests.unittest_utils import add_function_test, get_cuda_test_devices, get_test_devices


def _diagonalize(mass, device):
    n = len(mass)
    volume = mass @ np.ones(n, dtype=np.float32)
    diagonal = wp.empty((1, n), dtype=float, device=device)
    transform = wp.empty((1, n, n), dtype=float, device=device)
    coefficient_transform = wp.empty_like(transform)
    mean = wp.empty_like(diagonal)
    wp.launch(
        compute_nodal_diagonalization,
        dim=1,
        inputs=[
            wp.array((0, 1), dtype=int, device=device),
            wp.array((0,), dtype=int, device=device),
            wp.array(mass[None], dtype=wp.types.matrix(shape=(n, n), dtype=wp.float32), device=device),
            wp.array(volume[None], dtype=float, device=device),
        ],
        outputs=[diagonal, transform, mean, coefficient_transform],
        device=device,
    )
    return transform.numpy()[0], coefficient_transform.numpy()[0], diagonal.numpy()[0], mean.numpy()[0]


def test_nodal_full_rank_transform(test, device):
    """Preserve full-cell nodes and consistent primal/dual transforms for nonuniform mass."""
    for n in (4, 8):
        rng = np.random.default_rng(n)
        a = rng.normal(size=(n, n))
        a = a @ a.T
        a /= np.linalg.norm(a, 2)
        for perturbation in (0.0, 1.0e-5, 0.3):
            with test.subTest(n=n, perturbation=perturbation):
                mass = ((np.eye(n) + perturbation * a) / n).astype(np.float32)
                r, t, diagonal, mean = _diagonalize(mass, device)
                np.testing.assert_allclose(r @ mass @ r.T, np.eye(n) / n, atol=2.0e-6)
                np.testing.assert_allclose(t @ r.T, np.eye(n), atol=1.0e-5)
                np.testing.assert_allclose(r @ mass, np.diag(diagonal) @ t, atol=2.0e-6)
                test.assertTrue(np.all(mean >= 0.0))
                if perturbation <= 1.0e-5:
                    np.testing.assert_allclose(r, np.eye(n), atol=1.0e-5)


def test_nodal_signed_material_moments(test, device):
    """Keep material parameters admissible without dropping strain modes for signed moments."""
    average = np.array((10.0, 5.0, 1.0, 0.3, 0.0, 10.0), dtype=np.float32)
    moments = np.vstack((average, average, np.zeros(6), 4.0 * average)).astype(np.float32)
    moments[0, 0] *= -1.0
    moments[1, 5] *= -1.0
    parameters = wp.array(moments, dtype=YieldParamVec, device=device)
    wp.launch(
        normalize_strain_yield_parameters,
        dim=4,
        inputs=[
            4,
            wp.array((1.0, 1.0, 0.0, 2.0), dtype=float, device=device),
            wp.array(average[None], dtype=YieldParamVec, device=device),
            parameters,
        ],
        device=device,
    )
    np.testing.assert_allclose(parameters.numpy(), np.vstack((average, average, average, 2.0 * average)), atol=1.0e-6)


def test_nodal_rank_deficient_transform(test, device):
    """Orthogonalize and equally normalize the supported subspace of singular mass blocks."""
    for n in (4, 8):
        eigenvalues = np.append(np.linspace(0.25, 0.9, n - 1), 0.0) / n
        for mixed_nullspace in (False, True):
            with test.subTest(n=n, mixed_nullspace=mixed_nullspace):
                basis = np.eye(n)
                if mixed_nullspace:
                    basis[np.ix_((0, n - 1), (0, n - 1))] = np.array(((1.0, 1.0), (1.0, -1.0))) / np.sqrt(2.0)
                mass = (basis @ np.diag(eigenvalues) @ basis.T).astype(np.float32)
                r, t, diagonal, mean = _diagonalize(mass, device)
                test.assertTrue(np.isfinite(r).all())
                test.assertTrue(np.isfinite(t).all())
                test.assertEqual(np.linalg.matrix_rank(r), n - 1)
                active = np.linalg.norm(r, axis=1) > 0.0
                np.testing.assert_allclose(diagonal[active], 1.0 / n, atol=1.0e-6)
                np.testing.assert_allclose((r @ mass @ r.T)[active][:, active], np.eye(n - 1) / n, atol=1.0e-6)
                np.testing.assert_allclose(t @ r.T, np.diag(active.astype(float)), atol=1.0e-6)
                np.testing.assert_allclose(r @ mass, np.diag(diagonal) @ t, atol=1.0e-6)
                # Reconstructing a coefficient warmstart retains exactly its supported part.
                projector = basis[:, :-1] @ basis[:, :-1].T
                np.testing.assert_allclose(r.T @ t, projector, atol=1.0e-6)
                test.assertTrue(np.all(mean >= 0.0))

        # One particle produces a dense rank-one Gram block, not a diagonal block.
        phi = np.linspace(1.0, 2.0, n, dtype=np.float32)
        phi /= phi.sum()
        mass = 0.125 * np.outer(phi, phi)
        r, t, diagonal, mean = _diagonalize(mass, device)
        test.assertEqual(np.linalg.matrix_rank(r), 1)
        active = np.linalg.norm(r, axis=1) > 0.0
        np.testing.assert_allclose(diagonal[active], 1.0 / n, atol=1.0e-6)
        np.testing.assert_allclose((r @ mass @ r.T)[active][:, active], np.eye(1) / n, atol=1.0e-6)
        np.testing.assert_allclose(r.T @ t, np.outer(phi, phi) / np.dot(phi, phi), atol=1.0e-6)
        test.assertTrue(np.all(mean >= 0.0))


class _NodalSolver(SolverImplicitMPM):
    def _build_strain_eigenbasis(self, pic, scratch, inv_cell_volume):
        transform, mean = super()._build_strain_eigenbasis(pic, scratch, inv_cell_volume)
        element = int(pic.cell_indices.numpy().reshape(-1)[0])
        n = transform.shape[1]
        self.test_transform = transform.numpy()[element].copy()
        self.test_diagonal = scratch.strain_node_particle_volume.numpy().reshape(-1, n)[element].copy()
        return transform, mean


def _make_solver(device, basis, *, active_particle_count=8, **kwargs):
    a = (0.5 - 0.5 / np.sqrt(3.0), 0.5 + 0.5 / np.sqrt(3.0))
    positions = np.array(np.meshgrid(a, a, a)).reshape(3, -1).T
    if kwargs.get("grid_type") == "sparse":
        # Nanogrid cells are centered at integer voxel coordinates.
        positions -= 0.5
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    SolverImplicitMPM.register_custom_attributes(builder)
    for i, position in enumerate(positions):
        builder.add_particle(
            pos=position,
            vel=(0.1 * position[0], -0.05 * position[1], 0.02 * position[2]),
            mass=0.125,
            radius=0.25,
            flags=newton.ParticleFlags.ACTIVE if i < active_particle_count else 0,
            custom_attributes={
                "mpm:young_modulus": 1.0e4,
                "mpm:friction": 0.0,
                "mpm:yield_pressure": 300.0,
                "mpm:tensile_yield_ratio": 0.5,
                "mpm:yield_stress": 20.0,
                "mpm:viscosity": 10.0,
            },
        )
    model = builder.finalize(device=device)
    config = {"voxel_size": 1.0, "grid_type": "fixed", "max_iterations": 5, "warmstart_mode": "particles"}
    config.update(kwargs)
    solver = _NodalSolver(model, SolverImplicitMPM.Config(strain_basis=basis, **config))
    return solver, model.state()


def test_nodal_full_cell_solver(test, device):
    """Recover reference-node solves and finite histories for full P1d/Q1d cells."""
    for basis, n in (("P1d", 4), ("Q1d", 8)):
        for warmstart in ("particles", "grid"):
            with test.subTest(basis=basis, warmstart=warmstart):
                solver, state = _make_solver(device, basis, warmstart_mode=warmstart)
                solver.step(state, state, None, None, 0.001)
                np.testing.assert_allclose(solver.test_transform, np.eye(n), atol=3.0e-6)
                np.testing.assert_allclose(solver.test_diagonal, np.full(n, 1.0 / n), atol=1.0e-7)
                for _ in range(3):
                    solver.step(state, state, None, None, 0.001)
                for array in (
                    state.particle_q,
                    state.particle_qd,
                    state.mpm.particle_stress,
                    state.mpm.particle_qd_grad,
                    state.mpm.particle_elastic_strain,
                ):
                    test.assertTrue(np.isfinite(array.numpy()).all())


def test_nodal_sparse_graph_capture(test, device):
    """Match captured sparse nodal solves to ordinary stepping for both strain bases."""
    # This solver omits diagnostic host copies inside capture.
    for basis, active_particle_count in (("P1d", 8), ("Q1d", 8), ("P1d", 3), ("Q1d", 3)):
        settings = {"grid_type": "sparse", "grid_padding": 0, "max_active_cell_count": 64}
        diagnostic, reference = _make_solver(device, basis, active_particle_count=active_particle_count, **settings)
        model = diagnostic.model
        solver = SolverImplicitMPM(
            model,
            SolverImplicitMPM.Config(
                strain_basis=basis,
                voxel_size=1.0,
                max_iterations=5,
                warmstart_mode="particles",
                **settings,
            ),
        )
        captured = model.state()
        solver.step(captured, captured, None, None, 0.001)
        diagnostic.step(reference, reference, None, None, 0.001)
        expected_rank = min(active_particle_count, diagnostic.test_transform.shape[0])
        test.assertEqual(np.linalg.matrix_rank(diagnostic.test_transform), expected_rank)
        with wp.ScopedCapture(device=device) as capture:
            solver.step(captured, captured, None, None, 0.001)
        for _ in range(3):
            wp.capture_launch(capture.graph)
            diagnostic.step(reference, reference, None, None, 0.001)
        np.testing.assert_allclose(captured.particle_q.numpy(), reference.particle_q.numpy(), atol=1.0e-6)
        np.testing.assert_allclose(captured.particle_qd.numpy(), reference.particle_qd.numpy(), atol=1.0e-5)


class TestImplicitMPMNodalDiagonalization(unittest.TestCase):
    pass


for name, func in (
    ("test_nodal_full_rank_transform", test_nodal_full_rank_transform),
    ("test_nodal_rank_deficient_transform", test_nodal_rank_deficient_transform),
    ("test_nodal_signed_material_moments", test_nodal_signed_material_moments),
    ("test_nodal_full_cell_solver", test_nodal_full_cell_solver),
):
    add_function_test(TestImplicitMPMNodalDiagonalization, name, func, devices=get_test_devices(mode="basic"))

add_function_test(
    TestImplicitMPMNodalDiagonalization,
    "test_nodal_sparse_graph_capture",
    test_nodal_sparse_graph_capture,
    devices=get_cuda_test_devices(mode="basic"),
)

if __name__ == "__main__":
    unittest.main(verbosity=2)
