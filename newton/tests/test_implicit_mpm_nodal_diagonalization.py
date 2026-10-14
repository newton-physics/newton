# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np
import warp as wp
import warp.fem as fem

import newton
from newton._src.solvers.implicit_mpm.implicit_mpm_solver_kernels import (
    YieldMomentVec,
    _TransformedStrainBasis,
    compute_nodal_diagonalization,
    integrate_yield_parameters,
    inverse_scale_vector,
)
from newton._src.solvers.implicit_mpm.rheology_solver_kernels import YieldParamVec
from newton._src.solvers.implicit_mpm.solver_implicit_mpm import _make_grid_basis_space
from newton.solvers import SolverImplicitMPM
from newton.tests.unittest_utils import add_function_test, get_cuda_test_devices, get_test_devices


def _diagonalize(mass, device):
    n = len(mass)
    volume = mass @ np.ones(n, dtype=np.float32)
    diagonal = wp.empty((1, n), dtype=float, device=device)
    transform = wp.empty((1, n, n), dtype=float, device=device)
    coefficient_transform = wp.empty_like(transform)
    wp.launch(
        compute_nodal_diagonalization,
        dim=1,
        inputs=[
            wp.array((0, 1), dtype=int, device=device),
            wp.array((0,), dtype=int, device=device),
            wp.array(mass[None], dtype=wp.types.matrix(shape=(n, n), dtype=wp.float32), device=device),
            wp.array(volume[None], dtype=float, device=device),
        ],
        outputs=[diagonal, transform, coefficient_transform],
        device=device,
    )
    r = transform.numpy()[0]
    return r, coefficient_transform.numpy()[0], diagonal.numpy()[0], r @ volume


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


def test_nodal_mean_orientation(test, device):
    """Orient modes by their signed first moment, including nearly zero integrals."""
    for delta in (-1.0e-5, 0.0, 1.0e-5):
        root = np.array(((1.0, -1.0 + delta), (-1.0 + delta, 2.0)), dtype=np.float32)
        mass = root @ root
        r, t, _diagonal, mean = _diagonalize(mass, device)
        if delta < 0.0:
            test.assertLess(r[0, 0], 0.0)
        elif delta > 0.0:
            test.assertGreater(r[0, 0], 0.0)
        test.assertTrue(np.all(mean >= -2.0e-7))
        np.testing.assert_allclose(r @ mass @ r.T, np.eye(2) / 2, atol=3.0e-6)
        np.testing.assert_allclose(t @ r.T, np.eye(2), atol=3.0e-6)


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


def test_nodal_singular_completion(test, device):
    """Keep nodal completion directions independent of the supported mass eigenvectors."""
    for n in (4, 8):
        null_direction = np.arange(1, n + 1, dtype=float)
        null_direction /= np.linalg.norm(null_direction)
        completion, _ = np.linalg.qr(null_direction[:, None], mode="complete")
        frame = completion[:, 1:].T
        for angle in (0.0, 0.37):
            with test.subTest(n=n, angle=angle):
                rotation = np.eye(n - 1)
                rotation[:2, :2] = ((np.cos(angle), -np.sin(angle)), (np.sin(angle), np.cos(angle)))
                spectrum = np.linspace(0.25, 0.9, n - 1) / n
                mass = (frame.T @ rotation @ np.diag(spectrum) @ rotation.T @ frame).astype(np.float32)
                r, t, diagonal, _mean = _diagonalize(mass, device)
                active = np.linalg.norm(r, axis=1) > 0.0
                eigenvalues, eigenvectors = np.linalg.eigh(mass.astype(float))
                supported = eigenvalues > 1.0e-6
                inverse_root = (eigenvectors[:, supported] / np.sqrt(n * eigenvalues[supported])) @ eigenvectors[
                    :, supported
                ].T
                actual_frame = t[active] @ inverse_root
                test.assertEqual(len(actual_frame), n - 1)
                for actual, expected in zip(actual_frame, frame, strict=True):
                    # The final pressure orientation may flip a row, but not rotate it.
                    np.testing.assert_allclose(np.outer(actual, actual), np.outer(expected, expected), atol=3.0e-6)
                np.testing.assert_allclose(r @ mass @ r.T, np.diag(active.astype(float)) / n, atol=2.0e-6)
                np.testing.assert_allclose(t @ r.T, np.diag(active.astype(float)), atol=3.0e-6)
                np.testing.assert_allclose(r @ mass, np.diag(diagonal) @ t, atol=2.0e-6)

        r, t, diagonal, _mean = _diagonalize(np.zeros((n, n), dtype=np.float32), device)
        np.testing.assert_array_equal(r, np.zeros((n, n)))
        np.testing.assert_array_equal(t, np.zeros((n, n)))
        np.testing.assert_array_equal(diagonal, np.ones(n))


def test_nodal_material_averaging(test, device):
    """Respect pressure orientation, symmetric shear, quadratic viscosity, and inactive particles."""
    coords = np.array(((0.1, 0.3, 0.2), (0.4, 0.2, 0.4), (0.65, 0.4, 0.7), (0.9, 0.7, 0.5), (0.8, 0.3, 0.2)))
    volumes = np.array((0.1, 0.2, 0.3, 0.4, 100.0), dtype=np.float32)
    pressure = np.array((10.0, 20.0, 30.0, 40.0, 1.0e6))
    tensile_ratio = np.array((0.1, 0.2, 0.3, 0.4, 0.5))
    shear = np.array((1.0, 2.0, 3.0, 4.0, 1.0e6))
    viscosity = np.array((2.0, 3.0, 5.0, 7.0, 1.0e6))
    dt = 0.01
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    SolverImplicitMPM.register_custom_attributes(builder)
    for i, position in enumerate(coords):
        builder.add_particle(
            pos=position,
            vel=(0.0, 0.0, 0.0),
            mass=1.0,
            radius=0.1,
            flags=newton.ParticleFlags.ACTIVE if i < 4 else 0,
            custom_attributes={
                "mpm:friction": 0.3,
                "mpm:yield_pressure": pressure[i],
                "mpm:tensile_yield_ratio": tensile_ratio[i],
                "mpm:yield_stress": shear[i],
                "mpm:dilatancy": 0.2,
                "mpm:viscosity": viscosity[i],
            },
        )
    model = builder.finalize(device=device)
    solver = SolverImplicitMPM(model, SolverImplicitMPM.Config(voxel_size=1.0))
    encoded = np.column_stack(
        (
            pressure * np.sqrt(1.5),
            pressure * tensile_ratio * np.sqrt(1.5),
            shear,
            0.3 * pressure,
            np.full(5, 0.2),
            viscosity / dt,
        )
    )
    with wp.ScopedDevice(device):
        grid = fem.Grid3D(res=wp.vec3i(1), bounds_lo=wp.vec3(0.0), bounds_hi=wp.vec3(1.0))
        domain = fem.Cells(grid)
        pic = fem.PicQuadrature(domain, positions=model.state().particle_q, measures=wp.array(volumes, dtype=float))
        for basis_name in ("P1d", "Q1d", "P2d", "Q2d"):
            with test.subTest(basis=basis_name):
                family = fem.Polynomial.GAUSS_LEGENDRE if basis_name == "Q1d" else None
                basis = _make_grid_basis_space(grid, basis_name, family=family)
                space = fem.make_collocated_function_space(basis, dtype=YieldParamVec)
                field = fem.make_test(space, domain=domain)
                n = space.topology.MAX_NODES_PER_ELEMENT
                node_coords = basis.shape.make_node_coords_in_element()
                transform = np.ones((n, n), dtype=np.float32)
                transform[0] = [node_coords.func(j)[0] - 0.5 for j in range(n)]
                transform[1] = 7.0 * transform[0]
                transform[2] = -transform[0]
                transform[3] = 0.0
                mode_space = fem.make_collocated_function_space(basis, dtype=wp.types.vector(length=n, dtype=float))
                modes = mode_space.make_field(space_partition=field.space_partition)
                modes.dof_values = wp.array(transform.T.copy(), dtype=mode_space.dof_dtype)
                moment_space = fem.make_collocated_function_space(_TransformedStrainBasis(modes), dtype=YieldMomentVec)
                material_test = fem.make_test(moment_space, space_restriction=field.space_restriction)
                moments = fem.integrate(
                    integrate_yield_parameters,
                    quadrature=pic,
                    fields={"u": material_test},
                    values={
                        "material_parameters": solver._mpm_model.material_parameters,
                        "particle_Jp": model.state().mpm.particle_Jp,
                        "particle_flags": solver._mpm_model.material_particle_flags,
                        "inv_cell_volume": 1.0,
                        "dt": dt,
                        "transformed_basis": True,
                    },
                    output_dtype=YieldMomentVec,
                    assembly="generic",
                    kernel_options={"max_unroll": 4, "enable_backward": False},
                )
                values = moments.view(dtype=float)
                wp.launch(inverse_scale_vector, dim=(n, 6), inputs=[values[:, 6:], values[:, :6]], device=device)
                parameters = values[:, :6].view(YieldParamVec)
                psi = np.ones((n, 5))
                psi[0] = coords[:, 0] - 0.5
                psi[1] = 7.0 * psi[0]
                psi[2] = -psi[0]
                psi[3] = 0.0
                positive = np.maximum(psi, 0.0)
                absolute = np.abs(psi)
                weights = np.stack((positive, positive, absolute, positive, positive, psi**2), axis=-1)
                weights *= volumes[None, :, None]
                weights[:, 4] = 0.0
                total = weights.sum(axis=1)
                expected = np.divide(
                    (weights * encoded[None]).sum(axis=1), total, out=np.zeros((n, 6)), where=total > 0.0
                )
                actual = parameters.numpy()
                test.assertTrue(np.all(actual[:, 0] >= 0.0))
                np.testing.assert_allclose(actual, expected, rtol=3.0e-5, atol=1.0e-4)
                np.testing.assert_allclose(actual[0], actual[1], rtol=3.0e-5, atol=1.0e-4)
                np.testing.assert_allclose(actual[0, (2, 5)], actual[2, (2, 5)], rtol=3.0e-5)
                test.assertGreater(actual[0, 0], actual[2, 0])
                np.testing.assert_allclose(actual[(0, 1, 2), 3] / actual[(0, 1, 2), 0], 0.3 / np.sqrt(1.5), rtol=3.0e-5)


class _NodalSolver(SolverImplicitMPM):
    def _build_strain_basis_transform(self, pic, scratch, inv_cell_volume):
        transform, coefficient_transform = super()._build_strain_basis_transform(pic, scratch, inv_cell_volume)
        element = int(pic.cell_indices.numpy().reshape(-1)[0])
        n = transform.shape[1]
        self.test_transform = transform.numpy()[element].copy()
        self.test_diagonal = scratch.strain_node_particle_volume.numpy().reshape(-1, n)[element].copy()
        return transform, coefficient_transform


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
                material_test = solver._scratchpad.strain_yield_parameters_test
                test.assertIs(material_test.space.dtype, YieldMomentVec)
                np.testing.assert_allclose(solver.test_transform, np.eye(n), atol=3.0e-6)
                np.testing.assert_allclose(solver.test_diagonal, np.full(n, 1.0 / n), atol=1.0e-7)
                for _ in range(3):
                    solver.step(state, state, None, None, 0.001)
                    test.assertIs(solver._scratchpad.strain_yield_parameters_test, material_test)
                for array in (
                    state.particle_q,
                    state.particle_qd,
                    state.mpm.particle_stress,
                    state.mpm.particle_qd_grad,
                    state.mpm.particle_elastic_strain,
                ):
                    test.assertTrue(np.isfinite(array.numpy()).all())


def test_nodal_higher_order_solver(test, device):
    """Use the same supported-subspace orthogonalization for higher-order strain bases."""
    for basis in ("P2d", "Q2d"):
        with test.subTest(basis=basis):
            solver, state = _make_solver(device, basis, active_particle_count=3, max_iterations=0)
            solver.step(state, state, None, None, 0.001)
            test.assertEqual(np.linalg.matrix_rank(solver.test_transform), 3)
            for _ in range(2):
                solver.step(state, state, None, None, 0.001)
            for array in (state.particle_q, state.particle_qd, state.mpm.particle_stress):
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
    ("test_nodal_singular_completion", test_nodal_singular_completion),
    ("test_nodal_mean_orientation", test_nodal_mean_orientation),
    ("test_nodal_material_averaging", test_nodal_material_averaging),
    ("test_nodal_full_cell_solver", test_nodal_full_cell_solver),
    ("test_nodal_higher_order_solver", test_nodal_higher_order_solver),
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
