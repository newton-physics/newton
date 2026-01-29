# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest

import numpy as np
import warp as wp

from newton._src.sim.builder import ModelBuilder
from newton._src.solvers.vbd.particle_vbd_kernels import (
    evaluate_volumetric_neo_hookean_force_and_hessian,
    mat43,
    vec9,
)
from newton._src.solvers.vbd.solver_vbd import SolverVBD
from newton.tests.unittest_utils import add_function_test, get_test_devices


@wp.func
def assemble_tet_vertex_force(
    dE_dF: vec9,
    m1: float,
    m2: float,
    m3: float,
):
    f = wp.vec3(
        -(dE_dF[0] * m1 + dE_dF[3] * m2 + dE_dF[6] * m3),
        -(dE_dF[1] * m1 + dE_dF[4] * m2 + dE_dF[7] * m3),
        -(dE_dF[2] * m1 + dE_dF[5] * m2 + dE_dF[8] * m3),
    )

    return f


@wp.kernel
def compute_neo_hookean_energy_and_force_and_hessian(
    # inputs
    tet_id: int,
    dt: float,
    pos: wp.array(dtype=wp.vec3),
    tet_indices: wp.array(dtype=wp.int32, ndim=2),
    tet_poses: wp.array(dtype=wp.mat33),
    tet_materials: wp.array(dtype=float, ndim=2),
    # outputs: particle force and hessian
    particle_forces: wp.array(dtype=wp.vec3),
    particle_hessians: wp.array(dtype=wp.mat33),
):
    v_order = wp.tid()
    f, h = evaluate_volumetric_neo_hookean_force_and_hessian(
        tet_id,
        wp.tid(),
        pos,  # dont need damping
        pos,
        tet_indices,
        tet_poses[tet_id],
        tet_materials[tet_id, 0],  # k_mu
        tet_materials[tet_id, 1],  # k_lambda
        tet_materials[tet_id, 2],  # k_damp (was incorrectly [0,3] which is out of bounds!)
        dt,
    )

    particle_forces[tet_indices[tet_id, v_order]] = f

    particle_hessians[tet_indices[tet_id, v_order]] = h


@wp.kernel
def compute_neo_hookean_energy_and_force(
    # inputs
    tet_id: int,
    dt: float,
    pos: wp.array(dtype=wp.vec3),
    tet_indices: wp.array(dtype=wp.int32, ndim=2),
    tet_poses: wp.array(dtype=wp.mat33),
    tet_materials: wp.array(dtype=float, ndim=2),
    # outputs: particle force and hessian
    tet_energy: wp.array(dtype=float),
    particle_forces: wp.array(dtype=float),
):
    v0_idx = tet_indices[tet_id, 0]
    v1_idx = tet_indices[tet_id, 1]
    v2_idx = tet_indices[tet_id, 2]
    v3_idx = tet_indices[tet_id, 3]

    mu = tet_materials[tet_id, 0]
    lmbd = tet_materials[tet_id, 1]

    v0 = pos[v0_idx]
    v1 = pos[v1_idx]
    v2 = pos[v2_idx]
    v3 = pos[v3_idx]

    Dm_inv = tet_poses[tet_id]
    rest_volume = 1.0 / (wp.determinant(Dm_inv) * 6.0)

    diff_1 = v1 - v0
    diff_2 = v2 - v0
    diff_3 = v3 - v0
    Ds = wp.matrix_from_cols(diff_1, diff_2, diff_3)

    F = Ds * Dm_inv

    a = 1.0 + mu / lmbd

    det_F = wp.determinant(F)

    E = rest_volume * 0.5 * (mu * (wp.trace(F * wp.transpose(F)) - 3.0) + lmbd * (det_F - a) * (det_F - a))
    tet_energy[tet_id] = E

    F1_1 = F[0, 0]
    F2_1 = F[1, 0]
    F3_1 = F[2, 0]
    F1_2 = F[0, 1]
    F2_2 = F[1, 1]
    F3_2 = F[2, 1]
    F1_3 = F[0, 2]
    F2_3 = F[1, 2]
    F3_3 = F[2, 2]

    dPhi_D_dF = vec9(
        F1_1,
        F2_1,
        F3_1,
        F1_2,
        F2_2,
        F3_2,
        F1_3,
        F2_3,
        F3_3,
    )

    ddetF_dF = vec9(
        F2_2 * F3_3 - F2_3 * F3_2,
        F1_3 * F3_2 - F1_2 * F3_3,
        F1_2 * F2_3 - F1_3 * F2_2,
        F2_3 * F3_1 - F2_1 * F3_3,
        F1_1 * F3_3 - F1_3 * F3_1,
        F1_3 * F2_1 - F1_1 * F2_3,
        F2_1 * F3_2 - F2_2 * F3_1,
        F1_2 * F3_1 - F1_1 * F3_2,
        F1_1 * F2_2 - F1_2 * F2_1,
    )

    k = det_F - a
    dPhi_D_dF = dPhi_D_dF * mu
    dPhi_H_dF = ddetF_dF * lmbd * k

    dE_dF = (dPhi_D_dF + dPhi_H_dF) * rest_volume

    Dm_inv_1_1 = Dm_inv[0, 0]
    Dm_inv_2_1 = Dm_inv[1, 0]
    Dm_inv_3_1 = Dm_inv[2, 0]
    Dm_inv_1_2 = Dm_inv[0, 1]
    Dm_inv_2_2 = Dm_inv[1, 1]
    Dm_inv_3_2 = Dm_inv[2, 1]
    Dm_inv_1_3 = Dm_inv[0, 2]
    Dm_inv_2_3 = Dm_inv[1, 2]
    Dm_inv_3_3 = Dm_inv[2, 2]

    ms = mat43(
        -Dm_inv_1_1 - Dm_inv_2_1 - Dm_inv_3_1,
        -Dm_inv_1_2 - Dm_inv_2_2 - Dm_inv_3_2,
        -Dm_inv_1_3 - Dm_inv_2_3 - Dm_inv_3_3,
        Dm_inv_1_1,
        Dm_inv_1_2,
        Dm_inv_1_3,
        Dm_inv_2_1,
        Dm_inv_2_2,
        Dm_inv_2_3,
        Dm_inv_3_1,
        Dm_inv_3_2,
        Dm_inv_3_3,
    )

    for v_counter in range(4):
        f = assemble_tet_vertex_force(dE_dF, ms[v_counter, 0], ms[v_counter, 1], ms[v_counter, 2])
        particle_forces[tet_indices[tet_id, v_counter] * 3 + 0] = f[0]
        particle_forces[tet_indices[tet_id, v_counter] * 3 + 1] = f[1]
        particle_forces[tet_indices[tet_id, v_counter] * 3 + 2] = f[2]


# Pyramid-like fan around apex 4 with a quadrilateral base (0,1,2,3) split into four
# tets: (0,1,2,4), (0,2,3,4), (0,3,1,4), (1,3,2,4) plus the connected base layer.
PYRAMID_TET_INDICES = np.array(
    [
        [0, 1, 3, 9],
        [1, 4, 3, 13],
        [1, 3, 9, 13],
        [3, 9, 13, 12],
        [1, 9, 10, 13],
        [1, 2, 4, 10],
        [2, 5, 4, 14],
        [2, 4, 10, 14],
        [4, 10, 14, 13],
        [2, 10, 11, 14],
        [3, 4, 6, 12],
        [4, 7, 6, 16],
        [4, 6, 12, 16],
        [6, 12, 16, 15],
        [4, 12, 13, 16],
        [4, 5, 7, 13],
        [5, 8, 7, 17],
        [5, 7, 13, 17],
        [7, 13, 17, 16],
        [5, 13, 14, 17],
    ],
    dtype=np.int32,
)

PYRAMID_PARTICLES = [
    (0.0, 0.0, 0.0),  # 0
    (1.0, 0.0, 0.0),  # 1
    (2.0, 0.0, 0.0),  # 2
    (0.0, 1.0, 0.0),  # 3
    (1.0, 1.0, 0.0),  # 4
    (2.0, 1.0, 0.0),  # 5
    (0.0, 2.0, 0.0),  # 6
    (1.0, 2.0, 0.0),  # 7
    (2.0, 2.0, 0.0),  # 8
    (0.0, 0.0, 1.0),  # 9
    (1.0, 0.0, 1.0),  # 10
    (2.0, 0.0, 1.0),  # 11
    (0.0, 1.0, 1.0),  # 12
    (1.0, 1.0, 1.0),  # 13
    (2.0, 1.0, 1.0),  # 14
    (0.0, 2.0, 1.0),  # 15
    (1.0, 2.0, 1.0),  # 16
    (2.0, 2.0, 1.0),  # 17
]


def _build_model_with_soft_mesh(vertices: list[tuple[float, float, float]], tets: np.ndarray):
    """Use add_soft_mesh (full builder path) to create a soft-body model."""
    builder = ModelBuilder()
    builder.add_soft_mesh(
        pos=(0.0, 0.0, 0.0),
        rot=wp.quat_identity(),
        scale=1.0,
        vel=(0.0, 0.0, 0.0),
        vertices=vertices,
        indices=tets.flatten().tolist(),
        density=1.0,
        k_mu=1.0,
        k_lambda=1.0,
        k_damp=0.0,
    )
    builder.color()
    return builder.finalize(device="cpu")


def _expected_tet_adjacency(particle_count: int, tet_indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Enumerate adjacency exactly as the kernels do: append (tet_id, local_order) per vertex."""
    buckets = [[] for _ in range(particle_count)]
    for tet_id, tet in enumerate(tet_indices):
        for local_order, v in enumerate(tet):
            buckets[int(v)].append((tet_id, local_order))
    offsets = [0]
    flat = []
    for b in buckets:
        offsets.append(offsets[-1] + 2 * len(b))
        for tet_id, order in b:
            flat.extend([tet_id, order])
    return np.array(offsets, dtype=np.int32), np.array(flat, dtype=np.int32)


def _assert_adjacency_matches_tets(test, adjacency, tet_indices: np.ndarray):
    """Check each recorded (tet_id, local_order) really maps back to the vertex being visited."""
    offsets = adjacency.v_adj_tets_offsets.numpy()
    flat = adjacency.v_adj_tets.numpy()
    particle_count = len(offsets) - 1
    for v in range(particle_count):
        start, end = offsets[v], offsets[v + 1]
        entries = flat[start:end].reshape(-1, 2)
        for tet_id, local_order in entries:
            test.assertTrue(
                tet_indices[tet_id, local_order] == v, f"vertex {v} mismatch tet {tet_id} order {local_order}"
            )


def _color_groups_to_array(test, particle_count: int, color_groups: list[np.ndarray]) -> np.ndarray:
    """Convert the builder's color groups into a per-particle color array and validate assignment."""
    colors = -np.ones(particle_count, dtype=np.int32)
    for color_id, group in enumerate(color_groups):
        for vertex in np.asarray(group, dtype=np.int32):
            test.assertEqual(
                colors[vertex],
                -1,
                f"vertex {vertex} assigned to multiple color groups",
            )
            colors[vertex] = color_id
    test.assertFalse(
        np.any(colors < 0),
        "some particles were not assigned a color during mesh coloring",
    )
    return colors


def _assert_tet_graph_coloring(test, tet_indices: np.ndarray, colors: np.ndarray):
    """Ensure no two connected vertices share the same color."""
    for tet in tet_indices:
        for i in range(len(tet)):
            for j in range(i + 1, len(tet)):
                v0 = int(tet[i])
                v1 = int(tet[j])
                test.assertNotEqual(
                    colors[v0],
                    colors[v1],
                    f"vertices {v0} and {v1} share color {colors[v0]}",
                )


class TestSoftBody(unittest.TestCase):
    pass


def test_tet_adjacency_single_tet(test, device, solver):
    tet_indices = np.array([[0, 1, 2, 3]], dtype=np.int32)
    particles = [
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
    ]
    model = _build_model_with_soft_mesh(particles, tet_indices)

    solver = SolverVBD()
    solver.model = model
    solver.device = model.device

    adjacency = solver.compute_force_element_adjacency(model)

    exp_offsets, exp_flat = _expected_tet_adjacency(4, tet_indices)
    np.testing.assert_array_equal(adjacency.v_adj_tets_offsets.numpy(), exp_offsets)
    np.testing.assert_array_equal(adjacency.v_adj_tets.numpy(), exp_flat)
    _assert_adjacency_matches_tets(adjacency, tet_indices)


def test_tet_adjacency_complex_pyramid(test, device):
    model = _build_model_with_soft_mesh(PYRAMID_PARTICLES, PYRAMID_TET_INDICES)

    solver = SolverVBD(model)

    adjacency = solver.compute_particle_force_element_adjacency()

    exp_offsets, exp_flat = _expected_tet_adjacency(len(PYRAMID_PARTICLES), PYRAMID_TET_INDICES)
    np.testing.assert_array_equal(adjacency.v_adj_tets_offsets.numpy(), exp_offsets)
    np.testing.assert_array_equal(adjacency.v_adj_tets.numpy(), exp_flat)
    _assert_adjacency_matches_tets(test, adjacency, PYRAMID_TET_INDICES)


def test_tet_graph_coloring_is_valid(test, device):
    """Color a small tetrahedral mesh and verify the coloring respects graph adjacency."""
    builder = ModelBuilder()
    builder.add_soft_mesh(
        pos=(0.0, 0.0, 0.0),
        rot=wp.quat_identity(),
        scale=1.0,
        vel=(0.0, 0.0, 0.0),
        vertices=PYRAMID_PARTICLES,
        indices=PYRAMID_TET_INDICES.flatten().tolist(),
        density=1.0,
        k_mu=1.0,
        k_lambda=1.0,
        k_damp=0.0,
    )
    builder.color()

    colors = _color_groups_to_array(test, len(PYRAMID_PARTICLES), builder.particle_color_groups)
    _assert_tet_graph_coloring(test, PYRAMID_TET_INDICES, colors)


def test_tet_energy(test, device):
    rng = np.random.default_rng(seed=42)

    for _test in range(100):
        builder = ModelBuilder()

        vertices = [wp.vec3(rng.standard_normal((3,))) for _ in range(4)]

        p = np.array(vertices[0])
        q = np.array(vertices[1])
        r = np.array(vertices[2])
        s = np.array(vertices[3])

        qp = q - p
        rp = r - p
        sp = s - p

        Dm = np.array((qp, rp, sp)).T
        volume = np.linalg.det(Dm) / 6.0

        if volume < 0:
            vertices = [
                vertices[1],
                vertices[0],
                vertices[2],
                vertices[3],
            ]

        tet_indices = [0, 1, 2, 3]

        builder.add_soft_mesh(
            vertices=vertices,
            indices=tet_indices,
            rot=wp.quat_identity(),
            pos=wp.vec3(0.0),
            vel=wp.vec3(0.0),
            density=1000.0,
            scale=1.0,
            k_mu=rng.standard_normal(),
            k_lambda=rng.standard_normal(),
            k_damp=0.0,
        )
        dt = 0.001666

        model = builder.finalize(requires_grad=True)
        tet_energy = wp.zeros(1, dtype=float, requires_grad=True)
        particle_forces = wp.zeros(12, dtype=float, requires_grad=True)
        particle_hessian = wp.zeros(4, dtype=wp.mat33, requires_grad=False)

        state = model.state(requires_grad=True)
        state.particle_q.assign(state.particle_q.numpy() + rng.standard_normal((4, 3)))

        with wp.Tape() as tape:
            wp.launch(
                dim=1,
                kernel=compute_neo_hookean_energy_and_force,
                inputs=[
                    0,
                    dt,
                    state.particle_q,
                    model.tet_indices,
                    model.tet_poses,
                    model.tet_materials,
                    tet_energy,
                    particle_forces,
                ],
            )

        tape.backward(tet_energy)

        particle_force_auto_diff = -state.particle_q.grad.numpy()
        particle_forces_analytical_1 = particle_forces.numpy().copy().reshape(4, -1)

        force_autodiff_comparison = np.isclose(
            particle_force_auto_diff, particle_forces_analytical_1, rtol=1.0e-4, atol=0.1
        )
        if not force_autodiff_comparison.all():
            print("\n=== Autodiff Force vs Analytical Force Mismatch ===")
            print("autodiff force:\n", particle_force_auto_diff)
            print("\nanalytical force:\n", particle_forces_analytical_1)
            print("\ndifference:\n", particle_force_auto_diff - particle_forces_analytical_1)
        test.assertTrue(force_autodiff_comparison.all())

        # calculate hessians using auto diff
        particle_hessian_auto_diff = np.zeros((4, 3, 3), dtype=np.float32)

        def onehot(i, ndim):
            x = np.zeros(ndim, dtype=np.float32)
            x[i] = 1.0
            return wp.array(
                x,
            )

        for v_counter in range(4):
            for dim in range(3):
                tape.zero()
                tape.backward(grads={particle_forces: onehot(v_counter * 3 + dim, 12)})
                # force is the negative gradient so the hessian is the negative jacobian of it
                particle_hessian_auto_diff[v_counter, dim, :] = -state.particle_q.grad.numpy()[v_counter, :]

        particle_forces_vec3 = wp.zeros_like(state.particle_q)
        wp.launch(
            dim=4,
            kernel=compute_neo_hookean_energy_and_force_and_hessian,
            inputs=[
                0,
                dt,
                state.particle_q,
                model.tet_indices,
                model.tet_poses,
                model.tet_materials,
                particle_forces_vec3,
                particle_hessian,
            ],
        )
        particle_forces_analytical_2 = particle_forces_vec3.numpy()
        particle_hessian_analytical = particle_hessian.numpy()

        force_comparison = np.isclose(particle_forces_analytical_2, particle_forces_analytical_1, rtol=1.0e-4, atol=0.1)
        if not force_comparison.all():
            print("\n=== Force Mismatch ===")
            print("force from compute_neo_hookean_energy_and_force:\n", particle_forces_analytical_1)
            print("\nforce from compute_neo_hookean_energy_and_force_and_hessian:\n", particle_forces_analytical_2)
            print("\ndifference:\n", particle_forces_analytical_2 - particle_forces_analytical_1)
        test.assertTrue(force_comparison.all())

        for i in range(4):
            test.assertTrue(
                np.isclose(particle_hessian_auto_diff[i], particle_hessian_analytical[i], rtol=1.0e-2, atol=0.1).all()
            )


devices = get_test_devices(mode="basic")
add_function_test(
    TestSoftBody, "test_tet_adjacency_complex_pyramid", test_tet_adjacency_complex_pyramid, devices=devices
)
add_function_test(TestSoftBody, "test_tet_graph_coloring_is_valid", test_tet_graph_coloring_is_valid, devices=devices)
add_function_test(TestSoftBody, "test_tet_energy", test_tet_energy, devices=devices)

if __name__ == "__main__":
    unittest.main(verbosity=2, failfast=True)
