# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Constructor guard for heterogeneous multi-world models in SolverFeatherPGS.

Heterogeneous multi-world models (worlds whose per-world DOF counts differ)
are only safe on the matrix-free and split paths with the default
``articulated_contact_response="immediate"``: the dense path produces
deterministic wrong trajectories, and the propagation-family contact
responses use fixed-width per-world velocity windows that silently corrupt
velocities across world boundaries. These tests assert that the constructor
rejects the unsafe combinations with ``ValueError`` while the safe
combinations — and every mode on a homogeneous model — still construct.
"""

from __future__ import annotations

import unittest

import warp as wp

import newton
from newton.solvers import SolverFeatherPGS

PROPAGATION_RESPONSES = ("propagation", "propagation-fused", "propagation-colored")


def _make_chain_world(n_links: int) -> newton.ModelBuilder:
    """A fixed-base serial chain of ``n_links`` revolute links (n_links DOFs)."""
    builder = newton.ModelBuilder()
    joints = []
    prev = -1
    for i in range(n_links):
        link = builder.add_link()
        builder.add_shape_box(link, hx=0.15, hy=0.03, hz=0.03)
        if prev == -1:
            parent_xform = wp.transform(p=wp.vec3(0.0, 0.0, 1.0), q=wp.quat_identity())
        else:
            parent_xform = wp.transform(p=wp.vec3(0.15, 0.0, 0.0), q=wp.quat_identity())
        joints.append(
            builder.add_joint_revolute(
                parent=prev,
                child=link,
                axis=wp.vec3(0.0, 1.0, 0.0),
                parent_xform=parent_xform,
                child_xform=wp.transform(p=wp.vec3(-0.15, 0.0, 0.0), q=wp.quat_identity()),
                label=f"chain_joint_{i}",
            )
        )
        prev = link
    builder.add_articulation(joints, label="chain")
    return builder


def _build_model(world_link_counts: list[int], device: str) -> newton.Model:
    """One chain world per entry in ``world_link_counts``, plus a ground plane."""
    scene = newton.ModelBuilder()
    for n_links in world_link_counts:
        scene.add_world(_make_chain_world(n_links))
    scene.add_ground_plane()
    return scene.finalize(device=device)


class TestFeatherPGSHeteroGuard(unittest.TestCase):
    """Unsafe hetero-world configurations must fail at construction."""

    @classmethod
    def setUpClass(cls):
        if wp.get_cuda_device_count() == 0:
            raise unittest.SkipTest("SolverFeatherPGS construction tests require cuda:0")
        cls.device = "cuda:0"
        # Worlds alternate a 1-link pendulum and a 3-link chain: per-world DOF
        # counts [1, 3, 1, 3] -> heterogeneous.
        cls.hetero_model = _build_model([1, 3, 1, 3], cls.device)
        # All 3-link chains: per-world DOF counts uniform -> homogeneous.
        cls.homogeneous_model = _build_model([3, 3, 3, 3], cls.device)

    def test_hetero_propagation_family_raises(self):
        for response in PROPAGATION_RESPONSES:
            with self.subTest(articulated_contact_response=response):
                with self.assertRaises(ValueError) as ctx:
                    SolverFeatherPGS(
                        self.hetero_model,
                        pgs_mode="matrix_free",
                        articulated_contact_response=response,
                    )
                message = str(ctx.exception)
                self.assertIn(response, message)
                self.assertIn("heterogeneous", message)
                self.assertIn("1, 3", message)

    def test_hetero_matrix_free_constructs(self):
        solver = SolverFeatherPGS(self.hetero_model, pgs_mode="matrix_free")
        self.assertEqual(solver.pgs_mode, "matrix_free")

    def test_hetero_split_constructs(self):
        solver = SolverFeatherPGS(self.hetero_model, pgs_mode="split")
        self.assertEqual(solver.pgs_mode, "split")

    def test_homogeneous_all_modes_construct(self):
        for pgs_mode, response in (
            ("split", "immediate"),
            ("matrix_free", "immediate"),
            ("matrix_free", "propagation"),
            ("matrix_free", "propagation-fused"),
            ("matrix_free", "propagation-colored"),
        ):
            with self.subTest(pgs_mode=pgs_mode, articulated_contact_response=response):
                solver = SolverFeatherPGS(
                    self.homogeneous_model,
                    pgs_mode=pgs_mode,
                    articulated_contact_response=response,
                )
                self.assertEqual(solver.pgs_mode, pgs_mode)
                self.assertEqual(solver.articulated_contact_response, response)


if __name__ == "__main__":
    unittest.main()
