# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.core.types import vec5
from newton.solvers import SolverMuJoCo


class TestSolverMuJoCoPlanarMesh(unittest.TestCase):
    def setUp(self):
        try:
            SolverMuJoCo.import_mujoco()
        except ImportError as exc:
            self.skipTest(str(exc))

    @staticmethod
    def _build_mesh_model(vertices, indices):
        builder = newton.ModelBuilder()
        body = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 1.0), wp.quat_identity()))
        builder.add_shape_sphere(body=body, radius=0.01)
        mesh = newton.Mesh(vertices=vertices, indices=indices, compute_inertia=False)
        builder.add_shape_mesh(body=-1, mesh=mesh, label="flat_mesh")
        return builder.finalize(device="cpu")

    def test_planar_quad_compiles_with_newton_contacts(self):
        vertices = np.array(
            [
                [-5.0, -5.0, 0.0],
                [5.0, -5.0, 0.0],
                [-5.0, 5.0, 0.0],
                [5.0, 5.0, 0.0],
            ],
            dtype=np.float32,
        )
        indices = np.array([0, 1, 2, 1, 3, 2], dtype=np.int32)
        model = self._build_mesh_model(vertices, indices)

        solver = SolverMuJoCo(model, use_mujoco_cpu=True, use_mujoco_contacts=False)

        self.assertEqual(solver.mj_model.nmesh, 1)
        self.assertEqual(solver.mj_model.mesh_vertnum[0], 5)
        self.assertEqual(solver.mj_model.mesh_facenum[0], 3)
        self.assertEqual(model.shape_source[1].vertices.shape[0], 4)
        self.assertEqual(model.shape_source[1].indices.shape[0], 6)

    def test_planar_triangle_compiles_with_newton_contacts(self):
        vertices = np.array(
            [
                [-1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=np.float32,
        )
        indices = np.array([0, 1, 2], dtype=np.int32)
        model = self._build_mesh_model(vertices, indices)

        solver = SolverMuJoCo(model, use_mujoco_cpu=True, use_mujoco_contacts=False)

        self.assertEqual(solver.mj_model.nmesh, 1)
        self.assertEqual(solver.mj_model.mesh_vertnum[0], 4)
        self.assertEqual(solver.mj_model.mesh_facenum[0], 2)
        self.assertEqual(model.shape_source[1].vertices.shape[0], 3)
        self.assertEqual(model.shape_source[1].indices.shape[0], 3)

    def test_nonplanar_mesh_is_not_inflated(self):
        vertices = np.array(
            [
                [-1.0, -1.0, 0.0],
                [1.0, -1.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        indices = np.array([0, 1, 2, 0, 3, 1, 1, 3, 2, 2, 3, 0], dtype=np.int32)
        model = self._build_mesh_model(vertices, indices)

        solver = SolverMuJoCo(model, use_mujoco_cpu=True, use_mujoco_contacts=False)

        self.assertEqual(solver.mj_model.nmesh, 1)
        self.assertEqual(solver.mj_model.mesh_vertnum[0], 4)
        self.assertEqual(solver.mj_model.mesh_facenum[0], 4)

    def test_planar_mesh_rejects_mujoco_contacts(self):
        vertices = np.array(
            [
                [-5.0, -5.0, 0.0],
                [5.0, -5.0, 0.0],
                [-5.0, 5.0, 0.0],
                [5.0, 5.0, 0.0],
            ],
            dtype=np.float32,
        )
        indices = np.array([0, 1, 2, 1, 3, 2], dtype=np.int32)
        model = self._build_mesh_model(vertices, indices)

        with self.assertRaisesRegex(ValueError, "planar mesh collider"):
            SolverMuJoCo(model, use_mujoco_cpu=True, use_mujoco_contacts=True)

    def test_planar_mesh_rejects_explicit_mujoco_pair_contacts(self):
        vertices = np.array(
            [
                [-5.0, -5.0, 0.0],
                [5.0, -5.0, 0.0],
                [-5.0, 5.0, 0.0],
                [5.0, 5.0, 0.0],
            ],
            dtype=np.float32,
        )
        indices = np.array([0, 1, 2, 1, 3, 2], dtype=np.int32)

        builder = newton.ModelBuilder()
        SolverMuJoCo.register_custom_attributes(builder)
        body = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 1.0), wp.quat_identity()))
        sphere = builder.add_shape_sphere(body=body, radius=0.01)

        cfg = newton.ModelBuilder.ShapeConfig()
        cfg.has_shape_collision = False
        cfg.has_particle_collision = False
        cfg.collision_group = 0
        mesh = newton.Mesh(vertices=vertices, indices=indices, compute_inertia=False)
        flat_mesh = builder.add_shape_mesh(body=-1, mesh=mesh, cfg=cfg, label="flat_mesh")
        builder.add_custom_values(
            **{
                "mujoco:pair_world": 0,
                "mujoco:pair_geom1": flat_mesh,
                "mujoco:pair_geom2": sphere,
                "mujoco:pair_condim": 3,
                "mujoco:pair_solref": wp.vec2(0.02, 1.0),
                "mujoco:pair_solreffriction": wp.vec2(0.02, 1.0),
                "mujoco:pair_solimp": vec5(0.9, 0.95, 0.001, 0.5, 2.0),
                "mujoco:pair_margin": 0.0,
                "mujoco:pair_gap": 0.0,
                "mujoco:pair_friction": vec5(1.0, 1.0, 0.005, 0.0001, 0.0001),
            }
        )
        model = builder.finalize(device="cpu")

        with self.assertRaisesRegex(ValueError, "planar mesh collider"):
            SolverMuJoCo(model, use_mujoco_cpu=True, use_mujoco_contacts=True)


if __name__ == "__main__":
    unittest.main()
