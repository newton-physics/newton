# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import os
import unittest

import numpy as np
import warp as wp

import newton
from newton.tests.unittest_utils import assert_np_equal
from newton.viewer import ViewerNull


class _MeshProbe(ViewerNull):
    """Captures every ``log_mesh`` call keyed by object name."""

    def __init__(self):
        super().__init__(num_frames=1)
        self.calls: dict[str, dict] = {}

    def log_mesh(
        self,
        name,
        points,
        indices,
        normals=None,
        uvs=None,
        texture=None,
        hidden=False,
        backface_culling=True,
        colors=None,
        **kwargs,
    ):
        self.calls[name] = {
            "points": None if points is None else points.numpy(),
            "indices": None if indices is None else indices.numpy(),
            "normals": None if normals is None else normals.numpy(),
            "uvs": None if uvs is None else uvs.numpy(),
            "texture": texture,
            "hidden": hidden,
            "colors": None if colors is None else colors.numpy(),
        }


def _cloth_model():
    builder = newton.ModelBuilder()
    builder.add_cloth_grid(
        pos=(0.0, 0.0, 0.0),
        rot=wp.quat_identity(),
        vel=(0.0, 0.0, 0.0),
        dim_x=3,
        dim_y=3,
        cell_x=0.5,
        cell_y=0.5,
        mass=0.1,
    )
    n = builder.particle_count
    verts = np.array(builder.particle_q, dtype=np.float32)
    indices = np.array([0, 1, 2, 1, 3, 2], dtype=np.int32)
    uvs = np.linspace(0.0, 1.0, n * 2, dtype=np.float32).reshape(n, 2)
    builder.add_deformable_render_mesh(
        verts,
        indices,
        kind="cloth",
        particle_indices=np.arange(n, dtype=np.int32),
        uvs=uvs,
        label="skin",
    )
    return builder.finalize(), uvs


def _soft_model(render_verts):
    builder = newton.ModelBuilder()
    builder.add_soft_grid(
        pos=wp.vec3(0.0, 0.0, 0.0),
        rot=wp.quat_identity(),
        vel=wp.vec3(0.0, 0.0, 0.0),
        dim_x=2,
        dim_y=2,
        dim_z=2,
        cell_x=0.5,
        cell_y=0.5,
        cell_z=0.5,
        density=100.0,
        k_mu=1.0e4,
        k_lambda=1.0e4,
        k_damp=0.0,
    )
    idx = np.array([0, 1, 2, 0, 2, 3], dtype=np.int32)
    builder.add_deformable_render_mesh(render_verts, idx, kind="tet", label="skin")
    return builder.finalize()


class TestDeformableRenderMesh(unittest.TestCase):
    def test_cloth_skin_follows_particles(self):
        """A cloth render mesh with a 1:1 particle map must equal particle_q."""
        model, uvs = _cloth_model()
        self.assertEqual(model.deformable_render_mesh_count, 1)
        self.assertEqual(model.deformable_render_meshes[0].kind, newton.DeformableRenderKind.CLOTH_SHARED)

        viewer = _MeshProbe()
        viewer.set_model(model)
        state = model.state()
        moved = state.particle_q.numpy()
        moved[:, 2] += 1.5
        state.particle_q = wp.array(moved, dtype=wp.vec3)

        viewer.begin_frame(0.0)
        viewer.log_state(state)
        viewer.end_frame()

        call = viewer.calls["/model/render_meshes/skin"]
        self.assertFalse(call["hidden"])
        assert_np_equal(call["points"], moved, tol=1.0e-5)
        # UVs are preserved through the log_mesh path.
        assert_np_equal(call["uvs"], uvs, tol=1.0e-6)

    def test_render_mesh_suppresses_sim_triangles(self):
        """When a render mesh is shown the bare triangle view is hidden."""
        model, _ = _cloth_model()
        viewer = _MeshProbe()
        viewer.set_model(model)
        state = model.state()

        viewer.begin_frame(0.0)
        viewer.log_state(state)
        viewer.end_frame()
        self.assertTrue(viewer.calls["/model/triangles"]["hidden"])

        # Toggling render meshes off restores the triangle view.
        viewer.show_render_mesh = False
        viewer.begin_frame(1.0)
        viewer.log_state(state)
        viewer.end_frame()
        self.assertFalse(viewer.calls["/model/triangles"]["hidden"])

    def test_tet_embedding_partition_of_unity(self):
        """Barycentric weights sum to one and reference valid tets."""
        verts = np.array(
            [[0.25, 0.25, 0.25], [0.75, 0.75, 0.75], [0.5, 0.5, 0.5], [0.1, 0.9, 0.4]],
            dtype=np.float32,
        )
        model = _soft_model(verts)
        rm = model.deformable_render_meshes[0]
        self.assertEqual(rm.kind, newton.DeformableRenderKind.TET_EMBED)
        weights = rm.weights.numpy()
        assert_np_equal(weights.sum(axis=1), np.ones(len(verts), dtype=np.float32), tol=1.0e-5)
        parents = rm.parent.numpy()
        self.assertTrue((parents >= 0).all() and (parents < model.tet_count).all())

    def test_tet_skin_rigid_motion(self):
        """Embedded render verts must follow a rigid motion of the soft body exactly."""
        verts = np.array(
            [[0.25, 0.25, 0.25], [0.75, 0.75, 0.75], [0.5, 0.5, 0.5], [0.1, 0.9, 0.4]],
            dtype=np.float32,
        )
        model = _soft_model(verts)
        viewer = _MeshProbe()
        viewer.set_model(model)

        # Pure translation.
        state = model.state()
        shift = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        state.particle_q = wp.array(state.particle_q.numpy() + shift, dtype=wp.vec3)
        viewer.begin_frame(0.0)
        viewer.log_state(state)
        viewer.end_frame()
        assert_np_equal(viewer.calls["/model/render_meshes/skin"]["points"], verts + shift, tol=1.0e-4)

        # Pure rotation about z by 90 degrees.
        rot = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
        state2 = model.state()
        state2.particle_q = wp.array((model.state().particle_q.numpy() @ rot.T).astype(np.float32), dtype=wp.vec3)
        viewer.begin_frame(1.0)
        viewer.log_state(state2)
        viewer.end_frame()
        assert_np_equal(viewer.calls["/model/render_meshes/skin"]["points"], verts @ rot.T, tol=1.0e-4)

    def test_strain_coloring_emits_per_vertex_colors(self):
        """Enabling strain coloring routes per-vertex jet colors through log_mesh."""
        verts = np.array(
            [[0.25, 0.25, 0.25], [0.75, 0.75, 0.75], [0.5, 0.5, 0.5], [0.1, 0.9, 0.4]],
            dtype=np.float32,
        )
        model = _soft_model(verts)
        viewer = _MeshProbe()
        viewer.set_model(model)
        viewer.show_render_strain = True

        state = model.state()
        moved = state.particle_q.numpy()
        moved[:, 2] *= 1.5
        state.particle_q = wp.array(moved, dtype=wp.vec3)
        viewer.begin_frame(0.0)
        viewer.log_state(state)
        viewer.end_frame()

        call = viewer.calls["/model/render_meshes/skin"]
        self.assertFalse(call["hidden"])
        self.assertIsNotNone(call["colors"])
        self.assertEqual(call["colors"].shape, (len(verts), 3))
        # Jet colors are bounded to [0, 1].
        self.assertGreaterEqual(float(call["colors"].min()), 0.0)
        self.assertLessEqual(float(call["colors"].max()), 1.0)

    def test_rigid_skin_follows_bodies(self):
        """A rigid render mesh follows a rigid-body translation exactly."""
        builder = newton.ModelBuilder()
        nodes = np.stack([np.linspace(0.0, 1.0, 11), np.zeros(11), np.full(11, 1.0)], axis=1).astype(np.float32)
        bodies, _ = builder.add_rod(positions=[wp.vec3(*p) for p in nodes], radius=0.05, body_frame_origin="com")
        # A simple ring of render vertices around each node.
        verts, idx = [], []
        ring = 6
        for p in nodes:
            for j in range(ring):
                ang = 2.0 * np.pi * j / ring
                verts.append(p + 0.06 * np.array([0.0, np.cos(ang), np.sin(ang)], dtype=np.float32))
        for i in range(len(nodes) - 1):
            for j in range(ring):
                a, b, c, d = (
                    i * ring + j,
                    i * ring + (j + 1) % ring,
                    (i + 1) * ring + j,
                    (i + 1) * ring + (j + 1) % ring,
                )
                idx += [a, c, b, b, c, d]
        verts = np.asarray(verts, dtype=np.float32)
        builder.add_deformable_render_mesh(
            verts, np.asarray(idx, dtype=np.int32), kind="rigid", bodies=bodies, label="skin"
        )
        model = builder.finalize()
        rm = model.deformable_render_meshes[0]
        self.assertEqual(rm.kind, newton.DeformableRenderKind.RIGID_BODY)
        self.assertIsNotNone(rm.local_offsets)

        viewer = _MeshProbe()
        viewer.set_model(model)
        state = model.state()
        shift = np.array([0.5, -0.3, 0.2], dtype=np.float32)
        bq = state.body_q.numpy()
        bq[:, :3] += shift
        state.body_q = wp.array(bq, dtype=wp.transform)
        viewer.begin_frame(0.0)
        viewer.log_state(state)
        viewer.end_frame()
        assert_np_equal(viewer.calls["/model/render_meshes/skin"]["points"], verts + shift, tol=1.0e-4)

    def test_add_builder_rebases_parent_indices(self):
        """Merging a sub-builder must offset render-mesh driver indices."""
        sub = newton.ModelBuilder()
        sub.add_cloth_grid(
            pos=(0.0, 0.0, 0.0),
            rot=wp.quat_identity(),
            vel=(0.0, 0.0, 0.0),
            dim_x=2,
            dim_y=2,
            cell_x=0.5,
            cell_y=0.5,
            mass=0.1,
        )
        n = sub.particle_count
        verts = np.array(sub.particle_q, dtype=np.float32)
        sub.add_deformable_render_mesh(
            verts,
            np.array([0, 1, 2], dtype=np.int32),
            kind="cloth",
            particle_indices=np.arange(n, dtype=np.int32),
            label="skin",
        )

        main = newton.ModelBuilder()
        main.add_particle(pos=(9.0, 9.0, 9.0), vel=(0.0, 0.0, 0.0), mass=1.0)
        offset = main.particle_count
        main.add_builder(sub)
        model = main.finalize()

        parent = model.deformable_render_meshes[0].parent.numpy()
        assert_np_equal(parent, np.arange(n, dtype=np.int32) + offset)


class TestDeformableRenderMeshUSDImport(unittest.TestCase):
    def test_usd_render_mesh_relationship_import(self):
        """A ``newton:renderMesh`` rel on a TetMesh imports an embedded render mesh."""
        try:
            from pxr import Usd, UsdGeom
        except ImportError:
            self.skipTest("usd-core not available")

        asset = os.path.join(os.path.dirname(__file__), "assets", "tetmesh_simple.usda")
        stage = Usd.Stage.Open(asset)
        tet_prim = stage.GetPrimAtPath("/SimpleTetMesh")
        self.assertTrue(tet_prim.IsValid())

        pts = np.array(UsdGeom.TetMesh(tet_prim).GetPointsAttr().Get(), dtype=np.float32)
        # Author the render mesh as a child so it shares the tet's world xform;
        # reuse tet vertices so every render vertex embeds cleanly.
        render = UsdGeom.Mesh.Define(stage, "/SimpleTetMesh/RenderSkin")
        render.GetPointsAttr().Set([tuple(float(c) for c in p) for p in pts[:4]])
        render.GetFaceVertexCountsAttr().Set([3, 3])
        render.GetFaceVertexIndicesAttr().Set([0, 1, 2, 0, 2, 3])
        tet_prim.CreateRelationship("newton:renderMesh").SetTargets(["/SimpleTetMesh/RenderSkin"])

        builder = newton.ModelBuilder()
        builder.add_usd(stage)
        self.assertEqual(len(builder._deformable_render_meshes), 1)

        model = builder.finalize()
        self.assertEqual(model.deformable_render_mesh_count, 1)
        rm = model.deformable_render_meshes[0]
        self.assertEqual(rm.kind, newton.DeformableRenderKind.TET_EMBED)
        assert_np_equal(rm.weights.numpy().sum(axis=1), np.ones(4, dtype=np.float32), tol=1.0e-4)


if __name__ == "__main__":
    unittest.main()
