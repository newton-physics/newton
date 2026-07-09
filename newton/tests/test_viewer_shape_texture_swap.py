# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import unittest

import warp as wp

import newton
from newton.viewer import ViewerNull


class _ViewerTextureSwapProbe(ViewerNull):
    """A minimal viewer probe for testing per-shape texture swapping.

    Uses `ViewerNull` to avoid any rendering backend dependencies. Simulates a
    texture pool (normally backed by the concrete viewer, e.g. ViewerGL's
    GL_TEXTURE_2D_ARRAY) with a plain dict, and records mesh-level fallback calls.
    """

    def __init__(self):
        super().__init__(num_frames=1)
        self.pool_layers = {"pool_a.png": 0, "pool_b.png": 1}
        self.mesh_texture_updates = []
        self.logged_materials = []

    def _texture_pool_layer(self, key: str) -> int | None:
        return self.pool_layers.get(key)

    def update_mesh_texture(self, name: str, texture) -> None:
        self.mesh_texture_updates.append((name, texture))

    def log_instances(self, name, mesh, xforms, scales, colors, materials, hidden=False):
        self.logged_materials.append(materials)


def _build_shared_geometry_model() -> newton.Model:
    """Three identically-sized boxes: content-identical geometry is logged once."""
    builder = newton.ModelBuilder()
    for _ in range(3):
        body = builder.add_body(
            xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
            mass=1.0,
        )
        builder.add_shape_box(body=body, hx=0.1, hy=0.1, hz=0.1)
    return builder.finalize()


class TestViewerShapeTextureSwap(unittest.TestCase):
    def test_pool_texture_applies_per_instance_not_per_geometry(self):
        """Shapes sharing deduplicated geometry must be able to show different textures."""
        model = _build_shared_geometry_model()
        viewer = _ViewerTextureSwapProbe()
        viewer.show_collision = True
        viewer.set_model(model)

        batches = list(viewer._shape_instances.values())
        self.assertEqual(len(batches), 1)
        batch = batches[0]
        self.assertEqual(len(batch.model_shapes), 3)

        target_shape = batch.model_shapes[1]
        viewer.update_shape_textures([target_shape], "pool_b.png")

        materials = batch.materials.numpy()
        for slot, shape in enumerate(batch.model_shapes):
            if shape == target_shape:
                # shader convention: values >= 2 sample pool layer (value - 2)
                self.assertEqual(materials[slot, 3], float(viewer.pool_layers["pool_b.png"] + 2))
            else:
                self.assertLess(materials[slot, 3], 2.0)
        self.assertTrue(batch.materials_changed)
        # pool path must not fall back to mesh-level updates
        self.assertEqual(viewer.mesh_texture_updates, [])

    def test_materials_changed_flag_reaches_log_instances_once(self):
        """Layer writes must be passed to log_instances on the next frame, then settle."""
        model = _build_shared_geometry_model()
        state = model.state()
        viewer = _ViewerTextureSwapProbe()
        viewer.show_collision = True
        viewer.set_model(model)

        # first frame consumes model_changed; materials are always passed here
        viewer.begin_frame(0.0)
        viewer.log_state(state)

        batch = next(iter(viewer._shape_instances.values()))
        viewer.update_shape_textures([batch.model_shapes[0]], "pool_a.png")

        viewer.logged_materials.clear()
        viewer.begin_frame(1.0)
        viewer.log_state(state)
        self.assertTrue(any(materials is not None for materials in viewer.logged_materials))
        self.assertFalse(batch.materials_changed)

        # steady state: no further material uploads without new writes
        viewer.logged_materials.clear()
        viewer.begin_frame(2.0)
        viewer.log_state(state)
        self.assertTrue(all(materials is None for materials in viewer.logged_materials))

    def test_unregistered_texture_falls_back_to_one_mesh_update(self):
        """Sources missing from the pool update the logged mesh, once per shared geometry."""
        model = _build_shared_geometry_model()
        viewer = _ViewerTextureSwapProbe()
        viewer.show_collision = True
        viewer.set_model(model)

        batch = next(iter(viewer._shape_instances.values()))
        viewer.update_shape_textures(list(batch.model_shapes), "not_in_pool.png")

        # all three shapes share one logged mesh -> exactly one mesh-level update
        self.assertEqual(len(viewer.mesh_texture_updates), 1)
        name, texture = viewer.mesh_texture_updates[0]
        self.assertEqual(name, viewer._shape_mesh_names[int(batch.model_shapes[0])])
        self.assertEqual(texture, "not_in_pool.png")
        self.assertFalse(batch.materials_changed)


if __name__ == "__main__":
    unittest.main(verbosity=2, failfast=False)
