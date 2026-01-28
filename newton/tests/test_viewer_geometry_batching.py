# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
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

import newton
from newton.tests.unittest_utils import assert_np_equal
from newton.viewer import ViewerNull


class _ViewerGeometryBatchingProbe(ViewerNull):
    """A minimal viewer probe for testing viewer-side batching/caching.

    Uses `ViewerNull` to avoid any rendering backend dependencies.
    """

    def __init__(self):
        super().__init__(num_frames=1)
        self.log_capsules_calls = 0
        self.log_instances_calls = 0

    def _hash_geometry(self, geo_type: int, geo_scale, thickness: float, is_solid: bool, geo_src=None) -> int:
        # Match ViewerGL's capsule batching behavior: ignore capsule dimensions in the hash so
        # varying-radius / varying-half_height capsules can share one cached geometry identity.
        if geo_type == newton.GeoType.CAPSULE:
            geo_scale = (1.0, 1.0)
        return super()._hash_geometry(geo_type, geo_scale, thickness, is_solid, geo_src)

    def set_model(self, model, max_worlds=None):
        super().set_model(model, max_worlds=max_worlds)
        if self.model is None:
            return

        # ViewerGL uses instanced cylinder + spheres and expects capsule body instance scales to be
        # `(radius, radius, half_height)`. Since this probe shares the same capsule hashing behavior
        # (dimensions ignored), we re-materialize the per-shape dimensions into per-instance scales.
        shape_scale_np = self.model.shape_scale.numpy()

        for batch in self._shape_instances.values():
            if batch.geo_type != newton.GeoType.CAPSULE:
                continue

            idxs = list(batch.model_shapes)
            scales = [
                wp.vec3(
                    float(shape_scale_np[s][0]),
                    float(shape_scale_np[s][0]),
                    float(shape_scale_np[s][1]),
                )
                for s in idxs
            ]
            batch.scales = wp.array(scales, dtype=wp.vec3, device=self.device)

    def log_instances(self, *_args, **_kwargs):
        self.log_instances_calls += 1

    def log_capsules(self, name, mesh, xforms, scales, colors, materials, hidden=False):
        self.log_capsules_calls += 1

        # Fallback behavior: treat capsule batches like any other instanced geometry.
        self.log_instances(name, mesh, xforms, scales, colors, materials, hidden=hidden)


class TestViewerGeometryBatching(unittest.TestCase):
    def test_capsule_geometry_is_batched_across_scales(self):
        """Varying capsule dimensions should share one cached capsule geometry."""
        builder = newton.ModelBuilder()
        body = builder.add_body(
            xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
            mass=1.0,
        )

        # Three capsules with different (radius, half_height).
        builder.add_shape_capsule(body=body, radius=0.02, half_height=0.10)
        builder.add_shape_capsule(body=body, radius=0.03, half_height=0.15)
        builder.add_shape_capsule(body=body, radius=0.05, half_height=0.07)

        model = builder.finalize()
        viewer = _ViewerGeometryBatchingProbe()
        viewer.show_collision = True
        viewer.set_model(model)

        # All capsule shapes should end up in a single capsule batch (same flags/static).
        capsule_batches = [b for b in viewer._shape_instances.values() if b.geo_type == newton.GeoType.CAPSULE]
        self.assertEqual(len(capsule_batches), 1)

        batch = capsule_batches[0]

        # Geometry caching check: the capsule batch should reference one cached geometry path,
        # even though the model contains capsules with different (radius, half_height).
        self.assertIsInstance(batch.mesh, str)
        self.assertIn(batch.mesh, viewer._geometry_cache.values())

        # This test builds a capsules-only model, so the viewer should only have to create/cache
        # a single geometry mesh path in total.
        self.assertEqual(len(set(viewer._geometry_cache.values())), 1)
        self.assertEqual(len(batch.scales), 3)

        # Scales should be rewritten to (r, r, half_height) per shape.
        scales_np = batch.scales.numpy()
        expected = np.array(
            [
                [0.02, 0.02, 0.10],
                [0.03, 0.03, 0.15],
                [0.05, 0.05, 0.07],
            ],
            dtype=np.float32,
        )

        scales_sorted = scales_np[np.lexsort((scales_np[:, 2], scales_np[:, 0]))]
        expected_sorted = expected[np.lexsort((expected[:, 2], expected[:, 0]))]
        assert_np_equal(scales_sorted, expected_sorted, tol=1e-6)

    def test_log_state_routes_capsules_to_log_capsules(self):
        """`ViewerBase.log_state()` should dispatch capsule batches via `log_capsules()`."""
        builder = newton.ModelBuilder()
        body = builder.add_body(
            xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
            mass=1.0,
        )
        builder.add_shape_capsule(body=body, radius=0.05, half_height=0.2)
        builder.add_shape_box(body=body, hx=0.1, hy=0.1, hz=0.1)

        model = builder.finalize()
        state = model.state()

        viewer = _ViewerGeometryBatchingProbe()
        viewer.show_collision = True
        viewer.set_model(model)
        viewer.begin_frame(0.0)
        viewer.log_state(state)

        self.assertGreaterEqual(viewer.log_capsules_calls, 1)
        self.assertGreaterEqual(viewer.log_instances_calls, 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
