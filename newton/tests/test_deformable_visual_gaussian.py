# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for Gaussian visual payloads embedded in deformable bodies."""

import math
import unittest

import numpy as np
import warp as wp

import newton
from newton.sensors import SensorTiledCamera
from newton.tests.unittest_utils import USD_AVAILABLE
from newton.viewer import ViewerNull


def _soft_builder():
    builder = newton.ModelBuilder()
    builder.add_soft_grid(
        pos=wp.vec3(0.0),
        rot=wp.quat_identity(),
        vel=wp.vec3(0.0),
        dim_x=1,
        dim_y=1,
        dim_z=1,
        cell_x=1.0,
        cell_y=1.0,
        cell_z=1.0,
        density=100.0,
        k_mu=1.0e4,
        k_lambda=1.0e4,
        k_damp=0.0,
    )
    return builder


def _quat_matrix(quaternion):
    x, y, z, w = quaternion
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )


class _GaussianProbe(ViewerNull):
    """Record Gaussian assets sent through the ordinary viewer shape path."""

    def __init__(self):
        super().__init__(num_frames=1)
        self.calls = []

    def log_gaussian(self, name, gaussian, xform=None, hidden=False):
        self.calls.append((name, gaussian, hidden))


class TestDeformableVisualGaussianBuilder(unittest.TestCase):
    """Public builder and finalized model behavior."""

    def test_add_tet_bound_gaussian_visual(self):
        """Create a stable model record for a tetrahedron-bound Gaussian field."""
        builder = _soft_builder()
        positions = np.array([[0.2, 0.2, 0.2], [0.4, 0.2, 0.2], [0.2, 0.4, 0.2], [0.2, 0.2, 0.4]], dtype=np.float32)
        gaussian = newton.Gaussian(positions=positions, scales=np.full((4, 3), 0.05, dtype=np.float32))
        index = builder.add_deformable_visual_gaussian(
            gaussian,
            kind="tet",
            tet_range=(0, builder.tet_count),
            parent=np.zeros(4, dtype=np.int32),
            weights=np.full((4, 4), 0.25, dtype=np.float32),
            label="soft_splats",
        )

        model = builder.finalize()

        self.assertEqual(index, 0)
        self.assertEqual(model.deformable_visual_gaussian_count, 1)
        visual = model.deformable_visual_gaussians[index]
        self.assertIsInstance(visual, newton.DeformableVisualGaussian)
        self.assertIs(visual.gaussian, gaussian)
        self.assertEqual(visual.kind, newton.DeformableVisualBinding.Kind.TET)
        self.assertEqual(visual.count, 4)
        self.assertEqual(visual.label, "soft_splats")
        self.assertEqual(visual.index, index)

    def test_replicate_offsets_gaussian_drivers(self):
        """Replicate Gaussian bindings into distinct worlds without copying rest appearance."""
        source = _soft_builder()
        positions = np.array([[0.2, 0.2, 0.2], [0.4, 0.2, 0.2]], dtype=np.float32)
        gaussian = newton.Gaussian(positions=positions, scales=np.full((2, 3), 0.05, dtype=np.float32))
        source.add_deformable_visual_gaussian(
            gaussian,
            kind="tet",
            tet_range=(0, source.tet_count),
            parent=np.array([0, 1], dtype=np.int32),
            weights=np.full((2, 4), 0.25, dtype=np.float32),
            label="soft_splats",
        )

        builder = newton.ModelBuilder()
        builder.replicate(source, 2)
        model = builder.finalize()

        self.assertEqual(model.deformable_visual_gaussian_count, 2)
        self.assertEqual([visual.world for visual in model.deformable_visual_gaussians], [0, 1])
        np.testing.assert_array_equal(model.deformable_visual_gaussians[0].parent.numpy(), [0, 1])
        np.testing.assert_array_equal(
            model.deformable_visual_gaussians[1].parent.numpy(), np.array([0, 1]) + source.tet_count
        )
        self.assertIs(model.deformable_visual_gaussians[0].gaussian, gaussian)
        self.assertIs(model.deformable_visual_gaussians[1].gaussian, gaussian)

    def test_rejects_invalid_gaussian_visual_data(self):
        """Reject malformed appearance and unsupported bindings before finalization."""
        builder = _soft_builder()
        positions = np.array([[0.2, 0.2, 0.2]], dtype=np.float32)
        bad_sh = np.array([[np.nan, 0.0, 0.0]], dtype=np.float32)
        gaussian = newton.Gaussian(positions=positions, sh_coeffs=bad_sh)

        with self.assertRaisesRegex(ValueError, "Gaussian data must be finite"):
            builder.add_deformable_visual_gaussian(
                gaussian,
                kind="tet",
                tet_range=(0, builder.tet_count),
                parent=[0],
                weights=np.full((1, 4), 0.25, dtype=np.float32),
            )

        gaussian = newton.Gaussian(positions=positions)
        with self.assertRaisesRegex(ValueError, "supports only kind='tet'"):
            builder.add_deformable_visual_gaussian(
                gaussian,
                kind="particle",
                tet_range=(0, builder.tet_count),
            )

    def test_viewer_omits_static_copy_of_deformable_gaussian(self):
        """Do not draw an undeformed rest copy through the ordinary shape path."""
        builder = _soft_builder()
        ordinary = newton.Gaussian(positions=np.array([[2.0, 0.0, 0.0]], dtype=np.float32))
        builder.add_shape_gaussian(-1, gaussian=ordinary)

        visual_gaussian = newton.Gaussian(positions=np.array([[0.2, 0.2, 0.2]], dtype=np.float32))
        builder.add_deformable_visual_gaussian(
            visual_gaussian,
            kind="tet",
            tet_range=(0, builder.tet_count),
            parent=[0],
            weights=np.full((1, 4), 0.25, dtype=np.float32),
        )
        model = builder.finalize()
        viewer = _GaussianProbe()
        viewer.show_gaussians = True
        viewer.set_model(model)

        viewer.log_state(model.state())

        visible = [gaussian for _name, gaussian, hidden in viewer.calls if not hidden]
        self.assertEqual(visible, [ordinary])


@unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
class TestDeformableVisualGaussianUSDImport(unittest.TestCase):
    """USD Gaussian graphics embedded in a volume deformable."""

    @staticmethod
    def _stage():
        """Create a meter-scale, Z-up in-memory stage."""
        from pxr import Usd, UsdGeom

        stage = Usd.Stage.CreateInMemory()
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)
        return stage

    @staticmethod
    def _add_volume(stage, path, x=0.0):
        """Add one volume body containing a single tetrahedron."""
        from pxr import UsdGeom

        body = UsdGeom.Xform.Define(stage, path)
        body.GetPrim().AddAppliedSchema("PhysicsDeformableBodyAPI")
        tet = UsdGeom.TetMesh.Define(stage, f"{path}/Sim")
        tet.CreatePointsAttr([(x, 0.0, 0.0), (x + 1.0, 0.0, 0.0), (x, 1.0, 0.0), (x, 0.0, 1.0)])
        tet.CreateTetVertexIndicesAttr([(0, 1, 2, 3)])
        tet.GetPrim().AddAppliedSchema("PhysicsVolumeDeformableSimAPI")
        tet.GetPrim().AddAppliedSchema("PhysicsCollisionAPI")
        return tet

    @staticmethod
    def _add_gaussian(stage, path, positions, *, parent=None, weights=None):
        """Add one Gaussian field with optional authored embedding data."""
        from pxr import Sdf

        gaussian = stage.DefinePrim(path, "ParticleField3DGaussianSplat")
        gaussian.CreateAttribute("positions", Sdf.ValueTypeNames.Point3fArray).Set(positions)
        gaussian.CreateAttribute("scales", Sdf.ValueTypeNames.Float3Array).Set([(0.05, 0.04, 0.03)] * len(positions))
        if parent is not None:
            gaussian.CreateAttribute("newton:deformableSkin:tetIndices", Sdf.ValueTypeNames.IntArray).Set(parent)
        if weights is not None:
            gaussian.CreateAttribute("newton:deformableSkin:influenceWeights", Sdf.ValueTypeNames.Float4Array).Set(
                weights
            )
        return gaussian

    def test_imports_authored_tet_embedding_without_static_duplicate(self):
        """Import authored Gaussian bindings through the public USD path."""
        stage = self._stage()
        self._add_volume(stage, "/World/Bear")
        self._add_gaussian(
            stage,
            "/World/Bear/Gaussian",
            [(0.1, 0.1, 0.1), (0.4, 0.2, 0.1)],
            parent=[0, 0],
            weights=[(0.7, 0.1, 0.1, 0.1), (0.3, 0.4, 0.2, 0.1)],
        )

        builder = newton.ModelBuilder()
        result = builder.add_usd(stage, root_path="/World")
        model = builder.finalize()

        self.assertEqual(model.deformable_visual_gaussian_count, 1)
        visual = model.deformable_visual_gaussians[0]
        self.assertEqual(visual.graphics_path, "/World/Bear/Gaussian")
        self.assertEqual(visual.sim_path, "/World/Bear/Sim")
        np.testing.assert_array_equal(visual.parent.numpy(), [0, 0])
        np.testing.assert_allclose(visual.weights.numpy(), [[0.7, 0.1, 0.1, 0.1], [0.3, 0.4, 0.2, 0.1]])
        self.assertEqual(result["path_shape_map"], {"/World/Bear/Gaussian": visual.shape})

    def test_local_tet_indices_rebase_across_multiple_volume_bodies(self):
        """Rebase each field's local USD tet indices into the shared builder."""
        stage = self._stage()
        for name, x in (("Bear", 0.0), ("Rabbit", 2.0)):
            self._add_volume(stage, f"/World/{name}", x)
            self._add_gaussian(
                stage,
                f"/World/{name}/Gaussian",
                [(x + 0.2, 0.2, 0.2)],
                parent=[0],
                weights=[(0.4, 0.2, 0.2, 0.2)],
            )

        builder = newton.ModelBuilder()
        builder.add_usd(stage, root_path="/World")
        model = builder.finalize()

        self.assertEqual(model.deformable_visual_gaussian_count, 2)
        self.assertEqual([visual.parent.numpy().tolist() for visual in model.deformable_visual_gaussians], [[0], [1]])

    def test_computes_embedding_when_usd_omits_skinning_data(self):
        """Use Gaussian centers to compute a binding when no weights are authored."""
        stage = self._stage()
        self._add_volume(stage, "/World/Bear")
        self._add_gaussian(stage, "/World/Bear/Gaussian", [(0.25, 0.25, 0.25)])

        builder = newton.ModelBuilder()
        builder.add_usd(stage, root_path="/World")
        visual = builder.finalize().deformable_visual_gaussians[0]

        np.testing.assert_array_equal(visual.parent.numpy(), [0])
        np.testing.assert_allclose(visual.weights.numpy(), [[0.25, 0.25, 0.25, 0.25]])

    def test_malformed_embedding_is_skipped_without_static_fallback(self):
        """Do not leave a frozen Gaussian when the deformable binding is malformed."""
        stage = self._stage()
        self._add_volume(stage, "/World/Bear")
        self._add_gaussian(stage, "/World/Bear/Gaussian", [(0.25, 0.25, 0.25)], parent=[0])

        builder = newton.ModelBuilder()
        with self.assertWarnsRegex(UserWarning, "must be authored together"):
            result = builder.add_usd(stage, root_path="/World")

        self.assertEqual(builder.finalize().deformable_visual_gaussian_count, 0)
        self.assertNotIn("/World/Bear/Gaussian", result["path_shape_map"])

    def test_load_visual_shapes_false_skips_gaussian_visual(self):
        """Honor the shared visual-shape loading flag."""
        stage = self._stage()
        self._add_volume(stage, "/World/Bear")
        self._add_gaussian(stage, "/World/Bear/Gaussian", [(0.25, 0.25, 0.25)])

        builder = newton.ModelBuilder()
        result = builder.add_usd(stage, root_path="/World", load_visual_shapes=False)

        self.assertEqual(builder.finalize().deformable_visual_gaussian_count, 0)
        self.assertNotIn("/World/Bear/Gaussian", result["path_shape_map"])

    def test_replicates_imported_gaussian_paths_and_drivers(self):
        """Rebase USD ownership and tet drivers into replicated worlds."""
        stage = self._stage()
        self._add_volume(stage, "/World/envs/env_0/Bear")
        self._add_gaussian(
            stage,
            "/World/envs/env_0/Bear/Gaussian",
            [(0.25, 0.25, 0.25)],
            parent=[0],
            weights=[(0.25, 0.25, 0.25, 0.25)],
        )
        template = newton.ModelBuilder()
        template.add_usd(stage, root_path="/World/envs/env_0")

        builder = newton.ModelBuilder()
        builder.replicate(
            template,
            2,
            source_path_prefix="/World/envs/env_0",
            destination_path_prefixes=["/World/envs/env_0", "/World/envs/env_1"],
        )
        model = builder.finalize()

        self.assertEqual(model.deformable_visual_gaussian_count, 2)
        self.assertEqual([visual.world for visual in model.deformable_visual_gaussians], [0, 1])
        self.assertEqual([visual.parent.numpy().tolist() for visual in model.deformable_visual_gaussians], [[0], [1]])
        self.assertEqual(
            [visual.graphics_path for visual in model.deformable_visual_gaussians],
            ["/World/envs/env_0/Bear/Gaussian", "/World/envs/env_1/Bear/Gaussian"],
        )
        self.assertNotEqual(model.deformable_visual_gaussians[0].shape, model.deformable_visual_gaussians[1].shape)

    def test_nonuniform_usd_transform_preserves_gaussian_covariance(self):
        """Bake an affine prim transform into rotated anisotropic splats exactly."""
        from pxr import Gf, Sdf, UsdGeom

        stage = self._stage()
        self._add_volume(stage, "/World/Bear")
        UsdGeom.Xformable(stage.GetPrimAtPath("/World/Bear")).AddScaleOp().Set((2.0, 1.0, 0.5))
        gaussian = self._add_gaussian(stage, "/World/Bear/Gaussian", [(0.2, 0.2, 0.2)])
        angle = math.radians(45.0)
        local_rotation = np.array([0.0, 0.0, math.sin(0.5 * angle), math.cos(0.5 * angle)], dtype=np.float32)
        gaussian.CreateAttribute("orientations", Sdf.ValueTypeNames.QuatfArray).Set(
            [Gf.Quatf(float(local_rotation[3]), Gf.Vec3f(*[float(value) for value in local_rotation[:3]]))]
        )
        local_scale = np.array([0.12, 0.04, 0.02], dtype=np.float32)
        gaussian.GetAttribute("scales").Set([Gf.Vec3f(*[float(value) for value in local_scale])])

        builder = newton.ModelBuilder()
        builder.add_usd(stage, root_path="/World")
        imported = builder.finalize().deformable_visual_gaussians[0].gaussian

        linear = np.diag([2.0, 1.0, 0.5]).astype(np.float32)
        local_axes = _quat_matrix(local_rotation) @ np.diag(local_scale)
        expected_covariance = linear @ local_axes @ local_axes.T @ linear.T
        imported_axes = _quat_matrix(imported.rotations[0]) @ np.diag(imported.scales[0])
        np.testing.assert_allclose(imported_axes @ imported_axes.T, expected_covariance, rtol=1.0e-5, atol=1.0e-7)


class TestDeformableVisualGaussianEvaluation(unittest.TestCase):
    """Reusable current Gaussian transforms and scales."""

    def test_update_evaluates_tet_center_and_covariance(self):
        """Evaluate one Gaussian center and covariance through the shared visual result."""
        builder = _soft_builder()
        tet_indices = np.asarray(builder.tet_indices, dtype=np.int32).reshape(-1, 4)
        corners = np.asarray(builder.particle_q, dtype=np.float32)[tet_indices[0]]
        center = corners.mean(axis=0, keepdims=True)
        rest_scale = np.array([[0.1, 0.2, 0.3]], dtype=np.float32)
        gaussian = newton.Gaussian(positions=center, scales=rest_scale)
        builder.add_deformable_visual_gaussian(
            gaussian,
            kind="tet",
            tet_range=(0, builder.tet_count),
            parent=[0],
            weights=np.full((1, 4), 0.25, dtype=np.float32),
        )
        model = builder.finalize()
        state = model.state()
        visuals = model.deformable_visuals()

        returned = model.update_deformable_visuals(state, visuals)

        self.assertIs(returned, visuals)
        transforms = visuals.get_gaussian_transforms(0).numpy()
        scales = visuals.get_gaussian_scales(0).numpy()
        np.testing.assert_allclose(transforms[0, :3], center[0], atol=1.0e-6)
        np.testing.assert_allclose(np.sort(scales[0]), np.sort(rest_scale[0]), atol=1.0e-6)
        self.assertIs(visuals.state, state)

    def test_shear_updates_full_covariance(self):
        """Preserve the independently computed Gaussian covariance under affine shear."""
        builder = _soft_builder()
        tet_indices = np.asarray(builder.tet_indices, dtype=np.int32).reshape(-1, 4)
        rest_particles = np.asarray(builder.particle_q, dtype=np.float32)
        center = rest_particles[tet_indices[0]].mean(axis=0, keepdims=True)
        rest_scale = np.array([[0.08, 0.13, 0.21]], dtype=np.float32)
        gaussian = newton.Gaussian(positions=center, scales=rest_scale)
        builder.add_deformable_visual_gaussian(
            gaussian,
            kind="tet",
            tet_range=(0, builder.tet_count),
            parent=[0],
            weights=np.full((1, 4), 0.25, dtype=np.float32),
        )
        model = builder.finalize()
        state = model.state()
        shear = np.array([[1.0, 0.35, 0.0], [0.0, 1.0, 0.2], [0.0, 0.0, 0.7]], dtype=np.float32)
        state.particle_q.assign(rest_particles @ shear.T)
        visuals = model.deformable_visuals()

        model.update_deformable_visuals(state, visuals)

        transform = visuals.get_gaussian_transforms(0).numpy()[0]
        scales = visuals.get_gaussian_scales(0).numpy()[0]
        rotation = _quat_matrix(transform[3:7])
        actual_covariance = rotation @ np.diag(scales**2) @ rotation.T
        expected_covariance = shear @ np.diag(rest_scale[0] ** 2) @ shear.T
        np.testing.assert_allclose(actual_covariance, expected_covariance, atol=2.0e-6)
        np.testing.assert_allclose(transform[:3], (center @ shear.T)[0], atol=1.0e-6)

    def test_update_is_cuda_graph_capturable(self):
        """Replay Gaussian evaluation into the same output buffers during CUDA capture."""
        builder = _soft_builder()
        tet_indices = np.asarray(builder.tet_indices, dtype=np.int32).reshape(-1, 4)
        rest_particles = np.asarray(builder.particle_q, dtype=np.float32)
        center = rest_particles[tet_indices[0]].mean(axis=0, keepdims=True)
        builder.add_deformable_visual_gaussian(
            newton.Gaussian(positions=center, scales=np.full((1, 3), 0.1, dtype=np.float32)),
            kind="tet",
            tet_range=(0, builder.tet_count),
            parent=[0],
            weights=np.full((1, 4), 0.25, dtype=np.float32),
        )
        model = builder.finalize()
        if not model.device.is_cuda:
            self.skipTest("CUDA graph capture requires a CUDA device")
        state = model.state()
        visuals = model.deformable_visuals()
        model.update_deformable_visuals(state, visuals)
        transforms_ptr = visuals.gaussian_transforms.ptr
        scales_ptr = visuals.gaussian_scales.ptr

        with wp.ScopedCapture(model.device) as capture:
            model.update_deformable_visuals(state, visuals)

        moved = rest_particles.copy()
        moved[:, 2] += 0.4
        state.particle_q.assign(moved)
        wp.capture_launch(capture.graph)
        visuals.wait()

        self.assertEqual(visuals.gaussian_transforms.ptr, transforms_ptr)
        self.assertEqual(visuals.gaussian_scales.ptr, scales_ptr)
        np.testing.assert_allclose(visuals.get_gaussian_transforms(0).numpy()[0, :3], center[0] + [0, 0, 0.4])


class TestDeformableVisualGaussianSensor(unittest.TestCase):
    """Camera consumption through the public deformable visual output."""

    def test_tiled_camera_bounds_cover_large_gaussian_field(self):
        """Render samples outside the first lane of a multi-tile Gaussian field."""
        builder = _soft_builder()
        tet_indices = np.asarray(builder.tet_indices, dtype=np.int32).reshape(-1, 4)
        rest_particles = np.asarray(builder.particle_q, dtype=np.float32)
        corners = rest_particles[tet_indices[0]]
        count = 300
        positions = np.repeat(corners[0:1], count, axis=0)
        positions[1] = corners[3]
        weights = np.zeros((count, 4), dtype=np.float32)
        weights[:, 0] = 1.0
        weights[1] = (0.0, 0.0, 0.0, 1.0)
        builder.add_deformable_visual_gaussian(
            newton.Gaussian(
                positions=positions,
                scales=np.full((count, 3), 0.05, dtype=np.float32),
                opacities=np.full(count, 0.95, dtype=np.float32),
            ),
            kind="tet",
            tet_range=(0, builder.tet_count),
            parent=np.zeros(count, dtype=np.int32),
            weights=weights,
        )
        model = builder.finalize()
        state = model.state()
        sensor = SensorTiledCamera(
            model,
            default_render_config=SensorTiledCamera.RenderConfig(
                enable_particles=False,
                enable_simulation_triangles=False,
                gaussians_mode=SensorTiledCamera.GaussianRenderMode.QUALITY,
                max_distance=10.0,
            ),
        )
        width = 32
        height = 32
        camera_rays = sensor.utils.compute_camera_rays_pinhole(width, height, camera_fovs=math.radians(40.0))
        target = corners[3]
        camera_transforms = wp.array(
            [[wp.transformf(wp.vec3f(float(target[0]), float(target[1]), 2.0), wp.quat_identity())]],
            dtype=wp.transformf,
            device=model.device,
        )
        depth_image = sensor.utils.create_depth_image_output(width, height, camera_count=1)

        sensor.update(state, camera_transforms, camera_rays, depth_image=depth_image)

        center_depth = float(depth_image.numpy()[0, 0, height // 2, width // 2])
        self.assertGreater(center_depth, 0.0)

    def test_tiled_camera_tracks_tet_bound_gaussian(self):
        """Render a Gaussian visual without a separate static Gaussian shape."""
        builder = _soft_builder()
        tet_indices = np.asarray(builder.tet_indices, dtype=np.int32).reshape(-1, 4)
        rest_particles = np.asarray(builder.particle_q, dtype=np.float32)
        center = rest_particles[tet_indices[0]].mean(axis=0, keepdims=True)
        gaussian = newton.Gaussian(
            positions=center,
            scales=np.full((1, 3), 0.18, dtype=np.float32),
            opacities=np.array([0.95], dtype=np.float32),
        )
        builder.add_deformable_visual_gaussian(
            gaussian,
            kind="tet",
            tet_range=(0, builder.tet_count),
            parent=[0],
            weights=np.full((1, 4), 0.25, dtype=np.float32),
            label="soft_splat",
        )
        model = builder.finalize()
        state = model.state()

        sensor = SensorTiledCamera(
            model,
            default_render_config=SensorTiledCamera.RenderConfig(
                enable_particles=False,
                enable_simulation_triangles=False,
                gaussians_mode=SensorTiledCamera.GaussianRenderMode.QUALITY,
                max_distance=10.0,
            ),
        )
        width = 32
        height = 32
        camera_rays = sensor.utils.compute_camera_rays_pinhole(width, height, camera_fovs=math.radians(40.0))
        camera_transforms = wp.array(
            [[wp.transformf(wp.vec3f(float(center[0, 0]), float(center[0, 1]), 2.0), wp.quat_identity())]],
            dtype=wp.transformf,
            device=model.device,
        )
        depth_image = sensor.utils.create_depth_image_output(width, height, camera_count=1)

        sensor.update(state, camera_transforms, camera_rays, depth_image=depth_image)
        rest_depth = float(depth_image.numpy()[0, 0, height // 2, width // 2])

        moved = rest_particles.copy()
        moved[:, 2] += 0.4
        state.particle_q.assign(moved)
        sensor.update(state, camera_transforms, camera_rays, depth_image=depth_image)
        moved_depth = float(depth_image.numpy()[0, 0, height // 2, width // 2])

        self.assertGreater(rest_depth, 0.0)
        self.assertGreater(moved_depth, 0.0)
        self.assertAlmostEqual(rest_depth - moved_depth, 0.4, delta=0.1)

    def test_tiled_camera_keeps_replicated_gaussians_independent(self):
        """Render one independently deformed Gaussian field in each world."""
        prototype = _soft_builder()
        tet_indices = np.asarray(prototype.tet_indices, dtype=np.int32).reshape(-1, 4)
        rest_particles = np.asarray(prototype.particle_q, dtype=np.float32)
        center = rest_particles[tet_indices[0]].mean(axis=0, keepdims=True)
        prototype.add_deformable_visual_gaussian(
            newton.Gaussian(
                positions=center,
                scales=np.full((1, 3), 0.18, dtype=np.float32),
                opacities=np.array([0.95], dtype=np.float32),
            ),
            kind="tet",
            tet_range=(0, prototype.tet_count),
            parent=[0],
            weights=np.full((1, 4), 0.25, dtype=np.float32),
        )
        builder = newton.ModelBuilder()
        builder.replicate(prototype, 2)
        model = builder.finalize()
        state = model.state()

        moved = state.particle_q.numpy()
        particle_world = model.particle_world.numpy()
        moved[particle_world == 0, 2] += 0.4
        state.particle_q.assign(moved)

        sensor = SensorTiledCamera(
            model,
            default_render_config=SensorTiledCamera.RenderConfig(
                enable_particles=False,
                enable_simulation_triangles=False,
                gaussians_mode=SensorTiledCamera.GaussianRenderMode.QUALITY,
                max_distance=10.0,
            ),
        )
        width = 32
        height = 32
        camera_rays = sensor.utils.compute_camera_rays_pinhole(width, height, camera_fovs=math.radians(40.0))
        camera = wp.transformf(wp.vec3f(float(center[0, 0]), float(center[0, 1]), 2.0), wp.quat_identity())
        camera_transforms = wp.array([[camera, camera]], dtype=wp.transformf, device=model.device)
        depth_image = sensor.utils.create_depth_image_output(width, height, camera_count=1)

        sensor.update(state, camera_transforms, camera_rays, depth_image=depth_image)
        depth = depth_image.numpy()[:, 0, height // 2, width // 2]

        self.assertTrue(np.all(depth > 0.0))
        self.assertAlmostEqual(float(depth[1] - depth[0]), 0.4, delta=0.1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
