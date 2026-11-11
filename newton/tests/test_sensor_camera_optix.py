# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Functional parity checks for the optional OptiX camera backend."""

import importlib.util
import math
import unittest

import numpy as np
import warp as wp

import newton
from newton.sensors import SensorCamera


@unittest.skipUnless(
    importlib.util.find_spec("warp_optix") is not None and wp.is_cuda_available(),
    "requires warp-optix and a CUDA device",
)
class TestSensorCameraOptix(unittest.TestCase):
    def test_geometry_worlds_and_motion(self):
        """Match camera outputs across worlds and after moving primitive and mesh instances."""
        world = newton.ModelBuilder()
        for i in range(4):
            body = world.add_body(xform=wp.transform(wp.vec3((i - 1.5) * 0.6, 0.0, -3.0), wp.quat_identity()))
            if i == 0:
                world.add_shape_sphere(body, radius=0.25)
            elif i == 1:
                world.add_shape_box(body, hx=0.23, hy=0.2, hz=0.2)
            elif i == 2:
                world.add_shape_capsule(body, radius=0.18, half_height=0.22)
            else:
                world.add_shape_mesh(body, mesh=newton.Mesh.create_box(0.2, 0.2, 0.2, compute_inertia=False))
        builder = newton.ModelBuilder()
        builder.replicate(world, 2)
        builder.add_shape_plane(-1, xform=wp.transform(wp.vec3(0.0, 0.0, -4.0), wp.quat_identity()))
        model = builder.finalize(device="cuda:0")
        state = model.state()
        cameras = [SensorCamera(model, backend=backend) for backend in SensorCamera.Backend]
        for camera in cameras:
            camera.create_default_light(enable_shadows=True)
            camera.default_render_config.enable_shadows = True
        rays = SensorCamera.compute_camera_rays_pinhole(32, 24, camera_fov=math.radians(55), device=model.device)
        poses = wp.array([wp.transform_identity()] * 2, dtype=wp.transform, device=model.device)
        # Reverse the view-to-world map to catch accidental dependence on view order.
        worlds = wp.array([1, 0], dtype=wp.int32, device=model.device)
        outputs = [
            {
                "depth_image": camera.allocate_image_depth(2, 32, 24),
                "color_image": camera.allocate_image_color(2, 32, 24),
                "shape_index_image": camera.allocate_image(2, 32, 24, wp.uint32),
                "normal_image": camera.allocate_image(2, 32, 24, wp.vec3),
            }
            for camera in cameras
        ]
        for frame in range(2):
            if frame:
                transforms = state.body_q.numpy()
                transforms[0, 0] += 0.3
                transforms[-1, 1] += 0.25
                state.body_q.assign(transforms)
                model.bvh_refit_shapes(state)
            for camera, output in zip(cameras, outputs, strict=True):
                camera.update(state, poses, rays, world_indices=worlds, sync_shapes=bool(frame), **output)
            for name in ("depth_image", "normal_image"):
                np.testing.assert_allclose(outputs[0][name].numpy(), outputs[1][name].numpy(), atol=3e-4, rtol=1e-4)
            np.testing.assert_array_equal(
                outputs[0]["shape_index_image"].numpy(), outputs[1]["shape_index_image"].numpy()
            )
            color = [output["color_image"].numpy().view(np.uint8).astype(np.int16) for output in outputs]
            self.assertLessEqual(np.abs(color[0] - color[1]).max(), 2)
        # Compile the depth-only path separately from the shaded path.
        for camera, output in zip(cameras, outputs, strict=True):
            camera.update(
                state, poses, rays, world_indices=worlds, depth_image=output["depth_image"], sync_shapes=False
            )
        np.testing.assert_allclose(
            outputs[0]["depth_image"].numpy(), outputs[1]["depth_image"].numpy(), atol=3e-4, rtol=1e-4
        )

    def test_reject_particles_and_deformable_surface(self):
        """Reject unsupported geometry before attempting to compile an OptiX pipeline."""
        builder = newton.ModelBuilder()
        builder.add_cloth_mesh(
            pos=wp.vec3(0.0),
            rot=wp.quat_identity(),
            scale=1.0,
            vel=wp.vec3(0.0),
            vertices=[wp.vec3(-0.8, -0.5, -3.0), wp.vec3(0.8, -0.5, -3.0), wp.vec3(0.0, 0.8, -3.0)],
            indices=[0, 1, 2],
            density=1.0,
        )
        builder.add_particle(pos=wp.vec3(0.5, 0.5, -2.0), vel=wp.vec3(0.0), mass=1.0, radius=0.2)
        model = builder.finalize(device="cuda:0")
        state = model.state()
        rays = SensorCamera.compute_camera_rays_pinhole(16, 16, camera_fov=1.0, device=model.device)
        poses = wp.array([wp.transform_identity()], dtype=wp.transform, device=model.device)
        camera = SensorCamera(model, backend=SensorCamera.Backend.OPTIX)
        depth = camera.allocate_image_depth(1, 16, 16)
        with self.assertRaisesRegex(NotImplementedError, "particles or deformable surfaces"):
            camera.update(state, poses, rays, depth_image=depth)

    def test_empty_scene_and_capture(self):
        """Render an empty model and reject CUDA capture before submitting OptiX work."""
        model = newton.ModelBuilder().finalize(device="cuda:0")
        camera = SensorCamera(model, backend=SensorCamera.Backend.OPTIX)
        rays = SensorCamera.compute_camera_rays_pinhole(2, 2, camera_fov=1.0, device=model.device)
        poses = wp.array([wp.transform_identity()], dtype=wp.transform, device=model.device)
        depth = camera.allocate_image_depth(1, 2, 2)
        camera.update(model.state(), poses, rays, depth_image=depth)
        with wp.ScopedCapture(device=model.device):
            with self.assertRaisesRegex(RuntimeError, "CUDA graph capture"):
                camera.update(model.state(), poses, rays, depth_image=depth)


if __name__ == "__main__":
    unittest.main()
