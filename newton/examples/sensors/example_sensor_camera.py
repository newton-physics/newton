# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Sensor Camera
#
# Shows how to use the SensorCamera class and display its output
# via Viewer.log_image.
#
# Command: python -m newton.examples sensor_camera
#
###########################################################################

import math
import random

import numpy as np
import warp as wp
from pxr import Usd

import newton
import newton.examples
import newton.usd
from newton.sensors import SensorCamera
from newton.viewer import ViewerGL

SEMANTIC_COLOR_CYLINDER = (255, 0, 0)
SEMANTIC_COLOR_SPHERE = (255, 255, 0)
SEMANTIC_COLOR_CAPSULE = (0, 255, 255)
SEMANTIC_COLOR_BOX = (0, 0, 255)
SEMANTIC_COLOR_MESH = (0, 255, 0)
SEMANTIC_COLOR_ROBOT = (255, 0, 255)
SEMANTIC_COLOR_GAUSSIAN = (255, 153, 0)
SEMANTIC_COLOR_GROUND_PLANE = (68, 68, 68)


# Sweeping every Franka joint across its full URDF range yields poses that
# punch the wrist through the ground plane or fold the arm onto itself.
# Animate each joint as ``home + radius * sin(time + phase)`` instead, where
# ``home`` sits at the standard "ready" pose and ``radius = alpha * min(home -
# lower, upper - home)`` uses a per-joint fraction of the symmetric distance
# to the URDF limits.
_FRANKA_HOME_AND_ALPHA: dict[str, tuple[float, float]] = {
    "fr3_joint1": (0.0, 0.6),
    "fr3_joint2": (-math.pi / 4.0, 0.4),
    "fr3_joint3": (0.0, 0.5),
    "fr3_joint4": (-3.0 * math.pi / 4.0, 0.5),
    "fr3_joint5": (0.0, 0.6),
    "fr3_joint6": (math.pi / 2.0, 0.5),
    "fr3_joint7": (math.pi / 4.0, 0.7),
    "fr3_finger_joint1": (0.02, 1.0),
    "fr3_finger_joint2": (0.02, 1.0),
}


@wp.kernel(enable_backward=False)
def animate_franka(
    time: wp.float32,
    joint_type: wp.array[wp.int32],
    joint_dof_dim: wp.array2d[wp.int32],
    joint_q_start: wp.array[wp.int32],
    joint_qd_start: wp.array[wp.int32],
    dof_home: wp.array[wp.float32],
    dof_radius: wp.array[wp.float32],
    joint_q: wp.array[wp.float32],
):
    tid = wp.tid()

    if joint_type[tid] == newton.JointType.FREE:
        return

    rng = wp.rand_init(1234, tid)
    num_linear_dofs = joint_dof_dim[tid, 0]
    num_angular_dofs = joint_dof_dim[tid, 1]
    q_start = joint_q_start[tid]
    qd_start = joint_qd_start[tid]
    for i in range(num_linear_dofs + num_angular_dofs):
        joint_q[q_start + i] = dof_home[qd_start + i] + dof_radius[qd_start + i] * wp.sin(time + wp.randf(rng))


@wp.kernel(enable_backward=False)
def fill_world_camera_transforms(
    camera_transform: wp.transformf,
    out_transforms: wp.array[wp.transformf],
):
    # World-fixed camera: the same world-space pose renders every world.
    out_transforms[wp.tid()] = camera_transform


@wp.kernel(enable_backward=False)
def fill_body_camera_transforms(
    body_indices: wp.array[wp.int32],
    local_transform: wp.transformf,
    body_q: wp.array[wp.transform],
    out_transforms: wp.array[wp.transformf],
):
    # Body-mounted camera: compose each world's body pose with the local offset.
    tid = wp.tid()
    out_transforms[tid] = wp.transform_multiply(body_q[body_indices[tid]], local_transform)


class Example:
    def __init__(self, viewer: ViewerGL, args):
        self.worlds_per_row = 6
        self.worlds_per_col = 4
        self.world_count_total = self.worlds_per_row * self.worlds_per_col

        self.time = 0.0
        self.time_delta = 0.005

        self.viewer = viewer
        self.show_robot_camera = False
        self.sensor_color_as_main_view = False
        self.disable_clear_worlds = False
        self.disable_preserve_worlds = False

        self.sensor_render_width = 256
        self.sensor_render_height = 256

        fov = 45.0
        if isinstance(self.viewer, ViewerGL):
            fov = self.viewer.camera.fov
        # The camera rays only depend on the image model; the SensorCameras are
        # constructed once the render context exists (see below).
        self.observer_camera_fov = math.radians(fov)
        self.robot_camera_fov = math.radians(75.0)

        usd_stage = Usd.Stage.Open(newton.examples.get_asset("bunny.usd"))
        bunny_mesh = newton.usd.get_mesh(usd_stage.GetPrimAtPath("/root/bunny"))

        robot_asset = newton.utils.download_asset("franka_emika_panda") / "urdf/fr3_franka_hand.urdf"
        robot_builder = newton.ModelBuilder()
        robot_builder.add_urdf(robot_asset, floating=False)
        robot_camera_body = robot_builder.body_label.index("fr3/fr3_link6")

        gaussian = None
        if args.ply:
            gaussian = newton.Gaussian.create_from_ply(args.ply, args.min_response)

        builder = newton.ModelBuilder()

        semantic_colors = []
        robot_shape_indices: list[int] = []
        robot_camera_body_indices: list[int] = []

        rng = random.Random(1234)
        for _world_index in range(self.world_count_total):
            builder.begin_world()
            if rng.random() < 0.5:
                builder.add_shape_cylinder(
                    builder.add_body(xform=wp.transform(p=wp.vec3(0.0, -4.0, 0.5), q=wp.quat_identity())),
                    radius=0.4,
                    half_height=0.5,
                    color=(0.27, 0.47, 0.67),
                )
                semantic_colors.append(SEMANTIC_COLOR_CYLINDER)
            if rng.random() < 0.5:
                builder.add_shape_sphere(
                    builder.add_body(xform=wp.transform(p=wp.vec3(-2.0, -2.0, 0.5), q=wp.quat_identity())),
                    radius=0.5,
                    color=(0.40, 0.80, 0.93),
                )
                semantic_colors.append(SEMANTIC_COLOR_SPHERE)
            if rng.random() < 0.5:
                builder.add_shape_capsule(
                    builder.add_body(xform=wp.transform(p=wp.vec3(-4.0, 0.0, 0.75), q=wp.quat_identity())),
                    radius=0.25,
                    half_height=0.5,
                    color=(0.13, 0.53, 0.20),
                )
                semantic_colors.append(SEMANTIC_COLOR_CAPSULE)
            if rng.random() < 0.5:
                builder.add_shape_box(
                    builder.add_body(xform=wp.transform(p=wp.vec3(-2.0, 2.0, 0.5), q=wp.quat_identity())),
                    hx=0.5,
                    hy=0.35,
                    hz=0.5,
                    color=(0.80, 0.73, 0.27),
                )
                semantic_colors.append(SEMANTIC_COLOR_BOX)
            if rng.random() < 0.5:
                builder.add_shape_mesh(
                    builder.add_body(xform=wp.transform(p=wp.vec3(0.0, 4.0, 0.0), q=wp.quat(0.5, 0.5, 0.5, 0.5))),
                    mesh=bunny_mesh,
                    scale=(0.5, 0.5, 0.5),
                    color=(0.93, 0.40, 0.47),
                )
                semantic_colors.append(SEMANTIC_COLOR_MESH)

            if gaussian is not None:
                builder.add_shape_gaussian(
                    body=builder.add_body(xform=wp.transform(p=wp.vec3(0.0, 0.0, 0.4), q=wp.quat_identity())),
                    gaussian=gaussian,
                )
                semantic_colors.append(SEMANTIC_COLOR_GAUSSIAN)

            robot_shape_start = builder.shape_count
            robot_body_start = builder.body_count
            builder.add_builder(robot_builder, xform=wp.transform(p=wp.vec3(2.0, 0.0, 0.0), q=wp.quat_identity()))
            robot_shape_indices.extend(range(robot_shape_start, robot_shape_start + robot_builder.shape_count))
            semantic_colors.extend([SEMANTIC_COLOR_ROBOT] * robot_builder.shape_count)
            # The observer camera is world-fixed; the robot camera is mounted on
            # link6, so record its body index to compose its pose each frame.
            robot_camera_body_indices.append(robot_body_start + robot_camera_body)
            builder.end_world()

        ground_shape_index = builder.add_ground_plane(color=(0.6, 0.6, 0.6))
        semantic_colors.append(SEMANTIC_COLOR_GROUND_PLANE)

        self.model = builder.finalize()
        self.state = self.model.state()
        self.robot_shape_indices = np.asarray(robot_shape_indices, dtype=np.uint32)
        self.ground_shape_indices = np.asarray([ground_shape_index], dtype=np.uint32)
        self.robot_camera_body_indices = wp.array(robot_camera_body_indices, dtype=wp.int32, device=self.model.device)
        self.disable_clear_world_indices = np.arange(self.worlds_per_row, dtype=np.int32)
        self.disable_preserve_world_indices = np.arange(self.worlds_per_row, 2 * self.worlds_per_row, dtype=np.int32)
        # Identity mapping: view i renders world i. Disabled worlds become
        # negative WorldRenderFlag sentinels (see _update_world_indices).
        self.world_indices_np = np.arange(self.world_count_total, dtype=np.int32)
        self.world_indices = wp.array(self.world_indices_np, dtype=wp.int32, device=self.model.device)
        self._world_indices_dirty = False

        # Build per-DOF home pose and oscillation radius for animate_franka.
        # Joints not listed in _FRANKA_HOME_AND_ALPHA keep radius=0 and stay put.
        joint_qd_start = self.model.joint_qd_start.numpy()
        joint_limit_lower = self.model.joint_limit_lower.numpy()
        joint_limit_upper = self.model.joint_limit_upper.numpy()
        dof_home = np.zeros(self.model.joint_dof_count, dtype=np.float32)
        dof_radius = np.zeros(self.model.joint_dof_count, dtype=np.float32)
        for j_idx, label in enumerate(self.model.joint_label):
            # URDF parser produces hierarchical labels like "fr3/fr3_joint1".
            params = _FRANKA_HOME_AND_ALPHA.get(label.rsplit("/", 1)[-1])
            if params is None:
                continue
            home_val, alpha_val = params
            qd0 = int(joint_qd_start[j_idx])
            lower = float(joint_limit_lower[qd0])
            upper = float(joint_limit_upper[qd0])
            dof_home[qd0] = home_val
            dof_radius[qd0] = max(0.0, alpha_val * min(home_val - lower, upper - home_val))
        self.dof_home = wp.array(dof_home, dtype=wp.float32, device=self.model.device)
        self.dof_radius = wp.array(dof_radius, dtype=wp.float32, device=self.model.device)

        self.viewer.set_model(self.model)

        self.render_context = newton.RenderContext(self.model)
        self.render_context.create_default_light(enable_shadows=True)
        self.render_context.assign_checkerboard_material(shape_indices=self.ground_shape_indices)

        # Construct the SensorCameras with the render context (read-only once set)
        # and configure their render settings.
        self.sensor_camera = SensorCamera(
            SensorCamera.compute_camera_rays_pinhole(
                self.sensor_render_width, self.sensor_render_height, camera_fov=self.observer_camera_fov
            ),
            self.render_context,
        )
        self.robot_sensor_camera = SensorCamera(
            SensorCamera.compute_camera_rays_pinhole(
                self.sensor_render_width, self.sensor_render_height, camera_fov=self.robot_camera_fov
            ),
            self.render_context,
        )
        for sensor_camera in (self.sensor_camera, self.robot_sensor_camera):
            sensor_camera.render_config.enable_shadows = True
            sensor_camera.render_config.enable_textures = True
            sensor_camera.clear_data = newton.ClearData(clear_color=0xFF666666, clear_albedo=0xFF000000)

        # The caller owns the per-view world-space camera transforms passed to
        # SensorCamera.update(); fill them from the animated poses each frame.
        self.sensor_camera_transforms = wp.empty(
            self.sensor_camera.view_count, dtype=wp.transformf, device=self.model.device
        )
        self.robot_sensor_camera_transforms = wp.empty(
            self.robot_sensor_camera.view_count, dtype=wp.transformf, device=self.model.device
        )

        self.sensor_camera_color_image = self.sensor_camera.create_color_image_output()
        self.sensor_camera_albedo_image = self.sensor_camera.create_albedo_image_output()
        self.sensor_camera_depth_image = self.sensor_camera.create_depth_image_output()
        self.sensor_camera_normal_image = self.sensor_camera.create_normal_image_output()
        self.sensor_camera_shape_index_image = self.sensor_camera.create_shape_index_image_output()

        # Palette for the "semantic" debug view: looked up by shape index.
        # Indices written into shape_index_image come from builder shape order,
        # so the palette must be one entry per shape in that same order.
        assert len(semantic_colors) == self.model.shape_count, (
            f"semantic_colors out of sync with model: {len(semantic_colors)} vs {self.model.shape_count}"
        )
        self.semantic_palette = wp.array(
            np.asarray(semantic_colors, dtype=np.uint8),
            dtype=wp.uint8,
            device=self.sensor_camera_color_image.device,
        )

        device = self.sensor_camera_color_image.device
        n = self.world_count_total
        H = self.sensor_camera.height
        W = self.sensor_camera.width
        self.depth_rgba = wp.empty((n, H, W, 4), dtype=wp.uint8, device=device)
        self.normal_rgba = wp.empty((n, H, W, 4), dtype=wp.uint8, device=device)
        self.shape_rgba = wp.empty((n, H, W, 4), dtype=wp.uint8, device=device)
        self.semantic_rgba = wp.empty((n, H, W, 4), dtype=wp.uint8, device=device)
        self.color_main_rgba = wp.empty(
            (self.worlds_per_col * H, self.worlds_per_row * W, 4), dtype=wp.uint8, device=device
        )

    def step(self):
        wp.launch(
            animate_franka,
            self.model.joint_count,
            [
                self.time,
                self.model.joint_type,
                self.model.joint_dof_dim,
                self.model.joint_q_start,
                self.model.joint_qd_start,
                self.dof_home,
                self.dof_radius,
            ],
            outputs=[self.state.joint_q],
        )
        newton.eval_fk(self.model, self.state.joint_q, self.state.joint_qd, self.state)
        self.time += self.time_delta

    def render(self):
        sensor_image_is_main_view = self.render_sensors()

        self.viewer.begin_frame(0.0)
        if not sensor_image_is_main_view:
            self.viewer.log_state(self.state)
        self.viewer.end_frame()

    def render_sensors(self) -> bool:
        sensor_camera = self.robot_sensor_camera if self.show_robot_camera else self.sensor_camera
        camera_transforms = self._update_camera_transforms(sensor_camera)
        self.model.bvh_refit_shapes(self.state)
        self.model.bvh_refit_particles(self.state)
        self.render_context.update(self.state)
        self._update_world_indices()
        sensor_camera.update(
            self.state,
            camera_transforms,
            color_image=self.sensor_camera_color_image,
            albedo_image=self.sensor_camera_albedo_image,
            depth_image=self.sensor_camera_depth_image,
            normal_image=self.sensor_camera_normal_image,
            shape_index_image=self.sensor_camera_shape_index_image,
            world_indices=self.world_indices,
        )
        utils = sensor_camera.utils
        color_rgba = utils.to_rgba_from_color(self.sensor_camera_color_image)
        albedo_rgba = utils.to_rgba_from_color(self.sensor_camera_albedo_image)
        utils.to_rgba_from_depth(self.sensor_camera_depth_image, depth_range=(0.0, 10.0), out_buffer=self.depth_rgba)
        utils.to_rgba_from_normal(self.sensor_camera_normal_image, out_buffer=self.normal_rgba)
        utils.to_rgba_from_shape_index(self.sensor_camera_shape_index_image, out_buffer=self.shape_rgba)
        utils.to_rgba_from_shape_index(
            self.sensor_camera_shape_index_image, colors=self.semantic_palette, out_buffer=self.semantic_rgba
        )

        sensor_image_is_main_view = self.sensor_color_as_main_view and isinstance(self.viewer, ViewerGL)
        self.viewer.log_image("color", color_rgba)
        if sensor_image_is_main_view:
            # Flatten the per-world color views into one full-window image.
            color_main_rgba = utils.flatten_color_image_to_rgba(
                self.sensor_camera_color_image,
                out_buffer=self.color_main_rgba,
                worlds_per_row=self.worlds_per_row,
            )
            self.viewer.log_image("color", color_main_rgba, fullscreen=True)

        self.viewer.log_image("albedo", albedo_rgba)
        self.viewer.log_image("depth", self.depth_rgba)
        self.viewer.log_image("normal", self.normal_rgba)
        self.viewer.log_image("shape_index", self.shape_rgba)
        self.viewer.log_image("semantic", self.semantic_rgba)
        return sensor_image_is_main_view

    def _update_camera_transforms(self, sensor_camera):
        if sensor_camera is self.robot_sensor_camera:
            camera_transforms = self.robot_sensor_camera_transforms
            wp.launch(
                fill_body_camera_transforms,
                dim=self.world_count_total,
                inputs=[
                    self.robot_camera_body_indices,
                    self._get_robot_camera_transform(),
                    self.state.body_q,
                    camera_transforms,
                ],
                device=self.model.device,
            )
        else:
            camera_transforms = self.sensor_camera_transforms
            wp.launch(
                fill_world_camera_transforms,
                dim=self.world_count_total,
                inputs=[self._get_camera_transform(), camera_transforms],
                device=self.model.device,
            )
        return camera_transforms

    def _update_world_indices(self):
        if not self._world_indices_dirty:
            return

        # Reset to identity (view i -> world i), then disable selected worlds.
        self.world_indices_np[:] = np.arange(self.world_count_total, dtype=np.int32)
        if self.disable_clear_worlds:
            self.world_indices_np[self.disable_clear_world_indices] = int(newton.WorldRenderFlag.DISABLE_CLEAR)
        if self.disable_preserve_worlds:
            self.world_indices_np[self.disable_preserve_world_indices] = int(newton.WorldRenderFlag.DISABLE_PRESERVE)

        self.world_indices.assign(self.world_indices_np)
        self._world_indices_dirty = False

    def _get_camera_transform(self) -> wp.transformf:
        if isinstance(self.viewer, ViewerGL):
            return wp.transformf(
                self.viewer.camera.pos,
                wp.quat_from_matrix(wp.mat33f(self.viewer.camera.get_view_matrix().reshape(4, 4)[:3, :3])),
            )
        return wp.transformf(wp.vec3f(10.0, 0.0, 2.0), wp.quatf(0.5, 0.5, 0.5, 0.5))

    @staticmethod
    def _get_robot_camera_transform() -> wp.transformf:
        return wp.transformf(
            wp.vec3f(0.0, 0.06, 0.0),
            wp.quat_from_axis_angle(wp.vec3f(0.0, 1.0, 0.0), math.pi),
        )

    def test_final(self):
        """Verify sensor camera outputs and the sensor-color main-view fallback."""
        sensor_image_is_main_view = self.render_sensors()
        expected_main_view = self.sensor_color_as_main_view and isinstance(self.viewer, ViewerGL)
        assert sensor_image_is_main_view is expected_main_view

        if not isinstance(self.viewer, ViewerGL):
            self.sensor_color_as_main_view = True
            assert self.render_sensors() is False

        expected_shape = (24, self.sensor_camera.height, self.sensor_camera.width)

        color_image = self.sensor_camera_color_image.numpy()
        assert color_image.shape == expected_shape
        assert color_image.min() < color_image.max()

        depth_image = self.sensor_camera_depth_image.numpy()
        assert depth_image.shape == expected_shape
        assert depth_image.min() < depth_image.max()

        # Loose allocation-regression checks on the other outputs: just
        # verify the sensor wrote into arrays with the right shapes/dtypes.
        albedo_image = self.sensor_camera_albedo_image.numpy()
        assert albedo_image.shape == expected_shape
        assert albedo_image.dtype == np.uint32

        normal_image = self.sensor_camera_normal_image.numpy()
        assert normal_image.shape == (24, self.sensor_camera.height, self.sensor_camera.width, 3)
        assert normal_image.dtype == np.float32

        shape_index_image = self.sensor_camera_shape_index_image.numpy()
        assert shape_index_image.shape == expected_shape
        assert shape_index_image.dtype == np.uint32

        albedo_rgba = albedo_image.view(np.uint8).reshape(
            self.world_count_total, self.sensor_camera.height, self.sensor_camera.width, 4
        )
        ground_shape_mask = np.isin(shape_index_image.reshape(albedo_rgba.shape[:3]), self.ground_shape_indices)
        ground_albedo = albedo_rgba[..., :3][ground_shape_mask]
        assert ground_albedo.size > 0
        assert np.unique(ground_albedo, axis=0).shape[0] > 1

        robot_shape_mask = np.isin(shape_index_image.reshape(albedo_rgba.shape[:3]), self.robot_shape_indices)
        robot_albedo = albedo_rgba[..., :3][robot_shape_mask]
        assert robot_albedo.size > 0
        checker_swatches = np.array([[128, 128, 128], [191, 191, 191]], dtype=np.uint8)
        checker_swatch_mask = (robot_albedo[:, None, :] == checker_swatches[None, :, :]).all(axis=2).any(axis=1)
        assert not checker_swatch_mask.any()

        prev_depth_image = depth_image.copy()
        prev_shape_index_image = shape_index_image.copy()
        self.disable_clear_worlds = True
        self.disable_preserve_worlds = True
        self._world_indices_dirty = True
        self.render_sensors()

        world_indices = self.world_indices.numpy()
        assert np.all(world_indices[self.disable_clear_world_indices] == int(newton.WorldRenderFlag.DISABLE_CLEAR))
        assert np.all(
            world_indices[self.disable_preserve_world_indices] == int(newton.WorldRenderFlag.DISABLE_PRESERVE)
        )

        depth_image = self.sensor_camera_depth_image.numpy()
        shape_index_image = self.sensor_camera_shape_index_image.numpy()
        assert np.all(depth_image[self.disable_clear_world_indices] == 0.0)
        assert np.all(shape_index_image[self.disable_clear_world_indices] == np.iinfo(np.uint32).max)
        np.testing.assert_array_equal(
            depth_image[self.disable_preserve_world_indices], prev_depth_image[self.disable_preserve_world_indices]
        )
        np.testing.assert_array_equal(
            shape_index_image[self.disable_preserve_world_indices],
            prev_shape_index_image[self.disable_preserve_world_indices],
        )

        self.disable_clear_worlds = False
        self.disable_preserve_worlds = False
        self._world_indices_dirty = True

        self.show_robot_camera = True
        self.render_sensors()
        robot_camera_depth = self.sensor_camera_depth_image.numpy()
        assert robot_camera_depth.shape == expected_shape
        assert robot_camera_depth.min() < robot_camera_depth.max()
        self.show_robot_camera = False

    def gui(self, ui):
        show_compile_kernel_info = False
        if ui.radio_button("Observer Camera", not self.show_robot_camera):
            self.show_robot_camera = False
        if ui.radio_button("Robot Camera", self.show_robot_camera):
            self.show_robot_camera = True

        if isinstance(self.viewer, ViewerGL):
            _changed, self.sensor_color_as_main_view = ui.checkbox(
                "Sensor Color as Main View", self.sensor_color_as_main_view
            )

        changed, self.disable_clear_worlds = ui.checkbox("Disable First Row: Clear", self.disable_clear_worlds)
        if changed:
            self._world_indices_dirty = True
        changed, self.disable_preserve_worlds = ui.checkbox(
            "Disable Second Row: Preserve", self.disable_preserve_worlds
        )
        if changed:
            self._world_indices_dirty = True

        render_config = (self.robot_sensor_camera if self.show_robot_camera else self.sensor_camera).render_config

        if ui.radio_button(
            "Gaussians: Fast",
            render_config.gaussians_mode == newton.GaussianRenderMode.FAST,
        ):
            if render_config.gaussians_mode != newton.GaussianRenderMode.FAST:
                render_config.gaussians_mode = newton.GaussianRenderMode.FAST
                show_compile_kernel_info = True

        if ui.radio_button(
            "Gaussians: Quality",
            render_config.gaussians_mode == newton.GaussianRenderMode.QUALITY,
        ):
            if render_config.gaussians_mode != newton.GaussianRenderMode.QUALITY:
                render_config.gaussians_mode = newton.GaussianRenderMode.QUALITY
                show_compile_kernel_info = True

        changed, value = ui.slider_float(
            "Min Transmittance",
            render_config.gaussians_min_transmittance,
            0.0,
            1.0,
            "%.2f",
        )
        if changed:
            render_config.gaussians_min_transmittance = value
            show_compile_kernel_info = True

        changed, value = ui.slider_int(
            "Max Num Hits",
            render_config.gaussians_max_num_hits,
            1,
            40,
            "%d",
        )
        if changed:
            render_config.gaussians_max_num_hits = value
            show_compile_kernel_info = True

        if show_compile_kernel_info:
            display_width = self.viewer.renderer.window.width
            display_height = self.viewer.renderer.window.height

            overlay_width = 200
            overlay_height = 100

            text_width, text_height = ui.calc_text_size("Rebuilding Kernels")

            ui.set_next_window_pos(
                ui.ImVec2((display_width - overlay_width) * 0.5, (display_height - overlay_height) * 0.5)
            )
            ui.set_next_window_size(ui.ImVec2(overlay_width, overlay_height))

            if ui.begin(
                "Message",
                flags=(
                    ui.WindowFlags_.no_title_bar.value
                    | ui.WindowFlags_.no_mouse_inputs.value
                    | ui.WindowFlags_.no_scrollbar.value
                ),
            ):
                ui.set_cursor_pos(ui.ImVec2((overlay_width - text_width) * 0.5, (overlay_height - text_height) * 0.5))
                ui.text("Rebuilding Kernels")
            ui.end()

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.add_argument(
            "--ply",
            help="Gaussian filename.",
        )
        parser.add_argument(
            "-min",
            "--min-response",
            type=float,
            default=0.1,
            help="Gaussian min response.",
        )
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()

    # Parse arguments and initialize viewer
    viewer, args = newton.examples.init(parser)

    # Create viewer and run
    newton.examples.run(Example(viewer, args), args)
