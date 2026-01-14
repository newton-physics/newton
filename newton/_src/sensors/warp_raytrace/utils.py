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

import warp as wp


@wp.kernel(enable_backward=False)
def compute_mesh_bounds(in_meshes: wp.array(dtype=wp.uint64), out_bounds: wp.array2d(dtype=wp.vec3f)):
    tid = wp.tid()

    min_point = wp.vec3(wp.inf)
    max_point = wp.vec3(-wp.inf)

    if in_meshes[tid] != 0:
        mesh = wp.mesh_get(in_meshes[tid])
        for i in range(mesh.points.shape[0]):
            min_point = wp.min(min_point, mesh.points[i])
            max_point = wp.max(max_point, mesh.points[i])

    out_bounds[tid, 0] = min_point
    out_bounds[tid, 1] = max_point


@wp.kernel(enable_backward=False)
def compute_pinhole_camera_rays(
    width: int,
    height: int,
    camera_fovs: wp.array(dtype=wp.float32),
    out_rays: wp.array(dtype=wp.vec3f, ndim=4),
):
    camera_index, py, px = wp.tid()
    aspect_ratio = float(width) / float(height)
    u = (float(px) + 0.5) / float(width) - 0.5
    v = (float(py) + 0.5) / float(height) - 0.5
    h = wp.tan(camera_fovs[camera_index] / 2.0)
    ray_direction_camera_space = wp.vec3f(u * 2.0 * h * aspect_ratio, -v * 2.0 * h, -1.0)
    out_rays[camera_index, py, px, 0] = wp.vec3f(0.0)
    out_rays[camera_index, py, px, 1] = wp.normalize(ray_direction_camera_space)


@wp.kernel(enable_backward=False)
def flatten_color_image(
    color_image: wp.array(dtype=wp.uint32, ndim=3),
    buffer: wp.array(dtype=wp.uint8, ndim=3),
    width: wp.int32,
    height: wp.int32,
    num_cameras: wp.int32,
    num_worlds_per_row: wp.int32,
):
    world_id, camera_id, y, x = wp.tid()

    view_id = world_id * num_cameras + camera_id

    row = view_id // num_worlds_per_row
    col = view_id % num_worlds_per_row

    px = col * width + x
    py = row * height + y
    color = color_image[world_id, camera_id, y * width + x]

    buffer[py, px, 0] = wp.uint8((color >> wp.uint32(0)) & wp.uint32(0xFF))
    buffer[py, px, 1] = wp.uint8((color >> wp.uint32(8)) & wp.uint32(0xFF))
    buffer[py, px, 2] = wp.uint8((color >> wp.uint32(16)) & wp.uint32(0xFF))
    buffer[py, px, 3] = wp.uint8((color >> wp.uint32(24)) & wp.uint32(0xFF))


@wp.kernel(enable_backward=False)
def flatten_normal_image(
    normal_image: wp.array(dtype=wp.vec3f, ndim=3),
    buffer: wp.array(dtype=wp.uint8, ndim=3),
    width: wp.int32,
    height: wp.int32,
    num_cameras: wp.int32,
    num_worlds_per_row: wp.int32,
):
    world_id, camera_id, y, x = wp.tid()

    view_id = world_id * num_cameras + camera_id

    row = view_id // num_worlds_per_row
    col = view_id % num_worlds_per_row

    px = col * width + x
    py = row * height + y
    normal = normal_image[world_id, camera_id, y * width + x] * 0.5 + wp.vec3f(0.5)

    buffer[py, px, 0] = wp.uint8(normal[0] * 255.0)
    buffer[py, px, 1] = wp.uint8(normal[1] * 255.0)
    buffer[py, px, 2] = wp.uint8(normal[2] * 255.0)
    buffer[py, px, 3] = wp.uint8(255)


@wp.kernel(enable_backward=False)
def find_depth_range(depth_image: wp.array(dtype=wp.float32, ndim=3), depth_range: wp.array(dtype=wp.float32)):
    world_id, camera_id, yx = wp.tid()
    depth = depth_image[world_id, camera_id, yx]
    if depth > 0:
        wp.atomic_min(depth_range, 0, depth)
        wp.atomic_max(depth_range, 1, depth)


@wp.kernel(enable_backward=False)
def flatten_depth_image(
    depth_image: wp.array(dtype=wp.float32, ndim=3),
    buffer: wp.array(dtype=wp.uint8, ndim=3),
    depth_range: wp.array(dtype=wp.float32),
    width: wp.int32,
    height: wp.int32,
    num_cameras: wp.int32,
    num_worlds_per_row: wp.int32,
):
    world_id, camera_id, y, x = wp.tid()

    view_id = world_id * num_cameras + camera_id

    row = view_id // num_worlds_per_row
    col = view_id % num_worlds_per_row

    px = col * width + x
    py = row * height + y

    value = wp.uint8(0)
    depth = depth_image[world_id, camera_id, y * width + x]
    if depth > 0:
        denom = wp.max(depth_range[1] - depth_range[0], 1e-6)
        value = wp.uint8(255.0 - ((depth - depth_range[0]) / denom) * 205.0)

    buffer[py, px, 0] = value
    buffer[py, px, 1] = value
    buffer[py, px, 2] = value
    buffer[py, px, 3] = value
