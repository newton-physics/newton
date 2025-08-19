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

import numpy as np

# Default number of segments for mesh generation
default_num_segments = 32


def create_sphere_mesh(
    radius=1.0,
    num_latitudes=default_num_segments,
    num_longitudes=default_num_segments,
    reverse_winding=False,
):
    vertices = []
    indices = []

    for i in range(num_latitudes + 1):
        theta = i * np.pi / num_latitudes
        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)

        for j in range(num_longitudes + 1):
            phi = j * 2 * np.pi / num_longitudes
            sin_phi = np.sin(phi)
            cos_phi = np.cos(phi)

            x = cos_phi * sin_theta
            y = cos_theta
            z = sin_phi * sin_theta

            u = float(j) / num_longitudes
            v = float(i) / num_latitudes

            vertices.append([x * radius, y * radius, z * radius, x, y, z, u, v])

    for i in range(num_latitudes):
        for j in range(num_longitudes):
            first = i * (num_longitudes + 1) + j
            second = first + num_longitudes + 1

            if reverse_winding:
                indices.extend([first, second, first + 1, second, second + 1, first + 1])
            else:
                indices.extend([first, first + 1, second, second, first + 1, second + 1])

    return np.array(vertices, dtype=np.float32), np.array(indices, dtype=np.uint32)


def create_capsule_mesh(radius, half_height, up_axis=1, segments=default_num_segments):
    vertices = []
    indices = []

    x_dir, y_dir, z_dir = ((1, 2, 0), (2, 0, 1), (1, 2, 0))[up_axis]
    up_vector = np.zeros(3)
    up_vector[up_axis] = half_height

    for i in range(segments + 1):
        theta = i * np.pi / segments
        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)

        for j in range(segments + 1):
            phi = j * 2 * np.pi / segments
            sin_phi = np.sin(phi)
            cos_phi = np.cos(phi)

            z = cos_phi * sin_theta
            y = cos_theta
            x = sin_phi * sin_theta

            u = cos_theta * 0.5 + 0.5
            v = cos_phi * sin_theta * 0.5 + 0.5

            xyz = x, y, z
            x, y, z = xyz[x_dir], xyz[y_dir], xyz[z_dir]
            xyz = np.array((x, y, z), dtype=np.float32) * radius
            if j < segments // 2:
                xyz += up_vector
            else:
                xyz -= up_vector

            vertices.append([*xyz, x, y, z, u, v])

    nv = len(vertices)
    for i in range(segments + 1):
        for j in range(segments + 1):
            first = (i * (segments + 1) + j) % nv
            second = (first + segments + 1) % nv
            indices.extend([first, second, (first + 1) % nv, second, (second + 1) % nv, (first + 1) % nv])

    vertex_data = np.array(vertices, dtype=np.float32)
    index_data = np.array(indices, dtype=np.uint32)

    return vertex_data, index_data


def create_cone_mesh(radius, half_height, up_axis=1, segments=default_num_segments):
    # render it as a cylinder with zero top radius so we get correct normals on the sides
    return create_cylinder_mesh(radius, half_height, up_axis, segments, 0.0)


def create_cylinder_mesh(radius, half_height, up_axis=1, segments=default_num_segments, top_radius=None):
    if up_axis not in (0, 1, 2):
        raise ValueError("up_axis must be between 0 and 2")

    x_dir, y_dir, z_dir = (
        (1, 2, 0),
        (0, 1, 2),
        (2, 0, 1),
    )[up_axis]

    indices = []

    cap_vertices = []
    side_vertices = []

    # create center cap vertices
    position = np.array([0, -half_height, 0])[[x_dir, y_dir, z_dir]]
    normal = np.array([0, -1, 0])[[x_dir, y_dir, z_dir]]
    cap_vertices.append([*position, *normal, 0.5, 0.5])
    cap_vertices.append([*-position, *-normal, 0.5, 0.5])

    if top_radius is None:
        top_radius = radius
    side_slope = -np.arctan2(top_radius - radius, 2 * half_height)

    # create the cylinder base and top vertices
    for j in (-1, 1):
        center_index = max(j, 0)
        if j == 1:
            radius = top_radius
        for i in range(segments):
            theta = 2 * np.pi * i / segments

            cos_theta = np.cos(theta)
            sin_theta = np.sin(theta)

            x = cos_theta
            y = j * half_height
            z = sin_theta

            position = np.array([radius * x, y, radius * z])

            normal = np.array([x, side_slope, z])
            normal = normal / np.linalg.norm(normal)
            uv = (i / (segments - 1), (j + 1) / 2)
            vertex = np.hstack([position[[x_dir, y_dir, z_dir]], normal[[x_dir, y_dir, z_dir]], uv])
            side_vertices.append(vertex)

            normal = np.array([0, j, 0])
            uv = (cos_theta * 0.5 + 0.5, sin_theta * 0.5 + 0.5)
            vertex = np.hstack([position[[x_dir, y_dir, z_dir]], normal[[x_dir, y_dir, z_dir]], uv])
            cap_vertices.append(vertex)

            cs = center_index * segments
            indices.extend([center_index, i + cs + 2, (i + 1) % segments + cs + 2][::-j])

    # create the cylinder side indices
    for i in range(segments):
        index1 = len(cap_vertices) + i + segments
        index2 = len(cap_vertices) + ((i + 1) % segments) + segments
        index3 = len(cap_vertices) + i
        index4 = len(cap_vertices) + ((i + 1) % segments)

        indices.extend([index1, index2, index3, index2, index4, index3])

    vertex_data = np.array(np.vstack((cap_vertices, side_vertices)), dtype=np.float32)
    index_data = np.array(indices, dtype=np.uint32)

    return vertex_data, index_data


def create_arrow_mesh(
    base_radius, base_height, cap_radius=None, cap_height=None, up_axis=1, segments=default_num_segments
):
    if up_axis not in (0, 1, 2):
        raise ValueError("up_axis must be between 0 and 2")
    if cap_radius is None:
        cap_radius = base_radius * 1.8
    if cap_height is None:
        cap_height = base_height * 0.18

    up_vector = np.array([0, 0, 0])
    up_vector[up_axis] = 1

    base_vertices, base_indices = create_cylinder_mesh(base_radius, base_height / 2, up_axis, segments)
    cap_vertices, cap_indices = create_cone_mesh(cap_radius, cap_height / 2, up_axis, segments)

    base_vertices[:, :3] += base_height / 2 * up_vector
    # move cap slightly lower to avoid z-fighting
    cap_vertices[:, :3] += (base_height + cap_height / 2 - 1e-3 * base_height) * up_vector

    vertex_data = np.vstack((base_vertices, cap_vertices))
    index_data = np.hstack((base_indices, cap_indices + len(base_vertices)))

    return vertex_data, index_data


def create_box_mesh(extents):
    x_extent, y_extent, z_extent = extents

    vertices = [
        # Position                        Normal    UV
        [-x_extent, -y_extent, -z_extent, -1, 0, 0, 0, 0],
        [-x_extent, -y_extent, z_extent, -1, 0, 0, 1, 0],
        [-x_extent, y_extent, z_extent, -1, 0, 0, 1, 1],
        [-x_extent, y_extent, -z_extent, -1, 0, 0, 0, 1],
        [x_extent, -y_extent, -z_extent, 1, 0, 0, 0, 0],
        [x_extent, -y_extent, z_extent, 1, 0, 0, 1, 0],
        [x_extent, y_extent, z_extent, 1, 0, 0, 1, 1],
        [x_extent, y_extent, -z_extent, 1, 0, 0, 0, 1],
        [-x_extent, -y_extent, -z_extent, 0, -1, 0, 0, 0],
        [-x_extent, -y_extent, z_extent, 0, -1, 0, 1, 0],
        [x_extent, -y_extent, z_extent, 0, -1, 0, 1, 1],
        [x_extent, -y_extent, -z_extent, 0, -1, 0, 0, 1],
        [-x_extent, y_extent, -z_extent, 0, 1, 0, 0, 0],
        [-x_extent, y_extent, z_extent, 0, 1, 0, 1, 0],
        [x_extent, y_extent, z_extent, 0, 1, 0, 1, 1],
        [x_extent, y_extent, -z_extent, 0, 1, 0, 0, 1],
        [-x_extent, -y_extent, -z_extent, 0, 0, -1, 0, 0],
        [-x_extent, y_extent, -z_extent, 0, 0, -1, 1, 0],
        [x_extent, y_extent, -z_extent, 0, 0, -1, 1, 1],
        [x_extent, -y_extent, -z_extent, 0, 0, -1, 0, 1],
        [-x_extent, -y_extent, z_extent, 0, 0, 1, 0, 0],
        [-x_extent, y_extent, z_extent, 0, 0, 1, 1, 0],
        [x_extent, y_extent, z_extent, 0, 0, 1, 1, 1],
        [x_extent, -y_extent, z_extent, 0, 0, 1, 0, 1],
    ]

    # fmt: off
    indices = [
        0, 1, 2,
        0, 2, 3,
        4, 6, 5,
        4, 7, 6,
        8, 10, 9,
        8, 11, 10,
        12, 13, 14,
        12, 14, 15,
        16, 17, 18,
        16, 18, 19,
        20, 22, 21,
        20, 23, 22,
    ]
    # fmt: on
    return np.array(vertices, dtype=np.float32), np.array(indices, dtype=np.uint32)


def create_plane_mesh(width, length):
    return create_box_mesh((width, length, 0.005))
