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

"""Helper functions for computing rigid body inertia properties."""

from __future__ import annotations

import numpy as np
import warp as wp

from .types import (
    GEO_BOX,
    GEO_CAPSULE,
    GEO_CONE,
    GEO_CYLINDER,
    GEO_MESH,
    GEO_PLANE,
    GEO_SDF,
    GEO_SPHERE,
    SDF,
    Mesh,
    Vec3,
)


def compute_sphere_inertia(density: float, r: float) -> tuple[float, wp.vec3, wp.mat33]:
    """Helper to compute mass and inertia of a solid sphere

    Args:
        density: The sphere density
        r: The sphere radius

    Returns:

        A tuple of (mass, inertia) with inertia specified around the origin
    """

    v = 4.0 / 3.0 * wp.pi * r * r * r

    m = density * v
    Ia = 2.0 / 5.0 * m * r * r

    I = wp.mat33([[Ia, 0.0, 0.0], [0.0, Ia, 0.0], [0.0, 0.0, Ia]])

    return (m, wp.vec3(), I)


def compute_capsule_inertia(density: float, r: float, h: float) -> tuple[float, wp.vec3, wp.mat33]:
    """Helper to compute mass and inertia of a solid capsule extending along the y-axis

    Args:
        density: The capsule density
        r: The capsule radius
        h: The capsule height (full height of the interior cylinder)

    Returns:

        A tuple of (mass, inertia) with inertia specified around the origin
    """

    ms = density * (4.0 / 3.0) * wp.pi * r * r * r
    mc = density * wp.pi * r * r * h

    # total mass
    m = ms + mc

    # adapted from ODE
    Ia = mc * (0.25 * r * r + (1.0 / 12.0) * h * h) + ms * (0.4 * r * r + 0.375 * r * h + 0.25 * h * h)
    Ib = (mc * 0.5 + ms * 0.4) * r * r

    I = wp.mat33([[Ia, 0.0, 0.0], [0.0, Ib, 0.0], [0.0, 0.0, Ia]])

    return (m, wp.vec3(), I)


def compute_cylinder_inertia(density: float, r: float, h: float) -> tuple[float, wp.vec3, wp.mat33]:
    """Helper to compute mass and inertia of a solid cylinder extending along the y-axis

    Args:
        density: The cylinder density
        r: The cylinder radius
        h: The cylinder height (extent along the y-axis)

    Returns:

        A tuple of (mass, inertia) with inertia specified around the origin
    """

    m = density * wp.pi * r * r * h

    Ia = 1 / 12 * m * (3 * r * r + h * h)
    Ib = 1 / 2 * m * r * r

    I = wp.mat33([[Ia, 0.0, 0.0], [0.0, Ib, 0.0], [0.0, 0.0, Ia]])

    return (m, wp.vec3(), I)


def compute_cone_inertia(density: float, r: float, h: float) -> tuple[float, wp.vec3, wp.mat33]:
    """Helper to compute mass and inertia of a solid cone extending along the y-axis

    Args:
        density: The cone density
        r: The cone radius
        h: The cone height (extent along the y-axis)

    Returns:

        A tuple of (mass, inertia) with inertia specified around the origin
    """

    m = density * wp.pi * r * r * h / 3.0

    Ia = 1 / 20 * m * (3 * r * r + 2 * h * h)
    Ib = 3 / 10 * m * r * r

    I = wp.mat33([[Ia, 0.0, 0.0], [0.0, Ib, 0.0], [0.0, 0.0, Ia]])

    return (m, wp.vec3(), I)


def compute_box_inertia_from_mass(mass: float, w: float, h: float, d: float) -> wp.mat33:
    """Helper to compute 3x3 inertia matrix of a solid box with given mass
    and dimensions.

    Args:
        mass: The box mass
        w: The box width along the x-axis
        h: The box height along the y-axis
        d: The box depth along the z-axis

    Returns:

        A 3x3 inertia matrix with inertia specified around the origin
    """
    Ia = 1.0 / 12.0 * mass * (h * h + d * d)
    Ib = 1.0 / 12.0 * mass * (w * w + d * d)
    Ic = 1.0 / 12.0 * mass * (w * w + h * h)

    I = wp.mat33([[Ia, 0.0, 0.0], [0.0, Ib, 0.0], [0.0, 0.0, Ic]])

    return I


def compute_box_inertia(density: float, w: float, h: float, d: float) -> tuple[float, wp.vec3, wp.mat33]:
    """Helper to compute mass and inertia of a solid box

    Args:
        density: The box density
        w: The box width along the x-axis
        h: The box height along the y-axis
        d: The box depth along the z-axis

    Returns:

        A tuple of (mass, inertia) with inertia specified around the origin
    """

    v = w * h * d
    m = density * v
    I = compute_box_inertia_from_mass(m, w, h, d)

    return (m, wp.vec3(), I)


@wp.func
def triangle_inertia(
    v0: wp.vec3,
    v1: wp.vec3,
    v2: wp.vec3,
):
    vol = wp.dot(v0, wp.cross(v1, v2)) / 6.0  # tetra volume (0,v0,v1,v2)
    first = vol * (v0 + v1 + v2) / 4.0  # first-order integral

    # second-order integral (symmetric)
    o00, o11, o22 = wp.outer(v0, v0), wp.outer(v1, v1), wp.outer(v2, v2)
    o01, o02, o12 = wp.outer(v0, v1), wp.outer(v0, v2), wp.outer(v1, v2)
    o01t, o02t, o12t = wp.transpose(o01), wp.transpose(o02), wp.transpose(o12)

    second = (vol / 10.0) * (o00 + o11 + o22)
    second += (vol / 20.0) * (o01 + o01t + o02 + o02t + o12 + o12t)

    return vol, first, second


@wp.kernel
def compute_solid_mesh_inertia(
    indices: wp.array(dtype=int),
    vertices: wp.array(dtype=wp.vec3),
    # outputs
    volume: wp.array(dtype=float),
    first: wp.array(dtype=wp.vec3),
    second: wp.array(dtype=wp.mat33),
):
    i = wp.tid()
    p = vertices[indices[i * 3 + 0]]
    q = vertices[indices[i * 3 + 1]]
    r = vertices[indices[i * 3 + 2]]

    v, f, s = triangle_inertia(p, q, r)
    wp.atomic_add(volume, 0, v)
    wp.atomic_add(first, 0, f)
    wp.atomic_add(second, 0, s)


@wp.kernel
def compute_hollow_mesh_inertia(
    indices: wp.array(dtype=int),
    vertices: wp.array(dtype=wp.vec3),
    thickness: wp.array(dtype=float),
    # outputs
    volume: wp.array(dtype=float),
    first: wp.array(dtype=wp.vec3),
    second: wp.array(dtype=wp.mat33),
):
    tid = wp.tid()
    i = indices[tid * 3 + 0]
    j = indices[tid * 3 + 1]
    k = indices[tid * 3 + 2]

    vi = vertices[i]
    vj = vertices[j]
    vk = vertices[k]

    normal = -wp.normalize(wp.cross(vj - vi, vk - vi))
    ti = normal * thickness[i]
    tj = normal * thickness[j]
    tk = normal * thickness[k]

    # wedge vertices
    vi0 = vi - ti
    vi1 = vi + ti
    vj0 = vj - tj
    vj1 = vj + tj
    vk0 = vk - tk
    vk1 = vk + tk

    v_total = 0.0
    f_total = wp.vec3(0.0)
    s_total = wp.mat33(0.0)

    v, f, s = triangle_inertia(vi0, vj0, vk0)
    v_total += v
    f_total += f
    s_total += s
    v, f, s = triangle_inertia(vj0, vk1, vk0)
    v_total += v
    f_total += f
    s_total += s
    v, f, s = triangle_inertia(vj0, vj1, vk1)
    v_total += v
    f_total += f
    s_total += s
    v, f, s = triangle_inertia(vj0, vi1, vj1)
    v_total += v
    f_total += f
    s_total += s
    v, f, s = triangle_inertia(vj0, vi0, vi1)
    v_total += v
    f_total += f
    s_total += s
    v, f, s = triangle_inertia(vj1, vi1, vk1)
    v_total += v
    f_total += f
    s_total += s
    v, f, s = triangle_inertia(vi1, vi0, vk0)
    v_total += v
    f_total += f
    s_total += s
    v, f, s = triangle_inertia(vi1, vk0, vk1)
    v_total += v
    f_total += f
    s_total += s

    wp.atomic_add(volume, 0, v_total)
    wp.atomic_add(first, 0, f_total)
    wp.atomic_add(second, 0, s_total)


def compute_mesh_inertia(
    density: float,
    vertices: list,
    indices: list,
    is_solid: bool = True,
    thickness: list[float] | float = 0.001,
) -> tuple[float, wp.vec3, wp.mat33, float]:
    """
    Compute the mass, center of mass, inertia, and volume of a triangular mesh.

    Args:
        density: The density of the mesh material.
        vertices: A list of vertex positions (3D coordinates).
        indices: A list of triangle indices (each triangle is defined by 3 vertex indices).
        is_solid: If True, compute inertia for a solid mesh; if False, for a hollow mesh using the given thickness.
        thickness: Thickness of the mesh if it is hollow. Can be a single value or a list of values for each vertex.

    Returns:
        A tuple containing:
            - mass: The mass of the mesh.
            - com: The center of mass (3D coordinates).
            - I: The inertia tensor (3x3 matrix).
            - volume: The signed volume of the mesh.
    """

    indices = np.array(indices).flatten()
    num_tris = len(indices) // 3

    # Allocating for mass and inertia
    com_warp = wp.zeros(1, dtype=wp.vec3)
    I_warp = wp.zeros(1, dtype=wp.mat33)
    vol_warp = wp.zeros(1, dtype=float)

    wp_vertices = wp.array(vertices, dtype=wp.vec3)
    wp_indices = wp.array(indices, dtype=int)

    if is_solid:
        wp.launch(
            kernel=compute_solid_mesh_inertia,
            dim=num_tris,
            inputs=[
                wp_indices,
                wp_vertices,
            ],
            outputs=[
                vol_warp,
                com_warp,
                I_warp,
            ],
        )
    else:
        if isinstance(thickness, float):
            thickness = [thickness] * len(vertices)
        wp.launch(
            kernel=compute_hollow_mesh_inertia,
            dim=num_tris,
            inputs=[
                wp_indices,
                wp_vertices,
                wp.array(thickness, dtype=float),
            ],
            outputs=[
                vol_warp,
                com_warp,
                I_warp,
            ],
        )

    V_tot = float(vol_warp.numpy()[0])  # signed volume
    F_tot = com_warp.numpy()[0]  # first moment
    S_tot = I_warp.numpy()[0]  # second moment

    # If the winding is inward, flip signs
    if V_tot < 0:
        V_tot = -V_tot
        F_tot = -F_tot
        S_tot = -S_tot

    mass = density * V_tot
    if V_tot > 0.0:
        com = F_tot / V_tot
    else:
        com = F_tot

    S_tot *= density  # include density
    I_origin = np.trace(S_tot) * np.eye(3) - S_tot  # inertia about origin
    r = com
    I_com = I_origin - mass * ((r @ r) * np.eye(3) - np.outer(r, r))

    return mass, wp.vec3(*com), wp.mat33(*I_com), V_tot


def transform_inertia(m, I, p, q) -> wp.mat33:
    R = wp.quat_to_matrix(q)

    # Steiner's theorem
    return R @ I @ wp.transpose(R) + m * (wp.dot(p, p) * wp.mat33(np.eye(3)) - wp.outer(p, p))


def compute_shape_inertia(
    type: int,
    scale: Vec3,
    src: SDF | Mesh | None,
    density: float,
    is_solid: bool = True,
    thickness: list[float] | float = 0.001,
) -> tuple[float, wp.vec3, wp.mat33]:
    """Computes the mass, center of mass and 3x3 inertia tensor of a shape

    Args:
        type: The type of shape (GEO_SPHERE, GEO_BOX, etc.)
        scale: The scale of the shape
        src: The source shape (Mesh or SDF)
        density: The density of the shape
        is_solid: Whether the shape is solid or hollow
        thickness: The thickness of the shape (used for collision detection, and inertia computation of hollow shapes)

    Returns:
        The mass, center of mass and 3x3 inertia tensor of the shape
    """
    if density == 0.0 or type == GEO_PLANE:  # zero density means fixed
        return 0.0, wp.vec3(), wp.mat33()

    if type == GEO_SPHERE:
        solid = compute_sphere_inertia(density, scale[0])
        if is_solid:
            return solid
        else:
            assert isinstance(thickness, float), "thickness must be a float for a hollow sphere geom"
            hollow = compute_sphere_inertia(density, scale[0] - thickness)
            return solid[0] - hollow[0], solid[1], solid[2] - hollow[2]
    elif type == GEO_BOX:
        w, h, d = scale[0] * 2.0, scale[1] * 2.0, scale[2] * 2.0
        solid = compute_box_inertia(density, w, h, d)
        if is_solid:
            return solid
        else:
            assert isinstance(thickness, float), "thickness must be a float for a hollow box geom"
            hollow = compute_box_inertia(density, w - thickness, h - thickness, d - thickness)
            return solid[0] - hollow[0], solid[1], solid[2] - hollow[2]
    elif type == GEO_CAPSULE:
        r, h = scale[0], scale[1] * 2.0
        solid = compute_capsule_inertia(density, r, h)
        if is_solid:
            return solid
        else:
            assert isinstance(thickness, float), "thickness must be a float for a hollow capsule geom"
            hollow = compute_capsule_inertia(density, r - thickness, h - 2.0 * thickness)
            return solid[0] - hollow[0], solid[1], solid[2] - hollow[2]
    elif type == GEO_CYLINDER:
        r, h = scale[0], scale[1] * 2.0
        solid = compute_cylinder_inertia(density, r, h)
        if is_solid:
            return solid
        else:
            assert isinstance(thickness, float), "thickness must be a float for a hollow cylinder geom"
            hollow = compute_cylinder_inertia(density, r - thickness, h - 2.0 * thickness)
            return solid[0] - hollow[0], solid[1], solid[2] - hollow[2]
    elif type == GEO_CONE:
        r, h = scale[0], scale[1] * 2.0
        solid = compute_cone_inertia(density, r, h)
        if is_solid:
            return solid
        else:
            assert isinstance(thickness, float), "thickness must be a float for a hollow cone geom"
            hollow = compute_cone_inertia(density, r - thickness, h - 2.0 * thickness)
            return solid[0] - hollow[0], solid[1], solid[2] - hollow[2]
    elif type == GEO_MESH or type == GEO_SDF:
        assert src is not None, "src must be provided for mesh or SDF shapes"
        if src.has_inertia and src.mass > 0.0 and src.is_solid == is_solid:
            m, c, I = src.mass, src.com, src.I
            scale = wp.vec3(scale)
            sx, sy, sz = scale

            mass_ratio = sx * sy * sz * density
            m_new = m * mass_ratio

            c_new = wp.cw_mul(c, scale)

            Ixx = I[0, 0] * (sy**2 + sz**2) / 2 * mass_ratio
            Iyy = I[1, 1] * (sx**2 + sz**2) / 2 * mass_ratio
            Izz = I[2, 2] * (sx**2 + sy**2) / 2 * mass_ratio
            Ixy = I[0, 1] * sx * sy * mass_ratio
            Ixz = I[0, 2] * sx * sz * mass_ratio
            Iyz = I[1, 2] * sy * sz * mass_ratio

            I_new = wp.mat33([[Ixx, Ixy, Ixz], [Ixy, Iyy, Iyz], [Ixz, Iyz, Izz]])

            return m_new, c_new, I_new
        elif type == GEO_MESH:
            assert isinstance(src, Mesh), "src must be a Mesh for mesh shapes"
            # fall back to computing inertia from mesh geometry
            vertices = np.array(src.vertices) * np.array(scale)
            m, c, I, _vol = compute_mesh_inertia(density, vertices, src.indices, is_solid, thickness)
            return m, c, I
    raise ValueError(f"Unsupported shape type: {type}")
