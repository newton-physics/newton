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

import warnings

import numpy as np
import warp as wp

from .types import (
    SDF,
    GeoType,
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
    """Helper to compute mass and inertia of a solid capsule extending along the z-axis

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

    # For Z-axis orientation: I_xx = I_yy = Ia, I_zz = Ib
    I = wp.mat33([[Ia, 0.0, 0.0], [0.0, Ia, 0.0], [0.0, 0.0, Ib]])

    return (m, wp.vec3(), I)


def compute_cylinder_inertia(density: float, r: float, h: float) -> tuple[float, wp.vec3, wp.mat33]:
    """Helper to compute mass and inertia of a solid cylinder extending along the z-axis

    Args:
        density: The cylinder density
        r: The cylinder radius
        h: The cylinder height (extent along the z-axis)

    Returns:

        A tuple of (mass, inertia) with inertia specified around the origin
    """

    m = density * wp.pi * r * r * h

    Ia = 1 / 12 * m * (3 * r * r + h * h)
    Ib = 1 / 2 * m * r * r

    # For Z-axis orientation: I_xx = I_yy = Ia, I_zz = Ib
    I = wp.mat33([[Ia, 0.0, 0.0], [0.0, Ia, 0.0], [0.0, 0.0, Ib]])

    return (m, wp.vec3(), I)


def compute_cone_inertia(density: float, r: float, h: float) -> tuple[float, wp.vec3, wp.mat33]:
    """Helper to compute mass and inertia of a solid cone extending along the z-axis

    Args:
        density: The cone density
        r: The cone radius
        h: The cone height (extent along the z-axis)

    Returns:

        A tuple of (mass, center of mass, inertia) with inertia specified around the center of mass
    """

    m = density * wp.pi * r * r * h / 3.0

    # Center of mass is at -h/4 from the geometric center
    # Since the cone has base at -h/2 and apex at +h/2, the COM is 1/4 of the height from base toward apex
    com = wp.vec3(0.0, 0.0, -h / 4.0)

    # Inertia about the center of mass
    Ia = 3 / 20 * m * r * r + 3 / 80 * m * h * h
    Ib = 3 / 10 * m * r * r

    # For Z-axis orientation: I_xx = I_yy = Ia, I_zz = Ib
    I = wp.mat33([[Ia, 0.0, 0.0], [0.0, Ia, 0.0], [0.0, 0.0, Ib]])

    return (m, com, I)


def compute_ellipsoid_inertia(density: float, a: float, b: float, c: float) -> tuple[float, wp.vec3, wp.mat33]:
    """Helper to compute mass and inertia of a solid ellipsoid

    The ellipsoid is centered at the origin with semi-axes a, b, c along the x, y, z axes respectively.

    Args:
        density: The ellipsoid density
        a: The semi-axis along the x-axis
        b: The semi-axis along the y-axis
        c: The semi-axis along the z-axis

    Returns:

        A tuple of (mass, center of mass, inertia) with inertia specified around the center of mass
    """
    # Volume of ellipsoid: V = (4/3) * pi * a * b * c
    v = (4.0 / 3.0) * wp.pi * a * b * c
    m = density * v

    # Inertia tensor for a solid ellipsoid about its center of mass:
    # Ixx = (1/5) * m * (b² + c²)
    # Iyy = (1/5) * m * (a² + c²)
    # Izz = (1/5) * m * (a² + b²)
    Ixx = (1.0 / 5.0) * m * (b * b + c * c)
    Iyy = (1.0 / 5.0) * m * (a * a + c * c)
    Izz = (1.0 / 5.0) * m * (a * a + b * b)

    I = wp.mat33([[Ixx, 0.0, 0.0], [0.0, Iyy, 0.0], [0.0, 0.0, Izz]])

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


@wp.func
def transform_inertia(m: float, I: wp.mat33, p: wp.vec3, q: wp.quat) -> wp.mat33:
    """
    Compute a rigid body's inertia tensor expressed in a new coordinate frame.

    The transformation applies (1) a rotation by quaternion ``q`` and
    (2) a parallel-axis shift by vector ``p`` (Steiner's theorem).
    Let ``R`` be the rotation matrix corresponding to ``q``. The returned
    inertia tensor :math:`\\mathbf{I}'` is

    .. math::

        \\mathbf{I}' \\,=\\, \\mathbf{R}\\,\\mathbf{I}\\,\\mathbf{R}^\\top
        \\, + \\, m\\big(\\lVert\\mathbf{p}\\rVert^2\\,\\mathbf{I}_3
        \\, - \\, \\mathbf{p}\\,\\mathbf{p}^\\top\\big),

    where :math:`\\mathbf{I}_3` is the :math:`3\\times3` identity matrix.

    Args:
        m (float): Mass of the rigid body.
        I (wp.mat33): Inertia tensor expressed in the body's local frame, relative
            to its center of mass.
        p (wp.vec3): Position vector from the new frame's origin to the body's
            center of mass.
        q (wp.quat): Orientation of the body relative to the new frame, expressed
            as a quaternion.

    Returns:
        wp.mat33: The transformed inertia tensor expressed in the new frame.
    """

    R = wp.quat_to_matrix(q)

    # Steiner's theorem
    return R @ I @ wp.transpose(R) + m * (
        wp.dot(p, p) * wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0) - wp.outer(p, p)
    )


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
        type: The type of shape (GeoType.SPHERE, GeoType.BOX, etc.)
        scale: The scale of the shape
        src: The source shape (Mesh or SDF)
        density: The density of the shape
        is_solid: Whether the shape is solid or hollow
        thickness: The thickness of the shape (used for collision detection, and inertia computation of hollow shapes)

    Returns:
        The mass, center of mass and 3x3 inertia tensor of the shape
    """
    if density == 0.0 or type == GeoType.PLANE:  # zero density means fixed
        return 0.0, wp.vec3(), wp.mat33()

    if type == GeoType.SPHERE:
        solid = compute_sphere_inertia(density, scale[0])
        if is_solid:
            return solid
        else:
            assert isinstance(thickness, float), "thickness must be a float for a hollow sphere geom"
            hollow = compute_sphere_inertia(density, scale[0] - thickness)
            return solid[0] - hollow[0], solid[1], solid[2] - hollow[2]
    elif type == GeoType.BOX:
        w, h, d = scale[0] * 2.0, scale[1] * 2.0, scale[2] * 2.0
        solid = compute_box_inertia(density, w, h, d)
        if is_solid:
            return solid
        else:
            assert isinstance(thickness, float), "thickness must be a float for a hollow box geom"
            hollow = compute_box_inertia(density, w - thickness, h - thickness, d - thickness)
            return solid[0] - hollow[0], solid[1], solid[2] - hollow[2]
    elif type == GeoType.CAPSULE:
        r, h = scale[0], scale[1] * 2.0
        solid = compute_capsule_inertia(density, r, h)
        if is_solid:
            return solid
        else:
            assert isinstance(thickness, float), "thickness must be a float for a hollow capsule geom"
            hollow = compute_capsule_inertia(density, r - thickness, h - 2.0 * thickness)
            return solid[0] - hollow[0], solid[1], solid[2] - hollow[2]
    elif type == GeoType.CYLINDER:
        r, h = scale[0], scale[1] * 2.0
        solid = compute_cylinder_inertia(density, r, h)
        if is_solid:
            return solid
        else:
            assert isinstance(thickness, float), "thickness must be a float for a hollow cylinder geom"
            hollow = compute_cylinder_inertia(density, r - thickness, h - 2.0 * thickness)
            return solid[0] - hollow[0], solid[1], solid[2] - hollow[2]
    elif type == GeoType.CONE:
        r, h = scale[0], scale[1] * 2.0
        solid = compute_cone_inertia(density, r, h)
        if is_solid:
            return solid
        else:
            assert isinstance(thickness, float), "thickness must be a float for a hollow cone geom"
            hollow = compute_cone_inertia(density, r - thickness, h - 2.0 * thickness)
            return solid[0] - hollow[0], solid[1], solid[2] - hollow[2]
    elif type == GeoType.ELLIPSOID:
        a, b, c = scale[0], scale[1], scale[2]
        solid = compute_ellipsoid_inertia(density, a, b, c)
        if is_solid:
            return solid
        else:
            assert isinstance(thickness, float), "thickness must be a float for a hollow ellipsoid geom"
            hollow = compute_ellipsoid_inertia(density, a - thickness, b - thickness, c - thickness)
            return solid[0] - hollow[0], solid[1], solid[2] - hollow[2]
    elif type == GeoType.MESH or type == GeoType.SDF or type == GeoType.CONVEX_MESH:
        assert src is not None, "src must be provided for mesh, SDF, or convex hull shapes"
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
        elif type == GeoType.MESH or type == GeoType.CONVEX_MESH:
            assert isinstance(src, Mesh), "src must be a Mesh for mesh or convex hull shapes"
            # fall back to computing inertia from mesh geometry
            vertices = np.array(src.vertices) * np.array(scale)
            m, c, I, _vol = compute_mesh_inertia(density, vertices, src.indices, is_solid, thickness)
            return m, c, I
    raise ValueError(f"Unsupported shape type: {type}")


def verify_and_correct_inertia(
    mass: float,
    inertia: wp.mat33,
    balance_inertia: bool = True,
    bound_mass: float | None = None,
    bound_inertia: float | None = None,
    body_label: str | None = None,
) -> tuple[float, wp.mat33, bool]:
    """Verify and correct inertia values similar to MuJoCo's balanceinertia compiler setting.

    This function checks for invalid inertia values and corrects them if needed. It performs
    the following checks and corrections:
    1. Ensures mass is non-negative (and bounded if specified)
    2. Ensures inertia diagonal elements are non-negative (and bounded if specified)
    3. Ensures inertia matrix satisfies triangle inequality (principal moments satisfy Ixx + Iyy >= Izz etc.)
    4. Optionally balances inertia to satisfy the triangle inequality exactly

    Args:
        mass: The mass of the body
        inertia: The 3x3 inertia tensor
        balance_inertia: If True, adjust inertia to exactly satisfy triangle inequality (like MuJoCo's balanceinertia)
        bound_mass: If specified, clamp mass to be at least this value
        bound_inertia: If specified, clamp inertia diagonal elements to be at least this value
        body_label: Optional label/name of the body for more informative warnings

    Returns:
        A tuple of (corrected_mass, corrected_inertia, was_corrected) where was_corrected
        indicates if any corrections were made
    """
    was_corrected = False
    corrected_mass = mass
    inertia_array = np.array(inertia).reshape(3, 3)
    corrected_inertia = inertia_array.copy()

    # Format body identifier for warnings
    body_id = f" for body '{body_label}'" if body_label else ""

    # Check and correct mass
    if mass < 0:
        warnings.warn(f"Negative mass {mass} detected{body_id}, setting to 0", stacklevel=2)
        corrected_mass = 0.0
        was_corrected = True
    elif bound_mass is not None and mass < bound_mass and mass > 0:
        warnings.warn(f"Mass {mass} is below bound {bound_mass}{body_id}, clamping", stacklevel=2)
        corrected_mass = bound_mass
        was_corrected = True

    # For zero mass, inertia should also be zero
    if corrected_mass == 0.0:
        if np.any(inertia_array != 0):
            warnings.warn(f"Zero mass body{body_id} should have zero inertia, correcting", stacklevel=2)
            corrected_inertia = np.zeros((3, 3))
            was_corrected = True
        return corrected_mass, wp.mat33(corrected_inertia), was_corrected

    # Check that inertia matrix is symmetric
    if not np.allclose(inertia_array, inertia_array.T):
        warnings.warn(f"Inertia matrix{body_id} is not symmetric, making it symmetric", stacklevel=2)
        corrected_inertia = (inertia_array + inertia_array.T) / 2
        was_corrected = True

    # Compute eigenvalues (principal moments) for validation
    try:
        eigenvalues = np.linalg.eigvals(corrected_inertia)

        # Check for negative eigenvalues
        if np.any(eigenvalues < 0):
            warnings.warn(
                f"Negative eigenvalues detected{body_id}: {eigenvalues}, making positive definite",
                stacklevel=2,
            )
            # Make positive definite by adjusting eigenvalues
            min_eig = np.min(eigenvalues)
            adjustment = -min_eig + 1e-6
            corrected_inertia += np.eye(3) * adjustment
            eigenvalues += adjustment
            was_corrected = True

        # Apply inertia bounds to eigenvalues if specified
        if bound_inertia is not None:
            min_eig = np.min(eigenvalues)
            if min_eig < bound_inertia:
                warnings.warn(
                    f"Minimum eigenvalue {min_eig} is below bound {bound_inertia}{body_id}, adjusting", stacklevel=2
                )
                adjustment = bound_inertia - min_eig
                corrected_inertia += np.eye(3) * adjustment
                eigenvalues += adjustment
                was_corrected = True

        # Sort eigenvalues to get principal moments
        principal_moments = np.sort(eigenvalues)
        I1, I2, I3 = principal_moments

        # Check triangle inequality on principal moments
        # For a physically valid inertia tensor: I1 + I2 >= I3 (with tolerance)
        has_violations = I1 + I2 < I3 - 1e-10

    except np.linalg.LinAlgError:
        warnings.warn(f"Failed to compute eigenvalues for inertia tensor{body_id}, making it diagonal", stacklevel=2)
        was_corrected = True
        # Fallback: use diagonal elements
        trace = np.trace(corrected_inertia)
        if trace <= 0:
            trace = 1e-6
        corrected_inertia = np.eye(3) * (trace / 3.0)
        has_violations = False
        principal_moments = [trace / 3.0, trace / 3.0, trace / 3.0]

    if has_violations:
        warnings.warn(
            f"Inertia tensor{body_id} violates triangle inequality with principal moments ({I1:.6f}, {I2:.6f}, {I3:.6f})",
            stacklevel=2,
        )

        if balance_inertia:
            # For non-diagonal matrices, we need to adjust while preserving the rotation
            deficit = I3 - I1 - I2
            if deficit > 0:
                # Simple approach: add scalar to all eigenvalues to ensure validity
                # This preserves eigenvectors exactly
                # We need: (I1 + a) + (I2 + a) >= I3 + a
                # Which simplifies to: I1 + I2 + a >= I3
                # So: a >= I3 - I1 - I2 = deficit
                adjustment = deficit + 1e-6

                # Add scalar*I to shift all eigenvalues equally
                corrected_inertia = corrected_inertia + np.eye(3) * adjustment
                was_corrected = True

                # Update principal moments
                new_I1 = I1 + adjustment
                new_I2 = I2 + adjustment
                new_I3 = I3 + adjustment

                warnings.warn(
                    f"Balanced principal moments{body_id} from ({I1:.6f}, {I2:.6f}, {I3:.6f}) to "
                    f"({new_I1:.6f}, {new_I2:.6f}, {new_I3:.6f})",
                    stacklevel=2,
                )

    # Final check: ensure the corrected inertia matrix is positive definite
    if has_violations and balance_inertia:
        # Need to recompute after balancing since we modified the matrix
        try:
            eigenvalues = np.linalg.eigvals(corrected_inertia)
        except np.linalg.LinAlgError:
            warnings.warn(f"Failed to compute eigenvalues of inertia matrix{body_id}", stacklevel=2)
            eigenvalues = np.array([0.0, 0.0, 0.0])

    # Check final eigenvalues
    if np.any(eigenvalues <= 0) or np.any(~np.isfinite(eigenvalues)):
        warnings.warn(
            f"Corrected inertia matrix{body_id} is not positive definite, this should not happen", stacklevel=2
        )
        # As a last resort, make it positive definite by adding a small value to diagonal
        min_eigenvalue = np.min(eigenvalues[np.isfinite(eigenvalues)]) if np.any(np.isfinite(eigenvalues)) else -1e-6
        epsilon = abs(min_eigenvalue) + 1e-6
        corrected_inertia[0, 0] += epsilon
        corrected_inertia[1, 1] += epsilon
        corrected_inertia[2, 2] += epsilon
        was_corrected = True

    return corrected_mass, wp.mat33(corrected_inertia), was_corrected


@wp.kernel(enable_backward=False, module="unique")
def validate_and_correct_inertia_kernel(
    body_mass: wp.array(dtype=wp.float32),
    body_inertia: wp.array(dtype=wp.mat33),
    body_inv_mass: wp.array(dtype=wp.float32),
    body_inv_inertia: wp.array(dtype=wp.mat33),
    balance_inertia: wp.bool,
    bound_mass: wp.float32,
    bound_inertia: wp.float32,
    correction_flags: wp.array(dtype=wp.bool),  # Output: True if corrected, False otherwise
):
    """Warp kernel for parallel inertia validation and correction.

    This kernel performs basic validation and correction but doesn't support:
    - Warning messages (handled by caller)
    - Complex iterative balancing (falls back to simple correction)
    """
    tid = wp.tid()

    mass = body_mass[tid]
    inertia = body_inertia[tid]
    was_corrected = False

    # Check for negative mass
    if mass < 0.0:
        mass = 0.0
        was_corrected = True

    # Apply mass bound
    if bound_mass > 0.0 and mass < bound_mass:
        mass = bound_mass
        was_corrected = True

    # For zero mass, inertia should be zero
    if mass == 0.0:
        was_corrected = was_corrected or (wp.ddot(inertia, inertia) > 0.0)
        inertia = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    else:
        # Use eigendecomposition for proper validation
        _eigvecs, eigvals = wp.eig3(inertia)

        # Sort eigenvalues to get principal moments (I1 <= I2 <= I3)
        I1, I2, I3 = eigvals[0], eigvals[1], eigvals[2]
        if I1 > I2:
            I1, I2 = I2, I1
        if I2 > I3:
            I2, I3 = I3, I2
            if I1 > I2:
                I1, I2 = I2, I1

        # Check for negative eigenvalues
        if I1 < 0.0:
            adjustment = -I1 + 1e-6
            # Add scalar to all eigenvalues
            I1 += adjustment
            I2 += adjustment
            I3 += adjustment
            inertia = inertia + wp.mat33(adjustment, 0.0, 0.0, 0.0, adjustment, 0.0, 0.0, 0.0, adjustment)
            was_corrected = True

        # Apply eigenvalue bounds
        if bound_inertia > 0.0 and I1 < bound_inertia:
            adjustment = bound_inertia - I1
            I1 += adjustment
            I2 += adjustment
            I3 += adjustment
            inertia = inertia + wp.mat33(adjustment, 0.0, 0.0, 0.0, adjustment, 0.0, 0.0, 0.0, adjustment)
            was_corrected = True

        # Check triangle inequality: I1 + I2 >= I3 (with tolerance)
        # Use larger tolerance for float32 precision
        if balance_inertia and (I1 + I2 < I3 - 1e-6):
            deficit = I3 - I1 - I2
            adjustment = deficit + 1e-6
            # Add scalar*I to fix triangle inequality
            inertia = inertia + wp.mat33(adjustment, 0.0, 0.0, 0.0, adjustment, 0.0, 0.0, 0.0, adjustment)
            was_corrected = True

    # Write back corrected values
    body_mass[tid] = mass
    body_inertia[tid] = inertia

    # Update inverse mass
    if mass > 0.0:
        body_inv_mass[tid] = 1.0 / mass
    else:
        body_inv_mass[tid] = 0.0

    # Update inverse inertia
    if mass > 0.0:
        body_inv_inertia[tid] = wp.inverse(inertia)
    else:
        body_inv_inertia[tid] = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    correction_flags[tid] = was_corrected
