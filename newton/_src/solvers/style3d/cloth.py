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


"""Style3D cloth helpers built on ModelBuilder custom attributes."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
import warp as wp

from ...core.types import Vec2, Vec3
from ...geometry.flags import ParticleFlags
from ...geometry.kernels import compute_edge_aabbs
from ...utils.mesh import MeshAdjacency

if TYPE_CHECKING:
    from ...sim.builder import ModelBuilder


def _normalize_edge_aniso_values(edge_aniso_ke: Sequence[Vec3] | Vec3 | None, edge_count: int) -> list[Vec3]:
    """Normalize edge anisotropy values to a per-edge list."""
    if edge_aniso_ke is None:
        return [wp.vec3(0.0, 0.0, 0.0)] * edge_count
    if isinstance(edge_aniso_ke, (list, tuple, np.ndarray)):
        values = list(edge_aniso_ke)
        if len(values) == 1 and edge_count != 1:
            return [values[0]] * edge_count
        if len(values) != edge_count:
            raise ValueError(f"Expected {edge_count} edge_aniso_ke values, got {len(values)}")
        return values
    return [edge_aniso_ke] * edge_count


def _compute_panel_triangles(panel_verts: np.ndarray, panel_indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return per-triangle inverse rest matrices and areas from panel UVs."""
    p = panel_verts[panel_indices[:, 0]]
    q = panel_verts[panel_indices[:, 1]]
    r = panel_verts[panel_indices[:, 2]]

    qp = q - p
    rp = r - p

    D = np.concatenate((qp[..., None], rp[..., None]), axis=-1)
    areas = np.linalg.det(D) / 2.0
    areas[areas < 0.0] = 0.0

    D[areas == 0.0] = np.eye(2)[None, ...]
    inv_D = np.linalg.inv(D)
    return inv_D, areas


def _compute_edge_bending_data(
    panel_verts: np.ndarray,
    panel_indices: np.ndarray,
    tri_indices: np.ndarray,
    edge_aniso_ke: Sequence[Vec3] | Vec3 | None,
) -> tuple[
    np.ndarray,
    list[float] | None,
    list[float],
    list[tuple[float, float, float, float]],
    list[Vec3] | None,
]:
    """Compute edge indices and bending data from panel-space geometry."""
    adjacency = MeshAdjacency(tri_indices.tolist(), len(tri_indices))
    edge_indices = np.fromiter(
        (x for e in adjacency.edges.values() for x in (e.o0, e.o1, e.v0, e.v1, e.f0, e.f1)),
        int,
    ).reshape(-1, 6)

    edge_count = edge_indices.shape[0]
    edge_aniso_values = None
    edge_ke = None
    if edge_aniso_ke is not None:
        edge_aniso_values = _normalize_edge_aniso_values(edge_aniso_ke, edge_count)

    panel_tris = panel_indices
    panel_pos2d = panel_verts
    f0 = edge_indices[:, 4]
    f1 = edge_indices[:, 5]
    v0 = edge_indices[:, 0]
    v1 = edge_indices[:, 1]

    edge_v0_order = np.argmax(tri_indices[f0] == v0[:, None], axis=1)
    edge_v1_order = np.argmax(tri_indices[f1] == v1[:, None], axis=1)

    panel_tris_f0 = panel_tris[f0]
    panel_tris_f1 = panel_tris[f1]

    panel_x1_f0 = panel_pos2d[panel_tris_f0[np.arange(panel_tris_f0.shape[0]), edge_v0_order]]
    panel_x3_f0 = panel_pos2d[panel_tris_f0[np.arange(panel_tris_f0.shape[0]), (edge_v0_order + 1) % 3]]
    panel_x4_f0 = panel_pos2d[panel_tris_f0[np.arange(panel_tris_f0.shape[0]), (edge_v0_order + 2) % 3]]

    panel_x2_f1 = panel_pos2d[panel_tris_f1[np.arange(panel_tris_f1.shape[0]), edge_v1_order]]
    panel_x4_f1 = panel_pos2d[panel_tris_f1[np.arange(panel_tris_f1.shape[0]), (edge_v1_order + 1) % 3]]
    panel_x3_f1 = panel_pos2d[panel_tris_f1[np.arange(panel_tris_f1.shape[0]), (edge_v1_order + 2) % 3]]

    panel_x43_f0 = panel_x4_f0 - panel_x3_f0
    panel_x43_f1 = panel_x4_f1 - panel_x3_f1

    def dot(a, b):
        return (a * b).sum(axis=-1)

    if edge_aniso_values is not None:
        angle_f0 = np.atan2(panel_x43_f0[:, 1], panel_x43_f0[:, 0])
        angle_f1 = np.atan2(panel_x43_f1[:, 1], panel_x43_f1[:, 0])
        angle = (angle_f0 + angle_f1) * 0.5
        sin = np.sin(angle)
        cos = np.cos(angle)
        sin2 = np.pow(sin, 2)
        cos2 = np.pow(cos, 2)
        sin12 = np.pow(sin, 12)
        cos12 = np.pow(cos, 12)
        aniso_ke = np.array(edge_aniso_values, dtype=float).reshape(-1, 3)
        edge_ke = aniso_ke[:, 0] * sin12 + aniso_ke[:, 1] * cos12 + aniso_ke[:, 2] * 4.0 * sin2 * cos2

    edge_area = (
        np.abs(np.cross(panel_x43_f0, panel_x1_f0 - panel_x3_f0))
        + np.abs(np.cross(panel_x43_f1, panel_x2_f1 - panel_x3_f1))
        + 1.0e-8
    ) / 3.0

    def cot2d(a, b, c):
        ba = b - a
        ca = c - a
        dot_a = dot(ba, ca)
        cross_a = np.abs(np.cross(ba, ca)) + 1.0e-8
        return dot_a / cross_a

    cot1 = cot2d(panel_x3_f0, panel_x4_f0, panel_x1_f0)
    cot2 = cot2d(panel_x3_f1, panel_x4_f1, panel_x2_f1)
    cot3 = cot2d(panel_x4_f0, panel_x3_f0, panel_x1_f0)
    cot4 = cot2d(panel_x4_f1, panel_x3_f1, panel_x2_f1)
    edge_bending_cot = list(zip(cot1, cot2, cot3, cot4, strict=False))

    edge_ke_values = edge_ke.tolist() if edge_ke is not None else None
    return edge_indices[:, :4], edge_ke_values, edge_area.tolist(), edge_bending_cot, edge_aniso_values


def add_style3d_cloth_mesh(
    builder: ModelBuilder,
    *,
    pos: Vec3,
    rot: Any,
    vel: Vec3,
    vertices: Sequence[Vec3],
    indices: Sequence[int],
    density: float,
    scale: float = 1.0,
    panel_verts: Sequence[Vec2] | None = None,
    panel_indices: Sequence[int] | None = None,
    tri_aniso_ke: Sequence[Vec3] | Vec3 | None = None,
    edge_aniso_ke: Sequence[Vec3] | Vec3 | None = None,
    tri_ka: float | None = None,
    tri_kd: float | None = None,
    tri_drag: float | None = None,
    tri_lift: float | None = None,
    edge_kd: float | None = None,
    add_springs: bool = False,
    spring_ke: float | None = None,
    spring_kd: float | None = None,
    particle_radius: float | None = None,
    custom_attributes_particles: dict[str, Any] | None = None,
    custom_attributes_springs: dict[str, Any] | None = None,
) -> None:
    """Add a Style3D cloth mesh using ModelBuilder and custom attributes.

    This helper uses the standard ``ModelBuilder.add_particles``, ``add_triangles``,
    and ``add_edges`` APIs. It computes panel-space rest data for anisotropic
    stretch and bending, then injects the Style3D custom attributes:

    - ``style3d:tri_aniso_ke`` (triangle frequency)
    - ``style3d:edge_rest_area`` (edge frequency)
    - ``style3d:edge_bending_cot`` (edge frequency)
    - ``style3d:aniso_ke`` (edge frequency)

    It also overwrites ``builder.tri_poses`` and ``builder.tri_areas`` with
    panel-based rest data. Make sure to call
    ``SolverStyle3D.register_custom_attributes(builder)`` before invoking this.

    Args:
        builder: ModelBuilder to populate.
        pos: World-space translation for the mesh.
        rot: World-space rotation for the mesh.
        vel: Initial velocity for all particles.
        vertices: 3D mesh vertices (list of positions).
        indices: Triangle indices (3 entries per face).
        density: Mass per unit area in panel space.
        scale: Uniform scale applied to ``vertices`` and ``panel_verts``.
        panel_verts: 2D panel coordinates (UVs). Defaults to ``vertices`` XY.
        panel_indices: Triangle indices in panel space. Defaults to ``indices``.
        tri_aniso_ke: Anisotropic stretch stiffness (weft, warp, shear). Can be a
            single value or per-triangle list. If a list is provided for all
            triangles, it is filtered to valid faces.
        edge_aniso_ke: Anisotropic bending stiffness values. Can be a single
            value or per-edge list (computed from mesh adjacency).
        tri_ka: Triangle area stiffness (defaults to ``builder.default_tri_ka``).
        tri_kd: Triangle damping (defaults to ``builder.default_tri_kd``).
        tri_drag: Triangle drag coefficient (defaults to ``builder.default_tri_drag``).
        tri_lift: Triangle lift coefficient (defaults to ``builder.default_tri_lift``).
        edge_kd: Edge damping (defaults to ``builder.default_edge_kd``).
        add_springs: If True, add structural springs across mesh edges.
        spring_ke: Spring stiffness (defaults to ``builder.default_spring_ke``).
        spring_kd: Spring damping (defaults to ``builder.default_spring_kd``).
        particle_radius: Per-particle radius (defaults to ``builder.default_particle_radius``).
        custom_attributes_particles: Extra custom attributes for particles.
        custom_attributes_springs: Extra custom attributes for springs.
    """
    vertices_np = np.array(vertices, dtype=float) * scale
    rot_mat = np.array(wp.quat_to_matrix(rot), dtype=np.float32).reshape(3, 3)
    verts_3d = np.dot(vertices_np, rot_mat.T) + np.array(pos, dtype=float)

    panel_verts_np = np.array(panel_verts, dtype=float) if panel_verts is not None else vertices_np[:, :2]
    panel_indices_np = np.array(panel_indices if panel_indices is not None else indices, dtype=int).reshape(-1, 3)

    tri_indices_np = np.array(indices, dtype=int).reshape(-1, 3)
    panel_inv_D_all, panel_areas_all = _compute_panel_triangles(panel_verts_np, panel_indices_np)
    valid_inds = (panel_areas_all > 0.0).nonzero()[0]
    if len(valid_inds) < len(panel_areas_all):
        print("inverted or degenerate triangle elements")
    tri_indices_valid = tri_indices_np[valid_inds]
    panel_indices_valid = panel_indices_np[valid_inds]

    start_vertex = len(builder.particle_q)
    radius_value = particle_radius if particle_radius is not None else builder.default_particle_radius
    builder.add_particles(
        verts_3d.tolist(),
        [vel] * len(verts_3d),
        mass=[0.0] * len(verts_3d),
        radius=[radius_value] * len(verts_3d),
        custom_attributes=custom_attributes_particles,
    )

    tri_aniso_values = tri_aniso_ke or wp.vec3(builder.default_tri_ke)
    if isinstance(tri_aniso_values, (list, tuple, np.ndarray)):
        tri_aniso_values = list(tri_aniso_values)
        if len(tri_aniso_values) == len(tri_indices_np):
            tri_aniso_values = [tri_aniso_values[idx] for idx in valid_inds]
        elif len(tri_aniso_values) == 1 and len(tri_indices_valid) != 1:
            tri_aniso_values = [tri_aniso_values[0]] * len(tri_indices_valid)
        elif len(tri_aniso_values) != len(tri_indices_valid):
            raise ValueError(f"Expected {len(tri_indices_valid)} tri_aniso_ke values, got {len(tri_aniso_values)}")

    tri_start = len(builder.tri_indices)
    tri_areas = builder.add_triangles(
        (tri_indices_valid[:, 0] + start_vertex).tolist(),
        (tri_indices_valid[:, 1] + start_vertex).tolist(),
        (tri_indices_valid[:, 2] + start_vertex).tolist(),
        tri_ke=[builder.default_tri_ke] * len(tri_indices_valid),
        tri_ka=[tri_ka if tri_ka is not None else builder.default_tri_ka] * len(tri_indices_valid),
        tri_kd=[tri_kd if tri_kd is not None else builder.default_tri_kd] * len(tri_indices_valid),
        tri_drag=[tri_drag if tri_drag is not None else builder.default_tri_drag] * len(tri_indices_valid),
        tri_lift=[tri_lift if tri_lift is not None else builder.default_tri_lift] * len(tri_indices_valid),
        custom_attributes={"style3d:tri_aniso_ke": tri_aniso_values},
    )

    panel_inv_D = panel_inv_D_all[valid_inds]
    panel_areas = panel_areas_all[valid_inds]
    tri_end = tri_start + len(tri_areas)
    builder.tri_poses[tri_start:tri_end] = panel_inv_D.tolist()
    builder.tri_areas[tri_start:tri_end] = panel_areas.tolist()

    for t, area in enumerate(panel_areas.tolist()):
        i, j, k = tri_indices_valid[t]
        builder.particle_mass[start_vertex + i] += density * area / 3.0
        builder.particle_mass[start_vertex + j] += density * area / 3.0
        builder.particle_mass[start_vertex + k] += density * area / 3.0

    edge_indices_local, edge_ke, edge_rest_area, edge_bending_cot, edge_aniso_values = _compute_edge_bending_data(
        panel_verts_np,
        panel_indices_valid,
        tri_indices_valid,
        edge_aniso_ke,
    )
    edge_indices_global = edge_indices_local.copy()
    edge_indices_global[edge_indices_global >= 0] += start_vertex

    if edge_ke is None:
        edge_ke = [builder.default_edge_ke] * len(edge_indices_global)

    edge_kd_value = edge_kd if edge_kd is not None else builder.default_edge_kd
    edge_kd_list = [edge_kd_value] * len(edge_ke)

    edge_custom_attrs = {
        "style3d:edge_rest_area": edge_rest_area,
        "style3d:edge_bending_cot": edge_bending_cot,
    }
    if edge_aniso_values is not None:
        edge_custom_attrs["style3d:aniso_ke"] = edge_aniso_values

    builder.add_edges(
        edge_indices_global[:, 0].tolist(),
        edge_indices_global[:, 1].tolist(),
        edge_indices_global[:, 2].tolist(),
        edge_indices_global[:, 3].tolist(),
        edge_ke=edge_ke,
        edge_kd=edge_kd_list,
        custom_attributes=edge_custom_attrs,
    )

    if add_springs:
        spring_indices = set()
        for i, j, k, l in edge_indices_global:
            spring_indices.add((min(k, l), max(k, l)))
            if i != -1:
                spring_indices.add((min(i, k), max(i, k)))
                spring_indices.add((min(i, l), max(i, l)))
            if j != -1:
                spring_indices.add((min(j, k), max(j, k)))
                spring_indices.add((min(j, l), max(j, l)))
            if i != -1 and j != -1:
                spring_indices.add((min(i, j), max(i, j)))

        spring_ke_value = spring_ke if spring_ke is not None else builder.default_spring_ke
        spring_kd_value = spring_kd if spring_kd is not None else builder.default_spring_kd
        for i, j in spring_indices:
            builder.add_spring(
                i,
                j,
                spring_ke_value,
                spring_kd_value,
                control=0.0,
                custom_attributes=custom_attributes_springs,
            )


def add_style3d_cloth_grid(
    builder: ModelBuilder,
    *,
    pos: Vec3,
    rot: Any,
    vel: Vec3,
    dim_x: int,
    dim_y: int,
    cell_x: float,
    cell_y: float,
    mass: float,
    reverse_winding: bool = False,
    fix_left: bool = False,
    fix_right: bool = False,
    fix_top: bool = False,
    fix_bottom: bool = False,
    tri_aniso_ke: Sequence[Vec3] | Vec3 | None = None,
    tri_ka: float | None = None,
    tri_kd: float | None = None,
    tri_drag: float | None = None,
    tri_lift: float | None = None,
    edge_aniso_ke: Sequence[Vec3] | Vec3 | None = None,
    edge_kd: float | None = None,
    add_springs: bool = False,
    spring_ke: float | None = None,
    spring_kd: float | None = None,
    particle_radius: float | None = None,
    custom_attributes_particles: dict[str, Any] | None = None,
    custom_attributes_springs: dict[str, Any] | None = None,
) -> None:
    """Create a planar Style3D cloth grid.

    Call ``SolverStyle3D.register_custom_attributes(builder)`` before invoking
    this helper so the Style3D custom attributes are available on the builder.

    Args:
        builder: ModelBuilder to populate.
        pos: World-space translation for the grid.
        rot: World-space rotation for the grid.
        vel: Initial velocity for all particles.
        dim_x: Number of grid cells along X (creates dim_x + 1 vertices).
        dim_y: Number of grid cells along Y (creates dim_y + 1 vertices).
        cell_x: Cell size along X in panel space.
        cell_y: Cell size along Y in panel space.
        mass: Mass per particle (used to compute density).
        reverse_winding: If True, flip triangle winding.
        fix_left: Fix particles on the left edge.
        fix_right: Fix particles on the right edge.
        fix_top: Fix particles on the top edge.
        fix_bottom: Fix particles on the bottom edge.
        tri_aniso_ke: Anisotropic stretch stiffness (weft, warp, shear).
        tri_ka: Triangle area stiffness.
        tri_kd: Triangle damping.
        tri_drag: Triangle drag coefficient.
        tri_lift: Triangle lift coefficient.
        edge_aniso_ke: Anisotropic bending stiffness values.
        edge_kd: Edge damping.
        add_springs: If True, add structural springs across mesh edges.
        spring_ke: Spring stiffness.
        spring_kd: Spring damping.
        particle_radius: Per-particle radius.
        custom_attributes_particles: Extra custom attributes for particles.
        custom_attributes_springs: Extra custom attributes for springs.
    """

    def grid_index(x: int, y: int, dim_x: int) -> int:
        return y * dim_x + x

    indices: list[int] = []
    vertices: list[Vec3] = []
    panel_verts: list[Vec2] = []
    for y in range(0, dim_y + 1):
        for x in range(0, dim_x + 1):
            local_pos = wp.vec3(x * cell_x, y * cell_y, 0.0)
            vertices.append(local_pos)
            panel_verts.append(wp.vec2(local_pos[0], local_pos[1]))
            if x > 0 and y > 0:
                v0 = grid_index(x - 1, y - 1, dim_x + 1)
                v1 = grid_index(x, y - 1, dim_x + 1)
                v2 = grid_index(x, y, dim_x + 1)
                v3 = grid_index(x - 1, y, dim_x + 1)
                if reverse_winding:
                    indices.extend([v0, v1, v2])
                    indices.extend([v0, v2, v3])
                else:
                    indices.extend([v0, v1, v3])
                    indices.extend([v1, v2, v3])

    total_mass = mass * (dim_x + 1) * (dim_x + 1)
    total_area = cell_x * cell_y * dim_x * dim_y
    density = total_mass / total_area

    start_vertex = len(builder.particle_q)
    add_style3d_cloth_mesh(
        builder,
        pos=pos,
        rot=rot,
        vel=vel,
        vertices=vertices,
        indices=indices,
        density=density,
        scale=1.0,
        panel_verts=panel_verts,
        panel_indices=indices,
        tri_aniso_ke=tri_aniso_ke,
        edge_aniso_ke=edge_aniso_ke,
        tri_ka=tri_ka,
        tri_kd=tri_kd,
        tri_drag=tri_drag,
        tri_lift=tri_lift,
        edge_kd=edge_kd,
        add_springs=add_springs,
        spring_ke=spring_ke,
        spring_kd=spring_kd,
        particle_radius=particle_radius,
        custom_attributes_particles=custom_attributes_particles,
        custom_attributes_springs=custom_attributes_springs,
    )

    if fix_left or fix_right or fix_top or fix_bottom:
        vertex_id = 0
        for y in range(dim_y + 1):
            for x in range(dim_x + 1):
                if (
                    (x == 0 and fix_left)
                    or (x == dim_x and fix_right)
                    or (y == 0 and fix_bottom)
                    or (y == dim_y and fix_top)
                ):
                    builder.particle_flags[start_vertex + vertex_id] = (
                        builder.particle_flags[start_vertex + vertex_id] & ~ParticleFlags.ACTIVE
                    )
                    builder.particle_mass[start_vertex + vertex_id] = 0.0
                vertex_id += 1


@wp.kernel
def compute_sew_v(
    sew_dist: float,
    bvh_id: wp.uint64,
    pos: wp.array(dtype=wp.vec3),
    edge_indices: wp.array(dtype=wp.int32, ndim=2),
    vert_indices: wp.array(dtype=wp.int32),
    # outputs
    sew_vinds: wp.array(dtype=wp.vec2i, ndim=2),
    sew_vdists: wp.array(dtype=wp.float32, ndim=2),
):
    v_index = vert_indices[wp.tid()]
    v = pos[v_index]
    lower = wp.vec3(v[0] - sew_dist, v[1] - sew_dist, v[2] - sew_dist)
    upper = wp.vec3(v[0] + sew_dist, v[1] + sew_dist, v[2] + sew_dist)

    query = wp.bvh_query_aabb(bvh_id, lower, upper)

    edge_index = wp.int32(-1)
    vertex_num_sew = wp.int32(0)
    max_num_sew = sew_vinds.shape[1]
    while wp.bvh_query_next(query, edge_index):
        va_ind = edge_indices[edge_index, 2]
        vb_ind = edge_indices[edge_index, 3]
        if v_index == va_ind or v_index == vb_ind:
            continue
        va = pos[va_ind]
        vb = pos[vb_ind]
        check_va = bool(True)
        check_vb = bool(True)
        for i in range(vertex_num_sew):
            if sew_vinds[wp.tid()][i][1] == va_ind:
                check_va = False
                break

        for i in range(vertex_num_sew):
            if sew_vinds[wp.tid()][i][1] == vb_ind:
                check_vb = False
                break

        if v_index < va_ind and check_va:
            da = wp.length(va - v)
            if da <= sew_dist:
                if vertex_num_sew < max_num_sew:
                    sew_vinds[wp.tid()][vertex_num_sew][0] = v_index
                    sew_vinds[wp.tid()][vertex_num_sew][1] = va_ind
                    sew_vdists[wp.tid()][vertex_num_sew] = da
                    vertex_num_sew = vertex_num_sew + 1
                else:
                    for i in range(max_num_sew):
                        if da < sew_vdists[wp.tid()][i]:
                            sew_vinds[wp.tid()][i][0] = v_index
                            sew_vinds[wp.tid()][i][1] = va_ind
                            sew_vdists[wp.tid()][i] = da
                            break
        if v_index < vb_ind and check_vb:
            db = wp.length(vb - v)
            if db <= sew_dist:
                if vertex_num_sew < max_num_sew:
                    sew_vinds[wp.tid()][vertex_num_sew][0] = v_index
                    sew_vinds[wp.tid()][vertex_num_sew][1] = vb_ind
                    sew_vdists[wp.tid()][vertex_num_sew] = db
                    vertex_num_sew = vertex_num_sew + 1
                else:
                    for i in range(max_num_sew):
                        if db < sew_vdists[wp.tid()][i]:
                            sew_vinds[wp.tid()][i][0] = v_index
                            sew_vinds[wp.tid()][i][1] = vb_ind
                            sew_vdists[wp.tid()][i] = db
                            break


def create_trimesh_sew_springs(
    particle_q: np.ndarray,
    edge_indices: np.ndarray,
    sew_distance: float = 1.0e-3,
    sew_interior: bool = False,
):
    """
    A function that creates sew springs for trimesh.
    It will sew vertices within sew_distance.
    It returns the sew spring indices.

    Args:
        particle_q: The particle positions.
        edge_indices: The edge indices.
        sew_distance: The distance within which vertices will be connected by springs.
        sew_interior: If True, can sew between interior vertices, otherwise only sew boundary-interior
            or boundary-boundary vertices

    """

    trimesh_edge_indices = np.array(edge_indices)
    # compute unique vert indices
    flat_inds = trimesh_edge_indices.flatten()
    flat_inds = flat_inds[flat_inds >= 0]
    vert_inds = np.unique(flat_inds)
    # compute unique boundary vert indices
    bound_condition = trimesh_edge_indices[:, 1] < 0
    bound_edge_inds = trimesh_edge_indices[bound_condition]
    bound_edge_inds = bound_edge_inds[:, 2:4]
    bound_vert_inds = np.unique(bound_edge_inds.flatten())
    # compute edge bvh
    num_edge = trimesh_edge_indices.shape[0]
    lower_bounds_edges = wp.array(shape=(num_edge,), dtype=wp.vec3, device="cpu")
    upper_bounds_edges = wp.array(shape=(num_edge,), dtype=wp.vec3, device="cpu")
    wp_edge_indices = wp.array(edge_indices, dtype=wp.int32, device="cpu")
    wp_vert_pos = wp.array(particle_q, dtype=wp.vec3, device="cpu")
    wp.launch(
        kernel=compute_edge_aabbs,
        inputs=[wp_vert_pos, wp_edge_indices],
        outputs=[lower_bounds_edges, upper_bounds_edges],
        dim=num_edge,
        device="cpu",
    )
    bvh_edges = wp.Bvh(lower_bounds_edges, upper_bounds_edges)

    # compute sew springs
    max_num_sew = 5
    if sew_interior:
        num_vert = vert_inds.shape[0]
        wp_vert_inds = wp.array(vert_inds, dtype=wp.int32, device="cpu")
    else:
        num_vert = bound_vert_inds.shape[0]
        wp_vert_inds = wp.array(bound_vert_inds, dtype=wp.int32, device="cpu")

    wp_sew_vinds = wp.full(
        shape=(num_vert, max_num_sew), value=wp.vec2i(-1, -1), dtype=wp.vec2i, device="cpu"
    )  # each vert sew max 5 other verts
    wp_sew_vdists = wp.full(shape=(num_vert, max_num_sew), value=sew_distance, dtype=wp.float32, device="cpu")
    wp.launch(
        kernel=compute_sew_v,
        inputs=[sew_distance, bvh_edges.id, wp_vert_pos, wp_edge_indices, wp_vert_inds],
        outputs=[wp_sew_vinds, wp_sew_vdists],
        dim=num_vert,
        device="cpu",
    )

    np_sew_vinds = wp_sew_vinds.numpy().reshape(num_vert * max_num_sew, 2)
    np_sew_vinds = np_sew_vinds[np_sew_vinds[:, 0] >= 0]

    return np_sew_vinds


def sew_close_vertices(builder: ModelBuilder, sew_distance: float = 1.0e-3, sew_interior: bool = False) -> None:
    """Sew close vertices by creating springs between nearby mesh vertices.

    Args:
        builder: ModelBuilder with triangle/edge topology.
        sew_distance: Vertices within this distance are connected by springs.
        sew_interior: If True, allow interior-interior sewing; otherwise only
            boundary-interior or boundary-boundary vertices are sewn.
    """
    sew_springs = create_trimesh_sew_springs(
        builder.particle_q,
        builder.edge_indices,
        sew_distance,
        sew_interior,
    )
    for spring in sew_springs:
        builder.add_spring(
            spring[0],
            spring[1],
            builder.default_spring_ke,
            builder.default_spring_kd,
            control=0.0,
        )


__all__ = [
    "add_style3d_cloth_grid",
    "add_style3d_cloth_mesh",
    "create_trimesh_sew_springs",
    "sew_close_vertices",
]
