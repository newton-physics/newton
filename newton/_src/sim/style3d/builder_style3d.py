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

"""A module for building Style3D models."""

from __future__ import annotations

import numpy as np
import warp as wp

from ...core.types import (
    Axis,
    AxisType,
    Devicelike,
    Quat,
    Transform,
    Vec2,
    Vec3,
)
from ...geometry import ParticleFlags
from ..builder import ModelBuilder
from ..sew import create_trimesh_sew_springs
from .model_style3d import Style3DModel


class Style3DModelBuilder(ModelBuilder):
    """A helper class for building style3d simulation models at runtime.

    Use the Style3DModelBuilder to construct a simulation scene. The Style3DModelBuilder
    builds the scene representation using standard Python data structures (lists),
    this means it is not differentiable. Once :meth:`finalize()`
    has been called the ModelBuilder transfers all data to Warp tensors and returns
    an object that may be used for simulation.

    Example
    -------

    Example code::

        import newton
        from newton.solvers import SolverStyle3D
        from newton.sim import Style3DModelBuilder

        builder = Style3DModelBuilder()

        builder.add_cloth_grid(
            pos=wp.vec3(-0.5, 2.0, 0.0),
            rot=wp.quat_from_axis_angle(axis=wp.vec3(1, 0, 0), angle=wp.pi / 2.0),
            dim_x=grid_dim,
            dim_y=grid_dim,
            cell_x=grid_width / grid_dim,
            cell_y=grid_width / grid_dim,
            vel=wp.vec3(0.0, 0.0, 0.0),
            mass=cloth_density * (grid_width * grid_width) / (grid_dim * grid_dim),
            tri_ke=1.0e2,
            tri_ka=1.0e2,
            tri_kd=2.0e-6,
            edge_ke=1,
            tri_aniso_ke=wp.vec3(1.0e2, 1.0e2, 1.0e1),
            edge_aniso_ke=wp.vec3(2.0e-5, 1.0e-5, 5.0e-6),
        )

        # create model
        model = builder.finalize()

        state_0, state_1 = model.state(), model.state()
        control = model.control()
        solver = SolverStyle3D(model)

        for i in range(10):
            state_0.clear_forces()
            contacts = model.collide(state_0)
            solver.step(state_0, state_1, control, contacts, dt=1.0 / 60.0)
            state_0, state_1 = state_1, state_0

    Note:
        It is strongly recommended to use the Style3DModelBuilder to construct a simulation rather
        than creating your own Model object directly, however it is possible to do so if
        desired.
    """

    def __init__(self, up_axis: AxisType = Axis.Z, gravity: float = -9.81):
        """
        Initializes a new Style3DModelBuilder.

        Args:
            up_axis (AxisType, optional): The up axis for the simulation. Defaults to Axis.Z.
            gravity (float, optional): The gravitational acceleration. Defaults to -9.81.
        """
        super().__init__(up_axis=up_axis, gravity=gravity)

        # triangles
        self.tri_aniso_ke = []
        # edges (bending)
        self.edge_rest_area = []
        self.edge_bending_cot = []

    def add_builder(
        self,
        builder: Style3DModelBuilder,
        xform: Transform | None = None,
    ):
        """Copies the data from another `Style3DModelBuilder` to this `Style3DModelBuilder`.

        All entities from the source builder are added to this builder's current world context.
        Any world assignments that existed in the source builder are overwritten.

        Args:
            builder (Style3DModelBuilder): The model builder to copy data from.
            xform (Transform): Optional offset transform applied to root bodies.
        """

        super().add_builder(
            builder=builder,
            xform=xform,
        )

        style3d_builder_attrs = [
            "tri_aniso_ke",
            "edge_rest_area",
            "edge_bending_cot",
        ]

        for attr in style3d_builder_attrs:
            getattr(self, attr).extend(getattr(builder, attr))

    def add_aniso_triangles(
        self,
        i: list[int],
        j: list[int],
        k: list[int],
        density: float,
        panel_verts: list[Vec2],
        panel_indices: list[int],
        tri_aniso_ke: list[Vec3] | None = None,
        tri_ka: list[float] | None = None,
        tri_kd: list[float] | None = None,
        tri_drag: list[float] | None = None,
        tri_lift: list[float] | None = None,
    ) -> list[float]:
        """Adds anisotropic triangular FEM elements between groups of three particles in the system.

        Triangles are modeled as viscoelastic elements with elastic stiffness and damping
        Parameters specified on the model.

        Args:
            i: The indices of the first particle
            j: The indices of the second particle
            k: The indices of the third particle
            density: The density per-area of the mesh
            panel_indices: A list of triangle indices, 3 entries per-face
            panel_verts: A list of vertex 2D positions for panel-based cloth simulation.
            tri_aniso_ke: anisotropic stretch stiffness (weft, warp, shear) for pattern-based cloth simulation.

        Note:
            A triangle is created with a rest-length based on the distance
            between the particles in their initial configuration.

        """

        indices = np.array(panel_indices).reshape(-1, 3)

        # compute basis for 2D rest pose
        p = np.array(panel_verts)[indices[:, 0]]
        q = np.array(panel_verts)[indices[:, 1]]
        r = np.array(panel_verts)[indices[:, 2]]

        qp = q - p
        rp = r - p

        def normalized(a):
            l = np.linalg.norm(a, axis=-1, keepdims=True)
            l[l == 0] = 1.0
            return a / l

        D = np.concatenate((qp[..., None], rp[..., None]), axis=-1)

        areas = np.linalg.det(D) / 2.0
        areas[areas < 0.0] = 0.0
        valid_inds = (areas > 0.0).nonzero()[0]
        if len(valid_inds) < len(areas):
            print("inverted or degenerate triangle elements")

        D[areas == 0.0] = np.eye(2)[None, ...]
        inv_D = np.linalg.inv(D)

        inds = np.concatenate((i[valid_inds, None], j[valid_inds, None], k[valid_inds, None]), axis=-1)

        self.tri_indices.extend(inds.tolist())
        self.tri_poses.extend(inv_D[valid_inds].tolist())
        self.tri_activations.extend([0.0] * len(valid_inds))

        def init_if_none(arr, defaultValue):
            if arr is None:
                return [defaultValue] * len(areas)
            return arr

        tri_ke = [self.default_tri_ke] * len(areas)  # init tri_ke for rw safe
        tri_ka = init_if_none(tri_ka, self.default_tri_ka)
        tri_kd = init_if_none(tri_kd, self.default_tri_kd)
        tri_drag = init_if_none(tri_drag, self.default_tri_drag)
        tri_lift = init_if_none(tri_lift, self.default_tri_lift)
        tri_aniso_ke = tri_aniso_ke if tri_aniso_ke is not None else [wp.vec3(self.default_tri_ke)] * len(areas)

        self.tri_materials.extend(
            zip(
                np.array(tri_ke)[valid_inds],
                np.array(tri_ka)[valid_inds],
                np.array(tri_kd)[valid_inds],
                np.array(tri_drag)[valid_inds],
                np.array(tri_lift)[valid_inds],
                strict=False,
            )
        )
        self.tri_aniso_ke.extend(np.array(tri_aniso_ke)[valid_inds])
        areas = areas.tolist()
        self.tri_areas.extend(areas)

        for t in range(len(inds)):
            area = areas[t]
            self.particle_mass[inds[t, 0]] += density * area / 3.0
            self.particle_mass[inds[t, 1]] += density * area / 3.0
            self.particle_mass[inds[t, 2]] += density * area / 3.0

    def add_aniso_edges(
        self,
        i,
        j,
        k,
        l,
        f0,
        f1,
        v0_order,
        v1_order,
        panel_verts: list[Vec2],
        panel_indices: list[int],
        rest: list[float] | None = None,
        edge_aniso_ke: list[Vec3] | None = None,
        edge_kd: list[float] | None = None,
    ) -> None:
        """Adds bending edge elements between two adjacent triangles in the cloth mesh, defined by four vertices.

        The bending energy model follows the discrete shell formulation from [Grinspun et al. 2003].
        The anisotropic bending stiffness is controlled by the `edge_aniso_ke` parameter, and the bending damping by the `edge_kd` parameter.

        Args:
            i: The index of the first particle, i.e., opposite vertex 0
            j: The index of the second particle, i.e., opposite vertex 1
            k: The index of the third particle, i.e., vertex 0
            l: The index of the fourth particle, i.e., vertex 1
            rest: The rest angles across the edges in radians, if not specified they will be computed
            edge_aniso_ke: The anisotropic bending stiffness coefficient
            edge_kd: The bending damping coefficient

        Note:
            The edge lies between the particles indexed by 'k' and 'l' parameters with the opposing
            vertices indexed by 'i' and 'j'. This defines two connected triangles with counterclockwise
            winding: (i, k, l), (j, l, k).

        """

        # prepare panel edge data
        panel_tris = np.array(panel_indices).reshape(-1, 3)
        panel_pos2d = np.array(panel_verts).reshape(-1, 2)
        panel_tris_f0 = panel_tris[f0]
        panel_tris_f1 = panel_tris[f1]

        panel_x1_f0 = panel_pos2d[panel_tris_f0[np.arange(panel_tris_f0.shape[0]), v0_order]]
        panel_x3_f0 = panel_pos2d[panel_tris_f0[np.arange(panel_tris_f0.shape[0]), (v0_order + 1) % 3]]
        panel_x4_f0 = panel_pos2d[panel_tris_f0[np.arange(panel_tris_f0.shape[0]), (v0_order + 2) % 3]]

        panel_x2_f1 = panel_pos2d[panel_tris_f1[np.arange(panel_tris_f1.shape[0]), v1_order]]
        panel_x4_f1 = panel_pos2d[panel_tris_f1[np.arange(panel_tris_f1.shape[0]), (v1_order + 1) % 3]]
        panel_x3_f1 = panel_pos2d[panel_tris_f1[np.arange(panel_tris_f1.shape[0]), (v1_order + 2) % 3]]

        panel_x43_f0 = panel_x4_f0 - panel_x3_f0
        panel_x43_f1 = panel_x4_f1 - panel_x3_f1

        # x1 = np.array(self.particle_q)[i]
        # x2 = np.array(self.particle_q)[j]
        # x3 = np.array(self.particle_q)[k]
        # x4 = np.array(self.particle_q)[l]
        # x43 = x4 - x3

        inds = np.concatenate((i[:, None], j[:, None], k[:, None], l[:, None]), axis=-1)
        self.edge_indices.extend(inds.tolist())

        def dot(a, b):
            return (a * b).sum(axis=-1)

        # we still compute rest angle without panel, maybe used for folding edge in the future
        # Actually rest angle is always 0 on panel
        if rest is None:
            rest = np.zeros_like(i, dtype=float)
            valid_mask = (i != -1) & (j != -1)

            # compute rest angle
            x1_valid = np.array(self.particle_q)[i[valid_mask]]
            x2_valid = np.array(self.particle_q)[j[valid_mask]]
            x3_valid = np.array(self.particle_q)[k[valid_mask]]
            x4_valid = np.array(self.particle_q)[l[valid_mask]]

            def normalized(a):
                l = np.linalg.norm(a, axis=-1, keepdims=True)
                l[l == 0] = 1.0
                return a / l

            n1 = normalized(np.cross(x3_valid - x1_valid, x4_valid - x1_valid))
            n2 = normalized(np.cross(x4_valid - x2_valid, x3_valid - x2_valid))
            e = normalized(x4_valid - x3_valid)

            cos_theta = np.clip(dot(n1, n2), -1.0, 1.0)
            sin_theta = dot(np.cross(n1, n2), e)
            rest[valid_mask] = np.arctan2(sin_theta, cos_theta)

        self.edge_rest_angle.extend(rest.tolist())

        # compute rest length
        mean_edge_length = (np.linalg.norm(panel_x43_f0, axis=1) + np.linalg.norm(panel_x43_f1, axis=1)) * 0.5
        # self.edge_rest_length.extend(np.linalg.norm(x43, axis=1).tolist())
        self.edge_rest_length.extend(mean_edge_length.tolist())

        def init_if_none(arr, defaultValue):
            if arr is None:
                return [defaultValue] * len(i)
            return arr

        edge_ke = [self.default_edge_ke] * len(i)
        edge_kd = init_if_none(edge_kd, self.default_edge_kd)
        # compute final edge_ke based on edge_aniso_ke, ref Feng:2022:LBB
        if edge_aniso_ke is not None:
            angle_f0 = np.atan2(panel_x43_f0[:, 1], panel_x43_f0[:, 0])
            angle_f1 = np.atan2(panel_x43_f1[:, 1], panel_x43_f1[:, 0])
            angle = (angle_f0 + angle_f1) * 0.5
            sin = np.sin(angle)
            cos = np.cos(angle)
            sin2 = np.pow(sin, 2)
            cos2 = np.pow(cos, 2)
            sin12 = np.pow(sin, 12)
            cos12 = np.pow(cos, 12)
            aniso_ke = np.array(edge_aniso_ke).reshape(-1, 3)
            edge_ke = aniso_ke[:, 0] * sin12 + aniso_ke[:, 1] * cos12 + aniso_ke[:, 2] * 4.0 * sin2 * cos2

        self.edge_bending_properties.extend(zip(edge_ke, edge_kd, strict=False))

        # compute edge area
        edge_area = (
            np.abs(np.cross(panel_x43_f0, panel_x1_f0 - panel_x3_f0))
            + np.abs(np.cross(panel_x43_f1, panel_x2_f1 - panel_x3_f1))
            + 1.0e-8
        ) / 3.0
        self.edge_rest_area.extend(edge_area)

        # compute bending cotangents
        def cot2d(a, b, c):
            # compute cotangent of a
            ba = b - a
            ca = c - a
            dot_a = dot(ba, ca)
            cross_a = np.abs(np.cross(ba, ca)) + 1.0e-8
            return dot_a / cross_a

        cot1 = cot2d(panel_x3_f0, panel_x4_f0, panel_x1_f0)
        cot2 = cot2d(panel_x3_f1, panel_x4_f1, panel_x2_f1)
        cot3 = cot2d(panel_x4_f0, panel_x3_f0, panel_x1_f0)
        cot4 = cot2d(panel_x4_f1, panel_x3_f1, panel_x2_f1)
        self.edge_bending_cot.extend(zip(cot1, cot2, cot3, cot4, strict=False))

    def add_aniso_cloth_grid(
        self,
        pos: Vec3,
        rot: Quat,
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
        tri_aniso_ke: Vec3 | None = None,
        tri_ka: float | None = None,
        tri_kd: float | None = None,
        tri_drag: float | None = None,
        tri_lift: float | None = None,
        edge_aniso_ke: Vec3 | None = None,
        edge_kd: float | None = None,
        add_springs: bool = False,
        spring_ke: float | None = None,
        spring_kd: float | None = None,
        particle_radius: float | None = None,
    ):
        """Helper to create a regular planar cloth grid with anisotropic attributes

        Creates a rectangular grid of particles with FEM triangles and bending elements
        automatically.

        Args:
            pos: The position of the cloth in world space
            rot: The orientation of the cloth in world space
            vel: The velocity of the cloth in world space
            dim_x_: The number of rectangular cells along the x-axis
            dim_y: The number of rectangular cells along the y-axis
            cell_x: The width of each cell in the x-direction
            cell_y: The width of each cell in the y-direction
            mass: The mass of each particle
            reverse_winding: Flip the winding of the mesh
            fix_left: Make the left-most edge of particles kinematic (fixed in place)
            fix_right: Make the right-most edge of particles kinematic
            fix_top: Make the top-most edge of particles kinematic
            fix_bottom: Make the bottom-most edge of particles kinematic
        """

        def grid_index(x, y, dim_x):
            return y * dim_x + x

        indices, vertices, panel_verts = [], [], []
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

        start_vertex = len(self.particle_q)

        total_mass = mass * (dim_x + 1) * (dim_x + 1)
        total_area = cell_x * cell_y * dim_x * dim_y
        density = total_mass / total_area

        self.add_aniso_cloth_mesh(
            pos=pos,
            rot=rot,
            scale=1.0,
            vel=vel,
            vertices=vertices,
            indices=indices,
            density=density,
            tri_aniso_ke=tri_aniso_ke,
            tri_ka=tri_ka,
            tri_kd=tri_kd,
            tri_drag=tri_drag,
            tri_lift=tri_lift,
            edge_aniso_ke=edge_aniso_ke,
            edge_kd=edge_kd,
            add_springs=add_springs,
            spring_ke=spring_ke,
            spring_kd=spring_kd,
            particle_radius=particle_radius,
            panel_verts=panel_verts,
        )

        vertex_id = 0
        for y in range(dim_y + 1):
            for x in range(dim_x + 1):
                particle_mass = mass
                particle_flag = ParticleFlags.ACTIVE

                if (
                    (x == 0 and fix_left)
                    or (x == dim_x and fix_right)
                    or (y == 0 and fix_bottom)
                    or (y == dim_y and fix_top)
                ):
                    particle_flag = particle_flag & ~ParticleFlags.ACTIVE
                    particle_mass = 0.0

                self.particle_flags[start_vertex + vertex_id] = particle_flag
                self.particle_mass[start_vertex + vertex_id] = particle_mass
                vertex_id = vertex_id + 1

    def add_aniso_cloth_mesh(
        self,
        pos: Vec3,
        rot: Quat,
        vel: Vec3,
        scale: float,
        density: float,
        indices: list[int],
        vertices: list[Vec3],
        panel_verts: list[Vec2],
        panel_indices: list[int] | None = None,
        tri_aniso_ke: Vec3 | None = None,
        edge_aniso_ke: Vec3 | None = None,
        tri_ka: float | None = None,
        tri_kd: float | None = None,
        tri_drag: float | None = None,
        tri_lift: float | None = None,
        edge_kd: float | None = None,
        add_springs: bool = False,
        spring_ke: float | None = None,
        spring_kd: float | None = None,
        particle_radius: float | None = None,
    ) -> None:
        """Helper to create a cloth model from a regular triangle mesh with anisotropic attributes

        Creates one FEM triangle element and one bending element for every face
        and edge in the input triangle mesh

        Args:
            pos: The position of the cloth in world space
            rot: The orientation of the cloth in world space
            vel: The velocity of the cloth in world space
            vertices: A list of vertex positions
            indices: A list of triangle indices, 3 entries per-face
            density: The density per-area of the mesh
            panel_indices: A list of triangle indices, 3 entries per-face, passes None will use indices as panel_indices
            panel_verts: A list of vertex 2D positions for panel-based cloth simulation.
            particle_radius: The particle_radius which controls particle based collisions.
            tri_aniso_ke: anisotropic stretch stiffness (weft, warp, shear) for panel-based cloth simulation.
            edge_aniso_ke: anisotropic bend stiffness (weft, warp, shear) for panel-based cloth simulation.
        Note:

            The mesh should be two manifold.
        """

        tri_ka = tri_ka if tri_ka is not None else self.default_tri_ka
        tri_kd = tri_kd if tri_kd is not None else self.default_tri_kd
        tri_drag = tri_drag if tri_drag is not None else self.default_tri_drag
        tri_lift = tri_lift if tri_lift is not None else self.default_tri_lift

        edge_kd = edge_kd if edge_kd is not None else self.default_edge_kd
        spring_ke = spring_ke if spring_ke is not None else self.default_spring_ke
        spring_kd = spring_kd if spring_kd is not None else self.default_spring_kd
        particle_radius = particle_radius if particle_radius is not None else self.default_particle_radius
        panel_verts = panel_verts if panel_verts is not None else []

        if panel_indices is None:
            panel_indices = indices

        num_verts = int(len(vertices))
        num_tris = int(len(indices) / 3)

        start_vertex = len(self.particle_q)
        start_tri = len(self.tri_indices)

        # particles
        num_vert = int(len(vertices))
        num_vert2d = int(len(panel_verts))
        use_panel_mode = num_vert == num_vert2d
        if use_panel_mode:
            # use panel_verts to init particles for computing right panel-based anisotropic attributes, will reset with vertices at the end
            verts_2d_np = np.array(panel_verts) * scale
            verts_3d_np = np.hstack([verts_2d_np, np.zeros((len(panel_verts), 1))])
            self.add_particles(
                verts_3d_np.tolist(), [vel] * num_verts, mass=[0.0] * num_verts, radius=[particle_radius] * num_verts
            )
        else:
            vertices_np = np.array(vertices) * scale
            rot_mat_np = np.array(wp.quat_to_matrix(rot), dtype=np.float32).reshape(3, 3)
            verts_3d_np = np.dot(vertices_np, rot_mat_np.T) + pos
            self.add_particles(
                verts_3d_np.tolist(), [vel] * num_verts, mass=[0.0] * num_verts, radius=[particle_radius] * num_verts
            )

        # triangles
        inds = start_vertex + np.array(indices)
        inds = inds.reshape(-1, 3)
        tri_aniso_kes = [tri_aniso_ke] * num_tris if tri_aniso_ke is not None else None
        self.add_aniso_triangles(
            inds[:, 0],
            inds[:, 1],
            inds[:, 2],
            density,
            panel_verts,
            panel_indices,
            tri_aniso_kes,
            [tri_ka] * num_tris,
            [tri_kd] * num_tris,
            [tri_drag] * num_tris,
            [tri_lift] * num_tris,
        )

        end_tri = len(self.tri_indices)

        adj = wp.utils.MeshAdjacency(self.tri_indices[start_tri:end_tri], end_tri - start_tri)

        edge_indices = np.fromiter(
            (x for e in adj.edges.values() for x in (e.o0, e.o1, e.v0, e.v1, e.f0, e.f1)),
            int,
        ).reshape(-1, 6)
        # compute v0 and v1 order in each face
        edge_v0_order = np.argmax(inds[edge_indices[:, 4]][:, :3] == edge_indices[:, 0][:, None], axis=1)
        edge_v1_order = np.argmax(inds[edge_indices[:, 5]][:, :3] == edge_indices[:, 1][:, None], axis=1)

        edge_aniso_kes = [edge_aniso_ke] * len(edge_indices) if edge_aniso_ke is not None else None
        self.add_aniso_edges(
            edge_indices[:, 0],
            edge_indices[:, 1],
            edge_indices[:, 2],
            edge_indices[:, 3],
            edge_indices[:, 4],
            edge_indices[:, 5],
            edge_v0_order,
            edge_v1_order,
            panel_verts,
            panel_indices,
            edge_aniso_ke=edge_aniso_kes,
            edge_kd=[edge_kd] * len(edge_indices),
        )

        if add_springs:
            spring_indices = set()
            for i, j, k, l in edge_indices:
                spring_indices.add((min(k, l), max(k, l)))
                if i != -1:
                    spring_indices.add((min(i, k), max(i, k)))
                    spring_indices.add((min(i, l), max(i, l)))
                if j != -1:
                    spring_indices.add((min(j, k), max(j, k)))
                    spring_indices.add((min(j, l), max(j, l)))
                if i != -1 and j != -1:
                    spring_indices.add((min(i, j), max(i, j)))

            for i, j in spring_indices:
                self.add_spring(i, j, spring_ke, spring_kd, control=0.0)

        if use_panel_mode:
            # reset particle with vertices
            vertices_np = np.array(vertices) * scale
            rot_mat = np.array(wp.quat_to_matrix(rot), dtype=np.float32).reshape(3, 3)
            verts_3d_np = np.dot(vertices_np, rot_mat.T) + pos
            self.particle_q[start_vertex : start_vertex + num_vert] = verts_3d_np.tolist()

    def sew_close_vertices(
        self,
        sew_distance=1.0e-3,
        sew_interior=False,
    ):
        """
        Sew close vertices by creating sew springs.

        Args:
            sew_distance: Vertices within sew_distance will be connected by springs.
            sew_interior: If True, can sew between interior vertices, otherwise only sew boundary-interior or boundary-boundary vertices

        """
        sew_springs = create_trimesh_sew_springs(
            self.particle_q,
            self.edge_indices,
            sew_distance,
            sew_interior,
        )
        for spring in sew_springs:
            self.add_spring(spring[0], spring[1], self.default_spring_ke, self.default_spring_kd, control=0.0)

    def finalize(self, device: Devicelike | None = None, requires_grad: bool = False) -> Style3DModel:
        """Convert this builder object to a concrete model for simulation.

        After building simulation elements this method should be called to transfer
        all data to device memory ready for simulation.

        Args:
            device: The simulation device to use, e.g.: 'cpu', 'cuda'
            requires_grad: Whether to enable gradient computation for the model

        Returns:

            A model object.
        """

        model = super().finalize(device=device, requires_grad=requires_grad)
        style3d_model = Style3DModel.from_model(model)

        with wp.ScopedDevice(device):
            style3d_model.tri_aniso_ke = wp.array(self.tri_aniso_ke, dtype=wp.vec3, requires_grad=requires_grad)
            style3d_model.edge_rest_area = wp.array(self.edge_rest_area, dtype=wp.float32, requires_grad=requires_grad)
            style3d_model.edge_bending_cot = wp.array(self.edge_bending_cot, dtype=wp.vec4, requires_grad=requires_grad)

        return style3d_model
