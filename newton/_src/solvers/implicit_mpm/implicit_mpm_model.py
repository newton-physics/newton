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

"""Implicit MPM model."""

import math
from dataclasses import dataclass

import numpy as np
import warp as wp

import newton

from .rasterized_collisions import Collider

__all__ = ["ImplicitMPMModel", "ImplicitMPMOptions"]

_INFINITY = wp.constant(1.0e12)
"""Value above which quantities are considered infinite"""

_EPSILON = wp.constant(1.0 / _INFINITY)
"""Value below which quantities are considered zero"""

_DEFAULT_PROJECTION_THRESHOLD = 0.01
"""Default threshold for projection outside of collider, as a fraction of the voxel size"""

_DEFAULT_THICKNESS = 0.01
"""Default thickness for colliders, as a fraction of the voxel size"""
_DEFAULT_FRICTION = 0.5
"""Default friction coefficient for colliders"""
_DEFAULT_ADHESION = 0.0
"""Default adhesion coefficient for colliders (Pa)"""


@dataclass
class ImplicitMPMOptions:
    """Implicit MPM solver options."""

    # numerics
    max_iterations: int = 250
    """Maximum number of iterations for the rheology solver."""
    tolerance: float = 1.0e-5
    """Tolerance for the rheology solver."""
    strain_basis: str = "P0"
    """Strain basis functions. May be one of P0, Q1"""
    solver: str = "gauss-seidel"
    """Solver to use for the rheology solver. May be one of gauss-seidel, jacobi."""
    warmstart_mode: str = "auto"
    """Warmstart mode to use for the rheology solver. May be one of none, auto, particles, grid."""
    collider_velocity_mode: str = "instantaneous"
    """Collider velocity computation mode. May be one of instantaneous, finite_difference."""

    # grid
    voxel_size: float = 0.1
    """Size of the grid voxels."""
    grid_type: str = "sparse"
    """Type of grid to use. May be one of sparse, dense, fixed."""
    grid_padding: int = 0
    """Number of empty cells to add around particles when allocating the grid."""
    max_active_cell_count: int = -1
    """Maximum number of active cells to use for active subsets of dense grids. -1 means unlimited."""
    transfer_scheme: str = "apic"
    """Transfer scheme to use for particle-grid transfers. May be one of apic, pic."""

    # material / background
    critical_fraction: float = 0.0
    """Fraction for particles under which the yield surface collapses."""
    air_drag: float = 1.0
    """Numerical drag for the background air."""

    # experimental
    collider_normal_from_sdf_gradient: bool = False
    """Compute collider normals from sdf gradient rather than closest point"""
    collider_basis: str = "Q1"
    """Collider basis function string. Examples: P0 (piecewise constant), Q1 (trilinear), S2 (quadratic serendipity), pic8 (particle-based with max 8 points per cell)"""


def _particle_parameter(
    num_particles, model_value: float | wp.array | None = None, default_value=None, model_scale: wp.array | None = None
):
    """Helper function to create a particle-wise parameter array, taking defaults either from the model
    or the global options."""

    if model_value is None:
        return wp.full(num_particles, default_value, dtype=float)
    elif isinstance(model_value, wp.array):
        if model_value.shape[0] != num_particles:
            raise ValueError(f"Model value array must have {num_particles} elements")

        return model_value if model_scale is None else model_value * model_scale
    else:
        return wp.full(num_particles, model_value, dtype=float) if model_scale is None else model_value * model_scale


def _merge_meshes(
    points: list[np.array] = (),
    indices: list[np.array] = (),
    shape_ids: np.array = (),
    material_ids: np.array = (),
) -> tuple[wp.array, wp.array, wp.array, np.array]:
    """Merges the points and indices of several meshes into a single one"""

    pt_count = np.array([len(pts) for pts in points])
    face_count = np.array([len(idx) // 3 for idx in indices])
    offsets = np.cumsum(pt_count) - pt_count

    merged_points = np.vstack([pts[:, :3] for pts in points])
    merged_indices = np.concatenate([idx + offsets[k] for k, idx in enumerate(indices)])
    vertex_shape_ids = np.repeat(np.arange(len(points), dtype=int), repeats=pt_count)
    face_shape_ids = np.repeat(np.arange(len(points), dtype=int), repeats=face_count)

    return (
        wp.array(merged_points, dtype=wp.vec3),
        wp.array(merged_indices, dtype=int),
        wp.array(shape_ids[vertex_shape_ids], dtype=int),
        np.array(material_ids, dtype=int)[face_shape_ids],
    )


def _get_shape_mesh(model: newton.Model, shape_id: int, geo_type: newton.GeoType, geo_scale: wp.vec3):
    """Get a shape mesh from a model."""

    if geo_type == newton.GeoType.MESH:
        src_mesh = model.shape_source[shape_id]
        vertices = src_mesh.vertices * np.array(geo_scale)
        indices = src_mesh.indices
        return vertices, indices
    if geo_type == newton.GeoType.PLANE:
        # Handle "infinite" planes encoded with non-positive scales
        width = geo_scale[0] if len(geo_scale) > 0 and geo_scale[0] > 0.0 else 1000.0
        length = geo_scale[1] if len(geo_scale) > 1 and geo_scale[1] > 0.0 else 1000.0
        mesh = newton.Mesh.create_plane(
            width,
            length,
            compute_normals=False,
            compute_uvs=False,
            compute_inertia=False,
        )
        return mesh.vertices, mesh.indices
    elif geo_type == newton.GeoType.SPHERE:
        radius = geo_scale[0]
        mesh = newton.Mesh.create_sphere(
            radius,
            compute_normals=False,
            compute_uvs=False,
            compute_inertia=False,
        )
        return mesh.vertices, mesh.indices

    elif geo_type == newton.GeoType.CAPSULE:
        radius, half_height = geo_scale[:2]
        mesh = newton.Mesh.create_capsule(
            radius,
            half_height,
            up_axis=2,
            compute_normals=False,
            compute_uvs=False,
            compute_inertia=False,
        )
        return mesh.vertices, mesh.indices

    elif geo_type == newton.GeoType.CYLINDER:
        radius, half_height = geo_scale[:2]
        mesh = newton.Mesh.create_cylinder(
            radius,
            half_height,
            up_axis=2,
            compute_normals=False,
            compute_uvs=False,
            compute_inertia=False,
        )
        return mesh.vertices, mesh.indices

    elif geo_type == newton.GeoType.CONE:
        radius, half_height = geo_scale[:2]
        mesh = newton.Mesh.create_cone(
            radius,
            half_height,
            up_axis=2,
            compute_normals=False,
            compute_uvs=False,
            compute_inertia=False,
        )
        return mesh.vertices, mesh.indices

    elif geo_type == newton.GeoType.BOX:
        if len(geo_scale) == 1:
            ext = (geo_scale[0],) * 3
        else:
            ext = tuple(geo_scale[:3])
        mesh = newton.Mesh.create_box(
            ext[0],
            ext[1],
            ext[2],
            duplicate_vertices=False,
            compute_normals=False,
            compute_uvs=False,
            compute_inertia=False,
        )
        return mesh.vertices, mesh.indices

    raise NotImplementedError(f"Shape type {geo_type} not supported")


@wp.kernel
def _apply_shape_transforms(
    points: wp.array(dtype=wp.vec3), shape_ids: wp.array(dtype=int), shape_transforms: wp.array(dtype=wp.transform)
):
    v = wp.tid()
    p = points[v]
    shape_id = shape_ids[v]
    shape_transform = shape_transforms[shape_id]
    p = wp.transform_point(shape_transform, p)
    points[v] = p


def _get_body_collision_shapes(model: newton.Model, body_index: int):
    """Returns the ids of the shapes of a body with active collision flags."""

    shape_flags = model.shape_flags.numpy()
    body_shape_ids = np.array(model.body_shapes[body_index], dtype=int)

    return body_shape_ids[(shape_flags[body_shape_ids] & newton.ShapeFlags.COLLIDE_PARTICLES) > 0]


def _get_shape_collision_materials(model: newton.Model, shape_ids: list[int]):
    """Returns the collision materials from the model for a list of shapes"""
    thicknesses = model.shape_thickness.numpy()[shape_ids]
    friction = model.shape_material_mu.numpy()[shape_ids]

    return thicknesses, friction


def _create_body_collider_mesh(
    model: newton.Model,
    shape_ids: list[int],
    material_ids: list[int],
):
    """Create a collider mesh from a body."""

    shape_scale = model.shape_scale.numpy()
    shape_type = model.shape_type.numpy()

    shape_meshes = [_get_shape_mesh(model, sid, newton.GeoType(shape_type[sid]), shape_scale[sid]) for sid in shape_ids]

    collider_points, collider_indices, vertex_shape_ids, face_material_ids = _merge_meshes(
        *zip(*shape_meshes, strict=True),
        shape_ids=shape_ids,
        material_ids=material_ids,
    )

    wp.launch(
        _apply_shape_transforms,
        dim=collider_points.shape[0],
        inputs=[
            collider_points,
            vertex_shape_ids,
            model.shape_transform,
        ],
    )

    return wp.Mesh(collider_points, collider_indices, wp.zeros_like(collider_points)), face_material_ids


class ImplicitMPMModel:
    """Wrapper augmenting a ``newton.Model`` with implicit MPM data and setup.

    Holds particle material parameters, collider parameters, and convenience
    arrays derived from the wrapped ``model`` and ``ImplicitMPMOptions``. The
    instance is consumed by ``SolverImplicitMPM`` during time stepping.

    Args:
        model: The base Newton model to augment.
        options: Options controlling particle and collider defaults.
    """

    def __init__(self, model: newton.Model, options: ImplicitMPMOptions):
        self.model = model
        self._options = options

        # Global options from ImplicitMPMOptions
        self.voxel_size = float(options.voxel_size)
        """Size of the grid voxels"""

        self.critical_fraction = float(options.critical_fraction)
        """Maximum fraction of the grid volume that can be occupied by particles"""

        self.air_drag = float(options.air_drag)
        """Drag for the background air"""

        self.collider = Collider()
        """Collider struct"""

        self.collider_velocity_mode = options.collider_velocity_mode
        """Collider velocity computation mode (instantaneous or finite_difference)"""

        self.collider_body_mass = None
        self.collider_body_inv_inertia = None

        self.setup_particle_material()
        self.setup_collider()

    def notify_particle_material_changed(self):
        """Refresh cached extrema for material parameters.

        Tracks the minimum Young's modulus and maximum hardening across
        particles to quickly toggle code paths (e.g., compliant particles or
        hardening enabled) without recomputing per step.
        """
        model = self.model
        mpm_ns = getattr(model, "mpm", None)

        if mpm_ns is not None and hasattr(mpm_ns, "young_modulus"):
            self.min_young_modulus = float(np.min(mpm_ns.young_modulus.numpy()))
        else:
            self.min_young_modulus = _INFINITY

        if mpm_ns is not None and hasattr(mpm_ns, "hardening"):
            self.max_hardening = float(np.max(mpm_ns.hardening.numpy()))
        else:
            self.max_hardening = 0.0

    def notify_collider_changed(self):
        """Refresh cached extrema for collider parameters.

        Tracks the minimum collider mass to determine whether compliant
        colliders are present and to enable/disable related computations.
        """
        body_ids = self.collider.collider_body_index.numpy()
        body_mass = self.collider_body_mass.numpy()
        dynamic_body_ids = body_ids[body_ids >= 0]
        dynamic_body_ids = dynamic_body_ids[body_mass[dynamic_body_ids] > 0.0]
        dynamic_body_masses = body_mass[dynamic_body_ids]

        self.min_collider_mass = np.min(dynamic_body_masses, initial=np.inf)
        self.collider.query_max_dist = self.voxel_size * math.sqrt(3.0)
        self.collider_body_count = int(np.max(body_ids + 1, initial=0))

    def setup_particle_material(self):
        """Initialize derived per-particle fields from the model.

        Computes particle volumes and densities from the model's particle mass and radius.
        Also caches extrema used by the solver for fast feature toggles.

        Per-particle material parameters are read directly from the ``model.mpm.*`` namespace
        (registered via :meth:`SolverImplicitMPM.register_custom_attributes`).
        """
        model = self.model

        num_particles = model.particle_q.shape[0]

        with wp.ScopedDevice(model.device):
            # Assume that particles represent a cuboid volume of space
            # (they are typically laid out on a grid)
            self.particle_radius = _particle_parameter(num_particles, model.particle_radius)
            self.particle_volume = wp.array(8.0 * self.particle_radius.numpy() ** 3)
            self.particle_density = model.particle_mass / self.particle_volume

        self.notify_particle_material_changed()

    def setup_collider(
        self,
        collider_meshes: list[wp.Mesh] | None = None,
        collider_body_ids: list[int] | None = None,
        collider_thicknesses: list[float] | None = None,
        collider_friction: list[float] | None = None,
        collider_adhesion: list[float] | None = None,
        collider_projection_threshold: list[float] | None = None,
        model: newton.Model | None = None,
        body_com: wp.array | None = None,
        body_mass: wp.array | None = None,
        body_inv_inertia: wp.array | None = None,
        body_q: wp.array | None = None,
    ):
        """Initialize collider parameters and defaults from inputs.

        Populates the ``Collider`` struct with meshes, body mapping, and per-material
        properties (thickness, friction, adhesion, projection threshold).

        By default, this will setup collisions against all collision shapes in the model with flag `newton.ShapeFlag.COLLIDE_PARTICLES`.
        Rigid body colliders will be treated as kinematic if their mass is zero; for all model bodies to be treated as kinematic,
        pass ``body_mass=wp.zeros_like(model.body_mass)``.

        For any collider index `i`, only one of ``collider_meshes[i]`` and ``collider_body_ids`` may not be `None`.
        If material properties are not provided for a collider, but a body index is provided,
        the material will be read from the body shape material attributes on the model.

        Args:
            collider_meshes: Warp triangular meshes used as colliders.
            collider_body_ids: For dynamic colliders, per-mesh body ids.
            collider_thicknesses: Per-mesh signed distance offsets (m).
            collider_friction: Per-mesh Coulomb friction coefficients.
            collider_adhesion: Per-mesh adhesion (Pa).
            collider_projection_threshold: Per-mesh projection threshold, i.e. how far below the surface the
              particle may be before it is projected out. (m)
            model: The model to read collider properties from. Default to self.model.
            body_com: For dynamic colliders, per-body center of mass. Default to model.body_com.
            body_mass: For dynamic colliders, per-body mass. Default to model.body_mass.
            body_inv_inertia: For dynamic colliders, per-body inverse inertia. Default to model.body_inv_inertia.
            body_q: For dynamic colliders, per-body initial transform. Default to model.body_q.
        """

        if model is None:
            model = self.model

        if collider_body_ids is None:
            if collider_meshes is None:
                collider_body_ids = [
                    body_id
                    for body_id in range(-1, model.body_count)
                    if len(_get_body_collision_shapes(model, body_id)) > 0
                ]
            else:
                collider_body_ids = [None] * len(collider_meshes)
        if collider_meshes is None:
            collider_meshes = [None] * len(collider_body_ids)

        for collider_id, (mesh, body_id) in enumerate(zip(collider_meshes, collider_body_ids, strict=True)):
            if mesh is None:
                if body_id is None:
                    raise ValueError(
                        f"Either a mesh or a body_id must be provided for each collider; collider {collider_id} is missing both"
                    )
            elif body_id is not None:
                raise ValueError(
                    f"Either a mesh or a body_id must be provided for each collider; collider {collider_id} provides both"
                )

        collider_count = len(collider_body_ids)

        if collider_thicknesses is None:
            collider_thicknesses = [None] * collider_count
        if collider_projection_threshold is None:
            collider_projection_threshold = [None] * collider_count
        if collider_friction is None:
            collider_friction = [None] * collider_count
        if collider_adhesion is None:
            collider_adhesion = [None] * collider_count

        assert len(collider_body_ids) == len(collider_thicknesses)
        assert len(collider_body_ids) == len(collider_projection_threshold)
        assert len(collider_body_ids) == len(collider_friction)
        assert len(collider_body_ids) == len(collider_adhesion)

        if body_com is None:
            body_com = model.body_com
        if body_mass is None:
            body_mass = model.body_mass
        if body_inv_inertia is None:
            body_inv_inertia = model.body_inv_inertia
        if body_q is None:
            body_q = model.body_q

        # count materials and shapes
        material_count = 1  # default material
        body_shapes = {}
        collider_material_ids = []
        for body_id in collider_body_ids:
            if body_id is not None:
                shapes = _get_body_collision_shapes(model, body_id)
                if len(shapes) == 0:
                    raise ValueError(f"Body {body_id} has no collision shapes")

                body_shapes[body_id] = shapes
                collider_material_ids.append(list(range(material_count, material_count + len(shapes))))
                material_count += len(shapes)
            else:
                collider_material_ids.append([material_count])
                material_count += 1

        # assign material values
        material_thickness = [_DEFAULT_THICKNESS * self.voxel_size] * material_count
        material_friction = [_DEFAULT_FRICTION] * material_count
        material_adhesion = [_DEFAULT_ADHESION] * material_count
        material_projection_threshold = [_DEFAULT_PROJECTION_THRESHOLD * self.voxel_size] * material_count

        def assign_material(
            material_id: int,
            thickness: float | None = None,
            friction: float | None = None,
            adhesion: float | None = None,
            projection_threshold: float | None = None,
        ):
            if thickness is not None:
                material_thickness[material_id] = thickness
            if friction is not None:
                material_friction[material_id] = friction
            if adhesion is not None:
                material_adhesion[material_id] = adhesion
            if projection_threshold is not None:
                material_projection_threshold[material_id] = projection_threshold

        def assign_collider_material(material_id: int, collider_id: int):
            assign_material(
                material_id,
                collider_thicknesses[collider_id],
                collider_friction[collider_id],
                collider_adhesion[collider_id],
                collider_projection_threshold[collider_id],
            )

        for collider_id, body_id in enumerate(collider_body_ids):
            if body_id is not None:
                for material_id, shape_thickness, shape_friction in zip(
                    collider_material_ids[collider_id],
                    *_get_shape_collision_materials(model, body_shapes[body_id]),
                    strict=True,
                ):
                    # use material from shapes as default
                    assign_material(material_id, thickness=shape_thickness, friction=shape_friction)
                    # override with user-provided material
                    assign_collider_material(material_id, collider_id)
            else:
                # user-provided collider, single material
                assign_collider_material(collider_material_ids[collider_id][0], collider_id)

        collider_max_thickness = [
            max((material_thickness[material_id] for material_id in collider_material_ids[collider_id]), default=0.0)
            for collider_id in range(collider_count)
        ]

        # Create device arrays
        with wp.ScopedDevice(self.model.device):
            # Create collider meshes from bodies if necessary
            face_material_ids = [[]]
            for collider_id in range(collider_count):
                body_index = collider_body_ids[collider_id]

                if body_index is None:
                    # Set body index to -1 to indicate a static collider
                    # This may not correspond to the model's body -1, but as far as the collision kernels
                    # are concerned, it does not matter.

                    collider_body_ids[collider_id] = -1
                    material_id = collider_material_ids[collider_id][0]
                    face_count = collider_meshes[collider_id].indices.shape[0] // 3
                    mesh_face_material_ids = np.full(face_count, material_id, dtype=int)
                else:
                    collider_meshes[collider_id], mesh_face_material_ids = _create_body_collider_mesh(
                        model, body_shapes[body_index], collider_material_ids[collider_id]
                    )

                face_material_ids.append(mesh_face_material_ids)

            self.collider.collider_body_index = wp.array(collider_body_ids, dtype=int)
            self.collider.collider_mesh = wp.array([collider.id for collider in collider_meshes], dtype=wp.uint64)
            self.collider.collider_max_thickness = wp.array(collider_max_thickness, dtype=float)

            self.collider.face_material_index = wp.array(np.concatenate(face_material_ids), dtype=int)

            self.collider.material_thickness = wp.array(material_thickness, dtype=float)
            self.collider.material_friction = wp.array(material_friction, dtype=float)
            self.collider.material_adhesion = wp.array(material_adhesion, dtype=float)
            self.collider.material_projection_threshold = wp.array(material_projection_threshold, dtype=float)

        self.collider.body_com = body_com
        self.collider_body_mass = body_mass
        self.collider_body_inv_inertia = body_inv_inertia
        self.collider_body_q = body_q
        self._collider_meshes = collider_meshes  # Keep a ref so that meshes are not garbage collected

        # Toggle finite-difference collider velocities based on model setting
        self.collider.use_finite_difference_velocity = self.collider_velocity_mode == "finite_difference"

        self.notify_collider_changed()

    @property
    def has_compliant_particles(self):
        return self.min_young_modulus < _INFINITY

    @property
    def has_hardening(self):
        return self.max_hardening > 0.0

    @property
    def has_compliant_colliders(self):
        return self.min_collider_mass < _INFINITY
