# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Implicit MPM model."""

from __future__ import annotations

import math
import warnings
from typing import TYPE_CHECKING

import numpy as np
import warp as wp

import newton

from .rasterized_collisions import Collider

__all__ = ["ImplicitMPMModel"]

if TYPE_CHECKING:
    from .solver_implicit_mpm import SolverImplicitMPM

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


def _reuse_or_allocate(arr: wp.array | None, num_particles: int, dtype=float) -> wp.array:
    """Return ``arr`` if it is already sized for ``num_particles``, else allocate a fresh buffer."""
    if arr is not None and arr.shape == (num_particles,):
        return arr
    return wp.empty(num_particles, dtype=dtype)


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
        face_shape_ids,
    )


def _get_shape_mesh(model: newton.Model, shape_id: int, geo_type: newton.GeoType, geo_scale: wp.vec3) -> newton.Mesh:
    """Get a shape mesh from a model."""

    if geo_type == newton.GeoType.MESH:
        src_mesh = model.shape_source[shape_id]
        vertices = src_mesh.vertices * np.array(geo_scale)
        indices = src_mesh.indices
        return newton.Mesh(vertices, indices, compute_inertia=False)
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
        return mesh
    elif geo_type == newton.GeoType.SPHERE:
        radius = geo_scale[0]
        mesh = newton.Mesh.create_sphere(
            radius,
            compute_normals=False,
            compute_uvs=False,
            compute_inertia=False,
        )
        return mesh

    elif geo_type == newton.GeoType.CAPSULE:
        radius, half_height = geo_scale[:2]
        mesh = newton.Mesh.create_capsule(
            radius,
            half_height,
            up_axis=newton.Axis.Z,
            compute_normals=False,
            compute_uvs=False,
            compute_inertia=False,
        )
        return mesh

    elif geo_type == newton.GeoType.CYLINDER:
        radius, half_height = geo_scale[:2]
        mesh = newton.Mesh.create_cylinder(
            radius,
            half_height,
            up_axis=newton.Axis.Z,
            barrel_radius=geo_scale[2],
            compute_normals=False,
            compute_uvs=False,
            compute_inertia=False,
        )
        return mesh

    elif geo_type == newton.GeoType.CONE:
        radius, half_height = geo_scale[:2]
        mesh = newton.Mesh.create_cone(
            radius,
            half_height,
            up_axis=newton.Axis.Z,
            compute_normals=False,
            compute_uvs=False,
            compute_inertia=False,
        )
        return mesh

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
        return mesh

    raise NotImplementedError(f"Shape type {geo_type} not supported")


@wp.kernel
def _apply_shape_transforms(
    points: wp.array[wp.vec3], shape_ids: wp.array[int], shape_transforms: wp.array[wp.transform]
):
    v = wp.tid()
    p = points[v]
    shape_id = shape_ids[v]
    shape_transform = shape_transforms[shape_id]
    p = wp.transform_point(shape_transform, p)
    points[v] = p


@wp.kernel
def _compute_particle_volume_density(
    particle_radius: wp.array[float],
    particle_mass: wp.array[float],
    particle_volume: wp.array[float],
    particle_density: wp.array[float],
):
    i = wp.tid()
    r = particle_radius[i]
    v = 8.0 * r * r * r
    particle_volume[i] = v
    particle_density[i] = particle_mass[i] / v


@wp.kernel
def _refresh_particle_masks(
    model_flags: wp.array[wp.int32],
    is_collider_particle: wp.array[wp.int32],
    particle_volume: wp.array[float],
    transfer_flags: wp.array[wp.int32],
    material_flags: wp.array[wp.int32],
    material_particle_volume: wp.array[float],
):
    """Derive the transfer and material masks from ``model.particle_flags``.

    Collider particles are never active for either transfer or material;
    proxy particles transfer momentum but carry no material.
    """
    i = wp.tid()
    f = model_flags[i]
    if is_collider_particle[i] != 0:
        f = f & ~newton.ParticleFlags.ACTIVE
    transfer_flags[i] = f
    if f & newton.ParticleFlags.PROXY:
        f = f & ~newton.ParticleFlags.ACTIVE
    material_flags[i] = f
    if f & newton.ParticleFlags.ACTIVE:
        material_particle_volume[i] = particle_volume[i]
    else:
        material_particle_volume[i] = 0.0


_EXTREMA_MIN_YOUNG_MODULUS = 0
_EXTREMA_MAX_HARDENING = 1
_EXTREMA_MAX_VISCOSITY = 2
_EXTREMA_MAX_DILATANCY = 3
_EXTREMA_COUNT = 4


@wp.kernel
def _reset_material_extrema(extrema: wp.array[float]):
    extrema[_EXTREMA_MIN_YOUNG_MODULUS] = wp.inf
    extrema[_EXTREMA_MAX_HARDENING] = 0.0
    extrema[_EXTREMA_MAX_VISCOSITY] = 0.0
    extrema[_EXTREMA_MAX_DILATANCY] = 0.0


@wp.kernel
def _reduce_material_extrema(
    is_collider_particle: wp.array[wp.int32],
    young_modulus: wp.array[float],
    hardening: wp.array[float],
    viscosity: wp.array[float],
    dilatancy: wp.array[float],
    extrema: wp.array[float],
):
    """Reduce the material extrema over the particles that can carry material.

    Each atomic is guarded by a plain read of the current extremum, so most
    particles skip it and the contention stays low.
    """
    i = wp.tid()
    if is_collider_particle[i] != 0:
        return
    if young_modulus[i] < extrema[_EXTREMA_MIN_YOUNG_MODULUS]:
        wp.atomic_min(extrema, _EXTREMA_MIN_YOUNG_MODULUS, young_modulus[i])
    if hardening[i] > extrema[_EXTREMA_MAX_HARDENING]:
        wp.atomic_max(extrema, _EXTREMA_MAX_HARDENING, hardening[i])
    if viscosity[i] > extrema[_EXTREMA_MAX_VISCOSITY]:
        wp.atomic_max(extrema, _EXTREMA_MAX_VISCOSITY, viscosity[i])
    if dilatancy[i] > extrema[_EXTREMA_MAX_DILATANCY]:
        wp.atomic_max(extrema, _EXTREMA_MAX_DILATANCY, dilatancy[i])


def _get_body_collision_shapes(model: newton.Model, body_index: int):
    """Returns the ids of the shapes of a body with active collision flags."""

    shape_flags = model.shape_flags.numpy()
    body_shape_ids = np.array(model.body_shapes[body_index], dtype=int)

    return body_shape_ids[(shape_flags[body_shape_ids] & newton.ShapeFlags.COLLIDE_PARTICLES) > 0]


def _get_shape_collision_materials(model: newton.Model, shape_ids: list[int]):
    """Returns the collision materials from the model for a list of shapes"""
    thicknesses = model.shape_margin.numpy()[shape_ids]
    friction = model.shape_material_mu.numpy()[shape_ids]

    # Adhesion and projection threshold are optional ``mpm:*`` shape attributes
    mpm_attributes = getattr(model, "mpm", None)
    shape_adhesion = getattr(mpm_attributes, "collider_adhesion", None)
    if shape_adhesion is None:
        adhesion = np.full(len(shape_ids), _DEFAULT_ADHESION)
    else:
        adhesion = shape_adhesion.numpy()[shape_ids]

    # A negative value selects the solver default, which scales with the voxel size
    shape_projection_threshold = getattr(mpm_attributes, "collider_projection_threshold", None)
    if shape_projection_threshold is None:
        projection_threshold = np.full(len(shape_ids), -1.0)
    else:
        projection_threshold = shape_projection_threshold.numpy()[shape_ids]

    return thicknesses, friction, adhesion, projection_threshold


def _create_body_collider_mesh(
    model: newton.Model,
    shape_ids: list[int],
):
    """Create a collider mesh from a body."""

    shape_scale = model.shape_scale.numpy()
    shape_type = model.shape_type.numpy()

    shape_meshes = [_get_shape_mesh(model, sid, newton.GeoType(shape_type[sid]), shape_scale[sid]) for sid in shape_ids]

    collider_points, collider_indices, vertex_shape_ids, face_shape_ids = _merge_meshes(
        points=[mesh.vertices for mesh in shape_meshes],
        indices=[mesh.indices for mesh in shape_meshes],
        shape_ids=shape_ids,
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

    return wp.Mesh(collider_points, collider_indices, wp.zeros_like(collider_points)), face_shape_ids


@wp.struct
class MaterialParameters:
    """Convenience struct for passing material parameters to kernels."""

    young_modulus: wp.array[float]
    """Young's modulus for the material."""
    poisson_ratio: wp.array[float]
    """Poisson's ratio for the material."""
    damping: wp.array[float]
    """Damping for the material."""

    friction: wp.array[float]
    """Friction for the material."""
    yield_pressure: wp.array[float]
    """Yield pressure for the material."""
    tensile_yield_ratio: wp.array[float]
    """Tensile yield ratio for the material."""
    yield_stress: wp.array[float]
    """Yield stress for the material."""
    viscosity: wp.array[float]
    """Viscosity for the material."""

    hardening: wp.array[float]
    """Hardening for the material."""
    hardening_rate: wp.array[float]
    """Hardening rate for the material."""
    softening_rate: wp.array[float]
    """Softening rate for the material."""
    dilatancy: wp.array[float]
    """Dilatancy for the material."""


class ImplicitMPMModel:
    """Wrapper augmenting a ``newton.Model`` with implicit MPM data and setup.

    Holds particle material parameters, collider parameters, and convenience
    arrays derived from the wrapped ``model`` and ``SolverImplicitMPM.Config``.
    Consumed by ``SolverImplicitMPM`` during time stepping.

    Args:
        model: The base Newton model to augment.
        options: Options controlling particle and collider defaults.
    """

    def __init__(self, model: newton.Model, options: SolverImplicitMPM.Config):
        self.model = model
        self._options = options

        # Global options from SolverImplicitMPM.Config
        self.voxel_size = float(options.voxel_size)
        """Size of the grid voxels"""

        self.critical_fraction = float(options.critical_fraction)
        """Maximum fraction of the grid volume that can be occupied by particles"""

        self.air_drag = float(options.air_drag)
        """Drag for the background air"""

        self._separate_worlds = bool(options.separate_worlds and model.world_count > 1)
        """Whether colliders and particles are isolated by Newton world."""

        self.collider = Collider()
        """Collider struct"""

        self.material_parameters = MaterialParameters()
        """Material parameters struct"""

        self.collider_body_mass = None
        self.collider_body_inv_inertia = None
        self.collider_body_q = None
        self.deformable_collider_vertex_ranges: list[tuple[int, int, int]] = []

        self.min_young_modulus = math.inf
        """Smallest Young's modulus [Pa] over material-capable particles, as last read back on the host."""
        self.max_hardening = 0.0
        """Largest hardening factor over material-capable particles, as last read back on the host."""
        self.has_viscosity = False
        """Whether any material-capable particle has viscosity, as last read back on the host."""
        self.has_dilatancy = False
        """Whether any material-capable particle has dilatancy, as last read back on the host."""

        # Host copy of the device extrema is out of date (set by a notification).
        self._extrema_host_stale = False
        # A notification ran inside a CUDA graph capture, so graph replays may
        # change the device extrema without the host knowing.
        self._extrema_refreshed_in_graph = False
        # Code-path switches in effect when ``step()`` was last captured.
        self._captured_code_paths: tuple[bool, ...] | None = None

        self.notify_particle_material_changed()
        self.setup_collider()

    def notify_particle_material_changed(self):
        """Refresh the solver data derived from the particle properties.

        Called once during ``__init__`` and whenever particle flags, masses,
        radii, or any ``model.mpm.*`` material array change. The particle
        count must not change after construction. Binds references from the
        ``model.mpm.*`` namespace (registered via
        :meth:`SolverImplicitMPM.register_custom_attributes`) into
        ``self.material_parameters``, then recomputes on the device:

        - ``particle_volume`` and ``particle_density``, from
          ``model.particle_radius`` and ``model.particle_mass``.
        - The transfer and material particle masks (see :meth:`refresh_particle_flags`).
        - The material extrema that select solver code paths (see
          :meth:`sync_material_extrema`).

        After the first call, this only launches kernels that write existing
        arrays, so it may be recorded in a CUDA graph. Outside a graph capture
        it also reads the extrema back to the host.
        """
        model = self.model

        self.material_parameters.young_modulus = model.mpm.young_modulus
        self.material_parameters.poisson_ratio = model.mpm.poisson_ratio
        self.material_parameters.damping = model.mpm.damping
        self.material_parameters.friction = model.mpm.friction
        self.material_parameters.yield_pressure = model.mpm.yield_pressure
        self.material_parameters.tensile_yield_ratio = model.mpm.tensile_yield_ratio
        self.material_parameters.yield_stress = model.mpm.yield_stress
        self.material_parameters.hardening = model.mpm.hardening
        self.material_parameters.hardening_rate = model.mpm.hardening_rate
        self.material_parameters.softening_rate = model.mpm.softening_rate
        self.material_parameters.dilatancy = model.mpm.dilatancy
        self.material_parameters.viscosity = model.mpm.viscosity

        # Recompute particle volume and density from available particle data.
        # Assume that particles represent a cuboid volume of space, i.e., V = 8 r**3
        # (particles are typically laid out in a grid, and represent a uniform material).
        with wp.ScopedDevice(model.device):
            num_particles = model.particle_q.shape[0]
            self.particle_radius = _particle_parameter(num_particles, model.particle_radius)
            self.particle_volume = _reuse_or_allocate(getattr(self, "particle_volume", None), num_particles)
            self.particle_density = _reuse_or_allocate(getattr(self, "particle_density", None), num_particles)
            wp.launch(
                _compute_particle_volume_density,
                dim=num_particles,
                inputs=[self.particle_radius, model.particle_mass],
                outputs=[self.particle_volume, self.particle_density],
            )

        self._require_particle_arrays()
        self.refresh_particle_flags()
        self._refresh_material_extrema()

    def refresh_particle_flags(self):
        """Refresh the transfer and material particle masks from ``model.particle_flags``.

        Runs a single kernel that writes the existing ``particle_flags``,
        ``material_particle_flags`` and ``material_particle_volume`` arrays in
        place: no host synchronization or allocation, so it may be called every
        step and recorded in a CUDA graph. The material extrema do not depend on
        the ``ACTIVE`` or ``PROXY`` flags, so changing them never changes the
        solver's code paths.
        """
        model = self.model
        wp.launch(
            _refresh_particle_masks,
            dim=model.particle_count,
            inputs=[model.particle_flags, self._is_collider_particle, self.particle_volume],
            outputs=[self.particle_flags, self.material_particle_flags, self.material_particle_volume],
            device=model.device,
        )

    def _require_particle_arrays(self):
        """Allocate the particle masks and extrema on first use; later calls keep the same arrays.

        The arrays are written in place afterwards, so arrays bound in captured graphs stay valid.
        """
        model = self.model
        num_particles = model.particle_count
        with wp.ScopedDevice(model.device):
            self.particle_flags = _reuse_or_allocate(getattr(self, "particle_flags", None), num_particles, wp.int32)
            self.material_particle_flags = _reuse_or_allocate(
                getattr(self, "material_particle_flags", None), num_particles, wp.int32
            )
            self.material_particle_volume = _reuse_or_allocate(
                getattr(self, "material_particle_volume", None), num_particles
            )
            if getattr(self, "_is_collider_particle", None) is None:
                # Filled by setup_collider(); no particle is a collider before that.
                self._is_collider_particle = wp.zeros(num_particles, dtype=wp.int32)
            if getattr(self, "_material_extrema", None) is None:
                self._material_extrema = wp.empty(_EXTREMA_COUNT, dtype=float)

    def _refresh_collider_particle_mask(self):
        """Mark the particles that belong to deformable colliders, which never carry material."""
        is_collider_particle = np.zeros(self.model.particle_count, dtype=np.int32)
        collider_particle_ids = getattr(self.collider, "collider_particle_ids", None)
        if collider_particle_ids is not None:
            collider_particle_ids_np = collider_particle_ids.numpy()
            if collider_particle_ids_np.size:
                is_collider_particle[collider_particle_ids_np] = 1
        self._is_collider_particle.assign(is_collider_particle)

    def _refresh_material_extrema(self):
        """Reduce the material extrema on the device over every particle that can carry material.

        Collider particles never carry material, so they are excluded. Inactive
        and proxy particles are included, so that the ``ACTIVE`` and ``PROXY``
        flags can change at runtime without changing the solver's code paths.
        Unset ``model.mpm.*`` attributes default to rigid material without
        hardening, viscosity or dilatancy, and so do not enable additional code
        paths.

        Outside a graph capture, the extrema are read back to the host right
        away. Inside a capture, the readback is left to the next
        :meth:`sync_material_extrema` call outside a capture.
        """
        model = self.model
        params = self.material_parameters
        wp.launch(_reset_material_extrema, dim=1, outputs=[self._material_extrema], device=model.device)
        wp.launch(
            _reduce_material_extrema,
            dim=model.particle_count,
            inputs=[
                self._is_collider_particle,
                params.young_modulus,
                params.hardening,
                params.viscosity,
                params.dilatancy,
            ],
            outputs=[self._material_extrema],
            device=model.device,
        )
        self._extrema_host_stale = True
        if model.device.is_capturing:
            self._extrema_refreshed_in_graph = True
        else:
            self.sync_material_extrema()

    def code_path_switches(self) -> tuple[bool, bool, bool, bool]:
        """Return the host-side switches that select solver code paths.

        Returns:
            ``(has_compliant_particles, has_hardening, has_viscosity, has_dilatancy)``.
        """
        return (self.has_compliant_particles, self.has_hardening, self.has_viscosity, self.has_dilatancy)

    def record_captured_code_paths(self):
        """Remember the code-path switches in effect while ``step()`` is being captured in a CUDA graph."""
        self._captured_code_paths = self.code_path_switches()

    def sync_material_extrema(self):
        """Copy the device material extrema to the host if they may have changed.

        Does nothing during a graph capture, where the host keeps its last
        values. If a notification was ever recorded in a graph, every call
        outside a capture reads the extrema back, since graph replays can change
        them without the host knowing.

        Warns when the code-path switches changed since ``step()`` was last
        captured: the captured graph keeps the code paths in effect at capture
        time, so the change only takes effect in a graph captured afterwards.
        """
        if not (self._extrema_host_stale or self._extrema_refreshed_in_graph):
            return
        if self.model.device.is_capturing:
            return

        extrema = self._material_extrema.numpy()
        self.min_young_modulus = float(extrema[_EXTREMA_MIN_YOUNG_MODULUS])
        self.max_hardening = float(extrema[_EXTREMA_MAX_HARDENING])
        self.has_viscosity = bool(extrema[_EXTREMA_MAX_VISCOSITY] > 0.0)
        self.has_dilatancy = bool(extrema[_EXTREMA_MAX_DILATANCY] > 0.0)
        self._extrema_host_stale = False

        captured = self._captured_code_paths
        if captured is not None and captured != self.code_path_switches():
            names = ("compliant particles", "hardening", "viscosity", "dilatancy")
            changed = ", ".join(
                f"{name} {'on' if now else 'off'}"
                for name, was, now in zip(names, captured, self.code_path_switches(), strict=True)
                if was != now
            )
            warnings.warn(
                f"SolverImplicitMPM material change switches code paths ({changed}). CUDA graphs that "
                "captured step() before this change keep the previous code paths; capture them again.",
                stacklevel=3,
            )
            self._captured_code_paths = None

    def notify_collider_changed(self, body_mass: np.ndarray | None = None):
        """Refresh cached extrema for collider parameters.

        Tracks the minimum collider mass to determine whether compliant
        colliders are present and to enable/disable related computations.
        """
        body_ids = self.collider.collider_body_index.numpy()
        if body_mass is None:
            body_mass = self.collider_body_mass.numpy()
        dynamic_body_ids = body_ids[body_ids >= 0]
        dynamic_body_ids = dynamic_body_ids[body_mass[dynamic_body_ids] > 0.0]
        dynamic_body_masses = body_mass[dynamic_body_ids]
        deformable_particle_ids = self.collider.collider_particle_ids.numpy()
        if deformable_particle_ids.size:
            particle_mass = self.model.particle_mass.numpy()
            deformable_particle_masses = particle_mass[deformable_particle_ids]
            deformable_particle_masses = deformable_particle_masses[deformable_particle_masses > 0.0]
            dynamic_body_masses = np.concatenate((dynamic_body_masses, deformable_particle_masses))

        self.min_collider_mass = np.min(dynamic_body_masses, initial=np.inf)
        self.collider.query_max_dist = self.voxel_size * math.sqrt(3.0)
        self.collider_body_count = int(np.max(body_ids + 1, initial=0))

    def setup_collider(
        self,
        collider_meshes: list[wp.Mesh] | None = None,
        collider_body_ids: list[int] | None = None,
        collider_thicknesses: list[float] | None = None,
        collider_friction: list[float] | None = None,
        collider_adhesion: list[float] | None = None,
        collider_projection_threshold: list[float] | None = None,
        collider_particle_ids: list[list[int] | wp.array[int] | None] | None = None,
        model: newton.Model | None = None,
        body_com: wp.array | None = None,
        body_mass: wp.array | None = None,
        body_inv_inertia: wp.array | None = None,
        body_q: wp.array | None = None,
        collider_world_ids: list[int] | None = None,
        collider_material_counts: list[int | None] | None = None,
    ):
        """Initialize collider parameters and defaults from inputs.

        Populates the ``Collider`` struct with meshes, body mapping, and per-material
        properties (thickness, friction, adhesion, projection threshold).

        By default, this will setup collisions against all collision shapes in the model with flag `newton.ShapeFlag.COLLIDE_PARTICLES`.
        Rigid body colliders will be treated as kinematic if their effective mass is zero. When ``body_mass`` is omitted,
        bodies flagged with :attr:`newton.BodyFlags.KINEMATIC` have zero effective mass regardless of their stored mass.
        An explicit ``body_mass`` array is authoritative.

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
            collider_material_counts: Number of materials to allocate per collider (default 1). Body colliders
                allocate at least one material per shape, so ``max(shape count, count)`` materials; the first ones
                are the shape materials and faces start on them. Extra materials, and all materials of mesh
                colliders (whose faces start on the first one), start with the collider's material values. Face
                assignments and material values can then be changed through the views returned by
                :meth:`collider_face_materials` and :meth:`collider_materials`.
            collider_particle_ids: For deformable mesh colliders, solver-model particle IDs corresponding to each
                mesh vertex. These IDs cannot be combined with an external ``model``. In isolated multi-world mode,
                every ID must belong to the collider's local world; global deformable colliders are rejected.
            model: The model to read collider properties from. Default to self.model.
            body_com: For dynamic colliders, per-body center of mass. Default to model.body_com.
            body_mass: For dynamic colliders, per-body effective mass. By default, use ``model.body_mass`` with
                zero mass for bodies flagged with :attr:`newton.BodyFlags.KINEMATIC`.
            body_inv_inertia: For dynamic colliders, per-body inverse inertia. Default to model.body_inv_inertia.
            body_q: For dynamic colliders, per-body initial transform. Default to model.body_q.
            collider_world_ids: Per-collider Newton world IDs. Custom meshes default to global
                (``-1``), while body-backed colliders infer their body's world.

        Raises:
            ValueError: If collider inputs are inconsistent, world IDs are invalid, an isolated external model has a
                different world count, a global body-backed collider is dynamic, or deformable collider particle
                ownership cannot be mapped safely to the solver model and world.
        """

        if model is None:
            model = self.model
        elif self._separate_worlds and model is not self.model and model.world_count != self.model.world_count:
            raise ValueError(
                "An external collider model must have the same world_count as the isolated solver model; "
                f"got {model.world_count} and {self.model.world_count}."
            )

        collider_meshes = None if collider_meshes is None else list(collider_meshes)
        collider_body_ids = None if collider_body_ids is None else list(collider_body_ids)
        collider_thicknesses = None if collider_thicknesses is None else list(collider_thicknesses)
        collider_friction = None if collider_friction is None else list(collider_friction)
        collider_adhesion = None if collider_adhesion is None else list(collider_adhesion)
        collider_projection_threshold = (
            None if collider_projection_threshold is None else list(collider_projection_threshold)
        )
        collider_particle_ids = None if collider_particle_ids is None else list(collider_particle_ids)
        supplied_world_ids = None if collider_world_ids is None else list(collider_world_ids)

        shape_world = None
        body_world = None

        def get_shape_world():
            nonlocal shape_world
            if shape_world is None:
                shape_world = (
                    model.shape_world.numpy()
                    if model.shape_world is not None
                    else np.full(model.shape_count, -1, dtype=int)
                )
            return shape_world

        def get_body_world():
            nonlocal body_world
            if body_world is None:
                body_world = (
                    model.body_world.numpy()
                    if model.body_world is not None
                    else np.full(model.body_count, -1, dtype=int)
                )
            return body_world

        default_discovery = collider_meshes is None and collider_body_ids is None
        collider_shapes = []
        inferred_world_ids = []
        if default_discovery:
            collider_meshes = []
            collider_body_ids = []

            static_shapes = _get_body_collision_shapes(model, -1)
            if self._separate_worlds:
                static_collider_worlds = sorted({int(world) for world in get_shape_world()[static_shapes]})
            else:
                static_collider_worlds = [-1] if len(static_shapes) > 0 else []
            for world_id in static_collider_worlds:
                collider_meshes.append(None)
                collider_body_ids.append(-1)
                collider_shapes.append(
                    static_shapes[get_shape_world()[static_shapes] == world_id]
                    if self._separate_worlds
                    else static_shapes
                )
                inferred_world_ids.append(world_id)

            for body_id in range(model.body_count):
                shapes = _get_body_collision_shapes(model, body_id)
                if len(shapes) == 0:
                    continue
                collider_meshes.append(None)
                collider_body_ids.append(body_id)
                collider_shapes.append(shapes)
                inferred_world_ids.append(int(get_body_world()[body_id]) if self._separate_worlds else -1)
        else:
            if collider_body_ids is None:
                collider_body_ids = [None] * len(collider_meshes)
            elif collider_meshes is None:
                collider_meshes = [None] * len(collider_body_ids)
            elif len(collider_meshes) != len(collider_body_ids):
                raise ValueError(
                    "collider_meshes and collider_body_ids must have the same length; "
                    f"got {len(collider_meshes)} and {len(collider_body_ids)}."
                )
            collider_shapes = [None] * len(collider_body_ids)

        collider_count = len(collider_body_ids)

        def require_aligned(name, values, default=None):
            if values is None:
                return [default] * collider_count
            if len(values) != collider_count:
                raise ValueError(f"{name} must have one value per collider ({collider_count}); got {len(values)}.")
            return values

        collider_meshes = require_aligned("collider_meshes", collider_meshes)
        collider_thicknesses = require_aligned("collider_thicknesses", collider_thicknesses)
        collider_projection_threshold = require_aligned("collider_projection_threshold", collider_projection_threshold)
        collider_friction = require_aligned("collider_friction", collider_friction)
        collider_adhesion = require_aligned("collider_adhesion", collider_adhesion)
        collider_material_counts = require_aligned("collider_material_counts", collider_material_counts)
        collider_particle_ids = require_aligned("collider_particle_ids", collider_particle_ids)
        supplied_world_ids = require_aligned("collider_world_ids", supplied_world_ids)

        def validate_world_id(world_id, collider_id):
            if not isinstance(world_id, (int, np.integer)):
                raise ValueError(f"Invalid collider world ID {world_id!r} for collider {collider_id}.")
            world_id = int(world_id)
            if world_id < -1 or world_id >= self.model.world_count:
                raise ValueError(
                    f"Invalid collider world ID {world_id} for collider {collider_id}; expected -1 or an ID in "
                    f"[0, {self.model.world_count})."
                )
            return world_id

        if default_discovery:
            collider_world_ids = []
            for collider_id, raw_inferred_world_id in enumerate(inferred_world_ids):
                inferred_world_id = validate_world_id(raw_inferred_world_id, collider_id)
                supplied_world_id = supplied_world_ids[collider_id]
                if supplied_world_id is not None:
                    supplied_world_id = validate_world_id(supplied_world_id, collider_id)
                    if supplied_world_id != inferred_world_id:
                        raise ValueError(
                            f"Collider world ID {supplied_world_id} for collider {collider_id} does not match "
                            f"its inferred world ID {inferred_world_id}."
                        )
                collider_world_ids.append(inferred_world_id)
        else:
            collider_world_ids = []
            static_shapes = None
            for collider_id, (mesh, raw_body_id, requested_world_id) in enumerate(
                zip(collider_meshes, collider_body_ids, supplied_world_ids, strict=True)
            ):
                if mesh is None and raw_body_id is None:
                    raise ValueError(
                        f"Either a mesh or a body_id must be provided for each collider; collider {collider_id} is missing both"
                    )
                if mesh is not None and raw_body_id is not None:
                    raise ValueError(
                        f"Either a mesh or a body_id must be provided for each collider; collider {collider_id} provides both"
                    )

                if raw_body_id is None:
                    world_id = -1 if requested_world_id is None else requested_world_id
                else:
                    try:
                        body_id = int(raw_body_id)
                    except (TypeError, ValueError) as exc:
                        raise ValueError(
                            f"Invalid collider body ID {raw_body_id!r} for collider {collider_id}."
                        ) from exc
                    if body_id < -1 or body_id >= model.body_count:
                        raise ValueError(
                            f"Invalid collider body ID {body_id} for collider {collider_id}; expected -1 or an ID "
                            f"in [0, {model.body_count})."
                        )
                    collider_body_ids[collider_id] = body_id

                    if body_id >= 0:
                        if self._separate_worlds:
                            world_id = validate_world_id(get_body_world()[body_id], collider_id)
                        else:
                            world_id = -1
                        if requested_world_id is not None and self._separate_worlds:
                            validated_world_id = validate_world_id(requested_world_id, collider_id)
                            if validated_world_id != world_id:
                                raise ValueError(
                                    f"Collider world ID {validated_world_id} for collider {collider_id} does not "
                                    f"match body {body_id}'s inferred world ID {world_id}."
                                )
                        elif requested_world_id is not None:
                            world_id = requested_world_id
                        collider_shapes[collider_id] = _get_body_collision_shapes(model, body_id)
                    else:
                        if static_shapes is None:
                            static_shapes = _get_body_collision_shapes(model, -1)
                        if self._separate_worlds:
                            if requested_world_id is None:
                                static_world_ids = sorted({int(world) for world in get_shape_world()[static_shapes]})
                                if len(static_world_ids) != 1:
                                    raise ValueError(
                                        "An explicit body -1 collider is ambiguous in isolated multi-world mode; "
                                        "supply its collider world ID or use default discovery/custom meshes."
                                    )
                                world_id = static_world_ids[0]
                            else:
                                world_id = validate_world_id(requested_world_id, collider_id)
                            collider_shapes[collider_id] = static_shapes[get_shape_world()[static_shapes] == world_id]
                        else:
                            world_id = -1 if requested_world_id is None else requested_world_id
                            collider_shapes[collider_id] = static_shapes

                collider_world_ids.append(validate_world_id(world_id, collider_id))

        if body_com is None:
            body_com = model.body_com
        body_mass_is_explicit = body_mass is not None
        if body_mass is None:
            body_mass = model.body_mass
        if body_inv_inertia is None:
            body_inv_inertia = model.body_inv_inertia
        if body_q is None:
            body_q = model.body_q

        effective_body_mass = body_mass.numpy().copy()
        if not body_mass_is_explicit and model.body_flags is not None:
            body_flags = model.body_flags.numpy()
            kinematic_bodies = (body_flags & int(newton.BodyFlags.KINEMATIC)) != 0
            if np.any(kinematic_bodies & (effective_body_mass != 0.0)):
                effective_body_mass[kinematic_bodies] = 0.0
                body_mass = wp.array(effective_body_mass, dtype=body_mass.dtype, device=body_mass.device)
        if self._separate_worlds:
            for collider_id, (body_id, world_id) in enumerate(zip(collider_body_ids, collider_world_ids, strict=True)):
                if world_id == -1 and body_id is not None and body_id >= 0 and effective_body_mass[body_id] > 0.0:
                    raise ValueError(
                        f"Collider {collider_id} is a global dynamic collider backed by body {body_id}, which would "
                        "couple isolated worlds. Replicate it into each world, make it static or kinematic, or disable "
                        "Config.separate_worlds."
                    )

        # count materials and shapes
        material_count = 1  # default material
        collider_material_ids = []
        for collider_id, (body_id, shapes) in enumerate(zip(collider_body_ids, collider_shapes, strict=True)):
            requested_material_count = collider_material_counts[collider_id]
            if requested_material_count is not None and (
                not isinstance(requested_material_count, (int, np.integer)) or requested_material_count < 1
            ):
                raise ValueError(
                    f"collider_material_counts[{collider_id}] must be a positive integer; "
                    f"got {requested_material_count!r}."
                )
            if body_id is not None:
                if len(shapes) == 0:
                    raise ValueError(f"Body {body_id} has no collision shapes for its collider world ID")

            # body colliders own at least one material per shape
            shape_count = 0 if body_id is None else len(shapes)
            collider_material_count = max(
                shape_count, 1 if requested_material_count is None else requested_material_count
            )
            collider_material_ids.append(list(range(material_count, material_count + collider_material_count)))
            material_count += collider_material_count

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
                shape_count = len(collider_shapes[collider_id])
                for material_id, shape_margin, shape_friction, shape_adhesion, shape_projection_threshold in zip(
                    collider_material_ids[collider_id][:shape_count],
                    *_get_shape_collision_materials(model, collider_shapes[collider_id]),
                    strict=True,
                ):
                    # use material from shapes as default
                    has_projection_threshold = shape_projection_threshold >= 0.0
                    assign_material(
                        material_id,
                        thickness=shape_margin,
                        friction=shape_friction,
                        adhesion=shape_adhesion,
                        projection_threshold=shape_projection_threshold if has_projection_threshold else None,
                    )
                    # override with user-provided material
                    assign_collider_material(material_id, collider_id)
                # extra materials beyond the shapes start from the user-provided values
                for material_id in collider_material_ids[collider_id][shape_count:]:
                    assign_collider_material(material_id, collider_id)
            else:
                # user-provided collider, all materials start from the same values
                for material_id in collider_material_ids[collider_id]:
                    assign_collider_material(material_id, collider_id)

        collider_max_thickness = [
            max((material_thickness[material_id] for material_id in collider_material_ids[collider_id]), default=0.0)
            for collider_id in range(collider_count)
        ]
        has_deformable_colliders = any(particle_ids is not None for particle_ids in collider_particle_ids)
        if has_deformable_colliders and model is not self.model:
            raise ValueError(
                "collider_particle_ids are solver-state particle indices and may only be used with the solver model; "
                "external collider models are not supported"
            )

        solver_particle_world = (
            self.model.particle_world.numpy() if self._separate_worlds and has_deformable_colliders else None
        )
        collider_particle_offsets = [0]
        collider_particle_id_chunks = []
        for collider_id, particle_ids in enumerate(collider_particle_ids):
            if particle_ids is None:
                collider_particle_offsets.append(collider_particle_offsets[-1])
                continue

            if collider_body_ids[collider_id] is not None:
                raise ValueError("collider_particle_ids may only be provided for mesh colliders")
            if collider_meshes[collider_id] is None:
                raise ValueError("collider_particle_ids requires a collider mesh")

            if isinstance(particle_ids, wp.array):
                particle_ids_np = particle_ids.numpy()
            else:
                particle_ids_np = np.asarray(particle_ids, dtype=int)

            vertex_count = collider_meshes[collider_id].points.shape[0]
            if particle_ids_np.shape[0] != vertex_count:
                raise ValueError(
                    f"collider_particle_ids[{collider_id}] has {particle_ids_np.shape[0]} entries, "
                    f"but collider mesh has {vertex_count} vertices"
                )
            if particle_ids_np.size and (
                np.min(particle_ids_np) < 0 or np.max(particle_ids_np) >= self.model.particle_count
            ):
                raise ValueError(f"collider_particle_ids[{collider_id}] contains particle ids outside the solver model")

            if self._separate_worlds:
                collider_world_id = collider_world_ids[collider_id]
                if collider_world_id < 0:
                    raise ValueError(
                        f"collider_particle_ids[{collider_id}] cannot define a global deformable collider across "
                        "isolated worlds; assign the collider to one local world"
                    )

                particle_world_ids = np.unique(solver_particle_world[particle_ids_np])
                if np.any(particle_world_ids != collider_world_id):
                    raise ValueError(
                        f"collider_particle_ids[{collider_id}] must reference particles in collider world "
                        f"{collider_world_id}; found particle world IDs {particle_world_ids.tolist()}"
                    )

            collider_particle_id_chunks.append(particle_ids_np)
            collider_particle_offsets.append(collider_particle_offsets[-1] + particle_ids_np.shape[0])

        if not collider_particle_id_chunks:
            flat_collider_particle_ids = np.empty(0, dtype=int)
        elif len(collider_particle_id_chunks) == 1:
            flat_collider_particle_ids = collider_particle_id_chunks[0]
        else:
            flat_collider_particle_ids = np.concatenate(collider_particle_id_chunks)

        # Create device arrays
        with wp.ScopedDevice(self.model.device):
            # Create collider meshes from bodies if necessary
            packed_body_ids = []
            face_material_ids = []
            collider_face_offsets = []
            collider_face_counts = []
            face_offset = 0
            for collider_id in range(collider_count):
                body_index = collider_body_ids[collider_id]

                if body_index is None:
                    # Set body index to -1 to indicate a static collider
                    # This may not correspond to the model's body -1, but as far as the collision kernels
                    # are concerned, it does not matter.

                    packed_body_ids.append(-1)
                    face_count = collider_meshes[collider_id].indices.shape[0] // 3
                    mesh_face_material_ids = np.zeros(face_count, dtype=int)
                else:
                    # faces start on their shape's material, the first ones of the collider
                    collider_meshes[collider_id], mesh_face_material_ids = _create_body_collider_mesh(
                        model, collider_shapes[collider_id]
                    )
                    packed_body_ids.append(body_index)
                    face_count = collider_meshes[collider_id].indices.shape[0] // 3

                face_material_ids.append(mesh_face_material_ids)
                collider_face_offsets.append(face_offset)
                collider_face_counts.append(face_count)
                face_offset += face_count

            global_collider_ids = [
                collider_id for collider_id, collider_world_id in enumerate(collider_world_ids) if collider_world_id < 0
            ]
            world_collider_ids = []
            world_collider_offsets = [0]
            for world_id in range(self.model.world_count):
                world_collider_ids.extend(global_collider_ids)
                world_collider_ids.extend(
                    collider_id
                    for collider_id, collider_world_id in enumerate(collider_world_ids)
                    if collider_world_id == world_id
                )
                world_collider_offsets.append(len(world_collider_ids))

            self.collider.collider_body_index = wp.array(packed_body_ids, dtype=int)
            self.collider.collider_particle_offsets = wp.array(collider_particle_offsets, dtype=int)
            self.collider.collider_particle_ids = wp.array(flat_collider_particle_ids, dtype=int)
            self.collider.collider_mesh = wp.array([collider.id for collider in collider_meshes], dtype=wp.uint64)
            self.collider.collider_max_thickness = wp.array(collider_max_thickness, dtype=float)
            self.collider.collider_world = wp.array(collider_world_ids, dtype=int)
            self.collider.collider_face_offset = wp.array(collider_face_offsets, dtype=int)
            self.collider.world_collider_ids = wp.array(world_collider_ids, dtype=int)
            self.collider.world_collider_offsets = wp.array(world_collider_offsets, dtype=int)

            all_face_material_ids = np.concatenate(face_material_ids) if face_material_ids else np.empty(0, dtype=int)
            self.collider.face_material_index = wp.array(all_face_material_ids, dtype=int)
            self.collider.collider_material_offset = wp.array([ids[0] for ids in collider_material_ids], dtype=int)
            self.collider.collider_material_count = wp.array([len(ids) for ids in collider_material_ids], dtype=int)

            self.collider.material_thickness = wp.array(material_thickness, dtype=float)
            self.collider.material_friction = wp.array(material_friction, dtype=float)
            self.collider.material_adhesion = wp.array(material_adhesion, dtype=float)
            self.collider.material_projection_threshold = wp.array(material_projection_threshold, dtype=float)

        self.collider_face_ranges = list(zip(collider_face_offsets, collider_face_counts, strict=True))
        self.collider_material_ranges = [(ids[0], len(ids)) for ids in collider_material_ids]
        self.collider.body_com = body_com
        self.collider_body_mass = body_mass
        self.collider_body_inv_inertia = body_inv_inertia
        self.collider_body_q = body_q
        self._collider_meshes = collider_meshes  # Keep a ref so that meshes are not garbage collected
        self.deformable_collider_vertex_ranges = [
            (collider_id, collider_particle_offsets[collider_id], collider_particle_offsets[collider_id + 1])
            for collider_id in range(collider_count)
            if collider_particle_offsets[collider_id + 1] > collider_particle_offsets[collider_id]
        ]

        self._refresh_collider_particle_mask()
        self.refresh_particle_flags()
        self._refresh_material_extrema()
        self.notify_collider_changed(effective_body_mass)

    @property
    def has_compliant_particles(self):
        return self.min_young_modulus < _INFINITY

    @property
    def has_hardening(self):
        return self.max_hardening > 0.0

    @property
    def has_compliant_colliders(self):
        return self.min_collider_mass < _INFINITY
