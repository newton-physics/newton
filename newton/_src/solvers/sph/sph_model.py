# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Smoothed Particle Hydrodynamics model helpers."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import warp as wp

from ...sim import Model, ModelBuilder, State
from .kernels import sph_kernel_id
from .utils import sph_finite_scalar, sph_vec3_array

__all__ = ["SPHModel"]

if TYPE_CHECKING:
    from .solver_wcsph import SolverWCSPH


def _require_array_device(value: Any, name: str, device: object) -> Any:
    if getattr(value, "device", None) != device:
        raise ValueError(f"{name} must be allocated on the SPH solver device ({device})")
    return value


def _require_body_array(value: Any, name: str, model: Model, device: object) -> Any:
    value = _require_array_device(value, name, device)
    if not getattr(value, "shape", None) or int(value.shape[0]) != int(model.body_count):
        raise ValueError(f"{name} must contain one entry per collider-model body")
    return value


@dataclass
class SPHNeighborSearch:
    """Hash-grid neighbor search used by the WCSPH solver."""

    model: Model
    grid_dim_x: int = 128
    grid_dim_y: int = 128
    grid_dim_z: int = 128

    def __post_init__(self) -> None:
        if self.model.particle_count:
            self.reserve(self.model.particle_count)

    @property
    def grid(self) -> wp.HashGrid | None:
        return self.model.particle_grid

    @property
    def grid_id(self) -> wp.uint64:
        grid = self.grid
        if grid is None:
            raise RuntimeError("SPH neighbor grid has not been allocated")
        return grid.id

    def reserve(self, particle_count: int | None = None) -> None:
        """Ensure the underlying hash grid exists and can hold particles."""
        if particle_count is None:
            particle_count = self.model.particle_count
        if particle_count < 0:
            raise ValueError("particle_count must be non-negative")
        if particle_count <= 0:
            return

        grid_dimensions = (int(self.grid_dim_x), int(self.grid_dim_y), int(self.grid_dim_z))
        if (
            self.model.particle_grid is None
            or getattr(self.model, "_sph_particle_grid_dimensions", None) != grid_dimensions
        ):
            self.model.particle_grid = wp.HashGrid(
                self.grid_dim_x,
                self.grid_dim_y,
                self.grid_dim_z,
                device=self.model.device,
            )
            self.model._sph_particle_grid_dimensions = grid_dimensions

        with wp.ScopedDevice(self.model.device):
            self.model.particle_grid.reserve(particle_count)

    def build(self, state: State, radius: float) -> None:
        """Build the active neighbor grid for ``state``."""
        if radius <= 0.0 or not math.isfinite(radius):
            raise ValueError("neighbor search radius must be finite and positive")
        particle_count = self.model.particle_count
        if particle_count == 0:
            return

        self.reserve(particle_count)
        with wp.ScopedDevice(self.model.device):
            self.model.particle_grid.build(state.particle_q, radius=radius)


class SPHScratchpad:
    """Reusable per-particle work arrays for :class:`SolverWCSPH`."""

    def __init__(self, model: Model):
        self.model = model
        self.particle_count = -1
        self.resize()

    def resize(self) -> None:
        particle_count = int(self.model.particle_count)
        if particle_count == self.particle_count:
            return
        self.acceleration = wp.empty(particle_count, dtype=wp.vec3, device=self.model.device)
        self.velocity_delta = wp.empty(particle_count, dtype=wp.vec3, device=self.model.device)
        self.particle_count = particle_count


class SPHModel:
    """Wrapper augmenting a ``newton.Model`` with WCSPH runtime state."""

    def __init__(
        self,
        model: Model,
        config: SolverWCSPH.Config,
    ):
        from .boundaries import SPHBoundaryHandler  # noqa: PLC0415

        self.model = model
        self.config = config
        self.kernel_id = sph_kernel_id(config.kernel)

        self.neighbor_search = SPHNeighborSearch(model)
        self.scratch = SPHScratchpad(model)
        self.boundary_handler = SPHBoundaryHandler(model, enable_shape_boundaries=config.enable_shape_boundaries)

        self.collider_body_com = None
        self.collider_body_mass = None
        self.collider_body_inv_inertia = None
        self.collider_body_q = None
        self.collider_body_index = wp.empty(0, dtype=int, device=model.device)
        self.collider_impulse_body_index = wp.empty(0, dtype=int, device=model.device)
        self.collider_impulse_id = wp.empty(0, dtype=int, device=model.device)
        self.collider_impulse = wp.empty(0, dtype=wp.vec3, device=model.device)
        self.collider_impulse_position = wp.empty(0, dtype=wp.vec3, device=model.device)

        self.refresh_model()
        self.setup_collider()

    def setup_collider(
        self,
        collider_meshes: list[wp.Mesh] | None = None,
        collider_body_ids: list[int] | None = None,
        collider_margins: list[float] | None = None,
        collider_friction: list[float] | None = None,
        collider_projection_threshold: list[float] | None = None,
        model: Model | None = None,
        body_com: wp.array[wp.vec3] | None = None,
        body_mass: wp.array[float] | None = None,
        body_inv_inertia: wp.array[wp.mat33] | None = None,
        body_q: wp.array[wp.transform] | None = None,
    ) -> None:
        """Initialize collider parameters and defaults from inputs."""
        from .boundaries import _resolve_collider_body_ids  # noqa: PLC0415

        collider_model = self.model if model is None else model
        if collider_model.device != self.model.device:
            raise ValueError(
                f"Collider model device ({collider_model.device}) must match SPH model device ({self.model.device})."
            )

        self.collider_body_com = _require_body_array(
            collider_model.body_com if body_com is None else body_com,
            "body_com",
            collider_model,
            self.model.device,
        )
        self.collider_body_mass = _require_body_array(
            collider_model.body_mass if body_mass is None else body_mass,
            "body_mass",
            collider_model,
            self.model.device,
        )
        self.collider_body_inv_inertia = _require_body_array(
            collider_model.body_inv_inertia if body_inv_inertia is None else body_inv_inertia,
            "body_inv_inertia",
            collider_model,
            self.model.device,
        )
        self.collider_body_q = _require_body_array(
            collider_model.body_q if body_q is None else body_q,
            "body_q",
            collider_model,
            self.model.device,
        )

        self.boundary_handler.set_collider_model(collider_model)
        self.boundary_handler.save_collider_current_position(self.collider_body_q)

        body_ids = _resolve_collider_body_ids(collider_model, collider_body_ids, collider_meshes)
        if collider_meshes is None:
            self.boundary_handler.set_explicit_collider_meshes(None)
            self.boundary_handler.set_model_collider_material_overrides(
                body_ids,
                margins=collider_margins,
                friction=collider_friction,
                projection_threshold=collider_projection_threshold,
            )
        else:
            self.boundary_handler.set_explicit_collider_meshes(
                collider_meshes,
                body_ids=body_ids,
                margins=collider_margins,
                friction=collider_friction,
                projection_threshold=collider_projection_threshold,
            )
            self.boundary_handler.set_model_collider_material_overrides(
                (),
                margins=None,
                friction=None,
                projection_threshold=None,
            )
        self.collider_body_index = wp.array(body_ids, dtype=int, device=self.model.device)
        dynamic_collider_ids_list: list[int] = []
        dynamic_bodies: set[int] = set()
        for collider, body in enumerate(body_ids):
            body_index = int(body)
            if body_index >= 0 and body_index not in dynamic_bodies:
                dynamic_collider_ids_list.append(collider)
                dynamic_bodies.add(body_index)
        dynamic_collider_ids = np.asarray(dynamic_collider_ids_list, dtype=np.int32)
        self.collider_impulse_body_index = wp.array(body_ids[dynamic_collider_ids], dtype=int, device=self.model.device)
        impulse_count = 2 * dynamic_collider_ids.size
        self.collider_impulse_id = wp.empty(impulse_count, dtype=int, device=self.model.device)
        self.collider_impulse = wp.empty(impulse_count, dtype=wp.vec3, device=self.model.device)
        self.collider_impulse_position = wp.empty(impulse_count, dtype=wp.vec3, device=self.model.device)
        self._dynamic_collider_ids = wp.array(dynamic_collider_ids, dtype=int, device=self.model.device)

    def refresh_model(self) -> None:
        """Refresh caches derived from mutable model or config arrays."""
        self.default_support_radius = self.compute_default_support_radius()
        self.max_support_radius = self.compute_max_support_radius()
        self.max_depenetration_velocity = self.compute_max_depenetration_velocity()
        self.scratch.resize()
        if hasattr(self, "boundary_handler"):
            self.boundary_handler.refresh_model()
        if self.model.particle_count:
            self.neighbor_search.reserve(self.model.particle_count)

    def compute_default_support_radius(self) -> float:
        if self.config.smoothing_length is not None:
            return float(self.config.smoothing_length)
        return max(4.0 * self.model.particle_max_radius, 1.0e-3)

    def compute_max_support_radius(self) -> float:
        default_h = self.default_support_radius

        if not self.model.particle_count or not hasattr(self.model, "sph"):
            return float(default_h)

        values = self.model.sph.smoothing_length.numpy()
        if values.size == 0:
            return float(default_h)
        positive = values[values > 0.0]
        if positive.size == 0:
            return float(default_h)
        max_h = float(np.max(positive))
        if np.any(values <= 0.0):
            max_h = max(max_h, default_h)
        return max_h

    def compute_max_depenetration_velocity(self) -> float:
        """Bound contact stabilization below the WCSPH acoustic velocity scale."""
        sound_speed = self.config.sound_speed
        if sound_speed is None:
            sound_speed = (
                float(np.max(self.model.sph.sound_speed.numpy(), initial=0.0))
                if self.model.particle_count and hasattr(self.model, "sph")
                else 0.0
            )
        return min(float(self.model.particle_max_velocity), max(0.05, 0.01 * float(sound_speed)))

    def build_neighbor_grid(self, state: State) -> None:
        self.neighbor_search.build(state, self.max_support_radius)

    def collide_shape_boundaries(
        self,
        state: State,
        *,
        collider_velocity_mode: int,
        dt: float,
        mesh_query_max_distance: float | None = None,
    ) -> None:
        self.boundary_handler.enable_shape_boundaries = self.config.enable_shape_boundaries
        if mesh_query_max_distance is None:
            mesh_query_max_distance = self.max_support_radius
        self.boundary_handler.collide_analytic_shapes(
            state,
            boundary_margin=self.config.boundary_margin,
            boundary_friction=self.config.boundary_friction,
            mesh_query_max_distance=mesh_query_max_distance,
            body_com=self.collider_body_com,
            body_mass=self.collider_body_mass,
            body_inv_inertia=self.collider_body_inv_inertia,
            max_depenetration_velocity=self.max_depenetration_velocity,
            collider_velocity_mode=collider_velocity_mode,
            dt=dt,
        )


@dataclass(frozen=True)
class SPHCustomAttributeSpec:
    """Descriptor for one custom attribute required by Newton SPH."""

    name: str
    frequency: Model.AttributeFrequency
    assignment: Model.AttributeAssignment
    dtype: Any
    default: Any
    namespace: str = "sph"

    @property
    def qualified_name(self) -> str:
        return f"{self.namespace}:{self.name}"

    def custom_attribute(self) -> ModelBuilder.CustomAttribute:
        return ModelBuilder.CustomAttribute(
            name=self.name,
            frequency=self.frequency,
            assignment=self.assignment,
            dtype=self.dtype,
            default=self.default,
            namespace=self.namespace,
        )


_PARTICLE = Model.AttributeFrequency.PARTICLE
_MODEL = Model.AttributeAssignment.MODEL
_STATE = Model.AttributeAssignment.STATE


_SPH_MODEL_PARTICLE_ATTRIBUTES = tuple(
    SPHCustomAttributeSpec(name, _PARTICLE, _MODEL, dtype, default)
    for name, dtype, default in (
        ("rest_density", wp.float32, 1000.0),
        ("sound_speed", wp.float32, 20.0),
        ("stiffness", wp.float32, 0.0),
        ("pressure_exponent", wp.float32, 7.0),
        ("pressure_min", wp.float32, 0.0),
        ("pressure_max", wp.float32, 0.0),
        ("viscosity", wp.float32, 0.001),
        ("smoothing_length", wp.float32, 0.0),
    )
)

_SPH_STATE_PARTICLE_ATTRIBUTES = tuple(
    SPHCustomAttributeSpec(name, _PARTICLE, _STATE, dtype, default)
    for name, dtype, default in (
        ("density", wp.float32, 0.0),
        ("pressure", wp.float32, 0.0),
        ("volume", wp.float32, 0.0),
        ("boundary_impulse", wp.vec3, wp.vec3(0.0)),
    )
)


def _sph_custom_attribute_specs() -> tuple[SPHCustomAttributeSpec, ...]:
    return _SPH_MODEL_PARTICLE_ATTRIBUTES + _SPH_STATE_PARTICLE_ATTRIBUTES


def validate_sph_custom_attributes(model: Model) -> None:
    """Validate the model attributes required by the WCSPH solver."""
    if model.particle_count == 0:
        return

    namespace = getattr(model, "sph", None)
    missing: list[str] = []
    mismatched: list[str] = []
    for spec in _SPH_MODEL_PARTICLE_ATTRIBUTES:
        value = getattr(namespace, spec.name, None) if namespace is not None else None
        if value is None:
            missing.append(spec.qualified_name)
        elif not value.shape or int(value.shape[0]) != int(model.particle_count):
            mismatched.append(spec.qualified_name)

    if missing or mismatched:
        details = []
        if missing:
            details.append(f"missing {', '.join(missing)}")
        if mismatched:
            details.append(f"wrong particle count for {', '.join(mismatched)}")
        raise ValueError(
            "SolverWCSPH requires SPH custom attributes registered with "
            "SolverWCSPH.register_custom_attributes(builder) before finalizing the model: " + "; ".join(details)
        )


def register_sph_custom_attributes(builder: ModelBuilder) -> None:
    """Register the custom attributes used by Newton's SPH solver."""
    for spec in _sph_custom_attribute_specs():
        builder.add_custom_attribute(spec.custom_attribute())


def _validate_material_fields(
    material: object,
    names: Sequence[str],
    message: str,
    is_invalid: Callable[[Any], bool],
) -> None:
    for name in names:
        if is_invalid(_material_scalar(getattr(material, name), name)):
            raise ValueError(f"{name} must be {message}")


def _material_scalar(value: Any, name: str) -> float:
    return sph_finite_scalar(value, f"{name} must be finite")


@dataclass(frozen=True)
class SPHMaterial:
    """Material defaults used when adding SPH particles to a Newton model."""

    rest_density: float = 1000.0
    """Reference density [kg/m^3]."""
    sound_speed: float = 20.0
    """Artificial sound speed used by the WCSPH equation of state [m/s]."""
    stiffness: float = 0.0
    """Pressure stiffness used when ``sound_speed`` is zero [Pa]."""
    pressure_exponent: float = 7.0
    """Tait equation-of-state exponent."""
    pressure_min: float = 0.0
    """Minimum pressure clamp [Pa]."""
    pressure_max: float = 0.0
    """Maximum pressure clamp [Pa]. A non-positive value disables the upper clamp."""
    viscosity: float = 0.001
    """Dynamic viscosity coefficient [Pa s]."""
    smoothing_length: float | None = None
    """Kernel support radius [m]. ``None`` uses the solver default."""

    def validate(self) -> None:
        _validate_material_fields(self, ("rest_density",), "positive", lambda value: value <= 0.0)
        _validate_material_fields(
            self,
            (
                "sound_speed",
                "stiffness",
                "viscosity",
            ),
            "non-negative",
            lambda value: value < 0.0,
        )
        _validate_material_fields(
            self,
            ("pressure_exponent",),
            "finite and positive",
            lambda value: value <= 0.0,
        )
        pressure_min = _material_scalar(self.pressure_min, "pressure_min")
        pressure_max = _material_scalar(self.pressure_max, "pressure_max")
        if pressure_max < 0.0:
            raise ValueError("pressure_max must be finite and non-negative")
        if pressure_max > 0.0 and pressure_min > pressure_max:
            raise ValueError("pressure_min must be less than or equal to pressure_max")
        if self.smoothing_length is not None and _material_scalar(self.smoothing_length, "smoothing_length") <= 0.0:
            raise ValueError("smoothing_length must be positive")

    def custom_attributes(self) -> dict[str, Any]:
        """Return custom attributes matching ``SolverWCSPH.register_custom_attributes``."""
        self.validate()
        attrs = {
            "sph:rest_density": self.rest_density,
            "sph:sound_speed": self.sound_speed,
            "sph:stiffness": self.stiffness,
            "sph:pressure_exponent": self.pressure_exponent,
            "sph:pressure_min": self.pressure_min,
            "sph:pressure_max": self.pressure_max,
            "sph:viscosity": self.viscosity,
        }
        if self.smoothing_length is not None:
            attrs["sph:smoothing_length"] = self.smoothing_length
        return attrs


def make_sph_particle_attributes(
    *,
    material: SPHMaterial | None = None,
    custom_attributes: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compose SPH custom attributes for Newton particle builder calls."""
    if material is None:
        material = SPHMaterial()
    attrs = material.custom_attributes()
    if custom_attributes:
        attrs.update(custom_attributes)
    return attrs


def _finite_scalar(value: Any, name: str) -> float:
    return sph_finite_scalar(value, f"SPH {name} must be finite")


def _finite_positive_scalar(value: Any, name: str) -> float:
    result = _finite_scalar(value, name)
    if result <= 0.0:
        raise ValueError(f"SPH {name} must be finite and positive")
    return result


def _finite_nonnegative_scalar(value: Any, name: str) -> float:
    result = _finite_scalar(value, name)
    if result < 0.0:
        raise ValueError(f"SPH {name} must be finite and non-negative")
    return result


def _coerce_vec3(value: Any, name: str) -> wp.vec3:
    values = sph_vec3_array(
        value,
        f"SPH {name} must be a three-item vector",
        f"SPH {name} entries must be finite",
        bool_message=f"SPH {name} must be a three-item vector",
    )
    return wp.vec3(float(values[0]), float(values[1]), float(values[2]))


def _coerce_quat(value: Any, name: str) -> wp.quat:
    try:
        if any(isinstance(value[index], bool | np.bool_) for index in range(4)):
            raise ValueError
        values = (float(value[0]), float(value[1]), float(value[2]), float(value[3]))
    except (TypeError, IndexError, ValueError) as exc:
        raise ValueError(f"SPH {name} must be a four-item xyzw quaternion") from exc
    if any(not np.isfinite(component) for component in values):
        raise ValueError(f"SPH {name} entries must be finite")
    return wp.quat(*values)


def add_sph_particle_grid(
    builder: ModelBuilder,
    *,
    pos: Any,
    rot: Any | None = None,
    vel: Any | None = None,
    dim_x: int,
    dim_y: int,
    dim_z: int,
    cell_x: float,
    cell_y: float,
    cell_z: float,
    material: SPHMaterial | None = None,
    mass: float | None = None,
    jitter: float = 0.0,
    radius_mean: float | None = None,
    radius_std: float = 0.0,
    flags: list[int] | int | None = None,
    custom_attributes: dict[str, Any] | None = None,
) -> range:
    """Add an SPH-tagged particle grid and return the created particle indices.

    SPH bodies remain ordinary Newton particles with namespaced SPH metadata.

    Args:
        builder: Newton model builder that receives the particles and SPH attributes.
        pos: Grid origin in world coordinates.
        rot: Grid orientation as an xyzw quaternion. Defaults to identity.
        vel: Initial particle velocity. Defaults to zero.
        dim_x: Particle count along the local X axis.
        dim_y: Particle count along the local Y axis.
        dim_z: Particle count along the local Z axis.
        cell_x: Particle spacing along the local X axis [m].
        cell_y: Particle spacing along the local Y axis [m].
        cell_z: Particle spacing along the local Z axis [m].
        material: SPH material values stored on each particle.
        mass: Per-particle mass [kg]. Defaults to rest density times cell volume.
        jitter: Random position jitter applied by ``ModelBuilder.add_particle_grid``.
        radius_mean: Mean Newton collision radius [m]. Defaults to half the minimum cell spacing.
        radius_std: Standard deviation of the Newton collision radius [m].
        flags: Newton particle flags passed to the builder.
        custom_attributes: Additional per-particle custom attributes.

    Returns:
        Contiguous indices of the particles added to ``builder``.
    """
    if dim_x <= 0 or dim_y <= 0 or dim_z <= 0:
        raise ValueError("SPH particle grid dimensions must be positive")
    cell_x = _finite_positive_scalar(cell_x, "particle grid cell_x")
    cell_y = _finite_positive_scalar(cell_y, "particle grid cell_y")
    cell_z = _finite_positive_scalar(cell_z, "particle grid cell_z")
    jitter = _finite_scalar(jitter, "particle grid jitter")
    radius_std = _finite_nonnegative_scalar(radius_std, "particle grid radius_std")

    if material is None:
        material = SPHMaterial()
    material.validate()

    register_sph_custom_attributes(builder)

    if rot is None:
        rot = wp.quat_identity()
    else:
        rot = _coerce_quat(rot, "particle grid rot")
    pos = _coerce_vec3(pos, "particle grid pos")
    if vel is None:
        vel = wp.vec3(0.0)
    else:
        vel = _coerce_vec3(vel, "particle grid vel")
    if radius_mean is None:
        radius_mean = 0.5 * min(cell_x, cell_y, cell_z)
    else:
        radius_mean = _finite_positive_scalar(radius_mean, "particle grid radius_mean")
    if mass is None:
        mass = material.rest_density * cell_x * cell_y * cell_z
    else:
        mass = _finite_nonnegative_scalar(mass, "particle grid mass")

    start = len(builder.particle_q)
    attrs = make_sph_particle_attributes(
        material=material,
        custom_attributes=custom_attributes,
    )
    builder.add_particle_grid(
        pos=pos,
        rot=rot,
        vel=vel,
        dim_x=dim_x,
        dim_y=dim_y,
        dim_z=dim_z,
        cell_x=cell_x,
        cell_y=cell_y,
        cell_z=cell_z,
        mass=mass,
        jitter=jitter,
        radius_mean=radius_mean,
        radius_std=radius_std,
        flags=flags,
        custom_attributes=attrs,
    )
    return range(start, len(builder.particle_q))
