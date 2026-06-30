# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Smoothed Particle Hydrodynamics model helpers."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, fields
from enum import IntEnum
from typing import Any, Literal, get_args, get_type_hints

import numpy as np
import warp as wp

from ...geometry import GeoType, Heightfield, ParticleFlags
from ...sim import Model, ModelBuilder, State
from .kernels import sph_kernel_id, sph_kernel_names, sph_kernel_weight_np
from .utils import sph_finite_scalar, sph_vec3_array

SPHKernel = Literal["poly6", "cubic", "wendland", "spiky"]


__all__ = ["SPHModel"]


def _validated_int(value: Any, name: str, *, minimum: int) -> int:
    numeric = sph_finite_scalar(value, f"{name} must be an integer")
    if not numeric.is_integer():
        raise ValueError(f"{name} must be an integer")
    result = int(numeric)
    if result < minimum:
        if minimum == 1:
            raise ValueError(f"{name} must be positive")
        if minimum == 0:
            raise ValueError(f"{name} must be non-negative")
        raise ValueError(f"{name} must be at least {minimum}")
    return result


def _nonnegative_int(value: Any, name: str) -> int:
    return _validated_int(value, name, minimum=0)


def _require_array_device(value: Any, name: str, device: object) -> Any:
    if getattr(value, "device", None) != device:
        raise ValueError(f"{name} must be allocated on the SPH solver device ({device})")
    return value


class SPHRole(IntEnum):
    """Role of a particle in the SPH module."""

    FLUID = 0
    """Dynamic fluid particle."""

    BOUNDARY = 1
    """Static sampled boundary particle."""


def _validate_config_fields(
    owner: object,
    names: Sequence[str],
    message: str,
    is_invalid: Callable[[Any], bool],
) -> None:
    for name in names:
        value = getattr(owner, name)
        if is_invalid(value):
            raise ValueError(f"{name} must be {message}")


def _config_scalar_type(field_type: object) -> type | None:
    if field_type in (bool, int, float):
        return field_type
    args = get_args(field_type)
    if type(None) not in args:
        return None
    scalar_args = tuple(arg for arg in args if arg is not type(None))
    if len(scalar_args) == 1 and scalar_args[0] in (int, float):
        return scalar_args[0]
    return None


def _validate_config_scalar_fields(config: object) -> None:
    type_hints = get_type_hints(type(config))
    for field in fields(config):
        expected = _config_scalar_type(type_hints.get(field.name, field.type))
        if expected is None:
            continue
        value = getattr(config, field.name)
        if value is None:
            continue
        if expected is bool:
            if not isinstance(value, bool):
                raise ValueError(f"{field.name} must be a boolean")
        elif expected is int:
            numeric = sph_finite_scalar(value, f"{field.name} must be an integer")
            if not numeric.is_integer():
                raise ValueError(f"{field.name} must be an integer")
        elif expected is float:
            sph_finite_scalar(value, f"{field.name} must be finite")


@dataclass
class SPHConfig:
    """Configuration for :class:`~newton.solvers.SolverWCSPH`."""

    # kernel / equation of state
    kernel: SPHKernel = "poly6"
    """Smoothing kernel family."""

    smoothing_length: float | None = None
    """Default support radius [m] when per-particle attributes are absent or zero."""

    rest_density: float = 1000.0
    """Default fluid rest density [kg/m^3]."""

    sound_speed: float = 20.0
    """Default artificial sound speed [m/s] for WCSPH pressure."""

    stiffness: float = 0.0
    """Fallback pressure stiffness when ``sound_speed`` is zero."""

    pressure_exponent: float = 1.0
    """Tait-style equation-of-state exponent. ``1`` preserves the linear WCSPH law."""

    # viscosity / surface physics
    viscosity: float = 0.001
    """Default viscosity coefficient for the explicit viscosity term."""

    xsph: float = 0.0
    """XSPH velocity smoothing coefficient applied during integration."""

    enable_surface_tension: bool = False
    """Apply continuum-surface-force surface tension for particles with ``sph:surface_tension``."""

    surface_tension_normal_threshold: float = 1.0e-6
    """Minimum surface-normal length used before applying curvature-derived surface tension."""

    # boundaries / coupling
    enable_shape_boundaries: bool = True
    """Project fluid particles out of supported Newton collision shapes."""

    boundary_margin: float = 0.0
    """Additional particle-boundary separation distance [m]."""

    boundary_friction: float = 0.0
    """Tangential velocity damping applied by analytic boundaries."""

    collider_velocity_mode: Literal["forward", "backward"] = "forward"
    """Collider velocity computation mode. ``'forward'`` uses ``State.body_qd``;
    ``'backward'`` uses the previous timestep body transform.
    """

    enable_boundary_adhesion: bool = False
    """Apply SPH boundary adhesion for sampled boundaries and adhesive analytic colliders."""

    enable_boundary_wetting: bool = False
    """Apply sampled-boundary wetting using ``sph:wetting`` and ``sph:contact_angle``."""

    def validate(self) -> None:
        _validate_config_scalar_fields(self)
        if self.kernel not in sph_kernel_names():
            available = ", ".join(sph_kernel_names())
            raise ValueError(f"Unsupported SPH kernel '{self.kernel}'. Available kernels: {available}")
        _validate_config_fields(
            self,
            ("rest_density", "pressure_exponent"),
            "positive",
            lambda value: value <= 0.0,
        )
        _validate_config_fields(
            self,
            (
                "sound_speed",
                "stiffness",
                "viscosity",
                "xsph",
                "surface_tension_normal_threshold",
                "boundary_margin",
                "boundary_friction",
            ),
            "non-negative",
            lambda value: value < 0.0,
        )
        if self.smoothing_length is not None and self.smoothing_length <= 0.0:
            raise ValueError("smoothing_length must be positive")
        if self.collider_velocity_mode not in ("forward", "backward"):
            raise ValueError(f"Invalid collider velocity mode: {self.collider_velocity_mode}")


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


class SPHModel:
    """Wrapper augmenting a ``newton.Model`` with WCSPH runtime state."""

    def __init__(
        self,
        model: Model,
        config: SPHConfig,
    ):
        from .boundaries import SPHBoundaryHandler  # noqa: PLC0415

        self.model = model
        self.config = config
        self.kernel_id = sph_kernel_id(config.kernel)

        self.neighbor_search = SPHNeighborSearch(model)
        self.boundary_handler = SPHBoundaryHandler(model, enable_shape_boundaries=config.enable_shape_boundaries)

        self.collider_body_com = None
        self.collider_body_mass = None
        self.collider_body_inv_inertia = None
        self.collider_body_q = None
        self.collider_body_index_np = np.zeros(0, dtype=np.int32)
        self.collider_body_index = wp.array(self.collider_body_index_np, dtype=int, device=model.device)
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
        collider_adhesion: list[float] | None = None,
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

        self.collider_body_com = _require_array_device(
            collider_model.body_com if body_com is None else body_com,
            "body_com",
            self.model.device,
        )
        self.collider_body_mass = _require_array_device(
            collider_model.body_mass if body_mass is None else body_mass,
            "body_mass",
            self.model.device,
        )
        self.collider_body_inv_inertia = _require_array_device(
            collider_model.body_inv_inertia if body_inv_inertia is None else body_inv_inertia,
            "body_inv_inertia",
            self.model.device,
        )
        self.collider_body_q = _require_array_device(
            collider_model.body_q if body_q is None else body_q,
            "body_q",
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
                adhesion=collider_adhesion,
                projection_threshold=collider_projection_threshold,
            )
        else:
            self.boundary_handler.set_explicit_collider_meshes(
                collider_meshes,
                body_ids=body_ids,
                margins=collider_margins,
                friction=collider_friction,
                adhesion=collider_adhesion,
                projection_threshold=collider_projection_threshold,
            )
            self.boundary_handler.set_model_collider_material_overrides(
                None,
                margins=None,
                friction=None,
                adhesion=None,
                projection_threshold=None,
            )
        self.collider_body_index_np = body_ids
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
        self.collider_impulse_id = wp.empty(dynamic_collider_ids.size, dtype=int, device=self.model.device)
        self.collider_impulse = wp.empty(dynamic_collider_ids.size, dtype=wp.vec3, device=self.model.device)
        self.collider_impulse_position = wp.empty(dynamic_collider_ids.size, dtype=wp.vec3, device=self.model.device)
        self._dynamic_collider_ids = wp.array(dynamic_collider_ids, dtype=int, device=self.model.device)

    def refresh_model(self) -> None:
        """Refresh caches derived from mutable model or config arrays."""
        self.max_support_radius = self.compute_max_support_radius()
        if hasattr(self, "boundary_handler"):
            self.boundary_handler.refresh_model()
        if self.model.particle_count:
            self.neighbor_search.reserve(self.model.particle_count)

    def compute_max_support_radius(self) -> float:
        default_h = self.config.smoothing_length
        if default_h is None:
            default_h = max(2.0 * self.model.particle_max_radius, 1.0e-3)

        if not self.model.particle_count or not hasattr(self.model, "sph"):
            return float(default_h)

        values = self.model.sph.smoothing_length.numpy()
        if values.size == 0:
            return float(default_h)
        positive = values[values > 0.0]
        if positive.size == 0:
            return float(default_h)
        return float(np.max(positive))

    def build_neighbor_grid(self, state: State) -> None:
        self.neighbor_search.build(state, self.max_support_radius)

    def support_radius_np(self, support: np.ndarray | None = None) -> np.ndarray:
        support = np.asarray(self.model.sph.smoothing_length.numpy(), dtype=np.float32) if support is None else support
        return np.where(support > 0.0, support, self.max_support_radius).astype(np.float32)

    def collide_shape_boundaries(self, state: State, *, collider_velocity_mode: int, dt: float) -> None:
        self.boundary_handler.enable_shape_boundaries = self.config.enable_shape_boundaries
        self.boundary_handler.collide_analytic_shapes(
            state,
            boundary_margin=self.config.boundary_margin,
            boundary_friction=self.config.boundary_friction,
            collider_velocity_mode=collider_velocity_mode,
            enable_boundary_adhesion=self.config.enable_boundary_adhesion,
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
        ("role", wp.int32, int(SPHRole.FLUID)),
        ("rest_density", wp.float32, 1000.0),
        ("sound_speed", wp.float32, 20.0),
        ("stiffness", wp.float32, 0.0),
        ("pressure_exponent", wp.float32, 1.0),
        ("pressure_min", wp.float32, 0.0),
        ("pressure_max", wp.float32, 0.0),
        ("viscosity", wp.float32, 0.001),
        ("smoothing_length", wp.float32, 0.0),
        ("surface_tension", wp.float32, 0.0),
        ("adhesion", wp.float32, 0.0),
        ("wetting", wp.float32, 0.0),
        ("contact_angle", wp.float32, 0.5 * math.pi),
    )
)

_SPH_STATE_PARTICLE_ATTRIBUTES = tuple(
    SPHCustomAttributeSpec(name, _PARTICLE, _STATE, dtype, default)
    for name, dtype, default in (
        ("density", wp.float32, 0.0),
        ("pressure", wp.float32, 0.0),
        ("volume", wp.float32, 0.0),
        ("acceleration", wp.vec3, wp.vec3(0.0)),
        ("surface_acceleration", wp.vec3, wp.vec3(0.0)),
        ("adhesion_acceleration", wp.vec3, wp.vec3(0.0)),
        ("wetting_acceleration", wp.vec3, wp.vec3(0.0)),
        ("boundary_impulse", wp.vec3, wp.vec3(0.0)),
        ("normal", wp.vec3, wp.vec3(0.0)),
        ("boundary_normal", wp.vec3, wp.vec3(0.0)),
        ("color_field", wp.float32, 0.0),
        ("velocity_delta", wp.vec3, wp.vec3(0.0)),
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
    """Material defaults used when adding SPH particles to a Newton model.

    Attributes:
        rest_density: Reference density [kg/m^3].
        sound_speed: Artificial sound speed used by the WCSPH equation of state [m/s].
        stiffness: Pressure stiffness used when ``sound_speed`` is zero.
        pressure_exponent: Tait-style equation-of-state exponent.
        pressure_min: Minimum pressure clamp.
        pressure_max: Maximum pressure clamp. A non-positive value disables the upper clamp.
        viscosity: Dynamic viscosity coefficient.
        smoothing_length: Kernel support radius [m]. ``None`` uses the solver default.
        surface_tension: Continuum-surface-force surface-tension coefficient.
        adhesion: Sampled-boundary adhesion coefficient.
        wetting: Sampled-boundary wetting coefficient.
        contact_angle: Wetting contact angle [rad].
    """

    rest_density: float = 1000.0
    sound_speed: float = 20.0
    stiffness: float = 0.0
    pressure_exponent: float = 1.0
    pressure_min: float = 0.0
    pressure_max: float = 0.0
    viscosity: float = 0.001
    smoothing_length: float | None = None
    surface_tension: float = 0.0
    adhesion: float = 0.0
    wetting: float = 0.0
    contact_angle: float = 0.5 * math.pi

    def validate(self) -> None:
        _validate_material_fields(self, ("rest_density",), "positive", lambda value: value <= 0.0)
        _validate_material_fields(
            self,
            (
                "sound_speed",
                "stiffness",
                "viscosity",
                "surface_tension",
                "adhesion",
                "wetting",
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
        contact_angle = _material_scalar(self.contact_angle, "contact_angle")
        if contact_angle < 0.0 or contact_angle > math.pi:
            raise ValueError("contact_angle must be finite and in [0, pi]")

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
            "sph:surface_tension": self.surface_tension,
            "sph:adhesion": self.adhesion,
            "sph:wetting": self.wetting,
            "sph:contact_angle": self.contact_angle,
        }
        if self.smoothing_length is not None:
            attrs["sph:smoothing_length"] = self.smoothing_length
        return attrs


def make_sph_particle_attributes(
    *,
    material: SPHMaterial | None = None,
    role: SPHRole | int | None = None,
    boundary_normal: Any | None = None,
    custom_attributes: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compose SPH custom attributes for Newton particle builder calls."""
    if material is None:
        material = SPHMaterial()
    if role is None:
        role = SPHRole.FLUID
    else:
        role = _role_from_value(role)

    attrs: dict[str, Any] = {"sph:role": int(role)}
    if boundary_normal is not None:
        attrs["sph:boundary_normal"] = boundary_normal
    attrs.update(material.custom_attributes())
    if custom_attributes:
        attrs.update(custom_attributes)
    return attrs


def _role_from_value(role: SPHRole | int | str) -> SPHRole:
    if isinstance(role, SPHRole):
        return role
    if isinstance(role, str):
        key = role.strip().upper()
        if key.startswith("SPHROLE."):
            key = key.split(".", 1)[1]
        try:
            return SPHRole[key]
        except KeyError as exc:
            raise ValueError(f"Unknown SPH role '{role}'") from exc
    return SPHRole(_nonnegative_int(role, "SPH role"))


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


def _vec3_to_np(vec: wp.vec3) -> np.ndarray:
    return np.array([float(vec[0]), float(vec[1]), float(vec[2])], dtype=np.float64)


def _sampled_boundary_contributions(
    points: Sequence[wp.vec3],
    *,
    material: SPHMaterial,
    spacing: float,
    kernel: SPHKernel,
) -> list[float]:
    smoothing_length = material.smoothing_length if material.smoothing_length is not None else 2.0 * spacing
    h = _finite_positive_scalar(smoothing_length, "boundary smoothing length")
    kernel_id = sph_kernel_id(kernel)
    fallback = material.rest_density * spacing * spacing * spacing
    coords = np.asarray([_vec3_to_np(point) for point in points], dtype=np.float64)
    masses: list[float] = []

    for point in coords:
        delta = 0.0
        for neighbor in coords:
            delta += sph_kernel_weight_np(kernel_id, float(np.linalg.norm(point - neighbor)), h)
        if delta <= 1.0e-12:
            masses.append(float(fallback))
        else:
            masses.append(float(material.rest_density / delta))

    return masses


def _axis_samples(half_extent: float, spacing: float) -> np.ndarray:
    intervals = max(1, int(np.ceil((2.0 * half_extent) / spacing)))
    return np.linspace(-half_extent, half_extent, intervals + 1)


def _point_key(coords: Sequence[float]) -> tuple[float, float, float]:
    return tuple(round(float(c), 8) for c in coords)


def _sample_box_surface_points(half_extents: tuple[float, float, float], spacing: float) -> list[wp.vec3]:
    hx, hy, hz = half_extents
    if hx <= 0.0 or hy <= 0.0 or hz <= 0.0:
        raise ValueError("SPH box boundary half-extents must be positive")

    axes = (_axis_samples(hx, spacing), _axis_samples(hy, spacing), _axis_samples(hz, spacing))
    points: dict[tuple[float, float, float], wp.vec3] = {}

    for axis, half_extent in enumerate(half_extents):
        other_a = (axis + 1) % 3
        other_b = (axis + 2) % 3
        for side in (-half_extent, half_extent):
            for a in axes[other_a]:
                for b in axes[other_b]:
                    coords = [0.0, 0.0, 0.0]
                    coords[axis] = side
                    coords[other_a] = float(a)
                    coords[other_b] = float(b)
                    points[_point_key(coords)] = wp.vec3(*coords)

    return list(points.values())


def _sample_sphere_surface_points(radius: float, spacing: float) -> list[wp.vec3]:
    if radius <= 0.0:
        raise ValueError("SPH sphere boundary radius must be positive")

    count = max(6, int(np.ceil(4.0 * np.pi * radius * radius / (spacing * spacing))))
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))
    points = []

    for i in range(count):
        z = 1.0 - (2.0 * (i + 0.5) / count)
        radial = np.sqrt(max(0.0, 1.0 - z * z))
        theta = i * golden_angle
        points.append(wp.vec3(radius * radial * np.cos(theta), radius * radial * np.sin(theta), radius * z))

    return points


def _sample_ellipsoid_surface_points(axes: tuple[float, float, float], spacing: float) -> list[wp.vec3]:
    rx, ry, rz = axes
    if rx <= 0.0 or ry <= 0.0 or rz <= 0.0:
        raise ValueError("SPH ellipsoid boundary semi-axes must be positive")

    reference_radius = max(rx, ry, rz)
    unit_points = _sample_sphere_surface_points(1.0, spacing / reference_radius)
    return [wp.vec3(rx * p[0], ry * p[1], rz * p[2]) for p in unit_points]


def _sample_capsule_surface_points(radius: float, half_height: float, spacing: float) -> list[wp.vec3]:
    if radius <= 0.0:
        raise ValueError("SPH capsule boundary radius must be positive")
    if half_height < 0.0:
        raise ValueError("SPH capsule boundary half-height must be non-negative")

    if half_height == 0.0:
        return _sample_sphere_surface_points(radius, spacing)

    radial_count = max(6, int(np.ceil(2.0 * np.pi * radius / spacing)))
    axial_count = max(1, int(np.ceil((2.0 * half_height) / spacing)))
    cap_ring_count = max(2, int(np.ceil(0.5 * np.pi * radius / spacing)))
    points: dict[tuple[float, float, float], wp.vec3] = {}

    for axial in range(axial_count + 1):
        z = -half_height + 2.0 * half_height * axial / axial_count
        for radial in range(radial_count):
            theta = 2.0 * np.pi * radial / radial_count
            coords = (radius * np.cos(theta), radius * np.sin(theta), z)
            points[_point_key(coords)] = wp.vec3(*coords)

    for sign in (-1.0, 1.0):
        center_z = sign * half_height
        for ring in range(1, cap_ring_count + 1):
            phi = 0.5 * np.pi * ring / cap_ring_count
            ring_radius = radius * np.cos(phi)
            z = center_z + sign * radius * np.sin(phi)
            count = max(1, int(np.ceil(2.0 * np.pi * max(ring_radius, spacing * 0.5) / spacing)))
            for radial in range(count):
                theta = 2.0 * np.pi * radial / count
                coords = (ring_radius * np.cos(theta), ring_radius * np.sin(theta), z)
                points[_point_key(coords)] = wp.vec3(*coords)

    return list(points.values())


def _sample_cylinder_surface_points(radius: float, half_height: float, spacing: float) -> list[wp.vec3]:
    if radius <= 0.0:
        raise ValueError("SPH cylinder boundary radius must be positive")
    if half_height <= 0.0:
        raise ValueError("SPH cylinder boundary half-height must be positive")

    radial_count = max(6, int(np.ceil(2.0 * np.pi * radius / spacing)))
    axial_count = max(1, int(np.ceil((2.0 * half_height) / spacing)))
    ring_count = max(1, int(np.ceil(radius / spacing)))
    points: dict[tuple[float, float, float], wp.vec3] = {}

    for axial in range(axial_count + 1):
        z = -half_height + 2.0 * half_height * axial / axial_count
        for radial in range(radial_count):
            theta = 2.0 * np.pi * radial / radial_count
            coords = (radius * np.cos(theta), radius * np.sin(theta), z)
            points[_point_key(coords)] = wp.vec3(*coords)

    for z in (-half_height, half_height):
        points[_point_key((0.0, 0.0, z))] = wp.vec3(0.0, 0.0, z)
        for ring in range(1, ring_count + 1):
            ring_radius = radius * ring / ring_count
            count = max(6, int(np.ceil(2.0 * np.pi * ring_radius / spacing)))
            for radial in range(count):
                theta = 2.0 * np.pi * radial / count
                coords = (ring_radius * np.cos(theta), ring_radius * np.sin(theta), z)
                points[_point_key(coords)] = wp.vec3(*coords)

    return list(points.values())


def _sample_cone_surface_points(radius: float, half_height: float, spacing: float) -> list[wp.vec3]:
    if radius <= 0.0:
        raise ValueError("SPH cone boundary radius must be positive")
    if half_height <= 0.0:
        raise ValueError("SPH cone boundary half-height must be positive")

    axial_count = max(1, int(np.ceil((2.0 * half_height) / spacing)))
    ring_count = max(1, int(np.ceil(radius / spacing)))
    points: dict[tuple[float, float, float], wp.vec3] = {}

    for axial in range(axial_count + 1):
        z = -half_height + 2.0 * half_height * axial / axial_count
        ring_radius = radius * (half_height - z) / (2.0 * half_height)
        if ring_radius <= 1.0e-8:
            coords = (0.0, 0.0, half_height)
            points[_point_key(coords)] = wp.vec3(*coords)
            continue

        count = max(6, int(np.ceil(2.0 * np.pi * ring_radius / spacing)))
        for radial in range(count):
            theta = 2.0 * np.pi * radial / count
            coords = (ring_radius * np.cos(theta), ring_radius * np.sin(theta), z)
            points[_point_key(coords)] = wp.vec3(*coords)

    for ring in range(ring_count + 1):
        ring_radius = radius * ring / ring_count
        if ring == 0:
            coords = (0.0, 0.0, -half_height)
            points[_point_key(coords)] = wp.vec3(*coords)
            continue

        count = max(6, int(np.ceil(2.0 * np.pi * ring_radius / spacing)))
        for radial in range(count):
            theta = 2.0 * np.pi * radial / count
            coords = (ring_radius * np.cos(theta), ring_radius * np.sin(theta), -half_height)
            points[_point_key(coords)] = wp.vec3(*coords)

    return list(points.values())


def _normalize_np(vec: np.ndarray) -> wp.vec3:
    length = float(np.linalg.norm(vec))
    if length <= 1.0e-12:
        return wp.vec3(0.0)
    normalized = vec / length
    return wp.vec3(float(normalized[0]), float(normalized[1]), float(normalized[2]))


def _normalize_vec3(vec: wp.vec3) -> wp.vec3:
    return _normalize_np(np.array([vec[0], vec[1], vec[2]], dtype=np.float64))


def _box_surface_normal(point: wp.vec3, half_extents: tuple[float, float, float]) -> wp.vec3:
    normal = np.zeros(3, dtype=np.float64)
    for axis, half_extent in enumerate(half_extents):
        if np.isclose(abs(float(point[axis])), half_extent, atol=1.0e-8):
            normal[axis] = np.sign(float(point[axis]))
    return _normalize_np(normal)


def _sphere_surface_normal(point: wp.vec3) -> wp.vec3:
    return _normalize_np(np.array([point[0], point[1], point[2]], dtype=np.float64))


def _ellipsoid_surface_normal(point: wp.vec3, axes: tuple[float, float, float]) -> wp.vec3:
    return _normalize_np(
        np.array(
            [
                float(point[0]) / (axes[0] * axes[0]),
                float(point[1]) / (axes[1] * axes[1]),
                float(point[2]) / (axes[2] * axes[2]),
            ],
            dtype=np.float64,
        )
    )


def _capsule_surface_normal(point: wp.vec3, half_height: float) -> wp.vec3:
    z = float(point[2])
    if z > half_height:
        center_z = half_height
    elif z < -half_height:
        center_z = -half_height
    else:
        center_z = z
    return _normalize_np(np.array([point[0], point[1], z - center_z], dtype=np.float64))


def _cylinder_surface_normal(point: wp.vec3, radius: float, half_height: float) -> wp.vec3:
    radial = np.array([point[0], point[1], 0.0], dtype=np.float64)
    normal = np.zeros(3, dtype=np.float64)
    if np.isclose(np.linalg.norm(radial[:2]), radius, atol=1.0e-8):
        normal += radial
    if np.isclose(float(point[2]), half_height, atol=1.0e-8):
        normal[2] += 1.0
    elif np.isclose(float(point[2]), -half_height, atol=1.0e-8):
        normal[2] -= 1.0
    return _normalize_np(normal)


def _cone_surface_normal(point: wp.vec3, radius: float, half_height: float) -> wp.vec3:
    z = float(point[2])
    radial = np.array([point[0], point[1], 0.0], dtype=np.float64)
    radial_length = float(np.linalg.norm(radial[:2]))
    normal = np.zeros(3, dtype=np.float64)
    if np.isclose(z, -half_height, atol=1.0e-8):
        normal[2] -= 1.0
    side_radius = radius * (half_height - z) / (2.0 * half_height)
    if radial_length > 1.0e-12 and np.isclose(radial_length, side_radius, atol=1.0e-8):
        normal += np.array([point[0] / radial_length, point[1] / radial_length, radius / (2.0 * half_height)])
    elif np.isclose(z, half_height, atol=1.0e-8):
        normal[2] += 1.0
    return _normalize_np(normal)


def _sample_triangle_surface_points_and_normals(
    vertices: Any,
    indices: Any,
    scale: tuple[float, float, float],
    spacing: float,
) -> tuple[list[wp.vec3], list[wp.vec3]]:
    vertices = np.asarray(vertices, dtype=np.float64).reshape(-1, 3)
    indices = np.asarray(indices, dtype=np.int64).reshape(-1)
    if vertices.size == 0:
        raise ValueError("SPH mesh boundary sampling requires at least one mesh vertex")
    if len(indices) == 0 or len(indices) % 3 != 0:
        raise ValueError("SPH mesh boundary sampling requires triangle indices")
    if np.any(indices < 0) or np.any(indices >= len(vertices)):
        raise ValueError("SPH mesh boundary sampling found out-of-range triangle indices")

    scale_np = np.asarray(scale, dtype=np.float64)
    if np.any(scale_np <= 0.0):
        raise ValueError("SPH mesh boundary scale must be positive on all axes")
    scaled_vertices = vertices * scale_np[None, :]

    points: dict[tuple[float, float, float], wp.vec3] = {}
    normals: dict[tuple[float, float, float], np.ndarray] = {}
    triangles = indices.reshape(-1, 3)
    for tri in triangles:
        a = scaled_vertices[tri[0]]
        b = scaled_vertices[tri[1]]
        c = scaled_vertices[tri[2]]
        ab = b - a
        ac = c - a
        bc = c - b
        area = 0.5 * np.linalg.norm(np.cross(ab, ac))
        if area <= 1.0e-16:
            continue
        face_normal = np.cross(ab, ac)
        face_normal /= np.linalg.norm(face_normal)

        max_edge = max(float(np.linalg.norm(ab)), float(np.linalg.norm(ac)), float(np.linalg.norm(bc)))
        subdivision = max(1, int(np.ceil(max_edge / spacing)), int(np.ceil(np.sqrt(2.0 * area) / spacing)))
        for u_index in range(subdivision + 1):
            for v_index in range(subdivision + 1 - u_index):
                u = u_index / subdivision
                v = v_index / subdivision
                p = a + u * ab + v * ac
                key = _point_key(p)
                points[key] = wp.vec3(float(p[0]), float(p[1]), float(p[2]))
                if key in normals:
                    normals[key] += face_normal
                else:
                    normals[key] = face_normal.copy()

    if not points:
        raise ValueError("SPH mesh boundary sampling produced no particles")
    keys = list(points.keys())
    return [points[key] for key in keys], [_normalize_np(normals[key]) for key in keys]


def _sample_mesh_surface_points_and_normals(
    mesh: Any, scale: tuple[float, float, float], spacing: float
) -> tuple[list[wp.vec3], list[wp.vec3]]:
    return _sample_triangle_surface_points_and_normals(mesh.vertices, mesh.indices, scale, spacing)


def _heightfield_top_surface_mesh(
    heightfield: Heightfield,
    scale: tuple[float, float, float],
) -> tuple[np.ndarray, np.ndarray]:
    if heightfield.nrow < 2 or heightfield.ncol < 2:
        raise ValueError("SPH heightfield boundary sampling requires nrow and ncol >= 2")
    data = np.asarray(heightfield.data, dtype=np.float64).reshape(heightfield.nrow, heightfield.ncol)
    sx, sy, sz = (float(value) for value in scale)
    hx = abs(float(heightfield.hx) * sx)
    hy = abs(float(heightfield.hy) * sy)
    if hx <= 0.0 or hy <= 0.0:
        raise ValueError("SPH heightfield boundary scale must give positive x/y extents")
    min_z = float(heightfield.min_z) * sz
    max_z = float(heightfield.max_z) * sz
    z_range = max_z - min_z

    xs = np.linspace(-hx, hx, int(heightfield.ncol), dtype=np.float64)
    ys = np.linspace(-hy, hy, int(heightfield.nrow), dtype=np.float64)
    vertices = np.empty((int(heightfield.nrow) * int(heightfield.ncol), 3), dtype=np.float64)
    for row, y in enumerate(ys):
        for col, x in enumerate(xs):
            vertex_index = row * int(heightfield.ncol) + col
            vertices[vertex_index] = (float(x), float(y), min_z + float(data[row, col]) * z_range)

    indices: list[int] = []
    for row in range(int(heightfield.nrow) - 1):
        for col in range(int(heightfield.ncol) - 1):
            p00 = row * int(heightfield.ncol) + col
            p10 = p00 + 1
            p01 = (row + 1) * int(heightfield.ncol) + col
            p11 = p01 + 1
            indices.extend((p00, p10, p11, p00, p11, p01))
    return vertices, np.asarray(indices, dtype=np.int32)


def _sample_heightfield_surface_points_and_normals(
    heightfield: Heightfield,
    scale: tuple[float, float, float],
    spacing: float,
) -> tuple[list[wp.vec3], list[wp.vec3]]:
    vertices, indices = _heightfield_top_surface_mesh(heightfield, scale)
    return _sample_triangle_surface_points_and_normals(vertices, indices, (1.0, 1.0, 1.0), spacing)


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
    role: SPHRole | int = SPHRole.FLUID,
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
        role: SPH role assigned to each particle.
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
        role=role,
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


def add_sph_boundary_points(
    builder: ModelBuilder,
    *,
    points: Sequence[Any],
    normals: Sequence[Any],
    material: SPHMaterial | None = None,
    vel: Any | None = None,
    spacing: float | None = None,
    mass: float | None = None,
    radius: float | None = None,
    flags: int | None = None,
    kernel: SPHKernel = "poly6",
    custom_attributes: dict[str, Any] | None = None,
) -> range:
    """Add sampled-boundary point-cloud particles and return their indices.

    Args:
        builder: Newton model builder that receives the boundary particles.
        points: Boundary point positions in world coordinates.
        normals: Outward boundary normal corresponding to each point.
        material: SPH material values stored on each boundary particle.
        vel: Boundary velocity. Defaults to zero.
        spacing: Nominal sample spacing [m]. Required when ``mass`` is omitted.
        mass: Per-particle boundary mass [kg]. When omitted, kernel-normalized contributions are computed.
        radius: Newton collision radius [m]. Defaults to half ``spacing`` when available.
        flags: Newton particle flags. Defaults to ``ParticleFlags.ACTIVE``.
        kernel: SPH kernel used to normalize inferred boundary masses.
        custom_attributes: Additional per-particle custom attributes.

    Returns:
        Contiguous indices of the boundary particles added to ``builder``.
    """
    if not isinstance(points, Sequence) or isinstance(points, (str, bytes)) or len(points) == 0:
        raise ValueError("SPH boundary points must be a non-empty sequence")
    if not isinstance(normals, Sequence) or isinstance(normals, (str, bytes)) or len(normals) != len(points):
        raise ValueError("SPH boundary point normals must match the point count")
    if material is None:
        material = SPHMaterial()
    material.validate()
    if spacing is not None:
        spacing = _finite_positive_scalar(spacing, "boundary point spacing")
    if radius is None and spacing is not None:
        radius = 0.5 * spacing
    if radius is not None:
        radius = _finite_positive_scalar(radius, "boundary point radius")

    register_sph_custom_attributes(builder)

    positions = [_coerce_vec3(point, "boundary point") for point in points]
    boundary_normals = [_normalize_vec3(_coerce_vec3(normal, "boundary normal")) for normal in normals]
    if any(np.linalg.norm([normal[0], normal[1], normal[2]]) <= 1.0e-8 for normal in boundary_normals):
        raise ValueError("SPH boundary point normals must be non-zero")
    if mass is None:
        if spacing is None:
            raise ValueError("SPH boundary point mass requires explicit mass or spacing")
        masses = _sampled_boundary_contributions(positions, material=material, spacing=spacing, kernel=kernel)
    else:
        masses = [float(_finite_nonnegative_scalar(mass, "boundary point mass"))] * len(positions)
    if vel is None:
        vel = wp.vec3(0.0)
    vel = _coerce_vec3(vel, "boundary velocity")

    attrs = make_sph_particle_attributes(
        material=material,
        role=SPHRole.BOUNDARY,
        boundary_normal=boundary_normals,
        custom_attributes=custom_attributes,
    )
    particle_flags = int(ParticleFlags.ACTIVE) if flags is None else int(flags)
    start = len(builder.particle_q)
    builder.add_particles(
        pos=positions,
        vel=[vel] * len(positions),
        mass=masses,
        radius=None if radius is None else [float(radius)] * len(positions),
        flags=[particle_flags] * len(positions),
        custom_attributes=attrs,
    )
    return range(start, len(builder.particle_q))


def add_sph_boundary_from_shape(
    builder: ModelBuilder,
    shape: int,
    *,
    spacing: float,
    material: SPHMaterial | None = None,
    mass: float | None = None,
    radius: float | None = None,
    vel: Any | None = None,
    kernel: SPHKernel = "poly6",
    custom_attributes: dict[str, Any] | None = None,
) -> range:
    """Sample a Newton shape as SPH boundary particles.

    Currently supports `GeoType.BOX`, `GeoType.SPHERE`, `GeoType.ELLIPSOID`,
    `GeoType.CAPSULE`, `GeoType.CYLINDER`, `GeoType.CONE`, `GeoType.MESH`,
    `GeoType.CONVEX_MESH`, and `GeoType.HFIELD` shapes. The sampled particles
    are ordinary Newton particles with ``SPHRole.BOUNDARY`` metadata, which lets
    them contribute to SPH neighborhoods without being integrated as fluid
    particles.

    Args:
        builder: Newton model builder containing ``shape`` and receiving the sampled particles.
        shape: Builder shape index to sample.
        spacing: Target surface-sample spacing [m].
        material: SPH material values stored on each boundary particle.
        mass: Per-particle boundary mass [kg]. When omitted, kernel-normalized contributions are computed.
        radius: Newton collision radius [m]. Defaults to half ``spacing``.
        vel: Boundary velocity. Defaults to zero.
        kernel: SPH kernel used to normalize inferred boundary masses.
        custom_attributes: Additional per-particle custom attributes.

    Returns:
        Contiguous indices of the sampled boundary particles added to ``builder``.
    """
    if shape < 0 or shape >= builder.shape_count:
        raise IndexError(f"shape index {shape} is out of range")
    spacing = _finite_positive_scalar(spacing, "boundary sampling spacing")
    if builder.shape_world[shape] != builder.current_world:
        raise ValueError("Cannot sample an SPH boundary from a shape in a different builder world")
    shape_type = builder.shape_type[shape]
    if shape_type not in (
        GeoType.BOX,
        GeoType.SPHERE,
        GeoType.ELLIPSOID,
        GeoType.CAPSULE,
        GeoType.CYLINDER,
        GeoType.CONE,
        GeoType.MESH,
        GeoType.CONVEX_MESH,
        GeoType.HFIELD,
    ):
        raise ValueError(
            "Unsupported SPH boundary sampling shape. Supported shapes are GeoType.BOX, GeoType.SPHERE, "
            "GeoType.ELLIPSOID, GeoType.CAPSULE, GeoType.CYLINDER, GeoType.CONE, "
            "GeoType.MESH, GeoType.CONVEX_MESH, and GeoType.HFIELD shapes"
        )

    if material is None:
        material = SPHMaterial()
    material.validate()

    register_sph_custom_attributes(builder)

    shape_body = builder.shape_body[shape]
    shape_xform = builder.shape_transform[shape]
    if shape_body >= 0:
        shape_xform = wp.transform_multiply(builder.body_q[shape_body], shape_xform)

    shape_scale = tuple(float(x) for x in builder.shape_scale[shape])
    if shape_type == GeoType.BOX:
        local_points = _sample_box_surface_points(shape_scale, spacing)
        local_normals = [_box_surface_normal(point, shape_scale) for point in local_points]
    elif shape_type == GeoType.SPHERE:
        local_points = _sample_sphere_surface_points(shape_scale[0], spacing)
        local_normals = [_sphere_surface_normal(point) for point in local_points]
    elif shape_type == GeoType.ELLIPSOID:
        local_points = _sample_ellipsoid_surface_points(shape_scale, spacing)
        local_normals = [_ellipsoid_surface_normal(point, shape_scale) for point in local_points]
    elif shape_type == GeoType.CAPSULE:
        local_points = _sample_capsule_surface_points(shape_scale[0], shape_scale[1], spacing)
        local_normals = [_capsule_surface_normal(point, shape_scale[1]) for point in local_points]
    elif shape_type == GeoType.CYLINDER:
        local_points = _sample_cylinder_surface_points(shape_scale[0], shape_scale[1], spacing)
        local_normals = [_cylinder_surface_normal(point, shape_scale[0], shape_scale[1]) for point in local_points]
    elif shape_type == GeoType.CONE:
        local_points = _sample_cone_surface_points(shape_scale[0], shape_scale[1], spacing)
        local_normals = [_cone_surface_normal(point, shape_scale[0], shape_scale[1]) for point in local_points]
    elif shape_type == GeoType.HFIELD:
        heightfield = builder.shape_source[shape]
        if not isinstance(heightfield, Heightfield):
            raise ValueError("SPH heightfield boundary sampling requires a source heightfield")
        local_points, local_normals = _sample_heightfield_surface_points_and_normals(
            heightfield,
            shape_scale,
            spacing,
        )
    else:
        mesh = builder.shape_source[shape]
        if mesh is None:
            raise ValueError("SPH mesh boundary sampling requires a source mesh")
        local_points, local_normals = _sample_mesh_surface_points_and_normals(mesh, shape_scale, spacing)
    positions = [wp.transform_point(shape_xform, point) for point in local_points]
    normals = [
        _normalize_vec3(wp.quat_rotate(wp.transform_get_rotation(shape_xform), normal)) for normal in local_normals
    ]

    if vel is None:
        vel = wp.vec3(0.0)
    else:
        vel = _coerce_vec3(vel, "boundary sampling velocity")
    if radius is None:
        radius = 0.5 * spacing
    else:
        radius = _finite_positive_scalar(radius, "boundary sampling radius")
    if mass is None:
        masses = _sampled_boundary_contributions(positions, material=material, spacing=spacing, kernel=kernel)
    else:
        masses = [float(_finite_nonnegative_scalar(mass, "boundary sampling mass"))] * len(positions)
    start = len(builder.particle_q)
    attrs = make_sph_particle_attributes(
        material=material,
        role=SPHRole.BOUNDARY,
        boundary_normal=normals,
        custom_attributes=custom_attributes,
    )
    builder.add_particles(
        pos=positions,
        vel=[vel] * len(positions),
        mass=masses,
        radius=[radius] * len(positions),
        custom_attributes=attrs,
    )
    return range(start, len(builder.particle_q))
