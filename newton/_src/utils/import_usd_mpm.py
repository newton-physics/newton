# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""USD import helpers for authored implicit-MPM particles."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any

import numpy as np
import warp as wp

from ..sim.builder import ModelBuilder
from ..usd import utils as usd

_MATERIAL_ATTRIBUTES = {
    "newton:mpm:youngModulus": ("mpm:young_modulus", "pressure"),
    "newton:mpm:poissonRatio": ("mpm:poisson_ratio", "dimensionless"),
    "newton:mpm:damping": ("mpm:damping", "time"),
    "newton:mpm:friction": ("mpm:friction", "dimensionless"),
    "newton:mpm:yieldPressure": ("mpm:yield_pressure", "pressure"),
    "newton:mpm:tensileYieldRatio": ("mpm:tensile_yield_ratio", "dimensionless"),
    "newton:mpm:yieldStress": ("mpm:yield_stress", "pressure"),
    "newton:mpm:viscosity": ("mpm:viscosity", "pressure_time"),
    "newton:mpm:hardening": ("mpm:hardening", "dimensionless"),
    "newton:mpm:hardeningRate": ("mpm:hardening_rate", "dimensionless"),
    "newton:mpm:softeningRate": ("mpm:softening_rate", "dimensionless"),
    "newton:mpm:dilatancy": ("mpm:dilatancy", "dimensionless"),
}


@dataclass
class _MPMParticleData:
    """Validated data for one authored MPM Points prim."""

    path: str
    positions: list[wp.vec3]
    velocities: list[wp.vec3]
    masses: list[float]
    radii: list[float]
    material: dict[str, float | list[float]]


def _validate_units(linear_unit: float, mass_unit: float) -> None:
    """Validate USD stage units used by the MPM conversion."""
    if not math.isfinite(linear_unit) or linear_unit <= 0.0:
        raise ValueError(f"metersPerUnit must be finite and positive, got {linear_unit!r}.")
    if not math.isfinite(mass_unit) or mass_unit <= 0.0:
        raise ValueError(f"kilogramsPerUnit must be finite and positive, got {mass_unit!r}.")


def _array3(value: Any, count: int, path: str, name: str, *, optional: bool = False) -> np.ndarray:
    """Return a finite ``(count, 3)`` array or an optional zero array."""
    if value is None or len(value) == 0:
        if optional:
            return np.zeros((count, 3), dtype=np.float64)
        raise ValueError(f"{path}: {name} must contain at least one element.")
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (count, 3):
        raise ValueError(f"{path}: {name} must have shape ({count}, 3), got {array.shape}.")
    if not np.isfinite(array).all():
        raise ValueError(f"{path}: {name} contains a non-finite value.")
    return array


def _read_widths(points, count: int, path: str) -> np.ndarray | None:
    """Read authored widths, honoring a ``primvars:widths`` override."""
    from pxr import UsdGeom

    prim = points.GetPrim()
    primvar = UsdGeom.PrimvarsAPI(prim).GetPrimvar("widths")
    if primvar and primvar.GetAttr().HasAuthoredValue():
        value = primvar.ComputeFlattened()
    else:
        attr = points.GetWidthsAttr()
        value = attr.Get() if attr and attr.HasAuthoredValue() else None

    if value is None or len(value) == 0:
        return None
    widths = np.asarray(value, dtype=np.float64).reshape(-1)
    if widths.size == 1:
        widths = np.full(count, widths[0], dtype=np.float64)
    elif widths.size != count:
        raise ValueError(f"{path}: widths must contain one or {count} values, got {widths.size}.")
    if not np.isfinite(widths).all() or np.any(widths <= 0.0):
        raise ValueError(f"{path}: widths must contain only finite positive values.")
    return widths


def _bound_physics_material(prim):
    """Return the physics-purpose bound material prim, if any."""
    from pxr import UsdShade

    material, _relationship = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial("physics")
    if not material:
        return None
    material_prim = material.GetPrim()
    return material_prim if material_prim and material_prim.IsValid() else None


def _read_material_prim(
    material_prim,
    linear_unit: float,
    mass_unit: float,
) -> tuple[dict[str, float], float | None]:
    """Read MPM values and an optional positive density from one material."""
    if material_prim is None:
        return {}, None

    material_path = str(material_prim.GetPath())
    if not usd.has_applied_api_schema(material_prim, "NewtonMPMMaterialAPI"):
        raise ValueError(
            f"{material_path}: a material selected by an MPM physics binding must apply NewtonMPMMaterialAPI."
        )

    density = None
    density_attr = material_prim.GetAttribute("physics:density")
    density_value = density_attr.Get() if density_attr else None
    if density_value is not None:
        density_value = float(density_value)
        if not math.isfinite(density_value):
            raise ValueError(f"{material_path}: physics:density must be finite, got {density_value!r}.")
        if density_attr.HasAuthoredValue() and density_value < 0.0:
            raise ValueError(f"{material_path}: authored physics:density must be non-negative, got {density_value}.")
        if density_value > 0.0:
            density = density_value * mass_unit / (linear_unit**3)

    stress_scale = mass_unit / linear_unit
    values: dict[str, float] = {}
    for usd_name, (model_name, unit_kind) in _MATERIAL_ATTRIBUTES.items():
        attr = material_prim.GetAttribute(usd_name)
        if not attr or not attr.HasAuthoredValue():
            continue
        value = attr.Get()
        if value is None:
            continue
        try:
            value = float(value)
        except (TypeError, ValueError) as error:
            raise ValueError(f"{material_path}: {usd_name} must be a finite number, got {value!r}.") from error
        if not math.isfinite(value):
            raise ValueError(f"{material_path}: {usd_name} must be finite, got {value!r}.")
        if unit_kind in ("pressure", "pressure_time"):
            value *= stress_scale
        values[model_name] = value

    poisson = values.get("mpm:poisson_ratio")
    if poisson is not None and not -1.0 < poisson < 0.5:
        raise ValueError(f"{material_path}: newton:mpm:poissonRatio must be in (-1, 0.5), got {poisson}.")
    for name, value in values.items():
        if name != "mpm:poisson_ratio" and value < 0.0:
            raise ValueError(f"{material_path}: {name} must be non-negative, got {value}.")
    young_modulus = values.get("mpm:young_modulus")
    if young_modulus is not None and young_modulus == 0.0:
        raise ValueError(f"{material_path}: newton:mpm:youngModulus must be positive.")
    for name in ("mpm:tensile_yield_ratio", "mpm:dilatancy"):
        value = values.get(name)
        if value is not None and value > 1.0:
            raise ValueError(f"{material_path}: {name} must not exceed 1.0, got {value}.")

    return values, density


def _direct_physics_material(prim):
    """Return the material from a direct physics-purpose binding, if any."""
    from pxr import UsdShade

    material = UsdShade.MaterialBindingAPI(prim).GetDirectBinding("physics").GetMaterial()
    if not material:
        return None
    material_prim = material.GetPrim()
    return material_prim if material_prim and material_prim.IsValid() else None


def _point_material_subsets(points, count: int, path: str):
    """Return validated point material subsets carrying direct physics bindings."""
    from pxr import UsdGeom, UsdShade

    material_subsets = list(UsdShade.MaterialBindingAPI(points.GetPrim()).GetMaterialBindSubsets())
    if not material_subsets:
        return []

    point_subsets = []
    overlays = []
    for subset in material_subsets:
        material_prim = _direct_physics_material(subset.GetPrim())
        element_type = subset.GetElementTypeAttr().Get()
        if element_type != UsdGeom.Tokens.point:
            if material_prim is not None:
                raise ValueError(
                    f"{path}: physics material subset {subset.GetPath()} must have elementType='point', "
                    f"got {element_type!r}."
                )
            continue

        point_subsets.append(subset)
        if material_prim is not None:
            overlays.append((subset, material_prim))

    if not overlays:
        return []

    family_type = UsdGeom.Subset.GetFamilyType(points, UsdShade.Tokens.materialBind)
    if family_type not in (UsdGeom.Tokens.nonOverlapping, UsdGeom.Tokens.partition):
        raise ValueError(
            f"{path}: point material subsets must declare the materialBind family as "
            f"'nonOverlapping' or 'partition', got {family_type!r}."
        )

    # USD 26.3 rejects ValidateFamily(..., elementType='point') for Points even
    # though point subsets are supported. Validate the collected subsets against
    # the authored point count instead.
    valid, reason = UsdGeom.Subset.ValidateSubsets(point_subsets, count, family_type)
    if not valid:
        raise ValueError(f"{path}: invalid point material subsets: {reason}")

    return [
        (np.asarray(subset.GetIndicesAttr().Get(), dtype=np.int64).reshape(-1), material_prim)
        for subset, material_prim in overlays
    ]


def _read_particle_materials(
    builder: ModelBuilder,
    points,
    count: int,
    path: str,
    linear_unit: float,
    mass_unit: float,
) -> tuple[dict[str, float | list[float]], float | np.ndarray | None]:
    """Resolve the whole-prim material and point-subset material assignments."""
    parent_material = _bound_physics_material(points.GetPrim())
    parent_values, parent_density = _read_material_prim(parent_material, linear_unit, mass_unit)
    overlays = _point_material_subsets(points, count, path)
    if not overlays:
        return parent_values, parent_density

    defaults = {
        model_name: float(builder.custom_attributes[model_name].default)
        for model_name, _unit_kind in _MATERIAL_ATTRIBUTES.values()
    }
    resolved_parent = defaults.copy()
    if parent_material is not None:
        resolved_parent.update(parent_values)

    material_arrays = {name: np.full(count, value, dtype=np.float64) for name, value in resolved_parent.items()}
    density_values = np.full(count, parent_density if parent_density is not None else np.nan, dtype=np.float64)

    for indices, material_prim in overlays:
        resolved_subset = defaults.copy()
        subset_values, subset_density = _read_material_prim(material_prim, linear_unit, mass_unit)
        resolved_subset.update(subset_values)
        for name, value in resolved_subset.items():
            material_arrays[name][indices] = value
        density_values[indices] = subset_density if subset_density is not None else np.nan

    return {name: values.tolist() for name, values in material_arrays.items()}, density_values


def _read_particle_data(
    builder: ModelBuilder,
    prim,
    *,
    xform_cache,
    incoming_world_mat: wp.mat44,
    linear_unit: float,
    mass_unit: float,
) -> _MPMParticleData:
    """Validate and convert one MPM Points prim to Newton SI arrays."""
    from pxr import UsdGeom

    path = str(prim.GetPath())
    points = UsdGeom.Points(prim)
    raw_points = points.GetPointsAttr().Get()
    point_count = len(raw_points) if raw_points is not None else 0
    positions_local = _array3(raw_points, point_count, path, "points")
    velocities_local = _array3(points.GetVelocitiesAttr().Get(), point_count, path, "velocities", optional=True)

    stage_world_mat = np.asarray(
        usd.get_transform_matrix(prim, local=False, xform_cache=xform_cache), dtype=np.float64
    ).reshape(4, 4)
    stage_world_mat[:3, :] *= linear_unit
    world_mat = np.asarray(incoming_world_mat, dtype=np.float64).reshape(4, 4) @ stage_world_mat
    linear = world_mat[:3, :3]
    singular_values = np.linalg.svd(linear, compute_uv=False)
    scale = float(np.mean(singular_values))
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError(f"{path}: world transform must have finite non-zero scale.")
    if not np.allclose(singular_values, scale, rtol=1.0e-5, atol=max(1.0e-9, scale * 1.0e-7)):
        raise ValueError(
            f"{path}: MPM particle widths require a uniform, shear-free world transform; "
            f"got principal scales {singular_values.tolist()}."
        )

    positions_world = positions_local @ linear.T + world_mat[:3, 3]
    velocities_world = velocities_local @ linear.T
    if not np.isfinite(positions_world).all() or not np.isfinite(velocities_world).all():
        raise ValueError(f"{path}: transformed points or velocities contain a non-finite value.")

    widths = _read_widths(points, point_count, path)
    if widths is None:
        radii = np.full(point_count, builder.default_particle_radius, dtype=np.float64)
        representative_widths = 2.0 * radii
    else:
        representative_widths = widths * scale
        radii = 0.5 * representative_widths

    material, density = _read_particle_materials(builder, points, point_count, path, linear_unit, mass_unit)
    default_density = float(builder.default_shape_cfg.density)
    if isinstance(density, np.ndarray):
        resolved_density = density.copy()
        needs_default = np.isnan(resolved_density)
        if np.any(needs_default):
            if not math.isfinite(default_density) or default_density <= 0.0:
                raise ValueError(
                    f"{path}: particle mass requires a finite positive bound physics:density "
                    "or builder default density."
                )
            resolved_density[needs_default] = default_density
    else:
        resolved_density = density if density is not None else default_density
        if not math.isfinite(resolved_density) or resolved_density <= 0.0:
            raise ValueError(
                f"{path}: particle mass requires a finite positive bound physics:density or builder default density."
            )
    masses = resolved_density * representative_widths**3

    return _MPMParticleData(
        path=path,
        positions=[wp.vec3(*position) for position in positions_world],
        velocities=[wp.vec3(*velocity) for velocity in velocities_world],
        masses=masses.tolist(),
        radii=radii.tolist(),
        material=material,
    )


def import_mpm_particles(
    builder: ModelBuilder,
    root_prim,
    *,
    ignore_paths: list[str],
    xform_cache,
    incoming_world_mat: wp.mat44,
    linear_unit: float,
    mass_unit: float,
) -> dict[str, tuple[int, int]]:
    """Import API-opted-in Points prims and return their half-open ranges."""
    from pxr import Usd, UsdGeom

    _validate_units(linear_unit, mass_unit)
    particle_prims = []
    for prim in Usd.PrimRange(root_prim, Usd.TraverseInstanceProxies()):
        path = str(prim.GetPath())
        if any(re.match(pattern, path) for pattern in ignore_paths):
            continue
        if not prim.IsA(UsdGeom.Points) or not usd.has_applied_api_schema(prim, "NewtonMPMParticleAPI"):
            continue
        particle_prims.append(prim)

    if not particle_prims:
        return {}

    from ..solvers.implicit_mpm import SolverImplicitMPM  # noqa: PLC0415

    SolverImplicitMPM.register_custom_attributes(builder)
    payloads: list[_MPMParticleData] = []
    for prim in particle_prims:
        payloads.append(
            _read_particle_data(
                builder,
                prim,
                xform_cache=xform_cache,
                incoming_world_mat=incoming_world_mat,
                linear_unit=linear_unit,
                mass_unit=mass_unit,
            )
        )

    ranges: dict[str, tuple[int, int]] = {}
    for payload in payloads:
        start = builder.particle_count
        builder.add_particles(
            pos=payload.positions,
            vel=payload.velocities,
            mass=payload.masses,
            radius=payload.radii,
            custom_attributes=payload.material,
        )
        ranges[payload.path] = (start, builder.particle_count)
    return ranges


__all__ = ["import_mpm_particles"]
