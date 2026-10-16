# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""USD import helpers for authored particle simulation geometry."""

from __future__ import annotations

import math
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import warp as wp

from ..sim.builder import ModelBuilder
from . import utils as usd

_MATERIAL_ATTRIBUTES = {
    "newton:mpm:youngsModulus": ("mpm:young_modulus", "pressure"),
    "newton:mpm:poissonsRatio": ("mpm:poisson_ratio", "dimensionless"),
    "newton:mpm:elasticDamping": ("mpm:damping", "pressure_time"),
    "newton:mpm:internalFriction": ("mpm:friction", "dimensionless"),
    "newton:mpm:yieldPressure": ("mpm:yield_pressure", "pressure"),
    "newton:mpm:tensileYieldRatio": ("mpm:tensile_yield_ratio", "dimensionless"),
    "newton:mpm:yieldStress": ("mpm:yield_stress", "pressure"),
    "newton:mpm:viscosity": ("mpm:viscosity", "pressure_time"),
    "newton:mpm:hardening": ("mpm:hardening", "dimensionless"),
    "newton:mpm:hardeningRate": ("mpm:hardening_rate", "dimensionless"),
    "newton:mpm:softeningRate": ("mpm:softening_rate", "dimensionless"),
    "newton:mpm:dilatancy": ("mpm:dilatancy", "dimensionless"),
    "newton:mpm:initialPlasticVolumeStrain": ("mpm:particle_Jp", "dimensionless"),
}

_STANDARD_ELASTIC_ATTRIBUTES = {
    "physics:youngsModulus": ("mpm:young_modulus", "pressure"),
    "physics:poissonsRatio": ("mpm:poisson_ratio", "dimensionless"),
}

_STANDARD_DEFORMABLE_MATERIAL_API = "PhysicsVolumeDeformableMaterialAPI"

_SIMULATOR_DEFAULT_ATTRIBUTES = {
    "newton:mpm:yieldPressure",
    "newton:mpm:youngsModulus",
    "physics:youngsModulus",
}


@dataclass
class _ParticleData:
    """Validated data for one authored particle Points prim."""

    path: str
    positions: list[wp.vec3]
    velocities: list[wp.vec3]
    masses: list[float]
    radii: list[float]
    material: dict[str, float | list[float]]


def _validate_units(linear_unit: float, mass_unit: float) -> None:
    """Validate USD stage units used by the particle conversion."""
    if not math.isfinite(linear_unit) or linear_unit <= 0.0:
        raise ValueError(f"metersPerUnit must be finite and positive, got {linear_unit!r}.")
    if not math.isfinite(mass_unit) or mass_unit <= 0.0:
        raise ValueError(f"kilogramsPerUnit must be finite and positive, got {mass_unit!r}.")


def _convert_to_si(
    value: float,
    *,
    linear_unit: float,
    mass_unit: float,
    distance_power: int,
    path: str,
    name: str,
) -> float:
    """Convert a mass/length-derived value and reject floating-point overflow."""
    with np.errstate(divide="ignore", invalid="ignore", over="ignore", under="ignore"):
        converted = float(np.float64(value) * np.float64(mass_unit) / np.power(np.float64(linear_unit), distance_power))
    if not math.isfinite(converted) or (value != 0.0 and converted == 0.0):
        raise ValueError(f"{path}: {name} does not convert to a finite, representable SI value.")
    return converted


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
    """Read widths using the precedence and interpolation rules of ``UsdGeomPoints``."""
    from pxr import UsdGeom

    prim = points.GetPrim()
    primvars = UsdGeom.PrimvarsAPI(prim)
    primvar = primvars.GetPrimvar("widths")
    if not primvar:
        primvar = primvars.FindPrimvarWithInheritance("widths")

    if primvar and primvar.GetAttr().HasValue():
        value = primvar.ComputeFlattened()
        if value is None:
            raise ValueError(f"{path}: primvars:widths could not be flattened; check its values and indices.")
        interpolation = primvar.GetInterpolation()
    else:
        attr = points.GetWidthsAttr()
        value = attr.Get() if attr and attr.HasAuthoredValue() else None
        interpolation = points.GetWidthsInterpolation()

    if value is None:
        return None
    try:
        widths = np.asarray(value, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{path}: widths must contain numeric values, got {value!r}.") from error
    if widths.size == 0:
        return None
    if interpolation == UsdGeom.Tokens.constant:
        if widths.size != 1:
            raise ValueError(f"{path}: widths with interpolation='constant' must contain one value, got {widths.size}.")
        widths = np.full(count, widths[0], dtype=np.float64)
    elif interpolation in (UsdGeom.Tokens.vertex, UsdGeom.Tokens.varying):
        if widths.size != count:
            raise ValueError(
                f"{path}: widths with interpolation={str(interpolation)!r} must contain {count} values, "
                f"got {widths.size}."
            )
    else:
        raise ValueError(
            f"{path}: widths interpolation must be 'constant', 'vertex', or 'varying', got {interpolation!r}."
        )
    if not np.isfinite(widths).all() or np.any(widths <= 0.0):
        raise ValueError(f"{path}: widths must contain only finite positive values.")
    return widths


def _bound_physics_material(prim):
    """Return the physics-purpose bound material prim, if any."""
    from pxr import UsdShade

    material, relationship = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial("physics")
    if not material or not relationship:
        return None

    # ComputeBoundMaterial("physics") deliberately falls back to an all-purpose
    # binding. MPM material resolution must not turn a render-only binding into
    # a physics binding, so require the relationship itself to carry the
    # physics purpose. This covers inherited direct and collection bindings.
    relationship_name = str(relationship.GetName())
    if relationship_name != "material:binding:physics" and not relationship_name.startswith(
        "material:binding:collection:physics:"
    ):
        return None
    material_prim = material.GetPrim()
    return material_prim if material_prim and material_prim.IsValid() else None


def _read_material_prim(
    material_prim,
    linear_unit: float,
    mass_unit: float,
    default_young_modulus: float,
) -> tuple[dict[str, float], float | None]:
    """Read MPM values and an optional positive density from one material."""
    if material_prim is None:
        return {}, None

    material_path = str(material_prim.GetPath())
    has_newton_mpm_api = usd.has_applied_api_schema(material_prim, "NewtonMPMMaterialAPI")
    has_standard_deformable_api = usd.has_applied_api_schema(material_prim, _STANDARD_DEFORMABLE_MATERIAL_API)
    accepted_api_names = (
        "NewtonMPMMaterialAPI",
        "PhysicsMaterialAPI",
        _STANDARD_DEFORMABLE_MATERIAL_API,
    )
    if not any(usd.has_applied_api_schema(material_prim, api_name) for api_name in accepted_api_names):
        raise ValueError(
            f"{material_path}: a material selected by an MPM physics binding must apply NewtonMPMMaterialAPI, "
            "PhysicsMaterialAPI, or PhysicsVolumeDeformableMaterialAPI."
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
            density = _convert_to_si(
                density_value,
                linear_unit=linear_unit,
                mass_unit=mass_unit,
                distance_power=3,
                path=material_path,
                name="physics:density",
            )

    def authored_number(usd_name: str) -> tuple[bool, float | None]:
        attr = material_prim.GetAttribute(usd_name)
        if not attr or not attr.HasAuthoredValue():
            return False, None
        value = attr.Get()
        if value is None:
            raise ValueError(f"{material_path}: authored attribute {usd_name!r} has no value.")
        try:
            value = float(value)
        except (TypeError, ValueError) as error:
            raise ValueError(f"{material_path}: {usd_name} must be a finite number, got {value!r}.") from error
        if value == float("-inf") and usd_name in _SIMULATOR_DEFAULT_ATTRIBUTES:
            return True, value
        if not math.isfinite(value):
            raise ValueError(f"{material_path}: {usd_name} must be finite, got {value!r}.")
        return True, value

    values: dict[str, float] = {}
    sources: dict[str, str] = {}

    if has_standard_deformable_api:
        for usd_name, (model_name, unit_kind) in _STANDARD_ELASTIC_ATTRIBUTES.items():
            is_authored, value = authored_number(usd_name)
            if not is_authored or value == float("-inf"):
                continue
            assert value is not None
            if unit_kind in ("pressure", "pressure_time"):
                value = _convert_to_si(
                    value,
                    linear_unit=linear_unit,
                    mass_unit=mass_unit,
                    distance_power=1,
                    path=material_path,
                    name=usd_name,
                )
            values[model_name] = value
            sources[model_name] = usd_name

    # Newton's MPM-specific elasticity wins when both APIs are authored. The
    # proposed standard volume material is retained as an input compatibility
    # path, but it is not currently applicable to Points.
    if has_newton_mpm_api:
        for usd_name, (model_name, unit_kind) in _MATERIAL_ATTRIBUTES.items():
            is_authored, value = authored_number(usd_name)
            if not is_authored or value == float("-inf"):
                continue
            assert value is not None
            if unit_kind in ("pressure", "pressure_time"):
                value = _convert_to_si(
                    value,
                    linear_unit=linear_unit,
                    mass_unit=mass_unit,
                    distance_power=1,
                    path=material_path,
                    name=usd_name,
                )
            values[model_name] = value
            sources[model_name] = usd_name

    poisson = values.get("mpm:poisson_ratio")
    if poisson is not None and not -1.0 < poisson <= 0.5:
        source = sources["mpm:poisson_ratio"]
        raise ValueError(f"{material_path}: {source} must be in (-1, 0.5], got {poisson}.")
    for name, value in values.items():
        if name != "mpm:poisson_ratio" and value < 0.0:
            raise ValueError(f"{material_path}: {sources[name]} must be non-negative, got {value}.")
    young_modulus = values.get("mpm:young_modulus")
    for name in ("mpm:tensile_yield_ratio", "mpm:dilatancy"):
        value = values.get(name)
        if value is not None and value > 1.0:
            raise ValueError(f"{material_path}: {name} must not exceed 1.0, got {value}.")

    absolute_damping = values.get("mpm:damping")
    if absolute_damping is not None:
        damping_modulus = young_modulus if young_modulus is not None else default_young_modulus
        if absolute_damping == 0.0:
            normalized_damping = 0.0
        elif not math.isfinite(damping_modulus) or damping_modulus <= 0.0:
            raise ValueError(
                f"{material_path}: non-zero newton:mpm:elasticDamping requires a finite positive Young's modulus."
            )
        else:
            normalized_damping = absolute_damping / damping_modulus
        if not math.isfinite(normalized_damping) or (absolute_damping != 0.0 and normalized_damping == 0.0):
            raise ValueError(
                f"{material_path}: newton:mpm:elasticDamping does not convert to a finite, representable "
                "solver damping value."
            )
        values["mpm:damping"] = normalized_damping

    return values, density


def _read_particle_masses(prim, count: int, path: str, mass_unit: float) -> np.ndarray | None:
    """Read authored per-point masses and convert them to kilograms."""
    attr = prim.GetAttribute("physics:masses")
    if not attr or not attr.HasAuthoredValue():
        return None
    value = attr.Get()
    try:
        masses = np.asarray(value, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{path}: physics:masses must contain numeric values, got {value!r}.") from error
    if masses.size != count:
        raise ValueError(f"{path}: physics:masses must contain {count} values, got {masses.size}.")
    if not np.isfinite(masses).all() or np.any(masses <= 0.0):
        raise ValueError(f"{path}: physics:masses must contain only finite positive values.")
    with np.errstate(invalid="ignore", over="ignore", under="ignore"):
        converted = masses * np.float64(mass_unit)
    if not np.isfinite(converted).all() or np.any((masses != 0.0) & (converted == 0.0)):
        raise ValueError(f"{path}: physics:masses do not convert to finite, representable SI values.")
    return converted


def _read_body_mass_overrides(prim, linear_unit: float, mass_unit: float) -> tuple[float | None, float | None]:
    """Read positive total-mass and density overrides from the deformable body."""
    body_prim = usd._find_deformable_body_prim(prim)
    if body_prim is None:
        return None, None
    path = str(body_prim.GetPath())

    def read(name: str, distance_power: int) -> float | None:
        attr = body_prim.GetAttribute(name)
        if not attr or not attr.HasAuthoredValue():
            return None
        value = attr.Get()
        try:
            value = float(value)
        except (TypeError, ValueError) as error:
            raise ValueError(f"{path}: {name} must be a finite non-negative number, got {value!r}.") from error
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"{path}: {name} must be finite and non-negative, got {value!r}.")
        if value == 0.0:
            return None
        return _convert_to_si(
            value,
            linear_unit=linear_unit,
            mass_unit=mass_unit,
            distance_power=distance_power,
            path=path,
            name=name,
        )

    return read("physics:mass", 0), read("physics:density", 3)


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

    # Supported OpenUSD releases reject ValidateFamily(..., elementType='point')
    # for Points even though point subsets are supported. Validate the collected
    # subsets against the authored point count instead.
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
    default_young_modulus = float(builder.custom_attributes["mpm:young_modulus"].default)
    parent_material = _bound_physics_material(points.GetPrim())
    parent_values, parent_density = _read_material_prim(
        parent_material,
        linear_unit,
        mass_unit,
        default_young_modulus,
    )
    overlays = _point_material_subsets(points, count, path)
    if not overlays:
        return parent_values, parent_density

    model_names = {
        *(model_name for model_name, _unit_kind in _MATERIAL_ATTRIBUTES.values()),
        *(model_name for model_name, _unit_kind in _STANDARD_ELASTIC_ATTRIBUTES.values()),
    }
    defaults = {model_name: float(builder.custom_attributes[model_name].default) for model_name in model_names}
    resolved_parent = defaults.copy()
    if parent_material is not None:
        resolved_parent.update(parent_values)

    material_arrays = {name: np.full(count, value, dtype=np.float64) for name, value in resolved_parent.items()}
    density_values = np.full(count, parent_density if parent_density is not None else np.nan, dtype=np.float64)

    for indices, material_prim in overlays:
        resolved_subset = defaults.copy()
        subset_values, subset_density = _read_material_prim(
            material_prim,
            linear_unit,
            mass_unit,
            default_young_modulus,
        )
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
) -> _ParticleData:
    """Validate and convert one particle Points prim to Newton SI arrays."""
    from pxr import UsdGeom

    path = str(prim.GetPath())
    points = UsdGeom.Points(prim)
    raw_points = points.GetPointsAttr().Get()
    point_count = len(raw_points) if raw_points is not None else 0
    positions_local = _array3(raw_points, point_count, path, "points")
    velocities_local = _array3(points.GetVelocitiesAttr().Get(), point_count, path, "velocities", optional=True)

    with np.errstate(invalid="ignore", over="ignore", under="ignore"):
        stage_world_mat = np.asarray(
            usd.get_transform_matrix(prim, local=False, xform_cache=xform_cache), dtype=np.float64
        ).reshape(4, 4)
        stage_world_mat[:3, :] *= linear_unit
        world_mat = np.asarray(incoming_world_mat, dtype=np.float64).reshape(4, 4) @ stage_world_mat
    if not np.isfinite(world_mat).all():
        raise ValueError(f"{path}: world transform does not convert to finite SI values.")
    linear = world_mat[:3, :3]
    singular_values = np.linalg.svd(linear, compute_uv=False)
    scale = float(np.mean(singular_values))
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError(f"{path}: world transform must have finite non-zero scale.")
    if not np.allclose(singular_values, scale, rtol=1.0e-5, atol=max(1.0e-9, scale * 1.0e-7)):
        raise ValueError(
            f"{path}: particle widths require a uniform, shear-free world transform; "
            f"got principal scales {singular_values.tolist()}."
        )

    with np.errstate(invalid="ignore", over="ignore", under="ignore"):
        positions_world = positions_local @ linear.T + world_mat[:3, 3]
        velocities_world = velocities_local @ linear.T
    if not np.isfinite(positions_world).all() or not np.isfinite(velocities_world).all():
        raise ValueError(f"{path}: transformed points or velocities contain a non-finite value.")

    widths = _read_widths(points, point_count, path)
    if widths is None:
        radii = np.full(point_count, builder.default_particle_radius, dtype=np.float64)
        representative_widths = 2.0 * radii
    else:
        with np.errstate(invalid="ignore", over="ignore", under="ignore"):
            representative_widths = widths * scale
            radii = 0.5 * representative_widths
    if (
        not np.isfinite(representative_widths).all()
        or not np.isfinite(radii).all()
        or np.any(representative_widths <= 0.0)
        or np.any(radii <= 0.0)
    ):
        raise ValueError(f"{path}: particle widths and radii must convert to finite positive SI values.")

    material, density = _read_particle_materials(builder, points, point_count, path, linear_unit, mass_unit)
    masses = _read_particle_masses(prim, point_count, path, mass_unit)
    if masses is None:
        body_mass, body_density = _read_body_mass_overrides(prim, linear_unit, mass_unit)
        default_density = float(builder.default_shape_cfg.density)
        if body_density is not None:
            resolved_density = body_density
        elif isinstance(density, np.ndarray):
            resolved_density = density.copy()
            needs_default = np.isnan(resolved_density)
            if np.any(needs_default):
                if math.isfinite(default_density) and default_density > 0.0:
                    resolved_density[needs_default] = default_density
                elif body_mass is not None and np.all(needs_default):
                    resolved_density[needs_default] = 1.0
                else:
                    raise ValueError(
                        f"{path}: particle mass requires a finite positive bound physics:density "
                        "or builder default density."
                    )
        else:
            resolved_density = density if density is not None else default_density
            if not math.isfinite(resolved_density) or resolved_density <= 0.0:
                if body_mass is not None:
                    resolved_density = 1.0
                else:
                    raise ValueError(
                        f"{path}: particle mass requires a finite positive bound physics:density "
                        "or builder default density."
                    )
        if not np.isfinite(resolved_density).all() or np.any(resolved_density <= 0.0):
            raise ValueError(
                f"{path}: particle mass requires finite positive bound physics:density values or builder defaults."
            )
        with np.errstate(invalid="ignore", over="ignore", under="ignore"):
            masses = resolved_density * representative_widths**3
        if not np.isfinite(masses).all() or np.any(masses <= 0.0):
            raise ValueError(f"{path}: density and particle widths must produce finite positive SI masses.")
        if body_mass is not None:
            mass_sum = float(np.sum(masses, dtype=np.float64))
            if not math.isfinite(mass_sum) or mass_sum <= 0.0:
                raise ValueError(f"{path}: particle weights cannot represent the authored physics:mass.")
            with np.errstate(invalid="ignore", over="ignore", under="ignore"):
                masses = masses * (body_mass / mass_sum)
            if not np.isfinite(masses).all() or np.any(masses <= 0.0):
                raise ValueError(f"{path}: physics:mass does not produce finite positive SI particle masses.")

    return _ParticleData(
        path=path,
        positions=[wp.vec3(*position) for position in positions_world],
        velocities=[wp.vec3(*velocity) for velocity in velocities_world],
        masses=masses.tolist(),
        radii=radii.tolist(),
        material=material,
    )


def _resolve_simulation_owner(prim, default_scene):
    """Resolve and validate the simulation owner on the governing deformable body."""
    body_prim = usd._find_deformable_body_prim(prim)
    if body_prim is None:
        raise ValueError(
            f"{prim.GetPath()}: NewtonPointsDeformableSimAPI requires PhysicsDeformableBodyAPI "
            "on the Points prim or its direct parent."
        )
    relationship = body_prim.GetRelationship("physics:simulationOwner")
    targets = relationship.GetTargets() if relationship else []
    if not targets:
        return default_scene
    if len(targets) != 1:
        raise ValueError(
            f"{body_prim.GetPath()}: physics:simulationOwner must target exactly one PhysicsScene, "
            f"got {[str(target) for target in targets]}."
        )

    forwarded_targets = relationship.GetForwardedTargets()
    if len(forwarded_targets) != 1:
        raise ValueError(
            f"{body_prim.GetPath()}: physics:simulationOwner must resolve to exactly one PhysicsScene, "
            f"got {[str(target) for target in forwarded_targets]}."
        )
    owner = body_prim.GetStage().GetPrimAtPath(forwarded_targets[0])

    from pxr import UsdPhysics

    if not owner or not owner.IsValid() or not owner.IsA(UsdPhysics.Scene):
        raise ValueError(
            f"{body_prim.GetPath()}: physics:simulationOwner target {forwarded_targets[0]} "
            "must resolve to a PhysicsScene."
        )
    return owner


def find_particle_prims(root_prim, ignore_paths: list[str]) -> list[Any]:
    """Find opted-in particle simulation geometry below a USD root prim.

    Args:
        root_prim: Root prim of the USD subtree to traverse.
        ignore_paths: Regular expressions matching prim paths to ignore.

    Returns:
        Points prims carrying ``NewtonPointsDeformableSimAPI`` in traversal order.
    """
    from pxr import Usd, UsdGeom

    if not root_prim or not root_prim.IsValid():
        return []
    particle_prims = []
    for prim in Usd.PrimRange(root_prim, Usd.TraverseInstanceProxies()):
        path = str(prim.GetPath())
        if any(re.match(pattern, path) for pattern in ignore_paths):
            continue
        if prim.IsA(UsdGeom.Points) and usd.has_applied_api_schema(prim, "NewtonPointsDeformableSimAPI"):
            particle_prims.append(prim)
    return particle_prims


def import_particles(
    builder: ModelBuilder,
    root_prim,
    *,
    ignore_paths: list[str],
    xform_cache,
    incoming_world_mat: wp.mat44,
    linear_unit: float,
    mass_unit: float,
    scene_preflight: Callable[[Any], None] | None = None,
    particle_prims: list[Any] | None = None,
) -> tuple[dict[str, tuple[int, int]], Any | None]:
    """Import opted-in particle Points and return ranges plus their owner scene.

    Particle positions, widths, and radii are converted to meters [m],
    velocities to meters per second [m/s], and masses to kilograms [kg].

    ``NewtonPointsDeformableSimAPI`` marks a Points simulation geometry. The
    governing ``PhysicsDeformableBodyAPI`` prim's ``physics:simulationOwner``
    must resolve to a
    ``PhysicsScene`` carrying ``NewtonMPMSceneAPI``. Unauthored owner
    relationships use the first PhysicsScene in stage traversal order, matching
    the standard USD Physics ownership convention.

    Args:
        builder: Builder that receives the imported particles.
        root_prim: Root prim of the USD subtree to import.
        ignore_paths: Regular expressions matching prim paths to ignore.
        xform_cache: USD transform cache used to resolve world transforms.
        incoming_world_mat: Caller-provided world transform whose translation is in meters [m].
        linear_unit: Stage length scale in meters per stage length unit [m].
        mass_unit: Stage mass scale in kilograms per stage mass unit [kg].
        scene_preflight: Optional callback that validates the resolved MPM scene before particle mutation.
        particle_prims: Optional precomputed opted-in Points prims in traversal order.

    Returns:
        A tuple containing prim-path to half-open particle index ranges and the
        governing ``UsdPhysics.Scene`` prim, or ``None`` when no particles are
        imported.
    """
    from pxr import UsdPhysics

    if particle_prims is None:
        particle_prims = find_particle_prims(root_prim, ignore_paths)
    if not particle_prims:
        return {}, None

    stage = root_prim.GetStage()
    scene_prims = [prim for prim in stage.Traverse() if prim.IsA(UsdPhysics.Scene)]
    default_scene = scene_prims[0] if scene_prims else None
    imported_particle_prims = []
    particle_owners = []
    for prim in particle_prims:
        owner = _resolve_simulation_owner(prim, default_scene)
        if owner is None or not usd.has_applied_api_schema(owner, "NewtonMPMSceneAPI"):
            continue
        imported_particle_prims.append(prim)
        particle_owners.append(owner)

    if not imported_particle_prims:
        return {}, None

    scene_paths = {str(owner.GetPath()) for owner in particle_owners}
    if len(scene_paths) != 1:
        raise ValueError(
            "One USD import call cannot combine NewtonPointsDeformableSimAPI Points owned by different "
            f"NewtonMPMSceneAPI scenes; found {sorted(scene_paths)}."
        )
    scene_prim = particle_owners[0]

    _validate_units(linear_unit, mass_unit)

    from ..solvers.implicit_mpm import SolverImplicitMPM  # noqa: PLC0415

    if scene_preflight is not None:
        scene_preflight(scene_prim)
    SolverImplicitMPM.register_custom_attributes(builder)
    payloads: list[_ParticleData] = []
    for prim in imported_particle_prims:
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
    return ranges, scene_prim


__all__ = ["find_particle_prims", "import_particles"]
