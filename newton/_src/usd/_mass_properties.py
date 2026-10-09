# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Interpret and accumulate USD mass properties for one import.

Imported lazily by the USD importer so PXR remains an optional dependency.
"""

from __future__ import annotations

import collections
import math
import warnings
from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np
import warp as wp

from ..core import quat_between_axes
from ..core.types import Axis
from ..geometry import GeoType, Mesh, compute_inertia_shape, transform_inertia
from . import utils as usd

if TYPE_CHECKING:
    from pxr import Usd, UsdPhysics


def _is_enabled_collider(prim: Usd.Prim) -> bool:
    from pxr import UsdPhysics

    if collider := UsdPhysics.CollisionAPI(prim):
        return collider.GetCollisionEnabledAttr().Get()
    return False


class _UsdMassProperties:
    """Keep mass calculations and their records scoped to one import."""

    def __init__(self, stage: Usd.Stage, usd_axis_to_axis: dict, get_mesh_cached: Callable):
        self.stage = stage
        self.usd_axis_to_axis = usd_axis_to_axis
        self.get_mesh_cached = get_mesh_cached
        self.warned_invalid_density: set[str] = set()
        self.warned_invalid_diag_inertia: set[str] = set()
        self.bodies_requiring_mass_properties_fallback: set[str] = set()
        self.rigid_body_mass_info_map = {}
        self.rigid_body_mass_fallback_density = {}
        self.rigid_body_fallback_collider_paths = collections.defaultdict(list)
        self.expected_fallback_collider_paths: set[str] = set()
        self.warned_missing_collider_mass_info: set[str] = set()
        self.zero_mass_information = None

    # UsdPhysics.MassAPI value semantics: a schema fallback value (0 mass/density, zero
    # diagonal inertia or principal axes, non-finite center of mass) means "unspecified"
    # even when explicitly authored, so authoredness must not be used as the override signal.
    # A blocked attribute resolves to no value (Get() returns None) and is also unspecified.
    def effective_mass(self, mass_api: UsdPhysics.MassAPI) -> float | None:
        mass = mass_api.GetMassAttr().Get()
        return float(mass) if mass is not None and math.isfinite(mass) and mass > 0.0 else None

    def effective_density(self, mass_api: UsdPhysics.MassAPI, *, warn_invalid: bool = False) -> float | None:
        raw_density = mass_api.GetDensityAttr().Get()
        if raw_density is not None and math.isfinite(raw_density) and raw_density > 0.0:
            return float(raw_density)
        prim_path = str(mass_api.GetPrim().GetPath())
        if (
            warn_invalid
            and raw_density is not None
            and raw_density != 0.0
            and prim_path not in self.warned_invalid_density
        ):
            self.warned_invalid_density.add(prim_path)
            warnings.warn(
                f"{prim_path}: authored MassAPI density must be positive and finite; treating it as unspecified.",
                stacklevel=2,
            )
        return None

    def effective_diag_inertia(self, mass_api: UsdPhysics.MassAPI):
        diag = mass_api.GetDiagonalInertiaAttr().Get()
        if diag is None or all(v == 0.0 for v in diag):
            return None
        if all(math.isfinite(v) and v >= 0.0 for v in diag):
            return diag
        prim_path = str(mass_api.GetPrim().GetPath())
        if prim_path not in self.warned_invalid_diag_inertia:
            self.warned_invalid_diag_inertia.add(prim_path)
            warnings.warn(
                f"{prim_path}: authored MassAPI diagonalInertia must have finite, nonnegative components; "
                "treating it as unspecified.",
                stacklevel=2,
            )
        return None

    def effective_com(self, mass_api: UsdPhysics.MassAPI):
        com = mass_api.GetCenterOfMassAttr().Get()
        return com if com is not None and all(math.isfinite(v) for v in com) else None

    def effective_principal_axes(self, mass_api: UsdPhysics.MassAPI):
        from pxr import Gf

        axes = mass_api.GetPrincipalAxesAttr().Get()
        return axes if axes is not None and axes != Gf.Quatf(0.0) else None

    # WORKAROUND: UsdPhysicsRigidBodyAPI::ComputeMassProperties reads MassAPI attributes
    # into uninitialized locals (_ParseMassApi/_GetCoM in pxr/usd/usdPhysics/rigidBodyAPI.cpp;
    # usd-core <= 26.3, https://github.com/PixarAnimationStudios/OpenUSD/issues/4155).
    # A blocked attribute makes Get() fail, leaving stack garbage that can pass the
    # authored-value checks and yield nondeterministic mass properties. Supported versions
    # also apply authored mass from disabled colliders after the callback
    # (https://github.com/PixarAnimationStudios/OpenUSD/pull/4164).
    # Bypass ComputeMassProperties for either condition and use recorded enabled colliders.
    # Remove each workaround once the minimum supported usd-core ships its upstream fix.
    # Density is excluded from the blocked-attribute check: it is read into an initialized
    # struct member upstream and blocked density already resolves to "unspecified".
    def _has_blocked_attrs(self, prim: Usd.Prim) -> bool:
        from pxr import UsdPhysics

        mass_api = UsdPhysics.MassAPI(prim)
        if not mass_api:
            return False
        attrs = (
            mass_api.GetMassAttr(),
            mass_api.GetDiagonalInertiaAttr(),
            mass_api.GetPrincipalAxesAttr(),
            mass_api.GetCenterOfMassAttr(),
        )
        return any(attr.GetResolveInfo().ValueIsBlocked() for attr in attrs)

    def requires_recorded_fallback(self, body_prim: Usd.Prim) -> bool:
        """Detect inputs that supported OpenUSD versions cannot aggregate safely."""
        from pxr import Usd, UsdPhysics

        if self._has_blocked_attrs(body_prim):
            return True
        it = iter(Usd.PrimRange(body_prim, Usd.TraverseInstanceProxies()))
        for prim in it:
            if prim != body_prim and prim.HasAPI(UsdPhysics.RigidBodyAPI):
                it.PruneChildren()
                continue
            if prim.HasAPI(UsdPhysics.CollisionAPI):
                if UsdPhysics.MassAPI(prim) and not _is_enabled_collider(prim):
                    # OpenUSD reads authored mass after the callback, so a zero callback
                    # cannot exclude a disabled collider with MassAPI.
                    return True
                if self._has_blocked_attrs(prim):
                    return True
        return False

    def _build_from_effective_properties(
        self,
        prim: Usd.Prim,
        local_pos,
        local_rot,
        shape_geo_type: int,
        shape_scale: wp.vec3,
        shape_src: Mesh | None,
        shape_axis=None,
    ):
        """Build unit-density collider mass information from effective collider MassAPI properties.

        This helper is used for rigid-body fallback mass aggregation via
        ``UsdPhysics.RigidBodyAPI.ComputeMassProperties``. When a collider prim has effective
        ``MassAPI`` mass and diagonal inertia, we convert those values into a
        ``RigidBodyAPI.MassInformation`` payload that represents unit-density collider properties.
        """
        from pxr import Gf, UsdPhysics

        mass_api = UsdPhysics.MassAPI(prim)
        if not mass_api:
            return None

        self.effective_density(mass_api, warn_invalid=True)
        mass = self.effective_mass(mass_api)
        diag_val = self.effective_diag_inertia(mass_api)
        if mass is None or diag_val is None:
            # Warn when an authored override is dropped: mass carries a non-fallback value
            # that is unusable. The 0.0 schema fallback and blocked values stay silent.
            raw_mass = mass_api.GetMassAttr().Get()
            if mass is None and raw_mass is not None and raw_mass != 0.0:
                warnings.warn(
                    f"Skipping collider {prim.GetPath()}: authored MassAPI mass must be positive and finite "
                    "to derive volume and density.",
                    stacklevel=2,
                )
            return None

        shape_volume, _, _ = compute_inertia_shape(shape_geo_type, shape_scale, shape_src, density=1.0)
        if shape_volume <= 0.0:
            warnings.warn(
                f"Skipping collider {prim.GetPath()}: unable to derive positive collider volume from authored shape parameters.",
                stacklevel=2,
            )
            return None
        density = mass / shape_volume
        if density <= 0.0:
            warnings.warn(
                f"Skipping collider {prim.GetPath()}: derived density from authored mass is non-positive.",
                stacklevel=2,
            )
            return None

        inertia_diag_unit = np.array(diag_val, dtype=np.float32) / density

        principal_axes = self.effective_principal_axes(mass_api)
        if principal_axes is None:
            principal_axes = Gf.Quatf(1.0, 0.0, 0.0, 0.0)
        center_of_mass = self.effective_com(mass_api)
        if center_of_mass is None:
            center_of_mass = Gf.Vec3f(0.0, 0.0, 0.0)

        i_rot = usd.value_to_warp(principal_axes)
        rot = np.array(wp.quat_to_matrix(i_rot), dtype=np.float32).reshape(3, 3)
        inertia_full_unit = rot @ np.diag(inertia_diag_unit) @ rot.T

        mass_info = UsdPhysics.RigidBodyAPI.MassInformation()
        mass_info.volume = float(shape_volume)
        mass_info.centerOfMass = center_of_mass
        mass_info.localPos = Gf.Vec3f(*local_pos)
        mass_info.localRot = self._resolve_local_rotation(local_rot, shape_geo_type, shape_axis)
        mass_info.inertia = Gf.Matrix3f(*inertia_full_unit.flatten().tolist())
        return mass_info

    def _resolve_local_rotation(self, local_rot, shape_geo_type: int, shape_axis):
        """Match collider mass frame rotation with shape axis correction used by shape insertion."""
        from pxr import Gf, UsdPhysics

        if shape_geo_type not in {GeoType.CAPSULE, GeoType.CYLINDER, GeoType.CONE} or shape_axis is None:
            return local_rot

        axis = self.usd_axis_to_axis.get(shape_axis)
        if axis is None:
            axis_int_map = {
                int(UsdPhysics.Axis.X): Axis.X,
                int(UsdPhysics.Axis.Y): Axis.Y,
                int(UsdPhysics.Axis.Z): Axis.Z,
            }
            axis = axis_int_map.get(int(shape_axis))
        if axis is None or axis == Axis.Z:
            return local_rot

        local_rot_wp = usd.value_to_warp(local_rot)
        corrected_rot = wp.mul(local_rot_wp, quat_between_axes(Axis.Z, axis))
        return Gf.Quatf(
            float(corrected_rot[3]),
            float(corrected_rot[0]),
            float(corrected_rot[1]),
            float(corrected_rot[2]),
        )

    def _build_from_shape_geometry(
        self,
        prim: Usd.Prim,
        local_pos,
        local_rot,
        shape_geo_type: int,
        shape_scale: wp.vec3,
        shape_src: Mesh | None,
        shape_axis=None,
        is_solid: bool = True,
        thickness: float = 0.0,
    ):
        """Build unit-density collider mass information from geometric shape parameters.

        This fallback path derives collider volume, center of mass, and inertia from shape
        geometry (box/sphere/capsule/cylinder/cone/mesh) when collider-authored MassAPI mass
        properties are not available.
        """
        from pxr import Gf, UsdPhysics

        shape_mass, shape_com, shape_inertia = compute_inertia_shape(
            shape_geo_type, shape_scale, shape_src, density=1.0, is_solid=is_solid, thickness=thickness
        )
        if shape_mass <= 0.0:
            warnings.warn(
                f"Skipping collider {prim.GetPath()} in mass aggregation: unable to derive positive unit-density mass.",
                stacklevel=2,
            )
            return None

        shape_inertia_np = np.array(shape_inertia, dtype=np.float32).reshape(3, 3)
        mass_info = UsdPhysics.RigidBodyAPI.MassInformation()
        mass_info.volume = float(shape_mass)
        mass_info.centerOfMass = Gf.Vec3f(*shape_com)
        mass_info.localPos = Gf.Vec3f(*local_pos)
        mass_info.localRot = self._resolve_local_rotation(local_rot, shape_geo_type, shape_axis)
        mass_info.inertia = Gf.Matrix3f(*shape_inertia_np.flatten().tolist())
        return mass_info

    def record_collider(
        self,
        path: str,
        prim: Usd.Prim,
        shape_spec,
        shape_type,
        *,
        density: float,
        is_solid: bool,
        thickness: float,
        mesh_source: Mesh | None = None,
    ):
        """Record collider mass information used by the rigid-body fallback callback."""
        from pxr import UsdPhysics

        body_path = str(shape_spec.rigidBody)
        if body_path not in self.bodies_requiring_mass_properties_fallback or not _is_enabled_collider(prim):
            return

        shape_geo_type = None
        shape_scale = wp.vec3(1.0, 1.0, 1.0)
        shape_src = None
        if shape_type == UsdPhysics.ObjectType.CubeShape:
            shape_geo_type = GeoType.BOX
            hx, hy, hz = shape_spec.halfExtents
            shape_scale = wp.vec3(hx, hy, hz)
        elif shape_type == UsdPhysics.ObjectType.SphereShape:
            shape_geo_type = GeoType.SPHERE
            shape_scale = wp.vec3(shape_spec.radius, 0.0, 0.0)
        elif shape_type == UsdPhysics.ObjectType.CapsuleShape:
            shape_geo_type = GeoType.CAPSULE
            shape_scale = wp.vec3(shape_spec.radius, shape_spec.halfHeight, 0.0)
        elif shape_type == UsdPhysics.ObjectType.CylinderShape:
            shape_geo_type = GeoType.CYLINDER
            shape_scale = wp.vec3(shape_spec.radius, shape_spec.halfHeight, 0.0)
        elif shape_type == UsdPhysics.ObjectType.ConeShape:
            shape_geo_type = GeoType.CONE
            shape_scale = wp.vec3(shape_spec.radius, shape_spec.halfHeight, 0.0)
        elif shape_type == UsdPhysics.ObjectType.MeshShape:
            shape_geo_type = GeoType.MESH
            shape_scale = wp.vec3(*shape_spec.meshScale)
            # Visual meshes retain source mass properties; reuse those without
            # treating expanded visual topology as a geometry-only cache entry.
            shape_src = mesh_source if mesh_source is not None else self.get_mesh_cached(prim)
        if shape_geo_type is None:
            return

        self.expected_fallback_collider_paths.add(path)
        shape_axis = getattr(shape_spec, "axis", None)
        mass_info = self._build_from_effective_properties(
            prim,
            shape_spec.localPos,
            shape_spec.localRot,
            shape_geo_type,
            shape_scale,
            shape_src,
            shape_axis,
        )
        if mass_info is None:
            mass_info = self._build_from_shape_geometry(
                prim,
                shape_spec.localPos,
                shape_spec.localRot,
                shape_geo_type,
                shape_scale,
                shape_src,
                shape_axis,
                is_solid=is_solid,
                thickness=thickness,
            )
        if mass_info is not None:
            if path not in self.rigid_body_mass_info_map:
                self.rigid_body_fallback_collider_paths[body_path].append(path)
            self.rigid_body_mass_info_map[path] = mass_info
            self.rigid_body_mass_fallback_density[path] = density

    def _create_zero_mass_information(self):
        """Create a reusable zero-contribution collider mass payload for callback fallback."""
        from pxr import Gf, UsdPhysics

        mass_info = UsdPhysics.RigidBodyAPI.MassInformation()
        mass_info.volume = 0.0
        mass_info.centerOfMass = Gf.Vec3f(0.0)
        mass_info.localPos = Gf.Vec3f(0.0)
        mass_info.localRot = Gf.Quatf(1.0, 0.0, 0.0, 0.0)
        mass_info.inertia = Gf.Matrix3f(0.0)
        return mass_info

    def get_collision_mass_information(self, collider_prim: Usd.Prim):
        """MassInformation callback for ``ComputeMassProperties`` with one-time warning on misses."""
        if not _is_enabled_collider(collider_prim):
            return self.zero_mass_information
        collider_path = str(collider_prim.GetPath())
        is_expected_missing = (
            collider_path in self.expected_fallback_collider_paths
            and collider_path not in self.rigid_body_mass_info_map
        )
        if is_expected_missing and collider_path not in self.warned_missing_collider_mass_info:
            warnings.warn(
                f"Skipping collider {collider_path} in mass aggregation: missing usable collider mass information.",
                stacklevel=2,
            )
            self.warned_missing_collider_mass_info.add(collider_path)
        return self.rigid_body_mass_info_map.get(collider_path, self.zero_mass_information)

    def aggregate_recorded(self, body_path: str, body_density: float | None):
        """Aggregate callback mass data when OpenUSD cannot traverse the colliders."""
        from pxr import UsdPhysics

        total_mass = 0.0
        total_com = wp.vec3(0.0)
        total_inertia = wp.mat33(0.0)
        found = False
        for collider_path in self.rigid_body_fallback_collider_paths.get(body_path, ()):
            mass_info = self.rigid_body_mass_info_map[collider_path]
            shape_density = self.rigid_body_mass_fallback_density[collider_path]
            # The recording helpers reject nonpositive unit-density mass.
            volume = float(mass_info.volume)
            collider_prim = self.stage.GetPrimAtPath(collider_path)
            collider_mass_api = UsdPhysics.MassAPI(collider_prim)
            collider_mass = self.effective_mass(collider_mass_api) if collider_mass_api else None
            collider_density = self.effective_density(collider_mass_api) if collider_mass_api else None
            density = collider_mass / volume if collider_mass is not None else collider_density
            if density is None:
                density = body_density if body_density is not None else shape_density

            mass = density * volume
            local_rot = usd.value_to_warp(mass_info.localRot)
            local_xform = wp.transform(wp.vec3(*mass_info.localPos), local_rot)
            com = wp.transform_point(local_xform, wp.vec3(*mass_info.centerOfMass))
            inertia = wp.mat33(np.array(mass_info.inertia, dtype=np.float32).reshape(3, 3) * density)

            new_mass = total_mass + mass
            new_com = (total_com * total_mass + com * mass) / new_mass
            total_inertia = transform_inertia(
                total_mass, total_inertia, new_com - total_com, wp.quat_identity()
            ) + transform_inertia(mass, inertia, new_com - com, local_rot)
            total_mass = new_mass
            total_com = new_com
            found = True

        if not found:
            return None
        return total_mass, total_inertia, total_com
