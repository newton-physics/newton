# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""USD interchange for the experimental scene authoring application.

OpenUSD is imported only when these helpers are called. Imported builders are
owned by the caller and must be treated as read-only templates.
"""

from __future__ import annotations

import json
import math
import os
import tempfile
from collections.abc import Iterable
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit
from urllib.request import url2pathname

import numpy as np
import warp as wp

from ..geometry.bvh import (
    compute_box_bounds,
    compute_capsule_bounds,
    compute_cone_bounds,
    compute_cylinder_bounds,
    compute_ellipsoid_bounds,
    compute_shape_bounds,
    compute_sphere_bounds,
)
from ..geometry.flags import ShapeFlags
from ..geometry.types import GeoType, Heightfield, Mesh
from ..sim.builder import ModelBuilder
from ..utils.color import color_srgb_to_linear

_MARKER = "newtonSceneAuthoring"
_GROUND_FRICTION = 0.8
_SPHERE_ROLLING_FRICTION = 0.005


class NotAuthoringSceneError(ValueError):
    """Indicate an ordinary USD asset rather than an authored scene document."""


def _make_instances_editable(stage):
    from pxr import Usd

    paths = [prim.GetPath() for prim in stage.Traverse(Usd.TraverseInstanceProxies()) if prim.IsInstanceable()]
    for path in paths:
        stage.GetPrimAtPath(path).SetInstanceable(False)


def _ensure_articulation_roots(stage):
    """Give unowned joint networks standard roots for native runtime import."""
    from pxr import UsdPhysics

    roots = {prim.GetPath() for prim in stage.Traverse() if prim.HasAPI(UsdPhysics.ArticulationRootAPI)}
    rigid_bodies = {prim.GetPath() for prim in stage.Traverse() if prim.HasAPI(UsdPhysics.RigidBodyAPI)}
    neighbors = {}
    joints = {}
    children = set()
    world_joints = {}

    def rigid_body(targets):
        if len(targets) != 1:
            return None
        prim = stage.GetPrimAtPath(targets[0])
        while prim and not prim.IsPseudoRoot():
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                return prim.GetPath()
            prim = prim.GetParent()
        return None

    for prim in stage.Traverse():
        if not prim.IsA(UsdPhysics.Joint):
            continue
        joint = UsdPhysics.Joint(prim)
        if not joint.GetJointEnabledAttr().Get():
            continue
        parent = rigid_body(joint.GetBody0Rel().GetTargets())
        child = rigid_body(joint.GetBody1Rel().GetTargets())
        bodies = {body for body in (parent, child) if body is not None}
        for body in bodies:
            neighbors.setdefault(body, set()).update(bodies - {body})
            joints.setdefault(body, set()).add(prim.GetPath())
        if child is not None:
            children.add(child)
            if parent is None:
                world_joints[child] = prim.GetPath()
    pending = set(neighbors)
    while pending:
        component = set()
        stack = [min(pending)]
        while stack:
            body = stack.pop()
            if body not in component:
                component.add(body)
                stack.extend(neighbors[body] - component)
        pending -= component
        paths = component | set().union(*(joints[body] for body in component))
        if any(path.HasPrefix(root) for path in paths for root in roots):
            continue
        ordered = sorted(paths)
        ancestor = ordered[0]
        for path in ordered[1:]:
            ancestor = ancestor.GetCommonPrefix(path)
        if (
            ancestor.IsAbsoluteRootPath()
            or any(root.HasPrefix(ancestor) for root in roots)
            or any(body.HasPrefix(ancestor) for body in rigid_bodies - component)
        ):
            # Keep independent components separate: the native importer expects
            # one kinematic tree per root. USD permits a floating root body or
            # the fixed joint that anchors the component to the world.
            anchored = sorted(body for body in component if body in world_joints)
            ancestor = world_joints[anchored[0]] if anchored else min(component - children or component)
        UsdPhysics.ArticulationRootAPI.Apply(stage.GetPrimAtPath(ancestor))
        roots.add(ancestor)


def _open_asset(source: str):
    from pxr import Sdf, Usd, UsdGeom, UsdPhysics

    path = str(Path(source).expanduser().resolve())
    layer = Sdf.Layer.FindOrOpen(path)
    if layer is None:
        raise ValueError(f"Cannot open USD asset: {source}")
    session = Sdf.Layer.CreateAnonymous("authoring-session.usda")
    stage = Usd.Stage.Open(layer, session, Usd.Stage.LoadAll)
    stage.SetEditTarget(session)
    _make_instances_editable(stage)
    # SimReady assets ship independent backend variants. Select them in the
    # session layer, keeping downloaded packages and their defaults untouched.
    for prim in list(stage.Traverse()):
        for name in ("Newton", "MuJoCo", "PhysX"):
            variants = prim.GetVariantSets().GetVariantSet(name)
            names = variants.GetVariantNames()
            selection = "Enabled" if name == "Newton" else "Disabled"
            if selection in names:
                variants.SetVariantSelection(selection)
    _make_instances_editable(stage)
    for prim in stage.Traverse():
        if prim.IsA(UsdGeom.Xformable) and UsdGeom.Xformable(prim).GetResetXformStack():
            raise ValueError(f"Scene authoring does not support resetXformStack at {prim.GetPath()}")
        schemas = prim.GetPrimTypeInfo().GetAppliedAPISchemas()
        # Bare TetMesh has legacy soft-body semantics in native import. Points
        # and curves require an explicit simulation API; untagged grasp guides
        # and other annotations are harmless rigid-asset metadata.
        if prim.GetTypeName() == "TetMesh" or any("Deformable" in schema or "Particle" in schema for schema in schemas):
            raise ValueError(
                f"Scene authoring supports rigid USD assets only; unsupported simulation at {prim.GetPath()}"
            )
    linear_unit = UsdGeom.GetStageMetersPerUnit(stage)
    mass_unit = UsdPhysics.GetStageKilogramsPerUnit(stage)
    if not math.isfinite(linear_unit) or linear_unit <= 0 or not math.isfinite(mass_unit) or mass_unit <= 0:
        raise ValueError("USD length and mass units must be finite and positive")
    # Native rigid import does not convert stage units. Visual/static geometry
    # can be converted by a world scale; articulated or authored inertial data
    # requires a complete physical-parameter conversion, which is not supported.
    if not math.isclose(linear_unit, 1.0) or not math.isclose(mass_unit, 1.0):
        for prim in stage.Traverse():
            if prim.HasAPI(UsdPhysics.RigidBodyAPI) or prim.HasAPI(UsdPhysics.MassAPI) or prim.IsA(UsdPhysics.Joint):
                raise ValueError(
                    "USD assets with rigid-body, joint, or mass data must use metersPerUnit=1 and kilogramsPerUnit=1; "
                    f"found {linear_unit:g} and {mass_unit:g}"
                )
    # Jointed props may omit articulation metadata while remaining valid USD.
    # The owned runtime needs complete kinematic trees; let the native parser
    # choose roots, order joints, and preserve loop constraints using standard API.
    _ensure_articulation_roots(stage)
    return stage, path, linear_unit


def _asset_dependencies(source: str | Path) -> tuple[str, ...]:
    """Collect local USD dependencies, including anchored missing optional assets.

    Use the native recursive dependency scanner without modifying layers. Each
    package member watches its containing archive; nonlocal URLs are excluded.
    The owner layer anchors unresolved paths so later file creation is visible.
    """
    from pxr import Ar, Sdf, UsdUtils

    source = str(Path(source).expanduser().resolve())
    paths = set()

    def remember(value, layer=None):
        if not value or value.startswith("anon:"):
            return
        parsed = urlsplit(value)
        if parsed.scheme == "file" and parsed.netloc in ("", "localhost"):
            value = url2pathname(parsed.path)
        elif parsed.scheme and not Path(value).is_absolute():
            return
        if Ar.IsPackageRelativePath(value):
            value = Ar.SplitPackageRelativePathOuter(value)[0]
        path = Path(value).expanduser()
        if not path.is_absolute():
            owner = layer.realPath if layer is not None else source
            if Ar.IsPackageRelativePath(owner):
                path = Path(Ar.SplitPackageRelativePathOuter(owner)[0])
            else:
                path = Path(owner).parent / path
        paths.add(str(path.resolve()))

    def inspect(layer, dependency):
        for value in (dependency.assetPath, *dependency.dependencies):
            remember(Sdf.ComputeAssetPathRelativeToLayer(layer, value), layer)
        return dependency

    remember(source)
    layers, assets, _ = UsdUtils.ComputeAllDependencies(source, inspect)
    for layer in layers:
        remember(layer.realPath)
    for asset in assets:
        remember(asset)
    return tuple(sorted(paths))


def _source_scale(stage, scale, linear_unit):
    from pxr import UsdGeom

    values = tuple(float(value) for value in scale)
    if len(values) != 3 or any(not math.isfinite(value) or value <= 0 for value in values):
        raise ValueError("USD scale must contain three finite positive values")
    if UsdGeom.GetStageUpAxis(stage) == UsdGeom.Tokens.y:
        values = (values[0], values[2], values[1])
    return tuple(value * linear_unit for value in values)


def prepare_usd(source: str, *, scale=(1.0, 1.0, 1.0), mujoco: bool = False) -> ModelBuilder:
    """Import a rigid USD asset as a caller-owned, read-only builder template.

    Preserve bodies, joints, colliders, and visuals through native USD import.
    Visual/static assets are converted to meters; non-SI authored rigid physics
    is rejected. Placement uses a Z-up frame and scale uses that frame's axes.
    """
    from pxr import Gf, UsdGeom, UsdPhysics

    stage, _, linear_unit = _open_asset(source)
    factors = _source_scale(stage, scale, linear_unit)
    if factors != (1.0, 1.0, 1.0):
        pending = list(stage.GetPseudoRoot().GetChildren())
        while pending:
            prim = pending.pop()
            if prim.IsA(UsdGeom.Xformable):
                xform = UsdGeom.Xformable(prim)
                previous = xform.GetOrderedXformOps()
                op = xform.AddScaleOp(UsdGeom.XformOp.PrecisionDouble, "sceneAuthoring")
                op.Set(Gf.Vec3d(*factors))
                xform.SetXformOpOrder([op, *previous])
            else:
                pending.extend(prim.GetChildren())
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdPhysics.SetStageKilogramsPerUnit(stage, 1.0)
    builder = ModelBuilder()
    if mujoco:
        from ..solvers.mujoco.solver_mujoco import SolverMuJoCo  # noqa: PLC0415

        SolverMuJoCo.register_custom_attributes(builder)
    builder.add_usd(stage, collapse_fixed_joints=False)
    if builder.particle_count or builder.tri_count or builder.tet_count:
        raise ValueError("Scene authoring supports rigid USD assets only")
    if not builder.shape_count:
        raise ValueError(f"USD asset contains no supported rigid or visual geometry: {source}")
    return builder


def template_bounds(builder: ModelBuilder) -> tuple[np.ndarray, np.ndarray] | None:
    """Bound template collision geometry in its Z-up placement frame [m].

    Reuse native primitive bounds and host geometry without finalizing a model,
    mesh, or solver. Body and shape transforms retain the source asset's pivot.
    Include physical margins, excluding the contact search gap. Negative margins
    retain the unshrunk geometry for a conservative result. If there are no
    rigid colliders, include visual geometry. Bodyless templates include every
    shape because scene authoring promotes all their geometry to colliders.

    Return ``None`` when the selected geometry is empty, unbounded, unsupported,
    or has nonfinite bounds; explicit asset import can still proceed without a
    finite placement preview.
    """
    colliders = [index for index, flags in enumerate(builder.shape_flags) if flags & int(ShapeFlags.COLLIDE_SHAPES)]
    selected = (colliders if builder.body_count else ()) or range(builder.shape_count)
    primitive_bounds = {
        GeoType.BOX: compute_box_bounds,
        GeoType.CAPSULE: compute_capsule_bounds,
        GeoType.CYLINDER: compute_cylinder_bounds,
        GeoType.CONE: compute_cone_bounds,
        GeoType.ELLIPSOID: compute_ellipsoid_bounds,
    }
    aggregate_lower = np.full(3, np.inf)
    aggregate_upper = np.full(3, -np.inf)
    for index in selected:
        kind = builder.shape_type[index]
        scale = wp.vec3(builder.shape_scale[index])
        transform = builder.shape_transform[index]
        body = builder.shape_body[index]
        if body >= 0:
            transform = wp.transform_multiply(builder.body_q[body], transform)
        margin = float(builder.shape_margin[index])
        if (
            not np.isfinite(np.asarray(transform)).all()
            or not np.isfinite(np.asarray(scale)).all()
            or not math.isfinite(margin)
        ):
            return None
        if kind in primitive_bounds:
            lower, upper = primitive_bounds[kind](transform, scale)
        elif kind == GeoType.SPHERE:
            lower, upper = compute_sphere_bounds(wp.transform_get_translation(transform), scale[0])
        elif kind == GeoType.PLANE:
            if scale[0] <= 0 or scale[1] <= 0:
                return None
            lower, upper = compute_box_bounds(transform, wp.vec3(scale[0] * 0.5, scale[1] * 0.5, 0))
        elif kind in (GeoType.MESH, GeoType.CONVEX_MESH):
            source = builder.shape_source[index]
            if not isinstance(source, Mesh):
                return None
            vertices = np.asarray(source.vertices)
            if not len(vertices) or not np.isfinite(vertices).all():
                return None
            lower, upper = compute_shape_bounds(
                transform, scale, wp.vec3(vertices.min(axis=0)), wp.vec3(vertices.max(axis=0))
            )
        elif kind == GeoType.HFIELD:
            source = builder.shape_source[index]
            if not isinstance(source, Heightfield):
                return None
            heights = source.min_z + source.data * (source.max_z - source.min_z)
            if not heights.size or not np.isfinite(heights).all():
                return None
            # Finalization adds a bottom cap below the lowest heightfield sample.
            cap_gap = 1.0e-3 * max(1.0, source.hx, source.hy, abs(source.max_z - source.min_z))
            floor = min(source.min_z, source.max_z) - cap_gap
            lower, upper = compute_shape_bounds(
                transform, scale, wp.vec3(-source.hx, -source.hy, floor), wp.vec3(source.hx, source.hy, heights.max())
            )
        else:
            return None
        lower, upper = np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)
        if not np.isfinite([lower, upper]).all():
            return None
        aggregate_lower = np.minimum(aggregate_lower, lower - max(0.0, margin))
        aggregate_upper = np.maximum(aggregate_upper, upper + max(0.0, margin))
    if not np.isfinite([aggregate_lower, aggregate_upper]).all():
        return None
    return aggregate_lower, aggregate_upper


def _definition(obj) -> dict[str, Any]:
    definition = {
        "id": obj.id,
        "name": obj.name,
        "kind": obj.kind.value,
        "position": obj.position,
        "rotation": obj.rotation,
        "size": obj.size,
        "color": obj.color,
        "mass": obj.mass,
        "friction": obj.friction,
        "dynamic": obj.dynamic,
        "source": obj.source,
        "visible": obj.visible,
        "overrides": getattr(obj, "overrides", ("mass", "friction", "dynamic")),
    }
    _validate_definition(definition)
    definition["source"] = str(Path(obj.source).expanduser().resolve()) if obj.source else None
    definition["mass"], definition["friction"] = float(obj.mass), float(obj.friction)
    return {key: list(value) if isinstance(value, tuple) else value for key, value in definition.items()}


def _validate_definition(definition: dict[str, Any], *, normalize_rotation: bool = False) -> dict[str, Any]:
    """Validate a fresh object dictionary without accessing files or importing USD.

    Canonicalize vectors and explicit overrides to tuples. Editor inputs may
    request quaternion normalization; saved metadata must already be unit length.
    """
    if not isinstance(definition, dict):
        raise ValueError("Scene object metadata must be a dictionary")
    for field in (
        "id",
        "name",
        "kind",
        "position",
        "rotation",
        "size",
        "color",
        "mass",
        "friction",
        "dynamic",
        "source",
        "visible",
    ):
        if field not in definition:
            raise ValueError(f"Scene object is missing {field}")
    for field in ("id", "name"):
        if not isinstance(definition[field], str) or not definition[field].strip():
            raise ValueError(f"Scene object {field} must be a nonempty string")
    kind = definition["kind"]
    if not isinstance(kind, str) or kind not in {"box", "sphere", "capsule", "cylinder", "usd"}:
        raise ValueError(f"Unsupported scene object kind: {kind}")
    if not all(isinstance(definition[field], bool) for field in ("dynamic", "visible")):
        raise ValueError("Dynamic and visible flags must be booleans")
    for field, length in (("position", 3), ("rotation", 4), ("size", 3), ("color", 3)):
        try:
            values = tuple(float(value) for value in definition[field])
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"Invalid {field} in authoring object") from exc
        if len(values) != length or not all(math.isfinite(value) for value in values):
            raise ValueError(f"Invalid {field} in authoring object")
        definition[field] = values
    size = definition["size"]
    if min(size) <= 0 or not math.isfinite(definition["mass"]) or definition["mass"] <= 0:
        raise ValueError("Scene object dimensions and mass must be positive")
    if not math.isfinite(definition["friction"]) or definition["friction"] < 0:
        raise ValueError("Scene object friction must be finite and nonnegative")
    if any(value < 0 or value > 1 for value in definition["color"]):
        raise ValueError("Scene object color must lie between zero and one")
    rotation = definition["rotation"]
    norm = math.hypot(*rotation)
    if normalize_rotation:
        if norm < 1.0e-8:
            raise ValueError("Rotation must be a nonzero XYZW quaternion")
        scale = max(abs(value) for value in rotation)
        rotation = tuple(value / scale for value in rotation)
        norm = math.hypot(*rotation)
        definition["rotation"] = tuple(value / norm for value in rotation)
    elif not math.isclose(norm * norm, 1.0, abs_tol=1.0e-5):
        raise ValueError("Scene object rotation must be a unit quaternion")
    native_limit = float(np.finfo(np.float32).max)
    for field in ("position", "rotation", "size", "color", "mass", "friction"):
        values = (definition[field],) if field in ("mass", "friction") else definition[field]
        if any(abs(value) > native_limit for value in values):
            raise ValueError(f"Scene object {field} exceeds the finite float32 range")
        if field in ("size", "mass") and any(np.float32(value) <= 0 for value in values):
            raise ValueError(f"Scene object {field} must remain positive in float32")
    if kind == "sphere" and not all(math.isclose(value, size[0]) for value in size):
        raise ValueError("Sphere dimensions must be equal")
    if kind in {"capsule", "cylinder"} and not math.isclose(size[0], size[1]):
        raise ValueError("Capsule and cylinder radial dimensions must be equal")
    if kind == "capsule" and size[2] < size[0]:
        raise ValueError("Capsule total height must be at least its diameter")
    if kind == "usd" and (not isinstance(definition["source"], (str, os.PathLike)) or not str(definition["source"])):
        raise ValueError("USD scene objects require a source asset")
    overrides = definition.get("overrides", ("mass", "friction", "dynamic"))
    if any(value not in {"mass", "friction", "dynamic"} for value in overrides):
        raise ValueError("Unknown USD physical property override")
    definition["overrides"] = tuple(sorted(set(overrides)))
    return definition


def _write_references(stage, path, definition):
    from pxr import Gf, Usd, UsdGeom, UsdPhysics, UsdShade

    source_stage, source, linear_unit = _open_asset(definition["source"])
    factors = _source_scale(source_stage, definition["size"], linear_unit)
    template = prepare_usd(source, scale=definition["size"])
    overrides = set(definition.get("overrides", ("mass", "friction", "dynamic")))
    promote = not template.body_count
    asset = UsdGeom.Xform.Define(stage, f"{path}/Asset")
    if UsdGeom.GetStageUpAxis(source_stage) == UsdGeom.Tokens.y:
        asset.AddRotateXOp().Set(90.0)
    asset.AddScaleOp().Set(Gf.Vec3f(*factors))
    mappings = []
    for root in source_stage.GetPseudoRoot().GetChildren():
        destination = asset.GetPath().AppendChild(root.GetName())
        prim = stage.DefinePrim(destination)
        prim.GetReferences().AddReference(source, root.GetPath())
        prim.SetInstanceable(False)
        mappings.append((root.GetPath(), destination))

    # References to separate root prims preserve all source composition. Repair
    # cross-root relationships/connections, whose targets lie outside each
    # individual reference's namespace.
    def remap(target):
        for old, new in mappings:
            if target.HasPrefix(old):
                return target.ReplacePrefix(old, new)
        return target

    for source_prim in source_stage.Traverse():
        prim = stage.GetPrimAtPath(remap(source_prim.GetPath()))
        if not prim:
            continue
        for name in ("Newton", "MuJoCo", "PhysX"):
            selection = source_prim.GetVariantSets().GetVariantSet(name).GetVariantSelection()
            if selection:
                prim.GetVariantSets().GetVariantSet(name).SetVariantSelection(selection)
        # References and variants restore instanceability. Visit parents before
        # their descendants so overrides never target an instance proxy.
        if prim.IsInstanceable():
            prim.SetInstanceable(False)
        if source_prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            UsdPhysics.ArticulationRootAPI.Apply(prim)
        for relationship in source_prim.GetRelationships():
            targets = relationship.GetTargets()
            if targets:
                prim.CreateRelationship(relationship.GetName(), relationship.IsCustom()).SetTargets(
                    [remap(target) for target in targets]
                )
        for attribute in source_prim.GetAttributes():
            connections = attribute.GetConnections()
            if connections:
                prim.CreateAttribute(attribute.GetName(), attribute.GetTypeName(), attribute.IsCustom()).SetConnections(
                    [remap(target) for target in connections]
                )
        if "dynamic" in overrides and prim.HasAPI(UsdPhysics.RigidBodyAPI):
            UsdPhysics.RigidBodyAPI(prim).CreateKinematicEnabledAttr(not definition["dynamic"])
    if promote:
        UsdPhysics.RigidBodyAPI.Apply(asset.GetPrim()).CreateKinematicEnabledAttr(not definition["dynamic"])
        geometry_paths = set(template.shape_label)
        for prim in Usd.PrimRange(asset.GetPrim()):
            source_path = str(prim.GetPath()).replace(str(asset.GetPath()), "", 1)
            if prim.IsA(UsdGeom.Gprim) and source_path in geometry_paths:
                UsdPhysics.CollisionAPI.Apply(prim)
    if promote or "friction" in overrides:
        material = UsdShade.Material.Define(stage, f"{path}/PhysicsMaterial")
        physics_material = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
        physics_material.CreateStaticFrictionAttr(definition["friction"])
        physics_material.CreateDynamicFrictionAttr(definition["friction"])
        for prim in Usd.PrimRange(asset.GetPrim()):
            if prim.HasAPI(UsdPhysics.CollisionAPI):
                UsdShade.MaterialBindingAPI.Apply(prim).Bind(material, materialPurpose="physics")
    if definition["color"] != [1.0, 1.0, 1.0]:
        _tint_reference(asset.GetPrim(), definition["color"])
    # Record a total authored mass without replacing the source's rigid topology.
    bodies = [prim for prim in Usd.PrimRange(asset.GetPrim()) if prim.HasAPI(UsdPhysics.RigidBodyAPI)]
    if bodies and (promote or "mass" in overrides):
        masses = dict(zip(template.body_label, template.body_mass, strict=True))
        weights = [
            max(0.0, float(masses.get(str(prim.GetPath()).replace(str(asset.GetPath()), "", 1), 0.0)))
            for prim in bodies
        ]
        total = sum(weights)
        for prim, weight in zip(bodies, weights, strict=True):
            mass = definition["mass"] * (weight / total if total else 1.0 / len(bodies))
            UsdPhysics.MassAPI.Apply(prim).CreateMassAttr(mass)
            if weight > 0:
                for name in ("physics:diagonalInertia", "newton:inertia"):
                    attribute = prim.GetAttribute(name)
                    if attribute and attribute.HasAuthoredValueOpinion():
                        value = attribute.Get()
                        scaled = [component * mass / weight for component in value]
                        attribute.Set(Gf.Vec3f(*scaled) if name == "physics:diagonalInertia" else scaled)


def _tint_reference(root, tint):
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade, Vt

    from ..usd.utils import _color_to_display_space  # noqa: PLC0415 - defer optional USD parser imports

    def linear_tint(value, attribute):
        display = _color_to_display_space(tuple(value[:3]), attribute)
        return Gf.Vec3f(
            *color_srgb_to_linear(tuple(channel * factor for channel, factor in zip(display, tint, strict=True)))
        )

    visited = set()
    for prim in Usd.PrimRange(root):
        if not prim.IsA(UsdGeom.Gprim):
            continue
        material, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
        if material:
            shader, _, _ = material.ComputeSurfaceSource()
            if not shader or shader.GetIdAttr().Get() != "UsdPreviewSurface":
                raise ValueError(
                    "Nonwhite USD tint export supports displayColor and scalar UsdPreviewSurface only; use white tint for other shader graphs"
                )
            color = shader.GetInput("baseColor") or shader.GetInput("diffuseColor")
            if color and color.HasConnectedSource():
                raise ValueError(
                    "Nonwhite USD tint export does not support connected color textures or shader graphs; use white tint"
                )
            if shader.GetPath() not in visited:
                visited.add(shader.GetPath())
                if not color:
                    color = shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f)
                    color.Set(Gf.Vec3f(0.18))
                value = color.Get()
                if value is None or len(value) != 3:
                    raise ValueError("Nonwhite USD tint export requires a scalar RGB UsdPreviewSurface color")
                color.Set(linear_tint(value, color.GetAttr()))
                color.GetAttr().SetColorSpace("lin_rec709_scene")
        geometry = UsdGeom.Gprim(prim)
        attribute = geometry.GetDisplayColorAttr()
        colors = attribute.Get() or [Gf.Vec3f(0.18)]
        geometry.CreateDisplayColorAttr().Set(Vt.Vec3fArray([linear_tint(value, attribute) for value in colors]))
        attribute.SetColorSpace("lin_rec709_scene")


def save_scene(path: str | os.PathLike, objects: Iterable) -> None:
    """Atomically write authored objects using standard USD geometry and physics.

    Custom data stores editor definitions and stable IDs, never a private
    simulation schema. Referenced assets remain referenced rather than flattened.
    Nonwhite asset tint supports displayColor and scalar UsdPreviewSurface;
    connected textures and other shader graphs require white tint.
    """
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, Vt

    destination = Path(path).expanduser().resolve()
    if destination.suffix.lower() not in {".usd", ".usda", ".usdc"}:
        raise ValueError("Scene destination must end in .usd, .usda, or .usdc")
    definitions = [_definition(obj) for obj in objects]
    ids = [definition["id"] for definition in definitions]
    if len(set(ids)) != len(ids):
        raise ValueError("Scene object IDs must be unique")
    for definition in definitions:
        if definition["kind"] != "usd":
            continue
        for filename in _asset_dependencies(definition["source"]):
            dependency = Path(filename)
            if destination == dependency or (
                destination.is_file() and dependency.is_file() and destination.samefile(dependency)
            ):
                raise ValueError(f"Scene destination would overwrite an imported asset dependency: {dependency}")
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdPhysics.SetStageKilogramsPerUnit(stage, 1.0)
    stage.GetRootLayer().customLayerData = {_MARKER: 1}
    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())
    UsdPhysics.Scene.Define(stage, "/World/PhysicsScene")
    floor = UsdGeom.Plane.Define(stage, "/World/Ground")
    floor.CreateAxisAttr(UsdGeom.Tokens.z)
    floor.CreateWidthAttr(100.0)
    floor.CreateLengthAttr(100.0)
    floor.CreateDisplayColorAttr(Vt.Vec3fArray([Gf.Vec3f(*color_srgb_to_linear((0.28, 0.31, 0.34)))]))
    UsdPhysics.CollisionAPI.Apply(floor.GetPrim())
    floor_material = UsdShade.Material.Define(stage, "/World/GroundMaterial")
    floor_physics = UsdPhysics.MaterialAPI.Apply(floor_material.GetPrim())
    floor_physics.CreateStaticFrictionAttr(_GROUND_FRICTION)
    floor_physics.CreateDynamicFrictionAttr(_GROUND_FRICTION)
    UsdShade.MaterialBindingAPI.Apply(floor.GetPrim()).Bind(floor_material, materialPurpose="physics")
    for index, definition in enumerate(definitions):
        object_path = f"/World/Object_{index}"
        obj = UsdGeom.Xform.Define(stage, object_path)
        obj.GetPrim().SetCustomDataByKey(_MARKER, json.dumps(definition, allow_nan=False))
        obj.GetPrim().SetDisplayName(definition["name"])
        obj.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(*definition["position"]))
        x, y, z, w = definition["rotation"]
        obj.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Quatd(w, Gf.Vec3d(x, y, z)))
        obj.CreateVisibilityAttr(UsdGeom.Tokens.inherited if definition["visible"] else UsdGeom.Tokens.invisible)
        kind, size = definition["kind"], definition["size"]
        if kind == "usd":
            _write_references(stage, object_path, definition)
            continue
        geometry_path = f"{object_path}/Geometry"
        if kind == "box":
            geometry = UsdGeom.Cube.Define(stage, geometry_path)
            geometry.CreateSizeAttr(1.0)
            geometry.AddScaleOp().Set(Gf.Vec3f(*size))
        elif kind == "sphere":
            geometry = UsdGeom.Sphere.Define(stage, geometry_path)
            geometry.CreateRadiusAttr(size[0] / 2.0)
        else:
            geometry = (UsdGeom.Capsule if kind == "capsule" else UsdGeom.Cylinder).Define(stage, geometry_path)
            geometry.CreateRadiusAttr(size[0] / 2.0)
            geometry.CreateHeightAttr(size[2] - size[0] if kind == "capsule" else size[2])
            geometry.CreateAxisAttr(UsdGeom.Tokens.z)
        prim = geometry.GetPrim()
        geometry.CreateDisplayColorAttr(Vt.Vec3fArray([Gf.Vec3f(*color_srgb_to_linear(definition["color"]))]))
        UsdPhysics.CollisionAPI.Apply(prim)
        body = UsdPhysics.RigidBodyAPI.Apply(obj.GetPrim())
        body.CreateKinematicEnabledAttr(not definition["dynamic"])
        UsdPhysics.MassAPI.Apply(obj.GetPrim()).CreateMassAttr(definition["mass"])
        material = UsdShade.Material.Define(stage, f"{object_path}/Material")
        physics_material = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
        physics_material.CreateStaticFrictionAttr(definition["friction"])
        physics_material.CreateDynamicFrictionAttr(definition["friction"])
        if kind == "sphere":
            material.GetPrim().AddAppliedSchema("NewtonMaterialAPI")
            material.GetPrim().CreateAttribute("newton:rollingFriction", Sdf.ValueTypeNames.Float).Set(
                _SPHERE_ROLLING_FRICTION
            )
            prim.CreateAttribute("mjc:condim", Sdf.ValueTypeNames.Int).Set(6)
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(material, materialPurpose="physics")
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{destination.stem}-", suffix=destination.suffix, dir=destination.parent
    )
    os.close(descriptor)
    try:
        if not stage.GetRootLayer().Export(temporary):
            raise OSError(f"Cannot export scene to {destination}")
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def load_scene(path: str | os.PathLike) -> list[dict[str, Any]]:
    """Read definitions from an authored scene; reject ordinary USD assets distinctly."""
    from pxr import Usd

    stage = Usd.Stage.Open(str(Path(path).expanduser().resolve()))
    if stage is None:
        raise ValueError(f"Cannot open USD scene: {path}")
    if stage.GetRootLayer().customLayerData.get(_MARKER) != 1:
        raise NotAuthoringSceneError("USD file is an asset, not a saved authoring scene")
    world = stage.GetPrimAtPath("/World")
    if not world:
        raise ValueError("Saved authoring scene is missing /World")
    definitions = []
    for prim in world.GetChildren():
        encoded = prim.GetCustomDataByKey(_MARKER)
        if encoded is not None:
            try:
                definition = json.loads(encoded)
                _validate_definition(definition)
                definitions.append(
                    {key: list(value) if isinstance(value, tuple) else value for key, value in definition.items()}
                )
            except (KeyError, TypeError, json.JSONDecodeError) as exc:
                raise ValueError(f"Invalid authoring metadata at {prim.GetPath()}") from exc
    ids = [definition["id"] for definition in definitions]
    if len(set(ids)) != len(ids):
        raise ValueError("Saved scene object IDs must be unique")
    return definitions
