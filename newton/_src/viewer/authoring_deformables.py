# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Generate bounded, schema-authored deformables for the experimental editor."""

from __future__ import annotations

import itertools
import math

import numpy as np

from ..utils.color import color_srgb_to_linear

DEFORMABLE_KINDS = frozenset({"cloth", "cable", "soft", "particles"})
MAX_PARTICLES = 50_000
MPM_VOXEL_SIZE = 0.08
_DEFAULTS = {"resolution": 8, "density": 1000.0, "stiffness": 50_000.0, "thickness": 0.02}


def validate_deformable(definition: dict, *, check_points: bool = True) -> dict:
    """Validate and canonicalize generated physical parameters without importing USD."""
    kind = definition["kind"]
    if kind not in DEFORMABLE_KINDS:
        raise ValueError(f"Unsupported deformable kind: {kind}")
    resolution = definition.get("resolution", _DEFAULTS["resolution"])
    maximum = 16 if kind == "soft" else 32
    if isinstance(resolution, bool) or not isinstance(resolution, int) or not 2 <= resolution <= maximum:
        raise ValueError(f"{kind} resolution must be an integer between 2 and {maximum}")
    definition["resolution"] = resolution
    for name in ("density", "stiffness", "thickness"):
        try:
            value = float(definition.get(name, _DEFAULTS[name]))
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError(f"{name} must be finite and positive") from error
        if not math.isfinite(value) or not 0 < value <= float(np.finfo(np.float32).max) or np.float32(value) <= 0:
            raise ValueError(f"{name} must be finite, positive, and representable in float32")
        definition[name] = value
    if not definition.get("dynamic", True):
        raise ValueError("Generated deformables must be dynamic")
    points = definition.get("points", ())
    if not isinstance(points, (tuple, list)):
        raise ValueError("Deformable points must be a list or tuple of local XYZ positions")
    if kind == "particles":
        if not 1 <= len(points) <= MAX_PARTICLES:
            raise ValueError(f"Particles require between 1 and {MAX_PARTICLES} local points")
        if check_points:
            try:
                array = np.asarray(points, dtype=np.float64)
            except (TypeError, ValueError, OverflowError) as error:
                raise ValueError("Particle points must be finite XYZ positions") from error
            if array.shape != (len(points), 3) or not np.isfinite(array).all():
                raise ValueError("Particle points must be finite XYZ positions")
            if np.any(np.abs(array) > np.finfo(np.float32).max):
                raise ValueError("Particle points exceed the finite float32 range")
            definition["points"] = tuple(tuple(float(value) for value in point) for point in array)
    elif len(points):
        raise ValueError("Only particles accept explicit points")
    else:
        definition["points"] = ()
    size = definition["size"]
    volume = {
        "cloth": lambda: size[0] * size[1] * definition["thickness"],
        "cable": lambda: math.pi * (definition["thickness"] / 2) ** 2 * size[0],
        "soft": lambda: math.prod(size),
        "particles": lambda: len(points) * definition["thickness"] ** 3,
    }[kind]()
    mass = volume * definition["density"]
    if not math.isfinite(mass) or not 0 < mass <= float(np.finfo(np.float32).max) or np.float32(mass) <= 0:
        raise ValueError("Deformable dimensions and density must produce a finite positive float32 mass")
    return definition


def _geometry(definition: dict):
    """Return local simulation points and native element connectivity."""
    kind, size, count = definition["kind"], definition["size"], definition["resolution"]
    if kind == "particles":
        return definition["points"], None
    axes = [np.linspace(-size[axis] / 2, size[axis] / 2, count + 1) for axis in range(3)]
    if kind == "cable":
        return [(float(x), 0.0, 0.0) for x in axes[0]], None
    if kind == "cloth":
        points = [(float(x), float(y), 0.0) for y in axes[1] for x in axes[0]]
        faces = []
        for y, x in itertools.product(range(count), repeat=2):
            a = y * (count + 1) + x
            b, c, d = a + 1, a + count + 1, a + count + 2
            faces.extend(((a, b, d), (a, d, c)))
        return points, faces
    points = [(float(x), float(y), float(z)) for z in axes[2] for y in axes[1] for x in axes[0]]
    stride = count + 1
    pattern = ((0, 1, 3, 7), (0, 3, 2, 7), (0, 2, 6, 7), (0, 6, 4, 7), (0, 4, 5, 7), (0, 5, 1, 7))
    offsets = (0, 1, stride, stride + 1, stride**2, stride**2 + 1, stride**2 + stride, stride**2 + stride + 1)
    tets = []
    for z, y, x in itertools.product(range(count), repeat=3):
        start = z * stride**2 + y * stride + x
        tets.extend(tuple(start + offsets[index] for index in tet) for tet in pattern)
    return points, tets


def write_deformable(stage, object_path: str, definition: dict, *, physics_scene_path="/World/PhysicsScene") -> None:
    """Author validated geometry and physical material beneath an existing object Xform."""
    from pxr import Sdf, UsdGeom, UsdPhysics, UsdShade

    kind = definition["kind"]
    body = stage.GetPrimAtPath(object_path)
    body.AddAppliedSchema("PhysicsDeformableBodyAPI")
    path = f"{object_path}/Geometry"
    points, indices = _geometry(definition)
    if kind == "cloth":
        geometry = UsdGeom.Mesh.Define(stage, path)
        geometry.CreateFaceVertexCountsAttr([3] * len(indices))
        geometry.CreateFaceVertexIndicesAttr(np.asarray(indices).ravel().tolist())
        geometry.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
        geometry.GetPrim().AddAppliedSchema("PhysicsSurfaceDeformableSimAPI")
    elif kind == "cable":
        geometry = UsdGeom.BasisCurves.Define(stage, path)
        geometry.CreateTypeAttr(UsdGeom.Tokens.linear)
        geometry.CreateWrapAttr(UsdGeom.Tokens.nonperiodic)
        geometry.CreateCurveVertexCountsAttr([len(points)])
        geometry.CreateWidthsAttr([definition["thickness"]])
        geometry.SetWidthsInterpolation(UsdGeom.Tokens.constant)
        geometry.GetPrim().AddAppliedSchema("PhysicsCurvesDeformableSimAPI")
    elif kind == "soft":
        geometry = UsdGeom.TetMesh.Define(stage, path)
        geometry.CreateTetVertexIndicesAttr(indices)
        geometry.GetPrim().AddAppliedSchema("PhysicsVolumeDeformableSimAPI")
    else:
        geometry = UsdGeom.Points.Define(stage, path)
        geometry.CreateWidthsAttr([definition["thickness"]])
        geometry.SetWidthsInterpolation(UsdGeom.Tokens.constant)
        geometry.GetPrim().AddAppliedSchema("NewtonPointsDeformableSimAPI")
        scene = UsdPhysics.Scene.Define(stage, physics_scene_path)
        scene.GetPrim().AddAppliedSchema("NewtonMPMSceneAPI")
        scene.GetPrim().CreateAttribute("newton:mpm:voxelSize", Sdf.ValueTypeNames.Float).Set(MPM_VOXEL_SIZE)
        body.CreateRelationship("physics:simulationOwner").SetTargets([scene.GetPath()])
    geometry.CreatePointsAttr(points)
    geometry.CreateDisplayColorAttr([color_srgb_to_linear(definition["color"])])
    UsdPhysics.CollisionAPI.Apply(geometry.GetPrim())
    if kind in {"cloth", "cable"}:
        geometry.GetPrim().CreateAttribute("physics:thicknesses", Sdf.ValueTypeNames.FloatArray).Set(
            [definition["thickness"]]
        )
        geometry.GetPrim().CreateAttribute("physics:thicknesses:elementType", Sdf.ValueTypeNames.Token).Set("constant")
    material = UsdShade.Material.Define(stage, f"{object_path}/Material")
    physics_material = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    physics_material.CreateDensityAttr(definition["density"])
    if kind == "cable":
        physics_material.CreateStaticFrictionAttr(definition["friction"])
        physics_material.CreateDynamicFrictionAttr(definition["friction"])
    family = {"cloth": "Surface", "cable": "Curves", "soft": "Volume"}.get(kind)
    if family:
        material.GetPrim().AddAppliedSchema(f"Physics{family}DeformableMaterialAPI")
        material.GetPrim().CreateAttribute("physics:youngsModulus", Sdf.ValueTypeNames.Float).Set(
            definition["stiffness"]
        )
        material.GetPrim().CreateAttribute("physics:poissonsRatio", Sdf.ValueTypeNames.Float).Set(0.3)
    else:
        material.GetPrim().AddAppliedSchema("NewtonMPMMaterialAPI")
        material.GetPrim().CreateAttribute("newton:mpm:youngsModulus", Sdf.ValueTypeNames.Float).Set(
            definition["stiffness"]
        )
        material.GetPrim().CreateAttribute("newton:mpm:internalFriction", Sdf.ValueTypeNames.Float).Set(
            definition["friction"]
        )
    UsdShade.MaterialBindingAPI.Apply(geometry.GetPrim()).Bind(material, materialPurpose="physics")


def build_deformable(builder, definition: dict, *, label_prefix: str) -> dict:
    """Append generated native USD once, including its authored world transform."""
    from pxr import Gf, Tf, Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdPhysics.SetStageKilogramsPerUnit(stage, 1.0)
    path = "/" + "/".join(Tf.MakeValidIdentifier(part) for part in label_prefix.strip("/").split("/") if part)
    if path == "/":
        raise ValueError("A generated deformable requires a nonempty label prefix")
    obj = UsdGeom.Xform.Define(stage, path)
    obj.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(*definition["position"]))
    x, y, z, w = definition["rotation"]
    obj.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Quatd(w, Gf.Vec3d(x, y, z)))
    write_deformable(stage, path, definition)
    return builder.add_usd(stage, return_deformable_results=True, load_visual_shapes=False)
