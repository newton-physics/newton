# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Shared USD-authoring fixtures for the deformable-import test modules."""


def _add_cable_curve(stage, path, points, *, periodic=False, thickness=0.02, density=None):
    """Author a GeomBasisCurves marked as a curve deformable (cable).

    Binds a minimal canonical curve-deformable material carrying ``thickness`` (and optional
    ``density``) so the importer does not warn about an unauthored cable thickness. Pass
    ``thickness=None`` to leave the cable without a bound material, e.g. to exercise the
    default-radius fallback or a test's own material binding.
    """
    from pxr import UsdGeom

    curves = UsdGeom.BasisCurves.Define(stage, path)
    curves.CreateTypeAttr().Set(UsdGeom.Tokens.linear)
    curves.CreatePointsAttr([tuple(p) for p in points])
    curves.CreateCurveVertexCountsAttr([len(points)])
    curves.CreateWrapAttr().Set(UsdGeom.Tokens.periodic if periodic else UsdGeom.Tokens.nonperiodic)
    # Metadata-based discovery: apply the curve-deformable sim API by token so it
    # is found even when the deformable schema is not registered with USD.
    curves.GetPrim().AddAppliedSchema("PhysicsCurvesDeformableSimAPI")
    if thickness is not None:
        mat_attrs = {"thickness": thickness}
        if density is not None:
            mat_attrs["density"] = density
        _bind_deformable_material(stage, curves.GetPrim(), f"{path}Mat", **mat_attrs)
    return curves


def _bind_deformable_material(stage, prim, mat_path, *, namespace="physics", **attrs):
    """Author a deformable material and bind it to a prim.

    Authors under the canonical ``physics:`` namespace by default; pass
    ``namespace`` to author under a vendor namespace (e.g. ``omniphysics``) to
    exercise the schema-resolver compatibility path.
    """
    from pxr import Sdf, UsdGeom, UsdShade

    mat = UsdShade.Material.Define(stage, mat_path)
    # Declare the per-family deformable material API the importer's readers gate on.
    if prim.IsA(UsdGeom.BasisCurves):
        mat.GetPrim().AddAppliedSchema("PhysicsCurvesDeformableMaterialAPI")
    elif prim.IsA(UsdGeom.TetMesh):
        mat.GetPrim().AddAppliedSchema("PhysicsVolumeDeformableMaterialAPI")
    elif prim.IsA(UsdGeom.Mesh):
        mat.GetPrim().AddAppliedSchema("PhysicsSurfaceDeformableMaterialAPI")
    for name, value in attrs.items():
        mat.GetPrim().CreateAttribute(f"{namespace}:{name}", Sdf.ValueTypeNames.Float).Set(value)
    binding = UsdShade.MaterialBindingAPI.Apply(prim)
    binding.Bind(mat, materialPurpose="physics")
    return mat


def _add_physics_attachment(
    stage,
    path,
    *,
    src0,
    type0,
    indices0,
    src1="",
    type1="xform",
    indices1=None,
    coords0=None,
    coords1=None,
    enabled=True,
    stiffness=None,
    damping=None,
):
    """Author a proposal PhysicsAttachment prim by token, before the schema is registered."""
    from pxr import Sdf

    prim = stage.DefinePrim(path, "PhysicsAttachment")
    prim.CreateRelationship("physics:src0").SetTargets([src0])
    if src1:
        prim.CreateRelationship("physics:src1").SetTargets([src1])
    prim.CreateAttribute("physics:type0", Sdf.ValueTypeNames.Token).Set(type0)
    prim.CreateAttribute("physics:type1", Sdf.ValueTypeNames.Token).Set(type1)
    prim.CreateAttribute("physics:indices0", Sdf.ValueTypeNames.IntArray).Set(list(indices0))
    if indices1 is not None:
        prim.CreateAttribute("physics:indices1", Sdf.ValueTypeNames.IntArray).Set(list(indices1))
    if coords0 is not None:
        prim.CreateAttribute("physics:coords0", Sdf.ValueTypeNames.Vector3fArray).Set([tuple(c) for c in coords0])
    if coords1 is not None:
        prim.CreateAttribute("physics:coords1", Sdf.ValueTypeNames.Vector3fArray).Set([tuple(c) for c in coords1])
    prim.CreateAttribute("physics:attachmentEnabled", Sdf.ValueTypeNames.Bool).Set(enabled)
    if stiffness is not None:
        prim.CreateAttribute("physics:stiffness", Sdf.ValueTypeNames.Float).Set(stiffness)
    if damping is not None:
        prim.CreateAttribute("physics:damping", Sdf.ValueTypeNames.Float).Set(damping)
    return prim


def _add_element_collision_filter(
    stage, path, *, src0, src1, indices0=None, indices1=None, counts0=None, counts1=None, enabled=True
):
    """Author an AOUSD ``PhysicsElementCollisionFilter`` prim by token.

    ``counts0`` / ``counts1`` author the optional ``groupElemCounts`` arrays that slice the indices
    into paired groups; omit them to leave a single implicit group.
    """
    from pxr import Sdf

    prim = stage.DefinePrim(path, "PhysicsElementCollisionFilter")
    prim.CreateRelationship("physics:src0").SetTargets([src0])
    prim.CreateRelationship("physics:src1").SetTargets([src1])
    prim.CreateAttribute("physics:filterEnabled", Sdf.ValueTypeNames.Bool).Set(enabled)
    prim.CreateAttribute("physics:groupElemIndices0", Sdf.ValueTypeNames.IntArray).Set(list(indices0 or []))
    prim.CreateAttribute("physics:groupElemIndices1", Sdf.ValueTypeNames.IntArray).Set(list(indices1 or []))
    if counts0 is not None:
        prim.CreateAttribute("physics:groupElemCounts0", Sdf.ValueTypeNames.IntArray).Set(list(counts0))
    if counts1 is not None:
        prim.CreateAttribute("physics:groupElemCounts1", Sdf.ValueTypeNames.IntArray).Set(list(counts1))
    return prim


def _add_cloth_mesh(stage, path):
    """Author a two-triangle quad GeomMesh marked as a surface deformable (cloth)."""
    from pxr import UsdGeom

    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr([(0.0, 0.0, 1.0), (1.0, 0.0, 1.0), (1.0, 1.0, 1.0), (0.0, 1.0, 1.0)])
    mesh.CreateFaceVertexCountsAttr([3, 3])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 0, 2, 3])
    mesh.GetPrim().AddAppliedSchema("PhysicsSurfaceDeformableSimAPI")
    return mesh


def _apply_deformable_body_api(prim, *, mass=None, density=None):
    """Apply PhysicsDeformableBodyAPI with optional mass / density overrides."""
    from pxr import Sdf

    prim.AddAppliedSchema("PhysicsDeformableBodyAPI")
    if mass is not None:
        prim.CreateAttribute("physics:mass", Sdf.ValueTypeNames.Float).Set(mass)
    if density is not None:
        prim.CreateAttribute("physics:density", Sdf.ValueTypeNames.Float).Set(density)


def deformable_maps(builder):
    """Reconstruct per-prim deformable index maps from a builder's group registry.

    ``add_usd()`` no longer returns ``path_cable_map`` / ``path_cloth_map`` / ``path_soft_map`` —
    index ranges live on the builder/Model group registries. Tests rebuild the old dict shape here
    from those registries (cable body/joint ranges are contiguous, so ``range`` matches the original
    index lists) to keep per-prim range assertions concise. Returns ``(cable, cloth, soft)``.
    """
    cable = {
        builder.cable_label[i]: (
            list(range(builder.cable_body_start[i], builder.cable_body_end[i])),
            list(range(builder.cable_joint_start[i], builder.cable_joint_end[i])),
        )
        for i in range(len(builder.cable_label))
    }
    cloth = {
        builder.cloth_label[i]: {
            "particle": (builder.cloth_particle_start[i], builder.cloth_particle_end[i]),
            "tri": (builder.cloth_tri_start[i], builder.cloth_tri_end[i]),
            "edge": (builder.cloth_edge_start[i], builder.cloth_edge_end[i]),
        }
        for i in range(len(builder.cloth_label))
    }
    soft = {
        builder.soft_label[i]: {
            "particle": (builder.soft_particle_start[i], builder.soft_particle_end[i]),
            "tet": (builder.soft_tet_start[i], builder.soft_tet_end[i]),
        }
        for i in range(len(builder.soft_label))
    }
    return cable, cloth, soft
