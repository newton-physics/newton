# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Shared USD-authoring fixtures for the deformable-import test modules."""


def _add_cable_curve(stage, path, points, *, periodic=False, thickness=0.02, density=None, collision=True):
    """Author a GeomBasisCurves marked as a curve deformable (cable).

    Binds a minimal canonical curve-deformable material carrying ``thickness`` (and optional
    ``density``) so the importer does not warn about an unauthored cable thickness. Pass
    ``thickness=None`` to leave the cable without a bound material, e.g. to exercise the
    default-radius fallback or a test's own material binding. ``collision`` authors an
    enabled ``PhysicsCollisionAPI`` (the common colliding case); pass ``False`` for the
    collision-gating tests.
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
    if collision:
        curves.GetPrim().AddAppliedSchema("PhysicsCollisionAPI")
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
    # Declare the per-family deformable material API the importer's readers gate on. The
    # family APIs extend UsdPhysicsMaterialAPI, so proposal-shaped materials apply both
    # (schema inheritance is unavailable while the deformable schema is unregistered).
    mat.GetPrim().AddAppliedSchema("PhysicsMaterialAPI")
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


def _add_cloth_mesh(stage, path, *, collision=True):
    """Author a two-triangle quad GeomMesh marked as a surface deformable (cloth)."""
    from pxr import UsdGeom

    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr([(0.0, 0.0, 1.0), (1.0, 0.0, 1.0), (1.0, 1.0, 1.0), (0.0, 1.0, 1.0)])
    mesh.CreateFaceVertexCountsAttr([3, 3])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 0, 2, 3])
    mesh.GetPrim().AddAppliedSchema("PhysicsSurfaceDeformableSimAPI")
    if collision:
        mesh.GetPrim().AddAppliedSchema("PhysicsCollisionAPI")
    return mesh


def _apply_deformable_body_api(prim, *, mass=None, density=None):
    """Apply PhysicsDeformableBodyAPI with optional mass / density overrides."""
    from pxr import Sdf

    prim.AddAppliedSchema("PhysicsDeformableBodyAPI")
    if mass is not None:
        prim.CreateAttribute("physics:mass", Sdf.ValueTypeNames.Float).Set(mass)
    if density is not None:
        prim.CreateAttribute("physics:density", Sdf.ValueTypeNames.Float).Set(density)


def _deformable_stage(up_axis="z"):
    """Create an in-memory stage with the up axis and a physics scene already authored.

    ``add_usd()`` accepts a stage directly, so parsing tests skip the disk round-trip;
    author on disk only when file composition/resolution itself is under test.
    """
    from pxr import Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageUpAxis(stage, getattr(UsdGeom.Tokens, up_axis))
    UsdPhysics.Scene.Define(stage, "/PhysicsScene")
    return stage


def group_labels(builder, family):
    """Prim-path labels of a deformable family's imported groups (``cable``/``cloth``/``soft``).

    The single seam through which tests locate deformable groups: it reads the builder
    registries, so a change to how group metadata is stored or exposed only touches this
    helper, not the tests.
    """
    return list(getattr(builder, f"_{family}_label"))


def group_range(builder, family, label, kind, world=None):
    """``[start, end)`` index range of one deformable group's ``kind`` array.

    ``kind`` is ``body``/``joint`` for cables, ``particle``/``tri``/``edge`` for cloth, and
    ``particle``/``tet`` for soft volumes. See :func:`group_labels` for why tests must resolve
    ranges through this seam.
    """
    labels = getattr(builder, f"_{family}_label")
    worlds = getattr(builder, f"_{family}_world")
    matches = [i for i, group_label in enumerate(labels) if group_label == label]
    if world is not None:
        matches = [i for i in matches if worlds[i] == world]
    if len(matches) != 1:
        raise LookupError(f"{len(matches)} {family} groups labelled '{label}' (world={world})")
    (i,) = matches
    return getattr(builder, f"_{family}_{kind}_start")[i], getattr(builder, f"_{family}_{kind}_end")[i]
