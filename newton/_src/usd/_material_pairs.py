# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Import rigid contact overrides authored with NewtonMaterialPairAPI."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pxr import Usd

    from ..sim.builder import ModelBuilder


def _parse_material_pair(builder: ModelBuilder, prim: Usd.Prim, material_ids: dict[str, int]) -> None:
    """Register a USD pair and remember its material prim paths for collider import."""
    from pxr import UsdPhysics, UsdShade

    stage = prim.GetStage()
    material_paths = []
    for name in ("newton:materialPair:material0", "newton:materialPair:material1"):
        targets = prim.GetRelationship(name).GetForwardedTargets()
        if len(targets) != 1:
            raise ValueError(f"Material pair {prim.GetPath()} requires exactly one target for {name}.")
        material = stage.GetPrimAtPath(targets[0])
        if not material or not material.IsA(UsdShade.Material) or not material.HasAPI(UsdPhysics.MaterialAPI):
            raise ValueError(f"Material pair {prim.GetPath()} target {targets[0]} must be a physics material.")
        material_paths.append(str(targets[0]))

    properties = {}
    for field, token in (
        ("mu", "mu"),
        ("mu_static", "muStatic"),
        ("mu_torsional", "muTorsional"),
        ("mu_rolling", "muRolling"),
        ("restitution", "restitution"),
    ):
        name = f"newton:materialPair:{token}"
        attr = prim.GetAttribute(name)
        if not attr or not attr.HasAuthoredValueOpinion():
            continue
        value = attr.Get()
        if not isinstance(value, float):
            raise ValueError(f"Material pair {prim.GetPath()} requires a floating-point value for {name}.")
        properties[field] = value
    if not properties:
        raise ValueError(f"Material pair {prim.GetPath()} requires at least one coefficient.")

    ids = []
    for path in material_paths:
        if path not in material_ids:
            material_ids[path] = builder.add_material(path)
        ids.append(material_ids[path])
    builder.set_material_pair(*ids, **properties)
