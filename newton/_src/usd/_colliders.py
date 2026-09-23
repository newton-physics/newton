# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Add rigid USD colliders and approximate their meshes."""

from __future__ import annotations

import math
import re
import warnings
from collections.abc import Callable
from dataclasses import replace
from typing import TYPE_CHECKING, Any

import warp as wp

from ..core import quat_between_axes
from ..core.types import Axis
from ..geometry import ShapeFlags
from ..sim.builder import ModelBuilder
from . import utils as usd
from ._collision_filters import _collect_filtered_pairs
from ._mass_properties import _is_enabled_collider
from ._resolution_policy import (
    _resolve_shape_contact,
    _resolve_shape_hydroelastic,
    _resolve_shape_offsets,
    _resolve_shape_sdf,
    _resolve_shape_shell,
)
from .schema_resolver import PrimType

if TYPE_CHECKING:
    from pxr import Usd, UsdPhysics

    from ._mass_properties import _UsdMassProperties
    from ._resolution_policy import _PhysicsMaterial
    from ._visuals import _UsdVisuals
    from .schema_resolver import SchemaResolverManager


def _parse_colliders(
    *,
    builder: ModelBuilder,
    stage: Usd.Stage,
    ret_dict: dict[Any, Any],
    R: SchemaResolverManager,
    visuals: _UsdVisuals,
    mass_properties: _UsdMassProperties,
    material_specs: dict[str, _PhysicsMaterial],
    default_shape_density: float,
    path_body_map: dict[str, int],
    path_shape_map: dict[str, int],
    path_shape_scale: dict[str, wp.vec3],
    builder_custom_attr_shape: list[ModelBuilder.CustomAttribute],
    bodies_with_visual_shapes: set[int],
    incoming_world_xform: wp.transform,
    usd_axis_to_axis: dict[UsdPhysics.Axis, Axis],
    no_collision_shapes: set[int],
    imported_rigid_collider_groups: dict[str, tuple[str, ...]],
    ignore_paths: list[str],
    load_visual_shapes: bool,
    hide_collision_shapes: bool,
    force_show_colliders: bool,
    mesh_maxhullvert: int,
    skip_mesh_approximation: bool,
    collect_schema_attrs: bool,
    legacy_margin_gap: bool,
    verbose: bool,
    warn_invalid_desc: Callable[[Any, Any], bool],
    authored_filtered_path_pairs: set[tuple[str, str]],
    _is_uniform_scale: Callable[[Any], bool],
    _UNMATERIALED_VISUAL_COLOR: tuple[float, float, float],
) -> None:
    """Add colliders in descriptor order using the importer's shared maps and helpers."""
    from pxr import UsdPhysics

    # mapping from physics:approximation attribute (lower case) to remeshing method
    approximation_to_remeshing_method = {
        "convexdecomposition": "coacd",
        "convexhull": "convex_hull",
        "boundingsphere": "bounding_sphere",
        "boundingcube": "bounding_box",
        "meshsimplification": "quadratic",
    }
    # mapping from remeshing method to a list of shape indices
    remeshing_queue = {}
    # Approximated colliders whose prim is viewport geometry, and which therefore keep
    # their authored topology as a visual shape. See the approximation pass below.
    approximated_viewport_shapes: set[int] = set()

    for key, value in ret_dict.items():
        if key in {
            UsdPhysics.ObjectType.CubeShape,
            UsdPhysics.ObjectType.SphereShape,
            UsdPhysics.ObjectType.CapsuleShape,
            UsdPhysics.ObjectType.CylinderShape,
            UsdPhysics.ObjectType.ConeShape,
            UsdPhysics.ObjectType.MeshShape,
            UsdPhysics.ObjectType.PlaneShape,
        }:
            paths, shape_specs = value
            for xpath, shape_spec in zip(paths, shape_specs, strict=False):
                if warn_invalid_desc(xpath, shape_spec):
                    continue
                path = str(xpath)
                if any(re.match(p, path) for p in ignore_paths):
                    continue
                prim = stage.GetPrimAtPath(xpath)
                collider_is_enabled = _is_enabled_collider(prim)
                # Deformable-owned meshes never reach this loop: the scout excludes them
                # from the native parse. A sim-API mesh seen here was deliberately left
                # rigid (e.g. its body API conflicts with RigidBodyAPI), so import it.
                shape_already_added = path in path_shape_map
                body_path = str(shape_spec.rigidBody)
                if verbose:
                    print(f"collision shape {prim.GetPath()} ({prim.GetTypeName()}), body = {body_path}")
                body_id = path_body_map.get(body_path, -1)
                scale = usd.get_scale(prim, local=False)
                collision_group = builder.default_shape_cfg.collision_group
                collision_groups = tuple(sorted(str(group) for group in shape_spec.collisionGroups))
                material = material_specs[""]
                has_shape_material = len(shape_spec.materials) >= 1
                if has_shape_material:
                    if len(shape_spec.materials) > 1 and verbose:
                        print(f"Warning: More than one material found on shape at '{path}'.\nUsing only the first one.")
                    material = material_specs[str(shape_spec.materials[0])]
                    if verbose:
                        print(
                            f"\tMaterial of '{path}':\tfriction: {material.dynamicFriction},\ttorsional friction: {material.torsionalFriction},\trolling friction: {material.rollingFriction},\trestitution: {material.restitution},\tdensity: {material.density}"
                        )
                elif verbose:
                    print(f"No material found for shape at '{path}'.")

                # Non-MassAPI body mass accumulation in ModelBuilder uses shape cfg density.
                # Use per-shape physics material density when present; otherwise use default density.
                if not collider_is_enabled:
                    # Retain the disabled shape, but exclude it from builder mass aggregation.
                    shape_density = 0.0
                elif has_shape_material:
                    shape_density = material.density
                else:
                    shape_density = default_shape_density
                local_xform = wp.transform(shape_spec.localPos, usd.value_to_warp(shape_spec.localRot))
                if body_id == -1:
                    shape_xform = incoming_world_xform * local_xform
                else:
                    shape_xform = local_xform
                # Extract custom attributes for this shape
                shape_custom_attrs = usd.get_custom_attribute_values(
                    prim, builder_custom_attr_shape, context={"builder": builder}
                )
                if collect_schema_attrs:
                    R.collect_prim_attrs(prim)

                margin_val, gap_val = _resolve_shape_offsets(
                    prim, R, builder.default_shape_cfg, legacy_margin_gap=legacy_margin_gap, verbose=verbose
                )

                has_body_visual_shapes = load_visual_shapes and body_id in bodies_with_visual_shapes
                material_props = visuals.get_material_props_cached(prim)

                # Explicit hide_collision_shapes overrides drawability:
                # if the body already has visual shapes, hide its colliders unconditionally.
                hide_collider_for_body = hide_collision_shapes and has_body_visual_shapes
                # A collider is drawn when USD says it is drawn: ``purpose`` resolving to
                # ``default``/``proxy`` and the prim not being invisible. Not because a
                # render material happens to be bound, and not because nothing else in the
                # scene is visible -- an asset whose geometry is all ``guide`` has no render
                # geometry, and an empty viewport is the honest result of that. Reach for
                # ``force_show_colliders`` to inspect such a scene.
                collider_is_visible = (
                    force_show_colliders or visuals.is_viewport_drawn(prim)
                ) and not hide_collider_for_body
                # Approximating a viewport-drawn collider splits off its authored topology
                # as a visual shape (see the approximation pass below). That copy is subject
                # to ``hide_collision_shapes`` as well, so that the flag does not turn into a
                # no-op for exactly those colliders that carry ``physics:approximation``.
                splits_off_visual_copy = (
                    load_visual_shapes and visuals.is_viewport_drawn(prim) and not hide_collider_for_body
                )

                shape_contact = _resolve_shape_contact(prim, R, material, builder.default_shape_cfg, verbose=verbose)
                shape_ke = shape_contact["ke"]
                shape_kd = shape_contact["kd"]
                shape_kf = shape_contact["kf"]
                shape_ka = shape_contact["ka"]

                shape_color = material_props.get("color")
                carries_texture = material_props.get("texture") is not None and key == UsdPhysics.ObjectType.MeshShape
                if shape_color is None and not carries_texture and collider_is_visible:
                    shape_color = _UNMATERIALED_VISUAL_COLOR

                sdf = _resolve_shape_sdf(prim, R, builder.default_shape_cfg, verbose=verbose)
                has_sdf_api = sdf.has_api
                sdf_max_resolution = sdf.max_resolution
                sdf_narrow_band_range = sdf.narrow_band_range
                sdf_target_voxel_size = sdf.target_voxel_size
                sdf_texture_format = sdf.texture_format
                sdf_padding = sdf.padding
                is_hydroelastic, kh = _resolve_shape_hydroelastic(
                    prim,
                    R,
                    builder.default_shape_cfg,
                    sdf,
                    is_mesh=key == UsdPhysics.ObjectType.MeshShape,
                    verbose=verbose,
                )
                shape_is_solid, inertia_margin, shell_thickness_val = _resolve_shape_shell(prim, R, margin_val)

                if shape_already_added:
                    builder.shape_collision_group[path_shape_map[path]] = collision_group
                    imported_rigid_collider_groups[path] = collision_groups
                    mass_properties.record_collider(
                        path,
                        prim,
                        shape_spec,
                        key,
                        density=shape_density,
                        is_solid=shape_is_solid,
                        thickness=inertia_margin,
                    )
                    if verbose:
                        print(f"Shape at {path} already added; skipping duplicate geometry.")
                    continue

                shape_params = {
                    "body": body_id,
                    "xform": shape_xform,
                    "cfg": ModelBuilder.ShapeConfig(
                        ke=shape_ke,
                        kd=shape_kd,
                        kf=shape_kf,
                        ka=shape_ka,
                        margin=inertia_margin,
                        gap=gap_val,
                        mu=material.dynamicFriction,
                        restitution=material.restitution,
                        mu_torsional=material.torsionalFriction,
                        mu_rolling=material.rollingFriction,
                        density=shape_density,
                        collision_group=collision_group,
                        is_visible=collider_is_visible,
                        sdf_max_resolution=sdf_max_resolution,
                        sdf_narrow_band_range=sdf_narrow_band_range,
                        sdf_target_voxel_size=sdf_target_voxel_size,
                        sdf_texture_format=sdf_texture_format,
                        sdf_padding=sdf_padding,
                        is_hydroelastic=is_hydroelastic,
                        kh=kh,
                        is_solid=shape_is_solid,
                    ),
                    "label": path,
                    "custom_attributes": shape_custom_attrs,
                    "color": shape_color,
                }
                if collider_is_visible:
                    if material_props.get("color") is not None and material_props.get("texture") is None:
                        shape_params["color"] = material_props["color"]
                    if material_props.get("opacity") is not None:
                        shape_params["opacity"] = material_props["opacity"]
                # print(path, shape_params)
                if key == UsdPhysics.ObjectType.CubeShape:
                    hx, hy, hz = shape_spec.halfExtents
                    shape_id = builder.add_shape_box(
                        **shape_params,
                        hx=hx,
                        hy=hy,
                        hz=hz,
                    )
                elif key == UsdPhysics.ObjectType.SphereShape:
                    if not _is_uniform_scale(scale):
                        print(f"Warning: Non-uniform scaling of spheres is not supported, at {path}.")
                    radius = shape_spec.radius
                    shape_id = builder.add_shape_sphere(
                        **shape_params,
                        radius=radius,
                    )
                elif key == UsdPhysics.ObjectType.CapsuleShape:
                    # Apply axis rotation to transform
                    axis = int(shape_spec.axis)
                    shape_params["xform"] = wp.transform(
                        shape_params["xform"].p, shape_params["xform"].q * quat_between_axes(Axis.Z, axis)
                    )
                    radius = shape_spec.radius
                    half_height = shape_spec.halfHeight
                    shape_id = builder.add_shape_capsule(
                        **shape_params,
                        radius=radius,
                        half_height=half_height,
                    )
                elif key == UsdPhysics.ObjectType.CylinderShape:
                    # Apply axis rotation to transform
                    axis = int(shape_spec.axis)
                    shape_params["xform"] = wp.transform(
                        shape_params["xform"].p, shape_params["xform"].q * quat_between_axes(Axis.Z, axis)
                    )
                    radius = shape_spec.radius
                    half_height = shape_spec.halfHeight
                    shape_id = builder.add_shape_cylinder(
                        **shape_params,
                        radius=radius,
                        half_height=half_height,
                    )
                elif key == UsdPhysics.ObjectType.ConeShape:
                    # Apply axis rotation to transform
                    axis = int(shape_spec.axis)
                    shape_params["xform"] = wp.transform(
                        shape_params["xform"].p, shape_params["xform"].q * quat_between_axes(Axis.Z, axis)
                    )
                    radius = shape_spec.radius
                    half_height = shape_spec.halfHeight
                    shape_id = builder.add_shape_cone(
                        **shape_params,
                        radius=radius,
                        half_height=half_height,
                    )
                elif key == UsdPhysics.ObjectType.MeshShape:
                    # Resolve mesh hull vertex limit from schema with fallback to parameter
                    # The mesh needs its render material when anything will draw it: either
                    # the collider itself is visible, or it is viewport geometry whose
                    # authored topology is about to be split off as a visual shape.
                    if collider_is_visible or splits_off_visual_copy:
                        # Drawn colliders should render with the same visual material metadata
                        # as visual-only mesh imports.
                        mesh = visuals.get_mesh_with_visual_material(prim, path_name=path)
                    else:
                        # Not viewport-drawn, but the viewer still draws these under show_collision /
                        # show_static. Mutating the shared cache entry is safe: both caches key on the
                        # prim path, so every consumer resolves the same values.
                        mesh = visuals.get_mesh_cached(prim)
                        visuals.apply_visual_material(mesh, material_props)
                    mesh.maxhullvert = R.get_value(
                        prim,
                        prim_type=PrimType.SHAPE,
                        key="max_hull_vertices",
                        default=mesh_maxhullvert,
                        verbose=verbose,
                    )
                    # add_shape_mesh() rejects SDF cfg fields on meshes; strip them and
                    # write the SDF intent to the builder lists, deferring the build to finalize().
                    mesh_shape_params = dict(shape_params)
                    mesh_shape_params["cfg"] = replace(
                        shape_params["cfg"],
                        sdf_max_resolution=None,
                        sdf_target_voxel_size=None,
                        sdf_narrow_band_range=(-0.1, 0.1),
                        sdf_texture_format="uint16",
                        sdf_padding=None,
                        is_hydroelastic=False,
                    )
                    shape_id = builder.add_shape_mesh(
                        scale=wp.vec3(*shape_spec.meshScale),
                        mesh=mesh,
                        **mesh_shape_params,
                    )
                    builder.shape_sdf_max_resolution[shape_id] = sdf_max_resolution
                    builder.shape_sdf_target_voxel_size[shape_id] = sdf_target_voxel_size
                    builder.shape_sdf_narrow_band_range[shape_id] = sdf_narrow_band_range
                    builder.shape_sdf_texture_format[shape_id] = sdf_texture_format
                    builder.shape_sdf_padding[shape_id] = sdf_padding
                    # kh is a material param; persist regardless of hydro state.
                    builder.shape_material_kh[shape_id] = kh
                    if is_hydroelastic:
                        builder.shape_flags[shape_id] |= ShapeFlags.HYDROELASTIC
                    if collider_is_enabled and not skip_mesh_approximation:
                        approximation = usd.get_attribute(prim, "physics:approximation", None)
                        if approximation is not None:
                            if has_sdf_api and approximation.lower() != "none":
                                # physics:approximation belongs to PhysicsMeshCollisionAPI;
                                # it has no meaning on a NewtonSDFCollisionAPI prim.
                                warnings.warn(
                                    f"{prim.GetPath()}: physics:approximation={approximation!r} is "
                                    f"ignored on a shape with NewtonSDFCollisionAPI applied.",
                                    stacklevel=2,
                                )
                            else:
                                remeshing_method = approximation_to_remeshing_method.get(approximation.lower(), None)
                                if remeshing_method is None:
                                    if verbose:
                                        print(
                                            f"Warning: Unknown physics:approximation attribute '{approximation}' on shape at '{path}'."
                                        )
                                else:
                                    if remeshing_method not in remeshing_queue:
                                        remeshing_queue[remeshing_method] = []
                                    remeshing_queue[remeshing_method].append(shape_id)
                                    if splits_off_visual_copy:
                                        approximated_viewport_shapes.add(shape_id)

                elif key == UsdPhysics.ObjectType.PlaneShape:
                    # Warp uses +Z convention for planes
                    if shape_spec.axis != UsdPhysics.Axis.Z:
                        xform = shape_params["xform"]
                        axis_q = quat_between_axes(Axis.Z, usd_axis_to_axis[shape_spec.axis])
                        shape_params["xform"] = wp.transform(xform.p, xform.q * axis_q)
                    shape_id = builder.add_shape_plane(
                        **shape_params,
                        width=0.0,
                        length=0.0,
                    )
                else:
                    raise NotImplementedError(f"Shape type {key} not supported yet")

                path_shape_map[path] = shape_id
                path_shape_scale[path] = scale
                imported_rigid_collider_groups[path] = collision_groups

                # Restore the real collision margin when shell thickness was substituted.
                # TODO: Consider adding a dedicated shell_thickness field to ShapeConfig
                # so inertia thickness and collision margin don't share the same slot.
                if shell_thickness_val is not None and math.isfinite(float(shell_thickness_val)) and shape_id >= 0:
                    builder.shape_margin[shape_id] = margin_val

                mass_properties.record_collider(
                    path,
                    prim,
                    shape_spec,
                    key,
                    density=shape_density,
                    is_solid=shape_is_solid,
                    thickness=inertia_margin,
                    mesh_source=mesh if key == UsdPhysics.ObjectType.MeshShape else None,
                )

                _collect_filtered_pairs(prim, authored_filtered_path_pairs)

                if not collider_is_enabled:
                    no_collision_shapes.add(shape_id)
                    builder.shape_flags[shape_id] &= ~(ShapeFlags.COLLIDE_SHAPES | ShapeFlags.COLLIDE_PARTICLES)

    # Approximate meshes. ``physics:approximation`` belongs to
    # UsdPhysicsMeshCollisionAPI and is scoped to collision: it says which shape to
    # collide against, not which to draw. Approximating a prim that is viewport
    # geometry therefore splits it in two -- an approximated collider and a visual
    # carrying the authored topology -- rather than replacing what is drawn.
    #
    # Viewport geometry is decided by USD purpose and visibility alone. A prim whose
    # purpose resolves to ``default`` is drawable whether that value was authored or
    # inherited from the fallback, and whether or not a material is bound; the
    # collider display policy that governs pure colliders does not apply to a prim
    # that is also render geometry. ``approximate_meshes`` copies shapes carrying
    # VISIBLE, so mark these before handing them over.
    for remeshing_method, shape_ids in remeshing_queue.items():
        drawn = [s for s in shape_ids if s in approximated_viewport_shapes] if load_visual_shapes else []
        for shape_id in drawn:
            builder.shape_flags[shape_id] |= int(ShapeFlags.VISIBLE)
        if drawn:
            builder.approximate_meshes(method=remeshing_method, shape_indices=drawn, keep_visual_shapes=True)
        # Colliders that are not render geometry keep no visual: there is nothing
        # authored to preserve. If one is on screen it is because the collider
        # display policy put it there, and what it should show is the collider.
        rest = [s for s in shape_ids if s not in set(drawn)]
        if rest:
            builder.approximate_meshes(method=remeshing_method, shape_indices=rest, keep_visual_shapes=False)
