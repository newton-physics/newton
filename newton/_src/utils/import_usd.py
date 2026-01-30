# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import datetime
import itertools
import os
import re
import warnings
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import warp as wp

from ..core import quat_between_axes
from ..core.types import Axis, Transform
from ..geometry import MESH_MAXHULLVERT, ShapeFlags, compute_sphere_inertia
from ..sim.builder import ModelBuilder
from ..sim.joints import ActuatorMode, infer_actuator_mode
from ..sim.model import ModelAttributeFrequency
from ..usd import utils as usd
from ..usd.schema_resolver import PrimType, SchemaResolver, SchemaResolverManager
from ..usd.schemas import SchemaResolverNewton


def parse_usd(
    builder: ModelBuilder,
    source,
    *,
    xform: Transform | None = None,
    only_load_enabled_rigid_bodies: bool = False,
    only_load_enabled_joints: bool = True,
    joint_drive_gains_scaling: float = 1.0,
    verbose: bool = False,
    ignore_paths: list[str] | None = None,
    collapse_fixed_joints: bool = False,
    enable_self_collisions: bool = True,
    apply_up_axis_from_stage: bool = False,
    root_path: str = "/",
    joint_ordering: Literal["bfs", "dfs"] | None = "dfs",
    bodies_follow_joint_ordering: bool = True,
    skip_mesh_approximation: bool = False,
    load_sites: bool = True,
    load_visual_shapes: bool = True,
    hide_collision_shapes: bool = False,
    parse_mujoco_options: bool = True,
    mesh_maxhullvert: int = MESH_MAXHULLVERT,
    schema_resolvers: list[SchemaResolver] | None = None,
    force_position_velocity_actuation: bool = False,
) -> dict[str, Any]:
    """Parses a Universal Scene Description (USD) stage containing UsdPhysics schema definitions for rigid-body articulations and adds the bodies, shapes and joints to the given ModelBuilder.

    The USD description has to be either a path (file name or URL), or an existing USD stage instance that implements the `Stage <https://openusd.org/dev/api/class_usd_stage.html>`_ interface.

    See :ref:`usd_parsing` for more information.

    Args:
        builder (ModelBuilder): The :class:`~newton.ModelBuilder` to add the bodies and joints to.
        source (str | pxr.Usd.Stage): The file path to the USD file, or an existing USD stage instance.
        xform (Transform): The transform to apply to the entire scene.
        only_load_enabled_rigid_bodies (bool): If True, only rigid bodies which do not have `physics:rigidBodyEnabled` set to False are loaded.
        only_load_enabled_joints (bool): If True, only joints which do not have `physics:jointEnabled` set to False are loaded.
        joint_drive_gains_scaling (float): The default scaling of the PD control gains (stiffness and damping), if not set in the PhysicsScene with as "newton:joint_drive_gains_scaling".
        verbose (bool): If True, print additional information about the parsed USD file. Default is False.
        ignore_paths (List[str]): A list of regular expressions matching prim paths to ignore.
        collapse_fixed_joints (bool): If True, fixed joints are removed and the respective bodies are merged. Only considered if not set on the PhysicsScene as "newton:collapse_fixed_joints".
        enable_self_collisions (bool): Determines the default behavior of whether self-collisions are enabled for all shapes within an articulation. If an articulation has the attribute ``physxArticulation:enabledSelfCollisions`` defined, this attribute takes precedence.
        apply_up_axis_from_stage (bool): If True, the up axis of the stage will be used to set :attr:`newton.ModelBuilder.up_axis`. Otherwise, the stage will be rotated such that its up axis aligns with the builder's up axis. Default is False.
        root_path (str): The USD path to import, defaults to "/".
        joint_ordering (str): The ordering of the joints in the simulation. Can be either "bfs" or "dfs" for breadth-first or depth-first search, or ``None`` to keep joints in the order in which they appear in the USD. Default is "dfs".
        bodies_follow_joint_ordering (bool): If True, the bodies are added to the builder in the same order as the joints (parent then child body). Otherwise, bodies are added in the order they appear in the USD. Default is True.
        skip_mesh_approximation (bool): If True, mesh approximation is skipped. Otherwise, meshes are approximated according to the ``physics:approximation`` attribute defined on the UsdPhysicsMeshCollisionAPI (if it is defined). Default is False.
        load_sites (bool): If True, sites (prims with MjcSiteAPI) are loaded as non-colliding reference points. If False, sites are ignored. Default is True.
        load_visual_shapes (bool): If True, non-physics visual geometry is loaded. If False, visual-only shapes are ignored (sites are still controlled by ``load_sites``). Default is True.
        hide_collision_shapes (bool): If True, collision shapes are hidden. Default is False.
        parse_mujoco_options (bool): Whether MuJoCo solver options from the PhysicsScene should be parsed. If False, solver options are not loaded and custom attributes retain their default values. Default is True.
        mesh_maxhullvert (int): Maximum vertices for convex hull approximation of meshes. Note that an authored ``newton:maxHullVertices`` attribute on any shape with a ``NewtonMeshCollisionAPI`` will take priority over this value.
        schema_resolvers (list[SchemaResolver]): Resolver instances in priority order. Default is to only parse Newton-specific attributes.
            Schema resolvers collect per-prim "solver-specific" attributes, see :ref:`schema_resolvers` for more information.
            These include namespaced attributes such as ``newton:*``, ``physx*``
            (e.g., ``physxScene:*``, ``physxRigidBody:*``, ``physxSDFMeshCollision:*``), and ``mjc:*`` that
            are authored in the USD but not strictly required to build the simulation. This is useful for
            inspection, experimentation, or custom pipelines that read these values via
            :attr:`newton.usd.SchemaResolverManager.schema_attrs`.

            .. note::
                Using the ``schema_resolvers`` argument is an experimental feature that may be removed or changed significantly in the future.
        force_position_velocity_actuation (bool): If True and both stiffness (kp) and damping (kd)
            are non-zero, joints use :attr:`~newton.ActuatorMode.POSITION_VELOCITY` actuation mode.
            If False (default), actuator modes are inferred per joint via :func:`newton.infer_actuator_mode`:
            :attr:`~newton.ActuatorMode.POSITION` if stiffness > 0, :attr:`~newton.ActuatorMode.VELOCITY` if only
            damping > 0, :attr:`~newton.ActuatorMode.EFFORT` if a drive is present but both gains are zero
            (direct torque control), or :attr:`~newton.ActuatorMode.NONE` if no drive/actuation is applied.

    Returns:
        dict: Dictionary with the following entries:

        .. list-table::
            :widths: 25 75

            * - ``"fps"``
              - USD stage frames per second
            * - ``"duration"``
              - Difference between end time code and start time code of the USD stage
            * - ``"up_axis"``
              - :class:`Axis` representing the stage's up axis ("X", "Y", or "Z")
            * - ``"path_body_map"``
              - Mapping from prim path (str) of a rigid body prim (e.g. that implements the PhysicsRigidBodyAPI) to the respective body index in :class:`~newton.ModelBuilder`
            * - ``"path_joint_map"``
              - Mapping from prim path (str) of a joint prim (e.g. that implements the PhysicsJointAPI) to the respective joint index in :class:`~newton.ModelBuilder`
            * - ``"path_shape_map"``
              - Mapping from prim path (str) of the UsdGeom to the respective shape index in :class:`~newton.ModelBuilder`
            * - ``"path_shape_scale"``
              - Mapping from prim path (str) of the UsdGeom to its respective 3D world scale
            * - ``"mass_unit"``
              - The stage's Kilograms Per Unit (KGPU) definition (1.0 by default)
            * - ``"linear_unit"``
              - The stage's Meters Per Unit (MPU) definition (1.0 by default)
            * - ``"scene_attributes"``
              - Dictionary of all attributes applied to the PhysicsScene prim
            * - ``"collapse_results"``
              - Dictionary returned by :meth:`newton.ModelBuilder.collapse_fixed_joints` if ``collapse_fixed_joints`` is True, otherwise None.
            * - ``"physics_dt"``
              - The resolved physics scene time step (float or None)
            * - ``"schema_attrs"``
              - Dictionary of collected per-prim schema attributes (dict)
            * - ``"max_solver_iterations"``
              - The resolved maximum solver iterations (int or None)
            * - ``"path_body_relative_transform"``
              - Mapping from prim path to relative transform for bodies merged via ``collapse_fixed_joints``
            * - ``"path_original_body_map"``
              - Mapping from prim path to original body index before ``collapse_fixed_joints``
    """
    if schema_resolvers is None:
        schema_resolvers = [SchemaResolverNewton()]
    collect_schema_attrs = len(schema_resolvers) > 0

    try:
        from pxr import Sdf, Usd, UsdGeom, UsdPhysics  # noqa: PLC0415
    except ImportError as e:
        raise ImportError("Failed to import pxr. Please install USD (e.g. via `pip install usd-core`).") from e

    from .topology import topological_sort_undirected  # noqa: PLC0415

    @dataclass
    class PhysicsMaterial:
        staticFriction: float = builder.default_shape_cfg.mu
        dynamicFriction: float = builder.default_shape_cfg.mu
        torsionalFriction: float = builder.default_shape_cfg.torsional_friction
        rollingFriction: float = builder.default_shape_cfg.rolling_friction
        restitution: float = builder.default_shape_cfg.restitution
        density: float = builder.default_shape_cfg.density

    # load joint defaults
    default_joint_friction = builder.default_joint_cfg.friction
    default_joint_limit_ke = builder.default_joint_cfg.limit_ke
    default_joint_limit_kd = builder.default_joint_cfg.limit_kd
    default_joint_armature = builder.default_joint_cfg.armature

    # load shape defaults
    default_shape_density = builder.default_shape_cfg.density

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

    if ignore_paths is None:
        ignore_paths = []

    usd_axis_to_axis = {
        UsdPhysics.Axis.X: Axis.X,
        UsdPhysics.Axis.Y: Axis.Y,
        UsdPhysics.Axis.Z: Axis.Z,
    }

    if isinstance(source, str):
        stage = Usd.Stage.Open(source, Usd.Stage.LoadAll)
        _raise_on_stage_errors(stage, source)
    else:
        stage = source
        _raise_on_stage_errors(stage, "provided stage")

    DegreesToRadian = float(np.pi / 180)
    mass_unit = 1.0

    try:
        if UsdPhysics.StageHasAuthoredKilogramsPerUnit(stage):
            mass_unit = UsdPhysics.GetStageKilogramsPerUnit(stage)
    except Exception as e:
        if verbose:
            print(f"Failed to get mass unit: {e}")
    linear_unit = 1.0
    try:
        if UsdGeom.StageHasAuthoredMetersPerUnit(stage):
            linear_unit = UsdGeom.GetStageMetersPerUnit(stage)
    except Exception as e:
        if verbose:
            print(f"Failed to get linear unit: {e}")

    non_regex_ignore_paths = [path for path in ignore_paths if ".*" not in path]
    ret_dict = UsdPhysics.LoadUsdPhysicsFromRange(stage, [root_path], excludePaths=non_regex_ignore_paths)

    # Initialize schema resolver according to precedence
    R = SchemaResolverManager(schema_resolvers)

    # mapping from prim path to body index in ModelBuilder
    path_body_map: dict[str, int] = {}
    # mapping from prim path to shape index in ModelBuilder
    path_shape_map: dict[str, int] = {}
    path_shape_scale: dict[str, wp.vec3] = {}
    # mapping from prim path to joint index in ModelBuilder
    path_joint_map: dict[str, int] = {}

    physics_scene_prim = None
    physics_dt = None
    max_solver_iters = None

    visual_shape_cfg = ModelBuilder.ShapeConfig(
        density=0.0,
        has_shape_collision=False,
        has_particle_collision=False,
    )

    # Create a cache for world transforms to avoid recomputing them for each prim.
    xform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())

    def _is_enabled_collider(prim: Usd.Prim) -> bool:
        if collider := UsdPhysics.CollisionAPI(prim):
            return collider.GetCollisionEnabledAttr().Get()
        return False

    def _xform_to_mat44(xform: wp.transform) -> wp.mat44:
        return wp.transform_compose(xform.p, xform.q, wp.vec3(1.0))

    def _load_visual_shapes_impl(
        parent_body_id: int,
        prim: Usd.Prim,
        body_xform: wp.transform | None = None,
    ):
        """Load visual-only shapes (non-physics) for a prim subtree.

        Args:
            parent_body_id: ModelBuilder body id to attach shapes to. Use -1 for
                static shapes that are not bound to any rigid body.
            prim: USD prim to inspect for visual geometry and recurse into.
            body_xform: Rigid body transform actually used by the builder.
                This matches any physics-authored pose, scene-level transforms,
                and incoming transforms that were applied when the body was created.
        """
        if _is_enabled_collider(prim) or prim.HasAPI(UsdPhysics.RigidBodyAPI):
            return
        path_name = str(prim.GetPath())
        if any(re.match(path, path_name) for path in ignore_paths):
            return

        prim_world_mat = usd.get_transform_matrix(prim, local=False, xform_cache=xform_cache)
        if incoming_world_xform is not None and (parent_body_id == -1 or body_xform is not None):
            # Apply the incoming world transform in model space (static shapes or when using body_xform).
            incoming_mat = _xform_to_mat44(incoming_world_xform)
            prim_world_mat = incoming_mat @ prim_world_mat
        if body_xform is not None:
            # Use the body transform used by the builder to avoid USD/physics pose mismatches.
            body_world_mat = _xform_to_mat44(body_xform)
            rel_mat = wp.inverse(body_world_mat) @ prim_world_mat
        else:
            rel_mat = prim_world_mat

        xform_pos, xform_rot, scale = wp.transform_decompose(rel_mat)
        xform = wp.transform(xform_pos, xform_rot)

        if prim.IsInstance():
            proto = prim.GetPrototype()
            for child in proto.GetChildren():
                # remap prototype child path to this instance's path (instance proxy)
                inst_path = child.GetPath().ReplacePrefix(proto.GetPath(), prim.GetPath())
                inst_child = stage.GetPrimAtPath(inst_path)
                _load_visual_shapes_impl(parent_body_id, inst_child, body_xform)
            return
        type_name = str(prim.GetTypeName()).lower()
        if type_name.endswith("joint"):
            return

        shape_id = -1

        # Check if this prim is a site (has MjcSiteAPI applied)
        # First check if the API is formally applied (schema is registered)
        is_site = prim.HasAPI("MjcSiteAPI")
        # If not, check the apiSchemas metadata directly (for unregistered schemas)
        if not is_site:
            schemas_listop = prim.GetMetadata("apiSchemas")
            if schemas_listop:
                all_schemas = (
                    list(schemas_listop.prependedItems)
                    + list(schemas_listop.appendedItems)
                    + list(schemas_listop.explicitItems)
                )
                is_site = "MjcSiteAPI" in all_schemas

        # Skip based on granular loading flags
        if is_site and not load_sites:
            return
        if not is_site and not load_visual_shapes:
            return

        if path_name not in path_shape_map:
            if type_name == "cube":
                size = usd.get_float(prim, "size", 2.0)
                side_lengths = scale * size
                shape_id = builder.add_shape_box(
                    parent_body_id,
                    xform,
                    hx=side_lengths[0] / 2,
                    hy=side_lengths[1] / 2,
                    hz=side_lengths[2] / 2,
                    cfg=visual_shape_cfg,
                    as_site=is_site,
                    key=path_name,
                )
            elif type_name == "sphere":
                if not (scale[0] == scale[1] == scale[2]):
                    print("Warning: Non-uniform scaling of spheres is not supported.")
                radius = usd.get_float(prim, "radius", 1.0) * max(scale)
                shape_id = builder.add_shape_sphere(
                    parent_body_id,
                    xform,
                    radius,
                    cfg=visual_shape_cfg,
                    as_site=is_site,
                    key=path_name,
                )
            elif type_name == "plane":
                axis = usd.get_gprim_axis(prim)
                plane_xform = xform
                # Apply axis rotation to transform
                xform = wp.transform(xform.p, xform.q * quat_between_axes(Axis.Z, axis))
                width = usd.get_float(prim, "width", 0.0) * scale[0]
                length = usd.get_float(prim, "length", 0.0) * scale[1]
                shape_id = builder.add_shape_plane(
                    body=parent_body_id,
                    xform=plane_xform,
                    width=width,
                    length=length,
                    cfg=visual_shape_cfg,
                    key=path_name,
                )
            elif type_name == "capsule":
                axis = usd.get_gprim_axis(prim)
                radius = usd.get_float(prim, "radius", 0.5) * scale[0]
                half_height = usd.get_float(prim, "height", 2.0) / 2 * scale[1]
                # Apply axis rotation to transform
                xform = wp.transform(xform.p, xform.q * quat_between_axes(Axis.Z, axis))
                shape_id = builder.add_shape_capsule(
                    parent_body_id,
                    xform,
                    radius,
                    half_height,
                    cfg=visual_shape_cfg,
                    as_site=is_site,
                    key=path_name,
                )
            elif type_name == "cylinder":
                axis = usd.get_gprim_axis(prim)
                radius = usd.get_float(prim, "radius", 0.5) * scale[0]
                half_height = usd.get_float(prim, "height", 2.0) / 2 * scale[1]
                # Apply axis rotation to transform
                xform = wp.transform(xform.p, xform.q * quat_between_axes(Axis.Z, axis))
                shape_id = builder.add_shape_cylinder(
                    parent_body_id,
                    xform,
                    radius,
                    half_height,
                    cfg=visual_shape_cfg,
                    as_site=is_site,
                    key=path_name,
                )
            elif type_name == "cone":
                axis = usd.get_gprim_axis(prim)
                radius = usd.get_float(prim, "radius", 0.5) * scale[0]
                half_height = usd.get_float(prim, "height", 2.0) / 2 * scale[1]
                # Apply axis rotation to transform
                xform = wp.transform(xform.p, xform.q * quat_between_axes(Axis.Z, axis))
                shape_id = builder.add_shape_cone(
                    parent_body_id,
                    xform,
                    radius,
                    half_height,
                    cfg=visual_shape_cfg,
                    as_site=is_site,
                    key=path_name,
                )
            elif type_name == "mesh":
                mesh = usd.get_mesh(prim)
                shape_id = builder.add_shape_mesh(
                    parent_body_id,
                    xform,
                    scale=scale,
                    mesh=mesh,
                    cfg=visual_shape_cfg,
                    key=path_name,
                )
            elif len(type_name) > 0 and type_name != "xform" and verbose:
                print(f"Warning: Unsupported geometry type {type_name} at {path_name} while loading visual shapes.")

            if shape_id >= 0:
                path_shape_map[path_name] = shape_id
                path_shape_scale[path_name] = scale
                if verbose:
                    print(f"Added visual shape {path_name} ({type_name}) with id {shape_id}.")

        for child in prim.GetChildren():
            _load_visual_shapes_impl(parent_body_id, child, body_xform)

    def add_body(prim: Usd.Prim, xform: wp.transform, key: str, armature: float) -> int:
        """Add a rigid body to the builder and optionally load its visual shapes and sites among the body prim's children. Returns the resulting body index."""
        # Extract custom attributes for this body
        body_custom_attrs = usd.get_custom_attribute_values(prim, builder_custom_attr_body)

        b = builder.add_link(
            xform=xform,
            key=key,
            armature=armature,
            custom_attributes=body_custom_attrs,
        )
        path_body_map[key] = b
        if load_sites or load_visual_shapes:
            for child in prim.GetChildren():
                _load_visual_shapes_impl(b, child, body_xform=xform)
        return b

    def parse_body(
        rigid_body_desc: UsdPhysics.RigidBodyDesc,
        prim: Usd.Prim,
        incoming_xform: wp.transform | None = None,
        add_body_to_builder: bool = True,
    ) -> int | dict[str, Any]:
        """Parses a rigid body description.
        If `add_body_to_builder` is True, adds it to the builder and returns the resulting body index.
        Otherwise returns a dictionary of body data that can be passed to ModelBuilder.add_body()."""
        nonlocal path_body_map
        nonlocal physics_scene_prim

        if not rigid_body_desc.rigidBodyEnabled and only_load_enabled_rigid_bodies:
            return -1

        rot = rigid_body_desc.rotation
        origin = wp.transform(rigid_body_desc.position, usd.from_gfquat(rot))
        if incoming_xform is not None:
            origin = wp.mul(incoming_xform, origin)
        path = str(prim.GetPath())

        body_armature = usd.get_float_with_fallback(
            (prim, physics_scene_prim), "newton:armature", builder.default_body_armature
        )

        if add_body_to_builder:
            return add_body(prim, origin, path, body_armature)
        else:
            return {
                "prim": prim,
                "xform": origin,
                "key": path,
                "armature": body_armature,
            }

    def resolve_joint_parent_child(
        joint_desc: UsdPhysics.JointDesc,
        body_index_map: dict[str, int],
        get_transforms: bool = True,
    ):
        """Resolve the parent and child of a joint and return their parent + child transforms if requested."""
        if get_transforms:
            parent_tf = wp.transform(joint_desc.localPose0Position, usd.from_gfquat(joint_desc.localPose0Orientation))
            child_tf = wp.transform(joint_desc.localPose1Position, usd.from_gfquat(joint_desc.localPose1Orientation))
        else:
            parent_tf = None
            child_tf = None

        parent_path = str(joint_desc.body0)
        child_path = str(joint_desc.body1)
        parent_id = body_index_map.get(parent_path, -1)
        child_id = body_index_map.get(child_path, -1)
        # If child_id is -1, swap parent and child
        if child_id == -1:
            if parent_id == -1:
                raise ValueError(f"Unable to parse joint {joint_desc.primPath}: both bodies unresolved")
            parent_id, child_id = child_id, parent_id
            if get_transforms:
                parent_tf, child_tf = child_tf, parent_tf
            if verbose:
                print(f"Joint {joint_desc.primPath} connects {parent_path} to world")
        if get_transforms:
            return parent_id, child_id, parent_tf, child_tf
        else:
            return parent_id, child_id

    def parse_joint(
        joint_desc: UsdPhysics.JointDesc,
        incoming_xform: wp.transform | None = None,
    ) -> int | None:
        """Parse a joint description and add it to the builder. Returns the resulting joint index if successful, None otherwise."""
        if not joint_desc.jointEnabled and only_load_enabled_joints:
            return None
        key = joint_desc.type
        joint_path = str(joint_desc.primPath)
        joint_prim = stage.GetPrimAtPath(joint_desc.primPath)
        # collect engine-specific attributes on the joint prim if requested
        if collect_schema_attrs:
            R.collect_prim_attrs(joint_prim)
        parent_id, child_id, parent_tf, child_tf = resolve_joint_parent_child(  # pyright: ignore[reportAssignmentType]
            joint_desc, path_body_map, get_transforms=True
        )

        if incoming_xform is not None:
            parent_tf = incoming_xform * parent_tf

        joint_armature = R.get_value(
            joint_prim, prim_type=PrimType.JOINT, key="armature", default=default_joint_armature, verbose=verbose
        )
        joint_friction = R.get_value(
            joint_prim, prim_type=PrimType.JOINT, key="friction", default=default_joint_friction, verbose=verbose
        )

        # Extract custom attributes for this joint
        joint_custom_attrs = usd.get_custom_attribute_values(joint_prim, builder_custom_attr_joint)
        joint_params = {
            "parent": parent_id,
            "child": child_id,
            "parent_xform": parent_tf,
            "child_xform": child_tf,
            "key": joint_path,
            "enabled": joint_desc.jointEnabled,
            "custom_attributes": joint_custom_attrs,
        }

        joint_index: int | None = None
        if key == UsdPhysics.ObjectType.FixedJoint:
            joint_index = builder.add_joint_fixed(**joint_params)
        elif key == UsdPhysics.ObjectType.RevoluteJoint or key == UsdPhysics.ObjectType.PrismaticJoint:
            # we need to scale the builder defaults for the joint limits to degrees for revolute joints
            if key == UsdPhysics.ObjectType.RevoluteJoint:
                limit_gains_scaling = DegreesToRadian
            else:
                limit_gains_scaling = 1.0

            # Resolve limit gains with precedence, fallback to builder defaults when missing
            current_joint_limit_ke = R.get_value(
                joint_prim,
                prim_type=PrimType.JOINT,
                key="limit_angular_ke" if key == UsdPhysics.ObjectType.RevoluteJoint else "limit_linear_ke",
                default=default_joint_limit_ke * limit_gains_scaling,
                verbose=verbose,
            )
            current_joint_limit_kd = R.get_value(
                joint_prim,
                prim_type=PrimType.JOINT,
                key="limit_angular_kd" if key == UsdPhysics.ObjectType.RevoluteJoint else "limit_linear_kd",
                default=default_joint_limit_kd * limit_gains_scaling,
                verbose=verbose,
            )
            joint_params["axis"] = usd_axis_to_axis[joint_desc.axis]
            joint_params["limit_lower"] = joint_desc.limit.lower
            joint_params["limit_upper"] = joint_desc.limit.upper
            joint_params["limit_ke"] = current_joint_limit_ke
            joint_params["limit_kd"] = current_joint_limit_kd
            joint_params["armature"] = joint_armature
            joint_params["friction"] = joint_friction
            if joint_desc.drive.enabled:
                target_vel = joint_desc.drive.targetVelocity
                target_pos = joint_desc.drive.targetPosition
                target_ke = joint_desc.drive.stiffness
                target_kd = joint_desc.drive.damping

                joint_params["target_vel"] = target_vel
                joint_params["target_pos"] = target_pos
                joint_params["target_ke"] = target_ke
                joint_params["target_kd"] = target_kd
                joint_params["effort_limit"] = joint_desc.drive.forceLimit

                joint_params["actuator_mode"] = infer_actuator_mode(
                    target_ke, target_kd, force_position_velocity_actuation, has_drive=True
                )
            else:
                joint_params["actuator_mode"] = ActuatorMode.NONE

            # Read initial joint state BEFORE creating/overwriting USD attributes
            initial_position = None
            initial_velocity = None
            dof_type = "linear" if key == UsdPhysics.ObjectType.PrismaticJoint else "angular"

            # Resolve initial joint state from schema resolver
            if dof_type == "angular":
                initial_position = R.get_value(
                    joint_prim, PrimType.JOINT, "angular_position", default=None, verbose=verbose
                )
                initial_velocity = R.get_value(
                    joint_prim, PrimType.JOINT, "angular_velocity", default=None, verbose=verbose
                )
            else:  # linear
                initial_position = R.get_value(
                    joint_prim, PrimType.JOINT, "linear_position", default=None, verbose=verbose
                )
                initial_velocity = R.get_value(
                    joint_prim, PrimType.JOINT, "linear_velocity", default=None, verbose=verbose
                )

            if key == UsdPhysics.ObjectType.PrismaticJoint:
                joint_index = builder.add_joint_prismatic(**joint_params)
            else:
                if joint_desc.drive.enabled:
                    joint_params["target_pos"] *= DegreesToRadian
                    joint_params["target_vel"] *= DegreesToRadian
                    joint_params["target_kd"] /= DegreesToRadian / joint_drive_gains_scaling
                    joint_params["target_ke"] /= DegreesToRadian / joint_drive_gains_scaling

                joint_params["limit_lower"] *= DegreesToRadian
                joint_params["limit_upper"] *= DegreesToRadian
                joint_params["limit_ke"] /= DegreesToRadian
                joint_params["limit_kd"] /= DegreesToRadian

                joint_index = builder.add_joint_revolute(**joint_params)
        elif key == UsdPhysics.ObjectType.SphericalJoint:
            joint_index = builder.add_joint_ball(**joint_params)
        elif key == UsdPhysics.ObjectType.D6Joint:
            linear_axes = []
            angular_axes = []
            num_dofs = 0
            # Store initial state for D6 joints
            d6_initial_positions = {}
            d6_initial_velocities = {}
            # Track which axes were added as DOFs (in order)
            d6_dof_axes = []
            # print(joint_desc.jointLimits, joint_desc.jointDrives)
            # print(joint_desc.body0)
            # print(joint_desc.body1)
            # print(joint_desc.jointLimits)
            # print("Limits")
            # for limit in joint_desc.jointLimits:
            #     print("joint_path :", joint_path, limit.first, limit.second.lower, limit.second.upper)
            # print("Drives")
            # for drive in joint_desc.jointDrives:
            #     print("joint_path :", joint_path, drive.first, drive.second.targetPosition, drive.second.targetVelocity)

            for limit in joint_desc.jointLimits:
                dof = limit.first
                if limit.second.enabled:
                    limit_lower = limit.second.lower
                    limit_upper = limit.second.upper
                else:
                    limit_lower = builder.default_joint_cfg.limit_lower
                    limit_upper = builder.default_joint_cfg.limit_upper

                free_axis = limit_lower < limit_upper

                def define_joint_targets(dof, joint_desc):
                    target_pos = 0.0  # TODO: parse target from state:*:physics:appliedForce usd attribute when no drive is present
                    target_vel = 0.0
                    target_ke = 0.0
                    target_kd = 0.0
                    effort_limit = np.inf
                    has_drive = False
                    for drive in joint_desc.jointDrives:
                        if drive.first != dof:
                            continue
                        if drive.second.enabled:
                            has_drive = True
                            target_vel = drive.second.targetVelocity
                            target_pos = drive.second.targetPosition
                            target_ke = drive.second.stiffness
                            target_kd = drive.second.damping
                            effort_limit = drive.second.forceLimit
                    actuator_mode = infer_actuator_mode(
                        target_ke, target_kd, force_position_velocity_actuation, has_drive=has_drive
                    )
                    return target_pos, target_vel, target_ke, target_kd, effort_limit, actuator_mode

                target_pos, target_vel, target_ke, target_kd, effort_limit, actuator_mode = define_joint_targets(
                    dof, joint_desc
                )

                _trans_axes = {
                    UsdPhysics.JointDOF.TransX: (1.0, 0.0, 0.0),
                    UsdPhysics.JointDOF.TransY: (0.0, 1.0, 0.0),
                    UsdPhysics.JointDOF.TransZ: (0.0, 0.0, 1.0),
                }
                _rot_axes = {
                    UsdPhysics.JointDOF.RotX: (1.0, 0.0, 0.0),
                    UsdPhysics.JointDOF.RotY: (0.0, 1.0, 0.0),
                    UsdPhysics.JointDOF.RotZ: (0.0, 0.0, 1.0),
                }
                _rot_names = {
                    UsdPhysics.JointDOF.RotX: "rotX",
                    UsdPhysics.JointDOF.RotY: "rotY",
                    UsdPhysics.JointDOF.RotZ: "rotZ",
                }
                if free_axis and dof in _trans_axes:
                    # Per-axis translation names: transX/transY/transZ
                    trans_name = {
                        UsdPhysics.JointDOF.TransX: "transX",
                        UsdPhysics.JointDOF.TransY: "transY",
                        UsdPhysics.JointDOF.TransZ: "transZ",
                    }[dof]
                    # Store initial state for this axis
                    d6_initial_positions[trans_name] = R.get_value(
                        joint_prim,
                        PrimType.JOINT,
                        f"{trans_name}_position",
                        default=None,
                        verbose=verbose,
                    )
                    d6_initial_velocities[trans_name] = R.get_value(
                        joint_prim,
                        PrimType.JOINT,
                        f"{trans_name}_velocity",
                        default=None,
                        verbose=verbose,
                    )
                    current_joint_limit_ke = R.get_value(
                        joint_prim,
                        prim_type=PrimType.JOINT,
                        key=f"limit_{trans_name}_ke",
                        default=default_joint_limit_ke,
                        verbose=verbose,
                    )
                    current_joint_limit_kd = R.get_value(
                        joint_prim,
                        prim_type=PrimType.JOINT,
                        key=f"limit_{trans_name}_kd",
                        default=default_joint_limit_kd,
                        verbose=verbose,
                    )
                    linear_axes.append(
                        ModelBuilder.JointDofConfig(
                            axis=_trans_axes[dof],
                            limit_lower=limit_lower,
                            limit_upper=limit_upper,
                            limit_ke=current_joint_limit_ke,
                            limit_kd=current_joint_limit_kd,
                            target_pos=target_pos,
                            target_vel=target_vel,
                            target_ke=target_ke,
                            target_kd=target_kd,
                            armature=joint_armature,
                            effort_limit=effort_limit,
                            friction=joint_friction,
                            actuator_mode=actuator_mode,
                        )
                    )
                    # Track that this axis was added as a DOF
                    d6_dof_axes.append(trans_name)
                elif free_axis and dof in _rot_axes:
                    # Resolve per-axis rotational gains
                    rot_name = _rot_names[dof]
                    # Store initial state for this axis
                    d6_initial_positions[rot_name] = R.get_value(
                        joint_prim,
                        PrimType.JOINT,
                        f"{rot_name}_position",
                        default=None,
                        verbose=verbose,
                    )
                    d6_initial_velocities[rot_name] = R.get_value(
                        joint_prim,
                        PrimType.JOINT,
                        f"{rot_name}_velocity",
                        default=None,
                        verbose=verbose,
                    )
                    current_joint_limit_ke = R.get_value(
                        joint_prim,
                        prim_type=PrimType.JOINT,
                        key=f"limit_{rot_name}_ke",
                        default=default_joint_limit_ke * DegreesToRadian,
                        verbose=verbose,
                    )
                    current_joint_limit_kd = R.get_value(
                        joint_prim,
                        prim_type=PrimType.JOINT,
                        key=f"limit_{rot_name}_kd",
                        default=default_joint_limit_kd * DegreesToRadian,
                        verbose=verbose,
                    )

                    angular_axes.append(
                        ModelBuilder.JointDofConfig(
                            axis=_rot_axes[dof],
                            limit_lower=limit_lower * DegreesToRadian,
                            limit_upper=limit_upper * DegreesToRadian,
                            limit_ke=current_joint_limit_ke / DegreesToRadian,
                            limit_kd=current_joint_limit_kd / DegreesToRadian,
                            target_pos=target_pos * DegreesToRadian,
                            target_vel=target_vel * DegreesToRadian,
                            target_ke=target_ke / DegreesToRadian / joint_drive_gains_scaling,
                            target_kd=target_kd / DegreesToRadian / joint_drive_gains_scaling,
                            armature=joint_armature,
                            effort_limit=effort_limit,
                            friction=joint_friction,
                            actuator_mode=actuator_mode,
                        )
                    )
                    # Track that this axis was added as a DOF
                    d6_dof_axes.append(rot_name)
                    num_dofs += 1

            joint_index = builder.add_joint_d6(**joint_params, linear_axes=linear_axes, angular_axes=angular_axes)
        elif key == UsdPhysics.ObjectType.DistanceJoint:
            if joint_desc.limit.enabled and joint_desc.minEnabled:
                min_dist = joint_desc.limit.lower
            else:
                min_dist = -1.0  # no limit
            if joint_desc.limit.enabled and joint_desc.maxEnabled:
                max_dist = joint_desc.limit.upper
            else:
                max_dist = -1.0
            joint_index = builder.add_joint_distance(**joint_params, min_distance=min_dist, max_distance=max_dist)
        else:
            raise NotImplementedError(f"Unsupported joint type {key}")

        if joint_index is None:
            raise ValueError(f"Failed to add joint {joint_path}")

        # map the joint path to the index at insertion time
        path_joint_map[joint_path] = joint_index

        # Apply saved initial joint state after joint creation
        if key in (UsdPhysics.ObjectType.RevoluteJoint, UsdPhysics.ObjectType.PrismaticJoint):
            if initial_position is not None:
                q_start = builder.joint_q_start[joint_index]
                if key == UsdPhysics.ObjectType.RevoluteJoint:
                    builder.joint_q[q_start] = initial_position * DegreesToRadian
                else:
                    builder.joint_q[q_start] = initial_position
                if verbose:
                    joint_type_str = "revolute" if key == UsdPhysics.ObjectType.RevoluteJoint else "prismatic"
                    print(
                        f"Set {joint_type_str} joint {joint_index} position to {initial_position} ({'rad' if key == UsdPhysics.ObjectType.RevoluteJoint else 'm'})"
                    )
            if initial_velocity is not None:
                qd_start = builder.joint_qd_start[joint_index]
                if key == UsdPhysics.ObjectType.RevoluteJoint:
                    builder.joint_qd[qd_start] = initial_velocity  # velocity is already in rad/s
                else:
                    builder.joint_qd[qd_start] = initial_velocity
                if verbose:
                    joint_type_str = "revolute" if key == UsdPhysics.ObjectType.RevoluteJoint else "prismatic"
                    print(f"Set {joint_type_str} joint {joint_index} velocity to {initial_velocity} rad/s")
        elif key == UsdPhysics.ObjectType.D6Joint:
            # Apply D6 joint initial state
            q_start = builder.joint_q_start[joint_index]
            qd_start = builder.joint_qd_start[joint_index]

            # Get joint coordinate and DOF ranges
            if joint_index + 1 < len(builder.joint_q_start):
                q_end = builder.joint_q_start[joint_index + 1]
                qd_end = builder.joint_qd_start[joint_index + 1]
            else:
                q_end = len(builder.joint_q)
                qd_end = len(builder.joint_qd)

            # Apply initial values for each axis that was actually added as a DOF
            for dof_idx, axis_name in enumerate(d6_dof_axes):
                if dof_idx >= (qd_end - qd_start):
                    break

                is_rot = axis_name.startswith("rot")
                pos = d6_initial_positions.get(axis_name)
                vel = d6_initial_velocities.get(axis_name)

                if pos is not None and q_start + dof_idx < q_end:
                    coord_val = pos * DegreesToRadian if is_rot else pos
                    builder.joint_q[q_start + dof_idx] = coord_val
                    if verbose:
                        print(f"Set D6 joint {joint_index} {axis_name} position to {pos} ({'deg' if is_rot else 'm'})")

                if vel is not None and qd_start + dof_idx < qd_end:
                    vel_val = vel  # D6 velocities are already in correct units
                    builder.joint_qd[qd_start + dof_idx] = vel_val
                    if verbose:
                        print(f"Set D6 joint {joint_index} {axis_name} velocity to {vel} rad/s")

        return joint_index

    # Looking for and parsing the attributes on PhysicsScene prims
    scene_attributes = {}
    physics_scene_prim = None
    if UsdPhysics.ObjectType.Scene in ret_dict:
        paths, scene_descs = ret_dict[UsdPhysics.ObjectType.Scene]
        if len(paths) > 1 and verbose:
            print("Only the first PhysicsScene is considered")
        path, scene_desc = paths[0], scene_descs[0]
        if verbose:
            print("Found PhysicsScene:", path)
            print("Gravity direction:", scene_desc.gravityDirection)
            print("Gravity magnitude:", scene_desc.gravityMagnitude)
        builder.gravity = -scene_desc.gravityMagnitude * linear_unit

        # Storing Physics Scene attributes
        physics_scene_prim = stage.GetPrimAtPath(path)
        for a in physics_scene_prim.GetAttributes():
            scene_attributes[a.GetName()] = a.Get()

        # Parse custom attribute declarations from PhysicsScene prim
        # This must happen before processing any other prims
        declarations = usd.get_custom_attribute_declarations(physics_scene_prim)
        for attr in declarations.values():
            builder.add_custom_attribute(attr)

        # Updating joint_drive_gains_scaling if set of the PhysicsScene
        joint_drive_gains_scaling = usd.get_float(
            physics_scene_prim, "newton:joint_drive_gains_scaling", joint_drive_gains_scaling
        )

        time_steps_per_second = R.get_value(
            physics_scene_prim, prim_type=PrimType.SCENE, key="time_steps_per_second", default=1000, verbose=verbose
        )
        physics_dt = (1.0 / time_steps_per_second) if time_steps_per_second > 0 else 0.001

        gravity_enabled = R.get_value(
            physics_scene_prim, prim_type=PrimType.SCENE, key="gravity_enabled", default=True, verbose=verbose
        )
        if not gravity_enabled:
            builder.gravity = 0.0
        max_solver_iters = R.get_value(
            physics_scene_prim, prim_type=PrimType.SCENE, key="max_solver_iterations", default=None, verbose=verbose
        )

    stage_up_axis = Axis.from_string(str(UsdGeom.GetStageUpAxis(stage)))

    if apply_up_axis_from_stage:
        builder.up_axis = stage_up_axis
        axis_xform = wp.transform_identity()
        if verbose:
            print(f"Using stage up axis {stage_up_axis} as builder up axis")
    else:
        axis_xform = wp.transform(wp.vec3(0.0), quat_between_axes(stage_up_axis, builder.up_axis))
        if verbose:
            print(f"Rotating stage to align its up axis {stage_up_axis} with builder up axis {builder.up_axis}")
    if xform is None:
        incoming_world_xform = axis_xform
    else:
        incoming_world_xform = wp.transform(*xform) * axis_xform

    if verbose:
        print(
            f"Scaling PD gains by (joint_drive_gains_scaling / DegreesToRadian) = {joint_drive_gains_scaling / DegreesToRadian}, default scale for joint_drive_gains_scaling=1 is 1.0/DegreesToRadian = {1.0 / DegreesToRadian}"
        )

    # Process custom attributes defined for different kinds of prim.
    # Note that at this time we may have more custom attributes than before since they may have been
    # declared on the PhysicsScene prim.
    builder_custom_attr_shape: list[ModelBuilder.CustomAttribute] = builder.get_custom_attributes_by_frequency(
        [ModelAttributeFrequency.SHAPE]
    )
    builder_custom_attr_body: list[ModelBuilder.CustomAttribute] = builder.get_custom_attributes_by_frequency(
        [ModelAttributeFrequency.BODY]
    )
    builder_custom_attr_joint: list[ModelBuilder.CustomAttribute] = builder.get_custom_attributes_by_frequency(
        [ModelAttributeFrequency.JOINT, ModelAttributeFrequency.JOINT_DOF, ModelAttributeFrequency.JOINT_COORD]
    )
    builder_custom_attr_articulation: list[ModelBuilder.CustomAttribute] = builder.get_custom_attributes_by_frequency(
        [ModelAttributeFrequency.ARTICULATION]
    )

    if physics_scene_prim is not None:
        # Collect schema-defined attributes from the scene prim for inspection (e.g., mjc:* attributes)
        if collect_schema_attrs:
            R.collect_prim_attrs(physics_scene_prim)

        # Extract custom attributes for model (ONCE and WORLD frequency) from the PhysicsScene prim
        # WORLD frequency attributes use index 0 here; they get remapped during add_world()
        builder_custom_attr_model: list[ModelBuilder.CustomAttribute] = [
            attr
            for attr in builder.custom_attributes.values()
            if attr.frequency in (ModelAttributeFrequency.ONCE, ModelAttributeFrequency.WORLD)
        ]

        # Filter out MuJoCo attributes if parse_mujoco_options is False
        if not parse_mujoco_options:
            builder_custom_attr_model = [attr for attr in builder_custom_attr_model if attr.namespace != "mujoco"]

        # Read custom attribute values from the PhysicsScene prim
        scene_custom_attrs = usd.get_custom_attribute_values(physics_scene_prim, builder_custom_attr_model)
        scene_attributes.update(scene_custom_attrs)

        # Set values on builder's custom attributes
        for key, value in scene_custom_attrs.items():
            if key in builder.custom_attributes:
                builder.custom_attributes[key].values[0] = value

    joint_descriptions = {}
    # stores physics spec for every RigidBody in the selected range
    body_specs = {}
    # set of prim paths of rigid bodies that are ignored
    # (to avoid repeated regex evaluations)
    ignored_body_paths = set()
    material_specs = {}
    # maps from rigid body path to density value if it has been defined
    body_density = {}
    # maps from articulation_id to list of body_ids
    articulation_bodies = {}

    # TODO: uniform interface for iterating
    def data_for_key(physics_utils_results, key):
        if key not in physics_utils_results:
            return
        if verbose:
            print(physics_utils_results[key])

        yield from zip(*physics_utils_results[key], strict=False)

    # Setting up the default material
    material_specs[""] = PhysicsMaterial()

    def warn_invalid_desc(path, descriptor) -> bool:
        if not descriptor.isValid:
            warnings.warn(
                f'Warning: Invalid {type(descriptor).__name__} descriptor for prim at path "{path}".',
                stacklevel=2,
            )
            return True
        return False

    # Parsing physics materials from the stage
    for sdf_path, desc in data_for_key(ret_dict, UsdPhysics.ObjectType.RigidBodyMaterial):
        if warn_invalid_desc(sdf_path, desc):
            continue
        prim = stage.GetPrimAtPath(sdf_path)
        material_specs[str(sdf_path)] = PhysicsMaterial(
            staticFriction=desc.staticFriction,
            dynamicFriction=desc.dynamicFriction,
            restitution=desc.restitution,
            torsionalFriction=R.get_value(
                prim,
                prim_type=PrimType.MATERIAL,
                key="torsional_friction",
                default=builder.default_shape_cfg.torsional_friction,
                verbose=verbose,
            ),
            rollingFriction=R.get_value(
                prim,
                prim_type=PrimType.MATERIAL,
                key="rolling_friction",
                default=builder.default_shape_cfg.rolling_friction,
                verbose=verbose,
            ),
            # TODO: if desc.density is 0, then we should look for mass somewhere
            density=desc.density if desc.density > 0.0 else default_shape_density,
        )

    if UsdPhysics.ObjectType.RigidBody in ret_dict:
        prim_paths, rigid_body_descs = ret_dict[UsdPhysics.ObjectType.RigidBody]
        for prim_path, rigid_body_desc in zip(prim_paths, rigid_body_descs, strict=False):
            if warn_invalid_desc(prim_path, rigid_body_desc):
                continue
            body_path = str(prim_path)
            if any(re.match(p, body_path) for p in ignore_paths):
                ignored_body_paths.add(body_path)
                continue
            body_specs[body_path] = rigid_body_desc
            body_density[body_path] = default_shape_density
            prim = stage.GetPrimAtPath(prim_path)
            # Marking for deprecation --->
            if prim.HasRelationship("material:binding:physics"):
                other_paths = prim.GetRelationship("material:binding:physics").GetTargets()
                if len(other_paths) > 0:
                    material = material_specs[str(other_paths[0])]
                    if material.density > 0.0:
                        body_density[body_path] = material.density

            if prim.HasAPI(UsdPhysics.MassAPI):
                if usd.has_attribute(prim, "physics:density"):
                    d = usd.get_float(prim, "physics:density")
                    density = d * mass_unit  # / (linear_unit**3)
                    body_density[body_path] = density
            # <--- Marking for deprecation

    # Collect joint descriptions regardless of whether articulations are authored.
    for key, value in ret_dict.items():
        if key in {
            UsdPhysics.ObjectType.FixedJoint,
            UsdPhysics.ObjectType.RevoluteJoint,
            UsdPhysics.ObjectType.PrismaticJoint,
            UsdPhysics.ObjectType.SphericalJoint,
            UsdPhysics.ObjectType.D6Joint,
            UsdPhysics.ObjectType.DistanceJoint,
        }:
            paths, joint_specs = value
            for path, joint_spec in zip(paths, joint_specs, strict=False):
                joint_descriptions[str(path)] = joint_spec

    # maps from articulation_id to bool indicating if self-collisions are enabled
    articulation_has_self_collision = {}

    if UsdPhysics.ObjectType.Articulation in ret_dict:
        paths, articulation_descs = ret_dict[UsdPhysics.ObjectType.Articulation]

        articulation_id = builder.articulation_count
        parent_prim = None
        body_data = {}
        for path, desc in zip(paths, articulation_descs, strict=False):
            if warn_invalid_desc(path, desc):
                continue
            articulation_path = str(path)
            if any(re.match(p, articulation_path) for p in ignore_paths):
                continue
            articulation_prim = stage.GetPrimAtPath(path)
            articulation_root_xform = usd.get_transform(articulation_prim, local=False, xform_cache=xform_cache)
            # Joints are authored in the articulation-root frame, so always compose with it.
            articulation_incoming_xform = incoming_world_xform * articulation_root_xform
            # Collect engine-specific attributes for the articulation root on first encounter
            if collect_schema_attrs:
                R.collect_prim_attrs(articulation_prim)
                # Also collect on the parent prim (e.g. Xform with PhysxArticulationAPI)
                try:
                    parent_prim = articulation_prim.GetParent()
                except Exception:
                    parent_prim = None
                if parent_prim is not None and parent_prim.IsValid():
                    R.collect_prim_attrs(parent_prim)

            # Extract custom attributes for articulation frequency from the articulation root prim
            # (the one with PhysicsArticulationRootAPI, typically the articulation_prim itself or its parent)
            articulation_custom_attrs = {}
            # First check if articulation_prim itself has the PhysicsArticulationRootAPI
            if articulation_prim.HasAPI(UsdPhysics.ArticulationRootAPI):
                if verbose:
                    print(f"Extracting articulation custom attributes from {articulation_prim.GetPath()}")
                articulation_custom_attrs = usd.get_custom_attribute_values(
                    articulation_prim, builder_custom_attr_articulation
                )
            # If not, check the parent prim
            elif (
                parent_prim is not None and parent_prim.IsValid() and parent_prim.HasAPI(UsdPhysics.ArticulationRootAPI)
            ):
                if verbose:
                    print(f"Extracting articulation custom attributes from parent {parent_prim.GetPath()}")
                articulation_custom_attrs = usd.get_custom_attribute_values(
                    parent_prim, builder_custom_attr_articulation
                )
            if verbose and articulation_custom_attrs:
                print(f"Extracted articulation custom attributes: {articulation_custom_attrs}")
            body_ids = {}
            body_keys = []
            current_body_id = 0
            art_bodies = []
            if verbose:
                print(f"Bodies under articulation {path!s}:")
            for p in desc.articulatedBodies:
                if verbose:
                    print(f"\t{p!s}")
                if p == Sdf.Path.emptyPath:
                    continue
                key = str(p)
                if key in ignored_body_paths:
                    continue

                usd_prim = stage.GetPrimAtPath(p)
                if collect_schema_attrs:
                    # Collect on each articulated body prim encountered
                    R.collect_prim_attrs(usd_prim)

                if key in body_specs:
                    body_desc = body_specs[key]
                    desc_xform = wp.transform(body_desc.position, usd.from_gfquat(body_desc.rotation))
                    body_world = usd.get_transform(usd_prim, local=False, xform_cache=xform_cache)
                    desired_world = incoming_world_xform * body_world
                    body_incoming_xform = desired_world * wp.transform_inverse(desc_xform)
                    if bodies_follow_joint_ordering:
                        # we just parse the body information without yet adding it to the builder
                        body_data[current_body_id] = parse_body(
                            body_desc,
                            stage.GetPrimAtPath(p),
                            incoming_xform=body_incoming_xform,
                            add_body_to_builder=False,
                        )
                    else:
                        # look up description and add body to builder
                        bid: int = parse_body(  # pyright: ignore[reportAssignmentType]
                            body_desc,
                            stage.GetPrimAtPath(p),
                            incoming_xform=body_incoming_xform,
                            add_body_to_builder=True,
                        )
                        if bid >= 0:
                            art_bodies.append(bid)
                    # remove body spec once we inserted it
                    del body_specs[key]

                body_ids[key] = current_body_id
                body_keys.append(key)
                current_body_id += 1

            if len(body_ids) == 0:
                # no bodies under the articulation or we ignored all of them
                continue

            # determine the joint graph for this articulation
            joint_names: list[str] = []
            joint_edges: list[tuple[int, int]] = []
            # keys of joints that are excluded from the articulation (loop joints)
            joint_excluded: set[str] = set()
            for p in desc.articulatedJoints:
                joint_key = str(p)
                joint_desc = joint_descriptions[joint_key]
                #! it may be possible that a joint is filtered out in the middle of
                #! a chain of joints, which results in a disconnected graph
                #! we should raise an error in this case
                if any(re.match(p, joint_key) for p in ignore_paths):
                    continue
                if str(joint_desc.body0) in ignored_body_paths:
                    continue
                if str(joint_desc.body1) in ignored_body_paths:
                    continue
                parent_id, child_id = resolve_joint_parent_child(joint_desc, body_ids, get_transforms=False)  # pyright: ignore[reportAssignmentType]
                if joint_desc.excludeFromArticulation:
                    joint_excluded.add(joint_key)
                else:
                    joint_edges.append((parent_id, child_id))
                    joint_names.append(joint_key)

            articulation_joint_indices = []

            if len(joint_edges) == 0:
                # We have an articulation without joints, i.e. only free rigid bodies
                if bodies_follow_joint_ordering:
                    for i in body_ids.values():
                        child_body_id = add_body(**body_data[i])
                        joint_id = builder.add_joint_free(child=child_body_id)
                        # note the free joint's coordinates will be initialized by the body_q of the
                        # child body
                        builder.add_articulation(
                            [joint_id], key=body_data[i]["key"], custom_attributes=articulation_custom_attrs
                        )
                else:
                    for i, child_body_id in enumerate(art_bodies):
                        joint_id = builder.add_joint_free(child=child_body_id)
                        # note the free joint's coordinates will be initialized by the body_q of the
                        # child body
                        builder.add_articulation(
                            [joint_id], key=body_keys[i], custom_attributes=articulation_custom_attrs
                        )
                sorted_joints = []
            else:
                # we have an articulation with joints, we need to sort them topologically
                if joint_ordering is not None:
                    if verbose:
                        print(f"Sorting joints using {joint_ordering} ordering...")
                    sorted_joints, reversed_joint_list = topological_sort_undirected(
                        joint_edges, use_dfs=joint_ordering == "dfs", ensure_single_root=True
                    )
                    if reversed_joint_list:
                        reversed_joint_paths = [joint_names[joint_id] for joint_id in reversed_joint_list]
                        reversed_joint_names = ", ".join(reversed_joint_paths)
                        raise ValueError(
                            f"Reversed joints are not supported: {reversed_joint_names}. Ensure that the joint parent body is defined as physics:body0 and the child is defined as physics:body1 in the joint prim."
                        )
                    if verbose:
                        print("Joint ordering:", sorted_joints)
                else:
                    # we keep the original order of the joints
                    sorted_joints = np.arange(len(joint_names))

            if len(sorted_joints) > 0:
                # insert the bodies in the order of the joints
                if bodies_follow_joint_ordering:
                    inserted_bodies = set()
                    for jid in sorted_joints:
                        parent, child = joint_edges[jid]
                        if parent >= 0 and parent not in inserted_bodies:
                            b = add_body(**body_data[parent])
                            inserted_bodies.add(parent)
                            art_bodies.append(b)
                            path_body_map[body_data[parent]["key"]] = b
                        if child >= 0 and child not in inserted_bodies:
                            b = add_body(**body_data[child])
                            inserted_bodies.add(child)
                            art_bodies.append(b)
                            path_body_map[body_data[child]["key"]] = b

                first_joint_parent = joint_edges[sorted_joints[0]][0]
                if first_joint_parent != -1:
                    # the mechanism is floating since there is no joint connecting it to the world
                    # we explicitly add a free joint connecting the first body in the articulation to the world
                    # to make sure generalized-coordinate solvers can simulate it
                    if bodies_follow_joint_ordering:
                        child_body = body_data[first_joint_parent]
                        child_body_id = path_body_map[child_body["key"]]
                    else:
                        child_body_id = art_bodies[first_joint_parent]
                    # apply the articulation transform to the body
                    free_joint_id = builder.add_joint_free(child=child_body_id)
                    articulation_joint_indices.append(free_joint_id)

                # insert the remaining joints in topological order
                for joint_id, i in enumerate(sorted_joints):
                    if joint_id == 0 and first_joint_parent == -1:
                        # the articulation root joint receives the articulation transform as parent transform
                        # except if we already inserted a floating-base joint
                        joint = parse_joint(
                            joint_descriptions[joint_names[i]],
                            incoming_xform=articulation_incoming_xform,
                        )
                    else:
                        joint = parse_joint(
                            joint_descriptions[joint_names[i]],
                        )
                    if joint is not None:
                        articulation_joint_indices.append(joint)

                # insert loop joints
                for joint_key in joint_excluded:
                    joint = parse_joint(
                        joint_descriptions[joint_key],
                        incoming_xform=articulation_incoming_xform,
                    )

            # Create the articulation from all collected joints
            if articulation_joint_indices:
                builder.add_articulation(
                    articulation_joint_indices,
                    key=articulation_path,
                    custom_attributes=articulation_custom_attrs,
                )

            articulation_bodies[articulation_id] = art_bodies
            # determine if self-collisions are enabled
            articulation_has_self_collision[articulation_id] = usd.get_attribute(
                articulation_prim,
                "physxArticulation:enabledSelfCollisions",
                default=enable_self_collisions,
            )
            articulation_id += 1
    no_articulations = UsdPhysics.ObjectType.Articulation not in ret_dict
    has_joints = any(
        (
            not (only_load_enabled_joints and not joint_desc.jointEnabled)
            and not any(re.match(p, joint_key) for p in ignore_paths)
            and str(joint_desc.body0) not in ignored_body_paths
            and str(joint_desc.body1) not in ignored_body_paths
        )
        for joint_key, joint_desc in joint_descriptions.items()
    )

    # insert remaining bodies that were not part of any articulation so far
    for path, rigid_body_desc in body_specs.items():
        key = str(path)
        body_id: int = parse_body(  # pyright: ignore[reportAssignmentType]
            rigid_body_desc,
            stage.GetPrimAtPath(path),
            incoming_xform=incoming_world_xform,
            add_body_to_builder=True,
        )
        if not (no_articulations and has_joints):
            # add articulation and free joint for this body
            joint_id = builder.add_joint_free(child=body_id)
            builder.add_articulation([joint_id], key=key)

    if no_articulations and has_joints:
        # parse external joints that are not part of any articulation
        orphan_joints = []
        for joint_key, joint_desc in joint_descriptions.items():
            if any(re.match(p, joint_key) for p in ignore_paths):
                continue
            if str(joint_desc.body0) in ignored_body_paths or str(joint_desc.body1) in ignored_body_paths:
                continue
            try:
                parse_joint(joint_desc, incoming_xform=incoming_world_xform)
                orphan_joints.append(joint_key)
            except ValueError as exc:
                if verbose:
                    print(f"Skipping joint {joint_key}: {exc}")

        if len(orphan_joints) > 0:
            warn_str = (
                f"No articulation was found but {len(orphan_joints)} joints were parsed: [{', '.join(orphan_joints)}]. "
            )
            warn_str += (
                "Make sure your USD asset includes an articulation root prim with the PhysicsArticulationRootAPI.\n"
            )
            warn_str += "If you want to proceed with these orphan joints, make sure to call ModelBuilder.finalize(skip_validation_joints=True) "
            warn_str += "to avoid raising a ValueError. Note that not all solvers will support such a configuration."
            warnings.warn(warn_str, stacklevel=2)

    # parse shapes attached to the rigid bodies
    path_collision_filters = set()
    no_collision_shapes = set()
    collision_group_ids = {}
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
                if path in path_shape_map:
                    if verbose:
                        print(f"Shape at {path} already added, skipping.")
                    continue
                body_path = str(shape_spec.rigidBody)
                if verbose:
                    print(f"collision shape {prim.GetPath()} ({prim.GetTypeName()}), body = {body_path}")
                body_id = path_body_map.get(body_path, -1)
                scale = usd.get_scale(prim, local=False)
                collision_group = builder.default_shape_cfg.collision_group

                if len(shape_spec.collisionGroups) > 0:
                    cgroup_name = str(shape_spec.collisionGroups[0])
                    if cgroup_name not in collision_group_ids:
                        # Start from 1 to avoid collision_group = 0 (which means "no collisions")
                        collision_group_ids[cgroup_name] = len(collision_group_ids) + 1
                    collision_group = collision_group_ids[cgroup_name]
                material = material_specs[""]
                if len(shape_spec.materials) >= 1:
                    if len(shape_spec.materials) > 1 and verbose:
                        print(f"Warning: More than one material found on shape at '{path}'.\nUsing only the first one.")
                    material = material_specs[str(shape_spec.materials[0])]
                    if verbose:
                        print(
                            f"\tMaterial of '{path}':\tfriction: {material.dynamicFriction},\ttorsional friction: {material.torsionalFriction},\trolling friction: {material.rollingFriction},\trestitution: {material.restitution},\tdensity: {material.density}"
                        )
                elif verbose:
                    print(f"No material found for shape at '{path}'.")
                prim_and_scene = (prim, physics_scene_prim)
                local_xform = wp.transform(shape_spec.localPos, usd.from_gfquat(shape_spec.localRot))
                if body_id == -1:
                    shape_xform = incoming_world_xform * local_xform
                else:
                    shape_xform = local_xform
                # Extract custom attributes for this shape
                shape_custom_attrs = usd.get_custom_attribute_values(prim, builder_custom_attr_shape)
                if collect_schema_attrs:
                    R.collect_prim_attrs(prim)

                contact_margin = R.get_value(prim, prim_type=PrimType.SHAPE, key="contact_margin", verbose=verbose)
                if contact_margin == float("-inf"):
                    contact_margin = builder.default_shape_cfg.contact_margin

                shape_params = {
                    "body": body_id,
                    "xform": shape_xform,
                    "cfg": ModelBuilder.ShapeConfig(
                        ke=usd.get_float_with_fallback(
                            prim_and_scene, "newton:contact_ke", builder.default_shape_cfg.ke
                        ),
                        kd=usd.get_float_with_fallback(
                            prim_and_scene, "newton:contact_kd", builder.default_shape_cfg.kd
                        ),
                        kf=usd.get_float_with_fallback(
                            prim_and_scene, "newton:contact_kf", builder.default_shape_cfg.kf
                        ),
                        ka=usd.get_float_with_fallback(
                            prim_and_scene, "newton:contact_ka", builder.default_shape_cfg.ka
                        ),
                        thickness=usd.get_float_with_fallback(
                            prim_and_scene, "newton:contact_thickness", builder.default_shape_cfg.thickness
                        ),
                        contact_margin=contact_margin,
                        mu=material.dynamicFriction,
                        restitution=material.restitution,
                        torsional_friction=material.torsionalFriction,
                        rolling_friction=material.rollingFriction,
                        density=body_density.get(body_path, default_shape_density),
                        collision_group=collision_group,
                        is_visible=not hide_collision_shapes,
                    ),
                    "key": path,
                    "custom_attributes": shape_custom_attrs,
                }
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
                    if not (scale[0] == scale[1] == scale[2]):
                        print("Warning: Non-uniform scaling of spheres is not supported.")
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
                    mesh = usd.get_mesh(prim)
                    mesh.maxhullvert = R.get_value(
                        prim,
                        prim_type=PrimType.SHAPE,
                        key="max_hull_vertices",
                        default=mesh_maxhullvert,
                        verbose=verbose,
                    )
                    shape_id = builder.add_shape_mesh(
                        scale=wp.vec3(*shape_spec.meshScale),
                        mesh=mesh,
                        **shape_params,
                    )
                    if not skip_mesh_approximation:
                        approximation = usd.get_attribute(prim, "physics:approximation", None)
                        if approximation is not None:
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

                if prim.HasRelationship("physics:filteredPairs"):
                    other_paths = prim.GetRelationship("physics:filteredPairs").GetTargets()
                    for other_path in other_paths:
                        path_collision_filters.add((path, str(other_path)))

                if not _is_enabled_collider(prim):
                    no_collision_shapes.add(shape_id)
                    builder.shape_flags[shape_id] &= ~ShapeFlags.COLLIDE_SHAPES

    # approximate meshes
    for remeshing_method, shape_ids in remeshing_queue.items():
        builder.approximate_meshes(method=remeshing_method, shape_indices=shape_ids)

    # apply collision filters now that we have added all shapes
    for path1, path2 in path_collision_filters:
        shape1 = path_shape_map[path1]
        shape2 = path_shape_map[path2]
        builder.add_shape_collision_filter_pair(shape1, shape2)

    # apply collision filters to all shapes that have no collision
    for shape_id in no_collision_shapes:
        for other_shape_id in range(builder.shape_count):
            if other_shape_id != shape_id:
                builder.add_shape_collision_filter_pair(shape_id, other_shape_id)

    # apply collision filters from articulations that have self collisions disabled
    for art_id, bodies in articulation_bodies.items():
        if not articulation_has_self_collision[art_id]:
            for body1, body2 in itertools.combinations(bodies, 2):
                for shape1 in builder.body_shapes[body1]:
                    for shape2 in builder.body_shapes[body2]:
                        builder.add_shape_collision_filter_pair(shape1, shape2)

    # overwrite inertial properties of bodies that have PhysicsMassAPI schema applied
    if UsdPhysics.ObjectType.RigidBody in ret_dict:
        paths, rigid_body_descs = ret_dict[UsdPhysics.ObjectType.RigidBody]
        for path, _rigid_body_desc in zip(paths, rigid_body_descs, strict=False):
            prim = stage.GetPrimAtPath(path)
            if not prim.HasAPI(UsdPhysics.MassAPI):
                continue
            body_path = str(path)
            body_id = path_body_map.get(body_path, -1)
            if body_id == -1:
                continue
            mass = usd.get_float(prim, "physics:mass")
            if mass is not None:
                builder.body_mass[body_id] = mass
                builder.body_inv_mass[body_id] = 1.0 / mass
            com = usd.get_vector(prim, "physics:centerOfMass")
            if com is not None:
                builder.body_com[body_id] = wp.vec3(*com)
            i_diag = usd.get_vector(prim, "physics:diagonalInertia", np.zeros(3, dtype=np.float32))
            i_rot = usd.get_quat(prim, "physics:principalAxes", wp.quat_identity())
            if np.linalg.norm(i_diag) > 0.0:
                rot = np.array(wp.quat_to_matrix(i_rot), dtype=np.float32).reshape(3, 3)
                inertia = rot @ np.diag(i_diag) @ rot.T
                builder.body_inertia[body_id] = wp.mat33(inertia)
                if inertia.any():
                    builder.body_inv_inertia[body_id] = wp.inverse(wp.mat33(*inertia))
                else:
                    builder.body_inv_inertia[body_id] = wp.mat33(0.0)

            # Assign nonzero inertia if mass is nonzero to make sure the body can be simulated
            I_m = np.array(builder.body_inertia[body_id])
            mass = builder.body_mass[body_id]
            if I_m.max() == 0.0:
                if mass > 0.0:
                    # Heuristic: assume a uniform density sphere with the given mass
                    # For a sphere: I = (2/5) * m * r^2
                    # Estimate radius from mass assuming reasonable density (e.g., water density ~1000 kg/m³)
                    # This gives r = (3*m/(4*π*p))^(1/3)
                    density = default_shape_density  # kg/m³
                    volume = mass / density
                    radius = (3.0 * volume / (4.0 * np.pi)) ** (1.0 / 3.0)
                    _, _, I_default = compute_sphere_inertia(density, radius)

                    # Apply parallel axis theorem if center of mass is offset
                    com = builder.body_com[body_id]
                    if np.linalg.norm(com) > 1e-6:
                        # I = I_cm + m * d² where d is distance from COM to body origin
                        d_squared = np.sum(com**2)
                        I_default += wp.mat33(mass * d_squared * np.eye(3, dtype=np.float32))

                    builder.body_inertia[body_id] = I_default
                    builder.body_inv_inertia[body_id] = wp.inverse(I_default)

                    if verbose:
                        print(
                            f"Applied default inertia matrix for body {body_path}: diagonal elements = [{I_default[0, 0]}, {I_default[1, 1]}, {I_default[2, 2]}]"
                        )
                else:
                    warnings.warn(
                        f"Body {body_path} has zero mass and zero inertia despite having the MassAPI USD schema applied.",
                        stacklevel=2,
                    )

    # add free joints to floating bodies that's just been added by import_usd
    if not (no_articulations and has_joints):
        new_bodies = path_body_map.values()
        builder.add_free_joints_to_floating_bodies(new_bodies)

    # collapsing fixed joints to reduce the number of simulated bodies connected by fixed joints.
    collapse_results = None
    path_body_relative_transform = {}
    if scene_attributes.get("newton:collapse_fixed_joints", collapse_fixed_joints):
        collapse_results = builder.collapse_fixed_joints()
        body_merged_parent = collapse_results["body_merged_parent"]
        body_merged_transform = collapse_results["body_merged_transform"]
        body_remap = collapse_results["body_remap"]
        # remap body ids in articulation bodies
        for art_id, bodies in articulation_bodies.items():
            articulation_bodies[art_id] = [body_remap[b] for b in bodies if b in body_remap]

        for path, body_id in path_body_map.items():
            if body_id in body_remap:
                new_id = body_remap[body_id]
            elif body_id in body_merged_parent:
                # this body has been merged with another body
                new_id = body_remap[body_merged_parent[body_id]]
                path_body_relative_transform[path] = body_merged_transform[body_id]
            else:
                # this body has not been merged
                new_id = body_id

            path_body_map[path] = new_id

        # Joint indices may have shifted after collapsing fixed joints; refresh the joint path map accordingly.
        path_joint_map = {key: idx for idx, key in enumerate(builder.joint_key)}

    return {
        "fps": stage.GetFramesPerSecond(),
        "duration": stage.GetEndTimeCode() - stage.GetStartTimeCode(),
        "up_axis": stage_up_axis,
        "path_body_map": path_body_map,
        "path_joint_map": path_joint_map,
        "path_shape_map": path_shape_map,
        "path_shape_scale": path_shape_scale,
        "mass_unit": mass_unit,
        "linear_unit": linear_unit,
        "scene_attributes": scene_attributes,
        "physics_dt": physics_dt,
        "collapse_results": collapse_results,
        "schema_attrs": R.schema_attrs,
        # "articulation_roots": articulation_roots,
        # "articulation_bodies": articulation_bodies,
        "path_body_relative_transform": path_body_relative_transform,
        "max_solver_iterations": max_solver_iters,
    }


def resolve_usd_from_url(url: str, target_folder_name: str | None = None, export_usda: bool = False) -> str:
    """Download a USD file from a URL and resolves all references to other USD files to be downloaded to the given target folder.

    Args:
        url: URL to the USD file.
        target_folder_name: Target folder name. If ``None``, a time-stamped
          folder will be created in the current directory.
        export_usda: If ``True``, converts each downloaded USD file to USDA and
          saves the additional USDA file in the target folder with the same
          base name as the original USD file.

    Returns:
        File path to the downloaded USD file.
    """

    import requests  # noqa: PLC0415

    try:
        from pxr import Usd  # noqa: PLC0415
    except ImportError as e:
        raise ImportError("Failed to import pxr. Please install USD (e.g. via `pip install usd-core`).") from e

    response = requests.get(url, allow_redirects=True)
    if response.status_code != 200:
        raise RuntimeError(f"Failed to download USD file. Status code: {response.status_code}")
    file = response.content
    dot = os.path.extsep
    base = os.path.basename(url)
    url_folder = os.path.dirname(url)
    base_name = dot.join(base.split(dot)[:-1])
    if target_folder_name is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        target_folder_name = os.path.join(".usd_cache", f"{base_name}_{timestamp}")
    os.makedirs(target_folder_name, exist_ok=True)
    target_filename = os.path.join(target_folder_name, base)
    with open(target_filename, "wb") as f:
        f.write(file)

    stage = Usd.Stage.Open(target_filename, Usd.Stage.LoadNone)
    stage_str = stage.GetRootLayer().ExportToString()
    print(f"Downloaded USD file to {target_filename}.")
    if export_usda:
        usda_filename = os.path.join(target_folder_name, base_name + ".usda")
        with open(usda_filename, "w") as f:
            f.write(stage_str)
            print(f"Exported USDA file to {usda_filename}.")

    # parse referenced USD files like `references = @./franka_collisions.usd@`
    downloaded = set()
    for match in re.finditer(r"references.=.@(.*?)@", stage_str):
        refname = match.group(1)
        if refname.startswith("./"):
            refname = refname[2:]
        if refname in downloaded:
            continue
        try:
            response = requests.get(f"{url_folder}/{refname}", allow_redirects=True)
            if response.status_code != 200:
                print(f"Failed to download reference {refname}. Status code: {response.status_code}")
                continue
            file = response.content
            refdir = os.path.dirname(refname)
            if refdir:
                os.makedirs(os.path.join(target_folder_name, refdir), exist_ok=True)
            ref_filename = os.path.join(target_folder_name, refname)
            if not os.path.exists(ref_filename):
                with open(ref_filename, "wb") as f:
                    f.write(file)
            downloaded.add(refname)
            print(f"Downloaded USD reference {refname} to {ref_filename}.")
            if export_usda:
                ref_stage = Usd.Stage.Open(ref_filename, Usd.Stage.LoadNone)
                ref_stage_str = ref_stage.GetRootLayer().ExportToString()
                base = os.path.basename(ref_filename)
                base_name = dot.join(base.split(dot)[:-1])
                usda_filename = os.path.join(target_folder_name, base_name + ".usda")
                with open(usda_filename, "w") as f:
                    f.write(ref_stage_str)
                    print(f"Exported USDA file to {usda_filename}.")
        except Exception:
            print(f"Failed to download {refname}.")
    return target_filename


def _raise_on_stage_errors(usd_stage, stage_source: str):
    get_errors = getattr(usd_stage, "GetCompositionErrors", None)
    if get_errors is None:
        return
    errors = get_errors()
    if not errors:
        return
    messages = []
    for err in errors:
        try:
            messages.append(err.GetMessage())
        except Exception:
            messages.append(str(err))
    formatted = "\n".join(f"- {message}" for message in messages)
    raise RuntimeError(f"USD stage has composition errors while loading {stage_source}:\n{formatted}")
