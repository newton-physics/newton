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
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import warp as wp

from ..core import quat_between_axes
from ..core.types import Axis, Transform
from ..geometry import MESH_MAXHULLVERT, Mesh, ShapeFlags, compute_sphere_inertia
from ..sim.builder import ModelBuilder
from ..sim.joints import JointMode


def parse_usd(
    builder: ModelBuilder,
    source,
    xform: Transform | None = None,
    only_load_enabled_rigid_bodies: bool = False,
    only_load_enabled_joints: bool = True,
    joint_drive_gains_scaling: float = 1.0,
    invert_rotations: bool = True,
    verbose: bool = False,
    ignore_paths: list[str] | None = None,
    cloned_world: str | None = None,
    collapse_fixed_joints: bool = False,
    enable_self_collisions: bool = True,
    apply_up_axis_from_stage: bool = False,
    root_path: str = "/",
    joint_ordering: Literal["bfs", "dfs"] | None = "dfs",
    bodies_follow_joint_ordering: bool = True,
    skip_mesh_approximation: bool = False,
    load_non_physics_prims: bool = True,
    hide_collision_shapes: bool = False,
    mesh_maxhullvert: int = MESH_MAXHULLVERT,
) -> dict[str, Any]:
    """
    Parses a Universal Scene Description (USD) stage containing UsdPhysics schema definitions for rigid-body articulations and adds the bodies, shapes and joints to the given ModelBuilder.

    The USD description has to be either a path (file name or URL), or an existing USD stage instance that implements the `Stage <https://openusd.org/dev/api/class_usd_stage.html>`_ interface.

    Args:
        builder (ModelBuilder): The :class:`ModelBuilder` to add the bodies and joints to.
        source (str | pxr.Usd.Stage): The file path to the USD file, or an existing USD stage instance.
        xform (Transform): The transform to apply to the entire scene.
        only_load_enabled_rigid_bodies (bool): If True, only rigid bodies which do not have `physics:rigidBodyEnabled` set to False are loaded.
        only_load_enabled_joints (bool): If True, only joints which do not have `physics:jointEnabled` set to False are loaded.
        joint_drive_gains_scaling (float): The default scaling of the PD control gains (stiffness and damping), if not set in the PhysicsScene with as "newton:joint_drive_gains_scaling".
        invert_rotations (bool): If True, inverts any rotations defined in the shape transforms.
        verbose (bool): If True, print additional information about the parsed USD file. Default is False.
        ignore_paths (List[str]): A list of regular expressions matching prim paths to ignore.
        cloned_world (str): The prim path of a world which is cloned within this USD file. Siblings of this world prim will not be parsed but instead be replicated via `ModelBuilder.add_builder(builder, xform)` to speed up the loading of many instantiated worlds.
        collapse_fixed_joints (bool): If True, fixed joints are removed and the respective bodies are merged. Only considered if not set on the PhysicsScene as "newton:collapse_fixed_joints".
        enable_self_collisions (bool): Determines the default behavior of whether self-collisions are enabled for all shapes within an articulation. If an articulation has the attribute ``physxArticulation:enabledSelfCollisions`` defined, this attribute takes precedence.
        apply_up_axis_from_stage (bool): If True, the up axis of the stage will be used to set :attr:`newton.ModelBuilder.up_axis`. Otherwise, the stage will be rotated such that its up axis aligns with the builder's up axis. Default is False.
        root_path (str): The USD path to import, defaults to "/".
        joint_ordering (str): The ordering of the joints in the simulation. Can be either "bfs" or "dfs" for breadth-first or depth-first search, or ``None`` to keep joints in the order in which they appear in the USD. Default is "dfs".
        bodies_follow_joint_ordering (bool): If True, the bodies are added to the builder in the same order as the joints (parent then child body). Otherwise, bodies are added in the order they appear in the USD. Default is True.
        skip_mesh_approximation (bool): If True, mesh approximation is skipped. Otherwise, meshes are approximated according to the ``physics:approximation`` attribute defined on the UsdPhysicsMeshCollisionAPI (if it is defined). Default is False.
        load_non_physics_prims (bool): If True, prims that are children of a rigid body that do not have a UsdPhysics schema applied are loaded as visual shapes in a separate pass (may slow down the loading process). Otherwise, non-physics prims are ignored. Default is True.
        hide_collision_shapes (bool): If True, collision shapes are hidden. Default is False.
        mesh_maxhullvert (int): Maximum vertices for convex hull approximation of meshes.

    Returns:
        dict: Dictionary with the following entries:

        .. list-table::
            :widths: 25 75

            * - "fps"
              - USD stage frames per second
            * - "duration"
              - Difference between end time code and start time code of the USD stage
            * - "up_axis"
              - :class:`Axis` representing the stage's up axis ("X", "Y", or "Z")
            * - "path_shape_map"
              - Mapping from prim path (str) of the UsdGeom to the respective shape index in :class:`ModelBuilder`
            * - "path_body_map"
              - Mapping from prim path (str) of a rigid body prim (e.g. that implements the PhysicsRigidBodyAPI) to the respective body index in :class:`ModelBuilder`
            * - "path_shape_scale"
              - Mapping from prim path (str) of the UsdGeom to its respective 3D world scale
            * - "mass_unit"
              - The stage's Kilograms Per Unit (KGPU) definition (1.0 by default)
            * - "linear_unit"
              - The stage's Meters Per Unit (MPU) definition (1.0 by default)
            * - "scene_attributes"
              - Dictionary of all attributes applied to the PhysicsScene prim
            * - "collapse_results"
              - Dictionary returned by :meth:`newton.ModelBuilder.collapse_fixed_joints` if `collapse_fixed_joints` is True, otherwise None.
    """
    try:
        from pxr import Sdf, Usd, UsdGeom, UsdPhysics  # noqa: PLC0415
    except ImportError as e:
        raise ImportError("Failed to import pxr. Please install USD (e.g. via `pip install usd-core`).") from e

    from .topology import topological_sort  # noqa: PLC0415 (circular import)

    @dataclass
    class PhysicsMaterial:
        staticFriction: float = builder.default_shape_cfg.mu
        dynamicFriction: float = builder.default_shape_cfg.mu
        restitution: float = builder.default_shape_cfg.restitution
        density: float = builder.default_shape_cfg.density

    # load joint defaults
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

    def get_attribute(prim, name):
        return prim.GetAttribute(name)

    def has_attribute(prim, name):
        attr = get_attribute(prim, name)
        return attr.IsValid() and attr.HasAuthoredValue()

    def parse_float(prim, name, default=None):
        attr = get_attribute(prim, name)
        if not attr or not attr.HasAuthoredValue():
            return default
        val = attr.Get()
        if np.isfinite(val):
            return val
        return default

    def parse_float_with_fallback(prims: Iterable[Usd.Prim], name: str, default: float = 0.0) -> float:
        ret = default
        for prim in prims:
            if not prim:
                continue
            attr = get_attribute(prim, name)
            if not attr or not attr.HasAuthoredValue():
                continue
            val = attr.Get()
            if np.isfinite(val):
                ret = val
                break
        return ret

    def from_gfquat(gfquat):
        return wp.normalize(wp.quat(*gfquat.imaginary, gfquat.real))

    def parse_quat(prim, name, default=None):
        attr = get_attribute(prim, name)
        if not attr or not attr.HasAuthoredValue():
            return default
        val = attr.Get()
        quat = from_gfquat(val)
        l = wp.length(quat)
        if np.isfinite(l) and l > 0.0:
            return quat
        return default

    def parse_vec(prim, name, default=None):
        attr = get_attribute(prim, name)
        if not attr or not attr.HasAuthoredValue():
            return default
        val = attr.Get()
        if np.isfinite(val).all():
            return np.array(val, dtype=np.float32)
        return default

    def parse_generic(prim, name, default=None):
        attr = get_attribute(prim, name)
        if not attr or not attr.HasAuthoredValue():
            return default
        return attr.Get()

    def parse_xform(prim):
        xform = UsdGeom.Xform(prim)
        mat = np.array(xform.GetLocalTransformation(), dtype=np.float32)
        if invert_rotations:
            rot = wp.quat_from_matrix(wp.mat33(mat[:3, :3].T.flatten()))
        else:
            rot = wp.quat_from_matrix(wp.mat33(mat[:3, :3].flatten()))
        pos = mat[3, :3]
        return wp.transform(pos, rot)

    if ignore_paths is None:
        ignore_paths = []

    usd_axis_to_axis = {
        UsdPhysics.Axis.X: Axis.X,
        UsdPhysics.Axis.Y: Axis.Y,
        UsdPhysics.Axis.Z: Axis.Z,
    }

    if isinstance(source, str):
        stage = Usd.Stage.Open(source, Usd.Stage.LoadAll)
    else:
        stage = source

    DegreesToRadian = np.pi / 180
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

    # resolve cloned worlds
    if cloned_world is not None:
        cloned_world_prim = stage.GetPrimAtPath(cloned_world)
        if not cloned_world_prim:
            raise RuntimeError(f"Failed to resolve cloned world {cloned_world}")
        cloned_world_xforms = []
        cloned_world_paths = []
        # get paths of the siblings of the cloned world
        # and ignore them during parsing, later we use
        # ModelBuilder.add_builder() to instantiate these
        # worlds at their respective Xform transforms
        worlds_prim = cloned_world_prim.GetParent()
        for sibling in worlds_prim.GetChildren():
            # print(sibling.GetPath(), parse_xform(sibling))
            p = str(sibling.GetPath())
            cloned_world_xforms.append(parse_xform(sibling))
            cloned_world_paths.append(p)
            if sibling != cloned_world_prim:
                ignore_paths.append(p)

        # set xform of the cloned world (e.g. "world0") to identity
        # and later apply this xform via ModelBuilder.add_builder()
        # to instantiate the world at the correct location
        UsdGeom.Xform(cloned_world_prim).SetXformOpOrder([])

        # create a new builder for the cloned world, then instantiate
        # it back in the original builder
        multi_world_builder = builder
        builder = ModelBuilder()

    non_regex_ignore_paths = [path for path in ignore_paths if ".*" not in path]
    ret_dict = UsdPhysics.LoadUsdPhysicsFromRange(stage, [root_path], excludePaths=non_regex_ignore_paths)

    # for key, value in ret_dict.items():
    #     print(f"Object type: {key}")
    #     prims, scene_descs = value
    #     for prim, desc in zip(prims, scene_descs):
    #         print(prim)
    #         print(desc)
    #     if key == UsdPhysics.ObjectType.CapsuleShape:
    #         for prim, desc in zip(prims, scene_descs):
    #             print(desc.halfHeight)
    #     if key == UsdPhysics.ObjectType.RigidBody:
    #         for prim, desc in zip(prims, scene_descs):
    #             print(desc.simulationOwners)
    # print("***************************************************************************")

    # mapping from prim path to body ID in Warp sim
    path_body_map = {}
    # mapping from prim path to shape ID in Warp sim
    path_shape_map = {}
    path_shape_scale = {}

    physics_scene_prim = None

    visual_shape_cfg = ModelBuilder.ShapeConfig(
        density=0.0,
        has_shape_collision=False,
        has_particle_collision=False,
    )

    def load_visual_shapes(parent_body_id, prim, incoming_xform):
        if (
            prim.HasAPI(UsdPhysics.RigidBodyAPI)
            or prim.HasAPI(UsdPhysics.MassAPI)
            or prim.HasAPI(UsdPhysics.CollisionAPI)
            or prim.HasAPI(UsdPhysics.MeshCollisionAPI)
        ):
            return
        path_name = str(prim.GetPath())
        if any(re.match(path, path_name) for path in ignore_paths):
            return
        xform = incoming_xform * parse_xform(prim)
        if prim.IsInstance():
            proto = prim.GetPrototype()
            for child in proto.GetChildren():
                # remap prototype child path to this instance's path (instance proxy)
                inst_path = child.GetPath().ReplacePrefix(proto.GetPath(), prim.GetPath())
                inst_child = stage.GetPrimAtPath(inst_path)
                load_visual_shapes(parent_body_id, inst_child, xform)
            return
        type_name = str(prim.GetTypeName()).lower()
        if type_name.endswith("joint"):
            return
        scale = parse_scale(prim)
        shape_id = -1
        if path_name not in path_shape_map:
            if type_name == "cube":
                size = parse_float(prim, "size", 2.0)
                if has_attribute(prim, "extents"):
                    extents = parse_vec(prim, "extents") * scale
                    # TODO position geom at extents center?
                    # geo_pos = 0.5 * (extents[0] + extents[1])
                    extents = extents[1] - extents[0]
                else:
                    extents = scale * size
                shape_id = builder.add_shape_box(
                    parent_body_id,
                    xform,
                    hx=extents[0] / 2,
                    hy=extents[1] / 2,
                    hz=extents[2] / 2,
                    cfg=visual_shape_cfg,
                    key=path_name,
                )
            elif type_name == "sphere":
                if not (scale[0] == scale[1] == scale[2]):
                    print("Warning: Non-uniform scaling of spheres is not supported.")
                if has_attribute(prim, "extents"):
                    extents = parse_vec(prim, "extents") * scale
                    # TODO position geom at extents center?
                    # geo_pos = 0.5 * (extents[0] + extents[1])
                    extents = extents[1] - extents[0]
                    if not (extents[0] == extents[1] == extents[2]):
                        print("Warning: Non-uniform extents of spheres are not supported.")
                    radius = extents[0]
                else:
                    radius = parse_float(prim, "radius", 1.0) * scale[0]
                shape_id = builder.add_shape_sphere(
                    parent_body_id,
                    xform,
                    radius,
                    cfg=visual_shape_cfg,
                    key=path_name,
                )
            elif type_name == "plane":
                axis_str = parse_generic(prim, "axis", "Z").upper()
                plane_xform = xform
                if axis_str != "Z":
                    axis_q = quat_between_axes(Axis.Z, axis_str)
                    plane_xform = wp.transform(xform.p, xform.q * axis_q)
                width = parse_float(prim, "width", 0.0) * scale[0]
                length = parse_float(prim, "length", 0.0) * scale[1]
                shape_id = builder.add_shape_plane(
                    body=parent_body_id,
                    xform=plane_xform,
                    width=width,
                    length=length,
                    cfg=visual_shape_cfg,
                    key=path_name,
                )
            elif type_name == "capsule":
                axis_str = parse_generic(prim, "axis", "Z").upper()
                radius = parse_float(prim, "radius", 0.5) * scale[0]
                half_height = parse_float(prim, "height", 2.0) / 2 * scale[1]
                assert not has_attribute(prim, "extents"), "Capsule extents are not supported."
                # Apply axis rotation to transform
                axis_idx = "XYZ".index(axis_str)
                xform = wp.transform(xform.p, xform.q * quat_between_axes(Axis.Z, axis_idx))
                shape_id = builder.add_shape_capsule(
                    parent_body_id,
                    xform,
                    radius,
                    half_height,
                    cfg=visual_shape_cfg,
                    key=path_name,
                )
            elif type_name == "cylinder":
                axis_str = parse_generic(prim, "axis", "Z").upper()
                radius = parse_float(prim, "radius", 0.5) * scale[0]
                half_height = parse_float(prim, "height", 2.0) / 2 * scale[1]
                assert not has_attribute(prim, "extents"), "Cylinder extents are not supported."
                # Apply axis rotation to transform
                axis_idx = "XYZ".index(axis_str)
                xform = wp.transform(xform.p, xform.q * quat_between_axes(Axis.Z, axis_idx))
                shape_id = builder.add_shape_cylinder(
                    parent_body_id,
                    xform,
                    radius,
                    half_height,
                    cfg=visual_shape_cfg,
                    key=path_name,
                )
            elif type_name == "cone":
                axis_str = parse_generic(prim, "axis", "Z").upper()
                radius = parse_float(prim, "radius", 0.5) * scale[0]
                half_height = parse_float(prim, "height", 2.0) / 2 * scale[1]
                assert not has_attribute(prim, "extents"), "Cone extents are not supported."
                # Apply axis rotation to transform
                axis_idx = "XYZ".index(axis_str)
                xform = wp.transform(xform.p, xform.q * quat_between_axes(Axis.Z, axis_idx))
                shape_id = builder.add_shape_cone(
                    parent_body_id,
                    xform,
                    radius,
                    half_height,
                    cfg=visual_shape_cfg,
                    key=path_name,
                )
            elif type_name == "mesh":
                mesh = UsdGeom.Mesh(prim)
                points = np.array(mesh.GetPointsAttr().Get(), dtype=np.float32)
                indices = np.array(mesh.GetFaceVertexIndicesAttr().Get(), dtype=np.float32)
                counts = mesh.GetFaceVertexCountsAttr().Get()
                faces = []
                face_id = 0
                for count in counts:
                    if count == 3:
                        faces.append(indices[face_id : face_id + 3])
                    elif count == 4:
                        faces.append(indices[face_id : face_id + 3])
                        faces.append(indices[[face_id, face_id + 2, face_id + 3]])
                    else:
                        continue
                    face_id += count
                m = Mesh(points, np.array(faces, dtype=np.int32).flatten())
                shape_id = builder.add_shape_mesh(
                    parent_body_id,
                    xform,
                    scale=scale,
                    mesh=m,
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
            load_visual_shapes(parent_body_id, child, xform)

    def add_body(prim, xform, key, armature):
        b = builder.add_body(
            xform=xform,
            key=key,
            armature=armature,
        )
        path_body_map[key] = b
        if load_non_physics_prims:
            for child in prim.GetChildren():
                load_visual_shapes(b, child, wp.transform_identity())
        return b

    def parse_body(rigid_body_desc, prim, incoming_xform=None, add_body_to_builder=True):
        nonlocal path_body_map
        nonlocal physics_scene_prim

        if not rigid_body_desc.rigidBodyEnabled and only_load_enabled_rigid_bodies:
            return -1

        rot = rigid_body_desc.rotation
        origin = wp.transform(rigid_body_desc.position, from_gfquat(rot))
        if incoming_xform is not None:
            origin = wp.mul(incoming_xform, origin)
        path = str(prim.GetPath())

        body_armature = parse_float_with_fallback(
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

    def parse_scale(prim):
        xform = UsdGeom.Xform(prim)
        scale = np.ones(3, dtype=np.float32)
        for op in xform.GetOrderedXformOps():
            if op.GetOpType() == UsdGeom.XformOp.TypeScale:
                scale = np.array(op.Get(), dtype=np.float32)
        return scale

    def resolve_joint_parent_child(joint_desc, body_index_map: dict[str, int], get_transforms: bool = True):
        if get_transforms:
            parent_tf = wp.transform(joint_desc.localPose0Position, from_gfquat(joint_desc.localPose0Orientation))
            child_tf = wp.transform(joint_desc.localPose1Position, from_gfquat(joint_desc.localPose1Orientation))
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
                warnings.warn(f"Skipping joint {joint_desc.primPath}: both bodies unresolved", stacklevel=2)
                return
            parent_id, child_id = child_id, parent_id
            if get_transforms:
                parent_tf, child_tf = child_tf, parent_tf
            if verbose:
                print(f"Joint {joint_desc.primPath} connects {parent_path} to world")
        if get_transforms:
            return parent_id, child_id, parent_tf, child_tf
        else:
            return parent_id, child_id

    def parse_joint(joint_desc, joint_path, incoming_xform=None):
        if not joint_desc.jointEnabled and only_load_enabled_joints:
            return
        key = joint_desc.type
        joint_prim = stage.GetPrimAtPath(joint_desc.primPath)
        parent_id, child_id, parent_tf, child_tf = resolve_joint_parent_child(
            joint_desc, path_body_map, get_transforms=True
        )
        if incoming_xform is not None:
            parent_tf = wp.mul(incoming_xform, parent_tf)

        joint_armature = parse_float(joint_prim, "physxJoint:armature", default_joint_armature)
        joint_params = {
            "parent": parent_id,
            "child": child_id,
            "parent_xform": parent_tf,
            "child_xform": child_tf,
            "key": str(joint_path),
            "enabled": joint_desc.jointEnabled,
        }
        current_joint_limit_ke = parse_float_with_fallback(
            (joint_prim, physics_scene_prim), "newton:joint_limit_ke", default_joint_limit_ke
        )
        current_joint_limit_kd = parse_float_with_fallback(
            (joint_prim, physics_scene_prim), "newton:joint_limit_kd", default_joint_limit_kd
        )

        if key == UsdPhysics.ObjectType.FixedJoint:
            builder.add_joint_fixed(**joint_params)
        elif key == UsdPhysics.ObjectType.RevoluteJoint or key == UsdPhysics.ObjectType.PrismaticJoint:
            joint_params["axis"] = usd_axis_to_axis[joint_desc.axis]
            joint_params["limit_lower"] = joint_desc.limit.lower
            joint_params["limit_upper"] = joint_desc.limit.upper
            joint_params["limit_ke"] = current_joint_limit_ke
            joint_params["limit_kd"] = current_joint_limit_kd
            joint_params["armature"] = joint_armature
            if joint_desc.drive.enabled:
                # XXX take the target which is nonzero to decide between position vs. velocity target...
                if joint_desc.drive.targetVelocity:
                    joint_params["target"] = joint_desc.drive.targetVelocity
                    joint_params["mode"] = JointMode.TARGET_VELOCITY
                else:
                    joint_params["target"] = joint_desc.drive.targetPosition
                    joint_params["mode"] = JointMode.TARGET_POSITION

                joint_params["target_ke"] = joint_desc.drive.stiffness
                joint_params["target_kd"] = joint_desc.drive.damping
                joint_params["effort_limit"] = joint_desc.drive.forceLimit

            dof_type = "linear" if key == UsdPhysics.ObjectType.PrismaticJoint else "angular"
            joint_prim.CreateAttribute(f"physics:tensor:{dof_type}:dofOffset", Sdf.ValueTypeNames.UInt).Set(0)
            joint_prim.CreateAttribute(f"state:{dof_type}:physics:position", Sdf.ValueTypeNames.Float).Set(0)
            joint_prim.CreateAttribute(f"state:{dof_type}:physics:velocity", Sdf.ValueTypeNames.Float).Set(0)

            if key == UsdPhysics.ObjectType.PrismaticJoint:
                builder.add_joint_prismatic(**joint_params)
            else:
                if joint_desc.drive.enabled:
                    joint_params["target"] *= DegreesToRadian
                    joint_params["target_kd"] /= DegreesToRadian / joint_drive_gains_scaling
                    joint_params["target_ke"] /= DegreesToRadian / joint_drive_gains_scaling

                joint_params["limit_lower"] *= DegreesToRadian
                joint_params["limit_upper"] *= DegreesToRadian
                joint_params["limit_ke"] /= DegreesToRadian / joint_drive_gains_scaling
                joint_params["limit_kd"] /= DegreesToRadian / joint_drive_gains_scaling

                builder.add_joint_revolute(**joint_params)
        elif key == UsdPhysics.ObjectType.SphericalJoint:
            builder.add_joint_ball(**joint_params)
        elif key == UsdPhysics.ObjectType.D6Joint:
            linear_axes = []
            angular_axes = []
            num_dofs = 0
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
                    limit_lower = -np.inf
                    limit_upper = np.inf

                free_axis = limit_lower < limit_upper

                def define_joint_mode(dof, joint_desc):
                    target = 0.0  # TODO: parse target from state:*:physics:appliedForce usd attribute when no drive is present
                    mode = JointMode.NONE
                    target_ke = 0.0
                    target_kd = 0.0
                    effort_limit = np.inf
                    for drive in joint_desc.jointDrives:
                        if drive.first != dof:
                            continue
                        if drive.second.enabled:
                            if drive.second.targetVelocity != 0.0:
                                target = drive.second.targetVelocity
                                mode = JointMode.TARGET_VELOCITY
                            else:
                                target = drive.second.targetPosition
                                mode = JointMode.TARGET_POSITION
                            target_ke = drive.second.stiffness
                            target_kd = drive.second.damping
                            effort_limit = drive.second.forceLimit
                    return target, mode, target_ke, target_kd, effort_limit

                target, mode, target_ke, target_kd, effort_limit = define_joint_mode(dof, joint_desc)

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
                    linear_axes.append(
                        ModelBuilder.JointDofConfig(
                            axis=_trans_axes[dof],
                            limit_lower=limit_lower,
                            limit_upper=limit_upper,
                            limit_ke=current_joint_limit_ke,
                            limit_kd=current_joint_limit_kd,
                            target=target,
                            mode=mode,
                            target_ke=target_ke,
                            target_kd=target_kd,
                            armature=joint_armature,
                            effort_limit=effort_limit,
                        )
                    )
                elif free_axis and dof in _rot_axes:
                    angular_axes.append(
                        ModelBuilder.JointDofConfig(
                            axis=_rot_axes[dof],
                            limit_lower=limit_lower * DegreesToRadian,
                            limit_upper=limit_upper * DegreesToRadian,
                            limit_ke=current_joint_limit_ke / DegreesToRadian / joint_drive_gains_scaling,
                            limit_kd=current_joint_limit_kd / DegreesToRadian / joint_drive_gains_scaling,
                            target=target * DegreesToRadian,
                            mode=mode,
                            target_ke=target_ke / DegreesToRadian / joint_drive_gains_scaling,
                            target_kd=target_kd / DegreesToRadian / joint_drive_gains_scaling,
                            armature=joint_armature,
                            effort_limit=effort_limit,
                        )
                    )
                    joint_prim.CreateAttribute(
                        f"physics:tensor:{_rot_names[dof]}:dofOffset", Sdf.ValueTypeNames.UInt
                    ).Set(num_dofs)
                    joint_prim.CreateAttribute(
                        f"state:{_rot_names[dof]}:physics:position", Sdf.ValueTypeNames.Float
                    ).Set(0)
                    joint_prim.CreateAttribute(
                        f"state:{_rot_names[dof]}:physics:velocity", Sdf.ValueTypeNames.Float
                    ).Set(0)
                    num_dofs += 1

            builder.add_joint_d6(**joint_params, linear_axes=linear_axes, angular_axes=angular_axes)
        elif key == UsdPhysics.ObjectType.DistanceJoint:
            if joint_desc.limit.enabled and joint_desc.minEnabled:
                min_dist = joint_desc.limit.lower
            else:
                min_dist = -1.0  # no limit
            if joint_desc.limit.enabled and joint_desc.maxEnabled:
                max_dist = joint_desc.limit.upper
            else:
                max_dist = -1.0
            builder.add_joint_distance(**joint_params, min_distance=min_dist, max_distance=max_dist)
        else:
            raise NotImplementedError(f"Unsupported joint type {key}")

    # Looking for and parsing the attributes on PhysicsScene prims
    scene_attributes = {}
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
        axis = Axis.from_any(int(np.argmax(np.abs(scene_desc.gravityDirection))))

        # Storing Physics Scene attributes
        physics_scene_prim = stage.GetPrimAtPath(path)
        for a in physics_scene_prim.GetAttributes():
            scene_attributes[a.GetName()] = a.Get()

        # Updating joint_drive_gains_scaling if set of the PhysicsScene
        joint_drive_gains_scaling = parse_float(
            physics_scene_prim, "newton:joint_drive_gains_scaling", joint_drive_gains_scaling
        )
    else:
        # builder.up_vector, builder.up_axis = get_up_vector_and_axis(stage)
        axis = Axis.from_string(str(UsdGeom.GetStageUpAxis(stage)))

    if apply_up_axis_from_stage:
        builder.up_axis = axis
        axis_xform = wp.transform_identity()
        if verbose:
            print(f"Using stage up axis {axis} as builder up axis")
    else:
        axis_xform = wp.transform(wp.vec3(0.0), quat_between_axes(axis, builder.up_axis))
        if verbose:
            print(f"Rotating stage to align its up axis {axis} with builder up axis {builder.up_axis}")
    if xform is None:
        incoming_world_xform = axis_xform
    else:
        incoming_world_xform = wp.transform(*xform) * axis_xform

    if verbose:
        print(
            f"Scaling PD gains by (joint_drive_gains_scaling / DegreesToRadian) = {joint_drive_gains_scaling / DegreesToRadian}, default scale for joint_drive_gains_scaling=1 is 1.0/DegreesToRadian = {1.0 / DegreesToRadian}"
        )

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
    articulation_roots = []

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
        material_specs[str(sdf_path)] = PhysicsMaterial(
            staticFriction=desc.staticFriction,
            dynamicFriction=desc.dynamicFriction,
            restitution=desc.restitution,
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
                if has_attribute(prim, "physics:density"):
                    d = parse_float(prim, "physics:density")
                    density = d * mass_unit  # / (linear_unit**3)
                    body_density[body_path] = density
            # <--- Marking for deprecation

    # maps from articulation_id to bool indicating if self-collisions are enabled
    articulation_has_self_collision = {}

    if UsdPhysics.ObjectType.Articulation in ret_dict:
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

        paths, articulation_descs = ret_dict[UsdPhysics.ObjectType.Articulation]

        articulation_id = builder.articulation_count
        body_data = {}
        for path, desc in zip(paths, articulation_descs, strict=False):
            if warn_invalid_desc(path, desc):
                continue
            articulation_path = str(path)
            if any(re.match(p, articulation_path) for p in ignore_paths):
                continue
            articulation_prim = stage.GetPrimAtPath(path)
            body_ids = {}
            body_keys = []
            current_body_id = 0
            art_bodies = []
            if verbose:
                print(f"Bodies under articulation {path!s}:")
            for p in desc.articulatedBodies:
                if verbose:
                    print(f"\t{p!s}")
                key = str(p)
                if key in ignored_body_paths:
                    continue

                if p == Sdf.Path.emptyPath:
                    continue
                else:
                    usd_prim = stage.GetPrimAtPath(p)
                    if "TensorPhysicsArticulationRootAPI" in usd_prim.GetPrimTypeInfo().GetAppliedAPISchemas():
                        usd_prim.CreateAttribute(
                            "physics:newton:articulation_index", Sdf.ValueTypeNames.UInt, True
                        ).Set(articulation_id)
                        articulation_roots.append(key)

                if key in body_specs:
                    if bodies_follow_joint_ordering:
                        # we just parse the body information without yet adding it to the builder
                        body_data[current_body_id] = parse_body(
                            body_specs[key],
                            stage.GetPrimAtPath(p),
                            add_body_to_builder=False,
                        )
                    else:
                        # look up description and add body to builder
                        body_id = parse_body(
                            body_specs[key],
                            stage.GetPrimAtPath(p),
                            add_body_to_builder=True,
                        )
                        if body_id >= 0:
                            art_bodies.append(body_id)
                    # remove body spec once we inserted it
                    del body_specs[key]

                body_ids[key] = current_body_id
                body_keys.append(key)
                current_body_id += 1

            if len(body_ids) == 0:
                # no bodies under the articulation or we ignored all of them
                continue

            # determine the joint graph for this articulation
            joint_names = []
            joint_edges: list[tuple[int, int]] = []
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
                joint_names.append(joint_key)
                parent_id, child_id = resolve_joint_parent_child(joint_desc, body_ids, get_transforms=False)
                joint_edges.append((parent_id, child_id))

            articulation_xform = wp.mul(incoming_world_xform, parse_xform(articulation_prim))

            if len(joint_edges) == 0:
                # We have an articulation without joints, i.e. only free rigid bodies
                if bodies_follow_joint_ordering:
                    for i in body_ids.values():
                        builder.add_articulation(body_data[i]["key"])
                        child_body_id = add_body(**body_data[i])
                        # apply the articulation transform to the body
                        builder.body_q[child_body_id] = articulation_xform
                        builder.add_joint_free(child=child_body_id)
                        # note the free joint's coordinates will be initialized by the body_q of the
                        # child body
                else:
                    for i, child_body_id in enumerate(art_bodies):
                        builder.add_articulation(body_keys[i])
                        # apply the articulation transform to the body
                        builder.body_q[child_body_id] = articulation_xform
                        builder.add_joint_free(child=child_body_id)
                        # note the free joint's coordinates will be initialized by the body_q of the
                        # child body
                sorted_joints = []
            else:
                # we have an articulation with joints, we need to sort them topologically
                builder.add_articulation(articulation_path)
                if joint_ordering is not None:
                    if verbose:
                        print(f"Sorting joints using {joint_ordering} ordering...")
                    sorted_joints = topological_sort(joint_edges, use_dfs=joint_ordering == "dfs")
                    if verbose:
                        print("Joint ordering:", sorted_joints)
                else:
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
                    # to make sure Featherstone and MuJoCo can simulate it
                    if bodies_follow_joint_ordering:
                        child_body = body_data[first_joint_parent]
                        child_body_id = path_body_map[child_body["key"]]
                    else:
                        child_body_id = art_bodies[first_joint_parent]
                    # apply the articulation transform to the body
                    #! investigate why assigning body_q (joint_q) by art_xform * body_q is breaking the tests
                    # builder.body_q[child_body_id] = articulation_xform * builder.body_q[child_body_id]
                    builder.add_joint_free(child=child_body_id)
                    builder.joint_q[-7:] = articulation_xform

                # insert the remaining joints in topological order
                for joint_id, i in enumerate(sorted_joints):
                    if joint_id == 0 and first_joint_parent == -1:
                        # the articulation root joint receives the articulation transform as parent transform
                        # except if we already inserted a floating-base joint
                        parse_joint(
                            joint_descriptions[joint_names[i]],
                            joint_path=joint_names[i],
                            incoming_xform=articulation_xform,
                        )
                    else:
                        parse_joint(
                            joint_descriptions[joint_names[i]],
                            joint_path=joint_names[i],
                        )

            articulation_bodies[articulation_id] = art_bodies
            # determine if self-collisions are enabled
            articulation_has_self_collision[articulation_id] = parse_generic(
                articulation_prim,
                "physxArticulation:enabledSelfCollisions",
                default=enable_self_collisions,
            )
            articulation_id += 1

    # insert remaining bodies that were not part of any articulation so far
    for path, rigid_body_desc in body_specs.items():
        key = str(path)
        body_id = parse_body(
            rigid_body_desc,
            stage.GetPrimAtPath(path),
            incoming_xform=incoming_world_xform,
            add_body_to_builder=True,
        )
        # add articulation and free joint for this body
        builder.add_articulation(key)
        builder.add_joint_free(child=body_id)

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
                # print(prim)
                # print(shape_spec)
                if path in path_shape_map:
                    if verbose:
                        print(f"Shape at {path} already added, skipping.")
                    continue
                body_path = str(shape_spec.rigidBody)
                # print("shape ", prim, "body =" , body_path)
                body_id = path_body_map.get(body_path, -1)
                # scale = np.array(shape_spec.localScale)
                scale = parse_scale(prim)
                collision_group = -1
                if len(shape_spec.collisionGroups) > 0:
                    cgroup_name = str(shape_spec.collisionGroups[0])
                    if cgroup_name not in collision_group_ids:
                        collision_group_ids[cgroup_name] = len(collision_group_ids)
                    collision_group = collision_group_ids[cgroup_name]
                material = material_specs[""]
                if len(shape_spec.materials) >= 1:
                    if len(shape_spec.materials) > 1 and verbose:
                        print(f"Warning: More than one material found on shape at '{path}'.\nUsing only the first one.")
                    material = material_specs[str(shape_spec.materials[0])]
                    if verbose:
                        print(
                            f"\tMaterial of '{path}':\tfriction: {material.dynamicFriction},\trestitution: {material.restitution},\tdensity: {material.density}"
                        )
                elif verbose:
                    print(f"No material found for shape at '{path}'.")
                prim_and_scene = (prim, physics_scene_prim)
                local_xform = wp.transform(shape_spec.localPos, from_gfquat(shape_spec.localRot))
                if body_id == -1:
                    shape_xform = incoming_world_xform * local_xform
                else:
                    shape_xform = local_xform
                shape_params = {
                    "body": body_id,
                    "xform": shape_xform,
                    "cfg": ModelBuilder.ShapeConfig(
                        ke=parse_float_with_fallback(prim_and_scene, "newton:contact_ke", builder.default_shape_cfg.ke),
                        kd=parse_float_with_fallback(prim_and_scene, "newton:contact_kd", builder.default_shape_cfg.kd),
                        kf=parse_float_with_fallback(prim_and_scene, "newton:contact_kf", builder.default_shape_cfg.kf),
                        ka=parse_float_with_fallback(prim_and_scene, "newton:contact_ka", builder.default_shape_cfg.ka),
                        thickness=parse_float_with_fallback(
                            prim_and_scene, "newton:contact_thickness", builder.default_shape_cfg.thickness
                        ),
                        mu=material.dynamicFriction,
                        restitution=material.restitution,
                        density=body_density.get(body_path, default_shape_density),
                        collision_group=collision_group,
                        is_visible=not hide_collision_shapes,
                    ),
                    "key": path,
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
                    shape_id = builder.add_shape_sphere(
                        **shape_params,
                        radius=shape_spec.radius * scale[0],
                    )
                elif key == UsdPhysics.ObjectType.CapsuleShape:
                    # Apply axis rotation to transform
                    axis = int(shape_spec.axis)
                    shape_params["xform"] = wp.transform(
                        shape_params["xform"].p, shape_params["xform"].q * quat_between_axes(Axis.Z, axis)
                    )
                    shape_id = builder.add_shape_capsule(
                        **shape_params,
                        radius=shape_spec.radius * scale[(int(shape_spec.axis) + 1) % 3],
                        half_height=shape_spec.halfHeight * scale[int(shape_spec.axis)],
                    )
                elif key == UsdPhysics.ObjectType.CylinderShape:
                    # Apply axis rotation to transform
                    axis = int(shape_spec.axis)
                    shape_params["xform"] = wp.transform(
                        shape_params["xform"].p, shape_params["xform"].q * quat_between_axes(Axis.Z, axis)
                    )
                    shape_id = builder.add_shape_cylinder(
                        **shape_params,
                        radius=shape_spec.radius * scale[(int(shape_spec.axis) + 1) % 3],
                        half_height=shape_spec.halfHeight * scale[int(shape_spec.axis)],
                    )
                elif key == UsdPhysics.ObjectType.ConeShape:
                    # Apply axis rotation to transform
                    axis = int(shape_spec.axis)
                    shape_params["xform"] = wp.transform(
                        shape_params["xform"].p, shape_params["xform"].q * quat_between_axes(Axis.Z, axis)
                    )
                    shape_id = builder.add_shape_cone(
                        **shape_params,
                        radius=shape_spec.radius * scale[(int(shape_spec.axis) + 1) % 3],
                        half_height=shape_spec.halfHeight * scale[int(shape_spec.axis)],
                    )
                elif key == UsdPhysics.ObjectType.MeshShape:
                    mesh = UsdGeom.Mesh(prim)
                    points = np.array(mesh.GetPointsAttr().Get(), dtype=np.float32)
                    indices = np.array(mesh.GetFaceVertexIndicesAttr().Get(), dtype=np.float32)
                    counts = mesh.GetFaceVertexCountsAttr().Get()
                    faces = []
                    face_id = 0
                    for count in counts:
                        if count == 3:
                            faces.append(indices[face_id : face_id + 3])
                        elif count == 4:
                            faces.append(indices[face_id : face_id + 3])
                            faces.append(indices[[face_id, face_id + 2, face_id + 3]])
                        elif verbose:
                            print(
                                f"Error while parsing USD mesh {path}: encountered polygon with {count} vertices, but only triangles and quads are supported."
                            )
                            continue
                        face_id += count
                    m = Mesh(points, np.array(faces, dtype=np.int32).flatten(), maxhullvert=mesh_maxhullvert)
                    shape_id = builder.add_shape_mesh(
                        scale=scale,
                        mesh=m,
                        **shape_params,
                    )
                    if not skip_mesh_approximation:
                        approximation = parse_generic(prim, "physics:approximation", None)
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

                if not prim.HasAPI(UsdPhysics.CollisionAPI) or not parse_generic(
                    prim, "physics:collisionEnabled", True
                ):
                    no_collision_shapes.add(shape_id)
                    builder.shape_flags[shape_id] &= ~ShapeFlags.COLLIDE_SHAPES

    # approximate meshes
    for remeshing_method, shape_ids in remeshing_queue.items():
        builder.approximate_meshes(method=remeshing_method, shape_indices=shape_ids)

    # apply collision filters now that we have added all shapes
    for path1, path2 in path_collision_filters:
        shape1 = path_shape_map[path1]
        shape2 = path_shape_map[path2]
        builder.shape_collision_filter_pairs.append((shape1, shape2))

    # apply collision filters to all shapes that have no collision
    for shape_id in no_collision_shapes:
        for other_shape_id in range(builder.shape_count):
            if other_shape_id != shape_id:
                builder.shape_collision_filter_pairs.append((shape_id, other_shape_id))

    # apply collision filters from articulations that have self collisions disabled
    for art_id, bodies in articulation_bodies.items():
        if not articulation_has_self_collision[art_id]:
            for body1, body2 in itertools.combinations(bodies, 2):
                for shape1 in builder.body_shapes[body1]:
                    for shape2 in builder.body_shapes[body2]:
                        builder.shape_collision_filter_pairs.append((shape1, shape2))

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
            mass = parse_float(prim, "physics:mass")
            if mass is not None:
                builder.body_mass[body_id] = mass
                builder.body_inv_mass[body_id] = 1.0 / mass
            com = parse_vec(prim, "physics:centerOfMass")
            if com is not None:
                builder.body_com[body_id] = com
            i_diag = parse_vec(prim, "physics:diagonalInertia", np.zeros(3, dtype=np.float32))
            i_rot = parse_quat(prim, "physics:principalAxes", wp.quat_identity())
            if np.linalg.norm(i_diag) > 0.0:
                rot = np.array(wp.quat_to_matrix(i_rot), dtype=np.float32).reshape(3, 3)
                inertia = rot @ np.diag(i_diag) @ rot.T
                builder.body_inertia[body_id] = wp.mat33(inertia)
                if inertia.any():
                    builder.body_inv_inertia[body_id] = wp.inverse(wp.mat33(*inertia))
                else:
                    builder.body_inv_inertia[body_id] = wp.mat33(*np.zeros((3, 3), dtype=np.float32))

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
                        I_default += mass * d_squared * np.eye(3)

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
    new_bodies = path_body_map.values()
    builder.add_free_joints_to_floating_bodies(new_bodies)

    # collapsing fixed joints to reduce the number of simulated bodies connected by fixed joints.
    collapse_results = None
    merged_body_data = {}
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
        merged_body_data = collapse_results["merged_body_data"]

    path_original_body_map = path_body_map.copy()
    if cloned_world is not None:
        with wp.ScopedTimer("replicating worlds"):
            # instantiate worlds
            path_shape_map_updates = {}
            path_body_map_updates = {}
            path_shape_scale_updates = {}
            articulation_roots_updates = []
            articulation_bodies_updates = {}
            articulation_key = builder.articulation_key
            shape_key = builder.shape_key
            joint_key = builder.joint_key
            body_key = builder.body_key
            for world_path, world_xform in zip(cloned_world_paths, cloned_world_xforms, strict=False):
                shape_count = multi_world_builder.shape_count
                body_count = multi_world_builder.body_count
                original_body_count = multi_world_builder.body_count
                art_count = multi_world_builder.articulation_count
                # print("articulation_bodies = ", articulation_bodies)
                for path, shape_id in path_shape_map.items():
                    new_path = path.replace(cloned_world, world_path)
                    path_shape_map_updates[new_path] = shape_id + shape_count
                for path, body_id in path_body_map.items():
                    new_path = path.replace(cloned_world, world_path)
                    path_body_map_updates[new_path] = body_id + body_count
                    if collapse_fixed_joints:
                        original_body_id = path_original_body_map[path]
                        path_original_body_map[new_path] = original_body_id + original_body_count
                    if path in merged_body_data:
                        merged_body_data[new_path] = merged_body_data[path]
                        parent_path = merged_body_data[path]["parent_body"]
                        new_parent_path = parent_path.replace(cloned_world, world_path)
                        merged_body_data[new_path]["parent_body"] = new_parent_path

                for path, scale in path_shape_scale.items():
                    new_path = path.replace(cloned_world, world_path)
                    path_shape_scale_updates[new_path] = scale
                for art_id, bodies in articulation_bodies.items():
                    articulation_bodies_updates[art_id + art_count] = [b + body_count for b in bodies]
                for root in articulation_roots:
                    new_path = root.replace(cloned_world, world_path)
                    articulation_roots_updates.append(new_path)

                builder.articulation_key = [key.replace(cloned_world, world_path) for key in articulation_key]
                builder.shape_key = [key.replace(cloned_world, world_path) for key in shape_key]
                builder.joint_key = [key.replace(cloned_world, world_path) for key in joint_key]
                builder.body_key = [key.replace(cloned_world, world_path) for key in body_key]
                multi_world_builder.add_builder(builder, xform=world_xform)

            path_shape_map = path_shape_map_updates
            path_body_map = path_body_map_updates
            path_shape_scale = path_shape_scale_updates
            articulation_roots = articulation_roots_updates
            articulation_bodies = articulation_bodies_updates

            builder = multi_world_builder

    return {
        "fps": stage.GetFramesPerSecond(),
        "duration": stage.GetEndTimeCode() - stage.GetStartTimeCode(),
        "up_axis": Axis.from_string(UsdGeom.GetStageUpAxis(stage)),
        "path_shape_map": path_shape_map,
        "path_body_map": path_body_map,
        "path_shape_scale": path_shape_scale,
        "mass_unit": mass_unit,
        "linear_unit": linear_unit,
        "scene_attributes": scene_attributes,
        "collapse_results": collapse_results,
        # "articulation_roots": articulation_roots,
        # "articulation_bodies": articulation_bodies,
        "path_body_relative_transform": path_body_relative_transform,
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
