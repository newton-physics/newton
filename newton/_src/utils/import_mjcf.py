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

import math
import os
import re
import xml.etree.ElementTree as ET
from typing import Any

import numpy as np
import warp as wp

from ..core import quat_between_axes, quat_from_euler
from ..core.types import Axis, AxisType, Sequence, Transform
from ..geometry import MESH_MAXHULLVERT, Mesh
from ..sim import JointType, ModelBuilder
from ..sim.model import ModelAttributeFrequency
from ..usd.schemas import solref_to_stiffness_damping
from .import_utils import parse_custom_attributes, sanitize_xml_content


def parse_mjcf(
    builder: ModelBuilder,
    source: str,
    xform: Transform | None = None,
    floating: bool | None = None,
    base_joint: dict | str | None = None,
    armature_scale: float = 1.0,
    scale: float = 1.0,
    hide_visuals: bool = False,
    parse_visuals_as_colliders: bool = False,
    parse_meshes: bool = True,
    parse_sites: bool = True,
    parse_visuals: bool = True,
    up_axis: AxisType = Axis.Z,
    ignore_names: Sequence[str] = (),
    ignore_classes: Sequence[str] = (),
    visual_classes: Sequence[str] = ("visual",),
    collider_classes: Sequence[str] = ("collision",),
    no_class_as_colliders: bool = True,
    force_show_colliders: bool = False,
    enable_self_collisions: bool = False,
    ignore_inertial_definitions: bool = True,
    ensure_nonstatic_links: bool = True,
    static_link_mass: float = 1e-2,
    collapse_fixed_joints: bool = False,
    verbose: bool = False,
    skip_equality_constraints: bool = False,
    mesh_maxhullvert: int = MESH_MAXHULLVERT,
):
    """
    Parses MuJoCo XML (MJCF) file and adds the bodies and joints to the given ModelBuilder.

    Args:
        builder (ModelBuilder): The :class:`ModelBuilder` to add the bodies and joints to.
        source (str): The filename of the MuJoCo file to parse, or the MJCF XML string content.
        xform (Transform): The transform to apply to the imported mechanism.
        floating (bool): If True, the articulation is treated as a floating base. If False, the articulation is treated as a fixed base. If None, the articulation is treated as a floating base if a free joint is found in the MJCF, otherwise it is treated as a fixed base.
        base_joint (Union[str, dict]): The joint by which the root body is connected to the world. This can be either a string defining the joint axes of a D6 joint with comma-separated positional and angular axis names (e.g. "px,py,rz" for a D6 joint with linear axes in x, y and an angular axis in z) or a dict with joint parameters (see :meth:`ModelBuilder.add_joint`).
        armature_scale (float): Scaling factor to apply to the MJCF-defined joint armature values.
        scale (float): The scaling factor to apply to the imported mechanism.
        hide_visuals (bool): If True, hide visual shapes after loading them (affects visibility, not loading).
        parse_visuals_as_colliders (bool): If True, the geometry defined under the `visual_classes` tags is used for collision handling instead of the `collider_classes` geometries.
        parse_meshes (bool): Whether geometries of type `"mesh"` should be parsed. If False, geometries of type `"mesh"` are ignored.
        parse_sites (bool): Whether sites (non-colliding reference points) should be parsed. If False, sites are ignored.
        parse_visuals (bool): Whether visual geometries (non-collision shapes) should be loaded. If False, visual shapes are not loaded (different from `hide_visuals` which loads but hides them). Default is True.
        up_axis (AxisType): The up axis of the MuJoCo scene. The default is Z up.
        ignore_names (Sequence[str]): A list of regular expressions. Bodies and joints with a name matching one of the regular expressions will be ignored.
        ignore_classes (Sequence[str]): A list of regular expressions. Bodies and joints with a class matching one of the regular expressions will be ignored.
        visual_classes (Sequence[str]): A list of regular expressions. Visual geometries with a class matching one of the regular expressions will be parsed.
        collider_classes (Sequence[str]): A list of regular expressions. Collision geometries with a class matching one of the regular expressions will be parsed.
        no_class_as_colliders: If True, geometries without a class are parsed as collision geometries. If False, geometries without a class are parsed as visual geometries.
        force_show_colliders (bool): If True, the collision shapes are always shown, even if there are visual shapes.
        enable_self_collisions (bool): If True, self-collisions are enabled.
        ignore_inertial_definitions (bool): If True, the inertial parameters defined in the MJCF are ignored and the inertia is calculated from the shape geometry.
        ensure_nonstatic_links (bool): If True, links with zero mass are given a small mass (see `static_link_mass`) to ensure they are dynamic.
        static_link_mass (float): The mass to assign to links with zero mass (if `ensure_nonstatic_links` is set to True).
        collapse_fixed_joints (bool): If True, fixed joints are removed and the respective bodies are merged.
        verbose (bool): If True, print additional information about parsing the MJCF.
        skip_equality_constraints (bool): Whether <equality> tags should be parsed. If True, equality constraints are ignored.
        mesh_maxhullvert (int): Maximum vertices for convex hull approximation of meshes.
    """
    if xform is None:
        xform = wp.transform_identity()
    else:
        xform = wp.transform(*xform)

    if os.path.isfile(source):
        mjcf_dirname = os.path.dirname(source)
        file = ET.parse(source)
        root = file.getroot()
    else:
        xml_content = sanitize_xml_content(source)
        root = ET.fromstring(xml_content)
        mjcf_dirname = "."

    use_degrees = True  # angles are in degrees by default
    euler_seq = [0, 1, 2]  # XYZ by default

    # load joint defaults
    default_joint_limit_lower = builder.default_joint_cfg.limit_lower
    default_joint_limit_upper = builder.default_joint_cfg.limit_upper
    default_joint_stiffness = builder.default_joint_cfg.target_ke
    default_joint_damping = builder.default_joint_cfg.target_kd
    default_joint_armature = builder.default_joint_cfg.armature

    # load shape defaults
    default_shape_density = builder.default_shape_cfg.density

    # Process custom attributes defined for different kinds of shapes, bodies, joints, etc.
    builder_custom_attr_shape: list[ModelBuilder.CustomAttribute] = builder.get_custom_attributes_by_frequency(
        [ModelAttributeFrequency.SHAPE]
    )
    builder_custom_attr_body: list[ModelBuilder.CustomAttribute] = builder.get_custom_attributes_by_frequency(
        [ModelAttributeFrequency.BODY]
    )
    builder_custom_attr_joint: list[ModelBuilder.CustomAttribute] = builder.get_custom_attributes_by_frequency(
        [ModelAttributeFrequency.JOINT]
    )
    builder_custom_attr_dof: list[ModelBuilder.CustomAttribute] = builder.get_custom_attributes_by_frequency(
        [ModelAttributeFrequency.JOINT_DOF]
    )

    compiler = root.find("compiler")
    if compiler is not None:
        use_degrees = compiler.attrib.get("angle", "degree").lower() == "degree"
        euler_seq = ["xyz".index(c) for c in compiler.attrib.get("eulerseq", "xyz").lower()]
        mesh_dir = compiler.attrib.get("meshdir", ".")
    else:
        mesh_dir = "."

    mesh_assets = {}
    for asset in root.findall("asset"):
        for mesh in asset.findall("mesh"):
            if "file" in mesh.attrib:
                fname = os.path.join(mesh_dir, mesh.attrib["file"])
                # handle stl relative paths
                if not os.path.isabs(fname):
                    fname = os.path.abspath(os.path.join(mjcf_dirname, fname))
                name = mesh.attrib.get("name", ".".join(os.path.basename(fname).split(".")[:-1]))
                s = mesh.attrib.get("scale", "1.0 1.0 1.0")
                s = np.fromstring(s, sep=" ", dtype=np.float32)
                # parse maxhullvert attribute, default to mesh_maxhullvert if not specified
                maxhullvert = int(mesh.attrib.get("maxhullvert", str(mesh_maxhullvert)))
                mesh_assets[name] = {"file": fname, "scale": s, "maxhullvert": maxhullvert}

    class_parent = {}
    class_children = {}
    class_defaults = {"__all__": {}}

    def get_class(element) -> str:
        return element.get("class", "__all__")

    def parse_default(node, parent):
        nonlocal class_parent
        nonlocal class_children
        nonlocal class_defaults
        class_name = "__all__"
        if "class" in node.attrib:
            class_name = node.attrib["class"]
            class_parent[class_name] = parent
            parent = parent or "__all__"
            if parent not in class_children:
                class_children[parent] = []
            class_children[parent].append(class_name)

        if class_name not in class_defaults:
            class_defaults[class_name] = {}
        for child in node:
            if child.tag == "default":
                parse_default(child, node.get("class"))
            else:
                class_defaults[class_name][child.tag] = child.attrib

    for default in root.findall("default"):
        parse_default(default, None)

    def merge_attrib(default_attrib: dict, incoming_attrib: dict) -> dict:
        attrib = default_attrib.copy()
        for key, value in incoming_attrib.items():
            if key in attrib:
                if isinstance(attrib[key], dict):
                    attrib[key] = merge_attrib(attrib[key], value)
                else:
                    attrib[key] = value
            else:
                attrib[key] = value
        return attrib

    axis_xform = wp.transform(wp.vec3(0.0), quat_between_axes(up_axis, builder.up_axis))
    xform = xform * axis_xform

    def parse_float(attrib, key, default) -> float:
        if key in attrib:
            return float(attrib[key])
        else:
            return default

    def parse_vec(attrib, key, default):
        if key in attrib:
            out = np.fromstring(attrib[key], sep=" ", dtype=np.float32)
        else:
            out = np.array(default, dtype=np.float32)

        length = len(out)
        if length == 1:
            return wp.vec(len(default), wp.float32)(out[0], out[0], out[0])

        return wp.vec(length, wp.float32)(out)

    def parse_orientation(attrib) -> wp.quat:
        if "quat" in attrib:
            wxyz = np.fromstring(attrib["quat"], sep=" ")
            return wp.normalize(wp.quat(*wxyz[1:], wxyz[0]))
        if "euler" in attrib:
            euler = np.fromstring(attrib["euler"], sep=" ")
            if use_degrees:
                euler *= np.pi / 180
            return quat_from_euler(wp.vec3(euler), *euler_seq)
        if "axisangle" in attrib:
            axisangle = np.fromstring(attrib["axisangle"], sep=" ")
            angle = axisangle[3]
            if use_degrees:
                angle *= np.pi / 180
            axis = wp.normalize(wp.vec3(*axisangle[:3]))
            return wp.quat_from_axis_angle(axis, float(angle))
        if "xyaxes" in attrib:
            xyaxes = np.fromstring(attrib["xyaxes"], sep=" ")
            xaxis = wp.normalize(wp.vec3(*xyaxes[:3]))
            zaxis = wp.normalize(wp.vec3(*xyaxes[3:]))
            yaxis = wp.normalize(wp.cross(zaxis, xaxis))
            rot_matrix = np.array([xaxis, yaxis, zaxis]).T
            return wp.quat_from_matrix(wp.mat33(rot_matrix))
        if "zaxis" in attrib:
            zaxis = np.fromstring(attrib["zaxis"], sep=" ")
            zaxis = wp.normalize(wp.vec3(*zaxis))
            xaxis = wp.normalize(wp.cross(wp.vec3(0, 0, 1), zaxis))
            yaxis = wp.normalize(wp.cross(zaxis, xaxis))
            rot_matrix = np.array([xaxis, yaxis, zaxis]).T
            return wp.quat_from_matrix(wp.mat33(rot_matrix))
        return wp.quat_identity()

    def parse_shapes(defaults, body_name, link, geoms, density, visible=True, just_visual=False, incoming_xform=None):
        shapes = []
        for geo_count, geom in enumerate(geoms):
            geom_defaults = defaults
            if "class" in geom.attrib:
                geom_class = geom.attrib["class"]
                ignore_geom = False
                for pattern in ignore_classes:
                    if re.match(pattern, geom_class):
                        ignore_geom = True
                        break
                if ignore_geom:
                    continue
                if geom_class in class_defaults:
                    geom_defaults = merge_attrib(defaults, class_defaults[geom_class])
            if "geom" in geom_defaults:
                geom_attrib = merge_attrib(geom_defaults["geom"], geom.attrib)
            else:
                geom_attrib = geom.attrib

            geom_name = geom_attrib.get("name", f"{body_name}_geom_{geo_count}{'_visual' if just_visual else ''}")
            geom_type = geom_attrib.get("type", "sphere")
            if "mesh" in geom_attrib:
                geom_type = "mesh"

            ignore_geom = False
            for pattern in ignore_names:
                if re.match(pattern, geom_name):
                    ignore_geom = True
                    break
            if ignore_geom:
                continue

            geom_size = parse_vec(geom_attrib, "size", [1.0, 1.0, 1.0]) * scale
            geom_pos = parse_vec(geom_attrib, "pos", (0.0, 0.0, 0.0)) * scale
            geom_rot = parse_orientation(geom_attrib)
            tf = wp.transform(geom_pos, geom_rot)
            if link == -1 and incoming_xform is not None:
                tf = incoming_xform * tf
                geom_pos = tf.p
                geom_rot = tf.q

            geom_density = parse_float(geom_attrib, "density", density)

            shape_cfg = builder.default_shape_cfg.copy()
            shape_cfg.is_visible = visible
            shape_cfg.has_shape_collision = not just_visual
            shape_cfg.has_particle_collision = not just_visual
            shape_cfg.density = geom_density

            # Parse MJCF friction: "slide [torsion [roll]]"
            # Can't use parse_vec - it would replicate single values to all dimensions
            if "friction" in geom_attrib:
                friction_values = np.fromstring(geom_attrib["friction"], sep=" ", dtype=np.float32)

                if len(friction_values) >= 1:
                    shape_cfg.mu = float(friction_values[0])

                if len(friction_values) >= 2:
                    shape_cfg.torsional_friction = float(friction_values[1])

                if len(friction_values) >= 3:
                    shape_cfg.rolling_friction = float(friction_values[2])

            custom_attributes = parse_custom_attributes(geom_attrib, builder_custom_attr_shape, parsing_mode="mjcf")
            shape_kwargs = {
                "key": geom_name,
                "body": link,
                "cfg": shape_cfg,
                "custom_attributes": custom_attributes,
            }

            if incoming_xform is not None:
                tf = incoming_xform * tf

            if geom_type == "sphere":
                s = builder.add_shape_sphere(
                    xform=tf,
                    radius=geom_size[0],
                    **shape_kwargs,
                )
                shapes.append(s)

            elif geom_type == "box":
                s = builder.add_shape_box(
                    xform=tf,
                    hx=geom_size[0],
                    hy=geom_size[1],
                    hz=geom_size[2],
                    **shape_kwargs,
                )
                shapes.append(s)

            elif geom_type == "mesh" and parse_meshes:
                import trimesh  # noqa: PLC0415

                # use force='mesh' to load the mesh as a trimesh object
                # with baked in transforms, e.g. from COLLADA files
                stl_file = mesh_assets[geom_attrib["mesh"]]["file"]
                m = trimesh.load(stl_file, force="mesh")
                if "mesh" in geom_defaults:
                    mesh_scale = parse_vec(geom_defaults["mesh"], "scale", mesh_assets[geom_attrib["mesh"]]["scale"])
                else:
                    mesh_scale = mesh_assets[geom_attrib["mesh"]]["scale"]
                scaling = np.array(mesh_scale) * scale
                # as per the Mujoco XML reference, ignore geom size attribute
                assert len(geom_size) == 3, "need to specify size for mesh geom"

                # get maxhullvert value from mesh assets
                maxhullvert = mesh_assets[geom_attrib["mesh"]].get("maxhullvert", mesh_maxhullvert)

                if hasattr(m, "geometry"):
                    # multiple meshes are contained in a scene
                    for m_geom in m.geometry.values():
                        m_vertices = np.array(m_geom.vertices, dtype=np.float32) * scaling
                        m_faces = np.array(m_geom.faces.flatten(), dtype=np.int32)
                        m_mesh = Mesh(
                            m_vertices,
                            m_faces,
                            m.vertex_normals,
                            color=np.array(m.visual.main_color) / 255.0,
                            maxhullvert=maxhullvert,
                        )
                        s = builder.add_shape_mesh(
                            xform=tf,
                            mesh=m_mesh,
                            **shape_kwargs,
                        )
                        shapes.append(s)
                else:
                    # a single mesh
                    m_vertices = np.array(m.vertices, dtype=np.float32) * scaling
                    m_faces = np.array(m.faces.flatten(), dtype=np.int32)
                    m_color = np.array(m.visual.main_color) / 255.0 if hasattr(m.visual, "main_color") else None
                    m_mesh = Mesh(
                        m_vertices,
                        m_faces,
                        m.vertex_normals,
                        color=m_color,
                        maxhullvert=maxhullvert,
                    )
                    s = builder.add_shape_mesh(
                        xform=tf,
                        mesh=m_mesh,
                        **shape_kwargs,
                    )
                    shapes.append(s)

            elif geom_type in {"capsule", "cylinder"}:
                if "fromto" in geom_attrib:
                    geom_fromto = parse_vec(geom_attrib, "fromto", (0.0, 0.0, 0.0, 1.0, 0.0, 0.0))

                    start = wp.vec3(geom_fromto[0:3]) * scale
                    end = wp.vec3(geom_fromto[3:6]) * scale

                    # compute rotation to align the Warp capsule (along x-axis), with mjcf fromto direction
                    axis = wp.normalize(end - start)
                    angle = math.acos(wp.dot(axis, wp.vec3(0.0, 1.0, 0.0)))
                    axis = wp.normalize(wp.cross(axis, wp.vec3(0.0, 1.0, 0.0)))

                    geom_pos = (start + end) * 0.5
                    geom_rot = wp.quat_from_axis_angle(axis, -angle)
                    tf = wp.transform(geom_pos, geom_rot)

                    geom_radius = geom_size[0]
                    geom_height = wp.length(end - start) * 0.5
                    geom_up_axis = Axis.Y

                else:
                    geom_radius = geom_size[0]
                    geom_height = geom_size[1]
                    geom_up_axis = up_axis

                # Apply axis rotation to transform
                tf = wp.transform(tf.p, tf.q * quat_between_axes(Axis.Z, geom_up_axis))

                if geom_type == "cylinder":
                    s = builder.add_shape_cylinder(
                        xform=tf,
                        radius=geom_radius,
                        half_height=geom_height,
                        **shape_kwargs,
                    )
                    shapes.append(s)
                else:
                    s = builder.add_shape_capsule(
                        xform=tf,
                        radius=geom_radius,
                        half_height=geom_height,
                        **shape_kwargs,
                    )
                    shapes.append(s)

            elif geom_type == "plane":
                normal = wp.quat_rotate(geom_rot, wp.vec3(0.0, 0.0, 1.0))
                p = wp.dot(geom_pos, normal)
                s = builder.add_shape_plane(
                    plane=(*normal, p),
                    width=geom_size[0],
                    length=geom_size[1],
                    **shape_kwargs,
                )
                shapes.append(s)

            else:
                if verbose:
                    print(f"MJCF parsing shape {geom_name} issue: geom type {geom_type} is unsupported")

        return shapes

    def _parse_sites_impl(defaults, body_name, link, sites, incoming_xform=None):
        """Parse site elements from MJCF."""
        from ..geometry import GeoType  # noqa: PLC0415

        site_shapes = []
        for site_count, site in enumerate(sites):
            site_defaults = defaults
            if "class" in site.attrib:
                site_class = site.attrib["class"]
                ignore_site = False
                for pattern in ignore_classes:
                    if re.match(pattern, site_class):
                        ignore_site = True
                        break
                if ignore_site:
                    continue
                if site_class in class_defaults:
                    site_defaults = merge_attrib(defaults, class_defaults[site_class])

            if "site" in site_defaults:
                site_attrib = merge_attrib(site_defaults["site"], site.attrib)
            else:
                site_attrib = site.attrib

            site_name = site_attrib.get("name", f"{body_name}_site_{site_count}")

            # Check if site should be ignored by name
            ignore_site = False
            for pattern in ignore_names:
                if re.match(pattern, site_name):
                    ignore_site = True
                    break
            if ignore_site:
                continue

            # Parse site transform
            site_pos = parse_vec(site_attrib, "pos", (0.0, 0.0, 0.0)) * scale
            site_rot = parse_orientation(site_attrib)
            site_xform = wp.transform(site_pos, site_rot)

            if incoming_xform is not None:
                site_xform = incoming_xform * site_xform

            # Parse site type (defaults to sphere if not specified)
            site_type = site_attrib.get("type", "sphere")
            site_size = parse_vec(site_attrib, "size", [0.01, 0.01, 0.01]) * scale

            # Map MuJoCo site types to Newton GeoType
            type_map = {
                "sphere": GeoType.SPHERE,
                "box": GeoType.BOX,
                "capsule": GeoType.CAPSULE,
                "cylinder": GeoType.CYLINDER,
                "ellipsoid": GeoType.ELLIPSOID,
            }
            geo_type = type_map.get(site_type, GeoType.SPHERE)

            # Sites are typically hidden by default
            visible = False

            # Expand to 3-element vector
            if len(site_size) == 2:
                # Two values (e.g., capsule/cylinder: radius, half-height)
                radius = site_size[0]
                half_height = site_size[1]
                site_size = wp.vec3(radius, half_height, 0.0)

            # Add site using builder.add_site()
            s = builder.add_site(
                body=link,
                xform=site_xform,
                type=geo_type,
                scale=site_size,
                key=site_name,
                visible=visible,
            )
            site_shapes.append(s)

        return site_shapes

    def parse_body(
        body,
        parent,
        incoming_defaults: dict,
        childclass: str | None = None,
        parent_world_xform: Transform | None = None,
    ):
        body_class = body.get("class") or body.get("childclass")
        if body_class is None:
            body_class = childclass
            defaults = incoming_defaults
        else:
            for pattern in ignore_classes:
                if re.match(pattern, body_class):
                    return
            defaults = merge_attrib(incoming_defaults, class_defaults[body_class])
        if "body" in defaults:
            body_attrib = merge_attrib(defaults["body"], body.attrib)
        else:
            body_attrib = body.attrib
        body_name = body_attrib.get("name", f"body_{builder.body_count}")
        body_name = body_name.replace("-", "_")  # ensure valid USD path
        body_pos = parse_vec(body_attrib, "pos", (0.0, 0.0, 0.0))
        body_ori = parse_orientation(body_attrib)

        # Create local transform from parsed position and orientation
        local_xform = wp.transform(body_pos * scale, body_ori)

        # Compose with either the passed parent world transform or the import root xform
        world_xform = (parent_world_xform or xform) * local_xform

        # For joint positioning, we need the relative position/orientation scaled
        body_pos_for_joints = body_pos * scale
        body_ori_for_joints = body_ori

        joint_armature = []
        joint_name = []
        joint_pos = []
        joint_custom_attributes: dict[str, Any] = {}
        dof_custom_attributes: dict[str, dict[int, Any]] = {}

        linear_axes = []
        angular_axes = []
        joint_type = None

        freejoint_tags = body.findall("freejoint")
        if len(freejoint_tags) > 0:
            joint_type = JointType.FREE
            joint_name.append(freejoint_tags[0].attrib.get("name", f"{body_name}_freejoint"))
            joint_armature.append(0.0)
            joint_custom_attributes = parse_custom_attributes(
                freejoint_tags[0].attrib, builder_custom_attr_joint, parsing_mode="mjcf"
            )
        else:
            # DOF index relative to the joint being created (multiple MJCF joints in a body are combined into one Newton joint)
            current_dof_index = 0
            joints = body.findall("joint")
            for i, joint in enumerate(joints):
                joint_defaults = defaults
                if "class" in joint.attrib:
                    joint_class = joint.attrib["class"]
                    if joint_class in class_defaults:
                        joint_defaults = merge_attrib(joint_defaults, class_defaults[joint_class])
                if "joint" in joint_defaults:
                    joint_attrib = merge_attrib(joint_defaults["joint"], joint.attrib)
                else:
                    joint_attrib = joint.attrib

                # default to hinge if not specified
                joint_type_str = joint_attrib.get("type", "hinge")

                joint_name.append(joint_attrib.get("name") or f"{body_name}_joint_{i}")
                joint_pos.append(parse_vec(joint_attrib, "pos", (0.0, 0.0, 0.0)) * scale)
                joint_range = parse_vec(joint_attrib, "range", (default_joint_limit_lower, default_joint_limit_upper))
                joint_armature.append(parse_float(joint_attrib, "armature", default_joint_armature) * armature_scale)

                if joint_type_str == "free":
                    joint_type = JointType.FREE
                    break
                if joint_type_str == "fixed":
                    joint_type = JointType.FIXED
                    break
                is_angular = joint_type_str == "hinge"
                axis_vec = parse_vec(joint_attrib, "axis", (0.0, 0.0, 0.0))
                limit_lower = np.deg2rad(joint_range[0]) if is_angular and use_degrees else joint_range[0]
                limit_upper = np.deg2rad(joint_range[1]) if is_angular and use_degrees else joint_range[1]

                # Parse solreflimit for joint limit stiffness and damping
                solreflimit = parse_vec(joint_attrib, "solreflimit", (0.02, 1.0))
                limit_ke, limit_kd = solref_to_stiffness_damping(solreflimit)
                # Handle None return values (invalid solref)
                if limit_ke is None:
                    limit_ke = 2500.0  # From MuJoCo's default solref (0.02, 1.0)
                if limit_kd is None:
                    limit_kd = 100.0  # From MuJoCo's default solref (0.02, 1.0)

                ax = ModelBuilder.JointDofConfig(
                    axis=axis_vec,
                    limit_lower=limit_lower,
                    limit_upper=limit_upper,
                    limit_ke=limit_ke,
                    limit_kd=limit_kd,
                    target_ke=parse_float(joint_attrib, "stiffness", default_joint_stiffness),
                    target_kd=parse_float(joint_attrib, "damping", default_joint_damping),
                    armature=joint_armature[-1],
                )
                if is_angular:
                    angular_axes.append(ax)
                else:
                    linear_axes.append(ax)

                dof_attr = parse_custom_attributes(joint_attrib, builder_custom_attr_dof, parsing_mode="mjcf")
                # assemble custom attributes for each DOF (dict mapping DOF index to value)
                # Only store values that were explicitly specified in the source
                for key, value in dof_attr.items():
                    if key not in dof_custom_attributes:
                        dof_custom_attributes[key] = {}
                    dof_custom_attributes[key][current_dof_index] = value

                current_dof_index += 1

        body_custom_attributes = parse_custom_attributes(body_attrib, builder_custom_attr_body, parsing_mode="mjcf")
        link = builder.add_body(
            xform=world_xform,  # Use the composed world transform
            key=body_name,
            custom_attributes=body_custom_attributes,
        )

        if joint_type is None:
            joint_type = JointType.D6
            if len(linear_axes) == 0:
                if len(angular_axes) == 0:
                    joint_type = JointType.FIXED
                elif len(angular_axes) == 1:
                    joint_type = JointType.REVOLUTE
            elif len(linear_axes) == 1 and len(angular_axes) == 0:
                joint_type = JointType.PRISMATIC

        if len(freejoint_tags) > 0 and parent == -1 and (base_joint is not None or floating is not None):
            joint_pos = joint_pos[0] if len(joint_pos) > 0 else wp.vec3(0.0, 0.0, 0.0)
            _xform = wp.transform(body_pos_for_joints + joint_pos, body_ori_for_joints)

            if base_joint is not None:
                # in case of a given base joint, the position is applied first, the rotation only
                # after the base joint itself to not rotate its axis
                base_parent_xform = wp.transform(_xform.p, wp.quat_identity())
                base_child_xform = wp.transform((0.0, 0.0, 0.0), wp.quat_inverse(_xform.q))
                if isinstance(base_joint, str):
                    axes = base_joint.lower().split(",")
                    axes = [ax.strip() for ax in axes]
                    linear_axes = [ax[-1] for ax in axes if ax[0] in {"l", "p"}]
                    angular_axes = [ax[-1] for ax in axes if ax[0] in {"a", "r"}]
                    axes = {
                        "x": [1.0, 0.0, 0.0],
                        "y": [0.0, 1.0, 0.0],
                        "z": [0.0, 0.0, 1.0],
                    }
                    builder.add_joint_d6(
                        linear_axes=[ModelBuilder.JointDofConfig(axis=axes[a]) for a in linear_axes],
                        angular_axes=[ModelBuilder.JointDofConfig(axis=axes[a]) for a in angular_axes],
                        parent_xform=base_parent_xform,
                        child_xform=base_child_xform,
                        parent=-1,
                        child=link,
                        key="base_joint",
                    )
                elif isinstance(base_joint, dict):
                    base_joint["parent"] = -1
                    base_joint["child"] = root
                    base_joint["parent_xform"] = base_parent_xform
                    base_joint["child_xform"] = base_child_xform
                    base_joint["key"] = "base_joint"
                    builder.add_joint(**base_joint)
                else:
                    raise ValueError(
                        "base_joint must be a comma-separated string of joint axes or a dict with joint parameters"
                    )
            elif floating is not None and floating:
                builder.add_joint_free(link, key="floating_base")
            else:
                builder.add_joint_fixed(-1, link, parent_xform=_xform, key="fixed_base")

        else:
            joint_pos = joint_pos[0] if len(joint_pos) > 0 else wp.vec3(0.0, 0.0, 0.0)
            if len(joint_name) == 0:
                joint_name = [f"{body_name}_joint"]
            if joint_type == JointType.FREE:
                assert parent == -1, "Free joints must have the world body as parent"
                builder.add_joint_free(
                    link,
                    key="_".join(joint_name),
                    custom_attributes=joint_custom_attributes,
                )
            else:
                # TODO parse ref, springref values from joint_attrib
                builder.add_joint(
                    joint_type,
                    parent=parent,
                    child=link,
                    linear_axes=linear_axes,
                    angular_axes=angular_axes,
                    key="_".join(joint_name),
                    parent_xform=wp.transform(body_pos_for_joints + joint_pos, body_ori_for_joints),
                    child_xform=wp.transform(joint_pos, wp.quat_identity()),
                    custom_attributes=joint_custom_attributes | dof_custom_attributes,
                )

        # -----------------
        # add shapes

        geoms = body.findall("geom")
        visuals = []
        colliders = []
        for geo_count, geom in enumerate(geoms):
            geom_defaults = defaults
            if "class" in geom.attrib:
                geom_class = geom.attrib["class"]
                ignore_geom = False
                for pattern in ignore_classes:
                    if re.match(pattern, geom_class):
                        ignore_geom = True
                        break
                if ignore_geom:
                    continue
                if geom_class in class_defaults:
                    geom_defaults = merge_attrib(defaults, class_defaults[geom_class])
            if "geom" in geom_defaults:
                geom_attrib = merge_attrib(geom_defaults["geom"], geom.attrib)
            else:
                geom_attrib = geom.attrib

            geom_name = geom_attrib.get("name", f"{body_name}_geom_{geo_count}")

            contype = geom_attrib.get("contype", 1)
            conaffinity = geom_attrib.get("conaffinity", 1)
            collides_with_anything = not (int(contype) == 0 and int(conaffinity) == 0)

            if "class" in geom.attrib:
                neither_visual_nor_collider = True
                for pattern in visual_classes:
                    if re.match(pattern, geom_class):
                        visuals.append(geom)
                        neither_visual_nor_collider = False
                        break
                for pattern in collider_classes:
                    if re.match(pattern, geom_class):
                        colliders.append(geom)
                        neither_visual_nor_collider = False
                        break
                if neither_visual_nor_collider:
                    if no_class_as_colliders and collides_with_anything:
                        colliders.append(geom)
                    else:
                        visuals.append(geom)
            else:
                no_class_class = "collision" if no_class_as_colliders else "visual"
                if verbose:
                    print(f"MJCF parsing shape {geom_name} issue: no class defined for geom, assuming {no_class_class}")
                if no_class_as_colliders and collides_with_anything:
                    colliders.append(geom)
                else:
                    visuals.append(geom)

        if parse_visuals_as_colliders:
            colliders = visuals
        elif parse_visuals:
            s = parse_shapes(
                defaults,
                body_name,
                link,
                geoms=visuals,
                density=0.0,
                just_visual=True,
                visible=not hide_visuals,
            )
            visual_shapes.extend(s)

        show_colliders = force_show_colliders
        if parse_visuals_as_colliders:
            show_colliders = True
        elif len(visuals) == 0 or not parse_visuals:
            # we need to show the collision shapes since there are no visual shapes (or we're not loading them)
            show_colliders = True

        parse_shapes(
            defaults,
            body_name,
            link,
            geoms=colliders,
            density=default_shape_density,
            visible=show_colliders,
        )

        # Parse sites (non-colliding reference points)
        if parse_sites:
            sites = body.findall("site")
            if sites:
                _parse_sites_impl(
                    defaults,
                    body_name,
                    link,
                    sites=sites,
                )

        m = builder.body_mass[link]
        if not ignore_inertial_definitions and body.find("inertial") is not None:
            inertial = body.find("inertial")
            if "inertial" in defaults:
                inertial_attrib = merge_attrib(defaults["inertial"], inertial.attrib)
            else:
                inertial_attrib = inertial.attrib
            # overwrite inertial parameters if defined
            inertial_pos = parse_vec(inertial_attrib, "pos", (0.0, 0.0, 0.0)) * scale
            inertial_rot = parse_orientation(inertial_attrib)

            inertial_frame = wp.transform(inertial_pos, inertial_rot)
            com = inertial_frame.p
            if inertial_attrib.get("diaginertia") is not None:
                diaginertia = parse_vec(inertial_attrib, "diaginertia", (0.0, 0.0, 0.0))
                I_m = np.zeros((3, 3))
                I_m[0, 0] = diaginertia[0] * scale**2
                I_m[1, 1] = diaginertia[1] * scale**2
                I_m[2, 2] = diaginertia[2] * scale**2
            else:
                fullinertia = inertial_attrib.get("fullinertia")
                assert fullinertia is not None
                fullinertia = np.fromstring(fullinertia, sep=" ", dtype=np.float32)
                I_m = np.zeros((3, 3))
                I_m[0, 0] = fullinertia[0] * scale**2
                I_m[1, 1] = fullinertia[1] * scale**2
                I_m[2, 2] = fullinertia[2] * scale**2
                I_m[0, 1] = fullinertia[3] * scale**2
                I_m[0, 2] = fullinertia[4] * scale**2
                I_m[1, 2] = fullinertia[5] * scale**2
                I_m[1, 0] = I_m[0, 1]
                I_m[2, 0] = I_m[0, 2]
                I_m[2, 1] = I_m[1, 2]

            rot = wp.quat_to_matrix(inertial_frame.q)
            rot_np = np.array(rot).reshape(3, 3)
            I_m = rot_np @ I_m @ rot_np.T
            I_m = wp.mat33(I_m)
            m = float(inertial_attrib.get("mass", "0"))
            builder.body_mass[link] = m
            builder.body_inv_mass[link] = 1.0 / m if m > 0.0 else 0.0
            builder.body_com[link] = com
            builder.body_inertia[link] = I_m
            if any(x for x in I_m):
                builder.body_inv_inertia[link] = wp.inverse(I_m)
            else:
                builder.body_inv_inertia[link] = I_m
        if m == 0.0 and ensure_nonstatic_links:
            # set the mass to something nonzero to ensure the body is dynamic
            m = static_link_mass
            # cube with side length 0.5
            I_m = wp.mat33(np.eye(3)) * m / 12.0 * (0.5 * scale) ** 2 * 2.0
            I_m += wp.mat33(builder.default_body_armature * np.eye(3))
            builder.body_mass[link] = m
            builder.body_inv_mass[link] = 1.0 / m
            builder.body_inertia[link] = I_m
            builder.body_inv_inertia[link] = wp.inverse(I_m)

        # -----------------
        # recurse

        for child in body.findall("body"):
            _childclass = body.get("childclass")
            if _childclass is None:
                _childclass = childclass
                _incoming_defaults = defaults
            else:
                _incoming_defaults = merge_attrib(defaults, class_defaults[_childclass])
            parse_body(child, link, _incoming_defaults, childclass=_childclass, parent_world_xform=world_xform)

    def parse_equality_constraints(equality):
        def parse_common_attributes(element):
            return {
                "name": element.attrib.get("name"),
                "active": element.attrib.get("active", "true").lower() == "true",
                "solref": element.attrib.get("solref"),
                "solimp": element.attrib.get("solimp"),
            }

        for connect in equality.findall("connect"):
            common = parse_common_attributes(connect)
            body1_name = connect.attrib.get("body1", "").replace("-", "_") if connect.attrib.get("body1") else None
            body2_name = (
                connect.attrib.get("body2", "worldbody").replace("-", "_") if connect.attrib.get("body2") else None
            )
            anchor = connect.attrib.get("anchor")

            site1 = connect.attrib.get("site1")

            if body1_name and anchor:
                if verbose:
                    print(f"Connect constraint: {body1_name} to {body2_name} at anchor {anchor}")

                anchor_vec = wp.vec3(*[float(x) * scale for x in anchor.split()]) if anchor else None

                body1_idx = builder.body_key.index(body1_name) if body1_name and body1_name in builder.body_key else -1
                body2_idx = builder.body_key.index(body2_name) if body2_name and body2_name in builder.body_key else -1

                builder.add_equality_constraint_connect(
                    body1=body1_idx,
                    body2=body2_idx,
                    anchor=anchor_vec,
                    key=common["name"],
                    enabled=common["active"],
                )

            if site1:  # Implement site-based connect after Newton supports sites
                print("Warning: MuJoCo sites are not yet supported in Newton.")

        for weld in equality.findall("weld"):
            common = parse_common_attributes(weld)
            body1_name = weld.attrib.get("body1", "").replace("-", "_") if weld.attrib.get("body1") else None
            body2_name = weld.attrib.get("body2", "worldbody").replace("-", "_") if weld.attrib.get("body2") else None
            anchor = weld.attrib.get("anchor", "0 0 0")
            relpose = weld.attrib.get("relpose", "0 1 0 0 0 0 0")
            torquescale = weld.attrib.get("torquescale")

            site1 = weld.attrib.get("site1")

            if body1_name:
                if verbose:
                    print(f"Weld constraint: {body1_name} to {body2_name}")

                anchor_vec = wp.vec3(*[float(x) * scale for x in anchor.split()])

                body1_idx = builder.body_key.index(body1_name) if body1_name and body1_name in builder.body_key else -1
                body2_idx = builder.body_key.index(body2_name) if body2_name and body2_name in builder.body_key else -1

                relpose_list = [float(x) for x in relpose.split()]
                relpose_transform = wp.transform(
                    wp.vec3(relpose_list[0], relpose_list[1], relpose_list[2]),
                    wp.quat(relpose_list[4], relpose_list[5], relpose_list[6], relpose_list[3]),
                )

                builder.add_equality_constraint_weld(
                    body1=body1_idx,
                    body2=body2_idx,
                    anchor=anchor_vec,
                    relpose=relpose_transform,
                    torquescale=torquescale,
                    key=common["name"],
                    enabled=common["active"],
                )

            if site1:  # Implement site-based weld after Newton supports sites
                print("Warning: MuJoCo sites are not yet supported in Newton.")

        for joint in equality.findall("joint"):
            common = parse_common_attributes(joint)
            joint1_name = joint.attrib.get("joint1")
            joint2_name = joint.attrib.get("joint2")
            polycoef = joint.attrib.get("polycoef", "0 1 0 0 0")

            if joint1_name:
                if verbose:
                    print(f"Joint constraint: {joint1_name} coupled to {joint2_name} with polycoef {polycoef}")

                joint1_idx = (
                    builder.joint_key.index(joint1_name) if joint1_name and joint1_name in builder.joint_key else -1
                )
                joint2_idx = (
                    builder.joint_key.index(joint2_name) if joint2_name and joint2_name in builder.joint_key else -1
                )

                builder.add_equality_constraint_joint(
                    joint1=joint1_idx,
                    joint2=joint2_idx,
                    polycoef=[float(x) for x in polycoef.split()],
                    key=common["name"],
                    enabled=common["active"],
                )

        # add support for types "tendon" and "flex" once Newton supports them

    # -----------------
    # start articulation

    visual_shapes = []
    start_shape_count = len(builder.shape_type)
    builder.add_articulation(key=root.attrib.get("model"))

    world = root.find("worldbody")
    world_class = get_class(world)
    world_defaults = merge_attrib(class_defaults["__all__"], class_defaults.get(world_class, {}))

    # -----------------
    # add bodies

    for body in world.findall("body"):
        parse_body(body, -1, world_defaults, parent_world_xform=xform)

    # -----------------
    # add static geoms

    parse_shapes(
        defaults=world_defaults,
        body_name="world",
        link=-1,
        geoms=world.findall("geom"),
        density=default_shape_density,
        incoming_xform=xform,
    )

    if parse_sites:
        _parse_sites_impl(
            defaults=world_defaults,
            body_name="world",
            link=-1,
            sites=world.findall("site"),
            incoming_xform=xform,
        )

    # -----------------
    # add equality constraints

    equality = root.find("equality")
    if equality is not None and not skip_equality_constraints:
        parse_equality_constraints(equality)

    # -----------------

    end_shape_count = len(builder.shape_type)

    for i in range(start_shape_count, end_shape_count):
        for j in visual_shapes:
            builder.shape_collision_filter_pairs.append((i, j))

    if not enable_self_collisions:
        for i in range(start_shape_count, end_shape_count):
            for j in range(i + 1, end_shape_count):
                builder.shape_collision_filter_pairs.append((i, j))

    if collapse_fixed_joints:
        builder.collapse_fixed_joints()
