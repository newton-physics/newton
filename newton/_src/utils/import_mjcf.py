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
from collections.abc import Callable
from typing import Any

import numpy as np
import warp as wp

from ..core import quat_between_axes, quat_from_euler
from ..core.types import Axis, AxisType, Sequence, Transform
from ..geometry import MESH_MAXHULLVERT, Mesh, ShapeFlags
from ..sim import JointType, ModelBuilder
from ..sim.model import ModelAttributeFrequency
from ..usd.schemas import solref_to_stiffness_damping
from .import_utils import is_xml_content, parse_custom_attributes, sanitize_xml_content


def _default_path_resolver(base_dir: str | None, file_path: str) -> str:
    """Default path resolver - joins base_dir with file_path.

    Args:
        base_dir: Base directory for resolving relative paths (None for XML string input)
        file_path: The 'file' attribute value to resolve

    Returns:
        Resolved absolute file path

    Raises:
        ValueError: If file_path is relative and base_dir is None
    """
    if os.path.isabs(file_path):
        return os.path.normpath(file_path)
    elif base_dir:
        return os.path.normpath(os.path.join(base_dir, file_path))
    else:
        raise ValueError(f"Cannot resolve relative path '{file_path}' without base directory")


def _load_and_expand_mjcf(
    source: str,
    path_resolver: Callable[[str | None, str], str] = _default_path_resolver,
    included_files: set[str] | None = None,
) -> tuple[ET.Element, str | None]:
    """Load MJCF source and recursively expand <include> elements.

    Args:
        source: File path or XML string
        path_resolver: Callback to resolve file paths. Takes (base_dir, file_path) and returns:
            - For <include> elements: either an absolute file path or XML content directly
            - For asset elements (mesh, texture, etc.): must return an absolute file path
            Default resolver joins paths and returns absolute file paths.
        included_files: Set of already-included file paths for cycle detection

    Returns:
        Tuple of (root element, base directory or None for XML string input)

    Raises:
        ValueError: If a circular include is detected
    """
    if included_files is None:
        included_files = set()

    # Load source
    if is_xml_content(source):
        base_dir = None  # No base directory for XML strings
        root = ET.fromstring(sanitize_xml_content(source))
    else:
        # Treat as file path
        base_dir = os.path.dirname(source) or "."
        root = ET.parse(source).getroot()

    # Find all (parent, include) pairs in a single pass
    include_pairs = [(parent, child) for parent in root.iter() for child in parent if child.tag == "include"]

    for parent, include in include_pairs:
        file_attr = include.get("file")
        if not file_attr:
            continue

        resolved = path_resolver(base_dir, file_attr)

        if not is_xml_content(resolved):
            # Cycle detection for file paths
            if resolved in included_files:
                raise ValueError(f"Circular include detected: {resolved}")
            included_files.add(resolved)

        # Recursive call - handles both file paths and XML content
        included_root, included_base_dir = _load_and_expand_mjcf(resolved, path_resolver, included_files)

        # Resolve all file attributes in included content to absolute paths
        # This ensures assets from included files are resolved relative to their source
        for elem in included_root.iter():
            file_attr = elem.get("file")
            if file_attr and not os.path.isabs(file_attr):
                elem.set("file", path_resolver(included_base_dir, file_attr))

        # Replace include element with children of included root
        idx = list(parent).index(include)
        parent.remove(include)
        for i, child in enumerate(included_root):
            parent.insert(idx + i, child)

    return root, base_dir


def parse_mjcf(
    builder: ModelBuilder,
    source: str,
    *,
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
    parse_mujoco_options: bool = True,
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
    convert_3d_hinge_to_ball_joints: bool = False,
    mesh_maxhullvert: int = MESH_MAXHULLVERT,
    path_resolver: Callable[[str | None, str], str] | None = None,
):
    """
    Parses MuJoCo XML (MJCF) file and adds the bodies and joints to the given ModelBuilder.
    MuJoCo-specific custom attributes are registered on the builder automatically.

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
        parse_mujoco_options (bool): Whether solver options from the MJCF `<option>` tag should be parsed. If False, solver options are not loaded and custom attributes retain their default values. Default is True.
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
        convert_3d_hinge_to_ball_joints (bool): If True, series of three hinge joints are converted to a single ball joint. Default is False.
        mesh_maxhullvert (int): Maximum vertices for convex hull approximation of meshes.
        path_resolver (Callable): Callback to resolve file paths. Takes (base_dir, file_path) and returns a resolved path. For <include> elements, can return either a file path or XML content directly. For asset elements (mesh, texture, etc.), must return an absolute file path. The default resolver joins paths and returns absolute file paths.
    """
    if xform is None:
        xform = wp.transform_identity()
    else:
        xform = wp.transform(*xform)

    if path_resolver is None:
        path_resolver = _default_path_resolver

    # Convert Path objects to string
    source = os.fspath(source) if hasattr(source, "__fspath__") else source

    root, base_dir = _load_and_expand_mjcf(source, path_resolver)
    mjcf_dirname = base_dir or "."  # Backward compatible fallback for mesh paths

    use_degrees = True  # angles are in degrees by default
    euler_seq = [0, 1, 2]  # XYZ by default

    # load joint defaults
    default_joint_limit_lower = builder.default_joint_cfg.limit_lower
    default_joint_limit_upper = builder.default_joint_cfg.limit_upper
    default_joint_target_ke = builder.default_joint_cfg.target_ke
    default_joint_target_kd = builder.default_joint_cfg.target_kd
    default_joint_armature = builder.default_joint_cfg.armature
    default_joint_effort_limit = builder.default_joint_cfg.effort_limit

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
    builder_custom_attr_eq: list[ModelBuilder.CustomAttribute] = builder.get_custom_attributes_by_frequency(
        [ModelAttributeFrequency.EQUALITY_CONSTRAINT]
    )

    compiler = root.find("compiler")
    if compiler is not None:
        use_degrees = compiler.attrib.get("angle", "degree").lower() == "degree"
        euler_seq = ["xyz".index(c) for c in compiler.attrib.get("eulerseq", "xyz").lower()]
        mesh_dir = compiler.attrib.get("meshdir", ".")
    else:
        mesh_dir = "."

    # Parse MJCF option tag for ONCE and WORLD frequency custom attributes (solver options)
    # WORLD frequency attributes use index 0 here; they get remapped during add_world()
    if parse_mujoco_options:
        builder_custom_attr_option: list[ModelBuilder.CustomAttribute] = builder.get_custom_attributes_by_frequency(
            [ModelAttributeFrequency.ONCE, ModelAttributeFrequency.WORLD]
        )
        option_elem = root.find("option")
        if option_elem is not None and builder_custom_attr_option:
            option_attrs = parse_custom_attributes(option_elem.attrib, builder_custom_attr_option, "mjcf")
            for key, value in option_attrs.items():
                if key in builder.custom_attributes:
                    builder.custom_attributes[key].values[0] = value

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

    def resolve_defaults(class_name):
        if class_name in class_children:
            for child_name in class_children[class_name]:
                if class_name in class_defaults and child_name in class_defaults:
                    class_defaults[child_name] = merge_attrib(class_defaults[class_name], class_defaults[child_name])
                resolve_defaults(child_name)

    resolve_defaults("__all__")

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
            return wp.types.vector(len(default), wp.float32)(out[0], out[0], out[0])

        return wp.types.vector(length, wp.float32)(out)

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
            if incoming_xform is not None:
                tf = incoming_xform * tf

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

                    # Apply incoming_xform to fromto coordinates
                    if incoming_xform is not None:
                        start = wp.transform_point(incoming_xform, start)
                        end = wp.transform_point(incoming_xform, end)

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
                # Use tf (which has incoming_xform applied) for plane normal/distance
                normal = wp.quat_rotate(tf.q, wp.vec3(0.0, 0.0, 1.0))
                p = wp.dot(tf.p, normal)
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

            # Parse site size matching MuJoCo behavior:
            # - Default is [0.005, 0.005, 0.005]
            # - Partial values fill remaining with defaults (NOT replicating first value)
            # - size="0.001" → [0.001, 0.005, 0.005] (matches MuJoCo)
            # Note: This differs from parse_vec which would replicate single values
            site_size = np.array([0.005, 0.005, 0.005], dtype=np.float32)
            if "size" in site_attrib:
                size_values = np.fromstring(site_attrib["size"], sep=" ", dtype=np.float32)
                for i, val in enumerate(size_values):
                    if i < 3:
                        site_size[i] = val
            site_size = wp.vec3(site_size * scale)

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

    def get_frame_xform(frame_element, incoming_xform: wp.transform) -> wp.transform:
        """Compute composed transform for a frame element."""
        frame_pos = parse_vec(frame_element.attrib, "pos", (0.0, 0.0, 0.0)) * scale
        frame_rot = parse_orientation(frame_element.attrib)
        return incoming_xform * wp.transform(frame_pos, frame_rot)

    def _process_body_geoms(
        geoms,
        defaults: dict,
        body_name: str,
        link: int,
        incoming_xform: wp.transform | None = None,
    ) -> list:
        """Process geoms for a body, partitioning into visuals and colliders.

        This helper applies the same filtering/partitioning logic for geoms whether
        they appear directly in a <body> or inside a <frame> within a body.

        Args:
            geoms: Iterable of geom XML elements to process.
            defaults: The current defaults dictionary.
            body_name: Name of the parent body (for naming).
            link: The body index.
            incoming_xform: Optional transform to apply to geoms.

        Returns:
            List of visual shape indices (if parse_visuals is True).
        """
        visuals = []
        colliders = []

        for geo_count, geom in enumerate(geoms):
            geom_defaults = defaults
            geom_class = None
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

            if geom_class is not None:
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

        visual_shape_indices = []

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
                incoming_xform=incoming_xform,
            )
            visual_shape_indices.extend(s)

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
            incoming_xform=incoming_xform,
        )

        return visual_shape_indices

    def process_frames(
        frames,
        parent_body: int,
        defaults: dict,
        childclass: str | None,
        world_xform: wp.transform,
        body_relative_xform: wp.transform | None = None,
    ):
        """Process frame elements, composing transforms with children.

        Frames are pure coordinate transformations that can wrap bodies, geoms, sites, and nested frames.

        Args:
            frames: Iterable of frame XML elements to process.
            parent_body: The parent body index (-1 for world).
            defaults: The current defaults dictionary.
            childclass: The current childclass for body inheritance.
            world_xform: World transform for positioning child bodies.
            body_relative_xform: Body-relative transform for geoms/sites. If None, uses world_xform
                (appropriate for static geoms at worldbody level).
        """
        # Stack entries: (frame, world_xform, body_relative_xform, frame_defaults, frame_childclass)
        # For worldbody frames, body_relative equals world (static geoms use world coords)
        if body_relative_xform is None:
            frame_stack = [(f, world_xform, world_xform, defaults, childclass) for f in frames]
        else:
            frame_stack = [(f, world_xform, body_relative_xform, defaults, childclass) for f in frames]

        while frame_stack:
            frame, frame_world, frame_body_rel, frame_defaults, frame_childclass = frame_stack.pop()
            frame_local = get_frame_xform(frame, wp.transform_identity())
            composed_world = frame_world * frame_local
            composed_body_rel = frame_body_rel * frame_local

            # Resolve childclass for this frame's children
            _childclass = frame.get("childclass") or frame_childclass

            # Compute merged defaults for this frame's children
            if _childclass is None:
                _defaults = frame_defaults
            else:
                _defaults = merge_attrib(frame_defaults, class_defaults.get(_childclass, {}))

            # Process child bodies (need world transform)
            for child_body in frame.findall("body"):
                parse_body(child_body, parent_body, _defaults, childclass=_childclass, incoming_xform=composed_world)

            # Process child geoms (need body-relative transform)
            # Use the same visual/collider partitioning logic as parse_body
            child_geoms = frame.findall("geom")
            if child_geoms:
                body_name = "world" if parent_body == -1 else builder.body_key[parent_body]
                frame_visual_shapes = _process_body_geoms(
                    child_geoms,
                    _defaults,
                    body_name,
                    parent_body,
                    incoming_xform=composed_body_rel,
                )
                visual_shapes.extend(frame_visual_shapes)

            # Process child sites (need body-relative transform)
            if parse_sites:
                child_sites = frame.findall("site")
                if child_sites:
                    body_name = "world" if parent_body == -1 else builder.body_key[parent_body]
                    _parse_sites_impl(_defaults, body_name, parent_body, child_sites, incoming_xform=composed_body_rel)

            # Add nested frames to stack with current defaults and childclass (in reverse to maintain order)
            frame_stack.extend(
                (f, composed_world, composed_body_rel, _defaults, _childclass) for f in reversed(frame.findall("frame"))
            )

    def parse_body(
        body,
        parent,
        incoming_defaults: dict,
        childclass: str | None = None,
        incoming_xform: Transform | None = None,
    ):
        """Parse a body element from MJCF.

        Args:
            body: The XML body element.
            parent: Parent body index (-1 for world).
            incoming_defaults: Default attributes dictionary.
            childclass: Child class name for inheritance.
            incoming_xform: Accumulated transform from parent (may include frame offsets).
                If None, uses the import root xform.
        """
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

        # Compose with incoming transform (or import root xform if none)
        world_xform = (incoming_xform or xform) * local_xform

        # For joint positioning, compute body position relative to the actual parent body
        if parent >= 0:
            # Look up parent body's world transform and compute relative position
            parent_body_xform = builder.body_q[parent]
            relative_xform = wp.transform_inverse(parent_body_xform) * world_xform
            body_pos_for_joints = relative_xform.p
            body_ori_for_joints = relative_xform.q
        else:
            # World parent: use the composed world_xform (includes frame/import root transforms)
            body_pos_for_joints = world_xform.p
            body_ori_for_joints = world_xform.q

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

                effort_limit = default_joint_effort_limit
                if "actuatorfrcrange" in joint_attrib:
                    actuatorfrcrange = parse_vec(joint_attrib, "actuatorfrcrange", None)
                    if actuatorfrcrange is not None and len(actuatorfrcrange) == 2:
                        actuatorfrclimited = joint_attrib.get("actuatorfrclimited", "auto").lower()
                        if actuatorfrclimited in ("true", "auto"):
                            effort_limit = max(abs(actuatorfrcrange[0]), abs(actuatorfrcrange[1]))
                        elif verbose:
                            print(
                                f"Warning: Joint '{joint_attrib.get('name', 'unnamed')}' has actuatorfrcrange "
                                f"but actuatorfrclimited='{actuatorfrclimited}'. Force clamping will be disabled."
                            )

                ax = ModelBuilder.JointDofConfig(
                    axis=axis_vec,
                    limit_lower=limit_lower,
                    limit_upper=limit_upper,
                    limit_ke=limit_ke,
                    limit_kd=limit_kd,
                    target_ke=default_joint_target_ke,
                    target_kd=default_joint_target_kd,
                    armature=joint_armature[-1],
                    effort_limit=effort_limit,
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
        link = builder.add_link(
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
                elif convert_3d_hinge_to_ball_joints and len(angular_axes) == 3:
                    joint_type = JointType.BALL
            elif len(linear_axes) == 1 and len(angular_axes) == 0:
                joint_type = JointType.PRISMATIC

        if joint_type == JointType.FREE and parent == -1 and (base_joint is not None or floating is not None):
            joint_pos = joint_pos[0] if len(joint_pos) > 0 else wp.vec3(0.0, 0.0, 0.0)
            # Rotate joint_pos by body orientation before adding to body position
            rotated_joint_pos = wp.quat_rotate(body_ori_for_joints, joint_pos)
            _xform = wp.transform(body_pos_for_joints + rotated_joint_pos, body_ori_for_joints)

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
                    joint_indices.append(
                        builder.add_joint_d6(
                            linear_axes=[ModelBuilder.JointDofConfig(axis=axes[a]) for a in linear_axes],
                            angular_axes=[ModelBuilder.JointDofConfig(axis=axes[a]) for a in angular_axes],
                            parent_xform=base_parent_xform,
                            child_xform=base_child_xform,
                            parent=-1,
                            child=link,
                            key="base_joint",
                        )
                    )
                elif isinstance(base_joint, dict):
                    base_joint["parent"] = -1
                    base_joint["child"] = link
                    base_joint["parent_xform"] = base_parent_xform
                    base_joint["child_xform"] = base_child_xform
                    base_joint["key"] = "base_joint"
                    joint_indices.append(builder.add_joint(**base_joint))
                else:
                    raise ValueError(
                        "base_joint must be a comma-separated string of joint axes or a dict with joint parameters"
                    )
            elif floating is not None and floating:
                joint_indices.append(builder.add_joint_free(link, key="floating_base"))
            else:
                joint_indices.append(builder.add_joint_fixed(-1, link, parent_xform=world_xform, key="fixed_base"))

        else:
            joint_pos = joint_pos[0] if len(joint_pos) > 0 else wp.vec3(0.0, 0.0, 0.0)
            if len(joint_name) == 0:
                joint_name = [f"{body_name}_joint"]
            if joint_type == JointType.FREE:
                assert parent == -1, "Free joints must have the world body as parent"
                joint_indices.append(
                    builder.add_joint_free(
                        link,
                        key="_".join(joint_name),
                        custom_attributes=joint_custom_attributes,
                    )
                )
            else:
                # When parent is world (-1), use world_xform to respect the xform argument
                if parent == -1:
                    parent_xform_for_joint = world_xform * wp.transform(joint_pos, wp.quat_identity())
                else:
                    # Rotate joint_pos by body orientation before adding to body position
                    rotated_joint_pos = wp.quat_rotate(body_ori_for_joints, joint_pos)
                    parent_xform_for_joint = wp.transform(body_pos_for_joints + rotated_joint_pos, body_ori_for_joints)

                joint_indices.append(
                    builder.add_joint(
                        joint_type,
                        parent=parent,
                        child=link,
                        linear_axes=linear_axes,
                        angular_axes=angular_axes,
                        key="_".join(joint_name),
                        parent_xform=parent_xform_for_joint,
                        child_xform=wp.transform(joint_pos, wp.quat_identity()),
                        custom_attributes=joint_custom_attributes | dof_custom_attributes,
                    )
                )

        # -----------------
        # add shapes (using shared helper for visual/collider partitioning)

        geoms = body.findall("geom")
        body_visual_shapes = _process_body_geoms(geoms, defaults, body_name, link)
        visual_shapes.extend(body_visual_shapes)

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
            parse_body(child, link, _incoming_defaults, childclass=_childclass, incoming_xform=world_xform)

        # Process frame elements within this body
        # Use body's childclass if declared, otherwise inherit from parent
        frame_childclass = body.get("childclass") or childclass
        frame_defaults = (
            merge_attrib(defaults, class_defaults.get(frame_childclass, {})) if frame_childclass else defaults
        )
        process_frames(
            body.findall("frame"),
            parent_body=link,
            defaults=frame_defaults,
            childclass=frame_childclass,
            world_xform=world_xform,
            body_relative_xform=wp.transform_identity(),  # Geoms/sites need body-relative coords
        )

    def parse_equality_constraints(equality):
        def parse_common_attributes(element):
            return {
                "name": element.attrib.get("name"),
                "active": element.attrib.get("active", "true").lower() == "true",
            }

        def get_site_body_and_anchor(site_name: str) -> tuple[int, wp.vec3] | None:
            """Look up a site by name and return its body index and position (anchor).

            Returns:
                Tuple of (body_idx, anchor_position) or None if site not found or not a site.
            """
            if site_name not in builder.shape_key:
                if verbose:
                    print(f"Warning: Site '{site_name}' not found")
                return None
            site_idx = builder.shape_key.index(site_name)
            if not (builder.shape_flags[site_idx] & ShapeFlags.SITE):
                if verbose:
                    print(f"Warning: Shape '{site_name}' is not a site")
                return None
            body_idx = builder.shape_body[site_idx]
            site_xform = builder.shape_transform[site_idx]
            anchor = wp.vec3(site_xform[0], site_xform[1], site_xform[2])
            return (body_idx, anchor)

        for connect in equality.findall("connect"):
            common = parse_common_attributes(connect)
            custom_attrs = parse_custom_attributes(connect.attrib, builder_custom_attr_eq, parsing_mode="mjcf")
            body1_name = connect.attrib.get("body1", "").replace("-", "_") if connect.attrib.get("body1") else None
            body2_name = (
                connect.attrib.get("body2", "worldbody").replace("-", "_") if connect.attrib.get("body2") else None
            )
            anchor = connect.attrib.get("anchor")
            site1 = connect.attrib.get("site1")
            site2 = connect.attrib.get("site2")

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
                    custom_attributes=custom_attrs,
                )
            elif site1:
                if site2:
                    # Site-based connect: both site1 and site2 must be specified
                    site1_info = get_site_body_and_anchor(site1)
                    site2_info = get_site_body_and_anchor(site2)
                    if site1_info is None or site2_info is None:
                        if verbose:
                            print(f"Warning: Connect constraint '{common['name']}' failed.")
                        continue
                    body1_idx, anchor_vec = site1_info
                    body2_idx, _ = site2_info
                    if verbose:
                        print(
                            f"Connect constraint (site-based): site '{site1}' on body {body1_idx} to body {body2_idx}"
                        )
                    builder.add_equality_constraint_connect(
                        body1=body1_idx,
                        body2=body2_idx,
                        anchor=anchor_vec,
                        key=common["name"],
                        enabled=common["active"],
                        custom_attributes=custom_attrs,
                    )
                else:
                    if verbose:
                        print(
                            f"Warning: Connect constraint '{common['name']}' has site1 but no site2. "
                            "When using sites, both site1 and site2 must be specified. Skipping."
                        )

        for weld in equality.findall("weld"):
            common = parse_common_attributes(weld)
            custom_attrs = parse_custom_attributes(weld.attrib, builder_custom_attr_eq, parsing_mode="mjcf")
            body1_name = weld.attrib.get("body1", "").replace("-", "_") if weld.attrib.get("body1") else None
            body2_name = weld.attrib.get("body2", "worldbody").replace("-", "_") if weld.attrib.get("body2") else None
            anchor = weld.attrib.get("anchor", "0 0 0")
            relpose = weld.attrib.get("relpose", "0 1 0 0 0 0 0")
            torquescale = weld.attrib.get("torquescale")
            site1 = weld.attrib.get("site1")
            site2 = weld.attrib.get("site2")

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
                    custom_attributes=custom_attrs,
                )
            elif site1:
                if site2:
                    # Site-based weld: both site1 and site2 must be specified
                    site1_info = get_site_body_and_anchor(site1)
                    site2_info = get_site_body_and_anchor(site2)
                    if site1_info is None or site2_info is None:
                        if verbose:
                            print(f"Warning: Weld constraint '{common['name']}' failed.")
                        continue
                    body1_idx, _ = site1_info
                    body2_idx, anchor_vec = site2_info
                    relpose_list = [float(x) for x in relpose.split()]
                    relpose_transform = wp.transform(
                        wp.vec3(relpose_list[0], relpose_list[1], relpose_list[2]),
                        wp.quat(relpose_list[4], relpose_list[5], relpose_list[6], relpose_list[3]),
                    )
                    if verbose:
                        print(f"Weld constraint (site-based): body {body1_idx} to body {body2_idx}")
                    builder.add_equality_constraint_weld(
                        body1=body1_idx,
                        body2=body2_idx,
                        anchor=anchor_vec,
                        relpose=relpose_transform,
                        torquescale=torquescale,
                        key=common["name"],
                        enabled=common["active"],
                        custom_attributes=custom_attrs,
                    )
                else:
                    if verbose:
                        print(
                            f"Warning: Weld constraint '{common['name']}' has site1 but no site2. "
                            "When using sites, both site1 and site2 must be specified. Skipping."
                        )

        for joint in equality.findall("joint"):
            common = parse_common_attributes(joint)
            custom_attrs = parse_custom_attributes(joint.attrib, builder_custom_attr_eq, parsing_mode="mjcf")
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
                    custom_attributes=custom_attrs,
                )

        # add support for types "tendon" and "flex" once Newton supports them

    # -----------------
    # start articulation

    visual_shapes = []
    start_shape_count = len(builder.shape_type)
    joint_indices = []  # Collect joint indices as we create them

    world = root.find("worldbody")
    world_class = get_class(world)
    world_defaults = merge_attrib(class_defaults["__all__"], class_defaults.get(world_class, {}))

    # -----------------
    # add bodies

    for body in world.findall("body"):
        parse_body(body, -1, world_defaults, incoming_xform=xform)

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
    # process frame elements at worldbody level

    process_frames(
        world.findall("frame"),
        parent_body=-1,
        defaults=world_defaults,
        childclass=None,
        world_xform=xform,
        body_relative_xform=None,  # Static geoms use world coords
    )

    # -----------------
    # add equality constraints

    equality = root.find("equality")
    if equality is not None and not skip_equality_constraints:
        parse_equality_constraints(equality)

    # -----------------
    # parse contact pairs

    # Get custom attributes with custom frequency for pair parsing
    # Exclude pair_geom1/pair_geom2/pair_world as they're handled specially (geom name lookup, world assignment)
    builder_custom_attr_pair: list[ModelBuilder.CustomAttribute] = [
        attr
        for attr in builder.custom_attributes.values()
        if isinstance(attr.frequency_key, str)
        and attr.name.startswith("pair_")
        and attr.name not in ("pair_geom1", "pair_geom2", "pair_world")
    ]

    # Only parse contact pairs if custom attributes are registered
    has_pair_attrs = "mujoco:pair_geom1" in builder.custom_attributes
    contact = root.find("contact")
    if contact is not None and has_pair_attrs:
        # Parse <pair> elements - explicit contact pairs with custom properties
        for pair in contact.findall("pair"):
            geom1_name = pair.attrib.get("geom1")
            geom2_name = pair.attrib.get("geom2")

            if not geom1_name or not geom2_name:
                if verbose:
                    print("Warning: <pair> element missing geom1 or geom2 attribute, skipping")
                continue

            # Look up shape indices by geom name
            try:
                geom1_idx = builder.shape_key.index(geom1_name)
            except ValueError:
                if verbose:
                    print(f"Warning: <pair> references unknown geom '{geom1_name}', skipping")
                continue

            try:
                geom2_idx = builder.shape_key.index(geom2_name)
            except ValueError:
                if verbose:
                    print(f"Warning: <pair> references unknown geom '{geom2_name}', skipping")
                continue

            # Parse attributes using the standard custom attribute parsing
            pair_attrs = parse_custom_attributes(pair.attrib, builder_custom_attr_pair, parsing_mode="mjcf")

            # Build values dict for all pair attributes
            pair_values: dict[str, Any] = {
                "mujoco:pair_world": builder.current_world,
                "mujoco:pair_geom1": geom1_idx,
                "mujoco:pair_geom2": geom2_idx,
            }
            # Add remaining attributes with parsed values or defaults
            for attr in builder_custom_attr_pair:
                pair_values[attr.key] = pair_attrs.get(attr.key, attr.default)

            builder.add_custom_values(**pair_values)

            if verbose:
                print(f"Parsed contact pair: {geom1_name} ({geom1_idx}) <-> {geom2_name} ({geom2_idx})")

    # -----------------
    # Parse all fixed tendons in a single tendon section.

    # Get variable-length custom attributes for tendon parsing (frequency="tendon")
    # Exclude tendon_world, tendon_joint_adr, tendon_joint_num as they're handled specially
    builder_custom_attr_tendon: list[ModelBuilder.CustomAttribute] = [
        attr
        for attr in builder.custom_attributes.values()
        if isinstance(attr.frequency_key, str)
        and attr.name.startswith("tendon_")
        and attr.name not in ("tendon_world", "tendon_joint_adr", "tendon_joint_num", "tendon_joint", "tendon_coef")
    ]

    def parse_tendons(tendon_section):
        for fixed in tendon_section.findall("fixed"):
            tendon_name = fixed.attrib.get("name", "")

            # Parse joint elements within this fixed tendon
            joint_entries = []
            for joint_elem in fixed.findall("joint"):
                joint_name = joint_elem.attrib.get("joint")
                coef_str = joint_elem.attrib.get("coef", "1.0")

                if not joint_name:
                    if verbose:
                        print(f"Warning: <joint> in tendon '{tendon_name}' missing joint attribute, skipping")
                    continue

                # Look up joint index by name
                try:
                    joint_idx = builder.joint_key.index(joint_name)
                except ValueError:
                    if verbose:
                        print(
                            f"Warning: Tendon '{tendon_name}' references unknown joint '{joint_name}', skipping joint"
                        )
                    continue

                coef = float(coef_str)
                joint_entries.append((joint_idx, coef))

            if not joint_entries:
                if verbose:
                    print(f"Warning: Fixed tendon '{tendon_name}' has no valid joint elements, skipping")
                continue

            # Parse tendon-level attributes using the standard custom attribute parsing
            tendon_attrs = parse_custom_attributes(fixed.attrib, builder_custom_attr_tendon, parsing_mode="mjcf")

            # Determine wrap array start index
            tendon_joint_attr = builder.custom_attributes.get("mujoco:tendon_joint")
            joint_start = len(tendon_joint_attr.values) if tendon_joint_attr and tendon_joint_attr.values else 0

            # Add joints to the joint arrays
            for joint_idx, coef in joint_entries:
                builder.add_custom_values(
                    **{
                        "mujoco:tendon_joint": joint_idx,
                        "mujoco:tendon_coef": coef,
                    }
                )

            # Build values dict for tendon-level attributes
            tendon_values: dict[str, Any] = {
                "mujoco:tendon_world": builder.current_world,
                "mujoco:tendon_joint_adr": joint_start,
                "mujoco:tendon_joint_num": len(joint_entries),
            }
            # Add remaining attributes with parsed values or defaults
            for attr in builder_custom_attr_tendon:
                tendon_values[attr.key] = tendon_attrs.get(attr.key, attr.default)

            builder.add_custom_values(**tendon_values)

            if verbose:
                joint_names_str = ", ".join(f"{builder.joint_key[j]}*{c}" for j, c in joint_entries)
                print(f"Parsed fixed tendon: {tendon_name} ({joint_names_str})")

    # -----------------
    # parse actuators

    def parse_actuators(actuator_section):
        """Parse actuators and set target_ke/target_kd for joints."""
        for position_actuator in actuator_section.findall("position"):
            joint_name = position_actuator.attrib.get("joint")
            if not joint_name:
                continue

            if joint_name not in builder.joint_key:
                if verbose:
                    print(f"Warning: Actuator references unknown joint '{joint_name}'")
                continue

            joint_idx = builder.joint_key.index(joint_name)
            qd_start = builder.joint_qd_start[joint_idx]
            lin_dofs, ang_dofs = builder.joint_dof_dim[joint_idx]
            total_dofs = lin_dofs + ang_dofs

            kp = parse_float(position_actuator.attrib, "kp", 0.0)
            kv = parse_float(position_actuator.attrib, "kv", 0.0)

            for i in range(total_dofs):
                dof_idx = qd_start + i
                builder.joint_target_ke[dof_idx] = kp
                builder.joint_target_kd[dof_idx] = kv

            if verbose:
                print(f"Position actuator on joint '{joint_name}': kp={kp}, kv={kv}")

        for velocity_actuator in actuator_section.findall("velocity"):
            joint_name = velocity_actuator.attrib.get("joint")
            if not joint_name:
                continue

            if joint_name not in builder.joint_key:
                if verbose:
                    print(f"Warning: Actuator references unknown joint '{joint_name}'")
                continue

            joint_idx = builder.joint_key.index(joint_name)
            qd_start = builder.joint_qd_start[joint_idx]
            lin_dofs, ang_dofs = builder.joint_dof_dim[joint_idx]
            total_dofs = lin_dofs + ang_dofs
            kv = parse_float(velocity_actuator.attrib, "kv", 0.0)
            for i in range(total_dofs):
                dof_idx = qd_start + i
                builder.joint_target_kd[dof_idx] = kv

            if verbose:
                print(f"Velocity actuator on joint '{joint_name}': kv={kv}")

    # Only parse tendons if custom tendon attributes are registered
    has_tendon_attrs = "mujoco:tendon_world" in builder.custom_attributes
    if has_tendon_attrs:
        # Find all sections marked <tendon></tendon>
        tendon_sections = root.findall(".//tendon")
        for tendon_section in tendon_sections:
            parse_tendons(tendon_section)

    actuator_section = root.find("actuator")
    if actuator_section is not None:
        parse_actuators(actuator_section)

    # -----------------

    end_shape_count = len(builder.shape_type)

    for i in range(start_shape_count, end_shape_count):
        for j in visual_shapes:
            builder.shape_collision_filter_pairs.append((i, j))

    if not enable_self_collisions:
        for i in range(start_shape_count, end_shape_count):
            for j in range(i + 1, end_shape_count):
                builder.shape_collision_filter_pairs.append((i, j))

    # Create articulation from all collected joints
    if joint_indices:
        articulation_key = root.attrib.get("model")
        builder.add_articulation(joints=joint_indices, key=articulation_key)

    if collapse_fixed_joints:
        builder.collapse_fixed_joints()
