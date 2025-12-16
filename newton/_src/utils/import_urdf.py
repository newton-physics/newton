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

import os
import tempfile
import warnings
import xml.etree.ElementTree as ET
from typing import Literal
from urllib.parse import unquote, urlsplit

import numpy as np
import warp as wp

from ..core import Axis, AxisType, quat_between_axes
from ..core.types import Transform
from ..geometry import MESH_MAXHULLVERT, Mesh
from ..sim import ModelBuilder
from ..sim.model import ModelAttributeFrequency
from .import_utils import parse_custom_attributes, sanitize_xml_content
from .topology import topological_sort


def _download_file(dst, url: str) -> None:
    import requests  # noqa: PLC0415

    with requests.get(url, stream=True, timeout=10) as response:
        response.raise_for_status()
        for chunk in response.iter_content(chunk_size=8192):
            dst.write(chunk)


def download_asset_tmpfile(url: str):
    """Download a file into a NamedTemporaryFile.
    A closed NamedTemporaryFile is returned. It must be deleted by the caller."""
    urlpath = unquote(urlsplit(url).path)
    file_od = tempfile.NamedTemporaryFile("wb", suffix=os.path.splitext(urlpath)[1], delete=False)
    _download_file(file_od, url)
    file_od.close()

    return file_od


def parse_urdf(
    builder: ModelBuilder,
    source: str,
    xform: Transform | None = None,
    floating: bool = False,
    base_joint: dict | str | None = None,
    scale: float = 1.0,
    hide_visuals: bool = False,
    parse_visuals_as_colliders: bool = False,
    up_axis: AxisType = Axis.Z,
    force_show_colliders: bool = False,
    enable_self_collisions: bool = True,
    ignore_inertial_definitions: bool = True,
    ensure_nonstatic_links: bool = True,
    static_link_mass: float = 1e-2,
    joint_ordering: Literal["bfs", "dfs"] | None = "dfs",
    bodies_follow_joint_ordering: bool = True,
    collapse_fixed_joints: bool = False,
    mesh_maxhullvert: int = MESH_MAXHULLVERT,
):
    """
    Parses a URDF file and adds the bodies and joints to the given ModelBuilder.

    Args:
        builder (ModelBuilder): The :class:`ModelBuilder` to add the bodies and joints to.
        source (str): The filename of the URDF file to parse, or the URDF XML string content.
        xform (Transform): The transform to apply to the root body. If None, the transform is set to identity.
        floating (bool): If True, the root body is a free joint. If False, the root body is connected via a fixed joint to the world, unless a `base_joint` is defined.
        base_joint (Union[str, dict]): The joint by which the root body is connected to the world. This can be either a string defining the joint axes of a D6 joint with comma-separated positional and angular axis names (e.g. "px,py,rz" for a D6 joint with linear axes in x, y and an angular axis in z) or a dict with joint parameters (see :meth:`ModelBuilder.add_joint`).
        scale (float): The scaling factor to apply to the imported mechanism.
        hide_visuals (bool): If True, hide visual shapes.
        parse_visuals_as_colliders (bool): If True, the geometry defined under the `<visual>` tags is used for collision handling instead of the `<collision>` geometries.
        up_axis (AxisType): The up axis of the URDF. This is used to transform the URDF to the builder's up axis. It also determines the up axis of capsules and cylinders in the URDF. The default is Z.
        force_show_colliders (bool): If True, the collision shapes are always shown, even if there are visual shapes.
        enable_self_collisions (bool): If True, self-collisions are enabled.
        ignore_inertial_definitions (bool): If True, the inertial parameters defined in the URDF are ignored and the inertia is calculated from the shape geometry.
        ensure_nonstatic_links (bool): If True, links with zero mass are given a small mass (see `static_link_mass`) to ensure they are dynamic.
        static_link_mass (float): The mass to assign to links with zero mass (if `ensure_nonstatic_links` is set to True).
        joint_ordering (str): The ordering of the joints in the simulation. Can be either "bfs" or "dfs" for breadth-first or depth-first search, or ``None`` to keep joints in the order in which they appear in the URDF. Default is "dfs".
        bodies_follow_joint_ordering (bool): If True, the bodies are added to the builder in the same order as the joints (parent then child body). Otherwise, bodies are added in the order they appear in the URDF. Default is True.
        collapse_fixed_joints (bool): If True, fixed joints are removed and the respective bodies are merged.
        mesh_maxhullvert (int): Maximum vertices for convex hull approximation of meshes.
    """
    axis_xform = wp.transform(wp.vec3(0.0), quat_between_axes(up_axis, builder.up_axis))
    if xform is None:
        xform = axis_xform
    else:
        xform = wp.transform(*xform) * axis_xform

    if os.path.isfile(source):
        file = ET.parse(source)
        urdf_root = file.getroot()
    else:
        xml_content = sanitize_xml_content(source)
        urdf_root = ET.fromstring(xml_content)

    # load joint defaults
    default_joint_limit_lower = builder.default_joint_cfg.limit_lower
    default_joint_limit_upper = builder.default_joint_cfg.limit_upper
    default_joint_damping = builder.default_joint_cfg.target_kd

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
    builder_custom_attr_articulation: list[ModelBuilder.CustomAttribute] = builder.get_custom_attributes_by_frequency(
        [ModelAttributeFrequency.ARTICULATION]
    )

    def parse_transform(element):
        if element is None or element.find("origin") is None:
            return wp.transform()
        origin = element.find("origin")
        xyz = origin.get("xyz") or "0 0 0"
        rpy = origin.get("rpy") or "0 0 0"
        xyz = [float(x) * scale for x in xyz.split()]
        rpy = [float(x) for x in rpy.split()]
        return wp.transform(xyz, wp.quat_rpy(*rpy))

    def parse_shapes(link: int, geoms, density, incoming_xform=None, visible=True, just_visual=False):
        shape_cfg = builder.default_shape_cfg.copy()
        shape_cfg.density = density
        shape_cfg.is_visible = visible
        shape_cfg.has_shape_collision = not just_visual
        shape_cfg.has_particle_collision = not just_visual
        shape_kwargs = {
            "body": link,
            "cfg": shape_cfg,
            "custom_attributes": {},
        }
        shapes = []
        # add geometry
        for geom_group in geoms:
            geo = geom_group.find("geometry")
            if geo is None:
                continue
            custom_attributes = parse_custom_attributes(geo.attrib, builder_custom_attr_shape, parsing_mode="urdf")
            shape_kwargs["custom_attributes"] = custom_attributes

            tf = parse_transform(geom_group)
            if incoming_xform is not None:
                tf = incoming_xform * tf

            for box in geo.findall("box"):
                size = box.get("size") or "1 1 1"
                size = [float(x) for x in size.split()]
                s = builder.add_shape_box(
                    xform=tf,
                    hx=size[0] * 0.5 * scale,
                    hy=size[1] * 0.5 * scale,
                    hz=size[2] * 0.5 * scale,
                    **shape_kwargs,
                )
                shapes.append(s)

            for sphere in geo.findall("sphere"):
                s = builder.add_shape_sphere(
                    xform=tf,
                    radius=float(sphere.get("radius") or "1") * scale,
                    **shape_kwargs,
                )
                shapes.append(s)

            for cylinder in geo.findall("cylinder"):
                # Apply axis rotation to transform
                xform = wp.transform(tf.p, tf.q * quat_between_axes(Axis.Z, up_axis))
                s = builder.add_shape_cylinder(
                    xform=xform,
                    radius=float(cylinder.get("radius") or "1") * scale,
                    half_height=float(cylinder.get("length") or "1") * 0.5 * scale,
                    **shape_kwargs,
                )
                shapes.append(s)

            for capsule in geo.findall("capsule"):
                # Apply axis rotation to transform
                xform = wp.transform(tf.p, tf.q * quat_between_axes(Axis.Z, up_axis))
                s = builder.add_shape_capsule(
                    xform=xform,
                    radius=float(capsule.get("radius") or "1") * scale,
                    half_height=float(capsule.get("height") or "1") * 0.5 * scale,
                    **shape_kwargs,
                )
                shapes.append(s)

            for mesh in geo.findall("mesh"):
                file_tmp = None
                filename = mesh.get("filename")
                if filename is None:
                    continue
                if filename.startswith("package://"):
                    fn = filename.replace("package://", "")
                    package_name = fn.split("/")[0]
                    urdf_folder = os.path.dirname(source)
                    # resolve file path from package name, i.e. find
                    # the package folder from the URDF folder
                    if package_name in urdf_folder:
                        filename = os.path.join(urdf_folder[: urdf_folder.rindex(package_name)], fn)
                    else:
                        warnings.warn(
                            f'Warning: package "{package_name}" not found in URDF folder while loading mesh at "{filename}"',
                            stacklevel=2,
                        )
                elif filename.startswith(("http://", "https://")):
                    # download mesh
                    # note that the file must be deleted after use
                    file_tmp = download_asset_tmpfile(filename)
                    filename = file_tmp.name
                else:
                    filename = os.path.join(os.path.dirname(source), filename)
                if not os.path.exists(filename):
                    warnings.warn(f"Warning: mesh file {filename} does not exist", stacklevel=2)
                    continue

                import trimesh  # noqa: PLC0415

                # use force='mesh' to load the mesh as a trimesh object
                # with baked in transforms, e.g. from COLLADA files
                m = trimesh.load(filename, force="mesh")
                scaling = mesh.get("scale") or "1 1 1"
                scaling = np.array([float(x) * scale for x in scaling.split()])
                if hasattr(m, "geometry"):
                    # multiple meshes are contained in a scene
                    for m_geom in m.geometry.values():
                        m_vertices = np.array(m_geom.vertices, dtype=np.float32) * scaling
                        m_faces = np.array(m_geom.faces.flatten(), dtype=np.int32)
                        m_mesh = Mesh(m_vertices, m_faces, maxhullvert=mesh_maxhullvert)
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
                    m_mesh = Mesh(m_vertices, m_faces, maxhullvert=mesh_maxhullvert)
                    s = builder.add_shape_mesh(
                        xform=tf,
                        mesh=m_mesh,
                        **shape_kwargs,
                    )
                    shapes.append(s)
                if file_tmp is not None:
                    os.remove(file_tmp.name)
                    file_tmp = None

        return shapes

    joint_indices = []  # Collect joint indices as we create them

    # add joints

    # mapping from parent, child link names to joint
    parent_child_joint = {}

    joints = []
    for joint in urdf_root.findall("joint"):
        parent = joint.find("parent").get("link")
        child = joint.find("child").get("link")
        joint_custom_attributes = parse_custom_attributes(joint.attrib, builder_custom_attr_joint, parsing_mode="urdf")
        joint_data = {
            "name": joint.get("name"),
            "parent": parent,
            "child": child,
            "type": joint.get("type"),
            "origin": parse_transform(joint),
            "damping": default_joint_damping,
            "friction": 0.0,
            "axis": wp.vec3(1.0, 0.0, 0.0),
            "limit_lower": default_joint_limit_lower,
            "limit_upper": default_joint_limit_upper,
            "custom_attributes": joint_custom_attributes,
        }
        el_axis = joint.find("axis")
        if el_axis is not None:
            ax = el_axis.get("xyz", "1 0 0").strip().split()
            joint_data["axis"] = wp.vec3(float(ax[0]), float(ax[1]), float(ax[2]))
        el_dynamics = joint.find("dynamics")
        if el_dynamics is not None:
            joint_data["damping"] = float(el_dynamics.get("damping", default_joint_damping))
            joint_data["friction"] = float(el_dynamics.get("friction", 0))
        el_limit = joint.find("limit")
        if el_limit is not None:
            joint_data["limit_lower"] = float(el_limit.get("lower", default_joint_limit_lower))
            joint_data["limit_upper"] = float(el_limit.get("upper", default_joint_limit_upper))
        el_mimic = joint.find("mimic")
        if el_mimic is not None:
            joint_data["mimic_joint"] = el_mimic.get("joint")
            joint_data["mimic_multiplier"] = float(el_mimic.get("multiplier", 1))
            joint_data["mimic_offset"] = float(el_mimic.get("offset", 0))

        parent_child_joint[(parent, child)] = joint_data
        joints.append(joint_data)

    # topological sorting of joints because the FK function will resolve body transforms
    # in joint order and needs the parent link transform to be resolved before the child
    urdf_links = []
    sorted_joints = []
    if len(joints) > 0:
        if joint_ordering is not None:
            joint_edges = [(joint["parent"], joint["child"]) for joint in joints]
            sorted_joint_ids = topological_sort(joint_edges, use_dfs=joint_ordering == "dfs")
            sorted_joints = [joints[i] for i in sorted_joint_ids]
        else:
            sorted_joints = joints

        if bodies_follow_joint_ordering:
            body_order: list[str] = [sorted_joints[0]["parent"]] + [joint["child"] for joint in sorted_joints]
            for body in body_order:
                urdf_link = urdf_root.find(f"link[@name='{body}']")
                if urdf_link is None:
                    raise ValueError(f"Link {body} not found in URDF")
                urdf_links.append(urdf_link)
    if len(urdf_links) == 0:
        urdf_links = urdf_root.findall("link")

    # add links and shapes

    # maps from link name -> link index
    link_index: dict[str, int] = {}
    visual_shapes: list[int] = []
    start_shape_count = len(builder.shape_type)

    for urdf_link in urdf_links:
        name = urdf_link.get("name")
        if name is None:
            raise ValueError("Link has no name")
        link = builder.add_link(
            key=name,
            custom_attributes=parse_custom_attributes(urdf_link.attrib, builder_custom_attr_body, parsing_mode="urdf"),
        )

        # add ourselves to the index
        link_index[name] = link

        visuals = urdf_link.findall("visual")
        colliders = urdf_link.findall("collision")

        if parse_visuals_as_colliders:
            colliders = visuals
        else:
            s = parse_shapes(link, visuals, density=0.0, just_visual=True, visible=not hide_visuals)
            visual_shapes.extend(s)

        show_colliders = force_show_colliders
        if parse_visuals_as_colliders:
            show_colliders = True
        elif len(visuals) == 0:
            # we need to show the collision shapes since there are no visual shapes
            show_colliders = True

        parse_shapes(link, colliders, density=default_shape_density, visible=show_colliders)
        m = builder.body_mass[link]
        el_inertia = urdf_link.find("inertial")
        if not ignore_inertial_definitions and el_inertia is not None:
            # overwrite inertial parameters if defined
            inertial_frame = parse_transform(el_inertia)
            com = inertial_frame.p
            builder.body_com[link] = com
            I_m = np.zeros((3, 3))
            el_i_m = el_inertia.find("inertia")
            if el_i_m is not None:
                I_m[0, 0] = float(el_i_m.get("ixx", 0)) * scale**2
                I_m[1, 1] = float(el_i_m.get("iyy", 0)) * scale**2
                I_m[2, 2] = float(el_i_m.get("izz", 0)) * scale**2
                I_m[0, 1] = float(el_i_m.get("ixy", 0)) * scale**2
                I_m[0, 2] = float(el_i_m.get("ixz", 0)) * scale**2
                I_m[1, 2] = float(el_i_m.get("iyz", 0)) * scale**2
                I_m[1, 0] = I_m[0, 1]
                I_m[2, 0] = I_m[0, 2]
                I_m[2, 1] = I_m[1, 2]
                rot = wp.quat_to_matrix(inertial_frame.q)
                I_m = rot @ wp.mat33(I_m)
                builder.body_inertia[link] = I_m
                if any(x for x in I_m):
                    builder.body_inv_inertia[link] = wp.inverse(I_m)
                else:
                    builder.body_inv_inertia[link] = I_m
            el_mass = el_inertia.find("mass")
            if el_mass is not None:
                m = float(el_mass.get("value", 0))
                builder.body_mass[link] = m
                builder.body_inv_mass[link] = 1.0 / m if m > 0.0 else 0.0
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

    end_shape_count = len(builder.shape_type)

    # add base joint
    if len(sorted_joints) > 0:
        base_link_name = sorted_joints[0]["parent"]
    else:
        base_link_name = next(iter(link_index.keys()))
    root = link_index[base_link_name]
    if base_joint is not None:
        # in case of a given base joint, the position is applied first, the rotation only
        # after the base joint itself to not rotate its axis
        base_parent_xform = wp.transform(xform.p, wp.quat_identity())
        base_child_xform = wp.transform((0.0, 0.0, 0.0), wp.quat_inverse(xform.q))
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
                    linear_axes=[ModelBuilder.JointDofConfig(axes[a]) for a in linear_axes],
                    angular_axes=[ModelBuilder.JointDofConfig(axes[a]) for a in angular_axes],
                    parent_xform=base_parent_xform,
                    child_xform=base_child_xform,
                    parent=-1,
                    child=root,
                    key="base_joint",
                )
            )
        elif isinstance(base_joint, dict):
            base_joint["parent"] = -1
            base_joint["child"] = root
            base_joint["parent_xform"] = base_parent_xform
            base_joint["child_xform"] = base_child_xform
            base_joint["key"] = "base_joint"
            joint_indices.append(builder.add_joint(**base_joint))
        else:
            raise ValueError(
                "base_joint must be a comma-separated string of joint axes or a dict with joint parameters"
            )
    elif floating:
        floating_joint_id = builder.add_joint_free(root, key="floating_base")
        joint_indices.append(floating_joint_id)

        # set dofs to transform for the floating base joint
        start = builder.joint_q_start[floating_joint_id]

        builder.joint_q[start + 0] = xform.p[0]
        builder.joint_q[start + 1] = xform.p[1]
        builder.joint_q[start + 2] = xform.p[2]

        builder.joint_q[start + 3] = xform.q[0]
        builder.joint_q[start + 4] = xform.q[1]
        builder.joint_q[start + 5] = xform.q[2]
        builder.joint_q[start + 6] = xform.q[3]
    else:
        joint_indices.append(builder.add_joint_fixed(-1, root, parent_xform=xform, key="fixed_base"))

    # add joints, in the desired order starting from root body
    for joint in sorted_joints:
        parent = link_index[joint["parent"]]
        child = link_index[joint["child"]]
        if child == -1:
            # we skipped the insertion of the child body
            continue

        lower = joint.get("limit_lower", None)
        upper = joint.get("limit_upper", None)
        joint_damping = joint["damping"]

        parent_xform = joint["origin"]

        joint_params = {
            "parent": parent,
            "child": child,
            "parent_xform": parent_xform,
            "key": joint["name"],
            "custom_attributes": joint["custom_attributes"],
        }

        if joint["type"] == "revolute" or joint["type"] == "continuous":
            joint_indices.append(
                builder.add_joint_revolute(
                    axis=joint["axis"],
                    target_kd=joint_damping,
                    limit_lower=lower,
                    limit_upper=upper,
                    **joint_params,
                )
            )
        elif joint["type"] == "prismatic":
            joint_indices.append(
                builder.add_joint_prismatic(
                    axis=joint["axis"],
                    target_kd=joint_damping,
                    limit_lower=lower * scale,
                    limit_upper=upper * scale,
                    **joint_params,
                )
            )
        elif joint["type"] == "fixed":
            joint_indices.append(builder.add_joint_fixed(**joint_params))
        elif joint["type"] == "floating":
            joint_indices.append(builder.add_joint_free(**joint_params))
        elif joint["type"] == "planar":
            # find plane vectors perpendicular to axis
            axis = np.array(joint["axis"])
            axis /= np.linalg.norm(axis)

            # create helper vector that is not parallel to the axis
            helper = np.array([1, 0, 0]) if np.allclose(axis, [0, 1, 0]) else np.array([0, 1, 0])

            u = np.cross(helper, axis)
            u /= np.linalg.norm(u)

            v = np.cross(axis, u)
            v /= np.linalg.norm(v)

            joint_indices.append(
                builder.add_joint_d6(
                    linear_axes=[
                        ModelBuilder.JointDofConfig(
                            u,
                            limit_lower=lower * scale,
                            limit_upper=upper * scale,
                            target_kd=joint_damping,
                        ),
                        ModelBuilder.JointDofConfig(
                            v,
                            limit_lower=lower * scale,
                            limit_upper=upper * scale,
                            target_kd=joint_damping,
                        ),
                    ],
                    **joint_params,
                )
            )
        else:
            raise Exception("Unsupported joint type: " + joint["type"])

    # Create articulation from all collected joints
    if joint_indices:
        articulation_key = urdf_root.attrib.get("name")
        articulation_custom_attrs = parse_custom_attributes(
            urdf_root.attrib, builder_custom_attr_articulation, parsing_mode="urdf"
        )
        builder.add_articulation(
            joints=joint_indices,
            key=articulation_key,
            custom_attributes=articulation_custom_attrs,
        )

    for i in range(start_shape_count, end_shape_count):
        for j in visual_shapes:
            builder.shape_collision_filter_pairs.append((i, j))

    if not enable_self_collisions:
        for i in range(start_shape_count, end_shape_count):
            for j in range(i + 1, end_shape_count):
                builder.shape_collision_filter_pairs.append((i, j))

    if collapse_fixed_joints:
        builder.collapse_fixed_joints()
