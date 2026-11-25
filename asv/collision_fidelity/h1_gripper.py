# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Experimental original Unitree H1 hand pair with coupled prescribed joints.

The original URDF hand subtree retains collision/visual meshes, axes, inertias,
limits, and mimic relations. Local +X points from wrist toward fingers. Six
controls drive twelve revolute joints. Detached root motion makes no whole-arm
reach or actuator force claim. No deformable state is modified.

For external-rigid VBD, update input at t and output at t+dt on every substep.
For full-surface contact, configure builder.default_shape_cfg with
configure_sdf(force_sdf=True, max_resolution=64) before import. All fourteen
original mesh colliders need SDFs. Some source meshes have open edges; audits
report that fact and use normal-signed nearest-surface distance, not a proof
of a watertight solid or of triangle-level non-intersection.
"""

from __future__ import annotations

import copy
import hashlib
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import warp as wp

import newton

from .asset_paths import NEWTON_ASSETS_REVISION, newton_asset_path

__all__ = ["H1_CONTROL_NAMES", "H1_OPEN_CONTROLS", "H1HandDriver", "H1HandSpec", "add_h1_hand"]
H1_CONTROL_NAMES = ("thumb_yaw", "thumb_pitch", "index", "middle", "ring", "pinky")
H1_OPEN_CONTROLS = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
_JOINT_SUFFIXES = (
    "thumb_proximal_yaw_joint",
    "thumb_proximal_pitch_joint",
    "thumb_intermediate_joint",
    "thumb_distal_joint",
    "index_proximal_joint",
    "index_intermediate_joint",
    "middle_proximal_joint",
    "middle_intermediate_joint",
    "ring_proximal_joint",
    "ring_intermediate_joint",
    "pinky_proximal_joint",
    "pinky_intermediate_joint",
)
_CONTROL_INDICES = np.array([0, 1, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5])
_MULTIPLIERS = np.array([1, 1, 1.6, 2.4, 1, 1, 1, 1, 1, 1, 1, 1.0])


def _quaternion(value):
    q = np.asarray(value, dtype=float)
    if q.shape != (4,) or not np.isfinite(q).all() or np.linalg.norm(q) < 1e-10:
        raise ValueError("Expected a finite nonzero quaternion (x, y, z, w)")
    return q / np.linalg.norm(q)


@dataclass
class H1HandSpec:
    """Original geometry indices and source limits; positions in m, angles rad."""

    side: str
    body_ids: tuple[int, ...]
    shape_ids: tuple[int, ...]
    articulation: int
    root_q: int
    root_qd: int
    finger_q: tuple[int, ...]
    finger_qd: tuple[int, ...]
    lower: np.ndarray
    upper: np.ndarray
    velocity_limits: np.ndarray
    control_lower: np.ndarray
    control_upper: np.ndarray
    control_velocity_limits: np.ndarray
    provenance: dict
    collision_vertices: dict[int, np.ndarray]
    collision_triangles: dict[int, np.ndarray]
    shape_link_names: dict[int, str]


def _expand_controls(values, spec=None):
    controls = np.asarray(values, dtype=float)
    if controls.ndim not in (1, 2) or controls.shape[-1] != 6 or not np.isfinite(controls).all():
        raise ValueError("Expected six finite controls [thumb_yaw,thumb_pitch,index,middle,ring,pinky]")
    fingers = controls[..., _CONTROL_INDICES] * _MULTIPLIERS
    if spec is not None and (np.any(fingers < spec.lower - 1e-7) or np.any(fingers > spec.upper + 1e-7)):
        raise ValueError("Coupled command exceeds original H1 joint limits")
    return fingers


def _source_subtree(side, label, urdf):
    source = ET.parse(urdf).getroot()
    root_name = f"{side}_hand_link"
    links = {root_name}
    for _ in range(20):
        for joint in source.findall("joint"):
            if joint.find("parent").attrib["link"] in links:
                links.add(joint.find("child").attrib["link"])
    root = ET.Element("robot", name=label)
    for child in source:
        keep = child.tag == "link" and child.attrib["name"] in links
        if child.tag == "joint":
            keep = child.find("parent").attrib["link"] in links and child.find("child").attrib["link"] in links
        if keep:
            root.append(copy.deepcopy(child))
    for mesh in root.iter("mesh"):
        mesh.set("filename", str((urdf.parent / mesh.attrib["filename"]).resolve()))
    return root


def add_h1_hand(
    builder, *, side, root_pos=(0.0, 0.0, 0.0), rotation=(0.0, 0.0, 0.0, 1.0), controls=H1_OPEN_CONTROLS, label=None
):
    """Import one unchanged original hand, with six URDF-coupled controls.

    Root position is the original left/right_hand_link wrist origin, not a
    pinch TCP. Left/right source geometry is distinct; neither is a reflected
    copy. The arm and incoming wrist-roll joint are omitted. Source fixed palm
    joint is collapsed without moving its geometry. All bodies are kinematic.
    Materials, margins, gaps and SDF settings inherit builder defaults.
    """
    if side not in ("left", "right"):
        raise ValueError("side must be left or right")
    label = label or f"h1_{side}_hand"
    root_pos = np.asarray(root_pos, dtype=float)
    rotation = _quaternion(rotation)
    fingers = _expand_controls(controls)
    if root_pos.shape != (3,) or not np.isfinite(root_pos).all() or fingers.shape != (12,):
        raise ValueError("Expected one finite root position and six controls")
    asset = newton_asset_path("unitree_h1")
    urdf = asset / "urdf/h1_with_hand.urdf"
    subtree = _source_subtree(side, label, urdf)
    prefix = "L" if side == "left" else "R"
    names = [prefix + "_" + suffix for suffix in _JOINT_SUFFIXES]
    joints = {j.attrib["name"]: j for j in subtree.findall("joint")}
    limits = [joints[name].find("limit").attrib for name in names]
    lower = np.array([float(v["lower"]) for v in limits])
    upper = np.array([float(v["upper"]) for v in limits])
    speeds = np.array([float(v["velocity"]) for v in limits])
    for i, name in enumerate(names):
        mimic = joints[name].find("mimic")
        if i in (2, 3, 5, 7, 9, 11):
            reference = names[1 if i < 4 else i - 1]
            if (
                mimic is None
                or mimic.attrib["joint"] != reference
                or float(mimic.attrib.get("multiplier", 1)) != _MULTIPLIERS[i]
                or float(mimic.attrib.get("offset", 0)) != 0
            ):
                raise ValueError("Unexpected source mimic relation")
    if np.any(fingers < lower - 1e-7) or np.any(fingers > upper + 1e-7):
        raise ValueError("Coupled command exceeds original H1 joint limits")
    control_lower = np.array(
        [np.max(lower[_CONTROL_INDICES == i] / _MULTIPLIERS[_CONTROL_INDICES == i]) for i in range(6)]
    )
    control_upper = np.array(
        [np.min(upper[_CONTROL_INDICES == i] / _MULTIPLIERS[_CONTROL_INDICES == i]) for i in range(6)]
    )
    control_speeds = np.array(
        [np.min(speeds[_CONTROL_INDICES == i] / _MULTIPLIERS[_CONTROL_INDICES == i]) for i in range(6)]
    )
    body_start, shape_start, joint_start = builder.body_count, builder.shape_count, builder.joint_count
    articulation = builder.articulation_count
    builder.add_urdf(
        ET.tostring(subtree, encoding="unicode"),
        floating=True,
        xform=wp.transform(wp.vec3(*root_pos), wp.quat(*rotation)),
        collapse_fixed_joints=True,
        enable_self_collisions=False,
    )
    by_name = {builder.joint_label[j].rsplit("/", 1)[-1]: j for j in range(joint_start, builder.joint_count)}
    finger_q = tuple(builder.joint_q_start[by_name[name]] for name in names)
    finger_qd = tuple(builder.joint_qd_start[by_name[name]] for name in names)
    for index, value in zip(finger_q, fingers, strict=True):
        builder.joint_q[index] = float(value)
    bodies = tuple(range(body_start, builder.body_count))
    for body in bodies:
        builder.body_flags[body] = int(newton.BodyFlags.KINEMATIC)
    shapes = tuple(
        i
        for i in range(shape_start, builder.shape_count)
        if builder.shape_flags[i] & newton.ShapeFlags.COLLIDE_PARTICLES
    )
    if len(bodies) != 13 or len(shapes) != 14 or len(finger_q) != 12:
        raise ValueError("Unexpected original H1 subtree topology")
    vertices, triangles, shape_links = {}, {}, {}
    for i in shapes:
        if builder.shape_type[i] not in (newton.GeoType.MESH, newton.GeoType.CONVEX_MESH):
            raise ValueError("Unexpected nonmesh original H1 collider")
        vertices[i] = np.asarray(builder.shape_source[i].vertices) * np.asarray(builder.shape_scale[i])
        triangles[i] = np.asarray(builder.shape_source[i].indices).reshape(-1, 3)
        shape_links[i] = builder.shape_label[i]
    mesh_files = []
    import trimesh  # noqa: PLC0415 -- optional importer dependency, needed only while building a hand

    for link in subtree.findall("link"):
        mesh = link.find("collision/geometry/mesh")
        if mesh is None:
            continue
        path = Path(mesh.attrib["filename"])
        geometry = trimesh.load(path, force="mesh", process=True)
        counts = np.bincount(geometry.edges_unique_inverse)
        mesh_files.append(
            {
                "link": link.attrib["name"],
                "filename": path.name,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "vertices": len(geometry.vertices),
                "triangles": len(geometry.faces),
                "watertight": bool(geometry.is_watertight),
                "winding_consistent": bool(geometry.is_winding_consistent),
                "boundary_edges": int(np.count_nonzero(counts == 1)),
                "nonmanifold_edges": int(np.count_nonzero(counts > 2)),
            }
        )
    provenance = {
        "asset": f"Unitree H1 original {side} articulated hand",
        "label": label,
        "source": "https://github.com/newton-physics/newton-assets",
        "source_revision": NEWTON_ASSETS_REVISION,
        "upstream_source": "https://github.com/unitreerobotics/unitree_ros/tree/a3b70ca/robots/h1_description",
        "source_urdf": str(urdf),
        "source_urdf_sha256": hashlib.sha256(urdf.read_bytes()).hexdigest(),
        "license": "BSD-3-Clause",
        "license_sha256": hashlib.sha256((asset / "LICENSE").read_bytes()).hexdigest(),
        "geometry_changes": "None; retain original collision STL and visual DAE geometry at scale1. Only omit arm and resolve mesh paths.",
        "units": "metres, kilograms, seconds, radians",
        "root_frame": f"{side}_hand_link wrist origin; local +X toward fingers",
        "control_names": list(H1_CONTROL_NAMES),
        "joint_order": names,
        "joint_lower_rad": lower.tolist(),
        "joint_upper_rad": upper.tolist(),
        "joint_velocity_limits_rad_s": speeds.tolist(),
        "control_lower_rad": control_lower.tolist(),
        "control_upper_rad": control_upper.tolist(),
        "control_velocity_limits_rad_s": control_speeds.tolist(),
        "mimic_control_indices": _CONTROL_INDICES.tolist(),
        "mimic_multipliers": _MULTIPLIERS.tolist(),
        "collision_shape_count": len(shapes),
        "body_count": len(bodies),
        "collision_meshes": mesh_files,
        "shape_margin_m": [float(builder.shape_margin[i]) for i in shapes],
        "shape_gap_m": [float(builder.shape_gap[i]) for i in shapes],
        "mesh_sdf_configuration": [
            {
                "shape_id": i,
                "force_sdf": bool(builder.shape_force_sdf[i]),
                "max_resolution": builder.shape_sdf_max_resolution[i],
                "effective_default_max_resolution": builder.default_shape_cfg.sdf_max_resolution,
            }
            for i in shapes
        ],
        "command_support": "Prescribed detached root and coupled original joints via public eval_fk; force capacity and full arm reach not tested; no object pins",
        "self_collision": "Imported with hand self collision disabled; kinematic commands must be audited for interdigit/interhand overlap",
    }
    return H1HandSpec(
        side,
        bodies,
        shapes,
        articulation,
        builder.joint_q_start[joint_start],
        builder.joint_qd_start[joint_start],
        finger_q,
        finger_qd,
        lower,
        upper,
        speeds,
        control_lower,
        control_upper,
        control_speeds,
        provenance,
        vertices,
        triangles,
        shape_links,
    )


@wp.kernel
def _mesh_distances(
    points: wp.array[wp.vec3],
    shapes: wp.array[int],
    body_q: wp.array[wp.transform],
    shape_body: wp.array[int],
    shape_transform: wp.array[wp.transform],
    shape_scale: wp.array[wp.vec3],
    source_ptr: wp.array[wp.uint64],
    result: wp.array[float],
):
    i = wp.tid()
    distance = float(1e6)
    for j in range(shapes.shape[0]):
        shape = shapes[j]
        transform = body_q[shape_body[shape]] * shape_transform[shape]
        p = wp.transform_point(wp.transform_inverse(transform), points[i])
        scale = shape_scale[shape]
        query = wp.mesh_query_point_sign_normal(source_ptr[shape], wp.cw_div(p, scale), 1000.0)
        if query.result:
            nearest = wp.cw_mul(wp.mesh_eval_position(source_ptr[shape], query.face, query.u, query.v), scale)
            distance = wp.min(distance, wp.length(p - nearest) * query.sign)
    result[i] = distance


def _box_gap(center, axes, half, other_center, other_axes, other_half):
    delta = center - other_center
    directions = [*axes.T, *other_axes.T]
    directions.extend(np.cross(a, b) for a in axes.T for b in other_axes.T)
    gaps = []
    for axis in directions:
        length = np.linalg.norm(axis)
        if length < 1e-10:
            continue
        direction = axis / length
        gaps.append(abs(delta @ direction) - abs(direction @ axes) @ half - abs(direction @ other_axes) @ other_half)
    return float(max(gaps))


@wp.kernel
def _command(
    clock: wp.array[float],
    offset: float,
    times: wp.array[float],
    positions: wp.array[wp.vec3],
    rotations: wp.array[wp.quat],
    fingers: wp.array2d[float],
    root_q: int,
    root_qd: int,
    finger_q: wp.array[int],
    finger_qd: wp.array[int],
    root_com: wp.vec3,
    q: wp.array[float],
    qd: wp.array[float],
):
    t = clock[0] + offset
    index = int(0)
    blend = float(0.0)
    rate = float(0.0)
    if t >= times[times.shape[0] - 1]:
        index = times.shape[0] - 2
        blend = 1.0
    for i in range(times.shape[0] - 1):
        if t >= times[i] and t < times[i + 1]:
            index = i
            span = times[i + 1] - times[i]
            u = (t - times[i]) / span
            blend = u * u * (3.0 - 2.0 * u)
            rate = 6.0 * u * (1.0 - u) / span
    position = positions[index] + blend * (positions[index + 1] - positions[index])
    rotation = wp.quat_slerp(rotations[index], rotations[index + 1], blend)
    linear = rate * (positions[index + 1] - positions[index])
    delta = rotations[index + 1] * wp.quat_inverse(rotations[index])
    axis = wp.vec3(delta[0], delta[1], delta[2])
    length = wp.length(axis)
    angular = wp.vec3(0.0)
    if length > 1e-7:
        angular = axis / length * (2.0 * wp.atan2(length, delta[3]) * rate)
    linear += wp.cross(angular, wp.quat_rotate(rotation, root_com))
    for i in range(3):
        q[root_q + i] = position[i]
        qd[root_qd + i] = linear[i]
        qd[root_qd + 3 + i] = angular[i]
    for i in range(4):
        q[root_q + 3 + i] = rotation[i]
    for i in range(fingers.shape[1]):
        change = fingers[index + 1, i] - fingers[index, i]
        q[finger_q[i]] = fingers[index, i] + blend * change
        qd[finger_qd[i]] = rate * change


class H1HandDriver:
    """Smooth original hand FK, isolated to its articulation.

    For external-rigid VBD call update(clock, state0, state1, dt=substep_dt).
    For internal-rigid VBD command the kinematic hand endpoint at t+dt in
    state0 before step; the solver retains its own previous-pose history.
    Root positions are original hand_link wrist origins [m]; six coupled control values are [rad].
    """

    def __init__(self, model, spec, *, times, positions, rotations, controls):
        self.model, self.spec = model, spec
        times = np.asarray(times, dtype=float)
        positions = np.asarray(positions, dtype=float)
        fingers = _expand_controls(controls, spec)
        rotations = np.asarray(rotations, dtype=float)
        count = len(times)
        if count < 2 or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
            raise ValueError("Expected at least two finite strictly increasing keyframe times")
        if positions.shape != (count, 3) or not np.isfinite(positions).all() or fingers.shape != (count, 12):
            raise ValueError("Expected one palm position and 12 expanded finger coordinates per keyframe")
        if not np.isfinite(fingers).all() or np.any(fingers < spec.lower - 1e-7) or np.any(fingers > spec.upper + 1e-7):
            raise ValueError("Finger keyframe exceeds original joint position limits")
        if np.any(1.5 * np.abs(np.diff(fingers, axis=0)) / np.diff(times)[:, None] > spec.velocity_limits + 1e-7):
            raise ValueError("Finger trajectory exceeds original joint velocity limits")
        if rotations.shape == (4,):
            rotations = np.tile(rotations, (count, 1))
        if rotations.shape != (count, 4):
            raise ValueError("Expected one quaternion or one per keyframe")
        rotations = np.asarray([_quaternion(q) for q in rotations])
        for i in range(1, count):
            if np.dot(rotations[i - 1], rotations[i]) < 0:
                rotations[i] *= -1
        self.times = wp.array(times, dtype=float, device=model.device)
        self.positions = wp.array(positions, dtype=wp.vec3, device=model.device)
        self.rotations = wp.array(rotations, dtype=wp.quat, device=model.device)
        self.fingers = wp.array(fingers, dtype=float, device=model.device)
        self.q, self.qd = wp.clone(model.joint_q), wp.clone(model.joint_qd)
        self.finger_q = wp.array(spec.finger_q, dtype=int, device=model.device)
        self.finger_qd = wp.array(spec.finger_qd, dtype=int, device=model.device)
        self.articulations = wp.array([spec.articulation], dtype=int, device=model.device)
        self._shapes = wp.array(spec.shape_ids, dtype=int, device=model.device)
        self.root_com = wp.vec3(*model.body_com.numpy()[spec.body_ids[0]])

    def _update(self, clock, state, offset):
        wp.launch(
            _command,
            dim=1,
            inputs=[
                clock,
                offset,
                self.times,
                self.positions,
                self.rotations,
                self.fingers,
                self.spec.root_q,
                self.spec.root_qd,
                self.finger_q,
                self.finger_qd,
                self.root_com,
                self.q,
                self.qd,
            ],
            device=self.model.device,
        )
        newton.eval_fk(self.model, self.q, self.qd, state, indices=self.articulations)

    def update(self, clock, state0, state1=None, *, dt=0.0):
        """Set original hand FK at t and optionally t+dt [s]."""
        if not np.isfinite(dt) or dt < 0:
            raise ValueError("Expected finite nonnegative substep dt")
        self._update(clock, state0, 0.0)
        if state1 is not None:
            self._update(clock, state1, float(dt))

    def world_collision_meshes(self, state):
        """Return exact transformed original collider vertices/triangles by shape."""
        bodies = state.body_q.numpy()
        local = self.model.shape_transform.numpy()
        body = self.model.shape_body.numpy()
        result = {}
        for shape, vertices in self.spec.collision_vertices.items():
            transform = wp.transform(*bodies[body[shape]]) * wp.transform(*local[shape])
            rotation = np.asarray(wp.quat_to_matrix(wp.transform_get_rotation(transform))).reshape(3, 3)
            world = vertices @ rotation.T + np.asarray(wp.transform_get_translation(transform))
            result[shape] = (world, self.spec.collision_triangles[shape])
        return result

    def geometry_audit(
        self,
        state,
        *,
        points=None,
        radii=0.0,
        ground_z=None,
        full_surface_contact=None,
        box_center=None,
        box_half=None,
        box_rotation=(0.0, 0.0, 0.0, 1.0),
        box_margin=0.0,
    ):
        """Audit material vertices, plane and optional finite oriented support box.

        Signed vertex distances use original mesh triangles and nearest normal.
        Source open edges limit signed-solid interpretation. Plane clearance uses
        all transformed vertices. Finite-box clearance uses conservative local
        mesh bounds with both margins; negative gap is inconclusive overlap of
        bounds, positive gap proves separation at this sampled pose only.
        """
        result = {
            "full_surface_contact_enabled": full_surface_contact,
            "audit_scope": "Original mesh nearest-surface vertex audit; no triangle-intersection or continuous-trajectory proof",
            "signed_distance_method": "Warp normal-signed nearest original mesh triangle; source nonwatertight meshes disclosed in provenance",
        }
        if points is None:
            points = state.particle_q.numpy() if state.particle_q is not None else np.empty((0, 3))
        points = np.asarray(points, dtype=np.float32)
        if len(points):
            distance = wp.empty(len(points), dtype=float, device=self.model.device)
            wp.launch(
                _mesh_distances,
                dim=len(points),
                inputs=[
                    wp.array(points, dtype=wp.vec3, device=self.model.device),
                    self._shapes,
                    state.body_q,
                    self.model.shape_body,
                    self.model.shape_transform,
                    self.model.shape_scale,
                    self.model.shape_source_ptr,
                    distance,
                ],
                device=self.model.device,
            )
            signed = distance.numpy()
            result.update(
                vertices_inside_robot_solids=int(np.count_nonzero(signed < -1e-6)),
                minimum_vertex_to_robot_distance_m=float(signed.min()),
                minimum_shell_to_robot_distance_m=float((signed - np.asarray(radii)).min()),
            )
        meshes = self.world_collision_meshes(state)
        margin = self.model.shape_margin.numpy()
        low = min(float(vertices[:, 2].min()) for vertices, _ in meshes.values())
        low_margin = min(float(vertices[:, 2].min() - margin[shape]) for shape, (vertices, _) in meshes.items())
        result.update(minimum_robot_collision_z_m=low, collision_shape_count=len(meshes))
        if ground_z is not None:
            result.update(
                robot_clearance_above_ground_plane_m=low - float(ground_z),
                hand_ground_clearance_including_margin_m=low_margin - float(ground_z),
            )
        if box_center is not None:
            half = np.asarray(box_half, dtype=float)
            center = np.asarray(box_center, dtype=float)
            axes = np.asarray(wp.quat_to_matrix(wp.quat(*_quaternion(box_rotation)))).reshape(3, 3)
            bodies = state.body_q.numpy()
            local = self.model.shape_transform.numpy()
            body = self.model.shape_body.numpy()
            gaps = []
            for shape, vertices in self.spec.collision_vertices.items():
                transform = wp.transform(*bodies[body[shape]]) * wp.transform(*local[shape])
                rotation = np.asarray(wp.quat_to_matrix(wp.transform_get_rotation(transform))).reshape(3, 3)
                lo, hi = vertices.min(0), vertices.max(0)
                world = np.asarray(wp.transform_point(transform, wp.vec3(*((lo + hi) / 2))))
                gap = _box_gap(world, rotation, (hi - lo) / 2 + margin[shape], center, axes, half + box_margin)
                gaps.append((gap, shape))
            worst = min(gaps)
            result.update(
                minimum_conservative_box_separation_m=worst[0],
                closest_box_shape_id=worst[1],
                box_audit_method="Separating-axis gap for original collider local bounds; both shape margins included; sampled pose only",
            )
        return result
