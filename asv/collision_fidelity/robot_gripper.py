# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Experimental prescribed Franka hand adapter for contact smoke scenes.

The detached hand retains the cached robot's authored collision geometry,
visual geometry, joint axes, limits, and inertias. Its floating root and two
finger joints are commanded through forward kinematics. Use ordinary VBD internal rigid integration and command the input pose at
the end of each substep. This is prescribed robot motion,
not a test of dynamic robot actuation capacity. No soft particles are modified.
For full-surface rigid-soft contact, configure the builder's default shape
config with ``configure_sdf(force_sdf=True, max_resolution=64)`` before import.
The palm mesh requires that SDF; the original finger boxes remain analytic.
"""

from __future__ import annotations

import copy
import hashlib
import itertools
import xml.etree.ElementTree as ET
from dataclasses import dataclass

import numpy as np
import warp as wp

import newton

from .asset_paths import NEWTON_ASSETS_REVISION, newton_asset_path

__all__ = ["FrankaHandDriver", "FrankaHandSpec", "add_franka_hand"]

_TCP = (0.0, 0.0, 0.1034)


def _hand_xml(label, asset):
    source = ET.parse(asset / "urdf/fr3_franka_hand.urdf").getroot()
    root = ET.Element("robot", name=label)
    links = {"fr3_hand", "fr3_hand_tcp", "fr3_leftfinger", "fr3_rightfinger"}
    for child in source:
        keep = child.tag == "link" and child.attrib["name"] in links
        if child.tag == "joint":
            keep = child.find("parent").attrib["link"] in links and child.find("child").attrib["link"] in links
        if keep:
            root.append(copy.deepcopy(child))
    for mesh in root.iter("mesh"):
        mesh.set("filename", str(asset / mesh.attrib["filename"].removeprefix("package://franka_emika_panda/")))
    return ET.tostring(root, encoding="unicode")


def _opening(value):
    value = float(value)
    if not np.isfinite(value) or not 0.0 <= value <= 0.08:
        raise ValueError("Franka nominal opening must lie in [0, 0.08] m")
    return value


def _rotation(value):
    q = np.asarray(value, dtype=float)
    if q.shape != (4,) or not np.isfinite(q).all() or np.linalg.norm(q) < 1e-10:
        raise ValueError("rotation must be a finite nonzero quaternion (x, y, z, w)")
    return q / np.linalg.norm(q)


@dataclass
class FrankaHandSpec:
    """Imported hand indices and source provenance; lengths are in metres."""

    label: str
    body_ids: tuple[int, ...]
    shape_ids: tuple[int, ...]
    articulation: int
    root_q: int
    root_qd: int
    finger_q: tuple[int, int]
    finger_qd: tuple[int, int]
    provenance: dict
    collision_vertices: dict[int, np.ndarray]


def add_franka_hand(builder, *, pinch_pos, rotation=(0.0, 0.0, 0.0, 1.0), opening=0.06, label="franka_hand"):
    """Add the original detached hand, using its authored TCP as pinch center.

    Args:
        builder: Newton model builder.
        pinch_pos: World TCP position [m].
        rotation: Hand-to-world quaternion (x, y, z, w). Local Y closes;
            local +Z points from the palm toward the fingertips.
        opening: Twice the finger joint coordinate [m], in [0, 0.08].
            Authored inner tip faces give a geometric gap 0.00004 m smaller.
        label: Name used in provenance. Geometry remains at original scale.

    Shape contact materials, margins, gaps, and optional palm SDF construction
    inherit ``builder.default_shape_cfg``. Per-vertex rigid-soft contact does
    not require a palm SDF; full-surface contact does.
    """
    opening = _opening(opening)
    rotation = _rotation(rotation)
    pinch_pos = np.asarray(pinch_pos, dtype=float)
    if pinch_pos.shape != (3,) or not np.isfinite(pinch_pos).all():
        raise ValueError("pinch_pos must be a finite 3D position")
    asset = newton_asset_path("franka_emika_panda")
    urdf = asset / "urdf/fr3_franka_hand.urdf"
    xml = _hand_xml(label, asset)
    body_start, shape_start, joint_start = builder.body_count, builder.shape_count, builder.joint_count
    articulation = builder.articulation_count
    builder.add_urdf(
        xml,
        floating=True,
        xform=wp.transform_identity(),
        collapse_fixed_joints=True,
        enable_self_collisions=False,
        force_show_colliders=False,
    )
    root_joint = joint_start
    root_q, root_qd = builder.joint_q_start[root_joint], builder.joint_qd_start[root_joint]
    fingers = [j for j in range(joint_start, builder.joint_count) if "fr3_finger_joint" in builder.joint_label[j]]
    if len(fingers) != 2 or builder.joint_coord_count - root_q != 9:
        raise ValueError("Expected one floating hand root and two original prismatic finger joints")
    finger_q = tuple(builder.joint_q_start[j] for j in fingers)
    finger_qd = tuple(builder.joint_qd_start[j] for j in fingers)
    root_position = pinch_pos - np.asarray(wp.quat_rotate(wp.quat(*rotation), wp.vec3(*_TCP)))
    builder.joint_q[root_q : root_q + 7] = [*root_position, *rotation]
    for index in finger_q:
        builder.joint_q[index] = opening / 2
    bodies = tuple(range(body_start, builder.body_count))
    for body in bodies:
        builder.body_flags[body] = int(newton.BodyFlags.KINEMATIC)
    shapes = tuple(
        i
        for i in range(shape_start, builder.shape_count)
        if builder.shape_flags[i] & newton.ShapeFlags.COLLIDE_PARTICLES
    )
    vertices = {}
    tips = []
    for i in shapes:
        scale = np.asarray(builder.shape_scale[i])
        if builder.shape_type[i] == newton.GeoType.BOX:
            vertices[i] = np.asarray(list(itertools.product((-1.0, 1.0), repeat=3))) * scale
            if np.allclose(scale, (0.00875, 0.0076, 0.00925), atol=1e-8):
                tips.append(i)
        elif builder.shape_type[i] in (newton.GeoType.MESH, newton.GeoType.CONVEX_MESH):
            vertices[i] = np.asarray(builder.shape_source[i].vertices) * scale
        else:
            raise ValueError("Unexpected collision primitive in the original Franka hand")
    provenance = {
        "asset": "Franka FR3 detached hand subtree",
        "source": "https://github.com/newton-physics/newton-assets",
        "source_revision": NEWTON_ASSETS_REVISION,
        "source_urdf": str(urdf),
        "source_urdf_sha256": hashlib.sha256(urdf.read_bytes()).hexdigest(),
        "license_file": str(asset / "LICENSE"),
        "license_sha256": hashlib.sha256((asset / "LICENSE").read_bytes()).hexdigest(),
        "subtree_links": ["fr3_hand", "fr3_hand_tcp", "fr3_leftfinger", "fr3_rightfinger"],
        "geometry_changes": "None. Only the arm is omitted and package mesh paths are resolved locally.",
        "robot_command": "Prescribed floating root and original finger joints through eval_fk; no object pins",
        "collision_shape_count": len(shapes),
        "finger_tip_shape_ids": tips,
        "finger_joint_range_m": [0.0, 0.04],
        "finger_joint_velocity_limit_m_s": 0.2,
        "nominal_opening_range_m": [0.0, 0.08],
        "actual_tip_gap_offset_m": -0.00004,
        "tcp_in_hand_m": list(_TCP),
        "tip_face_width_x_m": 0.0175,
        "tip_face_length_z_m": 0.0185,
        "tip_face_center_z_relative_tcp_m": 0.00025,
        "shape_margin_m": [float(builder.shape_margin[i]) for i in shapes],
        "shape_gap_m": [float(builder.shape_gap[i]) for i in shapes],
        "mesh_sdf_configuration": [
            {
                "shape_id": i,
                "force_sdf": bool(builder.shape_force_sdf[i]),
                "max_resolution": builder.shape_sdf_max_resolution[i],
                "target_voxel_size_m": builder.shape_sdf_target_voxel_size[i],
                "representation": "Signed-distance field derived from the original collision mesh",
            }
            for i in shapes
            if builder.shape_type[i] in (newton.GeoType.MESH, newton.GeoType.CONVEX_MESH)
        ],
    }
    return FrankaHandSpec(
        label, bodies, shapes, articulation, root_q, root_qd, finger_q, finger_qd, provenance, vertices
    )


@wp.kernel
def _command(
    clock: wp.array[float],
    time_offset: float,
    times: wp.array[float],
    positions: wp.array[wp.vec3],
    rotations: wp.array[wp.quat],
    openings: wp.array[float],
    root_q: int,
    root_qd: int,
    finger_q: wp.vec2i,
    finger_qd: wp.vec2i,
    root_com: wp.vec3,
    q: wp.array[float],
    qd: wp.array[float],
):
    t = clock[0] + time_offset
    index = int(0)
    if t >= times[times.shape[0] - 1]:
        index = times.shape[0] - 1
    position = positions[index]
    rotation = rotations[index]
    opening = openings[index]
    linear = wp.vec3(0.0)
    angular = wp.vec3(0.0)
    opening_velocity = float(0.0)
    for i in range(times.shape[0] - 1):
        if t >= times[i] and t < times[i + 1]:
            span = times[i + 1] - times[i]
            u = (t - times[i]) / span
            blend = u * u * (3.0 - 2.0 * u)
            rate = 6.0 * u * (1.0 - u) / span
            position = positions[i] + blend * (positions[i + 1] - positions[i])
            linear = rate * (positions[i + 1] - positions[i])
            rotation = wp.quat_slerp(rotations[i], rotations[i + 1], blend)
            delta = rotations[i + 1] * wp.quat_inverse(rotations[i])
            axis = wp.vec3(delta[0], delta[1], delta[2])
            axis_length = wp.length(axis)
            if axis_length > 1e-7:
                angular = axis / axis_length * (2.0 * wp.atan2(axis_length, delta[3]) * rate)
            opening = openings[i] + blend * (openings[i + 1] - openings[i])
            opening_velocity = rate * (openings[i + 1] - openings[i])
    tcp = wp.vec3(0.0, 0.0, 0.1034)
    origin = position - wp.quat_rotate(rotation, tcp)
    com_velocity = linear + wp.cross(angular, wp.quat_rotate(rotation, root_com - tcp))
    for i in range(3):
        q[root_q + i] = origin[i]
        qd[root_qd + i] = com_velocity[i]
        qd[root_qd + i + 3] = angular[i]
    for i in range(4):
        q[root_q + i + 3] = rotation[i]
    for i in range(2):
        q[finger_q[i]] = 0.5 * opening
        qd[finger_qd[i]] = 0.5 * opening_velocity


@wp.kernel
def _signed_distances(
    points: wp.array[wp.vec3],
    shapes: wp.array[int],
    body_q: wp.array[wp.transform],
    shape_body: wp.array[int],
    shape_transform: wp.array[wp.transform],
    shape_type: wp.array[int],
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
        if shape_type[shape] == newton.GeoType.BOX:
            d = wp.vec3(wp.abs(p[0]), wp.abs(p[1]), wp.abs(p[2])) - shape_scale[shape]
            outside = wp.vec3(wp.max(d[0], 0.0), wp.max(d[1], 0.0), wp.max(d[2], 0.0))
            signed = wp.length(outside) + wp.min(wp.max(wp.max(d[0], d[1]), d[2]), 0.0)
            distance = wp.min(distance, signed)
        else:
            scale = shape_scale[shape]
            query = wp.mesh_query_point_sign_normal(source_ptr[shape], wp.cw_div(p, scale), 1000.0)
            if query.result:
                nearest = wp.cw_mul(wp.mesh_eval_position(source_ptr[shape], query.face, query.u, query.v), scale)
                distance = wp.min(distance, wp.length(p - nearest) * query.sign)
    result[i] = distance


class FrankaHandDriver:
    """Drive a hand along smooth keyframes, with actual FK and COM velocities.

    Keyframe positions are world TCP coordinates [m]. Openings are nominal
    full openings [m]. Rotations may be one constant quaternion or one per
    keyframe. Each interval uses cubic smoothstep; angular motion uses slerp.
    For ordinary VBD, set the clock to the substep endpoint and call
    ``update(clock, state0)``. The solver owns previous poses and collision
    detection. The optional second state is useful for initial-state audits;
    it is not a separate rigid dynamics solver.
    The clock is a one-element Warp float array in seconds, owned by the scene.
    """

    def __init__(self, model, spec, *, times, positions, rotations, openings):
        self.model, self.spec = model, spec
        times = np.asarray(times, dtype=float)
        positions = np.asarray(positions, dtype=float)
        openings = np.asarray(openings, dtype=float)
        rotations = np.asarray(rotations, dtype=float)
        count = len(times)
        if count < 2 or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
            raise ValueError("Hand keyframe times must be finite and strictly increasing, with at least two entries")
        if positions.shape != (count, 3) or not np.isfinite(positions).all() or openings.shape != (count,):
            raise ValueError("Expected one finite TCP position and opening per hand keyframe")
        for value in openings:
            _opening(value)
        max_speed = 0.75 * np.abs(np.diff(openings)) / np.diff(times)
        if np.any(max_speed > 0.2 + 1e-8):
            raise ValueError("Hand trajectory exceeds the original 0.2 m/s finger joint velocity limit")
        if rotations.shape == (4,):
            rotations = np.tile(rotations, (count, 1))
        if rotations.shape != (count, 4):
            raise ValueError("Expected one quaternion or one quaternion per hand keyframe")
        rotations = np.asarray([_rotation(q) for q in rotations])
        for i in range(1, count):
            if np.dot(rotations[i - 1], rotations[i]) < 0:
                rotations[i] *= -1
        self.times = wp.array(times, dtype=float, device=model.device)
        self.positions = wp.array(positions, dtype=wp.vec3, device=model.device)
        self.rotations = wp.array(rotations, dtype=wp.quat, device=model.device)
        self.openings = wp.array(openings, dtype=float, device=model.device)
        self.q, self.qd = wp.clone(model.joint_q), wp.clone(model.joint_qd)
        self._articulations = wp.array([spec.articulation], dtype=int, device=model.device)
        self._shapes = wp.array(spec.shape_ids, dtype=int, device=model.device)
        self._root_com = wp.vec3(*model.body_com.numpy()[spec.body_ids[0]])

    def _update_state(self, clock, state, time_offset):
        wp.launch(
            _command,
            dim=1,
            inputs=[
                clock,
                time_offset,
                self.times,
                self.positions,
                self.rotations,
                self.openings,
                self.spec.root_q,
                self.spec.root_qd,
                wp.vec2i(*self.spec.finger_q),
                wp.vec2i(*self.spec.finger_qd),
                self._root_com,
                self.q,
                self.qd,
            ],
            device=self.model.device,
        )
        newton.eval_fk(
            self.model,
            self.q,
            self.qd,
            state,
            indices=self._articulations,
            body_flag_filter=newton.BodyFlags.KINEMATIC,
        )

    def update(self, clock, state0, state1=None, *, dt=0.0):
        """Evaluate original hand FK at the clock and optionally a second time.

        Ordinary VBD uses only ``state0`` and an endpoint clock. Neither call
        changes particles or bodies outside this hand's articulation.
        """
        if not np.isfinite(dt) or dt < 0:
            raise ValueError("Hand update dt must be finite and nonnegative")
        self._update_state(clock, state0, 0.0)
        if state1 is not None:
            self._update_state(clock, state1, float(dt))

    def geometry_audit(self, state, *, points=None, radii=0.0, table_top=None, full_surface_contact=None):
        """Check material vertices against actual hand solids and plane height.

        Distances use exact box SDFs and the palm mesh's closest surface with
        normal-based sign. This is a vertex audit, not a triangle-intersection
        proof. Plane clearance does not test a finite table's XY footprint.
        """
        points = state.particle_q.numpy() if points is None else np.asarray(points, dtype=np.float32)
        result = dict(self.spec.provenance)
        result["full_surface_contact_enabled"] = full_surface_contact
        if len(points):
            distances = wp.empty(len(points), dtype=float, device=self.model.device)
            wp.launch(
                _signed_distances,
                dim=len(points),
                inputs=[
                    wp.array(points, dtype=wp.vec3, device=self.model.device),
                    self._shapes,
                    state.body_q,
                    self.model.shape_body,
                    self.model.shape_transform,
                    self.model.shape_type,
                    self.model.shape_scale,
                    self.model.shape_source_ptr,
                    distances,
                ],
                device=self.model.device,
            )
            signed = distances.numpy()
            result["vertices_inside_robot_solids"] = int(np.count_nonzero(signed < -1e-6))
            result["minimum_vertex_to_robot_distance_m"] = float(signed.min())
            result["minimum_shell_to_robot_distance_m"] = float((signed - np.asarray(radii)).min())
        bodies = state.body_q.numpy()
        transforms = self.model.shape_transform.numpy()
        shape_bodies = self.model.shape_body.numpy()
        root_transform = wp.transform(*bodies[self.spec.body_ids[0]])
        root_rotation = np.asarray(wp.quat_to_matrix(wp.transform_get_rotation(root_transform))).reshape(3, 3)
        tcp = np.asarray(wp.transform_point(root_transform, wp.vec3(*_TCP)))
        tip_bounds = []
        zmin = float("inf")
        for shape, vertices in self.spec.collision_vertices.items():
            transform = wp.transform(*bodies[shape_bodies[shape]]) * wp.transform(*transforms[shape])
            rotation = np.asarray(wp.quat_to_matrix(wp.transform_get_rotation(transform))).reshape(3, 3)
            world = vertices @ rotation.T + np.asarray(wp.transform_get_translation(transform))
            zmin = min(zmin, float(world[:, 2].min()))
            if shape in self.spec.provenance["finger_tip_shape_ids"]:
                local = (world - tcp) @ root_rotation
                tip_bounds.append((local.min(axis=0), local.max(axis=0)))
        tip_bounds.sort(key=lambda bounds: bounds[0][1])
        result["tip_collision_bounds_in_tcp_m"] = [[lo.tolist(), hi.tolist()] for lo, hi in tip_bounds]
        result["physical_tip_aperture_m"] = float(tip_bounds[1][0][1] - tip_bounds[0][1][1])
        result["tip_contact_span_x_in_tcp_m"] = [
            float(max(bounds[0][0] for bounds in tip_bounds)),
            float(min(bounds[1][0] for bounds in tip_bounds)),
        ]
        result["tip_contact_span_z_in_tcp_m"] = [
            float(max(bounds[0][2] for bounds in tip_bounds)),
            float(min(bounds[1][2] for bounds in tip_bounds)),
        ]
        result["minimum_robot_collision_z_m"] = zmin
        if table_top is not None:
            result["robot_clearance_above_table_plane_m"] = zmin - float(table_top)
        result["audit_scope"] = (
            "Material vertices against original robot solids; plane height excludes table XY footprint"
        )
        return result
