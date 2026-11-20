# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Experimental original Allegro hand with prescribed root and joint keyframes.

The original URDF collision geometry, joint axes, and limits remain unchanged.
Mass/inertia inputs retain the source values; Newton may correct invalid inertia
when finalizing the model. Commands do not test actuator force capacity.
No deformable particles or bodies outside the hand articulation are modified.
"""

from __future__ import annotations

import hashlib
import xml.etree.ElementTree as ET
from dataclasses import dataclass

import numpy as np
import warp as wp

import newton

from .asset_paths import NEWTON_ASSETS_REVISION, newton_asset_path

__all__ = ["GROUND_GRASP_FINGERS", "OPEN_FINGERS", "AllegroHandDriver", "AllegroHandSpec", "add_allegro_hand"]

OPEN_FINGERS = (0.0,) * 12 + (0.6, 0.0, 0.0, 0.0)
GROUND_GRASP_FINGERS = (
    -0.08192108,
    0.92523682,
    0.36568835,
    1.618,
    0.0,
    0.57200838,
    0.65679205,
    0.73815920,
    0.08192108,
    0.92523682,
    0.36568836,
    1.618,
    0.69388872,
    0.96651192,
    0.86169769,
    1.31911096,
)


def _quaternion(value):
    q = np.asarray(value, dtype=float)
    if q.shape != (4,) or not np.isfinite(q).all() or np.linalg.norm(q) < 1e-10:
        raise ValueError("Expected a finite nonzero quaternion (x, y, z, w)")
    return q / np.linalg.norm(q)


@dataclass
class AllegroHandSpec:
    """Original hand indices and source joint limits in canonical joint order."""

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
    provenance: dict


def add_allegro_hand(builder, *, root_pos, rotation, finger_positions=OPEN_FINGERS, label="allegro_hand"):
    """Add the unchanged cached left hand using the palm_link origin [m].

    Finger positions use original joint_0.0 through joint_15.0 order [rad].
    Local +X is the palm's contact side. A +pi/2 rotation about world Y makes
    the palm face down. Materials and margins inherit builder defaults.
    """
    root_pos = np.asarray(root_pos, dtype=float)
    rotation = _quaternion(rotation)
    fingers = np.asarray(finger_positions, dtype=float)
    asset = newton_asset_path("wonik_allegro")
    urdf = asset / "urdf/allegro_hand_description_left.urdf"
    source = ET.parse(urdf).getroot()
    source_joints = {j.attrib["name"]: j for j in source.findall("joint")}
    limits = [source_joints[f"joint_{i}.0"].find("limit").attrib for i in range(16)]
    lower = np.asarray([float(limit["lower"]) for limit in limits])
    upper = np.asarray([float(limit["upper"]) for limit in limits])
    speeds = np.asarray([float(limit["velocity"]) for limit in limits])
    if root_pos.shape != (3,) or not np.isfinite(root_pos).all():
        raise ValueError("Expected a finite palm origin [m]")
    if fingers.shape != (16,) or not np.isfinite(fingers).all():
        raise ValueError("Expected 16 finite original finger joint positions [rad]")
    if np.any(fingers < lower - 1e-7) or np.any(fingers > upper + 1e-7):
        raise ValueError("Finger command exceeds original Allegro joint position limits")
    body_start, shape_start, joint_start = builder.body_count, builder.shape_count, builder.joint_count
    articulation = builder.articulation_count
    builder.add_urdf(
        str(urdf),
        xform=wp.transform(wp.vec3(*root_pos), wp.quat(*rotation)),
        floating=True,
        collapse_fixed_joints=True,
        enable_self_collisions=False,
    )
    by_name = {builder.joint_label[j].rsplit("/", 1)[-1]: j for j in range(joint_start, builder.joint_count)}
    finger_joints = [by_name[f"joint_{i}.0"] for i in range(16)]
    finger_q = tuple(builder.joint_q_start[j] for j in finger_joints)
    finger_qd = tuple(builder.joint_qd_start[j] for j in finger_joints)
    for coordinate, value in zip(finger_q, fingers, strict=True):
        builder.joint_q[coordinate] = float(value)
    bodies = tuple(range(body_start, builder.body_count))
    for body in bodies:
        builder.body_flags[body] = int(newton.BodyFlags.KINEMATIC)
    shapes = tuple(
        i
        for i in range(shape_start, builder.shape_count)
        if builder.shape_flags[i] & newton.ShapeFlags.COLLIDE_PARTICLES
    )
    provenance = {
        "asset": "Original Wonik Allegro left hand",
        "label": label,
        "source": "https://github.com/newton-physics/newton-assets",
        "source_revision": NEWTON_ASSETS_REVISION,
        "source_urdf": str(urdf),
        "source_urdf_sha256": hashlib.sha256(urdf.read_bytes()).hexdigest(),
        "license_file": str(asset / "LICENSE"),
        "license_sha256": hashlib.sha256((asset / "LICENSE").read_bytes()).hexdigest(),
        "geometry_changes": "None; original collision boxes and fingertip spheres, visual meshes, and scale",
        "inertia_processing": "Original URDF inputs; Newton finalize validates/corrects inertia. All hand bodies are prescribed kinematic bodies.",
        "collision_shape_count": len(shapes),
        "joint_order": [f"joint_{i}.0" for i in range(16)],
        "joint_lower_rad": lower.tolist(),
        "joint_upper_rad": upper.tolist(),
        "joint_velocity_limits_rad_s": speeds.tolist(),
        "shape_margin_m": [float(builder.shape_margin[i]) for i in shapes],
        "shape_gap_m": [float(builder.shape_gap[i]) for i in shapes],
        "command_support": "Prescribed root and finger FK; actuator force capacity is not tested; no object pins",
    }
    return AllegroHandSpec(
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
        provenance,
    )


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
    for i in range(16):
        change = fingers[index + 1, i] - fingers[index, i]
        q[finger_q[i]] = fingers[index, i] + blend * change
        qd[finger_qd[i]] = rate * change


class AllegroHandDriver:
    """Smooth original hand FK, isolated to its articulation.

    For ordinary VBD command the kinematic hand endpoint at t+dt in
    state0 before step; the solver retains its own previous-pose history.
    Root positions are palm_link origins [m]; finger values are [rad].
    """

    def __init__(self, model, spec, *, times, positions, rotations, finger_positions):
        self.model, self.spec = model, spec
        times = np.asarray(times, dtype=float)
        positions = np.asarray(positions, dtype=float)
        fingers = np.asarray(finger_positions, dtype=float)
        rotations = np.asarray(rotations, dtype=float)
        count = len(times)
        if count < 2 or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
            raise ValueError("Expected at least two finite strictly increasing keyframe times")
        if positions.shape != (count, 3) or not np.isfinite(positions).all() or fingers.shape != (count, 16):
            raise ValueError("Expected one palm position and 16 finger coordinates per keyframe")
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
        newton.eval_fk(
            self.model,
            self.q,
            self.qd,
            state,
            indices=self.articulations,
            body_flag_filter=newton.BodyFlags.KINEMATIC,
        )

    def update(self, clock, state0, state1=None, *, dt=0.0):
        """Set original hand FK at t and optionally t+dt [s]."""
        if not np.isfinite(dt) or dt < 0:
            raise ValueError("Expected finite nonnegative substep dt")
        self._update(clock, state0, 0.0)
        if state1 is not None:
            self._update(clock, state1, float(dt))

    def geometry_audit(self, state, *, points=None, radii=0.0, ground_z=0.0):
        """Audit actual original boxes/spheres against material vertices [m].

        The ground result includes shape margins. Vertex distances use the
        authored solids, and do not prove triangle-level non-intersection.
        """
        q = state.body_q.numpy()
        body = self.model.shape_body.numpy()
        local = self.model.shape_transform.numpy()
        scale = self.model.shape_scale.numpy()
        kinds = self.model.shape_type.numpy()
        margin = self.model.shape_margin.numpy()
        points = np.empty((0, 3)) if points is None else np.asarray(points)
        distance = np.full(len(points), np.inf)
        min_z = np.inf
        for shape in self.spec.shape_ids:
            transform = wp.transform(*q[body[shape]]) * wp.transform(*local[shape])
            rotation = np.asarray(wp.quat_to_matrix(wp.transform_get_rotation(transform))).reshape(3, 3)
            center = np.asarray(wp.transform_get_translation(transform))
            p = (points - center) @ rotation
            if kinds[shape] == newton.GeoType.BOX:
                d = abs(p) - scale[shape]
                signed = np.linalg.norm(np.maximum(d, 0), axis=1) + np.minimum(d.max(axis=1), 0)
                low = center[2] - abs(rotation[2]) @ scale[shape]
            elif kinds[shape] == newton.GeoType.SPHERE:
                signed = np.linalg.norm(p, axis=1) - scale[shape, 0]
                low = center[2] - scale[shape, 0]
            else:
                raise ValueError("Unexpected primitive in original Allegro collision geometry")
            distance = np.minimum(distance, signed)
            min_z = min(min_z, float(low - margin[shape]))
        result = {"hand_ground_clearance_including_margin_m": min_z - ground_z}
        if len(points):
            result.update(
                vertices_inside_original_hand_solids=int(np.count_nonzero(distance < -1e-6)),
                minimum_vertex_to_hand_distance_m=float(distance.min()),
                minimum_shell_to_hand_distance_m=float((distance - np.asarray(radii)).min()),
            )
        return result
