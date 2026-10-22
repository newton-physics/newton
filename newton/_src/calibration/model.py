# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""The simulated cable a calibration run scores against.

:class:`CableWorld` builds a rod from candidate parameters, settles it, drives it
along a recorded trajectory, and renders it through each camera that observed the
recording. One instance holds a whole *population* of candidates, one per world:
the parameter arguments are lists with one entry per world.

An optimizer proposes many candidates per iteration. Simulating them together, in
one solver and render pass, makes the search affordable, so the evaluation takes
a population and not a single candidate.
"""

from __future__ import annotations

import copy
import math
import warnings
from collections.abc import Sequence
from typing import Any

import numpy as np
import warp as wp

import newton
import newton.sensors as sensors

from ..geometry.inertia import compute_inertia_capsule

SETTLE_MODES = ("dynamic",)
"""Ways :meth:`CableWorld.settle` can bring the cable to its initial equilibrium."""


@wp.kernel
def _shape_index_to_mask(
    shape_indices: wp.array4d[wp.uint32],
    selected_shapes: wp.array[wp.int32],
    masks: wp.array3d[wp.uint8],
):
    """Mark pixels whose nearest visible shape belongs to the cable."""
    world, y, x = wp.tid()
    shape = shape_indices[world, 0, y, x]
    value = wp.uint8(0)
    # Misses and non-shape geometry use sentinel IDs outside the shape table.
    if shape < wp.uint32(selected_shapes.shape[0]):
        if selected_shapes[int(shape)] != 0:
            value = wp.uint8(255)
    masks[world, y, x] = value


@wp.kernel
def _drive_clamp_kernel(
    body_indices: wp.array[wp.int32],
    transform_buffer: wp.array[wp.transform],
    anchor_pos: wp.vec3,
    anchor_rot: wp.quat,
    sim_frame: wp.array[wp.int32],
    substep_local: int,
    sim_substeps: int,
    body_q0: wp.array[wp.transform],
    body_q1: wp.array[wp.transform],
):
    """Move each kinematic clamp body to the recorded TCP pose for one substep.

    The pose is the TCP sample ``frame * sim_substeps + substep`` of
    ``transform_buffer``, or its last sample, composed with the TCP-to-anchor
    transform ``(anchor_pos, anchor_rot)``. :class:`CableWorld` builds the rod
    with the same composition, so the driven pose at frame 0 equals the built
    pose. The kernel writes the pose into both states.

    ``sim_frame`` is a device array so that the frame index advances when a CUDA
    graph replays the substeps. A scalar argument would keep its value from the
    time of the capture.
    """
    tid = wp.tid()
    body_id = body_indices[tid]

    ti = sim_frame[0] * sim_substeps + substep_local
    ti = wp.min(ti, transform_buffer.shape[0] - 1)

    tb = transform_buffer[ti]
    tb_rot = wp.transform_get_rotation(tb)
    pos = wp.transform_get_translation(tb) + wp.quat_rotate(tb_rot, anchor_pos)
    rot = wp.mul(tb_rot, anchor_rot)
    T = wp.transform(pos, rot)
    body_q0[body_id] = T
    body_q1[body_id] = T


@wp.kernel
def project_cable_kernel(
    body_q: wp.array[wp.transform],
    bodies: wp.array2d[wp.int32],  # (n_worlds, n_capsules)
    seg_len: float,
    cam_pos: wp.vec3,
    cam_quat: wp.quat,  # camera frame in the world frame: looks along -z, +y up
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    out: wp.array3d[wp.float32],  # (n_worlds, n_capsules + 1, 2)
):
    """Project each world's capsule chain to pixel coordinates.

    Runs over all worlds at once, so a loss that needs the cable's image-space
    geometry does not synchronize with the host during a sequence. Each capsule
    runs from its body origin along local +Z for ``seg_len`` [m]. Node ``i < n``
    is the origin of capsule ``i``, and node ``n`` is the far end of the last
    capsule. A node at or behind the image plane gets NaN.
    """
    w, i = wp.tid()
    n = bodies.shape[1]
    b = bodies[w, wp.min(i, n - 1)]
    tb = body_q[b]
    p = wp.transform_get_translation(tb)
    if i == n:  # far end of the final capsule
        p = p + wp.quat_rotate(wp.transform_get_rotation(tb), wp.vec3(0.0, 0.0, seg_len))
    # d = R^T (p - t): the world-frame offset in camera axes.
    d = wp.quat_rotate_inv(cam_quat, p - cam_pos)
    depth = -d[2]  # the camera looks along -z
    if depth <= 1.0e-6:  # at or behind the image plane
        out[w, i, 0] = wp.nan
        out[w, i, 1] = wp.nan
    else:
        # Image v grows downward, camera y upward.
        out[w, i, 0] = cx + fx * d[0] / depth
        out[w, i, 1] = cy - fy * d[1] / depth


# The only supported rest-angle parametrization: a joint's (alpha, beta) pair is the
# rotation vector (alpha, beta, 0), so a joint has no twist about the segment axis.
ANGLE_PARAM_EXP = "exp_map"


def bend_quat(alpha: float, beta: float) -> wp.quat:
    """Relative rotation of one joint under the ``exp_map`` parametrization.

    Args:
        alpha: Rotation-vector component about the local X axis [rad].
        beta: Rotation-vector component about the local Y axis [rad].

    Returns:
        The rotation by ``hypot(alpha, beta)`` about the axis ``(alpha, beta, 0)``.
    """
    theta = math.hypot(float(alpha), float(beta))
    if theta < 1e-12:
        return wp.quat_identity()  # limit of the exponential map at w = 0
    return wp.quat_from_axis_angle(wp.vec3(float(alpha) / theta, float(beta) / theta, 0.0), theta)


def quat_aligning_z_to(direction: Sequence[float]) -> wp.quat:
    """Quaternion that rotates local +Z onto ``direction``.

    The rotation is the shortest arc from +Z to ``direction``, so the roll about
    ``direction`` is fixed. A direction (2 of the 3 rotational DOF) is sufficient
    to orient a cable, which is rotationally symmetric about its own axis.

    Args:
        direction: Target direction ``(x, y, z)``. Any nonzero length.

    Returns:
        The aligning rotation.
    """
    ref = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    d0 = np.array([direction[0], direction[1], direction[2]], dtype=np.float64)
    d0 /= np.linalg.norm(d0)
    cross = np.cross(ref, d0)
    dot = float(np.dot(ref, d0))
    cross_len = float(np.linalg.norm(cross))
    if cross_len < 1e-8:
        # Parallel or anti-parallel to (0, 0, 1).
        q = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), 0.0 if dot > 0.0 else math.pi)
    else:
        axis = (cross / cross_len).tolist()
        q = wp.quat_from_axis_angle(wp.vec3(*axis), math.atan2(cross_len, dot))
    return q


def rod_points_from_bend_angles(
    segment_length: float,
    angles: Sequence[tuple[float, float]],
) -> tuple[list[wp.vec3], list[wp.quat]]:
    """Create rod points and segment quaternions from per-joint bend angles.

    The first segment starts at the origin and runs along +Z. Each next segment's
    frame is the previous frame bent by ``(alpha, beta)`` (see :func:`bend_quat`).
    All-zero angle pairs give a straight rod along +Z.

    Args:
        segment_length: Length of each segment [m].
        angles: ``(alpha, beta)`` bend angles [rad] per segment.

    Returns:
        Tuple ``(points, quats)`` compatible with :meth:`newton.ModelBuilder.add_rod`.
        ``points`` has ``len(angles) + 1`` entries, ``quats`` has ``len(angles)``.
    """
    q_frame = wp.quat_identity()
    points = [wp.vec3(0.0, 0.0, 0.0)]
    quats = []
    for alpha, beta in angles:
        quats.append(q_frame)
        d = wp.quat_rotate(q_frame, wp.vec3(0.0, 0.0, 1.0))
        points.append(points[-1] + d * segment_length)
        q_frame = wp.mul(q_frame, bend_quat(alpha, beta))

    return points, quats


def cable_points_clamped_at(
    clamp_capsule: int,
    clamp_pos: wp.vec3,
    clamp_rot: wp.quat,
    segment_length: float,
    angles: Sequence[tuple[float, float]],
) -> tuple[list[wp.vec3], list[wp.quat]]:
    """Build a rod whose capsule ``clamp_capsule`` sits at ``(clamp_pos, clamp_rot)``.

    The rod stays one continuous chain. An interior clamp capsule lets both sides
    hang, which models a grasp in the middle of the cable. The function builds the
    chain from its first node, then rotates and translates the whole chain so that
    the clamp capsule has the clamp pose.

    Args:
        clamp_capsule: Index of the capsule held at the clamp pose.
        clamp_pos: Start node of the clamp capsule [m].
        clamp_rot: Orientation of the clamp capsule's frame.
        segment_length: Length of each segment [m].
        angles: ``(alpha, beta)`` bend angles [rad] per segment.

    Returns:
        Tuple ``(points, quats)``, as :func:`rod_points_from_bend_angles`
        returns.
    """
    local_pts, local_quats = rod_points_from_bend_angles(segment_length, angles)
    fp = local_pts[clamp_capsule]
    fr = local_quats[clamp_capsule]
    # Align the selected capsule's frame with the explicit clamp frame.
    t_rot = wp.mul(clamp_rot, wp.quat_inverse(fr))
    t_pos = clamp_pos - wp.quat_rotate(t_rot, fp)
    points = [t_pos + wp.quat_rotate(t_rot, p) for p in local_pts]
    quats = [wp.mul(t_rot, q) for q in local_quats]
    return points, quats


def _require_twist_args(twist_stiffness: float | None, twist_damping: float | None) -> None:
    """Require both twist arguments before they go to :meth:`~newton.ModelBuilder.add_rod`.

    :meth:`~newton.ModelBuilder.add_joint_rod` sets the twist damping to 0.0
    without an error when it gets ``twist_stiffness`` but no ``twist_damping``.

    Raises:
        ValueError: If ``twist_stiffness`` or ``twist_damping`` is ``None``.
    """
    if twist_stiffness is None or twist_damping is None:
        raise ValueError(
            "add_rod needs both twist_stiffness and twist_damping: add_joint_rod sets the "
            "twist damping to 0.0 when only twist_stiffness is given. "
            f"Got twist_stiffness={twist_stiffness}, twist_damping={twist_damping}."
        )


class CableWorld:
    """Newton simulation of a population of candidate cables, one per world.

    :meth:`run_sequence` scores all worlds in one solver and render pass.

    Every value that changes what a fit means comes from the caller. Only
    ``cable_axis``, ``fps``, ``sim_substeps``, ``settle_check_every`` and
    ``settle_move_tol`` have defaults.

    The world frame is the robot base frame of the goals (see
    :class:`~.goal.CableGoal`).

    Args:
        cable_start: Position ``(x, y, z)`` [m] the cable is built at when
            ``transform_buffer`` is ``None``. The anchor transform is added to it.
        angles_list: Rest configuration per world. Each entry is a sequence of
            ``num_elements`` ``(alpha, beta)`` bend-angle pairs [rad]; see
            :func:`bend_quat`. Its length sets the world count.
        bend_stiffness_list: Bend stiffness [N·m/rad] per world.
        transform_buffer: Recorded TCP poses in the world frame, sampled at
            ``fps * sim_substeps`` and indexed as ``frame * sim_substeps + substep``.
            Sample 0 is the TCP pose at frame 0. ``None`` holds the clamp capsule
            static at ``cable_start``.
        cameras: The views of one recording, which share this simulation. Each
            entry is a dict with these keys:

            - ``"sensor_pos"``: camera position ``(x, y, z)`` [m] in the world frame.
              Required.
            - ``"sensor_quat"``: camera orientation ``(qx, qy, qz, qw)`` in the
              world frame. Required.
            - ``"camera_intrinsics"``: pinhole intrinsics
              ``(width, height, fx, fy, cx, cy)`` [px], or ``None``.
            - ``"render_size"``: image size ``(width, height)`` [px]; required when
              ``"camera_intrinsics"`` is ``None``.
            - ``"fov_deg"``: vertical field of view [deg]; required when
              ``"camera_intrinsics"`` is ``None``.
        sim_iterations: VBD solver iterations per substep.
        settle_mode: How :meth:`settle` reaches the initial equilibrium. One of
            :data:`SETTLE_MODES`.
        clamp_position: Arc length [m] from the node-0 end of the cable to the
            grasp point, in ``[0, num_elements * segment_length]``. 0.0 clamps the
            end of the cable. An interior value lets both sides hang.
        bend_damping_list: Bend damping [N·m·s/rad] per world.
        twist_stiffness_list: Twist stiffness [N·m/rad] per world.
        twist_damping_list: Twist damping [N·m·s/rad] per world.
        stretch_stiffness: Stretch stiffness of every joint [N/m].
        num_elements: Number of capsules in the cable.
        segment_length: Length of each capsule [m].
        cable_radius: Capsule radius [m].
        cable_mass: Total mass of the cable [kg].
        angle_parametrization: Parametrization of ``angles_list``. Must be
            :data:`ANGLE_PARAM_EXP`.
        attachment_transform: TCP-to-attachment transform
            ``((x, y, z), (qx, qy, qz, qw))``, translation [m].
        cable_axis: Direction ``(x, y, z)`` of the cable at the grasp, in the TCP
            frame. Replaces the rotation of ``attachment_transform``; its
            translation still applies. ``None`` keeps the attachment rotation.
        fps: Simulation frame rate [Hz].
        sim_substeps: Solver substeps per frame. ``transform_buffer`` must be
            sampled at ``fps * sim_substeps``. The solver settings are part of
            the model: VBD does not fully converge the bend constraint, so
            ``sim_substeps`` and ``sim_iterations`` change the equilibrium shape,
            and a fit is valid only at the settings it was fitted with.
        settle_check_every: Number of frames between two convergence checks of
            :meth:`settle`. At least 1.
        settle_move_tol: Largest body displacement [m] between two checks at
            which :meth:`settle` counts the cable as settled.

    Raises:
        ValueError: If a per-world list length
            differs from ``len(angles_list)``, ``angle_parametrization`` is not
            supported, ``settle_mode`` is not in :data:`SETTLE_MODES`,
            ``attachment_transform`` or ``cable_axis`` has the wrong shape,
            ``cable_axis`` is zero, ``clamp_position`` is outside the cable, a
            per-world twist value is ``None``, or ``settle_check_every`` is less
            than 1; if ``cameras`` is empty; or if a camera has no
            ``sensor_pos`` or ``sensor_quat``, has neither ``camera_intrinsics``
            nor both ``render_size`` and ``fov_deg``, or has a fractional image
            size.
    """

    def __init__(
        self,
        cable_start: Sequence[float],
        angles_list: Sequence[Sequence[tuple[float, float]]],
        bend_stiffness_list: Sequence[float],
        transform_buffer: wp.array[wp.transform] | None,
        *,
        cameras: Sequence[dict[str, Any]],
        sim_iterations: int,
        settle_mode: str,
        clamp_position: float,
        bend_damping_list: Sequence[float],
        twist_stiffness_list: Sequence[float],
        twist_damping_list: Sequence[float],
        stretch_stiffness: float,
        num_elements: int,
        segment_length: float,
        cable_radius: float,
        cable_mass: float,
        angle_parametrization: str,
        attachment_transform: Sequence[Sequence[float]],
        cable_axis: Sequence[float] | None = None,
        fps: int = 60,
        sim_substeps: int = 20,
        settle_check_every: int = 25,
        settle_move_tol: float = 1.0e-3,
    ) -> None:
        self.fps = fps
        self.sim_substeps = sim_substeps
        if settle_check_every < 1:
            raise ValueError(f"settle_check_every must be at least 1, got {settle_check_every}.")
        if not cameras:
            raise ValueError("cameras must hold at least one camera.")
        self.settle_check_every = settle_check_every
        self.settle_move_tol = settle_move_tol
        self.n = len(angles_list)
        # Rest angles in another parametrization are refused, not misread.
        if angle_parametrization != ANGLE_PARAM_EXP:
            raise ValueError(
                f"angle_parametrization {angle_parametrization!r} is not supported; only "
                f"{ANGLE_PARAM_EXP!r} is. The same (alpha, beta) pair describes a different "
                "joint rotation under a different parametrization, so rest angles recorded "
                "under one cannot be used. Fit the rest configuration again."
            )
        self.num_elements = num_elements
        self.segment_length = segment_length
        # Check the lengths here, so the error names the list.
        for name, values in (
            ("bend_stiffness_list", bend_stiffness_list),
            ("bend_damping_list", bend_damping_list),
            ("twist_stiffness_list", twist_stiffness_list),
            ("twist_damping_list", twist_damping_list),
        ):
            if len(values) != self.n:
                raise ValueError(
                    f"{name} has {len(values)} entries but the population is {self.n} "
                    "(from angles_list); every per-world list must match."
                )
        self.transform_buffer = transform_buffer
        self.has_control = transform_buffer is not None

        # The build and the drive apply the same attachment transform, so the anchor
        # does not jump at frame 0.
        attachment_pos, attachment_quat = attachment_transform
        if len(attachment_pos) != 3 or len(attachment_quat) != 4:
            raise ValueError(
                "attachment_transform must be ((x, y, z), (qx, qy, qz, qw)); "
                f"got position length {len(attachment_pos)}, quaternion length {len(attachment_quat)}."
            )
        attachment_pos = wp.vec3(*(float(v) for v in attachment_pos))
        attachment_rot = wp.quat(*(float(v) for v in attachment_quat))

        # cable_axis, in the TCP frame, replaces the attachment rotation: the clamp
        # capsule's local +Z points along it.
        if cable_axis is not None:
            if len(cable_axis) != 3:
                raise ValueError(f"cable_axis must be (x, y, z); got length {len(cable_axis)}.")
            if not any(cable_axis):
                raise ValueError("cable_axis must be a nonzero direction.")
            attachment_rot = quat_aligning_z_to(wp.vec3(*(float(v) for v in cable_axis)))

        self.sim_dt = (1.0 / self.fps) / self.sim_substeps
        if settle_mode not in SETTLE_MODES:
            raise ValueError(f"settle_mode {settle_mode!r} is not one of {list(SETTLE_MODES)}.")

        cable_length = num_elements * segment_length
        # The cable color and the black background of _render() set only the
        # appearance of the RGB frames; the masks come from shape indices.
        cable_color = (1.0, 1.0, 1.0)

        # The capsule that contains the grasp point is kinematic. The anchor is the
        # attachment moved back along its local +Z by the grasp point's offset in
        # that capsule, so the grasp point lands on the attachment.
        clamp_position = float(clamp_position)
        if not (0.0 <= clamp_position <= cable_length):
            raise ValueError(
                f"clamp_position {clamp_position} must be in [0, cable_length] "
                f"= [0, {cable_length}] (num_elements * segment_length)."
            )
        # 1e-9 makes a grasp on a node pick the capsule that starts there: in
        # floating point, 0.15 / 0.05 floors to 2.
        self.clamp_capsule = min(int(clamp_position / segment_length + 1e-9), num_elements - 1)
        self.clamp_offset = max(0.0, clamp_position - self.clamp_capsule * segment_length)
        self.anchor_rot = attachment_rot
        self.anchor_pos = attachment_pos + wp.quat_rotate(attachment_rot, wp.vec3(0.0, 0.0, -self.clamp_offset))

        builder = newton.ModelBuilder()

        # The mass is given as a density, so add_rod derives mass and inertia from one
        # value. Each capsule carries cable_mass / num_elements; the kinematic clamp
        # capsule's share is carried by the clamp.
        unit_mass, _, _ = compute_inertia_capsule(1.0, cable_radius, 0.5 * segment_length)
        rod_cfg = copy.copy(builder.default_shape_cfg)
        rod_cfg.density = (cable_mass / num_elements) / unit_mass
        # The capsules are the only shapes, and each candidate has its own world, so
        # a contact can only be a cable touching itself. The model has no contacts.
        rod_cfg.has_shape_collision = False
        rod_cfg.has_particle_collision = False

        self.cable_bodies_list = []
        cable_shape_ids = []
        kinematic_idx_list = []

        # The clamp capsule's pose at frame 0; cable_points_clamped_at moves the
        # built rod onto it.
        k = self.clamp_capsule
        if self.has_control:
            tcp0 = transform_buffer.numpy()[0]
            tcp0_pos = wp.vec3(float(tcp0[0]), float(tcp0[1]), float(tcp0[2]))
            tcp0_rot = wp.quat(float(tcp0[3]), float(tcp0[4]), float(tcp0[5]), float(tcp0[6]))
            clamp_body_pos = tcp0_pos + wp.quat_rotate(tcp0_rot, self.anchor_pos)
            clamp_body_rot = wp.mul(tcp0_rot, self.anchor_rot)
        else:
            clamp_body_pos = wp.vec3(*cable_start) + self.anchor_pos
            clamp_body_rot = self.anchor_rot

        for i in range(self.n):
            angles = angles_list[i]
            bend_stiffness = bend_stiffness_list[i]
            bend_damping = bend_damping_list[i]
            twist_stiffness = twist_stiffness_list[i]
            twist_damping = twist_damping_list[i]
            builder.begin_world(label=f"cable_{i}")
            cable_points, cable_edge_q = cable_points_clamped_at(
                k, clamp_body_pos, clamp_body_rot, segment_length, angles
            )
            _require_twist_args(twist_stiffness, twist_damping)
            rod = newton.Rod(cable_points, quaternions=cable_edge_q, radius=cable_radius)
            rod_bodies, _ = builder.add_rod(
                rod=rod,
                cfg=rod_cfg,
                stretch_stiffness=stretch_stiffness,
                bend_stiffness=bend_stiffness,
                bend_damping=bend_damping,
                twist_stiffness=twist_stiffness,
                twist_damping=twist_damping,
                # project_cable_kernel needs the body origins at the capsule start nodes.
                body_frame_origin="start",
                color=cable_color,
                label="cable",
            )
            for body in rod_bodies:
                cable_shape_ids.extend(builder.body_shapes[body])

            # rod_bodies[k] is the kinematic clamp capsule.
            clamp = rod_bodies[k]
            builder.body_mass[clamp] = 0.0
            builder.body_inv_mass[clamp] = 0.0
            builder.body_inertia[clamp] = wp.mat33(0.0)
            builder.body_inv_inertia[clamp] = wp.mat33(0.0)

            if self.has_control:
                kinematic_idx_list.append(rod_bodies[k])
            self.cable_bodies_list.append(rod_bodies)
            builder.end_world()

        builder.color()
        self.model = builder.finalize()
        selected_shapes = np.zeros(self.model.shape_count, dtype=np.int32)
        selected_shapes[cable_shape_ids] = 1
        self._cable_shape_flags = wp.array(selected_shapes, dtype=wp.int32, device=self.model.device)
        self._mask_buffers = {}

        self.kinematic_bodies = wp.array(kinematic_idx_list, dtype=wp.int32)

        # Views render one at a time, so views of equal size share one color buffer.
        self.sensor = sensors.SensorTiledCamera(self.model)
        self.cameras = []
        color_images = {}
        for cam in cameras:
            intrinsics = cam.get("camera_intrinsics")
            size = cam.get("render_size")
            fov_deg = cam.get("fov_deg")
            if intrinsics is not None:
                width, height, fx, fy, cx, cy = intrinsics
            elif size is not None and fov_deg is not None:
                width, height = size
            else:
                raise ValueError("each camera needs camera_intrinsics, or render_size and fov_deg.")
            # Stored sizes can be floats; the sensor needs integers.
            if width != int(width) or height != int(height):
                raise ValueError(f"camera image size must be whole pixels, got {width} x {height}.")
            width, height = int(width), int(height)
            cam_pos, cam_quat = cam.get("sensor_pos"), cam.get("sensor_quat")
            if cam_pos is None or cam_quat is None:
                raise ValueError("each camera needs sensor_pos and sensor_quat.")
            pose = wp.transformf(wp.vec3f(*cam_pos), wp.quatf(*cam_quat))
            if intrinsics is not None:
                # project_cable_kernel and the masks put the centre of pixel i at
                # coordinate i. The sensor puts it at i + 0.5, so shift the principal
                # point by half a pixel to render the same rays.
                rays = self.sensor.utils.compute_camera_rays_pinhole_opencv(width, height, fx, fy, cx + 0.5, cy + 0.5)
            else:
                rays = self.sensor.utils.compute_camera_rays_pinhole(width, height, camera_fovs=math.radians(fov_deg))
            if (width, height) not in color_images:
                color_images[(width, height)] = self.sensor.utils.create_color_image_output(width, height)
            self.cameras.append(
                {
                    "rays": rays,
                    "color_image": color_images[(width, height)],
                    # Shape (camera_count=1, world_count).
                    "transforms": wp.array([[pose] * self.n], dtype=wp.transformf, device=self.model.device),
                    # project_cable projects through the pose and intrinsics, not
                    # through the ray array.
                    "sensor_pos": cam_pos,
                    "sensor_quat": cam_quat,
                    "intrinsics": intrinsics,
                }
            )
        self.n_cameras = len(self.cameras)

        # (n_worlds, n_capsules) body indices, for the projection kernel.
        self._cable_bodies = wp.array(
            np.ascontiguousarray(np.array(self.cable_bodies_list, dtype=np.int32)),
            dtype=wp.int32,
            device=self.model.device,
        )
        self._proj_uv = wp.zeros((self.n, self.num_elements + 1, 2), dtype=wp.float32, device=self.model.device)

        self.solver = newton.solvers.SolverVBD(
            self.model,
            iterations=sim_iterations,
            rigid_compliant_alm=True,
        )
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.sim_frame_idx = wp.array([0], dtype=wp.int32, device=self.model.device)

        # The rest shape is the bent build in model.body_q. The simulation starts
        # from a straight cable with the same clamp pose.
        straight_angles = [(0.0, 0.0)] * self.num_elements
        straight_pts, straight_q = cable_points_clamped_at(
            k, clamp_body_pos, clamp_body_rot, segment_length, straight_angles
        )
        bq = self.state_0.body_q.numpy()
        for bodies in self.cable_bodies_list:
            for i, b in enumerate(bodies):
                p, q = straight_pts[i], straight_q[i]
                bq[b] = (p[0], p[1], p[2], q[0], q[1], q[2], q[3])
        self.state_0.body_q.assign(bq)
        self.state_1.body_q.assign(bq)
        self.state_0.body_qd.zero_()
        self.state_1.body_qd.zero_()

        # On CUDA, _step() replays one frame of _simulate() as a graph.
        if self.model.device.is_cuda:
            with wp.ScopedCapture() as capture:
                self._simulate()
            self.graph = capture.graph
        else:
            self.graph = None

    def settle(self, settle_frames: int) -> None:
        """Bring the cable to its initial equilibrium before :meth:`run_sequence`.

        :meth:`run_sequence` does not settle, so the caller calls this first. It is
        safe to call more than once, for example to settle again between two
        :meth:`run_sequence` calls.

        The ``"dynamic"`` mode steps the passive rod under gravity until it stops
        moving, for at most ``settle_frames`` frames (see :meth:`_settle_dynamic`).

        Args:
            settle_frames: Maximum number of frames to step. 0 or less does not
                settle and leaves the cable in its straight initial state.

        Warns:
            UserWarning: If the cable still moves after ``settle_frames`` frames.
        """
        self._settle_dynamic(settle_frames)

    def _settle_dynamic(self, max_frames: int) -> None:
        """Step at frame index 0 until the passive rod stops moving.

        Every ``settle_check_every`` frames, it measures the largest distance a
        cable body moved since the last check, over all worlds. Settling stops when
        that distance is less than ``settle_move_tol`` [m], so it continues until
        the slowest candidate is at rest, or after ``max_frames`` frames.

        All settle steps use the drive of frame 0, so a driven clamp moves only
        within that frame. The sequence then starts from the settled shape.

        Args:
            max_frames: Maximum number of frames to step. 0 or less does not settle.
        """
        if max_frames <= 0:
            return
        check_every, move_tol = self.settle_check_every, self.settle_move_tol
        cable_idx = [b for bodies in self.cable_bodies_list for b in bodies]

        def cable_positions() -> np.ndarray:
            return self.state_0.body_q.numpy()[cable_idx, :3].copy()

        prev = cable_positions()
        moved = float("inf")
        i = 0
        while i < max_frames:
            self._step(0)
            i += 1
            if i % check_every == 0:
                cur = cable_positions()
                moved = float(np.abs(cur - prev).max())
                prev = cur
                if moved < move_tol:
                    break
        # Warn only when a convergence check ran: a settle shorter than
        # check_every frames is a fixed-length settle, not a failed one.
        if i >= check_every and moved >= move_tol:
            warnings.warn(
                f"The settle stopped at its cap of {max_frames} frame(s) without converging: the cable "
                f"moved {moved:.2e} m over the last {check_every} frames, tolerance {move_tol:.0e} m.",
                stacklevel=3,
            )

    def _apply_drive(self, substep: int) -> None:
        """Set the kinematic clamp capsule to the recorded TCP pose for one substep.

        Uses the anchor transform, which is the attachment transform shifted by the
        clamp offset, so the grasp point follows the TCP. Does nothing without a
        drive.

        Args:
            substep: Substep index within the current frame.
        """
        if not self.has_control:
            return
        wp.launch(
            kernel=_drive_clamp_kernel,
            dim=self.kinematic_bodies.shape[0],
            inputs=[
                self.kinematic_bodies,
                self.transform_buffer,
                self.anchor_pos,
                self.anchor_rot,
                self.sim_frame_idx,
                substep,
                self.sim_substeps,
            ],
            outputs=[self.state_0.body_q, self.state_1.body_q],
        )

    def _simulate(self) -> None:
        """Advance all worlds by one frame of ``sim_substeps`` substeps."""
        for substep in range(self.sim_substeps):
            self.state_0.clear_forces()
            self._apply_drive(substep)
            self.solver.step(self.state_0, self.state_1, self.control, None, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def _step(self, frame_idx: int) -> None:
        """Advance one frame, with the drive at frame ``frame_idx``.

        Args:
            frame_idx: Frame of ``transform_buffer`` the drive reads.
        """
        self.sim_frame_idx.fill_(frame_idx)
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self._simulate()

    def project_cable(self, cam: int = 0) -> wp.array3d[wp.float32]:
        """Project each world's capsule chain to pixel coordinates for one view.

        Uses the same state as :meth:`_render`, so the projection matches the
        render of the same frame.

        Args:
            cam: Index of the view in ``cameras``.

        Returns:
            Device array of ``(u, v)`` [px], shape ``(n_worlds, num_elements + 1, 2)``,
            with NaN for a node at or behind the image plane. The
            next call overwrites the same array.

        Raises:
            ValueError: If the view has no ``camera_intrinsics``.
        """
        view = self.cameras[cam]
        if view["intrinsics"] is None:
            raise ValueError("project_cable needs camera_intrinsics; this view has none.")
        _, _, fx, fy, cx, cy = view["intrinsics"]
        wp.launch(
            project_cable_kernel,
            dim=self._proj_uv.shape[:2],
            inputs=[
                self.state_0.body_q,
                self._cable_bodies,
                float(self.segment_length),
                wp.vec3(*(float(v) for v in view["sensor_pos"])),
                wp.quat(*(float(v) for v in view["sensor_quat"])),
                float(fx),
                float(fy),
                float(cx),
                float(cy),
            ],
            outputs=[self._proj_uv],
            device=self.model.device,
        )
        return self._proj_uv

    def _render(
        self, readback: bool = True, cam: int = 0, refit: bool = True, *, with_mask: bool = False
    ) -> tuple[wp.array4d[wp.uint8], list[np.ndarray] | None, wp.array3d[wp.uint8] | None, list[np.ndarray] | None]:
        """Render ``state_0``, the newest state, through one view for all worlds.

        Masks are uint8 (0 or 255) and do not depend on material colors or
        lighting. Mask buffers are allocated only for calls with ``with_mask=True``.

        Args:
            readback: Copy the frames and masks to the host. When ``False``, the
                caller uses the device arrays.
            cam: Index of the view in ``cameras``.
            refit: Refit the BVH to ``state_0`` first. The refit depends only on
                the state, so when several views render one frame, only the first
                needs it.
            with_mask: Also render the cable mask.

        Returns:
            Tuple ``(color_rgba, frames, mask, masks)``. ``color_rgba`` is the
            device render, shape ``(world_count * camera_count, height, width, 4)``.
            ``frames`` is one host RGBA copy per world, or ``None`` without
            readback. ``mask`` is the device mask, shape
            ``(world_count, height, width)``, or ``None`` without ``with_mask``.
            ``masks`` is one host mask copy per world, or ``None`` without
            readback or ``with_mask``.
        """
        if refit:
            self.model.bvh_refit_shapes(self.state_0)
            self.model.bvh_refit_particles(self.state_0)
        view = self.cameras[cam]
        shape_image = mask = None
        if with_mask:
            _, _, height, width = view["color_image"].shape
            size = (width, height)
            if size not in self._mask_buffers:
                self._mask_buffers[size] = (
                    self.sensor.utils.create_shape_index_image_output(width, height),
                    wp.empty((self.n, height, width), dtype=wp.uint8, device=self.model.device),
                )
            shape_image, mask = self._mask_buffers[size]
        self.sensor.update(
            self.state_0,
            view["transforms"],
            view["rays"],
            color_image=view["color_image"],
            shape_index_image=shape_image,
            clear_data=sensors.SensorTiledCamera.ClearData(clear_color=0xFF000000, clear_albedo=0xFF000000),
        )
        if with_mask:
            wp.launch(
                _shape_index_to_mask,
                dim=mask.shape,
                inputs=[shape_image, self._cable_shape_flags],
                outputs=[mask],
                device=self.model.device,
            )
        color_rgba = self.sensor.utils.to_rgba_from_color(view["color_image"])
        if not readback:
            return color_rgba, None, mask, None
        arr = color_rgba.numpy()  # (world_count * camera_count, H, W, 4)
        # On CPU, .numpy() can alias the render buffer, which the next frame or view
        # overwrites. Copy so that each frame owns its pixels.
        frames = [np.array(arr[i], copy=True, order="C") for i in range(self.n)]
        masks = [m.copy() for m in mask.numpy()] if with_mask else None
        return color_rgba, frames, mask, masks

    def record_indices(self, num_frames: int, record_fps: float | None) -> list[int]:
        """Simulation frames that :meth:`run_sequence` records at ``record_fps``.

        A caller uses this to align other per-frame outputs, such as the matching
        goal frames, with the recorded frames.

        Args:
            num_frames: Number of frames in the sequence.
            record_fps: Recording rate [Hz]. ``None`` records nothing. Otherwise
                about ``num_frames * record_fps / fps`` frames are evenly spaced over
                the sequence, with at least one. A rate of ``fps`` or more records
                every frame.

        Returns:
            The recorded frame indices in ascending order.

        Raises:
            ValueError: If ``record_fps`` is not positive.
        """
        if record_fps is None:
            return []
        if record_fps <= 0:
            raise ValueError(f"record_fps must be positive, got {record_fps}.")
        if record_fps >= self.fps:
            return list(range(num_frames))
        n_record = max(1, round(num_frames * record_fps / self.fps))
        return sorted({round(k * (num_frames - 1) / max(1, n_record - 1)) for k in range(n_record)})

    def _goal_frames_at(self, num_frames: int, n_goal: int, goal_times: Sequence[float] | None) -> dict[int, list[int]]:
        """Map each simulation frame to the goal frames scored at it.

        Args:
            num_frames: Number of simulation frames.
            n_goal: Number of goal frames.
            goal_times: Capture time [s] of each goal frame, or ``None`` to spread
                the goal frames evenly over the sequence.

        Returns:
            Goal frame indices per simulation frame, for the frames that score any.
        """
        frames: dict[int, list[int]] = {}
        for j in range(n_goal):
            if goal_times is not None:
                f = round(goal_times[j] * self.fps)
            else:
                f = round(j / max(1, n_goal - 1) * (num_frames - 1))
            frames.setdefault(min(num_frames - 1, max(0, f)), []).append(j)
        return frames

    def _score_view(
        self,
        loss: Any,
        cam: int,
        goal_reprs: Sequence[Any],
        crop: Sequence[int],
        sim_mask: wp.array3d[wp.uint8] | None,
        masks: list[np.ndarray] | None,
        accum: Any,
        totals: list[float],
    ) -> None:
        """Score the current state of every world in one view against some goal frames.

        Args:
            loss: The loss, with the ``CalibrationLoss`` interface.
            cam: Index of the view in ``cameras``.
            goal_reprs: Representations of the goal frames scored at this frame.
            crop: ``[x0, y0, x1, y1]`` pixel crop for scoring.
            sim_mask: Device masks of all worlds, or ``None`` without a render.
            masks: Host masks, one per world, or ``None`` without a readback.
            accum: The loss's on-device accumulator, or ``None`` to score on the
                host.
            totals: Running loss per world. Host scores are added to it.
        """
        geom = self.project_cable(cam) if loss.wants_geometry else None
        if accum is not None:
            for goal_repr in goal_reprs:
                loss.accum(goal_repr, sim_mask, crop, accum, geom=geom)
            return
        geom_np = geom.numpy() if geom is not None else None
        for goal_repr in goal_reprs:
            for i in range(self.n):
                totals[i] += loss.score(
                    goal_repr,
                    masks[i] if masks is not None else None,
                    crop,
                    geom=geom_np[i] if geom_np is not None else None,
                )

    def run_sequence(
        self,
        num_frames: int,
        goal_reprs: Sequence[Any],
        crop: Sequence[int] | Sequence[Sequence[int]],
        loss: Any,
        *,
        record_fps: float | None = None,
        goal_times: Sequence[float] | Sequence[Sequence[float]] | None = None,
    ) -> tuple[list[Any], Any, Any]:
        """Run the frame sequence and score every world against the goal frames.

        This method does not settle the cable. Frame 0 is the state at the time of
        the call, so call :meth:`settle` first.

        Each goal frame is scored against one simulation frame. A world's loss is
        the sum of its frame scores divided by the number of goal frames.

        With several views (see ``cameras``), ``goal_reprs``, ``crop`` and
        ``goal_times`` take one entry per view, and ``losses``, ``recorded_frames``
        and ``recorded_masks`` get a leading per-view axis.

        When ``record_fps`` is set, ``self.last_frame_losses`` holds the per-frame
        loss of world 0 for each view, as ``(time [s], loss)`` pairs.

        Args:
            num_frames: Number of simulation frames to run.
            goal_reprs: Goal representations from the loss's ``prepare()``, one per
                goal frame. The caller computes them once for all candidates.
            crop: ``[x0, y0, x1, y1]`` pixel crop for scoring.
            loss: The loss, with the ``CalibrationLoss`` interface. If it
                supports on-device accumulation and the device is CUDA, all worlds
                are scored on the device and only the per-world totals are copied
                to the host. Otherwise the frames are read back and each world is
                scored on the host.
            record_fps: Recording rate [Hz]; see :meth:`record_indices`. ``None``
                records nothing.
            goal_times: Capture time [s] of each goal frame, relative to the start
                of the sequence. Goal frame ``j`` is then scored against simulation
                frame ``round(goal_times[j] * fps)``. ``None`` spaces the goal frames
                evenly over the sequence by index.

        Returns:
            Tuple ``(losses, recorded_frames, recorded_masks)``. ``losses`` has one
            value per world. Without ``record_fps``, the other two are ``None``.
            Otherwise ``recorded_frames[i]`` and ``recorded_masks[i]`` hold the
            RGBA frames and full-frame cable masks of world ``i``.

        Raises:
            ValueError: If there are several views and the number of
                ``goal_reprs``, ``crop`` or ``goal_times`` entries differs from the
                number of views, or if ``record_fps`` is not positive.
        """

        # With one view, the arguments and the results have no per-view axis.
        multi = self.n_cameras > 1
        if multi:
            reprs = [list(r) for r in goal_reprs]
            crops = list(crop)
            times = list(goal_times) if goal_times is not None else [None] * self.n_cameras
            if not (len(reprs) == len(crops) == len(times) == self.n_cameras):
                raise ValueError(
                    f"CableWorld has {self.n_cameras} cameras; run_sequence needs that many "
                    f"goal_reprs / crop / goal_times entries, got "
                    f"{len(reprs)} / {len(crops)} / {len(times)}."
                )
        else:
            reprs, crops, times = [goal_reprs], [crop], [goal_times]

        losses = [[0.0] * self.n for _ in range(self.n_cameras)]

        # Pair each goal frame with the simulation frame at its capture time, or
        # spread the goal frames evenly without capture times. One map per view.
        #
        # TODO: For a loss that uses only the projected axis, score each capture
        # against the axis interpolated linearly to its capture time between sim
        # frames f and f + 1, instead of against the nearest frame. Interpolate the
        # simulated axis only, never the recorded masks.
        sim_to_goal = [self._goal_frames_at(num_frames, len(reprs[c]), times[c]) for c in range(self.n_cameras)]

        # When recording, keep world 0's loss per scored frame, as the increase of
        # its running total.
        capture = record_fps is not None
        self.last_frame_losses = [[] for _ in range(self.n_cameras)] if capture else None
        prev_total = [0.0] * self.n_cameras

        record_indices: set[int] = set()
        recorded_frames = None
        recorded_masks = None
        if record_fps is not None:
            record_indices = set(self.record_indices(num_frames, record_fps))
            recorded_frames = [[[] for _ in range(self.n)] for _ in range(self.n_cameras)]
            recorded_masks = [[[] for _ in range(self.n)] for _ in range(self.n_cameras)]

        render_at = set().union(*(m.keys() for m in sim_to_goal)) | record_indices

        # On-device scoring copies only the per-world totals to the host.
        accums = [None] * self.n_cameras
        device = self.model.device
        use_device = loss.supports_accum and device is not None and device.is_cuda

        for f in range(num_frames):
            # Render before stepping, so frame f is the state at time f / fps.
            if f in render_at:
                recording = f in record_indices
                # The BVH refit depends only on the state, so only the first view
                # rendered at this frame does it.
                refit = True
                for c in range(self.n_cameras):
                    js = sim_to_goal[c].get(f, [])
                    if not (js or recording):
                        continue  # this view needs nothing from this frame
                    # Render only for a pixel loss or a recorded frame.
                    if loss.wants_render or recording:
                        _, frames, sim_mask, masks = self._render(
                            readback=recording or not use_device,
                            cam=c,
                            refit=refit,
                            with_mask=loss.wants_render or recording,
                        )
                        refit = False
                    else:
                        # The projection needs no BVH refit; leave it to the next
                        # view that renders.
                        frames, sim_mask, masks = None, None, None
                    if js:
                        if use_device and accums[c] is None:
                            accums[c] = loss.make_accum(self.n, self.model.device)
                        goals = [reprs[c][j] for j in js]
                        self._score_view(loss, c, goals, crops[c], sim_mask, masks, accums[c], losses[c])
                        if capture:
                            # The change of the running total of world 0 is the score
                            # of this frame, summed over the goal frames it matches.
                            cur = accums[c].totals()[0] if use_device else losses[c][0]
                            self.last_frame_losses[c].append((f / self.fps, cur - prev_total[c]))
                            prev_total[c] = cur
                    if recording:
                        for i in range(self.n):
                            recorded_frames[c][i].append(frames[i])
                            recorded_masks[c][i].append(masks[i])
            self._step(f)

        for c in range(self.n_cameras):
            if accums[c] is not None:
                losses[c] = accums[c].totals()
            n_goal_c = max(1, len(reprs[c]))
            losses[c] = [l / n_goal_c for l in losses[c]]
        if multi:
            return losses, recorded_frames, recorded_masks
        return (
            losses[0],
            recorded_frames[0] if recorded_frames is not None else None,
            recorded_masks[0] if recorded_masks is not None else None,
        )
