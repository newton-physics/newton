# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
import operator
from collections.abc import Sequence
from typing import Literal

import warp as wp

from ..math import quat_between_vectors_robust


@wp.kernel
def _set_indexed_body_transforms(
    body_indices: wp.array[wp.int32],
    transforms: wp.array[wp.transform],
    body_q: wp.array[wp.transform],
):
    tid = wp.tid()
    body_q[body_indices[tid]] = transforms[tid]


@wp.kernel
def _set_indexed_body_transforms_and_zero_velocities(
    body_indices: wp.array[wp.int32],
    transforms: wp.array[wp.transform],
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
):
    tid = wp.tid()
    body_id = body_indices[tid]
    body_q[body_id] = transforms[tid]
    body_qd[body_id] = wp.spatial_vector(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


def _as_vec3(value, name: str) -> wp.vec3:
    try:
        v = wp.vec3(float(value[0]), float(value[1]), float(value[2]))
    except (IndexError, TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a 3D vector") from exc

    if not all(math.isfinite(float(v[i])) for i in range(3)):
        raise ValueError(f"{name} must contain finite values")
    return v


def _as_points(points: Sequence[wp.vec3], name: str = "points") -> list[wp.vec3]:
    return [_as_vec3(p, f"{name}[{i}]") for i, p in enumerate(points)]


def _as_quat(value, name: str) -> wp.quat:
    try:
        q = wp.quat(float(value[0]), float(value[1]), float(value[2]), float(value[3]))
    except (IndexError, TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a quaternion") from exc

    if not all(math.isfinite(float(q[i])) for i in range(4)):
        raise ValueError(f"{name} must contain finite values")
    return q


def _as_body_indices(body_indices: Sequence[int]) -> list[int]:
    indices = []
    for i, value in enumerate(body_indices):
        if isinstance(value, bool):
            raise TypeError(f"body_indices[{i}] must be an integer body index")
        try:
            indices.append(operator.index(value))
        except TypeError as exc:
            raise TypeError(f"body_indices[{i}] must be an integer body index") from exc

    if not indices:
        raise ValueError("body_indices must contain at least one body index")
    if any(i < 0 for i in indices):
        raise ValueError("body_indices must be non-negative")
    if len(set(indices)) != len(indices):
        raise ValueError("body_indices must be unique")
    return indices


def _as_state_sequence(states) -> list[object]:
    if hasattr(states, "body_q"):
        return [states]
    return list(states)


def _validate_body_indices_in_bounds(indices: list[int], array: wp.array, name: str) -> None:
    count = array.shape[0]
    max_index = max(indices)
    if max_index >= count:
        raise ValueError(f"body_indices contain index {max_index}, but {name} has length {count}")


def _validate_quaternions_align_segments(points: list[wp.vec3], quats: list[wp.quat]) -> None:
    for i, q in enumerate(quats):
        seg = points[i + 1] - points[i]
        seg_len = float(wp.length(seg))
        if seg_len <= 0.0:
            raise ValueError("points must not contain duplicate consecutive points")

        seg_dir = seg / seg_len
        local_z_world = wp.quat_rotate(q, wp.vec3(0.0, 0.0, 1.0))
        local_z_len = float(wp.length(local_z_world))
        if local_z_len <= 0.0 or not math.isfinite(local_z_len):
            raise ValueError(f"quaternions[{i}] must rotate local +Z to a finite direction")

        alignment = float(wp.dot(seg_dir, local_z_world / local_z_len))
        if not math.isfinite(alignment) or alignment < 0.999:
            raise ValueError(
                "quaternions must align local +Z with the matching cable segment direction; "
                f"quaternions[{i}] has alignment {alignment:.6g}"
            )


def create_cable_stiffness_from_elastic_moduli(
    youngs_modulus: float,
    radius: float,
    segment_length: float,
) -> tuple[float, float]:
    """Create per-joint rod/cable stiffness parameters from elastic moduli.

    For a circular cross-section, this computes material stiffnesses and converts them to the
    per-joint stiffness values expected by ``ModelBuilder.add_rod()`` and
    ``ModelBuilder.add_rod_graph()``:

    - stretch_stiffness = E * A / L  [N/m]
    - bend_stiffness = E * I / L     [N*m]

    where:
    - A = pi * r^2
    - I = (pi * r^4) / 4  (area moment of inertia for a solid circular rod)
    - L = segment_length

    Args:
        youngs_modulus: Young's modulus E in Pascals [N/m^2].
        radius: Rod/cable radius r in meters.
        segment_length: Segment length L in meters.

    Returns:
        Tuple `(stretch_stiffness, bend_stiffness)` = `(E*A/L, E*I/L)`.
    """
    # Accept ints / numpy scalars, but return plain Python floats.
    E = float(youngs_modulus)
    r = float(radius)
    L = float(segment_length)

    if not math.isfinite(E):
        raise ValueError("youngs_modulus must be finite")
    if not math.isfinite(r):
        raise ValueError("radius must be finite")
    if not math.isfinite(L):
        raise ValueError("segment_length must be finite")

    if E < 0.0:
        raise ValueError("youngs_modulus must be >= 0")
    if r <= 0.0:
        raise ValueError("radius must be > 0")
    if L <= 0.0:
        raise ValueError("segment_length must be > 0")

    area = math.pi * r * r
    inertia = 0.25 * math.pi * r**4

    return E * area / L, E * inertia / L


def create_straight_cable_points(
    start: wp.vec3,
    direction: wp.vec3,
    length: float,
    num_segments: int,
) -> list[wp.vec3]:
    """Create straight cable polyline points.

    This is a convenience helper for constructing ``positions`` inputs for ``ModelBuilder.add_rod``.

    Args:
        start: First point in world space.
        direction: World-space direction of the cable (need not be normalized).
        length: Total length of the cable (meters).
        num_segments: Number of segments (edges). The number of points is ``num_segments + 1``.

    Returns:
        List of ``wp.vec3`` points of length ``num_segments + 1``.
    """
    if num_segments < 1:
        raise ValueError("num_segments must be >= 1")
    length_m = float(length)
    if not math.isfinite(length_m):
        raise ValueError("length must be finite")
    if length_m < 0.0:
        raise ValueError("length must be >= 0")

    dir_len = float(wp.length(direction))
    if dir_len <= 0.0:
        raise ValueError("direction must be non-zero")
    d = direction / dir_len

    ds = length_m / num_segments
    return [start + d * (ds * i) for i in range(num_segments + 1)]


def compute_cable_segment_lengths(points: Sequence[wp.vec3]) -> list[float]:
    """Compute consecutive segment lengths for a cable polyline.

    Args:
        points: Polyline points of length >= 2.

    Returns:
        List of per-segment lengths of length ``len(points) - 1``.
    """
    points_wp = _as_points(points)
    if len(points_wp) < 2:
        raise ValueError("points must have length >= 2")

    lengths: list[float] = []
    for i in range(len(points_wp) - 1):
        length = float(wp.length(points_wp[i + 1] - points_wp[i]))
        if not math.isfinite(length):
            raise ValueError("points must contain finite values")
        if length <= 0.0:
            raise ValueError("points must not contain duplicate consecutive points")
        lengths.append(length)
    return lengths


def check_matching_cable_segment_lengths(
    points_a: Sequence[wp.vec3],
    points_b: Sequence[wp.vec3],
    *,
    rtol: float = 1.0e-5,
    atol: float = 1.0e-7,
) -> None:
    """Check that two cable polylines have matching segment lengths.

    This is useful when rest and initial cable shapes differ but should start without stretch strain.

    Args:
        points_a: First cable polyline points.
        points_b: Second cable polyline points.
        rtol: Relative tolerance for each segment length.
        atol: Absolute tolerance for each segment length.

    Raises:
        ValueError: If the polylines have different segment counts or any segment length differs
            beyond ``atol + rtol * abs(length_a)``.
    """
    lengths_a = compute_cable_segment_lengths(points_a)
    lengths_b = compute_cable_segment_lengths(points_b)

    if len(lengths_a) != len(lengths_b):
        raise ValueError(
            "points_a and points_b must have the same number of segments "
            f"(got {len(lengths_a)} and {len(lengths_b)})"
        )

    rtol_f = float(rtol)
    atol_f = float(atol)
    if rtol_f < 0.0 or not math.isfinite(rtol_f):
        raise ValueError("rtol must be finite and >= 0")
    if atol_f < 0.0 or not math.isfinite(atol_f):
        raise ValueError("atol must be finite and >= 0")

    for i, (length_a, length_b) in enumerate(zip(lengths_a, lengths_b, strict=True)):
        error = abs(length_b - length_a)
        tolerance = atol_f + rtol_f * abs(length_a)
        if error > tolerance:
            raise ValueError(
                "cable segment lengths differ at segment "
                f"{i}: length_a={length_a:.9g}, length_b={length_b:.9g}, tolerance={tolerance:.3g}"
            )


def create_straight_cable_points_from_segment_lengths(
    start: wp.vec3,
    direction: wp.vec3,
    segment_lengths: Sequence[float],
) -> list[wp.vec3]:
    """Create straight cable points with explicit per-segment rest lengths.

    This is useful when a cable's initial shape is curved but its rest shape should be straight
    with matching segment lengths.

    Args:
        start: First point in world space.
        direction: World-space direction of the straight rest cable (need not be normalized).
        segment_lengths: Positive per-segment lengths.

    Returns:
        List of ``wp.vec3`` points of length ``len(segment_lengths) + 1``.
    """
    start_wp = _as_vec3(start, "start")
    direction_wp = _as_vec3(direction, "direction")

    dir_len = float(wp.length(direction_wp))
    if dir_len <= 0.0:
        raise ValueError("direction must be non-zero")
    d = direction_wp / dir_len

    lengths = [float(length) for length in segment_lengths]
    if not lengths:
        raise ValueError("segment_lengths must contain at least one length")
    for i, length in enumerate(lengths):
        if not math.isfinite(length):
            raise ValueError(f"segment_lengths[{i}] must be finite")
        if length <= 0.0:
            raise ValueError(f"segment_lengths[{i}] must be > 0")

    points = [start_wp]
    distance = 0.0
    for length in lengths:
        distance += length
        points.append(start_wp + d * distance)
    return points


def create_parallel_transport_cable_quaternions(
    points: Sequence[wp.vec3],
    *,
    twist_total: float = 0.0,
) -> list[wp.quat]:
    """Generate per-segment quaternions using a parallel-transport style construction.

    The intended use is for rod/cable capsules whose internal axis is local +Z.
    The returned quaternions rotate local +Z to each segment direction,
    while minimizing twist between successive segments. Optionally, a total twist can be
    distributed uniformly along the cable.

    Args:
        points: Polyline points of length >= 2.
        twist_total: Total twist (radians) distributed along the cable (applied about the segment direction).

    Returns:
        List of ``wp.quat`` of length ``len(points) - 1``.
    """
    if len(points) < 2:
        raise ValueError("points must have length >= 2")

    from_direction = wp.vec3(0.0, 0.0, 1.0)

    num_segments = len(points) - 1
    twist_total_rad = float(twist_total)
    twist_step = (twist_total_rad / num_segments) if twist_total_rad != 0.0 else 0.0
    eps = 1.0e-8

    quats: list[wp.quat] = []
    for i in range(num_segments):
        p0 = points[i]
        p1 = points[i + 1]
        seg = p1 - p0
        seg_len = float(wp.length(seg))
        if seg_len <= 0.0:
            raise ValueError("points must not contain duplicate consecutive points")
        to_direction = seg / seg_len

        # Robustly handle the anti-parallel (180-degree) case, e.g. +Z -> -Z.
        dq_dir = quat_between_vectors_robust(from_direction, to_direction, eps)

        q = dq_dir if i == 0 else wp.mul(dq_dir, quats[i - 1])

        if twist_total_rad != 0.0:
            twist_q = wp.quat_from_axis_angle(to_direction, twist_step)
            q = wp.mul(twist_q, q)

        quats.append(q)
        from_direction = to_direction

    return quats


def create_cable_body_transforms(
    points: Sequence[wp.vec3],
    quaternions: Sequence[wp.quat] | None = None,
    *,
    body_frame_origin: Literal["start", "com"],
    twist_total: float = 0.0,
) -> list[wp.transform]:
    """Create rod body transforms from cable points and per-segment quaternions.

    The returned transforms use the requested body-frame origin convention and orient each
    segment body so local +Z follows the corresponding cable segment.

    Args:
        points: Cable centerline points (segment endpoints) of length >= 2.
        quaternions: Optional per-segment orientations. If omitted, parallel-transport
            orientations are generated from ``points``.
        body_frame_origin: Location of each segment body frame. Choose this to match the
            convention used when the cable bodies were created. ``"start"`` places body ``i``
            at ``points[i]``. ``"com"`` places body ``i`` at the segment midpoint.
        twist_total: Optional total twist used only when ``quaternions`` is omitted.

    Returns:
        List of ``wp.transform`` values of length ``len(points) - 1``.
    """
    points_wp = _as_points(points)
    if len(points_wp) < 2:
        raise ValueError("points must have length >= 2")

    if quaternions is None:
        quats = create_parallel_transport_cable_quaternions(points_wp, twist_total=twist_total)
    else:
        quats = [_as_quat(q, f"quaternions[{i}]") for i, q in enumerate(quaternions)]
        expected = len(points_wp) - 1
        if len(quats) != expected:
            raise ValueError(f"quaternions must have {expected} elements, got {len(quats)}")
        _validate_quaternions_align_segments(points_wp, quats)

    if body_frame_origin == "start":
        origins = points_wp[:-1]
    elif body_frame_origin == "com":
        origins = [points_wp[i] + 0.5 * (points_wp[i + 1] - points_wp[i]) for i in range(len(quats))]
    else:
        raise ValueError('body_frame_origin must be "start" or "com"')

    return [wp.transform(origins[i], quats[i]) for i in range(len(quats))]


def create_straight_cable_rest_from_initial(
    initial_points: Sequence[wp.vec3],
    *,
    start: wp.vec3 | None = None,
    direction: wp.vec3 | None = None,
    twist_total: float = 0.0,
) -> tuple[list[wp.vec3], list[wp.quat]]:
    """Create a straight rest cable that preserves initial per-segment lengths.

    Use this when the model/rest cable should be straight, but the simulation starts from
    a curved cable pose. Matching segment lengths avoids unintended initial stretch while
    still allowing initial bend; it also keeps capsule sizing and body-frame offsets
    consistent between the rest and initial poses.

    Args:
        initial_points: Initial cable centerline points (segment endpoints).
        start: Optional start point for the straight rest cable. Defaults to
            ``initial_points[0]``.
        direction: Optional straight rest direction. Defaults to the chord from first to
            last point, falling back to the first segment direction when the chord is zero.
        twist_total: Total twist used to generate rest quaternions.

    Returns:
        Pair ``(rest_points, rest_quaternions)`` suitable for ``ModelBuilder.add_rod()``.
    """
    points_wp = _as_points(initial_points, "initial_points")
    lengths = compute_cable_segment_lengths(points_wp)

    start_wp = points_wp[0] if start is None else _as_vec3(start, "start")
    if direction is None:
        direction_wp = points_wp[-1] - points_wp[0]
        if float(wp.length(direction_wp)) <= 0.0:
            direction_wp = points_wp[1] - points_wp[0]
    else:
        direction_wp = _as_vec3(direction, "direction")

    rest_points = create_straight_cable_points_from_segment_lengths(start_wp, direction_wp, lengths)
    rest_quats = create_parallel_transport_cable_quaternions(rest_points, twist_total=twist_total)
    return rest_points, rest_quats


def create_straight_cable_points_and_quaternions(
    start: wp.vec3,
    direction: wp.vec3,
    length: float,
    num_segments: int,
    *,
    twist_total: float = 0.0,
) -> tuple[list[wp.vec3], list[wp.quat]]:
    """Generate straight cable points and matching per-segment quaternions.

    This is a convenience wrapper around:
    - :func:`create_straight_cable_points`
    - :func:`create_parallel_transport_cable_quaternions`
    """
    points = create_straight_cable_points(
        start=start,
        direction=direction,
        length=length,
        num_segments=num_segments,
    )
    quats = create_parallel_transport_cable_quaternions(points, twist_total=twist_total)
    return points, quats


def apply_cable_body_transforms(
    states,
    body_indices: Sequence[int],
    transforms: Sequence[wp.transform],
    *,
    body_q_prev: wp.array[wp.transform] | None = None,
    zero_velocities: bool = True,
) -> None:
    """Apply cable body transforms to states and optional previous-pose storage.

    Use this when a cable's simulation initial pose differs from its model/rest pose. It updates
    only the requested bodies and, when ``body_q_prev`` is provided, keeps previous-pose history
    in sync so the first step does not see an artificial teleportation. For
    :class:`~newton.solvers.SolverVBD`, pass ``solver.body_q_prev`` for best first-step behavior.

    Args:
        states: A single :class:`~newton.State` or a sequence of states whose ``body_q`` arrays
            should be updated.
        body_indices: Body indices corresponding to ``transforms``.
        transforms: Initial body transforms, one per body index.
        body_q_prev: Optional previous-pose buffer. For :class:`~newton.solvers.SolverVBD`,
            pass ``solver.body_q_prev`` so VBD's previous-body-pose history matches the applied
            initial pose.
        zero_velocities: If True, zero matching ``state.body_qd`` entries so the applied pose starts
            at rest.

    Example:
        Build a straight rest cable whose segment lengths match a curved initial pose, then
        push that initial pose into states and the VBD previous-pose history::

            rest_points, rest_quats = newton.utils.create_straight_cable_rest_from_initial(initial_points)
            body_ids, _ = builder.add_rod(positions=rest_points, quaternions=rest_quats, radius=0.02)
            model = builder.finalize()
            state_0, state_1 = model.state(), model.state()
            solver = newton.solvers.SolverVBD(model)

            # Use the same body_frame_origin convention that was used to create the cable bodies;
            # "start" is shown here as one example convention.
            initial_xforms = newton.utils.create_cable_body_transforms(initial_points, body_frame_origin="start")
            newton.utils.apply_cable_body_transforms(
                [state_0, state_1],
                body_ids,
                initial_xforms,
                body_q_prev=solver.body_q_prev,
            )
    """
    states_list = _as_state_sequence(states)
    indices = _as_body_indices(body_indices)
    xforms = list(transforms)
    if len(xforms) != len(indices):
        raise ValueError(f"transforms must have {len(indices)} elements, got {len(xforms)}")
    if not states_list and body_q_prev is None:
        raise ValueError("states must contain at least one state when body_q_prev is not provided")

    n = len(indices)
    # Upload the (index, transform) arrays once per device and reuse them across target buffers
    # instead of re-creating identical arrays per launch.
    device_arrays = {}

    def arrays_for_device(device):
        key = str(device)
        cached = device_arrays.get(key)
        if cached is None:
            cached = (
                wp.array(indices, dtype=wp.int32, device=device),
                wp.array(xforms, dtype=wp.transform, device=device),
            )
            device_arrays[key] = cached
        return cached

    for state_index, state in enumerate(states_list):
        body_q = getattr(state, "body_q", None)
        if body_q is None:
            raise ValueError(f"states[{state_index}] does not have a body_q array")
        _validate_body_indices_in_bounds(indices, body_q, f"states[{state_index}].body_q")
        body_indices_wp, transforms_wp = arrays_for_device(body_q.device)

        body_qd = getattr(state, "body_qd", None)
        if zero_velocities and body_qd is not None:
            if str(body_qd.device) != str(body_q.device):
                raise ValueError(
                    f"states[{state_index}].body_qd must be on the same device as states[{state_index}].body_q"
                )
            _validate_body_indices_in_bounds(indices, body_qd, f"states[{state_index}].body_qd")
            wp.launch(
                _set_indexed_body_transforms_and_zero_velocities,
                dim=n,
                inputs=[body_indices_wp, transforms_wp, body_q, body_qd],
                device=body_q.device,
            )
        else:
            wp.launch(
                _set_indexed_body_transforms,
                dim=n,
                inputs=[body_indices_wp, transforms_wp, body_q],
                device=body_q.device,
            )

    if body_q_prev is not None:
        _validate_body_indices_in_bounds(indices, body_q_prev, "body_q_prev")
        prev_indices_wp, prev_transforms_wp = arrays_for_device(body_q_prev.device)
        wp.launch(
            _set_indexed_body_transforms,
            dim=n,
            inputs=[prev_indices_wp, prev_transforms_wp, body_q_prev],
            device=body_q_prev.device,
        )
