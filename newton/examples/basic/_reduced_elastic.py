# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import warp as wp

import newton

BOX_RENDER_MAX_EDGE_LENGTH = 0.01
FRAME_AXES = (
    (np.array([1.0, 0.0, 0.0]), (1.0, 0.25, 0.25)),
    (np.array([0.0, 1.0, 0.0]), (0.25, 1.0, 0.25)),
    (np.array([0.0, 0.0, 1.0]), (0.35, 0.45, 1.0)),
)
ELASTIC_SOLVER_METRIC_KEYS = (
    "initial_residual_norm",
    "solve_residual_norm",
    "applied_residual_norm",
    "update_norm",
    "update_max",
)


def _segment_count(extent: float, edge: float) -> int:
    return max(1, int(np.ceil(float(extent) / edge - 1.0e-5)))


def box_surface_mesh(
    length: float,
    half_width_y: float,
    half_width_z: float,
    *,
    max_edge_length: float = BOX_RENDER_MAX_EDGE_LENGTH,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the subdivided box mesh used by reduced elastic beam examples."""
    half_length = 0.5 * length
    edge = max(float(max_edge_length), 1.0e-6)
    nx = _segment_count(length, edge)
    ny = _segment_count(2.0 * half_width_y, edge)
    nz = _segment_count(2.0 * half_width_z, edge)
    vertices: list[tuple[float, float, float]] = []
    indices: list[int] = []
    grid: dict[tuple[int, int, int], int] = {}

    for i in range(nx + 1):
        x = -half_length + length * (float(i) / float(nx))
        for j in range(ny + 1):
            y = -half_width_y + 2.0 * half_width_y * (float(j) / float(ny))
            for k in range(nz + 1):
                if i not in (0, nx) and j not in (0, ny) and k not in (0, nz):
                    continue
                z = -half_width_z + 2.0 * half_width_z * (float(k) / float(nz))
                grid[(i, j, k)] = len(vertices)
                vertices.append((x, y, z))

    def add_quad(a: int, b: int, c: int, d: int) -> None:
        indices.extend([a, b, c, a, c, d])

    for j in range(ny):
        for k in range(nz):
            add_quad(grid[(0, j, k)], grid[(0, j, k + 1)], grid[(0, j + 1, k + 1)], grid[(0, j + 1, k)])
            add_quad(
                grid[(nx, j, k)],
                grid[(nx, j + 1, k)],
                grid[(nx, j + 1, k + 1)],
                grid[(nx, j, k + 1)],
            )

    for i in range(nx):
        for k in range(nz):
            add_quad(grid[(i, 0, k)], grid[(i + 1, 0, k)], grid[(i + 1, 0, k + 1)], grid[(i, 0, k + 1)])
            add_quad(
                grid[(i, ny, k)],
                grid[(i, ny, k + 1)],
                grid[(i + 1, ny, k + 1)],
                grid[(i + 1, ny, k)],
            )

    for i in range(nx):
        for j in range(ny):
            add_quad(grid[(i, j, 0)], grid[(i, j + 1, 0)], grid[(i + 1, j + 1, 0)], grid[(i + 1, j, 0)])
            add_quad(
                grid[(i, j, nz)],
                grid[(i + 1, j, nz)],
                grid[(i + 1, j + 1, nz)],
                grid[(i, j + 1, nz)],
            )

    return np.asarray(vertices, dtype=np.float32), np.asarray(indices, dtype=np.int32)


def beam_render_sample_points(
    length: float,
    half_width_y: float,
    half_width_z: float,
    *,
    extra_points: Sequence[Sequence[float]] = (),
    max_edge_length: float = BOX_RENDER_MAX_EDGE_LENGTH,
    diagnostic_count: int = 33,
) -> np.ndarray:
    """Return modal sample points covering rendered beam vertices and overlays."""
    surface_vertices, _ = box_surface_mesh(length, half_width_y, half_width_z, max_edge_length=max_edge_length)
    diagnostic_points = np.column_stack(
        (
            np.linspace(-0.5 * length, 0.5 * length, diagnostic_count, dtype=np.float32),
            np.zeros(diagnostic_count, dtype=np.float32),
            np.full(diagnostic_count, half_width_z + 0.045, dtype=np.float32),
        )
    )
    point_sets = [surface_vertices, diagnostic_points]
    if extra_points:
        point_sets.append(np.asarray(extra_points, dtype=np.float32).reshape((-1, 3)))
    return np.vstack(point_sets).astype(np.float32, copy=False)


def poisson_axial_mode(
    points: np.ndarray, length: float, poisson_ratio: float, *, clamped_ends: bool = False
) -> np.ndarray:
    """Return axial extension with Poisson lateral strain [m per m]."""
    points = np.asarray(points, dtype=np.float32)
    phi = np.zeros_like(points, dtype=np.float32)
    inv_length = 1.0 / float(length)
    lateral_profile = 1.0
    if clamped_ends:
        xi = np.clip(2.0 * points[:, 0] * inv_length, -1.0, 1.0)
        lateral_profile = 1.5 * (1.0 - xi * xi)
    phi[:, 0] = points[:, 0] * inv_length
    phi[:, 1] = -float(poisson_ratio) * points[:, 1] * inv_length * lateral_profile
    phi[:, 2] = -float(poisson_ratio) * points[:, 2] * inv_length * lateral_profile
    return phi


def set_camera_from_bounds(viewer, bounds_min: np.ndarray, bounds_max: np.ndarray, offset_dir: np.ndarray):
    """Aim the viewer camera at a bounding box along a normalized offset direction."""
    center = 0.5 * (bounds_min + bounds_max)
    extent = float(np.max(bounds_max - bounds_min))
    distance = max(extent, 1.0) / (2.0 * math.tan(math.radians(45.0) * 0.5)) * 1.35
    offset_dir = offset_dir / np.linalg.norm(offset_dir)
    pos = center + offset_dir * distance
    front = center - pos
    front /= np.linalg.norm(front)
    yaw = math.degrees(math.atan2(front[1], front[0]))
    pitch = math.degrees(math.asin(front[2]))
    viewer.set_camera(wp.vec3(*pos), pitch, yaw)


def find_free_joint_q_start(model: newton.Model, body: int) -> tuple[int, int]:
    """Return the joint coordinate and velocity start indices of a body's free joint."""
    joint_parent = model.joint_parent.numpy()
    joint_child = model.joint_child.numpy()
    joint_q_start = model.joint_q_start.numpy()
    joint_qd_start = model.joint_qd_start.numpy()
    for j in range(len(joint_child)):
        if int(joint_child[j]) == body and int(joint_parent[j]) == -1:
            return int(joint_q_start[j]), int(joint_qd_start[j])
    raise RuntimeError(f"No free joint found for body {body}")


def cantilever_tip_mode(points: np.ndarray, length: float) -> np.ndarray:
    """Cubic cantilever tip mode: phi(0)=0 at the clamp, phi(L)=1 at the tip."""
    points = np.asarray(points, dtype=np.float32)
    s = np.clip(points[:, 0] + 0.5 * length, 0.0, length)
    phi = (s * s * (3.0 * length - s)) / (2.0 * length**3)
    slope = (3.0 * s * (2.0 * length - s)) / (2.0 * length**3)
    out = np.zeros_like(points, dtype=np.float32)
    out[:, 0] = -points[:, 2] * slope
    out[:, 2] = phi
    return out


def quat_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate a vector by a quaternion in xyzw order."""
    qv = np.asarray(q[:3], dtype=float)
    v = np.asarray(v, dtype=float)
    return v + 2.0 * np.cross(qv, np.cross(qv, v) + float(q[3]) * v)


def transform_point(xform: np.ndarray, point: np.ndarray) -> np.ndarray:
    """Transform a point by a Newton transform stored as position + xyzw quaternion."""
    return np.asarray(xform[:3], dtype=float) + quat_rotate(np.asarray(xform[3:7], dtype=float), point)


def elastic_solver_metric_snapshot(solver) -> dict[str, float]:
    """Return max-per-body reduced elastic modal solve metrics for one solver step."""
    if not hasattr(solver, "elastic_mode_solve_metrics"):
        return dict.fromkeys(ELASTIC_SOLVER_METRIC_KEYS, 0.0)

    metrics = solver.elastic_mode_solve_metrics()
    snapshot = {}
    for key in ELASTIC_SOLVER_METRIC_KEYS:
        values = np.asarray(metrics.get(key, ()), dtype=float)
        snapshot[key] = float(np.max(values)) if values.size > 0 else 0.0
    residual_scale = max(snapshot["initial_residual_norm"], 1.0e-12)
    snapshot["solve_residual_ratio"] = snapshot["solve_residual_norm"] / residual_scale
    snapshot["applied_residual_ratio"] = snapshot["applied_residual_norm"] / residual_scale
    return snapshot


def init_elastic_solver_metric_tracking(example) -> None:
    """Initialize common reduced elastic solver metric fields on an example."""
    example.final_modal_initial_residual_norm = 0.0
    example.final_modal_solve_residual_norm = 0.0
    example.final_modal_applied_residual_norm = 0.0
    example.final_modal_solve_residual_ratio = 0.0
    example.final_modal_applied_residual_ratio = 0.0
    example.final_modal_update_norm = 0.0
    example.final_modal_update_max = 0.0
    example.max_modal_initial_residual_norm = 0.0
    example.max_modal_solve_residual_norm = 0.0
    example.max_modal_applied_residual_norm = 0.0
    example.max_modal_solve_residual_ratio = 0.0
    example.max_modal_applied_residual_ratio = 0.0
    example.max_modal_update_norm = 0.0
    example.max_modal_update_max = 0.0


def update_elastic_solver_metric_tracking(example, solver=None) -> None:
    """Update common reduced elastic solver metric fields on an example."""
    if solver is None:
        solver = example.solver

    snapshot = elastic_solver_metric_snapshot(solver)
    example.final_modal_initial_residual_norm = snapshot["initial_residual_norm"]
    example.final_modal_solve_residual_norm = snapshot["solve_residual_norm"]
    example.final_modal_applied_residual_norm = snapshot["applied_residual_norm"]
    example.final_modal_solve_residual_ratio = snapshot["solve_residual_ratio"]
    example.final_modal_applied_residual_ratio = snapshot["applied_residual_ratio"]
    example.final_modal_update_norm = snapshot["update_norm"]
    example.final_modal_update_max = snapshot["update_max"]
    example.max_modal_initial_residual_norm = max(
        example.max_modal_initial_residual_norm, snapshot["initial_residual_norm"]
    )
    example.max_modal_solve_residual_norm = max(example.max_modal_solve_residual_norm, snapshot["solve_residual_norm"])
    example.max_modal_applied_residual_norm = max(
        example.max_modal_applied_residual_norm, snapshot["applied_residual_norm"]
    )
    example.max_modal_solve_residual_ratio = max(
        example.max_modal_solve_residual_ratio, snapshot["solve_residual_ratio"]
    )
    example.max_modal_applied_residual_ratio = max(
        example.max_modal_applied_residual_ratio, snapshot["applied_residual_ratio"]
    )
    example.max_modal_update_norm = max(example.max_modal_update_norm, snapshot["update_norm"])
    example.max_modal_update_max = max(example.max_modal_update_max, snapshot["update_max"])


def joint_endpoint_world(model, state, joint: int, side: str) -> np.ndarray:
    """Return a joint endpoint in world space, including reduced elastic displacement."""
    if side == "parent":
        body = int(model.joint_parent.numpy()[joint])
        xforms = model.joint_X_p.numpy()
        endpoint = int(model.joint_parent_elastic_endpoint.numpy()[joint])
    elif side == "child":
        body = int(model.joint_child.numpy()[joint])
        xforms = model.joint_X_c.numpy()
        endpoint = int(model.joint_child_elastic_endpoint.numpy()[joint])
    else:
        raise ValueError(f"side must be 'parent' or 'child', got {side!r}")

    local = np.array(xforms[joint, :3], dtype=float)
    if endpoint >= 0:
        elastic = int(model.body_elastic_index.numpy()[body])
        owner_joint = int(model.elastic_joint.numpy()[elastic])
        q_start = int(model.joint_q_start.numpy()[owner_joint])
        mode_count = int(model.elastic_mode_count.numpy()[elastic])
        max_modes = int(model.elastic_max_mode_count)
        phi = model.elastic_endpoint_phi.numpy().reshape((-1, max_modes, 3))
        q = state.joint_q.numpy()
        local += np.einsum("mc,m->c", phi[endpoint, :mode_count], q[q_start + 7 : q_start + 7 + mode_count])

    if body < 0:
        return local
    return transform_point(state.body_q.numpy()[body], local)


def finite_torsion_mode_fields(points: np.ndarray, length: float, tip_twist: float) -> tuple[np.ndarray, np.ndarray]:
    """Return circumferential and radial finite-twist displacement fields."""
    points = np.asarray(points, dtype=np.float32)
    s = np.clip(points[:, 0] + 0.5 * length, 0.0, length)
    theta = float(tip_twist) * s / float(length)
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)

    y = points[:, 1]
    z = points[:, 2]
    twist = np.zeros_like(points, dtype=np.float32)
    twist[:, 1] = -z * sin_theta
    twist[:, 2] = y * sin_theta

    radial = np.zeros_like(points, dtype=np.float32)
    radial[:, 1] = y * (cos_theta - 1.0)
    radial[:, 2] = z * (cos_theta - 1.0)
    return twist, radial


def finite_torsion_displacement(points: np.ndarray, length: float, tip_twist: float) -> np.ndarray:
    """Return exact finite-rotation twist displacements for beam-local points."""
    twist, radial = finite_torsion_mode_fields(points, length, tip_twist)
    return twist + radial


def beam_torsion_linear_modal_properties(
    length: float,
    half_width_y: float,
    half_width_z: float,
    density: float,
    shear_modulus: float,
) -> tuple[float, float]:
    """Return approximate modal mass and stiffness for a linear rectangular-beam twist."""
    length = float(length)
    if length <= 0.0:
        raise ValueError(f"length must be positive, got {length}")

    hy = float(half_width_y)
    hz = float(half_width_z)
    polar_moment = (4.0 / 3.0) * hy * hz**3 + (4.0 / 3.0) * hz * hy**3
    modal_mass = float(density) * polar_moment * length / 3.0
    modal_stiffness = float(shear_modulus) * polar_moment / length
    return modal_mass, modal_stiffness


def mesh_volume(vertices: np.ndarray, indices: np.ndarray) -> float:
    """Return the absolute volume enclosed by a closed triangle mesh."""
    vertices = np.asarray(vertices, dtype=np.float64)
    triangles = vertices[np.asarray(indices, dtype=np.int32).reshape((-1, 3))]
    signed = np.einsum("ij,ij->i", triangles[:, 0], np.cross(triangles[:, 1], triangles[:, 2]))
    return abs(float(np.sum(signed) / 6.0))


def elastic_shape_deformed_vertices(model, state, elastic_shape: int = 0) -> np.ndarray:
    """Return body-local deformed render vertices for one reduced elastic shape."""
    shape_count = int(getattr(model, "elastic_shape_count", 0))
    if elastic_shape < 0 or elastic_shape >= shape_count:
        raise IndexError(f"elastic_shape index {elastic_shape} out of range for {shape_count} shapes")

    vertex_start = model.elastic_shape_vertex_start.numpy()
    vertex_count = model.elastic_shape_vertex_count.numpy()
    elastic_shape_body = model.elastic_shape_body.numpy()
    body_elastic_index = model.body_elastic_index.numpy()
    elastic_mode_count = model.elastic_mode_count.numpy()
    elastic_joint = model.elastic_joint.numpy()
    joint_q_start = model.joint_q_start.numpy()

    start = int(vertex_start[elastic_shape])
    count = int(vertex_count[elastic_shape])
    body = int(elastic_shape_body[elastic_shape])
    elastic_index = int(body_elastic_index[body])
    mode_count = int(elastic_mode_count[elastic_index])
    q_start = int(joint_q_start[int(elastic_joint[elastic_index])]) + 7

    vertices = np.array(model.elastic_shape_vertex_local.numpy()[start : start + count], dtype=np.float64)
    if mode_count == 0:
        return vertices

    max_modes = int(model.elastic_max_mode_count)
    phi = model.elastic_shape_vertex_phi.numpy().reshape((-1, max_modes, 3))[start : start + count, :mode_count]
    q = state.joint_q.numpy()[q_start : q_start + mode_count]
    vertices += np.einsum("vmc,m->vc", phi, q)
    return vertices


def elastic_shape_volume_ratio(model, state, elastic_shape: int = 0) -> float:
    """Return deformed/rest volume for one reduced elastic render shape."""
    vertex_start = model.elastic_shape_vertex_start.numpy()
    vertex_count = model.elastic_shape_vertex_count.numpy()
    index_start = model.elastic_shape_index_start.numpy()
    index_count = model.elastic_shape_index_count.numpy()

    v_start = int(vertex_start[elastic_shape])
    v_count = int(vertex_count[elastic_shape])
    i_start = int(index_start[elastic_shape])
    i_count = int(index_count[elastic_shape])

    rest_vertices = np.array(model.elastic_shape_vertex_local.numpy()[v_start : v_start + v_count], dtype=np.float64)
    indices = model.elastic_shape_indices.numpy()[i_start : i_start + i_count]
    rest_volume = mesh_volume(rest_vertices, indices)
    if rest_volume <= 0.0:
        raise ValueError("elastic shape rest mesh has non-positive volume")

    deformed_vertices = elastic_shape_deformed_vertices(model, state, elastic_shape)
    return mesh_volume(deformed_vertices, indices) / rest_volume
