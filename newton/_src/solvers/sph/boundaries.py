# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import warp as wp

from ...geometry import GeoType, ParticleFlags, ShapeFlags
from ...sim import Model, State
from .kernels import sph_same_world_np as _same_world_np
from .shape_boundary_kernels import SPH_COLLIDER_VELOCITY_BACKWARD, SPH_COLLIDER_VELOCITY_FORWARD
from .shape_boundary_kernels import collide_particle_shapes as _collide_particle_shapes
from .sph_model import SPHRole

_SPH_ANALYTIC_BOUNDARY_SHAPE_TYPES = (
    "plane",
    "sphere",
    "box",
    "capsule",
    "cylinder",
    "ellipsoid",
    "cone",
    "mesh",
    "convex_mesh",
)


def _quat_rotate_np(quat_xyzw: np.ndarray, vec: np.ndarray) -> np.ndarray:
    qv = np.asarray(quat_xyzw[:3], dtype=np.float32)
    qw = float(quat_xyzw[3])
    return vec + 2.0 * qw * np.cross(qv, vec) + 2.0 * np.cross(qv, np.cross(qv, vec))


def _quat_multiply_np(left_xyzw: np.ndarray, right_xyzw: np.ndarray) -> np.ndarray:
    left = np.asarray(left_xyzw, dtype=np.float32)
    right = np.asarray(right_xyzw, dtype=np.float32)
    lx, ly, lz, lw = (float(left[0]), float(left[1]), float(left[2]), float(left[3]))
    rx, ry, rz, rw = (float(right[0]), float(right[1]), float(right[2]), float(right[3]))
    return np.asarray(
        (
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
            lw * rw - lx * rx - ly * ry - lz * rz,
        ),
        dtype=np.float32,
    )


def _quat_normalize_np(quat_xyzw: np.ndarray, *, eps: float = 1.0e-12) -> np.ndarray:
    quat = np.asarray(quat_xyzw, dtype=np.float32)
    norm = float(np.linalg.norm(quat))
    if norm <= float(eps):
        return np.asarray((0.0, 0.0, 0.0, 1.0), dtype=np.float32)
    return (quat / norm).astype(np.float32)


def _integrate_orientation_np(quat_xyzw: np.ndarray, angular_velocity: np.ndarray, dt: float) -> np.ndarray:
    omega = np.asarray(angular_velocity, dtype=np.float32)
    omega_norm = float(np.linalg.norm(omega))
    if omega_norm <= 1.0e-12 or dt == 0.0:
        return _quat_normalize_np(quat_xyzw)
    angle = omega_norm * float(dt)
    axis = omega / omega_norm
    half_angle = 0.5 * angle
    delta = np.asarray(
        (
            axis[0] * np.sin(half_angle),
            axis[1] * np.sin(half_angle),
            axis[2] * np.sin(half_angle),
            np.cos(half_angle),
        ),
        dtype=np.float32,
    )
    return _quat_normalize_np(_quat_multiply_np(delta, quat_xyzw))


def _quat_rotate_inv_np(quat_xyzw: np.ndarray, vec: np.ndarray) -> np.ndarray:
    q_inv = np.asarray((-quat_xyzw[0], -quat_xyzw[1], -quat_xyzw[2], quat_xyzw[3]), dtype=np.float32)
    return _quat_rotate_np(q_inv, vec)


def _quat_velocity_np(quat_now_xyzw: np.ndarray, quat_prev_xyzw: np.ndarray, dt: float) -> np.ndarray:
    if dt <= 0.0:
        return np.zeros(3, dtype=np.float32)
    q1 = _quat_normalize_np(quat_now_xyzw)
    q0 = _quat_normalize_np(quat_prev_xyzw)
    if float(np.dot(q1, q0)) < 0.0:
        q0 = -q0
    q0_inv = np.asarray((-q0[0], -q0[1], -q0[2], q0[3]), dtype=np.float32)
    dq = _quat_normalize_np(_quat_multiply_np(q1, q0_inv))
    axis_scaled = dq[:3]
    axis_norm = float(np.linalg.norm(axis_scaled))
    if axis_norm <= 1.0e-12:
        return np.zeros(3, dtype=np.float32)
    angle = 2.0 * math.atan2(axis_norm, float(np.clip(dq[3], -1.0, 1.0)))
    return (axis_scaled / axis_norm * (angle / float(dt))).astype(np.float32)


@dataclass
class SPHBoundaryHandler:
    """Boundary backend wrapper for SPH sampled and analytic boundaries.

    Sampled boundaries are represented as ordinary Newton particles tagged with
    ``SPHRole.BOUNDARY``. Analytic boundaries use lightweight projection passes
    over supported Newton collision shapes. This wrapper keeps both mechanisms
    in one place so ``SolverWCSPH`` can call a compact boundary backend instead
    of carrying boundary policy through the main step loop.
    """

    model: Model
    enable_shape_boundaries: bool = True
    collider_model: Model | None = None
    explicit_collider_meshes: tuple[object, ...] = ()
    explicit_collider_margins: tuple[float, ...] = ()
    explicit_collider_friction: tuple[float, ...] = ()
    explicit_collider_adhesion: tuple[float, ...] = ()
    explicit_collider_projection_threshold: tuple[float, ...] = ()
    explicit_collider_body_ids: tuple[int, ...] = ()
    explicit_collider_mesh_ids_wp: wp.array[wp.uint64] | None = None
    explicit_collider_margins_wp: wp.array[float] | None = None
    explicit_collider_friction_wp: wp.array[float] | None = None
    explicit_collider_adhesion_wp: wp.array[float] | None = None
    explicit_collider_projection_threshold_wp: wp.array[float] | None = None
    explicit_collider_body_ids_wp: wp.array[wp.int32] | None = None
    model_collider_shape_margin_wp: wp.array[float] | None = None
    model_collider_shape_friction_wp: wp.array[float] | None = None
    model_collider_shape_adhesion_wp: wp.array[float] | None = None
    model_collider_shape_projection_threshold_wp: wp.array[float] | None = None
    previous_collider_body_q_wp: wp.array[wp.transform] | None = None
    last_analytic_body_impulse: np.ndarray | None = None
    last_analytic_body_angular_impulse: np.ndarray | None = None
    analytic_body_impulse_wp: wp.array[wp.vec3] | None = None
    analytic_body_angular_impulse_wp: wp.array[wp.vec3] | None = None

    def __post_init__(self) -> None:
        self._refresh_explicit_collider_mesh_arrays()
        self.set_model_collider_material_overrides(
            None,
            margins=None,
            friction=None,
            adhesion=None,
            projection_threshold=None,
        )
        self._reset_analytic_body_impulses()

    def refresh_model(self) -> None:
        """Refresh caches sized from mutable Newton model topology/properties."""
        self._reset_analytic_body_impulses()

    def set_collider_model(self, model: Model | None) -> None:
        """Use ``model`` as the source of analytic collider shapes."""
        self.collider_model = self.model if model is None else model
        self.set_model_collider_material_overrides(
            None,
            margins=None,
            friction=None,
            adhesion=None,
            projection_threshold=None,
        )
        self.save_collider_current_position(self._collider_model().body_q)
        self._reset_analytic_body_impulses()

    def require_collider_previous_position(self, collider_body_q: wp.array[wp.transform] | None) -> None:
        if collider_body_q is None:
            self.previous_collider_body_q_wp = None
        elif (
            self.previous_collider_body_q_wp is None or self.previous_collider_body_q_wp.shape != collider_body_q.shape
        ):
            self.previous_collider_body_q_wp = wp.clone(collider_body_q)

    def save_collider_current_position(self, collider_body_q: wp.array[wp.transform] | None) -> None:
        self.require_collider_previous_position(collider_body_q)
        if collider_body_q is not None and self.previous_collider_body_q_wp is not None:
            self.previous_collider_body_q_wp.assign(collider_body_q)

    def set_model_collider_material_overrides(
        self,
        collider_body_ids: Sequence[int] | None,
        *,
        margins: Sequence[float | None] | None,
        friction: Sequence[float | None] | None,
        adhesion: Sequence[float | None] | None,
        projection_threshold: Sequence[float | None] | None,
    ) -> None:
        """Build per-shape material arrays for model-owned collider shapes."""
        collider_model = self._collider_model()
        shape_count = int(collider_model.shape_count)
        if shape_count == 0:
            empty = np.zeros(0, dtype=np.float32)
            self.model_collider_shape_margin_wp = wp.array(empty, dtype=wp.float32, device=self.model.device)
            self.model_collider_shape_friction_wp = wp.array(empty, dtype=wp.float32, device=self.model.device)
            self.model_collider_shape_adhesion_wp = wp.array(empty, dtype=wp.float32, device=self.model.device)
            self.model_collider_shape_projection_threshold_wp = wp.array(
                empty,
                dtype=wp.float32,
                device=self.model.device,
            )
            return

        shape_margin = np.asarray(collider_model.shape_margin.numpy(), dtype=np.float32).copy()
        shape_friction = np.asarray(collider_model.shape_material_mu.numpy(), dtype=np.float32).copy()
        shape_adhesion = np.zeros(shape_count, dtype=np.float32)
        shape_projection_threshold = np.zeros(shape_count, dtype=np.float32)
        if collider_body_ids is None:
            collider_body_ids = ()
        body_ids = [int(body) for body in collider_body_ids]
        shape_body = np.asarray(collider_model.shape_body.numpy(), dtype=np.int32)
        shape_flags = np.asarray(collider_model.shape_flags.numpy(), dtype=np.int32)

        if margins is not None:
            if len(margins) != len(body_ids):
                raise ValueError("SPH collider_margins must match collider_body_ids length")
            for body, margin in zip(body_ids, margins, strict=True):
                if margin is None:
                    continue
                if float(margin) < 0.0:
                    raise ValueError("SPH collider_margins must be non-negative")
                mask = (shape_body == body) & ((shape_flags & int(ShapeFlags.COLLIDE_PARTICLES)) != 0)
                shape_margin[mask] = float(margin)

        if friction is not None:
            if len(friction) != len(body_ids):
                raise ValueError("SPH collider_friction must match collider_body_ids length")
            for body, mu in zip(body_ids, friction, strict=True):
                if mu is None:
                    continue
                if float(mu) < 0.0:
                    raise ValueError("SPH collider_friction must be non-negative")
                mask = (shape_body == body) & ((shape_flags & int(ShapeFlags.COLLIDE_PARTICLES)) != 0)
                shape_friction[mask] = float(mu)

        if adhesion is not None:
            if len(adhesion) != len(body_ids):
                raise ValueError("SPH collider_adhesion must match collider_body_ids length")
            for body, value in zip(body_ids, adhesion, strict=True):
                if value is None:
                    continue
                if float(value) < 0.0:
                    raise ValueError("SPH collider_adhesion must be non-negative")
                mask = (shape_body == body) & ((shape_flags & int(ShapeFlags.COLLIDE_PARTICLES)) != 0)
                shape_adhesion[mask] = float(value)

        if projection_threshold is not None:
            if len(projection_threshold) != len(body_ids):
                raise ValueError("SPH collider_projection_threshold must match collider_body_ids length")
            for body, value in zip(body_ids, projection_threshold, strict=True):
                if value is None:
                    continue
                if float(value) < 0.0:
                    raise ValueError("SPH collider_projection_threshold must be non-negative")
                mask = (shape_body == body) & ((shape_flags & int(ShapeFlags.COLLIDE_PARTICLES)) != 0)
                shape_projection_threshold[mask] = float(value)

        self.model_collider_shape_margin_wp = wp.array(shape_margin, dtype=wp.float32, device=self.model.device)
        self.model_collider_shape_friction_wp = wp.array(shape_friction, dtype=wp.float32, device=self.model.device)
        self.model_collider_shape_adhesion_wp = wp.array(shape_adhesion, dtype=wp.float32, device=self.model.device)
        self.model_collider_shape_projection_threshold_wp = wp.array(
            shape_projection_threshold,
            dtype=wp.float32,
            device=self.model.device,
        )

    def set_explicit_collider_meshes(
        self,
        meshes: Sequence[object] | None,
        *,
        body_ids: Sequence[int] | None = None,
        margins: Sequence[float] | None = None,
        friction: Sequence[float] | None = None,
        adhesion: Sequence[float] | None = None,
        projection_threshold: Sequence[float] | None = None,
    ) -> None:
        """Use standalone triangle meshes as SPH colliders."""
        self.explicit_collider_meshes = () if meshes is None else tuple(meshes)
        mesh_count = len(self.explicit_collider_meshes)
        self.explicit_collider_body_ids = (
            tuple(-1 for _ in range(mesh_count)) if body_ids is None else tuple(int(body) for body in body_ids)
        )
        self.explicit_collider_margins = tuple(0.0 for _ in range(mesh_count)) if margins is None else tuple(margins)
        self.explicit_collider_friction = tuple(0.0 for _ in range(mesh_count)) if friction is None else tuple(friction)
        self.explicit_collider_adhesion = tuple(0.0 for _ in range(mesh_count)) if adhesion is None else tuple(adhesion)
        self.explicit_collider_projection_threshold = (
            tuple(0.0 for _ in range(mesh_count)) if projection_threshold is None else tuple(projection_threshold)
        )
        if len(self.explicit_collider_body_ids) != mesh_count:
            raise ValueError("SPH explicit collider body ids must match collider_meshes length")
        if len(self.explicit_collider_margins) != mesh_count:
            raise ValueError("SPH explicit collider mesh margins must match collider_meshes length")
        if len(self.explicit_collider_friction) != mesh_count:
            raise ValueError("SPH explicit collider mesh friction values must match collider_meshes length")
        if len(self.explicit_collider_adhesion) != mesh_count:
            raise ValueError("SPH explicit collider mesh adhesion values must match collider_meshes length")
        if len(self.explicit_collider_projection_threshold) != mesh_count:
            raise ValueError("SPH explicit collider mesh projection thresholds must match collider_meshes length")
        if any(float(value) < 0.0 for value in self.explicit_collider_margins):
            raise ValueError("SPH explicit collider mesh margins must be non-negative")
        if any(float(value) < 0.0 for value in self.explicit_collider_friction):
            raise ValueError("SPH explicit collider mesh friction values must be non-negative")
        if any(float(value) < 0.0 for value in self.explicit_collider_adhesion):
            raise ValueError("SPH explicit collider mesh adhesion values must be non-negative")
        if any(float(value) < 0.0 for value in self.explicit_collider_projection_threshold):
            raise ValueError("SPH explicit collider mesh projection thresholds must be non-negative")
        body_count = int(self._collider_model().body_count)
        for body in self.explicit_collider_body_ids:
            if body < -1 or body >= body_count:
                raise ValueError("SPH explicit collider body id is out of range")
        self._refresh_explicit_collider_mesh_arrays()

    def _refresh_explicit_collider_mesh_arrays(self) -> None:
        mesh_ids: list[int] = []
        for mesh in self.explicit_collider_meshes:
            mesh_id = getattr(mesh, "id", None)
            if mesh_id is None and getattr(mesh, "mesh", None) is not None:
                mesh_id = getattr(mesh.mesh, "id", None)
            if mesh_id is None and hasattr(mesh, "finalize"):
                mesh_id = mesh.finalize(device=self.model.device)
            if mesh_id is None:
                raise TypeError("SPH explicit collider meshes must be Newton Mesh or Warp Mesh objects")
            mesh_ids.append(int(mesh_id))

        self.explicit_collider_mesh_ids_wp = wp.array(
            np.asarray(mesh_ids, dtype=np.uint64),
            dtype=wp.uint64,
            device=self.model.device,
        )
        self.explicit_collider_margins_wp = wp.array(
            np.asarray(self.explicit_collider_margins, dtype=np.float32),
            dtype=wp.float32,
            device=self.model.device,
        )
        self.explicit_collider_friction_wp = wp.array(
            np.asarray(self.explicit_collider_friction, dtype=np.float32),
            dtype=wp.float32,
            device=self.model.device,
        )
        self.explicit_collider_adhesion_wp = wp.array(
            np.asarray(self.explicit_collider_adhesion, dtype=np.float32),
            dtype=wp.float32,
            device=self.model.device,
        )
        self.explicit_collider_projection_threshold_wp = wp.array(
            np.asarray(self.explicit_collider_projection_threshold, dtype=np.float32),
            dtype=wp.float32,
            device=self.model.device,
        )
        self.explicit_collider_body_ids_wp = wp.array(
            np.asarray(self.explicit_collider_body_ids, dtype=np.int32),
            dtype=wp.int32,
            device=self.model.device,
        )

    def _collider_model(self) -> Model:
        return self.model if self.collider_model is None else self.collider_model

    def analytic_shape_count(self) -> int:
        if not self.enable_shape_boundaries:
            return 0
        return int(self._collider_model().shape_count)

    def explicit_collider_mesh_count(self) -> int:
        if not self.enable_shape_boundaries:
            return 0
        return len(self.explicit_collider_meshes)

    def _reset_analytic_body_impulses(self) -> None:
        body_count = int(self._collider_model().body_count)
        self.last_analytic_body_impulse = np.zeros((body_count, 3), dtype=np.float32)
        self.last_analytic_body_angular_impulse = np.zeros((body_count, 3), dtype=np.float32)
        if self.analytic_body_impulse_wp is None or int(self.analytic_body_impulse_wp.shape[0]) != body_count:
            self.analytic_body_impulse_wp = wp.zeros(body_count, dtype=wp.vec3, device=self.model.device)
            self.analytic_body_angular_impulse_wp = wp.zeros(body_count, dtype=wp.vec3, device=self.model.device)
        else:
            self.analytic_body_impulse_wp.zero_()
            self.analytic_body_angular_impulse_wp.zero_()

    @staticmethod
    def _quat_rotate_z(quat: np.ndarray) -> np.ndarray:
        x, y, z, w = (float(value) for value in quat)
        xx = x * x
        yy = y * y
        wx = w * x
        wy = w * y
        xz = x * z
        yz = y * z
        return np.array((2.0 * (xz + wy), 2.0 * (yz - wx), 1.0 - 2.0 * (xx + yy)), dtype=np.float32)

    @staticmethod
    def _compose_transforms(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
        lhs_pos = np.asarray(lhs[0:3], dtype=np.float32)
        lhs_quat = np.asarray(lhs[3:7], dtype=np.float32)
        rhs_pos = np.asarray(rhs[0:3], dtype=np.float32)
        rhs_quat = np.asarray(rhs[3:7], dtype=np.float32)
        return np.concatenate(
            (
                lhs_pos + _quat_rotate_np(lhs_quat, rhs_pos),
                _quat_normalize_np(_quat_multiply_np(lhs_quat, rhs_quat), eps=1.0e-7),
            )
        ).astype(np.float32)

    @staticmethod
    def _clip_velocity_against_normal(
        velocity: np.ndarray,
        normal: np.ndarray,
        *,
        boundary_friction: float,
        shape_friction: float,
        boundary_velocity: np.ndarray | None = None,
    ) -> np.ndarray:
        wall_v = (
            np.zeros(3, dtype=np.float32)
            if boundary_velocity is None
            else np.asarray(boundary_velocity, dtype=np.float32)
        )
        v = np.asarray(velocity, dtype=np.float32).copy() - wall_v
        n = np.asarray(normal, dtype=np.float32)
        vn = float(np.dot(v, n))
        if vn < 0.0:
            v = v - n * vn

        vt = v - n * float(np.dot(v, n))
        friction = max(float(boundary_friction), float(shape_friction))
        return wall_v + v - vt * min(friction, 1.0)

    @staticmethod
    def _apply_adhesion_velocity(
        velocity: np.ndarray,
        normal: np.ndarray,
        *,
        adhesion: float,
        dt: float,
        enable_adhesion: bool,
    ) -> np.ndarray:
        if enable_adhesion and float(adhesion) > 0.0 and float(dt) > 0.0:
            return np.asarray(velocity, dtype=np.float32) - np.asarray(normal, dtype=np.float32) * float(
                adhesion
            ) * float(dt)
        return np.asarray(velocity, dtype=np.float32)

    @staticmethod
    def _fallback_local_axis(axis: int, sign: float = 1.0) -> np.ndarray:
        normal = np.zeros(3, dtype=np.float32)
        normal[axis] = sign
        return normal

    @staticmethod
    def _project_local_capsule(
        local: np.ndarray,
        *,
        radius: float,
        half_height: float,
        min_distance: float,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        expanded_radius = radius + min_distance
        axis_z = float(np.clip(local[2], -half_height, half_height))
        axis_point = np.array((0.0, 0.0, axis_z), dtype=np.float32)
        radial = local - axis_point
        distance = float(np.linalg.norm(radial))
        if distance >= expanded_radius:
            return None
        normal = radial / distance if distance > 1.0e-7 else SPHBoundaryHandler._fallback_local_axis(0)
        return axis_point + normal * expanded_radius, normal

    @staticmethod
    def _project_local_cylinder(
        local: np.ndarray,
        *,
        radius: float,
        half_height: float,
        min_distance: float,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        expanded_radius = radius + min_distance
        expanded_half_height = half_height + min_distance
        radial_distance = float(np.linalg.norm(local[0:2]))
        axial_distance = abs(float(local[2]))
        if radial_distance >= expanded_radius or axial_distance >= expanded_half_height:
            return None

        projected = local.copy()
        radial_penetration = expanded_radius - radial_distance
        axial_penetration = expanded_half_height - axial_distance
        if radial_penetration <= axial_penetration:
            normal_xy = (
                local[0:2] / radial_distance if radial_distance > 1.0e-7 else np.array((1.0, 0.0), dtype=np.float32)
            )
            projected[0:2] = normal_xy * expanded_radius
            normal = np.array((normal_xy[0], normal_xy[1], 0.0), dtype=np.float32)
        else:
            sign = 1.0 if local[2] >= 0.0 else -1.0
            projected[2] = sign * expanded_half_height
            normal = SPHBoundaryHandler._fallback_local_axis(2, sign)
        return projected, normal

    @staticmethod
    def _project_local_ellipsoid(
        local: np.ndarray,
        *,
        radii: np.ndarray,
        min_distance: float,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        expanded_radii = np.maximum(np.abs(np.asarray(radii, dtype=np.float32)) + min_distance, 1.0e-7)
        scaled = local / expanded_radii
        scaled_distance = float(np.linalg.norm(scaled))
        if scaled_distance >= 1.0:
            return None
        if scaled_distance <= 1.0e-7:
            axis = int(np.argmin(expanded_radii))
            normal = SPHBoundaryHandler._fallback_local_axis(axis)
            projected = normal * expanded_radii[axis]
        else:
            projected = local / scaled_distance
            gradient = projected / (expanded_radii * expanded_radii)
            gradient_length = float(np.linalg.norm(gradient))
            normal = (
                gradient / gradient_length if gradient_length > 1.0e-7 else SPHBoundaryHandler._fallback_local_axis(2)
            )
        return projected.astype(np.float32), normal.astype(np.float32)

    @staticmethod
    def _project_local_cone(
        local: np.ndarray,
        *,
        radius: float,
        half_height: float,
        min_distance: float,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        expanded_radius = radius + min_distance
        expanded_half_height = half_height + min_distance
        if expanded_radius <= 0.0 or expanded_half_height <= 0.0:
            return None

        z = float(local[2])
        radial_distance = float(np.linalg.norm(local[0:2]))
        if z <= -expanded_half_height or z >= expanded_half_height:
            return None

        allowed_radius = expanded_radius * (expanded_half_height - z) / (2.0 * expanded_half_height)
        if radial_distance >= allowed_radius:
            return None

        base_penetration = z + expanded_half_height
        side_penetration = allowed_radius - radial_distance
        projected = local.copy()
        if base_penetration <= side_penetration:
            projected[2] = -expanded_half_height
            normal = SPHBoundaryHandler._fallback_local_axis(2, -1.0)
        else:
            radial_dir = (
                local[0:2] / radial_distance if radial_distance > 1.0e-7 else np.array((1.0, 0.0), dtype=np.float32)
            )
            projected[0:2] = radial_dir * allowed_radius
            normal = np.array(
                (radial_dir[0], radial_dir[1], expanded_radius / (2.0 * expanded_half_height)),
                dtype=np.float32,
            )
            normal_length = float(np.linalg.norm(normal))
            normal = normal / normal_length if normal_length > 1.0e-7 else SPHBoundaryHandler._fallback_local_axis(0)
        return projected, normal

    @staticmethod
    def _closest_point_on_triangle(point: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
        ab = b - a
        ac = c - a
        ap = point - a
        d1 = float(np.dot(ab, ap))
        d2 = float(np.dot(ac, ap))
        if d1 <= 0.0 and d2 <= 0.0:
            return a

        bp = point - b
        d3 = float(np.dot(ab, bp))
        d4 = float(np.dot(ac, bp))
        if d3 >= 0.0 and d4 <= d3:
            return b

        vc = d1 * d4 - d3 * d2
        if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
            v = d1 / (d1 - d3)
            return a + v * ab

        cp = point - c
        d5 = float(np.dot(ab, cp))
        d6 = float(np.dot(ac, cp))
        if d6 >= 0.0 and d5 <= d6:
            return c

        vb = d5 * d2 - d1 * d6
        if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
            w = d2 / (d2 - d6)
            return a + w * ac

        va = d3 * d6 - d5 * d4
        if va <= 0.0 and (d4 - d3) >= 0.0 and (d5 - d6) >= 0.0:
            w = (d4 - d3) / ((d4 - d3) + (d5 - d6))
            return b + w * (c - b)

        denom = 1.0 / (va + vb + vc)
        v = vb * denom
        w = vc * denom
        return a + ab * v + ac * w

    @staticmethod
    def _project_local_mesh(
        local: np.ndarray,
        mesh: object,
        *,
        scale: np.ndarray,
        min_distance: float,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        vertices = np.asarray(getattr(mesh, "vertices", ()), dtype=np.float32)
        indices = np.asarray(getattr(mesh, "indices", ()), dtype=np.int64).reshape(-1, 3)
        if vertices.size == 0 or indices.size == 0:
            return None

        vertex_scale = np.asarray(scale, dtype=np.float32)
        if vertex_scale.shape[0] >= 3:
            vertices = vertices[:, :3] * vertex_scale[:3]
        else:
            vertices = vertices[:, :3] * float(vertex_scale[0])

        best_distance_sq = np.inf
        best_point = None
        best_normal = None
        for tri in indices:
            a, b, c = vertices[int(tri[0])], vertices[int(tri[1])], vertices[int(tri[2])]
            closest = SPHBoundaryHandler._closest_point_on_triangle(local, a, b, c)
            offset = local - closest
            distance_sq = float(np.dot(offset, offset))
            if distance_sq >= best_distance_sq:
                continue
            face_normal = np.cross(b - a, c - a).astype(np.float32)
            normal_length = float(np.linalg.norm(face_normal))
            if normal_length <= 1.0e-7:
                continue
            best_distance_sq = distance_sq
            best_point = closest.astype(np.float32)
            best_normal = face_normal / normal_length

        if best_point is None or best_normal is None:
            return None
        distance = float(np.sqrt(best_distance_sq))
        if distance >= min_distance:
            return None
        if distance > 1.0e-7 and float(np.dot(local - best_point, best_normal)) < 0.0:
            best_normal = -best_normal
        return best_point + best_normal.astype(np.float32) * min_distance, best_normal.astype(np.float32)

    def _collide_explicit_meshes_cpu(
        self,
        x: np.ndarray,
        v: np.ndarray,
        *,
        particle_mass: float,
        radius: float,
        boundary_margin: float,
        boundary_friction: float,
        enable_boundary_adhesion: bool,
        body_q: np.ndarray | None,
        body_qd: np.ndarray | None,
        body_q_prev: np.ndarray | None,
        body_com: np.ndarray,
        collider_velocity_mode: int,
        dt: float,
        analytic_body_impulse: np.ndarray | None,
        analytic_body_angular_impulse: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray, bool]:
        hit = False
        for mesh, mesh_body, mesh_margin, mesh_friction, mesh_adhesion, mesh_projection_threshold in zip(
            self.explicit_collider_meshes,
            self.explicit_collider_body_ids,
            self.explicit_collider_margins,
            self.explicit_collider_friction,
            self.explicit_collider_adhesion,
            self.explicit_collider_projection_threshold,
            strict=True,
        ):
            body = int(mesh_body)
            mesh_x = x
            normal_transform = None
            boundary_velocity = np.zeros(3, dtype=np.float32)
            body_transform = None
            if body_q is not None and 0 <= body < body_q.shape[0]:
                body_transform = body_q[body]
                mesh_x = _quat_rotate_inv_np(body_transform[3:7], x - body_transform[0:3])
                normal_transform = body_transform[3:7]
                if (
                    collider_velocity_mode == SPH_COLLIDER_VELOCITY_BACKWARD
                    and body_q_prev is not None
                    and 0 <= body < body_q_prev.shape[0]
                    and dt > 0.0
                ):
                    prev_transform = body_q_prev[body]
                    com_local = body_com[body] if body < body_com.shape[0] else np.zeros(3, dtype=np.float32)
                    com_world = body_transform[0:3] + _quat_rotate_np(body_transform[3:7], com_local)
                    prev_com_world = prev_transform[0:3] + _quat_rotate_np(prev_transform[3:7], com_local)
                    linear_velocity = (com_world - prev_com_world) / float(dt)
                    angular_velocity = _quat_velocity_np(body_transform[3:7], prev_transform[3:7], dt)
                    boundary_velocity = (linear_velocity + np.cross(angular_velocity, x - com_world)).astype(np.float32)
                elif body_qd is not None and 0 <= body < body_qd.shape[0]:
                    com_world = (
                        body_transform[0:3] + _quat_rotate_np(body_transform[3:7], body_com[body])
                        if body < body_com.shape[0]
                        else body_transform[0:3]
                    )
                    boundary_velocity = (body_qd[body, 0:3] + np.cross(body_qd[body, 3:6], x - com_world)).astype(
                        np.float32
                    )
            projection = self._project_local_mesh(
                mesh_x,
                mesh,
                scale=np.ones(3, dtype=np.float32),
                min_distance=radius + float(mesh_margin) + float(mesh_projection_threshold) + float(boundary_margin),
            )
            if projection is None:
                continue
            projected, normal = projection
            normal_length = float(np.linalg.norm(normal))
            if normal_length <= 1.0e-7:
                continue
            normal = normal / normal_length
            if normal_transform is not None and body_transform is not None:
                normal = _quat_rotate_np(normal_transform, normal)
                normal_length = float(np.linalg.norm(normal))
                if normal_length <= 1.0e-7:
                    continue
                normal = normal / normal_length
                x = (body_transform[0:3] + _quat_rotate_np(body_transform[3:7], projected)).astype(np.float32)
            else:
                x = projected.astype(np.float32)
            v_before_mesh = v.copy()
            v = self._clip_velocity_against_normal(
                v,
                normal,
                boundary_friction=boundary_friction,
                shape_friction=float(mesh_friction),
                boundary_velocity=boundary_velocity,
            )
            v = self._apply_adhesion_velocity(
                v,
                normal,
                adhesion=float(mesh_adhesion),
                dt=dt,
                enable_adhesion=enable_boundary_adhesion,
            )
            if (
                analytic_body_impulse is not None
                and analytic_body_angular_impulse is not None
                and body_transform is not None
                and 0 <= body < analytic_body_impulse.shape[0]
            ):
                particle_impulse = float(particle_mass) * (v - v_before_mesh)
                body_impulse = -particle_impulse
                analytic_body_impulse[body] += body_impulse.astype(np.float32)
                com_world = (
                    body_transform[0:3] + _quat_rotate_np(body_transform[3:7], body_com[body])
                    if body < body_com.shape[0]
                    else body_transform[0:3]
                )
                analytic_body_angular_impulse[body] += np.cross(x - com_world, body_impulse).astype(np.float32)
            hit = True
        return x, v, hit

    def _collide_analytic_shapes_cpu(
        self,
        state: State,
        *,
        boundary_margin: float,
        boundary_friction: float,
        collider_velocity_mode: int,
        enable_boundary_adhesion: bool,
        dt: float,
    ) -> None:
        model = self.model
        collider_model = self._collider_model()
        self._reset_analytic_body_impulses()
        particle_q = np.asarray(state.particle_q.numpy(), dtype=np.float32).copy()
        particle_qd = np.asarray(state.particle_qd.numpy(), dtype=np.float32).copy()
        particle_mass = np.asarray(model.particle_mass.numpy(), dtype=np.float32)
        particle_radius = np.asarray(model.particle_radius.numpy(), dtype=np.float32)
        particle_flags = np.asarray(model.particle_flags.numpy(), dtype=np.int32)
        particle_world = np.asarray(model.particle_world.numpy(), dtype=np.int32)
        sph_role = np.asarray(model.sph.role.numpy(), dtype=np.int32)
        shape_type = np.asarray(collider_model.shape_type.numpy(), dtype=np.int32)
        shape_flags = np.asarray(collider_model.shape_flags.numpy(), dtype=np.int32)
        shape_world = np.asarray(collider_model.shape_world.numpy(), dtype=np.int32)
        shape_body = np.asarray(collider_model.shape_body.numpy(), dtype=np.int32)
        shape_transform = np.asarray(collider_model.shape_transform.numpy(), dtype=np.float32)
        shape_scale = np.asarray(collider_model.shape_scale.numpy(), dtype=np.float32)
        shape_margin = np.asarray(self.model_collider_shape_margin_wp.numpy(), dtype=np.float32)
        shape_material_mu = np.asarray(self.model_collider_shape_friction_wp.numpy(), dtype=np.float32)
        shape_adhesion = np.asarray(self.model_collider_shape_adhesion_wp.numpy(), dtype=np.float32)
        shape_projection_threshold = np.asarray(
            self.model_collider_shape_projection_threshold_wp.numpy(),
            dtype=np.float32,
        )
        body_q = (
            np.asarray(state.body_q.numpy(), dtype=np.float32)
            if state.body_q is not None and getattr(state.body_q, "shape", (0,))[0] > 0
            else None
        )
        body_qd = (
            np.asarray(state.body_qd.numpy(), dtype=np.float32)
            if state.body_qd is not None and getattr(state.body_qd, "shape", (0,))[0] > 0
            else None
        )
        body_q_prev = (
            np.asarray(self.previous_collider_body_q_wp.numpy(), dtype=np.float32)
            if collider_velocity_mode == SPH_COLLIDER_VELOCITY_BACKWARD
            and self.previous_collider_body_q_wp is not None
            and getattr(self.previous_collider_body_q_wp, "shape", (0,))[0] > 0
            else None
        )
        body_com = (
            np.asarray(collider_model.body_com.numpy(), dtype=np.float32)
            if int(collider_model.body_count) > 0 and collider_model.body_com is not None
            else np.zeros((0, 3), dtype=np.float32)
        )
        analytic_body_impulse = self.last_analytic_body_impulse
        analytic_body_angular_impulse = self.last_analytic_body_angular_impulse
        boundary_impulse = np.asarray(state.sph.boundary_impulse.numpy(), dtype=np.float32).copy()

        for particle in range(int(model.particle_count)):
            if (particle_flags[particle] & int(ParticleFlags.ACTIVE)) == 0 or sph_role[particle] != int(SPHRole.FLUID):
                boundary_impulse[particle] = 0.0
                continue

            x = particle_q[particle].copy()
            v = particle_qd[particle].copy()
            v_initial = v.copy()
            radius = float(particle_radius[particle])
            world = int(particle_world[particle])
            hit = False

            for shape in range(int(collider_model.shape_count)):
                if (shape_flags[shape] & int(ShapeFlags.COLLIDE_PARTICLES)) == 0 or not _same_world_np(
                    world, int(shape_world[shape])
                ):
                    continue

                transform = shape_transform[shape]
                body = int(shape_body[shape])
                body_transform = None
                if body_q is not None and 0 <= body < body_q.shape[0]:
                    body_transform = body_q[body]
                    transform = self._compose_transforms(body_transform, transform)
                shape_velocity = np.zeros(3, dtype=np.float32)
                if (
                    collider_velocity_mode == SPH_COLLIDER_VELOCITY_BACKWARD
                    and body_q_prev is not None
                    and body_transform is not None
                    and 0 <= body < body_q_prev.shape[0]
                    and dt > 0.0
                ):
                    prev_transform = body_q_prev[body]
                    com_local = body_com[body] if body < body_com.shape[0] else np.zeros(3, dtype=np.float32)
                    com_world = body_transform[0:3] + _quat_rotate_np(body_transform[3:7], com_local)
                    prev_com_world = prev_transform[0:3] + _quat_rotate_np(prev_transform[3:7], com_local)
                    linear_velocity = (com_world - prev_com_world) / float(dt)
                    angular_velocity = _quat_velocity_np(body_transform[3:7], prev_transform[3:7], dt)
                    shape_velocity = (linear_velocity + np.cross(angular_velocity, x - com_world)).astype(np.float32)
                elif body_qd is not None and body_transform is not None and 0 <= body < body_qd.shape[0]:
                    if body < body_com.shape[0]:
                        com_world = body_transform[0:3] + _quat_rotate_np(body_transform[3:7], body_com[body])
                    else:
                        com_world = body_transform[0:3]
                    shape_velocity = (body_qd[body, 0:3] + np.cross(body_qd[body, 3:6], x - com_world)).astype(
                        np.float32
                    )
                point = transform[0:3]
                quat = transform[3:7]
                min_distance = (
                    radius
                    + float(shape_margin[shape])
                    + float(shape_projection_threshold[shape])
                    + float(boundary_margin)
                )
                v_before_shape = v.copy()
                hit_shape = False

                if shape_type[shape] == int(GeoType.PLANE):
                    normal = self._quat_rotate_z(quat)
                    normal_length = float(np.linalg.norm(normal))
                    if normal_length <= 1.0e-7:
                        continue
                    normal = normal / normal_length
                    signed_distance = float(np.dot(normal, x - point))
                    if signed_distance >= min_distance:
                        continue
                    x = x + normal * (min_distance - signed_distance)
                    v = self._clip_velocity_against_normal(
                        v,
                        normal,
                        boundary_friction=boundary_friction,
                        shape_friction=float(shape_material_mu[shape]),
                        boundary_velocity=shape_velocity,
                    )
                    hit_shape = True

                elif shape_type[shape] == int(GeoType.SPHERE):
                    offset = x - point
                    distance = float(np.linalg.norm(offset))
                    target_radius = abs(float(shape_scale[shape, 0])) + min_distance
                    if distance >= target_radius:
                        continue
                    normal = offset / distance if distance > 1.0e-7 else np.array((0.0, 1.0, 0.0), dtype=np.float32)
                    x = point + normal * target_radius
                    v = self._clip_velocity_against_normal(
                        v,
                        normal,
                        boundary_friction=boundary_friction,
                        shape_friction=float(shape_material_mu[shape]),
                        boundary_velocity=shape_velocity,
                    )
                    hit_shape = True

                elif shape_type[shape] == int(GeoType.BOX):
                    half_extents = np.abs(shape_scale[shape]) + min_distance
                    local = _quat_rotate_inv_np(quat, x - point)
                    if np.any(np.abs(local) >= half_extents):
                        continue
                    penetration = half_extents - np.abs(local)
                    axis = int(np.argmin(penetration))
                    sign = 1.0 if local[axis] >= 0.0 else -1.0
                    local[axis] = sign * half_extents[axis]
                    normal_local = np.zeros(3, dtype=np.float32)
                    normal_local[axis] = sign
                    normal = _quat_rotate_np(quat, normal_local)
                    normal_length = float(np.linalg.norm(normal))
                    if normal_length <= 1.0e-7:
                        continue
                    normal = normal / normal_length
                    x = point + _quat_rotate_np(quat, local)
                    v = self._clip_velocity_against_normal(
                        v,
                        normal,
                        boundary_friction=boundary_friction,
                        shape_friction=float(shape_material_mu[shape]),
                        boundary_velocity=shape_velocity,
                    )
                    hit_shape = True

                elif shape_type[shape] in (int(GeoType.CAPSULE), int(GeoType.CYLINDER)):
                    local = _quat_rotate_inv_np(quat, x - point)
                    primitive_radius = abs(float(shape_scale[shape, 0]))
                    primitive_half_height = abs(float(shape_scale[shape, 1]))
                    if shape_type[shape] == int(GeoType.CAPSULE):
                        projection = self._project_local_capsule(
                            local,
                            radius=primitive_radius,
                            half_height=primitive_half_height,
                            min_distance=min_distance,
                        )
                    else:
                        projection = self._project_local_cylinder(
                            local,
                            radius=primitive_radius,
                            half_height=primitive_half_height,
                            min_distance=min_distance,
                        )
                    if projection is None:
                        continue
                    projected_local, normal_local = projection
                    normal = _quat_rotate_np(quat, normal_local)
                    normal_length = float(np.linalg.norm(normal))
                    if normal_length <= 1.0e-7:
                        continue
                    normal = normal / normal_length
                    x = point + _quat_rotate_np(quat, projected_local)
                    v = self._clip_velocity_against_normal(
                        v,
                        normal,
                        boundary_friction=boundary_friction,
                        shape_friction=float(shape_material_mu[shape]),
                        boundary_velocity=shape_velocity,
                    )
                    hit_shape = True

                elif shape_type[shape] in (int(GeoType.ELLIPSOID), int(GeoType.CONE)):
                    local = _quat_rotate_inv_np(quat, x - point)
                    if shape_type[shape] == int(GeoType.ELLIPSOID):
                        projection = self._project_local_ellipsoid(
                            local,
                            radii=shape_scale[shape],
                            min_distance=min_distance,
                        )
                    else:
                        projection = self._project_local_cone(
                            local,
                            radius=abs(float(shape_scale[shape, 0])),
                            half_height=abs(float(shape_scale[shape, 1])),
                            min_distance=min_distance,
                        )
                    if projection is None:
                        continue
                    projected_local, normal_local = projection
                    normal = _quat_rotate_np(quat, normal_local)
                    normal_length = float(np.linalg.norm(normal))
                    if normal_length <= 1.0e-7:
                        continue
                    normal = normal / normal_length
                    x = point + _quat_rotate_np(quat, projected_local)
                    v = self._clip_velocity_against_normal(
                        v,
                        normal,
                        boundary_friction=boundary_friction,
                        shape_friction=float(shape_material_mu[shape]),
                        boundary_velocity=shape_velocity,
                    )
                    hit_shape = True

                elif shape_type[shape] in (int(GeoType.MESH), int(GeoType.CONVEX_MESH)):
                    mesh = collider_model.shape_source[shape]
                    if mesh is None:
                        continue
                    local = _quat_rotate_inv_np(quat, x - point)
                    projection = self._project_local_mesh(
                        local,
                        mesh,
                        scale=shape_scale[shape],
                        min_distance=min_distance,
                    )
                    if projection is None:
                        continue
                    projected_local, normal_local = projection
                    normal = _quat_rotate_np(quat, normal_local)
                    normal_length = float(np.linalg.norm(normal))
                    if normal_length <= 1.0e-7:
                        continue
                    normal = normal / normal_length
                    x = point + _quat_rotate_np(quat, projected_local)
                    v = self._clip_velocity_against_normal(
                        v,
                        normal,
                        boundary_friction=boundary_friction,
                        shape_friction=float(shape_material_mu[shape]),
                        boundary_velocity=shape_velocity,
                    )
                    hit_shape = True

                if hit_shape:
                    v = self._apply_adhesion_velocity(
                        v,
                        normal,
                        adhesion=float(shape_adhesion[shape]),
                        dt=dt,
                        enable_adhesion=enable_boundary_adhesion,
                    )
                    hit = True
                    if (
                        analytic_body_impulse is not None
                        and analytic_body_angular_impulse is not None
                        and body_transform is not None
                        and 0 <= body < analytic_body_impulse.shape[0]
                    ):
                        particle_impulse = float(particle_mass[particle]) * (v - v_before_shape)
                        body_impulse = -particle_impulse
                        analytic_body_impulse[body] += body_impulse.astype(np.float32)
                        if body < body_com.shape[0]:
                            com_world = body_transform[0:3] + _quat_rotate_np(body_transform[3:7], body_com[body])
                        else:
                            com_world = body_transform[0:3]
                        analytic_body_angular_impulse[body] += np.cross(x - com_world, body_impulse).astype(np.float32)

            x, v, hit_explicit_mesh = self._collide_explicit_meshes_cpu(
                x,
                v,
                particle_mass=float(particle_mass[particle]),
                radius=radius,
                boundary_margin=boundary_margin,
                boundary_friction=boundary_friction,
                enable_boundary_adhesion=enable_boundary_adhesion,
                body_q=body_q,
                body_qd=body_qd,
                body_q_prev=body_q_prev,
                body_com=body_com,
                collider_velocity_mode=collider_velocity_mode,
                dt=dt,
                analytic_body_impulse=analytic_body_impulse,
                analytic_body_angular_impulse=analytic_body_angular_impulse,
            )
            hit = hit or hit_explicit_mesh

            if hit:
                particle_q[particle] = x
                particle_qd[particle] = v
                boundary_impulse[particle] = boundary_impulse[particle] + float(particle_mass[particle]) * (
                    v - v_initial
                )

        state.particle_q.assign(particle_q.astype(np.float32))
        state.particle_qd.assign(particle_qd.astype(np.float32))
        state.sph.boundary_impulse.assign(boundary_impulse.astype(np.float32))
        if self.analytic_body_impulse_wp is not None:
            self.analytic_body_impulse_wp.assign(self.last_analytic_body_impulse.astype(np.float32))
        if self.analytic_body_angular_impulse_wp is not None:
            self.analytic_body_angular_impulse_wp.assign(self.last_analytic_body_angular_impulse.astype(np.float32))

    def collide_analytic_shapes(
        self,
        state: State,
        *,
        boundary_margin: float,
        boundary_friction: float,
        collider_velocity_mode: int = SPH_COLLIDER_VELOCITY_FORWARD,
        enable_boundary_adhesion: bool = False,
        dt: float = 0.0,
    ) -> None:
        """Project fluid particles against supported analytic Newton shapes."""
        model = self.model
        collider_model = self._collider_model()
        if collider_velocity_mode == SPH_COLLIDER_VELOCITY_BACKWARD:
            self.require_collider_previous_position(state.body_q)
        self._reset_analytic_body_impulses()
        if (
            not self.enable_shape_boundaries
            or not model.particle_count
            or (not collider_model.shape_count and not self.explicit_collider_mesh_count())
        ):
            self.save_collider_current_position(state.body_q)
            return

        if getattr(model.device, "is_cpu", False):
            self._collide_analytic_shapes_cpu(
                state,
                boundary_margin=boundary_margin,
                boundary_friction=boundary_friction,
                collider_velocity_mode=collider_velocity_mode,
                enable_boundary_adhesion=enable_boundary_adhesion,
                dt=dt,
            )
            self.save_collider_current_position(state.body_q)
            return

        body_q_prev = (
            self.previous_collider_body_q_wp
            if collider_velocity_mode == SPH_COLLIDER_VELOCITY_BACKWARD
            else state.body_q
        )

        wp.launch(
            _collide_particle_shapes,
            dim=model.particle_count,
            inputs=[
                collider_model.shape_count,
                collider_model.body_count,
                state.particle_q,
                state.particle_qd,
                model.particle_mass,
                model.particle_radius,
                model.particle_flags,
                model.particle_world,
                model.sph.role,
                collider_model.shape_type,
                collider_model.shape_flags,
                collider_model.shape_world,
                collider_model.shape_body,
                collider_model.shape_transform,
                collider_model.shape_scale,
                collider_model.shape_source_ptr,
                self.model_collider_shape_margin_wp,
                self.model_collider_shape_friction_wp,
                self.model_collider_shape_adhesion_wp,
                self.model_collider_shape_projection_threshold_wp,
                self.explicit_collider_mesh_count(),
                self.explicit_collider_mesh_ids_wp,
                self.explicit_collider_margins_wp,
                self.explicit_collider_friction_wp,
                self.explicit_collider_adhesion_wp,
                self.explicit_collider_projection_threshold_wp,
                self.explicit_collider_body_ids_wp,
                state.body_q,
                state.body_qd,
                body_q_prev,
                collider_model.body_com,
                self.analytic_body_impulse_wp,
                self.analytic_body_angular_impulse_wp,
                boundary_margin,
                boundary_friction,
                collider_velocity_mode,
                enable_boundary_adhesion,
                dt,
                state.sph.boundary_impulse,
            ],
            device=model.device,
        )
        self.save_collider_current_position(state.body_q)


def _resolve_collider_body_ids(
    collider_model: Model,
    collider_body_ids: list[int] | None,
    collider_meshes: list[wp.Mesh] | None,
) -> np.ndarray:
    if collider_meshes is not None:
        mesh_count = len(collider_meshes)
        if collider_body_ids is None:
            return np.full(mesh_count, -1, dtype=np.int32)
        body_ids = [int(body) if body is not None else -1 for body in collider_body_ids]
        if len(body_ids) != mesh_count:
            raise ValueError("SPH collider_body_ids must match collider_meshes length")
        for body in body_ids:
            if body < -1 or body >= int(collider_model.body_count):
                raise ValueError("SPH collider body id is out of range")
        return np.asarray(body_ids, dtype=np.int32)

    if collider_body_ids is None:
        body_ids = [
            body
            for body in range(-1, int(collider_model.body_count))
            if _collider_body_shape_count(collider_model, body) > 0
        ]
    else:
        body_ids = [int(body) for body in collider_body_ids]
        for body in body_ids:
            if body < -1 or body >= int(collider_model.body_count):
                raise ValueError("SPH collider body id is out of range")
            if _collider_body_shape_count(collider_model, body) == 0:
                raise ValueError(f"SPH collider body {body} has no particle-colliding shapes")
    return np.asarray(body_ids, dtype=np.int32)


def _collider_body_shape_count(collider_model: Model, body: int) -> int:
    if int(collider_model.shape_count) == 0:
        return 0
    shape_body = np.asarray(collider_model.shape_body.numpy(), dtype=np.int32)
    shape_flags = np.asarray(collider_model.shape_flags.numpy(), dtype=np.int32)
    mask = (shape_body == int(body)) & ((shape_flags & int(ShapeFlags.COLLIDE_PARTICLES)) != 0)
    return int(np.count_nonzero(mask))
