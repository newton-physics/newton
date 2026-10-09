# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Shared routed-tendon solver state helpers.

The routed tendon geometry is solver-independent: XPBD and VBD both need the
same tangent attachments, mutable free-span rest lengths, and segment-to-guide
mapping before applying their own numerical solve.
"""

from __future__ import annotations

import numpy as np
import warp as wp

from ..sim import Model
from ..sim.tendon import TendonGuideFlags, TendonGuideType
from .tendon_kernels import (
    mark_tendon_rebaseline_worlds,
    prepare_tendon_route,
    rebaseline_tendon_attachments,
    reset_tendon_state_array,
    snapshot_tendon_guide_active,
    solve_tendon_material,
    update_tendon_attachments,
    update_tendon_cone_rows,
    update_tendon_guide_active,
)

_TENDON_SEGMENT_STATE_FIELDS = (
    "tendon_seg_rest_length",
    "tendon_seg_rest_length_step",
    "tendon_seg_route_rest_length",
    "tendon_seg_stretch",
    "tendon_seg_material_tension",
    "tendon_seg_damping_tension",
    "tendon_seg_attachment_l",
    "tendon_seg_attachment_r",
    "tendon_seg_length",
    "tendon_seg_attachment_l_local",
    "tendon_seg_attachment_r_local",
    "tendon_seg_attachment_l_local_step",
    "tendon_seg_attachment_r_local_step",
    "tendon_seg_lambda",
    "tendon_seg_delta_lambda",
    "tendon_seg_rolling_delta_l",
    "tendon_seg_rolling_delta_r",
    "tendon_seg_active",
    "tendon_seg_active_guide_l",
    "tendon_seg_active_guide_r",
    "tendon_seg_active_compliance",
    "tendon_seg_active_damping",
)
_TENDON_GUIDE_STATE_FIELDS = (
    "tendon_guide_active",
    "tendon_guide_active_step",
    "tendon_guide_route_rest_length",
    "tendon_guide_cone_seg_l",
    "tendon_guide_cone_seg_r",
    "tendon_guide_cap_ratio",
)
_TENDON_STATE_FIELDS = (
    "tendon_cone_sweep_count",
    "tendon_total_cable",
)


def _transform_point_np(pose: np.ndarray, point: np.ndarray) -> np.ndarray:
    """Apply a Newton transform (px,py,pz,qx,qy,qz,qw) to a 3D point using numpy."""
    p = pose[:3]
    q = pose[3:]
    t = 2.0 * np.cross(q[:3], point)
    return point + q[3] * t + np.cross(q[:3], t) + p


def _transform_vector_np(pose: np.ndarray, vec: np.ndarray) -> np.ndarray:
    """Rotate a 3D vector by the quaternion in a Newton transform."""
    q = pose[3:]
    t = 2.0 * np.cross(q[:3], vec)
    return vec + q[3] * t + np.cross(q[:3], t)


def _tangent_point_circle_np(
    point: np.ndarray,
    center: np.ndarray,
    radius: float,
    plane_normal: np.ndarray,
    orientation: int,
) -> np.ndarray:
    """Compute the tangent point on a circle from an external point."""
    point = np.asarray(point, dtype=np.float64)
    center = np.asarray(center, dtype=np.float64)
    normal = np.asarray(plane_normal, dtype=np.float64)
    normal = normal / max(float(np.linalg.norm(normal)), 1.0e-12)

    d = center - point
    d_proj = d - np.dot(d, normal) * normal
    dist = float(np.linalg.norm(d_proj))
    if dist <= radius:
        if dist < 1.0e-8:
            fallback = np.array([1.0, 0.0, 0.0], dtype=np.float64)
            if abs(normal[0]) > 0.9:
                fallback = np.array([0.0, 1.0, 0.0], dtype=np.float64)
            fallback -= np.dot(fallback, normal) * normal
            return center + radius * fallback / float(np.linalg.norm(fallback))
        return center - radius * d_proj / dist

    u = d_proj / dist
    v = np.cross(normal, u)
    phi = np.arcsin(min(radius / dist, 1.0))
    angle = -0.5 * np.pi - phi if orientation > 0 else 0.5 * np.pi + phi
    return center + radius * (np.cos(angle) * u + np.sin(angle) * v)


def _segment_attachment_points_np(
    center_l: np.ndarray,
    center_r: np.ndarray,
    type_l: int,
    type_r: int,
    radius_l: float,
    radius_r: float,
    orient_l: int,
    orient_r: int,
    normal_l: np.ndarray,
    normal_r: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute free-span endpoints with the same tangent cases as the Warp kernel."""
    new_l = np.asarray(center_l, dtype=np.float64)
    new_r = np.asarray(center_r, dtype=np.float64)
    rolling = int(TendonGuideType.ROLLER)

    if type_l == rolling and type_r == rolling and radius_l > 0.0 and radius_r > 0.0:
        for _iter in range(10):
            new_r = _tangent_point_circle_np(new_l, center_r, radius_r, normal_r, orient_r)
            new_l = _tangent_point_circle_np(new_r, center_l, radius_l, normal_l, -orient_l)
    elif type_l == rolling and radius_l > 0.0:
        new_l = _tangent_point_circle_np(center_r, center_l, radius_l, normal_l, -orient_l)
        new_r = np.asarray(center_r, dtype=np.float64)
    elif type_r == rolling and radius_r > 0.0:
        new_l = np.asarray(center_l, dtype=np.float64)
        new_r = _tangent_point_circle_np(center_l, center_r, radius_r, normal_r, orient_r)

    return new_l, new_r


class TendonStateMixin:
    """Mixin that allocates routed-tendon mutable state on a solver instance."""

    def _init_tendon_state(self, model: Model) -> None:
        """Allocate mutable tendon state arrays and build segment/guide mappings."""
        if model.tendon_segment_count > 0 and model.requires_grad:
            raise NotImplementedError(
                "Routed tendon simulation is not differentiable; finalize the model with requires_grad=False"
            )

        self._has_dynamic_tendon_guides = False
        self._tendon_initial_state = {}
        self._tendon_segment_world = None
        self._tendon_guide_world = None
        self._tendon_world = None
        self._tendon_pose_rebaseline_mask = None
        # Solver-level cable cone parameters (a solver may override before calling this).
        if not hasattr(self, "tendon_max_sweeps"):
            self.tendon_max_sweeps = 256
        if not hasattr(self, "tendon_settle_tol"):
            self.tendon_settle_tol = 1.0e-3
        if not hasattr(self, "tendon_activation_tol"):
            self.tendon_activation_tol = 2.0e-3
        if not 1 <= self.tendon_max_sweeps <= 256:
            raise ValueError(f"tendon_max_sweeps must be between 1 and 256, got {self.tendon_max_sweeps}")
        if self.tendon_settle_tol < 0.0:
            raise ValueError(f"tendon_settle_tol must be non-negative, got {self.tendon_settle_tol}")
        if not 0.0 <= self.tendon_activation_tol < 1.0:
            raise ValueError(
                f"tendon_activation_tol must be between 0 (inclusive) and 1 (exclusive), "
                f"got {self.tendon_activation_tol}"
            )
        if model.tendon_segment_count == 0:
            self.tendon_seg_rest_length = None
            self.tendon_seg_rest_length_step = None
            self.tendon_seg_route_rest_length = None
            self.tendon_seg_stretch = None
            self.tendon_seg_material_tension = None
            self.tendon_seg_damping_tension = None
            self.tendon_seg_attachment_l = None
            self.tendon_seg_attachment_r = None
            self.tendon_seg_length = None
            self.tendon_seg_attachment_l_local = None
            self.tendon_seg_attachment_r_local = None
            self.tendon_seg_attachment_l_local_step = None
            self.tendon_seg_attachment_r_local_step = None
            self.tendon_seg_lambda = None
            self.tendon_seg_delta_lambda = None
            self.tendon_seg_rolling_delta_l = None
            self.tendon_seg_rolling_delta_r = None
            self.tendon_cone_sweep_count = None
            self.tendon_seg_guide_l = None
            self.tendon_seg_active = None
            self.tendon_seg_active_guide_l = None
            self.tendon_seg_active_guide_r = None
            self.tendon_seg_active_compliance = None
            self.tendon_seg_active_damping = None
            self.tendon_guide_active = None
            self.tendon_guide_active_step = None
            self.tendon_guide_route_rest_length = None
            self.tendon_guide_seg_left = None
            self.tendon_guide_tendon = None
            self.tendon_guide_cone_seg_l = None
            self.tendon_guide_cone_seg_r = None
            self.tendon_guide_cap_ratio = None
            self.tendon_total_cable = None
            return

        with wp.ScopedDevice(model.device):
            self.tendon_seg_attachment_l = wp.zeros(model.tendon_segment_count, dtype=wp.vec3)
            self.tendon_seg_attachment_r = wp.zeros(model.tendon_segment_count, dtype=wp.vec3)
            self.tendon_seg_length = wp.zeros(model.tendon_segment_count, dtype=float)
            self.tendon_seg_attachment_l_local = wp.zeros(model.tendon_segment_count, dtype=wp.vec3)
            self.tendon_seg_attachment_r_local = wp.zeros(model.tendon_segment_count, dtype=wp.vec3)
            self.tendon_seg_attachment_l_local_step = wp.zeros(model.tendon_segment_count, dtype=wp.vec3)
            self.tendon_seg_attachment_r_local_step = wp.zeros(model.tendon_segment_count, dtype=wp.vec3)
            # Unilateral constitutive tension; the signed damping term is reported separately.
            self.tendon_seg_material_tension = wp.zeros(model.tendon_segment_count, dtype=float)
            self.tendon_seg_lambda = wp.zeros(model.tendon_segment_count, dtype=float)
            self.tendon_seg_delta_lambda = wp.zeros(model.tendon_segment_count, dtype=float)
            self.tendon_seg_rolling_delta_l = wp.zeros(model.tendon_segment_count, dtype=float)
            self.tendon_seg_rolling_delta_r = wp.zeros(model.tendon_segment_count, dtype=float)
            # Cached instantaneous damping term used by routing and slip projections.
            self.tendon_seg_damping_tension = wp.zeros(model.tendon_segment_count, dtype=float)
            self.tendon_cone_sweep_count = wp.zeros(model.tendon_count, dtype=wp.int32)
            self.tendon_seg_active = wp.ones(model.tendon_segment_count, dtype=wp.int32)
            self.tendon_seg_active_guide_l = wp.zeros(model.tendon_segment_count, dtype=wp.int32)
            self.tendon_seg_active_guide_r = wp.zeros(model.tendon_segment_count, dtype=wp.int32)
            self.tendon_seg_active_compliance = wp.array(
                model.tendon_seg_compliance.numpy().copy(), dtype=float, device=model.device
            )
            self.tendon_seg_active_damping = wp.array(
                model.tendon_seg_damping.numpy().copy(), dtype=float, device=model.device
            )
            self.tendon_guide_active = wp.ones(model.tendon_guide_count, dtype=bool)
            self.tendon_guide_active_step = wp.ones(model.tendon_guide_count, dtype=bool)
            self.tendon_guide_route_rest_length = wp.zeros(model.tendon_guide_count, dtype=float)
            self.tendon_guide_cone_seg_l = wp.full(model.tendon_guide_count, -1, dtype=wp.int32)
            self.tendon_guide_cone_seg_r = wp.full(model.tendon_guide_count, -1, dtype=wp.int32)
            self.tendon_guide_cap_ratio = wp.ones(model.tendon_guide_count, dtype=float)
            self.tendon_total_cable = wp.zeros(model.tendon_count, dtype=float)

            tendon_start_np = model.tendon_start.numpy()
            seg_guide_l = []
            guide_seg_left = np.full(model.tendon_guide_count, -1, dtype=np.int32)
            guide_tendon = np.empty(model.tendon_guide_count, dtype=np.int32)
            seg = 0
            for t in range(model.tendon_count):
                start = tendon_start_np[t]
                end = tendon_start_np[t + 1]
                guide_tendon[start:end] = t
                for guide_idx in range(start, end - 1):
                    seg_guide_l.append(guide_idx)
                    if guide_idx + 1 < end - 1:
                        guide_seg_left[guide_idx + 1] = seg
                    seg += 1

            self.tendon_seg_guide_l = wp.array(seg_guide_l, dtype=wp.int32, device=model.device)
            self.tendon_seg_active_guide_l = wp.array(seg_guide_l, dtype=wp.int32, device=model.device)
            self.tendon_seg_active_guide_r = wp.array(
                np.asarray(seg_guide_l, dtype=np.int32) + 1, dtype=wp.int32, device=model.device
            )
            self.tendon_guide_seg_left = wp.array(guide_seg_left, dtype=wp.int32, device=model.device)
            self.tendon_guide_tendon = wp.array(guide_tendon, dtype=wp.int32, device=model.device)

            rest_np = model.tendon_seg_rest_length.numpy().copy()
            auto_mask = rest_np < 0.0
            rest_np[auto_mask] = 0.0
            self.tendon_seg_rest_length = wp.array(rest_np, dtype=float, device=model.device)
            self.tendon_seg_rest_length_step = wp.array(rest_np.copy(), dtype=float, device=model.device)
            self.tendon_seg_route_rest_length = wp.array(rest_np.copy(), dtype=float, device=model.device)
            # scratch: per-segment stretch d = len - rest, snapshot+telescoped inside the capstan
            # transport (kept at its own scale so stiff-cable friction transfers survive float32)
            self.tendon_seg_stretch = wp.zeros_like(self.tendon_seg_rest_length)

            guide_type_np = model.tendon_guide_type.numpy()
            guide_flags_np = model.tendon_guide_flags.numpy()
            self._has_dynamic_tendon_guides = bool(
                np.any(
                    (guide_type_np == int(TendonGuideType.ROLLER))
                    & ((guide_flags_np & int(TendonGuideFlags.DYNAMIC)) != 0)
                )
            )
            if self._has_dynamic_tendon_guides and model.body_q is not None:
                # Resolve the initial topology before measuring its free-span rest lengths.
                self._update_tendon_guide_active(model, model.body_q)
                wp.copy(self.tendon_guide_active_step, self.tendon_guide_active)

            route_rest_np, route_seg_mask = self._compute_active_route_rest_lengths(model)
            self.tendon_guide_route_rest_length = wp.array(route_rest_np, dtype=float, device=model.device)

            self._init_tendon_attachment_points(model, auto_mask, route_seg_mask)
            self._cache_tendon_initial_state(model)

    def _snapshot_tendon_step_state(self) -> None:
        """Snapshot mutable tendon material state at the start of a time step."""
        if self.tendon_seg_rest_length is None:
            return

        wp.copy(self.tendon_seg_rest_length_step, self.tendon_seg_rest_length)
        wp.copy(self.tendon_seg_attachment_l_local_step, self.tendon_seg_attachment_l_local)
        wp.copy(self.tendon_seg_attachment_r_local_step, self.tendon_seg_attachment_r_local)
        if self._has_dynamic_tendon_guides:
            wp.launch(
                kernel=snapshot_tendon_guide_active,
                dim=self.tendon_guide_active.shape[0],
                inputs=[self.tendon_guide_active, self.tendon_guide_active_step, self.model.tendon_guide_flags],
                device=self.tendon_guide_active.device,
            )

    def _update_tendon_guide_active(self, model: Model, body_q: wp.array[wp.transform]) -> None:
        """Update solver-owned dynamic routing flags from the current body poses."""
        if not self._has_dynamic_tendon_guides:
            return

        wp.launch(
            kernel=update_tendon_guide_active,
            dim=model.tendon_count,
            inputs=[
                body_q,
                model.tendon_start,
                model.tendon_guide_body,
                model.tendon_guide_type,
                model.tendon_guide_flags,
                model.tendon_guide_radius,
                model.tendon_guide_orientation,
                model.tendon_guide_offset,
                model.tendon_guide_axis,
                self.tendon_activation_tol,
                self.tendon_guide_active,
            ],
            device=model.device,
        )

    def _prepare_tendon_route(
        self,
        model: Model,
        body_q: wp.array[wp.transform],
        compliance_floor: float = 0.0,
        initialize: bool = False,
    ) -> None:
        """Build the active route and merged segment properties for one solver step."""
        if model.tendon_segment_count == 0:
            return

        wp.launch(
            kernel=prepare_tendon_route,
            dim=model.tendon_count,
            inputs=[
                body_q,
                model.tendon_start,
                model.tendon_guide_body,
                model.tendon_guide_type,
                model.tendon_guide_flags,
                model.tendon_guide_radius,
                model.tendon_guide_offset,
                model.tendon_guide_axis,
                model.tendon_seg_rest_length,
                self.tendon_seg_rest_length_step,
                model.tendon_seg_compliance,
                model.tendon_seg_damping,
                self.tendon_guide_active,
                self.tendon_guide_active_step,
                self.tendon_guide_route_rest_length,
                self.tendon_seg_attachment_l_local_step,
                self.tendon_seg_attachment_r_local_step,
                int(initialize),
                compliance_floor,
            ],
            outputs=[
                self.tendon_seg_route_rest_length,
                self.tendon_seg_active,
                self.tendon_seg_active_guide_l,
                self.tendon_seg_active_guide_r,
                self.tendon_seg_active_compliance,
                self.tendon_seg_active_damping,
            ],
            device=model.device,
        )

    def _rebaseline_tendon_geometry(self, body_q: wp.array[wp.transform]) -> None:
        """Rebaseline reset-selected tangent history from the accepted pose."""
        if self.tendon_seg_rest_length is None:
            return

        wp.launch(
            kernel=rebaseline_tendon_attachments,
            dim=self.model.tendon_segment_count,
            inputs=[
                body_q,
                self.model.tendon_guide_body,
                self.model.tendon_guide_type,
                self.model.tendon_guide_radius,
                self.model.tendon_guide_orientation,
                self.model.tendon_guide_offset,
                self.model.tendon_guide_axis,
                self.tendon_seg_active,
                self.tendon_seg_active_guide_l,
                self.tendon_seg_active_guide_r,
                self._tendon_segment_world,
                self._tendon_pose_rebaseline_mask,
                self.model.world_count,
            ],
            outputs=[
                self.tendon_seg_attachment_l_local,
                self.tendon_seg_attachment_r_local,
                self.tendon_seg_attachment_l_local_step,
                self.tendon_seg_attachment_r_local_step,
                self.tendon_seg_attachment_l,
                self.tendon_seg_attachment_r,
                self.tendon_seg_rolling_delta_l,
                self.tendon_seg_rolling_delta_r,
                self.tendon_seg_length,
            ],
            device=self.model.device,
        )
        self._tendon_pose_rebaseline_mask.zero_()

    def _update_tendon_cone_rows(
        self,
        model: Model,
        body_q: wp.array[wp.transform],
        report_unsupported_wrap: bool,
    ) -> None:
        """Cache geometry-dependent segment pairs and capstan ratios for material rows."""
        wp.launch(
            kernel=update_tendon_cone_rows,
            dim=model.tendon_guide_count,
            inputs=[
                body_q,
                model.tendon_start,
                self.tendon_guide_tendon,
                model.tendon_guide_body,
                model.tendon_guide_type,
                model.tendon_guide_flags,
                model.tendon_guide_radius,
                model.tendon_guide_orientation,
                model.tendon_guide_mu,
                model.tendon_guide_offset,
                model.tendon_guide_axis,
                self.tendon_guide_active,
                self.tendon_seg_active,
                self.tendon_seg_active_guide_l,
                self.tendon_seg_active_guide_r,
                self.tendon_seg_attachment_l,
                self.tendon_seg_attachment_r,
                self.tendon_seg_length,
                int(report_unsupported_wrap),
            ],
            outputs=[
                self.tendon_guide_cone_seg_l,
                self.tendon_guide_cone_seg_r,
                self.tendon_guide_cap_ratio,
            ],
            device=model.device,
        )

    def _compute_active_route_rest_lengths(self, model: Model) -> tuple[np.ndarray, np.ndarray]:
        """Compute bypass material lengths for dynamically routed rolling guides."""
        route_rest = np.zeros(model.tendon_guide_count, dtype=np.float32)
        route_seg_mask = np.zeros(model.tendon_segment_count, dtype=bool)
        body_q = model.body_q
        if body_q is None:
            return route_rest, route_seg_mask

        tendon_start = model.tendon_start.numpy()
        guide_body = model.tendon_guide_body.numpy()
        guide_type = model.tendon_guide_type.numpy()
        guide_radius = model.tendon_guide_radius.numpy()
        guide_orientation = model.tendon_guide_orientation.numpy()
        guide_flags = model.tendon_guide_flags.numpy()
        guide_active = self.tendon_guide_active.numpy()
        guide_offset = model.tendon_guide_offset.numpy()
        guide_axis = model.tendon_guide_axis.numpy()
        body_q_np = body_q.numpy()

        seg_base = 0
        for t in range(model.tendon_count):
            start = tendon_start[t]
            end = tendon_start[t + 1]
            for i in range(start + 1, end - 1):
                if (
                    guide_type[i] != int(TendonGuideType.ROLLER)
                    or (guide_flags[i] & int(TendonGuideFlags.DYNAMIC)) == 0
                ):
                    continue

                left_seg = seg_base + (i - start) - 1
                right_seg = left_seg + 1
                if not guide_active[i]:
                    route_seg_mask[left_seg] = True
                    route_seg_mask[right_seg] = True

                guide_l = i - 1
                guide_r = i + 1
                pose_l = body_q_np[guide_body[guide_l]]
                pose_r = body_q_np[guide_body[guide_r]]
                center_l = _transform_point_np(pose_l, guide_offset[guide_l]).astype(np.float64)
                center_r = _transform_point_np(pose_r, guide_offset[guide_r]).astype(np.float64)
                normal_l = _transform_vector_np(pose_l, guide_axis[guide_l])
                normal_r = _transform_vector_np(pose_r, guide_axis[guide_r])
                p0, p1 = _segment_attachment_points_np(
                    center_l,
                    center_r,
                    int(guide_type[guide_l]),
                    int(guide_type[guide_r]),
                    float(guide_radius[guide_l]),
                    float(guide_radius[guide_r]),
                    int(guide_orientation[guide_l]),
                    int(guide_orientation[guide_r]),
                    normal_l,
                    normal_r,
                )
                route_rest[i] = float(np.linalg.norm(p1 - p0))

            seg_base += end - start - 1

        return route_rest, route_seg_mask

    def _init_tendon_attachment_points(self, model: Model, auto_mask: np.ndarray, route_seg_mask: np.ndarray) -> None:
        """Compute initial tendon tangent attachments and rest lengths."""
        body_q = model.body_q
        if body_q is None:
            return

        tendon_start_np = model.tendon_start.numpy()
        guide_body_np = model.tendon_guide_body.numpy()
        guide_offset_np = model.tendon_guide_offset.numpy()
        body_q_np = body_q.numpy()

        att_l = np.zeros((model.tendon_segment_count, 3), dtype=np.float32)
        att_r = np.zeros((model.tendon_segment_count, 3), dtype=np.float32)
        att_l_local = np.zeros((model.tendon_segment_count, 3), dtype=np.float32)
        att_r_local = np.zeros((model.tendon_segment_count, 3), dtype=np.float32)

        seg = 0
        for t in range(model.tendon_count):
            start = tendon_start_np[t]
            end = tendon_start_np[t + 1]
            for i in range(start, end - 1):
                body_l = guide_body_np[i]
                body_r = guide_body_np[i + 1]
                off_l = guide_offset_np[i]
                off_r = guide_offset_np[i + 1]
                att_l[seg] = _transform_point_np(body_q_np[body_l], off_l)
                att_r[seg] = _transform_point_np(body_q_np[body_r], off_r)
                att_l_local[seg] = off_l
                att_r_local[seg] = off_r
                seg += 1

        with wp.ScopedDevice(model.device):
            self.tendon_seg_attachment_l = wp.array(att_l, dtype=wp.vec3, device=model.device)
            self.tendon_seg_attachment_r = wp.array(att_r, dtype=wp.vec3, device=model.device)
            self.tendon_seg_attachment_l_local = wp.array(att_l_local, dtype=wp.vec3, device=model.device)
            self.tendon_seg_attachment_r_local = wp.array(att_r_local, dtype=wp.vec3, device=model.device)

        self._prepare_tendon_route(model, body_q, initialize=True)

        wp.launch(
            kernel=update_tendon_attachments,
            dim=model.tendon_segment_count,
            inputs=[
                body_q,
                model.tendon_start,
                self.tendon_guide_tendon,
                model.tendon_guide_body,
                model.tendon_guide_type,
                model.tendon_guide_flags,
                model.tendon_guide_radius,
                model.tendon_guide_orientation,
                model.tendon_guide_offset,
                model.tendon_guide_axis,
                self.tendon_seg_active,
                self.tendon_seg_active_guide_l,
                self.tendon_seg_active_guide_r,
                self.tendon_guide_active,
                self.tendon_guide_active_step,
                self.tendon_seg_attachment_l_local_step,
                self.tendon_seg_attachment_r_local_step,
                0,
            ],
            outputs=[
                self.tendon_seg_attachment_l,
                self.tendon_seg_attachment_r,
                self.tendon_seg_attachment_l_local,
                self.tendon_seg_attachment_r_local,
                self.tendon_seg_rolling_delta_l,
                self.tendon_seg_rolling_delta_r,
                self.tendon_seg_length,
            ],
            device=model.device,
        )

        wp.launch(
            kernel=solve_tendon_material,
            dim=model.tendon_count,
            inputs=[
                body_q,
                model.body_qd,
                body_q,
                model.body_com,
                model.tendon_start,
                model.tendon_guide_body,
                model.tendon_guide_type,
                model.tendon_guide_radius,
                model.tendon_guide_offset,
                model.tendon_guide_axis,
                self.tendon_seg_rest_length,
                self.tendon_seg_rest_length_step,
                self.tendon_seg_route_rest_length,
                self.tendon_seg_stretch,
                self.tendon_seg_damping_tension,
                self.tendon_seg_active,
                self.tendon_seg_active_guide_l,
                self.tendon_seg_active_guide_r,
                self.tendon_seg_active_compliance,
                self.tendon_seg_active_damping,
                self.tendon_guide_active,
                self.tendon_guide_active_step,
                self.tendon_guide_route_rest_length,
                self.tendon_seg_attachment_l,
                self.tendon_seg_attachment_r,
                self.tendon_seg_length,
                self.tendon_seg_attachment_l_local,
                self.tendon_seg_attachment_r_local,
                self.tendon_seg_rolling_delta_l,
                self.tendon_seg_rolling_delta_r,
                self.tendon_guide_cone_seg_l,
                self.tendon_guide_cone_seg_r,
                self.tendon_guide_cap_ratio,
                self.tendon_cone_sweep_count,
                0,
                0.0,
                0,
                0,
                0,
                self.tendon_max_sweeps,
                self.tendon_settle_tol,
            ],
            device=model.device,
        )

        att_l_np = self.tendon_seg_attachment_l.numpy()
        att_r_np = self.tendon_seg_attachment_r.numpy()
        rest_np = self.tendon_seg_rest_length.numpy()
        for i in range(model.tendon_segment_count):
            if auto_mask[i] and not route_seg_mask[i]:
                rest_np[i] = np.linalg.norm(att_r_np[i] - att_l_np[i])
        self.tendon_seg_rest_length = wp.array(rest_np, dtype=float, device=model.device)
        self._snapshot_tendon_step_state()

        guide_type_np = model.tendon_guide_type.numpy()
        guide_radius_np = model.tendon_guide_radius.numpy()
        guide_offset_np = model.tendon_guide_offset.numpy()
        guide_axis_np = model.tendon_guide_axis.numpy()
        guide_active_np = self.tendon_guide_active.numpy()
        seg_active_np = self.tendon_seg_active.numpy()
        seg_active_guide_l_np = self.tendon_seg_active_guide_l.numpy()
        seg_active_guide_r_np = self.tendon_seg_active_guide_r.numpy()

        total_cable = np.zeros(model.tendon_count, dtype=np.float32)
        seg = 0
        for t in range(model.tendon_count):
            start = tendon_start_np[t]
            end = tendon_start_np[t + 1]
            num_guides = end - start
            seg_base = seg
            cable_len = 0.0
            for s in range(num_guides - 1):
                if seg_active_np[seg_base + s] != 0:
                    cable_len += rest_np[seg_base + s]
            for i in range(start + 1, end - 1):
                if guide_type_np[i] == int(TendonGuideType.ROLLER):
                    if not guide_active_np[i]:
                        continue
                    body_idx = guide_body_np[i]
                    q = body_q_np[body_idx]
                    center = _transform_point_np(q, guide_offset_np[i])
                    normal = _transform_vector_np(q, guide_axis_np[i])
                    radius = guide_radius_np[i]
                    pt_left = None
                    pt_right = None
                    for s in range(num_guides - 1):
                        seg_idx = seg_base + s
                        if seg_active_np[seg_idx] == 0:
                            continue
                        if seg_active_guide_r_np[seg_idx] == i:
                            pt_left = att_r_np[seg_idx]
                        if seg_active_guide_l_np[seg_idx] == i:
                            pt_right = att_l_np[seg_idx]
                    if pt_left is None or pt_right is None:
                        continue

                    r_l = pt_left - center
                    r_r = pt_right - center
                    cross_val = np.dot(np.cross(r_l, r_r), normal)
                    dot_val = np.dot(r_l, r_r)
                    theta = abs(np.arctan2(cross_val, dot_val))
                    cable_len += theta * radius
            total_cable[t] = cable_len
            seg += num_guides - 1

        self.tendon_total_cable = wp.array(total_cable, dtype=float, device=model.device)

    def _cache_tendon_initial_state(self, model: Model) -> None:
        """Preserve the initialized tendon state for solver resets."""
        guide_body = model.tendon_guide_body.numpy()
        body_world = model.body_world.numpy()
        guide_world = body_world[guide_body]
        segment_left_guide = self.tendon_seg_guide_l.numpy()
        tendon_start = model.tendon_start.numpy()

        self._tendon_segment_world = wp.array(guide_world[segment_left_guide], dtype=wp.int32, device=model.device)
        self._tendon_guide_world = wp.array(guide_world, dtype=wp.int32, device=model.device)
        self._tendon_world = wp.array(guide_world[tendon_start[:-1]], dtype=wp.int32, device=model.device)
        self._tendon_pose_rebaseline_mask = wp.zeros(model.world_count + 1, dtype=wp.bool, device=model.device)

        for field in (*_TENDON_SEGMENT_STATE_FIELDS, *_TENDON_GUIDE_STATE_FIELDS, *_TENDON_STATE_FIELDS):
            self._tendon_initial_state[field] = wp.clone(getattr(self, field))

    def _reset_tendon_state(self, world_mask: wp.array[wp.bool] | None) -> None:
        """Restore initialized tendon material and routing state for selected worlds."""
        if self.tendon_seg_rest_length is None:
            return

        groups = (
            (_TENDON_SEGMENT_STATE_FIELDS, self._tendon_segment_world),
            (_TENDON_GUIDE_STATE_FIELDS, self._tendon_guide_world),
            (_TENDON_STATE_FIELDS, self._tendon_world),
        )
        for fields, entity_world in groups:
            for field in fields:
                current = getattr(self, field)
                initial = self._tendon_initial_state[field]
                if world_mask is None:
                    wp.copy(current, initial)
                else:
                    wp.launch(
                        kernel=reset_tendon_state_array,
                        dim=current.shape[0],
                        inputs=[entity_world, world_mask, self.model.world_count, initial],
                        outputs=[current],
                        device=current.device,
                    )

        if world_mask is None:
            self._tendon_pose_rebaseline_mask.fill_(True)
        else:
            wp.launch(
                kernel=mark_tendon_rebaseline_worlds,
                dim=world_mask.shape[0],
                inputs=[world_mask],
                outputs=[self._tendon_pose_rebaseline_mask],
                device=self.model.device,
            )
