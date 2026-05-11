# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Shared routed-tendon solver state helpers.

The routed tendon geometry is solver-independent: XPBD and VBD both need the
same tangent attachments, mutable free-span rest lengths, and segment-to-link
mapping before applying their own numerical solve.
"""

from __future__ import annotations

import numpy as np
import warp as wp

from ..sim import Model
from ..sim.tendon import TendonLinkType
from .tendon_kernels import update_tendon_attachments


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


class TendonStateMixin:
    """Mixin that allocates routed-tendon mutable state on a solver instance."""

    def _init_tendon_state(self, model: Model, allocate_xpbd_lambdas: bool = True) -> None:
        """Allocate mutable tendon state arrays and build segment/link mappings."""
        if model.tendon_segment_count == 0:
            self.tendon_seg_rest_length = None
            self.tendon_seg_attachment_l = None
            self.tendon_seg_attachment_r = None
            self.tendon_seg_attachment_l_local = None
            self.tendon_seg_attachment_r_local = None
            self.tendon_seg_lambda = None
            self.tendon_seg_delta_lambda = None
            self.tendon_seg_rolling_delta_l = None
            self.tendon_seg_rolling_delta_r = None
            self.tendon_seg_link_l = None
            self.tendon_link_seg_left = None
            self.tendon_total_cable = None
            return

        with wp.ScopedDevice(model.device):
            self.tendon_seg_attachment_l = wp.zeros(model.tendon_segment_count, dtype=wp.vec3)
            self.tendon_seg_attachment_r = wp.zeros(model.tendon_segment_count, dtype=wp.vec3)
            self.tendon_seg_attachment_l_local = wp.zeros(model.tendon_segment_count, dtype=wp.vec3)
            self.tendon_seg_attachment_r_local = wp.zeros(model.tendon_segment_count, dtype=wp.vec3)
            self.tendon_seg_lambda = (
                wp.zeros(model.tendon_segment_count, dtype=float) if allocate_xpbd_lambdas else None
            )
            self.tendon_seg_delta_lambda = (
                wp.zeros(model.tendon_segment_count, dtype=float) if allocate_xpbd_lambdas else None
            )
            self.tendon_seg_rolling_delta_l = wp.zeros(model.tendon_segment_count, dtype=float)
            self.tendon_seg_rolling_delta_r = wp.zeros(model.tendon_segment_count, dtype=float)
            self.tendon_total_cable = wp.zeros(model.tendon_count, dtype=float)

            tendon_start_np = model.tendon_start.numpy()
            seg_link_l = []
            link_seg_left = np.full(model.tendon_link_count, -1, dtype=np.int32)
            seg = 0
            for t in range(model.tendon_count):
                start = tendon_start_np[t]
                end = tendon_start_np[t + 1]
                for link_idx in range(start, end - 1):
                    seg_link_l.append(link_idx)
                    if link_idx + 1 < end - 1:
                        link_seg_left[link_idx + 1] = seg
                    seg += 1

            self.tendon_seg_link_l = wp.array(seg_link_l, dtype=wp.int32, device=model.device)
            self.tendon_link_seg_left = wp.array(link_seg_left, dtype=wp.int32, device=model.device)

            rest_np = model.tendon_seg_rest_length.numpy().copy()
            auto_mask = rest_np < 0.0
            rest_np[auto_mask] = 0.0
            self.tendon_seg_rest_length = wp.array(rest_np, dtype=float, device=model.device)

            self._init_tendon_attachment_points(model, auto_mask)

    def _init_tendon_attachment_points(self, model: Model, auto_mask: np.ndarray) -> None:
        """Compute initial tendon tangent attachments and auto rest lengths."""
        body_q = model.body_q
        if body_q is None:
            return

        tendon_start_np = model.tendon_start.numpy()
        link_body_np = model.tendon_link_body.numpy()
        link_offset_np = model.tendon_link_offset.numpy()
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
                body_l = link_body_np[i]
                body_r = link_body_np[i + 1]
                off_l = link_offset_np[i]
                off_r = link_offset_np[i + 1]
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

        wp.launch(
            kernel=update_tendon_attachments,
            dim=model.tendon_count,
            inputs=[
                body_q,
                model.tendon_start,
                model.tendon_link_body,
                model.tendon_link_type,
                model.tendon_link_radius,
                model.tendon_link_orientation,
                model.tendon_link_mu,
                model.tendon_link_offset,
                model.tendon_link_axis,
                self.tendon_seg_rest_length,
                model.tendon_seg_compliance,
                self.tendon_seg_attachment_l,
                self.tendon_seg_attachment_r,
                self.tendon_seg_attachment_l_local,
                self.tendon_seg_attachment_r_local,
                self.tendon_seg_rolling_delta_l,
                self.tendon_seg_rolling_delta_r,
                0,
                0,
            ],
            device=model.device,
        )

        att_l_np = self.tendon_seg_attachment_l.numpy()
        att_r_np = self.tendon_seg_attachment_r.numpy()
        rest_np = self.tendon_seg_rest_length.numpy()
        for i in range(model.tendon_segment_count):
            if auto_mask[i]:
                rest_np[i] = np.linalg.norm(att_r_np[i] - att_l_np[i])
        self.tendon_seg_rest_length = wp.array(rest_np, dtype=float, device=model.device)

        link_type_np = model.tendon_link_type.numpy()
        link_radius_np = model.tendon_link_radius.numpy()
        link_offset_np = model.tendon_link_offset.numpy()
        link_axis_np = model.tendon_link_axis.numpy()

        total_cable = np.zeros(model.tendon_count, dtype=np.float32)
        seg = 0
        for t in range(model.tendon_count):
            start = tendon_start_np[t]
            end = tendon_start_np[t + 1]
            num_links = end - start
            seg_base = seg
            cable_len = 0.0
            for s in range(num_links - 1):
                cable_len += rest_np[seg_base + s]
            for i in range(start + 1, end - 1):
                if link_type_np[i] == int(TendonLinkType.ROLLING):
                    body_idx = link_body_np[i]
                    q = body_q_np[body_idx]
                    center = _transform_point_np(q, link_offset_np[i])
                    normal = _transform_vector_np(q, link_axis_np[i])
                    radius = link_radius_np[i]
                    seg_left = seg_base + (i - start) - 1
                    seg_right = seg_base + (i - start)
                    pt_dep = att_r_np[seg_left]
                    pt_arr = att_l_np[seg_right]
                    r_l = pt_dep - center
                    r_r = pt_arr - center
                    cross_val = np.dot(np.cross(r_l, r_r), normal)
                    dot_val = np.dot(r_l, r_r)
                    theta = abs(np.arctan2(cross_val, dot_val))
                    cable_len += theta * radius
            total_cable[t] = cable_len
            seg += num_links - 1

        self.tendon_total_cable = wp.array(total_cable, dtype=float, device=model.device)
