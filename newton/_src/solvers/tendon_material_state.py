# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Pack active guide routes for the linear direct material projection.

This workspace contains temporary trials and validated active-face caches, not
physical material history. Callers provide stretch from the current trajectory
and publish accepted rest lengths to State. A failed projection publishes neither
stretch nor slip; the caller decides whether the pose is provisional or accepted.
"""

import numpy as np
import warp as wp

from ..sim import Model
from ..sim.tendon import TendonGuideType
from .tendon_kernels import _initialize_tendon_material
from .tendon_material import TendonMaterialStatus, solve_tendon_material_component


@wp.struct
class _TendonMaterialWorkspace:
    previous_rest: wp.array[float]
    trial: wp.array[float]
    initial: wp.array[float]
    compliance: wp.array[float]
    cap: wp.array[float]
    upper: wp.array[float]
    output: wp.array[float]
    transfer: wp.array[wp.float64]
    status: wp.array[int]
    faces: wp.array[int]
    valid: wp.array[int]
    pieces: wp.array[int]
    count: wp.array[int]
    ids: wp.array[int]
    edge_ids: wp.array[int]
    next_guide: wp.array[int]
    incoming: wp.array[int]
    failed_component: wp.array[int]


def _create_tendon_material_workspace(model: Model) -> _TendonMaterialWorkspace:
    """Allocate packed scratch proportional to the number of authored spans."""
    if model.requires_grad:
        raise ValueError("The direct tendon material solve does not support differentiable simulation")
    if model.tendon_segment_count:
        compliance = model.tendon_segment_compliance.numpy()
        if not np.all(np.isfinite(compliance) & (compliance >= 1.0e-25)):
            raise ValueError("The direct tendon material solve requires finite compliance >= 1e-25 m/N")
        starts = model.tendon_start.numpy()
        kinds = model.tendon_guide_type.numpy()
        for tendon in range(model.tendon_count):
            count = 0
            for kind in kinds[starts[tendon] + 1 : starts[tendon + 1]]:
                count += 1
                if count > 32:
                    raise ValueError(
                        "The direct tendon material solve supports at most 32 authored spans "
                        f"between anchors (tendon {tendon})"
                    )
                if kind == int(TendonGuideType.ANCHOR):
                    count = 0
    workspace = _TendonMaterialWorkspace()
    for name in ("previous_rest", "trial", "initial", "compliance", "cap", "upper", "output"):
        setattr(workspace, name, wp.zeros(model.tendon_segment_count, dtype=float, device=model.device))
    workspace.transfer = wp.zeros(model.tendon_segment_count, dtype=wp.float64, device=model.device)
    for name in ("status", "faces", "valid", "pieces", "count", "next_guide", "incoming"):
        setattr(workspace, name, wp.zeros(model.tendon_segment_count, dtype=int, device=model.device))
    for name in ("ids", "edge_ids"):
        setattr(workspace, name, wp.full(model.tendon_segment_count, -1, dtype=int, device=model.device))
    workspace.failed_component = wp.full(model.tendon_count, -1, dtype=int, device=model.device)
    return workspace


@wp.func
def _reject_material(workspace: _TendonMaterialWorkspace, tendon: int, row: int, code: int) -> bool:
    workspace.status[row] = code
    workspace.failed_component[tendon] = row
    return False


@wp.func
def _project_component(workspace: _TendonMaterialWorkspace, tendon: int, row: int, count: int) -> bool:
    if workspace.count[row] != count:
        workspace.valid[row] = 0
    workspace.count[row] = count
    solve_tendon_material_component(
        workspace.initial,
        workspace.compliance,
        workspace.cap,
        workspace.upper,
        workspace.output,
        workspace.transfer,
        workspace.status,
        workspace.faces,
        workspace.valid,
        workspace.pieces,
        count,
        1,
        row,
        1,
        1,
    )
    if workspace.status[row] < 0:
        workspace.failed_component[tendon] = row
        return False
    return True


@wp.func
def _project_tendon_material(
    workspace: _TendonMaterialWorkspace,
    tendon: int,
    guide_start: int,
    guide_end: int,
    segment_start: int,
    segment_end: int,
    guide_type: wp.array[int],
    guide_active: wp.array[bool],
    cone_segment_l: wp.array[int],
    cone_segment_r: wp.array[int],
    cap_ratio: wp.array[float],
    segment_active: wp.array[int],
    raw_compliance: wp.array[float],
    compliance: wp.array[float],
    damping_force: wp.array[float],
    length: wp.array[float],
    rolling_delta_l: wp.array[float],
    rolling_delta_r: wp.array[float],
    apply_rolling: int,
    apply_pinhole: int,
    min_rest: float,
    stretch: wp.array[float],
    material_transfer: wp.array[float],
) -> bool:
    """Project a complete tendon, preserving outputs if any component is rejected.

    Stretch and rolling increments are measured from the same step snapshot.
    Damping is a frozen signed force [N], as in the current linear material law.
    Returned transfer is net sliding relative to the full no-slip rolling trial.
    The caller must inspect the result on the accepted pose, not just trial poses.
    """
    workspace.failed_component[tendon] = -1
    for seg in range(segment_start, segment_end):
        if not wp.isfinite(raw_compliance[seg]) or raw_compliance[seg] < 1.0e-25:
            return _reject_material(workspace, tendon, segment_start, int(TendonMaterialStatus.BAD_INPUT))
        workspace.next_guide[seg] = -1
        workspace.incoming[seg] = -1
        workspace.trial[seg] = stretch[seg]

    for guide in range(guide_start + 1, guide_end - 1):
        kind = guide_type[guide]
        rolling = kind == int(TendonGuideType.ROLLER) and guide_active[guide]
        pinhole = kind == int(TendonGuideType.PINHOLE)
        if not ((apply_rolling != 0 and rolling) or (apply_pinhole != 0 and pinhole)):
            continue
        left = cone_segment_l[guide]
        right = cone_segment_r[guide]
        if left < 0 or right < 0:
            continue
        if left < segment_start or right >= segment_end or left >= right:
            return _reject_material(workspace, tendon, segment_start, int(TendonMaterialStatus.INVALID_ROUTE))
        if (
            segment_active[left] == 0
            or segment_active[right] == 0
            or workspace.next_guide[left] >= 0
            or workspace.incoming[right] >= 0
        ):
            return _reject_material(workspace, tendon, segment_start, int(TendonMaterialStatus.INVALID_ROUTE))
        for skipped in range(left + 1, right):
            if segment_active[skipped] != 0:
                return _reject_material(workspace, tendon, segment_start, int(TendonMaterialStatus.INVALID_ROUTE))
        workspace.next_guide[left] = guide
        workspace.incoming[right] = guide
        if rolling:
            # Preserve the validated float32 order for common arc exchange and
            # differential transport. Do not clip this trial before projection.
            common = 0.5 * (rolling_delta_r[left] + rolling_delta_l[right])
            differential = 0.5 * (rolling_delta_r[left] - rolling_delta_l[right])
            workspace.trial[left] = workspace.trial[left] - common
            workspace.trial[right] = workspace.trial[right] - common
            workspace.trial[left] = workspace.trial[left] - differential
            workspace.trial[right] = workspace.trial[right] + differential

    row = int(-1)
    count = int(0)
    previous = int(-1)
    for seg in range(segment_start, segment_end):
        if segment_active[seg] == 0:
            continue
        guide = int(-1)
        if previous >= 0:
            guide = workspace.next_guide[previous]
        if guide < 0:
            if count > 0 and not _project_component(workspace, tendon, row, count):
                return False
            row = seg
            count = 0
        if count >= 32:
            return _reject_material(workspace, tendon, row, int(TendonMaterialStatus.COMPONENT_TOO_LARGE))
        index = row + count
        if workspace.ids[index] != seg:
            workspace.valid[row] = 0
        workspace.ids[index] = seg
        offset = compliance[seg] * damping_force[seg]
        workspace.initial[index] = workspace.trial[seg] + offset
        workspace.compliance[index] = compliance[seg]
        workspace.upper[index] = (length[seg] - min_rest) + offset
        if count > 0:
            edge = index - 1
            if workspace.edge_ids[edge] != guide:
                workspace.valid[row] = 0
            workspace.edge_ids[edge] = guide
            workspace.cap[edge] = cap_ratio[guide]
        count += 1
        previous = seg
    if count > 0 and not _project_component(workspace, tendon, row, count):
        return False

    # A later failing component must not publish an earlier successful component.
    for guide in range(guide_start, guide_end):
        material_transfer[guide] = 0.0
    row = int(-1)
    count = int(0)
    previous = int(-1)
    transfer = wp.float64(0.0)
    for seg in range(segment_start, segment_end):
        if segment_active[seg] == 0:
            continue
        new_component = previous < 0
        if previous >= 0:
            new_component = workspace.next_guide[previous] < 0
        if new_component:
            row = seg
            count = 0
            transfer = wp.float64(0.0)
        incoming_transfer = transfer
        transfer = workspace.transfer[row + count]
        # Apply only conservative sliding to the physical rolling trial. Do not
        # recover stretch by subtracting damping from a rounded effective extension.
        stretch[seg] = float(wp.float64(workspace.trial[seg]) + incoming_transfer - transfer)
        guide = workspace.next_guide[seg]
        if guide >= 0:
            material_transfer[guide] = float(transfer)
        count += 1
        previous = seg
    return True


@wp.kernel(enable_backward=False)
def solve_tendon_material_direct(
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    body_q_prev: wp.array[wp.transform],
    body_com: wp.array[wp.vec3],
    tendon_start: wp.array[int],
    tendon_guide_body: wp.array[int],
    tendon_guide_type: wp.array[int],
    tendon_guide_radius: wp.array[float],
    tendon_guide_offset: wp.array[wp.vec3],
    tendon_guide_axis: wp.array[wp.vec3],
    seg_rest_length: wp.array[float],
    seg_rest_length_step: wp.array[float],
    seg_route_rest_length: wp.array[float],
    seg_stretch: wp.array[float],
    seg_damping_tension: wp.array[float],
    seg_active: wp.array[int],
    seg_active_guide_l: wp.array[int],
    seg_active_guide_r: wp.array[int],
    seg_active_compliance: wp.array[float],
    seg_active_damping: wp.array[float],
    tendon_guide_active: wp.array[bool],
    tendon_guide_active_step: wp.array[bool],
    tendon_guide_route_rest_length: wp.array[float],
    seg_attachment_l: wp.array[wp.vec3],
    seg_attachment_r: wp.array[wp.vec3],
    seg_length: wp.array[float],
    seg_attachment_l_local: wp.array[wp.vec3],
    seg_attachment_r_local: wp.array[wp.vec3],
    seg_rolling_delta_l: wp.array[float],
    seg_rolling_delta_r: wp.array[float],
    tendon_guide_cone_seg_l: wp.array[int],
    tendon_guide_cone_seg_r: wp.array[int],
    tendon_guide_cap_ratio: wp.array[float],
    tendon_cone_sweep_count: wp.array[int],
    damping_from_pose_delta: int,
    dt: float,
    apply_rolling_transfer: int,
    apply_pinhole_slip: int,
    guide_material_transfer: wp.array[float],
    workspace: _TendonMaterialWorkspace,
    raw_compliance: wp.array[float],
    failure: wp.array[wp.vec2i],
    accepted_pose: int,
):
    """Solve the linear material return map; latch rejection only on accepted poses."""
    tendon_id = wp.tid()
    guide_start = tendon_start[tendon_id]
    guide_end = tendon_start[tendon_id + 1]
    segment_start = guide_start - tendon_id
    segment_end = guide_end - tendon_id - 1
    for seg in range(segment_start, segment_end):
        workspace.previous_rest[seg] = seg_rest_length[seg]
    _initialize_tendon_material(
        tendon_id,
        body_q,
        body_qd,
        body_q_prev,
        body_com,
        tendon_start,
        tendon_guide_body,
        tendon_guide_type,
        tendon_guide_radius,
        tendon_guide_offset,
        tendon_guide_axis,
        seg_rest_length,
        seg_rest_length_step,
        seg_route_rest_length,
        seg_stretch,
        seg_damping_tension,
        seg_active,
        seg_active_guide_l,
        seg_active_guide_r,
        seg_active_damping,
        tendon_guide_active,
        tendon_guide_active_step,
        tendon_guide_route_rest_length,
        seg_attachment_l,
        seg_attachment_r,
        seg_length,
        seg_attachment_l_local,
        seg_attachment_r_local,
        tendon_cone_sweep_count,
        damping_from_pose_delta,
        dt,
        apply_rolling_transfer,
        apply_pinhole_slip,
        0,
        guide_material_transfer,
    )
    valid = _project_tendon_material(
        workspace,
        tendon_id,
        guide_start,
        guide_end,
        segment_start,
        segment_end,
        tendon_guide_type,
        tendon_guide_active,
        tendon_guide_cone_seg_l,
        tendon_guide_cone_seg_r,
        tendon_guide_cap_ratio,
        seg_active,
        raw_compliance,
        seg_active_compliance,
        seg_damping_tension,
        seg_length,
        seg_rolling_delta_l,
        seg_rolling_delta_r,
        apply_rolling_transfer,
        apply_pinhole_slip,
        1.0e-6,
        seg_stretch,
        guide_material_transfer,
    )
    if not valid:
        # Provisional iterates may be infeasible. Preserve the previous material
        # allocation and retry after the body update; never fall back to sweeps.
        for seg in range(segment_start, segment_end):
            seg_rest_length[seg] = workspace.previous_rest[seg]
            seg_stretch[seg] = seg_length[seg] - seg_rest_length[seg]
        if accepted_pose != 0 and failure[tendon_id][0] == 0:
            row = workspace.failed_component[tendon_id]
            code = workspace.status[row]
            failure[tendon_id] = wp.vec2i(code, row)
            wp.printf(
                "Error: direct tendon material solve rejected accepted pose "
                "(tendon=%d, component=%d, status=%d). "
                "Discard this trajectory and call solver.check_tendon_material(state).\n",
                tendon_id,
                row,
                code,
            )
        return
    for seg in range(segment_start, segment_end):
        if seg_active[seg] != 0:
            seg_rest_length[seg] = wp.max(seg_length[seg] - seg_stretch[seg], 1.0e-6)
