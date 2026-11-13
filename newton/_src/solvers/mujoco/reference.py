# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Reference coordinates shared by model finalization and MuJoCo export."""

from __future__ import annotations

import warp as wp

from ...sim.articulation import eval_fk
from ...sim.model import Model
from ...sim.state import State
from .kernels import build_ref_q_kernel


def build_ref_q(model: Model, ref_q: wp.array | None = None) -> wp.array:
    """Use zero scalar coordinates, identity ball rotations, and initial free-body poses."""
    if ref_q is None:
        ref_q = wp.zeros(model.joint_coord_count, dtype=wp.float32, device=model.device)
    wp.launch(
        kernel=build_ref_q_kernel,
        dim=model.joint_count,
        inputs=[
            model.joint_type,
            model.joint_q,
            model.joint_q_start,
            model.joint_dof_dim,
        ],
        outputs=[
            ref_q,
        ],
        device=model.device,
    )
    return ref_q


def compute_body_poses_at_qref(
    model: Model,
    ref_q: wp.array,
    ref_qd: wp.array | None = None,
    ref_body_q: wp.array | None = None,
    ref_body_qd: wp.array | None = None,
) -> wp.array:
    """Evaluate the supplied reference coordinates with zero velocities."""
    if ref_qd is None:
        ref_qd = wp.zeros(model.joint_dof_count, dtype=wp.float32, device=model.device)
    if ref_body_q is None:
        ref_body_q = wp.zeros(model.body_count, dtype=wp.transform, device=model.device)
    if ref_body_qd is None:
        ref_body_qd = wp.zeros(model.body_count, dtype=wp.spatial_vector, device=model.device)

    ref_state = State()
    ref_state.body_q = ref_body_q
    ref_state.body_qd = ref_body_qd
    eval_fk(model, ref_q, ref_qd, ref_state)
    return ref_body_q
