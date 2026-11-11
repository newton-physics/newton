# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Portable (CPU) twins of the matrix-free solve's native CUDA row kernels.

The fused matrix-free Gauss-Seidel kernel and the propagation contact kernel
are warp-cooperative CUDA snippets that compile to no-ops on the CPU. The
kernels here run one thread per world and visit the rows in the same order
with the same update math, so a CPU solve tracks the CUDA one to float
round-off (the CUDA kernels reduce dot products in a different order).

Only the configurations the CPU accepts are mirrored: ``friction_mode=
"fixed_normal"`` without contact torsion, contact compliance or joint friction.
"""

import warp as wp

from .friction import friction_pair_candidate
from .friction_patches import WARP_DETERMINISM_OFF, patch_normal_load
from .kernels import (
    PGS_CONSTRAINT_TYPE_CONTACT,
    PGS_CONSTRAINT_TYPE_FRICTION,
    PGS_CONSTRAINT_TYPE_JOINT_LIMIT,
    PGS_CONSTRAINT_TYPE_JOINT_TARGET,
    PGS_CONSTRAINT_TYPE_JOINT_VELOCITY_LIMIT,
)

wp.set_module_options({"enable_backward": False, **WARP_DETERMINISM_OFF})


def reject_cpu_unsupported_features(solver, *, enable_sleeping: bool) -> None:
    """Raise :class:`NotImplementedError` for matrix-free features that only have native CUDA row kernels.

    Contact torsion rejects CPU devices when it is configured, so it is not repeated here.
    """
    unsupported = []
    if solver._contact_compliance:
        unsupported.append("contact_compliance=True")
    if solver._friction_mode != "fixed_normal":
        unsupported.append(f"friction_mode={solver._friction_mode!r}")
    if solver._enable_joint_friction:
        unsupported.append("enable_joint_friction=True")
    if unsupported:
        raise NotImplementedError(
            "SolverSteel on CPU does not support " + ", ".join(unsupported) + "; these features require CUDA."
        )
    if enable_sleeping and solver._solve_mode == "colored":
        raise NotImplementedError(
            "Experimental sleeping with solve_mode='colored' requires CUDA: on CPU the colored mode runs the "
            "serial propagation sweep, which sleeping does not support"
        )


@wp.func
def _dense_dot(J_world: wp.array3d[float], world: int, row: int, v: wp.array2d[float], dofs: int):
    s = float(0.0)
    for d in range(dofs):
        s += J_world[world, row, d] * v[world, d]
    return s


@wp.func
def _dense_apply(Y_world: wp.array3d[float], world: int, row: int, delta: float, v: wp.array2d[float], dofs: int):
    for d in range(dofs):
        v[world, d] = v[world, d] + Y_world[world, row, d] * delta


@wp.func
def _mf_dot(
    J_a: wp.array3d[float],
    J_b: wp.array3d[float],
    world: int,
    row: int,
    dof_a: int,
    dof_b: int,
    v: wp.array2d[float],
):
    s = float(0.0)
    for k in range(6):
        if dof_a >= 0:
            s += J_a[world, row, k] * v[world, dof_a + k]
        if dof_b >= 0:
            s += J_b[world, row, k] * v[world, dof_b + k]
    return s


@wp.func
def _mf_apply(
    MiJt_a: wp.array3d[float],
    MiJt_b: wp.array3d[float],
    world: int,
    row: int,
    dof_a: int,
    dof_b: int,
    delta: float,
    v: wp.array2d[float],
):
    for k in range(6):
        if dof_a >= 0:
            v[world, dof_a + k] = v[world, dof_a + k] + MiJt_a[world, row, k] * delta
        if dof_b >= 0:
            v[world, dof_b + k] = v[world, dof_b + k] + MiJt_b[world, row, k] * delta


@wp.kernel(enable_backward=False)
def pgs_solve_mf_gs_cpu(
    max_world_dofs: int,
    world_constraint_count: wp.array[int],
    world_dof_indices: wp.array2d[int],
    rhs_bias: wp.array2d[float],
    world_diag: wp.array2d[float],
    world_row_w: wp.array2d[float],
    J_world: wp.array3d[float],
    Y_world: wp.array3d[float],
    world_row_type: wp.array2d[int],
    world_row_parent: wp.array2d[int],
    has_drive_rows: int,
    fuse_vel_limits: int,
    drive_target_vel_bias: wp.array2d[float],
    drive_vel_multiplier: wp.array2d[float],
    drive_impulse_multiplier: wp.array2d[float],
    drive_max_impulse: wp.array2d[float],
    drive_vel_limit: wp.array2d[float],
    mf_constraint_count: wp.array[int],
    mf_contact_rows_end: wp.array[int],
    mf_dof_a: wp.array2d[int],
    mf_dof_b: wp.array2d[int],
    mf_eff_mass_inv: wp.array2d[float],
    mf_rhs: wp.array2d[float],
    mf_row_type: wp.array2d[int],
    mf_row_parent: wp.array2d[int],
    mf_J_a: wp.array3d[float],
    mf_J_b: wp.array3d[float],
    mf_MiJt_a: wp.array3d[float],
    mf_MiJt_b: wp.array3d[float],
    world_row_mu: wp.array2d[float],
    mf_row_mu: wp.array2d[float],
    mf_row_w: wp.array2d[float],
    iterations: int,
    omega: float,
    regularize: int,
    freeze_drive_rows: int,
    row_phase: int,
    friction_start_iteration: int,
    iteration_offset: int,
    # in/out
    world_impulses: wp.array2d[float],
    mf_impulses: wp.array2d[float],
    v_scratch: wp.array2d[float],
    v_out: wp.array[float],
):
    """World-serial twin of ``_get_pgs_solve_mf_gs_kernel`` (fixed-normal friction).

    Per sweep: the dense rows through ``J_world``/``Y_world``, then the free-body contact
    rows on their 6+6 body DOFs, then the velocity limits (dense velocity-limit rows, the
    fused clamp of driven DOFs and the free-body velocity-limit rows). ``row_phase``
    restricts a sweep to row families as the CUDA kernel does: ``1`` and ``4`` the
    contact rows, ``2`` the internal rows and velocity limits, ``3`` the internal rows,
    ``5`` the velocity limits and ``0`` every family. The world velocity is staged in
    ``v_scratch`` as the CUDA kernel stages it in shared memory.
    """
    world = wp.tid()
    m_dense = world_constraint_count[world]
    m_mf = mf_constraint_count[world]
    if m_dense == 0 and m_mf == 0:
        return
    m_dense = wp.min(m_dense, world_impulses.shape[1])
    m_mf = wp.min(m_mf, mf_impulses.shape[1])
    # Free-body rows are laid out as [contacts and friction][velocity limits].
    mf_contact_end = wp.min(mf_contact_rows_end[world], m_mf)
    D = max_world_dofs

    dense_main_end = m_dense
    mf_main_end = mf_contact_end
    velocity_limit_pass = int(1)
    contact_rows = int(1)
    internal_rows = int(1)
    if row_phase == 1 or row_phase == 4:
        internal_rows = 0
        velocity_limit_pass = 0
    elif row_phase == 2 or row_phase == 3:
        contact_rows = 0
        mf_main_end = 0
        if row_phase == 3:
            velocity_limit_pass = 0
    elif row_phase == 5:
        dense_main_end = 0
        mf_main_end = 0
        contact_rows = 0

    for d in range(D):
        global_dof = world_dof_indices[world, d]
        if global_dof >= 0:
            v_scratch[world, d] = v_out[global_dof]
        else:
            v_scratch[world, d] = 0.0

    for it in range(iterations):
        global_iter = iteration_offset + it
        changed = int(0)

        # Dense rows.
        for i in range(dense_main_end):
            row_type = world_row_type[world, i]
            if row_type == PGS_CONSTRAINT_TYPE_JOINT_VELOCITY_LIMIT:
                continue
            if freeze_drive_rows != 0 and row_type == PGS_CONSTRAINT_TYPE_JOINT_TARGET:
                continue
            is_contact_row = row_type == PGS_CONSTRAINT_TYPE_CONTACT or row_type == PGS_CONSTRAINT_TYPE_FRICTION
            if is_contact_row and contact_rows == 0:
                continue
            if not is_contact_row and internal_rows == 0:
                continue
            parent = world_row_parent[world, i]
            if row_type == PGS_CONSTRAINT_TYPE_FRICTION:
                if global_iter < friction_start_iteration:
                    world_impulses[world, i] = 0.0
                    continue
                # The pair solve at the first friction row also writes the second.
                if i != parent + 1:
                    continue
            denom = world_diag[world, i]
            if denom <= 0.0 and row_type != PGS_CONSTRAINT_TYPE_FRICTION:
                continue

            jv = _dense_dot(J_world, world, i, v_scratch, D)
            old_impulse = world_impulses[world, i]
            residual = jv + rhs_bias[world, i]
            # Regularized rows (weight w < 1) also relax the impulse toward zero.
            w = float(1.0)
            if regularize != 0:
                w = world_row_w[world, i]
            delta = float(0.0)
            if denom > 0.0:
                delta = -residual / denom * w - (1.0 - w) * old_impulse
            new_impulse = old_impulse + omega * delta

            if row_type == PGS_CONSTRAINT_TYPE_JOINT_TARGET and has_drive_rows != 0:
                # PhysX force-drive update; not relaxed by omega.
                max_impulse = drive_max_impulse[world, i]
                new_impulse = (
                    old_impulse * drive_impulse_multiplier[world, i]
                    + jv * drive_vel_multiplier[world, i]
                    + drive_target_vel_bias[world, i]
                )
                new_impulse = wp.min(wp.max(new_impulse, -max_impulse), max_impulse)
            elif row_type == PGS_CONSTRAINT_TYPE_FRICTION:
                sib = parent + 2
                # A patch's normal rows form a cycle through their parent links and share its load.
                radius = wp.max(
                    world_row_mu[world, i] * patch_normal_load(world_row_parent, world_impulses, world, parent), 0.0
                )
                sibling_residual = rhs_bias[world, sib]
                cross = float(0.0)
                for d in range(D):
                    sibling_residual += J_world[world, sib, d] * v_scratch[world, d]
                    cross += J_world[world, i, d] * Y_world[world, sib, d]
                old_sib = world_impulses[world, sib]
                pair = friction_pair_candidate(
                    denom,
                    cross,
                    world_diag[world, sib],
                    wp.vec2(residual, sibling_residual),
                    wp.vec2(old_impulse, old_sib),
                    radius,
                    omega,
                )
                mag = wp.length(pair)
                scale = float(1.0)
                if mag > radius:
                    scale = radius / mag
                new_impulse = pair[0] * scale
                sib_delta = pair[1] * scale - old_sib
                world_impulses[world, sib] = pair[1] * scale
                if sib_delta != 0.0:
                    changed = 1
                    _dense_apply(Y_world, world, sib, sib_delta, v_scratch, D)
            elif new_impulse < 0.0 and (
                row_type == PGS_CONSTRAINT_TYPE_CONTACT or row_type == PGS_CONSTRAINT_TYPE_JOINT_LIMIT
            ):
                # Contact and joint-limit rows are unilateral; mimic and connect rows are bilateral.
                new_impulse = 0.0

            delta_impulse = new_impulse - old_impulse
            world_impulses[world, i] = new_impulse
            if delta_impulse != 0.0:
                changed = 1
                _dense_apply(Y_world, world, i, delta_impulse, v_scratch, D)

        # Free-body contact and friction rows.
        i = int(0)
        while i < mf_main_end:
            rt = mf_row_type[world, i]
            parent = mf_row_parent[world, i]
            cur = i
            # The first row of a friction pair also solves the second, so the sweep steps over it.
            i = cur + 1
            if rt == PGS_CONSTRAINT_TYPE_FRICTION and cur == parent + 1:
                i = cur + 2
            if rt == PGS_CONSTRAINT_TYPE_FRICTION and global_iter < friction_start_iteration:
                mf_impulses[world, cur] = 0.0
                if i == cur + 2:
                    mf_impulses[world, cur + 1] = 0.0
                continue
            if rt == PGS_CONSTRAINT_TYPE_FRICTION and cur != parent + 1:
                continue
            mf_diag = mf_eff_mass_inv[world, cur]
            if mf_diag <= 0.0 and rt != PGS_CONSTRAINT_TYPE_FRICTION:
                continue
            radius = float(0.0)
            if rt == PGS_CONSTRAINT_TYPE_FRICTION:
                radius = wp.max(
                    mf_row_mu[world, cur] * patch_normal_load(mf_row_parent, mf_impulses, world, parent), 0.0
                )
                # A zero disk with no carried impulse cannot change the velocity.
                if radius == 0.0 and mf_impulses[world, cur] == 0.0 and mf_impulses[world, cur + 1] == 0.0:
                    continue

            dof_a = mf_dof_a[world, cur]
            dof_b = mf_dof_b[world, cur]
            jv = _mf_dot(mf_J_a, mf_J_b, world, cur, dof_a, dof_b, v_scratch)
            residual = jv + mf_rhs[world, cur]
            old_impulse = mf_impulses[world, cur]
            delta = -residual * mf_diag
            if rt == PGS_CONSTRAINT_TYPE_CONTACT:
                w = float(1.0)
                if regularize != 0:
                    w = mf_row_w[world, cur]
                delta = -residual * mf_diag * w - (1.0 - w) * old_impulse
            new_impulse = old_impulse + omega * delta

            if rt == PGS_CONSTRAINT_TYPE_CONTACT:
                new_impulse = wp.max(new_impulse, 0.0)
            elif rt == PGS_CONSTRAINT_TYPE_FRICTION:
                sib = parent + 2
                old_sib = mf_impulses[world, sib]
                pair = wp.vec2(0.0)
                # Zero load gives a zero disk.
                if radius > 0.0:
                    sibling_residual = mf_rhs[world, sib]
                    cross = float(0.0)
                    for k in range(6):
                        if dof_a >= 0:
                            sibling_residual += mf_J_a[world, sib, k] * v_scratch[world, dof_a + k]
                            cross += mf_J_a[world, cur, k] * mf_MiJt_a[world, sib, k]
                        if dof_b >= 0:
                            sibling_residual += mf_J_b[world, sib, k] * v_scratch[world, dof_b + k]
                            cross += mf_J_b[world, cur, k] * mf_MiJt_b[world, sib, k]
                    inv_sib = mf_eff_mass_inv[world, sib]
                    first_diag = float(0.0)
                    if mf_diag > 0.0:
                        first_diag = 1.0 / mf_diag
                    sibling_diag = float(0.0)
                    if inv_sib > 0.0:
                        sibling_diag = 1.0 / inv_sib
                    pair = friction_pair_candidate(
                        first_diag,
                        cross,
                        sibling_diag,
                        wp.vec2(residual, sibling_residual),
                        wp.vec2(old_impulse, old_sib),
                        radius,
                        omega,
                    )
                mag = wp.length(pair)
                scale = float(1.0)
                if mag > radius:
                    scale = radius / mag
                new_impulse = pair[0] * scale
                sib_delta = pair[1] * scale - old_sib
                mf_impulses[world, sib] = pair[1] * scale
                if sib_delta != 0.0:
                    changed = 1
                    _mf_apply(mf_MiJt_a, mf_MiJt_b, world, sib, dof_a, dof_b, sib_delta, v_scratch)

            delta_impulse = new_impulse - old_impulse
            mf_impulses[world, cur] = new_impulse
            if delta_impulse != 0.0:
                changed = 1
                _mf_apply(mf_MiJt_a, mf_MiJt_b, world, cur, dof_a, dof_b, delta_impulse, v_scratch)

        # Velocity limits last: dense joint velocity limits, the fused clamp of driven DOFs,
        # then free-body velocity limits.
        if velocity_limit_pass != 0:
            for i in range(m_dense):
                if world_row_type[world, i] != PGS_CONSTRAINT_TYPE_JOINT_VELOCITY_LIMIT:
                    continue
                denom = world_diag[world, i]
                if denom <= 0.0:
                    continue
                # Stateless projection onto the velocity box: each visit applies only the
                # impulse needed for the current overshoot.
                residual = _dense_dot(J_world, world, i, v_scratch, D) + rhs_bias[world, i]
                delta_impulse = float(0.0)
                if residual < 0.0:
                    delta_impulse = -residual / denom
                world_impulses[world, i] = delta_impulse
                if delta_impulse != 0.0:
                    changed = 1
                    _dense_apply(Y_world, world, i, delta_impulse, v_scratch, D)

            if fuse_vel_limits != 0:
                for i in range(m_dense):
                    if world_row_type[world, i] != PGS_CONSTRAINT_TYPE_JOINT_TARGET:
                        continue
                    denom = world_diag[world, i]
                    if denom <= 0.0:
                        continue
                    qdot_max = drive_vel_limit[world, i]
                    jv = _dense_dot(J_world, world, i, v_scratch, D)
                    if wp.abs(jv) > qdot_max:
                        delta_clamp = (wp.min(wp.max(jv, -qdot_max), qdot_max) - jv) / denom
                        if delta_clamp != 0.0:
                            changed = 1
                            _dense_apply(Y_world, world, i, delta_clamp, v_scratch, D)

            for i in range(mf_contact_end, m_mf):
                if mf_row_type[world, i] != PGS_CONSTRAINT_TYPE_JOINT_VELOCITY_LIMIT:
                    continue
                mf_diag = mf_eff_mass_inv[world, i]
                if mf_diag <= 0.0:
                    continue
                dof_a = mf_dof_a[world, i]
                dof_b = mf_dof_b[world, i]
                residual = _mf_dot(mf_J_a, mf_J_b, world, i, dof_a, dof_b, v_scratch) + mf_rhs[world, i]
                delta_impulse = float(0.0)
                if residual < 0.0:
                    delta_impulse = -residual * mf_diag
                mf_impulses[world, i] = delta_impulse
                if delta_impulse != 0.0:
                    changed = 1
                    _mf_apply(mf_MiJt_a, mf_MiJt_b, world, i, dof_a, dof_b, delta_impulse, v_scratch)

        # An exactly stationary sweep is a fixed point once friction rows are active.
        if global_iter >= friction_start_iteration and changed == 0:
            break

    for d in range(D):
        global_dof = world_dof_indices[world, d]
        if global_dof >= 0:
            v_out[global_dof] = v_scratch[world, d]


@wp.kernel(enable_backward=False)
def pgs_solve_propagation_contact_cpu(
    propagation_constraint_count: wp.array[int],
    propagation_body_a: wp.array2d[int],
    propagation_body_b: wp.array2d[int],
    propagation_MiJt_a: wp.array3d[float],
    propagation_MiJt_b: wp.array3d[float],
    propagation_J_a: wp.array3d[float],
    propagation_J_b: wp.array3d[float],
    propagation_eff_mass_inv: wp.array2d[float],
    propagation_rhs: wp.array2d[float],
    propagation_row_w: wp.array2d[float],
    propagation_row_type: wp.array2d[int],
    propagation_row_parent: wp.array2d[int],
    propagation_row_mu: wp.array2d[float],
    omega: float,
    regularize: int,
    friction_start_iteration: int,
    global_iter: int,
    # in/out
    propagation_impulses: wp.array2d[float],
    propagation_body_qd: wp.array2d[float],
    propagation_body_impulses: wp.array2d[float],
):
    """World-serial twin of one sweep of ``_get_pgs_solve_propagation_contact_kernel`` (contact and friction rows).

    Each row reads the current body twists, updates its impulse, and applies ``MiJt * delta``
    to ``propagation_body_qd`` and ``J * delta`` to ``propagation_body_impulses``; the tree
    propagation then makes the twists exact.
    """
    world = wp.tid()
    m = wp.min(propagation_constraint_count[world], propagation_impulses.shape[1])
    for i in range(m):
        row_type = propagation_row_type[world, i]
        if row_type == PGS_CONSTRAINT_TYPE_FRICTION and global_iter < friction_start_iteration:
            propagation_impulses[world, i] = 0.0
            continue
        eff_inv = propagation_eff_mass_inv[world, i]
        if eff_inv <= 0.0 and row_type != PGS_CONSTRAINT_TYPE_FRICTION:
            continue
        ba = propagation_body_a[world, i]
        bb = propagation_body_b[world, i]
        jv = float(0.0)
        for k in range(6):
            if ba >= 0:
                jv += propagation_J_a[world, i, k] * propagation_body_qd[ba, k]
            if bb >= 0:
                jv += propagation_J_b[world, i, k] * propagation_body_qd[bb, k]
        residual = jv + propagation_rhs[world, i]
        old_impulse = propagation_impulses[world, i]
        delta = -residual * eff_inv
        if row_type == PGS_CONSTRAINT_TYPE_CONTACT and regularize != 0:
            # Regularized rows (weight w < 1) also relax the impulse toward zero.
            w = propagation_row_w[world, i]
            delta = delta * w - (1.0 - w) * old_impulse
        new_impulse = old_impulse + omega * delta

        if row_type == PGS_CONSTRAINT_TYPE_CONTACT:
            new_impulse = wp.max(new_impulse, 0.0)
        elif row_type == PGS_CONSTRAINT_TYPE_FRICTION:
            parent = propagation_row_parent[world, i]
            if i != parent + 1:
                # The first tangent row solves both tangents of its contact.
                new_impulse = old_impulse
            else:
                radius = wp.max(
                    propagation_row_mu[world, i]
                    * patch_normal_load(propagation_row_parent, propagation_impulses, world, parent),
                    0.0,
                )
                sib = parent + 2
                other = propagation_impulses[world, sib]
                sibling_residual = propagation_rhs[world, sib]
                cross = float(0.0)
                for k in range(6):
                    if ba >= 0:
                        sibling_residual += propagation_J_a[world, sib, k] * propagation_body_qd[ba, k]
                        cross += propagation_J_a[world, i, k] * propagation_MiJt_a[world, sib, k]
                    if bb >= 0:
                        sibling_residual += propagation_J_b[world, sib, k] * propagation_body_qd[bb, k]
                        cross += propagation_J_b[world, i, k] * propagation_MiJt_b[world, sib, k]
                inv_sib = propagation_eff_mass_inv[world, sib]
                first_diag = float(0.0)
                if eff_inv > 0.0:
                    first_diag = 1.0 / eff_inv
                sibling_diag = float(0.0)
                if inv_sib > 0.0:
                    sibling_diag = 1.0 / inv_sib
                pair = friction_pair_candidate(
                    first_diag,
                    cross,
                    sibling_diag,
                    wp.vec2(residual, sibling_residual),
                    wp.vec2(old_impulse, other),
                    radius,
                    omega,
                )
                mag = wp.length(pair)
                scale = float(1.0)
                if mag > radius:
                    scale = radius / mag
                new_impulse = pair[0] * scale
                sib_new = pair[1] * scale
                sib_delta = sib_new - other
                propagation_impulses[world, sib] = sib_new
                if sib_delta != 0.0:
                    sib_ba = propagation_body_a[world, sib]
                    sib_bb = propagation_body_b[world, sib]
                    for k in range(6):
                        if sib_ba >= 0:
                            propagation_body_qd[sib_ba, k] += propagation_MiJt_a[world, sib, k] * sib_delta
                            propagation_body_impulses[sib_ba, k] += propagation_J_a[world, sib, k] * sib_delta
                        if sib_bb >= 0:
                            propagation_body_qd[sib_bb, k] += propagation_MiJt_b[world, sib, k] * sib_delta
                            propagation_body_impulses[sib_bb, k] += propagation_J_b[world, sib, k] * sib_delta

        delta_impulse = new_impulse - old_impulse
        propagation_impulses[world, i] = new_impulse
        if delta_impulse != 0.0:
            for k in range(6):
                if ba >= 0:
                    propagation_body_qd[ba, k] += propagation_MiJt_a[world, i, k] * delta_impulse
                    propagation_body_impulses[ba, k] += propagation_J_a[world, i, k] * delta_impulse
                if bb >= 0:
                    propagation_body_qd[bb, k] += propagation_MiJt_b[world, i, k] * delta_impulse
                    propagation_body_impulses[bb, k] += propagation_J_b[world, i, k] * delta_impulse
