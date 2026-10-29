# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Experimental rolling and spin friction rows for free-body contacts.

A shape opts in through the ``steel:condim`` custom attribute, borrowing MuJoCo's 3/4/6
convention: at 4 or more it contributes :attr:`~newton.Model.shape_material_mu_torsional` [m]
to a spin row about the contact normal, at 6 also :attr:`~newton.Model.shape_material_mu_rolling`
[m] to a rolling pair about the contact tangents. A contact takes the larger coefficient of its
two shapes. The rows follow the contact's sliding pair, ``[normal, t1, t2, roll1, roll2, spin]``,
and are bounded by their coefficient times the pooled normal load of the contact's patch: the
rolling pair on a disk, the spin row on a box. Only contacts on the free-body route (free bodies
against free bodies or static and kinematic geometry) receive them.
"""

import numpy as np
import warp as wp

from .kernels import (
    PGS_CONSTRAINT_TYPE_CONTACT_ROLLING,
    PGS_CONSTRAINT_TYPE_CONTACT_SPIN,
    build_mf_angular_friction_rows,
    gate_free_rigid_angular_coefficients,
)


def configure_free_rigid_angular_friction(solver, model) -> None:
    """Decide from the shapes' ``steel:condim`` whether free-body contacts get angular rows."""
    condim = solver._namespaced_attribute("condim")
    values = condim.numpy() if condim is not None and model.shape_count else None
    if values is not None and np.any((values < 1) | (values > 6)):
        raise ValueError("steel:condim must be between 1 and 6")
    solver._free_rigid_angular_rows = bool(values is not None and np.any(values >= 4))


def refresh_free_rigid_angular_friction(solver) -> None:
    """Write each shape's gated torsional and rolling coefficients [m] on the device.

    Capturable; outside capture it also rejects a condim above 3 on a solver constructed without angular rows.
    """
    model = solver.model
    condim = solver._namespaced_attribute("condim")
    if not solver._free_rigid_angular_rows:
        if condim is not None and model.shape_count and not model.device.is_capturing:
            if np.any(condim.numpy() >= 4):
                raise ValueError("steel:condim above 3 needs a solver constructed with it; construct a new solver")
        return
    wp.launch(
        gate_free_rigid_angular_coefficients,
        dim=model.shape_count,
        inputs=[condim, model.shape_material_mu_torsional, model.shape_material_mu_rolling],
        outputs=[solver._free_rigid_mu_torsional, solver._free_rigid_mu_rolling],
        device=model.device,
    )


def free_rigid_route_bodies(solver) -> wp.array | None:
    """Mark the bodies whose contacts can take the free-body route; ``None`` without angular rows.

    A contact takes that route when each side is a free rigid body or has no response DOFs, as the allocator decides.
    """
    if not solver._free_rigid_angular_rows:
        return None
    model = solver.model
    route = np.zeros(max(model.body_count, 1), dtype=np.int32)
    if model.body_count and solver._has_free_rigid_bodies:
        body_art = solver._body_to_articulation.numpy()[: model.body_count]
        response = solver._articulation_response_dof_count.numpy()
        free = solver._is_free_rigid.numpy()
        safe = np.maximum(body_art, 0)
        route[: model.body_count] = (body_art < 0) | (response[safe] == 0) | (free[safe] != 0)
    return wp.array(route, dtype=wp.int32, device=model.device)


def validate_free_rigid_angular_friction_mode(solver) -> None:
    """Reject solver options whose free-body solves do not visit the angular rows."""
    if not solver._free_rigid_angular_rows:
        return
    model = solver.model
    if (
        not model.device.is_cuda
        or model.requires_grad
        or solver._solve_mode != "matrix_free"
        or solver._articulated_contact_response != "immediate"
        or solver._row_schedule != "interleaved"
        or solver._friction_mode != "fixed_normal"
        or solver._contact_compliance
        or solver._sparse_mass_matrix_size is not None
        or (solver._has_free_rigid_bodies and solver._pgs_solve_mf_gs_kernel is None)
    ):
        raise ValueError(
            "steel:condim above 3 requires CUDA, no gradients, solve_mode='matrix_free', "
            "articulated_contact_response='immediate', row_schedule='interleaved', friction_mode='fixed_normal', "
            "the fused matrix-free GS solve and no contact compliance"
        )


def launch_free_rigid_angular_friction_rows(solver, state_aug, contacts, threads: int) -> None:
    """Write the rolling and spin rows that follow each free-body contact's sliding pair."""
    wp.launch(
        build_mf_angular_friction_rows,
        dim=threads,
        inputs=[
            contacts.rigid_contact_count,
            threads,
            contacts.rigid_contact_normal,
            contacts.rigid_contact_shape0,
            contacts.rigid_contact_shape1,
            solver._contact_world,
            solver._contact_slot,
            solver._contact_path,
            solver._contact_slots_needed,
            solver._contact_art_a,
            solver._contact_art_b,
            solver._articulation_response_dof_count,
            solver.model.shape_body,
            state_aug._body_v_s,
            solver._prescribed_articulation,
            int(solver._has_prescribed_response),
            solver._free_rigid_mu_torsional,
            solver._free_rigid_mu_rolling,
        ],
        outputs=[
            solver._mf_body_a,
            solver._mf_body_b,
            solver._mf_J_a,
            solver._mf_J_b,
            solver._mf_row_type,
            solver._mf_row_parent,
            solver._mf_row_mu,
            solver._mf_phi,
            solver._mf_target_velocity,
            solver._mf_row_restitution,
        ],
        device=solver.model.device,
    )


def free_rigid_angular_friction_gs_sources() -> tuple[str, str, str]:
    """Return the fused GS sweep's angular-row skip, bound and projection sources.

    The rolling pair is solved as one disk block at its first row, as the sliding pair is; the spin row is a box.
    Both take their bound from the coefficient times the pooled normal load at each visit.
    """
    rolling = int(PGS_CONSTRAINT_TYPE_CONTACT_ROLLING)
    spin = int(PGS_CONSTRAINT_TYPE_CONTACT_SPIN)
    angular = f"(mf_rt == {rolling} || mf_rt == {spin})"
    skip = f"""
            if ({angular} && global_iter < friction_start_iteration) {{
                s_lam_mf[i] = 0.0f;
                __syncwarp();
                continue;
            }}
            // The rolling pair is solved at its first row, three after the normal.
            if (mf_rt == {rolling} && i != mf_par + 3) continue;"""
    radius = f"""
            float angular_radius = 0.0f;
            if ({angular}) {{
                float lambda_n = s_lam_mf[mf_par];
                for (int patch_row = (mf_meta.data[off_meta + mf_par * 4 + 3] >> 16);
                     patch_row >= 0 && patch_row != mf_par;
                     patch_row = (mf_meta.data[off_meta + patch_row * 4 + 3] >> 16))
                    lambda_n += s_lam_mf[patch_row];
                angular_radius = fmaxf(mf_row_mu.data[off_mf + i] * lambda_n, 0.0f);
                // A zero bound with no carried impulse cannot change the velocity.
                if (angular_radius == 0.0f && s_lam_mf[i] == 0.0f && (mf_rt != {rolling} || s_lam_mf[i + 1] == 0.0f))
                    continue;
            }}"""
    update = f""" else if (mf_rt == {spin}) {{
                new_impulse = fminf(fmaxf(new_impulse, -angular_radius), angular_radius);
            }} else if (mf_rt == {rolling}) {{
                int sib = i + 1;
                int sib_mf6 = mf6_base + sib * 6;
                float2 pair = make_float2(0.0f, 0.0f);
                if (angular_radius > 0.0f) {{
                    float sibling_residual = 0.0f;
                    float cross = 0.0f;
                    if (lane < 6 && dof_a >= 0) {{
                        sibling_residual = mf_J_a.data[sib_mf6 + lane] * s_v[dof_a + lane];
                        cross = cur_Ja * mf_MiJt_a.data[sib_mf6 + lane];
                    }}
                    if (lane >= 6 && lane < 12 && dof_b >= 0) {{
                        sibling_residual = mf_J_b.data[sib_mf6 + lane - 6] * s_v[dof_b + lane - 6];
                        cross = cur_Jb * mf_MiJt_b.data[sib_mf6 + lane - 6];
                    }}
                    sibling_residual = warp_sum(sibling_residual) + __int_as_float(mf_meta.data[off_meta + sib * 4 + 2]);
                    cross = warp_sum(cross);
                    float inv_sib = __int_as_float(mf_meta.data[off_meta + sib * 4 + 1]);
                    pair = friction_pair_candidate(mf_diag > 0.0f ? 1.0f / mf_diag : 0.0f, cross,
                        inv_sib > 0.0f ? 1.0f / inv_sib : 0.0f, residual, sibling_residual, old_impulse,
                        s_lam_mf[sib], angular_radius, omega);
                }}
                float mag = sqrtf(pair.x * pair.x + pair.y * pair.y);
                float scale = mag > angular_radius ? angular_radius / mag : 1.0f;
                new_impulse = pair.x * scale;
                float sib_delta = pair.y * scale - s_lam_mf[sib];
                __syncwarp();
                s_lam_mf[sib] = pair.y * scale;
                if (sib_delta != 0.0f) {{
                    iteration_changed = 1;
                    if (lane < 6 && dof_a >= 0) s_v[dof_a + lane] += mf_MiJt_a.data[sib_mf6 + lane] * sib_delta;
                    if (lane >= 6 && lane < 12 && dof_b >= 0)
                        s_v[dof_b + lane - 6] += mf_MiJt_b.data[sib_mf6 + lane - 6] * sib_delta;
                }}
            }}"""
    return skip, radius, update
