# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Experimental implicit unilateral material contacts for FeatherPGS.

For signed separation phi, separating velocity u and impulse lambda:
F = max(0, -k*phi_next - c*u_next), gamma = 1/(h*(h*k+c)),
bias = k*phi/(h*k+c), residual = u_next + bias + gamma*lambda.
Native hydro stiffness is already area/pressure weighted [N/m], not shape
bulk stiffness [N/m^3]. Friction weighting is applied once to the existing
pair coefficient, with the cone bounded by the compliant normal impulse.

The current implementation synchronizes row metadata to the host once per
physical step and launches one original PGS iteration at a time. It is a
deliberately bounded experimental reference, not a graph-compatible fast path.
"""

import math

import numpy as np
import warp as wp


def material_coefficients(stiffness, damping, friction_scale, *, shape_friction=0.0):
    """Resolve one hydro contact's material fields with explicit SI semantics.

    Positive exported stiffness is required [N/m]. Damping is the exported
    contact coefficient [N s/m]; zero stays zero, without copying a shape
    damping coefficient to every quadrature sample. Zero/unset friction scale means 1, NOT
    frictionless. The returned friction coefficient is pair-mixed shape mu
    times that scale, applied once (stiffness already includes quadrature).
    No critical-damping or real-rubber calibration is silently invented.
    """
    values = (stiffness, damping, friction_scale, shape_friction)
    if any(not math.isfinite(x) or x < 0 for x in values) or stiffness == 0:
        raise ValueError("Require finite positive hydro stiffness and non-negative material coefficients")
    return stiffness, damping, shape_friction * (friction_scale or 1.0)


def normal_coefficients(stiffness, damping, separation, *, dt):
    """Return impulse-space compliance [1/kg] and velocity bias [m/s]."""
    if not math.isfinite(dt) or dt <= 0:
        raise ValueError("dt must be finite and positive")
    k, c, _ = material_coefficients(stiffness, damping, 0.0)
    if not math.isfinite(separation):
        raise ValueError("separation must be finite")
    # A dashpot must not act across an open gap. At a speculative contact,
    # use the spring-only implicit law until penetration has begun.
    denominator = dt * k + (c if separation <= 0.0 else 0.0)
    return 1.0 / (dt * denominator), k * separation / denominator


@wp.kernel
def update_compliant_rhs(
    base: wp.array2d[float], gamma: wp.array2d[float], impulses: wp.array2d[float], rhs: wp.array2d[float]
):
    """Update a frozen row's physical residual without synchronizing to the host."""
    world, row = wp.tid()
    rhs[world, row] = base[world, row] + gamma[world, row] * impulses[world, row]


def validate_configuration(model, settings):
    """Reject combinations without a defined and tested compliant row update."""
    required = {
        "pgs_mode": "matrix_free",
        "articulated_contact_response": "immediate",
        "pgs_schedule": "interleaved",
        "pgs_velocity_iterations": 0,
        "pgs_warmstart": False,
        "mf_warmstart": False,
        "enable_restitution": False,
        "pgs_contact_regularization": 0.0,
        "pgs_debug": False,
        "contact_friction_position_iterations": -1,
        "friction_mode": "current",
        "contact_friction_anchor_limit": 0,
        "contact_friction_shared_anchor": False,
        "contact_shared_anchor": False,
    }
    for key, expected in required.items():
        if settings[key] != expected:
            raise ValueError(f"contact_compliance requires {key}={expected!r}; got {settings[key]!r}")
    if settings["pgs_iterations"] <= 0:
        raise ValueError("contact_compliance requires positive pgs_iterations")
    if not model.device.is_cuda:
        raise ValueError("contact_compliance currently requires CUDA")


def validate_step(solver):
    """Reject unqualified combinations before any contact preprocessing."""
    if wp.get_stream(solver.model.device).is_capturing:
        raise RuntimeError("contact_compliance does not support CUDA graph capture")
    if getattr(solver, "contact_torsion_radius", 0.0) > 0:
        raise ValueError("contact_compliance is not validated with contact_torsion_radius > 0")
    # The persistent-patch implementation allocates a dummy buffer even when OFF.
    if getattr(solver, "friction_anchor_beta", 0.0) > 0 or getattr(solver, "_friction_anchors_enabled", False):
        raise ValueError("contact_compliance is not validated with friction_anchor_beta > 0")


def start_step(solver, contacts, dt):
    """Reset step-local material bindings before row construction."""
    validate_step(solver)
    if not math.isfinite(dt) or dt <= 0:
        raise ValueError("contact_compliance requires a finite positive dt")
    if contacts is None or any(
        getattr(contacts, key, None) is None
        for key in ("rigid_contact_stiffness", "rigid_contact_damping", "rigid_contact_friction")
    ):
        raise ValueError("contact_compliance requires native per-contact material arrays")
    if getattr(contacts, "rigid_contacts_body_pair_reduced", False):
        raise ValueError("contact_compliance does not support body-pair contact reduction")
    count = int(contacts.rigid_contact_count.numpy()[0])
    if count < 0 or count > contacts.rigid_contact_max:
        raise RuntimeError("contact_compliance rejects overflowing contact input")
    solver._compliant_contacts = contacts
    solver._compliant_dt = dt
    solver._compliant_prepared = False
    solver.compliance_contact_count = 0
    solver.compliance_skipped_contact_count = 0


def _prepare_compliant_rows(self):
    """Replace normal biases/diagonals and weight their existing friction children."""
    contacts, dt = self._compliant_contacts, self._compliant_dt
    count = int(contacts.rigid_contact_count.numpy()[0])
    if count > contacts.rigid_contact_max:
        raise RuntimeError("contact_compliance rejects overflowing contact input")
    dense_counts, mf_counts = self.constraint_count.numpy(), self.mf_constraint_count.numpy()
    # Failed contact reservations roll back the live row count. Check loss counters
    # even when warning output and watermark diagnostics are disabled.
    dropped = sum(
        int(rows.numpy().sum())
        for rows in (self._row_dropped_dense, self._row_dropped_mf, self._row_dropped_propagation)
    )
    if dropped or np.any(dense_counts > self.dense_max_constraints) or np.any(mf_counts > self.mf_max_constraints):
        raise RuntimeError("contact_compliance rejects overflowing solver rows")
    stiffness = contacts.rigid_contact_stiffness.numpy()[:count]
    damping = contacts.rigid_contact_damping.numpy()[:count]
    scale = contacts.rigid_contact_friction.numpy()[:count]
    paths, slots, worlds = (x.numpy()[:count] for x in (self.contact_path, self.contact_slot, self.contact_world))
    slots_needed = self.contact_slots_needed.numpy()[:count]
    dense_diag, mf_inv = self.diag.numpy(), self.mf_eff_mass_inv.numpy()
    self._compliant_dense_base, self._compliant_mf_base = self.rhs.numpy(), self.mf_rhs.numpy()
    self._compliant_dense_gamma, self._compliant_mf_gamma = np.zeros_like(dense_diag), np.zeros_like(mf_inv)
    dense_phi, mf_phi = self.phi.numpy(), self.mf_phi.numpy()
    dense_mu, mf_mu = self.row_mu.numpy(), self.mf_row_mu.numpy()
    dense_parent, mf_parent = self.row_parent.numpy(), self.mf_row_parent.numpy()
    dense_type, mf_type = self.row_type.numpy(), self.mf_row_type.numpy()
    dense_cfm = self.row_cfm.numpy()
    dense_target = self.target_velocity.numpy()
    # Without prescribed bodies Newton allocates only a (1, 1) placeholder.
    mf_target = self.mf_target_velocity.numpy() if self._has_prescribed_response else None
    active = 0
    skipped = 0
    seen_rows = set()
    for contact, k in enumerate(stiffness):
        if not np.isfinite(k) or k < 0:
            raise ValueError("Invalid exported contact stiffness")
        if k == 0:
            continue
        path, slot, world = int(paths[contact]), int(slots[contact]), int(worlds[contact])
        # No capacity request means the allocator intentionally excluded this pair
        # (nonresponding bodies, world filtering, or a positive-gap gate).
        if path == -1 and slot == -1 and slots_needed[contact] == 0:
            skipped += 1
            continue
        shape = dense_diag.shape if path == 0 else mf_inv.shape
        if path not in (0, 1) or not 0 <= world < shape[0] or not 0 <= slot < shape[1]:
            raise RuntimeError(f"Compliant contact was dropped or routed to unsupported path {path}, slot {slot}")
        if slot >= (dense_counts if path == 0 else mf_counts)[world]:
            raise RuntimeError("Compliant contact was dropped or mapped beyond active solver rows")
        row = (path, world, slot)
        if row in seen_rows:
            raise RuntimeError("Compliant contacts must map one-to-one to normal rows")
        seen_rows.add(row)
        _, c, friction_weight = material_coefficients(
            float(k),
            float(damping[contact]),
            float(scale[contact]),
            shape_friction=1.0,
        )
        phi = (dense_phi if path == 0 else mf_phi)[world, slot]
        gamma, bias = normal_coefficients(float(k), c, float(phi), dt=dt)
        if max(abs(gamma), abs(bias)) > np.finfo(np.float32).max:
            raise ValueError("Contact compliance coefficients exceed float32 range")
        if path == 0:
            if dense_type[world, slot] != 0:
                raise RuntimeError("Contact map no longer points to a dense normal row")
            self._compliant_dense_gamma[world, slot] = gamma
            self._compliant_dense_base[world, slot] = bias - dense_target[world, slot]
            dense_diag[world, slot] += gamma - dense_cfm[world, slot]
            children = (dense_parent[world] == slot) & (dense_type[world] == 2)
            dense_mu[world, children] *= friction_weight
        else:
            if mf_type[world, slot] != 0 or mf_inv[world, slot] <= 0:
                raise RuntimeError("Contact map no longer points to an effective MF normal row")
            self._compliant_mf_gamma[world, slot] = gamma
            self._compliant_mf_base[world, slot] = bias - (mf_target[world, slot] if mf_target is not None else 0.0)
            mf_inv[world, slot] = 1.0 / (1.0 / mf_inv[world, slot] - self.pgs_cfm + gamma)
            children = (mf_parent[world] == slot) & (mf_type[world] == 2)
            mf_mu[world, children] *= friction_weight
        active += 1
    self.diag.assign(dense_diag)
    self.mf_eff_mass_inv.assign(mf_inv)
    self.row_mu.assign(dense_mu)
    self.mf_row_mu.assign(mf_mu)
    self._compliant_dense_base_gpu = wp.array(self._compliant_dense_base, device=self.model.device)
    self._compliant_mf_base_gpu = wp.array(self._compliant_mf_base, device=self.model.device)
    self._compliant_dense_gamma_gpu = wp.array(self._compliant_dense_gamma, device=self.model.device)
    self._compliant_mf_gamma_gpu = wp.array(self._compliant_mf_gamma, device=self.model.device)
    self.compliance_contact_count = active
    self.compliance_skipped_contact_count = skipped
    self._compliant_prepared = True


def solve(solver, *, iterations, friction_start_iteration, iteration_offset):
    """Solve compliance inside one physical step, never integrate per iteration."""
    if not solver._compliant_prepared:
        _prepare_compliant_rows(solver)
    for iteration in range(iterations):
        wp.launch(
            update_compliant_rhs,
            dim=solver.rhs.shape,
            inputs=[solver._compliant_dense_base_gpu, solver._compliant_dense_gamma_gpu, solver.impulses, solver.rhs],
            device=solver.model.device,
        )
        wp.launch(
            update_compliant_rhs,
            dim=solver.mf_rhs.shape,
            inputs=[solver._compliant_mf_base_gpu, solver._compliant_mf_gamma_gpu, solver.mf_impulses, solver.mf_rhs],
            device=solver.model.device,
        )
        solver._pack_mf_meta(solver.mf_rhs)
        solver._launch_matrix_free_gs_solve(
            dense_rhs=solver.rhs,
            mf_meta=solver.mf_meta_packed,
            iterations=1,
            omega=solver.pgs_omega,
            friction_start_iteration=friction_start_iteration,
            iteration_offset=iteration_offset + iteration,
        )
