# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Experimental implicit unilateral material contacts for FeatherPGS.

For signed separation ``phi``, separating velocity ``u`` and impulse ``lambda``:
``F = max(0, -k*phi_next - c*u_next)``, ``gamma = 1/(h*(h*k+c))``,
``bias = k*phi/(h*k+c)``, ``residual = u_next + bias + gamma*lambda``.
Hydroelastic contact stiffness is already area/pressure weighted [N/m], not shape
bulk stiffness [N/m^3]. Friction weighting is applied once to the existing pair
coefficient, with the cone bounded by the compliant normal impulse.

The implementation synchronizes row metadata to the host once per step and launches
one Gauss-Seidel sweep at a time. It is a bounded experimental reference, not a
graph-compatible fast path.
"""

import math

import numpy as np
import warp as wp

from .kernels import PGS_CONSTRAINT_TYPE_CONTACT, PGS_CONSTRAINT_TYPE_FRICTION


def material_coefficients(stiffness, damping, friction_scale, *, shape_friction=0.0):
    """Resolve one hydroelastic contact's material fields with explicit SI semantics.

    Positive stiffness is required [N/m]. Damping is the contact coefficient [N s/m];
    zero stays zero. A zero friction scale means 1, not frictionless. The returned
    friction coefficient is the pair friction times that scale, applied once (the
    stiffness already includes the quadrature weight).
    """
    values = (stiffness, damping, friction_scale, shape_friction)
    if any(not math.isfinite(x) or x < 0 for x in values) or stiffness == 0:
        raise ValueError("Require finite positive hydroelastic stiffness and non-negative material coefficients")
    return stiffness, damping, shape_friction * (friction_scale or 1.0)


def normal_coefficients(stiffness, damping, separation, *, dt):
    """Return the impulse-space compliance [1/kg] and the velocity bias [m/s] of one row."""
    if not math.isfinite(dt) or dt <= 0:
        raise ValueError("dt must be finite and positive")
    k, c, _ = material_coefficients(stiffness, damping, 0.0)
    if not math.isfinite(separation):
        raise ValueError("separation must be finite")
    # A dashpot must not act across an open gap: a speculative contact uses the
    # spring-only implicit law until penetration has begun.
    denominator = dt * k + (c if separation <= 0.0 else 0.0)
    return 1.0 / (dt * denominator), k * separation / denominator


@wp.kernel
def update_compliant_rhs(
    base: wp.array2d[float], gamma: wp.array2d[float], impulses: wp.array2d[float], rhs: wp.array2d[float]
):
    """Update a row's physical residual from its current impulse without a host synchronization."""
    world, row = wp.tid()
    rhs[world, row] = base[world, row] + gamma[world, row] * impulses[world, row]


def validate_configuration(settings):
    """Reject constructor combinations without a defined and tested compliant row update."""
    required = {
        "pgs_velocity_iterations": 0,
        "pgs_warmstart": False,
        "pgs_contact_regularization": 0.0,
        "contact_shared_anchor": False,
        "contact_friction_shared_anchor": False,
        "friction_mode": "current",
        "contact_friction_position_iterations": -1,
        "pgs_debug": False,
    }
    for key, expected in required.items():
        if settings[key] != expected:
            raise ValueError(f"contact_compliance requires {key}={expected!r}; got {settings[key]!r}")
    if settings["pgs_iterations"] <= 0:
        raise ValueError("contact_compliance requires positive pgs_iterations")


def validate_step(solver):
    """Reject unsupported combinations before any contact preprocessing."""
    if wp.get_stream(solver.model.device).is_capturing:
        raise RuntimeError("contact_compliance does not support CUDA graph capture")
    if getattr(solver, "friction_anchor_beta", 0.0) > 0 or getattr(solver, "_friction_anchors_enabled", False):
        raise ValueError("contact_compliance is not validated with friction_anchor_beta > 0")


def start_step(solver, contacts, dt):
    """Validate the step's inputs and reset step-local material bindings before row construction."""
    validate_step(solver)
    if not math.isfinite(dt) or dt <= 0:
        raise ValueError("contact_compliance requires a finite positive dt")
    if contacts is None or any(
        getattr(contacts, key, None) is None
        for key in ("rigid_contact_stiffness", "rigid_contact_damping", "rigid_contact_friction")
    ):
        raise ValueError("contact_compliance requires per-contact hydroelastic material arrays")
    count = int(contacts.rigid_contact_count.numpy()[0])
    if count < 0 or count > contacts.rigid_contact_max:
        raise RuntimeError("contact_compliance rejects overflowing contact input")
    restitution = getattr(solver, "shape_material_restitution", None)
    if getattr(solver, "enable_restitution", True) and restitution is not None and np.any(restitution.numpy() > 0.0):
        # The compliant law replaces the position bias; a rebound target has no
        # defined composition with it.
        raise ValueError("contact_compliance requires zero shape restitution")
    solver._compliant_contacts = contacts
    solver._compliant_dt = dt
    solver._compliant_prepared = False
    solver.compliance_contact_count = 0
    solver.compliance_skipped_contact_count = 0


def clear(solver):
    """Discard the step-local compliance bindings."""
    solver._compliant_contacts = None
    solver._compliant_prepared = False
    solver.compliance_contact_count = 0
    solver.compliance_skipped_contact_count = 0


def _prepare_compliant_rows(self):
    """Replace the normal biases and diagonals of compliant rows and weight their friction children."""
    contacts, dt = self._compliant_contacts, self._compliant_dt
    count = int(contacts.rigid_contact_count.numpy()[0])
    if count > contacts.rigid_contact_max:
        raise RuntimeError("contact_compliance rejects overflowing contact input")
    dense_counts, mf_counts = self.constraint_count.numpy(), self.mf_constraint_count.numpy()
    # Failed contact reservations roll back the live row count; check the loss
    # counters even when overflow warnings are disabled.
    dropped = int(self._row_dropped_dense.numpy().sum()) + int(self._row_dropped_mf.numpy().sum())
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
    dense_target = self.target_velocity.numpy()
    # Without prescribed bodies the free-body target is a (1, 1) placeholder.
    mf_target = self.mf_target_velocity.numpy() if self._has_prescribed_response else None
    active = 0
    skipped = 0
    seen_rows = set()
    for contact, k in enumerate(stiffness):
        if not np.isfinite(k) or k < 0:
            raise ValueError("Invalid contact stiffness")
        if k == 0:
            continue
        path, slot, world = int(paths[contact]), int(slots[contact]), int(worlds[contact])
        # No capacity request means the allocator intentionally excluded this pair
        # (non-responding bodies, world filtering or a positive-gap gate).
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
            float(k), float(damping[contact]), float(scale[contact]), shape_friction=1.0
        )
        phi = (dense_phi if path == 0 else mf_phi)[world, slot]
        gamma, bias = normal_coefficients(float(k), c, float(phi), dt=dt)
        if max(abs(gamma), abs(bias)) > np.finfo(np.float32).max:
            raise ValueError("Contact compliance coefficients exceed float32 range")
        if path == 0:
            if dense_type[world, slot] != PGS_CONSTRAINT_TYPE_CONTACT:
                raise RuntimeError("Contact map no longer points to a dense normal row")
            self._compliant_dense_gamma[world, slot] = gamma
            self._compliant_dense_base[world, slot] = bias - dense_target[world, slot]
            # The compliance replaces the row's constraint force mixing.
            dense_diag[world, slot] += gamma - self.pgs_cfm
            children = (dense_parent[world] == slot) & (dense_type[world] == PGS_CONSTRAINT_TYPE_FRICTION)
            dense_mu[world, children] *= friction_weight
        else:
            if mf_type[world, slot] != PGS_CONSTRAINT_TYPE_CONTACT or mf_inv[world, slot] <= 0:
                raise RuntimeError("Contact map no longer points to an effective free-body normal row")
            self._compliant_mf_gamma[world, slot] = gamma
            self._compliant_mf_base[world, slot] = bias - (mf_target[world, slot] if mf_target is not None else 0.0)
            mf_inv[world, slot] = 1.0 / (1.0 / mf_inv[world, slot] - self.pgs_cfm + gamma)
            children = (mf_parent[world] == slot) & (mf_type[world] == PGS_CONSTRAINT_TYPE_FRICTION)
            mf_mu[world, children] *= friction_weight
        active += 1
    device = self.model.device
    self.diag.assign(dense_diag)
    self.mf_eff_mass_inv.assign(mf_inv)
    self.row_mu.assign(dense_mu)
    self.mf_row_mu.assign(mf_mu)
    self._compliant_dense_base_gpu = wp.array(self._compliant_dense_base, device=device)
    self._compliant_mf_base_gpu = wp.array(self._compliant_mf_base, device=device)
    self._compliant_dense_gamma_gpu = wp.array(self._compliant_dense_gamma, device=device)
    self._compliant_mf_gamma_gpu = wp.array(self._compliant_mf_gamma, device=device)
    self.compliance_contact_count = active
    self.compliance_skipped_contact_count = skipped
    self._compliant_prepared = True


def solve(solver, *, iterations):
    """Solve the compliant rows within one step, updating their residuals after every sweep."""
    if not solver._compliant_prepared:
        _prepare_compliant_rows(solver)
    device = solver.model.device
    for _ in range(iterations):
        wp.launch(
            update_compliant_rhs,
            dim=solver.rhs.shape,
            inputs=[solver._compliant_dense_base_gpu, solver._compliant_dense_gamma_gpu, solver.impulses, solver.rhs],
            device=device,
        )
        wp.launch(
            update_compliant_rhs,
            dim=solver.mf_rhs.shape,
            inputs=[solver._compliant_mf_base_gpu, solver._compliant_mf_gamma_gpu, solver.mf_impulses, solver.mf_rhs],
            device=device,
        )
        solver._pack_mf_meta(solver.mf_rhs)
        solver._launch_pgs_solve(solver.rhs, 1, regularize=False)
