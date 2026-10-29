# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Experimental implicit unilateral material contacts for Steel.

For signed separation ``phi``, separating velocity ``u`` and impulse ``lambda``:
``F = max(0, -k*phi_next - c*u_next)``, ``gamma = 1/(h*(h*k+c))``,
``bias = k*phi/(h*k+c)``, ``residual = u_next + bias + gamma*lambda``.
Hydroelastic contact stiffness is already area/pressure weighted [N/m], not shape
bulk stiffness [N/m^3]. Friction weighting is applied once to the existing pair
coefficient, with the cone bounded by the compliant normal impulse.

Row coefficients are prepared on the device in persistent buffers, and the solve
kernels add ``gamma * lambda`` to compliant normal residuals in every sweep, so a
step needs no host synchronization and can be graph captured. Errors latch in device
status words: an eager step raises from its own word, and captured steps latch into a
word that :meth:`SolverSteel.check_constraint_capacity` reports until :meth:`SolverSteel.reset`.
"""

import math

import warp as wp

from .kernels import PGS_CONSTRAINT_TYPE_CONTACT, PGS_CONSTRAINT_TYPE_FRICTION, WARP_DETERMINISM_OFF

# SolverSteel rejects models with ``requires_grad``, so no kernel needs an adjoint.
# Unfused arithmetic keeps the float64 coefficient law equal to the host law.
wp.set_module_options({"enable_backward": False, "fuse_fp": False, **WARP_DETERMINISM_OFF})

# Status codes, in the order the per-contact checks run.
_INPUT_OVERFLOW = 0
_ROW_OVERFLOW = 1
_RESTITUTION = 2
_INVALID_STIFFNESS = 3
_UNMAPPED = 4
_BEYOND_ROWS = 5
_DUPLICATE = 6
_INVALID_MATERIAL = 7
_FLOAT32_RANGE = 8
_DENSE_TYPE = 9
_MF_TYPE = 10
_INVALID_SEPARATION = 11
_STATUS_CODES = 12
# status[_STATUS_CODES] holds the smallest (contact + 1) * 16 + code; global errors use contact = -1.
_NO_ERROR = 2147483647

_ERRORS = {
    _INPUT_OVERFLOW: (RuntimeError, "contact_compliance rejects overflowing contact input"),
    _ROW_OVERFLOW: (RuntimeError, "contact_compliance rejects overflowing solver rows"),
    _RESTITUTION: (ValueError, "contact_compliance requires zero shape restitution"),
    _INVALID_STIFFNESS: (ValueError, "Invalid contact stiffness"),
    _UNMAPPED: (RuntimeError, "Compliant contact was dropped or routed to an unsupported path"),
    _BEYOND_ROWS: (RuntimeError, "Compliant contact was dropped or mapped beyond active solver rows"),
    _DUPLICATE: (RuntimeError, "Compliant contacts must map one-to-one to normal rows"),
    _INVALID_MATERIAL: (
        ValueError,
        "Require finite positive hydroelastic stiffness and non-negative material coefficients",
    ),
    _FLOAT32_RANGE: (ValueError, "Contact compliance coefficients exceed float32 range"),
    _DENSE_TYPE: (RuntimeError, "Contact map no longer points to a dense normal row"),
    _MF_TYPE: (RuntimeError, "Contact map no longer points to an effective free-body normal row"),
    _INVALID_SEPARATION: (ValueError, "separation must be finite"),
}


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


def validate_configuration(settings):
    """Reject constructor combinations without a defined and tested compliant row update."""
    required = {
        "velocity_iterations": 0,
        "contact_warmstart": False,
        "contact_regularization": 0.0,
        "contact_shared_anchor": False,
        "contact_friction_shared_anchor": False,
        "friction_mode": "fixed_normal",
        "contact_friction_position_iterations": -1,
    }
    for key, expected in required.items():
        if settings[key] != expected:
            raise ValueError(f"contact_compliance requires {key}={expected!r}; got {settings[key]!r}")
    if settings["iterations"] <= 0:
        raise ValueError("contact_compliance requires positive iterations")


def validate_step(solver):
    """Reject unsupported combinations before any contact preprocessing."""
    if getattr(solver, "_friction_anchor_beta", 0.0) > 0 or getattr(solver, "_friction_anchors_enabled", False):
        raise ValueError("contact_compliance is not validated with friction_anchor_beta > 0")


def start_step(solver, contacts, dt):
    """Check the step's host-side inputs; reads no device data, so it is capture-safe."""
    validate_step(solver)
    if not math.isfinite(dt) or dt <= 0:
        raise ValueError("contact_compliance requires a finite positive dt")
    if contacts is None or any(
        getattr(contacts, key, None) is None
        for key in ("rigid_contact_stiffness", "rigid_contact_damping", "rigid_contact_friction")
    ):
        raise ValueError("contact_compliance requires per-contact hydroelastic material arrays")


class ComplianceBuffers:
    """Persistent per-solver storage for compliant row coefficients and the latched status."""

    def __init__(self, solver):
        device = solver.model.device
        # Same shapes as the row storage they shadow, so routing bounds match the row arrays.
        self.dense_gamma = wp.zeros(solver._diag.shape, dtype=float, device=device)
        self.mf_gamma = wp.zeros(solver._mf_eff_mass_inv.shape, dtype=float, device=device)
        self.dense_weight = wp.zeros(solver._diag.shape, dtype=float, device=device)
        self.mf_weight = wp.zeros(solver._mf_eff_mass_inv.shape, dtype=float, device=device)
        # [active, skipped] for the latest prepared step.
        self.counts = wp.zeros(2, dtype=wp.int32, device=device)
        # Captured steps latch into ``status``; eager steps check and clear ``eager_status``.
        self.status = wp.empty(_STATUS_CODES + 1, dtype=wp.int32, device=device)
        self.eager_status = wp.empty(_STATUS_CODES + 1, dtype=wp.int32, device=device)
        clear_status(self)
        self.dummy_target = wp.zeros((1, 1), dtype=float, device=device)


def clear_status(buffers):
    """Clear both latched status words."""
    for status in (buffers.status, buffers.eager_status):
        wp.launch(_reset_status, dim=1, inputs=[status], device=status.device)


def begin_rows(solver):
    """Clear the step-local coefficients before row construction."""
    buffers = solver._compliance
    buffers.dense_gamma.zero_()
    buffers.mf_gamma.zero_()
    buffers.counts.zero_()
    # The allocator writes slots_needed only after its skip gates, so a stale request
    # from an earlier step could make an excluded contact look overflowed.
    solver._contact_slots_needed.zero_()


def prepare_rows(solver, contacts, dt):
    """Replace the bias and CFM of compliant normal rows on the device, then weight their friction rows."""
    buffers = solver._compliance
    device = solver.model.device
    capturing = wp.get_stream(device).is_capturing
    status = buffers.status if capturing else buffers.eager_status
    wp.launch(
        _check_inputs,
        dim=max(solver._world_count, 1),
        inputs=[
            contacts.rigid_contact_count,
            contacts.rigid_contact_max,
            solver._constraint_count,
            solver._articulation_constraint_max,
            solver._mf_constraint_count,
            solver._free_body_constraint_max,
            solver._row_dropped_all,
            status,
        ],
        device=device,
    )
    if solver._enable_restitution:
        wp.launch(
            _check_restitution,
            dim=solver._shape_restitution.shape[0],
            inputs=[solver._shape_restitution, status],
            device=device,
        )
    has_target = bool(solver._has_prescribed_response)
    wp.launch(
        _prepare_contacts,
        dim=contacts.rigid_contact_max,
        inputs=[
            contacts.rigid_contact_count,
            contacts.rigid_contact_stiffness,
            contacts.rigid_contact_damping,
            contacts.rigid_contact_friction,
            solver._contact_path,
            solver._contact_slot,
            solver._contact_world,
            solver._contact_slots_needed,
            wp.float64(dt),
            solver._constraint_count,
            solver._row_type,
            solver._phi,
            solver._target_velocity,
            solver._mf_constraint_count,
            solver._mf_row_type,
            solver._mf_phi,
            solver._mf_target_velocity if has_target else buffers.dummy_target,
            int(has_target),
            float(solver._cfm),
            wp.float64(solver._cfm),
        ],
        outputs=[
            solver._diag,
            solver._rhs,
            solver._mf_eff_mass_inv,
            solver._mf_rhs,
            buffers.dense_gamma,
            buffers.mf_gamma,
            buffers.dense_weight,
            buffers.mf_weight,
            buffers.counts,
            status,
        ],
        device=device,
    )
    wp.launch(
        _weight_friction_rows,
        dim=solver._row_mu.shape,
        inputs=[
            solver._constraint_count,
            solver._row_type,
            solver._row_parent,
            buffers.dense_gamma,
            buffers.dense_weight,
        ],
        outputs=[solver._row_mu],
        device=device,
    )
    if solver._has_free_rigid_bodies:
        wp.launch(
            _weight_friction_rows,
            dim=solver._mf_row_mu.shape,
            inputs=[
                solver._mf_constraint_count,
                solver._mf_row_type,
                solver._mf_row_parent,
                buffers.mf_gamma,
                buffers.mf_weight,
            ],
            outputs=[solver._mf_row_mu],
            device=device,
        )
    if not capturing:
        # An eager step raises at once; the next step starts from a clear status.
        validate(status, clear=True)


def validate(status, *, clear):
    """Raise the first error latched in ``status`` with one small synchronous readback, optionally clearing it."""
    values = status.numpy()
    if not values[:_STATUS_CODES].any():
        return
    first = int(values[_STATUS_CODES])
    if clear:
        wp.launch(_reset_status, dim=1, inputs=[status], device=status.device)
    error, message = _ERRORS[first & 15]
    contact = first // 16 - 1
    raise error(message if contact < 0 else f"{message} (contact {contact})")


def counts(solver):
    """Return the (active, skipped) contact counts of the latest prepared step."""
    active, skipped = solver._compliance.counts.numpy()
    return int(active), int(skipped)


@wp.kernel
def _reset_status(status: wp.array[wp.int32]):
    for code in range(_STATUS_CODES):
        status[code] = 0
    status[_STATUS_CODES] = _NO_ERROR


@wp.func
def _latch(status: wp.array[wp.int32], code: int, contact: int):
    status[code] = 1
    wp.atomic_min(status, _STATUS_CODES, (contact + 1) * 16 + code)


@wp.kernel
def _check_inputs(
    contact_count: wp.array[int],
    contact_capacity: int,
    dense_count: wp.array[int],
    dense_capacity: int,
    mf_count: wp.array[int],
    mf_capacity: int,
    dropped: wp.array2d[wp.int32],
    status: wp.array[wp.int32],
):
    world = wp.tid()
    if world == 0 and (contact_count[0] < 0 or contact_count[0] > contact_capacity):
        _latch(status, _INPUT_OVERFLOW, -1)
    # Failed reservations roll back the live row count, so check the loss counters too.
    lost = int(0)
    for family in range(dropped.shape[0]):
        if world < dropped.shape[1]:
            lost += dropped[family, world]
    if world < dense_count.shape[0] and dense_count[world] > dense_capacity:
        lost += 1
    if world < mf_count.shape[0] and mf_count[world] > mf_capacity:
        lost += 1
    if lost != 0:
        _latch(status, _ROW_OVERFLOW, -1)


@wp.kernel
def _check_restitution(shape_restitution: wp.array[float], status: wp.array[wp.int32]):
    # The compliant law replaces the position bias; a rebound target has no defined composition with it.
    if shape_restitution[wp.tid()] > 0.0:
        _latch(status, _RESTITUTION, -1)


@wp.func_native("""
return __ddiv_rn(1.0, __dmul_rn(dt, __dadd_rn(__dmul_rn(dt, k), c)));
""")
def _implicit_gamma(k: wp.float64, c: wp.float64, dt: wp.float64) -> wp.float64: ...


@wp.func_native("""
return __ddiv_rn(__dmul_rn(k, phi), __dadd_rn(__dmul_rn(dt, k), c));
""")
def _implicit_bias(k: wp.float64, c: wp.float64, phi: wp.float64, dt: wp.float64) -> wp.float64: ...


@wp.kernel
def _prepare_contacts(
    contact_count: wp.array[int],
    stiffness: wp.array[float],
    damping: wp.array[float],
    friction_scale: wp.array[float],
    paths: wp.array[int],
    slots: wp.array[int],
    worlds: wp.array[int],
    slots_needed: wp.array[int],
    dt: wp.float64,
    dense_count: wp.array[int],
    dense_type: wp.array2d[int],
    dense_phi: wp.array2d[float],
    dense_target: wp.array2d[float],
    mf_count: wp.array[int],
    mf_type: wp.array2d[int],
    mf_phi: wp.array2d[float],
    mf_target: wp.array2d[float],
    has_mf_target: int,
    pgs_cfm: float,
    pgs_cfm64: wp.float64,
    diag: wp.array2d[float],
    rhs: wp.array2d[float],
    mf_inv: wp.array2d[float],
    mf_rhs: wp.array2d[float],
    dense_gamma: wp.array2d[float],
    mf_gamma: wp.array2d[float],
    dense_weight: wp.array2d[float],
    mf_weight: wp.array2d[float],
    counts: wp.array[wp.int32],
    status: wp.array[wp.int32],
):
    contact = wp.tid()
    if contact >= contact_count[0]:
        return
    k = stiffness[contact]
    if not wp.isfinite(k) or k < 0.0:
        _latch(status, _INVALID_STIFFNESS, contact)
        return
    if k == 0.0:
        return
    path = paths[contact]
    slot = slots[contact]
    world = worlds[contact]
    # No capacity request means the allocator intentionally excluded this pair
    # (non-responding bodies, world filtering or a positive-gap gate).
    if path == -1 and slot == -1 and slots_needed[contact] == 0:
        wp.atomic_add(counts, 1, 1)
        return
    rows = dense_gamma.shape[1]
    world_count = dense_gamma.shape[0]
    if path == 1:
        rows = mf_gamma.shape[1]
        world_count = mf_gamma.shape[0]
    if path < 0 or path > 1 or world < 0 or world >= world_count or slot < 0 or slot >= rows:
        _latch(status, _UNMAPPED, contact)
        return
    active_rows = dense_count[world]
    if path == 1:
        active_rows = mf_count[world]
    if slot >= active_rows:
        _latch(status, _BEYOND_ROWS, contact)
        return
    c = damping[contact]
    scale = friction_scale[contact]
    if not wp.isfinite(c) or c < 0.0 or not wp.isfinite(scale) or scale < 0.0:
        _latch(status, _INVALID_MATERIAL, contact)
        return
    weight = scale
    if scale == 0.0:
        weight = 1.0
    # Read only the separation array of the validated route; dense and free-body capacities differ.
    phi = float(0.0)
    if path == 0:
        phi = dense_phi[world, slot]
    else:
        phi = mf_phi[world, slot]
    if not wp.isfinite(phi):
        _latch(status, _INVALID_SEPARATION, contact)
        return
    k64 = wp.float64(k)
    c64 = wp.float64(0.0)
    if phi <= 0.0:
        c64 = wp.float64(c)
    gamma = _implicit_gamma(k64, c64, dt)
    bias = _implicit_bias(k64, c64, wp.float64(phi), dt)
    limit = wp.float64(3.4028234663852886e38)
    if wp.abs(gamma) > limit or wp.abs(bias) > limit:
        _latch(status, _FLOAT32_RANGE, contact)
        return
    gamma32 = float(gamma)
    bias32 = float(bias)
    if path == 0:
        if dense_type[world, slot] != PGS_CONSTRAINT_TYPE_CONTACT:
            _latch(status, _DENSE_TYPE, contact)
            return
        if wp.atomic_add(dense_gamma, world, slot, gamma32) != 0.0:
            _latch(status, _DUPLICATE, contact)
            return
        dense_weight[world, slot] = weight
        rhs[world, slot] = bias32 - dense_target[world, slot]
        # The compliance replaces the row's constraint force mixing, rounded once from float64.
        diag[world, slot] = diag[world, slot] + float(gamma - pgs_cfm64)
    else:
        inv = mf_inv[world, slot]
        if mf_type[world, slot] != PGS_CONSTRAINT_TYPE_CONTACT or inv <= 0.0:
            _latch(status, _MF_TYPE, contact)
            return
        if wp.atomic_add(mf_gamma, world, slot, gamma32) != 0.0:
            _latch(status, _DUPLICATE, contact)
            return
        mf_weight[world, slot] = weight
        target = 0.0
        if has_mf_target != 0:
            target = mf_target[world, slot]
        mf_rhs[world, slot] = bias32 - target
        mf_inv[world, slot] = 1.0 / (1.0 / inv - pgs_cfm + gamma32)
    wp.atomic_add(counts, 0, 1)


@wp.kernel
def _weight_friction_rows(
    row_count: wp.array[int],
    row_type: wp.array2d[int],
    row_parent: wp.array2d[int],
    gamma: wp.array2d[float],
    weight: wp.array2d[float],
    row_mu: wp.array2d[float],
):
    world, row = wp.tid()
    if row >= row_count[world] or row_type[world, row] != PGS_CONSTRAINT_TYPE_FRICTION:
        return
    parent = row_parent[world, row]
    if parent < 0 or parent >= gamma.shape[1] or gamma[world, parent] == 0.0:
        return
    row_mu[world, row] = row_mu[world, row] * weight[world, parent]
