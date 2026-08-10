# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Differentiable elastic-foundation midsole for gradient-based simulation.

This is the autodiff-ready sibling of :mod:`projects.digital_instron_v2.dynamics`.
It exposes the same calibrated Hyperfoam-Maxwell-Pasternak column bed as a live
Warp force model, but re-shaped so the whole per-substep force integration can be
recorded on a :class:`warp.Tape` and differentiated end to end. Two structural
changes make the shipped forward model differentiable:

* **Tape-safe viscoelastic recurrence.** The forward
  :func:`~projects.digital_instron_v2.dynamics.foundation_pressure` updates the
  generalized-Maxwell overstress state *in place* (``q_state[i] = qn``), which
  aliases the same array across substeps and is unsafe to differentiate. Here
  :func:`foundation_pressure_diff` takes the previous substep's state as a
  read-only input and writes the next substep's state into a *separate* array, so
  every buffer on the loss path is written exactly once per rollout.

* **Cone-respecting smooth Coulomb friction.** The forward model's anchored
  bristle friction switches on an integer stick/slip flag, which has no useful
  gradient. :func:`foundation_apply_diff` instead uses
  :func:`warp.smooth_normalize` (the pseudo-Huber smoothed direction), giving a
  friction force ``-mu * fn * v_tan / sqrt(delta^2 + |v_tan|^2)`` that is smooth
  through zero relative velocity and never exceeds the cone ``mu * fn``.

The constitutive parameters that a fit would vary -- the equilibrium shear
modulus, Hyperfoam exponent, Maxwell overstress ratio, and Pasternak coupling --
are held in a length-4 ``requires_grad`` device array so gradients of any
simulation objective with respect to the foam material are available directly
from ``material_params.grad``. The Coulomb friction coefficient ``mu`` is held in
a separate length-1 ``requires_grad`` ``friction_params`` array, so a lateral- or
shear-force objective can be differentiated with respect to friction as well
(friction identification), independent of the constitutive fit.

See :mod:`projects.digital_instron_v2.dynamics` for the (faster, forward-only)
production force model and the geometry/calibration helpers reused here.
"""

from __future__ import annotations

import numpy as np
import warp as wp

from .core import EFFECTIVE_POISSON_RATIO, Material
from .dynamics import POISSON, TAU_S, FoundationConfig, FoundationParams

# Indices into the differentiable ``material_params`` vector.
MAT_G_EQ = wp.constant(0)  # equilibrium shear modulus G_inst * equilibrium_fraction [Pa]
MAT_ALPHA = wp.constant(1)  # Hyperfoam exponent
MAT_OVERSTRESS = wp.constant(2)  # (1 - equilibrium_fraction) / equilibrium_fraction
MAT_PASTERNAK = wp.constant(3)  # Pasternak lateral coupling [N/m]

# Index into the differentiable ``friction_params`` vector.
FRIC_MU = wp.constant(0)  # Coulomb friction coefficient (smooth-cone bound)


@wp.func
def _hyperfoam_pressure_diff(
    strain: wp.float32, g_eq: wp.float32, alpha: wp.float32, p: FoundationParams
) -> wp.float32:
    """Positive uniaxial compression pressure from first-order Hyperfoam.

    Identical law to :func:`~projects.digital_instron_v2.dynamics._hyperfoam_pressure`
    but with the differentiable modulus ``g_eq`` and exponent ``alpha`` passed as
    scalars so gradients flow into them.
    """
    stretch = 1.0 - strain
    if stretch < p.stretch_floor:
        stretch = p.stretch_floor
    volume_ratio = wp.pow(stretch, 1.0 - 2.0 * POISSON)
    return 2.0 * g_eq / (alpha * stretch) * (wp.pow(volume_ratio, -alpha * p.beta) - wp.pow(stretch, alpha))


@wp.kernel
def foundation_pressure_diff(
    carrier: wp.int32,
    dt: wp.float32,
    body_q: wp.array[wp.transform],
    anchor_local: wp.array[wp.vec3],
    z_free: wp.array[wp.float32],
    rest_len: wp.array[wp.float32],
    params: FoundationParams,
    material_params: wp.array[wp.float32],
    q_prev: wp.array[wp.float32],
    peq_prev: wp.array[wp.float32],
    q_out: wp.array[wp.float32],
    peq_out: wp.array[wp.float32],
    compression: wp.array[wp.float32],
    base_pressure: wp.array[wp.float32],
):
    """Compression, Hyperfoam equilibrium pressure, and a tape-safe Maxwell recurrence.

    Reads the previous substep's overstress state (``q_prev``) and equilibrium
    pressure (``peq_prev``) and writes this substep's state into the separate
    ``q_out``/``peq_out`` arrays (no in-place aliasing), so the recurrence can be
    replayed on a backward pass. ``g_eq``, ``alpha`` and the overstress ratio are
    read from the differentiable ``material_params`` vector.
    """
    i = wp.tid()
    g_eq = material_params[MAT_G_EQ]
    alpha = material_params[MAT_ALPHA]
    overstress = material_params[MAT_OVERSTRESS]

    world = wp.transform_point(body_q[carrier], anchor_local[i])
    comp = z_free[i] - world[2]
    if comp < 0.0:
        comp = 0.0
    compression[i] = comp
    strain = comp / rest_len[i]
    peq = _hyperfoam_pressure_diff(strain, g_eq, alpha, params)
    decay = wp.exp(-dt / TAU_S)
    ramp = TAU_S * (1.0 - decay) / dt
    qn = decay * q_prev[i] + overstress * ramp * (peq - peq_prev[i])
    q_out[i] = qn
    peq_out[i] = peq
    base_pressure[i] = peq + qn


@wp.kernel
def foundation_apply_diff(
    carrier: wp.int32,
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    body_com: wp.array[wp.vec3],
    anchor_local: wp.array[wp.vec3],
    area: wp.array[wp.float32],
    neighbors: wp.array2d[wp.int32],
    compression: wp.array[wp.float32],
    base_pressure: wp.array[wp.float32],
    params: FoundationParams,
    material_params: wp.array[wp.float32],
    friction_params: wp.array[wp.float32],
    friction_smoothing: wp.float32,
    body_f: wp.array[wp.spatial_vector],
    normal_force: wp.array[wp.float32],
    cop_moment: wp.array[wp.vec3],
    active_count: wp.array[wp.int32],
):
    """Pasternak coupling, per-column wrench into ``body_f``, and force diagnostics.

    Matches the normal-force path of
    :func:`~projects.digital_instron_v2.dynamics.foundation_apply` (Pasternak
    Laplacian, pressure floor, Kelvin-Voigt normal damping) but replaces the
    non-differentiable anchored bristle friction with a smooth, cone-respecting
    Coulomb law built on :func:`warp.smooth_normalize`. The Pasternak coupling is
    read from the differentiable ``material_params`` vector and the friction
    coefficient ``mu`` from the differentiable ``friction_params`` vector, so a
    lateral-force objective can be differentiated with respect to friction too.
    """
    i = wp.tid()
    pasternak = material_params[MAT_PASTERNAK]

    ci = compression[i]
    lap = -4.0 * ci
    for side in range(4):
        j = neighbors[i, side]
        if j >= 0:
            lap += compression[j]
        elif j == -1:
            lap += ci  # natural (zero-gradient) footprint boundary
    lap *= params.inv_h2

    pressure = base_pressure[i] - pasternak * lap
    if pressure < 0.0:
        pressure = 0.0
    fn = pressure * area[i]

    q_body = body_q[carrier]
    world = wp.transform_point(q_body, anchor_local[i])
    com_world = wp.transform_point(q_body, body_com[carrier])
    r = world - com_world
    vel = body_qd[carrier]
    point_vel = wp.spatial_top(vel) + wp.cross(wp.spatial_bottom(vel), r)

    if ci > 0.0:
        fn = fn - params.normal_damping * point_vel[2]
    if fn < 0.0:
        fn = 0.0

    # Cone-respecting smooth Coulomb friction: ft = -mu * fn * smooth_normalize(v_tan).
    # smooth_normalize(v, delta) = v / sqrt(delta^2 + |v|^2) has magnitude < 1, so the
    # tangential force is bounded by the cone mu * fn and is smooth through v_tan = 0.
    mu = friction_params[FRIC_MU]
    f_tan = wp.vec2(0.0, 0.0)
    if fn > 0.0 and mu > 0.0:
        v_tan = wp.vec2(point_vel[0], point_vel[1])
        f_tan = -mu * fn * wp.smooth_normalize(v_tan, friction_smoothing)

    force = wp.vec3(f_tan[0], f_tan[1], fn)
    wp.atomic_add(body_f, carrier, wp.spatial_vector(force, wp.cross(r, force)))
    wp.atomic_add(normal_force, 0, fn)
    wp.atomic_add(cop_moment, 0, wp.vec3(world[0] * fn, world[1] * fn, 0.0))
    if ci > 0.0:
        wp.atomic_add(active_count, 0, 1)


class DifferentiableMidsoleFoundation:
    """Autodiff-ready elastic-foundation force model for a fixed-length rollout.

    Unlike :class:`~projects.digital_instron_v2.dynamics.MidsoleFoundation`, which
    ping-pongs a single set of buffers for speed, this variant keeps a separate
    compression/pressure/overstress history for each of ``num_substeps`` substeps
    so the whole rollout can be recorded on one :class:`warp.Tape` and
    differentiated. Drive it exactly like the forward model, but pass the current
    substep index so the correct history slot and the previous overstress state
    are used::

        foundation = DifferentiableMidsoleFoundation(..., num_substeps=N)
        tape = wp.Tape()
        with tape:
            for t in range(N):
                states[t].body_f.zero_()
                foundation.apply(states[t], t, dt)
                solver.step(states[t], states[t + 1], None, None, dt)
            wp.launch(loss_kernel, ...)
        tape.backward(loss)
        # gradients w.r.t. the foam material and the friction coefficient:
        foundation.material_params.grad.numpy()
        foundation.friction_params.grad.numpy()

    Args:
        anchor_local: Column attachment points in the carrier body frame [m],
            shape ``[column_count, 3]``.
        z_free: World height of each uncompressed foam column top [m], shape
            ``[column_count]``.
        rest_len: Column rest thickness [m], shape ``[column_count]``.
        area: Tributary area per column [m^2], shape ``[column_count]``.
        neighbors: Pasternak 4-neighbour indices, shape ``[column_count, 4]``.
        spacing_m: Column grid spacing [m].
        material: Calibrated :class:`~projects.digital_instron_v2.core.Material`.
        carrier_body: Index of the rigid body carrying the foundation.
        body_com: Model center-of-mass array (``model.body_com``).
        num_substeps: Number of substeps in one differentiated rollout.
        config: Dynamic :class:`~projects.digital_instron_v2.dynamics.FoundationConfig`.
        friction_smoothing: Tangential velocity smoothing scale for the smooth
            Coulomb friction [m/s].
        device: Warp device (must match the carrier state's device).
    """

    def __init__(
        self,
        anchor_local: np.ndarray,
        z_free: np.ndarray,
        rest_len: np.ndarray,
        area: np.ndarray,
        neighbors: np.ndarray,
        spacing_m: float,
        material: Material,
        carrier_body: int,
        body_com,
        num_substeps: int,
        config: FoundationConfig | None = None,
        friction_smoothing: float = 0.05,
        device=None,
    ) -> None:
        config = config or FoundationConfig()
        self.device = device
        self.carrier = int(carrier_body)
        self.body_com = body_com
        self.column_count = int(len(rest_len))
        self.num_substeps = int(num_substeps)
        self.friction_smoothing = float(friction_smoothing)

        params = FoundationParams()
        params.g_eq = material.instantaneous_shear_modulus_pa * material.equilibrium_fraction
        params.alpha = material.hyperfoam_exponent
        params.beta = EFFECTIVE_POISSON_RATIO / (1.0 - 2.0 * EFFECTIVE_POISSON_RATIO)
        params.overstress = (1.0 - material.equilibrium_fraction) / material.equilibrium_fraction
        params.pasternak = material.pasternak_n_per_m
        params.inv_h2 = 1.0 / spacing_m**2
        params.stretch_floor = config.stretch_floor
        params.normal_damping = config.normal_damping
        params.friction_kt = config.friction_stiffness
        params.friction_kv = config.friction
        params.mu = config.mu
        self.params = params

        # Differentiable constitutive vector [g_eq, alpha, overstress, pasternak].
        self.material_params = wp.array(
            np.array([params.g_eq, params.alpha, params.overstress, params.pasternak], np.float32),
            dtype=wp.float32,
            device=device,
            requires_grad=True,
        )
        # Differentiable friction vector [mu] for gradient-based friction identification.
        self.friction_params = wp.array(
            np.array([params.mu], np.float32),
            dtype=wp.float32,
            device=device,
            requires_grad=True,
        )

        m = self.column_count
        self.anchor_local = wp.array(np.ascontiguousarray(anchor_local, np.float32), dtype=wp.vec3, device=device)
        self.z_free = wp.array(np.ascontiguousarray(z_free, np.float32), dtype=wp.float32, device=device)
        self.rest_len = wp.array(np.ascontiguousarray(rest_len, np.float32), dtype=wp.float32, device=device)
        self.area = wp.array(np.ascontiguousarray(area, np.float32), dtype=wp.float32, device=device)
        self.neighbors = wp.array(np.ascontiguousarray(neighbors, np.int32), dtype=wp.int32, device=device)

        def grad_zeros():
            return wp.zeros(m, dtype=wp.float32, device=device, requires_grad=True)

        # Per-substep history so nothing on the loss path is overwritten within a rollout.
        self.compression = [grad_zeros() for _ in range(self.num_substeps)]
        self.base_pressure = [grad_zeros() for _ in range(self.num_substeps)]
        self.q_state = [grad_zeros() for _ in range(self.num_substeps)]
        self.peq_prev = [grad_zeros() for _ in range(self.num_substeps)]
        # Fixed zero initial overstress state (substep 0 reads these).
        self.q_init = grad_zeros()
        self.peq_init = grad_zeros()

        # Diagnostics (off the loss path); overwritten each substep.
        self.normal_force = wp.zeros(1, dtype=wp.float32, device=device, requires_grad=True)
        self.cop_moment = wp.zeros(1, dtype=wp.vec3, device=device, requires_grad=True)
        self.active = wp.zeros(1, dtype=wp.int32, device=device)

    def apply(self, state, substep: int, dt: float) -> None:
        """Accumulate the foundation wrench into ``state.body_f`` for one substep.

        Args:
            state: Simulation state supplying the carrier pose/velocity and
                receiving the wrench; ``state.body_f`` must be cleared beforehand.
            substep: Substep index in ``[0, num_substeps)``; selects the history
                slot and the previous overstress state for the recurrence.
            dt: Substep duration [s].
        """
        t = int(substep)
        q_prev = self.q_init if t == 0 else self.q_state[t - 1]
        peq_prev = self.peq_init if t == 0 else self.peq_prev[t - 1]

        self.normal_force.zero_()
        self.cop_moment.zero_()
        self.active.zero_()

        wp.launch(
            foundation_pressure_diff,
            dim=self.column_count,
            inputs=[
                self.carrier,
                dt,
                state.body_q,
                self.anchor_local,
                self.z_free,
                self.rest_len,
                self.params,
                self.material_params,
                q_prev,
                peq_prev,
                self.q_state[t],
                self.peq_prev[t],
                self.compression[t],
                self.base_pressure[t],
            ],
            device=self.device,
        )
        wp.launch(
            foundation_apply_diff,
            dim=self.column_count,
            inputs=[
                self.carrier,
                state.body_q,
                state.body_qd,
                self.body_com,
                self.anchor_local,
                self.area,
                self.neighbors,
                self.compression[t],
                self.base_pressure[t],
                self.params,
                self.material_params,
                self.friction_params,
                self.friction_smoothing,
                state.body_f,
                self.normal_force,
                self.cop_moment,
                self.active,
            ],
            device=self.device,
        )

    def zero_grad(self) -> None:
        """Zero the accumulated gradients on the differentiable buffers."""
        self.material_params.grad.zero_()
        self.friction_params.grad.zero_()
        for buf in (*self.compression, *self.base_pressure, *self.q_state, *self.peq_prev):
            buf.grad.zero_()
        self.q_init.grad.zero_()
        self.peq_init.grad.zero_()

    def diagnostics(self) -> dict[str, float]:
        """Return the last substep's total normal force, center of pressure, and active count."""
        fz = float(self.normal_force.numpy()[0])
        moment = self.cop_moment.numpy()[0]
        cop = (float(moment[0] / fz), float(moment[1] / fz)) if fz > 1.0e-9 else (0.0, 0.0)
        return {
            "normal_force_n": fz,
            "cop_x_m": cop[0],
            "cop_y_m": cop[1],
            "active_columns": int(self.active.numpy()[0]),
        }
