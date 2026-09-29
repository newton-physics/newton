# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Portable, GPU-native runtime for an identified digital shoe foundation."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import warp as wp

from .contact import (
    bristle_step,
    contact_kinematics,
    contact_wrench,
    normal_reaction,
    pasternak_coupling,
    pasternak_flux,
    surround_balance,
)
from .contact import cone_viscous_scale as _cone_viscous_scale  # noqa: F401  # compatibility export
from .material import (
    HYPERFOAM_ALPHA_FLOOR,  # noqa: F401  # compatibility export
    hyperfoam_pressure,
    maxwell_coefficients,
    maxwell_coefficients_numpy,
    maxwell_step,
)
from .material import (
    ogden_hill_term as _hyperfoam_term,  # noqa: F401  # compatibility export
)


@dataclass(frozen=True)
class ShoeMaterial:
    """Effective intact-shoe constitutive parameters used by the runtime.

    The equilibrium network is a **two-term** Ogden-Hill (Abaqus Hyperfoam)
    series, ``p_eq = sum_n 2 mu_n / (alpha_n lambda) (J^(-alpha_n beta) -
    lambda^alpha_n)``. One first-order term cannot cover both the 0-10% secant
    the published foam tables report and the 74-90% peak strains the bench
    fixtures reach, so a second term is carried. ``instantaneous_shear_modulus_2_pa
    = 0`` disables it exactly and recovers the historical single-term artifact.

    ``pasternak_n_per_m`` is a *reported* quantity, kept so the artifact schema
    stays readable and backward compatible. Nothing consumes it: the runtime
    derives each column's own Pasternak coefficient from the equilibrium
    Ogden-Hill shear modulus and that column's rest thickness,
    ``k_i = mu_eq * t_i``, which is the shear-layer definition ``G * t`` and adds
    no free parameter. Exporters write the bed-mean of that rule here.
    """

    instantaneous_shear_modulus_pa: float
    hyperfoam_exponent: float
    equilibrium_fraction: float
    pasternak_n_per_m: float
    effective_poisson_ratio: float = 0.0
    maxwell_relaxation_time_s: float = 0.08
    instantaneous_shear_modulus_2_pa: float = 0.0
    hyperfoam_exponent_2: float = 1.0

    def __post_init__(self) -> None:
        values = tuple(self.__dict__.values())
        if not np.all(np.isfinite(values)):
            raise ValueError("shoe material parameters must be finite")
        if self.instantaneous_shear_modulus_pa <= 0.0 or self.hyperfoam_exponent <= 0.0:
            raise ValueError("shear modulus and Hyperfoam exponent must be positive")
        if self.instantaneous_shear_modulus_2_pa < 0.0:
            raise ValueError("second Ogden-Hill shear modulus must be nonnegative")
        if not 0.0 < self.equilibrium_fraction <= 1.0:
            raise ValueError("equilibrium fraction must be in (0, 1]")
        if self.pasternak_n_per_m < 0.0:
            raise ValueError("Pasternak coupling must be nonnegative")
        if not -1.0 < self.effective_poisson_ratio < 0.5:
            raise ValueError("effective Poisson ratio must be in (-1, 0.5)")
        if self.maxwell_relaxation_time_s <= 0.0:
            raise ValueError("Maxwell relaxation time must be positive")

    @property
    def equilibrium_shear_modulus_pa(self) -> float:
        """Equilibrium Ogden-Hill shear modulus, the SUM over both terms [Pa].

        ``mu_eq = (G_1 + G_2) * f_eq``. Every term of an Ogden-Hill series adds
        ``2 mu_n`` to the small-strain compressive tangent regardless of its
        exponent, so the series modulus is the sum and this is the quantity the
        per-column Pasternak rule ``k_i = mu_eq * t_i`` uses.
        """
        return (self.instantaneous_shear_modulus_pa + self.instantaneous_shear_modulus_2_pa) * (
            self.equilibrium_fraction
        )


@wp.struct
class FoundationParams:
    """Device-side constitutive and contact constants for the column bed."""

    g_eq: wp.float32  # first-term equilibrium shear modulus G_inst * equilibrium_fraction [Pa]
    alpha: wp.float32  # first-term Hyperfoam exponent
    g_eq2: wp.float32  # second-term equilibrium shear modulus [Pa]; zero disables the term
    alpha2: wp.float32  # second-term Hyperfoam exponent
    beta: wp.float32  # poisson / (1 - 2 poisson)
    one_minus_two_poisson: wp.float32  # volumetric stretch exponent
    tau_s: wp.float32  # Maxwell relaxation time [s]
    overstress: wp.float32  # (1 - equilibrium_fraction) / equilibrium_fraction
    inv_h2: wp.float32  # 1 / spacing^2 [1/m^2]
    stretch_floor: wp.float32  # minimum stretch (foam densification limit)
    normal_damping: wp.float32  # per-column Kelvin-Voigt normal damping [N.s/m]
    friction_kt: wp.float32  # uniform bristle tangential stiffness [N/m per column]
    friction_kv: wp.float32  # uniform bristle tangential damping [N.s/m per column]
    friction_viscous_ratio: wp.float32  # viscous cap as a fraction of the cone mu*fn
    friction_release_dwell_s: wp.float32  # unloaded dwell before the stick point is discarded [s]
    mu: wp.float32  # Coulomb friction coefficient
    friction_model: wp.int32  # 0: legacy, 1: maxwell, 2: column_maxwell, 3: elastic_coulomb
    friction_relaxation_time_s: wp.float32  # shear relaxation time [s]


@wp.func
def _hyperfoam_pressure(strain: wp.float32, p: FoundationParams) -> wp.float32:
    """Positive uniaxial compression pressure from the two-term Hyperfoam law.

    Sums :func:`_hyperfoam_term` over both Ogden-Hill terms. One first-order term
    has a single shape exponent and cannot match both the 0-10% secant of the
    published foam tables and the 74-90% peak strains the bench fixtures reach;
    the second term restores that freedom without a second ``pow`` per term.

    At the measured zero effective Poisson ratio ``beta`` is zero and
    ``one_minus_two_poisson`` is one, so the volumetric factor is ``pow(x, 0)``
    with ``x >= stretch_floor > 0``. That is exactly one on CPU and CUDA and
    needs no special case; the stretch floor is what keeps it away from
    ``pow(0, 0)``.
    """
    return hyperfoam_pressure(
        strain, p.g_eq, p.alpha, p.g_eq2, p.alpha2, p.beta, p.one_minus_two_poisson, p.stretch_floor
    )


def set_material_block(params: FoundationParams, material) -> None:
    """Write every constitutive constant of ``material`` into ``params``.

    The one place a host derives a whole device-side material, so the constructor
    and :meth:`MidsoleFoundation.set_world_material` cannot drift apart and leave a
    randomized world carrying half of the old foam.

    Args:
        params: Device-side constants to fill in place. Contact and friction
            fields are left untouched; they belong to
            :class:`FoundationConfig`, not to the material.
        material: Any material carrying the Ogden-Hill series,
            ``equilibrium_fraction``, ``effective_poisson_ratio`` and
            ``maxwell_relaxation_time_s``.
    """
    set_hyperfoam_series(params, material)
    poisson = float(getattr(material, "effective_poisson_ratio", 0.0))
    params.beta = poisson / (1.0 - 2.0 * poisson)
    params.one_minus_two_poisson = 1.0 - 2.0 * poisson
    params.tau_s = float(getattr(material, "maxwell_relaxation_time_s", 0.08))
    params.overstress = (1.0 - material.equilibrium_fraction) / material.equilibrium_fraction


def clone_params(params: FoundationParams) -> FoundationParams:
    """Return a field-by-field copy of a device-side constitutive block."""
    clone = FoundationParams()
    for name in FoundationParams.vars:
        setattr(clone, name, getattr(params, name))
    return clone


def set_hyperfoam_series(params: FoundationParams, material) -> None:
    """Write both equilibrium Ogden-Hill terms of ``material`` into ``params``.

    The only place a host builds the device-side constitutive block, so a term
    cannot be forgotten at one call site and silently drop out of one solver.

    Args:
        params: Device-side constants to fill in place.
        material: Any material carrying ``instantaneous_shear_modulus_pa``,
            ``hyperfoam_exponent``, ``instantaneous_shear_modulus_2_pa``,
            ``hyperfoam_exponent_2`` and ``equilibrium_fraction``
            (:class:`ShoeMaterial` or
            :class:`projects.digital_instron_v2.core.Material`).
    """
    fraction = float(material.equilibrium_fraction)
    params.g_eq = float(material.instantaneous_shear_modulus_pa) * fraction
    params.alpha = float(material.hyperfoam_exponent)
    params.g_eq2 = float(material.instantaneous_shear_modulus_2_pa) * fraction
    params.alpha2 = float(material.hyperfoam_exponent_2)


@wp.func
def _pasternak_coupling(t_i: wp.float32, t_j: wp.float32, p: FoundationParams) -> wp.float32:
    """Pasternak coefficient of the shear layer between two columns [N/m].

    A Pasternak layer coefficient is ``G * t``. The shear modulus is the foam's
    own equilibrium Ogden-Hill modulus ``mu_eq``, which for a two-term series is
    the *sum* :attr:`FoundationParams.g_eq` + :attr:`FoundationParams.g_eq2`, and
    the layer thickness at the shared face is the mean of the two column rest
    thicknesses. Averaging keeps the pair conductance symmetric, so the lateral
    flux summed over the bed is exactly zero and the layer can only move load,
    never create it.
    """
    return pasternak_coupling(t_i, t_j, p.g_eq + p.g_eq2)


@wp.func
def _foundation_pressure(
    i: int,
    carrier: wp.array[wp.int32],
    column_count: wp.int32,
    dt: wp.float32,
    body_q: wp.array[wp.transform],
    anchor_local: wp.array[wp.vec3],
    z_free: wp.array[wp.float32],
    rest_len: wp.array[wp.float32],
    params: wp.array[FoundationParams],
    q_state: wp.array[wp.float32],
    peq_prev: wp.array[wp.float32],
    compression: wp.array[wp.float32],
    base_pressure: wp.array[wp.float32],
):
    """Compression, Hyperfoam equilibrium pressure, and real-time Maxwell overstress.

    Launched over ``world_count * column_count`` threads. Per-column state is
    tiled per world and indexed by the thread id, the shared per-column constants
    are indexed by the column alone, and the constitutive constants are indexed by
    the world, so every world can carry its own material. See
    :class:`MidsoleFoundation`.
    """
    world_index = i // column_count
    column = i - world_index * column_count
    p = params[world_index]
    world = wp.transform_point(body_q[carrier[world_index]], anchor_local[column])
    comp = z_free[i] - world[2]
    if comp < 0.0:
        comp = 0.0
    compression[i] = comp
    strain = comp / rest_len[column]
    peq = _hyperfoam_pressure(strain, p)
    decay, ramp = maxwell_coefficients(dt, p.tau_s)
    qn = maxwell_step(q_state[i], peq, peq_prev[i], p.overstress, decay, ramp)
    q_state[i] = qn
    peq_prev[i] = peq
    base_pressure[i] = peq + qn


@wp.kernel
def foundation_pressure(
    carrier: wp.array[wp.int32],
    column_count: wp.int32,
    dt: wp.float32,
    body_q: wp.array[wp.transform],
    anchor_local: wp.array[wp.vec3],
    z_free: wp.array[wp.float32],
    rest_len: wp.array[wp.float32],
    params: wp.array[FoundationParams],
    q_state: wp.array[wp.float32],
    peq_prev: wp.array[wp.float32],
    compression: wp.array[wp.float32],
    base_pressure: wp.array[wp.float32],
):
    """Run :func:`_foundation_pressure` once per independent column."""
    _foundation_pressure(
        wp.tid(),
        carrier,
        column_count,
        dt,
        body_q,
        anchor_local,
        z_free,
        rest_len,
        params,
        q_state,
        peq_prev,
        compression,
        base_pressure,
    )


@wp.func
def _foundation_column_forces(
    i: wp.int32,
    carrier: wp.array[wp.int32],
    column_count: wp.int32,
    dt: wp.float32,
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    body_com: wp.array[wp.vec3],
    anchor_local: wp.array[wp.vec3],
    area: wp.array[wp.float32],
    rest_len: wp.array[wp.float32],
    neighbors: wp.array2d[wp.int32],
    compression: wp.array[wp.float32],
    base_pressure: wp.array[wp.float32],
    tangent_anchor: wp.array[wp.vec2],
    tangent_stuck: wp.array[wp.int32],
    tangent_dwell: wp.array[wp.float32],
    friction_kt: wp.array[wp.float32],
    friction_kv: wp.array[wp.float32],
    params: wp.array[FoundationParams],
    column_force: wp.array[wp.vec3],
    column_pressed: wp.array[wp.float32],
    ground_plane: wp.int32,
    ground_height: wp.float32,
) -> tuple[wp.vec3, wp.vec3]:
    """Compute signed carrier transfer and, when requested, the external plane traction."""
    world_index = i // column_count
    base = world_index * column_count
    column = i - base
    p = params[world_index]
    ci = compression[i]
    flux = pasternak_flux(column, base, compression, rest_len, neighbors, p.g_eq + p.g_eq2)

    body = carrier[world_index]
    point, _com_world, point_vel, gap = contact_kinematics(
        body_q[body], body_qd[body], body_com[body], anchor_local[column], ground_height, ground_plane
    )
    reaction = normal_reaction(
        ci,
        base_pressure[i],
        area[column],
        p.normal_damping,
        point_vel[2],
        gap,
        ground_plane,
    )
    # The shear-layer flux stays unclamped: it redistributes load between columns and
    # clipping it per column would invent net support under uncompressed foam.
    fn = reaction - flux
    transfer_pressed = fn
    if transfer_pressed < 0.0:
        transfer_pressed = 0.0
    pressed = transfer_pressed
    if ground_plane != 0:
        pressed = reaction

    f_tan, next_anchor, next_stuck, next_dwell = bristle_step(
        wp.vec2(point[0], point[1]),
        wp.vec2(point_vel[0], point_vel[1]),
        dt,
        pressed,
        friction_kt[column],
        friction_kv[column],
        p.mu,
        p.friction_viscous_ratio,
        p.friction_release_dwell_s,
        tangent_anchor[i],
        tangent_stuck[i],
        tangent_dwell[i],
    )
    tangent_anchor[i] = next_anchor
    tangent_stuck[i] = next_stuck
    tangent_dwell[i] = next_dwell

    column_force[i] = wp.vec3(f_tan[0], f_tan[1], fn)
    column_pressed[i] = transfer_pressed
    return wp.vec3(f_tan[0], f_tan[1], reaction), point


@wp.kernel
def foundation_apply(
    carrier: wp.array[wp.int32],
    column_count: wp.int32,
    dt: wp.float32,
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    body_com: wp.array[wp.vec3],
    anchor_local: wp.array[wp.vec3],
    area: wp.array[wp.float32],
    rest_len: wp.array[wp.float32],
    neighbors: wp.array2d[wp.int32],
    compression: wp.array[wp.float32],
    base_pressure: wp.array[wp.float32],
    tangent_anchor: wp.array[wp.vec2],
    tangent_stuck: wp.array[wp.int32],
    tangent_dwell: wp.array[wp.float32],
    friction_kt: wp.array[wp.float32],
    friction_kv: wp.array[wp.float32],
    params: wp.array[FoundationParams],
    column_force: wp.array[wp.vec3],
    column_pressed: wp.array[wp.float32],
):
    """Evaluate the legacy generic foundation at its nominal carrier anchors.

    This exported kernel retains its original signature and transfer-traction
    convention. :func:`foundation_apply_ground` opts into external plane contact.
    Both paths share the same material, shear transfer and bristle law.
    """
    _foundation_column_forces(
        wp.tid(),
        carrier,
        column_count,
        dt,
        body_q,
        body_qd,
        body_com,
        anchor_local,
        area,
        rest_len,
        neighbors,
        compression,
        base_pressure,
        tangent_anchor,
        tangent_stuck,
        tangent_dwell,
        friction_kt,
        friction_kv,
        params,
        column_force,
        column_pressed,
        0,
        0.0,
    )


@wp.kernel
def foundation_apply_ground(
    carrier: wp.array[wp.int32],
    column_count: wp.int32,
    dt: wp.float32,
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    body_com: wp.array[wp.vec3],
    anchor_local: wp.array[wp.vec3],
    area: wp.array[wp.float32],
    rest_len: wp.array[wp.float32],
    neighbors: wp.array2d[wp.int32],
    compression: wp.array[wp.float32],
    base_pressure: wp.array[wp.float32],
    tangent_anchor: wp.array[wp.vec2],
    tangent_stuck: wp.array[wp.int32],
    tangent_dwell: wp.array[wp.float32],
    friction_kt: wp.array[wp.float32],
    friction_kv: wp.array[wp.float32],
    params: wp.array[FoundationParams],
    column_force: wp.array[wp.vec3],
    column_pressed: wp.array[wp.float32],
    ground_height: wp.float32,
    ground_force: wp.array[wp.vec3],
    contact_point: wp.array[wp.vec3],
):
    """Write external plane traction separately from signed internal load transfer.

    Each world owns a tile of column state. Neighbor indices are shared and
    offset within that tile, so neither forces nor histories cross worlds.
    These arrays are reduced in a fixed order rather than atomically per column.
    """
    i = wp.tid()
    force, point = _foundation_column_forces(
        i,
        carrier,
        column_count,
        dt,
        body_q,
        body_qd,
        body_com,
        anchor_local,
        area,
        rest_len,
        neighbors,
        compression,
        base_pressure,
        tangent_anchor,
        tangent_stuck,
        tangent_dwell,
        friction_kt,
        friction_kv,
        params,
        column_force,
        column_pressed,
        1,
        ground_height,
    )
    ground_force[i] = force
    contact_point[i] = point


@wp.func
def _cycle_overstress_column(
    i: wp.int32,
    compression: wp.array2d[wp.float32],
    slack: wp.array[wp.float32],
    dt_s: wp.array[wp.float32],
    params: FoundationParams,
    fraction: wp.float32,
    tau_s: wp.float32,
    overstress_out: wp.array2d[wp.float32],
):
    """Evaluate the shared periodic Maxwell recurrence for one column."""
    frames = compression.shape[0]
    thickness = slack[i]
    state = float(0.0)
    decay_product = float(1.0)
    previous = _hyperfoam_pressure(compression[frames - 1, i] / thickness, params)
    for frame in range(frames):
        equilibrium = _hyperfoam_pressure(compression[frame, i] / thickness, params)
        decay, ramp = maxwell_coefficients(dt_s[frame], tau_s)
        state = maxwell_step(state, equilibrium, previous, fraction, decay, ramp)
        decay_product *= decay
        previous = equilibrium
    state = state / (1.0 - decay_product)
    previous = _hyperfoam_pressure(compression[frames - 1, i] / thickness, params)
    for frame in range(frames):
        equilibrium = _hyperfoam_pressure(compression[frame, i] / thickness, params)
        decay, ramp = maxwell_coefficients(dt_s[frame], tau_s)
        state = maxwell_step(state, equilibrium, previous, fraction, decay, ramp)
        previous = equilibrium
        overstress_out[frame, i] = state


@wp.kernel
def cycle_overstress(
    compression: wp.array2d[wp.float32],
    slack: wp.array[wp.float32],
    dt_s: wp.array[wp.float32],
    params: FoundationParams,
    fraction: wp.float32,
    tau_s: wp.float32,
    overstress_out: wp.array2d[wp.float32],
):
    """Write one column's periodic Maxwell overstress for every frame of a cycle.

    Each thread owns a column and walks the cycle twice: the first pass finds the
    periodic state, the second records it frame by frame. :func:`surround_sweep`
    and :func:`cycle_force` then share this overstress, so the relaxed surround
    balances the same load the summed reaction later reports.
    """
    _cycle_overstress_column(wp.tid(), compression, slack, dt_s, params, fraction, tau_s, overstress_out)


@wp.func
def _cycle_force_frame(
    frame: wp.int32,
    compression: wp.array2d[wp.float32],
    overstress: wp.array2d[wp.float32],
    slack: wp.array[wp.float32],
    params: FoundationParams,
    area: wp.float32,
) -> wp.float32:
    """Sum one cycle frame in the shared deterministic column order [N]."""
    # A fixed column order makes repeated identification independent of atomic scheduling.
    total = float(0.0)
    for i in range(compression.shape[1]):
        equilibrium = _hyperfoam_pressure(compression[frame, i] / slack[i], params)
        total += normal_reaction(compression[frame, i], equilibrium + overstress[frame, i], area, 0.0, 0.0, 0.0, 0)
    return total


@wp.kernel
def cycle_force(
    compression: wp.array2d[wp.float32],
    overstress: wp.array2d[wp.float32],
    slack: wp.array[wp.float32],
    params: FoundationParams,
    area: wp.float32,
    force_out: wp.array[wp.float32],
):
    """Sum the unilateral ground reaction of every column into each frame of a cycle.

    The overstress comes from :func:`cycle_overstress` on the same compression,
    the clamp keeps the reaction unilateral, and the shear flux cancels
    internally, so the sum is the load an Instron would measure.
    """
    frame, lane = wp.tid()
    if lane != 0:
        return
    force_out[frame] += _cycle_force_frame(frame, compression, overstress, slack, params, area)


@wp.func
def _surround_balance(
    c: wp.float32,
    rigid: wp.float32,
    pull: wp.float32,
    coupling_sum: wp.float32,
    thickness: wp.float32,
    overstress_base: wp.float32,
    overstress_gain: wp.float32,
    params: FoundationParams,
    area: wp.float32,
    attachment: wp.float32,
    max_strain: wp.float32,
    relaxation: wp.float32,
    carrier_bond: wp.int32,
) -> wp.float32:
    """One damped Newton step of an undriven column toward its local balance.

    This is the single implementation of the passive-surround balance. The
    identification sweeps it over a whole cycle at once (:func:`surround_sweep`)
    and the live runtime sweeps it every substep (:func:`surround_relax`), so a
    fitted shoe and a simulated shoe settle their untouched foam identically.

    The column carries its own unilateral ground reaction and the Pasternak shear
    ``pull`` from its neighbours, whose per-face coefficients come from
    :func:`_pasternak_coupling`. ``attachment`` is an optional vertical bond to
    the shoe above it and defaults to zero everywhere, because its reaction never
    reached the reported force or the carrier wrench: booked that way it was an
    undeclared rigid support, not a bond. ``rigid`` is the compression the carrier
    would impose on this column if it rode along rigidly:

    * ``carrier_bond == 0``: the column top is a free shoe surface the carrier
      never touches. Callers pass ``rigid = 0``, which is the bench-fixture
      surround of the identification.
    * ``carrier_bond != 0``: passive compression is bounded above by nonnegative
      nominal carrier penetration. This one-sided bound carries the free
      surround with the shoe; it does not glue each passive top to the rigid
      carrier or solve independent endpoint positions.

    The Maxwell overstress the column will carry once the step is taken is
    ``overstress_base + overstress_gain * p_eq(c)``, so a relaxation that moves
    ``c`` is balanced against the load it actually ends up under. Ignoring the
    gain understates the reaction tangent by ``1 + gain`` (about a factor of ten
    for this foam), which makes the Newton step overshoot and the compression
    ring at the substep rate. The identification holds the overstress fixed
    inside one pass and refreshes it between passes, so it passes ``gain = 0``.

    Args:
        c: Current column compression [m].
        rigid: Compression the rigid carrier imposes on this column [m].
        pull: Summed neighbour shear ``sum_j k_ij * (c_j - c)`` [N].
        coupling_sum: Summed face coefficients ``sum_j k_ij`` [N/m], the shear
            part of the local tangent.
        thickness: Column rest thickness [m].
        overstress_base: Maxwell overstress the column carries at zero
            equilibrium pressure [Pa].
        overstress_gain: Overstress produced per unit equilibrium pressure by the
            step about to be taken.
        params: Device-side constitutive constants.
        area: Tributary area of the column [m^2].
        attachment: Vertical bond stiffness to the shoe [N/m], normally zero.
        max_strain: Compression limit as a fraction of rest thickness.
        relaxation: Fraction of the Newton step taken, ``1`` for quasi-static.
        carrier_bond: Nonzero to bound passive compression by carrier penetration.

    Returns:
        The updated column compression [m].
    """
    return surround_balance(
        c,
        rigid,
        pull,
        coupling_sum,
        thickness,
        overstress_base,
        overstress_gain,
        params.g_eq,
        params.alpha,
        params.g_eq2,
        params.alpha2,
        params.beta,
        params.one_minus_two_poisson,
        params.stretch_floor,
        area,
        attachment,
        max_strain,
        relaxation,
        carrier_bond,
    )


@wp.func
def _surround_sweep_cell(
    frame: wp.int32,
    i: wp.int32,
    compression_in: wp.array2d[wp.float32],
    overstress: wp.array2d[wp.float32],
    driven: wp.array[wp.int32],
    neighbors: wp.array2d[wp.int32],
    slack: wp.array[wp.float32],
    params: FoundationParams,
    area: wp.float32,
    coupling_scale: wp.float32,
    attachment: wp.float32,
    max_strain: wp.float32,
    relaxation: wp.float32,
    compression_out: wp.array2d[wp.float32],
):
    """Apply the shared whole-cycle surround step to one frame and column."""
    if driven[i] != 0:
        compression_out[frame, i] = compression_in[frame, i]
        return
    c = compression_in[frame, i]
    pull = float(0.0)
    coupling_sum = float(0.0)
    for side in range(4):
        j = neighbors[i, side]
        if j >= 0:
            coupling = coupling_scale * _pasternak_coupling(slack[i], slack[j], params)
            pull += coupling * (compression_in[frame, j] - c)
            coupling_sum += coupling
    compression_out[frame, i] = _surround_balance(
        c,
        0.0,
        pull,
        coupling_sum,
        slack[i],
        overstress[frame, i],
        0.0,
        params,
        area,
        attachment,
        max_strain,
        relaxation,
        0,
    )


@wp.kernel
def surround_sweep(
    compression_in: wp.array2d[wp.float32],
    overstress: wp.array2d[wp.float32],
    driven: wp.array[wp.int32],
    neighbors: wp.array2d[wp.int32],
    slack: wp.array[wp.float32],
    params: FoundationParams,
    area: wp.float32,
    coupling_scale: wp.float32,
    attachment: wp.float32,
    max_strain: wp.float32,
    relaxation: wp.float32,
    compression_out: wp.array2d[wp.float32],
):
    """Relax one untouched column toward its own quasi-static balance.

    Each thread owns one frame and one column. Driven columns pass straight
    through, so the indenter keeps its imposed compression while the surrounding
    foam settles against neighbour shear and its unilateral ground reaction.
    Frames are independent, so a whole test relaxes at once.

    The support is the same equilibrium-plus-overstress reaction :func:`cycle_force`
    later sums, and the tangent differentiates that same sum. Balancing the
    equilibrium pressure alone would relax the surround against a support the
    loading then multiplies by ``1 + q / p_eq``, which is what drove the fit
    toward a short relaxation time. The overstress is held fixed inside one
    relaxation and refreshed by :func:`cycle_overstress` between passes.

    The bench fixture never touches this foam, so ``rigid = 0`` in
    :func:`_surround_balance`. ``relaxation`` is the fraction of the local Newton
    step taken: one is plain damped Jacobi, above one is successive
    over-relaxation, which reaches the same fixed point in fewer sweeps.
    """
    frame, i = wp.tid()
    _surround_sweep_cell(
        frame,
        i,
        compression_in,
        overstress,
        driven,
        neighbors,
        slack,
        params,
        area,
        coupling_scale,
        attachment,
        max_strain,
        relaxation,
        compression_out,
    )


@wp.kernel
def surround_relax(
    carrier: wp.array[wp.int32],
    column_count: wp.int32,
    body_q: wp.array[wp.transform],
    driven: wp.array[wp.int32],
    neighbors: wp.array2d[wp.int32],
    anchor_local: wp.array[wp.vec3],
    z_free_rigid: wp.array[wp.float32],
    rest_len: wp.array[wp.float32],
    area: wp.array[wp.float32],
    q_state: wp.array[wp.float32],
    peq_prev: wp.array[wp.float32],
    params: wp.array[FoundationParams],
    decay: wp.array[wp.float32],
    overstress_gain: wp.array[wp.float32],
    coupling_scale: wp.float32,
    attachment: wp.float32,
    max_strain: wp.float32,
    relaxation: wp.float32,
    carrier_bond: wp.int32,
    compression_in: wp.array[wp.float32],
    compression_out: wp.array[wp.float32],
):
    """Sweep the live column bed once toward the same balance the fit relaxes.

    Driven columns take the compression their carrier pose imposes, so the
    untouched foam reads the indenter through the shear layer exactly as
    :func:`surround_sweep` does during identification. The balance itself is
    :func:`_surround_balance`; only the source of the driven compression and the
    per-substep damping differ.

    ``decay`` and ``overstress_gain`` are the substep's Maxwell update, so the
    surround settles against the overstress :func:`foundation_pressure` is about
    to write for the compression this sweep produces, not against the previous
    substep's value. Both are per world, because a world's Maxwell update follows
    its own material.

    Launched over ``world_count * column_count`` threads with the same tiled
    state and shared, world-offset neighbour lookup as :func:`foundation_apply`.
    """
    i = wp.tid()
    world_index = i // column_count
    base = world_index * column_count
    column = i - base
    p = params[world_index]
    gain = overstress_gain[world_index]
    world = wp.transform_point(body_q[carrier[world_index]], anchor_local[column])
    rigid = z_free_rigid[column] - world[2]
    if driven[column] != 0:
        compression_out[i] = wp.max(rigid, 0.0)
        return
    c = compression_in[i]
    pull = float(0.0)
    coupling_sum = float(0.0)
    for side in range(4):
        j = neighbors[column, side]
        if j >= 0:
            coupling = coupling_scale * _pasternak_coupling(rest_len[column], rest_len[j], p)
            pull += coupling * (compression_in[base + j] - c)
            coupling_sum += coupling
    compression_out[i] = _surround_balance(
        c,
        rigid,
        pull,
        coupling_sum,
        rest_len[column],
        decay[world_index] * q_state[i] - gain * peq_prev[i],
        gain,
        p,
        area[column],
        attachment,
        max_strain,
        relaxation,
        carrier_bond,
    )


@wp.func
def _surround_write_free_top(
    i: int,
    carrier: wp.array[wp.int32],
    column_count: wp.int32,
    inv_dt: wp.float32,
    body_q: wp.array[wp.transform],
    driven: wp.array[wp.int32],
    anchor_local: wp.array[wp.vec3],
    z_free_rigid: wp.array[wp.float32],
    compression: wp.array[wp.float32],
    previous: wp.array[wp.float32],
    z_free: wp.array[wp.float32],
    rate: wp.array[wp.float32],
):
    """Publish the relaxed free surface the shared pressure kernel then consumes.

    :func:`foundation_pressure` reads ``compression = z_free - world_z``, so
    moving ``z_free`` to ``world_z + c`` hands it the relaxed compression without
    a second contact path. Driven columns keep their rigid free top.

    Launched over ``world_count * column_count`` threads; ``z_free`` is tiled
    per world because the relaxed free surface is state, not a constant.
    """
    world_index = i // column_count
    column = i - world_index * column_count
    if driven[column] != 0:
        z_free[i] = z_free_rigid[column]
        rate[i] = 0.0
        return
    world = wp.transform_point(body_q[carrier[world_index]], anchor_local[column])
    c = compression[i]
    z_free[i] = world[2] + c
    rate[i] = (c - previous[i]) * inv_dt
    previous[i] = c


@wp.kernel
def surround_write_free_top(
    carrier: wp.array[wp.int32],
    column_count: wp.int32,
    inv_dt: wp.float32,
    body_q: wp.array[wp.transform],
    driven: wp.array[wp.int32],
    anchor_local: wp.array[wp.vec3],
    z_free_rigid: wp.array[wp.float32],
    compression: wp.array[wp.float32],
    previous: wp.array[wp.float32],
    z_free: wp.array[wp.float32],
    rate: wp.array[wp.float32],
):
    """Run :func:`_surround_write_free_top` once per independent column."""
    _surround_write_free_top(
        wp.tid(),
        carrier,
        column_count,
        inv_dt,
        body_q,
        driven,
        anchor_local,
        z_free_rigid,
        compression,
        previous,
        z_free,
        rate,
    )


@wp.kernel
def surround_seed_driven(
    driven_compression: wp.array2d[wp.float32],
    driven_index: wp.array[wp.int32],
    compression: wp.array2d[wp.float32],
):
    """Stamp the indenter's imposed compression onto a warm-started field.

    :func:`surround_sweep` passes driven columns straight through, so a field
    reused from an earlier solve is only valid once its driven columns carry the
    compression of the current call.
    """
    frame, k = wp.tid()
    compression[frame, driven_index[k]] = driven_compression[frame, k]


@wp.kernel
def surround_update_max(
    a: wp.array2d[wp.float32],
    b: wp.array2d[wp.float32],
    out: wp.array[wp.float32],
):
    """Reduce the largest compression change between two consecutive sweeps [m]."""
    frame, i = wp.tid()
    wp.atomic_max(out, 0, wp.abs(a[frame, i] - b[frame, i]))


def relax_surround(
    driven_compression: np.ndarray,
    driven: np.ndarray,
    neighbors: np.ndarray,
    slack_m: np.ndarray,
    params: FoundationParams,
    *,
    area_m2: float,
    spacing_m: float,
    attachment_n_m: float,
    max_strain: float,
    sweeps: int,
    coupling_scale: float = 1.0,
    over_relaxation: float = 1.0,
    overstress: wp.array2d[wp.float32] | None = None,
    initial: wp.array2d[wp.float32] | None = None,
    tolerance_m: float = 0.0,
    check_every: int = 25,
    stats: dict[str, float] | None = None,
    device=None,
) -> np.ndarray:
    """Return whole-midsole compression for every frame of a test.

    The identification and the live runtime therefore share one contact model
    and one geometry: the indenter drives its columns and the rest relax.

    The sweep count is a cap, not a schedule. ``tolerance_m`` uses the legacy
    interval-ratio estimate ``u q / (1 - q)``, where ``u`` is a last-sweep update
    and ``q`` is the ratio between sampled updates. Because samples may be many
    sweeps apart, this is not a certified remaining-error bound. It is retained
    unchanged for execution compatibility. ``initial`` warms the compression
    field; callers must qualify numerical convergence separately.

    Args:
        driven_compression: Imposed compression of the driven columns [m],
            shape ``[frames, driven_count]``.
        driven: Nonzero for columns the indenter drives, shape ``[column_count]``.
        neighbors: Four in-plane neighbour indices; negative is a free edge.
        slack_m: Rest thickness per column [m].
        params: Device-side constitutive constants.
        area_m2: Tributary area per column [m^2].
        spacing_m: Column grid spacing [m].
        attachment_n_m: Vertical bond of untouched foam to the shoe [N/m];
            zero is the booked-consistently default.
        max_strain: Compression limit as a fraction of rest thickness.
        sweeps: Hard cap on relaxation sweeps.
        coupling_scale: Fraction of the material-pinned face coefficient
            ``mu_eq * t`` that drives the passive surface.
        over_relaxation: Fraction of the local Newton step taken per sweep. One
            is plain damped Jacobi; above one it is successive over-relaxation.
        overstress: Maxwell overstress carried by every column and frame [Pa],
            shape ``[frames, column_count]``, from :func:`cycle_overstress`.
            ``None`` relaxes against the equilibrium pressure alone, which is
            only the first pass of a self-consistent solve.
        initial: Warm-start compression [m], shape ``[frames, column_count]``,
            normally the previous converged solve. Its driven columns are
            overwritten with ``driven_compression``.
        tolerance_m: Extrapolated remaining compression travel that ends the
            solve [m]. Zero always runs the full ``sweeps`` cap.
        check_every: Sweeps between convergence tests. Each test costs one
            reduction and one device synchronization.
        stats: Optional mapping that receives ``sweeps``, ``update_m`` and
            ``remaining_m`` so the convergence stays visible to the caller.

    Returns:
        Device compression for every column and frame [m], shape
        ``[frames, column_count]``, ready for :func:`cycle_force`.
    """
    driven = np.ascontiguousarray(driven, np.int32)
    frames = len(driven_compression)
    count = len(slack_m)
    driven_index = np.ascontiguousarray(np.flatnonzero(driven != 0), np.int32)
    if initial is None:
        current = wp.zeros((frames, count), dtype=wp.float32, device=device)
    else:
        current = wp.clone(initial)
    device = current.device
    wp.launch(
        surround_seed_driven,
        dim=(frames, len(driven_index)),
        inputs=[
            wp.array(np.ascontiguousarray(driven_compression, np.float32), dtype=wp.float32, device=device),
            wp.array(driven_index, dtype=wp.int32, device=device),
            current,
        ],
        device=device,
    )
    scratch = wp.zeros_like(current)
    if overstress is None:
        overstress = wp.zeros((frames, count), dtype=wp.float32, device=device)
    driven_device = wp.array(driven, dtype=wp.int32, device=device)
    neighbor_device = wp.array(np.ascontiguousarray(neighbors, np.int32), dtype=wp.int32, device=device)
    slack_device = wp.array(np.ascontiguousarray(slack_m, np.float32), dtype=wp.float32, device=device)
    update = wp.zeros(1, dtype=wp.float32, device=device)
    interval = max(int(check_every), 1)
    previous_update = float("inf")
    used = 0
    change = float("nan")
    remaining = float("nan")
    for sweep in range(max(sweeps, 0)):
        wp.launch(
            surround_sweep,
            dim=(frames, count),
            inputs=[
                current,
                overstress,
                driven_device,
                neighbor_device,
                slack_device,
                params,
                float(area_m2),
                float(coupling_scale),
                float(attachment_n_m),
                float(max_strain),
                float(over_relaxation),
                scratch,
            ],
            device=device,
        )
        used = sweep + 1
        if tolerance_m > 0.0 and used % interval == 0:
            update.zero_()
            wp.launch(surround_update_max, dim=(frames, count), inputs=[current, scratch, update], device=device)
            change = float(update.numpy()[0])
            # The first interval has nothing to measure a decay against, and an
            # iteration that is not contracting has no finite extrapolation, so
            # both keep the solve running rather than declaring success.
            decay = change / previous_update if 0.0 < previous_update < float("inf") else float("inf")
            previous_update = change
            if change == 0.0:
                remaining = 0.0
            elif decay >= 1.0:
                remaining = float("inf")
            else:
                remaining = change * decay / (1.0 - decay)
            current, scratch = scratch, current
            if remaining < tolerance_m:
                break
            continue
        current, scratch = scratch, current
    if stats is not None:
        stats.update({"sweeps": float(used), "update_m": change, "remaining_m": remaining})
    return current


# Groups per world in the deterministic reduction. Group ``g`` walks its world's tile
# with stride ``group_count``, so neighbouring lanes read neighbouring columns, and the
# final pass folds ``group_count`` partials per world. The count trades the width of the
# first pass against the length of the second: measured on an A6000 with the 910-column
# bed, the captured substep costs 2.16 / 1.34 / 1.53 / 1.85 ns per column at 4 / 16 / 32
# / 64 groups (64 worlds), so sixteen is the floor and the curve is flat around it.
FOUNDATION_REDUCTION_GROUPS = 16


@wp.func
def _foundation_partial(
    tid: wp.int32,
    carrier: wp.array[wp.int32],
    column_count: wp.int32,
    group_count: wp.int32,
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    body_com: wp.array[wp.vec3],
    anchor_local: wp.array[wp.vec3],
    compression: wp.array[wp.float32],
    column_force: wp.array[wp.vec3],
    column_pressed: wp.array[wp.float32],
    part_force: wp.array[wp.vec3],
    part_torque: wp.array[wp.vec3],
    part_moment: wp.array[wp.vec3],
    part_cop: wp.array[wp.vec3],
    part_normal: wp.array[wp.float32],
    part_pressed: wp.array[wp.float32],
    part_power: wp.array[wp.float32],
    part_max: wp.array[wp.float32],
    part_active: wp.array[wp.int32],
    ground_plane: wp.int32,
    contact_point: wp.array[wp.vec3],
) -> None:
    """Reduce one fixed-order group using either nominal anchors or external plane points."""
    world_index = tid // group_count
    group = tid - world_index * group_count
    base = world_index * column_count
    body = carrier[world_index]
    q_body = body_q[body]
    com_world = wp.transform_point(q_body, body_com[body])
    vel = body_qd[body]

    force_sum = wp.vec3(0.0, 0.0, 0.0)
    torque_sum = wp.vec3(0.0, 0.0, 0.0)
    moment_sum = wp.vec3(0.0, 0.0, 0.0)
    cop_sum = wp.vec3(0.0, 0.0, 0.0)
    normal_sum = float(0.0)
    pressed_sum = float(0.0)
    power_sum = float(0.0)
    max_comp = float(0.0)
    active = int(0)
    for column in range(group, column_count, group_count):
        i = base + column
        force = column_force[i]
        pressed = column_pressed[i]
        world = wp.vec3(0.0, 0.0, 0.0)
        if ground_plane != 0:
            world = contact_point[i]
            pressed = force[2]
        else:
            world = wp.transform_point(q_body, anchor_local[column])
        torque, moment, power = contact_wrench(world, force, com_world, vel)
        force_sum += force
        torque_sum += torque
        moment_sum += moment
        cop_sum += wp.vec3(world[0] * pressed, world[1] * pressed, 0.0)
        normal_sum += force[2]
        pressed_sum += pressed
        power_sum += power
        ci = compression[i]
        max_comp = wp.max(max_comp, ci)
        if ci > 0.0:
            active += 1
    part_force[tid] = force_sum
    part_torque[tid] = torque_sum
    part_moment[tid] = moment_sum
    part_cop[tid] = cop_sum
    part_normal[tid] = normal_sum
    part_pressed[tid] = pressed_sum
    part_power[tid] = power_sum
    part_max[tid] = max_comp
    part_active[tid] = active


@wp.kernel
def foundation_partial(
    carrier: wp.array[wp.int32],
    column_count: wp.int32,
    group_count: wp.int32,
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    body_com: wp.array[wp.vec3],
    anchor_local: wp.array[wp.vec3],
    compression: wp.array[wp.float32],
    column_force: wp.array[wp.vec3],
    column_pressed: wp.array[wp.float32],
    part_force: wp.array[wp.vec3],
    part_torque: wp.array[wp.vec3],
    part_moment: wp.array[wp.vec3],
    part_cop: wp.array[wp.vec3],
    part_normal: wp.array[wp.float32],
    part_pressed: wp.array[wp.float32],
    part_power: wp.array[wp.float32],
    part_max: wp.array[wp.float32],
    part_active: wp.array[wp.int32],
):
    """Sum legacy transfer tractions at nominal anchors in a deterministic fixed order.

    Group ``g`` owns columns ``g, g + group_count, ...`` within one world.
    The exported signature and default reduction convention stay unchanged.
    """
    _foundation_partial(
        wp.tid(),
        carrier,
        column_count,
        group_count,
        body_q,
        body_qd,
        body_com,
        anchor_local,
        compression,
        column_force,
        column_pressed,
        part_force,
        part_torque,
        part_moment,
        part_cop,
        part_normal,
        part_pressed,
        part_power,
        part_max,
        part_active,
        0,
        anchor_local,
    )


@wp.kernel
def foundation_partial_ground(
    carrier: wp.array[wp.int32],
    column_count: wp.int32,
    group_count: wp.int32,
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    body_com: wp.array[wp.vec3],
    contact_point: wp.array[wp.vec3],
    compression: wp.array[wp.float32],
    ground_force: wp.array[wp.vec3],
    column_pressed: wp.array[wp.float32],
    part_force: wp.array[wp.vec3],
    part_torque: wp.array[wp.vec3],
    part_moment: wp.array[wp.vec3],
    part_cop: wp.array[wp.vec3],
    part_normal: wp.array[wp.float32],
    part_pressed: wp.array[wp.float32],
    part_power: wp.array[wp.float32],
    part_max: wp.array[wp.float32],
    part_active: wp.array[wp.int32],
):
    """Transfer the complete external ground wrench through the massless attached shoe.

    Internal Pasternak forces cancel in total force, but their separated vertical
    tractions alone do not cancel in moment. Reducing external forces at their
    ground points includes the equivalent transfer couple without inventing a
    separate support. Friction, carrier torque and rigid-wrench power use this
    same contact-height lever arm.
    """
    _foundation_partial(
        wp.tid(),
        carrier,
        column_count,
        group_count,
        body_q,
        body_qd,
        body_com,
        contact_point,
        compression,
        ground_force,
        column_pressed,
        part_force,
        part_torque,
        part_moment,
        part_cop,
        part_normal,
        part_pressed,
        part_power,
        part_max,
        part_active,
        1,
        contact_point,
    )


@wp.func
def _foundation_finalize(
    world_index: int,
    carrier: wp.array[wp.int32],
    group_count: wp.int32,
    part_force: wp.array[wp.vec3],
    part_torque: wp.array[wp.vec3],
    part_moment: wp.array[wp.vec3],
    part_cop: wp.array[wp.vec3],
    part_normal: wp.array[wp.float32],
    part_pressed: wp.array[wp.float32],
    part_power: wp.array[wp.float32],
    part_max: wp.array[wp.float32],
    part_active: wp.array[wp.int32],
    body_f: wp.array[wp.spatial_vector],
    normal_force: wp.array[wp.float32],
    cop_moment: wp.array[wp.vec3],
    active_count: wp.array[wp.int32],
    resultant_force: wp.array[wp.vec3],
    resultant_moment_origin: wp.array[wp.vec3],
    contact_power: wp.array[wp.float32],
    max_compression: wp.array[wp.float32],
    pressed_force: wp.array[wp.float32],
):
    """Fold one world's partials into its diagnostics and its carrier wrench.

    One thread per world sums the groups in index order and *writes* the result, so the
    accumulators need no reset pass and carry no atomic history. Only the carrier wrench
    is added, because other forces may already be staged in ``body_f``.
    """
    base = world_index * group_count
    force_sum = wp.vec3(0.0, 0.0, 0.0)
    torque_sum = wp.vec3(0.0, 0.0, 0.0)
    moment_sum = wp.vec3(0.0, 0.0, 0.0)
    cop_sum = wp.vec3(0.0, 0.0, 0.0)
    normal_sum = float(0.0)
    pressed_sum = float(0.0)
    power_sum = float(0.0)
    max_comp = float(0.0)
    active = int(0)
    for group in range(group_count):
        k = base + group
        force_sum += part_force[k]
        torque_sum += part_torque[k]
        moment_sum += part_moment[k]
        cop_sum += part_cop[k]
        normal_sum += part_normal[k]
        pressed_sum += part_pressed[k]
        power_sum += part_power[k]
        max_comp = wp.max(max_comp, part_max[k])
        active += part_active[k]
    normal_force[world_index] = normal_sum
    pressed_force[world_index] = pressed_sum
    cop_moment[world_index] = cop_sum
    resultant_force[world_index] = force_sum
    resultant_moment_origin[world_index] = moment_sum
    contact_power[world_index] = power_sum
    max_compression[world_index] = max_comp
    active_count[world_index] = active
    wp.atomic_add(body_f, carrier[world_index], wp.spatial_vector(force_sum, torque_sum))


@wp.kernel
def foundation_finalize(
    carrier: wp.array[wp.int32],
    group_count: wp.int32,
    part_force: wp.array[wp.vec3],
    part_torque: wp.array[wp.vec3],
    part_moment: wp.array[wp.vec3],
    part_cop: wp.array[wp.vec3],
    part_normal: wp.array[wp.float32],
    part_pressed: wp.array[wp.float32],
    part_power: wp.array[wp.float32],
    part_max: wp.array[wp.float32],
    part_active: wp.array[wp.int32],
    body_f: wp.array[wp.spatial_vector],
    normal_force: wp.array[wp.float32],
    cop_moment: wp.array[wp.vec3],
    active_count: wp.array[wp.int32],
    resultant_force: wp.array[wp.vec3],
    resultant_moment_origin: wp.array[wp.vec3],
    contact_power: wp.array[wp.float32],
    max_compression: wp.array[wp.float32],
    pressed_force: wp.array[wp.float32],
):
    """Run :func:`_foundation_finalize` once per independent world."""
    _foundation_finalize(
        wp.tid(),
        carrier,
        group_count,
        part_force,
        part_torque,
        part_moment,
        part_cop,
        part_normal,
        part_pressed,
        part_power,
        part_max,
        part_active,
        body_f,
        normal_force,
        cop_moment,
        active_count,
        resultant_force,
        resultant_moment_origin,
        contact_power,
        max_compression,
        pressed_force,
    )


@wp.kernel
def foundation_reset(
    carrier: wp.array[wp.int32],
    clear_body_force: wp.int32,
    body_f: wp.array[wp.spatial_vector],
    normal_force: wp.array[wp.float32],
    cop_moment: wp.array[wp.vec3],
    active_count: wp.array[wp.int32],
    resultant_force: wp.array[wp.vec3],
    resultant_moment_origin: wp.array[wp.vec3],
    contact_power: wp.array[wp.float32],
    max_compression: wp.array[wp.float32],
    pressed_force: wp.array[wp.float32],
):
    """Zero the per-substep foundation accumulators (and optionally the carrier wrench).

    :func:`foundation_finalize` *writes* every accumulator, so the zeroing is only
    kept for callers that reach for the arrays before the first substep.
    :meth:`MidsoleFoundation.apply` therefore launches this kernel only when
    ``clear_body_force`` asks it to zero the carrier wrench as well, which saves a
    graph node per substep.

    Launched over ``world_count`` threads, one per world.
    """
    w = wp.tid()
    normal_force[w] = 0.0
    cop_moment[w] = wp.vec3(0.0, 0.0, 0.0)
    active_count[w] = 0
    resultant_force[w] = wp.vec3(0.0, 0.0, 0.0)
    resultant_moment_origin[w] = wp.vec3(0.0, 0.0, 0.0)
    contact_power[w] = 0.0
    max_compression[w] = 0.0
    pressed_force[w] = 0.0
    if clear_body_force != 0:
        body_f[carrier[w]] = wp.spatial_vector(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


# ---------------------------------------------------------------------------
# Foundation driver
# ---------------------------------------------------------------------------
@dataclass
class FoundationConfig:
    """Tunable dynamic parameters layered on the calibrated constitutive law.

    The Instron replay leaves ``normal_damping``, ``friction_stiffness`` and
    ``friction`` at zero and keeps ``stretch_floor`` below the calibration's peak
    strain so the collected loop reproduces the fitted force-displacement response
    exactly. The default friction model is an area-scaled elastic Coulomb bristle.
    ``friction_model="elastic_coulomb"`` derives each column's stiffness as
    ``G_eq * area / rest_length`` and applies no tangential damping. It is a
    geometry-scaled contact assumption; normal compression does not identify
    outsole friction. Maxwell shear remains available explicitly: an equilibrium
    spring in parallel with a spring/dashpot branch, in series with a Coulomb slider.
    ``friction_stiffness`` is the equilibrium tangential stiffness, ``friction``
    is the internal dashpot viscosity, and ``mu`` bounds traction at ``mu * fn``.
    Select ``friction_model="legacy"`` for the previous anchored spring with direct
    velocity damping. No public kernel or legacy symbol is removed.

    For configured-coefficient models, the tangential bed is a property of the
    contact area, not of the sampling grid, so the per-column values are
    converted to ``friction_stiffness_per_area``
    ``k'' = kt / A`` [N/m^3] and ``friction_damping_per_area`` ``c'' = kv / A``
    [N.s/m^3] and re-expanded as ``kt_i = k'' * A_i``. Set the per-area fields directly
    to declare the tangential layer independently of the column count; leaving them at
    zero derives them from the per-column values and the mean tributary area, which is
    exactly the previous behaviour on a uniform grid. The legacy shoe settings,
    kt = 1e4 N/m over 25 mm^2, correspond to ``k'' = 4.0e8 N/m^3``, i.e. a thin outsole
    rubber layer (G ~ 0.8 MPa over 2 mm). That is a declared assumption: it is 2-3
    orders of magnitude above the shear stiffness of the identified foam, and the
    Instron identification is compression only.

    ``friction_model="maxwell"`` uses the configured tangential stiffness and
    viscosity with a Maxwell branch. ``friction_model="column_maxwell"`` derives
    each column's shear stiffness and viscosity from the material modulus, column
    area, rest length, and relaxation time. That law extrapolates the fitted normal
    material into large shear strain; independent outsole shear calibration is not
    available. ``friction_viscous_ratio`` applies only to legacy direct velocity
    damping. Maxwell models use ``friction_relaxation_time_s``, or the material
    relaxation time when omitted.
    ``friction_release_dwell_s`` preserves contact history through short normal dropouts.

    ``ground_height_m`` opts into a horizontal external contact plane [m]. In this
    mode ``anchor_local`` denotes nominal bottom points of a massless attached
    shoe and the initial ``z_free`` is the plane height. Friction capacity is
    ``mu * R`` for the unilateral ground reaction ``R``, not signed load transfer.
    Ground forces act at the nominal bottom's world XY projected onto the plane;
    their complete wrench reaches the carrier. A separated nominal bottom carries
    no ground force, even when the Maxwell history is nonzero. The constitutive
    and passive-relaxation states still follow the same material update. All
    initial ``z_free`` values must equal the plane height at device precision.
    A surround with passive columns requires ``carrier_bond=True`` so the
    relaxation cannot load nominal bottoms that are separated from the plane.

    ``None`` retains the generic bench convention: ``anchor_local`` can denote
    carrier/indenter top points, ``z_free`` is the uncompressed foam top, and
    transfer tractions are reduced at nominal anchors. Do not infer a ground
    plane from ``z_free`` in that convention.
    """

    stretch_floor: float = 0.05
    normal_damping: float = 0.0
    friction_stiffness: float = 0.0
    friction: float = 0.0
    friction_stiffness_per_area: float = 0.0
    friction_damping_per_area: float = 0.0
    friction_viscous_ratio: float = 0.2
    friction_release_dwell_s: float = 0.0005
    mu: float = 0.0
    ground_height_m: float | None = None
    friction_model: str = "elastic_coulomb"
    friction_relaxation_time_s: float | None = None

    def __post_init__(self) -> None:
        if self.friction_model not in ("elastic_coulomb", "maxwell", "column_maxwell", "legacy"):
            raise ValueError(
                f"friction_model must be 'elastic_coulomb', 'maxwell', 'column_maxwell', or 'legacy', got {self.friction_model!r}"
            )
        if self.friction_relaxation_time_s is not None and (
            not np.isfinite(self.friction_relaxation_time_s) or self.friction_relaxation_time_s <= 0.0
        ):
            raise ValueError("friction_relaxation_time_s must be positive and finite if specified")
        if self.ground_height_m is not None and (
            not np.isfinite(self.ground_height_m) or abs(self.ground_height_m) > float(np.finfo(np.float32).max)
        ):
            raise ValueError("ground_height_m must be finite in float32 or None")
        negative = (
            self.friction_stiffness < 0.0
            or self.friction < 0.0
            or self.friction_stiffness_per_area < 0.0
            or self.friction_damping_per_area < 0.0
            or self.friction_viscous_ratio < 0.0
            or self.friction_release_dwell_s < 0.0
        )
        if negative:
            raise ValueError("friction stiffness, damping, viscous ratio and release dwell must be nonnegative")


@dataclass
class SurroundConfig:
    """Assumed relaxation settings for the columns the carrier does not drive.

    The identification relaxes the foam outside the indenter against neighbour
    shear and its own unilateral ground reaction (:func:`surround_sweep`). This
    config gives the live runtime the same surround, so one geometry and one
    contact model serve both.

    Args:
        driven: Nonzero where the carrier drives the column, shape
            ``[column_count]``. An all-driven mask makes the relaxation a no-op.
        attachment_n_m: Vertical bond of untouched foam to the shoe [N/m].
            Zero by default: its reaction never entered the reported force or
            the carrier wrench, so any nonzero value is an undeclared support.
        max_strain: Compression limit as a fraction of rest thickness.
        coupling_scale: Assumed lateral drive on the passive surface as a
            fraction of the material-pinned face coefficient ``mu_eq * t``.
        sweeps: Relaxation sweeps per substep. The field is warm started from the
            previous substep, so few sweeps track a converged quasi-static solve.
        relaxation_time_s: First-order lag toward the local balance [s]. Zero
            takes the full Newton step, which is the quasi-static solve the
            identification uses.
        carrier_bond: Bound passive compression between zero and nonnegative
            nominal carrier penetration. The free surround is carried laterally
            by the shoe, not glued at each top to the rigid last. This scalar
            constraint does not solve independent endpoint geometry or a
            separate flight shape-recovery mode. False keeps the bench's free
            surround. See :func:`_surround_balance`.
    """

    driven: np.ndarray
    attachment_n_m: float = 0.0
    max_strain: float = 0.9
    coupling_scale: float = 1.0
    sweeps: int = 8
    relaxation_time_s: float = 0.0
    carrier_bond: bool = False

    def __post_init__(self) -> None:
        self.driven = np.ascontiguousarray(np.asarray(self.driven) != 0, np.int32)
        if self.driven.ndim != 1 or not np.any(self.driven):
            raise ValueError("the surround needs a one-dimensional mask with at least one driven column")
        if self.attachment_n_m < 0.0 or not 0.0 < self.max_strain < 1.0:
            raise ValueError("surround bond stiffness must be nonnegative and its strain limit inside (0, 1)")
        if self.coupling_scale < 0.0 or self.sweeps < 1 or self.relaxation_time_s < 0.0:
            raise ValueError("surround coupling, sweeps, and relaxation time must be nonnegative and sweeps positive")


class MidsoleFoundation:
    """Live Warp elastic-foundation force model attached to one carrier body per world.

    One instance drives ``world_count`` independent copies of the same column bed
    in a single launch, each with its own carrier body and its own material, so a
    population of controller candidates or a domain-randomized batch costs about
    what one shoe costs: the bed is only ~1000 columns, which leaves a modern GPU
    idle at one world per launch.

    **Array layout.** Per-column *state* is tiled: world ``w`` owns
    ``[w * column_count : (w + 1) * column_count]`` of
    :attr:`compression`, :attr:`base_pressure`, :attr:`column_force`,
    :attr:`column_pressed`, :attr:`ground_force`, :attr:`contact_point`,
    :attr:`z_free`, ``q_state``, ``peq_prev``, :attr:`tangent_anchor`,
    :attr:`tangent_stuck`, ``tangent_dwell`` and the surround fields. Per-column
    *constants* are **shared**, i.e. kept at length ``column_count`` and indexed
    by the column alone: :attr:`anchor_local`, :attr:`area`, :attr:`rest_len`,
    ``z_free_rigid``, :attr:`neighbors`, ``friction_kt``, ``friction_kv`` and
    ``driven``. ``neighbors`` therefore stores plain column indices and every
    kernel offsets them by its own world's tile base, which makes a cross-world
    Pasternak coupling structurally impossible. The reductions
    (:attr:`normal_force`, :attr:`cop_moment`, :attr:`active`,
    :attr:`resultant_force`, :attr:`resultant_moment_origin`,
    :attr:`contact_power`, :attr:`max_compression`, :attr:`pressed_force`) have
    length ``world_count`` and are indexed by world. The constitutive constants
    live in :attr:`world_params`, one :class:`FoundationParams` block per world;
    :meth:`set_world_material` rewrites one world's block.

    ``column_force`` is signed load transfer [N], with vertical component
    ``R - neighbor_flux``. ``column_pressed`` remains its positive vertical part.
    Neither is local ground pressure. With ``config.ground_height_m`` set,
    ``ground_force`` holds external friction and nonnegative ``R`` [N], and
    ``contact_point`` holds the corresponding projected plane point [m], including
    potential points for unloaded columns. These two arrays stay zero in generic
    mode. Plane-mode resultants, COP and ``pressed_force`` use external ground
    reactions, not the transfer diagnostics. ``contact_power`` is the carrier's
    rigid-wrench power [W], using its velocity field at those same points; it
    does not include independent foam deformation or stored bristle-energy rates.

    With default ground settings and ``world_count = 1`` the existing arrays keep
    their old lengths and arithmetic, reproducing the single-world generic runtime.

    Args:
        anchor_local: Column attachment points in the carrier body frame [m],
            shape ``[column_count, 3]``. Use nominal bottom points when
            ``config.ground_height_m`` is set.
        z_free: World height of each uncompressed foam column top in generic
            mode, or the ground-plane height for bottom-anchor mode [m], shape
            ``[column_count]``; tiled over the worlds internally. Passive columns
            publish effective pressure references here, not geometric endpoints.
        rest_len: Column rest thickness [m], shape ``[column_count]``.
        area: Tributary area per column [m^2], shape ``[column_count]``.
        neighbors: Pasternak 4-neighbour indices, shape ``[column_count, 4]``.
        spacing_m: Column grid spacing [m].
        material: Calibrated :class:`ShoeMaterial`, given to every world until
            :meth:`set_world_material` changes one.
        carrier_body: Index of the rigid body carrying the foundation, or one
            body index per world. A bare index is only allowed when
            ``world_count`` is one, because two worlds sharing a carrier would
            sum their wrenches into the same body.
        body_com: Model center-of-mass array (``model.body_com``).
        config: Dynamic :class:`FoundationConfig`.
        device: Warp device.
        surround: Optional :class:`SurroundConfig` letting the columns the
            carrier does not drive relax passively every substep, exactly as the
            identification relaxes them.
        world_count: Number of independent copies of the bed evaluated in one
            launch.
    """

    def __init__(
        self,
        anchor_local: np.ndarray,
        z_free: np.ndarray,
        rest_len: np.ndarray,
        area: np.ndarray,
        neighbors: np.ndarray,
        spacing_m: float,
        material: ShoeMaterial,
        carrier_body: int | Sequence[int],
        body_com,
        config: FoundationConfig | None = None,
        device=None,
        surround: SurroundConfig | None = None,
        world_count: int = 1,
    ) -> None:
        config = config or FoundationConfig()
        self.ground_height_m = config.ground_height_m
        if self.ground_height_m is not None:
            if not np.all(np.asarray(z_free, dtype=np.float32) == np.float32(self.ground_height_m)):
                raise ValueError("initial z_free must equal ground_height_m in ground-plane mode")
            if surround is not None and not surround.carrier_bond and np.any(surround.driven == 0):
                raise ValueError("ground-plane passive columns require surround.carrier_bond=True")
        self.device = device
        self.world_count = int(world_count)
        if self.world_count < 1:
            raise ValueError("world_count must be at least one")
        carriers = np.atleast_1d(np.ascontiguousarray(carrier_body, np.int32)).reshape(-1)
        if len(carriers) != self.world_count:
            raise ValueError("carrier_body must give one body index per world")
        if len(np.unique(carriers)) != len(carriers):
            raise ValueError("each world needs its own carrier body")
        self.carrier_bodies = carriers
        self.carrier = wp.array(carriers, dtype=wp.int32, device=device)
        self.body_com = body_com
        self.column_count = int(len(rest_len))
        self.friction_solver = None

        if config.friction_model in ("column_maxwell", "elastic_coulomb"):
            column_area = np.asarray(area, dtype=np.float64)
            column_length = np.asarray(rest_len, dtype=np.float64)
            if (
                column_area.shape != (self.column_count,)
                or column_length.shape != (self.column_count,)
                or not np.isfinite(column_area).all()
                or not np.isfinite(column_length).all()
                or np.any(column_area <= 0.0)
                or np.any(column_length <= 0.0)
            ):
                raise ValueError(
                    f"{config.friction_model} requires positive finite area and rest length for every column"
                )

        params = FoundationParams()
        set_material_block(params, material)
        params.inv_h2 = 1.0 / spacing_m**2
        params.stretch_floor = config.stretch_floor
        params.normal_damping = config.normal_damping
        params.friction_kt = config.friction_stiffness
        params.friction_kv = config.friction
        params.friction_viscous_ratio = config.friction_viscous_ratio
        params.friction_release_dwell_s = config.friction_release_dwell_s
        params.mu = config.mu
        params.friction_model = {"legacy": 0, "maxwell": 1, "column_maxwell": 2, "elastic_coulomb": 3}[
            config.friction_model
        ]
        params.friction_relaxation_time_s = float(
            config.friction_relaxation_time_s
            if config.friction_relaxation_time_s is not None
            else getattr(material, "maxwell_relaxation_time_s", 0.08)
        )
        self.params = params
        self.config = config

        m = self.column_count
        n = self.world_count * m  # tiled per-column state, world w at [w * m : (w + 1) * m]
        # Tangential layer per unit area, so refining the column grid keeps the same patch
        # stiffness. Deriving from the mean area reproduces the per-column setting exactly
        # on a uniform grid.
        area_m2 = np.ascontiguousarray(area, np.float64).reshape(-1)
        mean_area = float(area_m2.mean())
        if mean_area <= 0.0:
            raise ValueError("column tributary areas must be positive")
        self.friction_stiffness_per_area_n_m3 = float(
            config.friction_stiffness_per_area or config.friction_stiffness / mean_area
        )
        self.friction_damping_per_area_n_s_m3 = float(config.friction_damping_per_area or config.friction / mean_area)
        self.friction_viscous_ratio = float(config.friction_viscous_ratio)
        self.friction_release_dwell_s = float(config.friction_release_dwell_s)
        self.anchor_local = wp.array(np.ascontiguousarray(anchor_local, np.float32), dtype=wp.vec3, device=device)
        # The relaxed free surface is state, not a constant, so it is tiled per world.
        self.z_free = wp.array(
            np.tile(np.ascontiguousarray(z_free, np.float32).reshape(-1), self.world_count),
            dtype=wp.float32,
            device=device,
        )
        self.rest_len = wp.array(np.ascontiguousarray(rest_len, np.float32), dtype=wp.float32, device=device)
        self.area = wp.array(np.ascontiguousarray(area, np.float32), dtype=wp.float32, device=device)
        self.neighbors = wp.array(np.ascontiguousarray(neighbors, np.int32), dtype=wp.int32, device=device)
        self.q_state = wp.zeros(n, dtype=wp.float32, device=device)
        self.peq_prev = wp.zeros(n, dtype=wp.float32, device=device)
        self.compression = wp.zeros(n, dtype=wp.float32, device=device)
        self.base_pressure = wp.zeros(n, dtype=wp.float32, device=device)
        self.tangent_anchor = wp.zeros(n, dtype=wp.vec2, device=device)  # world XY stick point
        self.tangent_stuck = wp.zeros(n, dtype=wp.int32, device=device)  # 1 while the bristle grips
        self.tangent_dwell = wp.zeros(n, dtype=wp.float32, device=device)  # unloaded time held by a grip [s]
        self.tangent_deflection = wp.zeros(n, dtype=wp.vec2, device=device)
        self.tangent_maxwell_force = wp.zeros(n, dtype=wp.vec2, device=device)
        self.friction_kt = wp.array(
            np.ascontiguousarray(self.friction_stiffness_per_area_n_m3 * area_m2, np.float32),
            dtype=wp.float32,
            device=device,
        )
        self.friction_kv = wp.array(
            np.ascontiguousarray(self.friction_damping_per_area_n_s_m3 * area_m2, np.float32),
            dtype=wp.float32,
            device=device,
        )
        w = self.world_count
        self.normal_force = wp.zeros(w, dtype=wp.float32, device=device)
        self.cop_moment = wp.zeros(w, dtype=wp.vec3, device=device)
        self.active = wp.zeros(w, dtype=wp.int32, device=device)
        self.resultant_force = wp.zeros(w, dtype=wp.vec3, device=device)
        self.resultant_moment_origin = wp.zeros(w, dtype=wp.vec3, device=device)
        self.contact_power = wp.zeros(w, dtype=wp.float32, device=device)
        self.max_compression = wp.zeros(w, dtype=wp.float32, device=device)
        # Signed transfer traction [N]: (ground friction x/y, R - neighbor flux).
        # Its normal component is not a local ground reaction and must not be clamped.
        self.column_force = wp.zeros(n, dtype=wp.vec3, device=device)
        # Positive transfer traction [N], retained for legacy consumers in both modes.
        self.column_pressed = wp.zeros(n, dtype=wp.float32, device=device)
        # External traction [N] and world plane points [m], populated only in plane mode.
        self.ground_force = wp.zeros(n, dtype=wp.vec3, device=device)
        self.contact_point = wp.zeros(n, dtype=wp.vec3, device=device)
        self.pressed_force = wp.zeros(w, dtype=wp.float32, device=device)
        # Deterministic two-pass reduction of the bed. Groups partition each world's
        # columns by stride, so the partial sums are over fixed sets in a fixed order.
        self.reduction_groups = int(min(FOUNDATION_REDUCTION_GROUPS, max(1, m)))
        partials = w * self.reduction_groups
        self.partial_force = wp.zeros(partials, dtype=wp.vec3, device=device)
        self.partial_torque = wp.zeros(partials, dtype=wp.vec3, device=device)
        self.partial_moment = wp.zeros(partials, dtype=wp.vec3, device=device)
        self.partial_cop = wp.zeros(partials, dtype=wp.vec3, device=device)
        self.partial_normal = wp.zeros(partials, dtype=wp.float32, device=device)
        self.partial_pressed = wp.zeros(partials, dtype=wp.float32, device=device)
        self.partial_power = wp.zeros(partials, dtype=wp.float32, device=device)
        self.partial_max = wp.zeros(partials, dtype=wp.float32, device=device)
        self.partial_active = wp.zeros(partials, dtype=wp.int32, device=device)
        # One constitutive block per world. Domain randomization rewrites these and
        # nothing else, so a randomized batch shares one geometry and one contact law.
        self.world_blocks = [clone_params(params) for _ in range(w)]
        self.world_params = wp.array(self.world_blocks, dtype=FoundationParams, device=device)
        # Per-world Maxwell update of the surround sweep, refreshed on the host whenever
        # the substep or a material changes. See :meth:`relax_surround`.
        self.surround_decay = wp.zeros(w, dtype=wp.float32, device=device)
        self.surround_gain = wp.zeros(w, dtype=wp.float32, device=device)
        self._surround_dt_s = 0.0
        self._materials_dirty = True

        self.surround = surround
        self.free_column_count = 0
        self.driven = wp.full(m, 1, dtype=wp.int32, device=device)
        if surround is not None:
            if len(surround.driven) != m:
                raise ValueError("the surround mask must cover every column")
            self.free_column_count = int(m - int(surround.driven.sum()))
            # The rigid free top the carrier imposes. surround_write_free_top overwrites
            # z_free for the undriven columns every substep, so the rigid heights need
            # their own copy.
            self.z_free_rigid = wp.array(np.ascontiguousarray(z_free, np.float32), dtype=wp.float32, device=device)
            self.driven = wp.array(surround.driven, dtype=wp.int32, device=device)
            self.surround_compression = wp.zeros(n, dtype=wp.float32, device=device)
            self.surround_scratch = wp.zeros(n, dtype=wp.float32, device=device)
            self.surround_previous = wp.zeros(n, dtype=wp.float32, device=device)
            # Compression rate of the passive surface [m/s], a diagnostic of whether the
            # relaxation is still travelling when the substep ends.
            self.surround_rate = wp.zeros(n, dtype=wp.float32, device=device)

        if self.config.friction_model in ("maxwell", "column_maxwell", "elastic_coulomb"):
            self._install_default_friction_adapter()

    def _install_default_friction_adapter(self) -> None:
        """Auto-install the configured non-legacy friction parameter adapter."""
        if self.config.mu <= 0.0 or (
            self.config.friction_model not in ("column_maxwell", "elastic_coulomb")
            and self.config.friction_stiffness <= 0.0
            and self.config.friction_stiffness_per_area <= 0.0
        ):
            self.friction_solver = None
            return

        from .friction_parameter_adapter import FrictionParameterAdapter  # noqa: PLC0415

        rows = []
        for w in range(self.world_count):
            tau_w = float(self.world_blocks[w].friction_relaxation_time_s)
            rows.append(
                [
                    9.0
                    if self.config.friction_model == "elastic_coulomb"
                    else (8.0 if self.config.friction_model == "column_maxwell" else 7.0),
                    float(self.config.mu),
                    1.0,
                    1.0,
                    0.0,
                    float(self.config.friction_release_dwell_s),
                    0.0,
                    float(self.config.mu),
                    0.1,
                    1.0e12,
                    0.001,
                    tau_w,
                ]
            )
        adapter = FrictionParameterAdapter(
            self,
            self.world_count,
            base_kt=self.friction_kt,
            base_kv=self.friction_kv,
            is_default=True,
            initial_parameters=np.array(rows, dtype=np.float32),
            deflection=self.tangent_deflection,
            maxwell_force=self.tangent_maxwell_force,
        )
        self.friction_solver = adapter

    def reset(self) -> None:
        """Clear the viscoelastic overstress history and release the friction bristles."""
        self.q_state.zero_()
        self.peq_prev.zero_()
        self.tangent_stuck.zero_()
        self.tangent_dwell.zero_()
        self.ground_force.zero_()
        self.contact_point.zero_()
        self.tangent_deflection.zero_()
        self.tangent_maxwell_force.zero_()
        if self.friction_solver is not None:
            self.friction_solver.reset()
        if self.free_column_count:
            self.surround_compression.zero_()
            self.surround_scratch.zero_()
            self.surround_previous.zero_()
            self.surround_rate.zero_()

    def set_world_material(self, world: int, material: ShoeMaterial) -> None:
        """Give one world its own foam, leaving every other world untouched.

        Only the constitutive block changes: the geometry, the tributary areas,
        the friction bed and the contact settings stay shared, which is what makes
        a randomized batch a study of the material and of nothing else.

        Host-side and allocation-free on the device, but it does copy to the
        device, so call it before a CUDA graph capture, not inside one.

        Args:
            world: Index of the world to re-materialize.
            material: Replacement :class:`ShoeMaterial`.
        """
        if not 0 <= int(world) < self.world_count:
            raise IndexError("world index is outside the batch")
        set_material_block(self.world_blocks[int(world)], material)
        if self.config.friction_relaxation_time_s is None:
            tau = float(getattr(material, "maxwell_relaxation_time_s", 0.08))
            self.world_blocks[int(world)].friction_relaxation_time_s = tau
        self.world_params.assign(self.world_blocks)
        self._materials_dirty = True
        if self.friction_solver is not None and getattr(self.friction_solver, "is_default", False):
            self.friction_solver.update_world_tau(
                int(world), float(self.world_blocks[int(world)].friction_relaxation_time_s)
            )

    def set_world_materials(self, materials: Sequence[ShoeMaterial]) -> None:
        """Give every world its own foam in one call.

        Args:
            materials: One :class:`ShoeMaterial` per world, in world order.
        """
        materials = list(materials)
        if len(materials) != self.world_count:
            raise ValueError("set_world_materials needs one material per world")
        for block, material in zip(self.world_blocks, materials, strict=True):
            set_material_block(block, material)
            if self.config.friction_relaxation_time_s is None:
                block.friction_relaxation_time_s = float(getattr(material, "maxwell_relaxation_time_s", 0.08))
        self.world_params.assign(self.world_blocks)
        self._materials_dirty = True
        if self.friction_solver is not None and getattr(self.friction_solver, "is_default", False):
            taus = [float(b.friction_relaxation_time_s) for b in self.world_blocks]
            self.friction_solver.update_taus(taus)

    def _refresh_surround_constants(self, dt: float) -> None:
        """Recompute the per-world Maxwell update the surround sweep relaxes against.

        The decay and the overstress gain depend on the substep and on the world's
        own relaxation time, and they are evaluated in double precision on the host
        exactly as the single-material runtime evaluated them, so a batch of equal
        materials reproduces it bit for bit.

        The copy only happens when the substep or a material actually changed, so a
        captured CUDA graph replaying at a fixed substep never reaches it.
        """
        if self._surround_dt_s == dt and not self._materials_dirty:
            return
        decay = np.empty(self.world_count, np.float32)
        gain = np.empty(self.world_count, np.float32)
        for index, block in enumerate(self.world_blocks):
            world_decay, ramp = maxwell_coefficients_numpy(dt, block.tau_s)
            decay[index] = world_decay
            gain[index] = float(block.overstress * ramp)
        self.surround_decay.assign(decay)
        self.surround_gain.assign(gain)
        self._surround_dt_s = float(dt)
        self._materials_dirty = False

    def relax_surround(self, state, dt: float) -> None:
        """Relax the columns the carrier does not drive for one substep.

        Sweeps :func:`surround_relax` over the warm-started compression field and
        publishes the result through :func:`surround_write_free_top`, so the
        pressure and wrench kernels see one bed with one contact law.
        """
        cfg = self.surround
        sweeps = int(cfg.sweeps)
        sub_dt = dt / sweeps
        tau = float(cfg.relaxation_time_s)
        relaxation = 1.0 if tau <= 0.0 else 1.0 - float(np.exp(-sub_dt / tau))
        # The Maxwell update foundation_pressure applies after this relaxation, so the
        # surround balances the load it is about to carry rather than the previous one.
        self._refresh_surround_constants(dt)
        inputs = [
            self.carrier,
            self.column_count,
            state.body_q,
            self.driven,
            self.neighbors,
            self.anchor_local,
            self.z_free_rigid,
            self.rest_len,
            self.area,
            self.q_state,
            self.peq_prev,
            self.world_params,
            self.surround_decay,
            self.surround_gain,
            float(cfg.coupling_scale),
            float(cfg.attachment_n_m),
            float(cfg.max_strain),
            relaxation,
            int(bool(cfg.carrier_bond)),
        ]
        for _ in range(sweeps):
            wp.launch(
                surround_relax,
                dim=self.world_count * self.column_count,
                inputs=[*inputs, self.surround_compression, self.surround_scratch],
                device=self.device,
            )
            self.surround_compression, self.surround_scratch = self.surround_scratch, self.surround_compression
        if sweeps % 2:
            # An odd sweep count leaves the result in the other buffer, so a captured
            # CUDA graph would replay the next substep from the stale one. The copy
            # restores the entry-time buffer roles; the values are unchanged.
            self.surround_compression, self.surround_scratch = self.surround_scratch, self.surround_compression
            wp.copy(self.surround_compression, self.surround_scratch)
        wp.launch(
            surround_write_free_top,
            dim=self.world_count * self.column_count,
            inputs=[
                self.carrier,
                self.column_count,
                float(1.0 / dt),
                state.body_q,
                self.driven,
                self.anchor_local,
                self.z_free_rigid,
                self.surround_compression,
                self.surround_previous,
                self.z_free,
                self.surround_rate,
            ],
            device=self.device,
        )

    def apply(self, state, dt: float, clear_body_force: bool = False) -> None:
        """Accumulate the foundation wrench into ``state.body_f`` for one substep.

        Args:
            state: Simulation state supplying the carrier pose/velocity and receiving the wrench.
            dt: Substep duration [s].
            clear_body_force: Also zero the carrier's ``body_f`` in the fused reset launch, so a
                caller that only loads the foundation wrench can skip a separate
                :meth:`newton.State.clear_forces`. Leave False when other forces are staged into
                ``body_f`` before this call (e.g. an external probe load).
        """
        if self.free_column_count:
            self.relax_surround(state, dt)
        if clear_body_force:
            wp.launch(
                foundation_reset,
                dim=self.world_count,
                inputs=[
                    self.carrier,
                    int(clear_body_force),
                    state.body_f,
                    self.normal_force,
                    self.cop_moment,
                    self.active,
                    self.resultant_force,
                    self.resultant_moment_origin,
                    self.contact_power,
                    self.max_compression,
                    self.pressed_force,
                ],
                device=self.device,
            )
        wp.launch(
            foundation_pressure,
            dim=self.world_count * self.column_count,
            inputs=[
                self.carrier,
                self.column_count,
                dt,
                state.body_q,
                self.anchor_local,
                self.z_free,
                self.rest_len,
                self.world_params,
                self.q_state,
                self.peq_prev,
                self.compression,
                self.base_pressure,
            ],
            device=self.device,
        )
        plane_contact = self.ground_height_m is not None
        ground_inputs = [float(self.ground_height_m), self.ground_force, self.contact_point] if plane_contact else []
        wp.launch(
            foundation_apply_ground if plane_contact else foundation_apply,
            dim=self.world_count * self.column_count,
            inputs=[
                self.carrier,
                self.column_count,
                dt,
                state.body_q,
                state.body_qd,
                self.body_com,
                self.anchor_local,
                self.area,
                self.rest_len,
                self.neighbors,
                self.compression,
                self.base_pressure,
                self.tangent_anchor if self.friction_solver is None else self.friction_solver.scratch_anchor,
                self.tangent_stuck if self.friction_solver is None else self.friction_solver.scratch_stuck,
                self.tangent_dwell if self.friction_solver is None else self.friction_solver.scratch_dwell,
                self.friction_kt if self.friction_solver is None else self.friction_solver.zero_stiffness,
                self.friction_kv,
                self.world_params,
                self.column_force,
                self.column_pressed,
                *ground_inputs,
            ],
            device=self.device,
        )
        if self.friction_solver is not None:
            self.friction_solver.apply(state, dt)
        wp.launch(
            foundation_partial_ground if plane_contact else foundation_partial,
            dim=self.world_count * self.reduction_groups,
            inputs=[
                self.carrier,
                self.column_count,
                self.reduction_groups,
                state.body_q,
                state.body_qd,
                self.body_com,
                self.contact_point if plane_contact else self.anchor_local,
                self.compression,
                self.ground_force if plane_contact else self.column_force,
                self.column_pressed,
                self.partial_force,
                self.partial_torque,
                self.partial_moment,
                self.partial_cop,
                self.partial_normal,
                self.partial_pressed,
                self.partial_power,
                self.partial_max,
                self.partial_active,
            ],
            device=self.device,
            block_dim=32 if self.world_count <= 256 else 256,
        )
        wp.launch(
            foundation_finalize,
            dim=self.world_count,
            inputs=[
                self.carrier,
                self.reduction_groups,
                self.partial_force,
                self.partial_torque,
                self.partial_moment,
                self.partial_cop,
                self.partial_normal,
                self.partial_pressed,
                self.partial_power,
                self.partial_max,
                self.partial_active,
                state.body_f,
                self.normal_force,
                self.cop_moment,
                self.active,
                self.resultant_force,
                self.resultant_moment_origin,
                self.contact_power,
                self.max_compression,
                self.pressed_force,
            ],
            device=self.device,
            # Small batches need more blocks, not more inactive lanes in one block.
            block_dim=1 if self.world_count <= 256 else 256,
        )

    def diagnostics(self, world: int = 0) -> dict[str, float]:
        """Return the last substep's total normal force, center of pressure, and active count.

        Args:
            world: Index of the world to report; the accumulators carry one entry
                per world. Reads the device, so it cannot be used inside a CUDA
                graph capture.
        """
        fz = float(self.normal_force.numpy()[world])
        pressed = float(self.pressed_force.numpy()[world])
        moment = self.cop_moment.numpy()[world]
        cop = (float(moment[0] / pressed), float(moment[1] / pressed)) if pressed > 1.0e-9 else (0.0, 0.0)
        return {
            "normal_force_n": fz,
            "pressed_force_n": pressed,
            "cop_x_m": cop[0],
            "cop_y_m": cop[1],
            "active_columns": int(self.active.numpy()[world]),
        }
