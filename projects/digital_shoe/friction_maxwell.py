# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tangential standard-linear-solid bristle with a Coulomb slider.

The equilibrium spring is parallel to a Maxwell branch (spring + dashpot in
series). Unlike a dashpot applied directly to relative velocity, its stress is
an internal mechanical state with finite high-frequency stiffness. Backward
Euler and one radial return update both mechanical states consistently.
This is not an output-force filter. Normal mechanics remain external inputs.
"""

import warp as wp


@wp.func
def column_maxwell_parameters(
    equilibrium_shear_modulus: float,
    overstress_ratio: float,
    area: float,
    rest_length: float,
    relaxation_time_s: float,
) -> tuple[float, float]:
    """Return material-derived equilibrium shear stiffness [N/m] and viscosity [N s/m].

    The foam column is treated as a linear shear layer with ``k = G A / L``.
    Its Maxwell branch uses the material's branch-to-equilibrium modulus ratio
    and relaxation time. This is a large-strain extrapolation of the fitted
    normal material, not independent outsole shear calibration.
    """
    scale = area / wp.max(rest_length, 1.0e-12)
    stiffness = equilibrium_shear_modulus * scale
    viscosity = equilibrium_shear_modulus * overstress_ratio * scale * relaxation_time_s
    return stiffness, viscosity


@wp.func
def elastic_coulomb_stiffness(equilibrium_shear_modulus: float, area: float, rest_length: float) -> float:
    """Return area-scaled elastic shear stiffness [N/m] for one column."""
    return equilibrium_shear_modulus * area / wp.max(rest_length, 1.0e-12)


@wp.func
def bristle_maxwell_step(
    velocity: wp.vec2,
    dt: float,
    normal: float,
    kt: float,
    viscosity: float,
    relaxation_time_s: float,
    mu: float,
    release_dwell: float,
    deflection: wp.vec2,
    maxwell_stress: wp.vec2,
    stuck: int,
    dwell: float,
) -> tuple[wp.vec2, wp.mat22, wp.vec2, wp.vec2, int, float]:
    """Advance coupled elastic, viscous and plastic tangential states.

    Args:
        velocity: Relative tangential velocity [m/s].
        dt: Timestep [s].
        normal: Prescribed normal force [N].
        kt: Equilibrium tangential stiffness [N/m].
        viscosity: Maxwell dashpot coefficient [N s/m].
        relaxation_time_s: Maxwell relaxation time [s]; branch stiffness is viscosity/time.
        mu: Coulomb coefficient.
        release_dwell: Unloaded release delay [s].
        deflection: Elastic-slider displacement [m].
        maxwell_stress: Maxwell branch tangential force [N], not stress per unit area.
        stuck: Incoming contact flag.
        dwell: Incoming unloaded dwell [s].

    Returns:
        Force [N], velocity Jacobian [N s/m], deflection [m], branch force [N],
        contact flag, unloaded dwell [s].
    """
    tau = wp.max(relaxation_time_s, 1.0e-12)
    branch_k = wp.max(viscosity, 0.0) / tau
    decay = tau / (tau + dt)
    zero = wp.vec2(0.0)
    if normal <= 0.0 or kt <= 0.0:
        elapsed = dwell + dt
        if stuck == 0 or elapsed > release_dwell or kt <= 0.0:
            return zero, wp.mat22(0.0), zero, zero, 0, 0.0
        return zero, wp.mat22(0.0), deflection, decay * maxwell_stress, stuck, elapsed
    z_old = zero
    q_old = zero
    if stuck != 0:
        z_old = deflection
        q_old = maxwell_stress
    if viscosity <= 0.0:
        q_old = zero
    increment = dt * velocity
    algorithmic_k = kt + decay * branch_k
    trial = kt * (z_old + increment) + decay * (q_old + branch_k * increment)
    magnitude = wp.length(trial)
    capacity = wp.max(mu, 0.0) * normal
    plastic = zero
    traction = trial
    tangent = wp.identity(2, float) * (-algorithmic_k * dt)
    if magnitude > capacity and magnitude > 1.0e-12:
        direction = trial / magnitude
        plastic = ((magnitude - capacity) / algorithmic_k) * direction
        traction = capacity * direction
        tangent = (wp.identity(2, float) - wp.outer(direction, direction)) * (
            -algorithmic_k * dt * capacity / magnitude
        )
    z_new = z_old + increment - plastic
    q_new = decay * (q_old + branch_k * (increment - plastic))
    force = -traction
    return force, tangent, z_new, q_new, 1, 0.0


@wp.func
def bristle_elastic_coulomb_step(
    velocity: wp.vec2,
    dt: float,
    normal: float,
    kt: float,
    mu: float,
    release_dwell: float,
    deflection: wp.vec2,
    stuck: int,
    dwell: float,
) -> tuple[wp.vec2, wp.mat22, wp.vec2, int, float]:
    """Advance an area-scaled elastic bristle with a Coulomb slider."""
    force, tangent, z, _q, next_stuck, next_dwell = bristle_maxwell_step(
        velocity,
        dt,
        normal,
        kt,
        0.0,
        1.0,
        mu,
        release_dwell,
        deflection,
        wp.vec2(0.0),
        stuck,
        dwell,
    )
    return force, tangent, z, next_stuck, next_dwell


@wp.func
def maxwell_bristle_energy(z: wp.vec2, q: wp.vec2, kt: float, viscosity: float, relaxation_time_s: float) -> float:
    """Return stored mechanical energy [J] of both tangential springs."""
    energy = 0.5 * kt * wp.dot(z, z)
    if viscosity > 0.0:
        energy += 0.5 * relaxation_time_s * wp.dot(q, q) / viscosity
    return energy
