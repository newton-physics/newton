# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Shoe descriptors and hypotheses for how a runner's impedance adapts to a shoe.

Runners adjust leg stiffness to surface stiffness (Ferris, Louie & Farley,
Proc. R. Soc. B 1998). The hypotheses here are interchangeable maps from a
shoe descriptor to gain multipliers, so data from several shoes can decide
between them without changing the controller or rollout.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .mechanics import Chain
from .plan import CONTACT_THRESHOLD_N, Plan

ADAPTATION_MODES = ("fixed", "series", "learned")
# Hip, knee, and ankle adapt; pelvis residual gains stay fixed.
_ADAPTED = slice(3, 6)


@dataclass(frozen=True)
class ShoeDescriptor:
    """Summarize one shoe from a standardized virtual compression test.

    Attributes:
        stiffness_n_m: Secant vertical stiffness at peak load [N/m].
        energy_return: Unloading work divided by loading work [-].
        peak_force_n: Test peak load [N].
        speed_m_s: Test loading and unloading speed [m/s].
    """

    stiffness_n_m: float
    energy_return: float
    peak_force_n: float
    speed_m_s: float

    def features(self) -> np.ndarray:
        """Return ``[ln(stiffness [N/m]), energy_return]`` for learned adaptation."""
        return np.array([math.log(self.stiffness_n_m), self.energy_return])


def characterize_shoe(shoe, *, peak_force_n: float = 1500.0, speed_m_s: float = 0.2, dt_s: float = 1.25e-4):
    """Press a level shoe vertically to a peak load and release it at constant speed.

    This is a flat full-foot test at the shoe's static pitch. Match ``peak_force_n``
    and ``speed_m_s`` to the lab compression protocol before comparing shoes.

    Args:
        shoe: :class:`..cartesian.shoe.Shoe`. Its history is reset before and after.
        peak_force_n: Load at which compression reverses [N].
        speed_m_s: Constant ankle speed [m/s].
        dt_s: Contact step [s].

    Returns:
        The descriptor and the force-displacement curve [m, N], shape [N, 2].
    """
    if min(peak_force_n, speed_m_s, dt_s) <= 0:
        raise ValueError("peak_force_n, speed_m_s, and dt_s must be positive")
    shoe.foundation.reset()
    touch = -float(np.min(shoe.anchor_local_m[:, 2]))
    travel_limit = 0.9 * float(np.max(shoe.shoe.column_bed.rest_length_m))
    pitch = shoe.static_pitch_rad
    step = speed_m_s * dt_s
    displacement, force = -0.002, 0.0
    curve = []
    while force < peak_force_n:
        if displacement > travel_limit:
            raise ValueError("Shoe did not reach the peak load before 90% compression")
        wrench, _ = shoe.apply([0.0, touch - displacement], [0.0, -speed_m_s], pitch, 0.0, dt_s)
        force = float(wrench[1])
        curve.append((displacement, force))
        displacement += step
    peak_index = len(curve) - 1
    while force > 0.0 and displacement > -0.002:
        displacement -= step
        wrench, _ = shoe.apply([0.0, touch - displacement], [0.0, speed_m_s], pitch, 0.0, dt_s)
        force = float(wrench[1])
        curve.append((displacement, force))
    shoe.foundation.reset()
    curve = np.asarray(curve)
    peak_displacement, peak = curve[peak_index]
    if peak_displacement <= 0:
        raise ValueError("Shoe reached the peak load before geometric contact; check the mount")
    loading = curve[: peak_index + 1]
    unloading = curve[peak_index:]
    work_in = float(np.trapezoid(np.maximum(loading[:, 1], 0.0), loading[:, 0]))
    work_out = -float(np.trapezoid(np.maximum(unloading[:, 1], 0.0), unloading[:, 0]))
    descriptor = ShoeDescriptor(
        stiffness_n_m=float(peak / peak_displacement),
        energy_return=work_out / work_in,
        peak_force_n=peak_force_n,
        speed_m_s=speed_m_s,
    )
    return descriptor, curve


def leg_stiffness(plan: Plan, chain: Chain, contact_threshold_n: float = CONTACT_THRESHOLD_N) -> float:
    """Estimate spring-mass leg stiffness [N/m] as peak vertical force over hip-ankle shortening."""
    contact = plan.grf_n[:, 1] > contact_threshold_n
    rows = np.flatnonzero(contact)
    if len(rows) == 0:
        raise ValueError("Plan contains no contact")
    length = np.array([np.linalg.norm(np.subtract(*chain.kinematics(plan.q[k])[[0, 2]])) for k in rows])
    shortening = float(length[0] - np.min(length))
    if shortening <= 0:
        raise ValueError("Hip-ankle distance does not shorten during contact")
    return float(np.max(plan.grf_n[rows, 1]) / shortening)


@dataclass(frozen=True)
class ShoeAdaptation:
    """Map a shoe descriptor to per-channel stiffness and damping multipliers.

    - ``fixed``: keep the identified gains in every shoe.
    - ``series``: keep leg and shoe series stiffness constant, so
      ``1/k_leg' = 1/k_leg + 1/k_shoe - 1/k_shoe'``. Joint stiffness scales by
      ``k_leg'/k_leg``, which assumes leg stiffness changes come equally from each joint.
    - ``learned``: ``ln(K'/K) = (z' - z) @ weights`` with ``z`` from
      :meth:`ShoeDescriptor.features`. Fit ``weights`` once several shoes are measured.

    Damping scales with the square root of the stiffness multiplier, keeping
    each damping ratio. Pelvis residual gains never adapt.

    Attributes:
        mode: One of :data:`ADAPTATION_MODES`.
        reference_shoe: Shoe in which the gains were identified.
        leg_stiffness_n_m: Leg stiffness in the reference shoe [N/m]; required by ``series``.
        weights: Log-gain sensitivities for hip, knee, ankle, shape (2, 3); required by ``learned``.
    """

    mode: str
    reference_shoe: ShoeDescriptor
    leg_stiffness_n_m: float | None = None
    weights: np.ndarray | None = None

    def __post_init__(self):
        if self.mode not in ADAPTATION_MODES:
            raise ValueError(f"mode must be one of {ADAPTATION_MODES}")
        if self.mode == "series" and not (self.leg_stiffness_n_m and self.leg_stiffness_n_m > 0):
            raise ValueError("series adaptation requires a positive leg stiffness")
        if self.mode == "learned" and np.shape(self.weights) != (2, 3):
            raise ValueError("learned adaptation requires weights with shape (2, 3)")

    def factors(self, shoe: ShoeDescriptor) -> tuple[np.ndarray, np.ndarray]:
        """Return stiffness and damping multipliers, each shape (6,)."""
        stiffness = np.ones(6)
        if self.mode == "series":
            compliance = (
                1.0 / self.leg_stiffness_n_m + 1.0 / self.reference_shoe.stiffness_n_m - 1.0 / shoe.stiffness_n_m
            )
            if compliance <= 0:
                raise ValueError("Shoe is too compliant for a constant series stiffness")
            stiffness[_ADAPTED] = 1.0 / (compliance * self.leg_stiffness_n_m)
        elif self.mode == "learned":
            stiffness[_ADAPTED] = np.exp((shoe.features() - self.reference_shoe.features()) @ self.weights)
        return stiffness, np.sqrt(stiffness)
