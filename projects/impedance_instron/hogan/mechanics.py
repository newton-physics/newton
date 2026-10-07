# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Analytic planar mechanics for a floating pelvis carrying one thigh, shank, and foot.

Coordinates are ``[hip_x, hip_z, pelvis, hip, knee, ankle]``. The pelvis origin
is the hip center, so ``hip_x``/``hip_z`` match :mod:`..cartesian.mechanics`.
Absolute angles increase from world +x toward +z. The pelvis local +x axis
points up the trunk (``pi/2`` when upright). Thigh absolute angle is
``pelvis + hip + pi``, so the hip angle is zero for a vertical thigh under an
upright pelvis and positive in flexion. Knee flexion is negative and ankle
dorsiflexion positive, as in the Cartesian model.

The pelvis body lumps the trunk, arms, head, and contralateral leg rigidly.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

BODY_NAMES = ("pelvis", "thigh", "shank", "foot")
COORDINATE_NAMES = ("hip_x", "hip_z", "pelvis", "hip", "knee", "ankle")
UPRIGHT_PELVIS_RAD = 0.5 * math.pi
_ANGLE_OFFSETS = np.array([0.0, math.pi, math.pi, 1.5 * math.pi])


def _array(value, shape: tuple[int, ...], name: str) -> np.ndarray:
    """Convert a finite value to the required floating-point array shape."""
    result = np.asarray(value, dtype=float)
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError(f"{name} must be finite with shape {shape}")
    return result


class Chain:
    """Describe the four integrated bodies, ordered pelvis, thigh, shank, foot.

    Args:
        lengths_m: Thigh and shank lengths [m], shape (2,).
        endpoint_local_m: Foot endpoint offset from the ankle [m], shape (2,).
        masses_kg: Segment masses [kg], shape (4,).
        com_local_m: COM offsets from each segment origin [m], shape (4, 2).
        inertias_kg_m2: Planar inertias about each segment COM [kg m^2], shape (4,).
    """

    def __init__(self, lengths_m, endpoint_local_m, masses_kg, com_local_m, inertias_kg_m2):
        self.lengths_m = _array(lengths_m, (2,), "lengths_m").copy()
        self.endpoint_local_m = _array(endpoint_local_m, (2,), "endpoint_local_m").copy()
        self.masses_kg = _array(masses_kg, (4,), "masses_kg").copy()
        self.com_local_m = _array(com_local_m, (4, 2), "com_local_m").copy()
        self.inertias_kg_m2 = _array(inertias_kg_m2, (4,), "inertias_kg_m2").copy()
        for name in ("lengths_m", "masses_kg", "inertias_kg_m2"):
            if np.any(getattr(self, name) <= 0):
                raise ValueError(f"{name} must be positive")
        self._angular = np.zeros((4, 6))
        self._angular[:, 2:] = np.tril(np.ones((4, 4)))
        self._rotational_mass = self._angular.T @ (self.inertias_kg_m2[:, None] * self._angular)
        self._mass_weights = np.repeat(self.masses_kg, 2)[:, None]
        # The thigh is attached at the pelvis origin (hip center).
        self._offsets = np.array(
            [[0.0, 0.0], [self.lengths_m[0], 0.0], [self.lengths_m[1], 0.0], self.endpoint_local_m]
        )
        self._cached_q: np.ndarray | None = None

    @property
    def total_mass_kg(self) -> float:
        """Return the summed mass of all four bodies [kg]."""
        return float(np.sum(self.masses_kg))

    @staticmethod
    def _body(body: int) -> int:
        """Check a segment index without accepting boolean values."""
        if isinstance(body, bool) or not isinstance(body, (int, np.integer)) or not 0 <= body < 4:
            raise ValueError("body must be 0 (pelvis), 1 (thigh), 2 (shank), or 3 (foot)")
        return int(body)

    def _geometry(self, q) -> None:
        """Cache rotations and point Jacobians shared by contact and dynamics."""
        q = _array(q, (6,), "q")
        if self._cached_q is not None and np.array_equal(q, self._cached_q):
            return
        self._angles = np.cumsum(q[2:]) + _ANGLE_OFFSETS
        cosines, sines = np.cos(self._angles), np.sin(self._angles)
        self._rotations = np.empty((4, 2, 2))
        self._rotations[:, 0, 0] = cosines
        self._rotations[:, 0, 1] = -sines
        self._rotations[:, 1, 0] = sines
        self._rotations[:, 1, 1] = cosines
        self._radii = (self._rotations @ self._offsets[:, :, None])[:, :, 0]
        self._com_radii = (self._rotations @ self.com_local_m[:, :, None])[:, :, 0]
        self._joints = np.empty((5, 2))
        self._joints[0] = q[:2]
        self._joints[1:] = q[:2] + np.cumsum(self._radii, axis=0)
        perpendicular = self._radii[:, ::-1] * np.array([-1.0, 1.0])
        increments = perpendicular[:, :, None] * self._angular[:, None, :]
        self._origin_jacobians = np.zeros((4, 2, 6))
        self._origin_jacobians[:, :, :2] = np.eye(2)
        self._origin_jacobians[1:] += np.cumsum(increments[:3], axis=0)
        com_perpendicular = self._com_radii[:, ::-1] * np.array([-1.0, 1.0])
        self._com_jacobians = self._origin_jacobians + com_perpendicular[:, :, None] * self._angular[:, None, :]
        self._cached_q = q.copy()

    def angular_jacobian(self, body: int) -> np.ndarray:
        """Return the constant angular Jacobian, shape (6,)."""
        return self._angular[self._body(body)].copy()

    def angle(self, q, body: int) -> float:
        """Return the segment's absolute angle [rad]."""
        self._geometry(q)
        return float(self._angles[self._body(body)])

    def point(self, q, body: int, local, v=None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return a point's position [m], Jacobian, shape (2, 6), and ``Jdot @ v`` [m/s^2]."""
        self._geometry(q)
        body = self._body(body)
        radius = self._rotations[body] @ _array(local, (2,), "local")
        position = self._joints[body] + radius
        jacobian = self._origin_jacobians[body] + np.outer(np.array([-radius[1], radius[0]]), self._angular[body])
        centripetal = np.zeros(2)
        if v is not None:
            omega_squared = np.square(self._angular @ _array(v, (6,), "v"))
            centripetal = -np.sum(self._radii[:body] * omega_squared[:body, None], axis=0)
            centripetal -= radius * omega_squared[body]
        return position, jacobian, centripetal

    def kinematics(self, q) -> np.ndarray:
        """Return hip, knee, ankle, and foot endpoint positions [m], shape (4, 2)."""
        self._geometry(q)
        return self._joints[1:].copy()

    def reach(self, q, ankle_m) -> np.ndarray:
        """Re-solve hip and knee so the ankle lands on ``ankle_m``, keeping the foot's absolute angle.

        Args:
            q: Coordinates, shape [N, 6].
            ankle_m: Ankle targets [m], shape [N, 2]. Out-of-reach targets straighten the knee.

        Returns:
            Coordinates with the hip, knee, and ankle angles replaced, shape [N, 6].
        """
        q = np.array(q, dtype=float)
        delta = np.asarray(ankle_m, dtype=float) - q[:, :2]
        thigh, shank = self.lengths_m
        distance = np.clip(np.linalg.norm(delta, axis=1), abs(thigh - shank) + 1e-9, (thigh + shank) * (1 - 1e-9))
        hip_angle = np.arccos((thigh**2 + distance**2 - shank**2) / (2 * thigh * distance))
        knee = np.arccos((thigh**2 + shank**2 - distance**2) / (2 * thigh * shank)) - math.pi
        # Rotating the thigh forward of the hip-ankle line keeps the knee anterior (flexion negative).
        thigh_angle = np.arctan2(delta[:, 1], delta[:, 0]) + hip_angle
        old_thigh = q[:, 2] + q[:, 3] + math.pi
        total = q[:, 2:].sum(axis=1)
        q[:, 3] += np.angle(np.exp(1j * (thigh_angle - old_thigh)))
        q[:, 4] = knee
        q[:, 5] = total - q[:, 2:5].sum(axis=1)
        return q

    def com(self, q) -> np.ndarray:
        """Return the whole-chain center of mass [m], shape (2,)."""
        self._geometry(q)
        coms = self._joints[:4] + self._com_radii
        return self.masses_kg @ coms / self.total_mass_kg

    def dynamics(self, q, v, gravity: float = 9.81) -> tuple[np.ndarray, np.ndarray]:
        """Return the mass matrix, shape (6, 6), and bias in ``M @ acceleration + bias = load``.

        The bias holds centripetal and gravity terms [N, N, N m, N m, N m, N m].
        """
        self._geometry(q)
        v = _array(v, (6,), "v")
        if not np.isfinite(gravity) or gravity < 0:
            raise ValueError("gravity must be finite and nonnegative")
        omega_squared = np.square(self._angular @ v)
        centripetal = -self._com_radii * omega_squared[:, None]
        centripetal[1:] -= np.cumsum(self._radii[:3] * omega_squared[:3, None], axis=0)
        centripetal[:, 1] += gravity
        jacobian = self._com_jacobians.reshape(8, 6)
        mass = jacobian.T @ (self._mass_weights * jacobian) + self._rotational_mass
        bias = jacobian.T @ (self.masses_kg[:, None] * centripetal).reshape(8)
        return mass, bias


@dataclass(frozen=True)
class RestOfBody:
    """Lumped trunk, arms, head, and contralateral leg carried by the pelvis.

    The defaults are rough adult-runner estimates, not subject measurements:
    the lumped COM sits 0.19 m above the hip center when upright, with a
    0.35 m sagittal radius of gyration. Replace them with subject data.

    Attributes:
        com_local_m: COM offset from the hip center in the pelvis frame [m].
        radius_of_gyration_m: Sagittal radius of gyration about the COM [m].
        mass_kg: Lumped mass [kg]. ``None`` uses subject mass minus the modeled leg.
    """

    com_local_m: tuple[float, float] = (0.19, 0.0)
    radius_of_gyration_m: float = 0.35
    mass_kg: float | None = None


def chain_from_profile(reference: dict, profile: dict, rest: RestOfBody | None = None) -> Chain:
    """Add a lumped pelvis to a Cartesian single-leg profile and reference geometry."""
    rest = rest or RestOfBody()
    leg_masses = np.asarray(profile["masses_kg"], dtype=float)
    mass = float(reference["subject_mass_kg"]) - float(np.sum(leg_masses)) if rest.mass_kg is None else rest.mass_kg
    if not np.isfinite(mass) or mass <= 0:
        raise ValueError("Rest-of-body mass must be positive; check subject mass and leg masses")
    if not np.isfinite(rest.radius_of_gyration_m) or rest.radius_of_gyration_m <= 0:
        raise ValueError("Rest-of-body radius of gyration must be positive")
    return Chain(
        reference["lengths_m"],
        reference["endpoint_local_m"],
        np.concatenate(([mass], leg_masses)),
        np.vstack((np.asarray(rest.com_local_m, dtype=float), np.asarray(profile["com_local_m"], dtype=float))),
        np.concatenate(([mass * rest.radius_of_gyration_m**2], np.asarray(profile["inertias_kg_m2"], dtype=float))),
    )


def from_leg_coordinates(state, velocity, pelvis_rad, pelvis_rate_rad_s) -> tuple[np.ndarray, np.ndarray]:
    """Convert Cartesian-model rows ``[hip_x, hip_z, thigh, knee, ankle]`` to chain coordinates.

    Args:
        state: Cartesian-model coordinates, shape [N, 5].
        velocity: Cartesian-model velocities, shape [N, 5].
        pelvis_rad: Absolute pelvis angle [rad], shape [N].
        pelvis_rate_rad_s: Pelvis angular velocity [rad/s], shape [N].

    Returns:
        Chain coordinates and velocities, each shape [N, 6]. Hip angles are
        wrapped near zero and unwrapped over time.
    """
    state = np.asarray(state, dtype=float)
    velocity = np.asarray(velocity, dtype=float)
    pelvis = np.broadcast_to(np.asarray(pelvis_rad, dtype=float), len(state))
    rate = np.broadcast_to(np.asarray(pelvis_rate_rad_s, dtype=float), len(state))
    hip = np.unwrap(np.mod(state[:, 2] - pelvis, 2.0 * math.pi) - math.pi)
    q = np.column_stack((state[:, :2], pelvis, hip, state[:, 3:5]))
    v = np.column_stack((velocity[:, :2], rate, velocity[:, 2] - rate, velocity[:, 3:5]))
    return q, v
