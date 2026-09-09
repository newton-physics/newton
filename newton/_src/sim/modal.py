# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Reduced modal basis utilities for elastic links."""

from __future__ import annotations

import math
import warnings
from collections.abc import Sequence
from typing import Any

import numpy as np


class ModalBasis:
    """Sampled linear modal basis for a reduced elastic body.

    The basis stores displacement samples ``phi`` at body-local material points.
    A deformation at point ``X`` is evaluated linearly as ``sum_i phi_i(X) q_i``.
    Unknown points are evaluated by inverse-distance interpolation from the
    stored samples, keeping the runtime representation independent of how the
    modes were generated.

    Args:
        sample_points: Body-local sample points [m], shape ``[sample_count, 3]``.
        sample_phi: Translational mode values [m per modal coordinate], shape
            ``[sample_count, mode_count, 3]``.
        sample_psi: Optional angular mode values [rad per modal coordinate], shape
            ``[sample_count, mode_count, 3]``. Each entry is the body-local
            infinitesimal rotation vector of the material frame at the sample for
            a unit modal coordinate. Defaults to zeros, which disables rotational
            joint coupling.
        sample_mass: Optional per-sample lumped mass [kg], shape ``[sample_count]``.
            When supplied, ``mode_mass`` and any unspecified inertia coupling
            integrals (``mode_coupling_linear``, ``_angular``, ``_centrifugal``,
            ``_coriolis``) are computed from it.
        mode_mass: Optional modal mass [kg], shape ``[mode_count]``. An explicit
            value always takes precedence; when omitted it is derived from
            ``sample_mass`` as ``sum_s mass_s |phi_i(x_s)|^2``, sharing the mass
            distribution used for the coupling integrals.
        mode_stiffness: Modal stiffness values [N/m], shape ``[mode_count]``.
        mode_damping: Modal damping values [N s/m], shape ``[mode_count]``.
        mode_coupling_linear: Optional linear coupling integral
            ``S_i = sum_s mass_s phi_i(x_s)`` [kg], shape ``[mode_count, 3]``.
            Couples floating-frame translational acceleration and gravity into
            each mode. ``None`` when unavailable.
        mode_coupling_angular: Optional angular coupling integral
            ``sum_s mass_s (phi_i(x_s) cross x_s)`` [kg m], shape
            ``[mode_count, 3]``. Couples floating-frame angular acceleration
            into each mode. ``None`` when unavailable.
        mode_coupling_centrifugal: Optional centrifugal coupling integral
            ``sum_s mass_s (phi_i(x_s) outer x_s)`` [kg m^2], shape
            ``[mode_count, 3, 3]``. Couples the floating-frame centrifugal
            acceleration into each mode. ``None`` when unavailable.
        mode_coupling_coriolis: Optional Coriolis coupling integral
            ``sum_s mass_s (phi_j(x_s) cross phi_i(x_s))`` [kg m^2], shape
            ``[mode_count, mode_count, 3]``. Couples the floating-frame angular
            velocity with the modal velocities. Antisymmetric in ``(i, j)``, so
            it vanishes for a single mode. ``None`` when unavailable.
        label: Optional basis label.
        interpolation_epsilon: Distance regularizer [m] used for inverse-distance
            interpolation.
    """

    def __init__(
        self,
        sample_points: Sequence[Sequence[float]] | np.ndarray | None = None,
        sample_phi: Sequence[Sequence[Sequence[float]]] | np.ndarray | None = None,
        sample_psi: Sequence[Sequence[Sequence[float]]] | np.ndarray | None = None,
        sample_mass: Sequence[float] | np.ndarray | None = None,
        mode_mass: Sequence[float] | np.ndarray | None = None,
        mode_stiffness: Sequence[float] | np.ndarray | None = None,
        mode_damping: Sequence[float] | np.ndarray | None = None,
        mode_coupling_linear: Sequence[Sequence[float]] | np.ndarray | None = None,
        mode_coupling_angular: Sequence[Sequence[float]] | np.ndarray | None = None,
        mode_coupling_centrifugal: Sequence[Sequence[Sequence[float]]] | np.ndarray | None = None,
        mode_coupling_coriolis: Sequence[Sequence[Sequence[float]]] | np.ndarray | None = None,
        label: str | None = None,
        interpolation_epsilon: float = 1.0e-8,
    ):
        self.label = label
        self.interpolation_epsilon = float(interpolation_epsilon)

        if sample_points is None:
            points = np.zeros((0, 3), dtype=np.float32)
        else:
            points = np.asarray(sample_points, dtype=np.float32)
            if points.ndim == 1:
                points = points.reshape((1, 3))
            if points.ndim != 2 or points.shape[1] != 3:
                raise ValueError(f"sample_points must have shape [sample_count, 3], got {points.shape}")

        if sample_phi is None:
            mode_count = self._infer_mode_count(mode_mass, mode_stiffness, mode_damping)
            phi = np.zeros((points.shape[0], mode_count, 3), dtype=np.float32)
        else:
            phi = np.asarray(sample_phi, dtype=np.float32)
            if phi.ndim == 2 and points.shape[0] == 1:
                phi = phi.reshape((1, phi.shape[0], 3))
            if phi.ndim != 3 or phi.shape[0] != points.shape[0] or phi.shape[2] != 3:
                raise ValueError(
                    "sample_phi must have shape [sample_count, mode_count, 3], "
                    f"got {phi.shape} for {points.shape[0]} sample points"
                )
            mode_count = int(phi.shape[1])

        if sample_psi is None:
            psi = np.zeros((points.shape[0], mode_count, 3), dtype=np.float32)
        else:
            psi = np.asarray(sample_psi, dtype=np.float32)
            if psi.ndim == 2 and points.shape[0] == 1:
                psi = psi.reshape((1, psi.shape[0], 3))
            if psi.ndim != 3 or psi.shape != (points.shape[0], mode_count, 3):
                raise ValueError(
                    "sample_psi must have shape [sample_count, mode_count, 3], "
                    f"got {psi.shape} for {points.shape[0]} sample points and {mode_count} modes"
                )

        self.sample_points = np.array(points, dtype=np.float32, copy=True)
        self.sample_phi = np.array(phi, dtype=np.float32, copy=True)
        self.sample_psi = np.array(psi, dtype=np.float32, copy=True)
        self.sample_mass = self._coerce_sample_mass(sample_mass, points.shape[0])

        lumped_mass = lumped_linear = lumped_angular = lumped_centrifugal = lumped_coriolis = None
        if self.sample_mass is not None and mode_count > 0:
            (
                lumped_mass,
                lumped_linear,
                lumped_angular,
                lumped_centrifugal,
                lumped_coriolis,
            ) = self._lumped_reduced_inertia(self.sample_points, self.sample_phi, self.sample_mass)

        self.mode_mass = self._coerce_mode_array(mode_mass, mode_count, 1.0, "mode_mass")
        self.mode_stiffness = self._coerce_mode_array(mode_stiffness, mode_count, 0.0, "mode_stiffness")
        self.mode_damping = self._coerce_mode_array(mode_damping, mode_count, 0.0, "mode_damping")
        self.mode_coupling_linear = self._coerce_coupling(mode_coupling_linear, mode_count, "mode_coupling_linear")
        self.mode_coupling_angular = self._coerce_coupling(mode_coupling_angular, mode_count, "mode_coupling_angular")
        self.mode_coupling_centrifugal = self._coerce_coupling_matrix(
            mode_coupling_centrifugal, mode_count, "mode_coupling_centrifugal"
        )
        self.mode_coupling_coriolis = self._coerce_coupling_pairs(
            mode_coupling_coriolis, mode_count, "mode_coupling_coriolis"
        )

        if mode_mass is None and lumped_mass is not None:
            self.mode_mass = lumped_mass
        if self.mode_coupling_linear is None:
            self.mode_coupling_linear = lumped_linear
        if self.mode_coupling_angular is None:
            self.mode_coupling_angular = lumped_angular
        if self.mode_coupling_centrifugal is None:
            self.mode_coupling_centrifugal = lumped_centrifugal
        if self.mode_coupling_coriolis is None:
            self.mode_coupling_coriolis = lumped_coriolis

    @property
    def mode_count(self) -> int:
        """Number of modal coordinates."""
        return int(self.sample_phi.shape[1])

    @property
    def sample_count(self) -> int:
        """Number of stored sample points."""
        return int(self.sample_points.shape[0])

    @staticmethod
    def _infer_mode_count(*arrays: Sequence[float] | np.ndarray | None) -> int:
        mode_count = 0
        for values in arrays:
            if values is None:
                continue
            count = int(np.asarray(values, dtype=np.float32).reshape((-1,)).shape[0])
            if mode_count not in (0, count):
                raise ValueError(f"mode property lengths disagree: {mode_count} and {count}")
            mode_count = count
        return mode_count

    @staticmethod
    def _coerce_mode_array(
        values: Sequence[float] | np.ndarray | None,
        mode_count: int,
        default: float,
        name: str,
    ) -> np.ndarray:
        if values is None:
            return np.full(mode_count, default, dtype=np.float32)
        array = np.asarray(values, dtype=np.float32).reshape((-1,))
        if array.shape[0] != mode_count:
            raise ValueError(f"{name} must have length {mode_count}, got {array.shape[0]}")
        return np.array(array, dtype=np.float32, copy=True)

    @staticmethod
    def _coerce_sample_mass(
        values: Sequence[float] | np.ndarray | None,
        sample_count: int,
    ) -> np.ndarray | None:
        if values is None:
            return None
        array = np.asarray(values, dtype=np.float32).reshape((-1,))
        if array.shape[0] != sample_count:
            raise ValueError(f"sample_mass must have length {sample_count}, got {array.shape[0]}")
        return np.array(array, dtype=np.float32, copy=True)

    @staticmethod
    def _coerce_coupling(
        values: Sequence[Sequence[float]] | np.ndarray | None,
        mode_count: int,
        name: str,
    ) -> np.ndarray | None:
        if values is None:
            return None
        array = np.asarray(values, dtype=np.float32)
        if array.shape != (mode_count, 3):
            raise ValueError(f"{name} must have shape ({mode_count}, 3), got {array.shape}")
        return np.array(array, dtype=np.float32, copy=True)

    @staticmethod
    def _coerce_coupling_matrix(
        values: Sequence[Sequence[Sequence[float]]] | np.ndarray | None,
        mode_count: int,
        name: str,
    ) -> np.ndarray | None:
        if values is None:
            return None
        array = np.asarray(values, dtype=np.float32)
        if array.shape != (mode_count, 3, 3):
            raise ValueError(f"{name} must have shape ({mode_count}, 3, 3), got {array.shape}")
        return np.array(array, dtype=np.float32, copy=True)

    @staticmethod
    def _coerce_coupling_pairs(
        values: Sequence[Sequence[Sequence[float]]] | np.ndarray | None,
        mode_count: int,
        name: str,
    ) -> np.ndarray | None:
        if values is None:
            return None
        array = np.asarray(values, dtype=np.float32)
        if array.shape != (mode_count, mode_count, 3):
            raise ValueError(f"{name} must have shape ({mode_count}, {mode_count}, 3), got {array.shape}")
        return np.array(array, dtype=np.float32, copy=True)

    @staticmethod
    def _lumped_reduced_inertia(
        points: np.ndarray,
        phi: np.ndarray,
        sample_mass: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return lumped modal mass and the floating-frame coupling integrals.

        ``mass[i] = sum_s mass_s |phi_i(x_s)|^2`` [kg], shape ``[mode_count]``;
        ``linear[i] = sum_s mass_s phi_i(x_s)`` [kg] and
        ``angular[i] = sum_s mass_s (phi_i(x_s) cross x_s)`` [kg m], each shape
        ``[mode_count, 3]``; ``centrifugal[i] = sum_s mass_s (phi_i(x_s) outer x_s)``
        [kg m^2], shape ``[mode_count, 3, 3]``; ``coriolis[i, j] = sum_s mass_s
        (phi_j(x_s) cross phi_i(x_s))`` [kg m^2], shape ``[mode_count, mode_count, 3]``.
        """
        mass = sample_mass.astype(np.float64)
        phi64 = phi.astype(np.float64)
        points64 = points.astype(np.float64)
        modal_mass = np.einsum("s,smc,smc->m", mass, phi64, phi64)
        linear = np.einsum("s,smc->mc", mass, phi64)
        cross = np.cross(phi64, points64[:, None, :])
        angular = np.einsum("s,smc->mc", mass, cross)
        centrifugal = np.einsum("s,smc,sd->mcd", mass, phi64, points64)
        cross_pairs = np.cross(phi64[:, None, :, :], phi64[:, :, None, :])
        coriolis = np.einsum("s,sijc->ijc", mass, cross_pairs)
        return (
            modal_mass.astype(np.float32),
            linear.astype(np.float32),
            angular.astype(np.float32),
            centrifugal.astype(np.float32),
            coriolis.astype(np.float32),
        )

    def copy(self) -> ModalBasis:
        """Return a deep copy of this basis."""
        return ModalBasis(
            sample_points=self.sample_points,
            sample_phi=self.sample_phi,
            sample_psi=self.sample_psi,
            sample_mass=self.sample_mass,
            mode_mass=self.mode_mass,
            mode_stiffness=self.mode_stiffness,
            mode_damping=self.mode_damping,
            mode_coupling_linear=self.mode_coupling_linear,
            mode_coupling_angular=self.mode_coupling_angular,
            mode_coupling_centrifugal=self.mode_coupling_centrifugal,
            mode_coupling_coriolis=self.mode_coupling_coriolis,
            label=self.label,
            interpolation_epsilon=self.interpolation_epsilon,
        )

    def find_sample(self, point: Sequence[float] | np.ndarray, tolerance: float = 1.0e-7) -> int:
        """Return the index of a matching sample point, or ``-1`` if none exists.

        Args:
            point: Body-local point [m], shape ``[3]``.
            tolerance: Matching tolerance [m].

        Returns:
            The local sample index, or ``-1``.
        """
        if self.sample_count == 0:
            return -1
        query = np.asarray(point, dtype=np.float32).reshape((3,))
        dist2 = np.sum((self.sample_points - query.reshape((1, 3))) ** 2, axis=1)
        nearest = int(np.argmin(dist2))
        return nearest if float(dist2[nearest]) <= tolerance * tolerance else -1

    def add_sample(
        self,
        point: Sequence[float] | np.ndarray,
        phi: Sequence[Sequence[float]] | np.ndarray | None = None,
        psi: Sequence[Sequence[float]] | np.ndarray | None = None,
        tolerance: float = 1.0e-7,
    ) -> int:
        """Add a body-local sample and return its local index.

        If ``point`` already exists, its existing index is returned. When ``phi``
        or ``psi`` is not provided, that value is interpolated from the current
        samples.

        Args:
            point: Body-local point [m], shape ``[3]``.
            phi: Translational mode values [m per modal coordinate], shape
                ``[mode_count, 3]``.
            psi: Angular mode values [rad per modal coordinate], shape
                ``[mode_count, 3]``.
            tolerance: Duplicate-point tolerance [m].

        Returns:
            The local sample index.
        """
        phi_batch = None if phi is None else np.asarray(phi, dtype=np.float32).reshape((1, self.mode_count, 3))
        psi_batch = None if psi is None else np.asarray(psi, dtype=np.float32).reshape((1, self.mode_count, 3))
        return int(
            self.add_samples(np.asarray(point, dtype=np.float32).reshape((1, 3)), phi_batch, psi_batch, tolerance)[0]
        )

    def add_samples(
        self,
        points: Sequence[Sequence[float]] | np.ndarray,
        phi: Sequence[Sequence[Sequence[float]]] | np.ndarray | None = None,
        psi: Sequence[Sequence[Sequence[float]]] | np.ndarray | None = None,
        tolerance: float = 1.0e-7,
    ) -> np.ndarray:
        """Add body-local samples in one batch and return their local indices.

        Duplicate points reuse the first matching sample. Missing mode values are
        interpolated from the samples that existed before this call. If the basis
        carries lumped sample masses, appended interpolation samples receive zero
        mass so they do not alter the basis's inertial integrals.

        Args:
            points: Body-local points [m], shape ``[sample_count, 3]``.
            phi: Translational mode values [m per modal coordinate], shape
                ``[sample_count, mode_count, 3]``.
            psi: Angular mode values [rad per modal coordinate], shape
                ``[sample_count, mode_count, 3]``.
            tolerance: Duplicate-point tolerance [m].

        Returns:
            Local sample indices, shape ``[sample_count]``.
        """
        queries = np.asarray(points, dtype=np.float32)
        if queries.ndim == 1:
            queries = queries.reshape((1, 3))
        if queries.ndim != 2 or queries.shape[1] != 3:
            raise ValueError(f"points must have shape [sample_count, 3], got {queries.shape}")
        if tolerance < 0.0:
            raise ValueError(f"tolerance must be non-negative, got {tolerance}")

        query_count = int(queries.shape[0])
        expected_shape = (query_count, self.mode_count, 3)
        phi_values = None if phi is None else np.asarray(phi, dtype=np.float32)
        psi_values = None if psi is None else np.asarray(psi, dtype=np.float32)
        if phi_values is not None and phi_values.shape != expected_shape:
            raise ValueError(f"phi must have shape {expected_shape}, got {phi_values.shape}")
        if psi_values is not None and psi_values.shape != expected_shape:
            raise ValueError(f"psi must have shape {expected_shape}, got {psi_values.shape}")
        if query_count == 0:
            return np.zeros(0, dtype=np.int32)

        all_points = list(self.sample_points)
        result = np.empty(query_count, dtype=np.int32)
        new_query_indices: list[int] = []

        # A spatial hash keeps duplicate detection linear in the common case while
        # retaining the Euclidean tolerance used by find_sample().
        cell_size = tolerance if tolerance > 0.0 else 1.0
        buckets: dict[tuple[int, int, int], list[int]] = {}

        def cell(point: np.ndarray) -> tuple[int, int, int]:
            if tolerance == 0.0:
                return tuple(int(value) for value in point.view(np.int32))
            return tuple(int(value) for value in np.floor(point / cell_size))

        for sample_index, point in enumerate(all_points):
            buckets.setdefault(cell(point), []).append(sample_index)

        neighbor_offsets = (
            [(0, 0, 0)]
            if tolerance == 0.0
            else [(dx, dy, dz) for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)]
        )
        tolerance2 = tolerance * tolerance
        for query_index, query in enumerate(queries):
            query_cell = cell(query)
            nearest = -1
            nearest_dist2 = math.inf
            for dx, dy, dz in neighbor_offsets:
                candidate_cell = (query_cell[0] + dx, query_cell[1] + dy, query_cell[2] + dz)
                for sample_index in buckets.get(candidate_cell, ()):
                    delta = all_points[sample_index] - query
                    dist2 = float(np.dot(delta, delta))
                    if dist2 <= tolerance2 and dist2 < nearest_dist2:
                        nearest = sample_index
                        nearest_dist2 = dist2
            if nearest >= 0:
                result[query_index] = nearest
                continue

            sample_index = len(all_points)
            result[query_index] = sample_index
            new_query_indices.append(query_index)
            all_points.append(query)
            buckets.setdefault(query_cell, []).append(sample_index)

        if not new_query_indices:
            return result

        new_queries = queries[new_query_indices]
        interpolated_phi, interpolated_psi = self._evaluate_many(new_queries)
        new_phi = interpolated_phi if phi_values is None else phi_values[new_query_indices]
        new_psi = interpolated_psi if psi_values is None else psi_values[new_query_indices]
        self.sample_points = np.concatenate((self.sample_points, new_queries), axis=0)
        self.sample_phi = np.concatenate((self.sample_phi, new_phi), axis=0)
        self.sample_psi = np.concatenate((self.sample_psi, new_psi), axis=0)
        if self.sample_mass is not None:
            self.sample_mass = np.concatenate((self.sample_mass, np.zeros(len(new_query_indices), dtype=np.float32)))
        return result

    def _evaluate_many(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Evaluate interpolation points in bounded-memory NumPy batches."""
        query_count = int(points.shape[0])
        if self.mode_count == 0 or self.sample_count == 0:
            zeros = np.zeros((query_count, self.mode_count, 3), dtype=np.float32)
            return zeros, zeros.copy()

        phi = np.empty((query_count, self.mode_count, 3), dtype=np.float32)
        psi = np.empty_like(phi)
        chunk_size = max(1, min(query_count, 1_000_000 // self.sample_count))
        eps2 = self.interpolation_epsilon * self.interpolation_epsilon
        exact_tol2 = max(eps2, 1.0e-14)
        for start in range(0, query_count, chunk_size):
            stop = min(start + chunk_size, query_count)
            delta = self.sample_points[None, :, :] - points[start:stop, None, :]
            dist2 = np.einsum("qsc,qsc->qs", delta, delta)
            nearest = np.argmin(dist2, axis=1)
            exact = dist2[np.arange(stop - start), nearest] <= exact_tol2
            weights = (1.0 / (dist2 + eps2)).astype(np.float32)
            weights /= np.sum(weights, axis=1, keepdims=True)
            phi[start:stop] = np.einsum("qs,smc->qmc", weights, self.sample_phi)
            psi[start:stop] = np.einsum("qs,smc->qmc", weights, self.sample_psi)
            if np.any(exact):
                phi[start:stop][exact] = self.sample_phi[nearest[exact]]
                psi[start:stop][exact] = self.sample_psi[nearest[exact]]
        return phi, psi

    def evaluate(self, point: Sequence[float] | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Inverse-distance interpolate the modal samples at a body-local point.

        Args:
            point: Body-local point [m], shape ``[3]``.

        Returns:
            The translational and angular mode values ``(phi, psi)`` at ``point``,
            each shape ``[mode_count, 3]``, in [m] and [rad] per modal coordinate.
        """
        if self.mode_count == 0:
            return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.float32)
        if self.sample_count == 0:
            return np.zeros((self.mode_count, 3), dtype=np.float32), np.zeros((self.mode_count, 3), dtype=np.float32)

        query = np.asarray(point, dtype=np.float32).reshape((3,))
        delta = self.sample_points - query.reshape((1, 3))
        dist2 = np.sum(delta * delta, axis=1)
        nearest = int(np.argmin(dist2))
        exact_tol2 = max(self.interpolation_epsilon * self.interpolation_epsilon, 1.0e-14)
        if float(dist2[nearest]) <= exact_tol2:
            return (
                np.array(self.sample_phi[nearest], dtype=np.float32, copy=True),
                np.array(self.sample_psi[nearest], dtype=np.float32, copy=True),
            )

        eps2 = self.interpolation_epsilon * self.interpolation_epsilon
        weights = (1.0 / (dist2 + eps2)).astype(np.float32)
        weights /= np.sum(weights)
        return (
            np.einsum("s,smc->mc", weights, self.sample_phi).astype(np.float32),
            np.einsum("s,smc->mc", weights, self.sample_psi).astype(np.float32),
        )


def _estimate_sample_psi(
    sample_points: Sequence[Sequence[float]] | np.ndarray,
    sample_phi: Sequence[Sequence[Sequence[float]]] | np.ndarray,
    neighbor_count: int = 12,
    regularization: float = 1.0e-9,
) -> np.ndarray:
    """Estimate angular mode samples from a translational displacement point cloud.

    The body-local rotation of the material frame is the skew part of the
    displacement gradient, i.e. the half-curl of the mode shape:
    ``psi_i = 0.5 * curl(phi_i)``. For generators that only carry sampled (or
    nodal) displacement values, with no analytic shape or mesh connectivity, this
    fits a local affine displacement gradient over each sample's nearest
    neighbors by least squares and returns the axial vector of its skew part.

    Args:
        sample_points: Body-local sample points [m], shape ``[sample_count, 3]``.
        sample_phi: Translational mode samples [m per modal coordinate], shape
            ``[sample_count, mode_count, 3]``.
        neighbor_count: Number of nearest neighbors used for each local fit.
        regularization: Relative Tikhonov term added to the normal equations so
            that degenerate (collinear or coplanar) neighborhoods stay stable;
            directions the geometry cannot resolve fall back toward zero.

    Returns:
        Angular mode samples [rad per modal coordinate], shape
        ``[sample_count, mode_count, 3]``.
    """
    points = np.asarray(sample_points, dtype=np.float64)
    phi = np.asarray(sample_phi, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"sample_points must have shape [sample_count, 3], got {points.shape}")
    sample_count = points.shape[0]
    if phi.ndim != 3 or phi.shape[0] != sample_count or phi.shape[2] != 3:
        raise ValueError(
            "sample_phi must have shape [sample_count, mode_count, 3], "
            f"got {phi.shape} for {sample_count} sample points"
        )

    mode_count = phi.shape[1]
    psi = np.zeros((sample_count, mode_count, 3), dtype=np.float32)
    if sample_count < 4 or mode_count == 0:
        # An affine 3D gradient needs at least four non-degenerate samples.
        return psi

    k = min(int(neighbor_count), sample_count - 1)
    degenerate = 0
    for s in range(sample_count):
        delta = points - points[s]
        order = np.argsort(np.sum(delta * delta, axis=1))[1 : k + 1]
        dx = delta[order]  # [k, 3]
        gram = dx.T @ dx  # [3, 3]
        scale = np.trace(gram) / 3.0
        if scale <= 0.0:
            continue
        if np.linalg.svd(dx, compute_uv=False)[-1] <= 1.0e-6 * math.sqrt(scale):
            degenerate += 1
        gram_inv = np.linalg.inv(gram + regularization * scale * np.eye(3))
        du = phi[order] - phi[s]  # [k, mode, 3]
        # grad[m] = (du_mᵀ dx)(dxᵀ dx)⁻¹  ->  grad[m, i, j] = d(phi_m)_i / dx_j
        grad = np.einsum("kmi,kj->mij", du, dx) @ gram_inv
        psi[s, :, 0] = 0.5 * (grad[:, 2, 1] - grad[:, 1, 2])
        psi[s, :, 1] = 0.5 * (grad[:, 0, 2] - grad[:, 2, 0])
        psi[s, :, 2] = 0.5 * (grad[:, 1, 0] - grad[:, 0, 1])

    if degenerate:
        warnings.warn(
            f"derive_psi: {degenerate}/{sample_count} sample neighborhoods are rank-deficient "
            "(collinear or coplanar); the unresolved rotation directions are damped toward zero. "
            "Provide a fuller 3D sample distribution for accurate angular coupling.",
            stacklevel=2,
        )
    return psi


class ModalGeneratorSampled:
    """Build a :class:`ModalBasis` from externally supplied sampled modes.

    Args:
        sample_points: Body-local sample points [m], shape ``[sample_count, 3]``.
        sample_phi: Translational mode samples [m per modal coordinate], shape
            ``[sample_count, mode_count, 3]``.
        sample_psi: Optional angular mode samples [rad per modal coordinate],
            shape ``[sample_count, mode_count, 3]``. Pass these when known;
            otherwise set ``derive_psi`` to estimate them from ``sample_phi``.
        mode_mass: Optional per-mode modal mass [kg].
        mode_stiffness: Optional per-mode modal stiffness [N/m].
        mode_damping: Optional per-mode modal damping [N·s/m].
        derive_psi: When ``True`` and ``sample_psi`` is not given, estimate the
            angular samples from the displacement gradient of ``sample_phi``.
        label: Optional basis label.
    """

    def __init__(
        self,
        sample_points: Sequence[Sequence[float]] | np.ndarray,
        sample_phi: Sequence[Sequence[Sequence[float]]] | np.ndarray,
        sample_psi: Sequence[Sequence[Sequence[float]]] | np.ndarray | None = None,
        mode_mass: Sequence[float] | np.ndarray | None = None,
        mode_stiffness: Sequence[float] | np.ndarray | None = None,
        mode_damping: Sequence[float] | np.ndarray | None = None,
        derive_psi: bool = False,
        label: str | None = None,
    ):
        self.sample_points = np.asarray(sample_points, dtype=np.float32)
        self.sample_phi = np.asarray(sample_phi, dtype=np.float32)
        self.sample_psi = None if sample_psi is None else np.asarray(sample_psi, dtype=np.float32)
        self.mode_mass = None if mode_mass is None else np.asarray(mode_mass, dtype=np.float32)
        self.mode_stiffness = None if mode_stiffness is None else np.asarray(mode_stiffness, dtype=np.float32)
        self.mode_damping = None if mode_damping is None else np.asarray(mode_damping, dtype=np.float32)
        self.derive_psi = bool(derive_psi)
        self.label = label

    def build(self) -> ModalBasis:
        """Build the sampled modal basis."""
        sample_psi = self.sample_psi
        if sample_psi is None and self.derive_psi:
            sample_psi = _estimate_sample_psi(self.sample_points, self.sample_phi)
        return ModalBasis(
            sample_points=self.sample_points,
            sample_phi=self.sample_phi,
            sample_psi=sample_psi,
            mode_mass=self.mode_mass,
            mode_stiffness=self.mode_stiffness,
            mode_damping=self.mode_damping,
            label=self.label,
        )


class ModalGeneratorCraigBampton:
    """Build a Newton modal basis from Craig-Bampton reduced matrices.

    The generator accepts already-loaded arrays. Input coordinates must contain
    six DOFs per interface in ``[tx, ty, tz, rx, ry, rz]`` order, followed by
    any fixed-interface modes. A setup-time generalized eigensolve separates the
    six global rigid-body modes and transforms all remaining coordinates into
    ``reduced_dof_count - 6`` mass-normalized elastic modes. The resulting modal
    mass and stiffness are diagonal, so importing dense reduced matrices does not
    add matrix bandwidth to Newton's runtime elastic solve.

    Because :class:`ModalBasis` stores diagonal structural damping, off-diagonal
    entries in the transformed damping matrix are omitted with a warning. Modal,
    Rayleigh, and other damping models diagonalized by the same modes remain
    exact.

    The reduced matrices determine rigid-to-modal linear and angular inertia
    coupling. They do not contain the higher-order centrifugal and Coriolis
    integrals used by Newton's nonlinear floating-frame coupling, so those
    optional terms remain unavailable in the returned basis.

    Args:
        interface_positions: Body-local interface-frame positions [m], shape
            ``[interface_count, 3]``. Their order defines the six-DOF blocks in
            the reduced matrices.
        mass_matrix: Craig-Bampton reduced mass matrix, shape
            ``[reduced_dof_count, reduced_dof_count]``. The first
            ``6 * interface_count`` rows and columns are the interface block.
        stiffness_matrix: Craig-Bampton reduced stiffness matrix, with the same
            shape and coordinate order as ``mass_matrix``.
        sample_points: Body-local recovery sample points [m], shape
            ``[sample_count, 3]``.
        recovery_matrix: Translational recovery matrix mapping reduced
            coordinates to interleaved sample displacements, shape
            ``[3 * sample_count, reduced_dof_count]``.
        damping_matrix: Optional Craig-Bampton reduced damping matrix, with the
            same shape and coordinate order as ``mass_matrix``.
        interface_names: Optional unique names for the interface frames. Defaults
            to ``interface_0``, ``interface_1``, and so on.
        rigid_mode_tolerance: Relative eigenvalue tolerance used to identify the
            six global rigid-body modes and reject additional zero-energy modes.
        damping_coupling_tolerance: Relative off-diagonal damping magnitude above
            which the diagonal approximation emits a warning.
        label: Optional basis label.
    """

    def __init__(
        self,
        interface_positions: Sequence[Sequence[float]] | np.ndarray,
        mass_matrix: Sequence[Sequence[float]] | np.ndarray,
        stiffness_matrix: Sequence[Sequence[float]] | np.ndarray,
        sample_points: Sequence[Sequence[float]] | np.ndarray,
        recovery_matrix: Sequence[Sequence[float]] | np.ndarray,
        damping_matrix: Sequence[Sequence[float]] | np.ndarray | None = None,
        interface_names: Sequence[str] | None = None,
        rigid_mode_tolerance: float = 1.0e-8,
        damping_coupling_tolerance: float = 1.0e-5,
        label: str | None = None,
    ):
        self.interface_positions = np.asarray(interface_positions, dtype=np.float64)
        if self.interface_positions.ndim != 2 or self.interface_positions.shape[1] != 3:
            raise ValueError(
                f"interface_positions must have shape [interface_count, 3], got {self.interface_positions.shape}"
            )
        self.interface_count = int(self.interface_positions.shape[0])
        if self.interface_count < 1:
            raise ValueError("At least one interface is required")

        if interface_names is None:
            names = tuple(f"interface_{i}" for i in range(self.interface_count))
        else:
            names = tuple(str(name) for name in interface_names)
            if len(names) != self.interface_count:
                raise ValueError(f"interface_names must have length {self.interface_count}, got {len(names)}")
            if any(not name for name in names):
                raise ValueError("interface_names cannot contain empty names")
            if len(set(names)) != len(names):
                raise ValueError("interface_names must be unique")
        self.interface_names = names

        mass = np.asarray(mass_matrix, dtype=np.float64)
        if mass.ndim != 2 or mass.shape[0] != mass.shape[1]:
            raise ValueError(f"mass_matrix must be square, got {mass.shape}")
        self.reduced_dof_count = int(mass.shape[0])
        self.interface_dof_count = 6 * self.interface_count
        if self.reduced_dof_count < self.interface_dof_count:
            raise ValueError(
                f"mass_matrix must provide at least {self.interface_dof_count} interface DOFs, "
                f"got {self.reduced_dof_count}"
            )
        self.mass_matrix = self._coerce_square_matrix(mass, self.reduced_dof_count, "mass_matrix")
        self.stiffness_matrix = self._coerce_square_matrix(stiffness_matrix, self.reduced_dof_count, "stiffness_matrix")
        self.damping_matrix = (
            None
            if damping_matrix is None
            else self._coerce_square_matrix(damping_matrix, self.reduced_dof_count, "damping_matrix")
        )

        self.sample_points = np.asarray(sample_points, dtype=np.float64)
        if self.sample_points.ndim != 2 or self.sample_points.shape[1] != 3:
            raise ValueError(f"sample_points must have shape [sample_count, 3], got {self.sample_points.shape}")
        if self.sample_points.shape[0] == 0:
            raise ValueError("At least one recovery sample point is required")
        self.recovery_matrix = np.asarray(recovery_matrix, dtype=np.float64)
        expected_recovery_shape = (3 * self.sample_points.shape[0], self.reduced_dof_count)
        if self.recovery_matrix.shape != expected_recovery_shape:
            raise ValueError(
                f"recovery_matrix must have shape {expected_recovery_shape}, got {self.recovery_matrix.shape}"
            )

        self.rigid_mode_tolerance = float(rigid_mode_tolerance)
        if self.rigid_mode_tolerance <= 0.0:
            raise ValueError(f"rigid_mode_tolerance must be positive, got {self.rigid_mode_tolerance}")
        self.damping_coupling_tolerance = float(damping_coupling_tolerance)
        if self.damping_coupling_tolerance < 0.0:
            raise ValueError(f"damping_coupling_tolerance must be non-negative, got {self.damping_coupling_tolerance}")
        self.label = label

        self.fixed_interface_mode_count = self.reduced_dof_count - self.interface_dof_count
        self.discarded_mode_count = 0
        self.rigid_mode_count = 6
        self.mode_count = self.reduced_dof_count - self.rigid_mode_count
        self.mass = 0.0
        self.com = np.zeros(3, dtype=np.float64)
        self.inertia = np.zeros((3, 3), dtype=np.float64)
        self.spatial_mass = np.zeros((6, 6), dtype=np.float64)
        self.frequencies = np.zeros(0, dtype=np.float64)
        self.rigid_matrix = np.zeros((self.reduced_dof_count, 6), dtype=np.float64)
        self.modal_matrix = np.zeros((self.reduced_dof_count, self.mode_count), dtype=np.float64)
        self.damping_off_diagonal_ratio = 0.0
        self.interface_sample_indices: dict[str, int] = {}

    @staticmethod
    def _coerce_square_matrix(
        matrix: Sequence[Sequence[float]] | np.ndarray,
        size: int,
        name: str,
    ) -> np.ndarray:
        array = np.asarray(matrix, dtype=np.float64)
        if array.shape != (size, size):
            raise ValueError(f"{name} must have shape ({size}, {size}), got {array.shape}")
        return 0.5 * (array + array.T)

    @staticmethod
    def _skew(value: np.ndarray) -> np.ndarray:
        x, y, z = value
        return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=np.float64)

    def _rigid_map(self) -> np.ndarray:
        rigid = np.zeros((self.reduced_dof_count, 6), dtype=np.float64)
        for interface, position in enumerate(self.interface_positions):
            start = 6 * interface
            rigid[start : start + 3, :3] = np.eye(3)
            rigid[start : start + 3, 3:] = -self._skew(position)
            rigid[start + 3 : start + 6, 3:] = np.eye(3)
        return rigid

    def _extract_rigid_properties(self, spatial_mass: np.ndarray) -> None:
        translation_mass = spatial_mass[:3, :3]
        mass = float(np.trace(translation_mass) / 3.0)
        scale = max(float(np.max(np.abs(spatial_mass))), 1.0)
        if mass <= 0.0:
            raise ValueError(f"The Craig-Bampton mass matrix has non-positive rigid mass {mass}")
        if np.max(np.abs(translation_mass - mass * np.eye(3))) > 1.0e-7 * scale:
            raise ValueError("The Craig-Bampton mass matrix does not contain a physical rigid translational block")

        com_skew = -spatial_mass[:3, 3:] / mass
        if np.max(np.abs(com_skew + com_skew.T)) > 1.0e-7 * scale:
            raise ValueError(
                "The Craig-Bampton mass matrix does not contain a physical rigid translation-rotation block"
            )
        com = np.array([com_skew[2, 1], com_skew[0, 2], com_skew[1, 0]], dtype=np.float64)

        inertia_origin = 0.5 * (spatial_mass[3:, 3:] + spatial_mass[3:, 3:].T)
        parallel_axis = mass * ((com @ com) * np.eye(3) - np.outer(com, com))
        inertia = inertia_origin - parallel_axis
        if np.min(np.linalg.eigvalsh(inertia)) < -1.0e-7 * scale:
            raise ValueError("The Craig-Bampton mass matrix produces a non-positive rigid inertia")

        self.mass = mass
        self.com = com
        self.inertia = inertia

    def build(self) -> ModalBasis:
        """Build the mass-normalized elastic modal basis."""
        mass = self.mass_matrix
        stiffness = self.stiffness_matrix
        damping = self.damping_matrix
        rigid = self._rigid_map()

        stiffness_scale = max(float(np.linalg.norm(stiffness, ord=np.inf)), 1.0)
        rigid_scale = max(float(np.linalg.norm(rigid, ord=np.inf)), 1.0)
        rigid_stiffness_ratio = float(np.linalg.norm(stiffness @ rigid, ord=np.inf)) / (stiffness_scale * rigid_scale)
        if rigid_stiffness_ratio > 1.0e-7:
            raise ValueError(
                "The Craig-Bampton stiffness matrix does not preserve the six rigid-body modes "
                f"(relative residual {rigid_stiffness_ratio:.3g})"
            )
        if damping is not None:
            damping_scale = max(float(np.linalg.norm(damping, ord=np.inf)), 1.0)
            rigid_damping_ratio = float(np.linalg.norm(damping @ rigid, ord=np.inf)) / (damping_scale * rigid_scale)
            if rigid_damping_ratio > 1.0e-7:
                raise ValueError(
                    "The Craig-Bampton damping matrix does not preserve the six rigid-body modes "
                    f"(relative residual {rigid_damping_ratio:.3g})"
                )

        try:
            chol = np.linalg.cholesky(mass)
        except np.linalg.LinAlgError as exc:
            raise ValueError("The Craig-Bampton mass matrix must be positive definite") from exc

        inv_chol_stiffness = np.linalg.solve(chol, stiffness)
        standard = np.linalg.solve(chol, inv_chol_stiffness.T).T
        standard = 0.5 * (standard + standard.T)
        eigenvalues, eigenvectors = np.linalg.eigh(standard)
        eigenvalue_scale = max(float(np.max(np.abs(eigenvalues))), 1.0)
        rigid_threshold = self.rigid_mode_tolerance * eigenvalue_scale
        if float(np.max(np.abs(eigenvalues[: self.rigid_mode_count]))) > rigid_threshold:
            raise ValueError(
                "The Craig-Bampton system does not contain exactly six global rigid-body modes "
                f"within relative tolerance {self.rigid_mode_tolerance:.3g}"
            )
        if self.mode_count > 0 and float(eigenvalues[self.rigid_mode_count]) <= rigid_threshold:
            raise ValueError(
                "The Craig-Bampton system contains additional zero-energy modes beyond the six global rigid-body modes"
            )

        modal_matrix = np.linalg.solve(chol.T, eigenvectors[:, self.rigid_mode_count :])
        for mode in range(self.mode_count):
            pivot = int(np.argmax(np.abs(modal_matrix[:, mode])))
            if modal_matrix[pivot, mode] < 0.0:
                modal_matrix[:, mode] *= -1.0

        modal_mass = modal_matrix.T @ mass @ modal_matrix
        modal_stiffness = modal_matrix.T @ stiffness @ modal_matrix
        mode_mass = np.diag(modal_mass)
        mode_stiffness = np.maximum(np.diag(modal_stiffness), 0.0)

        if damping is None:
            mode_damping = np.zeros(self.mode_count, dtype=np.float64)
            self.damping_off_diagonal_ratio = 0.0
        else:
            modal_damping = modal_matrix.T @ damping @ modal_matrix
            modal_damping = 0.5 * (modal_damping + modal_damping.T)
            diagonal = np.diag(modal_damping)
            off_diagonal = modal_damping - np.diag(diagonal)
            if self.mode_count == 0:
                self.damping_off_diagonal_ratio = 0.0
            else:
                diagonal_scale = max(float(np.max(np.abs(diagonal))), np.finfo(np.float64).eps)
                self.damping_off_diagonal_ratio = float(np.max(np.abs(off_diagonal))) / diagonal_scale
            if self.damping_off_diagonal_ratio > self.damping_coupling_tolerance:
                warnings.warn(
                    "Craig-Bampton damping is not diagonal in the retained elastic modal basis; "
                    f"discarding off-diagonal terms with relative magnitude {self.damping_off_diagonal_ratio:.3g}",
                    stacklevel=2,
                )
            mode_damping = np.maximum(diagonal, 0.0)

        sample_rigid = np.hstack(
            (
                np.tile(np.eye(3), (self.sample_points.shape[0], 1)),
                np.vstack([-self._skew(point) for point in self.sample_points]),
            )
        )
        recovery_scale = max(float(np.max(np.abs(sample_rigid))), 1.0)
        recovery_rigid_error = float(np.max(np.abs(self.recovery_matrix @ rigid - sample_rigid)))
        if recovery_rigid_error > 1.0e-6 * recovery_scale:
            raise ValueError(
                "recovery_matrix is inconsistent with rigid motion of sample_points "
                f"(maximum error {recovery_rigid_error:.3g})"
            )

        recovered_modes = self.recovery_matrix @ modal_matrix
        sample_phi = np.transpose(
            recovered_modes.reshape((self.sample_points.shape[0], 3, self.mode_count)),
            (0, 2, 1),
        )
        sample_psi = _estimate_sample_psi(self.sample_points, sample_phi)
        basis_points = np.array(self.sample_points, dtype=np.float32, copy=True)
        basis_phi = np.array(sample_phi, dtype=np.float32, copy=True)
        basis_psi = np.array(sample_psi, dtype=np.float32, copy=True)

        self.interface_sample_indices = {}
        for interface, (name, point) in enumerate(zip(self.interface_names, self.interface_positions, strict=True)):
            start = 6 * interface
            interface_phi = modal_matrix[start : start + 3].T.astype(np.float32)
            interface_psi = modal_matrix[start + 3 : start + 6].T.astype(np.float32)
            distances = np.linalg.norm(basis_points.astype(np.float64) - point, axis=1)
            matching = np.nonzero(distances <= 1.0e-7)[0]
            if matching.shape[0] > 0:
                sample_index = int(matching[0])
                basis_phi[sample_index] = interface_phi
                basis_psi[sample_index] = interface_psi
            else:
                sample_index = int(basis_points.shape[0])
                basis_points = np.vstack((basis_points, point.astype(np.float32)))
                basis_phi = np.concatenate((basis_phi, interface_phi[None]), axis=0)
                basis_psi = np.concatenate((basis_psi, interface_psi[None]), axis=0)
            self.interface_sample_indices[name] = sample_index

        spatial_mass = rigid.T @ mass @ rigid
        self._extract_rigid_properties(spatial_mass)
        rigid_modal_mass = rigid.T @ mass @ modal_matrix
        coupling_linear = rigid_modal_mass[:3].T
        coupling_angular = -rigid_modal_mass[3:].T

        self.spatial_mass = np.array(spatial_mass, dtype=np.float64, copy=True)
        self.rigid_matrix = np.array(rigid, dtype=np.float64, copy=True)
        self.modal_matrix = np.array(modal_matrix, dtype=np.float64, copy=True)
        self.frequencies = np.sqrt(np.maximum(mode_stiffness / np.maximum(mode_mass, 1.0e-12), 0.0)) / (2.0 * math.pi)

        return ModalBasis(
            sample_points=basis_points,
            sample_phi=basis_phi,
            sample_psi=basis_psi,
            mode_mass=mode_mass,
            mode_stiffness=mode_stiffness,
            mode_damping=mode_damping,
            mode_coupling_linear=coupling_linear,
            mode_coupling_angular=coupling_angular,
            label=self.label,
        )


class ModalGeneratorPOD:
    """Build linear modal basis functions with proper orthogonal decomposition.

    Args:
        sample_points: Rest sample points [m], shape ``[sample_count, 3]``.
        displacements: Exemplar displacement fields [m], shape
            ``[snapshot_count, sample_count, 3]``.
        deformed_points: Exemplar deformed points [m], shape
            ``[snapshot_count, sample_count, 3]``. Used only when
            ``displacements`` is not provided.
        mode_count: Number of POD modes to retain.
        total_mass: Total mass [kg] represented by the samples. If provided,
            modal masses are estimated by lumped sample weights.
        stiffness_scale: Heuristic stiffness scale [N/m].
        damping_ratio: Modal damping ratio.
        subtract_mean: Whether to remove the mean exemplar displacement before
            computing POD modes.
        derive_psi: Whether to estimate angular mode samples from the
            displacement gradient of the POD modes, enabling joint rotational
            coupling.
        label: Optional basis label.
    """

    def __init__(
        self,
        sample_points: Sequence[Sequence[float]] | np.ndarray,
        displacements: Sequence[Sequence[Sequence[float]]] | np.ndarray | None = None,
        deformed_points: Sequence[Sequence[Sequence[float]]] | np.ndarray | None = None,
        mode_count: int | None = None,
        total_mass: float | None = None,
        stiffness_scale: float = 1.0,
        damping_ratio: float = 0.0,
        subtract_mean: bool = False,
        derive_psi: bool = False,
        label: str | None = None,
    ):
        self.sample_points = np.asarray(sample_points, dtype=np.float32)
        if self.sample_points.ndim != 2 or self.sample_points.shape[1] != 3:
            raise ValueError(f"sample_points must have shape [sample_count, 3], got {self.sample_points.shape}")

        if displacements is None:
            if deformed_points is None:
                raise ValueError("Either displacements or deformed_points must be provided")
            deformed = np.asarray(deformed_points, dtype=np.float32)
            displacements = deformed - self.sample_points.reshape((1, self.sample_points.shape[0], 3))

        self.displacements = np.asarray(displacements, dtype=np.float32)
        if self.displacements.ndim != 3 or self.displacements.shape[1:] != self.sample_points.shape:
            raise ValueError(
                f"displacements must have shape [snapshot_count, sample_count, 3], got {self.displacements.shape}"
            )

        self.mode_count = mode_count
        self.total_mass = total_mass
        self.stiffness_scale = float(stiffness_scale)
        self.damping_ratio = float(damping_ratio)
        self.subtract_mean = bool(subtract_mean)
        self.derive_psi = bool(derive_psi)
        self.label = label

    def build(self) -> ModalBasis:
        """Build a sampled POD modal basis."""
        snapshot_count = int(self.displacements.shape[0])
        if snapshot_count == 0:
            raise ValueError("At least one displacement snapshot is required")

        matrix = self.displacements.reshape((snapshot_count, -1)).astype(np.float64)
        if self.subtract_mean:
            matrix = matrix - np.mean(matrix, axis=0, keepdims=True)

        _, singular_values, vt = np.linalg.svd(matrix, full_matrices=False)
        mode_count = int(vt.shape[0]) if self.mode_count is None else self.mode_count
        mode_count = min(int(mode_count), int(vt.shape[0]))
        if mode_count < 0:
            raise ValueError(f"mode_count must be non-negative, got {mode_count}")

        modes = vt[:mode_count].reshape((mode_count, self.sample_points.shape[0], 3)).astype(np.float32)
        for i in range(mode_count):
            scale = float(np.max(np.linalg.norm(modes[i], axis=1)))
            if scale > 0.0:
                modes[i] /= scale

        sample_phi = np.transpose(modes, (1, 0, 2))
        sample_psi = _estimate_sample_psi(self.sample_points, sample_phi) if self.derive_psi else None
        mode_mass = self._estimate_mode_mass(sample_phi)
        mode_stiffness = self._estimate_mode_stiffness(mode_count, singular_values)
        mode_damping = np.zeros(mode_count, dtype=np.float32)
        if self.damping_ratio > 0.0:
            for i in range(mode_count):
                mode_damping[i] = (
                    2.0
                    * self.damping_ratio
                    * math.sqrt(max(float(mode_mass[i]), 0.0) * max(float(mode_stiffness[i]), 0.0))
                )

        # Lump the total mass uniformly over the samples (as the modal-mass
        # estimate does) so the basis can form the coupling integrals.
        sample_mass = None
        if self.total_mass is not None and self.sample_points.shape[0] > 0:
            sample_mass = np.full(
                self.sample_points.shape[0],
                float(self.total_mass) / float(self.sample_points.shape[0]),
                dtype=np.float32,
            )

        return ModalBasis(
            sample_points=self.sample_points,
            sample_phi=sample_phi,
            sample_psi=sample_psi,
            sample_mass=sample_mass,
            mode_mass=mode_mass,
            mode_stiffness=mode_stiffness,
            mode_damping=mode_damping,
            label=self.label,
        )

    def _estimate_mode_mass(self, sample_phi: np.ndarray) -> np.ndarray:
        mode_count = int(sample_phi.shape[1])
        if self.total_mass is None or self.sample_points.shape[0] == 0:
            return np.ones(mode_count, dtype=np.float32)

        sample_mass = float(self.total_mass) / float(self.sample_points.shape[0])
        mode_mass = np.zeros(mode_count, dtype=np.float32)
        for i in range(mode_count):
            norm2 = np.sum(sample_phi[:, i, :] * sample_phi[:, i, :], axis=1)
            mode_mass[i] = float(sample_mass * np.sum(norm2))
        return mode_mass

    def _estimate_mode_stiffness(self, mode_count: int, singular_values: np.ndarray) -> np.ndarray:
        if mode_count == 0:
            return np.zeros(0, dtype=np.float32)

        stiffness = np.zeros(mode_count, dtype=np.float32)
        if singular_values.shape[0] > 0 and singular_values[0] > 0.0:
            energy = singular_values[:mode_count] / singular_values[0]
        else:
            energy = np.ones(mode_count, dtype=np.float64)

        for i in range(mode_count):
            # POD modes are ordered by explained displacement energy. Keep the
            # default heuristic gentle so higher-order exemplar modes are not
            # accidentally locked out before users provide calibrated values.
            stiffness[i] = float(self.stiffness_scale * max(energy[i], 1.0e-6) / ((i + 1) * (i + 1)))
        return stiffness


class ModalGeneratorFEM:
    """Build linear modal modes from nodal finite-element matrices.

    This generator solves the generalized eigenvalue problem
    ``K phi = lambda M phi`` on translational nodal DOFs, mass-normalizes the
    selected eigenvectors, and returns a sampled :class:`ModalBasis`.

    Args:
        node_positions: Rest node positions [m], shape ``[node_count, 3]``.
        mass_matrix: Nodal mass matrix [kg], shape
            ``[3 * node_count, 3 * node_count]``.
        stiffness_matrix: Nodal stiffness matrix [N/m], shape
            ``[3 * node_count, 3 * node_count]``.
        damping_matrix: Optional nodal damping matrix [N s/m], shape
            ``[3 * node_count, 3 * node_count]``.
        sample_points: Body-local points [m] where the returned basis is
            sampled. Defaults to ``node_positions``.
        sample_node_indices: Optional node index for each sample point. When
            provided, sample values are copied exactly from those FEM nodes.
            Otherwise values are interpolated from nodal modes.
        mode_count: Number of modes to retain.
        fixed_node_indices: Nodes whose translational DOFs are fixed during the
            eigen solve. Their modal displacement is zero in the returned basis.
        fixed_dof_indices: Individual nodal DOF indices fixed during the solve.
        discard_mode_count: Number of lowest eigenmodes to skip. Use this to
            drop rigid modes in free-free models.
        eigenvalue_tolerance: Minimum eigenvalue [1/s^2] retained after fixed
            DOFs and discarded modes are removed.
        damping_ratio: Modal damping ratio used when ``damping_matrix`` is not
            provided.
        derive_psi: Whether to estimate angular mode samples from the
            displacement gradient of the nodal modes, enabling joint rotational
            coupling.
        label: Optional basis label.
    """

    def __init__(
        self,
        node_positions: Sequence[Sequence[float]] | np.ndarray,
        mass_matrix: Sequence[Sequence[float]] | np.ndarray,
        stiffness_matrix: Sequence[Sequence[float]] | np.ndarray,
        damping_matrix: Sequence[Sequence[float]] | np.ndarray | None = None,
        sample_points: Sequence[Sequence[float]] | np.ndarray | None = None,
        sample_node_indices: Sequence[int] | np.ndarray | None = None,
        mode_count: int | None = None,
        fixed_node_indices: Sequence[int] | np.ndarray | None = None,
        fixed_dof_indices: Sequence[int] | np.ndarray | None = None,
        discard_mode_count: int = 0,
        eigenvalue_tolerance: float = 1.0e-8,
        damping_ratio: float = 0.0,
        derive_psi: bool = False,
        label: str | None = None,
    ):
        self.node_positions = np.asarray(node_positions, dtype=np.float32)
        if self.node_positions.ndim != 2 or self.node_positions.shape[1] != 3:
            raise ValueError(f"node_positions must have shape [node_count, 3], got {self.node_positions.shape}")

        self.node_count = int(self.node_positions.shape[0])
        self.dof_count = 3 * self.node_count
        self.mass_matrix = self._coerce_square_matrix(mass_matrix, self.dof_count, "mass_matrix")
        self.stiffness_matrix = self._coerce_square_matrix(stiffness_matrix, self.dof_count, "stiffness_matrix")
        self.damping_matrix = (
            None
            if damping_matrix is None
            else self._coerce_square_matrix(damping_matrix, self.dof_count, "damping_matrix")
        )

        if sample_points is None:
            self.sample_points = np.array(self.node_positions, dtype=np.float32, copy=True)
        else:
            self.sample_points = np.asarray(sample_points, dtype=np.float32)
            if self.sample_points.ndim != 2 or self.sample_points.shape[1] != 3:
                raise ValueError(f"sample_points must have shape [sample_count, 3], got {self.sample_points.shape}")

        self.sample_node_indices = None
        if sample_node_indices is not None:
            sample_node_indices = np.asarray(sample_node_indices, dtype=np.int32).reshape((-1,))
            if sample_node_indices.shape[0] != self.sample_points.shape[0]:
                raise ValueError(
                    "sample_node_indices must have one entry per sample point, "
                    f"got {sample_node_indices.shape[0]} for {self.sample_points.shape[0]} samples"
                )
            if np.any(sample_node_indices < 0) or np.any(sample_node_indices >= self.node_count):
                raise ValueError("sample_node_indices contains out-of-range node indices")
            self.sample_node_indices = sample_node_indices

        self.mode_count = None if mode_count is None else int(mode_count)
        if self.mode_count is not None and self.mode_count < 0:
            raise ValueError(f"mode_count must be non-negative, got {self.mode_count}")

        fixed: set[int] = set()
        if fixed_node_indices is not None:
            for node_value in np.asarray(fixed_node_indices, dtype=np.int32).reshape((-1,)):
                node = int(node_value)
                if node < 0 or node >= self.node_count:
                    raise ValueError(f"fixed_node_indices contains out-of-range node index {node}")
                fixed.update((3 * node, 3 * node + 1, 3 * node + 2))
        if fixed_dof_indices is not None:
            for dof_value in np.asarray(fixed_dof_indices, dtype=np.int32).reshape((-1,)):
                dof = int(dof_value)
                if dof < 0 or dof >= self.dof_count:
                    raise ValueError(f"fixed_dof_indices contains out-of-range DOF index {dof}")
                fixed.add(dof)
        self.fixed_dof_indices = np.asarray(sorted(fixed), dtype=np.int32)
        self.discard_mode_count = int(discard_mode_count)
        if self.discard_mode_count < 0:
            raise ValueError(f"discard_mode_count must be non-negative, got {self.discard_mode_count}")
        self.eigenvalue_tolerance = float(eigenvalue_tolerance)
        self.damping_ratio = float(damping_ratio)
        self.derive_psi = bool(derive_psi)
        self.label = label

        self.eigenvalues = np.zeros(0, dtype=np.float64)
        self.frequencies = np.zeros(0, dtype=np.float64)
        self.modal_matrix = np.zeros((self.dof_count, 0), dtype=np.float64)

    @staticmethod
    def _coerce_square_matrix(matrix: Sequence[Sequence[float]] | np.ndarray, size: int, name: str) -> np.ndarray:
        array = np.asarray(matrix, dtype=np.float64)
        if array.shape != (size, size):
            raise ValueError(f"{name} must have shape ({size}, {size}), got {array.shape}")
        return 0.5 * (array + array.T)

    def build(self) -> ModalBasis:
        """Build a sampled modal basis."""
        free_dofs = np.ones(self.dof_count, dtype=bool)
        free_dofs[self.fixed_dof_indices] = False
        free = np.nonzero(free_dofs)[0]
        if free.shape[0] == 0:
            raise ValueError("At least one free DOF is required to build FEM modes")

        mass = self.mass_matrix[np.ix_(free, free)]
        stiffness = self.stiffness_matrix[np.ix_(free, free)]
        try:
            chol = np.linalg.cholesky(mass)
        except np.linalg.LinAlgError as exc:
            raise ValueError("mass_matrix must be symmetric positive definite on free DOFs") from exc

        inv_chol_stiffness = np.linalg.solve(chol, stiffness)
        standard = np.linalg.solve(chol, inv_chol_stiffness.T).T
        standard = 0.5 * (standard + standard.T)
        eigenvalues, eigenvectors = np.linalg.eigh(standard)
        order = np.argsort(eigenvalues)
        eigenvalues = eigenvalues[order]
        eigenvectors = eigenvectors[:, order]

        valid = eigenvalues > self.eigenvalue_tolerance
        valid_indices = np.nonzero(valid)[0]
        if self.discard_mode_count:
            valid_indices = valid_indices[self.discard_mode_count :]
        if self.mode_count is not None:
            valid_indices = valid_indices[: self.mode_count]
        if valid_indices.shape[0] == 0:
            raise ValueError("No eigenmodes remain after filtering")

        free_modes = np.linalg.solve(chol.T, eigenvectors[:, valid_indices])
        full_modes = np.zeros((self.dof_count, valid_indices.shape[0]), dtype=np.float64)
        full_modes[free, :] = free_modes

        mode_mass = np.zeros(valid_indices.shape[0], dtype=np.float64)
        mode_stiffness = np.zeros(valid_indices.shape[0], dtype=np.float64)
        mode_damping = np.zeros(valid_indices.shape[0], dtype=np.float64)
        for i in range(valid_indices.shape[0]):
            phi = full_modes[:, i]
            modal_mass = float(phi @ self.mass_matrix @ phi)
            if modal_mass <= 0.0:
                raise ValueError(f"Mode {i} has non-positive modal mass {modal_mass}")
            scale = 1.0 / math.sqrt(modal_mass)
            full_modes[:, i] *= scale
            phi = full_modes[:, i]
            mode_mass[i] = float(phi @ self.mass_matrix @ phi)
            mode_stiffness[i] = max(float(phi @ self.stiffness_matrix @ phi), 0.0)
            if self.damping_matrix is not None:
                mode_damping[i] = max(float(phi @ self.damping_matrix @ phi), 0.0)
            elif self.damping_ratio > 0.0:
                mode_damping[i] = 2.0 * self.damping_ratio * math.sqrt(mode_mass[i] * mode_stiffness[i])

        nodal_phi = np.transpose(full_modes.reshape((self.node_count, 3, -1)), (0, 2, 1))
        nodal_psi = _estimate_sample_psi(self.node_positions, nodal_phi) if self.derive_psi else None
        if self.sample_node_indices is None:
            nodal_basis = ModalBasis(
                sample_points=self.node_positions,
                sample_phi=nodal_phi,
                sample_psi=nodal_psi,
                mode_mass=mode_mass,
                mode_stiffness=mode_stiffness,
                mode_damping=mode_damping,
                label=self.label,
            )
            evaluated = [nodal_basis.evaluate(point) for point in self.sample_points]
            sample_phi = np.asarray([phi for phi, _ in evaluated], dtype=np.float32)
            sample_psi = np.asarray([psi for _, psi in evaluated], dtype=np.float32) if self.derive_psi else None
        else:
            sample_phi = nodal_phi[self.sample_node_indices]
            sample_psi = None if nodal_psi is None else nodal_psi[self.sample_node_indices]

        self.eigenvalues = mode_stiffness / np.maximum(mode_mass, 1.0e-12)
        self.frequencies = np.sqrt(np.maximum(self.eigenvalues, 0.0)) / (2.0 * math.pi)
        self.modal_matrix = np.array(full_modes, dtype=np.float64, copy=True)

        # Exact inertia coupling integrals from the consistent mass matrix
        translation_modes = np.tile(np.eye(3, dtype=np.float64), (self.node_count, 1))
        rotation_modes = np.zeros((self.dof_count, 3), dtype=np.float64)
        for j in range(self.node_count):
            position = self.node_positions[j].astype(np.float64)
            for axis in range(3):
                unit = np.zeros(3, dtype=np.float64)
                unit[axis] = 1.0
                rotation_modes[3 * j : 3 * j + 3, axis] = np.cross(unit, position)
        mass_modes = self.mass_matrix @ full_modes
        coupling_linear = (mass_modes.T @ translation_modes).astype(np.float32)
        coupling_angular = (-mass_modes.T @ rotation_modes).astype(np.float32)
        mass_modes_n = mass_modes.T.reshape((-1, self.node_count, 3))  # [mode, node, 3]
        node_positions = self.node_positions.astype(np.float64)
        coupling_centrifugal = np.einsum("isc,sd->icd", mass_modes_n, node_positions).astype(np.float32)
        mass_phi = mass_modes_n.transpose(1, 0, 2)  # [node, mode, 3]
        nodal_phi64 = nodal_phi.astype(np.float64)
        coupling_coriolis = np.cross(nodal_phi64[:, None, :, :], mass_phi[:, :, None, :]).sum(axis=0).astype(np.float32)

        return ModalBasis(
            sample_points=self.sample_points,
            sample_phi=sample_phi,
            sample_psi=sample_psi,
            mode_mass=mode_mass,
            mode_stiffness=mode_stiffness,
            mode_damping=mode_damping,
            mode_coupling_linear=coupling_linear,
            mode_coupling_angular=coupling_angular,
            mode_coupling_centrifugal=coupling_centrifugal,
            mode_coupling_coriolis=coupling_coriolis,
            label=self.label,
        )


class ModalGeneratorBeam:
    """Build sampled linear modes for a straight Euler-Bernoulli beam.

    The beam lies along local ``x`` with its midpoint at the body origin.

    Args:
        length: Beam length [m].
        half_width_y: Half width in local ``y`` [m].
        half_width_z: Half width in local ``z`` [m].
        mode_specs: Sequence of mode dictionaries. Supported ``type`` values are
            ``"axial"``, ``"bending_y"``, ``"bending_z"``, and ``"torsion"``.
        sample_count: Number of samples along the beam length.
        density: Density [kg/m^3].
        young_modulus: Young's modulus [Pa].
        shear_modulus: Shear modulus [Pa]. Defaults to ``young_modulus / 2.6``.
        area: Cross-section area [m^2]. Defaults to the rectangular section.
        area_moment_y: Second moment about local ``y`` [m^4].
        area_moment_z: Second moment about local ``z`` [m^4].
        polar_moment: Polar second moment [m^4].
        damping_ratio: Modal damping ratio.
        label: Optional basis label.
    """

    class Mode:
        """Beam mode type names."""

        AXIAL = "axial"
        BENDING_Y = "bending_y"
        BENDING_Z = "bending_z"
        TORSION = "torsion"

    class Boundary:
        """Beam boundary names used by mode specifications."""

        PINNED_PINNED = "pinned-pinned"
        CANTILEVER_TIP = "cantilever-tip"
        FIXED_FREE = "fixed-free"
        LINEAR = "linear"

    def __init__(
        self,
        length: float,
        half_width_y: float = 0.0,
        half_width_z: float = 0.0,
        mode_specs: Sequence[dict[str, Any] | str] | None = None,
        sample_count: int = 33,
        density: float = 1000.0,
        young_modulus: float = 1.0e6,
        shear_modulus: float | None = None,
        area: float | None = None,
        area_moment_y: float | None = None,
        area_moment_z: float | None = None,
        polar_moment: float | None = None,
        damping_ratio: float = 0.0,
        label: str | None = None,
    ):
        if length <= 0.0:
            raise ValueError(f"length must be positive, got {length}")
        self.length = float(length)
        self.half_width_y = float(half_width_y)
        self.half_width_z = float(half_width_z)
        self.mode_specs = [self._coerce_mode_spec(spec) for spec in (mode_specs or [self.Mode.BENDING_Z])]
        self.sample_count = max(2, int(sample_count))
        self.density = float(density)
        self.young_modulus = float(young_modulus)
        self.shear_modulus = float(shear_modulus) if shear_modulus is not None else self.young_modulus / 2.6
        self.area = float(area) if area is not None else max(4.0 * self.half_width_y * self.half_width_z, 0.0)
        self.area_moment_y = (
            float(area_moment_y)
            if area_moment_y is not None
            else (4.0 / 3.0) * self.half_width_y * self.half_width_z**3
        )
        self.area_moment_z = (
            float(area_moment_z)
            if area_moment_z is not None
            else (4.0 / 3.0) * self.half_width_z * self.half_width_y**3
        )
        self.polar_moment = (
            float(polar_moment) if polar_moment is not None else max(self.area_moment_y + self.area_moment_z, 0.0)
        )
        self.damping_ratio = float(damping_ratio)
        self.label = label

    @staticmethod
    def _coerce_mode_spec(spec: dict[str, Any] | str) -> dict[str, Any]:
        if isinstance(spec, str):
            return {"type": spec}
        result = dict(spec)
        if "type" not in result:
            raise ValueError("Each beam mode spec must include a 'type' value")
        order = int(result.get("order", 1))
        if order < 1:
            raise ValueError(f"Beam mode order must be at least 1, got {order}")
        return result

    def build(self, sample_points: Sequence[Sequence[float]] | np.ndarray | None = None) -> ModalBasis:
        """Build a sampled beam modal basis.

        Args:
            sample_points: Optional body-local sample points [m], shape
                ``[sample_count, 3]``. If omitted, centerline and rectangular
                cross-section samples are generated.

        Returns:
            A sampled modal basis.
        """
        points = self._sample_points() if sample_points is None else np.asarray(sample_points, dtype=np.float32)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError(f"sample_points must have shape [sample_count, 3], got {points.shape}")

        sample_phi = np.zeros((points.shape[0], len(self.mode_specs), 3), dtype=np.float32)
        sample_psi = np.zeros((points.shape[0], len(self.mode_specs), 3), dtype=np.float32)
        for sample_index, point in enumerate(points):
            for mode_index, spec in enumerate(self.mode_specs):
                phi, psi = self._evaluate_mode(spec, point)
                sample_phi[sample_index, mode_index] = phi
                sample_psi[sample_index, mode_index] = psi

        mode_mass = np.zeros(len(self.mode_specs), dtype=np.float32)
        mode_stiffness = np.zeros(len(self.mode_specs), dtype=np.float32)
        mode_damping = np.zeros(len(self.mode_specs), dtype=np.float32)
        for i, spec in enumerate(self.mode_specs):
            mode_mass[i], mode_stiffness[i] = self._mode_properties(spec)
            if self.damping_ratio > 0.0:
                mode_damping[i] = (
                    2.0
                    * self.damping_ratio
                    * math.sqrt(max(float(mode_mass[i]), 0.0) * max(float(mode_stiffness[i]), 0.0))
                )

        return ModalBasis(
            sample_points=points,
            sample_phi=sample_phi,
            sample_psi=sample_psi,
            mode_mass=mode_mass,
            mode_stiffness=mode_stiffness,
            mode_damping=mode_damping,
            label=self.label,
        )

    def _sample_points(self) -> np.ndarray:
        yz: list[tuple[float, float]] = [(0.0, 0.0)]
        if self.half_width_y > 0.0:
            yz.extend([(-self.half_width_y, 0.0), (self.half_width_y, 0.0)])
        if self.half_width_z > 0.0:
            yz.extend([(0.0, -self.half_width_z), (0.0, self.half_width_z)])
        if self.half_width_y > 0.0 and self.half_width_z > 0.0:
            yz.extend(
                [
                    (-self.half_width_y, -self.half_width_z),
                    (self.half_width_y, -self.half_width_z),
                    (self.half_width_y, self.half_width_z),
                    (-self.half_width_y, self.half_width_z),
                ]
            )

        unique_yz = list(dict.fromkeys(yz))
        points: list[tuple[float, float, float]] = []
        for x in np.linspace(-0.5 * self.length, 0.5 * self.length, self.sample_count):
            for y, z in unique_yz:
                points.append((float(x), y, z))
        return np.asarray(points, dtype=np.float32)

    def _evaluate_mode(self, spec: dict[str, Any], point: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        mode_type = str(spec["type"])
        order = int(spec.get("order", 1))
        boundary = str(spec.get("boundary", self._default_boundary(mode_type)))
        x = float(point[0])
        y = float(point[1])
        z = float(point[2])
        s = min(max(x + 0.5 * self.length, 0.0), self.length)

        if mode_type == self.Mode.AXIAL:
            phi = np.array([x / self.length, 0.0, 0.0], dtype=np.float32)
            return phi, np.zeros(3, dtype=np.float32)

        if mode_type == self.Mode.BENDING_Y:
            phi, slope = self._bending_shape(s, order, boundary)
            return np.array([-y * slope, phi, 0.0], dtype=np.float32), np.array([0.0, 0.0, slope], dtype=np.float32)

        if mode_type == self.Mode.BENDING_Z:
            phi, slope = self._bending_shape(s, order, boundary)
            return np.array([-z * slope, 0.0, phi], dtype=np.float32), np.array([0.0, -slope, 0.0], dtype=np.float32)

        if mode_type == self.Mode.TORSION:
            theta, _ = self._torsion_shape(s, order, boundary)
            return np.array([0.0, -z * theta, y * theta], dtype=np.float32), np.array(
                [theta, 0.0, 0.0], dtype=np.float32
            )

        raise ValueError(f"Unsupported beam mode type '{mode_type}'")

    def _default_boundary(self, mode_type: str) -> str:
        if mode_type == self.Mode.TORSION:
            return self.Boundary.FIXED_FREE
        if mode_type == self.Mode.AXIAL:
            return self.Boundary.LINEAR
        return self.Boundary.CANTILEVER_TIP

    def _bending_shape(self, s: float, order: int, boundary: str) -> tuple[float, float]:
        if boundary == self.Boundary.PINNED_PINNED:
            k = float(order) * math.pi / self.length
            return math.sin(k * s), k * math.cos(k * s)

        if boundary == self.Boundary.CANTILEVER_TIP:
            if order != 1:
                k = (float(order) - 0.5) * math.pi / self.length
                tip = math.sin(k * self.length)
                scale = 1.0 / tip if abs(tip) > 1.0e-8 else 1.0
                return scale * math.sin(k * s), scale * k * math.cos(k * s)

            phi = (s * s * (3.0 * self.length - s)) / (2.0 * self.length**3)
            slope = (3.0 * s * (2.0 * self.length - s)) / (2.0 * self.length**3)
            return phi, slope

        raise ValueError(f"Unsupported bending boundary '{boundary}'")

    def _torsion_shape(self, s: float, order: int, boundary: str) -> tuple[float, float]:
        if boundary == self.Boundary.LINEAR:
            power = max(1, order)
            xi = s / self.length
            theta = xi**power
            dtheta = float(power) * xi ** (power - 1) / self.length
            return theta, dtheta

        if boundary == self.Boundary.FIXED_FREE:
            k = (float(order) - 0.5) * math.pi / self.length
            tip = math.sin(k * self.length)
            scale = 1.0 / tip if abs(tip) > 1.0e-8 else 1.0
            return scale * math.sin(k * s), scale * k * math.cos(k * s)

        raise ValueError(f"Unsupported torsion boundary '{boundary}'")

    def _mode_properties(self, spec: dict[str, Any]) -> tuple[float, float]:
        mode_type = str(spec["type"])
        order = int(spec.get("order", 1))
        boundary = str(spec.get("boundary", self._default_boundary(mode_type)))
        mass_per_length = self.density * self.area

        if mode_type == self.Mode.AXIAL:
            mass = mass_per_length * self.length / 12.0
            stiffness = self.young_modulus * self.area / self.length if self.area > 0.0 else 0.0
            return mass, stiffness

        if mode_type in (self.Mode.BENDING_Y, self.Mode.BENDING_Z):
            inertia = self.area_moment_z if mode_type == self.Mode.BENDING_Y else self.area_moment_y
            ei = self.young_modulus * inertia
            if boundary == self.Boundary.PINNED_PINNED:
                k = float(order) * math.pi / self.length
                mass = mass_per_length * self.length / 2.0
                stiffness = ei * k**4 * self.length / 2.0
                return mass, stiffness
            if boundary == self.Boundary.CANTILEVER_TIP and order == 1:
                mass = mass_per_length * self.length * (33.0 / 140.0)
                stiffness = 3.0 * ei / (self.length**3)
                return mass, stiffness
            k = (float(order) - 0.5) * math.pi / self.length
            mass = mass_per_length * self.length / 2.0
            stiffness = ei * k**4 * self.length / 2.0
            return mass, stiffness

        if mode_type == self.Mode.TORSION:
            rotary_mass = self.density * self.polar_moment
            gj = self.shear_modulus * self.polar_moment
            if boundary == self.Boundary.LINEAR:
                mass = rotary_mass * self.length / float(2 * order + 1)
                stiffness = gj * float(order * order) / (self.length * float(2 * order - 1))
                return mass, stiffness
            k = (float(order) - 0.5) * math.pi / self.length
            mass = rotary_mass * self.length / 2.0
            stiffness = gj * k * k * self.length / 2.0
            return mass, stiffness

        raise ValueError(f"Unsupported beam mode type '{mode_type}'")


__all__ = [
    "ModalBasis",
    "ModalGeneratorBeam",
    "ModalGeneratorCraigBampton",
    "ModalGeneratorFEM",
    "ModalGeneratorPOD",
    "ModalGeneratorSampled",
]
