# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""GPU-resident parameter adapter for candidate friction evaluation in Cartesian Engine.

This adapter hooks into MidsoleFoundation via foundation.friction_solver. It provides
the zero-stiffness and scratch history arrays so the foundation's normal mechanics
run unmodified, then executes a postnormal GPU kernel to apply per-world friction
parameters (scaling baseline kt and kv) using canonical bristle_step (method 0, legacy)
or bristle_deflection_step (method 1, deflection). Normal forces and material laws remain untouched.
"""

from __future__ import annotations

import numpy as np
import warp as wp

from projects.digital_shoe.contact import bristle_step, contact_kinematics
from projects.digital_shoe.friction_deflection import bristle_deflection_step
from projects.digital_shoe.friction_maxwell import (
    bristle_elastic_coulomb_step,
    bristle_maxwell_step,
    column_maxwell_parameters,
    elastic_coulomb_stiffness,
    maxwell_bristle_energy,
)
from projects.digital_shoe.friction_pressure import bristle_pressure_step, pressure_coefficient
from projects.digital_shoe.friction_slip_history import bristle_slip_history_step, slip_history_coefficient
from projects.digital_shoe.friction_stribeck import bristle_stribeck_step, stribeck_coefficient
from projects.digital_shoe.runtime import FoundationParams

wp.set_module_options({"enable_backward": False})

SUPPORTED_METHODS = {
    0: "legacy",
    1: "deflection",
    4: "stribeck",
    5: "pressure",
    6: "slip_history",
    7: "maxwell",
    8: "column_maxwell",
    9: "elastic_coulomb",
}


@wp.kernel
def _postnormal_friction_step(
    column_count: int,
    plane: int,
    ground_height: float,
    carrier: wp.array[int],
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    body_com: wp.array[wp.vec3],
    anchor_local: wp.array[wp.vec3],
    ground_force: wp.array[wp.vec3],
    column_force: wp.array[wp.vec3],
    settings: wp.array2d[float],
    base_kt: wp.array[float],
    base_kv: wp.array[float],
    area: wp.array[float],
    rest_len: wp.array[float],
    world_params: wp.array[FoundationParams],
    tangent_anchor: wp.array[wp.vec2],
    tangent_stuck: wp.array[int],
    tangent_dwell: wp.array[float],
    deflection: wp.array[wp.vec2],
    sliding_distance: wp.array[float],
    maxwell_force: wp.array[wp.vec2],
    stored_energy: wp.array[float],
    column_diagnostics: wp.array[wp.vec4],
    step_diagnostics: wp.array[wp.vec4],
    dt: float,
):
    """Evaluate candidate tangential friction and track physical diagnostics."""
    i = wp.tid()
    w = i // column_count
    c = i % column_count
    body = carrier[w]

    point, _com, point_vel, _gap = contact_kinematics(
        body_q[body], body_qd[body], body_com[body], anchor_local[c], ground_height, plane
    )

    pos = wp.vec2(point[0], point[1])
    v = wp.vec2(point_vel[0], point_vel[1])

    # Normal contact force magnitude from foundation solve
    n = ground_force[i][2] if plane != 0 else wp.max(column_force[i][2], 0.0)

    method = int(settings[w, 0])
    mu = settings[w, 1]
    kt = base_kt[c] * settings[w, 2]
    kv = base_kv[c] * settings[w, 3]
    gamma = settings[w, 4]
    release = settings[w, 5]
    width = settings[w, 6]

    force = wp.vec2(0.0)
    z = deflection[i]
    a = tangent_anchor[i]
    s = tangent_stuck[i]
    elapsed = tangent_dwell[i]
    old_energy = stored_energy[i]

    if method == 9:
        p = world_params[w]
        kt = elastic_coulomb_stiffness(p.g_eq + p.g_eq2, area[c], rest_len[c]) * settings[w, 2]
        force, _jac, z, s, elapsed = bristle_elastic_coulomb_step(v, dt, n, kt, mu, release, z, s, elapsed)
        maxwell_force[i] = wp.vec2(0.0)
        a = pos + dt * v - z
    elif method == 7:
        shear_tau = settings[w, 11]
        force, _jac, z, q, s, elapsed = bristle_maxwell_step(
            v,
            dt,
            n,
            kt,
            kv,
            shear_tau,
            mu,
            release,
            z,
            maxwell_force[i],
            s,
            elapsed,
        )
        maxwell_force[i] = q
        a = pos + dt * v - z
    elif method == 8:
        p = world_params[w]
        shear_tau = p.friction_relaxation_time_s
        kt, kv = column_maxwell_parameters(
            p.g_eq + p.g_eq2,
            p.overstress,
            area[c],
            rest_len[c],
            shear_tau,
        )
        kt *= settings[w, 2]
        kv *= settings[w, 3]
        force, _jac, z, q, s, elapsed = bristle_maxwell_step(
            v, dt, n, kt, kv, shear_tau, mu, release, z, maxwell_force[i], s, elapsed
        )
        maxwell_force[i] = q
        a = pos + dt * v - z
    elif method == 6:
        force, z, s, elapsed, distance = bristle_slip_history_step(
            v,
            dt,
            n,
            kt,
            kv,
            mu,
            settings[w, 7],
            settings[w, 10],
            gamma,
            release,
            z,
            s,
            elapsed,
            sliding_distance[i],
        )
        sliding_distance[i] = distance
        mu = slip_history_coefficient(distance, mu, settings[w, 7], settings[w, 10])
        a = pos + dt * v - z
    elif method == 5:
        force, _jac, z, s, elapsed = bristle_pressure_step(
            v, dt, n, area[c], kt, kv, mu, settings[w, 9], gamma, release, z, s, elapsed
        )
        mu = pressure_coefficient(n, area[c], mu, settings[w, 9])
        a = pos + dt * v - z
    elif method == 4:
        force, _jac, z, s, elapsed = bristle_stribeck_step(
            v, dt, n, kt, kv, mu, settings[w, 7], settings[w, 8], gamma, release, z, s, elapsed
        )
        mu = stribeck_coefficient(v, mu, settings[w, 7], settings[w, 8])
        a = pos + dt * v - z
    elif method == 1:
        # Consistent deflection tracking
        force, _jac, z, s, elapsed = bristle_deflection_step(v, dt, n, kt, kv, mu, gamma, release, width, z, s, elapsed)
        if s != 0:
            a = pos + dt * v - z
        else:
            a = pos
    else:
        # Method 0: Canonical anchored bristle
        force, a, s, elapsed = bristle_step(pos, v, dt, n, kt, kv, mu, gamma, release, a, s, elapsed)
        if n > 0.0 and kt > 0.0:
            z = pos + dt * v - a
        elif s == 0:
            z = wp.vec2(0.0)

    # Elastic stored energy and mechanical dissipation diagnostics
    energy = 0.5 * kt * wp.dot(z, z)
    if method == 7:
        energy = maxwell_bristle_energy(z, maxwell_force[i], kt, kv, settings[w, 11])
    elif method == 8:
        energy = maxwell_bristle_energy(z, maxwell_force[i], kt, kv, world_params[w].friction_relaxation_time_s)
    work = wp.dot(force, v) * dt
    residual = energy - old_energy + work
    tolerance = 1.0e-6 * (energy + old_energy + wp.abs(work) + 1.0e-6)
    violation = wp.max(residual - tolerance, 0.0)
    cone_excess = wp.max(wp.length(force) - mu * wp.max(n, 0.0), 0.0)

    # Write ONLY tangential XY, preserving normal Z
    column_force[i] = wp.vec3(force[0], force[1], column_force[i][2])
    if plane != 0:
        ground_force[i] = wp.vec3(force[0], force[1], ground_force[i][2])

    tangent_anchor[i] = a
    tangent_stuck[i] = s
    tangent_dwell[i] = elapsed
    deflection[i] = z
    stored_energy[i] = energy

    # Per-column cumulative diagnostics
    diag = column_diagnostics[i]
    column_diagnostics[i] = wp.vec4(
        diag[0] + violation,
        wp.max(diag[1], wp.length(z)),
        wp.max(diag[2], cone_excess),
        diag[3] + work,
    )

    # Step-wise diagnostic for reduction into totals
    step_diagnostics[i] = wp.vec4(violation, wp.length(z), cone_excess, work)


@wp.kernel
def _reduce_column_diagnostics(
    count: int,
    groups: int,
    step_diagnostics: wp.array[wp.vec4],
    partial_diagnostics: wp.array[wp.vec4],
):
    """Reduce column diagnostics across stride groups."""
    i = wp.tid()
    w = i // groups
    group = i % groups
    d = wp.vec4(0.0)
    for c in range(group, count, groups):
        k = w * count + c
        val = step_diagnostics[k]
        d = wp.vec4(
            d[0] + val[0],
            wp.max(d[1], val[1]),
            wp.max(d[2], val[2]),
            d[3] + val[3],
        )
    partial_diagnostics[i] = d


@wp.kernel
def _finish_world_diagnostics(
    groups: int,
    partial_diagnostics: wp.array[wp.vec4],
    totals: wp.array[wp.vec4],
):
    """Accumulate reduced group diagnostics into per-world running totals."""
    w = wp.tid()
    d = totals[w]
    for group in range(groups):
        i = w * groups + group
        val = partial_diagnostics[i]
        d = wp.vec4(
            d[0] + val[0],
            wp.max(d[1], val[1]),
            wp.max(d[2], val[2]),
            d[3] + val[3],
        )
    totals[w] = d


class FrictionParameterAdapter:
    """GPU-resident adapter injecting per-world friction parameters into MidsoleFoundation.

    Hooks into foundation.friction_solver. Normal mechanics, contact points, and
    compression kinematics remain unmodified.

    Args:
        foundation: MidsoleFoundation instance from Cartesian Engine.
        world_count: Number of parallel simulation worlds.
        base_kt: Base per-column tangential stiffness array [N/m].
        base_kv: Base per-column tangential damping array [N*s/m].
    """

    def __init__(
        self,
        foundation,
        world_count: int,
        base_kt: np.ndarray | wp.array | None = None,
        base_kv: np.ndarray | wp.array | None = None,
        *,
        is_default: bool = False,
        initial_parameters: np.ndarray | list | None = None,
        deflection: wp.array | None = None,
        maxwell_force: wp.array | None = None,
    ):
        if world_count != foundation.world_count or world_count < 1:
            raise ValueError("Adapter worlds must match the foundation")
        current_solver = getattr(foundation, "friction_solver", None)
        if current_solver is not None:
            if not getattr(current_solver, "is_default", False) or is_default:
                raise ValueError("Foundation already has a friction adapter")

        # Baseline per-column stiffness and damping validation
        column_count = int(foundation.column_count)
        if base_kt is None:
            base_kt = foundation.friction_kt.numpy().copy()
        if base_kv is None:
            base_kv = foundation.friction_kv.numpy().copy()
        kt_values = base_kt.numpy() if isinstance(base_kt, wp.array) else np.asarray(base_kt)
        kv_values = base_kv.numpy() if isinstance(base_kv, wp.array) else np.asarray(base_kv)
        for name, values in (("base_kt", kt_values), ("base_kv", kv_values)):
            if values.shape != (column_count,) or not np.isfinite(values).all() or np.any(values < 0):
                raise ValueError(f"{name} must be finite nonnegative per-column values")

        self.is_default = bool(is_default)
        device = foundation.compression.device
        self.foundation = foundation
        self.world_count = int(world_count)
        self.column_count = column_count
        self.device = device
        n = self.world_count * self.column_count

        # Disposable buffers required by MidsoleFoundation when friction_solver is active
        self.zero_stiffness = wp.zeros(self.column_count, dtype=float, device=device)
        self.scratch_anchor = wp.zeros(n, dtype=wp.vec2, device=device)
        self.scratch_stuck = wp.zeros(n, dtype=int, device=device)
        self.scratch_dwell = wp.zeros(n, dtype=float, device=device)

        # Deflection and physical history tracked by adapter
        if deflection is not None:
            self.deflection = deflection
        elif hasattr(foundation, "tangent_deflection") and foundation.tangent_deflection is not None:
            self.deflection = foundation.tangent_deflection
        else:
            self.deflection = wp.zeros(n, dtype=wp.vec2, device=device)

        self.sliding_distance = wp.zeros(n, dtype=float, device=device)

        if maxwell_force is not None:
            self.maxwell_force = maxwell_force
        elif hasattr(foundation, "tangent_maxwell_force") and foundation.tangent_maxwell_force is not None:
            self.maxwell_force = foundation.tangent_maxwell_force
        else:
            self.maxwell_force = wp.zeros(n, dtype=wp.vec2, device=device)

        self.stored_energy = wp.zeros(n, dtype=float, device=device)
        self.column_diagnostics = wp.zeros(n, dtype=wp.vec4, device=device)

        # Reduction buffers for totals
        self.groups = min(self.column_count, 16)
        self.step_diagnostics = wp.zeros(n, dtype=wp.vec4, device=device)
        self.partial_diagnostics = wp.zeros(self.world_count * self.groups, dtype=wp.vec4, device=device)
        self.totals = wp.zeros(self.world_count, dtype=wp.vec4, device=device)

        self.base_kt = wp.array(kt_values, dtype=float, device=device)
        self.base_kv = wp.array(kv_values, dtype=float, device=device)

        # Per-world candidate parameters
        self.settings = wp.zeros((self.world_count, 12), dtype=float, device=device)
        if initial_parameters is not None:
            self.set_parameters(initial_parameters)
        elif not is_default:
            default_params = np.array([[0.0, 0.8, 1.0, 1.0, 0.2, 0.0005, 0.0]], dtype=np.float32)
            self.set_parameters(default_params)

        # Only detach auto-default after all validation and allocations succeed
        if current_solver is not None and getattr(current_solver, "is_default", False):
            current_solver.detach(restore_default=False)

        foundation.tangent_deflection = self.deflection
        foundation.tangent_maxwell_force = self.maxwell_force
        foundation.friction_solver = self

    def set_parameters(self, parameters: np.ndarray | list) -> None:
        """Set candidate friction parameters across worlds.

        Args:
            parameters: Array of shape (K, 7) or (7,) where rows contain:
                [method, mu, kt_scale, kv_scale, viscous_ratio, release_dwell_s, yield_width].
                Rows are padded/replicated to fill world_count if K < world_count.
        """
        arr = np.asarray(parameters, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr[None, :]
        if arr.ndim != 2 or arr.shape[1] not in (7, 9, 10, 11, 12):
            raise ValueError(f"Parameters must have seven to twelve supported fields, got {arr.shape}")
        if arr.shape[1] == 7:
            arr = np.column_stack((arr, arr[:, 1], np.full(len(arr), 0.1, np.float32)))
        if arr.shape[1] == 9:
            arr = np.column_stack((arr, np.full(len(arr), 1.0e12, np.float32)))
        if arr.shape[1] == 10:
            arr = np.column_stack((arr, np.full(len(arr), 0.001, np.float32)))
        if arr.shape[1] == 11:
            arr = np.column_stack((arr, np.full(len(arr), 0.005, np.float32)))
        if len(arr) == 0 or not np.isfinite(arr).all():
            raise ValueError("Candidate parameters must be nonempty and finite")
        if np.any(arr[:, 0] != arr[:, 0].astype(int)):
            raise ValueError("Friction methods must be exact integer identifiers")
        if len(arr) > self.world_count:
            raise ValueError(f"Parameter row count ({len(arr)}) exceeds world count ({self.world_count})")

        # Validate all supported methods before assigning world parameters.
        methods = arr[:, 0].astype(int)
        for m in methods:
            if m not in SUPPORTED_METHODS:
                raise ValueError(
                    f"Friction method {m} is unsupported. Supported method codes are 0, 1, 4, 5, 6, 7, 8 and 9. Diagnostic methods 2/3 are rejected."
                )

        if np.any(methods == 5):
            areas = self.foundation.area.numpy()
            if not np.isfinite(areas).all() or np.any(areas <= 0):
                raise ValueError("Pressure-dependent friction requires finite positive column areas")
        if np.any(np.isin(methods, [8, 9])):
            areas = self.foundation.area.numpy()
            lengths = self.foundation.rest_len.numpy()
            if not np.isfinite(areas).all() or np.any(areas <= 0.0):
                raise ValueError("Column-derived friction requires finite positive column areas")
            if not np.isfinite(lengths).all() or np.any(lengths <= 0.0):
                raise ValueError("Column-derived friction requires finite positive column rest lengths")

        # Validate physics parameters
        if np.any(arr[:, 1] < 0.0):
            raise ValueError("mu must be nonnegative")
        if np.any(arr[:, 2] <= 0.0):
            raise ValueError("kt_scale must be strictly positive")
        if np.any(arr[:, 3] < 0.0):
            raise ValueError("kv_scale must be nonnegative")
        if np.any(arr[:, 4] < 0.0):
            raise ValueError("viscous_ratio must be nonnegative")
        if np.any(arr[:, 5] < 0.0):
            raise ValueError("release_dwell_s must be nonnegative")
        if np.any(arr[:, 6] != 0.0):
            raise ValueError("yield_width must be 0.0 for qualified production friction")
        if np.any(arr[:, 7] < 0) or np.any(arr[:, 7] > arr[:, 1]):
            raise ValueError("Dynamic friction must be nonnegative and no larger than static friction")
        if np.any(arr[:, 8] < 1e-6):
            raise ValueError("Transition speed must be at least 1e-6 m/s")
        if np.any(arr[:, 9] <= 0.0):
            raise ValueError("Pressure scale must be positive [Pa]")
        if np.any(arr[:, 10] <= 0.0):
            raise ValueError("Sliding scale must be positive [m]")
        if np.any(arr[:, 11] <= 0.0):
            raise ValueError("Shear relaxation time must be positive [s]")
        if np.any(np.isin(methods, [7, 8]) & (arr[:, 4] != 0.0)):
            raise ValueError("Maxwell friction requires viscous_ratio=0; viscosity is internal to the branch")

        padded = np.repeat(arr[-1:], self.world_count, axis=0)
        padded[: len(arr)] = arr
        self.settings.assign(padded)

    def reset(self) -> None:
        """Clear bristle scratch state, deflection, stored energy, and diagnostics."""
        for arr in (
            self.scratch_anchor,
            self.scratch_stuck,
            self.scratch_dwell,
            self.deflection,
            self.sliding_distance,
            self.maxwell_force,
            self.stored_energy,
            self.column_diagnostics,
            self.step_diagnostics,
            self.partial_diagnostics,
            self.totals,
        ):
            arr.zero_()

    def apply(self, state, dt: float) -> None:
        """Execute postnormal friction solve and track resident diagnostics."""
        f = self.foundation
        plane = int(f.ground_height_m is not None)
        height = float(f.ground_height_m or 0.0)

        wp.launch(
            _postnormal_friction_step,
            dim=self.world_count * self.column_count,
            inputs=[
                self.column_count,
                plane,
                height,
                f.carrier,
                state.body_q,
                state.body_qd,
                f.body_com,
                f.anchor_local,
                f.ground_force,
                f.column_force,
                self.settings,
                self.base_kt,
                self.base_kv,
                f.area,
                f.rest_len,
                f.world_params,
                f.tangent_anchor,
                f.tangent_stuck,
                f.tangent_dwell,
                self.deflection,
                self.sliding_distance,
                self.maxwell_force,
                self.stored_energy,
                self.column_diagnostics,
                self.step_diagnostics,
                dt,
            ],
            device=self.device,
        )

        wp.launch(
            _reduce_column_diagnostics,
            dim=self.world_count * self.groups,
            inputs=[
                self.column_count,
                self.groups,
                self.step_diagnostics,
                self.partial_diagnostics,
            ],
            device=self.device,
        )

        wp.launch(
            _finish_world_diagnostics,
            dim=self.world_count,
            inputs=[
                self.groups,
                self.partial_diagnostics,
                self.totals,
            ],
            device=self.device,
        )

    def update_world_tau(self, world: int, tau: float) -> None:
        """Update shear relaxation time for a specific world."""
        settings_np = self.settings.numpy()
        settings_np[world, 11] = float(tau)
        self.settings.assign(settings_np)

    def update_taus(self, taus: Sequence[float]) -> None:
        """Update shear relaxation times across all worlds."""
        settings_np = self.settings.numpy()
        for w, tau in enumerate(taus):
            settings_np[w, 11] = float(tau)
        self.settings.assign(settings_np)

    def detach(self, restore_default: bool = True) -> None:
        """Detach from foundation and restore configured default friction solver if requested."""
        if getattr(self.foundation, "friction_solver", None) is self:
            self.foundation.friction_solver = None
            if restore_default and getattr(self.foundation, "config", None) is not None:
                if getattr(self.foundation.config, "friction_model", "legacy") in (
                    "elastic_coulomb",
                    "maxwell",
                    "column_maxwell",
                ):
                    self.foundation._install_default_friction_adapter()
