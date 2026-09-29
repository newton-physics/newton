# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Store controller contact histories out of place for full discrete backpropagation."""

from __future__ import annotations

from typing import ClassVar

import numpy as np
import warp as wp

from projects.digital_shoe.contact import bristle_step, contact_kinematics, normal_reaction, pasternak_flux
from projects.digital_shoe.friction_maxwell import (
    bristle_elastic_coulomb_step,
    bristle_maxwell_step,
    column_maxwell_parameters,
    elastic_coulomb_stiffness,
)
from projects.digital_shoe.material import maxwell_coefficients, maxwell_step
from projects.digital_shoe.runtime import FoundationParams, _hyperfoam_pressure, _pasternak_coupling, _surround_balance

wp.set_module_options({"enable_backward": True, "fuse_fp": True})


@wp.kernel
def _rigid_penetration(
    columns: int,
    body_q: wp.array[wp.transform],
    anchor: wp.array[wp.vec3],
    rigid_top: wp.array[float],
    rigid: wp.array[float],
):
    """Stage pose-dependent penetration once so eight reverse sweeps do not contend on the same pose."""
    i = wp.tid()
    c = i % columns
    point = wp.transform_point(body_q[i // columns], anchor[c])
    rigid[i] = rigid_top[c] - point[2]


@wp.kernel
def _surround_sweep(
    columns: int,
    rigid: wp.array[float],
    driven: wp.array[int],
    neighbors: wp.array2d[int],
    rest: wp.array[float],
    area: wp.array[float],
    q_state: wp.array[float],
    peq_prev: wp.array[float],
    params: wp.array[FoundationParams],
    decay: wp.array[float],
    gain_values: wp.array[float],
    coupling_scale: float,
    attachment: float,
    max_strain: float,
    relaxation: float,
    carrier_bond: int,
    current: wp.array[float],
    output: wp.array[float],
):
    """Apply the same fixed Jacobi sweep with a retained, per-column penetration input."""
    i = wp.tid()
    w = i // columns
    c = i % columns
    r = rigid[i]
    if driven[c] != 0:
        output[i] = wp.max(r, 0.0)
        return
    p = params[w]
    gain = gain_values[w]
    value = current[i]
    pull = float(0.0)
    coupling_sum = float(0.0)
    for side in range(4):
        j = neighbors[c, side]
        if j >= 0:
            coupling = coupling_scale * _pasternak_coupling(rest[c], rest[j], p)
            pull += coupling * (current[w * columns + j] - value)
            coupling_sum += coupling
    output[i] = _surround_balance(
        value,
        r,
        pull,
        coupling_sum,
        rest[c],
        decay[w] * q_state[i] - gain * peq_prev[i],
        gain,
        p,
        area[c],
        attachment,
        max_strain,
        relaxation,
        carrier_bond,
    )


@wp.kernel
def _free_top(
    columns: int,
    body_q: wp.array[wp.transform],
    driven: wp.array[int],
    anchor: wp.array[wp.vec3],
    rigid_top: wp.array[float],
    compression: wp.array[float],
    top: wp.array[float],
):
    """Publish each world's free surface without modifying a previous step."""
    i = wp.tid()
    w = i // columns
    c = i % columns
    if driven[c] != 0:
        top[i] = rigid_top[c]
    else:
        point = wp.transform_point(body_q[w], anchor[c])
        top[i] = point[2] + compression[i]


@wp.kernel
def _pressure(
    columns: int,
    dt: float,
    body_q: wp.array[wp.transform],
    anchor: wp.array[wp.vec3],
    top: wp.array[float],
    rest: wp.array[float],
    params: wp.array[FoundationParams],
    q_prev: wp.array[float],
    peq_prev: wp.array[float],
    q_next: wp.array[float],
    peq_next: wp.array[float],
    compression: wp.array[float],
    pressure: wp.array[float],
):
    """Evaluate the shared Maxwell and Hyperfoam expressions into separate histories."""
    i = wp.tid()
    w = i // columns
    c = i % columns
    p = params[w]
    point = wp.transform_point(body_q[w], anchor[c])
    comp = top[i] - point[2]
    if comp < 0.0:
        comp = 0.0
    compression[i] = comp
    peq = _hyperfoam_pressure(comp / rest[c], p)
    decay, ramp = maxwell_coefficients(dt, p.tau_s)
    qn = maxwell_step(q_prev[i], peq, peq_prev[i], p.overstress, decay, ramp)
    q_next[i] = qn
    peq_next[i] = peq
    pressure[i] = peq + qn


@wp.kernel
def _forces(
    columns: int,
    dt: float,
    ground_height: float,
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    body_com: wp.array[wp.vec3],
    anchor: wp.array[wp.vec3],
    area: wp.array[float],
    rest: wp.array[float],
    neighbors: wp.array2d[int],
    compression: wp.array[float],
    pressure: wp.array[float],
    params: wp.array[FoundationParams],
    kt: wp.array[float],
    kv: wp.array[float],
    anchor_prev: wp.array[wp.vec2],
    stuck_prev: wp.array[int],
    dwell_prev: wp.array[float],
    deflection_prev: wp.array[wp.vec2],
    maxwell_prev: wp.array[wp.vec2],
    deflection_next: wp.array[wp.vec2],
    maxwell_next: wp.array[wp.vec2],
    anchor_next: wp.array[wp.vec2],
    stuck_next: wp.array[int],
    dwell_next: wp.array[float],
    transfer: wp.array[wp.vec3],
    ground: wp.array[wp.vec3],
    points: wp.array[wp.vec3],
):
    """Apply the same carried-ground bristle law without overwriting its input state."""
    i = wp.tid()
    w = i // columns
    c = i % columns
    p = params[w]
    flux = pasternak_flux(c, w * columns, compression, rest, neighbors, p.g_eq + p.g_eq2)
    point, _com, velocity, gap = contact_kinematics(body_q[w], body_qd[w], body_com[w], anchor[c], ground_height, 1)
    reaction = normal_reaction(compression[i], pressure[i], area[c], p.normal_damping, velocity[2], gap, 1)
    if p.friction_model == 3:
        shear_kt = elastic_coulomb_stiffness(p.g_eq + p.g_eq2, area[c], rest[c])
        tangent, _jac, z, _q, next_stuck, next_dwell = bristle_elastic_coulomb_step(
            wp.vec2(velocity[0], velocity[1]),
            dt,
            reaction,
            shear_kt,
            p.mu,
            p.friction_release_dwell_s,
            deflection_prev[i],
            stuck_prev[i],
            dwell_prev[i],
        )
        deflection_next[i] = z
        maxwell_next[i] = wp.vec2(0.0)
        next_anchor = wp.vec2(point[0], point[1]) + dt * wp.vec2(velocity[0], velocity[1]) - z
    elif p.friction_model == 1 or p.friction_model == 2:
        shear_kt = kt[c]
        shear_kv = kv[c]
        if p.friction_model == 2:
            shear_kt, shear_kv = column_maxwell_parameters(
                p.g_eq + p.g_eq2,
                p.overstress,
                area[c],
                rest[c],
                p.friction_relaxation_time_s,
            )
        tangent, _jac, z, q, next_stuck, next_dwell = bristle_maxwell_step(
            wp.vec2(velocity[0], velocity[1]),
            dt,
            reaction,
            shear_kt,
            shear_kv,
            p.friction_relaxation_time_s,
            p.mu,
            p.friction_release_dwell_s,
            deflection_prev[i],
            maxwell_prev[i],
            stuck_prev[i],
            dwell_prev[i],
        )
        deflection_next[i] = z
        maxwell_next[i] = q
        next_anchor = wp.vec2(point[0], point[1]) + dt * wp.vec2(velocity[0], velocity[1]) - z
    else:
        tangent, next_anchor, next_stuck, next_dwell = bristle_step(
            wp.vec2(point[0], point[1]),
            wp.vec2(velocity[0], velocity[1]),
            dt,
            reaction,
            kt[c],
            kv[c],
            p.mu,
            p.friction_viscous_ratio,
            p.friction_release_dwell_s,
            anchor_prev[i],
            stuck_prev[i],
            dwell_prev[i],
        )
        deflection_next[i] = wp.vec2(0.0)
        maxwell_next[i] = wp.vec2(0.0)
    anchor_next[i] = next_anchor
    stuck_next[i] = next_stuck
    dwell_next[i] = next_dwell
    transfer[i] = wp.vec3(tangent[0], tangent[1], reaction - flux)
    ground[i] = wp.vec3(tangent[0], tangent[1], reaction)
    points[i] = point


@wp.func
def _group_wrench(
    base: int,
    group: int,
    columns: int,
    groups: int,
    com_world: wp.vec3,
    force: wp.array[wp.vec3],
    point: wp.array[wp.vec3],
) -> wp.spatial_vector:
    """Preserve the forward foundation's column and arithmetic reduction order."""
    force_sum = wp.vec3(0.0)
    torque_sum = wp.vec3(0.0)
    for c in range(group, columns, groups):
        i = base + c
        force_sum += force[i]
        torque_sum += wp.cross(point[i] - com_world, force[i])
    return wp.spatial_vector(force_sum, torque_sum)


@wp.func_grad(_group_wrench)
def _adj_group_wrench(
    base: int,
    group: int,
    columns: int,
    groups: int,
    com_world: wp.vec3,
    force: wp.array[wp.vec3],
    point: wp.array[wp.vec3],
    adj_result: wp.spatial_vector,
):
    """Replay each column's local wrench VJP rather than differentiating a dynamic nonlinear loop."""
    force_seed = wp.spatial_top(adj_result)
    torque_seed = wp.spatial_bottom(adj_result)
    for c in range(group, columns, groups):
        i = base + c
        arm = point[i] - com_world
        arm_seed = wp.cross(force[i], torque_seed)
        wp.atomic_add(wp.adjoint[force], i, force_seed + wp.cross(torque_seed, arm))
        wp.atomic_add(wp.adjoint[point], i, arm_seed)
        wp.adjoint[com_world] -= arm_seed


@wp.kernel
def _partial_wrench(
    columns: int,
    groups: int,
    body_q: wp.array[wp.transform],
    body_com: wp.array[wp.vec3],
    force: wp.array[wp.vec3],
    points: wp.array[wp.vec3],
    partial: wp.array[wp.spatial_vector],
):
    """Reduce fixed-order groups with explicit per-column reverse derivatives."""
    i = wp.tid()
    w = i // groups
    group = i % groups
    com = wp.transform_point(body_q[w], body_com[w])
    partial[i] = _group_wrench(w * columns, group, columns, groups, com, force, points)


@wp.kernel
def _total_wrench(groups: int, partial: wp.array[wp.spatial_vector], wrench: wp.array[wp.spatial_vector]):
    """Fold groups in the original order into a fresh per-step wrench."""
    w = wp.tid()
    total = wp.spatial_vector()
    for g in range(groups):
        total += partial[w * groups + g]
    wrench[w] = total


class ContactTape:
    """Own non-aliased contact state for a fixed-length carried-shoe rollout.

    Geometry, material, and friction remain read-only views of a forward
    foundation. This adapter supports one carrier per world on a fixed plane.
    It does not run dynamics, score incomplete trajectories, or change contact laws.
    """

    class State:
        """Retain all continuous and discrete shoe memory needed at a step boundary."""

        FIELDS: ClassVar[dict[str, type]] = {
            "q_state": float,
            "peq_prev": float,
            "tangent_anchor": wp.vec2,
            "tangent_stuck": int,
            "tangent_dwell": float,
            "tangent_deflection": wp.vec2,
            "tangent_maxwell_force": wp.vec2,
            "surround_compression": float,
        }

        def __init__(self, count: int, device, *, requires_grad: bool = True):
            for name, dtype in self.FIELDS.items():
                setattr(
                    self,
                    name,
                    wp.zeros(count, dtype=dtype, device=device, requires_grad=requires_grad and dtype is not int),
                )

        @classmethod
        def from_storage(cls, storage, index):
            """Create disjoint timestep views while the parent storage owns their lifetime."""
            result = cls.__new__(cls)
            for name, array in storage.items():
                setattr(result, name, array[index])
            return result

        def copy_from(self, source):
            """Copy a complete state boundary; call outside Tape when restoring a fixed checkpoint."""
            for name in self.FIELDS:
                wp.copy(getattr(self, name), getattr(source, name))

    class Step:
        """Retain disjoint views of this step's pressure, force, and sweep intermediates."""

        FIELDS: ClassVar[dict[str, type]] = {
            "compression": float,
            "rigid": float,
            "base_pressure": float,
            "z_free": float,
            "column_force": wp.vec3,
            "ground_force": wp.vec3,
            "contact_point": wp.vec3,
        }

        def __init__(self, tape, index, output_state):
            for name, array in tape._step_storage.items():
                setattr(self, name, array[index])
            self.partial = tape._partial_storage[index]
            self.body_f = tape._wrench_storage[index]
            self.sweeps = []
            if tape.source.free_column_count:
                if tape._sweep_storage is not None:
                    self.sweeps = [
                        tape._sweep_storage[index, sweep] for sweep in range(tape.source.surround.sweeps - 1)
                    ]
                self.sweeps.append(output_state.surround_compression)

    def __init__(self, source, steps: int, dt: float):
        if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
            raise ValueError("steps must be a positive integer")
        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError("dt must be finite and positive")
        if source.ground_height_m is None:
            raise ValueError("Controller contact requires a declared ground plane")
        if not np.array_equal(source.carrier.numpy(), np.arange(source.world_count)):
            raise ValueError("Controller contact requires one indexed carrier per world")
        adapter = getattr(source, "friction_solver", None)
        if adapter is not None and not getattr(adapter, "is_default", False):
            raise ValueError(
                "ContactTape supports the configured default friction model, not arbitrary attached overrides"
            )
        self.source = source
        self.device = source.device
        self.dt = float(dt)
        count = source.world_count * source.column_count
        self._state_storage = {
            name: wp.zeros((steps + 1, count), dtype=dtype, device=self.device, requires_grad=dtype is not int)
            for name, dtype in self.State.FIELDS.items()
        }
        self._step_storage = {
            name: wp.zeros((steps, count), dtype=dtype, device=self.device, requires_grad=True)
            for name, dtype in self.Step.FIELDS.items()
        }
        self._partial_storage = wp.zeros(
            (steps, source.world_count * source.reduction_groups),
            dtype=wp.spatial_vector,
            device=self.device,
            requires_grad=True,
        )
        self._wrench_storage = wp.zeros(
            (steps, source.world_count), dtype=wp.spatial_vector, device=self.device, requires_grad=True
        )
        self._sweep_storage = None
        if source.free_column_count and source.surround.sweeps > 1:
            self._sweep_storage = wp.zeros(
                (steps, source.surround.sweeps - 1, count), dtype=float, device=self.device, requires_grad=True
            )
        self.states = [self.State.from_storage(self._state_storage, t) for t in range(steps + 1)]
        self.steps = [self.Step(self, t, self.states[t + 1]) for t in range(steps)]
        source._refresh_surround_constants(dt)
        cfg = source.surround
        self.relaxation = (
            1.0 if cfg.relaxation_time_s <= 0 else 1.0 - float(np.exp(-(dt / cfg.sweeps) / cfg.relaxation_time_s))
        )

    def zero_grad(self):
        """Clear owning gradient buffers once, not once for every timestep view."""
        arrays = [
            *self._state_storage.values(),
            *self._step_storage.values(),
            self._partial_storage,
            self._wrench_storage,
            self._sweep_storage,
        ]
        for array in arrays:
            if array is not None and array.grad is not None:
                array.grad.zero_()

    def apply(self, index: int, body_q, body_qd):
        """Advance one shared contact update and retain its complete differentiable history."""
        s = self.source
        prev, nxt, out = self.states[index], self.states[index + 1], self.steps[index]
        cfg = s.surround
        count = s.world_count * s.column_count
        current = prev.surround_compression
        if s.free_column_count:
            wp.launch(
                _rigid_penetration,
                dim=count,
                inputs=[s.column_count, body_q, s.anchor_local, s.z_free_rigid, out.rigid],
                device=self.device,
            )
            for buffer in out.sweeps:
                wp.launch(
                    _surround_sweep,
                    dim=count,
                    inputs=[
                        s.column_count,
                        out.rigid,
                        s.driven,
                        s.neighbors,
                        s.rest_len,
                        s.area,
                        prev.q_state,
                        prev.peq_prev,
                        s.world_params,
                        s.surround_decay,
                        s.surround_gain,
                        float(cfg.coupling_scale),
                        float(cfg.attachment_n_m),
                        float(cfg.max_strain),
                        self.relaxation,
                        int(bool(cfg.carrier_bond)),
                        current,
                        buffer,
                    ],
                    device=self.device,
                )
                current = buffer
            wp.launch(
                _free_top,
                dim=count,
                inputs=[s.column_count, body_q, s.driven, s.anchor_local, s.z_free_rigid, current, out.z_free],
                device=self.device,
            )
            top = out.z_free
        else:
            wp.copy(nxt.surround_compression, current)
            top = s.z_free
        wp.launch(
            _pressure,
            dim=count,
            inputs=[
                s.column_count,
                self.dt,
                body_q,
                s.anchor_local,
                top,
                s.rest_len,
                s.world_params,
                prev.q_state,
                prev.peq_prev,
                nxt.q_state,
                nxt.peq_prev,
                out.compression,
                out.base_pressure,
            ],
            device=self.device,
        )
        wp.launch(
            _forces,
            dim=count,
            inputs=[
                s.column_count,
                self.dt,
                float(s.ground_height_m),
                body_q,
                body_qd,
                s.body_com,
                s.anchor_local,
                s.area,
                s.rest_len,
                s.neighbors,
                out.compression,
                out.base_pressure,
                s.world_params,
                s.friction_kt,
                s.friction_kv,
                prev.tangent_anchor,
                prev.tangent_stuck,
                prev.tangent_dwell,
                prev.tangent_deflection,
                prev.tangent_maxwell_force,
                nxt.tangent_deflection,
                nxt.tangent_maxwell_force,
                nxt.tangent_anchor,
                nxt.tangent_stuck,
                nxt.tangent_dwell,
                out.column_force,
                out.ground_force,
                out.contact_point,
            ],
            device=self.device,
        )
        wp.launch(
            _partial_wrench,
            dim=s.world_count * s.reduction_groups,
            inputs=[
                s.column_count,
                s.reduction_groups,
                body_q,
                s.body_com,
                out.ground_force,
                out.contact_point,
                out.partial,
            ],
            device=self.device,
            block_dim=32,
        )
        wp.launch(
            _total_wrench,
            dim=s.world_count,
            inputs=[s.reduction_groups, out.partial, out.body_f],
            device=self.device,
            block_dim=1,
        )
        return out
