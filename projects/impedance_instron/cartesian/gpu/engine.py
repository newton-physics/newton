# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Integrate independent Cartesian leg/shoe candidates without stepwise readbacks."""

from __future__ import annotations

import math
from copy import copy, deepcopy
from time import perf_counter
from types import SimpleNamespace

import numpy as np
import warp as wp

from projects.digital_shoe.runtime import FoundationConfig

from ..diagnostics import summarize as summarize_diagnostics
from ..fit import FitConfig
from ..mechanics import Body
from ..phase import hip_contact_gate
from ..run import Config
from ..shoe import Shoe
from ..trajectory import Spline, basis
from .foundation import FoundationFused
from .mechanics import Params, Vec5, ankle, dynamics, foot_angle, make_params, solve
from .objective import MeasuredObjective

wp.set_module_options({"enable_backward": False, "fuse_fp": False})


@wp.struct
class _Settings:
    stiffness: wp.vec4d
    damping: wp.vec4d
    lower: wp.vec2d
    upper: wp.vec2d
    dt: wp.float64
    gravity: wp.float64
    pitch: wp.float64
    hip_floor: wp.float64
    max_speed: wp.float64
    max_force: wp.float64
    compression_limit: wp.float64
    passive_cap: wp.float64
    joint_diagnostic: int
    steps: int
    controls: int


@wp.func
def _finite(q: Vec5) -> bool:
    valid = True
    for i in range(5):
        valid = valid and wp.isfinite(q[i])
    return valid


@wp.kernel
def _initialize(
    q0: Vec5,
    v0: Vec5,
    states: wp.array2d[Vec5],
    velocities: wp.array2d[Vec5],
    integrated: wp.array[int],
    recorded: wp.array[int],
    failure: wp.array[int],
    failure_step: wp.array[int],
    range_step: wp.array[int],
    range_mask: wp.array[int],
):
    w = wp.tid()
    states[0, w] = q0
    velocities[0, w] = v0
    integrated[w] = 0
    recorded[w] = 0
    failure[w] = 0
    failure_step[w] = -1
    range_step[w] = -1
    range_mask[w] = 0


@wp.func
def _prepare_world(
    w: int,
    s: int,
    state_row: int,
    p: Params,
    cfg: _Settings,
    basis_values: wp.array2d[wp.float64],
    hip_gate: wp.array[wp.float64],
    coefficients: wp.array3d[wp.float64],
    states: wp.array2d[Vec5],
    velocities: wp.array2d[Vec5],
    equilibrium: wp.array2d[wp.vec4d],
    actuator: wp.array2d[wp.vec4d],
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    failure: wp.array[int],
    failure_step: wp.array[int],
    range_step: wp.array[int],
    range_mask: wp.array[int],
):
    """Stage one world with an immutable time index and caller-owned history row."""
    if failure[w] != 0:
        return
    q = states[state_row, w]
    v = velocities[state_row, w]
    if not _finite(q) or not _finite(v):
        failure[w] = 1
        failure_step[w] = s
        return
    eq = wp.vec4d(wp.float64(0.0))
    if cfg.controls == 12:
        # Replay must reconstruct this value before evaluating the force screen.
        # Warp does not replay intermediate values from a dynamic loop.
        for row in range(12):
            for c in range(4):
                eq[c] += basis_values[s, row] * coefficients[w, row, c]
    else:
        for row in range(cfg.controls):
            for c in range(4):
                eq[c] += basis_values[s, row] * coefficients[w, row, c]
    gate = hip_gate[s]
    control = wp.vec4d(
        gate * (cfg.stiffness[0] * (eq[0] - q[0]) - cfg.damping[0] * v[0]),
        gate * (cfg.stiffness[1] * (eq[1] - q[1]) - cfg.damping[1] * v[1]),
        cfg.stiffness[2] * (eq[2] - q[3]) - cfg.damping[2] * v[3],
        cfg.stiffness[3] * (eq[3] - q[4]) - cfg.damping[3] * v[4],
    )
    for c in range(4):
        if not wp.isfinite(control[c]):
            failure[w] = 2
            failure_step[w] = s
            return
    equilibrium[state_row, w] = eq
    actuator[state_row, w] = control
    outside = int(0)
    for j in range(2):
        if q[j + 3] < cfg.lower[j] or q[j + 3] > cfg.upper[j]:
            outside = outside | (1 << j)
    if outside != 0 and range_step[w] == -1:
        range_step[w] = s
        range_mask[w] = outside
    code = int(0)
    if outside != 0 and cfg.joint_diagnostic == 0:
        code = code | 4
    if q[1] < cfg.hip_floor:
        code = code | 8
    if wp.length(wp.vec2d(v[0], v[1])) > cfg.max_speed:
        code = code | 16
    for j in range(2, 5):
        if wp.abs(v[j]) > cfg.max_speed:
            code = code | 16
    if wp.length(wp.vec2d(control[0], control[1])) > cfg.max_force:
        code = code | 32
    if code != 0:
        failure[w] = code
        failure_step[w] = s
        return
    if s == cfg.steps:
        return
    position, jx, jz = ankle(q, p)
    angle = (foot_angle(q) - cfg.pitch) / wp.float64(2.0)
    body_q[w] = wp.transform(
        wp.vec3(wp.float32(position[0]), 0.0, wp.float32(position[1])),
        wp.quat(0.0, wp.float32(-wp.sin(angle)), 0.0, wp.float32(wp.cos(angle))),
    )
    body_qd[w] = wp.spatial_vector(
        wp.float32(wp.dot(jx, v)),
        0.0,
        wp.float32(wp.dot(jz, v)),
        0.0,
        wp.float32(-((v[2] + v[3]) + v[4])),
        0.0,
    )


@wp.func_replay(_prepare_world)
def _replay_prepare_world(
    w: int,
    s: int,
    state_row: int,
    p: Params,
    cfg: _Settings,
    basis_values: wp.array2d[wp.float64],
    hip_gate: wp.array[wp.float64],
    coefficients: wp.array3d[wp.float64],
    states: wp.array2d[Vec5],
    velocities: wp.array2d[Vec5],
    equilibrium: wp.array2d[wp.vec4d],
    actuator: wp.array2d[wp.vec4d],
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    failure: wp.array[int],
    failure_step: wp.array[int],
    range_step: wp.array[int],
    range_mask: wp.array[int],
):
    """Reuse retained array outputs without replaying primal writes or counters."""
    pass


@wp.kernel
def _prepare(
    p: Params,
    cfg: _Settings,
    clock: wp.array[int],
    basis_values: wp.array2d[wp.float64],
    hip_gate: wp.array[wp.float64],
    coefficients: wp.array3d[wp.float64],
    states: wp.array2d[Vec5],
    velocities: wp.array2d[Vec5],
    equilibrium: wp.array2d[wp.vec4d],
    actuator: wp.array2d[wp.vec4d],
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    failure: wp.array[int],
    failure_step: wp.array[int],
    range_step: wp.array[int],
    range_mask: wp.array[int],
):
    """Stage the mutable forward engine through the shared step function."""
    _prepare_world(
        wp.tid(),
        clock[0],
        clock[0],
        p,
        cfg,
        basis_values,
        hip_gate,
        coefficients,
        states,
        velocities,
        equilibrium,
        actuator,
        body_q,
        body_qd,
        failure,
        failure_step,
        range_step,
        range_mask,
    )


@wp.kernel
def _prepare_masked(
    enabled: wp.array[int],
    p: Params,
    cfg: _Settings,
    clock: wp.array[int],
    basis_values: wp.array2d[wp.float64],
    hip_gate: wp.array[wp.float64],
    coefficients: wp.array3d[wp.float64],
    states: wp.array2d[Vec5],
    velocities: wp.array2d[Vec5],
    equilibrium: wp.array2d[wp.vec4d],
    actuator: wp.array2d[wp.vec4d],
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    failure: wp.array[int],
    failure_step: wp.array[int],
    range_step: wp.array[int],
    range_mask: wp.array[int],
):
    """Stage the mutable forward engine through the shared step function."""
    if enabled[wp.tid()] == 0:
        return
    _prepare_world(
        wp.tid(),
        clock[0],
        clock[0],
        p,
        cfg,
        basis_values,
        hip_gate,
        coefficients,
        states,
        velocities,
        equilibrium,
        actuator,
        body_q,
        body_qd,
        failure,
        failure_step,
        range_step,
        range_mask,
    )


@wp.kernel
def _compression_partial(
    groups: int,
    column_count: int,
    passive_cap: wp.float64,
    compression: wp.array[wp.float32],
    rest: wp.array[wp.float64],
    driven: wp.array[int],
    maxima: wp.array2d[wp.vec2d],
    caps: wp.array2d[int],
    nonfinite: wp.array2d[int],
):
    """Reduce column diagnostics cooperatively without changing float64 division."""
    w, group, lane = wp.tid()
    driven_max = wp.float64(0.0)
    passive_max = wp.float64(0.0)
    count = int(0)
    invalid = int(0)
    for c in range(group * 32 + lane, column_count, groups * 32):
        fraction = wp.float64(compression[w * column_count + c]) / rest[c]
        if not wp.isfinite(fraction):
            invalid = 1
        elif driven[c] != 0:
            driven_max = wp.max(driven_max, fraction)
        else:
            passive_max = wp.max(passive_max, fraction)
            if fraction >= passive_cap - wp.float64(1.0e-6):
                count = count + 1
    dm = wp.tile_max(wp.tile(driven_max))
    pm = wp.tile_max(wp.tile(passive_max))
    ct = wp.tile_sum(wp.tile(count))
    inv = wp.tile_max(wp.tile(invalid))
    if lane == 0:
        maxima[w, group] = wp.vec2d(dm[0], pm[0])
        caps[w, group] = ct[0]
        nonfinite[w, group] = inv[0]


@wp.func
def _advance_world(
    w: int,
    s: int,
    state_row: int,
    next_row: int,
    p: Params,
    cfg: _Settings,
    states: wp.array2d[Vec5],
    velocities: wp.array2d[Vec5],
    next_states: wp.array2d[Vec5],
    next_velocities: wp.array2d[Vec5],
    actuator: wp.array2d[wp.vec4d],
    body_f: wp.array[wp.spatial_vector],
    groups: int,
    partial_maxima: wp.array2d[wp.vec2d],
    partial_caps: wp.array2d[int],
    partial_nonfinite: wp.array2d[int],
    forces: wp.array2d[wp.vec2d],
    moments: wp.array2d[wp.float64],
    fractions: wp.array2d[wp.vec3d],
    cap_count: wp.array2d[int],
    integrated: wp.array[int],
    recorded: wp.array[int],
    failure: wp.array[int],
    failure_step: wp.array[int],
):
    """Advance one world through the shared screens, load mapping, and mass solve."""
    if failure[w] != 0:
        return
    wrench = body_f[w]
    force = wp.vec2d(wp.float64(wrench[0]), wp.float64(wrench[2]))
    moment = -wp.float64(wrench[4])
    if not wp.isfinite(force[0]) or not wp.isfinite(force[1]) or not wp.isfinite(moment):
        failure[w] = 64
        failure_step[w] = s
        return
    driven_max = wp.float64(0.0)
    passive_max = wp.float64(0.0)
    caps = int(0)
    for group in range(groups):
        if partial_nonfinite[w, group] != 0:
            failure[w] = 128
            failure_step[w] = s
            return
        maxima = partial_maxima[w, group]
        driven_max = wp.max(driven_max, maxima[0])
        passive_max = wp.max(passive_max, maxima[1])
        caps = caps + partial_caps[w, group]
    forces[state_row, w] = force
    moments[state_row, w] = moment
    fractions[state_row, w] = wp.vec3d(wp.max(driven_max, passive_max), driven_max, passive_max)
    cap_count[state_row, w] = caps
    recorded[w] = recorded[w] + 1
    code = int(0)
    if driven_max > cfg.compression_limit + wp.float64(1.0e-6):
        code = code | 256
    if force[1] < wp.float64(-1.0e-6):
        code = code | 512
    if wp.length(force) > cfg.max_force:
        code = code | 1024
    if code != 0:
        failure[w] = code
        failure_step[w] = s
        return
    q = states[state_row, w]
    v = velocities[state_row, w]
    control = actuator[state_row, w]
    _position, jx, jz = ankle(q, p)
    load = Vec5(control[0], control[1], wp.float64(0.0), control[2], control[3])
    for j in range(5):
        external = jx[j] * force[0] + jz[j] * force[1]
        if j >= 2:
            external = external + moment
        load[j] += external
    mass, bias = dynamics(q, v, p, cfg.gravity)
    acceleration = solve(mass, load - bias)
    next_v = v + cfg.dt * acceleration
    next_q = q + cfg.dt * next_v
    if not _finite(next_q) or not _finite(next_v):
        failure[w] = 2048
        failure_step[w] = s
        return
    next_states[next_row, w] = next_q
    next_velocities[next_row, w] = next_v
    integrated[w] = integrated[w] + 1


@wp.func_replay(_advance_world)
def _replay_advance_world(
    w: int,
    s: int,
    state_row: int,
    next_row: int,
    p: Params,
    cfg: _Settings,
    states: wp.array2d[Vec5],
    velocities: wp.array2d[Vec5],
    next_states: wp.array2d[Vec5],
    next_velocities: wp.array2d[Vec5],
    actuator: wp.array2d[wp.vec4d],
    body_f: wp.array[wp.spatial_vector],
    groups: int,
    partial_maxima: wp.array2d[wp.vec2d],
    partial_caps: wp.array2d[int],
    partial_nonfinite: wp.array2d[int],
    forces: wp.array2d[wp.vec2d],
    moments: wp.array2d[wp.float64],
    fractions: wp.array2d[wp.vec3d],
    cap_count: wp.array2d[int],
    integrated: wp.array[int],
    recorded: wp.array[int],
    failure: wp.array[int],
    failure_step: wp.array[int],
):
    """Reuse retained array outputs without replaying primal writes or counters."""
    pass


@wp.kernel
def _advance(
    p: Params,
    cfg: _Settings,
    clock: wp.array[int],
    states: wp.array2d[Vec5],
    velocities: wp.array2d[Vec5],
    actuator: wp.array2d[wp.vec4d],
    body_f: wp.array[wp.spatial_vector],
    groups: int,
    partial_maxima: wp.array2d[wp.vec2d],
    partial_caps: wp.array2d[int],
    partial_nonfinite: wp.array2d[int],
    forces: wp.array2d[wp.vec2d],
    moments: wp.array2d[wp.float64],
    fractions: wp.array2d[wp.vec3d],
    cap_count: wp.array2d[int],
    integrated: wp.array[int],
    recorded: wp.array[int],
    failure: wp.array[int],
    failure_step: wp.array[int],
):
    """Advance the mutable forward engine through the shared step function."""
    _advance_world(
        wp.tid(),
        clock[0],
        clock[0],
        clock[0] + 1,
        p,
        cfg,
        states,
        velocities,
        states,
        velocities,
        actuator,
        body_f,
        groups,
        partial_maxima,
        partial_caps,
        partial_nonfinite,
        forces,
        moments,
        fractions,
        cap_count,
        integrated,
        recorded,
        failure,
        failure_step,
    )


@wp.kernel
def _advance_masked(
    enabled: wp.array[int],
    p: Params,
    cfg: _Settings,
    clock: wp.array[int],
    states: wp.array2d[Vec5],
    velocities: wp.array2d[Vec5],
    actuator: wp.array2d[wp.vec4d],
    body_f: wp.array[wp.spatial_vector],
    groups: int,
    partial_maxima: wp.array2d[wp.vec2d],
    partial_caps: wp.array2d[int],
    partial_nonfinite: wp.array2d[int],
    forces: wp.array2d[wp.vec2d],
    moments: wp.array2d[wp.float64],
    fractions: wp.array2d[wp.vec3d],
    cap_count: wp.array2d[int],
    integrated: wp.array[int],
    recorded: wp.array[int],
    failure: wp.array[int],
    failure_step: wp.array[int],
):
    """Advance the mutable forward engine through the shared step function."""
    if enabled[wp.tid()] == 0:
        return
    _advance_world(
        wp.tid(),
        clock[0],
        clock[0],
        clock[0] + 1,
        p,
        cfg,
        states,
        velocities,
        states,
        velocities,
        actuator,
        body_f,
        groups,
        partial_maxima,
        partial_caps,
        partial_nonfinite,
        forces,
        moments,
        fractions,
        cap_count,
        integrated,
        recorded,
        failure,
        failure_step,
    )


@wp.kernel
def _advance_prepare(
    enabled: wp.array[int],
    early_tick: int,
    basis_values: wp.array2d[wp.float64],
    hip_gate: wp.array[wp.float64],
    coefficients: wp.array3d[wp.float64],
    equilibrium: wp.array2d[wp.vec4d],
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    range_step: wp.array[int],
    range_mask: wp.array[int],
    p: Params,
    cfg: _Settings,
    clock: wp.array[int],
    states: wp.array2d[Vec5],
    velocities: wp.array2d[Vec5],
    actuator: wp.array2d[wp.vec4d],
    body_f: wp.array[wp.spatial_vector],
    groups: int,
    partial_maxima: wp.array2d[wp.vec2d],
    partial_caps: wp.array2d[int],
    partial_nonfinite: wp.array2d[int],
    forces: wp.array2d[wp.vec2d],
    moments: wp.array2d[wp.float64],
    fractions: wp.array2d[wp.vec3d],
    cap_count: wp.array2d[int],
    integrated: wp.array[int],
    recorded: wp.array[int],
    failure: wp.array[int],
    failure_step: wp.array[int],
):
    """Advance one world, then stage its next carrier without a global dependency."""
    if enabled[wp.tid()] == 0:
        return
    _advance_world(
        wp.tid(),
        clock[0] - early_tick,
        clock[0] - early_tick,
        (clock[0] - early_tick) + 1,
        p,
        cfg,
        states,
        velocities,
        states,
        velocities,
        actuator,
        body_f,
        groups,
        partial_maxima,
        partial_caps,
        partial_nonfinite,
        forces,
        moments,
        fractions,
        cap_count,
        integrated,
        recorded,
        failure,
        failure_step,
    )

    _prepare_world(
        wp.tid(),
        (clock[0] - early_tick) + 1,
        (clock[0] - early_tick) + 1,
        p,
        cfg,
        basis_values,
        hip_gate,
        coefficients,
        states,
        velocities,
        equilibrium,
        actuator,
        body_q,
        body_qd,
        failure,
        failure_step,
        range_step,
        range_mask,
    )


@wp.kernel
def _tick(clock: wp.array[int]):
    clock[0] = clock[0] + 1


@wp.kernel
def _gather(
    world: int,
    states: wp.array2d[Vec5],
    velocities: wp.array2d[Vec5],
    equilibrium: wp.array2d[wp.vec4d],
    actuator: wp.array2d[wp.vec4d],
    forces: wp.array2d[wp.vec2d],
    moments: wp.array2d[wp.float64],
    fractions: wp.array2d[wp.vec3d],
    caps: wp.array2d[int],
    q: wp.array[Vec5],
    v: wp.array[Vec5],
    eq: wp.array[wp.vec4d],
    load: wp.array[wp.vec4d],
    f: wp.array[wp.vec2d],
    m: wp.array[wp.float64],
    comp: wp.array[wp.vec3d],
    cap: wp.array[int],
):
    i = wp.tid()
    q[i] = states[i, world]
    v[i] = velocities[i, world]
    eq[i] = equilibrium[i, world]
    load[i] = actuator[i, world]
    if i < forces.shape[0]:
        f[i] = forces[i, world]
        m[i] = moments[i, world]
        comp[i] = fractions[i, world]
        cap[i] = caps[i, world]


@wp.kernel
def _loop_condition(clock: wp.array[int], condition: wp.array[int], limit: int):
    condition[0] = int(clock[0] < limit)


@wp.kernel
def _snapshot(
    winner: wp.array[int],
    states: wp.array2d[Vec5],
    velocities: wp.array2d[Vec5],
    equilibrium: wp.array2d[wp.vec4d],
    actuator: wp.array2d[wp.vec4d],
    forces: wp.array2d[wp.vec2d],
    moments: wp.array2d[wp.float64],
    fractions: wp.array2d[wp.vec3d],
    caps: wp.array2d[int],
    integrated: wp.array[int],
    recorded: wp.array[int],
    failure: wp.array[int],
    failure_step: wp.array[int],
    range_step: wp.array[int],
    range_mask: wp.array[int],
    loss: wp.array[wp.float64],
    rmse: wp.array2d[wp.float64],
    maximum_error: wp.array2d[wp.float64],
    costs: wp.array2d[wp.float64],
    residual: wp.array2d[wp.float64],
    diagnostics: wp.array2d[wp.float64],
    out_states: wp.array2d[Vec5],
    out_velocities: wp.array2d[Vec5],
    out_equilibrium: wp.array2d[wp.vec4d],
    out_actuator: wp.array2d[wp.vec4d],
    out_forces: wp.array2d[wp.vec2d],
    out_moments: wp.array2d[wp.float64],
    out_fractions: wp.array2d[wp.vec3d],
    out_caps: wp.array2d[int],
    out_integrated: wp.array[int],
    out_recorded: wp.array[int],
    out_failure: wp.array[int],
    out_failure_step: wp.array[int],
    out_range_step: wp.array[int],
    out_range_mask: wp.array[int],
    out_loss: wp.array[wp.float64],
    out_rmse: wp.array2d[wp.float64],
    out_maximum_error: wp.array2d[wp.float64],
    out_costs: wp.array2d[wp.float64],
    out_residual: wp.array2d[wp.float64],
    out_diagnostics: wp.array2d[wp.float64],
):
    i = wp.tid()
    w = winner[0]
    if w < 0:
        return
    if i < states.shape[0]:
        out_states[i, 0] = states[i, w]
        out_velocities[i, 0] = velocities[i, w]
        out_equilibrium[i, 0] = equilibrium[i, w]
        out_actuator[i, 0] = actuator[i, w]
    if i < forces.shape[0]:
        out_forces[i, 0] = forces[i, w]
        out_moments[i, 0] = moments[i, w]
        out_fractions[i, 0] = fractions[i, w]
        out_caps[i, 0] = caps[i, w]
    if i < residual.shape[0]:
        out_residual[i, 0] = residual[i, w]
    if i == 0:
        out_integrated[0] = integrated[w]
        out_recorded[0] = recorded[w]
        out_failure[0] = failure[w]
        out_failure_step[0] = failure_step[w]
        out_range_step[0] = range_step[w]
        out_range_mask[0] = range_mask[w]
        out_loss[0] = loss[w]
        for ch in range(6):
            out_rmse[0, ch] = rmse[w, ch]
            out_maximum_error[0, ch] = maximum_error[w, ch]
        for ch in range(3):
            out_costs[0, ch] = costs[w, ch]
        for ch in range(11):
            out_diagnostics[0, ch] = diagnostics[w, ch]


class Engine:
    """Run independent leg/shoe candidates through fixed-size CUDA graph chunks.

    Only candidate coefficients enter at a batch boundary. All integration,
    contact histories, screens, and measured residuals remain on the device.
    Failed worlds freeze their leg and remain invalid for the rest of the batch.
    The shared contact kernels may continue updating those unused worlds; their
    isolated history tiles cannot load another candidate.

    Args:
        reference: Frozen single-leg recorded inputs and initial state.
        profile: Fixed three-body masses, gains, and spline bounds.
        artifact: Shared shoe artifact path.
        mount_m: Fixed ankle location in the intrinsic shoe frame [m].
        static_pitch_rad: Fixed calibration pitch [rad].
        config: Unchanged numerical timestep and failure screens.
        settings: Unchanged measured-objective settings.
        world_count: Fixed batch size, one independent leg/shoe per candidate.
        device: CUDA device; the engine does not offer a CPU fallback.
        chunk_steps: Number of integration steps captured in a reusable graph.
    """

    def __init__(
        self,
        reference,
        profile,
        artifact,
        mount_m,
        static_pitch_rad,
        *,
        config: Config,
        settings: FitConfig,
        world_count: int = 49,
        device: str = "cuda:0",
        chunk_steps: int = 32,
        friction_model: str = "elastic_coulomb",
    ):
        started = perf_counter()
        self.device = wp.get_device(device)
        if not self.device.is_cuda:
            raise ValueError("The Cartesian GPU engine requires CUDA; no CPU fallback")
        for name, value in (("world_count", world_count), ("chunk_steps", chunk_steps)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        reference, profile = deepcopy(reference), deepcopy(profile)
        if "shoe_static_pitch_rad" in reference and not np.isclose(
            float(reference["shoe_static_pitch_rad"]), static_pitch_rad, rtol=0, atol=1e-12
        ):
            raise ValueError("Ground-angle reference and shoe use different fixed pitch frames")
        self.reference, self.profile = reference, profile
        self.config, self.settings = config, settings
        self.world_count, self.chunk_steps = world_count, chunk_steps
        self.params = make_params(reference, profile)
        self.duration = float(reference["time_s"][-1])
        self.steps = math.ceil(self.duration / config.dt_s)
        self.dt = self.duration / self.steps
        self.time_s = np.linspace(0.0, self.duration, self.steps + 1)
        self.hip_gate_values = (
            hip_contact_gate(reference, self.time_s, config.hip_flight_gate_ramp_s)
            if config.hip_flight_gate_enabled
            else np.ones_like(self.time_s)
        )
        # Use the canonical adapter once for artifact registration and footprint selection.
        self.shoe = Shoe(artifact, mount_m, static_pitch_rad, device=str(self.device), friction_model=friction_model)
        if np.any(self.shoe.model.body_com.numpy()):
            raise ValueError("The reference shoe carrier COM must remain at its ankle origin")
        self.carriers = SimpleNamespace(
            body_q=wp.zeros(world_count, dtype=wp.transform, device=self.device),
            body_qd=wp.zeros(world_count, dtype=wp.spatial_vector, device=self.device),
            body_f=wp.zeros(world_count, dtype=wp.spatial_vector, device=self.device),
        )
        bed = self.shoe.shoe.column_bed
        self.foundation = FoundationFused(
            self.shoe.anchor_local_m,
            np.zeros(len(bed.rest_length_m)),
            bed.rest_length_m,
            bed.area_m2,
            bed.neighbors,
            bed.spacing_m,
            self.shoe.shoe.material,
            np.arange(world_count),
            wp.zeros(world_count, dtype=wp.vec3, device=self.device),
            FoundationConfig(
                ground_height_m=0.0,
                normal_damping=0.0,
                friction_stiffness=10000.0
                if friction_model == "legacy"
                else (1000.0 if friction_model == "maxwell" else 0.0),
                friction=10.0 if friction_model in ("legacy", "maxwell") else 0.0,
                mu=0.8,
                friction_model=friction_model,
            ),
            self.device,
            self.shoe.foundation.surround,
            world_count=world_count,
        )
        self.rest = wp.array(bed.rest_length_m, dtype=wp.float64, device=self.device)
        cfg = _Settings()
        cfg.stiffness = wp.vec4d(*profile["hip_stiffness_n_m"], *profile["joint_stiffness_nm_rad"])
        cfg.damping = wp.vec4d(*profile["hip_damping_ns_m"], *profile["joint_damping_nms_rad"])
        cfg.lower, cfg.upper = wp.vec2d(*profile["joint_lower_rad"]), wp.vec2d(*profile["joint_upper_rad"])
        cfg.dt, cfg.gravity, cfg.pitch = self.dt, config.gravity_m_s2, static_pitch_rad
        cfg.hip_floor, cfg.max_speed = config.minimum_hip_height_m, config.maximum_speed
        cfg.max_force, cfg.compression_limit = config.maximum_force_n, config.compression_limit
        cfg.passive_cap = float(self.foundation.surround.max_strain)
        cfg.joint_diagnostic = int(config.joint_limits_diagnostic)
        cfg.steps, cfg.controls = self.steps, settings.control_count
        self.kernel_config = cfg
        self.coefficients = wp.zeros((world_count, settings.control_count, 4), dtype=wp.float64, device=self.device)
        self.basis = wp.array(
            basis(self.time_s, self.duration, settings.control_count), dtype=wp.float64, device=self.device
        )
        self.hip_gate = wp.array(self.hip_gate_values, dtype=wp.float64, device=self.device)
        self.states = wp.zeros((self.steps + 1, world_count), dtype=Vec5, device=self.device)
        self.velocities = wp.zeros_like(self.states)
        self.equilibrium = wp.zeros((self.steps + 1, world_count), dtype=wp.vec4d, device=self.device)
        self.actuator = wp.zeros_like(self.equilibrium)
        self.forces = wp.zeros((self.steps, world_count), dtype=wp.vec2d, device=self.device)
        self.moments = wp.zeros((self.steps, world_count), dtype=wp.float64, device=self.device)
        self.fractions = wp.zeros((self.steps, world_count), dtype=wp.vec3d, device=self.device)
        self.caps = wp.zeros((self.steps, world_count), dtype=wp.int32, device=self.device)
        # Each group reduces a contiguous tile; extrema and integer sums are order-independent.
        self.reduction_groups = min(32, self.foundation.column_count)
        self.partial_maxima = wp.zeros((world_count, self.reduction_groups), dtype=wp.vec2d, device=self.device)
        self.partial_caps = wp.zeros((world_count, self.reduction_groups), dtype=wp.int32, device=self.device)
        self.partial_nonfinite = wp.zeros_like(self.partial_caps)
        self.clock = wp.zeros(1, dtype=wp.int32, device=self.device)
        self.foundation.diagnostics = (
            self.rest,
            self.kernel_config.passive_cap,
            self.partial_maxima,
            self.partial_caps,
            self.partial_nonfinite,
            self.clock,
        )
        for name in ("integrated", "recorded", "failure", "failure_step", "range_step", "range_mask"):
            setattr(self, name, wp.zeros(world_count, dtype=wp.int32, device=self.device))
        self.objective = MeasuredObjective(
            reference,
            settings,
            self.time_s,
            world_count,
            self.device,
            profile["hip_damping_ns_m"],
            self.shoe.metadata["friction_mu"],
            self.dt,
            self.hip_gate,
        )
        self.graph = None
        self.tail_graph = None
        self.resident_graph = None
        self.resident_condition = wp.zeros(1, dtype=wp.int32, device=self.device)
        self.setup_wall_s = perf_counter() - started
        self.capture_wall_s = None
        self.last_wall_s = None

    def _reset(self):
        """Reset each physical history once at the beginning of a candidate rollout."""
        self.foundation.reset()
        self.clock.zero_()
        wp.launch(
            _initialize,
            dim=self.world_count,
            inputs=[
                Vec5(*self.reference["state"][0]),
                Vec5(*self.reference["velocity"][0]),
                self.states,
                self.velocities,
                self.integrated,
                self.recorded,
                self.failure,
                self.failure_step,
                self.range_step,
                self.range_mask,
            ],
            device=self.device,
        )

    def _prepare(self):
        """Evaluate the four-channel controller and stage the ankle carrier on device."""
        # One independent world per block spreads small batches across CUDA SMs.
        wp.launch(
            _prepare_masked,
            dim=self.world_count,
            inputs=[
                self.foundation.enabled,
                self.params,
                self.kernel_config,
                self.clock,
                self.basis,
                self.hip_gate,
                self.coefficients,
                self.states,
                self.velocities,
                self.equilibrium,
                self.actuator,
                self.carriers.body_q,
                self.carriers.body_qd,
                self.failure,
                self.failure_step,
                self.range_step,
                self.range_mask,
            ],
            device=self.device,
            block_dim=1,
        )

    def _step(self, *, staged=False):
        """Queue unchanged physics; optionally stage the next carrier inside integration."""
        if not staged:
            self._prepare()
        early_tick = staged and self.foundation.fused_diagnostics
        self.foundation.apply(self.carriers, self.dt, clear_body_force=True, tick=early_tick)
        if not self.foundation.fused_diagnostics:
            wp.launch(
                _compression_partial,
                dim=(self.world_count, self.reduction_groups, 32),
                inputs=[
                    self.reduction_groups,
                    self.foundation.column_count,
                    self.kernel_config.passive_cap,
                    self.foundation.compression,
                    self.rest,
                    self.foundation.driven,
                    self.partial_maxima,
                    self.partial_caps,
                    self.partial_nonfinite,
                ],
                device=self.device,
                block_dim=32,
            )
        wp.launch(
            _advance_prepare if staged else _advance_masked,
            dim=self.world_count,
            inputs=[
                self.foundation.enabled,
                *(
                    [
                        int(early_tick),
                        self.basis,
                        self.hip_gate,
                        self.coefficients,
                        self.equilibrium,
                        self.carriers.body_q,
                        self.carriers.body_qd,
                        self.range_step,
                        self.range_mask,
                    ]
                    if staged
                    else []
                ),
                self.params,
                self.kernel_config,
                self.clock,
                self.states,
                self.velocities,
                self.actuator,
                self.carriers.body_f,
                self.reduction_groups,
                self.partial_maxima,
                self.partial_caps,
                self.partial_nonfinite,
                self.forces,
                self.moments,
                self.fractions,
                self.caps,
                self.integrated,
                self.recorded,
                self.failure,
                self.failure_step,
            ],
            device=self.device,
            block_dim=1,
        )
        if not early_tick:
            wp.launch(_tick, dim=1, inputs=[self.clock], device=self.device)

    def capture(self, coefficients):
        """Compile and capture fixed-size chunks separately from warm rollout timing."""
        values = self._validate_coefficients(coefficients)
        started = perf_counter()
        self.coefficients.assign(values)
        self.foundation.enabled.fill_(1)
        self._reset()
        self._step()
        self.objective.launch(self.states, self.velocities, self.actuator, self.forces, self.integrated, self.failure)
        self.objective.loss.numpy()
        with wp.ScopedCapture(device=self.device) as capture:
            for _ in range(min(self.chunk_steps, self.steps)):
                self._step()
        self.graph = capture.graph
        tail = self.steps % self.chunk_steps if self.steps >= self.chunk_steps else 0
        if tail:
            with wp.ScopedCapture(device=self.device) as capture:
                for _ in range(tail):
                    self._step()
            self.tail_graph = capture.graph
        self.resident_graph = None
        self.capture_resident()
        self.capture_wall_s = perf_counter() - started

    def capture_resident(self):
        """Capture a full rollout with a device-side graph loop and no array copies.

        Call :meth:`capture` with the initial bounded coefficients first. The
        conditional graph node requires CUDA 12.4 or newer. Shared step functions
        retain their arithmetic, but next-carrier staging joins integration and
        the shoe block writes compression diagnostics and advances the clock.
        The chunk path remains available with separate staging for comparison.
        """
        if self.resident_graph is not None:
            return
        if self.graph is None:
            raise RuntimeError("Call capture(initial_coefficients) before capture_resident()")
        chunk_limit = (self.steps // self.chunk_steps) * self.chunk_steps
        wp.load_module(module=__name__, device=self.device)

        def body():
            for _ in range(self.chunk_steps):
                self._step(staged=True)
            wp.launch(
                _loop_condition, dim=1, inputs=[self.clock, self.resident_condition, chunk_limit], device=self.device
            )

        with wp.ScopedCapture(device=self.device) as capture:
            self._reset()
            self._prepare()
            if chunk_limit:
                wp.launch(
                    _loop_condition,
                    dim=1,
                    inputs=[self.clock, self.resident_condition, chunk_limit],
                    device=self.device,
                )
                wp.capture_while(self.resident_condition, body)
                for _ in range(self.steps % self.chunk_steps):
                    self._step(staged=True)
            else:
                for _ in range(self.steps):
                    self._step(staged=True)
            self.objective.launch(
                self.states, self.velocities, self.actuator, self.forces, self.integrated, self.failure
            )
        self.resident_graph = capture.graph

    def evaluate_device(self):
        """Submit the full rollout graph using resident coefficients without copies or waits."""
        if self.resident_graph is None:
            raise RuntimeError("Call capture(initial_coefficients) before evaluate_device()")
        wp.capture_launch(self.resident_graph)

    class Snapshot:
        """Keep one accepted full history on CUDA until final result extraction."""

        def __init__(self, source):
            self.engine = copy(source)
            self.engine.world_count = 1
            for name in source._SNAPSHOT_FIELDS:
                array = getattr(source, name)
                shape = (array.shape[0], 1) if array.ndim == 2 else (1,)
                setattr(self.engine, name, wp.empty(shape, dtype=array.dtype, device=source.device))
            self.engine.objective = copy(source.objective)
            for name in source._SNAPSHOT_SCORES:
                array = getattr(source.objective, name)
                shape = (1,) if array.ndim == 1 else (1, array.shape[1])
                if name == "residual":
                    shape = (array.shape[0], 1)
                setattr(self.engine.objective, name, wp.empty(shape, dtype=array.dtype, device=source.device))

        def trace(self):
            """Unload the saved winning trajectory without rerunning physics."""
            return self.engine.trace(0)

        def score(self):
            """Unload the saved winning score and completion diagnostics."""
            scores = self.engine.objective.read()
            scores.update(integrated_steps=self.engine.integrated.numpy(), failure_code=self.engine.failure.numpy())
            return scores

    _SNAPSHOT_FIELDS = (
        "states",
        "velocities",
        "equilibrium",
        "actuator",
        "forces",
        "moments",
        "fractions",
        "caps",
        "integrated",
        "recorded",
        "failure",
        "failure_step",
        "range_step",
        "range_mask",
    )
    _SNAPSHOT_SCORES = ("loss", "rmse", "maximum_error", "costs", "residual", "diagnostics")

    def create_snapshot(self):
        """Allocate one best-history buffer before search starts."""
        return self.Snapshot(self)

    def snapshot_device(self, winner, snapshot):
        """Save device-selected history; a negative winner leaves the snapshot unchanged."""
        inputs = [winner]
        for source in (self, snapshot.engine):
            inputs.extend(getattr(source, name) for name in self._SNAPSHOT_FIELDS)
            inputs.extend(getattr(source.objective, name) for name in self._SNAPSHOT_SCORES)
        wp.launch(_snapshot, dim=max(self.steps + 1, self.objective.residual_dim), inputs=inputs, device=self.device)

    def _validate_coefficients(self, coefficients):
        """Check the same global spline bounds before a batch enters the device."""
        values = np.asarray(coefficients, dtype=np.float64)
        if values.shape != (self.world_count, self.settings.control_count, 4):
            raise ValueError("Coefficients must have shape (world_count, control_count, 4)")
        bounds = [
            self.profile[k]
            for k in (
                "equilibrium_lower",
                "equilibrium_upper",
                "equilibrium_rate_limit",
                "equilibrium_acceleration_limit",
            )
        ]
        for row in values:
            if not Spline(self.duration, row).bounds(*bounds):
                raise ValueError("Equilibrium exceeds position, rate, or acceleration limits")
        return np.ascontiguousarray(values)

    def evaluate(self, coefficients):
        """Return batch scores after device-only rollouts and one reporting boundary."""
        values = self._validate_coefficients(coefficients)
        if self.graph is None:
            self.capture(values)
        started = perf_counter()
        self.coefficients.assign(values)
        self.foundation.enabled.fill_(1)
        self.evaluate_device()
        scores = self.objective.read()
        scores.update(integrated_steps=self.integrated.numpy(), failure_code=self.failure.numpy())
        self.last_wall_s = perf_counter() - started
        return scores

    def trace(self, world: int = 0):
        """Copy one selected trace only after the batch has completed."""
        if not 0 <= world < self.world_count:
            raise IndexError("World index is outside this batch")
        recorded = int(self.recorded.numpy()[world])
        integrated = int(self.integrated.numpy()[world])
        q = wp.empty(self.steps + 1, dtype=Vec5, device=self.device)
        v = wp.empty_like(q)
        eq = wp.empty(self.steps + 1, dtype=wp.vec4d, device=self.device)
        load = wp.empty_like(eq)
        f = wp.empty(self.steps, dtype=wp.vec2d, device=self.device)
        m = wp.empty(self.steps, dtype=wp.float64, device=self.device)
        comp = wp.empty(self.steps, dtype=wp.vec3d, device=self.device)
        cap = wp.empty(self.steps, dtype=wp.int32, device=self.device)
        wp.launch(
            _gather,
            dim=self.steps + 1,
            inputs=[
                world,
                self.states,
                self.velocities,
                self.equilibrium,
                self.actuator,
                self.forces,
                self.moments,
                self.fractions,
                self.caps,
                q,
                v,
                eq,
                load,
                f,
                m,
                comp,
                cap,
            ],
            device=self.device,
        )
        q, v, eq, load = q.numpy(), v.numpy(), eq.numpy(), load.numpy()
        f, m, comp, cap = f.numpy()[:recorded], m.numpy()[:recorded], comp.numpy()[:recorded], cap.numpy()[:recorded]
        gate = self.hip_gate_values[:recorded]
        hip_stiffness = np.asarray(self.profile["hip_stiffness_n_m"], dtype=float)
        hip_damping = np.asarray(self.profile["hip_damping_ns_m"], dtype=float)
        spring_force = gate[:, None] * hip_stiffness * (eq[:recorded, :2] - q[:recorded, :2])
        damping_force = -gate[:, None] * hip_damping * v[:recorded, :2]
        body = Body(
            self.reference["lengths_m"],
            self.reference["endpoint_local_m"],
            self.profile["masses_kg"],
            self.profile["com_local_m"],
            self.profile["inertias_kg_m2"],
        )
        # Plot-only joint positions are derived after integration, never in the fitting loop.
        joints = np.asarray([body.kinematics(row) for row in q[:recorded]]).reshape(recorded, 4, 2)
        trace = {
            "time_s": self.time_s[:recorded].copy(),
            "state": q[:recorded],
            "velocity": v[:recorded],
            "joints_m": joints,
            "equilibrium": eq[:recorded],
            "hip_force_n": load[:recorded, :2],
            "hip_spring_force_n": spring_force,
            "hip_damping_force_n": damping_force,
            "hip_gate": gate.copy(),
            "joint_torque_nm": load[:recorded, 2:],
            "grf_n": f,
            "ankle_contact_moment_nm": m,
            "compression_fraction": comp[:, 0],
            "driven_compression_fraction": comp[:, 1],
            "passive_compression_fraction": comp[:, 2],
            "passive_cap_column_count": cap,
        }
        code = int(self.failure.numpy()[world])
        first_failure = int(self.failure_step.numpy()[world])
        range_step = int(self.range_step.numpy()[world])
        range_mask = int(self.range_mask.numpy()[world])
        reasons = {
            1: "Nonfinite leg state or velocity",
            2: "Nonfinite actuator load",
            4: "Joint range exceeded",
            8: "Hip height screen exceeded; leg-ground collision is not modeled",
            16: "Numerical speed screen exceeded",
            32: "Hip force screen exceeded",
            64: "Nonfinite or invalid shoe wrench",
            128: "Nonfinite shoe compression",
            256: "Driven shoe compression screen exceeded",
            512: "Shoe supplied tensile ground normal force",
            1024: "Ground force screen exceeded",
            2048: "Nonfinite integrated state or velocity",
        }
        cap_rows = np.flatnonzero(cap > 0)
        summary = {
            "status": "failed" if code else "completed",
            "failure": {
                "time_s": float(self.time_s[first_failure]),
                "reasons": [text for bit, text in reasons.items() if bit & code],
            }
            if code
            else None,
            "model": "cartesian_single_leg",
            "body_count": 3,
            "shoe_count": 1,
            "actuated_channels": 4,
            "hip_flight_gate_enabled": self.config.hip_flight_gate_enabled,
            "hip_flight_gate_ramp_s": self.config.hip_flight_gate_ramp_s,
            "hip_flight_gate_contact_threshold_n": 5.0,
            "hip_flight_gate_reference_source": "grf_time_s/grf_target_n vertical force",
            "mechanics_backend": "Warp float64 CUDA; resident batched limb/contact/objective",
            "shoe_device": str(self.device),
            "actual_dt_s": self.dt,
            "integrated_steps": integrated,
            "integrated_duration_s": integrated * self.dt,
            "requested_duration_s": self.duration,
            "terminal_state": q[integrated].tolist(),
            "terminal_velocity": v[integrated].tolist(),
            "leg_mass_kg": float(np.sum(body.masses_kg)),
            "segment_inertial_properties": {
                "body_order": ["thigh", "shank", "foot"],
                "masses_kg": np.asarray(self.profile["masses_kg"], dtype=float).tolist(),
                "com_local_m": np.asarray(self.profile["com_local_m"], dtype=float).tolist(),
                "sagittal_inertias_kg_m2": np.asarray(self.profile["inertias_kg_m2"], dtype=float).tolist(),
                "provenance": self.profile["provenance"].get("inertial"),
                "population_comparison": self.profile["provenance"].get("de_leva_1996_population_comparison"),
            },
            "controller": (
                "Fhip = gate*(Khip*(p_eq-p)-Dhip*v); tau = Kjoint*(theta_eq-theta)-Djoint*theta_dot"
                if self.config.hip_flight_gate_enabled
                else "Fhip = Khip*(p_eq-p)-Dhip*v; tau = Kjoint*(theta_eq-theta)-Djoint*theta_dot"
            ),
            "external_loads": "hip point force, gravity on three leg masses, and one shoe-ground wrench only",
            "initial_contact_state": "zero material/friction histories; not a settled or periodic contact state",
            "trace_sampling": "preintegration, one contact update per row; terminal state/velocity in summary",
            "joint_limits_diagnostic": self.config.joint_limits_diagnostic,
            "joint_exceedances": {
                "joint range": {
                    "first_time_s": float(self.time_s[range_step]),
                    "joints": [j for j in range(2) if range_mask & (1 << j)],
                }
            }
            if range_step >= 0
            else {},
            "limit_handling": "no force/torque clipping, joint stops, prescribed motion, or state projection",
            "driven_compression_limit": self.config.compression_limit,
            "maximum_driven_compression_fraction": float(comp[:, 1].max()) if recorded else None,
            "maximum_passive_compression_fraction": float(comp[:, 2].max()) if recorded else None,
            "passive_cap_fraction": float(self.foundation.surround.max_strain),
            "passive_cap_hit": bool(len(cap_rows)),
            "passive_cap_steps": len(cap_rows),
            "passive_cap_first_time_s": float(self.time_s[cap_rows[0]]) if len(cap_rows) else None,
            "passive_cap_max_columns": int(cap.max()) if recorded else 0,
            "qualification": "Exploratory numerical screens only; not physical or physiological validation. "
            "The existing passive cap and footprint attachment approximation are unchanged. "
            "No trunk, opposite leg, upper-body weight, hip torque, or measured-force input is present.",
        }
        summary["rollout_diagnostics"] = summarize_diagnostics(
            trace, self.reference, float(self.shoe.metadata["friction_mu"])
        )
        return trace, summary
