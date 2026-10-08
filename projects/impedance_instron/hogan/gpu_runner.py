# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""CUDA candidate rollouts of the causal runner, without tracking-plan inputs.

The chain, actuator, oscillator, and load memory use float64; the unchanged shoe
runtime uses float32. Worlds are candidate-major within each group. Groups share
one shoe instance and an *exact* integration timestep, including the duration /
ceil(duration / maximum_dt) adjustment. Different geometries and timesteps never
silently inherit the first trial's settings.

Only graph chunks are dispatched from Python during integration. Full traces
remain resident until all groups finish; summaries and offline identification
scores can then be computed on the host. Trace memory is O(candidates * trials *
steps), not bounded by chunk_steps. max_trials_per_group bounds padding and the
world count per group, not aggregate trace storage; callers should batch large
datasets/populations themselves. Separate but equivalent Shoe objects are kept
in separate groups deliberately, avoiding assumptions about mutable shoe setup.
"""

from __future__ import annotations

import math
from copy import deepcopy
from time import perf_counter
from types import SimpleNamespace

import numpy as np
import warp as wp

from projects.digital_shoe.runtime import clone_params

from ..cartesian.gpu.foundation import FoundationFused
from ..cartesian.shoe import Shoe
from .batch import (
    ChainParams,
    Settings,
    Vec6,
    _angular,
    _ankle,
    _chain_params,
    _cholesky_solve,
    _dynamics,
    _finite6,
    _stage,
    _tick,
)
from .mechanics import Chain
from .runner import FEATURE_NAMES, RolloutConfig, Runner, State, Task

wp.set_module_options({"enable_backward": False, "fuse_fp": False})

_FEATURE_COUNT = wp.constant(len(FEATURE_NAMES))
_Features = wp.types.vector(_FEATURE_COUNT, wp.float64)
_TWO_PI = wp.constant(wp.float64(2.0 * math.pi))
_HALF_PI = wp.constant(wp.float64(math.pi / 2.0))
_FAILURES = {
    2: "Hip height screen exceeded",
    3: "Numerical speed screen exceeded",
    4: "Nonfinite actuator/contact output",
    5: "Contact force screen exceeded",
    6: "Driven shoe compression screen exceeded",
    7: "Singular mass matrix",
    8: "Nonfinite integrated state",
    9: "Integrated state exceeded height/speed screen",
}


@wp.struct
class _Model:
    lower: wp.vec3d
    upper: wp.vec3d
    stiffness: wp.vec3d
    damping: wp.vec3d
    torque_cap: wp.vec3d
    torque_step: wp.vec3d
    response_fraction: wp.float64
    load_fraction: wp.float64
    frequency: wp.float64
    reference_speed: wp.float64
    speed_scale: wp.float64
    cadence_gain: wp.float64
    phase_feedback: wp.float64
    intrinsic_damping: wp.vec3d


@wp.struct
class _Output:
    load: Vec6
    grf_n: wp.vec2d
    ankle_contact_moment_nm: wp.float64
    compression_fraction: wp.float64
    equilibrium_rad: wp.vec3d
    stiffness_nm_rad: wp.vec3d
    damping_nms_rad: wp.vec3d
    torque_saturated: wp.vec3i


@wp.struct
class _Buffers:
    state: wp.array[Vec6]
    velocity: wp.array[Vec6]
    torque: wp.array[wp.vec3d]
    sensory: wp.array[wp.vec2d]
    status: wp.array[int]
    recorded: wp.array[int]
    clock: wp.array[int]
    enabled: wp.array[int]
    fraction: wp.array[wp.float64]
    invalid_compression: wp.array[int]
    states: wp.array2d[Vec6]
    velocities: wp.array2d[Vec6]
    senses: wp.array2d[wp.vec2d]
    outputs: wp.array2d[_Output]


@wp.kernel
def _reset(
    q0: wp.array[Vec6],
    v0: wp.array[Vec6],
    torque0: wp.array[wp.vec3d],
    sensory0: wp.array[wp.vec2d],
    trials: int,
    data: _Buffers,
):
    w = wp.tid()
    s = w % trials
    data.state[w] = q0[s]
    data.velocity[w] = v0[s]
    data.torque[w] = torque0[s]
    data.sensory[w] = sensory0[s]
    data.status[w] = 0
    data.recorded[w] = 0
    data.enabled[w] = 1
    data.states[0, w] = q0[s]
    data.velocities[0, w] = v0[s]
    data.senses[0, w] = sensory0[s]


@wp.func
def _too_fast(v: Vec6, maximum: wp.float64):
    fast = wp.length(wp.vec2d(v[0], v[1])) > maximum
    for j in range(2, 6):
        fast = fast or wp.abs(v[j]) > maximum
    return fast


@wp.kernel
def _actuate(
    params: wp.array[ChainParams],
    models: wp.array[_Model],
    weights: wp.array3d[wp.float64],
    speeds: wp.array[wp.float64],
    steps: wp.array[int],
    cfg: Settings,
    data: _Buffers,
):
    w = wp.tid()
    if data.status[w] != 0:
        return
    s = w % cfg.stance_count
    c = w // cfg.stance_count
    k = data.clock[0]
    code = int(0)
    q = data.state[w]
    v = data.velocity[w]
    if k >= steps[s]:
        code = 1
    elif q[1] < cfg.hip_floor:
        code = 2
    elif _too_fast(v, cfg.max_speed):
        code = 3
    if code != 0:
        data.status[w] = code
        data.enabled[w] = 0
        return
    data.fraction[w] = wp.float64(0.0)
    data.invalid_compression[w] = 0
    p = params[s]
    m = models[c]
    sense = data.sensory[w]
    ankle, _jx, _jz, _angle = _ankle(q, p)
    phase = sense[0]
    load_bw = wp.clamp(sense[1], wp.float64(0.0), wp.float64(4.0))
    features = _Features(
        wp.float64(1.0),
        wp.sin(phase),
        wp.cos(phase),
        wp.sin(wp.float64(2.0) * phase),
        wp.cos(wp.float64(2.0) * phase),
        wp.sin(wp.float64(3.0) * phase),
        wp.cos(wp.float64(3.0) * phase),
        load_bw,
        load_bw * wp.sin(phase),
        load_bw * wp.cos(phase),
        wp.clamp((q[0] - ankle[0]) / (p.lengths[0] + p.lengths[1]), wp.float64(-1.0), wp.float64(1.0)),
        wp.clamp(q[2] - _HALF_PI, wp.float64(-1.0), wp.float64(1.0)),
        wp.clamp((speeds[s] - v[0]) / m.speed_scale, wp.float64(-2.0), wp.float64(2.0)),
        wp.clamp((speeds[s] - m.reference_speed) / m.speed_scale, wp.float64(-2.0), wp.float64(2.0)),
    )
    equilibrium = wp.vec3d(wp.float64(0.0))
    stiffness = wp.vec3d(wp.float64(0.0))
    damping = wp.vec3d(wp.float64(0.0))
    for output in range(3):
        for joint in range(3):
            logit = wp.float64(0.0)
            for feature in range(_FEATURE_COUNT):
                logit += weights[c, output * 3 + joint, feature] * features[feature]
            logit = wp.clamp(logit, wp.float64(-40.0), wp.float64(40.0))
            fraction = wp.float64(1.0) / (wp.float64(1.0) + wp.exp(-logit))
            if output == 0:
                equilibrium[joint] = m.lower[joint] + (m.upper[joint] - m.lower[joint]) * fraction
            elif output == 1:
                stiffness[joint] = m.stiffness[joint] * fraction
            else:
                damping[joint] = m.damping[joint] * fraction
    load = Vec6(wp.float64(0.0))
    torque = data.torque[w]
    saturated = wp.vec3i(0)
    for j in range(3):
        raw = stiffness[j] * (equilibrium[j] - q[j + 3]) - damping[j] * v[j + 3]
        cap = m.torque_cap[j]
        desired = wp.clamp(raw, -cap, cap)
        change = wp.clamp(m.response_fraction * (desired - torque[j]), -m.torque_step[j], m.torque_step[j])
        torque[j] = wp.clamp(torque[j] + change, -cap, cap)
        load[j + 3] = wp.clamp(torque[j] - m.intrinsic_damping[j] * v[j + 3], -cap, cap)
        saturated[j] = int(wp.abs(raw) > cap)
    data.torque[w] = torque
    out = _Output()
    out.load = load
    out.equilibrium_rad = equilibrium
    out.stiffness_nm_rad = stiffness
    out.damping_nms_rad = damping
    out.torque_saturated = saturated
    data.outputs[k, w] = out


@wp.kernel
def _compression(
    columns: int,
    driven: wp.array[int],
    rest: wp.array[wp.float64],
    compression: wp.array[wp.float32],
    data: _Buffers,
):
    i = wp.tid()
    w = i // columns
    c = i % columns
    if data.status[w] != 0 or driven[c] == 0:
        return
    # The CPU divides float32 compression by the artifact's float64 thickness.
    fraction = wp.float64(compression[i]) / rest[c]
    if not wp.isfinite(fraction):
        wp.atomic_max(data.invalid_compression, w, 1)
    else:
        wp.atomic_max(data.fraction, w, fraction)


@wp.kernel
def _advance(
    params: wp.array[ChainParams],
    models: wp.array[_Model],
    speeds: wp.array[wp.float64],
    steps: wp.array[int],
    cfg: Settings,
    dt: wp.float64,
    body_f: wp.array[wp.spatial_vector],
    data: _Buffers,
):
    w = wp.tid()
    if data.status[w] != 0:
        return
    s = w % cfg.stance_count
    k = data.clock[0]
    p = params[s]
    m = models[w // cfg.stance_count]
    q = data.state[w]
    v = data.velocity[w]
    out = data.outputs[k, w]
    f = body_f[w]
    fx = wp.float64(f[0])
    fz = wp.float64(f[2])
    moment = -wp.float64(f[4])
    compression = data.fraction[w]
    code = int(0)
    if (
        not _finite6(out.load)
        or not (wp.isfinite(fx) and wp.isfinite(fz) and wp.isfinite(moment))
        or data.invalid_compression[w] != 0
    ):
        code = 4
    elif fz < wp.float64(-1.0e-6) or wp.length(wp.vec2d(fx, fz)) > cfg.max_force:
        code = 5
    elif compression > cfg.compression_limit + wp.float64(1.0e-6):
        code = 6
    if code != 0:
        data.status[w] = code
        data.enabled[w] = 0
        return
    mass, bias = _dynamics(q, v, p, cfg.gravity)
    _position, jx, jz, _angle = _ankle(q, p)
    force = out.load + fx * jx + fz * jz + moment * _angular(3)
    acceleration, ok = _cholesky_solve(mass, force - bias)
    velocity = v + dt * acceleration
    position = q + dt * velocity
    if not ok:
        code = 7
    elif not _finite6(position) or not _finite6(velocity):
        code = 8
    elif position[1] < cfg.hip_floor or _too_fast(velocity, cfg.max_speed):
        code = 9
    if code != 0:
        data.status[w] = code
        data.enabled[w] = 0
        return
    # Commit only accepted intervals, then update sensors using this contact.
    out.grf_n = wp.vec2d(fx, fz)
    out.ankle_contact_moment_nm = moment
    out.compression_fraction = compression
    data.outputs[k, w] = out
    sense = data.sensory[w]
    weight = (p.masses[0] + p.masses[1] + p.masses[2] + p.masses[3]) * cfg.gravity
    load_bw = wp.max(fz, wp.float64(0.0)) / weight
    filtered = sense[1] + m.load_fraction * (load_bw - sense[1])
    speed = wp.clamp((speeds[s] - m.reference_speed) / m.speed_scale, wp.float64(-2.0), wp.float64(2.0))
    frequency = m.frequency * wp.exp(wp.clamp(m.cadence_gain * speed, wp.float64(-2.0), wp.float64(2.0)))
    modulation = wp.float64(1.0) - m.phase_feedback * wp.min(filtered, wp.float64(1.0)) * wp.sin(sense[0])
    phase = sense[0] + _TWO_PI * frequency * modulation * dt
    phase = phase - wp.floor(phase / _TWO_PI) * _TWO_PI
    sense = wp.vec2d(phase, filtered)
    data.state[w] = position
    data.velocity[w] = velocity
    data.sensory[w] = sense
    data.states[k + 1, w] = position
    data.velocities[k + 1, w] = velocity
    data.senses[k + 1, w] = sense
    data.recorded[w] = k + 1
    if k + 1 == steps[s]:
        data.status[w] = 1
        data.enabled[w] = 0


def _model_params(model: Runner, dt: float) -> _Model:
    p = _Model()
    bounds = model.bounds
    p.lower = wp.vec3d(*bounds.equilibrium_lower_rad)
    p.upper = wp.vec3d(*bounds.equilibrium_upper_rad)
    p.stiffness = wp.vec3d(*bounds.stiffness_max_nm_rad)
    p.damping = wp.vec3d(*bounds.damping_max_nms_rad)
    p.torque_cap = wp.vec3d(*bounds.torque_max_nm)
    p.torque_step = wp.vec3d(*(np.asarray(bounds.torque_rate_max_nm_s) * dt))
    # expm1 is unavailable in Warp; compute these constant fractions exactly once.
    p.response_fraction = -math.expm1(-dt / model.response_time_s)
    p.load_fraction = -math.expm1(-dt / model.load_time_s)
    p.frequency = model.frequency_hz
    p.reference_speed = model.reference_speed_m_s
    p.speed_scale = model.speed_scale_m_s
    p.cadence_gain = model.cadence_speed_gain
    p.phase_feedback = model.phase_feedback
    p.intrinsic_damping = wp.vec3d(*model.intrinsic_damping_nms_rad)
    return p


def _summary(trace: dict, code: int, duration: float, dt: float, cfg: RolloutConfig) -> dict:
    count = len(trace["load"])
    failure = None if code == 1 else _FAILURES[code]
    power = trace["load"][:, 3:] * trace["velocity"][:-1, 3:]
    contact = trace["grf_n"][:, 1] > cfg.contact_threshold_n
    indices = np.flatnonzero(contact)
    return {
        "status": "completed" if failure is None else "failed",
        "failure": failure,
        "failure_time_s": None if failure is None else count * dt,
        "dt_s": dt,
        "contact_threshold_n": cfg.contact_threshold_n,
        "requested_duration_s": duration,
        "integrated_duration_s": count * dt,
        "reference_inputs_used": False,
        "external_actuator_load": [0.0, 0.0, 0.0],
        "peak_grf_n": trace["grf_n"].max(0).tolist() if count else [0.0, 0.0],
        "grf_impulse_ns": (trace["grf_n"].sum(0) * dt).tolist(),
        "contact_duration_s": float(np.count_nonzero(contact) * dt),
        "touchdown_s": float(indices[0] * dt) if len(indices) else None,
        "toeoff_s": float((indices[-1] + 1) * dt) if len(indices) else None,
        "joint_positive_work_j": (np.maximum(power, 0).sum(0) * dt).tolist(),
        "joint_negative_work_j": (np.minimum(power, 0).sum(0) * dt).tolist(),
        "torque_saturation_fraction": trace["torque_saturated"].mean(0).tolist() if count else [0.0] * 3,
        "terminal_state": trace["state"][-1].tolist(),
        "terminal_velocity": trace["velocity"][-1].tolist(),
        "validated": False,
    }


class _Group:
    def __init__(self, batch, indices, chains, shoes, initials, tasks):
        self.indices = tuple(indices)
        self.device = batch.device
        self.config = batch.config
        self.candidates = batch.candidates
        self.trial_count = len(indices)
        self.world_count = self.candidates * self.trial_count
        self.steps = batch.steps[list(indices)]
        self.dt = float(batch.dts[indices[0]])
        self.durations = batch.durations_s[list(indices)]
        self.chunk_steps = min(batch.chunk_steps, int(self.steps.max()))
        self.graph = None
        self.max_steps = int(self.steps.max())
        d, w = self.device, self.world_count
        self.params = wp.array([_chain_params(chains[i]) for i in indices], dtype=ChainParams, device=d)
        self.models = wp.zeros(self.candidates, dtype=_Model, device=d)
        self.weights = wp.zeros((self.candidates, 9, _FEATURE_COUNT), dtype=wp.float64, device=d)
        self.speeds = wp.array([tasks[i].speed_m_s for i in indices], dtype=wp.float64, device=d)
        self.steps_d = wp.array(self.steps, dtype=int, device=d)
        self.q0 = wp.array(np.array([initials[i].q for i in indices]), dtype=Vec6, device=d)
        self.v0 = wp.array(np.array([initials[i].v for i in indices]), dtype=Vec6, device=d)
        self.torque0 = wp.array(np.array([initials[i].torque_nm for i in indices]), dtype=wp.vec3d, device=d)
        self.sensory0 = wp.array(
            np.array([[initials[i].phase_rad, initials[i].normal_load_bw] for i in indices]), dtype=wp.vec2d, device=d
        )
        shoe = shoes[indices[0]]
        source = shoe.foundation
        bed = shoe.shoe.column_bed
        self.foundation = FoundationFused(
            source.anchor_local.numpy(),
            np.full(len(bed.rest_length_m), source.config.ground_height_m),
            source.rest_len.numpy(),
            source.area.numpy(),
            source.neighbors.numpy(),
            bed.spacing_m,
            shoe.shoe.material,
            np.arange(w),
            wp.zeros(w, dtype=wp.vec3, device=d),
            deepcopy(source.config),
            d,
            deepcopy(source.surround),
            world_count=w,
        )
        # Preserve runtime material/config blocks and default-adapter parameters,
        # rather than reconstructing them from artifact defaults.
        foundation = self.foundation
        foundation.world_blocks = [clone_params(source.world_blocks[0]) for _ in range(w)]
        foundation.world_params.assign(foundation.world_blocks)
        foundation._materials_dirty = True
        wp.copy(foundation.friction_kt, source.friction_kt)
        wp.copy(foundation.friction_kv, source.friction_kv)
        if source.friction_solver is not None:
            adapter = foundation.friction_solver
            adapter.settings.assign(np.repeat(source.friction_solver.settings.numpy(), w, axis=0))
            wp.copy(adapter.base_kt, source.friction_solver.base_kt)
            wp.copy(adapter.base_kv, source.friction_solver.base_kv)
        self.rest = wp.array(bed.rest_length_m, dtype=wp.float64, device=d)
        self.carriers = SimpleNamespace(
            body_q=wp.zeros(w, dtype=wp.transform, device=d),
            body_qd=wp.zeros(w, dtype=wp.spatial_vector, device=d),
            body_f=wp.zeros(w, dtype=wp.spatial_vector, device=d),
        )
        cfg = self.config
        settings = Settings()
        settings.gravity = cfg.gravity_m_s2
        settings.threshold = cfg.contact_threshold_n
        settings.compression_limit = cfg.compression_limit
        settings.max_force = cfg.maximum_force_n
        settings.hip_floor = cfg.minimum_hip_height_m
        settings.max_speed = cfg.maximum_speed
        settings.pitch = shoe.static_pitch_rad
        settings.stance_count = self.trial_count
        self.settings = settings
        data = _Buffers()
        data.state = wp.zeros(w, dtype=Vec6, device=d)
        data.velocity = wp.zeros(w, dtype=Vec6, device=d)
        data.torque = wp.zeros(w, dtype=wp.vec3d, device=d)
        data.sensory = wp.zeros(w, dtype=wp.vec2d, device=d)
        data.status = wp.zeros(w, dtype=int, device=d)
        data.recorded = wp.zeros(w, dtype=int, device=d)
        data.clock = wp.zeros(1, dtype=int, device=d)
        data.enabled = self.foundation.enabled
        data.fraction = wp.zeros(w, dtype=wp.float64, device=d)
        data.invalid_compression = wp.zeros(w, dtype=int, device=d)
        data.states = wp.zeros((self.max_steps + 1, w), dtype=Vec6, device=d)
        data.velocities = wp.zeros((self.max_steps + 1, w), dtype=Vec6, device=d)
        data.senses = wp.zeros((self.max_steps + 1, w), dtype=wp.vec2d, device=d)
        data.outputs = wp.zeros((self.max_steps, w), dtype=_Output, device=d)
        self.data = data

    def _reset(self):
        self.foundation.reset()
        self.data.clock.zero_()
        wp.launch(
            _reset,
            dim=self.world_count,
            inputs=[self.q0, self.v0, self.torque0, self.sensory0, self.trial_count, self.data],
            device=self.device,
        )

    def _step(self):
        data, d = self.data, self.device
        wp.launch(
            _actuate,
            dim=self.world_count,
            inputs=[self.params, self.models, self.weights, self.speeds, self.steps_d, self.settings, data],
            device=d,
        )
        wp.launch(
            _stage,
            dim=self.world_count,
            inputs=[
                self.params,
                self.settings,
                data.status,
                data.state,
                data.velocity,
                self.carriers.body_q,
                self.carriers.body_qd,
            ],
            device=d,
        )
        self.foundation.apply(self.carriers, self.dt, clear_body_force=True)
        wp.launch(
            _compression,
            dim=self.world_count * self.foundation.column_count,
            inputs=[self.foundation.column_count, self.foundation.driven, self.rest, self.foundation.compression, data],
            device=d,
        )
        wp.launch(
            _advance,
            dim=self.world_count,
            inputs=[
                self.params,
                self.models,
                self.speeds,
                self.steps_d,
                self.settings,
                self.dt,
                self.carriers.body_f,
                data,
            ],
            device=d,
        )
        wp.launch(_tick, dim=1, inputs=[data.clock], device=d)

    def launch(self, models):
        self.models.assign([_model_params(model, self.dt) for model in models])
        self.weights.assign(np.array([model.weights.reshape(9, _FEATURE_COUNT) for model in models]))
        self._reset()
        if self.graph is None:
            # Settle compilation and shoe dt caches before capture, not at every step.
            self._step()
            wp.synchronize_device(self.device)
            self._reset()
            with wp.ScopedCapture(device=self.device) as capture:
                for _ in range(self.chunk_steps):
                    self._step()
            self.graph = capture.graph
        for _ in range(math.ceil(self.max_steps / self.chunk_steps)):
            wp.capture_launch(self.graph)

    def collect(self):
        data = self.data
        status, counts = data.status.numpy(), data.recorded.numpy()
        states, velocities = data.states.numpy(), data.velocities.numpy()
        senses, outputs = data.senses.numpy(), data.outputs.numpy()
        result = []
        for c in range(self.candidates):
            row = []
            for s in range(self.trial_count):
                w = c * self.trial_count + s
                count = int(counts[w])
                trace = {key: outputs[key][:count, w].copy() for key in outputs.dtype.names}
                trace["torque_saturated"] = trace["torque_saturated"].astype(bool)
                trace.update(
                    time_s=np.arange(count + 1) * self.dt,
                    state=states[: count + 1, w].copy(),
                    velocity=velocities[: count + 1, w].copy(),
                    phase_rad=senses[: count + 1, w, 0].copy(),
                    normal_load_bw=senses[: count + 1, w, 1].copy(),
                )
                row.append((trace, _summary(trace, int(status[w]), float(self.durations[s]), self.dt, self.config)))
            result.append(row)
        return result


class GpuBatch:
    """Evaluate candidate Runner models over fixed, independent predictive inputs.

    Each evaluation resets all runner and shoe history, never modifying supplied
    shoes or initial states. Groups are dispatched sequentially on one CUDA stream,
    but all candidate/trial worlds *within* a group run concurrently. No measured
    trajectory, feedforward, target force, or reference event is accepted.

    Args:
        chains: Body mechanics, one per trial.
        shoes: Ordinary ground-plane Shoe instances, one per trial. Reuse an
            instance to batch trials with identical geometry/material/config.
            Runtime material blocks and default-adapter settings are copied at
            construction; custom friction adapters are not supported.
        initials: Complete causal runner initial states, one per trial.
        tasks: Known task inputs, one per trial.
        durations_s: Positive prediction horizons [s], one per trial.
        candidates: Exact number of models evaluated concurrently in each group.
        config: Shared integration and numerical-screen settings.
        device: CUDA device; CPU execution is deliberately not a fallback.
        chunk_steps: Maximum steps per reusable captured graph.
        max_trials_per_group: Limit trials per same-shoe, same-dt group.

    Attributes:
        group_trial_indices: Trial indices in each execution group.
        steps: Exact integration step count for each trial.
        dts: Actual integration timestep for each trial [s].
        setup_wall_s: Constructor wall time [s], excluding lazy graph capture.
        last_wall_s: Last evaluation wall time [s], including copies, summaries,
            and first-use compilation/capture where applicable.
    """

    def __init__(
        self,
        chains: list[Chain],
        shoes: list[Shoe],
        initials: list[State],
        tasks: list[Task],
        durations_s,
        *,
        candidates: int,
        config: RolloutConfig | None = None,
        device: str = "cuda:0",
        chunk_steps: int = 64,
        max_trials_per_group: int = 16,
    ):
        started = perf_counter()
        self.config = config or RolloutConfig()
        self.trial_count = len(chains)
        if not self.trial_count or any(len(values) != self.trial_count for values in (shoes, initials, tasks)):
            raise ValueError("chains, shoes, initials, and tasks must be nonempty and have equal length")
        for name, value in (
            ("candidates", candidates),
            ("chunk_steps", chunk_steps),
            ("max_trials_per_group", max_trials_per_group),
        ):
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        self.candidates = int(candidates)
        self.chunk_steps = int(chunk_steps)
        self.durations_s = np.array(durations_s, dtype=float, copy=True)
        if (
            self.durations_s.shape != (self.trial_count,)
            or not np.isfinite(self.durations_s).all()
            or np.any(self.durations_s <= 0)
        ):
            raise ValueError("durations_s must contain one finite positive horizon per trial")
        steps = [math.ceil(float(duration) / self.config.dt_s) for duration in self.durations_s]
        if max(steps) > np.iinfo(np.int32).max - self.chunk_steps:
            raise ValueError("Trial step count exceeds the GPU clock range")
        self.steps = np.asarray(steps, dtype=np.int32)
        self.dts = self.durations_s / self.steps
        self.initials = tuple(initial.copy() for initial in initials)
        for shoe in shoes:
            source = shoe.foundation
            if source.world_count != 1 or source.config.ground_height_m is None or source.surround is None:
                raise ValueError("Each shoe needs a single-world ground-plane foundation with a surround mask")
            if source.friction_solver is not None and not getattr(source.friction_solver, "is_default", False):
                raise ValueError("Custom friction adapters are not supported")
        self.device = wp.get_device(device)
        if not self.device.is_cuda:
            raise ValueError("GpuBatch requires a CUDA device")
        groups = {}
        for i, (shoe, dt) in enumerate(zip(shoes, self.dts, strict=True)):
            groups.setdefault((id(shoe), float(dt)), []).append(i)
        self.group_trial_indices = tuple(
            tuple(indices[start : start + max_trials_per_group])
            for indices in groups.values()
            for start in range(0, len(indices), max_trials_per_group)
        )
        self._groups = [
            _Group(self, indices, chains, shoes, self.initials, tasks) for indices in self.group_trial_indices
        ]
        self.setup_wall_s = perf_counter() - started
        self.last_wall_s = None

    def evaluate(self, models: list[Runner]) -> list[list[tuple[dict, dict]]]:
        """Return candidate-major, then input-trial-order ``(trace, summary)`` pairs.

        Both dictionaries have the same fields and accepted-interval convention
        as :func:`runner.simulate`. All traces are recorded, including the initial
        and last accepted states on failure. Returned arrays own their storage.
        Offline ``identify.score(trace, summary, trial, model)`` can consume each
        pair directly; targets never enter the GPU dynamics or this API.
        """
        if len(models) != self.candidates:
            raise ValueError(f"Expected exactly {self.candidates} Runner models")
        for model in models:
            if not isinstance(model, Runner):
                raise TypeError("models must contain Runner instances")
            if any(np.any(np.abs(initial.torque_nm) > model.bounds.torque_max_nm) for initial in self.initials):
                raise ValueError("Initial torque exceeds the model bounds")
        started = perf_counter()
        for group in self._groups:
            group.launch(models)
        # Download only after dispatching every rollout; no host per-step reads.
        indexed = [{} for _ in models]
        for group in self._groups:
            values = group.collect()
            for c, row in enumerate(values):
                indexed[c].update(zip(group.indices, row, strict=True))
        result = [[row[s] for s in range(self.trial_count)] for row in indexed]
        self.last_wall_s = perf_counter() - started
        return result
