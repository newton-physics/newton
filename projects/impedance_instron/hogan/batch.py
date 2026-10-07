# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Roll out many stances under many impedance schedules in lockstep on CUDA.

World ``w`` runs stance ``w % stance_count`` under gain candidate
``w // stance_count``. Each world reproduces :func:`.rollout.simulate`: the same
control law, semi-implicit Euler step, contact staging, and failure screens,
with the chain mechanics in double precision and the batched shoe of
:class:`..cartesian.gpu.foundation.FoundationFused`. Only per-world summaries
are kept on the device, so batch size is bounded by the shoe state.

Gains are scheduled on a normalized gait phase shared by all stances:
``[0, 1)`` from the start of the window to the reference touchdown, ``[1, 2)``
over reference contact, and ``[2, 3]`` to the end of the window.
"""

from __future__ import annotations

import math
from time import perf_counter
from types import SimpleNamespace

import numpy as np
import warp as wp

from projects.digital_shoe.runtime import FoundationConfig

from ..cartesian.gpu.foundation import FoundationFused
from ..cartesian.shoe import Shoe
from .control import Impedance
from .mechanics import Chain
from .plan import Plan
from .rollout import Config

wp.set_module_options({"enable_backward": False, "fuse_fp": False})

Vec6 = wp.types.vector(6, wp.float64)
Mat6 = wp.types.matrix((6, 6), wp.float64)
_PI = wp.constant(wp.float64(math.pi))
_HALF_PI = wp.constant(wp.float64(0.5 * math.pi))

FAILURE_REASONS = {
    2: "Nonfinite actuator load or contact wrench",
    4: "Hip height screen exceeded",
    8: "Numerical speed screen exceeded",
    16: "Ground force screen exceeded",
    32: "Driven shoe compression screen exceeded",
    64: "Shoe supplied tensile ground normal force",
    128: "Singular mass matrix",
    256: "Nonfinite integrated state or velocity",
}


@wp.struct
class ChainParams:
    lengths: wp.vec2d
    endpoint: wp.vec2d
    masses: wp.vec4d
    inertias: wp.vec4d
    com_x: wp.vec4d
    com_z: wp.vec4d


@wp.struct
class Settings:
    gravity: wp.float64
    threshold: wp.float64
    compression_limit: wp.float64
    max_force: wp.float64
    hip_floor: wp.float64
    max_speed: wp.float64
    pitch: wp.float64
    touchdown_phase: int
    stance_count: int


@wp.func
def _rotate(angle: wp.float64, x: wp.float64, z: wp.float64):
    c = wp.cos(angle)
    s = wp.sin(angle)
    return wp.vec2d(c * x - s * z, s * x + c * z)


@wp.func
def _angular(body: int):
    a = Vec6(wp.float64(0.0))
    for j in range(4):
        if j <= body:
            a[j + 2] = wp.float64(1.0)
    return a


@wp.func
def _angles(q: Vec6):
    a0 = q[2]
    a1 = a0 + q[3] + _PI
    a2 = a1 + q[4]
    a3 = a2 + q[5] + _HALF_PI
    return wp.vec4d(a0, a1, a2, a3)


@wp.func
def _ankle(q: Vec6, p: ChainParams):
    """Return ankle position [m], its x/z Jacobian rows, and the foot angle [rad]."""
    ang = _angles(q)
    r1 = _rotate(ang[1], p.lengths[0], wp.float64(0.0))
    r2 = _rotate(ang[2], p.lengths[1], wp.float64(0.0))
    jx = Vec6(wp.float64(0.0))
    jz = Vec6(wp.float64(0.0))
    jx[0] = wp.float64(1.0)
    jz[1] = wp.float64(1.0)
    a1 = _angular(1)
    a2 = _angular(2)
    jx = jx - r1[1] * a1 - r2[1] * a2
    jz = jz + r1[0] * a1 + r2[0] * a2
    position = wp.vec2d(q[0] + r1[0] + r2[0], q[1] + r1[1] + r2[1])
    return position, jx, jz, ang[3]


@wp.func
def _body_terms(
    mass: Mat6,
    bias: Vec6,
    m: wp.float64,
    inertia: wp.float64,
    a: Vec6,
    ox: Vec6,
    oz: Vec6,
    angle: wp.float64,
    cx: wp.float64,
    cz: wp.float64,
    omega: wp.float64,
    prior: wp.vec2d,
    gravity: wp.float64,
):
    c = _rotate(angle, cx, cz)
    jx = ox - c[1] * a
    jz = oz + c[0] * a
    w2 = omega * omega
    ax = prior[0] - c[0] * w2
    az = prior[1] - c[1] * w2 + gravity
    mass = mass + m * (wp.outer(jx, jx) + wp.outer(jz, jz)) + inertia * wp.outer(a, a)
    bias = bias + m * (ax * jx + az * jz)
    return mass, bias


@wp.func
def _dynamics(q: Vec6, v: Vec6, p: ChainParams, gravity: wp.float64):
    """Mirror :meth:`.mechanics.Chain.dynamics`: ``M @ acceleration + bias = load``."""
    ang = _angles(q)
    r1 = _rotate(ang[1], p.lengths[0], wp.float64(0.0))
    r2 = _rotate(ang[2], p.lengths[1], wp.float64(0.0))
    a0 = _angular(0)
    a1 = _angular(1)
    a2 = _angular(2)
    a3 = _angular(3)
    w1 = wp.dot(a1, v)
    w2 = wp.dot(a2, v)
    ox = Vec6(wp.float64(0.0))
    oz = Vec6(wp.float64(0.0))
    ox[0] = wp.float64(1.0)
    oz[1] = wp.float64(1.0)
    mass = Mat6(wp.float64(0.0))
    bias = Vec6(wp.float64(0.0))
    zero = wp.vec2d(wp.float64(0.0), wp.float64(0.0))
    mass, bias = _body_terms(
        mass, bias, p.masses[0], p.inertias[0], a0, ox, oz, ang[0], p.com_x[0], p.com_z[0], wp.dot(a0, v), zero, gravity
    )
    mass, bias = _body_terms(
        mass, bias, p.masses[1], p.inertias[1], a1, ox, oz, ang[1], p.com_x[1], p.com_z[1], w1, zero, gravity
    )
    ox2 = ox - r1[1] * a1
    oz2 = oz + r1[0] * a1
    prior2 = -r1 * (w1 * w1)
    mass, bias = _body_terms(
        mass, bias, p.masses[2], p.inertias[2], a2, ox2, oz2, ang[2], p.com_x[2], p.com_z[2], w2, prior2, gravity
    )
    ox3 = ox2 - r2[1] * a2
    oz3 = oz2 + r2[0] * a2
    prior3 = prior2 - r2 * (w2 * w2)
    mass, bias = _body_terms(
        mass,
        bias,
        p.masses[3],
        p.inertias[3],
        a3,
        ox3,
        oz3,
        ang[3],
        p.com_x[3],
        p.com_z[3],
        wp.dot(a3, v),
        prior3,
        gravity,
    )
    return mass, bias


@wp.func
def _cholesky_solve(m: Mat6, b: Vec6):
    """Solve the symmetric positive-definite system; report failure on a nonpositive pivot."""
    lower = Mat6(wp.float64(0.0))
    ok = True
    for j in range(6):
        d = m[j, j]
        for k in range(j):
            d -= lower[j, k] * lower[j, k]
        if not (d > wp.float64(0.0)):
            ok = False
            d = wp.float64(1.0)
        d = wp.sqrt(d)
        lower[j, j] = d
        for i in range(j + 1, 6):
            t = m[i, j]
            for k in range(j):
                t -= lower[i, k] * lower[j, k]
            lower[i, j] = t / d
    y = Vec6(wp.float64(0.0))
    for i in range(6):
        t = b[i]
        for k in range(i):
            t -= lower[i, k] * y[k]
        y[i] = t / lower[i, i]
    x = Vec6(wp.float64(0.0))
    for ii in range(6):
        i = 5 - ii
        t = y[i]
        for k in range(i + 1, 6):
            t -= lower[k, i] * x[k]
        x[i] = t / lower[i, i]
    return x, ok


@wp.func
def _sample(values: wp.array2d[Vec6], s: int, n: int, position: wp.float64):
    x = wp.clamp(position, wp.float64(0.0), wp.float64(n))
    i = wp.min(int(x), n - 1)
    w = x - wp.float64(i)
    return (wp.float64(1.0) - w) * values[i, s] + w * values[i + 1, s]


@wp.func
def _gait_phase(phase: wp.float64, timing: wp.vec3d):
    """Map the plan clock [s] to the normalized gait phase described in the module docstring."""
    touchdown = timing[0]
    toeoff = timing[1]
    tiny = wp.float64(1.0e-9)
    if phase < touchdown:
        return wp.max(phase, wp.float64(0.0)) / wp.max(touchdown, tiny)
    if phase < toeoff:
        return wp.float64(1.0) + (phase - touchdown) / wp.max(toeoff - touchdown, tiny)
    return wp.min(wp.float64(2.0) + (phase - toeoff) / wp.max(timing[2] - toeoff, tiny), wp.float64(3.0))


@wp.func
def _gains(knots: wp.array[wp.float64], table: wp.array2d[Vec6], c: int, phi: wp.float64):
    n = knots.shape[0]
    if n == 1 or phi <= knots[0]:
        return table[c, 0]
    if phi >= knots[n - 1]:
        return table[c, n - 1]
    i = int(0)
    for j in range(n - 1):
        if knots[j] <= phi:
            i = j
    w = (phi - knots[i]) / (knots[i + 1] - knots[i])
    return (wp.float64(1.0) - w) * table[c, i] + w * table[c, i + 1]


@wp.func
def _finite6(x: Vec6):
    ok = True
    for i in range(6):
        ok = ok and wp.isfinite(x[i])
    return ok


@wp.kernel
def _reset(
    q0: wp.array[Vec6],
    v0: wp.array[Vec6],
    stance_count: int,
    state: wp.array[Vec6],
    velocity: wp.array[Vec6],
    touchdown: wp.array[wp.float64],
    status: wp.array[int],
    recorded: wp.array[int],
    contact_steps: wp.array[int],
    sq_q: wp.array[Vec6],
    sq_f: wp.array[wp.vec2d],
    peak: wp.array[wp.vec2d],
    max_fraction: wp.array[wp.float64],
    fraction: wp.array[wp.float32],
):
    w = wp.tid()
    s = w % stance_count
    state[w] = q0[s]
    velocity[w] = v0[s]
    touchdown[w] = wp.float64(-1.0)
    status[w] = 0
    recorded[w] = 0
    contact_steps[w] = 0
    sq_q[w] = Vec6(wp.float64(0.0))
    sq_f[w] = wp.vec2d(wp.float64(0.0), wp.float64(0.0))
    peak[w] = wp.vec2d(wp.float64(-1.0e30), wp.float64(-1.0e30))
    max_fraction[w] = wp.float64(0.0)
    fraction[w] = wp.float32(0.0)


@wp.kernel
def _stage(
    params: wp.array[ChainParams],
    cfg: Settings,
    status: wp.array[int],
    state: wp.array[Vec6],
    velocity: wp.array[Vec6],
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
):
    w = wp.tid()
    if status[w] != 0:
        return
    q = state[w]
    v = velocity[w]
    position, jx, jz, angle = _ankle(q, params[w % cfg.stance_count])
    half = (angle - cfg.pitch) / wp.float64(2.0)
    body_q[w] = wp.transform(
        wp.vec3(wp.float32(position[0]), 0.0, wp.float32(position[1])),
        wp.quat(0.0, wp.float32(-wp.sin(half)), 0.0, wp.float32(wp.cos(half))),
    )
    omega = v[2] + v[3] + v[4] + v[5]
    body_qd[w] = wp.spatial_vector(
        wp.float32(wp.dot(jx, v)), 0.0, wp.float32(wp.dot(jz, v)), 0.0, wp.float32(-omega), 0.0
    )


@wp.kernel
def _compression(
    column_count: int,
    driven: wp.array[wp.int32],
    rest: wp.array[wp.float32],
    compression: wp.array[wp.float32],
    status: wp.array[int],
    fraction: wp.array[wp.float32],
):
    i = wp.tid()
    w = i // column_count
    column = i - w * column_count
    if status[w] != 0 or driven[column] == 0:
        return
    wp.atomic_max(fraction, w, compression[i] / rest[column])


@wp.kernel
def _advance(
    params: wp.array[ChainParams],
    cfg: Settings,
    clock: wp.array[int],
    steps: wp.array[int],
    dts: wp.array[wp.float64],
    timing: wp.array[wp.vec3d],
    ref_q: wp.array2d[Vec6],
    ref_v: wp.array2d[Vec6],
    ref_ff: wp.array2d[Vec6],
    ref_grf: wp.array2d[wp.vec2d],
    knots: wp.array[wp.float64],
    stiffness: wp.array2d[Vec6],
    damping: wp.array2d[Vec6],
    body_f: wp.array[wp.spatial_vector],
    fraction: wp.array[wp.float32],
    state: wp.array[Vec6],
    velocity: wp.array[Vec6],
    touchdown: wp.array[wp.float64],
    status: wp.array[int],
    recorded: wp.array[int],
    contact_steps: wp.array[int],
    sq_q: wp.array[Vec6],
    sq_f: wp.array[wp.vec2d],
    peak: wp.array[wp.vec2d],
    max_fraction: wp.array[wp.float64],
    enabled: wp.array[int],
):
    w = wp.tid()
    if status[w] != 0:
        return
    s = w % cfg.stance_count
    c = w // cfg.stance_count
    k = clock[0]
    n = steps[s]
    if k >= n:
        status[w] = 1
        enabled[w] = 0
        return
    p = params[s]
    q = state[w]
    v = velocity[w]
    dt = dts[s]
    t = wp.float64(k) * dt
    phase = t
    if cfg.touchdown_phase != 0 and touchdown[w] >= wp.float64(0.0):
        phase = t - touchdown[w] + timing[s][0]
    position = phase / dt
    q_ref = _sample(ref_q, s, n, position)
    v_ref = _sample(ref_v, s, n, position)
    feedforward = _sample(ref_ff, s, n, position)
    phi = _gait_phase(phase, timing[s])
    gain_k = _gains(knots, stiffness, c, phi)
    gain_d = _gains(knots, damping, c, phi)
    load = feedforward + wp.cw_mul(gain_k, q_ref - q) + wp.cw_mul(gain_d, v_ref - v)
    f = body_f[w]
    fx = wp.float64(f[0])
    fz = wp.float64(f[2])
    moment = -wp.float64(f[4])
    compressed = wp.float64(fraction[w])
    fraction[w] = wp.float32(0.0)

    error = q - ref_q[k, s]
    sq_q[w] = sq_q[w] + wp.cw_mul(error, error)
    force_error = wp.vec2d(fx, fz) - ref_grf[k, s]
    sq_f[w] = sq_f[w] + wp.cw_mul(force_error, force_error)
    top = peak[w]
    peak[w] = wp.vec2d(wp.max(top[0], fx), wp.max(top[1], fz))
    max_fraction[w] = wp.max(max_fraction[w], compressed)
    recorded[w] = k + 1
    if fz > cfg.threshold:
        contact_steps[w] = contact_steps[w] + 1

    code = int(0)
    if not _finite6(load) or not (wp.isfinite(fx) and wp.isfinite(fz) and wp.isfinite(moment)):
        code = code | 2
    if q[1] < cfg.hip_floor:
        code = code | 4
    speed = cfg.max_speed
    if wp.length(wp.vec2d(v[0], v[1])) > speed:
        code = code | 8
    for j in range(2, 6):
        if wp.abs(v[j]) > speed:
            code = code | 8
    if wp.length(wp.vec2d(fx, fz)) > cfg.max_force:
        code = code | 16
    if compressed > cfg.compression_limit + wp.float64(1.0e-6):
        code = code | 32
    if fz < wp.float64(-1.0e-6):
        code = code | 64
    if code != 0:
        status[w] = code
        enabled[w] = 0
        return
    if touchdown[w] < wp.float64(0.0) and fz > cfg.threshold:
        touchdown[w] = t

    mass, bias = _dynamics(q, v, p, cfg.gravity)
    _ankle_position, jx, jz, _foot_angle = _ankle(q, p)
    generalized = load + fx * jx + fz * jz + moment * _angular(3) - bias
    acceleration, ok = _cholesky_solve(mass, generalized)
    if not ok:
        status[w] = 128
        enabled[w] = 0
        return
    v = v + dt * acceleration
    q = q + dt * v
    if not _finite6(q) or not _finite6(v):
        status[w] = 256
        enabled[w] = 0
        return
    state[w] = q
    velocity[w] = v


@wp.kernel
def _tick(clock: wp.array[int]):
    clock[0] = clock[0] + 1


def _chain_params(chain: Chain) -> ChainParams:
    p = ChainParams()
    p.lengths = wp.vec2d(*chain.lengths_m)
    p.endpoint = wp.vec2d(*chain.endpoint_local_m)
    p.masses = wp.vec4d(*chain.masses_kg)
    p.inertias = wp.vec4d(*chain.inertias_kg_m2)
    p.com_x = wp.vec4d(*chain.com_local_m[:, 0])
    p.com_z = wp.vec4d(*chain.com_local_m[:, 1])
    return p


def reference_timing(plan: Plan, threshold_n: float) -> np.ndarray:
    """Return reference touchdown, toe-off, and duration on the plan clock [s], shape (3,)."""
    contact = np.flatnonzero(plan.grf_n[:, 1] > threshold_n)
    return np.array([plan.touchdown_s, float(plan.time_s[contact[-1]]), plan.duration_s])


def gait_phase(phase_s, timing) -> np.ndarray:
    """Map plan-clock phases [s] to the normalized gait phase; see the module docstring."""
    touchdown, toeoff, duration = timing
    phase = np.asarray(phase_s, dtype=float)
    return np.interp(phase, [0.0, touchdown, toeoff, duration], [0.0, 1.0, 2.0, 3.0])


def schedule_impedance(knots_phase, stiffness, damping, timing) -> Impedance:
    """Convert a normalized-phase schedule to a plan-clock :class:`.control.Impedance` for one stance.

    Piecewise-linear gains in the normalized phase stay piecewise linear on the
    plan clock when the knots include the phase breakpoints 1 and 2.
    """
    touchdown, toeoff, duration = timing
    knot_s = np.interp(np.asarray(knots_phase, dtype=float), [0.0, 1.0, 2.0, 3.0], [0.0, touchdown, toeoff, duration])
    return Impedance(knot_s, stiffness, damping)


class Batch:
    """Evaluate gain candidates over a fixed set of stances, one CUDA world per pair.

    Args:
        plans: Reference plans, one per stance.
        chains: Pelvis-leg mechanics, one per stance.
        shoe_artifact: Digital shoe artifact path.
        mount_m: Shoe mount offset [m], shape (3,).
        static_pitch_rad: Static shoe pitch [rad].
        knots_phase: Strictly increasing normalized gait phases of the gain knots.
        candidates: Number of gain schedules evaluated together.
        config: Phase policy and failure screens shared with :func:`.rollout.simulate`.
        device: CUDA device.
        chunk_steps: Steps per captured CUDA graph.
        friction_model: Shoe friction model.
    """

    def __init__(
        self,
        plans: list[Plan],
        chains: list[Chain],
        shoe_artifact,
        mount_m,
        static_pitch_rad: float,
        knots_phase,
        *,
        candidates: int,
        config: Config | None = None,
        device: str = "cuda:0",
        chunk_steps: int = 64,
        friction_model: str = "elastic_coulomb",
    ):
        started = perf_counter()
        cfg = config or Config()
        if len(plans) != len(chains) or not plans:
            raise ValueError("plans and chains must be nonempty and of equal length")
        if not cfg.residual_feedforward:
            raise ValueError("Batch rollouts keep the residual pelvis feedforward")
        self.device = wp.get_device(device)
        self.config = cfg
        self.stance_count = len(plans)
        self.candidates = int(candidates)
        self.world_count = self.stance_count * self.candidates
        self.knots_phase = np.asarray(knots_phase, dtype=float)
        if self.knots_phase.ndim != 1 or np.any(np.diff(self.knots_phase) <= 0):
            raise ValueError("knots_phase must be strictly increasing")
        self.chunk_steps = int(chunk_steps)
        self.steps = np.array([len(plan.time_s) - 1 for plan in plans], dtype=np.int32)
        self.max_steps = int(self.steps.max())
        self.dt = float(plans[0].dt_s)
        self.timing = np.array([reference_timing(plan, cfg.contact_threshold_n) for plan in plans])

        rows = self.max_steps + 1
        padded = {name: np.zeros((rows, self.stance_count, 6)) for name in ("q", "v", "feedforward")}
        grf = np.zeros((rows, self.stance_count, 2))
        for s, plan in enumerate(plans):
            n = len(plan.time_s)
            for name, table in padded.items():
                values = getattr(plan, name)
                table[:n, s] = values
                table[n:, s] = values[-1]
            grf[:n, s] = plan.grf_n
        d = self.device
        self.ref_q = wp.array(padded["q"], dtype=Vec6, device=d)
        self.ref_v = wp.array(padded["v"], dtype=Vec6, device=d)
        self.ref_ff = wp.array(padded["feedforward"], dtype=Vec6, device=d)
        self.ref_grf = wp.array(grf, dtype=wp.vec2d, device=d)
        self.q0 = wp.array(np.array([plan.q[0] for plan in plans]), dtype=Vec6, device=d)
        self.v0 = wp.array(np.array([plan.v[0] for plan in plans]), dtype=Vec6, device=d)
        self.steps_d = wp.array(self.steps, dtype=int, device=d)
        self.dts = wp.array(np.array([plan.dt_s for plan in plans]), dtype=wp.float64, device=d)
        self.timing_d = wp.array(self.timing, dtype=wp.vec3d, device=d)
        self.params = wp.array([_chain_params(chain) for chain in chains], dtype=ChainParams, device=d)
        self.knots = wp.array(self.knots_phase, dtype=wp.float64, device=d)
        self.stiffness = wp.zeros((self.candidates, len(self.knots_phase)), dtype=Vec6, device=d)
        self.damping = wp.zeros_like(self.stiffness)

        settings = Settings()
        settings.gravity = cfg.gravity_m_s2
        settings.threshold = cfg.contact_threshold_n
        settings.compression_limit = cfg.compression_limit
        settings.max_force = cfg.maximum_force_n
        settings.hip_floor = cfg.minimum_hip_height_m
        settings.max_speed = cfg.maximum_speed
        settings.pitch = float(static_pitch_rad)
        settings.touchdown_phase = int(cfg.phase == "touchdown")
        settings.stance_count = self.stance_count
        self.settings = settings

        w = self.world_count
        self.shoe = Shoe(shoe_artifact, mount_m, static_pitch_rad, device=str(d), friction_model=friction_model)
        bed = self.shoe.shoe.column_bed
        self.carriers = SimpleNamespace(
            body_q=wp.zeros(w, dtype=wp.transform, device=d),
            body_qd=wp.zeros(w, dtype=wp.spatial_vector, device=d),
            body_f=wp.zeros(w, dtype=wp.spatial_vector, device=d),
        )
        self.foundation = FoundationFused(
            self.shoe.anchor_local_m,
            np.zeros(len(bed.rest_length_m)),
            bed.rest_length_m,
            bed.area_m2,
            bed.neighbors,
            bed.spacing_m,
            self.shoe.shoe.material,
            np.arange(w),
            wp.zeros(w, dtype=wp.vec3, device=d),
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
            d,
            self.shoe.foundation.surround,
            world_count=w,
        )

        self.state = wp.zeros(w, dtype=Vec6, device=d)
        self.velocity = wp.zeros_like(self.state)
        self.touchdown = wp.zeros(w, dtype=wp.float64, device=d)
        self.status = wp.zeros(w, dtype=int, device=d)
        self.recorded = wp.zeros(w, dtype=int, device=d)
        self.contact_steps = wp.zeros(w, dtype=int, device=d)
        self.sq_q = wp.zeros(w, dtype=Vec6, device=d)
        self.sq_f = wp.zeros(w, dtype=wp.vec2d, device=d)
        self.peak = wp.zeros(w, dtype=wp.vec2d, device=d)
        self.max_fraction = wp.zeros(w, dtype=wp.float64, device=d)
        self.fraction = wp.zeros(w, dtype=wp.float32, device=d)
        self.clock = wp.zeros(1, dtype=int, device=d)
        self.graph = None
        self.setup_wall_s = perf_counter() - started
        self.last_wall_s = None

    def _reset(self):
        self.foundation.reset()
        self.foundation.enabled.fill_(1)
        self.clock.zero_()
        wp.launch(
            _reset,
            dim=self.world_count,
            inputs=[
                self.q0,
                self.v0,
                self.stance_count,
                self.state,
                self.velocity,
                self.touchdown,
                self.status,
                self.recorded,
                self.contact_steps,
                self.sq_q,
                self.sq_f,
                self.peak,
                self.max_fraction,
                self.fraction,
            ],
            device=self.device,
        )

    def _step(self):
        d = self.device
        wp.launch(
            _stage,
            dim=self.world_count,
            inputs=[
                self.params,
                self.settings,
                self.status,
                self.state,
                self.velocity,
                self.carriers.body_q,
                self.carriers.body_qd,
            ],
            device=d,
        )
        self.foundation.apply(self.carriers, self.dt, clear_body_force=True)
        wp.launch(
            _compression,
            dim=self.world_count * self.foundation.column_count,
            inputs=[
                self.foundation.column_count,
                self.foundation.driven,
                self.foundation.rest_len,
                self.foundation.compression,
                self.status,
                self.fraction,
            ],
            device=d,
        )
        wp.launch(
            _advance,
            dim=self.world_count,
            inputs=[
                self.params,
                self.settings,
                self.clock,
                self.steps_d,
                self.dts,
                self.timing_d,
                self.ref_q,
                self.ref_v,
                self.ref_ff,
                self.ref_grf,
                self.knots,
                self.stiffness,
                self.damping,
                self.carriers.body_f,
                self.fraction,
                self.state,
                self.velocity,
                self.touchdown,
                self.status,
                self.recorded,
                self.contact_steps,
                self.sq_q,
                self.sq_f,
                self.peak,
                self.max_fraction,
                self.foundation.enabled,
            ],
            device=d,
        )
        wp.launch(_tick, dim=1, inputs=[self.clock], device=d)

    def evaluate(self, stiffness, damping) -> dict:
        """Roll out every stance under every candidate schedule.

        Args:
            stiffness: Nonnegative stiffness at each knot, shape [candidates, knots, 6].
            damping: Nonnegative damping at each knot, shape [candidates, knots, 6].

        Returns:
            Per-world summaries, each with leading shape [candidates, stances]:
            ``status`` (1 completed, otherwise a bitmask of :data:`FAILURE_REASONS`),
            ``recorded`` steps, ``tracking_rmse`` [m or rad] (6), ``grf_rmse_n`` (2),
            ``peak_grf_n`` (2), ``contact_duration_s``, ``maximum_compression_fraction``,
            and ``terminal_state``/``terminal_velocity`` (6).
        """
        shape = (self.candidates, len(self.knots_phase), 6)
        stiffness = np.asarray(stiffness, dtype=np.float64)
        damping = np.asarray(damping, dtype=np.float64)
        if stiffness.shape != shape or damping.shape != shape:
            raise ValueError(f"stiffness and damping must have shape {shape}")
        if not (np.isfinite(stiffness).all() and np.isfinite(damping).all()):
            raise ValueError("gains must be finite")
        if np.any(stiffness < 0) or np.any(damping < 0):
            raise ValueError("gains must be nonnegative")
        started = perf_counter()
        self.stiffness.assign(stiffness)
        self.damping.assign(damping)
        self._reset()
        if self.graph is None:
            # Compile and settle host-side caches outside the capture.
            self._step()
            wp.synchronize_device(self.device)
            self._reset()
            with wp.ScopedCapture(device=self.device) as capture:
                for _ in range(self.chunk_steps):
                    self._step()
            self.graph = capture.graph
        for _ in range(math.ceil((self.max_steps + 1) / self.chunk_steps)):
            wp.capture_launch(self.graph)
        status = self.status.numpy()
        recorded = self.recorded.numpy()
        count = np.maximum(recorded, 1)[:, None]
        grid = (self.candidates, self.stance_count)
        result = {
            "status": status,
            "recorded": recorded,
            "tracking_rmse": np.sqrt(self.sq_q.numpy() / count),
            "grf_rmse_n": np.sqrt(self.sq_f.numpy() / count),
            "peak_grf_n": self.peak.numpy(),
            "contact_duration_s": self.contact_steps.numpy() * self.dt,
            "maximum_compression_fraction": self.max_fraction.numpy(),
            "terminal_state": self.state.numpy(),
            "terminal_velocity": self.velocity.numpy(),
        }
        self.last_wall_s = perf_counter() - started
        return {name: value.reshape(*grid, *value.shape[1:]) for name, value in result.items()}
