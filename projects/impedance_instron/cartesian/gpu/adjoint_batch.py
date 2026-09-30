# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Retain four heterogeneous measured stances in one differentiable shoe rollout."""

from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
from time import perf_counter

import numpy as np
import warp as wp

from ..data import load as load_reference
from ..fit import FitConfig, _mapping, _weights
from ..phase import hip_contact_gate
from ..profile import load as load_profile
from ..run import Config
from ..trajectory import basis
from .adjoint import _measure_force
from .adjoint_contact import ContactTape
from .adjoint_objective import _batch_costs, _batch_loss, _batch_residuals
from .engine import Engine, Vec6, _advance_world, _compression_partial, _prepare_world, _Settings
from .mechanics import Params, Vec5, ankle, foot_angle, make_params, make_params_initial

wp.set_module_options({"enable_backward": True, "fuse_fp": False})


@wp.kernel
def _prepare_batch(
    step: int,
    valid_steps: wp.array[int],
    params: wp.array[Params],
    configs: wp.array[_Settings],
    basis0: wp.array2d[wp.float64],
    basis1: wp.array2d[wp.float64],
    basis2: wp.array2d[wp.float64],
    basis3: wp.array2d[wp.float64],
    gate0: wp.array[wp.float64],
    gate1: wp.array[wp.float64],
    gate2: wp.array[wp.float64],
    gate3: wp.array[wp.float64],
    coefficients: wp.array3d[wp.float64],
    q: wp.array2d[Vec5],
    v: wp.array2d[Vec5],
    equilibrium: wp.array2d[Vec6],
    actuator: wp.array2d[wp.vec4d],
    ankle_force: wp.array2d[wp.vec2d],
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    failure: wp.array[int],
    failure_step: wp.array[int],
    range_step: wp.array[int],
    range_mask: wp.array[int],
):
    """Call the unchanged reference controller with each world's own spline grid."""
    w = wp.tid()
    if step > valid_steps[w]:
        return
    if w == 0:
        _prepare_world(
            w,
            step,
            0,
            params[w],
            configs[w],
            basis0,
            gate0,
            coefficients,
            q,
            v,
            equilibrium,
            actuator,
            ankle_force,
            body_q,
            body_qd,
            failure,
            failure_step,
            range_step,
            range_mask,
        )
    elif w == 1:
        _prepare_world(
            w,
            step,
            0,
            params[w],
            configs[w],
            basis1,
            gate1,
            coefficients,
            q,
            v,
            equilibrium,
            actuator,
            ankle_force,
            body_q,
            body_qd,
            failure,
            failure_step,
            range_step,
            range_mask,
        )
    elif w == 2:
        _prepare_world(
            w,
            step,
            0,
            params[w],
            configs[w],
            basis2,
            gate2,
            coefficients,
            q,
            v,
            equilibrium,
            actuator,
            ankle_force,
            body_q,
            body_qd,
            failure,
            failure_step,
            range_step,
            range_mask,
        )
    else:
        _prepare_world(
            w,
            step,
            0,
            params[w],
            configs[w],
            basis3,
            gate3,
            coefficients,
            q,
            v,
            equilibrium,
            actuator,
            ankle_force,
            body_q,
            body_qd,
            failure,
            failure_step,
            range_step,
            range_mask,
        )


@wp.kernel
def _stage_batch(
    step: int,
    valid_steps: wp.array[int],
    params: wp.array[Params],
    configs: wp.array[_Settings],
    q: wp.array2d[Vec5],
    v: wp.array2d[Vec5],
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
):
    """Expose reference carrier kinematics without the controller's empty replay hook."""
    w = wp.tid()
    if step >= valid_steps[w]:
        return
    state = q[0, w]
    velocity = v[0, w]
    position, jx, jz = ankle(state, params[w])
    angle = (foot_angle(state) - configs[w].pitch) / wp.float64(2.0)
    body_q[w] = wp.transform(
        wp.vec3(wp.float32(position[0]), 0.0, wp.float32(position[1])),
        wp.quat(0.0, wp.float32(-wp.sin(angle)), 0.0, wp.float32(wp.cos(angle))),
    )
    body_qd[w] = wp.spatial_vector(
        wp.float32(wp.dot(jx, velocity)),
        0.0,
        wp.float32(wp.dot(jz, velocity)),
        0.0,
        wp.float32(-((velocity[2] + velocity[3]) + velocity[4])),
        0.0,
    )


@wp.kernel
def _advance_batch(
    step: int,
    valid_steps: wp.array[int],
    params: wp.array[Params],
    configs: wp.array[_Settings],
    q: wp.array2d[Vec5],
    v: wp.array2d[Vec5],
    next_q: wp.array2d[Vec5],
    next_v: wp.array2d[Vec5],
    actuator: wp.array2d[wp.vec4d],
    ankle_force: wp.array2d[wp.vec2d],
    body_f: wp.array[wp.spatial_vector],
    groups: int,
    partial_maxima: wp.array2d[wp.vec2d],
    partial_caps: wp.array2d[int],
    partial_nonfinite: wp.array2d[int],
    forces: wp.array2d[wp.vec2d],
    moments: wp.array2d[wp.float64],
    fractions: wp.array2d[wp.vec3d],
    caps: wp.array2d[int],
    integrated: wp.array[int],
    recorded: wp.array[int],
    failure: wp.array[int],
    failure_step: wp.array[int],
):
    """Integrate only a world's real steps and retain its terminal leg state."""
    w = wp.tid()
    if step >= valid_steps[w]:
        next_q[0, w] = q[0, w]
        next_v[0, w] = v[0, w]
        return
    _advance_world(
        w,
        step,
        0,
        0,
        params[w],
        configs[w],
        q,
        v,
        next_q,
        next_v,
        actuator,
        ankle_force,
        body_f,
        groups,
        partial_maxima,
        partial_caps,
        partial_nonfinite,
        forces,
        moments,
        fractions,
        caps,
        integrated,
        recorded,
        failure,
        failure_step,
    )


class BatchAdjoint:
    """One fixed-address four-world VJP over stance-specific measured objectives.

    The controller gradient for world ``w`` is the unaveraged derivative of
    that world's complete native-grid loss. ``mean_gradient`` averages active
    worlds and is suitable for minibatch training after all validity checks.
    """

    class Step:
        """Expose one timestep's disjoint carrier, load, and diagnostic views."""

        def __init__(self, storage, index):
            for name, array in storage.items():
                setattr(self, name, array[index])

    def __init__(
        self,
        template_fit_dir: Path,
        dataset_root: Path,
        *,
        capacity: int = 4,
        controller_duration_s: float | None = None,
        human_lengths_m=None,
    ):
        if capacity != 4:
            raise ValueError("The current captured batch has exactly four world slots")
        template_fit_dir, dataset_root = Path(template_fit_dir), Path(dataset_root)
        metadata = json.loads((template_fit_dir / "optimization_inputs.json").read_text())
        self.profile = load_profile(template_fit_dir / "profile.json")
        if len(self.profile["equilibrium_lower"]) != 6:
            raise ValueError("Batch fitting requires the saved six-channel profile")
        self.config = Config(**metadata["simulation_config"])
        self.settings = FitConfig(**metadata["fit_config"])
        self.controller_duration_s = None if controller_duration_s is None else float(controller_duration_s)
        if self.controller_duration_s is not None and (
            not math.isfinite(self.controller_duration_s) or self.controller_duration_s <= 0.0
        ):
            raise ValueError("controller_duration_s must be finite and positive")
        self.human_lengths_m = None if human_lengths_m is None else np.asarray(human_lengths_m, dtype=np.float64)
        if self.human_lengths_m is not None and (
            self.human_lengths_m.shape != (2,)
            or not np.isfinite(self.human_lengths_m).all()
            or np.any(self.human_lengths_m <= 0.0)
        ):
            raise ValueError("human_lengths_m must contain two finite, positive segment lengths")
        if self.controller_duration_s is not None and self.config.hip_flight_gate_enabled:
            raise ValueError("A fixed controller clock cannot use the reference-derived hip flight gate")
        shoe = metadata["shoe"]
        shoe_path = Path(shoe["path"])
        if hashlib.sha256(shoe_path.read_bytes()).hexdigest() != shoe["sha256"]:
            raise ValueError("Saved shoe asset differs from fit provenance")
        self.shoe = shoe
        manifest = json.loads((dataset_root / "manifest.json").read_text())
        members = manifest["members"]
        if len(members) != 110 or {m["trial"] for m in members} != {"FR3_1", "FR3_2"}:
            raise ValueError("Expected the prepared FR3_1/FR3_2 100+10 stance dataset")
        references = [load_reference(dataset_root / member["reference"]) for member in members]
        steps = [math.ceil(float(r["time_s"][-1]) / self.config.dt_s) for r in references]
        self.max_steps = max(steps)
        self.max_motion = max(len(r["time_s"]) for r in references)
        self.max_force = max(len(r["grf_time_s"]) for r in references)
        self.max_samples = 2 * self.max_motion + self.max_force
        self.capacity = capacity
        self.source = Engine(
            references[int(np.argmax(steps))],
            self.profile,
            shoe_path,
            shoe["mount_m"],
            shoe["static_pitch_rad"],
            config=self.config,
            settings=self.settings,
            world_count=capacity,
            friction_model=shoe["friction_model"],
        )
        self.device = self.source.device
        self.params = wp.array([self.source.params] * capacity, dtype=Params, device=self.device)
        self.configs = wp.array([self.source.kernel_config] * capacity, dtype=_Settings, device=self.device)
        self.valid_steps = wp.zeros(capacity, dtype=int, device=self.device)
        self.active = wp.zeros(capacity, dtype=int, device=self.device)
        self.coefficients = wp.zeros((capacity, 12, 6), dtype=wp.float64, device=self.device, requires_grad=True)
        self.basis = [wp.zeros((self.max_steps + 1, 12), dtype=wp.float64, device=self.device) for _ in range(capacity)]
        self.gates = [wp.zeros(self.max_steps + 1, dtype=wp.float64, device=self.device) for _ in range(capacity)]
        self._q_storage = wp.zeros(
            (self.max_steps + 1, 1, capacity), dtype=Vec5, device=self.device, requires_grad=True
        )
        self._v_storage = wp.zeros_like(self._q_storage, requires_grad=True)
        self.q = [self._q_storage[t] for t in range(self.max_steps + 1)]
        self.v = [self._v_storage[t] for t in range(self.max_steps + 1)]
        trace_types = {
            "body_q": ((capacity,), wp.transform, True),
            "body_qd": ((capacity,), wp.spatial_vector, True),
            "carrier_q": ((capacity,), wp.transform, True),
            "carrier_qd": ((capacity,), wp.spatial_vector, True),
            "equilibrium": ((1, capacity), Vec6, True),
            "actuator": ((1, capacity), wp.vec4d, True),
            "ankle_force": ((1, capacity), wp.vec2d, True),
            "forces": ((1, capacity), wp.vec2d, True),
            "moments": ((1, capacity), wp.float64, False),
            "fractions": ((1, capacity), wp.vec3d, False),
            "caps": ((1, capacity), int, False),
            "partial_maxima": ((capacity, self.source.reduction_groups), wp.vec2d, False),
            "partial_caps": ((capacity, self.source.reduction_groups), int, False),
            "partial_nonfinite": ((capacity, self.source.reduction_groups), int, False),
        }
        self._trace_storage = {
            name: wp.zeros((self.max_steps + 1, *shape), dtype=dtype, device=self.device, requires_grad=grad)
            for name, (shape, dtype, grad) in trace_types.items()
        }
        self.trace = [self.Step(self._trace_storage, t) for t in range(self.max_steps + 1)]
        self._measured_force_storage = wp.zeros(
            (self.max_steps, 1, capacity), dtype=wp.vec2d, device=self.device, requires_grad=True
        )
        self.measured_force = [self._measured_force_storage[t] for t in range(self.max_steps)]
        self.contact = ContactTape(self.source.foundation, self.max_steps, self.source.dt)
        for name in ("integrated", "recorded", "failure", "failure_step", "range_step", "range_mask"):
            setattr(self, name, wp.zeros(capacity, dtype=int, device=self.device))
        self.motion_count = wp.zeros(capacity, dtype=int, device=self.device)
        self.force_count = wp.zeros(capacity, dtype=int, device=self.device)
        self.lower = wp.zeros((capacity, self.max_samples), dtype=int, device=self.device)
        self.upper = wp.zeros_like(self.lower)
        self.fraction = wp.zeros((capacity, self.max_samples), dtype=wp.float64, device=self.device)
        self.targets = wp.zeros((capacity, self.max_samples), dtype=wp.vec2d, device=self.device)
        self.weights = wp.zeros((capacity, self.max_samples), dtype=wp.float64, device=self.device)
        self.residual_weights = wp.zeros_like(self.weights)
        self.scales = wp.array(
            [self.settings.hip_tolerance_m, self.settings.joint_tolerance_rad, self.settings.force_tolerance_n],
            dtype=wp.float64,
            device=self.device,
        )
        self.ground_angle = wp.zeros(capacity, dtype=int, device=self.device)
        self.ground_offset = wp.zeros(capacity, dtype=wp.float64, device=self.device)
        self.residual = wp.zeros(
            (2 * self.max_samples, capacity), dtype=wp.float64, device=self.device, requires_grad=True
        )
        self.costs = wp.zeros((capacity, 3), dtype=wp.float64, device=self.device, requires_grad=True)
        self.losses_device = wp.zeros(capacity, dtype=wp.float64, device=self.device, requires_grad=True)
        self.loss_seed = wp.zeros(capacity, dtype=wp.float64, device=self.device)
        self._state_history = self._q_storage.reshape((self.max_steps + 1, capacity))
        self._force_history = self._measured_force_storage.reshape((self.max_steps, capacity))
        self._active_count = 0
        self._counts = np.zeros(capacity, dtype=np.int32)
        self._motion_counts = np.zeros(capacity, dtype=np.int32)
        self._force_counts = np.zeros(capacity, dtype=np.int32)
        self.tape = None
        self.forward_graph = None
        self.backward_graph = None

    def set_batch(self, references: list[dict], coefficients: np.ndarray) -> None:
        """Upload one to four independent stances without changing graph addresses."""
        if not 1 <= len(references) <= self.capacity:
            raise ValueError("Batch needs one to four stances")
        coefficients = np.asarray(coefficients, dtype=np.float64)
        if coefficients.shape != (len(references), 12, 6) or not np.isfinite(coefficients).all():
            raise ValueError("Coefficients must be finite [active, 12, 6]")
        wmax = self.capacity
        padded_coeff = np.zeros((wmax, 12, 6), dtype=np.float64)
        padded_coeff[: len(references)] = coefficients
        self.coefficients.assign(padded_coeff)
        initial_q = np.zeros((1, wmax, 5), dtype=np.float64)
        initial_v = np.zeros_like(initial_q)
        counts = np.zeros(wmax, dtype=np.int32)
        motion_counts = np.zeros_like(counts)
        force_counts = np.zeros_like(counts)
        active = np.zeros_like(counts)
        lower = np.zeros((wmax, self.max_samples), dtype=np.int32)
        upper = np.zeros_like(lower)
        fraction = np.zeros((wmax, self.max_samples), dtype=np.float64)
        targets = np.zeros((wmax, self.max_samples, 2), dtype=np.float64)
        weights = np.zeros((wmax, self.max_samples), dtype=np.float64)
        residual_weights = np.zeros_like(weights)
        angles = np.zeros(wmax, dtype=np.int32)
        offsets = np.zeros(wmax, dtype=np.float64)
        timesteps = np.full(wmax, self.source.dt, dtype=np.float64)
        params = [self.source.params] * wmax
        configs = [self.source.kernel_config] * wmax
        for w in range(wmax):
            b = np.zeros((self.max_steps + 1, 12), dtype=np.float64)
            gate = np.zeros(self.max_steps + 1, dtype=np.float64)
            if w < len(references):
                reference = references[w]
                duration = float(reference["time_s"][-1])
                if self.controller_duration_s is not None and duration > self.controller_duration_s:
                    raise ValueError("Stance horizon exceeds the supported controller period")
                steps = math.ceil(duration / self.config.dt_s)
                if steps > self.max_steps:
                    raise ValueError("Stance exceeds the captured maximum duration")
                dt = duration / steps
                clock = np.linspace(0.0, duration, steps + 1)
                b[: steps + 1] = basis(clock, self.controller_duration_s or duration, 12)
                gate[: steps + 1] = (
                    hip_contact_gate(reference, clock, self.config.hip_flight_gate_ramp_s)
                    if self.config.hip_flight_gate_enabled
                    else 1.0
                )
                counts[w] = steps
                active[w] = 1
                timesteps[w] = dt
                initial_q[0, w] = reference["state"][0]
                initial_v[0, w] = reference["velocity"][0]
                params[w] = (
                    make_params(reference, self.profile)
                    if self.human_lengths_m is None
                    else make_params_initial(self.human_lengths_m, self.profile)
                )
                cfg = copy.deepcopy(self.source.kernel_config)
                cfg.dt = dt
                cfg.steps = steps
                configs[w] = cfg
                motion_time = np.asarray(reference["time_s"])
                force_time = np.asarray(reference["grf_time_s"])
                force_mask = (force_time >= clock[0]) & (force_time <= clock[-2])
                if not np.any(force_mask):
                    raise ValueError("Stance has no native GRF samples in simulation support")
                motion = len(motion_time)
                force = int(np.count_nonzero(force_mask))
                motion_counts[w], force_counts[w] = motion, force
                maps = (_mapping(clock, motion_time),) * 2 + (_mapping(clock[:-1], force_time[force_mask]),)
                sample_count = 2 * motion + force
                lower[w, :sample_count] = np.concatenate([item[0] for item in maps])
                upper[w, :sample_count] = np.concatenate([item[1] for item in maps])
                fraction[w, :sample_count] = np.concatenate([item[2].ravel() for item in maps])
                weight_values = np.concatenate(
                    (_weights(motion_time), _weights(motion_time), _weights(force_time[force_mask]))
                )
                weights[w, :sample_count] = weight_values
                residual_weights[w, :sample_count] = np.sqrt(weight_values / 2)
                angle_targets = np.array(reference["joint_target_rad"], copy=True)
                angles[w] = int("foot_ground_target_rad" in reference)
                offsets[w] = np.pi / 2 - float(reference.get("shoe_static_pitch_rad", 0.0))
                if angles[w]:
                    angle_targets[:, 1] = reference["foot_ground_target_rad"]
                targets[w, :sample_count] = np.concatenate(
                    (reference["hip_target_m"], angle_targets, reference["grf_target_n"][force_mask])
                )
            self.basis[w].assign(b)
            self.gates[w].assign(gate)
        self.params.assign(params)
        self.configs.assign(configs)
        self.valid_steps.assign(counts)
        self.contact.set_valid_steps(counts)
        self.contact.set_world_timesteps(timesteps)
        self.active.assign(active)
        self.loss_seed.assign(active.astype(np.float64))
        self.q[0].assign(initial_q)
        self.v[0].assign(initial_v)
        for name in self.contact.states[0].FIELDS:
            getattr(self.contact.states[0], name).zero_()
        self.motion_count.assign(motion_counts)
        self.force_count.assign(force_counts)
        self.lower.assign(lower)
        self.upper.assign(upper)
        self.fraction.assign(fraction)
        self.targets.assign(targets)
        self.weights.assign(weights)
        self.residual_weights.assign(residual_weights)
        self.ground_angle.assign(angles)
        self.ground_offset.assign(offsets)
        self._active_count = len(references)
        self._counts, self._motion_counts, self._force_counts = counts, motion_counts, force_counts

    def _reset_diagnostics(self):
        for name in ("integrated", "recorded", "failure", "range_mask"):
            getattr(self, name).zero_()
        for name in ("failure_step", "range_step"):
            getattr(self, name).fill_(-1)

    def _forward(self):
        s = self.source
        for t in range(self.max_steps + 1):
            out = self.trace[t]
            wp.launch(
                _prepare_batch,
                dim=self.capacity,
                inputs=[
                    t,
                    self.valid_steps,
                    self.params,
                    self.configs,
                    *self.basis,
                    *self.gates,
                    self.coefficients,
                    self.q[t],
                    self.v[t],
                    out.equilibrium,
                    out.actuator,
                    out.ankle_force,
                    out.body_q,
                    out.body_qd,
                    self.failure,
                    self.failure_step,
                    self.range_step,
                    self.range_mask,
                ],
                device=self.device,
                block_dim=1,
            )
            if t == self.max_steps:
                break
            wp.launch(
                _stage_batch,
                dim=self.capacity,
                inputs=[
                    t,
                    self.valid_steps,
                    self.params,
                    self.configs,
                    self.q[t],
                    self.v[t],
                    out.carrier_q,
                    out.carrier_qd,
                ],
                device=self.device,
                block_dim=1,
            )
            contact = self.contact.apply(t, out.carrier_q, out.carrier_qd)
            wp.launch(
                _compression_partial,
                dim=(self.capacity, s.reduction_groups, 32),
                inputs=[
                    s.reduction_groups,
                    s.foundation.column_count,
                    s.kernel_config.passive_cap,
                    contact.compression,
                    s.rest,
                    s.foundation.driven,
                    out.partial_maxima,
                    out.partial_caps,
                    out.partial_nonfinite,
                ],
                device=self.device,
                block_dim=32,
                record_tape=False,
            )
            wp.launch(
                _advance_batch,
                dim=self.capacity,
                inputs=[
                    t,
                    self.valid_steps,
                    self.params,
                    self.configs,
                    self.q[t],
                    self.v[t],
                    self.q[t + 1],
                    self.v[t + 1],
                    out.actuator,
                    out.ankle_force,
                    contact.body_f,
                    s.reduction_groups,
                    out.partial_maxima,
                    out.partial_caps,
                    out.partial_nonfinite,
                    out.forces,
                    out.moments,
                    out.fractions,
                    out.caps,
                    self.integrated,
                    self.recorded,
                    self.failure,
                    self.failure_step,
                ],
                device=self.device,
                block_dim=1,
            )
            wp.launch(
                _measure_force,
                dim=self.capacity,
                inputs=[contact.body_f, self.measured_force[t]],
                device=self.device,
                block_dim=1,
            )
        wp.launch(
            _batch_residuals,
            dim=(2 * self.max_samples, self.capacity),
            inputs=[
                self._state_history,
                self._force_history,
                self.motion_count,
                self.force_count,
                self.lower,
                self.upper,
                self.fraction,
                self.targets,
                self.residual_weights,
                self.scales,
                self.ground_angle,
                self.ground_offset,
                self.residual,
            ],
            device=self.device,
        )
        wp.launch(
            _batch_costs,
            dim=(self.capacity, 3),
            inputs=[self.residual, self.motion_count, self.force_count, self.costs],
            device=self.device,
        )
        wp.launch(
            _batch_loss, dim=self.capacity, inputs=[self.costs, self.active, self.losses_device], device=self.device
        )

    def _zero_grad(self):
        arrays = [
            self.coefficients,
            self._q_storage,
            self._v_storage,
            self._measured_force_storage,
            self.residual,
            self.costs,
            self.losses_device,
            *self._trace_storage.values(),
        ]
        for array in arrays:
            if array.grad is not None:
                array.grad.zero_()
        self.contact.zero_grad()

    def _results(self, *, with_gradient: bool, timings: dict) -> dict:
        active = self._active_count
        counts = self.integrated.numpy()[:active]
        failures = self.failure.numpy()[:active]
        valid = (counts == self._counts[:active]) & (failures == 0)
        losses = self.losses_device.numpy()[:active].copy()
        losses[~valid] = np.inf
        residual = self.residual.numpy()[:, :active]
        rmse = np.full((active, 6), np.inf, dtype=np.float64)
        scales = np.asarray(
            [self.settings.hip_tolerance_m, self.settings.joint_tolerance_rad, self.settings.force_tolerance_n]
        )
        for w in range(active):
            if valid[w]:
                m, f = self._motion_counts[w], self._force_counts[w]
                for block, n in enumerate((m, m, f)):
                    for c in range(2):
                        row = residual[2 * block * m + c : 2 * (block * m + n) : 2, w]
                        rmse[w, 2 * block + c] = scales[block] * math.sqrt(2.0 * float(np.sum(row * row)))
        result = {
            "valid": valid,
            "losses": losses,
            "rmse": rmse,
            "mean_loss": float(np.mean(losses)),
            "timings": timings,
            "failure": failures,
            "integrated_steps": counts,
        }
        if with_gradient:
            gradients = self.coefficients.grad.numpy()[:active].copy()
            if not np.isfinite(gradients).all():
                valid[:] = False
            result["gradients"] = gradients
            result["mean_gradient"] = np.mean(gradients, axis=0)
        return result

    def forward_only(self) -> dict:
        """Score the complete native-grid objective for the current active worlds."""
        if self._active_count == 0:
            raise ValueError("Call set_batch before simulation")
        started = perf_counter()
        if self.forward_graph is None:
            self._reset_diagnostics()
            self._forward()
        else:
            wp.capture_launch(self.forward_graph)
        result = self._results(
            with_gradient=False,
            timings={"forward_wall_s": perf_counter() - started, "captured": self.forward_graph is not None},
        )
        return result

    def value_and_grad(self) -> dict:
        """Return separate full-stance losses and unaveraged coefficient VJPs."""
        if self._active_count == 0:
            raise ValueError("Call set_batch before simulation")
        started = perf_counter()
        if self.forward_graph is None:
            if self.tape is not None:
                self._zero_grad()
            self._reset_diagnostics()
            with wp.Tape() as self.tape:
                self._forward()
        else:
            wp.capture_launch(self.forward_graph)
        forward_s = perf_counter() - started
        counts = self.integrated.numpy()[: self._active_count]
        failures = self.failure.numpy()[: self._active_count]
        if np.any(counts != self._counts[: self._active_count]) or np.any(failures != 0):
            return self._results(
                with_gradient=False, timings={"forward_wall_s": forward_s, "captured": self.forward_graph is not None}
            )
        backward_started = perf_counter()
        if self.backward_graph is None:
            self.tape.backward(grads={self.losses_device: self.loss_seed})
        else:
            wp.capture_launch(self.backward_graph)
        return self._results(
            with_gradient=True,
            timings={
                "forward_wall_s": forward_s,
                "backward_wall_s": perf_counter() - backward_started,
                "total_wall_s": perf_counter() - started,
                "captured": self.forward_graph is not None,
            },
        )

    def capture(self) -> None:
        """Capture one fixed-shape forward and backward graph for all later batches."""
        if self.forward_graph is not None:
            return
        if self.tape is None:
            result = self.value_and_grad()
            if not np.all(result["valid"]):
                raise RuntimeError("A complete batch is required before graph capture")
        with wp.ScopedCapture(device=self.device) as forward:
            self._reset_diagnostics()
            self._forward()
        with wp.ScopedCapture(device=self.device) as backward:
            self._zero_grad()
            self.tape.backward(grads={self.losses_device: self.loss_seed})
        self.forward_graph, self.backward_graph = forward.graph, backward.graph
