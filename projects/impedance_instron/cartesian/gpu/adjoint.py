# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Experimental coupled-rollout backpropagation using shared physics.

Short-window checks pass; full-stance finite-difference validation is incomplete.
This module is a diagnostic backend, not a qualified controller optimizer.
"""

from __future__ import annotations

from time import perf_counter

import numpy as np
import warp as wp

from .adjoint_contact import ContactTape
from .adjoint_objective import ObjectiveAdjoint
from .engine import Vec6, _advance_world, _compression_partial, _prepare_world, _Settings
from .mechanics import Params, Vec5, ankle, foot_angle

wp.set_module_options({"enable_backward": True, "fuse_fp": False})


@wp.kernel
def _stage_carrier(
    p: Params,
    cfg: _Settings,
    q: wp.array2d[Vec5],
    v: wp.array2d[Vec5],
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
):
    """Expose the reference carrier kinematics through a plain differentiable kernel."""
    w = wp.tid()
    state = q[0, w]
    velocity = v[0, w]
    position, jx, jz = ankle(state, p)
    angle = (foot_angle(state) - cfg.pitch) / wp.float64(2.0)
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
def _prepare_taped(
    s: int,
    p: Params,
    cfg: _Settings,
    basis_values: wp.array2d[wp.float64],
    hip_gate: wp.array[wp.float64],
    coefficients: wp.array3d[wp.float64],
    states: wp.array2d[Vec5],
    velocities: wp.array2d[Vec5],
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
    """Call the reference step using a separate history row and immutable time."""
    _prepare_world(
        wp.tid(),
        s,
        0,
        p,
        cfg,
        basis_values,
        hip_gate,
        coefficients,
        states,
        velocities,
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
def _advance_taped(
    s: int,
    p: Params,
    cfg: _Settings,
    states: wp.array2d[Vec5],
    velocities: wp.array2d[Vec5],
    next_states: wp.array2d[Vec5],
    next_velocities: wp.array2d[Vec5],
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
    cap_count: wp.array2d[int],
    integrated: wp.array[int],
    recorded: wp.array[int],
    failure: wp.array[int],
    failure_step: wp.array[int],
):
    """Call the reference step using a separate history row and immutable time."""
    _advance_world(
        wp.tid(),
        s,
        0,
        0,
        p,
        cfg,
        states,
        velocities,
        next_states,
        next_velocities,
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
        cap_count,
        integrated,
        recorded,
        failure,
        failure_step,
    )


@wp.kernel
def _force_seed(force: wp.array2d[wp.vec2d], scale: wp.float64, loss: wp.array[wp.float64]):
    """Accumulate a diagnostic force scalar, never a replacement measured fit score."""
    w = wp.tid()
    f = force[0, w]
    wp.atomic_add(loss, 0, scale * (wp.float64(0.002) * f[0] + wp.float64(0.001) * f[1]))


@wp.kernel
def _measure_force(body_f: wp.array[wp.spatial_vector], force: wp.array2d[wp.vec2d]):
    """Retain a differentiable force read from the same shoe wrench used by integration."""
    w = wp.tid()
    wrench = body_f[w]
    force[0, w] = wp.vec2d(wp.float64(wrench[0]), wp.float64(wrench[2]))


@wp.kernel
def _terminal_seed(q: wp.array2d[Vec5], v: wp.array2d[Vec5], loss: wp.array[wp.float64]):
    """Seed position and velocity to audit the complete state-transition derivative."""
    w = wp.tid()
    q_seed = Vec5(wp.float64(0.3), wp.float64(-0.2), wp.float64(0.1), wp.float64(-0.4), wp.float64(0.2))
    v_seed = Vec5(wp.float64(0.2), wp.float64(-0.3), wp.float64(0.4), wp.float64(0.1), wp.float64(-0.2))
    wp.atomic_add(loss, 0, wp.dot(q[0, w], q_seed) + wp.dot(v[0, w], v_seed))


class EngineAdjoint:
    """Retain a coupled leg/shoe window and its complete discrete reverse path.

    Diagnostic windows hold their full input checkpoint fixed and do not
    differentiate an omitted prefix. The measured objective is available only
    for one complete stance from step zero. This engine is not an optimizer
    and never assigns a partial trajectory a measured fit score.
    """

    class Step:
        """Retain disjoint views of carrier staging, loads, and diagnostics."""

        def __init__(self, storage, index):
            for name, array in storage.items():
                setattr(self, name, array[index])

    def __init__(self, source, steps: int, *, start_step: int = 0, objective: str = "diagnostic"):
        if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
            raise ValueError("steps must be a positive integer")
        if isinstance(start_step, bool) or not isinstance(start_step, int) or start_step < 0:
            raise ValueError("start_step must be a nonnegative integer")
        if start_step + steps > source.steps:
            raise ValueError("The window must remain within the original stance")
        if source.settings.control_count != 12:
            raise ValueError("The adjoint controller requires twelve control points")
        if objective not in (
            "diagnostic",
            "diagnostic_force",
            "diagnostic_terminal",
            "measured",
            "measured_motion",
            "measured_force",
        ):
            raise ValueError("Unknown audit objective")
        if objective.startswith("measured") and (start_step != 0 or steps != source.steps or source.world_count != 1):
            raise ValueError("Measured gradients require one complete stance from its fixed initial state")
        self.objective_mode = objective
        if int(source.clock.numpy()[0]) != start_step:
            raise ValueError("The source must be parked at the exact input checkpoint")
        self.source = source
        self.device = source.device
        self.steps = steps
        self.start_step = start_step
        w = source.world_count
        self.coefficients = wp.clone(source.coefficients, requires_grad=True)
        self._q_storage = wp.zeros((steps + 1, 1, w), dtype=Vec5, device=self.device, requires_grad=True)
        self._v_storage = wp.zeros_like(self._q_storage, requires_grad=True)
        self.q = [self._q_storage[t] for t in range(steps + 1)]
        self.v = [self._v_storage[t] for t in range(steps + 1)]
        wp.copy(self.q[0], source.states[start_step : start_step + 1])
        wp.copy(self.v[0], source.velocities[start_step : start_step + 1])
        self.contact = ContactTape(source.foundation, steps, source.dt)
        self.contact.states[0].copy_from(source.foundation)
        groups = source.reduction_groups
        field_types = {
            "body_q": ((w,), wp.transform, True),
            "body_qd": ((w,), wp.spatial_vector, True),
            "carrier_q": ((w,), wp.transform, True),
            "carrier_qd": ((w,), wp.spatial_vector, True),
            "equilibrium": ((1, w), Vec6, True),
            "actuator": ((1, w), wp.vec4d, True),
            "ankle_force": ((1, w), wp.vec2d, True),
            "forces": ((1, w), wp.vec2d, True),
            "moments": ((1, w), wp.float64, False),
            "fractions": ((1, w), wp.vec3d, False),
            "caps": ((1, w), int, False),
            "partial_maxima": ((w, groups), wp.vec2d, False),
            "partial_caps": ((w, groups), int, False),
            "partial_nonfinite": ((w, groups), int, False),
        }
        self._trace_storage = {
            name: wp.zeros((steps + 1, *shape), dtype=dtype, device=self.device, requires_grad=grad)
            for name, (shape, dtype, grad) in field_types.items()
        }
        self.trace = [self.Step(self._trace_storage, t) for t in range(steps + 1)]
        self._measured_force_storage = wp.zeros((steps, 1, w), dtype=wp.vec2d, device=self.device, requires_grad=True)
        self.measured_force = [self._measured_force_storage[t] for t in range(steps)]
        self._initial_diagnostics = {}
        for name in ("failure", "failure_step", "range_step", "range_mask", "integrated", "recorded"):
            self._initial_diagnostics[name] = wp.clone(getattr(source, name))
            setattr(self, name, wp.clone(getattr(source, name)))
        self.loss = wp.zeros(1, dtype=wp.float64, device=self.device, requires_grad=True)
        self.tape = None
        self.forward_graph = None
        self.backward_graph = None
        self.objective = None
        self.state_history = None
        self.force_history = None
        if objective.startswith("measured"):
            self.objective = ObjectiveAdjoint(source.objective)
            self.state_history = self._q_storage.reshape((steps + 1, w))
            self.force_history = self._measured_force_storage.reshape((steps, w))

    def zero_grad(self):
        """Clear each owning gradient buffer once while preserving all primal histories."""
        arrays = [
            self.coefficients,
            self.loss,
            self._q_storage,
            self._v_storage,
            self._measured_force_storage,
            *self._trace_storage.values(),
        ]
        if self.objective is not None:
            arrays.extend((self.objective.residual, self.objective.costs))
        for array in arrays:
            if array.grad is not None:
                array.grad.zero_()
        self.contact.zero_grad()

    def reset_diagnostics(self):
        """Restore counters outside Tape without overwriting the fixed input checkpoint."""
        for name, initial in self._initial_diagnostics.items():
            wp.copy(getattr(self, name), initial)
        self.loss.zero_()

    def _prepare(self, t):
        s = self.source
        out = self.trace[t]
        wp.launch(
            _prepare_taped,
            dim=s.world_count,
            inputs=[
                self.start_step + t,
                s.params,
                s.kernel_config,
                s.basis,
                s.hip_gate,
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

    def forward(self):
        """Record an unmodified contact/integration window into non-aliased buffers."""
        s = self.source
        for t in range(self.steps):
            self._prepare(t)
            out = self.trace[t]
            wp.launch(
                _stage_carrier,
                dim=s.world_count,
                inputs=[s.params, s.kernel_config, self.q[t], self.v[t], out.carrier_q, out.carrier_qd],
                device=self.device,
                block_dim=1,
            )
            contact = self.contact.apply(t, out.carrier_q, out.carrier_qd)
            # These extrema only select failure screens and diagnostics. Successful
            # trajectories never differentiate the pass/fail decision.
            wp.launch(
                _compression_partial,
                dim=(s.world_count, s.reduction_groups, 32),
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
                _advance_taped,
                dim=s.world_count,
                inputs=[
                    self.start_step + t,
                    s.params,
                    s.kernel_config,
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
                dim=s.world_count,
                inputs=[contact.body_f, self.measured_force[t]],
                device=self.device,
                block_dim=1,
            )
            if self.objective_mode in ("diagnostic", "diagnostic_force"):
                wp.launch(
                    _force_seed,
                    dim=s.world_count,
                    inputs=[self.measured_force[t], 1.0 / self.steps, self.loss],
                    device=self.device,
                )
        self._prepare(self.steps)
        if self.objective_mode in ("diagnostic", "diagnostic_terminal"):
            wp.launch(_terminal_seed, dim=s.world_count, inputs=[self.q[-1], self.v[-1], self.loss], device=self.device)
        elif self.objective_mode.startswith("measured"):
            self.objective.launch(self.state_history, self.force_history, self.loss, mode=self.objective_mode)

    def complete(self):
        """Require every world to finish the requested window without a failed screen."""
        return bool(
            np.all(self.failure.numpy() == 0) and np.all(self.integrated.numpy() == self.start_step + self.steps)
        )

    def capture(self):
        """Capture validated forward and backward work outside measured update timing.

        Coefficient values may change between launches. Geometry, configuration,
        material, checkpoint, and buffer addresses must remain fixed.
        """
        if not self.device.is_cuda:
            raise ValueError("CUDA graph capture requires a CUDA device")
        if self.forward_graph is not None:
            return
        if self.tape is None:
            result = self.value_and_grad()
            if not result["valid"]:
                raise RuntimeError("A complete finite-gradient warmup is required before capture")
        with wp.ScopedCapture(device=self.device) as forward_capture:
            self.reset_diagnostics()
            self.forward()
        with wp.ScopedCapture(device=self.device) as backward_capture:
            self.zero_grad()
            self.tape.backward(self.loss)
        self.forward_graph = forward_capture.graph
        self.backward_graph = backward_capture.graph

    def value_and_grad(self):
        """Return the selected scalar and coefficient gradient only for a complete rollout."""
        started = perf_counter()
        if self.forward_graph is not None:
            wp.capture_launch(self.forward_graph)
        else:
            if self.tape is not None:
                self.zero_grad()
            self.reset_diagnostics()
            with wp.Tape() as self.tape:
                self.forward()
        if not self.complete():
            return {"valid": False, "loss": None, "gradient": None, "reason": "incomplete_window"}
        value = float(self.loss.numpy()[0])
        forward_wall_s = perf_counter() - started
        if not np.isfinite(value):
            return {"valid": False, "loss": None, "gradient": None, "reason": "nonfinite_diagnostic"}
        backward_started = perf_counter()
        if self.backward_graph is not None:
            wp.capture_launch(self.backward_graph)
        else:
            self.tape.backward(self.loss)
        if not self.complete():
            raise RuntimeError("Backward replay modified the retained forward counters or failure state")
        gradient = self.coefficients.grad.numpy()
        if not np.isfinite(gradient).all():
            return {"valid": False, "loss": value, "gradient": None, "reason": "nonfinite_gradient"}
        return {
            "valid": True,
            "loss": value,
            "gradient": gradient,
            "reason": None,
            "timings": {
                "forward_wall_s": forward_wall_s,
                "backward_wall_s": perf_counter() - backward_started,
                "total_wall_s": perf_counter() - started,
                "captured": self.forward_graph is not None,
            },
        }
