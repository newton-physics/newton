# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Score causal CUDA rollouts without moving trajectories to the host.

Targets are immutable construction-time snapshots owned only by this module.
The physics batches receive chains, shoes, initial states, tasks and horizons,
never observations. Warm evaluations upload models and download one 16-byte
aggregate per candidate on the resident path; no CPU prediction or scoring is
performed. If the combined estimate of all chunks exceeds the budget, trials
stream through one resident chunk at a time. Streaming downloads the same small
aggregate per candidate *per chunk*, then combines trial-weighted losses and
failures on the host. Neither path copies per-step results or reads full traces.

The default 512 MiB resident-array estimate includes padded traces, targets,
reductions, and a conservative shoe allowance (4 KiB per world-column plus
1 MiB per group). It excludes existing input shoes, CPU snapshots, CUDA
context/module/graph storage and allocator-pool reservations; it is not a cap
on process VRAM. Streaming halves oversized trial chunks before any batch is
allocated; a single trial that cannot fit must reduce candidates or horizon.
Every streamed evaluation rebuilds batches, preprocesses/uploads frozen
targets, reads fixed shoe setup arrays, and recaptures graphs per chunk. This
trades setup/capture overhead for bounded live rollout storage, not bounded
allocator reservations. Resident batches/graphs are reused only when all fit.
"""

from __future__ import annotations

import math
from copy import copy
from typing import TYPE_CHECKING

import numpy as np
import warp as wp

from .batch import Vec6
from .gpu_runner import GpuBatch, _Buffers, _Model, _Output
from .runner import RolloutConfig, Runner

if TYPE_CHECKING:
    from .identify import Trial

wp.set_module_options({"enable_backward": False, "fuse_fp": False})

_MEMORY_BUDGET_BYTES = 512 * 1024**2


@wp.struct
class _Targets:
    offsets: wp.array[int]
    time: wp.array[wp.float64]
    q: wp.array[Vec6]
    left: wp.array[int]
    fraction: wp.array[wp.float64]
    force: wp.array2d[wp.vec2d]
    duration: wp.array[wp.float64]
    peak: wp.array[wp.float64]
    impulse: wp.array[wp.vec2d]
    contact: wp.array[wp.float64]


@wp.struct
class _Aggregate:
    loss: wp.float64
    failed: int


@wp.kernel
def _score_worlds(
    data: _Buffers,
    models: wp.array[_Model],
    targets: _Targets,
    trials: int,
    dt: wp.float64,
    threshold: wp.float64,
    losses: wp.array[wp.float64],
    failed: wp.array[int],
):
    w = wp.tid()
    s = w % trials
    count = data.recorded[w]
    end = wp.float64(count) * dt
    squared = Vec6(wp.float64(0.0))
    observed = int(0)
    for i in range(targets.offsets[s], targets.offsets[s + 1]):
        if targets.time[i] > end + wp.float64(1.0e-12):
            break
        # np.interp holds the last accepted state, including the 1e-12
        # observation-window tolerance. Never read a rejected/stale state.
        left = wp.min(targets.left[i], count)
        right = wp.min(left + 1, count)
        q = data.states[left, w] + targets.fraction[i] * (data.states[right, w] - data.states[left, w])
        error = q - targets.q[i]
        for j in range(6):
            squared[j] += error[j] * error[j]
        observed += 1
    loss = wp.float64(0.0)
    if observed > 0:
        for j in range(6):
            scale = wp.float64(0.05)
            dimensions = wp.float64(4.0)
            if j < 2:
                scale = wp.float64(0.02)
                dimensions = wp.float64(2.0)
            loss += squared[j] / wp.float64(observed) / (scale * scale) / dimensions
    force_error = wp.vec2d(wp.float64(0.0))
    force_sum = wp.vec2d(wp.float64(0.0))
    peak = wp.float64(0.0)
    if count > 0:
        peak = data.outputs[0, w].grf_n[1]
    contact = int(0)
    effort = wp.float64(0.0)
    caps = models[w // trials].torque_cap
    for k in range(count):
        out = data.outputs[k, w]
        difference = out.grf_n - targets.force[k, s]
        for j in range(2):
            force_error[j] += difference[j] * difference[j]
        force_sum += out.grf_n
        peak = wp.max(peak, out.grf_n[1])
        if out.grf_n[1] > threshold:
            contact += 1
        for j in range(3):
            torque = out.load[j + 3] / caps[j]
            effort += torque * torque
    if count > 0:
        loss += (force_error[0] + force_error[1]) / wp.float64(count) / wp.float64(20000.0)
        loss += wp.float64(0.01) * effort / (wp.float64(count) * wp.float64(3.0))
    peak_error = (peak - targets.peak[s]) / wp.float64(100.0)
    impulse_error = (force_sum * dt - targets.impulse[s]) / wp.float64(20.0)
    contact_error = (wp.float64(contact) * dt - targets.contact[s]) / wp.float64(0.02)
    loss += peak_error * peak_error + wp.dot(impulse_error, impulse_error) / wp.float64(2.0)
    loss += contact_error * contact_error
    is_failed = int(data.status[w] != 1)
    if is_failed != 0:
        loss += wp.float64(1000.0) * (wp.float64(2.0) - end / targets.duration[s])
    losses[w] = loss
    failed[w] = is_failed


@wp.kernel
def _accumulate_candidates(
    losses: wp.array[wp.float64],
    failed: wp.array[int],
    trials: int,
    aggregates: wp.array[_Aggregate],
):
    c = wp.tid()
    result = aggregates[c]
    for s in range(trials):
        w = c * trials + s
        result.loss += losses[w]
        result.failed += failed[w]
    aggregates[c] = result


def _target_arrays(trials, group) -> _Targets:
    """Snapshot native motion and full-horizon force targets once per group."""
    offsets, times, coordinates, lefts, fractions = [0], [], [], [], []
    force = np.zeros((group.max_steps, group.trial_count, 2))
    peaks, impulses, contacts = [], [], []
    for s, index in enumerate(group.indices):
        trial = trials[index]
        clock = np.arange(int(group.steps[s]) + 1) * group.dt
        native = trial.time_s[1:]
        left = np.clip(np.searchsorted(clock, native, side="right") - 1, 0, len(clock) - 2)
        times.append(native)
        coordinates.append(trial.q[1:])
        lefts.append(left)
        # The multiply-generated last clock can fall one ulp below duration.
        fractions.append(np.clip((native - clock[left]) / (clock[left + 1] - clock[left]), 0.0, 1.0))
        offsets.append(offsets[-1] + len(native))
        force[: len(clock) - 1, s] = np.column_stack(
            [np.interp(clock[:-1], trial.force_time_s, trial.grf_n[:, j]) for j in range(2)]
        )
        interior = (trial.force_time_s > 0) & (trial.force_time_s < trial.duration_s)
        target_clock = np.concatenate(([0.0], trial.force_time_s[interior], [trial.duration_s]))
        measured = np.column_stack([np.interp(target_clock, trial.force_time_s, trial.grf_n[:, j]) for j in range(2)])
        intervals = np.diff(target_clock)
        peaks.append(measured[:, 1].max())
        impulses.append(np.sum(0.5 * (measured[1:] + measured[:-1]) * intervals[:, None], axis=0))
        contacts.append(np.sum(intervals * (measured[:-1, 1] > group.config.contact_threshold_n)))
    targets = _Targets()
    for name, values, dtype in (
        ("offsets", offsets, int),
        ("time", np.concatenate(times), wp.float64),
        ("q", np.concatenate(coordinates), Vec6),
        ("left", np.concatenate(lefts), int),
        ("fraction", np.concatenate(fractions), wp.float64),
        ("force", force, wp.vec2d),
        ("duration", group.durations, wp.float64),
        ("peak", peaks, wp.float64),
        ("impulse", impulses, wp.vec2d),
        ("contact", contacts, wp.float64),
    ):
        setattr(targets, name, wp.array(values, dtype=dtype, device=group.device))
    return targets


def _estimate_bytes(chunks, candidates: int, config: RolloutConfig) -> int:
    """Account for exact shoe/dt grouping and trace padding before allocation."""
    total = candidates * np.dtype(_Aggregate.numpy_dtype()).itemsize
    output_bytes = np.dtype(_Output.numpy_dtype()).itemsize
    for trials in chunks:
        groups = {}
        for trial in trials:
            steps = math.ceil(trial.duration_s / config.dt_s)
            if steps > np.iinfo(np.int32).max - 64:
                raise ValueError("Trial step count exceeds the GPU clock range")
            key = (id(trial.shoe), trial.duration_s / steps)
            groups.setdefault(key, []).append((trial, steps))
        for members in groups.values():
            steps = max(steps for _, steps in members)
            worlds = candidates * len(members)
            # Two Vec6 histories, one vec2d history, and interval _Output.
            total += worlds * ((steps + 1) * 112 + steps * output_bytes)
            total += steps * len(members) * 16
            total += sum((len(trial.time_s) - 1) * 68 + 44 for trial, _ in members) + 4
            columns = len(members[0][0].shoe.shoe.column_bed.rest_length_m)
            total += worlds * (4096 * columns + 4096) + 1024**2
    return total


class GpuEvaluator:
    """Simulate and score fixed trials on CUDA, streaming when necessary.

    Args:
        trials: Nonempty offline trials. Targets are snapshotted at construction;
            streaming also copies initial states but retains chain/shoe/task
            references. Do not mutate physics while using an evaluator; construct
            a new evaluator after changing observations or physics.
        candidates: Exact positive number of Runner models per evaluation.
        config: Shared integration and numerical-screen settings.
        device: CUDA device, without a CPU fallback.
        max_trials_per_batch: Positive trial chunk limit. All batches are retained
            only if their combined conservative array estimate fits the budget;
            otherwise chunks are adaptively halved and streamed sequentially.
        memory_budget_bytes: Positive resident-array estimate cap [bytes].
            ``None`` selects 512 MiB. Not a process VRAM or allocator-pool cap.

    Attributes:
        estimated_static_bytes: Combined estimate before adaptive splitting [bytes].
        estimated_resident_bytes: Combined resident estimate, or the largest
            streamed chunk estimate [bytes].
        memory_budget_bytes: Resident-array admission budget [bytes].
        streaming: Whether evaluation reconstructs and releases one chunk at a time.

    Evaluation returns only ``mean_loss`` and integer ``failed`` in candidate
    order, matching ``identify.evaluate`` selection fields. Diagnostic trial rows
    are deliberately omitted. Reductions use float64, with sequential sums within
    a world for deterministic replay; streaming combines small per-candidate
    results on the host, weighted by trial count, not horizon. Roundoff may differ
    across chunkings or from NumPy. Streaming incurs batch setup and graph capture
    on every call. Instances are reusable but not reentrant or thread-safe.
    """

    def __init__(
        self,
        trials: list[Trial],
        *,
        candidates: int,
        config: RolloutConfig | None = None,
        device: str = "cuda:0",
        max_trials_per_batch: int = 8,
        memory_budget_bytes: int | None = None,
    ):
        trials = tuple(trials)
        if not trials:
            raise ValueError("Evaluation needs at least one trial")
        if memory_budget_bytes is None:
            memory_budget_bytes = _MEMORY_BUDGET_BYTES
        for name, value in (
            ("candidates", candidates),
            ("max_trials_per_batch", max_trials_per_batch),
            ("memory_budget_bytes", memory_budget_bytes),
        ):
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        self.candidates = int(candidates)
        self.trial_count = len(trials)
        self.config = config or RolloutConfig()
        self.device = wp.get_device(device)
        if not self.device.is_cuda:
            raise ValueError("GpuEvaluator requires a CUDA device")
        chunks = [trials[start : start + max_trials_per_batch] for start in range(0, len(trials), max_trials_per_batch)]
        self.memory_budget_bytes = int(memory_budget_bytes)
        self.estimated_static_bytes = _estimate_bytes(chunks, self.candidates, self.config)
        self.estimated_resident_bytes = self.estimated_static_bytes
        self.streaming = self.estimated_static_bytes > self.memory_budget_bytes
        self._initial_torques = np.array([trial.initial.torque_nm for trial in trials])
        self._batches = []
        self._scorers = []
        self._aggregates = None
        self._streaming_chunks = ()
        if self.streaming:
            pending, admitted = list(reversed(chunks)), []
            self.estimated_resident_bytes = 0
            while pending:
                chunk = pending.pop()
                estimate = _estimate_bytes([chunk], self.candidates, self.config)
                if estimate > self.memory_budget_bytes:
                    if len(chunk) == 1:
                        raise ValueError(
                            f"Single-trial GPU objective {chunk[0].id!r} resident-array estimate {estimate} bytes "
                            f"exceeds the {self.memory_budget_bytes}-byte budget; reduce candidates/horizon "
                            "or increase memory_budget_bytes"
                        )
                    middle = len(chunk) // 2
                    pending.extend((chunk[middle:], chunk[:middle]))
                else:
                    admitted.append(chunk)
                    self.estimated_resident_bytes = max(self.estimated_resident_bytes, estimate)
            # Freeze clocks too: duration and target interpolation must not drift
            # between construction and later chunk reconstruction.
            snapshots = []
            for chunk in admitted:
                frozen = []
                for trial in chunk:
                    snapshot = copy(trial)
                    snapshot.initial = trial.initial.copy()
                    for name in ("time_s", "q", "force_time_s", "grf_n"):
                        values = getattr(trial, name).copy()
                        values.setflags(write=False)
                        setattr(snapshot, name, values)
                    frozen.append(snapshot)
                snapshots.append(tuple(frozen))
            self._streaming_chunks = tuple(snapshots)
            return
        for chunk in chunks:
            batch = GpuBatch(
                [trial.chain for trial in chunk],
                [trial.shoe for trial in chunk],
                [trial.initial for trial in chunk],
                [trial.task for trial in chunk],
                [trial.duration_s for trial in chunk],
                candidates=self.candidates,
                config=self.config,
                device=self.device,
                max_trials_per_group=int(max_trials_per_batch),
            )
            self._batches.append(batch)
            for group in batch._groups:
                targets = _target_arrays(chunk, group)
                losses = wp.zeros(group.world_count, dtype=wp.float64, device=self.device)
                failed = wp.zeros(group.world_count, dtype=int, device=self.device)
                self._scorers.append((group, targets, losses, failed))
        self._aggregates = wp.zeros(self.candidates, dtype=_Aggregate, device=self.device)

    def _release(self):
        """Release a finished streamed chunk without waiting for cyclic GC."""
        for batch in self._batches:
            for group in batch._groups:
                group.graph = None
                # The cloned foundation and its friction adapter own each other.
                # Break that cycle, never detach the original input shoe's adapter.
                group.foundation.friction_solver = None
        self._scorers.clear()
        self._batches.clear()
        self._aggregates = None

    def _evaluate_chunk(self, chunk, models):
        """Scope all resident ownership to one admitted chunk and return tiny totals."""
        evaluator = GpuEvaluator(
            list(chunk),
            candidates=self.candidates,
            config=self.config,
            device=self.device,
            max_trials_per_batch=len(chunk),
            memory_budget_bytes=self.memory_budget_bytes,
        )
        try:
            return evaluator.evaluate(models)
        except Exception:
            # Successful aggregate download already synchronizes; only errors
            # need an explicit barrier before destroying in-flight resources.
            wp.synchronize_device(self.device)
            raise
        finally:
            evaluator._release()

    def evaluate(self, models: list[Runner]) -> list[dict]:
        """Reset, simulate, score, and download only candidate loss/failure totals."""
        if len(models) != self.candidates:
            raise ValueError(f"Expected exactly {self.candidates} Runner models")
        for model in models:
            if not isinstance(model, Runner):
                raise TypeError("models must contain Runner instances")
            if np.any(np.abs(self._initial_torques) > model.bounds.torque_max_nm):
                raise ValueError("Initial torque exceeds the model bounds")
        if self.streaming:
            losses = np.zeros(self.candidates, dtype=np.float64)
            failures = [0] * self.candidates
            for chunk in self._streaming_chunks:
                values = self._evaluate_chunk(chunk, models)
                for c, row in enumerate(values):
                    losses[c] += row["mean_loss"] * len(chunk)
                    failures[c] += row["failed"]
            return [
                {"mean_loss": float(losses[c] / self.trial_count), "failed": failures[c]}
                for c in range(self.candidates)
            ]
        self._aggregates.zero_()
        for group, targets, losses, failed in self._scorers:
            group.launch(models)
            wp.launch(
                _score_worlds,
                dim=group.world_count,
                inputs=[
                    group.data,
                    group.models,
                    targets,
                    group.trial_count,
                    group.dt,
                    self.config.contact_threshold_n,
                    losses,
                    failed,
                ],
                device=self.device,
            )
            wp.launch(
                _accumulate_candidates,
                dim=self.candidates,
                inputs=[losses, failed, group.trial_count, self._aggregates],
                device=self.device,
            )
        values = self._aggregates.numpy()
        return [{"mean_loss": float(row["loss"] / self.trial_count), "failed": int(row["failed"])} for row in values]
