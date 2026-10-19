# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
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

"""Playback of a recorded palletizer cycle (``palletizer_trajectory.csv``).

Simulator-agnostic: this module only loads the recording and steps through it. It has no Newton (or
any other framework) dependency -- callers read each :class:`Frame` and apply it to their own robot
(joint targets, gripper) via callbacks. Used by ``example_surface_gripper``.

Recording format -- a CSV table with a header row and one row per frame (~24 Hz)::

    time_s, J1_deg, J2_deg, J3_deg_rel_J2, J4_deg, J5_deg, J6_deg, gripper_close, prepare_to_engage
    0.000, 160.5714, -3.4991, -7.3408, 0.0000, -82.6592, -116.5714, 0, 0

``time_s`` is the time since the first frame [s]. ``J1_deg`` to ``J6_deg`` are the arm joint
targets [deg]. J3 is stored relative to J2 -- the real J3 command is ``J3_deg_rel_J2 + J2_deg``, a
coupling the caller applies at drive time. ``gripper_close`` is the gripper signal (1 = closed;
rising edge = close, falling edge = open). ``prepare_to_engage`` is 1 in the short lead-in before
each close.
"""

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import warp as wp

JOINT_COLUMNS = ("J1_deg", "J2_deg", "J3_deg_rel_J2", "J4_deg", "J5_deg", "J6_deg")


@dataclass
class Frame:
    """One recorded frame, as stored (no coupling or unit conversion applied)."""

    time_s: float  # time since the first frame [s]
    joints_deg: list[float]  # 6 raw joint angles [deg], J1-J6; J3 is relative to J2
    gripper_close: bool  # gripper signal: True = closed
    prepare_to_engage: bool  # True in the lead-in before each close


def load_recording(path: str | Path) -> list[Frame]:
    """Read a ``.csv`` recording into a flat list of :class:`Frame`.

    Raises:
        FileNotFoundError: if ``path`` does not exist (with a clear message rather than a deep parse
            error).
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"robot recording not found: {path}")
    frames: list[Frame] = []
    with path.open(newline="") as file:
        for row in csv.DictReader(file):
            joints_deg = []
            for column in JOINT_COLUMNS:
                joints_deg.append(float(row[column]))
            frames.append(
                Frame(
                    time_s=float(row["time_s"]),
                    joints_deg=joints_deg,
                    gripper_close=bool(int(row["gripper_close"])),
                    prepare_to_engage=bool(int(row["prepare_to_engage"])),
                )
            )
    return frames


def recorded_times(frames: list[Frame]) -> list[float]:
    """Seconds of each frame relative to the first.

    Used for time-accurate playback -- interpolating targets between frames by time rather than
    stepping one frame per simulation step.
    """
    times = []
    for frame in frames:
        times.append(frame.time_s - frames[0].time_s)
    return times


@wp.kernel
def sample_playback_kernel(
    rec_times: wp.array[float],  # [N] recorded sample times [s], monotonic
    rec_targets: wp.array2d[float],  # [N, num_dofs] coupled arm targets [rad]
    rec_engaged: wp.array[wp.bool],  # [N] suction-cup engagement command (gripper_close) per frame
    rec_preparing: wp.array[wp.bool],  # [N] preparing-to-engage signal per frame
    sim_step_count: wp.array[int],  # in: device sub-step counter (current time = sim_step_count[0] * dt)
    last_lo: wp.array[int],  # in: cached lower sample index; the forward search resumes from here
    dt: float,  # physics sub-step [s]; sim time = sim_step_count * dt
    # outputs
    joint_target_q: wp.array[float],  # [num_dofs] interpolated position targets [rad]
    engaged: wp.array[wp.bool],  # [1] engagement command sampled at the current time
    preparing: wp.array[wp.bool],  # [1] preparing-to-engage signal sampled at the current time
):
    """Interpolate the recorded joint position targets and sample the engagement command at the
    current time (one thread per DOF).

    The time is the integer sub-step count times ``dt`` (exact, no float accumulation). Since sim time
    only advances, the bracketing samples are found by a forward search resumed from the cached
    ``last_lo`` (usually 0-1 steps) rather than a fresh binary search. Engagement is a step signal, so
    its value at ``t`` is ``rec_engaged[lo]``. Clamps to the last sample past the end, so the arm holds
    the final recorded pose. ``preparing`` is sampled the same way from ``rec_preparing`` (a lead-in
    before each engagement).
    """
    dof = wp.tid()
    n = rec_times.shape[0]

    step = sim_step_count[0]
    t = float(step) * dt

    # forward search from the cached index for the largest lo with rec_times[lo] <= t
    lo = last_lo[0]
    while lo < n - 1 and rec_times[lo + 1] <= t:
        lo += 1

    if dof == 0:
        engaged[0] = rec_engaged[lo]  # step signal: value of the most recent frame at or before t
        preparing[0] = rec_preparing[lo]  # step signal, same as engaged

    if lo >= n - 1:
        joint_target_q[dof] = rec_targets[n - 1, dof]  # past the end: hold the last recorded pose
        return

    # Cubic (Catmull-Rom) interpolation through the four surrounding knots. It passes through the two
    # bracketing knots (p1, p2) with tangents estimated from their neighbors, so the target is
    # C1-continuous (no slope kink at the knots) -- unlike linear interpolation, whose kinks inject
    # jerk that the stiff drives ring on.
    frac = (t - rec_times[lo]) / (rec_times[lo + 1] - rec_times[lo])
    p0 = rec_targets[wp.max(lo - 1, 0), dof]
    p1 = rec_targets[lo, dof]
    p2 = rec_targets[lo + 1, dof]
    p3 = rec_targets[wp.min(lo + 2, n - 1), dof]
    f2 = frac * frac
    f3 = f2 * frac
    joint_target_q[dof] = 0.5 * (
        (2.0 * p1)
        + (-p0 + p2) * frac
        + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * f2
        + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * f3
    )


@wp.kernel
def advance_playback_kernel(
    rec_times: wp.array[float],  # [N] recorded sample times [s], monotonic
    dt: float,  # physics sub-step [s]; sim time = sim_step_count * dt
    sim_step_count: wp.array[int],  # out: device sub-step counter, advanced in place
    last_lo: wp.array[int],  # out: cached lower sample index, advanced in place
):
    """Advance the sub-step counter and commit the forward-search index for the next sub-step
    (single thread).
    """
    n = rec_times.shape[0]
    step = sim_step_count[0]
    t = float(step) * dt

    lo = last_lo[0]
    while lo < n - 1 and rec_times[lo + 1] <= t:
        lo += 1

    last_lo[0] = lo
    sim_step_count[0] = step + 1


def _arm_targets_rad(frame: Frame, num_arm_dofs: int) -> np.ndarray:
    """Recorded :class:`Frame` -> the ``num_arm_dofs`` arm joint position targets [rad].

    Takes J1-J6, applies the J3-relative-to-J2 coupling (real J3 = recorded J3 + J2), and converts
    degrees to radians. This is where the coupling/units are applied -- the recording stores raw.
    """
    j = list(frame.joints_deg[:num_arm_dofs])
    j[2] += j[1]  # J3 is recorded relative to J2
    return np.deg2rad(np.asarray(j, dtype=np.float32))


def _load_playback(path: str | Path, num_arm_dofs: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """Load a recording and extract the arrays the sim consumes.

    Returns ``(rec_times, rec_targets, rec_engaged, rec_preparing, rec_duration)``:
        - ``rec_times``: sample times [s], shape [N], starting at 0 (:func:`recorded_times`).
        - ``rec_targets``: coupled arm joint targets [rad], shape [N, num_arm_dofs].
        - ``rec_engaged``: suction-cup engagement command (``gripper_close``) per frame, shape [N] bool.
        - ``rec_preparing``: preparing-to-engage signal (``prepare_to_engage``) per frame, shape [N] bool.
        - ``rec_duration``: recording length [s] (``rec_times[-1]``).
    """
    frames = load_recording(path)
    rec_times = np.asarray(recorded_times(frames), dtype=np.float64)
    rec_targets = np.stack([_arm_targets_rad(f, num_arm_dofs) for f in frames]).astype(np.float64)
    rec_engaged = np.array([f.gripper_close for f in frames], dtype=bool)  # [N]
    rec_preparing = np.array([f.prepare_to_engage for f in frames], dtype=bool)  # [N]
    return rec_times, rec_targets, rec_engaged, rec_preparing, float(rec_times[-1])


def _gaussian_smooth(times: np.ndarray, values: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian-smooth ``values`` ([N, D]) over the non-uniform sample ``times`` ([N]).

    Each output sample is a Gaussian-weighted average of all samples by *time* distance
    (``w_ij = exp(-((t_i - t_j) / sigma)^2 / 2)``), so it correctly handles the non-uniform sample
    rate. ``sigma <= 0`` returns the input unchanged.
    """
    if sigma <= 0.0:
        return values
    dt = times[:, None] - times[None, :]  # [N, N] pairwise time differences [s]
    weights = np.exp(-0.5 * (dt / sigma) ** 2)  # Gaussian weights by time distance
    weights /= weights.sum(axis=1, keepdims=True)  # normalize per output sample
    return weights @ values  # [N, D] smoothed targets


class RobotPlayback:
    """Container for the recorded playback arrays the sim consumes, loaded onto the device.

    Loads the recording at ``path``, applies the J3->J2 coupling and Gaussian smoothing (width
    ``smoothing_sigma`` [s]), and holds the sample times, coupled arm drive targets, and per-frame
    suction engagement command as device arrays, plus the recording duration. ``num_arm_dofs`` is the
    number of arm joints to extract (J1-J6).
    """

    def __init__(self, path: str | Path, smoothing_sigma: float, num_arm_dofs: int):
        rec_times, rec_targets, rec_engaged, rec_preparing, self.rec_duration = _load_playback(path, num_arm_dofs)
        rec_targets = _gaussian_smooth(rec_times, rec_targets, smoothing_sigma)  # smooth the coarse waypoints
        self.rec_times_wp = wp.array(rec_times, dtype=wp.float32)  # [N] sample times [s]
        self.rec_targets_wp = wp.array(rec_targets, dtype=wp.float32)  # [N, num_arm_dofs] coupled targets [rad]
        self.rec_engaged_wp = wp.array(rec_engaged, dtype=wp.bool)  # [N] engagement command per frame
        self.rec_preparing_wp = wp.array(rec_preparing, dtype=wp.bool)  # [N] preparing-to-engage signal per frame
        # engage / disengage events: frame indices where the suction command rises (a pick) / falls (a release)
        self.rising = [i for i in range(1, len(rec_engaged)) if rec_engaged[i] and not rec_engaged[i - 1]]
        self.falling = [i for i in range(1, len(rec_engaged)) if not rec_engaged[i] and rec_engaged[i - 1]]

    def step(
        self,
        sim_step_count: wp.array[int],
        last_lo: wp.array[int],
        dt: float,
        joint_target_q: wp.array[float],
        engaged: wp.array[wp.bool],
        preparing: wp.array[wp.bool],
    ) -> None:
        """Launch :func:`sample_playback_kernel`: interpolate the arm drive targets and sample the
        engagement (and preparing-to-engage) commands at the current sub-step time, then advance the
        clock with :func:`advance_playback_kernel`.

        Args:
            sim_step_count: [1] in/out device sub-step counter (current time = ``sim_step_count * dt``).
            last_lo: [1] in/out cached lower sample index; the forward search resumes from here.
            dt: physics sub-step [s].
            joint_target_q: [num_arm_dofs] out interpolated position targets [rad].
            engaged: [1] out engagement command sampled at the current time.
            preparing: [1] out preparing-to-engage flag sampled at the current time.
        """
        wp.launch(
            sample_playback_kernel,
            dim=self.rec_targets_wp.shape[1],  # one thread per arm DOF
            inputs=[
                self.rec_times_wp,
                self.rec_targets_wp,
                self.rec_engaged_wp,
                self.rec_preparing_wp,
                sim_step_count,
                last_lo,
                float(dt),
            ],
            outputs=[joint_target_q, engaged, preparing],
        )
        wp.launch(
            advance_playback_kernel,
            dim=1,
            inputs=[self.rec_times_wp, float(dt)],
            outputs=[sim_step_count, last_lo],
        )
