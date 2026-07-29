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

"""Optional debug telemetry recorders for ``example_suction_cup_isaac_sim_repro``.

Each recorder logs a quantity per sim sub-step over the first engaged (suction-on) window and writes it
to CSV at the first disengagement. Recording is host-side (reads device state each sub-step), so it only
runs on CPU -- on CUDA it is skipped by the caller to avoid breaking graph capture (see ``RECORD_DEBUG``).
"""

import csv

import warp as wp


class EndEffectorAccelerationRecorder:
    """Records the end-effector acceleration each sim step while the suction is engaged, then writes it
    to CSV at the first disengagement.

    Gated by the caller (see the record calls in ``simulate``). Recording is host-side (reads state
    each sub-step), so it only runs on CPU -- on CUDA it is skipped to avoid breaking graph capture.
    """

    def __init__(self, ee_body, sim_dt):
        self.ee_body = ee_body
        self.sim_dt = sim_dt
        self.accel_log = []  # [ang_x, ang_y, ang_z, lin_x, lin_y, lin_z], EE frame
        self.time_log = []  # matching sim time [s]
        self.done = False  # set True at the first disengagement -> record no more

    def record(self, prev_state, curr_state, engaged, sim_step_count):
        """Record one sub-step. Call after the solver step, before the state swap.

        ``prev_state`` / ``curr_state`` are the pre-/post-step states (finite-difference acceleration);
        ``engaged`` / ``sim_step_count`` are the device arrays for the engagement command and sub-step
        counter.
        """
        if self.done:
            return
        if not bool(engaged.numpy()[0]):
            if self.accel_log:  # was engaged, now disengaged -> stop recording and dump
                self.done = True
                self._dump()
            return

        prev_v = prev_state.body_qd.numpy()[self.ee_body]  # [ang, lin] world, before the step
        curr_v = curr_state.body_qd.numpy()[self.ee_body]  # after the step
        accel = (curr_v - prev_v) / self.sim_dt  # world-frame spatial acceleration
        # rotate into the end-effector frame (inverse of the current EE orientation)
        quat = wp.quat(*curr_state.body_q.numpy()[self.ee_body][3:7])  # EE orientation [x, y, z, w]
        ang = wp.quat_rotate_inv(quat, wp.vec3(*accel[0:3]))  # angular accel, EE frame [rad/s^2]
        lin = wp.quat_rotate_inv(quat, wp.vec3(*accel[3:6]))  # linear accel, EE frame [m/s^2]
        self.accel_log.append([ang[0], ang[1], ang[2], lin[0], lin[1], lin[2]])
        self.time_log.append(float(sim_step_count.numpy()[0]) * self.sim_dt)

    def _dump(self, path="ee_accelerations.csv"):
        """Write the accel log (time + 6 EE-frame components) to CSV."""
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["time_s", "ang_x", "ang_y", "ang_z", "lin_x", "lin_y", "lin_z"])
            for t, accel in zip(self.time_log, self.accel_log, strict=True):
                writer.writerow([t, *accel])
        print(f"wrote {len(self.time_log)} rows to {path}")


class DriveTargetRecorder:
    """Records the smoothed runtime arm drive targets (the interpolated ``control.joint_target_q``
    applied each sim step) while the suction is engaged, then writes them to CSV at the first
    disengagement.

    Gated by the caller (see the record calls in ``simulate``). Host-side (reads state each sub-step),
    so CPU only.
    """

    def __init__(self, sim_dt, num_arm_dofs):
        self.sim_dt = sim_dt
        self.num_arm_dofs = num_arm_dofs
        self.target_log = []  # applied arm drive targets [rad], J1..J6
        self.time_log = []  # matching sim time [s]
        self.done = False  # set True at the first disengagement -> record no more

    def record(self, engaged, joint_target_q, sim_step_count):
        """Record one sub-step. ``engaged`` / ``joint_target_q`` / ``sim_step_count`` are device arrays
        for the engagement command, the applied drive targets, and the sub-step counter.
        """
        if self.done:
            return
        if not bool(engaged.numpy()[0]):
            if self.target_log:  # was engaged, now disengaged -> stop recording and dump
                self.done = True
                self._dump()
            return
        self.target_log.append(list(joint_target_q.numpy()[: self.num_arm_dofs]))
        self.time_log.append(float(sim_step_count.numpy()[0]) * self.sim_dt)

    def _dump(self, path="drive_targets.csv"):
        """Write the drive-target log (time + J1..J6 [rad]) to CSV."""
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["time_s"] + [f"J{i + 1}" for i in range(self.num_arm_dofs)])
            for t, q in zip(self.time_log, self.target_log, strict=True):
                writer.writerow([t, *q])
        print(f"wrote {len(self.time_log)} rows to {path}")
