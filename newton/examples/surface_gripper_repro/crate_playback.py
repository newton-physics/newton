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

"""Advancing the crates from their waiting pose to their grip pose for ``example_surface_gripper_repro``.

The crate bodies and their pallet are authored up front into ``Assets/pick_scene.usda`` (baked by
``bake_pick_scene.py``); this module owns only the runtime advancing. Newton
models are fixed after ``finalize``, so the conveyor is faked: in the idle gap before each pick cycle
the next crate is teleported from its parked waiting pose onto the shared pick pallet (its grip pose),
driven each frame from the sim clock.
"""

import numpy as np
import warp as wp


class CratePlayback:
    """Moves each crate from its parked waiting pose to its grip pose, one per disengagement event.

    Every crate starts parked at its waiting pose. Each disengagement -- the seal releasing the previous
    box -- cues the next crate: ``falling[i]`` (the release before crate ``i``'s pick) is when crate
    ``i`` should be on the pick pallet, so its time is the threshold. When the sim clock crosses that
    threshold, the crate's free-joint DOFs are teleported to its grip pose. Crate 0 fires on the panel's
    release (``falling[0]``).

    Args:
        playback: :class:`~newton.examples.surface_gripper_repro.robot_playback.RobotPlayback`; its ``falling`` edges
            and sample times set the thresholds.
        model: finalized Newton ``Model`` (for the crate free-joint DOF slices).
        crate_bodies: crate body ids, in pick order.
        grip_poses: each crate's grip pose (``wp.transform``), in pick order.
    """

    def __init__(self, playback, model, crate_bodies, grip_poses):
        self._crate_bodies = crate_bodies
        # The time of every release (disengagement) event, in order. playback.falling lists the release
        # frames: falling[0] releases the panel, falling[1] releases crate 0, falling[2] crate 1, ...
        # The nth release is the cue to move the nth crate onto the pick pallet (crate 0 on the panel's
        # release), so its time is that crate's threshold. step() pairs releases to crates in order
        # and stops when either runs out, so no assumption about their counts matching is needed.
        rec_times = playback.rec_times_wp.numpy()  # rec_times[frame] = that frame's time [s]
        self._disengage_times = []
        for i in range(len(playback.falling)):
            release_frame = playback.falling[i]  # the ith release event
            release_time = float(rec_times[release_frame])
            self._disengage_times.append(release_time)
        # each crate's grip pose as free-joint DOFs [x, y, z, qx, qy, qz, qw]
        self._grip_q = np.zeros((len(grip_poses), 7), dtype=np.float32)
        for i, t in enumerate(grip_poses):
            pos = wp.transform_get_translation(t)
            quat = wp.transform_get_rotation(t)
            self._grip_q[i] = [pos[0], pos[1], pos[2], quat[0], quat[1], quat[2], quat[3]]
        # each crate body's free-joint DOF slice (7 joint_q = pos + quat xyzw, 6 joint_qd)
        joint_child = model.joint_child.numpy()
        joint_q_start = model.joint_q_start.numpy()
        joint_qd_start = model.joint_qd_start.numpy()
        self._dof = []
        for b in crate_bodies:
            j = int(np.where(joint_child == b)[0][0])  # the crate body's (free) joint
            self._dof.append((int(joint_q_start[j]), int(joint_qd_start[j])))
        self._next = 0

    def step(self, sim_time, state):
        """Move every crate whose disengagement time has passed from its waiting pose to its grip pose.

        Returns the body id of the last crate moved this call (so the caller can retarget the seal to
        it), or ``None`` if none were due. The crate free-joint DOFs are captured by reference in the
        caller's CUDA graph, so the in-place assigns take effect on the next launch.
        """
        active = None
        # pair releases to crates in order; stop when either runs out (extra releases move no crate)
        while (
            self._next < len(self._crate_bodies)
            and self._next < len(self._disengage_times)
            and sim_time >= self._disengage_times[self._next]
        ):
            qs, qds = self._dof[self._next]
            q = state.joint_q.numpy()
            qd = state.joint_qd.numpy()
            q[qs : qs + 7] = self._grip_q[self._next]  # teleport onto the pick pallet at its grip pose
            qd[qds : qds + 6] = 0.0
            state.joint_q.assign(q)
            state.joint_qd.assign(qd)
            active = self._crate_bodies[self._next]
            self._next += 1
        return active
