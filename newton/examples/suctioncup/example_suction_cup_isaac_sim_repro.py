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

###########################################################################
# Example Suction Cup Isaac Sim Repro
#
# Reproduction scene for the suction-cup gripper on a robot arm. Loads the robot arm from a USD stage
# (Assets/robot_only_newton_flattened.usda) with a fixed base on a ground plane, then plays back a
# recorded FANUC palletizer cycle (Assets/robot_recording_truncated.jsonl -- the leading idle removed
# from robot_recording.jsonl so the arm moves right away). Playback is time-accurate: the six
# arm joint position targets are interpolated from the recorded timestamps at the current simulation
# time (J3 coupled to J2, degrees -> radians) and updated before every physics sub-step, so the arm
# follows the recording at its true speed. The recording's suction-cup engagement command (ro[0]) is
# extracted per frame; the suction gripper itself is wired up and added in later steps.
#
# Command: python -m newton.examples suction_cup_isaac_sim_repro
###########################################################################

from pathlib import Path

import numpy as np
import warp as wp

import newton
import newton.examples
from newton.examples.suctioncup.robot_playback import load_recording, recorded_times

# assets live alongside this example
ASSETS = Path(__file__).parent / "Assets"
# robot USD with convex-hull collision added to the suction-gripper meshes (see add_gripper_collision.py)
ROBOT_USD = ASSETS / "robot_with_gripper_collision.usda"
# recording with the leading idle removed (truncated from robot_recording.jsonl); it starts just
# before the first joint motion, so the arm moves right away.
RECORDING_JSONL = ASSETS / "robot_recording_truncated.jsonl"

FPS = 60  # rendered frames per second
SIM_HZ = 240  # target physics rate; sim_substeps = SIM_HZ / FPS physics steps per render frame
NUM_ARM_DOFS = 6  # J1-J6; recorded joints 6-8 are unused finger DOFs
BOX_HALF = 0.5  # half-extent of the static support box (size [1, 1, 1]) [m]
PICK_BOX_HALF = (0.5, 0.5, 0.05)  # half-extents of the dynamic pick box (size [1, 1, 0.1]) [m]
PICK_BOX_DENSITY = 100.0  # density of the dynamic pick box [kg/m^3]

# Box centers at the arm's first-engagement pick pose, precomputed (centered under the end-effector,
# with the pick box's top face 1 cm below the gripper geometry; the static box's top is the pick
# box's bottom). Hard-coded here so the scene builds in one pass -- no forward-kinematics probe.
STATIC_BOX_CENTER = (-0.494, 1.589, 0.772)  # 1x1x1 support box (pallet) [m]
PICK_BOX_CENTER = (-0.494, 1.589, 1.322)  # 1x1x0.1 dynamic pick box [m]


def arm_targets_rad(frame) -> np.ndarray:
    """Recorded :class:`Frame` -> the six arm joint position targets [rad].

    Takes J1-J6, applies the J3-relative-to-J2 coupling (real J3 = recorded J3 + J2), and converts
    degrees to radians. This is where the coupling/units are applied -- the recording stores raw.
    """
    j = list(frame.joints_deg[:NUM_ARM_DOFS])
    j[2] += j[1]  # J3 is recorded relative to J2
    return np.deg2rad(np.asarray(j, dtype=np.float32))


def load_playback(path):
    """Load a recording and extract the arrays the sim consumes.

    Returns ``(rec_times, rec_targets, rec_engaged, rec_duration)``:
        - ``rec_times``: sample times [s], shape [N], starting at 0 (:func:`recorded_times`).
        - ``rec_targets``: coupled arm joint targets [rad], shape [N, NUM_ARM_DOFS] (:func:`arm_targets_rad`).
        - ``rec_engaged``: suction-cup engagement command (robot output ro[0]) per frame, shape [N] bool.
        - ``rec_duration``: recording length [s] (``rec_times[-1]``).
    """
    frames = load_recording(path)
    rec_times = np.asarray(recorded_times(frames), dtype=np.float64)
    rec_targets = np.stack([arm_targets_rad(f) for f in frames]).astype(np.float64)  # [N, NUM_ARM_DOFS]
    rec_engaged = np.array([f.ro[0] for f in frames], dtype=bool)  # [N]
    return rec_times, rec_targets, rec_engaged, float(rec_times[-1])


@wp.kernel
def sample_playback_kernel(
    rec_times: wp.array[float],  # [N] recorded sample times [s], monotonic
    rec_targets: wp.array2d[float],  # [N, num_dofs] coupled arm targets [rad]
    rec_engaged: wp.array[wp.bool],  # [N] suction-cup engagement command (ro[0]) per frame
    sim_step_count: wp.array[int],  # in/out: device sub-step counter (current time = sim_step_count[0] * dt); advanced in place
    last_lo: wp.array[int],  # in/out: cached lower sample index; the forward search resumes from here
    dt: float,  # physics sub-step [s]; sim time = sim_step_count * dt
    # outputs
    joint_target_q: wp.array[float],  # [num_dofs] interpolated position targets [rad]
    engaged: wp.array[wp.bool],  # [1] engagement command sampled at the current time
):
    """Interpolate the recorded joint position targets and sample the engagement command at the
    current time (one thread per DOF); advance the sub-step counter for the next sub-step.

    The time is the integer sub-step count times ``dt`` (exact, no float accumulation). Since sim time
    only advances, the bracketing samples are found by a forward search resumed from the cached
    ``last_lo`` (usually 0-1 steps) rather than a fresh binary search, and the new index is cached
    back. Engagement is a step signal, so its value at ``t`` is ``rec_engaged[lo]``. Clamps to the
    last sample past the end, so the arm holds the final recorded pose.
    """
    dof = wp.tid()
    n = rec_times.shape[0]

    # every thread reads the shared scratch (counter, index) into a local first; then a single thread
    # writes them back, so the reads and writes don't race (this launch is one warp, dim = NUM_ARM_DOFS).
    step = sim_step_count[0]
    if dof == 0:
        sim_step_count[0] = step + 1
    t = float(step) * dt

    # forward search from the cached index for the largest lo with rec_times[lo] <= t
    lo = last_lo[0]
    while lo < n - 1 and rec_times[lo + 1] <= t:
        lo += 1

    if dof == 0:
        last_lo[0] = lo
        engaged[0] = rec_engaged[lo]  # step signal: value of the most recent frame at or before t

    if lo >= n - 1:
        joint_target_q[dof] = rec_targets[n - 1, dof]  # past the end: hold the last recorded pose
        return

    frac = (t - rec_times[lo]) / (rec_times[lo + 1] - rec_times[lo])
    joint_target_q[dof] = rec_targets[lo, dof] * (1.0 - frac) + rec_targets[lo + 1, dof] * frac


class Example:
    def __init__(self, viewer, args):

        # Cache the viewer
        self.viewer = viewer

        # FPS and sim step dt        
        self.fps = FPS  # rendered frames per second
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = max(1, round(self.frame_dt * SIM_HZ))
        self.sim_dt = self.frame_dt / self.sim_substeps

        # Device scratch for sample_playback_kernel: the sub-step counter (drives the sim clock),
        # the cached lower sample index for the forward time search, and the engagement command
        # sampled at the current time (kernel output, for the seal wired up later).
        self.sim_step_count_wp = wp.zeros(1, dtype=wp.int32)
        self.last_lo_wp = wp.zeros(1, dtype=wp.int32)
        self.engaged_wp = wp.zeros(1, dtype=wp.bool)

        # RECORDING_JSONL contains time-stamped joint drive target positions and suction pad engagement
        # states. Load and extract the time-stamps, the joint drive target positions and the
        # suction pad engagement states.
        rec_times, rec_targets, rec_engaged, self.rec_duration = load_playback(RECORDING_JSONL)
        self.rec_times_wp = wp.array(rec_times, dtype=wp.float32)
        self.rec_targets_wp = wp.array(rec_targets, dtype=wp.float32)  # 2d [N, NUM_ARM_DOFS]
        self.rec_engaged_wp = wp.array(rec_engaged, dtype=wp.bool)  # [N]; suction engagement command per frame

        # Load the Fanuc robot arm on a ground plane.
        self.initial_arm_q = rec_targets[0].astype(np.float32)  # drive target at t=0, the start pose
        builder = newton.ModelBuilder()
        newton.solvers.SolverMuJoCo.register_custom_attributes(builder)
        builder.add_usd(str(ROBOT_USD), floating=False, collapse_fixed_joints=True)
        builder.add_ground_plane()

        # Static support box (1x1x1, collidable) at the pick pose -- the pallet the pick box sits on.
        builder.add_shape_box(
            -1,
            xform=wp.transform(wp.vec3(*STATIC_BOX_CENTER), wp.quat_identity()),
            hx=BOX_HALF,
            hy=BOX_HALF,
            hz=BOX_HALF,
        )
        # Dynamic pick box (1x1x0.1, the object to pick) resting on the static box.
        pick_cfg = builder.default_shape_cfg.copy()
        pick_cfg.density = PICK_BOX_DENSITY
        self.pick_box = builder.add_body(
            xform=wp.transform(wp.vec3(*PICK_BOX_CENTER), wp.quat_identity()), label="pick_box"
        )
        builder.add_shape_box(self.pick_box, hx=PICK_BOX_HALF[0], hy=PICK_BOX_HALF[1], hz=PICK_BOX_HALF[2], cfg=pick_cfg)

        self._setup(builder.finalize())
        self.viewer.set_model(self.model)

    def _setup(self, model):
        """Bind ``model``, build the solver/state/control/contacts, start the arm at the first
        recorded pose, and capture the step graph."""
        self.model = model
        self.solver = newton.solvers.SolverMuJoCo(self.model)
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.model.contacts()
        self._reset_initial_state()
        self.capture()
        self._reset_initial_state()  # capture runs one frame; restore the clean start pose

    def _reset_initial_state(self):
        # set only the arm DOFs; any extra DOFs (the dynamic pick box's free joint) keep their
        # built-in rest pose from the model, so the box starts resting on the static box.
        joint_q = self.state_0.joint_q.numpy()
        joint_q[:NUM_ARM_DOFS] = self.initial_arm_q
        self.state_0.joint_q.assign(joint_q)
        self.state_0.joint_qd.zero_()
        newton.eval_fk(self.model, self.state_0.joint_q, self.state_0.joint_qd, self.state_0)

    def capture(self):
        # capturing runs one frame for real, which advances the device sub-step counter and search
        # index, so reset both to 0 afterwards.
        self.graph = None
        if wp.get_device().is_cuda:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph
            self.sim_step_count_wp.zero_()
            self.last_lo_wp.zero_()

    def simulate(self):
        for _ in range(self.sim_substeps):
            # at the device sub-step time (sim_step_count * sim_dt): interpolate the drive target,
            # sample the engagement command, advance the counter and the search index -- all inside the
            # kernel (on-device, so this stays graph-capturable).
            wp.launch(
                sample_playback_kernel,
                dim=NUM_ARM_DOFS,
                inputs=[
                    self.rec_times_wp,
                    self.rec_targets_wp,
                    self.rec_engaged_wp,
                    self.sim_step_count_wp,  # in/out: read as the current time, then advanced in place
                    self.last_lo_wp,  # in/out: forward-search index, resumed and cached
                    float(self.sim_dt),
                ],
                outputs=[self.control.joint_target_q, self.engaged_wp],
            )
            self.state_0.clear_forces()  # zero body_f each sub-step (the suction cup will accumulate into it)
            self.model.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        # the target kernel interpolates and applies the drive targets and advances the sub-step
        # counter before each physics sub-step, so step() just runs one frame.
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()

    def render(self):
        # wall-clock time = physics sub-steps elapsed (read back from the device) * sim_dt
        sim_time = int(self.sim_step_count_wp.numpy()[0]) * self.sim_dt
        self.viewer.begin_frame(int(self.sim_step_count_wp.numpy()[0]) * self.sim_dt)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()

    def gui(self, ui):
        # show the recorded suction-cup command (sampled per sub-step by sample_playback_kernel)
        engaged = bool(self.engaged_wp.numpy()[0])
        ui.text(f"Suction: {'On' if engaged else 'Off'}")

    def test_final(self):
        # the fixed-base arm should hold together on its stiff joint drives: bodies stay at or above
        # the ground (no explosion, no fall-through).
        newton.examples.test_body_state(
            self.model,
            self.state_0,
            "robot arm bodies stay at or above the ground",
            lambda q, qd: q[2] > -0.05,
        )


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
