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

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import warp as wp

import newton
import newton.examples
from newton.examples.suctioncup.box_placements import PlacementConfig, compute_box_placements
from newton.examples.suctioncup.crate_playback import CratePlayback
from newton.examples.suctioncup.debug_recorders import DriveTargetRecorder, EndEffectorAccelerationRecorder
from newton.examples.suctioncup.robot_playback import RobotPlayback
from newton.examples.suctioncup.surface_gripper import (
    PadShape,
    SurfaceGripper,
    SurfaceGripperBuilder,
    evaluate_gripper_force,
    attach_seal,
    reset_seal_on_contact,
)

# assets live alongside this example
ASSETS = Path(__file__).parent / "Assets"
# robot USD with convex-hull collision added to the suction-gripper (EOAT) meshes
ROBOT_USD = ASSETS / "fanuc_arm_flattened_collision.usda"
# recording with the leading idle removed (truncated from robot_recording.jsonl); it starts just
# before the first joint motion, so the arm moves right away.
RECORDING_JSONL = ASSETS / "robot_recording_truncated.jsonl"

# Gaussian smoothing of the recorded drive targets [s]. The recording is a coarse waypoint staircase
# (values held, then stepped ~17 deg), so smoothing recovers a continuous motion. 0 = raw recording;
# larger = smoother (and further from the exact recorded knots). ~1 waypoint interval (~0.08 s) is a
# reasonable start.
SMOOTHING_SIGMA = 0.06

FPS = 60  # rendered frames per second
SIM_HZ = 120  # target physics rate; sim_substeps = SIM_HZ / FPS physics steps per render frame

NUM_ARM_DOFS = 6  # J1-J6; recorded joints 6-8 are unused finger DOFs

# The arm picks two boxes in sequence over the recording's two engage/disengage cycles: a wide shallow
# panel at the 1st engagement, then a deep crate at the 2nd. Each rests on its own static pallet until
# the suction grips it (the gripper parameters are fixed -- a statement of the tool). (half-extents [m],
# mass [kg]); shape density is 0, so mass/inertia are set on the body.
PANEL = ((0.5, 0.5, 0.04), 30.0)  # 1st engagement -- wide shallow panel, size [1.0, 1.0, 0.08] m
CRATE = ((0.242, 0.166, 0.137), 12.0)  # 2nd engagement -- deep crate, size [0.484, 0.332, 0.274] m
PANEL_PALLET_HALF = (0.54, 0.54, 0.5)
CRATE_PALLET_HALF = (0.206, 0.282, 0.5)
NUM_CRATES = 6

# Deepest reach of the finger collision geometry along the suction axis, in the J6_link (flange)
# frame [m] -- the point that would first penetrate the box. The box top is seated here so the fingers
# rest on the box without sinking in. Resolved from the USD (max +x over the Finger_0x meshes).
FINGER_HULL_DEEPEST_X = 0.3109

# Set False to disable the suction cup: the seal wrench is never applied, so the arm plays back the
# recorded trajectory and the pick box just sits on the pallet (useful for inspecting the bare arm
# motion). Read at graph-capture time, so set it before constructing the example.
ENABLE_GRIPPER = True

# Enable the pick-box <-> gripper-geometry rigid contact. On: the box presses flush against the tool
# and tipping is resisted by contact (matches a real vacuum grip on rigid tooling); pair with a
# tension-only seal (surface_gripper.SEAL_TENSION_ONLY). Off: the soft seal alone owns the hold.
ENABLE_PAD_BOX_CONTACT = False

# When True, re-anchor a pad's seal frame whenever its held body is in external contact (e.g. a crate
# lands on the held panel). The seal snaps to the current relative pose so it yields to the contact
# instead of building a large elastic restoring spike. Uses the previous sub-step's contact set.
SEAL_RESET_ON_CONTACT = True

# Draw a small non-colliding disk at each suction cup (GRIPPER_PADS) so the cup layout is visible in
# the viewer. Purely visual (has_shape_collision off); does not affect the physics.
SHOW_CUP_MARKERS = True

# Set False to disable the debug CSV recording -- the end-effector acceleration and the smoothed
# runtime drive targets (see EndEffectorAccelerationRecorder / DriveTargetRecorder). Recording is
# host-side, so it only takes effect on CPU regardless.
RECORD_DEBUG = False

# Seal fractures (releases) once its brittle break metric exceeds this. 1.0 = nominal capacity; a
# value > 1 is a capacity safety factor (the seal tolerates sqrt(threshold)x the nominal elastic peel
# before breaking). Set to 5 (~2.2x): the wide panel (box 1) overhangs the cups, so the arm's
# reorientations spike its peel to ~2x nominal -- the holding force still carries it, so a strict 1.0
# would drop it ~60 frames before the recorded release. The crate (box 0) stays far below either way.
BREAK_THRESHOLD = 5.0

# The break metric must stay over BREAK_THRESHOLD for at least this long before the seal fractures.
# Debounces lone transient spikes (a genuine overload is sustained), so a held box is not dropped by a
# brief sub-step spike. Expressed as a time so it is independent of the sim rate; the sub-step count is
# round(BREAK_HOLD_TIME / sim_dt), floored at 1.
BREAK_HOLD_TIME = 0.033  # [s]

# Suction gripper on the end-effector (body EE_BODY / J6_link). Four cups at the suction vents at the
# tips of Finger_01..04, resolved into the J6_link (flange) frame from the USD. Each finger is an
# L-shaped arm whose vent faces the box; the vents are wide-set at the crate edges (a cross ~+/-6 cm on
# one axis, ~+/-13 cm on the other), giving the tilt leverage a real palletizer needs. x=0.309 is the
# vent (suction-face) plane. The suction axis is the flange +x (world-down at the pick), so each pad's
# local +z is rotated onto +x. Positions in the EE body frame [m].
GRIPPER_PADS = (
    (0.2880, 0.1286, 0.0035),
    (0.2885, -0.0024, -0.2022),
    (0.2880, -0.1323, 0.0035),
    (0.2886, -0.0025, 0.2085),
)


@dataclass(frozen=True)
class GripperParams:
    """Per-pad suction-seal tuning (see :class:`~newton.examples.suctioncup.surface_gripper.SurfaceGripper`).

    Field-for-field the ``SurfaceGripper`` keyword arguments except the runtime ``body_id`` / ``xform``,
    so it unpacks straight into the constructor (``SurfaceGripper(body_id=..., xform=..., **asdict(...))``).
    """

    k_normal: float  # normal stiffness [N/m]
    d_normal: float  # normal damping [N.s/m]
    f_normal_max: float  # per-pad break threshold [N]
    f_grip_max: float  # per-pad suction preload [N]
    k_shear_x: float  # shear stiffness [N/m]
    k_shear_y: float
    mu_x: float  # shear friction coefficient
    mu_y: float
    d_peel_x: float  # peel damping [N.m.s/rad]
    d_peel_y: float
    shape: int  # PadShape
    dim_a: float  # pad radius (CIRCLE) [m]
    dim_b: float
    d_shear_x: float = 0.0  # shear damping [N.s/m]; not in gripper.pdf, kept at 0
    d_shear_y: float = 0.0
    peel_capacity_scale: float = 1.0  # multiplies the geometric peel capacity (peel-limited lifts)


# Tuned for the light pick box (~1 kg, weight ~10 N). Preload ~= box weight so the box rests against
# the pads (constant contact); the break threshold is well above the carry loads so the seal holds.
# Damped springs so the four redundant pads settle, not ring. Stiff seal so the box tracks the flange
# rigidly (a soft seal lets it swing like a pendulum under the fast arm). The seal forces are applied
# explicitly, so with the small box (m = 1, I ~ 4e-3) at 240 Hz: near-critical damping keeps k stable
# up to ~m/dt^2, but the angular d_peel must stay tiny (dt < 2*I/d_peel) or it diverges. Seal
# stiffness is bounded by explicit stability at 240 Hz (omega*dt must stay well below 2, or the seal
# rings): k ~ 6000 tracks the box with ~mm lag while staying smooth.
GRIPPER_PARAMS = GripperParams(
    # Normal - translation - z
    k_normal=96000.0,  # stiff, like a vacuum cup on rigid tooling; sets the tilt tracking (peel angle ~0.2deg)
    d_normal=40.0,  # low: the wide-cup couple amplifies damping into the explicit-integrator limit at 240 Hz
    # f_normal_max is the seal's real hold/break capacity: the clamp on fz, i.e. the most tension a cup
    # can pull. 2000 N/cup sits well above any pick here (4 cups >> 100 kg), so the seal holds robustly
    # and the brittle-break trips only on genuine peel/overload, not on the panel's overhang transient.
    f_normal_max=2000.0,  # per-pad normal hold / break threshold [N]
    f_grip_max=50.0,  # per-pad suction preload [N]; gentle baseline push + shear/peel capacity floor
    # Shear - translation - x,y
    k_shear_x=6000.0,
    k_shear_y=6000.0,
    # High friction: when the arm holds the flange with the suction axis near-horizontal, the box
    # weight is a pure shear load, and shear capacity = mu * |holding force|. High mu keeps ample
    # margin through the arm's fast reorientation so the box doesn't slip and dangle.
    mu_x=16.0,
    mu_y=16.0,
    # Shear damping must stay small with the wide-set cups: the far cups' tangential velocity scales
    # with the cup spread, so d_shear's effective twist damping is amplified by ~Sigma r^2 (75x vs a
    # tight cluster) and overshoots the explicit-integrator limit at 240 Hz if raised much above ~30.
    d_shear_x=20.0,
    d_shear_y=20.0,
    # peel rotation- x,y. Small, for the same explicit-integrator reason as the shear/normal damping
    # above (the wide cup couple amplifies it); the wide-set cups need little peel damping to stay put.
    d_peel_x=0.5,
    d_peel_y=0.5,
    peel_capacity_scale=1.0,  # geometric capacity only; the real wide-set cups should hold without a fudge
    shape=int(PadShape.CIRCLE),
    dim_a=0.03,
    dim_b=0.03,
)


@wp.kernel
def update_seal_break_kernel(
    gripper_command_engaged: wp.array[wp.bool],  # [1] recorded engagement command (ro[0])
    pad_break_metric: wp.array[float],  # [pads] brittle break envelope from the previous force eval
    pad_engaged: wp.array[wp.bool],  # [pads] whether each pad held last sub-step (from attach_seal)
    break_threshold: float,  # break metric above this counts as over-capacity (1.0 = nominal capacity)
    break_hold_steps: int,  # sub-steps a cup must stay over threshold before the gripper fractures
    pad_offsets: wp.array[int],  # [grippers+1] CSR offsets: gripper g owns pads [pad_offsets[g], pad_offsets[g+1])
    pad_seal_break_count: wp.array[int],  # [pads] in/out: consecutive over-threshold sub-steps, per pad
    gripper_seal_broken: wp.array[wp.bool],  # [grippers] in/out: latched gripper-wide break within an engaged window
    # outputs
    pad_seal_engaged: wp.array[wp.bool],  # [pads] out: seal command fed to attach_seal
):
    """
    One thread per gripper.
    If the gripper is commanded to disengage then 
    1) set the per pad seal break count to 0
    2) set the per pad engagement state to False
    3) set the per gripper seal broken state to False
    If the gripper is commanded to engage and the pad is engaged then 
    1) increment the per pad break count if the break metric is True
    2) if any pad break count exceeds a threshold then break the seal on all pads
    """
    g = wp.tid()  # one thread per gripper -> sole owner of this gripper's latch, counters, and commands
    lo = pad_offsets[g]  # this gripper's pads are [lo, hi)
    hi = pad_offsets[g + 1]
    cmd = gripper_command_engaged[0]
    if not cmd:
        gripper_seal_broken[g] = False  # recorded release clears the gripper latch for the next cycle
        for pad in range(lo, hi):
            pad_seal_break_count[pad] = 0
    else:
        for pad in range(lo, hi):
            if pad_engaged[pad] and pad_break_metric[pad] > break_threshold:
                pad_seal_break_count[pad] = pad_seal_break_count[pad] + 1
                if pad_seal_break_count[pad] >= break_hold_steps:
                    gripper_seal_broken[g] = True  # sustained overload at this cup vents the whole gripper
            else:
                pad_seal_break_count[pad] = 0  # dipped back under -> not a sustained overload
    hold = cmd and not gripper_seal_broken[g]  # whole gripper engages or releases as a unit
    for pad in range(lo, hi):
        pad_seal_engaged[pad] = hold


def seal_modes_for(gripper, spec):
    """The seal's three spring-damper modes for a gripped box ``spec = ((hx, hy, hz), mass)``, as
    ``(name, omega_n [rad/s], zeta)`` per mode. ``ixx`` is the box's tilt inertia about a horizontal
    grip axis; ``hz`` is the COM depth below the top-face grip. Shown in the side panel."""
    (_hx, hy, hz), mass = spec
    ixx = mass / 3.0 * (hy * hy + hz * hz)
    return (
        (
            "peel",
            gripper.peel_natural_frequency(ixx, mass, hz),
            gripper.peel_damping_ratio(ixx, mass, GRIPPER_PARAMS.d_peel_x, hz),
        ),
        ("normal", gripper.normal_natural_frequency(mass), gripper.normal_damping_ratio(mass, GRIPPER_PARAMS.d_normal)),
        ("shear", gripper.shear_natural_frequency(mass), gripper.shear_damping_ratio(mass, GRIPPER_PARAMS.d_shear_x)),
    )


class Example:
    def __init__(self, viewer, args):

        # Cache the viewer
        self.viewer = viewer

        # FPS and sim step dt
        self.fps = FPS  # rendered frames per second
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = max(1, round(self.frame_dt * SIM_HZ))
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.break_hold_steps = max(1, round(BREAK_HOLD_TIME / self.sim_dt))  # debounce span in sub-steps

        # sim_step_count_wp stores the number of completed simulation steps.
        # last_lo_wp is used to iterate through the recording of the robot arm.
        # gripper_command_engaged_wp is the engagement state of the gripper as read from the recording of the robot arm.
        # gripper_seal_broken_wp is the fracture state of the gripper.
        # pad_seal_engaged_wp is the per pad engagement state after accounting for gripper_command_engaged_wp and gripper_seal_broken_wp.
        # pad_offsets maps each gripper to the range of pads owned by the gripper.
        self.sim_step_count_wp = wp.zeros(1, dtype=wp.int32)
        self.last_lo_wp = wp.zeros(1, dtype=wp.int32)
        self.gripper_command_engaged_wp = wp.zeros(1, dtype=wp.bool)  # [gripper] recorded engagement command
        self.pad_seal_break_count_wp = wp.zeros(len(GRIPPER_PADS), dtype=wp.int32)  # consecutive over-threshold steps, per pad
        self.gripper_seal_broken_wp = wp.zeros(1, dtype=wp.bool)  # [gripper] latched fracture
        self.pad_seal_engaged_wp = wp.zeros(len(GRIPPER_PADS), dtype=wp.bool)  # [pads] per-pad seal command
        self.pad_offsets = wp.array([0, len(GRIPPER_PADS)], dtype=wp.int32)

        # RECORDING_JSONL contains time-stamped joint drive target positions and suction pad engagement
        # states. Load and extract the time-stamps, the joint drive target positions and the
        # suction pad engagement states.
        # Apply gaussian smoothing to the raw drive target after loading.
        self.robot_arm_playback = RobotPlayback(RECORDING_JSONL, SMOOTHING_SIGMA, NUM_ARM_DOFS)

        # Load the Fanuc robot arm on a ground plane.
        builder = newton.ModelBuilder()
        builder.add_usd(str(ROBOT_USD), floating=False, collapse_fixed_joints=True)
        ee_body = builder.body_count - 1  # last arm link (J6_link) is the end-effector flange
        builder.add_ground_plane()

        # Compute poses for every pick box (panel + crates) and every pallet (one for the panel,
        # one for the crates).
        # Each crate is initially posed at a waiting pose and then moved one at a time to the
        # grip pose on the corresponding pallet so that the crate may be gripped by the gripper.
        # The panel is immediately ready for gripping so its wait pose is equal to its grip pose.
        # The pick and pallet poses are computed using the pose of the end effector at engagement time.
        # These poses are computed using the recording of the robot arm motion.
        placement_config = PlacementConfig(
            num_arm_dofs=NUM_ARM_DOFS,
            finger_hull_deepest_x=FINGER_HULL_DEEPEST_X,
            gripper_pads=GRIPPER_PADS,
            panel=PANEL,
            crate=CRATE,
            num_crates=NUM_CRATES,
            panel_pallet_half=PANEL_PALLET_HALF,
            crate_pallet_half=CRATE_PALLET_HALF,
        )
        placements = compute_box_placements(builder, self.robot_arm_playback, ee_body, placement_config)

        # Add each static pallet: one for the panel and one for the crates.
        nb_pallets = len(placements.pallet_poses)
        for i in range(nb_pallets):
            pallet_pose = placements.pallet_poses[i]
            hx, hy, hz = placements.pallet_dims[i]  # (hx, hy, hz) half-extents [m]
            builder.add_shape_box(-1, xform=pallet_pose, hx=hx, hy=hy, hz=hz)

        # Add the panel and crates that will be gripped by the gripper.
        box_body_ids, box_shape_ids = [], []
        nb_boxes = len(placements.masses)
        for i in range(nb_boxes):
            (hx, hy, hz) = placements.dims[i]
            label = "panel" if i == 0 else f"crate_{i - 1}"
            body = builder.add_body(
                xform=placements.wait_poses[i], mass=placements.masses[i], inertia=placements.inertias[i], label=label
            )
            cfg = builder.default_shape_cfg.copy()
            cfg.density = 0.0  # body mass is authoritative; the shape adds none
            box_body_ids.append(body)
            box_shape_ids.append(builder.add_shape_box(body, hx=hx, hy=hy, hz=hz, cfg=cfg))

        # Store the body and shape ids for panel and crates.
        # Store the grip poses of the crates so that the crates may be moved to their grip pose
        # and made ready for gripping.
        panel_body_id, panel_shape_id = box_body_ids[0], box_shape_ids[0]
        crate_body_ids, crate_shape_ids = box_body_ids[1:], box_shape_ids[1:]
        crate_grip_poses = placements.pick_poses[1:]  # where each crate is moved to be gripped

        # Filter every pick box against the whole robot arm (bodies 0..ee_body): the seal owns the
        # hold, and the wide panel swings up against the wrist/forearm links during the carry, so
        # letting them collide would fight the seal (see ENABLE_PAD_BOX_CONTACT / SEAL_TENSION_ONLY).
        # Boxes still collide with the pallets, the panel, each other (so the crates stack), and the ground.
        if not ENABLE_PAD_BOX_CONTACT:
            for shape in range(len(builder.shape_body)):
                if 0 <= builder.shape_body[shape] <= ee_body:  # any robot-arm link (base..gripper)
                    builder.add_shape_collision_filter_pair(panel_shape_id, shape)
                    for cs in crate_shape_ids:
                        builder.add_shape_collision_filter_pair(cs, shape)

        # Cup markers: a thin non-colliding disk at each suction cup so the cup layout is visible in the
        # viewer (radius = the modeled cup radius dim_a; oriented so the disk faces along the suction axis).
        if SHOW_CUP_MARKERS:
            marker_down = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), np.pi / 2.0)  # disk axis -> flange +x
            marker_cfg = builder.default_shape_cfg.copy()
            marker_cfg.density = 0.0
            marker_cfg.has_shape_collision = False
            for px, py, pz in GRIPPER_PADS:
                builder.add_shape_cylinder(
                    ee_body,
                    xform=wp.transform(wp.vec3(px, py, pz), marker_down),
                    radius=GRIPPER_PARAMS.dim_a,
                    half_height=0.004,
                    cfg=marker_cfg,
                )

        # The newton scene is complete.
        self.model = builder.finalize()
        self.solver = newton.solvers.SolverMuJoCo(self.model, nconmax=256, njmax=2048, iterations=10)
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.model.contacts()

        # Suction gripper on the end-effector: one SurfaceGripper on the flange with four pads at the
        # recorded finger offsets, suction axis along the flange +x (pad local +z rotated onto +x).
        # Driven by the recorded ro[0] command -- all four pads engage/release together, sealing the
        # dynamic pick box.
        gripper = SurfaceGripper(
            body_id=ee_body,
            xform=wp.transform_identity(),  # gripper frame == flange body frame
            **asdict(GRIPPER_PARAMS),
        )
        pad_down = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), np.pi / 2.0)  # pad +z -> flange +x
        for px, py, pz in GRIPPER_PADS:
            gripper.add_pad(wp.transform(wp.vec3(px, py, pz), pad_down))
        gripper_builder = SurfaceGripperBuilder()
        gripper_builder.add_gripper(gripper)
        self.gripper_model = gripper_builder.finalize(device=self.model.device)
        self.gripper_state = self.gripper_model.state()
        self.gripper_control = self.gripper_model.control()
        self.gripper_control.pad_grip_control.fill_(1.0)  # full suction command

        self.pad_body_b = wp.full(len(GRIPPER_PADS), panel_body_id, dtype=wp.int32)
        # Moves each parked crate onto the pick pallet on its disengagement cue (see CratePlayback).
        self.crate_playback = CratePlayback(self.robot_arm_playback, self.model, crate_body_ids, crate_grip_poses)

        # The seal's spring-damper modes depend on the gripped box (see seal_modes_for); precompute a set
        # per box, shown in the side panel with the active one selected as the seal retargets (gui()).
        self.panel_seal_modes = seal_modes_for(gripper, PANEL)
        self.crate_seal_modes = seal_modes_for(gripper, CRATE)
        self.seal_modes = self.panel_seal_modes  # panel is gripped first; swapped to the crate at the switch

        # Start the arm at the first recorded pose. Set only the arm DOFs; the pick box's free-joint
        # DOFs keep their built-in rest pose (from add_body), so it starts resting on the static box.
        initial_arm_q = self.robot_arm_playback.rec_targets_wp.numpy()[0]  # drive target at t=0, the start pose
        joint_q = self.state_0.joint_q.numpy()
        joint_q[:NUM_ARM_DOFS] = initial_arm_q
        self.state_0.joint_q.assign(joint_q)
        self.state_0.joint_qd.zero_()
        newton.eval_fk(self.model, self.state_0.joint_q, self.state_0.joint_qd, self.state_0)

        # Capture one frame of physics into a CUDA graph, then restore the clean start pose (capturing
        # runs the frame for real, advancing the state).
        self.capture()
        self.state_0.joint_q.assign(joint_q)
        self.state_0.joint_qd.zero_()
        newton.eval_fk(self.model, self.state_0.joint_q, self.state_0.joint_qd, self.state_0)

        self.viewer.set_model(self.model)

        # Record the EE acceleration and the smoothed drive targets over the 1st engaged window to CSV
        if RECORD_DEBUG and not wp.get_device().is_cuda:
            self.accel_recorder = EndEffectorAccelerationRecorder(ee_body, self.sim_dt)
            self.drive_target_recorder = DriveTargetRecorder(self.sim_dt, NUM_ARM_DOFS)

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
            # Interpolate the arm drive targets (joint_target_q) and sample the engagement command
            # (gripper_command_engaged_wp) at the current sim time (sim_step_count_wp*sim_dt)
            # Advance the search index (last_lo_wp) through the recording, 
            # and advance sim time (sim_step_count_wp) for the next sub-step.
            self.robot_arm_playback.step(
                self.sim_step_count_wp,  # in/out: read as the current time, then advanced in place
                self.last_lo_wp,  # in/out: forward-search index, resumed and cached
                self.sim_dt,
                self.control.joint_target_q,
                self.gripper_command_engaged_wp,
            )
            self.state_0.clear_forces()  # zero body_f each sub-step (the suction cup accumulates into it)

            # Break the gripper (and per pad) seal based on pad_break_metric and a threshold time for 
            # pad_break_metric being continuously True.
            wp.launch(
                update_seal_break_kernel,
                dim=self.gripper_seal_broken_wp.shape[0],  # one thread per gripper
                inputs=[
                    self.gripper_command_engaged_wp,
                    self.gripper_state.pad_break_metric,
                    self.gripper_state.pad_engaged,
                    float(BREAK_THRESHOLD),
                    int(self.break_hold_steps),
                    self.pad_offsets,
                    self.pad_seal_break_count_wp,
                    self.gripper_seal_broken_wp,
                ],
                outputs=[self.pad_seal_engaged_wp],
            )
            # Commit this sub-step's per-pad seal command into the gripper state: set pad_engaged, and on
            # each pad's rising edge cache its seal anchor frame relative to its target body (pad_body_b).
            attach_seal(
                self.model,
                self.state_0,
                self.contacts,
                self.gripper_model,
                self.gripper_state,
                self.pad_seal_engaged_wp,
                self.pad_body_b,
            )
            if ENABLE_GRIPPER and SEAL_RESET_ON_CONTACT:
                reset_seal_on_contact(self.model, self.state_0, self.contacts, self.gripper_model, self.gripper_state)
            if ENABLE_GRIPPER:
                evaluate_gripper_force(
                    self.model, self.state_0, self.gripper_model, self.gripper_state, self.gripper_control, self.sim_dt
                )

            self.model.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            if RECORD_DEBUG and not wp.get_device().is_cuda:
                self.accel_recorder.record(self.state_0, self.state_1, self.gripper_command_engaged_wp, self.sim_step_count_wp)
                self.drive_target_recorder.record(self.gripper_command_engaged_wp, self.control.joint_target_q, self.sim_step_count_wp)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        # the target kernel interpolates and applies the drive targets and advances the sub-step
        # counter before each physics sub-step, so step() just runs one frame.
        # On each crate's disengagement cue, move it onto the pick pallet and retarget the seal to it.
        # pad_body_b and the crate free-joint DOFs are captured by reference, so the in-place assigns
        # take effect on the next graph launch.
        sim_time = int(self.sim_step_count_wp.numpy()[0]) * self.sim_dt
        active_crate = self.crate_playback.step(sim_time, self.state_0)
        if active_crate is not None:
            self.pad_body_b.assign(np.full(len(GRIPPER_PADS), active_crate, dtype=np.int32))
            self.seal_modes = self.crate_seal_modes  # side-panel modes now describe the crate
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()

    def render(self):
        # wall-clock time = physics sub-steps elapsed (read back from the device) * sim_dt
        sim_time = int(self.sim_step_count_wp.numpy()[0]) * self.sim_dt
        self.viewer.begin_frame(sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()

    def gui(self, ui):
        # commanded suction (recorded ro[0], sampled per sub-step by sample_playback_kernel) vs the
        # actual latched seal (command AND-ed with the break/proximity logic in update_seal_break_kernel /
        # attach_seal) -- the two differ if a seal fractured or failed to grab.
        commanded = bool(self.gripper_command_engaged_wp.numpy()[0])
        held = int(self.pad_seal_engaged_wp.numpy().sum())
        ui.text(f"Suction cmd:  {'On' if commanded else 'Off'}  (recording)")
        ui.text(f"Seal engaged: {held}/{len(GRIPPER_PADS)} pads  (actual)")
        # seal spring-damper modes for the picked box (constant): natural frequency and damping ratio
        ui.text("Seal modes:")
        for name, omega_n, zeta in self.seal_modes:
            ui.text(f"  {name:6s} f_n={omega_n / (2.0 * np.pi):5.2f} Hz  zeta={zeta:.2f}")

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
