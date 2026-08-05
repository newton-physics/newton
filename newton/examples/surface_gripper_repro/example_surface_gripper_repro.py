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
# Example Surface Gripper Isaac Sim Repro
#
# Reproduction scene for the surface-gripper on a robot arm. Loads the robot arm from a USD stage
# (Assets/robot_only_newton_flattened.usda) with a fixed base on a ground plane, then plays back a
# recorded FANUC palletizer cycle (Assets/robot_recording_truncated.jsonl). Playback is time-accurate:
# the six arm joint position targets are interpolated from the recorded timestamps at the current simulation
# time (J3 coupled to J2, degrees -> radians) and updated before every physics sub-step, so the arm
# follows the recording at its true speed. The recording's surface-gripper engagement and disengagement
# commands are extracted per frame. Objects are placed in the scene so that they may be gripped
# and manipulated by the the surface gripper as required by the motion of the robot arm and the
# engagement/disengagement commands.

# Command: python -m newton.examples surface_gripper_repro
###########################################################################

from pathlib import Path

import numpy as np
import warp as wp

import newton
import newton.examples
from newton.examples.surface_gripper_repro.box_placements import PlacementConfig, compute_box_placements
from newton.examples.surface_gripper_repro.crate_playback import CratePlayback
from newton.examples.surface_gripper_repro.debug_recorders import (
    DriveTargetRecorder,
    EndEffectorAccelerationRecorder,
    GripperForceRecorder,
    PadBreakMetricRecorder,
)
from newton.examples.surface_gripper_repro.robot_playback import RobotPlayback
from newton.examples.surface_gripper_repro.surface_gripper import (
    SurfaceGripper,
    SurfaceGripperBuilder,
    attach_seal,
    attach_seal_seated,
    evaluate_gripper_force,
)

# assets live alongside this example
ASSETS = Path(__file__).parent / "Assets"

# rendered frames per second
FPS = 60
# target physics rate; sim_substeps = SIM_HZ / FPS physics steps per render frame
SIM_HZ = 120

# recording of the robot arm motion and surface gripper engagement/disengagement.
RECORDING_JSONL = ASSETS / "robot_recording_truncated.jsonl"
# Gaussian smoothing of the recorded drive targets [s]. The recording is a coarse waypoint staircase
# (values held, then stepped ~17 deg), so smoothing recovers a continuous motion. 0 = raw recording;
# larger = smoother (and further from the exact recorded knots). ~1 waypoint interval (~0.08 s) is a
# reasonable start.
SMOOTHING_SIGMA = 0.06

# robot arm USD
ROBOT_USD = ASSETS / "fanuc_arm_flattened_collision.usda"
# J1-J6; recorded joints 6-8 are unused finger DOFs
NUM_ARM_DOFS = 6
# Deepest reach of the finger collision geometry along the grip axis, in the J6_link (flange)
# frame [m] -- the point that would first penetrate the box. The box top is seated here so the fingers
# rest on the box without sinking in. Resolved from the USD (max +x over the Finger_0x meshes).
FINGER_HULL_DEEPEST_X = 0.3109
# Surface gripper on the end-effector (body EE_BODY / J6_link). Four pads at the vents at the
# tips of Finger_01..04, resolved into the J6_link (flange) frame from the USD. Each finger is an
# L-shaped arm whose vent faces the box; the vents are wide-set at the crate edges (a cross ~+/-6 cm on
# one axis, ~+/-13 cm on the other), giving the tilt leverage a real palletizer needs. x=0.309 is the
# vent plane. The grip axis is the flange +x (world-down at the pick), so each pad's
# local +z is rotated onto +x. Positions in the EE body frame [m].
GRIPPER_PADS = (
    (0.2880, 0.1286, 0.0035),
    (0.2885, -0.0024, -0.2022),
    (0.2880, -0.1323, 0.0035),
    (0.2886, -0.0025, 0.2085),
)
# Each pad is a short cylinder of this radius and half-height [m] (also the SHOW_PAD_MARKERS
# disk). Its lip is the circle of PAD_RADIUS on the pad's bottom face (the face toward the box,
# +PAD_HALF_HEIGHT along the grip axis); PAD_LIP_SAMPLES points are sampled around that lip for
# the on-device seating fit (attach_seal_seated).
PAD_RADIUS = 0.03
PAD_HALF_HEIGHT = 0.004
PAD_LIP_SAMPLES = 16
# Grip force (vacuum) [N] per pad. Static hold of the 30 kg panel needs ~74 N/pad, but the
# recorded palletizer motion drives the normal load to ~384 N/pad, so the vacuum must clear that.
F_GRIP_MAX = 450.0
# Per-DOF seal modes (angular natural frequency mu [rad/s], damping ratio). Converted to stiffness/
# damping against the crate by SurfaceGripper.set_natural_frequency_damping_ratio (translation DOFs use
# the crate mass, peel/twist its box inertia).
NORMAL_MODE = (89.44272, 0.018634)
SHEAR_X_MODE = (22.36068, 0.037268)
SHEAR_Y_MODE = (22.36068, 0.037268)
PEEL_X_MODE = (10.79666, 0.124961)
PEEL_Y_MODE = (8.35631, 0.096717)
TWIST_MODE = (2.79962, 0.0)

# The arm picks two boxes in sequence over the recording's two engage/disengage cycles: a wide shallow
# panel at the 1st engagement, then a deep crate at the 2nd and subsequent engagements. Each rests
# on a static pallet until the surface gripper grips and manipulates it.
# (half-extents [m],
# mass [kg]); shape density is 0, so mass/inertia are set on the body.
PANEL = ((0.5, 0.5, 0.04), 30.0)  # 1st engagement -- wide shallow panel, size [1.0, 1.0, 0.08] m
CRATE = ((0.242, 0.166, 0.137), 12.0)  # 2nd engagement -- deep crate, size [0.484, 0.332, 0.274] m
PANEL_PALLET_HALF = (0.54, 0.54, 0.5)
CRATE_PALLET_HALF = (0.206, 0.282, 0.5)
NUM_CRATES = 6

# Set False to disable the surface gripper: the seal wrench is never applied, so the arm plays back the
# recorded trajectory and the pick box just sits on the pallet (useful for inspecting the bare arm
# motion). Read at graph-capture time, so set it before constructing the example.
ENABLE_GRIPPER = True

# Enable the pick-box <-> gripper-geometry rigid contact. On: the box presses flush against the tool
# and tipping is resisted by contact (matches a real vacuum grip on rigid tooling). Off: the soft seal
# alone owns the hold.
ENABLE_PAD_BOX_CONTACT = False

# On engagement, fit the gripped box's pose to the pad lips (surface_gripper.attach_seal_seated,
# an on-device Gauss-Newton fit) and anchor the seal to that fitted pose, so the seal seats the box flush
# on the pads (the lip-SDF standoff becomes a bias the seal pulls closed). Fully kernel-driven, so it
# graph-captures and runs on CPU and graphed CUDA alike.
SEAL_SEAT_ON_ENGAGE = True

# Gauss-Newton iterations for the on-engagement seat fit. 1 is exact for a planar (box) face; raise it for
# curved gripped objects, where each iteration re-samples the SDF at the updated pose to converge.
SEAT_ITERS = 4

# Number of simulation worlds (environments). The whole scene (arm + boxes + pallets + gripper) is
# replicated NUM_WORLDS times, all overlapping at the origin -- Newton's broad phase does not collide
# across worlds, so the copies don't interact, and only world 0 is rendered (see set_visible_worlds).
NUM_WORLDS = 1

# Draw a small non-colliding disk at each pad (GRIPPER_PADS) so the pad layout is visible in
# the viewer. Purely visual (has_shape_collision off); does not affect the physics.
SHOW_PAD_MARKERS = True

# Set False to disable the debug CSV recording -- the end-effector acceleration and the smoothed
# runtime drive targets (see EndEffectorAccelerationRecorder / DriveTargetRecorder). Recording is
# host-side, so it only takes effect on CPU regardless.
RECORD_DEBUG = False

# Seal fractures (releases) once its brittle break metric exceeds this. 1.0 = nominal capacity; a
# value > 1 is a capacity safety factor (the seal tolerates sqrt(threshold)x the nominal elastic peel
# before breaking). Set to 5 (~2.2x): the wide panel (box 1) overhangs the pads, so the arm's
# reorientations spike its peel to ~2x nominal -- the holding force still carries it, so a strict 1.0
# would drop it ~60 frames before the recorded release. The crate (box 0) stays far below either way.
BREAK_THRESHOLD = 5.0

# The break metric must stay over BREAK_THRESHOLD for at least this long before the seal fractures.
# Debounces lone transient spikes (a genuine overload is sustained), so a held box is not dropped by a
# brief sub-step spike. Expressed as a time so it is independent of the sim rate; the sub-step count is
# round(BREAK_HOLD_TIME / sim_dt), floored at 1.
BREAK_HOLD_TIME = 0.033  # [s]


@wp.kernel
def update_seal_break_kernel(
    gripper_command_engaged: wp.array[wp.bool],  # [1] recorded engagement command (ro[0])
    pad_break_metric: wp.array[float],  # [pads] brittle break envelope from the previous force eval
    pad_engaged: wp.array[wp.bool],  # [pads] whether each pad held last sub-step (from attach_seal)
    break_threshold: float,  # break metric above this counts as over-capacity (1.0 = nominal capacity)
    break_hold_steps: int,  # sub-steps a pad must stay over threshold before the gripper fractures
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
                    gripper_seal_broken[g] = True  # sustained overload at this pad vents the whole gripper
            else:
                pad_seal_break_count[pad] = 0  # dipped back under -> not a sustained overload
    hold = cmd and not gripper_seal_broken[g]  # whole gripper engages or releases as a unit
    for pad in range(lo, hi):
        pad_seal_engaged[pad] = hold


def picked_box_seal_modes(gripper_model, state, model, body_b):
    """Per-DOF ``(name, angular natural frequency [rad/s], damping ratio)`` the seal presents for the
    picked body ``body_b``. The tool's stiffness/damping are fixed, so the modes follow the body's mass
    (translation DOFs) and its inertia about each seal axis (rotation DOFs) -- the latter uses the
    body's pose relative to the seal frame (:func:`nat_freq_damping_ratio_to_stiffness_damping` inverted
    per DOF, ``mu = sqrt(k/m_eff)``, ``zeta = d / (2*sqrt(k*m_eff))``).
    """
    gm = gripper_model
    mass = float(model.body_mass.numpy()[body_b])
    i_box = wp.mat33(*model.body_inertia.numpy()[body_b].flatten().tolist())  # inertia tensor, body frame
    q_box = wp.quat(*(float(v) for v in state.body_q.numpy()[body_b][3:7]))
    a = int(gm.gripper_body_id.numpy()[0])
    q_a = wp.quat(*(float(v) for v in state.body_q.numpy()[a][3:7]))
    q_gx = wp.quat(*(float(v) for v in gm.gripper_xform.numpy()[0][3:7]))
    q_px = wp.quat(*(float(v) for v in gm.pad_xform.numpy()[0][3:7]))
    q_seal = q_a * q_gx * q_px  # seal-frame world orientation (rotation of any pad; all share it)

    def i_about(axis):  # box inertia about a world seal axis: rotate the axis into the box body frame
        n = wp.quat_rotate_inv(q_box, wp.quat_rotate(q_seal, axis))
        return float(wp.dot(n, i_box * n))

    def val(name):
        return float(getattr(gm, "gripper_" + name).numpy()[0])

    dofs = (
        ("normal", val("k_normal"), val("d_normal"), mass),
        ("shear_x", val("k_shear_x"), val("d_shear_x"), mass),
        ("shear_y", val("k_shear_y"), val("d_shear_y"), mass),
        ("peel_x", val("k_peel_x"), val("d_peel_x"), i_about(wp.vec3(1.0, 0.0, 0.0))),
        ("peel_y", val("k_peel_y"), val("d_peel_y"), i_about(wp.vec3(0.0, 1.0, 0.0))),
        ("twist", val("k_torsion"), val("d_torsion"), i_about(wp.vec3(0.0, 0.0, 1.0))),
    )
    modes = []
    for name, k, d, m_eff in dofs:
        if k > 0.0 and m_eff > 0.0:
            mu = (k / m_eff) ** 0.5
            zeta = d / (2.0 * (k * m_eff) ** 0.5)
        else:
            mu, zeta = 0.0, 0.0
        modes.append((name, mu, zeta))
    return modes


def box_sdf_mesh(hx, hy, hz, device=None):
    """A ``wp.Mesh`` of an axis-aligned box (half-extents [m]) for SDF queries only (not collision).

    In general the gripped object would supply its own surface mesh; here the pick boxes are primitives,
    so tessellate them once (:meth:`newton.Mesh.create_box`) for the seating fit's SDF queries.
    """
    m = newton.Mesh.create_box(hx, hy, hz)
    verts = np.asarray(m.vertices, dtype=np.float32).reshape(-1, 3)
    idx = np.asarray(m.indices, dtype=np.int32).reshape(-1)
    return wp.Mesh(
        points=wp.array(verts, dtype=wp.vec3, device=device), indices=wp.array(idx, dtype=wp.int32, device=device)
    )


@wp.kernel
def broadcast_arm_targets_kernel(
    num_arm_dofs: int,
    world_stride: int,  # per-world length of joint_target_q
    joint_target_q: wp.array[float],  # in/out: world 0's arm targets [0:num_arm_dofs] copied to every world
):
    """Copy world 0's recorded arm drive targets into every other world's arm DOFs (the worlds are
    identical replicas, so they all follow the same recording). One thread per (world >= 1, arm DOF)."""
    tid = wp.tid()  # dim = (num_worlds - 1) * num_arm_dofs
    w = tid // num_arm_dofs + 1  # target worlds 1..num_worlds-1 (world 0 is the source)
    d = tid % num_arm_dofs
    joint_target_q[w * world_stride + d] = joint_target_q[d]


@wp.kernel
def broadcast_command_kernel(gripper_command_engaged: wp.array[wp.bool]):
    """Copy world 0's sampled engagement command to every gripper (one per world)."""
    g = wp.tid()  # dim = n_grippers
    if g > 0:
        gripper_command_engaged[g] = gripper_command_engaged[0]


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
        # The per-gripper / per-pad runtime arrays are sized from the finalized gripper model further down
        # (one gripper per world, so they scale with NUM_WORLDS).

        # RECORDING_JSONL contains time-stamped joint drive target positions and pad engagement
        # states. Load and extract the time-stamps, the joint drive target positions and the
        # pad engagement states.
        # Apply gaussian smoothing to the raw drive target after loading.
        self.robot_arm_playback = RobotPlayback(RECORDING_JSONL, SMOOTHING_SIGMA, NUM_ARM_DOFS)

        # Build ONE environment (arm + pick boxes + pallets + pad markers), then replicate it across
        # NUM_WORLDS worlds. The copies overlap at the origin (Newton's broad phase never collides across
        # worlds), and the ground plane is added globally (world -1) so a single floor is shared by all.
        env = newton.ModelBuilder()
        env.add_usd(str(ROBOT_USD), floating=False, collapse_fixed_joints=True)
        ee_body_local = env.body_count - 1  # last arm link (J6_link), the flange, within one env

        # Poses for the pick boxes (panel + crates) and pallets, from the arm's recorded engagement poses.
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
        placements = compute_box_placements(env, self.robot_arm_playback, ee_body_local, placement_config)

        # Static pallets (one for the panel, one for the crates).
        for i in range(len(placements.pallet_poses)):
            hx, hy, hz = placements.pallet_dims[i]  # (hx, hy, hz) half-extents [m]
            env.add_shape_box(-1, xform=placements.pallet_poses[i], hx=hx, hy=hy, hz=hz)

        # Pick boxes (panel + crates). Record each box's env-local body id and half-extents (for the SDF meshes).
        box_shape_locals = []
        env_box_half_extents = {}  # env-local body id -> (hx, hy, hz)
        for i in range(len(placements.masses)):
            (hx, hy, hz) = placements.dims[i]
            label = "panel" if i == 0 else f"crate_{i - 1}"
            body = env.add_body(
                xform=placements.wait_poses[i], mass=placements.masses[i], inertia=placements.inertias[i], label=label
            )
            cfg = env.default_shape_cfg.copy()
            cfg.density = 0.0  # body mass is authoritative; the shape adds none
            env_box_half_extents[body] = (hx, hy, hz)
            box_shape_locals.append(env.add_shape_box(body, hx=hx, hy=hy, hz=hz, cfg=cfg))
        box_body_locals = list(env_box_half_extents.keys())
        panel_body_local = box_body_locals[0]
        crate_body_locals = box_body_locals[1:]
        crate_grip_poses = placements.pick_poses[1:]  # where each crate is moved to be gripped

        # Filter every pick box against the whole arm (bodies 0..ee_body_local): the seal owns the hold.
        if not ENABLE_PAD_BOX_CONTACT:
            for shape in range(len(env.shape_body)):
                if 0 <= env.shape_body[shape] <= ee_body_local:  # any robot-arm link (base..gripper)
                    for bs in box_shape_locals:
                        env.add_shape_collision_filter_pair(bs, shape)

        # Pad markers: a thin non-colliding disk of the pad radius at each pad, for viewer visibility only.
        if SHOW_PAD_MARKERS:
            marker_down = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), np.pi / 2.0)  # disk axis -> flange +x
            marker_cfg = env.default_shape_cfg.copy()
            marker_cfg.density = 0.0
            marker_cfg.has_shape_collision = False
            for px, py, pz in GRIPPER_PADS:
                env.add_shape_cylinder(
                    ee_body_local,
                    xform=wp.transform(wp.vec3(px, py, pz), marker_down),
                    radius=PAD_RADIUS,
                    half_height=PAD_HALF_HEIGHT,
                    cfg=marker_cfg,
                )

        # Main scene: shared global ground plane + NUM_WORLDS overlapping copies of the env.
        builder = newton.ModelBuilder()
        builder.add_ground_plane()  # global (world -1): one floor collides with every world
        builder.replicate(env, NUM_WORLDS, spacing=(0.0, 0.0, 0.0))
        self.model = builder.finalize()

        # Map each env-local body id to its global body id in each world (bodies group contiguously per world).
        bw = self.model.body_world.numpy()
        world_bodies = [[b for b in range(self.model.body_count) if bw[b] == w] for w in range(NUM_WORLDS)]

        def gbody(w, local):  # global body id of env-local body ``local`` in world ``w``
            return world_bodies[w][local]

        self.ee_body = gbody(0, ee_body_local)  # world-0 flange (for the debug EE-accel recorder)

        # Per-box SDF meshes + body id -> mesh id for every world's pick boxes (seat fit / queries only;
        # the box shape still owns collision).
        self.box_half_extents = {gbody(w, lb): he for w in range(NUM_WORLDS) for lb, he in env_box_half_extents.items()}
        self.sdf_meshes = {b: box_sdf_mesh(*he, device=self.model.device) for b, he in self.box_half_extents.items()}
        body_mesh_id = np.zeros(self.model.body_count, dtype=np.uint64)
        for b, m in self.sdf_meshes.items():
            body_mesh_id[b] = m.id
        self.body_mesh_id = wp.array(body_mesh_id, dtype=wp.uint64, device=self.model.device)

        # use_mujoco_contacts=False: Newton's collide pipeline (model.collide, run each sub-step) owns
        # collision -- the solver consumes those contacts instead of MuJoCo's internal collision.
        # Contact/constraint budgets scale with the world count (each world adds its own contacts).
        self.solver = newton.solvers.SolverMuJoCo(
            self.model, nconmax=256 * NUM_WORLDS, njmax=2048 * NUM_WORLDS, iterations=10, use_mujoco_contacts=False
        )
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.model.contacts()

        # One surface gripper per world, on that world's flange, tagged world=w. Four pads at the recorded
        # finger offsets (grip axis along flange +x). Seal tuned per-DOF against the crate.
        gripper_builder = SurfaceGripperBuilder()
        pad_down = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), np.pi / 2.0)  # pad +z -> flange +x
        for w in range(NUM_WORLDS):
            gripper = SurfaceGripper(gbody(w, ee_body_local), wp.transform_identity(), world=w)
            gripper.set_natural_frequency_damping_ratio(
                CRATE, F_GRIP_MAX, NORMAL_MODE, SHEAR_X_MODE, SHEAR_Y_MODE, PEEL_X_MODE, PEEL_Y_MODE, TWIST_MODE
            )
            for px, py, pz in GRIPPER_PADS:
                gripper.add_pad(wp.transform(wp.vec3(px, py, pz), pad_down))
            gripper_builder.add_gripper(gripper)
        self.gripper_model = gripper_builder.finalize(device=self.model.device)
        self.gripper_state = self.gripper_model.state()
        self.gripper_control = self.gripper_model.control()
        self.gripper_control.pad_grip_control.fill_(1.0)  # full grip command

        # Per-gripper / per-pad runtime arrays, sized from the gripper model (one gripper per world).
        n_grippers = self.gripper_model.gripper_body_id.shape[0]
        n_pads = self.gripper_model.pad_xform.shape[0]
        self.gripper_command_engaged_wp = wp.zeros(n_grippers, dtype=wp.bool)  # [grippers] recorded engagement command
        self.pad_seal_break_count_wp = wp.zeros(n_pads, dtype=wp.int32)  # consecutive over-threshold steps, per pad
        self.gripper_seal_broken_wp = wp.zeros(n_grippers, dtype=wp.bool)  # [grippers] latched fracture
        self.pad_seal_engaged_wp = wp.zeros(n_pads, dtype=wp.bool)  # [pads] per-pad seal command
        self.pad_offsets = wp.array(
            [g * len(GRIPPER_PADS) for g in range(n_grippers + 1)], dtype=wp.int32
        )  # [grippers+1] CSR: gripper g owns pads [g*npads, (g+1)*npads)

        # Each pad starts targeting its own world's panel body (the gripper model groups pads by world).
        pad_world = self.gripper_model.pad_world.numpy()
        panel_ids = [gbody(w, panel_body_local) for w in range(NUM_WORLDS)]
        self.pad_body_b = wp.array(
            np.array([panel_ids[pad_world[p]] for p in range(n_pads)], dtype=np.int32), dtype=wp.int32
        )

        # One CratePlayback per world (worlds are identical copies, so each moves its own crates on the
        # same disengagement cues). pad_world_start[w] slices the gripper model's pads for world w.
        self.crate_playbacks = [
            CratePlayback(
                self.robot_arm_playback, self.model, [gbody(w, cb) for cb in crate_body_locals], crate_grip_poses
            )
            for w in range(NUM_WORLDS)
        ]
        self.pad_world_start = self.gripper_model.pad_world_start.numpy()  # CSR: world w's pads
        # Per-world stride into joint_target_q, for broadcasting world 0's recorded arm targets to all worlds.
        self.arm_target_stride = self.control.joint_target_q.shape[0] // NUM_WORLDS

        # Start each world's arm at the first recorded pose (its arm DOFs are the first NUM_ARM_DOFS of its
        # per-world block). The pick boxes keep their built-in rest pose, resting on their static box.
        initial_arm_q = self.robot_arm_playback.rec_targets_wp.numpy()[0]  # drive target at t=0, the start pose
        joint_q = self.state_0.joint_q.numpy()
        q_stride = joint_q.shape[0] // NUM_WORLDS
        for w in range(NUM_WORLDS):
            joint_q[w * q_stride : w * q_stride + NUM_ARM_DOFS] = initial_arm_q
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
        # All worlds overlap at the origin; render only world 0 (the others still simulate).
        self.viewer.set_visible_worlds([0])

        # Record the EE acceleration and the smoothed drive targets over the 1st engaged window to CSV
        if RECORD_DEBUG and not wp.get_device().is_cuda:
            self.accel_recorder = EndEffectorAccelerationRecorder(self.ee_body, self.sim_dt)
            self.drive_target_recorder = DriveTargetRecorder(self.sim_dt, NUM_ARM_DOFS)
            self.pad_break_recorder = PadBreakMetricRecorder(
                self.sim_dt, self.robot_arm_playback.rec_duration, len(GRIPPER_PADS)
            )
            # per-pad seal forces over the first lift (simple linear model only; see GripperForceRecorder)
            self.gripper_force_recorder = GripperForceRecorder(self.sim_dt, len(GRIPPER_PADS))

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
            # The playback fills only world 0's arm targets and gripper command; fan them out to the other
            # identical worlds (in-graph, so the whole thing stays graph-capturable).
            if NUM_WORLDS > 1:
                wp.launch(
                    broadcast_arm_targets_kernel,
                    dim=(NUM_WORLDS - 1) * NUM_ARM_DOFS,
                    inputs=[NUM_ARM_DOFS, self.arm_target_stride, self.control.joint_target_q],
                )
                wp.launch(
                    broadcast_command_kernel,
                    dim=self.gripper_command_engaged_wp.shape[0],
                    inputs=[self.gripper_command_engaged_wp],
                )
            self.state_0.clear_forces()  # zero body_f each sub-step (the surface gripper accumulates into it)

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
            # SEAL_SEAT_ON_ENGAGE: anchor to the on-device fitted (seated) box pose instead of the actual
            # one, so the seal seats the box on the pads. Fully kernel-driven, so it graph-captures.
            if SEAL_SEAT_ON_ENGAGE:
                attach_seal_seated(
                    self.state_0,
                    self.gripper_model,
                    self.gripper_state,
                    self.pad_seal_engaged_wp,
                    self.pad_body_b,
                    self.body_mesh_id,
                    PAD_RADIUS,
                    PAD_HALF_HEIGHT,
                    PAD_LIP_SAMPLES,
                    iters=SEAT_ITERS,
                )
            else:
                attach_seal(
                    self.state_0,
                    self.gripper_model,
                    self.gripper_state,
                    self.pad_seal_engaged_wp,
                    self.pad_body_b,
                )
            if ENABLE_GRIPPER:
                evaluate_gripper_force(
                    self.model, self.state_0, self.gripper_model, self.gripper_state, self.gripper_control, self.sim_dt
                )

            self.model.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            if RECORD_DEBUG and not wp.get_device().is_cuda:
                self.accel_recorder.record(
                    self.state_0, self.state_1, self.gripper_command_engaged_wp, self.sim_step_count_wp
                )
                self.drive_target_recorder.record(
                    self.gripper_command_engaged_wp, self.control.joint_target_q, self.sim_step_count_wp
                )
                self.pad_break_recorder.record(self.gripper_state.pad_break_metric, self.sim_step_count_wp)
                self.gripper_force_recorder.record(
                    self.gripper_command_engaged_wp, self.gripper_state.pad_dof_force, self.sim_step_count_wp
                )
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        # the target kernel interpolates and applies the drive targets and advances the sub-step
        # counter before each physics sub-step, so step() just runs one frame.
        # On each crate's disengagement cue, move it onto the pick pallet and retarget the seal to it.
        # pad_body_b and the crate free-joint DOFs are captured by reference, so the in-place assigns
        # take effect on the next graph launch.
        sim_time = int(self.sim_step_count_wp.numpy()[0]) * self.sim_dt
        # Advance each world's crates; when a world's crate becomes active, retarget that world's pads to it.
        pad_body_b = None
        for w in range(NUM_WORLDS):
            active_crate = self.crate_playbacks[w].step(sim_time, self.state_0)
            if active_crate is not None:
                if pad_body_b is None:
                    pad_body_b = self.pad_body_b.numpy()
                pad_body_b[int(self.pad_world_start[w]) : int(self.pad_world_start[w + 1])] = active_crate
        if pad_body_b is not None:
            self.pad_body_b.assign(pad_body_b)
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
        # commanded grip (recorded ro[0], sampled per sub-step by sample_playback_kernel) vs the
        # actual latched seal (command AND-ed with the break/proximity logic in update_seal_break_kernel /
        # attach_seal) -- the two differ if a seal fractured or failed to grab.
        commanded = bool(self.gripper_command_engaged_wp.numpy()[0])
        held = int(self.pad_seal_engaged_wp.numpy().sum())
        ui.text(f"Grip cmd:  {'On' if commanded else 'Off'}  (recording)")
        ui.text(f"Seal engaged: {held}/{len(GRIPPER_PADS)} pads  (actual)")
        # Seal modes for the box currently gripped: same tool k/d, but natural frequency and damping
        # ratio depend on that box's mass/inertia and its pose relative to the seal. Zero when nothing
        # is gripped (no pad engaged).
        ui.text("Picked-box seal modes:")
        modes = picked_box_seal_modes(self.gripper_model, self.state_0, self.model, int(self.pad_body_b.numpy()[0]))
        if held == 0:
            modes = [(name, 0.0, 0.0) for name, _, _ in modes]
        for name, mu, zeta in modes:
            ui.text(f"  {name:7s} wn={mu:6.1f} rad/s  zeta={zeta:.3f}")

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
