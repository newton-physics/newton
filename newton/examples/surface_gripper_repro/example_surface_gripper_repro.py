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
# (Assets/fanuc_arm.usda) with a fixed base on a ground plane, then plays back a
# recorded FANUC palletizer cycle (Assets/robot_recording.jsonl). Playback is time-accurate:
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
from newton.examples.surface_gripper_repro.robot_playback import RobotPlayback
from newton.selection import ArticulationView
from newton.examples.surface_gripper_repro.surface_gripper import (
    SurfaceGripper,
    SurfaceGripperBuilder,
    attach_seal,
    attach_seal_seated,
    evaluate_gripper_force,
    evaluate_seal_quality,
)

# Asset paths (global constants). All assets live in the Assets/ directory alongside this example.
ASSETS = Path(__file__).parent / "Assets"
# robot arm USD
ROBOT_USD = ASSETS / "fanuc_arm.usda"
# ArticulationView pattern matching the robot arm's USD root prim label (used to find its joint_q offsets).
ROBOT_ARTICULATION_PATTERN = "*Robot*"
# Pick scene (2 static pallets + panel + 6 crates as box colliders at their waiting poses).
PICK_SCENE_USD = ASSETS / "pick_scene.usda"
# recording of the robot arm motion and surface gripper engagement/disengagement.
RECORDING_JSONL = ASSETS / "robot_recording.jsonl"

# rendered frames per second
FPS = 60
# target physics rate; sim_substeps = SIM_HZ / FPS physics steps per render frame
SIM_HZ = 120

# Number of simulation worlds (environments). The whole scene (arm + boxes + pallets + gripper) is
# replicated NUM_WORLDS times, all overlapping at the origin -- Newton's broad phase does not collide
# across worlds, so the copies don't interact, and only world 0 is rendered (see set_visible_worlds).
NUM_WORLDS = 1

# Gaussian smoothing of the recorded drive targets [s]. The recording is a coarse waypoint staircase
# (values held, then stepped ~17 deg), so smoothing recovers a continuous motion. 0 = raw recording;
# larger = smoother (and further from the exact recorded knots). ~1 waypoint interval (~0.08 s) is a
# reasonable start.
SMOOTHING_SIGMA = 0.06

# J1-J6; recorded joints 6-8 are unused finger DOFs
NUM_ARM_DOFS = 6

# USD prim path of the panel rigid body (the 1st picked box) in pick_scene.usda.
PANEL_PRIM = "/pick_scene/PickBoxes/panel"
# USD prim paths of the six crate rigid bodies in pick_scene.usda, in pick order.
CRATE_PRIMS = (
    "/pick_scene/PickBoxes/crate_0",
    "/pick_scene/PickBoxes/crate_1",
    "/pick_scene/PickBoxes/crate_2",
    "/pick_scene/PickBoxes/crate_3",
    "/pick_scene/PickBoxes/crate_4",
    "/pick_scene/PickBoxes/crate_5",
)
# Each crate is authored at a waiting pose, where it cannot be reached by the surface gripper, and then moved
# to a grip pose (wp.transform, position [m] + quat), where it can be reached. At each disengagement signal in
# RECORDING_JSONL, the next crate to be gripped is teleported from the waiting pose to the grip pose.
CRATE_GRIP_POSES = (
    wp.transform(wp.vec3(-1.53891122341156, -1.339923620223999, 0.7991625070571899), wp.quat(0.0, 0.0, 0.7071067690849304, 0.7071067690849304)),
    wp.transform(wp.vec3(-1.5389012098312378, -1.339895486831665, 0.7957268357276917), wp.quat(0.0, 0.0, 0.7071067690849304, 0.7071067690849304)),
    wp.transform(wp.vec3(-1.5389126539230347, -1.3399271965026855, 0.8007364869117737), wp.quat(0.0, 0.0, 0.7071067690849304, 0.7071067690849304)),
    wp.transform(wp.vec3(-1.538902759552002, -1.3398996591567993, 0.7959489226341248), wp.quat(0.0, 0.0, 0.7071067690849304, 0.7071067690849304)),
    wp.transform(wp.vec3(-1.5389012098312378, -1.3398951292037964, 0.7962862849235535), wp.quat(0.0, 0.0, 0.7071067690849304, 0.7071067690849304)),
    wp.transform(wp.vec3(-1.538907527923584, -1.33991277217865, 0.7989855408668518), wp.quat(0.0, 0.0, 0.7071067690849304, 0.7071067690849304)),
)

# Each pad is represented with a thin cylinder in ROBOT_USD
PAD_PRIMS = (
    "/Robot/J6_link/GripperPads/pad_0",
    "/Robot/J6_link/GripperPads/pad_1",
    "/Robot/J6_link/GripperPads/pad_2",
    "/Robot/J6_link/GripperPads/pad_3",
)

# A pad's lip is the circle of the pad radius on its bottom face (toward the box, +half-height along the
# grip axis); PAD_LIP_SAMPLES points are sampled around that lip for the on-device seating fit (attach_seal_seated).
PAD_LIP_SAMPLES = 16
# Gauss-Newton iterations for the on-engagement seat fit. 1 is exact for a planar (box) face; raise it for
# curved gripped objects, where each iteration re-samples the SDF at the updated pose to converge.
SEAT_ITERS = 4
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


# On engagement, fit the gripped box's pose to the pad lips (surface_gripper.attach_seal_seated,
# an on-device Gauss-Newton fit) and anchor the seal to that fitted pose, so the seal seats the box flush
# on the pads (the lip-SDF standoff becomes a bias the seal pulls closed). Fully kernel-driven, so it
# graph-captures and runs on CPU and graphed CUDA alike.
SEAL_SEAT_ON_ENGAGE = True

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
    pad_break_metric: wp.array[float],  # [pads] brittle break envelope from the previous force eval
    pad_engaged_bs_prev: wp.array[wp.vec2i],  # [pads] gripped body/shape last sub-step (``[0] < 0`` = was released)
    break_threshold: float,  # break metric above this counts as over-capacity (1.0 = nominal capacity)
    break_hold_steps: int,  # sub-steps a pad must stay over threshold before the gripper fractures
    pad_offsets: wp.array[int],  # [grippers+1] start indices: gripper g owns pads [pad_offsets[g] : pad_offsets[g+1]]
    pad_seal_break_count_prev: wp.array[int],  # [pads] consecutive over-threshold sub-steps from the previous sub-step (read)
    pad_seal_break_count_curr: wp.array[int],  # [pads] consecutive over-threshold sub-steps for the current sub-step (written)
    # in/out: initialised by update_engagement_signals_kernel; overwritten with the break-logic result
    pad_engaged_bs_curr: wp.array[wp.vec2i],  # [pads] gripper_state_input_curr.pad_engaged_bs
):
    """One thread per gripper. Reads the engagement state from pad_engaged_bs_curr (set by
    update_engagement_signals_kernel), checks whether any pad's break metric has been over threshold for
    break_hold_steps consecutive sub-steps, and clears pad_engaged_bs_curr to (-1, -1) for the whole
    gripper if any pad sustained an overload.
    """
    g = wp.tid()
    lo = pad_offsets[g]  # this gripper's pads are [lo, hi)
    hi = pad_offsets[g + 1]
    cmd = pad_engaged_bs_curr[lo][0] >= 0
    any_broken = wp.bool(False)
    for p in range(lo, hi):
        if not cmd:
            pad_seal_break_count_curr[p] = 0
        elif pad_engaged_bs_prev[p][0] >= 0 and pad_break_metric[p] > break_threshold:
            pad_seal_break_count_curr[p] = pad_seal_break_count_prev[p] + 1
            if pad_seal_break_count_curr[p] >= break_hold_steps:
                any_broken = True
        else:
            pad_seal_break_count_curr[p] = 0
    hold = cmd and not any_broken
    for p in range(lo, hi):
        if not hold:
            pad_engaged_bs_curr[p] = wp.vec2i(-1, -1)


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
    body_q = state.body_q.numpy()
    qb = body_q[body_b]  # [px,py,pz, qx,qy,qz,qw]; [3:7] is the quaternion
    q_box = wp.quat(float(qb[3]), float(qb[4]), float(qb[5]), float(qb[6]))
    a = int(gm.gripper_body_id.numpy()[0])
    qa = body_q[a]
    q_a = wp.quat(float(qa[3]), float(qa[4]), float(qa[5]), float(qa[6]))
    gx = gm.gripper_xform.numpy()[0]
    q_gx = wp.quat(float(gx[3]), float(gx[4]), float(gx[5]), float(gx[6]))
    px = gm.pad_xform.numpy()[0]
    q_px = wp.quat(float(px[3]), float(px[4]), float(px[5]), float(px[6]))
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


def read_pad_dimensions(builder, robot) -> tuple[list[float], list[float]]:
    """Per-pad radius and half-height [m], read from the loaded arm USD.

    Each pad is a visual cylinder under J6_link/GripperPads; a cylinder's ``shape_scale`` is
    ``(radius, half_height, 0)``. The pads need not be identical, so each is read separately.

    Args:
        builder: the ``ModelBuilder`` the arm USD was loaded into.
        robot: the dict returned by ``add_usd`` for the arm; its ``path_shape_map`` maps prim path -> shape id.

    Returns:
        ``(radii, half_heights)`` -- two parallel lists [m], one entry per pad in :data:`PAD_PRIMS` order.
    """
    path_shape_map = robot["path_shape_map"]
    radii = []
    half_heights = []
    for prim in PAD_PRIMS:
        radius, half_height, _ = builder.shape_scale[path_shape_map[prim]]
        radii.append(float(radius))
        half_heights.append(float(half_height))
    return radii, half_heights


def read_pad_transforms(builder, robot) -> list[wp.transform]:
    """Placement transform of each surface-gripper pad, in the flange (J6_link) frame, read from the arm USD.
    Args:
        builder: the ``ModelBuilder`` the arm USD was loaded into.
        robot: the dict returned by ``add_usd`` for the arm; its ``path_shape_map`` maps prim path -> shape id.

    Returns:
        One ``wp.transform`` per pad, in the order of :data:`PAD_PRIMS`.
    """
    path_shape_map = robot["path_shape_map"]
    transforms = []
    for prim in PAD_PRIMS:
        transforms.append(builder.shape_transform[path_shape_map[prim]])
    return transforms


def _box_half_extents(pick, body_prim) -> tuple[float, float, float]:
    """(hx, hy, hz) half-extents [m] of a pick box's collider (the ``/collision`` child of ``body_prim``)."""
    s = pick["path_shape_scale"][body_prim + "/collision"]
    return (float(s[0]), float(s[1]), float(s[2]))


def read_panel_box(pick) -> tuple[int, tuple[float, float, float]]:
    """Panel's env-local body id and half-extents, read from the loaded pick-scene USD.

    Args:
        pick: the dict returned by ``add_usd`` for the pick scene (``path_body_map`` -> body id,
            ``path_shape_scale`` -> collider half-extents).

    Returns:
        ``(body, half_extents)`` -- the panel's env-local body id and its ``(hx, hy, hz)`` half-extents [m].
    """
    return pick["path_body_map"][PANEL_PRIM], _box_half_extents(pick, PANEL_PRIM)


def read_crate_boxes(pick) -> tuple[list[int], list[tuple[float, float, float]]]:
    """Crates' env-local body ids and half-extents, read from the loaded pick-scene USD.

    Args:
        pick: the dict returned by ``add_usd`` for the pick scene (``path_body_map`` -> body id,
            ``path_shape_scale`` -> collider half-extents).

    Returns:
        ``(bodies, half_extents)`` -- two parallel lists in pick order (:data:`CRATE_PRIMS`): each crate's
        env-local body id, and its ``(hx, hy, hz)`` half-extents [m].
    """
    bodies = []
    half_extents = []
    for prim in CRATE_PRIMS:
        bodies.append(pick["path_body_map"][prim])
        half_extents.append(_box_half_extents(pick, prim))
    return bodies, half_extents


def filter_pick_boxes_against_arm(builder, pick, ee_body_id) -> None:
    """Disable collision between every pick box and every robot-arm link (bodies 0..``ee_body_id``).

    With the pad-box rigid contact off, the soft seal alone holds the box, so the box colliders must not
    also collide with the arm/tool. Adds a collision filter pair for each (box collider, arm-link shape).

    Args:
        builder: the ``ModelBuilder`` the arm + pick scene were loaded into.
        pick: the dict returned by ``add_usd`` for the pick scene (``path_shape_map`` -> shape id).
        ee_body_id: env-local body id of the last arm link (the flange); arm links are bodies 0..ee_body_id.
    """
    psm = pick["path_shape_map"]  # collision prim path -> env-local shape id
    box_shapes = [psm[PANEL_PRIM + "/collision"]]
    for prim in CRATE_PRIMS:
        box_shapes.append(psm[prim + "/collision"])
    for shape in range(len(builder.shape_body)):
        if 0 <= builder.shape_body[shape] <= ee_body_id:  # any robot-arm link (base..gripper)
            for bs in box_shapes:
                builder.add_shape_collision_filter_pair(bs, shape)



@wp.kernel
def broadcast_arm_targets_kernel(
    num_arm_dofs: int,
    arm_offset: int,    # start of world 0's arm DOFs in joint_target_q
    world_stride: int,  # per-world stride in joint_target_q
    joint_target_q: wp.array[float],
):
    """Copy world 0's recorded arm drive targets into every other world's arm DOFs (the worlds are
    identical replicas, so they all follow the same recording). One thread per (world >= 1, arm DOF)."""
    tid = wp.tid()  # dim = (num_worlds - 1) * num_arm_dofs
    w = tid // num_arm_dofs + 1  # target worlds 1..num_worlds-1 (world 0 is the source)
    d = tid % num_arm_dofs
    joint_target_q[arm_offset + w * world_stride + d] = joint_target_q[arm_offset + d]


@wp.kernel
def teleport_crate_kernel(
    gripper_curr_box_prev: wp.array[int],   # [grippers] box index before this sub-step
    gripper_curr_box_curr: wp.array[int],   # [grippers] box index after this sub-step
    n_crates_per_world: int,
    crate_joint_q_start: wp.array[int],     # [n_grippers * n_crates_per_world] joint_q start per gripper per crate
    crate_joint_qd_start: wp.array[int],    # [n_grippers * n_crates_per_world] joint_qd start per gripper per crate
    crate_grip_q: wp.array[float],          # [n_grippers * n_crates_per_world * 7] grip-pose DOFs (pos+quat xyzw)
    joint_q: wp.array[float],               # in/out: simulation joint positions
    joint_qd: wp.array[float],              # in/out: simulation joint velocities
):
    """One thread per gripper. When curr_box advances and the new box is a crate (index >= 1),
    teleport it to its grip pose by writing directly into the free-joint DOFs."""
    g = wp.tid()
    prev_box = gripper_curr_box_prev[g]
    curr_box = gripper_curr_box_curr[g]
    if curr_box == prev_box or curr_box == 0:
        return  # no advance, or wrapped to panel (panel is never teleported)
    box_idx = g * n_crates_per_world + curr_box - 1  # curr_box >= 1; subtract 1 since panel has no slot
    qs = crate_joint_q_start[box_idx]
    qds = crate_joint_qd_start[box_idx]
    grip_start = box_idx * 7
    for i in range(7):
        joint_q[qs + i] = crate_grip_q[grip_start + i]
    for i in range(6):
        joint_qd[qds + i] = 0.0


def teleport_crate(
    example_state_prev: "ExampleState",
    example_state_curr: "ExampleState",
    crate_joint_q_start_wp: wp.array,
    crate_joint_qd_start_wp: wp.array,
    crate_grip_q_wp: wp.array,
    state,
) -> None:
    """Teleport the next crate to its grip pose when curr_box advances (engagement fell this sub-step).

    Compares ``example_state_prev.gripper_curr_box_wp`` with ``example_state_curr.gripper_curr_box_wp``;
    when the index advanced and the new box is a crate (index >= 1), writes its grip-pose DOFs into
    ``state.joint_q`` / ``state.joint_qd`` directly on device (graph-capturable).
    """
    n_grippers = example_state_curr.gripper_curr_box_wp.shape[0]
    if n_grippers == 0:
        return
    n_crates_per_world = crate_joint_q_start_wp.shape[0] // n_grippers
    wp.launch(
        teleport_crate_kernel,
        dim=n_grippers,
        inputs=[
            example_state_prev.gripper_curr_box_wp,
            example_state_curr.gripper_curr_box_wp,
            n_crates_per_world,
            crate_joint_q_start_wp,
            crate_joint_qd_start_wp,
            crate_grip_q_wp,
            state.joint_q,
            state.joint_qd,
        ],
    )


@wp.kernel
def update_engagement_signals_kernel(
    gripper_engaged_curr: wp.array[wp.bool],     # [grippers] current step's engaged command
    gripper_engaged_prev: wp.array[wp.bool],     # [grippers] previous step's engaged command
    gripper_preparing_curr: wp.array[wp.bool],   # [grippers] current step's preparing flag
    gripper_preparing_prev: wp.array[wp.bool],   # [grippers] previous step's preparing flag
    gripper_curr_box_prev: wp.array[int],        # [grippers] previous box index (read)
    gripper_curr_box_curr: wp.array[int],        # [grippers] current box index (written: carry or advance)
    gripper_box_body_ids: wp.array[int],         # [n_grippers * n_boxes_per_world] body ids in pick order per gripper
    gripper_box_shape_ids: wp.array[int],        # [n_grippers * n_boxes_per_world] shape ids in pick order per gripper
    n_boxes_per_world: int,
    pad_gripper: wp.array[int],                  # [pads] pad -> gripper
    pad_offsets: wp.array[int],                  # [grippers+1] first pad index per gripper
    # _prev state: read for carry-forward
    pad_engaged_bs_prev: wp.array[wp.vec2i],
    pad_preparing_bs_prev: wp.array[wp.vec2i],
    # _curr state: written
    pad_engaged_bs_curr: wp.array[wp.vec2i],
    pad_preparing_bs_curr: wp.array[wp.vec2i],
):
    """One thread per pad. Fans per-gripper engaged/preparing signals out to per-pad state, with
    rising-edge latching, sustained carry-forward, and falling-edge clearing. The first pad of each
    gripper carries or advances gripper_curr_box_curr from gripper_curr_box_prev."""
    pad = wp.tid()
    g = pad_gripper[pad]

    eng_curr = gripper_engaged_curr[g]
    eng_prev = gripper_engaged_prev[g]
    prep_curr = gripper_preparing_curr[g]
    prep_prev = gripper_preparing_prev[g]
    curr_box = gripper_curr_box_prev[g]
    box_slot = g * n_boxes_per_world + curr_box
    body_id = gripper_box_body_ids[box_slot]
    shape_id = gripper_box_shape_ids[box_slot]

    if eng_curr and not eng_prev:       # rising edge: latch
        pad_engaged_bs_curr[pad] = wp.vec2i(body_id, shape_id)
    elif eng_curr:                       # sustained: carry from previous sub-step
        pad_engaged_bs_curr[pad] = pad_engaged_bs_prev[pad]
    else:                                # command off: clear
        pad_engaged_bs_curr[pad] = wp.vec2i(-1, -1)

    if prep_curr and not prep_prev:     # rising edge: latch
        pad_preparing_bs_curr[pad] = wp.vec2i(body_id, shape_id)
    elif prep_curr:                      # sustained: carry
        pad_preparing_bs_curr[pad] = pad_preparing_bs_prev[pad]
    else:                                # off: clear
        pad_preparing_bs_curr[pad] = wp.vec2i(-1, -1)

    # Carry or advance the box index. Only the first pad of each gripper writes.
    if pad == pad_offsets[g]:
        if not eng_curr and eng_prev:   # falling edge: advance to next box
            gripper_curr_box_curr[g] = (curr_box + 1) % n_boxes_per_world
        else:                           # otherwise: carry
            gripper_curr_box_curr[g] = curr_box


def update_engagement_signals(
    example_state_prev: "ExampleState",
    example_state_curr: "ExampleState",
    gripper_box_body_ids: wp.array,
    gripper_box_shape_ids: wp.array,
    n_boxes_per_world: int,
    gripper_model,
    pad_offsets: wp.array,
    gripper_state_input_prev,
    gripper_state_input_curr,
) -> None:
    """Fan per-gripper engaged/preparing signals out to per-pad state with edge detection and box advance.

    Reads the current and previous per-gripper recording signals from ``example_state_curr/prev``,
    applies rising-edge latching, sustained carry-forward, and falling-edge clearing to the per-pad
    input state arrays (both body-ID and shape-ID), and advances
    ``example_state_curr.gripper_curr_box_wp`` on the falling edge of engage.
    """
    n_pads = gripper_model.pad_xform.shape[0]
    if n_pads == 0:
        return
    wp.launch(
        update_engagement_signals_kernel,
        dim=n_pads,
        inputs=[
            example_state_curr.gripper_command_engaged_wp,
            example_state_prev.gripper_command_engaged_wp,
            example_state_curr.gripper_command_preparing_wp,
            example_state_prev.gripper_command_preparing_wp,
            example_state_prev.gripper_curr_box_wp,
            example_state_curr.gripper_curr_box_wp,
            gripper_box_body_ids,
            gripper_box_shape_ids,
            n_boxes_per_world,
            gripper_model.pad_gripper,
            pad_offsets,
            gripper_state_input_prev.pad_engaged_bs,
            gripper_state_input_prev.pad_preparing_bs,
            gripper_state_input_curr.pad_engaged_bs,
            gripper_state_input_curr.pad_preparing_bs,
        ],
    )


class ExampleState:
    """Per-gripper recording signals and counters, double-buffered as example_state_prev / example_state_curr."""

    def __init__(self, n_grippers: int, n_pads: int):
        self.gripper_command_engaged_wp = wp.zeros(n_grippers, dtype=wp.bool)    # engagement command (ro[0])
        self.gripper_command_preparing_wp = wp.zeros(n_grippers, dtype=wp.bool)  # preparing-to-engage flag (ro[2])
        self.gripper_curr_box_wp = wp.zeros(n_grippers, dtype=wp.int32)          # current box index
        self.pad_seal_break_count_wp = wp.zeros(n_pads, dtype=wp.int32)          # consecutive over-threshold sub-steps per pad


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
        self.sim_step_count_wp = wp.zeros(1, dtype=wp.int32)
        self.last_lo_wp = wp.zeros(1, dtype=wp.int32)

        # RECORDING_JSONL contains time-stamped joint drive target positions and pad engagement
        # states. Load and extract the time-stamps, the joint drive target positions and the
        # pad engagement states.
        # Apply gaussian smoothing to the raw drive target after loading.
        self.robot_arm_playback = RobotPlayback(RECORDING_JSONL, SMOOTHING_SIGMA, NUM_ARM_DOFS)

        # Build ONE environment (arm with pad cylinders + pick boxes + pallets), then replicate it across
        # NUM_WORLDS worlds. The copies overlap at the origin (Newton's broad phase never collides across
        # worlds), and the ground plane is added globally (world -1) so a single floor is shared by all.
        env = newton.ModelBuilder()

        # Load the robot arm.
        robot = env.add_usd(str(ROBOT_USD), floating=False, collapse_fixed_joints=True)
        ee_body_local = env.body_count - 1  # last arm link (J6_link), the flange, within one env
        # The pad geometry lives in the robot arm USD: four visual cylinders under J6_link/GripperPads (see the
        # asset). Read each pad's placement transform and its radius/half-height.
        pad_transforms = read_pad_transforms(env, robot)
        self.pad_radii, self.pad_half_heights = read_pad_dimensions(env, robot)  # per-pad, in PAD_PRIMS order

        # Load the pick scene (pallets + panel + crates at their waiting poses).
        pick = env.add_usd(str(PICK_SCENE_USD))

        # Get the local id of the panel body and the local ids of the crate bodies.
        # These ids allow quick retrieval of panel and crate attributes after replicating 
        # the simulation configuration to multiple worlds.
        panel_body_local_id, panel_half_extents = read_panel_box(pick)
        crate_body_local_ids, crate_half_extents = read_crate_boxes(pick)


        # Get the half-extents of the panel and crates.
        # The half-extents are used later to create sdf meshes of the pick objects.
        env_box_half_extents = {panel_body_local_id: panel_half_extents}
        for i in range(len(crate_body_local_ids)):
            crate_body_id = crate_body_local_ids[i]
            crate_extent = crate_half_extents[i]
            env_box_half_extents[crate_body_id] = crate_extent

        # Get the mass/inertia of panel and crates.
        # The masses/inertias are used translate from natural frequency/damping ratio
        # to stiffness/damping and vice versa.
        self.panel_mass = float(env.body_mass[panel_body_local_id])
        self.panel_inertia = env.body_inertia[panel_body_local_id]
        self.crate_masses = []
        self.crate_inertias = []
        for i in range(len(crate_body_local_ids)):
            crate_body_id = crate_body_local_ids[i]
            self.crate_masses.append(float(env.body_mass[crate_body_id]))
            self.crate_inertias.append(env.body_inertia[crate_body_id])

        # Create sdf meshes of the panel and crates.
        # sdf meshes are used to help compute the relative pose of pad and 
        # picked object that minimises the distance between the two bodies.
        device = wp.get_device()
        self.sdf_meshes = {}  # env-local box body id -> its SDF mesh
        for lb, he in env_box_half_extents.items():
            self.sdf_meshes[lb] = box_sdf_mesh(*he, device=device)

        # Filter every pick box against the whole arm: the seal owns the hold.
        filter_pick_boxes_against_arm(env, pick, ee_body_local)

        # Create multiple identical worlds of the simulation configuration 
        # and add a shared ground plane.
        builder = newton.ModelBuilder()
        builder.add_ground_plane()  # global (world -1): adds no body, so per-world bodies stay contiguous
        for _ in range(NUM_WORLDS):
            builder.add_world(env)
        self.model = builder.finalize(device=device)  # same device the SDF meshes were built on

        # Every world is an identical copy of the world loaded from usd.
        # As a consequence, there is no need to have unique sdf meshes for each world.
        # Build a map from env-local body id to env-local collision shape id for each pick box,
        # so that the shape-indexed arrays can be filled correctly.
        env_body_to_shape_id = {}
        env_body_to_shape_id[panel_body_local_id] = pick["path_shape_map"][PANEL_PRIM + "/collision"]
        for i in range(len(crate_body_local_ids)):
            crate_prim = CRATE_PRIMS[i]
            env_body_to_shape_id[crate_body_local_ids[i]] = pick["path_shape_map"][crate_prim + "/collision"]

        # shape_mesh_id_wp[shape_id] is the SDF mesh id for that collision shape (0 for non-pick shapes).
        # model.shape_transform[shape_id] gives T_bs (mesh-in-body transform), replacing body_mesh_xform_wp.
        # For our axis-aligned boxes the mesh origin coincides with the body origin, so T_bs is identity
        # and model.shape_transform already holds the correct value from the builder.
        env_shape_count = len(env.shape_body)  # per-world shape stride
        shape_mesh_id = np.zeros(self.model.shape_count, dtype=np.uint64)
        for w in range(NUM_WORLDS):
            for lb, mesh in self.sdf_meshes.items():
                global_shape_id = w * env_shape_count + env_body_to_shape_id[lb]
                shape_mesh_id[global_shape_id] = mesh.id
        self.shape_mesh_id_wp = wp.array(shape_mesh_id, dtype=wp.uint64, device=self.model.device)

        # Note: Newton's collision pipeline is used in this example so set use_mujoco_contacts=False
        self.solver = newton.solvers.SolverMuJoCo(
            self.model, nconmax=256 * NUM_WORLDS, njmax=2048 * NUM_WORLDS, iterations=10, use_mujoco_contacts=False
        )
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.model.contacts()

        # One surface gripper per world, attached to the end effector body. Pads placed at the
        # transforms read from the arm USD. The seal's stiffness/damping are set once, from the first crate's
        # mass/inertia as the design reference -- the SAME k/d then apply to the panel and every crate. (The
        # natural frequency/damping ratio those k/d yield vary with the gripped box's mass; picked_box_seal_modes
        # reports that in the GUI, but the seal parameters do not change.)
        gripper_builder = SurfaceGripperBuilder()
        for w in range(NUM_WORLDS):
            gripper = SurfaceGripper(
                w * env.body_count + ee_body_local, wp.transform_identity(), world=w, n_lip_samples=PAD_LIP_SAMPLES
            )
            gripper.set_natural_frequency_damping_ratio(
                self.crate_masses[0],
                self.crate_inertias[0],
                F_GRIP_MAX,
                NORMAL_MODE,
                SHEAR_X_MODE,
                SHEAR_Y_MODE,
                PEEL_X_MODE,
                PEEL_Y_MODE,
                TWIST_MODE,
            )
            for i in range(len(pad_transforms)):
                gripper.add_pad(pad_transforms[i], self.pad_radii[i], self.pad_half_heights[i])
            gripper_builder.add_gripper(gripper)


        # Create gripper model, state and control.
        # These classes mirror Newton's model, state and control classes.
        self.gripper_model = gripper_builder.finalize(device=self.model.device)
        self.gripper_state_input_prev = self.gripper_model.state_input()
        self.gripper_state_input_curr = self.gripper_model.state_input()
        self.gripper_state_output = self.gripper_model.state_output()
        self.gripper_control = self.gripper_model.control()
        self.gripper_control.pad_grip_control.fill_(1.0)  # full grip command

        # Create ExampleState instances.
        n_grippers = self.gripper_model.gripper_body_id.shape[0]
        n_pads = self.gripper_model.pad_xform.shape[0]
        self.example_state_prev = ExampleState(n_grippers, n_pads)
        self.example_state_curr = ExampleState(n_grippers, n_pads)

        # Boxes (panel or crates) are picked in strict order.
        # For each world, compute the global body ids and global shape ids of the boxes.
        n_boxes_per_world = 1 + len(CRATE_PRIMS)
        self.n_boxes_per_world = n_boxes_per_world
        box_body_ids = []
        box_shape_ids = []
        for w in range(NUM_WORLDS):
            box_body_ids.append(w * env.body_count + panel_body_local_id)
            box_shape_ids.append(w * env_shape_count + env_body_to_shape_id[panel_body_local_id])
            for i in range(len(crate_body_local_ids)):
                box_body_ids.append(w * env.body_count + crate_body_local_ids[i])
                box_shape_ids.append(w * env_shape_count + env_body_to_shape_id[crate_body_local_ids[i]])
        self.gripper_box_body_ids_wp = wp.array(box_body_ids, dtype=wp.int32, device=self.model.device)
        # Parallel to gripper_box_body_ids_wp: global shape ID of each box's collision shape.
        self.gripper_box_shape_ids_wp = wp.array(box_shape_ids, dtype=wp.int32, device=self.model.device)

        # Each gripper g owns pads [g*npads : (g+1)*npads]
        pad_offsets = []
        for g in range(n_grippers + 1):
            pad_offsets.append(g * len(PAD_PRIMS))
        self.pad_offsets_wp = wp.array(pad_offsets, dtype=wp.int32)

        # The crates are teleported from their waiting pose to their grip pose.
        # This requires knowledge of the indices of the array elements in 
        # state.joint_q used to pose each crate.
        # Use ArticulationView to compute the array elements in state.joint_q 
        # that correspond to each crate in each world.
        n_crates = len(CRATE_PRIMS)
        crate_joint_q_start = np.zeros((NUM_WORLDS, n_crates), dtype=np.int32)
        crate_joint_qd_start = np.zeros((NUM_WORLDS, n_crates), dtype=np.int32)
        crate_grip_q = np.zeros((NUM_WORLDS, n_crates, 7), dtype=np.float32)
        for crate_index in range(n_crates):
            crate_name = CRATE_PRIMS[crate_index].split("/")[-1]  # e.g. "crate_0"
            crate_view = ArticulationView(self.model, pattern=f"*{crate_name}*")
            coord_layout = crate_view.frequency_layouts[newton.Model.AttributeFrequency.JOINT_COORD]
            dof_layout = crate_view.frequency_layouts[newton.Model.AttributeFrequency.JOINT_DOF]
            grip = CRATE_GRIP_POSES[crate_index]
            pos = wp.transform_get_translation(grip)
            quat = wp.transform_get_rotation(grip)
            for w in range(NUM_WORLDS):
                crate_joint_q_start[w, crate_index] = coord_layout.offset + w * coord_layout.stride_between_worlds
                crate_joint_qd_start[w, crate_index] = dof_layout.offset + w * dof_layout.stride_between_worlds
                crate_grip_q[w, crate_index] = [pos[0], pos[1], pos[2], quat[0], quat[1], quat[2], quat[3]]
        self.crate_joint_q_start_wp = wp.array(crate_joint_q_start.flatten(), dtype=wp.int32, device=self.model.device)
        self.crate_joint_qd_start_wp = wp.array(crate_joint_qd_start.flatten(), dtype=wp.int32, device=self.model.device)
        self.crate_grip_q_wp = wp.array(crate_grip_q.flatten(), dtype=wp.float32, device=self.model.device)

        # Cache the per-world offset and stride in joint_target_q for the arm, used by
        # broadcast_arm_targets_kernel. ArticulationView finds the arm by its USD root prim label.
        arm_view = ArticulationView(self.model, pattern=ROBOT_ARTICULATION_PATTERN)
        arm_coord_layout = arm_view.frequency_layouts[newton.Model.AttributeFrequency.JOINT_COORD]
        self.arm_target_offset = arm_coord_layout.offset
        self.arm_target_stride = arm_coord_layout.stride_between_worlds

        # Start each world's arm at the first recorded pose.
        initial_arm_q = self.robot_arm_playback.rec_targets_wp.numpy()[0]  # drive target at t=0, the start pose
        joint_q = self.state_0.joint_q.numpy()
        for w in range(NUM_WORLDS):
            start = self.arm_target_offset + w * self.arm_target_stride
            joint_q[start : start + NUM_ARM_DOFS] = initial_arm_q
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
            # (gripper_command_engaged_curr_wp) at the current sim time (sim_step_count_wp*sim_dt)
            # Advance the search index (last_lo_wp) through the recording,
            # and advance sim time (sim_step_count_wp) for the next sub-step.
            self.robot_arm_playback.step(
                self.sim_step_count_wp,  # in/out: read as the current time, then advanced in place
                self.last_lo_wp,  # in/out: forward-search index, resumed and cached
                self.sim_dt,
                self.control.joint_target_q,
                self.example_state_curr.gripper_command_engaged_wp,   # out: engagement command (ro[0]) for world 0
                self.example_state_curr.gripper_command_preparing_wp,  # out: preparing flag (ro[2]) for world 0
            )
            # Fan world 0's arm targets out to the other worlds.
            if NUM_WORLDS > 1:
                wp.launch(
                    broadcast_arm_targets_kernel,
                    dim=(NUM_WORLDS - 1) * NUM_ARM_DOFS,
                    inputs=[NUM_ARM_DOFS, self.arm_target_offset, self.arm_target_stride, self.control.joint_target_q],
                )
            # Fan per-gripper engaged/preparing signals to all pads, with edge detection and box advance.
            update_engagement_signals(
                self.example_state_prev,
                self.example_state_curr,
                self.gripper_box_body_ids_wp,
                self.gripper_box_shape_ids_wp,
                self.n_boxes_per_world,
                self.gripper_model,
                self.pad_offsets_wp,
                self.gripper_state_input_prev,
                self.gripper_state_input_curr,
            )
            # Teleport the next crate to its grip pose when curr_box advances (engagement fell).
            teleport_crate(
                self.example_state_prev,
                self.example_state_curr,
                self.crate_joint_q_start_wp,
                self.crate_joint_qd_start_wp,
                self.crate_grip_q_wp,
                self.state_0,
            )
            self.state_0.clear_forces()  # zero body_f each sub-step (the surface gripper accumulates into it)

            # Break the gripper (and per pad) seal based on pad_break_metric and a threshold time for
            # pad_break_metric being continuously True.
            wp.launch(
                update_seal_break_kernel,
                dim=self.example_state_curr.gripper_command_engaged_wp.shape[0],  # one thread per gripper
                inputs=[
                    self.gripper_state_output.pad_break_metric,
                    self.gripper_state_input_prev.pad_engaged_bs,
                    float(BREAK_THRESHOLD),
                    int(self.break_hold_steps),
                    self.pad_offsets_wp,
                    self.example_state_prev.pad_seal_break_count_wp,
                    self.example_state_curr.pad_seal_break_count_wp,
                ],
                outputs=[self.gripper_state_input_curr.pad_engaged_bs],
            )
            # On each pad's rising edge cache pad_anchor_b (the seal frame in the gripped body).
            # SEAL_SEAT_ON_ENGAGE: anchor to the on-device fitted (seated) box pose instead of the actual
            # one, so the seal seats the box on the pads. Fully kernel-driven, so it graph-captures.
            if SEAL_SEAT_ON_ENGAGE:
                attach_seal_seated(
                    self.model,
                    self.state_0,
                    self.gripper_model,
                    self.gripper_state_input_prev,
                    self.gripper_state_output,
                    self.gripper_state_input_curr,
                    self.shape_mesh_id_wp,
                    iters=SEAT_ITERS,
                )
            else:
                attach_seal(
                    self.state_0,
                    self.gripper_model,
                    self.gripper_state_input_prev,
                    self.gripper_state_output,
                    self.gripper_state_input_curr,
                )
            evaluate_gripper_force(
                self.model, self.state_0, self.gripper_model, self.gripper_state_input_prev, self.gripper_state_output, self.gripper_control, self.sim_dt
            )

            self.model.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)

            self.state_0, self.state_1 = self.state_1, self.state_0
            self.gripper_state_input_prev, self.gripper_state_input_curr = self.gripper_state_input_curr, self.gripper_state_input_prev
            self.example_state_prev, self.example_state_curr = self.example_state_curr, self.example_state_prev

        # Per-pad seal-quality metric (each pad's RMS lip-gap deviation from its seated pose), once per frame
        # on the final state. Part of the captured graph; the GUI reads it back. Engaged pads use the cached
        # sdf0; pads whose gripper is preparing recompute the seated pose + sdf0 live -- both in one call.
        evaluate_seal_quality(
            self.model,
            self.state_0,
            self.gripper_model,
            self.gripper_state_input_prev,
            self.gripper_state_output,
            self.shape_mesh_id_wp,
            self.gripper_state_output.pad_seal_quality_rms,
            iters=SEAT_ITERS,
        )

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
        self.viewer.begin_frame(sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()

    def gui(self, ui):
        # commanded grip (recorded ro[0], sampled per sub-step by sample_playback_kernel) vs the
        # actual latched seal (command AND-ed with the break/proximity logic in update_seal_break_kernel /
        # attach_seal) -- the two differ if a seal fractured or failed to grab.
        commanded = bool(self.example_state_prev.gripper_command_engaged_wp.numpy()[0])
        preparing = bool(self.example_state_prev.gripper_command_preparing_wp.numpy()[0])
        held = int((self.gripper_state_input_prev.pad_engaged_bs.numpy()[:, 0] >= 0).sum())
        ui.text(f"Grip cmd:  {'On' if commanded else 'Off'}  (recording)")
        ui.text(f"Preparing: {'On' if preparing else 'Off'}  (lead-in before engage)")
        ui.text(f"Seal engaged: {held}/{len(PAD_PRIMS)} pads  (actual)")
        # Seal quality, per pad: each pad's RMS deviation of its current lip signed distances from their
        # seated (engagement) values [mm]. 0 = that pad holds the box at its seated pose; grows as the box
        # shifts. A pad that is neither gripping nor preparing reads -1 (shown as "--"). World 0's pads shown.
        pad_rms_mm = self.gripper_state_output.pad_seal_quality_rms.numpy()[: len(PAD_PRIMS)] * 1000.0
        parts = []
        for i in range(len(pad_rms_mm)):
            if pad_rms_mm[i] < 0.0:
                parts.append("--")
            else:
                parts.append(f"{pad_rms_mm[i]:.3f}")
        ui.text(f"RMS lip gap [mm]: {', '.join(parts)}")
        # Pick-box masses read from the USD; the seal is tuned against the first crate.
        ui.text(f"Pick boxes: panel {self.panel_mass:.0f} kg, crate[0] {self.crate_masses[0]:.0f} kg")
        # Seal modes for the box currently gripped: same tool k/d, but natural frequency and damping
        # ratio depend on that box's mass/inertia and its pose relative to the seal. Zero when nothing
        # is gripped (no pad engaged).
        ui.text("Picked-box seal modes:")
        modes = picked_box_seal_modes(
            self.gripper_model, self.state_0, self.model, int(self.gripper_state_input_prev.pad_preparing_bs.numpy()[0, 0])
        )
        if held == 0:
            zeroed_modes = []
            for name, _, _ in modes:
                zeroed_modes.append((name, 0.0, 0.0))
            modes = zeroed_modes
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
