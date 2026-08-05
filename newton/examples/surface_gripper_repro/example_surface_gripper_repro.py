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

# Asset paths (global constants). All assets live in the Assets/ directory alongside this example.
ASSETS = Path(__file__).parent / "Assets"
# robot arm USD
ROBOT_USD = ASSETS / "fanuc_arm_flattened_collision.usda"
# Pick scene (2 static pallets + panel + 6 crates as box colliders at their waiting poses), baked from
# the deterministic FK placements by bake_pick_scene.py. Re-bake only if the arm USD or recording changes.
PICK_SCENE_USD = ASSETS / "pick_scene.usda"
# recording of the robot arm motion and surface gripper engagement/disengagement.
RECORDING_JSONL = ASSETS / "robot_recording_truncated.jsonl"

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

# Draw a small non-colliding sphere at each pad-lip sample point (the points attach_seal_seated samples
# the gripped object's SDF at during the seat fit), so the seat-fit probe geometry is visible in the
# viewer. Sized to the pad half-height (read from the USD). Purely visual (collision off); no physics.
SHOW_LIP_POINTS = True

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


def add_lip_point_markers(builder, ee_body_id, pad_transforms, pad_radii, pad_half_heights) -> None:
    """Add a small non-colliding sphere at each pad-lip sample point, for viewer visibility.

    Places :data:`PAD_LIP_SAMPLES` markers around each pad's lip -- the circle of that pad's radius on the
    pad's bottom face (+that pad's half-height along the grip axis), in the pad frame -- matching the points
    :func:`attach_seal_seated` samples the gripped object's SDF at. Rigidly fixed to the flange (body
    ``ee_body_id``); purely visual (collision off).

    Args:
        builder: the ``ModelBuilder`` the arm was loaded into.
        ee_body_id: env-local body id of the flange (the pads' parent).
        pad_transforms: each pad's placement transform in the flange frame (see :func:`read_pad_transforms`).
        pad_radii: per-pad radius [m], one per pad (parallel to ``pad_transforms``).
        pad_half_heights: per-pad half-height [m] (also the marker sphere radius), one per pad.
    """
    lip_cfg = builder.default_shape_cfg.copy()
    lip_cfg.density = 0.0
    lip_cfg.has_shape_collision = False
    for i in range(len(pad_transforms)):
        pad_tf = pad_transforms[i]
        pad_radius = pad_radii[i]
        pad_half_height = pad_half_heights[i]
        for s in range(PAD_LIP_SAMPLES):
            th = 2.0 * np.pi * s / PAD_LIP_SAMPLES
            lip_local = wp.vec3(pad_radius * np.cos(th), pad_radius * np.sin(th), pad_half_height)
            builder.add_shape_sphere(
                ee_body_id,
                xform=wp.transform(wp.transform_point(pad_tf, lip_local), wp.quat_identity()),
                radius=pad_half_height,
                cfg=lip_cfg,
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
        self.sim_step_count_wp = wp.zeros(1, dtype=wp.int32)
        self.last_lo_wp = wp.zeros(1, dtype=wp.int32)
        # The per-gripper / per-pad runtime arrays are sized from the finalized gripper model further down
        # (one gripper per world, so they scale with NUM_WORLDS).

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
        panel_body_local_id, panel_half_extents = read_panel_box(pick)
        crate_body_local_ids, crate_half_extents = read_crate_boxes(pick)
        # Half-extents of every pick box, keyed by env-local body id (for the SDF seat meshes): the panel,
        # then every crate.
        env_box_half_extents = {panel_body_local_id: panel_half_extents}
        for i in range(len(crate_body_local_ids)):
            crate_body_id = crate_body_local_ids[i]
            crate_extent = crate_half_extents[i]
            env_box_half_extents[crate_body_id] = crate_extent

        # Reference mass/inertia of each pick box, read from the loaded USD (both are authored in
        # pick_scene.usda, so Newton uses them verbatim). Both boxes are exposed (the GUI shows each box's
        # mass); the seal is tuned against the crate's (set_natural_frequency_damping_ratio below).
        self.panel_mass = float(env.body_mass[panel_body_local_id])
        self.panel_inertia = env.body_inertia[panel_body_local_id]
        # One mass/inertia per crate (the crates need not be identical).
        self.crate_masses = []
        self.crate_inertias = []
        for i in range(len(crate_body_local_ids)):
            crate_body_id = crate_body_local_ids[i]
            self.crate_masses.append(float(env.body_mass[crate_body_id]))
            self.crate_inertias.append(env.body_inertia[crate_body_id])

        # SDF seat meshes (queries only; the box shape still owns collision). They depend only on the box
        # half-extents, so build them here -- one per distinct box, before the model is finalized -- on the
        # device the model will use. Shared across worlds; body_mesh_id (after finalize) maps each world's box
        # to its mesh. Kept on self so the wp.Mesh objects outlive body_mesh_id's references to their ids.
        device = wp.get_device()
        self.sdf_meshes = {}  # env-local box body id -> its SDF mesh
        for lb, he in env_box_half_extents.items():
            self.sdf_meshes[lb] = box_sdf_mesh(*he, device=device)

        # Filter every pick box against the whole arm: the seal owns the hold.
        if not ENABLE_PAD_BOX_CONTACT:
            filter_pick_boxes_against_arm(env, pick, ee_body_local)

        # Seat-fit sample points: markers at the pad-lip points attach_seal_seated samples the box SDF at.
        # Only meaningful when the seat fit actually runs (SEAL_SEAT_ON_ENGAGE).
        if SHOW_LIP_POINTS and SEAL_SEAT_ON_ENGAGE:
            add_lip_point_markers(env, ee_body_local, pad_transforms, self.pad_radii, self.pad_half_heights)

        # Main scene: shared global ground plane + NUM_WORLDS overlapping copies of the env, each added
        # as its own world (add_world). No spacing -- the copies overlap at the origin (broad phase never
        # collides across worlds), and only world 0 is rendered.
        env_body_count = env.body_count  # bodies per world
        builder = newton.ModelBuilder()
        builder.add_ground_plane()  # global (world -1): adds no body, so per-world bodies stay contiguous
        for _ in range(NUM_WORLDS):
            builder.add_world(env)
        self.model = builder.finalize(device=device)  # same device the SDF meshes were built on

        # add_world appends each env copy contiguously, so env-local body ``local`` in world ``w`` has global
        # id ``w * env_body_count + local``. World 0's global ids therefore equal the env-local ids (e.g. ee_body_local).

        # body_mesh_id maps every world's box body (global id w * env_body_count + lb) to its shared SDF mesh id.
        body_mesh_id = np.zeros(self.model.body_count, dtype=np.uint64)
        for w in range(NUM_WORLDS):
            for lb, mesh in self.sdf_meshes.items():
                body_mesh_id[w * env_body_count + lb] = mesh.id
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

        # One surface gripper per world, attached to the end effector body. Pads placed at the
        # transforms read from the arm USD. The seal's stiffness/damping are set once, from the first crate's
        # mass/inertia as the design reference -- the SAME k/d then apply to the panel and every crate. (The
        # natural frequency/damping ratio those k/d yield vary with the gripped box's mass; picked_box_seal_modes
        # reports that in the GUI, but the seal parameters do not change.)
        gripper_builder = SurfaceGripperBuilder()
        for w in range(NUM_WORLDS):
            gripper = SurfaceGripper(w * env_body_count + ee_body_local, wp.transform_identity(), world=w)
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
            for pad_xform in pad_transforms:
                gripper.add_pad(pad_xform)
            gripper_builder.add_gripper(gripper)
        self.gripper_model = gripper_builder.finalize(device=self.model.device)
        self.gripper_state = self.gripper_model.state()
        self.gripper_control = self.gripper_model.control()
        self.gripper_control.pad_grip_control.fill_(1.0)  # full grip command

        # Per-gripper / per-pad runtime arrays, sized from the gripper model (one gripper per world).
        n_grippers = self.gripper_model.gripper_body_id.shape[0]
        n_pads = self.gripper_model.pad_xform.shape[0]

        # Per-pad lip geometry the seat fit samples, one entry per pad in the gripper model. Every gripper
        # has the same pads (added in PAD_PRIMS order), so pad i's within-gripper index is i % pads_per_gripper.
        pads_per_gripper = len(PAD_PRIMS)
        pad_radius_list = []
        pad_face_offset_list = []
        for p in range(n_pads):
            pad_radius_list.append(self.pad_radii[p % pads_per_gripper])
            pad_face_offset_list.append(self.pad_half_heights[p % pads_per_gripper])
        self.pad_radius_wp = wp.array(pad_radius_list, dtype=float, device=self.model.device)
        self.pad_face_offset_wp = wp.array(pad_face_offset_list, dtype=float, device=self.model.device)
        # gripper_command_engaged_wp is the engagement state of the gripper as read from the recording of the robot arm.
        # gripper_seal_broken_wp is the fracture state of the gripper
        # pad_seal_break_count_wp is the number of continuous steps that each pad has exceeded the maximum force.
        # pad_seal_engaged_wp is the per pad engagement state after accounting for gripper_command_engaged_wp and gripper_seal_broken_wp.
        self.gripper_command_engaged_wp = wp.zeros(n_grippers, dtype=wp.bool)  # [grippers] recorded engagement command
        self.pad_seal_break_count_wp = wp.zeros(n_pads, dtype=wp.int32)  # consecutive over-threshold steps, per pad
        self.gripper_seal_broken_wp = wp.zeros(n_grippers, dtype=wp.bool)  # [grippers] latched fracture
        self.pad_seal_engaged_wp = wp.zeros(n_pads, dtype=wp.bool)  # [pads] per-pad seal command
        # [grippers+1] CSR offsets: gripper g owns pads [g*npads, (g+1)*npads)
        pad_offsets = []
        for g in range(n_grippers + 1):
            pad_offsets.append(g * len(PAD_PRIMS))
        self.pad_offsets = wp.array(pad_offsets, dtype=wp.int32)

        # Each pad starts targeting its own world's panel body (the gripper model groups pads by world).
        pad_world = self.gripper_model.pad_world.numpy()
        pad_body_b = []
        for p in range(n_pads):
            pad_body_b.append(pad_world[p] * env_body_count + panel_body_local_id)  # this pad's world's panel body
        self.pad_body_b = wp.array(np.array(pad_body_b, dtype=np.int32), dtype=wp.int32)

        # One CratePlayback per world (worlds are identical copies, so each moves its own crates on the
        # same disengagement cues). pad_world_start[w] slices the gripper model's pads for world w.
        self.crate_playbacks = []
        for w in range(NUM_WORLDS):
            crate_bodies = []  # this world's crate body ids
            for cb in crate_body_local_ids:
                crate_bodies.append(w * env_body_count + cb)
            self.crate_playbacks.append(
                CratePlayback(self.robot_arm_playback, self.model, crate_bodies, CRATE_GRIP_POSES)
            )
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
            self.accel_recorder = EndEffectorAccelerationRecorder(ee_body_local, self.sim_dt)
            self.drive_target_recorder = DriveTargetRecorder(self.sim_dt, NUM_ARM_DOFS)
            self.pad_break_recorder = PadBreakMetricRecorder(
                self.sim_dt, self.robot_arm_playback.rec_duration, len(PAD_PRIMS)
            )
            # per-pad seal forces over the first lift (simple linear model only; see GripperForceRecorder)
            self.gripper_force_recorder = GripperForceRecorder(self.sim_dt, len(PAD_PRIMS))

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
                    self.pad_radius_wp,
                    self.pad_face_offset_wp,
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
        ui.text(f"Seal engaged: {held}/{len(PAD_PRIMS)} pads  (actual)")
        # Pick-box masses read from the USD; the seal is tuned against the first crate.
        ui.text(f"Pick boxes: panel {self.panel_mass:.0f} kg, crate[0] {self.crate_masses[0]:.0f} kg")
        # Seal modes for the box currently gripped: same tool k/d, but natural frequency and damping
        # ratio depend on that box's mass/inertia and its pose relative to the seal. Zero when nothing
        # is gripped (no pad engaged).
        ui.text("Picked-box seal modes:")
        modes = picked_box_seal_modes(self.gripper_model, self.state_0, self.model, int(self.pad_body_b.numpy()[0]))
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
