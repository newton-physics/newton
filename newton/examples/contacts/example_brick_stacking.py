# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Brick Stacking
#
# Demonstrates a Franka Panda robot picking up bricks from a table
# and stacking them, using SDF-based mesh collision and the MuJoCo solver.
# The arm is controlled with IK and a finite-state machine that sequences
# approach, grasp, lift, move, place and release for each brick.
#
# Command: python -m newton.examples brick_stacking
#
###########################################################################

import enum

import numpy as np
import warp as wp

import newton
import newton.examples
import newton.ik as ik

# Brick dimensions [m]
PITCH = 0.008
BRICK_HEIGHT = 0.0096

BRICK_SCALE = 1.0
BRICK_DENSITY = 565.0  # ABS plastic [kg/m³]


BRICK_MASS = BRICK_DENSITY * (4 * PITCH) * (2 * PITCH) * BRICK_HEIGHT
BRICK_KE = 9.81 * BRICK_MASS / 1.25e-6
BRICK_KD = 2.0 * np.sqrt(BRICK_KE * BRICK_MASS) * 10.0  # 10x critical damping for fast settling
BRICK_MARGIN = 4.0e-5

# SDF mesh parameters
SDF_RESOLUTION = 256
SDF_NARROW_BAND = 0.01
SDF_MARGIN = 0.01

# Gripper finger positions [m]
GRIPPER_OPEN = 0.5 * (2 * PITCH * BRICK_SCALE + 0.004)
GRIPPER_RELEASE = 0.5 * (2 * PITCH * BRICK_SCALE * 2.0)
GRIPPER_CLOSED = 0.5 * (2 * PITCH * BRICK_SCALE - 0.003)


def _build_mesh_with_sdf(verts, faces, color, scale=1.0):
    scaled_verts = verts * scale
    mesh = newton.Mesh(scaled_verts, faces.flatten(), color=color)
    mesh.build_sdf(
        max_resolution=SDF_RESOLUTION,
        narrow_band_range=(-SDF_NARROW_BAND * scale, SDF_NARROW_BAND * scale),
        margin=SDF_MARGIN * scale,
    )
    return mesh


def _cylinder_mesh(radius, height, segments, cx=0.0, cy=0.0, cz=0.0, bottom_cap=True):
    """Cylinder with split vertices at rims for sharp edges.

    Separate vertex rings for side walls and caps prevent normal averaging
    across the sharp rim, giving crisp cylindrical edges.
    """
    n = segments
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    cos_a, sin_a = np.cos(angles), np.sin(angles)

    ring_x = cx + radius * cos_a
    ring_y = cy + radius * sin_a

    verts = []
    faces = []

    # Side wall: own bottom ring [0..n) and top ring [n..2n)
    side_bot = np.column_stack([ring_x, ring_y, np.full(n, cz)]).astype(np.float32)
    side_top = np.column_stack([ring_x, ring_y, np.full(n, cz + height)]).astype(np.float32)
    verts.append(side_bot)
    verts.append(side_top)
    for i in range(n):
        j = (i + 1) % n
        faces.append([i, n + j, n + i])
        faces.append([i, j, n + j])

    # Top cap: separate ring + center vertex
    off_top = 2 * n
    cap_top_ring = np.column_stack([ring_x, ring_y, np.full(n, cz + height)]).astype(np.float32)
    cap_top_center = np.array([[cx, cy, cz + height]], dtype=np.float32)
    verts.append(cap_top_ring)
    verts.append(cap_top_center)
    tc = off_top + n
    for i in range(n):
        j = (i + 1) % n
        faces.append([tc, off_top + i, off_top + j])

    # Bottom cap (optional): separate ring + center vertex
    if bottom_cap:
        off_bot = off_top + n + 1
        cap_bot_ring = np.column_stack([ring_x, ring_y, np.full(n, cz)]).astype(np.float32)
        cap_bot_center = np.array([[cx, cy, cz]], dtype=np.float32)
        verts.append(cap_bot_ring)
        verts.append(cap_bot_center)
        bc = off_bot + n
        for i in range(n):
            j = (i + 1) % n
            faces.append([bc, off_bot + j, off_bot + i])

    return np.vstack(verts), np.array(faces, dtype=np.int32)


def _combine_meshes(mesh_list):
    all_v, all_f, off = [], [], 0
    for v, f in mesh_list:
        all_v.append(v)
        all_f.append(f + off)
        off += len(v)
    return np.vstack(all_v).astype(np.float32), np.vstack(all_f).astype(np.int32)


STUD_RADIUS = 0.0024
STUD_COLLIDER_RADIUS = STUD_RADIUS - 0.0002
STUD_HEIGHT = 0.0017
COLLIDER_INSET = 0.0001
WALL_THICKNESS = 0.0012
TOP_THICKNESS = 0.001
TUBE_OUTER_RADIUS = 0.003255
TUBE_HEIGHT = BRICK_HEIGHT - TOP_THICKNESS
CYLINDER_SEGMENTS = 48


def _make_shell_mesh(nx, ny):
    """Watertight hollow box shell for an *nx* x *ny* brick.

    Origin at the centre-bottom (z=0).  Inner cavity is open at the
    bottom and sealed by a top plate.
    """
    ox = nx * PITCH / 2.0
    oy = ny * PITCH / 2.0
    inx = ox - WALL_THICKNESS
    iny = oy - WALL_THICKNESS
    H = BRICK_HEIGHT
    T = TOP_THICKNESS

    v = np.array(
        [
            [-ox, -oy, 0],
            [+ox, -oy, 0],
            [+ox, +oy, 0],
            [-ox, +oy, 0],
            [-ox, -oy, H],
            [+ox, -oy, H],
            [+ox, +oy, H],
            [-ox, +oy, H],
            [-inx, -iny, 0],
            [+inx, -iny, 0],
            [+inx, +iny, 0],
            [-inx, +iny, 0],
            [-inx, -iny, H - T],
            [+inx, -iny, H - T],
            [+inx, +iny, H - T],
            [-inx, +iny, H - T],
        ],
        dtype=np.float32,
    )
    f = np.array(
        [
            [4, 5, 6],
            [4, 6, 7],
            [0, 1, 5],
            [0, 5, 4],
            [2, 3, 7],
            [2, 7, 6],
            [3, 0, 4],
            [3, 4, 7],
            [1, 2, 6],
            [1, 6, 5],
            [0, 8, 9],
            [0, 9, 1],
            [1, 9, 10],
            [1, 10, 2],
            [2, 10, 11],
            [2, 11, 3],
            [3, 11, 8],
            [3, 8, 0],
            [9, 8, 12],
            [9, 12, 13],
            [11, 10, 14],
            [11, 14, 15],
            [8, 11, 15],
            [8, 15, 12],
            [10, 9, 13],
            [10, 13, 14],
            [12, 15, 14],
            [12, 14, 13],
        ],
        dtype=np.int32,
    )
    return v, f


def _make_brick_mesh(nx=4, ny=2):
    """Full brick mesh (shell + studs + interior tubes) for an *nx* x *ny* brick.

    Each sub-component is a closed surface with consistent outward normals.
    Dimensions follow the standard 8mm pitch system. The mesh is centered
    at the origin in XY with the bottom at Z=0.
    """
    shell_v, shell_f = _make_shell_mesh(nx, ny)
    stud_meshes = []
    for i in range(nx):
        for j in range(ny):
            sx = (i - (nx - 1) / 2.0) * PITCH
            sy = (j - (ny - 1) / 2.0) * PITCH
            stud_meshes.append(
                _cylinder_mesh(
                    STUD_RADIUS, STUD_HEIGHT, CYLINDER_SEGMENTS, cx=sx, cy=sy, cz=BRICK_HEIGHT, bottom_cap=False
                )
            )

    tube_meshes = []
    if ny == 2:
        for i in range(nx - 1):
            tx = (i - (nx - 2) / 2.0) * PITCH
            tube_meshes.append(_cylinder_mesh(TUBE_OUTER_RADIUS, TUBE_HEIGHT, CYLINDER_SEGMENTS, cx=tx, cy=0.0, cz=0.0))

    return _combine_meshes([(shell_v, shell_f), *stud_meshes, *tube_meshes])


class TaskType(enum.IntEnum):
    APPROACH = 0
    REFINE_APPROACH = 1
    GRASP = 2
    LIFT = 3
    MOVE_TO_DROP_OFF = 4
    REFINE_DROP_OFF = 5
    RELEASE = 6
    HOME = 7


@wp.kernel(enable_backward=False)
def set_target_pose_kernel(
    task_schedule: wp.array[wp.int32],
    task_time_limits: wp.array[float],
    task_pick_body: wp.array[int],
    task_drop_body: wp.array[int],
    task_drop_layer: wp.array[int],
    task_idx: wp.array[int],
    task_time_elapsed: wp.array[float],
    task_dt: float,
    offset_approach: wp.vec3,
    offset_lift: wp.vec3,
    grasp_z_offset: wp.vec3,
    drop_z_offset: wp.vec3,
    brick_stack_height: float,
    home_pos: wp.vec3,
    task_init_body_q: wp.array[wp.transform],
    body_q: wp.array[wp.transform],
    ee_index: int,
    # outputs
    ee_pos_target: wp.array[wp.vec3],
    ee_pos_interp: wp.array[wp.vec3],
    ee_rot_target: wp.array[wp.vec4],
    ee_rot_interp: wp.array[wp.vec4],
    gripper_target: wp.array2d[wp.float32],
):
    tid = wp.tid()

    idx = task_idx[tid]
    task = task_schedule[idx]
    time_limit = task_time_limits[idx]
    pick_body = task_pick_body[idx]
    drop_body = task_drop_body[idx]
    drop_layer = task_drop_layer[idx]

    task_time_elapsed[tid] += task_dt
    t_lin = wp.min(1.0, task_time_elapsed[tid] / time_limit)
    # Smoothstep easing
    t = t_lin * t_lin * (3.0 - 2.0 * t_lin)

    ee_pos_prev = wp.transform_get_translation(task_init_body_q[ee_index])
    ee_quat_prev = wp.transform_get_rotation(task_init_body_q[ee_index])
    ee_quat_down = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), wp.pi)

    pick_pos = wp.transform_get_translation(task_init_body_q[pick_body])
    pick_quat = wp.transform_get_rotation(task_init_body_q[pick_body])

    drop_pos = wp.transform_get_translation(task_init_body_q[drop_body])
    drop_quat = wp.transform_get_rotation(task_init_body_q[drop_body])
    layer_offset = wp.float32(drop_layer) * brick_stack_height * wp.vec3(0.0, 0.0, 1.0)
    ee_quat_drop = ee_quat_down * wp.quat_inverse(drop_quat)

    t_gripper = 0.0
    target_pos = home_pos
    target_quat = ee_quat_down

    if task == TaskType.APPROACH.value:
        target_pos = pick_pos + offset_approach
        target_quat = ee_quat_down * wp.quat_inverse(pick_quat)
    elif task == TaskType.REFINE_APPROACH.value:
        target_pos = pick_pos + grasp_z_offset
        target_quat = ee_quat_prev
    elif task == TaskType.GRASP.value:
        target_pos = ee_pos_prev
        target_quat = ee_quat_prev
        t_gripper = t
    elif task == TaskType.LIFT.value:
        target_pos = ee_pos_prev + offset_lift
        target_quat = ee_quat_prev
        t_gripper = 1.0
    elif task == TaskType.MOVE_TO_DROP_OFF.value:
        target_pos = drop_pos + layer_offset + offset_approach
        target_quat = ee_quat_drop
        t_gripper = 1.0
    elif task == TaskType.REFINE_DROP_OFF.value:
        target_pos = drop_pos + layer_offset + grasp_z_offset + drop_z_offset
        target_quat = ee_quat_drop
        t_gripper = 1.0
    elif task == TaskType.RELEASE.value:
        target_pos = drop_pos + layer_offset + grasp_z_offset + drop_z_offset
        target_quat = ee_quat_drop
        t_gripper = 1.0 - t
    elif task == TaskType.HOME.value:
        target_pos = home_pos
        target_quat = ee_quat_down

    ee_pos_target[tid] = target_pos
    interp_pos = ee_pos_prev * (1.0 - t) + target_pos * t

    # XY alignment correction for IK convergence
    ee_pos_actual = wp.transform_get_translation(body_q[ee_index])
    xy_err = wp.vec3(
        interp_pos[0] - ee_pos_actual[0],
        interp_pos[1] - ee_pos_actual[1],
        0.0,
    )
    use_align = 1.0
    if task == TaskType.APPROACH.value or task == TaskType.HOME.value:
        use_align = 0.0
    ee_pos_interp[tid] = interp_pos + use_align * xy_err

    ee_rot_target[tid] = target_quat[:4]
    ee_rot_interp[tid] = wp.quat_slerp(ee_quat_prev, target_quat, t)[:4]

    gripper_open = GRIPPER_OPEN
    if task == TaskType.RELEASE.value or task == TaskType.HOME.value:
        gripper_open = GRIPPER_RELEASE
    gripper_pos = gripper_open * (1.0 - t_gripper) + GRIPPER_CLOSED * t_gripper
    gripper_target[tid, 0] = gripper_pos
    gripper_target[tid, 1] = gripper_pos


@wp.kernel(enable_backward=False)
def advance_task_kernel(
    task_time_limits: wp.array[float],
    ee_pos_target: wp.array[wp.vec3],
    ee_rot_target: wp.array[wp.vec4],
    body_q: wp.array[wp.transform],
    ee_index: int,
    # outputs
    task_idx: wp.array[int],
    task_time_elapsed: wp.array[float],
    task_init_body_q: wp.array[wp.transform],
):
    tid = wp.tid()
    idx = task_idx[tid]
    time_limit = task_time_limits[idx]

    ee_pos_current = wp.transform_get_translation(body_q[ee_index])
    ee_quat_current = wp.transform_get_rotation(body_q[ee_index])

    pos_err = wp.length(ee_pos_target[tid] - ee_pos_current)

    ee_quat_tgt = wp.quaternion(ee_rot_target[tid][:3], ee_rot_target[tid][3])
    quat_rel = ee_quat_current * wp.quat_inverse(ee_quat_tgt)
    rot_err = wp.degrees(2.0 * wp.acos(wp.clamp(wp.abs(quat_rel[3]), 0.0, 1.0)))

    if (
        task_time_elapsed[tid] >= time_limit
        and pos_err < 0.003
        and rot_err < 1.5
        and task_idx[tid] < wp.len(task_time_limits) - 1
    ):
        task_idx[tid] += 1
        task_time_elapsed[tid] = 0.0
        num_bodies = wp.len(body_q)
        for i in range(num_bodies):
            task_init_body_q[i] = body_q[i]


class Example:
    def __init__(self, viewer, args=None):
        self.viewer = viewer
        self.sim_time = 0.0
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = 16
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.ee_index = 11
        self.brick_count = 3

        self.table_height = 0.1
        self.table_pos = wp.vec3(0.0, -0.5, 0.5 * self.table_height)
        self.table_top_center = self.table_pos + wp.vec3(0.0, 0.0, 0.5 * self.table_height)
        self.robot_base_pos = self.table_top_center + wp.vec3(-0.5, 0.0, 0.0)

        self.brick_height_scaled = BRICK_HEIGHT * BRICK_SCALE
        self.brick_width_scaled = 2 * PITCH * BRICK_SCALE
        self.brick_length_scaled = 4 * PITCH * BRICK_SCALE

        # Task offsets (TCP frame) [m]
        self.offset_approach = wp.vec3(0.0, 0.0, 0.025)
        self.offset_lift = wp.vec3(0.0, -0.001, 0.042)
        self.grasp_z_offset = wp.vec3(0.0, 0.0, 0.012)
        self.drop_z_offset = wp.vec3(0.0, 0.0, -0.001)

        # Generate brick mesh procedurally
        self.v_2x4, self.f_2x4 = _make_brick_mesh()

        # Build Franka + table, finalize IK model from default pose
        franka_builder = self.build_franka_with_table()
        self.model_ik = franka_builder.finalize()

        # Record home EE position from the default URDF configuration
        state_tmp = self.model_ik.state()
        newton.eval_fk(self.model_ik, self.model_ik.joint_q, self.model_ik.joint_qd, state_tmp)
        self.home_pos = wp.vec3(*state_tmp.body_q.numpy()[self.ee_index][:3])

        # Solve IK for the approach pose above the red brick so the gripper
        # starts there and the first visible motion is a smooth descent.
        init_joints = self._solve_approach_ik()
        franka_builder.joint_q[:7] = init_joints.tolist()
        franka_builder.joint_q[7] = GRIPPER_OPEN
        franka_builder.joint_q[8] = GRIPPER_OPEN
        franka_builder.joint_target_pos[:9] = franka_builder.joint_q[:9]

        # Build full scene
        scene = newton.ModelBuilder()
        scene.add_builder(franka_builder)
        self.add_bricks(scene)
        scene.add_ground_plane(cfg=newton.ModelBuilder.ShapeConfig(mu=0.75))

        self.model = scene.finalize()

        contact_max = 16384
        self.model.rigid_contact_max = contact_max

        self.collision_pipeline = newton.CollisionPipeline(
            self.model,
            reduce_contacts=True,
            rigid_contact_max=contact_max,
            broad_phase="nxn",
        )

        self.solver = newton.solvers.SolverMuJoCo(
            self.model,
            solver="newton",
            integrator="implicitfast",
            iterations=15,
            ls_iterations=100,
            nconmax=contact_max,
            njmax=contact_max * 2,
            cone="elliptic",
            impratio=50.0,
            use_mujoco_contacts=False,
        )

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.collision_pipeline.contacts()

        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)
        wp.copy(self.control.joint_target_pos[:9], self.model.joint_q[:9])

        self.setup_ik()
        self.setup_tasks()

        self.viewer.set_model(self.model)
        if hasattr(self.viewer, "picking"):
            ps = self.viewer.picking.pick_state.numpy()
            ps[0]["pick_stiffness"] = 0.1
            ps[0]["pick_damping"] = 0.01
            self.viewer.picking.pick_state.assign(ps)

        cam_pos = self.table_top_center + wp.vec3(0.22, -0.18, 0.15)
        self.viewer.set_camera(pos=cam_pos, pitch=-30.0, yaw=135.0)

        self.capture()

    # -- scene construction --------------------------------------------------

    def build_franka_with_table(self):
        builder = newton.ModelBuilder()
        newton.solvers.SolverMuJoCo.register_custom_attributes(builder)

        builder.add_urdf(
            newton.utils.download_asset("franka_emika_panda") / "urdf/fr3_franka_hand.urdf",
            xform=wp.transform(self.robot_base_pos, wp.quat_identity()),
            floating=False,
            enable_self_collisions=False,
            parse_visuals_as_colliders=False,
        )

        builder.joint_q[:9] = [
            -3.6802115e-03,
            2.3901723e-02,
            3.6804110e-03,
            -2.3683236e00,
            -1.2918962e-04,
            2.3922248e00,
            7.8549200e-01,
            GRIPPER_OPEN,
            GRIPPER_OPEN,
        ]
        builder.joint_target_pos[:9] = builder.joint_q[:9]
        builder.joint_target_ke[:9] = [400, 400, 400, 400, 400, 400, 400, 100, 100]
        builder.joint_target_kd[:9] = [40, 40, 40, 40, 40, 40, 40, 10, 10]
        builder.joint_effort_limit[:9] = [87, 87, 87, 87, 12, 12, 12, 100, 100]
        builder.joint_armature[:9] = [0.3] * 4 + [0.11] * 3 + [0.15] * 2

        # Gravity compensation
        gravcomp_attr = builder.custom_attributes["mujoco:jnt_actgravcomp"]
        if gravcomp_attr.values is None:
            gravcomp_attr.values = {}
        for dof_idx in range(7):
            gravcomp_attr.values[dof_idx] = True

        gravcomp_body = builder.custom_attributes["mujoco:gravcomp"]
        if gravcomp_body.values is None:
            gravcomp_body.values = {}
        for body_idx in range(2, 14):
            gravcomp_body.values[body_idx] = 1.0

        solimp_attr = builder.custom_attributes.get("mujoco:geom_solimp")
        priority_attr = builder.custom_attributes.get("mujoco:geom_priority")
        if solimp_attr is not None and priority_attr is not None:
            if solimp_attr.values is None:
                solimp_attr.values = {}
            if priority_attr.values is None:
                priority_attr.values = {}
            for s, b in enumerate(builder.shape_body):
                if b in (12, 13):
                    solimp_attr.values[s] = (0.7, 0.95, 0.0001, 0.5, 2.0)
                    priority_attr.values[s] = 1

        # Table
        table_cfg = newton.ModelBuilder.ShapeConfig(
            margin=1e-3,
            density=1000.0,
            ke=5.0e4,
            kd=5.0e2,
            mu=1.0,
        )
        builder.add_shape_box(
            body=-1,
            hx=0.4,
            hy=0.4,
            hz=0.5 * self.table_height,
            xform=wp.transform(self.table_pos, wp.quat_identity()),
            cfg=table_cfg,
        )

        return builder

    def add_board_floor(self, scene, center_x, center_y, brick_cfg, collider_cfg):
        """Add a static gray brick floor centered at (center_x, center_y)."""
        gray_mesh = _build_mesh_with_sdf(
            self.v_2x4,
            self.f_2x4,
            color=(0.35, 0.35, 0.35),
            scale=BRICK_SCALE,
        )
        sqrt2_2 = float(np.sqrt(2.0) / 2.0)
        floor_rot = wp.quat(0.0, 0.0, sqrt2_2, sqrt2_2)
        floor_z = self.table_top_center[2] - 0.8 * self.brick_height_scaled

        solimp_attr = scene.custom_attributes.get("mujoco:geom_solimp")
        bw = self.brick_width_scaled
        bl = self.brick_length_scaled
        inset = COLLIDER_INSET * BRICK_SCALE
        box_hz = 0.5 * (BRICK_HEIGHT - STUD_HEIGHT) * BRICK_SCALE - inset
        box_cz = STUD_HEIGHT * BRICK_SCALE + inset + box_hz
        stud_hh = 0.5 * STUD_HEIGHT * BRICK_SCALE - inset
        stud_cz = BRICK_HEIGHT * BRICK_SCALE + stud_hh
        nx, ny = 4, 2
        ox = 0.5 * nx * PITCH * BRICK_SCALE
        oy = 0.5 * ny * PITCH * BRICK_SCALE
        wt = WALL_THICKNESS * BRICK_SCALE
        wall_hz = 0.5 * BRICK_HEIGHT * BRICK_SCALE - inset
        wall_cz = wall_hz + inset
        wall_boxes = [
            (wp.vec3(ox - 0.5 * wt, 0.0, wall_cz), 0.5 * wt - inset, oy - inset, wall_hz),
            (wp.vec3(-(ox - 0.5 * wt), 0.0, wall_cz), 0.5 * wt - inset, oy - inset, wall_hz),
            (wp.vec3(0.0, oy - 0.5 * wt, wall_cz), ox - inset, 0.5 * wt - inset, wall_hz),
            (wp.vec3(0.0, -(oy - 0.5 * wt), wall_cz), ox - inset, 0.5 * wt - inset, wall_hz),
        ]
        for dx in (-1.5 * bw, -0.5 * bw, 0.5 * bw, 1.5 * bw):
            for dy in (-0.5 * bl, 0.5 * bl):
                pos = wp.vec3(center_x + dx, center_y + dy, floor_z)
                brick_xform = wp.transform(pos, floor_rot)
                shape_idx = scene.shape_count
                scene.add_shape_mesh(
                    body=-1,
                    mesh=gray_mesh,
                    cfg=brick_cfg,
                    xform=brick_xform,
                )
                if solimp_attr is not None:
                    if solimp_attr.values is None:
                        solimp_attr.values = {}
                    solimp_attr.values[shape_idx] = (0.6, 0.95, 0.00075, 0.5, 2.5)
                box_local = wp.transform(wp.vec3(0.0, 0.0, box_cz), wp.quat_identity())
                scene.add_shape_box(
                    body=-1,
                    hx=ox - inset,
                    hy=oy - inset,
                    hz=box_hz,
                    xform=wp.transform_multiply(brick_xform, box_local),
                    cfg=collider_cfg,
                )
                for w_pos, w_hx, w_hy, w_hz in wall_boxes:
                    w_local = wp.transform(w_pos, wp.quat_identity())
                    scene.add_shape_box(
                        body=-1,
                        hx=w_hx,
                        hy=w_hy,
                        hz=w_hz,
                        xform=wp.transform_multiply(brick_xform, w_local),
                        cfg=collider_cfg,
                    )
                for si in range(nx):
                    for sj in range(ny):
                        sx = (si - (nx - 1) / 2.0) * PITCH * BRICK_SCALE
                        sy = (sj - (ny - 1) / 2.0) * PITCH * BRICK_SCALE
                        stud_local = wp.transform(wp.vec3(sx, sy, stud_cz), wp.quat_identity())
                        scene.add_shape_cylinder(
                            body=-1,
                            radius=STUD_COLLIDER_RADIUS * BRICK_SCALE,
                            half_height=stud_hh,
                            xform=wp.transform_multiply(brick_xform, stud_local),
                            cfg=collider_cfg,
                        )

    def add_bricks(self, scene):
        brick_cfg = newton.ModelBuilder.ShapeConfig(
            density=BRICK_DENSITY,
            ke=BRICK_KE,
            kd=BRICK_KD,
            mu=0.7,
            margin=BRICK_MARGIN,
            gap=SDF_MARGIN,
        )
        # Invisible primitive colliders (box walls, floor slab, stud cylinders)
        # approximate the brick geometry to make brick-brick contacts more robust,
        # i.e. reduce interpenetration that can occur with the compliant SDF model
        # when bricks move quickly or collide violently, outside the gentle Franka
        # pick-and-place regime this example is designed around.
        #
        # Keep the proxies collision-only with zero density so they do not add to
        # mass and inertia on top of the visible SDF mesh.
        collider_cfg = newton.ModelBuilder.ShapeConfig(
            density=0.0,
            ke=BRICK_KE,
            kd=BRICK_KD,
            mu=0.7,
            margin=BRICK_MARGIN,
            gap=SDF_MARGIN,
            is_visible=False,
        )
        bh = 0.5 * self.brick_height_scaled
        sqrt2_2 = float(np.sqrt(2.0) / 2.0)
        rot_90z = wp.quat(0.0, 0.0, sqrt2_2, sqrt2_2)

        blue_x = self.table_top_center[0] - 0.05
        blue_y = self.table_top_center[1] - 0.04
        self.add_board_floor(scene, blue_x, blue_y, brick_cfg, collider_cfg)

        positions = [
            self.table_top_center + wp.vec3(0.0, 0.06, bh),
            self.table_top_center + wp.vec3(0.05, -0.04, bh),
            wp.vec3(blue_x, blue_y, self.table_top_center[2] + 0.2 * self.brick_height_scaled),
        ]
        rotations = [rot_90z, rot_90z, wp.quat_identity()]
        colors = [(0.8, 0.1, 0.1), (0.1, 0.7, 0.1), (0.1, 0.2, 0.8)]
        labels = ["brick_red", "brick_green", "brick_blue"]

        solimp_attr = scene.custom_attributes.get("mujoco:geom_solimp")
        nx, ny = 4, 2
        inset = COLLIDER_INSET * BRICK_SCALE
        box_hz = 0.5 * (BRICK_HEIGHT - STUD_HEIGHT) * BRICK_SCALE - inset
        box_cz = STUD_HEIGHT * BRICK_SCALE + inset + box_hz
        stud_hh = 0.5 * STUD_HEIGHT * BRICK_SCALE - inset
        stud_cz = BRICK_HEIGHT * BRICK_SCALE + stud_hh
        ox = 0.5 * nx * PITCH * BRICK_SCALE
        oy = 0.5 * ny * PITCH * BRICK_SCALE
        wt = WALL_THICKNESS * BRICK_SCALE
        wall_hz = 0.5 * BRICK_HEIGHT * BRICK_SCALE - inset
        wall_cz = wall_hz + inset
        wall_boxes = [
            (wp.vec3(ox - 0.5 * wt, 0.0, wall_cz), 0.5 * wt - inset, oy - inset, wall_hz),
            (wp.vec3(-(ox - 0.5 * wt), 0.0, wall_cz), 0.5 * wt - inset, oy - inset, wall_hz),
            (wp.vec3(0.0, oy - 0.5 * wt, wall_cz), ox - inset, 0.5 * wt - inset, wall_hz),
            (wp.vec3(0.0, -(oy - 0.5 * wt), wall_cz), ox - inset, 0.5 * wt - inset, wall_hz),
        ]
        self.brick_bodies = []
        for i in range(self.brick_count):
            mesh = _build_mesh_with_sdf(self.v_2x4, self.f_2x4, color=colors[i], scale=BRICK_SCALE)
            body = scene.add_body(xform=wp.transform(positions[i], rotations[i]), label=labels[i])
            shape_idx = scene.shape_count
            scene.add_shape_mesh(body, mesh=mesh, cfg=brick_cfg)
            if solimp_attr is not None:
                if solimp_attr.values is None:
                    solimp_attr.values = {}
                solimp_attr.values[shape_idx] = (0.6, 0.95, 0.00075, 0.5, 2.5)
            scene.add_shape_box(
                body=body,
                hx=ox - inset,
                hy=oy - inset,
                hz=box_hz,
                xform=wp.transform(wp.vec3(0.0, 0.0, box_cz), wp.quat_identity()),
                cfg=collider_cfg,
            )
            for w_pos, w_hx, w_hy, w_hz in wall_boxes:
                scene.add_shape_box(
                    body=body,
                    hx=w_hx,
                    hy=w_hy,
                    hz=w_hz,
                    xform=wp.transform(w_pos, wp.quat_identity()),
                    cfg=collider_cfg,
                )
            for si in range(nx):
                for sj in range(ny):
                    sx = (si - (nx - 1) / 2.0) * PITCH * BRICK_SCALE
                    sy = (sj - (ny - 1) / 2.0) * PITCH * BRICK_SCALE
                    scene.add_shape_cylinder(
                        body=body,
                        radius=STUD_COLLIDER_RADIUS * BRICK_SCALE,
                        half_height=stud_hh,
                        xform=wp.transform(wp.vec3(sx, sy, stud_cz), wp.quat_identity()),
                        cfg=collider_cfg,
                    )
            self.brick_bodies.append(body)

    # -- IK ------------------------------------------------------------------

    def _solve_approach_ik(self):
        """Solve IK for the approach pose above the red brick."""
        bh = 0.5 * self.brick_height_scaled
        sqrt2_2 = np.sqrt(2.0) / 2.0

        red_pos = np.array(
            [
                float(self.table_top_center[0]),
                float(self.table_top_center[1]) + 0.06,
                float(self.table_top_center[2]) + bh,
            ]
        )
        target_pos = red_pos + np.array([0.0, 0.0, float(self.offset_approach[2])])

        down = np.array([1.0, 0.0, 0.0, 0.0])
        inv_pick = np.array([0.0, 0.0, -sqrt2_2, sqrt2_2])
        x1, y1, z1, w1 = down
        x2, y2, z2, w2 = inv_pick
        target_quat = np.array(
            [
                w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
                w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            ]
        )

        ik_dofs = self.model_ik.joint_coord_count
        seed = np.zeros(ik_dofs, dtype=np.float32)
        seed[:7] = [0.0, 0.5, 0.0, -1.5, 0.0, 2.0, 0.78]
        joint_q = wp.array(seed.reshape(1, -1), dtype=wp.float32)

        solver = ik.IKSolver(
            model=self.model_ik,
            n_problems=1,
            objectives=[
                ik.IKObjectivePosition(
                    link_index=self.ee_index,
                    link_offset=wp.vec3(0.0, 0.0, 0.0),
                    target_positions=wp.array([wp.vec3(*target_pos.tolist())], dtype=wp.vec3),
                ),
                ik.IKObjectiveRotation(
                    link_index=self.ee_index,
                    link_offset_rotation=wp.quat_identity(),
                    target_rotations=wp.array([wp.vec4(*target_quat.tolist())], dtype=wp.vec4),
                ),
                ik.IKObjectiveJointLimit(
                    joint_limit_lower=self.model_ik.joint_limit_lower[:ik_dofs],
                    joint_limit_upper=self.model_ik.joint_limit_upper[:ik_dofs],
                ),
            ],
            lambda_initial=0.1,
            jacobian_mode=ik.IKJacobianType.ANALYTIC,
        )
        for _ in range(30):
            solver.step(joint_q, joint_q, iterations=24)

        return joint_q.flatten().numpy()[:7]

    def setup_ik(self):
        state_ik = self.model.state()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, state_ik)
        body_q_np = state_ik.body_q.numpy()

        self.pos_obj = ik.IKObjectivePosition(
            link_index=self.ee_index,
            link_offset=wp.vec3(0.0, 0.0, 0.0),
            target_positions=wp.array([self.home_pos], dtype=wp.vec3),
        )
        self.rot_obj = ik.IKObjectiveRotation(
            link_index=self.ee_index,
            link_offset_rotation=wp.quat_identity(),
            target_rotations=wp.array([body_q_np[self.ee_index][3:][:4]], dtype=wp.vec4),
        )

        ik_dofs = self.model_ik.joint_coord_count
        obj_joint_limits = ik.IKObjectiveJointLimit(
            joint_limit_lower=wp.clone(self.model_ik.joint_limit_lower[:ik_dofs].reshape((1, ik_dofs))).flatten(),
            joint_limit_upper=wp.clone(self.model_ik.joint_limit_upper[:ik_dofs].reshape((1, ik_dofs))).flatten(),
        )
        self.joint_q_ik = wp.clone(self.model.joint_q[:ik_dofs].reshape((1, ik_dofs)))

        self.ik_iters = 24
        self.ik_solver = ik.IKSolver(
            model=self.model_ik,
            n_problems=1,
            objectives=[self.pos_obj, self.rot_obj, obj_joint_limits],
            lambda_initial=0.1,
            jacobian_mode=ik.IKJacobianType.ANALYTIC,
        )

    # -- task FSM ------------------------------------------------------------

    def setup_tasks(self):
        # Round 1: pick red, place on green, release.
        round_1 = [
            TaskType.APPROACH,
            TaskType.REFINE_APPROACH,
            TaskType.GRASP,
            TaskType.LIFT,
            TaskType.MOVE_TO_DROP_OFF,
            TaskType.REFINE_DROP_OFF,
            TaskType.RELEASE,
        ]
        round_1_times = [2.0, 1.0, 0.5, 1.0, 1.5, 1.5, 1.0]

        # Round 2: grip red+green pair, place on blue, release, go home.
        round_2 = [
            TaskType.GRASP,
            TaskType.LIFT,
            TaskType.MOVE_TO_DROP_OFF,
            TaskType.REFINE_DROP_OFF,
            TaskType.RELEASE,
            TaskType.HOME,
        ]
        round_2_times = [0.5, 1.0, 1.5, 1.5, 1.0, 2.5]

        red, green, blue = self.brick_bodies

        task_schedule, task_pick, task_drop, task_layer, time_limits = [], [], [], [], []
        for tasks, times, pick, drop, layer in [
            (round_1, round_1_times, red, green, 1),
            (round_2, round_2_times, red, blue, 2),
        ]:
            task_schedule.extend(tasks)
            task_pick.extend([pick] * len(tasks))
            task_drop.extend([drop] * len(tasks))
            task_layer.extend([layer] * len(tasks))
            time_limits.extend(times)

        self.task_schedule = wp.array(task_schedule, dtype=wp.int32)
        self.task_time_limits = wp.array(time_limits, dtype=float)
        self.task_pick_body = wp.array(task_pick, dtype=wp.int32)
        self.task_drop_body = wp.array(task_drop, dtype=wp.int32)
        self.task_drop_layer = wp.array(task_layer, dtype=wp.int32)

        self.task_idx = wp.zeros(1, dtype=wp.int32)
        self.task_time_elapsed = wp.zeros(1, dtype=wp.float32)
        self.task_init_body_q = wp.clone(self.state_0.body_q)

        self.ee_pos_target = wp.zeros(1, dtype=wp.vec3)
        self.ee_pos_interp = wp.zeros(1, dtype=wp.vec3)
        self.ee_rot_target = wp.zeros(1, dtype=wp.vec4)
        self.ee_rot_interp = wp.zeros(1, dtype=wp.vec4)
        self.gripper_target = wp.zeros(shape=(1, 2), dtype=wp.float32)

    def set_joint_targets(self):
        wp.launch(
            set_target_pose_kernel,
            dim=1,
            inputs=[
                self.task_schedule,
                self.task_time_limits,
                self.task_pick_body,
                self.task_drop_body,
                self.task_drop_layer,
                self.task_idx,
                self.task_time_elapsed,
                self.frame_dt,
                self.offset_approach,
                self.offset_lift,
                self.grasp_z_offset,
                self.drop_z_offset,
                self.brick_height_scaled,
                self.home_pos,
                self.task_init_body_q,
                self.state_0.body_q,
                self.ee_index,
            ],
            outputs=[
                self.ee_pos_target,
                self.ee_pos_interp,
                self.ee_rot_target,
                self.ee_rot_interp,
                self.gripper_target,
            ],
        )

        self.pos_obj.set_target_positions(self.ee_pos_interp)
        self.rot_obj.set_target_rotations(self.ee_rot_interp)

        if self.graph_ik is not None:
            wp.capture_launch(self.graph_ik)
        else:
            self.ik_solver.step(self.joint_q_ik, self.joint_q_ik, iterations=self.ik_iters)

        wp.copy(dest=self.control.joint_target_pos[:7], src=self.joint_q_ik.flatten()[:7])
        wp.copy(dest=self.control.joint_target_pos[7:9], src=self.gripper_target.flatten()[:2])

        wp.launch(
            advance_task_kernel,
            dim=1,
            inputs=[
                self.task_time_limits,
                self.ee_pos_target,
                self.ee_rot_target,
                self.state_0.body_q,
                self.ee_index,
            ],
            outputs=[self.task_idx, self.task_time_elapsed, self.task_init_body_q],
        )

    # -- simulation loop -----------------------------------------------------

    def capture(self):
        self.graph = None
        self.graph_ik = None
        if wp.get_device().is_cuda:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph
            with wp.ScopedCapture() as capture:
                self.ik_solver.step(self.joint_q_ik, self.joint_q_ik, iterations=self.ik_iters)
            self.graph_ik = capture.graph

    def simulate(self):
        self.collision_pipeline.collide(self.state_0, self.contacts)
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def reset(self):
        self.sim_time = 0.0
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)
        wp.copy(self.control.joint_target_pos[:9], self.model.joint_q[:9])
        self.joint_q_ik = wp.clone(self.model.joint_q[: self.model_ik.joint_coord_count].reshape((1, -1)))
        self.setup_tasks()
        self.capture()

    def step(self):
        self.set_joint_targets()

        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()

        self.sim_time += self.frame_dt

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()

    def test_final(self):
        task_idx = self.task_idx.numpy()[0]
        total_tasks = len(self.task_schedule)
        if task_idx < total_tasks - 1:
            raise ValueError(f"Task sequence incomplete: reached step {task_idx}/{total_tasks - 1}")

        body_q = self.state_0.body_q.numpy()
        bh = self.brick_height_scaled
        red, green, blue = self.brick_bodies

        blue_z = body_q[blue][2]
        green_z = body_q[green][2]
        red_z = body_q[red][2]

        blue_xy = body_q[blue][:2]
        green_xy = body_q[green][:2]
        red_xy = body_q[red][:2]

        errors = []
        tol_z = 0.15 * bh

        for name, idx in [("Blue", blue), ("Green", green), ("Red", red)]:
            if not np.all(np.isfinite(body_q[idx])):
                errors.append(f"{name} brick (body {idx}) has non-finite transform")

        # All three bricks should be roughly aligned in XY
        if np.linalg.norm(green_xy - blue_xy) > 0.01:
            errors.append(f"Green brick XY offset from blue: {np.linalg.norm(green_xy - blue_xy):.4f} m (max 0.01)")
        if np.linalg.norm(red_xy - blue_xy) > 0.01:
            errors.append(f"Red brick XY offset from blue: {np.linalg.norm(red_xy - blue_xy):.4f} m (max 0.01)")

        # Green should be ~1 brick height above blue
        dz_green = green_z - blue_z
        if abs(dz_green - bh) > tol_z:
            errors.append(f"Green-Blue height gap: {dz_green:.4f} m, expected ~{bh:.4f} m")

        # Red should be ~2 brick heights above blue
        dz_red = red_z - blue_z
        if abs(dz_red - 2.0 * bh) > tol_z:
            errors.append(f"Red-Blue height gap: {dz_red:.4f} m, expected ~{2.0 * bh:.4f} m")

        if errors:
            raise ValueError("Brick stacking verification failed:\n  " + "\n  ".join(errors))


if __name__ == "__main__":
    parser = newton.examples.create_parser()

    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args)
    newton.examples.run(example, args)
