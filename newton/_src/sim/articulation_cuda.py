# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import functools

import warp as wp

from .articulation import eval_joint_child_state, eval_joint_motion
from .enums import JointType

TILE_BLOCK_DIM = 32
# The kernel stages one wp.transform (28 B) per joint in shared memory; 1024
# joints stay within the 48 KiB per-block floor of supported GPUs.
FK_TILE_MAX_JOINTS = 1024


@functools.cache
def create_eval_articulation_fk_tile(joint_capacity: int, write_all: bool, has_cable: bool):
    joint_capacity = wp.constant(joint_capacity)
    preserve_body_q = not write_all or has_cable

    @wp.kernel(enable_backward=False, module="unique")
    def eval_articulation_fk_tile(
        articulation_start: wp.array[int],
        articulation_end: wp.array[int],
        articulation_level_start: wp.array[int],
        level_joint_start: wp.array[int],
        level_joints: wp.array[int],
        fk_joint_parent: wp.array[int],
        articulation_count: int,
        articulation_mask: wp.array[bool],
        articulation_indices: wp.array[int],
        joint_q: wp.array[float],
        joint_qd: wp.array[float],
        joint_q_start: wp.array[int],
        joint_qd_start: wp.array[int],
        joint_type: wp.array[int],
        joint_parent: wp.array[int],
        joint_child: wp.array[int],
        joint_X_p: wp.array[wp.transform],
        joint_X_c: wp.array[wp.transform],
        joint_axis: wp.array[wp.vec3],
        joint_dof_dim: wp.array2d[int],
        body_com: wp.array[wp.vec3],
        body_flags: wp.array[int],
        body_flag_filter: int,
        body_q: wp.array[wp.transform],
        body_qd: wp.array[wp.spatial_vector],
    ):
        block, thread = wp.tid()

        articulation = block
        if articulation_indices:
            articulation = articulation_indices[block]
            if articulation < 0 or articulation >= articulation_count:
                return

        if articulation_mask and not articulation_mask[articulation]:
            return

        joint_begin = articulation_start[articulation]
        joint_end = articulation_end[articulation]
        joint_count = joint_end - joint_begin
        body_q_work = wp.tile_zeros(shape=joint_capacity, dtype=wp.transform, storage="shared")

        if wp.static(preserve_body_q):
            chunk_count = (joint_count + wp.block_dim() - 1) // wp.block_dim()
            for chunk in range(chunk_count):
                child_slot = chunk * wp.block_dim() + thread
                active = child_slot < joint_count
                target = int(0)
                child_q = wp.transform_identity()
                if active:
                    joint = joint_begin + child_slot
                    target = child_slot
                    child_q = body_q[joint_child[joint]]
                wp.tile_scatter_masked(body_q_work, target, child_q, active)

        level_begin = articulation_level_start[articulation]
        level_count = articulation_level_start[articulation + 1] - level_begin

        for local_level in range(level_count):
            level = level_begin + local_level
            level_joint_begin = level_joint_start[level]
            level_joint_count = level_joint_start[level + 1] - level_joint_begin
            chunk_count = (level_joint_count + wp.block_dim() - 1) // wp.block_dim()

            for chunk in range(chunk_count):
                scheduled_joint = chunk * wp.block_dim() + thread
                active = scheduled_joint < level_joint_count
                child_slot = int(0)
                X_wc = wp.transform_identity()
                write_child = bool(False)

                if active:
                    joint = level_joints[level_joint_begin + scheduled_joint]
                    type = joint_type[joint]
                    evaluate_joint = bool(True)
                    if wp.static(has_cable):
                        evaluate_joint = type != JointType.CABLE

                    if evaluate_joint:
                        parent = joint_parent[joint]
                        child = joint_child[joint]
                        parent_joint = fk_joint_parent[joint]

                        X_pj = joint_X_p[joint]
                        X_cj = joint_X_c[joint]
                        q_start = joint_q_start[joint]
                        qd_start = joint_qd_start[joint]
                        lin_axis_count = joint_dof_dim[joint, 0]
                        ang_axis_count = joint_dof_dim[joint, 1]
                        X_j, v_j = eval_joint_motion(
                            type,
                            q_start,
                            qd_start,
                            lin_axis_count,
                            ang_axis_count,
                            joint_q,
                            joint_qd,
                            joint_axis,
                        )

                        X_wp = wp.transform_identity()
                        if parent_joint >= joint_begin and parent_joint < joint_end:
                            X_wp = body_q_work[parent_joint - joint_begin]
                        elif parent >= 0:
                            X_wp = body_q[parent]
                        X_wc, v_wc = eval_joint_child_state(
                            type,
                            parent,
                            child,
                            X_wp,
                            X_pj,
                            X_cj,
                            X_j,
                            v_j,
                            body_qd,
                            body_com,
                        )

                        write_child = bool(True)
                        if wp.static(not write_all):
                            write_child = (body_flags[child] & body_flag_filter) != 0
                        if write_child:
                            body_q[child] = X_wc
                            body_qd[child] = v_wc
                            child_slot = joint - joint_begin

                wp.tile_scatter_masked(body_q_work, child_slot, X_wc, write_child)

    return eval_articulation_fk_tile
