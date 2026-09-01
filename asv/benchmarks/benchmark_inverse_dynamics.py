# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Used for benchmarking the Newton inverse-dynamics evaluator.
#
# This module provides shared logic for inverse-dynamics benchmarks on
# the Franka Emika Panda robot.
###########################################################################

from __future__ import annotations

import numpy as np

import newton
import newton.utils

# Franka "ready" pose: arm bent at the elbow, end-effector pointing forward.
# Indexed against the FR3's seven arm joints; any extra joint coordinates
# (e.g. gripper fingers) stay at zero.
_FRANKA_READY_POSE = np.array(
    [0.0, -np.pi / 4.0, 0.0, -3.0 * np.pi / 4.0, 0.0, np.pi / 2.0, np.pi / 4.0],
    dtype=np.float32,
)


def create_franka_model(world_count: int = 1) -> newton.Model:
    """Build a fixed-base Franka FR3 model replicated across ``world_count`` worlds."""
    articulation_builder = newton.ModelBuilder()
    articulation_builder.num_rigid_contacts_per_world = 0
    articulation_builder.default_shape_cfg.density = 100.0
    asset_path = newton.utils.download_asset("franka_emika_panda") / "urdf/fr3.urdf"
    articulation_builder.add_urdf(asset_path, floating=False, scale=1.0)

    builder = newton.ModelBuilder()
    builder.replicate(articulation_builder, world_count)
    return builder.finalize(requires_grad=False)


def set_default_pose(model: newton.Model, state: newton.State) -> None:
    """Populate ``state.joint_q`` with the Franka ready pose tiled across
    every world and ``state.joint_qd`` with a small non-zero ramp (also
    tiled) so the Coriolis bias ``C(q, q_dot)*q_dot`` is non-trivial.
    Forward kinematics is evaluated so ``state.body_q`` stays consistent
    with ``state.joint_q`` (the documented precondition of
    :func:`~newton.eval_inverse_dynamics_passive`).
    """
    world_count = max(model.world_count, 1)

    n_coords = model.joint_coord_count
    coords_per_world = n_coords // world_count
    q_per_world = np.zeros(coords_per_world, dtype=np.float32)
    n = min(_FRANKA_READY_POSE.shape[0], coords_per_world)
    q_per_world[:n] = _FRANKA_READY_POSE[:n]
    state.joint_q.assign(np.tile(q_per_world, world_count))

    n_dofs = model.joint_dof_count
    dofs_per_world = n_dofs // world_count
    qd_per_world = np.linspace(0.1, 0.7, dofs_per_world, dtype=np.float32)
    state.joint_qd.assign(np.tile(qd_per_world, world_count))

    newton.eval_fk(model, state.joint_q, state.joint_qd, state)
