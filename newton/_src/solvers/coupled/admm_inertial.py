# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Reduced-coordinate target momentum and bias for numerical ADMM inertia."""

import warp as wp

from ...sim.articulation import eval_fk
from ...sim.inverse_dynamics import _InverseDynamicsScratchBuffer, _rnea_compensation_pass
from .model_view import ModelView


@wp.kernel(enable_backward=False)
def _isotropic_inertia(lump: wp.array[float], tensor: wp.array[wp.mat33]):
    body = wp.tid()
    tensor[body] = lump[body] * wp.identity(n=3, dtype=float)


@wp.kernel(enable_backward=False)
def _bias_at_com(
    articulation_start: wp.array[int],
    articulation_end: wp.array[int],
    joint_child: wp.array[int],
    body_q_com: wp.array[wp.transform],
    solve_origin: wp.array[wp.vec3],
    spatial_force: wp.array[wp.spatial_vector],
    force: wp.array[wp.spatial_vector],
):
    articulation = wp.tid()
    start = articulation_start[articulation]
    end = articulation_end[articulation]
    for joint in range(start, end):
        body = joint_child[joint]
        wrench = spatial_force[body]
        r = wp.transform_get_translation(body_q_com[body]) - solve_origin[body]
        linear = wp.spatial_top(wrench)
        angular = wp.spatial_bottom(wrench) - wp.cross(r, linear)
        force[body] = wp.spatial_vector(linear, angular)


class ReducedInertia:
    """Evaluate gamma target velocities and bias without a dynamics solve.

    The forward-kinematics pass evaluates ``J(q_n) qd_k``. The inverse-dynamics
    pass runs once per timestep on the added proximal inertia alone to supply its Coriolis
    force. Gravity is compensated separately by the body's force kernel.
    """

    def __init__(self, entry, mass, inertia, *, isotropic=True):
        self.view = ModelView(entry.view, f"{entry.name}_inertia")
        self.view.body_mass = mass
        self.view.body_inertia = (
            wp.empty(entry.view.body_count, dtype=wp.mat33, device=entry.view.device) if isotropic else inertia
        )
        self.inertia_lump = inertia if isotropic else None
        self.target = entry.view.state()
        self.bias = wp.zeros(entry.view.body_count, dtype=wp.spatial_vector, device=entry.view.device)
        self.joint_bias = wp.zeros(entry.view.joint_dof_count, dtype=float, device=entry.view.device)
        self.scratch = _InverseDynamicsScratchBuffer(
            body_count=entry.view.body_count,
            articulation_count=entry.view.articulation_count,
            joint_dof_count=entry.view.joint_dof_count,
            joint_target_q_count=entry.view.joint_target_q.shape[0],
            # Only the recursive bias pass is needed, not the dense mass matrix.
            max_dofs_per_articulation=0,
            max_joints_per_articulation=0,
            world_count=entry.view.world_count,
            device=entry.view.device,
        )

    def refresh(self, state):
        if self.inertia_lump is not None:
            wp.launch(
                _isotropic_inertia,
                dim=self.view.body_count,
                inputs=[self.inertia_lump, self.view.body_inertia],
                device=self.view.device,
            )
        _rnea_compensation_pass(
            self.view, state, self.scratch, state.joint_qd, self.scratch.zero_gravity, self.joint_bias
        )

        wp.launch(
            _bias_at_com,
            dim=self.view.articulation_count,
            inputs=[
                self.view.articulation_start,
                self.view.articulation_end,
                self.view.joint_child,
                self.scratch.body_q_com,
                self.scratch.body_solve_origin,
                self.scratch.body_f_s,
                self.bias,
            ],
            device=self.view.device,
        )

    def target_velocity(self, q_n, qd_k):
        eval_fk(self.view, q_n, qd_k, self.target)
        return self.target.body_qd
