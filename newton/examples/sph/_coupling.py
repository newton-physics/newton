# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Helpers for the SPH rigid-body coupling examples."""

import warp as wp


@wp.kernel
def _apply_collider_impulse_forces(
    dt: float,
    collider_ids: wp.array[wp.int32],
    collider_impulses: wp.array[wp.vec3],
    collider_impulse_positions: wp.array[wp.vec3],
    collider_body_index: wp.array[wp.int32],
    body_q: wp.array[wp.transform],
    body_com: wp.array[wp.vec3],
    body_f: wp.array[wp.spatial_vector],
):
    i = wp.tid()
    collider = collider_ids[i]
    if collider < 0 or collider >= collider_body_index.shape[0]:
        return

    body = collider_body_index[collider]
    if body < 0:
        return

    force = collider_impulses[i] / dt
    com_world = wp.transform_point(body_q[body], body_com[body])
    moment_arm = collider_impulse_positions[i] - com_world
    wp.atomic_add(body_f, body, wp.spatial_vector(force, wp.cross(moment_arm, force)))


@wp.kernel
def _update_body_impulse_norm_max(
    collider_ids: wp.array[wp.int32],
    collider_impulses: wp.array[wp.vec3],
    collider_body_index: wp.array[wp.int32],
    body_impulse_norm_max: wp.array[float],
):
    i = wp.tid()
    collider = collider_ids[i]
    if collider < 0 or collider >= collider_body_index.shape[0]:
        return

    body = collider_body_index[collider]
    if body < 0:
        return

    impulse_norm = wp.length(collider_impulses[i])
    wp.atomic_max(body_impulse_norm_max, body, impulse_norm)


@wp.kernel
def _subtract_body_force(
    dt: float,
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    body_f: wp.array[wp.spatial_vector],
    body_inv_inertia: wp.array[wp.mat33],
    body_inv_mass: wp.array[float],
    body_q_res: wp.array[wp.transform],
    body_qd_res: wp.array[wp.spatial_vector],
):
    body = wp.tid()
    force = body_f[body]
    linear_delta = dt * body_inv_mass[body] * wp.spatial_top(force)
    rot = wp.transform_get_rotation(body_q[body])
    angular_delta = dt * wp.quat_rotate(
        rot,
        body_inv_inertia[body] * wp.quat_rotate_inv(rot, wp.spatial_bottom(force)),
    )

    body_q_res[body] = body_q[body]
    body_qd_res[body] = body_qd[body] - wp.spatial_vector(linear_delta, angular_delta)


class SPHRigidBodyCoupling:
    """Bookkeeping shared by the SPH rigid-body coupling examples."""

    def __init__(self, model, sph_solver, state, fluid_states, substep_dt: float):
        if substep_dt <= 0.0:
            raise ValueError("SPH rigid-body coupling substep_dt must be positive.")
        self.model = model
        self.sph_solver = sph_solver
        self.substep_dt = float(substep_dt)

        self.sph_solver.setup_collider(model=self.model)
        self.body_sph_forces = wp.zeros_like(state.body_f)
        self.body_impulse_norm_max = wp.zeros(state.body_f.shape[0], dtype=float, device=self.model.device)
        self._max_collider_impulse_norm = 0.0

        fluid_states = tuple(fluid_states)
        if not fluid_states:
            raise ValueError("SPH rigid-body coupling requires at least one fluid state.")
        for fluid_state in fluid_states:
            self.init_fluid_state(state, fluid_state)
            self.update_fluid_state(state, fluid_state)

        self.collider_body_index = self.sph_solver.collider_body_index
        self.collect_impulses(fluid_states[0])

    @property
    def max_collider_impulse_norm(self) -> float:
        for norm in self.body_impulse_norm_max.numpy():
            self._max_collider_impulse_norm = max(self._max_collider_impulse_norm, float(norm))
        return self._max_collider_impulse_norm

    def init_fluid_state(self, state, fluid_state) -> None:
        fluid_state.body_q = wp.empty_like(state.body_q)
        fluid_state.body_qd = wp.empty_like(state.body_qd)
        fluid_state.body_f = wp.empty_like(state.body_f)

    def update_fluid_state(self, state, fluid_state) -> None:
        wp.launch(
            _subtract_body_force,
            dim=state.body_q.shape[0],
            inputs=[
                self.substep_dt,
                state.body_q,
                state.body_qd,
                self.body_sph_forces,
                self.model.body_inv_inertia,
                self.model.body_inv_mass,
                fluid_state.body_q,
                fluid_state.body_qd,
            ],
            device=self.model.device,
        )
        if fluid_state.body_f is not None:
            fluid_state.body_f.zero_()

    def apply_forces(self, state) -> None:
        if self.collider_impulses.shape[0] == 0:
            return
        wp.launch(
            _apply_collider_impulse_forces,
            dim=self.collider_impulses.shape[0],
            inputs=[
                self.substep_dt,
                self.collider_ids,
                self.collider_impulses,
                self.collider_impulse_positions,
                self.collider_body_index,
                state.body_q,
                self.model.body_com,
                state.body_f,
            ],
            device=self.model.device,
        )

    def save_applied_forces(self, state) -> None:
        self.body_sph_forces.assign(state.body_f)

    def collect_impulses(self, fluid_state) -> None:
        self.collider_impulses, self.collider_impulse_positions, self.collider_ids = (
            self.sph_solver.collect_collider_impulses(fluid_state)
        )
        if self.collider_impulses.shape[0] == 0:
            return
        wp.launch(
            _update_body_impulse_norm_max,
            dim=self.collider_impulses.shape[0],
            inputs=[
                self.collider_ids,
                self.collider_impulses,
                self.collider_body_index,
                self.body_impulse_norm_max,
            ],
            device=self.model.device,
        )


__all__ = ["SPHRigidBodyCoupling"]
