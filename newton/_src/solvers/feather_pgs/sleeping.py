# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Experimental passive-island sleeping with collision-driven wake propagation."""

import math

import numpy as np
import warp as wp

from ...sim import Model
from ...sim.enums import BodyFlags, JointType, ModelFlags


class _SleepState:
    def __init__(self, solver, linear_threshold, angular_threshold, quiet_time, skip_constraints=True):
        for value in (linear_threshold, angular_threshold, quiet_time):
            if not math.isfinite(value) or value <= 0:
                raise ValueError("Sleeping thresholds and quiet time must be finite and positive")
        model = solver.model
        if (
            model.requires_grad
            or model.particle_count
            or solver.pgs_warmstart
            or solver._mf_warmstart_enabled
            or solver.contact_compliance
            or solver.pgs_velocity_iterations
            or solver.articulated_contact_response != "immediate"
        ):
            raise ValueError(
                "Experimental sleeping requires rigid immediate response, no compliance, warmstart or velocity passes"
            )
        if model.articulation_count == 0:
            raise ValueError("Experimental sleeping requires at least one articulation")
        self.model = model
        self.solver = solver
        self.linear_threshold = linear_threshold
        self.angular_threshold = angular_threshold
        self.quiet_time = quiet_time
        self.skip_constraints = skip_constraints
        self.carry_frozen_patches = True
        # Skip the articulated dynamics of islands that sleep through a step; their outputs are frozen anyway.
        self.skip_dynamics = skip_constraints
        body_nodes = solver.body_to_articulation.numpy().copy()
        response_counts = solver.articulation_response_dof_count.numpy()
        prescribed = (body_nodes >= 0) & (response_counts[np.maximum(body_nodes, 0)] == 0)
        body_nodes[prescribed] = -1
        device = model.device
        self.body_nodes = wp.array(body_nodes, dtype=int, device=device)
        self.body_prescribed = wp.array(prescribed.astype(np.int32), dtype=int, device=device)
        self.contact_response_mask = wp.clone(solver.body_has_response_dofs)
        self.limit_q_index = wp.clone(solver._joint_limit_q_index)
        count = model.articulation_count
        self.parent = wp.empty(count, dtype=int, device=device)
        self.wake_parent = wp.empty(count, dtype=int, device=device)
        self.previous_root = wp.array(np.arange(count, dtype=np.int32), dtype=int, device=device)
        self.art_awake = wp.ones(count, dtype=int, device=device)
        self.step_asleep = wp.zeros(count, dtype=int, device=device)
        self.quiet_age = wp.zeros(count, dtype=float, device=device)
        self.root_awake = wp.zeros(count, dtype=int, device=device)
        self.root_veto = wp.zeros(count, dtype=int, device=device)
        self.root_supported = wp.zeros(count, dtype=int, device=device)
        self.root_ready = wp.ones(count, dtype=int, device=device)
        self.old_group_wake = wp.zeros(count, dtype=int, device=device)
        self.body_awake = wp.ones(model.body_count, dtype=int, device=device)
        self.body_island = wp.full(model.body_count, -1, dtype=int, device=device)
        self.valid = wp.zeros(1, dtype=int, device=device)
        self.incomplete = wp.zeros(1, dtype=int, device=device)
        self.last_q = wp.clone(model.joint_q)
        self.last_qd = wp.clone(model.joint_qd)
        self.last_body_q = wp.clone(model.body_q)
        self.last_body_qd = wp.zeros(model.body_count, dtype=wp.spatial_vector, device=device)
        self.last_gravity = wp.clone(model.gravity)
        gravity_world = model.articulation_world.numpy().copy()
        gravity_world[gravity_world < 0] = model.gravity.size - 1
        self.gravity_world = wp.array(gravity_world, dtype=int, device=device)
        joint_art = model.joint_articulation.numpy()
        q_start, qd_start = model.joint_q_start.numpy(), model.joint_qd_start.numpy()
        coord_art = np.full(model.joint_coord_count, -1, dtype=np.int32)
        dof_art = np.full(model.joint_dof_count, -1, dtype=np.int32)
        for joint, art in enumerate(joint_art):
            coord_art[q_start[joint] : q_start[joint + 1]] = art
            dof_art[qd_start[joint] : qd_start[joint + 1]] = art
        self.coord_art = wp.array(coord_art, dtype=int, device=device)
        # Articulations without response DOFs are prescribed; their kinematics always run.
        skippable = response_counts > 0
        self.art_skippable = wp.array(skippable.astype(np.int32), dtype=int, device=device)
        self.joint_skip_art = wp.array(
            np.where(np.isin(joint_art, np.flatnonzero(skippable)), joint_art, -1).astype(np.int32),
            dtype=int,
            device=device,
        )
        self.dof_skip_art = wp.array(
            np.where(np.isin(dof_art, np.flatnonzero(skippable)), dof_art, -1).astype(np.int32),
            dtype=int,
            device=device,
        )
        self.dof_art = wp.array(dof_art, dtype=int, device=device)
        starts = model.articulation_start.numpy()[:-1]
        root_types = model.joint_type.numpy()[starts]
        root_parents = model.joint_parent.numpy()[starts]
        fixed = (root_types == int(JointType.FIXED)) & (root_parents < 0)
        self.anchored = wp.array(fixed.astype(np.int32), dtype=int, device=device)
        # Loop and mimic rows are built for every articulation, so their owners never sleep.
        pinned = np.zeros(count, dtype=np.int32)
        body_art = solver.body_to_articulation.numpy()
        loops = np.flatnonzero(solver._model_plan.loop_joint_articulation >= 0)
        for body in np.concatenate((model.joint_parent.numpy()[loops], model.joint_child.numpy()[loops])):
            if body >= 0 and body_art[body] >= 0:
                pinned[body_art[body]] = 1
        if solver._mimic_count:
            pinned[np.flatnonzero(np.diff(solver._mimic_art_start_np))] = 1
            for dofs in (solver._mimic_dof0.numpy(), solver._mimic_dof1.numpy()):
                arts = dof_art[dofs[dofs >= 0]]
                pinned[arts[arts >= 0]] = 1
        self.pinned = wp.array(pinned, dtype=int, device=device)
        self.changed = wp.zeros(count, dtype=int, device=device)
        self.asleep_steps = wp.zeros(count, dtype=int, device=device)
        self.frozen_bodies = wp.zeros(model.body_count, dtype=int, device=device)
        shape_art = np.full(model.shape_count, -1, dtype=np.int32)
        shape_body = model.shape_body.numpy()
        shape_art[shape_body >= 0] = body_nodes[shape_body[shape_body >= 0]]
        frequency = Model.AttributeFrequency
        entity_arts = {
            ModelFlags.JOINT_PROPERTIES: (frequency.JOINT, joint_art),
            ModelFlags.JOINT_DOF_PROPERTIES: (frequency.JOINT_DOF, dof_art),
            ModelFlags.BODY_PROPERTIES: (frequency.BODY, body_nodes),
            ModelFlags.BODY_INERTIAL_PROPERTIES: (frequency.BODY, body_nodes),
            ModelFlags.SHAPE_PROPERTIES: (frequency.SHAPE, shape_art),
        }
        # Per-entity model arrays, compared on notification so only islands whose properties changed wake.
        self.property_arrays = {
            flag: [(name, value.numpy().copy(), arts) for name, value in _entity_arrays(model, kind, arts.size)]
            for flag, (kind, arts) in entity_arts.items()
        }

    @property
    def patch_frozen_bodies(self):
        """Bodies whose friction-patch history can be carried; only row-free sleeping qualifies."""
        return self.frozen_bodies if self.skip_constraints and self.carry_frozen_patches else None

    def wake(self, world_mask=None):
        """Wake every component and invalidate state-change history, or wake only the masked worlds."""
        requested = self.solver._mass_update_requested
        if world_mask is None:
            wp.launch(_wake_all, self.art_awake.size, [self.art_awake, requested], device=self.model.device)
            self.body_awake.fill_(1)
            self.quiet_age.zero_()
            self.valid.zero_()
            self.asleep_steps.zero_()
            return
        device = self.model.device
        world = self.solver.art_to_world
        wp.launch(
            _wake_worlds,
            self.art_awake.size,
            [world_mask, world, self.art_awake, self.quiet_age, requested],
            device=device,
        )
        wp.launch(
            _wake_world_bodies,
            self.body_awake.size,
            [world_mask, world, self.body_nodes, self.body_awake],
            device=device,
        )

    def notify(self, flags):
        """Wake islands whose per-entity model properties changed, or every island for other changes."""
        flags = int(flags)
        tracked = 0
        for flag in self.property_arrays:
            tracked |= int(flag)
        if flags & ~tracked:
            self.wake()
            return
        # Several notifications can arrive before the next step consumes the mask; keep every pending wake.
        changed = self.changed.numpy().copy()
        unowned = False
        for flag, entries in self.property_arrays.items():
            if not flags & int(flag):
                continue
            for index, (name, snapshot, arts) in enumerate(entries):
                # Read the model's current allocation: callers may replace an array instead of assigning into it.
                current = getattr(self.model, name).numpy()
                if current.shape != snapshot.shape:
                    raise ValueError(f"Model array {name!r} changed shape from {snapshot.shape} to {current.shape}.")
                differs = (current != snapshot) & ~(_isnan(current) & _isnan(snapshot))
                if differs.ndim > 1:
                    differs = differs.reshape(differs.shape[0], -1).any(axis=1)
                owners = arts[differs]
                changed[owners[owners >= 0]] = 1
                # Static geometry and prescribed bodies support sleeping islands without belonging to one.
                unowned |= bool(np.any(owners < 0))
                entries[index] = (name, current.copy(), arts)
        if unowned:
            self.wake()
        else:
            self.changed.assign(changed)

    def begin(self, state, control, contacts):
        """Build conservative islands and propagate input wake events."""
        model = self.model
        device = model.device
        self.root_awake.zero_()
        self.root_veto.zero_()
        self.root_supported.zero_()
        self.root_ready.fill_(1)
        self.old_group_wake.zero_()
        self.incomplete.zero_()
        wp.launch(_reset_parent, self.parent.size, [self.parent], device=device)
        if contacts is not None:
            wp.launch(
                _join_contacts,
                contacts.rigid_contact_max,
                [
                    contacts.rigid_contact_count,
                    contacts.rigid_contact_shape0,
                    contacts.rigid_contact_shape1,
                    model.shape_body,
                    self.body_nodes,
                    model.body_flags,
                    self.solver.art_to_world,
                    self.parent,
                    self.incomplete,
                ],
                device=device,
            )
        wp.launch(_compress, self.parent.size, [self.parent], device=device)
        wp.copy(self.wake_parent, self.parent)
        wp.launch(
            _join_previous, self.parent.size, [self.art_awake, self.previous_root, self.wake_parent], device=device
        )
        wp.launch(_compress, self.parent.size, [self.wake_parent], device=device)
        wp.launch(
            _input_components,
            self.parent.size,
            [self.parent, self.anchored, self.art_awake, self.root_awake, self.root_supported],
            device=device,
        )
        wp.launch(
            _input_bodies,
            model.body_count,
            [
                self.body_nodes,
                self.parent,
                model.body_flags,
                state.body_q,
                state.body_qd,
                state.body_f,
                self.last_body_q,
                self.last_body_qd,
                self.valid,
                self.root_veto,
            ],
            device=device,
        )
        wp.launch(
            _input_coords,
            model.joint_coord_count,
            [self.coord_art, self.parent, state.joint_q, self.last_q, self.valid, self.root_veto],
            device=device,
        )
        wp.launch(
            _input_dofs,
            model.joint_dof_count,
            [
                self.dof_art,
                self.parent,
                state.joint_qd,
                self.last_qd,
                control.joint_f,
                model.joint_target_ke,
                model.joint_target_kd,
                self.solver._passive_spring_stiffness,
                self.valid,
                self.root_veto,
            ],
            device=device,
        )
        wp.launch(_veto_pinned, self.parent.size, [self.parent, self.pinned, self.root_veto], device=device)
        wp.launch(_veto_pinned, self.parent.size, [self.parent, self.changed, self.root_veto], device=device)
        self.changed.zero_()
        wp.launch(
            _gravity_change,
            self.parent.size,
            [model.gravity, self.last_gravity, self.gravity_world, self.parent, self.root_veto],
            device=device,
        )
        if contacts is not None:
            wp.launch(
                _contact_boundaries,
                contacts.rigid_contact_max,
                [
                    contacts.rigid_contact_count,
                    contacts.rigid_contact_shape0,
                    contacts.rigid_contact_shape1,
                    model.shape_body,
                    self.body_nodes,
                    model.body_flags,
                    self.body_prescribed,
                    self.parent,
                    contacts.rigid_contact_point0,
                    contacts.rigid_contact_point1,
                    contacts.rigid_contact_normal,
                    contacts.rigid_contact_margin0,
                    contacts.rigid_contact_margin1,
                    state.body_q,
                    state.body_qd,
                    self.last_body_q,
                    self.valid,
                    self.root_veto,
                    self.root_supported,
                ],
                device=device,
            )
        wp.launch(
            _veto_unsafe_islands,
            self.parent.size,
            [
                self.parent,
                self.root_supported,
                self.incomplete,
                contacts._reduction_overflow if contacts is not None else self.incomplete,
                self.solver.art_to_world,
                self.solver.constraint_overflow,
                self.root_veto,
            ],
            device=device,
        )
        wp.launch(
            _mark_old_wake,
            self.parent.size,
            [self.parent, self.wake_parent, self.art_awake, self.root_awake, self.root_veto, self.old_group_wake],
            device=device,
        )
        wp.launch(
            _propagate_old_wake,
            self.parent.size,
            [self.parent, self.wake_parent, self.old_group_wake, self.root_veto],
            device=device,
        )
        wp.launch(
            _wake_components,
            self.parent.size,
            [
                self.parent,
                self.root_awake,
                self.root_veto,
                self.art_awake,
                self.quiet_age,
                self.solver._mass_update_requested,
            ],
            device=device,
        )
        solver = self.solver
        skip = int(self.skip_constraints and self.skip_dynamics)
        wp.launch(
            _articulation_activity,
            self.parent.size,
            [
                self.art_awake,
                self.art_skippable,
                int(self.skip_constraints),
                skip,
                self.step_asleep,
                solver._dynamics_art_active,
                solver._dynamics_art_mask,
                solver._constraint_art_active,
                solver._fk_id_cache_valid,
            ],
            device=device,
        )
        wp.launch(
            _entity_activity,
            model.joint_dof_count,
            [self.dof_skip_art, skip, self.art_awake, solver._dynamics_dof_active],
            device=device,
        )
        wp.launch(
            _entity_activity,
            model.joint_count,
            [self.joint_skip_art, skip, self.art_awake, solver._dynamics_joint_active],
            device=device,
        )
        wp.launch(
            _entity_activity,
            model.body_count,
            [self.body_nodes, skip, self.art_awake, solver._dynamics_body_active],
            device=device,
        )
        if self.skip_constraints:
            wp.launch(
                _frozen_bodies,
                model.body_count,
                [self.body_nodes, self.art_awake, self.asleep_steps, self.frozen_bodies],
                device=device,
            )
            wp.launch(
                _contact_response_mask,
                model.body_count,
                [self.body_nodes, self.art_awake, self.solver.body_has_response_dofs, self.contact_response_mask],
                device=device,
            )
            wp.launch(
                _limit_coordinates,
                model.joint_dof_count,
                [self.dof_art, self.art_awake, self.solver._joint_limit_q_index, self.limit_q_index],
                device=device,
            )

    def finish(self, state_in, state_out, state_aug, dt):
        """Freeze quiet supported islands and publish synchronized state history."""
        model = self.model
        device = model.device
        wp.launch(
            _output_motion,
            model.body_count,
            [
                self.body_nodes,
                self.parent,
                self.art_awake,
                int(self.skip_constraints),
                int(self.skip_constraints and self.skip_dynamics),
                self.step_asleep,
                state_out.body_q,
                state_out.body_qd,
                self.linear_threshold,
                self.angular_threshold,
                self.root_veto,
            ],
            device=device,
        )
        wp.launch(
            _output_finite,
            model.joint_coord_count,
            [
                self.coord_art,
                self.parent,
                int(self.skip_constraints and self.skip_dynamics),
                self.step_asleep,
                state_out.joint_q,
                self.root_veto,
            ],
            device=device,
        )
        wp.launch(
            _output_finite,
            model.joint_dof_count,
            [
                self.dof_art,
                self.parent,
                int(self.skip_constraints and self.skip_dynamics),
                self.step_asleep,
                state_out.joint_qd,
                self.root_veto,
            ],
            device=device,
        )
        wp.launch(
            _quiet_components,
            self.parent.size,
            [
                self.parent,
                self.root_veto,
                self.root_supported,
                self.solver.art_to_world,
                self.solver.constraint_overflow,
                self.incomplete,
                dt,
                self.quiet_time,
                self.quiet_age,
                self.root_ready,
            ],
            device=device,
        )
        wp.launch(
            _publish_components,
            self.parent.size,
            [self.parent, self.root_ready, self.art_awake, self.previous_root, self.asleep_steps],
            device=device,
        )
        wp.launch(
            _freeze_coords,
            model.joint_coord_count,
            [
                self.coord_art,
                self.parent,
                self.art_awake,
                self.step_asleep,
                self.root_veto,
                state_in.joint_q,
                state_out.joint_q,
            ],
            device=device,
        )
        wp.launch(
            _freeze_dofs,
            model.joint_dof_count,
            [
                self.dof_art,
                self.parent,
                self.art_awake,
                self.step_asleep,
                self.root_veto,
                state_out.joint_qd,
                state_aug.joint_qdd,
                self.solver.v_out,
            ],
            device=device,
        )
        wp.launch(
            _freeze_bodies,
            model.body_count,
            [
                self.body_nodes,
                self.parent,
                self.art_awake,
                self.step_asleep,
                self.root_veto,
                state_in.body_q,
                state_out.body_q,
                state_out.body_qd,
                self.body_awake,
                self.body_island,
            ],
            device=device,
        )
        wp.copy(self.last_q, state_out.joint_q)
        wp.copy(self.last_qd, state_out.joint_qd)
        wp.copy(self.last_body_q, state_out.body_q)
        wp.copy(self.last_body_qd, state_out.body_qd)
        wp.copy(self.last_gravity, model.gravity)
        self.valid.fill_(1)
        if self.solver._fk_id_cache_enabled:
            self.solver._fk_id_cache_valid.zero_()


@wp.func
def _nonzero(norm: float):
    """Whether a norm is positive or nonfinite, so invalid inputs wake rather than freeze."""
    return not wp.isfinite(norm) or norm > 0.0


@wp.func
def _pose_changed(a: wp.transform, b: wp.transform):
    changed = False
    for j in range(7):
        if a[j] != b[j]:
            changed = True
    return changed


@wp.func
def _root(parent: wp.array[int], node: int):
    while parent[node] != node:
        node = parent[node]
    return node


@wp.func
def _union(parent: wp.array[int], a: int, b: int):
    while True:
        a = _root(parent, a)
        b = _root(parent, b)
        if a == b:
            break
        high, low = wp.max(a, b), wp.min(a, b)
        if wp.atomic_cas(parent, high, high, low) == high:
            break


@wp.func
def _node(body: int, body_nodes: wp.array[int], flags: wp.array[int]):
    node = int(-1)
    if body >= 0 and (flags[body] & int(BodyFlags.KINEMATIC)) == 0:
        node = body_nodes[body]
    return node


@wp.kernel(enable_backward=False)
def _wake_all(awake: wp.array[int], requested: wp.array[int]):
    i = wp.tid()
    if awake[i] == 0:
        requested[i] = 1
    awake[i] = 1


@wp.kernel(enable_backward=False)
def _wake_worlds(
    mask: wp.array[wp.bool], world: wp.array[int], awake: wp.array[int], age: wp.array[float], requested: wp.array[int]
):
    i = wp.tid()
    if world[i] >= 0 and mask[world[i]]:
        if awake[i] == 0:
            requested[i] = 1
        awake[i] = 1
        age[i] = 0.0


@wp.kernel(enable_backward=False)
def _wake_world_bodies(mask: wp.array[wp.bool], world: wp.array[int], nodes: wp.array[int], awake: wp.array[int]):
    i = wp.tid()
    node = nodes[i]
    if node >= 0 and world[node] >= 0 and mask[world[node]]:
        awake[i] = 1


@wp.kernel(enable_backward=False)
def _reset_parent(parent: wp.array[int]):
    i = wp.tid()
    parent[i] = i


@wp.kernel(enable_backward=False)
def _join_previous(awake: wp.array[int], previous: wp.array[int], parent: wp.array[int]):
    i = wp.tid()
    if awake[i] == 0:
        _union(parent, i, previous[i])


@wp.kernel(enable_backward=False)
def _join_contacts(
    count: wp.array[int],
    shape0: wp.array[int],
    shape1: wp.array[int],
    shape_body: wp.array[int],
    body_nodes: wp.array[int],
    flags: wp.array[int],
    world: wp.array[int],
    parent: wp.array[int],
    incomplete: wp.array[int],
):
    i = wp.tid()
    if count[0] > shape0.shape[0] or count[0] < 0:
        wp.atomic_max(incomplete, 0, 1)
    if i < count[0]:
        if shape0[i] < 0 or shape1[i] < 0:
            wp.atomic_max(incomplete, 0, 1)
            return
        a = _node(shape_body[shape0[i]], body_nodes, flags)
        b = _node(shape_body[shape1[i]], body_nodes, flags)
        if a >= 0 and b >= 0:
            if world[a] == world[b]:
                _union(parent, a, b)
            else:
                wp.atomic_max(incomplete, 0, 1)


@wp.kernel(enable_backward=False)
def _compress(parent: wp.array[int]):
    i = wp.tid()
    parent[i] = _root(parent, i)


@wp.kernel(enable_backward=False)
def _input_components(
    parent: wp.array[int],
    anchored: wp.array[int],
    awake: wp.array[int],
    root_awake: wp.array[int],
    supported: wp.array[int],
):
    i = wp.tid()
    wp.atomic_max(root_awake, parent[i], awake[i])
    wp.atomic_max(supported, parent[i], anchored[i])


@wp.kernel(enable_backward=False)
def _input_bodies(
    nodes: wp.array[int],
    parent: wp.array[int],
    flags: wp.array[int],
    q: wp.array[wp.transform],
    qd: wp.array[wp.spatial_vector],
    force: wp.array[wp.spatial_vector],
    last_q: wp.array[wp.transform],
    last_qd: wp.array[wp.spatial_vector],
    valid: wp.array[int],
    veto: wp.array[int],
):
    i = wp.tid()
    node = nodes[i]
    if node >= 0:
        changed = False
        if valid[0] != 0:
            changed = _pose_changed(q[i], last_q[i]) or _nonzero(wp.length(qd[i] - last_qd[i]))
        if changed or _nonzero(wp.length(force[i])) or (flags[i] & int(BodyFlags.KINEMATIC)) != 0:
            wp.atomic_max(veto, parent[node], 1)


@wp.kernel(enable_backward=False)
def _input_coords(
    nodes: wp.array[int],
    parent: wp.array[int],
    q: wp.array[float],
    last: wp.array[float],
    valid: wp.array[int],
    veto: wp.array[int],
):
    i = wp.tid()
    if nodes[i] >= 0 and valid[0] != 0 and q[i] != last[i]:
        wp.atomic_max(veto, parent[nodes[i]], 1)


@wp.kernel(enable_backward=False)
def _input_dofs(
    nodes: wp.array[int],
    parent: wp.array[int],
    qd: wp.array[float],
    last: wp.array[float],
    force: wp.array[float],
    ke: wp.array[float],
    kd: wp.array[float],
    spring: wp.array[float],
    valid: wp.array[int],
    veto: wp.array[int],
):
    i = wp.tid()
    if nodes[i] >= 0:
        if force[i] != 0.0 or ke[i] != 0.0 or kd[i] != 0.0 or spring[i] != 0.0 or (valid[0] != 0 and qd[i] != last[i]):
            wp.atomic_max(veto, parent[nodes[i]], 1)


@wp.kernel(enable_backward=False)
def _veto_pinned(parent: wp.array[int], pinned: wp.array[int], veto: wp.array[int]):
    i = wp.tid()
    if pinned[i] != 0:
        wp.atomic_max(veto, parent[i], 1)


@wp.kernel(enable_backward=False)
def _gravity_change(
    gravity: wp.array[wp.vec3],
    last: wp.array[wp.vec3],
    world: wp.array[int],
    parent: wp.array[int],
    veto: wp.array[int],
):
    i = wp.tid()
    if _nonzero(wp.length(gravity[world[i]] - last[world[i]])):
        wp.atomic_max(veto, parent[i], 1)


@wp.kernel(enable_backward=False)
def _contact_boundaries(
    count: wp.array[int],
    shape0: wp.array[int],
    shape1: wp.array[int],
    shape_body: wp.array[int],
    nodes: wp.array[int],
    flags: wp.array[int],
    prescribed: wp.array[int],
    parent: wp.array[int],
    point0: wp.array[wp.vec3],
    point1: wp.array[wp.vec3],
    normal: wp.array[wp.vec3],
    margin0: wp.array[float],
    margin1: wp.array[float],
    q: wp.array[wp.transform],
    qd: wp.array[wp.spatial_vector],
    last: wp.array[wp.transform],
    valid: wp.array[int],
    veto: wp.array[int],
    supported: wp.array[int],
):
    i = wp.tid()
    if i < count[0] and shape0[i] >= 0 and shape1[i] >= 0:
        bodies = wp.vec2i(shape_body[shape0[i]], shape_body[shape1[i]])
        pa = wp.vec3(point0[i])
        pb = wp.vec3(point1[i])
        if bodies[0] >= 0:
            pa = wp.transform_point(q[bodies[0]], pa)
        if bodies[1] >= 0:
            pb = wp.transform_point(q[bodies[1]], pb)
        gap = wp.dot(-normal[i], pa - pb) - margin0[i] - margin1[i]
        for side in range(2):
            a = _node(bodies[side], nodes, flags)
            b = _node(bodies[1 - side], nodes, flags)
            if a >= 0 and b < 0:
                if gap <= 1.0e-5:
                    wp.atomic_max(supported, parent[a], 1)
                other = bodies[1 - side]
                if other >= 0:
                    if nodes[other] < 0 and prescribed[other] == 0 and (flags[other] & int(BodyFlags.DYNAMIC)) != 0:
                        wp.atomic_max(veto, parent[a], 1)
                    speed = wp.length(qd[other])
                    if (
                        not wp.isfinite(speed)
                        or speed > 0.0
                        or (valid[0] != 0 and _pose_changed(q[other], last[other]))
                    ):
                        wp.atomic_max(veto, parent[a], 1)


@wp.kernel(enable_backward=False)
def _mark_old_wake(
    parent: wp.array[int],
    previous: wp.array[int],
    awake: wp.array[int],
    root_awake: wp.array[int],
    veto: wp.array[int],
    old_wake: wp.array[int],
):
    i = wp.tid()
    if awake[i] == 0 and (root_awake[parent[i]] != 0 or veto[parent[i]] != 0):
        wp.atomic_max(old_wake, previous[i], 1)


@wp.kernel(enable_backward=False)
def _propagate_old_wake(parent: wp.array[int], previous: wp.array[int], old_wake: wp.array[int], veto: wp.array[int]):
    i = wp.tid()
    if old_wake[previous[i]] != 0:
        wp.atomic_max(veto, parent[i], 1)


@wp.kernel(enable_backward=False)
def _wake_components(
    parent: wp.array[int],
    root_awake: wp.array[int],
    veto: wp.array[int],
    awake: wp.array[int],
    age: wp.array[float],
    requested: wp.array[int],
):
    i = wp.tid()
    if veto[parent[i]] != 0 or (awake[i] == 0 and root_awake[parent[i]] != 0):
        age[i] = 0.0
        # A woken articulation's mass matrix may predate its frozen pose; refresh it on this step.
        if awake[i] == 0:
            requested[i] = 1
        awake[i] = 1


@wp.kernel(enable_backward=False)
def _articulation_activity(
    awake: wp.array[int],
    skippable: wp.array[int],
    skip_constraints: int,
    skip: int,
    step_asleep: wp.array[int],
    active: wp.array[int],
    active_mask: wp.array[wp.bool],
    rows_active: wp.array[int],
    fk_id_cache_valid: wp.array[int],
):
    i = wp.tid()
    step_asleep[i] = 1 - awake[i]
    rows_active[i] = 1
    if skip_constraints != 0 and awake[i] == 0:
        rows_active[i] = 0
    active[i] = 1
    active_mask[i] = True
    if skip != 0:
        if awake[i] == 0 and skippable[i] != 0:
            active[i] = 0
            active_mask[i] = False
        # Sleeping articulations keep stale FK/ID terms, which only their discarded dynamics read.
        fk_id_cache_valid[i] = 1 - active[i]


@wp.kernel(enable_backward=False)
def _entity_activity(nodes: wp.array[int], skip: int, awake: wp.array[int], active: wp.array[int]):
    i = wp.tid()
    active[i] = 1
    if skip != 0 and nodes[i] >= 0 and awake[nodes[i]] == 0:
        active[i] = 0


@wp.kernel(enable_backward=False)
def _output_motion(
    nodes: wp.array[int],
    parent: wp.array[int],
    awake: wp.array[int],
    skip_constraints: int,
    skip_dynamics: int,
    step_asleep: wp.array[int],
    q: wp.array[wp.transform],
    qd: wp.array[wp.spatial_vector],
    linear: float,
    angular: float,
    veto: wp.array[int],
):
    i = wp.tid()
    # Sleepers whose dynamics were skipped hold stale outputs rather than trial results.
    if nodes[i] >= 0 and not (skip_dynamics != 0 and step_asleep[nodes[i]] != 0):
        pose = q[i]
        for j in range(7):
            if not wp.isfinite(pose[j]):
                wp.atomic_max(veto, parent[nodes[i]], 1)
        velocity = qd[i]
        if not wp.isfinite(wp.length(velocity)) or (
            (skip_constraints == 0 or awake[nodes[i]] != 0)
            and (wp.length(wp.spatial_top(velocity)) > linear or wp.length(wp.spatial_bottom(velocity)) > angular)
        ):
            wp.atomic_max(veto, parent[nodes[i]], 1)


@wp.kernel(enable_backward=False)
def _output_finite(
    nodes: wp.array[int],
    parent: wp.array[int],
    skip_dynamics: int,
    step_asleep: wp.array[int],
    value: wp.array[float],
    veto: wp.array[int],
):
    i = wp.tid()
    if nodes[i] < 0 or (skip_dynamics != 0 and step_asleep[nodes[i]] != 0):
        return
    if not wp.isfinite(value[i]):
        wp.atomic_max(veto, parent[nodes[i]], 1)


@wp.kernel(enable_backward=False)
def _quiet_components(
    parent: wp.array[int],
    veto: wp.array[int],
    supported: wp.array[int],
    world: wp.array[int],
    overflow: wp.array[wp.bool],
    incomplete: wp.array[int],
    dt: float,
    quiet_time: float,
    age: wp.array[float],
    ready: wp.array[int],
):
    i = wp.tid()
    root = parent[i]
    if veto[root] != 0 or supported[root] == 0 or overflow[world[i]] or incomplete[0] != 0:
        age[i] = 0.0
    else:
        age[i] = wp.min(age[i] + dt, quiet_time)
    if age[i] < quiet_time:
        wp.atomic_min(ready, root, 0)


@wp.kernel(enable_backward=False)
def _publish_components(
    parent: wp.array[int], ready: wp.array[int], awake: wp.array[int], previous: wp.array[int], asleep: wp.array[int]
):
    i = wp.tid()
    awake[i] = 1 - ready[parent[i]]
    previous[i] = parent[i]
    if awake[i] == 0:
        asleep[i] += 1
    else:
        asleep[i] = 0


@wp.func
def _frozen(node: int, parent: wp.array[int], awake: wp.array[int], step_asleep: wp.array[int], veto: wp.array[int]):
    # A step sleeper keeps its frozen state even if it wakes now, unless its own trial output vetoed it.
    return awake[node] == 0 or (step_asleep[node] != 0 and veto[parent[node]] == 0)


@wp.kernel(enable_backward=False)
def _freeze_coords(
    nodes: wp.array[int],
    parent: wp.array[int],
    awake: wp.array[int],
    step_asleep: wp.array[int],
    veto: wp.array[int],
    source: wp.array[float],
    target: wp.array[float],
):
    i = wp.tid()
    if nodes[i] >= 0 and _frozen(nodes[i], parent, awake, step_asleep, veto):
        target[i] = source[i]


@wp.kernel(enable_backward=False)
def _freeze_dofs(
    nodes: wp.array[int],
    parent: wp.array[int],
    awake: wp.array[int],
    step_asleep: wp.array[int],
    veto: wp.array[int],
    qd: wp.array[float],
    qdd: wp.array[float],
    v: wp.array[float],
):
    i = wp.tid()
    if nodes[i] >= 0 and _frozen(nodes[i], parent, awake, step_asleep, veto):
        qd[i] = 0.0
        qdd[i] = 0.0
        v[i] = 0.0


@wp.kernel(enable_backward=False)
def _freeze_bodies(
    nodes: wp.array[int],
    parent: wp.array[int],
    awake: wp.array[int],
    step_asleep: wp.array[int],
    veto: wp.array[int],
    source: wp.array[wp.transform],
    target: wp.array[wp.transform],
    qd: wp.array[wp.spatial_vector],
    body_awake: wp.array[int],
    body_island: wp.array[int],
):
    i = wp.tid()
    node = nodes[i]
    if node >= 0:
        body_awake[i] = awake[node]
        body_island[i] = parent[node]
        if _frozen(node, parent, awake, step_asleep, veto):
            target[i] = source[i]
            qd[i] = wp.spatial_vector()


@wp.kernel(enable_backward=False)
def _veto_unsafe_islands(
    parent: wp.array[int],
    supported: wp.array[int],
    incomplete: wp.array[int],
    reduction_overflow: wp.array[int],
    world: wp.array[int],
    overflow: wp.array[wp.bool],
    veto: wp.array[int],
):
    i = wp.tid()
    if supported[parent[i]] == 0 or incomplete[0] != 0 or reduction_overflow[0] != 0 or overflow[world[i]]:
        wp.atomic_max(veto, parent[i], 1)


@wp.kernel(enable_backward=False)
def _frozen_bodies(nodes: wp.array[int], awake: wp.array[int], asleep: wp.array[int], frozen: wp.array[int]):
    """Bodies whose poses, contacts and friction history repeat: asleep now and through the previous step."""
    i = wp.tid()
    frozen[i] = 0
    if nodes[i] >= 0 and awake[nodes[i]] == 0 and asleep[nodes[i]] >= 1:
        frozen[i] = 1


@wp.kernel(enable_backward=False)
def _contact_response_mask(
    nodes: wp.array[int],
    awake: wp.array[int],
    response: wp.array[int],
    mask: wp.array[int],
):
    i = wp.tid()
    mask[i] = response[i]
    if nodes[i] >= 0 and awake[nodes[i]] == 0:
        mask[i] = 0


@wp.kernel(enable_backward=False)
def _limit_coordinates(nodes: wp.array[int], awake: wp.array[int], source: wp.array[int], target: wp.array[int]):
    i = wp.tid()
    target[i] = source[i]
    if nodes[i] >= 0 and awake[nodes[i]] == 0:
        target[i] = -1


def _entity_arrays(model, frequency, size):
    """Model arrays declared at ``frequency`` with one row per entity, so coincident counts never mix owners."""
    arrays = []
    for name, spec in model.attribute_specs.items():
        value = vars(model).get(name)
        if (
            spec.frequency != frequency
            or spec.assignment not in (None, Model.AttributeAssignment.MODEL)
            or spec.deprecated
        ):
            continue
        if isinstance(value, wp.array) and value.ndim >= 1 and value.shape[0] == size and size > 0:
            arrays.append((name, value))
    return arrays


def _isnan(values):
    return np.isnan(values) if np.issubdtype(values.dtype, np.floating) else np.zeros(values.shape, dtype=bool)
