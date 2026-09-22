# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Experimental spatial contact integration."""

from __future__ import annotations

import warp as wp

from ...core.types import mat36f, mat66f, vec6f
from .contact_cache import AngularContactCache
from .spatial_contact import (
    SpatialContactPrepared,
    SpatialContactResult,
    compute_spatial_contact_residual,
    prepare_spatial_contact,
    solve_spatial_contact,
)

wp.set_module_options({"enable_backward": False})


@wp.struct
class ContactLawData:
    """Device views for one prepared projection schedule."""

    enabled: wp.bool
    jacobian_first: wp.array[mat66f]
    jacobian_second: wp.array[mat66f]
    bias: wp.array[wp.float32]
    friction: wp.array[wp.vec3f]
    angular_reaction: wp.array[wp.vec3f]
    angular_trial: wp.array[wp.vec3f]
    angular_previous: wp.array[wp.vec3f]
    prepared: wp.array[SpatialContactPrepared]
    mechanical: wp.array[mat66f]
    status: wp.array[wp.int32]


@wp.func
def pack_contact_reaction(linear: wp.vec3f, angular: wp.vec3f) -> vec6f:
    return vec6f(linear[2], linear[0], linear[1], angular[0], angular[1], angular[2])


@wp.func
def linear_contact_reaction(value: vec6f) -> wp.vec3f:
    return wp.vec3f(value[1], value[2], value[0])


@wp.func
def angular_contact_reaction(value: vec6f) -> wp.vec3f:
    return wp.vec3f(value[3], value[4], value[5])


@wp.func
def gather_contact_velocity(
    contact: int, first: int, second: int, twist: wp.array[vec6f], data: ContactLawData
) -> vec6f:
    velocity = vec6f(0.0)
    if first >= 0:
        velocity += data.jacobian_first[contact] @ twist[first]
    if second >= 0:
        velocity += data.jacobian_second[contact] @ twist[second]
    return velocity


@wp.func
def project_extended_contact(
    contact: int, first: int, second: int, linear: wp.vec3f, twist: wp.array[vec6f], data: ContactLawData
) -> SpatialContactResult:
    old = pack_contact_reaction(linear, data.angular_reaction[contact])
    free = gather_contact_velocity(contact, first, second, twist, data) - data.mechanical[contact] @ old
    free[0] += data.bias[contact]
    result = solve_spatial_contact(data.prepared[contact], free)
    data.status[contact] = result.status
    return result


@wp.func
def clamp_contact_reaction(value: vec6f, friction: wp.vec3f) -> vec6f:
    # Clamp the friction budget while preserving the cached normal impulse.
    scale = vec6f(1.0, friction[0], friction[0], friction[1], friction[2], friction[2])
    result = value
    result[0] = wp.max(value[0], 0.0)
    norm_sq = float(0.0)
    for axis in range(1, 6):
        if scale[axis] > 0.0:
            norm_sq += (value[axis] / scale[axis]) ** 2.0
        else:
            result[axis] = 0.0
    factor = wp.min(1.0, result[0] / wp.max(wp.sqrt(norm_sq), 1.0e-30))
    for axis in range(1, 6):
        result[axis] *= factor
    return result


@wp.kernel
def _initialize_contacts(
    source_count: wp.array[wp.int32],
    source_map: wp.array[wp.int32],
    source_frame: wp.array[wp.quatf],
    source_material: wp.array[wp.vec2f],
    source_angular_friction: wp.array[wp.vec2f],
    world: wp.array[wp.int32],
    first: wp.array[wp.int32],
    second: wp.array[wp.int32],
    linear_jacobian_first: wp.array[mat36f],
    linear_jacobian_second: wp.array[mat36f],
    legacy_bias: wp.array[wp.vec3f],
    spatial_friction: bool,
    data: ContactLawData,
):
    source = wp.tid()
    if source >= source_count[0]:
        return
    contact = source_map[source]
    if contact < 0:
        return
    rotation = wp.quat_to_matrix(source_frame[source])
    ja = mat66f(0.0)
    jb = mat66f(0.0)
    for row in range(3):
        local = (row + 2) % 3
        for col in range(6):
            ja[row, col] = linear_jacobian_first[contact][local, col]
            jb[row, col] = linear_jacobian_second[contact][local, col]
        for axis in range(3):
            if first[contact] >= 0:
                ja[row + 3, axis + 3] = -rotation[axis, local]
            if second[contact] >= 0:
                jb[row + 3, axis + 3] = rotation[axis, local]
    data.jacobian_first[contact] = ja
    data.jacobian_second[contact] = jb
    material = source_material[source]
    angular = wp.vec2f(0.0)
    if spatial_friction:
        angular = source_angular_friction[source]
    data.friction[contact] = wp.vec3f(material[0], angular[0], angular[1])
    data.bias[contact] = legacy_bias[contact][2]
    data.status[contact] = 0


@wp.kernel
def _prepare_contacts(
    world: wp.array[wp.int32],
    local: wp.array[wp.int32],
    counts: wp.array[wp.int32],
    first: wp.array[wp.int32],
    second: wp.array[wp.int32],
    inverse_weight: wp.array[mat66f],
    multiplicity: wp.array[wp.int32],
    static_multiplicity: wp.array[wp.int32],
    occupancy: wp.array2d[wp.int32],
    colors: wp.array[wp.int32],
    mode: int,
    data: ContactLawData,
    world_status: wp.array[wp.int32],
):
    contact = wp.tid()
    wid = world[contact]
    if local[contact] >= counts[wid]:
        return
    d = mat66f(0.0)
    for endpoint in range(2):
        body = first[contact]
        jacobian = data.jacobian_first[contact]
        if endpoint == 1:
            body = second[contact]
            jacobian = data.jacobian_second[contact]
        if body >= 0:
            m = int(1)
            if mode == 1:
                m = wp.max(1, multiplicity[body] - static_multiplicity[body])
            elif mode == 2:
                m = wp.max(1, occupancy[body, colors[contact]])
            d += float(m) * (jacobian @ inverse_weight[body] @ wp.transpose(jacobian))
    data.mechanical[contact] = d
    prepared = prepare_spatial_contact(d, data.friction[contact])
    data.prepared[contact] = prepared
    data.status[contact] = prepared.status
    if prepared.status != 0:
        world_status[wid] = 0


@wp.kernel
def warmstart_extended_contacts(
    world: wp.array[wp.int32],
    local: wp.array[wp.int32],
    counts: wp.array[wp.int32],
    active: wp.array[wp.bool],
    status: wp.array[wp.int32],
    first: wp.array[wp.int32],
    second: wp.array[wp.int32],
    inverse_weight: wp.array[mat66f],
    weighted: bool,
    linear_reaction: wp.array[wp.vec3f],
    data: ContactLawData,
    delta: wp.array[vec6f],
):
    contact = wp.tid()
    wid = world[contact]
    if local[contact] >= counts[wid] or not active[wid] or status[wid] != 1:
        return
    reaction = clamp_contact_reaction(
        pack_contact_reaction(linear_reaction[contact], data.angular_reaction[contact]), data.friction[contact]
    )
    linear_reaction[contact] = linear_contact_reaction(reaction)
    data.angular_reaction[contact] = angular_contact_reaction(reaction)
    for endpoint in range(2):
        body = first[contact]
        jacobian = data.jacobian_first[contact]
        if endpoint == 1:
            body = second[contact]
            jacobian = data.jacobian_second[contact]
        if body >= 0:
            value = wp.transpose(jacobian) @ reaction
            if weighted:
                value = inverse_weight[body] @ value
            wp.atomic_add(delta, body, value)


@wp.kernel
def _compute_residuals(
    world: wp.array[wp.int32],
    local: wp.array[wp.int32],
    counts: wp.array[wp.int32],
    active: wp.array[wp.bool],
    status: wp.array[wp.int32],
    first: wp.array[wp.int32],
    second: wp.array[wp.int32],
    linear_reaction: wp.array[wp.vec3f],
    twist: wp.array[vec6f],
    data: ContactLawData,
    physical: wp.array[SpatialContactPrepared],
    velocity: wp.array[wp.vec3f],
    residual: wp.array[wp.float32],
    maximum: wp.array[wp.float32],
):
    contact = wp.tid()
    wid = world[contact]
    if local[contact] >= counts[wid] or not active[wid] or status[wid] != 1:
        return
    reaction = pack_contact_reaction(linear_reaction[contact], data.angular_reaction[contact])
    c = gather_contact_velocity(contact, first[contact], second[contact], twist, data)
    velocity[contact] = linear_contact_reaction(c)
    c[0] += data.bias[contact]
    value = wp.max(wp.abs(compute_spatial_contact_residual(physical[contact], reaction, c)))
    if not wp.isfinite(value):
        status[wid] = 0
        value = 1.0e30
    residual[contact] = value
    wp.atomic_max(maximum, wid, value)


@wp.kernel
def _extend_convergence(
    world: wp.array[wp.int32],
    local: wp.array[wp.int32],
    counts: wp.array[wp.int32],
    active: wp.array[wp.bool],
    status: wp.array[wp.int32],
    first: wp.array[wp.int32],
    second: wp.array[wp.int32],
    global_twist: wp.array[vec6f],
    previous_twist: wp.array[vec6f],
    data: ContactLawData,
    law_residual: wp.array[wp.float32],
    tolerance: float,
    residual: wp.array[wp.float32],
    required: wp.array[wp.int32],
):
    contact = wp.tid()
    wid = world[contact]
    if local[contact] >= counts[wid] or not active[wid] or status[wid] != 1:
        return
    prepared = data.prepared[contact]
    trace = prepared.normal_delassus
    active_count = float(1.0)
    for axis in range(1, 6):
        coefficient = prepared.scaling[axis]
        if coefficient > 0.0:
            trace += coefficient * coefficient * prepared.delassus[axis, axis]
            active_count += 1.0
    # Undo the natural map's velocity scaling before comparing to the
    # configured velocity tolerance; angular rows use their friction lengths.
    scale = wp.sqrt(wp.max(trace / active_count, 1.0e-30))
    value = law_residual[contact] * scale / tolerance
    current = gather_contact_velocity(contact, first[contact], second[contact], global_twist, data)
    previous = gather_contact_velocity(contact, first[contact], second[contact], previous_twist, data)
    for axis in range(3, 6):
        if prepared.scaling[axis] > 0.0:
            value = wp.max(value, wp.abs(current[axis] - previous[axis]) / tolerance)
    wp.atomic_max(residual, wid, value)
    wp.atomic_max(required, wid, 1)


@wp.kernel
def _write_angular_outputs(
    world: wp.array[wp.int32],
    local: wp.array[wp.int32],
    counts: wp.array[wp.int32],
    first: wp.array[wp.int32],
    second: wp.array[wp.int32],
    inverse_time_step: wp.array[wp.float32],
    data: ContactLawData,
    wrench: wp.array[vec6f],
):
    contact = wp.tid()
    wid = world[contact]
    if local[contact] >= counts[wid]:
        return
    angular = data.angular_reaction[contact] * inverse_time_step[wid]
    reaction = pack_contact_reaction(wp.vec3f(0.0), angular)
    if first[contact] >= 0:
        wp.atomic_add(wrench, first[contact], wp.transpose(data.jacobian_first[contact]) @ reaction)
    if second[contact] >= 0:
        wp.atomic_add(wrench, second[contact], wp.transpose(data.jacobian_second[contact]) @ reaction)


@wp.kernel
def _reset_contacts(world: wp.array[wp.int32], mask: wp.array[wp.bool], data: ContactLawData):
    contact = wp.tid()
    if not mask or mask[world[contact]]:
        data.angular_reaction[contact] = wp.vec3f(0.0)
        data.angular_trial[contact] = wp.vec3f(0.0)
        data.angular_previous[contact] = wp.vec3f(0.0)
        data.status[contact] = 0


class ContactLaw:
    """Own optional contact state without increasing default solver storage."""

    def __init__(self, problem, config):
        self.problem = problem
        self.config = config
        self.data = ContactLawData()
        data = self.data
        data.enabled = True
        for name, dtype in (
            ("jacobian_first", mat66f),
            ("jacobian_second", mat66f),
            ("bias", wp.float32),
            ("friction", wp.vec3f),
            ("angular_reaction", wp.vec3f),
            ("angular_trial", wp.vec3f),
            ("angular_previous", wp.vec3f),
            ("status", wp.int32),
        ):
            setattr(data, name, wp.zeros(problem.contact_capacity, dtype=dtype, device=problem.device))
        self.jacobi = self._schedule_data()
        self.colored = self._schedule_data()
        self.physical = self._schedule_data()
        self.cache = AngularContactCache(
            problem.contacts.model_max_contacts_host
            if config.contact_spatial_friction and problem.contacts is not None
            else 0,
            problem.device,
        )

    def _schedule_data(self):
        data = ContactLawData()
        for name in ContactLawData.vars:
            setattr(data, name, getattr(self.data, name))
        data.prepared = wp.zeros(
            self.problem.contact_capacity, dtype=SpatialContactPrepared, device=self.problem.device
        )
        data.mechanical = wp.zeros(self.problem.contact_capacity, dtype=mat66f, device=self.problem.device)
        return data

    def initialize(self, time_step, begin_step):
        p = self.problem
        if begin_step:
            self.data.angular_reaction.zero_()
            self.data.angular_trial.zero_()
            self.data.angular_previous.zero_()
            self.data.status.zero_()
        if p.contacts is None or p.contact_capacity == 0:
            return
        wp.launch(
            _initialize_contacts,
            dim=p.contacts.model_max_contacts_host,
            inputs=[
                p.contacts.model_active_contacts,
                p.contact_source_to_internal,
                p.contacts.frame,
                p.contacts.material,
                p.contacts.angular_friction,
                p.contact_world,
                p.contact_body_first,
                p.contact_body_second,
                p.contact_jacobian_first,
                p.contact_jacobian_second,
                p.contact_bias,
                self.config.contact_spatial_friction,
                self.data,
            ],
            device=p.device,
        )

        if begin_step:
            self.cache.import_reactions(
                p.contacts,
                p.contact_source_to_internal,
                p.data.bodies.q_i,
                time_step,
                self.data.angular_reaction,
            )

    def prepare(self, inverse_weight, status, *, colored=None):
        p = self.problem
        if p.contact_capacity == 0:
            return
        schedules = [(0, self.physical), (1, self.jacobi)] if colored is None else [(2, self.colored)]
        for mode, data in schedules:
            wp.launch(
                _prepare_contacts,
                dim=p.contact_capacity,
                inputs=[
                    p.contact_world,
                    p.contact_local,
                    p.world_contact_count,
                    p.contact_body_first,
                    p.contact_body_second,
                    inverse_weight,
                    p.body_constraint_count,
                    p.static_body_constraint_count,
                    colored.body_occupancy if colored is not None else None,
                    colored.contact.colors if colored is not None else None,
                    mode,
                    data,
                    status,
                ],
                device=p.device,
            )

    def residuals(self, active, twist):
        p = self.problem
        p.world_contact_residual_max.zero_()
        wp.launch(
            _compute_residuals,
            dim=p.contact_capacity,
            inputs=[
                p.contact_world,
                p.contact_local,
                p.world_contact_count,
                active,
                p.projection_status,
                p.contact_body_first,
                p.contact_body_second,
                p.contact_reaction,
                twist,
                self.jacobi,
                self.physical.prepared,
                p.contact_velocity,
                p.contact_residual,
                p.world_contact_residual_max,
            ],
            device=p.device,
        )

    def finish_iteration(self, active, global_twist, previous_twist, projected_twist, tolerance):
        p = self.problem
        self.residuals(active, projected_twist)
        wp.launch(
            _extend_convergence,
            dim=p.contact_capacity,
            inputs=[
                p.contact_world,
                p.contact_local,
                p.world_contact_count,
                active,
                p.projection_status,
                p.contact_body_first,
                p.contact_body_second,
                global_twist,
                previous_twist,
                self.physical,
                p.contact_residual,
                tolerance,
                p.world_lagged_velocity_residual,
                p.world_lagged_velocity_required,
            ],
            device=p.device,
        )

    def write_outputs(self, inverse_time_step, wrench):
        p = self.problem
        wp.launch(
            _write_angular_outputs,
            dim=p.contact_capacity,
            inputs=[
                p.contact_world,
                p.contact_local,
                p.world_contact_count,
                p.contact_body_first,
                p.contact_body_second,
                inverse_time_step,
                self.data,
                wrench,
            ],
            device=p.device,
        )

        if p.contacts is not None:
            self.cache.export_reactions(
                p.contacts, p.contact_source_to_internal, inverse_time_step, self.data.angular_reaction
            )

    def reset(self, world_mask=None):
        p = self.problem
        self.cache.reset(world_mask)
        wp.launch(
            _reset_contacts, dim=p.contact_capacity, inputs=[p.contact_world, world_mask, self.data], device=p.device
        )
