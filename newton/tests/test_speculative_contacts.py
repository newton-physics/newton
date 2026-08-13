# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for velocity-expanded rigid-contact candidate generation."""

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.geometry.contact_reduction_global import (
    EXPORT_REDUCED_CONTACTS_BLOCK_DIM,
    PREDICTIVE_BIN_ID,
    GlobalContactReducer,
    GlobalContactReducerData,
    create_export_reduced_contacts_kernel,
    export_and_reduce_contact_centered_two_spatial_depths,
    export_and_reduce_predictive_contact,
    export_contact_to_buffer,
    make_contact_value,
    reclaim_contact_id,
    reduce_buffered_contacts_speculative_kernel,
    reduction_finalize_slot,
)
from newton._src.geometry.narrow_phase import ContactWriterData, NarrowPhase, write_contact_simple
from newton._src.geometry.types import GeoType
from newton.tests.unittest_utils import add_function_test, get_cuda_test_devices, get_test_devices


@wp.kernel
def _register_regular_and_predictive_contact(
    reducer_data: GlobalContactReducerData,
    shape_transform: wp.array[wp.transform],
    shape_linear_velocity: wp.array[wp.vec3],
    shape_angular_velocity: wp.array[wp.vec3],
    contact_ids: wp.array[wp.int32],
):
    position = wp.vec3(0.0)
    normal = wp.vec3(1.0, 0.0, 0.0)
    contact_id = export_and_reduce_contact_centered_two_spatial_depths(
        0,
        1,
        position,
        normal,
        0.05,
        17,
        position,
        0.0,
        0.1,
        position,
        wp.vec3(-1.0),
        wp.vec3(1.0),
        wp.vec3i(1),
        reducer_data,
    )
    contact_ids[0] = contact_id
    contact_ids[1] = export_and_reduce_predictive_contact(
        0,
        1,
        position,
        normal,
        0.05,
        0.0,
        0.0,
        0.0,
        17,
        shape_transform,
        shape_linear_velocity,
        shape_angular_velocity,
        0.1,
        0.1,
        contact_id,
        reducer_data,
    )
    contact_ids[2] = export_and_reduce_predictive_contact(
        0,
        1,
        position,
        normal,
        0.05,
        0.0,
        0.0,
        0.0,
        17,
        shape_transform,
        shape_linear_velocity,
        shape_angular_velocity,
        0.1,
        0.1,
        contact_id,
        reducer_data,
    )


@wp.kernel
def _register_inner_and_rotating_leading_contact(
    reducer_data: GlobalContactReducerData,
    shape_transform: wp.array[wp.transform],
    shape_linear_velocity: wp.array[wp.vec3],
    shape_angular_velocity: wp.array[wp.vec3],
    contact_ids: wp.array[wp.int32],
):
    normal = wp.vec3(1.0, 0.0, 0.0)
    inner_position = wp.vec3(0.0)
    contact_ids[0] = export_and_reduce_contact_centered_two_spatial_depths(
        0,
        1,
        inner_position,
        normal,
        -0.01,
        11,
        inner_position,
        0.01,
        0.1,
        inner_position,
        wp.vec3(-2.0),
        wp.vec3(2.0),
        wp.vec3i(1),
        reducer_data,
    )

    leading_position = wp.vec3(0.0, 1.0, 0.0)
    regular_contact_id = export_and_reduce_contact_centered_two_spatial_depths(
        0,
        1,
        leading_position,
        normal,
        0.05,
        23,
        leading_position,
        0.01,
        0.1,
        leading_position,
        wp.vec3(-2.0),
        wp.vec3(2.0),
        wp.vec3i(1),
        reducer_data,
    )
    contact_ids[1] = regular_contact_id
    contact_ids[2] = export_and_reduce_predictive_contact(
        0,
        1,
        leading_position,
        normal,
        0.05,
        0.0,
        0.0,
        0.0,
        23,
        shape_transform,
        shape_linear_velocity,
        shape_angular_velocity,
        0.1,
        0.1,
        regular_contact_id,
        reducer_data,
    )


@wp.kernel
def _register_predictive_clearance_candidates(
    reducer_data: GlobalContactReducerData,
    shape_transform: wp.array[wp.transform],
    shape_linear_velocity: wp.array[wp.vec3],
    shape_angular_velocity: wp.array[wp.vec3],
    contact_ids: wp.array[wp.int32],
):
    normal = wp.vec3(1.0, 0.0, 0.0)
    for candidate_idx in range(8):
        clearance = float(0.07)
        fingerprint = int(7)
        position_y = float(0.0)
        if candidate_idx == 1:
            clearance = 0.06
            fingerprint = 3
        elif candidate_idx == 2:
            clearance = 0.05
            fingerprint = 14
        elif candidate_idx == 3:
            clearance = 0.04
            fingerprint = 2
        elif candidate_idx == 4:
            clearance = 0.03
            fingerprint = 1
        elif candidate_idx == 5:
            clearance = 0.02
            fingerprint = 16
        elif candidate_idx == 6:
            clearance = 0.01
            fingerprint = 12
        elif candidate_idx == 7:
            clearance = 0.08
            fingerprint = 18
            position_y = 10.0
        contact_ids[candidate_idx] = export_and_reduce_predictive_contact(
            0,
            1,
            wp.vec3(0.0, position_y, clearance),
            normal,
            clearance,
            0.0,
            0.0,
            0.0,
            fingerprint,
            shape_transform,
            shape_linear_velocity,
            shape_angular_velocity,
            0.1,
            0.1,
            -1,
            reducer_data,
        )


@wp.kernel
def _register_predictive_clearance_candidates_contended(
    reducer_data: GlobalContactReducerData,
    shape_transform: wp.array[wp.transform],
    shape_linear_velocity: wp.array[wp.vec3],
    shape_angular_velocity: wp.array[wp.vec3],
    clearance_order: int,
):
    candidate_idx = wp.tid()
    clearance_rank = candidate_idx
    if clearance_order == 1:
        clearance_rank = 254 - wp.min(candidate_idx, 254)
    elif clearance_order == 2:
        clearance_rank = (candidate_idx * 73) % 255
    clearance = 0.01 + float(clearance_rank) * 0.0001
    fingerprint = (candidate_idx << 2) | ((candidate_idx & 1) << 1)
    position_y = 0.0
    if candidate_idx == 255:
        clearance = 0.08
        position_y = 10.0
    export_and_reduce_predictive_contact(
        0,
        1,
        wp.vec3(0.0, position_y, clearance),
        wp.vec3(1.0, 0.0, 0.0),
        clearance,
        0.0,
        0.0,
        0.0,
        fingerprint,
        shape_transform,
        shape_linear_velocity,
        shape_angular_velocity,
        0.1,
        0.1,
        -1,
        reducer_data,
    )


@wp.kernel
def _buffer_one_contact(reducer_data: GlobalContactReducerData):
    export_contact_to_buffer(
        0,
        1,
        wp.vec3(0.25, -0.5, 0.75),
        wp.vec3(1.0, 0.0, 0.0),
        -0.01,
        7,
        reducer_data,
    )


@wp.kernel
def _buffer_separated_axial_contact(reducer_data: GlobalContactReducerData):
    """Buffer one separated axial-shape contact produced by a mesh triangle."""
    export_contact_to_buffer(
        0,
        1,
        wp.vec3(0.0),
        wp.vec3(1.0, 0.0, 0.0),
        0.05,
        7,
        reducer_data,
    )


@wp.kernel
def _replace_full_buffer_predictive_winner(
    reducer_data: GlobalContactReducerData,
    shape_transform: wp.array[wp.transform],
    shape_linear_velocity: wp.array[wp.vec3],
    shape_angular_velocity: wp.array[wp.vec3],
    contact_ids: wp.array[wp.int32],
):
    normal = wp.vec3(1.0, 0.0, 0.0)
    fingerprint = int(7)
    original_clearance = 0.08
    original_contact_id = export_contact_to_buffer(
        0,
        1,
        wp.vec3(0.0, 0.0, original_clearance),
        normal,
        original_clearance,
        fingerprint,
        reducer_data,
    )
    contact_ids[0] = export_and_reduce_predictive_contact(
        0,
        1,
        wp.vec3(0.0, 0.0, original_clearance),
        normal,
        original_clearance,
        0.0,
        0.0,
        0.0,
        fingerprint,
        shape_transform,
        shape_linear_velocity,
        shape_angular_velocity,
        0.1,
        0.1,
        original_contact_id,
        reducer_data,
    )
    contact_ids[1] = export_and_reduce_predictive_contact(
        0,
        1,
        wp.vec3(0.0, 0.0, 0.05),
        normal,
        0.05,
        0.0,
        0.0,
        0.0,
        fingerprint,
        shape_transform,
        shape_linear_velocity,
        shape_angular_velocity,
        0.1,
        0.1,
        -1,
        reducer_data,
    )


@wp.kernel
def _replace_validated_predictive_claims(
    reducer_data: GlobalContactReducerData,
    allocated_ids: wp.array[wp.int32],
):
    entry_idx = int(0)
    clearance_slot = int(0)
    impact_slot = int(6)
    fingerprint = int(7)
    provisional_clearance = make_contact_value(-0.1, fingerprint, 0, reducer_data.deterministic)
    provisional_impact = make_contact_value(0.1, fingerprint, 0, reducer_data.deterministic)
    clearance_idx = clearance_slot * reducer_data.ht_capacity + entry_idx
    impact_idx = impact_slot * reducer_data.ht_capacity + entry_idx
    reducer_data.ht_values[clearance_idx] = provisional_clearance
    reducer_data.ht_values[impact_idx] = provisional_impact

    contact_id = export_contact_to_buffer(
        0,
        1,
        wp.vec3(0.0),
        wp.vec3(1.0, 0.0, 0.0),
        0.1,
        fingerprint,
        reducer_data,
    )

    # Reproduce two stronger contenders replacing both claims after validation.
    reducer_data.ht_values[clearance_idx] = make_contact_value(0.0, fingerprint + 1, 0, reducer_data.deterministic)
    reducer_data.ht_values[impact_idx] = make_contact_value(0.2, fingerprint + 1, 0, reducer_data.deterministic)
    clearance_final = make_contact_value(-0.1, fingerprint, contact_id, reducer_data.deterministic)
    impact_final = make_contact_value(0.1, fingerprint, contact_id, reducer_data.deterministic)
    retained = reduction_finalize_slot(
        entry_idx,
        clearance_slot,
        provisional_clearance,
        clearance_final,
        reducer_data.ht_values,
        reducer_data.ht_capacity,
    )
    if reduction_finalize_slot(
        entry_idx,
        impact_slot,
        provisional_impact,
        impact_final,
        reducer_data.ht_values,
        reducer_data.ht_capacity,
    ):
        retained = True
    if not retained:
        reclaim_contact_id(contact_id, reducer_data)

    for i in range(reducer_data.capacity):
        allocated_ids[i] = export_contact_to_buffer(
            0,
            1,
            wp.vec3(float(i), 0.0, 0.0),
            wp.vec3(1.0, 0.0, 0.0),
            0.0,
            i,
            reducer_data,
        )


@wp.kernel
def _reclaim_and_allocate_contact_ids(
    reducer_data: GlobalContactReducerData,
    allocated_ids: wp.array[wp.int32],
):
    """Reclaim one ID per thread and immediately contend for the available IDs."""
    tid = wp.tid()
    reclaim_contact_id(tid + 1, reducer_data)
    allocated_ids[tid] = export_contact_to_buffer(
        0,
        1,
        wp.vec3(float(tid), 0.0, 0.0),
        wp.vec3(1.0, 0.0, 0.0),
        0.0,
        tid,
        reducer_data,
    )


@wp.kernel
def _register_predictive_clearance_candidates_sequential(
    reducer_data: GlobalContactReducerData,
    shape_transform: wp.array[wp.transform],
    shape_linear_velocity: wp.array[wp.vec3],
    shape_angular_velocity: wp.array[wp.vec3],
):
    candidate_idx = int(0)
    while candidate_idx < 256:
        clearance_rank = (candidate_idx * 73) % 255
        clearance = 0.01 + float(clearance_rank) * 0.0001
        fingerprint = (candidate_idx << 2) | ((candidate_idx & 1) << 1)
        position_y = 0.0
        if candidate_idx == 255:
            clearance = 0.08
            position_y = 10.0
        export_and_reduce_predictive_contact(
            0,
            1,
            wp.vec3(0.0, position_y, clearance),
            wp.vec3(1.0, 0.0, 0.0),
            clearance,
            0.0,
            0.0,
            0.0,
            fingerprint,
            shape_transform,
            shape_linear_velocity,
            shape_angular_velocity,
            0.1,
            0.1,
            -1,
            reducer_data,
        )
        candidate_idx = candidate_idx + 1


def _expected_contended_clearances(clearance_order: int) -> list[float]:
    """Return the nearest clearance in each fingerprint shard plus the impact guard."""
    nearest_by_shard: dict[int, float] = {}
    for candidate_idx in range(255):
        if clearance_order == 0:
            clearance_rank = candidate_idx
        elif clearance_order == 1:
            clearance_rank = 254 - candidate_idx
        else:
            clearance_rank = (candidate_idx * 73) % 255
        clearance = 0.01 + clearance_rank * 0.0001
        fingerprint = (candidate_idx << 2) | ((candidate_idx & 1) << 1)
        shard_hash = fingerprint
        shard_hash ^= shard_hash >> 16
        shard_hash = (shard_hash * 0x7FEB352D) & 0xFFFFFFFF
        shard_hash ^= shard_hash >> 15
        shard_hash = (shard_hash * 0x846CA68B) & 0xFFFFFFFF
        shard_hash ^= shard_hash >> 16
        shard = shard_hash % 6
        nearest_by_shard[shard] = min(clearance, nearest_by_shard.get(shard, float("inf")))
    return sorted([*nearest_by_shard.values(), 0.08])


_export_reduced_contacts = create_export_reduced_contacts_kernel(write_contact_simple)


def _make_predictive_reducer(capacity, device, deterministic):
    """Create reducer storage for predictive reservation tests."""
    return GlobalContactReducer(
        capacity=capacity,
        device=device,
        deterministic=deterministic,
        enable_contact_reclamation=True,
    )


def _build_spheres(device, velocity: float, separation: float = 0.3, gap: float = 0.0):
    """Build two spheres separated along X with the first sphere moving."""
    builder = newton.ModelBuilder(gravity=wp.vec3(0.0))
    builder.rigid_gap = gap
    body_a = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.0)))
    builder.add_shape_sphere(body_a, radius=0.1)
    builder.body_qd[body_a] = (velocity, 0.0, 0.0, 0.0, 0.0, 0.0)
    body_b = builder.add_body(xform=wp.transform(wp.vec3(separation, 0.0, 0.0)))
    builder.add_shape_sphere(body_b, radius=0.1)
    model = builder.finalize(device=device)
    return model, model.state()


def _collide(model, state, speculative: bool):
    """Run one collision pass and return the populated contact buffer."""
    config = None
    if speculative:
        config = newton.CollisionPipeline.SpeculativeContactConfig(
            max_speculative_extension=0.25,
        )
    pipeline = newton.CollisionPipeline(model, broad_phase="nxn", speculative_config=config)
    contacts = pipeline.contacts()
    pipeline.collide(state, contacts, dt=0.02)
    return contacts


def _export_reducer_contacts(reducer: GlobalContactReducer, device):
    """Export reducer winners and return their count and world positions."""
    contact_count = wp.zeros(1, dtype=wp.int32, device=device)
    contact_position = wp.zeros(8, dtype=wp.vec3, device=device)
    writer_data = ContactWriterData()
    writer_data.contact_max = 8
    writer_data.contact_count = contact_count
    writer_data.contact_pair = wp.zeros(8, dtype=wp.vec2i, device=device)
    writer_data.contact_position = contact_position
    writer_data.contact_normal = wp.zeros(8, dtype=wp.vec3, device=device)
    writer_data.contact_penetration = wp.zeros(8, dtype=wp.float32, device=device)
    writer_data.contact_tangent = wp.empty(0, dtype=wp.vec3, device=device)
    writer_data.contact_sort_key = wp.empty(0, dtype=wp.int64, device=device)
    shape_gap = wp.full(2, 0.1, dtype=wp.float32, device=device)
    writer_data.shape_gap = shape_gap
    writer_data.shape_transform = wp.empty(0, dtype=wp.transform, device=device)
    writer_data.shape_linear_velocity = wp.empty(0, dtype=wp.vec3, device=device)
    writer_data.shape_angular_velocity = wp.empty(0, dtype=wp.vec3, device=device)
    writer_data.collision_update_dt = 0.0
    writer_data.max_speculative_extension = 0.0
    reducer.exported_flags.zero_()
    total_blocks = 128
    wp.launch_tiled(
        _export_reduced_contacts,
        dim=total_blocks,
        inputs=[
            reducer.hashtable.keys,
            reducer.ht_values,
            reducer.hashtable.active_slots,
            reducer.position_depth,
            reducer.normal,
            reducer.shape_pairs,
            reducer.contact_fingerprints,
            reducer.exported_flags,
            wp.zeros(2, dtype=wp.int32, device=device),
            wp.zeros(2, dtype=wp.vec4, device=device),
            shape_gap,
            writer_data,
            total_blocks,
            int(not device.is_cpu),
            int(reducer.deterministic),
        ],
        device=device,
        block_dim=EXPORT_REDUCED_CONTACTS_BLOCK_DIM,
    )
    return int(contact_count.numpy()[0]), contact_position.numpy()


def test_speculative_candidates_are_opt_in(test, device):
    """Verify separated approaching shapes only emit a candidate when enabled."""
    model, state = _build_spheres(device, velocity=10.0)
    test.assertEqual(int(_collide(model, state, speculative=False).rigid_contact_count.numpy()[0]), 0)
    test.assertGreater(int(_collide(model, state, speculative=True).rigid_contact_count.numpy()[0]), 0)


def test_speculative_candidates_require_approach(test, device):
    """Verify separated stationary and diverging shapes emit no candidate."""
    for velocity in (0.0, -10.0):
        model, state = _build_spheres(device, velocity=velocity)
        contacts = _collide(model, state, speculative=True)
        test.assertEqual(int(contacts.rigid_contact_count.numpy()[0]), 0)


def test_speculative_candidates_require_dt(test, device):
    """Require a current horizon and suppress candidates that cannot reach it."""
    model, state = _build_spheres(device, velocity=10.0)
    config = newton.CollisionPipeline.SpeculativeContactConfig(
        max_speculative_extension=0.25,
    )
    pipeline = newton.CollisionPipeline(model, broad_phase="nxn", speculative_config=config)

    contacts = pipeline.contacts()
    with test.assertRaisesRegex(ValueError, "dt must be provided"):
        pipeline.collide(state, contacts)

    pipeline.collide(state, contacts, dt=0.02)
    test.assertGreater(int(contacts.rigid_contact_count.numpy()[0]), 0)

    pipeline.collide(state, contacts, dt=0.005)
    test.assertEqual(int(contacts.rigid_contact_count.numpy()[0]), 0)


def test_speculative_gap_uses_larger_fixed_or_velocity_distance(test, device):
    """Use the larger fixed or velocity-based gap without adding them."""
    model, state = _build_spheres(device, velocity=1.0, separation=0.33, gap=0.05)
    pipeline = newton.CollisionPipeline(
        model,
        broad_phase="nxn",
        speculative_config=newton.CollisionPipeline.SpeculativeContactConfig(max_speculative_extension=0.25),
    )

    contacts = pipeline.contacts()
    pipeline.collide(state, contacts, dt=0.05)
    test.assertEqual(int(contacts.rigid_contact_count.numpy()[0]), 0)

    pipeline.collide(state, contacts, dt=0.15)
    test.assertGreater(int(contacts.rigid_contact_count.numpy()[0]), 0)


def test_speculative_candidates_reject_invalid_dt_override(test, device):
    """Reject negative and non-finite per-call speculative horizons."""
    model, state = _build_spheres(device, velocity=10.0)
    pipeline = newton.CollisionPipeline(
        model,
        broad_phase="nxn",
        speculative_config=newton.CollisionPipeline.SpeculativeContactConfig(),
    )
    contacts = pipeline.contacts()
    for dt in (-0.01, float("nan"), float("inf"), float("-inf")):
        with test.subTest(dt=dt), test.assertRaisesRegex(ValueError, "dt must be a non-negative finite number"):
            pipeline.collide(state, contacts, dt=dt)


def test_speculative_candidates_reject_common_motion(test, device):
    """Reject separated shapes whose large common motion makes their swept unions overlap."""
    builder = newton.ModelBuilder(gravity=wp.vec3(0.0))
    builder.rigid_gap = 0.0
    body_a = builder.add_body(xform=wp.transform_identity())
    builder.add_shape_sphere(body_a, radius=0.1)
    builder.body_qd[body_a] = (20.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    body_b = builder.add_body(xform=wp.transform(wp.vec3(0.4, 0.0, 0.0)))
    builder.add_shape_sphere(body_b, radius=0.1)
    builder.body_qd[body_b] = (20.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    model = builder.finalize(device=device)
    shape_pairs = wp.array([wp.vec2i(0, 1)], dtype=wp.vec2i, device=device)
    config = newton.CollisionPipeline.SpeculativeContactConfig(
        max_speculative_extension=0.25,
    )

    for broad_phase in ("nxn", "sap", "explicit"):
        with test.subTest(broad_phase=broad_phase):
            pipeline = newton.CollisionPipeline(
                model,
                broad_phase=broad_phase,
                shape_pairs_filtered=shape_pairs if broad_phase == "explicit" else None,
                speculative_config=config,
            )
            contacts = pipeline.contacts()
            pipeline.collide(model.state(), contacts, dt=0.1)
            test.assertEqual(int(pipeline.broad_phase_pair_count.numpy()[0]), 0)
            test.assertEqual(int(contacts.rigid_contact_count.numpy()[0]), 0)


def test_speculative_candidates_preserve_physical_geometry(test, device):
    """Verify candidate generation does not enlarge stored physical margins."""
    model, state = _build_spheres(device, velocity=10.0)
    contacts = _collide(model, state, speculative=True)
    test.assertGreater(int(contacts.rigid_contact_count.numpy()[0]), 0)
    test.assertAlmostEqual(float(contacts.rigid_contact_margin0.numpy()[0]), 0.1, places=6)
    test.assertAlmostEqual(float(contacts.rigid_contact_margin1.numpy()[0]), 0.1, places=6)


def test_speculative_candidates_include_angular_motion(test, device):
    """Verify an offset shape's angular motion expands and filters candidates."""
    builder = newton.ModelBuilder(gravity=wp.vec3(0.0))
    builder.rigid_gap = 0.0
    body = builder.add_body(
        xform=wp.transform_identity(),
        mass=1.0,
        inertia=wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
        lock_inertia=True,
    )
    builder.add_shape_sphere(body, radius=0.1, xform=wp.transform(wp.vec3(0.0, 1.0, 0.0)))
    builder.body_qd[body] = (0.0, 0.0, 0.0, 0.0, 0.0, -10.0)
    builder.add_shape_sphere(-1, radius=0.1, xform=wp.transform(wp.vec3(0.3, 1.0, 0.0)))
    model = builder.finalize(device=device)
    pipeline = newton.CollisionPipeline(
        model,
        broad_phase="nxn",
        speculative_config=newton.CollisionPipeline.SpeculativeContactConfig(
            max_speculative_extension=0.25,
        ),
    )
    contacts = pipeline.contacts()
    pipeline.collide(model.state(), contacts, dt=0.02)
    test.assertGreater(int(pipeline.broad_phase_pair_count.numpy()[0]), 0)
    test.assertGreater(int(contacts.rigid_contact_count.numpy()[0]), 0)


def test_speculative_cone_reaches_infinite_plane(test, device):
    """Retain a swept GJK candidate before its current bounds reach an infinite plane."""
    builder = newton.ModelBuilder(gravity=wp.vec3(0.0))
    builder.rigid_gap = 0.0
    body = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.5)))
    builder.add_shape_cone(body, radius=0.1, half_height=0.1)
    builder.body_qd[body] = (0.0, 0.0, -20.0, 0.0, 0.0, 0.0)
    builder.add_shape_plane(width=0.0, length=0.0)
    model = builder.finalize(device=device)
    pipeline = newton.CollisionPipeline(
        model,
        broad_phase="nxn",
        speculative_config=newton.CollisionPipeline.SpeculativeContactConfig(
            max_speculative_extension=0.75,
        ),
    )
    contacts = pipeline.contacts()
    pipeline.collide(model.state(), contacts, dt=0.03)

    test.assertGreater(int(pipeline.broad_phase_pair_count.numpy()[0]), 0)
    test.assertGreater(int(contacts.rigid_contact_count.numpy()[0]), 0)


def test_stationary_contacts_match_non_speculative_pipeline(test, device):
    """Match contacts for non-moving shapes with speculative generation on and off."""
    builder = newton.ModelBuilder(gravity=wp.vec3(0.0))
    builder.rigid_gap = 0.0
    body_a = builder.add_body()
    builder.add_shape_box(body_a, hx=0.1, hy=0.1, hz=0.1)
    body_b = builder.add_body(xform=wp.transform(wp.vec3(0.15, 0.0, 0.0)))
    builder.add_shape_box(
        body_b,
        hx=0.1,
        hy=0.1,
        hz=0.1,
    )
    model = builder.finalize(device=device)
    state = model.state()

    pipelines = (
        newton.CollisionPipeline(model, broad_phase="nxn", deterministic=True),
        newton.CollisionPipeline(
            model,
            broad_phase="nxn",
            deterministic=True,
            speculative_config=newton.CollisionPipeline.SpeculativeContactConfig(
                max_speculative_extension=0.25,
            ),
        ),
    )
    outputs = []
    for pipeline in pipelines:
        contacts = pipeline.contacts()
        pipeline.collide(state, contacts, dt=0.03)
        count = int(contacts.rigid_contact_count.numpy()[0])
        test.assertGreater(count, 0)
        outputs.append(
            (
                contacts.rigid_contact_shape0.numpy()[:count],
                contacts.rigid_contact_shape1.numpy()[:count],
                contacts.rigid_contact_point0.numpy()[:count],
                contacts.rigid_contact_point1.numpy()[:count],
                contacts.rigid_contact_normal.numpy()[:count],
                contacts.rigid_contact_margin0.numpy()[:count],
                contacts.rigid_contact_margin1.numpy()[:count],
            )
        )

    for regular, speculative in zip(*outputs, strict=True):
        np.testing.assert_allclose(speculative, regular, rtol=0.0, atol=1.0e-6)


def test_speculative_contacts_prevent_dynamic_tunneling(test, device):
    """Prevent a fast dynamic sphere from crossing an infinite plane in one XPBD step."""
    builder = newton.ModelBuilder(gravity=wp.vec3(0.0))
    builder.rigid_gap = 0.0
    body = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.5)))
    builder.add_shape_sphere(body, radius=0.05)
    builder.body_qd[body] = (0.0, 0.0, -20.0, 0.0, 0.0, 0.0)
    builder.add_shape_plane(width=0.0, length=0.0)
    model = builder.finalize(device=device)
    dt = 0.03

    def step(speculative):
        config = None
        if speculative:
            config = newton.CollisionPipeline.SpeculativeContactConfig(
                max_speculative_extension=0.75,
            )
        pipeline = newton.CollisionPipeline(model, broad_phase="nxn", speculative_config=config)
        contacts = pipeline.contacts()
        state_in = model.state()
        state_out = model.state()
        pipeline.collide(state_in, contacts, dt=dt)
        newton.solvers.SolverXPBD(model, iterations=5).step(state_in, state_out, None, contacts, dt)
        return float(state_out.body_q.numpy()[body, 2])

    test.assertLess(step(False), 0.0)
    test.assertGreaterEqual(step(True), 0.04)


def test_speculative_narrow_phase_launch(test, device):
    """Verify the public narrow-phase convenience API performs exact admission."""
    shape_transform = wp.array(
        [wp.transform_identity(), wp.transform(wp.vec3(0.3, 0.0, 0.0))],
        dtype=wp.transform,
        device=device,
    )
    shape_aabb_lower = wp.full(2, wp.vec3(-0.1), dtype=wp.vec3, device=device)
    shape_aabb_upper = wp.full(2, wp.vec3(0.1), dtype=wp.vec3, device=device)
    narrow_phase = NarrowPhase(
        max_candidate_pairs=1,
        reduce_contacts=False,
        device=device,
        shape_aabb_lower=shape_aabb_lower,
        shape_aabb_upper=shape_aabb_upper,
        shape_voxel_resolution=wp.full(2, wp.vec3i(1), dtype=wp.vec3i, device=device),
        has_meshes=False,
        contact_max=4,
        verify_buffers=False,
        speculative=True,
    )

    candidate_pair = wp.array([wp.vec2i(0, 1)], dtype=wp.vec2i, device=device)
    candidate_pair_count = wp.array([1], dtype=wp.int32, device=device)
    shape_types = wp.array([int(GeoType.SPHERE), int(GeoType.SPHERE)], dtype=wp.int32, device=device)
    shape_data = wp.array(
        [wp.vec4(0.1, 0.1, 0.1, 0.0), wp.vec4(0.1, 0.1, 0.1, 0.0)],
        dtype=wp.vec4,
        device=device,
    )
    shape_gap = wp.array([0.2, 0.0], dtype=wp.float32, device=device)
    shape_base_gap = wp.zeros(2, dtype=wp.float32, device=device)
    shape_angular_velocity = wp.zeros(2, dtype=wp.vec3, device=device)

    for velocity, expected_count in ((10.0, 1), (-10.0, 0)):
        contact_count = wp.zeros(1, dtype=wp.int32, device=device)
        narrow_phase.launch(
            candidate_pair=candidate_pair,
            candidate_pair_count=candidate_pair_count,
            shape_types=shape_types,
            shape_data=shape_data,
            shape_transform=shape_transform,
            shape_source=wp.zeros(2, dtype=wp.uint64, device=device),
            shape_sdf_index=wp.full(2, -1, dtype=wp.int32, device=device),
            shape_gap=shape_gap,
            shape_base_gap=shape_base_gap,
            shape_collision_radius=wp.full(2, 0.1, dtype=wp.float32, device=device),
            shape_flags=wp.zeros(2, dtype=wp.int32, device=device),
            shape_collision_aabb_lower=shape_aabb_lower,
            shape_collision_aabb_upper=shape_aabb_upper,
            shape_voxel_resolution=wp.full(2, wp.vec3i(1), dtype=wp.vec3i, device=device),
            contact_pair=wp.zeros(4, dtype=wp.vec2i, device=device),
            contact_position=wp.zeros(4, dtype=wp.vec3, device=device),
            contact_normal=wp.zeros(4, dtype=wp.vec3, device=device),
            contact_penetration=wp.zeros(4, dtype=wp.float32, device=device),
            contact_count=contact_count,
            contact_tangent=wp.empty(0, dtype=wp.vec3, device=device),
            shape_linear_velocity=wp.array([wp.vec3(velocity, 0.0, 0.0), wp.vec3(0.0)], dtype=wp.vec3, device=device),
            shape_angular_velocity=shape_angular_velocity,
            collision_update_dt=0.02,
            max_speculative_extension=0.25,
            device=device,
        )
        test.assertEqual(int(contact_count.numpy()[0]), expected_count)


def test_speculative_narrow_phase_rejects_hydroelastic(test, device):
    """Verify indexed hydroelastic writers cannot silently bypass exact admission."""
    with test.assertRaisesRegex(NotImplementedError, "does not yet support hydroelastic"):
        NarrowPhase(
            max_candidate_pairs=1,
            device=device,
            hydroelastic_sdf=object(),
            speculative=True,
        )


def test_speculative_narrow_phase_rejects_unmarked_custom_writer(test, device):
    """Reject custom writers that do not explicitly implement speculative admission."""
    with test.assertRaisesRegex(ValueError, "contact_writer_supports_speculative=True"):
        NarrowPhase(
            max_candidate_pairs=1,
            device=device,
            contact_writer_warp_func=write_contact_simple,
            speculative=True,
        )


def test_speculative_pipeline_rejects_hydroelastic_before_sdf_construction(test, device):
    """Reject speculative hydroelastic pairs before constructing their SDF pipeline."""
    builder = newton.ModelBuilder(gravity=wp.vec3(0.0))
    body_a = builder.add_body()
    builder.add_shape_sphere(body_a, radius=0.1)
    body_b = builder.add_body(xform=wp.transform(wp.vec3(0.15, 0.0, 0.0)))
    builder.add_shape_sphere(body_b, radius=0.1)
    model = builder.finalize(device=device)
    shape_flags = model.shape_flags.numpy()
    shape_flags |= int(newton.ShapeFlags.HYDROELASTIC)
    model.shape_flags.assign(shape_flags)

    with test.assertRaisesRegex(NotImplementedError, "does not yet support hydroelastic"):
        newton.CollisionPipeline(
            model,
            speculative_config=newton.CollisionPipeline.SpeculativeContactConfig(),
        )


def test_speculative_pipeline_allows_missing_explicit_pairs(test, device):
    """Allow non-explicit speculative pipelines when the model pair array is absent."""
    model, _state = _build_spheres(device, velocity=0.0)
    model.shape_contact_pairs = None
    newton.CollisionPipeline(
        model,
        broad_phase="nxn",
        speculative_config=newton.CollisionPipeline.SpeculativeContactConfig(),
    )


def test_speculative_mesh_sdf_candidates(test, device):
    """Verify separated approaching mesh SDFs retain leading candidates."""
    projectile = newton.Mesh.create_box(0.05, compute_normals=False, compute_uvs=False)
    projectile.build_sdf(device=device, max_resolution=32)
    wall = newton.Mesh.create_box(0.02, 0.3, 0.3, compute_normals=False, compute_uvs=False)
    wall.build_sdf(device=device, max_resolution=32)

    builder = newton.ModelBuilder(gravity=wp.vec3(0.0))
    builder.rigid_gap = 0.0
    body = builder.add_body(xform=wp.transform(wp.vec3(-0.2, 0.0, 0.0)))
    builder.add_shape_mesh(body, mesh=projectile)
    builder.body_qd[body] = (20.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    builder.add_shape_mesh(-1, mesh=wall)
    model = builder.finalize(device=device)

    contacts = _collide(model, model.state(), speculative=True)
    test.assertGreater(int(contacts.rigid_contact_count.numpy()[0]), 0)


def test_speculative_axial_shapes_reach_triangle_mesh(test, device):
    """Retain separated sphere and capsule contacts against a triangle mesh."""
    for shape_type in ("sphere", "capsule"):
        with test.subTest(shape_type=shape_type):
            wall = newton.Mesh.create_box(0.02, 0.3, 0.3, compute_normals=False, compute_uvs=False)
            builder = newton.ModelBuilder(gravity=wp.vec3(0.0))
            builder.rigid_gap = 0.0
            body = builder.add_body(xform=wp.transform(wp.vec3(-0.25, 0.0, 0.0)))
            if shape_type == "sphere":
                builder.add_shape_sphere(body, radius=0.1)
            else:
                builder.add_shape_capsule(body, radius=0.1, half_height=0.1)
            builder.body_qd[body] = (10.0, 0.0, 0.0, 0.0, 0.0, 0.0)
            builder.add_shape_mesh(-1, mesh=wall)
            model = builder.finalize(device=device)

            contacts = _collide(model, model.state(), speculative=True)
            test.assertGreater(int(contacts.rigid_contact_count.numpy()[0]), 0)


def test_speculative_mesh_sdf_manifold_is_bounded(test, device):
    """Limit a separated mesh-SDF shape pair to seven predictive contacts."""
    projectile = newton.Mesh.create_sphere(
        0.2,
        num_latitudes=32,
        num_longitudes=32,
        compute_normals=False,
        compute_uvs=False,
    )
    projectile.build_sdf(device=device, max_resolution=64)
    wall = newton.Mesh.create_box(0.02, 0.5, 0.5, compute_normals=False, compute_uvs=False)
    wall.build_sdf(device=device, max_resolution=64)

    builder = newton.ModelBuilder(gravity=wp.vec3(0.0))
    builder.rigid_gap = 0.0
    body = builder.add_body(xform=wp.transform(wp.vec3(-0.35, 0.0, 0.0)))
    builder.add_shape_mesh(body, mesh=projectile)
    builder.body_qd[body] = (10.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    builder.add_shape_mesh(-1, mesh=wall)
    model = builder.finalize(device=device)

    pipeline = newton.CollisionPipeline(
        model,
        broad_phase="nxn",
        speculative_config=newton.CollisionPipeline.SpeculativeContactConfig(
            max_speculative_extension=0.25,
        ),
    )
    contacts = pipeline.contacts()
    pipeline.collide(model.state(), contacts, dt=0.03)

    count = int(contacts.rigid_contact_count.numpy()[0])
    test.assertGreater(count, 0)
    test.assertLessEqual(count, 7)


def test_speculative_mesh_sdf_retains_rotating_leading_feature(test, device):
    """Verify an inner SDF contact cannot hide the rod end rotating toward the board."""
    rod = newton.Mesh.create_box(0.5, 0.04, 0.04, compute_normals=False, compute_uvs=False)
    rod.build_sdf(device=device, max_resolution=64)
    board = newton.Mesh.create_box(0.7, 0.3, 0.02, compute_normals=False, compute_uvs=False)
    board.build_sdf(device=device, max_resolution=64)

    builder = newton.ModelBuilder(gravity=wp.vec3(0.0))
    builder.rigid_gap = 0.0
    rod_body = builder.add_body(
        xform=wp.transform(
            wp.vec3(0.0, 0.0, 0.095),
            wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), -0.1),
        )
    )
    builder.add_shape_mesh(rod_body, mesh=rod)
    builder.body_qd[rod_body] = (0.0, 0.0, 0.0, 0.0, 10.0, 0.0)
    builder.add_shape_mesh(-1, mesh=board)
    model = builder.finalize(device=device)

    pipeline = newton.CollisionPipeline(
        model,
        broad_phase="nxn",
        speculative_config=newton.CollisionPipeline.SpeculativeContactConfig(
            max_speculative_extension=0.15,
        ),
    )
    contacts = pipeline.contacts()
    pipeline.collide(model.state(), contacts, dt=0.03)

    count = int(contacts.rigid_contact_count.numpy()[0])
    test.assertGreater(count, 1)
    rod_points = contacts.rigid_contact_point0.numpy()[:count]
    test.assertLess(float(rod_points[:, 0].min()), -0.3)
    test.assertGreater(float(rod_points[:, 0].max()), 0.3)


def test_predictive_reducer_reuses_regular_contact(test, device, deterministic):
    """Verify one candidate winning regular and predictive keys exports once."""
    reducer = GlobalContactReducer(capacity=8, device=device, deterministic=deterministic)
    shape_transform = wp.array([wp.transform_identity(), wp.transform_identity()], dtype=wp.transform, device=device)
    shape_linear_velocity = wp.array([wp.vec3(1.0, 0.0, 0.0), wp.vec3(0.0)], dtype=wp.vec3, device=device)
    shape_angular_velocity = wp.zeros(2, dtype=wp.vec3, device=device)
    contact_ids = wp.zeros(3, dtype=wp.int32, device=device)
    wp.launch(
        _register_regular_and_predictive_contact,
        dim=1,
        inputs=[
            reducer.get_data_struct(),
            shape_transform,
            shape_linear_velocity,
            shape_angular_velocity,
            contact_ids,
        ],
        device=device,
    )

    ids = contact_ids.numpy()
    test.assertGreaterEqual(int(ids[0]), 0)
    test.assertEqual(int(ids[1]), int(ids[0]))
    test.assertEqual(int(ids[2]), int(ids[0]))
    test.assertEqual(int(reducer.contact_count.numpy()[0]), 1)
    exported_count, _ = _export_reducer_contacts(reducer, device)
    test.assertEqual(exported_count, 1)


def test_speculative_buffered_reducer_uses_one_based_ids(test, device, deterministic):
    """Verify buffered speculative reduction processes every real contact and skips reserved ID zero."""
    reducer = GlobalContactReducer(capacity=8, device=device, deterministic=deterministic)
    wp.launch(_buffer_one_contact, dim=1, inputs=[reducer.get_data_struct()], device=device)

    shape_transform = wp.array([wp.transform_identity(), wp.transform_identity()], dtype=wp.transform, device=device)
    wp.launch(
        reduce_buffered_contacts_speculative_kernel,
        dim=1,
        inputs=[
            reducer.get_data_struct(),
            wp.full(2, int(GeoType.BOX), dtype=wp.int32, device=device),
            wp.zeros(2, dtype=wp.vec4, device=device),
            wp.zeros(2, dtype=wp.float32, device=device),
            shape_transform,
            wp.zeros(2, dtype=wp.vec3, device=device),
            wp.zeros(2, dtype=wp.vec3, device=device),
            wp.full(2, wp.vec3(-1.0), dtype=wp.vec3, device=device),
            wp.full(2, wp.vec3(1.0), dtype=wp.vec3, device=device),
            wp.full(2, wp.vec3i(1), dtype=wp.vec3i, device=device),
            0.1,
            0.1,
            1,
        ],
        device=device,
    )

    test.assertEqual(int(reducer.contact_count.numpy()[0]), 1)
    exported_count, positions = _export_reducer_contacts(reducer, device)
    test.assertEqual(exported_count, 1)
    for actual, expected in zip(positions[0], (0.25, -0.5, 0.75), strict=True):
        test.assertAlmostEqual(float(actual), expected, places=6)


def test_speculative_buffered_axial_contacts_use_predictive_manifold(test, device):
    """Route separated sphere and capsule mesh contacts by their physical clearance."""
    for shape_type in (GeoType.SPHERE, GeoType.CAPSULE):
        with test.subTest(shape_type=shape_type):
            reducer = GlobalContactReducer(capacity=8, device=device)
            wp.launch(_buffer_separated_axial_contact, dim=1, inputs=[reducer.get_data_struct()], device=device)

            wp.launch(
                reduce_buffered_contacts_speculative_kernel,
                dim=1,
                inputs=[
                    reducer.get_data_struct(),
                    wp.array([int(shape_type), int(GeoType.MESH)], dtype=wp.int32, device=device),
                    wp.array([wp.vec4(0.1, 0.1, 0.1, 0.0), wp.vec4(0.0)], dtype=wp.vec4, device=device),
                    wp.zeros(2, dtype=wp.float32, device=device),
                    wp.array([wp.transform_identity(), wp.transform_identity()], dtype=wp.transform, device=device),
                    wp.array([wp.vec3(1.0, 0.0, 0.0), wp.vec3(0.0)], dtype=wp.vec3, device=device),
                    wp.zeros(2, dtype=wp.vec3, device=device),
                    wp.full(2, wp.vec3(-1.0), dtype=wp.vec3, device=device),
                    wp.full(2, wp.vec3(1.0), dtype=wp.vec3, device=device),
                    wp.full(2, wp.vec3i(1), dtype=wp.vec3i, device=device),
                    0.1,
                    0.1,
                    1,
                ],
                device=device,
            )

            bins = (reducer.hashtable.keys.numpy() >> np.uint64(55)) & np.uint64(0xFF)
            test.assertIn(PREDICTIVE_BIN_ID, bins)


def test_predictive_reducer_reclaims_replaced_reservation(test, device, deterministic):
    """Reclaim capacity when both validated predictive claims are replaced."""
    capacity = 8
    reducer = _make_predictive_reducer(capacity, device, deterministic)
    allocated_ids = wp.full(capacity, -1, dtype=wp.int32, device=device)
    wp.launch(
        _replace_validated_predictive_claims,
        dim=1,
        inputs=[reducer.get_data_struct(), allocated_ids],
        device=device,
    )

    test.assertEqual(sorted(int(contact_id) for contact_id in allocated_ids.numpy()), list(range(1, capacity + 1)))
    test.assertEqual(int(reducer.contact_count.numpy()[0]), capacity)


def test_predictive_reducer_retains_rotating_leading_contact(test, device, deterministic):
    """Verify an inner winner cannot suppress an imminent angular-only feature."""
    reducer = _make_predictive_reducer(8, device, deterministic)
    shape_transform = wp.array([wp.transform_identity(), wp.transform_identity()], dtype=wp.transform, device=device)
    shape_linear_velocity = wp.zeros(2, dtype=wp.vec3, device=device)
    shape_angular_velocity = wp.array([wp.vec3(0.0, 0.0, -1.0), wp.vec3(0.0)], dtype=wp.vec3, device=device)
    contact_ids = wp.zeros(3, dtype=wp.int32, device=device)
    wp.launch(
        _register_inner_and_rotating_leading_contact,
        dim=1,
        inputs=[
            reducer.get_data_struct(),
            shape_transform,
            shape_linear_velocity,
            shape_angular_velocity,
            contact_ids,
        ],
        device=device,
    )

    ids = contact_ids.numpy()
    test.assertGreaterEqual(int(ids[0]), 0)
    test.assertEqual(int(ids[1]), -1)
    test.assertGreaterEqual(int(ids[2]), 0)
    test.assertNotEqual(int(ids[2]), int(ids[0]))
    test.assertEqual(int(reducer.contact_count.numpy()[0]), 2)
    exported_count, positions = _export_reducer_contacts(reducer, device)
    test.assertEqual(exported_count, 2)
    test.assertAlmostEqual(float(max(positions[:exported_count, 1])), 1.0, places=6)


def test_predictive_reducer_retains_clearance_manifold(test, device, deterministic):
    """Verify the nearest clearance in each deterministic shard and the impact guard survive."""
    reducer = _make_predictive_reducer(16, device, deterministic)
    shape_transform = wp.array([wp.transform_identity(), wp.transform_identity()], dtype=wp.transform, device=device)
    shape_linear_velocity = wp.array([wp.vec3(1.0, 0.0, 0.0), wp.vec3(0.0)], dtype=wp.vec3, device=device)
    shape_angular_velocity = wp.array([wp.vec3(0.0, 0.0, -1.0), wp.vec3(0.0)], dtype=wp.vec3, device=device)
    contact_ids = wp.full(8, -1, dtype=wp.int32, device=device)
    wp.launch(
        _register_predictive_clearance_candidates,
        dim=1,
        inputs=[
            reducer.get_data_struct(),
            shape_transform,
            shape_linear_velocity,
            shape_angular_velocity,
            contact_ids,
        ],
        device=device,
    )

    test.assertEqual(int(reducer.contact_count.numpy()[0]), 8)
    exported_count, positions = _export_reducer_contacts(reducer, device)
    test.assertEqual(exported_count, 7)
    clearances = sorted(float(position[2]) for position in positions[:exported_count])
    for actual, expected in zip(clearances, (0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08), strict=True):
        test.assertAlmostEqual(actual, expected, places=6)


def test_predictive_reducer_retains_clearance_manifold_under_contention(test, device, deterministic):
    """Verify concurrent blocks retain all winners within a bounded contact buffer."""
    shape_transform = wp.array([wp.transform_identity(), wp.transform_identity()], dtype=wp.transform, device=device)
    shape_linear_velocity = wp.array([wp.vec3(1.0, 0.0, 0.0), wp.vec3(0.0)], dtype=wp.vec3, device=device)
    shape_angular_velocity = wp.array([wp.vec3(0.0, 0.0, -1.0), wp.vec3(0.0)], dtype=wp.vec3, device=device)
    for clearance_order in (0, 1, 2):
        with test.subTest(clearance_order=clearance_order):
            reducer = _make_predictive_reducer(16, device, deterministic)
            wp.launch(
                _register_predictive_clearance_candidates_contended,
                dim=256,
                inputs=[
                    reducer.get_data_struct(),
                    shape_transform,
                    shape_linear_velocity,
                    shape_angular_velocity,
                    clearance_order,
                ],
                block_dim=32,
                device=device,
            )

            test.assertLessEqual(int(reducer.contact_count.numpy()[0]), reducer.capacity)
            exported_count, positions = _export_reducer_contacts(reducer, device)
            test.assertEqual(exported_count, 7)
            clearances = sorted(float(position[2]) for position in positions[:exported_count])
            expected = _expected_contended_clearances(clearance_order)
            for actual, expected_clearance in zip(clearances, expected, strict=True):
                test.assertAlmostEqual(actual, expected_clearance, places=6)


def test_predictive_reducer_preserves_winners_with_small_buffer(test, device, deterministic):
    """Verify 256 ordered contenders preserve shard winners within a 16-contact buffer."""
    reducer = _make_predictive_reducer(16, device, deterministic)
    shape_transform = wp.array([wp.transform_identity(), wp.transform_identity()], dtype=wp.transform, device=device)
    shape_linear_velocity = wp.array([wp.vec3(1.0, 0.0, 0.0), wp.vec3(0.0)], dtype=wp.vec3, device=device)
    shape_angular_velocity = wp.array([wp.vec3(0.0, 0.0, -1.0), wp.vec3(0.0)], dtype=wp.vec3, device=device)
    wp.launch(
        _register_predictive_clearance_candidates_sequential,
        dim=1,
        inputs=[
            reducer.get_data_struct(),
            shape_transform,
            shape_linear_velocity,
            shape_angular_velocity,
        ],
        device=device,
    )

    allocated_count = int(reducer.contact_count.numpy()[0])
    test.assertLessEqual(allocated_count, reducer.capacity)
    exported_count, positions = _export_reducer_contacts(reducer, device)
    test.assertEqual(exported_count, 7)
    test.assertLessEqual(exported_count, allocated_count)
    clearances = sorted(float(position[2]) for position in positions[:exported_count])
    for actual, expected in zip(clearances, (0.01, 0.0101, 0.0102, 0.0104, 0.0107, 0.011, 0.08), strict=True):
        test.assertAlmostEqual(actual, expected, places=6)


def test_predictive_reducer_restores_winner_when_buffer_is_full(test, device, deterministic):
    """Preserve displaced winners when a stronger candidate cannot allocate storage."""
    reducer = _make_predictive_reducer(1, device, deterministic)
    shape_transform = wp.array([wp.transform_identity(), wp.transform_identity()], dtype=wp.transform, device=device)
    shape_linear_velocity = wp.array([wp.vec3(1.0, 0.0, 0.0), wp.vec3(0.0)], dtype=wp.vec3, device=device)
    shape_angular_velocity = wp.zeros(2, dtype=wp.vec3, device=device)
    contact_ids = wp.full(2, -1, dtype=wp.int32, device=device)
    wp.launch(
        _replace_full_buffer_predictive_winner,
        dim=1,
        inputs=[
            reducer.get_data_struct(),
            shape_transform,
            shape_linear_velocity,
            shape_angular_velocity,
            contact_ids,
        ],
        device=device,
    )

    ids = contact_ids.numpy()
    test.assertEqual(int(ids[0]), 1)
    test.assertEqual(int(ids[1]), -1)
    exported_count, positions = _export_reducer_contacts(reducer, device)
    test.assertEqual(exported_count, 1)
    test.assertAlmostEqual(float(positions[0, 2]), 0.08, places=6)


def test_predictive_reducer_reclamation_storage_is_opt_in(test, device):
    """Avoid allocating predictive reservation storage in the default reducer."""
    reducer = GlobalContactReducer(capacity=32, device=device)
    test.assertEqual(reducer.reclaimed_contact_bits.shape[0], 0)
    test.assertEqual(reducer.reclaimed_contact_cursor.shape[0], 0)


def test_predictive_reducer_reclaims_ids_without_duplicates(test, device):
    """Allocate each concurrently reclaimed contact ID exactly once."""
    capacity = 1024
    reducer = _make_predictive_reducer(capacity, device, deterministic=False)
    reducer.contact_count.fill_(capacity)
    allocated_ids = wp.full(capacity, -1, dtype=wp.int32, device=device)

    wp.launch(
        _reclaim_and_allocate_contact_ids,
        dim=capacity,
        inputs=[reducer.get_data_struct(), allocated_ids],
        device=device,
    )

    ids = np.sort(allocated_ids.numpy())
    np.testing.assert_array_equal(ids, np.arange(1, capacity + 1, dtype=np.int32))
    test.assertEqual(int(reducer.contact_count.numpy()[0]), capacity)
    test.assertEqual(int(np.count_nonzero(reducer.reclaimed_contact_bits.numpy())), 0)


class TestSpeculativeContacts(unittest.TestCase):
    """Test inexpensive speculative-candidate behavior."""


class TestSpeculativeMeshContacts(unittest.TestCase):
    """Test speculative mesh/SDF candidate generation on CUDA."""


for _name, _test in (
    ("test_speculative_candidates_are_opt_in", test_speculative_candidates_are_opt_in),
    ("test_speculative_candidates_require_approach", test_speculative_candidates_require_approach),
    ("test_speculative_candidates_require_dt", test_speculative_candidates_require_dt),
    (
        "test_speculative_gap_uses_larger_fixed_or_velocity_distance",
        test_speculative_gap_uses_larger_fixed_or_velocity_distance,
    ),
    ("test_speculative_candidates_reject_invalid_dt_override", test_speculative_candidates_reject_invalid_dt_override),
    ("test_speculative_candidates_reject_common_motion", test_speculative_candidates_reject_common_motion),
    ("test_speculative_candidates_preserve_physical_geometry", test_speculative_candidates_preserve_physical_geometry),
    ("test_speculative_candidates_include_angular_motion", test_speculative_candidates_include_angular_motion),
    ("test_speculative_cone_reaches_infinite_plane", test_speculative_cone_reaches_infinite_plane),
    (
        "test_stationary_contacts_match_non_speculative_pipeline",
        test_stationary_contacts_match_non_speculative_pipeline,
    ),
    ("test_speculative_contacts_prevent_dynamic_tunneling", test_speculative_contacts_prevent_dynamic_tunneling),
    ("test_speculative_narrow_phase_launch", test_speculative_narrow_phase_launch),
    (
        "test_speculative_narrow_phase_rejects_hydroelastic",
        test_speculative_narrow_phase_rejects_hydroelastic,
    ),
    (
        "test_speculative_narrow_phase_rejects_unmarked_custom_writer",
        test_speculative_narrow_phase_rejects_unmarked_custom_writer,
    ),
    (
        "test_speculative_pipeline_rejects_hydroelastic_before_sdf_construction",
        test_speculative_pipeline_rejects_hydroelastic_before_sdf_construction,
    ),
    (
        "test_speculative_pipeline_allows_missing_explicit_pairs",
        test_speculative_pipeline_allows_missing_explicit_pairs,
    ),
    (
        "test_predictive_reducer_reclamation_storage_is_opt_in",
        test_predictive_reducer_reclamation_storage_is_opt_in,
    ),
    (
        "test_speculative_buffered_axial_contacts_use_predictive_manifold",
        test_speculative_buffered_axial_contacts_use_predictive_manifold,
    ),
    (
        "test_predictive_reducer_reclaims_ids_without_duplicates",
        test_predictive_reducer_reclaims_ids_without_duplicates,
    ),
):
    add_function_test(TestSpeculativeContacts, _name, _test, devices=get_test_devices())

for _deterministic in (False, True):
    _suffix = "deterministic" if _deterministic else "fast"
    add_function_test(
        TestSpeculativeContacts,
        f"test_predictive_reducer_reuses_regular_contact_{_suffix}",
        test_predictive_reducer_reuses_regular_contact,
        devices=get_test_devices(),
        deterministic=_deterministic,
    )
    add_function_test(
        TestSpeculativeContacts,
        f"test_speculative_buffered_reducer_uses_one_based_ids_{_suffix}",
        test_speculative_buffered_reducer_uses_one_based_ids,
        devices=get_test_devices(),
        deterministic=_deterministic,
    )
    add_function_test(
        TestSpeculativeContacts,
        f"test_predictive_reducer_reclaims_replaced_reservation_{_suffix}",
        test_predictive_reducer_reclaims_replaced_reservation,
        devices=get_test_devices(),
        deterministic=_deterministic,
    )
    add_function_test(
        TestSpeculativeContacts,
        f"test_predictive_reducer_retains_rotating_leading_contact_{_suffix}",
        test_predictive_reducer_retains_rotating_leading_contact,
        devices=get_test_devices(),
        deterministic=_deterministic,
    )
    add_function_test(
        TestSpeculativeContacts,
        f"test_predictive_reducer_retains_clearance_manifold_{_suffix}",
        test_predictive_reducer_retains_clearance_manifold,
        devices=get_test_devices(),
        deterministic=_deterministic,
    )
    add_function_test(
        TestSpeculativeContacts,
        f"test_predictive_reducer_retains_clearance_manifold_under_contention_{_suffix}",
        test_predictive_reducer_retains_clearance_manifold_under_contention,
        devices=get_cuda_test_devices(),
        deterministic=_deterministic,
    )
    add_function_test(
        TestSpeculativeContacts,
        f"test_predictive_reducer_preserves_winners_with_small_buffer_{_suffix}",
        test_predictive_reducer_preserves_winners_with_small_buffer,
        devices=get_test_devices(),
        deterministic=_deterministic,
    )
    add_function_test(
        TestSpeculativeContacts,
        f"test_predictive_reducer_restores_winner_when_buffer_is_full_{_suffix}",
        test_predictive_reducer_restores_winner_when_buffer_is_full,
        devices=get_test_devices(),
        deterministic=_deterministic,
    )

add_function_test(
    TestSpeculativeMeshContacts,
    "test_speculative_axial_shapes_reach_triangle_mesh",
    test_speculative_axial_shapes_reach_triangle_mesh,
    devices=get_test_devices(),
)
add_function_test(
    TestSpeculativeMeshContacts,
    "test_speculative_mesh_sdf_candidates",
    test_speculative_mesh_sdf_candidates,
    devices=get_cuda_test_devices(),
)
add_function_test(
    TestSpeculativeMeshContacts,
    "test_speculative_mesh_sdf_manifold_is_bounded",
    test_speculative_mesh_sdf_manifold_is_bounded,
    devices=get_cuda_test_devices(),
)
add_function_test(
    TestSpeculativeMeshContacts,
    "test_speculative_mesh_sdf_retains_rotating_leading_feature",
    test_speculative_mesh_sdf_retains_rotating_leading_feature,
    devices=get_cuda_test_devices(),
)


if __name__ == "__main__":
    unittest.main(verbosity=2)
