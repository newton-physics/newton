# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for frame-to-frame contact matching."""

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.geometry.contact_data import CONTACT_SORT_CONVEX_SUB_KEY_BITS, CONTACT_SORT_SUB_KEY_BITS
from newton._src.geometry.contact_match import _CLAIM_SENTINEL, MATCH_BROKEN, MATCH_NOT_FOUND
from newton.tests.unittest_utils import add_function_test, get_cuda_test_devices, get_test_devices


class TestContactMatching(unittest.TestCase):
    pass


class TestContactMatchingSticky(unittest.TestCase):
    pass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_simple_scene(device):
    """Build a scene with 3 spheres resting on a ground plane.

    Returns (model, state).  Spheres at x = -0.5, 0.0, 0.5, all at z = radius
    so they touch the plane.
    """
    builder = newton.ModelBuilder()
    builder.add_ground_plane()

    for x in (-0.5, 0.0, 0.5):
        b = builder.add_body(xform=wp.transform(wp.vec3(x, 0.0, 0.1)))
        builder.add_shape_sphere(body=b, radius=0.1)

    model = builder.finalize(device=device)
    state = model.state()
    return model, state


def _collide_once(pipeline, state, contacts):
    """Clear and collide, returning the contact count on host."""
    contacts.clear()
    pipeline.collide(state, contacts)
    return contacts.rigid_contact_count.numpy()[0]


def _build_two_world_contact_scene(device):
    """Build two local sphere-plane contacts sharing one global ground."""
    world = newton.ModelBuilder()
    body = world.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.1)))
    world.add_shape_sphere(body=body, radius=0.1)
    builder = newton.ModelBuilder()
    builder.add_ground_plane()
    builder.add_world(world, xform=wp.transform(wp.vec3(-0.5, 0.0, 0.0)))
    builder.add_world(world, xform=wp.transform(wp.vec3(0.5, 0.0, 0.0)))
    model = builder.finalize(device=device)
    state = model.state()
    pipeline = newton.CollisionPipeline(
        model,
        broad_phase="nxn",
        contact_matching="latest",
        contact_report=True,
    )
    return model, state, pipeline, pipeline.contacts()


# ---------------------------------------------------------------------------
# Test functions
# ---------------------------------------------------------------------------


def test_first_frame_all_not_found(test, device):
    """First frame: prev_count is 0, so every contact must get MATCH_NOT_FOUND."""
    with wp.ScopedDevice(device):
        model, state = _build_simple_scene(device)
        pipeline = newton.CollisionPipeline(model, broad_phase="nxn", contact_matching="latest")
        contacts = pipeline.contacts()

        count = _collide_once(pipeline, state, contacts)
        test.assertGreater(count, 0, "Expected contacts between spheres and ground plane")

        match_idx = contacts.rigid_contact_match_index.numpy()[:count]
        test.assertTrue(
            np.all(match_idx == MATCH_NOT_FOUND),
            f"First frame should have all MATCH_NOT_FOUND, got unique values: {np.unique(match_idx)}",
        )


def test_stable_scene_identity_match(test, device):
    """Stable scene: deterministic sort + identical state means match_index[i] == i.

    This is the strongest possible invariant: each sorted contact maps to the
    same sorted position in the previous frame.  It verifies binary search,
    position/normal threshold acceptance, sort permutation of match_index,
    and the save-then-match round-trip through the sorter's scratch buffers.
    """
    with wp.ScopedDevice(device):
        model, state = _build_simple_scene(device)
        pipeline = newton.CollisionPipeline(model, broad_phase="nxn", contact_matching="latest")
        contacts = pipeline.contacts()

        # Frame 1: populate previous-frame data.
        count1 = _collide_once(pipeline, state, contacts)
        test.assertGreater(count1, 0)

        # Frame 2: identical state.
        count2 = _collide_once(pipeline, state, contacts)
        test.assertEqual(count1, count2, "Contact count must be stable between identical frames")

        match_idx = contacts.rigid_contact_match_index.numpy()[:count2]
        expected = np.arange(count2, dtype=np.int32)
        np.testing.assert_array_equal(
            match_idx,
            expected,
            err_msg="Stable scene: match_index[i] must be i (identity mapping)",
        )


def test_stable_scene_identity_across_three_frames(test, device):
    """Identity match must hold across 3+ frames, not just the first pair."""
    with wp.ScopedDevice(device):
        model, state = _build_simple_scene(device)
        pipeline = newton.CollisionPipeline(model, broad_phase="nxn", contact_matching="latest")
        contacts = pipeline.contacts()

        _collide_once(pipeline, state, contacts)  # frame 1
        for frame in range(2, 5):
            count = _collide_once(pipeline, state, contacts)
            match_idx = contacts.rigid_contact_match_index.numpy()[:count]
            expected = np.arange(count, dtype=np.int32)
            np.testing.assert_array_equal(
                match_idx,
                expected,
                err_msg=f"Frame {frame}: match_index must be identity",
            )


def test_save_initializes_next_frame_claims(test, device):
    """Initialize saved claim slots for the next matching frame."""
    with wp.ScopedDevice(device):
        model, state = _build_simple_scene(device)
        pipeline = newton.CollisionPipeline(model, broad_phase="nxn", contact_matching="latest")
        contacts = pipeline.contacts()
        matcher = pipeline._contact_matcher
        test.assertIsNotNone(matcher)

        matcher._prev_claim.fill_(wp.int64(0))
        count = _collide_once(pipeline, state, contacts)
        test.assertGreater(count, 0)
        np.testing.assert_array_equal(
            matcher._prev_claim.numpy()[:count],
            np.full(count, int(_CLAIM_SENTINEL), dtype=np.int64),
        )

        test.assertEqual(_collide_once(pipeline, state, contacts), count)


def test_masked_reset_restarts_only_selected_contact_history(test, device):
    """Restart local or global contact pairs while preserving other matches."""
    with wp.ScopedDevice(device):
        model, state, pipeline, contacts = _build_two_world_contact_scene(device)

        count = _collide_once(pipeline, state, contacts)
        test.assertEqual(_collide_once(pipeline, state, contacts), count)
        np.testing.assert_array_equal(
            contacts.rigid_contact_match_index.numpy()[:count],
            np.arange(count, dtype=np.int32),
        )

        pipeline.reset_contact_matching(wp.array((False, False, False), dtype=wp.bool, device=device))
        _collide_once(pipeline, state, contacts)
        test.assertTrue(np.all(contacts.rigid_contact_match_index.numpy()[:count] >= 0))

        pipeline.reset_contact_matching(wp.array((True, False, False), dtype=wp.bool, device=device))
        _collide_once(pipeline, state, contacts)
        shape_world = model.shape_world.numpy()
        shape0 = contacts.rigid_contact_shape0.numpy()[:count]
        shape1 = contacts.rigid_contact_shape1.numpy()[:count]
        world0 = shape_world[shape0]
        world1 = shape_world[shape1]
        contact_world = np.where(world0 >= 0, world0, world1)
        matches = contacts.rigid_contact_match_index.numpy()[:count]
        np.testing.assert_array_equal(matches[contact_world == 0], MATCH_NOT_FOUND)
        test.assertTrue(np.all(matches[contact_world == 1] >= 0))
        test.assertEqual(int(contacts.rigid_contact_broken_count.numpy()[0]), 0)

        _collide_once(pipeline, state, contacts)
        pipeline.reset_contact_matching(wp.array((False, False, True), dtype=wp.bool, device=device))
        _collide_once(pipeline, state, contacts)
        np.testing.assert_array_equal(contacts.rigid_contact_match_index.numpy()[:count], MATCH_NOT_FOUND)
        test.assertEqual(int(contacts.rigid_contact_broken_count.numpy()[0]), 0)


def test_reset_rejects_invalid_world_masks(test, device):
    """Reject malformed contact reset masks through the shared contract."""
    with wp.ScopedDevice(device):
        _model, _state, pipeline, _contacts = _build_two_world_contact_scene(device)
        invalid_masks = (
            ((False, False, False), TypeError),
            (wp.zeros(3, dtype=wp.int32, device=device), TypeError),
            (wp.zeros((1, 3), dtype=wp.bool, device=device), ValueError),
            (wp.zeros(2, dtype=wp.bool, device=device), ValueError),
            (wp.zeros(4, dtype=wp.bool, device=device), ValueError),
        )
        for world_mask, error in invalid_masks:
            with test.subTest(world_mask=world_mask), test.assertRaises(error):
                pipeline.reset_contact_matching(world_mask)

        if device.is_cuda:
            with test.assertRaises(ValueError):
                pipeline.reset_contact_matching(wp.zeros(3, dtype=wp.bool, device="cpu"))


def test_reset_accumulates_live_mask_during_cuda_graph_replay(test, device):
    """Accumulate the current device mask each time a reset graph is replayed."""
    with wp.ScopedDevice(device):
        _model, _state, pipeline, _contacts = _build_two_world_contact_scene(device)
        matcher = pipeline._contact_matcher
        test.assertIsNotNone(matcher)

        # Materialize the reset kernel before capture, then clear its pending state.
        pipeline.reset_contact_matching(wp.array((True, False, False), dtype=wp.bool, device=device))
        pipeline.reset_contact_matching()
        world_mask = wp.zeros(3, dtype=wp.bool, device=device)
        with wp.ScopedCapture(device=device) as capture:
            pipeline.reset_contact_matching(world_mask)

        world_mask.assign((True, False, False))
        wp.capture_launch(capture.graph)
        np.testing.assert_array_equal(matcher._reset_world_mask.numpy(), (True, False, False))

        world_mask.assign((False, True, True))
        wp.capture_launch(capture.graph)
        np.testing.assert_array_equal(matcher._reset_world_mask.numpy(), (True, True, True))


def test_new_contact_detection(test, device):
    """A new sphere that enters the scene produces MATCH_NOT_FOUND,
    while existing contacts keep their identity match.
    """
    with wp.ScopedDevice(device):
        builder = newton.ModelBuilder()
        builder.add_ground_plane()
        for x in (-0.5, 0.5):
            b = builder.add_body(xform=wp.transform(wp.vec3(x, 0.0, 0.1)))
            builder.add_shape_sphere(body=b, radius=0.1)
        # Third sphere far away — no contacts in frame 1.
        b3 = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 10.0)))
        builder.add_shape_sphere(body=b3, radius=0.1)

        model = builder.finalize(device=device)
        state = model.state()
        pipeline = newton.CollisionPipeline(model, broad_phase="nxn", contact_matching="latest")
        contacts = pipeline.contacts()

        # Frame 1: 2 sphere-plane contacts.
        count1 = _collide_once(pipeline, state, contacts)
        test.assertGreater(count1, 0)

        # Move third sphere to ground for frame 2.
        q = state.body_q.numpy()
        q[2][0:3] = [0.0, 0.0, 0.1]
        state.body_q = wp.array(q, dtype=wp.transform, device=device)

        count2 = _collide_once(pipeline, state, contacts)
        test.assertGreater(count2, count1, "More contacts expected with third sphere on ground")

        match_idx = contacts.rigid_contact_match_index.numpy()[:count2]

        n_new = np.sum(match_idx == -1)
        n_matched = np.sum(match_idx >= 0)
        test.assertGreater(n_new, 0, "New sphere should produce MATCH_NOT_FOUND contacts")
        test.assertEqual(n_matched, count1, f"All {count1} old contacts should still match, got {n_matched}")

        # Matched indices must be unique (no two new contacts claim the same old).
        matched_vals = match_idx[match_idx >= 0]
        test.assertEqual(len(np.unique(matched_vals)), len(matched_vals), "Matched indices must be unique")


def test_broken_pos_threshold_all_contacts(test, device):
    """Moving all spheres beyond pos_threshold must break ALL contacts (not just some).

    Uses the default :attr:`CollisionPipeline.contact_matching_pos_threshold` so
    the test follows any future retune of the default.  ``contact_report=True``
    lets us close the loop and verify each broken new contact has a matching
    entry in ``rigid_contact_broken_indices`` (the old contact was also
    reported as broken — broken-on-both-sides).
    """
    with wp.ScopedDevice(device):
        model, state = _build_simple_scene(device)
        pipeline = newton.CollisionPipeline(
            model,
            broad_phase="nxn",
            contact_matching="latest",
            contact_report=True,
        )
        contacts = pipeline.contacts()

        count1 = _collide_once(pipeline, state, contacts)
        test.assertGreater(count1, 0)

        # Shift all dynamic bodies along x by 0.2 m — well above the default
        # (0.0005 m) pos_threshold but small enough to keep them on the plane.
        q = state.body_q.numpy()
        for i in range(len(q)):
            q[i][0] += 0.2
        state.body_q = wp.array(q, dtype=wp.transform, device=device)

        count2 = _collide_once(pipeline, state, contacts)
        match_idx = contacts.rigid_contact_match_index.numpy()[:count2]

        # Every new contact should be MATCH_BROKEN: key matches but position
        # drifted beyond threshold.
        test.assertTrue(
            np.all(match_idx == MATCH_BROKEN),
            f"All contacts should be MATCH_BROKEN. Unique values: {np.unique(match_idx)}",
        )

        # And every old contact should appear in broken_contact_indices:
        # if the new side is broken, the old side must also be broken
        # (nothing matched it).
        broken_count = contacts.rigid_contact_broken_count.numpy()[0]
        test.assertEqual(
            broken_count,
            count1,
            f"All {count1} old contacts should be reported as broken, got {broken_count}",
        )
        broken_indices = contacts.rigid_contact_broken_indices.numpy()[:broken_count]
        np.testing.assert_array_equal(
            np.sort(broken_indices),
            np.arange(count1, dtype=np.int32),
            err_msg="broken_contact_indices must enumerate every old contact",
        )


def test_within_pos_threshold_still_matches(test, device):
    """Moving spheres less than pos_threshold must still produce matches.

    Uses the default :attr:`CollisionPipeline.contact_matching_pos_threshold`
    (0.0005 m) so the test follows any future retune of the default.
    """
    with wp.ScopedDevice(device):
        model, state = _build_simple_scene(device)
        pipeline = newton.CollisionPipeline(
            model,
            broad_phase="nxn",
            contact_matching="latest",
        )
        contacts = pipeline.contacts()

        count1 = _collide_once(pipeline, state, contacts)
        test.assertGreater(count1, 0)

        # Shift all dynamic bodies along x by 0.0002 m — below the default
        # (0.0005 m) pos_threshold.
        q = state.body_q.numpy()
        for i in range(len(q)):
            q[i][0] += 0.0002
        state.body_q = wp.array(q, dtype=wp.transform, device=device)

        count2 = _collide_once(pipeline, state, contacts)
        match_idx = contacts.rigid_contact_match_index.numpy()[:count2]

        test.assertTrue(
            np.all(match_idx >= 0),
            f"All contacts should match within default threshold. Unique: {np.unique(match_idx)}",
        )


def test_broken_normal_threshold(test, device):
    """Moving a sphere so the contact normal direction changes beyond threshold
    produces MATCH_BROKEN.

    Two spheres (radius 0.1) overlap in frame 1 along x-axis (normal ≈ (1,0,0)).
    In frame 2, sphere B moves so they overlap along y-axis (normal ≈ (0,1,0)).
    Same shape pair / sub_key, generous pos_threshold, but dot((1,0,0), (0,1,0)) = 0
    which is below any reasonable normal_dot_threshold → MATCH_BROKEN.
    """
    with wp.ScopedDevice(device):
        builder = newton.ModelBuilder()
        # Two spheres overlapping along x-axis.
        ba = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.0)))
        builder.add_shape_sphere(body=ba, radius=0.1)
        bb = builder.add_body(xform=wp.transform(wp.vec3(0.19, 0.0, 0.0)))
        builder.add_shape_sphere(body=bb, radius=0.1)

        model = builder.finalize(device=device)
        state = model.state()

        pipeline = newton.CollisionPipeline(
            model,
            broad_phase="nxn",
            contact_matching="latest",
            contact_matching_pos_threshold=10.0,  # very generous — ignore position
            contact_matching_normal_dot_threshold=0.5,  # cos(60°) — perpendicular normals break
        )
        contacts = pipeline.contacts()

        count1 = _collide_once(pipeline, state, contacts)
        test.assertGreater(count1, 0, "Overlapping spheres must produce contacts")

        # Move sphere B so they overlap along y-axis instead.
        q = state.body_q.numpy()
        q[1][0:3] = [0.0, 0.19, 0.0]
        state.body_q = wp.array(q, dtype=wp.transform, device=device)

        count2 = _collide_once(pipeline, state, contacts)
        test.assertGreater(count2, 0, "Repositioned spheres must still produce contacts")

        match_idx = contacts.rigid_contact_match_index.numpy()[:count2]
        test.assertTrue(
            np.all(match_idx == -2),
            f"Normal changed ~90°, all should be MATCH_BROKEN. Unique: {np.unique(match_idx)}",
        )


def test_contact_report_indices_correct(test, device):
    """Contact report indices must be consistent with match_index values."""
    with wp.ScopedDevice(device):
        model, state = _build_simple_scene(device)
        pipeline = newton.CollisionPipeline(model, broad_phase="nxn", contact_matching="latest", contact_report=True)
        contacts = pipeline.contacts()

        # Frame 1: all contacts are new.
        count1 = _collide_once(pipeline, state, contacts)
        test.assertGreater(count1, 0)

        new_count1 = contacts.rigid_contact_new_count.numpy()[0]
        test.assertEqual(new_count1, count1, "First frame: all contacts should be new")

        # Verify new_contact_indices point to valid sorted positions.
        new_indices1 = contacts.rigid_contact_new_indices.numpy()[:new_count1]
        test.assertTrue(np.all(new_indices1 >= 0) and np.all(new_indices1 < count1))

        # Verify new_contact_indices match the actual -1 positions in match_index.
        match_idx1 = contacts.rigid_contact_match_index.numpy()[:count1]
        expected_new = np.where(match_idx1 < 0)[0].astype(np.int32)
        np.testing.assert_array_equal(
            np.sort(new_indices1),
            np.sort(expected_new),
            err_msg="rigid_contact_new_indices must match positions where match_index < 0",
        )

        # Frame 2: stable scene — no new, no broken.
        _collide_once(pipeline, state, contacts)
        test.assertEqual(contacts.rigid_contact_new_count.numpy()[0], 0)
        test.assertEqual(contacts.rigid_contact_broken_count.numpy()[0], 0)


def test_save_resets_next_frame_report_flags(test, device):
    """Reset saved report flags before the next matching frame."""
    with wp.ScopedDevice(device):
        model, state = _build_simple_scene(device)
        pipeline = newton.CollisionPipeline(
            model,
            broad_phase="nxn",
            contact_matching="latest",
            contact_report=True,
        )
        contacts = pipeline.contacts()
        matcher = pipeline._contact_matcher
        test.assertIsNotNone(matcher)

        count = _collide_once(pipeline, state, contacts)
        test.assertGreater(count, 0)
        matcher._prev_was_matched.fill_(wp.int32(1))
        matcher.save_sorted_state(
            sorted_keys=pipeline._contact_sorter.sorted_keys_view,
            contact_count=contacts.rigid_contact_count,
            sorted_point0=contacts.rigid_contact_point0,
            sorted_point1=contacts.rigid_contact_point1,
            sorted_shape0=contacts.rigid_contact_shape0,
            sorted_shape1=contacts.rigid_contact_shape1,
            sorted_normal=contacts.rigid_contact_normal,
            body_q=state.body_q,
            shape_body=model.shape_body,
            device=device,
        )
        np.testing.assert_array_equal(matcher._prev_was_matched.numpy()[:count], np.zeros(count, dtype=np.int32))

        test.assertEqual(_collide_once(pipeline, state, contacts), count)
        test.assertEqual(int(contacts.rigid_contact_new_count.numpy()[0]), 0)
        test.assertEqual(int(contacts.rigid_contact_broken_count.numpy()[0]), 0)


def test_contact_report_broken_indices(test, device):
    """Broken contact report must list old contacts that disappeared."""
    with wp.ScopedDevice(device):
        builder = newton.ModelBuilder()
        builder.add_ground_plane()
        for x in (-0.5, 0.5):
            b = builder.add_body(xform=wp.transform(wp.vec3(x, 0.0, 0.1)))
            builder.add_shape_sphere(body=b, radius=0.1)

        model = builder.finalize(device=device)
        state = model.state()

        pipeline = newton.CollisionPipeline(model, broad_phase="nxn", contact_matching="latest", contact_report=True)
        contacts = pipeline.contacts()

        # Frame 1: 2 sphere-plane contacts.
        count1 = _collide_once(pipeline, state, contacts)
        test.assertGreater(count1, 0)

        # Frame 2: move one sphere far away so its contact disappears.
        q = state.body_q.numpy()
        q[1][0:3] = [0.5, 0.0, 10.0]  # second sphere flies away
        state.body_q = wp.array(q, dtype=wp.transform, device=device)

        count2 = _collide_once(pipeline, state, contacts)
        test.assertLess(count2, count1, "Fewer contacts after removing a sphere")

        broken_count = contacts.rigid_contact_broken_count.numpy()[0]
        test.assertGreater(broken_count, 0, "Should have broken contacts from the removed sphere")

        # Broken indices must be valid positions in the OLD sorted buffer.
        broken_indices = contacts.rigid_contact_broken_indices.numpy()[:broken_count]
        test.assertTrue(
            np.all(broken_indices >= 0) and np.all(broken_indices < count1),
            f"Broken indices must be in [0, {count1}), got: {broken_indices}",
        )


def test_deterministic_implied(test, device):
    """Any non-disabled contact_matching mode should imply deterministic=True."""
    with wp.ScopedDevice(device):
        model, _state = _build_simple_scene(device)
        pipeline = newton.CollisionPipeline(model, broad_phase="nxn", contact_matching="latest")
        test.assertTrue(pipeline.deterministic)
        test.assertEqual(pipeline.contact_matching, "latest")
        test.assertEqual(pipeline._contact_sort_sub_key_bits, CONTACT_SORT_CONVEX_SUB_KEY_BITS)


def test_matching_sort_width_follows_contact_family(test, device):
    """Use compact persistent keys only for bounded manifold sub-keys."""
    with wp.ScopedDevice(device):
        model, _state = _build_simple_scene(device)
        primitive_pipeline = newton.CollisionPipeline(model, broad_phase="nxn", contact_matching="latest")
        test.assertEqual(primitive_pipeline._contact_sort_sub_key_bits, CONTACT_SORT_CONVEX_SUB_KEY_BITS)

        builder = newton.ModelBuilder()
        builder.add_ground_plane()
        body = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.5)))
        builder.add_shape_mesh(body, mesh=newton.Mesh.create_box(0.5, compute_inertia=False))
        mesh_model = builder.finalize(device=device)
        mesh_pipeline = newton.CollisionPipeline(mesh_model, broad_phase="nxn", contact_matching="latest")
        test.assertEqual(mesh_pipeline._contact_sort_sub_key_bits, CONTACT_SORT_SUB_KEY_BITS)


def test_contacts_exposes_matching_mode(test, device):
    """Expose the associated or most recent contact matching mode."""
    with wp.ScopedDevice(device):
        model, state = _build_simple_scene(device)
        test.assertEqual(newton.Contacts(0, 0, device=device).contact_matching_mode, "disabled")

        latest_pipeline = newton.CollisionPipeline(model, broad_phase="nxn", contact_matching="latest")
        contacts = latest_pipeline.contacts()

        test.assertEqual(contacts.contact_matching_mode, "latest")
        with test.assertRaises(AttributeError):
            contacts.contact_matching_mode = "sticky"

        incompatible_pipeline = newton.CollisionPipeline(
            model,
            broad_phase="nxn",
            rigid_contact_max=contacts.rigid_contact_max + 1,
            contact_matching="sticky",
        )
        with test.assertRaisesRegex(ValueError, "capacity"):
            incompatible_pipeline.collide(state, contacts)
        test.assertEqual(contacts.contact_matching_mode, "latest")

        for mode in ("sticky", "disabled"):
            pipeline = newton.CollisionPipeline(
                model,
                broad_phase="nxn",
                rigid_contact_max=contacts.rigid_contact_max,
                contact_matching=mode,
            )
            pipeline.collide(state, contacts)
            test.assertEqual(contacts.contact_matching_mode, mode)


def test_matching_disabled_no_allocation(test, device):
    """DISABLED mode: match_index and report arrays should be None."""
    with wp.ScopedDevice(device):
        model, _state = _build_simple_scene(device)
        pipeline = newton.CollisionPipeline(model, broad_phase="nxn", deterministic=True)
        contacts = pipeline.contacts()
        test.assertIsNone(contacts.rigid_contact_match_index)
        test.assertIsNone(contacts.rigid_contact_new_indices)
        test.assertIsNone(contacts.rigid_contact_broken_indices)
        test.assertEqual(pipeline.contact_matching, "disabled")


def test_match_index_valid_after_sort(test, device):
    """After sorting, match indices must be in valid range and unique."""
    with wp.ScopedDevice(device):
        model, state = _build_simple_scene(device)
        pipeline = newton.CollisionPipeline(model, broad_phase="nxn", contact_matching="latest")
        contacts = pipeline.contacts()

        _collide_once(pipeline, state, contacts)  # frame 1
        count = _collide_once(pipeline, state, contacts)  # frame 2

        match_idx = contacts.rigid_contact_match_index.numpy()[:count]
        matched = match_idx[match_idx >= 0]

        test.assertTrue(np.all(matched < count), f"Indices must be < {count}, max: {matched.max()}")
        test.assertEqual(len(np.unique(matched)), len(matched), "Matched indices must be unique")


def test_dynamic_body_world_transform(test, device):
    """Two dynamic spheres (no ground plane) must produce identity match.

    This exercises the ``body_q[bid]`` world-space transform path in both the
    match and save kernels (bid != -1), which the ground-plane tests skip.
    """
    with wp.ScopedDevice(device):
        builder = newton.ModelBuilder()
        ba = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.0)))
        builder.add_shape_sphere(body=ba, radius=0.1)
        bb = builder.add_body(xform=wp.transform(wp.vec3(0.19, 0.0, 0.0)))
        builder.add_shape_sphere(body=bb, radius=0.1)

        model = builder.finalize(device=device)
        state = model.state()

        # Verify shape0 is a dynamic body (not ground).
        sb = model.shape_body.numpy()
        test.assertNotEqual(sb[0], -1, "shape0 should be a dynamic body in this test")

        pipeline = newton.CollisionPipeline(model, broad_phase="nxn", contact_matching="latest")
        contacts = pipeline.contacts()

        count1 = _collide_once(pipeline, state, contacts)
        test.assertGreater(count1, 0)

        # Frame 2: identical state → identity match.
        count2 = _collide_once(pipeline, state, contacts)
        test.assertEqual(count1, count2)
        match_idx = contacts.rigid_contact_match_index.numpy()[:count2]
        np.testing.assert_array_equal(
            match_idx,
            np.arange(count2, dtype=np.int32),
            err_msg="Dynamic-body stable scene must produce identity match",
        )


def test_box_on_plane_multiple_contacts(test, device):
    """A box on a plane produces multiple contacts per shape pair (sub_keys 0-3).

    This verifies matching works when a single shape pair generates several
    contacts with distinct sort sub-keys, and that the identity invariant
    holds for all of them.
    """
    with wp.ScopedDevice(device):
        builder = newton.ModelBuilder()
        builder.add_ground_plane()
        b = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.15)))
        builder.add_shape_box(body=b, hx=0.1, hy=0.1, hz=0.1)

        model = builder.finalize(device=device)
        state = model.state()

        pipeline = newton.CollisionPipeline(model, broad_phase="nxn", contact_matching="latest")
        contacts = pipeline.contacts()

        count1 = _collide_once(pipeline, state, contacts)
        test.assertGreater(count1, 1, "Box on plane should produce multiple contacts")

        # Frame 2: identical state → identity match for all contacts.
        count2 = _collide_once(pipeline, state, contacts)
        test.assertEqual(count1, count2)
        match_idx = contacts.rigid_contact_match_index.numpy()[:count2]
        np.testing.assert_array_equal(
            match_idx,
            np.arange(count2, dtype=np.int32),
            err_msg="Box multi-contact stable scene must produce identity match",
        )


def test_invalid_mode_raises(test, device):
    """Invalid contact_matching values must raise ValueError."""
    with wp.ScopedDevice(device):
        model, _state = _build_simple_scene(device)

        with test.assertRaises(ValueError):
            newton.CollisionPipeline(model, broad_phase="nxn", contact_matching="bogus")

        with test.assertRaises(ValueError):
            # Booleans no longer accepted.
            newton.CollisionPipeline(model, broad_phase="nxn", contact_matching=True)


def test_contact_report_requires_matching(test, device):
    """contact_report=True requires a non-disabled matching mode."""
    with wp.ScopedDevice(device):
        model, _state = _build_simple_scene(device)
        with test.assertRaises(ValueError):
            newton.CollisionPipeline(
                model,
                broad_phase="nxn",
                contact_matching="disabled",
                contact_report=True,
            )


# ---------------------------------------------------------------------------
# Sticky mode tests
# ---------------------------------------------------------------------------


def test_sticky_matched_rows_replayed(test, device):
    """STICKY mode: matched rows carry exact previous-frame geometry even when
    the narrow phase's fresh output differs on a perturbed second frame.

    Frame 2 perturbs the bodies slightly (less than the match threshold) so
    the narrow phase produces a different-but-close contact record.  Sticky
    replay must overwrite ``point0``/``point1``/``offset0``/``offset1`` with
    the previous frame's values, so after frame 2 those columns equal the
    frame-1 snapshot even though the narrow phase would have produced
    something slightly different.
    """
    with wp.ScopedDevice(device):
        model, state = _build_simple_scene(device)
        pipeline = newton.CollisionPipeline(model, broad_phase="nxn", contact_matching="sticky")
        contacts = pipeline.contacts()

        count1 = _collide_once(pipeline, state, contacts)
        test.assertGreater(count1, 0)
        snap_point0 = contacts.rigid_contact_point0.numpy()[:count1].copy()
        snap_point1 = contacts.rigid_contact_point1.numpy()[:count1].copy()
        snap_offset0 = contacts.rigid_contact_offset0.numpy()[:count1].copy()
        snap_offset1 = contacts.rigid_contact_offset1.numpy()[:count1].copy()
        snap_normal = contacts.rigid_contact_normal.numpy()[:count1].copy()

        # Perturb every body by 0.1 mm in x -- well below the 0.5 mm default
        # pos threshold so every contact still matches, but enough for the
        # narrow phase to produce a detectably different fresh record.
        q = state.body_q.numpy()
        for i in range(len(q)):
            q[i][0] += 0.0001
        state.body_q = wp.array(q, dtype=wp.transform, device=device)

        # Also run the narrow phase on a fresh (non-sticky) pipeline with
        # the same state, so we can confirm the fresh contact values really
        # differ from frame 1 -- otherwise the sticky assertion below would
        # pass trivially.
        pipeline_fresh = newton.CollisionPipeline(model, broad_phase="nxn")
        contacts_fresh = pipeline_fresh.contacts()
        _collide_once(pipeline_fresh, state, contacts_fresh)
        fresh_point0 = contacts_fresh.rigid_contact_point0.numpy()[:count1]

        count2 = _collide_once(pipeline, state, contacts)
        test.assertEqual(count1, count2)
        match_idx = contacts.rigid_contact_match_index.numpy()[:count2]
        test.assertTrue(
            np.all(match_idx >= 0),
            f"All perturbed contacts should still match. Unique: {np.unique(match_idx)}",
        )

        # Sanity: fresh narrow phase really did produce different point0 values
        # on the perturbed frame, so the sticky assertion below is non-trivial.
        test.assertFalse(
            np.array_equal(fresh_point0, snap_point0),
            "Precondition: perturbation must change fresh narrow-phase point0",
        )

        # Sticky contract: replayed fields equal the frame-1 snapshot.
        for field, prev in (
            ("point0", snap_point0),
            ("point1", snap_point1),
            ("offset0", snap_offset0),
            ("offset1", snap_offset1),
            ("normal", snap_normal),
        ):
            current = getattr(contacts, f"rigid_contact_{field}").numpy()[:count2]
            np.testing.assert_array_equal(
                current,
                prev,
                err_msg=f"Sticky mode: matched rows must carry prev-frame {field} byte-for-byte",
            )


def test_sticky_unmatched_rows_pass_through(test, device):
    """STICKY mode: unmatched rows keep the current frame's narrow-phase data.

    Add a new sphere to the scene in frame 2.  Its contacts have
    match_index < 0, so sticky replay must NOT overwrite them — their
    shape indices must reflect the newly added shape.
    """
    with wp.ScopedDevice(device):
        builder = newton.ModelBuilder()
        builder.add_ground_plane()
        for x in (-0.5, 0.5):
            b = builder.add_body(xform=wp.transform(wp.vec3(x, 0.0, 0.1)))
            builder.add_shape_sphere(body=b, radius=0.1)
        # Third sphere parked out of the way for frame 1.
        b3 = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 10.0)))
        new_shape = builder.add_shape_sphere(body=b3, radius=0.1)

        model = builder.finalize(device=device)
        state = model.state()
        pipeline = newton.CollisionPipeline(model, broad_phase="nxn", contact_matching="sticky")
        contacts = pipeline.contacts()

        count1 = _collide_once(pipeline, state, contacts)
        test.assertGreater(count1, 0)

        # Bring the third sphere down onto the ground.
        q = state.body_q.numpy()
        q[2][0:3] = [0.0, 0.0, 0.1]
        state.body_q = wp.array(q, dtype=wp.transform, device=device)

        count2 = _collide_once(pipeline, state, contacts)
        test.assertGreater(count2, count1)

        match_idx = contacts.rigid_contact_match_index.numpy()[:count2]
        shape0 = contacts.rigid_contact_shape0.numpy()[:count2]
        shape1 = contacts.rigid_contact_shape1.numpy()[:count2]

        unmatched = match_idx < 0
        test.assertTrue(unmatched.any(), "Frame 2 must introduce at least one unmatched contact")

        # At least one unmatched row must reference the newly added shape,
        # proving sticky replay did not overwrite new contacts with stale data.
        involves_new = (shape0 == new_shape) | (shape1 == new_shape)
        test.assertTrue(
            (involves_new & unmatched).any(),
            "Unmatched rows must pass through the new narrow-phase contacts for the new shape",
        )

        # Sanity: matched rows still carry valid shape indices (not -1 from
        # the default-fill sentinel).
        matched_mask = match_idx >= 0
        test.assertTrue(
            np.all(shape0[matched_mask] >= 0) and np.all(shape1[matched_mask] >= 0),
            "Matched rows must have non-sentinel shape indices after replay",
        )


def test_sticky_disabled_no_sticky_buffers(test, device):
    """LATEST and DISABLED modes must not allocate sticky buffers."""
    with wp.ScopedDevice(device):
        model, _state = _build_simple_scene(device)

        p_latest = newton.CollisionPipeline(model, broad_phase="nxn", contact_matching="latest")
        test.assertIsNotNone(p_latest._contact_matcher)
        test.assertFalse(p_latest._contact_matcher.is_sticky)
        test.assertIsNone(p_latest._contact_matcher._prev_point0)
        test.assertIsNone(p_latest._contact_matcher._prev_point1)
        test.assertIsNone(p_latest._contact_matcher._prev_offset0)
        test.assertIsNone(p_latest._contact_matcher._prev_offset1)

        p_off = newton.CollisionPipeline(model, broad_phase="nxn", contact_matching="disabled")
        test.assertIsNone(p_off._contact_matcher)

        p_sticky = newton.CollisionPipeline(model, broad_phase="nxn", contact_matching="sticky")
        test.assertTrue(p_sticky._contact_matcher.is_sticky)
        test.assertIsNotNone(p_sticky._contact_matcher._prev_point0)
        test.assertIsNotNone(p_sticky._contact_matcher._prev_point1)
        test.assertIsNotNone(p_sticky._contact_matcher._prev_offset0)
        test.assertIsNotNone(p_sticky._contact_matcher._prev_offset1)


# ---------------------------------------------------------------------------
# Register tests
# ---------------------------------------------------------------------------

devices = get_test_devices()
cuda_devices = get_cuda_test_devices()

add_function_test(
    TestContactMatching, "test_first_frame_all_not_found", test_first_frame_all_not_found, devices=devices
)
add_function_test(
    TestContactMatching, "test_stable_scene_identity_match", test_stable_scene_identity_match, devices=devices
)
add_function_test(
    TestContactMatching,
    "test_stable_scene_identity_across_three_frames",
    test_stable_scene_identity_across_three_frames,
    devices=devices,
)
add_function_test(
    TestContactMatching,
    "test_save_initializes_next_frame_claims",
    test_save_initializes_next_frame_claims,
    devices=devices,
)
add_function_test(
    TestContactMatching,
    "test_masked_reset_restarts_only_selected_contact_history",
    test_masked_reset_restarts_only_selected_contact_history,
    devices=devices,
)
add_function_test(
    TestContactMatching,
    "test_reset_rejects_invalid_world_masks",
    test_reset_rejects_invalid_world_masks,
    devices=get_test_devices(mode="basic"),
)
add_function_test(
    TestContactMatching,
    "test_reset_accumulates_live_mask_during_cuda_graph_replay",
    test_reset_accumulates_live_mask_during_cuda_graph_replay,
    devices=cuda_devices,
)
add_function_test(TestContactMatching, "test_new_contact_detection", test_new_contact_detection, devices=devices)
add_function_test(
    TestContactMatching,
    "test_broken_pos_threshold_all_contacts",
    test_broken_pos_threshold_all_contacts,
    devices=devices,
)
add_function_test(
    TestContactMatching,
    "test_within_pos_threshold_still_matches",
    test_within_pos_threshold_still_matches,
    devices=devices,
)
add_function_test(TestContactMatching, "test_broken_normal_threshold", test_broken_normal_threshold, devices=devices)
add_function_test(
    TestContactMatching, "test_contact_report_indices_correct", test_contact_report_indices_correct, devices=devices
)
add_function_test(
    TestContactMatching,
    "test_save_resets_next_frame_report_flags",
    test_save_resets_next_frame_report_flags,
    devices=devices,
)
add_function_test(
    TestContactMatching, "test_contact_report_broken_indices", test_contact_report_broken_indices, devices=devices
)
add_function_test(TestContactMatching, "test_deterministic_implied", test_deterministic_implied, devices=devices)
add_function_test(
    TestContactMatching, "test_contacts_exposes_matching_mode", test_contacts_exposes_matching_mode, devices=devices
)
add_function_test(
    TestContactMatching, "test_matching_disabled_no_allocation", test_matching_disabled_no_allocation, devices=devices
)
add_function_test(
    TestContactMatching, "test_match_index_valid_after_sort", test_match_index_valid_after_sort, devices=devices
)
add_function_test(
    TestContactMatching, "test_dynamic_body_world_transform", test_dynamic_body_world_transform, devices=devices
)
add_function_test(
    TestContactMatching, "test_box_on_plane_multiple_contacts", test_box_on_plane_multiple_contacts, devices=devices
)
add_function_test(TestContactMatching, "test_invalid_mode_raises", test_invalid_mode_raises, devices=devices)
add_function_test(
    TestContactMatching, "test_contact_report_requires_matching", test_contact_report_requires_matching, devices=devices
)

add_function_test(
    TestContactMatchingSticky, "test_sticky_matched_rows_replayed", test_sticky_matched_rows_replayed, devices=devices
)
add_function_test(
    TestContactMatchingSticky,
    "test_sticky_unmatched_rows_pass_through",
    test_sticky_unmatched_rows_pass_through,
    devices=devices,
)
add_function_test(
    TestContactMatchingSticky,
    "test_sticky_disabled_no_sticky_buffers",
    test_sticky_disabled_no_sticky_buffers,
    devices=devices,
)

if __name__ == "__main__":
    wp.clear_kernel_cache()
    unittest.main(verbosity=2)
