# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
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

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.geometry.contact_matcher import ContactMatcher
from newton._src.sim.collide_unified import BroadPhaseMode, CollisionPipelineUnified
from newton.tests.unittest_utils import add_function_test, get_test_devices

wp.config.quiet = True


class TestContactMatcher(unittest.TestCase):
    pass


def test_contact_matcher_basic(test: TestContactMatcher, device):
    """Test basic contact matching with simple key/payload pairs."""
    max_contacts = 10
    matcher = ContactMatcher(max_contacts, device=device)

    # Create test data for timestep 0
    keys_0 = wp.array([1, 2, 3, 4, 5], dtype=wp.uint64, device=device)
    payloads_0 = wp.array([10, 20, 30, 40, 50], dtype=wp.uint32, device=device)
    num_keys_0 = wp.array([5], dtype=wp.int32, device=device)
    result_map_0 = wp.zeros(max_contacts, dtype=wp.int32, device=device)

    # First timestep - no previous contacts, all should be -1
    matcher.launch(keys_0, num_keys_0, payloads_0, result_map_0, device=device)
    wp.synchronize_device(device)
    result_0 = result_map_0.numpy()
    test.assertTrue(np.all(result_0[:5] == -1), "First timestep should have all new contacts")

    # Create test data for timestep 1 - some contacts persist, some are new
    keys_1 = wp.array([2, 3, 4, 6, 7], dtype=wp.uint64, device=device)
    payloads_1 = wp.array([20, 30, 40, 60, 70], dtype=wp.uint32, device=device)
    num_keys_1 = wp.array([5], dtype=wp.int32, device=device)
    result_map_1 = wp.zeros(max_contacts, dtype=wp.int32, device=device)

    # Second timestep - should match contacts 2, 3, 4
    matcher.launch(keys_1, num_keys_1, payloads_1, result_map_1, device=device)
    wp.synchronize_device(device)
    result_1 = result_map_1.numpy()

    # Contacts at indices 0, 1, 2 should match (keys 2, 3, 4 from previous step at indices 1, 2, 3)
    test.assertEqual(result_1[0], 1, "Key 2 should match previous index 1")
    test.assertEqual(result_1[1], 2, "Key 3 should match previous index 2")
    test.assertEqual(result_1[2], 3, "Key 4 should match previous index 3")
    # Contacts at indices 3, 4 are new (keys 6, 7)
    test.assertEqual(result_1[3], -1, "Key 6 should be new")
    test.assertEqual(result_1[4], -1, "Key 7 should be new")


def test_contact_matcher_duplicate_keys(test: TestContactMatcher, device):
    """Test contact matching with duplicate keys (differentiated by payload)."""
    max_contacts = 10
    matcher = ContactMatcher(max_contacts, device=device)

    # Create test data with duplicate keys but different payloads
    keys_0 = wp.array([1, 1, 1, 2, 2], dtype=wp.uint64, device=device)
    payloads_0 = wp.array([10, 20, 30, 40, 50], dtype=wp.uint32, device=device)
    num_keys_0 = wp.array([5], dtype=wp.int32, device=device)
    result_map_0 = wp.zeros(max_contacts, dtype=wp.int32, device=device)

    # First timestep
    matcher.launch(keys_0, num_keys_0, payloads_0, result_map_0, device=device)
    wp.synchronize_device(device)

    # Second timestep - same keys, some payloads match
    keys_1 = wp.array([1, 1, 2, 2, 3], dtype=wp.uint64, device=device)
    payloads_1 = wp.array([20, 30, 40, 60, 70], dtype=wp.uint32, device=device)
    num_keys_1 = wp.array([5], dtype=wp.int32, device=device)
    result_map_1 = wp.zeros(max_contacts, dtype=wp.int32, device=device)

    matcher.launch(keys_1, num_keys_1, payloads_1, result_map_1, device=device)
    wp.synchronize_device(device)
    result_1 = result_map_1.numpy()

    # Key 1 with payload 20 should match previous index 1
    test.assertEqual(result_1[0], 1, "Key 1, payload 20 should match")
    # Key 1 with payload 30 should match previous index 2
    test.assertEqual(result_1[1], 2, "Key 1, payload 30 should match")
    # Key 2 with payload 40 should match previous index 3
    test.assertEqual(result_1[2], 3, "Key 2, payload 40 should match")
    # Key 2 with payload 60 is new (different payload)
    test.assertEqual(result_1[3], -1, "Key 2, payload 60 should be new")
    # Key 3 is completely new
    test.assertEqual(result_1[4], -1, "Key 3 should be new")


def test_contact_matcher_stacked_cubes(test: TestContactMatcher, device):
    """Test contact matcher with realistic scenario: 2D grid of rotated stacked cubes in static equilibrium.

    This test verifies that:
    1. Frame 0: Initial contacts are detected, but none can be matched (no previous frame)
    2. Frame 1+: ALL contacts should be matched (100%) in a static configuration
       using (shape_pair_key, feature_id) pairs.

    The cubes are at rest, so geometry and contact features should be identical
    frame-to-frame. If contacts are not matched, the test prints detailed debug info
    showing positions, pair keys, and feature keys for analysis.

    Scene: 5x5 grid where each position has 5 stacked cubes, each rotated 30 degrees
    more than the one below it (125 total cubes).
    """
    # Build model with ground plane and 2D grid of stacked cubes
    # Use very high stiffness and damping to minimize motion
    builder = newton.ModelBuilder()
    builder.default_shape_cfg.ke = 1e6  # Very stiff contacts
    builder.default_shape_cfg.kd = 1e4  # High damping to prevent oscillation

    # Ground plane
    builder.add_ground_plane()

    # Parameters for 2D grid of stacked cubes
    cube_size = 0.5
    n_cubes_per_stack = 4  # Number of cubes in each stack
    s = 3  # Grid dimension in x
    t = 3  # Grid dimension in y
    spacing = 2.0  # Spacing between stacks
    rotation_increment = np.pi / 6.0  # 30 degrees in radians

    # Create 2D grid of stacks
    for i in range(s):
        for j in range(t):
            # Calculate base position for this stack
            x_base = (i - (s - 1) / 2.0) * spacing
            y_base = (j - (t - 1) / 2.0) * spacing

            # Create stack of n cubes at this grid position
            for k in range(n_cubes_per_stack):
                # Z position (stacked vertically)
                z_pos = cube_size / 2 + k * cube_size

                # Rotation increases by 30 degrees for each level
                rotation_angle = k * rotation_increment
                rotation = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), rotation_angle)

                # Create cube
                cube = builder.add_body(xform=wp.transform([x_base, y_base, z_pos], rotation))
                builder.add_shape_box(
                    body=cube,
                    hx=cube_size / 2,
                    hy=cube_size / 2,
                    hz=cube_size / 2,
                )

    model = builder.finalize(device=device)
    model.ground = True

    # Create states
    state_0 = model.state()
    state_1 = model.state()

    # Initialize contact matcher
    max_contacts = 1000
    matcher = ContactMatcher(max_contacts, device=device)

    # Create collision pipeline with contact matching enabled
    collision_pipeline = CollisionPipelineUnified.from_model(
        model,
        rigid_contact_margin=0.1,
        broad_phase_mode=BroadPhaseMode.NXN,
        enable_contact_matching=True,
    )

    # Simulate for a few timesteps and track contacts
    solver = newton.solvers.SolverXPBD(model, iterations=2)
    sim_dt = 1.0 / 60.0

    contact_counts = []
    persistent_counts = []
    contact_history = []  # Store detailed contact info for debugging

    for frame in range(5):
        # Get contacts using unified pipeline
        contacts = collision_pipeline.collide(model, state_0)
        num_contacts = collision_pipeline.narrow_contact_count.numpy()[0]
        contact_counts.append(num_contacts)

        frame_data = {
            "frame": frame,
            "num_contacts": num_contacts,
            "positions": [],
            "pair_keys": [],
            "feature_keys": [],
            "match_results": [],
        }

        if num_contacts > 0:
            # Use contact_pair_key and contact_key directly from narrow phase
            keys_wp = collision_pipeline.narrow_contact_pair_key
            payloads_wp = collision_pipeline.narrow_contact_key
            num_keys_wp = collision_pipeline.narrow_contact_count
            result_map = wp.zeros(max_contacts, dtype=wp.int32, device=device)

            # Collect data for debugging
            positions_np = collision_pipeline.narrow_contact_position.numpy()[:num_contacts]
            pair_keys_np = keys_wp.numpy()[:num_contacts]
            feature_keys_np = payloads_wp.numpy()[:num_contacts]

            # Match contacts
            matcher.launch(keys_wp, num_keys_wp, payloads_wp, result_map, device=device)
            wp.synchronize_device(device)
            result = result_map.numpy()[:num_contacts]

            # Store frame data
            frame_data["positions"] = positions_np.copy()
            frame_data["pair_keys"] = pair_keys_np.copy()
            frame_data["feature_keys"] = feature_keys_np.copy()
            frame_data["match_results"] = result.copy()

            # Count persistent contacts (matched to previous timestep)
            persistent = np.sum(result >= 0)
            persistent_counts.append(persistent)
        else:
            persistent_counts.append(0)

        contact_history.append(frame_data)

        # Step simulation
        state_0.clear_forces()
        solver.step(state_0, state_1, model.control(), contacts, sim_dt)
        state_0, state_1 = state_1, state_0

    # Verify results
    test.assertGreater(contact_counts[0], 0, "Frame 0: Should have contacts in first timestep")
    test.assertEqual(persistent_counts[0], 0, "Frame 0: First timestep should have no previous contacts to match")

    # Starting from frame 1, ALL contacts should be matched in static configuration
    # The cubes are at rest, so contact features must persist frame-to-frame
    if len(persistent_counts) > 1:
        for i in range(1, len(persistent_counts)):
            if contact_counts[i] > 0:
                persistence_ratio = persistent_counts[i] / contact_counts[i]
                if persistent_counts[i] != contact_counts[i]:
                    # Print detailed contact history for debugging
                    print(f"\n=== CONTACT MATCHING FAILURE at Frame {i} ===")
                    print(f"Expected: {contact_counts[i]} matched, Got: {persistent_counts[i]}")
                    print(f"Persistence ratio: {persistence_ratio:.2%}\n")

                    # Print previous frame and current frame contact details
                    for frame_idx in [i - 1, i]:
                        frame_data = contact_history[frame_idx]
                        print(f"--- Frame {frame_idx} ({frame_data['num_contacts']} contacts) ---")
                        for c in range(frame_data["num_contacts"]):
                            pos = frame_data["positions"][c]
                            pair_key = frame_data["pair_keys"][c]
                            feature_key = frame_data["feature_keys"][c]
                            match_result = frame_data["match_results"][c] if frame_idx == i else "N/A"
                            print(
                                f"  Contact {c}: pos=({pos[0]:.6f}, {pos[1]:.6f}, {pos[2]:.6f}) "
                                f"pair_key=0x{pair_key:016x} feature_key={feature_key} match={match_result}"
                            )
                        print()

                test.assertEqual(
                    persistent_counts[i],
                    contact_counts[i],
                    f"Frame {i}: ALL {contact_counts[i]} contacts should be matched in static configuration "
                    f"(found {persistent_counts[i]}, ratio: {persistence_ratio:.2%})",
                )


def test_contact_matcher_empty_contacts(test: TestContactMatcher, device):
    """Test contact matcher with zero contacts."""
    max_contacts = 10
    matcher = ContactMatcher(max_contacts, device=device)

    # Create empty contact data
    keys = wp.zeros(max_contacts, dtype=wp.uint64, device=device)
    payloads = wp.zeros(max_contacts, dtype=wp.uint32, device=device)
    num_keys = wp.array([0], dtype=wp.int32, device=device)
    result_map = wp.zeros(max_contacts, dtype=wp.int32, device=device)

    # Should handle empty contacts gracefully
    matcher.launch(keys, num_keys, payloads, result_map, device=device)

    # No assertion needed - just verify it doesn't crash
    test.assertTrue(True, "Should handle empty contacts without crashing")


def test_contact_matcher_max_capacity(test: TestContactMatcher, device):
    """Test contact matcher at maximum capacity."""
    max_contacts = 20
    matcher = ContactMatcher(max_contacts, device=device)

    # Fill to capacity
    keys_0 = wp.array(list(range(max_contacts)), dtype=wp.uint64, device=device)
    payloads_0 = wp.array(list(range(100, 100 + max_contacts)), dtype=wp.uint32, device=device)
    num_keys_0 = wp.array([max_contacts], dtype=wp.int32, device=device)
    result_map_0 = wp.zeros(max_contacts, dtype=wp.int32, device=device)

    # First timestep
    matcher.launch(keys_0, num_keys_0, payloads_0, result_map_0, device=device)
    wp.synchronize_device(device)

    # Second timestep - all contacts persist
    matcher.launch(keys_0, num_keys_0, payloads_0, result_map_0, device=device)
    wp.synchronize_device(device)
    result = result_map_0.numpy()

    # All contacts should match their previous indices
    expected = np.arange(max_contacts, dtype=np.int32)
    test.assertTrue(
        np.array_equal(result, expected), "All contacts at max capacity should match their previous indices"
    )


# Register tests for all devices
devices = get_test_devices()
for device in devices:
    add_function_test(
        TestContactMatcher, f"test_contact_matcher_basic_{device.alias}", test_contact_matcher_basic, devices=[device]
    )
    add_function_test(
        TestContactMatcher,
        f"test_contact_matcher_duplicate_keys_{device.alias}",
        test_contact_matcher_duplicate_keys,
        devices=[device],
    )
    add_function_test(
        TestContactMatcher,
        f"test_contact_matcher_stacked_cubes_{device.alias}",
        test_contact_matcher_stacked_cubes,
        devices=[device],
    )
    add_function_test(
        TestContactMatcher,
        f"test_contact_matcher_empty_contacts_{device.alias}",
        test_contact_matcher_empty_contacts,
        devices=[device],
    )
    add_function_test(
        TestContactMatcher,
        f"test_contact_matcher_max_capacity_{device.alias}",
        test_contact_matcher_max_capacity,
        devices=[device],
    )


if __name__ == "__main__":
    unittest.main(verbosity=2)
