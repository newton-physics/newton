# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.geometry.hashtable import hashtable_find_or_insert
from newton._src.geometry.sdf_planar_recovery import (
    MeshRecoveryContact,
    MeshRecoveryData,
    PlanarExit,
    _pair_key,
    active_recovery_contact,
    allocate_mesh_recovery,
    planar_exit,
    reset_mesh_recovery,
)
from newton._src.geometry.sdf_texture import TextureSDFData
from newton.tests.unittest_utils import add_function_test, get_test_devices


@wp.kernel(enable_backward=False)
def _find_exit(
    wall: wp.uint64, probe: wp.uint64, offset: float, flags: wp.array[wp.uint8], exits: wp.array[PlanarExit]
):
    exits[0] = planar_exit(
        wall,
        probe,
        wp.transform(wp.vec3(offset, 0.0, 0.0), wp.quat_identity()),
        wp.vec3(1.0),
        wp.vec3(1.0),
        wp.vec3(-0.004),
        wp.vec3(0.004),
        wp.vec3(offset, 0.0, 0.0) - wp.vec3(0.004),
        wp.vec3(offset, 0.0, 0.0) + wp.vec3(0.004),
        TextureSDFData(),
        TextureSDFData(),
        flags,
        -1,
    )


def test_thin_wall_exit(test, device):
    """Accept a unique exit only when its swept and lateral corridor is clear."""
    wall = newton.Mesh.create_box(0.0005, 0.015, 0.015, duplicate_vertices=False)
    probe = newton.Mesh.create_capsule(0.004, 0.0, segments=16)
    dense = newton.Mesh.create_capsule(0.004, 0.0, segments=48)
    flat_vertices = dense.vertices.copy()
    flat_vertices[:, 0] = np.minimum(flat_vertices[:, 0], 0.0039)
    broad = newton.Mesh(flat_vertices, dense.indices)
    obstacle = newton.Mesh.create_box(0.0002, 0.0002, 0.0002, duplicate_vertices=False)
    # The obstacle is outside the initial probe bounds but inside its swept footprint.
    obstructed = newton.Mesh(
        np.vstack((wall.vertices, obstacle.vertices + np.array([-0.005, 0.0005, 0.0]))),
        np.concatenate((wall.indices, obstacle.indices + len(wall.vertices))),
    )
    # A held row may travel tangentially by one eighth of the wall thickness.
    side = newton.Mesh.create_box(0.00002, 0.00002, 0.00002, duplicate_vertices=False)
    obstructed_side = newton.Mesh(
        np.vstack((wall.vertices, side.vertices + np.array([-0.002, 0.00408, 0.0]))),
        np.concatenate((wall.indices, side.indices + len(wall.vertices))),
    )
    exits = wp.empty(1, dtype=PlanarExit, device=device)
    flags = wp.empty(0, dtype=wp.uint8, device=device)
    for mesh, tip, offset, expected in (
        (wall, probe, -0.0005, True),
        (wall, dense, -0.0005, True),
        (wall, broad, -0.0005, False),
        (wall, probe, 0.0, False),
        (obstructed, probe, -0.0005, False),
        (obstructed_side, probe, -0.0005, False),
    ):
        with test.subTest(offset=offset, vertices=len(tip.vertices), broad=tip is broad, obstructed=mesh is not wall):
            wp.launch(
                _find_exit,
                1,
                inputs=[mesh.finalize(device), tip.finalize(device), offset, flags, exits],
                device=device,
            )
            result = exits.numpy()[0]
            test.assertEqual(bool(result["valid"]), expected)
            if expected:
                test.assertAlmostEqual(result["depth"], 0.004, places=6)
                test.assertLess(result["normal"][0], -0.99)


@wp.kernel(enable_backward=False)
def _check_held_contact(recovery: MeshRecoveryData, result: wp.array[int]):
    slot = hashtable_find_or_insert(_pair_key(0, 1), recovery.keys, recovery.active_slots)
    contact = MeshRecoveryContact()
    normal = wp.vec3(-1.0, 0.0, 0.0)
    contact.thickness = 0.001
    contact.initial_depth = 0.004
    contact.rotation0 = wp.quat_identity()
    contact.rotation1 = wp.quat_identity()
    recovery.contacts[slot] = contact
    for i in range(5):
        pose = wp.transform_identity()
        if i == 1:
            pose.p = wp.vec3(-0.001, 0.0, 0.0)
        elif i == 2:
            pose.p = wp.vec3(0.0, 0.0001, 0.0)
        elif i == 3:
            pose.q = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), 0.1)
        elif i == 4:
            pose.p = wp.vec3(0.0, 0.0002, 0.0)
        result[i] = active_recovery_contact(recovery, 0, 0, 1, wp.transform_identity(), pose, normal)
    # A fresh rotated contact must survive the precision of world-space
    # transforms when its coordinates are large relative to its thickness.
    rotated = wp.transform(wp.vec3(100.125, 3.5, 0.0), wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), 0.8))
    contact.relative_translation = rotated.p
    contact.rotation1 = rotated.q
    recovery.contacts[slot] = contact
    result[5] = active_recovery_contact(
        recovery, 0, 0, 1, wp.transform_identity(), rotated, wp.quat_rotate(rotated.q, normal)
    )
    result[6] = active_recovery_contact(
        recovery, 0, 1, 0, rotated, wp.transform_identity(), -wp.quat_rotate(rotated.q, normal)
    )
    reset_mesh_recovery(recovery)
    result[7] = active_recovery_contact(recovery, 0, 0, 1, wp.transform_identity(), wp.transform_identity(), normal)


def test_held_contact_corridor(test, device):
    """Use a held row only while the current poses remain inside its proven corridor."""
    recovery = allocate_mesh_recovery(8, device)
    result = wp.empty(8, dtype=int, device=device)
    wp.launch(_check_held_contact, 1, inputs=[recovery, result], device=device)
    slots = result.numpy()
    test.assertGreaterEqual(slots[0], 0)
    test.assertEqual(slots[1], slots[0])
    np.testing.assert_array_equal(slots[2:], [slots[0], -2, -2, slots[0], slots[0], -1])
    test.assertTrue(np.all(recovery.keys.numpy() == np.iinfo(np.uint64).max))


class TestSDFPlanarRecovery(unittest.TestCase):
    pass


add_function_test(TestSDFPlanarRecovery, "test_thin_wall_exit", test_thin_wall_exit, devices=get_test_devices())
add_function_test(
    TestSDFPlanarRecovery, "test_held_contact_corridor", test_held_contact_corridor, devices=get_test_devices()
)


if __name__ == "__main__":
    unittest.main(verbosity=2)
