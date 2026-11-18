# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.geometry.hashtable import hashtable_find_or_insert
from newton._src.geometry.sdf_planar_recovery import (
    MeshRecoveryContact,
    MeshRecoveryContacts,
    MeshRecoveryData,
    PlanarExit,
    _pair_key,
    active_recovery_contact,
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
        TextureSDFData(),
        TextureSDFData(),
        flags,
        -1,
    )


def test_thin_wall_exit(test, device):
    wall = newton.Mesh.create_box(0.0005, 0.015, 0.015, duplicate_vertices=False)
    probe = newton.Mesh.create_capsule(0.004, 0.0, segments=16)
    obstacle = newton.Mesh.create_box(0.0002, 0.0002, 0.0002, duplicate_vertices=False)
    # The obstacle is outside the initial probe bounds but inside its swept footprint.
    obstructed = newton.Mesh(
        np.vstack((wall.vertices, obstacle.vertices + np.array([-0.005, 0.0005, 0.0]))),
        np.concatenate((wall.indices, obstacle.indices + len(wall.vertices))),
    )
    exits = wp.empty(1, dtype=PlanarExit, device=device)
    flags = wp.empty(0, dtype=wp.uint8, device=device)
    for mesh, offset, expected in ((wall, -0.0005, True), (wall, 0.0, False), (obstructed, -0.0005, False)):
        with test.subTest(offset=offset, obstructed=mesh is obstructed):
            wp.launch(
                _find_exit,
                1,
                inputs=[mesh.finalize(device), probe.finalize(device), offset, flags, exits],
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
    for i in range(4):
        pose = wp.transform_identity()
        if i == 1:
            pose.p = wp.vec3(-0.001, 0.0, 0.0)
        elif i == 2:
            pose.p = wp.vec3(0.0, 0.0001, 0.0)
        elif i == 3:
            pose.q = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), 0.1)
        result[i] = active_recovery_contact(recovery, 0, 0, 1, wp.transform_identity(), pose, normal)
    # A fresh rotated contact must survive the precision of world-space
    # transforms when its coordinates are large relative to its thickness.
    rotated = wp.transform(wp.vec3(100.125, 3.5, 0.0), wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), 0.8))
    contact.relative_translation = rotated.p
    contact.rotation1 = rotated.q
    recovery.contacts[slot] = contact
    result[4] = active_recovery_contact(
        recovery, 0, 0, 1, wp.transform_identity(), rotated, wp.quat_rotate(rotated.q, normal)
    )
    reset_mesh_recovery(recovery)
    result[5] = active_recovery_contact(recovery, 0, 0, 1, wp.transform_identity(), wp.transform_identity(), normal)


def test_held_contact_corridor(test, device):
    recovery = MeshRecoveryContacts(8, device)
    result = wp.empty(6, dtype=int, device=device)
    wp.launch(_check_held_contact, 1, inputs=[recovery.data, result], device=device)
    slots = result.numpy()
    test.assertGreaterEqual(slots[0], 0)
    test.assertEqual(slots[1], slots[0])
    np.testing.assert_array_equal(slots[2:], [-2, -2, slots[0], -1])
    test.assertTrue(np.all(recovery.table.keys.numpy() == np.iinfo(np.uint64).max))


class TestSDFPlanarRecovery(unittest.TestCase):
    pass


add_function_test(TestSDFPlanarRecovery, "test_thin_wall_exit", test_thin_wall_exit, devices=get_test_devices())
add_function_test(
    TestSDFPlanarRecovery, "test_held_contact_corridor", test_held_contact_corridor, devices=get_test_devices()
)


if __name__ == "__main__":
    unittest.main(verbosity=2)
