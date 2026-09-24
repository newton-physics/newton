# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Verify stable native-contact compaction without losing fields or diagnostics."""

import dataclasses
import unittest

import numpy as np
import warp as wp

from newton._src.solvers.mujoco.heterogeneous import HeterogeneousContactFilter
from newton.tests.unittest_utils import get_test_devices


def _make_filter(device, *, capacity=8):
    """Create native contact buffers and two shared filtering variants."""
    import mujoco
    import mujoco_warp

    model = mujoco.MjModel.from_xml_string("""
        <mujoco>
          <worldbody>
            <geom type="plane" size="2 2 .1"/>
            <body pos="0 0 1">
              <freejoint/>
              <geom type="sphere" size=".1" pos="-.2 0 0"/>
              <geom type="sphere" size=".1" pos=".2 0 0"/>
            </body>
          </worldbody>
        </mujoco>
    """)
    with wp.ScopedDevice(device):
        mjw_model = mujoco_warp.put_model(model)
        data = mujoco_warp.put_data(
            model, mujoco.MjData(model), nworld=3, naconmax=capacity, naccdmax=capacity, njmax=64
        )
        mjw_model.opt.warn_overflow = False
        allowed = wp.array([[True, False, False], [False, True, False]], dtype=bool)
        world_to_filter = wp.array([0, 1, 0], dtype=int)
        contact_filter = HeterogeneousContactFilter(data, allowed, 3, world_to_filter=world_to_filter)
    return mjw_model, data, allowed, contact_filter


def _seed_contacts(data):
    """Give every contact field distinct bits and include reversed geom pairs."""
    for field_id, field in enumerate(dataclasses.fields(data.contact)):
        array = getattr(data.contact, field.name)
        if array.size == 0:
            continue
        values = array.numpy().copy()
        words = values.view(np.uint32).reshape(data.naconmax, -1)
        words[:] = np.arange(words.size, dtype=np.uint32).reshape(words.shape) + 10000 * field_id
        array.assign(values)
    data.contact.geom.assign(np.array([[0, 1], [0, 2], [0, 2], [0, 1], [1, 0], [0, 2], [2, 0], [1, 2]], dtype=np.int32))
    data.contact.worldid.assign(np.array([0, 0, 1, 1, 2, 2, 1, 2], dtype=np.int32))
    data.nacon.fill_(8)
    return {
        field.name: getattr(data.contact, field.name).numpy().copy()
        for field in dataclasses.fields(data.contact)
        if getattr(data.contact, field.name).size > 0
    }


def _assert_fields(test, data, original, indices):
    """Compare each retained field bitwise, independent of its scalar type."""
    test.assertEqual(int(data.nacon.numpy()[0]), len(indices))
    for name, values in original.items():
        actual = getattr(data.contact, name).numpy()[: len(indices)]
        expected = values[indices]
        np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32), err_msg=name)


class TestMuJoCoHeterogeneousFilter(unittest.TestCase):
    def test_stable_compaction_preserves_every_field(self):
        """Keep all field bits in source order across shared filtering variants."""
        for device in get_test_devices():
            with self.subTest(device=device), wp.ScopedDevice(device):
                model, data, _, contact_filter = _make_filter(device)
                original = _seed_contacts(data)
                contact_filter(model, data)
                _assert_fields(self, data, original, [0, 2, 4, 6])

    def test_empty_all_and_no_contacts(self):
        """Handle an empty count and fully accepting or rejecting pair tables."""
        for device in get_test_devices():
            with self.subTest(device=device), wp.ScopedDevice(device):
                model, data, allowed, contact_filter = _make_filter(device)
                for count, allow, indices in ((0, True, []), (8, True, list(range(8))), (8, False, [])):
                    original = _seed_contacts(data)
                    data.nacon.fill_(count)
                    allowed.fill_(allow)
                    contact_filter(model, data)
                    _assert_fields(self, data, original, indices)

    def test_compaction_preserves_collision_overflows(self):
        """Retain native overflow flags before compaction hides the raw counts."""
        from mujoco_warp import OverflowType

        for device in get_test_devices():
            with self.subTest(device=device), wp.ScopedDevice(device):
                model, data, _, contact_filter = _make_filter(device)
                original = _seed_contacts(data)
                data.nacon.fill_(data.naconmax + 7)
                data.ncollision.fill_(data.naconmax + 3)
                data.overflow.fill_(int(OverflowType.CCD))
                contact_filter(model, data)
                _assert_fields(self, data, original, [0, 2, 4, 6])
                expected = int(OverflowType.CCD | OverflowType.BROADPHASE | OverflowType.NARROWPHASE)
                np.testing.assert_array_equal(data.overflow.numpy(), np.full(data.nworld, expected))

    def test_zero_capacity(self):
        """Accept empty native contact buffers on CPU and CUDA graph replay."""
        for device in get_test_devices():
            with self.subTest(device=device), wp.ScopedDevice(device):
                model, data, _, contact_filter = _make_filter(device, capacity=0)
                contact_filter(model, data)
                self.assertEqual(int(data.nacon.numpy()[0]), 0)
                if wp.get_device(device).is_cuda:
                    with wp.ScopedCapture(device=device) as capture:
                        contact_filter(model, data)
                    wp.capture_launch(capture.graph)
                    self.assertEqual(int(data.nacon.numpy()[0]), 0)

    def test_graph_replay_observes_changed_filter_rules(self):
        """Replay accepting and rejecting branches without recapturing contact buffers."""
        if not wp.is_cuda_available():
            self.skipTest("CUDA graph replay requires a CUDA device")
        device = wp.get_device("cuda:0")
        with wp.ScopedDevice(device):
            model, data, allowed, contact_filter = _make_filter(device)
            original = _seed_contacts(data)
            allowed.fill_(True)
            contact_filter(model, data)
            with wp.ScopedCapture(device=device) as capture:
                contact_filter(model, data)
            for allow, indices in ((True, list(range(8))), (False, []), (True, list(range(8)))):
                for name, values in original.items():
                    getattr(data.contact, name).assign(values)
                data.nacon.fill_(8)
                allowed.fill_(allow)
                wp.capture_launch(capture.graph)
                _assert_fields(self, data, original, indices)
            for name, values in original.items():
                getattr(data.contact, name).assign(values)
            data.nacon.fill_(8)
            allowed.assign(np.array([[True, False, False], [False, True, False]]))
            wp.capture_launch(capture.graph)
            _assert_fields(self, data, original, [0, 2, 4, 6])


if __name__ == "__main__":
    unittest.main()
