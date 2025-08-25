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

import os
import tempfile
import unittest

import numpy as np
import warp as wp

import newton
import newton.utils
from newton.tests.unittest_utils import add_function_test, get_test_devices

wp.config.quiet = True


class TestRecorder(unittest.TestCase):
    pass


def test_body_transform_recorder(test: TestRecorder, device):
    recorder = newton.utils.BasicRecorder()

    transform1 = wp.array([wp.transform([1, 2, 3], [0, 0, 0, 1])], dtype=wp.transform, device=device)
    transform2 = wp.array([wp.transform([4, 5, 6], [0, 0, 0, 1])], dtype=wp.transform, device=device)

    recorder.record(transform1)
    recorder.record(transform2)

    test.assertEqual(len(recorder.transforms_history), 2)

    np.testing.assert_allclose(recorder.transforms_history[0].numpy(), transform1.numpy())
    np.testing.assert_allclose(recorder.transforms_history[1].numpy(), transform2.numpy())

    with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as tmp:
        file_path = tmp.name

    try:
        recorder.save_to_file(file_path)

        new_recorder = newton.utils.BasicRecorder()
        new_recorder.load_from_file(file_path, device=device)

        test.assertEqual(len(new_recorder.transforms_history), 2)
        np.testing.assert_allclose(new_recorder.transforms_history[0].numpy(), transform1.numpy())
        np.testing.assert_allclose(new_recorder.transforms_history[1].numpy(), transform2.numpy())

    finally:
        if os.path.exists(file_path):
            os.remove(file_path)


def _compare_serialized_data(test, data1, data2):
    test.assertEqual(type(data1), type(data2))
    if isinstance(data1, dict):
        test.assertEqual(set(data1.keys()), set(data2.keys()))
        for key in data1:
            _compare_serialized_data(test, data1[key], data2[key])
    elif isinstance(data1, list) or isinstance(data1, tuple):
        test.assertEqual(len(data1), len(data2))
        for item1, item2 in zip(data1, data2):
            _compare_serialized_data(test, item1, item2)
    elif isinstance(data1, set):
        test.assertEqual(data1, data2)
    elif isinstance(data1, wp.array):
        np.testing.assert_allclose(data1.numpy(), data2.numpy(), atol=1e-6)
    elif isinstance(data1, np.ndarray):
        test.assertEqual(data1.shape, data2.shape)
        test.assertEqual(data1.dtype, data2.dtype)
        for idx in np.ndindex(data1.shape):
            test.assertAlmostEqual(data1[idx], data2[idx], delta=1e-6)
    elif isinstance(data1, float):
        test.assertAlmostEqual(data1, data2)
    elif isinstance(data1, (int, bool, str, type(None), bytes, bytearray, complex)):
        test.assertEqual(data1, data2)
    else:
        test.fail(f"Unhandled type for comparison: {type(data1)}")


def _test_model_and_state_recorder_with_format(test: TestRecorder, device, file_extension: str):
    """Helper function to test model and state recorder with a specific file format."""
    builder = newton.ModelBuilder()
    body = builder.add_body()
    builder.add_shape_capsule(body)
    builder.add_joint_free(body)
    model = builder.finalize(device=device)

    states = []
    for i in range(3):
        state = model.state()
        state.body_q.fill_(wp.transform([1.0 + i, 2.0 + i, 3.0 + i], wp.quat_identity()))
        state.body_qd.fill_(wp.spatial_vector([0.1 * i, 0.2 * i, 0.3 * i, 0.4 * i, 0.5 * i, 0.6 * i]))
        states.append(state)

    recorder = newton.utils.ModelAndStateRecorder()
    recorder.record_model(model)
    for state in states:
        recorder.record(state)

    with tempfile.NamedTemporaryFile(suffix=file_extension, delete=False) as tmp:
        file_path = tmp.name

    try:
        recorder.save_to_file(file_path)

        # Verify the file was created with the expected format
        test.assertTrue(os.path.exists(file_path), f"File {file_path} was not created")

        # For binary files, verify it's actually binary data
        if file_extension == ".bin":
            with open(file_path, "rb") as f:
                data = f.read(10)  # Read first 10 bytes
                # CBOR2 binary data should not be readable as text
                test.assertIsInstance(data, bytes, "Binary file should contain bytes")

        new_recorder = newton.utils.ModelAndStateRecorder()
        new_recorder.load_from_file(file_path)

        # Test that the model was loaded correctly
        test.assertIsNotNone(new_recorder.deserialized_model)

        # Test that we can create a new model and restore it
        restored_model = newton.Model(device=device)
        new_recorder.playback_model(restored_model)

        # Basic model validation - check that key properties match
        test.assertEqual(restored_model.body_count, model.body_count)
        test.assertEqual(restored_model.joint_count, model.joint_count)
        test.assertEqual(restored_model.shape_count, model.shape_count)

        # Test state history
        test.assertEqual(len(recorder.history), len(new_recorder.history))

        # Test that we can restore states
        for i, original_state in enumerate(states):
            restored_state = restored_model.state()
            new_recorder.playback(restored_state, i)

            # Compare the restored state with the original
            if restored_state.body_q is not None and original_state.body_q is not None:
                np.testing.assert_allclose(restored_state.body_q.numpy(), original_state.body_q.numpy(), atol=1e-6)
            if restored_state.body_qd is not None and original_state.body_qd is not None:
                np.testing.assert_allclose(restored_state.body_qd.numpy(), original_state.body_qd.numpy(), atol=1e-6)

    finally:
        if os.path.exists(file_path):
            os.remove(file_path)


def test_model_and_state_recorder_json(test: TestRecorder, device):
    """Test model and state recorder with JSON format."""
    _test_model_and_state_recorder_with_format(test, device, ".json")


def test_model_and_state_recorder_binary(test: TestRecorder, device):
    """Test model and state recorder with binary CBOR2 format."""
    # Skip binary test if CBOR2 is not available
    try:
        import cbor2  # noqa: F401, PLC0415
    except ImportError:
        test.skipTest("cbor2 library not available for binary format testing")

    _test_model_and_state_recorder_with_format(test, device, ".bin")


devices = get_test_devices()

add_function_test(
    TestRecorder,
    "test_body_transform_recorder",
    test_body_transform_recorder,
    devices=devices,
)

add_function_test(
    TestRecorder,
    "test_model_and_state_recorder_json",
    test_model_and_state_recorder_json,
    devices=devices,
)

add_function_test(
    TestRecorder,
    "test_model_and_state_recorder_binary",
    test_model_and_state_recorder_binary,
    devices=devices,
)

if __name__ == "__main__":
    wp.clear_kernel_cache()
    unittest.main(verbosity=2)
