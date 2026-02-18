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
import newton.examples
from newton._src.utils.import_mjcf import parse_mjcf
from newton._src.viewer.viewer_file import (
    HAS_CBOR2,
    RingBuffer,
    depointer_as_key,
    pointer_as_key,
)
from newton.tests.unittest_utils import add_function_test, get_test_devices
from newton.viewer import ViewerFile


class TestRecorder(unittest.TestCase):
    pass


def test_ringbuffer_basic(test: TestRecorder, device):
    """Test basic RingBuffer functionality."""
    # Test with capacity 3
    rb = RingBuffer(3)

    # Test empty buffer
    test.assertEqual(len(rb), 0)
    test.assertEqual(rb.to_list(), [])

    # Test adding items within capacity
    rb.append("a")
    test.assertEqual(len(rb), 1)
    test.assertEqual(rb[0], "a")
    test.assertEqual(rb.to_list(), ["a"])

    rb.append("b")
    test.assertEqual(len(rb), 2)
    test.assertEqual(rb[0], "a")
    test.assertEqual(rb[1], "b")
    test.assertEqual(rb.to_list(), ["a", "b"])

    rb.append("c")
    test.assertEqual(len(rb), 3)
    test.assertEqual(rb[0], "a")
    test.assertEqual(rb[1], "b")
    test.assertEqual(rb[2], "c")
    test.assertEqual(rb.to_list(), ["a", "b", "c"])

    # Test overflow (should overwrite oldest)
    rb.append("d")
    test.assertEqual(len(rb), 3)  # Still capacity 3
    test.assertEqual(rb[0], "b")  # "a" was overwritten
    test.assertEqual(rb[1], "c")
    test.assertEqual(rb[2], "d")
    test.assertEqual(rb.to_list(), ["b", "c", "d"])

    rb.append("e")
    test.assertEqual(len(rb), 3)
    test.assertEqual(rb[0], "c")  # "b" was overwritten
    test.assertEqual(rb[1], "d")
    test.assertEqual(rb[2], "e")
    test.assertEqual(rb.to_list(), ["c", "d", "e"])


def test_ringbuffer_edge_cases(test: TestRecorder, device):
    """Test RingBuffer edge cases."""
    rb = RingBuffer(2)

    # Test index errors
    with test.assertRaises(IndexError):
        _ = rb[0]

    with test.assertRaises(IndexError):
        rb[0] = "test"

    # Test iteration on empty buffer
    items = list(rb)
    test.assertEqual(items, [])

    # Add items and test iteration
    rb.append("x")
    rb.append("y")
    items = list(rb)
    test.assertEqual(items, ["x", "y"])

    # Test overflow and iteration
    rb.append("z")
    items = list(rb)
    test.assertEqual(items, ["y", "z"])

    # Test clear
    rb.clear()
    test.assertEqual(len(rb), 0)
    test.assertEqual(rb.to_list(), [])

    # Test from_list
    rb.from_list(["1", "2", "3", "4"])  # More than capacity
    test.assertEqual(len(rb), 2)  # Should only keep last 2
    test.assertEqual(rb.to_list(), ["3", "4"])


def test_recorder_with_ringbuffer(test: TestRecorder, device):
    """Test ViewerFile with RingBuffer-backed history."""
    # Test with ring buffer (capacity 3)
    recorder_rb = ViewerFile("recording.json", auto_save=False, max_history_size=3)

    # Simulate recording states
    for i in range(5):
        state_data = {"frame": i, "data": f"state_{i}"}
        recorder_rb.history.append(state_data)

    # Should only keep last 3 states
    test.assertEqual(len(recorder_rb.history), 3)
    test.assertEqual(recorder_rb.history[0]["frame"], 2)  # Oldest kept
    test.assertEqual(recorder_rb.history[1]["frame"], 3)
    test.assertEqual(recorder_rb.history[2]["frame"], 4)  # Newest

    # Test playback-style access
    for i in range(len(recorder_rb.history)):
        state_data = recorder_rb.history[i]
        expected_frame = 2 + i
        test.assertEqual(state_data["frame"], expected_frame)


def test_recorder_backward_compatibility(test: TestRecorder, device):
    """Test that ViewerFile keeps backward-compatible unlimited history behavior."""
    # Test with default (unlimited history)
    recorder_list = ViewerFile("recording.json", auto_save=False)

    # Should use regular list
    test.assertIsInstance(recorder_list.history, list)

    # Simulate recording many states
    for i in range(10):
        state_data = {"frame": i, "data": f"state_{i}"}
        recorder_list.history.append(state_data)

    # Should keep all states
    test.assertEqual(len(recorder_list.history), 10)
    test.assertEqual(recorder_list.history[0]["frame"], 0)
    test.assertEqual(recorder_list.history[9]["frame"], 9)


def test_recorder_ringbuffer_save_load(test: TestRecorder, device):
    """Test ViewerFile with RingBuffer save/load functionality."""
    builder = newton.ModelBuilder()
    body = builder.add_body()
    builder.add_shape_capsule(body)
    model = builder.finalize(device=device)

    # Create recorder with ring buffer (capacity 3)
    recorder = ViewerFile("recording.json", auto_save=False, max_history_size=3)
    recorder.record_model(model)

    # Record 5 states (should only keep last 3)
    states = []
    for i in range(5):
        state = model.state()
        state.body_q.fill_(wp.transform([1.0 + i, 2.0 + i, 3.0 + i], wp.quat_identity()))
        state.body_qd.fill_(wp.spatial_vector([0.1 * i, 0.2 * i, 0.3 * i, 0.4 * i, 0.5 * i, 0.6 * i]))
        recorder.record(state)
        states.append(state)

    # Should only have last 3 states
    test.assertEqual(len(recorder.history), 3)

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
        file_path = tmp.name

    try:
        recorder.save_recording(file_path)

        # Load into a new recorder with different capacity
        new_recorder = ViewerFile(file_path, auto_save=False, max_history_size=5)
        new_recorder.load_recording()

        # Should have loaded the 3 states that were saved
        test.assertEqual(len(new_recorder.history), 3)

        # Test that we can create a new model and restore it
        restored_model = newton.Model(device=device)
        new_recorder.playback_model(restored_model)

        # Basic model validation
        test.assertEqual(restored_model.body_count, model.body_count)
        test.assertEqual(restored_model.joint_count, model.joint_count)
        test.assertEqual(restored_model.shape_count, model.shape_count)

        # Test state history comparison
        for original_state_data, loaded_state_data in zip(recorder.history, new_recorder.history, strict=False):
            _compare_serialized_data(test, original_state_data, loaded_state_data)

    finally:
        if os.path.exists(file_path):
            os.remove(file_path)


def test_viewer_file_playback(test: TestRecorder, device):
    """Test ViewerFile load_recording, load_model, and load_state for playback."""
    builder = newton.ModelBuilder()
    body = builder.add_body()
    builder.add_shape_capsule(body)
    model = builder.finalize(device=device)

    states = []
    for i in range(3):
        state = model.state()
        state.body_q.fill_(wp.transform([1.0 + i, 2.0 + i, 3.0 + i], wp.quat_identity()))
        state.body_qd.fill_(wp.spatial_vector([0.1 * i, 0.2 * i, 0.3 * i, 0.4 * i, 0.5 * i, 0.6 * i]))
        states.append(state)

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
        file_path = tmp.name

    try:
        # Record via ViewerFile
        viewer_file_record = ViewerFile(file_path, auto_save=False)
        viewer_file_record.set_model(model)
        for state in states:
            viewer_file_record.log_state(state)
        viewer_file_record.save_recording()
        viewer_file_record.close()

        # Playback via ViewerFile
        viewer_file_play = ViewerFile(file_path)
        viewer_file_play.load_recording()

        test.assertTrue(viewer_file_play.has_model())
        test.assertEqual(viewer_file_play.get_frame_count(), 3)

        restored_model = newton.Model(device=device)
        viewer_file_play.load_model(restored_model)

        test.assertEqual(restored_model.body_count, model.body_count)
        test.assertEqual(restored_model.shape_count, model.shape_count)

        for frame_id in range(3):
            restored_state = restored_model.state()
            viewer_file_play.load_state(restored_state, frame_id)
            np.testing.assert_allclose(
                restored_state.body_q.numpy(),
                states[frame_id].body_q.numpy(),
                atol=1e-6,
                err_msg=f"body_q mismatch at frame {frame_id}",
            )
            np.testing.assert_allclose(
                restored_state.body_qd.numpy(),
                states[frame_id].body_qd.numpy(),
                atol=1e-6,
                err_msg=f"body_qd mismatch at frame {frame_id}",
            )
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
        for item1, item2 in zip(data1, data2, strict=False):
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
    elif isinstance(data1, int | bool | str | type(None) | bytes | bytearray | complex):
        test.assertEqual(data1, data2)
    else:
        test.fail(f"Unhandled type for comparison: {type(data1)}")


def _test_model_and_state_recorder_with_format(test: TestRecorder, device, file_extension: str):
    """Helper function to test model and state recorder with a specific file format."""
    builder = newton.ModelBuilder()
    body = builder.add_body()
    builder.add_shape_capsule(body)
    model = builder.finalize(device=device)

    states = []
    for i in range(3):
        state = model.state()
        state.body_q.fill_(wp.transform([1.0 + i, 2.0 + i, 3.0 + i], wp.quat_identity()))
        state.body_qd.fill_(wp.spatial_vector([0.1 * i, 0.2 * i, 0.3 * i, 0.4 * i, 0.5 * i, 0.6 * i]))
        states.append(state)

    recorder = ViewerFile("recording.json", auto_save=False)
    recorder.record_model(model)
    for state in states:
        recorder.record(state)

    with tempfile.NamedTemporaryFile(suffix=file_extension, delete=False) as tmp:
        file_path = tmp.name

    try:
        recorder.save_recording(file_path)

        # Verify the file was created with the expected format
        test.assertTrue(os.path.exists(file_path), f"File {file_path} was not created")

        # For binary files, verify it's actually binary data
        if file_extension == ".bin":
            with open(file_path, "rb") as f:
                data = f.read(10)  # Read first 10 bytes
                # CBOR2 binary data should not be readable as text
                test.assertIsInstance(data, bytes, "Binary file should contain bytes")

        new_recorder = ViewerFile(file_path, auto_save=False)
        new_recorder.load_recording()

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
        for original_state_data, loaded_state_data in zip(recorder.history, new_recorder.history, strict=False):
            _compare_serialized_data(test, original_state_data, loaded_state_data)

    finally:
        if os.path.exists(file_path):
            os.remove(file_path)


def test_model_and_state_recorder_json(test: TestRecorder, device):
    """Test model and state recorder with JSON format."""
    _test_model_and_state_recorder_with_format(test, device, ".json")


def test_model_and_state_recorder_binary(test: TestRecorder, device):
    """Test model and state recorder with binary CBOR2 format."""
    # Skip binary test if CBOR2 is not available
    if not HAS_CBOR2:
        test.skipTest("cbor2 library not available for binary format testing")

    _test_model_and_state_recorder_with_format(test, device, ".bin")


devices = get_test_devices()

add_function_test(
    TestRecorder,
    "test_ringbuffer_basic",
    test_ringbuffer_basic,
    devices=devices,
)

add_function_test(
    TestRecorder,
    "test_ringbuffer_edge_cases",
    test_ringbuffer_edge_cases,
    devices=devices,
)

add_function_test(
    TestRecorder,
    "test_recorder_with_ringbuffer",
    test_recorder_with_ringbuffer,
    devices=devices,
)

add_function_test(
    TestRecorder,
    "test_recorder_backward_compatibility",
    test_recorder_backward_compatibility,
    devices=devices,
)

add_function_test(
    TestRecorder,
    "test_recorder_ringbuffer_save_load",
    test_recorder_ringbuffer_save_load,
    devices=devices,
)

add_function_test(
    TestRecorder,
    "test_viewer_file_playback",
    test_viewer_file_playback,
    devices=devices,
    check_output=False,  # ViewerFile prints save/load messages
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
    check_output=False,  # Ignore "Please install 'psutil'" UserWarning
)


def test_warp_dtype_roundtrip(test: TestRecorder, device):
    """
    Test that all warp dtypes can be serialized and deserialized correctly.

    This test ensures that recordings remain loadable across warp versions by:
    1. Testing both built-in types (vec3f, mat33f) and dynamic types (vec5, vec7)
    2. Verifying data integrity after round-trip serialization
    3. Catching type resolution issues early (like the vec_t bug)
    """
    # Test cases: (dtype, shape, description)
    # This comprehensive list covers all dtypes used in Newton Model/State/Control/Contacts
    test_cases = [
        # Built-in scalar types (all used in Newton)
        (wp.float32, (10,), "float32 scalar array"),
        (wp.float64, (5,), "float64 scalar array"),
        (wp.int32, (8,), "int32 scalar array"),
        (wp.int64, (4,), "int64 scalar array"),
        (wp.uint32, (6,), "uint32 scalar array"),
        (wp.uint64, (3,), "uint64 scalar array"),  # Used by shape_source_ptr
        # Boolean type (used by shape_is_solid, joint_enabled, jnt_actgravcomp)
        (wp.bool, (7,), "bool array"),
        # Smaller integer types (for completeness)
        (wp.int8, (5,), "int8 array"),
        (wp.int16, (5,), "int16 array"),
        (wp.uint8, (5,), "uint8 array"),
        (wp.uint16, (5,), "uint16 array"),
        # Built-in vector types
        (wp.vec2, (5,), "vec2 array"),
        (wp.vec3, (5,), "vec3 array"),
        (wp.vec4, (5,), "vec4 array"),
        (wp.vec2f, (5,), "vec2f array"),
        (wp.vec3f, (5,), "vec3f array"),
        (wp.vec4f, (5,), "vec4f array"),
        (wp.vec2d, (5,), "vec2d (float64) array"),
        (wp.vec3d, (5,), "vec3d (float64) array"),
        (wp.vec2i, (5,), "vec2i (int32) array"),
        (wp.vec3i, (5,), "vec3i (int32) array"),
        # Built-in matrix types
        (wp.mat22, (3,), "mat22 array"),
        (wp.mat33, (3,), "mat33 array"),
        (wp.mat44, (3,), "mat44 array"),
        (wp.mat22f, (3,), "mat22f array"),
        (wp.mat33f, (3,), "mat33f array"),
        (wp.mat44f, (3,), "mat44f array"),
        # Built-in special types
        (wp.quat, (4,), "quaternion array"),
        (wp.quatf, (4,), "quatf array"),
        (wp.transform, (3,), "transform array"),
        (wp.transformf, (3,), "transformf array"),
        (wp.spatial_vector, (3,), "spatial_vector array"),
        (wp.spatial_vectorf, (3,), "spatial_vectorf array"),
        # Dynamic vector types (non-standard sizes) - THIS CATCHES THE vec_t BUG
        (wp.types.vector(5, wp.float32), (4,), "dynamic vec5f array"),
        (wp.types.vector(6, wp.float32), (3,), "dynamic vec6f array"),
        (wp.types.vector(7, wp.float64), (2,), "dynamic vec7d array"),
        # Dynamic matrix types (non-standard sizes)
        (wp.types.matrix((2, 3), wp.float32), (3,), "dynamic mat2x3f array"),
        (wp.types.matrix((3, 2), wp.float32), (3,), "dynamic mat3x2f array"),
        (wp.types.matrix((5, 5), wp.float32), (2,), "dynamic mat5x5f array"),
    ]

    rng = np.random.default_rng(42)  # Reproducibility

    for dtype, shape, description in test_cases:
        with test.subTest(dtype=description):
            # Create test array with random data
            arr = wp.zeros(shape, dtype=dtype, device=device)

            # Fill with non-zero values to verify data integrity
            np_data = arr.numpy()
            np_data[:] = rng.standard_normal(np_data.shape).astype(np_data.dtype)
            arr = wp.array(np_data, dtype=dtype, device=device)

            # Serialize
            serialized = pointer_as_key({"test_array": arr}, format_type="json")

            # Deserialize
            deserialized = depointer_as_key(serialized, format_type="json")

            # Verify
            test.assertIn("test_array", deserialized, f"Array missing after deserialization: {description}")
            result_arr = deserialized["test_array"]
            test.assertIsNotNone(result_arr, f"Array is None after deserialization: {description}")
            test.assertIsInstance(result_arr, wp.array, f"Result is not wp.array: {description}")

            # Compare data
            np.testing.assert_allclose(
                result_arr.numpy(),
                arr.numpy(),
                atol=1e-6,
                err_msg=f"Data mismatch for {description}",
            )


def test_warp_dtype_roundtrip_binary(test: TestRecorder, device):
    """Test dtype round-trip with binary CBOR2 format."""
    if not HAS_CBOR2:
        test.skipTest("cbor2 library not available")

    # Test a subset of types with binary format
    test_dtypes = [
        wp.vec3f,
        wp.mat33f,
        wp.transform,
        wp.types.vector(5, wp.float32),  # Dynamic type
        wp.types.matrix((3, 4), wp.float32),  # Dynamic matrix
    ]

    rng = np.random.default_rng(42)

    for dtype in test_dtypes:
        dtype_name = getattr(dtype, "__name__", str(dtype))
        with test.subTest(dtype=dtype_name):
            arr = wp.zeros((3,), dtype=dtype, device=device)
            np_data = arr.numpy()
            np_data[:] = rng.standard_normal(np_data.shape).astype(np_data.dtype)
            arr = wp.array(np_data, dtype=dtype, device=device)

            # Test binary round-trip
            serialized = pointer_as_key({"arr": arr}, format_type="cbor2")
            deserialized = depointer_as_key(serialized, format_type="cbor2")

            test.assertIsNotNone(deserialized["arr"])
            np.testing.assert_allclose(deserialized["arr"].numpy(), arr.numpy(), atol=1e-6)


def test_warp_dtype_file_roundtrip(test: TestRecorder, device):
    """
    Test complete file save/load cycle with various dtypes.

    This simulates the real-world scenario where recordings are saved to disk
    and loaded later, potentially by different code versions.
    """

    # Create a mock "state" object with various array types
    class MockState:
        def __init__(self):
            self.vec3_array = wp.zeros((10,), dtype=wp.vec3f, device=device)
            self.mat33_array = wp.zeros((5,), dtype=wp.mat33f, device=device)
            self.transform_array = wp.zeros((3,), dtype=wp.transformf, device=device)
            # Dynamic types that caused issues
            self.vec5_array = wp.zeros((8,), dtype=wp.types.vector(5, wp.float32), device=device)
            self.vec6_array = wp.zeros((4,), dtype=wp.types.vector(6, wp.float32), device=device)

    # Fill with random data
    rng = np.random.default_rng(123)
    state = MockState()
    for attr_name in ["vec3_array", "mat33_array", "transform_array", "vec5_array", "vec6_array"]:
        arr = getattr(state, attr_name)
        np_data = arr.numpy()
        np_data[:] = rng.standard_normal(np_data.shape).astype(np_data.dtype)
        setattr(state, attr_name, wp.array(np_data, dtype=arr.dtype, device=device))

    # Test with both JSON and binary formats
    for suffix, format_name in [(".json", "JSON"), (".bin", "Binary")]:
        if suffix == ".bin" and not HAS_CBOR2:
            continue  # Skip binary if cbor2 not available

        with test.subTest(format=format_name):
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                file_path = tmp.name

            try:
                # Record
                recorder = ViewerFile("recording.json", auto_save=False)
                recorder.record(state)

                # Save
                recorder.save_recording(file_path)

                # Load into new recorder
                new_recorder = ViewerFile(file_path, auto_save=False)
                new_recorder.load_recording()

                # Verify
                test.assertEqual(len(new_recorder.history), 1)
                loaded_state = new_recorder.history[0]

                for attr_name in ["vec3_array", "mat33_array", "transform_array", "vec5_array", "vec6_array"]:
                    test.assertIn(attr_name, loaded_state, f"Missing {attr_name} in {format_name}")
                    loaded_arr = loaded_state[attr_name]
                    original_arr = getattr(state, attr_name)
                    test.assertIsNotNone(loaded_arr, f"{attr_name} is None in {format_name}")
                    np.testing.assert_allclose(
                        loaded_arr.numpy(),
                        original_arr.numpy(),
                        atol=1e-6,
                        err_msg=f"Data mismatch for {attr_name} in {format_name}",
                    )

            finally:
                if os.path.exists(file_path):
                    os.remove(file_path)


add_function_test(
    TestRecorder,
    "test_warp_dtype_roundtrip",
    test_warp_dtype_roundtrip,
    devices=devices,
)

add_function_test(
    TestRecorder,
    "test_warp_dtype_roundtrip_binary",
    test_warp_dtype_roundtrip_binary,
    devices=devices,
    check_output=False,
)

add_function_test(
    TestRecorder,
    "test_warp_dtype_file_roundtrip",
    test_warp_dtype_file_roundtrip,
    devices=devices,
    check_output=False,
)


def test_real_model_recording_roundtrip(test: TestRecorder, device):
    """
    Test recording and replay with a real Newton Model.

    This is the most comprehensive test - it uses an actual Model with:
    - Bodies, shapes, joints (standard Newton dtypes)
    - MuJoCo custom attributes (including dynamic vec5 types that caused the vec_t bug)
    - State objects with all standard arrays

    If warp changes dtype serialization in any way, this test will catch it.
    """
    # Build a real model with MuJoCo solver attributes
    mjcf_filename = newton.examples.get_asset("nv_humanoid.xml")

    builder = newton.ModelBuilder()
    newton.solvers.SolverMuJoCo.register_custom_attributes(builder)
    parse_mjcf(
        builder,
        mjcf_filename,
        ignore_names=["floor", "ground"],
        up_axis="Z",
    )

    model = builder.finalize(device=device)
    state = model.state()

    # Record the model and state
    recorder = ViewerFile("recording.json", auto_save=False)
    recorder.record_model(model)
    recorder.record(state)

    # Test with both formats
    for suffix, format_name in [(".json", "JSON"), (".bin", "Binary")]:
        if suffix == ".bin" and not HAS_CBOR2:
            continue

        with test.subTest(format=format_name):
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                file_path = tmp.name

            try:
                # Save
                recorder.save_recording(file_path)

                # Load
                new_recorder = ViewerFile(file_path, auto_save=False)
                new_recorder.load_recording()

                # Verify model loaded
                test.assertIsNotNone(new_recorder.deserialized_model, f"Model not loaded in {format_name}")

                # Verify state loaded
                test.assertEqual(len(new_recorder.history), 1, f"State count mismatch in {format_name}")

                # Restore and verify model
                restored_model = newton.Model(device=device)
                new_recorder.playback_model(restored_model)

                test.assertEqual(restored_model.body_count, model.body_count)
                test.assertEqual(restored_model.joint_count, model.joint_count)
                test.assertEqual(restored_model.shape_count, model.shape_count)

                # Verify MuJoCo attributes loaded (these use dynamic vec5 types)
                if hasattr(model, "mujoco") and hasattr(restored_model, "mujoco"):
                    for attr_name in ["geom_solimp", "solimplimit", "solimpfriction"]:
                        if hasattr(model.mujoco, attr_name):
                            original = getattr(model.mujoco, attr_name)
                            restored = getattr(restored_model.mujoco, attr_name, None)
                            test.assertIsNotNone(
                                restored, f"MuJoCo attribute {attr_name} not restored in {format_name}"
                            )
                            if original is not None and restored is not None:
                                np.testing.assert_allclose(
                                    restored.numpy(),
                                    original.numpy(),
                                    atol=1e-6,
                                    err_msg=f"MuJoCo {attr_name} data mismatch in {format_name}",
                                )

            finally:
                if os.path.exists(file_path):
                    os.remove(file_path)


add_function_test(
    TestRecorder,
    "test_real_model_recording_roundtrip",
    test_real_model_recording_roundtrip,
    devices=devices,
    check_output=False,
)


if __name__ == "__main__":
    unittest.main(verbosity=2)
