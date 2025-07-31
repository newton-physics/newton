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

from __future__ import annotations

import pickle

import numpy as np
import warp as wp

from newton.sim.model import Model
from newton.sim.state import State
from newton.sim.types import ShapeGeometry, ShapeMaterials


class BasicRecorder:
    """A class to record and playback simulation body transforms."""

    def __init__(self):
        """
        Initializes the Recorder.
        """
        self.transforms_history: list[wp.array] = []
        self.point_clouds_history: list[list[wp.array]] = []

    def record(self, body_transforms: wp.array, point_clouds: list[wp.array] | None = None):
        """
        Records a snapshot of body transforms.

        Args:
            body_transforms (wp.array): A warp array representing the body transforms.
                This is typically retrieved from `state.body_q`.
            point_clouds (list[wp.array] | None): An optional list of warp arrays representing point clouds.
        """
        self.transforms_history.append(wp.clone(body_transforms))
        if point_clouds:
            self.point_clouds_history.append([wp.clone(pc) for pc in point_clouds if pc is not None and pc.size > 0])
        else:
            self.point_clouds_history.append([])

    def playback(self, frame_id: int) -> tuple[wp.array | None, list[wp.array] | None]:
        """
        Plays back a recorded frame by returning the stored body transforms and point cloud.

        Args:
            frame_id (int): The integer index of the frame to be played back.

        Returns:
            A tuple containing the body transforms and point clouds for the given
            frame, or (None, None) if the frame_id is out of bounds.
        """
        if not (0 <= frame_id < len(self.transforms_history)):
            print(f"Warning: frame_id {frame_id} is out of bounds. Playback skipped.")
            return None, None

        transforms = self.transforms_history[frame_id]
        point_clouds = self.point_clouds_history[frame_id] if frame_id < len(self.point_clouds_history) else None
        return transforms, point_clouds

    def save_to_file(self, file_path: str):
        """
        Saves the recorded transforms history to a file.

        Args:
            file_path (str): The full path to the file where the transforms will be saved.
        """
        history_np = {f"frame_{i}": t.numpy() for i, t in enumerate(self.transforms_history)}
        for i, pc_list in enumerate(self.point_clouds_history):
            history_np[f"frame_{i}_points_count"] = len(pc_list)
            for j, pc in enumerate(pc_list):
                if pc is not None:
                    history_np[f"frame_{i}_points_{j}"] = pc.numpy()
        np.savez_compressed(file_path, **history_np)

    def load_from_file(self, file_path: str, device=None):
        """
        Loads recorded transforms from a file, replacing the current history.

        Args:
            file_path (str): The full path to the file from which to load the transforms.
            device: The device to load the transforms onto. If None, uses CPU.
        """
        self.transforms_history.clear()
        self.point_clouds_history.clear()
        with np.load(file_path) as data:
            try:
                transform_keys = [k for k in data.keys() if k.startswith("frame_") and "_points" not in k]
                frame_keys = sorted(transform_keys, key=lambda x: int(x.split("_")[1]))
            except (IndexError, ValueError) as e:
                raise ValueError(f"Invalid frame key format in file: {e}") from e
            for key in frame_keys:
                frame_index_str = key.split("_")[1]
                transform_np = data[key]
                transform_wp = wp.array(transform_np, dtype=wp.transform, device=device)
                self.transforms_history.append(transform_wp)

                pc_list = []
                count_key = f"frame_{frame_index_str}_points_count"
                if count_key in data:
                    count = int(data[count_key].item())
                    for j in range(count):
                        points_key = f"frame_{frame_index_str}_points_{j}"
                        if points_key in data:
                            points_np = data[points_key]
                            points_wp = wp.array(points_np, dtype=wp.vec3, device=device)
                            pc_list.append(points_wp)
                self.point_clouds_history.append(pc_list)


class ModelAndStateRecorder:
    """A class to record and playback simulation model and state using pickle serialization.

    WARNING: This class uses pickle for serialization which is UNSAFE and can execute
    arbitrary code when loading files. Only load recordings from TRUSTED sources that
    you have verified. Loading recordings from untrusted sources could lead to malicious
    code execution and compromise your system."""

    def __init__(self):
        """
        Initializes the Recorder.
        """
        self.history: list[dict] = []
        self.model_data: dict = {}

    def _get_device_from_state(self, state: State):
        """
        Retrieves the device from a simulation state object.

        This is done by finding the first `wp.array` attribute in the state
        and returning its device.

        Args:
            state (State): The simulation state.

        Returns:
            The device of the state's arrays, or None if no wp.array is found.
        """
        # device can be retrieved from any warp array attribute in the state
        for _name, value in state.__dict__.items():
            if isinstance(value, wp.array):
                return value.device
        return None

    def _serialize_object_attributes(self, obj):
        """
        Serializes the attributes of an object.

        This method handles both standard Python objects and Warp structs,
        preparing them for serialization by converting attributes to a
        serializable format.

        Args:
            obj: The object to serialize.

        Returns:
            A dictionary containing the serialized attributes.
        """
        data = {}
        attrs = wp.attr(obj) if hasattr(type(obj), "_wp_struct_meta_") else obj.__dict__
        for name, value in attrs.items():
            serialized_value = self._serialize_value(value)
            if serialized_value is not None:
                data[name] = serialized_value
        return data

    def _serialize_value(self, value):
        """
        Serializes a single value into a Pickle-compatible format.

        Handles various types including primitives, numpy arrays, and warp arrays.
        Warp arrays are converted to numpy arrays and stored in a dictionary
        with a type hint. Special handling for `ShapeMaterials` and `ShapeGeometry`.

        Args:
            value: The value to serialize.

        Returns:
            A serializable representation of the value, or None if the value
            type is not supported for serialization.
        """
        if value is None:
            return None

        if isinstance(value, (int, float, bool, str, list, dict, set, tuple)):
            return value
        elif isinstance(value, np.ndarray):
            return value
        elif isinstance(value, wp.array):
            if value.size > 0:
                return {
                    "__type__": "wp.array",
                    "data": value.numpy(),
                }
        elif hasattr(type(value), "_wp_struct_meta_"):
            type_name = type(value).__name__
            if type_name in ("ShapeMaterials", "ShapeGeometry"):
                return {
                    "__type__": type_name,
                    "data": self._serialize_object_attributes(value),
                }
        return None

    def _deserialize_and_restore_value(self, value, device):
        """
        Deserializes a value and restores it, including to a specific device.

        This is the counterpart to `_serialize_value`. It reconstructs objects,
        numpy arrays, and warp arrays from their serialized representation.

        Args:
            value: The serialized value.
            device: The device to load warp arrays onto.

        Returns:
            The deserialized value.
        """
        if isinstance(value, dict) and "__type__" in value:
            type_name = value["__type__"]
            obj_data = value["data"]

            if type_name == "wp.array":
                return wp.array(obj_data, device=device)

            instance = None
            if type_name == "ShapeMaterials":
                instance = ShapeMaterials()
            elif type_name == "ShapeGeometry":
                instance = ShapeGeometry()

            if instance:
                for name, s_value in obj_data.items():
                    # For wp.structs, we need to handle attribute setting carefully.
                    if hasattr(type(instance), "_wp_struct_meta_"):
                        restored_value = self._deserialize_and_restore_value(s_value, device)
                        if restored_value is not None:
                            setattr(instance, name, restored_value)
                    else:
                        setattr(instance, name, self._deserialize_and_restore_value(s_value, device))
                return instance
        elif isinstance(value, np.ndarray):
            return value
        return value

    def record(self, state: State):
        """
        Records a snapshot of the state.

        Args:
            state (State): The simulation state.
        """
        state_data = {}
        for name, value in state.__dict__.items():
            if isinstance(value, wp.array):
                state_data[name] = value.numpy()
        self.history.append(state_data)

    def record_model(self, model: Model):
        """
        Records a snapshot of the model's serializable attributes.
        It stores warp arrays as numpy arrays, and primitive types as-is.

        Args:
            model (Model): The simulation model.
        """
        self.model_data = self._serialize_object_attributes(model)

    def playback(self, state: State, frame_id: int):
        """
        Plays back a recorded frame by updating the state.

        Args:
            state (State): The simulation state to restore.
            frame_id (int): The integer index of the frame to be played back.
        """
        if not (0 <= frame_id < len(self.history)):
            print(f"Warning: frame_id {frame_id} is out of bounds. Playback skipped.")
            return

        state_data = self.history[frame_id]
        try:
            device = self._get_device_from_state(state)
        except ValueError:
            print("Warning: Unable to determine device from state. Playback skipped.")
            return

        for name, value_np in state_data.items():
            if hasattr(state, name):
                value_wp = wp.array(value_np, device=device)
                setattr(state, name, value_wp)

    def playback_model(self, model: Model):
        """
        Plays back a recorded model by updating its attributes.

        Args:
            model (Model): The simulation model to restore.
        """
        device = model.device

        for name, value in self.model_data.items():
            if hasattr(model, name):
                restored_value = self._deserialize_and_restore_value(value, device)
                if restored_value is not None:
                    setattr(model, name, restored_value)

    def save_to_file(self, file_path: str):
        """
        Saves the recorded history to a file using pickle.

        Args:
            file_path (str): The full path to the file.
        """
        with open(file_path, "wb") as f:
            data_to_save = {"model": self.model_data, "states": self.history}
            pickle.dump(data_to_save, f)

    def load_from_file(self, file_path: str):
        """
        Loads a recorded history from a file, replacing the current history.

        Args:
            file_path (str): The full path to the file.
        """
        with open(file_path, "rb") as f:
            data = pickle.load(f)
            if isinstance(data, dict) and "states" in data:
                self.history = data.get("states", [])
                self.model_data = data.get("model", {})
            else:
                # For backward compatibility with old format.
                self.history = data
                self.model_data = {}
