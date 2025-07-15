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

import numpy as np
import warp as wp
import pickle

from newton.utils.render import SimRendererOpenGL
from newton.sim.state import State
from newton.sim.model import Model
from newton.sim.types import ShapeMaterials, ShapeGeometry


class BodyTransformRecorder:
    """A class to record and playback simulation body transforms."""

    def __init__(self, renderer: SimRendererOpenGL):
        """
        Initializes the Recorder.

        Args:
            renderer: The simulation renderer instance.
        """
        self.renderer = renderer
        self.transforms_history: list[wp.array] = []

    def record(self, body_transforms: wp.array):
        """
        Records a snapshot of body transforms.

        Args:
            body_transforms (wp.array): A warp array representing the body transforms.
                This is typically retrieved from `state.body_q`.
        """
        self.transforms_history.append(wp.clone(body_transforms))

    def playback(self, frame_id: int):
        """
        Plays back a recorded frame by updating the renderer with the stored body transforms.

        Args:
            frame_id (int): The integer index of the frame to be played back.
        """
        if not (0 <= frame_id < len(self.transforms_history)):
            print(f"Warning: frame_id {frame_id} is out of bounds. Playback skipped.")
            return

        body_transforms = self.transforms_history[frame_id]
        self.renderer.update_body_transforms(body_transforms)

    def save_to_file(self, file_path: str):
        """
        Saves the recorded transforms history to a file.

        Args:
            file_path (str): The full path to the file where the transforms will be saved.
        """
        history_np = {f"frame_{i}": t.numpy() for i, t in enumerate(self.transforms_history)}
        np.savez_compressed(file_path, **history_np)

    def load_from_file(self, file_path: str, device=None):
        """
        Loads recorded transforms from a file, replacing the current history.

        Args:
            file_path (str): The full path to the file from which to load the transforms.
            device: The device to load the transforms onto. If None, uses CPU.
        """
        self.transforms_history.clear()
        with np.load(file_path) as data:
            frame_keys = sorted(data.keys(), key=lambda x: int(x.split("_")[1]))
            for key in frame_keys:
                transform_np = data[key]
                transform_wp = wp.array(transform_np, dtype=wp.transform, device=device)
                self.transforms_history.append(transform_wp)


class ModelAndStateRecorder:
    """A class to record and playback simulation model and state."""

    def __init__(self):
        """
        Initializes the Recorder.
        """
        self.history: list[dict] = []
        self.model_data: dict = {}

    def _get_device_from_state(self, state: State):
        # device can be retrieved from any warp array attribute in the state
        for _name, value in state.__dict__.items():
            if isinstance(value, wp.array):
                return value.device
        return None

    def _serialize_value(self, value):
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

    def _serialize_object_attributes(self, obj):
        data = {}
        attrs = wp.attr(obj) if hasattr(type(obj), "_wp_struct_meta_") else obj.__dict__
        for name, value in attrs.items():
            serialized_value = self._serialize_value(value)
            if serialized_value is not None:
                data[name] = serialized_value
        return data

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
        device = self._get_device_from_state(state)

        for name, value_np in state_data.items():
            if hasattr(state, name):
                value_wp = wp.array(value_np, device=device)
                setattr(state, name, value_wp)

    def _deserialize_and_restore_value(self, value, device):
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
