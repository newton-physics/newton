# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Learn one shared equilibrium spline for a runner's stance initial states."""

from __future__ import annotations

import json

import numpy as np

from ..trajectory import Spline
from .gradient_fit import _constraints, _project_proposal
from .shared_controller import nominal_six_channel


class RunnerController:
    """Map initial leg states to one runner-specific equilibrium trajectory.

    The model has one shared 12-by-6 spline template. In ``initial`` mode, a
    low-rank adjustment depends only on the initial state and velocity. The
    input order is ``[hip_z, hip_angle, knee, ankle, hip_x_velocity,
    hip_z_velocity, hip_angle_velocity, knee_velocity, ankle_velocity]``;
    absolute hip x is omitted so the trajectory is expressed in a local ground
    frame centered at the training mean initial hip x. Hip and ankle horizontal
    equilibrium channels translate together with each initial hip x. Duration
    and geometry are fixed to training-set summaries, and are not inference
    inputs.
    """

    def __init__(self, training_references, profile, scale, *, mode="initial", rank=4, seed=17, duration=None):
        if mode not in ("fixed", "initial"):
            raise ValueError("Mode must be 'fixed' or 'initial'")
        if rank < 1:
            raise ValueError("Rank must be positive")
        if not training_references:
            raise ValueError("At least one training reference is required")

        self.profile = profile
        self.mode = mode
        self.rank = int(rank)
        self.scale = np.asarray(scale, dtype=np.float64)
        if self.scale.shape != (6,) or not np.isfinite(self.scale).all() or np.any(self.scale <= 0):
            raise ValueError("Scale must contain six finite positive channel scales")

        durations = np.asarray([float(reference["time_s"][-1]) for reference in training_references])
        if not np.isfinite(durations).all() or np.any(durations <= 0):
            raise ValueError("Training references must have positive finite durations")
        self.duration = float(np.median(durations) if duration is None else duration)
        if not np.isfinite(self.duration) or self.duration <= 0:
            raise ValueError("Duration must be positive and finite")
        geometries = np.asarray([reference["lengths_m"] for reference in training_references], dtype=np.float64)
        if geometries.ndim != 2 or geometries.shape[1] != 2 or not np.isfinite(geometries).all():
            raise ValueError("Training references must provide two finite segment lengths")
        self.mean_geometry = geometries.mean(axis=0)
        self.origin_x = float(np.mean([reference["state"][0][0] for reference in training_references]))

        nominals = np.asarray([nominal_six_channel(reference, profile) for reference in training_references])
        centered_nominals = nominals.copy()
        for index, reference in enumerate(training_references):
            offset = float(reference["state"][0][0]) - self.origin_x
            centered_nominals[index, :, 0] -= offset
            centered_nominals[index, :, 4] -= offset
        normalized_seed = centered_nominals.mean(axis=0).reshape(72) / np.tile(self.scale, 12)
        self._constraint_matrix, self._constraint_limits = _constraints(profile, self.duration, np.tile(self.scale, 12))
        self.translation_range = (
            np.asarray(
                [
                    min(float(r["state"][0, 0]) for r in training_references),
                    max(float(r["state"][0, 0]) for r in training_references),
                ]
            )
            - self.origin_x
        )
        translations = np.zeros((2, 72))
        translations[:, 0::6] = self.translation_range[:, None] / self.scale[0]
        translations[:, 4::6] = self.translation_range[:, None] / self.scale[4]
        self._fixed_limits = self._constraint_limits - np.max(self._constraint_matrix @ translations.T, axis=1)
        zero = np.zeros(72, dtype=np.float64)
        template = _project_proposal(
            zero,
            normalized_seed,
            self._constraint_matrix,
            self._fixed_limits if mode == "fixed" else self._constraint_limits,
        )[0]

        raw_features = self._features_from_references(training_references)
        self.feature_mean = raw_features.mean(axis=0)
        self.feature_scale = raw_features.std(axis=0)
        self.feature_scale = np.where(self.feature_scale > 1e-10, self.feature_scale, 1.0)

        rng = np.random.default_rng(seed)
        self.parameters = {"template": template.copy()}
        if mode == "initial":
            self.parameters["encoder_weight"] = rng.normal(size=(9, self.rank)) / np.sqrt(9)
            self.parameters["decoder_weight"] = np.zeros((self.rank, 72), dtype=np.float64)
        self.first = {name: np.zeros_like(value) for name, value in self.parameters.items()}
        self.second = {name: np.zeros_like(value) for name, value in self.parameters.items()}
        self.updates = 0

    @staticmethod
    def _feature_rows(states, velocities):
        """Build centered, reference-independent initial-condition features."""
        states = np.asarray(states, dtype=np.float64)
        velocities = np.asarray(velocities, dtype=np.float64)
        if states.ndim != 2 or states.shape[1] != 5 or velocities.shape != states.shape:
            raise ValueError("Initial states and velocities must both have shape [batch, 5]")
        if not np.isfinite(states).all() or not np.isfinite(velocities).all():
            raise ValueError("Initial states and velocities must be finite")
        return np.column_stack((states[:, 1:], velocities))

    def _features_from_references(self, references):
        states = np.asarray([reference["state"][0] for reference in references], dtype=np.float64)
        velocities = np.asarray([reference["velocity"][0] for reference in references], dtype=np.float64)
        return self._feature_rows(states, velocities)

    def predict_initial(self, initial_states, initial_velocities):
        """Predict bounded shared-controller splines from initial conditions only."""
        features = self._feature_rows(initial_states, initial_velocities)
        if len(features) == 0:
            raise ValueError("At least one initial condition is required")
        normalized_features = (features - self.feature_mean) / self.feature_scale
        p = self.parameters
        if self.mode == "initial":
            encoded = np.tanh(normalized_features @ p["encoder_weight"])
            correction = encoded @ p["decoder_weight"]
        else:
            encoded = np.empty((len(features), 0), dtype=np.float64)
            correction = np.zeros((len(features), 72), dtype=np.float64)

        coefficients = np.empty((len(features), 12, 6), dtype=np.float64)
        faces = []
        projection_records = []
        if self.mode == "fixed":
            projected_template, shared_record = _project_proposal(
                np.zeros(72), p["template"], self._constraint_matrix, self._fixed_limits
            )
            shared_active = self._constraint_matrix[
                (self._fixed_limits - self._constraint_matrix @ projected_template) <= 1e-8
            ]
        for index, displacement in enumerate(correction):
            translated = p["template"].copy()
            offset = float(np.asarray(initial_states)[index, 0]) - self.origin_x
            translated[0::6] += offset / self.scale[0]
            translated[4::6] += offset / self.scale[4]
            proposed = translated + displacement
            if self.mode == "fixed":
                projected = projected_template + translated - p["template"]
                record = shared_record
            else:
                update, record = _project_proposal(
                    translated, proposed - translated, self._constraint_matrix, self._constraint_limits
                )
                projected = translated + update
            coefficient = projected.reshape(12, 6) * self.scale
            if not Spline(self.duration, coefficient).bounds(
                self.profile["equilibrium_lower"],
                self.profile["equilibrium_upper"],
                self.profile["equilibrium_rate_limit"],
                self.profile["equilibrium_acceleration_limit"],
            ):
                raise ValueError("Projected controller exceeds original spline bounds")
            active = (
                shared_active
                if self.mode == "fixed"
                else self._constraint_matrix[(self._constraint_limits - self._constraint_matrix @ projected) <= 1e-8]
            )
            if len(active):
                _, singular, vectors = np.linalg.svd(active, full_matrices=False)
                rank = int(np.count_nonzero(singular > max(float(singular[0]), 1.0) * 1e-10))
                face = vectors[:rank]
            else:
                face = np.empty((0, 72))
            coefficients[index] = coefficient
            faces.append(face)
            projection_records.append({**record, "active_rank": len(face)})
        cache = {"features": normalized_features, "encoded": encoded, "faces": faces, "projection": projection_records}
        return coefficients, cache

    def trajectory(self, initial_state, initial_velocity):
        """Construct a :class:`Spline` over the controller's fixed stance duration."""
        coefficients, _ = self.predict_initial(
            np.asarray(initial_state, dtype=np.float64)[None, :],
            np.asarray(initial_velocity, dtype=np.float64)[None, :],
        )
        return Spline(self.duration, coefficients[0])

    def backward(self, coefficient_gradients, cache):
        """Return the batch-mean VJP through projection and the shared model."""
        gradients = np.asarray(coefficient_gradients, dtype=np.float64)
        if gradients.ndim != 3 or gradients.shape[1:] != (12, 6) or len(gradients) != len(cache["faces"]):
            raise ValueError("Coefficient gradients must have shape [batch, 12, 6]")
        if len(gradients) == 0:
            raise ValueError("At least one coefficient gradient is required")
        gradient = (gradients * self.scale).reshape(-1, 72) / len(gradients)
        if self.mode != "fixed":
            for index, face in enumerate(cache["faces"]):
                gradient[index] -= face.T @ (face @ gradient[index])
        result = {"template": gradient.sum(axis=0)}
        if self.mode == "initial":
            encoded = cache["encoded"]
            result["decoder_weight"] = encoded.T @ gradient
            hidden = (gradient @ self.parameters["decoder_weight"].T) * (1.0 - encoded * encoded)
            result["encoder_weight"] = cache["features"].T @ hidden
        return result

    def apply_adam(self, gradients, *, learning_rate=0.003, clip_norm=10.0):
        """Apply a clipped Adam update to the shared runner controller."""
        if set(gradients) != set(self.parameters):
            raise ValueError("Gradient parameter names do not match controller parameters")
        norm = float(np.sqrt(sum(np.sum(value * value) for value in gradients.values())))
        if not np.isfinite(norm):
            raise ValueError("Nonfinite controller gradient")
        if learning_rate <= 0 or clip_norm <= 0:
            raise ValueError("Learning rate and clip norm must be positive")
        factor = min(1.0, clip_norm / max(norm, 1e-30))
        self.updates += 1
        for name, value in gradients.items():
            gradient = np.asarray(value, dtype=np.float64) * factor
            self.first[name] = 0.9 * self.first[name] + 0.1 * gradient
            self.second[name] = 0.999 * self.second[name] + 0.001 * gradient * gradient
            first = self.first[name] / (1.0 - 0.9**self.updates)
            second = self.second[name] / (1.0 - 0.999**self.updates)
            self.parameters[name] -= learning_rate * first / (np.sqrt(second) + 1e-8)
        if self.mode == "fixed":
            self.parameters["template"] = _project_proposal(
                np.zeros(72), self.parameters["template"], self._constraint_matrix, self._fixed_limits
            )[0]
        return {"norm": norm, "clipped_norm": norm * factor}

    def snapshot(self):
        """Copy model and optimizer state for rejecting a proposed rollout update."""
        return {
            "parameters": {name: value.copy() for name, value in self.parameters.items()},
            "first": {name: value.copy() for name, value in self.first.items()},
            "second": {name: value.copy() for name, value in self.second.items()},
            "updates": self.updates,
        }

    def restore(self, snapshot):
        """Restore model and Adam state from a snapshot."""
        for name in ("parameters", "first", "second"):
            setattr(self, name, {key: value.copy() for key, value in snapshot[name].items()})
        self.updates = int(snapshot["updates"])

    def save(self, path, metadata=None):
        """Save model, training-only statistics and resumable optimizer state."""
        arrays = {
            "mode": self.mode,
            "rank": self.rank,
            "duration": self.duration,
            "origin_x": self.origin_x,
            "translation_range": self.translation_range,
            "fixed_limits": self._fixed_limits,
            "mean_geometry": self.mean_geometry,
            "feature_mean": self.feature_mean,
            "feature_scale": self.feature_scale,
            "scale": self.scale,
            "updates": self.updates,
            "metadata_json": json.dumps(metadata or {}, sort_keys=True),
        }
        arrays.update(self.parameters)
        arrays.update({f"first_{name}": value for name, value in self.first.items()})
        arrays.update({f"second_{name}": value for name, value in self.second.items()})
        np.savez_compressed(path, **arrays)

    @classmethod
    def load(cls, path, profile):
        """Restore a saved controller without recomputing training statistics."""
        result = cls.__new__(cls)
        result.profile = profile
        result._constraint_matrix = None
        result._constraint_limits = None
        with np.load(path, allow_pickle=False) as archive:
            result.mode = str(archive["mode"])
            result.rank = int(archive["rank"])
            result.duration = float(archive["duration"])
            for name in ("mean_geometry", "feature_mean", "feature_scale", "scale"):
                setattr(result, name, archive[name].copy())
            result.origin_x = float(archive["origin_x"])
            result.translation_range = archive["translation_range"].copy()
            result._fixed_limits = archive["fixed_limits"].copy()
            names = ["template"] + (["encoder_weight", "decoder_weight"] if result.mode == "initial" else [])
            result.parameters = {name: archive[name].copy() for name in names}
            result.first = {name: archive[f"first_{name}"].copy() for name in names}
            result.second = {name: archive[f"second_{name}"].copy() for name in names}
            result.updates = int(archive["updates"])
            metadata = json.loads(str(archive["metadata_json"]))
        if result.mode not in ("fixed", "initial"):
            raise ValueError("Saved controller has an invalid mode")
        result._constraint_matrix, result._constraint_limits = _constraints(
            profile, result.duration, np.tile(result.scale, 12)
        )
        return result, metadata
