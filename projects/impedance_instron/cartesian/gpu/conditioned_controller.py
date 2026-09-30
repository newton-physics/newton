# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Predict bounded stance-specific spline corrections with a shared NumPy network."""

from __future__ import annotations

import json

import numpy as np

from ..trajectory import Spline
from .gradient_fit import _constraints, _project_proposal
from .shared_controller import nominal_six_channel


def features(reference, nominal, scale):
    """Encode the desired kinematics and native GRF without prescribing simulation forces."""
    duration = float(reference["time_s"][-1])
    query = np.linspace(0.0, duration, 12)
    desired_force = np.column_stack(
        [np.interp(query, reference["grf_time_s"], reference["grf_target_n"][:, c]) for c in range(2)]
    )
    return np.r_[
        (nominal / scale).ravel(),
        reference["state"][0],
        reference["velocity"][0],
        duration,
        reference["lengths_m"],
        (desired_force / 100.0).ravel(),
    ]


class Controller:
    """Share a reference encoder and coefficient decoder across all training stances."""

    def __init__(self, training_references, profile, scale, *, seed=17, components=16, hidden=32):
        self.profile = profile
        self.scale = np.asarray(scale, dtype=np.float64)
        self._constraint_cache = {}
        nominals = [nominal_six_channel(reference, profile) for reference in training_references]
        raw = np.asarray(
            [
                features(reference, nominal, self.scale)
                for reference, nominal in zip(training_references, nominals, strict=True)
            ]
        )
        self.feature_mean = raw.mean(axis=0)
        self.feature_scale = raw.std(axis=0)
        self.feature_scale = np.where(self.feature_scale > 1e-10, self.feature_scale, 1.0)
        standardized = (raw - self.feature_mean) / self.feature_scale
        _, singular, vectors = np.linalg.svd(standardized, full_matrices=False)
        count = min(components, len(training_references) - 1, len(singular))
        self.components = vectors[:count].copy()
        self.component_scale = np.maximum(singular[:count] / np.sqrt(len(training_references) - 1), 1e-8)
        rng = np.random.default_rng(seed)
        self.parameters = {
            "encoder_weight": rng.normal(size=(count, hidden)) / np.sqrt(count),
            "encoder_bias": np.zeros(hidden),
            "decoder_weight": np.zeros((hidden, 72)),
            "decoder_bias": np.zeros(72),
        }
        self.first = {name: np.zeros_like(value) for name, value in self.parameters.items()}
        self.second = {name: np.zeros_like(value) for name, value in self.parameters.items()}
        self.updates = 0

    def _bounds(self, duration):
        if duration not in self._constraint_cache:
            self._constraint_cache[duration] = _constraints(self.profile, duration, np.tile(self.scale, 12))
        return self._constraint_cache[duration]

    def predict(self, references):
        """Return projected coefficients and the retained network/projection VJP data."""
        nominals = np.asarray([nominal_six_channel(reference, self.profile) for reference in references])
        raw_features = np.asarray(
            [features(reference, nominal, self.scale) for reference, nominal in zip(references, nominals, strict=True)]
        )
        encoded = ((raw_features - self.feature_mean) / self.feature_scale) @ self.components.T / self.component_scale
        p = self.parameters
        activation = np.tanh(encoded @ p["encoder_weight"] + p["encoder_bias"])
        correction = activation @ p["decoder_weight"] + p["decoder_bias"]
        coefficients = np.empty_like(nominals)
        faces = []
        projection_records = []
        normalized = (nominals / self.scale).reshape(len(references), 72)
        for i, reference in enumerate(references):
            duration = float(reference["time_s"][-1])
            matrix, limits = self._bounds(duration)
            displacement, record = _project_proposal(normalized[i], correction[i], matrix, limits)
            projected = normalized[i] + displacement
            coefficients[i] = projected.reshape(12, 6) * self.scale
            bounds = [
                self.profile[name]
                for name in (
                    "equilibrium_lower",
                    "equilibrium_upper",
                    "equilibrium_rate_limit",
                    "equilibrium_acceleration_limit",
                )
            ]
            if not Spline(duration, coefficients[i]).bounds(*bounds):
                raise ValueError("Projected controller exceeds original spline bounds")
            active = matrix[(limits - matrix @ projected) <= 1e-8]
            if len(active):
                _, singular, vectors = np.linalg.svd(active, full_matrices=False)
                rank = int(np.count_nonzero(singular > max(float(singular[0]), 1.0) * 1e-10))
                face = vectors[:rank]
            else:
                face = np.empty((0, 72))
            faces.append(face)
            projection_records.append(
                {**record, "active_rank": len(face), "correction_norm": float(np.linalg.norm(correction[i]))}
            )
        return coefficients, {
            "encoded": encoded,
            "activation": activation,
            "faces": faces,
            "projection": projection_records,
        }

    def backward(self, coefficient_gradients, cache):
        """Differentiate the active projection face and shared network for a batch mean loss."""
        gradient = (np.asarray(coefficient_gradients) * self.scale).reshape(-1, 72) / len(coefficient_gradients)
        for i, face in enumerate(cache["faces"]):
            gradient[i] -= face.T @ (face @ gradient[i])
        activation, encoded = cache["activation"], cache["encoded"]
        hidden = (gradient @ self.parameters["decoder_weight"].T) * (1.0 - activation * activation)
        return {
            "encoder_weight": encoded.T @ hidden,
            "encoder_bias": hidden.sum(axis=0),
            "decoder_weight": activation.T @ gradient,
            "decoder_bias": gradient.sum(axis=0),
        }

    def apply_adam(self, gradients, *, learning_rate=0.003, clip_norm=10.0):
        """Update shared parameters with finite, clipped batch gradients."""
        norm = float(np.sqrt(sum(np.sum(value * value) for value in gradients.values())))
        if not np.isfinite(norm):
            raise ValueError("Nonfinite controller gradient")
        factor = min(1.0, clip_norm / max(norm, 1e-30))
        self.updates += 1
        for name, value in gradients.items():
            gradient = value * factor
            self.first[name] = 0.9 * self.first[name] + 0.1 * gradient
            self.second[name] = 0.999 * self.second[name] + 0.001 * gradient * gradient
            first = self.first[name] / (1.0 - 0.9**self.updates)
            second = self.second[name] / (1.0 - 0.999**self.updates)
            self.parameters[name] -= learning_rate * first / (np.sqrt(second) + 1e-8)
        return {"norm": norm, "clipped_norm": norm * factor}

    def snapshot(self):
        """Copy controller and optimizer state for rollback after a failed proposed rollout."""
        return {
            "parameters": {name: value.copy() for name, value in self.parameters.items()},
            "first": {name: value.copy() for name, value in self.first.items()},
            "second": {name: value.copy() for name, value in self.second.items()},
            "updates": self.updates,
        }

    def restore(self, snapshot):
        """Restore all model and Adam state after rejecting an update."""
        for name in ("parameters", "first", "second"):
            setattr(self, name, {key: value.copy() for key, value in snapshot[name].items()})
        self.updates = snapshot["updates"]

    def save(self, path, metadata):
        """Save train-only feature statistics, network parameters and resumable Adam state."""
        arrays = {
            "feature_mean": self.feature_mean,
            "feature_scale": self.feature_scale,
            "components": self.components,
            "component_scale": self.component_scale,
            "scale": self.scale,
            "updates": self.updates,
            "metadata_json": json.dumps(metadata, sort_keys=True),
        }
        arrays.update(self.parameters)
        arrays.update({f"first_{name}": value for name, value in self.first.items()})
        arrays.update({f"second_{name}": value for name, value in self.second.items()})
        np.savez_compressed(path, **arrays)

    @classmethod
    def load(cls, path, profile):
        """Restore a checkpoint without fitting feature statistics to new references."""
        result = cls.__new__(cls)
        result.profile = profile
        result._constraint_cache = {}
        with np.load(path, allow_pickle=False) as archive:
            for name in ("feature_mean", "feature_scale", "components", "component_scale", "scale"):
                setattr(result, name, archive[name].copy())
            names = ("encoder_weight", "encoder_bias", "decoder_weight", "decoder_bias")
            result.parameters = {name: archive[name].copy() for name in names}
            result.first = {name: archive[f"first_{name}"].copy() for name in names}
            result.second = {name: archive[f"second_{name}"].copy() for name in names}
            result.updates = int(archive["updates"])
            metadata = json.loads(str(archive["metadata_json"]))
        return result, metadata
