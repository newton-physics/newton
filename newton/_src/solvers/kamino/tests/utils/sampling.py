# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""
KAMINO: UNIT TESTS: Various utilities for sampling model quantities such as states, coordinates, masks etc
"""

from __future__ import annotations

import numpy as np

from ..._src.core.model import ModelKamino
from .joints import actuator_coords_from_units, actuator_dofs_from_units, get_actuators_q_quaternion_first_ids

###
# Module interface
###

__all__ = [
    "sample_actuator_coords",
    "sample_actuator_velocities",
    "sample_base_state",
    "sample_body_poses",
    "sample_world_mask",
]


def sample_world_mask(
    num_worlds: int,
    rng: np.random.Generator,
    num_samples: int = 1,
) -> np.ndarray:
    """
    Helper sampling random non-trivial boolean world masks given the number of worlds.
    Generated masks are majoritarily True, but neither all-False, nor (unless of size 1) all-True.

    Args:
        num_worlds: number of worlds.
        rng: Random number generator.
        num_samples: number of sample masks to generate.

    Returns:
        world_masks: sampled boolean masks, with shape (num_samples, num_worlds)
    """
    # Sample ids in [0, num_worlds - 1] to set to False
    # Target about 10% of inactive worlds, at least one, and at most num_worlds - 1
    num_false = min(num_worlds - 1, max(1, round(0.1 * num_worlds)))
    false_ids = rng.integers(low=0, high=num_worlds, endpoint=False, size=num_samples * num_false)
    false_ids = np.reshape(false_ids, (num_samples, num_false))

    # Build mask, setting sampled ids to zero
    # Note: non-unique false_ids do not affect the non-trivial character
    mask = np.full((num_samples, num_worlds), True)
    mask[np.arange(num_samples)[:, None], false_ids] = False
    return mask


def sample_base_state(
    num_worlds: int,
    rng: np.random.Generator,
    num_samples: int = 1,
    max_pos: float = 1.0,
    max_quat: float = 1.0,
    max_lin_vel: float = 0.5,
    max_ang_vel: float = np.radians(90.0),
    unit_quaternions: bool = True,
) -> np.ndarray:
    """
    Helper sampling random base_q, base_u given the number of worlds.

    Args:
        num_worlds: number of worlds.
        rng: Random number generator.
        num_samples: number of sample base states to generate.
        max_pos: maximal absolute sample value, for positions.
        max_quat: maximal absolute sample value, for unit quaternion coefficients.
        max_lin_vel: maximal absolute sample value, for linear velocities.
        max_ang_vel: maximal absolute sample value, for angular velocities.
        unit_quaternions: whether to normalize sampled quaternions.

    Returns:
        base_q: sampled base poses, with shape (num_samples, num_worlds, 7)
        base_u: sampled base velocities, with shape (num_samples, num_worlds, 6)
    """
    # Sample base state
    base_q = np.empty((num_samples, num_worlds, 7))
    base_u = np.empty((num_samples, num_worlds, 6))
    base_q[:, :, :3].flat = rng.uniform(-max_pos, max_pos, num_samples * num_worlds * 3)
    base_q[:, :, 3:].flat = rng.uniform(-max_quat, max_quat, num_samples * num_worlds * 4)
    base_u[:, :, :3].flat = rng.uniform(-max_lin_vel, max_lin_vel, num_samples * num_worlds * 3)
    base_u[:, :, 3:].flat = rng.uniform(-max_ang_vel, max_ang_vel, num_samples * num_worlds * 3)

    # Normalize quaternions
    if unit_quaternions:
        base_q[:, :, 3:] /= np.linalg.norm(base_q[:, :, 3:], axis=2)[:, :, None]

    return base_q, base_u


def sample_actuator_coords(
    model: ModelKamino,
    rng: np.random.Generator,
    num_samples: int = 1,
    max_pos: float = 0.1,
    max_angle: float = np.radians(20.0),
    max_quat: float = 1.0,
    unit_quaternions: bool = True,
) -> np.ndarray:
    """
    Helper sampling random actuator coords for a given model.

    Args:
        model: Kamino model.
        rng: Random number generator.
        num_samples: number of sample coord vectors to generate.
        max_pos: maximal absolute sample value, for coordinates that are positions.
        max_angle: maximal absolute sample value, for coordinates that are angles.
        max_quat: maximal absolute sample value, for coordinates that are unit quaternion coefficients.
        unit_quaternions: whether to normalize sampled quaternions.

    Returns:
        actuator_q: sampled coordinates, with shape (num_samples, num_actuator_coords)
    """
    # Generate sampling bounds
    max_coords = actuator_coords_from_units(model, max_pos, max_angle, max_quat)

    # Sample coordinates
    actuator_q = np.zeros((num_samples, model.size.sum_of_num_actuated_joint_coords), dtype=np.float32)
    for i in range(actuator_q.shape[1]):
        actuator_q[:, i] = rng.uniform(-max_coords[i], max_coords[i], size=num_samples)

    # Normalize quaternions
    if unit_quaternions:
        quat_ids = get_actuators_q_quaternion_first_ids(model)
        for i in quat_ids:
            actuator_q[:, i : i + 4] /= np.linalg.norm(actuator_q[:, i : i + 4], axis=1)[:, None]

    return actuator_q


def sample_actuator_velocities(
    model: ModelKamino,
    rng: np.random.Generator,
    num_samples: int = 1,
    max_lin_vel: float = 0.5,
    max_ang_vel: float = np.radians(90.0),
) -> np.ndarray:
    """
    Helper sampling random actuator velocities for a given model.

    Args:
        model: Kamino model.
        rng: Random number generator.
        num_samples: number of sample velocity vectors to generate.
        max_lin_vel: maximal absolute sample value, for linear dofs.
        max_ang_vel: maximal absolute sample value, for angular dofs.

    Returns:
        actuator_u: sampled velocities, with shape (num_samples, num_actuator_dofs)
    """
    # Generate sampling bounds
    max_vel = actuator_dofs_from_units(model, max_lin_vel, max_ang_vel)

    # Sample velocities
    actuator_u = np.zeros((num_samples, model.size.sum_of_num_actuated_joint_dofs), dtype=np.float32)
    for i in range(actuator_u.shape[1]):
        actuator_u[:, i] = rng.uniform(-max_vel[i], max_vel[i], size=num_samples)

    return actuator_u


def sample_body_poses(
    num_bodies: int,
    rng: np.random.Generator,
    num_samples: int = 1,
    max_pos: float = 0.1,
    max_quat: float = 1.0,
    unit_quaternions=True,
) -> np.ndarray:
    """
    Helper sampling random body poses given the number of bodies.

    Args:
        num_bodies: number of bodies in the model.
        rng: Random number generator.
        num_samples: number of sample body poses vectors to generate.
        max_pos: maximal absolute sample value, for positions.
        max_quat: maximal absolute sample value, for unit quaternion coefficients.
        unit_quaternions: whether to normalize sampled quaternions.

    Returns:
        body_q: sampled body poses, with shape (num_samples, num_bodies, 7)
    """
    # Sample body poses
    body_q = np.empty((num_samples, num_bodies, 7))
    body_q[:, :, :3].flat = rng.uniform(-max_pos, max_pos, num_samples * num_bodies * 3)
    body_q[:, :, 3:].flat = rng.uniform(-max_quat, max_quat, num_samples * num_bodies * 4)

    # Normalize quaternions
    if unit_quaternions:
        body_q[:, :, 3:] /= np.linalg.norm(body_q[:, :, 3:], axis=2)[:, :, None]

    return body_q
