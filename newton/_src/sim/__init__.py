# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from .articulation import eval_fk, eval_ik, eval_jacobian, eval_mass_matrix
from .builder import ModelBuilder
from .collide import CollisionPipeline
from .contacts import Contacts
from .control import Control
from .enums import (
    BodyFlags,
    EqType,
    JointTargetMode,
    JointType,
)
from .model import Model
from .state import State

__all__ = [
    "BodyFlags",
    "CollisionPipeline",
    "Contacts",
    "Control",
    "EqType",
    "JointTargetMode",
    "JointType",
    "Model",
    "ModelBuilder",
    "State",
    "eval_fk",
    "eval_ik",
    "eval_jacobian",
    "eval_mass_matrix",
    "reset_state",
]


def reset_state(model: Model, state: State, eval_fk: bool = True) -> None:
    """Reset a state to the model's initial configuration.

    Convenience wrapper for :meth:`Model.reset_state`. See that method for
    full documentation.

    Args:
        model: The model whose initial configuration to restore.
        state: The state object to reset.
        eval_fk: Whether to re-evaluate forward kinematics.
    """
    model.reset_state(state, eval_fk=eval_fk)
