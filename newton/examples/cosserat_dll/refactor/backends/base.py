# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Base class for Cosserat rod solver backends."""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..model import CosseratRodModel


class BackendBase(ABC):
    """Abstract base class for solver backends.

    All backends implement the same simulation algorithm:
    1. Predict positions and rotations
    2. Solve constraints (direct method with banded matrix)
    3. Integrate positions and rotations

    Subclasses must implement the step() method and may override
    notification methods for state changes.

    Attributes:
        model: Reference to the rod model.
        name: Human-readable name of the backend.
    """

    def __init__(self, model: "CosseratRodModel"):
        """Initialize backend with model reference.

        Args:
            model: The rod model to operate on.
        """
        self.model = model

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name of this backend."""
        pass

    @abstractmethod
    def step(self, dt: float):
        """Advance simulation by one timestep.

        This method must implement the full simulation step:
        1. Predict positions (semi-implicit Euler)
        2. Predict rotations (quaternion integration)
        3. Prepare constraints (compute compliance)
        4. Update constraint values
        5. Compute Jacobians
        6. Assemble JMJT matrix
        7. Solve and apply corrections
        8. Integrate positions
        9. Integrate rotations
        10. Clear forces

        Args:
            dt: Time step size in seconds.
        """
        pass

    # =========================================================================
    # State change notifications
    # =========================================================================
    # These methods are called when model state changes. Backends that cache
    # state (like Warp arrays) should override these to sync their caches.

    def on_gravity_changed(self):
        """Called when gravity vector changes."""
        pass

    def on_material_changed(self):
        """Called when material properties change."""
        pass

    def on_rest_shape_changed(self):
        """Called when rest Darboux vectors change."""
        pass

    def on_stiffness_changed(self):
        """Called when stiffness coefficients change."""
        pass

    def on_reset(self):
        """Called when simulation is reset."""
        pass
