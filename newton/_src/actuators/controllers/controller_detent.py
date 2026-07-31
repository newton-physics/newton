# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
from typing import Any, ClassVar

import numpy as np
import warp as wp

from .base import Controller


@wp.func
def _interpolate_detent_effort(
    position: float,
    lookup_positions: wp.array[float],
    lookup_efforts: wp.array[float],
    lookup_size: int,
) -> float:
    if position <= lookup_positions[0]:
        return lookup_efforts[0]
    if position >= lookup_positions[lookup_size - 1]:
        return lookup_efforts[lookup_size - 1]

    for sample_index in range(lookup_size - 1):
        upper_position = lookup_positions[sample_index + 1]
        if position <= upper_position:
            lower_position = lookup_positions[sample_index]
            fraction = (position - lower_position) / (upper_position - lower_position)
            lower_effort = lookup_efforts[sample_index]
            return lower_effort + fraction * (lookup_efforts[sample_index + 1] - lower_effort)

    return lookup_efforts[lookup_size - 1]


@wp.kernel
def _detent_effort_kernel(
    positions: wp.array[float],
    velocities: wp.array[float],
    pos_indices: wp.array[wp.uint32],
    vel_indices: wp.array[wp.uint32],
    lookup_positions: wp.array[float],
    lookup_efforts: wp.array[float],
    lookup_size: int,
    damping: wp.array[float],
    efforts: wp.array[float],
):
    actuator_index = wp.tid()
    position = positions[pos_indices[actuator_index]]
    velocity = velocities[vel_indices[actuator_index]]
    efforts[actuator_index] = _interpolate_detent_effort(
        position, lookup_positions, lookup_efforts, lookup_size
    ) - damping[actuator_index] * velocity


class ControllerDetent(Controller):
    """Stateless controller for mechanical switches with multiple stable detents.

    The controller evaluates a signed piecewise-linear effort curve from the
    current joint position and adds viscous damping::

        effort = signed_lookup(position) - damping * velocity

    Define the curve either with high-level detent parameters or directly with
    ``lookup_positions`` and ``lookup_efforts``. All DOFs grouped into one
    :class:`~newton.actuators.Actuator` share the curve, while ``damping`` is a
    per-DOF parameter.
    """

    SHARED_PARAMS: ClassVar[set[str]] = {
        "detent_positions",
        "breakover_positions",
        "holding_efforts",
        "breakaway_efforts",
        "transition_width",
        "lookup_positions",
        "lookup_efforts",
    }

    @staticmethod
    def _validate_finite(values: tuple[float, ...], name: str) -> None:
        if not all(math.isfinite(value) for value in values):
            raise ValueError(f"{name} must contain only finite values")

    @classmethod
    def _resolve_lookup(cls, args: dict[str, Any]) -> tuple[tuple[float, ...], tuple[float, ...]]:
        has_lookup = "lookup_positions" in args or "lookup_efforts" in args
        high_level_names = {
            "detent_positions",
            "breakover_positions",
            "holding_efforts",
            "breakaway_efforts",
            "transition_width",
        }
        has_high_level = any(name in args for name in high_level_names)

        if has_lookup and has_high_level:
            raise ValueError("Provide either detent parameters or lookup_positions and lookup_efforts, not both")
        if has_lookup:
            if "lookup_positions" not in args or "lookup_efforts" not in args:
                raise ValueError("Both lookup_positions and lookup_efforts are required")
            positions = tuple(args["lookup_positions"])
            efforts = tuple(args["lookup_efforts"])
            if len(positions) < 2:
                raise ValueError("lookup_positions and lookup_efforts must contain at least two samples")
            if len(positions) != len(efforts):
                raise ValueError(
                    f"lookup_positions length ({len(positions)}) must match "
                    f"lookup_efforts length ({len(efforts)})"
                )
            cls._validate_finite(positions, "lookup_positions")
            cls._validate_finite(efforts, "lookup_efforts")
            if not all(positions[i] < positions[i + 1] for i in range(len(positions) - 1)):
                raise ValueError("lookup_positions must be strictly increasing")
            return positions, efforts

        required_names = {"detent_positions", "holding_efforts", "breakaway_efforts", "transition_width"}
        missing_names = sorted(required_names - args.keys())
        if missing_names:
            raise ValueError(f"Missing required detent parameter(s): {', '.join(missing_names)}")

        detents = tuple(args["detent_positions"])
        holding_efforts = tuple(args["holding_efforts"])
        breakaway_efforts = tuple(args["breakaway_efforts"])
        transition_width = args["transition_width"]
        if len(detents) < 2:
            raise ValueError("detent_positions must contain at least two values")
        cls._validate_finite(detents, "detent_positions")
        if not all(detents[i] < detents[i + 1] for i in range(len(detents) - 1)):
            raise ValueError("detent_positions must be strictly increasing")
        if len(holding_efforts) != len(detents):
            raise ValueError(
                f"holding_efforts length ({len(holding_efforts)}) must match "
                f"detent_positions length ({len(detents)})"
            )
        if len(breakaway_efforts) != len(detents) - 1:
            raise ValueError(
                f"breakaway_efforts length ({len(breakaway_efforts)}) must be one less than "
                f"detent_positions length ({len(detents)})"
            )
        cls._validate_finite(holding_efforts, "holding_efforts")
        cls._validate_finite(breakaway_efforts, "breakaway_efforts")
        if any(effort < 0.0 for effort in holding_efforts):
            raise ValueError("holding_efforts must contain non-negative values")
        if any(effort < 0.0 for effort in breakaway_efforts):
            raise ValueError("breakaway_efforts must contain non-negative values")
        if not math.isfinite(transition_width) or transition_width <= 0.0:
            raise ValueError(f"transition_width must be positive and finite, got {transition_width}")

        if "breakover_positions" in args and args["breakover_positions"] is not None:
            breakovers = tuple(args["breakover_positions"])
            if len(breakovers) != len(detents) - 1:
                raise ValueError(
                    f"breakover_positions length ({len(breakovers)}) must be one less than "
                    f"detent_positions length ({len(detents)})"
                )
            cls._validate_finite(breakovers, "breakover_positions")
            for index, breakover in enumerate(breakovers):
                if not detents[index] < breakover < detents[index + 1]:
                    raise ValueError(
                        f"breakover_positions[{index}] must lie strictly between its adjacent detents"
                    )
        else:
            breakovers = tuple(0.5 * (detents[i] + detents[i + 1]) for i in range(len(detents) - 1))

        crossings: list[tuple[float, bool, float]] = []
        for index, detent in enumerate(detents):
            crossings.append((detent, True, holding_efforts[index]))
            if index < len(breakovers):
                crossings.append((breakovers[index], False, breakaway_efforts[index]))

        for index in range(len(crossings) - 1):
            if crossings[index][0] + transition_width >= crossings[index + 1][0] - transition_width:
                raise ValueError("transition_width neighborhoods must not overlap adjacent zero crossings")

        positions: list[float] = []
        efforts: list[float] = []
        for crossing, is_stable, magnitude in crossings:
            left_effort = magnitude if is_stable else -magnitude
            positions.extend((crossing - transition_width, crossing, crossing + transition_width))
            efforts.extend((left_effort, 0.0, -left_effort))
        return tuple(positions), tuple(efforts)

    @classmethod
    def resolve_arguments(cls, args: dict[str, Any]) -> dict[str, Any]:
        """Resolve and validate detent controller parameters.

        Args:
            args: User-provided controller arguments.

        Returns:
            A signed lookup curve and per-DOF damping value.
        """
        damping = args.get("damping", 0.0)
        if not math.isfinite(damping) or damping < 0.0:
            raise ValueError(f"damping must be non-negative and finite, got {damping}")
        positions, efforts = cls._resolve_lookup(args)
        resolved = {"lookup_positions": positions, "lookup_efforts": efforts, "damping": damping}
        for name in cls.SHARED_PARAMS - {"lookup_positions", "lookup_efforts"}:
            if name in args:
                resolved[name] = args[name]
        return resolved

    def __init__(
        self,
        damping: wp.array[float],
        lookup_positions: tuple[float, ...],
        lookup_efforts: tuple[float, ...],
        detent_positions: tuple[float, ...] | None = None,
        breakover_positions: tuple[float, ...] | None = None,
        holding_efforts: tuple[float, ...] | None = None,
        breakaway_efforts: tuple[float, ...] | None = None,
        transition_width: float | None = None,
    ):
        """Initialize a detent controller.

        Args:
            damping: Viscous damping coefficients [N·s/m or N·m·s/rad]. Shape ``(N,)``.
            lookup_positions: Sorted joint positions [m or rad]. Shape ``(K,)``.
            lookup_efforts: Signed output efforts [N or N·m]. Shape ``(K,)``.
            detent_positions: Authored stable joint positions [m or rad].
            breakover_positions: Authored unstable joint positions [m or rad].
            holding_efforts: Authored restoring efforts [N or N·m].
            breakaway_efforts: Authored breakaway efforts [N or N·m].
            transition_width: Authored zero-crossing transition width [m or rad].
        """
        if len(lookup_positions) != len(lookup_efforts):
            raise ValueError("lookup_positions and lookup_efforts must have matching lengths")
        self.damping = damping
        self._lookup_positions = lookup_positions
        self._lookup_efforts = lookup_efforts
        self.lookup_size = len(lookup_positions)
        self.lookup_positions: wp.array[float] | None = None
        self.lookup_efforts: wp.array[float] | None = None

    def finalize(self, device: wp.Device, num_actuators: int) -> None:
        """Upload the shared lookup curve to the actuator device.

        Args:
            device: Warp device to use.
            num_actuators: Number of actuator DOFs sharing the curve.
        """
        if self.damping.shape != (num_actuators,):
            raise ValueError(f"damping shape {self.damping.shape} must be ({num_actuators},)")
        self.lookup_positions = wp.array(
            np.asarray(self._lookup_positions, dtype=np.float32), dtype=wp.float32, device=device
        )
        self.lookup_efforts = wp.array(
            np.asarray(self._lookup_efforts, dtype=np.float32), dtype=wp.float32, device=device
        )
        self._lookup_positions = ()
        self._lookup_efforts = ()

    def is_stateful(self) -> bool:
        return False

    def is_graphable(self) -> bool:
        return True

    def compute(
        self,
        positions: wp.array[float],
        velocities: wp.array[float],
        target_pos: wp.array[float],
        target_vel: wp.array[float],
        feedforward: wp.array[float] | None,
        pos_indices: wp.array[wp.uint32],
        vel_indices: wp.array[wp.uint32],
        target_pos_indices: wp.array[wp.uint32],
        target_vel_indices: wp.array[wp.uint32],
        forces: wp.array[float],
        state: Controller.State | None,
        dt: float,
        device: wp.Device | None = None,
    ) -> None:
        wp.launch(
            kernel=_detent_effort_kernel,
            dim=len(forces),
            inputs=[
                positions,
                velocities,
                pos_indices,
                vel_indices,
                self.lookup_positions,
                self.lookup_efforts,
                self.lookup_size,
                self.damping,
            ],
            outputs=[forces],
            device=device,
        )
