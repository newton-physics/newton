# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Serializable description of a complete calibration run."""

from __future__ import annotations

import copy
import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from typing import Any

from .model import SETTLE_MODES
from .optimizer import optimizer_from_spec
from .schema import SCHEMA_VERSION, check_fields, check_schema_version
from .schema import is_real as _is_real
from .setup_spec import CableSetupSpec


@dataclass
class CalibrationOutputSpec:
    """Optional run artifacts, separate from the calibration problem."""

    directory: str | None = None
    """Artifact directory. ``None`` writes no files."""

    trace_every: int = 0
    """Record a rendered rollout every N iterations. 0 disables traces."""

    def validate(self) -> None:
        """Check the output settings.

        Raises:
            ValueError: If :attr:`directory` is empty, :attr:`trace_every` is
                negative or not an integer, or traces have no directory.
        """
        if self.directory is not None and (not isinstance(self.directory, str) or not self.directory):
            raise ValueError("output.directory must be a nonempty string or None.")
        if type(self.trace_every) is not int or self.trace_every < 0:
            raise ValueError("output.trace_every must be a nonnegative integer.")
        if self.trace_every and self.directory is None:
            raise ValueError("traces require an output directory.")


@dataclass
class CableRunSpec:
    """Physical setup plus the objective, search, simulation and output settings."""

    setup: CableSetupSpec
    """Physical cable setup."""

    optimizer: dict[str, Any]
    """Optimizer settings for :func:`~.optimizer.optimizer_from_spec`, for example ``{"kind": "cma", "seed": 42}``."""

    loss: str
    """Name of the per-frame objective. Only ``"chamfer"`` is supported."""

    init_bend_stiffness: float
    """Search start of the per-joint bend stiffness [N·m/rad], or its fixed value when it is not searched."""

    init_bend_damping: float
    """Search start of the per-joint bend damping [N·m·s/rad], or its fixed value when it is not searched."""

    settle_mode: str
    """How to reach the initial equilibrium. One of :data:`~.model.SETTLE_MODES`."""

    settle_frames: int
    """Maximum number of frames spent settling before frame 0. 0 does not settle."""

    sim_iterations: int
    """Solver iterations per substep."""

    opt_rest_config: bool = True
    """Search the rest angles."""

    opt_bend_stiffness: bool = True
    """Search the bend stiffness."""

    opt_bend_damping: bool = True
    """Search the bend damping."""

    twist_stiffness_mode: str | float = "coupled"
    """``"coupled"`` uses the bend stiffness, ``"fit"`` searches it, and a number fixes it [N·m/rad]."""

    twist_damping_mode: str | float = "coupled"
    """As :attr:`twist_stiffness_mode`, for the twist damping [N·m·s/rad]."""

    init_angles: list[list[float]] | None = None
    """Rest angles ``[[alpha, beta], ...]`` [rad], one pair per element, or ``None`` for a straight cable."""

    init_twist_stiffness: float | None = None
    """Search start of a ``"fit"`` twist stiffness [N·m/rad], or ``None`` to start at :attr:`init_bend_stiffness`."""

    init_twist_damping: float | None = None
    """Search start of a ``"fit"`` twist damping [N·m·s/rad], or ``None`` to start at :attr:`init_bend_damping`."""

    output: CalibrationOutputSpec = field(default_factory=CalibrationOutputSpec)
    """Optional run artifacts."""

    schema_version: int = SCHEMA_VERSION
    """Format version of the serialized run spec."""

    def to_dict(self) -> dict[str, Any]:
        """Return the run spec as a JSON-serializable mapping."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> CableRunSpec:
        """Build and validate a run spec from :meth:`to_dict` output.

        Raises:
            ValueError: If the schema version is not supported, ``d`` has a
                field this class does not define, ``setup`` or ``output`` is
                not a mapping, or :meth:`validate` fails.
            TypeError: If ``d`` has no value for a required field.
        """
        d = copy.deepcopy(dict(d))
        check_schema_version(d.get("schema_version"), "CableRunSpec")
        check_fields(cls, d, "CableRunSpec")
        for name in ("setup", "output"):
            if name in d and not isinstance(d[name], Mapping):
                raise ValueError(f"{name} must be a mapping, got {type(d[name]).__name__}.")
        if "setup" in d:
            d["setup"] = CableSetupSpec.from_dict(d["setup"])
        if "output" in d:
            output = dict(d["output"])
            check_fields(CalibrationOutputSpec, output, "CalibrationOutputSpec")
            d["output"] = CalibrationOutputSpec(**output)
        result = cls(**d)
        result.validate()
        return result

    def validate(self) -> None:
        """Check the problem, the optimizer settings and the output settings.

        Raises:
            ValueError: If a part of the run spec is not valid.
        """
        self.validate_problem()
        optimizer_from_spec(self.optimizer)
        if not isinstance(self.output, CalibrationOutputSpec):
            raise ValueError(f"output must be a CalibrationOutputSpec, got {type(self.output).__name__}.")
        self.output.validate()

    def validate_problem(self) -> None:
        """Check the setup, objective and search settings, but not the optimizer or output.

        Raises:
            ValueError: If a setting is not valid or no parameter is searched.
        """
        check_schema_version(self.schema_version, "CableRunSpec")
        if not isinstance(self.setup, CableSetupSpec):
            raise ValueError(f"setup must be a CableSetupSpec, got {type(self.setup).__name__}.")
        self.setup.validate()
        if self.loss != "chamfer":
            raise ValueError(f"loss {self.loss!r} is not supported; use 'chamfer'.")
        if self.settle_mode not in SETTLE_MODES:
            raise ValueError(f"settle_mode {self.settle_mode!r} is not one of {list(SETTLE_MODES)}.")
        if type(self.settle_frames) is not int or self.settle_frames < 0:
            raise ValueError(f"settle_frames must be a nonnegative integer, got {self.settle_frames!r}.")
        if type(self.sim_iterations) is not int or self.sim_iterations <= 0:
            raise ValueError(f"sim_iterations must be a positive integer, got {self.sim_iterations!r}.")
        for name in ("init_bend_stiffness", "init_bend_damping"):
            value = getattr(self, name)
            if not _is_real(value) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be a finite number >= 0, got {value!r}.")
        for name in ("init_twist_stiffness", "init_twist_damping"):
            value = getattr(self, name)
            if value is not None and (not _is_real(value) or not math.isfinite(value) or value < 0):
                raise ValueError(f"{name} must be None or a finite number >= 0, got {value!r}.")
        for name in ("twist_stiffness_mode", "twist_damping_mode"):
            mode = getattr(self, name)
            if mode not in ("coupled", "fit") and (not _is_real(mode) or not math.isfinite(mode) or mode < 0):
                raise ValueError(f"{name} must be 'coupled', 'fit' or a finite number >= 0, got {mode!r}.")
        angles = self.init_angles
        if angles is not None and (
            not isinstance(angles, (list, tuple))
            or len(angles) != self.setup.num_elements
            or not all(
                isinstance(pair, (list, tuple))
                and len(pair) == 2
                and all(_is_real(v) and math.isfinite(v) for v in pair)
                for pair in angles
            )
        ):
            raise ValueError(f"init_angles must be None or {self.setup.num_elements} pairs of finite numbers.")
        if not self.searched_groups():
            raise ValueError("no parameter is searched: enable at least one parameter group.")

    def searched_groups(self) -> list[str]:
        """Return the names of the searched parameter groups, in decision-vector order.

        The rest angles are searched only when the cable has two or more elements.
        """
        return [
            name
            for name, searched in (
                ("rest_config", self.opt_rest_config and self.setup.num_elements > 1),
                ("bend_stiffness", self.opt_bend_stiffness),
                ("twist_stiffness", self.twist_stiffness_mode == "fit"),
                ("bend_damping", self.opt_bend_damping),
                ("twist_damping", self.twist_damping_mode == "fit"),
            )
            if searched
        ]
