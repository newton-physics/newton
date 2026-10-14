# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Map between decision vectors and cable parameters."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from .evaluate import CableParameters
from .schema import is_real as _is_real

SEARCHABLE = ("angles", "bend_stiffness", "twist_stiffness", "bend_damping", "twist_damping")
"""The :class:`~.evaluate.CableParameters` fields that a search can change, in decision-vector order."""

_SCALARS = SEARCHABLE[1:]


class CableSearchSpace:
    """Map a decision vector to cable parameters, and give the search start and bounds.

    The vector holds the searched fields in the order of :data:`SEARCHABLE`:
    the rest angles, then log10 of each searched stiffness and damping. A field
    that is not searched keeps its start value.

    The rest angles hold one ``(alpha, beta)`` pair per element except the last.
    The last pair orients no segment, so it stays at its start value. The rest
    angles have no bounds.

    Args:
        start: The search start, and the value of each field that is not searched.
        search: Names of the fields to search, from :data:`SEARCHABLE`.
        bounds: ``(lower, upper)`` of each searched stiffness and damping field,
            in the units of that field. Both bounds are positive, and the start
            value is within them.
        couple_twist: Make the twist stiffness and damping equal to the bend
            stiffness and damping for every candidate. The twist fields are then
            not searched, and their start values equal the bend values.

    Raises:
        ValueError: If a field name is not known, a start value is not valid,
            the bounds of a searched field are missing or not valid, the
            coupling contradicts the search or the start, or nothing is searched.
    """

    def __init__(
        self,
        start: CableParameters,
        *,
        search: Sequence[str],
        bounds: Mapping[str, Sequence[float]],
        couple_twist: bool,
    ):
        if not isinstance(start, CableParameters):
            raise ValueError(f"start must be CableParameters, got {type(start).__name__}.")
        if isinstance(search, str) or not all(isinstance(name, str) for name in search):
            raise ValueError(f"search must be a sequence of field names, got {search!r}.")
        unknown = sorted(set(search) - set(SEARCHABLE))
        if unknown:
            raise ValueError(f"unknown field(s) {unknown} in search; choose from {list(SEARCHABLE)}.")
        if type(couple_twist) is not bool:
            raise ValueError(f"couple_twist must be True or False, got {couple_twist!r}.")

        angles = start.angles
        if not isinstance(angles, (list, tuple)) or not all(
            isinstance(pair, (list, tuple)) and len(pair) == 2 and all(_is_real(v) and math.isfinite(v) for v in pair)
            for pair in angles
        ):
            raise ValueError("start.angles must be pairs of finite numbers.")
        for name in _SCALARS:
            value = getattr(start, name)
            if not _is_real(value) or not math.isfinite(value) or value < 0:
                raise ValueError(f"start.{name} must be a finite number >= 0, got {value!r}.")

        self.search = [name for name in SEARCHABLE if name in search]
        """The searched fields, in decision-vector order."""

        if couple_twist:
            for twist, bend in (("twist_stiffness", "bend_stiffness"), ("twist_damping", "bend_damping")):
                if twist in self.search:
                    raise ValueError(f"{twist} follows {bend} when couple_twist is set; do not search it.")
                if getattr(start, twist) != getattr(start, bend):
                    raise ValueError(f"start.{twist} must equal start.{bend} when couple_twist is set.")

        searched_scalars = [name for name in self.search if name != "angles"]
        if set(bounds) != set(searched_scalars):
            raise ValueError(
                f"bounds must be given for exactly the searched stiffness and damping fields "
                f"{searched_scalars}, got {sorted(bounds)}."
            )
        self.bounds_by_field = {}
        """``(lower, upper)`` of each searched stiffness and damping field."""
        for name in searched_scalars:
            pair = bounds[name]
            if (
                isinstance(pair, str)
                or len(pair) != 2
                or not all(_is_real(v) and math.isfinite(v) for v in pair)
                or not 0 < pair[0] < pair[1]
            ):
                raise ValueError(f"bounds of {name} must be (lower, upper) with 0 < lower < upper, got {pair!r}.")
            if not pair[0] <= getattr(start, name) <= pair[1]:
                raise ValueError(f"start.{name} = {getattr(start, name)!r} is outside its bounds {tuple(pair)}.")
            self.bounds_by_field[name] = (float(pair[0]), float(pair[1]))

        self.start = start
        self.couple_twist = couple_twist
        self.n_bend = max(len(angles) - 1, 0)
        if self.dim == 0:
            raise ValueError("nothing is searched: a cable with one element has no rest angle to search.")

    @property
    def dim(self) -> int:
        """Number of searched dimensions."""
        return (2 * self.n_bend if "angles" in self.search else 0) + len(self.bounds_by_field)

    def searched_scalars(self) -> list[str]:
        """Return the names of the searched stiffness and damping fields, in decision-vector order."""
        return list(self.bounds_by_field)

    def x0(self) -> list[float]:
        """Return the search start as a decision vector."""
        x = []
        if "angles" in self.search:
            x += [float(v) for pair in self.start.angles[: self.n_bend] for v in pair]
        x += [math.log10(getattr(self.start, name)) for name in self.bounds_by_field]
        return x

    def bounds(self) -> tuple[list[float], list[float]]:
        """Return the lower and upper bound of each decision variable. The rest angles are not bounded."""
        n_angles = 2 * self.n_bend if "angles" in self.search else 0
        lower = [-math.inf] * n_angles + [math.log10(lo) for lo, _ in self.bounds_by_field.values()]
        upper = [math.inf] * n_angles + [math.log10(hi) for _, hi in self.bounds_by_field.values()]
        return lower, upper

    def describe(self, x: Sequence[float]) -> str:
        """Return the searched values of a decision vector as text, for progress logs."""
        parameters = self.decode(x)
        parts = [f"{name}={getattr(parameters, name):.4g}" for name in self.bounds_by_field]
        if "angles" in self.search and self.n_bend:
            parts.append(f"{2 * self.n_bend} rest angles")
        return ", ".join(parts)

    def decode(self, x: Sequence[float]) -> CableParameters:
        """Return the cable parameters that a decision vector describes.

        Raises:
            ValueError: If ``x`` does not have :attr:`dim` finite values within :meth:`bounds`.
        """
        lower, upper = self.bounds()
        if len(x) != self.dim or not all(
            math.isfinite(float(v)) and lo <= float(v) <= hi for v, lo, hi in zip(x, lower, upper, strict=True)
        ):
            raise ValueError(f"expected {self.dim} finite decision variables within the bounds.")
        values = {name: getattr(self.start, name) for name in SEARCHABLE}
        values["angles"] = [tuple(pair) for pair in self.start.angles]
        i = 0
        if "angles" in self.search:
            values["angles"][: self.n_bend] = [(float(x[2 * k]), float(x[2 * k + 1])) for k in range(self.n_bend)]
            i = 2 * self.n_bend
        for name in self.bounds_by_field:
            values[name] = 10.0 ** float(x[i])
            i += 1
        if self.couple_twist:
            values["twist_stiffness"] = values["bend_stiffness"]
            values["twist_damping"] = values["bend_damping"]
        return CableParameters(**values)

    def to_dict(self) -> dict[str, Any]:
        """Return the start, the searched fields, the bounds and the coupling, for the provenance of a result."""
        return {
            "start": {
                "angles": [list(pair) for pair in self.start.angles],
                **{name: getattr(self.start, name) for name in _SCALARS},
            },
            "search": list(self.search),
            "bounds": {name: list(pair) for name, pair in self.bounds_by_field.items()},
            "couple_twist": self.couple_twist,
        }
