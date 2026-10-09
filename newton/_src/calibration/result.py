# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""The outcome of a cable calibration run.

:class:`CableCalibrationResult` records what a run produced and enough context
to judge whether to trust it: the fitted values, the metrics they scored, the
evidence and spec that produced them, and the versions of the software that ran.

It separates the *fit* from the *judgement of the fit*. ``fit`` holds the fitted
values as a flat mapping, which keeps the file readable by tools that only want
the numbers. ``metrics``, ``provenance`` and ``artifacts`` describe the run
around them.

On comparing runs
-----------------
Recording the seed is necessary to compare two runs, and it is not sufficient.
Reproducibility also depends on the solver, the renderer and the hardware, which
this record knows nothing about: kernels that reduce in floating point can give
different results run to run purely from scheduling order. So the seed is stored
as provenance, and whether two runs are genuinely comparable is a judgement for
the caller, who knows what produced them.

What does follow from an absent seed: the run was a single sample of a
stochastic search, and any difference from another run may be the seed rather
than the change under test.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from typing import Any

from .schema import SCHEMA_VERSION, check_fields, check_schema_version

STATUS_OK = "ok"
STATUS_INTERRUPTED = "interrupted"
STATUS_FAILED = "failed"
STATUSES = (STATUS_OK, STATUS_INTERRUPTED, STATUS_FAILED)


@dataclass
class CableCalibrationResult:
    """Fitted values plus the context needed to judge and reproduce them."""

    status: str
    """One of :data:`STATUSES`. ``interrupted`` means the search stopped early and
    the fit holds the best values found, which are not converged."""

    fit: dict[str, Any]
    """Fitted values as a flat mapping, for example ``bend_angles`` and ``bend_stiffness``."""

    metrics: dict[str, Any] = field(default_factory=dict)
    """What the fit scored: objective value, iterations, and any per-view
    breakdown the runtime reports."""

    diagnostics: dict[str, Any] = field(default_factory=dict)
    """Optional run detail, such as convergence behaviour or per-view losses."""

    provenance: dict[str, Any] = field(default_factory=dict)
    """Bundle path, run spec, seed, and software versions."""

    artifacts: dict[str, str] = field(default_factory=dict)
    """Paths written by the run, relative to the artifact directory."""

    schema_version: int = SCHEMA_VERSION
    """Format version of the serialized result."""

    def to_dict(self) -> dict[str, Any]:
        """Return the result as a JSON-serializable mapping."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> CableCalibrationResult:
        """Build a result from :meth:`to_dict` output.

        Raises:
            ValueError: If the schema version is not supported, or ``d`` has a
                field this class does not define.
        """
        d = dict(d)
        check_schema_version(d.get("schema_version"), "CableCalibrationResult")
        check_fields(cls, d, "CableCalibrationResult")
        return cls(**d)

    def validate(self) -> None:
        """Check the status is known and the result has fitted values.

        Raises:
            ValueError: If the schema version, status or fitted values are invalid.
        """
        check_schema_version(self.schema_version, "CableCalibrationResult")
        if self.status not in STATUSES:
            raise ValueError(f"unknown status {self.status!r}; expected one of {list(STATUSES)}.")
        if not self.fit:
            raise ValueError("result has no fitted values.")
