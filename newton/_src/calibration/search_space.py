# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Map between decision vectors and cable candidates."""

from __future__ import annotations

import math
from collections.abc import Sequence

from .evaluate import CableCandidate


class CableSearchSpace:
    """Map a decision vector to a candidate, and give the search start.

    The vector holds the searched groups in this order: the rest angles, then
    log10 of the bend stiffness, twist stiffness, bend damping and twist
    damping. A group has a slot only when it is searched. A bend group is
    searched when its ``opt_*`` flag is set. A twist group is searched when its
    mode is ``"fit"``.

    The rest angles hold one ``(alpha, beta)`` pair per element except the last.
    The last pair orients no segment, so it stays at its start value.

    Args:
        run_spec: The :class:`~.run_spec.CableRunSpec` that sets what to search.

    Raises:
        ValueError: If ``init_angles`` has the wrong length, or a searched
            scalar has no positive start value.
    """

    def __init__(self, run_spec):
        self.spec = run_spec
        self.n = run_spec.setup.num_elements
        self.n_bend = self.n - 1

        self.base_angles = run_spec.init_angles if run_spec.init_angles is not None else [(0.0, 0.0)] * self.n
        if len(self.base_angles) != self.n:
            raise ValueError(f"init_angles has {len(self.base_angles)} pair(s) for {self.n} element(s).")

        self.init_bend_stiffness = run_spec.init_bend_stiffness
        self.init_bend_damping = run_spec.init_bend_damping
        self.init_twist_stiffness = (
            run_spec.init_twist_stiffness if run_spec.init_twist_stiffness is not None else run_spec.init_bend_stiffness
        )
        self.init_twist_damping = (
            run_spec.init_twist_damping if run_spec.init_twist_damping is not None else run_spec.init_bend_damping
        )

        # Each scalar is searched in log10, so its start must be positive.
        for searched, name, value in (
            (run_spec.opt_bend_stiffness, "init_bend_stiffness", self.init_bend_stiffness),
            (run_spec.twist_stiffness_mode == "fit", "init_twist_stiffness", self.init_twist_stiffness),
            (run_spec.opt_bend_damping, "init_bend_damping", self.init_bend_damping),
            (run_spec.twist_damping_mode == "fit", "init_twist_damping", self.init_twist_damping),
        ):
            if searched and value <= 0:
                raise ValueError(f"a log-space search of this term needs a positive {name}, got {value!r}.")

    def _offsets(self) -> dict[str, int]:
        """Return the first index of each scalar slot, in layout order, and the total size."""
        s = self.spec
        off = 2 * self.n_bend if s.opt_rest_config else 0
        out = {}
        for name, searched in (
            ("bend_stiffness", s.opt_bend_stiffness),
            ("twist_stiffness", s.twist_stiffness_mode == "fit"),
            ("bend_damping", s.opt_bend_damping),
            ("twist_damping", s.twist_damping_mode == "fit"),
        ):
            out[name] = off
            off += 1 if searched else 0
        out["dim"] = off
        return out

    @property
    def dim(self) -> int:
        """Number of searched dimensions."""
        return self._offsets()["dim"]

    def searched_scalars(self) -> list[str]:
        """Return the names of the searched scalars, in layout order.

        The names are :class:`~.evaluate.CableCandidate` attributes. The rest
        angles are not included.
        """
        off = self._offsets()
        names = [n for n in off if n != "dim"]
        return [n for n, nxt in zip(names, [*names[1:], "dim"], strict=True) if off[nxt] > off[n]]

    def x0(self) -> list[float]:
        """Return the search start as a decision vector."""
        s = self.spec
        x = []
        if s.opt_rest_config:
            x += [float(v) for pair in self.base_angles[: self.n_bend] for v in pair]
        if s.opt_bend_stiffness:
            x.append(math.log10(self.init_bend_stiffness))
        if s.twist_stiffness_mode == "fit":
            x.append(math.log10(self.init_twist_stiffness))
        if s.opt_bend_damping:
            x.append(math.log10(self.init_bend_damping))
        if s.twist_damping_mode == "fit":
            x.append(math.log10(self.init_twist_damping))
        return x

    def describe(self, x: Sequence[float]) -> str:
        """Return the searched values of a decision vector as text, for progress logs."""
        candidate = self.decode(x)
        parts = [f"{name}={getattr(candidate, name):.4g}" for name in self.searched_scalars()]
        if self.spec.opt_rest_config and self.n_bend:
            parts.append(f"{2 * self.n_bend} rest angles")
        return ", ".join(parts)

    def decode(self, x: Sequence[float]) -> CableCandidate:
        """Return the candidate that a decision vector describes.

        Raises:
            ValueError: If ``x`` does not have :attr:`dim` finite values.
        """
        if len(x) != self.dim or not all(math.isfinite(float(v)) for v in x):
            raise ValueError(f"expected {self.dim} finite decision variables.")
        s = self.spec
        off = self._offsets()

        if s.opt_rest_config:
            angles = [(float(x[2 * i]), float(x[2 * i + 1])) for i in range(self.n_bend)]
            angles.append(tuple(self.base_angles[-1]))
        else:
            angles = [tuple(a) for a in self.base_angles]

        bend_stiffness = 10.0 ** float(x[off["bend_stiffness"]]) if s.opt_bend_stiffness else self.init_bend_stiffness
        bend_damping = 10.0 ** float(x[off["bend_damping"]]) if s.opt_bend_damping else self.init_bend_damping

        if s.twist_stiffness_mode == "coupled":
            twist_stiffness = bend_stiffness
        elif s.twist_stiffness_mode == "fit":
            twist_stiffness = 10.0 ** float(x[off["twist_stiffness"]])
        else:
            twist_stiffness = float(s.twist_stiffness_mode)

        if s.twist_damping_mode == "coupled":
            twist_damping = bend_damping
        elif s.twist_damping_mode == "fit":
            twist_damping = 10.0 ** float(x[off["twist_damping"]])
        else:
            twist_damping = float(s.twist_damping_mode)

        return CableCandidate(
            angles=angles,
            bend_stiffness=bend_stiffness,
            twist_stiffness=twist_stiffness,
            bend_damping=bend_damping,
            twist_damping=twist_damping,
        )
