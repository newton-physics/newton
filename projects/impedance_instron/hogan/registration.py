# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Register measured motion to the shoe geometry from the static standing trial."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .adaptation import characterize_shoe


@dataclass(frozen=True)
class StaticRegistration:
    """Vertical shift that seats the static ankle on the shoe compressed by standing load.

    Attributes:
        height_offset_m: Shift added to the measured hip height [m].
        static_ankle_height_m: Measured static ankle-center height [m].
        unloaded_ankle_height_m: Ankle height at geometric shoe contact, level shoe [m].
        static_compression_m: Shoe compression under the static load [m].
        static_load_n: Vertical load on the registered foot while standing [N].
    """

    height_offset_m: float
    static_ankle_height_m: float
    unloaded_ankle_height_m: float
    static_compression_m: float
    static_load_n: float

    def to_dict(self) -> dict:
        """Return the registration as JSON-ready values."""
        return asdict(self)


def register_static_height(
    reference: dict, shoe, *, static_load_n: float, speed_m_s: float = 0.005, dt_s: float = 5e-4
) -> StaticRegistration:
    """Shift motion so the static ankle sits where the shoe model supports the standing load.

    The malleolus-based ankle center and the shoe's ankle mount are located
    independently. In the static trial the shoe is level by construction (its
    pitch comes from the static foot markers), so the only unknown is how far
    the standing load compresses it. A slow press approximates the relaxed
    standing response. The offset depends on the reference shoe only and must be
    held fixed when the shoe is changed.

    Args:
        reference: Cartesian single-leg reference with ``static_ankle_m``.
        shoe: :class:`..cartesian.shoe.Shoe` worn in the static trial.
        static_load_n: Vertical load on this foot while standing [N].
        speed_m_s: Quasi-static press speed [m/s].
        dt_s: Contact step [s].
    """
    _, curve = characterize_shoe(shoe, peak_force_n=static_load_n, speed_m_s=speed_m_s, dt_s=dt_s)
    compression = float(curve[np.argmax(curve[:, 1] >= static_load_n), 0])
    unloaded = -float(np.min(shoe.anchor_local_m[:, 2]))
    static_ankle = float(reference["static_ankle_m"][1])
    return StaticRegistration(
        height_offset_m=unloaded - compression - static_ankle,
        static_ankle_height_m=static_ankle,
        unloaded_ankle_height_m=unloaded,
        static_compression_m=compression,
        static_load_n=static_load_n,
    )
