# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Stage-driven simulation runtime: derive a Newton simulation from USD."""

from .._src.usd.runtime import Simulation, load_usd, step

__all__ = ["Simulation", "load_usd", "step"]

if __name__ == "__main__":
    from .._src.usd.runtime import _main

    _main()
