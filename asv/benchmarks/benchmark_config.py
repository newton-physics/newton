# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import os


def pr_gate_repeat(default: int, pr_repeat: int = 3) -> int:
    """Return a bounded PR sample count without changing full ASV runs."""
    if os.environ.get("NEWTON_ASV_PR_GATE"):
        return min(default, pr_repeat)
    return default
