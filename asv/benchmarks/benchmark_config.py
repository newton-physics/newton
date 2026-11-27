# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import os


def pr_gate_repeat(default: int, pr_repeat: int = 2) -> int:
    """Return a bounded PR sample count without changing full ASV runs."""
    # 3 repeats over ASV's 2 rounds give 6 samples per revision, where ASV can never report a change.
    if os.environ.get("NEWTON_ASV_PR_GATE"):
        return min(default, pr_repeat)
    return default
