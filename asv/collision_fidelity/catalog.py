# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Lightweight case metadata; listing never imports a physics backend."""

import json
from pathlib import Path


def load_cases():
    """Return independent case records keyed by their stable identifier."""
    manifest = json.loads(Path(__file__).with_name("cases.json").read_text())
    cases = {case["case_id"]: case for case in manifest["cases"]}
    if len(cases) != manifest["case_count"]:
        raise ValueError("Duplicate or missing collision-fidelity case identifier")
    return cases


def get_case(case_id):
    """Return one case or raise KeyError for an unknown identifier."""
    return load_cases()[case_id]
