# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import unittest
from pathlib import Path


class TestArticulationDocs(unittest.TestCase):
    def test_issue_1213_common_workflows_documented(self):
        docs_path = Path(__file__).resolve().parents[2] / "docs" / "concepts" / "articulations.rst"
        docs_text = docs_path.read_text(encoding="utf-8")

        self.assertIn("Common articulation workflows", docs_text)
        self.assertIn("exclude_joint_types=[newton.JointType.FREE, newton.JointType.BALL]", docs_text)
        self.assertIn("set_root_transforms()", docs_text)
        self.assertIn("Model.joint_X_p", docs_text)
        self.assertIn("newton.selection.ArticulationView", docs_text)
