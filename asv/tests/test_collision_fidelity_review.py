# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check the static review page against saved results without physics or video decoding."""

import importlib
import importlib.util
import json
import sys
import tempfile
import unittest
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote

sys.path.insert(0, str(Path(__file__).parents[1]))


class _Page(HTMLParser):
    """Collect rendered text, links and table cells from the generated artifact."""

    def __init__(self, source):
        super().__init__()
        self.tags, self.visible_text, self.rows = [], [], {}
        self.depth, self.table, self.row, self.cell = 0, None, None, None
        self.feed(source)

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        self.tags.append((tag, attrs))
        if tag == "details":
            self.depth += 1
        elif tag == "table":
            self.table = attrs.get("id")
            self.rows[self.table] = []
        elif tag == "tr":
            self.row = []
        elif tag in ("th", "td"):
            self.cell = []

    def handle_endtag(self, tag):
        if tag == "details":
            self.depth -= 1
        elif tag in ("th", "td") and self.cell is not None:
            self.row.append("".join(self.cell).strip())
            self.cell = None
        elif tag == "tr" and self.table is not None:
            self.rows[self.table].append(self.row)
            self.row = None
        elif tag == "table":
            self.table = None

    def handle_data(self, text):
        if not self.depth:
            self.visible_text.append(text)
        if self.cell is not None:
            self.cell.append(text)


class TestFidelityReview(unittest.TestCase):
    def render(self, directory, results, cases):
        """Write real JSON inputs and invoke the page writer through its public API."""
        self.assertIsNotNone(importlib.util.find_spec("collision_fidelity.review"), "Review writer is missing")
        module = importlib.import_module("collision_fidelity.review")
        for result in results:
            (directory / f"{result['case_id']}.json").write_text(json.dumps(result))
        path = module.write_review(directory, results, cases)
        self.assertEqual(path, directory / "index.html")
        return _Page(path.read_text())

    def test_counts_values_and_missing_evidence_remain_distinct(self):
        """Do not turn missing windows into zeros or incomplete outcomes into failures."""
        results = [
            {
                "case_id": "complete",
                "execution": {"status": "completed", "full_timeline_completed": True},
                "measurements": {
                    "finite": {"all_samples_finite": True},
                    "stability": {
                        "jitter_m": 0.001234,
                        "rms_speed_m_s": 0.02,
                        "complete_window_count": 2,
                        "declared_complete_window_count": 2,
                    },
                    "penetration": {"rigid_soft_penetration_m": 0.0},
                },
                "criteria": {"hold": {"status": "incomplete"}, "overall": {"status": "incomplete"}},
            },
            {
                "case_id": "failed",
                "execution": {"status": "nonfinite"},
                "measurements": {
                    "stability": {
                        "jitter_m": 99.0,
                        "rms_speed_m_s": None,
                        "complete_window_count": 1,
                        "declared_complete_window_count": 2,
                    }
                },
                "criteria": {"hold": {"status": "fail"}, "overall": {"status": "fail"}},
            },
            {
                "case_id": "unsupported",
                "execution": {"status": "unsupported"},
                "measurements": None,
                "criteria": {"overall": {"status": "unsupported"}},
            },
        ]
        with tempfile.TemporaryDirectory() as temporary:
            page = self.render(Path(temporary), results, {})
        self.assertIn(["Hold", "1", "1", "2", "1", "0", "1", "100", "50"], page.rows["aggregate-criteria"])
        self.assertIn(["Jitter (mm)", "1.234", "1"], page.rows["aggregate-measurements"])
        self.assertIn(["RMS speed (m/s)", "0.02", "1"], page.rows["aggregate-measurements"])
        self.assertIn(["Rigid-soft penetration: fixed supports (mm)", "0", "1"], page.rows["aggregate-measurements"])
        text = " ".join(page.visible_text)
        self.assertIn("Incomplete", text)
        self.assertIn("Fail", text)
        self.assertIn("Missing", text)
        self.assertNotIn("None", text)
        self.assertEqual(len([tag for tag, _ in page.tags if tag == "article"]), 3)

    def test_unevaluated_criteria_have_no_failure_rate(self):
        """Display an empty evaluated denominator as missing, not a zero failure rate."""
        result = {
            "case_id": "unknown",
            "execution": {"status": "completed"},
            "criteria": {"overall": {"status": "incomplete"}},
        }
        with tempfile.TemporaryDirectory() as temporary:
            page = self.render(Path(temporary), [result], {})
        self.assertIn(["Overall", "0", "0", "1", "1", "0", "0", "Missing", "100"], page.rows["aggregate-criteria"])

    def test_details_are_folded_but_actual_integration_is_visible(self):
        """Keep actual step settings visible and long descriptions closed by default."""
        result = {
            "case_id": "case",
            "execution": {"status": "completed"},
            "provenance": {"frame_dt_s": 0.02, "dt_s": 0.002, "substeps": 10, "iterations": 40},
            "inputs": {"physics_parameters": {"soft_contact_ke": 18000}},
        }
        with tempfile.TemporaryDirectory() as temporary:
            page = self.render(
                Path(temporary),
                [result],
                {"case": {"title": "Readable title", "description": "Long material explanation", "iterations": 3}},
            )
        text = " ".join(page.visible_text)
        for required in ("Readable title", "Frame dt 0.02 s", "Physical dt 0.002 s", "Substeps 10", "Iterations 40"):
            self.assertIn(required, text)
        self.assertNotIn("Long material explanation", text)
        self.assertTrue(any(tag == "details" for tag, _ in page.tags))
        self.assertTrue(all("open" not in attrs for tag, attrs in page.tags if tag == "details"))

    def test_untrusted_text_and_relative_links_cannot_create_markup(self):
        """Escape titles, reasons, metadata and filenames rather than injecting HTML."""
        case_id = "case' onmouseover='alert(1)"
        video = 'clip"><img src=x onerror=alert(1)>.mp4'
        malicious = '<script>alert("title")</script>'
        result = {
            "case_id": case_id,
            "execution": {"status": "error", "reason": malicious},
            "inputs": {"material": malicious},
            "criteria": {"overall": {"status": "incomplete", "reason": malicious}},
            "recording": {"status": "partial", "path": video, "frame_count": 2, "fps": 30},
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / video).write_bytes(b"recorded bytes")
            page = self.render(root, [result], {case_id: {"title": malicious, "description": malicious}})
        self.assertFalse(any(tag in ("img", "script") for tag, _ in page.tags))
        self.assertFalse(any(key.startswith("on") for _, attrs in page.tags for key in attrs))
        sources = [attrs["src"] for tag, attrs in page.tags if tag == "video"]
        self.assertEqual([unquote(source) for source in sources], [video])
        links = [unquote(attrs["href"]) for tag, attrs in page.tags if tag == "a"]
        self.assertIn(f"{case_id}.json", links)
        self.assertTrue(all(not link.startswith(("http:", "https:", "javascript:", "//")) for link in links))

    def test_only_usable_local_completed_or_partial_recordings_are_embedded(self):
        """Keep missing, empty, failed and escaping paths out of the video elements."""
        cases = [
            ("partial", "partial", 3, True),
            ("complete", "completed", 3, True),
            ("failed", "error", 3, True),
            ("empty", "completed", 0, True),
            ("missing", "completed", 3, False),
            ("unsupported", "unsupported", 0, False),
        ]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            results = []
            for name, status, count, exists in cases:
                if exists:
                    (root / f"{name}.mp4").write_bytes(b"recorded bytes")
                results.append(
                    {
                        "case_id": name,
                        "execution": {"status": "completed"},
                        "recording": {"status": status, "path": f"{name}.mp4", "frame_count": count},
                    }
                )
            results.append(
                {
                    "case_id": "outside",
                    "execution": {"status": "error"},
                    "recording": {"status": "partial", "path": "../outside.mp4", "frame_count": 3},
                }
            )
            page = self.render(root, results, {})
        self.assertEqual([attrs["src"] for tag, attrs in page.tags if tag == "video"], ["partial.mp4", "complete.mp4"])
        text = " ".join(page.visible_text)
        self.assertIn("Partial recording", text)
        self.assertEqual(text.count("Not visually reviewed"), 2)


if __name__ == "__main__":
    unittest.main()
