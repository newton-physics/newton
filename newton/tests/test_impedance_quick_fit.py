# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check the quick comparison's nested models and matched search protocol."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from projects.impedance_instron.hogan.identify import Parameterization
from projects.impedance_instron.hogan.quick_fit import active_parameters, plot, search
from projects.impedance_instron.hogan.runner import Runner, Task


class TestQuickRunnerFit(unittest.TestCase):
    """Keep comparison arms different only in variable-gain capacity."""

    def test_constant_mask_keeps_equilibrium_free(self):
        """Freeze nonconstant gain weights while retaining identical equilibrium capacity."""
        data = Runner.seed().to_dict()
        weights = np.array(data["weights"])
        weights[1:, :, 1:] = 0.0
        data["weights"] = weights.tolist()
        parameters = Parameterization(Runner.from_dict(data), [3.7])
        active = active_parameters(parameters, False)
        self.assertEqual(active.sum(), 47)
        self.assertEqual(active_parameters(parameters, True).sum(), 119)
        offsets = np.full(parameters.size, 0.1)
        offsets[~active] = 0.0
        model = parameters.model(offsets)
        np.testing.assert_array_equal(model.weights[1:, :, 1:], 0.0)
        self.assertTrue(np.any(model.weights[0] != weights[0]))

    def test_matched_budget_with_no_evaluation_trials(self):
        """Use equal candidate counts and never supply held-out data to search."""
        trial = SimpleNamespace(task=Task(3.7), split="train")
        protocol = {
            "search_dt_s": 0.0005,
            "seed": 17,
            "sigma": 0.2,
            "population": 6,
            "generations": 2,
            "bound": 1.5,
            "regularization": 0.01,
        }
        counts = []
        with tempfile.TemporaryDirectory() as directory:
            for variable in (False, True):
                destination = Path(directory) / str(variable)
                destination.mkdir()
                with patch("projects.impedance_instron.hogan.quick_fit.evaluate") as mocked:
                    mocked.return_value = {"failed": 0, "mean_loss": 1.0}
                    search(Runner.seed(), [trial], variable=variable, output=destination, protocol=protocol)
                    counts.append(mocked.call_count)
                    for call in mocked.call_args_list:
                        self.assertIs(call.args[1][0], trial)
        self.assertEqual(counts, [13, 13])

    def test_plot_handles_empty_failed_trace(self):
        """Render measurements even if a failed model produced no force intervals."""
        svg = plot(
            [("measured", np.array([0.0, 1.0]), np.array([0.0, 1.0])), ("failed", np.array([]), np.array([]))], "Force"
        )
        self.assertIn("<svg", svg)
        self.assertIn("measured", svg)


if __name__ == "__main__":
    unittest.main()
