# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example USD Runtime
#
# Stage-driven simulation: the USD file authors the physics scene, the
# solver choice, and the solver configuration. No manual scene wiring.
#
# Command: python -m newton.examples usd_runtime
###########################################################################

import numpy as np

import newton
import newton.examples
import newton.usd.runtime


class Example:
    def __init__(self, viewer, args):
        self.viewer = viewer
        self.sim = newton.usd.runtime.load_usd(str(newton.examples.get_asset("usd_runtime_boxes.usda")))
        fps = self.sim.usd_info.get("fps") or 60.0
        self.steps_per_frame = max(1, round(1.0 / (fps * self.sim.dt)))
        self.viewer.set_model(self.sim.model)

    def step(self):
        for _ in range(self.steps_per_frame):
            newton.usd.runtime.step(self.sim)

    def render(self):
        self.viewer.begin_frame(self.sim.time)
        self.viewer.log_state(self.sim.state)
        self.viewer.end_frame()

    def test_final(self):
        body_q = self.sim.state.body_q.numpy()
        assert np.isfinite(body_q).all(), "non-finite body transforms"
        assert (body_q[:, 2] > 0.0).all(), "boxes fell through the ground"
        assert (body_q[:, 2] < 3.0).all(), "boxes did not settle"


if __name__ == "__main__":
    viewer, args = newton.examples.init()
    newton.examples.run(Example(viewer, args), args)
