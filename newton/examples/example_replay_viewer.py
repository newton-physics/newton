# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

###########################################################################
# Example Integrated Viewer
#
# Shows how to use the replay UI extension with ViewerGL to load and
# display previously recorded simulation data.
#
# Recording is done automatically using ViewerFile (like ViewerUSD):
#   viewer = newton.viewer.ViewerFile("my_recording.json")
#   viewer.set_model(model)
#   viewer.log_state(state)  # Records automatically
#   viewer.close()  # Saves automatically
#
# Command: python -m newton.examples.example_replay_viewer
#
###########################################################################


import newton
import newton.examples


class Example:
    def __init__(self, viewer):
        """Initialize the integrated viewer example with replay UI."""
        self.viewer = viewer

        # Add replay UI extension to the viewer
        self.replay_ui = newton.viewer.ui_extensions.ReplayUI()
        self.viewer.add_custom_ui(self.replay_ui.render, "ReplayControls")

        # No simulation - this example is purely for replay
        self.sim_time = 0.0

    def step(self):
        """No simulation step needed - replay is handled by UI."""
        pass

    def render(self):
        """Render the current state (managed by replay UI)."""
        self.viewer.begin_frame(self.sim_time)
        # Current state is logged by the replay UI when frames are loaded
        # No need to call viewer.log_state() here
        self.viewer.end_frame()

    def test(self):
        pass


if __name__ == "__main__":
    # Parse arguments and initialize viewer
    viewer, args = newton.examples.init()

    # Create example and run
    example = Example(viewer)

    newton.examples.run(example)
