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

"""Direct Cosserat rod: native reference + GPU-resident Warp candidate.

This example runs two direct rods side-by-side:
- Reference rod: full native DefKitAdv.dll pipeline.
- Candidate rod: GPU-resident Warp implementation that avoids host round-trips.

Command:
    uv run python newton/examples/cosserat_codex/gpu_warp_cosserat_codex.py --dll-path "C:\\path\\to\\DefKitAdv.dll"
"""

from __future__ import annotations

import argparse

import newton
import newton.examples

from newton.examples.cosserat_codex.cli import create_parser as create_base_parser
from newton.examples.cosserat_codex.simulation import Example


def create_parser():
    """Create argument parser with vsync toggle."""
    parser = create_base_parser()
    parser.add_argument(
        "--vsync",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable vertical sync.",
    )
    return parser


if __name__ == "__main__":
    viewer, args = newton.examples.init(create_parser())

    if isinstance(viewer, newton.viewer.ViewerGL):
        viewer.show_particles = True
        viewer.vsync = args.vsync

    example = Example(viewer, args)
    newton.examples.run(example, args)
