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

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import warp as wp

###
# Module interface
###

__all__ = ["setup_tests", "test_context"]


###
# Global test context
###


@dataclass
class TestContext:
    setup_done: bool = False
    """Whether the global test setup has already run """

    verbose: bool = False
    """Global default verbosity flag to be used by unit tests """

    device: wp.DeviceLike | None = None
    """Global default device to be used by unit tests """

    output_path: Path | None = None
    """Global cache directory for tests to use, if any."""


test_context = TestContext()
""" Global object shared across unit tests, containing status & settings regarding test execution. """


###
# Functions
###


def setup_tests(verbose: bool = False, device: wp.DeviceLike | str | None = None, clear_cache: bool = True):
    # Numpy configuration
    np.set_printoptions(
        linewidth=999999, edgeitems=999999, threshold=999999, precision=10, suppress=True
    )  # Suppress scientific notation

    # Warp configuration
    wp.init()
    wp.config.mode = "release"
    wp.config.enable_backward = False
    wp.config.verbose = False
    wp.config.verify_fp = False
    wp.config.verify_cuda = False

    # Clear cache
    if clear_cache:
        wp.clear_kernel_cache()
        wp.clear_lto_cache()

    # Update test context
    test_context.verbose = verbose
    test_context.device = wp.get_device(device)
    test_context.setup_done = True

    # Set the cache directory for optional test output, if any
    # Data directory (contains perfprof.csv)
    test_context.output_path = Path(__file__).parent / "output"
    test_context.output_path.mkdir(parents=True, exist_ok=True)
