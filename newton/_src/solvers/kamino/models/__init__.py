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

"""
Predefined models for testing and demonstration of Kamino.

This module provides a collection of model builders and relevant utilities
for testing and demonstrating the features of the Kamino physics solver.

These include:

- A set of utility functions to retrieve paths to USD asset directories

- A set of 'example' models in the form of USD assets.\n
    This directory currently provides a symbolic link to `newton-assets/disneyreasearch`.

- A set of 'basic' models used for demonstrating fundamental features of Kamino and for testing purposes.
    These are provided both in the form of USD assets as well as manually constructed model builders.

- A set of `testing` models that are used to almost exclusively for unit testing, and include:
    - supported geometric shapes, e.g. boxes, spheres, capsules, etc.
    - supported joint types,e.g. revolute, prismatic, spherical, etc.
"""

import os

from .builders import basics, testing, utils

__all__ = [
    "basics",
    "builders",
    "get_basics_usd_assets_path",
    "get_examples_usd_assets_path",
    "get_testing_usd_assets_path",
    "testing",
    "utils",
]

###
# Asset path utilities
###


def get_examples_usd_assets_path() -> str | None:
    """
    Returns the path to the USD assets for example models.

    If the path does not exist, returns None.
    """
    path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "assets/examples")
    if not os.path.exists(path):
        return None
    return path


def get_basics_usd_assets_path() -> str:
    """
    Returns the path to the USD assets for basic models.
    """
    path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "assets/basics")
    if not os.path.exists(path):
        raise FileNotFoundError(f"The USD assets path for basic models does not exist: {path}")
    return path


def get_testing_usd_assets_path() -> str:
    """
    Returns the path to the USD assets for testing models.
    """
    path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "assets/testing")
    if not os.path.exists(path):
        raise FileNotFoundError(f"The USD assets path for testing models does not exist: {path}")
    return path
