# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

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

from .builders import basics, basics_newton, testing, utils
from .builders.utils import get_basics_usd_assets_path, get_testing_usd_assets_path

__all__ = [
    "basics",
    "basics_newton",
    "builders",
    "get_basics_usd_assets_path",
    "get_testing_usd_assets_path",
    "testing",
    "utils",
]
