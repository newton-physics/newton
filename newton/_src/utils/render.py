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

from __future__ import annotations


def bourke_color_map(low: float, high: float, v: float) -> list[float]:
    """Map a scalar value to an RGB color using Bourke's color ramp.

    Apply smooth rainbow color mapping where the value is linearly
    interpolated across five color bands: blue → cyan → green → yellow → red.
    Values outside the [low, high] range are clamped.

    Based on Paul Bourke's colour ramping method:
    https://paulbourke.net/texture_colour/colourspace/

    Args:
        low: Minimum value of the input range.
        high: Maximum value of the input range.
        v: The scalar value to map to a color.

    Returns:
        RGB color as a list of three floats in the range [0.0, 1.0].
    """
    c = [1.0, 1.0, 1.0]

    if v < low:
        v = low
    if v > high:
        v = high
    dv = high - low

    if v < (low + 0.25 * dv):
        c[0] = 0.0
        c[1] = 4.0 * (v - low) / dv
    elif v < (low + 0.5 * dv):
        c[0] = 0.0
        c[2] = 1.0 + 4.0 * (low + 0.25 * dv - v) / dv
    elif v < (low + 0.75 * dv):
        c[0] = 4.0 * (v - low - 0.5 * dv) / dv
        c[2] = 0.0
    else:
        c[1] = 1.0 + 4.0 * (low + 0.75 * dv - v) / dv
        c[2] = 0.0

    return c
