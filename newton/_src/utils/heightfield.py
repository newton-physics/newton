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

import os

import numpy as np


def load_heightfield_elevation(
    filename: str,
    nrow: int,
    ncol: int,
) -> np.ndarray:
    """Load elevation data from a PNG or binary file.

    Supports two formats following MuJoCo conventions:
    - PNG: Grayscale image where white=high, black=low
      (normalized to [0, 1])
    - Binary: MuJoCo custom format with int32 header
      (nrow, ncol) followed by float32 data

    Args:
        filename: Path to the heightfield file (PNG or binary).
        nrow: Expected number of rows.
        ncol: Expected number of columns.

    Returns:
        (nrow, ncol) float32 array of elevation values.
    """
    ext = os.path.splitext(filename)[1].lower()

    if ext == ".png":
        from PIL import Image  # noqa: PLC0415

        img = Image.open(filename).convert("L")
        data = np.array(img, dtype=np.float32) / 255.0
        if data.shape != (nrow, ncol):
            raise ValueError(f"PNG heightfield dimensions {data.shape} don't match expected ({nrow}, {ncol})")
        return data

    # Default: MuJoCo binary format
    # Header: (int32) nrow, (int32) ncol; payload: float32[nrow*ncol]
    with open(filename, "rb") as f:
        header = np.fromfile(f, dtype=np.int32, count=2)
        if header.size != 2 or header[0] <= 0 or header[1] <= 0:
            raise ValueError(
                f"Invalid binary heightfield header in '{filename}': expected 2 positive int32 values, got {header}"
            )
        expected_count = int(header[0]) * int(header[1])
        data = np.fromfile(f, dtype=np.float32, count=expected_count)
        if data.size != expected_count:
            raise ValueError(
                f"Binary heightfield '{filename}' payload size mismatch: "
                f"expected {expected_count} float32 values for {header[0]}x{header[1]} grid, got {data.size}"
            )
    return data.reshape(header[0], header[1])
