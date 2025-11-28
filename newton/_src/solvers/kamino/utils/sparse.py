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
KAMINO: Utilities
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

###
# Matrix Sparsity
###


def sparseview(
    matrix: np.ndarray,
    title: str = "Matrix Sparsity",
    tick_fontsize: int = 5,
    max_ticks: int = 20,
    grid: bool = False,
    path: str | None = None,
):
    """
    Visualize the sparsity pattern of a matrix.
    Zero entries are shown in red.
    Non-zero entries are shown in grayscale (black to white).

    Parameters:
    - matrix: 2D numpy array
    - title: Title for the plot (used for display or saved image)
    - save_path: If provided, saves the image to this file path; otherwise, displays it.
    """

    # Check if the input is a 2D NumPy array
    if not isinstance(matrix, np.ndarray):
        raise ValueError("Input must be a NumPy array.")

    # A helper function  to compute sparse ticks
    def get_sparse_ticks(length, max_ticks):
        if length <= max_ticks:
            return np.arange(length)
        step = max(1, int(np.ceil(length / max_ticks)))
        return np.arange(0, length, step)

    # Create color image: start with a gray image
    color_image = np.zeros((*matrix.shape, 3))  # RGB image

    # Normalize non-zero values to 0-1 for grayscale
    non_zero_mask = matrix != 0
    zero_mask = matrix == 0

    if np.any(non_zero_mask):
        norm = Normalize(vmin=matrix[non_zero_mask].min(), vmax=matrix[non_zero_mask].max())
        gray_values = norm(matrix)  # normalized to [0,1]
        gray_image = np.stack([gray_values] * 3, axis=-1)
        color_image[non_zero_mask] = gray_image[non_zero_mask]

    # Set exact zeros to red
    color_image[zero_mask] = [1, 0, 0]

    # Plot the image
    _fig, ax = plt.subplots()
    ax.imshow(color_image, origin="upper")

    # Configure figure tick labels
    xticks = get_sparse_ticks(matrix.shape[1], max_ticks)
    yticks = get_sparse_ticks(matrix.shape[0], max_ticks)
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    ax.set_xticklabels(xticks, fontsize=tick_fontsize)
    ax.set_yticklabels(yticks, fontsize=tick_fontsize)

    # Minor ticks for grid alignment (optional)
    ax.set_xticks(np.arange(matrix.shape[1] + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(matrix.shape[0] + 1) - 0.5, minor=True)
    ax.grid(False)
    ax.tick_params(which="minor", size=0)

    # Add colorbar only for the non-zero values
    if np.any(non_zero_mask):
        sm = ScalarMappable(cmap="gray", norm=norm)
        sm.set_array([])  # dummy array for colorbar
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label("non-zero values")

    # Set title and layout
    plt.title(title)
    plt.tight_layout()

    # Set grid for better visibility
    if grid:
        plt.grid(True, which="major", color="blue", linestyle="-", linewidth=0.1)

    # Save or show the plot
    if path:
        plt.savefig(path, dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show()
