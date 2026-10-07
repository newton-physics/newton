# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Dense multi-block system info with linear-time offset computation."""

from __future__ import annotations

import numpy as np
import warp as wp

from ....linalg.core import DenseSquareMultiLinearInfo

###
# Module interface
###

__all__ = ["DenseBlockInfo"]


###
# Interfaces
###


class DenseBlockInfo(DenseSquareMultiLinearInfo):
    """Kamino's square multi-linear system info, with offsets from prefix sums.

    Kamino computes the block offsets with a loop that is quadratic in the
    number of blocks, which dominates setup when LOX has many factor blocks.
    """

    def finalize(
        self,
        dimensions: list[int],
        dtype: type = wp.float32,
        itype: type = wp.int32,
        device: wp.DeviceLike = None,
    ) -> None:
        """Allocate the block dimensions and offsets on the specified device."""
        self.dimensions = self._check_dimensions(dimensions)
        self.dtype = dtype
        self.itype = itype
        if device is not None:
            self.device = device

        dimensions_np = np.asarray(self.dimensions, dtype=np.int64)
        mat_offsets = np.concatenate(([0], np.cumsum(dimensions_np * dimensions_np)))
        vec_offsets = np.concatenate(([0], np.cumsum(dimensions_np)))
        self.num_blocks = len(self.dimensions)
        self.max_dimension = max(self.dimensions)
        self.total_mat_size = int(mat_offsets[-1])
        self.total_vec_size = int(vec_offsets[-1])
        with wp.ScopedDevice(self.device):
            self.maxdim = wp.array(self.dimensions, dtype=self.itype)
            self.dim = wp.array(self.dimensions, dtype=self.itype)
            self.mio = wp.array(mat_offsets[:-1], dtype=self.itype)
            self.vio = wp.array(vec_offsets[:-1], dtype=self.itype)
