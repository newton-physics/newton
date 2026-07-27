# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Provide small Warp utility functions for conveyor belt kernels."""

import warp as wp

# not needed for the purpose of this sample
wp.config.enable_backward = False


@wp.struct
class Vec3Pair:
    """Pair of orthogonal vectors returned by ``compute_basis_vectors``."""

    v0: wp.vec3
    v1: wp.vec3


@wp.func
def compute_basis_vectors(dir: wp.vec3) -> Vec3Pair:
    """Compute two unit vectors orthogonal to ``dir`` and to each other, returning them as a Vec3Pair.

    ``dir`` must be a normalized vector.
    """
    basis_vectors = Vec3Pair()

    if wp.abs(dir.y) <= 0.9999:
        basis_vectors.v0.x = dir.z
        basis_vectors.v0.y = 0.0
        basis_vectors.v0.z = -dir.x

        basis_vectors.v0 = wp.normalize(basis_vectors.v0)

        basis_vectors.v1.x = dir.y * basis_vectors.v0.z
        basis_vectors.v1.y = (dir.z * basis_vectors.v0.x) - (dir.x * basis_vectors.v0.z)
        basis_vectors.v1.z = -dir.y * basis_vectors.v0.x
    else:
        basis_vectors.v0.x = 1.0
        basis_vectors.v0.y = 0.0
        basis_vectors.v0.z = 0.0

        basis_vectors.v1.x = 0.0
        basis_vectors.v1.y = dir.z
        basis_vectors.v1.z = -dir.y

        basis_vectors.v1 = wp.normalize(basis_vectors.v1)

    return basis_vectors
