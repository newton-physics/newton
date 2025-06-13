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


import warp as wp


@wp.struct
class ShapeMaterials:
    """
    Represents the contact material properties of a shape.
        ke: The contact elastic stiffness (only used by the Euler integrators)
        kd: The contact damping stiffness (only used by the Euler integrators)
        kf: The contact friction stiffness (only used by the Euler integrators)
        ka: The contact adhesion distance (values greater than 0 mean adhesive contact; only used by the Euler integrators)
        mu: The coefficient of friction
        restitution: The coefficient of restitution (only used by XPBD integrator)
    """

    ke: wp.array(dtype=float)
    kd: wp.array(dtype=float)
    kf: wp.array(dtype=float)
    ka: wp.array(dtype=float)
    mu: wp.array(dtype=float)
    restitution: wp.array(dtype=float)


@wp.struct
class ShapeGeometry:
    """
    Represents the geometry of a shape.
        type: The type of geometry (GEO_SPHERE, GEO_BOX, etc.)
        is_solid: Indicates whether the shape is solid or hollow
        thickness: The thickness of the shape (used for collision detection, and inertia computation of hollow shapes)
        source: Pointer to the source geometry (can be a mesh or SDF index, zero otherwise)
        scale: The 3D scale of the shape
        filter: The filter group of the shape
        transform: The transform of the shape in world space
    """

    type: wp.array(dtype=wp.int32)
    is_solid: wp.array(dtype=bool)
    thickness: wp.array(dtype=float)
    source: wp.array(dtype=wp.uint64)
    scale: wp.array(dtype=wp.vec3)
    filter: wp.array(dtype=int)
