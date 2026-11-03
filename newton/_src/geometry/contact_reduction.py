# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Contact reduction utilities for mesh collision.

This module provides constants, helper functions, and shared-memory utilities
used by the contact reduction system. The reduction selects a representative
subset (up to ``MAX_CONTACTS_PER_PAIR`` contacts per shape pair) that preserves
simulation stability.

**Configuration:**

Edit the constants in the "Contact Reduction Configuration" block below to
tune the reduction. All values are plain Python integers evaluated at module
import time and consumed via ``wp.static()`` in kernels, so there is zero
runtime overhead. Changing them requires restarting the process (standard Warp
kernel-caching behavior).

.. note::
   Only the default ``"icosahedron"`` polyhedron configuration is currently
   tested on CI. Other polyhedra (dodecahedron, octahedron, hexahedron) are
   functional but should be considered experimental.

**Contact Reduction Strategy Overview:**

When complex meshes collide, thousands of triangle pairs may generate contacts.
Contact reduction selects a representative subset that preserves simulation
stability while keeping memory and computation bounded.

The reduction uses three complementary strategies:

1. **Spatial Extreme Slots** (``NUM_NORMAL_BINS`` x ``NUM_SPATIAL_DIRECTIONS``)

   For each normal bin (polyhedron face), finds the most extreme contacts in
   evenly-spaced 2D scan directions on the face plane. This builds the convex
   hull / support polygon boundary, critical for stable stacking.

2. **Per-Bin Max-Depth Slots** (``NUM_NORMAL_BINS`` x 1)

   Each normal bin tracks its deepest contact unconditionally. This ensures
   deeply penetrating contacts from any normal direction are never dropped.
   Critical for gear-like contacts with varied normal orientations.

3. **Voxel-Based Depth Slots** (``NUM_VOXEL_DEPTH_SLOTS``)

   The mesh is divided into a virtual voxel grid. Each voxel independently
   tracks its deepest contact, providing spatial coverage and preventing
   sudden contact jumps when different mesh regions become deepest.

All normal-bin configurations balance clearance from bin boundaries across
the coordinate axes, treating X-, Y-, and Z-up scenes equally. This stabilizes
nearly axis-aligned ground and wall contacts. Binning remains memoryless:
contacts near other bin boundaries can still change groups as their normals move.

See Also:
    :class:`GlobalContactReducer` in ``contact_reduction_global.py`` for the
    hashtable-based approach used for mesh-mesh (SDF) collisions.
"""

import warp as wp

# =====================================================================
# Contact Reduction Configuration
# =====================================================================
# Polyhedron for normal binning.  Determines NUM_NORMAL_BINS.
#   "icosahedron"  -> 20 bins  (default, finer normal resolution)
#   "dodecahedron" -> 12 bins  (good balance)
#   "octahedron"   ->  8 bins  (cheaper, coarser)
#   "hexahedron"   ->  6 bins  (cheapest, coarsest)
#
# NOTE: Only the default "icosahedron" configuration is currently tested
# on CI. Other polyhedra are functional but should be considered
# experimental. Use at your own discretion.
NORMAL_BINNING_POLYHEDRON = "icosahedron"

# Scan directions per normal bin (2D extremes on each face plane).
# Range 3-6. More directions = more accurate convex hull but more slots.
NUM_SPATIAL_DIRECTIONS = 6

# Voxel-based depth slots for spatial coverage.
NUM_VOXEL_DEPTH_SLOTS = 100
# =====================================================================

# Hard architectural limit — keeps per-pair indices representable in 8 bits.
MAX_CONTACTS_PER_PAIR = 255

# ---------------------------------------------------------------------------
# Derived constants (do not edit — computed from the config above)
# ---------------------------------------------------------------------------
_POLYHEDRON_BINS = {
    "hexahedron": 6,
    "octahedron": 8,
    "dodecahedron": 12,
    "icosahedron": 20,
}

NUM_NORMAL_BINS = _POLYHEDRON_BINS[NORMAL_BINNING_POLYHEDRON]

_total_slots = NUM_NORMAL_BINS * (NUM_SPATIAL_DIRECTIONS + 1) + NUM_VOXEL_DEPTH_SLOTS

assert _total_slots <= MAX_CONTACTS_PER_PAIR, (
    f"Total reduction slots ({_total_slots}) exceed MAX_CONTACTS_PER_PAIR "
    f"({MAX_CONTACTS_PER_PAIR}). Reduce NUM_SPATIAL_DIRECTIONS, "
    f"NUM_VOXEL_DEPTH_SLOTS, or switch to a coarser polyhedron."
)


# http://stereopsis.com/radix.html
@wp.func_native("""
uint32_t i = reinterpret_cast<uint32_t&>(f);
uint32_t mask = (uint32_t)(-(int)(i >> 31)) | 0x80000000;
return i ^ mask;
""")
def float_flip(f: float) -> wp.uint32: ...


# =====================================================================
# Polyhedron face normals
# =====================================================================
# Each polyhedron is stored as a flat tuple of (x, y, z) triples.
# Only the selected polyhedron is compiled into a Warp matrix constant.

# fmt: off
_HEXAHEDRON_NORMALS = (
    1.0,  0.0,  0.0,
   -1.0,  0.0,  0.0,
    0.0,  1.0,  0.0,
    0.0, -1.0,  0.0,
    0.0,  0.0,  1.0,
    0.0,  0.0, -1.0,
)

# Rotate the cube-vertex normals 60 degrees around (1, 1, 1).
# Every coordinate axis then has 19.4712 degrees of clearance from a bin boundary.
_OCTAHEDRON_NORMALS = (
     0.57735027,  0.57735027,  0.57735027,
    -0.19245009,  0.96225045, -0.19245009,
    -0.19245009, -0.19245009,  0.96225045,
    -0.96225045,  0.19245009,  0.19245009,
     0.96225045, -0.19245009, -0.19245009,
     0.19245009,  0.19245009, -0.96225045,
     0.19245009, -0.96225045,  0.19245009,
    -0.57735027, -0.57735027, -0.57735027,
)

# Rotate the canonical icosahedral vertices 23.4309116 degrees around
# (1, 1, 1), giving every coordinate axis 14.8960 degrees of bin clearance.
# Preserve the original top/equatorial/bottom ordering for the pruned search;
# its cap axis is the original +Y axis after the same rotation.
_DODECAHEDRON_NORMALS = (
     0.32491969,  0.93903535,  0.11242688,
    -0.66874031,  0.66874030,  0.32491970,
     0.11242688,  0.32491969,  0.93903535,
    -0.32491970,  0.66874031, -0.66874030,
     0.93903535,  0.11242688,  0.32491969,
     0.66874030,  0.32491970, -0.66874031,
    -0.66874030, -0.32491970,  0.66874031,
    -0.93903535, -0.11242688, -0.32491969,
     0.32491970, -0.66874031,  0.66874030,
    -0.11242688, -0.32491969, -0.93903535,
     0.66874031, -0.66874030, -0.32491970,
    -0.32491969, -0.93903535, -0.11242688,
)

# Align the original face normals with the dual of the canonical icosahedron,
# then rotate 97.7612433 degrees around (1, 1, 1). All coordinate axes have
# 14.0816 degrees of bin clearance. Preserve the original cap/belt ordering
# for the pruned search, whose cap axis rotates with the face normals.
_ICOSAHEDRON_NORMALS = (
     0.80103071,  0.29640806, -0.52008851,
     0.29640808,  0.80103070, -0.52008851,
    -0.29640806,  0.52008851, -0.80103071,
    -0.15816595, -0.15816596, -0.97466254,
     0.52008851, -0.29640808, -0.80103070,
     0.80103070, -0.52008851,  0.29640808,
     0.52008851, -0.80103071, -0.29640806,
     0.57735027,  0.57735027,  0.57735027,
     0.97466254,  0.15816595,  0.15816596,
    -0.52008851,  0.80103071,  0.29640806,
     0.15816596,  0.97466254,  0.15816595,
    -0.97466254, -0.15816595, -0.15816596,
    -0.80103070,  0.52008851, -0.29640808,
    -0.15816596, -0.97466254, -0.15816595,
    -0.57735027, -0.57735027, -0.57735027,
     0.15816595,  0.15816596,  0.97466254,
    -0.52008851,  0.29640808,  0.80103070,
    -0.80103071, -0.29640806,  0.52008851,
    -0.29640808, -0.80103070,  0.52008851,
     0.29640806, -0.52008851,  0.80103071,
)
# fmt: on

_POLYHEDRON_NORMALS_DATA = {
    "hexahedron": _HEXAHEDRON_NORMALS,
    "octahedron": _OCTAHEDRON_NORMALS,
    "dodecahedron": _DODECAHEDRON_NORMALS,
    "icosahedron": _ICOSAHEDRON_NORMALS,
}

_face_normals_mat_type = wp.types.matrix(shape=(NUM_NORMAL_BINS, 3), dtype=wp.float32)
FACE_NORMALS = _face_normals_mat_type(*_POLYHEDRON_NORMALS_DATA[NORMAL_BINNING_POLYHEDRON])

# Backward-compatible alias used by tests.
DODECAHEDRON_FACE_NORMALS = (
    _face_normals_mat_type(*_DODECAHEDRON_NORMALS)
    if NORMAL_BINNING_POLYHEDRON == "dodecahedron"
    else wp.types.matrix(shape=(12, 3), dtype=wp.float32)(*_DODECAHEDRON_NORMALS)
)


@wp.func
def get_face_normal(face_idx: int) -> wp.vec3:
    """Return a face normal without materializing the full icosahedron matrix."""
    if wp.static(NORMAL_BINNING_POLYHEDRON != "icosahedron"):
        return FACE_NORMALS[face_idx]
    if face_idx == 0:
        return wp.vec3(0.80103071, 0.29640806, -0.52008851)
    if face_idx == 1:
        return wp.vec3(0.29640808, 0.80103070, -0.52008851)
    if face_idx == 2:
        return wp.vec3(-0.29640806, 0.52008851, -0.80103071)
    if face_idx == 3:
        return wp.vec3(-0.15816595, -0.15816596, -0.97466254)
    if face_idx == 4:
        return wp.vec3(0.52008851, -0.29640808, -0.80103070)
    if face_idx == 5:
        return wp.vec3(0.80103070, -0.52008851, 0.29640808)
    if face_idx == 6:
        return wp.vec3(0.52008851, -0.80103071, -0.29640806)
    if face_idx == 7:
        return wp.vec3(0.57735027, 0.57735027, 0.57735027)
    if face_idx == 8:
        return wp.vec3(0.97466254, 0.15816595, 0.15816596)
    if face_idx == 9:
        return wp.vec3(-0.52008851, 0.80103071, 0.29640806)
    if face_idx == 10:
        return wp.vec3(0.15816596, 0.97466254, 0.15816595)
    if face_idx == 11:
        return wp.vec3(-0.97466254, -0.15816595, -0.15816596)
    if face_idx == 12:
        return wp.vec3(-0.80103070, 0.52008851, -0.29640808)
    if face_idx == 13:
        return wp.vec3(-0.15816596, -0.97466254, -0.15816595)
    if face_idx == 14:
        return wp.vec3(-0.57735027, -0.57735027, -0.57735027)
    if face_idx == 15:
        return wp.vec3(0.15816595, 0.15816596, 0.97466254)
    if face_idx == 16:
        return wp.vec3(-0.52008851, 0.29640808, 0.80103070)
    if face_idx == 17:
        return wp.vec3(-0.80103071, -0.29640806, 0.52008851)
    if face_idx == 18:
        return wp.vec3(-0.29640808, -0.80103070, 0.52008851)
    return wp.vec3(0.29640806, -0.52008851, 0.80103071)


@wp.func
def get_slot(normal: wp.vec3) -> int:
    """Return the normal-bin index whose face normal best matches *normal*.

    Each polyhedron has a compile-time specialization selected via
    ``NORMAL_BINNING_POLYHEDRON``:

    * **hexahedron** (6 faces) — O(1) axis-aligned comparison, no dot products.
    * **dodecahedron** (12 faces) — rotated-cap-axis range pruning over
      top / equatorial / bottom groups (4-8 dot products).
    * **icosahedron** (20 faces) — rotated-cap-axis range pruning over
      top cap / equatorial belt / bottom cap (5-15 dot products).
    * **octahedron** and any other polyhedron — full linear scan.

    Args:
        normal: Normal vector to match.

    Returns:
        Index of the best matching face in ``FACE_NORMALS``.
    """
    if wp.static(NORMAL_BINNING_POLYHEDRON == "hexahedron"):
        # Faces are axis-aligned: 0=+X, 1=-X, 2=+Y, 3=-Y, 4=+Z, 5=-Z.
        ax = wp.abs(normal[0])
        ay = wp.abs(normal[1])
        az = wp.abs(normal[2])
        if ax >= ay and ax >= az:
            if normal[0] >= 0.0:
                return 0
            else:
                return 1
        elif ay >= az:
            if normal[1] >= 0.0:
                return 2
            else:
                return 3
        else:
            if normal[2] >= 0.0:
                return 4
            else:
                return 5

    elif wp.static(NORMAL_BINNING_POLYHEDRON == "dodecahedron"):
        up_dot = wp.dot(normal, wp.vec3(-0.20209268, 0.94502682, 0.25706587))

        # Conservative thresholds: only skip regions when clearly in a polar cap.
        # Face layout: 0-3 = top group, 4-7 = equatorial, 8-11 = bottom group.
        if up_dot > 0.65:
            start_idx = 0
            end_idx = 4
        elif up_dot < -0.65:
            start_idx = 8
            end_idx = 12
        elif up_dot >= 0.0:
            start_idx = 0
            end_idx = 8
        else:
            start_idx = 4
            end_idx = 12

        best_slot = start_idx
        max_dot = wp.dot(normal, FACE_NORMALS[start_idx])

        for i in range(start_idx + 1, end_idx):
            d = wp.dot(normal, FACE_NORMALS[i])
            if d > max_dot:
                max_dot = d
                best_slot = i

        return best_slot

    elif wp.static(NORMAL_BINNING_POLYHEDRON == "icosahedron"):
        # Dot with the rotated cap axis, not world Y. Rotation preserves the
        # original conservative cap thresholds and candidate ranges.
        up_dot = wp.dot(normal, wp.vec3(0.29269405, 0.29269407, -0.91030784))

        # Keep each candidate range compile-time constant so Warp embeds only the selected normals.
        if up_dot > 0.65:
            best_slot = 0
            max_dot = wp.dot(normal, FACE_NORMALS[0])
            for i in range(1, 5):
                d = wp.dot(normal, FACE_NORMALS[i])
                if d > max_dot:
                    max_dot = d
                    best_slot = i
            return best_slot
        elif up_dot < -0.65:
            best_slot = 15
            max_dot = wp.dot(normal, FACE_NORMALS[15])
            for i in range(16, 20):
                d = wp.dot(normal, FACE_NORMALS[i])
                if d > max_dot:
                    max_dot = d
                    best_slot = i
            return best_slot
        elif up_dot >= 0.0:
            best_slot = 0
            max_dot = wp.dot(normal, FACE_NORMALS[0])
            for i in range(1, 15):
                d = wp.dot(normal, FACE_NORMALS[i])
                if d > max_dot:
                    max_dot = d
                    best_slot = i
            return best_slot
        else:
            best_slot = 5
            max_dot = wp.dot(normal, FACE_NORMALS[5])
            for i in range(6, 20):
                d = wp.dot(normal, FACE_NORMALS[i])
                if d > max_dot:
                    max_dot = d
                    best_slot = i
            return best_slot

    else:
        best_slot = 0
        max_dot = wp.dot(normal, FACE_NORMALS[0])
        for i in range(1, wp.static(NUM_NORMAL_BINS)):
            d = wp.dot(normal, FACE_NORMALS[i])
            if d > max_dot:
                max_dot = d
                best_slot = i
        return best_slot


@wp.func
def project_point_to_plane(bin_normal_idx: wp.int32, point: wp.vec3) -> wp.vec2:
    """Project a 3D point onto the 2D plane of a normal-bin face.

    Creates a local 2D coordinate system on the face plane using the face
    normal and constructs orthonormal basis vectors u and v.

    Args:
        bin_normal_idx: Index of the face in ``FACE_NORMALS``.
        point: 3D point to project.

    Returns:
        2D coordinates of the point in the face's local coordinate system.
    """
    face_normal = get_face_normal(bin_normal_idx)

    if wp.abs(face_normal[1]) < 0.9:
        ref = wp.vec3(0.0, 1.0, 0.0)
    else:
        ref = wp.vec3(1.0, 0.0, 0.0)

    u = wp.normalize(ref - wp.dot(ref, face_normal) * face_normal)
    v = wp.cross(face_normal, u)

    return wp.vec2(wp.dot(point, u), wp.dot(point, v))


@wp.func
def get_spatial_direction_2d(dir_idx: int) -> wp.vec2:
    """Get evenly-spaced 2D direction for spatial binning.

    Args:
        dir_idx: Direction index in the range 0..NUM_SPATIAL_DIRECTIONS-1.

    Returns:
        Unit 2D vector at angle ``dir_idx * 2pi / NUM_SPATIAL_DIRECTIONS``.
    """
    angle = float(dir_idx) * (2.0 * wp.pi / float(wp.static(NUM_SPATIAL_DIRECTIONS)))
    return wp.vec2(wp.cos(angle), wp.sin(angle))


def compute_num_reduction_slots() -> int:
    """Total reduction slots per shape pair.

    Returns:
        ``NUM_NORMAL_BINS * (NUM_SPATIAL_DIRECTIONS + 1) + NUM_VOXEL_DEPTH_SLOTS``,
        guaranteed to be at most ``MAX_CONTACTS_PER_PAIR`` (255).
    """
    return NUM_NORMAL_BINS * (NUM_SPATIAL_DIRECTIONS + 1) + NUM_VOXEL_DEPTH_SLOTS


@wp.func
def compute_voxel_index(
    pos_local: wp.vec3,
    aabb_lower: wp.vec3,
    aabb_upper: wp.vec3,
    resolution: wp.vec3i,
) -> int:
    """Compute voxel index for a position in local space.

    Args:
        pos_local: Position in mesh local space
        aabb_lower: Local AABB lower bound
        aabb_upper: Local AABB upper bound
        resolution: Voxel grid resolution (nx, ny, nz)

    Returns:
        Linear voxel index in [0, nx*ny*nz)
    """
    size = aabb_upper - aabb_lower
    # Normalize position to [0, 1]
    rel = wp.vec3(0.0, 0.0, 0.0)
    if size[0] > 1e-6:
        rel = wp.vec3((pos_local[0] - aabb_lower[0]) / size[0], rel[1], rel[2])
    if size[1] > 1e-6:
        rel = wp.vec3(rel[0], (pos_local[1] - aabb_lower[1]) / size[1], rel[2])
    if size[2] > 1e-6:
        rel = wp.vec3(rel[0], rel[1], (pos_local[2] - aabb_lower[2]) / size[2])

    # Clamp to [0, 1) and map to voxel indices
    nx = resolution[0]
    ny = resolution[1]
    nz = resolution[2]

    vx = wp.clamp(int(rel[0] * float(nx)), 0, nx - 1)
    vy = wp.clamp(int(rel[1] * float(ny)), 0, ny - 1)
    vz = wp.clamp(int(rel[2] * float(nz)), 0, nz - 1)

    return vx + vy * nx + vz * nx * ny
