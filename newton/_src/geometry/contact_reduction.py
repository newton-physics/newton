# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Contact reduction utilities for mesh collision.

This module provides constants, helper functions, and shared-memory utilities
used by the contact reduction system. The reduction selects a representative
subset (up to 240 contacts per pair) that preserves simulation stability.

**Contact Reduction Strategy Overview:**

When complex meshes collide, thousands of triangle pairs may generate contacts.
Contact reduction selects a representative subset that preserves simulation
stability while keeping memory and computation bounded.

The reduction uses three complementary strategies:

1. **Spatial Extreme Slots (120 total = 20 bins x 6 directions)**

   For each of 20 normal bins (icosahedron faces), finds the 6 most extreme
   contacts in 2D scan directions on the face plane. This builds the convex
   hull / support polygon boundary, critical for stable stacking.

2. **Per-Bin Max-Depth Slots (20 total = 20 bins x 1)**

   Each normal bin tracks its deepest contact unconditionally. This ensures
   deeply penetrating contacts from any normal direction are never dropped.
   Critical for gear-like contacts with varied normal orientations.

3. **Voxel-Based Depth Slots (100 total)**

   The mesh is divided into a virtual voxel grid. Each voxel independently
   tracks its deepest contact, providing spatial coverage and preventing
   sudden contact jumps when different mesh regions become deepest.

**Slot Calculation:**

::

    Per-bin slots:  20 bins x (6 spatial + 1 max-depth) = 140 slots
    Voxel slots:    100 slots
    Total:          240 slots per shape pair

See Also:
    :class:`GlobalContactReducer` in ``contact_reduction_global.py`` for the
    hashtable-based approach used for mesh-mesh (SDF) collisions.
"""

import warp as wp


# http://stereopsis.com/radix.html
@wp.func_native("""
uint32_t i = reinterpret_cast<uint32_t&>(f);
uint32_t mask = (uint32_t)(-(int)(i >> 31)) | 0x80000000;
return i ^ mask;
""")
def float_flip(f: float) -> wp.uint32: ...


@wp.func_native("""
#if defined(__CUDA_ARCH__)
__syncthreads();
#endif
""")
def synchronize(): ...


_mat20x3 = wp.types.matrix(shape=(20, 3), dtype=wp.float32)

# Face normals ordered: top cap (0-4), equatorial (5-14), bottom cap (15-19)
# This layout enables contiguous range searches for all cases.
ICOSAHEDRON_FACE_NORMALS = _mat20x3(
    # Top cap (faces 0-4, Y ≈ +0.795)
    0.49112338,
    0.79465455,
    0.35682216,
    -0.18759243,
    0.7946545,
    0.57735026,
    -0.6070619,
    0.7946545,
    0.0,
    -0.18759237,
    0.7946545,
    -0.57735026,
    0.4911234,
    0.79465455,
    -0.3568221,
    # Equatorial band (faces 5-14, Y ≈ ±0.188)
    0.9822469,
    -0.18759257,
    0.0,
    0.7946544,
    0.18759239,
    -0.5773503,
    0.30353096,
    -0.18759252,
    0.93417233,
    0.7946544,
    0.18759243,
    0.5773503,
    -0.7946545,
    -0.18759249,
    0.5773503,
    -0.30353105,
    0.18759243,
    0.9341724,
    -0.7946544,
    -0.1875924,
    -0.5773503,
    -0.9822469,
    0.18759254,
    0.0,
    0.30353096,
    -0.1875925,
    -0.93417233,
    -0.30353084,
    0.18759246,
    -0.9341724,
    # Bottom cap (faces 15-19, Y ≈ -0.795)
    0.18759249,
    -0.7946544,
    0.57735026,
    -0.49112338,
    -0.7946545,
    0.35682213,
    -0.49112338,
    -0.79465455,
    -0.35682213,
    0.18759243,
    -0.7946544,
    -0.57735026,
    0.607062,
    -0.7946544,
    0.0,
)


@wp.func
def get_slot(normal: wp.vec3) -> int:
    """Returns the index of the icosahedron face that best matches the normal.

    Uses Y-component to select search region:
    - Faces 0-4: top cap (Y ≈ +0.795)
    - Faces 5-14: equatorial band (Y ≈ ±0.188)
    - Faces 15-19: bottom cap (Y ≈ -0.795)

    Args:
        normal: Normal vector to match

    Returns:
        Index of the best matching icosahedron face (0-19)
    """
    up_dot = normal[1]

    # Conservative thresholds: only skip regions when clearly in a polar cap.
    # Top/bottom cap faces have Y ≈ ±0.795, equatorial faces have |Y| ≈ 0.188.
    # Threshold 0.65 ensures we don't miss better matches in adjacent regions.
    # Face layout: 0-4 = top cap, 5-14 = equatorial, 15-19 = bottom cap.
    if up_dot > 0.65:
        # Clearly pointing up - only check top cap (5 faces)
        start_idx = 0
        end_idx = 5
    elif up_dot < -0.65:
        # Clearly pointing down - only check bottom cap (5 faces)
        start_idx = 15
        end_idx = 20
    elif up_dot >= 0.0:
        # Leaning up - check top cap + equatorial (15 faces)
        start_idx = 0
        end_idx = 15
    else:
        # Leaning down - check equatorial + bottom cap (15 faces)
        start_idx = 5
        end_idx = 20

    best_slot = start_idx
    max_dot = wp.dot(normal, ICOSAHEDRON_FACE_NORMALS[start_idx])

    for i in range(start_idx + 1, end_idx):
        d = wp.dot(normal, ICOSAHEDRON_FACE_NORMALS[i])
        if d > max_dot:
            max_dot = d
            best_slot = i

    return best_slot


@wp.func
def project_point_to_plane(bin_normal_idx: wp.int32, point: wp.vec3) -> wp.vec2:
    """Project a 3D point onto the 2D plane of an icosahedron face.

    Creates a local 2D coordinate system on the face plane using the face normal
    and constructs orthonormal basis vectors u and v.

    Args:
        bin_normal_idx: Index of the icosahedron face (0-19)
        point: 3D point to project

    Returns:
        2D coordinates of the point in the face's local coordinate system
    """
    face_normal = ICOSAHEDRON_FACE_NORMALS[bin_normal_idx]

    # Create orthonormal basis on the plane
    # Choose reference vector that's not parallel to normal
    if wp.abs(face_normal[1]) < 0.9:
        ref = wp.vec3(0.0, 1.0, 0.0)
    else:
        ref = wp.vec3(1.0, 0.0, 0.0)

    # u = normalize(ref - dot(ref, normal) * normal)
    u = wp.normalize(ref - wp.dot(ref, face_normal) * face_normal)
    # v = cross(normal, u)
    v = wp.cross(face_normal, u)

    # Project point onto u and v axes
    return wp.vec2(wp.dot(point, u), wp.dot(point, v))


@wp.func
def get_spatial_direction_2d(dir_idx: int) -> wp.vec2:
    """Get evenly-spaced 2D direction for spatial binning.

    Args:
        dir_idx: Direction index in the range 0..NUM_SPATIAL_DIRECTIONS-1

    Returns:
        Unit 2D vector at angle (dir_idx * 2pi / NUM_SPATIAL_DIRECTIONS)
    """
    angle = float(dir_idx) * (2.0 * wp.pi / 6.0)
    return wp.vec2(wp.cos(angle), wp.sin(angle))


NUM_SPATIAL_DIRECTIONS = 6  # Evenly-spaced 2D directions (60 degrees apart)
NUM_NORMAL_BINS = 20  # Icosahedron faces
NUM_VOXEL_DEPTH_SLOTS = 100  # Voxel-based depth slots for spatial coverage


def compute_num_reduction_slots() -> int:
    """Compute the number of reduction slots.

    Returns:
        Total number of reduction slots:
        - 20 normal bins * (6 spatial directions + 1 max-depth) (per-bin slots)
        - + 100 voxel-based depth slots (deepest contact per voxel region)
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


def create_shared_memory_pointer_block_dim_func(
    add: int,
):
    """Create a shared memory pointer function for a block-dimension-dependent array size.

    Args:
        add: Number of additional int elements beyond WP_TILE_BLOCK_DIM.

    Returns:
        A Warp function that returns a pointer to shared memory
    """

    snippet = f"""
#if defined(__CUDA_ARCH__)
    constexpr int array_size = WP_TILE_BLOCK_DIM +{add};
    __shared__ int s[array_size];
    auto ptr = &s[0];
    return (uint64_t)ptr;
#else
    return (uint64_t)0;
#endif
    """

    @wp.func_native(snippet)
    def get_shared_memory_pointer() -> wp.uint64: ...

    return get_shared_memory_pointer


def create_shared_memory_pointer_block_dim_mul_func(
    mul: int,
):
    """Create a shared memory pointer whose size scales with the block dimension.

    Allocates ``WP_TILE_BLOCK_DIM * mul`` int32 elements of shared memory.

    Args:
        mul: Multiplier applied to WP_TILE_BLOCK_DIM.

    Returns:
        A Warp function that returns a pointer to shared memory
    """

    snippet = f"""
#if defined(__CUDA_ARCH__)
    constexpr int array_size = WP_TILE_BLOCK_DIM * {mul};
    __shared__ int s[array_size];
    auto ptr = &s[0];
    return (uint64_t)ptr;
#else
    return (uint64_t)0;
#endif
    """

    @wp.func_native(snippet)
    def get_shared_memory_pointer() -> wp.uint64: ...

    return get_shared_memory_pointer


get_shared_memory_pointer_block_dim_plus_2_ints = create_shared_memory_pointer_block_dim_func(2)
