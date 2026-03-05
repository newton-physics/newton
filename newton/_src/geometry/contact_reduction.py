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

"""Contact reduction for mesh collision using GPU shared memory.

This module provides efficient contact reduction for mesh-plane and mesh-convex
collisions where all contacts for a shape pair can be processed within a single
GPU thread block using shared memory.

**Contact Reduction Strategy Overview:**

When complex meshes collide, thousands of triangle pairs may generate contacts.
Contact reduction selects a representative subset (up to 240 contacts per pair)
that preserves simulation stability while keeping memory and computation bounded.

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

**Usage:**

- This shared-memory approach is used for mesh-plane and mesh-convex contacts
- For mesh-mesh (SDF) collisions, see ``contact_reduction_global.py`` which
  uses a hashtable-based approach for contacts spanning multiple GPU blocks

See Also:
    :class:`ContactReductionFunctions` for the main API and detailed documentation.
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


@wp.func
def pack_value_thread_id(value: float, thread_id: int) -> wp.uint64:
    """Pack float value and thread_id into uint64 for atomic argmax.

    High 32 bits: float_flip(value) - makes floats comparable as unsigned ints
    Low 32 bits: thread_id - for deterministic tie-breaking

    atomicMax on this packed value will select:
    1. The thread with the highest value
    2. For equal values, the thread with the highest thread_id (deterministic)
    """
    return (wp.uint64(float_flip(value)) << wp.uint64(32)) | wp.uint64(thread_id)


# Use native func because warp tries to convert 0xFFFFFFFF to int32 which is not the intended behavior
@wp.func_native("""
return static_cast<int32_t>(packed & 0xFFFFFFFFull);
""")
def unpack_thread_id(packed: wp.uint64) -> int: ...


@wp.struct
class ContactStruct:
    position: wp.vec3
    normal: wp.vec3
    depth: wp.float32
    feature: wp.int32  # Feature ID for deduplication (e.g., triangle index)
    projection: wp.float32


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


def create_shared_memory_pointer_func_4_byte_aligned(
    array_size: int,
):
    """Create a shared memory pointer function for a specific array size.

    Args:
        array_size: Number of int elements in the shared memory array.

    Returns:
        A Warp function that returns a pointer to shared memory
    """

    snippet = f"""
#if defined(__CUDA_ARCH__)
    constexpr int array_size = {array_size};
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


def create_shared_memory_pointer_func_8_byte_aligned(
    array_size: int,
):
    """Create a shared memory pointer function for a specific array size.

    Args:
        array_size: Number of int elements in the shared memory array.

    Returns:
        A Warp function that returns a pointer to shared memory
    """

    snippet = f"""
#if defined(__CUDA_ARCH__)
    constexpr int array_size = {array_size};
    __shared__ uint64_t s[array_size];
    auto ptr = &s[0];
    return (uint64_t)ptr;
#else
    return (uint64_t)0;
#endif
    """

    @wp.func_native(snippet)
    def get_shared_memory_pointer() -> wp.uint64: ...

    return get_shared_memory_pointer


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


class ContactReductionFunctions:
    """Reduces many candidate contacts to a representative subset using GPU shared memory.

    When colliding complex meshes, thousands of triangle pairs may generate contacts.
    This class provides functions to efficiently reduce them to a manageable set while
    preserving contacts that are important for stable simulation.

    **Algorithm Overview:**

    The reduction uses three complementary strategies to select representative contacts:

    1. **Spatial Extreme Slots** - Find the support polygon boundary per normal direction
    2. **Per-Bin Max-Depth Slots** - Track deepest contact per normal direction
    3. **Voxel-Based Depth Slots** - Track deepest contact per mesh region

    Winners are determined via atomic operations in GPU shared memory. Contacts that
    win multiple slots are deduplicated before output.

    **Slot Layout (240 total slots per shape pair):**

    ::

        Per-bin slots:  20 bins x (6 spatial + 1 max-depth) = 140 slots
        Voxel slots:    100 slots
        Total:          240 slots

    **1. Spatial Extreme Slots (6 per normal bin = 120 total):**

    For contacts with depth < beta (near-penetrating), finds the most extreme contact
    in each of 6 evenly-spaced 2D scan directions (60° apart) on the icosahedron face
    plane. This builds the convex hull / support polygon of the contact patch, which
    is critical for stable stacking and preventing objects from tipping.

    ::

        if depth < beta:
            score = dot(projected_position_2d, scan_direction)  # higher wins

    **2. Per-Bin Max-Depth Slots (1 per normal bin = 20 total):**

    Each normal bin unconditionally tracks its deepest contact (most negative depth).
    This ensures the most penetrated contact per normal direction is always kept,
    regardless of the beta threshold. Critical for:

    - **Gear-like contacts** where tooth surfaces have different normal orientations
    - **Stable response** to deeply penetrating contacts from any direction

    ::

        score = -depth  # most negative depth wins (deepest penetration)

    **3. Voxel-Based Depth Slots (100 total):**

    The mesh is divided into a virtual voxel grid (up to 100 voxels based on local
    AABB). Each voxel independently tracks its deepest contact, providing spatial
    coverage across the mesh surface. This prevents:

    - **Sudden contact jumps** when different mesh regions become deepest
    - **Late contact detection** at mesh centers (contacts show up early, not just
      after they become the global deepest)

    ::

        score = -depth  # most negative depth wins per voxel

    **Why This Hybrid Approach:**

    - **Spatial extremes alone** miss deeply penetrating contacts at the center
    - **Max-depth alone** misses the support polygon boundary needed for stability
    - **Voxels alone** don't capture normal-direction information for angled contacts
    - **Combined** they provide robust contact selection for diverse scenarios:
      flat stacking, gear meshing, screw threading, tilting objects, etc.

    **Depth Threshold (Beta):**

    Contacts with depth < 0.0001m (0.1mm) participate in spatial extreme competition.
    This small positive threshold avoids contact flickering due to numerical noise
    while effectively selecting only near-penetrating contacts for the support polygon.

    The overall contact detection range is controlled by the contact margin parameter
    on shapes, not by the reduction system.

    See Also:
        :class:`GlobalContactReducer` in ``contact_reduction_global.py`` for the
        hashtable-based variant used for mesh-mesh (SDF) collisions.
    """

    BETA_THRESHOLD = 0.0001
    """Penetration depth threshold for spatial extreme slot competition.

    Only contacts with depth below this value (i.e., penetrating or near-touching)
    compete for spatial extreme slots that build the support polygon boundary.
    A small positive value (rather than zero) accounts for numerical tolerances,
    preventing contact flickering when stacked objects have near-zero depths.
    """

    def __init__(self):
        self.reduction_slot_count = compute_num_reduction_slots()

        # Shared memory pointers
        self.get_smem_slots_plus_1 = create_shared_memory_pointer_func_4_byte_aligned(self.reduction_slot_count + 1)
        self.get_smem_slots_contacts = create_shared_memory_pointer_func_4_byte_aligned(self.reduction_slot_count * 9)
        self.get_smem_reduction = create_shared_memory_pointer_func_8_byte_aligned(self.reduction_slot_count)

        # Warp functions
        self.store_reduced_contact = self._create_store_reduced_contact()
        self.filter_unique_contacts = self._create_filter_unique_contacts()

    def _create_store_reduced_contact(self):
        """Create the store_reduced_contact warp function.

        The returned function competes contacts for reduction slots using atomic max.
        Each thread with a contact computes scores for all spatial directions
        and atomically competes for the corresponding slots. Additionally, contacts
        compete for voxel-based depth slots.

        Winners write their contact to the shared buffer.
        """
        NUM_REDUCTION_SLOTS = self.reduction_slot_count
        BETA = self.BETA_THRESHOLD
        get_smem = self.get_smem_reduction
        # Number of per-bin slots (6 spatial directions + 1 max-depth per bin)
        num_per_bin_slots = NUM_NORMAL_BINS * (NUM_SPATIAL_DIRECTIONS + 1)

        @wp.func
        def store_reduced_contact(
            thread_id: int,
            active: bool,
            c: ContactStruct,
            buffer: wp.array(dtype=ContactStruct),
            active_ids: wp.array(dtype=int),
            empty_marker: float,
            voxel_index: int,
        ):
            """Compete this thread's contact for reduction slots via atomic max.

            Args:
                thread_id: Thread index within the block
                active: Whether this thread has a valid contact to store
                c: Contact data (position, normal, depth, mode)
                buffer: Shared memory buffer for winning contacts
                active_ids: Tracks which slots contain valid contacts
                empty_marker: Sentinel value indicating empty slots
                voxel_index: Voxel index for the contact position (0 to NUM_VOXEL_DEPTH_SLOTS-1)
            """
            # Slot layout per bin: 6 spatial directions + 1 max-depth
            slots_per_bin = wp.static(NUM_SPATIAL_DIRECTIONS + 1)

            winner_slots = wp.array(
                ptr=wp.static(get_smem)(),
                shape=(NUM_REDUCTION_SLOTS,),
                dtype=wp.uint64,
            )

            for i in range(thread_id, NUM_REDUCTION_SLOTS, wp.block_dim()):
                winner_slots[i] = wp.uint64(0)
            synchronize()

            bin_id = 0
            pos_2d = wp.vec2(0.0, 0.0)
            if active:
                bin_id = get_slot(c.normal)
                # Project position to 2D plane once, reuse for all directions
                pos_2d = project_point_to_plane(bin_id, c.position)

            if active:
                base_key = bin_id * slots_per_bin
                # Compete for spatial direction slots (contacts with depth < beta)
                use_beta = c.depth < wp.static(BETA)
                for dir_i in range(wp.static(NUM_SPATIAL_DIRECTIONS)):
                    if use_beta:
                        dir_2d = get_spatial_direction_2d(dir_i)
                        score = wp.dot(pos_2d, dir_2d)
                        key = base_key + dir_i
                        wp.atomic_max(winner_slots, key, pack_value_thread_id(score, thread_id))

                # Compete for per-bin max-depth slot (last slot in bin)
                max_depth_key = base_key + slots_per_bin - 1
                # Use -depth as score so atomicMax selects the deepest (most negative depth)
                wp.atomic_max(winner_slots, max_depth_key, pack_value_thread_id(-c.depth, thread_id))

                # Compete for voxel-based depth slot
                # Voxel slots start after per-bin slots
                voxel_key = wp.static(num_per_bin_slots) + wp.clamp(
                    voxel_index, 0, wp.static(NUM_VOXEL_DEPTH_SLOTS - 1)
                )
                wp.atomic_max(winner_slots, voxel_key, pack_value_thread_id(-c.depth, thread_id))
            synchronize()

            if active:
                base_key = bin_id * slots_per_bin
                # Check spatial direction slots
                for dir_i in range(wp.static(NUM_SPATIAL_DIRECTIONS)):
                    if use_beta:
                        key = base_key + dir_i
                        if unpack_thread_id(winner_slots[key]) == thread_id:
                            p = buffer[key].projection
                            if p == empty_marker:
                                slot_id = wp.atomic_add(active_ids, NUM_REDUCTION_SLOTS, 1)
                                if slot_id < NUM_REDUCTION_SLOTS:
                                    active_ids[slot_id] = key
                            dir_2d = get_spatial_direction_2d(dir_i)
                            score = wp.dot(pos_2d, dir_2d)
                            if score > p:
                                c.projection = score
                                buffer[key] = c

                # Check per-bin max-depth slot
                max_depth_key = base_key + slots_per_bin - 1
                if unpack_thread_id(winner_slots[max_depth_key]) == thread_id:
                    p = buffer[max_depth_key].projection
                    if p == empty_marker:
                        slot_id = wp.atomic_add(active_ids, NUM_REDUCTION_SLOTS, 1)
                        if slot_id < NUM_REDUCTION_SLOTS:
                            active_ids[slot_id] = max_depth_key
                    score = -c.depth
                    if score > p:
                        c.projection = score
                        buffer[max_depth_key] = c

                # Check voxel depth slot
                voxel_key = wp.static(num_per_bin_slots) + wp.clamp(
                    voxel_index, 0, wp.static(NUM_VOXEL_DEPTH_SLOTS - 1)
                )
                if unpack_thread_id(winner_slots[voxel_key]) == thread_id:
                    p = buffer[voxel_key].projection
                    if p == empty_marker:
                        slot_id = wp.atomic_add(active_ids, NUM_REDUCTION_SLOTS, 1)
                        if slot_id < NUM_REDUCTION_SLOTS:
                            active_ids[slot_id] = voxel_key
                    score = -c.depth
                    if score > p:
                        c.projection = score
                        buffer[voxel_key] = c
            synchronize()

        return store_reduced_contact

    def _create_filter_unique_contacts(self):
        """Create the filter_unique_contacts warp function.

        The returned function removes duplicate contacts that won multiple slots
        but originate from the same geometric feature (e.g., same triangle).
        Only the first occurrence per feature is kept.
        """
        NUM_REDUCTION_SLOTS = self.reduction_slot_count
        get_smem = self.get_smem_reduction
        # Number of per-bin slots (6 spatial directions + 1 max-depth per bin)
        num_per_bin_slots = NUM_NORMAL_BINS * (NUM_SPATIAL_DIRECTIONS + 1)

        @wp.func
        def filter_unique_contacts(
            thread_id: int,
            buffer: wp.array(dtype=ContactStruct),
            active_ids: wp.array(dtype=int),
            empty_marker: float,
        ):
            """Remove duplicate contacts, keeping first occurrence per feature.

            Args:
                thread_id: Thread index within the block
                buffer: Shared memory buffer containing reduced contacts
                active_ids: Output array of unique contact slot indices
                empty_marker: Sentinel value indicating empty slots
            """
            # Slot layout per bin: 6 spatial directions + 1 max-depth
            slots_per_bin = wp.static(NUM_SPATIAL_DIRECTIONS + 1)

            keep_flags = wp.array(
                ptr=wp.static(get_smem)(),
                shape=(NUM_REDUCTION_SLOTS,),
                dtype=wp.int32,
            )

            for i in range(thread_id, NUM_REDUCTION_SLOTS, wp.block_dim()):
                keep_flags[i] = 0
            synchronize()

            # Phase 2a: Duplicate detection within each normal bin (20 threads active)
            # Each bin is processed by one thread to find unique contacts
            if thread_id < wp.static(NUM_NORMAL_BINS):
                bin_id = thread_id
                base_key = bin_id * slots_per_bin
                for slot_i in range(slots_per_bin):
                    key_i = base_key + slot_i
                    if buffer[key_i].projection > empty_marker:
                        feature_i = buffer[key_i].feature
                        is_dup = int(0)
                        for slot_j in range(slot_i):
                            key_j = base_key + slot_j
                            if buffer[key_j].projection > empty_marker and buffer[key_j].feature == feature_i:
                                is_dup = 1
                        if is_dup == 0:
                            keep_flags[key_i] = 1
            synchronize()

            # Phase 2b: Duplicate detection for voxel slots
            # We check if a voxel slot's feature already exists in per-bin slots or earlier voxel slots
            # Use strided parallel processing
            for voxel_slot in range(thread_id, wp.static(NUM_VOXEL_DEPTH_SLOTS), wp.block_dim()):
                voxel_key = wp.static(num_per_bin_slots) + voxel_slot
                if buffer[voxel_key].projection > empty_marker:
                    feature_v = buffer[voxel_key].feature
                    is_dup = int(0)

                    # Check against all per-bin slots (spatial extremes + max-depth)
                    for per_bin_key in range(wp.static(num_per_bin_slots)):
                        if buffer[per_bin_key].projection > empty_marker and buffer[per_bin_key].feature == feature_v:
                            is_dup = 1

                    # Check against earlier voxel slots
                    for earlier_voxel in range(voxel_slot):
                        earlier_key = wp.static(num_per_bin_slots) + earlier_voxel
                        if buffer[earlier_key].projection > empty_marker and buffer[earlier_key].feature == feature_v:
                            is_dup = 1

                    if is_dup == 0:
                        keep_flags[voxel_key] = 1

            # Reset counter for parallel compaction
            if thread_id == 0:
                active_ids[NUM_REDUCTION_SLOTS] = 0
            synchronize()

            # Phase 3: Parallel compaction - all threads participate
            # Each thread checks its subset of slots and uses atomic_add for write index
            for key in range(thread_id, NUM_REDUCTION_SLOTS, wp.block_dim()):
                if keep_flags[key] == 1:
                    write_idx = wp.atomic_add(active_ids, NUM_REDUCTION_SLOTS, 1)
                    active_ids[write_idx] = key
            synchronize()

        return filter_unique_contacts


get_shared_memory_pointer_block_dim_plus_2_ints = create_shared_memory_pointer_block_dim_func(2)
