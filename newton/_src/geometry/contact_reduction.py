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

import numpy as np
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
    feature: wp.int32
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

_mat60x3 = wp.types.matrix(shape=(60, 3), dtype=wp.float32)

# Triangle vertices ordered: top cap (0-14), equatorial (15-44), bottom cap (45-59)
# Each face has 3 consecutive vertices. Layout matches ICOSAHEDRON_FACE_NORMALS.
ICOSAHEDRON_TRIANGLES = _mat60x3(
    # Top cap faces 0-4 (rows 0-14)
    0.0,
    1.0,
    0.0,
    0.27639318,
    0.4472136,
    0.85065085,
    0.8944273,
    0.44721365,
    0.0,
    0.0,
    1.0,
    0.0,
    -0.7236069,
    0.44721365,
    0.5257311,
    0.27639318,
    0.4472136,
    0.85065085,
    0.0,
    1.0,
    0.0,
    -0.72360677,
    0.4472136,
    -0.5257312,
    -0.7236069,
    0.44721365,
    0.5257311,
    0.0,
    1.0,
    0.0,
    0.27639332,
    0.4472136,
    -0.8506508,
    -0.72360677,
    0.4472136,
    -0.5257312,
    0.0,
    1.0,
    0.0,
    0.8944273,
    0.44721365,
    0.0,
    0.27639332,
    0.4472136,
    -0.8506508,
    # Equatorial band faces 5-14 (rows 15-44)
    0.8944273,
    0.44721365,
    0.0,
    0.7236068,
    -0.4472136,
    0.5257311,
    0.72360677,
    -0.4472136,
    -0.52573115,
    0.8944273,
    0.44721365,
    0.0,
    0.72360677,
    -0.4472136,
    -0.52573115,
    0.27639332,
    0.4472136,
    -0.8506508,
    0.27639318,
    0.4472136,
    0.85065085,
    -0.27639323,
    -0.4472136,
    0.8506508,
    0.7236068,
    -0.4472136,
    0.5257311,
    0.27639318,
    0.4472136,
    0.85065085,
    0.7236068,
    -0.4472136,
    0.5257311,
    0.8944273,
    0.44721365,
    0.0,
    -0.7236069,
    0.44721365,
    0.5257311,
    -0.8944273,
    -0.44721365,
    0.0,
    -0.27639323,
    -0.4472136,
    0.8506508,
    -0.7236069,
    0.44721365,
    0.5257311,
    -0.27639323,
    -0.4472136,
    0.8506508,
    0.27639318,
    0.4472136,
    0.85065085,
    -0.72360677,
    0.4472136,
    -0.5257312,
    -0.2763933,
    -0.4472136,
    -0.8506508,
    -0.8944273,
    -0.44721365,
    0.0,
    -0.72360677,
    0.4472136,
    -0.5257312,
    -0.8944273,
    -0.44721365,
    0.0,
    -0.7236069,
    0.44721365,
    0.5257311,
    0.27639332,
    0.4472136,
    -0.8506508,
    0.72360677,
    -0.4472136,
    -0.52573115,
    -0.2763933,
    -0.4472136,
    -0.8506508,
    0.27639332,
    0.4472136,
    -0.8506508,
    -0.2763933,
    -0.4472136,
    -0.8506508,
    -0.72360677,
    0.4472136,
    -0.5257312,
    # Bottom cap faces 15-19 (rows 45-59)
    0.0,
    -1.0,
    0.0,
    0.7236068,
    -0.4472136,
    0.5257311,
    -0.27639323,
    -0.4472136,
    0.8506508,
    0.0,
    -1.0,
    0.0,
    -0.27639323,
    -0.4472136,
    0.8506508,
    -0.8944273,
    -0.44721365,
    0.0,
    0.0,
    -1.0,
    0.0,
    -0.8944273,
    -0.44721365,
    0.0,
    -0.2763933,
    -0.4472136,
    -0.8506508,
    0.0,
    -1.0,
    0.0,
    -0.2763933,
    -0.4472136,
    -0.8506508,
    0.72360677,
    -0.4472136,
    -0.52573115,
    0.0,
    -1.0,
    0.0,
    0.72360677,
    -0.4472136,
    -0.52573115,
    0.7236068,
    -0.4472136,
    0.5257311,
)


@wp.func
def get_scan_dir(icosahedron_face_id: int, i: int) -> wp.vec3:
    """Get scan direction for contact reduction.

    Args:
        icosahedron_face_id: ID of the icosahedron face
        i: Index in the range 0...5

    Returns:
        Edge of the triangle (not normalized).
        Indices 3, 4, 5 return the edges with negated direction.
    """
    face_base = 3 * icosahedron_face_id
    v0 = ICOSAHEDRON_TRIANGLES[face_base + (i + 1) % 3]
    v1 = ICOSAHEDRON_TRIANGLES[face_base + i % 3]
    result = v0 - v1
    if i >= 3:
        result = -result
    return result


@wp.func
def project_point_to_plane(bin_normal_idx: wp.int32, face_center: wp.vec3f) -> wp.vec2f:
    """Projects a 3D point onto the 2D plane defined by an icosahedron face normal.

    Args:
        bin_normal_idx: Index of the icosahedron face (0-19) defining the plane normal.
        face_center: 3D point to project.

    Returns:
        2D coordinates of the projected point in the plane's local basis.
    """
    bin_normal_dir = ICOSAHEDRON_FACE_NORMALS[bin_normal_idx]
    # Project face center to the plane corresponding to the bin normal
    temp_vec = wp.vec3f(0.0, 1.0, 0.0)
    if wp.abs(wp.dot(bin_normal_dir, temp_vec)) > 0.95:
        temp_vec = wp.vec3f(0.0, 0.0, 1.0)

    # Create orthogonal basis vectors in the plane
    plane_u = wp.normalize(wp.cross(bin_normal_dir, temp_vec))
    plane_v = wp.normalize(wp.cross(bin_normal_dir, plane_u))

    # Convert face_center to 2D coordinates in the plane basis
    face_center_2d = wp.vec2f(wp.dot(face_center, plane_u), wp.dot(face_center, plane_v))

    return face_center_2d


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
def get_slot_flat(normal: wp.vec3) -> int:
    """Returns the index of the icosahedron face that best matches the normal.

    Unlike get_slot, this function searches all bins without early termination.

    Args:
        normal: Normal vector to match

    Returns:
        Index of the best matching icosahedron face (0-19)
    """
    max_dot_product = wp.float32(-1e10)
    bin_normal_idx = wp.int32(-1)
    for n_idx in range(20):
        dp = wp.dot(normal, ICOSAHEDRON_FACE_NORMALS[n_idx])
        dp -= float(n_idx) * 1e-6  # for breaking ties in symmetric shapes
        if dp > max_dot_product:
            max_dot_product = dp
            bin_normal_idx = n_idx
    return bin_normal_idx


NUM_SPATIAL_DIRECTIONS = 6  # Triangle edge directions
NUM_NORMAL_BINS = 20  # Icosahedron faces


def compute_num_reduction_slots(num_betas: int) -> int:
    """Compute the number of reduction slots for a given number of beta values.

    Args:
        num_betas: Number of beta values

    Returns:
        Total number of reduction slots (20 bins * (6 directions * num_betas + 1))
        The +1 is for the max depth slot per bin.
    """
    return NUM_NORMAL_BINS * (NUM_SPATIAL_DIRECTIONS * num_betas + 1)


def create_betas_array(betas: tuple = (10.0,), device=None) -> wp.array:
    """Create a warp array with the beta values for contact reduction.

    Args:
        betas: Tuple of beta values (default: (10.0,))
        device: Device to create the array on (default: current device)

    Returns:
        wp.array of shape (num_betas,) with dtype float32
    """
    betas_np = np.array(betas, dtype=np.float32)
    return wp.array(betas_np, dtype=wp.float32, device=device)


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

    1. Contacts are binned by normal direction (20 bins based on icosahedron faces)
    2. Within each bin, contacts compete for slots using a scoring function
    3. Winners are determined via atomic operations in shared memory
    4. Duplicates (same geometric feature) are filtered out

    **Scoring Function:**

    Contacts are filtered by depth threshold, then compete with pure spatial score::

        if depth < beta:
            score = dot(position, scan_direction)

    Where:
        - ``position``: Contact point in centered world coordinates
        - ``scan_direction``: One of 6 spatial probing directions (±X, ±Y, ±Z)
        - ``depth``: Signed penetration depth (negative = penetrating)
        - ``beta``: Depth threshold controlling which contacts participate

    **Beta Parameter (Depth Threshold):**

    - ``beta = inf`` (or large value like 1000000): All contacts participate
    - ``beta = 0``: Only penetrating contacts (depth < 0) participate
    - ``beta = -0.01``: Only contacts with at least 1cm penetration participate

    Multiple betas can be specified to keep contacts at different depth thresholds.
    Each beta adds 6 slots per normal bin (one per spatial direction).

    Args:
        betas: Tuple of depth thresholds. Default ``(1000000.0, 0.0001)`` keeps both
               all spatial extremes and near-penetrating spatial extremes.
    """

    def __init__(self, betas: tuple = (1000000.0, 0.0001)):
        self.betas = betas
        self.num_betas = len(betas)
        self.num_reduction_slots = compute_num_reduction_slots(self.num_betas)

        # Shared memory pointers
        self.get_smem_slots_plus_1 = create_shared_memory_pointer_func_4_byte_aligned(self.num_reduction_slots + 1)
        self.get_smem_slots_contacts = create_shared_memory_pointer_func_4_byte_aligned(self.num_reduction_slots * 9)
        self.get_smem_reduction = create_shared_memory_pointer_func_8_byte_aligned(self.num_reduction_slots)

        # Warp functions
        self.store_reduced_contact = self._create_store_reduced_contact()
        self.filter_unique_contacts = self._create_filter_unique_contacts()

    def create_betas_array(self, device=None) -> wp.array:
        """Create a warp array with the beta values."""
        return create_betas_array(self.betas, device)

    def _create_store_reduced_contact(self):
        """Create the store_reduced_contact warp function.

        The returned function competes contacts for reduction slots using atomic max.
        Each thread with a contact computes scores for all (direction, beta) combinations
        and atomically competes for the corresponding slots. Winners write their contact
        to the shared buffer.
        """
        num_reduction_slots = self.num_reduction_slots
        num_betas = self.num_betas
        get_smem = self.get_smem_reduction

        @wp.func
        def store_reduced_contact(
            thread_id: int,
            active: bool,
            c: ContactStruct,
            buffer: wp.array(dtype=ContactStruct),
            active_ids: wp.array(dtype=int),
            betas_arr: wp.array(dtype=wp.float32),
            empty_marker: float,
        ):
            """Compete this thread's contact for reduction slots via atomic max.

            Args:
                thread_id: Thread index within the block
                active: Whether this thread has a valid contact to store
                c: Contact data (position, normal, depth, feature)
                buffer: Shared memory buffer for winning contacts
                active_ids: Tracks which slots contain valid contacts
                betas_arr: Array of depth thresholds (contact participates if depth < beta)
                empty_marker: Sentinel value indicating empty slots
            """
            slots_per_bin = wp.static(NUM_SPATIAL_DIRECTIONS * num_betas + 1)
            num_slots = wp.static(NUM_NORMAL_BINS * slots_per_bin)

            winner_slots = wp.array(
                ptr=wp.static(get_smem)(),
                shape=(wp.static(num_reduction_slots),),
                dtype=wp.uint64,
            )

            for i in range(thread_id, num_slots, wp.block_dim()):
                winner_slots[i] = wp.uint64(0)
            synchronize()

            bin_id = 0
            if active:
                bin_id = get_slot(c.normal)

            if active:
                base_key = bin_id * slots_per_bin
                # Compete for spatial direction + beta slots
                for dir_i in range(wp.static(NUM_SPATIAL_DIRECTIONS)):
                    scan_dir = get_scan_dir(bin_id, dir_i)
                    spatial_dp = wp.dot(scan_dir, c.position)
                    for beta_i in range(num_betas):
                        if c.depth < betas_arr[beta_i]:
                            score = spatial_dp  # - betas_arr[beta_i] * c.depth
                            key = base_key + dir_i * wp.static(num_betas) + beta_i
                            wp.atomic_max(winner_slots, key, pack_value_thread_id(score, thread_id))
                # Compete for max depth slot (last slot in bin)
                max_depth_key = base_key + slots_per_bin - 1
                # Use -depth as score so atomicMax selects the deepest (most negative depth)
                wp.atomic_max(winner_slots, max_depth_key, pack_value_thread_id(-c.depth, thread_id))
            synchronize()

            if active:
                base_key = bin_id * slots_per_bin
                # Check spatial direction + beta slots
                for dir_i in range(wp.static(NUM_SPATIAL_DIRECTIONS)):
                    scan_dir = get_scan_dir(bin_id, dir_i)
                    spatial_dp = wp.dot(scan_dir, c.position)
                    for beta_i in range(num_betas):
                        if c.depth < betas_arr[beta_i]:
                            key = base_key + dir_i * wp.static(num_betas) + beta_i
                            if unpack_thread_id(winner_slots[key]) == thread_id:
                                p = buffer[key].projection
                                if p == empty_marker:
                                    id = wp.atomic_add(active_ids, num_slots, 1)
                                    if id < num_slots:
                                        active_ids[id] = key
                                score = spatial_dp  # - betas_arr[beta_i] * c.depth
                                if score > p:
                                    c.projection = score
                                    buffer[key] = c
                # Check max depth slot
                max_depth_key = base_key + slots_per_bin - 1
                if unpack_thread_id(winner_slots[max_depth_key]) == thread_id:
                    p = buffer[max_depth_key].projection
                    if p == empty_marker:
                        id = wp.atomic_add(active_ids, num_slots, 1)
                        if id < num_slots:
                            active_ids[id] = max_depth_key
                    score = -c.depth
                    if score > p:
                        c.projection = score
                        buffer[max_depth_key] = c
            synchronize()

        return store_reduced_contact

    def _create_filter_unique_contacts(self):
        """Create the filter_unique_contacts warp function.

        The returned function removes duplicate contacts that won multiple slots
        but originate from the same geometric feature (e.g., same triangle).
        Only the first occurrence per feature is kept.
        """
        num_reduction_slots = self.num_reduction_slots
        num_betas = self.num_betas
        get_smem = self.get_smem_reduction

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
            slots_per_bin = wp.static(NUM_SPATIAL_DIRECTIONS * num_betas + 1)
            num_slots = wp.static(NUM_NORMAL_BINS * slots_per_bin)

            keep_flags = wp.array(
                ptr=wp.static(get_smem)(),
                shape=(wp.static(num_reduction_slots),),
                dtype=wp.int32,
            )

            for i in range(thread_id, num_slots, wp.block_dim()):
                keep_flags[i] = 0
            synchronize()

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

            if thread_id == 0:
                write_idx = int(0)
                for key in range(num_slots):
                    if keep_flags[key] == 1:
                        active_ids[write_idx] = key
                        write_idx += 1
                active_ids[num_slots] = write_idx
            synchronize()

        return filter_unique_contacts


get_shared_memory_pointer_block_dim_plus_2_ints = create_shared_memory_pointer_block_dim_func(2)
