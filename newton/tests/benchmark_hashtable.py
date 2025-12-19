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

"""Benchmark for the hash table performance.

This script measures hash table operations:
1. Insert performance (with varying collision rates)
2. Clear active performance
3. Memory bandwidth utilization
"""

import time
from dataclasses import dataclass

import numpy as np
import warp as wp

from newton._src.geometry.contact_reduction_global import reduction_insert_slot
from newton._src.geometry.hashtable import HashTable


@dataclass
class BenchmarkResult:
    """Result from a single benchmark."""

    name: str
    num_ops: int
    time_ms: float
    ops_per_second: float
    bandwidth_gb_s: float = 0.0


@wp.func
def make_key(shape_a: int, shape_b: int, bin_id: int) -> wp.uint64:
    """Create a hash table key from shape pair and bin."""
    key = wp.uint64(shape_a) & wp.uint64(0x1FFFFFFF)
    key = key | ((wp.uint64(shape_b) & wp.uint64(0x1FFFFFFF)) << wp.uint64(29))
    key = key | ((wp.uint64(bin_id) & wp.uint64(0x1F)) << wp.uint64(58))
    return key


@wp.func
def make_value(score: float, contact_id: int) -> wp.uint64:
    """Pack score and contact_id into a uint64 value.

    Uses a simple encoding where higher score = higher value.
    """
    # Simple encoding: score as high bits (scaled to int), contact_id as low bits
    # This gives correct ordering for positive scores
    score_int = wp.uint64(wp.max(0.0, score * 1000000.0))
    return (score_int << wp.uint64(32)) | wp.uint64(contact_id)


@wp.kernel
def insert_low_collision_kernel(
    num_insertions: int,
    keys: wp.array(dtype=wp.uint64),
    values: wp.array(dtype=wp.uint64),
    active_slots: wp.array(dtype=wp.int32),
):
    """Insert with low collision rate - each thread inserts to unique key."""
    tid = wp.tid()
    if tid >= num_insertions:
        return

    # Unique key per thread
    shape_a = tid % 1000
    shape_b = (tid // 1000) % 1000
    bin_id = (tid // 1000000) % 20

    key = make_key(shape_a, shape_b, bin_id)
    slot_id = tid % 13
    value = make_value(float(tid), tid)

    reduction_insert_slot(key, slot_id, value, keys, values, active_slots)


@wp.kernel
def insert_medium_collision_kernel(
    num_insertions: int,
    keys: wp.array(dtype=wp.uint64),
    values: wp.array(dtype=wp.uint64),
    active_slots: wp.array(dtype=wp.int32),
):
    """Insert with medium collision rate - groups of threads share keys."""
    tid = wp.tid()
    if tid >= num_insertions:
        return

    # ~10 threads per key on average
    group = tid // 10
    shape_a = group % 100
    shape_b = (group // 100) % 100
    bin_id = (group // 10000) % 20

    key = make_key(shape_a, shape_b, bin_id)
    slot_id = tid % 13
    value = make_value(float(tid), tid)

    reduction_insert_slot(key, slot_id, value, keys, values, active_slots)


@wp.kernel
def insert_high_collision_kernel(
    num_insertions: int,
    keys: wp.array(dtype=wp.uint64),
    values: wp.array(dtype=wp.uint64),
    active_slots: wp.array(dtype=wp.int32),
):
    """Insert with high collision rate - many threads compete for same keys."""
    tid = wp.tid()
    if tid >= num_insertions:
        return

    # ~100 threads per key on average
    group = tid // 100
    shape_a = group % 10
    shape_b = (group // 10) % 10
    bin_id = group % 20

    key = make_key(shape_a, shape_b, bin_id)
    slot_id = tid % 13
    value = make_value(float(tid), tid)

    reduction_insert_slot(key, slot_id, value, keys, values, active_slots)


@wp.kernel
def insert_extreme_collision_kernel(
    num_insertions: int,
    keys: wp.array(dtype=wp.uint64),
    values: wp.array(dtype=wp.uint64),
    active_slots: wp.array(dtype=wp.int32),
):
    """Insert with extreme collision - all threads write to same few keys."""
    tid = wp.tid()
    if tid >= num_insertions:
        return

    # Only 20 unique keys (one per bin)
    shape_a = 0
    shape_b = 1
    bin_id = tid % 20

    key = make_key(shape_a, shape_b, bin_id)
    slot_id = tid % 13
    value = make_value(float(tid), tid)

    reduction_insert_slot(key, slot_id, value, keys, values, active_slots)


def run_insert_benchmark(
    kernel,
    name: str,
    num_insertions: int,
    device: str = "cuda:0",
    num_iterations: int = 10,
) -> BenchmarkResult:
    """Run an insertion benchmark with the given kernel."""
    values_per_key = 13
    # Large capacity to minimize hash collisions
    capacity = max(num_insertions * 10, 1024)

    ht = HashTable(capacity, device=device)
    values = wp.zeros(ht.capacity * values_per_key, dtype=wp.uint64, device=device)

    # Warm up
    wp.launch(
        kernel,
        dim=num_insertions,
        inputs=[num_insertions, ht.keys, values, ht.active_slots],
        device=device,
    )
    wp.synchronize()
    ht.clear()
    values.zero_()
    wp.synchronize()

    # Benchmark
    times = []
    for _ in range(num_iterations):
        ht.clear()
        values.zero_()
        wp.synchronize()

        start = time.perf_counter()
        wp.launch(
            kernel,
            dim=num_insertions,
            inputs=[num_insertions, ht.keys, values, ht.active_slots],
            device=device,
        )
        wp.synchronize()
        times.append((time.perf_counter() - start) * 1000)

    avg_time = np.mean(times)
    ops_per_second = num_insertions / (avg_time / 1000)

    # Estimate bandwidth: each insert reads/writes key (8B) + value (8B) + active_slot (4B)
    bytes_per_op = 8 + 8 + 4
    bandwidth_gb_s = (num_insertions * bytes_per_op) / (avg_time / 1000) / 1e9

    return BenchmarkResult(
        name=name,
        num_ops=num_insertions,
        time_ms=avg_time,
        ops_per_second=ops_per_second,
        bandwidth_gb_s=bandwidth_gb_s,
    )


def run_clear_active_benchmark(
    num_active: int,
    device: str = "cuda:0",
    num_iterations: int = 10,
) -> BenchmarkResult:
    """Benchmark clear_active performance (keys only, not values)."""
    values_per_key = 13
    capacity = num_active * 2

    ht = HashTable(capacity, device=device)
    values = wp.zeros(ht.capacity * values_per_key, dtype=wp.uint64, device=device)

    # Fill with data using low collision kernel
    wp.launch(
        insert_low_collision_kernel,
        dim=num_active,
        inputs=[num_active, ht.keys, values, ht.active_slots],
        device=device,
    )
    wp.synchronize()

    # Benchmark clear_active
    times = []
    for _ in range(num_iterations):
        # Re-fill
        wp.launch(
            insert_low_collision_kernel,
            dim=num_active,
            inputs=[num_active, ht.keys, values, ht.active_slots],
            device=device,
        )
        wp.synchronize()

        start = time.perf_counter()
        ht.clear_active()
        wp.synchronize()
        times.append((time.perf_counter() - start) * 1000)

    avg_time = np.mean(times)
    ops_per_second = num_active / (avg_time / 1000) if avg_time > 0 else 0

    # Bandwidth: clearing key (8B) only - values are caller's responsibility now
    bytes_per_entry = 8
    bandwidth_gb_s = (num_active * bytes_per_entry) / (avg_time / 1000) / 1e9 if avg_time > 0 else 0

    return BenchmarkResult(
        name="clear_active",
        num_ops=num_active,
        time_ms=avg_time,
        ops_per_second=ops_per_second,
        bandwidth_gb_s=bandwidth_gb_s,
    )


def main():
    """Run the benchmark suite."""
    wp.init()

    device = "cuda:0"
    print("Hash Table Benchmark")
    print(f"Device: {device}")
    print("=" * 80)

    insertion_counts = [10_000, 100_000, 500_000, 1_000_000]

    # 1. Low collision benchmark
    print("\n1. LOW COLLISION (unique keys per thread)")
    print("-" * 80)
    print(f"{'Insertions':>12} {'Time (ms)':>12} {'Ops/sec':>15} {'BW (GB/s)':>12}")
    print("-" * 80)
    for n in insertion_counts:
        result = run_insert_benchmark(insert_low_collision_kernel, "low_collision", n, device)
        print(
            f"{result.num_ops:>12,} {result.time_ms:>12.3f} {result.ops_per_second:>15,.0f} {result.bandwidth_gb_s:>12.2f}"
        )

    # 2. Medium collision benchmark
    print("\n2. MEDIUM COLLISION (~10 threads per key)")
    print("-" * 80)
    print(f"{'Insertions':>12} {'Time (ms)':>12} {'Ops/sec':>15} {'BW (GB/s)':>12}")
    print("-" * 80)
    for n in insertion_counts:
        result = run_insert_benchmark(insert_medium_collision_kernel, "medium_collision", n, device)
        print(
            f"{result.num_ops:>12,} {result.time_ms:>12.3f} {result.ops_per_second:>15,.0f} {result.bandwidth_gb_s:>12.2f}"
        )

    # 3. High collision benchmark
    print("\n3. HIGH COLLISION (~100 threads per key)")
    print("-" * 80)
    print(f"{'Insertions':>12} {'Time (ms)':>12} {'Ops/sec':>15} {'BW (GB/s)':>12}")
    print("-" * 80)
    for n in insertion_counts:
        result = run_insert_benchmark(insert_high_collision_kernel, "high_collision", n, device)
        print(
            f"{result.num_ops:>12,} {result.time_ms:>12.3f} {result.ops_per_second:>15,.0f} {result.bandwidth_gb_s:>12.2f}"
        )

    # 4. Extreme collision benchmark
    print("\n4. EXTREME COLLISION (all threads -> 20 keys)")
    print("-" * 80)
    print(f"{'Insertions':>12} {'Time (ms)':>12} {'Ops/sec':>15} {'BW (GB/s)':>12}")
    print("-" * 80)
    for n in insertion_counts:
        result = run_insert_benchmark(insert_extreme_collision_kernel, "extreme_collision", n, device)
        print(
            f"{result.num_ops:>12,} {result.time_ms:>12.3f} {result.ops_per_second:>15,.0f} {result.bandwidth_gb_s:>12.2f}"
        )

    # 5. Clear active benchmark
    print("\n5. CLEAR ACTIVE (keys only)")
    print("-" * 80)
    print(f"{'Active entries':>14} {'Time (ms)':>12} {'Entries/sec':>15} {'BW (GB/s)':>12}")
    print("-" * 80)
    for n in insertion_counts:
        result = run_clear_active_benchmark(n, device)
        print(
            f"{result.num_ops:>14,} {result.time_ms:>12.3f} {result.ops_per_second:>15,.0f} {result.bandwidth_gb_s:>12.2f}"
        )

    print("\n" + "=" * 80)
    print("Benchmark complete.")


if __name__ == "__main__":
    main()
