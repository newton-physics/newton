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

"""Benchmark for global contact reduction performance.

This script measures the performance of the global contact reduction system
and its individual components:
1. Contact collection (insert into hash table)
2. Hash table clear_active
3. Contact export (read from hash table and write to output)
"""

import time
from dataclasses import dataclass

import numpy as np
import warp as wp

from newton._src.geometry.contact_reduction_global import (
    GlobalContactReducer,
    GlobalContactReducerData,
    export_and_reduce_contact,
    make_contact_key,
    make_contact_value,
    reduction_insert_slot,
)


@dataclass
class BenchmarkResults:
    """Results from a benchmark run."""

    name: str
    num_contacts: int
    insert_time_ms: float
    clear_time_ms: float
    export_time_ms: float
    total_time_ms: float
    contacts_per_second: float


@wp.kernel
def benchmark_insert_kernel(
    num_contacts: int,
    reducer_data: GlobalContactReducerData,
    beta0: float,
    beta1: float,
):
    """Kernel that simulates inserting contacts into the reducer."""
    tid = wp.tid()
    if tid >= num_contacts:
        return

    # Simulate contact data with some distribution
    shape_a = tid % 100  # 100 different shapes
    shape_b = (tid // 100) % 100 + 100  # 100 different shapes
    x = float(tid % 50) * 0.1
    y = float((tid // 50) % 50) * 0.1
    z = float((tid // 2500) % 50) * 0.1
    position = wp.vec3(x, y, z)

    # Normal varies based on contact
    nx = wp.sin(float(tid) * 0.1)
    ny = wp.cos(float(tid) * 0.1)
    nz = wp.sqrt(1.0 - nx * nx - ny * ny)
    normal = wp.vec3(nx, ny, nz)

    depth = -0.01 + float(tid % 100) * 0.0001  # Varies from -0.01 to 0
    feature = tid % 10

    export_and_reduce_contact(shape_a, shape_b, position, normal, depth, feature, reducer_data, beta0, beta1)


@wp.kernel
def benchmark_simple_insert_kernel(
    num_insertions: int,
    keys: wp.array(dtype=wp.uint64),
    values: wp.array(dtype=wp.uint64),
    active_slots: wp.array(dtype=wp.int32),
):
    """Kernel that benchmarks raw hash table insertions."""
    tid = wp.tid()
    if tid >= num_insertions:
        return

    # Create key from (shape_pair, bin)
    shape_a = tid % 100
    shape_b = (tid // 100) % 100
    bin_id = tid % 20

    key = make_contact_key(shape_a, shape_b, bin_id)
    slot_id = tid % 13  # One of 13 slots

    # Value encodes score and contact_id
    score = float(tid)
    contact_id = tid
    value = make_contact_value(score, contact_id)

    reduction_insert_slot(key, slot_id, value, keys, values, active_slots)


def run_benchmark(num_contacts: int, device: str = "cuda:0", num_iterations: int = 10) -> BenchmarkResults:
    """Run a benchmark with the specified number of contacts.

    Args:
        num_contacts: Number of contacts to simulate
        device: Warp device to use
        num_iterations: Number of iterations to average over

    Returns:
        BenchmarkResults with timing data
    """
    # Create reducer with capacity for the contacts
    reducer = GlobalContactReducer(
        capacity=num_contacts * 2,  # 2x headroom
        device=device,
        num_betas=2,
    )

    beta0 = 1000000.0
    beta1 = 0.0001

    # Warm up
    reducer_data = reducer.get_data_struct()
    wp.launch(
        benchmark_insert_kernel,
        dim=num_contacts,
        inputs=[num_contacts, reducer_data, beta0, beta1],
        device=device,
    )
    wp.synchronize()
    reducer.clear_active()
    wp.synchronize()

    # Benchmark insert
    insert_times = []
    for _ in range(num_iterations):
        reducer.clear()
        wp.synchronize()

        reducer_data = reducer.get_data_struct()
        start = time.perf_counter()
        wp.launch(
            benchmark_insert_kernel,
            dim=num_contacts,
            inputs=[num_contacts, reducer_data, beta0, beta1],
            device=device,
        )
        wp.synchronize()
        insert_times.append((time.perf_counter() - start) * 1000)

    # Benchmark clear_active
    clear_times = []
    for _ in range(num_iterations):
        # First insert some data
        reducer_data = reducer.get_data_struct()
        wp.launch(
            benchmark_insert_kernel,
            dim=num_contacts,
            inputs=[num_contacts, reducer_data, beta0, beta1],
            device=device,
        )
        wp.synchronize()

        start = time.perf_counter()
        reducer.clear_active()
        wp.synchronize()
        clear_times.append((time.perf_counter() - start) * 1000)

    # Calculate statistics
    insert_time = np.mean(insert_times)
    clear_time = np.mean(clear_times)
    export_time = 0.0  # TODO: Add export benchmark when kernel is integrated
    total_time = insert_time + clear_time + export_time
    contacts_per_second = num_contacts / (total_time / 1000) if total_time > 0 else 0

    return BenchmarkResults(
        name=f"contacts_{num_contacts}",
        num_contacts=num_contacts,
        insert_time_ms=insert_time,
        clear_time_ms=clear_time,
        export_time_ms=export_time,
        total_time_ms=total_time,
        contacts_per_second=contacts_per_second,
    )


def run_hashtable_benchmark(num_insertions: int, device: str = "cuda:0", num_iterations: int = 10) -> float:
    """Benchmark raw hash table insertion performance.

    Args:
        num_insertions: Number of insertions to perform
        device: Warp device
        num_iterations: Number of iterations to average

    Returns:
        Average time in milliseconds
    """
    values_per_key = 13
    capacity = max(num_insertions * 4, 1024)  # Reduced headroom to avoid OOM

    from newton._src.geometry.hashtable import HashTable  # noqa: PLC0415

    ht = HashTable(capacity, device=device)
    values = wp.zeros(ht.capacity * values_per_key, dtype=wp.uint64, device=device)

    # Warm up
    wp.launch(
        benchmark_simple_insert_kernel,
        dim=num_insertions,
        inputs=[num_insertions, ht.keys, values, ht.active_slots],
        device=device,
    )
    wp.synchronize()
    ht.clear_active()
    values.zero_()
    wp.synchronize()

    times = []
    for _ in range(num_iterations):
        ht.clear()
        values.zero_()
        wp.synchronize()

        start = time.perf_counter()
        wp.launch(
            benchmark_simple_insert_kernel,
            dim=num_insertions,
            inputs=[num_insertions, ht.keys, values, ht.active_slots],
            device=device,
        )
        wp.synchronize()
        times.append((time.perf_counter() - start) * 1000)

    return np.mean(times)


def main():
    """Run the benchmark suite."""
    wp.init()

    device = "cuda:0"
    print(f"Running benchmarks on {device}")
    print("=" * 70)

    # Test different contact counts
    contact_counts = [1000, 10000, 50000, 100000, 500000]

    print("\n1. Full contact reduction benchmark (insert + clear_active)")
    print("-" * 70)
    print(f"{'Contacts':>12} {'Insert (ms)':>12} {'Clear (ms)':>12} {'Total (ms)':>12} {'Contacts/s':>15}")
    print("-" * 70)

    for num_contacts in contact_counts:
        results = run_benchmark(num_contacts, device=device)
        print(
            f"{results.num_contacts:>12} "
            f"{results.insert_time_ms:>12.3f} "
            f"{results.clear_time_ms:>12.3f} "
            f"{results.total_time_ms:>12.3f} "
            f"{results.contacts_per_second:>15,.0f}"
        )

    print("\n2. Raw hash table insertion benchmark")
    print("-" * 70)
    print(f"{'Insertions':>12} {'Time (ms)':>12} {'Insertions/s':>15}")
    print("-" * 70)

    insertion_counts = [10000, 100000, 500000, 1000000]
    for num_insertions in insertion_counts:
        time_ms = run_hashtable_benchmark(num_insertions, device=device)
        insertions_per_second = num_insertions / (time_ms / 1000)
        print(f"{num_insertions:>12} {time_ms:>12.3f} {insertions_per_second:>15,.0f}")

    print("\n" + "=" * 70)
    print("Benchmark complete.")


if __name__ == "__main__":
    main()
