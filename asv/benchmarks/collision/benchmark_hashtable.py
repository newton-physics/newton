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

"""ASV benchmarks for hash table performance.

Measures hash table operations:
1. Insert performance (with varying collision rates)
2. Clear active performance
"""

import warp as wp
from asv_runner.benchmarks.mark import skip_benchmark_if

wp.config.quiet = True

from newton._src.geometry.contact_reduction_global import reduction_insert_slot
from newton._src.geometry.hashtable import HashTable


@wp.func
def make_key(shape_a: int, shape_b: int, bin_id: int) -> wp.uint64:
    """Create a hash table key from shape pair and bin."""
    key = wp.uint64(shape_a) & wp.uint64(0x1FFFFFFF)
    key = key | ((wp.uint64(shape_b) & wp.uint64(0x1FFFFFFF)) << wp.uint64(29))
    key = key | ((wp.uint64(bin_id) & wp.uint64(0x1F)) << wp.uint64(58))
    return key


@wp.func
def make_value(score: float, contact_id: int) -> wp.uint64:
    """Pack score and contact_id into a uint64 value."""
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

    shape_a = tid % 1000
    shape_b = (tid // 1000) % 1000
    bin_id = (tid // 1000000) % 20

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

    group = tid // 100
    shape_a = group % 10
    shape_b = (group // 10) % 10
    bin_id = group % 20

    key = make_key(shape_a, shape_b, bin_id)
    slot_id = tid % 13
    value = make_value(float(tid), tid)

    reduction_insert_slot(key, slot_id, value, keys, values, active_slots)


class FastHashTableInsertLowCollision:
    """Benchmark hash table insertion with low collision rate."""

    repeat = 3
    number = 1
    params = [[100_000, 500_000, 1_000_000]]
    param_names = ["num_insertions"]

    def setup(self, num_insertions):
        self.num_insertions = num_insertions
        self.values_per_key = 13
        capacity = max(num_insertions * 10, 1024)

        self.ht = HashTable(capacity, device="cuda:0")
        self.values = wp.zeros(self.ht.capacity * self.values_per_key, dtype=wp.uint64, device="cuda:0")

        # Warm up
        wp.launch(
            insert_low_collision_kernel,
            dim=num_insertions,
            inputs=[num_insertions, self.ht.keys, self.values, self.ht.active_slots],
            device="cuda:0",
        )
        wp.synchronize()

        # Clear for first iteration
        self.ht.clear()
        self.values.zero_()
        wp.synchronize()

    @skip_benchmark_if(True)
    def time_insert(self, num_insertions):
        wp.launch(
            insert_low_collision_kernel,
            dim=num_insertions,
            inputs=[num_insertions, self.ht.keys, self.values, self.ht.active_slots],
            device="cuda:0",
        )
        wp.synchronize()

    def teardown(self, num_insertions):
        # Clear after each iteration
        self.ht.clear()
        self.values.zero_()
        wp.synchronize()


class FastHashTableInsertHighCollision:
    """Benchmark hash table insertion with high collision rate."""

    repeat = 3
    number = 1
    params = [[100_000, 500_000, 1_000_000]]
    param_names = ["num_insertions"]

    def setup(self, num_insertions):
        self.num_insertions = num_insertions
        self.values_per_key = 13
        capacity = max(num_insertions * 10, 1024)

        self.ht = HashTable(capacity, device="cuda:0")
        self.values = wp.zeros(self.ht.capacity * self.values_per_key, dtype=wp.uint64, device="cuda:0")

        # Warm up
        wp.launch(
            insert_high_collision_kernel,
            dim=num_insertions,
            inputs=[num_insertions, self.ht.keys, self.values, self.ht.active_slots],
            device="cuda:0",
        )
        wp.synchronize()

        # Clear for first iteration
        self.ht.clear()
        self.values.zero_()
        wp.synchronize()

    @skip_benchmark_if(True)
    def time_insert(self, num_insertions):
        wp.launch(
            insert_high_collision_kernel,
            dim=num_insertions,
            inputs=[num_insertions, self.ht.keys, self.values, self.ht.active_slots],
            device="cuda:0",
        )
        wp.synchronize()

    def teardown(self, num_insertions):
        # Clear after each iteration
        self.ht.clear()
        self.values.zero_()
        wp.synchronize()


class FastHashTableClearActive:
    """Benchmark hash table clear_active operation."""

    repeat = 3
    number = 1
    params = [[10_000, 100_000, 500_000]]
    param_names = ["num_active"]

    def setup(self, num_active):
        self.num_active = num_active
        self.values_per_key = 13
        capacity = num_active * 2

        self.ht = HashTable(capacity, device="cuda:0")
        self.values = wp.zeros(self.ht.capacity * self.values_per_key, dtype=wp.uint64, device="cuda:0")

        # Fill with data before timing clear_active
        wp.launch(
            insert_low_collision_kernel,
            dim=num_active,
            inputs=[num_active, self.ht.keys, self.values, self.ht.active_slots],
            device="cuda:0",
        )
        wp.synchronize()

    @skip_benchmark_if(True)
    def time_clear_active(self, num_active):
        self.ht.clear_active()
        wp.synchronize()

    def teardown(self, num_active):
        # Re-fill for next iteration
        wp.launch(
            insert_low_collision_kernel,
            dim=num_active,
            inputs=[num_active, self.ht.keys, self.values, self.ht.active_slots],
            device="cuda:0",
        )
        wp.synchronize()


if __name__ == "__main__":
    import argparse

    from newton.utils import run_benchmark

    benchmark_list = {
        "FastHashTableInsertLowCollision": FastHashTableInsertLowCollision,
        "FastHashTableInsertHighCollision": FastHashTableInsertHighCollision,
        "FastHashTableClearActive": FastHashTableClearActive,
    }

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "-b", "--bench", default=None, action="append", choices=benchmark_list.keys(), help="Run a single benchmark."
    )
    args = parser.parse_known_args()[0]

    if args.bench is None:
        benchmarks = benchmark_list.keys()
    else:
        benchmarks = args.bench

    for key in benchmarks:
        benchmark = benchmark_list[key]
        run_benchmark(benchmark)
