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

"""ASV benchmarks for global contact reduction performance.

Measures the performance of the global contact reduction system:
1. Contact collection (insert into hash table)
2. Hash table clear_active
"""

import warp as wp
from asv_runner.benchmarks.mark import skip_benchmark_if

wp.config.quiet = True

from newton._src.geometry.contact_reduction_global import (
    GlobalContactReducer,
    GlobalContactReducerData,
    export_and_reduce_contact,
)


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
    shape_a = tid % 100
    shape_b = (tid // 100) % 100 + 100
    x = float(tid % 50) * 0.1
    y = float((tid // 50) % 50) * 0.1
    z = float((tid // 2500) % 50) * 0.1
    position = wp.vec3(x, y, z)

    # Normal varies based on contact
    nx = wp.sin(float(tid) * 0.1)
    ny = wp.cos(float(tid) * 0.1)
    nz = wp.sqrt(1.0 - nx * nx - ny * ny)
    normal = wp.vec3(nx, ny, nz)

    depth = -0.01 + float(tid % 100) * 0.0001

    export_and_reduce_contact(shape_a, shape_b, position, normal, depth, reducer_data, beta0, beta1)


class FastGlobalContactReducerInsert:
    """Benchmark contact insertion into GlobalContactReducer."""

    repeat = 3
    number = 1
    params = [[10_000, 100_000, 500_000]]
    param_names = ["num_contacts"]

    def setup(self, num_contacts):
        self.num_contacts = num_contacts
        self.beta0 = 1000000.0
        self.beta1 = 0.0001

        self.reducer = GlobalContactReducer(
            capacity=num_contacts * 2,
            device="cuda:0",
            num_betas=2,
        )
        self.reducer_data = self.reducer.get_data_struct()

        # Warm up
        wp.launch(
            benchmark_insert_kernel,
            dim=num_contacts,
            inputs=[num_contacts, self.reducer_data, self.beta0, self.beta1],
            device="cuda:0",
        )
        wp.synchronize()

        # Clear for first iteration (setup runs before each timing)
        self.reducer.clear()
        wp.synchronize()

    @skip_benchmark_if(True)
    def time_insert(self, num_contacts):
        wp.launch(
            benchmark_insert_kernel,
            dim=num_contacts,
            inputs=[num_contacts, self.reducer_data, self.beta0, self.beta1],
            device="cuda:0",
        )
        wp.synchronize()

    def teardown(self, num_contacts):
        # Clear after each iteration so next setup starts fresh
        self.reducer.clear()
        wp.synchronize()


class FastGlobalContactReducerClearActive:
    """Benchmark clear_active operation on GlobalContactReducer."""

    repeat = 3
    number = 1
    params = [[10_000, 100_000, 500_000]]
    param_names = ["num_contacts"]

    def setup(self, num_contacts):
        self.num_contacts = num_contacts
        self.beta0 = 1000000.0
        self.beta1 = 0.0001

        self.reducer = GlobalContactReducer(
            capacity=num_contacts * 2,
            device="cuda:0",
            num_betas=2,
        )

        # Fill with data before timing clear_active
        reducer_data = self.reducer.get_data_struct()
        wp.launch(
            benchmark_insert_kernel,
            dim=num_contacts,
            inputs=[num_contacts, reducer_data, self.beta0, self.beta1],
            device="cuda:0",
        )
        wp.synchronize()

    @skip_benchmark_if(True)
    def time_clear_active(self, num_contacts):
        self.reducer.clear_active()
        wp.synchronize()

    def teardown(self, num_contacts):
        # Re-fill for next iteration
        reducer_data = self.reducer.get_data_struct()
        wp.launch(
            benchmark_insert_kernel,
            dim=num_contacts,
            inputs=[num_contacts, reducer_data, self.beta0, self.beta1],
            device="cuda:0",
        )
        wp.synchronize()


class FastGlobalContactReducerFullCycle:
    """Benchmark full insert + clear_active cycle."""

    repeat = 3
    number = 1
    params = [[10_000, 100_000, 500_000]]
    param_names = ["num_contacts"]

    def setup(self, num_contacts):
        self.num_contacts = num_contacts
        self.beta0 = 1000000.0
        self.beta1 = 0.0001

        self.reducer = GlobalContactReducer(
            capacity=num_contacts * 2,
            device="cuda:0",
            num_betas=2,
        )

        self.reducer_data = self.reducer.get_data_struct()

        # Warm up
        wp.launch(
            benchmark_insert_kernel,
            dim=num_contacts,
            inputs=[num_contacts, self.reducer_data, self.beta0, self.beta1],
            device="cuda:0",
        )
        wp.synchronize()
        self.reducer.clear_active()
        wp.synchronize()

    @skip_benchmark_if(True)
    def time_full_cycle(self, num_contacts):
        # Clear from previous iteration
        self.reducer.clear_active()

        # Insert contacts
        wp.launch(
            benchmark_insert_kernel,
            dim=num_contacts,
            inputs=[num_contacts, self.reducer_data, self.beta0, self.beta1],
            device="cuda:0",
        )
        wp.synchronize()


if __name__ == "__main__":
    import argparse

    from newton.utils import run_benchmark

    benchmark_list = {
        "FastGlobalContactReducerInsert": FastGlobalContactReducerInsert,
        "FastGlobalContactReducerClearActive": FastGlobalContactReducerClearActive,
        "FastGlobalContactReducerFullCycle": FastGlobalContactReducerFullCycle,
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
