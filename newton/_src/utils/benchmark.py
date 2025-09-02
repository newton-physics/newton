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

import itertools
import time


def run_benchmark(benchmark_cls, number=1, print_results=True):
    """
    Simple scaffold to run a benchmark class.

    Parameters:
      benchmark_cls    : ASV-compatible benchmark class.
      number  : Number of iterations to run each benchmark method.

    Returns:
      A dictionary mapping (method name, parameter tuple) to the average result.
    """

    # Determine all parameter combinations (if any).
    if hasattr(benchmark_cls, "params"):
        param_lists = benchmark_cls.params
        combinations = list(itertools.product(*param_lists))
    else:
        combinations = [()]

    results = {}
    # For each parameter combination:
    for params in combinations:
        # Create a fresh benchmark instance.
        instance = benchmark_cls()
        if hasattr(instance, "setup"):
            instance.setup(*params)
        # Iterate over all attributes to find benchmark methods.
        for attr in dir(instance):
            if attr.startswith("time_") or attr.startswith("track_"):
                method = getattr(instance, attr)
                print(f"\n[Benchmark] Running {benchmark_cls.__name__}.{attr} with parameters {params}")
                samples = []
                if attr.startswith("time_"):
                    # Run timing benchmarks multiple times and measure elapsed time.
                    for _ in range(number):
                        start = time.perf_counter()
                        method(*params)
                        t = time.perf_counter() - start
                        samples.append(t)
                elif attr.startswith("track_"):
                    # Run tracking benchmarks multiple times and record returned values.
                    for _ in range(number):
                        val = method(*params)
                        samples.append(val)
                # Compute the average result.
                avg = sum(samples) / len(samples)
                results[(attr, params)] = avg
        if hasattr(instance, "teardown"):
            instance.teardown(*params)

    if print_results:
        print("\n=== Benchmark Results ===")
        for (method_name, params), avg in results.items():
            print(f"{benchmark_cls.__name__}.{method_name} {params}: {avg:.6f}")

    return results
