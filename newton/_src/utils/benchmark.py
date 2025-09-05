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
        # If param_lists contains multiple lists, generate all combinations
        # If it's a single list, just use it directly
        if len(param_lists) > 1 and all(isinstance(item, (list, tuple)) for item in param_lists):
            combinations = itertools.product(*param_lists)
        else:
            combinations = param_lists
    else:
        combinations = [()]

    results = {}
    cached_data = None
    
    # For each parameter combination:
    for i, params in enumerate(combinations):
        
        # Create a fresh benchmark instance.
        instance = benchmark_cls()
        
        # Convert params to keyword arguments using param_names
        if hasattr(benchmark_cls, "param_names"):
            # Handle single value case by wrapping in tuple
            if not isinstance(params, (list, tuple)):
                params = (params,)
            param_dict = dict(zip(benchmark_cls.param_names, params))
        else:
            # Fallback to positional args
            if not isinstance(params, (list, tuple)):
                params = (params,)
            param_dict = None
        
        # Call setup_cache on the first combination only
        if i == 0 and hasattr(benchmark_cls, "setup_cache"):
            cached_data = instance.setup_cache()
        
        if hasattr(instance, "setup"):
            if param_dict is not None:
                instance.setup(**param_dict)
            else:
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
                        if param_dict is not None:
                            if cached_data is not None:
                                method(cached_data, **param_dict)
                            else:
                                method(**param_dict)
                        else:
                            if cached_data is not None:
                                method(cached_data, *params)
                            else:
                                method(*params)
                        t = time.perf_counter() - start
                        samples.append(t)
                elif attr.startswith("track_"):
                    # Run tracking benchmarks multiple times and record returned values.
                    for _ in range(number):
                        if param_dict is not None:
                            if cached_data is not None:
                                val = method(cached_data, **param_dict)
                            else:
                                val = method(**param_dict)
                        else:
                            if cached_data is not None:
                                val = method(cached_data, *params)
                            else:
                                val = method(*params)
                        samples.append(val)
                # Compute the average result.
                avg = sum(samples) / len(samples)
                results[(attr, params)] = avg
        if hasattr(instance, "teardown"):
            if param_dict is not None:
                if cached_data is not None:
                    instance.teardown(cached_data, **param_dict)
                else:
                    instance.teardown(**param_dict)
            else:
                if cached_data is not None:
                    instance.teardown(cached_data, *params)
                else:
                    instance.teardown(*params)

    if print_results:
        print("\n=== Benchmark Results ===")
        for (method_name, params), avg in results.items():
            print(f"{benchmark_cls.__name__}.{method_name} {params}: {avg:.6f}")

    return results
