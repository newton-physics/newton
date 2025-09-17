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


def _convert_params_to_dict(params, param_names):
    """Convert params to keyword arguments using param_names."""
    if not param_names:
        return None

    # Handle single value case by wrapping in tuple
    if not isinstance(params, (list | tuple)):
        params = (params,)

    return dict(zip(param_names, params, strict=False))


def _call_with_params(method, param_dict, params, cached_data):
    """Call a method with appropriate parameters (keyword or positional)."""
    if param_dict is not None:
        if cached_data is not None:
            return method(cached_data, **param_dict)
        else:
            return method(**param_dict)
    else:
        if cached_data is not None:
            return method(cached_data, *params)
        else:
            return method(*params)


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
        if len(param_lists) > 1 and all(isinstance(item, (list | tuple)) for item in param_lists):
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
        param_names = getattr(benchmark_cls, "param_names", None)
        param_dict = _convert_params_to_dict(params, param_names)

        # Ensure params is always a tuple for consistent handling
        params_tuple = params
        if not isinstance(params, (list | tuple)):
            params_tuple = (params,)

        # Call setup_cache on the first combination only
        if i == 0 and hasattr(benchmark_cls, "setup_cache"):
            print(f"\n[Benchmark] Running {benchmark_cls.__name__}.setup_cache")
            cached_data = instance.setup_cache()

        if hasattr(instance, "setup"):
            _call_with_params(instance.setup, param_dict, params_tuple, cached_data)

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
                        _call_with_params(method, param_dict, params_tuple, cached_data)
                        t = time.perf_counter() - start
                        samples.append(t)
                elif attr.startswith("track_"):
                    # Run tracking benchmarks multiple times and record returned values.
                    for _ in range(number):
                        val = _call_with_params(method, param_dict, params_tuple, cached_data)
                        samples.append(val)
                # Compute the average result.
                avg = sum(samples) / len(samples)
                results[(attr, params)] = avg
        if hasattr(instance, "teardown"):
            _call_with_params(instance.teardown, param_dict, params_tuple, cached_data)

    if print_results:
        print("\n=== Benchmark Results ===")
        for (method_name, params), avg in results.items():
            print(f"{benchmark_cls.__name__}.{method_name} {params}: {avg:.6f}")

    return results
