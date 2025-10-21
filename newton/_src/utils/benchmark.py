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
import inspect

import warp as wp
import functools


_STACK = None


class EventTracer:
    """Calculates elapsed times of functions annotated with `event_scope`.

    This class hes been copied from:
    https://github.com/google-deepmind/mujoco_warp/blob/660f8e2f0fb3ccde78c4e70cf24658a1a14ecf1b/mujoco_warp/_src/warp_util.py#L28

    Use as a context manager like so:

      @event_trace
      def my_warp_function(...):
        ...

      with EventTracer() as tracer:
        my_warp_function(...)
        print(tracer.trace())
    """

    def __init__(self, enabled: bool = True):
        global _STACK
        if _STACK is not None:
            raise ValueError("only one EventTracer can run at a time")
        if enabled:
            _STACK = {}

    def __enter__(self):
        return self

    def trace(self) -> dict:
        """Calculates elapsed times for every node of the trace."""
        global _STACK

        if _STACK is None:
            return {}

        ret = {}

        for k, v in _STACK.items():
            events, sub_stack = v
            # push into next level of stack
            saved_stack, _STACK = _STACK, sub_stack
            sub_trace = self.trace()
            # pop!
            _STACK = saved_stack
            events = tuple(wp.get_event_elapsed_time(beg, end) for beg, end in events)
            ret[k] = (events, sub_trace)

        return ret

    def __exit__(self, type, value, traceback):
        global _STACK
        _STACK = None


def _merge(a: dict, b: dict) -> dict:
    """Merges two event trace stacks.
    This function hes been copied from:
    https://github.com/google-deepmind/mujoco_warp/blob/660f8e2f0fb3ccde78c4e70cf24658a1a14ecf1b/mujoco_warp/_src/warp_util.py#L78
    """
    ret = {}
    if not a or not b:
        return dict(**a, **b)
    if set(a) != set(b):
        raise ValueError("incompatible stacks")
    for key in a:
        a1_events, a1_substack = a[key]
        a2_events, a2_substack = b[key]
        ret[key] = (a1_events + a2_events, _merge(a1_substack, a2_substack))
    return ret


def event_scope(fn, name: str = ""):
    """Wraps a function and records an event before and after the function invocation.

    This function hes been copied from:
    https://github.com/google-deepmind/mujoco_warp/blob/660f8e2f0fb3ccde78c4e70cf24658a1a14ecf1b/mujoco_warp/_src/warp_util.py#L92
    """
    name = name or getattr(fn, "__name__")

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        global _STACK
        if _STACK is None:
            return fn(*args, **kwargs)

        for frame_info in inspect.stack():
            if frame_info.function in ("capture_while", "capture_if"):
                return fn(*args, **kwargs)

        # push into next level of stack
        saved_stack, _STACK = _STACK, {}
        beg = wp.Event(enable_timing=True)
        end = wp.Event(enable_timing=True)
        wp.record_event(beg)
        res = fn(*args, **kwargs)
        wp.record_event(end)
        # pop back up to current level
        sub_stack, _STACK = _STACK, saved_stack
        # append events and substack
        prev_events, prev_substack = _STACK.get(name, ((), {}))
        events = prev_events + ((beg, end),)
        sub_stack = _merge(prev_substack, sub_stack)
        _STACK[name] = (events, sub_stack)
        return res

    return wrapper


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
