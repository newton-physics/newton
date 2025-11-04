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

import functools
import itertools
import time

import warp as wp


class EventTracer:
    """
    Calculates elapsed times of functions annotated with `event_scope`.

    .. note::

        This class has been copied from:
        https://github.com/google-deepmind/mujoco_warp/blob/660f8e2f0fb3ccde78c4e70cf24658a1a14ecf1b/mujoco_warp/_src/warp_util.py#L28
        Then modified to change _STACK from being a global.

    Example
    -------

    .. code-block:: python

      @event_trace
      def my_warp_function(...):
        ...

      with EventTracer() as tracer:
        my_warp_function(...)
        print(tracer.trace())
    """

    _STACK = None
    _active_instance = None

    def __new__(cls, enabled):
        if EventTracer._active_instance is not None and enabled:
            raise ValueError("only one EventTracer can run at a time")
        return super().__new__(cls)

    def __init__(self, enabled: bool = True):
        """
        Args:
            enabled (bool): If True, elapsed times of annotated functions are measured.
        """
        if enabled:
            self._STACK = {}
            EventTracer._active_instance = self

    def __enter__(self):
        return self

    def trace(self) -> dict:
        """Calculates elapsed times for every node of the trace."""

        if self._STACK is None:
            return {}

        ret = {}

        for k, v in self._STACK.items():
            events, sub_stack = v
            # push into next level of stack
            saved_stack, self._STACK = self._STACK, sub_stack
            sub_trace = self.trace()
            # pop!
            self._STACK = saved_stack
            events = tuple(wp.get_event_elapsed_time(beg, end) for beg, end in events)
            ret[k] = (events, sub_trace)

        return ret

    def add_trace(self, stack, new_stack):
        """Sums elapsed times from two difference traces."""
        ret = {}
        for k in new_stack:
            times, sub_stack = stack[k] if k in stack.keys() else (0, {})
            new_times, new_sub_stack = new_stack[k]
            times = times + sum(new_times)
            ret[k] = (times, self.add_trace(sub_stack, new_sub_stack))
        return ret

    def __exit__(self, type, value, traceback):
        self._STACK = None
        if EventTracer._active_instance is self:
            EventTracer._active_instance = None


def _merge(a: dict, b: dict) -> dict:
    """
    Merges two event trace stacks.

    .. note::

        This function has been copied from:
        https://github.com/google-deepmind/mujoco_warp/blob/660f8e2f0fb3ccde78c4e70cf24658a1a14ecf1b/mujoco_warp/_src/warp_util.py#L78
        Then modified to change how the dictionary items were accessed.

    Parameters:
      a  : Base event trace stack.
      b  : Second event trace stack to add to the base event trace stack.

    Returns:
      A dictionary where the two event traces are merged.
    """
    ret = {}
    if not a or not b:
        return dict(**a, **b)
    if set(a) != set(b):
        raise ValueError("incompatible stacks")
    for key, (a1_events, a1_substack) in a.items():
        a2_events, a2_substack = b[key]
        ret[key] = (a1_events + a2_events, _merge(a1_substack, a2_substack))
    return ret


def event_scope(fn, name: str = ""):
    """
    Wraps a function and records an event before and after the function invocation.

    .. note::

        This function has been copied from:
        https://github.com/google-deepmind/mujoco_warp/blob/660f8e2f0fb3ccde78c4e70cf24658a1a14ecf1b/mujoco_warp/_src/warp_util.py#L92
        Then modified to change _STACK from being a global.

    Parameters:
      fn    : Function to be wrapped.
      name  : Custom name associated with the function.
    """
    name = name or fn.__name__

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        if EventTracer._active_instance is None:
            return fn(*args, **kwargs)

        # push into next level of stack
        saved_stack, EventTracer._active_instance._STACK = EventTracer._active_instance._STACK, {}
        beg = wp.Event(enable_timing=True)
        end = wp.Event(enable_timing=True)
        wp.record_event(beg)
        res = fn(*args, **kwargs)
        wp.record_event(end)
        # pop back up to current level
        sub_stack, EventTracer._active_instance._STACK = EventTracer._active_instance._STACK, saved_stack
        # append events and substack
        prev_events, prev_substack = EventTracer._active_instance._STACK.get(name, ((), {}))
        events = (*prev_events, (beg, end))
        sub_stack = _merge(prev_substack, sub_stack)
        EventTracer._active_instance._STACK[name] = (events, sub_stack)
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
