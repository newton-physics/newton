# Implementation Plan: --benchmark option

## Overview

Two files to change, one constant to define. No changes to individual examples,
the viewer base class, or other viewer backends.

## Step 1: Add benchmark state to ViewerNull

**File:** `newton/_src/viewer/viewer_null.py`

### 1a. Update constructor signature

Change `__init__` from:

```python
def __init__(self, num_frames: int = 1000):
```

to:

```python
def __init__(
    self,
    num_frames: int = 1000,
    benchmark: bool = False,
    benchmark_timeout: float | None = None,
    benchmark_start_frame: int = 3,
):
```

Add to `__init__` body:

```python
self.benchmark = benchmark or benchmark_timeout is not None
self.benchmark_timeout = benchmark_timeout
self.benchmark_start_frame = benchmark_start_frame
self._bench_start_time: float | None = None
self._bench_frames = 0
self._bench_elapsed = 0.0
```

### 1b. Update `end_frame()`

Currently:

```python
def end_frame(self):
    self.frame_count += 1
```

Change to:

```python
def end_frame(self):
    self.frame_count += 1

    if self.benchmark:
        if self.frame_count == self.benchmark_start_frame:
            self._bench_start_time = time.perf_counter()
        elif self._bench_start_time is not None:
            self._bench_frames = self.frame_count - self.benchmark_start_frame
            self._bench_elapsed = time.perf_counter() - self._bench_start_time
```

Add `import time` at the top of the file (alongside the existing stdlib imports).

### 1c. Update `is_running()`

Currently:

```python
def is_running(self) -> bool:
    return self.frame_count < self.num_frames
```

Change to:

```python
def is_running(self) -> bool:
    if self.frame_count >= self.num_frames:
        return False
    if (
        self.benchmark_timeout is not None
        and self._bench_start_time is not None
        and self._bench_elapsed >= self.benchmark_timeout
    ):
        return False
    return True
```

### 1d. Add `benchmark_result()` method

Add after `is_running()`:

```python
def benchmark_result(self) -> dict | None:
    """Return benchmark results, or None if benchmarking was not enabled.

    Returns:
        Dictionary with ``fps``, ``frames``, and ``elapsed`` keys,
        or ``None`` if benchmarking is not enabled.
    """
    if not self.benchmark:
        return None
    if self._bench_frames == 0 or self._bench_elapsed == 0.0:
        return {"fps": 0.0, "frames": 0, "elapsed": 0.0}
    return {
        "fps": self._bench_frames / self._bench_elapsed,
        "frames": self._bench_frames,
        "elapsed": self._bench_elapsed,
    }
```

## Step 2: Add --benchmark argument and wiring in the example runner

**File:** `newton/examples/__init__.py`

### 2a. Add `--benchmark` to `create_parser()`

Insert after the `--quiet` argument block (after line 438):

```python
parser.add_argument(
    "--benchmark",
    type=float,
    default=False,
    nargs="?",
    const=None,
    metavar="SECONDS",
    help="Run in benchmark mode: measure FPS after a warmup period. "
    "If SECONDS is given, stop after that many seconds or --num-frames, "
    "whichever comes first.",
)
```

### 2b. Update `init()` to handle --benchmark

In the `init()` function, after `wp.set_device(args.device)` and
before the viewer creation block, add:

```python
# Benchmark mode forces null viewer
if args.benchmark is not False:
    args.viewer = "null"
```

Then update the `elif args.viewer == "null":` branch from:

```python
elif args.viewer == "null":
    viewer = newton.viewer.ViewerNull(num_frames=args.num_frames)
```

to:

```python
elif args.viewer == "null":
    viewer = newton.viewer.ViewerNull(
        num_frames=args.num_frames,
        benchmark=args.benchmark is not False,
        benchmark_timeout=args.benchmark or None,
    )
```

`args.benchmark` is `False` when absent, `None` for bare `--benchmark`,
or a `float` for `--benchmark SECONDS`. `args.benchmark or None` maps
both `False` and `None` to `None` (no timeout), passing through any
positive float as-is.

### 2c. Print benchmark results after `run()`

In the `run()` function, after `viewer.close()` and before the
`if perform_test:` block, add:

```python
if hasattr(viewer, "benchmark_result"):
    result = viewer.benchmark_result()
    if result is not None:
        print(
            f"Benchmark: {_format_fps(result['fps'])} FPS ({result['frames']} frames in {result['elapsed']:.2f}s)"
        )
```

## Step 3: Verify

### Manual smoke tests

```bash
# Benchmark until num_frames (default 1000) on GPU
uv run -m newton.examples basic_shapes --benchmark

# Custom 5s timeout on CPU
uv run -m newton.examples basic_shapes --benchmark 5 --device cpu

# Combined with --test (should print both FPS and run test assertions)
uv run -m newton.examples basic_shapes --benchmark --test

# Verify --num-frames still caps (set low so it finishes before timeout)
uv run -m newton.examples basic_shapes --benchmark 60 --num-frames 10

# Verify existing behavior unchanged (no --benchmark)
uv run -m newton.examples basic_shapes --viewer null --num-frames 20
```

### Edge cases to check

- `--benchmark 0`: equivalent to bare `--benchmark`, no timeout
- `--benchmark` with an example that's slower than 1 frame per
  `benchmark_timeout`: should complete at least 1 frame after warmup, then
  stop; FPS will be very low but valid
- `--benchmark` combined with `--viewer gl`: the benchmark flag should
  silently override the viewer to `null`
- Examples that crash during warmup: normal error propagation, no benchmark
  output printed

## Files changed

| File | Change |
|---|---|
| `newton/_src/viewer/viewer_null.py` | Add benchmark timing logic |
| `newton/examples/__init__.py` | Add `--benchmark` arg + wiring + FPS output |

No other files need changes.
