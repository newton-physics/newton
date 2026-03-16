**Title:** Add --benchmark option for FPS measurement

---

### Description

Add a `--benchmark` option to the example runner that measures average FPS, excluding startup/compilation overhead.

`--benchmark` takes an optional float argument (seconds). When no value is given, the example runs until `--num-frames` is reached. When a value is given, the example stops after that many seconds or `--num-frames`, whichever comes first. In both cases, `--viewer` is implicitly set to `null`. The first 3 frames run as warmup (JIT compilation, data caching), then a timer starts. On exit, the result is printed:

```
Benchmark: 45.2 FPS (452 frames in 10.01s)
```

Usage:

```bash
# Benchmark until num_frames is reached (default 1000)
uv run -m newton.examples basic_shapes --benchmark

# Benchmark for at most 30 seconds
uv run -m newton.examples basic_shapes --benchmark 30

# Benchmark on CPU with a time limit
uv run -m newton.examples cloth_hanging --benchmark 10 --device cpu
```

Implementation touches two files:

- **`ViewerNull`**: add `benchmark`/`benchmark_timeout`/`benchmark_start_frame` constructor params, start timing after warmup frames, stop when timeout reached (if set), expose `benchmark_result()` method.
- **`newton/examples/__init__.py`**: add `--benchmark` arg (`type=float, nargs="?", default=False, const=None`), force `null` viewer when set, print FPS after `run()`.

No changes to individual examples, the viewer base class, or other viewer backends. `--test` remains orthogonal.

---

### Motivation / Use Case

When benchmarking CPU vs GPU performance, `--num-frames` is impractical — a count that finishes in seconds on GPU can take 20+ minutes on CPU for heavy examples (e.g. `cloth_franka`). An external `subprocess.run(timeout=...)` kills the process without reporting FPS and counts JIT compilation in the timing. A time-based benchmark mode lets both fast and slow devices produce a comparable FPS metric within a predictable wall-clock budget.

---

### Alternatives Considered

- **`--num-frames` only**: Requires per-example, per-device tuning to get reasonable run times. Not practical for automated cross-device comparison.
- **External script with timeout + frame counting**: Requires killing the process and parsing stdout for frame counts. Fragile, and compilation overhead contaminates the measurement.
- **Adding timing to the GL viewer**: Would conflate rendering overhead with simulation performance. The null viewer isolates simulation cost.
