# Cosserat Codex Warp GPU Performance Optimization Plan

## Executive Summary

Analysis of `newton/examples/cosserat_codex` reveals several GPU performance bottlenecks that significantly limit throughput. The most critical issues are:

1. **Sequential kernels** running O(N) work on single GPU thread
2. **GPU-CPU round-trips** in the render path every frame
3. **Per-rod kernel launches** in Python loops instead of batched operations
4. **No CUDA graph capture** for rendering

This plan prioritizes optimizations by impact and implementation complexity.

---

## Phase 1: Critical - Parallelize Sequential Kernels

**Impact: HIGH | Risk: MEDIUM | Files: `kernels/collision.py`, `kernels/solvers.py`**

### 1.1 Parallelize `_warp_constraint_max` Reduction (collision.py:204-223)

**Current:** Single thread iterates over all edges computing max norm
```python
tid = wp.tid()
if tid != 0:
    return
for edge in range(n_edges):  # O(N) on single thread
    ...
```

**Optimization:** Parallel tree reduction using Warp atomics or tile operations

```python
@wp.kernel
def _warp_constraint_max_parallel(
    constraint_values: wp.array(dtype=wp.float32),
    n_edges: int,
    partial_max: wp.array(dtype=wp.float32),  # Per-block partial results
):
    tid = wp.tid()
    if tid >= n_edges:
        return

    # Each thread computes its edge's norm
    base_idx = tid * 6
    norm_sq = 0.0
    for j in range(6):
        val = constraint_values[base_idx + j]
        norm_sq += val * val
    norm = wp.sqrt(norm_sq)

    # Atomic max to shared output
    wp.atomic_max(partial_max, 0, norm)
```

**Alternative:** Use `wp.tile_reduce()` for better performance on newer Warp versions.

### 1.2 Parallelize `_warp_apply_direct_corrections` (collision.py:290-425)

**Current:** Single thread loops over all edges sequentially
```python
tid = wp.tid()
if tid != 0:
    return
for edge in range(n_edges):  # Sequential
    # Apply corrections to particles edge and edge+1
```

**Optimization:** Per-edge parallelization with careful handling of shared particles

```python
@wp.kernel
def _warp_apply_direct_corrections_parallel(
    ...,
    n_edges: int,
):
    edge = wp.tid()
    if edge >= n_edges:
        return

    # Load delta_lambda for this edge
    base_idx = edge * 6
    dl = [delta_lambda[base_idx + i] for i in range(6)]

    # Update lambda_sum (independent per-edge)
    for i in range(6):
        lambda_sum[base_idx + i] += dl[i]

    # Position correction for particle 'edge' (shared with edge-1)
    # Use atomic add for thread safety
    corr_p0 = compute_position_correction(jacobian_pos, edge, dl, inv_masses[edge])
    wp.atomic_add(predicted_positions, edge, corr_p0)

    # Position correction for particle 'edge+1' (shared with edge+1)
    corr_p1 = compute_position_correction_p1(jacobian_pos, edge, dl, inv_masses[edge+1])
    wp.atomic_add(predicted_positions, edge + 1, corr_p1)

    # Rotation corrections similarly with atomics
```

**Risk:** Atomic operations may cause contention for adjacent edges. Test performance vs sequential.

---

## Phase 2: High Priority - Eliminate Render Path GPU-CPU Transfers

**Impact: HIGH | Risk: LOW | Files: `simulation/example.py`, `kernels/collision.py`**

### 2.1 GPU-Native Director Line Building (example.py:974-998)

**Current:** Every render frame:
```python
gpu_positions = rod.positions_numpy()  # GPU -> CPU transfer
gpu_orientations = rod.orientations_numpy()  # GPU -> CPU transfer
gpu_starts, gpu_ends, gpu_colors = build_director_lines(...)  # CPU compute
self.viewer.log_lines(..., wp.array(gpu_starts, ...))  # CPU -> GPU transfer
```

**Optimization:** Create new kernel to build director lines directly on GPU

```python
# New file or add to kernels/collision.py
@wp.kernel
def _warp_build_director_lines(
    positions: wp.array(dtype=wp.vec3),
    orientations: wp.array(dtype=wp.quat),
    offset: wp.vec3,
    scale: float,
    starts: wp.array(dtype=wp.vec3),  # Output: 3 * num_points
    ends: wp.array(dtype=wp.vec3),
    colors: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()
    pos = positions[tid] + offset
    q = orientations[tid]

    # Director d1 (red)
    d1 = wp.quat_rotate(q, wp.vec3(1.0, 0.0, 0.0))
    starts[tid * 3 + 0] = pos
    ends[tid * 3 + 0] = pos + d1 * scale
    colors[tid * 3 + 0] = wp.vec3(1.0, 0.0, 0.0)

    # Director d2 (green)
    d2 = wp.quat_rotate(q, wp.vec3(0.0, 1.0, 0.0))
    starts[tid * 3 + 1] = pos
    ends[tid * 3 + 1] = pos + d2 * scale
    colors[tid * 3 + 1] = wp.vec3(0.0, 1.0, 0.0)

    # Director d3 (blue)
    d3 = wp.quat_rotate(q, wp.vec3(0.0, 0.0, 1.0))
    starts[tid * 3 + 2] = pos
    ends[tid * 3 + 2] = pos + d3 * scale
    colors[tid * 3 + 2] = wp.vec3(0.0, 0.0, 1.0)
```

**Implementation:**
1. Pre-allocate GPU arrays for director visualization in `__init__`
2. Replace CPU `build_director_lines()` with kernel launch
3. Pass GPU arrays directly to viewer

### 2.2 GPU-Native Mesh Update (example.py:1020-1034)

**Current:** Downloads positions, generates mesh on CPU, re-uploads

**Optimization:** The `RodMesher` class already has `_update_rod_mesh_kernel` in `meshing/kernels.py`. Ensure it's being used with GPU-resident arrays:

```python
# Instead of:
gpu_positions = rod.positions_numpy() + offset  # CPU
self._gpu_meshers[idx].update_numpy(gpu_positions)  # CPU mesh gen

# Use:
self._gpu_meshers[idx].update_warp(rod.positions_wp, offset_wp)  # GPU mesh gen
```

**Note:** Check if `update_warp()` method exists; if not, add it to `RodMesher`.

---

## Phase 3: High Priority - Batch Kernel Launches

**Impact: MEDIUM-HIGH | Risk: LOW | Files: `simulation/example.py`**

### 3.1 Batch Per-Rod Operations (example.py:442-459)

**Current:** Python loop launches 2 kernels per rod
```python
for idx, rod in enumerate(self.gpu_state.rods):
    wp.launch(_warp_copy_with_offset, dim=rod.num_points, ...)
    wp.launch(_warp_copy_with_offset, dim=rod.num_points, ...)
```

**Optimization:** Single batched kernel for all rods

```python
@wp.kernel
def _warp_copy_positions_batched(
    rod_positions: wp.array(dtype=wp.vec3),  # Flattened all rods
    offsets: wp.array(dtype=wp.vec3),        # Per-rod offsets
    rod_starts: wp.array(dtype=int),         # Rod start indices
    rod_counts: wp.array(dtype=int),         # Points per rod
    num_rods: int,
    output: wp.array(dtype=wp.vec3),
    output_starts: wp.array(dtype=int),
):
    tid = wp.tid()
    # Binary search or linear scan to find which rod this thread belongs to
    # Apply appropriate offset
```

**Simpler Alternative:** Pre-concatenate rod data into single arrays at init time.

### 3.2 Batch Constraint Applications (example.py:588-647)

Similar pattern - consolidate track constraint and concentric constraint kernels.

---

## Phase 4: Medium Priority - CUDA Graph for Rendering

**Impact: MEDIUM | Risk: LOW | Files: `simulation/example.py`**

### 4.1 Add Render Graph Capture

**Current:** `step()` uses CUDA graphs, but `render()` does not

**Optimization:**
```python
def __init__(self, ...):
    # ... existing code ...
    self._render_graph = None
    self._render_graph_valid = False

def render(self):
    if self.use_cuda_graph and not self._render_graph_valid:
        with wp.ScopedCapture(device=self.model.device) as capture:
            self._render_impl()
        self._render_graph = capture.graph
        self._render_graph_valid = True

    if self._render_graph_valid:
        wp.capture_launch(self._render_graph)
    else:
        self._render_impl()
```

**Caveat:** Graph invalidation needed when visualization toggles change (show_directors, show_rod_mesh).

---

## Phase 5: Medium Priority - Memory Access Optimization

**Impact: MEDIUM | Risk: MEDIUM | Files: `kernels/assembly.py`**

### 5.1 Optimize Jacobian Memory Layout

**Current:** Complex indexing via `_warp_jacobian_index(edge, row, col)` causes scattered access

**Analysis needed:** Profile to determine if this is actually a bottleneck. The indexing is:
```python
def _warp_jacobian_index(edge: int, row: int, col: int) -> int:
    return edge * 36 + row * 6 + col
```

This is row-major within each edge's 6x6 block, which should be reasonably cache-friendly.

**Potential optimization:** Transpose jacobian storage to be column-major if column access dominates.

---

## Phase 6: Low Priority - Minor Optimizations

### 6.1 Pre-allocate Constant Warp Objects (example.py)

Cache `wp.vec3()` constants instead of recreating:
```python
# In __init__:
self._zero_offset_wp = wp.vec3(0.0, 0.0, 0.0)

# In step:
# Instead of: zero_offset = wp.vec3(0.0, 0.0, 0.0)
# Use: self._zero_offset_wp
```

### 6.2 GPU Vertex Transformation (example.py:338-343)

Move Python loop for mesh vertex transformation to GPU kernel (one-time init cost).

---

## Files to Modify

| File | Changes |
|------|---------|
| `kernels/collision.py` | Add parallel reduction, parallel corrections, director kernel |
| `simulation/example.py` | Use new kernels, add render graph, batch operations |
| `rod/warp_rod.py` | Update to use parallel kernels |
| `meshing/rod_mesher.py` | Add `update_warp()` method if missing |

---

## Verification Strategy

### Per-Optimization Testing

1. **Parallel kernels:** Compare output values against sequential baseline
   ```bash
   uv run -m newton.examples cosserat_codex --compare-baseline
   ```

2. **Render optimization:** Visual inspection + frame time profiling
   ```bash
   uv run -m newton.examples cosserat_codex --profile-render
   ```

3. **Batched kernels:** Ensure simulation behavior unchanged via `test_final()`

### Performance Measurement

```bash
# Before/after timing comparison
uv run -m newton.examples cosserat_codex --benchmark --frames 1000

# Warp profiling
WARP_PROFILE=1 uv run -m newton.examples cosserat_codex --frames 100
```

### Existing Tests

```bash
uv run --extra dev -m newton.tests.test_examples -k cosserat
```

---

## Implementation Order (Recommended)

1. **Phase 2.1** - GPU director lines (safest, high impact)
2. **Phase 1.1** - Parallel constraint max (isolated change)
3. **Phase 3.1** - Batch kernel launches (moderate refactor)
4. **Phase 1.2** - Parallel corrections (requires careful atomics)
5. **Phase 4.1** - Render CUDA graph (needs invalidation logic)
6. **Phase 2.2** - GPU mesh update (depends on RodMesher)
7. **Phase 5-6** - Lower priority items as time permits

---

## Risk Assessment

| Optimization | Risk | Mitigation |
|-------------|------|------------|
| Parallel corrections with atomics | MEDIUM | Extensive testing, fallback to sequential |
| CUDA graph invalidation | LOW | Conservative invalidation on any toggle |
| Batched kernels | LOW | Maintain per-rod fallback |
| Memory layout changes | MEDIUM | Profile before changing |

---

## Expected Performance Gains

| Optimization | Expected Speedup |
|-------------|------------------|
| Parallel constraint_max | 10-50x for this kernel |
| Parallel corrections | 5-20x for this kernel |
| GPU director lines | Eliminates ~6 transfers/frame/rod |
| Batched launches | Reduces launch overhead ~4-8x |
| Render CUDA graph | Reduces per-frame overhead |

**Overall:** Expect 2-5x improvement in frame rate for typical scenarios with multiple rods.
