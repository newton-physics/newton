# ChysX

A minimal CUDA physics simulator demonstrating how to plug a custom C++/CUDA
backend into [Newton](https://github.com/newton-physics/newton).

The integrator is just semi-implicit Euler with gravity (no contacts, no
elasticity), but the shape of the API mirrors a "real" engine:

- All physics code lives in this folder under `src/` (CUDA C++).
- It builds into a standalone Python wheel `chysx` via
  `scikit-build-core` + `pybind11`.
- It exposes a single function that accepts **raw CUDA device pointers** so it
  can share GPU buffers with Warp (and therefore with Newton's `State`) with
  zero copies — the same pattern libuipc uses for its `BufferView`.

## Layout

```
ChysX/
├── pyproject.toml           # scikit-build-core + pybind11 build deps
├── CMakeLists.txt           # CUDA + pybind11 build rules
├── src/
│   ├── bindings.cpp         # pybind11 module: ClothMaterial + ClothSimulator
│   ├── memory/              # GPU memory primitives (no domain logic)
│   │   ├── cuda_array.{h,cu}    # RAII paired host (pinned) + device buffer
│   │   └── device_span.h        # non-owning typed view (T*, count)
│   ├── math/                # tiny CUDA-aware Vec/Mat math library
│   │   ├── common.cuh
│   │   ├── vec.cuh
│   │   └── matrix.cuh
│   ├── geometry/            # geometric primitives
│   │   └── triangle_mesh.h
│   └── cloth/               # cloth simulator (material + buffers + kernels)
│       ├── cloth_material.h
│       ├── cloth_buffers.h
│       └── cloth_simulator.{h,cu}
└── chysx/
    └── __init__.py          # Python facade: ClothMaterial, ClothSimulator
```

### Internal: `chysx::CudaArray<T>`

`src/cuda_array.{h,cu}` defines a templated C++ helper used by ChysX
itself (e.g. for staging tables, CPU reference state, or unit tests) —
it is *not* exposed to Python.  Newton continues to share its own GPU
buffers with ChysX through `wp.array.ptr`, so this class never appears
on the bridge.

The class is parameterised on the element type, so callers can write
`CudaArray<float>`, `CudaArray<chysx::math::Vec3f>`, `CudaArray<int>`,
etc., and the byte counts are derived automatically.

Each instance owns four core members, accessible through getters of the
same name:

| member     | meaning                                                  |
|------------|----------------------------------------------------------|
| `cpu_ptr`  | host pointer (page-locked / pinned), `0` if unallocated  |
| `gpu_ptr`  | CUDA device pointer, `0` if unallocated                  |
| `cpu_size` | element count currently allocated on the host            |
| `gpu_size` | element count currently allocated on the device          |

Convenience accessors: `cpu_data()` / `gpu_data()` return typed `T*`,
`cpu_bytes()` / `gpu_bytes()` return raw byte counts, and host-side
`operator[](i)` indexes into the host buffer.

```cpp
#include "memory/cuda_array.h"   // also pulls in memory/device_span.h
#include "math/vec.cuh"

using chysx::CudaArray;
using chysx::math::Vec3f;

CudaArray<Vec3f> particles(1024);   // 1024 Vec3f on both host + device
particles[0] = Vec3f(0.0f, 0.0f, 1.0f);
particles.copy_to_device();         // synchronous H2D
// ... launch kernel using particles.gpu_data() ...
particles.copy_to_host();           // synchronous D2H
```

The two sides can be sized independently (`allocate_host(count)` /
`allocate_device(count)`) or together (`resize(count)`), and
`copy_to_device(stream)` / `copy_to_host(stream)` move data in either
direction.  Host memory is allocated with `cudaMallocHost` so transfers
can run asynchronously on a caller-supplied CUDA stream.

## Build & install

Requirements:

- NVIDIA CUDA Toolkit (tested with 12.8); `nvcc` on `PATH`.
- A C++17 compiler (MSVC 2019+ on Windows, GCC 9+ / Clang on Linux).
- CMake ≥ 3.18.
- A Newton dev environment (`uv sync --extra examples` in the repo root).

From the Newton repository root:

```powershell
# default architecture is sm_120 (RTX 5090); override if you have a different GPU
$env:CHYSX_CUDA_ARCH = "86"  # RTX 30xx
uv pip install --no-build-isolation ./ChysX
```

`--no-build-isolation` reuses the dev venv (which already has Warp + Newton),
which means the resulting `chysx` wheel can be `import`-ed alongside them.

Sanity check:

```powershell
uv run python -c "import chysx; print(chysx.__version__)"
```

## How it talks to Newton

`newton/_src/solvers/chysx/solver_chysx.py` defines `SolverChysX`, a
`SolverBase` subclass that:

1. **At construction**, builds a `chysx.ClothMaterial` from the Newton
   model (gravity, eventually density / Lamé from `model.tri_*`) and
   *copies* it into a `chysx.ClothSimulator` via `set_material(...)`.
2. **Each `step()`**, copies `state_in` to `state_out` (so callers can
   keep double-buffering), then *assigns* the device pointers
   `state_out.particle_q.ptr` / `state_out.particle_qd.ptr` to the
   simulator via `set_external_buffers(...)`, and finally calls
   `sim.step(dt)`.

Because Warp's `wp.array.ptr` returns the same CUDA device pointer that
the underlying `cudaMalloc` produced, the ChysX kernel writes directly
into Newton's particle state — no host round-trip.  Material parameters
are *copied* (values), buffer pointers are *referenced* (views).

## Run the demo

```powershell
uv run python -m newton.examples chysx_freefall
```

This drops a 10×10 cloth patch 2 m above the ground and lets the whole sheet
fall as a rigid plane (the toy integrator has no internal forces, so every
particle accelerates the same way).
