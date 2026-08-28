# C++ controller runtime

Runs a Newton controller from C++ with no Python interpreter. Newton captures a
controller's step as a Warp APIC graph and saves it to a `.wrp` file; this
library loads that file, writes inputs, launches it, and reads outputs back.

Newton itself is pure Python — nothing here is built by `pip install newton`, and
`newton.tests` does not exercise it. It is built separately, and only the export
side (`newton.controllers.export_controller_graph`) is covered by the Python test suite.

## Export a controller

```python
from newton.controllers import ControllerJointImpedance, export_controller_graph

controller = ControllerJointImpedance(model, ...)
inputs, outputs = controller.input(), controller.output()

export_controller_graph(controller=controller, inputs=inputs, outputs=outputs, path="joint_impedance")
```

This writes `joint_impedance.wrp` and `joint_impedance_modules/`. Both must ship
together — the modules directory holds the compiled kernels.

The device to replay on is not something you choose here — it is read directly
out of the `.wrp` file's own header, which records whichever device
`export_controller_graph` captured on. A `device="cuda"` capture binds CUDA
device 0 and replays there; a `device="cpu"` capture never touches CUDA. A
graph replays only on the device it was captured for — the compiled kernels in
`joint_impedance_modules/` are architecture-specific (PTX/cubin for CUDA, an
LLVM-JIT'd object file for CPU), so there is no "record once, replay anywhere."
CPU replay resolves those kernels through Warp's LLVM JIT (`warp-clang`),
linked automatically as part of this build — see `FindWarp.cmake`.

## Build

Needs CMake 3.24+, a C++20 compiler, and Newton's environment on the same
machine (the build locates `warp.so` and Warp's headers through it).

```bash
uv sync
cmake -S newton/_src/controllers/cpp -B build/controllers
cmake --build build/controllers
ctest --test-dir build/controllers --output-on-failure
```

Two tests run: `controller_artifacts` captures a graph from Python and records
the torques it produced, then `controller_runtime` loads that graph in C++ and
checks it produces the same values. They are ordinary asserts with a non-zero
exit code — no test framework is fetched.

Point the build at a different environment with `-DWARP_ROOT=...`,
`-DWARP_PYTHON=...`, or `-DNEWTON_PYTHON=...`.

## Use

```cpp
#include <newton/controllers/controller.hpp>

newton::controllers::Controller controller{"joint_impedance.wrp"};
auto input = controller.input();
auto output = controller.output();

input["joint_q"] = {0.35f, -0.62f, 0.18f, 0.9f};
input["joint_q_des"] = {0.0f, -0.3f, 0.5f};

if (!controller.step(input, output, 0.01f)) {
    std::cerr << controller.last_error() << "\n";  // step never throws
    return 1;
}
// output["joint_f"] now holds one torque per controlled DOF
```

`input()` and `output()` return buffers sized from the graph's parameter table,
keyed by field name with the `input.` / `output.` prefix stripped. Every
parameter is addressed as float32, which is why `export_controller_graph`
refuses ports built from anything else. `params()` lists what the graph exposes
along with each size.

## Ports bound to a view

A Python port can be bound to a view of a simulation-sized array
(`inputs.joint_q_des = sim_q_des[indices]`). The exported parameter is
then that whole array, and the indices travel inside the graph, so the buffer
you exchange has the simulation's length rather than the controlled-DOF count:

| Python binding | field length in C++ |
| --- | --- |
| compact array of 2, bound directly | 2 |
| view of 2 elements into an array of 5 | 5 |

Fill every element of such an input — the graph reads the ones the view
addresses. Read back every element of such an output — the graph has written the
addressed ones and left the rest at the values they held when the graph was
captured. A struct may mix both kinds of binding; `params()` tells you which is
which.
