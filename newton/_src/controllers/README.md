# Controllers

> **Status:** **superseded.** The design described below (with `Resource`, `Reference`, and `Block`) is an earlier sketch. The current v0 design uses `Controller` and `ControlGroup` without a Resource/Reference layer — see [`DESIGN_DOC.md`](DESIGN_DOC.md). The content here is preserved for historical context only; do not implement from it.

A binding layer for reusable control blocks. Write a PID, a filter, or a differential-kinematics block once; bind it to any robot's named input and output interfaces. Multiple blocks operate on the same robot without stepping on each other's output writes, and each block owns its own internal state.

See [INTERNALS.md](INTERNALS.md) for how binding works underneath.

---

## Core concepts

- **Resource** — The hardware contract. Constructed from concrete, allocated `input` and `output` struct instances: `Resource(input, output)`. Exposes `resource.inputs` and `resource.outputs`, proxy namespaces that produce References to fields on those structs.
- **Reference** — What `resource.inputs.joint_position` or `resource.outputs.joint_effort[:, 0:3]` returns. A handle to a specific field (or a sliced region of a field) on the hardware contract. References are the library's currency for write-claim tracking.
- **Block** — A compiled, runnable unit. Reads from wherever its author wants (References to hardware-contract fields, or external `wp.array`s passed directly) and writes to specific output targets. Owns any internal state it needs. Runs on its own with `block.run()`.
- **Controller** — A group of blocks operating on one resource. At construction, the library checks that no two blocks' Reference writes overlap. Exposes `run()`, `reset(indices=None)`, and a static `valid_mask`.

---

## Binding a block to a robot

```python
import warp as wp
from controllers import Resource
from controllers.blocks import pid

@wp.struct
class ArmInput:
    joint_position: wp.array(dtype=wp.float32, ndim=2)   # (robot, joint)
    joint_setpoint: wp.array(dtype=wp.float32, ndim=2)
    kp: wp.array(dtype=wp.float32, ndim=2)               # tunable gain

@wp.struct
class ArmOutput:
    joint_effort: wp.array(dtype=wp.float32, ndim=2)

# allocate
input = ArmInput()
input.joint_position = wp.zeros((10, 6), dtype=wp.float32)
input.joint_setpoint = wp.zeros((10, 6), dtype=wp.float32)
input.kp = wp.full((10, 6), 1.0, dtype=wp.float32)

output = ArmOutput()
output.joint_effort = wp.zeros((10, 6), dtype=wp.float32)

# resource bundles the hardware contract
resource = Resource(input, output)

# compile a block
pid_block = pid(
    resource,
    measurement=resource.inputs.joint_position,
    setpoint=resource.inputs.joint_setpoint,
    kp=resource.inputs.kp,
    ki=0.1,                                             # direct binding: plain float
    output=resource.outputs.joint_effort,
)

# run
pid_block.run()
pid_block.reset(indices=reset_indices)
```

The block is runnable on its own — you don't need a Controller for a single block. Tuning is done by writing fields on `input` between runs: `input.kp = new_gain` updates what the kernel sees on the next `run()`.

## Three ways to pass an argument

1. **Reference to a hardware-contract field** — `resource.inputs.X` or `resource.outputs.X`. Tracked by the library. Writes to output References participate in disjoint-writes checks and in `valid_mask`.
2. **Sliced Reference** — `resource.outputs.joint_effort[:, 0:3]`. Same as above, but claims only the slice. Supports basic slicing (slice objects with step, integer indices, negative indices, ellipsis, and combinations). Fancy indexing is not supported.
3. **Direct binding** — any `wp.array` or scalar the user has in hand. Passed to the kernel verbatim; not tracked. The user is responsible for lifetime and for avoiding unintended overlap with other blocks writing to the same external buffer.

A block's author decides which arguments are expected to be References and which can be direct bindings. The user decides at binding time which form to pass.

---

## Composing blocks into a Controller

```python
from controllers import Controller

pid_first_three = pid(
    resource,
    measurement=resource.inputs.joint_position,
    setpoint=resource.inputs.joint_setpoint,
    kp=resource.inputs.kp,
    ki=0.1,
    output=resource.outputs.joint_effort[:, 0:3],
)

pid_last_three = pid(
    resource,
    measurement=resource.inputs.joint_position,
    setpoint=resource.inputs.joint_setpoint,
    kp=resource.inputs.kp,
    ki=0.1,
    output=resource.outputs.joint_effort[:, 3:6],
)

controller = Controller(resource, [pid_first_three, pid_last_three])
controller.run()
controller.reset(indices=reset_indices)
```

At construction, `Controller` collects every block's Reference writes and verifies no two overlap. Overlapping claims raise with a message naming the conflicting blocks and the offending Reference.

Blocks composed in a Controller are semantically independent: any permutation of their run order produces the same result. If two blocks need to exchange data, either fuse them into a single block (multiple kernels inside one block function can share private state) or route the intermediate value through a buffer outside the hardware contract.

---

## Writing a block

A block is a function that takes `resource`, a set of References and/or direct bindings, and returns a compiled `Block`. The block function runs once at compile time: it resolves References to concrete values, allocates whatever private state it needs, and packages everything into a ready-to-run unit.

```python
import warp as wp
from controllers import Block, Reference

@wp.kernel
def _pid_kernel(
    measurement: wp.array(dtype=wp.float32, ndim=2),
    setpoint: wp.array(dtype=wp.float32, ndim=2),
    kp: wp.array(dtype=wp.float32, ndim=2),
    ki: wp.float32,
    integral: wp.array(dtype=wp.float32, ndim=2),
    output: wp.array(dtype=wp.float32, ndim=2),
):
    robot, joint = wp.tid()
    error = setpoint[robot, joint] - measurement[robot, joint]
    integral[robot, joint] = integral[robot, joint] + ki * error
    output[robot, joint] = kp[robot, joint] * error + integral[robot, joint]


@wp.kernel
def _reset_pid_integral(
    integral: wp.array(dtype=wp.float32, ndim=2),
    indices: wp.array(dtype=wp.int32, ndim=1),
):
    i, joint = wp.tid()
    robot = indices[i]
    integral[robot, joint] = 0.0


def pid(resource, *, measurement, setpoint, kp, ki, output, name=None):
    # Resolve references to concrete values; direct bindings pass through.
    measurement_array = resource.resolve(measurement)
    output_array = resource.resolve(output)
    integral = wp.zeros_like(measurement_array)

    # Block-author's own build-time checks.
    assert measurement_array.shape == output_array.shape, (
        f"pid: measurement shape {measurement_array.shape} != "
        f"output shape {output_array.shape}"
    )

    kernel_args = tuple(
        resource.resolve(argument)
        for argument in (measurement, setpoint, kp, ki, integral, output)
    )

    def reset(indices):
        if indices is None:
            indices = wp.array(
                list(range(measurement_array.shape[0])), dtype=wp.int32
            )
        wp.launch(
            _reset_pid_integral,
            dim=(indices.shape[0], integral.shape[1]),
            inputs=[integral, indices],
        )

    return Block(
        kernel_calls=[(_pid_kernel, measurement_array.shape, kernel_args)],
        private={"integral": integral} if name is None else {f"{name}/integral": integral},
        writes=[output] if isinstance(output, Reference) else [],
        reset=reset,
    )
```

What the block author provides to `Block`:

- **`kernel_calls`** — a list of `(kernel, launch_dim, args)` tuples. `launch_dim` is already concrete (`int` or `tuple[int, ...]`), `args` is a tuple of concrete values (arrays or scalars). Nothing dynamic remains for run time.
- **`private`** — a dict of name → `wp.array` for any internal state the block wants to expose. Names are optional in the sense that any naming convention works; blocks that expect to be instantiated more than once typically accept a `name=` kwarg and prefix their private keys with it.
- **`writes`** — a list of References the block writes to via the hardware contract. Used by `Controller` for the disjoint-writes check and by `valid_mask`. Direct-binding writes are not listed here because the library can't track them.
- **`reset`** — optional callable taking `indices`, called when the host invokes `block.reset(...)` or `controller.reset(...)`. The block author decides what reset means for their state. If a block's state never needs resetting, omit this.

Build-time dtype and ndim checks happen for free from each kernel's annotations. Any other check (shape compatibility across arguments, value ranges, etc.) goes directly in the block function — it runs at compile time with real arrays.

---

## Running and resetting

```python
block.run()
block.reset(indices=None)

controller.run()
controller.reset(indices=None)
```

- `indices` is a 1-D `wp.array(dtype=wp.int32)` of robot indices, or `None` to reset every robot.
- After compilation, the concrete `wp.array`s are captured inside the block. Mutations to the contents of those arrays (`input.kp = new_gain`, for example) are visible on the next `run()`. Reassigning a struct field to a different array after compilation is not supported.
- Reset runs outside the `run()` hot path and is not subject to CUDA-graph capture constraints. Block authors can launch whatever kernels they need.

---

## Inspecting private state

Each block exposes a `private` dict:

```python
integrator_state = pid_block.private["integral"]   # wp.array
```

This is the raw `wp.array` — readable, and writeable at the user's risk. Useful for logging during development, asserting in tests, or dumping state to disk for offline analysis.

---

## Validity

`controller.valid_mask` is a `tuple[bool, ...]` aligned with the output struct's field order: `mask[i]` is true if any block in the controller writes (fully or partially) to the *i*-th output field via a Reference. It is fixed at Controller construction time.

For a sequence of Controllers driving one resource, combine their masks once at setup:

```python
combined = tuple(any(bits) for bits in zip(*(c.valid_mask for c in controllers)))
```

Direct-binding writes do not contribute to `valid_mask` because they are not part of the hardware contract.

---

## Package layout

- `controllers/contract.py` — `Resource`, `Reference`.
- `controllers/block.py` — `Block`, `Controller`.
- `controllers/blocks/` — reusable block functions (PID, filters, differential kinematics, etc.).
- `controllers/kernels/` — Warp kernel implementations used by blocks.

---

## Common errors

- **Overlapping Reference writes.** Two blocks in one Controller claim the same output field or overlapping slices of it. Raised at `Controller(...)` construction.
- **Missing allocation.** Any input field a block references must be allocated on the `input` struct before constructing the Resource — the block reads its shape and dtype at compile time.
- **Unknown field.** `resource.inputs.not_a_real_field` raises at access time, naming the valid fields.
- **Type mismatch.** If a resolved argument's dtype or ndim doesn't match the kernel's annotation, the block raises with the kernel and argument name.
