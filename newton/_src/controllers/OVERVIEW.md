# Newton Controllers — Overview

A small, composable library for **control laws** — the blocks that read sensor data and emit joint targets / efforts each step. PID, differential IK, gravity compensation, low-pass filters — anything you'd write between "I have a measurement" and "the actuator gets a setpoint."

This document is a tour. For the full spec, see `DESIGN_DOC.md`.

---

## The two types

**`ControlLaw`** — a single control law. Examples: `ControlLawPID`, `ControlLawDifferentialIK`. Each subclass implements one specific algorithm and declares which named ports it reads from and writes to.

**`Controller`** — a thin composer that owns a list of `ControlLaw`s. It runs them in order each step, zeroing their declared output slots first so multiple laws writing to the same slot accumulate naturally via `+=`.

```python
import newton.controllers as nc

pid = nc.ControlLawPID(...)
controller = nc.Controller([pid])
```

The `Controller` is also where you opt into gradient-tracked allocations (`requires_grad=True`) and pick the Warp device.

---

## Reading inputs and writing outputs

Every port on a `ControlLaw` is a **string** giving the attribute name where that port's array lives on the `input` / `output` object you pass to `step()`. The Controller resolves each name via `getattr` and feeds the array to the kernel.

```python
controller.step(input, output, current_state, next_state, dt)
```

- `input` — any duck-typed object whose attributes hold the read ports (measurement, setpoint, gains, targets…).
- `output` — any duck-typed object whose attributes hold the write ports (joint_f, joint_target_q, …). The Controller zeros the declared output slots at the start of each step.
- `current_state` / `next_state` — `Controller.State` objects holding per-law internal state (e.g. PID integrals). Double-buffered.
- `dt` — timestep in seconds.

`input` and `output` are usually a `types.SimpleNamespace` or a small `@dataclass` you build once at setup. They can hold anything: live simulation arrays from `newton.State` / `newton.Control`, gain arrays, app-specific RL targets — whatever the laws ask for.

```python
from types import SimpleNamespace

input = SimpleNamespace(
    joint_q=state.joint_q,                                  # live sim data
    joint_qd=state.joint_qd,
    setpoint=wp.array([1.0, 2.0, 0.5], dtype=wp.float32),   # your own array
    kp=wp.full(3, 100.0, dtype=wp.float32),                 # your own gains
    ki=wp.zeros(3, dtype=wp.float32),
    kd=wp.full(3, 10.0, dtype=wp.float32),
    integral_max=wp.full(3, float("inf"), dtype=wp.float32),
    setpoint_rate=wp.zeros(3, dtype=wp.float32),
)
output = SimpleNamespace(joint_f=control.joint_f)

controller.step(input, output, cs0, cs1, dt=0.01)
```

The shape matches `newton.actuators.Actuator.step` and `newton.solvers.Solver.step` — explicit data in, explicit data out.

---

## Port forms

Every per-DOF port accepts one of two forms:

| Spec | Meaning | Kernel access |
|---|---|---|
| `"name"` | look up `input.name` (or `output.name`) and use the controller's `indices` array | `getattr(source, name)[indices[i]]` |
| `("name", port_indices)` | same lookup, but use custom `port_indices` for the inner index | `getattr(source, name)[port_indices[i]]` |

The tuple form is for the (less common) case where a port's source array has a different layout than the controller's `indices` — e.g. reading gains from a packed local array while writing outputs to a sparse global one.

**Per-group ports** (per-robot, not per-DOF — e.g. `target_pos` on DiffIK) accept just `"name"`. At step, the array is fetched and validated to be `length == num_robots` with the documented dtype (`wp.vec3` / `wp.quat` / `wp.float32`).

---

## Indices

Every `ControlLaw` takes an `indices: wp.array[wp.uint32]`. This is the controller-level lookup that tells the kernel *which slots* in the output array it writes to. If your robot has 9 DOFs and you want this controller to drive the first 7, `indices = wp.array([0, 1, 2, 3, 4, 5, 6], dtype=wp.uint32)`.

The convention across the codebase is that `len(indices)` = `num_outputs` — the number of slots this law produces per step. Per-DOF ports default to using `indices` as their port_indices.

For coupled controllers (like `ControlLawDifferentialIK`) the indices length also encodes the replication factor: `len(indices) // model_builder.joint_dof_count == R`, the number of robot copies.

---

## Lifecycle

```python
# 1. Build the input/output containers (cheap; do this once at setup).
input  = SimpleNamespace(joint_q=..., kp=..., setpoint=..., ...)
output = SimpleNamespace(joint_f=...)

# 2. Construct one or more ControlLaws. Validation happens here:
#    port specs are normalized, port_indices shapes are checked.
pid = nc.ControlLawPID(
    indices=indices,
    measurement="joint_q",
    measurement_rate="joint_qd",
    setpoint="setpoint",
    setpoint_rate="setpoint_rate",
    kp="kp", ki="ki", kd="kd",
    integral_max="integral_max",
    output="joint_f",
)

# 3. Wrap in a Controller. This is where device allocations happen
#    (PID integrals, DiffIK internal Model + Jacobian buffers, etc.).
controller = nc.Controller([pid])

# 4. Allocate the double-buffered State.
cs0 = controller.state()
cs1 = controller.state()

# 5. Step.
for _ in range(steps):
    controller.step(input, output, cs0, cs1, dt=0.01)
    cs0, cs1 = cs1, cs0

# 6. Reset selected slots (e.g. when an RL episode ends).
mask = wp.array([True, False, True, ...], dtype=wp.bool)
controller.reset(cs0, mask)
```

---

## Output accumulation

The Controller zeros every declared output slot at the start of each `step()`. Each `ControlLaw.compute()` then writes via `+=` into its output array, and the laws run in registration order. Two laws targeting overlapping slots accumulate their contributions — useful for stacking, say, a gravity-compensation term and a PD term into the same `joint_f`.

Multi-output laws (e.g. `ControlLawDifferentialIK` writes both `output_qd` and `output_q`) declare every binding via `outputs()`; the Controller zeros and accumulates each one.

---

## State and reset

A `ControlLaw` may carry internal state — most commonly an integrator term (`ControlLawPID.State.integral`) or a filter history. This lives in the per-law `State` dataclass, allocated by `controller.state()`.

Calls follow a double-buffer pattern: `current_state` is read, `next_state` is written, then the caller swaps. The `Controller` orchestrates this — each law's state is plumbed through to its `compute()` automatically.

Reset writes a per-law `reset_state` (zero-initialized at finalize; user-mutable) into the live state at slots flagged by a boolean mask. `mask` is a `wp.array[wp.bool]` of length `controller.num_outputs`; `mask[i] = True` means "reset slot `i`."

```python
pid.reset_state.integral.fill_(0.0)                    # set the reset target
controller.reset(cs0, mask=wp.array([True, False, True, ...], dtype=wp.bool))
```

Stateless laws (like `ControlLawDifferentialIK`) have `state() → None` and ignore reset.

---

## A complete example: PID

A 3-DOF PD controller (`ki=0`) that drives `joint_q` toward a setpoint and writes the resulting torque into `joint_f`:

```python
import warp as wp
import numpy as np
from types import SimpleNamespace
import newton.controllers as nc

device = wp.get_device()
N = 3

# Inputs and outputs — these are just attribute bags. Live arrays here.
input = SimpleNamespace(
    joint_q=wp.zeros(N, dtype=wp.float32, device=device),
    joint_qd=wp.zeros(N, dtype=wp.float32, device=device),
    setpoint=wp.array([1.0, 2.0, 0.5], dtype=wp.float32, device=device),
    setpoint_rate=wp.zeros(N, dtype=wp.float32, device=device),
    kp=wp.full(N, 100.0, dtype=wp.float32, device=device),
    ki=wp.zeros(N, dtype=wp.float32, device=device),
    kd=wp.full(N, 10.0, dtype=wp.float32, device=device),
    integral_max=wp.full(N, float("inf"), dtype=wp.float32, device=device),
)
output = SimpleNamespace(joint_f=wp.zeros(N, dtype=wp.float32, device=device))

pid = nc.ControlLawPID(
    indices=wp.array(np.arange(N, dtype=np.uint32), device=device),
    measurement="joint_q",
    measurement_rate="joint_qd",
    setpoint="setpoint",
    setpoint_rate="setpoint_rate",
    kp="kp", ki="ki", kd="kd",
    integral_max="integral_max",
    output="joint_f",
)
controller = nc.Controller([pid])

cs0, cs1 = controller.state(), controller.state()
for _ in range(steps):
    controller.step(input, output, cs0, cs1, dt=0.005)
    cs0, cs1 = cs1, cs0
```

---

## A complete example: Differential IK

`ControlLawDifferentialIK` runs a damped-least-squares Jacobian solve per robot to drive a named **site** (a frame attached to a body via `builder.add_site`) toward a target pose. It takes a `newton.ModelBuilder` containing K topologically-identical articulations; the law replicates that template R times internally so `num_robots = K * R`.

```python
builder = newton.ModelBuilder()
link0 = builder.add_link()
link1 = builder.add_link()
j0 = builder.add_joint_revolute(parent=-1, child=link0, axis=wp.vec3(0, 0, 1))
j1 = builder.add_joint_revolute(
    parent=link0, child=link1, axis=wp.vec3(0, 0, 1),
    parent_xform=wp.transform(p=wp.vec3(1.0, 0, 0)),
)
builder.add_articulation([j0, j1], label="arm")
builder.add_site(link1, label="tool",
                 xform=wp.transform(p=wp.vec3(1.0, 0, 0), q=wp.quat_identity()))

N = 2  # dofs in the template
input = SimpleNamespace(
    joint_q=wp.zeros(N, dtype=wp.float32),
    joint_qd=wp.zeros(N, dtype=wp.float32),
    target_pos=wp.array([wp.vec3(2.0, 0.1, 0.0)], dtype=wp.vec3),
    target_quat=wp.array([wp.quat(0.0, 0.0, 0.0, 1.0)], dtype=wp.quat),
    damping=wp.array([0.05], dtype=wp.float32),
    gain=wp.array([1.0], dtype=wp.float32),
)
output = SimpleNamespace(
    output_qd=wp.zeros(N, dtype=wp.float32),
    output_q=wp.zeros(N, dtype=wp.float32),
)

diffik = nc.ControlLawDifferentialIK(
    model_builder=builder,
    indices=wp.array([0, 1], dtype=wp.uint32),
    site="tool",
    measurement="joint_q",
    measurement_rate="joint_qd",
    target_pos="target_pos",
    target_quat="target_quat",
    damping="damping",
    gain="gain",
    output_qd="output_qd",
    output_q="output_q",
)
controller = nc.Controller([diffik])

cs0, cs1 = controller.state(), controller.state()
controller.step(input, output, cs0, cs1, dt=0.01)
```

Each step the controller emits:

- `output.output_qd` — the joint-velocity command from the DLS solve.
- `output.output_q` — `q_current + q_dot * dt`, a position target one step ahead. Useful when feeding a downstream PD that tracks position targets (e.g. MuJoCo's `joint_target_q` actuator).

For a full interactive demo with four parallel Franka Pandas, four draggable 6DOF gizmos, and MuJoCo's built-in joint-PD wired to the controller's output, see `newton/examples/controllers/example_diff_ik_panda.py`.

---

## Differentiability

`Controller(..., requires_grad=True)` propagates gradient tracking to every internal allocation (PID's integrals, DiffIK's replicated Model and Jacobian buffers). User-supplied arrays carry their own `requires_grad`. The module is tape-agnostic — wrap your code in `wp.Tape()` externally.

- `ControlLawPID` is fully differentiable.
- `ControlLawDifferentialIK` is tape-safe but **forward-only through the DLS solve**. The solve kernel uses `wp.tile_cholesky` / `wp.tile_cholesky_solve`; their adjoints are advertised but non-functional in Warp 1.14.0 (backward returns zero gradients). The solve is therefore marked `enable_backward=False`. Every other kernel in the chain is autograd-able by default. Revisit when upstream Warp fixes tile-cholesky backward.

---

## Where things live

- `newton/_src/controllers/control_law.py` — `ControlLaw` abstract base.
- `newton/_src/controllers/controller.py` — `Controller` composer.
- `newton/_src/controllers/utils.py` — port normalization (`_normalize_port`, `_normalize_per_group_port`) and step-time resolution (`_resolve_input_array`, `_resolve_per_group_array`).
- `newton/_src/controllers/impl/controller_pid.py` — `ControlLawPID`.
- `newton/_src/controllers/impl/controller_diff_ik.py` — `ControlLawDifferentialIK`.
- `newton/_src/controllers/DESIGN_DOC.md` — full spec (cross-cutting decisions, subclassing guide, planned controllers).
- `newton/examples/controllers/` — runnable demos.
- `newton/tests/test_controllers.py` — analytical and integration tests.

Public surface (`from newton.controllers import …`):
`Controller`, `ControlLaw`, `ControlLawPID`, `ControlLawDifferentialIK`.

---

## Writing your own ControlLaw

Subclass `ControlLaw`, decorate kernels with `@wp.kernel`, and implement:

- `__init__(**ports)` — validate and store port specs (use `_normalize_port` / `_normalize_per_group_port` from `utils.py`). Cheap CPU-side work only — device-buffer allocation happens in `finalize()`.
- `finalize(device, num_outputs, requires_grad=False)` — allocate private device buffers and `self.reset_state`.
- `state(num_outputs, device, requires_grad=False)` — allocate a fresh `State`, or return `None` if stateless.
- `is_stateful()`, `is_graphable()` — small predicates the `Controller` reads.
- `outputs()` — return `list[(attr_name, port_indices)]` so the `Controller` knows what to zero.
- `compute(input, output, cur_state, nxt_state, dt)` — fetch port arrays via `_resolve_input_array` / `_resolve_per_group_array`, then launch kernels.
- `reset(state, mask)` — if stateful, write `self.reset_state` into `state` at masked slots.

`ControlLawPID` and `ControlLawDifferentialIK` are the canonical references — the former for independent per-DOF laws, the latter for coupled / per-robot laws with an internal Newton Model.
