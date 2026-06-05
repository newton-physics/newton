# Newton Controllers — Design Doc

A library for composable control blocks. A `ControlLaw` is a single, runnable control law (PID, differential IK, gravity comp, …). A `Controller` is a composer that wraps one or more `ControlLaw`s and orchestrates the per-step zero / compute sequence.

Controllers typically run **before** actuators in a simulation step: a `ControlLaw` produces a desired joint position, velocity, or force; downstream actuators turn that target into effort.

**Data is separated from functions.** A `ControlLaw` never owns the live arrays it reads or writes; it stores only *attribute names* declaring where to look. At step time, the caller passes in an `input` object (read ports) and an `output` object (write ports), and the law resolves each name via `getattr(input, name)` / `getattr(output, name)`. The signature is `Controller.step(input, output, current_state, next_state, dt)` — mirroring `Actuator.step(sim_state, sim_control, …)` and the various `Solver.step(state_in, state_out, control, contacts, dt)`. `input` and `output` are duck-typed user objects (`SimpleNamespace`, dataclasses, or even `newton.State`/`newton.Control` when the attribute names happen to match).

---

## Core concepts

- **ControlLaw** — Abstract base for a single control law. Subclass with prefix-first naming: `ControlLawPID`, `ControlLawFilter`, `ControlLawGravityComp`. Lifecycle methods: `finalize(device, num_outputs)`, `state(...)`, `is_stateful()`, `is_graphable()`, `outputs()`, `compute(input, output, state, next_state, dt)`. Holds a nested `State` dataclass.
- **Controller** — Composer of one or more `ControlLaw`s. Owns step / reset / state orchestration.

Multiple `ControlLaw`s may bind to overlapping output slots; their contributions accumulate via `+=` directly into the output array (see *Output accumulation*).

---

## Two ControlLaw flavors

ControlLaws fall into two categories, distinguished by whether each output depends on its sibling DOFs. The base class is shared, but the underlying kernel launch may be a different shape.

### Independent per-DOF ControlLaws

Output `i` depends only on input `i` plus per-DOF parameters. Examples: `ControlLawPID`, low-pass filter, saturation, feedforward.

- Construction needs only the bindings + an `indices` array of global DOF indices.
- Kernel launches 1D with `dim=len(indices)`; `i = wp.tid()`.
- No Newton `Model` required.

### Coupled / structural ControlLaws

Each robot's outputs depend on the full state of that robot. Examples: `ControlLawDifferentialIK`, `ControlLawGravityComp`, `ControlLawDifferentialDrive`, `ControlLawHolonomic`, `ControlLawOperationalSpace`.

- Articulated ControlLaws (`ControlLawDifferentialIK`, `ControlLawGravityComp`, etc.) take a `model: newton.Model` containing one or more **topologically identical** articulations (`K = model.articulation_count`). The K articulations must share DOF count, link/joint count, and joint types; they may differ in physical parameters (mass, inertia, friction, joint limits). The ControlLaw reads `dofs_per_robot = model.joint_dof_count // K` directly from the Model at `__init__` time.
- Mobile-base ControlLaws (`ControlLawDifferentialDrive`, `ControlLawHolonomic`) take raw geometry instead of a Model. Wheel-base geometry isn't `Model` state in Newton.
- For articulated ControlLaws: `len(indices) % model.joint_dof_count == 0`. The replication count is `R = len(indices) // model.joint_dof_count`, and the ControlLaw internally tiles the user-supplied Model `R` times at `finalize()`. `num_robots = K * R`. The K=1 case is the single-robot template; K>1 lets the user supply categorical variants (e.g. for RL randomization) which are replicated across the batch.
- For mobile ControlLaws: `len(indices) % wheels_per_robot == 0`. `num_robots = len(indices) // wheels_per_robot`.
- Layout convention. After replication, robot `r` is variant `r % K` from replication `r // K`. So with `K=4, R=3` the robots are `[v0, v1, v2, v3, v0, v1, v2, v3, v0, v1, v2, v3]`. Per-group input arrays (`target_pos`, `target_quat`, `damping`, …) of length `num_robots` follow this layout.
- For both flavors: `indices[r * outputs_per_robot + j]` is the global output index of robot `r`'s local slot `j` (where `outputs_per_robot = dofs_per_robot` for articulated or `wheels_per_robot` for mobile). Robots are contiguous in the flat layout.
- Kernel launches 2D with `dim=(num_robots, outputs_per_robot)`; inside, `robot, local_slot = wp.tid()`, and the flat output index is `robot * outputs_per_robot + local_slot`.

`Controller` does not need to know which flavor a ControlLaw is. Each ControlLaw declares its bound output arrays + indices via `outputs()`; the Controller's zero pass walks the union of all ControlLaws' output destinations.

---

## The unified port form

Every input/output port that addresses per-DOF data is a **string attribute name** that is resolved against the `input` (read ports) or `output` (write ports) object at step time. Ports accept one of two forms:

| Form | Meaning | Kernel access |
|---|---|---|
| `"attr_name"` (bare str) | use the controller-level `indices` as the lookup | `getattr(source, attr_name)[indices[i]]` |
| `("attr_name", port_indices)` | tuple; use `port_indices` as the lookup | `getattr(source, attr_name)[port_indices[i]]` |

For many controllers, the indices of all inputs will align, and the bare string is enough. The tuple form exists when a port's source array uses a different layout than the controller's `indices`.

### Validation

| Stage | Form | Check |
|---|---|---|
| `__init__` | bare `str` | `isinstance(spec, str)` |
| `__init__` | `(str, port_indices)` | `port_indices.shape == indices.shape` |
| step | both | `getattr(source, attr_name)` exists and is a `wp.array`. Shape/dtype mismatches fail at the kernel launch with a precise Warp error. |

The user picks the attribute names freely; the same value can come from any container (a `SimpleNamespace`, a custom dataclass, or even `newton.State` itself if its fields match the chosen names).

### Per-group ports

Some controllers have ports keyed by *robot index*, not per-DOF index — e.g. `ControlLawDifferentialIK`'s `target_pos` (one 3D target per robot). These also use strings:

- The port spec is just `"attr_name"` (no tuple form).
- At step, the kernel does `getattr(input, attr_name)[robot]` and the resolver checks `arr.shape == (num_robots,)` plus the documented dtype (`wp.vec3` / `wp.quat` / `wp.float32`).

### One-big-array example

Measurement, setpoint, and output all live on a single `state` object:

```python
pid = nc.ControlLawPID(
    indices=output_indices,
    measurement=("x", measurement_indices),
    measurement_rate=("x", measurement_rate_indices),
    setpoint=("x", setpoint_indices),
    setpoint_rate=("x", setpoint_rate_indices),
    kp="kp",                          # bare str: looked up via the controller's own `indices`
    ki="ki",
    kd="kd",
    integral_max="integral_max",
    output=("x", output_indices),
)

# At step time:
input  = SimpleNamespace(x=state.x, kp=kp_array, ki=ki_array, kd=kd_array, integral_max=integral_max_array)
output = SimpleNamespace(x=state.x)
controller.step(input, output, cs0, cs1, dt)
```

The same `wp.array` reference appears under one attribute (`input.x` / `output.x`); different `port_indices` arrays disambiguate which slots each port reads/writes.

---

## Lifecycle

```python
import warp as wp
import numpy as np
from types import SimpleNamespace
import newton
import newton.controllers as nc

N = 60                  # 10 robots * 6 DOFs
N_global = 200          # total DOFs in the simulator
dof_indices = wp.array(np.arange(N, dtype=np.uint32))    # this ControlLaw's output slots

# 1. Construct a ControlLaw. Every port is a string giving the attribute name
#    on the input/output object passed to step(). Bare string defaults the
#    port indices to the ControlLaw-level `indices`; tuple form takes custom
#    port_indices for ports whose source layout differs.
pid = nc.ControlLawPID(
    indices=dof_indices,
    measurement="joint_q",
    measurement_rate="joint_qd",
    setpoint="joint_target_pos",
    setpoint_rate="joint_target_vel",
    kp="kp",
    ki="ki",
    kd="kd",
    integral_max="integral_max",
    output="joint_target_force",
)

# 2. Compose into a Controller. Controller validates that every ControlLaw
#    agrees on num_outputs, calls finalize() on each (allocating per-law
#    private buffers + the public reset_state), and collects every law's
#    outputs() into a flat list to be zeroed at the start of each step.
controller = nc.Controller([pid])

# 3. Allocate state pair (double buffer).
state_0 = controller.state()
state_1 = controller.state()

# 4. Build the input/output containers. Plain duck-typed objects — fields
#    are whatever the ControlLaws ask for. Live sim arrays can come from
#    newton.State / newton.Control; gains and per-step targets can be your
#    own arrays. Anything resolvable by getattr works.
arm_in = SimpleNamespace(
    joint_q          = wp.zeros(N_global, dtype=wp.float32),
    joint_qd         = wp.zeros(N_global, dtype=wp.float32),
    joint_target_pos = wp.zeros(N_global, dtype=wp.float32),
    joint_target_vel = wp.zeros(N_global, dtype=wp.float32),
    kp               = wp.full(N, 50.0, dtype=wp.float32),
    ki               = wp.full(N,  1.0, dtype=wp.float32),
    kd               = wp.full(N,  5.0, dtype=wp.float32),
    integral_max     = wp.full(N, float("inf"), dtype=wp.float32),
)
arm_out = SimpleNamespace(joint_target_force=wp.zeros(N_global, dtype=wp.float32))

# 5. Step loop.
for _ in range(steps):
    controller.step(arm_in, arm_out, state_0, state_1, dt=0.005)
    state_0, state_1 = state_1, state_0

    # ... actuators, stepping sim, etc etc ...

# 6. Reset (bool mask, length len(indices)).
controller.reset(state_0, mask=reset_mask)
```

A `ControlLaw`'s `__init__` validates port specs and stashes them as `(attr_name, port_indices)` pairs; it does not allocate device buffers. `Controller.__init__` validates all ControlLaws agree on `num_outputs`, calls `control_law.finalize(device, num_outputs)` on each (allocating per-ControlLaw private buffers and the public `control_law.reset_state`), and collects every law's `outputs()` bindings into a flat list to be zeroed at the start of each step.

---

## Output accumulation (direct write)

`Controller` zeros all output destinations before any ControlLaw's `compute()` is called. At the start of each `step()`, it walks the flat list of `(attr_name, port_indices)` bindings collected from every ControlLaw's `outputs()`, resolves each `attr_name` against the passed-in `output` object once, and launches a kernel that writes zero into `output_array[port_indices[i]]`. Two ControlLaws binding overlapping slots will have those slots zeroed twice.

Each `ControlLaw.compute()` then writes directly into its bound output array(s) using `+=`:

```python
# Inside a per-DOF kernel:
i = wp.tid()
out_idx = output_indices[i]
output_array[out_idx] += control_law_contribution
```

ControlLaws run **serially** (sequential `wp.launch` calls, made much more efficient by graphing). Two ControlLaws with overlapping output indices simply accumulate; no atomic_add is needed.

**Composition semantics.** `ControlLawPID + ControlLawGravityComp + ControlLawFeedforward` all writing to the same `joint_target_force` array produce the sum of their contributions. There are no overlap checks; users compose at their own risk.

**Multi-output ControlLaws** (e.g. `ControlLawDifferentialIK` writes both `joint_target_qd` and `joint_target_q`) declare multiple `(output_array, output_indices)` bindings via `outputs()`. The framework treats each binding equivalently for zero / accumulate purposes.

---

## Reset semantics

Reset is a ControlLaw-defined operation that updates the live State from a per-ControlLaw "reset target" State. Each ControlLaw has a public attribute `reset_state: ControlLaw.State`, allocated at `finalize()` and zero-initialized. The user mutates `control_law.reset_state` to customize what reset writes; they can do so any time after `finalize()` and before the next reset call.

**Mask shape.** `mask` is a `wp.array[wp.bool]` of length equal to the ControlLaw's `num_outputs` — the shared outer length of every output binding the ControlLaw declares via `outputs()`. `mask[i] = True` means "reset output slot `i`." For multi-output ControlLaws, *all* of the ControlLaw's output arrays must share the same outer length (e.g., a ControlLaw with a `wp.array[wp.vec3]` position output and a `wp.array[wp.quat]` orientation output must size both arrays to the same length); the single mask then refers to corresponding slots across all outputs.

**Controller-level invariant.** A `Controller` requires every ControlLaw to share `num_outputs`; this is validated at `Controller.__init__`. The Controller-wide mask passed to `controller.reset(state, mask)` has that shared length.

**User-facing call:**

```python
controller.reset(state, mask)
```

- `state` is a `Controller.State`.
- `mask` is a `wp.array[wp.bool]` of length `controller.num_outputs`.

To customize what reset writes, mutate the ControlLaw's stash ahead of time:

```python
pid = nc.ControlLawPID(...)
controller = nc.Controller([pid])           # finalize allocates pid.reset_state (zeros)
pid.reset_state.integral.fill_(0.5)         # bias the reset target

controller.reset(state_0, mask=reset_mask)  # masked slots → 0.5
```

**Slot, not necessarily one float.** Each entry in `mask` refers to *one output slot*. The chunk of state associated with a slot is ControlLaw-defined: it may be a single scalar (e.g., PID's `integral[i]`), a `wp.vec3` (a per-robot position), a `wp.quat` (a per-robot orientation), or a wider buffer (e.g., a filter that holds K floats per slot). The ControlLaw's `reset` implementation knows the stride and writes the right thing.

**How it works.** `Controller.reset(state, mask)` walks `(control_law, sub_state)` pairs and calls `control_law.reset(sub_state, mask)` for each non-None sub_state. The ControlLaw's kernel iterates over `num_outputs` threads and reads `mask[i]` directly — no indirection through any per-ControlLaw index array, because the mask is already in the ControlLaw's local frame.

The same kernel is callable from C++ directly with the raw `(state_field, reset_state_field, mask)` array pointers — there is no Python-only sugar in the kernel-level contract.

---

## State and the double-buffer swap

Every `ControlLaw` defines a nested `State` dataclass holding whatever internal buffers the control law needs (e.g. PID integrals, filter history). `ControlLaw.is_stateful()` returns true if it holds any. `ControlLaw.state(num_outputs, device)` allocates a fresh `State`.

`Controller.State` composes per-ControlLaw states by index. `controller.state()` returns a composed state with every ControlLaw's state already allocated.

```python
state_0 = controller.state()
state_1 = controller.state()
for _ in range(steps):
    controller.step(input, output, state_0, state_1, dt=dt)
    state_0, state_1 = state_1, state_0
```

The Controller reads from `state_0` and writes to `state_1` on each step. After step, the caller swaps. Reset uses a bool mask; see *Reset semantics* above.

---

## Differentiability

`Controller.__init__` accepts a single `requires_grad: bool = False` flag. It is the **only** site where gradient support is configured; the value propagates into every ControlLaw's `finalize()` and `state()` call and from there into every internally-allocated buffer (PID's `integral` and `reset_state`, DiffIK's replicated `Model`, internal `State`, `_jacobian`, `_qd_target_local`, …).

User-provided input arrays (`measurement`, `target_pos`, `kp`, …) carry their own `requires_grad` — the ControlLaws don't own those allocations.

Kernels mostly use the default `@wp.kernel` decorator, which supports Warp autograd. The module does not manage a `wp.Tape`; the caller (e.g. Isaac Lab) wraps the relevant block with `wp.Tape()` externally and calls `tape.backward(loss)` on its own. This mirrors `newton.actuators.Actuator.step`, which is also tape-agnostic.

**Per-ControlLaw status:**

- `ControlLawPID`: fully differentiable. Gradients flow from `output` back through `measurement`, `setpoint`, `kp`, `ki`, `kd`, etc.
- `ControlLawDifferentialIK`: tape-safe, forward-only through the solve. The compute chain is split into per-element kernels (gather, build site Jacobian, build DLS matrix, q_dot back-projection, accumulate) plus a single tile-Cholesky solve kernel. Every kernel except the solve is autograd-able by default. The solve uses `wp.tile_cholesky` + `wp.tile_cholesky_solve` — the tile primitives' docstrings claim registered adjoints, but the backward path is non-functional in Warp 1.14.0 (verified directly: standalone test gives correct forward but zero gradients for both A and the rhs). The solve kernel is therefore marked `enable_backward=False`; gradients propagate from the loss back to `output_qd` and stop at the solve. Useful for RL pipelines that wrap a whole sim in `wp.Tape` without needing IK gradients; not yet usable for end-to-end diff-physics through the solve. Revisit when upstream `wp.tile_cholesky` backward is fixed.

Users with mixed-grad needs (some ControlLaws grad-tracked, others not) split into multiple `Controller`s; there is no per-ControlLaw override.

---

## Subclassing a ControlLaw

```python
class ControlLaw:
    @dataclass
    class State:
        """Pure data container. Subclasses declare their fields (e.g.
        integral arrays, history buffers). No methods — reset is on ControlLaw."""

    # Set at finalize() to a zero-initialized State. Users mutate this attribute
    # to customize what subsequent reset() calls write. Stateless ControlLaws
    # leave this as None.
    reset_state: ControlLaw.State | None

    def __init__(self, **ports):
        """Validate every port spec, normalize each to (attr_name, port_indices)
        via _normalize_port (or just attr_name via _normalize_per_group_port),
        stash on self. Does NOT allocate device buffers — finalize() does that.

        Subclasses declare which kwargs they accept; missing required ports
        raise here, unknown ports raise here, and port_indices shape
        mismatches raise here. The live array shape/dtype is checked at
        step time (the array doesn't exist until then). Scalar value-range
        checks (e.g. ``kp >= 0``) are NOT performed at this layer — they
        would force a synchronous device-to-host copy. Such checks belong
        in a config layer above the ControlLaw."""

    def finalize(self, device: wp.Device, num_outputs: int) -> None:
        """Allocate device-side private buffers (e.g. internal Model +
        State + Jacobian buffers for coupled ControlLaws) AND the
        reset_state (zero-initialized via self.state(num_outputs, device)
        for stateful ControlLaws). Called by Controller after
        construction. num_outputs == len(indices)."""

    def state(self, num_outputs: int, device: wp.Device) -> ControlLaw.State | None:
        """Allocate a fresh State, or None if stateless."""

    def is_stateful(self) -> bool: ...
    def is_graphable(self) -> bool: ...

    def outputs(self) -> list[tuple[str, wp.array[wp.uint32]]]:
        """Return the (output_attr_name, output_port_indices) bindings this
        ControlLaw writes to. Controller collects these from every
        ControlLaw, resolves each attr_name against the `output` arg of
        step() once per step, and zeros the listed slots before any
        compute() runs. Most ControlLaws return a single binding;
        multi-output ControlLaws (e.g. ControlLawDifferentialIK) return
        multiple."""

    def compute(
        self,
        input,                                     # any object with the read-port attributes
        output,                                    # any object with the write-port attributes
        state: ControlLaw.State | None,
        next_state: ControlLaw.State | None,
        dt: float,
    ) -> None:
        """Resolve port arrays via getattr(input, name) / getattr(output, name),
        then launch kernels: read live data, write `+=` into outputs,
        populate next_state. Called by Controller.step. The device is
        fixed at finalize() time, so compute() does not take one."""

    def reset(
        self,
        state: ControlLaw.State,
        mask: wp.array[wp.bool],
    ) -> None:
        """Update `state` from `self.reset_state` where `mask` is True.
        `mask` is a bool array of length num_outputs (the shared outer
        length of every binding returned by outputs()). `mask[i] = True`
        means "reset slot i." The implementation typically launches a
        kernel of num_outputs threads doing a direct mask[i] check.

        Stateless ControlLaws leave this as a no-op (or don't override)."""
```

The base class is neutral about the per-DOF vs. coupled distinction. Subclasses decide whether to take a `model=` kwarg, what their `indices`-length divisibility constraint is, and what kernel launch dimensionality to use.

### Example: using ControlLawPID (independent flavor)

Reference implementation: `newton/_src/controllers/impl/controller_pid.py`.

```python
import warp as wp
import numpy as np
from types import SimpleNamespace
import newton.controllers as nc

device = wp.get_device()
N = 6                                                            # DOFs this ControlLaw manages
indices = wp.array(np.arange(N, dtype=np.uint32), device=device)

# Construct the law: every port is a string attribute name.
pid = nc.ControlLawPID(
    indices=indices,
    measurement="measurement",
    measurement_rate="measurement_rate",
    setpoint="setpoint",
    setpoint_rate="setpoint_rate",
    kp="kp",
    ki="ki",
    kd="kd",
    integral_max="integral_max",
    output="output",
)
controller = nc.Controller([pid])

# Read/write containers (any duck-typed object works).
input = SimpleNamespace(
    measurement=wp.zeros(N, dtype=wp.float32, device=device),
    measurement_rate=wp.zeros(N, dtype=wp.float32, device=device),
    setpoint=wp.zeros(N, dtype=wp.float32, device=device),
    setpoint_rate=wp.zeros(N, dtype=wp.float32, device=device),
    kp=wp.full(N, 50.0, dtype=wp.float32, device=device),
    ki=wp.full(N,  1.0, dtype=wp.float32, device=device),
    kd=wp.full(N,  5.0, dtype=wp.float32, device=device),
    integral_max=wp.full(N, float("inf"), dtype=wp.float32, device=device),
)
output = SimpleNamespace(output=wp.zeros(N, dtype=wp.float32, device=device))

state_0, state_1 = controller.state(), controller.state()

# Step loop:
for _ in range(steps):
    controller.step(input, output, state_0, state_1, dt=0.005)
    state_0, state_1 = state_1, state_0

# Reset some DOFs to the ControlLaw's reset_state values:
#   (pid.reset_state.integral was zero-initialized; mutate it for non-zero reset)
mask = wp.array([True, False, True, False, True, False], dtype=wp.bool, device=device)
controller.reset(state_0, mask=mask)
```

### Example: using ControlLawDifferentialIK (coupled flavor)

Reference implementation: `newton/_src/controllers/impl/controller_diff_ik.py`.

The ControlLaw takes a `newton.ModelBuilder` containing K topologically-identical articulations (K ≥ 1), replicates it R times internally at `finalize()` via `ModelBuilder.replicate`, and runs a damped-least-squares Jacobian solve per robot at every step. Stateless. Drives a **site** (a named frame attached to a body, added to the builder via `builder.add_site(...)`) to a target pose.

```python
import warp as wp
import newton
import newton.controllers as nc

# Build a single-robot 2-DOF planar arm template (K=1). Each link is 1 unit long.
builder = newton.ModelBuilder()
link0 = builder.add_link()
link1 = builder.add_link()
j0 = builder.add_joint_revolute(parent=-1, child=link0, axis=wp.vec3(0.0, 0.0, 1.0))
j1 = builder.add_joint_revolute(parent=link0, child=link1, axis=wp.vec3(0.0, 0.0, 1.0),
                                parent_xform=wp.transform(p=wp.vec3(1.0, 0.0, 0.0)))
builder.add_articulation([j0, j1], label="arm")

# Define a "tool" site at the tip of link1. The ControlLaw looks up both the EE
# link (= link1) and the body-frame offset xform from the builder by this label.
builder.add_site(link1, label="tool",
                 xform=wp.transform(p=wp.vec3(1.0, 0.0, 0.0), q=wp.quat_identity()))

device = wp.get_device()
N = 2                                                            # DOFs (one robot, two joints)
indices = wp.array([0, 1], dtype=wp.uint32, device=device)

# Target pose is the site's world-frame pose. At q=[0,0] the site is at (2, 0, 0).
diffik = nc.ControlLawDifferentialIK(
    model_builder=builder,
    indices=indices,
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

input = SimpleNamespace(
    joint_q=wp.zeros(N, dtype=wp.float32, device=device),
    joint_qd=wp.zeros(N, dtype=wp.float32, device=device),
    target_pos=wp.array([wp.vec3(2.0, 0.1, 0.0)], dtype=wp.vec3, device=device),
    target_quat=wp.array([wp.quat(0.0, 0.0, 0.0, 1.0)], dtype=wp.quat, device=device),
    damping=wp.array([0.05], dtype=wp.float32, device=device),
    gain=wp.array([1.0], dtype=wp.float32, device=device),
)
output = SimpleNamespace(
    output_qd=wp.zeros(N, dtype=wp.float32, device=device),
    output_q=wp.zeros(N, dtype=wp.float32, device=device),
)

state_0, state_1 = controller.state(), controller.state()
for _ in range(steps):
    controller.step(input, output, state_0, state_1, dt=0.01)
    state_0, state_1 = state_1, state_0
```

Internally the compute step is: gather `joint_q`/`joint_qd` → `eval_fk` → `eval_jacobian` → in-kernel 6x6 Cholesky DLS solve per robot (converts Newton's COM-frame Jacobian to a site-frame Jacobian via `omega × (site - COM)`) → accumulate `q_dot` and integrate `q = q_current + q_dot * dt`.

---

## Controller

```python
class Controller:
    def __init__(self, control_laws: list[ControlLaw],
                 requires_grad: bool = False,
                 device: wp.Device | None = None):
        """Validate that every ControlLaw agrees on num_outputs (the outer
        length of every binding their outputs() return), finalize() each
        (allocating per-law buffers + reset_state), collect every law's
        outputs() into a flat list to be resolved + zeroed at step time."""

    def is_stateful(self) -> bool: ...
    def is_graphable(self) -> bool: ...

    def state(self) -> Controller.State | None:
        """Allocate composed state with one entry per stateful ControlLaw."""

    def step(self, input, output, current_state, next_state, dt: float) -> None:
        """1. Resolve every output binding's attr_name against `output` once,
              and zero each one's declared slots. Two ControlLaws binding
              overlapping slots will have those slots zeroed twice —
              harmless, and avoids any wp.array-identity comparison.
           2. For each ControlLaw (in registration order), call
              compute(input, output, ...) which fetches its own port
              arrays via getattr and += writes into its outputs."""

    def reset(self, state: Controller.State, mask: wp.array[wp.bool]) -> None:
        """For each (control_law, sub_state) pair where sub_state is not
        None, call control_law.reset(sub_state, mask). No framework-level
        interpretation of `mask` — each ControlLaw handles it according
        to its own state layout."""
```

```python
@dataclass
class Controller.State:
    control_law_states: list[ControlLaw.State | None]
```

`Controller.State` is a pure data container; reset is driven from `Controller.reset(state, mask)` (which has access to each ControlLaw's `reset_state`).

---

## Where this fits in the simulation step

```python
controller = nc.Controller([pid, diff_ik])
actuator = newton.actuators.Actuator(controller=..., ...)

state_0 = controller.state(); state_1 = controller.state()
act_state_0 = actuator.state(); act_state_1 = actuator.state()

for _ in range(steps):
    # 1. ControlLaws run first. Their outputs typically land in arrays the
    #    actuator reads from (e.g. joint_target_force, joint_target_qd).
    controller.step(ctrl_input, ctrl_output, state_0, state_1, dt=dt)

    # 2. Actuator translates target → joint effort.
    actuator.step(sim_state, sim_control, act_state_0, act_state_1, dt=dt)

    # 3. Physics solver.
    solver.step(sim_state, sim_state_next, sim_control, contacts, dt)

    state_0, state_1 = state_1, state_0
    act_state_0, act_state_1 = act_state_1, act_state_0
```

The user can point `ctrl_output`'s fields at the *same* `wp.array`s that `sim_control` exposes (e.g. `joint_target_q`, `joint_f`), forming the bridge from controllers to actuators. That sharing is up to the user — the controllers module does not introspect Newton sim objects.

---

## Public API surface

Per AGENTS.md, examples and docs must not import from `newton._src`. The internal package is `newton/_src/controllers/`; the public shim is `newton/controllers.py`:

```python
# newton/controllers.py
from ._src.controllers import (
    ControlLaw,
    Controller,
    ControlLawPID,
    ControlLawDifferentialIK,
)

__all__ = [
    "Controller",
    "ControlLaw",
    "ControlLawDifferentialIK",
    "ControlLawPID",
]
```

Users write `from newton.controllers import Controller, ControlLawPID, ControlLawDifferentialIK`.

---

## v0 scope

**Framework.**

- `ControlLaw` base class with `finalize`, `state`, `is_stateful`, `is_graphable`, `outputs`, `compute`.
- `Controller` composer with `state`, `step`, `reset`, `is_stateful`, `is_graphable`, plus the upfront-zero pass machinery.
- `newton/controllers.py` public shim.
- Helpers in `newton/_src/controllers/utils.py`:
  - `_normalize_port(spec, control_law_indices, name)` — validate `str` or `(str, port_indices)` at `__init__`, return `(attr_name, port_indices)`.
  - `_normalize_per_group_port(spec, name)` — validate the per-group `str` spec at `__init__`.
  - `_resolve_input_array(source, attr_name, name)` — step-time `getattr` with a type check.
  - `_resolve_per_group_array(source, attr_name, num_robots, dtype, name)` — step-time `getattr` with the per-group shape + dtype check (DiffIK's `target_pos`, `target_quat`, etc.).
  - Replication uses :meth:`newton.ModelBuilder.replicate` directly at `finalize()` time; no separate helper. K-articulation homogeneity is asserted via `joint_dof_count % articulation_count == 0` in the consuming ControlLaw's `__init__` (necessary check; users are responsible for matching link/joint counts across articulations).
- For each shipped ControlLaw: math-correctness tests, integral / state accumulation across the `state_0` / `state_1` swap, masked reset, accumulation when multiple ControlLaws overlap, one-big-array binding sanity tests.

**Implementation order.**

1. `ControlLawPID` — proves independent-per-DOF flavor + the framework end-to-end (direct-write outputs, double-buffer state, mask reset).
2. `ControlLawDifferentialIK` — proves the coupled flavor: K-articulation Model + internal replication, multi-output (`q̇` and `q`), `newton.eval_fk` + `newton.eval_jacobian` reuse, in-kernel small-system Cholesky.
3. `ControlLawDifferentialDrive` — first mobile ControlLaw; proves `wheels_per_robot` divisibility convention.
4. `ControlLawHolonomic` — second mobile; same family.
5. *(gated on Newton's upcoming inverse-dynamics function)* `ControlLawGravityComp`, `ControlLawJointImpedance`, `ControlLawOperationalSpace`.

**Out of scope for v0.**

- USD parsing.
- CUDA-graph capture testing.
- `ModelBuilder.add_controller` analog.
- Nullspace projection (joint centering, joint-limit avoidance) for `ControlLawDifferentialIK` — compose a separate ControlLaw later if needed.
- Multiple end-effectors per `ControlLawDifferentialIK` instance — users compose two instances under one `Controller`.

---

## ControlLaws to design

> Per-ControlLaw specs that still need decisions before implementation.

### `ControlLawGravityComp`

- **Category:** coupled, articulated. **Gated on Newton's upcoming generic inverse-dynamics function.**
- **Sketch:** `τ_g = -∂U_gravity/∂q`. Equivalent to inverse dynamics with `q̇ = q̈ = 0`.
- **Open:** port for the gravity vector (default `(0, 0, -9.81)`, but allow override). When the inverse-dynamics function lands, confirm its signature matches the per-robot batched layout the coupled-controller convention assumes.

### `ControlLawJointImpedance`

- **Category:** coupled, articulated. **Gated on the inverse-dynamics function** for any variant that includes `g(q)` or `h(q,q̇)`.
- **Sketch:** `τ = M(q)(q̈_d + K_d (q̇_d - q̇) + K_p (q_d - q)) + h(q, q̇)` (or simpler stiffness-only variants).
- **Open:** variants to ship — full impedance, PD + gravity, Cartesian impedance (later)? `M(q)` is public via `newton.eval_mass_matrix`; `h(q,q̇)` needs the inverse-dynamics function.

### `ControlLawOperationalSpace`

- **Category:** coupled, articulated, task-space. **Gated on the inverse-dynamics function** for the bias term μ and gravity p.
- **Sketch:** task-space inertia `Λ = (J M⁻¹ Jᵀ)⁻¹`; task force `F = Λ (ẍ_d + K_d (ẋ_d - ẋ) + K_p (x_d - x)) + μ + p`; torque `τ = Jᵀ F + (I - JᵀJ̄ᵀ) τ_null`, where `J̄ = M⁻¹ Jᵀ Λ`.
- `J` and `M` are public; `μ` and `p` need the inverse-dynamics function.
- **Open:** all-in-one OSC or split (TaskSpaceForce + JointMapping)? Inertia-weighted vs. plain damped pseudoinverse — both, or pick one? Nullspace handling — separate `ControlLawNullspaceProjection` block that accumulates, or built in?

### `ControlLawDifferentialDrive`

- **Category:** coupled, mobile.
- **Sketch:** input `(v, ω)` (linear m/s, angular rad/s) → wheel velocities `(ω_L, ω_R)` via `ω_L = (v - ω·L/2) / r`, `ω_R = (v + ω·L/2) / r` where `r` = wheel radius, `L` = axle width.
- **Construction convention:** takes raw scalars `wheel_radius: float`, `axle_width: float` directly. `wheels_per_robot = 2`. The user-supplied `indices` array orders `(left, right, left, right, …)` per-robot.
- **Open:** output port — wheel velocity (`joint_target_vel`) or wheel torque via an inner PI loop? Per-robot input `(v, ω)` as separate `target_v` / `target_omega` per-group ports, or combined `target_twist: wp.array[wp.vec2]`?

### `ControlLawHolonomic`

- **Category:** coupled, mobile.
- **Sketch:** input body twist `(v_x, v_y, ω)` → wheel velocities for an N-wheeled omni base (3-wheel Kiwi, 4-wheel mecanum, etc.). Mapping is `wheel_velocities = W(geometry) · [v_x, v_y, ω]ᵀ`.
- **Construction convention:** takes a `wheel_geometry: wp.array2d[float]` of shape `(wheels_per_robot, 3)` directly. Each row encodes one wheel's contribution to the velocity-mapping matrix. Helpers to build the matrix for known configurations (Kiwi, mecanum) live alongside the controller.
- **Open:** which geometry helpers to ship (Kiwi / mecanum / generic builder)? Output port — wheel velocity or wheel torque?

---

## Open questions

1. **Per-ControlLaw specs.** See *ControlLaws to design* above.
2. **In-kernel small-system solver helper.** `ControlLawDifferentialIK` ships an inline 6x6 Cholesky. If a second coupled ControlLaw needs the same SPD solve, extract it to `controllers/utils.py`.
