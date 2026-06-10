# Newton Controllers — Design Doc

## Background

Many control laws are implemented across Isaac Lab and Isaac Sim. The goal of this module is to centralise them in Newton, re-implemented to be CUDA-graphable and vectorized. Status of the controllers being ported:

| Controller | Original repo | In Newton? |
| --- | --- | --- |
| Differential IK    | Isaac Lab | :heavy_check_mark: |
| Operational Space  | Isaac Lab | :x: |
| Joint Impedance    | Isaac Lab | :x: |
| Differential Drive | Isaac Sim | :x: |
| Holonomic Drive    | Isaac Sim | :x: |
| Ackermann          | Isaac Sim | :x: |

*NOTE:* the list may not be complete.

General-purpose blocks (linear filters, saturation, feedforward, …) will land alongside the ported controllers as they become useful.

---

## Architecture

The module is built from two base classes plus two data types:

- **`ControlSignal`** — a *slot type*. Carries `(dtype, ndim, description)`; no attribute name. Module-level constants are canonical (identity-equal). Newton ships a small joint-level vocabulary (`JOINT_Q`, `JOINT_QD`, `JOINT_TARGET_Q`, `JOINT_TARGET_QD`, `JOINT_F`); user-defined signals (setpoint, gains, IK targets, …) are demonstrated in tests and examples.
- **`HardwareInterface`** — per-deployment wiring: two `dict[ControlSignal, str]`s, `inputs` and `outputs`, mapping each signal to the attribute name on the runtime `input` / `output` object. The interface is the *only* place where signal-to-name resolution lives.
- **`ControlLaw`** — abstract base for a single law. Subclasses (`ControlLawPID`, `ControlLawDifferentialIK`, …) declare `INPUT_PORTS` and `OUTPUT_PORTS` (frozen sets of port-name strings) and accept `(ControlSignal, port_indices)` tuples as their constructor port kwargs. Each law records the set of signals it consumes / produces.
- **`Controller`** — composer that takes a `HardwareInterface` and a list of `ControlLaw`s. Validates that every law's used signals are covered by the interface in the right direction, asks each law to resolve its signal bindings against the interface (so step-time is plain `getattr`), and owns the per-step "zero outputs, then accumulate via `+=`" pattern.

The split keeps `Controller` algorithm-agnostic: it routes data and orchestrates the step, the laws hold the actual math, and the interface holds the deployment-specific naming. Multiple laws bound to the same output signal accumulate via `+=` (see *Output accumulation*) — useful for modular control like summing a gravity-compensation term with a PID feedback term.

---

## Signals, interfaces, and the input/output structs at step time

A user wiring up a controller takes three steps:

1. **Pick or define signals.** Newton ships canonical joint-level signals; if a law needs something Newton doesn't ship (setpoint, gains, target poses), the user defines it as a module-level `ControlSignal`.
2. **Build a `HardwareInterface`** mapping each signal to the attribute name on whatever object will be passed as `input` / `output` at step time. The interface is reusable across configurations and shared by every law in the same `Controller`.
3. **Construct each `ControlLaw`** by binding each port kwarg to a `(signal, port_indices)` tuple, then compose them under a `Controller(hw, [law_a, law_b, ...])`.

```python
# Per-application signals (Newton doesn't ship these — they belong to the
# user's controller configuration):
SETPOINT = ControlSignal(dtype=wp.float32, ndim=1, description="PID setpoint")
KP       = ControlSignal(dtype=wp.float32, ndim=1, description="proportional gain")
# ... KI, KD, INTEGRAL_MAX, ...

# Wiring for this application: which attribute on input/output holds each signal.
hw = HardwareInterface(
    inputs={
        JOINT_Q:       "joint_q",
        JOINT_QD:      "joint_qd",
        SETPOINT:      "setpoint",
        SETPOINT_RATE: "setpoint_rate",
        KP: "kp", KI: "ki", KD: "kd",
        INTEGRAL_MAX:  "integral_max",
    },
    outputs={JOINT_F: "joint_f"},
)

# Law construction takes only signals + port_indices — no attribute names.
pid = ControlLawPID(
    measurement      = (JOINT_Q,       indices),
    measurement_rate = (JOINT_QD,      indices),
    setpoint         = (SETPOINT,      indices),
    setpoint_rate    = (SETPOINT_RATE, indices),
    kp               = (KP,            indices),
    ki               = (KI,            indices),
    kd               = (KD,            indices),
    integral_max     = (INTEGRAL_MAX,  indices),
    output           = (JOINT_F,       indices),
)

controller = Controller(hw, [pid])

# At step time, the user supplies whatever input/output object they like;
# the attributes on it must match the interface's names.
controller.step(input, output, current_state, next_state, dt)
```

The `input` and `output` objects are duck-typed — `SimpleNamespace`, an `@dataclass`, or anything that exposes the right attributes. They don't have to be `newton.State` / `newton.Control`; the interface mediates between the canonical signal vocabulary and whatever object the user passes.

### Port form

Every port spec is a 2-tuple `(ControlSignal, wp.array[wp.uint32])`:

| Element | Type | Meaning |
|---|---|---|
| `signal` | `ControlSignal` | The slot type. Identifies which `HardwareInterface` entry this port reads from / writes to. |
| `port_indices` | `wp.array[wp.uint32]` | Per-element kernel lookup. `arr[port_indices[i]]` for per-DOF ports, `arr[port_indices[r]]` for per-robot ports. |

Per-DOF and per-robot ports use the same shape of spec — the only difference is the length of `port_indices`:

- **Per-DOF**: `port_indices.shape == (num_outputs,)` where `num_outputs` is derived from the output port's `port_indices` length and cross-checked against every other per-DOF port at `__init__`.
- **Per-robot**: `port_indices.shape == (num_robots,)` where `num_robots = model_builder.articulation_count` for articulated coupled laws.

### Validation

| Stage | Check |
|---|---|
| `__init__` | Each port spec is a 2-tuple `(ControlSignal, wp.array[wp.uint32])`. Per-DOF / per-robot `port_indices` lengths cross-check against the law's structural sizes. |
| `Controller.__init__` | Every signal the law's `_used_inputs` contains is a key in `hw.inputs`; same for `_used_outputs` / `hw.outputs`. |
| `step` | `getattr(input, attr_name)` resolves to a `wp.array`. Dtype/shape mismatches surface at the kernel launch with Warp's diagnostic. No construction-time dtype check yet — relying on kernel launches keeps subclass authoring minimal. |

### Shared backing array

A single backing `wp.array` can be referenced under one attribute on both `input` and `output`. Distinct signals with different `port_indices` then read/write different slots of that backing array — useful for "controller writes joint_target_q; downstream PID reads joint_target_q" wiring.

---

## Two flavors of ControlLaw

The class hierarchy is flat — both flavors derive from `ControlLaw`. The distinction shows up in kernel-launch shape and structural arguments.

### Independent per-DOF

Output `i` depends only on input `i` plus per-DOF parameters. Examples: `ControlLawPID`, low-pass filter, saturation, feedforward.

- Construction takes only port specs.
- Kernel launches 1D with `dim=num_outputs`; `i = wp.tid()`.
- No `newton.Model` required.

### Coupled / structural

Each robot's outputs depend on the full state of that robot. Examples: `ControlLawDifferentialIK`, `ControlLawGravityComp`, `ControlLawDifferentialDrive`, `ControlLawHolonomic`, `ControlLawOperationalSpace`.

The law must carry a model of the robots it controls. Two patterns, depending on whether the data lives in a `newton.ModelBuilder` or in raw parameter arrays:

#### Articulated case

The caller passes a `model_builder: newton.ModelBuilder` containing exactly `N` topologically-identical articulations — `N = model_builder.articulation_count = num_robots`. The N articulations share DOF count, link/joint count, and joint types. They may differ in physical parameters (mass, inertia, friction, joint limits) and per-articulation site placement.

The controller does **no replication** — if the user wants `N` copies of a single-robot template, they call `newton.ModelBuilder.replicate(template, world_count=N)` themselves before construction.

At `finalize()` the law calls `model_builder.finalize(device, requires_grad)` to get its internal `Model`. Per-step compute uses `eval_fk`, `eval_jacobian`, etc. on this internal model — independent of the simulated scene's model, so the user can deliberately introduce modelling errors between controller and sim.

#### Mobile case

Mobile-robot controllers are parameterized by data that doesn't live in `newton.Model` (wheel radius, axle width, wheel-mapping matrix). The user passes those parameters as plain `wp.array`s of length `N = num_robots`. No replication.

#### Indexing

For coupled laws, the per-DOF output port's `port_indices` is required to be **robot-contiguous**:

```
port_indices = [
    # robot 0's DOFs:
    robot_0_dof_0, robot_0_dof_1, ..., robot_0_dof_{D-1},
    # robot 1's DOFs:
    robot_1_dof_0, robot_1_dof_1, ..., robot_1_dof_{D-1},
    ...
]
```

Each entry is the global output index (slot in the user's `output` array) for that robot's local DOF. Kernels launch 2D with `dim=(num_robots, dofs_per_robot)`; the flat index into `port_indices` is `robot * dofs_per_robot + local_dof`.

---

## Output accumulation

At the start of each `step()` the `Controller` resolves every output binding's `attr_name` against the passed-in `output`, then zeros the slots indicated by each binding's `port_indices`. The ControlLaws then run serially, in registration order; each writes via `+=` into its output arrays:

```python
# Inside a per-DOF kernel:
i = wp.tid()
out_idx = output_indices[i]
output_array[out_idx] += contribution
```

Composition is sum-of-contributions: a PD term + a gravity-compensation term + a feedforward term all writing to the same `joint_f` produce their pointwise sum. There are no overlap checks — laws compose at the user's discretion. Two laws binding overlapping slots have those slots zeroed twice in the upfront pass; idempotent.

**Multi-output laws** (e.g. `ControlLawDifferentialIK` writes both `output_qd` and `output_q`) declare multiple bindings via `outputs()`. Each is treated identically for zero / accumulate purposes. All of a law's outputs must share the same outer length (`num_outputs`).

---

## State and double-buffering

Each `ControlLaw` defines a nested `State` dataclass holding whatever per-step internal buffers it needs (PID integrals, filter history, …). `ControlLaw.is_stateful()` reports whether it carries any; `ControlLaw.state(device, requires_grad)` allocates a fresh one.

`Controller.State` composes per-law states as a flat list keyed by registration order. `controller.state()` returns a composed state with every stateful law's state already allocated (and `None` entries for stateless laws).

The step protocol is double-buffered: the Controller reads from `current_state`, writes to `next_state`, and the caller swaps:

```python
state_0 = controller.state()
state_1 = controller.state()
for _ in range(steps):
    controller.step(input, output, state_0, state_1, dt=dt)
    state_0, state_1 = state_1, state_0
```

This mirrors the in/out State pattern used by `Solver.step(state_in, state_out, ...)` and keeps autograd-safe kernels free of aliased read/write arrays.

---

## Differentiability

`Controller.__init__(..., requires_grad=False)` is the single source of truth for gradient support. The flag propagates into every `ControlLaw.finalize(device, requires_grad)` and `ControlLaw.state(device, requires_grad)` call, and from there into every internally allocated buffer.

User-supplied arrays bound to ports (`measurement`, `target_pos`, `kp`, …) carry their own `requires_grad` — the laws don't own those allocations.

Kernels use the default `@wp.kernel` decorator, which records adjoints onto an active `wp.Tape`. The module is tape-agnostic: the caller (Isaac Lab, a custom training loop) wraps the relevant block in `wp.Tape()` and runs `tape.backward(loss=…)` on its own.

### Per-law status

- **`ControlLawPID`** — fully differentiable end-to-end. Gradients flow from `output` back through every read port.
- **`ControlLawDifferentialIK`** — tape-safe; forward-only through the damped least squares (DLS) solve. Every kernel except the solve is autograd-able by default. The solve uses `wp.tile_cholesky` + `wp.tile_cholesky_solve`; their adjoints are advertised but return zero gradients in Warp 1.14.0 (verified directly). The solve kernel is therefore marked `enable_backward=False`. Useful for RL pipelines that wrap a whole sim in `wp.Tape` without needing IK gradients; not yet usable for end-to-end diff-physics training through the IK. Revisit if upstream `wp.tile_cholesky` backward lands.

Mixed grad-tracking needs (some laws gradient-tracked, others not) split into multiple `Controller`s.

---

## How controllers compose with actuators and solvers

```python
controller = Controller(hw, [pid, diff_ik])
actuator   = newton.actuators.Actuator(controller=..., ...)

ctrl_state_0 = controller.state(); ctrl_state_1 = controller.state()
act_state_0  = actuator.state();   act_state_1  = actuator.state()

for _ in range(steps):
    # 1. Control laws compute targets (joint position, velocity, or feedforward effort).
    controller.step(ctrl_in, ctrl_out, ctrl_state_0, ctrl_state_1, dt=dt)

    # 2. Actuators translate target -> joint effort.
    actuator.step(sim_state, sim_control, act_state_0, act_state_1, dt=dt)

    # 3. Physics solver advances the state.
    solver.step(sim_state, sim_state_next, sim_control, contacts, dt)

    ctrl_state_0, ctrl_state_1 = ctrl_state_1, ctrl_state_0
    act_state_0,  act_state_1  = act_state_1,  act_state_0
    sim_state, sim_state_next  = sim_state_next, sim_state
```

The bridge from controllers to actuators is the runtime `wp.array` reference: the user typically points an attribute of `ctrl_out` (named per the `HardwareInterface`) at the same `wp.array` exposed on `sim_control` (e.g. `joint_target_q`, `joint_f`). The controllers module never introspects Newton sim objects — that wiring is the user's responsibility.

For controllers whose only output is a joint target, the user can point `ctrl_out` directly at the corresponding attribute on `sim_control` (skipping a separate `Output` namespace).

---

## Subclassing a ControlLaw

The base class is a minimal contract. Subclasses choose their kernel shapes, declare their port roles, and (for coupled laws) declare what structural arguments their constructor takes beyond ports.

```python
class ControlLaw:
    INPUT_PORTS:  ClassVar[frozenset[str]] = frozenset()
    OUTPUT_PORTS: ClassVar[frozenset[str]] = frozenset()

    @dataclass
    class State:
        """Pure data container. Subclasses declare fields (integral
        arrays, filter history, …). No methods."""

    _used_inputs:  frozenset[ControlSignal]
    _used_outputs: frozenset[ControlSignal]

    def __init__(self, **port_bindings):
        """Validate every port spec (a 2-tuple of (ControlSignal,
        wp.array[uint32])), cross-check per-DOF / per-robot lengths,
        record _used_inputs / _used_outputs, stash (signal, port_indices)
        keyed by port name. Cheap CPU-side work; device-side buffers are
        allocated in finalize()."""

    def _resolve(self, hw: HardwareInterface) -> None:
        """Convert each stashed (signal, port_indices) into
        (attr_name, port_indices) using hw. Called by the composing
        Controller once at construction. After this call, compute() can
        do plain getattr(input, self._<port>_attr)."""

    def finalize(self, device: wp.Device, requires_grad: bool = False) -> None:
        """Allocate device-side private buffers (e.g. internal Model +
        Jacobian buffers for articulated laws). Called by the Controller
        after _resolve."""

    def state(self, device: wp.Device,
              requires_grad: bool = False) -> ControlLaw.State | None:
        """Allocate a fresh State, or None for stateless laws."""

    def is_stateful(self) -> bool: ...
    def is_graphable(self) -> bool: ...

    def inputs(self) -> list[tuple[str, wp.array[wp.uint32]]]:
        """Resolved (attr_name, port_indices) pairs for read ports. Used
        by Controller.input() to size the auto-allocated namespace."""

    def outputs(self) -> list[tuple[str, wp.array[wp.uint32]]]:
        """Resolved (attr_name, port_indices) pairs for write ports.
        Used by Controller.step()'s upfront zero pass and by
        Controller.output() for auto-allocation."""

    def compute(self,
                input, output,
                state: ControlLaw.State | None,
                next_state: ControlLaw.State | None,
                dt: float) -> None:
        """Read live arrays via getattr(input, self._<port>_attr) (the
        resolved name was stashed by _resolve), launch kernels, write
        via += into outputs, populate next_state."""
```

The `Controller` itself is small:

```python
class Controller:
    def __init__(self,
                 hw: HardwareInterface,
                 control_laws: list[ControlLaw],
                 requires_grad: bool = False,
                 device: wp.Device | None = None):
        """For each law: verify _used_inputs ⊆ hw.inputs.keys() and
        _used_outputs ⊆ hw.outputs.keys(); call law._resolve(hw) to stash
        attr_name per port; call law.finalize(device, requires_grad).
        Collect a flat list of (attr_name, port_indices) output bindings
        for the upfront zero pass at step time."""

    def state(self)  -> Controller.State: ...
    def input(self)  -> SimpleNamespace: ...    # auto-allocated; mutate fields as needed
    def output(self) -> SimpleNamespace: ...

    def step(self, input, output,
             current_state: Controller.State,
             next_state: Controller.State,
             dt: float) -> None:
        """1. For each (attr_name, port_indices) output binding, zero
              output[attr_name][port_indices].
           2. For each law in registration order, call compute(input,
              output, cur, nxt, dt), which += writes into outputs."""
```

```python
@dataclass
class Controller.State:
    control_law_states: list[ControlLaw.State | None]
```

### Internal helpers (`newton/_src/controllers/utils.py`)

- `_normalize_port(spec, name)` — validate the 2-tuple `(ControlSignal, wp.array[uint32])` spec at `__init__`, return `(signal, port_indices)`.
- `_resolve_input_array(source, attr_name, name)` — step-time `getattr` plus `wp.array` type check.

### Choosing between independent and coupled

If output `i` depends only on input `i` and per-DOF parameters, write it independent: 1D launch, no `Model`. Pure scalar math.

If each robot's outputs depend on its full configuration (Jacobians, mass matrix, kinematic chain, base velocity), write it coupled: take a `model_builder` (articulated case) or raw per-robot parameter vectors (mobile case); finalize the builder at `finalize()`; 2D launch over `(num_robots, outputs_per_robot)`.

Reference implementations: `controller_pid.py` (independent) and `controller_diff_ik.py` (coupled, articulated).

---

## Roadmap

Implemented:

1. **`ControlLawPID`** — independent per-DOF, fully differentiable.
2. **`ControlLawDifferentialIK`** — coupled articulated, tape-safe (forward-only through the solve).

Planned, blocked by lack of a public inverse-dynamics function in Newton:

- `ControlLawGravityComp`
- `ControlLawJointImpedance`
- `ControlLawOperationalSpace`

Planned, unblocked:

- `ControlLawDifferentialDrive`
- `ControlLawHolonomic`
- `ControlLawAckermann`
- Linear filters (low-pass, band-pass, notch).
