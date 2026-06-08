# Newton Controllers — Design Doc

## Background

There are many controllers which have been implemented across Isaac Lab/Sim. The goal of this module is to centralize all of these controllers to the Newton repository, re-implementing each controller so that they are completely CUDA-graphable and vectorized. A list of controllers, their original location, and their status in this module is given below:

| Controller | Current Repo | Completed?
| --- | --- | --- 
| Differential IK | Isaac Lab | :heavy_check_mark: |
| Operational Space | Isaac Lab | :x:  |
| Joint Impedance | Isaac Lab | :x: |
| Differential Drive | Isaac Sim | :x: |
| Holonomic Drive | Isaac Sim | :x: |
| Ackermann | Isaac Sim | :x: |

*NOTE* List above may not be complete.

Further, we may want to add some general purpose control algorithms which are very common in robotics, such as linear filtering (low-pass, band-pass, notch, etc).

---

## Architecture

This module is built from two base classes:

- **`ControlLaw`** — abstract base for a single law. Subclasses (`ControlLawPID`, `ControlLawDifferentialIK`, …) implement one algorithm. Each declares which named ports it reads and writes, owns any algorithm-specific buffers, and exposes a `compute(input, output, current_state, next_state, dt)` method.
- **`Controller`** — composer that wraps a list of `ControlLaw`s. Owns the per-step zero / compute sequence, the device, the gradient-tracking flag, and reset orchestration.

The split keeps `Controller` algorithm-agnostic: it routes data and orchestrates the step, the laws hold the actual math. Multiple laws bound to the same output slot accumulate via `+=` (see *Output accumulation*), which is useful for modular control, such as adding gravity compensation to an existing controller.

---

## Duck-typing input/output data structures

Every port on a `ControlLaw` is a **string** giving an attribute name. At step time, the Controller resolves the name via `getattr(input, name)` or `getattr(output, name)` against the duck-typed objects you pass to `step()`. Concretely:

```python
controller.step(input, output, current_state, next_state, dt)
```

- `input` — any object whose attributes hold the read ports (measurement, setpoint, gains, targets…).
- `output` — any object whose attributes hold the write ports (joint_f, joint_target_q, …).

Importantly, the `input` and `output` structures are not neccesarilly the newton `simData` and `controlData` objects. This is because controllers may require input information which is not tracked as part of the simulation state (for example, targets pose for a given site on the robot).

### Flexibility on input/output indices

Every per-DOF port accepts one of two specs at `__init__`:

| Spec | Meaning | Kernel access |
|---|---|---|
| `"attr_name"` | use the controller's own `indices` as port_indices | `getattr(source, name)[indices[i]]` |
| `("attr_name", port_indices)` | use a custom `port_indices` array | `getattr(source, name)[port_indices[i]]` |

The bare-string form handles the common case where every port reads from the same flat layout the controller writes to. The tuple form covers the layout-mismatch case (e.g. reading densely packed gains while writing into sparse global output slots).

**Per-robot ports** (e.g. controller targets which cannot be set per-dof, such as `target_pos` on an end-effector) accept just `"attr_name"`. At step the resolver checks `shape == (num_robots,)` and the documented dtype.

On all controllers, basic input validation is performed at the `__init__` and `step` functions:

| Stage | Check |
|---|---|
| `__init__` (per-DOF inputs) | `spec` is `str` or `(str, wp.array[uint32])`. For the tuple form, `port_indices.shape == indices.shape`. |
| `__init__` (per-robot inputs) | `spec` is `str`. |
| `step` (per-DOF inputs) | `getattr(source, name)` resolves to a `wp.array`. Shape/dtype mismatches surface at the kernel launch with Warp's diagnostic. |
| `step` (per-robot inputs) | `getattr(source, name)` resolves to a `wp.array` with shape `(num_robots,)` and the documented dtype. |

Shape and dtype of live arrays are checked at step time because the array identity is unknown until then. Init-time port_indices validation is enough to catch the structural mistakes; everything else fails loudly on the first launch.

### Example: shared backing array

The same backing array can show up under one attribute, with different per-port `port_indices` selecting different slots:

```python
pid = nc.ControlLawPID(
    indices=output_indices,
    measurement      = ("x", measurement_indices),
    measurement_rate = ("x", measurement_rate_indices),
    setpoint         = ("x", setpoint_indices),
    setpoint_rate    = ("x", setpoint_rate_indices),
    kp="kp", ki="ki", kd="kd",
    integral_max="integral_max",
    output=("x", output_indices),
)

input  = SimpleNamespace(x=state.x, kp=kp, ki=ki, kd=kd, integral_max=imax)
output = SimpleNamespace(x=state.x)

controller_state_0, controller_state_1 = controller.state(), controller.state()

controller.step(input, output, controller_state_0, controller_state_1, dt)
```

---

## Two flavors of ControlLaw

While there is no difference at the class-level, it is worth discussing the conventions for when a `ControlLaw` is either:

* independent on every DOF
* structurally coupled

### Independent per-DOF

Output `i` depends only on input `i` plus per-DOF parameters. Examples: `ControlLawPID`, low-pass filter, saturation, feedforward.

- Construction takes only the port specs plus an `indices` array of global DOF indices.
- Kernel launches 1D with `dim=len(indices)`; `i = wp.tid()`.
- Usually no `newton.Model` required at `__init__`.

### Coupled / structural

Each robot's outputs depend on the full state of that robot. Examples: `ControlLawDifferentialIK`, `ControlLawGravityComp`, `ControlLawDifferentialDrive`, `ControlLawHolonomic`, `ControlLawOperationalSpace`.

In this case, the `ControlLaw` must contain a model of the robots it controls. Below are two examples of how that model can be passed to a controller:

#### Common case: Articulation
In many cases, all information which is needed about the robot can be expressed by passing a single `model_builder: newton.ModelBuilder` argument containing `K = model_builder.articulation_count` topologically-identical articulations, i.e. the K articulations share:
    - DOF count
    - link/joint count
    - joint types

Controllers can then use functions such as `eval_fk`, `eval_jacobian` using robot models which are _independent_ of the simulated robot models, allowing the user to build modelling errors between the controller and the simulated robots.

#### Common case: Mobile robots
For most mobile robot controllers, they are parameterized by data which is not obvious to extract from a `newton.Model`, such as wheel radius, axle width, wheel-mapping matrix. In this case, model data for `K` topilogically-identical mobile robots can be passed as `wp.array` inputs for each required parameter.

In either case (`model_builder` or directly passing parameters) the conventional global output index of robot `r`'s local slot `j` is `output_indices[r * outputs_per_robot + j]`.

---

## Output accumulation

At the start of each `step()` the `Controller` resolves every output binding's `attr_name` against the passed-in `output`, then zeros the slots indicated by each binding's `port_indices`. The ControlLaws then run serially, in registration order; each writes via `+=` into its output arrays:

```python
# Inside a per-DOF kernel:
i = wp.tid()
out_idx = output_indices[i]
output_array[out_idx] += contribution
```

Composition is sum-of-contributions: a PD term + a gravity-compensation term + a feedforward term all writing to the same `joint_f` produce their pointwise sum. There are no overlap checks — laws compose at the user's discretion. Two laws binding overlapping slots have those slots zeroed twice in the upfront pass; idempotent and avoids any wp.array identity comparison.

**Multi-output laws** (e.g. `ControlLawDifferentialIK` writes both `output_qd` and `output_q`) declare multiple bindings via `outputs()`. Each is treated identically for zero / accumulate purposes. All of a law's outputs must share the same outer length (`num_outputs`).

---

## State and double-buffering

Each `ControlLaw` defines a nested `State` dataclass holding whatever per-step internal buffers it needs (PID integrals, filter history, …). `ControlLaw.is_stateful()` reports whether it carries any; `ControlLaw.state(num_outputs, device, requires_grad)` allocates a fresh one.

`Controller.State` composes per-law states as a flat list. `controller.state()` returns a composed state with every stateful law's state already allocated (and `None` entries for stateless laws).

The step protocol is double-buffered: the Controller reads from `current_state`, writes to `next_state`, and the caller swaps:

```python
state_0 = controller.state()
state_1 = controller.state()
for _ in range(steps):
    controller.step(input, output, state_0, state_1, dt=dt)
    state_0, state_1 = state_1, state_0
```

This mirrors the in/out State pattern used by `Solver.step(state_in, state_out, ...)` and avoids tape entanglement when running under `wp.Tape`.

---

## Differentiability

`Controller.__init__(control_laws, requires_grad=False)` is the single source of truth for gradient support. The flag propagates into every `ControlLaw.finalize(device, num_outputs, requires_grad)` and `ControlLaw.state(num_outputs, device, requires_grad)` call, and from there into every internally allocated buffer (PID's `integral` and `reset_state`; DiffIK's replicated `Model`, `_jacobian`, `_qd_target_local`, …).

User-supplied input arrays (`measurement`, `target_pos`, `kp`, …) carry their own `requires_grad` — the laws don't own those allocations.

Kernels use the default `@wp.kernel` decorator, which records adjoints onto an active `wp.Tape`. The module is tape-agnostic: the caller (Isaac Lab, a custom training loop) wraps the relevant block in `wp.Tape()` and runs `tape.backward(loss=…)` on its own. This matches `newton.actuators.Actuator.step`.

### Per-law status

- **`ControlLawPID`** — fully differentiable end-to-end. Gradients flow from `output` back through `measurement`, `measurement_rate`, `setpoint`, `setpoint_rate`, `kp`, `ki`, `kd`, `integral_max`.
- **`ControlLawDifferentialIK`** — tape-safe; forward-only through the damped least squares (DLS) solve. The compute chain is split into per-element kernels (gather --> build site Jacobian --> build the 6×6 DLS matrix --> q_dot back-projection --> accumulate) plus one tile-cooperative Cholesky-solve kernel. Every kernel except the solve is autograd-able by default. The solve uses `wp.tile_cholesky` + `wp.tile_cholesky_solve`; their adjoints are advertised but return zero gradients in Warp 1.14.0 (verified directly: forward correct, backward gives zero gradients on both A and the rhs). The solve kernel is therefore marked `enable_backward=False` to make the zero-gradient behaviour explicit. Gradients propagate from the loss back to `output_qd` and stop at the solve. Useful for RL pipelines that wrap a whole sim in `wp.Tape` without needing IK gradients; not yet usable for end-to-end diff-physics training through the IK. Revisit if upstream `wp.tile_cholesky` backward lands.

Mixed grad-tracking needs (some laws gradient-tracked, others not) split into multiple `Controller`s; there is no per-law override.

---

## How controllers compose with actuators and solvers

The standard ordering inside one simulation step:

```python
controller = nc.Controller([pid, diff_ik])
actuator   = newton.actuators.Actuator(controller=..., ...)

ctrl_state_0 = controller.state(); ctrl_state_1 = controller.state()
act_state_0  = actuator.state();  act_state_1  = actuator.state()

for _ in range(steps):
    # 1. Control laws compute targets (joint position, velocity, or feedforward effort).
    controller.step(ctrl_in, ctrl_out, ctrl_state_0, ctrl_state_1, dt=dt)

    # 2. Actuators translate target -> joint effort, applying delay / clamping / etc.
    actuator.step(sim_state, sim_control, act_state_0, act_state_1, dt=dt)

    # 3. Physics solver advances the state.
    solver.step(sim_state, sim_state_next, sim_control, contacts, dt)

    ctrl_state_0, ctrl_state_1 = ctrl_state_1, ctrl_state_0
    act_state_0,  act_state_1  = act_state_1,  act_state_0
    sim_state, sim_state_next = sim_state_next, sim_state
```

The bridge is `ctrl_out` <--> `sim_control`: the user typically points `ctrl_out`'s attributes at the same `wp.array`s exposed on `sim_control` (e.g. `joint_target_q`, `joint_f`). The controllers module never introspects Newton sim objects — that wiring is the user's responsibility, and it stays explicit at the call site.

For controllers which only result in joint targets, they can write directly to the `sim_control` object rather than constructing a separate `ctrl_out` object.

---

## Subclassing a ControlLaw

The base class is a minimal contract. Subclasses are free to choose their kernel shapes, decide whether to take a `model_builder` argument, etc.

```python
class ControlLaw:
    @dataclass
    class State:
        """Pure data container. Subclasses declare their fields (integral
        arrays, history buffers). No methods."""

    indices: wp.array[wp.uint32]
    """Global output indices this law writes to. Set in __init__."""

    def __init__(self, **ports):
        """Validate port specs and stash them as (attr_name, port_indices)
        pairs (per-DOF) or attr_name strings (per-robot). Cheap CPU-side
        work only: device buffers are allocated in finalize().

        Subclasses declare which kwargs they accept; missing required ports
        raise here, unknown ports raise here, and port_indices shape
        mismatches raise here."""

    def finalize(self, device: wp.Device, num_outputs: int,
                 requires_grad: bool = False) -> None:
        """Allocate device-side private buffers (e.g. internal Model +
        Jacobian buffers for articulated laws). Called by Controller after
        construction. num_outputs == len(self.indices)."""

    def state(self, num_outputs: int, device: wp.Device,
              requires_grad: bool = False) -> ControlLaw.State | None:
        """Allocate a fresh State, or None for stateless laws."""

    def is_stateful(self) -> bool: ...
    def is_graphable(self) -> bool: ...

    def outputs(self) -> list[tuple[str, wp.array[wp.uint32]]]:
        """Return the law's (output_attr_name, output_port_indices)
        bindings. The Controller resolves each attr_name against the
        `output` arg of step() once per step and zeros the listed slots
        before any compute() runs."""

    def compute(self,
                input, 
                output,
                state: ControlLaw.State | None,
                next_state: ControlLaw.State | None,
                dt: float) -> None:
        """Resolve port arrays via _resolve_input_array / _resolve_per_robot_array,
        launch kernels: read live data, write += into outputs, populate
        next_state. Device is fixed at finalize() time."""
```

The `Controller` itself is small:

```python
class Controller:
    def __init__(self,
                 control_laws: list[ControlLaw],
                 requires_grad: bool = False,
                 device: wp.Device | None = None):
        """Validate that every law agrees on num_outputs, finalize() each
        (allocating per-law buffers + reset_state), collect every law's
        outputs() into a flat list to be resolved + zeroed at step time."""

    def is_stateful(self) -> bool: ...
    def is_graphable(self) -> bool: ...

    def state(self) -> Controller.State:
        """Allocate composed state — one entry per ControlLaw (None for
        stateless laws)."""

    def step(self, 
             input, 
             output,
             current_state: Controller.State,
             next_state: Controller.State,
             dt: float) -> None:
        """1. Resolve each output binding's attr_name against `output`,
              and zero the declared slots. Two laws binding overlapping
              slots get zeroed twice — idempotent.
           2. For each law in registration order, call compute(input,
              output, cur, nxt, dt), which += writes into outputs."""
```

```python
@dataclass
class Controller.State:
    control_law_states: list[ControlLaw.State | None]
```

### Internal helpers (`newton/_src/controllers/utils.py`)

- `_normalize_port(spec, control_law_indices, name)` — validate `str` or `(str, port_indices)` at `__init__`, return `(attr_name, port_indices)`.
- `_normalize_per_robot_port(spec, name)` — validate the per-robot `str` spec at `__init__`.
- `_resolve_input_array(source, attr_name, name)` — step-time `getattr` plus `wp.array` type check.
- `_resolve_per_robot_array(source, attr_name, num_robots, dtype, name)` — step-time `getattr` plus per-robot shape + dtype check.

### Choosing between independent and coupled

If output `i` depends only on input `i` and per-DOF parameters, write it independent: 1D launch, no `Model`, `len(indices) = N`. Pure scalar math.

If each robot's outputs depend on its full configuration (Jacobians, mass matrix, kinematic chain, base velocity), write it coupled: take a `model_builder` or raw geometry, replicate at finalize, 2D launch over `(num_robots, outputs_per_robot)`.

The reference implementations are `controller_pid.py` (independent) and `controller_diff_ik.py` (coupled, articulated).

---

## Roadmap

ControlLaws currently implemented:

1. **`ControlLawPID`** — independent per-DOF, fully differentiable.
2. **`ControlLawDifferentialIK`** — coupled articulated, tape-safe (forward-only through the solve).

Planned, but blocked by lack of inverse-dynamics function:

- `ControlLawGravityComp`
- `ControlLawJointImpedance`
- `ControlLawOperationalSpace`