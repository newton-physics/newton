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

- **`ControlLaw`** — abstract base for a single law. Subclasses (`ControlLawPID`, `ControlLawDifferentialIK`, …) implement one algorithm. Each carries a unique-within-`Controller` `label`, declares which named ports it reads and writes, owns any algorithm-specific buffers, and exposes a `compute(input, output, current_state, next_state, dt)` method.
- **`Controller`** — composer that wraps a list of `ControlLaw`s. Owns the per-step zero / compute sequence, the device, the gradient-tracking flag, and the composed state.

The split keeps `Controller` algorithm-agnostic: it routes data and orchestrates the step, the laws hold the actual math. Multiple laws bound to the same output slot accumulate via `+=` (see *Output accumulation*), which is useful for modular control, such as adding gravity compensation to an existing controller.

---

## Duck-typing input/output data structures

Every port on a `ControlLaw` is a **string** giving an attribute name. At step time, the Controller resolves the name via `getattr(input, name)` or `getattr(output, name)` against the duck-typed `input`/`output` objects you pass to `step()`. Concretely:

```python
controller.step(input, output, current_state, next_state, dt)
```

- `input` — any object whose attributes hold the read ports (measurement, setpoint, gains, targets…).
- `output` — any object whose attributes hold the write ports (joint_f, joint_target_q, …).

Importantly, the `input` and `output` structures are not neccesarilly the newton `simData` and `controlData` objects. This is because controllers may require input information which is not tracked as part of the simulation state (for example, a target pose for a given site on the robot, or a live update to a controller gain if it has variable impedance).

### Port form

Every port spec is a 2-tuple ``("attr_name", port_indices)``:

| Element | Type | Meaning |
|---|---|---|
| `attr_name` | `str` | The attribute name on the `input` / `output` object. Resolved at step time via `getattr(source, attr_name)`. |
| `port_indices` | `wp.array[wp.uint32]` | The inner lookup. Kernel reads `arr[port_indices[i]]` (per-DOF) or `arr[port_indices[r]]` (per-robot). |

Per-DOF and per-robot ports use the same shape of spec — the only difference is the length of `port_indices`:

- **Per-DOF**: `port_indices.shape == (num_outputs,)` where `num_outputs` is the controller's "size" (derived from the output port's indices length and cross-checked against every other per-DOF port at `__init__`).
- **Per-robot**: `port_indices.shape == (num_robots,)` where `num_robots = model_builder.articulation_count` for articulated coupled laws.

### Validation

| Stage | Check |
|---|---|
| `__init__` | `spec` is `(str, wp.array[wp.uint32])`. |
| `__init__` (per-DOF) | `port_indices.shape == (num_outputs,)`. |
| `__init__` (per-robot) | `port_indices.shape == (num_robots,)`. |
| `step` | `getattr(source, attr_name)` resolves to a `wp.array`. Per-robot ports additionally check the dtype matches the documented contract (`wp.vec3` / `wp.quat` / `wp.float32`). Out-of-bounds reads via wrongly-sized source arrays surface at the kernel launch with Warp's diagnostic. |

### Example: shared backing array
In particular example below, the same `wp.array` (`state.x`) appears under one attribute on both `input` and `output`; different `port_indices` arrays disambiguate which slots each port reads or writes.

```python
pid = nc.ControlLawPID(
    label            = "arm_pid",
    measurement      = ("x", measurement_indices),
    measurement_rate = ("x", measurement_rate_indices),
    setpoint         = ("x", setpoint_indices),
    setpoint_rate    = ("x", setpoint_rate_indices),
    kp               = ("kp", gain_indices),
    ki               = ("ki", gain_indices),
    kd               = ("kd", gain_indices),
    integral_max     = ("integral_max", gain_indices),
    output           = ("x", output_indices),
)

input  = SimpleNamespace(x=state.x, kp=kp, ki=ki, kd=kd, integral_max=imax)
output = SimpleNamespace(x=state.x)

cs0, cs1 = controller.state(), controller.state()
controller.step(input, output, cs0, cs1, dt)
```

---

## Two flavors of ControlLaw

There is no difference at the class level, but the conventions diverge between independent-per-DOF laws and structurally coupled laws.

### Independent per-DOF

Output `i` depends only on input `i` plus per-DOF parameters. Examples: `ControlLawPID`, low-pass filter, saturation, feedforward.

- Construction takes only the port specs.
- Kernel launches 1D with `dim=num_outputs`; `i = wp.tid()`.
- No `newton.Model` required.

### Coupled / structural

Each robot's outputs depend on the full state of that robot. Examples: `ControlLawDifferentialIK`, `ControlLawGravityComp`, `ControlLawDifferentialDrive`, `ControlLawHolonomic`, `ControlLawOperationalSpace`.

The law must carry a model of the robots it controls. Two patterns, depending on whether the data/functionality we need is found in a `newton.ModelBuilder` or must be passed as independent parameters.

#### Newton model case

The caller passes a `model_builder: newton.ModelBuilder` containing exactly `N` topologically-identical articulations — `N = model_builder.articulation_count = num_robots` is the number of robots this controller manages. The N articulations share:

- DOF count
- link/joint count
- joint types

They may differ in physical parameters (mass, inertia, friction, joint limits) and in per-articulation site placement (so e.g. a fleet with mixed gripper lengths can sit under one `ControlLawDifferentialIK`).

The controller does **no replication** — if the user wants `N` copies of a single-robot template, they call `newton.ModelBuilder.replicate(template, world_count=N)` themselves before construction. Keeping the responsibility on the caller means a single, obvious mental model: the builder you hand in is exactly the set of robots the controller will manage.

At `finalize()` the controller calls `model_builder.finalize(device, requires_grad)` to get an internal `Model`. Per-step compute uses `eval_fk`, `eval_jacobian`, etc. on this internal model — which is independent of the simulated scene's model, so the user can deliberately introduce modelling errors between controller and sim if they want to.

#### Raw parameters case

Mobile-robot controllers are parameterized by data that doesn't live in `newton.Model` (wheel radius, axle width, wheel-mapping matrix). The user passes those parameters as plain `wp.array` arguments of length `N = num_robots`. No replication; the controller just consumes the length-N parameter vectors directly.

#### Indexing

For both kinds of coupled law, the per-DOF output port's `port_indices` is required to be **robot-contiguous**: all of robot 0's DOFs come first, then all of robot 1's DOFs, and so on. Concretely, with `N = num_robots` and `D = dofs_per_robot`, the array looks like:

```
port_indices = [
    # robot 0's DOFs, in local order:
    robot_0_dof_0, robot_0_dof_1, ..., robot_0_dof_{D-1},

    # robot 1's DOFs, in local order:
    robot_1_dof_0, robot_1_dof_1, ..., robot_1_dof_{D-1},

    ...

    # robot N-1's DOFs, in local order:
    robot_{N-1}_dof_0, robot_{N-1}_dof_1, ..., robot_{N-1}_dof_{D-1},
]
```

Each entry is the global output index (the slot in the user's `output` array) for that robot's local DOF. So `port_indices[robot * dofs_per_robot + local_dof]` gives the global slot for robot `robot`'s `local_dof`-th DOF.

Kernels launch 2D with `dim=(num_robots, dofs_per_robot)`; inside, `robot, local_dof = wp.tid()` and the flat index into `port_indices` is `robot * dofs_per_robot + local_dof`.

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

Each `ControlLaw` defines a nested `State` dataclass holding whatever per-step internal buffers it needs (PID integrals, filter history, …). `ControlLaw.is_stateful()` reports whether it carries any; `ControlLaw.state(num_outputs, device, requires_grad)` allocates a fresh state or returns `None` if the control law is not stateful.

`Controller.State` composes per-law states as a `dict[str, ControlLaw.State | None]` keyed by each law's `label` (with `None` entries for stateless laws). Dict insertion order matches the order the laws were passed to the `Controller`, so step iteration is deterministic. Keying by label makes external introspection (`state.control_law_states["arm_pid"].integral.numpy()`) stable across reordering and easier to read than positional indexing.

The step protocol is double-buffered: the Controller reads from `current_state`, writes to `next_state`, and the caller swaps:

```python
controller_state_0 = controller.state()
controller_state_1 = controller.state()
for _ in range(steps):
    controller.step(input, output, controller_state_0, controller_state_1, dt=dt)
    controller_state_0, controller_state_1 = controller_state_1, controller_state_0
```

This mirrors the in/out State pattern used by `Solver.step(state_in, state_out, ...)` and avoids tape entanglement when running under `wp.Tape`.

---

## Differentiability

`Controller.__init__(control_laws, requires_grad=False)` is the single source of truth for gradient support. The flag propagates into every `ControlLaw.finalize(device, num_outputs, requires_grad)` and `ControlLaw.state(num_outputs, device, requires_grad)` call, and from there into every internally allocated buffer (PID's `integral`; DiffIK's internal `Model`, `_jacobian`, `_qd_target_local`, …).

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
# A controller which runs both a PID and Differential Inverse Kinematics
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

    label: str
    """Unique-within-Controller string identifier. Set in __init__."""

    def __init__(self, *, label: str, **ports):
        """Validate port specs and stash them as (attr_name, port_indices)
        pairs. Cheap CPU-side work only: device buffers are allocated in
        finalize().

        Subclasses declare which kwargs they accept; missing required ports
        raise here, unknown ports raise here, and port_indices shape
        mismatches raise here. The output port carries the controller's
        num_outputs (= its port_indices length) — every per-DOF port is
        cross-checked against it at __init__."""

    def finalize(self, device: wp.Device, num_outputs: int,
                 requires_grad: bool = False) -> None:
        """Allocate device-side private buffers (e.g. internal Model +
        Jacobian buffers for articulated laws). Called by Controller after
        construction."""

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
        """Resolve port arrays via _resolve_input_array /
        _resolve_per_robot_array, launch kernels: read live data, write +=
        into outputs, populate next_state. Device is fixed at finalize()
        time."""
```

The `Controller` itself is small:

```python
class Controller:
    def __init__(self,
                 control_laws: list[ControlLaw],
                 requires_grad: bool = False,
                 device: wp.Device | None = None):
        """Validate that every law has a unique label and agrees on
        num_outputs, finalize() each, collect every law's outputs() into
        a flat list to be resolved + zeroed at step time."""

    def is_stateful(self) -> bool: ...
    def is_graphable(self) -> bool: ...

    def state(self) -> Controller.State:
        """Allocate composed state — keyed by each ControlLaw's label
        (None entries for stateless laws)."""

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
    control_law_states: dict[str, ControlLaw.State | None]
```

### Internal helpers (`newton/_src/controllers/utils.py`)

- `_normalize_port(spec, name)` — validate the 2-tuple `(str, wp.array[uint32])` spec at `__init__`, return `(attr_name, port_indices)`.
- `_resolve_input_array(source, attr_name, name)` — step-time `getattr` plus `wp.array` type check.
- `_resolve_per_robot_array(source, attr_name, dtype, name)` — step-time `getattr` plus dtype check (per-robot ports).

### Choosing between independent and coupled

If output `i` depends only on input `i` and per-DOF parameters, write it independent: 1D launch, no `Model`. Pure scalar math.

If each robot's outputs depend on its full configuration (Jacobians, mass matrix, kinematic chain, base velocity), write it coupled: take a `model_builder` (articulated case) or raw per-robot parameter vectors (mobile case) or both; finalize the builder once at finalize time; 2D launch over `(num_robots, outputs_per_robot)`.

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