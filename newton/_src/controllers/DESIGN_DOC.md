# Newton Controllers — Design Doc

**NOTE** This is a temporary document for design review purposes only. It should be deleted
and replaced by proper documentation before merging.

## Background

There are many controllers implemented across Isaac Lab/Sim. The goal of this module is to centralize them in Newton, re-implementing each so that they are completely vectorized and CUDA-graphable.

Below is a table summarizing existing Isaac Sim/Lab controllers, where their implementation currently lives, and whether there is a replacement already completed in this module.

| Controller | Current Repo | Completed? |
| --- | --- | --- |
| Differential IK | Isaac Lab | :heavy_check_mark: |
| PID | — | :heavy_check_mark: |
| Operational Space | Isaac Lab | :x:  |
| Joint Impedance | Isaac Lab | :x: |
| Differential Drive | Isaac Sim | :heavy_check_mark: |
| Holonomic Drive | Isaac Sim | :x: |
| Ackermann | Isaac Sim | :x: |

An open question is whether to broaden this "control toolbox" beyond the controllers in the table above to include general-purpose algorithms common in robotics. For example, a linear-filter class could implement:

- low-pass
- band-pass
- notch
---

## Architecture

The module exposes a single abstract base class — `Controller(ABC)` — that every concrete control law (`ControllerPID`, `ControllerDifferentialKinematics`, `ControllerDifferentialDrive`, …) subclasses directly. There is **no framework-level composition**: callers wanting to combine multiple control laws invoke each one's `compute()` in sequence themselves.

`Controller` declares the surface every law implements:

- `is_graphable() -> bool` — CUDA-graph compatibility predicate.
- `is_stateful() -> bool` — whether the law carries state between steps.
- `state()` — allocate a fresh per-step `State` (subclass-specific `@dataclass`), or `None` for stateless laws.
- `input_struct()` — allocate a fresh, auto-generated dataclass holding one `wp.array` field per input port.
- `output_struct()` — same idea for write ports.
- `compute(input_struct, output_struct, controller_state_now, controller_state_next, time_step) -> None`.

Users compose laws by simply instantiating many `Controller`s and composing the outputs in whatever way they desire.

---

## Controller flavors

Concrete controllers fall into two flavors based on **whether the control law needs a model of the robot**. This drives the constructor shape and what counts as a "goal."

### Decoupled controllers (DOFs independent)

`ControllerPID` is the canonical example. Each output depends only on the matching entries of its own ports: `output[i]` is a function of `joint_measured[i]`, `joint_target[i]`, the gains at `[i]`, and the state at `[i]` — nothing else. As a result it:

- **Carries no robot model.** It never needs topology, mass, or geometry — only `default_dof_indices` to know which slots to read and write.
- **Takes per-DOF goals.** A goal is a scalar target for a single controlled DOF (e.g. a target joint angle), in the same layout as every other port.
- **Has no concept of a "robot."** `num_robots` never appears; only `num_outputs`. One DOF is indistinguishable from the next, so the law vectorizes across any mix of robots for free.

### Model-based controllers (DOFs coupled, per-robot goals)

`ControllerDifferentialKinematics` and `ControllerDifferentialDrive` are the canonical examples. A robot's output DOFs are coupled — they cannot be solved independently of one another — and the goal is specified once per robot rather than per DOF. Such a controller:

- **Carries a model of the robot.** This can be a full `newton.ModelBuilder` of `N` topologically-identical articulations (`ControllerDifferentialKinematics`, finalized internally for FK + Jacobian evaluation) or just the handful of geometry parameters the law needs (`ControllerDifferentialDrive`'s per-robot `wheel_radius` / `wheel_base`). Either way it describes the robot the law reasons about.

### Examples

| | Decoupled (`PID`) | Model-based (`DifferentialKinematics`, `DifferentialDrive`) |
| --- | --- | --- |
| Robot model at construction | none | `ModelBuilder` or geometry parameters |
| Goal shape | per-DOF scalar | per-robot target |
| DOF coupling | none | within a robot |
| Knows about "robots"? | no (`num_outputs`) | yes (`num_robots`) |

---

## Controller Initialization

Every controller implementation declares a fixed set of inputs/outputs/parameters when it is initialized. There are two shapes for how these declarations happen:

### Live ports (`*_attr` + `*_idx`)

A live port connects the controller to an array that the caller supplies fresh on every step. Each live port is declared by a pair of constructor arguments: a `*_attr` string and an optional `*_idx` array.

```python
ControllerPID(
    default_dof_indices: wp.array,
    joint_measured_attr: str = "joint_q",
    joint_measured_idx: wp.array | None = None,
    joint_measured_rate_attr: str = "joint_qd",
    joint_measured_rate_idx: wp.array | None = None,
    joint_target_attr: str = "joint_target_q",
    joint_target_idx: wp.array | None = None,
    joint_target_rate_attr: str = "joint_target_qd",
    joint_target_rate_idx: wp.array | None = None,
    output_attr: str = "joint_f",
    output_idx: wp.array | None = None,
    ...
)
```

#### `*_attr` — which array

The string names an attribute on the object the caller passes to `compute()`; the controller resolves it each step via `getattr`. Different controllers default to different names, and the caller can override any of them.

#### `*_idx` — which slots

An optional `wp.array[wp.uint32]`. The kernel touches only `arr[idx[i]]`, leaving every other slot of the array untouched. When `*_idx` is `None`, the port falls back to `default_dof_indices`, so by default the read and write ports line up on the same slots.

Passing an explicit `*_idx` lets the controller read from one layout and write into another. In `test_controllers` the controller reads `joint_target_q[1, 2]` but writes `output[5, 7]`:

```python
ControllerPID(
    default_dof_indices=wp.array([5, 7], dtype=wp.uint32, device=device),
    joint_target_idx=wp.array([1, 2], dtype=wp.uint32, device=device),
    output_attr="output",
    ...
)
```

### Parameter ports (`wp.array | str`)

Parameter ports are configuration knobs (PID gains, DLS damping, IK bandwidth, wheel geometry, …). Each accepts either a `wp.array` (*baked*) or a `str` (*live*).

```python
ControllerPID(
    default_dof_indices: wp.array,
    kp: wp.array | str | None = None,
    kd: wp.array | str | None = None,
    ki: wp.array | str | None = None,
    integral_max: wp.array | str | None = None,
    ...
)
```

`ControllerPID(kp=wp.array(...))` is an example of **baked** parameters; `ControllerPID(kp="kp")` is an example of **live** parameters.

#### Baked parameters (`wp.array`)

The controller copies the array at construction. Mutating the user's original afterward has no effect. Length must be `num_outputs` (per-DOF) or `num_robots` (per-robot); dtype `wp.float32`. Read in natural order (`arr[i]`) — no `_idx` override.

`example_pid_pendulum` bakes PID gains; `example_diff_drive_swarm` bakes per-robot `wheel_radius` / `wheel_base`:

```python
ControllerPID(
    default_dof_indices=default_dof_indices,
    kp=wp.array([3600.0, 1200.0], dtype=wp.float32, device=device),
    kd=wp.array([320.0, 160.0], dtype=wp.float32, device=device),
)

ControllerDifferentialDrive(
    num_robots=16,
    wheel_radius=wp.full(16, 0.05, dtype=wp.float32, device=device),
    wheel_base=wp.full(16, 0.2, dtype=wp.float32, device=device),
    default_dof_indices=default_dof_indices,
)
```

#### Live parameters (`str`)

If a parameter is given as a string, then it is expected that the parameter is given as at every step as part of the input object. Each `compute()` resolves it with `getattr` and reads `arr[i]` in natural order. Same length/dtype rules as baked.

For example, the test file `test_controllers` passes `kp="kp"` so gains can change between steps without reconstructing the controller; `ControllerDifferentialKinematics` accepts `bandwidth="my_band"` the same way:

```python
ControllerPID(
    default_dof_indices=default_dof_indices,
    kp="kp",
    ...
)

ControllerDifferentialKinematics(
    model_builder=builder,
    controlled_site_label="tool",
    default_dof_indices=default_dof_indices,
    bandwidth="my_band",
    ...
)
```

---

## State

Stateful laws (PID) define a nested `@dataclass class State(Controller.State)` with their internal buffers (PID's `integral`). The abstract function `state()` allocates a fresh instance with zero-initialized fields. Users typically double-buffer:

```python
s0, s1 = controller.state(), controller.state()
for ... :
    controller.compute(input_struct, output_struct, s0, s1, time_step=dt)
    s0, s1 = s1, s0
```

Stateless laws (`ControllerDifferentialKinematics`, `ControllerDifferentialDrive`) return `None` from `state()` and accept `None` for both state slots.

---

## Struct factories

The inputs and outputs of particular controllers are highly dependent on the specific algorithm. This can be contrasted to `newton.actuators`, where the input is predictably a set of desired position/velocity/effort floats, and the output is an effort float.

To simplify working with different controllers, each `Controller` implements `input_struct()` and `output_struct()` functions to auto-allocate correctly-sized struct instances which meet that controllers interface. Field names match the user's `*_attr` strings. Baked parameters are absent from either struct (they live on the controller). Each field is sized minimally for the controller's view — `max(idx)+1` for ports with an explicit `_idx`, otherwise the natural-order length (`num_outputs` / `num_robots`).

Users can:

- Use the returned struct as-is (each call gives a fresh allocation):

```python
inp = controller.input_struct()       # fresh wp.zeros fields
inp.joint_target_q.assign([0.6, -1.2])
out = controller.output_struct()
controller.compute(inp, out, s0, s1, time_step=dt)
```

- Reassign fields to point at live sim buffers:

```python
inp = controller.input_struct()
inp.joint_q = state.joint_q           # share the solver's array, no copy
out = controller.output_struct()
out.joint_f = control.joint_f         # write directly into the solver's input
```

- Skip the factory entirely and pass any duck-typed object with the right attributes:

```python
inp = SimpleNamespace(joint_q=state.joint_q, joint_qd=state.joint_qd,
                      joint_target_q=target_q, joint_target_qd=target_qd)
out = SimpleNamespace(joint_f=control.joint_f)
controller.compute(inp, out, s0, s1, time_step=dt)
```

---

## Concrete controllers

### `ControllerPID`

Per-DOF PID with symmetric anti-windup clamping. Stateful (`State.integral` shape `[num_outputs]`, float32).

```python
output[output_idx[i]] = kp[i] * (joint_target - joint_measured)
                      + ki[i] * clamp(integral + pos_err * dt, -integral_max[i], +integral_max[i])
                      + kd[i] * (joint_target_rate - joint_measured_rate)
```

Constructor:

```python
ControllerPID(
    default_dof_indices: wp.array,
    kp: wp.array | str | None = None,
    kd: wp.array | str | None = None,
    ki: wp.array | str | None = None,
    integral_max: wp.array | str | None = None,
    joint_measured_attr: str = "joint_q",
    joint_measured_idx: wp.array | None = None,
    joint_measured_rate_attr: str = "joint_qd",
    joint_measured_rate_idx: wp.array | None = None,
    joint_target_attr: str = "joint_target_q",
    joint_target_idx: wp.array | None = None,
    joint_target_rate_attr: str = "joint_target_qd",
    joint_target_rate_idx: wp.array | None = None,
    output_attr: str = "joint_f",
    output_idx: wp.array | None = None,
    device: Any = None,
    requires_grad: bool = False,
)
```

Omitted gain ports default to zero arrays of length `num_outputs`. `integral_max` defaults to `+inf` per DOF (no clamping).

### `ControllerDifferentialKinematics`

One-step differential IK for one end-effector per robot. The user passes a `newton.ModelBuilder` with `N` topologically-identical articulations; the controller finalizes that internally for FK + Jacobian evaluation. Stateless (`state()` returns `None`).

Per-robot solve (`IkMethod.DAMPED_LEAST_SQUARES`, default):

```
e        = [target_pos - site_pos ;  2 * sign(q_err.w) * q_err.xyz]
A        = J_site J_site^T + lambda^2 * I_6                     (6×6 SPD)
y        = A^{-1} e                                             (tile Cholesky)
q_dot    = bandwidth * J_site^T y
output_q = q_current + q_dot * dt
```

With `IkMethod.TRANSPOSE`: `q_dot = bandwidth * J_site^T e`.

`solver_damping` is `lambda`; `bandwidth` is the scalar multiplier on the solve output (both per-robot gain ports).

**Tape-safe forward, zero-grad through the solve.** Every kernel except `_cholesky_solve_kernel` is autograd-able by default; the solve uses `wp.tile_cholesky` / `wp.tile_cholesky_solve` whose registered adjoints return zero gradients in Warp 1.14.0, so that one kernel is marked `enable_backward=False`. Revisit when upstream tile-Cholesky backward lands.

Constructor:

```python
ControllerDifferentialKinematics(
    model_builder: ModelBuilder,
    controlled_site_label: str,
    default_dof_indices: wp.array,
    bandwidth: wp.array | str,
    solver_damping: wp.array | str | None = None,
    ik_method: IkMethod = IkMethod.DAMPED_LEAST_SQUARES,
    target_pos_attr: str = "site_target_position",
    target_pos_idx: wp.array | None = None,
    target_quat_attr: str = "site_target_quaternion",
    target_quat_idx: wp.array | None = None,
    joint_measurement_attr: str = "joint_q",
    joint_measurement_idx: wp.array | None = None,
    joint_measurement_rate_attr: str = "joint_qd",
    joint_measurement_rate_idx: wp.array | None = None,
    joint_target_q_attr: str = "joint_target_q",
    joint_target_q_idx: wp.array | None = None,
    joint_target_qd_attr: str = "joint_target_qd",
    joint_target_qd_idx: wp.array | None = None,
    device: Any = None,
    requires_grad: bool = False,
)
```

`num_robots = model_builder.articulation_count`. `default_dof_indices` length is `num_robots * dofs_per_robot`, laid out `[r0_d0, r0_d1, …, r1_d0, …]`. `solver_damping` defaults to `DEFAULT_SOLVER_DAMPING` (`0.05`) per robot when `None`.

### `ControllerDifferentialDrive`

Vectorized unicycle differential-drive: per robot, converts body-frame `(linear_speed, angular_speed)` into left/right wheel angular velocities and writes them into a `joint_target_qd`-style output array via `default_dof_indices` (`[r0_left, r0_right, r1_left, r1_right, …]`). Commands and wheel rates are clamped per robot; parameter ports (`wheel_radius`, `wheel_base`, speed limits) are per-robot baked or live arrays. Stateless (`state()` returns `None`). Example: `python -m newton.examples diff_drive_swarm`.

Constructor:

```python
ControllerDifferentialDrive(
    num_robots: int,
    wheel_radius: wp.array | str,
    wheel_base: wp.array | str,
    default_dof_indices: wp.array,
    max_linear_speed: wp.array | str | None = None,
    max_angular_speed: wp.array | str | None = None,
    max_wheel_speed: wp.array | str | None = None,
    linear_speed_attr: str = "linear_speed_command",
    linear_speed_idx: wp.array | None = None,
    angular_speed_attr: str = "angular_speed_command",
    angular_speed_idx: wp.array | None = None,
    joint_target_qd_attr: str = "joint_target_qd",
    joint_target_qd_idx: wp.array | None = None,
    device: Any = None,
    requires_grad: bool = False,
)
```

Omitted clamp ports default to `+inf` per robot (no clamping). `default_dof_indices` must be `wp.array[uint32]` with length `2 * num_robots`.

---

## End-to-end example

A minimal PID loop driving two joints to a fixed target (condensed from `example_pid_pendulum`):

```python
controller = ControllerPID(
    default_dof_indices=wp.array([0, 1], dtype=wp.uint32),
    kp=wp.array([3600.0, 1200.0], dtype=wp.float32),
    kd=wp.array([320.0, 160.0], dtype=wp.float32),
)

# Read ports default to joint_q / joint_qd; the write port defaults to joint_f.
inp = controller.input_struct()
inp.joint_target_q.assign([0.6, -1.2])      # setpoint [rad]
out = controller.output_struct()
out.joint_f = control.joint_f               # write straight into the solver's array

s0, s1 = controller.state(), controller.state()   # PID integral, double-buffered

for _ in range(num_steps):
    inp.joint_q = state_0.joint_q           # rebind to the current sim state
    inp.joint_qd = state_0.joint_qd
    controller.compute(inp, out, s0, s1, time_step=dt)
    s0, s1 = s1, s0
    solver.step(state_0, state_1, control, contacts, dt)
    state_0, state_1 = state_1, state_0
```