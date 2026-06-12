# Newton Controllers — Design Doc

## Background

There are many controllers implemented across Isaac Lab/Sim. The goal of this module is to centralize them in Newton, re-implementing each so that they are completely CUDA-graphable and vectorized.

| Controller | Current Repo | Completed? |
| --- | --- | --- |
| Differential IK | Isaac Lab | :heavy_check_mark: |
| PID | — | :heavy_check_mark: |
| Operational Space | Isaac Lab | :x:  |
| Joint Impedance | Isaac Lab | :x: |
| Differential Drive | Isaac Sim | :heavy_check_mark: |
| Holonomic Drive | Isaac Sim | :x: |
| Ackermann | Isaac Sim | :x: |

Further, we may want to add general-purpose algorithms common in robotics (linear filters: low-pass, band-pass, notch, …).

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

## Ports

Every controller declares a fixed set of *ports* — named pieces of data it reads or writes per step. There are two port shapes:

### Live ports (`*_attr` + `*_idx`)

For data that comes from the simulation each step (joint positions, target poses, …):

- `*_attr: str` — attribute name on the user-supplied `input_struct` / `output_struct`. Resolved at step time via `getattr(struct, attr_name)`.
- `*_idx: wp.array[wp.uint32] | None` — kernel reads `arr[idx[i]]`. When `None`, the controller's `default_dof_indices` (per-DOF ports) or natural-order `wp.arange(num_robots)` (per-robot ports) is used.

The pattern lets a controller view an arbitrary slice of a larger sim-side array: a per-DOF index of `[5, 7]` writes only those two slots and leaves the others alone.

### Parameter ports (`wp.array | str`)

For configuration knobs (PID gains, DLS damping, IK bandwidth) the user can pick:

- Pass a `wp.array` — *baked*: the controller takes a copy at construction. Mutating the user's original later has no effect. Must have length `num_outputs` (per-DOF) or `num_robots` (per-robot) and dtype `wp.float32`.
- Pass a `str` — *live*: at step time the controller resolves `getattr(input_struct, value)` and reads that array. Same length/dtype requirement.

Parameter ports are always read in natural order (`arr[i]`) — no `_idx` override. The user's "leave them in order" choice keeps the constructor surface flat and matches how gain arrays are typically authored.

---

## State

Stateful laws (PID) define a nested `@dataclass class State(Controller.State)` with their internal buffers (PID's `integral`). `state()` allocates a fresh instance with zero-initialized fields. Users typically double-buffer:

```python
s0, s1 = controller.state(), controller.state()
for ... :
    controller.compute(input_struct, output_struct, s0, s1, time_step=dt)
    s0, s1 = s1, s0
```

Stateless laws (`ControllerDifferentialKinematics`, `ControllerDifferentialDrive`) return `None` from `state()` and accept `None` for both state slots.

---

## Struct factories

`input_struct()` and `output_struct()` allocate auto-generated struct instances with one `wp.zeros` field per live port the user declared. Field names match the user's `*_attr` strings (and live-gain strings). Baked parameters are absent (they live on the controller). Each field is sized minimally for the controller's view — `max(idx)+1` for ports with an explicit `_idx`, otherwise the natural-order length (`num_outputs` / `num_robots`).

Users can:

- Use the returned struct as-is (each call gives a fresh allocation).
- Reassign fields to point at live sim buffers (`s.joint_q = state.joint_q`).
- Skip the factory entirely and pass any duck-typed object with the right attributes — `SimpleNamespace`, a hand-written dataclass, even `newton.State` if its field names happen to match.

---

## Concrete controllers

### `ControllerPID`

Per-DOF PID with symmetric anti-windup clamping.

```python
output[output_idx[i]] = kp[i] * (joint_target - joint_measured)
                      + ki[i] * clamp(integral + pos_err * dt, -integral_max[i], +integral_max[i])
                      + kd[i] * (joint_target_rate - joint_measured_rate)
```

Constructor (abbreviated):

```python
ControllerPID(
    kp: wp.array | str,
    kd: wp.array | str,
    ki: wp.array | str,
    integral_max: wp.array | str,
    default_dof_indices: wp.array,
    joint_measured_attr: str = "joint_q",
    joint_measured_idx: wp.array | None = None,
    joint_target_attr: str = "joint_target_q",
    joint_target_idx: wp.array | None = None,
    output_attr: str = "joint_f",
    output_idx: wp.array | None = None,
    ...   # mirror for *_rate
    device=None,
    requires_grad=False,
)
```

`State.integral` shape `[num_outputs]`, float32.

### `ControllerDifferentialKinematics`

One-step damped-least-squares differential IK for one end-effector per robot. The user passes a `newton.ModelBuilder` with `N` topologically-identical articulations; the controller finalizes that internally for FK + Jacobian evaluation.

Per-robot solve:

```
e        = [target_pos - site_pos ;  2 * sign(q_err.w) * q_err.xyz]
A        = J_site J_site^T + lambda^2 * I_6                     (6×6 SPD)
y        = A^{-1} e                                             (tile Cholesky)
q_dot    = bandwidth * J_site^T y
output_q = q_current + q_dot * dt
```

`solver_damping` is `lambda`; `bandwidth` is the scalar multiplier on the solve output (both per-robot gain ports).

**Tape-safe forward, zero-grad through the solve.** Every kernel except `_cholesky_solve_kernel` is autograd-able by default; the solve uses `wp.tile_cholesky` / `wp.tile_cholesky_solve` whose registered adjoints return zero gradients in Warp 1.14.0, so that one kernel is marked `enable_backward=False`. Revisit when upstream tile-Cholesky backward lands.

Stateless (`state()` returns `None`).

### `ControllerDifferentialDrive`

Vectorized unicycle differential-drive: per robot, converts body-frame `(linear_speed, angular_speed)` into left/right wheel angular velocities and writes them into a `joint_target_qd`-style output array via `default_dof_indices` (`[r0_left, r0_right, r1_left, r1_right, …]`). Commands and wheel rates are clamped per robot; parameter ports (`wheel_radius`, `wheel_base`, speed limits) are per-robot baked or live arrays.

Stateless (`state()` returns `None`). Example: `python -m newton.examples diff_drive_swarm`.