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

### Port forms

Every per-DOF port accepts one of two specs at `__init__`:

| Spec | Meaning | Kernel access |
|---|---|---|
| `"attr_name"` | use the controller's own `indices` as port_indices | `getattr(source, name)[indices[i]]` |
| `("attr_name", port_indices)` | use a custom `port_indices` array | `getattr(source, name)[port_indices[i]]` |

The bare-string form handles the common case where every port reads from the same flat layout the controller writes to. The tuple form covers the layout-mismatch case (e.g. reading densely packed gains while writing into sparse global output slots).

**Per-group ports** (per-robot, not per-DOF — e.g. DiffIK's `target_pos`, `damping`, `gain`) accept just `"attr_name"`. At step the resolver checks `shape == (num_robots,)` and the documented dtype.

### Validation

| Stage | Check |
|---|---|
| `__init__` (per-DOF) | `spec` is `str` or `(str, wp.array[uint32])`. For the tuple form, `port_indices.shape == indices.shape`. |
| `__init__` (per-group) | `spec` is `str`. |
| `step` (per-DOF) | `getattr(source, name)` resolves to a `wp.array`. Shape/dtype mismatches surface at the kernel launch with Warp's diagnostic. |
| `step` (per-group) | `getattr(source, name)` resolves to a `wp.array` with shape `(num_robots,)` and the documented dtype. |

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
controller.step(input, output, cs0, cs1, dt)
```

---

## Two flavors of ControlLaw

Subclasses fall into two categories by whether each output slot is independent of its siblings. The base class is shared; the difference shows up in kernel-launch shape and what constructor arguments are needed.

### Independent per-DOF

Output `i` depends only on input `i` plus per-DOF parameters. Examples: `ControlLawPID`, low-pass filter, saturation, feedforward.

- Construction takes only the port specs plus an `indices` array of global DOF indices.
- Kernel launches 1D with `dim=len(indices)`; `i = wp.tid()`.
- No `newton.Model` required.

### Coupled / structural

Each robot's outputs depend on the full state of that robot. Examples: `ControlLawDifferentialIK`, `ControlLawGravityComp`, `ControlLawDifferentialDrive`, `ControlLawHolonomic`, `ControlLawOperationalSpace`.

There are two sub-flavors here, distinguished by whether they need an articulated model:

- **Articulated coupled** controllers take a `model_builder: newton.ModelBuilder` containing `K = model_builder.articulation_count` topologically-identical articulations. The K articulations share DOF count, link/joint count, and joint types; they may differ in physical parameters (mass, inertia, friction, joint limits) and in per-articulation site placement (see below). At `finalize()` the controller replicates the builder `R = len(indices) // model_builder.joint_dof_count` times via `ModelBuilder.replicate`, finalizes it on the chosen device, and stores the resulting `Model` plus a working `State` for the per-step `eval_fk` / `eval_jacobian` calls.
  - The K=1 case is a single template. K>1 is the "categorical variants" case: each variant is a distinct articulation in the template, replicated R times to fill out the batch. Useful for RL with structural domain randomization (e.g. different gripper lengths).
  - The constraint is `len(indices) % model_builder.joint_dof_count == 0`. `num_robots = K * R`.
- **Mobile coupled** controllers take raw geometry instead — wheel radius, axle width, wheel-mapping matrix. There's no Newton `Model` for a wheel base in the current schema. `num_robots = len(indices) // wheels_per_robot`.

For both: `indices[r * outputs_per_robot + j]` is the global output index of robot `r`'s local slot `j` (`outputs_per_robot = dofs_per_robot` for articulated, `wheels_per_robot` for mobile). Robots are contiguous in the flat layout. Kernels launch 2D with `dim=(num_robots, outputs_per_robot)`; inside, `robot, local_slot = wp.tid()`.

### Variant-interleaved replication layout

After `ModelBuilder.replicate(R)`, robot `r` corresponds to variant `r % K` of replication `r // K`. So with `K=4, R=3` the robots are `[v0, v1, v2, v3, v0, v1, v2, v3, v0, v1, v2, v3]`. Per-group input arrays (`target_pos`, `target_quat`, `damping`, `gain`, …) of length `num_robots` follow this layout.

This convention is the one `ModelBuilder.replicate` already uses, so the controller's internal `Model` and any user-built scene that calls `replicate` on the same template will line up automatically.

`Controller` is flavor-agnostic. Each ControlLaw declares its outputs via `outputs() -> list[(attr_name, port_indices)]`; the Controller's zero pass walks the union.

---

## Output accumulation

At the start of each `step()` the Controller resolves every output binding's `attr_name` against the passed-in `output`, then zeros the slots indicated by each binding's `port_indices`. The ControlLaws then run serially, in registration order; each writes via `+=` into its output arrays:

```python
# Inside a per-DOF kernel:
i = wp.tid()
out_idx = output_indices[i]
output_array[out_idx] += contribution
```

Composition is sum-of-contributions: a PD term + a gravity-compensation term + a feedforward term all writing to the same `joint_f` produce their pointwise sum. There are no overlap checks — laws compose at the user's discretion. Two laws binding overlapping slots have those slots zeroed twice in the upfront pass; idempotent and avoids any wp.array identity comparison.

**Multi-output laws** (e.g. `ControlLawDifferentialIK` writes both `output_qd` and `output_q`) declare multiple bindings via `outputs()`. Each is treated identically for zero / accumulate purposes. All of a law's outputs must share the same outer length (`num_outputs`), so a single reset mask covers the law.

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

## Reset

Reset writes a per-ControlLaw `reset_state` into the live state at slots flagged by a boolean mask. Each `ControlLaw` exposes a public `reset_state: ControlLaw.State` allocated at `finalize()` and zero-initialized; the user mutates it to customize what subsequent resets write.

```python
pid = nc.ControlLawPID(...)
controller = nc.Controller([pid])           # finalize allocates pid.reset_state (zeros)
pid.reset_state.integral.fill_(0.5)         # bias the reset target

controller.reset(state_0, mask=reset_mask)  # masked slots get 0.5
```

### Mask shape and the slot abstraction

`mask` is a `wp.array[wp.bool]` of length equal to the ControlLaw's `num_outputs` — the shared outer length of every binding the law declares via `outputs()`. `mask[i] = True` means "reset output slot `i`."

A *slot* is one unit of meaning to the law. For PID, that's a single scalar (the `integral[i]` value). For a controller that emits a `wp.vec3` position + a `wp.quat` orientation per robot, both output arrays have length `num_robots`, and `mask[i] = True` resets both arrays' slot `i` together. For a filter that carries K floats of history per slot, the law's reset kernel knows the stride.

Each law writes its own reset kernel; it iterates over `num_outputs` threads and reads `mask[i]` directly — the mask is already in the law's local frame, no indirection through `indices`.

### Controller-wide invariant

A `Controller` requires every ControlLaw it composes to agree on `num_outputs`, so a single Controller-wide mask makes sense across the group. This is validated at `Controller.__init__`. `controller.reset(state, mask)` walks `(law, sub_state)` pairs and dispatches `law.reset(sub_state, mask)` for every non-`None` sub-state.

---

## Differentiability

`Controller.__init__(control_laws, requires_grad=False)` is the single source of truth for gradient support. The flag propagates into every `ControlLaw.finalize(device, num_outputs, requires_grad)` and `ControlLaw.state(num_outputs, device, requires_grad)` call, and from there into every internally allocated buffer (PID's `integral` and `reset_state`; DiffIK's replicated `Model`, `_jacobian`, `_qd_target_local`, …).

User-supplied input arrays (`measurement`, `target_pos`, `kp`, …) carry their own `requires_grad` — the laws don't own those allocations.

Kernels use the default `@wp.kernel` decorator, which records adjoints onto an active `wp.Tape`. The module is tape-agnostic: the caller (Isaac Lab, a custom training loop) wraps the relevant block in `wp.Tape()` and runs `tape.backward(loss=…)` on its own. This matches `newton.actuators.Actuator.step`.

### Per-law status

- **`ControlLawPID`** — fully differentiable end-to-end. Gradients flow from `output` back through `measurement`, `measurement_rate`, `setpoint`, `setpoint_rate`, `kp`, `ki`, `kd`, `integral_max`.
- **`ControlLawDifferentialIK`** — tape-safe; forward-only through the DLS solve. The compute chain is split into per-element kernels (gather → build site Jacobian → build the 6×6 DLS matrix → q_dot back-projection → accumulate) plus one tile-cooperative Cholesky-solve kernel. Every kernel except the solve is autograd-able by default. The solve uses `wp.tile_cholesky` + `wp.tile_cholesky_solve`; their adjoints are advertised but return zero gradients in Warp 1.14.0 (verified directly: forward correct, backward gives zero gradients on both A and the rhs). The solve kernel is therefore marked `enable_backward=False` to make the zero-gradient behaviour explicit. Gradients propagate from the loss back to `output_qd` and stop at the solve. Useful for RL pipelines that wrap a whole sim in `wp.Tape` without needing IK gradients; not yet usable for end-to-end diff-physics training through the IK. Revisit when upstream `wp.tile_cholesky` backward lands.

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
```

The bridge is `ctrl_out` ↔ `sim_control`: the user typically points `ctrl_out`'s attributes at the same `wp.array`s exposed on `sim_control` (e.g. `joint_target_q`, `joint_f`). The controllers module never introspects Newton sim objects — that wiring is the user's responsibility, and it stays explicit at the call site.

When a control law produces a target that the actuator should track at constant gains (e.g. DiffIK outputs `joint_target_q` consumed by MuJoCo's built-in joint-position PD), the actuator step can be skipped entirely; the controller writes straight into `sim_control.joint_target_q`. See `newton/examples/controllers/example_diff_ik_panda.py` for that pattern.

---

## Subclassing a ControlLaw

The base class is a minimal contract. Subclasses are free to choose their kernel shapes, decide whether to take a `model_builder` argument, and pick whatever divisibility constraint their replication scheme needs.

```python
class ControlLaw:
    @dataclass
    class State:
        """Pure data container. Subclasses declare their fields (integral
        arrays, history buffers). No methods."""

    # Allocated at finalize() to a zero-initialized State. Users mutate this
    # attribute to customize what subsequent reset() calls write. Stateless
    # laws leave it as None.
    reset_state: ControlLaw.State | None

    indices: wp.array[wp.uint32]
    """Global output indices this law writes to. Set in __init__."""

    def __init__(self, **ports):
        """Validate port specs and stash them as (attr_name, port_indices)
        pairs (per-DOF) or attr_name strings (per-group). Cheap CPU-side
        work only: device buffers are allocated in finalize().

        Subclasses declare which kwargs they accept; missing required ports
        raise here, unknown ports raise here, and port_indices shape
        mismatches raise here. Scalar value-range checks (kp >= 0 etc.)
        live in a config layer above the ControlLaw to avoid a synchronous
        device-to-host copy."""

    def finalize(self, device: wp.Device, num_outputs: int,
                 requires_grad: bool = False) -> None:
        """Allocate device-side private buffers (e.g. internal Model +
        Jacobian buffers for articulated laws) and self.reset_state
        (zero-initialized via self.state(num_outputs, device,
        requires_grad) for stateful laws). Called by Controller after
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
                input, output,
                state: ControlLaw.State | None,
                next_state: ControlLaw.State | None,
                dt: float) -> None:
        """Resolve port arrays via _resolve_input_array / _resolve_per_group_array,
        launch kernels: read live data, write += into outputs, populate
        next_state. Device is fixed at finalize() time."""

    def reset(self, state: ControlLaw.State,
              mask: wp.array[wp.bool]) -> None:
        """Update `state` from `self.reset_state` where mask is True.
        mask is in the law's local frame (length num_outputs); no
        indirection through self.indices."""
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

    def step(self, input, output,
             current_state: Controller.State,
             next_state: Controller.State,
             dt: float) -> None:
        """1. Resolve each output binding's attr_name against `output`,
              and zero the declared slots. Two laws binding overlapping
              slots get zeroed twice — idempotent.
           2. For each law in registration order, call compute(input,
              output, cur, nxt, dt), which += writes into outputs."""

    def reset(self, state: Controller.State,
              mask: wp.array[wp.bool]) -> None:
        """For each (law, sub_state) pair where sub_state is not None,
        call law.reset(sub_state, mask). The mask is in the laws' shared
        local frame (length controller.num_outputs)."""
```

```python
@dataclass
class Controller.State:
    control_law_states: list[ControlLaw.State | None]
```

### Internal helpers (`newton/_src/controllers/utils.py`)

- `_normalize_port(spec, control_law_indices, name)` — validate `str` or `(str, port_indices)` at `__init__`, return `(attr_name, port_indices)`.
- `_normalize_per_group_port(spec, name)` — validate the per-group `str` spec at `__init__`.
- `_resolve_input_array(source, attr_name, name)` — step-time `getattr` plus `wp.array` type check.
- `_resolve_per_group_array(source, attr_name, num_robots, dtype, name)` — step-time `getattr` plus per-group shape + dtype check.

Replication uses `newton.ModelBuilder.replicate` directly at `finalize()` time; no separate helper. K-articulation homogeneity is asserted via `joint_dof_count % articulation_count == 0` in the consuming law's `__init__`; users are responsible for matching link/joint counts across articulations.

### Choosing between independent and coupled

If output `i` depends only on input `i` and per-DOF parameters, write it independent: 1D launch, no `Model`, `len(indices) = N`. Pure scalar math.

If each robot's outputs depend on its full configuration (Jacobians, mass matrix, kinematic chain, base velocity), write it coupled: take a `model_builder` or raw geometry, replicate at finalize, 2D launch over `(num_robots, outputs_per_robot)`.

The reference implementations are `controller_pid.py` (independent) and `controller_diff_ik.py` (coupled, articulated).

---

## ControlLawDifferentialIK: implementation notes

Worth a deeper look because the same patterns will recur in future articulated coupled controllers.

### Solve form

Per robot, with `J_site` the 6×N site-frame Jacobian:

```
e        = [target_pos - site_pos ;  2 * sign(q_err.w) * q_err.xyz]
A        = J_site J_site^T + lambda^2 * I_6      (6x6 SPD)
L L^T    = A                                     (Cholesky)
L L^T y  = e                                     (solve)
q_dot    = gain * J_site^T y
```

`q_err = target_quat * conj(site_quat)`; `gain` is a per-robot scalar applied uniformly to every output DOF after the DLS solve. `output_q` is written as `q_current + q_dot * dt`.

### Compute chain

Five Warp kernels, each as small and focused as possible:

1. `_gather_local_kernel` — pull `joint_q` / `joint_qd` slices from the user-supplied `input` arrays into the controller's internal `Model` `State` buffers via the controller's `indices`.
2. `_build_site_jacobian_kernel` — run `eval_fk` + `eval_jacobian` on the internal `Model`, then convert Newton's COM-frame Jacobian rows to site-frame via `v_site = v_com + cross(omega, site - com)`. Outputs the 6×N site Jacobian and the 6-vector task-space error.
3. `_build_dls_matrix_kernel` — form `A = J_site J_site^T + λ² I` per element.
4. `_cholesky_solve_kernel` — tile-cooperative `wp.tile_cholesky` / `wp.tile_cholesky_solve` to produce `y = A⁻¹ e`. The only kernel that uses tile primitives.
5. `_qd_from_y_kernel` + `_accumulate_outputs_kernel` — back-project `q_dot = gain · J_site^T y` and accumulate into `output_qd` and `output_q`.

The split is deliberate: the autograd-friendly per-element kernels see only contiguous reads/writes and stay fully differentiable. The tile kernel sees only pure tile primitives (`tile_load → tile_cholesky → tile_cholesky_solve → tile_store`). The cost of the split is a few extra kernel launches; the benefit is that the chain is autograd-able by default everywhere except the tile-Cholesky kernel itself (whose adjoint is broken in Warp 1.14.0, see *Differentiability*).

### Site lookup is per-variant

The user identifies the end-effector by a site **label** added to the template builder via `builder.add_site(body, label="…", xform=…)`. The controller's `__init__` finds every shape in the template labeled with that name, asserts there's exactly one per articulation (K total), and stashes both the within-variant body index and the body-local xform per variant.

At `finalize()` these fan out into length-`num_robots` Warp arrays (`_ee_body_per_robot`, `_ee_link_per_robot`, `_site_xform_per_robot`) in the variant-interleaved layout. The site-Jacobian kernel indexes them by robot.

This lets each of the K variants legitimately have different end-effector geometry — e.g. a Franka template where variant 0 has a short gripper and variant 1 has a long one, all under the same controller. The site label is the contract; positions and body indices may differ.

### Restrictions

- Scalar-DOF joints only (revolute, prismatic). `joint_q.shape == joint_qd.shape == (joint_dof_count,)`. Floating bases and ball joints aren't supported.
- One end-effector per controller. Multi-task IK composes via two `ControlLawDifferentialIK` instances under one `Controller`.
- Left-damped LS only. No nullspace projection — compose a separate law if you need joint centering or limit avoidance.

### Pitfalls

- `wp.quat_mul` doesn't exist in Warp — use the `q1 * q2` operator.
- The unicode `×` triggers ruff RUF002/RUF003. Use `cross(...)` or `x` in comments.

---

## Roadmap

ControlLaws currently implemented:

1. **`ControlLawPID`** — independent per-DOF, fully differentiable.
2. **`ControlLawDifferentialIK`** — coupled articulated, tape-safe (forward-only through the solve).

Planned, ordered by implementation difficulty / blocking dependencies:

### `ControlLawDifferentialDrive`

- **Category:** coupled, mobile.
- **Math:** input `(v, ω)` (linear m/s, angular rad/s) → wheel velocities `(ω_L, ω_R)` via `ω_L = (v − ω·L/2) / r`, `ω_R = (v + ω·L/2) / r` where `r` = wheel radius, `L` = axle width.
- **Construction:** takes raw scalars `wheel_radius: float`, `axle_width: float` directly. `wheels_per_robot = 2`. The user-supplied `indices` array orders `(left, right, left, right, …)` per robot.
- **Open:** output port — wheel velocity (`joint_target_qd`) or wheel torque via an inner PI loop? Per-robot input `(v, ω)` as separate `target_v` / `target_omega` per-group ports, or combined `target_twist: wp.array[wp.vec2]`?

### `ControlLawHolonomic`

- **Category:** coupled, mobile.
- **Math:** input body twist `(v_x, v_y, ω)` → wheel velocities for an N-wheeled omni base (3-wheel Kiwi, 4-wheel mecanum). Mapping is `wheel_velocities = W(geometry) · [v_x, v_y, ω]ᵀ`.
- **Construction:** takes a `wheel_geometry: wp.array2d[float]` of shape `(wheels_per_robot, 3)` directly. Each row encodes one wheel's contribution to the velocity-mapping matrix. Helpers to build the matrix for known configurations (Kiwi, mecanum) live alongside the controller.
- **Open:** which geometry helpers to ship (Kiwi / mecanum / generic builder)? Output port — wheel velocity or wheel torque?

### `ControlLawGravityComp` *(gated on Newton's generic inverse-dynamics function)*

- **Category:** coupled, articulated.
- **Math:** `τ_g = −∂U_gravity/∂q`. Equivalent to inverse dynamics with `q̇ = q̈ = 0`.
- **Open:** port for the gravity vector (default `(0, 0, −9.81)`, allow override). When the inverse-dynamics function lands, confirm its signature matches the per-robot batched layout articulated controllers assume.

### `ControlLawJointImpedance` *(gated on the inverse-dynamics function for any variant including `g(q)` or `h(q,q̇)`)*

- **Category:** coupled, articulated.
- **Math:** `τ = M(q)(q̈_d + K_d (q̇_d − q̇) + K_p (q_d − q)) + h(q, q̇)` or simpler stiffness-only variants.
- **Open:** which variants to ship — full impedance, PD + gravity, Cartesian impedance? `M(q)` is public via `newton.eval_mass_matrix`; `h(q,q̇)` waits on inverse dynamics.

### `ControlLawOperationalSpace` *(gated on the inverse-dynamics function)*

- **Category:** coupled, articulated, task-space.
- **Math:** task-space inertia `Λ = (J M⁻¹ Jᵀ)⁻¹`; task force `F = Λ (ẍ_d + K_d (ẋ_d − ẋ) + K_p (x_d − x)) + μ + p`; torque `τ = Jᵀ F + (I − Jᵀ J̄ᵀ) τ_null` where `J̄ = M⁻¹ Jᵀ Λ`. `J`, `M` are public; `μ`, `p` wait on inverse dynamics.
- **Open:** monolithic OSC or split (TaskSpaceForce + JointMapping)? Inertia-weighted vs. plain damped pseudoinverse — ship both, or pick one? Nullspace handling — separate `ControlLawNullspaceProjection` block that accumulates, or built in?

---

## Open design questions

- **Promoting a small-system solver helper.** `ControlLawDifferentialIK`'s `_cholesky_solve_kernel` (tile-cooperative 6×6 SPD solve) lives inline in `controller_diff_ik.py`. If a second coupled law needs the same primitive (likely candidates: `ControlLawOperationalSpace`'s 6×6 task-space inertia inverse), extract to `controllers/utils.py` once the second consumer exists.
- **Composition graph.** Today `Controller` runs laws in registration order with `+=` accumulation. If a case appears where one law's output should be a *named input* to a subsequent law (rather than an additive contribution to a shared output), a `ControllerBuilder` with symbolic references between laws would formalize the wire — keeping the data/function separation at the law level while letting the builder synthesize the `input`/`output` objects and the dispatch order. No use case requires this today.
- **DiffIK gradient through the solve.** Currently blocked by Warp 1.14.0's non-functional `tile_cholesky` adjoint. When upstream fixes it, drop `enable_backward=False` from `_cholesky_solve_kernel` and turn `test_runs_inside_wp_tape_without_crashing` into an analytical-gradient assertion.
