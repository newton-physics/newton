# Newton Controllers — Design Doc

> **Status:** design proposal for v0. Supersedes the prior `README.md` sketch. Implementation does not yet exist.

A library for composable control blocks. A `Controller` is a single, runnable control law (PID, filter, gravity comp, …). A `ControlGroup` is a composer that wraps one or more `Controller`s and orchestrates the per-step zero / compute / accumulate sequence. The API mirrors `newton.actuators.Actuator` so the two modules feel like the same engine — finalize lifecycle, `State` dataclasses, `mask`-based reset, the `state_0, state_1 = state_1, state_0` swap pattern, `is_stateful()` / `is_graphable()` flags, and the `(global_array, indices)` binding convention used throughout Newton.

Controllers typically run **before** actuators in a simulation step: a `Controller` produces a desired joint position, velocity, or force; downstream actuators turn that target into effort. There is no central hardware-contract object — users author their own `@wp.struct` instances for inputs and outputs (or reuse Newton's `State` / `Control`), and controllers bind directly to fields on them via `(array, indices)` pairs.

---

## Core concepts

- **Controller** — Abstract base for a single control law. Subclass with prefix-first naming: `ControllerPID`, `ControllerFilter`, `ControllerGravityComp`. Mirrors `newton.actuators.Controller` in shape: nested `State` dataclass, `finalize(device, num_outputs)`, `state(...)`, `is_stateful()`, `is_graphable()`, `compute(state, next_state, dt)`.
- **ControlGroup** — Composer of one or more `Controller`s. Owns step / reset / state orchestration. Analogous to `newton.actuators.Actuator`, but composes peer control blocks rather than a controller + clamping + delay stack.

Multiple `Controller`s may bind to overlapping output slots; their contributions are scatter-added by the `ControlGroup`. There are no overlap checks.

---

## Two controller flavors

Controllers fall into two categories, distinguished by whether each output depends on its sibling DOFs. The base class is shared; only the constructor signature and the kernel launch differ.

### Independent per-DOF controllers

Output `i` depends only on input `i` plus per-DOF parameters. Examples: `ControllerPID`, low-pass filter, saturation, feedforward.

- Construction needs only the bindings + an `indices` array of global DOF indices.
- Kernel launches 1D with `dim=len(indices)`; `i = wp.tid()`.
- No Newton `Model` required.

### Coupled / structural controllers

Each robot's outputs depend on the full state of that robot. Examples: `ControllerGravityComp`, `ControllerInverseDynamics`, `ControllerDifferentialDrive`, `ControllerHolonomicBase`, `ControllerOperationalSpace`.

- Construction **requires** a `model: newton.Model` representing **one** copy of the robot. The controller reads `dofs_per_robot` (plus joint axes, link masses, parent indices, anything else it needs) from the Model at `__init__` time.
- The `indices` array is the same flat `wp.array[wp.uint32]` shape as the per-DOF case, but validated: `len(indices) % dofs_per_robot == 0`. `num_robots = len(indices) // dofs_per_robot`.
- Convention: `indices[r * dofs_per_robot + j]` is the global DOF index of robot `r`'s local joint `j`. Robots are contiguous in the flat layout.
- Kernel launches 2D with `dim=(num_robots, dofs_per_robot)`; inside, `robot, local_joint = wp.tid()`, and the flat scratch index is `robot * dofs_per_robot + local_joint`.

`ControlGroup` does not need to know which flavor a controller is. Each controller declares its `(output_array, output_indices, scratch)` triple after `finalize()`; the group's zero / accumulate kernels only need those.

---

## Lifecycle

```python
import warp as wp
import newton
import newton.controllers as nc

N = 60                  # 10 robots * 6 DOFs
dof_indices = wp.array(np.arange(N, dtype=np.uint32))

@wp.struct
class ArmInputs:
    joint_q:           wp.array[float]   # global, length >= max(dof_indices) + 1
    joint_qd:          wp.array[float]
    joint_target_pos:  wp.array[float]
    kp:                wp.array[float]   # local, length len(indices)
    integral_max:      wp.array[float]   # local

@wp.struct
class ArmOutputs:
    joint_target_force: wp.array[float]  # global

arm_in  = ArmInputs();  arm_in.joint_q = wp.zeros(N_global, ...);  ...
arm_out = ArmOutputs(); arm_out.joint_target_force = wp.zeros(N_global, ...)

# 1. Construct a controller. The `indices` kwarg is the default for every
#    per-DOF port. Per-port indices (`measurement_indices=...`,
#    `output_indices=...`) override on a case-by-case basis, mirroring
#    Actuator's `pos_indices` / `target_pos_indices` / `effort_indices`.
pid = nc.ControllerPID(
    indices=dof_indices,
    measurement=arm_in.joint_q,                # global; indices default to `indices`
    measurement_rate=arm_in.joint_qd,
    setpoint=arm_in.joint_target_pos,
    kp=arm_in.kp,                              # local array, length len(indices)
    ki=0.1,                                    # scalar
    kd=wp.full(N, 2.0, dtype=wp.float32),      # local array
    integral_max=arm_in.integral_max,
    output=arm_out.joint_target_force,         # global; output_indices default to `indices`
)

# 2. Compose into a group. ControlGroup picks the device from the
#    controllers' bound arrays, validates agreement, and calls
#    finalize() on each.
group = nc.ControlGroup([pid])

# 3. Allocate state pair (Newton-style double buffer).
state_0 = group.state()
state_1 = group.state()

# 4. Step loop.
for _ in range(steps):
    group.step(state_0, state_1, dt=0.005)
    state_0, state_1 = state_1, state_0

# 5. Reset (bool mask, length len(indices), matching actuators).
group.reset(state_0, mask=reset_mask)
```

**Why the two-phase construction.** A `Controller`'s `__init__` validates and stashes bindings; it does not allocate device buffers. `ControlGroup.__init__` picks the device, validates all controllers agree, then calls `controller.finalize(device, num_outputs)` on each — that's when scratch and private buffers are allocated. Same pattern as `Actuator.__init__` calling `controller.finalize(...)`.

---

## Forms a per-DOF port accepts

| Form | Meaning | Kernel access |
|---|---|---|
| `(array, indices)` | global array; look up by index | `array[indices[i]]` |
| `array` (paired with controller-level `indices`) | global array; uses the constructor's default `indices` | `array[indices[i]]` |
| `array` (length `len(indices)`) | already in local layout | `array[i]` |
| scalar (`float` / `int`) | constant for all DOFs | `c` |

Disambiguation between the two bare-`array` forms is by length: if `array.shape[0] == len(indices)`, it's local; otherwise it is treated as global and uses the default `indices`. This mirrors how `ControllerPD` today accepts `kp` as a length-N local array while `positions` arrives as a global state array.

**For the "one big array" case** — measurement, setpoint, and output all living in the same global `state.x`:

```python
pid = nc.ControllerPID(
    indices=output_indices,                              # default for unspecified per-port indices
    measurement=(state.x, measurement_indices),
    setpoint=(state.x, setpoint_indices),
    output=(state.x, output_indices),
    kp=local_kp,
    ki=0.1,
    ...
)
```

Same `wp.array` reference appears in three ports; three different index arrays disambiguate the slots.

---

## State and the double-buffer swap

Every `Controller` defines a nested `State` dataclass holding whatever internal buffers the control law needs (e.g. PID integrals, filter history). `Controller.is_stateful()` returns true if it holds any. `Controller.state(num_outputs, device)` allocates a fresh `State`.

`ControlGroup.State` composes per-controller states by index. `group.state()` returns a composed state with every controller's state already allocated.

```python
state_0 = group.state()
state_1 = group.state()
for _ in range(steps):
    group.step(state_0, state_1, dt=dt)
    state_0, state_1 = state_1, state_0
```

The group reads from `state_0` and writes to `state_1` on each step. After step, the caller swaps. Reset uses a bool mask, identical to `Actuator.State.reset`:

```python
group.reset(state_0, mask: wp.array[wp.bool] | None = None)
```

`mask` is per-controller (length `len(indices)`). `mask=None` resets every entry.

---

## Output accumulation (scatter-add, no overlap checks)

Each `Controller` allocates a private `wp.array[float]` scratch of length `len(indices)` during `finalize`. `ControlGroup.step` then, for each controller:

1. **Zero** the controller's destination slots: `output_array[output_indices[i]] = 0.0` for `i` in `0..len(output_indices)`.
2. **Compute** — call `controller.compute(state, next_state, dt)`, which writes its scratch and next state.
3. **Accumulate** — `output_array[output_indices[i]] += scratch[i]` for `i` in `0..len(output_indices)`.

Zeroing is idempotent — two controllers writing to overlapping `(array, indices)` slots zero the overlap twice, then both accumulate, producing the sum. This is the scatter-add semantics that lets a user compose `ControllerPID + ControllerGravityComp + ControllerFeedforward` all writing to `joint_target_force` without conflict. There are no overlap checks; users compose at their own risk.

---

## Authoring a Controller — the class shape

`Controller` is a base class deliberately shaped like `newton.actuators.Controller`:

```python
class Controller:
    @dataclass
    class State:
        def reset(self, mask: wp.array[wp.bool] | None = None) -> None: ...

    def __init__(self, **ports):
        """Validate shapes / dtypes / scalar value ranges, normalize each
        per-DOF port to (array, indices) form, stash bindings on self.
        Does NOT allocate device buffers — finalize() does that.

        Subclasses declare which kwargs they accept; missing required ports
        raise here, unknown ports raise here, and shape / dtype / length
        mismatches raise here."""

    def finalize(self, device: wp.Device, num_outputs: int) -> None:
        """Allocate device-side scratch and per-controller buffers.
        Called by ControlGroup after construction. num_outputs == len(indices)."""

    def state(self, num_outputs: int, device: wp.Device) -> Controller.State | None:
        """Allocate a fresh State, or None if stateless."""

    def is_stateful(self) -> bool: ...
    def is_graphable(self) -> bool: ...

    def compute(
        self,
        state: Controller.State | None,
        next_state: Controller.State | None,
        dt: float,
    ) -> None:
        """Read bound inputs, write to self._scratch, write next_state.
        Called by ControlGroup.step. The device is fixed at finalize()
        time, so compute() does not take one."""
```

The four lifecycle method names — `finalize`, `state`, `is_stateful`, `is_graphable` — line up with `newton.actuators.Controller`. The `update_state` step is folded into `compute(state, next_state, dt)`, matching `Actuator`'s control-flow shape. `resolve_arguments` from the actuator base class is dropped for v0: scalar validation moves into `__init__` directly.

The base class is neutral about the per-DOF vs. coupled distinction. Subclasses decide whether to take a `model=` kwarg, what their `indices`-length divisibility constraint is, and what kernel launch dimensionality to use.

### Example: ControllerPID (independent flavor)

```python
class ControllerPID(Controller):
    """Stateful PID controller producing a target signal from a measurement
    and setpoint. Independent per-DOF: output[i] depends only on input[i].

    Ports:
        indices                                       — default global DOF indices
        measurement, measurement_rate, setpoint       — signals
        kp, ki, kd, integral_max                      — gains (local arrays or scalars)
        output                                        — destination
        *_indices (optional, per port)                — override default `indices`
    """

    @dataclass
    class State(Controller.State):
        integral: wp.array[float] | None = None

        def reset(self, mask: wp.array[wp.bool] | None = None) -> None:
            if mask is None:
                self.integral.zero_()
            else:
                wp.launch(_masked_zero_1d, dim=len(mask), inputs=[self.integral, mask])

    def __init__(
        self,
        *,
        indices,
        measurement, measurement_rate, setpoint,
        kp, ki, kd, integral_max,
        output,
        measurement_indices=None,
        measurement_rate_indices=None,
        setpoint_indices=None,
        output_indices=None,
    ):
        self._indices = indices
        # Per-port indices default to the controller-level `indices`.
        self._measurement, self._measurement_indices = _normalize_port(measurement, measurement_indices, indices)
        self._measurement_rate, self._measurement_rate_indices = _normalize_port(measurement_rate, measurement_rate_indices, indices)
        self._setpoint, self._setpoint_indices = _normalize_port(setpoint, setpoint_indices, indices)
        self._output, self._output_indices = _normalize_port(output, output_indices, indices)
        # Gains: local array of length len(indices), or scalar.
        self._kp = _normalize_gain(kp, len(indices))
        self._ki = _normalize_gain(ki, len(indices))
        self._kd = _normalize_gain(kd, len(indices))
        self._integral_max = _normalize_gain(integral_max, len(indices))
        # Scalar value-range validation (e.g. ki >= 0) happens here.

    def finalize(self, device, num_outputs):
        self._scratch = wp.zeros(num_outputs, dtype=wp.float32, device=device)

    def is_stateful(self): return True
    def is_graphable(self): return True

    def state(self, num_outputs, device):
        return ControllerPID.State(integral=wp.zeros(num_outputs, dtype=wp.float32, device=device))

    def compute(self, state, next_state, dt):
        wp.launch(
            _pid_kernel,
            dim=len(self._indices),
            inputs=[
                self._measurement, self._measurement_indices,
                self._measurement_rate, self._measurement_rate_indices,
                self._setpoint, self._setpoint_indices,
                self._kp, self._ki, self._kd, self._integral_max,
                dt, state.integral,
            ],
            outputs=[self._scratch, next_state.integral],
        )
```

The kernel does `measurement[measurement_indices[i]] - setpoint[setpoint_indices[i]]` and `kp[i] * error + …`, writing `scratch[i]`.

### Sketch: ControllerGravityComp (coupled flavor)

```python
class ControllerGravityComp(Controller):
    """Gravity compensation for an articulated robot. Coupled per-robot:
    a robot's per-joint torque depends on its full configuration q."""

    def __init__(
        self,
        *,
        model: newton.Model,            # single-robot Model
        indices,                        # flat; len(indices) % dofs_per_robot == 0
        measurement,                    # global joint_q
        output,                         # global joint torque
        gravity=(0.0, 0.0, -9.81),      # vec3 scalar
        measurement_indices=None,
        output_indices=None,
    ):
        self._dofs_per_robot = _dofs_in_single_robot_model(model)
        if len(indices) % self._dofs_per_robot != 0:
            raise ValueError(
                f"ControllerGravityComp: len(indices)={len(indices)} is not a multiple "
                f"of dofs_per_robot={self._dofs_per_robot} (from the supplied Model)."
            )
        self._num_robots = len(indices) // self._dofs_per_robot
        # Extract link masses, COMs, joint axes, parent indices, etc.
        # from the single-robot Model. These live as device arrays on self.
        self._link_masses = ...
        self._link_coms = ...
        self._joint_axes = ...
        self._parent_indices = ...
        # Stash bindings.
        ...

    def compute(self, state, next_state, dt):
        wp.launch(
            _gravity_comp_kernel,
            dim=(self._num_robots, self._dofs_per_robot),
            inputs=[
                self._measurement, self._measurement_indices,
                self._link_masses, self._link_coms, self._joint_axes, self._parent_indices,
                self._gravity,
            ],
            outputs=[self._scratch],
        )
```

The kernel uses `robot, local_joint = wp.tid()`, `flat = robot * dofs_per_robot + local_joint`, and writes `scratch[flat]`. `ControlGroup` then scatter-adds via `output_indices` exactly as for `ControllerPID` — the group does not see the 2D launch.

---

## ControlGroup

```python
class ControlGroup:
    def __init__(self, controllers: list[Controller]):
        """Pick the device from controllers' bound output arrays, validate
        all agree, finalize() every controller (each controller's
        num_outputs is len(its indices)), precompute the per-output
        zero / accumulate plan."""

    def is_stateful(self) -> bool: ...
    def is_graphable(self) -> bool: ...

    def state(self) -> ControlGroup.State | None:
        """Allocate composed state with one entry per stateful controller."""

    def step(self, current_state, next_state, dt: float) -> None:
        """For each controller: zero output slots, compute, scatter-add scratch."""

    def reset(self, state, mask: wp.array[wp.bool] | None = None) -> None:
        """Fan the mask out to each controller's State.reset(mask)."""
```

```python
@dataclass
class ControlGroup.State:
    controller_states: list[Controller.State | None]
    def reset(self, mask=None): ...
```

---

## Where this fits in the simulation step

```python
group = nc.ControlGroup([pid, gravity_comp])
actuator = newton.actuators.Actuator(controller=..., ...)

state_0 = group.state(); state_1 = group.state()
act_state_0 = actuator.state(); act_state_1 = actuator.state()

for _ in range(steps):
    # 1. Controllers run first. Their outputs typically land in arrays the
    #    actuator reads from (e.g. joint_target_force).
    group.step(state_0, state_1, dt=dt)

    # 2. Actuator translates target → joint effort.
    actuator.step(sim_state, sim_control, act_state_0, act_state_1, dt=dt)

    # 3. Physics solver.
    solver.step(model, sim_state, sim_state_next, dt)

    state_0, state_1 = state_1, state_0
    act_state_0, act_state_1 = act_state_1, act_state_0
```

The user's output `@wp.struct` fields can be the *same* `wp.array`s set on `sim_control` (e.g. `joint_target_pos`, `joint_f`). That sharing is up to the user — the controllers module does not introspect Newton sim objects.

---

## Public API surface

Per AGENTS.md, examples and docs must not import from `newton._src`. The internal package is `newton/_src/controllers/`; the public shim is `newton/controllers.py`:

```python
# newton/controllers.py
from ._src.controllers import (
    Controller,
    ControlGroup,
    ControllerPID,
)

__all__ = [
    "ControlGroup",
    "Controller",
    "ControllerPID",
]
```

Users write `from newton.controllers import ControlGroup, ControllerPID`. Same pattern as `newton/actuators.py`.

---

## v0 scope

**Framework.**

- `Controller` base class with `finalize`, `state`, `is_stateful`, `is_graphable`, `compute`.
- `ControlGroup` composer with `state`, `step`, `reset`, `is_stateful`, `is_graphable`, plus the zero / scatter-add machinery via `(output_array, output_indices)`.
- `newton/controllers.py` public shim.
- Helpers: `_normalize_port`, `_normalize_gain` for the port-form normalization described above.
- For each shipped controller: math-correctness tests, integral / state accumulation across the `state_0` / `state_1` swap, masked reset, scatter-add when multiple controllers overlap, sliced / one-big-array binding sanity tests.

**Suggested implementation order** (each step exercises a new piece of the framework):

1. `ControllerPID` — proves independent-per-DOF flavor + the framework end-to-end.
2. `ControllerGravityComp` — first coupled controller; proves the single-robot `Model` + `dofs_per_robot` divisibility convention + 2D kernel launch.
3. `ControllerJointImpedance` — coupled but very close to gravity comp in shape; tests parameter porting.
4. `ControllerDifferentialDrive` — first mobile controller; proves the body-twist-to-wheels mapping (output not the same shape as input).
5. `ControllerHolonomic` — second mobile; same family.
6. `ControllerDifferentialIK` — first task-space controller; proves Jacobian reuse from `newton.ik`.
7. `ControllerOperationalSpace` — most complex; needs full dynamics from the Model.

See *Controllers to design* below for per-controller notes.

**Out of scope for v0.**

- USD parsing.
- Differentiability flag (`requires_grad`) — slot reserved in `finalize` signature, not exercised.
- CUDA-graph capture testing.
- `ModelBuilder.add_controller` analog.

---

## Controllers to design

> Notes for a follow-up design session. Each controller below needs a concrete spec before implementation. The framework (Controller, ControlGroup, indices binding, coupled-controller convention) is already settled; what's left is per-controller decisions about ports, kernel layout, and what to lift out of `newton.Model`.

### Model-based controllers

#### `ControllerDifferentialIK`

- **Category:** coupled, task-space.
- **Sketch:** maps a desired task-space velocity `ẋ_d` (per end-effector) to joint velocities via `q̇ = J⁺(q) ẋ_d` (damped pseudoinverse, with nullspace and joint-limit terms).
- **Relationship to `newton.ik`.** Should *not* be built on `IKSolver` — that's a full nonlinear multi-seed LM / L-BFGS optimizer (`newton/_src/sim/ik/ik_solver.py`). Differential IK is a one-step Jacobian solve. What it *should* reuse from `newton.ik`:
  - The Jacobian machinery (`IKJacobianType.AUTODIFF` or analytical) used inside `IKOptimizerLM`. Need to expose / refactor that into a reusable Jacobian-at-q routine that doesn't require an optimizer loop.
  - The `IKObjective` family (`IKObjectivePosition`, `IKObjectiveRotation`) as the way to declare "what task-space quantity does ẋ_d refer to".
- **Open design questions to resume on:**
  - Does the controller take a list of `IKObjective`s the same way `IKSolver` does, or a flatter "frame_a, frame_b, type" spec?
  - Damping λ — scalar, per-DOF, or adaptive?
  - Nullspace projection (e.g. joint centering) — port or built-in?
  - Output: joint velocity (`joint_qd_target`) or joint position by integrating ẋ_d?

#### `ControllerJointImpedance`

- **Category:** coupled.
- **Sketch:** `τ = M(q)(q̈_d + K_d (q̇_d - q̇) + K_p (q_d - q)) + h(q, q̇)` (or simpler stiffness-only variants).
- **Open design questions:**
  - Include mass-matrix premultiplication, or stop at `τ = K_p e + K_d ė + g(q)` (PD + gravity)?
  - Variants to ship: full impedance, stiffness-only (PD + gravity), Cartesian impedance (later)?
  - Where does `M(q)` come from in Newton? Composite Rigid Body? RNEA-derived? Check what dynamics primitives `Model` exposes.

#### `ControllerGravityComp`

- **Category:** coupled.
- **Sketch:** `τ_g = -∂U_gravity/∂q`. Equivalent to RNEA with `q̇ = q̈ = 0`.
- **Open design questions:**
  - Recursive Newton-Euler or analytic per-joint? RNEA is more general (handles all joint types Newton supports); analytic-per-joint is simpler for revolute-only chains.
  - What does the Model need to expose to make this clean? Probably: per-link mass, COM in body frame, joint type, joint axis, parent index, joint offset transform.
  - Does Newton already have an RNEA kernel we can reuse, or do we author one here?

### Manipulator controllers

#### `ControllerOperationalSpace`

- **Category:** coupled, task-space.
- **Sketch:** task-space inertia `Λ = (J M⁻¹ Jᵀ)⁻¹`; task force `F = Λ (ẍ_d + K_d (ẋ_d - ẋ) + K_p (x_d - x)) + μ + p`; torque `τ = Jᵀ F + (I - JᵀJ̄ᵀ) τ_null`, where `J̄ = M⁻¹ Jᵀ Λ`.
- **Open design questions:**
  - All-in-one OSC or split (TaskSpaceForce + JointMapping)?
  - Inertia-weighted pseudoinverse vs. plain damped pseudoinverse — both, or pick one?
  - Same Jacobian-machinery reuse question as `ControllerDifferentialIK`.
  - Nullspace handling — separate `ControllerNullspaceProjection` block that scatter-adds, or built in?

### Mobile-robot controllers

These break the per-DOF / per-robot symmetry: the controller's input dimension (body-twist command, 3 numbers for 2D planar) is different from its output dimension (wheel count). The "coupled controller" convention still applies — a single-robot `Model` describes one mobile base — but the indices length divides by *wheel count*, not by DOF count.

#### `ControllerDifferentialDrive`

- **Category:** coupled, mobile.
- **Sketch:** input `(v, ω)` (linear m/s, angular rad/s) → wheel velocities `(ω_L, ω_R)` via `ω_L = (v - ω·L/2) / r`, `ω_R = (v + ω·L/2) / r` where `r` = wheel radius, `L` = axle width.
- **Open design questions:**
  - Where does Newton store wheel-radius and axle-width? Likely needs new metadata on the Model, or a small auxiliary struct passed in.
  - Output port: wheel velocity (`joint_target_vel`) or wheel torque via an inner PI loop?
  - Are we taking a `Model` of a diff-drive base, or just the two scalars `(r, L)` directly? Suggest Model for consistency with the coupled-controller convention.
  - Does Newton have a canonical way to identify "this is the left wheel joint" vs "right wheel joint" in a Model? Probably via joint names or user-supplied indices.

#### `ControllerHolonomic`

- **Category:** coupled, mobile.
- **Sketch:** input body twist `(v_x, v_y, ω)` → wheel velocities for an N-wheeled omni base (3-wheel Kiwi, 4-wheel mecanum, etc.). Mapping is `wheel_velocities = W(geometry) · [v_x, v_y, ω]ᵀ`.
- **Open design questions:**
  - Which geometries to ship (Kiwi / mecanum / generic)? Suggest one generic `wheel_geometry: wp.array2d[float]` representing the per-wheel placement + roller angle, with helpers to build the matrix for known configurations.
  - Same Model-vs-direct-scalars question as `ControllerDifferentialDrive`.

### Cross-cutting questions to revisit

- **Scalar-vs-array gain ports.** Still unresolved (see Open questions below). Affects every controller with gains.
- **What lives in `newton.Model` vs. controllers' own metadata?** Coupled controllers need: link masses / COMs / joint axes / parent indices / joint types / wheel geometry / etc. Decide whether `Model` grows accessors or whether each controller introspects fields directly.
- **Where do Jacobian / dynamics primitives (M, J, h, g) live?** Today they're internal to `newton/_src/sim/ik/`. Several controllers need them in non-IK contexts. Worth pulling into a shared `newton.dynamics` (or similar) before authoring `ControllerOperationalSpace`.
- **Mobile-robot output shape.** Output port length ≠ DOF count. The `(output_array, output_indices)` pattern still works (output indices select wheel-velocity slots in a global control array), but the divisibility convention is `len(indices) % wheels_per_robot == 0` rather than DOFs. Document this carefully.

---

## Open questions

1. **Scalar vs. array kernel variants.** When a per-DOF gain port (`kp`, `kd`) can be either a scalar `float` or a length-`len(indices)` `wp.array[float]`, the controller author needs a strategy: ship two kernel variants, dispatch on type at `__init__`, broadcast scalars to a length-1 array, or rely on Warp's overload mechanism. Pick a convention before the second controller lands.
2. **`Model` introspection API for coupled controllers.** v0 doesn't ship one, but the v1 design needs `_dofs_in_single_robot_model(model)`, plus accessors for link masses, COMs, joint axes, parent indices. Decide whether these live in `newton._src.controllers.utils`, or if `newton.Model` itself should expose convenience accessors usable by both controllers and any user code.
3. **Composed reset shape.** `group.reset(state, mask)` is per-DOF on the output dimension (length `len(indices)`). If a controller's state has a different leading dim (e.g. history buffers, per-robot quantities for coupled controllers), it handles the mask internally — same as `ControllerNeuralMLP.State.reset` does today.
4. **Future need for sliced-output provenance.** Dropped in v0 because zeroing-overlapping-indices-then-accumulating is correct (just slightly redundant). If a future user has many controllers binding overlapping `output_indices` and the redundant zeroing becomes a measurable cost, dedupe the zero pass in `ControlGroup` by `(array.ptr, indices.ptr)`.
