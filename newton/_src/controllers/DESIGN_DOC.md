# Newton Controllers — Design Doc

> **Status:** design proposal for v0. Supersedes the prior `README.md` sketch. Implementation does not yet exist.

A library for composable control blocks. A `Controller` is a single, runnable control law (PID, filter, gravity comp, …). A `ControlGroup` is a composer that wraps one or more `Controller`s and orchestrates the per-step zero / compute / accumulate sequence. The API mirrors `newton.actuators.Actuator` so the two modules feel like the same engine — finalize lifecycle, `State` dataclasses, `mask`-based reset, the `state_0, state_1 = state_1, state_0` swap pattern, `is_stateful()` / `is_graphable()` flags.

Controllers typically run **before** actuators in a simulation step: a `Controller` produces a desired joint position, velocity, or force; downstream actuators turn that target into effort. There is no central hardware-contract object — users author their own `@wp.struct` instances for inputs and outputs, and controllers bind directly to fields on them.

---

## Core concepts

- **Controller** — Abstract base for a single control law. Subclass with prefix-first naming: `ControllerPID`, `ControllerFilter`, `ControllerGravityComp`. Mirrors `newton.actuators.Controller` in shape: nested `State` dataclass, `finalize(device, num_outputs)`, `state(...)`, `is_stateful()`, `is_graphable()`, `compute(state, next_state, dt)`.
- **ControlGroup** — Composer of one or more `Controller`s. Owns the step / reset / state orchestration. Analogous to `newton.actuators.Actuator`, but composes peer control blocks rather than a controller + clamping + delay stack.

Multiple `Controller`s may bind to the same output field; their contributions are scatter-added by the `ControlGroup` (see *Output accumulation* below). There are no disjoint-write checks.

---

## Lifecycle

```python
import warp as wp
import newton.controllers as nc

@wp.struct
class ArmInputs:
    joint_q:           wp.array[float]
    joint_qd:          wp.array[float]
    joint_target_pos:  wp.array[float]
    kp:                wp.array[float]
    integral_max:      wp.array[float]

@wp.struct
class ArmOutputs:
    joint_target_force: wp.array[float]   # consumed by a downstream Actuator

# 1. Allocate the user-owned hardware contract.
arm_in = ArmInputs()
arm_in.joint_q = wp.zeros(N, dtype=wp.float32)
arm_in.joint_qd = wp.zeros(N, dtype=wp.float32)
arm_in.joint_target_pos = wp.zeros(N, dtype=wp.float32)
arm_in.kp = wp.full(N, 50.0, dtype=wp.float32)
arm_in.integral_max = wp.full(N, 10.0, dtype=wp.float32)

arm_out = ArmOutputs()
arm_out.joint_target_force = wp.zeros(N, dtype=wp.float32)

# 2. Construct a controller. Every port (signal *and* gain) is bound here,
#    uniformly. Each port may be a wp.array (sourced from the hardware
#    contract or anywhere else) or a scalar constant. The controller
#    author does not decide on the user's behalf which form a user must use.
pid = nc.ControllerPID(
    measurement=arm_in.joint_q,
    measurement_rate=arm_in.joint_qd,
    setpoint=arm_in.joint_target_pos,
    kp=arm_in.kp,                                    # tunable via the contract
    ki=0.1,                                          # scalar constant
    kd=wp.full(N, 2.0, dtype=wp.float32),            # user-owned external array
    integral_max=arm_in.integral_max,
    output=arm_out.joint_target_force,
)

# 3. Compose into a group. ControlGroup calls finalize() on every controller.
group = nc.ControlGroup([pid])

# 4. Allocate state pair (Newton-style double buffer).
state_0 = group.state()
state_1 = group.state()

# 5. Step loop.
for _ in range(steps):
    group.step(state_0, state_1, dt=0.005)
    state_0, state_1 = state_1, state_0

# 6. Reset (bool mask matching actuators).
group.reset(state_0, mask=reset_mask)
```

**Why the two-phase construction.** A `Controller`'s `__init__` records the bindings, runs shape / dtype / value-range checks, and stashes the resolved values. It does **not** allocate per-controller scratch or state buffers, because `ControlGroup` is the authority on device-level allocation: it picks the device (inferred from the controllers' arrays), validates they all agree, and *then* calls `controller.finalize(device, num_outputs)` on each. This matches the actuator pattern verbatim — `Actuator.__init__` calls `controller.finalize(device, num_actuators)` on its components.

---

## Two ways to pass a port

Every named port on a `Controller`'s constructor accepts either of the following, and the controller author does not get to dictate which form a user must use:

1. **`wp.array`** — any warp array of the right shape and dtype. May be a field on the user's input `@wp.struct` (tunable between steps), a fully external user-owned array, or a basic-sliced view (`arm_out.joint_target_force[0:3]`). Passed through verbatim.
2. **Scalar constant** — a plain `float` or `int`. Baked in at construction. The user can later replace the entire controller if they want a different value.

A given port is *just* a `wp.array[...]` or scalar from the kernel's perspective. The same `ControllerPID` instance accepts `kp` as a hardware-contract field from one user, a constant from another, and an external array from a third — no special-casing by the author.

Sliced views work uniformly. `arm_out.joint_target_force[0:3]` is a `wp.array` view; element-wise reads and writes hit the right offset of the underlying buffer. This applies equally to input and output bindings.

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

The group reads from `state_0` and writes to `state_1` on each step. After step, the caller swaps. This matches the Newton convention used elsewhere in the simulator.

Reset uses a bool mask, identical to `Actuator.State.reset`:

```python
group.reset(state_0, mask: wp.array[wp.bool] | None = None)
```

`mask=None` resets every output.

---

## Output accumulation (scatter-add, no overlap checks)

Each `Controller` writes its result into a private `wp.array[float]` scratch buffer of length `num_outputs`, allocated during `finalize`. `ControlGroup.step` then:

1. For each controller, zero its bound output (element-wise on the array or view).
2. For each controller, compute the controller (writing its scratch and next state).
3. For each controller, element-wise-add its scratch into the bound output.

Zeroing is idempotent — two controllers binding to the same output field cause two zero launches, but the second is a no-op effect-wise. Two controllers binding to overlapping slices of the same field similarly zero the overlap twice and then both accumulate into the shared region, producing the sum. This is the scatter-add semantics that lets a user compose, e.g., `ControllerPID + ControllerGravityComp + ControllerFeedforward` all writing to `joint_target_force` without conflict. There are no overlap checks; users compose at their own risk.

---

## Authoring a Controller — the class shape

`Controller` is a base class deliberately shaped like `newton.actuators.Controller`:

```python
class Controller:
    @dataclass
    class State:
        def reset(self, mask: wp.array[wp.bool] | None = None) -> None: ...

    def __init__(self, **ports):
        """Validate shapes / dtypes / scalar value ranges, stash bindings
        on self. Does NOT allocate device buffers — finalize() does that.

        Subclasses declare which kwargs they accept; missing required ports
        raise here, unknown ports raise here, and shape / dtype mismatches
        across ports raise here."""

    def finalize(self, device: wp.Device, num_outputs: int) -> None:
        """Allocate device-side scratch and per-controller buffers.
        Called by ControlGroup after construction."""

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

The four lifecycle method names — `finalize`, `state`, `is_stateful`, `is_graphable` — line up with `newton.actuators.Controller`. The `update_state` step is folded into `compute(state, next_state, dt)`, matching `Actuator`'s control-flow shape. `resolve_arguments` from the actuator base class is intentionally dropped for v0: scalar validation moves into `__init__` directly, since there is no longer a separation between "scalar config" and "array bindings" — every port is uniform.

### Example: ControllerPID

```python
class ControllerPID(Controller):
    """Stateful PID controller producing a target signal from a measurement
    and setpoint.

    Ports (each accepts a wp.array or a scalar):
        measurement, measurement_rate, setpoint   — signals
        kp, ki, kd, integral_max                  — gains
        output                                    — wp.array (or sliced view)
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
        measurement,
        measurement_rate,
        setpoint,
        kp,
        ki,
        kd,
        integral_max,
        output,
    ):
        # Each port is already a wp.array or a scalar; just stash + validate.
        # Cross-port shape compatibility, scalar value ranges (e.g. ki >= 0),
        # and dtype checks happen here.
        self._measurement = measurement
        self._measurement_rate = measurement_rate
        self._setpoint = setpoint
        self._kp = kp
        self._ki = ki
        self._kd = kd
        self._integral_max = integral_max
        self._output = output  # wp.array or sliced view; ControlGroup zeros/accumulates

    def finalize(self, device, num_outputs):
        self._scratch = wp.zeros(num_outputs, dtype=wp.float32, device=device)

    def is_stateful(self): return True
    def is_graphable(self): return True

    def state(self, num_outputs, device):
        return ControllerPID.State(integral=wp.zeros(num_outputs, dtype=wp.float32, device=device))

    def compute(self, state, next_state, dt):
        # Kernel signature is uniform whether kp / ki / kd / integral_max
        # are arrays or scalars — Warp lowers them appropriately.
        wp.launch(
            _pid_kernel,
            dim=self._scratch.shape[0],
            inputs=[
                self._measurement, self._measurement_rate, self._setpoint,
                self._kp, self._ki, self._kd, self._integral_max,
                dt, state.integral,
            ],
            outputs=[self._scratch, next_state.integral],
        )
```

The per-controller output scratch (`self._scratch`) is the buffer `ControlGroup` accumulates into `self._output`. The controller is otherwise self-contained — anything the kernel needs is on `self`, and nothing about group composition leaks into the kernel signature. `ControllerPID` may need to ship two kernel variants (one with scalar gains, one with array gains) or use Warp's polymorphism so the same author code accepts either form for `kp` / `ki` / `kd`; that is a v0 implementation detail, not an API one.

---

## ControlGroup

```python
class ControlGroup:
    def __init__(self, controllers: list[Controller]):
        """Pick the device from controllers' bound outputs, validate
        all agree, finalize() every controller, precompute the per-output
        zero / accumulate plan."""

    def is_stateful(self) -> bool:
        """True if any controller is stateful."""

    def is_graphable(self) -> bool:
        """True if every controller is graphable."""

    def state(self) -> ControlGroup.State | None:
        """Allocate composed state with one entry per stateful controller."""

    def step(self, current_state, next_state, dt: float) -> None:
        """For each controller: zero the bound output; compute; accumulate scratch into output."""

    def reset(self, state, mask: wp.array[wp.bool] | None = None) -> None:
        """Reset every controller's per-DOF state where mask is true."""
```

`ControlGroup.State` is the composed analogue of `Actuator.State`:

```python
@dataclass
class ControlGroup.State:
    controller_states: list[Controller.State | None]
    def reset(self, mask=None): ...
```

---

## Where this fits in the simulation step

```python
group = nc.ControlGroup([...])
actuator = newton.actuators.Actuator(controller=..., ...)

state_0 = group.state()
state_1 = group.state()
act_state_0 = actuator.state()
act_state_1 = actuator.state()

for _ in range(steps):
    # 1. Controllers run first. They typically write a target into the
    #    user's output @wp.struct, which the user may have wired to the
    #    arrays the actuator reads from.
    group.step(state_0, state_1, dt=dt)

    # 2. Actuator translates target → joint effort.
    actuator.step(sim_state, sim_control, act_state_0, act_state_1, dt=dt)

    # 3. Physics solver.
    solver.step(model, sim_state, sim_state_next, dt)

    state_0, state_1 = state_1, state_0
    act_state_0, act_state_1 = act_state_1, act_state_0
```

The user's output `@wp.struct` fields can be the *same* `wp.array`s set on `sim_control` (e.g. `joint_target_pos`). That sharing is up to the user — the controllers module does not introspect Newton sim objects.

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

**In scope.**

- `Controller` base class with `finalize`, `state`, `is_stateful`, `is_graphable`, `compute`.
- `ControlGroup` composer with `state`, `step`, `reset`, `is_stateful`, `is_graphable`.
- `ControllerPID` — first and only concrete controller in v0.
- `newton/controllers.py` public shim.
- Unit tests for `ControllerPID` (stateless behavior, integral accumulation, masked reset, swap pattern, scatter-add when two PIDs write to the same field, sliced output binding).

**Out of scope for v0.**

- USD parsing (no `controllers/usd_parser.py`).
- Differentiability flag (`requires_grad`) — slot reserved in `finalize` signature, not exercised.
- CUDA graph capture testing — `is_graphable()` returns the truthful value but we do not ship a captured-graph example.
- Additional controllers (filter, gravity comp, diff-IK, neural). These follow once the base shape is settled.
- Connecting to `ModelBuilder` (e.g. an `add_controller` analog of `add_actuator`).

---

## Open questions

1. **Scalar vs. array kernel variants.** When a port like `kp` can be either a scalar `float` or a `wp.array[float]`, the controller author needs a strategy: ship two kernel variants, dispatch on type at `__init__`, broadcast scalars to a length-1 array, or rely on Warp's overload mechanism. Pick a convention and document it before authoring the second controller.
2. **Composed reset shape.** `group.reset(state, mask)` is per-DOF on the output dimension. If a controller's state has a different leading dim (e.g. history buffers), it handles the mask internally — same as `ControllerNeuralMLP.State.reset` does today.
3. **Future need for sliced-output provenance.** Dropped in v0 because zeroing-overlapping-slices-then-accumulating is correct (just slightly redundant). If a future user has many controllers binding to slices of one field and the redundant zeroing becomes a measurable cost, reintroduce a narrow `Output` wrapper that carries `(parent_array, slice)` so `ControlGroup` can dedup the zero pass.
