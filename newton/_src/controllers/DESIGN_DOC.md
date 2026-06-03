# Newton Controllers — Design Doc

A library for composable control blocks. A `Controller` is a single, runnable control law (PID, differential IK, gravity comp, …). A `ControlGroup` is a composer that wraps one or more `Controller`s and orchestrates the per-step zero / compute sequence.

Controllers typically run **before** actuators in a simulation step: a `Controller` produces a desired joint position, velocity, or force; downstream actuators turn that target into effort. Users author their own `@wp.struct` instances for inputs and outputs (which can be views/slices of the `newton.State` or `newton.Control` objects).

---

## Core concepts

- **Controller** — Abstract base for a single control law. Subclass with prefix-first naming: `ControllerPID`, `ControllerFilter`, `ControllerGravityComp`. Lifecycle methods: `finalize(device, num_outputs)`, `state(...)`, `is_stateful()`, `is_graphable()`, `outputs()`, `compute(state, next_state, dt)`. Holds a nested `State` dataclass.
- **ControlGroup** — Composer of one or more `Controller`s. Owns step / reset / state orchestration.

Multiple `Controller`s may bind to overlapping output slots; their contributions accumulate via `+=` directly into the output array (see *Output accumulation*).

---

## Two controller flavors

Controllers fall into two categories, distinguished by whether each output depends on its sibling DOFs. The base class is shared, but the underlying kernel launch may be a different shape.

### Independent per-DOF controllers

Output `i` depends only on input `i` plus per-DOF parameters. Examples: `ControllerPID`, low-pass filter, saturation, feedforward.

- Construction needs only the bindings + an `indices` array of global DOF indices.
- Kernel launches 1D with `dim=len(indices)`; `i = wp.tid()`.
- No Newton `Model` required.

### Coupled / structural controllers

Each robot's outputs depend on the full state of that robot. Examples: `ControllerDifferentialIK`, `ControllerGravityComp`, `ControllerDifferentialDrive`, `ControllerHolonomic`, `ControllerOperationalSpace`.

- Articulated controllers (`ControllerDifferentialIK`, `ControllerGravityComp`, etc.) take a `model: newton.Model` containing one or more **topologically identical** articulations (`K = model.articulation_count`). The K articulations must share DOF count, link/joint count, and joint types; they may differ in physical parameters (mass, inertia, friction, joint limits). The controller reads `dofs_per_robot = model.joint_dof_count // K` directly from the Model at `__init__` time.
- Mobile-base controllers (`ControllerDifferentialDrive`, `ControllerHolonomic`) take raw geometry instead of a Model. Wheel-base geometry isn't `Model` state in Newton.
- For articulated controllers: `len(indices) % model.joint_dof_count == 0`. The replication count is `R = len(indices) // model.joint_dof_count`, and the controller internally tiles the user-supplied Model `R` times at `finalize()`. `num_robots = K * R`. The K=1 case is the single-robot template; K>1 lets the user supply categorical variants (e.g. for RL randomization) which are replicated across the batch.
- For mobile controllers: `len(indices) % wheels_per_robot == 0`. `num_robots = len(indices) // wheels_per_robot`.
- Layout convention. After replication, robot `r` is variant `r % K` from replication `r // K`. So with `K=4, R=3` the robots are `[v0, v1, v2, v3, v0, v1, v2, v3, v0, v1, v2, v3]`. Per-group input arrays (`target_pos`, `target_quat`, `damping`, …) of length `num_robots` follow this layout.
- For both flavors: `indices[r * outputs_per_robot + j]` is the global output index of robot `r`'s local slot `j` (where `outputs_per_robot = dofs_per_robot` for articulated or `wheels_per_robot` for mobile). Robots are contiguous in the flat layout.
- Kernel launches 2D with `dim=(num_robots, outputs_per_robot)`; inside, `robot, local_slot = wp.tid()`, and the flat output index is `robot * outputs_per_robot + local_slot`.

`ControlGroup` does not need to know which flavor a controller is. Each controller declares its bound output arrays + indices via `outputs()`; the group's zero pass walks the union of all controllers' output destinations.

---

## The unified port form

Every input/output port that addresses per-DOF data accepts one of two forms:

| Form | Meaning | Kernel access |
|---|---|---|
| `array` (bare) | use the controller-level `indices` as the lookup | `array[indices[i]]` |
| `(array, port_indices)` | tuple; use `port_indices` as the lookup | `array[port_indices[i]]` |

For many controllers, the indices of all inputs will align, and the bare `wp.array` input is enough. The second option exists for when more flexibility is needed.

### Validation at `__init__`

| Form | Check |
|---|---|
| bare `array` | `array.shape[0] >= max(indices) + 1` |
| `(array, port_indices)` | `len(port_indices) == len(indices)` and `array.shape[0] >= max(port_indices) + 1` |

The user can store their data wherever is most natural — locally allocated for one controller, globally shared across the sim, sliced into a bigger struct, etc.

### Per-group ports

Some controllers have ports keyed by *robot index*, not by per-DOF index — e.g. `ControllerDifferentialIK`'s `target_pos` (one 3D target per robot). These are a separate, simpler shape:

- Bare `wp.array[D]` (with appropriate dtype `D`) of length `num_robots`.
- Kernel does `target_pos[robot]` where `robot = wp.tid()[0]` (in a 2D launch) or `robot = i // dofs_per_robot` (in a 1D launch).
- No tuple form. Each port's docstring identifies it as per-group.

### One-big-array example

Measurement, setpoint, and output all live in the same global `state.x`:

```python
pid = nc.ControllerPID(
    indices=output_indices,
    measurement=(state.x, measurement_indices),
    measurement_rate=(state.x, measurement_rate_indices),
    setpoint=(state.x, setpoint_indices),
    setpoint_rate=(state.x, setpoint_rate_indices),
    kp=(kp_array, identity),                        # uses a local array, not part of the sim data.
    ki=(ki_array, identity),                        # uses a local array, not part of the sim data.
    kd=(kd_array, identity),                        # uses a local array, not part of the sim data.
    integral_max=(integral_max_array, identity),    # uses a local array, not part of the sim data.
    output=(state.x, output_indices),
)
```

The same `wp.array` reference appears in multiple ports; different index arrays disambiguate the slots.

---

## Lifecycle

```python
import warp as wp
import numpy as np
import newton
import newton.controllers as nc

N = 60                  # 10 robots * 6 DOFs
N_global = 200          # total DOFs in the simulator
dof_indices = wp.array(np.arange(N, dtype=np.uint32))    # this controller's output slots
identity = wp.arange(N, dtype=wp.uint32)                 # for any local-layout ports

@wp.struct
class ArmInputs:
    joint_q:           wp.array[float]   # global, length N_global
    joint_qd:          wp.array[float]
    joint_target_pos:  wp.array[float]
    joint_target_vel:  wp.array[float]
    kp:                wp.array[float]   # length N, laid out for this controller
    ki:                wp.array[float]
    kd:                wp.array[float]
    integral_max:      wp.array[float]

@wp.struct
class ArmOutputs:
    joint_target_force: wp.array[float]  # global

arm_in  = ArmInputs();  arm_in.joint_q = wp.zeros(N_global, dtype=wp.float32); ...
arm_out = ArmOutputs(); arm_out.joint_target_force = wp.zeros(N_global, dtype=wp.float32)

# 1. Construct a controller. Bare-array ports use the controller-level `indices`
#    as the lookup; tuple ports use their own.
pid = nc.ControllerPID(
    indices=dof_indices,
    measurement=arm_in.joint_q,
    measurement_rate=arm_in.joint_qd,
    setpoint=arm_in.joint_target_pos,
    setpoint_rate=arm_in.joint_target_vel,
    kp=(arm_in.kp, identity),
    ki=(arm_in.ki, identity),
    kd=(arm_in.kd, identity),
    integral_max=(arm_in.integral_max, identity),
    output=arm_out.joint_target_force,
)

# 2. Compose into a group. ControlGroup picks the device from the controllers'
#    bound arrays, validates agreement, calls finalize() on each, and
#    precomputes the union of all output destinations for the upfront zero pass.
group = nc.ControlGroup([pid])

# 3. Allocate state pair (double buffer).
state_0 = group.state()
state_1 = group.state()

# 4. Step loop.
for _ in range(steps):
    group.step(state_0, state_1, dt=0.005)
    state_0, state_1 = state_1, state_0

    # ... actuators, stepping sim, etc etc...

# 5. Reset (bool mask, length len(indices)).
# NOTE: see more about resetting later in this doc.
group.reset(state_0, mask=reset_mask)
```

A `Controller`'s `__init__` validates and stashes bindings; it does not allocate device buffers. `ControlGroup.__init__` picks the device, validates all controllers agree, calls `controller.finalize(device, num_outputs)` on each (allocating per-controller private buffers and the public `controller.reset_state`), and collects every controller's `outputs()` bindings into a flat list to be zeroed at the start of each step.

---

## Output accumulation (direct write)

`ControlGroup` zeros all output destinations before any controller's `compute()` is called. At the start of each `step()`, it walks the flat list of `(array, port_indices)` bindings collected from every controller's `outputs()` and, for each one, launches a kernel that writes zero into `array[port_indices[i]]`. Two controllers binding overlapping slots will have those slots zeroed twice.

Each `Controller.compute()` then writes directly into its bound output array(s) using `+=`:

```python
# Inside a per-DOF kernel:
i = wp.tid()
out_idx = output_indices[i]
output_array[out_idx] += controller_contribution
```

Controllers run **serially** (sequential `wp.launch` calls, made much more efficient by graphing). Two controllers with overlapping output indices simply accumulate; no atomic_add is needed.

**Composition semantics.** `ControllerPID + ControllerGravityComp + ControllerFeedforward` all writing to the same `joint_target_force` array produce the sum of their contributions. There are no overlap checks; users compose at their own risk.

**Multi-output controllers** (e.g. `ControllerDifferentialIK` writes both `joint_target_qd` and `joint_target_q`) declare multiple `(output_array, output_indices)` bindings via `outputs()`. The framework treats each binding equivalently for zero / accumulate purposes.

---

## Reset semantics

Reset is a controller-defined operation that updates the live State from a per-controller "reset target" State. Each controller has a public attribute `reset_state: Controller.State`, allocated at `finalize()` and zero-initialized. The user mutates `controller.reset_state` to customize what reset writes; they can do so any time after `finalize()` and before the next reset call.

**User-facing call:**

```python
group.reset(state, mask)
```

- `state` is a `ControlGroup.State`.
- `mask` is a `wp.array[wp.bool]` containing one boolean per *global DOF starting index*. `mask[g] = True` means "reset the DOF at global starting index `g`." The mask is the same shape regardless of how many controllers exist or how their internal state is laid out.

To customize what reset writes, mutate the controller's stash ahead of time:

```python
pid = nc.ControllerPID(...)
group = nc.ControlGroup([pid])              # finalize allocates pid.reset_state (zeros)
pid.reset_state.integral.fill_(0.5)         # bias the reset target

group.reset(state_0, mask=reset_mask)       # PID's masked entries → 0.5
```

**DOF starting indices, not slot indices.** Each entry in `mask` corresponds to one DOF — but a DOF's footprint inside a controller's State may be larger than one element (e.g., a quaternion-tracking filter that stores 4 floats per DOF). The `mask` is still indexed by DOF starting index, not by per-element slot. How a DOF's starting index expands into a chunk of state (1 element, 4 elements, …) is the controller author's business.

**How it works.** `ControlGroup.reset(state, mask)` walks `(controller, sub_state)` pairs and calls `controller.reset(sub_state, mask)` for each non-None sub_state. There is *no* framework-level interpretation of what the mask means for each controller's state — the controller is handed the raw mask and decides how to use it. A typical pattern is to launch a kernel of `len(self._indices)` threads where thread `i` reads `mask[self._indices[i]]` and resets the corresponding chunk of state from `self.reset_state`.

The same kernel is callable from C++ directly with the raw `(state_field, reset_state_field, controller_indices, mask)` array pointers — there is no Python-only sugar in the kernel-level contract.

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

The group reads from `state_0` and writes to `state_1` on each step. After step, the caller swaps. Reset uses a bool mask; see *Reset semantics* above.

---

## Subclassing a Controller

```python
class Controller:
    @dataclass
    class State:
        """Pure data container. Subclasses declare their fields (e.g.
        integral arrays, history buffers). No methods — reset is on Controller."""

    # Set at finalize() to a zero-initialized State. Users mutate this attribute
    # to customize what subsequent reset() calls write. Stateless controllers
    # leave this as None.
    reset_state: Controller.State | None

    def __init__(self, **ports):
        """Validate shapes / dtypes, normalize each port to (array, port_indices)
        form via _normalize_port, stash bindings on self. Does NOT allocate
        device buffers — finalize() does that.

        Subclasses declare which kwargs they accept; missing required ports
        raise here, unknown ports raise here, and shape / dtype / length
        mismatches raise here. Scalar value-range checks (e.g. ``kp >= 0``)
        are NOT performed at this layer — they would force a synchronous
        device-to-host copy. Such checks belong in a config layer above
        the controller."""

    def finalize(self, device: wp.Device, num_outputs: int) -> None:
        """Allocate device-side private buffers (e.g. internal Model +
        State + Jacobian buffers for coupled controllers) AND the
        reset_state (zero-initialized via self.state(num_outputs, device)
        for stateful controllers). Called by ControlGroup after
        construction. num_outputs == len(indices)."""

    def state(self, num_outputs: int, device: wp.Device) -> Controller.State | None:
        """Allocate a fresh State, or None if stateless."""

    def is_stateful(self) -> bool: ...
    def is_graphable(self) -> bool: ...

    def outputs(self) -> list[tuple[wp.array[float], wp.array[wp.uint32]]]:
        """Return the (output_array, output_port_indices) bindings this
        controller writes to. ControlGroup collects these from every
        controller and zeros each one before every step. Most controllers
        return a single binding; multi-output controllers (e.g.
        ControllerDifferentialIK) return multiple."""

    def compute(
        self,
        state: Controller.State | None,
        next_state: Controller.State | None,
        dt: float,
    ) -> None:
        """Read bound inputs, write `+=` into bound outputs, write next_state.
        Called by ControlGroup.step. The device is fixed at finalize()
        time, so compute() does not take one."""

    def reset(
        self,
        state: Controller.State,
        mask: wp.array[wp.bool],
    ) -> None:
        """Update `state` from `self.reset_state` for the DOFs flagged in
        `mask`. `mask` is a bool array indexed by global DOF starting
        index — `mask[g] = True` means "reset DOF g." A typical
        implementation launches a kernel of len(self._indices) threads
        where thread i reads mask[self._indices[i]] and resets the
        corresponding chunk of state from self.reset_state.

        Stateless controllers leave this as a no-op (or don't override)."""
```

The base class is neutral about the per-DOF vs. coupled distinction. Subclasses decide whether to take a `model=` kwarg, what their `indices`-length divisibility constraint is, and what kernel launch dimensionality to use.

### Example: using ControllerPID (independent flavor)

Reference implementation: `newton/_src/controllers/impl/controller_pid.py`.

```python
import warp as wp
import numpy as np
import newton.controllers as nc

device = wp.get_device()
N = 6                                                            # DOFs this controller manages
indices  = wp.array(np.arange(N, dtype=np.uint32), device=device)
identity = wp.arange(N, dtype=wp.uint32, device=device)          # for any local-layout port

zeros = lambda: wp.zeros(N, dtype=wp.float32, device=device)
gain  = lambda v: wp.array([v] * N, dtype=wp.float32, device=device)

measurement      = zeros()      # bare → looked up via the controller-level `indices`
measurement_rate = zeros()
setpoint         = zeros()
setpoint_rate    = zeros()
output           = zeros()

pid = nc.ControllerPID(
    indices=indices,
    measurement=measurement,
    measurement_rate=measurement_rate,
    setpoint=setpoint,
    setpoint_rate=setpoint_rate,
    kp=(gain(50.0), identity),                                    # local-layout gain
    ki=(gain( 1.0), identity),
    kd=(gain( 5.0), identity),
    integral_max=(gain(float("inf")), identity),                  # disable clamping
    output=output,
)
group = nc.ControlGroup([pid])

state_0, state_1 = group.state(), group.state()

# Step loop:
for _ in range(steps):
    group.step(state_0, state_1, dt=0.005)
    state_0, state_1 = state_1, state_0

# Reset some DOFs to the controller's reset_state values:
#   (pid.reset_state.integral was zero-initialized; mutate it for non-zero reset)
mask = wp.array([True, False, True, False, True, False], dtype=wp.bool, device=device)
group.reset(state_0, mask=mask)
```

### Example: ControllerDifferentialIK (coupled flavor)

```python
class ControllerDifferentialIK(Controller):
    """One-step damped-least-squares differential IK for a single end-effector
    per robot. Coupled per-robot: each robot's joint-velocity solution depends
    on its full configuration q.

    Ports:
        indices                       — flat global DOF indices; len(indices) % model.joint_dof_count == 0
        measurement                   — global joint_q (per-DOF lookup)
        measurement_rate              — global joint_qd (per-DOF lookup)
        target_pos                    — per-group: wp.array[wp.vec3], length num_robots, in base frame
        target_quat                   — per-group: wp.array[wp.quat], length num_robots, in base frame
        damping                       — per-group: wp.array[float], length num_robots (DLS λ)
        output_qd                     — global joint_qd_target (per-DOF lookup)
        output_q                      — global joint_q_target (per-DOF lookup)

    Model assumption: the supplied Model contains K=model.articulation_count
    topologically identical articulations, each with its base at the origin.
    All K articulations must share DOF count, link/joint count, and joint
    types; they may differ in physical parameters. Target poses are
    expressed in that shared base frame. The controller tiles the Model
    R = len(indices) // model.joint_dof_count times internally at finalize().
    num_robots = K * R.
    """

    def __init__(
        self,
        *,
        model: newton.Model,            # K articulations, topologically identical
        indices,                        # flat; len(indices) % model.joint_dof_count == 0
        end_effector_link: int,         # link index within one articulation
        measurement,
        measurement_rate,
        target_pos,                     # wp.array[wp.vec3], length num_robots
        target_quat,                    # wp.array[wp.quat], length num_robots
        damping,                        # wp.array[float], length num_robots
        output_qd,
        output_q,
    ):
        self._template_model    = model
        self._end_effector_link = end_effector_link
        K = model.articulation_count
        if model.joint_dof_count % K != 0:
            raise ValueError(
                f"ControllerDifferentialIK: model.joint_dof_count={model.joint_dof_count} is not "
                f"divisible by model.articulation_count={K}; the K articulations must share DOF count."
            )
        self._dofs_per_robot = model.joint_dof_count // K
        _validate_homogeneous_articulations(model)   # same dof count, link count, joint types per articulation
        if len(indices) % model.joint_dof_count != 0:
            raise ValueError(
                f"ControllerDifferentialIK: len(indices)={len(indices)} is not a multiple "
                f"of model.joint_dof_count={model.joint_dof_count} (K={K} articulations)."
            )
        self._replication_count = len(indices) // model.joint_dof_count
        self._num_robots = K * self._replication_count
        self._indices = indices
        self._target_pos  = _validate_per_group(target_pos,  self._num_robots, wp.vec3,    "target_pos")
        self._target_quat = _validate_per_group(target_quat, self._num_robots, wp.quat,    "target_quat")
        self._damping     = _validate_per_group(damping,     self._num_robots, wp.float32, "damping")
        self._measurement      = _normalize_port(measurement,      indices, name="measurement")
        self._measurement_rate = _normalize_port(measurement_rate, indices, name="measurement_rate")
        self._output_qd        = _normalize_port(output_qd,        indices, name="output_qd")
        self._output_q         = _normalize_port(output_q,         indices, name="output_q")

    def finalize(self, device, num_outputs):
        self._model = _replicate_model(self._template_model, self._replication_count, device)
        self._state = self._model.state()
        self._joint_q_local  = wp.zeros((self._num_robots, self._dofs_per_robot), dtype=wp.float32, device=device)
        self._joint_qd_local = wp.zeros((self._num_robots, self._dofs_per_robot), dtype=wp.float32, device=device)
        self._jacobian = wp.zeros(
            (self._num_robots, 6 * self._model.body_count_per_articulation, self._dofs_per_robot),
            dtype=wp.float32, device=device,
        )
        self._qd_target_local = wp.zeros((self._num_robots, self._dofs_per_robot), dtype=wp.float32, device=device)

    def is_stateful(self): return False
    def is_graphable(self): return True

    def outputs(self):
        return [self._output_qd, self._output_q]

    def compute(self, state, next_state, dt):
        # 1. Gather joint_q / joint_qd from global arrays into local buffers
        #    via the controller's indices.
        meas, meas_idx       = self._measurement
        meas_rate, mrate_idx = self._measurement_rate
        wp.launch(_gather_to_local_2d,
                  dim=(self._num_robots, self._dofs_per_robot),
                  inputs=[meas, meas_idx, self._dofs_per_robot],
                  outputs=[self._joint_q_local])
        wp.launch(_gather_to_local_2d,
                  dim=(self._num_robots, self._dofs_per_robot),
                  inputs=[meas_rate, mrate_idx, self._dofs_per_robot],
                  outputs=[self._joint_qd_local])

        # 2. Forward kinematics on the replicated Model.
        newton.eval_fk(self._model, self._joint_q_local, self._joint_qd_local, self._state)

        # 3. Spatial Jacobian.
        newton.eval_jacobian(self._model, self._state, J=self._jacobian)

        # 4. Damped least squares solve per robot. For each robot r:
        #    - Extract J_r (the 6-row sub-Jacobian for end_effector_link).
        #    - Compute task-space error e_r from (current EE pose vs. target pose).
        #    - Solve (J_r J_rᵀ + λ²I) y = e_r via in-kernel Cholesky on the 6×6 SPD system.
        #    - q̇_r = J_rᵀ y.
        wp.launch(_diff_ik_dls_kernel,
                  dim=self._num_robots,
                  inputs=[self._jacobian, self._state.body_q, self._target_pos, self._target_quat,
                          self._damping, self._end_effector_link, self._dofs_per_robot],
                  outputs=[self._qd_target_local])

        # 5. Accumulate into the global output arrays via the bound output indices.
        out_qd, out_qd_idx = self._output_qd
        out_q,  out_q_idx  = self._output_q
        wp.launch(_accumulate_qd_and_integrate_q_kernel,
                  dim=(self._num_robots, self._dofs_per_robot),
                  inputs=[self._qd_target_local, self._joint_q_local, dt,
                          out_qd_idx, out_q_idx, self._dofs_per_robot],
                  outputs=[out_qd, out_q])
```

Kernel sequence: gather → FK → Jacobian → DLS solve (in-kernel 6×6 Cholesky, left-damping form `(JJᵀ + λ²I)y = ẋ_d, q̇ = Jᵀy`) → accumulate `q̇` and integrate `q = q_current + q̇ * dt`. The "fixed base" assumption is in the template Model: the user constructs it with the root body at the origin; per-robot world-frame placement isn't tracked because the controller operates entirely in the base frame.

---

## ControlGroup

```python
class ControlGroup:
    def __init__(self, controllers: list[Controller]):
        """Pick the device from controllers' bound output arrays, validate
        all agree, finalize() every controller (each controller's
        num_outputs is len(its indices)), collect every controller's
        outputs() into a flat list to be zeroed at the start of each step."""

    def is_stateful(self) -> bool: ...
    def is_graphable(self) -> bool: ...

    def state(self) -> ControlGroup.State | None:
        """Allocate composed state with one entry per stateful controller."""

    def step(self, current_state, next_state, dt: float) -> None:
        """1. Walk the collected list of output bindings and zero each one.
              Two controllers binding overlapping slots will have those
              slots zeroed twice — harmless (idempotent), and avoids any
              wp.array-identity comparison.
           2. For each controller (in registration order), call compute()
              which += writes into its bound output array(s)."""

    def reset(self, state: ControlGroup.State, mask: wp.array[wp.bool]) -> None:
        """For each (controller, sub_state) pair where sub_state is not
        None, call controller.reset(sub_state, mask). No framework-level
        interpretation of `mask` — each controller handles it according
        to its own state layout."""
```

```python
@dataclass
class ControlGroup.State:
    controller_states: list[Controller.State | None]
```

`ControlGroup.State` is a pure data container; reset is driven from `ControlGroup.reset(state, mask)` (which has access to each controller's `reset_state`).

---

## Where this fits in the simulation step

```python
group = nc.ControlGroup([pid, diff_ik])
actuator = newton.actuators.Actuator(controller=..., ...)

state_0 = group.state(); state_1 = group.state()
act_state_0 = actuator.state(); act_state_1 = actuator.state()

for _ in range(steps):
    # 1. Controllers run first. Their outputs typically land in arrays the
    #    actuator reads from (e.g. joint_target_force, joint_target_qd).
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
    ControllerDifferentialIK,
)

__all__ = [
    "ControlGroup",
    "Controller",
    "ControllerDifferentialIK",
    "ControllerPID",
]
```

Users write `from newton.controllers import ControlGroup, ControllerPID, ControllerDifferentialIK`.

---

## v0 scope

**Framework.**

- `Controller` base class with `finalize`, `state`, `is_stateful`, `is_graphable`, `outputs`, `compute`.
- `ControlGroup` composer with `state`, `step`, `reset`, `reset_per_robot`, `is_stateful`, `is_graphable`, plus the upfront-zero pass machinery.
- `newton/controllers.py` public shim.
- Helpers in `newton/_src/controllers/utils.py`:
  - `_normalize_port(port, controller_indices, name)` — normalize bare-array or tuple to `(array, port_indices)` with validation.
  - `_validate_per_group(array, num_robots, dtype, name)` — shape + dtype check for per-group ports.
  - `_validate_homogeneous_articulations(model)` — verify all `K = model.articulation_count` articulations share DOF count, link/joint count, and joint types. Raises on mismatch.
  - `_replicate_model(template, replication_count, device)` — tile a K-articulation Model `replication_count` times, producing a Model with `K * replication_count` articulations. The K=1 case yields a straight replication; K>1 yields variant-interleaved replication (used by `ControllerDifferentialIK` and any future articulated coupled controller).
- For each shipped controller: math-correctness tests, integral / state accumulation across the `state_0` / `state_1` swap, masked reset, accumulation when multiple controllers overlap, one-big-array binding sanity tests.

**Implementation order.**

1. `ControllerPID` — proves independent-per-DOF flavor + the framework end-to-end (direct-write outputs, double-buffer state, mask reset).
2. `ControllerDifferentialIK` — proves the coupled flavor: K-articulation Model + internal replication, multi-output (`q̇` and `q`), `newton.eval_fk` + `newton.eval_jacobian` reuse, in-kernel small-system Cholesky.
3. `ControllerDifferentialDrive` — first mobile controller; proves `wheels_per_robot` divisibility convention.
4. `ControllerHolonomic` — second mobile; same family.
5. *(gated on Newton's upcoming inverse-dynamics function)* `ControllerGravityComp`, `ControllerJointImpedance`, `ControllerOperationalSpace`.

**Out of scope for v0.**

- USD parsing.
- Differentiability flag (`requires_grad`) — slot reserved in `finalize` signature, not exercised.
- CUDA-graph capture testing.
- `ModelBuilder.add_controller` analog.
- Nullspace projection (joint centering, joint-limit avoidance) for `ControllerDifferentialIK` — compose a separate controller later if needed.
- Multiple end-effectors per `ControllerDifferentialIK` instance — users compose two instances under one `ControlGroup`.

---

## Controllers to design

> Per-controller specs that still need decisions before implementation.

### `ControllerGravityComp`

- **Category:** coupled, articulated. **Gated on Newton's upcoming generic inverse-dynamics function.**
- **Sketch:** `τ_g = -∂U_gravity/∂q`. Equivalent to inverse dynamics with `q̇ = q̈ = 0`.
- **Open:** port for the gravity vector (default `(0, 0, -9.81)`, but allow override). When the inverse-dynamics function lands, confirm its signature matches the per-robot batched layout the coupled-controller convention assumes.

### `ControllerJointImpedance`

- **Category:** coupled, articulated. **Gated on the inverse-dynamics function** for any variant that includes `g(q)` or `h(q,q̇)`.
- **Sketch:** `τ = M(q)(q̈_d + K_d (q̇_d - q̇) + K_p (q_d - q)) + h(q, q̇)` (or simpler stiffness-only variants).
- **Open:** variants to ship — full impedance, PD + gravity, Cartesian impedance (later)? `M(q)` is public via `newton.eval_mass_matrix`; `h(q,q̇)` needs the inverse-dynamics function.

### `ControllerOperationalSpace`

- **Category:** coupled, articulated, task-space. **Gated on the inverse-dynamics function** for the bias term μ and gravity p.
- **Sketch:** task-space inertia `Λ = (J M⁻¹ Jᵀ)⁻¹`; task force `F = Λ (ẍ_d + K_d (ẋ_d - ẋ) + K_p (x_d - x)) + μ + p`; torque `τ = Jᵀ F + (I - JᵀJ̄ᵀ) τ_null`, where `J̄ = M⁻¹ Jᵀ Λ`.
- `J` and `M` are public; `μ` and `p` need the inverse-dynamics function.
- **Open:** all-in-one OSC or split (TaskSpaceForce + JointMapping)? Inertia-weighted vs. plain damped pseudoinverse — both, or pick one? Nullspace handling — separate `ControllerNullspaceProjection` block that accumulates, or built in?

### `ControllerDifferentialDrive`

- **Category:** coupled, mobile.
- **Sketch:** input `(v, ω)` (linear m/s, angular rad/s) → wheel velocities `(ω_L, ω_R)` via `ω_L = (v - ω·L/2) / r`, `ω_R = (v + ω·L/2) / r` where `r` = wheel radius, `L` = axle width.
- **Construction convention:** takes raw scalars `wheel_radius: float`, `axle_width: float` directly. `wheels_per_robot = 2`. The user-supplied `indices` array orders `(left, right, left, right, …)` per-robot.
- **Open:** output port — wheel velocity (`joint_target_vel`) or wheel torque via an inner PI loop? Per-robot input `(v, ω)` as separate `target_v` / `target_omega` per-group ports, or combined `target_twist: wp.array[wp.vec2]`?

### `ControllerHolonomic`

- **Category:** coupled, mobile.
- **Sketch:** input body twist `(v_x, v_y, ω)` → wheel velocities for an N-wheeled omni base (3-wheel Kiwi, 4-wheel mecanum, etc.). Mapping is `wheel_velocities = W(geometry) · [v_x, v_y, ω]ᵀ`.
- **Construction convention:** takes a `wheel_geometry: wp.array2d[float]` of shape `(wheels_per_robot, 3)` directly. Each row encodes one wheel's contribution to the velocity-mapping matrix. Helpers to build the matrix for known configurations (Kiwi, mecanum) live alongside the controller.
- **Open:** which geometry helpers to ship (Kiwi / mecanum / generic builder)? Output port — wheel velocity or wheel torque?

---

## Open questions

1. **Per-controller specs.** See *Controllers to design* above.
2. **In-kernel small-system solver helper.** `ControllerDifferentialIK` needs a 6×6 Cholesky; future controllers may need similar. Inline for v0; extract to `controllers/utils.py` if a second user appears.
3. **Task-space orientation error formulation for `ControllerDifferentialIK`.** Rotation-vector (axis-angle log map) vs. quaternion vector-part? Pick one and document.
4. **`_replicate_model` implementation.** Use `newton.ModelBuilder` to clone? Or a lower-level Model-array tiling?
