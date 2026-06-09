# Controllers + Hardware Interface (exploration)

## Context

Today's control laws identify their ports by string attribute names directly:

```python
pid = ControlLawPID(
    measurement=("joint_q", indices),
    ...
)
```

That's flexible but has weaknesses:

- The string `"joint_q"` is a magic value coupled to whatever object the user later passes to `step()`. Typos surface at runtime, and there's nothing pinning down what `"joint_q"` *means* (its dtype, its expected shape, that it's the joint-position-coordinate-vector and not something else).
- A `Controller` composing multiple control laws has no way to confirm that all its laws are reading from / writing to the *same* logical interface. Two laws could each invent their own attribute names and the framework wouldn't notice.
- The control law itself is tied to a specific attribute name. A `ControlLawPID` that knows "joint_q" is not portable to another system whose joint positions live at `"q"` or `"positions"`.

The proposal: split the concept into two pieces.

1. **`ControlSignal`** — the *slot type* describing a kind of array (`dtype`, `ndim`, `description`). No attribute name. Newton ships canonical signals as module-level constants.
2. **`HardwareInterface`** — the *runtime wiring* per deployment: a mapping `signal → attribute_name`. The user writes this once for their system; it answers "where on `input` does this signal live?"

A `ControlLaw` is wired against signals (canonically by default; overridable per-port). The user provides indices. The `Controller` is constructed with a `HardwareInterface` and resolves each law's signals to attribute names at step time. The control law itself never knows or cares about attribute names — fully portable across systems whose signals live at any names.

## User sketch (target API shape)

```python
# Newton ships canonical signals as module-level constants. Each carries
# dtype + ndim + description; NO attribute name (the name is per-deployment).
JOINT_Q  = ControlSignal(dtype=wp.float32, ndim=1, description="joint positions [m or rad]")
JOINT_QD = ControlSignal(dtype=wp.float32, ndim=1, description="joint velocities")
JOINT_F  = ControlSignal(dtype=wp.float32, ndim=1, description="joint efforts [N or N·m]")
SETPOINT = ControlSignal(dtype=wp.float32, ndim=1, description="PID setpoint")
SETPOINT_RATE = ControlSignal(dtype=wp.float32, ndim=1, description="PID setpoint rate")
KP = ControlSignal(dtype=wp.float32, ndim=1, description="proportional gain")
KI = ControlSignal(dtype=wp.float32, ndim=1, description="integral gain")
KD = ControlSignal(dtype=wp.float32, ndim=1, description="derivative gain")
INTEGRAL_MAX = ControlSignal(dtype=wp.float32, ndim=1, description="anti-windup clamp")
# ...

# A control-law subclass declares its canonical signal per port at the
# class level:
class ControlLawPID(ControlLaw):
    CANONICAL_INPUTS = {
        "measurement":      JOINT_Q,
        "measurement_rate": JOINT_QD,
        "setpoint":         SETPOINT,
        "setpoint_rate":    SETPOINT_RATE,
        "kp": KP, "ki": KI, "kd": KD,
        "integral_max":     INTEGRAL_MAX,
    }
    CANONICAL_OUTPUTS = {"output": JOINT_F}

    def __init__(
        self,
        *,
        # Each kwarg accepts:
        #   - wp.array[wp.uint32] (bare port_indices; signal defaults to canonical)
        #   - (ControlSignal, wp.array[wp.uint32]) (explicit signal override)
        measurement, measurement_rate, setpoint, setpoint_rate,
        kp, ki, kd, integral_max,
        output,
    ): ...

# Common case — user provides just indices; signals default to canonical:
pid = ControlLawPID(
    measurement=idx, measurement_rate=idx,
    setpoint=idx, setpoint_rate=idx,
    kp=idx, ki=idx, kd=idx, integral_max=idx,
    output=idx,
)

# Override case — user binds a non-canonical signal at one port (e.g., a
# user-defined filtered-joint-position signal):
MY_FILTERED_Q = ControlSignal(dtype=wp.float32, ndim=1, description="low-pass filtered joint positions")
pid_with_filter = ControlLawPID(
    measurement=(MY_FILTERED_Q, idx),   # explicit override
    measurement_rate=idx,
    # ... rest default ...
    output=idx,
)

# The HardwareInterface is the user's per-system wiring. It maps each
# signal the laws will reference to the attribute name on the input/output
# object the user will pass at step time.
hw = HardwareInterface(
    inputs={
        JOINT_Q: "joint_q",   JOINT_QD: "joint_qd",
        SETPOINT: "setpoint", SETPOINT_RATE: "setpoint_rate",
        KP: "kp", KI: "ki", KD: "kd",
        INTEGRAL_MAX: "integral_max",
    },
    outputs={JOINT_F: "joint_f"},
)

controller = Controller(hw, control_laws=[pid, ...])

# At step time the Controller (or the law via the hw passed through it)
# resolves each port's signal → attribute name via `hw`, then does
# getattr(input, name) for the live array.
controller.step(input, output, cs0, cs1, dt)
```

## What this buys us

- **Portable control laws.** A `ControlLawPID` knows nothing about attribute names. It declares "I need the JOINT_Q slot for `measurement`." A user with a different system just writes a `HardwareInterface` whose `JOINT_Q` maps to whatever attribute their input object uses. The controller code stays the same.
- **Single source of truth for runtime wiring.** A signal name (the attribute) lives in exactly one place per system: the `HardwareInterface`. Renaming "joint_q" → "q" is one line.
- **Composition discipline.** A `Controller`'s `HardwareInterface` must cover (be a superset of) every signal every law uses, in the right direction. Two laws can't silently bind disjoint signal sets without the interface noticing.
- **Semantic typing.** A `ControlSignal` carries dtype + ndim + description; the law can validate at construction that the user's signal override makes sense (e.g., that a per-DOF float port isn't being bound to a `wp.vec3` signal).
- **Extensibility through new signals.** A variable-impedance PD controller adds a new `ControlSignal` (`KP_LIVE`) without modifying any existing code — and a user pulls it in by adding one entry to their `HardwareInterface`.

## Resolved decisions

- **Vocabulary unit:** `ControlSignal`, the slot type. Carries only `(dtype, ndim, description)` — no attribute name. Module-level constants are canonical; identity-equal (two same-field constructions are different signals).
- **`HardwareInterface`:** Per-deployment wiring. Two `dict[ControlSignal, str]`s, `inputs` and `outputs`, mapping each signal to its attribute name on the runtime input/output object.
- **Signal direction:** Exclusive. A signal lives in `interface.inputs` or `interface.outputs`, not both. If a signal flows both ways in some application (e.g., joint targets that DiffIK writes and a downstream PID reads), Newton ships two distinct signal constants for the two directions.
- **Canonical wiring on a `ControlLaw`:** Each subclass declares `CANONICAL_INPUTS` / `CANONICAL_OUTPUTS` class attributes mapping each port name to a default `ControlSignal`. The constructor's port kwargs accept either a bare `port_indices` (canonical signal used) or a `(signal, port_indices)` tuple (explicit override). This makes `ControlLawPID` portable: it knows JOINT_Q is its canonical measurement signal but not what attribute name a given user puts JOINT_Q at.
- **Step-time access:** Per-law `compute()` resolves each port via the Controller's `HardwareInterface`: `name = hw.inputs[signal]; arr = getattr(input, name)`. The interface is the only place where signal-to-name resolution happens.
- **Indices:** Per-port at the control law's constructor. Either bare `port_indices` (canonical signal) or `(signal, port_indices)`.
- **Law's signal tracking:** Each law records the set of signals it actually uses (after canonical defaults are filled in). The Controller validates `law._used_inputs ⊆ hw.inputs.keys()` and `law._used_outputs ⊆ hw.outputs.keys()` at construction.
- **Controller composition:** `+=` accumulation. Controller resolves the union of declared output signals' attribute names via the interface, zeros those slots, then each law `+=` writes into the resolved arrays. Two laws binding the same output signal sum into the same slot.
- **Per-law dtype validation:** Each `ControlLaw` subclass validates `signal.dtype` (and `ndim`) against the per-port expected dtype at `__init__`. PID's `measurement` rejects a `wp.vec3` signal at construction with a clear error. The canonical signals already match their ports' expected dtypes; overrides get checked.
- **Newton's standard signal set:** Newton publishes a vocabulary of canonical `ControlSignal`s as module-level constants — `JOINT_Q`, `JOINT_QD`, `JOINT_F`, plus the input/output duals of `JOINT_TARGET_Q` / `JOINT_TARGET_QD`, plus PID staples (`SETPOINT`, `SETPOINT_RATE`, `KP`, `KI`, `KD`, `INTEGRAL_MAX`) and DiffIK staples (`TARGET_POS`, `TARGET_QUAT`, `DAMPING`, `GAIN`). No pre-built `HardwareInterface`; every user assembles their own.
- **Input/output factories:** `controller.input()` / `controller.output()` allocate fresh objects whose attributes match the interface's name strings, with `wp.zeros(<size>, dtype=signal.dtype)` per used signal. Mirrors `controller.state()`. User mutates fields to share arrays with sim.
- **`num_outputs` on per-DOF laws:** Derived from the output port's `port_indices` length. Every other per-port `port_indices` is cross-checked against it at `__init__`.
- **Signal aliasing within a law:** Allowed. Two ports on one law can bind the same signal with different `port_indices`.
- **Migration:** Hard cut-over. The string-port API gets replaced with the signal API in a single PR. Tests, examples, and the panda demo all migrate together.

## Proposed data model

```python
@dataclass(frozen=True, eq=False)  # eq=False => identity equality
class ControlSignal:
    """A slot type. Describes what kind of array fills this slot. No
    attribute name — that's per-deployment via HardwareInterface."""
    dtype: type        # wp.float32, wp.vec3, wp.quat, ...
    ndim: int          # 1 for the common case; 2+ reserved
    description: str   # human-readable; for docstring / debug

@dataclass(frozen=True)
class HardwareInterface:
    """Per-deployment wiring: which attribute on the runtime input/output
    object holds each signal's live array."""
    inputs:  dict[ControlSignal, str]
    outputs: dict[ControlSignal, str]

    def covers_inputs(self,  signals: set[ControlSignal]) -> bool:
        return signals <= self.inputs.keys()
    def covers_outputs(self, signals: set[ControlSignal]) -> bool:
        return signals <= self.outputs.keys()
```

The `ControlLaw` base class records its signal usage:

```python
class ControlLaw:
    # Subclass class attributes — canonical signal per port:
    CANONICAL_INPUTS:  dict[str, ControlSignal]  # port_name -> signal
    CANONICAL_OUTPUTS: dict[str, ControlSignal]

    _used_inputs:  frozenset[ControlSignal]
    _used_outputs: frozenset[ControlSignal]
    # Per port, the law stashes (signal, port_indices) after construction.

    def __init__(self, **port_bindings):
        # Each port kwarg is either:
        #   - wp.array[uint32] (bare port_indices; signal = CANONICAL_*[name])
        #   - (ControlSignal, wp.array[uint32]) (explicit override)
        # Normalize, validate dtype against the law's per-port expectation,
        # cross-check per-DOF lengths, record _used_inputs / _used_outputs.
        ...
```

The `Controller`:

```python
class Controller:
    def __init__(self, hw: HardwareInterface, control_laws: list[ControlLaw]):
        for law in control_laws:
            if not hw.covers_inputs(law._used_inputs):
                raise ValueError(...)
            if not hw.covers_outputs(law._used_outputs):
                raise ValueError(...)
        self._hw = hw
        self._laws = control_laws

    def input(self) -> SimpleNamespace:
        used = frozenset().union(*(law._used_inputs for law in self._laws))
        return SimpleNamespace(**{
            self._hw.inputs[signal]: wp.zeros(<size>, dtype=signal.dtype, device=self._device)
            for signal in used
        })

    def output(self) -> SimpleNamespace: ...   # same idea over _used_outputs
    def state(self)  -> Controller.State: ...

    def step(self, input, output, current_state, next_state, dt):
        # Resolve every output signal -> attribute name via self._hw,
        # zero the declared slots, then call each law's compute() passing
        # the interface (or pre-resolved arrays) through.
        ...
```

## ControlLaw subclass authoring (PID example)

```python
class ControlLawPID(ControlLaw):
    # Canonical signals — used as the default for each port when the user
    # passes only port_indices.
    CANONICAL_INPUTS = {
        "measurement":      JOINT_Q,
        "measurement_rate": JOINT_QD,
        "setpoint":         SETPOINT,
        "setpoint_rate":    SETPOINT_RATE,
        "kp":               KP,
        "ki":               KI,
        "kd":               KD,
        "integral_max":     INTEGRAL_MAX,
    }
    CANONICAL_OUTPUTS = {"output": JOINT_F}

    # Per-port expected dtype/ndim — used to validate signal overrides at
    # __init__. (Could be folded into CANONICAL_*; tentatively separate.)
    PORT_DTYPES = {p: wp.float32 for p in CANONICAL_INPUTS} | {"output": wp.float32}

    def __init__(
        self,
        *,
        # Each kwarg accepts:
        #   - wp.array[wp.uint32]                 (bare indices; canonical signal used)
        #   - (ControlSignal, wp.array[wp.uint32]) (explicit signal override)
        measurement, measurement_rate, setpoint, setpoint_rate,
        kp, ki, kd, integral_max,
        output,
    ):
        # Per port (let `m` stand in for the kwarg value):
        #   if isinstance(m, wp.array):
        #       signal = CANONICAL_INPUTS["measurement"]
        #       port_indices = m
        #   else:
        #       signal, port_indices = m
        #   assert signal.dtype == PORT_DTYPES["measurement"]
        # The output port's port_indices defines num_outputs; all other
        # ports' port_indices lengths cross-check against it.
        # Record _used_inputs / _used_outputs.
        # Stash (signal, port_indices) per port for compute().
```

At step time, the law's `compute()` resolves each port through the Controller's `HardwareInterface`:

```python
def compute(self, hw, input, output, cur_state, nxt_state, dt):
    meas_signal, meas_idx = self._measurement
    meas_array = getattr(input, hw.inputs[meas_signal])
    # ... and so on for every port; then launch kernels.
```

(The Controller either passes `hw` to each `compute()` directly, or pre-resolves every law's port to `(array, port_indices)` once per step and hands those pre-resolved bindings to the law. The latter avoids per-law `getattr` overhead in the hot loop; tentative choice is "pre-resolve in `Controller.step`.")

## Open details (tentative answers; revisit at implementation time)

- **Interface containing signals no law uses:** Allowed. Interfaces are reusable across configurations. No warning.
- **A signal appearing twice in the same direction map (e.g., two entries with different attribute names):** A dict naturally rules this out (signal is the key, attribute name is the value). Catching the *reverse* — two different signals pointing at the same attribute name — happens at interface construction with a clear error.
- **Concrete type returned by `controller.input()`:** `SimpleNamespace` for simplicity. Can promote to a generated `@dataclass` later if static analysis benefits become valuable.
- **`HardwareInterface` mutability:** Frozen `dict` (e.g., `MappingProxyType`); immutable after construction.
- **`compute()` signature:** Tentatively, the Controller pre-resolves each law's `(signal, port_indices)` to `(wp.array, port_indices)` once per `step()` and passes the resolved bindings to the law's compute (so compute never does `getattr`). Cleaner; takes the interface lookup out of the hot path.
- **CUDA-graph compatibility:** Almost certainly fine — resolution happens once per step (pre-step); the kernels see plain `wp.array`s. Confirm once an implementation exists.
- **Extension examples to walk through before implementing:** variable-impedance PD, Cartesian operational-space, gravity comp. Confirms the design absorbs these without forced patterns.

## Migration / files to touch (when execution begins)

This is a substantial refactor; rough scope:

- `newton/_src/controllers/utils.py` — add `ControlSignal`, `HardwareInterface`. Replace string-name port normalizers with signal-based normalizers that accept bare indices (using class-canonical signal) or `(signal, indices)` tuples.
- `newton/_src/controllers/standard_signals.py` (new) — canonical `ControlSignal` module-level constants: `JOINT_Q`, `JOINT_QD`, `JOINT_F`, input/output duals of `JOINT_TARGET_Q` / `JOINT_TARGET_QD`, plus PID staples (`SETPOINT`, `SETPOINT_RATE`, `KP`, `KI`, `KD`, `INTEGRAL_MAX`) and DiffIK staples (`TARGET_POS`, `TARGET_QUAT`, `DAMPING`, `GAIN`). No pre-built `HardwareInterface`.
- `newton/_src/controllers/control_law.py` — `ControlLaw` base records `_used_inputs`, `_used_outputs`. Subclasses declare `CANONICAL_INPUTS` / `CANONICAL_OUTPUTS` / `PORT_DTYPES` class attributes.
- `newton/_src/controllers/controller.py` — `Controller` takes `hw: HardwareInterface` and `control_laws=`; validates that every law's used signal set is covered by `hw`; pre-resolves each port to `(array, port_indices)` once per `step()` via the interface and passes resolved bindings down to each law's compute.
- `newton/_src/controllers/impl/controller_pid.py` — declare `CANONICAL_INPUTS` / `CANONICAL_OUTPUTS` / `PORT_DTYPES`. Each kwarg accepts bare `wp.array[uint32]` (canonical default) or `(signal, indices)`. Derive `num_outputs` from the output port. Validate dtype per port. Track `_used_inputs` / `_used_outputs`.
- `newton/_src/controllers/impl/controller_diff_ik.py` — same canonical-port pattern; keep `model_builder`, `num_robots`, `dofs_per_robot` on the law's constructor.
- `newton/tests/test_controllers.py` — every test rewritten to construct a `HardwareInterface`, bind ports either canonically or with explicit signals, and assert behaviour. Hard cut-over; no string-port form supported.
- `newton/examples/controllers/*.py` — same.
- `newton/_src/controllers/DESIGN_DOC.md` — rewrite for the signal + interface model.

## Next step

Plan is captured for review. Open for iteration. When ready, the natural follow-ups are: (a) walk through extension examples (variable-impedance PD, operational space, gravity comp) to confirm the design absorbs them; (b) decide tentative answers in *Open details* above; (c) commit and start implementation.
