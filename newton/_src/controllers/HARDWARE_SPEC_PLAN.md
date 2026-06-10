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
# --- Newton ships a small set of canonical signals (newton.State only) ---
JOINT_Q         = ControlSignal(dtype=wp.float32,        ndim=1, description="joint positions [m or rad]")
JOINT_QD        = ControlSignal(dtype=wp.float32,        ndim=1, description="joint velocities")
JOINT_TARGET_Q  = ControlSignal(dtype=wp.float32,        ndim=1, description="commanded joint positions")
JOINT_TARGET_QD = ControlSignal(dtype=wp.float32,        ndim=1, description="commanded joint velocities")
JOINT_F         = ControlSignal(dtype=wp.float32,        ndim=1, description="joint efforts [N or N·m]")
BODY_Q          = ControlSignal(dtype=wp.transform,      ndim=1, description="body world transforms")
BODY_QD         = ControlSignal(dtype=wp.spatial_vector, ndim=1, description="body spatial velocities")

# --- Other signals are demonstrated in tests/examples/docs, not shipped ---
SETPOINT = ControlSignal(dtype=wp.float32, ndim=1, description="PID setpoint")
KP       = ControlSignal(dtype=wp.float32, ndim=1, description="proportional gain")
KI       = ControlSignal(dtype=wp.float32, ndim=1, description="integral gain")
KD       = ControlSignal(dtype=wp.float32, ndim=1, description="derivative gain")
INTEGRAL_MAX = ControlSignal(dtype=wp.float32, ndim=1, description="anti-windup clamp")
# ...

# Control laws take full (signal, indices) tuples at every port (no bare-indices
# shortcut in the initial version):
pid = ControlLawPID(
    measurement      = (JOINT_Q,        idx),
    measurement_rate = (JOINT_QD,       idx),
    setpoint         = (SETPOINT,       idx),
    setpoint_rate    = (SETPOINT_RATE,  idx),
    kp               = (KP,             idx),
    ki               = (KI,             idx),
    kd               = (KD,             idx),
    integral_max     = (INTEGRAL_MAX,   idx),
    output           = (JOINT_F,        idx),
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

# At step time the user supplies whatever input / output structs they like.
# Each law has pre-resolved each port to (attr_name, port_indices) via the
# Controller's hw at construction; compute() just does getattr(input, attr_name).
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
- **Signal direction:** A signal may appear in either or both of `interface.inputs` and `interface.outputs`. If `JOINT_TARGET_Q` is written by DiffIK and also read by a downstream PID, the user lists it in both dicts; the attribute name may match (one underlying array) or differ (different fields on the input vs output struct).
- **Construction signatures (initial):** Maximally simple. Every port kwarg requires the full `(signal, port_indices)` tuple. No bare-indices shortcut, no defaulted-canonical-signal form. (Revisit later — the user wants shorter signatures, but that's a follow-up once the base design is in.)
- **Step-time access:** Each law records, per port, the *resolved* attribute name when the Controller is constructed (the Controller hands the law its `HardwareInterface` once, the law walks its ports and stashes `attr_name = hw.inputs[signal]` per input port, `attr_name = hw.outputs[signal]` per output). At step time `compute()` does plain `getattr(input, attr_name)[port_indices[i]]` — no per-step interface lookup in the hot path.
- **Indices:** Per-port at the control law's constructor. `(signal, port_indices)`.
- **Law's signal tracking:** Each law records the set of signals it received per port. The Controller validates `law._used_inputs ⊆ hw.inputs.keys()` and `law._used_outputs ⊆ hw.outputs.keys()` at construction.
- **Controller composition:** `+=` accumulation. Controller resolves the union of declared output signals' attribute names via the interface, zeros those slots, then each law `+=` writes into the resolved arrays. Two laws binding the same output signal sum into the same slot.
- **Dtype validation:** No construction-time dtype check in the initial cut (the `PORT_DTYPES` concept is dropped). Kernel-launch errors surface dtype mismatches loudly. A later refinement can validate dtype at construction by reading the bound signal's `dtype` against an expectation declared on the law's class, but that's deferred to keep the initial code minimal.
- **Newton's canonical signal set:** Newton publishes a short, Newton-state-only vocabulary of canonical `ControlSignal`s as module-level constants. Day-one list: `JOINT_Q`, `JOINT_QD`, `JOINT_TARGET_Q`, `JOINT_TARGET_QD`, `JOINT_F`, `BODY_Q` (wp.transform), `BODY_QD` (wp.spatial_vector). No `_IN` / `_OUT` direction duals — `JOINT_TARGET_Q` is a single signal, listed in whichever direction(s) a given user's `HardwareInterface` needs. PID/DiffIK-specific concepts (`SETPOINT`, `SETPOINT_RATE`, `KP`, `KI`, `KD`, `INTEGRAL_MAX`, `TARGET_POS`, `TARGET_QUAT`, `DAMPING`, `GAIN`) are **not** shipped as canonical — they're demonstrated in tests / examples / docs as the natural pattern for user-defined signals.
- **Direction exclusivity, revised:** A given `ControlSignal` may appear in *both* `hw.inputs` and `hw.outputs` of a `HardwareInterface` (different uses on different laws). The earlier "ship two distinct constants for the two directions" rule is dropped, since Newton no longer ships per-direction duals.
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
    # Subclasses declare which port-name kwargs are inputs vs outputs.
    INPUT_PORTS:  ClassVar[frozenset[str]]
    OUTPUT_PORTS: ClassVar[frozenset[str]]

    _used_inputs:  frozenset[ControlSignal]   # populated at __init__
    _used_outputs: frozenset[ControlSignal]

    def __init__(self, **port_bindings):
        # Each port kwarg is a (ControlSignal, wp.array[uint32]) tuple.
        # Validate the kwarg names against INPUT_PORTS ∪ OUTPUT_PORTS,
        # record _used_inputs / _used_outputs, stash (signal, port_indices)
        # keyed by port name. Per-DOF length cross-checks happen in subclasses.
        ...

    def _resolve(self, hw: HardwareInterface) -> None:
        # Called by Controller.__init__. For every port the subclass
        # registered, look up its attribute name via hw and stash it on
        # the law instance for fast getattr at step time. Raise if any
        # used signal is not in the right direction of hw.
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
            law._resolve(hw)   # stash attr_name per port on the law
        self._hw = hw
        self._laws = control_laws
        # ... record per-output (attr_name, port_indices) pairs for the
        # upfront zero pass at step time ...

    def input(self) -> SimpleNamespace:
        used = frozenset().union(*(law._used_inputs for law in self._laws))
        return SimpleNamespace(**{
            self._hw.inputs[signal]: wp.zeros(<size>, dtype=signal.dtype, device=self._device)
            for signal in used
        })

    def output(self) -> SimpleNamespace: ...   # same idea over _used_outputs
    def state(self)  -> Controller.State: ...

    def step(self, input, output, current_state, next_state, dt):
        # 1. For every (attr_name, port_indices) output binding collected
        #    at __init__, zero output[attr_name][port_indices].
        # 2. For each law in registration order, call
        #      law.compute(input, output, cur, nxt, dt)
        #    which += writes into its outputs.
        ...
```

## ControlLaw subclass authoring (PID example)

```python
class ControlLawPID(ControlLaw):
    # Declare which kwargs are read ports and which are write ports.
    # No canonical signal defaults yet (full-tuple-everywhere); revisit
    # when we add a bare-indices shortcut.
    INPUT_PORTS  = {"measurement", "measurement_rate",
                    "setpoint", "setpoint_rate",
                    "kp", "ki", "kd", "integral_max"}
    OUTPUT_PORTS = {"output"}

    def __init__(
        self,
        *,
        # Every kwarg is the explicit form: (ControlSignal, wp.array[wp.uint32]).
        measurement, measurement_rate, setpoint, setpoint_rate,
        kp, ki, kd, integral_max,
        output,
    ):
        # For each port:
        #   signal, port_indices = kwarg_value
        #   record into _used_inputs or _used_outputs (per INPUT_PORTS/OUTPUT_PORTS).
        #   stash (signal, port_indices) keyed by port name for later.
        # num_outputs derived from output's port_indices length;
        # other per-DOF ports cross-check against it.
        # No dtype validation at construction (initial cut). Kernel-launch
        # errors surface dtype mismatches if a user binds the wrong-typed
        # signal.
```

At construction the base class records `_used_inputs` / `_used_outputs` from the kwargs. When the law is handed to a `Controller`, the Controller calls a `law._resolve(hw)` step that:

- checks every used input signal is in `hw.inputs` (raise if not),
- checks every used output signal is in `hw.outputs` (raise if not),
- stashes the resolved `attr_name` per port on the law (`self._measurement_attr = hw.inputs[meas_signal]`, etc.).

After that, `compute()` is the same shape as today's string-port API:

```python
def compute(self, input, output, cur_state, nxt_state, dt):
    meas = getattr(input, self._measurement_attr)
    out  = getattr(output, self._output_attr)
    # ... and so on for every port; then launch kernels.
```

The HardwareInterface is only consulted at construction / `Controller.__init__`. The hot path is plain attribute access, identical to today.

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

- `newton/_src/controllers/utils.py` — add `ControlSignal`, `HardwareInterface`. Replace string-name port normalizers with signal-based normalizers that take `(ControlSignal, wp.array[uint32])` tuples.
- `newton/_src/controllers/standard_signals.py` (new) — Newton-state canonical `ControlSignal` module-level constants only: `JOINT_Q`, `JOINT_QD`, `JOINT_TARGET_Q`, `JOINT_TARGET_QD`, `JOINT_F`, `BODY_Q`, `BODY_QD`. Nothing PID/DiffIK-specific. No pre-built `HardwareInterface`.
- `newton/_src/controllers/control_law.py` — `ControlLaw` base records `_used_inputs`, `_used_outputs`. Adds a `_resolve(hw)` hook the `Controller` calls at composition time to stash per-port `attr_name` on the law. Subclasses declare `INPUT_PORTS` / `OUTPUT_PORTS` (sets of port-name strings) — no canonical-signal defaults yet, no `PORT_DTYPES`.
- `newton/_src/controllers/controller.py` — `Controller` takes `hw: HardwareInterface` and `control_laws=`; validates that every law's used signals are covered by `hw` in the right direction; calls `law._resolve(hw)` on each; records output `(attr_name, port_indices)` for the upfront zero pass; passes `input`/`output` straight to each `compute()`.
- `newton/_src/controllers/impl/controller_pid.py` — declare `INPUT_PORTS` / `OUTPUT_PORTS`. Every kwarg accepts only `(signal, indices)`. Derive `num_outputs` from the output port. Track `_used_inputs` / `_used_outputs`. No construction-time dtype validation (kernel-launch handles it).
- `newton/_src/controllers/impl/controller_diff_ik.py` — same per-port pattern; keep `model_builder`, `num_robots`, `dofs_per_robot` on the law's constructor.
- `newton/tests/test_controllers.py` — every test rewritten to: define the signals it needs (Newton's canonical + locally-defined), construct a `HardwareInterface`, bind ports as `(signal, indices)`, assert behaviour. Hard cut-over; no string-port form supported.
- `newton/examples/controllers/*.py` — same. PID example demonstrates user-defined SETPOINT / KP / KI / KD / INTEGRAL_MAX signals; DiffIK example demonstrates user-defined TARGET_POS / TARGET_QUAT / DAMPING / GAIN.
- `newton/_src/controllers/DESIGN_DOC.md` — rewrite for the signal + interface model.

## Next step

Plan is captured for review. Open for iteration. When ready, the natural follow-ups are: (a) walk through extension examples (variable-impedance PD, operational space, gravity comp) to confirm the design absorbs them; (b) decide tentative answers in *Open details* above; (c) commit and start implementation.
