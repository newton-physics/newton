# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Stage-driven simulation runtime: derive a Newton simulation from USD."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import newton
import newton.usd
from newton._src.solvers import SolverFeatherstone, SolverMuJoCo, SolverSemiImplicit, SolverVBD, SolverXPBD
from newton._src.usd.utils import _get_raw_api_schemas, get_attribute

if TYPE_CHECKING:
    from pxr import Usd


@dataclass
class _SolverEntry:
    schema: str
    cls: type
    param_attrs: dict[str, str]
    uses_newton_contacts: bool = True


_SOLVER_REGISTRY: dict[str, _SolverEntry] = {
    e.schema: e
    for e in [
        _SolverEntry(
            schema="NewtonSolverXpbdAPI",
            cls=SolverXPBD,
            param_attrs={
                "newton:xpbd:iterations": "iterations",
                "newton:xpbd:angularDamping": "angular_damping",
                "newton:xpbd:enableRestitution": "enable_restitution",
            },
        ),
        _SolverEntry(
            schema="NewtonSolverMujocoAPI",
            cls=SolverMuJoCo,
            param_attrs={
                "newton:mujoco:iterations": "iterations",
                "newton:mujoco:lsIterations": "ls_iterations",
                "newton:mujoco:njmax": "njmax",
                "newton:mujoco:nconmax": "nconmax",
            },
            # MVP runs MuJoCo with its internal collision pipeline only.
            uses_newton_contacts=False,
        ),
        _SolverEntry(
            schema="NewtonSolverFeatherstoneAPI",
            cls=SolverFeatherstone,
            param_attrs={
                "newton:featherstone:angularDamping": "angular_damping",
                "newton:featherstone:updateMassMatrixInterval": "update_mass_matrix_interval",
                "newton:featherstone:frictionSmoothing": "friction_smoothing",
            },
        ),
        _SolverEntry(
            schema="NewtonSolverSemiImplicitAPI",
            cls=SolverSemiImplicit,
            param_attrs={
                "newton:semiImplicit:angularDamping": "angular_damping",
                "newton:semiImplicit:frictionSmoothing": "friction_smoothing",
                "newton:semiImplicit:jointAttachKe": "joint_attach_ke",
                "newton:semiImplicit:jointAttachKd": "joint_attach_kd",
            },
        ),
        _SolverEntry(
            schema="NewtonSolverVbdAPI",
            cls=SolverVBD,
            param_attrs={
                "newton:vbd:iterations": "iterations",
                "newton:vbd:frictionEpsilon": "friction_epsilon",
            },
        ),
    ]
}


def _find_physics_scene(stage: Usd.Stage) -> Usd.Prim:
    from pxr import UsdPhysics

    scenes = [p for p in stage.Traverse() if p.IsA(UsdPhysics.Scene)]
    if len(scenes) != 1:
        raise ValueError(
            f"newton.usd.runtime requires exactly one PhysicsScene prim, found {len(scenes)}."
            " Multiple scenes are reserved for future multi-solver support."
        )
    return scenes[0]


def _resolve_solver_entry(scene_prim: Usd.Prim) -> _SolverEntry:
    applied = set(scene_prim.GetAppliedSchemas()) | set(_get_raw_api_schemas(scene_prim))
    tokens = sorted(t for t in applied if t.startswith("NewtonSolver") and t.endswith("API"))
    registered = ", ".join(sorted(_SOLVER_REGISTRY))
    if not tokens:
        raise ValueError(f"No NewtonSolver*API applied to {scene_prim.GetPath()}. Apply exactly one of: {registered}")
    if len(tokens) > 1:
        raise ValueError(
            f"exactly one NewtonSolver*API must be applied to {scene_prim.GetPath()}, found: {', '.join(tokens)}"
        )
    entry = _SOLVER_REGISTRY.get(tokens[0])
    if entry is None:
        raise ValueError(f"Unknown solver schema {tokens[0]}. Registered: {registered}")
    return entry


@dataclass
class Simulation:
    """A stage-driven simulation: the USD stage is the spec, this owns the derived runtime objects.

    All fields are public; external control is opt-in by writing into
    ``control`` or ``state.body_f`` between steps.
    """

    stage: Any
    model: newton.Model
    solvers: list
    state: newton.State
    control: newton.Control
    contacts: newton.Contacts | None
    dt: float
    collision_interval: int
    usd_info: dict
    time: float = 0.0
    step_count: int = 0
    _graphs: tuple | None = field(default=None, repr=False)

    @property
    def solver(self):
        return self.solvers[0]


def load_usd(source: str | Usd.Stage, *, requires_grad: bool = False, use_graph: bool | None = None) -> Simulation:
    """Load a USD-authored simulation and derive the Newton runtime objects from it.

    Args:
        source: Path to a USD file or an open ``Usd.Stage``. The stage is read, never written.
        requires_grad: Finalize the model with gradient arrays. ``step()`` remains
            forward-only; gradient workflows drive ``sim.solver`` directly.
        use_graph: Force graph capture on/off; ``None`` follows the examples' criteria.
    """
    from pxr import Usd

    stage = Usd.Stage.Open(source) if isinstance(source, str) else source
    scene_prim = _find_physics_scene(stage)
    entry = _resolve_solver_entry(scene_prim)

    builder = newton.ModelBuilder()
    entry.cls.register_custom_attributes(builder)
    usd_info = builder.add_usd(
        stage,
        schema_resolvers=[
            newton.usd.SchemaResolverNewton(),
            newton.usd.SchemaResolverPhysx(),
            newton.usd.SchemaResolverMjc(),
        ],
        apply_up_axis_from_stage=True,
    )
    if entry.cls is SolverVBD:
        builder.color()
    model = builder.finalize(requires_grad=requires_grad)

    scene_attrs = usd_info["scene_attributes"]
    params = {
        kwarg: scene_attrs[attr] for attr, kwarg in entry.param_attrs.items() if scene_attrs.get(attr) is not None
    }
    solver = entry.cls(model, **params)

    state = model.state()
    control = model.control()
    contacts = model.contacts() if entry.uses_newton_contacts else None
    if model.joint_count:
        newton.eval_fk(model, state.joint_q, state.joint_qd, state)

    sim = Simulation(
        stage=stage,
        model=model,
        solvers=[solver],
        state=state,
        control=control,
        contacts=contacts,
        dt=usd_info["physics_dt"],
        collision_interval=int(get_attribute(scene_prim, "newton:collisionInterval", 1)),
        usd_info=usd_info,
    )
    _maybe_capture(sim, use_graph)
    return sim


def _maybe_capture(sim: Simulation, use_graph: bool | None) -> None:
    pass  # implemented in Task 5
