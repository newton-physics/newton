# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Stage-driven simulation runtime: derive a Newton simulation from USD."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from newton._src.solvers import SolverFeatherstone, SolverMuJoCo, SolverSemiImplicit, SolverVBD, SolverXPBD
from newton._src.usd.utils import _get_raw_api_schemas

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


def load_usd(source: str | Usd.Stage, *, requires_grad: bool = False, use_graph: bool | None = None):
    from pxr import Usd

    stage = Usd.Stage.Open(source) if isinstance(source, str) else source
    scene_prim = _find_physics_scene(stage)
    _resolve_solver_entry(scene_prim)
    raise NotImplementedError  # completed in Task 3
