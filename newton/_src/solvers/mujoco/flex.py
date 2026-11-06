# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Programmatic authoring helpers for MuJoCo flexes."""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from typing import Any

import numpy as np
import warp as wp

from ...core._warnings import _external_warning_stacklevel
from ...core.types import vec5
from ...exceptions import NewtonWarning
from ...sim import Model, ModelBuilder
from ._authoring import _ensure_mujoco_attributes, _prepare_custom_frequency_row, _vector

ELASTIC2D = {"none": 0, "bend": 1, "stretch": 2, "both": 3}
SELFCOLLIDE = {"none": 0, "narrow": 1, "bvh": 2, "sap": 3, "auto": 4}

# Per-flex attributes as (dtype, default). A negative ``young``, ``edgestiffness``,
# ``edgedamping``, ``radius``, or sliding ``friction`` means "derive from the Newton
# elements and particles", which is also how elements without an authored flex convert.
FLEX_ATTRIBUTES = {
    "label": (str, ""),
    "young": (wp.float32, -1.0),
    "poisson": (wp.float32, 0.0),
    "damping": (wp.float32, 0.0),
    "edgestiffness": (wp.float32, -1.0),
    "edgedamping": (wp.float32, -1.0),
    "thickness": (wp.float32, 1.0),
    "elastic2d": (wp.int32, ELASTIC2D["stretch"]),
    "radius": (wp.float32, -1.0),
    "selfcollide": (wp.int32, SELFCOLLIDE["none"]),
    "friction": (wp.vec3, wp.vec3(-1.0, 0.005, 0.0001)),
    "margin": (wp.float32, 0.0),
    "gap": (wp.float32, 0.0),
    "solref": (wp.vec2, wp.vec2(0.02, 1.0)),
    "solimp": (vec5, vec5(0.9, 0.95, 0.001, 0.5, 2.0)),
}

# Newton element source -> (flex dim, per-element attribute naming its ``mujoco:flex`` row, frequency, indices).
ELEMENTS = {
    "tetrahedra": (3, "tet_flex", Model.AttributeFrequency.TETRAHEDRON, "tet_indices"),
    "triangles": (2, "tri_flex", Model.AttributeFrequency.TRIANGLE, "tri_indices"),
    "springs": (1, "spring_flex", Model.AttributeFrequency.SPRING, "spring_indices"),
}


def _register_flex_attributes(builder: ModelBuilder) -> None:
    """Declare the ``mujoco:flex`` rows and the per-element flex assignments on ``builder``."""
    builder.add_custom_frequency(ModelBuilder.CustomFrequency(name="flex", namespace="mujoco"))
    for name, (dtype, default) in FLEX_ATTRIBUTES.items():
        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name=f"flex_{name}", frequency="mujoco:flex", dtype=dtype, default=default, namespace="mujoco"
            )
        )
    for _, attribute, frequency, _ in ELEMENTS.values():
        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name=attribute,
                frequency=frequency,
                dtype=wp.int32,
                default=-1,
                namespace="mujoco",
                references="mujoco:flex",
            )
        )


def _choice(value: str | int, options: dict[str, int], name: str) -> int:
    if isinstance(value, str):
        if value not in options:
            raise ValueError(f"{name} must be one of {', '.join(options)}, got {value!r}.")
        return options[value]
    if int(value) not in options.values():
        raise ValueError(f"{name} must be in [0, {len(options)}), got {value!r}.")
    return int(value)


def add_flex(
    builder: ModelBuilder,
    *,
    tetrahedra: Sequence[int] | None = None,
    triangles: Sequence[int] | None = None,
    springs: Sequence[int] | None = None,
    label: str | None = None,
    young: float | None = None,
    poisson: float | None = None,
    damping: float = 0.0,
    edgestiffness: float | None = None,
    edgedamping: float | None = None,
    thickness: float | None = None,
    elastic2d: str | int | None = None,
    radius: float | None = None,
    selfcollide: str | int = "none",
    friction: Sequence[float] | None = None,
    margin: float = 0.0,
    gap: float = 0.0,
    solref: Sequence[float] = (0.02, 1.0),
    solimp: Sequence[float] = (0.9, 0.95, 0.001, 0.5, 2.0),
    custom_attributes: dict[str, Any] | None = None,
) -> int:
    """Add a MuJoCo flex over existing Newton tetrahedra, triangles, or springs.

    Exactly one element source must be given. Its elements, and their particles,
    must belong to :attr:`~newton.ModelBuilder.current_world`. The flex is only
    simulated by :class:`~newton.solvers.SolverMuJoCo` with ``use_flex=True``;
    elements without an authored flex are grouped into one flex per dimension.

    Args:
        builder: Model builder receiving the flex.
        tetrahedra: Newton tetrahedron indices; creates a 3D flex.
        triangles: Newton triangle indices; creates a 2D flex.
        springs: Newton spring indices; creates a 1D flex.
        label: Optional flex label.
        young: Young's modulus [Pa]. ``None`` derives it and ``poisson`` from the
            elements' Lamé parameters (springs have none, so 1D flexes stay stretch-only).
        poisson: Poisson ratio; requires ``young``.
        damping: Rayleigh damping coefficient [s].
        edgestiffness: 1D edge stiffness [N/m]. ``None`` uses the spring stiffness.
        edgedamping: 1D edge damping [N·s/m]. ``None`` uses the spring damping.
        thickness: 2D shell thickness [m], defaults to 1 so that Newton's
            per-area triangle stiffness carries over.
        elastic2d: 2D elasticity: ``"none"``, ``"bend"``, ``"stretch"`` (default), or ``"both"``.
        radius: Collision radius [m]. ``None`` uses the largest particle radius.
        selfcollide: ``"none"`` (default), ``"narrow"``, ``"bvh"``, ``"sap"``, or ``"auto"``.
        friction: Sliding, torsional, and rolling friction. ``None`` uses
            ``particle_mu`` with MuJoCo's default torsional and rolling friction.
        margin: Contact inclusion margin [m].
        gap: Inactive contact gap [m].
        solref: Contact constraint reference parameters.
        solimp: Contact constraint impedance parameters.
        custom_attributes: Additional registered ``mujoco:flex`` attributes.

    Returns:
        The ``mujoco:flex`` row index.
    """
    _ensure_mujoco_attributes(builder, "mujoco:flex_young")
    sources = {
        k: v for k, v in (("tetrahedra", tetrahedra), ("triangles", triangles), ("springs", springs)) if v is not None
    }
    if len(sources) != 1:
        raise ValueError("add_flex requires exactly one of tetrahedra, triangles, or springs.")
    ((source, indices),) = sources.items()
    dim, attribute, _, field = ELEMENTS[source]
    indices = np.asarray(indices, dtype=np.int64).ravel()
    if indices.size == 0:
        raise ValueError(f"add_flex requires at least one element in {source}.")
    if dim != 2 and (thickness is not None or elastic2d is not None):
        raise ValueError("thickness and elastic2d only apply to triangles.")
    if young is None and poisson is not None:
        raise ValueError("poisson requires young.")
    if dim != 1 and (edgestiffness is not None or edgedamping is not None):
        raise ValueError("edgestiffness and edgedamping only apply to springs.")

    elements = np.asarray(getattr(builder, field), dtype=np.int64).reshape(-1, dim + 1)
    if indices.min() < 0 or indices.max() >= len(elements):
        raise IndexError(f"{source} indices must be in [0, {len(elements)}).")
    owner = builder.custom_attributes[f"mujoco:{attribute}"].values
    taken = [int(i) for i in indices if owner.get(int(i), -1) >= 0]
    if taken:
        raise ValueError(f"{source} {taken[:5]} already belong to a flex.")
    worlds = np.unique(np.asarray(builder.particle_world)[elements[indices]])
    if worlds.tolist() != [builder.current_world]:
        raise ValueError(
            f"{source} use particles from worlds {worlds.tolist()}, but the flex is added to world {builder.current_world}."
        )

    def derived(value: float | None) -> float:
        return -1.0 if value is None else float(value)

    values = {
        "mujoco:flex_label": label or "",
        "mujoco:flex_young": derived(young),
        "mujoco:flex_poisson": 0.0 if poisson is None else float(poisson),
        "mujoco:flex_damping": float(damping),
        "mujoco:flex_edgestiffness": derived(edgestiffness),
        "mujoco:flex_edgedamping": derived(edgedamping),
        "mujoco:flex_thickness": 1.0 if thickness is None else float(thickness),
        "mujoco:flex_elastic2d": _choice("stretch" if elastic2d is None else elastic2d, ELASTIC2D, "elastic2d"),
        "mujoco:flex_radius": derived(radius),
        "mujoco:flex_selfcollide": _choice(selfcollide, SELFCOLLIDE, "selfcollide"),
        "mujoco:flex_friction": _vector(
            builder, "mujoco:flex_friction", (-1.0, 0.005, 0.0001) if friction is None else friction, 3
        ),
        "mujoco:flex_margin": float(margin),
        "mujoco:flex_gap": float(gap),
        "mujoco:flex_solref": _vector(builder, "mujoco:flex_solref", solref, 2),
        "mujoco:flex_solimp": _vector(builder, "mujoco:flex_solimp", solimp, 5),
    }
    row = _prepare_custom_frequency_row(builder, "mujoco:flex", values, custom_attributes)
    flex = builder.add_custom_values(**row)["mujoco:flex_young"]
    for i in indices:
        owner[int(i)] = flex
    return flex


def _uniform(values: np.ndarray, what: str) -> Any:
    """Return the mean of ``values`` along the first axis, warning if they differ, since a flex has one material."""
    if np.ptp(values, axis=0).any():
        warnings.warn(
            f"A MuJoCo flex uses a single {what}; averaging non-uniform Newton values. "
            "Use newton.solvers.mujoco.add_flex to split them into separate flexes.",
            NewtonWarning,
            stacklevel=_external_warning_stacklevel(),
        )
    return values.mean(axis=0)


__all__ = ["add_flex"]
