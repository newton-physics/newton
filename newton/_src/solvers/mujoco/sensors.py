# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""MuJoCo metadata for imported MJCF site touch and body/site frame sensors."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import numpy as np
import warp as wp

from ...sim.model import Model

if TYPE_CHECKING:
    from ...sim.builder import ModelBuilder

FRAME_SENSOR_TYPES = frozenset(
    {
        "framepos",
        "framequat",
        "framexaxis",
        "frameyaxis",
        "framezaxis",
        "framelinvel",
        "frameangvel",
        "framelinacc",
        "frameangacc",
    }
)
"""MJCF frame sensor elements that can be imported."""

SENSOR_OBJECT_TYPES = frozenset({"body", "xbody", "site"})
"""MJCF object and reference types that imported sensors may name."""

INTERPOLATION_ORDERS = {"zoh": 0, "linear": 1, "cubic": 2}
"""MJCF sensor ``interp`` keywords and their MuJoCo interpolation orders."""


def _register_sensor_attributes(builder: ModelBuilder) -> None:
    """Register the ``mujoco:sensor`` frequency and the per-sensor attributes (idempotent)."""
    frequency = "mujoco:sensor"
    builder.add_custom_frequency(type(builder).CustomFrequency(name="sensor", namespace="mujoco"))
    for name, dtype, default, references in (
        ("type", str, "", None),
        ("attributes", str, "{}", None),
        ("objtype", str, "body", None),
        ("body", wp.int32, -1, "body"),
        ("site", wp.int32, -1, "shape"),
        ("reftype", str, "", None),
        ("refbody", wp.int32, -1, "body"),
        ("refsite", wp.int32, -1, "shape"),
        ("world", wp.int32, -1, "world"),
    ):
        builder.add_custom_attribute(
            type(builder).CustomAttribute(
                name=f"sensor_{name}",
                frequency=frequency,
                assignment=Model.AttributeAssignment.MODEL,
                dtype=dtype,
                default=default,
                references=references,
                namespace="mujoco",
            )
        )


def _sensor_spec_kwargs(attributes: dict[str, str]) -> dict[str, Any]:
    """Convert authored MJCF sensor attributes to ``MjSpec.add_sensor`` keyword arguments."""
    kwargs: dict[str, Any] = {key: attributes[key] for key in ("name",) if key in attributes}
    for key in ("cutoff", "noise", "delay"):
        if key in attributes:
            kwargs[key] = float(attributes[key])
    if "nsample" in attributes:
        kwargs["nsample"] = int(attributes["nsample"])
    if "interp" in attributes:
        kwargs["interp"] = INTERPOLATION_ORDERS[attributes["interp"]]
    if "interval" in attributes:
        # MJCF may omit the phase or the whole interval, which default to zero.
        interval = [float(value) for value in attributes["interval"].split()]
        if len(interval) > 2:
            raise ValueError(
                f"Sensor {attributes.get('name', '')!r} has invalid interval {attributes['interval']!r}; "
                "expected a period and an optional phase."
            )
        kwargs["interval"] = [*interval, 0.0, 0.0][:2]
    if "user" in attributes:
        kwargs["userdata"] = [float(value) for value in attributes["user"].split()]
    return kwargs


def _exported_sensor_rows(model: Model, first_world: int, separate_worlds: bool) -> list[int]:
    """Return the sensor rows SolverMuJoCo exports: all rows, or only the template world's with separate worlds."""
    count = model.custom_frequency_counts.get("mujoco:sensor", 0)
    if getattr(model, "mujoco", None) is None or not count:
        return []
    world = model.mujoco.sensor_world.numpy()
    return [row for row in range(count) if not separate_worlds or int(world[row]) in (-1, first_world)]


def _sensor_required_sites(model: Model, first_world: int, separate_worlds: bool) -> set[int]:
    """Return the site shapes that exported sensors name, which SolverMuJoCo keeps when ``include_sites=False``."""
    rows = _exported_sensor_rows(model, first_world, separate_worlds)
    if not rows:
        return set()
    sites = np.concatenate([model.mujoco.sensor_site.numpy()[rows], model.mujoco.sensor_refsite.numpy()[rows]])
    return set(sites[sites >= 0].tolist())


def _validate_sensor_worlds(model: Model, world_count: int) -> None:
    """Require every world to have the template world's sensors, since only those are exported.

    Raises:
        ValueError: If the worlds' imported sensors differ in count, type, attributes, or targets.
    """
    count = model.custom_frequency_counts.get("mujoco:sensor", 0)
    namespace = getattr(model, "mujoco", None)
    if namespace is None or not count or world_count <= 1:
        return
    world = namespace.sensor_world.numpy()
    rows = np.flatnonzero(world >= 0)
    counts = np.bincount(world[rows], minlength=world_count)
    if np.any(counts != counts[0]):
        mismatch = int(np.flatnonzero(counts != counts[0])[0])
        raise ValueError(
            "SolverMuJoCo requires homogeneous worlds. "
            f"World 0 has {counts[0]} imported sensors, but world {mismatch} has {counts[mismatch]}."
        )

    def local_index(entity_world: np.ndarray, indices: np.ndarray) -> np.ndarray:
        # Index of each entity within its world, so equivalent rows compare equal across worlds.
        order = np.argsort(entity_world, kind="stable")
        local = np.empty(len(entity_world), dtype=np.int64)
        local[order] = np.arange(len(entity_world)) - np.searchsorted(entity_world[order], entity_world[order])
        result = np.full(len(indices), -1, dtype=np.int64)
        result[indices >= 0] = local[indices[indices >= 0]]
        return result

    rows = rows[np.argsort(world[rows], kind="stable")].reshape(world_count, counts[0])
    columns = [
        np.asarray(getattr(namespace, f"sensor_{name}"), dtype=object)[rows]
        for name in ("type", "objtype", "reftype", "attributes")
    ]
    for entity_world, names in ((model.body_world, ("body", "refbody")), (model.shape_world, ("site", "refsite"))):
        columns += [
            local_index(entity_world.numpy(), getattr(namespace, f"sensor_{name}").numpy())[rows] for name in names
        ]
    mismatch = np.any([column != column[0] for column in columns], axis=0)
    if np.any(mismatch):
        raise ValueError(
            "SolverMuJoCo requires homogeneous worlds. "
            f"The imported sensors of world {int(np.argwhere(mismatch)[0][0])} differ from those of world 0."
        )


def _add_imported_sensors(
    spec: Any,
    model: Model,
    first_world: int,
    separate_worlds: bool,
    body_name: Callable[[int], str],
    site_names: dict[int, str],
) -> None:
    """Add the template world's imported sensors to a MuJoCo spec.

    Args:
        spec: MuJoCo ``MjSpec`` being built.
        model: Newton model with ``mujoco:sensor_*`` attributes.
        first_world: Template world mapped to the MuJoCo model.
        separate_worlds: Whether each Newton world is a separate MuJoCo world.
        body_name: Maps a Newton body index to its MuJoCo body name.
        site_names: Maps a Newton site shape index to its MuJoCo site name.
    """
    import mujoco

    rows = _exported_sensor_rows(model, first_world, separate_worlds)
    if not rows:
        return
    namespace = model.mujoco
    values = {name: getattr(namespace, f"sensor_{name}").numpy() for name in ("body", "site", "refbody", "refsite")}

    def object_name(kind: str, body: int, site: int) -> str:
        return site_names[int(site)] if kind == "site" else body_name(int(body))

    used_names: set[str] = set()
    for row in rows:
        kind = namespace.sensor_objtype[row]
        reference = namespace.sensor_reftype[row]
        kwargs = _sensor_spec_kwargs(json.loads(namespace.sensor_attributes[row]))
        if kwargs.get("name"):
            # Copies of one MJCF in a MuJoCo model would repeat sensor names; suffix them like tendon names.
            name, suffix = kwargs["name"], 1
            while kwargs["name"] in used_names:
                kwargs["name"] = f"{name}_{suffix}"
                suffix += 1
            used_names.add(kwargs["name"])
        if reference:
            kwargs["reftype"] = getattr(mujoco.mjtObj, f"mjOBJ_{reference.upper()}")
            kwargs["refname"] = object_name(reference, values["refbody"][row], values["refsite"][row])
        spec.add_sensor(
            type=getattr(mujoco.mjtSensor, f"mjSENS_{namespace.sensor_type[row].upper()}"),
            objtype=getattr(mujoco.mjtObj, f"mjOBJ_{kind.upper()}"),
            objname=object_name(kind, values["body"][row], values["site"][row]),
            **kwargs,
        )
