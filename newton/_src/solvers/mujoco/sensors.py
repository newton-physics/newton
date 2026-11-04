# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""MuJoCo metadata for imported MJCF site touch and body/site frame sensors."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

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
        # MJCF may omit the phase, which defaults to zero.
        interval = [float(value) for value in attributes["interval"].split()]
        kwargs["interval"] = [*interval, 0.0][:2]
    if "user" in attributes:
        kwargs["userdata"] = [float(value) for value in attributes["user"].split()]
    return kwargs


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

    namespace = getattr(model, "mujoco", None)
    count = model.custom_frequency_counts.get("mujoco:sensor", 0)
    if namespace is None or not count:
        return
    values = {
        name: getattr(namespace, f"sensor_{name}").numpy() for name in ("body", "site", "refbody", "refsite", "world")
    }

    def object_name(kind: str, body: int, site: int) -> str:
        return site_names[int(site)] if kind == "site" else body_name(int(body))

    for row in range(count):
        world = int(values["world"][row])
        if separate_worlds and world not in (-1, first_world):
            continue
        kind = namespace.sensor_objtype[row]
        reference = namespace.sensor_reftype[row]
        kwargs = _sensor_spec_kwargs(json.loads(namespace.sensor_attributes[row]))
        if reference:
            kwargs["reftype"] = getattr(mujoco.mjtObj, f"mjOBJ_{reference.upper()}")
            kwargs["refname"] = object_name(reference, values["refbody"][row], values["refsite"][row])
        spec.add_sensor(
            type=getattr(mujoco.mjtSensor, f"mjSENS_{namespace.sensor_type[row].upper()}"),
            objtype=getattr(mujoco.mjtObj, f"mjOBJ_{kind.upper()}"),
            objname=object_name(kind, values["body"][row], values["site"][row]),
            **kwargs,
        )
