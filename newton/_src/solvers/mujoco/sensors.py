# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""MuJoCo metadata for imported site touch and body/site frame sensors."""

import json

import warp as wp

from ...sim.model import Model


def _register_sensor_attributes(builder):
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


def _add_imported_sensors(spec, model, first_world, separate_worlds, body_name, sites):
    import mujoco

    namespace = model.mujoco
    count = model.custom_frequency_counts.get("mujoco:sensor", 0)
    if not count:
        return
    values = {
        name: getattr(namespace, f"sensor_{name}").numpy() for name in ("body", "site", "refbody", "refsite", "world")
    }

    def object_name(kind, body, site):
        return sites[int(site)] if kind == "site" else body_name(int(body))

    for row in range(count):
        world = int(values["world"][row])
        if separate_worlds and world not in (-1, first_world):
            continue
        kind = namespace.sensor_objtype[row]
        reference = namespace.sensor_reftype[row]
        attributes = json.loads(namespace.sensor_attributes[row])
        kwargs = {key: attributes[key] for key in ("name",) if key in attributes}
        for key in ("cutoff", "noise", "delay"):
            if key in attributes:
                kwargs[key] = float(attributes[key])
        for key in ("nsample", "interp"):
            if key in attributes:
                kwargs[key] = int(attributes[key])
        for key in ("userdata", "interval"):
            if key in attributes:
                kwargs[key] = list(map(float, attributes[key].split()))
        if reference:
            kwargs["reftype"] = getattr(mujoco.mjtObj, f"mjOBJ_{reference.upper()}")
            kwargs["refname"] = object_name(reference, values["refbody"][row], values["refsite"][row])
        spec.add_sensor(
            type=getattr(mujoco.mjtSensor, f"mjSENS_{namespace.sensor_type[row].upper()}"),
            objtype=getattr(mujoco.mjtObj, f"mjOBJ_{kind.upper()}"),
            objname=object_name(kind, values["body"][row], values["site"][row]),
            **kwargs,
        )
