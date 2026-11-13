# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Compare fresh MJCF-to-USD conversion with native import on small physical fixtures."""

import argparse
import importlib.metadata
import json
import re
import traceback
from functools import partial
from pathlib import Path

import numpy as np
import warp as wp

import newton
from newton.solvers import SolverMuJoCo
from newton.tests.mjcf_usd_comparison import UsdComparisonImporter


def _fixtures(directory):
    body = '<body name="link"><joint name="joint" axis="0 1 0" {joint}/><geom name="shape" type="box" size=".1 .2 .3" mass="1"/>{child}</body>'

    def xml(*, joint="", child="", actuators="", asset="", extra="", defaults="", world=None):
        world = world if world is not None else body.format(joint=joint, child=child)
        return f'<mujoco model="fixture">{defaults}<asset>{asset}</asset><worldbody>{world}</worldbody><actuator>{actuators}</actuator>{extra}</mujoco>'

    drive = '<position name="drive" joint="joint" kp="20" {}/>'
    hand = '<body name="hand" pos="0 0 .5"><geom name="hand_shape" size=".1" mass="2"/></body>'
    vertices = "0 0 0 .3 0 0 0 .4 0 0 0 .5"
    faces = "0 2 1 0 1 3 0 3 2 1 2 3"
    mesh_body = '<body name="link"><joint name="joint"/><geom name="shape" type="mesh" mesh="mesh"/></body>'
    mesh = directory / "tetra.obj"
    mesh.write_text("v 0 0 0\nv .3 0 0\nv 0 .4 0\nv 0 0 .5\nf 1 3 2\nf 1 2 4\nf 1 4 3\nf 2 3 4\n")
    cases = {
        "explicit_joint": xml(joint='armature=".05" damping=".3" ref="15" range="-30 60"'),
        "explicit_zero_defaults": xml(defaults='<default><joint armature="0" damping="0"/></default>'),
        "position_kv": xml(actuators=drive.format('kv="3"')),
        "position_dampratio": xml(actuators=drive.format('dampratio=".7"')),
        "dcmotor": xml(actuators='<dcmotor name="drive" joint="joint" motorconst=".2" resistance="2"/>'),
        "paired_damping": xml(actuators=drive.format('kv="3"') + '<velocity name="velocity" joint="joint" kv="5"/>'),
        "inherited_range": xml(joint='type="slide" ref=".2" range="-1 2"', actuators=drive.format('inheritrange=".8"')),
        "springdamper": xml(joint='springdamper=".2 .7"'),
        "contact_pair": xml(
            world=body.format(joint="", child="") + '<geom name="ground" type="plane" size="1 1 .1"/>',
            extra='<contact><pair geom1="ground" geom2="shape" friction=".4 .4 .03 .01 .01"/></contact>',
        ),
        "collision_masks": xml(
            world="".join(
                f'<body name="{name}"><joint name="{name}_joint"/><geom name="{name}_shape" size=".1" mass="1" contype="{ct}" conaffinity="{ca}"/></body>'
                for name, ct, ca in (("one", 2, 4), ("two", 8, 16), ("three", 4, 2))
            )
        ),
        "inline_mesh": xml(asset=f'<mesh name="mesh" vertex="{vertices}" face="{faces}"/>', world=mesh_body),
        "obj_mesh": xml(asset=f'<mesh name="mesh" file="{mesh}"/>', world=mesh_body),
        "ellipsoid": xml(world=body.format(joint="", child="").replace('type="box"', 'type="ellipsoid"')),
        "actuator_history": xml(actuators=drive.format('kv="3" delay=".004" nsample="4" interp="linear"')),
        "actuator_group": xml(actuators=drive.format('kv="3" group="2"'), extra='<option actuatorgroupdisable="2"/>'),
        "tendon": xml(
            extra='<tendon><fixed name="tendon" stiffness="2" damping=".3"><joint joint="joint" coef="1"/></fixed></tendon>'
        ),
        "composed_dampratio": xml(child=hand, actuators=drive.format('dampratio=".7"')),
    }
    return cases, xml(actuators=drive.format('dampratio=".7"')), xml(world=hand)


def _snapshot(model, body_names, geom_names):
    import mujoco

    def canonical(name, candidates):
        variants = (name, re.sub(r"_\d+$", "", name))
        matches = [
            candidate
            for candidate in candidates
            if any(value == candidate or value.endswith(("_" + candidate, "/" + candidate)) for value in variants)
        ]
        if not matches:
            raise ValueError(f"Cannot map compiled name {name!r} to fixture names {candidates}")
        return max(matches, key=len)

    bodies = {}
    for i in range(1, model.nbody):
        rotation = np.empty(9)
        mujoco.mju_quat2Mat(rotation, model.body_iquat[i])
        rotation = rotation.reshape(3, 3)
        parent = int(model.body_parentid[i])
        parent = body_names.index(canonical(model.body(parent).name, body_names)) if parent else -1
        pose_rotation = np.empty(9)
        mujoco.mju_quat2Mat(pose_rotation, model.body_quat[i])
        bodies[canonical(model.body(i).name, body_names)] = np.concatenate(
            (
                [model.body_mass[i], parent],
                model.body_ipos[i],
                (rotation @ np.diag(model.body_inertia[i]) @ rotation.T).ravel(),
                model.body_pos[i],
                pose_rotation,
            )
        )
    fields = (
        "jnt_type",
        "jnt_pos",
        "jnt_axis",
        "jnt_limited",
        "jnt_range",
        "jnt_stiffness",
        "dof_damping",
        "dof_armature",
        "dof_frictionloss",
        "geom_type",
        "geom_size",
        "geom_friction",
        "pair_dim",
        "pair_friction",
        "tendon_stiffness",
        "tendon_damping",
        "actuator_gainprm",
        "actuator_biasprm",
        "actuator_gear",
        "actuator_ctrlrange",
        "actuator_group",
        "actuator_delay",
        "actuator_history",
    )
    snapshot = {f"body:{name}": value for name, value in bodies.items()}
    snapshot.update({name: np.asarray(getattr(model, name)).copy() for name in fields})
    snapshot["jnt_range"][model.jnt_limited == 0] = 0
    # Unused size slots and infinite-plane display dimensions do not affect physics.
    for i, kind in enumerate(map(int, model.geom_type)):
        if kind in (mujoco.mjtGeom.mjGEOM_PLANE, mujoco.mjtGeom.mjGEOM_MESH):
            snapshot["geom_size"][i] = 0
        elif kind == mujoco.mjtGeom.mjGEOM_SPHERE:
            snapshot["geom_size"][i, 1:] = 0
        elif kind in (mujoco.mjtGeom.mjGEOM_CAPSULE, mujoco.mjtGeom.mjGEOM_CYLINDER):
            snapshot["geom_size"][i, 2:] = 0
    geom_ids = {i: geom_names.index(canonical(model.geom(i).name, geom_names)) for i in range(model.ngeom)}
    order = sorted(geom_ids, key=geom_ids.get)
    for field in ("geom_type", "geom_size", "geom_friction"):
        snapshot[field] = snapshot[field][order]
    snapshot["disableactuator"] = np.asarray(model.opt.disableactuator)
    data = mujoco.MjData(model)
    data.qvel[:] = 0.1
    data.ctrl[:] = 0.15
    flags = model.opt.disableflags
    try:
        model.opt.disableflags &= ~int(mujoco.mjtDisableBit.mjDSBL_CONTACT)
        mujoco.mj_forward(model, data)
    finally:
        model.opt.disableflags = flags
    snapshot["contact_pairs"] = np.asarray(
        sorted({tuple(sorted((geom_ids[int(c.geom1)], geom_ids[int(c.geom2)]))) for c in data.contact}), dtype=int
    ).reshape(-1, 2)
    for field in ("qfrc_bias", "qfrc_passive", "qfrc_actuator"):
        snapshot[field] = getattr(data, field).copy()
    return snapshot


def _differences(actual, expected):
    differences = {}
    for name in sorted(actual.keys() | expected.keys()):
        if name not in actual or name not in expected:
            differences[name] = "missing in " + ("actual" if name not in actual else "reference")
            continue
        a, b = np.asarray(actual[name]), np.asarray(expected[name])
        if a.shape != b.shape:
            differences[name] = {"actual_shape": a.shape, "reference_shape": b.shape}
        elif not np.allclose(a, b, rtol=3e-5, atol=2e-6):
            differences[name] = {"actual": a.tolist(), "reference": b.tolist()}
    return differences


def main():
    import mujoco

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--artifacts", required=True)
    args = parser.parse_args()
    directory = Path(args.artifacts)
    directory.mkdir(parents=True, exist_ok=True)
    cases, arm, hand = _fixtures(directory)
    report = {
        "versions": {
            name: importlib.metadata.version(name)
            for name in (
                "mujoco",
                "mujoco-warp",
                "warp-lang",
                "usd-exchange",
                "newton-usd-schemas",
                "mujoco-usd-converter",
            )
        },
        "device": str(wp.get_device()),
        "cases": {},
    }
    for name, xml in cases.items():
        case = report["cases"][name] = {}
        reference_model = mujoco.MjModel.from_xml_string(xml)
        body_names = [reference_model.body(i).name for i in range(1, reference_model.nbody)]
        geom_names = sorted(reference_model.geom(i).name for i in range(reference_model.ngeom))
        reference = _snapshot(reference_model, body_names, geom_names)
        snapshots = {}
        for route in ("native", "usd"):
            phase = "import"
            adapter = UsdComparisonImporter(directory / name)
            try:
                builder = newton.ModelBuilder()
                builder.default_joint_cfg.armature = 0.4 if name == "explicit_zero_defaults" else 0
                builder.default_joint_cfg.damping = 0.8 if name == "explicit_zero_defaults" else 0
                method = builder.add_mjcf_native if route == "native" else partial(adapter, builder)
                if name == "composed_dampratio":
                    method(arm)
                    method(hand, parent_body=0)
                else:
                    method(xml)
                phase = "finalize"
                model = builder.finalize()
                before = model.joint_target_kd.numpy().tolist()
                phase = "solver"
                (directory / name).mkdir(parents=True, exist_ok=True)
                solver = SolverMuJoCo(
                    model,
                    disable_contacts=True,
                    kinematic_switchable_roots=False,
                    save_to_mjcf=str(directory / name / f"{route}.xml"),
                )
                snapshots[route] = _snapshot(solver.mj_model, body_names, geom_names)
                snapshots[route].update(
                    {
                        "core:target_mode": model.joint_target_mode.numpy(),
                        "core:target_ke": model.joint_target_ke.numpy(),
                        "core:target_kd": model.joint_target_kd.numpy(),
                        "core:ctrl_source": model.mujoco.ctrl_source.numpy()
                        if model.custom_frequency_counts.get("mujoco:actuator", 0)
                        else np.empty(0),
                    }
                )
                physical = {key: value for key, value in snapshots[route].items() if not key.startswith("core:")}
                case[route] = {
                    "status": "completed",
                    "native_reference_differences": _differences(physical, reference),
                    "kd_before_solver": before,
                    "kd_after_solver": model.joint_target_kd.numpy().tolist(),
                }
                if route == "usd":
                    case[route]["conversions"] = adapter.records
            except Exception:
                case[route] = {
                    "status": "error",
                    "phase": phase,
                    "traceback": traceback.format_exc(),
                    "conversions": adapter.records if route == "usd" else [],
                }
        if "usd" in snapshots and "native" in snapshots:
            case["usd_vs_native_importer"] = _differences(snapshots["usd"], snapshots["native"])
        print(name, {route: case[route]["status"] for route in ("native", "usd")}, flush=True)
        Path(args.output).write_text(json.dumps(report, indent=2))
    return int(any(case["native"]["status"] != "completed" for case in report["cases"].values()))


if __name__ == "__main__":
    raise SystemExit(main())
