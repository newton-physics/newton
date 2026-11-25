# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Build declared scenes using the unmodified public Newton VBD API."""

import importlib

from .catalog import get_case


def build_scene(case_id, device=None):
    """Build one scene with its declared inputs and one-substep interface.

    The caller owns the physical timeline, observation, and rendering. This
    function does not capture a graph or intercept solver or collision calls.
    """
    case = get_case(case_id)
    name = case["scene_id"]
    if name == "r2_franka_duck":
        raise NotImplementedError(
            "The archived articulated duck recipe requires a second dynamics solver. "
            "This benchmark calls only ordinary VBD; no replacement recipe is supplied."
        )
    if case["runner"] == "h1":
        from .h1_scenes import build_scene as build_h1  # noqa: PLC0415

        return build_h1(case_id, device=device)
    if case["runner"] == "soft":
        module = "soft_scenes"
    elif case["runner"] == "poker":
        module = "poker_cards_scene"
    elif case["runner"] == "cable":
        module = "cable_scenes"
    elif name.startswith("r5_"):
        module = "cloth_robot_scenes"
    elif name.startswith("r6_"):
        module = "solid_robot_scenes"
    elif name.startswith("m3_"):
        module = "bag_robot_scenes"
    elif name == "m2_franka_pouch":
        module = "pouch_scene"
    elif name in ("m2_franka_cable", "m2_allegro_cable"):
        module = "robot_cable_scene"
    elif name.startswith("c9_"):
        module = "cable_robot_retrofit"
    elif name == "r3_allegro_ground_sphere":
        module = "hand_scene"
    else:
        raise ValueError(f"No scene builder is declared for {case_id}")
    factory = importlib.import_module(f".{module}", __package__).build_scene
    scene = factory(name, device=device, **case["kwargs"])
    scene.case_id = case_id
    scene.protocol = case
    scene.source_parameters = dict(case["kwargs"])
    if abs(scene.frame_dt / scene.substeps - case["substep_dt_s"]) > 1e-12:
        raise ValueError(f"Scene {case_id} does not match its declared physical step")
    if scene.substeps > 10:
        raise ValueError("Benchmark scenes support at most ten substeps per frame")
    return scene
