#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Build static web reports for VBD cable bend/twist verification.

The generated pages are intentionally plain static HTML so they can be copied
directly to the ``gh-pages`` branch under ``reports/``.  Each visual validation
case gets one focused report with:

* a Newton-rendered OpenGL video,
* scalar test results from ``metrics.json``,
* the plot(s) that support the acceptance criterion,
* an explicit note about why the case is not redundant with the others.
"""

from __future__ import annotations

import argparse
import html
import importlib
import json
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont  # noqa: TID253

REPO_ROOT = Path(__file__).resolve().parents[1]
REPORTS_ROOT = REPO_ROOT / "reports"
FIGURE_ROOT = REPO_ROOT / "docs" / "design" / "figures" / "cable_bend_twist"
CABLE_RENDER_COLOR = (0.86, 0.90, 0.94)

FIGURE_CAPTIONS = {
    "analytic_bend_focus.png": "Bend centerline, angle error, and RMS history against exact discrete arcs.",
    "analytic_twist_focus.png": "Twist profile, linearity error, and transverse leakage for pure torsion.",
    "bend_stiffness_sweep.png": "Tip deflection and deflection times stiffness for the loaded cantilever sweep.",
    "torsion_law_transducer.png": (
        "Material-mapped pure torsion: twist profile against phi(x) = theta x / L and transverse bend leakage."
    ),
    "torsion_material_time.png": (
        "During the ramp and hold: commanded versus simulated tip twist, twist-profile error, and bend leakage."
    ),
    "torsion_material_scaling.png": (
        "Material-property scaling sweep: GJ/h from E, nu, r, h, and explicit G compared with "
        "the kernel torque response divided by the imposed twist angle."
    ),
    "twist_transfer_profiles.png": (
        "Measured twist profiles for straight, sharp-kink, and smooth-curve routes. "
        "Only the straight path has the analytical reference shown here."
    ),
    "localized_buckling.png": (
        "DER Figure 7-inspired localized buckling prototype: final Newton centerline "
        "and tangent-envelope diagnostic across three segment counts."
    ),
    "localized_buckling_dynamic.png": (
        "Dynamic reproduction: final Newton centerline plus tangent-envelope and material-twist transfer diagnostics."
    ),
    "localized_buckling_quasistatic.png": (
        "Quasistatic reproduction: analytical localized-buckling branch initialization "
        "followed by a no-inertia Newton cable-energy solve."
    ),
    "localized_buckling_quasistatic_convergence.png": (
        "Quasistatic convergence: no-inertia Newton cable-energy solves with real-valued imposed material twist "
        "across multiple resolutions."
    ),
    "mechanical_laws.png": ("Cantilever bending moment law and discrete-to-continuum bend convergence."),
    "twisted_ring_writhe.png": (
        "Control-vs-twisted ring checks: best-fit-plane coplanarity and visible best-fit-plane span."
    ),
    "michell_threshold_sweep.png": (
        "Michell/Zajac threshold sweep: coplanarity and out-of-plane span versus total twist divided "
        "by the analytical critical twist."
    ),
    "michell_threshold_curve.png": "Michell threshold curve and Newton sweep summary.",
    "michell_paper_criterion_grid.png": "Michell/Zajac paper-criterion ring grid.",
    "michell_paper_criterion_curve.png": "Paper-criterion curve compared with Newton sweep.",
    "michell_figure8_grid.png": "Figure-8 shaped ring attempt grid.",
    "der_reproductions.png": "DER-inspired summary across torsion, routed twist, and Michell threshold checks.",
    "verification_coverage.png": "Coverage matrix for the bend/twist verification suite.",
    "dahl_hysteresis.png": (
        "Standard hysteresis diagnostics: load/response loops, time histories, and residual memory summaries "
        "for bend force-deflection and twist torque-angle response."
    ),
}


@dataclass(frozen=True)
class CameraSpec:
    pos: tuple[float, float, float]
    target: tuple[float, float, float]
    pitch: float = 0.0
    yaw: float = 0.0
    fov: float = 45.0


@dataclass(frozen=True)
class ViewPanel:
    label: str
    camera: CameraSpec
    show_joints: bool | None = None
    joint_scale: float | None = None


@dataclass(frozen=True)
class VideoVariant:
    suffix: str
    title: str
    caption: str
    camera: CameraSpec
    example_mode: str | None = None
    show_joints: bool | None = None
    joint_scale: float | None = None
    view_panels: tuple[ViewPanel, ...] = ()


@dataclass(frozen=True)
class FigureGroup:
    title: str
    body: str
    figures: tuple[str, ...]


@dataclass(frozen=True)
class ReportSpec:
    slug: str
    title: str
    subtitle: str
    alias: str
    module: str
    camera: CameraSpec
    steps: int
    stride: int
    fps: int
    figures: tuple[str, ...]
    setup: str
    result: str
    proves: tuple[str, ...]
    nonredundant: str
    visual_reference: tuple[str, ...]
    failure_cues: tuple[str, ...]
    numerical_setup: tuple[tuple[str, str], ...] = ()
    reproduce_command: str | None = None
    video_summary: str = ""
    video_metric_notes: tuple[tuple[str, str], ...] = ()
    figure_groups: tuple[FigureGroup, ...] = ()
    target_heading: str = "Verification Target"
    checks_heading: str = "Acceptance Criteria"
    show_checks: bool = True
    show_joints: bool = False
    joint_scale: float = 1.5
    show_main_video: bool = True
    extra_videos: tuple[VideoVariant, ...] = ()
    uniform_cable_color: bool = True
    preserve_shape_color_prefixes: tuple[str, ...] = ()


REPORT_SPECS: tuple[ReportSpec, ...] = (
    ReportSpec(
        slug="cable_bend_twist_analytic",
        title="Cable Bend/Twist Analytic Validation",
        subtitle="Exact discrete bend arcs and pure twist profiles checked against analytic targets.",
        alias="cable_bend_twist_analytic",
        module="newton.examples.vbd.example_cable_bend_twist_analytic",
        camera=CameraSpec((2.15, -4.20, 1.62), (0.68, 0.0, 0.42), fov=38.0),
        steps=390,
        stride=2,
        fps=30,
        figures=(),
        setup=(
            "This test isolates the bend/twist split in the simplest setting with exact "
            "answers. Root and tip bodies prescribe either a constant-curvature bend arc "
            "or a pure tangent-axis twist; VBD solves the interior bodies and must put "
            "the deformation into the matching subspace only."
        ),
        result=(
            "The solved centerlines and material frames overlap the analytic references. "
            "Bend angle error stays near 0.001 degrees, bend shape error stays near "
            "0.0001% of length, and pure twist creates almost no transverse motion."
        ),
        proves=(
            "Constant-curvature bend reaches the exact discrete arc",
            "Pure twist stays straight and follows the exact linear twist profile",
            "Bend and twist stay isolated in full SolverVBD stepping",
        ),
        nonredundant=(
            "This is the baseline accuracy gate for the split. Later reports add loads, "
            "history, routed geometry, and ring instabilities."
        ),
        visual_reference=(
            "Blue/green bend overlays coincide for the 30-, 60-, and 90-degree rows",
            "Orange/cyan twist ticks stay aligned in the 150-degree visual stress case",
            "The twist centerline remains straight while the material frame rotates",
        ),
        failure_cues=(
            "Visible gaps open between the blue and green bend overlays",
            "Orange twist ticks roll ahead of or behind the cyan reference",
            "The twist row bows, shifts sideways, or loses straightness",
        ),
        numerical_setup=(
            ("Solver fps", "60 Hz"),
            ("Solver timestep", "1/600 s (10 substeps per frame)"),
            ("VBD iterations", "20 per substep"),
            ("Segments / radius", "14 segments, h = 0.10 m, r = 0.010 m"),
            ("Stretch stiffness", "1.0e6 N/m"),
            ("Bend rows", "bend = 400 N m, twist = 1000 N m"),
            ("Twist rows", "bend = 2000 N m, twist = 120 N m"),
            ("Targets", "30, 60, 90 deg; visual twist stress = 150 deg"),
        ),
        video_summary=(
            "The videos are visual checks for the two exact boundary-value solutions. "
            "Bend focus compares the VBD centerline to the exact discrete arc. Twist "
            "inspection compares simulated material-frame phase to exact linear phase "
            "in a deliberately high 150-degree pure-twist case."
        ),
        figure_groups=(
            FigureGroup(
                title="Exact Bend Arc",
                body=(
                    "The left panel overlays the 30-, 60-, and 90-degree exact arcs with the "
                    "settled VBD centerlines. The time plots report the maximum joint-angle "
                    "and centerline RMS errors during the ramp and hold."
                ),
                figures=("analytic_bend_focus.png",),
            ),
            FigureGroup(
                title="Pure Twist Distribution",
                body=(
                    "The left panel overlays the 30-, 60-, and 90-degree exact linear-twist "
                    "profiles with the VBD material-frame phase. The time plots report twist "
                    "linearity error and bend leakage during pure twist."
                ),
                figures=("analytic_twist_focus.png",),
            ),
        ),
        show_joints=True,
        joint_scale=1.5,
        show_main_video=False,
        extra_videos=(
            VideoVariant(
                suffix="bend_focus",
                title="Cable Bend Analytic Focus",
                caption=(
                    "Bend sweep at settle. Blue is the VBD centerline, green is the exact arc, "
                    "and both overlays are offset above the white tube for visibility."
                ),
                camera=CameraSpec((0.65, 90.70, 0.05), (0.65, 0.72, 0.05), fov=1.333),
                example_mode="bend",
            ),
            VideoVariant(
                suffix="twist_inspection",
                title="Cable Twist Multi-Angle Inspection",
                caption=(
                    "150-degree pure twist from three views. Orange is the VBD material-frame "
                    "phase; cyan is the exact linear-twist phase drawn at a larger radius."
                ),
                camera=CameraSpec((0.65, -3.55, 0.32), (0.65, -0.72, 0.02), fov=10.0),
                example_mode="twist_max",
                show_joints=False,
                view_panels=(
                    ViewPanel(
                        "Profile: cyan exact, orange VBD",
                        CameraSpec((0.65, -3.55, 0.32), (0.65, -0.72, 0.02), fov=17.0),
                    ),
                    ViewPanel(
                        "Oblique: overlap and straight centerline",
                        CameraSpec((1.65, -3.35, 0.95), (0.65, -0.72, 0.02), fov=18.0),
                    ),
                    ViewPanel(
                        "Down-axis: phase fan, no bowing",
                        CameraSpec((3.10, -0.72, 0.05), (0.65, -0.72, 0.02), fov=15.0),
                    ),
                ),
            ),
        ),
    ),
    ReportSpec(
        slug="cable_bend_stiffness",
        title="Cable Bend Cantilever Sweep",
        subtitle="Loaded cantilevers verify the bend stiffness path and monotone stiffness response.",
        alias="cable_bend_stiffness",
        module="newton.examples.vbd.example_cable_bend_stiffness",
        camera=CameraSpec((2.25, -2.95, 1.32), (0.78, 0.0, -0.12), fov=36.0),
        steps=510,
        stride=3,
        fps=30,
        figures=(),
        setup=(
            "This test verifies the loaded bend force-response path. Three horizontal "
            "cantilever cables receive the same downward tip force; only bend stiffness "
            "changes, so stiffer cables should deflect less and deflection times stiffness "
            "should remain nearly constant."
        ),
        result=(
            "The three cables settle in order: low stiffness sags most, high stiffness "
            "sags least. The quantitative scaling check is in the plot below the video."
        ),
        proves=(
            "Bend stiffness controls loaded bend response",
            "Loaded bend remains stable in the full solver",
            "The bend path behaves approximately Hookean over a stiffness sweep",
        ),
        nonredundant=(
            "The analytic boundary-value case is kinematic. This case uses an actual "
            "external force, so it checks the solver path that turns load into bend."
        ),
        visual_reference=(
            "The lowest-stiffness cable sags the most",
            "The highest-stiffness cable sags the least",
            "Tip markers stay ordered and close to the green 1/k guide",
            "Sideways drift remains near zero",
        ),
        failure_cues=(
            "A stiffer cable sags more than a softer cable",
            "The cables drift sideways under a purely vertical load",
            "Tip deflection keeps oscillating instead of settling",
        ),
        numerical_setup=(
            ("Solver fps", "60 Hz"),
            ("Solver timestep", "1/600 s (10 substeps per frame)"),
            ("VBD iterations", "10 per substep"),
            ("Segments / radius", "16 segments, h = 0.10 m, r = 0.010 m"),
            ("Cable stiffness", "stretch = 1.0e6 N/m, bend = 100, 300, 900 N m"),
            ("Twist stiffness", "0.77 * bend stiffness"),
            ("Load schedule", "0.50 N tip force, 2 s ramp + 6 s hold"),
        ),
        video_summary=(
            "The video shows the same vertical tip load applied to three cantilevers. "
            "The visible ordering is the main sanity check: low bend stiffness should "
            "sag most, high bend stiffness least. The plot below is the quantitative "
            "flat-delta-k check."
        ),
        figure_groups=(
            FigureGroup(
                title="Loaded Bend Response",
                body=(
                    "The sweep plot checks the primary assertion. Deflection must decrease "
                    "monotonically as bend stiffness increases, and deflection times stiffness "
                    "should stay nearly flat."
                ),
                figures=("bend_stiffness_sweep.png",),
            ),
            FigureGroup(
                title="Bend Reference Laws",
                body=(
                    "This plot is interpretive rather than the main pass/fail gate. It shows "
                    "the cantilever moment law and the discrete-to-continuum bend convergence trend."
                ),
                figures=("mechanical_laws.png",),
            ),
        ),
    ),
    ReportSpec(
        slug="cable_torsion_material_mapping",
        title="Cable Torsion Material Mapping",
        subtitle=(
            "Material properties mapped to k_twist = GJ / h, checked by material scaling "
            "and a straight-shaft torsion profile."
        ),
        alias="cable_torsion_material_mapping",
        module="newton.examples.vbd.example_cable_torsion_material_mapping",
        camera=CameraSpec((1.85, -3.00, 1.08), (0.0, -0.14, 0.45), fov=32.0),
        steps=510,
        stride=2,
        fps=30,
        figures=(),
        setup=(
            "This report starts from material properties, not a hand-picked twist stiffness: "
            "Young's modulus E, Poisson ratio nu, cable radius r, and segment length h. "
            "For an isotropic circular rod it computes G = E / (2(1 + nu)), "
            "J = pi r^4 / 2, then gives Newton the per-joint twist stiffness "
            "k_twist = GJ / h. "
            "With both endpoint positions fixed and the tip rotated by theta, the "
            "analytical twist profile is phi(x) = theta x / L. The appended scaling "
            "check changes E, explicit G, nu, r, and h, then verifies that the measured local torque "
            "response changes by the same GJ/h law."
        ),
        result=(
            "Orange simulated material-frame ticks overlay the cyan analytic ticks, "
            "and the centerline stays nearly straight."
        ),
        proves=(
            "GJ/h from E, nu, and r matches the measured torsional response across a one-at-a-time scaling sweep",
            "linear twist profile phi(x) = theta x / L matches the kinematic boundary conditions",
            "pure torsion does not leak into the bend subspace",
        ),
        nonredundant=(
            "The pure-twist profile overlaps with the analytic boundary-value report. "
            "The unique purpose here is the material-property mapping: E, G, nu, r, and h "
            "produce k_twist = GJ / h for the cable, and changing those material inputs "
            "changes the measured torque response by the predicted amount."
        ),
        visual_reference=(
            "Orange simulated material-frame ticks overlay the cyan analytic ticks",
            "The centerline remains nearly straight along the shaft axis",
            "Scaling sweep points fall on the identity line; radius cases show the r^4 slope",
        ),
        failure_cues=(
            "Simulated ticks lead or lag the analytic profile",
            "Centerline bows during the twist ramp or hold",
            "Scaling sweep points scatter far from the identity line",
        ),
        numerical_setup=(
            ("Solver fps", "60 Hz"),
            ("Solver timestep", "1/600 s (10 substeps per frame)"),
            ("VBD iterations", "60 per substep"),
            ("Segments / radius", "24 segments, h = 0.08 m, r = 0.012 m"),
            ("Material", "E = 2.0e9 Pa, nu = 0.30"),
            ("Derived shear modulus", "G ~= 7.69e8 Pa"),
            ("Derived polar inertia", "J ~= 3.26e-8 m^4"),
            ("Cable stiffness from material", "stretch ~= 1.13e7 N/m, bend ~= 407 N m, twist ~= 313 N m"),
            ("Drive", "90 deg tip twist, 3 s ramp + 5 s hold"),
            ("Scaling sweep", "16 one-at-a-time cases over E, G, radius, h, and nu at 5 deg local twist"),
        ),
        video_summary=(
            "Cyan ticks show the analytical linear twist profile for the commanded tip "
            "rotation. Orange ticks show the simulated material-frame phase on the same "
            "centerline. The right-side plots show twist-profile error and bend leakage."
        ),
        figure_groups=(
            FigureGroup(
                title="Material Scaling Check",
                body=(
                    "This is the stiffness-mapping proof. It sweeps Young's modulus, explicit "
                    "shear modulus, radius, segment length, and Poisson ratio. For each case "
                    "the report computes G = E / (2(1 + nu)) or uses the provided G, then "
                    "J = pi r^4 / 2 and k_twist = GJ / h. The VBD angular-force kernel is "
                    "queried for torque divided by imposed angle. The points should sit on "
                    "the identity line; the radius cases are especially diagnostic because "
                    "torsional stiffness scales as r^4."
                ),
                figures=("torsion_material_scaling.png",),
            ),
            FigureGroup(
                title="Ramp And Hold",
                body=(
                    "This is the motion-time check. It verifies that the material-derived "
                    "stiffness path stays well behaved while the tip is being twisted, then "
                    "settles cleanly after the ramp ends."
                ),
                figures=("torsion_material_time.png",),
            ),
            FigureGroup(
                title="Final Material-Mapped Torsion",
                body=(
                    "At the final frame, the simulated material-frame twist should match "
                    "the straight-shaft analytical profile phi(x) = theta x / L, while "
                    "the centerline remains nearly straight."
                ),
                figures=("torsion_law_transducer.png",),
            ),
        ),
    ),
    ReportSpec(
        slug="cable_twist_transfer",
        title="Cable DER-Inspired Routed Twist Behavior Demo",
        subtitle="Straight, V-kink, and semicircular paths demonstrate tangent-axis twist transfer through routed geometry.",
        alias="cable_twist_transfer",
        module="newton.examples.vbd.example_cable_twist_transfer",
        camera=CameraSpec((3.85, 2.30, 1.72), (0.0, 0.0, 0.35), fov=40.0),
        steps=390,
        stride=2,
        fps=30,
        figures=(),
        setup=(
            "Three endpoint-held paths receive the same 90-degree kinematic root rotation: "
            "a straight line, a sharp V-kink, and a smooth semicircle. The tip stays "
            "fixed, so the measurement is how much material-frame twist reaches the "
            "middle and tip side of each routed path. Gravity is disabled; the scene "
            "isolates twist transfer through the prescribed route. No external torque "
            "is applied."
        ),
        result=(
            "The straight path distributes twist toward the middle, the V-kink keeps "
            "most twist upstream of the sharp bend under this kinematic rotation, and "
            "the semicircle carries a visible amount of twist around smooth curvature."
        ),
        proves=(
            "the straight path matches its analytical linear twist profile",
            "sharp and smooth paths produce different qualitative transfer behavior",
            "shape motion stays bounded during routed twist",
        ),
        nonredundant=(
            "The straight analytic torsion tests do not exercise path routing. This "
            "behavior demo shows whether the same tangent-axis twist variable behaves "
            "differently on straight, sharply kinked, and smoothly curved geometry. "
            "It is inspired by the Discrete Elastic Rods Figure 5 asymmetry-of-twist "
            "example, but it is not a reproduction of that experiment and should not "
            "be read as a quantitative DER validation."
        ),
        visual_reference=(
            "Straight row: twist follows the straight-cable reference.",
            "V-kink row: twist mostly stays before the sharp bend.",
            "Semicircle row: twist visibly reaches the tip side of the smooth curve.",
            "All paths keep their overall shape recognizable.",
        ),
        failure_cues=(
            "V-kink transfers twist like a smooth curve",
            "semicircle fails to carry twist around the arc",
            "paths move so much that the route is no longer recognizable",
        ),
        numerical_setup=(
            ("Solver fps", "60 Hz"),
            ("Solver timestep", "1/600 s (10 substeps per frame)"),
            ("VBD iterations", "16 per substep"),
            ("Segments / radius", "32 segments per path, r = 0.012 m"),
            ("Cable stiffness", "stretch = 1.0e6 N/m, bend = 5.0e3 N m, twist = 2.0e2 N m"),
            ("Gravity", "off"),
            ("Kinematic root rotation", "90 deg about local tangent, 2 s ramp + 4 s hold"),
            ("Paths", "straight, V-kink, semicircle"),
        ),
        video_summary=(
            "The video shows the three routed paths under the same prescribed root "
            "rotation. The side plots are behavior metrics, not analytical error curves: "
            "they track straight-path distribution, V-kink localization, smooth-curve "
            "transfer, and shape motion."
        ),
        video_metric_notes=(
            (
                "straight mid twist",
                "Twist angle at the middle body of the straight path; it should rise toward about half the prescribed root rotation.",
            ),
            (
                "V-kink tip-side twist",
                "Largest twist after the bend; low values mean most twist stayed before the sharp bend.",
            ),
            (
                "semicircle tip-side twist",
                "Largest twist after the midpoint; nonzero values show transfer through smooth curvature.",
            ),
            (
                "max shape motion",
                "Largest body displacement from its starting path, normalized by path length.",
            ),
        ),
        figure_groups=(
            FigureGroup(
                title="Routed Twist Profiles",
                body=(
                    "The black dashed line is the exact straight-cable solution for the prescribed root and fixed tip orientations. "
                    "The cyan straight cable should overlap it. The orange and green routed curves do not have a simple closed-form "
                    "reference in this report; they are behavior checks. The V-kink should keep most twist before the bend, while "
                    "the semicircle should carry some twist around the smooth curve. This follows the qualitative idea of DER "
                    "Figure 5, where V-shaped and semi-circular reference rods show non-uniform twist when one end is rotated. "
                    "The scalar amounts are listed in Key Metrics."
                ),
                figures=("twist_transfer_profiles.png",),
            ),
        ),
    ),
    ReportSpec(
        slug="cable_plectoneme",
        title="Cable Plectoneme Formation Demo",
        subtitle="Dynamic fixed-end cable supercoiling with hard-history self-contact.",
        alias="cable_plectoneme",
        module="newton.examples.cable.example_cable_plectoneme",
        camera=CameraSpec((0.0, -4.0, 1.30), (0.0, 0.0, 1.05), fov=35.0),
        steps=780,
        stride=1,
        fps=60,
        figures=(),
        show_joints=False,
        joint_scale=2.0,
        setup=(
            "This dynamic SolverVBD cable example starts from a twist-free hanging "
            "arc, keeps both endpoint positions fixed, and gradually counter-twists "
            "the endpoint material frames. Past the buckling threshold the centerline "
            "leaves the initial plane and folds into a plectoneme held open by "
            "hard-history self-contact."
        ),
        result=(
            "Left: clean shape. Right: the same frame with Newton joint axes. The "
            "dynamic cable supercoils and reaches strand self-contact while "
            "the endpoints remain fixed."
        ),
        proves=(
            "fixed endpoint positions during dynamic twist loading",
            "out-of-plane deformation develops from endpoint counter-twist",
            "self-contact prevents the strands from passing through each other",
        ),
        nonredundant=(
            "The Michell and twisted-ring pages cover closed-loop buckling behavior. "
            "This page is the open-cable dynamic contact case."
        ),
        visual_reference=(
            "Left panel shows the clean centerline.",
            "Right panel adds Newton joint axes at the same timestep.",
            "The endpoints should stay fixed while the centerline supercoils.",
        ),
        failure_cues=(
            "centerline remains planar",
            "fixed endpoints move",
            "endpoint-adjacent segments visibly stretch",
            "strands pass through instead of contacting",
        ),
        numerical_setup=(
            ("Segments / radius", "80 segments, radius = 0.42 mean segment length"),
            ("Initial shape", "twist-free hanging arc"),
            ("Boundary", "fixed endpoint positions, counter-twisted endpoint frames"),
            ("Twist command", "6 turns over 8 s after a 2 s settle, then 3 s hold"),
            ("Contact", "hard-history self-contact, gap = 0.6 mean segment length"),
            ("Solver", "20 VBD iterations, 8 substeps per frame"),
            ("Scope", "dynamic SolverVBD plectoneme/contact demonstration"),
        ),
        video_summary=(
            "Both panels show the same timestep. Use the left panel for the shape and "
            "the right panel to inspect orientation."
        ),
        target_heading="Scenario",
        checks_heading="Review Checks",
        show_checks=False,
        show_main_video=False,
        extra_videos=(
            VideoVariant(
                suffix="clean_axes",
                title="Cable Plectoneme Formation Demo",
                caption=("Synchronized split view: clean shape on the left, Newton joint axes on the right."),
                camera=CameraSpec((2.0, -3.85, 1.95), (0.0, 0.0, 0.92), fov=39.0),
                view_panels=(
                    ViewPanel(
                        "Clean shape",
                        CameraSpec((0.0, -4.0, 1.30), (0.0, 0.0, 1.05), fov=35.0),
                        show_joints=False,
                    ),
                    ViewPanel(
                        "Newton joint axes",
                        CameraSpec((0.0, -4.0, 1.30), (0.0, 0.0, 1.05), fov=35.0),
                        show_joints=True,
                        joint_scale=2.0,
                    ),
                ),
            ),
        ),
    ),
    ReportSpec(
        slug="cable_localized_buckling",
        title="Cable Localized Helical Buckling",
        subtitle="Clamped twisted rod with axial shortening, using DER localized-buckling parameters.",
        alias="cable_localized_buckling",
        module="newton.examples.cable._example_cable_localized_buckling",
        camera=CameraSpec((1.65, -10.25, 3.35), (4.55, 0.0, 0.05), fov=34.0),
        steps=420,
        stride=2,
        fps=45,
        figures=(),
        setup=(
            "This example uses the DER localized-buckling parameter set: a naturally "
            "straight rod of length 9.29, bend modulus 1.345, twist modulus "
            "0.789, 27 clamp turns, and 0.3 units of axial shortening. "
            "Root and tip are kinematic; the interior rod uses Newton's cable "
            "stretch, geometric bend, and transported-twist energy. The report "
            "keeps two cases: a no-inertia quasistatic reproduction "
            "for validation, and a standard dynamic SolverVBD run for behavior. "
            "The imposed 27 turns are recorded as real-valued material twist, so the "
            "reported twist is not reduced to the quaternion principal branch. "
            "Endpoint-quaternion driving is intentionally excluded from the "
            "validation path because it is branch-limited for this 27-turn case."
        ),
        result=(
            "The quasistatic reproduction uses Newton's real-valued imposed "
            "material twist and a no-inertia solve on the analytical branch. Across "
            "n = 60, 80, and 100, the tangent-angle envelope converges toward the "
            "analytical tanh^2 profile and the peak tangent angle approaches the "
            "analytical value. The matching dynamic runs use the same resolutions "
            "as a sensitivity check, not as the analytical convergence proof. "
            "A direct endpoint-drive diagnostic loses roughly half the transported "
            "material twist and selects an end-localized branch, explaining the "
            "old dynamic discrepancy."
        ),
        proves=(
            "Newton's cable energy reproduces the localized-buckling equilibrium branch",
            "Newton retains the 27-turn material path instead of losing it to a quaternion branch",
            "large imposed twist couples into centerline buckling",
            "the kinematic clamp position remains enforced during the run",
        ),
        nonredundant=(
            "The analytic boundary-value report checks exact low-deformation bend and "
            "twist separately. This example stresses large twist, axial shortening, "
            "and bend/twist coupling in one open-rod setup."
        ),
        visual_reference=(
            "Orange centerline should leave the gray straight reference.",
            "The buckle amplitude should localize instead of becoming a uniform sideways drift.",
            "Blue clamp markers should remain fixed.",
        ),
        failure_cues=(
            "rod stays nearly straight",
            "tip clamp drifts away from the prescribed shortened position",
            "segments visibly stretch or collapse",
        ),
        numerical_setup=(
            ("Dynamic solver fps", "60 Hz"),
            ("Dynamic timestep", "1/600 s (10 substeps per frame)"),
            ("Dynamic iterations", "60 per substep"),
            (
                "Dynamic run command",
                "uv run --extra examples python -m newton.examples.cable._example_cable_localized_buckling",
            ),
            (
                "Quasistatic run command",
                "uv run --extra examples python -m newton.examples.cable._example_cable_localized_buckling "
                "--loading-mode quasistatic_reproduction --twist-mode initial_material --static-iterations 300",
            ),
            (
                "Quasistatic convergence command",
                "uv run --extra examples python -m newton.examples.cable._example_cable_localized_buckling "
                "--convergence --viewer null",
            ),
            ("Video capture", "both videos use the same camera, 1280 px width, 45 fps, and 420 simulated frames"),
            ("Quasistatic sweep", "n = 60, 80, 100 no-inertia static branch solves"),
            ("Rod", "L = 9.29 m, 100 default segments, r = 0.020 m"),
            ("Material", "bend modulus = 1.345, twist modulus = 0.789"),
            ("Stretch stiffness", "2.0e7 N/m"),
            ("Dynamic damping", "10.0 for stretch, bend, and twist"),
            ("Dynamic clamp drive", "initial 27-turn material twist; 0.3 m axial shortening over 4 s"),
            ("Quasistatic budget", "one 300-iteration no-inertia solve per plotted resolution"),
            ("Seed", "localized 5-turn helical seed, amplitude 5e-3 m, sigma = 0.12 L"),
            ("Gravity", "off"),
            ("Scope", "quasistatic validation plus dynamic behavior demo"),
        ),
        reproduce_command=(
            "uv run --extra examples python -m newton.examples.cable._example_cable_localized_buckling "
            "--test --viewer null"
        ),
        video_summary=(
            "The quasistatic video is shown first because it is the validation "
            "path: the analytical branch is advanced with inertia removed. The "
            "dynamic video follows as the finite-time SolverVBD behavior demo "
            "under the same parameters."
        ),
        figure_groups=(
            FigureGroup(
                title="Matched Final-State Diagnostics",
                body=(
                    "These two plots use the same three-panel layout, axes, camera, "
                    "target line, and diagnostics. The left plot is the no-inertia "
                    "quasistatic solve; the right plot is the standard dynamic "
                    "SolverVBD run under the same physical parameters."
                ),
                figures=("localized_buckling_quasistatic.png", "localized_buckling_dynamic.png"),
            ),
            FigureGroup(
                title="Quasistatic Resolution Sweep",
                body=(
                    "The resolution sweep uses the same quasistatic solver and the same "
                    "real-valued imposed material twist as the quasistatic final-state "
                    "plot. It is kept separate from the matched dynamic/quasistatic "
                    "A/B diagnostics because it answers a different question: refinement "
                    "toward the analytical envelope."
                ),
                figures=("localized_buckling_quasistatic_convergence.png",),
            ),
        ),
        show_main_video=False,
        extra_videos=(
            VideoVariant(
                suffix="quasistatic",
                title="Quasistatic Reproduction",
                caption=(
                    "No-inertia analytical-branch solve using Newton's cable energy, "
                    "rendered with the same camera and duration as the dynamic video."
                ),
                camera=CameraSpec((1.65, -10.25, 3.35), (4.55, 0.0, 0.05), fov=34.0),
                example_mode="quasistatic_guided",
            ),
            VideoVariant(
                suffix="dynamic",
                title="Dynamic Reproduction",
                caption=(
                    "Standard SolverVBD dynamics under the same localized-buckling "
                    "parameters, rendered with the same camera and duration as the "
                    "quasistatic video."
                ),
                camera=CameraSpec((1.65, -10.25, 3.35), (4.55, 0.0, 0.05), fov=34.0),
            ),
        ),
    ),
    ReportSpec(
        slug="cable_der_twisted_ring",
        title="Cable Twisted Ring Writhe Demo",
        subtitle="Closed-ring twist-to-writhe behavior with an untwisted control.",
        alias="cable_der_twisted_ring",
        module="newton.examples.cable._example_cable_der_twisted_ring",
        camera=CameraSpec((1.45, -3.25, 1.70), (0.0, 0.0, 0.60), fov=37.0),
        steps=240,
        stride=2,
        fps=30,
        figures=(),
        setup=(
            "Two closed rings are simulated with gravity off: an untwisted control "
            "and a ring initialized with one material-frame turn plus a tiny "
            "deterministic out-of-plane seed."
        ),
        result=(
            "The control remains planar, while the twisted ring converts internal "
            "twist energy into visible out-of-plane writhe."
        ),
        proves=(
            "closed-loop twist energy can become writhe",
            "the same ring without twist stays planar",
            "the basic twist-to-writhe response is visible before the Michell threshold check",
        ),
        nonredundant=(
            "The Michell page is the quantitative threshold test. This page is the "
            "simpler control-vs-twisted behavior demo. It is inspired by DER/Cosserat "
            "twisted-ring validations, but it does not measure DER topological "
            "quantities such as linking number, writhe, or a twist-writhe decomposition."
        ),
        visual_reference=(
            "Blue control ring should remain close to its gray rest outline.",
            "Orange twisted ring should leave its best-fit plane after the seed grows.",
            "Coplanarity is the main pass/fail metric; best-fit-plane span shows the visible writhe size.",
        ),
        failure_cues=(
            "control ring becomes non-coplanar",
            "twisted ring remains planar",
            "coplanarity separation is small",
        ),
        numerical_setup=(
            ("Solver fps", "60 Hz"),
            ("Solver timestep", "1/600 s (10 substeps per frame)"),
            ("VBD iterations", "32 per substep"),
            ("Ring / cable radius", "ring R = 0.55 m, cable r = 0.010 m"),
            ("Cable stiffness", "stretch = 1.0e7 N/m, bend = 50 N m, twist = 500 N m"),
            ("Twist cases", "control = 0 turn, twisted = 1 material-frame turn"),
            ("Gravity", "off"),
            ("Run time", "4.0 s"),
        ),
        video_summary=(
            "The video shows two separate closed rings. Blue is the untwisted control; "
            "orange is the ring initialized with one material-frame turn. Gray outlines "
            "show the starting ring planes."
        ),
        figure_groups=(
            FigureGroup(
                title="Control vs Twisted Ring",
                body=(
                    "Coplanarity is a scale-free best-fit-plane metric: zero means perfectly "
                    "planar, larger values mean the centerline has left its plane. The control "
                    "must stay below 1e-2; the twisted ring must exceed 5e-2 and separate clearly "
                    "from the control. The span panel measures distance through each ring's best-fit "
                    "plane, so it is not confused by a planar ring tilting in world space."
                ),
                figures=("twisted_ring_writhe.png",),
            ),
        ),
        uniform_cable_color=False,
    ),
    ReportSpec(
        slug="cable_michell_threshold",
        title="Cable Michell/Zajac Twist Threshold",
        subtitle="Closed-ring twist-to-writhe behavior checked against the analytical critical twist.",
        alias="cable_michell_threshold",
        module="newton.examples.vbd.example_cable_michell_threshold",
        camera=CameraSpec((0.0, -7.30, 2.25), (0.0, 0.0, 0.60), fov=44.0),
        steps=210,
        stride=1,
        fps=30,
        figures=(),
        setup=(
            "Closed rings sweep total twist divided by the analytical Michell/Zajac "
            "critical total twist, theta_c = 2*pi*sqrt(3*bend_stiffness/twist_stiffness). "
            "Clearly subcritical rings are left planar. Rings above 1.00x receive a tiny "
            "out-of-plane seed to select the writhe mode. The 0.95x case is a below-threshold "
            "near-boundary sample, and the 1.00x critical case is a diagnostic reference "
            "rather than a pass/fail point."
        ),
        result=(
            "Clearly subcritical rings remain nearly planar, while rings just above "
            "the analytical threshold leave the plane and form writhe."
        ),
        proves=(
            "closed-loop twist-to-writhe coupling appears around the analytical critical twist",
            "clearly subcritical planar rings stay planar",
            "above-threshold seeded rings leave the plane",
        ),
        nonredundant=(
            "The twisted-ring demo is a single control-vs-one-turn behavior check. "
            "This report is the threshold check: it sweeps total twist normalized "
            "by the analytical critical twist and shows where the response changes. "
            "The 1.00x ring marks the analytical boundary, but the seed protocol "
            "makes this a threshold-bracketing regression, not a full linear-stability "
            "eigenmode solve."
        ),
        visual_reference=(
            "Rings are ordered left-to-right by total twist / critical twist.",
            "Blue clearly subcritical rings should stay planar.",
            "Purple 1.00x ring is diagnostic only.",
            "Orange/red above-threshold rings should leave their best-fit planes.",
        ),
        failure_cues=(
            "clearly subcritical ring becomes strongly nonplanar",
            "above-threshold ring stays planar",
            "the observed transition is far from the analytical critical twist",
        ),
        numerical_setup=(
            ("Solver fps", "60 Hz"),
            ("Solver timestep", "1/480 s (8 substeps per frame)"),
            ("VBD iterations", "28 per substep"),
            ("Ring / cable radius", "ring R = 0.55 m, cable r = 0.010 m"),
            ("Cable stiffness", "stretch = 1.0e7 N/m, bend = 50 N m, twist = 500 N m"),
            ("Threshold sweep", "0.70x, 0.85x, 0.95x, 1.00x, 1.05x, 1.20x, 1.50x critical twist"),
            ("Seed amplitude", "1.0e-3 m on above-threshold rings"),
            ("Gravity", "off"),
            ("Run time", "3.5 s"),
        ),
        video_summary=(
            "The video shows a threshold sweep of closed rings ordered by total twist divided "
            "by the analytical critical twist. Gray outlines show the starting planes; colored "
            "lines show the current centerlines."
        ),
        figure_groups=(
            FigureGroup(
                title="Paper Threshold Map",
                body=(
                    "Criterion used here: undamped final shape. Blue means the final ring is "
                    "coplanar by the best-fit-plane metric; purple means the final ring is "
                    "non-coplanar. Both plots use the Michell/Zajac paper axes: twist stiffness "
                    "divided by bend stiffness on x, imposed twist theta^n on y, and the "
                    "analytical threshold curve in black. The left plot is Newton's one-turn "
                    "slice; the right plot is the full grid with the same color rule."
                ),
                figures=("michell_threshold_curve.png", "michell_figure8_grid.png"),
            ),
            FigureGroup(
                title="Paper Growth/Decay Criterion",
                body=(
                    "Criterion used here: perturbation amplitude on a lightly damped run. "
                    "Blue means the seeded perturbation amplitude decreases or stays small; "
                    "purple means it grows. The left plot is the same one-turn slice; the "
                    "right plot is the same full grid. The damping removes phase-of-oscillation "
                    "ambiguity; it does not change the elastic stiffnesses or analytical "
                    "threshold curve."
                ),
                figures=("michell_paper_criterion_curve.png", "michell_paper_criterion_grid.png"),
            ),
            FigureGroup(
                title="Threshold Sweep",
                body=(
                    "The left panel shows coplanarity: values near zero are planar, while larger "
                    "values indicate writhe. The dashed vertical line is the analytical "
                    "Michell/Zajac critical twist; the dotted horizontal lines are the self-test "
                    "acceptance gates, not additional theory. The 1.00x point sits on the "
                    "analytical boundary and is diagnostic only. The right panel shows "
                    "out-of-plane span through each ring's best-fit plane, so a rigid tilt of the "
                    "whole ring is not counted as writhe."
                ),
                figures=("michell_threshold_sweep.png",),
            ),
        ),
    ),
    ReportSpec(
        slug="cable_dahl_hysteresis",
        title="Cable Dahl Hysteresis Validation",
        subtitle="Elastic and Dahl cantilevers verify bend and twist history without subspace leakage.",
        alias="cable_dahl_hysteresis",
        module="newton.examples.vbd.example_cable_dahl_hysteresis",
        camera=CameraSpec((2.42, -3.65, 1.45), (1.05, -0.08, -0.02), fov=43.0),
        steps=840,
        stride=3,
        fps=30,
        figures=(),
        setup=(
            "Four cantilevers run in matched pairs. The bend pair receives the same "
            "one-sided cyclic tip force. The twist pair receives the same one-sided "
            "kinematic tip-twist cycle, and the report measures the internal reaction "
            "torque proxy from twist stiffness plus Dahl history stress."
        ),
        result=(
            "Dahl changes both histories: the bend cable has lower peak deflection "
            "and a larger settled residual, while the twist cable keeps a residual "
            "reaction torque after the commanded angle returns to zero."
        ),
        proves=(
            "Dahl history remains stable under cyclic bend",
            "Dahl history also acts in the twist subspace",
            "pure bend history does not leak into twist, and pure twist history does not leak into bend",
        ),
        nonredundant=(
            "This is the only stateful split-cable report. The other pages validate "
            "elastic stiffness, analytic geometry, and thresholds; this one validates "
            "persistent Dahl memory in both angular subspaces."
        ),
        visual_reference=(
            "Orange traces are elastic; blue traces have Dahl history enabled.",
            "The bend video shows force-driven shape change and residual set.",
            "The twist video shows prescribed tip rotation, material-frame tick marks, and residual reaction torque.",
            "The video inset plots use the same load/response axes as the first report-plot column; scene depth only separates the orange and blue traces.",
            "The first plot column contains the usual hysteresis loops: force versus deflection for bend, and reaction torque versus commanded angle for twist.",
            "The middle plot column is the same motion over time, with the gray dashed command overlaid.",
            "The right plot column summarizes peak response and settled residual memory.",
        ),
        failure_cues=(
            "Dahl looks identical to elastic",
            "bend residual set or twist residual reaction disappears after unloading",
            "pure bend creates twist history, or pure twist creates bend history",
        ),
        numerical_setup=(
            ("Solver fps", "60 Hz"),
            ("Solver timestep", "1/600 s (10 substeps per frame)"),
            ("VBD iterations", "10 per substep"),
            ("Segments / radius", "16 segments, h = 0.10 m, r = 0.010 m"),
            ("Cable stiffness", "stretch = 1.0e6 N/m, bend = 250 N m, twist = 250 N m"),
            ("Dahl parameters", "eps_max = 0.10, tau = 0.05"),
            ("Bend drive", "0.50 N one-sided cyclic tip force, 12 s cycle + 2 s settle"),
            ("Twist drive", "45 deg one-sided kinematic tip twist, 12 s cycle + 2 s settle"),
        ),
        video_summary=(
            "The two videos separate the two behaviors. Bend is force-driven, so correctness "
            "is visible as deflection and residual set. Twist is kinematically driven, so "
            "correctness is visible as tick rotation plus residual internal reaction torque."
        ),
        figure_groups=(
            FigureGroup(
                title="Hysteresis Checks",
                body=(
                    "The left column is the standard hysteresis view. For bend, it is tip "
                    "force versus tip deflection; for twist, it is reaction torque versus "
                    "commanded tip angle. The middle column shows the same cycle over time, "
                    "so the loop direction and unloading branch are explicit. The right "
                    "column reduces the cycle to peak response and residual memory after unloading."
                ),
                figures=("dahl_hysteresis.png",),
            ),
        ),
        show_main_video=False,
        extra_videos=(
            VideoVariant(
                suffix="bend",
                title="Cable Dahl Bend Hysteresis",
                caption=(
                    "Bend-only view: orange is elastic, blue has Dahl history. "
                    "Both see the same cyclic tip force. In the inset loop, horizontal "
                    "is tip force and vertical is downward deflection."
                ),
                camera=CameraSpec((2.05, -3.35, 1.05), (0.82, -0.56, -0.10), fov=39.0),
                example_mode="bend",
            ),
            VideoVariant(
                suffix="twist",
                title="Cable Dahl Twist Hysteresis",
                caption=(
                    "Twist-only view: the tip angle is prescribed, tick marks show twist, "
                    "and the in-scene loop uses the same axes as the report: horizontal "
                    "is commanded tip angle and vertical is signed reaction torque."
                ),
                camera=CameraSpec((2.02, -3.15, 1.18), (0.80, 0.34, 0.05), fov=38.0),
                example_mode="twist",
            ),
        ),
    ),
)

# (slug, title, subtitle, headline, n_videos) — external sibling reports (href="../{slug}/")
EXTERNAL_REPORTS: tuple[tuple[str, str, str, str, int], ...] = (
    (
        "cable_twist_unwrap",
        "Cable Incremental Twist Unwrapping (Not Supported)",
        "Archived diagnostic for a deferred solver feature; not part of the landing implementation.",
        "unsupported deferred feature",
        2,
    ),
    (
        "cable_twist_wiggle_buckling",
        "Cable Twist Wiggle Buckling",
        "Three fixed-span twist-only cases with hard-history self-contact.",
        "bend/twist stiffness cases",
        3,
    ),
)

# (slug, title, subtitle, headline, n_videos) — sub-pages inside cable_bend_twist_verification/ (href="{slug}/")
OVERVIEW_SUBPAGES: tuple[tuple[str, str, str, str, int], ...] = (
    (
        "rod_solver_validation",
        "Romero 2021 Rod Master-Curve Validation",
        "Quantitative cantilever and Bend-Twist master-curve checks for SolverVBD cable strain.",
        "Romero master curves",
        10,
    ),
)


CSS = r"""
:root {
  color-scheme: light;
  --ink: #17202a;
  --muted: #5f6f82;
  --line: #d9e0e8;
  --panel: #f7f9fb;
  --accent: #1f6f8b;
  --good: #146c43;
  --warn: #9a5b00;
  --max: 1280px;
}
* { box-sizing: border-box; }
body {
  margin: 0;
  background: #ffffff;
  color: var(--ink);
  font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  line-height: 1.5;
}
header { border-bottom: 1px solid var(--line); background: #fbfcfd; }
main, .header-inner, footer { width: min(var(--max), calc(100vw - 32px)); margin: 0 auto; }
.header-inner { padding: 36px 0 28px; }
h1 { margin: 0 0 10px; font-size: clamp(2rem, 4vw, 3.35rem); line-height: 1.05; letter-spacing: 0; }
.subtitle { max-width: 850px; margin: 0; color: var(--muted); font-size: 1.05rem; }
.metric-row { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 12px; margin-top: 22px; }
.metric { border: 1px solid var(--line); border-radius: 8px; background: #ffffff; padding: 14px 16px; }
.metric .label { color: var(--muted); font-size: 0.82rem; font-weight: 650; text-transform: uppercase; }
.metric .value { display: block; margin-top: 4px; font-size: 1.25rem; font-weight: 760; }
main { padding: 28px 0 48px; }
section { margin: 0 0 34px; }
h2 { margin: 0 0 10px; font-size: 1.45rem; line-height: 1.2; letter-spacing: 0; }
h3 { margin: 0 0 10px; font-size: 1rem; line-height: 1.25; }
p { margin: 0 0 12px; }
a { color: var(--accent); }
code {
  padding: 0.08em 0.25em;
  border-radius: 4px;
  background: #edf2f7;
  font-family: ui-monospace, SFMono-Regular, Consolas, "Liberation Mono", monospace;
  font-size: 0.92em;
}
pre { margin: 12px 0 0; padding: 14px 16px; overflow-x: auto; border: 1px solid var(--line); border-radius: 8px; background: var(--panel); }
pre code { padding: 0; background: transparent; }
.two-col { display: grid; grid-template-columns: minmax(0, 1fr) minmax(320px, 0.9fr); gap: 22px; align-items: start; }
.plot-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(420px, 1fr)); gap: 18px; }
.plot-grid-single { grid-template-columns: minmax(0, 1fr); }
.plot-block { margin-top: 18px; }
.plot-block:first-child { margin-top: 0; }
.breadcrumb { margin: 0 0 14px; font-size: 0.92rem; font-weight: 650; }
.breadcrumb a { color: var(--accent); text-decoration: none; }
.breadcrumb a:hover { text-decoration: underline; }
.video-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 18px; max-width: 1120px; margin: 18px auto 0; }
.video-grid-single { grid-template-columns: minmax(0, 1fr); max-width: 1280px; }
.video-main { max-width: 820px; margin: 14px auto 18px; }
figure { margin: 0; border: 1px solid var(--line); border-radius: 8px; background: #ffffff; overflow: hidden; }
figure img, video { display: block; width: 100%; height: auto; }
video { background: #111827; }
figcaption { padding: 10px 12px 12px; border-top: 1px solid var(--line); color: var(--muted); font-size: 0.92rem; }
.takeaways, .card-grid { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 12px; margin-top: 14px; }
.takeaway, .card { border: 1px solid var(--line); border-radius: 8px; padding: 14px; background: #ffffff; }
.card { display: block; color: inherit; text-decoration: none; }
.takeaway strong, .card strong { display: block; margin-bottom: 6px; }
.check-list { margin: 0; padding-left: 1.15rem; }
.check-list li { margin: 0.35rem 0; }
.note { border-left: 4px solid var(--accent); background: var(--panel); padding: 12px 14px; }
table { width: 100%; border-collapse: collapse; border: 1px solid var(--line); background: white; }
th, td { padding: 9px 10px; border-bottom: 1px solid var(--line); text-align: left; vertical-align: top; }
th { background: var(--panel); color: var(--muted); font-size: 0.86rem; text-transform: uppercase; }
footer { padding: 0 0 42px; color: var(--muted); font-size: 0.9rem; }
@media (max-width: 820px) {
  .metric-row, .takeaways, .two-col, .plot-grid, .video-grid, .card-grid { grid-template-columns: 1fr; }
}
"""


def _escape(text: Any) -> str:
    return html.escape(str(text), quote=True)


def _font(size: int):
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            pass
    return ImageFont.load_default()


FONT_TITLE = _font(24)
FONT_LABEL = _font(18)
FONT_SIDEBAR_TITLE = _font(19)
FONT_SIDEBAR = _font(15)
FONT_SIDEBAR_SMALL = _font(13)


class RawVideoWriter:
    def __init__(self, output: Path, width: int, height: int, fps: int, crf: int = 22):
        output.parent.mkdir(parents=True, exist_ok=True)
        self.output = output
        cmd = [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{width}x{height}",
            "-r",
            str(fps),
            "-i",
            "-",
            "-an",
            "-c:v",
            "libx264",
            "-crf",
            str(crf),
            "-pix_fmt",
            "yuv420p",
            str(output),
        ]
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    def write(self, frame: np.ndarray) -> None:
        assert self.proc.stdin is not None
        self.proc.stdin.write(np.ascontiguousarray(frame).tobytes())

    def close(self) -> None:
        assert self.proc.stdin is not None
        self.proc.stdin.close()
        code = self.proc.wait()
        if code != 0:
            raise RuntimeError(f"ffmpeg failed for {self.output} with exit code {code}")


def _configure_headless_newton():
    import pyglet

    pyglet.options["headless"] = True

    import warp as wp  # noqa: PLC0415

    wp.config.kernel_cache_dir = "/tmp/warp-cache"

    import newton  # noqa: PLC0415
    import newton.viewer  # noqa: PLC0415

    return newton, wp


def _make_args(newton, frames: int):
    parser = newton.examples.create_parser()
    args = newton.examples.default_args(parser)
    args.viewer = "gl"
    args.headless = True
    args.test = False
    args.quiet = True
    args.num_frames = frames
    return args


def _apply_uniform_cable_color(
    model,
    newton,
    wp,
    preserve_label_prefixes: tuple[str, ...] = (),
) -> None:
    if model.shape_count == 0 or model.shape_color is None or model.shape_type is None:
        return

    colors = model.shape_color.numpy().astype(np.float32, copy=True)
    shape_type = model.shape_type.numpy()
    cable_mask = shape_type != int(newton.GeoType.PLANE)
    if preserve_label_prefixes:
        preserved = np.zeros_like(cable_mask, dtype=bool)
        for idx, label in enumerate(getattr(model, "shape_label", ())):
            if idx >= preserved.shape[0]:
                break
            preserved[idx] = any(str(label).startswith(prefix) for prefix in preserve_label_prefixes)
        cable_mask &= ~preserved
    colors[cable_mask] = np.asarray(CABLE_RENDER_COLOR, dtype=np.float32)
    model.shape_color.assign(wp.array(colors, dtype=wp.vec3, device=model.shape_color.device))


def _capture_rgb(viewer) -> np.ndarray:
    gl = viewer.renderer.gl
    w = int(viewer.renderer._screen_width)
    h = int(viewer.renderer._screen_height)
    buf = (gl.GLubyte * (w * h * 3))()
    gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, viewer.renderer._frame_fbo)
    gl.glReadPixels(0, 0, w, h, gl.GL_RGB, gl.GL_UNSIGNED_BYTE, buf)
    gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0)
    return np.ctypeslib.as_array(buf).reshape(h, w, 3)[::-1].copy()


def _wrap_text(draw: ImageDraw.ImageDraw, text: str, font, width: int) -> list[str]:
    words = text.split()
    lines: list[str] = []
    line = ""
    for word in words:
        candidate = word if not line else f"{line} {word}"
        if draw.textlength(candidate, font=font) <= width:
            line = candidate
        else:
            if line:
                lines.append(line)
            line = word
    if line:
        lines.append(line)
    return lines


def _draw_wrapped(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    font,
    width: int,
    fill: tuple[int, int, int] = (23, 32, 42),
    line_gap: int = 4,
) -> int:
    x, y = xy
    for line in _wrap_text(draw, text, font, width):
        draw.text((x, y), line, font=font, fill=fill)
        y += font.size + line_gap
    return y


def _draw_sidebar_section(
    draw: ImageDraw.ImageDraw,
    x: int,
    y: int,
    width: int,
    title: str,
    items: tuple[str, ...] | list[str],
    *,
    max_items: int | None = None,
) -> int:
    draw.text((x, y), title, font=FONT_SIDEBAR_TITLE, fill=(23, 32, 42))
    y += 27
    for item in list(items)[:max_items]:
        bullet_x = x
        text_x = x + 18
        draw.ellipse((bullet_x + 2, y + 7, bullet_x + 8, y + 13), fill=(31, 111, 139))
        y = _draw_wrapped(draw, (text_x, y), item, FONT_SIDEBAR, width - 18, fill=(42, 52, 65), line_gap=3)
        y += 8
    return y + 8


def _decorate_frame(image: np.ndarray, spec: ReportSpec, label: str, title: str | None = None) -> np.ndarray:
    panel = Image.fromarray(image)
    title_h = 46
    out_w = panel.width
    out_h = panel.height + title_h
    if out_w % 2:
        out_w += 1
    if out_h % 2:
        out_h += 1
    canvas = Image.new("RGB", (out_w, out_h), (18, 24, 34))
    draw = ImageDraw.Draw(canvas)
    draw.text((14, 9), title or spec.title, font=FONT_TITLE, fill=(245, 247, 250))
    if label:
        tw = draw.textlength(label, font=FONT_LABEL)
        draw.text((max(14, out_w - int(tw) - 16), 13), label, font=FONT_LABEL, fill=(203, 213, 225))
    canvas.paste(panel, (0, title_h))
    return np.asarray(canvas)


def _configure_video_viewer(viewer, spec: ReportSpec, show_joints: bool, joint_scale: float | None) -> None:
    viewer.show_collision = True
    viewer.show_static = True
    viewer.show_contacts = False
    viewer.show_joints = show_joints
    if hasattr(viewer, "renderer"):
        viewer.renderer.line_width = 2.8
        if show_joints:
            viewer.renderer.joint_scale = spec.joint_scale if joint_scale is None else joint_scale


def _configure_camera(viewer, wp, camera: CameraSpec) -> None:
    viewer.set_camera(pos=wp.vec3(*camera.pos), pitch=camera.pitch, yaw=camera.yaw)
    if hasattr(viewer, "camera"):
        viewer.camera.look_at(camera.target)
        viewer.camera.fov = camera.fov


def _panel_with_label(image: np.ndarray, label: str) -> np.ndarray:
    panel = Image.fromarray(image)
    draw = ImageDraw.Draw(panel, "RGBA")
    pad = 8
    text_w = int(draw.textlength(label, font=FONT_LABEL))
    box_w = min(panel.width - 2 * pad, text_w + 2 * pad)
    box_h = FONT_LABEL.size + 2 * pad
    draw.rounded_rectangle((pad, pad, pad + box_w, pad + box_h), radius=5, fill=(17, 24, 39, 185))
    draw.text((2 * pad, 2 * pad - 1), label, font=FONT_LABEL, fill=(245, 247, 250, 255))
    return np.asarray(panel)


def _current_analytic_metrics(example) -> dict[str, float]:
    if not hasattr(example, "_load_scale"):
        return {}

    scale = example._load_scale(max(0.0, getattr(example, "sim_time", 0.0) - getattr(example, "frame_dt", 0.0)))
    metrics: dict[str, float] = {}

    bend_angle = []
    bend_shape = []
    for case in getattr(example, "bend_cases", []):
        measured_angles = example._measure_case_angles(case, axis_index=1)
        expected_angles = example._analytic_angles(case["target"], scale)
        if abs(measured_angles[-1] + expected_angles[-1]) < abs(measured_angles[-1] - expected_angles[-1]):
            measured_angles = -measured_angles
        measured_points = example._rod_points(case["bodies"])
        expected_points = example._analytic_bend_points(case["rest_points"], case["target"], scale)
        bend_angle.append(float(np.degrees(np.sqrt(np.mean((measured_angles - expected_angles) ** 2)))))
        shape_rms = float(np.sqrt(np.mean(np.sum((measured_points - expected_points) ** 2, axis=1))))
        bend_shape.append(100.0 * shape_rms / max(float(example.cable_length), 1.0e-12))
    if bend_angle:
        metrics["bend angle RMS [deg]"] = max(bend_angle)
        metrics["bend shape RMS [%L]"] = max(bend_shape)

    twist_linear = []
    twist_transverse = []
    for case in getattr(example, "twist_cases", []):
        measured_angles = example._measure_case_angles(case, axis_index=0)
        expected_angles = example._analytic_angles(case["target"], scale)
        if abs(measured_angles[-1] + expected_angles[-1]) < abs(measured_angles[-1] - expected_angles[-1]):
            measured_angles = -measured_angles
        points = example._rod_points(case["bodies"])
        rest = case["rest_points"]
        twist_linear.append(float(np.degrees(np.sqrt(np.mean((measured_angles - expected_angles) ** 2)))))
        trans = float(np.max(np.linalg.norm((points - rest)[:, 1:3], axis=1)))
        twist_transverse.append(100.0 * trans / max(float(example.cable_length), 1.0e-12))
    if twist_linear:
        metrics["twist linear RMS [deg]"] = max(twist_linear)
        metrics["twist transverse [%L]"] = max(twist_transverse)

    return metrics


def _current_bend_metrics(example) -> dict[str, float]:
    if not hasattr(example, "_measured_tip_state"):
        return {}

    states = example._measured_tip_state()
    deflections = np.asarray([s[0] for s in states], dtype=np.float64)
    dispy = np.asarray([s[2] for s in states], dtype=np.float64)
    k = np.asarray(example.BEND_STIFFNESS_VALUES, dtype=np.float64)
    cable_length = max(float(example.cable_length), 1.0e-12)

    side_drift = 100.0 * float(np.max(np.abs(dispy))) / cable_length
    metrics: dict[str, float] = {}
    force = example._force_at_time(max(0.0, getattr(example, "sim_time", 0.0) - getattr(example, "frame_dt", 0.0)))
    if force < 0.05 * float(example.TIP_FORCE_MAX):
        metrics["bend scale residual [%]"] = 0.0
        reference_inv = float(getattr(example, "HOOKE_REFERENCE_DELTA_TIMES_K", 1.0))
        metrics["bend guide RMS [%L]"] = 100.0 * float(np.sqrt(np.mean((reference_inv / k) ** 2))) / cable_length
        metrics["bend side drift [%L]"] = side_drift
        return metrics

    invariants = deflections * k
    mean_inv = float(np.mean(invariants))
    if abs(mean_inv) > 1.0e-12:
        metrics["bend scale residual [%]"] = 100.0 * float(np.max(np.abs(invariants / mean_inv - 1.0)))
    else:
        metrics["bend scale residual [%]"] = 0.0

    reference_inv = float(getattr(example, "HOOKE_REFERENCE_DELTA_TIMES_K", mean_inv))
    expected = reference_inv / k
    rms = float(np.sqrt(np.mean((deflections - expected) ** 2)))
    metrics["bend guide RMS [%L]"] = 100.0 * rms / cable_length
    metrics["bend side drift [%L]"] = side_drift
    return metrics


def _current_der_torque_metrics(example) -> dict[str, float]:
    if not all(hasattr(example, attr) for attr in ("_measure_twists", "_current_pos", "rest_pos")):
        return {}

    twists = np.asarray(example._measure_twists(), dtype=np.float64)
    if len(twists) < 2:
        return {}

    target_tip_twist = float(getattr(example, "TARGET_TIP_TWIST", twists[-1]))
    ramp_time = max(float(getattr(example, "RAMP_TIME", 1.0)), 1.0e-12)
    sim_time = float(getattr(example, "sim_time", 0.0))
    commanded_tip_twist = min(max(sim_time / ramp_time, 0.0), 1.0) * target_tip_twist
    expected_profile = np.linspace(0.0, commanded_tip_twist, len(twists), dtype=np.float64)
    profile_rms = float(np.sqrt(np.mean((twists - expected_profile) ** 2)))

    current_pos = np.asarray(example._current_pos(), dtype=np.float64)
    rest_pos = np.asarray(example.rest_pos, dtype=np.float64)
    transverse = current_pos - rest_pos
    transverse[:, 0] = 0.0
    max_transverse = float(np.max(np.linalg.norm(transverse, axis=1)))
    cable_length = max(float(getattr(example, "cable_length", 1.0)), 1.0e-12)

    return {
        "twist profile error [deg]": float(np.degrees(profile_rms)),
        "bend leakage [%L]": 100.0 * max_transverse / cable_length,
    }


def _current_der_twist_transfer_metrics(example) -> dict[str, float]:
    if not all(hasattr(example, attr) for attr in ("cases", "_measure_twist_profile", "_current_points")):
        return {}

    rows = {}
    max_drift_pct = 0.0
    for case in example.cases:
        twists = np.asarray(example._measure_twist_profile(case), dtype=np.float64)
        if len(twists) < 3:
            continue
        mid = len(twists) // 2
        points = np.asarray(example._current_points(case), dtype=np.float64)
        rest_pos = np.asarray(case["rest_pos"], dtype=np.float64)
        arc_length = max(float(case.get("arc_length", 1.0)), 1.0e-12)
        drift_pct = 100.0 * float(np.max(np.linalg.norm(points - rest_pos, axis=1))) / arc_length
        max_drift_pct = max(max_drift_pct, drift_pct)
        rows[case["label"]] = {
            "mid_deg": abs(float(np.degrees(twists[mid]))),
            "tip_side_deg": float(np.degrees(np.max(np.abs(twists[mid:])))),
        }

    if not rows:
        return {}

    return {
        "straight mid twist [deg]": rows.get("straight", {}).get("mid_deg", 0.0),
        "V-kink tip-side twist [deg]": rows.get("v_kink", {}).get("tip_side_deg", 0.0),
        "semicircle tip-side twist [deg]": rows.get("semicircle", {}).get("tip_side_deg", 0.0),
        "max shape motion [%L]": max_drift_pct,
    }


def _current_localized_buckling_metrics(example) -> dict[str, float]:
    if not hasattr(example, "metrics"):
        return {}
    row = example.metrics()
    return {
        "radial buckle [m]": float(row.get("max_radial", 0.0)),
        "peak angle [rad]": float(row.get("max_tangent_deviation_rad", 0.0)),
        "material twist [turns]": float(row.get("material_turns", 0.0)),
        "twist profile err [turns]": float(row.get("twist_profile_rms_error_turns", 0.0)),
        "segment stretch [%]": 100.0 * float(row.get("max_segment_stretch", 0.0)),
    }


def _current_video_metrics(spec: ReportSpec, example) -> dict[str, float]:
    if spec.slug == "cable_bend_twist_analytic":
        return _current_analytic_metrics(example)
    if spec.slug == "cable_bend_stiffness":
        return _current_bend_metrics(example)
    if spec.slug == "cable_torsion_material_mapping":
        return _current_der_torque_metrics(example)
    if spec.slug == "cable_twist_transfer":
        return _current_der_twist_transfer_metrics(example)
    if spec.slug == "cable_localized_buckling":
        return _current_localized_buckling_metrics(example)
    return {}


def _append_metric_history(history: dict[str, list[float]], metrics: dict[str, float]) -> None:
    for key, value in metrics.items():
        series = history.setdefault(key, [])
        series.append(float(value))
        del series[:-180]


def _draw_sparkline(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    values: list[float],
    color: tuple[int, int, int],
    fixed_ymax: float,
) -> None:
    x0, y0, x1, y1 = box
    draw.rectangle(box, outline=(70, 82, 98), width=1)
    if len(values) < 2:
        return
    vals = np.asarray(values[-120:], dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    if len(vals) < 2:
        return
    ymax = max(float(fixed_ymax), 1.0e-9)
    xs = np.linspace(x0 + 3, x1 - 3, len(vals))
    ys = y1 - 3 - (np.clip(vals, 0.0, ymax) / ymax) * (y1 - y0 - 6)
    draw.line(list(zip(xs, ys, strict=True)), fill=color, width=2)


def _draw_error_panel(
    width: int,
    height: int,
    metrics: dict[str, float],
    history: dict[str, list[float]],
) -> np.ndarray:
    panel = Image.new("RGB", (width, height), (35, 43, 55))
    draw = ImageDraw.Draw(panel)
    x = 18
    y = 18

    if not metrics:
        return np.asarray(panel)

    colors = [(48, 209, 88), (0, 190, 255), (255, 159, 10), (255, 214, 10)]
    metric_colors = {
        "straight mid twist [deg]": (26, 217, 255),
        "V-kink tip-side twist [deg]": (255, 140, 26),
        "semicircle tip-side twist [deg]": (64, 255, 89),
        "max shape motion [%L]": (255, 214, 10),
        "radial buckle [m]": (255, 140, 26),
        "tip twist [turns]": (26, 217, 255),
        "segment stretch [%]": (255, 214, 10),
    }
    plot_w = width - 2 * x
    plot_h = max(36, (height - 2 * y) // max(len(metrics), 1) - 24)
    short_names = {
        "bend angle RMS [deg]": "angle",
        "bend shape RMS [%L]": "shape",
        "twist linear RMS [deg]": "twist",
        "twist transverse [%L]": "drift",
        "bend side drift [%L]": "side",
        "bend scale residual [%]": "scale err",
        "bend guide RMS [%L]": "guide",
        "twist profile error [deg]": "profile err",
        "bend leakage [%L]": "bend leakage",
        "straight mid twist [deg]": "straight mid deg",
        "V-kink tip-side twist [deg]": "V-kink tip deg",
        "semicircle tip-side twist [deg]": "curve tip deg",
        "max shape motion [%L]": "shape %L",
        "radial buckle [m]": "radial",
        "tip twist [turns]": "twist turns",
        "segment stretch [%]": "stretch",
    }
    fixed_ranges = {
        "bend angle RMS [deg]": 0.002,
        "bend shape RMS [%L]": 0.001,
        "twist linear RMS [deg]": 0.025,
        "twist transverse [%L]": 0.001,
        "bend side drift [%L]": 0.05,
        "bend scale residual [%]": 5.0,
        "bend guide RMS [%L]": 1.0,
        "twist profile error [deg]": 0.5,
        "bend leakage [%L]": 0.2,
        "straight mid twist [deg]": 60.0,
        "V-kink tip-side twist [deg]": 15.0,
        "semicircle tip-side twist [deg]": 30.0,
        "max shape motion [%L]": 8.0,
        "radial buckle [m]": 0.5,
        "tip twist [turns]": 27.0,
        "segment stretch [%]": 6.0,
    }
    for i, (name, value) in enumerate(metrics.items()):
        color = metric_colors.get(name, colors[i % len(colors)])
        if y + plot_h + 22 > height - 12:
            break
        label = short_names.get(name, name)
        draw.text((x, y), label, font=FONT_SIDEBAR_SMALL, fill=(214, 222, 235))
        value_text = f"{value:.4g}"
        tw = draw.textlength(value_text, font=FONT_SIDEBAR_SMALL)
        draw.text((x + plot_w - int(tw), y), value_text, font=FONT_SIDEBAR_SMALL, fill=color)
        y += 16
        _draw_sparkline(
            draw,
            (x, y, x + plot_w, y + plot_h),
            history.get(name, []),
            color,
            fixed_ranges.get(name, max(value * 1.2, 1.0e-6)),
        )
        y += plot_h + 8
    return np.asarray(panel)


def render_video(
    spec: ReportSpec,
    output: Path,
    *,
    width: int,
    height: int,
    overwrite: bool,
    camera: CameraSpec | None = None,
    title: str | None = None,
    example_mode: str | None = None,
    steps: int | None = None,
    stride: int | None = None,
    show_joints: bool | None = None,
    joint_scale: float | None = None,
) -> None:
    if output.exists() and not overwrite:
        print(f"[video] keep existing {output}")
        return

    newton, wp = _configure_headless_newton()
    module = importlib.import_module(spec.module)
    render_steps = int(spec.steps if steps is None else steps)
    render_stride = int(spec.stride if stride is None else stride)
    if spec.slug == "cable_localized_buckling" and example_mode == "quasistatic":
        render_steps = (
            module.Example.CONTINUATION_LOAD_STEPS + 1
        ) * module.Example.CONTINUATION_SETTLE_FRAMES + module.Example.CONTINUATION_HOLD_FRAMES
    elif spec.slug == "cable_localized_buckling" and example_mode == "static":
        render_steps = module.Example.STATIC_LOAD_STEPS + module.Example.STATIC_HOLD_FRAMES
    elif spec.slug == "cable_localized_buckling" and example_mode == "static_refine":
        render_steps = spec.steps + module.Example.STATIC_REFINE_FRAMES
    elif spec.slug == "cable_localized_buckling" and example_mode == "paper_static":
        render_steps = module.Example.PAPER_BRANCH_HOLD_FRAMES
    elif spec.slug == "cable_localized_buckling" and example_mode in ("paper_guided", "quasistatic_guided"):
        render_steps = spec.steps
    show_side_panel = spec.slug in {
        "cable_bend_twist_analytic",
        "cable_torsion_material_mapping",
        "cable_twist_transfer",
        "cable_localized_buckling",
    }
    side_panel_width = 260 if show_side_panel else 0
    viewer_width = width - side_panel_width
    viewer = newton.viewer.ViewerGL(width=viewer_width, height=height, headless=True)

    args = _make_args(newton, render_steps + 1)
    original_twist_mode = getattr(module.Example, "TWIST_MODE", None)
    if example_mode is not None:
        args.cable_analytic_mode = example_mode
        args.cable_split_demo_mode = example_mode
        args.cable_dahl_mode = example_mode
        if spec.slug == "cable_localized_buckling":
            args.localized_loading_mode = example_mode
            if example_mode in ("paper_guided", "quasistatic_guided"):
                args.localized_loading_mode = "quasistatic_guided"
                args.localized_clamp_motion = "root-fixed"
                args.localized_static_iterations = 180
                args.localized_paper_branch_phase_scale = module.Example.PAPER_BRANCH_PHASE_SCALE
                module.Example.TWIST_MODE = "initial_material"
    elif spec.slug == "cable_localized_buckling":
        args.localized_clamp_motion = "root-fixed"
    try:
        example = module.Example(viewer, args)
    finally:
        if original_twist_mode is not None:
            module.Example.TWIST_MODE = original_twist_mode
    if spec.slug == "cable_localized_buckling" and example_mode in ("paper_guided", "quasistatic_guided"):
        example.PAPER_GUIDED_FRAMES = render_steps
    if spec.uniform_cable_color:
        _apply_uniform_cable_color(example.model, newton, wp, spec.preserve_shape_color_prefixes)
    show_joints = spec.show_joints if show_joints is None else show_joints
    _configure_video_viewer(viewer, spec, show_joints, joint_scale)
    camera = camera or spec.camera
    _configure_camera(viewer, wp, camera)

    writer: RawVideoWriter | None = None
    metric_history: dict[str, list[float]] = {}
    try:
        for step in range(render_steps + 1):
            if step == 0 or step % render_stride == 0 or step == render_steps:
                example.render()
                label = f"t={getattr(example, 'sim_time', 0.0):.2f}s"
                scene = _capture_rgb(viewer)
                if side_panel_width:
                    metrics = _current_video_metrics(spec, example)
                    _append_metric_history(metric_history, metrics)
                    side_panel = _draw_error_panel(
                        side_panel_width,
                        scene.shape[0],
                        metrics,
                        metric_history,
                    )
                    scene = np.hstack((scene, side_panel))
                frame = _decorate_frame(scene, spec, label, title=title)
                if writer is None:
                    writer = RawVideoWriter(output, frame.shape[1], frame.shape[0], fps=spec.fps)
                writer.write(frame)
            if step == render_steps:
                break
            example.step()
    finally:
        if writer is not None:
            writer.close()
        viewer.close()
    print(f"[video] wrote {output}")


def render_multiview_video(
    spec: ReportSpec,
    variant: VideoVariant,
    output: Path,
    *,
    width: int,
    height: int,
    overwrite: bool,
) -> None:
    if output.exists() and not overwrite:
        print(f"[video] keep existing {output}")
        return
    if not variant.view_panels:
        raise ValueError(f"{variant.suffix} has no view panels")

    newton, wp = _configure_headless_newton()
    module = importlib.import_module(spec.module)
    panel_width = width // 2
    panel_height = height if len(variant.view_panels) == 2 else height // 2
    viewer = newton.viewer.ViewerGL(width=panel_width, height=panel_height, headless=True)

    args = _make_args(newton, spec.steps + 1)
    if variant.example_mode is not None:
        args.cable_analytic_mode = variant.example_mode
        args.cable_split_demo_mode = variant.example_mode
        args.cable_dahl_mode = variant.example_mode
        if spec.slug == "cable_localized_buckling":
            args.localized_loading_mode = variant.example_mode
            if variant.example_mode in ("paper_guided", "quasistatic_guided"):
                args.localized_loading_mode = "quasistatic_guided"
                args.localized_clamp_motion = "root-fixed"
                args.localized_static_iterations = 180
                args.localized_paper_branch_phase_scale = module.Example.PAPER_BRANCH_PHASE_SCALE
    elif spec.slug == "cable_localized_buckling":
        args.localized_clamp_motion = "root-fixed"
    example = module.Example(viewer, args)
    if spec.uniform_cable_color:
        _apply_uniform_cable_color(example.model, newton, wp, spec.preserve_shape_color_prefixes)
    show_joints = spec.show_joints if variant.show_joints is None else variant.show_joints
    _configure_video_viewer(viewer, spec, show_joints, variant.joint_scale)

    writer: RawVideoWriter | None = None
    metric_history: dict[str, list[float]] = {}
    try:
        for step in range(spec.steps + 1):
            if step == 0 or step % spec.stride == 0 or step == spec.steps:
                panels = []
                for view_panel in variant.view_panels:
                    panel_show_joints = show_joints if view_panel.show_joints is None else view_panel.show_joints
                    panel_joint_scale = (
                        variant.joint_scale if view_panel.joint_scale is None else view_panel.joint_scale
                    )
                    _configure_video_viewer(viewer, spec, panel_show_joints, panel_joint_scale)
                    _configure_camera(viewer, wp, view_panel.camera)
                    example.render()
                    panels.append(_panel_with_label(_capture_rgb(viewer), view_panel.label))
                if len(panels) == 2:
                    tiled = np.hstack((panels[0], panels[1]))
                else:
                    while len(panels) < 3:
                        panels.append(np.zeros_like(panels[0]))
                    metrics = _current_video_metrics(spec, example)
                    _append_metric_history(metric_history, metrics)
                    error_panel = _draw_error_panel(
                        panel_width,
                        panel_height,
                        metrics,
                        metric_history,
                    )
                    tiled = np.vstack(
                        (
                            np.hstack((panels[0], panels[1])),
                            np.hstack((panels[2], error_panel)),
                        )
                    )
                label = f"t={getattr(example, 'sim_time', 0.0):.2f}s"
                frame = _decorate_frame(tiled, spec, label, title=variant.title)
                if writer is None:
                    writer = RawVideoWriter(output, frame.shape[1], frame.shape[0], fps=spec.fps)
                writer.write(frame)
            if step == spec.steps:
                break
            example.step()
    finally:
        if writer is not None:
            writer.close()
        viewer.close()
    print(f"[video] wrote {output}")


def load_metrics() -> dict[str, Any]:
    metrics_path = FIGURE_ROOT / "metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(
            f"{metrics_path} does not exist. Run scripts/generate_cable_bend_twist_report.py first."
        )
    return json.loads(metrics_path.read_text(encoding="utf-8"))


def _metric_rows(spec: ReportSpec, metrics: dict[str, Any]) -> list[tuple[str, str]]:
    if spec.slug == "cable_bend_twist_analytic":
        bend = metrics["analytic"]["bend_rows"]
        twist = metrics["analytic"]["twist_rows"]
        return [
            ("Bend angle RMS", f"{max(row['angle_rms_deg'] for row in bend):.4f} deg max"),
            ("Bend shape RMS", f"{max(row['shape_rms_pct_l'] for row in bend):.5f}% L max"),
            ("Twist linear RMS", f"{max(row['linear_rms_deg'] for row in twist):.4f} deg max"),
            ("Twist transverse", f"{max(row['transverse_pct_l'] for row in twist):.5f}% L max"),
        ]
    if spec.slug == "cable_bend_stiffness":
        bend = metrics["bend"]
        spread = (
            100.0
            * (max(bend["delta_times_k"]) - min(bend["delta_times_k"]))
            / max(np.mean(bend["delta_times_k"]), 1.0e-12)
        )
        residual = 100.0 * max(
            abs(v / max(np.mean(bend["delta_times_k"]), 1.0e-12) - 1.0) for v in bend["delta_times_k"]
        )
        return [
            ("Stiffness sweep", ", ".join(f"{k:.0f}" for k in bend["k"])),
            ("Deflection*k", ", ".join(f"{v:.2f}" for v in bend["delta_times_k"])),
            ("Max scale residual", f"{residual:.2f}%"),
            ("Delta*k range", f"{spread:.2f}%"),
            ("Max side drift", f"{max(abs(v) for v in bend['dispy']):.2e} m"),
        ]
    if spec.slug == "cable_torsion_material_mapping":
        row = metrics["der_torque"]
        scaling = metrics.get("der_torque_scaling", {})
        return [
            ("Material sweep cases", f"{int(scaling.get('case_count', 0))}"),
            ("Material scaling error", f"{100.0 * scaling.get('max_scale_relative_error', 0.0):.3e}%"),
            ("Twist stiffness from material", f"{row.get('twist_stiffness', 313.0):.2f} N m"),
            ("Twist profile RMS", f"{row['profile_rms_deg']:.4f} deg"),
            ("Bend leakage", f"{row['transverse_pct_l']:.5f}% L"),
        ]
    if spec.slug == "cable_twist_transfer":
        rows = {row["label"]: row for row in metrics["der_twist_transfer"]["rows"]}
        return [
            ("Straight profile RMS", f"{metrics['der_twist_transfer']['straight_profile_rms_deg']:.4f} deg"),
            ("V-kink tip-side twist", f"{rows['v_kink']['max_second_half_deg']:.2f} deg"),
            ("Semicircle tip-side twist", f"{rows['semicircle']['max_second_half_deg']:.2f} deg"),
            ("Max shape motion", f"{max(row['centerline_drift_pct_l'] for row in rows.values()):.2f}% L"),
        ]
    if spec.slug == "cable_plectoneme":
        final = metrics["plectoneme"]["final"]
        return [
            ("Endpoint twist command", f"{final['twist_command_deg']:.0f} deg"),
            ("Out-of-plane span", f"{final['out_of_plane_span_radii']:.2f} radii"),
            ("Closest strand distance", f"{final['min_strand_distance_radii']:.2f} radii"),
            ("Endpoint drift", f"{final['endpoint_drift']:.1e} m"),
            ("Endpoint segment length", f"{final['endpoint_segment_ratio']:.3f}x rest"),
        ]
    if spec.slug == "cable_localized_buckling":
        localized = metrics["localized_buckling"]
        paper_row = next(
            (row for row in localized.get("loading_modes", []) if row.get("label") == "quasistatic reproduction"),
            {},
        )
        dynamic_row = next(
            (row for row in localized.get("loading_modes", []) if row.get("label") == "dynamic reproduction"),
            {},
        )
        convergence_rows = localized.get("paper_convergence_rows", [])
        finest_row = max(convergence_rows, key=lambda row: row.get("segments", 0)) if convergence_rows else {}
        return [
            ("Target clamp twist", "27.0 turns"),
            ("Target axial shortening", "0.300 m"),
            ("Theory tangent angle", f"{localized['theory_phi0']:.3f} rad"),
            ("Quasistatic finest n", str(finest_row.get("segments", "not generated"))),
            ("Finest peak angle", f"{finest_row.get('peak_phi', 0.0):.3f} rad" if finest_row else "not generated"),
            (
                "Finest envelope R2",
                f"{finest_row.get('sech_fit_r_squared', 0.0):.3f}" if finest_row else "not generated",
            ),
            (
                "Dynamic peak angle",
                f"{dynamic_row.get('peak_phi', 0.0):.3f} rad" if dynamic_row else "not generated",
            ),
            (
                "Dynamic peak location",
                f"{dynamic_row.get('peak_location_s', 0.0):.3f} L" if dynamic_row else "not generated",
            ),
            (
                "Quasistatic peak angle",
                f"{paper_row.get('peak_phi', 0.0):.3f} rad" if paper_row else "not generated",
            ),
            (
                "Quasistatic peak location",
                f"{paper_row.get('peak_location_s', 0.0):.3f} L" if paper_row else "not generated",
            ),
            (
                "Quasistatic envelope R2",
                f"{paper_row.get('sech_fit_r_squared', 0.0):.3f}" if paper_row else "not generated",
            ),
            ("Quasistatic budget", "one 300-iteration no-inertia solve"),
            ("Current status", "quasistatic validation plus dynamic sensitivity"),
        ]
    if spec.slug == "cable_der_twisted_ring":
        rows = {row["label"]: row for row in metrics["der_ring"]["rows"]}
        separation = rows["twisted"]["coplanarity"] / max(rows["control"]["coplanarity"], 1.0e-12)
        return [
            ("Control coplanarity", f"{rows['control']['coplanarity']:.3e}"),
            ("Twisted coplanarity", f"{rows['twisted']['coplanarity']:.3e}"),
            ("Coplanarity separation", f"{separation:.1f}x"),
            ("Twisted plane span", f"{rows['twisted']['plane_span']:.3f} m"),
        ]
    if spec.slug == "cable_michell_threshold":
        rows = metrics["michell"]["rows"]
        stable_rows = [row for row in rows if row.get("expected") == "stable"]
        above_threshold = [row for row in rows if row["factor"] > 1.0]
        critical_rows = [row for row in rows if row.get("expected") == "critical"]
        max_stable = max(row["coplanarity"] for row in stable_rows)
        min_above_threshold = min(row["coplanarity"] for row in above_threshold)
        stable_limit = metrics["michell"].get("stable_coplanarity_max", 5.0e-3)
        writhe_limit = metrics["michell"].get("writhe_coplanarity_min", 5.0e-2)
        stable_factors = [
            row["factor"] for row in rows if row.get("expected") == "stable" and row["coplanarity"] < stable_limit
        ]
        writhe_factors = [row["factor"] for row in rows if row["factor"] > 1.0 and row["coplanarity"] > writhe_limit]
        if stable_factors and writhe_factors:
            observed_bracket = f"{max(stable_factors):.2f}x - {min(writhe_factors):.2f}x"
        else:
            observed_bracket = "not bracketed"
        critical_value = f"{critical_rows[0]['coplanarity']:.3e}" if critical_rows else "not sampled"
        return [
            ("Critical twist", f"{metrics['michell']['critical_twist_deg']:.2f} deg"),
            ("Observed transition", observed_bracket),
            ("Sweep points", f"{len(rows)}"),
            (
                "Acceptance gates",
                f"stable < {stable_limit:.1e}, above-threshold > {writhe_limit:.1e}",
            ),
            ("1.00x diagnostic coplanarity", critical_value),
            ("Max stable coplanarity", f"{max_stable:.3e}"),
            ("Min above-threshold coplanarity", f"{min_above_threshold:.3e}"),
            ("Coplanarity separation", f"{min_above_threshold / max(max_stable, 1.0e-12):.1f}x"),
        ]
    if spec.slug == "cable_dahl_hysteresis":
        dahl = metrics["dahl"]
        bend_rows = {row["name"]: row for row in dahl["bend_rows"]}
        twist_rows = {row["name"]: row for row in dahl["twist_rows"]}
        be = bend_rows["bend_elastic"]
        bd = bend_rows["bend_dahl"]
        te = twist_rows["twist_elastic"]
        td = twist_rows["twist_dahl"]
        max_leak = max(row["leak_sigma"] for row in list(bend_rows.values()) + list(twist_rows.values()))
        return [
            ("Bend peak ratio", f"{bd['max_deflection'] / be['max_deflection']:.2f}x elastic"),
            ("Bend residual ratio", f"{bd['residual'] / max(be['residual'], 1.0e-12):.1f}x elastic"),
            ("Twist residual reaction", f"{td['residual_reaction']:.2f} N*m"),
            ("Twist loop area", f"{td['loop_area']:.2f} vs {te['loop_area']:.2e} elastic"),
            ("Max subspace leak", f"{max_leak:.2e} N*m"),
        ]
    raise ValueError(spec.slug)


def _metric_cards(rows: list[tuple[str, str]], limit: int = 3) -> str:
    cards = []
    for label, value in rows[:limit]:
        cards.append(
            f'<div class="metric"><span class="label">{_escape(label)}</span>'
            f'<span class="value">{_escape(value)}</span></div>'
        )
    return "\n".join(cards)


def _metric_table(rows: list[tuple[str, str]]) -> str:
    body = "\n".join(f"<tr><td>{_escape(label)}</td><td>{_escape(value)}</td></tr>" for label, value in rows)
    return f"<table><thead><tr><th>Metric</th><th>Value</th></tr></thead><tbody>{body}</tbody></table>"


def _setup_table(rows: list[tuple[str, str]]) -> str:
    body = "\n".join(f"<tr><td>{_escape(label)}</td><td>{_escape(value)}</td></tr>" for label, value in rows)
    return f"<table><thead><tr><th>Setting</th><th>Value</th></tr></thead><tbody>{body}</tbody></table>"


def _copy_asset(src: Path, dst: Path) -> None:
    if not src.exists():
        raise FileNotFoundError(src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _versioned_asset_url(asset_path: Path, relative_url: str) -> str:
    if not asset_path.exists():
        return relative_url
    stat = asset_path.stat()
    return f"{relative_url}?v={int(stat.st_mtime)}-{stat.st_size}"


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    clean_text = "\n".join(line.rstrip() for line in text.splitlines()) + "\n"
    path.write_text(clean_text, encoding="utf-8")
    print(f"[report] wrote {path}")


def build_case_report(spec: ReportSpec, metrics: dict[str, Any]) -> None:
    report_dir = REPORTS_ROOT / spec.slug
    assets = report_dir / "assets"
    rows = _metric_rows(spec, metrics)
    setup_rows = list(spec.numerical_setup)
    setup_rows.append(("Report video", f"{spec.fps} fps render, {spec.steps} solver frames, stride {spec.stride}"))
    video_name = f"{spec.slug}.mp4"
    video_url = _versioned_asset_url(assets / video_name, f"assets/{video_name}")
    main_video_html = ""
    if spec.show_main_video:
        main_video_html = f"""
      <figure class="video-main">
        <video controls preload="metadata" muted playsinline>
          <source src="{_escape(video_url)}" type="video/mp4">
        </video>
        <figcaption>{_escape(spec.result)}</figcaption>
      </figure>
"""
    extra_video_html = ""
    if spec.extra_videos:
        video_grid_class = "video-grid video-grid-single" if len(spec.extra_videos) == 1 else "video-grid"
        extra_video_html = f"""
      <div class="{video_grid_class}">
        {
            "".join(
                f'''
        <figure>
          <video controls preload="metadata" muted playsinline>
            <source src="{_escape(_versioned_asset_url(assets / f'{spec.slug}_{variant.suffix}.mp4', f'assets/{spec.slug}_{variant.suffix}.mp4'))}" type="video/mp4">
          </video>
          <figcaption>{_escape(variant.caption)}</figcaption>
        </figure>
            '''
                for variant in spec.extra_videos
            )
        }
      </div>
"""

    figures_to_copy = list(spec.figures)
    for group in spec.figure_groups:
        figures_to_copy.extend(group.figures)
    for figure in dict.fromkeys(figures_to_copy):
        _copy_asset(FIGURE_ROOT / figure, assets / figure)

    def _figure_html(figure: str) -> str:
        caption = FIGURE_CAPTIONS.get(figure, figure)
        src = _versioned_asset_url(assets / figure, f"assets/{figure}")
        return f"""
        <figure>
          <img src="{_escape(src)}" alt="{_escape(spec.title)} plot {figure}">
          <figcaption>{_escape(caption)}</figcaption>
        </figure>
        """

    if spec.figure_groups:
        blocks = []
        for group in spec.figure_groups:
            grid_class = "plot-grid"
            if spec.slug == "cable_localized_buckling" and group.title == "Matched Final-State Diagnostics":
                grid_class = "plot-grid plot-grid-single"
            blocks.append(
                f"""
        <div class="plot-block">
          <h3>{_escape(group.title)}</h3>
          <p>{_escape(group.body)}</p>
          <div class="{grid_class}">
            {"".join(_figure_html(figure) for figure in group.figures)}
          </div>
        </div>
            """
            )
        figure_html = "\n".join(blocks)
    elif spec.figures:
        figure_html = f"""
      <div class="plot-grid">
        {"".join(_figure_html(figure) for figure in spec.figures)}
      </div>
"""
    else:
        figure_html = ""
    plot_checks_html = ""
    if figure_html:
        plot_checks_html = f"""
    <section>
      <h2>Plot Checks</h2>
      {figure_html}
    </section>
"""

    def _bullet_list(items: tuple[str, ...]) -> str:
        return f'<ul class="check-list">{"".join(f"<li>{_escape(item)}</li>" for item in items)}</ul>'

    video_summary = spec.video_summary or (
        "The video is captured from Newton's OpenGL viewer in headless mode, "
        "using the same example scene as the headless validation run. Visual aids "
        "are drawn in the scene itself when they make correctness easier to inspect. "
        "The video is the visual sanity check; the metric table and plots are the accuracy gate."
    )
    video_metric_notes_html = ""
    if spec.video_metric_notes:
        rows_html = "\n".join(
            f"<tr><td>{_escape(label)}</td><td>{_escape(description)}</td></tr>"
            for label, description in spec.video_metric_notes
        )
        video_metric_notes_html = f"""
      <h3>Video Metric Plots</h3>
      <table><thead><tr><th>Plot</th><th>Meaning</th></tr></thead><tbody>{rows_html}</tbody></table>
"""
    run_command = (
        spec.reproduce_command or f"uv run --extra examples python -m newton.examples {spec.alias} --test --viewer null"
    )
    acceptance_cards: list[str] = []
    if spec.visual_reference:
        acceptance_cards.append(
            f"""
        <div class="card">
          <h3>Passes When</h3>
          {_bullet_list(spec.visual_reference)}
        </div>
            """
        )
    if spec.failure_cues:
        acceptance_cards.append(
            f"""
        <div class="card">
          <h3>Watch For</h3>
          {_bullet_list(spec.failure_cues)}
        </div>
            """
        )
    if spec.proves:
        acceptance_cards.append(
            f"""
        <div class="card">
          <h3>Validated Behavior</h3>
          {_bullet_list(spec.proves)}
        </div>
            """
        )
    acceptance_html = ""
    if acceptance_cards and spec.show_checks:
        acceptance_html = f"""
    <section>
      <h2>{_escape(spec.checks_heading)}</h2>
      <div class="card-grid">
        {"".join(acceptance_cards)}
      </div>
    </section>
"""

    html_text = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{_escape(spec.title)}</title>
  <style>{CSS}</style>
</head>
<body>
  <header>
    <div class="header-inner">
      <nav class="breadcrumb"><a href="../cable_bend_twist_verification/">&larr; Cable Bend/Twist Verification</a></nav>
      <h1>{_escape(spec.title)}</h1>
      <p class="subtitle">{_escape(spec.subtitle)}</p>
      <div class="metric-row">
        {_metric_cards(rows)}
      </div>
    </div>
  </header>

  <main>
    <section class="two-col">
      <div>
        <h2>{_escape(spec.target_heading)}</h2>
        <p>{_escape(spec.setup)}</p>
        <p class="note">{_escape(spec.nonredundant)}</p>
      </div>
      <div>
        <h2>Key Metrics</h2>
        {_metric_table(rows)}
      </div>
    </section>

    <section>
      <h2>Simulation Setup</h2>
      {_setup_table(setup_rows)}
    </section>

    <section>
      <h2>Video Checks</h2>
      <p>{_escape(video_summary)}</p>
      {main_video_html}
      {extra_video_html}
      {video_metric_notes_html}
    </section>

    {acceptance_html}

    {plot_checks_html}

    <section>
      <h2>Reproduce</h2>
      <pre><code>{_escape(run_command)}</code></pre>
    </section>
  </main>

  <footer>
    Report payload: this page, Newton-rendered MP4 video(s), and selected PNG plots in <code>assets/</code>.
    Parent hub: <a href="../cable_bend_twist_verification/">Cable Bend/Twist Verification</a>.
  </footer>
</body>
</html>
"""
    _write(report_dir / "index.html", html_text)


def build_overview(metrics: dict[str, Any]) -> None:
    report_dir = REPORTS_ROOT / "cable_bend_twist_verification"

    cards = []
    for spec in REPORT_SPECS:
        rows = _metric_rows(spec, metrics)
        cards.append(
            f"""
            <a class="card" href="../{_escape(spec.slug)}/">
              <strong>{_escape(spec.title)}</strong>
              <span>{_escape(spec.subtitle)}</span><br>
              <code>{_escape(rows[0][1])}</code>
            </a>
            """
        )
    for slug, title, subtitle, headline, _n in EXTERNAL_REPORTS:
        cards.append(
            f"""
            <a class="card" href="../{_escape(slug)}/">
              <strong>{_escape(title)}</strong>
              <span>{_escape(subtitle)}</span><br>
              <code>{_escape(headline)}</code>
            </a>
            """
        )
    for slug, title, subtitle, headline, _n in OVERVIEW_SUBPAGES:
        cards.append(
            f"""
            <a class="card" href="{_escape(slug)}/">
              <strong>{_escape(title)}</strong>
              <span>{_escape(subtitle)}</span><br>
              <code>{_escape(headline)}</code>
            </a>
            """
        )

    _n_reports = len(REPORT_SPECS) + len(EXTERNAL_REPORTS) + len(OVERVIEW_SUBPAGES)
    _n_videos = (
        sum((1 if spec.show_main_video else 0) + len(spec.extra_videos) for spec in REPORT_SPECS)
        + sum(n for *_, n in EXTERNAL_REPORTS)
        + sum(n for *_, n in OVERVIEW_SUBPAGES)
    )
    html_text = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Cable Bend/Twist Verification</title>
  <style>{CSS}</style>
</head>
<body>
  <header>
    <div class="header-inner">
      <h1>Cable Bend/Twist Verification</h1>
      <p class="subtitle">
        Nonredundant visual and quantitative reports proving that Newton VBD cable bend and twist are split,
        accurate, stable, physically calibrated where expected, and visible in Newton-rendered scenes.
      </p>
      <div class="metric-row">
        <div class="metric"><span class="label">Visual Reports</span><span class="value">{_n_reports}</span></div>
        <div class="metric"><span class="label">Total Videos</span><span class="value">{_n_videos} MP4</span></div>
        <div class="metric"><span class="label">Primary Gate</span><span class="value">--test + plots</span></div>
      </div>
    </div>
  </header>

  <main>
    <section>
      <h2>Report Index</h2>
      <div class="card-grid">
        {"".join(cards)}
      </div>
    </section>

  </main>

  <footer>
    Designed to publish under <code>https://jumyungc.github.io/newton/reports/</code>.
  </footer>
</body>
</html>
"""
    _write(report_dir / "index.html", html_text)


def build_reports_index() -> None:
    cards = [
        """
        <a class="card" href="cable_bend_twist_verification/">
          <strong>Cable Bend/Twist Verification</strong>
          <span>Overview hub for the nonredundant VBD cable bend/twist reports.</span>
        </a>
        """,
    ]
    for spec in REPORT_SPECS:
        cards.append(
            f"""
            <a class="card" href="{_escape(spec.slug)}/">
              <strong>{_escape(spec.title)}</strong>
              <span>{_escape(spec.subtitle)}</span>
            </a>
            """
        )
    for slug, title, subtitle, _headline, _n in EXTERNAL_REPORTS:
        cards.append(
            f"""
            <a class="card" href="{_escape(slug)}/">
              <strong>{_escape(title)}</strong>
              <span>{_escape(subtitle)}</span>
            </a>
            """
        )
    html_text = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Newton Reports</title>
  <style>{CSS}</style>
</head>
<body>
  <header>
    <div class="header-inner">
      <h1>Newton Reports</h1>
      <p class="subtitle">Static verification reports designed to publish under <code>https://jumyungc.github.io/newton/reports/</code>.</p>
    </div>
  </header>
  <main>
    <section>
      <h2>Available Reports</h2>
      <div class="card-grid">{"".join(cards)}</div>
    </section>
  </main>
  <footer>Report payload: static HTML, PNG plots, and MP4 videos.</footer>
</body>
</html>
"""
    _write(REPORTS_ROOT / "index.html", html_text)


def render_videos(
    specs: tuple[ReportSpec, ...],
    *,
    width: int,
    height: int,
    overwrite: bool,
    metrics: dict[str, Any],
) -> None:
    for spec in specs:
        if spec.show_main_video:
            render_video(
                spec,
                REPORTS_ROOT / spec.slug / "assets" / f"{spec.slug}.mp4",
                width=width,
                height=height,
                overwrite=overwrite,
            )
        for variant in spec.extra_videos:
            output = REPORTS_ROOT / spec.slug / "assets" / f"{spec.slug}_{variant.suffix}.mp4"
            if variant.view_panels:
                render_multiview_video(spec, variant, output, width=width, height=height, overwrite=overwrite)
            else:
                render_video(
                    spec,
                    output,
                    width=width,
                    height=height,
                    overwrite=overwrite,
                    camera=variant.camera,
                    title=variant.title,
                    example_mode=variant.example_mode,
                    show_joints=variant.show_joints,
                    joint_scale=variant.joint_scale,
                )


def _render_one_video_variant(
    spec: ReportSpec,
    variant_name: str,
    *,
    width: int,
    height: int,
    overwrite: bool,
) -> None:
    if variant_name == "main":
        if not spec.show_main_video:
            return
        render_video(
            spec,
            REPORTS_ROOT / spec.slug / "assets" / f"{spec.slug}.mp4",
            width=width,
            height=height,
            overwrite=overwrite,
        )
        return

    for variant in spec.extra_videos:
        if variant.suffix == variant_name:
            output = REPORTS_ROOT / spec.slug / "assets" / f"{spec.slug}_{variant.suffix}.mp4"
            if variant.view_panels:
                render_multiview_video(spec, variant, output, width=width, height=height, overwrite=overwrite)
            else:
                render_video(
                    spec,
                    output,
                    width=width,
                    height=height,
                    overwrite=overwrite,
                    camera=variant.camera,
                    title=variant.title,
                    example_mode=variant.example_mode,
                    show_joints=variant.show_joints,
                    joint_scale=variant.joint_scale,
                )
            return

    raise ValueError(f"Unknown video variant {variant_name!r} for {spec.slug}")


def _video_variant_names(spec: ReportSpec) -> tuple[str, ...]:
    names = ("main",) if spec.show_main_video else ()
    return names + tuple(variant.suffix for variant in spec.extra_videos)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--videos", action="store_true", help="Render Newton OpenGL MP4 videos before writing HTML.")
    parser.add_argument("--overwrite-videos", action="store_true", help="Regenerate existing MP4 files.")
    parser.add_argument("--width", type=int, default=1280, help="OpenGL capture width.")
    parser.add_argument("--height", type=int, default=720, help="OpenGL capture height.")
    parser.add_argument(
        "--only",
        choices=[spec.slug for spec in REPORT_SPECS],
        nargs="*",
        default=None,
        help="Limit generation to selected report slugs.",
    )
    parser.add_argument(
        "--video-variant",
        default=None,
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args()

    metrics = load_metrics()
    selected = tuple(spec for spec in REPORT_SPECS if args.only is None or spec.slug in set(args.only))

    for spec in selected:
        (REPORTS_ROOT / spec.slug / "assets").mkdir(parents=True, exist_ok=True)
    (REPORTS_ROOT / "cable_bend_twist_verification" / "assets").mkdir(parents=True, exist_ok=True)

    if args.video_variant is not None:
        if not args.videos:
            raise ValueError("--video-variant is only valid with --videos")
        if len(selected) != 1:
            raise ValueError("--video-variant requires exactly one --only report slug")
        _render_one_video_variant(
            selected[0],
            args.video_variant,
            width=args.width,
            height=args.height,
            overwrite=args.overwrite_videos,
        )
        return

    if args.videos:
        for spec in selected:
            for variant_name in _video_variant_names(spec):
                cmd = [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--videos",
                    "--only",
                    spec.slug,
                    "--video-variant",
                    variant_name,
                    "--width",
                    str(args.width),
                    "--height",
                    str(args.height),
                ]
                if args.overwrite_videos:
                    cmd.append("--overwrite-videos")
                subprocess.run(cmd, cwd=REPO_ROOT, check=True)

    for spec in selected:
        build_case_report(spec, metrics)
    if args.only is None:
        build_overview(metrics)
        build_reports_index()


if __name__ == "__main__":
    main()
