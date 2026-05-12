<!-- SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers -->
<!-- SPDX-License-Identifier: CC-BY-4.0 -->

# Routed Cable Joint Formulation

The routed cable joint formulation is maintained as a LaTeX white paper:

- [LaTeX source](cable_joints_formulation.tex)
- [Rendered PDF](https://reports.mmacklin.com/cable-sim-research/cable_joints_formulation.pdf)
- [Capstan SVG figure](images/cable_joints_formulation/capstan_geometry.svg)

The paper derives the equations of motion and the exact solver rows used by
the current XPBD and VBD tendon paths:

- route and tangent-point updates,
- unilateral free-span stretch rows,
- capstan slip as a rest-length projection,
- rolling spin-axis angular coupling,
- pinhole friction from local bend angle, and
- capstan hysteresis from stateful rest-length distribution.
