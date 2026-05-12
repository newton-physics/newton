# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Generate vector figures for the routed cable joint formulation note."""

from __future__ import annotations

import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "images" / "cable_joints_formulation"


def tangent_point_circle(
    p: tuple[float, float],
    center: tuple[float, float],
    radius: float,
    orientation: int,
) -> tuple[float, float]:
    """2-D equivalent of Newton's circular rolling-link tangent construction."""
    px, py = p
    cx, cy = center
    dx = cx - px
    dy = cy - py
    dist = math.hypot(dx, dy)
    if dist <= radius:
        raise ValueError("external point must be outside the circle")

    ux = dx / dist
    uy = dy / dist
    # plane_normal is +z, so cross(plane_normal, u) = (-u_y, u_x).
    vx = -uy
    vy = ux
    phi = math.asin(min(radius / dist, 1.0))
    angle = -math.pi / 2.0 - phi if orientation > 0 else math.pi / 2.0 + phi
    return (
        cx + radius * (math.cos(angle) * ux + math.sin(angle) * vx),
        cy + radius * (math.cos(angle) * uy + math.sin(angle) * vy),
    )


def build_capstan_svg() -> str:
    width = 760
    height = 430
    xmin, xmax = -3.7, 3.9
    ymin, ymax = -3.05, 1.75
    scale = min(width / (xmax - xmin), height / (ymax - ymin))
    x_pad = (width - (xmax - xmin) * scale) * 0.5
    y_pad = (height - (ymax - ymin) * scale) * 0.5

    def xy(p: tuple[float, float]) -> tuple[float, float]:
        x, y = p
        return (x_pad + (x - xmin) * scale, y_pad + (ymax - y) * scale)

    def fmt(p: tuple[float, float]) -> str:
        x, y = xy(p)
        return f"{x:.2f},{y:.2f}"

    center = (0.0, 0.0)
    radius = 1.0
    p_l = (-1.0, -2.55)
    p_r = (1.0, -2.55)
    t_l = tangent_point_circle(p_l, center, radius, orientation=-1)
    t_r = tangent_point_circle(p_r, center, radius, orientation=1)

    arc_points = [
        (radius * math.cos(theta), radius * math.sin(theta))
        for theta in [math.pi - i * math.pi / 72.0 for i in range(73)]
    ]
    cable_points = [p_l, t_l, *arc_points[1:-1], t_r, p_r]
    cable_path = "M " + " L ".join(fmt(p) for p in cable_points)

    angle_arc = [
        (0.46 * math.cos(theta), 0.46 * math.sin(theta)) for theta in [math.pi - i * math.pi / 48.0 for i in range(49)]
    ]
    angle_path = "M " + " L ".join(fmt(p) for p in angle_arc)

    # Arrow marker is intentionally small so the diagram reads cleanly when
    # downscaled inside the paper.
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-label="Capstan cable constraint geometry">
  <defs>
    <marker id="arrow-blue" markerWidth="9" markerHeight="7" refX="8" refY="3.5" orient="auto">
      <path d="M 0 0 L 9 3.5 L 0 7 z" fill="#1f67d2"/>
    </marker>
    <marker id="arrow-red" markerWidth="8" markerHeight="6" refX="7" refY="3" orient="auto">
      <path d="M 0 0 L 8 3 L 0 6 z" fill="#c84b31"/>
    </marker>
    <style>
      text {{ font-family: DejaVu Sans, Arial, sans-serif; fill: #16202a; }}
      .mono {{ font-family: DejaVu Sans Mono, Consolas, monospace; }}
      .muted {{ fill: #5b6776; }}
    </style>
  </defs>
  <rect x="0" y="0" width="{width}" height="{height}" rx="16" fill="#ffffff"/>
  <text x="36" y="44" font-size="24" font-weight="700">Capstan cable row</text>
  <text x="36" y="72" font-size="14" class="muted">Geometry generated with the same circular tangent construction used for rolling links.</text>

  <circle cx="{xy(center)[0]:.2f}" cy="{xy(center)[1]:.2f}" r="{radius * scale:.2f}" fill="#fff7e6" stroke="#c77a1f" stroke-width="4"/>
  <circle cx="{xy(center)[0]:.2f}" cy="{xy(center)[1]:.2f}" r="4" fill="#c77a1f"/>
  <text x="{xy(center)[0] - 23:.2f}" y="{xy(center)[1] + 7:.2f}" font-size="15">g</text>
  <text x="{xy(center)[0] + 18:.2f}" y="{xy(center)[1] + 33:.2f}" font-size="14" class="muted">axis a</text>

  <path d="{cable_path}" fill="none" stroke="#202a34" stroke-width="7" stroke-linecap="round" stroke-linejoin="round"/>
  <circle cx="{xy(t_l)[0]:.2f}" cy="{xy(t_l)[1]:.2f}" r="6" fill="#202a34"/>
  <circle cx="{xy(t_r)[0]:.2f}" cy="{xy(t_r)[1]:.2f}" r="6" fill="#202a34"/>

  <path d="{angle_path}" fill="none" stroke="#c84b31" stroke-width="3" marker-end="url(#arrow-red)"/>
  <text x="{xy((-0.18, 0.58))[0]:.2f}" y="{xy((-0.18, 0.58))[1]:.2f}" font-size="17" fill="#c84b31">theta</text>

  <path d="M {fmt((-1.34, -2.05))} L {fmt((-1.34, -0.75))}" stroke="#1f67d2" stroke-width="3" marker-end="url(#arrow-blue)"/>
  <text x="{xy((-1.72, -1.32))[0]:.2f}" y="{xy((-1.72, -1.32))[1]:.2f}" font-size="18" fill="#1f67d2">T_l</text>
  <path d="M {fmt((1.22, -0.75))} L {fmt((1.22, -2.05))}" stroke="#1f67d2" stroke-width="3" marker-end="url(#arrow-blue)"/>
  <text x="{xy((0.55, -1.34))[0]:.2f}" y="{xy((0.55, -1.34))[1]:.2f}" font-size="18" fill="#1f67d2">T_r</text>

  <text x="{xy((-1.95, -2.85))[0]:.2f}" y="{xy((-1.95, -2.85))[1]:.2f}" font-size="14" class="muted">rest state: r_l + r_r = const</text>

  <rect x="484" y="124" width="224" height="146" rx="10" fill="#f5f9ff" stroke="#b9d6ff"/>
  <text x="504" y="154" font-size="16" font-weight="700">Capstan admissibility</text>
  <text x="504" y="186" font-size="15" class="mono">alpha = exp(mu theta)</text>
  <text x="504" y="214" font-size="15" class="mono">T_l &lt;= alpha T_r</text>
  <text x="504" y="240" font-size="15" class="mono">T_r &lt;= alpha T_l</text>

  <rect x="484" y="292" width="224" height="72" rx="10" fill="#fff8f0" stroke="#ecc28b"/>
  <text x="504" y="322" font-size="15" class="mono">slip moves rest length</text>
  <text x="504" y="348" font-size="15" class="mono">until a bound is active</text>
</svg>
"""


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "capstan_geometry.svg").write_text(build_capstan_svg())


if __name__ == "__main__":
    main()
