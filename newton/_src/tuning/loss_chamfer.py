# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Chamfer loss: distance to the goal, which does not saturate."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import warp as wp

from .loss import TuningLoss

# --- Chamfer: distance to the goal, which does not saturate ------------------
#
# The loss grows with the distance between the simulated and the recorded cable and
# does not saturate — a cable 100 px away scores 100 — so the search can still rank
# candidates whose cable does not overlap the goal at all.
#
#   L_frame = 1/2 ( mean_{p on axis} EDT_goal[p]                     sim -> goal
#                 + mean_{q in goal} dist(q, projected cable axis)   goal -> sim
#                 ) / goal_extent
#
# BOTH directions are needed. sim->goal alone is degenerate: it only asks that every
# point of the simulated cable be near the goal, which a cable curled into a ball
# satisfies perfectly while leaving most of the goal uncovered. An optimizer finds
# that shape: the silhouette gets shorter and wider while the loss keeps improving.
#
# The two halves are computed differently, because only one of them can precompute.
# A distance field can be built ahead of time for the thing being measured TO, and
# only the goal is fixed:
#   sim->goal   query points vary, field is the goal's  -> one distance transform in
#               prepare(), then a table lookup per axis sample.
#   goal->sim   query points fixed, field would be the SIM's, which differs for every
#               candidate at every frame -> one transform per candidate and frame.
# So the second direction skips the field entirely: the projected cable axis is a
# polyline with one segment per capsule, and the distance to it is closed-form
# point-to-segment. No transform and no rasterisation. The goal side uses only the
# mask's pixels; no centreline is extracted from the recording.
#
# Distance is to the cable's AXIS, not its surface — deliberately. Subtracting the
# radius would make a perfect fit score 0 instead of ~a quarter of the goal's width,
# but it also clamps to zero every goal pixel lying inside the cable, discarding their
# signal and flattening the valley around the optimum. A search that ranks candidates
# is not affected by the constant offset, but it gains from the steeper valley.
#
# Both terms are dimensionless: dividing by the goal's on-screen length makes the
# result "error in cable-lengths", comparable across cameras at different distances
# and across image resolutions.
#
# Distances are accumulated as fixed-point integers: integer atomics are exactly
# associative, so the loss stays bit-reproducible regardless of the order blocks
# finish in. The goal->sim sum runs over every goal pixel, so it is int64; the
# sim->goal sum covers a fixed number of axis samples and fits int32.
_CHAMFER_FIXED = wp.constant(16.0)  # fixed-point scale, 1/16 px resolution
_CHAMFER_MAX_PX = 1024.0  # distance clamp; bounds each sample's contribution
_PROJ_INVALID = wp.constant(-1.0e8)  # a projected node below this is behind the lens
CHAMFER_MISS = 2.0  # frame score when the sim projects nothing

# Samples along the projected cable axis for the sim->goal direction, at uniform
# arc length: sample k sits at (k + 0.5) / N of the visible length, so each sample
# stands for an equal piece of the curve however the capsules foreshorten. The
# count is fixed so the launch shape does not depend on the candidate and the
# samples move continuously with the parameters.
_CHAMFER_AXIS_SAMPLES = wp.constant(512)

# Goal pixels one thread sums before one atomic add, to reduce contention on the
# per-world counters.
_PIXEL_STRIP = wp.constant(32)


def _goal_edt(mask: np.ndarray) -> np.ndarray:
    """Distance [px] from every pixel to the nearest set pixel of ``mask``.

    Computed once per goal frame in prepare(), which is what makes the sim->goal
    direction a table lookup per sample rather than a search over goal pixels.

    SciPy is imported here rather than at module level, because it is an optional
    dependency.
    """
    from scipy.ndimage import distance_transform_edt

    inv = np.where(mask > 0, 0, 255).astype(np.uint8)
    # Distance to the nearest ZERO, and the inversion above makes the cable the
    # zeros, so this is distance-to-cable: the exact Euclidean distance.
    edt = distance_transform_edt(inv)
    return np.minimum(edt, _CHAMFER_MAX_PX)


def _mask_extent(mask: np.ndarray) -> float:
    """The mask's length along its principal axis [px] — the cable's on-screen length.

    Used to normalize the distance term. Taken from the goal (not the sim) so the
    scale is a property of the observation, fixed for the whole run.
    """
    ys, xs = np.nonzero(mask > 0)
    if len(xs) < 2:
        return 1.0
    pts = np.stack([xs, ys]).astype(np.float64)
    pts -= pts.mean(1, keepdims=True)
    u, _, _ = np.linalg.svd(pts, full_matrices=False)
    t = u[:, 0] @ pts
    return max(1.0, float(t.max() - t.min()))


@wp.func
def _point_seg_dist(px: float, py: float, ax: float, ay: float, bx: float, by: float) -> float:
    """Distance from (px, py) to the segment a-b, in pixels."""
    vx = bx - ax
    vy = by - ay
    wx = px - ax
    wy = py - ay
    l2 = vx * vx + vy * vy
    t = float(0.0)
    if l2 > 1.0e-12:
        t = wp.clamp((wx * vx + wy * vy) / l2, 0.0, 1.0)
    dx = wx - t * vx
    dy = wy - t * vy
    return wp.sqrt(dx * dx + dy * dy)


@wp.func
def _axis_length(uv: wp.array3d[wp.float32], i: int, n_nodes: int):
    """Total projected length [px] of world ``i``'s cable axis.

    Segments with an endpoint behind the lens are excluded, so the arc-length
    parametrisation below covers only the visible part of the cable.
    """
    total = float(0.0)
    for s in range(n_nodes - 1):
        ax = uv[i, s, 0]
        bx = uv[i, s + 1, 0]
        if ax > _PROJ_INVALID and bx > _PROJ_INVALID:
            dx = bx - ax
            dy = uv[i, s + 1, 1] - uv[i, s, 1]
            total += wp.sqrt(dx * dx + dy * dy)
    return total


@wp.kernel
def _chamfer_axis_kernel(
    edt_q: wp.array2d[wp.int32],  # (crop_h, crop_w) fixed-point EDT
    uv: wp.array3d[wp.float32],  # (n_worlds, n_nodes, 2) pixel coords
    x0: int,
    y0: int,
    crop_w: int,
    crop_h: int,
    max_px: float,
    sim_sum: wp.array[wp.int32],  # (n_worlds,) samples taken
    dist_sum: wp.array[wp.int32],
):  # (n_worlds,) sum EDT over samples
    """sim->goal: distance from the projected cable AXIS to the goal mask.

    Sampling the axis, rather than a rendered silhouette, makes both directions
    relate the same pair of objects and removes the need to render.

    Samples sit at uniform arc length: sample k at (k + 0.5)/N of the total
    visible length, so each represents an equal share of the curve regardless of
    how the capsules foreshorten.
    """
    i, k = wp.tid()
    n_nodes = uv.shape[1]
    total = _axis_length(uv, i, n_nodes)
    if total <= 0.0:
        return  # nothing visible; _chamfer_reduce_kernel charges CHAMFER_MISS
    target = (float(k) + 0.5) / float(_CHAMFER_AXIS_SAMPLES) * total

    # Walk the polyline to the segment containing `target`. n_nodes is ~11, so a
    # linear walk per sample is cheaper than building a prefix table.
    acc = float(0.0)
    px = float(0.0)
    py = float(0.0)
    found = int(0)
    for s in range(n_nodes - 1):
        if found == 0:
            ax = uv[i, s, 0]
            bx = uv[i, s + 1, 0]
            if ax > _PROJ_INVALID and bx > _PROJ_INVALID:
                ay = uv[i, s, 1]
                by = uv[i, s + 1, 1]
                dx = bx - ax
                dy = by - ay
                seg = wp.sqrt(dx * dx + dy * dy)
                if acc + seg >= target:
                    t = float(0.0)
                    if seg > 1.0e-12:
                        t = (target - acc) / seg
                    px = ax + t * dx
                    py = ay + t * dy
                    found = 1
                else:
                    acc += seg
    if found == 0:
        return

    # The EDT is defined on the crop only. A sample projecting outside it takes
    # the clamp rather than being dropped, so a candidate cannot lower its score
    # by steering the cable off-frame.
    c = int(wp.round(px)) - x0
    r = int(wp.round(py)) - y0
    dq = int(wp.round(max_px * _CHAMFER_FIXED))
    if c >= 0 and c < crop_w and r >= 0 and r < crop_h:
        dq = edt_q[r, c]
    wp.atomic_add(sim_sum, i, 1)
    wp.atomic_add(dist_sum, i, dq)


@wp.kernel
def _chamfer_goal_kernel(
    edt_q: wp.array2d[wp.int32],  # (crop_h, crop_w) fixed-point EDT
    uv: wp.array3d[wp.float32],  # (n_worlds, n_nodes, 2) pixel coords
    x0: int,
    y0: int,
    crop_w: int,
    max_px: float,
    gdist_sum: wp.array[wp.int64],
):  # (n_worlds,) sum axis-dist over goal
    """goal->sim: distance from every goal pixel to the projected cable axis."""
    i, r, cb = wp.tid()
    y = y0 + r
    d_goal = int(0)
    n_nodes = uv.shape[1]
    for k in range(_PIXEL_STRIP):
        c = cb * _PIXEL_STRIP + k
        if c < crop_w:
            x = x0 + c
            # A goal pixel is exactly where the goal's own distance transform is 0,
            # so the mask needs no separate upload.
            if edt_q[r, c] == 0:
                best = max_px
                for s in range(n_nodes - 1):
                    ax = uv[i, s, 0]
                    bx = uv[i, s + 1, 0]
                    if ax > _PROJ_INVALID and bx > _PROJ_INVALID:
                        d = _point_seg_dist(float(x), float(y), ax, uv[i, s, 1], bx, uv[i, s + 1, 1])
                        best = wp.min(best, d)
                d_goal += int(wp.round(best * _CHAMFER_FIXED))
    if d_goal > 0:
        wp.atomic_add(gdist_sum, i, wp.int64(d_goal))


@wp.kernel
def _chamfer_reduce_kernel(
    sim_sum: wp.array[wp.int32],
    dist_sum: wp.array[wp.int32],
    gdist_sum: wp.array[wp.int64],
    goal_count: int,
    extent: float,
    miss: float,
    accum: wp.array[wp.float32],
):
    """Add this frame's mean of the two directions, in cable-lengths, to each world."""
    i = wp.tid()
    n_sim = sim_sum[i]
    if n_sim == 0:
        # No axis sample landed, i.e. the whole cable is behind the lens.
        accum[i] = accum[i] + miss
    else:
        s2g = float(dist_sum[i]) / (_CHAMFER_FIXED * float(n_sim))
        g2s = float(gdist_sum[i]) / (_CHAMFER_FIXED * float(goal_count))
        accum[i] = accum[i] + 0.5 * (s2g + g2s) / extent


class ChamferAccumulator:
    """Per-world running chamfer loss plus the scratch counters the kernels fill."""

    def __init__(self, n: int, device: wp.DeviceLike) -> None:
        self.device = device
        self.sim_sum = wp.zeros(n, dtype=wp.int32, device=device)
        self.dist_sum = wp.zeros(n, dtype=wp.int32, device=device)
        self.gdist_sum = wp.zeros(n, dtype=wp.int64, device=device)
        self.loss = wp.zeros(n, dtype=wp.float32, device=device)

    def totals(self) -> list[float]:
        """Per-world summed loss; the single device->host copy of the sequence."""
        return [float(v) for v in self.loss.numpy()]


class _ChamferGoal:
    """Chamfer goal representation: the mask, its distance transform and extent.

    The distance transform is the whole point of preparing once — it is the static
    goal-side work that would otherwise be redone for every candidate.
    """

    __slots__ = ("_dev_edt", "count", "edt", "extent", "mask")

    def __init__(self, mask: np.ndarray) -> None:
        self.mask = mask
        self.count = int(np.count_nonzero(mask))
        if self.count == 0:
            # A distance field to nothing has no meaning: SciPy would measure to a
            # point outside the mask and the frame would pull the cable toward it.
            raise ValueError("chamfer goal mask has no pixels; the goal frame must show the cable.")
        self.edt = _goal_edt(mask)
        self.extent = _mask_extent(mask)
        self._dev_edt: dict[str, wp.array2d[wp.int32]] = {}

    def device_edt(self, device: wp.DeviceLike) -> wp.array2d[wp.int32]:
        key = wp.get_device(device).alias
        if key not in self._dev_edt:
            q = np.rint(self.edt * float(_CHAMFER_FIXED)).astype(np.int32)
            self._dev_edt[key] = wp.array(np.ascontiguousarray(q), dtype=wp.int32, device=device)
        return self._dev_edt[key]


class TuningLossChamfer(TuningLoss):
    """Symmetric chamfer distance between the goal mask and the projected cable axis.

    .. experimental::

    Works from the projected cable alone, so it needs no render. The loss is
    normalized by the goal's extent and does not reach 0 at a perfect fit.
    """

    wants_geometry = True
    wants_render = False
    supports_accum = True

    def prepare(self, goal_mask: np.ndarray) -> _ChamferGoal:
        """Build the goal's distance field and extent.

        Raises:
            ValueError: If ``goal_mask`` has no set pixel.
        """
        return _ChamferGoal(goal_mask)

    def score(
        self,
        goal: _ChamferGoal,
        sim_mask: np.ndarray | None,
        crop: Sequence[int],
        *,
        geom: np.ndarray | None = None,
    ) -> float:
        """Score one world's projected cable axis with the kernels of :meth:`accum` on the CPU.

        ``geom`` is the ``(n_nodes, 2)`` projected cable axis in full-image pixel
        coordinates. ``sim_mask`` is unused.
        """
        if geom is None:
            raise ValueError(
                "the chamfer loss needs the projected cable (geom=); callers supply it for losses with wants_geometry."
            )
        accumulator = ChamferAccumulator(1, "cpu")
        uv = wp.array(np.asarray(geom, dtype=np.float32)[None], dtype=wp.float32, device="cpu")
        self.accum(goal, sim_mask, crop, accumulator, geom=uv)
        return accumulator.totals()[0]

    def make_accum(self, n_worlds: int, device: wp.DeviceLike) -> ChamferAccumulator:
        return ChamferAccumulator(n_worlds, device)

    def accum(
        self,
        goal: _ChamferGoal,
        sim_mask: wp.array | None,
        crop: Sequence[int],
        accumulator: ChamferAccumulator,
        *,
        geom: wp.array3d[wp.float32] | None = None,
    ) -> None:
        """Score every world on-device, adding into ``accumulator``.

        ``geom`` is the ``(n_worlds, n_nodes, 2)`` projected cable axis in full-image
        pixel coordinates, with nodes behind the lens marked by a value below
        ``-1e8``. ``sim_mask`` is unused and may be None -- this loss declares
        ``wants_render=False``, so the caller need not render at all.
        """
        if geom is None:
            raise ValueError("the chamfer loss needs the projected cable (geom=).")
        x0, y0, x1, y1 = (int(v) for v in crop)
        crop_w, crop_h = x1 - x0, y1 - y0
        n = geom.shape[0]
        accumulator.sim_sum.zero_()
        accumulator.dist_sum.zero_()
        accumulator.gdist_sum.zero_()
        edt = goal.device_edt(accumulator.device)
        wp.launch(
            _chamfer_axis_kernel,
            dim=(n, int(_CHAMFER_AXIS_SAMPLES)),
            inputs=[edt, geom, x0, y0, crop_w, crop_h, float(_CHAMFER_MAX_PX)],
            outputs=[accumulator.sim_sum, accumulator.dist_sum],
            device=accumulator.device,
        )
        wp.launch(
            _chamfer_goal_kernel,
            dim=(n, crop_h, -(-crop_w // _PIXEL_STRIP)),
            inputs=[edt, geom, x0, y0, crop_w, float(_CHAMFER_MAX_PX)],
            outputs=[accumulator.gdist_sum],
            device=accumulator.device,
        )
        wp.launch(
            _chamfer_reduce_kernel,
            dim=n,
            inputs=[
                accumulator.sim_sum,
                accumulator.dist_sum,
                accumulator.gdist_sum,
                goal.count,
                float(goal.extent),
                float(CHAMFER_MISS),
            ],
            outputs=[accumulator.loss],
            device=accumulator.device,
        )


CHAMFER = TuningLossChamfer()
"""The shared :class:`TuningLossChamfer` instance."""
