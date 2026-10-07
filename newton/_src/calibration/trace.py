# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Write the rendered trace of a run to disk.

A traced iteration is one candidate rolled out with its frames kept (see
:meth:`~.evaluate.CableEvaluator.record`). :class:`CableTraceWriter` writes this
layout, with one ``LABEL`` per camera view:

``iter_NNN_LABEL/``
    Rendered sensor frames of iteration ``NNN``.
``iter_NNN_LABEL_simmask/``
    Binary masks of the visible cable shapes, cropped like the reference masks.
``goal_LABEL.png`` / ``goal_mask_LABEL.png``
    The first reference frame and its mask.
``goal_seq_LABEL/`` / ``goal_seq_LABEL_mask/``
    The reference frame that pairs with each recorded simulation frame, so
    ``iter_NNN_LABEL/frame_k`` and ``goal_seq_LABEL/frame_k`` pair 1:1. They
    are the same in every iteration, so they are written once.
``mean_fit.dat``
    The objective of the traced candidate per iteration.

The reference images come from the evidence bundle. An evaluation does not
decode the RGB frames.

Requires Pillow (``newton[calibration]``).
"""

from __future__ import annotations

import os
from collections.abc import Sequence

import numpy as np


def _save_sequence(frames, out_dir):
    """Write RGB uint8 frames to ``out_dir`` as ``frame_NNN.png``."""
    from PIL import Image

    os.makedirs(out_dir, exist_ok=True)
    for k, frame in enumerate(frames):
        Image.fromarray(np.asarray(frame)[..., :3]).save(os.path.join(out_dir, f"frame_{k:03d}.png"))


def _save_mask_sequence(masks, out_dir):
    """Write single-channel uint8 masks to ``out_dir`` as ``frame_NNN.png``."""
    from PIL import Image

    os.makedirs(out_dir, exist_ok=True)
    for k, mask in enumerate(masks):
        Image.fromarray(np.asarray(mask, dtype=np.uint8)).save(os.path.join(out_dir, f"frame_{k:03d}.png"))


def _read_rgb(path):
    """Read one RGB frame as an ``(H, W, 3)`` uint8 array."""
    from PIL import Image

    with Image.open(path) as img:
        return np.asarray(img.convert("RGB"))


def _read_mask(path):
    """Read one mask as an ``(H, W)`` uint8 array."""
    from PIL import Image

    with Image.open(path) as img:
        return np.asarray(img.convert("L"))


class CableTraceWriter:
    """Write the trace artifacts of a run into one directory.

    Args:
        output_dir: Directory of the trace layout.
        bundle: The :class:`~.evidence.CableEvidenceBundle` of the run, which
            supplies the reference images.
        bundle_dir: Directory that the bundle's relative paths resolve against.
    """

    def __init__(self, output_dir: str, bundle, bundle_dir: str):
        self.output_dir = output_dir
        self.bundle_dir = bundle_dir
        self.recordings = {r.label: r for r in bundle.recordings}
        self._goals_written = False

    def write(self, iteration: int, trace_views: Sequence, objective: float) -> None:
        """Write one traced iteration.

        Args:
            iteration: Iteration number, used in the ``iter_NNN`` directory names.
            trace_views: The :class:`~.evaluate.CableTraceView` entries from
                :meth:`~.evaluate.CableEvaluator.record`.
            objective: The objective of the traced candidate.

        Raises:
            ValueError: If a view has a different number of masks and frames,
                or has no recording in the bundle.
        """
        os.makedirs(self.output_dir, exist_ok=True)
        if not self._goals_written:
            self._write_goal_images(trace_views)
            self._goals_written = True

        for view in trace_views:
            tag = f"_{view.label}"
            _save_sequence(view.frames, os.path.join(self.output_dir, f"iter_{iteration:03d}{tag}"))
            if len(view.masks) != len(view.frames):
                raise ValueError("trace masks must align with rendered frames.")
            x0, y0, x1, y1 = (int(v) for v in view.crop)
            sim_masks = [mask[y0:y1, x0:x1] for mask in view.masks]
            _save_mask_sequence(sim_masks, os.path.join(self.output_dir, f"iter_{iteration:03d}{tag}_simmask"))

        path = os.path.join(self.output_dir, "mean_fit.dat")
        header = not os.path.exists(path)
        with open(path, "a") as fh:
            if header:
                fh.write('% # columns="iteration, mean objective function value"\n')
            fh.write(f"{iteration} {objective}\n")

    def _write_goal_images(self, trace_views):
        """Write the first reference frame and mask, and the reference sequence, of each view."""
        from PIL import Image

        for view in trace_views:
            rec = self.recordings.get(view.label)
            if rec is None:
                raise ValueError(f"traced view '{view.label}' has no recording in the bundle.")
            tag = f"_{view.label}"
            frames = [_read_rgb(os.path.join(self.bundle_dir, p)) for p in rec.rgb_frames]
            masks = [_read_mask(os.path.join(self.bundle_dir, p)) for p in rec.masks]
            Image.fromarray(frames[0]).save(os.path.join(self.output_dir, f"goal{tag}.png"))
            Image.fromarray(masks[0]).save(os.path.join(self.output_dir, f"goal_mask{tag}.png"))
            _save_sequence([frames[j] for j in view.goal_indices], os.path.join(self.output_dir, f"goal_seq{tag}"))
            _save_mask_sequence(
                [masks[j] for j in view.goal_indices], os.path.join(self.output_dir, f"goal_seq{tag}_mask")
            )
