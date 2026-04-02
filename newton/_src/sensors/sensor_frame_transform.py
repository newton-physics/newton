# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Frame Transform Sensor - measures transforms relative to sites."""

import warp as wp

from ..geometry import ShapeFlags
from ..sim.model import Model
from ..sim.state import State
from ..utils.selection import match_labels


@wp.kernel
def compute_shape_transforms_kernel(
    shapes: wp.array[int],
    shape_body: wp.array[int],
    shape_transform: wp.array[wp.transform],
    body_q: wp.array[wp.transform],
    # output
    world_transforms: wp.array[wp.transform],
):
    """Compute world transforms for a list of shape indices.

    Args:
        shapes: Array of shape indices
        shape_body: Model's shape_body array (body parent of each shape)
        shape_transform: Model's shape_transform array (local transforms)
        body_q: State's body_q array (body world transforms)
        world_transforms: Output array for computed world transforms
    """
    tid = wp.tid()
    shape_idx = shapes[tid]

    body_idx = shape_body[shape_idx]
    if body_idx >= 0:
        # Shape attached to a body
        X_wb = body_q[body_idx]
        X_bs = shape_transform[shape_idx]
        world_transforms[shape_idx] = wp.transform_multiply(X_wb, X_bs)
    else:
        # Static shape in world frame
        world_transforms[shape_idx] = shape_transform[shape_idx]


@wp.kernel
def compute_relative_transforms_kernel(
    all_shape_transforms: wp.array[wp.transform],
    shapes: wp.array[int],
    reference_sites: wp.array[int],
    # output
    relative_transforms: wp.array[wp.transform],
):
    """Compute relative transforms expressing object poses in reference frame coordinates.

    Args:
        all_shape_transforms: Array of world transforms for all shapes (indexed by shape index)
        shapes: Indices of target shapes
        reference_sites: Indices of reference sites
        relative_transforms: Output array of relative transforms

    Computes X_ro = X_wr^{-1} * X_wo for each pair, where:
    - X_wo is the world transform of the object shape (object to world)
    - X_wr is the world transform of the reference site (reference to world)
    - X_ro is the transform from object to reference (expresses object pose in reference frame)
    """
    tid = wp.tid()
    shape_idx = shapes[tid]
    ref_idx = reference_sites[tid]

    X_wo = all_shape_transforms[shape_idx]
    X_wr = all_shape_transforms[ref_idx]

    # Compute relative transform: express object pose in reference frame coordinates
    X_ro = wp.transform_multiply(wp.transform_inverse(X_wr), X_wo)
    relative_transforms[tid] = X_ro


class SensorFrameTransform:
    """Sensor that measures transforms of shapes/sites relative to reference sites.

    This sensor computes the relative transform from each reference site to each
    target shape: ``X_ro = inverse(X_wr) * X_wo``, where *X_wo* is the world
    transform of the target, *X_wr* is the world transform of the reference site,
    and *X_ro* expresses the target's pose in the reference frame.

    **Objects** (``shapes``) can be any shape index, including both regular shapes
    and sites. **Reference frames** (``reference_sites``) must be sites (validated
    at initialization). A single reference site broadcasts to all shapes;
    otherwise the counts must match 1:1.

    Attributes:
        transforms: Relative transforms [m, unitless quaternion], shape
            ``(N,)`` (updated after each call to :meth:`update`).

    The ``shapes`` and ``reference_sites`` parameters accept label patterns -- see :ref:`label-matching`.

    Example:
        Measure a shape's pose relative to a reference site:

        .. testcode::

            import warp as wp
            import newton
            from newton.sensors import SensorFrameTransform

            builder = newton.ModelBuilder()
            body = builder.add_body(xform=wp.transform((0, 0, 1), wp.quat_identity()))
            builder.add_shape_box(body, hx=0.1, hy=0.1, hz=0.1, label="box")
            builder.add_site(body, label="ref")
            model = builder.finalize()

            sensor = SensorFrameTransform(model, shapes="box", reference_sites="ref")
            state = model.state()

            sensor.update(state)
            transforms = sensor.transforms.numpy()  # shape: (N, 7)
    """

    def __init__(
        self,
        model: Model,
        shapes: str | list[str] | list[int],
        reference_sites: str | list[str] | list[int],
        *,
        verbose: bool | None = None,
    ):
        """Initialize the SensorFrameTransform.

        Args:
            model: The model to measure.
            shapes: List of shape indices, single pattern to match against shape
                labels, or list of patterns where any one matches.
            reference_sites: List of site indices, single pattern to match against
                site labels, or list of patterns where any one matches. Must expand
                to one site or the same number as ``shapes``.
            verbose: If True, print details. If None, uses ``wp.config.verbose``.

        Raises:
            ValueError: If arguments are invalid or no labels match.
        """
        self.model = model
        self.verbose = verbose if verbose is not None else wp.config.verbose

        # Resolve label patterns to indices
        original_shapes = shapes
        shapes = match_labels(model.shape_label, shapes)
        original_reference_sites = reference_sites
        reference_sites = match_labels(model.shape_label, reference_sites)

        # Validate shape indices
        if not shapes:
            if isinstance(original_shapes, list) and len(original_shapes) == 0:
                raise ValueError("'shapes' must not be empty")
            raise ValueError(f"No shapes matched the given pattern {original_shapes!r}")
        if any(idx < 0 or idx >= model.shape_count for idx in shapes):
            raise ValueError(f"Invalid shape indices. Must be in range [0, {model.shape_count})")

        # Validate reference site indices
        if not reference_sites:
            if isinstance(original_reference_sites, list) and len(original_reference_sites) == 0:
                raise ValueError("'reference_sites' must not be empty")
            raise ValueError(f"No reference sites matched the given pattern {original_reference_sites!r}")
        if any(idx < 0 or idx >= model.shape_count for idx in reference_sites):
            raise ValueError(f"Invalid reference site indices. Must be in range [0, {model.shape_count})")

        # Verify that reference indices are actually sites
        shape_flags = model.shape_flags.numpy()
        for idx in reference_sites:
            if not (shape_flags[idx] & ShapeFlags.SITE):
                raise ValueError(f"Reference index {idx} (label: {model.shape_label[idx]}) is not a site")

        # Handle reference site matching
        if len(reference_sites) == 1:
            # Single reference site for all shapes
            reference_sites_matched = reference_sites * len(shapes)
        elif len(reference_sites) == len(shapes):
            reference_sites_matched = list(reference_sites)
        else:
            raise ValueError(
                f"Number of reference sites ({len(reference_sites)}) must match "
                f"number of shapes ({len(shapes)}) or be 1"
            )

        # Build list of unique shape indices that need transforms computed
        all_indices = set(shapes) | set(reference_sites_matched)
        self._unique_shape_indices = sorted(all_indices)

        # Allocate transform array for all shapes (indexed by shape index)
        # Only the shapes we care about will be computed, rest stay zero
        self._all_shape_transforms = wp.zeros(
            model.shape_count,
            dtype=wp.transform,
            device=model.device,
        )

        # Allocate output array
        self.transforms = wp.zeros(
            len(shapes),
            dtype=wp.transform,
            device=model.device,
        )

        # Convert indices to warp arrays (done once at init)
        self._unique_indices_arr = wp.array(self._unique_shape_indices, dtype=int, device=model.device)
        self._shape_indices_arr = wp.array(shapes, dtype=int, device=model.device)
        self._reference_indices_arr = wp.array(reference_sites_matched, dtype=int, device=model.device)

        if self.verbose:
            print("SensorFrameTransform initialized:")
            print(f"  Shapes: {len(shapes)}")
            print(f"  Reference sites: {len(set(reference_sites_matched))} unique")
            print(
                f"  Unique shapes to compute: {len(self._unique_shape_indices)} (optimized from {len(shapes) + len(reference_sites_matched)})"
            )

    def update(self, state: State):
        """Update sensor measurements based on current state.

        Reads ``state.body_q`` to compute world-frame shape transforms.

        Args:
            state: The current state. Reads ``body_q``, which is updated by a solver step or :func:`~newton.eval_fk`.
        """
        # Compute world transforms for all unique shapes directly into the all_shape_transforms array
        wp.launch(
            compute_shape_transforms_kernel,
            dim=len(self._unique_shape_indices),
            inputs=[self._unique_indices_arr, self.model.shape_body, self.model.shape_transform, state.body_q],
            outputs=[self._all_shape_transforms],
            device=self.model.device,
        )

        # Compute relative transforms by indexing directly into all_shape_transforms
        wp.launch(
            compute_relative_transforms_kernel,
            dim=len(self._shape_indices_arr),
            inputs=[self._all_shape_transforms, self._shape_indices_arr, self._reference_indices_arr],
            outputs=[self.transforms],
            device=self.model.device,
        )
