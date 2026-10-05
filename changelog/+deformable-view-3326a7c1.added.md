Add experimental `newton.selection.DeformableCurveView`, `DeformableSurfaceView`, and `DeformableVolumeView` for label-pattern access to finalized deformable objects, including batched state reads, masked full-buffer updates with gradient support, uneven per-world partitions, and raw ranges for deformable objects with different element counts. Use the same selection API for native builder calls and USD imports.

Expose experimental per-family labels, world indices, counts, and simulation ranges on `Model` for inspection without a selection view.

Update values must not share storage with the target array. Use `wp.clone()` to copy a live getter result before passing it back to a setter.

Convert selected-object indices to a caller-owned Boolean mask with `set_mask_from_indices()`. Values remain in view order, and preallocated device inputs support CUDA graph capture.
