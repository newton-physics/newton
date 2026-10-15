# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Experimental G1 kernel dispatch; opt in with NEWTON_G1_KERNELS=1.

These specializations preserve the scalar implementations as the default and
fallback. They require CUDA, Warp 1.18+, 49 response DOFs, and 43 PD actuators
with one maximum-effort clamp. No solver settings are changed.
"""

import os

import warp as wp


def enabled(device) -> bool:
    """Select the experiment only on supported CUDA runtimes."""
    version = tuple(int(part) for part in wp.__version__.split(".")[:2])
    return os.environ.get("NEWTON_G1_KERNELS") == "1" and version >= (1, 18) and wp.get_device(device).is_cuda


@wp.kernel(enable_backward=False)
def _add_armature(
    starts: wp.array[int],
    ends: wp.array[int],
    qd_starts: wp.array[int],
    mask: wp.array[bool],
    armature: wp.array[float],
    H: wp.array3d[float],
):
    articulation, dof = wp.tid()
    if mask:
        if not mask[articulation]:
            return
    begin = qd_starts[starts[articulation]]
    count = qd_starts[ends[articulation]] - begin
    if dof < count:
        H[articulation, dof, dof] += armature[begin + dof]


def launch_mass(kernel, dim, inputs, outputs, device):
    """Dispatch the small-batch mass kernel after the caller zeros H."""
    J, H = inputs[7], outputs[0]
    if not (
        enabled(device)
        and J.shape[0] == dim == H.shape[0]
        and J.shape[0] <= 2
        and not H.requires_grad
        and not any(isinstance(value, wp.array) and value.requires_grad for value in inputs)
        and J.shape[1:] == (264, 49)
        and H.shape[1:] == (49, 49)
    ):
        return wp.launch(kernel, dim=dim, inputs=inputs, outputs=outputs, device=device)
    from .mass_matrix import fused_mass_matrix  # noqa: PLC0415

    wp.launch_tiled(
        fused_mass_matrix,
        dim=(J.shape[0], 49, 13),
        inputs=inputs[:8],
        outputs=outputs,
        block_dim=64,
        device=device,
    )
    if inputs[8] is not None:
        wp.launch(
            _add_armature,
            dim=(J.shape[0], 49),
            inputs=[inputs[0], inputs[1], inputs[5], inputs[3], inputs[8], H],
            device=device,
        )


def launch_inverse(kernel, dim, inputs, outputs, device):
    """Preserve zero padding and use the tiled 49-DOF inverse when enabled."""
    if not (enabled(device) and inputs[0].shape[1:] == (49, 49)):
        return wp.launch(kernel, dim=dim, inputs=inputs, outputs=outputs, device=device)
    from .inverse_block import inverse_kernel  # noqa: PLC0415

    return wp.launch_tiled(inverse_kernel, dim=dim, inputs=[*inputs, outputs[-1]], block_dim=64, device=device)


def launch_coupled(kernel, dim, inputs, outputs, device, *, compatible=False):
    """Use the PD/max-effort specialization only for a compatible group layout."""
    if not (compatible and enabled(device)):
        return wp.launch(kernel, dim=dim, inputs=inputs, outputs=outputs, device=device)
    from .coupled_solve import solve  # noqa: PLC0415

    return wp.launch(solve, dim=(dim, 64), inputs=inputs[:23], outputs=outputs, block_dim=64, device=device)
