# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

""":func:`export_controller_graph` — save a controller's compute graph for the C++ runtime.

The saved artifact is a Warp APIC graph: the kernel launches of one
:meth:`~newton.controllers.ControllerBase.step` call, recorded with the device
addresses they ran against, plus a table of named parameters the host can read
and write. ``newton/_src/controllers/cpp`` loads one and steps it without a
Python interpreter.
"""

from __future__ import annotations

import ctypes
from pathlib import Path
from typing import TYPE_CHECKING, Any

import warp as wp

if TYPE_CHECKING:
    from .controller import ControllerBase

# The C++ runtime addresses every parameter as a buffer of float32, sizing it
# from the parameter's byte count. Warp exposes the underlying ctypes scalar as
# ``_type_`` on scalar and compound types alike, so this one check covers
# float32 and everything built from it (vec3, quat, transform, mat33, ...).
_FLOAT32_CTYPE = ctypes.c_float


def export_controller_graph(
    *,
    controller: ControllerBase,
    inputs: Any,
    outputs: Any,
    path: str | Path,
) -> None:
    """Capture one controller step and save it as a ``.wrp`` graph artifact.

    Every :class:`~warp.array` field of ``inputs`` and ``outputs`` becomes a
    named graph parameter — ``input.<field>`` and ``output.<field>`` — matching
    the prefixes the C++ runtime looks for. A ``dt`` parameter is added for the
    host to write each step.

    Warp appends the ``.wrp`` extension and writes a sibling ``<path>_modules``
    directory holding the compiled kernels, so both must ship together.

    **Ports bound to a view.** A view has no device address of its own, so what
    is registered is the simulation-sized array it reads or writes through, and
    the indices travel inside the graph. The host exchanges that whole array:
    it writes every element of such an input, of which the graph reads the
    addressed ones, and reads back every element of such an output, of which the
    graph has written the addressed ones and left the rest at their capture-time
    values. A port bound to a plain array keeps its own length, so a struct
    mixing the two yields parameters of both sizes; the C++ ``params()`` reports
    the length of each.

    Args:
        controller: Controller to capture. Must report
            :meth:`~newton.controllers.ControllerBase.is_graphable`.
        inputs: Populated input struct, typically from ``controller.input()``.
            Bind every port before calling: capture records the addresses these
            arrays have now.
        outputs: Output struct, typically from ``controller.output()``.
        path: Destination stem. ``<path>.wrp`` and ``<path>_modules`` are written.

    Raises:
        RuntimeError: If the controller is not graphable.
        TypeError: If a port's dtype is not composed of float32.
    """
    if not controller.is_graphable():
        raise RuntimeError(f"{type(controller).__name__}.is_graphable() is False, so its step cannot be captured.")

    device = _port_device(inputs)

    params: dict[str, wp.array] = {}
    for prefix, struct in (("input", inputs), ("output", outputs)):
        for field, port in vars(struct).items():
            if isinstance(port, wp.indexedarray):
                _check_dtype(f"{prefix}.{field}", port)
                # A view has no device pointer of its own; the graph reads and
                # writes through the array underneath it.
                params[f"{prefix}.{field}"] = port.data
            elif isinstance(port, wp.array):
                _check_dtype(f"{prefix}.{field}", port)
                params[f"{prefix}.{field}"] = port

    # Allocated before the capture so the recorded launches hold this buffer's
    # address; the host writes it through the "dt" parameter each step.
    dt = wp.zeros(1, dtype=wp.float32, device=device)

    # Run once outside the capture so lazy kernel compilation does not happen
    # inside it.
    controller.step(inputs=inputs, outputs=outputs, dt=dt)
    wp.synchronize_device(device)

    with wp.ScopedCapture(device=device, apic=True, force_module_load=False) as capture:
        controller.step(inputs=inputs, outputs=outputs, dt=dt)

    wp.capture_save(
        capture.graph,
        str(path),
        inputs={name: array for name, array in params.items() if name.startswith("input.")} | {"dt": dt},
        outputs={name: array for name, array in params.items() if name.startswith("output.")},
    )


def _check_dtype(name: str, port: wp.array | wp.indexedarray) -> None:
    scalar = getattr(port.dtype, "_type_", None)
    if scalar is not _FLOAT32_CTYPE:
        raise TypeError(
            f"{name} has dtype {port.dtype.__name__}, which is not composed of float32. The C++ runtime "
            f"reads every parameter as float32, so only float32 and types built from it (vec3, quat, "
            f"transform, mat33, ...) can be exported."
        )


def _port_device(struct: Any) -> Any:
    for port in vars(struct).values():
        if isinstance(port, wp.array | wp.indexedarray):
            return port.device
    return wp.get_device()
