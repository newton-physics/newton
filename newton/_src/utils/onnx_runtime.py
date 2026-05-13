# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Graph-capturable ONNX inference runtime for Newton policy networks.

Only the ``onnx`` package (pure protobuf parser) is required -- no
``onnxruntime`` or ``torch``.  Weights are loaded once onto the target
Warp device; inference executes a pre-built list of lightweight op
descriptors that dispatch to Warp kernels without host round-trips or
device allocation.

Supported ONNX operators (all graph-capturable after one warmup call):

* **Gemm** -- ``C = alpha * A @ B.T + beta * bias`` with ``transB=1``
* **Elu** -- element-wise activation
* **Squeeze** -- alias passthrough (the output array shares memory with the
  input).  Only used to drop unit dims, no copy is performed.
* **LSTM** -- forward, single-direction, single-layer, ``seq_length=1``.  The
  full step (gate GEMM + cell update) executes in two on-device kernels.
  This is the layout produced by single-step policy LSTMs.

Example::

    from newton.utils import OnnxRuntime

    rt = OnnxRuntime("policy.onnx", device="cuda:0")
    out = rt({"observation": wp.array2d(obs, dtype=wp.float32, device="cuda:0")})
    actions = out["action"]
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import warp as wp


def _require_onnx():
    """Lazy import of the ``onnx`` package with a friendly error message.

    Returns ``(onnx, numpy_helper)``.  ``onnx`` is an optional dependency;
    importing it eagerly at module load would force every Newton install to
    pull protobuf and ml-dtypes even when the user never loads a policy.
    """
    try:
        import onnx
        from onnx import numpy_helper
    except ImportError as exc:  # pragma: no cover - exercised only on missing dep
        raise ImportError(
            "OnnxRuntime requires the optional `onnx` package. "
            "Install it with `pip install onnx>=1.16.0` or `pip install newton[onnx]`."
        ) from exc
    return onnx, numpy_helper


# ---------------------------------------------------------------------------
# Warp kernels
# ---------------------------------------------------------------------------
#
# These are simple per-output-element kernels: one thread writes one cell.
# Policies seen in practice are tiny (batch=1, hidden<=128), so the tiled /
# tensor-core variants are unnecessary and only add nvJitLink/LTO compile
# pressure that races with mujoco_warp's tile_cholesky build under heavy
# parallel test runs.


@wp.kernel
def _gemm_transb_kernel(
    A: wp.array2d[float],  # (M, K)
    B: wp.array2d[float],  # (N, K) — stored transposed
    bias: wp.array[float],  # (N,)
    C: wp.array2d[float],  # (M, N)
    K: int,
    alpha: float,
    beta: float,
):
    """``C = alpha * A @ B.T + beta * bias`` with ``transB=1``."""
    i, j = wp.tid()

    s = float(0.0)
    for k in range(K):
        s += A[i, k] * B[j, k]

    C[i, j] = alpha * s + beta * bias[j]


@wp.kernel
def _elu_kernel(
    x: wp.array2d[float],
    y: wp.array2d[float],
    alpha: float,
):
    i, j = wp.tid()
    v = x[i, j]
    y[i, j] = wp.where(v >= 0.0, v, alpha * (wp.exp(v) - 1.0))


# ---------------------------------------------------------------------------
# LSTM cell.  Two on-device kernels: a thread-per-gate GEMM that computes the
# four gates ``x @ W.T + h_prev @ R.T`` into a (batch, 4*hidden) workspace,
# and a pointwise update that adds the biases, applies sigmoid/tanh, and
# writes ``h_out`` / ``c_out``.  Both are graph-capturable after warmup.
# ---------------------------------------------------------------------------


@wp.kernel
def _lstm_gates_kernel(
    x: wp.array2d[float],  # (batch, input_size)
    h_prev: wp.array2d[float],  # (batch, hidden_size)
    W: wp.array2d[float],  # (4*hidden_size, input_size)
    R: wp.array2d[float],  # (4*hidden_size, hidden_size)
    gates: wp.array2d[float],  # (batch, 4*hidden_size) output
    input_size: int,
    hidden_size: int,
):
    """``gates = x @ W.T + h_prev @ R.T`` (one thread per (batch, gate))."""
    b, j = wp.tid()

    s = float(0.0)
    for k in range(input_size):
        s += x[b, k] * W[j, k]
    for k in range(hidden_size):
        s += h_prev[b, k] * R[j, k]

    gates[b, j] = s


@wp.kernel
def _lstm_cell_update_kernel(
    gates: wp.array2d[float],  # (batch, 4*hidden_size); already x@W.T + h_prev@R.T
    c_prev: wp.array2d[float],  # (batch, hidden_size)
    Bx: wp.array[float],  # (4*hidden_size,)
    Bh: wp.array[float],  # (4*hidden_size,)
    h_out: wp.array2d[float],  # (batch, hidden_size)
    c_out: wp.array2d[float],  # (batch, hidden_size)
    hidden_size: int,
):
    b, h = wp.tid()

    s_i = gates[b, 0 * hidden_size + h] + Bx[0 * hidden_size + h] + Bh[0 * hidden_size + h]
    s_o = gates[b, 1 * hidden_size + h] + Bx[1 * hidden_size + h] + Bh[1 * hidden_size + h]
    s_f = gates[b, 2 * hidden_size + h] + Bx[2 * hidden_size + h] + Bh[2 * hidden_size + h]
    s_c = gates[b, 3 * hidden_size + h] + Bx[3 * hidden_size + h] + Bh[3 * hidden_size + h]

    g_i = 1.0 / (1.0 + wp.exp(-s_i))
    g_o = 1.0 / (1.0 + wp.exp(-s_o))
    g_f = 1.0 / (1.0 + wp.exp(-s_f))
    g_c = wp.tanh(s_c)

    c_new = g_f * c_prev[b, h] + g_i * g_c
    c_out[b, h] = c_new
    h_out[b, h] = g_o * wp.tanh(c_new)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


_ATTR_DECODERS = {
    1: lambda a: a.f,  # FLOAT
    2: lambda a: a.i,  # INT
    3: lambda a: a.s.decode("utf-8") if isinstance(a.s, (bytes, bytearray)) else a.s,  # STRING
}


@dataclass
class _Op:
    op_type: str
    inputs: list[str]
    outputs: list[str]
    attrs: dict[str, Any] = field(default_factory=dict)
    attr_names: set[str] = field(default_factory=set)


def _decode_attrs(node) -> tuple[dict[str, Any], set[str]]:
    """Decode all attributes of an ONNX ``NodeProto`` in a single pass.

    Returns ``(decoded, all_names)`` where ``decoded`` only contains attributes
    of supported types (FLOAT/INT/STRING) and ``all_names`` is the full set of
    attribute names present on the node (used for fail-fast validation of
    unsupported features even when their value type isn't decoded).
    """
    out: dict[str, Any] = {}
    all_names: set[str] = set()
    for attr in node.attribute:
        all_names.add(attr.name)
        decoder = _ATTR_DECODERS.get(attr.type)
        if decoder is not None:
            out[attr.name] = decoder(attr)
    return out, all_names


def _np_to_warp(arr_np: np.ndarray, device: wp.context.Device) -> wp.array:
    arr_np = np.ascontiguousarray(arr_np, dtype=np.float32)
    return wp.array(arr_np, dtype=wp.float32, device=device)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


class OnnxRuntime:
    """Lightweight ONNX inference engine for graph-capturable MLP policies.

    Args:
        path: Path to an ``.onnx`` file.
        device: Warp device string (e.g. ``"cuda:0"``).  ``None`` uses the
            current default device.
        batch_size: Fixed batch dimension used to pre-allocate intermediate
            buffers.  Defaults to ``1``.
        input_batch_axes: Optional batch-axis override for graph inputs.  If
            an integer is provided, it is applied to every graph input; if a
            dictionary is provided, it maps graph input names to their batch
            axis.  The selected axes are replaced with ``batch_size`` even
            when the ONNX model exported them as fixed dimensions.
    """

    def __init__(
        self,
        path: str,
        device: str | None = None,
        batch_size: int = 1,
        input_batch_axes: int | dict[str, int] | None = None,
    ):
        self._device = wp.get_device(device)

        onnx, numpy_helper = _require_onnx()
        model = onnx.load(path)
        graph = model.graph

        self._tensors: dict[str, wp.array] = {}
        self._shapes: dict[str, tuple[int, ...]] = {}

        for init in graph.initializer:
            arr_np = numpy_helper.to_array(init).astype(np.float32)
            self._tensors[init.name] = _np_to_warp(arr_np, self._device)
            self._shapes[init.name] = tuple(arr_np.shape)

        initializer_names = {init.name for init in graph.initializer}
        self.input_names: list[str] = [inp.name for inp in graph.input if inp.name not in initializer_names]
        self.output_names: list[str] = [out.name for out in graph.output]

        if isinstance(input_batch_axes, dict):
            unknown_inputs = set(input_batch_axes) - set(self.input_names)
            if unknown_inputs:
                raise KeyError(
                    f"OnnxRuntime: input_batch_axes references unknown graph inputs {sorted(unknown_inputs)}"
                )

        for inp in graph.input:
            if inp.name in initializer_names:
                continue
            dims = list(inp.type.tensor_type.shape.dim)
            batch_axis = None
            if input_batch_axes is not None:
                if isinstance(input_batch_axes, dict):
                    batch_axis = input_batch_axes.get(inp.name)
                else:
                    batch_axis = input_batch_axes
                if batch_axis is not None:
                    if batch_axis < 0:
                        batch_axis += len(dims)
                    if batch_axis < 0 or batch_axis >= len(dims):
                        raise ValueError(
                            f"OnnxRuntime: input '{inp.name}' batch axis {batch_axis} is out of range "
                            f"for rank-{len(dims)} input"
                        )
            shape = []
            for axis, d in enumerate(dims):
                if axis == batch_axis:
                    shape.append(batch_size)
                elif d.HasField("dim_value") and d.dim_value > 0:
                    shape.append(d.dim_value)
                else:
                    shape.append(batch_size)
            self._shapes[inp.name] = tuple(shape)

        self._ops: list[_Op] = []
        for node in graph.node:
            decoded, all_names = _decode_attrs(node)
            self._ops.append(
                _Op(
                    op_type=node.op_type,
                    inputs=list(node.input),
                    outputs=list(node.output),
                    attrs=decoded,
                    attr_names=all_names,
                )
            )

        self._preallocate_buffers()

    # ------------------------------------------------------------------
    # Buffer pre-allocation
    # ------------------------------------------------------------------

    def _preallocate_buffers(self) -> None:
        for op in self._ops:
            handler = _SHAPE_DISPATCH.get(op.op_type)
            if handler is None:
                raise NotImplementedError(
                    f"OnnxRuntime: unsupported op '{op.op_type}'.  Supported ops: {sorted(_OP_DISPATCH.keys())}"
                )
            handler(op, self._shapes, self._tensors, self._device)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def __call__(self, inputs: dict[str, wp.array]) -> dict[str, wp.array]:
        """Run forward inference.

        Args:
            inputs: Mapping of ONNX input names to Warp arrays already on
                the correct device.  2-D ``wp.array2d`` is the typical case.

        Returns:
            Mapping of ONNX output names to Warp result arrays.
        """
        tensors = self._tensors

        # Restrict provided keys to the declared graph inputs so callers
        # cannot overwrite initializers (weights) or internal tensors by
        # passing a matching name.
        declared_inputs = set(self.input_names)
        for name in inputs:
            if name not in declared_inputs:
                raise KeyError(f"OnnxRuntime: unknown input '{name}'")

        for name in self.input_names:
            if name not in inputs:
                raise KeyError(f"OnnxRuntime: missing input '{name}'")
            arr = inputs[name]
            expected_shape = self._shapes[name]
            if tuple(arr.shape) != expected_shape:
                raise ValueError(f"OnnxRuntime: input '{name}' has shape {tuple(arr.shape)}, expected {expected_shape}")
            tensors[name] = arr

        for op in self._ops:
            dispatch = _OP_DISPATCH.get(op.op_type)
            if dispatch is None:
                raise NotImplementedError(f"OnnxRuntime: unsupported op '{op.op_type}'")
            dispatch(op, tensors, self._shapes, self._device)

        return {name: tensors[name] for name in self.output_names}


# ---------------------------------------------------------------------------
# Shape inference (per op)
# ---------------------------------------------------------------------------


def _shape_gemm(op, shapes, tensors, device):
    A_shape = shapes[op.inputs[0]]
    B_shape = shapes[op.inputs[1]]
    transA = int(op.attrs.get("transA", 0))
    transB = int(op.attrs.get("transB", 0))
    if transA:
        raise NotImplementedError("OnnxRuntime Gemm: transA=1 is not graph-capturable in this runtime")
    if transB != 1:
        raise NotImplementedError("OnnxRuntime Gemm: only transB=1 policy weights are supported")
    if len(op.inputs) < 3 or not op.inputs[2]:
        raise NotImplementedError("OnnxRuntime Gemm: bias input is required for graph-capturable policy execution")
    if len(A_shape) != 2 or len(B_shape) != 2:
        raise NotImplementedError("OnnxRuntime Gemm: only 2-D tensors are supported")
    M = A_shape[0]
    N = B_shape[0]
    K = A_shape[1]
    if B_shape[1] != K:
        raise ValueError(f"OnnxRuntime Gemm: incompatible shapes {A_shape} and {B_shape}")
    bias_shape = shapes[op.inputs[2]]
    if bias_shape != (N,):
        raise ValueError(f"OnnxRuntime Gemm: bias '{op.inputs[2]}' has shape {bias_shape}, expected {(N,)}")
    out_shape = (M, N)
    out_name = op.outputs[0]
    if out_name not in tensors:
        tensors[out_name] = wp.zeros(out_shape, dtype=wp.float32, device=device)
    shapes[out_name] = out_shape


def _shape_elementwise_unary(op, shapes, tensors, device):
    in_shape = shapes[op.inputs[0]]
    if len(in_shape) != 2:
        raise NotImplementedError("OnnxRuntime Elu: only 2-D tensors are supported")
    out_name = op.outputs[0]
    if out_name not in tensors:
        tensors[out_name] = wp.zeros(in_shape, dtype=wp.float32, device=device)
    shapes[out_name] = in_shape


def _shape_squeeze(op, shapes, tensors, device):
    """Squeeze is implemented as a no-copy alias on a 2-D Warp buffer.

    The input is expected to be a higher-rank tensor whose unit dims will be
    removed by the ``axes`` initializer; the resulting layout must be 2-D and
    contiguous, which matches the LSTM ``Y -> Y_2d`` decoder pattern.
    """
    in_shape = shapes[op.inputs[0]]
    axes = None
    if len(op.inputs) > 1 and op.inputs[1] in tensors:
        axes_tensor = tensors[op.inputs[1]]
        if hasattr(axes_tensor, "numpy"):
            axes = [int(v) for v in axes_tensor.numpy().tolist()]
    if axes is None:
        out_shape = tuple(d for d in in_shape if d != 1)
    else:
        rank = len(in_shape)
        axes_norm = {a if a >= 0 else a + rank for a in axes}
        out_shape = tuple(d for i, d in enumerate(in_shape) if i not in axes_norm)
    if len(out_shape) != 2:
        raise NotImplementedError(
            f"OnnxRuntime Squeeze: only squeezes that produce a 2-D tensor are supported (got {out_shape})"
        )
    shapes[op.outputs[0]] = out_shape
    op.attrs["_out_shape"] = out_shape


def _shape_lstm(op, shapes, tensors, device):
    """Single-step (seq_length=1), single-direction, single-layer LSTM.

    Weights are reshaped into device-resident 2-D tiles and the gates
    workspace is preallocated, so per-call inference is two Warp launches.
    """
    # The kernel hardcodes ONNX-default activations (sigmoid, tanh, tanh) and
    # has no peephole / clip / input_forget / sequence_lens support.  Reject
    # any model that asks for non-default behavior so we never silently
    # produce wrong inferences.  ``activations`` / ``activation_alpha`` /
    # ``activation_beta`` are rejected on presence (we don't decode them and
    # any explicit setting overrides the defaults), but ``clip`` and
    # ``input_forget`` are checked by value: many exporters serialize their
    # default values (``clip=0.0``, ``input_forget=0``) explicitly, and those
    # match our hardcoded behavior so we should accept them.
    for unsupported in ("activations", "activation_alpha", "activation_beta"):
        if unsupported in op.attr_names:
            raise NotImplementedError(
                f"OnnxRuntime LSTM: attribute '{unsupported}' is not supported "
                f"(only default sigmoid/tanh/tanh activations)"
            )
    if op.attrs.get("clip", 0.0):
        raise NotImplementedError(
            f"OnnxRuntime LSTM: non-default 'clip' attribute is not supported (got {op.attrs['clip']})"
        )
    if op.attrs.get("input_forget", 0):
        raise NotImplementedError(
            f"OnnxRuntime LSTM: non-default 'input_forget' attribute is not supported (got {op.attrs['input_forget']})"
        )

    # sequence_lens (input index 4) and P / peepholes (input index 7) are
    # optional ONNX inputs.  Empty strings denote "not provided" in ONNX.
    if len(op.inputs) > 4 and op.inputs[4]:
        raise NotImplementedError("OnnxRuntime LSTM: 'sequence_lens' input is not supported")
    if len(op.inputs) > 7 and op.inputs[7]:
        raise NotImplementedError("OnnxRuntime LSTM: peephole input 'P' is not supported")

    direction = op.attrs.get("direction", "forward")
    if direction not in ("forward", b"forward"):
        raise NotImplementedError("OnnxRuntime LSTM: only forward direction is supported")

    layout = int(op.attrs.get("layout", 0))
    if layout != 0:
        # layout=1 would require batch-major state shapes (batch, num_directions,
        # hidden_size) for both the optional input states and the Yh/Yc output
        # states, but the rest of this implementation hardcodes direction-major
        # ((num_directions, batch, hidden_size)) shapes.  Reject layout=1 up
        # front rather than silently producing incorrectly shaped state tensors.
        raise NotImplementedError("OnnxRuntime LSTM: layout must be 0 (layout=1 not supported)")

    X_shape = shapes[op.inputs[0]]
    if len(X_shape) != 3:
        raise NotImplementedError("OnnxRuntime LSTM: input X must be 3-D")
    if layout == 0:
        seq_len, batch, input_size = X_shape
    else:
        batch, seq_len, input_size = X_shape
    if seq_len != 1:
        raise NotImplementedError("OnnxRuntime LSTM: only seq_length=1 is supported (single-step inference)")

    W_shape = shapes[op.inputs[1]]
    if len(W_shape) != 3 or W_shape[0] != 1:
        raise NotImplementedError("OnnxRuntime LSTM: only num_directions=1 is supported")
    hidden_size = int(op.attrs.get("hidden_size", W_shape[1] // 4))

    if W_shape != (1, 4 * hidden_size, input_size):
        raise ValueError(f"OnnxRuntime LSTM: W has shape {W_shape}, expected {(1, 4 * hidden_size, input_size)}")

    R_shape = shapes[op.inputs[2]]
    if R_shape != (1, 4 * hidden_size, hidden_size):
        raise ValueError(f"OnnxRuntime LSTM: R has shape {R_shape}, expected {(1, 4 * hidden_size, hidden_size)}")

    # Materialize reshaped W/R/B as device-resident 2-D arrays.  The
    # initializers are already on device; use ``reshape`` to produce 2-D
    # views without re-uploading.
    W_full = tensors[op.inputs[1]]
    R_full = tensors[op.inputs[2]]
    cache: dict[str, wp.array] = {}
    cache["W"] = W_full.reshape((4 * hidden_size, input_size))
    cache["R"] = R_full.reshape((4 * hidden_size, hidden_size))

    if len(op.inputs) > 3 and op.inputs[3] and op.inputs[3] in tensors:
        B_full = tensors[op.inputs[3]]
        B_shape_in = shapes[op.inputs[3]]
        if B_shape_in != (1, 8 * hidden_size):
            raise ValueError(f"OnnxRuntime LSTM: B has shape {B_shape_in}, expected {(1, 8 * hidden_size)}")
        B_2d = B_full.reshape((8 * hidden_size,))
        # Slice into the two 4H halves once at preallocation time.  Warp
        # array slices are zero-copy views; per-call inference reuses them.
        cache["Bx"] = B_2d[: 4 * hidden_size]
        cache["Bh"] = B_2d[4 * hidden_size :]
    else:
        cache["Bx"] = wp.zeros(4 * hidden_size, dtype=wp.float32, device=device)
        cache["Bh"] = wp.zeros(4 * hidden_size, dtype=wp.float32, device=device)

    cache["gates"] = wp.zeros((batch, 4 * hidden_size), dtype=wp.float32, device=device)
    cache["input_size"] = input_size
    cache["hidden_size"] = hidden_size
    cache["batch"] = batch
    cache["layout"] = layout
    op.attrs["_cache"] = cache

    # Output Y has shape (seq, num_directions, batch, hidden) for layout=0
    # and (batch, seq, num_directions, hidden) for layout=1.  We allocate a
    # 2-D ``(batch, hidden)`` buffer that the kernel writes directly, plus a
    # 4-D shape entry so downstream ops (like Squeeze) see the ONNX shape.
    h_buf = wp.zeros((batch, hidden_size), dtype=wp.float32, device=device)
    c_buf = wp.zeros((batch, hidden_size), dtype=wp.float32, device=device)
    cache["h_out"] = h_buf
    cache["c_out"] = c_buf

    if layout == 0:
        Y_shape = (1, 1, batch, hidden_size)
    else:
        Y_shape = (batch, 1, 1, hidden_size)
    Yh_shape = (1, batch, hidden_size)

    if len(op.outputs) > 0 and op.outputs[0]:
        # Y is just the (batch, hidden) buffer with extra unit dims.  Store
        # the same Warp buffer reshaped to the ONNX-expected rank so a
        # following Squeeze can alias it back to 2-D for free.
        tensors[op.outputs[0]] = h_buf.reshape(Y_shape)
        shapes[op.outputs[0]] = Y_shape
    if len(op.outputs) > 1 and op.outputs[1]:
        tensors[op.outputs[1]] = h_buf.reshape(Yh_shape)
        shapes[op.outputs[1]] = Yh_shape
    if len(op.outputs) > 2 and op.outputs[2]:
        tensors[op.outputs[2]] = c_buf.reshape(Yh_shape)
        shapes[op.outputs[2]] = Yh_shape


# ---------------------------------------------------------------------------
# Op implementations
# ---------------------------------------------------------------------------


def _exec_gemm(op, tensors, shapes, device):
    A = tensors[op.inputs[0]]
    B = tensors[op.inputs[1]]
    bias = tensors[op.inputs[2]]
    alpha = float(op.attrs.get("alpha", 1.0))
    beta = float(op.attrs.get("beta", 1.0))

    A_shape = shapes[op.inputs[0]]
    B_shape = shapes[op.inputs[1]]
    M = A_shape[0]
    N = B_shape[0]
    K = B_shape[1]

    out = tensors[op.outputs[0]]
    wp.launch(
        _gemm_transb_kernel,
        dim=(M, N),
        inputs=[A, B, bias, out, K, alpha, beta],
        device=device,
    )


def _exec_elu(op, tensors, shapes, device):
    x = tensors[op.inputs[0]]
    alpha = float(op.attrs.get("alpha", 1.0))
    out = tensors[op.outputs[0]]
    shape = shapes[op.inputs[0]]
    wp.launch(_elu_kernel, dim=shape, inputs=[x, out, alpha], device=device)


def _exec_squeeze(op, tensors, shapes, device):
    """Squeeze is an alias: the output Warp buffer reuses the input's storage.

    The runtime shape table is updated to reflect the squeezed rank, so
    downstream ops see a 2-D tensor without any data movement.
    """
    src = tensors[op.inputs[0]]
    out_shape = op.attrs["_out_shape"]
    tensors[op.outputs[0]] = src.reshape(out_shape)
    shapes[op.outputs[0]] = out_shape


def _exec_lstm(op, tensors, shapes, device):
    """Run one LSTM step entirely on device.

    No host transfers, no allocation -- just two Warp launches into
    preallocated buffers, so this path is graph-capturable after warmup.
    """
    cache = op.attrs["_cache"]
    input_size: int = cache["input_size"]
    hidden_size: int = cache["hidden_size"]
    batch: int = cache["batch"]
    layout: int = cache["layout"]

    X = tensors[op.inputs[0]]
    if layout == 0:
        # X has shape (1, batch, input_size); reshape to (batch, input_size).
        x_t = X.reshape((batch, input_size))
    else:
        # X has shape (batch, 1, input_size); reshape similarly.
        x_t = X.reshape((batch, input_size))

    # Initial hidden / cell states are optional but always provided by
    # ``ControllerNeuralLSTM``.  Reshape from (num_dirs, batch, hidden) to
    # (batch, hidden) on the fly without copying.
    if len(op.inputs) > 5 and op.inputs[5] and op.inputs[5] in tensors:
        h_prev = tensors[op.inputs[5]].reshape((batch, hidden_size))
    else:
        if "h_prev_zero" not in cache:
            cache["h_prev_zero"] = wp.zeros((batch, hidden_size), dtype=wp.float32, device=device)
        h_prev = cache["h_prev_zero"]
    if len(op.inputs) > 6 and op.inputs[6] and op.inputs[6] in tensors:
        c_prev = tensors[op.inputs[6]].reshape((batch, hidden_size))
    else:
        if "c_prev_zero" not in cache:
            cache["c_prev_zero"] = wp.zeros((batch, hidden_size), dtype=wp.float32, device=device)
        c_prev = cache["c_prev_zero"]

    gates = cache["gates"]
    h_out = cache["h_out"]
    c_out = cache["c_out"]

    wp.launch(
        _lstm_gates_kernel,
        dim=(batch, 4 * hidden_size),
        inputs=[x_t, h_prev, cache["W"], cache["R"], gates, input_size, hidden_size],
        device=device,
    )
    wp.launch(
        _lstm_cell_update_kernel,
        dim=(batch, hidden_size),
        inputs=[gates, c_prev, cache["Bx"], cache["Bh"], h_out, c_out, hidden_size],
        device=device,
    )


# ---------------------------------------------------------------------------
# Dispatch tables
# ---------------------------------------------------------------------------


_OP_DISPATCH: dict[str, Any] = {
    "Gemm": _exec_gemm,
    "Elu": _exec_elu,
    "Squeeze": _exec_squeeze,
    "LSTM": _exec_lstm,
}

_SHAPE_DISPATCH: dict[str, Any] = {
    "Gemm": _shape_gemm,
    "Elu": _shape_elementwise_unary,
    "Squeeze": _shape_squeeze,
    "LSTM": _shape_lstm,
}
