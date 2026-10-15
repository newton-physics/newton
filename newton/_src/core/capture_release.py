# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Releases of CUDA resources that wait until no graph capture is active.

Warp frees device memory capture-safely but destroys textures and pinned host memory immediately, which
invalidates an active stream capture. The garbage collector can finalize either inside any capture.

A release is deferred while any CUDA device reports a Warp-tracked capture. The check and the release are not
atomic with respect to capture start on another thread, which is safe for Warp's default thread-local capture
mode: CUDA restricts only the capturing thread. A global-mode capture begun on another thread during a release,
and captures Warp does not track (started externally on a non-current stream), are not covered.
"""

from __future__ import annotations

import collections
import functools
import gc
from collections.abc import Callable

import warp as wp

# ``deque.popleft`` hands each release to exactly one caller, also across threads and reentrant drains.
_pending_releases: collections.deque[Callable[[], None]] = collections.deque()


def release_deferred() -> None:
    """Run the releases a graph capture deferred, stopping as soon as a capture is active."""
    while _pending_releases and not _any_cuda_device_capturing():
        try:
            release = _pending_releases.popleft()
        except IndexError:
            return
        release()


class CaptureSafeTexture3D(wp.Texture3D):
    """A :class:`warp.Texture3D` whose destruction waits until no graph capture is active."""

    def __del__(self):
        try:
            if self.device.is_cuda:
                # The bound release keeps the texture alive until it runs; Python finalizes an object once.
                _release_or_defer(functools.partial(wp.Texture3D.__del__, self))
                return
        except (TypeError, AttributeError):
            # Interpreter shutdown, as in Warp's own finalizers.
            return
        super().__del__()


def make_pinned_release_capture_safe(array: wp.array) -> wp.array:
    """Make a pinned host array's ``cudaFreeHost`` wait until no graph capture is active."""
    free = array.deleter
    array.deleter = lambda ptr, capacity: _release_or_defer(functools.partial(free, ptr, capacity))
    return array


def _any_cuda_device_capturing() -> bool:
    return any(device.is_capturing for device in wp.get_cuda_devices())


def _release_or_defer(release: Callable[[], None]) -> None:
    if _any_cuda_device_capturing():
        _pending_releases.append(release)
        return
    release()
    release_deferred()


def _release_deferred_after_collection(phase: str, info: dict) -> None:
    # Collections keep happening after a capture ends (e.g. while replaying graphs), so they reclaim deferred releases.
    if phase == "stop" and _pending_releases:
        try:
            release_deferred()
        except (TypeError, AttributeError):
            # Interpreter shutdown, as in Warp's own finalizers.
            pass


gc.callbacks.append(_release_deferred_after_collection)
