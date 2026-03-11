# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""KAMINO: Utilities: CPU/GPU Warp Device Info"""

from typing import Literal

import warp as wp

###
# Module interface
###

__all__ = [
    "get_device_malloc_info",
    "get_device_spec_info",
]


###
# Functions
###


def _fmt_bytes(bytes: int) -> str:
    """
    Helper function to format a byte value into a human-readable string with appropriate units.

    Args:
        bytes (int):
            The number of bytes to format.
    Returns:
        str:
            A formatted string representing the byte value in appropriate units (bytes, KB, MB, GB, etc.).
    """
    if bytes < 1024:
        return f"{bytes} bytes"
    elif bytes < 1024**2:
        return f"{bytes / 1024:.3f} KB"
    elif bytes < 1024**3:
        return f"{bytes / (1024**2):.3f} MB"
    elif bytes < 1024**4:
        return f"{bytes / (1024**3):.3f} GB"
    else:
        return f"{bytes / (1024**4):.3f} TB"


def get_device_spec_info(device: wp.DeviceLike) -> str:
    """
    Retrieves detailed specifications of a given Warp device as a formatted string.

    Args:
        device (wp.DeviceLike):
            The device for which to retrieve specifications.

    Returns:
        str:
            A formatted string containing detailed specifications for the specified device.
    """
    spec_info: str = f"[device: `{device}`]:\n"
    spec_info += f"                name: {device.name}\n"
    spec_info += f"               alias: {device.alias}\n"
    spec_info += f"                arch: {device.arch}\n"
    spec_info += f"                uuid: {device.uuid}\n"
    spec_info += f"             ordinal: {device.ordinal}\n"
    spec_info += f"          pci_bus_id: {device.pci_bus_id}\n"
    spec_info += f"              is_uva: {device.is_uva}\n"
    spec_info += f"          is_primary: {device.is_primary}\n"
    spec_info += f"  is_cubin_supported: {device.is_cubin_supported}\n"
    spec_info += f"is_mempool_supported: {device.is_mempool_supported}\n"
    spec_info += f"  is_mempool_enabled: {device.is_mempool_enabled}\n"
    spec_info += f"    is_ipc_supported: {device.is_ipc_supported}\n"
    spec_info += f"              is_cpu: {device.is_cpu}\n"
    spec_info += f"             is_cuda: {device.is_cuda}\n"
    spec_info += f"        is_capturing: {device.is_capturing}\n"
    spec_info += f"         has_context: {device.has_context}\n"
    spec_info += f"             context: {device.context}\n"
    spec_info += f"          has_stream: {device.has_stream}\n"
    spec_info += f"            sm_count: {device.sm_count}\n"
    spec_info += f"        total_memory: {device.total_memory} (~{_fmt_bytes(device.total_memory)})\n"
    spec_info += f"         free_memory: {device.free_memory} (~{_fmt_bytes(device.free_memory)})\n"
    return spec_info


def get_device_malloc_info(
    device: wp.DeviceLike,
    usage: Literal["current", "high"] = "current",
    resolution: Literal["auto", "bytes", "MB", "GB"] = "auto",
) -> str:
    """
    Retrieves memory allocation information for the specified device as a formatted string.

    Args:
        device (wp.DeviceLike):
            The device for which to retrieve memory allocation information.

    Returns:
        str:
            A formatted string containing memory allocation information for the specified device.
    """
    # Initialize the info string
    malloc_info: str = f"[device: `{device}`][{usage}]: "

    # Check resolution argument validity
    if resolution not in ["auto", "bytes", "MB", "GB"]:
        raise ValueError(f"Invalid resolution `{resolution}`. Must be one of 'auto', 'bytes', 'MB', or 'GB'.")

    # Get memory allocation info if a CUDA device
    if device.is_cuda:
        if usage == "current":
            mem_used_bytes = wp.get_mempool_used_mem_current(device)
        elif usage == "high":
            mem_used_bytes = wp.get_mempool_used_mem_high(device)
        else:
            raise ValueError(f"Invalid usage `{usage}`. Must be one of 'current' or 'high'.")
        mem_used_mb = float(mem_used_bytes) / (1024**2)
        mem_used_gb = float(mem_used_bytes) / (1024**3)
        if resolution == "bytes":
            malloc_info += f"{mem_used_bytes} bytes"
        elif resolution == "MB":
            malloc_info += f"{mem_used_mb:.3f} MB"
        elif resolution == "GB":
            malloc_info += f"{mem_used_gb:.3f} GB"
        else:  # resolution == "auto"
            malloc_info += f"{_fmt_bytes(mem_used_bytes)}"
    else:
        malloc_info: str = f"[device: `{device}`][{usage}]: ERROR: Device does not support CUDA"

    # Return the formatted memory allocation info
    return malloc_info
