#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Print a short hash of the host CPU's ISA feature set.

Warp 1.13+ compiles CPU kernels with ``-march=native``, so cached object
files are only safe to reuse on a CPU with the same instruction set.
This script detects the ISA features and prints a 16-character hex hash
suitable for use in a CI cache key.

Supported platforms:
  - x86_64 Linux / macOS / Windows (via system C compiler)
  - AArch64 Linux (via /proc/cpuinfo)
  - AArch64 macOS (via sysctl)
"""

import hashlib
import platform
import subprocess
import sys

# ISA-related macro keywords emitted by ``cc -march=native -dM -E``.
# These correspond to the instruction set extensions that affect codegen.
_X86_ISA_KEYWORDS = (
    "ADX",
    "AES",
    "AVX",
    "BMI",
    "CLFLUSH",
    "F16C",
    "FMA",
    "LZCNT",
    "MMX",
    "MOVBE",
    "PCLMUL",
    "POPCNT",
    "RDRAND",
    "RDSEED",
    "SHA",
    "SSE",
    "VAES",
    "VPCLMUL",
)


def _x86_features() -> str:
    """Query ISA features by asking the system C compiler what -march=native enables."""
    for cc in ("cc", "gcc", "clang"):
        try:
            out = subprocess.check_output(
                [cc, "-march=native", "-dM", "-E", "-x", "c", "-"],
                input="",
                text=True,
                stderr=subprocess.DEVNULL,
            )
        except (FileNotFoundError, subprocess.CalledProcessError):
            continue

        macros = sorted(
            line.split()[1]
            for line in out.splitlines()
            if line.startswith("#define __") and any(k in line for k in _X86_ISA_KEYWORDS)
        )
        if macros:
            return " ".join(macros)

    return ""


def _aarch64_features() -> str:
    """Query ISA features from /proc/cpuinfo (Linux) or sysctl (macOS)."""
    # Linux: kernel exposes HWCAP flags in /proc/cpuinfo.
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("Features"):
                    return " ".join(sorted(line.split(":")[1].split()))
    except FileNotFoundError:
        pass

    # macOS: sysctl exposes CPU features.
    try:
        out = subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.features"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        if out.strip():
            return " ".join(sorted(out.strip().split()))
    except (FileNotFoundError, subprocess.CalledProcessError):
        pass

    return ""


def get_features() -> str:
    machine = platform.machine().lower()
    if machine in ("x86_64", "amd64", "x86", "i686"):
        return _x86_features()
    if machine in ("aarch64", "arm64"):
        return _aarch64_features()
    return ""


def main() -> None:
    features = get_features() or platform.processor()
    h = hashlib.sha256(features.encode()).hexdigest()[:16]
    print(f"features: {features}", file=sys.stderr)
    print(h)


if __name__ == "__main__":
    main()
