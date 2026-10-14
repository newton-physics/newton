# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Output Warp's CPU compilation target for keying the CPU kernel cache.

With the default ``-march=native``, Warp names CPU objects after a hash of the
LLVM version and the host's ISA features, so cached objects only hit on runners
with the same features. Hosted runners vary, so the cache key must include it.
"""

import os
import sys


def cpu_target() -> tuple[str, str]:
    """Return Warp's CPU target hash and the LLVM host CPU name."""
    import warp as wp  # noqa: PLC0415
    from warp._src import context  # noqa: PLC0415

    wp.config.quiet = True
    wp.init()
    flags = context._resolve_cpu_compiler_flags(None, wp.config.cpu_compiler_flags)
    return context._get_cpu_target_hash(flags) or "none", context._get_host_cpu_name()


def main() -> int:
    try:
        target, cpu = cpu_target()
    except Exception as e:
        # Relies on private Warp API; a shared key keeps caching working, just with fewer hits.
        print(f"::warning::Could not determine the Warp CPU target: {e!r}")
        target, cpu = "unknown", "unknown"

    print(f"Warp CPU target: {target} (LLVM host CPU: {cpu})")
    if output := os.environ.get("GITHUB_OUTPUT"):
        with open(output, "a", encoding="utf-8") as f:
            f.write(f"target={target}\n")
    if summary := os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(summary, "a", encoding="utf-8") as f:
            f.write(f"Warp CPU target: `{target}` (LLVM host CPU: `{cpu}`)\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
