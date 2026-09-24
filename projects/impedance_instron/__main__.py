# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Run the controller pipeline or ingest processed Visual3D measurements."""

import sys


def main(argv: list[str] | None = None) -> None:
    """Dispatch data commands while preserving the controller command line."""
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments and arguments[0] == "visual3d":
        from .cartesian.visual3d import _main as run  # noqa: PLC0415

        run(arguments[1:])
    elif arguments and arguments[0] == "prepare-visual3d":
        from .cartesian.prepare_visual3d import main as run  # noqa: PLC0415

        run(arguments[1:])
    else:
        from .pipeline import main as run  # noqa: PLC0415

        run(arguments)


if __name__ == "__main__":
    main()
