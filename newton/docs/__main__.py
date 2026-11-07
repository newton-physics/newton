# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Print the installed Newton documentation location."""

import argparse

from newton import info
from newton._src.documentation import _location

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--path", action="store_true", help="Print only the documentation directory")
arguments = parser.parse_args()
try:
    if arguments.path:
        root, _mode = _location()
        if root is None:
            parser.exit(1, "No local Newton documentation found. Run without --path for online documentation.\n")
        print(root)
    else:
        info()
except (RuntimeError, OSError, ValueError) as error:
    parser.exit(1, f"{error}\n")
