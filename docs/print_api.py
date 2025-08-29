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

import importlib
import inspect


def public_symbols(mod) -> list[str]:
    """Return the list of public names for *mod* (honours ``__all__``)."""

    if hasattr(mod, "__all__") and isinstance(mod.__all__, list | tuple):
        return list(mod.__all__)

    def is_public(name: str) -> bool:
        if name.startswith("_"):
            return False
        return not inspect.ismodule(getattr(mod, name))

    return sorted(filter(is_public, dir(mod)))


def get_symbols(mod_name: str):
    module = importlib.import_module(mod_name)
    all_symbols = public_symbols(module)

    children = []
    for name in all_symbols:
        attr = getattr(module, name)
        if inspect.ismodule(attr):
            children.append(get_symbols(f"{mod_name}.{name}"))
        else:
            children.append(name)

    return (mod_name.split(".")[-1], children)


def print_symbols(sym_dict, indent=0):
    name, children = sym_dict[0], sym_dict[1]
    print(f"{'   ' * indent}{name}:")

    symbols = []
    submodules = []
    for child in children:
        if isinstance(child, str):
            symbols.append(child)
        else:
            submodules.append(child)

    # sort
    symbols.sort()
    submodules.sort(key=lambda x: x[0])

    for sym in symbols:
        print(f"{'   ' * (indent + 1)}{sym}")
    print()
    for sub in submodules:
        print_symbols(sub, indent=indent + 1)


print_symbols(get_symbols("newton"))
