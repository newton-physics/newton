// SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>

namespace newton::controllers::detail {

// Load a shared library (.so / .dylib / .dll) by path. Returns nullptr on
// failure; dynamic_library_last_error() then describes why. The library is
// never unloaded -- it lives for the rest of the process, same as Warp's own
// Python bindings and C++ consumers (e.g. github.com/erwincoumans/warp_cpp)
// assume.
void* dynamic_library_open(const char* path);

// Resolve a symbol's address out of an already-loaded library. Returns
// nullptr if the symbol isn't exported; dynamic_library_last_error() then
// describes why.
void* dynamic_library_resolve(void* handle, const char* name);

// Describes the most recent dynamic_library_open() / dynamic_library_resolve()
// failure on this thread. Only meaningful immediately after one of those calls
// returns nullptr.
std::string dynamic_library_last_error();

}  // namespace newton::controllers::detail
