// SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
// SPDX-License-Identifier: Apache-2.0

#include "dynamic_library.hpp"

#if defined(_WIN32)
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace newton::controllers::detail {

void* dynamic_library_open(const char* path) {
#if defined(_WIN32)
    return static_cast<void*>(LoadLibraryA(path));
#else
    return dlopen(path, RTLD_NOW);
#endif
}

void* dynamic_library_resolve(void* handle, const char* name) {
#if defined(_WIN32)
    return reinterpret_cast<void*>(GetProcAddress(static_cast<HMODULE>(handle), name));
#else
    return dlsym(handle, name);
#endif
}

std::string dynamic_library_last_error() {
#if defined(_WIN32)
    const DWORD code = GetLastError();
    char* message = nullptr;
    FormatMessageA(
        FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS, nullptr, code,
        0, reinterpret_cast<char*>(&message), 0, nullptr
    );
    std::string result = message ? message : ("error code " + std::to_string(code));
    if (message) LocalFree(message);
    return result;
#else
    const char* detail = dlerror();
    return detail ? detail : "unknown error";
#endif
}

}  // namespace newton::controllers::detail
