# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
#
# Locate warp-lang from Newton's uv environment: its headers (WARP_INCLUDE_DIR),
# its runtime library (WARP_LIBRARY, warp.so/.dylib/.dll), and its LLVM-JIT
# backend (WARP_CLANG_LIBRARY, warp-clang.so/.dylib/.dll), which resolves a
# CPU-captured graph's compiled kernels at replay time.
#
# Neither library is exposed as a linkable IMPORTED target: the pip wheel does
# not ship an import library (.lib) for either on Windows, so this build
# dlopen()s/LoadLibrary()s them at runtime instead of linking at build time --
# on every platform, not just Windows, matching how Warp's own C++ consumers
# do it (see github.com/erwincoumans/warp_cpp). WARP_LIBRARY / WARP_CLANG_LIBRARY
# are only used to bake absolute paths into the build via
# target_compile_definitions (see CMakeLists.txt) and to fail fast here with a
# clear message if either is missing, rather than a cryptic dlopen error later.
#
# Run `uv sync` before configuring. Override discovery with
# -DWARP_ROOT=/path/to/site-packages/warp or -DWARP_PYTHON=/path/to/python

include(FindPackageHandleStandardArgs)

if(WARP_ROOT)
    set(_warp_root "${WARP_ROOT}")
else()
    set(_warp_python "${WARP_PYTHON}")
    if(NOT _warp_python AND EXISTS "${NEWTON_ROOT}/.venv/bin/python")
        set(_warp_python "${NEWTON_ROOT}/.venv/bin/python")
    elseif(NOT _warp_python AND EXISTS "${NEWTON_ROOT}/.venv/Scripts/python.exe")
        # uv's venv layout on Windows is Scripts/, not bin/.
        set(_warp_python "${NEWTON_ROOT}/.venv/Scripts/python.exe")
    endif()
    if(_warp_python)
        set(Python3_EXECUTABLE "${_warp_python}")
    else()
        find_package(Python3 COMPONENTS Interpreter REQUIRED)
        set(_warp_python "${Python3_EXECUTABLE}")
    endif()

    execute_process(
        COMMAND "${_warp_python}" -c
            "import pathlib; import warp; print(pathlib.Path(warp.__file__).resolve().parent)"
        OUTPUT_VARIABLE _warp_root
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_VARIABLE _warp_error
        RESULT_VARIABLE _warp_result
    )

    if(_warp_result)
        message(FATAL_ERROR
            "Failed to import warp from ${_warp_python}: ${_warp_error}\n"
            "Install Warp with: uv sync")
    endif()
endif()

set(WARP_INCLUDE_DIR "${_warp_root}/native")

if(WIN32)
    set(_warp_library "${_warp_root}/bin/warp.dll")
    set(_warp_clang_library "${_warp_root}/bin/warp-clang.dll")
elseif(APPLE)
    set(_warp_library "${_warp_root}/bin/libwarp.dylib")
    set(_warp_clang_library "${_warp_root}/bin/libwarp-clang.dylib")
else()
    set(_warp_library "${_warp_root}/bin/warp.so")
    set(_warp_clang_library "${_warp_root}/bin/warp-clang.so")
endif()

if(NOT EXISTS "${WARP_INCLUDE_DIR}/apic.h")
    message(FATAL_ERROR
        "Warp headers not found at ${WARP_INCLUDE_DIR}\n"
        "Expected apic.h from a uv-installed warp-lang package.")
endif()

if(NOT EXISTS "${_warp_library}")
    message(FATAL_ERROR
        "Warp runtime library not found at ${_warp_library}\n"
        "Reinstall with: uv sync")
endif()

if(NOT EXISTS "${_warp_clang_library}")
    message(FATAL_ERROR
        "Warp's LLVM-JIT backend not found at ${_warp_clang_library}\n"
        "Reinstall with: uv sync")
endif()

set(WARP_ROOT "${_warp_root}" CACHE PATH "Root of the uv-installed warp package")
set(WARP_LIBRARY "${_warp_library}" CACHE FILEPATH "Path to the Warp runtime shared library")
set(WARP_CLANG_LIBRARY "${_warp_clang_library}" CACHE FILEPATH "Path to Warp's LLVM-JIT backend library")

find_package_handle_standard_args(Warp
    REQUIRED_VARS WARP_ROOT WARP_INCLUDE_DIR WARP_LIBRARY WARP_CLANG_LIBRARY
)
