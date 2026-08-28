# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
#
# Locate warp-lang from Newton's uv environment and expose warp::runtime and
# warp::clang. warp::clang is the LLVM-JIT backend (warp-clang.so/.dylib/.dll)
# that resolves a CPU-captured graph's compiled kernels at replay time; it
# ships alongside warp.so in every uv-installed warp-lang package, so it is
# required just like the runtime library itself.
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

if(NOT TARGET warp::runtime)
    add_library(warp::runtime UNKNOWN IMPORTED GLOBAL)
    set_target_properties(warp::runtime PROPERTIES
        IMPORTED_LOCATION "${_warp_library}"
        IMPORTED_NO_SONAME TRUE
        INTERFACE_INCLUDE_DIRECTORIES "${WARP_INCLUDE_DIR}"
    )
endif()

if(NOT TARGET warp::clang)
    add_library(warp::clang UNKNOWN IMPORTED GLOBAL)
    set_target_properties(warp::clang PROPERTIES
        IMPORTED_LOCATION "${_warp_clang_library}"
        IMPORTED_NO_SONAME TRUE
    )
endif()

set(WARP_ROOT "${_warp_root}" CACHE PATH "Root of the uv-installed warp package")
set(WARP_LIBRARY "${_warp_library}" CACHE FILEPATH "Path to the Warp runtime shared library")
set(WARP_CLANG_LIBRARY "${_warp_clang_library}" CACHE FILEPATH "Path to Warp's LLVM-JIT backend library")

find_package_handle_standard_args(Warp
    REQUIRED_VARS WARP_ROOT WARP_INCLUDE_DIR WARP_LIBRARY WARP_CLANG_LIBRARY
)
