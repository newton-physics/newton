# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
#
# Standalone build for the C++ controller runtime. Newton itself is pure Python;
# nothing here is built by `pip install newton` or exercised by `newton.tests`.
#
#   cmake -S newton/_src/controllers/cpp -B build/controllers
#   cmake --build build/controllers
#   ctest --test-dir build/controllers --output-on-failure
#
# See README.md in this directory.

cmake_minimum_required(VERSION 3.24)
project(newton_controllers_cpp CXX)

# Repo root, four levels up: newton/_src/controllers/cpp -> newton
get_filename_component(NEWTON_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../../.." ABSOLUTE)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

find_package(Warp REQUIRED)

add_library(newton_controllers STATIC src/controller.cpp)

target_compile_features(newton_controllers PUBLIC cxx_std_20)
target_include_directories(newton_controllers PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_include_directories(newton_controllers PRIVATE "${WARP_INCLUDE_DIR}")

# warp.so and warp-clang.so are dlopen()'d/LoadLibrary()'d at runtime, not
# linked -- see FindWarp.cmake for why. CMAKE_DL_LIBS is dl on Linux (needed
# for dlopen/dlsym) and empty elsewhere (LoadLibrary/GetProcAddress live in
# kernel32.dll, always linked on Windows; macOS's libSystem always has dlopen).
target_link_libraries(newton_controllers PUBLIC ${CMAKE_DL_LIBS})
target_compile_definitions(newton_controllers PRIVATE
    "NEWTON_WARP_LIBRARY=\"${WARP_LIBRARY}\""
    "NEWTON_WARP_CLANG_LIBRARY=\"${WARP_CLANG_LIBRARY}\""
)

option(NEWTON_CONTROLLERS_BUILD_TESTS "Build the C++ controller runtime tests" ON)
if(NEWTON_CONTROLLERS_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
