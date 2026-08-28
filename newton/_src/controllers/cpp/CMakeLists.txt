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
target_link_libraries(newton_controllers PUBLIC warp::runtime warp::clang)

# warp.so defers its Python C API references to runtime (-Wl,-z,lazy) so that
# non-Python hosts can load it. Tell the static linker to do the same instead of
# failing on symbols it cannot resolve.
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    target_link_options(newton_controllers INTERFACE "LINKER:--unresolved-symbols=ignore-in-shared-libs")
endif()

get_filename_component(WARP_BIN_DIR "${WARP_LIBRARY}" DIRECTORY)
set_target_properties(newton_controllers PROPERTIES BUILD_RPATH "${WARP_BIN_DIR}")

option(NEWTON_CONTROLLERS_BUILD_TESTS "Build the C++ controller runtime tests" ON)
if(NEWTON_CONTROLLERS_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
