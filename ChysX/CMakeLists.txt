cmake_minimum_required(VERSION 3.18)
project(chysx_native LANGUAGES CXX CUDA)

# Emit compile_commands.json for clangd / IDEs.  Real builds go through
# scikit-build-core, but if you build with a plain `cmake -B build` this
# file ends up in build/ and can be symlinked / copied to the project
# root so clangd picks it up.
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# C++17 for both host and device code
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Need Python.h + the pybind11 CMake config (provided by the build env)
find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)

# Locate pybind11 from whichever Python was used to invoke the build.
# scikit-build-core puts pybind11 into PYTHONPATH for us.
execute_process(
    COMMAND "${Python_EXECUTABLE}" -c "import pybind11, sys; sys.stdout.write(pybind11.get_cmake_dir())"
    OUTPUT_VARIABLE pybind11_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
find_package(pybind11 CONFIG REQUIRED)

# CUDA runtime library (for cudaGetLastError, cudaStream_t, ...).
find_package(CUDAToolkit REQUIRED)

pybind11_add_module(_chysx_native MODULE
    src/memory/cuda_array.cu
    src/cloth/cloth_simulator.cu
    src/bindings.cpp
)

target_include_directories(_chysx_native PRIVATE src)
target_link_libraries(_chysx_native PRIVATE CUDA::cudart)

set_target_properties(_chysx_native PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# scikit-build-core copies installed targets into the wheel.
install(TARGETS _chysx_native LIBRARY DESTINATION chysx)
