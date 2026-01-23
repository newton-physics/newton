cmake_minimum_required(VERSION 3.14)
project(DefKitAdv)

set(CMAKE_CXX_STANDARD 14)

find_package(Eigen3 QUIET)
if(NOT Eigen3_FOUND)
    message(STATUS "Eigen3 not found, fetching...")
    include(FetchContent)
    FetchContent_Declare(
      Eigen3
      GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
      GIT_TAG 3.4.0
      SHALLOW TRUE
    )
    FetchContent_MakeAvailable(Eigen3)
    include_directories(${eigen3_SOURCE_DIR})
else()
    include_directories(${EIGEN3_INCLUDE_DIR})
endif()

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/LinearMath)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/PositionBasedDynamics)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/PositionBasedDynamics/Common)

# Source files
file(GLOB SOURCES 
    "DefKit.cpp"
    "ElasticRod.cpp"
)

file(GLOB PBD_SOURCES "PositionBasedDynamics/*.cpp")
file(GLOB MATH_SOURCES "LinearMath/*.cpp")

# Create shared library
add_library(DefKitAdv SHARED ${SOURCES} ${PBD_SOURCES} ${MATH_SOURCES})

# Definitions
if(MSVC)
    add_definitions(-D_CRT_SECURE_NO_WARNINGS)
endif()
