cmake_minimum_required(VERSION 3.0)
cmake_policy(SET CMP0048 NEW)

project(Discrete-Hankel-Transform VERSION 0.1)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "-Wall -g -fPIC -O2 -march=native")

# C++17 support bessel functions; use boost if it doesn't exist
find_package(Boost REQUIRED)
include_directories(${Boost_INCLUDE_DIR})

find_package(Eigen3 3.3 REQUIRED NO_MODULE)
include_directories(${EIGEN3_INCLUDE_DIR})

# fmt
add_subdirectory(${CMAKE_SOURCE_DIR}/external/fmt EXCLUDE_FROM_ALL)

# pybind11
# python 3.8 may be used for new system, but python in anaconda is 3.7
find_package(PythonInterp 3.7 REQUIRED)
find_package(PythonLibs 3.7 REQUIRED)

add_subdirectory(${CMAKE_SOURCE_DIR}/external/pybind11 EXCLUDE_FROM_ALL)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)

add_subdirectory(src)

