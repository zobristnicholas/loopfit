cmake_minimum_required(VERSION 3.5.0)

project(loopfit)

find_package(Cython REQUIRED)
find_package(NumPy REQUIRED)
find_package(PythonExtensions REQUIRED)
find_package(Ceres REQUIRED)

include_directories(${NumPy_INCLUDE_DIR})
include_directories(${CERES_INCLUDE_DIRS})
include_directories(src/loopfit)

add_subdirectory(src/loopfit)