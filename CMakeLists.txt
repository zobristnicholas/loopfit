### setup project ###
cmake_minimum_required(VERSION 3.17.3)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
project(loopfit)

### find python extensions (from scikit-build) ###
find_package(PythonExtensions REQUIRED)

### find cython (from scikit-build) ###
find_package(Cython REQUIRED)

### find numpy (not using scikit-build method) ###
# find_package(NumPy REQUIRED) will grab a system install of numpy over the
# local environment, so we don't use skbuild's method and use
# numpy.get_include() instead using the PYTHON_EXECUTABLE defined by
# PythonExtensions package
execute_process(
  COMMAND "${PYTHON_EXECUTABLE}"
  -c "import numpy; print(numpy.get_include())"
  OUTPUT_VARIABLE NumPy_INCLUDE_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE
)
include_directories(${NumPy_INCLUDE_DIR})

### configure ceres-solver ###
find_package(Ceres REQUIRED)
include_directories(${CERES_INCLUDE_DIRS})

### configure loopfit ###
add_subdirectory(src/loopfit)
