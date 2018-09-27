project(regression)
cmake_minimum_required(VERSION 3.9)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/")

find_package(Python3Interp REQUIRED)
find_package(Eigen3 REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -Wall --pedantic -fPIC")

set(PYBIND11_PYTHON_VERSION ${PYTHON_VERSION_STRING})
add_subdirectory(pybind11)

add_library(regression
  LinearRegression.cpp
  )

target_link_libraries(regression Eigen3::Eigen )

pybind11_add_module(PyRegression
                    PyRegression.cpp
                    )
target_include_directories(PyRegression PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}> $<INSTALL_INTERFACE:include>)

target_link_libraries(PyRegression PUBLIC regression)

