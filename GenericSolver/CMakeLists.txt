cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
project(python-solver LANGUAGES CXX)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_POSITION_INDEPENDENT_CODE True)
set(CMAKE_INCLUDE_CURRENT_DIR TRUE)


set(PYTHON_SOLVER_MAJOR_VERSION 0)
set(PYTHON_SOLVER_MINOR_VERSION 3)
set(PYTHON_SOLVER_VERSION_STRING
    ${PYTHON_SOLVER_MAJOR_VERSION}.${genericIO_MINOR_VERSION})

#Compiler flags
SET(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -O3")

#Required packages
find_package(Boost REQUIRED)
find_package(OpenMP)

#Adding genericIO submodule
# execute_process(COMMAND git submodule add http://cees-gitlab.Stanford.EDU/bob/genericIO.git external/genericIO WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
#Updating genericIO library
if(NOT DEFINED AddGenericIO)
    set(AddGenericIO TRUE)
endif()
if(${AddGenericIO})
  execute_process(COMMAND git submodule update --init --recursive -- external/genericIO WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
  add_subdirectory(external/genericIO)
  include_directories(${CMAKE_CURRENT_BINARY_DIR}/external/genericIO)
  include_directories(${CMAKE_CURRENT_BINARY_DIR}/external/genericIO/external/sepVector)
  include_directories(${CMAKE_CURRENT_SOURCE_DIR}/external/genericIO/cpp)
  include_directories(${CMAKE_CURRENT_SOURCE_DIR}/external/genericIO/external/sepVector/lib)
  include_directories(${CMAKE_CURRENT_SOURCE_DIR}/external/genericIO/external/sepVector/external/hypercube/lib)
  include_directories(${CMAKE_CURRENT_SOURCE_DIR}/external/genericIO/external/buffers/external/json)
  include_directories(${CMAKE_CURRENT_SOURCE_DIR}/external/genericIO/external/buffers/lib)
  set(AddGenericIO FALSE)
endif()

#Finding multithread libraries
find_package(Threads)
# Adding TBB library
find_package(TBB REQUIRED)

add_subdirectory(python_modules)

include(InstallRequiredSystemLibraries)
set(CPACK_RESOURCE_FILE_LICENSE "${PROJECT_SOURCE_DIR}/LICENSE")
set(CPACK_PACKAGE_VERSION_MAJOR "${PYTHON_SOLVER_MAJOR_VERSION}")
set(CPACK_PACKAGE_VERSION_MINOR "${PYTHON_SOLVER_MINOR_VERSION}")
set(CPACK_PACKAGE_VERSION_PATCH "0")
set(CPACK_PACKAGE_NAME "GenericSolver")
set(CPACK_PACKAGE_CONTACT "Ettore Biondi & Robert Clapp")
include(CPack)
