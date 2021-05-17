cmake_minimum_required(VERSION 3.13)

# Set the project name to
project(Xped CXX)

set(XPED_DIR ${CMAKE_CURRENT_SOURCE_DIR})

include(cmake/StandardProjectSettings.cmake)

include(cmake/PreventInSourceBuilds.cmake)

###################################################################################################################################
###################################                          define options                   #####################################
###################################################################################################################################
option(XPED_ENABLE_LRU_CACHE "Use lru cache library from github." OFF)

set(XPED_SUPPORTED_EFFICIENCY_MODELS XPED_MEMORY_EFFICIENT XPED_TIME_EFFICIENT)
set(XPED_EFFICIENCY_MODEL "XPED_TIME_EFFICIENT" CACHE STRING "Xped tries to be time efficient.")
if(NOT (${XPED_EFFICIENCY_MODEL} IN_LIST XPED_SUPPORTED_EFFICIENCY_MODELS))
  message( FATAL_ERROR "You specified an unsupported efficiency model.")
endif()
message( STATUS "Build uses the ${XPED_EFFICIENCY_MODEL} model.")

option(XPED_USE_LIBCXX "Use libc++ from llvm." OFF)
if(NOT CMAKE_CXX_COMPILER_ID MATCHES ".*Clang")
 if(XPED_USE_LIBCXX)
  message( FATAL_ERROR "It is only possible to use libc++ in combination with clang")
 endif()
endif()

set(XPED_SUPPORTED_TENSOR_LIBS EIGEN_TENSOR ARRAY_TENSOR CYCLOPS_TENSOR)
set(XPED_TENSOR_LIB "EIGEN_TENSOR" CACHE STRING "Used tensor library for plain tensor operations.")
if(NOT (${XPED_TENSOR_LIB} IN_LIST XPED_SUPPORTED_TENSOR_LIBS))
  message( FATAL_ERROR "You specified a tensor library which is not supported.")
endif()
message( STATUS "Build uses the ${XPED_TENSOR_LIB} library.")

option(XPED_BUILD_TESTS "Build the tests." ON)

option(XPED_BUILD_CYCLOPS "Build the cyclops library from source." ON)

option(XPED_ENABLE_CACHE "Enable a compiler cache if available" OFF)

option(XPED_USE_OPENMP "Enable openmp parallelization" ON)
option(XPED_USE_OPENMPI "Enable openmpi parallelization" OFF)
if(${XPED_TENSOR_LIB} STREQUAL "CYCLOPS_TENSOR")
  set(XPED_USE_OPENMPI ON)
endif()

option(XPED_USE_BLAS "Enable blas linking." ON)
option(XPED_USE_LAPACK "Enable lapack linking." ON)

option(XPED_ENABLE_BUILD_WITH_TIME_TRACE "Enable -ftime-trace to generate time tracing .json files on clang" OFF)

option(XPED_ENABLE_COVERAGE "Enable coverage reporting for gcc/clang" FALSE)

option(XPED_ENABLE_SANITIZER_ADDRESS "Enable address sanitizer" FALSE)
option(XPED_ENABLE_SANITIZER_LEAK "Enable leak sanitizer" FALSE)
option(XPED_ENABLE_SANITIZER_UNDEFINED_BEHAVIOR "Enable undefined behavior sanitizer" FALSE)
option(XPED_ENABLE_SANITIZER_THREAD "Enable thread sanitizer" FALSE)
option(XPED_ENABLE_SANITIZER_MEMORY "Enable memory sanitizer" FALSE)

option(XPED_ENABLE_DOXYGEN "Enable doxygen doc builds of source" OFF)

option(XPED_ENABLE_CPPCHECK "Enable static analysis with cppcheck" OFF)
option(XPED_ENABLE_CLANG_TIDY "Enable static analysis with clang-tidy" OFF)
option(XPED_ENABLE_INCLUDE_WHAT_YOU_USE "Enable static analysis with include-what-you-use" OFF)
option(XPED_ENABLE_CLANG_FORMAT "Enable clang-format target." OFF)
###################################################################################################################################
##########################################                          end                   #########################################
###################################################################################################################################

# Link this 'library' to set the c++ standard / compile-time options requested
add_library(project_options INTERFACE)

# include compiler options from cmake/CompilerOptions.cmake
include(cmake/CompilerOptions.cmake)
set_project_options(project_options)

# Link this 'library' to use the warnings specified in CompilerWarnings.cmake
add_library(project_warnings INTERFACE)

# standard compiler warnings
include(cmake/CompilerWarnings.cmake)
set_project_warnings(project_warnings)

if(CMAKE_CXX_COMPILER_ID MATCHES ".*Clang")
  if(XPED_ENABLE_BUILD_WITH_TIME_TRACE)
    target_compile_options(project_options INTERFACE -ftime-trace)
  endif()
endif()

# enable cache system
include(cmake/Cache.cmake)

# sanitizer options if supported by compiler
include(cmake/Sanitizers.cmake)
enable_sanitizers(project_options)

# enable doxygen
include(cmake/Doxygen.cmake)
enable_doxygen()

# allow for static analysis options
include(cmake/StaticAnalyzers.cmake)

# add clang-format target
include(cmake/clang-format.cmake)

find_package(GSL REQUIRED)
find_package(Boost REQUIRED)

if(XPED_USE_OPENMP)
  find_package(OpenMP)
endif()

if(XPED_USE_OPENMPI)
  find_package(MPI)
endif()

if(XPED_USE_BLAS)
  find_package(BLAS)
  list(FIND BLAS_LIBRARIES /opt/intel/mkl/lib/intel64_lin/libmkl_core.so XPED_INTERN_USE_MKL)
  if(NOT (${XPED_INTERN_USE_MKL} EQUAL -1))
    set(XPED_USE_MKL ON)
    find_path(XPED_MKL_INCLUDE_DIR mkl.h HINTS /opt/intel/mkl/include /usr/lib/intel/mkl/include /usr/lib64/intel/mkl/include)
    if(NOT XPED_MKL_INCLUDE_DIR)
      message(FATAL_ERROR "Could not find intel mkl headers.")
    else()
      message(STATUS "Found mkl headers: ${XPED_MKL_INCLUDE_DIR}")
    endif()
  else()
    set(XPED_USE_MKL OFF)
  endif()
else()
  set(XPED_USE_MKL OFF)
endif()

if(XPED_USE_LAPACK)
  find_package(LAPACK)
endif()

find_package(GSL REQUIRED)
find_package(Boost REQUIRED)

#git is required to clone libraries used from github/gitlab
find_package(Git REQUIRED)
include(cmake/addGitLibs.cmake)

if(${XPED_TENSOR_LIB} STREQUAL "CYCLOPS_TENSOR" AND NOT XPED_BUILD_CYCLOPS)
  find_library(CYCLOPS_LIB NAMES ctf
    #HINTS /home/user/__LIBS__/ctf/lib
    #NO_DEFAULT_PATH
    REQUIRED
  )
  add_library(cyclops_lib::all UNKNOWN IMPORTED)
  set_target_properties(cyclops_lib::all PROPERTIES
    IMPORTED_LOCATION "${CYCLOPS_LIB}"
    )
  message(STATUS ${CYCLOPS_LIB})

  find_path(CYCLOPS_INCLUDE_DIR
    NAMES ctf.hpp
    #NO_DEFAULT_PATH
    #HINTS /home/user/__LIBS__/ctf/include
    REQUIRED
  )
  message(STATUS ${CYCLOPS_INCLUDE_DIR})
endif()

add_subdirectory(src)

if (${XPED_BUILD_TESTS})
  add_subdirectory(tests)
endif()
