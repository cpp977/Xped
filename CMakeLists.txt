cmake_minimum_required(VERSION 3.13)

# Set the project name to
project(Xped CXX)

include(cmake/StandardProjectSettings.cmake)

include(cmake/PreventInSourceBuilds.cmake)

############################################################################################
#############                          define options                   ####################
############################################################################################
option(XPED_ENABLE_LRU_CACHE "Use lru cache library from github." OFF)

option(XPED_USE_LIBCXX "Use libc++ from llvm." OFF)
if(NOT CMAKE_CXX_COMPILER_ID MATCHES ".*Clang")
 if(XPED_USE_LIBCXX)
  message( FATAL_ERROR "It is only possible to use libc++ in combination with clang")
 endif()
endif()

set(SUPPORTED_TENSOR_LIBS EIGEN_TENSOR ARRAY_TENSOR)
set(XPED_TENSOR_LIB "EIGEN_TENSOR" CACHE STRING "Used tensor library for plain tensor operations.")
if(NOT (${XPED_TENSOR_LIB} IN_LIST SUPPORTED_TENSOR_LIBS))
  message( FATAL_ERROR "You specified a tensor library which is not supported.")
endif()
message( STATUS "Build uses the ${XPED_TENSOR_LIB} library.")

option(XPED_BUILD_TESTS "Build the tests." ON)

option(XPED_ENABLE_CACHE "Enable cache if available" OFF)

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
############################################################################################

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

#git is required to clone libraries used from github/gitlab
find_package(Git REQUIRED)
include(cmake/addGitLibs.cmake)

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

add_subdirectory(src)
if (${XPED_BUILD_TESTS})
  add_subdirectory(tests)
endif()
