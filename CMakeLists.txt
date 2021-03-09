cmake_minimum_required(VERSION 3.13)

# Set the project name to
project(Multiped CXX)

include(cmake/StandardProjectSettings.cmake)

include(cmake/PreventInSourceBuilds.cmake)

#define options
option(ENABLE_LRU_CACHE "Use lru cache library from github." OFF)
option(USE_LIBCXX "Use libc++ from llvm." OFF)
if(NOT CMAKE_CXX_COMPILER_ID MATCHES ".*Clang")
 if(USE_LIBCXX)
  message( FATAL_ERROR "It is only possible to use libc++ in combination with clang")
 endif()
endif()
   
# Link this 'library' to set the c++ standard / compile-time options requested
add_library(project_options INTERFACE)

# include compiler options from cmake/CompilerOptions.cmake
include(cmake/CompilerOptions.cmake)
set_project_options(project_options)

set(CMAKE_CXX_FLAGS_RELEASE "")
message("CMAKE_CXX_FLAGS_RELEASE is ${CMAKE_CXX_FLAGS_RELEASE}")

# target_compile_features(project_options INTERFACE cxx_std_17)

# Link this 'library' to use the warnings specified in CompilerWarnings.cmake
add_library(project_warnings INTERFACE)

# standard compiler warnings
include(cmake/CompilerWarnings.cmake)
set_project_warnings(project_warnings)

if(CMAKE_CXX_COMPILER_ID MATCHES ".*Clang")
  option(ENABLE_BUILD_WITH_TIME_TRACE "Enable -ftime-trace to generate time tracing .json files on clang" OFF)
  if(ENABLE_BUILD_WITH_TIME_TRACE)
    add_compile_definitions(project_options INTERFACE -ftime-trace)
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

option(BUILD_SHARED_LIBS "Enable compilation of shared libraries" OFF)

# Very basic PCH example
option(ENABLE_PCH "Enable Precompiled Headers" OFF)
if(ENABLE_PCH)
  # This sets a global PCH parameter, each project will build its own PCH, which is a good idea if any #define's change
  #
  # consider breaking this out per project as necessary
  target_precompile_headers(
    project_options
    INTERFACE
    <vector>
    <string>
    <map>
    <utility>)
endif()

add_subdirectory(src)
add_subdirectory(tests)
