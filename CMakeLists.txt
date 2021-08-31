cmake_minimum_required(VERSION 3.18)

# Set the project name to
project(Xped CXX C Fortran)
set(XPED_DIR ${CMAKE_CURRENT_SOURCE_DIR})

include(cmake/StandardProjectSettings.cmake)

include(cmake/PreventInSourceBuilds.cmake)

include(FeatureSummary)

###################################################################################################################################
###################################                          define options                   #####################################
###################################################################################################################################
option(XPED_ENABLE_LRU_CACHE "Use lru cache library from github." OFF)

set(XPED_EFFICIENCY_MODEL "XPED_TIME_EFFICIENT" CACHE STRING "Xped tries to be time efficient.")
set(XPED_SUPPORTED_EFFICIENCY_MODELS "XPED_TIME_EFFICIENT;XPED_MEMORY_EFFICIENT")
set_property(CACHE XPED_EFFICIENCY_MODEL PROPERTY STRINGS ${XPED_SUPPORTED_EFFICIENCY_MODELS})
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

set(XPED_TENSOR_LIB "EIGEN_TENSOR" CACHE STRING "Used tensor library for plain tensor operations.")
set(XPED_SUPPORTED_TENSOR_LIBS "EIGEN_TENSOR;ARRAY_TENSOR;CYCLOPS_TENSOR")
set_property(CACHE XPED_TENSOR_LIB PROPERTY STRINGS ${XPED_SUPPORTED_TENSOR_LIBS})
if(NOT (${XPED_TENSOR_LIB} IN_LIST XPED_SUPPORTED_TENSOR_LIBS))
  message( STATUS "Build wanted to use the ${XPED_TENSOR_LIB} library.")
  message( FATAL_ERROR "You specified a tensor library which is not supported.")  
endif()
message( STATUS "Build uses the ${XPED_TENSOR_LIB} library.")

add_feature_info(TensorLib XPED_TENSOR_LIB "Build will use the ${XPED_TENSOR_LIB} library.")

set(XPED_MATRIX_LIB "EIGEN_MATRIX" CACHE STRING "Used matrix library for plain tensor operations.")
set(XPED_SUPPORTED_MATRIX_LIBS "EIGEN_MATRIX;CYCLOPS_MATRIX")
set_property(CACHE XPED_MATRIX_LIB PROPERTY STRINGS ${XPED_SUPPORTED_MATRIX_LIBS})
if(NOT (${XPED_MATRIX_LIB} IN_LIST XPED_SUPPORTED_MATRIX_LIBS))
  message( STATUS "Build wanted to use the ${XPED_MATRIX_LIB} library.")
  message( FATAL_ERROR "You specified a matrix library which is not supported.")  
endif()
message( STATUS "Build uses the ${XPED_MATRIX_LIB} library.")

set(XPED_VECTOR_LIB "EIGEN_VECTOR" CACHE STRING "Used matrix library for plain tensor operations.")
set(XPED_SUPPORTED_VECTOR_LIBS "EIGEN_VECTOR;CYCLOPS_VECTOR")
set_property(CACHE XPED_VECTOR_LIB PROPERTY STRINGS ${XPED_SUPPORTED_VECTOR_LIBS})
if(NOT (${XPED_VECTOR_LIB} IN_LIST XPED_SUPPORTED_VECTOR_LIBS))
  message( STATUS "Build wanted to use the ${XPED_VECTOR_LIB} library.")
  message( FATAL_ERROR "You specified a vector library which is not supported.")  
endif()
message( STATUS "Build uses the ${XPED_VECTOR_LIB} library.")

option(XPED_BUILD_TESTS "Build the tests." ON)
add_feature_info(Tests XPED_BUILD_TESTS "The tests will be compiled.")

option(XPED_BUILD_BENCHMARKS "Build the benchmarks." ON)

option(XPED_BUILD_CYCLOPS "Build the cyclops library from source." OFF)

option(XPED_ENABLE_CACHE "Enable a compiler cache if available" OFF)

option(XPED_USE_OPENMP "Enable openmp parallelization" ON)
option(XPED_USE_MPI "Enable message parsing interface (mpi) parallelization" OFF)

option(XPED_USE_BLAS "Enable blas linking." ON)
option(XPED_USE_LAPACK "Enable lapack linking." ON)
option(XPED_USE_SCALAPACK "Enable scalapack linking (only useful for MPI programs)." OFF)

option(XPED_USE_MKL "Enable use of intel math kernel library (MKL)." OFF)
if(XPED_USE_MKL)
  set(XPED_USE_BLAS OFF)
  set(XPED_USE_LAPACK OFF)
  set(XPED_USE_SCALAPACK OFF)
  set(XPED_USE_OPENMP OFF)
  set(XPED_USE_MPI OFF)
  
  # set(XPED_MKL_THREADING "OpenMP" CACHE STRING "Which threading is used by MKL.")
  # set(XPED_MKL_SUPPORTED_THREADINGS "OpenMP;TBB;Seq")
  # set_property(CACHE XPED_MKL_THREADING PROPERTY STRINGS ${XPED_MKL_SUPPORTED_THREADINGS})
  # if(NOT (${XPED_MKL_THREADING} IN_LIST XPED_MKL_SUPPORTED_THREADINGS))
  #   message( STATUS "MKL theading was set to ${XPED_MKL_THREADING}.")
  #   message( FATAL_ERROR "You specified a MKL threading which is not supported.")  
  # endif()
  # if(XPED_MKL_THREADING STREQUAL "OpenMP")
  #   set(XPED_MKL_THREADING_STR "omp")
  # elseif(XPED_MKL_THREADING STREQUAL "TBB")
  #   set(XPED_MKL_THREADING_STR "tbb")
  # elseif(XPED_MKL_THREADING STREQUAL "Seq")
  #   set(XPED_MKL_THREADING_STR "seq")
  # endif()
  # if(${CMAKE_SIZEOF_VOID_P} STREQUAL "8")
  #   set(XPED_MKL_BITNESS_STR "32bit")
  # else()
  #   set(XPED_MKL_BITNESS_STR "32bit")
  # endif()
  option(XPED_MKL_USE_MPI "Use mkl with intel MPI and Scalapack enabled." OFF)
endif()

option(XPED_ENABLE_BUILD_WITH_TIME_TRACE "Enable -ftime-trace to generate time tracing .json files on clang" OFF)

option(XPED_ENABLE_COVERAGE "Enable coverage reporting for gcc/clang" OFF)

option(XPED_ENABLE_SANITIZER_ADDRESS "Enable address sanitizer" OFF)
option(XPED_ENABLE_SANITIZER_LEAK "Enable leak sanitizer" OFF)
option(XPED_ENABLE_SANITIZER_UNDEFINED_BEHAVIOR "Enable undefined behavior sanitizer" OFF)
option(XPED_ENABLE_SANITIZER_THREAD "Enable thread sanitizer" OFF)
option(XPED_ENABLE_SANITIZER_MEMORY "Enable memory sanitizer" OFF)

option(XPED_ENABLE_DOXYGEN "Enable doxygen doc builds of source" OFF)

option(XPED_ENABLE_CPPCHECK "Enable static analysis with cppcheck" OFF)
option(XPED_ENABLE_CLANG_TIDY "Enable static analysis with clang-tidy" OFF)
option(XPED_ENABLE_INCLUDE_WHAT_YOU_USE "Enable static analysis with include-what-you-use" OFF)
option(XPED_ENABLE_CLANG_FORMAT "Enable clang-format target." OFF)

set(XPED_LOG_LEVEL "SPDLOG_LEVEL_OFF" CACHE STRING "Compile time log level.")
set(XPED_SUPPORTED_LOG_LEVELS "SPDLOG_LEVEL_TRACE;SPDLOG_LEVEL_DEBUG;SPDLOG_LEVEL_INFO;SPDLOG_LEVEL_WARN;SPDLOG_LEVEL_ERROR;SPDLOG_LEVEL_CRITICAL;SPDLOG_LEVEL_OFF")
set_property(CACHE XPED_LOG_LEVEL PROPERTY STRINGS ${XPED_SUPPORTED_LOG_LEVELS})
if(NOT (${XPED_LOG_LEVEL} IN_LIST XPED_SUPPORTED_LOG_LEVELS))
  message( STATUS "Log level was set to ${XPED_LOG_LEVEL}.")
  message( FATAL_ERROR "You specified a log level which is not supported.")  
endif()
###################################################################################################################################
##########################################                          end                   #########################################
###################################################################################################################################

if(XPED_USE_OPENMP)
  find_package(OpenMP REQUIRED)
endif()

if(XPED_USE_MPI)
  find_package(MPI REQUIRED)
  execute_process(COMMAND ${MPIEXEC_EXECUTABLE} --version OUTPUT_VARIABLE XPED_MPI_VENDOR_CHECK)
  string(FIND ${XPED_MPI_VENDOR_CHECK} "open-mpi" XPED_USE_OPENMPI)
  if(NOT (${XPED_USE_OPENMPI} EQUAL -1))
    set(XPED_USE_OPENMPI ON)
    set(XPED_USE_MPICH OFF)
    set(XPED_USE_INTELMPI OFF)
    message(STATUS "Found openmpi library.")
  else()
    string(FIND ${XPED_MPI_VENDOR_CHECK} "Intel" XPED_USE_INTELMPI)
    if(NOT (${XPED_USE_INTELMPI} EQUAL -1))
      set(XPED_USE_OPENMPI OFF)
      set(XPED_USE_MPICH OFF)
      set(XPED_USE_INTELMPI ON)
      message(FATAL_ERROR "Found intel mkl library but icpc is not the CXX compiler.")
    else()
      set(XPED_USE_OPENMPI OFF)
      set(XPED_USE_INTELMPI OFF)
      set(XPED_USE_MPICH ON)
      message(STATUS "Found mpich library.")
    endif()
  endif()
endif()

if(XPED_USE_BLAS)
  find_package(BLAS REQUIRED)
  set_package_properties(BLAS PROPERTIES
    URL "www.netlib.org"
    DESCRIPTION "(B)asic (L)inear (A)lgebra (S)ubprograms, by vendor ${BLA_VENDOR}."
    TYPE REQUIRED
    )
  
  # list(FIND BLAS_LIBRARIES $ENV{MKLROOT}/lib/intel64_lin/libmkl_core.so XPED_INTERN_USE_MKL)
#   if(NOT (${XPED_INTERN_USE_MKL} EQUAL -1))
#     set(XPED_USE_MKL ON)
#     find_path(XPED_MKL_INCLUDE_DIR mkl.h HINTS $ENV{MKLROOT}/include /usr/lib/intel/mkl/include /usr/lib64/intel/mkl/include)
#     if(NOT XPED_MKL_INCLUDE_DIR)
#       message(FATAL_ERROR "Could not find intel mkl headers.")
#     else()
#       message(STATUS "Found mkl headers: ${XPED_MKL_INCLUDE_DIR}")
#     endif()
#     set_target_properties(BLAS::BLAS PROPERTIES
#       INTERFACE_INCLUDE_DIRECTORIES ${XPED_MKL_INCLUDE_DIR}
#       )
#   else()
#     set(XPED_USE_MKL OFF)
#   endif()
# else()
#   set(XPED_USE_MKL OFF)
endif()

if(XPED_USE_LAPACK)
  find_package(LAPACK REQUIRED)
  # if(XPED_USE_MKL)
  # set(test_prog_for_lapacke "\#include <mkl_lapacke.h> \\nint main() {lapack_int m, n, lda, info\; double *a, *tau\; info = LAPACKE_dgeqrf( LAPACK_COL_MAJOR, m, n, a, lda, tau )\;}")
  # else()
  set(test_prog_for_lapacke "\#include <lapacke.h> \\nint main() {lapack_int m, n, lda, info\; double *a, *tau\; info = LAPACKE_dgeqrf( LAPACK_COL_MAJOR, m, n, a, lda, tau )\;}")
  # endif()
  set(CMAKE_REQUIRED_LIBRARIES LAPACK::LAPACK;BLAS::BLAS)
  include(CheckCXXSourceCompiles)
  check_cxx_source_compiles(${test_prog_for_lapacke} XPED_COMPILES_LAPACKE)
  if (NOT XPED_COMPILES_LAPACKE)    
    find_library(XPED_LAPACKE NAMES lapacke REQUIRED)
    add_library(LAPACKE::LAPACKE UNKNOWN IMPORTED)
    set_target_properties(LAPACKE::LAPACKE PROPERTIES IMPORTED_LOCATION "${XPED_LAPACKE}")
    set(test_prog_for_lapacke2 "\#include <lapacke.h> \\nint main() {lapack_int m, n, lda, info\; double *a, *tau\; info = LAPACKE_dgeqrf( LAPACK_COL_MAJOR, m, n, a, lda, tau )\;}")
    set(CMAKE_REQUIRED_LIBRARIES LAPACKE::LAPACKE)
    check_cxx_source_compiles(${test_prog_for_lapacke2} XPED_COMPILES_LAPACKE2)
    set(XPED_USE_LAPACKE ON)
    if (NOT XPED_COMPILES_LAPACKE2)
      message(FATAL_ERROR "Lapacke support is missing.")
    endif()
  endif()
endif()

if(XPED_USE_MKL)
  list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules")
  find_package(MKL REQUIRED)

  if(NOT TARGET mkl::mkl_intel_32bit_omp_dyn)
    message(FATAL_ERROR "Intel MKL support not sufficient (Core libraries with BLAS and LAPACK support not found).")
  endif()
  if(XPED_MKL_USE_MPI)
    if(NOT TARGET mkl::scalapack_mpich_intel_32bit_omp_dyn)
      message(FATAL_ERROR "Intel MKL support not sufficient. (SCALAPACK with mpich not found)")
    endif()
  endif()
endif()

find_package(GSL 1.16 REQUIRED)
set(Boost_NO_BOOST_CMAKE ON)
find_package(Boost 1.34.0 REQUIRED)

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

#git is required to clone libraries used from github/gitlab
find_package(Git 1.6.5 REQUIRED)
include(cmake/addGitLibs.cmake)

if(${XPED_TENSOR_LIB} STREQUAL "CYCLOPS_TENSOR" AND NOT XPED_BUILD_CYCLOPS)
  find_library(CYCLOPS_LIB NAMES ctf REQUIRED)
  find_library(HPTT_LIB NAMES hptt REQUIRED)
  
  add_library(cyclops_lib::all UNKNOWN IMPORTED)
  target_link_libraries(cyclops_lib::all INTERFACE ${CYCLOPS_LIB})
  target_link_libraries(cyclops_lib::all INTERFACE ${HPTT_LIB})
  set_target_properties(cyclops_lib::all PROPERTIES
    IMPORTED_LOCATION "${CYCLOPS_LIB}"
    )
  message(STATUS ${CYCLOPS_LIB})

  find_path(CYCLOPS_INCLUDE_DIR NAMES ctf.hpp REQUIRED)
  target_include_directories(cyclops_lib::all INTERFACE ${CYCLOPS_INCLUDE_DIR})
  message(STATUS ${CYCLOPS_INCLUDE_DIR})
  include(cmake/PrintProperties.cmake)
  print_target_properties(cyclops_lib::all)
endif()

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

add_subdirectory(src)

if (XPED_BUILD_TESTS)
  find_package(doctest 2.4.6 REQUIRED)
  add_subdirectory(tests)
endif()

if (XPED_BUILD_BENCHMARKS)
  add_subdirectory(benchmarks)
endif()

feature_summary(WHAT ALL
                FILENAME ${CMAKE_BINARY_DIR}/all.log APPEND)
