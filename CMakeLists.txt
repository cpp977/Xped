cmake_minimum_required(VERSION 3.18)

project(Xped CXX C Fortran)
set(XPED_DIR ${CMAKE_CURRENT_SOURCE_DIR})

include(cmake/StandardProjectSettings.cmake)

include(cmake/PreventInSourceBuilds.cmake)

include(cmake/Functions.cmake)

include(FeatureSummary)
include(CMakeDependentOption)

###################################################################################################################################
###################################                          define options                   #####################################
###################################################################################################################################
option(XPED_COMPILED_LIB "Configure the library as a compiled library. Long compile times." OFF)
add_feature_info("Xped is configured to be a compiled library only." XPED_COMPILED_LIB "")

option(XPED_BUILD_TESTS "Build the tests." ON)
add_feature_info("The tests will be build." XPED_BUILD_TESTS "")

option(XPED_BUILD_BENCHMARKS "Build the benchmarks." ON)
add_feature_info("The benchmarks will be build." XPED_BUILD_BENCHMARKS "")

set(XPED_STORAGE "VecOfMat" CACHE STRING "Used storage for Xped::Tensor.")
set(XPED_SUPPORTED_STORAGES "Contiguous;VecOfMat")
set_property(CACHE XPED_STORAGE PROPERTY STRINGS ${XPED_SUPPORTED_STORAGES})
if(NOT (${XPED_STORAGE} IN_LIST XPED_SUPPORTED_STORAGES))
  message( STATUS "Build wanted to use the storage type ${XPED_STORAGE}.")
  message( FATAL_ERROR "You specified a storage type which is not supported.")  
endif()
add_feature_info("Build uses the storage type ${XPED_STORAGE}" XPED_STORAGE "")

set(XPED_TENSOR_LIB "Eigen" CACHE STRING "Used tensor library for plain tensor operations.")
set(XPED_SUPPORTED_LINA_LIBS "Eigen;Array;Cyclops")
set_property(CACHE XPED_TENSOR_LIB PROPERTY STRINGS ${XPED_SUPPORTED_LINA_LIBS})
if(NOT (${XPED_TENSOR_LIB} IN_LIST XPED_SUPPORTED_LINA_LIBS))
  message( STATUS "Build wanted to use the ${XPED_TENSOR_LIB} library for tensor operations.")
  message( FATAL_ERROR "You specified a tensor library which is not supported.")  
endif()
add_feature_info("Build uses the ${XPED_TENSOR_LIB} library for tensor operations." XPED_TENSOR_LIB "")

set(XPED_MATRIX_LIB "Eigen" CACHE STRING "Used matrix library for plain tensor operations.")
set_property(CACHE XPED_MATRIX_LIB PROPERTY STRINGS ${XPED_SUPPORTED_LINA_LIBS})
if(NOT (${XPED_MATRIX_LIB} IN_LIST XPED_SUPPORTED_LINA_LIBS))
  message( STATUS "Build wanted to use the ${XPED_MATRIX_LIB} library for matrix operations.")
  message( FATAL_ERROR "You specified a matrix library which is not supported.")  
endif()
add_feature_info("Build uses the ${XPED_MATRIX_LIB} library for matrix operations." XPED_MATRIX_LIB "")

set(XPED_VECTOR_LIB "Eigen" CACHE STRING "Used matrix library for plain tensor operations.")
set_property(CACHE XPED_VECTOR_LIB PROPERTY STRINGS ${XPED_SUPPORTED_LINA_LIBS})
if(NOT (${XPED_VECTOR_LIB} IN_LIST XPED_SUPPORTED_LINA_LIBS))
  message( STATUS "Build wanted to use the ${XPED_VECTOR_LIB} library for vector operations.")
  message( FATAL_ERROR "You specified a vector library which is not supported.")  
endif()
add_feature_info("Build uses the ${XPED_VECTOR_LIB} library for vector operations." XPED_VECTOR_LIB "")

create_plain_name(XPED_PLAIN_LIB ${XPED_VECTOR_LIB} ${XPED_MATRIX_LIB} ${XPED_TENSOR_LIB})
message( STATUS "Build will use ${XPED_PLAIN_LIB} for linear algebra operations." )

option(XPED_BUILD_CYCLOPS "Build the cyclops library from source." OFF)
add_feature_info("Cyclops library will be build in ${CMAKE_BINARY_DIR}/thirdparty/cyclops." XPED_BUILD_CYCLOPS "")

option(XPED_ENABLE_CCACHE "Enable a compiler cache if available" OFF)
add_feature_info("Compilation uses ccache to speed up recompilation." XPED_ENABLE_CCACHE "")

option(XPED_USE_OPENMP "Enable openmp parallelization" ON)
option(XPED_USE_MPI "Enable message parsing interface (mpi) parallelization" OFF)
option(XPED_USE_BLAS "Enable blas linking." ON)
if(NOT BLA_VENDOR)
  set(BLA_VENDOR OpenBLAS)
endif()
option(XPED_USE_LAPACK "Enable lapack linking." ON)
option(XPED_USE_SCALAPACK "Enable scalapack linking (only useful for MPI programs)." OFF)
option(XPED_USE_MKL "Enable use of intel math kernel library (MKL)." OFF)
add_feature_info("External MKL libraries from Intel will be linked." XPED_USE_MKL "")
if(XPED_USE_MKL)
  set(XPED_USE_BLAS OFF)
  set(XPED_USE_LAPACK OFF)
  set(XPED_USE_SCALAPACK OFF)
  set(XPED_USE_OPENMP OFF)
  set(XPED_USE_MPI OFF)  
endif()
cmake_dependent_option(XPED_MKL_USE_MPI "Use mkl with intel MPI and Scalapack enabled." OFF XPED_USE_MKL OFF)
add_feature_info("ScaLAPACK of Intel MKL implementation wll be linked." XPED_MKL_USE_MPI "")
add_feature_info("Threading with OpenMP is enabled." XPED_USE_OPENMP "")
add_feature_info("Distributed parallelism with MPI (message passing interface) is enabled." XPED_USE_MPI "")
add_feature_info("External BLAS implementation will be linked." XPED_USE_BLAS "")
add_feature_info("External LAPACK implementation will be linked." XPED_USE_LAPACK "")
add_feature_info("External SCALAPACK implementation will be linked." XPED_USE_SCALAPACK "")

option(XPED_ENABLE_BUILD_WITH_TIME_TRACE "Enable -ftime-trace to generate time tracing .json files on clang" OFF)
add_feature_info("Compiler option -ftime-trace is appended when building with clang." XPED_ENABLE_BUILD_WITH_TIME_TRACE "")

option(XPED_ENABLE_COVERAGE "Enable coverage reporting for gcc/clang" OFF)
add_feature_info("Compile flags for coverage are added." XPED_ENABLE_COVERGAE "")

option(XPED_ENABLE_SANITIZER_ADDRESS "Enable address sanitizer" OFF)
add_feature_info("Address sanitizer is enabled." XPED_ENABLE_SANITIZER_ADDRESS "")
option(XPED_ENABLE_SANITIZER_LEAK "Enable leak sanitizer" OFF)
add_feature_info("Leak sanitizer is enabled." XPED_ENABLE_SANITIZER_LEAK "")
option(XPED_ENABLE_SANITIZER_UNDEFINED_BEHAVIOR "Enable undefined behavior sanitizer" OFF)
add_feature_info("Undefined behavior sanitizer is enabled." XPED_ENABLE_SANITIZER_UNDEFINED_BEHAVIOR "")
option(XPED_ENABLE_SANITIZER_THREAD "Enable thread sanitizer" OFF)
add_feature_info("Thread sanitizer is enabled." XPED_ENABLE_SANITIZER_THREAD "")
option(XPED_ENABLE_SANITIZER_MEMORY "Enable memory sanitizer" OFF)
add_feature_info("Memory sanitizer is created." XPED_ENABLE_SANITIZER_MEMORY "")

option(XPED_ENABLE_DOXYGEN "Enable doxygen doc builds of source" OFF)
add_feature_info("Target for doxygen is enabled." XPED_ENABLE_DOXYGEN "")

option(XPED_ENABLE_CPPCHECK "Enable static analysis with cppcheck" OFF)
add_feature_info("Tool cppcheck is enabled." XPED_ENABLE_CPPCHECKT "")
option(XPED_ENABLE_CLANG_TIDY "Enable static analysis with clang-tidy" OFF)
add_feature_info("Tool clang-tidy is enabled." XPED_ENABLE_CLANG_TIDY "")
option(XPED_ENABLE_INCLUDE_WHAT_YOU_USE "Enable static analysis with include-what-you-use" OFF)
add_feature_info("Tool include-what-you-use is enabled." XPED_ENABLE_INCLUDE_WHAT_YOU_USE "")
option(XPED_ENABLE_CLANG_FORMAT "Enable clang-format target." ON)
add_feature_info("Clang format target is included." XPED_ENABLE_CLANG_FORMAT "")

option(XPED_ENABLE_LRU_CACHE "Use lru cache library from github." OFF)
add_feature_info("Build uses lru cache to cache results for fusion tree manipulations." XPED_ENABLE_LRU_CACHE "")

set(XPED_EFFICIENCY_MODEL "XPED_TIME_EFFICIENT" CACHE STRING "Xped tries to be time efficient.")
set(XPED_SUPPORTED_EFFICIENCY_MODELS "XPED_TIME_EFFICIENT;XPED_MEMORY_EFFICIENT")
set_property(CACHE XPED_EFFICIENCY_MODEL PROPERTY STRINGS ${XPED_SUPPORTED_EFFICIENCY_MODELS})
if(NOT (${XPED_EFFICIENCY_MODEL} IN_LIST XPED_SUPPORTED_EFFICIENCY_MODELS))
  message( FATAL_ERROR "You specified an unsupported efficiency model.")
endif()
add_feature_info("Build uses the ${XPED_EFFICIENCY_MODEL} model." XPED_EFFICIENCY_MODEL "")

option(XPED_USE_LIBCXX "Use libc++ from llvm." OFF)
add_feature_info("c++ standard library implementation from clang (libc++) will be used." XPED_USE_LIBCXX "")
if(XPED_USE_LIBCXX)
  set(USED_LIBCXX libc++)
else()
  set (USED_LIBCXX libstdc++)
endif()
if(NOT CMAKE_CXX_COMPILER_ID MATCHES ".*Clang")
 if(XPED_USE_LIBCXX)
  message( FATAL_ERROR "It is only possible to use libc++ in combination with clang")
 endif()
endif()

option(XPED_USE_AD "Use automatic differentiation (AD) with Xped Tensors." ON)
add_feature_info("Build uses the stan math library for automatic differentitation of algorithms performed with Xped::Tensor." XPED_USE_AD "")

option(XPED_USE_NLO "Use nonlinear optimization algorithms." OFF)
add_feature_info("Build uses nonlinear optimization algorithms." XPED_USE_NLO "")

if(XPED_USE_NLO)
  set(XPED_OPTIM_LIB "ceres" CACHE STRING "Used library for nonlinear gradient-based optimization.")
  set(XPED_SUPPORTED_OPTIM_LIBS "ceres")
  set_property(CACHE XPED_OPTIM_LIB PROPERTY STRINGS ${XPED_SUPPORTED_OPTIM_LIBS})
  if(NOT (${XPED_OPTIM_LIB} IN_LIST XPED_SUPPORTED_OPTIM_LIBS))
    message(STATUS "Build wanted to use the ${XPED_OPTIM_LIB} library for nonlinear optimization.")
    message(FATAL_ERROR "You specified a library which is not supported.")
  endif()
  add_feature_info("Build uses the ${XPED_OPTIM_LIB} library for nonlinear gradient-based optimization." XPED_OPTIM_LIB "")
endif()

set(XPED_LOG_LEVEL "SPDLOG_LEVEL_OFF" CACHE STRING "Compile time log level.")
set(XPED_SUPPORTED_LOG_LEVELS "SPDLOG_LEVEL_TRACE;SPDLOG_LEVEL_DEBUG;SPDLOG_LEVEL_INFO;SPDLOG_LEVEL_WARN;SPDLOG_LEVEL_ERROR;SPDLOG_LEVEL_CRITICAL;SPDLOG_LEVEL_OFF")
set_property(CACHE XPED_LOG_LEVEL PROPERTY STRINGS ${XPED_SUPPORTED_LOG_LEVELS})
if(NOT (${XPED_LOG_LEVEL} IN_LIST XPED_SUPPORTED_LOG_LEVELS))
  message( STATUS "Log level was set to ${XPED_LOG_LEVEL}.")
  message( FATAL_ERROR "You specified a log level which is not supported.")  
endif()
add_feature_info("Log level set to ${XPED_LOG_LEVEL}." XPED_LOG_LEVEL "")
###################################################################################################################################
##########################################                   end options                  #########################################
###################################################################################################################################

if(XPED_USE_OPENMP)
  find_package(OpenMP REQUIRED)
  set_package_properties(OpenMP PROPERTIES
    URL "https://www.openmp.org/"
    DESCRIPTION "Open (M)ulti (P)rocessing."
    TYPE REQUIRED
    PURPOSE "Allows computations to be processed by several threads. This is supported by all backends for plain tensor and matrix operations."
    )
endif()

if(XPED_USE_MPI)
  find_package(MPI REQUIRED)
  execute_process(COMMAND ${MPIEXEC_EXECUTABLE} --version OUTPUT_VARIABLE XPED_MPI_VENDOR_CHECK)
  string(FIND ${XPED_MPI_VENDOR_CHECK} "open-mpi" XPED_USE_OPENMPI)
  if(NOT (${XPED_USE_OPENMPI} EQUAL -1))
    set(XPED_USE_OPENMPI ON)
    set(XPED_USE_MPICH OFF)
    set(XPED_USE_INTELMPI OFF)
    set(XPED_MPI_IMPL "openmpi")
    # message(STATUS "Found openmpi library.")
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
      set(XPED_MPI_IMPL "MPICH")
      # message(STATUS "Found mpich library.")
    endif()
  endif()
  set_package_properties(MPI PROPERTIES
    URL "https://en.wikipedia.org/wiki/Message_Passing_Interface"
    DESCRIPTION "(M)essage (P)assing (I)nterface, implementation by ${XPED_MPI_IMPL}."
    TYPE REQUIRED
    PURPOSE "When using the cyclops library as a backend for plain tensor and matrix operations, computations can be distributed among different cpus using the MPI."
    )
endif()

if(XPED_USE_BLAS)
  find_package(BLAS REQUIRED)
  set_package_properties(BLAS PROPERTIES
    URL "https://en.wikipedia.org/wiki/Basic_Linear_Algebra_Subprograms"
    DESCRIPTION "(B)asic (L)inear (A)lgebra (S)ubprograms, by vendor ${BLA_VENDOR}."
    TYPE REQUIRED
    PURPOSE "The Eigen and cyclops backend can export matrix arithmetics to external BLAS implementations."
    )
endif()

if(XPED_USE_LAPACK)
  find_package(LAPACK REQUIRED)
  set_package_properties(LAPACK PROPERTIES
    URL "https://en.wikipedia.org/wiki/LAPACK"
    DESCRIPTION "(L)inear (A)lgebra (Pack)age, by vendor ${BLA_VENDOR}."
    TYPE REQUIRED
    PURPOSE "The Eigen and cyclops backend can export matrix decompositions (SVD, QR, ...) to external LAPACK implementations."
    )
  set(test_prog_for_lapacke "\#include <lapacke.h> \\nint main() {lapack_int m, n, lda, info\; double *a, *tau\; info = LAPACKE_dgeqrf( LAPACK_COL_MAJOR, m, n, a, lda, tau )\;}")
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
  set_package_properties(LAPACK PROPERTIES
    URL "https://software.intel.com/content/www/us/en/develop/tools/oneapi/components/onemkl.html#gs.acpjiq"
    DESCRIPTION "(M)ath (K)ernel (L)ibrary, by intel."
    TYPE REQUIRED
    PURPOSE "The Eigen and cyclops backend can export matrix decompositions (SVD, QR, ...) to calls to MKL. The cyclops backend can also use the Intel MPI and ScaLAPACK implementation."
    )
  
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

include(cmake/CPM.cmake)

set(CMAKE_POLICY_DEFAULT_CMP0075 NEW)
CPMAddPackage(
  NAME spdlog
  VERSION 1.9.2
  GIT_REPOSITORY https://github.com/gabime/spdlog
  )
target_compile_definitions(spdlog PRIVATE SPDLOG_ACTIVE_LEVEL=${XPED_LOG_LEVEL})
if(CMAKE_CXX_COMPILER_ID MATCHES ".*Clang")
  target_compile_options(spdlog PRIVATE -stdlib=${USED_LIBCXX})
endif()

CPMAddPackage(
  NAME Eigen
  DOWNLOAD_ONLY YES
  GIT_REPOSITORY https://gitlab.com/libeigen/eigen
  GIT_TAG 3.4.0
  # GIT_TAG master
  )
if(Eigen_ADDED)
  add_library(Eigen INTERFACE IMPORTED)
  target_include_directories(Eigen INTERFACE ${Eigen_SOURCE_DIR})
endif()

CPMAddPackage(
  NAME tabulate
  VERSION 1.4
  GIT_REPOSITORY https://github.com/p-ranav/tabulate
  OPTIONS "USE_CPP17 ON"
  )

CPMAddPackage(
  NAME seq
  VERSION 0.2.1
  GIT_REPOSITORY https://github.com/n0phx/seq
  OPTIONS "SEQ_BUILD_TESTS OFF"
  )

if(XPED_ENABLE_LRU_CACHE)
  CPMAddPackage(
    NAME lrucache
    DOWNLOAD_ONLY YES
    GIT_REPOSITORY https://github.com/goldsborough/lru-cache
    GIT_TAG master
    )
  if(lrucache_ADDED)
    add_library(lrucache INTERFACE IMPORTED)
    target_include_directories(lrucache INTERFACE ${lrucache_SOURCE_DIR}/include)
  endif()
endif()

CPMAddPackage(
  NAME yas
  DOWNLOAD_ONLY YES
  GIT_REPOSITORY https://github.com/niXman/yas
  GIT_TAG 7.1.0
)
if(yas_ADDED)
  add_library(yas INTERFACE IMPORTED)
  target_include_directories(yas INTERFACE ${yas_SOURCE_DIR}/include)
endif()

if(XPED_USE_AD)
  CPMAddPackage(
    NAME stan
    GIT_REPOSITORY https://github.com/cpp977/math
    GIT_TAG develop
    #VERSION 4.3.1
    DOWNLOAD_ONLY YES
    )

  if (stan_ADDED)
    # stan has no CMake support, so we create our own target
    
    add_library(stan INTERFACE)
    target_include_directories(stan INTERFACE ${stan_SOURCE_DIR})

    set(TBB_ROOT ${stan_SOURCE_DIR}/lib/tbb_2020.3)
    set(TBB_DIR ${TBB_ROOT}/cmake)
    include(${TBB_DIR}/TBBBuild.cmake)
    tbb_build(TBB_ROOT ${TBB_ROOT} CONFIG_DIR ${TBB_DIR}) # MAKE_ARGS
    find_package(TBB REQUIRED tbb)

    CPMAddPackage(
      NAME sundials
      GIT_REPOSITORY https://github.com/LLNL/sundials
      VERSION 6.0.0
      )

    target_link_libraries(stan INTERFACE TBB::tbb sundials_kinsol)
    # FILE(GLOB stan_sources ${lua_SOURCE_DIR}/*.c)
    # list(REMOVE_ITEM stan_sources "${stan_SOURCE_DIR}/stan.c" "${stan_SOURCE_DIR}/stan.c")
    # add_library(stan STATIC ${stan_sources})
    
    # target_include_directories(stan
    #   PUBLIC
    #   $<BUILD_INTERFACE:${stan_SOURCE_DIR}>
    #   )
  endif()
endif()

if(XPED_USE_NLO)
  if(${XPED_OPTIM_LIB} STREQUAL "ceres")
    CPMAddPackage(
      NAME glog
      GIT_REPOSITORY https://github.com/google/glog
      VERSION 0.5.0
      OPTIONS "WITH_GFLAGS OFF"
      )

    CPMAddPackage(
      NAME ceres
      GIT_REPOSITORY https://github.com/ceres-solver/ceres-solver
      GIT_TAG 2.0.0
      OPTIONS "GFLAGS OFF" "SUITESPARSE OFF" "CXSPARSE OFF" "ACCELERATESPARSE OFF" "CERES_THREADING_MODEL OPENMP" "BUILD_TESTING OFF" "BUILD_EXAMPLES OFF" "BUILD_BENCHMARKS OFF" "MINIGLOG ON"
      )
    target_link_libraries(ceres PRIVATE glog)
  endif()
endif()

find_path(TOOLS_INCLUDE_DIR NAMES ArgParser.h PATHS ${TOOLS_ROOT} PATH_SUFFIXES TOOLS REQUIRED)
add_library(tools INTERFACE IMPORTED)
target_include_directories(tools INTERFACE ${TOOLS_INCLUDE_DIR}/..)
  
if(${XPED_TENSOR_LIB} STREQUAL "Array")
  CPMAddPackage(
    NAME array
    DOWNLOAD_ONLY YES
    GIT_REPOSITORY https://github.com/dsharlet/array
    GIT_TAG a8b42e241ca50bee73e200a952ea8c936d03a856
    )
  if(array_ADDED)
    add_library(ndarray INTERFACE IMPORTED)
    target_include_directories(ndarray INTERFACE ${array_SOURCE_DIR})
  endif()  
endif()

if(${XPED_TENSOR_LIB} STREQUAL "Cyclops" AND NOT XPED_BUILD_CYCLOPS)
  message(STATUS ${CYCLOPS_ROOT})
  message(STATUS ${HPTT_ROOT})
  find_library(CYCLOPS_LIB NAMES ctf PATHS ${CYCLOPS_ROOT} PATH_SUFFIXES lib REQUIRED)
  find_library(HPTT_LIB NAMES hptt PATHS ${HPTT_ROOT} PATH_SUFFIXES lib REQUIRED)
  
  add_library(cyclops_lib::all UNKNOWN IMPORTED)
  target_link_libraries(cyclops_lib::all INTERFACE ${CYCLOPS_LIB})
  target_link_libraries(cyclops_lib::all INTERFACE ${HPTT_LIB})
  set_target_properties(cyclops_lib::all PROPERTIES
    IMPORTED_LOCATION "${CYCLOPS_LIB}"
    )
  message(STATUS ${CYCLOPS_LIB})

  find_path(CYCLOPS_INCLUDE_DIR NAMES ctf.hpp PATHS ${CYCLOPS_ROOT} PATH_SUFFIXES include REQUIRED)
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

feature_summary(WHAT ENABLED_FEATURES PACKAGES_FOUND PACKAGES_NOT_FOUND
  FILENAME ${CMAKE_BINARY_DIR}/ConfigSummary.log
  VAR XPED_FEATURE_SUMMARY
  DESCRIPTION "******************************************************************\n*                  Summary of build configuration                *\n******************************************************************")

#message(STATUS ${XPED_FEATURE_SUMMARY})
