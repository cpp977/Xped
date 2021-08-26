include_directories(${XPED_DIR}/src)

file(GLOB SOURCES "Core/*.hpp"
                  "Symmetry/*.hpp"
                  "Hash/*.hpp"
                  "Interfaces/*.hpp"
                  "MPS/*.hpp"
                  "Util/*.hpp")
                
add_executable(benchmark
               bench.cpp
               ${SOURCES})
               
add_dependencies(benchmark
  tabulate
  spdlog
  yas
#  TOOLS
  Eigen
  seq)
                 
if(XPED_ENABLE_LRU_CACHE)
  add_dependencies(benchmark lru_cache)
endif()
if(${XPED_TENSOR_LIB} STREQUAL "ARRAY_TENSOR")
  add_dependencies(benchmark array)
endif()
if(${XPED_TENSOR_LIB} STREQUAL "CYCLOPS_TENSOR" AND ${XPED_BUILD_CYCLOPS})
  add_dependencies(benchmark cyclops)
endif()


target_link_libraries (benchmark
  project_options
  project_warnings
  GSL::gsl
  GSL::gslcblas
  Boost::headers
  XPED_EIGEN
#  XPED_TOOLS
  XPED_TABULATE
  XPED_SPDLOG
  XPED_YAS
  XPED_SEQ
  )

if((CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 9) OR CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
  message(STATUS "Adding explicit linker flag for cpp-17 std::filesystem.")
  target_link_libraries(benchmark stdc++fs)
endif()
  
if(${XPED_TENSOR_LIB} STREQUAL "CYCLOPS_TENSOR")
  target_link_libraries(benchmark cyclops_lib::all)
elseif(${XPED_TENSOR_LIB} STREQUAL "ARRAY_TENSOR")
  target_link_libraries(benchmark XPED_ARRAY)
endif()

if(XPED_ENABLE_LRU_CACHE)
  target_link_libraries(benchmark XPED_LRUCACHE)
endif()

if(XPED_USE_MPI)
  target_link_libraries(benchmark MPI::MPI_CXX)
endif()

if(XPED_USE_OPENMP)
  target_link_libraries (benchmark OpenMP::OpenMP_CXX)
endif()

if(XPED_USE_BLAS)
  target_link_libraries (benchmark BLAS::BLAS)
endif()

if(XPED_USE_LAPACK)
  target_link_libraries (benchmark LAPACK::LAPACK)
  if(XPED_USE_LAPACKE)
    target_link_libraries (benchmark LAPACKE::LAPACKE)
  endif()
endif()

if(XPED_USE_MKL)
  target_link_libraries(benchmark mkl::mkl_intel_32bit_omp_dyn)
  if(XPED_MKL_USE_MPI)
    target_link_libraries(benchmark mkl::scalapack_mpich_intel_32bit_omp_dyn)
  endif()
endif()
