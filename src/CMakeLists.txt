add_subdirectory(Util)
add_subdirectory(Hash)
add_subdirectory(Interfaces)
add_subdirectory(MPS)
add_subdirectory(Core)
add_subdirectory(Symmetry)

include_directories(${XPED_DIR}/src)

file(GLOB SOURCES "Core/*.hpp"
                  "Symmetry/*.hpp"
                  "Hash/*.hpp"
                  "Interfaces/*.hpp"
                  "MPS/*.hpp"
                  "Util/*.hpp")
                
add_executable(run
               main.cpp
               ${SOURCES})
               
add_dependencies(run
  tabulate
  spdlog
  yas
  TOOLS
  Eigen
  seq)
                 
if(XPED_ENABLE_LRU_CACHE)
  add_dependencies(run lru_cache)
endif()
if(${XPED_TENSOR_LIB} STREQUAL "ARRAY_TENSOR")
  add_dependencies(run array)
endif()
if(${XPED_TENSOR_LIB} STREQUAL "CYCLOPS_TENSOR" AND ${XPED_BUILD_CYCLOPS})
  add_dependencies(run cyclops)
endif()


target_link_libraries (run
  project_options
  project_warnings
  GSL::gsl
  GSL::gslcblas
  XPED_EIGEN
  XPED_TOOLS
  XPED_TABULATE
  XPED_SPDLOG
  XPED_YAS
  XPED_SEQ
  )

if(${XPED_TENSOR_LIB} STREQUAL "CYCLOPS_TENSOR")
  target_link_libraries(run cyclops_lib::all)
elseif(${XPED_TENSOR_LIB} STREQUAL "ARRAY_TENSOR")
  target_link_libraries(run XPED_ARRAY)
endif()

if(XPED_ENABLE_LRU_CACHE)
  target_link_libraries(run XPED_LRUCACHE)
endif()

if(XPED_USE_MPI)
  target_link_libraries(run MPI::MPI_CXX)
endif()

if(XPED_USE_BLAS)
  target_link_libraries (run BLAS::BLAS)
endif()

if(XPED_USE_LAPACK)
  target_link_libraries (run LAPACK::LAPACK)
  if(XPED_USE_LAPACKE)
    target_link_libraries (run LAPACKE::LAPACKE)
  endif()
endif()

if(XPED_USE_OPENMP)
  target_link_libraries (run OpenMP::OpenMP_CXX)
endif()
