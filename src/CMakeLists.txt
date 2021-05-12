add_subdirectory(Util)
add_subdirectory(Hash)
add_subdirectory(Interfaces)
add_subdirectory(MPS)
add_subdirectory(Core)
add_subdirectory(Symmetry)

include_directories(${XPED_DIR}/src
                    ${TOOLS_INCLUDE_DIR}
                    ${TABULATE_INCLUDE_DIR}
                    ${EIGEN_INCLUDE_DIR}
                    ${LRUCACHE_INCLUDE_DIR}
                    ${SEQ_INCLUDE_DIR}
                    ${ARRAY_INCLUDE_DIR})
if(XPED_USE_MKL)
  include_directories(${XPED_MKL_INCLUDE_DIR})
endif()

if (${XPED_TENSOR_LIB} STREQUAL "CYCLOPS_TENSOR")
  include_directories(home/user/__LIBS__/ctf/include)
endif()

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
                 TOOLS
                 Eigen
                 seq)
                 
if(XPED_ENABLE_LRU_CACHE)
  add_dependencies(run lru_cache)
endif()
if(${XPED_TENSOR_LIB} STREQUAL "ARRAY_TENSOR")
  add_dependencies(run array)
endif()


target_link_libraries (run
                       project_options
                       project_warnings
                       GSL::gsl
                       GSL::gslcblas)

if (${XPED_TENSOR_LIB} STREQUAL "CYCLOPS_TENSOR")
  target_link_libraries(run /home/user/__LIBS__/ctf/lib/libctf.a)
endif()

if(XPED_USE_OPENMPI OR ${XPED_TENSOR_LIB} STREQUAL "CYCLOPS_TENSOR")
  target_link_libraries(run MPI::MPI_CXX)
endif()

if(XPED_USE_BLAS)
  target_link_libraries (run BLAS::BLAS)
endif()

if(XPED_USE_LAPACK)
  target_link_libraries (run LAPACK::LAPACK)
endif()

if(XPED_USE_OPENMP)
  target_link_libraries (run OpenMP::OpenMP_CXX)
endif()
