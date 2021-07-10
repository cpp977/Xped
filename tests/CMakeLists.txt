

include_directories(${XPED_DIR}/src)

# include_directories(${XPED_DIR}/src
#                     ${TOOLS_INCLUDE_DIR}
#                     ${TABULATE_INCLUDE_DIR}
#                     ${DOCTEST_INCLUDE_DIR}
#                     ${EIGEN_INCLUDE_DIR}
#                     ${LRUCACHE_INCLUDE_DIR}
#                     ${SEQ_INCLUDE_DIR})
                    
# if(XPED_USE_MKL)
#   include_directories(${XPED_MKL_INCLUDE_DIR})
# endif()
# if (${XPED_TENSOR_LIB} STREQUAL "CYCLOPS_TENSOR")
#   include_directories(${CYCLOPS_INCLUDE_DIR})
#   if(${XPED_BUILD_CYCLOPS})
#     include_directories(${CYCLOPS_HPTT_INCLUDE_DIR})
#   endif()
# endif()
# if (${XPED_TENSOR_LIB} STREQUAL "ARRAY_TENSOR")
#   include_directories(${ARRAY_INCLUDE_DIR})
# endif()

include(doctest)

enable_testing()

file(GLOB SOURCES "Core/*.hpp"
                  "Symmetry/*.hpp"
                  "Hash/*.hpp"
                  "Interfaces/*.hpp"
                  "MPS/*.hpp"
                  "Util/*.hpp")

add_library(test_suite OBJECT driver.cpp ${SOURCES})
add_dependencies(test_suite spdlog)
if(${XPED_TENSOR_LIB} STREQUAL "CYCLOPS_TENSOR" AND ${XPED_BUILD_CYCLOPS})
  add_dependencies(test_suite cyclops)
endif()
target_link_libraries (test_suite
  project_options
  project_warnings
  doctest::doctest
  XPED_SPDLOG)

if(XPED_USE_MPI)
  target_link_libraries(test_suite MPI::MPI_CXX)
endif()
if(${XPED_TENSOR_LIB} STREQUAL "CYCLOPS_TENSOR")
  target_link_libraries(test_suite cyclops_lib::all)
endif()

#########################
#       basis_test      #
#########################
add_executable(basis_test basis.cpp $<TARGET_OBJECTS:test_suite>)
add_dependencies(basis_test
  tabulate
  spdlog
  yas
  TOOLS
  Eigen
  seq)
                 
if(XPED_ENABLE_LRU_CACHE)
  add_dependencies(basis_test lru_cache)
endif()
if(${XPED_TENSOR_LIB} STREQUAL "ARRAY_TENSOR")
  add_dependencies(basis_test array)
endif()
if(${XPED_TENSOR_LIB} STREQUAL "CYCLOPS_TENSOR" AND ${XPED_BUILD_CYCLOPS})
  add_dependencies(basis_test cyclops)
endif()
target_link_libraries (basis_test
  project_options
  project_warnings
  doctest::doctest
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
  target_link_libraries(basis_test cyclops_lib::all)
elseif(${XPED_TENSOR_LIB} STREQUAL "ARRAY_TENSOR")
  target_link_libraries(basis_test XPED_ARRAY)
endif()
if(XPED_ENABLE_LRU_CACHE)
  target_link_libraries(basis_test XPED_LRUCACHE)
endif()
if(XPED_USE_OPENMP)
  target_link_libraries (basis_test OpenMP::OpenMP_CXX)
endif()
if(XPED_USE_BLAS)
  target_link_libraries (basis_test BLAS::BLAS)
endif()
if(XPED_USE_LAPACK)
  target_link_libraries (basis_test LAPACK::LAPACK)
  if(XPED_USE_LAPACKE)
    target_link_libraries (basis_test LAPACKE::LAPACKE)
  endif()
endif()
if(XPED_USE_MPI)
  target_link_libraries(basis_test MPI::MPI_CXX)
endif()
# doctest_discover_tests(basis_test TEST_PREFIX basis__ ADD_LABELS "1")
add_test(NAME basis COMMAND basis_test)
#########################
#        tree_test      #
#########################
add_executable(tree_test tree.cpp $<TARGET_OBJECTS:test_suite>)
add_dependencies(tree_test
  tabulate
  spdlog
  yas
  TOOLS
  Eigen
  seq)
                 
if(XPED_ENABLE_LRU_CACHE)
  add_dependencies(tree_test lru_cache)
endif()
if(${XPED_TENSOR_LIB} STREQUAL "ARRAY_TENSOR")
  add_dependencies(tree_test array)
endif()
if(${XPED_TENSOR_LIB} STREQUAL "CYCLOPS_TENSOR" AND ${XPED_BUILD_CYCLOPS})
  add_dependencies(tree_test cyclops)
endif()
target_link_libraries (tree_test
  project_options
  project_warnings
  doctest::doctest
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
  target_link_libraries(tree_test cyclops_lib::all)
elseif(${XPED_TENSOR_LIB} STREQUAL "ARRAY_TENSOR")
  target_link_libraries(tree_test XPED_ARRAY)
endif()
if(XPED_ENABLE_LRU_CACHE)
  target_link_libraries(tree_test XPED_LRUCACHE)
endif()
if(XPED_USE_OPENMP)
  target_link_libraries (tree_test OpenMP::OpenMP_CXX)
endif()
if(XPED_USE_BLAS)
  target_link_libraries (tree_test BLAS::BLAS)
endif()
if(XPED_USE_LAPACK)
  target_link_libraries (tree_test LAPACK::LAPACK)
  if(XPED_USE_LAPACKE)
    target_link_libraries (tree_test LAPACKE::LAPACKE)
  endif()
endif()
if(XPED_USE_MPI)
  target_link_libraries(tree_test MPI::MPI_CXX)
endif()
# doctest_discover_tests(tree_test TEST_PREFIX tree__ ADD_LABELS "1")
add_test(NAME tree COMMAND tree_test)

#########################
#       tensor_test     #
#########################
add_executable(tensor_test tensor.cpp $<TARGET_OBJECTS:test_suite>)
add_dependencies(tensor_test
  tabulate
  spdlog
  yas
  TOOLS
  Eigen
  seq)
                 
if(XPED_ENABLE_LRU_CACHE)
  add_dependencies(tensor_test lru_cache)
endif()
if(${XPED_TENSOR_LIB} STREQUAL "ARRAY_TENSOR")
  add_dependencies(tensor_test array)
endif()
if(${XPED_TENSOR_LIB} STREQUAL "CYCLOPS_TENSOR" AND ${XPED_BUILD_CYCLOPS})
  add_dependencies(tensor_test cyclops)
endif()
target_link_libraries (tensor_test
  project_options
  project_warnings
  doctest::doctest
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
  target_link_libraries(tensor_test cyclops_lib::all)
elseif(${XPED_TENSOR_LIB} STREQUAL "ARRAY_TENSOR")
  target_link_libraries(tensor_test XPED_ARRAY)
endif()
if(XPED_ENABLE_LRU_CACHE)
  target_link_libraries(tensor_test XPED_LRUCACHE)
endif()
if(XPED_USE_OPENMP)
  target_link_libraries (tensor_test OpenMP::OpenMP_CXX)
endif()
if(XPED_USE_BLAS)
  target_link_libraries (tensor_test BLAS::BLAS)
endif()
if(XPED_USE_LAPACK)
  target_link_libraries (tensor_test LAPACK::LAPACK)
  if(XPED_USE_LAPACKE)
    target_link_libraries (tensor_test LAPACKE::LAPACKE)
  endif()
endif()
if(XPED_USE_MPI)
  target_link_libraries(tensor_test MPI::MPI_CXX)
endif()
# doctest_discover_tests(tensor_test TEST_PREFIX tensor__ ADD_LABELS "1")
add_test(NAME tensor COMMAND tensor_test)
#########################
#        mps_test       #
#########################
add_executable(mps_test mps.cpp $<TARGET_OBJECTS:test_suite>)
add_dependencies(mps_test
  tabulate
  spdlog
  yas
  TOOLS
  Eigen
  seq)
                 
if(XPED_ENABLE_LRU_CACHE)
  add_dependencies(mps_test lru_cache)
endif()
if(${XPED_TENSOR_LIB} STREQUAL "ARRAY_TENSOR")
  add_dependencies(mps_test array)
endif()
if(${XPED_TENSOR_LIB} STREQUAL "CYCLOPS_TENSOR" AND ${XPED_BUILD_CYCLOPS})
  add_dependencies(mps_test cyclops)
endif()
target_link_libraries (mps_test
  project_options
  project_warnings
  doctest::doctest
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
  target_link_libraries(mps_test cyclops_lib::all)
elseif(${XPED_TENSOR_LIB} STREQUAL "ARRAY_TENSOR")
  target_link_libraries(mps_test XPED_ARRAY)
endif()
if(XPED_ENABLE_LRU_CACHE)
  target_link_libraries(mps_test XPED_LRUCACHE)
endif()
if(XPED_USE_OPENMP)
  target_link_libraries (mps_test OpenMP::OpenMP_CXX)
endif()
if(XPED_USE_BLAS)
  target_link_libraries (mps_test BLAS::BLAS)
endif()
if(XPED_USE_LAPACK)
  target_link_libraries (mps_test LAPACK::LAPACK)
  if(XPED_USE_LAPACKE)
    target_link_libraries (mps_test LAPACKE::LAPACKE)
  endif()
endif()
if(XPED_USE_MPI)
  target_link_libraries(mps_test MPI::MPI_CXX)
endif()
# doctest_discover_tests(mps_test TEST_PREFIX mps__ ADD_LABELS "1")
add_test(NAME MPS COMMAND mps_test)
