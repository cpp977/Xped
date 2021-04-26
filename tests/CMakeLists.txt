find_package(doctest REQUIRED)

include_directories(${XPED_DIR}/src
                    ${TOOLS_INCLUDE_DIR}
                    ${TABULATE_INCLUDE_DIR}
                    ${DOCTEST_INCLUDE_DIR}
                    ${EIGEN_INCLUDE_DIR}
                    ${LRUCACHE_INCLUDE_DIR}
                    ${SEQ_INCLUDE_DIR}
                    ${ARRAY_INCLUDE_DIR})
if(XPED_USE_MKL)
  include_directories(${XPED_MKL_INCLUDE_DIR})
endif()

include(doctest)

enable_testing()

file(GLOB SOURCES "Core/*.hpp"
                  "Symmetry/*.hpp"
                  "Hash/*.hpp"
                  "Interfaces/*.hpp"
                  "MPS/*.hpp"
                  "Util/*.hpp")

add_library(test_suite OBJECT driver.cpp ${SOURCES})
target_link_libraries (test_suite project_options project_warnings doctest::doctest)

add_executable(basis_test basis.cpp $<TARGET_OBJECTS:test_suite>)
add_dependencies(basis_test tabulate TOOLS Eigen seq)
if(XPED_ENABLE_LRU_CACHE)
  add_dependencies(basis_test lru_cache)
endif()
if(${XPED_TENSOR_LIB} STREQUAL "ARRAY_TENSOR")
  add_dependencies(basis_test array)
endif()
target_link_libraries (basis_test project_options project_warnings doctest::doctest GSL::gsl GSL::gslcblas)
if(XPED_USE_OPENMP)
  target_link_libraries (basis_test OpenMP::OpenMP_CXX)
endif()
if(XPED_USE_BLAS)
  target_link_libraries (basis_test BLAS::BLAS)
endif()
if(XPED_USE_LAPACK)
  target_link_libraries (basis_test LAPACK::LAPACK)
endif()
doctest_discover_tests(basis_test TEST_PREFIX basis__ ADD_LABELS "1")

add_executable(tree_test tree.cpp $<TARGET_OBJECTS:test_suite>)
add_dependencies(tree_test tabulate TOOLS Eigen seq)
if(XPED_ENABLE_LRU_CACHE)
  add_dependencies(tree_test lru_cache)
endif()
if(${XPED_TENSOR_LIB} STREQUAL "ARRAY_TENSOR")
  add_dependencies(tree_test array)
endif()
target_link_libraries (tree_test project_options project_warnings doctest::doctest GSL::gsl GSL::gslcblas)
if(XPED_USE_OPENMP)
  target_link_libraries (tree_test OpenMP::OpenMP_CXX)
endif()
if(XPED_USE_BLAS)
  target_link_libraries (tree_test BLAS::BLAS)
endif()
if(XPED_USE_LAPACK)
  target_link_libraries (tree_test LAPACK::LAPACK)
endif()
doctest_discover_tests(tree_test TEST_PREFIX tree__ ADD_LABELS "1")

add_executable(tensor_test tensor.cpp $<TARGET_OBJECTS:test_suite>)
add_dependencies(tensor_test tabulate TOOLS Eigen seq)
if(XPED_ENABLE_LRU_CACHE)
  add_dependencies(tensor_test lru_cache)
endif()
if(${XPED_TENSOR_LIB} STREQUAL "ARRAY_TENSOR")
  add_dependencies(tensor_test array)
endif()
target_link_libraries (tensor_test project_options project_warnings doctest::doctest GSL::gsl GSL::gslcblas)
if(XPED_USE_OPENMP)
  target_link_libraries (tensor_test OpenMP::OpenMP_CXX)
endif()
if(XPED_USE_BLAS)
  target_link_libraries (tensor_test BLAS::BLAS)
endif()
if(XPED_USE_LAPACK)
  target_link_libraries (tensor_test LAPACK::LAPACK)
endif()
doctest_discover_tests(tensor_test TEST_PREFIX tensor__ ADD_LABELS "1")
