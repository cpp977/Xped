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

#include(/home/user/__LIBS__/doctest/scripts/cmake/doctest.cmake)

enable_testing()

file(GLOB SOURCES "./*.hpp"
                  "Symmetry/*.hpp"
                  "Hash/*.hpp")

add_library(test_suite OBJECT driver.cpp ${SOURCES})
add_dependencies(test_suite doctest)
target_link_libraries (test_suite project_options project_warnings)

add_executable(basis_test basis.cpp $<TARGET_OBJECTS:test_suite>)
add_dependencies(basis_test tabulate TOOLS Eigen doctest seq)
if(XPED_ENABLE_LRU_CACHE)
  add_dependencies(basis_test lru_cache)
endif()
if(${XPED_TENSOR_LIB} STREQUAL "ARRAY_TENSOR")
  add_dependencies(basis_test array)
endif()
target_link_libraries (basis_test project_options project_warnings GSL::gsl GSL::gslcblas)
if(XPED_USE_OPENMP)
  target_link_libraries (run OpenMP::OpenMP_CXX)
endif()
#doctest_discover_tests(basis_test)
add_test(NAME basis_test COMMAND basis_test )

add_executable(tree_test tree.cpp $<TARGET_OBJECTS:test_suite>)
add_dependencies(tree_test tabulate TOOLS Eigen doctest seq)
if(XPED_ENABLE_LRU_CACHE)
  add_dependencies(tree_test lru_cache)
endif()
if(${XPED_TENSOR_LIB} STREQUAL "ARRAY_TENSOR")
  add_dependencies(tree_test array)
endif()
target_link_libraries (tree_test project_options project_warnings GSL::gsl GSL::gslcblas)
if(XPED_USE_OPENMP)
  target_link_libraries (run OpenMP::OpenMP_CXX)
endif()
#doctest_discover_tests(tree_test)
add_test(NAME tree_test COMMAND tree_test)

add_executable(tensor_test tensor.cpp $<TARGET_OBJECTS:test_suite>)
add_dependencies(tensor_test tabulate TOOLS Eigen doctest seq)
if(XPED_ENABLE_LRU_CACHE)
  add_dependencies(tensor_test lru_cache)
endif()
if(${XPED_TENSOR_LIB} STREQUAL "ARRAY_TENSOR")
  add_dependencies(tensor_test array)
endif()
target_link_libraries (tensor_test project_options project_warnings GSL::gsl GSL::gslcblas)
if(XPED_USE_OPENMP)
  target_link_libraries (run OpenMP::OpenMP_CXX)
endif()
#doctest_discover_tests(tree_test)
add_test(NAME tensor_test COMMAND tensor_test)
