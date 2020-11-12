#find_package(Eigen3 REQUIRED NO_MODULE)
find_package(GSL REQUIRED)
find_package(Boost REQUIRED)
find_package(doctest REQUIRED)

include_directories(${TOOLS_INCLUDE_DIR} ${TEXTTABLE_INCLUDE_DIR} ${EIGEN_INCLUDE_DIR})
#include(/home/user/__LIBS__/doctest/scripts/cmake/doctest.cmake)

enable_testing()

file(GLOB SOURCES "./*.hpp"
                  "symmetry/*.hpp"
                  "hash/*.hpp")

add_library(test_suite OBJECT driver.cpp ${SOURCES})

add_executable(basis_test basis.cpp $<TARGET_OBJECTS:test_suite>)
add_dependencies(basis_test TextTable TOOLS)
target_link_libraries (basis_test GSL::gsl GSL::gslcblas doctest::doctest)
#doctest_discover_tests(basis_test)
add_test(NAME basis_test COMMAND basis_test )

add_executable(tree_test tree.cpp $<TARGET_OBJECTS:test_suite>)
add_dependencies(tree_test TextTable TOOLS)
target_link_libraries (tree_test GSL::gsl GSL::gslcblas doctest::doctest)
#doctest_discover_tests(tree_test)
add_test(NAME tree_test COMMAND tree_test)

add_executable(tensor_test tensor.cpp $<TARGET_OBJECTS:test_suite>)
add_dependencies(tensor_test TextTable TOOLS)
target_link_libraries (tensor_test GSL::gsl GSL::gslcblas doctest::doctest)
#doctest_discover_tests(tree_test)
add_test(NAME tensor_test COMMAND tensor_test)

#add_executable(test_suite driver.cpp basis.cpp ${SOURCES})

