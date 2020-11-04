#add_subdirectory(../src)

find_package(Eigen3 REQUIRED NO_MODULE)
find_package(GSL REQUIRED)
find_package(Boost REQUIRED)
find_package(doctest REQUIRED)

include_directories(/home/user/Nextcloud/postdoc/code/dmrg/TOOLS/ /home/user/__LIBS__/cpp-text-table)

file(GLOB SOURCES "./*.cpp"	
                  "./*.hpp"
                  "symmetry/*.hpp"
                  "hash/*.hpp")
add_executable(tree_test tree.cpp ${SOURCES})
target_link_libraries (tree_test Eigen3::Eigen GSL::gsl GSL::gslcblas doctest::doctest)
