add_subdirectory(symmetry)
add_subdirectory(hash)

find_package (Eigen3 REQUIRED NO_MODULE)
find_package(GSL REQUIRED)
find_package(Boost REQUIRED)

include_directories(/home/user/Nextcloud/postdoc/code/dmrg/TOOLS/ /home/user/__LIBS__/cpp-text-table)

file(GLOB SOURCES "./*.cpp"	
                  "./*.hpp"
                  "symmetry/*.hpp"
                  "hash/*.hpp")
add_executable(run main.cpp ${SOURCES})
target_link_libraries (run Eigen3::Eigen GSL::gsl GSL::gslcblas)
