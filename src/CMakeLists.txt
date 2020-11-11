add_subdirectory(util)
add_subdirectory(symmetry)
add_subdirectory(hash)

find_package (Eigen3 REQUIRED NO_MODULE)
find_package(GSL REQUIRED)
find_package(Boost REQUIRED)

set(TEXTTABLE_ROOT ${CMAKE_BINARY_DIR}/thirdparty/texttable)
#set(TEXTTABLE_INCLUDE_DIR ${TEXTTABLE_ROOT}/bin/include)
#message(${TEXTTABLE_INCLUDE_DIR})

include_directories(/home/user/Nextcloud/postdoc/code/dmrg/TOOLS/ ${TEXTTABLE_INCLUDE_DIR})

file(GLOB SOURCES "./*.cpp"	
                  "./*.hpp"
                  "symmetry/*.hpp"
                  "hash/*.hpp")
add_executable(run main.cpp ${SOURCES})
add_dependencies(run TextTable)
target_link_libraries (run Eigen3::Eigen GSL::gsl GSL::gslcblas)
