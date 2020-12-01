add_subdirectory(util)
add_subdirectory(symmetry)
add_subdirectory(hash)

#find_package (Eigen3 REQUIRED NO_MODULE)
find_package(GSL REQUIRED)
find_package(Boost REQUIRED)

include_directories(${TOOLS_INCLUDE_DIR} ${TEXTTABLE_INCLUDE_DIR} ${EIGEN_INCLUDE_DIR} ${LRUCACHE_INCLUDE_DIR})

file(GLOB SOURCES "./*.cpp"	
                  "./*.hpp"
                  "symmetry/*.hpp"
                  "hash/*.hpp")
add_executable(run main.cpp ${SOURCES})
add_dependencies(run TextTable TOOLS Eigen lru_cache)
target_link_libraries (run GSL::gsl GSL::gslcblas)
