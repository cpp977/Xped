add_subdirectory(util)
add_subdirectory(symmetry)
add_subdirectory(hash)
add_subdirectory(interfaces)

#find_package (Eigen3 REQUIRED NO_MODULE)
find_package(GSL REQUIRED)
find_package(Boost REQUIRED)

include_directories(${TOOLS_INCLUDE_DIR} ${TEXTTABLE_INCLUDE_DIR} ${EIGEN_INCLUDE_DIR} ${LRUCACHE_INCLUDE_DIR} ${SEQ_INCLUDE_DIR} ${ARRAY_INCLUDE_DIR})

file(GLOB SOURCES "./*.cpp"	
                  "./*.hpp"
                  "symmetry/*.hpp"
                  "hash/*.hpp"
                  "interfaces/*.hpp")
                  
add_executable(run
               main.cpp
               ${SOURCES})
               
add_dependencies(run
                 TextTable
                 TOOLS
                 Eigen
                 seq
                 array)
if(ENABLE_LRU_CACHE)
add_dependencies(run lru_cache)
endif()

target_link_libraries (run
                       project_options
                       project_warnings
                       GSL::gsl
                       GSL::gslcblas)
