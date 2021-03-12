add_subdirectory(util)
add_subdirectory(symmetry)
add_subdirectory(hash)
add_subdirectory(interfaces)

#find_package (Eigen3 REQUIRED NO_MODULE)
find_package(GSL REQUIRED)
find_package(Boost REQUIRED)

include_directories(${XPED_DIR}/src ${TOOLS_INCLUDE_DIR} ${TEXTTABLE_INCLUDE_DIR} ${EIGEN_INCLUDE_DIR} ${LRUCACHE_INCLUDE_DIR} ${SEQ_INCLUDE_DIR} ${ARRAY_INCLUDE_DIR})

file(GLOB SOURCES "./*.cpp"	
                  "./*.hpp"
                  "symmetry/*.hpp"
                  "hash/*.hpp"
                  "interfaces/*.hpp")
                  
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
if(XPED_USE_OPENMP)
target_link_libraries (run OpenMP::OpenMP_CXX)
endif()
