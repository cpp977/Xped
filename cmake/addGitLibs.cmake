include(ExternalProject)

set(TABULATE_ROOT ${CMAKE_BINARY_DIR}/thirdparty/tabulate)
set(TABULATE_INCLUDE_DIR ${TABULATE_ROOT}/src/tabulate/single_include)

ExternalProject_Add(
        tabulate
        PREFIX ${TABULATE_ROOT}
        GIT_REPOSITORY "https://github.com/p-ranav/tabulate"
        GIT_SHALLOW ON
        TIMEOUT 10
        UPDATE_COMMAND ""
        CONFIGURE_COMMAND ""
        BUILD_COMMAND ""
        INSTALL_COMMAND ""
        LOG_DOWNLOAD ON
        LOG_MERGED_STDOUTERR ON
        USES_TERMINAL_DOWNLOAD ON
)

set(SEQ_ROOT ${CMAKE_BINARY_DIR}/thirdparty/seq)
set(SEQ_INCLUDE_DIR ${SEQ_ROOT}/src/seq/include)

ExternalProject_Add(
        seq
        PREFIX ${SEQ_ROOT}
        GIT_REPOSITORY "https://github.com/integricho/seq.git"
        GIT_SHALLOW ON
        TIMEOUT 10
#        UPDATE_COMMAND ${GIT_EXECUTABLE} pull
        UPDATE_COMMAND ""
        CONFIGURE_COMMAND ""
        BUILD_COMMAND ""
        INSTALL_COMMAND ""
        LOG_DOWNLOAD ON
        LOG_MERGED_STDOUTERR ON
        USES_TERMINAL_DOWNLOAD ON
)

if(${XPED_TENSOR_LIB} STREQUAL "ARRAY_TENSOR")
set(ARRAY_ROOT ${CMAKE_BINARY_DIR}/thirdparty/array)
set(ARRAY_INCLUDE_DIR ${ARRAY_ROOT}/src)

ExternalProject_Add(
        array
        PREFIX ${ARRAY_ROOT}
        GIT_REPOSITORY "https://github.com/dsharlet/array.git"
        GIT_SHALLOW ON
        TIMEOUT 10
#        UPDATE_COMMAND ${GIT_EXECUTABLE} pull
        UPDATE_COMMAND ""
        CONFIGURE_COMMAND ""
        BUILD_COMMAND ""
        INSTALL_COMMAND ""
        LOG_DOWNLOAD ON
        LOG_MERGED_STDOUTERR ON
        USES_TERMINAL_DOWNLOAD ON
)
endif()

set(TOOLS_ROOT ${CMAKE_BINARY_DIR}/thirdparty/tools)
set(TOOLS_INCLUDE_DIR ${TOOLS_ROOT}/src/TOOLS)

ExternalProject_Add(
        TOOLS
        PREFIX ${TOOLS_ROOT}
        GIT_REPOSITORY "https://cpp977:A*u1t5o!s@github.com/cpp977/TOOLS.git"
        GIT_SHALLOW ON
        TIMEOUT 10
#        UPDATE_COMMAND ${GIT_EXECUTABLE} pull
        UPDATE_COMMAND ""
        CONFIGURE_COMMAND ""
        BUILD_COMMAND ""
        INSTALL_COMMAND ""
        LOG_DOWNLOAD ON
        LOG_MERGED_STDOUTERR ON
        USES_TERMINAL_DOWNLOAD ON
)

set(EIGEN_ROOT ${CMAKE_BINARY_DIR}/thirdparty/eigen)
set(EIGEN_INCLUDE_DIR ${EIGEN_ROOT}/src/Eigen)

ExternalProject_Add(
        Eigen
        PREFIX ${EIGEN_ROOT}
        GIT_REPOSITORY "https://gitlab.com/libeigen/eigen"
        GIT_SHALLOW ON
        TIMEOUT 10
        UPDATE_COMMAND ""
        CONFIGURE_COMMAND "" 
        BUILD_COMMAND ""
        INSTALL_COMMAND "" 
        TEST_COMMAND ""
        LOG_DOWNLOAD ON
        LOG_MERGED_STDOUTERR ON
        USES_TERMINAL_DOWNLOAD ON
)

if(XPED_ENABLE_LRU_CACHE)
set(LRUCACHE_ROOT ${CMAKE_BINARY_DIR}/thirdparty/lru)
set(LRUCACHE_INCLUDE_DIR ${LRUCACHE_ROOT}/src/lru_cache/include)

ExternalProject_Add(
        lru_cache
        PREFIX ${LRUCACHE_ROOT}
        GIT_REPOSITORY "https://github.com/goldsborough/lru-cache"
        GIT_SHALLOW ON
        TIMEOUT 10
        UPDATE_COMMAND ""
        CONFIGURE_COMMAND "" 
        BUILD_COMMAND ""
        INSTALL_COMMAND "" 
        TEST_COMMAND ""
        LOG_DOWNLOAD ON
        LOG_MERGED_STDOUTERR ON
        USES_TERMINAL_DOWNLOAD ON
)
endif()