include(ExternalProject)

set(TEXTTABLE_ROOT ${CMAKE_BINARY_DIR}/thirdparty/texttable)
set(TEXTTABLE_INCLUDE_DIR ${TEXTTABLE_ROOT}/src/TextTable)

ExternalProject_Add(
        TextTable
        PREFIX ${TEXTTABLE_ROOT}
        GIT_REPOSITORY "https://github.com/haarcuba/cpp-text-table.git"
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

if(ENABLE_LRU_CACHE)
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