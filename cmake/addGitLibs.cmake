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
#         UPDATE_COMMAND ${GIT_EXECUTABLE} pull
          UPDATE_COMMAND ""
          CONFIGURE_COMMAND ""
          BUILD_COMMAND ""
          INSTALL_COMMAND ""
          LOG_DOWNLOAD ON
          LOG_MERGED_STDOUTERR ON
          USES_TERMINAL_DOWNLOAD ON
  )
endif()

if(${XPED_TENSOR_LIB} STREQUAL "CYCLOPS_TENSOR" AND XPED_BUILD_CYCLOPS)
  set(CYCLOPS_ROOT ${CMAKE_BINARY_DIR}/thirdparty/cyclops)
  set(CYCLOPS_INCLUDE_DIR ${CYCLOPS_ROOT}/src/cyclops/include)
  set(CYCLOPS_LIB_DIR ${CYCLOPS_ROOT}/lib)
  set(CYCLOPS_HPTT_LIB_DIR ${CYCLOPS_ROOT}/src/cyclops-build/hptt/lib)
  set(CYCLOPS_HPTT_INCLUDE_DIR ${CYCLOPS_ROOT}/src/cyclops-build/hptt/include)

  set(cmd_configure "OMPI_CXX=${CMAKE_CXX_COMPILER} ../cyclops/configure")
  list(APPEND cmd_configure " CXX=\"mpicxx\"")
  if(XPED_USE_OPENMP)
    list(APPEND cmd_configure " CXXFLAGS=\"-O3 -march=native -fopenmp\"")
  else()
    list(APPEND cmd_configure " CXXFLAGS=\"-O3 -march=native -DOMP_OFF\"")
  endif()
  if(XPED_USE_BLAS)
    list(APPEND cmd_configure " LIBS=\"${BLAS_LIBRARIES}\"")
    list(APPEND cmd_configure " LD_LIBS=\"${BLAS_LIBRARIES}\"")
  endif()
  list(APPEND cmd_configure " --install-dir=${CYCLOPS_ROOT}")
  list(APPEND cmd_configure " --with-hptt")
  list(APPEND cmd_configure " --build-hptt")
  
  message(STATUS ${cmd_configure})
  file(WRITE ${CYCLOPS_ROOT}/src/configure.sh ${cmd_configure})
#  get_target_property(MAIN_CXXFLAGS project_options INTERFACE_COMPILE_OPTIONS)
#  message(STATUS ${MAIN_CXXFLAGS})

  set(cmd_make "OMPI_CXX=${CMAKE_CXX_COMPILER} make")
  file(WRITE ${CYCLOPS_ROOT}/src/cmd_make.sh ${cmd_make})
  
  ExternalProject_Add(
          cyclops
          PREFIX ${CYCLOPS_ROOT}
          BINARY_DIR "${CYCLOPS_ROOT}/src/cyclops-build"
          SOURCE_DIR "${CYCLOPS_ROOT}/src/cyclops"
          INSTALL_DIR "${CYCLOPS_ROOT}/src/cyclops-build"
          GIT_REPOSITORY "https://github.com/cyclops-community/ctf.git"
          GIT_SHALLOW ON
          TIMEOUT 10
#          PATCH_COMMAND bash ../patch.sh
#         UPDATE_COMMAND ${GIT_EXECUTABLE} pull
          UPDATE_COMMAND ""
#          CONFIGURE_COMMAND ../cyclops/configure CXX="mpicxx -cxx=${CMAKE_CXX_COMPILER}" CXXFLAGS=-march=native --install-dir=${CYCLOPS_ROOT} --with-hptt --build-hptt
          CONFIGURE_COMMAND bash ../configure.sh
          BUILD_COMMAND bash ../cmd_make.sh
          INSTALL_COMMAND make install
          LOG_DOWNLOAD ON
          LOG_MERGED_STDOUTERR ON
          USES_TERMINAL_DOWNLOAD ON
  )
  add_library(cyclops_lib::cyclops_lib UNKNOWN IMPORTED)
  set_target_properties(cyclops_lib::cyclops_lib PROPERTIES
    #INTERFACE_INCLUDE_DIRECTORIES ${MyLibBar_INCLUDE_DIR}
    IMPORTED_LOCATION ${CYCLOPS_LIB_DIR}/libctf.a
    )
  add_library(cyclops_lib::hptt UNKNOWN IMPORTED)
  set_target_properties(cyclops_lib::hptt PROPERTIES
    #INTERFACE_INCLUDE_DIRECTORIES ${MyLibBar_INCLUDE_DIR}
    IMPORTED_LOCATION ${CYCLOPS_HPTT_LIB_DIR}/libhptt.a
    )
  add_library(cyclops_lib::all INTERFACE IMPORTED)
  set_property(TARGET cyclops_lib::all PROPERTY
  INTERFACE_LINK_LIBRARIES cyclops_lib::cyclops_lib cyclops_lib::hptt)
  
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