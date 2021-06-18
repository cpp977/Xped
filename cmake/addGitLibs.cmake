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

add_library(XPED_TABULATE INTERFACE)
set_target_properties(XPED_TABULATE PROPERTIES
  INTERFACE_INCLUDE_DIRECTORIES ${TABULATE_INCLUDE_DIR}
  )

set(SPDLOG_ROOT ${CMAKE_BINARY_DIR}/thirdparty/spdlog)
set(SPDLOG_INCLUDE_DIR ${SPDLOG_ROOT}/src/spdlog/include)

ExternalProject_Add(
  spdlog
  PREFIX ${SPDLOG_ROOT}
  GIT_REPOSITORY "https://github.com/gabime/spdlog"
  GIT_TAG origin/v1.x
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

add_library(XPED_SPDLOG INTERFACE)
set_target_properties(XPED_SPDLOG PROPERTIES
  INTERFACE_INCLUDE_DIRECTORIES ${SPDLOG_INCLUDE_DIR}
  )

set(YAS_ROOT ${CMAKE_BINARY_DIR}/thirdparty/yas)
set(YAS_INCLUDE_DIR ${YAS_ROOT}/src/yas/include)

ExternalProject_Add(
  yas
  PREFIX ${YAS_ROOT}
  GIT_REPOSITORY "https://github.com/niXman/yas"
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

add_library(XPED_YAS INTERFACE)
set_target_properties(XPED_YAS PROPERTIES
  INTERFACE_INCLUDE_DIRECTORIES ${YAS_INCLUDE_DIR}
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

add_library(XPED_SEQ INTERFACE)
set_target_properties(XPED_SEQ PROPERTIES
  INTERFACE_INCLUDE_DIRECTORIES ${SEQ_INCLUDE_DIR}
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

  add_library(XPED_ARRAY INTERFACE)
  set_target_properties(XPED_ARRAY PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES ${ARRAY_INCLUDE_DIR}
    )
endif()

if(${XPED_TENSOR_LIB} STREQUAL "CYCLOPS_TENSOR" AND XPED_BUILD_CYCLOPS)
  set(CYCLOPS_ROOT ${CMAKE_BINARY_DIR}/thirdparty/cyclops)
  set(CYCLOPS_INCLUDE_DIR ${CYCLOPS_ROOT}/include)
  set(CYCLOPS_LIB_DIR ${CYCLOPS_ROOT}/lib)
  set(CYCLOPS_HPTT_LIB_DIR ${CYCLOPS_ROOT}/src/cyclops-build/hptt/lib)
  set(CYCLOPS_HPTT_INCLUDE_DIR ${CYCLOPS_ROOT}/src/cyclops-build/hptt/include)
  set(CYCLOPS_SCALAPACK_LIB_DIR ${CYCLOPS_ROOT}/src/cyclops-build/scalapack/build/lib)
  set(CYCLOPS_SCALAPACK_INCLUDE_DIR ${CYCLOPS_ROOT}/src/cyclops-build/scalapack/include)

  set(cmd_configure "OMPI_CXX=${CMAKE_CXX_COMPILER} ../cyclops/configure")
  list(APPEND cmd_configure " CXX=\"mpicxx\"")
  if(XPED_USE_OPENMP)
    list(APPEND cmd_configure " CXXFLAGS=\"-O3 -fPIC -march=native -fopenmp\"")
  else()
    list(APPEND cmd_configure " CXXFLAGS=\"-O3 -fPIC -march=native -DOMP_OFF\"")
  endif()
  if(XPED_USE_BLAS)
    get_target_property(XPED_USED_BLAS_LINKER_FLAGS BLAS::BLAS INTERFACE_LINK_LIBRARIES)
    set(XPED_USED_BLAS_LIBS "")
    foreach(MKL_LIB ${XPED_USED_BLAS_LINKER_FLAGS})
      set(XPED_USED_BLAS_LIBS "${XPED_USED_BLAS_LIBS} ${MKL_LIB}")
    endforeach()
    list(APPEND cmd_configure " LIBS=\"${XPED_USED_BLAS_LIBS}\"")
    list(APPEND cmd_configure " LD_LIBS=\"${XPED_USED_BLAS_LIBS}\"")
  endif()
  list(APPEND cmd_configure " --install-dir=${CYCLOPS_ROOT}")
  list(APPEND cmd_configure " --with-hptt")
  list(APPEND cmd_configure " --build-hptt")
  list(APPEND cmd_configure " --with-scalapack")
  list(APPEND cmd_configure " --build-scalapack")
  
  message(STATUS ${cmd_configure})
  file(WRITE ${CYCLOPS_ROOT}/src/configure.sh ${cmd_configure})

  set(cmd_patch "sed -i 's/\\&//g' ${CYCLOPS_ROOT}/src/cyclops/src/scripts/expand_includes.sh\;")
  list(APPEND cmd_patch " sed -i 's/ctf_all.hpp/ctf_all.hpp 2> \\/dev\\/null/g' ${CYCLOPS_ROOT}/src/cyclops/src/scripts/expand_includes.sh\;")
  list(APPEND cmd_patch " sed -i 's/bool tensor_name_less::operator()(CTF::Idx_Tensor\\* A, CTF::Idx_Tensor\\* B)/bool tensor_name_less::operator()(CTF::Idx_Tensor\\* A, CTF::Idx_Tensor\\* B) const/g' ${CYCLOPS_ROOT}/src/cyclops/src/interface/term.cxx\;")
  list(APPEND cmd_patch " sed -i 's/bool operator()(CTF::Idx_Tensor\\* A, CTF::Idx_Tensor\\* B)/bool operator()(CTF::Idx_Tensor\\* A, CTF::Idx_Tensor\\* B) const/g' ${CYCLOPS_ROOT}/src/cyclops/src/interface/term.h\;")
  list(APPEND cmd_patch " sed -i 's/cmake \\.\\. -DBUILD_STATIC_LIBS=ON -DBUILD_SHARED_LIBS=OFF/cmake \\.\\. -DBUILD_STATIC_LIBS=ON -DBUILD_SHARED_LIBS=OFF -DCMAKE_Fortran_FLAGS=-fallow-argument-mismatch/g' ${CYCLOPS_ROOT}/src/cyclops/configure\;")
  list(APPEND cmd_patch " sed -i 's/read -p \\\"found/echo \\\"found/g' ${CYCLOPS_ROOT}/src/cyclops/configure\;")
  list(APPEND cmd_patch " sed -i 's/overwrite?  (Y\\/N)? \\\" -n 1 -r/overwrite.\\\"/g' ${CYCLOPS_ROOT}/src/cyclops/configure\;")
  list(APPEND cmd_patch " sed -i 's/if \\[\\[ \\$REPLY =~ \\^\\[Yy]\\$ ]]/if \\[\\[ 1 == 1 ]]/g' ${CYCLOPS_ROOT}/src/cyclops/configure\;")
  file(WRITE ${CYCLOPS_ROOT}/src/patch.sh ${cmd_patch})
  
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
    PATCH_COMMAND bash ../patch.sh
    #         UPDATE_COMMAND ${GIT_EXECUTABLE} pull
    UPDATE_COMMAND ""
    #          CONFIGURE_COMMAND ../cyclops/configure CXX="mpicxx -cxx=${CMAKE_CXX_COMPILER}" CXXFLAGS=-march=native --install-dir=${CYCLOPS_ROOT} --with-hptt --build-hptt
    CONFIGURE_COMMAND bash ../configure.sh
    #	  CONFIGURE_COMMAND ""
    BUILD_COMMAND bash ../cmd_make.sh
    #	  BUILD_COMMAND ""
    INSTALL_COMMAND make install
    #	  INSTALL_COMMAND ""
    LOG_DOWNLOAD ON
    LOG_MERGED_STDOUTERR ON
    USES_TERMINAL_DOWNLOAD ON
    )
  add_library(cyclops_lib::cyclops_lib UNKNOWN IMPORTED)
  set_target_properties(cyclops_lib::cyclops_lib PROPERTIES
    IMPORTED_LOCATION ${CYCLOPS_LIB_DIR}/libctf.a
    )
  file(MAKE_DIRECTORY ${CYCLOPS_INCLUDE_DIR})
  target_include_directories(cyclops_lib::cyclops_lib INTERFACE ${CYCLOPS_INCLUDE_DIR})
  add_library(cyclops_lib::hptt UNKNOWN IMPORTED)
  set_target_properties(cyclops_lib::hptt PROPERTIES
    IMPORTED_LOCATION ${CYCLOPS_HPTT_LIB_DIR}/libhptt.a
    )
  file(MAKE_DIRECTORY ${CYCLOPS_HPTT_INCLUDE_DIR})
  target_include_directories(cyclops_lib::hptt INTERFACE ${CYCLOPS_HPTT_INCLUDE_DIR})

  add_library(cyclops_lib::scalapack UNKNOWN IMPORTED)
  set_target_properties(cyclops_lib::scalapack PROPERTIES
    IMPORTED_LOCATION ${CYCLOPS_SCALAPACK_LIB_DIR}/libscalapack.a
    )
  target_link_libraries(cyclops_lib::scalapack INTERFACE gfortran)
  file(MAKE_DIRECTORY ${CYCLOPS_SCALAPACK_INCLUDE_DIR})

  add_library(cyclops_lib::all INTERFACE IMPORTED)
  set_property(TARGET cyclops_lib::all PROPERTY
    INTERFACE_LINK_LIBRARIES cyclops_lib::cyclops_lib cyclops_lib::hptt cyclops_lib::scalapack
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

add_library(XPED_TOOLS INTERFACE)
set_target_properties(XPED_TOOLS PROPERTIES
  INTERFACE_INCLUDE_DIRECTORIES ${TOOLS_INCLUDE_DIR}
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

add_library(XPED_EIGEN INTERFACE)
set_target_properties(XPED_EIGEN PROPERTIES
  INTERFACE_INCLUDE_DIRECTORIES ${EIGEN_INCLUDE_DIR}
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
  
  add_library(XPED_LRUCACHE INTERFACE)
  set_target_properties(XPED_LRUCACHE PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES ${LRUCACHE_INCLUDE_DIR}
    )
endif()
