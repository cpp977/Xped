include(ExternalProject)

if(${XPED_TENSOR_LIB} STREQUAL "Cyclops" AND XPED_BUILD_CYCLOPS)
  find_package(Wget REQUIRED) #needed for extra downloads performed by cyclops build.
  
  set(CYCLOPS_ROOT ${CMAKE_BINARY_DIR}/thirdparty/cyclops)
  set(CYCLOPS_INCLUDE_DIR ${CYCLOPS_ROOT}/include)
  set(CYCLOPS_LIB_DIR ${CYCLOPS_ROOT}/lib)
  set(CYCLOPS_HPTT_LIB_DIR ${CYCLOPS_ROOT}/src/cyclops-build/hptt/lib)
  set(CYCLOPS_HPTT_INCLUDE_DIR ${CYCLOPS_ROOT}/src/cyclops-build/hptt/include)
  set(CYCLOPS_SCALAPACK_LIB_DIR ${CYCLOPS_ROOT}/src/cyclops-build/scalapack/build/lib)
  set(CYCLOPS_SCALAPACK_INCLUDE_DIR ${CYCLOPS_ROOT}/src/cyclops-build/scalapack/include)

  if(XPED_USE_OPENMPI)
    set(cmd_configure "OMPI_CXX=${CMAKE_CXX_COMPILER} ../cyclops/configure")
  elseif(XPED_USE_MPICH)
    set(cmd_configure "MPICH_CXX=${CMAKE_CXX_COMPILER} ../cyclops/configure")
  elseif(XPED_MKL_USE_MPI)
    set(cmd_configure "I_MPI_CXX=${CMAKE_CXX_COMPILER} ../cyclops/configure")
  endif()
  list(APPEND cmd_configure " CXX=\"${MPI_CXX_COMPILER}\"")
  
  set(XPED_USED_CXXFLAGS "")
  if (${CMAKE_BUILD_TYPE} STREQUAL "Release")
    list(APPEND XPED_USED_CXXFLAGS " -O3")
  elseif(${CMAKE_BUILD_TYPE} STREQUAL "Debug")
    list(APPEND XPED_USED_CXXFLAGS " -O0 -g")
  endif()
  list(APPEND XPED_USED_CXXFLAGS " -fPIC")
  list(APPEND XPED_USED_CXXFLAGS " -march=native")
  list(APPEND XPED_USED_CXXFLAGS " ${OpenMP_CXX_FLAGS}")
  list(APPEND cmd_configure " CXXFLAGS=\"${XPED_USED_CXXFLAGS}\"")
  
  if(XPED_USE_BLAS)
    get_target_property(XPED_USED_BLAS_LINKER_FLAGS BLAS::BLAS INTERFACE_LINK_LIBRARIES)
    set(XPED_USED_BLAS_LIBS "")
    foreach(BLAS_LIB ${XPED_USED_BLAS_LINKER_FLAGS})
      set(XPED_USED_BLAS_LIBS "${XPED_USED_BLAS_LIBS} ${BLAS_LIB}")
    endforeach()
    list(APPEND cmd_configure " LIBS=\"${XPED_USED_BLAS_LIBS}\"")
    list(APPEND cmd_configure " LD_LIBS=\"${XPED_USED_BLAS_LIBS}\"")
  endif()
  if(XPED_USE_MKL)
    get_target_property(XPED_USED_MKL_LINKER_FLAGS mkl::mkl_intel_32bit_omp_dyn INTERFACE_LINK_LIBRARIES)
    set(XPED_USED_MKL_LIBS "")
    foreach(MKL_LIB ${XPED_USED_MKL_LINKER_FLAGS})
      string(FIND ${MKL_LIB} ".so" IS_CMAKE_TARGET)
      if(${IS_CMAKE_TARGET} EQUAL -1)
        get_target_property(XPED_USED_MKL_LINKER_FLAGS_REC ${MKL_LIB} INTERFACE_LINK_LIBRARIES)
        foreach(MKL_LIB_REC ${XPED_USED_MKL_LINKER_FLAGS_REC})
          list(APPEND XPED_USED_MKL_LIBS " ${MKL_LIB_REC}")
        endforeach()
      else()
        list(APPEND XPED_USED_MKL_LIBS " ${MKL_LIB}")
      endif()
    endforeach()
    if(XPED_MKL_USE_MPI)
      get_target_property(XPED_USED_MKL_SCALAPACK_LINKER_FLAGS mkl::scalapack_mpich_intel_32bit_omp_dyn INTERFACE_LINK_LIBRARIES)
      foreach(MKL_LIB ${XPED_USED_MKL_SCALAPACK_LINKER_FLAGS})
        string(FIND ${MKL_LIB} ".so" IS_CMAKE_TARGET)
        if(${IS_CMAKE_TARGET} EQUAL -1)
          get_target_property(XPED_USED_MKL_SCALAPACK_LINKER_FLAGS_REC ${MKL_LIB} INTERFACE_LINK_LIBRARIES)
          foreach(MKL_LIB_REC ${XPED_USED_MKL_SCALAPACK_LINKER_FLAGS_REC})
            string(FIND ${MKL_LIB_REC} ".so" IS_CMAKE_TARGET_REC)
            if(${IS_CMAKE_TARGET_REC} EQUAL -1)
              get_target_property(XPED_USED_MKL_SCALAPACK_LINKER_FLAGS_REC_REC ${MKL_LIB_REC} INTERFACE_LINK_LIBRARIES)
              foreach(MKL_LIB_REC_REC ${XPED_USED_MKL_SCALAPACK_LINKER_FLAGS_REC_REC})
                list(APPEND XPED_USED_MKL_LIBS " ${MKL_LIB_REC_REC}")
              endforeach()
            else()
              list(APPEND XPED_USED_MKL_LIBS " ${MKL_LIB_REC}")
            endif()
          endforeach()
        else()
          list(APPEND XPED_USED_MKL_LIBS " ${MKL_LIB}")
        endif()
      endforeach()
    endif()
    list(REMOVE_DUPLICATES XPED_USED_MKL_LIBS)
    
    # list(APPEND cmd_configure " LIB_PATH=\"-L$ENV{MKLROOT}/lib/intel64\"")
    # if(XPED_MKL_USE_MPI)
    #   # list(APPEND cmd_configure " LIBS=\"-lmkl_scalapack_lp64 -lmkl_blacs_intelmpi_lp64 -liomp5 -lpthread -lm -ldl\"")
    #   list(APPEND cmd_configure " LIBS=\"-lmkl_scalapack_lp64 -lmkl_blacs_intelmpi_lp64 -ltbb -lstdc++ -lpthread -lm -ldl\"")
    # else()
    #   list(APPEND cmd_configure " LIBS=\"-liomp5 -lpthread -lm -ldl\"")
    # endif()
   list(APPEND cmd_configure " LIBS=\"${XPED_USED_MKL_LIBS}\"")
   list(APPEND cmd_configure " LD_LIBS=\"${XPED_USED_MKL_LIBS}\"")
  endif()
  
  list(APPEND cmd_configure " --install-dir=${CYCLOPS_ROOT}")
  list(APPEND cmd_configure " --with-hptt")
  list(APPEND cmd_configure " --build-hptt")
  list(APPEND cmd_configure " --with-scalapack")
  if(XPED_USE_MPI)
    list(APPEND cmd_configure " --build-scalapack")
  endif()
  
  # message(STATUS ${cmd_configure})
  file(WRITE ${CYCLOPS_ROOT}/src/configure.sh ${cmd_configure})

  set(cmd_patch "sed -i 's/\\&//g' ${CYCLOPS_ROOT}/src/cyclops/src/scripts/expand_includes.sh\;")
  list(APPEND cmd_patch " sed -i 's/ctf_all.hpp/ctf_all.hpp 2> \\/dev\\/null/g' ${CYCLOPS_ROOT}/src/cyclops/src/scripts/expand_includes.sh\;")
  list(APPEND cmd_patch " sed -i 's/bool tensor_name_less::operator()(CTF::Idx_Tensor\\* A, CTF::Idx_Tensor\\* B)/bool tensor_name_less::operator()(CTF::Idx_Tensor\\* A, CTF::Idx_Tensor\\* B) const/g' ${CYCLOPS_ROOT}/src/cyclops/src/interface/term.cxx\;")
  list(APPEND cmd_patch " sed -i 's/bool operator()(CTF::Idx_Tensor\\* A, CTF::Idx_Tensor\\* B)/bool operator()(CTF::Idx_Tensor\\* A, CTF::Idx_Tensor\\* B) const/g' ${CYCLOPS_ROOT}/src/cyclops/src/interface/term.h\;")
  string(REPLACE "/" "\\/" XPED_MPIEXEC_EXECUTABLE ${MPIEXEC_EXECUTABLE})
  string(REPLACE "/" "\\/" XPED_FC_COMPILER ${CMAKE_Fortran_COMPILER})
  string(REPLACE "/" "\\/" XPED_CXX_COMPILER ${CMAKE_CXX_COMPILER})
  string(REPLACE "/" "\\/" XPED_C_COMPILER ${CMAKE_C_COMPILER})
  if(${CMAKE_Fortran_COMPILER_VERSION} VERSION_GREATER_EQUAL 10)
    list(APPEND cmd_patch " sed -i 's/cmake \\.\\. -DBUILD_STATIC_LIBS=ON -DBUILD_SHARED_LIBS=OFF/CC=${XPED_C_COMPILER} CXX=${XPED_CXX_COMPILER} FC=${XPED_FC_COMPILER} cmake \\.\\. -DBUILD_STATIC_LIBS=ON -DBUILD_SHARED_LIBS=OFF -DCMAKE_Fortran_FLAGS=-fallow-argument-mismatch -DMPIEXEC_EXECUTABLE=${XPED_MPIEXEC_EXECUTABLE}/g' ${CYCLOPS_ROOT}/src/cyclops/configure\;")
  else()
    list(APPEND cmd_patch " sed -i 's/cmake \\.\\. -DBUILD_STATIC_LIBS=ON -DBUILD_SHARED_LIBS=OFF/CC=${XPED_C_COMPILER} CXX=${XPED_CXX_COMPILER} FC=${XPED_FC_COMPILER} cmake \\.\\. -DBUILD_STATIC_LIBS=ON -DBUILD_SHARED_LIBS=OFF -DMPIEXEC_EXECUTABLE=${XPED_MPIEXEC_EXECUTABLE}/g' ${CYCLOPS_ROOT}/src/cyclops/configure\;")
  endif()
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
    GIT_PROGRESS ON
    LOG_MERGED_STDOUTERR ON
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

  if(XPED_USE_MPI)
    add_library(cyclops_lib::scalapack UNKNOWN IMPORTED)
    set_target_properties(cyclops_lib::scalapack PROPERTIES
      IMPORTED_LOCATION ${CYCLOPS_SCALAPACK_LIB_DIR}/libscalapack.a
      )
    target_link_libraries(cyclops_lib::scalapack INTERFACE gfortran)
    file(MAKE_DIRECTORY ${CYCLOPS_SCALAPACK_INCLUDE_DIR})
  endif()

  add_library(cyclops_lib::all INTERFACE IMPORTED)
  # set_property(TARGET cyclops_lib::all PROPERTY
  #   INTERFACE_LINK_LIBRARIES cyclops_lib::cyclops_lib)
  target_link_libraries(cyclops_lib::all INTERFACE cyclops_lib::cyclops_lib)
  target_link_libraries(cyclops_lib::all INTERFACE cyclops_lib::hptt)
  if(XPED_USE_MPI)
    target_link_libraries(cyclops_lib::all INTERFACE cyclops_lib::scalapack)
  endif()
endif()
