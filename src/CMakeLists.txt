if(NOT XPED_COMPILED_LIB)
  add_library(Xped INTERFACE)
  target_include_directories(Xped BEFORE INTERFACE ${XPED_DIR}/src)
  target_include_directories(Xped BEFORE INTERFACE ${XPED_DIR}/include)
  target_link_libraries(Xped INTERFACE
    project_options
    project_warnings
    GSL::gsl
    GSL::gslcblas
    Eigen
    tabulate
    spdlog
    yas
    seq
    )
  if(${XPED_TENSOR_LIB} STREQUAL "CYCLOPS_TENSOR")
    target_link_libraries(Xped INTERFACE cyclops_lib::all)
  elseif(${XPED_TENSOR_LIB} STREQUAL "ARRAY_TENSOR")
    target_link_libraries(Xped INTERFACE ndarray)
  endif()

  if(XPED_ENABLE_LRU_CACHE)
    target_link_libraries(Xped INTERFACE lrucache)
  endif()

  if(XPED_USE_MPI)
    target_link_libraries(Xped INTERFACE MPI::MPI_CXX)
  endif()

  if(XPED_USE_OPENMP)
    target_link_libraries (Xped INTERFACE OpenMP::OpenMP_CXX)
  endif()

  if(XPED_USE_BLAS)
    target_link_libraries (Xped INTERFACE BLAS::BLAS)
  endif()

  if(XPED_USE_LAPACK)
    target_link_libraries (Xped INTERFACE LAPACK::LAPACK)
    if(XPED_USE_LAPACKE)
      target_link_libraries (Xped INTERFACE LAPACKE::LAPACKE)
    endif()
  endif()

  if(XPED_USE_MKL)
    target_link_libraries(Xped INTERFACE mkl::mkl_intel_32bit_omp_dyn)
    if(XPED_MKL_USE_MPI)
      target_link_libraries(Xped INTERFACE mkl::scalapack_mpich_intel_32bit_omp_dyn)
    endif()
  endif()
else()
  add_library(Xped
    ${XPED_DIR}/src/Symmetry/SU2.cpp
    ${XPED_DIR}/src/Core/Basis.cpp
    ${XPED_DIR}/src/Core/Qbasis.cpp
    ${XPED_DIR}/src/Core/FusionTree.cpp
    ${XPED_DIR}/src/Core/treepair.cpp
    ${XPED_DIR}/src/Core/XpedBase.cpp
    ${XPED_DIR}/src/Core/XpedHelper.cpp
    ${XPED_DIR}/src/Core/Xped.cpp
    ${XPED_DIR}/src/Util/Mpi.cpp
    )
  target_include_directories(Xped BEFORE PUBLIC ${XPED_DIR}/include)
  target_link_libraries(Xped PUBLIC
    project_options
    project_warnings
    GSL::gsl
    GSL::gslcblas
    Eigen
    tabulate
    spdlog
    yas
    seq
    )

  if(${XPED_TENSOR_LIB} STREQUAL "CYCLOPS_TENSOR")
    target_link_libraries(Xped PUBLIC cyclops_lib::all)
  elseif(${XPED_TENSOR_LIB} STREQUAL "ARRAY_TENSOR")
    target_link_libraries(Xped PUBLIC ndarray)
  endif()

  if(XPED_ENABLE_LRU_CACHE)
    target_link_libraries(Xped PUBLIC lrucache)
  endif()

  if(XPED_USE_MPI)
    target_link_libraries(Xped PUBLIC MPI::MPI_CXX)
  endif()

  if(XPED_USE_OPENMP)
    target_link_libraries (Xped PUBLIC OpenMP::OpenMP_CXX)
  endif()

  if(XPED_USE_BLAS)
    target_link_libraries (Xped PUBLIC BLAS::BLAS)
  endif()

  if(XPED_USE_LAPACK)
    target_link_libraries (Xped PUBLIC LAPACK::LAPACK)
    if(XPED_USE_LAPACKE)
      target_link_libraries (Xped PUBLIC LAPACKE::LAPACKE)
    endif()
  endif()

  if(XPED_USE_MKL)
    target_link_libraries(Xped PUBLIC mkl::mkl_intel_32bit_omp_dyn)
    if(XPED_MKL_USE_MPI)
      target_link_libraries(Xped PUBLIC mkl::scalapack_mpich_intel_32bit_omp_dyn)
    endif()
  endif()
endif()

if(${XPED_TENSOR_LIB} STREQUAL "CYCLOPS_TENSOR" AND ${XPED_BUILD_CYCLOPS})
  add_dependencies(Xped cyclops)
endif()

add_executable(run main.cpp)

target_include_directories(run BEFORE PUBLIC ${XPED_DIR}/inlcude)

target_link_libraries (run Xped)
