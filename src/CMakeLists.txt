add_executable(run main.cpp)

target_include_directories(run BEFORE PUBLIC ${XPED_DIR}/src)

if(${XPED_TENSOR_LIB} STREQUAL "CYCLOPS_TENSOR" AND ${XPED_BUILD_CYCLOPS})
  add_dependencies(run cyclops)
endif()

target_link_libraries (run
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
  target_link_libraries(run cyclops_lib::all)
elseif(${XPED_TENSOR_LIB} STREQUAL "ARRAY_TENSOR")
  target_link_libraries(run ndarray)
endif()

if(XPED_ENABLE_LRU_CACHE)
  target_link_libraries(run lrucache)
endif()

if(XPED_USE_MPI)
  target_link_libraries(run MPI::MPI_CXX)
endif()

if(XPED_USE_OPENMP)
  target_link_libraries (run OpenMP::OpenMP_CXX)
endif()

if(XPED_USE_BLAS)
  target_link_libraries (run BLAS::BLAS)
endif()

if(XPED_USE_LAPACK)
  target_link_libraries (run LAPACK::LAPACK)
  if(XPED_USE_LAPACKE)
    target_link_libraries (run LAPACKE::LAPACKE)
  endif()
endif()

if(XPED_USE_MKL)
  target_link_libraries(run mkl::mkl_intel_32bit_omp_dyn)
  if(XPED_MKL_USE_MPI)
    target_link_libraries(run mkl::scalapack_mpich_intel_32bit_omp_dyn)
  endif()
endif()
