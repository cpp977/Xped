if(NOT XPED_COMPILED_LIB)
  add_library(Xped INTERFACE)
  target_include_directories(Xped BEFORE INTERFACE ${XPED_DIR}/src)
  target_include_directories(Xped BEFORE INTERFACE ${XPED_DIR}/include)
  target_include_directories(Xped BEFORE INTERFACE ${YAS_INCLUDE_DIRS})
  target_link_libraries(Xped INTERFACE
    project_options
    project_warnings
    GSL::gsl
    GSL::gslcblas
    Boost::boost
    Eigen3::Eigen
    tabulate::tabulate
    spdlog::spdlog
    seq::seq
    WignerSUN::wignersun
    libdeps
    HighFive
    HighFive_libdeps
    HighFive_HighFive
    )
  if(${XPED_TENSOR_LIB} STREQUAL "Cyclops")
    target_link_libraries(Xped INTERFACE cyclops_lib::all)
  elseif(${XPED_TENSOR_LIB} STREQUAL "Array")
    target_include_directories(Xped BEFORE INTERFACE ${ARRAY_INCLUDE_DIRS})
  endif()

  if(XPED_USE_AD)
    target_link_libraries(Xped INTERFACE SUNDIALS::kinsol)
    target_link_libraries(Xped INTERFACE TBB::tbb)
    target_include_directories(Xped BEFORE INTERFACE ${STAN_MATH_INCLUDE_DIRS})
    # target_compile_definitions(Xped INTERFACE STAN_THREADS)
  endif()

  if(XPED_USE_NLO)
    if(${XPED_OPTIM_LIB} STREQUAL "ceres")
      target_link_libraries(Xped INTERFACE Ceres::ceres)
    endif()
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
    ${XPED_DIR}/src/Core/TensorBase.cpp
    ${XPED_DIR}/src/Core/TensorHelper.cpp
    ${XPED_DIR}/src/Core/Tensor.cpp
    ${XPED_DIR}/src/Util/Mpi.cpp
    ${XPED_DIR}/src/MPS/Mps.cpp
    ${XPED_DIR}/src/MPS/MpsAlgebra.cpp
    ${XPED_DIR}/src/MPS/MpsContractions.cpp
    ${XPED_DIR}/src/PEPS/PEPSContractions.cpp
    ${XPED_DIR}/src/PEPS/CTM.cpp
    ${XPED_DIR}/src/PEPS/iPEPS.cpp
    ${XPED_DIR}/src/PEPS/Pattern.cpp
    ${XPED_DIR}/src/PEPS/UnitCell.cpp
    ${XPED_DIR}/src/Interfaces/MatrixInterface_${XPED_MATRIX_LIB}_impl.cpp
    ${XPED_DIR}/src/Interfaces/TensorInterface_${XPED_TENSOR_LIB}_impl.cpp
    ${XPED_DIR}/src/Interfaces/VectorInterface_${XPED_VECTOR_LIB}_impl.cpp
    ${XPED_DIR}/src/Interfaces/PlainInterface_${XPED_PLAIN_LIB}_impl.cpp
    )
  target_include_directories(Xped BEFORE PUBLIC ${XPED_DIR}/include)
  target_include_directories(Xped BEFORE PUBLIC ${YAS_INCLUDE_DIRS})
  if(XPED_USE_AD)
    target_include_directories(Xped BEFORE INTERFACE ${STAN_MATH_INCLUDE_DIRS})
  endif()
  target_link_libraries(Xped PUBLIC
    project_options
    project_warnings
    GSL::gsl
    GSL::gslcblas
    Boost::boost
    Eigen3::Eigen
    tabulate::tabulate
    spdlog::spdlog
    seq::seq
    )

  if(${XPED_TENSOR_LIB} STREQUAL "Cyclops")
    target_link_libraries(Xped PUBLIC cyclops_lib::all)
  elseif(${XPED_TENSOR_LIB} STREQUAL "Array")
    target_include_directories(Xped BEFORE INTERFACE ${ARRAY_INCLUDE_DIRS})
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

if(${XPED_TENSOR_LIB} STREQUAL "Cyclops" AND ${XPED_BUILD_CYCLOPS})
  add_dependencies(Xped cyclops)
endif()

add_executable(run main.cpp)

target_link_libraries (run Xped)
