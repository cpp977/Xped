if(NOT XPED_COMPILED_LIB)
  add_library(Xped INTERFACE ${XPED_DIR}/src/Util/Logging.cpp)
  target_include_directories(Xped BEFORE INTERFACE ${XPED_DIR}/src)
  target_include_directories(Xped BEFORE INTERFACE ${XPED_DIR}/include)
  target_include_directories(Xped BEFORE INTERFACE ${YAS_INCLUDE_DIRS})
  if(XPED_USE_GSL_COEFFS)
    target_link_libraries(Xped INTERFACE GSL::gsl GSL::gslcblas)
  endif()
  target_link_libraries(
    Xped
    INTERFACE project_options
              project_warnings
              Boost::boost
              Eigen3::Eigen
              tabulate::tabulate
              spdlog::spdlog
              seq::seq
              WignerSUN::wignersun
              HighFive::HighFive
              libassert::assert)
  if(${XPED_TENSOR_LIB} STREQUAL "Cyclops")
    target_link_libraries(Xped INTERFACE ${CYCLOPS_LIB})
    target_include_directories(Xped BEFORE INTERFACE ${CYCLOPS_INCLUDE_DIR})
  elseif(${XPED_TENSOR_LIB} STREQUAL "Array")
    target_include_directories(Xped BEFORE INTERFACE ${ARRAY_INCLUDE_DIRS})
  endif()

  # if(XPED_USE_AD)
  if(XPED_USE_EXTERNAL_STAN)
    target_link_libraries(Xped INTERFACE SUNDIALS::kinsol)
    target_link_libraries(Xped INTERFACE TBB::tbb)
    target_include_directories(Xped BEFORE INTERFACE ${STAN_MATH_INCLUDE_DIRS})
    # target_compile_definitions(Xped INTERFACE STAN_THREADS)
  else()
    # bundled self-contained AD engine: drop-in stan-math API subset, no deps
    target_include_directories(Xped BEFORE INTERFACE ${XPED_DIR}/include/selfad)
  endif()
  # endif()

  if(XPED_USE_NLO)
    if(${XPED_OPTIM_LIB} STREQUAL "ceres")
      if(XPED_USE_EXTERNAL_CERES)
        target_link_libraries(Xped INTERFACE Ceres::ceres)
      else()
        # bundled self-contained optimizer: drop-in ceres API subset, no deps
        target_include_directories(Xped BEFORE INTERFACE ${XPED_DIR}/include/selfnlo)
      endif()
    endif()
  endif()

  if(XPED_USE_MPI)
    target_link_libraries(Xped INTERFACE MPI::MPI_CXX)
  endif()

  if(XPED_USE_OPENMP)
    target_link_libraries(Xped INTERFACE OpenMP::OpenMP_CXX)
  endif()

  if(XPED_USE_BLAS)
    target_link_libraries(Xped INTERFACE BLAS::BLAS)
  endif()

  if(XPED_USE_LAPACK)
    target_link_libraries(Xped INTERFACE LAPACK::LAPACK)
    if(XPED_USE_LAPACKE)
      target_link_libraries(Xped INTERFACE LAPACKE::LAPACKE)
    endif()
  endif()

  if(XPED_USE_SCALAPACK)
    target_link_libraries(Xped INTERFACE scalapack)
  endif()

  if(XPED_USE_MKL)
    target_link_libraries(Xped INTERFACE MKL::MKL)
    if(XPED_MKL_USE_MPI)
      target_link_libraries(Xped INTERFACE MKL::MKL)
    endif()
  endif()
else()
  add_library(
    Xped
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
    ${XPED_DIR}/src/PEPS/iPEPS.cpp
    ${XPED_DIR}/src/PEPS/CTM.cpp
    ${XPED_DIR}/src/PEPS/LinearAlgebra.cpp
    ${XPED_DIR}/src/PEPS/PEPSContractions.cpp
    ${XPED_DIR}/src/PEPS/Pattern.cpp
    ${XPED_DIR}/src/PEPS/UnitCell.cpp
    ${XPED_DIR}/src/PEPS/CTMSolver.cpp
    ${XPED_DIR}/src/PEPS/TMatrix.cpp
    ${XPED_DIR}/src/PEPS/TimePropagator.cpp
    ${XPED_DIR}/src/PEPS/SimpleUpdate.cpp
    ${XPED_DIR}/src/Physics/SiteOperator.cpp
    ${XPED_DIR}/src/Interfaces/MatrixInterface_${XPED_MATRIX_LIB}_impl.cpp
    ${XPED_DIR}/src/Interfaces/TensorInterface_${XPED_TENSOR_LIB}_impl.cpp
    ${XPED_DIR}/src/Interfaces/VectorInterface_${XPED_VECTOR_LIB}_impl.cpp
    ${XPED_DIR}/src/Interfaces/PlainInterface_${XPED_PLAIN_LIB}_impl.cpp
    ${XPED_DIR}/src/Util/Mpi.cpp
    ${XPED_DIR}/src/Util/Logging.cpp)

  target_include_directories(Xped BEFORE PUBLIC ${XPED_DIR}/include)
  target_include_directories(Xped BEFORE PUBLIC ${YAS_INCLUDE_DIRS})
  if(XPED_USE_EXTERNAL_STAN)
    target_link_libraries(Xped INTERFACE SUNDIALS::kinsol)
    target_link_libraries(Xped INTERFACE TBB::tbb)
    target_include_directories(Xped BEFORE INTERFACE ${STAN_MATH_INCLUDE_DIRS})
  else()
    # bundled self-contained AD engine: drop-in stan-math API subset, no deps
    target_include_directories(Xped BEFORE PUBLIC ${XPED_DIR}/include/selfad)
  endif()

  if(XPED_USE_GSL_COEFFS)
    target_link_libraries(Xped PUBLIC GSL::gsl GSL::gslcblas)
  endif()
  if(XPED_USE_NLO)
    if(${XPED_OPTIM_LIB} STREQUAL "ceres")
      if(XPED_USE_EXTERNAL_CERES)
        target_link_libraries(Xped PUBLIC Ceres::ceres)
      else()
        # bundled self-contained optimizer: drop-in ceres API subset, no deps
        target_include_directories(Xped BEFORE PUBLIC ${XPED_DIR}/include/selfnlo)
      endif()
    endif()
  endif()
  target_link_libraries(
    Xped
    PUBLIC project_options
           project_warnings
           Boost::boost
           Eigen3::Eigen
           tabulate::tabulate
           spdlog::spdlog
           seq::seq
           libassert::assert
           HighFive::HighFive
           WignerSUN::wignersun
           libassert::assert)

  if(XPED_USE_NLO)
    if(${XPED_OPTIM_LIB} STREQUAL "ceres")
      target_link_libraries(Xped INTERFACE Ceres::ceres)
    endif()
  endif()

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
    target_link_libraries(Xped PUBLIC OpenMP::OpenMP_CXX)
  endif()

  if(XPED_USE_BLAS)
    target_link_libraries(Xped PUBLIC BLAS::BLAS)
  endif()

  if(XPED_USE_LAPACK)
    target_link_libraries(Xped PUBLIC LAPACK::LAPACK)
    if(XPED_USE_LAPACKE)
      target_link_libraries(Xped PUBLIC LAPACKE::LAPACKE)
    endif()
  endif()

  if(XPED_USE_MKL)
    target_link_libraries(Xped PUBLIC mkl::mkl_intel_32bit_omp_dyn)
    if(XPED_MKL_USE_MPI)
      target_link_libraries(Xped
                            PUBLIC mkl::scalapack_mpich_intel_32bit_omp_dyn)
    endif()
  endif()
endif()

add_executable(run main.cpp)

target_link_libraries(run Xped)
