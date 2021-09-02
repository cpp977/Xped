include(doctest)
enable_testing()

add_library(test_suite OBJECT driver.cpp)
target_include_directories(test_suite BEFORE PUBLIC ${XPED_DIR}/src)

if(${XPED_TENSOR_LIB} STREQUAL "CYCLOPS_TENSOR" AND ${XPED_BUILD_CYCLOPS})
  add_dependencies(test_suite cyclops)
endif()


target_link_libraries (test_suite PUBLIC
  project_options
  project_warnings
  doctest::doctest
  GSL::gsl
  GSL::gslcblas
  Eigen
  tabulate
  spdlog
  yas
  seq)

if(${XPED_TENSOR_LIB} STREQUAL "CYCLOPS_TENSOR")
  target_link_libraries(test_suite PUBLIC cyclops_lib::all)
elseif(${XPED_TENSOR_LIB} STREQUAL "ARRAY_TENSOR")
  target_link_libraries(test_suite PUBLIC ndarray)
endif()
if(XPED_ENABLE_LRU_CACHE)
  target_link_libraries(test_suite PUBLIC lrucache)
endif()
if(XPED_USE_MPI)
  target_link_libraries(test_suite PUBLIC MPI::MPI_CXX)
endif()
if(XPED_USE_OPENMP)
  target_link_libraries (test_suite PUBLIC OpenMP::OpenMP_CXX)
endif()
if(XPED_USE_BLAS)
  target_link_libraries (test_suite PUBLIC BLAS::BLAS)
endif()
if(XPED_USE_LAPACK)
  target_link_libraries (test_suite PUBLIC LAPACK::LAPACK)
  if(XPED_USE_LAPACKE)
    target_link_libraries (test_suite PUBLIC LAPACKE::LAPACKE)
  endif()
endif()
if(XPED_USE_MKL)
  target_link_libraries(test_suite PUBLIC mkl::mkl_intel_32bit_omp_dyn)
  if(XPED_MKL_USE_MPI)
    target_link_libraries(test_suite PUBLIC mkl::scalapack_mpich_intel_32bit_omp_dyn)
  endif()
endif()

#########################
#       basis_test      #
#########################
add_executable(basis_test basis.cpp $<TARGET_OBJECTS:test_suite>)
target_link_libraries (basis_test test_suite)

# doctest_discover_tests(basis_test TEST_PREFIX basis__ ADD_LABELS "1")
if (XPED_USE_MPI)
  add_test(NAME basis COMMAND ${MPIEXEC_EXECUTABLE} --mca opal_warn_on_missing_libcuda 0 ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS} ./basis_test ${MPIEXEC_POSTFLAGS})
else ()
  add_test(NAME basis COMMAND basis_test)
endif()
#########################
#        tree_test      #
#########################
add_executable(tree_test tree.cpp $<TARGET_OBJECTS:test_suite>)
target_link_libraries (tree_test test_suite)

# doctest_discover_tests(tree_test TEST_PREFIX tree__ ADD_LABELS "1")
if(XPED_USE_MPI)
  add_test(NAME tree COMMAND ${MPIEXEC_EXECUTABLE} --mca opal_warn_on_missing_libcuda 0 ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS} ./tree_test ${MPIEXEC_POSTFLAGS})
else()
  add_test(NAME tree COMMAND tree_test)
endif()

#########################
#       tensor_test     #
#########################
add_executable(tensor_test tensor.cpp $<TARGET_OBJECTS:test_suite>)
target_link_libraries (tensor_test test_suite)

# doctest_discover_tests(tensor_test TEST_PREFIX tensor__ ADD_LABELS "1")
if(XPED_USE_MPI)
  add_test(NAME tensor COMMAND ${MPIEXEC_EXECUTABLE} --mca opal_warn_on_missing_libcuda 0 ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS} ./tensor_test ${MPIEXEC_POSTFLAGS})
else()
  add_test(NAME tensor COMMAND tensor_test)
endif()

#########################
#        mps_test       #
#########################
add_executable(mps_test mps.cpp $<TARGET_OBJECTS:test_suite>)
target_link_libraries (mps_test test_suite)

# doctest_discover_tests(mps_test TEST_PREFIX mps__ ADD_LABELS "1")
if(XPED_USE_MPI)
  add_test(NAME MPS COMMAND ${MPIEXEC_EXECUTABLE} --mca opal_warn_on_missing_libcuda 0 ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS} ./mps_test ${MPIEXEC_POSTFLAGS})
else()
  add_test(NAME MPS COMMAND mps_test)
endif()
