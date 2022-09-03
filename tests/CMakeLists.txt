include(doctest)
enable_testing()

add_library(test_suite OBJECT driver.cpp)
target_link_libraries (test_suite PUBLIC
  Xped
  doctest::doctest
  )

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
#       physics_test      #
#########################
add_executable(physics_test physics.cpp $<TARGET_OBJECTS:test_suite>)
target_link_libraries (physics_test test_suite)

# doctest_discover_tests(basis_test TEST_PREFIX basis__ ADD_LABELS "1")
if (XPED_USE_MPI)
  add_test(NAME physics COMMAND ${MPIEXEC_EXECUTABLE} --mca opal_warn_on_missing_libcuda 0 ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS} ./physics_test ${MPIEXEC_POSTFLAGS})
else ()
  add_test(NAME physics COMMAND physics_test)
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

#########################
#        peps_ad_test      #
#########################
add_executable(peps_ad peps_ad.cpp)# $<TARGET_OBJECTS:test_suite>)
target_link_libraries (peps_ad Xped)
