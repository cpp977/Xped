vcpkg_from_github(
    OUT_SOURCE_PATH SOURCE_PATH
    REPO Reference-ScaLAPACK/scalapack
    REF v2.2.1
    SHA512 de356e69e9d91437d2563ea02d5f18a99bac815644bd5cbb0f5fc8737febf379ce8cd2574f4137876ca5da6723d5452c15d760f82b222b39f0d5e61580094a95
    HEAD_REF master
)

vcpkg_cmake_configure(
    SOURCE_PATH "${SOURCE_PATH}"
    OPTIONS
      -DSCALAPACK_BUILD_TESTS=OFF
      -DCMAKE_Fortran_FLAGS=-fallow-argument-mismatch
)

vcpkg_install_cmake()

vcpkg_fixup_cmake_targets(CONFIG_PATH lib/cmake/scalapack-2.2.1)
# vcpkg_cmake_config_fixup(CONFIG_PATH "lib/cmake/scalapack-2.2.1")

# vcpkg_copy_pdbs()
# file(INSTALL ${SOURCE_PATH}/ DESTINATION ${CURRENT_PACKAGES_DIR}/share/${PORT} RENAME copyright)

file(REMOVE_RECURSE "${CURRENT_PACKAGES_DIR}/debug/include")
# file(REMOVE_RECURSE "${CURRENT_PACKAGES_DIR}/debug/share")
file(INSTALL ${SOURCE_PATH}/LICENSE DESTINATION ${CURRENT_PACKAGES_DIR}/share/${PORT} RENAME copyright)
