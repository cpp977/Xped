vcpkg_from_github(
    OUT_SOURCE_PATH SOURCE_PATH
    REPO Reference-ScaLAPACK/scalapack
    REF 1f505f7
    SHA512 4f151ff081cd92e732c911c7e5f53676d5edd6f16f1b716a9a24d1b8271d47dcad272d94e951a3ccb3082576027aaf4804d7d9731f11a0a337df2916e9f76dbd
    HEAD_REF master
)

vcpkg_cmake_configure(
    SOURCE_PATH "${SOURCE_PATH}"
    OPTIONS
      -DSCALAPACK_BUILD_TESTS=OFF
      -DCMAKE_Fortran_FLAGS=-fallow-argument-mismatch
)

vcpkg_install_cmake()

vcpkg_fixup_cmake_targets(CONFIG_PATH lib/cmake/scalapack-2.2.2)
# vcpkg_cmake_config_fixup(CONFIG_PATH "lib/cmake/scalapack-2.2.1")

# vcpkg_copy_pdbs()
# file(INSTALL ${SOURCE_PATH}/ DESTINATION ${CURRENT_PACKAGES_DIR}/share/${PORT} RENAME copyright)

file(REMOVE_RECURSE "${CURRENT_PACKAGES_DIR}/debug/include")
# file(REMOVE_RECURSE "${CURRENT_PACKAGES_DIR}/debug/share")
file(INSTALL ${SOURCE_PATH}/LICENSE DESTINATION ${CURRENT_PACKAGES_DIR}/share/${PORT} RENAME copyright)
