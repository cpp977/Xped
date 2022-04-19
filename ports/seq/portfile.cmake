vcpkg_from_github(
    OUT_SOURCE_PATH SOURCE_PATH
    REPO n0phx/seq
    REF v0.2.1
    SHA512 9cebe2d455f7406c2b4eba19f7c923cc224ac6114f9aaf533d909c885c0d9e7fc3026bb1b8c1f1fa804d40963cc1147f860fe8a5de0183f33cee5481c3ef5cdf
    HEAD_REF master
)

vcpkg_cmake_configure(
    SOURCE_PATH ${SOURCE_PATH}
    PREFER_NINJA
    )
  
vcpkg_cmake_install()

vcpkg_cmake_config_fixup(PACKAGE_NAME "seq" CONFIG_PATH "lib/cmake/seq")
vcpkg_fixup_pkgconfig()

file(INSTALL ${SOURCE_PATH}/LICENSE DESTINATION ${CURRENT_PACKAGES_DIR}/share/seq RENAME copyright)
file(REMOVE_RECURSE "${CURRENT_PACKAGES_DIR}/debug/include")
file(REMOVE_RECURSE "${CURRENT_PACKAGES_DIR}/lib" "${CURRENT_PACKAGES_DIR}/debug/lib" "${CURRENT_PACKAGES_DIR}/debug")
