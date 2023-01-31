vcpkg_from_github(
    OUT_SOURCE_PATH SOURCE_PATH
    REPO jeremy-rifkin/libassert
    REF v1.1
    SHA512 5a99ab9809bb4f51b23f52b35b35d9f342bb478ac2cbde9da1e9c6d4ab6b732e2ea1def897ab7d6f4f8e4389cba7102ec4fd3e2d8136fbef29d4e9635b563497
    HEAD_REF master
)

vcpkg_cmake_configure(
    SOURCE_PATH "${SOURCE_PATH}"
)

vcpkg_cmake_install()

vcpkg_cmake_config_fixup(CONFIG_PATH "lib${LIB_SUFFIX}/cmake/assert")

# vcpkg_copy_pdbs()

file(REMOVE_RECURSE "${CURRENT_PACKAGES_DIR}/debug/include")
# file(REMOVE_RECURSE "${CURRENT_PACKAGES_DIR}/debug/share")
file(INSTALL ${SOURCE_PATH}/LICENSE DESTINATION ${CURRENT_PACKAGES_DIR}/share/${PORT} RENAME copyright)
