# header-only library
vcpkg_from_github(
    OUT_SOURCE_PATH SOURCE_PATH
    REPO doctest/doctest
    REF 1fd763bcc903fb55447cd4c78435dd4dec281ba2
    SHA512 300d1804a9fdecbba9d2b3183cab95170f834e61519539a1691b0aeff14a49c210c3b2e686cd5612b39411eb51a01b3336d16866ce8585caed922a3c4b9a7ff0
    HEAD_REF dev
)

vcpkg_cmake_configure(
    SOURCE_PATH "${SOURCE_PATH}"
    DISABLE_PARALLEL_CONFIGURE
    OPTIONS
        -DDOCTEST_WITH_TESTS=OFF
)

vcpkg_cmake_install()

vcpkg_cmake_config_fixup(CONFIG_PATH lib/cmake/${PORT})

file(REMOVE_RECURSE "${CURRENT_PACKAGES_DIR}/debug" "${CURRENT_PACKAGES_DIR}/lib")

vcpkg_install_copyright(FILE_LIST "${SOURCE_PATH}/LICENSE.txt")
