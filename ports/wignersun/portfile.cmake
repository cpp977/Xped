vcpkg_download_distfile(ARCHIVE
    URLS "https://homepages.physik.uni-muenchen.de/~vondelft/Papers/ClebschGordan/ClebschGordan.cpp"
    FILENAME "ClebschGordan.cpp"
    SHA512 bdee7297308423653de3608048b6276932b9a4d78e082f59034c3030431dd9ddebd83c6e5092bb91b0da42658969ffc551324a99b34f726c8aaefee39cb3cbfe
)

file(COPY "${ARCHIVE}" DESTINATION "${CURRENT_BUILDTREES_DIR}/src/")
file(WRITE "${CURRENT_BUILDTREES_DIR}/src/ClebschGordan.hpp" "")

vcpkg_apply_patches(
    SOURCE_PATH "${CURRENT_BUILDTREES_DIR}/src/"
    PATCHES split_to_header_and_souce.patch
)

file(COPY ${CMAKE_CURRENT_LIST_DIR}/CMakeLists.txt DESTINATION "${CURRENT_BUILDTREES_DIR}/src/")
file(COPY ${CMAKE_CURRENT_LIST_DIR}/config.cmake.in DESTINATION "${CURRENT_BUILDTREES_DIR}/src/")

vcpkg_cmake_configure(
    SOURCE_PATH ${CURRENT_BUILDTREES_DIR}/src
    PREFER_NINJA
    )

vcpkg_cmake_install()

file(REMOVE_RECURSE "${CURRENT_PACKAGES_DIR}/debug/include")
file(REMOVE_RECURSE "${CURRENT_PACKAGES_DIR}/debug/share")

FILE(WRITE "${CURRENT_PACKAGES_DIR}/share/${PORT}/copyright" "Unknown")

# file(INSTALL "${SOURCE_PATH}/stan" DESTINATION "${CURRENT_PACKAGES_DIR}/include")

# file(INSTALL "${SOURCE_PATH}/LICENSE.md" DESTINATION "${CURRENT_PACKAGES_DIR}/share/${PORT}" RENAME copyright)

# set(TBB_ROOT ${SOURCE_PATH}/lib/tbb_2020.3)
# set(TBB_DIR ${TBB_ROOT}/cmake)

# include(${TBB_DIR}/TBBBuild.cmake)
# tbb_build(TBB_ROOT ${TBB_ROOT} MAKE_ARGS compiler=g++ CONFIG_DIR ${CURRENT_PACKAGES_DIR}/share/tbb)
