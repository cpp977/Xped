vcpkg_from_github(
    OUT_SOURCE_PATH SOURCE_PATH
    REPO LLNL/sundials
    REF v6.0.0
    # SHA512 a009bc77f31ad426cf02670f06363058ef83dfd6fd84c868e4c8713ccb453ceff481f98266b46c7a6de0ef4d4ecca74a8c8e78150b88cecc7cce41ed8f056dbb
    SHA512 14e1b42aa6a1bb1c54a13bf2b1a9c5a6ab92bf8017878ee67e0d4ef22d58d9d41fd6fd439877216ebe86b7f1d98c6f27049b60b87741e9127d46eafb238eedda
    HEAD_REF main
    PATCHES
    install-dlls-in-bin.patch
    cmake_config.patch
)

string(COMPARE EQUAL "${VCPKG_LIBRARY_LINKAGE}" "static" SUN_BUILD_STATIC)
string(COMPARE EQUAL "${VCPKG_LIBRARY_LINKAGE}" "dynamic" SUN_BUILD_SHARED)

vcpkg_cmake_configure(
    SOURCE_PATH ${SOURCE_PATH}
    OPTIONS 
        -D_BUILD_EXAMPLES=OFF
        -DBUILD_STATIC_LIBS=${SUN_BUILD_STATIC}
        -DBUILD_SHARED_LIBS=${SUN_BUILD_SHARED}
        -DENABLE_CALIPER=OFF
)

vcpkg_cmake_install(DISABLE_PARALLEL)

file(REMOVE_RECURSE ${CURRENT_PACKAGES_DIR}/debug/include)

file(INSTALL ${SOURCE_PATH}/LICENSE DESTINATION ${CURRENT_PACKAGES_DIR}/share/${PORT} RENAME copyright)
file(REMOVE "${CURRENT_PACKAGES_DIR}/LICENSE")
file(REMOVE "${CURRENT_PACKAGES_DIR}/debug/LICENSE")

vcpkg_copy_pdbs()
vcpkg_cmake_config_fixup(CONFIG_PATH lib/cmake/${PORT})
