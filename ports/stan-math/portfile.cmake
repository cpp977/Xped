vcpkg_from_github(
    OUT_SOURCE_PATH SOURCE_PATH
    REPO stan-dev/math
    REF v4.4.0
    SHA512 d841adfe67c59af24e4122fe088858b16fde2ee4fb2565008631af944ac84dd34b0766cffa5908a29d6b4ba991630557f74dc77643fbb5cffc2fa4a5834220ea
    HEAD_REF develop
)

file(INSTALL "${SOURCE_PATH}/stan" DESTINATION "${CURRENT_PACKAGES_DIR}/include")

file(INSTALL "${SOURCE_PATH}/LICENSE.md" DESTINATION "${CURRENT_PACKAGES_DIR}/share/${PORT}" RENAME copyright)

# set(TBB_ROOT ${SOURCE_PATH}/lib/tbb_2020.3)
# set(TBB_DIR ${TBB_ROOT}/cmake)

# include(${TBB_DIR}/TBBBuild.cmake)
# tbb_build(TBB_ROOT ${TBB_ROOT} MAKE_ARGS compiler=g++ CONFIG_DIR ${CURRENT_PACKAGES_DIR}/share/tbb)
