cmake_minimum_required(VERSION 3.18)

project(wignersun CXX)

add_library(wignersun ClebschGordan.cpp)
add_library(WignerSUN::wignersun ALIAS wignersun)

include(GNUInstallDirs)

target_include_directories(wignersun PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)

set(WIGNERSUN_INSTALL_CONFIGDIR "${CMAKE_INSTALL_DATAROOTDIR}/wignersun")

install(TARGETS wignersun
  EXPORT wignersun-targets
  INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  )

install(FILES ClebschGordan.hpp DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

# Write build-tree targets
export(TARGETS wignersun
    FILE ${CMAKE_CURRENT_BINARY_DIR}/wignersun-targets.cmake
    NAMESPACE WignerSUN::)
# Allow find_package to locate package without installing it (find it's build-tree)
export(PACKAGE wignersun)

install(EXPORT wignersun-targets
    FILE wignersun-targets.cmake
    NAMESPACE WignerSUN::
    DESTINATION ${WIGNERSUN_INSTALL_CONFIGDIR})
  
include(CMakePackageConfigHelpers)

# Generate build-tree configuration
set(INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
configure_package_config_file("${CMAKE_CURRENT_LIST_DIR}/config.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/wignersun-config.cmake"
    INSTALL_DESTINATION ${CMAKE_CURRENT_BINARY_DIR}
    PATH_VARS INCLUDE_DIR
    INSTALL_PREFIX ${CMAKE_CURRENT_BINARY_DIR})

# Generate install-tree configuration
set(INCLUDE_DIR ${CMAKE_INSTALL_INCLUDEDIR})
configure_package_config_file("${CMAKE_CURRENT_LIST_DIR}/config.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/wignersun-config.install.cmake"
    INSTALL_DESTINATION ${WIGNERSUN_INSTALL_CONFIGDIR}
    PATH_VARS INCLUDE_DIR)

# Generate package version file
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/wignersun-config-version.cmake"
    VERSION 1.0.0
    COMPATIBILITY SameMajorVersion)

# Install the install-tree configuration
install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/wignersun-config.install.cmake"
    DESTINATION ${WIGNERSUN_INSTALL_CONFIGDIR}
    RENAME "wignersun-config.cmake")

# Install package version file
install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/wignersun-config-version.cmake"
    DESTINATION ${WIGNERSUN_INSTALL_CONFIGDIR})
