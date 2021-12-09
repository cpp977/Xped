if(XPED_ENABLE_CLANG_FORMAT)
# additional target to perform clang-format run, requires clang-format

# get all project files
set(PROJECT_TRDPARTY_DIR1 ${CMAKE_BINARY_DIR}/thirdparty)
set(PROJECT_TRDPARTY_DIR2 ${CMAKE_BINARY_DIR}/_deps)

file(GLOB_RECURSE ALL_SOURCE_FILES *.cpp *.hpp)
foreach (SOURCE_FILE ${ALL_SOURCE_FILES})
    string(FIND ${SOURCE_FILE} ${PROJECT_TRDPARTY_DIR1} PROJECT_TRDPARTY_DIR1_FOUND)
    if (NOT ${PROJECT_TRDPARTY_DIR1_FOUND} EQUAL -1)
        list(REMOVE_ITEM ALL_SOURCE_FILES ${SOURCE_FILE})
    endif ()
    string(FIND ${SOURCE_FILE} ${PROJECT_TRDPARTY_DIR2} PROJECT_TRDPARTY_DIR2_FOUND)
    if (NOT ${PROJECT_TRDPARTY_DIR2_FOUND} EQUAL -1)
        list(REMOVE_ITEM ALL_SOURCE_FILES ${SOURCE_FILE})
    endif ()

endforeach ()

add_custom_target(
        clangformat
        COMMAND /usr/bin/clang-format
        -style=file
        -i
        ${ALL_SOURCE_FILES}
)
endif()

