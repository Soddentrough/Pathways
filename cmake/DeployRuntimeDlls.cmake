# DeployRuntimeDlls.cmake
# Resolves and copies transitive runtime DLL dependencies from COMPILER_DIR to TARGET_DIR on Windows.

if(NOT DEFINED TARGET_EXE OR NOT EXISTS "${TARGET_EXE}")
    return()
endif()

get_filename_component(TARGET_DIR "${TARGET_EXE}" DIRECTORY)
if(EXISTS "${COMPILER_DIR}/objdump.exe")
    set(CMAKE_OBJDUMP "${COMPILER_DIR}/objdump.exe")
endif()

file(GET_RUNTIME_DEPENDENCIES
    EXECUTABLES "${TARGET_EXE}"
    RESOLVED_DEPENDENCIES_VAR RESOLVED_DEPS
    UNRESOLVED_DEPENDENCIES_VAR UNRESOLVED_DEPS
    CONFLICTING_DEPENDENCIES_PREFIX CONFLICT_DEPS
    DIRECTORIES "${COMPILER_DIR}" "${TARGET_DIR}"
    PRE_EXCLUDE_REGEXES "^api-ms-.*" "^ext-ms-.*" "^vulkan-1\\.dll$" "^dxgi\\.dll$"
    POST_EXCLUDE_REGEXES ".*[Ww]indows[\\\\/][Ss]ystem32.*" ".*[Ww]indows[\\\\/][Ss]ys[Ww][Oo][Ww]64.*"
)

foreach(DEP_DLL IN LISTS RESOLVED_DEPS)
    get_filename_component(DLL_NAME "${DEP_DLL}" NAME)
    set(DST_PATH "${TARGET_DIR}/${DLL_NAME}")
    file(COPY_FILE "${DEP_DLL}" "${DST_PATH}" ONLY_IF_DIFFERENT)
endforeach()
