cmake_minimum_required(VERSION 3.28)
project(Pathways VERSION 1.11.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Default to Release build if none specified
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose build type: Debug Release RelWithDebInfo MinSizeRel" FORCE)
endif()

# Output directories
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Compiler options and warnings
option(ENABLE_NATIVE_OPTIMIZATION "Enable CPU native architecture instruction optimizations (-march=native)" ON)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(
        -Wall
        -Wextra
        -Wpedantic
        -Wno-unused-parameter
        -Wno-missing-field-initializers
        -Wno-unused-variable
        -Wno-array-bounds
    )

    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
        add_compile_options(-Wno-stringop-overflow)
    endif()

    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        add_compile_options(
            -Wno-nullability-extension
            -Wno-unused-private-field
        )
    endif()

    if(ENABLE_NATIVE_OPTIMIZATION)
        add_compile_options(-march=native -mtune=native)
    endif()

    # Release optimizations
    if(CMAKE_BUILD_TYPE STREQUAL "Release" OR CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
        add_compile_options(-O3)
    endif()
endif()

if(MSVC)
    add_compile_options(
        /W3
        /permissive-
        /utf-8
        /MP
        /EHsc
    )
    if(ENABLE_NATIVE_OPTIMIZATION)
        add_compile_options(/arch:AVX2)
    endif()
    if(CMAKE_BUILD_TYPE STREQUAL "Release" OR CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
        add_compile_options(/O2 /Oi /Ot)
    endif()
endif()

# Windows definitions and toolchain options
if(WIN32)
    add_compile_definitions(NOMINMAX WIN32_LEAN_AND_MEAN)
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        add_link_options(-static-libgcc -static-libstdc++)
    endif()
    set(CMAKE_MAP_IMPORTED_CONFIG_MINSIZEREL Release)
    set(CMAKE_MAP_IMPORTED_CONFIG_RELWITHDEBINFO Release)
    set(CMAKE_MAP_IMPORTED_CONFIG_DEBUG Release)
endif()

# Find system packages
find_package(Vulkan REQUIRED)
find_package(SDL3 REQUIRED)
find_package(Threads REQUIRED)
find_package(ZLIB REQUIRED)

message(STATUS "Vulkan Include Dir: ${Vulkan_INCLUDE_DIRS}")
message(STATUS "Vulkan Libraries: ${Vulkan_LIBRARIES}")
message(STATUS "SDL3 Include Dir: ${SDL3_INCLUDE_DIRS}")

# Third-party include directories (marked as SYSTEM to avoid external warnings)
set(THIRD_PARTY_DIR ${CMAKE_CURRENT_SOURCE_DIR}/third_party)
include_directories(SYSTEM
    ${THIRD_PARTY_DIR}
    ${THIRD_PARTY_DIR}/glm
    ${THIRD_PARTY_DIR}/vma
    ${THIRD_PARTY_DIR}/cgltf
    ${THIRD_PARTY_DIR}/stb
    ${THIRD_PARTY_DIR}/tinyexr
    ${THIRD_PARTY_DIR}/imgui
    ${THIRD_PARTY_DIR}/imgui/backends
)
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

# ImGui sources
set(IMGUI_SOURCES
    ${THIRD_PARTY_DIR}/imgui/imgui.cpp
    ${THIRD_PARTY_DIR}/imgui/imgui_draw.cpp
    ${THIRD_PARTY_DIR}/imgui/imgui_tables.cpp
    ${THIRD_PARTY_DIR}/imgui/imgui_widgets.cpp
    ${THIRD_PARTY_DIR}/imgui/backends/imgui_impl_sdl3.cpp
    ${THIRD_PARTY_DIR}/imgui/backends/imgui_impl_vulkan.cpp
)

# Shader compilation
find_program(GLSLC glslc REQUIRED)
message(STATUS "Found glslc: ${GLSLC}")

set(SHADER_DIR ${CMAKE_CURRENT_SOURCE_DIR}/shaders)
set(SPV_OUTPUT_DIR ${CMAKE_BINARY_DIR}/shaders)
set(BIN_SPV_DIR ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/shaders)
file(MAKE_DIRECTORY ${SPV_OUTPUT_DIR})
file(MAKE_DIRECTORY ${BIN_SPV_DIR})

set(SHADER_FILES
    ${SHADER_DIR}/compute/tonemap_aces.comp
    ${SHADER_DIR}/compute/accum_merge.comp
    ${SHADER_DIR}/compute/ffx_shadow_tileclassify.comp
    ${SHADER_DIR}/compute/ffx_shadow_filter.comp
    ${SHADER_DIR}/compute/taa_resolve.comp
    ${SHADER_DIR}/compute/atrous_denoise.comp
    ${SHADER_DIR}/compute/wavefront_classify.comp
    ${SHADER_DIR}/compute/wavefront_intersect.comp
    ${SHADER_DIR}/compute/wavefront_shade.comp
    ${SHADER_DIR}/compute/wavefront_shade_diffuse.comp
    ${SHADER_DIR}/compute/wavefront_shade_dielectric.comp
    ${SHADER_DIR}/compute/wavefront_shade_conductor.comp
    ${SHADER_DIR}/compute/wavefront_shade_complex.comp
    ${SHADER_DIR}/compute/wavefront_shadow.comp
    ${SHADER_DIR}/compute/wavefront_resolve.comp
    ${SHADER_DIR}/rt/raytrace.rgen
    ${SHADER_DIR}/rt/raytrace.rmiss
    ${SHADER_DIR}/rt/shadow.rmiss
    ${SHADER_DIR}/rt/raytrace.rchit
)

set(SPV_FILES "")
foreach(SHADER ${SHADER_FILES})
    if(EXISTS ${SHADER})
        get_filename_component(SHADER_NAME ${SHADER} NAME)
        set(SPV_FILE ${SPV_OUTPUT_DIR}/${SHADER_NAME}.spv)
        set(BIN_SPV_FILE ${BIN_SPV_DIR}/${SHADER_NAME}.spv)
        add_custom_command(
            OUTPUT ${SPV_FILE}
            COMMAND ${GLSLC} -I ${SHADER_DIR}/compute --target-env=vulkan1.4 -O ${SHADER} -o ${SPV_FILE}
            COMMAND ${CMAKE_COMMAND} -E copy_if_different ${SPV_FILE} ${BIN_SPV_FILE}
            DEPENDS ${SHADER}
            COMMENT "Compiling shader ${SHADER_NAME} to SPIR-V"
        )
        list(APPEND SPV_FILES ${SPV_FILE})
    endif()
endforeach()

add_custom_target(compile_shaders ALL DEPENDS ${SPV_FILES})

# Engine sources
file(GLOB_RECURSE SRC_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp
)

# Executable target
add_executable(pathways
    ${SRC_FILES}
    ${IMGUI_SOURCES}
)

add_dependencies(pathways compile_shaders)

target_link_libraries(pathways
    PRIVATE
    Vulkan::Vulkan
    SDL3::SDL3
    Threads::Threads
    ZLIB::ZLIB
)

if(UNIX)
    target_link_libraries(pathways PRIVATE m)
endif()

target_include_directories(pathways PRIVATE
    ${Vulkan_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${SPV_OUTPUT_DIR}
    ${BIN_SPV_DIR}
)

# Definitions
target_compile_definitions(pathways PRIVATE
    SHADER_DIR="${SPV_OUTPUT_DIR}"
    VK_ENABLE_BETA_EXTENSIONS=0
)

# Windows runtime DLL deployment
if(WIN32)
    # Collect extra runtime DLLs (e.g. zlib1.dll, libwinpthread-1.dll) from toolchain directory
    get_filename_component(COMPILER_DIR "${CMAKE_CXX_COMPILER}" DIRECTORY)
    set(EXTRA_RUNTIME_DLLS "")
    foreach(DLL_NAME zlib1.dll libwinpthread-1.dll)
        if(EXISTS "${COMPILER_DIR}/${DLL_NAME}")
            list(APPEND EXTRA_RUNTIME_DLLS "${COMPILER_DIR}/${DLL_NAME}")
        endif()
    endforeach()

    # Guard deployment: copy runtime DLLs if any exist, avoid empty argument errors
    add_custom_command(TARGET pathways POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E
            $<IF:$<BOOL:$<TARGET_RUNTIME_DLLS:pathways>>,copy_if_different,echo>
            $<TARGET_RUNTIME_DLLS:pathways>
            $<IF:$<BOOL:$<TARGET_RUNTIME_DLLS:pathways>>,$<TARGET_FILE_DIR:pathways>,"No target runtime DLLs to deploy">
        COMMAND_EXPAND_LISTS
        COMMENT "Deploying target runtime DLL dependencies"
    )

    if(EXTRA_RUNTIME_DLLS)
        add_custom_command(TARGET pathways POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                ${EXTRA_RUNTIME_DLLS}
                $<TARGET_FILE_DIR:pathways>
            COMMENT "Deploying extra toolchain runtime DLLs"
        )
    endif()
endif()

# Testing
enable_testing()

# Camera & FPS Controls Unit Test Target
add_executable(test_camera_controls
    tests/test_camera_controls.cpp
    src/scene/Camera.cpp
    src/scene/ProceduralScene.cpp
    src/core/Config.cpp
    src/core/Logger.cpp
)
target_include_directories(test_camera_controls PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${THIRD_PARTY_DIR}/glm
)

if(UNIX)
    target_link_libraries(test_camera_controls PRIVATE m)
endif()

add_test(NAME CameraControls COMMAND test_camera_controls)


# GUI Mouse Wheel & Scroll Indicator Unit Test Target
add_executable(test_gui_scroll
    tests/test_gui_scroll.cpp
    ${THIRD_PARTY_DIR}/imgui/imgui.cpp
    ${THIRD_PARTY_DIR}/imgui/imgui_draw.cpp
    ${THIRD_PARTY_DIR}/imgui/imgui_tables.cpp
    ${THIRD_PARTY_DIR}/imgui/imgui_widgets.cpp
)
target_include_directories(test_gui_scroll PRIVATE
    ${THIRD_PARTY_DIR}/imgui
)
if(UNIX)
    target_link_libraries(test_gui_scroll PRIVATE m)
endif()
add_test(NAME GuiScroll COMMAND test_gui_scroll)


# ReSTIR DI Quality & Frame Pacing Unit Test Target
add_executable(test_restir_quality
    tests/test_restir_quality.cpp
    src/scene/ProceduralScene.cpp
    src/core/QualityGovernor.cpp
)
target_include_directories(test_restir_quality PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${THIRD_PARTY_DIR}/glm
)
if(UNIX)
    target_link_libraries(test_restir_quality PRIVATE m Threads::Threads)
endif()
add_test(NAME RestirQuality COMMAND test_restir_quality)

# FidelityFX Shadow Denoiser Unit Test Target
add_executable(test_shadow_denoiser
    tests/test_shadow_denoiser.cpp
    src/core/Config.cpp
    src/core/Logger.cpp
)
target_include_directories(test_shadow_denoiser PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${THIRD_PARTY_DIR}/glm
)
if(UNIX)
    target_link_libraries(test_shadow_denoiser PRIVATE m)
endif()
add_test(NAME ShadowDenoiser COMMAND test_shadow_denoiser)

# Temporal Anti-Aliasing (TAA) Unit Test Target
add_executable(test_taa
    tests/test_taa.cpp
    src/core/Config.cpp
    src/core/Logger.cpp
    src/scene/Camera.cpp
)
target_include_directories(test_taa PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${THIRD_PARTY_DIR}/glm
)
if(UNIX)
    target_link_libraries(test_taa PRIVATE m)
endif()
add_test(NAME TemporalAntiAliasing COMMAND test_taa)

if(UNIX)
    # P2P Direct BAR DMA-BUF Unit Test Target
    add_executable(test_p2p_direct_bar
        tests/test_p2p_direct_bar.cpp
    )
    target_link_libraries(test_p2p_direct_bar PRIVATE Vulkan::Vulkan)
    add_test(NAME P2PDirectBar COMMAND test_p2p_direct_bar)
endif()

# Pipeline Comparison & Telemetry Unit Test Target
add_executable(test_pipeline_comparison
    tests/test_pipeline_comparison.cpp
    src/core/Config.cpp
    src/core/Logger.cpp
    src/utils/ImageDumper.cpp
)
target_include_directories(test_pipeline_comparison PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${THIRD_PARTY_DIR}/glm
)
if(UNIX)
    target_link_libraries(test_pipeline_comparison PRIVATE m Threads::Threads ZLIB::ZLIB)
endif()
add_test(NAME PipelineComparison COMMAND test_pipeline_comparison)



# ==============================================================================
# Installation & CPack Packaging Configuration
# ==============================================================================
include(GNUInstallDirs)

install(TARGETS pathways DESTINATION ${CMAKE_INSTALL_BINDIR})
install(DIRECTORY ${SPV_OUTPUT_DIR}/ DESTINATION ${CMAKE_INSTALL_DATADIR}/pathways/shaders FILES_MATCHING PATTERN "*.spv")
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/scenes")
    install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/scenes/ DESTINATION ${CMAKE_INSTALL_DATADIR}/pathways/scenes)
endif()
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/data/pathways.desktop")
    install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/data/pathways.desktop DESTINATION ${CMAKE_INSTALL_DATADIR}/applications)
endif()
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/README.md DESTINATION ${CMAKE_INSTALL_DOCDIR} OPTIONAL)

set(CPACK_PACKAGE_NAME "pathways")
set(CPACK_PACKAGE_VENDOR "Soddentrough")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "Pure Vulkan 1.4 Real-Time Path Tracer with Multi-GPU & Hardware RT")
set(CPACK_PACKAGE_VERSION_MAJOR "${PROJECT_VERSION_MAJOR}")
set(CPACK_PACKAGE_VERSION_MINOR "${PROJECT_VERSION_MINOR}")
set(CPACK_PACKAGE_VERSION_PATCH "${PROJECT_VERSION_PATCH}")
set(CPACK_PACKAGE_CONTACT "https://github.com/Soddentrough/Pathways")
set(CPACK_RESOURCE_FILE_README "${CMAKE_CURRENT_SOURCE_DIR}/README.md")

# RPM Generator options (Fedora / RHEL)
set(CPACK_RPM_PACKAGE_LICENSE "MIT")
set(CPACK_RPM_PACKAGE_GROUP "Amusements/Graphics")
set(CPACK_RPM_PACKAGE_URL "https://github.com/Soddentrough/Pathways")
set(CPACK_RPM_PACKAGE_AUTOREQ "yes")
set(CPACK_RPM_PACKAGE_AUTOPROV "yes")
set(CPACK_RPM_PACKAGE_ARCHITECTURE "x86_64")
if(NOT CPACK_RPM_PACKAGE_RELEASE)
    set(CPACK_RPM_PACKAGE_RELEASE "1.fc44")
endif()
set(CPACK_RPM_FILE_NAME "RPM-DEFAULT")

# Default generators
set(CPACK_GENERATOR "RPM;TGZ")

include(CPack)
