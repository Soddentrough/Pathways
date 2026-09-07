cmake_minimum_required(VERSION 3.25)
project(Pathways LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Output directories
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Compiler warnings and options
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(
        -Wall
        -Wextra
        -Wpedantic
        -Wno-unused-parameter
        -Wno-missing-field-initializers
        -Wno-unused-variable
        -Wno-array-bounds
        -Wno-stringop-overflow
    )
endif()

# Find system packages (Fedora 44)
find_package(Vulkan REQUIRED)
find_package(SDL3 REQUIRED)
find_package(Threads REQUIRED)
find_package(ZLIB REQUIRED)

# Check Vulkan version (baseline 1.4.341)
message(STATUS "Vulkan Include Dir: ${Vulkan_INCLUDE_DIRS}")
message(STATUS "Vulkan Libraries: ${Vulkan_LIBRARIES}")
message(STATUS "SDL3 Include Dir: ${SDL3_INCLUDE_DIRS}")

# Third-party include directories
set(THIRD_PARTY_DIR ${CMAKE_CURRENT_SOURCE_DIR}/third_party)
include_directories(
    ${THIRD_PARTY_DIR}
    ${THIRD_PARTY_DIR}/glm
    ${THIRD_PARTY_DIR}/vma
    ${THIRD_PARTY_DIR}/cgltf
    ${THIRD_PARTY_DIR}/stb
    ${THIRD_PARTY_DIR}/tinyexr
    ${THIRD_PARTY_DIR}/imgui
    ${THIRD_PARTY_DIR}/imgui/backends
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
file(MAKE_DIRECTORY ${SPV_OUTPUT_DIR})

set(SHADER_FILES
    ${SHADER_DIR}/compute/tonemap_aces.comp
    ${SHADER_DIR}/compute/raytrace_comp.comp
    ${SHADER_DIR}/compute/dgc_compact.comp
    ${SHADER_DIR}/compute/accum_merge.comp
)

set(SPV_FILES "")
foreach(SHADER ${SHADER_FILES})
    if(EXISTS ${SHADER})
        get_filename_component(SHADER_NAME ${SHADER} NAME)
        set(SPV_FILE ${SPV_OUTPUT_DIR}/${SHADER_NAME}.spv)
        add_custom_command(
            OUTPUT ${SPV_FILE}
            COMMAND ${GLSLC} --target-env=vulkan1.4 -O ${SHADER} -o ${SPV_FILE}
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
    m
)

target_include_directories(pathways PRIVATE
    ${Vulkan_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${SPV_OUTPUT_DIR}
)

# Definitions
target_compile_definitions(pathways PRIVATE
    SHADER_DIR="${SPV_OUTPUT_DIR}"
    VK_ENABLE_BETA_EXTENSIONS=0
)

# Camera & FPS Controls Unit Test Target
add_executable(test_camera_controls
    tests/test_camera_controls.cpp
    src/scene/Camera.cpp
)
target_include_directories(test_camera_controls PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${THIRD_PARTY_DIR}/glm
)
target_link_libraries(test_camera_controls PRIVATE m)
