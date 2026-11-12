cmake_minimum_required(VERSION 3.28)
project(Pathways VERSION 1.24.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_compile_definitions(
    PATHWAYS_VERSION_STRING="${PROJECT_VERSION}"
    PATHWAYS_VERSION_MAJOR=${PROJECT_VERSION_MAJOR}
    PATHWAYS_VERSION_MINOR=${PROJECT_VERSION_MINOR}
    PATHWAYS_VERSION_PATCH=${PROJECT_VERSION_PATCH}
)

# Default to Release build if none specified
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose build type: Debug Release RelWithDebInfo MinSizeRel" FORCE)
endif()

# Output directories
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Compiler options and warnings
option(ENABLE_NATIVE_OPTIMIZATION "Enable CPU native architecture instruction optimizations (-march=native)" OFF)
option(PATHWAYS_PORTABLE_CPU "Build portable x86-64-v3 binaries for distribution and CI" OFF)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(
        -Wall
        -Wextra
        -Wpedantic
        -Wno-unused-parameter
        -Wno-unused-variable
        -Wno-missing-field-initializers # Required for Vulkan 1.4 C-struct aggregate zero-initialization
    )


    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        add_compile_options(
            -Wno-nullability-extension
            -Wno-unused-private-field
        )
    endif()

    if(ENABLE_NATIVE_OPTIMIZATION)
        add_compile_options(-march=native -mtune=native)
    elseif(PATHWAYS_PORTABLE_CPU AND CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
        add_compile_options(-march=x86-64-v3)
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
    if(ENABLE_NATIVE_OPTIMIZATION OR PATHWAYS_PORTABLE_CPU)
        add_compile_options(/arch:AVX2)
    endif()
    if(CMAKE_BUILD_TYPE STREQUAL "Release" OR CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
        add_compile_options(/O2 /Oi /Ot)
    endif()
endif()

# Windows definitions and toolchain options
if(WIN32)
    add_compile_definitions(NOMINMAX WIN32_LEAN_AND_MEAN _CRT_SECURE_NO_WARNINGS)
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        # Use dynamic runtime consistent with OpenUSD (libusd_ms.dll) and TBB runtime DLLs
    endif()
    set(CMAKE_MAP_IMPORTED_CONFIG_MINSIZEREL Release)
    set(CMAKE_MAP_IMPORTED_CONFIG_RELWITHDEBINFO Release)
    set(CMAKE_MAP_IMPORTED_CONFIG_DEBUG Release)
endif()

# Find system packages
find_package(Vulkan 1.4 REQUIRED)
if(Vulkan_VERSION VERSION_LESS "1.4.341")
    message(FATAL_ERROR "Pathways requires Vulkan SDK/Runtime >= 1.4.341. Found: ${Vulkan_VERSION}")
endif()
find_package(SDL3 REQUIRED)
find_package(Threads REQUIRED)
find_package(ZLIB REQUIRED)
find_package(PkgConfig QUIET)
if(PkgConfig_FOUND)
    pkg_check_modules(FFMPEG IMPORTED_TARGET libavformat libavcodec libswscale libavutil)
endif()

if(FFMPEG_FOUND)
    add_compile_definitions(PATHWAYS_HAS_FFMPEG)
endif()

message(STATUS "Vulkan Version: ${Vulkan_VERSION}")
message(STATUS "Vulkan Include Dir: ${Vulkan_INCLUDE_DIRS}")
message(STATUS "Vulkan Libraries: ${Vulkan_LIBRARIES}")
message(STATUS "SDL3 Include Dir: ${SDL3_INCLUDE_DIRS}")
message(STATUS "FFmpeg Found: ${FFMPEG_FOUND}")

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
    ${SHADER_DIR}/compute/accum_running_avg.comp
    ${SHADER_DIR}/compute/accum_tonemap_fused.comp
    ${SHADER_DIR}/compute/wavefront_classify.comp
    ${SHADER_DIR}/compute/wavefront_intersect.comp
    ${SHADER_DIR}/compute/wavefront_shade.comp
    ${SHADER_DIR}/compute/wavefront_shade_diffuse.comp
    ${SHADER_DIR}/compute/wavefront_shade_dielectric.comp
    ${SHADER_DIR}/compute/wavefront_shade_conductor.comp
    ${SHADER_DIR}/compute/wavefront_shade_complex.comp
    ${SHADER_DIR}/compute/wavefront_shade_emissive.comp
    ${SHADER_DIR}/compute/wavefront_shade_passthrough.comp
    ${SHADER_DIR}/compute/wavefront_shadow.comp
    ${SHADER_DIR}/compute/wavefront_tail_megakernel.comp
    ${SHADER_DIR}/compute/update_tlas_instances.comp
    ${SHADER_DIR}/compute/nrc_encode_infer.comp
    ${SHADER_DIR}/compute/nrc_train.comp
    ${SHADER_DIR}/compute/nrc_resolve.comp
    ${SHADER_DIR}/compute/wavefront_resolve.comp
    ${SHADER_DIR}/compute/upways_temporal.comp
    ${SHADER_DIR}/compute/upways_neural_guide.comp
    ${SHADER_DIR}/compute/upways_upscale_resolve.comp
    ${SHADER_DIR}/compute/ffx_sdk_prep.comp
    ${SHADER_DIR}/compute/fsr3_blend.comp
    ${SHADER_DIR}/compute/caustic_photon_trace.comp
    ${SHADER_DIR}/compute/caustic_splat.comp
    ${SHADER_DIR}/compute/caustic_filter.comp
    ${SHADER_DIR}/compute/restir_debug_view.comp
    ${SHADER_DIR}/compute/denoise_fused_temporal.comp
    ${SHADER_DIR}/compute/denoise_atrous_step1.comp
    ${SHADER_DIR}/compute/denoise_atrous_step2.comp
    ${SHADER_DIR}/compute/denoise_atrous_step3.comp
    ${SHADER_DIR}/compute/denoise_atrous_coarse_remodulate.comp
    ${SHADER_DIR}/rt/raytrace.rgen
    ${SHADER_DIR}/rt/raytrace.rmiss
    ${SHADER_DIR}/rt/shadow.rmiss
    ${SHADER_DIR}/rt/raytrace.rchit
    ${SHADER_DIR}/ui/imgui_hdr.frag
)

set(SPV_FILES "")
# Every shared include is a dependency of every shader (cheap, and robust
# against missing hand-maintained lists when headers are added).
file(GLOB SHADER_INCLUDE_DEPS CONFIGURE_DEPENDS ${SHADER_DIR}/compute/*.glsl)
foreach(SHADER ${SHADER_FILES})
    if(EXISTS ${SHADER})
        get_filename_component(SHADER_NAME ${SHADER} NAME)
        set(SPV_FILE ${SPV_OUTPUT_DIR}/${SHADER_NAME}.spv)
        set(BIN_SPV_FILE ${BIN_SPV_DIR}/${SHADER_NAME}.spv)
        add_custom_command(
            OUTPUT ${SPV_FILE}
            COMMAND ${GLSLC} -I ${SHADER_DIR}/compute --target-env=vulkan1.4 -O ${SHADER} -o ${SPV_FILE}
            COMMAND ${CMAKE_COMMAND} -E copy_if_different ${SPV_FILE} ${BIN_SPV_FILE}
            DEPENDS ${SHADER} ${SHADER_INCLUDE_DEPS}
            COMMENT "Compiling shader ${SHADER_NAME} to SPIR-V"
        )
        list(APPEND SPV_FILES ${SPV_FILE})
    endif()
endforeach()

# Streamlined Secondary Shaders (Specialized compile-time dead-code elimination)
set(SEC_DIFFUSE_SPV ${SPV_OUTPUT_DIR}/wavefront_shade_diffuse_sec.comp.spv)
set(BIN_SEC_DIFFUSE_SPV ${BIN_SPV_DIR}/wavefront_shade_diffuse_sec.comp.spv)
add_custom_command(
    OUTPUT ${SEC_DIFFUSE_SPV}
    COMMAND ${GLSLC} -I ${SHADER_DIR}/compute --target-env=vulkan1.4 -DIS_SECONDARY_BOUNCE=1 -O ${SHADER_DIR}/compute/wavefront_shade_diffuse.comp -o ${SEC_DIFFUSE_SPV}
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${SEC_DIFFUSE_SPV} ${BIN_SEC_DIFFUSE_SPV}
    DEPENDS ${SHADER_DIR}/compute/wavefront_shade_diffuse.comp ${SHADER_INCLUDE_DEPS}
    COMMENT "Compiling streamlined secondary shader wavefront_shade_diffuse_sec.comp to SPIR-V"
)
list(APPEND SPV_FILES ${SEC_DIFFUSE_SPV})

set(SEC_COMPLEX_SPV ${SPV_OUTPUT_DIR}/wavefront_shade_complex_sec.comp.spv)
set(BIN_SEC_COMPLEX_SPV ${BIN_SPV_DIR}/wavefront_shade_complex_sec.comp.spv)
add_custom_command(
    OUTPUT ${SEC_COMPLEX_SPV}
    COMMAND ${GLSLC} -I ${SHADER_DIR}/compute --target-env=vulkan1.4 -DIS_SECONDARY_BOUNCE=1 -O ${SHADER_DIR}/compute/wavefront_shade_complex.comp -o ${SEC_COMPLEX_SPV}
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${SEC_COMPLEX_SPV} ${BIN_SEC_COMPLEX_SPV}
    DEPENDS ${SHADER_DIR}/compute/wavefront_shade_complex.comp ${SHADER_INCLUDE_DEPS}
    COMMENT "Compiling streamlined secondary shader wavefront_shade_complex_sec.comp to SPIR-V"
)
list(APPEND SPV_FILES ${SEC_COMPLEX_SPV})

# Alpha-mask specialized secondary intersection shader (zero texture/UV overhead on opaque scenes)
set(INTERSECT_ALPHA_SPV ${SPV_OUTPUT_DIR}/wavefront_intersect_alpha.comp.spv)
set(BIN_INTERSECT_ALPHA_SPV ${BIN_SPV_DIR}/wavefront_intersect_alpha.comp.spv)
add_custom_command(
    OUTPUT ${INTERSECT_ALPHA_SPV}
    COMMAND ${GLSLC} -I ${SHADER_DIR}/compute --target-env=vulkan1.4 -DENABLE_ALPHA_MASK=1 -O ${SHADER_DIR}/compute/wavefront_intersect.comp -o ${INTERSECT_ALPHA_SPV}
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${INTERSECT_ALPHA_SPV} ${BIN_INTERSECT_ALPHA_SPV}
    DEPENDS ${SHADER_DIR}/compute/wavefront_intersect.comp ${SHADER_INCLUDE_DEPS}
    COMMENT "Compiling alpha-mask intersection shader wavefront_intersect_alpha.comp to SPIR-V"
)
list(APPEND SPV_FILES ${INTERSECT_ALPHA_SPV})

# ReSTIR PT specialized material shading variants (isolates reservoir VGPR pressure from non-ReSTIR paths)
function(add_specialized_wf_shader OUT_NAME SRC_FILE)
    set(EXTRA_DEFS ${ARGN})
    set(OUT_SPV ${SPV_OUTPUT_DIR}/${OUT_NAME}.spv)
    set(BIN_OUT_SPV ${BIN_SPV_DIR}/${OUT_NAME}.spv)
    add_custom_command(
        OUTPUT ${OUT_SPV}
        COMMAND ${GLSLC} -I ${SHADER_DIR}/compute --target-env=vulkan1.4 ${EXTRA_DEFS} -O ${SHADER_DIR}/compute/${SRC_FILE} -o ${OUT_SPV}
        COMMAND ${CMAKE_COMMAND} -E copy_if_different ${OUT_SPV} ${BIN_OUT_SPV}
        DEPENDS ${SHADER_DIR}/compute/${SRC_FILE} ${SHADER_INCLUDE_DEPS}
        COMMENT "Compiling specialized shader ${OUT_NAME} to SPIR-V"
    )
    set(SPV_FILES ${SPV_FILES} ${OUT_SPV} PARENT_SCOPE)
endfunction()

add_specialized_wf_shader(wavefront_shade_diffuse_restir.comp wavefront_shade_diffuse.comp -DENABLE_RESTIR_PT=1)
add_specialized_wf_shader(wavefront_shade_diffuse_sec_restir.comp wavefront_shade_diffuse.comp -DIS_SECONDARY_BOUNCE=1 -DENABLE_RESTIR_PT=1)
add_specialized_wf_shader(wavefront_shade_complex_restir.comp wavefront_shade_complex.comp -DENABLE_RESTIR_PT=1)
add_specialized_wf_shader(wavefront_shade_complex_sec_restir.comp wavefront_shade_complex.comp -DIS_SECONDARY_BOUNCE=1 -DENABLE_RESTIR_PT=1)
add_specialized_wf_shader(wavefront_shade_dielectric_restir.comp wavefront_shade_dielectric.comp -DENABLE_RESTIR_PT=1)
add_specialized_wf_shader(wavefront_shade_emissive_restir.comp wavefront_shade_emissive.comp -DENABLE_RESTIR_PT=1)

add_custom_target(compile_shaders ALL DEPENDS ${SPV_FILES})

# Engine sources
file(GLOB_RECURSE SRC_FILES CONFIGURE_DEPENDS
    ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp
)

# Windows Resource File (Application Icon & Version Info)
set(WINDOWS_RESOURCE_FILES "")
if(WIN32)
    set(WINDOWS_RESOURCE_FILES ${CMAKE_CURRENT_SOURCE_DIR}/data/pathways.rc)
    set_source_files_properties(${CMAKE_CURRENT_SOURCE_DIR}/data/pathways.rc
        PROPERTIES
        COMPILE_FLAGS "-I ${CMAKE_CURRENT_SOURCE_DIR}/data"
    )
endif()

# Engine core static library (shared across pathways executable and all CTests)
set(ENGINE_CORE_SOURCES ${SRC_FILES})
list(REMOVE_ITEM ENGINE_CORE_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp")

add_library(pathways_engine_core STATIC
    ${ENGINE_CORE_SOURCES}
    ${IMGUI_SOURCES}
)

add_dependencies(pathways_engine_core compile_shaders)

target_link_libraries(pathways_engine_core
    PUBLIC
    Vulkan::Vulkan
    SDL3::SDL3
    Threads::Threads
    ZLIB::ZLIB
)

if(FFMPEG_FOUND)
    target_link_libraries(pathways_engine_core PUBLIC PkgConfig::FFMPEG)
endif()

if(WIN32)
    target_link_libraries(pathways_engine_core PUBLIC dxgi)
endif()

if(UNIX)
    target_link_libraries(pathways_engine_core PUBLIC m ${CMAKE_DL_LIBS})
endif()

target_include_directories(pathways_engine_core PUBLIC
    ${Vulkan_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${SPV_OUTPUT_DIR}
    ${BIN_SPV_DIR}
)

target_compile_definitions(pathways_engine_core PUBLIC
    SHADER_DIR="${SPV_OUTPUT_DIR}"
)

# Executable target
add_executable(pathways
    src/main.cpp
    ${WINDOWS_RESOURCE_FILES}
)

target_link_libraries(pathways PRIVATE pathways_engine_core)

# AMD FidelityFX SDK 2.4.0+ Support
set(PATHWAYS_DEFAULT_FFX_SDK_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../FidelityFX-SDK")
if(NOT EXISTS "${PATHWAYS_DEFAULT_FFX_SDK_DIR}" AND DEFINED ENV{HOME} AND EXISTS "$ENV{HOME}/Development/FidelityFX-SDK")
    set(PATHWAYS_DEFAULT_FFX_SDK_DIR "$ENV{HOME}/Development/FidelityFX-SDK")
elseif(NOT EXISTS "${PATHWAYS_DEFAULT_FFX_SDK_DIR}" AND DEFINED ENV{USERPROFILE} AND EXISTS "$ENV{USERPROFILE}/Development/FidelityFX-SDK")
    file(TO_CMAKE_PATH "$ENV{USERPROFILE}/Development/FidelityFX-SDK" PATHWAYS_DEFAULT_FFX_SDK_DIR)
endif()
set(PATHWAYS_FFX_SDK_DIR "${PATHWAYS_DEFAULT_FFX_SDK_DIR}" CACHE PATH "Path to AMD FidelityFX SDK 2.4.0+ root directory")

if(NOT EXISTS "${PATHWAYS_FFX_SDK_DIR}/Kits/FidelityFX/api/include/ffx_api.h")
    set(PATHWAYS_FETCHED_FFX_SDK_DIR "${CMAKE_BINARY_DIR}/_deps/FidelityFX-SDK")
    if(NOT EXISTS "${PATHWAYS_FETCHED_FFX_SDK_DIR}/Kits/FidelityFX/api/include/ffx_api.h")
        find_package(Git QUIET)
        if(GIT_FOUND)
            message(STATUS "Cloning mandatory AMD FidelityFX SDK v2.4.0 into ${PATHWAYS_FETCHED_FFX_SDK_DIR}...")
            execute_process(
                COMMAND "${GIT_EXECUTABLE}" clone --branch v2.4.0 --depth 1
                        https://github.com/GPUOpen-LibrariesAndSDKs/FidelityFX-SDK.git
                        "${PATHWAYS_FETCHED_FFX_SDK_DIR}"
                RESULT_VARIABLE FFX_GIT_RESULT
            )
        endif()
    endif()
    if(EXISTS "${PATHWAYS_FETCHED_FFX_SDK_DIR}/Kits/FidelityFX/api/include/ffx_api.h")
        set(PATHWAYS_FFX_SDK_DIR "${PATHWAYS_FETCHED_FFX_SDK_DIR}" CACHE PATH "Path to AMD FidelityFX SDK 2.4.0+ root directory" FORCE)
    endif()
endif()

if(EXISTS "${PATHWAYS_FFX_SDK_DIR}/Kits/FidelityFX/api/include/ffx_api.h")
    message(STATUS "AMD FidelityFX SDK found at: ${PATHWAYS_FFX_SDK_DIR}")
    set(FFX_SDK_INCLUDE_DIRS
        "${PATHWAYS_FFX_SDK_DIR}/Kits/FidelityFX/api/include"
        "${PATHWAYS_FFX_SDK_DIR}/Kits/FidelityFX/upscalers/include"
        "${PATHWAYS_FFX_SDK_DIR}/Kits/FidelityFX/denoisers/include"
        "${PATHWAYS_FFX_SDK_DIR}/Kits/FidelityFX/radiancecache/include"
        "${PATHWAYS_FFX_SDK_DIR}/Kits/FidelityFX/framegeneration/include"
    )
    target_include_directories(pathways_engine_core SYSTEM PUBLIC ${FFX_SDK_INCLUDE_DIRS})
    target_compile_definitions(pathways_engine_core PUBLIC
        PATHWAYS_HAS_FFX_SDK=1
        PATHWAYS_FFX_SDK_ROOT_DIR="${PATHWAYS_FFX_SDK_DIR}"
    )
    if(UNIX AND NOT APPLE AND EXISTS "/usr/bin/python3")
        if(NOT DEFINED Python3_EXECUTABLE OR NOT Python3_EXECUTABLE MATCHES "^/usr/bin/")
            foreach(_py_var
                    Python3_EXECUTABLE Python3_INCLUDE_DIR Python3_LIBRARY Python3_LIBRARY_RELEASE Python3_LIBRARY_DEBUG
                    _Python3_EXECUTABLE _Python3_INCLUDE_DIR _Python3_LIBRARY_RELEASE _Python3_LIBRARY_DEBUG
                    _Python3_INTERPRETER_PROPERTIES _Python3_DEVELOPMENT_MODULE_SIGNATURE _Python3_DEVELOPMENT_EMBED_SIGNATURE)
                unset(${_py_var} CACHE)
                unset(${_py_var})
            endforeach()
            set(Python3_EXECUTABLE "/usr/bin/python3" CACHE FILEPATH "System Python3 executable" FORCE)
            set(Python3_FIND_VIRTUALENV STANDARD)
        endif()
    endif()
    if(UNIX AND EXISTS "${PATHWAYS_FFX_SDK_DIR}/Kits/FidelityFX/signedbin/amd_fidelityfx_upscaler_vk.dll")
        find_package(Python3 COMPONENTS Interpreter REQUIRED)
        set(FFX_UPSCALER_VK_SO "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/libamd_fidelityfx_upscaler_vk.so")
        add_custom_command(
            OUTPUT "${FFX_UPSCALER_VK_SO}"
            COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/scripts/build_ffx_upscaler_vk_linux.py"
                    "${PATHWAYS_FFX_SDK_DIR}"
                    "${CMAKE_BINARY_DIR}/ffx_sdk_generated"
                    "${FFX_UPSCALER_VK_SO}"
                    "${CMAKE_CXX_COMPILER}"
            DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/scripts/build_ffx_upscaler_vk_linux.py"
                    "${PATHWAYS_FFX_SDK_DIR}/Kits/FidelityFX/signedbin/amd_fidelityfx_upscaler_vk.dll"
                    "${PATHWAYS_FFX_SDK_DIR}/Kits/FidelityFX/backend/vk/ffx_vk.cpp"
                    "${PATHWAYS_FFX_SDK_DIR}/Kits/FidelityFX/upscalers/fsr3/internal/ffx_fsr3upscaler.cpp"
            COMMENT "Building native Linux FidelityFX Upscaler 3.1.6 Vulkan provider (libamd_fidelityfx_upscaler_vk.so)"
        )
        add_custom_target(ffx_upscaler_vk_linux ALL DEPENDS "${FFX_UPSCALER_VK_SO}")
        add_dependencies(pathways_engine_core ffx_upscaler_vk_linux)
    endif()
else()
    message(FATAL_ERROR "AMD FidelityFX SDK 2.4.0+ is mandatory for FSR 3.1 / FSR 4.1 ML support, but was not found at ${PATHWAYS_FFX_SDK_DIR}.")
endif()

# OpenUSD (Pixar USD) Support
set(USD_FALLBACK_DEPS "${CMAKE_CURRENT_SOURCE_DIR}/third_party/usd_deps")
if(UNIX AND (EXISTS "${USD_FALLBACK_DEPS}/include/tbb/concurrent_queue.h" OR EXISTS "/usr/include/tbb/concurrent_queue.h"))
    set(PXR_FIND_TBB_IN_CONFIG OFF)
    if(EXISTS "/usr/lib64/libXt.so.6" AND NOT DEFINED X11_Xt_LIB)
        set(X11_Xt_LIB "/usr/lib64/libXt.so.6" CACHE FILEPATH "Path to Xt library")
        set(X11_Xt_INCLUDE_PATH "${USD_FALLBACK_DEPS}/include" CACHE PATH "Path to Xt include")
        set(X11_Xt_FOUND TRUE CACHE BOOL "Xt found")
    endif()
endif()

find_package(Python3 COMPONENTS Interpreter Development QUIET)
set(PATHWAYS_HAS_PYTHON_MATERIALX FALSE)
if(Python3_FOUND)
    if((DEFINED Python3_SITELIB AND EXISTS "${Python3_SITELIB}/materialx") OR
       (DEFINED Python3_STDARCH AND EXISTS "${Python3_STDARCH}/materialx") OR
       EXISTS "/usr/lib/python${Python3_VERSION_MAJOR}.${Python3_VERSION_MINOR}/site-packages/materialx" OR
       EXISTS "/usr/lib64/python${Python3_VERSION_MAJOR}.${Python3_VERSION_MINOR}/site-packages/materialx")
        set(PATHWAYS_HAS_PYTHON_MATERIALX TRUE)
    endif()
endif()

if(WIN32 OR PATHWAYS_HAS_PYTHON_MATERIALX)
    find_package(OpenGL QUIET)
    find_package(X11 QUIET COMPONENTS Xt)
    find_package(pxr QUIET HINTS "$ENV{USD_ROOT}" "$ENV{PXR_ROOT}" "${CMAKE_CURRENT_SOURCE_DIR}/../USD" "C:/Users/naoki/Development/USD" "/usr/lib64/cmake/pxr")
    if(pxr_FOUND)
        message(STATUS "OpenUSD (pxr) found: ${PXR_VERSION}")
        if(TARGET hioAvif)
            set_target_properties(hioAvif PROPERTIES
                INTERFACE_INCLUDE_DIRECTORIES "/usr/include"
                INTERFACE_SYSTEM_INCLUDE_DIRECTORIES "/usr/include"
            )
        endif()
        if(TARGET TBB::tbb)
            get_target_property(_tbb_loc TBB::tbb IMPORTED_LOCATION)
            if(NOT _tbb_loc)
                if(EXISTS "/usr/lib64/libtbb.so.12")
                    set_target_properties(TBB::tbb PROPERTIES IMPORTED_LOCATION "/usr/lib64/libtbb.so.12")
                elseif(EXISTS "/usr/lib64/libtbb.so")
                    set_target_properties(TBB::tbb PROPERTIES IMPORTED_LOCATION "/usr/lib64/libtbb.so")
                endif()
            endif()
            if(UNIX AND EXISTS "${USD_FALLBACK_DEPS}/include")
                set_property(TARGET TBB::tbb APPEND PROPERTY INTERFACE_INCLUDE_DIRECTORIES "${USD_FALLBACK_DEPS}/include")
            endif()
        endif()
        if(TARGET usd_m)
            set_target_properties(usd_m PROPERTIES INTERFACE_LINK_LIBRARIES "")
            get_target_property(_usd_dirs usd_m INTERFACE_INCLUDE_DIRECTORIES)
            if(_usd_dirs)
                list(FILTER _usd_dirs EXCLUDE REGEX "rav1e|svt-av1|libvmaf")
                set_target_properties(usd_m PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${_usd_dirs}")
            endif()
            set(PATHWAYS_USD_LIBS usd_m)
        else()
            set(PATHWAYS_USD_LIBS arch tf gf plug vt ar sdf pcp usd usdGeom usdShade usdLux)
        endif()
        find_package(Python3 COMPONENTS Development QUIET)
        target_compile_definitions(pathways_engine_core PUBLIC
            PATHWAYS_ENABLE_USD=1
            TBB_SUPPRESS_DEPRECATED_MESSAGES=1
            TBB_INTERFACE_VERSION_MAJOR=12
        )
        target_link_libraries(pathways_engine_core PUBLIC ${PATHWAYS_USD_LIBS})
        if(TARGET TBB::tbb)
            target_link_libraries(pathways_engine_core PUBLIC TBB::tbb)
        elseif(EXISTS "/usr/lib64/libtbb.so.12")
            target_link_libraries(pathways_engine_core PUBLIC /usr/lib64/libtbb.so.12)
        elseif(EXISTS "/usr/lib64/libtbb.so")
            target_link_libraries(pathways_engine_core PUBLIC /usr/lib64/libtbb.so)
        endif()
        if(TARGET Python3::Python)
            target_link_libraries(pathways_engine_core PUBLIC Python3::Python)
            target_include_directories(pathways_engine_core SYSTEM PUBLIC ${Python3_INCLUDE_DIRS})
        endif()
        if(UNIX AND EXISTS "${USD_FALLBACK_DEPS}/include")
            target_include_directories(pathways_engine_core SYSTEM BEFORE PUBLIC "${USD_FALLBACK_DEPS}/include")
        endif()
        target_include_directories(pathways_engine_core SYSTEM PUBLIC ${PXR_INCLUDE_DIRS})
    endif()
else()
    message(STATUS "OpenUSD prerequisites not fully installed (requires usd-devel, imath-devel, python3-materialx, libXt-devel). Building without native OpenUSD support.")
endif()

# Windows runtime DLL deployment
if(WIN32)
    # Collect extra runtime DLLs (e.g. zlib1.dll, libwinpthread-1.dll) from toolchain directory
    get_filename_component(COMPILER_DIR "${CMAKE_CXX_COMPILER}" DIRECTORY)
    set(EXTRA_RUNTIME_DLLS "")
    foreach(DLL_NAME zlib1.dll libwinpthread-1.dll libtbb12.dll libstdc++-6.dll libgcc_s_seh-1.dll)
        if(EXISTS "${COMPILER_DIR}/${DLL_NAME}")
            list(APPEND EXTRA_RUNTIME_DLLS "${COMPILER_DIR}/${DLL_NAME}")
        endif()
    endforeach()
    if(pxr_FOUND)
        foreach(CANDIDATE_DLL "${PXR_CMAKE_DIR}/lib/libusd_ms.dll" "${PXR_CMAKE_DIR}/bin/libusd_ms.dll")
            if(EXISTS "${CANDIDATE_DLL}")
                list(APPEND EXTRA_RUNTIME_DLLS "${CANDIDATE_DLL}")
            endif()
        endforeach()
    endif()
    if(EXISTS "${PATHWAYS_FFX_SDK_DIR}/Kits/FidelityFX/signedbin")
        foreach(FFX_DLL_NAME
                amd_fidelityfx_loader_vk.dll
                amd_fidelityfx_upscaler_vk.dll
                amd_fidelityfx_denoiser_vk.dll
                amd_fidelityfx_radiancecache_vk.dll
                amd_fidelityfx_framegeneration_vk.dll)
            if(EXISTS "${PATHWAYS_FFX_SDK_DIR}/Kits/FidelityFX/signedbin/${FFX_DLL_NAME}")
                list(APPEND EXTRA_RUNTIME_DLLS "${PATHWAYS_FFX_SDK_DIR}/Kits/FidelityFX/signedbin/${FFX_DLL_NAME}")
            endif()
        endforeach()
    endif()

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

    add_custom_command(TARGET pathways POST_BUILD
        COMMAND ${CMAKE_COMMAND}
            -DTARGET_EXE="$<TARGET_FILE:pathways>"
            -DCOMPILER_DIR="${COMPILER_DIR}"
            -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/DeployRuntimeDlls.cmake"
        COMMENT "Deploying transitive toolchain DLL dependencies"
    )

    if(pxr_FOUND AND EXISTS "${PXR_CMAKE_DIR}/lib/usd")
        add_custom_command(TARGET pathways POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_directory
                "${PXR_CMAKE_DIR}/lib/usd"
                "$<TARGET_FILE_DIR:pathways>/usd"
            COMMENT "Deploying OpenUSD runtime plugin directory"
        )
    endif()
endif()

# Package and embed Upways neural reconstruction weights
set(UPWAYS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../Upways" CACHE PATH "Path to Upways ML repository")
option(PATHWAYS_AUTO_PACKAGE_UPWAYS "Auto-package updated weights from Upways during build" ON)

include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/PackageUpwaysWeights.cmake")

add_custom_target(package_upways_weights
    COMMAND ${CMAKE_COMMAND}
        -DPATHWAYS_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}"
        -DUPWAYS_DIR="${UPWAYS_DIR}"
        -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/PackageUpwaysWeights.cmake"
    COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_BINARY_DIR}/data/models"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${CMAKE_CURRENT_SOURCE_DIR}/data/models/upways_weights.bin"
        "${CMAKE_BINARY_DIR}/data/models/upways_weights.bin"
    COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_BINARY_DIR}/bin/data/models"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${CMAKE_CURRENT_SOURCE_DIR}/data/models/upways_weights.bin"
        "${CMAKE_BINARY_DIR}/bin/data/models/upways_weights.bin"
    BYPRODUCTS
        "${CMAKE_CURRENT_SOURCE_DIR}/data/models/upways_weights.bin"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/rt/upways_default_weights.hpp"
        "${CMAKE_BINARY_DIR}/data/models/upways_weights.bin"
        "${CMAKE_BINARY_DIR}/bin/data/models/upways_weights.bin"
    COMMENT "Checking and packaging updated Upways neural weights"
)

add_dependencies(pathways_engine_core package_upways_weights)

# Deploy default neural reconstruction weights to build tree
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/data/models/upways_weights.bin")
    file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/data/models")
    file(COPY "${CMAKE_CURRENT_SOURCE_DIR}/data/models/upways_weights.bin"
         DESTINATION "${CMAKE_BINARY_DIR}/data/models")
    if(EXISTS "${CMAKE_BINARY_DIR}/bin")
        file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/bin/data/models")
        file(COPY "${CMAKE_CURRENT_SOURCE_DIR}/data/models/upways_weights.bin"
             DESTINATION "${CMAKE_BINARY_DIR}/bin/data/models")
    endif()
    add_custom_command(TARGET pathways POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_BINARY_DIR}/data/models"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${CMAKE_CURRENT_SOURCE_DIR}/data/models/upways_weights.bin"
            "${CMAKE_BINARY_DIR}/data/models/upways_weights.bin"
        COMMAND ${CMAKE_COMMAND} -E make_directory "$<TARGET_FILE_DIR:pathways>/data/models"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${CMAKE_CURRENT_SOURCE_DIR}/data/models/upways_weights.bin"
            "$<TARGET_FILE_DIR:pathways>/data/models/upways_weights.bin"
        COMMENT "Deploying Upways neural weights to build tree"
    )
endif()

# Testing
option(BUILD_TESTING "Build unit test executables" ON)
if(BUILD_TESTING)
enable_testing()

# Camera & FPS Controls Unit Test Target
add_executable(test_camera_controls tests/test_camera_controls.cpp)
target_link_libraries(test_camera_controls PRIVATE pathways_engine_core)
add_test(NAME CameraControls COMMAND test_camera_controls)

# Camera Paths & Trajectories Unit Test Target
add_executable(test_camera_paths tests/test_camera_paths.cpp)
target_link_libraries(test_camera_paths PRIVATE pathways_engine_core)
add_test(NAME CameraPaths COMMAND test_camera_paths)

# GUI Mouse Wheel, Scroll Indicator & GuiManager Unit Test Target
add_executable(test_gui_scroll tests/test_gui_scroll.cpp)
target_link_libraries(test_gui_scroll PRIVATE pathways_engine_core)
add_test(NAME GuiScroll COMMAND test_gui_scroll)

# Temporal Anti-Aliasing (TAA) & SuperResolutionManager Unit Test Target
add_executable(test_taa tests/test_taa.cpp)
target_link_libraries(test_taa PRIVATE pathways_engine_core)
add_test(NAME TemporalAntiAliasing COMMAND test_taa)

if(UNIX)
    # P2P Direct BAR DMA-BUF Cross-GPU Copy Unit Test Target
    add_executable(test_p2p_direct_bar tests/test_p2p_direct_bar.cpp)
    target_link_libraries(test_p2p_direct_bar PRIVATE pathways_engine_core)
    add_test(NAME P2PDirectBar COMMAND test_p2p_direct_bar)

    # Cross-GPU Hardware Synchronization (VK_KHR_external_semaphore_fd + Timeline) Unit Test Target
    add_executable(test_cross_gpu_sync tests/test_cross_gpu_sync.cpp)
    target_link_libraries(test_cross_gpu_sync PRIVATE pathways_engine_core)
    add_test(NAME CrossGpuSync COMMAND test_cross_gpu_sync)
endif()

# Pipeline Comparison, CLI Validation & Telemetry Unit Test Target
add_executable(test_pipeline_comparison tests/test_pipeline_comparison.cpp)
target_link_libraries(test_pipeline_comparison PRIVATE pathways_engine_core)
add_test(NAME PipelineComparison COMMAND test_pipeline_comparison)

# Multi-GPU Frame Pacing, Coordinator & Shared Buffer Unit Test Target
add_executable(test_mgpu_frame_pacing tests/test_mgpu_frame_pacing.cpp)
target_link_libraries(test_mgpu_frame_pacing PRIVATE pathways_engine_core)
add_test(NAME MgpuFramePacing COMMAND test_mgpu_frame_pacing)

# Tier 3 GPU TLAS Refit & GpuTlasUpdatePipeline Unit Test Target
add_executable(test_tlas_gpu_refit tests/test_tlas_gpu_refit.cpp)
target_link_libraries(test_tlas_gpu_refit PRIVATE pathways_engine_core)
add_test(NAME TlasGpuRefit COMMAND test_tlas_gpu_refit)

# Tier 4 Neural Radiance Caching (NRC) Wave32 WMMA Unit Test Target
add_executable(test_nrc_wmma tests/test_nrc_wmma.cpp)
target_link_libraries(test_nrc_wmma PRIVATE pathways_engine_core)
add_test(NAME NrcWmma COMMAND test_nrc_wmma)

# Upways Neural Denoiser & Super-Resolution Wave32 WMMA Unit Test Target
add_executable(test_upways_pipeline tests/test_upways_pipeline.cpp)
target_link_libraries(test_upways_pipeline PRIVATE pathways_engine_core)
add_test(NAME UpwaysPipeline COMMAND test_upways_pipeline)

# Adaptive E-SVGF Denoiser Pipeline (5-Dispatch Dilated GPU Execution) Test Target
add_executable(test_denoiser_pipeline tests/test_denoiser_pipeline.cpp)
target_compile_definitions(test_denoiser_pipeline PRIVATE
    PATHWAYS_SPV_DIR="${SPV_OUTPUT_DIR}"
)
target_link_libraries(test_denoiser_pipeline PRIVATE pathways_engine_core)
add_test(NAME DenoiserPipeline COMMAND test_denoiser_pipeline)

# ReSTIR PT Mathematical Foundations & Estimator Unit Test Target
add_executable(test_restir_pt_math tests/test_restir_pt_math.cpp)
target_link_libraries(test_restir_pt_math PRIVATE pathways_engine_core)
add_test(NAME RestirPtMath COMMAND test_restir_pt_math)

# DGC Multi-Slice Ring Buffer & Dedicated Compute Queue Unit Test Target
add_executable(test_dgc_async_queue tests/test_dgc_async_queue.cpp)
target_link_libraries(test_dgc_async_queue PRIVATE pathways_engine_core)
add_test(NAME DgcAsyncQueue COMMAND test_dgc_async_queue)

# Batched BLAS Builds & Compaction Unit Test Target
add_executable(test_blas_batch tests/test_blas_batch.cpp)
target_link_libraries(test_blas_batch PRIVATE pathways_engine_core)
add_test(NAME BlasBatch COMMAND test_blas_batch)

# Hierarchical Light Tree & Emissive Geometry Pipeline Unit Test Target
add_executable(test_light_tree tests/test_light_tree.cpp)
target_link_libraries(test_light_tree PRIVATE pathways_engine_core)
add_test(NAME LightTree COMMAND test_light_tree)

# glTF KHR_mesh_quantization, KHR_materials_specular & MiniJson Unit Test Target
add_executable(test_mesh_quantization tests/test_mesh_quantization.cpp)
target_link_libraries(test_mesh_quantization PRIVATE pathways_engine_core)
add_test(NAME MeshQuantization COMMAND test_mesh_quantization)

# Procedural Cyber City Megastructure Unit Test Target
add_executable(test_procedural_cyber_city tests/test_procedural_cyber_city.cpp)
target_link_libraries(test_procedural_cyber_city PRIVATE pathways_engine_core)
add_test(NAME ProceduralCyberCity COMMAND test_procedural_cyber_city)

# Real-Time Caustics Dielectric Target & Bounding Volume Unit Test Target
add_executable(test_caustics_detection tests/test_caustics_detection.cpp)
target_link_libraries(test_caustics_detection PRIVATE pathways_engine_core)
add_test(NAME CausticsDetection COMMAND test_caustics_detection)

# AMD FidelityFX SDK 2.4 Feature, Performance & Visual Assessment Test Target
add_executable(test_fidelityfx_features tests/test_fidelityfx_features.cpp)
target_link_libraries(test_fidelityfx_features PRIVATE pathways_engine_core)
add_dependencies(test_fidelityfx_features pathways)
add_test(NAME FidelityFxFeatures COMMAND test_fidelityfx_features)

# OpenUSD Native Scene Ingestion Unit Test Target
if(pxr_FOUND)
    add_executable(test_usd_loader tests/test_usd_loader.cpp)
    target_link_libraries(test_usd_loader PRIVATE pathways_engine_core)
    add_test(NAME UsdLoader COMMAND test_usd_loader)
endif()
endif() # BUILD_TESTING




# ==============================================================================
# Installation & CPack Packaging Configuration
# ==============================================================================
include(GNUInstallDirs)

set(CURATED_SCENE_FILES
    scenes/DamagedHelmet.glb
    scenes/DragonAttenuation.glb
    scenes/DragonDispersion.glb
)

set(CURATED_SCENE_DIRS
    bmw-m6
    breakfast-room
    classroom
    coffee-maker
    cornell-caustic
    cyber_city
    glass-of-water
    living-room
    modern-hall
    PointInstancedMedCity
    Kitchen_set
    BuickRiviera
    veach-ajar
)

if(WIN32)
    # Windows runtime deployment and installation
    install(TARGETS pathways DESTINATION ${CMAKE_INSTALL_BINDIR})

    # Deploy required runtime DLLs
    install(PROGRAMS $<TARGET_RUNTIME_DLLS:pathways> DESTINATION ${CMAKE_INSTALL_BINDIR})
    if(EXTRA_RUNTIME_DLLS)
        install(PROGRAMS ${EXTRA_RUNTIME_DLLS} DESTINATION ${CMAKE_INSTALL_BINDIR})
    endif()

    if(pxr_FOUND AND EXISTS "${PXR_CMAKE_DIR}/lib/usd")
        install(DIRECTORY "${PXR_CMAKE_DIR}/lib/usd" DESTINATION ${CMAKE_INSTALL_BINDIR})
    endif()

    # Install SPIR-V shaders directly adjacent to executable
    install(DIRECTORY ${SPV_OUTPUT_DIR}/ DESTINATION ${CMAKE_INSTALL_BINDIR}/shaders FILES_MATCHING PATTERN "*.spv")

    # Install curated Tier 1 research scenes
    foreach(SCENE_FILE ${CURATED_SCENE_FILES})
        if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${SCENE_FILE}")
            install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/${SCENE_FILE}" DESTINATION scenes)
        endif()
    endforeach()

    foreach(SCENE_DIR ${CURATED_SCENE_DIRS})
        if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/scenes/${SCENE_DIR}")
            install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/scenes/${SCENE_DIR}/"
                    DESTINATION "scenes/${SCENE_DIR}"
                    FILES_MATCHING
                    PATTERN "*_extended.glb"
                    PATTERN "*.glb"
                    PATTERN "*.gltf"
                    PATTERN "*.bin"
                    PATTERN "*.usd"
                    PATTERN "*.usda"
                    PATTERN "*.usdc"
                    PATTERN "*.png"
                    PATTERN "*.jpg"
                    PATTERN "*.jpeg"
                    PATTERN "*.exr"
                    PATTERN "*.hdr"
                    PATTERN "*.mp4"
                    PATTERN "LICENSE.txt"
                    PATTERN "VERSION.txt"
                    PATTERN "conversion.yaml")
        endif()
    endforeach()
    file(GLOB ROOT_HDR_MP4_SCENES "${CMAKE_CURRENT_SOURCE_DIR}/scenes/*.hdr" "${CMAKE_CURRENT_SOURCE_DIR}/scenes/*.mp4")
    if(ROOT_HDR_MP4_SCENES)
        install(FILES ${ROOT_HDR_MP4_SCENES} DESTINATION scenes)
    endif()

    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/data/models/upways_weights.bin")
        install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/data/models/upways_weights.bin" DESTINATION data/models)
    endif()
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/data/pathways.ico")
        install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/data/pathways.ico" DESTINATION ${CMAKE_INSTALL_BINDIR})
    endif()
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE")
        install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/LICENSE DESTINATION .)
    endif()
    install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/README.md DESTINATION . OPTIONAL)
else()
    # Standard Linux FHS installation
    set_target_properties(pathways PROPERTIES
        INSTALL_RPATH "$ORIGIN/../${CMAKE_INSTALL_LIBDIR}/pathways"
    )
    install(TARGETS pathways DESTINATION ${CMAKE_INSTALL_BINDIR})
    install(DIRECTORY ${SPV_OUTPUT_DIR}/ DESTINATION ${CMAKE_INSTALL_DATADIR}/pathways/shaders FILES_MATCHING PATTERN "*.spv")
    if(DEFINED FFX_UPSCALER_VK_SO)
        install(PROGRAMS "${FFX_UPSCALER_VK_SO}" DESTINATION ${CMAKE_INSTALL_LIBDIR}/pathways)
    endif()

    foreach(SCENE_FILE ${CURATED_SCENE_FILES})
        if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${SCENE_FILE}")
            install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/${SCENE_FILE}" DESTINATION ${CMAKE_INSTALL_DATADIR}/pathways/scenes)
        endif()
    endforeach()

    foreach(SCENE_DIR ${CURATED_SCENE_DIRS})
        if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/scenes/${SCENE_DIR}")
            install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/scenes/${SCENE_DIR}/"
                    DESTINATION "${CMAKE_INSTALL_DATADIR}/pathways/scenes/${SCENE_DIR}"
                    FILES_MATCHING
                    PATTERN "*_extended.glb"
                    PATTERN "*.glb"
                    PATTERN "*.gltf"
                    PATTERN "*.bin"
                    PATTERN "*.usd"
                    PATTERN "*.usda"
                    PATTERN "*.usdc"
                    PATTERN "*.png"
                    PATTERN "*.jpg"
                    PATTERN "*.jpeg"
                    PATTERN "*.exr"
                    PATTERN "*.hdr"
                    PATTERN "*.mp4"
                    PATTERN "LICENSE.txt"
                    PATTERN "VERSION.txt"
                    PATTERN "conversion.yaml")
        endif()
    endforeach()
    file(GLOB ROOT_HDR_MP4_SCENES "${CMAKE_CURRENT_SOURCE_DIR}/scenes/*.hdr" "${CMAKE_CURRENT_SOURCE_DIR}/scenes/*.mp4")
    if(ROOT_HDR_MP4_SCENES)
        install(FILES ${ROOT_HDR_MP4_SCENES} DESTINATION ${CMAKE_INSTALL_DATADIR}/pathways/scenes)
    endif()

    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/data/pathways.desktop")
        install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/data/pathways.desktop DESTINATION ${CMAKE_INSTALL_DATADIR}/applications)
    endif()
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/data/pathways.svg")
        install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/data/pathways.svg DESTINATION ${CMAKE_INSTALL_DATADIR}/icons/hicolor/scalable/apps)
    endif()
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/data/pathways.png")
        install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/data/pathways.png DESTINATION ${CMAKE_INSTALL_DATADIR}/icons/hicolor/1024x1024/apps)
        install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/data/pathways.png DESTINATION ${CMAKE_INSTALL_DATADIR}/pixmaps)
    endif()
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/data/models/upways_weights.bin")
        install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/data/models/upways_weights.bin" DESTINATION ${CMAKE_INSTALL_DATADIR}/pathways/data/models)
    endif()
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE")
        install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/LICENSE DESTINATION ${CMAKE_INSTALL_DATADIR}/licenses/pathways)
    endif()
    install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/README.md DESTINATION ${CMAKE_INSTALL_DOCDIR} OPTIONAL)
endif()

set(CPACK_PACKAGE_NAME "pathways")
set(CPACK_PACKAGE_VENDOR "Soddentrough")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "Pure Vulkan 1.4 Real-Time Path Tracer with Multi-GPU & Hardware RT")
set(CPACK_PACKAGE_VERSION_MAJOR "${PROJECT_VERSION_MAJOR}")
set(CPACK_PACKAGE_VERSION_MINOR "${PROJECT_VERSION_MINOR}")
set(CPACK_PACKAGE_VERSION_PATCH "${PROJECT_VERSION_PATCH}")
set(CPACK_PACKAGE_CONTACT "https://github.com/Soddentrough/Pathways")
set(CPACK_RESOURCE_FILE_README "${CMAKE_CURRENT_SOURCE_DIR}/README.md")
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE")
    set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE")
endif()

if(WIN32)
    # Windows NSIS Installer & ZIP Distribution
    set(CPACK_GENERATOR "NSIS;ZIP")
    set(CPACK_NSIS_DISPLAY_NAME "Pathways")
    set(CPACK_NSIS_PACKAGE_NAME "Pathways")
    set(CPACK_PACKAGE_INSTALL_DIRECTORY "Pathways")
    set(CPACK_NSIS_MODIFY_PATH ON)

    # Shortcuts
    set(CPACK_PACKAGE_EXECUTABLES "pathways" "Pathways Path Tracer")
    set(CPACK_CREATE_DESKTOP_LINKS "pathways")

    # Icon branding
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/data/pathways.ico")
        set(CPACK_NSIS_MUI_ICON "${CMAKE_CURRENT_SOURCE_DIR}/data/pathways.ico")
        set(CPACK_NSIS_MUI_UNIICON "${CMAKE_CURRENT_SOURCE_DIR}/data/pathways.ico")
        set(CPACK_PACKAGE_ICON "${CMAKE_CURRENT_SOURCE_DIR}/data/pathways.ico")
    endif()

    set(CPACK_NSIS_MENU_LINKS
        "bin/pathways.exe" "Pathways Path Tracer"
        "https://github.com/Soddentrough/Pathways" "Pathways GitHub"
    )
    set(CPACK_NSIS_ENABLE_UNINSTALL_BEFORE_INSTALL ON)
else()
    # RPM Generator options (Fedora / RHEL)
    set(CPACK_RPM_PACKAGE_LICENSE "MIT")
    set(CPACK_RPM_PACKAGE_GROUP "Graphics")
    set(CPACK_RPM_PACKAGE_URL "https://github.com/Soddentrough/Pathways")
    set(CPACK_RPM_PACKAGE_AUTOREQ "yes")
    set(CPACK_RPM_PACKAGE_AUTOPROV "yes")
    set(CPACK_RPM_PACKAGE_ARCHITECTURE "x86_64")
    if(NOT CPACK_RPM_PACKAGE_RELEASE)
        set(CPACK_RPM_PACKAGE_RELEASE "1.fc45")
    endif()
    set(CPACK_RPM_FILE_NAME "RPM-DEFAULT")
    set(CPACK_RPM_PACKAGE_DESCRIPTION "Pathways is a high-performance, real-time Vulkan 1.4 path tracer optimized for modern AMD RDNA architectures (e.g. Dual Radeon AI PRO R9700). Features include pure Wave32 wavefront dispatch via Device Generated Commands (DGC), hardware ray tracing, unified Wavefront ReSTIR PT, Upways 3.0 FP8 WMMA neural reconstruction, multi-GPU scaling (checkerboard tiling and sample parallelism), and canonical Tier 1 PBR research assets.")

    # Default generators
    set(CPACK_GENERATOR "RPM;TGZ")
endif()

# Exclude large references and repository internals from release source packages
set(CPACK_SOURCE_IGNORE_FILES
    "/tests/references/"
    "/\\\\.git/"
    "/build/"
    "\\\\.DS_Store"
)

include(CPack)
