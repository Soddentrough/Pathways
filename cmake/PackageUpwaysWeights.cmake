# PackageUpwaysWeights.cmake
# Standalone CMake script executed at build/configure time to package and embed Upways weights.

if(NOT DEFINED CMAKE_CURRENT_SOURCE_DIR)
    get_filename_component(CMAKE_CURRENT_SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}/.." ABSOLUTE)
endif()

if(NOT DEFINED UPWAYS_DIR)
    if(DEFINED ENV{UPWAYS_DIR})
        set(UPWAYS_DIR "$ENV{UPWAYS_DIR}")
    else()
        get_filename_component(UPWAYS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../Upways" ABSOLUTE)
    endif()
endif()

set(UPWAYS_PRODUCTION_BIN "${UPWAYS_DIR}/checkpoints/upways_kpn_production/vulkan_export/upways_kpn_weights.bin")
set(UPWAYS_PRODUCTION_HPP "${UPWAYS_DIR}/checkpoints/upways_kpn_production/vulkan_export/upways_kpn_weights.hpp")

set(PATHWAYS_MODEL_DIR "${CMAKE_CURRENT_SOURCE_DIR}/data/models")
set(PATHWAYS_MODEL_BIN "${PATHWAYS_MODEL_DIR}/upways_weights.bin")
set(PATHWAYS_WEIGHTS_HPP "${CMAKE_CURRENT_SOURCE_DIR}/src/rt/upways_weights.hpp")
set(PATHWAYS_DEFAULT_HPP "${CMAKE_CURRENT_SOURCE_DIR}/src/rt/upways_default_weights.hpp")

# Function to generate upways_default_weights.hpp directly from upways_weights.bin
function(generate_embedded_weights bin_file out_header)
    if(NOT EXISTS "${bin_file}")
        return()
    endif()

    file(READ "${bin_file}" HEX_DATA HEX)
    string(LENGTH "${HEX_DATA}" HEX_LEN)
    math(EXPR BYTE_COUNT "${HEX_LEN} / 2")

    string(REGEX REPLACE "([0-9a-f][0-9a-f])" "0x\\1, " FORMATTED_BYTES "${HEX_DATA}")
    string(REGEX REPLACE "(0x[0-9a-f][0-9a-f], ){16}" "\\0\n    " FORMATTED_LINES "${FORMATTED_BYTES}")

    file(WRITE "${out_header}"
"#pragma once

// Auto-generated embedded default neural reconstructor weights
// Generated from data/models/upways_weights.bin by CMake build process
#include <cstdint>
#include <cstddef>

namespace upways {

constexpr size_t EMBEDDED_WEIGHTS_SIZE = ${BYTE_COUNT};

inline constexpr uint8_t DEFAULT_UPWAYS_WEIGHTS[] = {
    ${FORMATTED_LINES}
};

} // namespace upways
")
    message(STATUS "UpwaysPackaging: Embedded ${BYTE_COUNT} bytes into ${out_header}")
endfunction()

# 1. Check if upstream Upways has newer production weights to ingest
if(EXISTS "${UPWAYS_PRODUCTION_BIN}")
    set(NEEDS_SYNC FALSE)
    if(NOT EXISTS "${PATHWAYS_MODEL_BIN}")
        set(NEEDS_SYNC TRUE)
    elseif("${UPWAYS_PRODUCTION_BIN}" IS_NEWER_THAN "${PATHWAYS_MODEL_BIN}")
        set(NEEDS_SYNC TRUE)
    endif()

    if(NEEDS_SYNC)
        message(STATUS "UpwaysPackaging: Ingesting newer Upways weights from ${UPWAYS_PRODUCTION_BIN}")
        file(MAKE_DIRECTORY "${PATHWAYS_MODEL_DIR}")
        file(COPY_FILE "${UPWAYS_PRODUCTION_BIN}" "${PATHWAYS_MODEL_BIN}")
        if(EXISTS "${UPWAYS_PRODUCTION_HPP}")
            file(COPY_FILE "${UPWAYS_PRODUCTION_HPP}" "${PATHWAYS_WEIGHTS_HPP}")
        endif()
        generate_embedded_weights("${PATHWAYS_MODEL_BIN}" "${PATHWAYS_DEFAULT_HPP}")
    endif()
endif()

# 2. Always ensure upways_default_weights.hpp is synchronized with data/models/upways_weights.bin
if(EXISTS "${PATHWAYS_MODEL_BIN}")
    set(NEEDS_EMBED FALSE)
    if(NOT EXISTS "${PATHWAYS_DEFAULT_HPP}")
        set(NEEDS_EMBED TRUE)
    elseif("${PATHWAYS_MODEL_BIN}" IS_NEWER_THAN "${PATHWAYS_DEFAULT_HPP}")
        set(NEEDS_EMBED TRUE)
    endif()

    if(NEEDS_EMBED)
        message(STATUS "UpwaysPackaging: Synchronizing embedded C++ header from ${PATHWAYS_MODEL_BIN}")
        generate_embedded_weights("${PATHWAYS_MODEL_BIN}" "${PATHWAYS_DEFAULT_HPP}")
    endif()
endif()
