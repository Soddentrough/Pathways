# PackageUpwaysWeights.cmake
# Standalone CMake script executed at build/configure time to package and embed Upways weights.

if(NOT DEFINED PATHWAYS_SOURCE_DIR)
    get_filename_component(PATHWAYS_SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}/.." ABSOLUTE)
endif()

if(NOT DEFINED UPWAYS_DIR)
    if(DEFINED ENV{UPWAYS_DIR})
        set(UPWAYS_DIR "$ENV{UPWAYS_DIR}")
    else()
        get_filename_component(UPWAYS_DIR "${PATHWAYS_SOURCE_DIR}/../Upways" ABSOLUTE)
    endif()
endif()

set(UPWAYS_PRODUCTION_BIN "${UPWAYS_DIR}/data/models/upways_weights.bin")

set(PATHWAYS_MODEL_DIR "${PATHWAYS_SOURCE_DIR}/data/models")
set(PATHWAYS_MODEL_BIN "${PATHWAYS_MODEL_DIR}/upways_weights.bin")
set(PATHWAYS_WEIGHTS_HPP "${PATHWAYS_SOURCE_DIR}/src/rt/upways_weights.hpp")
set(PATHWAYS_DEFAULT_HPP "${PATHWAYS_SOURCE_DIR}/src/rt/upways_default_weights.hpp")

# Function to generate upways_default_weights.hpp directly from upways_weights.bin
function(generate_embedded_weights bin_file out_header)
    if(NOT EXISTS "${bin_file}")
        return()
    endif()

    file(SHA256 "${bin_file}" BIN_SHA256)
    file(READ "${bin_file}" HEX_DATA HEX)
    string(LENGTH "${HEX_DATA}" HEX_LEN)
    math(EXPR BYTE_COUNT "${HEX_LEN} / 2")

    string(REGEX REPLACE "([0-9a-f][0-9a-f])" "0x\\1, " FORMATTED_BYTES "${HEX_DATA}")
    string(REGEX REPLACE "(0x[0-9a-f][0-9a-f], ){16}" "\\0\n    " FORMATTED_LINES "${FORMATTED_BYTES}")

    file(WRITE "${out_header}"
"#pragma once

// Auto-generated embedded default neural reconstructor weights
// Generated from data/models/upways_weights.bin by CMake build process
// Source SHA256: ${BIN_SHA256}
#include <cstdint>
#include <cstddef>

namespace upways {

constexpr size_t EMBEDDED_WEIGHTS_SIZE = ${BYTE_COUNT};

inline constexpr uint8_t DEFAULT_UPWAYS_WEIGHTS[] = {
    ${FORMATTED_LINES}
};

} // namespace upways
")
    message(STATUS "UpwaysPackaging: Embedded ${BYTE_COUNT} bytes (SHA256: ${BIN_SHA256}) into ${out_header}")
endfunction()

# 1. Check if upstream Upways has updated production weights to ingest
if(EXISTS "${UPWAYS_PRODUCTION_BIN}")
    set(NEEDS_SYNC FALSE)
    if(NOT EXISTS "${PATHWAYS_MODEL_BIN}")
        set(NEEDS_SYNC TRUE)
    else()
        file(SHA256 "${UPWAYS_PRODUCTION_BIN}" UPSTREAM_SHA256)
        file(SHA256 "${PATHWAYS_MODEL_BIN}" LOCAL_SHA256)
        if(NOT "${UPSTREAM_SHA256}" STREQUAL "${LOCAL_SHA256}")
            set(NEEDS_SYNC TRUE)
        endif()
    endif()

    if(NEEDS_SYNC)
        message(STATUS "UpwaysPackaging: Ingesting updated Upways weights from ${UPWAYS_PRODUCTION_BIN}")
        file(MAKE_DIRECTORY "${PATHWAYS_MODEL_DIR}")
        file(COPY_FILE "${UPWAYS_PRODUCTION_BIN}" "${PATHWAYS_MODEL_BIN}")
        generate_embedded_weights("${PATHWAYS_MODEL_BIN}" "${PATHWAYS_DEFAULT_HPP}")
    endif()
endif()

# 2. Always ensure upways_default_weights.hpp is synchronized with data/models/upways_weights.bin
if(EXISTS "${PATHWAYS_MODEL_BIN}")
    file(SHA256 "${PATHWAYS_MODEL_BIN}" LOCAL_BIN_SHA256)
    set(NEEDS_EMBED FALSE)
    if(NOT EXISTS "${PATHWAYS_DEFAULT_HPP}")
        set(NEEDS_EMBED TRUE)
    else()
        file(READ "${PATHWAYS_DEFAULT_HPP}" DEFAULT_HPP_CONTENT)
        string(FIND "${DEFAULT_HPP_CONTENT}" "${LOCAL_BIN_SHA256}" SHA_FOUND)
        if(SHA_FOUND EQUAL -1)
            set(NEEDS_EMBED TRUE)
        endif()
    endif()

    if(NEEDS_EMBED)
        message(STATUS "UpwaysPackaging: Synchronizing embedded C++ header from ${PATHWAYS_MODEL_BIN}")
        generate_embedded_weights("${PATHWAYS_MODEL_BIN}" "${PATHWAYS_DEFAULT_HPP}")
    endif()
endif()

