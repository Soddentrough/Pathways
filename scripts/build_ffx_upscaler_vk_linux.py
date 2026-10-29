#!/usr/bin/env python3
"""
Builds native Linux libamd_fidelityfx_upscaler_vk.so (AMD FSR 3.1.6 Vulkan Provider)
from the open-source FidelityFX SDK 2.4.0 C++ host and Vulkan backend sources.
"""
import os
import shutil
import struct
import subprocess
import sys
import zlib


def main():
    if len(sys.argv) < 4:
        print("Usage: build_ffx_upscaler_vk_linux.py <ffx_sdk_dir> <gen_dir> <out_so_path> [cxx_compiler]")
        sys.exit(1)

    sdk_dir = os.path.realpath(sys.argv[1])
    out_dir = os.path.realpath(sys.argv[2])
    out_so = os.path.realpath(sys.argv[3])
    cxx = "clang++" if shutil.which("clang++") else (sys.argv[4] if len(sys.argv) > 4 else "g++")

    dll_path = os.path.join(sdk_dir, "Kits/FidelityFX/signedbin/amd_fidelityfx_upscaler_vk.dll")
    if not os.path.exists(dll_path):
        print(f"FidelityFX upscaler binary not found at {dll_path}")
        sys.exit(1)

    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.dirname(out_so), exist_ok=True)

    with open(dll_path, "rb") as f:
        data = f.read()

    e_lfanew = struct.unpack_from("<I", data, 0x3c)[0]
    num_sections = struct.unpack_from("<H", data, e_lfanew + 6)[0]
    opt_hdr_size = struct.unpack_from("<H", data, e_lfanew + 20)[0]
    image_base = struct.unpack_from("<Q", data, e_lfanew + 24 + 24)[0]

    sections = []
    sec_offset = e_lfanew + 24 + opt_hdr_size
    for i in range(num_sections):
        name, vsize, rva, raw_size, raw_ptr = struct.unpack_from("<8sIIII", data, sec_offset + i * 40)
        sections.append((name.rstrip(b'\x00').decode(), image_base + rva, vsize, raw_ptr, raw_size))

    def va_to_offset(va):
        if va == 0:
            return None
        for _, sec_va, vsize, raw_ptr, raw_size in sections:
            if sec_va <= va < sec_va + max(vsize, raw_size):
                off = va - sec_va
                if off < raw_size:
                    return raw_ptr + off
        return None

    def read_cstr(va):
        off = va_to_offset(va)
        if off is None:
            return ""
        end = data.find(b'\x00', off)
        return data[off:end].decode('utf-8', errors='replace')

    def read_str_array(va, count):
        if count == 0 or va == 0:
            return []
        off = va_to_offset(va)
        ptrs = struct.unpack_from(f"<{count}Q", data, off)
        return [read_cstr(p) for p in ptrs]

    def read_u32_array(va, count):
        if count == 0 or va == 0:
            return []
        off = va_to_offset(va)
        return list(struct.unpack_from(f"<{count}I", data, off))

    funcs = [
        ("prepare_reactivity_pass", 0x180112750, 0x180112fc0, ["_wave64_16bit", "_wave64", "_16bit", ""]),
        ("shading_change_pass", 0x180112fc0, 0x180113830, ["_wave64_16bit", "_wave64", "_16bit", ""]),
        ("prepare_inputs_pass", 0x180113830, 0x180113c80, ["_wave64", ""]),
        ("accumulate_pass", 0x180113c80, 0x180114500, ["_wave64_16bit", "_wave64", "_16bit", ""]),
        ("rcas_pass", 0x180114500, 0x180114950, ["_wave64", ""]),
        ("luma_pyramid_pass", 0x180114950, 0x180114da0, ["_wave64", ""]),
        ("autogen_reactive_pass", 0x180114da0, 0x180115610, ["_wave64_16bit", "_wave64", "_16bit", ""]),
        ("shading_change_pyramid_pass", 0x180115610, 0x180115a60, ["_wave64", ""]),
        ("luma_instability_pass", 0x180115a60, 0x180115eb0, ["_wave64", ""]),
        ("debug_view_pass", 0x180115eb0, 0x180116300, ["_wave64", ""]),
    ]

    def map_options_to_dll_key(pass_name, options):
        if pass_name == "accumulate_pass":
            return options
        elif pass_name == "prepare_inputs_pass":
            return ((options & 0x3) << 3) | ((options >> 2) & 0x7) | (options & 0x20)
        else:
            return ((options & 0x7) << 2) | ((options >> 3) & 0x3) | (options & 0x20)

    all_suffixes = ["", "_wave64", "_16bit", "_wave64_16bit"]

    for pass_name, start_va, end_va, suffixes in funcs:
        off_s = va_to_offset(start_va)
        off_e = va_to_offset(end_va)
        chunk = data[off_s:off_e]
        idx = 0
        tables = []
        while True:
            idx = chunk.find(b'\x48\x69\xc8\x30\x01\x00\x00', idx)
            if idx == -1:
                break
            ind_rva = struct.unpack_from("<I", chunk, idx - 4)[0]
            info_field_rva = struct.unpack_from("<I", chunk, idx + 7 + 4)[0]
            info_base_rva = info_field_rva - 8
            tables.append((ind_rva, info_base_rva))
            idx += 7

        suffix_map = dict(zip(suffixes, tables))
        for sfx in all_suffixes:
            hdr_name = f"ffx_fsr3upscaler_{pass_name}{sfx}_permutations.h"
            hdr_path = os.path.join(out_dir, hdr_name)
            prefix = f"ffx_fsr3upscaler_{pass_name}{sfx}"
            lines = ["#pragma once", "#include <stdint.h>", ""]
            if sfx == "":
                lines.append(f"""#ifndef FFX_FSR3UPSCALER_{pass_name.upper()}_PERMUTATIONKEY_DEFINED
#define FFX_FSR3UPSCALER_{pass_name.upper()}_PERMUTATIONKEY_DEFINED
typedef union ffx_fsr3upscaler_{pass_name}_PermutationKey {{
    struct {{
        uint32_t FFX_FSR3UPSCALER_OPTION_REPROJECT_USE_LANCZOS_TYPE : 1;
        uint32_t FFX_FSR3UPSCALER_OPTION_HDR_COLOR_INPUT : 1;
        uint32_t FFX_FSR3UPSCALER_OPTION_LOW_RESOLUTION_MOTION_VECTORS : 1;
        uint32_t FFX_FSR3UPSCALER_OPTION_JITTERED_MOTION_VECTORS : 1;
        uint32_t FFX_FSR3UPSCALER_OPTION_INVERTED_DEPTH : 1;
        uint32_t FFX_FSR3UPSCALER_OPTION_APPLY_SHARPENING : 1;
    }};
    uint32_t index;
}} ffx_fsr3upscaler_{pass_name}_PermutationKey;
#endif
""")
            lines.append(f"""typedef struct {prefix}_PermutationInfo {{
    const uint32_t       blobSize;
    const unsigned char* blobData;
    const char*          entryName;
    const uint32_t       numConstantBuffers;
    const uint32_t       numSRVTextures;
    const uint32_t       numUAVTextures;
    const uint32_t       numSRVBuffers;
    const uint32_t       numUAVBuffers;
    const uint32_t       numSamplers;
    const uint32_t       numRTAccelerationStructures;
    const char**         constantBufferNames;
    const uint32_t*      constantBufferBindings;
    const uint32_t*      constantBufferCounts;
    const uint32_t*      constantBufferSpaces;
    const char**         srvTextureNames;
    const uint32_t*      srvTextureBindings;
    const uint32_t*      srvTextureCounts;
    const uint32_t*      srvTextureSpaces;
    const char**         uavTextureNames;
    const uint32_t*      uavTextureBindings;
    const uint32_t*      uavTextureCounts;
    const uint32_t*      uavTextureSpaces;
    const char**         srvBufferNames;
    const uint32_t*      srvBufferBindings;
    const uint32_t*      srvBufferCounts;
    const uint32_t*      srvBufferSpaces;
    const char**         uavBufferNames;
    const uint32_t*      uavBufferBindings;
    const uint32_t*      uavBufferCounts;
    const uint32_t*      uavBufferSpaces;
    const char**         samplerNames;
    const uint32_t*      samplerBindings;
    const uint32_t*      samplerCounts;
    const uint32_t*      samplerSpaces;
    const char**         rtAccelerationStructureNames;
    const uint32_t*      rtAccelerationStructureBindings;
    const uint32_t*      rtAccelerationStructureCounts;
    const uint32_t*      rtAccelerationStructureSpaces;
}} {prefix}_PermutationInfo;
""")

            if sfx not in suffix_map:
                with open(hdr_path, "w") as fp:
                    fp.write("\n".join(lines))
                continue

            ind_rva, info_rva = suffix_map[sfx]
            ind_off = va_to_offset(image_base + ind_rva)
            dll_ind_table = struct.unpack_from("<64i", data, ind_off)
            remapped_ind = [dll_ind_table[map_options_to_dll_key(pass_name, opt)] for opt in range(64)]
            max_idx = max(dll_ind_table)
            info_off = va_to_offset(image_base + info_rva)

            info_entries_str = []
            for ti in range(max_idx + 1):
                e_off = info_off + ti * 0x130
                blob_size, blob_va, entry_va = struct.unpack_from("<IxxxxQQ", data, e_off)
                b_off = va_to_offset(blob_va)
                raw_blob = data[b_off:b_off + blob_size]
                spv = zlib.decompress(raw_blob[8:])
                entry_name = read_cstr(entry_va)

                hex_bytes = ", ".join(f"0x{b:02x}" for b in spv)
                lines.append(f"static const unsigned char g_{prefix}_blob_{ti}[] = {{ {hex_bytes} }};")

                groups = [
                    ("cb", 0x18),
                    ("srvTex", 0x40),
                    ("uavTex", 0x68),
                    ("srvBuf", 0x90),
                    ("uavBuf", 0xb8),
                    ("samp", 0xe0),
                    ("rtas", 0x108),
                ]
                counts = {}
                ptrs_code = {}
                for gname, goff in groups:
                    cnt, n_va, b_va, c_va, s_va = struct.unpack_from("<IxxxxQQQQ", data, e_off + goff)
                    counts[gname] = cnt
                    if cnt > 0:
                        names = read_str_array(n_va, cnt)
                        binds = read_u32_array(b_va, cnt)
                        cnts = read_u32_array(c_va, cnt)
                        spcs = read_u32_array(s_va, cnt)
                        names_str = ", ".join(f'"{n}"' for n in names)
                        binds_str = ", ".join(str(x) for x in binds)
                        cnts_str = ", ".join(str(x) for x in cnts)
                        spcs_str = ", ".join(str(x) for x in spcs)
                        lines.append(f"static const char* g_{prefix}_{ti}_{gname}_names[] = {{ {names_str} }};")
                        lines.append(f"static const uint32_t g_{prefix}_{ti}_{gname}_bindings[] = {{ {binds_str} }};")
                        lines.append(f"static const uint32_t g_{prefix}_{ti}_{gname}_counts[] = {{ {cnts_str} }};")
                        lines.append(f"static const uint32_t g_{prefix}_{ti}_{gname}_spaces[] = {{ {spcs_str} }};")
                        ptrs_code[gname] = f"g_{prefix}_{ti}_{gname}_names, g_{prefix}_{ti}_{gname}_bindings, g_{prefix}_{ti}_{gname}_counts, g_{prefix}_{ti}_{gname}_spaces"
                    else:
                        ptrs_code[gname] = "nullptr, nullptr, nullptr, nullptr"

                info_entries_str.append(
                    f"    {{ {len(spv)}, g_{prefix}_blob_{ti}, \"{entry_name}\", "
                    f"{counts['cb']}, {counts['srvTex']}, {counts['uavTex']}, {counts['srvBuf']}, {counts['uavBuf']}, {counts['samp']}, {counts['rtas']}, "
                    f"{ptrs_code['cb']}, {ptrs_code['srvTex']}, {ptrs_code['uavTex']}, {ptrs_code['srvBuf']}, {ptrs_code['uavBuf']}, {ptrs_code['samp']}, {ptrs_code['rtas']} }}"
                )

            ind_str = ", ".join(str(x) for x in remapped_ind)
            lines.append(f"static const int32_t g_{prefix}_IndirectionTable[] = {{ {ind_str} }};")
            lines.append(f"static const {prefix}_PermutationInfo g_{prefix}_PermutationInfo[] = {{\n" + ",\n".join(info_entries_str) + "\n};")

            with open(hdr_path, "w") as fp:
                fp.write("\n".join(lines) + "\n")

    with open(os.path.join(out_dir, "miniz.h"), "w") as f:
        f.write("""#pragma once
#include <stdint.h>
#include <zlib.h>
typedef uint8_t mz_uint8;
typedef unsigned long mz_ulong;
#define MZ_OK Z_OK
static inline int mz_uncompress(unsigned char* pDest, mz_ulong* pDest_len, const unsigned char* pSource, mz_ulong source_len) {
    return uncompress(pDest, pDest_len, pSource, source_len);
}
""")

    with open(os.path.join(out_dir, "git_hash_branch.h"), "w") as f:
        f.write('#pragma once\n#define GIT_HASH_BRANCH L"FidelityFX-SDK-2.4.0"\n')

    with open(os.path.join(out_dir, "ffx_linux_compat.h"), "w") as f:
        f.write("""#pragma once
#pragma clang diagnostic ignored "-Wignored-attributes"
#pragma clang diagnostic ignored "-Wunknown-warning-option"
#pragma clang diagnostic ignored "-Wunknown-pragmas"
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#pragma clang diagnostic ignored "-Wabsolute-value"

#include <stddef.h>
#include <stdint.h>
#include <string.h>
#include <wchar.h>

#ifdef __cplusplus
#include <cmath>
#include <locale>
#include <codecvt>
using std::log2;
using std::floor;
#endif

#ifndef UNREFERENCED_PARAMETER
#define UNREFERENCED_PARAMETER(x) ((void)(x))
#endif

#ifndef FFX_UNUSED
#define FFX_UNUSED(x) ((void)(x))
#endif

#ifndef _countof
#define _countof(arr) (sizeof(arr) / sizeof((arr)[0]))
#endif

#ifndef _TRUNCATE
#define _TRUNCATE ((size_t)-1)
#endif

#ifndef swprintf_s
#define swprintf_s swprintf
#endif

#ifdef __cplusplus
class ffxProviderExternal;

template <size_t N>
static inline int wcscpy_s(wchar_t (&dst)[N], const wchar_t* src) {
    wcsncpy(dst, src ? src : L"", N);
    dst[N - 1] = L'\\0';
    return 0;
}

static inline int wcscpy_s(wchar_t* dst, size_t dstSize, const wchar_t* src) {
    if (!dst || dstSize == 0) return -1;
    wcsncpy(dst, src ? src : L"", dstSize);
    dst[dstSize - 1] = L'\\0';
    return 0;
}

static inline int strcpy_s(char* dst, size_t dstSize, const char* src) {
    if (!dst || dstSize == 0) return -1;
    strncpy(dst, src ? src : "", dstSize - 1);
    dst[dstSize - 1] = '\\0';
    return 0;
}

static inline int strncpy_s(char* dst, size_t dstSize, const char* src, size_t count) {
    if (!dst || dstSize == 0) return -1;
    size_t copyLen = (count == _TRUNCATE || count >= dstSize) ? (dstSize - 1) : count;
    strncpy(dst, src ? src : "", copyLen);
    dst[copyLen] = '\\0';
    return 0;
}
#endif
""")

    with open(os.path.join(sdk_dir, "Kits/FidelityFX/upscalers/fsr3/internal/ffx_provider_fsr3upscale.cpp"), "r") as f:
        src = f.read()
    src = src.replace(
        '#define MAKE_VERSION_STRING(major, minor, patch) STRINGIFY major "." STRINGIFY minor "." STRINGIFY patch',
        '#define MAKE_VERSION_STRING(major, minor, patch) "3.1.6"'
    )
    src = src.replace("0xF5A5'CA1Eui64", "0xF5A5CA1EULL")
    with open(os.path.join(out_dir, "ffx_provider_fsr3upscale_linux.cpp"), "w") as f:
        f.write(src)

    with open(os.path.join(sdk_dir, "Kits/FidelityFX/api/internal/ffx_internal_types.h"), "r") as f:
        itypes = f.read()
    itypes = itypes.replace("#define FFX_SDK_DEFAULT_CONTEXT_SIZE (1024 * 128)", "#define FFX_SDK_DEFAULT_CONTEXT_SIZE (1024 * 256)")
    with open(os.path.join(out_dir, "ffx_internal_types.h"), "w") as f:
        f.write(itypes)

    with open(os.path.join(out_dir, "ffx_provider.h"), "w") as f:
        f.write(f"""#pragma once
#include "{sdk_dir}/Kits/FidelityFX/api/include/ffx_api.hpp"
#include "{sdk_dir}/Kits/FidelityFX/api/include/ffx_api_types.h"
#include "{sdk_dir}/Kits/FidelityFX/api/internal/ffx_internal_types.h"
#include "{sdk_dir}/Kits/FidelityFX/api/internal/ffx_api_helper.h"

#include <optional>
#include <span>
#include <utility>

class ffxProvider
{{
public:
    ffxProvider(uint64_t i_id, uint64_t i_effect_id, const char* i_version_name)
        : RefCount(1), Id(i_id), EffectId(i_effect_id), VersionName(i_version_name)
    {{
    }}

    virtual ~ffxProvider() = default;

    size_t GetRefCount() const {{ return RefCount; }}
    uint64_t GetId() const {{ return Id; }}
    const char* GetVersionName() const {{ return VersionName; }}

    virtual bool CanProvide(uint64_t descType) const
    {{
        return (descType & FFX_API_EFFECT_MASK) == EffectId;
    }}

    virtual bool IsSupported(void* device) const
    {{
        (void)device;
        return true;
    }}

    virtual ffxReturnCode_t CreateContext(ffxContext* context, ffxCreateContextDescHeader* desc, Allocator& alloc) = 0;
    virtual ffxReturnCode_t DestroyContext(ffxContext* context, Allocator& alloc) = 0;
    virtual ffxReturnCode_t Configure(ffxContext* context, const ffxConfigureDescHeader* desc) const = 0;
    virtual ffxReturnCode_t Query(ffxContext* context, ffxQueryDescHeader* desc) const = 0;
    virtual ffxReturnCode_t Dispatch(ffxContext* context, const ffxDispatchDescHeader* desc) const = 0;

protected:
    size_t RefCount = 0;
    uint64_t Id;
    uint64_t EffectId;
    const char* VersionName;
}};

class ffxProviderExternal : public ffxProvider
{{
public:
    ffxProviderExternal() : ffxProvider(0, 0, "") {{}}
    ffxProviderExternal(void*, uint64_t) : ffxProvider(0, 0, "") {{}}
    bool Valid() const {{ return false; }}
    ffxReturnCode_t CreateContext(ffxContext*, ffxCreateContextDescHeader*, Allocator&) override {{ return FFX_API_RETURN_NO_PROVIDER; }}
    ffxReturnCode_t DestroyContext(ffxContext*, Allocator&) override {{ return FFX_API_RETURN_NO_PROVIDER; }}
    ffxReturnCode_t Configure(ffxContext*, const ffxConfigureDescHeader*) const override {{ return FFX_API_RETURN_NO_PROVIDER; }}
    ffxReturnCode_t Query(ffxContext*, ffxQueryDescHeader*) const override {{ return FFX_API_RETURN_NO_PROVIDER; }}
    ffxReturnCode_t Dispatch(ffxContext*, const ffxDispatchDescHeader*) const override {{ return FFX_API_RETURN_NO_PROVIDER; }}
}};

struct InternalContextHeader
{{
    ffxProvider* provider;
}};

inline ffxProvider* GetAssociatedProvider(ffxContext context)
{{
    return reinterpret_cast<InternalContextHeader*>(context)->provider;
}}

ffxProvider* GetProvider(ffxStructType_t descType, uint64_t overrideId, void* device, std::optional<ffxProviderExternal>& extProviderSlot);

inline ffxProvider* GetProvider(ffxStructType_t descType, uint64_t overrideId, void* device, std::optional<ffxProviderExternal>& extProviderSlot, std::span<ffxProvider* const> providers)
{{
    (void)extProviderSlot;
    if (overrideId)
    {{
        for (auto provider : providers)
        {{
            if (provider->GetId() == overrideId && provider->CanProvide(descType) && provider->IsSupported(device))
                return provider;
        }}
    }}
    else
    {{
        for (auto provider : providers)
        {{
            if (provider->CanProvide(descType) && provider->IsSupported(device))
                return provider;
        }}
    }}
    return nullptr;
}}

uint64_t GetProviderVersions(ffxStructType_t descType, void* device, uint64_t capacity, uint64_t* versionIds, const char** versionNames, std::optional<ffxProviderExternal>& extProviderSlot);

inline uint64_t GetProviderVersions(ffxStructType_t descType, void* device, uint64_t capacity, uint64_t* versionIds, const char** versionNames, std::optional<ffxProviderExternal>& extProviderSlot, std::span<ffxProvider* const> providers)
{{
    (void)extProviderSlot;
    uint64_t count = 0;
    for (auto provider : providers)
    {{
        if (count >= capacity) break;
        if (provider->CanProvide(descType) && provider->IsSupported(device))
        {{
            auto index = count++;
            if (versionIds) versionIds[index] = provider->GetId();
            if (versionNames) versionNames[index] = provider->GetVersionName();
        }}
    }}
    return count;
}}

inline uint64_t GetProviderCount(ffxStructType_t descType, void* device, std::optional<ffxProviderExternal>& extProviderSlot)
{{
    return GetProviderVersions(descType, device, UINT64_MAX, nullptr, nullptr, extProviderSlot);
}}
""")

    with open(os.path.join(out_dir, "vfs_overlay.yaml"), "w") as f:
        f.write(f"""{{
  'version': 0,
  'roots': [
    {{
      'name': '{sdk_dir}/Kits/FidelityFX/api/internal',
      'type': 'directory',
      'contents': [
        {{
          'name': 'ffx_provider.h',
          'type': 'file',
          'external-contents': '{out_dir}/ffx_provider.h'
        }},
        {{
          'name': 'ffx_internal_types.h',
          'type': 'file',
          'external-contents': '{out_dir}/ffx_internal_types.h'
        }}
      ]
    }}
  ]
}}
""")

    with open(os.path.join(out_dir, "ffx_upscaler_vk_provider_entry.cpp"), "w") as f:
        f.write(f"""#include "{sdk_dir}/Kits/FidelityFX/api/internal/ffx_provider.h"
#include "{sdk_dir}/Kits/FidelityFX/upscalers/include/ffx_upscale.hpp"
#include "{sdk_dir}/Kits/FidelityFX/upscalers/fsr3/include/ffx_provider_fsr3upscale.h"
#include "{sdk_dir}/Kits/FidelityFX/upscalers/fsr3/include/ffx_fsr3upscaler.h"

static ffxProvider* const s_upscaleProviders[] = {{
    &ffxProvider_FSR3Upscale::GetInstance(),
}};

ffxProvider* GetProvider(ffxStructType_t descType, uint64_t overrideId, void* device, std::optional<ffxProviderExternal>& extProviderSlot)
{{
    return GetProvider(descType, overrideId, device, extProviderSlot, s_upscaleProviders);
}}

uint64_t GetProviderVersions(ffxStructType_t descType, void* device, uint64_t capacity, uint64_t* versionIds, const char** versionNames, std::optional<ffxProviderExternal>& extProviderSlot)
{{
    return GetProviderVersions(descType, device, capacity, versionIds, versionNames, extProviderSlot, s_upscaleProviders);
}}

#include "{sdk_dir}/Kits/FidelityFX/api/internal/ffx_api.cpp"
""")

    cmd = [
        cxx, "-std=c++20", "-O3", "-shared", "-fPIC", "-fms-extensions",
        "-ivfsoverlay", os.path.join(out_dir, "vfs_overlay.yaml"),
        "-include", os.path.join(out_dir, "ffx_linux_compat.h"),
        "-DFFX_UPSCALER=1",
        f"-I{out_dir}",
        f"-I{sdk_dir}/Kits/FidelityFX/api/include",
        f"-I{sdk_dir}/Kits/FidelityFX/api/internal",
        f"-I{sdk_dir}/Kits/FidelityFX/backend/vk",
        f"-I{sdk_dir}/Kits/FidelityFX/upscalers/include",
        f"-I{sdk_dir}/Kits/FidelityFX/upscalers/fsr3/include",
        f"-I{sdk_dir}/Kits/FidelityFX/upscalers/fsr3/internal",
        os.path.join(out_dir, "ffx_upscaler_vk_provider_entry.cpp"),
        os.path.join(out_dir, "ffx_provider_fsr3upscale_linux.cpp"),
        os.path.join(sdk_dir, "Kits/FidelityFX/api/internal/ffx_assert.cpp"),
        os.path.join(sdk_dir, "Kits/FidelityFX/api/internal/ffx_message.cpp"),
        os.path.join(sdk_dir, "Kits/FidelityFX/api/internal/ffx_object_management.cpp"),
        os.path.join(sdk_dir, "Kits/FidelityFX/api/internal/ffx_query_fallback.cpp"),
        os.path.join(sdk_dir, "Kits/FidelityFX/backend/vk/ffx_backends_vk.cpp"),
        os.path.join(sdk_dir, "Kits/FidelityFX/backend/vk/ffx_vk.cpp"),
        os.path.join(sdk_dir, "Kits/FidelityFX/upscalers/fsr3/internal/ffx_fsr3upscaler.cpp"),
        os.path.join(sdk_dir, "Kits/FidelityFX/upscalers/fsr3/internal/ffx_fsr3upscaler_shaderblobs.cpp"),
        "-lvulkan", "-lz",
        "-o", out_so,
    ]
    subprocess.check_call(cmd)


if __name__ == "__main__":
    main()
