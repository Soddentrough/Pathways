#!/usr/bin/env python3
"""
Builds a native Linux shared library (libamd_fidelityfx_upscaler_fsr4_vk.so) for
AMD FidelityFX Super Resolution 4.1.2 (ML Super-Resolution) on Vulkan.

Because AMD FidelityFX SDK 2.4.0 ships the FSR 4.1.2 FP8/INT8 cooperative-matrix
weights and SPIR-V shaders exclusively inside the signed Windows PE32+ binary
(Kits/FidelityFX/signedbin/amd_fidelityfx_upscaler_vk.dll), this generator builds
a lightweight, zero-dependency Linux ELF shared library that maps the PE32+ image
in-process and bridges Linux SysV x86_64 calls to the DLL's Microsoft x64 ABI
(__attribute__((ms_abi))) entry points and Vulkan dispatch table.

Usage:
    python3 scripts/build_ffx_upscaler_fsr4_vk_linux.py <ffx_sdk_dir> <build_gen_dir> <output_so> [cxx_compiler]
"""

import os
import subprocess
import sys
from pathlib import Path


BRIDGE_CPP_SOURCE = r'''#include <vulkan/vulkan.h>

#include <algorithm>
#include <cmath>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <cwchar>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include <pthread.h>
#include <sys/mman.h>
#include <unistd.h>

#define MS_ABI __attribute__((ms_abi))

namespace {

// Wrap a static SysV function into an MS_ABI function
template <auto RealFn>
struct MsAbiStatic;

template <typename R, typename... Args, R (*RealFn)(Args...)>
struct MsAbiStatic<RealFn> {
    static MS_ABI R call(Args... args) {
        return RealFn(args...);
    }
};

// Wrap a dynamic SysV function pointer slot into an MS_ABI function
template <typename FnPtr, FnPtr* Slot>
struct MsAbiDynamic;

template <typename R, typename... Args, R (**Slot)(Args...)>
struct MsAbiDynamic<R (*)(Args...), Slot> {
    static MS_ABI R call(Args... args) {
        return (*Slot)(args...);
    }
};

struct VkThunkEntry {
    const char* name;
    PFN_vkVoidFunction* realSlot;
    PFN_vkVoidFunction msThunk;
};

#define DEFINE_VK_THUNK(fn) \
    static PFN_##fn s_real_##fn = nullptr;

DEFINE_VK_THUNK(vkAllocateDescriptorSets)
DEFINE_VK_THUNK(vkAllocateMemory)
DEFINE_VK_THUNK(vkBindBufferMemory)
DEFINE_VK_THUNK(vkBindImageMemory)
DEFINE_VK_THUNK(vkCmdBeginDebugUtilsLabelEXT)
DEFINE_VK_THUNK(vkCmdBindDescriptorSets)
DEFINE_VK_THUNK(vkCmdBindPipeline)
DEFINE_VK_THUNK(vkCmdClearColorImage)
DEFINE_VK_THUNK(vkCmdCopyBuffer)
DEFINE_VK_THUNK(vkCmdCopyBufferToImage)
DEFINE_VK_THUNK(vkCmdCopyImage)
DEFINE_VK_THUNK(vkCmdDispatch)
DEFINE_VK_THUNK(vkCmdDispatchIndirect)
DEFINE_VK_THUNK(vkCmdEndDebugUtilsLabelEXT)
DEFINE_VK_THUNK(vkCmdFillBuffer)
DEFINE_VK_THUNK(vkCmdPipelineBarrier)
DEFINE_VK_THUNK(vkCmdWriteBufferMarker2AMD)
DEFINE_VK_THUNK(vkCmdWriteBufferMarkerAMD)
DEFINE_VK_THUNK(vkCreateBuffer)
DEFINE_VK_THUNK(vkCreateBufferView)
DEFINE_VK_THUNK(vkCreateComputePipelines)
DEFINE_VK_THUNK(vkCreateDescriptorPool)
DEFINE_VK_THUNK(vkCreateDescriptorSetLayout)
DEFINE_VK_THUNK(vkCreateImage)
DEFINE_VK_THUNK(vkCreateImageView)
DEFINE_VK_THUNK(vkCreatePipelineLayout)
DEFINE_VK_THUNK(vkCreateSampler)
DEFINE_VK_THUNK(vkCreateShaderModule)
DEFINE_VK_THUNK(vkDestroyBuffer)
DEFINE_VK_THUNK(vkDestroyBufferView)
DEFINE_VK_THUNK(vkDestroyDescriptorPool)
DEFINE_VK_THUNK(vkDestroyDescriptorSetLayout)
DEFINE_VK_THUNK(vkDestroyImage)
DEFINE_VK_THUNK(vkDestroyImageView)
DEFINE_VK_THUNK(vkDestroyPipeline)
DEFINE_VK_THUNK(vkDestroyPipelineLayout)
DEFINE_VK_THUNK(vkDestroySampler)
DEFINE_VK_THUNK(vkDestroyShaderModule)
DEFINE_VK_THUNK(vkFlushMappedMemoryRanges)
DEFINE_VK_THUNK(vkFreeDescriptorSets)
DEFINE_VK_THUNK(vkFreeMemory)
DEFINE_VK_THUNK(vkGetBufferMemoryRequirements)
DEFINE_VK_THUNK(vkGetBufferMemoryRequirements2KHR)
DEFINE_VK_THUNK(vkGetImageMemoryRequirements)
DEFINE_VK_THUNK(vkMapMemory)
DEFINE_VK_THUNK(vkSetDebugUtilsObjectNameEXT)
DEFINE_VK_THUNK(vkUnmapMemory)
DEFINE_VK_THUNK(vkUpdateDescriptorSets)

#define ENTRY_VK_THUNK(fn) \
    { #fn, reinterpret_cast<PFN_vkVoidFunction*>(&s_real_##fn), reinterpret_cast<PFN_vkVoidFunction>(MsAbiDynamic<PFN_##fn, &s_real_##fn>::call) }

static VkThunkEntry s_vkThunks[] = {
    ENTRY_VK_THUNK(vkAllocateDescriptorSets),
    ENTRY_VK_THUNK(vkAllocateMemory),
    ENTRY_VK_THUNK(vkBindBufferMemory),
    ENTRY_VK_THUNK(vkBindImageMemory),
    ENTRY_VK_THUNK(vkCmdBeginDebugUtilsLabelEXT),
    ENTRY_VK_THUNK(vkCmdBindDescriptorSets),
    ENTRY_VK_THUNK(vkCmdBindPipeline),
    ENTRY_VK_THUNK(vkCmdClearColorImage),
    ENTRY_VK_THUNK(vkCmdCopyBuffer),
    ENTRY_VK_THUNK(vkCmdCopyBufferToImage),
    ENTRY_VK_THUNK(vkCmdCopyImage),
    ENTRY_VK_THUNK(vkCmdDispatch),
    ENTRY_VK_THUNK(vkCmdDispatchIndirect),
    ENTRY_VK_THUNK(vkCmdEndDebugUtilsLabelEXT),
    ENTRY_VK_THUNK(vkCmdFillBuffer),
    ENTRY_VK_THUNK(vkCmdPipelineBarrier),
    ENTRY_VK_THUNK(vkCmdWriteBufferMarker2AMD),
    ENTRY_VK_THUNK(vkCmdWriteBufferMarkerAMD),
    ENTRY_VK_THUNK(vkCreateBuffer),
    ENTRY_VK_THUNK(vkCreateBufferView),
    ENTRY_VK_THUNK(vkCreateComputePipelines),
    ENTRY_VK_THUNK(vkCreateDescriptorPool),
    ENTRY_VK_THUNK(vkCreateDescriptorSetLayout),
    ENTRY_VK_THUNK(vkCreateImage),
    ENTRY_VK_THUNK(vkCreateImageView),
    ENTRY_VK_THUNK(vkCreatePipelineLayout),
    ENTRY_VK_THUNK(vkCreateSampler),
    ENTRY_VK_THUNK(vkCreateShaderModule),
    ENTRY_VK_THUNK(vkDestroyBuffer),
    ENTRY_VK_THUNK(vkDestroyBufferView),
    ENTRY_VK_THUNK(vkDestroyDescriptorPool),
    ENTRY_VK_THUNK(vkDestroyDescriptorSetLayout),
    ENTRY_VK_THUNK(vkDestroyImage),
    ENTRY_VK_THUNK(vkDestroyImageView),
    ENTRY_VK_THUNK(vkDestroyPipeline),
    ENTRY_VK_THUNK(vkDestroyPipelineLayout),
    ENTRY_VK_THUNK(vkDestroySampler),
    ENTRY_VK_THUNK(vkDestroyShaderModule),
    ENTRY_VK_THUNK(vkFlushMappedMemoryRanges),
    ENTRY_VK_THUNK(vkFreeDescriptorSets),
    ENTRY_VK_THUNK(vkFreeMemory),
    ENTRY_VK_THUNK(vkGetBufferMemoryRequirements),
    ENTRY_VK_THUNK(vkGetBufferMemoryRequirements2KHR),
    ENTRY_VK_THUNK(vkGetImageMemoryRequirements),
    ENTRY_VK_THUNK(vkMapMemory),
    ENTRY_VK_THUNK(vkSetDebugUtilsObjectNameEXT),
    ENTRY_VK_THUNK(vkUnmapMemory),
    ENTRY_VK_THUNK(vkUpdateDescriptorSets),
};

static MS_ABI PFN_vkVoidFunction ms_vkGetDeviceProcAddr(VkDevice device, const char* pName) {
    if (!pName) return nullptr;
    if (std::strcmp(pName, "vkGetDeviceProcAddr") == 0) {
        return reinterpret_cast<PFN_vkVoidFunction>(&ms_vkGetDeviceProcAddr);
    }
    PFN_vkVoidFunction realFn = vkGetDeviceProcAddr(device, pName);
    if (!realFn) {
        return nullptr;
    }
    for (auto& entry : s_vkThunks) {
        if (std::strcmp(entry.name, pName) == 0) {
            *entry.realSlot = realFn;
            return entry.msThunk;
        }
    }
    return nullptr;
}

// CRT & Win32 shims
static MS_ABI void* ms_malloc(size_t sz) { return std::malloc(sz); }
static MS_ABI void ms_free(void* p) { std::free(p); }
static MS_ABI void* ms_calloc(size_t n, size_t sz) { return std::calloc(n, sz); }
static MS_ABI int ms_callnewh(size_t) { return 0; }
static MS_ABI void* ms_memcpy(void* d, const void* s, size_t n) { return std::memcpy(d, s, n); }
static MS_ABI void* ms_memmove(void* d, const void* s, size_t n) { return std::memmove(d, s, n); }
static MS_ABI void* ms_memset(void* d, int c, size_t n) { return std::memset(d, c, n); }
static MS_ABI int ms_memcmp(const void* a, const void* b, size_t n) { return std::memcmp(a, b, n); }
static MS_ABI int ms_strcmp(const char* a, const char* b) { return std::strcmp(a, b); }

static MS_ABI int ms_strncpy_s(char* dst, size_t dstSize, const char* src, size_t count) {
    if (!dst || dstSize == 0) return 22;
    if (!src) { dst[0] = '\0'; return 0; }
    size_t copyLen = (count == static_cast<size_t>(-1) || count >= dstSize) ? (dstSize - 1) : count;
    std::strncpy(dst, src, copyLen);
    dst[copyLen] = '\0';
    return 0;
}

static MS_ABI int ms_wcscpy_s(uint16_t* dst, size_t dstSize, const uint16_t* src) {
    if (!dst || dstSize == 0) return 22;
    if (!src) { dst[0] = 0; return 0; }
    size_t i = 0;
    for (; i + 1 < dstSize && src[i] != 0; ++i) {
        dst[i] = src[i];
    }
    dst[i] = 0;
    return 0;
}

static MS_ABI int ms_getenv_s(size_t* pReturnValue, char* buffer, size_t numberOfElements, const char* varname) {
    const char* val = varname ? std::getenv(varname) : nullptr;
    if (!val) {
        if (pReturnValue) *pReturnValue = 0;
        if (buffer && numberOfElements > 0) buffer[0] = '\0';
        return 0;
    }
    size_t needed = std::strlen(val) + 1;
    if (pReturnValue) *pReturnValue = needed;
    if (buffer && numberOfElements >= needed) {
        std::memcpy(buffer, val, needed);
        return 0;
    }
    return 34;
}

static MS_ABI double ms_log2(double x) { return std::log2(x); }
static MS_ABI float ms_log2f(float x) { return std::log2f(x); }
static MS_ABI double ms_floor(double x) { return std::floor(x); }
static MS_ABI float ms_floorf(float x) { return std::floorf(x); }
static MS_ABI float ms_cosf(float x) { return std::cosf(x); }
static MS_ABI float ms_sinf(float x) { return std::sinf(x); }
static MS_ABI float ms_tanf(float x) { return std::tanf(x); }
static MS_ABI float ms_atanf(float x) { return std::atanf(x); }
static MS_ABI float ms_powf(float x, float y) { return std::powf(x, y); }
static MS_ABI float ms_exp2f(float x) { return std::exp2f(x); }
static MS_ABI float ms_roundf(float x) { return std::roundf(x); }

static MS_ABI int16_t ms_dclass(double x) { return static_cast<int16_t>(std::fpclassify(x)); }
static MS_ABI int16_t ms_fdclass(float x) { return static_cast<int16_t>(std::fpclassify(x)); }
static MS_ABI int16_t ms_ldclass(long double x) { return static_cast<int16_t>(std::fpclassify(x)); }
static MS_ABI int ms_dsign(double x) { return std::signbit(x) ? 0x8000 : 0; }
static MS_ABI int ms_fdsign(float x) { return std::signbit(x) ? 0x8000 : 0; }
static MS_ABI int ms_ldsign(long double x) { return std::signbit(x) ? 0x8000 : 0; }

static std::recursive_mutex s_globalMtx;
static MS_ABI void ms_Mtx_init_in_situ(void* mtx, int) {
    if (mtx) std::memset(mtx, 0, 64);
}
static MS_ABI int ms_Mtx_lock(void*) {
    s_globalMtx.lock();
    return 0;
}
static MS_ABI int ms_Mtx_unlock(void*) {
    s_globalMtx.unlock();
    return 0;
}

static MS_ABI void ms_AcquireSRWLockExclusive(void*) { s_globalMtx.lock(); }
static MS_ABI void ms_ReleaseSRWLockExclusive(void*) { s_globalMtx.unlock(); }
static MS_ABI void ms_WakeAllConditionVariable(void*) {}
static MS_ABI int ms_SleepConditionVariableSRW(void*, void*, uint32_t, uint32_t) { return 1; }
static MS_ABI int ms_DisableThreadLibraryCalls(void*) { return 1; }
static MS_ABI void ms_InitializeSListHead(void* p) { if (p) std::memset(p, 0, 16); }
static MS_ABI uint32_t ms_GetCurrentThreadId() { return static_cast<uint32_t>(gettid()); }
static MS_ABI uint32_t ms_GetCurrentProcessId() { return static_cast<uint32_t>(getpid()); }
static MS_ABI void* ms_GetCurrentProcess() { return reinterpret_cast<void*>(static_cast<intptr_t>(-1)); }
static MS_ABI int ms_IsDebuggerPresent() { return 0; }
static MS_ABI int ms_IsProcessorFeaturePresent(uint32_t) { return 1; }
static MS_ABI uint64_t ms_GetTickCount64() {
    struct timespec ts{};
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return static_cast<uint64_t>(ts.tv_sec) * 1000ULL + static_cast<uint64_t>(ts.tv_nsec) / 1000000ULL;
}
static MS_ABI int ms_QueryPerformanceCounter(uint64_t* out) {
    struct timespec ts{};
    clock_gettime(CLOCK_MONOTONIC, &ts);
    if (out) *out = static_cast<uint64_t>(ts.tv_sec) * 1000000000ULL + static_cast<uint64_t>(ts.tv_nsec);
    return 1;
}
static MS_ABI void ms_GetSystemTimeAsFileTime(uint64_t* out) {
    if (out) *out = 0;
}
static MS_ABI void ms_OutputDebugStringA(const char*) {}
static MS_ABI void ms_OutputDebugStringW(const uint16_t*) {}
static MS_ABI void* ms_LoadLibraryA(const char*) { return nullptr; }
static MS_ABI void* ms_LoadLibraryExA(const char*, void*, uint32_t) { return nullptr; }
static MS_ABI void* ms_GetModuleHandleA(const char*) { return nullptr; }
static MS_ABI void* ms_GetProcAddress(void*, const char*) { return nullptr; }
static MS_ABI int ms_FreeLibrary(void*) { return 1; }
static MS_ABI uint32_t ms_GetModuleFileNameA(void*, char* buf, uint32_t sz) {
    if (buf && sz > 0) buf[0] = '\0';
    return 0;
}
static MS_ABI int ms_MultiByteToWideChar(uint32_t, uint32_t, const char* src, int srcLen, uint16_t* dst, int dstLen) {
    if (!src) return 0;
    int len = (srcLen < 0) ? (static_cast<int>(std::strlen(src)) + 1) : srcLen;
    if (!dst || dstLen == 0) return len;
    int n = (len < dstLen) ? len : dstLen;
    for (int i = 0; i < n; ++i) dst[i] = static_cast<uint8_t>(src[i]);
    return n;
}
static MS_ABI int ms_WideCharToMultiByte(uint32_t, uint32_t, const uint16_t* src, int srcLen, char* dst, int dstLen, const char*, int*) {
    if (!src) return 0;
    int len = 0;
    if (srcLen < 0) {
        while (src[len] != 0) ++len;
        ++len;
    } else {
        len = srcLen;
    }
    if (!dst || dstLen == 0) return len;
    int n = (len < dstLen) ? len : dstLen;
    for (int i = 0; i < n; ++i) dst[i] = static_cast<char>(src[i] & 0xFF);
    return n;
}
static MS_ABI uint32_t ms_GetFileVersionInfoSizeA(const char*, uint32_t* lpdwHandle) {
    if (lpdwHandle) *lpdwHandle = 0;
    return 0;
}
static MS_ABI int ms_GetFileVersionInfoA(const char*, uint32_t, uint32_t, void*) { return 0; }
static MS_ABI int ms_VerQueryValueA(const void*, const char*, void**, uint32_t*) { return 0; }

#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wformat-nonliteral"
#endif
static MS_ABI int ms_stdio_vsprintf(uint64_t, char* buf, size_t sz, const char* fmt, void*, __builtin_ms_va_list ap) {
    if (!buf || sz == 0 || !fmt) return -1;
    uint64_t a0 = __builtin_va_arg(ap, uint64_t);
    uint64_t a1 = __builtin_va_arg(ap, uint64_t);
    uint64_t a2 = __builtin_va_arg(ap, uint64_t);
    uint64_t a3 = __builtin_va_arg(ap, uint64_t);
    return std::snprintf(buf, sz, fmt, a0, a1, a2, a3);
}
#if defined(__clang__)
#pragma clang diagnostic pop
#endif

static MS_ABI int ms_stdio_vswprintf_s(uint64_t, uint16_t* buf, size_t sz, const uint16_t*, void*, __builtin_ms_va_list) {
    if (buf && sz > 0) buf[0] = 0;
    return 0;
}
static MS_ABI int ms_noop() { return 0; }

// FidelityFX ABI structures
struct ffxApiHeader {
    uint64_t type;
    ffxApiHeader* pNext;
};

struct VkDeviceContext {
    VkDevice vkDevice;
    VkPhysicalDevice vkPhysicalDevice;
    PFN_vkGetDeviceProcAddr vkDeviceProcAddr;
};

struct ffxCreateBackendVKDesc {
    ffxApiHeader header;
    VkDeviceContext deviceContext;
};

struct FfxApiDimensions2D {
    uint32_t width;
    uint32_t height;
};

struct ffxCreateContextDescUpscale {
    ffxApiHeader header;
    uint32_t flags;
    FfxApiDimensions2D maxRenderSize;
    FfxApiDimensions2D maxUpscaleSize;
    void* fpMessage;
};

struct ffxQueryDescGetVersions {
    ffxApiHeader header;
    uint64_t createDescType;
    void* device;
    uint64_t* outputCount;
    uint64_t* versionIds;
    const char** versionNames;
};

struct ffxQueryDescUpscaleGetGPUMemoryUsageV2 {
    ffxApiHeader header;
    void* device;
    FfxApiDimensions2D maxRenderSize;
    FfxApiDimensions2D maxUpscaleSize;
    uint32_t flags;
    void* gpuMemoryUsageUpscaler;
};

using PfnMsFfxCreateContext  = MS_ABI uint32_t (*)(void** context, ffxApiHeader* desc, const void* memCb);
using PfnMsFfxDestroyContext = MS_ABI uint32_t (*)(void** context, const void* memCb);
using PfnMsFfxConfigure      = MS_ABI uint32_t (*)(void** context, const ffxApiHeader* desc);
using PfnMsFfxQuery          = MS_ABI uint32_t (*)(void** context, ffxApiHeader* desc);
using PfnMsFfxDispatch       = MS_ABI uint32_t (*)(void** context, const ffxApiHeader* desc);

static std::once_flag s_initOnce;
static bool s_loaded = false;
static PfnMsFfxCreateContext  s_msCreateContext  = nullptr;
static PfnMsFfxDestroyContext s_msDestroyContext = nullptr;
static PfnMsFfxConfigure      s_msConfigure      = nullptr;
static PfnMsFfxQuery          s_msQuery          = nullptr;
static PfnMsFfxDispatch       s_msDispatch       = nullptr;
static int32_t s_tlsEpoch = INT32_MIN;

static std::string locatePeDll() {
    std::vector<std::filesystem::path> candidates;
    if (const char* dllEnv = std::getenv("FFX_DLL_PATH")) {
        candidates.push_back(std::filesystem::path(dllEnv) / "amd_fidelityfx_upscaler_vk.dll");
    }
    if (const char* sdkEnv = std::getenv("FFX_SDK_PATH")) {
        candidates.push_back(std::filesystem::path(sdkEnv) / "Kits/FidelityFX/signedbin/amd_fidelityfx_upscaler_vk.dll");
    }
#ifdef FFX_DEFAULT_DLL_PATH
    candidates.emplace_back(FFX_DEFAULT_DLL_PATH);
#endif
    if (const char* home = std::getenv("HOME")) {
        candidates.push_back(std::filesystem::path(home) / "Development/FidelityFX-SDK/Kits/FidelityFX/signedbin/amd_fidelityfx_upscaler_vk.dll");
    }
    std::error_code ec;
    auto exePath = std::filesystem::read_symlink("/proc/self/exe", ec);
    if (!ec && exePath.has_parent_path()) {
        candidates.push_back(exePath.parent_path() / "amd_fidelityfx_upscaler_vk.dll");
    }

    for (const auto& p : candidates) {
        if (std::filesystem::exists(p, ec)) {
            return p.string();
        }
    }
    return {};
}

static void initPeModule() {
    std::string dllPath = locatePeDll();
    if (dllPath.empty()) {
        return;
    }

    std::ifstream f(dllPath, std::ios::binary | std::ios::ate);
    if (!f) {
        return;
    }
    size_t fileSize = static_cast<size_t>(f.tellg());
    if (fileSize < 0x1000) {
        return;
    }
    std::vector<uint8_t> raw(fileSize);
    f.seekg(0);
    f.read(reinterpret_cast<char*>(raw.data()), static_cast<std::streamsize>(fileSize));

    if (raw[0] != 'M' || raw[1] != 'Z') {
        return;
    }
    uint32_t e_lfanew = *reinterpret_cast<const uint32_t*>(raw.data() + 0x3c);
    if (e_lfanew + 24 + 240 > raw.size()) {
        return;
    }
    uint16_t numSections = *reinterpret_cast<const uint16_t*>(raw.data() + e_lfanew + 6);
    uint16_t optHdrSize  = *reinterpret_cast<const uint16_t*>(raw.data() + e_lfanew + 20);
    uint16_t peMagic     = *reinterpret_cast<const uint16_t*>(raw.data() + e_lfanew + 24);
    if (peMagic != 0x20B) {
        return;
    }

    uint64_t imageBase     = *reinterpret_cast<const uint64_t*>(raw.data() + e_lfanew + 24 + 24);
    uint32_t sizeOfImage   = *reinterpret_cast<const uint32_t*>(raw.data() + e_lfanew + 24 + 56);
    uint32_t sizeOfHeaders = *reinterpret_cast<const uint32_t*>(raw.data() + e_lfanew + 24 + 60);

    void* mapped = mmap(reinterpret_cast<void*>(imageBase), sizeOfImage,
                        PROT_READ | PROT_WRITE | PROT_EXEC,
                        MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (mapped == MAP_FAILED) {
        return;
    }
    uint8_t* base = static_cast<uint8_t*>(mapped);

    std::memcpy(base, raw.data(), std::min<size_t>(sizeOfHeaders, raw.size()));
    size_t secOffset = e_lfanew + 24 + optHdrSize;
    for (uint16_t i = 0; i < numSections; ++i) {
        const uint8_t* sh = raw.data() + secOffset + i * 40;
        uint32_t vsize   = *reinterpret_cast<const uint32_t*>(sh + 8);
        uint32_t rva     = *reinterpret_cast<const uint32_t*>(sh + 12);
        uint32_t rawSize = *reinterpret_cast<const uint32_t*>(sh + 16);
        uint32_t rawPtr  = *reinterpret_cast<const uint32_t*>(sh + 20);
        uint32_t copyLen = std::min(vsize, rawSize);
        if (copyLen > 0 && rawPtr + copyLen <= raw.size() && rva + copyLen <= sizeOfImage) {
            std::memcpy(base + rva, raw.data() + rawPtr, copyLen);
        }
    }

    // Apply base relocations if mapped address differs from preferred imageBase
    int64_t delta = reinterpret_cast<int64_t>(base) - static_cast<int64_t>(imageBase);
    if (delta != 0) {
        uint32_t relocRva  = *reinterpret_cast<const uint32_t*>(raw.data() + e_lfanew + 24 + 112 + 5 * 8);
        uint32_t relocSize = *reinterpret_cast<const uint32_t*>(raw.data() + e_lfanew + 24 + 112 + 5 * 8 + 4);
        uint32_t off = 0;
        while (off + 8 <= relocSize) {
            uint32_t pageRva   = *reinterpret_cast<const uint32_t*>(base + relocRva + off);
            uint32_t blockSize = *reinterpret_cast<const uint32_t*>(base + relocRva + off + 4);
            if (blockSize < 8) break;
            uint32_t count = (blockSize - 8) / 2;
            const uint16_t* entries = reinterpret_cast<const uint16_t*>(base + relocRva + off + 8);
            for (uint32_t i = 0; i < count; ++i) {
                uint16_t type   = entries[i] >> 12;
                uint16_t offset = entries[i] & 0x0FFF;
                if (type == 10) {
                    *reinterpret_cast<uint64_t*>(base + pageRva + offset) += delta;
                }
            }
            off += blockSize;
        }
    }

    // Resolve Import Address Table (IAT)
    std::unordered_map<std::string, void*> symMap = {
        {"vkGetBufferMemoryRequirements", reinterpret_cast<void*>(&MsAbiStatic<vkGetBufferMemoryRequirements>::call)},
        {"vkGetPhysicalDeviceProperties2", reinterpret_cast<void*>(&MsAbiStatic<vkGetPhysicalDeviceProperties2>::call)},
        {"vkCreateImage", reinterpret_cast<void*>(&MsAbiStatic<vkCreateImage>::call)},
        {"vkDestroyImage", reinterpret_cast<void*>(&MsAbiStatic<vkDestroyImage>::call)},
        {"vkEnumerateDeviceExtensionProperties", reinterpret_cast<void*>(&MsAbiStatic<vkEnumerateDeviceExtensionProperties>::call)},
        {"vkGetPhysicalDeviceProperties", reinterpret_cast<void*>(&MsAbiStatic<vkGetPhysicalDeviceProperties>::call)},
        {"vkAllocateMemory", reinterpret_cast<void*>(&MsAbiStatic<vkAllocateMemory>::call)},
        {"vkDestroyBuffer", reinterpret_cast<void*>(&MsAbiStatic<vkDestroyBuffer>::call)},
        {"vkGetDeviceProcAddr", reinterpret_cast<void*>(&ms_vkGetDeviceProcAddr)},
        {"vkCreateBuffer", reinterpret_cast<void*>(&MsAbiStatic<vkCreateBuffer>::call)},
        {"vkGetPhysicalDeviceFeatures2", reinterpret_cast<void*>(&MsAbiStatic<vkGetPhysicalDeviceFeatures2>::call)},
        {"vkGetPhysicalDeviceMemoryProperties", reinterpret_cast<void*>(&MsAbiStatic<vkGetPhysicalDeviceMemoryProperties>::call)},
        {"vkGetImageMemoryRequirements", reinterpret_cast<void*>(&MsAbiStatic<vkGetImageMemoryRequirements>::call)},
        {"malloc", reinterpret_cast<void*>(&ms_malloc)},
        {"free", reinterpret_cast<void*>(&ms_free)},
        {"calloc", reinterpret_cast<void*>(&ms_calloc)},
        {"_callnewh", reinterpret_cast<void*>(&ms_callnewh)},
        {"memcpy", reinterpret_cast<void*>(&ms_memcpy)},
        {"memmove", reinterpret_cast<void*>(&ms_memmove)},
        {"memset", reinterpret_cast<void*>(&ms_memset)},
        {"memcmp", reinterpret_cast<void*>(&ms_memcmp)},
        {"strcmp", reinterpret_cast<void*>(&ms_strcmp)},
        {"strncpy_s", reinterpret_cast<void*>(&ms_strncpy_s)},
        {"wcscpy_s", reinterpret_cast<void*>(&ms_wcscpy_s)},
        {"getenv_s", reinterpret_cast<void*>(&ms_getenv_s)},
        {"log2", reinterpret_cast<void*>(&ms_log2)},
        {"log2f", reinterpret_cast<void*>(&ms_log2f)},
        {"floor", reinterpret_cast<void*>(&ms_floor)},
        {"floorf", reinterpret_cast<void*>(&ms_floorf)},
        {"cosf", reinterpret_cast<void*>(&ms_cosf)},
        {"sinf", reinterpret_cast<void*>(&ms_sinf)},
        {"tanf", reinterpret_cast<void*>(&ms_tanf)},
        {"atanf", reinterpret_cast<void*>(&ms_atanf)},
        {"powf", reinterpret_cast<void*>(&ms_powf)},
        {"exp2f", reinterpret_cast<void*>(&ms_exp2f)},
        {"roundf", reinterpret_cast<void*>(&ms_roundf)},
        {"_dclass", reinterpret_cast<void*>(&ms_dclass)},
        {"_fdclass", reinterpret_cast<void*>(&ms_fdclass)},
        {"_ldclass", reinterpret_cast<void*>(&ms_ldclass)},
        {"_dsign", reinterpret_cast<void*>(&ms_dsign)},
        {"_fdsign", reinterpret_cast<void*>(&ms_fdsign)},
        {"_ldsign", reinterpret_cast<void*>(&ms_ldsign)},
        {"__stdio_common_vsprintf", reinterpret_cast<void*>(&ms_stdio_vsprintf)},
        {"__stdio_common_vswprintf_s", reinterpret_cast<void*>(&ms_stdio_vswprintf_s)},
        {"_Mtx_init_in_situ", reinterpret_cast<void*>(&ms_Mtx_init_in_situ)},
        {"_Mtx_lock", reinterpret_cast<void*>(&ms_Mtx_lock)},
        {"_Mtx_unlock", reinterpret_cast<void*>(&ms_Mtx_unlock)},
        {"GetFileVersionInfoSizeA", reinterpret_cast<void*>(&ms_GetFileVersionInfoSizeA)},
        {"GetFileVersionInfoA", reinterpret_cast<void*>(&ms_GetFileVersionInfoA)},
        {"VerQueryValueA", reinterpret_cast<void*>(&ms_VerQueryValueA)},
        {"AcquireSRWLockExclusive", reinterpret_cast<void*>(&ms_AcquireSRWLockExclusive)},
        {"ReleaseSRWLockExclusive", reinterpret_cast<void*>(&ms_ReleaseSRWLockExclusive)},
        {"WakeAllConditionVariable", reinterpret_cast<void*>(&ms_WakeAllConditionVariable)},
        {"SleepConditionVariableSRW", reinterpret_cast<void*>(&ms_SleepConditionVariableSRW)},
        {"DisableThreadLibraryCalls", reinterpret_cast<void*>(&ms_DisableThreadLibraryCalls)},
        {"InitializeSListHead", reinterpret_cast<void*>(&ms_InitializeSListHead)},
        {"GetCurrentThreadId", reinterpret_cast<void*>(&ms_GetCurrentThreadId)},
        {"GetCurrentProcessId", reinterpret_cast<void*>(&ms_GetCurrentProcessId)},
        {"GetCurrentProcess", reinterpret_cast<void*>(&ms_GetCurrentProcess)},
        {"IsDebuggerPresent", reinterpret_cast<void*>(&ms_IsDebuggerPresent)},
        {"IsProcessorFeaturePresent", reinterpret_cast<void*>(&ms_IsProcessorFeaturePresent)},
        {"GetTickCount64", reinterpret_cast<void*>(&ms_GetTickCount64)},
        {"QueryPerformanceCounter", reinterpret_cast<void*>(&ms_QueryPerformanceCounter)},
        {"GetSystemTimeAsFileTime", reinterpret_cast<void*>(&ms_GetSystemTimeAsFileTime)},
        {"OutputDebugStringA", reinterpret_cast<void*>(&ms_OutputDebugStringA)},
        {"OutputDebugStringW", reinterpret_cast<void*>(&ms_OutputDebugStringW)},
        {"LoadLibraryA", reinterpret_cast<void*>(&ms_LoadLibraryA)},
        {"LoadLibraryExA", reinterpret_cast<void*>(&ms_LoadLibraryExA)},
        {"GetModuleHandleA", reinterpret_cast<void*>(&ms_GetModuleHandleA)},
        {"GetProcAddress", reinterpret_cast<void*>(&ms_GetProcAddress)},
        {"FreeLibrary", reinterpret_cast<void*>(&ms_FreeLibrary)},
        {"GetModuleFileNameA", reinterpret_cast<void*>(&ms_GetModuleFileNameA)},
        {"MultiByteToWideChar", reinterpret_cast<void*>(&ms_MultiByteToWideChar)},
        {"WideCharToMultiByte", reinterpret_cast<void*>(&ms_WideCharToMultiByte)},
    };

    uint32_t impRva = *reinterpret_cast<const uint32_t*>(raw.data() + e_lfanew + 24 + 112 + 1 * 8);
    if (impRva != 0) {
        for (uint32_t idx = 0;; ++idx) {
            const uint32_t* desc = reinterpret_cast<const uint32_t*>(base + impRva + idx * 20);
            uint32_t origFirstThunk = desc[0];
            uint32_t nameRva        = desc[3];
            uint32_t firstThunk     = desc[4];
            if (!origFirstThunk && !nameRva && !firstThunk) break;
            uint32_t thunkRva = origFirstThunk ? origFirstThunk : firstThunk;
            const uint64_t* lookup = reinterpret_cast<const uint64_t*>(base + thunkRva);
            uint64_t* iat = reinterpret_cast<uint64_t*>(base + firstThunk);
            for (uint32_t j = 0; lookup[j] != 0; ++j) {
                if (!(lookup[j] & (1ULL << 63))) {
                    const char* fnName = reinterpret_cast<const char*>(base + (lookup[j] & 0x7FFFFFFFULL) + 2);
                    auto it = symMap.find(fnName);
                    if (it != symMap.end()) {
                        iat[j] = reinterpret_cast<uint64_t>(it->second);
                    } else {
                        iat[j] = reinterpret_cast<uint64_t>(&ms_noop);
                    }
                } else {
                    iat[j] = reinterpret_cast<uint64_t>(&ms_noop);
                }
            }
        }
    }

    // Patch Windows TEB %gs:0x58 / %gs:0x30 TLS epoch accesses in .text
    uint64_t* fakeTlsArray = reinterpret_cast<uint64_t*>(base + 0x2259020);
    fakeTlsArray[0] = reinterpret_cast<uint64_t>(&s_tlsEpoch) - 4;
    for (uint32_t rva : {0x1014u, 0x11c043u, 0x11c0c1u}) {
        uint8_t* p = base + rva;
        int32_t rel = static_cast<int32_t>(0x2259020 - (rva + 7));
        p[0] = 0x48; p[1] = 0x8d; p[2] = 0x05; // leaq fakeTlsArray(%rip), %rax
        std::memcpy(p + 3, &rel, 4);
        p[7] = 0x66; p[8] = 0x90;             // 2-byte nop
    }
    {
        uint8_t* p = base + 0x11c105;
        p[0] = 0x31; p[1] = 0xc0;             // xorl %eax, %eax
        p[2] = 0x0f; p[3] = 0x1f; p[4] = 0x80; p[5] = 0x00; p[6] = 0x00; p[7] = 0x00; p[8] = 0x00;
    }
    // Patch __chkstk (0x11cb70) to retq and _onexit (0x11c450) to xorl %eax, %eax; retq
    base[0x11cb70] = 0xc3;
    base[0x11c450] = 0x31; base[0x11c451] = 0xc0; base[0x11c452] = 0xc3;

    // Run C++ static initializers (__xc_a..__xc_z: 0x1000 and 0x15d0)
    using InitFn = MS_ABI void (*)();
    reinterpret_cast<InitFn>(base + 0x1000)();
    reinterpret_cast<InitFn>(base + 0x15d0)();

    // Resolve PE exports from Export Directory
    uint32_t expRva = *reinterpret_cast<const uint32_t*>(raw.data() + e_lfanew + 24 + 112 + 0 * 8);
    if (expRva != 0) {
        const uint32_t* expDir = reinterpret_cast<const uint32_t*>(base + expRva);
        uint32_t numNames  = expDir[6];
        uint32_t funcsRva  = expDir[7];
        uint32_t namesRva  = expDir[8];
        uint32_t ordsRva   = expDir[9];
        const uint32_t* funcs = reinterpret_cast<const uint32_t*>(base + funcsRva);
        const uint32_t* names = reinterpret_cast<const uint32_t*>(base + namesRva);
        const uint16_t* ords  = reinterpret_cast<const uint16_t*>(base + ordsRva);
        for (uint32_t i = 0; i < numNames; ++i) {
            const char* name = reinterpret_cast<const char*>(base + names[i]);
            void* fnAddr = base + funcs[ords[i]];
            if (std::strcmp(name, "ffxCreateContext") == 0) {
                s_msCreateContext = reinterpret_cast<PfnMsFfxCreateContext>(fnAddr);
            } else if (std::strcmp(name, "ffxDestroyContext") == 0) {
                s_msDestroyContext = reinterpret_cast<PfnMsFfxDestroyContext>(fnAddr);
            } else if (std::strcmp(name, "ffxConfigure") == 0) {
                s_msConfigure = reinterpret_cast<PfnMsFfxConfigure>(fnAddr);
            } else if (std::strcmp(name, "ffxQuery") == 0) {
                s_msQuery = reinterpret_cast<PfnMsFfxQuery>(fnAddr);
            } else if (std::strcmp(name, "ffxDispatch") == 0) {
                s_msDispatch = reinterpret_cast<PfnMsFfxDispatch>(fnAddr);
            }
        }
    }

    s_loaded = (s_msCreateContext && s_msDestroyContext && s_msConfigure && s_msQuery && s_msDispatch);
}

static bool ensureLoaded() {
    std::call_once(s_initOnce, initPeModule);
    return s_loaded;
}

// Temporarily patch any VkDeviceContext::vkDeviceProcAddr in a header chain to ms_vkGetDeviceProcAddr
struct ScopedDescPatch {
    struct SavedProcAddr {
        PFN_vkGetDeviceProcAddr* slot;
        PFN_vkGetDeviceProcAddr orig;
    };
    struct SavedVoidPtr {
        void** slot;
        void* orig;
    };
    std::vector<SavedProcAddr> savedProcs;
    std::vector<SavedVoidPtr> savedPtrs;

    explicit ScopedDescPatch(ffxApiHeader* head) {
        for (ffxApiHeader* h = head; h != nullptr; h = h->pNext) {
            // FFX_API_CREATE_CONTEXT_DESC_TYPE_BACKEND_VK (0x01000003) or DEPRECATED (0x0000003)
            if (h->type == 0x01000003u || h->type == 0x0000003u) {
                auto* vkBackend = reinterpret_cast<ffxCreateBackendVKDesc*>(h);
                savedProcs.push_back({&vkBackend->deviceContext.vkDeviceProcAddr, vkBackend->deviceContext.vkDeviceProcAddr});
                vkBackend->deviceContext.vkDeviceProcAddr = reinterpret_cast<PFN_vkGetDeviceProcAddr>(&ms_vkGetDeviceProcAddr);
            } else if (h->type == 0x00010000u) { // FFX_API_CREATE_CONTEXT_DESC_TYPE_UPSCALE
                auto* up = reinterpret_cast<ffxCreateContextDescUpscale*>(h);
                if (up->fpMessage != nullptr) {
                    savedPtrs.push_back({&up->fpMessage, up->fpMessage});
                    up->fpMessage = nullptr;
                }
            } else if (h->type == 4u) { // FFX_API_QUERY_DESC_TYPE_GET_VERSIONS
                auto* vq = reinterpret_cast<ffxQueryDescGetVersions*>(h);
                if (vq->device != nullptr) {
                    auto* devCtx = static_cast<VkDeviceContext*>(vq->device);
                    savedProcs.push_back({&devCtx->vkDeviceProcAddr, devCtx->vkDeviceProcAddr});
                    devCtx->vkDeviceProcAddr = reinterpret_cast<PFN_vkGetDeviceProcAddr>(&ms_vkGetDeviceProcAddr);
                }
            } else if (h->type == 0x00010005u) { // FFX_API_QUERY_DESC_TYPE_UPSCALE_GPU_MEMORY_USAGE_V2
                auto* mq = reinterpret_cast<ffxQueryDescUpscaleGetGPUMemoryUsageV2*>(h);
                if (mq->device != nullptr) {
                    auto* devCtx = static_cast<VkDeviceContext*>(mq->device);
                    savedProcs.push_back({&devCtx->vkDeviceProcAddr, devCtx->vkDeviceProcAddr});
                    devCtx->vkDeviceProcAddr = reinterpret_cast<PFN_vkGetDeviceProcAddr>(&ms_vkGetDeviceProcAddr);
                }
            }
        }
    }

    ~ScopedDescPatch() {
        for (auto& s : savedProcs) {
            *s.slot = s.orig;
        }
        for (auto& s : savedPtrs) {
            *s.slot = s.orig;
        }
    }
};

static bool isMajorVersion4(uint64_t versionId, const char* versionName) {
    if (versionName && versionName[0] == '4' && versionName[1] == '.') {
        return true;
    }
    uint32_t major32 = static_cast<uint32_t>((versionId >> 32) & 0xFFFFu);
    return major32 == 0x0100u || major32 == 4u;
}

} // namespace

extern "C" {

__attribute__((visibility("default"))) uint32_t ffxCreateContext(void** context, ffxApiHeader* desc, const void* /*memCb*/) {
    if (!context || !desc) {
        return 2; // FFX_API_RETURN_ERROR_PARAMETER
    }
    if (!ensureLoaded()) {
        return 4; // FFX_API_RETURN_NO_PROVIDER
    }
    ScopedDescPatch patch(desc);
    return s_msCreateContext(context, desc, nullptr);
}

__attribute__((visibility("default"))) uint32_t ffxDestroyContext(void** context, const void* /*memCb*/) {
    if (!context || !*context) {
        return 0;
    }
    if (!ensureLoaded()) {
        return 4;
    }
    return s_msDestroyContext(context, nullptr);
}

__attribute__((visibility("default"))) uint32_t ffxConfigure(void** context, const ffxApiHeader* desc) {
    if (!desc) {
        return 2;
    }
    if (!ensureLoaded()) {
        return 4;
    }
    return s_msConfigure(context, desc);
}

__attribute__((visibility("default"))) uint32_t ffxQuery(void** context, ffxApiHeader* desc) {
    if (!desc) {
        return 2;
    }
    if (!ensureLoaded()) {
        return 4;
    }
    ScopedDescPatch patch(desc);

    // Filter FFX_API_QUERY_DESC_TYPE_GET_VERSIONS so this FSR 4 module only reports FSR 4.x versions
    // (leaving FSR 3.1.6 to the native Linux C++ provider libamd_fidelityfx_upscaler_vk.so).
    if (desc->type == 4u) {
        auto* q = reinterpret_cast<ffxQueryDescGetVersions*>(desc);
        uint64_t rawCount = 0;
        ffxQueryDescGetVersions probe = *q;
        probe.outputCount  = &rawCount;
        probe.versionIds   = nullptr;
        probe.versionNames = nullptr;
        uint32_t rc = s_msQuery(context, &probe.header);
        if (rc != 0 || rawCount == 0) {
            if (q->outputCount) *q->outputCount = 0;
            return rc;
        }
        std::vector<uint64_t> ids(rawCount, 0);
        std::vector<const char*> names(rawCount, nullptr);
        probe.versionIds   = ids.data();
        probe.versionNames = names.data();
        rc = s_msQuery(context, &probe.header);
        if (rc != 0) {
            if (q->outputCount) *q->outputCount = 0;
            return rc;
        }
        std::vector<uint64_t> fsr4Ids;
        std::vector<const char*> fsr4Names;
        for (uint64_t i = 0; i < rawCount; ++i) {
            if (isMajorVersion4(ids[i], names[i])) {
                fsr4Ids.push_back(ids[i]);
                fsr4Names.push_back(names[i]);
            }
        }
        if (q->outputCount) {
            uint64_t cap = *q->outputCount;
            if (!q->versionIds && !q->versionNames) {
                *q->outputCount = static_cast<uint64_t>(fsr4Ids.size());
            } else {
                uint64_t n = std::min<uint64_t>(cap, fsr4Ids.size());
                for (uint64_t i = 0; i < n; ++i) {
                    if (q->versionIds) q->versionIds[i] = fsr4Ids[i];
                    if (q->versionNames) q->versionNames[i] = fsr4Names[i];
                }
                *q->outputCount = n;
            }
        }
        return 0;
    }

    return s_msQuery(context, desc);
}

__attribute__((visibility("default"))) uint32_t ffxDispatch(void** context, const ffxApiHeader* desc) {
    if (!context || !*context || !desc) {
        return 2;
    }
    if (!ensureLoaded()) {
        return 4;
    }
    return s_msDispatch(context, desc);
}

} // extern "C"
'''


def main() -> int:
    if len(sys.argv) < 4:
        print(
            "Usage: build_ffx_upscaler_fsr4_vk_linux.py <ffx_sdk_dir> <build_gen_dir> <output_so> [cxx_compiler]",
            file=sys.stderr,
        )
        return 1

    sdk_dir = Path(sys.argv[1]).resolve()
    gen_dir = Path(sys.argv[2]).resolve()
    output_so = Path(sys.argv[3]).resolve()
    cxx_compiler = sys.argv[4] if len(sys.argv) > 4 else os.environ.get("CXX", "clang++")

    dll_path = sdk_dir / "Kits" / "FidelityFX" / "signedbin" / "amd_fidelityfx_upscaler_vk.dll"
    if not dll_path.exists():
        print(f"ERROR: Could not find {dll_path}", file=sys.stderr)
        return 1

    gen_dir.mkdir(parents=True, exist_ok=True)
    output_so.parent.mkdir(parents=True, exist_ok=True)

    bridge_cpp = gen_dir / "ffx_upscaler_fsr4_vk_bridge.cpp"
    bridge_cpp.write_text(BRIDGE_CPP_SOURCE, encoding="utf-8")

    cmd = [
        cxx_compiler,
        "-std=c++20",
        "-O2",
        "-fPIC",
        "-shared",
        "-fvisibility=hidden",
        "-Wall",
        "-Wextra",
        f'-DFFX_DEFAULT_DLL_PATH="{dll_path}"',
        str(bridge_cpp),
        "-lvulkan",
        "-lpthread",
        "-o",
        str(output_so),
    ]

    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"ERROR compiling {output_so}:\n{res.stderr}", file=sys.stderr)
        return res.returncode
    if res.stderr.strip():
        print(res.stderr, file=sys.stderr)

    print(f"Built Linux PE32+ ms_abi FidelityFX Upscaler 4.1.2 ML Vulkan provider: {output_so}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
