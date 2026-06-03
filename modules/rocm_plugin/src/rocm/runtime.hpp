// Copyright (C) 2021-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "hip/hip_runtime.h"
//#include <hip_runtime_api.h>
#include <atomic>
#include <rocm/device_pointers.hpp>
#include <error.hpp>
#include <functional>


inline void throwIfError(
    hipError_t err,
    const std::experimental::source_location& location = std::experimental::source_location::current()) {
    if (err != hipSuccess) ov::rocm_gpu::throw_ov_exception(hipGetErrorString(err), location);
}

inline void logIfError(
    hipError_t err,
    const std::experimental::source_location& location = std::experimental::source_location::current()) {
    if (err != hipSuccess) ov::rocm_gpu::logError(hipGetErrorString(err), location);
}

namespace rocm {

template <typename T>
auto toNative(T&& a) noexcept(noexcept(std::forward<T>(a).get())) -> decltype(std::forward<T>(a).get()) {
    return std::forward<T>(a).get();
}

template <typename T>
auto toNative(T&& a) noexcept(noexcept(std::forward<T>(a).data())) -> decltype(std::forward<T>(a).data()) {
    return std::forward<T>(a).data();
}

template <typename T>
typename std::enable_if<std::is_scalar<typename std::decay<T>::type>::value, typename std::decay<T>::type>::type
toNative(T t) noexcept {
    return t;
}

template <typename T, typename R, typename... Args>
auto createFirstArg(R (*creator)(T*, Args... args), Args... args) {
    T t;
    throwIfError(creator(&t, toNative(std::forward<Args>(args))...));
    return t;
}

template <typename R, typename... ConArgs, typename... Args>
auto createLastArg(R (*creator)(ConArgs...), Args... args) {
    using LastType = typename std::remove_pointer<
        typename std::tuple_element<sizeof...(ConArgs) - 1, std::tuple<ConArgs...>>::type>::type;
    LastType t;
    throwIfError(creator(toNative(std::forward<Args>(args))..., &t));
    return t;
}

class Device {
    int id;

public:
    Device() : Device{currentId()} {}
    explicit Device(int id) noexcept : id{id} {}
    static int currentId() { return createFirstArg(hipGetDevice); }
    static int count() { return createFirstArg(hipGetDeviceCount); }
    hipDeviceProp_t props() const { return createFirstArg(hipGetDeviceProperties, id); }
    const Device& setCurrent() const {
        throwIfError(hipSetDevice(id));
        return *this;
    }
    void synchronize() { throwIfError(::hipDeviceSynchronize()); }
};

constexpr auto memoryAlignment = 256;
constexpr auto defaultResidentGrids = 16;

inline int residentGrids(const hipDeviceProp_t& p) {
    return p.concurrentKernels;
}

inline int max_concurrent_streams(rocm::Device d) {
    auto p = d.props();
    int r = p.asyncEngineCount;
    if (!p.concurrentKernels) return r + 1;
    return r + p.concurrentKernels;
}

inline bool isHalfSupported(rocm::Device d) {
    return false;  // half_float::half is CPU-only; use f32 path
}

inline bool isInt8Supported(rocm::Device d) {
    return true;
}

template <typename T>
class Handle {
public:
    using Native = T;
    using Shared = std::shared_ptr<Native>;

    template <typename R, typename... Args>
    using Construct = R (*)(T*, Args... args);

    template <typename R>
    using Destruct = R (*)(Native);

    virtual ~Handle() = 0;

    explicit operator bool() const { return native_.operator bool(); }

    const Native& get() const noexcept { return *native_; }
    const Shared& get_shared() const noexcept { return native_; }

protected:
    template <typename R, typename... Args>
    Handle(Construct<R, Args...> constructor, Destruct<R> destructor, Args... args) {
        auto native = Native{createFirstArg(constructor, args...)};
        try {
            native_ =
                std::shared_ptr<Native>(std::make_unique<Native>(native).release(), [destructor](const Native* native) {
                    if (destructor) {
                        logIfError(destructor(*native));
                    }
                    delete native;
                });
        } catch (...) {
            if (destructor) {
                logIfError(destructor(native));
            }
            throw;
        }
    }

    template <typename R, typename... Args>
    Handle(Construct<R, Args...> constructor, std::nullptr_t, Args... args) {
        auto native = Native{createFirstArg(constructor, args...)};
        native_ = std::shared_ptr<Native>(std::make_unique<Native>(native).release());
    }

private:
    std::shared_ptr<Native> native_;
};

template <typename T>
inline Handle<T>::~Handle() {}

class DefaultAllocation {
    struct Deleter {
        void operator()(void* p) const noexcept { logIfError(hipFree(p)); }
    };
    std::shared_ptr<void> p;

public:
    explicit DefaultAllocation(void* p) noexcept : p{p, Deleter{}} {}
    void* get() const noexcept { return p.get(); }
    template <typename T, typename std::enable_if<std::is_void<T>::value>::type* = nullptr>
    operator DevicePointer<T*>() const noexcept {
        return DevicePointer<T*>{get()};
    }
};

class Allocation {
    class Deleter {
        Handle<hipStream_t>::Shared stream;

        auto freeImpl(void* p) const noexcept {
//#if rocmRT_VERSION >= 11020
//            return hipFreeAsync(p, *stream);
//#else
            return hipFree(p);
//#endif
        }

    public:
        Deleter(const Handle<hipStream_t>& stream) noexcept : stream{stream.get_shared()} {}
        void operator()(void* p) const noexcept { logIfError(freeImpl(p)); }
    };

    std::shared_ptr<void> p;

public:
    Allocation(void* p, const Handle<hipStream_t>& stream) noexcept : p{p, Deleter{stream}} {}
    void* get() const noexcept { return p.get(); }
    template <typename T, typename std::enable_if<std::is_void<T>::value>::type* = nullptr>
    operator DevicePointer<T*>() const noexcept {
        return DevicePointer<T*>{get()};
    }
};

class Stream : public Handle<hipStream_t> {
public:
    Stream() : Handle((hipStreamCreate), hipStreamDestroy) {}

    Allocation malloc(std::size_t size) const { return {mallocImpl(size), *this}; }
    void upload(rocm::DevicePointer<void*> dst, const void* src, std::size_t count) const {
        uploadImpl(dst.get(), src, count);
    }
    void transfer(rocm::DevicePointer<void*> dst, rocm::DevicePointer<const void*> src, std::size_t count) const {
        throwIfError(hipMemcpyAsync(dst.get(), src.get(), count, hipMemcpyDeviceToDevice, get()));
    }
    void upload(const Allocation& dst, const void* src, std::size_t count) const { uploadImpl(dst.get(), src, count); }
    void download(void* dst, const Allocation& src, std::size_t count) const { downloadImpl(dst, src.get(), count); }
    void download(void* dst, rocm::DevicePointer<const void*> src, std::size_t count) const {
        downloadImpl(dst, src.get(), count);
    }
    void download(void* dst, rocm::DevicePointer<void*> src, std::size_t count) const {
        downloadImpl(dst, src.get(), count);
    }
    void memset(const Allocation& dst, int value, std::size_t count) const { memsetImpl(dst.get(), value, count); }
    void memset(rocm::DevicePointer<void*> dst, int value, std::size_t count) const {
        memsetImpl(dst.get(), value, count);
    }
    void synchronize() const { throwIfError(hipStreamSynchronize(get())); }

    template <typename... Args>
    void run(dim3 gridDim, dim3 blockDim, void (*kernel)(Args...), Args... args) const {
        //kernel<<<dim3(gridDim), dim3(blockDim), 0, get()>>>(args...);
        hipLaunchKernelGGL(kernel, dim3(gridDim), dim3(blockDim), 0, get(), args...);
        //kernel(args...);
    }


private:
    void uploadImpl(void* dst, const void* src, std::size_t count) const {
        throwIfError(hipMemcpyAsync(dst, src, count, hipMemcpyHostToDevice, get()));
    }
    void downloadImpl(void* dst, const void* src, std::size_t count) const {
        throwIfError(hipMemcpyAsync(dst, src, count, hipMemcpyDeviceToHost, get()));
    }
    void* mallocImpl(std::size_t size) const {
        return createFirstArg<void*, hipError_t>(
//#if rocmRT_VERSION >= 11020
//            hipMallocAsync, size, get()
//#else
            hipMalloc, size
//#endif
        );
    }
    void memsetImpl(void* dst, int value, size_t count) const {
        throwIfError(hipMemsetAsync(dst, value, count, get()));
    }
};

class DefaultStream {
    void uploadImpl(void* dst, const void* src, std::size_t count) const {
        throwIfError(hipMemcpy(dst, src, count, hipMemcpyHostToDevice));
    }
    void downloadImpl(void* dst, const void* src, std::size_t count) const {
        throwIfError(hipMemcpy(dst, src, count, hipMemcpyDeviceToHost));
    }
    void memsetImpl(void* dst, int value, std::size_t count) const { throwIfError(hipMemset(dst, value, count)); }
    DefaultStream() = default;

public:
    static DefaultStream& stream() {
        static DefaultStream stream{};
        return stream;
    }

    auto malloc(std::size_t size) const {
        return DefaultAllocation{createFirstArg<void*, hipError_t>(hipMalloc, size)};
    }
    void upload(DevicePointer<void*> dst, const void* src, std::size_t count) const {
        uploadImpl(dst.get(), src, count);
    }
    void upload(const DefaultAllocation& dst, const void* src, std::size_t count) const {
        uploadImpl(dst.get(), src, count);
    }
    void download(void* dst, const DefaultAllocation& src, std::size_t count) const {
        downloadImpl(dst, src.get(), count);
    }
    void download(void* dst, DevicePointer<const void*> src, std::size_t count) const {
        downloadImpl(dst, src.get(), count);
    }
    void memset(const DefaultAllocation& dst, int value, std::size_t count) const {
        memsetImpl(dst.get(), value, count);
    }
    void memset(rocm::DevicePointer<void*> dst, int value, std::size_t count) const {
        memsetImpl(dst.get(), value, count);
    }
};

}  // namespace rocm
