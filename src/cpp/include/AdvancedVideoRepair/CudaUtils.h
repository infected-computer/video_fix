#pragma once

#ifdef HAVE_CUDA
#include <cuda_runtime.h>
#include <memory>
#include <stdexcept>

namespace VideoRepair {

/**
 * @brief RAII wrapper for CUDA device memory
 * 
 * Automatically manages the lifecycle of CUDA device memory allocations,
 * ensuring proper cleanup and preventing memory leaks.
 */
template<typename T>
class CudaDeviceBuffer {
private:
    T* d_ptr_;
    size_t count_;

public:
    // Constructor that allocates memory
    explicit CudaDeviceBuffer(size_t count) : d_ptr_(nullptr), count_(count) {
        if (count > 0) {
            cudaError_t result = cudaMalloc(&d_ptr_, count * sizeof(T));
            if (result != cudaSuccess) {
                throw std::runtime_error("Failed to allocate CUDA device memory: " + 
                                       std::string(cudaGetErrorString(result)));
            }
        }
    }
    
    // Destructor ensures proper cleanup
    ~CudaDeviceBuffer() {
        if (d_ptr_) {
            cudaFree(d_ptr_);
        }
    }
    
    // Move constructor
    CudaDeviceBuffer(CudaDeviceBuffer&& other) noexcept 
        : d_ptr_(other.d_ptr_), count_(other.count_) {
        other.d_ptr_ = nullptr;
        other.count_ = 0;
    }
    
    // Move assignment
    CudaDeviceBuffer& operator=(CudaDeviceBuffer&& other) noexcept {
        if (this != &other) {
            if (d_ptr_) {
                cudaFree(d_ptr_);
            }
            d_ptr_ = other.d_ptr_;
            count_ = other.count_;
            other.d_ptr_ = nullptr;
            other.count_ = 0;
        }
        return *this;
    }
    
    // Delete copy operations (CUDA memory can't be safely copied)
    CudaDeviceBuffer(const CudaDeviceBuffer&) = delete;
    CudaDeviceBuffer& operator=(const CudaDeviceBuffer&) = delete;
    
    // Access operations
    T* get() const { return d_ptr_; }
    size_t size() const { return count_; }
    size_t size_bytes() const { return count_ * sizeof(T); }
    
    // Conversion to bool for null checks
    explicit operator bool() const { return d_ptr_ != nullptr; }
    
    // Release ownership
    T* release() {
        T* temp = d_ptr_;
        d_ptr_ = nullptr;
        count_ = 0;
        return temp;
    }
    
    // Reset with new allocation
    void reset(size_t new_count = 0) {
        if (d_ptr_) {
            cudaFree(d_ptr_);
        }
        d_ptr_ = nullptr;
        count_ = new_count;
        
        if (new_count > 0) {
            cudaError_t result = cudaMalloc(&d_ptr_, new_count * sizeof(T));
            if (result != cudaSuccess) {
                throw std::runtime_error("Failed to allocate CUDA device memory: " + 
                                       std::string(cudaGetErrorString(result)));
            }
        }
    }
    
    // Copy data from host to device
    cudaError_t copy_from_host(const T* host_data, size_t elements = 0, cudaStream_t stream = 0) {
        if (!host_data || !d_ptr_) return cudaErrorInvalidValue;
        
        size_t copy_count = (elements == 0) ? count_ : std::min(elements, count_);
        size_t copy_bytes = copy_count * sizeof(T);
        
        if (stream == 0) {
            return cudaMemcpy(d_ptr_, host_data, copy_bytes, cudaMemcpyHostToDevice);
        } else {
            return cudaMemcpyAsync(d_ptr_, host_data, copy_bytes, cudaMemcpyHostToDevice, stream);
        }
    }
    
    // Copy data from device to host
    cudaError_t copy_to_host(T* host_data, size_t elements = 0, cudaStream_t stream = 0) const {
        if (!host_data || !d_ptr_) return cudaErrorInvalidValue;
        
        size_t copy_count = (elements == 0) ? count_ : std::min(elements, count_);
        size_t copy_bytes = copy_count * sizeof(T);
        
        if (stream == 0) {
            return cudaMemcpy(host_data, d_ptr_, copy_bytes, cudaMemcpyDeviceToHost);
        } else {
            return cudaMemcpyAsync(host_data, d_ptr_, copy_bytes, cudaMemcpyDeviceToHost, stream);
        }
    }
    
    // Set memory to zero
    cudaError_t zero(cudaStream_t stream = 0) {
        if (!d_ptr_) return cudaErrorInvalidValue;
        
        if (stream == 0) {
            return cudaMemset(d_ptr_, 0, size_bytes());
        } else {
            return cudaMemsetAsync(d_ptr_, 0, size_bytes(), stream);
        }
    }
};

/**
 * @brief RAII wrapper for CUDA streams
 */
class CudaStreamPtr {
private:
    cudaStream_t stream_;

public:
    CudaStreamPtr() : stream_(nullptr) {
        cudaError_t result = cudaStreamCreate(&stream_);
        if (result != cudaSuccess) {
            throw std::runtime_error("Failed to create CUDA stream: " + 
                                   std::string(cudaGetErrorString(result)));
        }
    }
    
    ~CudaStreamPtr() {
        if (stream_) {
            cudaStreamDestroy(stream_);
        }
    }
    
    // Move semantics
    CudaStreamPtr(CudaStreamPtr&& other) noexcept : stream_(other.stream_) {
        other.stream_ = nullptr;
    }
    
    CudaStreamPtr& operator=(CudaStreamPtr&& other) noexcept {
        if (this != &other) {
            if (stream_) {
                cudaStreamDestroy(stream_);
            }
            stream_ = other.stream_;
            other.stream_ = nullptr;
        }
        return *this;
    }
    
    // Delete copy operations
    CudaStreamPtr(const CudaStreamPtr&) = delete;
    CudaStreamPtr& operator=(const CudaStreamPtr&) = delete;
    
    // Access operations
    cudaStream_t get() const { return stream_; }
    explicit operator bool() const { return stream_ != nullptr; }
    
    // Stream operations
    cudaError_t synchronize() const {
        return cudaStreamSynchronize(stream_);
    }
    
    cudaError_t query() const {
        return cudaStreamQuery(stream_);
    }
};

/**
 * @brief RAII wrapper for CUDA events
 */
class CudaEventPtr {
private:
    cudaEvent_t event_;

public:
    explicit CudaEventPtr(unsigned int flags = cudaEventDefault) : event_(nullptr) {
        cudaError_t result = cudaEventCreateWithFlags(&event_, flags);
        if (result != cudaSuccess) {
            throw std::runtime_error("Failed to create CUDA event: " + 
                                   std::string(cudaGetErrorString(result)));
        }
    }
    
    ~CudaEventPtr() {
        if (event_) {
            cudaEventDestroy(event_);
        }
    }
    
    // Move semantics
    CudaEventPtr(CudaEventPtr&& other) noexcept : event_(other.event_) {
        other.event_ = nullptr;
    }
    
    CudaEventPtr& operator=(CudaEventPtr&& other) noexcept {
        if (this != &other) {
            if (event_) {
                cudaEventDestroy(event_);
            }
            event_ = other.event_;
            other.event_ = nullptr;
        }
        return *this;
    }
    
    // Delete copy operations
    CudaEventPtr(const CudaEventPtr&) = delete;
    CudaEventPtr& operator=(const CudaEventPtr&) = delete;
    
    // Access operations
    cudaEvent_t get() const { return event_; }
    explicit operator bool() const { return event_ != nullptr; }
    
    // Event operations
    cudaError_t record(cudaStream_t stream = 0) const {
        return cudaEventRecord(event_, stream);
    }
    
    cudaError_t synchronize() const {
        return cudaEventSynchronize(event_);
    }
    
    cudaError_t query() const {
        return cudaEventQuery(event_);
    }
    
    float elapsed_time(const CudaEventPtr& start_event) const {
        float milliseconds = 0.0f;
        cudaEventElapsedTime(&milliseconds, start_event.get(), event_);
        return milliseconds;
    }
};

/**
 * @brief RAII wrapper for CUDA context management
 */
class CudaContextGuard {
private:
    CUcontext prev_context_;
    bool context_switched_;

public:
    explicit CudaContextGuard(CUcontext new_context) : prev_context_(nullptr), context_switched_(false) {
        CUresult result = cuCtxGetCurrent(&prev_context_);
        if (result == CUDA_SUCCESS && prev_context_ != new_context) {
            result = cuCtxSetCurrent(new_context);
            if (result == CUDA_SUCCESS) {
                context_switched_ = true;
            }
        }
    }
    
    ~CudaContextGuard() {
        if (context_switched_) {
            cuCtxSetCurrent(prev_context_);
        }
    }
    
    // Delete copy and move operations
    CudaContextGuard(const CudaContextGuard&) = delete;
    CudaContextGuard& operator=(const CudaContextGuard&) = delete;
    CudaContextGuard(CudaContextGuard&&) = delete;
    CudaContextGuard& operator=(CudaContextGuard&&) = delete;
};

/**
 * @brief Utility functions for CUDA error checking
 */
class CudaErrorChecker {
public:
    static void check_cuda_error(cudaError_t result, const char* file, int line) {
        if (result != cudaSuccess) {
            throw std::runtime_error("CUDA error at " + std::string(file) + ":" + 
                                   std::to_string(line) + " - " + 
                                   std::string(cudaGetErrorString(result)));
        }
    }
    
    static void check_last_cuda_error(const char* file, int line) {
        cudaError_t result = cudaGetLastError();
        if (result != cudaSuccess) {
            throw std::runtime_error("CUDA error at " + std::string(file) + ":" + 
                                   std::to_string(line) + " - " + 
                                   std::string(cudaGetErrorString(result)));
        }
    }
};

// Convenience macros for error checking
#define CUDA_CHECK(call) VideoRepair::CudaErrorChecker::check_cuda_error((call), __FILE__, __LINE__)
#define CUDA_CHECK_LAST() VideoRepair::CudaErrorChecker::check_last_cuda_error(__FILE__, __LINE__)

} // namespace VideoRepair

#endif // HAVE_CUDA