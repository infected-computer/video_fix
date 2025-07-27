#ifndef ENHANCED_PERFORMANCE_ENGINE_H
#define ENHANCED_PERFORMANCE_ENGINE_H

#include "AdvancedVideoRepairEngine.h"
#include <memory_resource>
#include <thread_pool>
#include <cuda_runtime.h>
#include <npp.h>
#include <cufft.h>

namespace AdvancedVideoRepair {

/**
 * @brief Enhanced Performance Engine with Critical Optimizations
 * 
 * This addresses the major bottlenecks in the current implementation:
 * 1. Memory management optimization with custom allocators
 * 2. True streaming processing for large files
 * 3. Custom CUDA kernels for video-specific operations
 * 4. Lock-free multi-threading architecture
 * 5. Advanced caching strategies
 */

class EnhancedPerformanceEngine {
public:
    struct PerformanceConfig {
        // Memory management
        size_t frame_pool_size = 64;           // Pre-allocated frame pool
        size_t gpu_memory_pool_mb = 2048;      // GPU memory pool
        bool use_memory_mapping = true;        // Memory-mapped file I/O
        bool enable_zero_copy = true;          // Zero-copy GPU transfers
        
        // Threading
        int decode_threads = 4;                // Dedicated decode threads
        int process_threads = 8;               // Processing thread pool
        int encode_threads = 2;                // Encode threads
        bool use_lockfree_queues = true;       // Lock-free data structures
        
        // GPU optimization
        bool use_custom_cuda_kernels = true;   // Custom CUDA implementations
        bool enable_cuda_streams = true;       // Multiple CUDA streams
        int cuda_stream_count = 4;             // Number of CUDA streams
        bool use_tensor_cores = true;          // Tensor Core acceleration
        
        // Caching
        bool enable_motion_cache = true;       // Cache motion vectors
        bool enable_frame_cache = true;        // Cache decoded frames
        size_t cache_size_mb = 1024;          // Cache memory limit
        
        // Advanced processing
        bool enable_simd_optimization = true;  // SIMD/AVX optimizations
        bool use_npp_acceleration = true;      // NVIDIA Performance Primitives
        bool enable_async_processing = true;   // Asynchronous pipeline
    };

private:
    // Advanced memory management
    class FrameMemoryPool;
    class GPUMemoryManager;
    class LockFreeFrameQueue;
    
    // High-performance processing pipeline
    class StreamingProcessor;
    class CUDAKernelManager;
    class SIMDOptimizedProcessor;
    
    // Caching systems
    class MotionVectorCache;
    class FrameCache;
    
public:
    explicit EnhancedPerformanceEngine(const PerformanceConfig& config = {});
    ~EnhancedPerformanceEngine();
    
    bool initialize();
    void shutdown();
    
    // High-performance repair interface
    AdvancedRepairResult repair_video_streaming(
        const std::string& input_file,
        const std::string& output_file,
        const RepairStrategy& strategy
    );
    
    // Performance monitoring
    struct PerformanceMetrics {
        double frames_per_second = 0.0;
        double gpu_utilization_percent = 0.0;
        double memory_bandwidth_gbps = 0.0;
        size_t peak_memory_usage_mb = 0;
        double cache_hit_ratio = 0.0;
        int active_threads = 0;
    };
    
    PerformanceMetrics get_performance_metrics() const;
    void set_performance_config(const PerformanceConfig& config);

private:
    PerformanceConfig m_config;
    bool m_initialized = false;
    
    // Core components
    std::unique_ptr<FrameMemoryPool> m_frame_pool;
    std::unique_ptr<GPUMemoryManager> m_gpu_memory;
    std::unique_ptr<StreamingProcessor> m_streaming_processor;
    std::unique_ptr<CUDAKernelManager> m_cuda_kernels;
    std::unique_ptr<MotionVectorCache> m_motion_cache;
    std::unique_ptr<FrameCache> m_frame_cache;
    
    // Performance monitoring
    mutable std::mutex m_metrics_mutex;
    PerformanceMetrics m_current_metrics;
    std::chrono::steady_clock::time_point m_last_metrics_update;
    
    // Internal methods
    bool initialize_memory_pools();
    bool initialize_cuda_kernels();
    bool initialize_thread_pools();
    void update_performance_metrics();
};

/**
 * @brief Custom CUDA Kernels for Video-Specific Operations
 */
class CUDAKernelManager {
public:
    explicit CUDAKernelManager();
    ~CUDAKernelManager();
    
    bool initialize(int device_id);
    void cleanup();
    
    // Custom kernels for video repair
    cudaError_t motion_compensated_interpolation_kernel(
        const PtrStepSz<uchar3>& prev_frame,
        const PtrStepSz<uchar3>& next_frame,
        const PtrStepSz<float2>& motion_vectors,
        PtrStepSz<uchar3>& result,
        float temporal_position,
        cudaStream_t stream = 0
    );
    
    cudaError_t hierarchical_motion_estimation_kernel(
        const PtrStepSz<uchar>& frame1,
        const PtrStepSz<uchar>& frame2,
        PtrStepSz<float2>& motion_vectors,
        int block_size,
        int search_range,
        cudaStream_t stream = 0
    );
    
    cudaError_t corruption_detection_kernel(
        const PtrStepSz<uchar3>& frame,
        PtrStepSz<uchar>& corruption_mask,
        float threshold,
        cudaStream_t stream = 0
    );
    
    cudaError_t content_aware_inpainting_kernel(
        const PtrStepSz<uchar3>& input_frame,
        const PtrStepSz<uchar>& mask,
        PtrStepSz<uchar3>& output_frame,
        int patch_size,
        cudaStream_t stream = 0
    );
    
    cudaError_t temporal_denoising_kernel(
        const PtrStepSz<uchar3>& current_frame,
        const PtrStepSz<uchar3>& reference_frame,
        PtrStepSz<uchar3>& denoised_frame,
        float noise_strength,
        cudaStream_t stream = 0
    );

private:
    bool m_initialized = false;
    int m_device_id = 0;
    
    // CUDA resources
    std::vector<cudaStream_t> m_streams;
    cufftHandle m_fft_plan;
    
    // Texture and surface references for optimized memory access
    cudaArray_t m_temp_arrays[4];
    cudaTextureObject_t m_textures[4];
    cudaSurfaceObject_t m_surfaces[4];
    
    bool initialize_cuda_resources();
    void cleanup_cuda_resources();
};

/**
 * @brief High-Performance Frame Memory Pool with Zero-Copy Operations
 */
class FrameMemoryPool {
public:
    explicit FrameMemoryPool(size_t pool_size, const cv::Size& frame_size);
    ~FrameMemoryPool();
    
    struct Frame {
        cv::Mat cpu_mat;
        cv::cuda::GpuMat gpu_mat;
        bool is_gpu_valid = false;
        bool is_cpu_valid = false;
        std::atomic<bool> in_use{false};
        uint64_t timestamp = 0;
    };
    
    std::shared_ptr<Frame> acquire_frame();
    void release_frame(std::shared_ptr<Frame> frame);
    
    // Zero-copy operations
    bool upload_to_gpu(Frame& frame, cudaStream_t stream = 0);
    bool download_from_gpu(Frame& frame, cudaStream_t stream = 0);
    
    // Statistics
    size_t get_pool_size() const { return m_pool_size; }
    size_t get_available_frames() const;
    size_t get_peak_usage() const { return m_peak_usage; }

private:
    std::vector<std::shared_ptr<Frame>> m_frame_pool;
    std::queue<std::shared_ptr<Frame>> m_available_frames;
    std::mutex m_pool_mutex;
    
    size_t m_pool_size;
    cv::Size m_frame_size;
    std::atomic<size_t> m_peak_usage{0};
    
    // CUDA pinned memory for zero-copy transfers
    void* m_pinned_memory = nullptr;
    size_t m_pinned_memory_size = 0;
    
    bool allocate_pinned_memory();
    void deallocate_pinned_memory();
};

/**
 * @brief Lock-Free Multi-Producer Multi-Consumer Queue for High Throughput
 */
template<typename T>
class LockFreeQueue {
public:
    explicit LockFreeQueue(size_t capacity);
    ~LockFreeQueue();
    
    bool enqueue(T&& item);
    bool dequeue(T& item);
    
    size_t size() const;
    bool empty() const;
    bool full() const;

private:
    struct Node {
        std::atomic<T*> data{nullptr};
        std::atomic<Node*> next{nullptr};
    };
    
    std::atomic<Node*> m_head{nullptr};
    std::atomic<Node*> m_tail{nullptr};
    std::atomic<size_t> m_size{0};
    size_t m_capacity;
    
    // Memory pool for nodes to avoid allocation overhead
    std::vector<Node> m_node_pool;
    std::atomic<size_t> m_node_index{0};
    
    Node* allocate_node();
    void deallocate_node(Node* node);
};

/**
 * @brief Advanced Streaming Processor for Large Files
 */
class StreamingProcessor {
public:
    explicit StreamingProcessor(const PerformanceConfig& config);
    ~StreamingProcessor();
    
    struct StreamingContext {
        std::string input_file;
        std::string output_file;
        RepairStrategy strategy;
        
        // Streaming parameters
        size_t chunk_size_frames = 32;        // Process in chunks
        size_t overlap_frames = 4;            // Overlap for temporal consistency
        bool enable_progressive_loading = true; // Load data progressively
        
        // Callbacks
        std::function<void(double)> progress_callback;
        std::function<void(const std::string&)> log_callback;
    };
    
    bool process_video_stream(const StreamingContext& context);
    
    // Streaming statistics
    struct StreamingStats {
        size_t total_chunks_processed = 0;
        size_t bytes_processed = 0;
        double average_chunk_time_ms = 0.0;
        double throughput_mbps = 0.0;
    };
    
    StreamingStats get_streaming_stats() const;

private:
    PerformanceConfig m_config;
    
    // Thread pools for different stages
    class ThreadPool;
    std::unique_ptr<ThreadPool> m_decode_pool;
    std::unique_ptr<ThreadPool> m_process_pool;
    std::unique_ptr<ThreadPool> m_encode_pool;
    
    // Lock-free queues between stages
    std::unique_ptr<LockFreeQueue<std::shared_ptr<FrameMemoryPool::Frame>>> m_decode_queue;
    std::unique_ptr<LockFreeQueue<std::shared_ptr<FrameMemoryPool::Frame>>> m_process_queue;
    std::unique_ptr<LockFreeQueue<std::shared_ptr<FrameMemoryPool::Frame>>> m_encode_queue;
    
    // Streaming statistics
    mutable std::mutex m_stats_mutex;
    StreamingStats m_stats;
    
    // Pipeline stages
    void decode_worker(const StreamingContext& context);
    void process_worker(const StreamingContext& context);
    void encode_worker(const StreamingContext& context);
    
    bool setup_streaming_pipeline(const StreamingContext& context);
    void cleanup_streaming_pipeline();
};

/**
 * @brief SIMD-Optimized Processor for CPU-based Operations
 */
class SIMDOptimizedProcessor {
public:
    SIMDOptimizedProcessor();
    ~SIMDOptimizedProcessor();
    
    // SIMD-optimized implementations
    void motion_estimation_avx2(
        const uint8_t* frame1, const uint8_t* frame2,
        int width, int height, int stride,
        float* motion_vectors_x, float* motion_vectors_y,
        int block_size, int search_range
    );
    
    void temporal_interpolation_avx2(
        const uint8_t* prev_frame, const uint8_t* next_frame,
        uint8_t* interpolated_frame,
        int width, int height, int stride,
        float temporal_position
    );
    
    void corruption_detection_avx2(
        const uint8_t* frame_data,
        uint8_t* corruption_mask,
        int width, int height, int stride,
        float threshold
    );
    
    void denoising_avx2(
        const uint8_t* noisy_frame,
        uint8_t* denoised_frame,
        int width, int height, int stride,
        float noise_sigma
    );

private:
    bool m_avx2_available = false;
    bool m_avx512_available = false;
    
    void detect_cpu_features();
};

/**
 * @brief Advanced Motion Vector Cache with LRU Policy
 */
class MotionVectorCache {
public:
    explicit MotionVectorCache(size_t cache_size_mb);
    ~MotionVectorCache();
    
    struct MotionData {
        cv::Mat motion_vectors;
        int64_t timestamp;
        uint64_t frame_hash;
        double confidence_score;
    };
    
    bool get_motion_vectors(uint64_t frame_pair_hash, MotionData& motion_data);
    void cache_motion_vectors(uint64_t frame_pair_hash, const MotionData& motion_data);
    
    // Cache statistics
    double get_hit_ratio() const;
    size_t get_cache_size_bytes() const;
    void clear_cache();

private:
    struct CacheEntry {
        MotionData data;
        int64_t last_access_time;
        std::shared_ptr<CacheEntry> prev;
        std::shared_ptr<CacheEntry> next;
    };
    
    std::unordered_map<uint64_t, std::shared_ptr<CacheEntry>> m_cache_map;
    std::shared_ptr<CacheEntry> m_lru_head;
    std::shared_ptr<CacheEntry> m_lru_tail;
    
    size_t m_max_cache_size_bytes;
    std::atomic<size_t> m_current_cache_size{0};
    std::atomic<size_t> m_cache_hits{0};
    std::atomic<size_t> m_cache_misses{0};
    
    mutable std::shared_mutex m_cache_mutex;
    
    void evict_lru_entries(size_t bytes_needed);
    void move_to_front(std::shared_ptr<CacheEntry> entry);
    uint64_t calculate_frame_hash(const cv::Mat& frame);
};

} // namespace AdvancedVideoRepair

#endif // ENHANCED_PERFORMANCE_ENGINE_H