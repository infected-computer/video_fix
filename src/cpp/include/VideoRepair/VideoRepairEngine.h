#ifndef VIDEO_REPAIR_ENGINE_H
#define VIDEO_REPAIR_ENGINE_H

#include <memory>
#include <vector>
#include <unordered_map>
#include <functional>
#include <atomic>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <queue>

// FFmpeg headers
extern "C" {
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libavutil/avutil.h>
#include <libavutil/imgutils.h>
#include <libavfilter/avfilter.h>
#include <libavfilter/buffersink.h>
#include <libavfilter/buffersrc.h>
#include <libswscale/swscale.h>
}

// OpenCV with CUDA
#include <opencv2/opencv.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudafeatures2d.hpp>

// CUDA Runtime
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include <cublas_v2.h>
#include <cudnn.h>

// Internal headers
#include "Core/MemoryManager.h"
#include "Core/ErrorHandling.h"
#include "VideoRepair/FormatParsers.h"
#include "VideoRepair/RepairAlgorithms.h"

/**
 * @brief Core video repair engine with hybrid C++/Python architecture
 * 
 * This is the main processing engine that handles:
 * - Multi-format video parsing and analysis
 * - GPU-accelerated frame processing
 * - Professional codec support (ProRes, BRAW, R3D, etc.)
 * - Real-time repair algorithms
 * - Memory-optimized streaming processing
 */
class VideoRepairEngine {
public:
    // Forward declarations
    struct RepairContext;
    struct StreamInfo;
    struct RepairParameters;
    struct RepairResult;
    class GPUProcessor;
    class FormatParser;
    class RepairAlgorithm;

    // Repair status enumeration
    enum class RepairStatus {
        PENDING,
        ANALYZING,
        PROCESSING,
        REPAIRING,
        FINALIZING,
        COMPLETED,
        FAILED,
        CANCELLED
    };

    // Supported video formats with priority levels
    enum class VideoFormat {
        // Professional formats (highest priority)
        PRORES_422,
        PRORES_422_HQ,
        PRORES_422_LT,
        PRORES_422_PROXY,
        PRORES_4444,
        PRORES_4444_XQ,
        PRORES_RAW,
        BLACKMAGIC_RAW,
        RED_R3D,
        ARRI_RAW,
        SONY_XAVC,
        CANON_CRM,
        MXF_OP1A,
        MXF_OP_ATOM,
        
        // Broadcast formats
        AVCHD,
        XDCAM_HD,
        
        // Consumer formats
        MP4_H264,
        MP4_H265,
        MOV_H264,
        MOV_H265,
        AVI_DV,
        AVI_MJPEG,
        MKV_H264,
        MKV_H265,
        
        // Legacy formats
        DV,
        HDV,
        
        UNKNOWN
    };

    // Repair techniques available
    enum class RepairTechnique {
        HEADER_RECONSTRUCTION,
        INDEX_REBUILD,
        FRAGMENT_RECOVERY,
        CONTAINER_REMUX,
        FRAME_INTERPOLATION,
        AI_INPAINTING,
        SUPER_RESOLUTION,
        DENOISING,
        METADATA_RECOVERY
    };

    // GPU processing capabilities
    struct GPUCapabilities {
        bool cuda_available = false;
        bool opencv_cuda_available = false;
        bool tensorrt_available = false;
        bool cudnn_available = false;
        int device_count = 0;
        size_t total_memory = 0;
        size_t free_memory = 0;
        int compute_capability_major = 0;
        int compute_capability_minor = 0;
        std::string device_name;
        std::vector<int> available_devices;
    };

    // Stream information structure
    struct StreamInfo {
        int stream_index = -1;
        AVMediaType media_type = AVMEDIA_TYPE_UNKNOWN;
        AVCodecID codec_id = AV_CODEC_ID_NONE;
        VideoFormat detected_format = VideoFormat::UNKNOWN;
        
        // Video specific
        int width = 0;
        int height = 0;
        AVRational frame_rate = {0, 1};
        AVRational time_base = {0, 1};
        AVPixelFormat pixel_format = AV_PIX_FMT_NONE;
        int bit_depth = 8;
        AVColorSpace color_space = AVCOL_SPC_UNSPECIFIED;
        AVColorPrimaries color_primaries = AVCOL_PRI_UNSPECIFIED;
        AVColorTransferCharacteristic color_trc = AVCOL_TRC_UNSPECIFIED;
        
        // Audio specific
        int sample_rate = 0;
        int channels = 0;
        AVSampleFormat sample_format = AV_SAMPLE_FMT_NONE;
        
        // Professional metadata
        std::string timecode;
        std::string camera_model;
        std::string lens_model;
        std::unordered_map<std::string, std::string> metadata;
        
        // Corruption indicators
        bool has_corruption = false;
        std::vector<std::pair<int64_t, int64_t>> corrupted_ranges;
        double corruption_percentage = 0.0;
        
        // Quality metrics
        double psnr = 0.0;
        double ssim = 0.0;
        double vmaf = 0.0;
    };

    // Repair parameters configuration
    struct RepairParameters {
        // Input/Output
        std::string input_file;
        std::string output_file;
        std::string reference_file; // For header reconstruction
        
        // Processing options
        std::vector<RepairTechnique> techniques;
        bool use_gpu = true;
        int gpu_device_id = 0;
        bool enable_ai_processing = true;
        bool preserve_original_quality = true;
        
        // Performance tuning
        int max_threads = 0; // 0 = auto-detect
        size_t memory_limit_mb = 0; // 0 = no limit
        int processing_queue_size = 4;
        bool enable_hardware_decoding = true;
        bool enable_hardware_encoding = true;
        
        // Quality settings
        bool maintain_bit_depth = true;
        bool maintain_color_space = true;
        bool maintain_frame_rate = true;
        double quality_factor = 1.0; // 0.0-1.0, higher = better quality
        
        // AI-specific settings
        double ai_strength = 0.8; // 0.0-1.0
        bool mark_ai_regions = true;
        std::string ai_model_path;
        
        // Progress callback
        std::function<void(double progress, const std::string& status)> progress_callback;
        std::function<void(const std::string& log_message)> log_callback;
    };

    // Comprehensive repair result
    struct RepairResult {
        bool success = false;
        RepairStatus final_status = RepairStatus::PENDING;
        std::string error_message;
        
        // Processing statistics
        double processing_time_seconds = 0.0;
        double gpu_utilization_average = 0.0;
        size_t memory_peak_usage_mb = 0;
        int frames_processed = 0;
        int frames_repaired = 0;
        
        // Quality metrics (before/after)
        struct QualityMetrics {
            double psnr_before = 0.0;
            double psnr_after = 0.0;
            double ssim_before = 0.0;
            double ssim_after = 0.0;
            double vmaf_before = 0.0;
            double vmaf_after = 0.0;
        } quality_metrics;
        
        // Repair details
        std::vector<RepairTechnique> techniques_applied;
        std::unordered_map<std::string, std::string> repair_details;
        
        // AI processing results
        struct AIResults {
            bool ai_processing_used = false;
            int frames_ai_processed = 0;
            std::vector<cv::Rect> ai_processed_regions;
            double ai_confidence_average = 0.0;
        } ai_results;
        
        // Output file information
        StreamInfo output_stream_info;
        std::string output_file_path;
        int64_t output_file_size = 0;
        
        // Warnings and recommendations
        std::vector<std::string> warnings;
        std::vector<std::string> recommendations;
    };

private:
    // Internal repair context for thread-safe operations
    struct RepairContext {
        std::string session_id;
        RepairParameters parameters;
        RepairResult result;
        
        // FFmpeg contexts
        AVFormatContext* input_format_ctx = nullptr;
        AVFormatContext* output_format_ctx = nullptr;
        std::vector<AVCodecContext*> decoder_contexts;
        std::vector<AVCodecContext*> encoder_contexts;
        
        // OpenCV/CUDA contexts
        cv::cuda::GpuMat gpu_frame_buffer;
        cv::cuda::Stream cuda_stream;
        std::unique_ptr<GPUProcessor> gpu_processor;
        
        // Processing state
        std::atomic<RepairStatus> current_status{RepairStatus::PENDING};
        std::atomic<double> progress{0.0};
        std::atomic<bool> should_cancel{false};
        std::mutex context_mutex;
        
        // Memory management
        std::unique_ptr<MemoryManager> memory_manager;
        
        // Format-specific parsers
        std::unordered_map<VideoFormat, std::unique_ptr<FormatParser>> format_parsers;
        
        // Repair algorithms
        std::vector<std::unique_ptr<RepairAlgorithm>> repair_algorithms;
        
        ~RepairContext() { cleanup(); }
        void cleanup();
    };

public:
    // Constructor/Destructor
    explicit VideoRepairEngine();
    ~VideoRepairEngine();

    // Core initialization
    bool initialize();
    void shutdown();
    bool is_initialized() const { return m_initialized; }

    // GPU capabilities
    GPUCapabilities get_gpu_capabilities() const;
    bool set_gpu_device(int device_id);
    int get_current_gpu_device() const { return m_current_gpu_device; }

    // Format support detection
    bool is_format_supported(const std::string& file_path) const;
    VideoFormat detect_video_format(const std::string& file_path) const;
    std::vector<VideoFormat> get_supported_formats() const;

    // File analysis (non-destructive)
    StreamInfo analyze_file(const std::string& file_path) const;
    std::vector<RepairTechnique> recommend_repair_techniques(const StreamInfo& stream_info) const;
    double estimate_repair_time(const StreamInfo& stream_info, 
                               const std::vector<RepairTechnique>& techniques) const;

    // Repair operations
    std::string start_repair_async(const RepairParameters& parameters);
    RepairResult repair_file_sync(const RepairParameters& parameters);
    
    // Session management
    RepairStatus get_repair_status(const std::string& session_id) const;
    double get_repair_progress(const std::string& session_id) const;
    bool cancel_repair(const std::string& session_id);
    RepairResult get_repair_result(const std::string& session_id) const;
    void cleanup_session(const std::string& session_id);

    // Batch processing
    struct BatchJob {
        std::string job_id;
        std::vector<std::string> input_files;
        std::string output_directory;
        RepairParameters base_parameters;
        std::function<void(const std::string& file, const RepairResult& result)> completion_callback;
    };
    
    std::string start_batch_repair(const BatchJob& batch_job);
    std::vector<RepairResult> get_batch_results(const std::string& job_id) const;

    // Configuration and tuning
    void set_memory_limit(size_t limit_mb);
    void set_thread_count(int thread_count);
    void enable_debug_output(bool enable);
    void set_log_level(int level);

    // Performance monitoring
    struct PerformanceMetrics {
        double cpu_usage_percent = 0.0;
        double gpu_usage_percent = 0.0;
        size_t memory_usage_mb = 0;
        size_t gpu_memory_usage_mb = 0;
        int active_sessions = 0;
        double average_processing_fps = 0.0;
    };
    
    PerformanceMetrics get_performance_metrics() const;
    void reset_performance_counters();

private:
    // Core members
    bool m_initialized = false;
    int m_current_gpu_device = 0;
    GPUCapabilities m_gpu_capabilities;
    
    // Threading and synchronization
    std::vector<std::thread> m_worker_threads;
    std::mutex m_sessions_mutex;
    std::condition_variable m_work_available;
    std::atomic<bool> m_shutdown_requested{false};
    
    // Session management
    std::unordered_map<std::string, std::unique_ptr<RepairContext>> m_active_sessions;
    std::queue<std::string> m_processing_queue;
    
    // Global configuration
    size_t m_memory_limit_mb = 0;
    int m_thread_count = 0;
    bool m_debug_output_enabled = false;
    int m_log_level = 2; // 0=error, 1=warning, 2=info, 3=debug
    
    // Performance monitoring
    mutable std::mutex m_metrics_mutex;
    PerformanceMetrics m_current_metrics;
    std::chrono::steady_clock::time_point m_metrics_last_update;
    
    // Private methods
    bool initialize_ffmpeg();
    bool initialize_opencv_cuda();
    bool initialize_gpu_resources();
    void initialize_format_parsers();
    void initialize_worker_threads();
    
    std::string generate_session_id() const;
    std::unique_ptr<RepairContext> create_repair_context(const RepairParameters& parameters);
    
    void worker_thread_function();
    void process_repair_session(RepairContext* context);
    
    // Core repair pipeline
    bool analyze_input_file(RepairContext* context);
    bool setup_processing_pipeline(RepairContext* context);
    bool execute_repair_algorithms(RepairContext* context);
    bool finalize_output(RepairContext* context);
    
    void update_progress(RepairContext* context, double progress, const std::string& status);
    void log_message(RepairContext* context, const std::string& message, int level = 2);
    
    void update_performance_metrics();
    
    // Utility methods
    void cleanup_all_sessions();
    bool validate_repair_parameters(const RepairParameters& parameters) const;
    VideoFormat detect_format_from_header(const std::string& file_path) const;
    VideoFormat detect_format_from_extension(const std::string& file_path) const;
};

/**
 * @brief GPU-accelerated processing component
 */
class VideoRepairEngine::GPUProcessor {
public:
    explicit GPUProcessor(int device_id);
    ~GPUProcessor();

    bool initialize();
    void shutdown();

    // Frame processing operations
    bool upload_frame(const AVFrame* cpu_frame, cv::cuda::GpuMat& gpu_frame);
    bool download_frame(const cv::cuda::GpuMat& gpu_frame, AVFrame* cpu_frame);
    
    // GPU-accelerated repairs
    bool denoise_frame(cv::cuda::GpuMat& frame, float noise_sigma);
    bool enhance_frame(cv::cuda::GpuMat& frame, float enhancement_factor);
    bool interpolate_missing_regions(cv::cuda::GpuMat& frame, const cv::cuda::GpuMat& mask);
    bool super_resolve_frame(const cv::cuda::GpuMat& input, cv::cuda::GpuMat& output, int scale_factor);
    
    // Quality assessment
    double calculate_psnr_gpu(const cv::cuda::GpuMat& img1, const cv::cuda::GpuMat& img2);
    double calculate_ssim_gpu(const cv::cuda::GpuMat& img1, const cv::cuda::GpuMat& img2);
    
    // Memory management
    size_t get_available_memory() const;
    void clear_cache();

private:
    int m_device_id;
    bool m_initialized = false;
    
    // CUDA contexts
    cudaStream_t m_cuda_stream;
    cublasHandle_t m_cublas_handle;
    cudnnHandle_t m_cudnn_handle;
    
    // OpenCV CUDA contexts
    cv::cuda::Stream m_opencv_stream;
    
    // Memory pools
    std::vector<cv::cuda::GpuMat> m_frame_buffer_pool;
    std::mutex m_buffer_pool_mutex;
    
    bool initialize_cuda_context();
    bool initialize_cublas();
    bool initialize_cudnn();
    void cleanup_resources();
};

/**
 * @brief Base class for format-specific parsers
 */
class VideoRepairEngine::FormatParser {
public:
    explicit FormatParser(VideoFormat format) : m_format(format) {}
    virtual ~FormatParser() = default;

    VideoFormat get_format() const { return m_format; }
    
    virtual bool can_parse(const std::string& file_path) const = 0;
    virtual StreamInfo parse_stream_info(AVFormatContext* format_ctx, int stream_index) const = 0;
    virtual bool detect_corruption(AVFormatContext* format_ctx, StreamInfo& stream_info) const = 0;
    virtual std::vector<RepairTechnique> recommend_techniques(const StreamInfo& stream_info) const = 0;
    
protected:
    VideoFormat m_format;
};

/**
 * @brief Base class for repair algorithms
 */
class VideoRepairEngine::RepairAlgorithm {
public:
    explicit RepairAlgorithm(RepairTechnique technique) : m_technique(technique) {}
    virtual ~RepairAlgorithm() = default;

    RepairTechnique get_technique() const { return m_technique; }
    
    virtual bool can_repair(const StreamInfo& stream_info) const = 0;
    virtual bool apply_repair(RepairContext* context) = 0;
    virtual double estimate_processing_time(const StreamInfo& stream_info) const = 0;
    
protected:
    RepairTechnique m_technique;
};

// Global utility functions
namespace VideoRepairUtils {
    std::string format_to_string(VideoRepairEngine::VideoFormat format);
    std::string technique_to_string(VideoRepairEngine::RepairTechnique technique);
    std::string status_to_string(VideoRepairEngine::RepairStatus status);
    
    bool is_professional_format(VideoRepairEngine::VideoFormat format);
    bool requires_reference_file(VideoRepairEngine::RepairTechnique technique);
    
    AVPixelFormat get_optimal_pixel_format(VideoRepairEngine::VideoFormat format, int bit_depth);
    AVCodecID get_codec_for_format(VideoRepairEngine::VideoFormat format);
}

#endif // VIDEO_REPAIR_ENGINE_H