#ifndef ADVANCED_VIDEO_REPAIR_ENGINE_H
#define ADVANCED_VIDEO_REPAIR_ENGINE_H

#include <memory>
#include <vector>
#include <unordered_map>
#include <functional>
#include <atomic>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <queue>
#include <fstream>
#include <chrono>

// FFmpeg headers for professional video processing
extern "C" {
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libavutil/avutil.h>
#include <libavutil/imgutils.h>
#include <libavfilter/avfilter.h>
#include <libavfilter/buffersink.h>
#include <libavfilter/buffersrc.h>
#include <libswscale/swscale.h>
#include <libavutil/opt.h>
#include <libavutil/mathematics.h>
#include <libavutil/timestamp.h>
}

// OpenCV for advanced image processing
#include <opencv2/opencv.hpp>
#ifdef HAVE_CUDA
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudafeatures2d.hpp>
#endif

// RAII wrappers for FFmpeg
#include "FFmpegUtils.h"

// Thread-safe utilities
#include "ThreadSafeFrameBuffer.h"

/**
 * @brief Advanced Video Repair Engine
 * 
 * A sophisticated video repair system that actually implements advanced algorithms:
 * - Container structure analysis and reconstruction
 * - Frame-level corruption detection and repair
 * - Temporal interpolation for missing frames
 * - Header reconstruction with format-specific knowledge
 * - Bitstream analysis and correction
 * - Motion-compensated error concealment
 */
namespace AdvancedVideoRepair {

// Forward declarations
class ContainerAnalyzer;
class BitstreamAnalyzer;
class FrameReconstructor;
class MotionEstimator;
class TemporalInterpolator;

/**
 * @brief Video container formats with detailed parsing capabilities
 */
enum class ContainerFormat {
    MP4_ISOBMFF,        // ISO Base Media File Format
    MOV_QUICKTIME,      // QuickTime Movie
    AVI_RIFF,          // Audio Video Interleave
    MKV_MATROSKA,      // Matroska Video
    MXF_SMPTE,         // Material Exchange Format
    TS_MPEG,           // MPEG Transport Stream
    M2TS_BLURAY,       // Blu-ray MPEG-2 Transport Stream
    UNKNOWN
};

/**
 * @brief Video codecs with specific repair strategies
 */
enum class VideoCodec {
    H264_AVC,          // Advanced Video Coding
    H265_HEVC,         // High Efficiency Video Coding
    VP9_GOOGLE,        // VP9 by Google
    AV1_AOMedia,       // AV1 by Alliance for Open Media
    PRORES_APPLE,      // Apple ProRes family
    DNX_AVID,          // Avid DNxHD/DNxHR
    CINEFORM_GOPRO,    // GoPro CineForm
    MJPEG_MOTION,      // Motion JPEG
    DV_DIGITAL,        // Digital Video
    UNKNOWN_CODEC
};

/**
 * @brief Corruption types that can be detected and repaired
 */
enum class CorruptionType {
    CONTAINER_STRUCTURE,    // Corrupted container metadata
    BITSTREAM_ERRORS,      // Corrupted video bitstream
    MISSING_FRAMES,        // Missing or damaged frames
    SYNC_LOSS,            // Audio/video synchronization issues
    INDEX_CORRUPTION,      // Corrupted seek tables/indices
    HEADER_DAMAGE,        // Damaged file headers
    INCOMPLETE_FRAMES,     // Partially written frames
    TEMPORAL_ARTIFACTS    // Temporal inconsistencies
};

/**
 * @brief Repair strategy configuration
 */
struct RepairStrategy {
    bool use_reference_frames = true;     // Use neighboring frames for reconstruction
    bool enable_motion_compensation = true; // Use motion vectors for interpolation
    bool preserve_original_quality = true;  // Maintain original compression level
    bool use_temporal_analysis = true;     // Analyze temporal patterns
    double error_concealment_strength = 0.8; // 0.0-1.0, strength of error concealment
    int max_interpolation_distance = 5;    // Maximum frames for interpolation
    bool enable_post_processing = true;    // Apply post-processing filters
};

/**
 * @brief Detailed corruption analysis result
 */
struct CorruptionAnalysis {
    std::vector<CorruptionType> detected_issues;
    std::vector<std::pair<int64_t, int64_t>> corrupted_byte_ranges;
    std::vector<int> corrupted_frame_numbers;
    double overall_corruption_percentage = 0.0;
    bool is_repairable = false;
    std::string detailed_report;
    
    // Container-specific analysis
    struct ContainerIssues {
        bool missing_moov_atom = false;
        bool corrupted_mdat_atom = false;
        bool invalid_chunk_offsets = false;
        bool missing_index_data = false;
        std::vector<std::string> missing_required_boxes;
    } container_issues;
    
    // Bitstream-specific analysis
    struct BitstreamIssues {
        int corrupted_macroblocks = 0;
        int missing_reference_frames = 0;
        bool corrupted_sps_pps = false;
        std::vector<int> frames_with_errors;
    } bitstream_issues;
};

/**
 * @brief Advanced repair result with comprehensive metrics
 */
struct AdvancedRepairResult {
    bool success = false;
    std::string input_file;
    std::string output_file;
    
    // Processing statistics
    std::chrono::milliseconds processing_time{0};
    size_t bytes_repaired = 0;
    int frames_reconstructed = 0;
    int frames_interpolated = 0;
    
    // Quality metrics
    struct QualityMetrics {
        double psnr_improvement = 0.0;
        double ssim_improvement = 0.0;
        double temporal_consistency_score = 0.0;
        double artifact_reduction_percentage = 0.0;
    } quality_metrics;
    
    // Repair details
    std::vector<std::string> repairs_performed;
    std::vector<std::string> warnings;
    CorruptionAnalysis original_analysis;
    
    // Validation results
    bool output_playable = false;
    bool audio_sync_maintained = false;
    std::string validation_report;
};

/**
 * @brief Main Advanced Video Repair Engine class
 */
class AdvancedVideoRepairEngine {
public:
    explicit AdvancedVideoRepairEngine();
    ~AdvancedVideoRepairEngine();
    
    // Initialization and configuration
    bool initialize();
    void shutdown();
    bool is_initialized() const { return m_initialized; }
    
    // Main repair interface
    AdvancedRepairResult repair_video_file(
        const std::string& input_file,
        const std::string& output_file,
        const RepairStrategy& strategy = RepairStrategy{}
    );
    
    // Analysis capabilities
    CorruptionAnalysis analyze_corruption(const std::string& file_path);
    bool can_repair_file(const std::string& file_path);
    std::vector<RepairStrategy> suggest_repair_strategies(const CorruptionAnalysis& analysis);
    
    // Format detection and support
    ContainerFormat detect_container_format(const std::string& file_path);
    VideoCodec detect_video_codec(const std::string& file_path);
    bool is_format_supported(ContainerFormat format, VideoCodec codec);
    
    // Configuration
    void set_thread_count(int threads) { m_thread_count = threads; }
    void enable_gpu_acceleration(bool enable) { m_gpu_enabled = enable; }
    void set_memory_limit_mb(size_t limit_mb) { m_memory_limit_mb = limit_mb; }
    void set_log_level(int level) { m_log_level = level; }
    
    // Progress monitoring
    void set_progress_callback(std::function<void(double, const std::string&)> callback) {
        m_progress_callback = callback;
    }

private:
    // Core components
    std::unique_ptr<ContainerAnalyzer> m_container_analyzer;
    std::unique_ptr<BitstreamAnalyzer> m_bitstream_analyzer;
    std::unique_ptr<FrameReconstructor> m_frame_reconstructor;
    std::unique_ptr<MotionEstimator> m_motion_estimator;
    std::unique_ptr<TemporalInterpolator> m_temporal_interpolator;
    
    // FFmpeg contexts using RAII wrappers
    VideoRepair::AVFormatContextPtr m_input_format_ctx;
    VideoRepair::AVFormatContextPtr m_output_format_ctx;
    std::vector<VideoRepair::AVCodecContextPtr> m_decoder_contexts;
    std::vector<VideoRepair::AVCodecContextPtr> m_encoder_contexts;
    
    // State management
    bool m_initialized = false;
    int m_thread_count = 0;
    bool m_gpu_enabled = false;
    size_t m_memory_limit_mb = 4096;
    int m_log_level = 2;
    
    std::function<void(double, const std::string&)> m_progress_callback;
    
    // Internal repair methods
    bool initialize_ffmpeg();
    bool setup_input_context(const std::string& input_file);
    bool setup_output_context(const std::string& output_file, const RepairStrategy& strategy);
    
    AdvancedRepairResult perform_container_repair(
        const std::string& input_file,
        const std::string& output_file,
        const CorruptionAnalysis& analysis,
        const RepairStrategy& strategy
    );
    
    AdvancedRepairResult perform_bitstream_repair(
        const std::string& input_file,
        const std::string& output_file,
        const CorruptionAnalysis& analysis,
        const RepairStrategy& strategy
    );
    
    bool validate_repair_result(const std::string& output_file, AdvancedRepairResult& result);
    
    void update_progress(double progress, const std::string& status);
    void log_message(const std::string& message, int level = 2);
    
    void cleanup_contexts();
    
    // Internal analysis methods
    bool validate_file_header(const std::string& file_path);
    bool check_stream_integrity(const std::string& file_path, CorruptionAnalysis& analysis);
    double calculate_corruption_percentage(const CorruptionAnalysis& analysis);
    bool determine_repairability(const CorruptionAnalysis& analysis);
    std::string generate_analysis_report(const CorruptionAnalysis& analysis);
};

/**
 * @brief Advanced container analyzer that understands format internals
 */
class ContainerAnalyzer {
public:
    explicit ContainerAnalyzer(AdvancedVideoRepairEngine* engine);
    ~ContainerAnalyzer();
    
    CorruptionAnalysis analyze_mp4_structure(const std::string& file_path);
    CorruptionAnalysis analyze_avi_structure(const std::string& file_path);
    CorruptionAnalysis analyze_mkv_structure(const std::string& file_path);
    
    bool repair_mp4_container(
        const std::string& input_file,
        const std::string& output_file,
        const CorruptionAnalysis& analysis
    );
    
    bool repair_avi_container(
        const std::string& input_file,
        const std::string& output_file,
        const CorruptionAnalysis& analysis
    );

private:
    AdvancedVideoRepairEngine* m_engine;
    
    // MP4/MOV specific methods
    struct MP4Box {
        uint32_t size;
        std::string type;
        int64_t offset;
        std::vector<uint8_t> data;
    };
    
    std::vector<MP4Box> parse_mp4_boxes(const std::string& file_path);
    bool reconstruct_moov_atom(const std::vector<MP4Box>& boxes, std::vector<uint8_t>& moov_data);
    bool rebuild_stco_offsets(std::vector<MP4Box>& boxes);
    
    // AVI specific methods
    struct AVIChunk {
        std::string fourcc;
        uint32_t size;
        int64_t offset;
        std::vector<uint8_t> data;
    };
    
    std::vector<AVIChunk> parse_avi_chunks(const std::string& file_path);
    bool reconstruct_avi_index(const std::vector<AVIChunk>& chunks, std::vector<uint8_t>& idx1_data);
};

/**
 * @brief Bitstream analyzer for codec-specific corruption detection
 */
class BitstreamAnalyzer {
public:
    explicit BitstreamAnalyzer(AdvancedVideoRepairEngine* engine);
    ~BitstreamAnalyzer();
    
    CorruptionAnalysis analyze_h264_bitstream(AVCodecContext* codec_ctx, AVPacket* packet);
    CorruptionAnalysis analyze_h265_bitstream(AVCodecContext* codec_ctx, AVPacket* packet);
    
    bool repair_h264_packet(AVPacket* packet, const CorruptionAnalysis& analysis);
    bool repair_h265_packet(AVPacket* packet, const CorruptionAnalysis& analysis);

private:
    AdvancedVideoRepairEngine* m_engine;
    
    // H.264 specific analysis
    struct H264NALUnit {
        uint8_t nal_type;
        std::vector<uint8_t> data;
        bool is_corrupted = false;
    };
    
    std::vector<H264NALUnit> parse_h264_nals(const uint8_t* data, int size);
    bool validate_h264_sps(const H264NALUnit& nal);
    bool validate_h264_pps(const H264NALUnit& nal);
    bool repair_h264_slice(H264NALUnit& nal);
};

/**
 * @brief Advanced frame reconstructor using temporal and spatial techniques
 */
class FrameReconstructor {
public:
    explicit FrameReconstructor(AdvancedVideoRepairEngine* engine);
    ~FrameReconstructor();
    
    bool reconstruct_missing_frame(
        const std::vector<cv::Mat>& reference_frames,
        cv::Mat& output_frame,
        int target_frame_number,
        const RepairStrategy& strategy
    );
    
    // New thread-safe version using ThreadSafeFrameBuffer
    bool reconstruct_missing_frame_safe(
        const VideoRepair::ThreadSafeFrameBuffer& frame_buffer,
        cv::Mat& output_frame,
        int target_frame_number,
        const RepairStrategy& strategy
    );
    
    bool repair_corrupted_regions(
        cv::Mat& frame,
        const cv::Mat& corruption_mask,
        const std::vector<cv::Mat>& reference_frames,
        const RepairStrategy& strategy
    );
    
    // New thread-safe version for corrupted regions
    bool repair_corrupted_regions_safe(
        cv::Mat& frame,
        const cv::Mat& corruption_mask,
        const VideoRepair::ThreadSafeFrameBuffer& frame_buffer,
        const RepairStrategy& strategy
    );

private:
    AdvancedVideoRepairEngine* m_engine;
    std::unique_ptr<MotionEstimator> m_motion_estimator;
    
    // Thread-safe frame buffer for concurrent operations
    mutable std::mutex reconstruction_mutex_;
    std::atomic<size_t> active_reconstructions_{0};
    
    bool perform_temporal_interpolation(
        const cv::Mat& prev_frame,
        const cv::Mat& next_frame,
        cv::Mat& interpolated_frame,
        double temporal_position
    );
    
    bool perform_spatial_inpainting(
        cv::Mat& frame,
        const cv::Mat& mask
    );
    
    bool apply_motion_compensation(
        const cv::Mat& reference_frame,
        cv::Mat& target_frame,
        const std::vector<cv::Point2f>& motion_vectors
    );
    
    // Additional interpolation methods
    bool perform_optical_flow_interpolation(
        const cv::Mat& prev_frame,
        const cv::Mat& next_frame,
        cv::Mat& interpolated_frame,
        double temporal_position
    );
    
    bool perform_feature_based_interpolation(
        const cv::Mat& prev_frame,
        const cv::Mat& next_frame,
        cv::Mat& interpolated_frame,
        double temporal_position
    );
    
    cv::Mat blend_reconstruction_results(
        const cv::Mat& result1,
        const cv::Mat& result2,
        double weight1,
        double weight2
    );
};

/**
 * @brief Motion estimator for advanced temporal processing
 */
class MotionEstimator {
public:
    explicit MotionEstimator();
    ~MotionEstimator();
    
    std::vector<cv::Point2f> estimate_motion_vectors(
        const cv::Mat& frame1,
        const cv::Mat& frame2,
        int block_size = 16
    );
    
    cv::Mat create_motion_compensated_frame(
        const cv::Mat& reference_frame,
        const std::vector<cv::Point2f>& motion_vectors,
        int block_size = 16
    );

private:
    cv::Ptr<cv::cuda::FarnebackOpticalFlow> m_optical_flow;
    bool m_gpu_available = false;
};

} // namespace AdvancedVideoRepair

#endif // ADVANCED_VIDEO_REPAIR_ENGINE_H