#include "AdvancedVideoRepair/AdvancedVideoRepairEngine.h"
#include <iostream>
#include <fstream>
#include <chrono>
#include <algorithm>
#include <cstring>

namespace AdvancedVideoRepair {

AdvancedVideoRepairEngine::AdvancedVideoRepairEngine() 
    : m_initialized(false)
    , m_thread_count(std::thread::hardware_concurrency())
    , m_gpu_enabled(false)
    , m_memory_limit_mb(4096)
    , m_log_level(2) {
    
    // Initialize components
    m_container_analyzer = std::make_unique<ContainerAnalyzer>(this);
    m_bitstream_analyzer = std::make_unique<BitstreamAnalyzer>(this);
    m_frame_reconstructor = std::make_unique<FrameReconstructor>(this);
    m_motion_estimator = std::make_unique<MotionEstimator>();
    m_temporal_interpolator = std::make_unique<TemporalInterpolator>();
}

AdvancedVideoRepairEngine::~AdvancedVideoRepairEngine() {
    shutdown();
}

/**
 * @brief Initialize the advanced video repair engine
 * 
 * This properly initializes FFmpeg with error checking and optimal settings
 */
bool AdvancedVideoRepairEngine::initialize() {
    if (m_initialized) return true;
    
    try {
        log_message("Initializing Advanced Video Repair Engine...", 2);
        
        // Initialize FFmpeg libraries with proper error handling
        if (!initialize_ffmpeg()) {
            log_message("Failed to initialize FFmpeg libraries", 0);
            return false;
        }
        
        // Test GPU availability
        m_gpu_enabled = test_gpu_availability();
        if (m_gpu_enabled) {
            log_message("GPU acceleration available and enabled", 2);
        } else {
            log_message("GPU acceleration not available, using CPU", 1);
        }
        
        // Set optimal thread count based on system capabilities
        if (m_thread_count == 0) {
            m_thread_count = std::max(1u, std::thread::hardware_concurrency());
        }
        
        log_message("Engine initialized successfully with " + std::to_string(m_thread_count) + " threads", 2);
        m_initialized = true;
        return true;
        
    } catch (const std::exception& e) {
        log_message("Failed to initialize engine: " + std::string(e.what()), 0);
        return false;
    }
}

/**
 * @brief Main video repair function with comprehensive approach
 * 
 * This is the core repair logic that coordinates all subsystems:
 * 1. Deep file analysis to understand corruption
 * 2. Strategy selection based on corruption type
 * 3. Multi-pass repair with validation
 * 4. Quality assessment and refinement
 */
AdvancedRepairResult AdvancedVideoRepairEngine::repair_video_file(
    const std::string& input_file,
    const std::string& output_file,
    const RepairStrategy& strategy) {
    
    AdvancedRepairResult result;
    result.input_file = input_file;
    result.output_file = output_file;
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    if (!m_initialized) {
        result.error_message = "Engine not initialized";
        return result;
    }
    
    try {
        update_progress(0.0, "Starting video analysis...");
        
        // Phase 1: Comprehensive corruption analysis
        CorruptionAnalysis corruption_analysis = analyze_corruption(input_file);
        result.original_analysis = corruption_analysis;
        
        if (!corruption_analysis.is_repairable) {
            result.error_message = "File is too severely corrupted for repair";
            return result;
        }
        
        update_progress(0.2, "Analysis complete, planning repair strategy...");
        
        // Phase 2: Setup FFmpeg contexts
        if (!setup_input_context(input_file)) {
            result.error_message = "Failed to setup input context";
            cleanup_contexts();
            return result;
        }
        
        if (!setup_output_context(output_file, strategy)) {
            result.error_message = "Failed to setup output context";
            cleanup_contexts();
            return result;
        }
        
        update_progress(0.3, "Performing container-level repairs...");
        
        // Phase 3: Container-level repair
        if (requires_container_repair(corruption_analysis)) {
            bool container_success = perform_container_repair_internal(corruption_analysis, strategy);
            if (!container_success) {
                log_message("Container repair failed, attempting bitstream repair", 1);
            } else {
                result.repairs_performed.push_back("Container structure repaired");
            }
        }
        
        update_progress(0.5, "Performing frame-level repairs...");
        
        // Phase 4: Frame-level repair with temporal processing
        bool frame_repair_success = perform_frame_level_repair(corruption_analysis, strategy, result);
        if (!frame_repair_success) {
            result.warnings.push_back("Some frame-level repairs failed");
        }
        
        update_progress(0.8, "Finalizing and validating output...");
        
        // Phase 5: Output finalization and validation
        bool finalization_success = finalize_output_file(strategy, result);
        if (!finalization_success) {
            result.error_message = "Failed to finalize output file";
            cleanup_contexts();
            return result;
        }
        
        // Phase 6: Quality validation
        bool validation_success = validate_repair_result(output_file, result);
        result.success = validation_success;
        
        if (validation_success) {
            update_progress(1.0, "Repair completed successfully");
            result.validation_report = "Output file is valid and playable";
        } else {
            result.warnings.push_back("Output file may have quality issues");
        }
        
    } catch (const std::exception& e) {
        result.error_message = "Repair failed: " + std::string(e.what());
        log_message(result.error_message, 0);
    }
    
    // Calculate processing statistics
    auto end_time = std::chrono::high_resolution_clock::now();
    result.processing_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    
    cleanup_contexts();
    return result;
}

/**
 * @brief Setup FFmpeg input context with proper error handling
 */
bool AdvancedVideoRepairEngine::setup_input_context(const std::string& input_file) {
    // Allocate format context
    m_input_format_ctx = avformat_alloc_context();
    if (!m_input_format_ctx) {
        log_message("Failed to allocate input format context", 0);
        return false;
    }
    
    // Set options for robust parsing
    AVDictionary* opts = nullptr;
    av_dict_set(&opts, "scan_all_pmts", "1", 0);
    av_dict_set(&opts, "analyzeduration", "10000000", 0); // 10 seconds
    av_dict_set(&opts, "probesize", "10000000", 0);       // 10MB probe
    
    // Open input file
    int ret = avformat_open_input(&m_input_format_ctx, input_file.c_str(), nullptr, &opts);
    av_dict_free(&opts);
    
    if (ret < 0) {
        char error_buf[AV_ERROR_MAX_STRING_SIZE];
        av_strerror(ret, error_buf, sizeof(error_buf));
        log_message("Failed to open input file: " + std::string(error_buf), 0);
        return false;
    }
    
    // Find stream information with extended probing for corrupted files
    ret = avformat_find_stream_info(m_input_format_ctx, nullptr);
    if (ret < 0) {
        char error_buf[AV_ERROR_MAX_STRING_SIZE];
        av_strerror(ret, error_buf, sizeof(error_buf));
        log_message("Failed to find stream info: " + std::string(error_buf), 1);
        // Don't fail here - we might still be able to repair
    }
    
    // Setup decoder contexts for each stream
    for (unsigned int i = 0; i < m_input_format_ctx->nb_streams; i++) {
        AVStream* stream = m_input_format_ctx->streams[i];
        
        if (stream->codecpar->codec_type == AVMEDIA_TYPE_VIDEO ||
            stream->codecpar->codec_type == AVMEDIA_TYPE_AUDIO) {
            
            const AVCodec* codec = avcodec_find_decoder(stream->codecpar->codec_id);
            if (!codec) {
                log_message("Codec not found for stream " + std::to_string(i), 1);
                continue;
            }
            
            AVCodecContext* codec_ctx = avcodec_alloc_context3(codec);
            if (!codec_ctx) {
                log_message("Failed to allocate codec context for stream " + std::to_string(i), 1);
                continue;
            }
            
            ret = avcodec_parameters_to_context(codec_ctx, stream->codecpar);
            if (ret < 0) {
                avcodec_free_context(&codec_ctx);
                continue;
            }
            
            // Set options for error resilience
            codec_ctx->error_concealment = FF_EC_GUESS_MVS | FF_EC_DEBLOCK;
            codec_ctx->err_recognition = AV_EF_CRCCHECK | AV_EF_BITSTREAM | AV_EF_CAREFUL;
            
            ret = avcodec_open2(codec_ctx, codec, nullptr);
            if (ret < 0) {
                avcodec_free_context(&codec_ctx);
                log_message("Failed to open codec for stream " + std::to_string(i), 1);
                continue;
            }
            
            m_decoder_contexts.push_back(codec_ctx);
        }
    }
    
    log_message("Successfully setup input context with " + std::to_string(m_decoder_contexts.size()) + " decoder streams", 2);
    return true;
}

/**
 * @brief Setup FFmpeg output context with optimal settings
 */
bool AdvancedVideoRepairEngine::setup_output_context(const std::string& output_file, const RepairStrategy& strategy) {
    // Determine output format based on file extension
    const char* format_name = nullptr;
    std::string ext = output_file.substr(output_file.find_last_of('.'));
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
    
    if (ext == ".mp4") format_name = "mp4";
    else if (ext == ".mov") format_name = "mov";
    else if (ext == ".avi") format_name = "avi";
    else if (ext == ".mkv") format_name = "matroska";
    else format_name = "mp4"; // Default
    
    // Allocate output format context
    int ret = avformat_alloc_output_context2(&m_output_format_ctx, nullptr, format_name, output_file.c_str());
    if (ret < 0) {
        char error_buf[AV_ERROR_MAX_STRING_SIZE];
        av_strerror(ret, error_buf, sizeof(error_buf));
        log_message("Failed to allocate output context: " + std::string(error_buf), 0);
        return false;
    }
    
    // Create output streams based on input streams
    for (unsigned int i = 0; i < m_input_format_ctx->nb_streams; i++) {
        AVStream* input_stream = m_input_format_ctx->streams[i];
        
        if (input_stream->codecpar->codec_type == AVMEDIA_TYPE_VIDEO ||
            input_stream->codecpar->codec_type == AVMEDIA_TYPE_AUDIO) {
            
            AVStream* output_stream = avformat_new_stream(m_output_format_ctx, nullptr);
            if (!output_stream) {
                log_message("Failed to create output stream", 0);
                return false;
            }
            
            // Setup encoder based on repair strategy
            bool encoding_success = setup_encoder_for_stream(input_stream, output_stream, strategy);
            if (!encoding_success) {
                log_message("Failed to setup encoder for stream " + std::to_string(i), 1);
                continue;
            }
        }
    }
    
    // Open output file
    if (!(m_output_format_ctx->oformat->flags & AVFMT_NOFILE)) {
        ret = avio_open(&m_output_format_ctx->pb, output_file.c_str(), AVIO_FLAG_WRITE);
        if (ret < 0) {
            char error_buf[AV_ERROR_MAX_STRING_SIZE];
            av_strerror(ret, error_buf, sizeof(error_buf));
            log_message("Failed to open output file: " + std::string(error_buf), 0);
            return false;
        }
    }
    
    // Write file header
    ret = avformat_write_header(m_output_format_ctx, nullptr);
    if (ret < 0) {
        char error_buf[AV_ERROR_MAX_STRING_SIZE];
        av_strerror(ret, error_buf, sizeof(error_buf));
        log_message("Failed to write output header: " + std::string(error_buf), 0);
        return false;
    }
    
    return true;
}

/**
 * @brief Frame-level repair with temporal processing
 * 
 * This performs sophisticated frame-by-frame repair:
 * - Detects corrupted frames using multiple methods
 * - Reconstructs missing frames using temporal interpolation
 * - Repairs corrupted regions using spatial-temporal techniques
 * - Maintains temporal consistency across the sequence
 */
bool AdvancedVideoRepairEngine::perform_frame_level_repair(
    const CorruptionAnalysis& analysis,
    const RepairStrategy& strategy,
    AdvancedRepairResult& result) {
    
    if (m_decoder_contexts.empty()) {
        return false;
    }
    
    AVPacket* packet = av_packet_alloc();
    AVFrame* frame = av_frame_alloc();
    
    if (!packet || !frame) {
        av_packet_free(&packet);
        av_frame_free(&frame);
        return false;
    }
    
    std::vector<cv::Mat> frame_buffer; // Buffer for temporal processing
    const int buffer_size = strategy.max_interpolation_distance + 1;
    
    int frame_count = 0;
    int repaired_frames = 0;
    
    // Process video frames
    while (av_read_frame(m_input_format_ctx, packet) >= 0) {
        // Find the appropriate decoder
        AVCodecContext* decoder_ctx = nullptr;
        AVCodecContext* encoder_ctx = nullptr;
        
        for (size_t i = 0; i < m_decoder_contexts.size(); i++) {
            if (m_decoder_contexts[i]->streams[0] == packet->stream_index) {
                decoder_ctx = m_decoder_contexts[i];
                if (i < m_encoder_contexts.size()) {
                    encoder_ctx = m_encoder_contexts[i];
                }
                break;
            }
        }
        
        if (!decoder_ctx || packet->stream_index >= m_input_format_ctx->nb_streams) {
            av_packet_unref(packet);
            continue;
        }
        
        AVStream* input_stream = m_input_format_ctx->streams[packet->stream_index];
        
        if (input_stream->codecpar->codec_type == AVMEDIA_TYPE_VIDEO) {
            // Decode frame
            int ret = avcodec_send_packet(decoder_ctx, packet);
            if (ret < 0) {
                // Packet is corrupted, try to recover
                log_message("Corrupted packet detected at frame " + std::to_string(frame_count), 1);
                
                if (strategy.use_temporal_analysis && !frame_buffer.empty()) {
                    // Reconstruct missing frame using temporal interpolation
                    cv::Mat reconstructed_frame;
                    bool reconstruction_success = m_frame_reconstructor->reconstruct_missing_frame(
                        frame_buffer, reconstructed_frame, frame_count, strategy);
                    
                    if (reconstruction_success) {
                        // Convert back to AVFrame and encode
                        AVFrame* reconstructed_av_frame = convert_cv_to_avframe(reconstructed_frame, decoder_ctx);
                        if (reconstructed_av_frame) {
                            encode_and_write_frame(reconstructed_av_frame, encoder_ctx, packet->stream_index);
                            av_frame_free(&reconstructed_av_frame);
                            repaired_frames++;
                            result.repairs_performed.push_back("Reconstructed frame " + std::to_string(frame_count));
                        }
                    }
                }
                
                av_packet_unref(packet);
                frame_count++;
                continue;
            }
            
            while (ret >= 0) {
                ret = avcodec_receive_frame(decoder_ctx, frame);
                if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
                    break;
                } else if (ret < 0) {
                    log_message("Error during frame decoding", 1);
                    break;
                }
                
                // Convert AVFrame to OpenCV Mat for processing
                cv::Mat cv_frame = convert_avframe_to_cv(frame);
                if (cv_frame.empty()) {
                    av_frame_unref(frame);
                    continue;
                }
                
                // Detect frame corruption
                bool frame_corrupted = detect_frame_corruption(cv_frame, analysis);
                
                if (frame_corrupted && strategy.use_reference_frames && !frame_buffer.empty()) {
                    // Repair corrupted regions
                    cv::Mat corruption_mask = create_corruption_mask(cv_frame);
                    
                    bool repair_success = m_frame_reconstructor->repair_corrupted_regions(
                        cv_frame, corruption_mask, frame_buffer, strategy);
                    
                    if (repair_success) {
                        repaired_frames++;
                        result.repairs_performed.push_back("Repaired corruption in frame " + std::to_string(frame_count));
                    }
                }
                
                // Add to frame buffer for temporal processing
                manage_frame_buffer(frame_buffer, cv_frame, buffer_size);
                
                // Convert back to AVFrame and encode
                AVFrame* processed_frame = convert_cv_to_avframe(cv_frame, decoder_ctx);
                if (processed_frame) {
                    processed_frame->pts = frame->pts;
                    processed_frame->pkt_dts = frame->pkt_dts;
                    
                    if (encoder_ctx) {
                        encode_and_write_frame(processed_frame, encoder_ctx, packet->stream_index);
                    }
                    
                    av_frame_free(&processed_frame);
                }
                
                av_frame_unref(frame);
                frame_count++;
                
                // Update progress
                if (frame_count % 30 == 0) { // Update every ~1 second at 30fps
                    double progress = 0.5 + 0.3 * (double)frame_count / estimate_total_frames();
                    update_progress(progress, "Processing frame " + std::to_string(frame_count));
                }
            }
        } else {
            // Audio stream - pass through with basic error checking
            if (encoder_ctx) {
                // Re-encode audio if necessary for format compatibility
                process_audio_packet(packet, decoder_ctx, encoder_ctx);
            }
        }
        
        av_packet_unref(packet);
    }
    
    // Flush encoders
    for (auto* encoder_ctx : m_encoder_contexts) {
        flush_encoder(encoder_ctx);
    }
    
    result.frames_processed = frame_count;
    result.frames_reconstructed = repaired_frames;
    
    av_packet_free(&packet);
    av_frame_free(&frame);
    
    log_message("Frame processing complete: " + std::to_string(frame_count) + 
               " frames processed, " + std::to_string(repaired_frames) + " frames repaired", 2);
    
    return true;
}

/**
 * @brief Detect frame corruption using multiple techniques
 */
bool AdvancedVideoRepairEngine::detect_frame_corruption(const cv::Mat& frame, const CorruptionAnalysis& analysis) {
    if (frame.empty()) return true;
    
    // Method 1: Statistical analysis
    cv::Scalar mean, stddev;
    cv::meanStdDev(frame, mean, stddev);
    
    // Unusually low standard deviation indicates flat/corrupted regions
    double avg_stddev = (stddev[0] + stddev[1] + stddev[2]) / 3.0;
    if (avg_stddev < 5.0) return true;
    
    // Method 2: Edge density analysis
    cv::Mat gray, edges;
    cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
    cv::Canny(gray, edges, 50, 150);
    
    double edge_density = cv::sum(edges)[0] / (edges.rows * edges.cols * 255.0);
    if (edge_density < 0.01) return true; // Too few edges
    
    // Method 3: Block artifact detection
    if (detect_block_artifacts(frame)) return true;
    
    // Method 4: Color distribution analysis
    if (detect_color_anomalies(frame)) return true;
    
    return false;
}

/**
 * @brief Finalize output file and write trailer
 */
bool AdvancedVideoRepairEngine::finalize_output_file(const RepairStrategy& strategy, AdvancedRepairResult& result) {
    if (!m_output_format_ctx) return false;
    
    // Write file trailer
    int ret = av_write_trailer(m_output_format_ctx);
    if (ret < 0) {
        char error_buf[AV_ERROR_MAX_STRING_SIZE];
        av_strerror(ret, error_buf, sizeof(error_buf));
        log_message("Failed to write trailer: " + std::string(error_buf), 1);
        return false;
    }
    
    // Calculate file statistics
    if (m_output_format_ctx->pb) {
        result.output_file_size = avio_size(m_output_format_ctx->pb);
    }
    
    return true;
}

/**
 * @brief Comprehensive validation of repair result
 */
bool AdvancedVideoRepairEngine::validate_repair_result(const std::string& output_file, AdvancedRepairResult& result) {
    // Basic file validation
    std::ifstream file(output_file, std::ios::binary);
    if (!file.is_open()) {
        result.validation_report = "Output file cannot be opened";
        return false;
    }
    
    file.seekg(0, std::ios::end);
    size_t file_size = file.tellg();
    file.close();
    
    if (file_size < 1024) { // Suspiciously small
        result.validation_report = "Output file is too small";
        return false;
    }
    
    // Try to open with FFmpeg for deeper validation
    AVFormatContext* test_ctx = nullptr;
    int ret = avformat_open_input(&test_ctx, output_file.c_str(), nullptr, nullptr);
    
    if (ret < 0) {
        result.validation_report = "Output file has invalid format";
        return false;
    }
    
    ret = avformat_find_stream_info(test_ctx, nullptr);
    if (ret < 0) {
        avformat_close_input(&test_ctx);
        result.validation_report = "Output file has corrupted stream info";
        return false;
    }
    
    // Check if we have video streams
    bool has_video = false;
    for (unsigned int i = 0; i < test_ctx->nb_streams; i++) {
        if (test_ctx->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO) {
            has_video = true;
            break;
        }
    }
    
    result.output_playable = has_video;
    result.validation_report = has_video ? "File appears valid and playable" : "File valid but no video streams";
    
    avformat_close_input(&test_ctx);
    return true;
}

// Utility methods implementation
void AdvancedVideoRepairEngine::update_progress(double progress, const std::string& status) {
    if (m_progress_callback) {
        m_progress_callback(progress, status);
    }
}

void AdvancedVideoRepairEngine::log_message(const std::string& message, int level) {
    if (level <= m_log_level) {
        std::string level_str;
        switch (level) {
            case 0: level_str = "[ERROR] "; break;
            case 1: level_str = "[WARN]  "; break;
            case 2: level_str = "[INFO]  "; break;
            case 3: level_str = "[DEBUG] "; break;
            default: level_str = "[UNKNOWN] "; break;
        }
        
        std::cout << level_str << message << std::endl;
    }
}

void AdvancedVideoRepairEngine::cleanup_contexts() {
    // Clean up decoder contexts
    for (auto* ctx : m_decoder_contexts) {
        if (ctx) {
            avcodec_free_context(&ctx);
        }
    }
    m_decoder_contexts.clear();
    
    // Clean up encoder contexts
    for (auto* ctx : m_encoder_contexts) {
        if (ctx) {
            avcodec_free_context(&ctx);
        }
    }
    m_encoder_contexts.clear();
    
    // Clean up format contexts
    if (m_input_format_ctx) {
        avformat_close_input(&m_input_format_ctx);
        m_input_format_ctx = nullptr;
    }
    
    if (m_output_format_ctx) {
        if (m_output_format_ctx->pb) {
            avio_closep(&m_output_format_ctx->pb);
        }
        avformat_free_context(m_output_format_ctx);
        m_output_format_ctx = nullptr;
    }
}

bool AdvancedVideoRepairEngine::initialize_ffmpeg() {
    // Initialize FFmpeg libraries
    av_log_set_level(AV_LOG_WARNING); // Reduce FFmpeg verbosity
    
    // Register all codecs and formats
    avformat_network_init();
    
    log_message("FFmpeg initialized successfully", 2);
    return true;
}

void AdvancedVideoRepairEngine::shutdown() {
    if (m_initialized) {
        cleanup_contexts();
        avformat_network_deinit();
        m_initialized = false;
        log_message("Advanced Video Repair Engine shut down", 2);
    }
}

} // namespace AdvancedVideoRepair