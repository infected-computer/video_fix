#include "AdvancedVideoRepair/AdvancedVideoRepairEngine.h"
#include "AdvancedVideoRepair/FFmpegUtils.h"
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
    // Allocate format context using RAII wrapper
    AVFormatContext* ctx = avformat_alloc_context();
    if (!ctx) {
        log_message("Failed to allocate input format context", 0);
        return false;
    }
    
    // Set options for robust parsing using RAII wrapper
    VideoRepair::AVDictionaryPtr opts;
    av_dict_set(opts.get_ptr(), "scan_all_pmts", "1", 0);
    av_dict_set(opts.get_ptr(), "analyzeduration", "10000000", 0); // 10 seconds
    av_dict_set(opts.get_ptr(), "probesize", "10000000", 0);       // 10MB probe
    
    // Open input file
    int ret = avformat_open_input(&ctx, input_file.c_str(), nullptr, opts.get_ptr());
    
    // Move ownership to RAII wrapper
    m_input_format_ctx.reset(ctx);
    
    if (ret < 0) {
        char error_buf[AV_ERROR_MAX_STRING_SIZE];
        av_strerror(ret, error_buf, sizeof(error_buf));
        log_message("Failed to open input file: " + std::string(error_buf), 0);
        return false;
    }
    
    // Find stream information with extended probing for corrupted files
    ret = avformat_find_stream_info(m_input_format_ctx.get(), nullptr);
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
            
            VideoRepair::AVCodecContextPtr codec_ctx(avcodec_alloc_context3(codec));
            if (!codec_ctx) {
                log_message("Failed to allocate codec context for stream " + std::to_string(i), 1);
                continue;
            }
            
            ret = avcodec_parameters_to_context(codec_ctx.get(), stream->codecpar);
            if (ret < 0) {
                continue;
            }
            
            // Set options for error resilience
            codec_ctx->error_concealment = FF_EC_GUESS_MVS | FF_EC_DEBLOCK;
            codec_ctx->err_recognition = AV_EF_CRCCHECK | AV_EF_BITSTREAM | AV_EF_CAREFUL;
            
            ret = avcodec_open2(codec_ctx.get(), codec, nullptr);
            if (ret < 0) {
                log_message("Failed to open codec for stream " + std::to_string(i), 1);
                continue;
            }
            
            m_decoder_contexts.push_back(std::move(codec_ctx));
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
    
    // Allocate output format context using RAII wrapper
    AVFormatContext* output_ctx = nullptr;
    int ret = avformat_alloc_output_context2(&output_ctx, nullptr, format_name, output_file.c_str());
    if (ret < 0) {
        char error_buf[AV_ERROR_MAX_STRING_SIZE];
        av_strerror(ret, error_buf, sizeof(error_buf));
        log_message("Failed to allocate output context: " + std::string(error_buf), 0);
        return false;
    }
    
    // Move ownership to RAII wrapper
    m_output_format_ctx.reset(output_ctx);
    
    // Create output streams based on input streams
    for (unsigned int i = 0; i < m_input_format_ctx->nb_streams; i++) {
        AVStream* input_stream = m_input_format_ctx->streams[i];
        
        if (input_stream->codecpar->codec_type == AVMEDIA_TYPE_VIDEO ||
            input_stream->codecpar->codec_type == AVMEDIA_TYPE_AUDIO) {
            
            AVStream* output_stream = avformat_new_stream(m_output_format_ctx.get(), nullptr);
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
    // Clean up decoder contexts - RAII wrappers handle automatic cleanup
    m_decoder_contexts.clear();
    
    // Clean up encoder contexts - RAII wrappers handle automatic cleanup
    m_encoder_contexts.clear();
    
    // Clean up format contexts - RAII wrappers handle automatic cleanup
    if (m_output_format_ctx && m_output_format_ctx->pb) {
        avio_closep(&m_output_format_ctx->pb);
    }
    
    // Reset RAII wrappers - they will automatically call proper cleanup
    m_input_format_ctx.reset();
    m_output_format_ctx.reset();
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

/**
 * @brief Comprehensive corruption analysis for video files
 * 
 * Implements detailed analysis as specified in the requirements:
 * 1. Check file header integrity
 * 2. Parse container structure 
 * 3. Validate essential atoms/boxes
 * 4. Check stream integrity
 */
CorruptionAnalysis AdvancedVideoRepairEngine::analyze_corruption(const std::string& file_path) {
    CorruptionAnalysis result;
    
    try {
        log_message("Starting corruption analysis for: " + file_path, 2);
        
        // 1. Check file header
        if (!validate_file_header(file_path)) {
            result.detected_issues.push_back(CorruptionType::HEADER_DAMAGE);
            log_message("Header corruption detected", 1);
        }
        
        // 2. Parse container structure
        ContainerFormat format = detect_container_format(file_path);
        
        if (format == ContainerFormat::MP4_ISOBMFF || format == ContainerFormat::MOV_QUICKTIME) {
            // MP4/MOV specific analysis
            CorruptionAnalysis mp4_analysis = m_container_analyzer->analyze_mp4_structure(file_path);
            
            // Merge analysis results
            result.detected_issues.insert(result.detected_issues.end(), 
                                        mp4_analysis.detected_issues.begin(), 
                                        mp4_analysis.detected_issues.end());
            result.container_issues = mp4_analysis.container_issues;
            
        } else if (format == ContainerFormat::AVI_RIFF) {
            // AVI specific analysis
            CorruptionAnalysis avi_analysis = m_container_analyzer->analyze_avi_structure(file_path);
            
            result.detected_issues.insert(result.detected_issues.end(), 
                                        avi_analysis.detected_issues.begin(), 
                                        avi_analysis.detected_issues.end());
            
        } else if (format == ContainerFormat::MKV_MATROSKA) {
            // MKV specific analysis
            CorruptionAnalysis mkv_analysis = m_container_analyzer->analyze_mkv_structure(file_path);
            
            result.detected_issues.insert(result.detected_issues.end(), 
                                        mkv_analysis.detected_issues.begin(), 
                                        mkv_analysis.detected_issues.end());
        }
        
        // 3. Validate essential atoms/boxes (format-specific)
        if (format == ContainerFormat::MP4_ISOBMFF || format == ContainerFormat::MOV_QUICKTIME) {
            // Check for missing moov atom
            if (result.container_issues.missing_moov_atom) {
                result.detected_issues.push_back(CorruptionType::MISSING_FRAMES);
                log_message("Missing moov atom detected", 1);
            }
            
            // Check for corrupted mdat atom
            if (result.container_issues.corrupted_mdat_atom) {
                result.detected_issues.push_back(CorruptionType::BITSTREAM_ERRORS);
                log_message("Corrupted mdat atom detected", 1);
            }
            
            // Check index corruption
            if (result.container_issues.invalid_chunk_offsets) {
                result.detected_issues.push_back(CorruptionType::INDEX_CORRUPTION);
                log_message("Invalid chunk offsets detected", 1);
            }
        }
        
        // 4. Check stream integrity using FFmpeg
        bool stream_integrity_ok = check_stream_integrity(file_path, result);
        if (!stream_integrity_ok) {
            result.detected_issues.push_back(CorruptionType::BITSTREAM_ERRORS);
            log_message("Stream integrity issues detected", 1);
        }
        
        // Calculate overall corruption percentage
        result.overall_corruption_percentage = calculate_corruption_percentage(result);
        
        // Determine if file is repairable
        result.is_repairable = determine_repairability(result);
        
        // Generate detailed report
        result.detailed_report = generate_analysis_report(result);
        
        log_message("Corruption analysis complete. Issues found: " + 
                   std::to_string(result.detected_issues.size()), 2);
        
        return result;
        
    } catch (const std::exception& e) {
        log_message("Error during corruption analysis: " + std::string(e.what()), 0);
        result.is_repairable = false;
        result.detailed_report = "Analysis failed: " + std::string(e.what());
        return result;
    }
}

/**
 * @brief Validate file header integrity
 */
bool AdvancedVideoRepairEngine::validate_file_header(const std::string& file_path) {
    std::ifstream file(file_path, std::ios::binary);
    if (!file.is_open()) {
        return false;
    }
    
    // Read first 32 bytes for header analysis
    char header[32];
    file.read(header, sizeof(header));
    
    if (file.gcount() < 8) {
        return false; // File too small
    }
    
    // Check for known video file signatures
    
    // MP4/MOV - starts with ftyp box
    if (header[4] == 'f' && header[5] == 't' && header[6] == 'y' && header[7] == 'p') {
        return true;
    }
    
    // AVI - starts with RIFF
    if (header[0] == 'R' && header[1] == 'I' && header[2] == 'F' && header[3] == 'F') {
        // Check for AVI signature at offset 8
        if (header[8] == 'A' && header[9] == 'V' && header[10] == 'I' && header[11] == ' ') {
            return true;
        }
    }
    
    // MKV - EBML signature
    if (header[0] == 0x1A && header[1] == 0x45 && header[2] == 0xDF && header[3] == 0xA3) {
        return true;
    }
    
    // If no known signature found, consider header corrupted
    return false;
}

/**
 * @brief Check stream integrity using FFmpeg
 */
bool AdvancedVideoRepairEngine::check_stream_integrity(const std::string& file_path, CorruptionAnalysis& analysis) {
    // Try to open the file with FFmpeg
    VideoRepair::AVFormatContextPtr format_ctx;
    AVFormatContext* ctx = avformat_alloc_context();
    if (!ctx) {
        return false;
    }
    
    format_ctx.reset(ctx);
    
    // Set options for maximum error tolerance
    VideoRepair::AVDictionaryPtr opts;
    av_dict_set(opts.get_ptr(), "fflags", "+ignidx+igndts+genpts", 0);
    av_dict_set(opts.get_ptr(), "analyzeduration", "10000000", 0);
    av_dict_set(opts.get_ptr(), "probesize", "10000000", 0);
    
    int ret = avformat_open_input(format_ctx.get_ptr(), file_path.c_str(), nullptr, opts.get_ptr());
    if (ret < 0) {
        // Severe corruption - cannot even open
        analysis.bitstream_issues.corrupted_sps_pps = true;
        return false;
    }
    
    // Try to find stream info
    ret = avformat_find_stream_info(format_ctx.get(), nullptr);
    if (ret < 0) {
        // Stream info problems
        analysis.bitstream_issues.missing_reference_frames++;
        return false;
    }
    
    // Check each stream
    bool has_issues = false;
    for (unsigned int i = 0; i < format_ctx->nb_streams; i++) {
        AVStream* stream = format_ctx->streams[i];
        
        if (stream->codecpar->codec_type == AVMEDIA_TYPE_VIDEO) {
            // Try to create decoder for video stream
            const AVCodec* codec = avcodec_find_decoder(stream->codecpar->codec_id);
            if (!codec) {
                has_issues = true;
                continue;
            }
            
            VideoRepair::AVCodecContextPtr codec_ctx(avcodec_alloc_context3(codec));
            if (!codec_ctx) {
                has_issues = true;
                continue;
            }
            
            ret = avcodec_parameters_to_context(codec_ctx.get(), stream->codecpar);
            if (ret < 0) {
                has_issues = true;
                continue;
            }
            
            ret = avcodec_open2(codec_ctx.get(), codec, nullptr);
            if (ret < 0) {
                has_issues = true;
                analysis.bitstream_issues.corrupted_sps_pps = true;
            }
        }
    }
    
    return !has_issues;
}

/**
 * @brief Calculate overall corruption percentage
 */
double AdvancedVideoRepairEngine::calculate_corruption_percentage(const CorruptionAnalysis& analysis) {
    if (analysis.detected_issues.empty()) {
        return 0.0;
    }
    
    // Weight different corruption types
    double total_weight = 0.0;
    double corruption_weight = 0.0;
    
    for (const auto& issue : analysis.detected_issues) {
        switch (issue) {
            case CorruptionType::HEADER_DAMAGE:
                total_weight += 20.0;
                corruption_weight += 20.0;
                break;
            case CorruptionType::CONTAINER_STRUCTURE:
                total_weight += 30.0;
                corruption_weight += 30.0;
                break;
            case CorruptionType::BITSTREAM_ERRORS:
                total_weight += 25.0;
                corruption_weight += 25.0;
                break;
            case CorruptionType::MISSING_FRAMES:
                total_weight += 15.0;
                corruption_weight += 15.0;
                break;
            case CorruptionType::INDEX_CORRUPTION:
                total_weight += 10.0;
                corruption_weight += 10.0;
                break;
            default:
                total_weight += 5.0;
                corruption_weight += 5.0;
                break;
        }
    }
    
    return total_weight > 0 ? (corruption_weight / 100.0) * 100.0 : 0.0;
}

/**
 * @brief Determine if file is repairable based on analysis
 */
bool AdvancedVideoRepairEngine::determine_repairability(const CorruptionAnalysis& analysis) {
    // File is considered unrepairable if:
    // 1. Corruption percentage is too high (>80%)
    // 2. Critical structures are completely missing
    // 3. Both header and container structure are corrupted
    
    if (analysis.overall_corruption_percentage > 80.0) {
        return false;
    }
    
    bool has_header_damage = false;
    bool has_container_damage = false;
    
    for (const auto& issue : analysis.detected_issues) {
        if (issue == CorruptionType::HEADER_DAMAGE) {
            has_header_damage = true;
        }
        if (issue == CorruptionType::CONTAINER_STRUCTURE) {
            has_container_damage = true;
        }
    }
    
    // If both header and container are damaged, very difficult to repair
    if (has_header_damage && has_container_damage) {
        return false;
    }
    
    return true;
}

/**
 * @brief Generate detailed analysis report
 */
std::string AdvancedVideoRepairEngine::generate_analysis_report(const CorruptionAnalysis& analysis) {
    std::string report = "=== Video Corruption Analysis Report ===\n\n";
    
    report += "Overall Corruption: " + std::to_string(analysis.overall_corruption_percentage) + "%\n";
    report += "Repairable: " + std::string(analysis.is_repairable ? "Yes" : "No") + "\n\n";
    
    if (!analysis.detected_issues.empty()) {
        report += "Detected Issues:\n";
        for (const auto& issue : analysis.detected_issues) {
            switch (issue) {
                case CorruptionType::HEADER_DAMAGE:
                    report += "- File header corruption\n";
                    break;
                case CorruptionType::CONTAINER_STRUCTURE:
                    report += "- Container structure damage\n";
                    break;
                case CorruptionType::BITSTREAM_ERRORS:
                    report += "- Video bitstream errors\n";
                    break;
                case CorruptionType::MISSING_FRAMES:
                    report += "- Missing or damaged frames\n";
                    break;
                case CorruptionType::INDEX_CORRUPTION:
                    report += "- Index/seek table corruption\n";
                    break;
                case CorruptionType::SYNC_LOSS:
                    report += "- Audio/video sync issues\n";
                    break;
                case CorruptionType::INCOMPLETE_FRAMES:
                    report += "- Incomplete frame data\n";
                    break;
                case CorruptionType::TEMPORAL_ARTIFACTS:
                    report += "- Temporal inconsistencies\n";
                    break;
            }
        }
    }
    
    // Container-specific issues
    if (analysis.container_issues.missing_moov_atom) {
        report += "\nContainer Issues:\n";
        report += "- Missing moov atom (MP4/MOV)\n";
    }
    if (analysis.container_issues.corrupted_mdat_atom) {
        report += "- Corrupted mdat atom\n";
    }
    if (analysis.container_issues.invalid_chunk_offsets) {
        report += "- Invalid chunk offset table\n";
    }
    
    // Bitstream issues
    if (analysis.bitstream_issues.corrupted_sps_pps) {
        report += "\nBitstream Issues:\n";
        report += "- Corrupted SPS/PPS headers\n";
    }
    if (analysis.bitstream_issues.missing_reference_frames > 0) {
        report += "- Missing reference frames: " + 
                 std::to_string(analysis.bitstream_issues.missing_reference_frames) + "\n";
    }
    
    return report;
}

/**
 * @brief Detect container format from file header
 */
ContainerFormat AdvancedVideoRepairEngine::detect_container_format(const std::string& file_path) {
    std::ifstream file(file_path, std::ios::binary);
    if (!file.is_open()) {
        return ContainerFormat::UNKNOWN;
    }
    
    // Read first 32 bytes for format detection
    char header[32];
    file.read(header, sizeof(header));
    
    if (file.gcount() < 12) {
        return ContainerFormat::UNKNOWN;
    }
    
    // MP4/MOV detection - look for ftyp box
    if (header[4] == 'f' && header[5] == 't' && header[6] == 'y' && header[7] == 'p') {
        // Check brand to distinguish MP4 from MOV
        std::string brand(header + 8, 4);
        if (brand == "isom" || brand == "mp41" || brand == "mp42" || brand == "avc1") {
            return ContainerFormat::MP4_ISOBMFF;
        } else if (brand == "qt  ") {
            return ContainerFormat::MOV_QUICKTIME;
        } else {
            return ContainerFormat::MP4_ISOBMFF; // Default to MP4
        }
    }
    
    // AVI detection - RIFF header with AVI signature
    if (header[0] == 'R' && header[1] == 'I' && header[2] == 'F' && header[3] == 'F') {
        if (header[8] == 'A' && header[9] == 'V' && header[10] == 'I' && header[11] == ' ') {
            return ContainerFormat::AVI_RIFF;
        }
    }
    
    // MKV detection - EBML signature
    if (header[0] == 0x1A && header[1] == 0x45 && header[2] == 0xDF && header[3] == 0xA3) {
        return ContainerFormat::MKV_MATROSKA;
    }
    
    // MPEG-TS detection - sync byte pattern
    if (header[0] == 0x47) {
        // Check for additional sync bytes at 188-byte intervals
        file.seekg(188, std::ios::beg);
        char sync_byte;
        file.read(&sync_byte, 1);
        if (sync_byte == 0x47) {
            return ContainerFormat::TS_MPEG;
        }
    }
    
    // M2TS detection (Blu-ray)
    if (header[0] == 0x47 && header[4] == 0x47) {
        // M2TS has 4-byte timestamp prefix
        return ContainerFormat::M2TS_BLURAY;
    }
    
    // MXF detection - KLV signature
    if (header[0] == 0x06 && header[1] == 0x0E && header[2] == 0x2B && header[3] == 0x34) {
        return ContainerFormat::MXF_SMPTE;
    }
    
    return ContainerFormat::UNKNOWN;
}

/**
 * @brief Detect video codec from file
 */
VideoCodec AdvancedVideoRepairEngine::detect_video_codec(const std::string& file_path) {
    // Use FFmpeg to detect codec
    VideoRepair::AVFormatContextPtr format_ctx;
    AVFormatContext* ctx = avformat_alloc_context();
    if (!ctx) {
        return VideoCodec::UNKNOWN_CODEC;
    }
    
    format_ctx.reset(ctx);
    
    int ret = avformat_open_input(format_ctx.get_ptr(), file_path.c_str(), nullptr, nullptr);
    if (ret < 0) {
        return VideoCodec::UNKNOWN_CODEC;
    }
    
    ret = avformat_find_stream_info(format_ctx.get(), nullptr);
    if (ret < 0) {
        return VideoCodec::UNKNOWN_CODEC;
    }
    
    // Find first video stream
    for (unsigned int i = 0; i < format_ctx->nb_streams; i++) {
        if (format_ctx->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO) {
            AVCodecID codec_id = format_ctx->streams[i]->codecpar->codec_id;
            
            switch (codec_id) {
                case AV_CODEC_ID_H264:
                    return VideoCodec::H264_AVC;
                case AV_CODEC_ID_HEVC:
                    return VideoCodec::H265_HEVC;
                case AV_CODEC_ID_VP9:
                    return VideoCodec::VP9_GOOGLE;
                case AV_CODEC_ID_AV1:
                    return VideoCodec::AV1_AOMedia;
                case AV_CODEC_ID_PRORES:
                    return VideoCodec::PRORES_APPLE;
                case AV_CODEC_ID_DNXHD:
                    return VideoCodec::DNX_AVID;
                case AV_CODEC_ID_MJPEG:
                    return VideoCodec::MJPEG_MOTION;
                case AV_CODEC_ID_DVVIDEO:
                    return VideoCodec::DV_DIGITAL;
                default:
                    return VideoCodec::UNKNOWN_CODEC;
            }
        }
    }
    
    return VideoCodec::UNKNOWN_CODEC;
}

} // namespace AdvancedVideoRepair