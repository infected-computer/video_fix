#include "VideoRepair/FormatParsers/ProResParser.h"
#include "Core/ErrorHandling.h"
#include <algorithm>
#include <iomanip>
#include <sstream>
#include <cstring>

extern "C" {
#include <libavutil/timecode.h>
#include <libavutil/dict.h>
}

// Static constant definitions
const std::unordered_map<uint32_t, ProResParser::ProResVariant> ProResParser::FOURCC_TO_VARIANT = {
    {0x6F637061, ProResVariant::PROXY},      // 'apco'
    {0x73637061, ProResVariant::LT},         // 'apcs'
    {0x6E637061, ProResVariant::STANDARD},   // 'apcn'
    {0x68637061, ProResVariant::HQ},         // 'apch'
    {0x68347061, ProResVariant::PRORES_4444}, // 'ap4h'
    {0x78347061, ProResVariant::PRORES_4444_XQ}, // 'ap4x'
    {0x6E727061, ProResVariant::RAW},        // 'aprn'
    {0x68727061, ProResVariant::RAW_HQ}      // 'aprh'
};

const std::unordered_map<ProResParser::ProResVariant, std::string> ProResParser::VARIANT_NAMES = {
    {ProResVariant::PROXY, "ProRes 422 Proxy"},
    {ProResVariant::LT, "ProRes 422 LT"},
    {ProResVariant::STANDARD, "ProRes 422"},
    {ProResVariant::HQ, "ProRes 422 HQ"},
    {ProResVariant::PRORES_4444, "ProRes 4444"},
    {ProResVariant::PRORES_4444_XQ, "ProRes 4444 XQ"},
    {ProResVariant::RAW, "ProRes RAW"},
    {ProResVariant::RAW_HQ, "ProRes RAW HQ"}
};

const std::unordered_map<ProResParser::ProResVariant, size_t> ProResParser::VARIANT_BITRATES = {
    {ProResVariant::PROXY, 45000000},      // 45 Mbps for 1920x1080@25p
    {ProResVariant::LT, 102000000},        // 102 Mbps
    {ProResVariant::STANDARD, 147000000},   // 147 Mbps
    {ProResVariant::HQ, 220000000},        // 220 Mbps
    {ProResVariant::PRORES_4444, 330000000}, // 330 Mbps
    {ProResVariant::PRORES_4444_XQ, 500000000}, // 500 Mbps
    {ProResVariant::RAW, 1000000000},      // 1 Gbps (varies greatly)
    {ProResVariant::RAW_HQ, 2000000000}    // 2 Gbps (varies greatly)
};

const std::unordered_map<uint8_t, std::string> ProResParser::FRAMERATE_CODES = {
    {1, "23.976"}, {2, "24"}, {3, "25"}, {4, "29.97"}, {5, "30"},
    {6, "50"}, {7, "59.94"}, {8, "60"}, {9, "100"}, {10, "119.88"}, {11, "120"}
};

const std::unordered_map<uint8_t, std::string> ProResParser::CHROMA_FORMATS = {
    {2, "4:2:2"}, {3, "4:4:4"}
};

ProResParser::ProResParser() : VideoRepairEngine::FormatParser(VideoRepairEngine::VideoFormat::PRORES_422) {
    // Initialize parser configuration with professional defaults
    m_config.strict_validation = true;
    m_config.extract_all_metadata = true;
    m_config.deep_frame_analysis = false;
    m_config.corruption_threshold = 0.05;
    m_config.max_frames_to_analyze = 100;
}

bool ProResParser::can_parse(const std::string& file_path) const {
    AVFormatContext* format_ctx = nullptr;
    
    // Open file and check for ProRes streams
    if (avformat_open_input(&format_ctx, file_path.c_str(), nullptr, nullptr) < 0) {
        return false;
    }
    
    bool can_parse = false;
    if (avformat_find_stream_info(format_ctx, nullptr) >= 0) {
        for (unsigned int i = 0; i < format_ctx->nb_streams; i++) {
            AVStream* stream = format_ctx->streams[i];
            if (stream->codecpar->codec_type == AVMEDIA_TYPE_VIDEO) {
                // Check for ProRes codec IDs
                switch (stream->codecpar->codec_id) {
                    case AV_CODEC_ID_PRORES:
                        can_parse = true;
                        break;
                    default:
                        // Additional check by examining codec tag
                        if (stream->codecpar->codec_tag != 0) {
                            ProResVariant variant = identify_variant_from_fourcc(
                                reinterpret_cast<const uint8_t*>(&stream->codecpar->codec_tag));
                            if (variant != ProResVariant::UNKNOWN) {
                                can_parse = true;
                            }
                        }
                        break;
                }
                if (can_parse) break;
            }
        }
    }
    
    avformat_close_input(&format_ctx);
    return can_parse;
}

VideoRepairEngine::StreamInfo ProResParser::parse_stream_info(AVFormatContext* format_ctx, int stream_index) const {
    VideoRepairEngine::StreamInfo stream_info;
    
    if (!format_ctx || stream_index < 0 || stream_index >= static_cast<int>(format_ctx->nb_streams)) {
        return stream_info;
    }
    
    AVStream* stream = format_ctx->streams[stream_index];
    AVCodecParameters* codecpar = stream->codecpar;
    
    // Basic stream information
    stream_info.stream_index = stream_index;
    stream_info.media_type = codecpar->codec_type;
    stream_info.codec_id = codecpar->codec_id;
    
    if (codecpar->codec_type == AVMEDIA_TYPE_VIDEO) {
        stream_info.width = codecpar->width;
        stream_info.height = codecpar->height;
        stream_info.frame_rate = stream->avg_frame_rate;
        stream_info.time_base = stream->time_base;
        stream_info.pixel_format = static_cast<AVPixelFormat>(codecpar->format);
        stream_info.bit_depth = av_pix_fmt_desc_get(stream_info.pixel_format)->comp[0].depth;
        stream_info.color_space = codecpar->color_space;
        stream_info.color_primaries = codecpar->color_primaries;
        stream_info.color_trc = codecpar->color_trc;
        
        // Detect ProRes variant
        ProResVariant variant = detect_prores_variant(codecpar->extradata, codecpar->extradata_size);
        
        // Map variant to VideoFormat
        switch (variant) {
            case ProResVariant::PROXY:
                stream_info.detected_format = VideoRepairEngine::VideoFormat::PRORES_422_PROXY;
                break;
            case ProResVariant::LT:
                stream_info.detected_format = VideoRepairEngine::VideoFormat::PRORES_422_LT;
                break;
            case ProResVariant::STANDARD:
                stream_info.detected_format = VideoRepairEngine::VideoFormat::PRORES_422;
                break;
            case ProResVariant::HQ:
                stream_info.detected_format = VideoRepairEngine::VideoFormat::PRORES_422_HQ;
                break;
            case ProResVariant::PRORES_4444:
                stream_info.detected_format = VideoRepairEngine::VideoFormat::PRORES_4444;
                break;
            case ProResVariant::PRORES_4444_XQ:
                stream_info.detected_format = VideoRepairEngine::VideoFormat::PRORES_4444_XQ;
                break;
            case ProResVariant::RAW:
            case ProResVariant::RAW_HQ:
                stream_info.detected_format = VideoRepairEngine::VideoFormat::PRORES_RAW;
                break;
            default:
                stream_info.detected_format = VideoRepairEngine::VideoFormat::PRORES_422;
                break;
        }
        
        // Extract professional metadata
        extract_timecode_from_prores(format_ctx, stream_info.timecode);
        
        std::vector<ProResAtom> atoms = parse_professional_metadata(format_ctx);
        extract_camera_metadata(atoms, stream_info);
        extract_color_metadata(atoms, stream_info);
        
        // Populate metadata map with professional information
        stream_info.metadata["prores_variant"] = variant_to_string(variant);
        stream_info.metadata["expected_bitrate"] = std::to_string(
            get_expected_bitrate(variant, stream_info.width, stream_info.height, 
                               av_q2d(stream_info.frame_rate)));
        
        // Add professional format indicators
        stream_info.metadata["supports_alpha"] = supports_alpha_channel(variant) ? "true" : "false";
        stream_info.metadata["supports_12bit"] = supports_12bit_depth(variant) ? "true" : "false";
        stream_info.metadata["recommended_encoder"] = get_recommended_encoder_settings(variant);
    }
    
    return stream_info;
}

bool ProResParser::detect_corruption(AVFormatContext* format_ctx, VideoRepairEngine::StreamInfo& stream_info) const {
    if (!format_ctx || stream_info.stream_index < 0) {
        return false;
    }
    
    // Find corrupted frame ranges
    std::vector<std::pair<int64_t, int64_t>> corrupted_ranges = find_corrupted_prores_frames(format_ctx);
    
    if (!corrupted_ranges.empty()) {
        stream_info.has_corruption = true;
        stream_info.corrupted_ranges = corrupted_ranges;
        
        // Calculate corruption percentage
        int64_t total_frames = format_ctx->streams[stream_info.stream_index]->nb_frames;
        if (total_frames > 0) {
            stream_info.corruption_percentage = calculate_corruption_severity(corrupted_ranges, total_frames);
        }
        
        // Add corruption details to metadata
        stream_info.metadata["corruption_detected"] = "true";
        stream_info.metadata["corrupted_frame_ranges"] = std::to_string(corrupted_ranges.size());
        stream_info.metadata["corruption_severity"] = std::to_string(stream_info.corruption_percentage);
        
        return true;
    }
    
    return false;
}

std::vector<VideoRepairEngine::RepairTechnique> ProResParser::recommend_techniques(
    const VideoRepairEngine::StreamInfo& stream_info) const {
    
    return get_prores_specific_techniques(stream_info);
}

ProResParser::ProResVariant ProResParser::detect_prores_variant(const uint8_t* frame_data, size_t data_size) const {
    if (!frame_data || data_size < 8) {
        return ProResVariant::UNKNOWN;
    }
    
    // Look for ProRes frame header signature
    if (!validate_prores_signature(frame_data, data_size)) {
        return ProResVariant::UNKNOWN;
    }
    
    // Extract codec identifier (4 bytes at offset 4 in frame header)
    if (data_size >= 8) {
        const uint8_t* fourcc = frame_data + 4;
        return identify_variant_from_fourcc(fourcc);
    }
    
    return ProResVariant::UNKNOWN;
}

bool ProResParser::parse_prores_frame_header(const uint8_t* frame_data, size_t data_size, 
                                            ProResFrameHeader& header) const {
    if (!frame_data || data_size < sizeof(ProResFrameHeader)) {
        return false;
    }
    
    // Clear header structure
    std::memset(&header, 0, sizeof(header));
    
    // Parse frame header fields (Big-endian format)
    const uint8_t* ptr = frame_data;
    
    // Frame size (4 bytes)
    header.frame_size = (ptr[0] << 24) | (ptr[1] << 16) | (ptr[2] << 8) | ptr[3];
    ptr += 4;
    
    // Frame identifier - should be 'icpf'
    std::memcpy(header.frame_identifier, ptr, 4);
    ptr += 4;
    
    // Header size (2 bytes)
    header.header_size = (ptr[0] << 8) | ptr[1];
    ptr += 2;
    
    // Version (1 byte)
    header.version = *ptr++;
    
    // Encoder identifier (4 bytes)
    std::memcpy(header.encoder_identifier, ptr, 4);
    ptr += 4;
    
    // Video dimensions (4 bytes total)
    header.width = (ptr[0] << 8) | ptr[1];
    ptr += 2;
    header.height = (ptr[0] << 8) | ptr[1];
    ptr += 2;
    
    // Video format parameters
    header.chroma_format = *ptr++;
    header.interlaced_mode = *ptr++;
    header.aspect_ratio_info = *ptr++;
    header.framerate_code = *ptr++;
    header.color_primaries = *ptr++;
    header.transfer_characteristics = *ptr++;
    header.matrix_coefficients = *ptr++;
    header.source_format = *ptr++;
    header.alpha_channel_type = *ptr++;
    
    // Reserved bytes
    std::memcpy(header.reserved, ptr, 3);
    
    // Validate frame identifier
    if (std::memcmp(header.frame_identifier, "icpf", 4) != 0) {
        header.is_valid = false;
        return false;
    }
    
    // Detect ProRes variant from encoder identifier
    header.variant = identify_variant_from_fourcc(header.encoder_identifier);
    header.is_valid = (header.variant != ProResVariant::UNKNOWN);
    
    // Calculate quality metrics
    if (header.is_valid) {
        header.quality_factor = calculate_prores_quality_score(header, header.frame_size);
        header.expected_bitrate = get_expected_bitrate(header.variant, header.width, header.height, 25.0);
    }
    
    return header.is_valid;
}

std::vector<ProResParser::ProResAtom> ProResParser::parse_professional_metadata(AVFormatContext* format_ctx) const {
    std::vector<ProResAtom> atoms;
    
    if (!format_ctx) return atoms;
    
    // Parse metadata from AVFormatContext
    AVDictionaryEntry* entry = nullptr;
    while ((entry = av_dict_get(format_ctx->metadata, "", entry, AV_DICT_IGNORE_SUFFIX))) {
        ProResAtom atom;
        atom.type = 0; // Generic metadata atom
        
        // Convert metadata to atom structure
        std::string key = entry->key;
        std::string value = entry->value;
        
        if (key.find("timecode") != std::string::npos) {
            atom.professional_metadata.timecode = value;
        } else if (key.find("camera") != std::string::npos) {
            atom.professional_metadata.camera_model = value;
        } else if (key.find("lens") != std::string::npos) {
            atom.professional_metadata.lens_model = value;
        } else {
            atom.professional_metadata.custom_metadata[key] = value;
        }
        
        atoms.push_back(atom);
    }
    
    return atoms;
}

double ProResParser::calculate_prores_quality_score(const ProResFrameHeader& header, size_t actual_frame_size) const {
    if (!header.is_valid || header.variant == ProResVariant::UNKNOWN) {
        return 0.0;
    }
    
    // Get expected frame size based on variant and resolution
    double frame_rate = 25.0; // Default frame rate for calculation
    size_t expected_bitrate = get_expected_bitrate(header.variant, header.width, header.height, frame_rate);
    size_t expected_frame_size = expected_bitrate / (8 * frame_rate); // Convert to bytes per frame
    
    // Calculate quality score based on size ratio
    if (expected_frame_size == 0) return 0.0;
    
    double size_ratio = static_cast<double>(actual_frame_size) / expected_frame_size;
    
    // Quality score ranges from 0.0 to 1.0
    // Higher ratios indicate better quality (within reasonable bounds)
    if (size_ratio > 2.0) {
        return 1.0; // Capped at maximum quality
    } else if (size_ratio < 0.5) {
        return size_ratio * 2.0; // Linear scaling for low ratios
    } else {
        return 0.5 + (size_ratio - 0.5) * 1.0; // Adjusted scaling for normal ratios
    }
}

bool ProResParser::validate_prores_frame_integrity(const uint8_t* frame_data, size_t data_size) const {
    if (!frame_data || data_size < 32) return false;
    
    ProResFrameHeader header;
    if (!parse_prores_frame_header(frame_data, data_size, header)) {
        return false;
    }
    
    // Additional integrity checks
    return !detect_header_corruption(header) && 
           !detect_payload_corruption(frame_data, data_size, header) &&
           analyze_prores_macroblock_structure(frame_data, data_size) &&
           validate_prores_slice_structure(frame_data, data_size);
}

std::vector<VideoRepairEngine::RepairTechnique> ProResParser::get_prores_specific_techniques(
    const VideoRepairEngine::StreamInfo& stream_info) const {
    
    std::vector<VideoRepairEngine::RepairTechnique> techniques;
    
    if (stream_info.has_corruption) {
        // Header-level corruption
        if (stream_info.corruption_percentage > 0.8) {
            techniques.push_back(VideoRepairEngine::RepairTechnique::HEADER_RECONSTRUCTION);
        }
        
        // Index corruption
        techniques.push_back(VideoRepairEngine::RepairTechnique::INDEX_REBUILD);
        
        // Frame-level corruption
        if (stream_info.corruption_percentage < 0.3) {
            techniques.push_back(VideoRepairEngine::RepairTechnique::FRAGMENT_RECOVERY);
        }
        
        // Container issues
        techniques.push_back(VideoRepairEngine::RepairTechnique::CONTAINER_REMUX);
        
        // AI-based repairs for professional content
        if (stream_info.corruption_percentage > 0.1) {
            techniques.push_back(VideoRepairEngine::RepairTechnique::FRAME_INTERPOLATION);
            techniques.push_back(VideoRepairEngine::RepairTechnique::AI_INPAINTING);
        }
        
        // Metadata recovery
        techniques.push_back(VideoRepairEngine::RepairTechnique::METADATA_RECOVERY);
    }
    
    return techniques;
}

bool ProResParser::requires_professional_tools(const VideoRepairEngine::StreamInfo& stream_info) const {
    // ProRes is a professional format - always recommend professional tools
    return true;
}

bool ProResParser::extract_timecode_from_prores(AVFormatContext* format_ctx, std::string& timecode) const {
    if (!format_ctx) return false;
    
    // Look for timecode in metadata
    AVDictionaryEntry* tc_entry = av_dict_get(format_ctx->metadata, "timecode", nullptr, 0);
    if (tc_entry && tc_entry->value) {
        timecode = tc_entry->value;
        return true;
    }
    
    // Look for timecode in first video stream
    for (unsigned int i = 0; i < format_ctx->nb_streams; i++) {
        AVStream* stream = format_ctx->streams[i];
        if (stream->codecpar->codec_type == AVMEDIA_TYPE_VIDEO) {
            AVDictionaryEntry* stream_tc = av_dict_get(stream->metadata, "timecode", nullptr, 0);
            if (stream_tc && stream_tc->value) {
                timecode = stream_tc->value;
                return true;
            }
        }
    }
    
    return false;
}

bool ProResParser::extract_camera_metadata(const std::vector<ProResAtom>& atoms, 
                                          VideoRepairEngine::StreamInfo& stream_info) const {
    bool metadata_found = false;
    
    for (const auto& atom : atoms) {
        if (!atom.professional_metadata.camera_model.empty()) {
            stream_info.camera_model = atom.professional_metadata.camera_model;
            stream_info.metadata["camera_model"] = atom.professional_metadata.camera_model;
            metadata_found = true;
        }
        
        if (!atom.professional_metadata.camera_serial.empty()) {
            stream_info.metadata["camera_serial"] = atom.professional_metadata.camera_serial;
            metadata_found = true;
        }
        
        // Add all custom metadata
        for (const auto& [key, value] : atom.professional_metadata.custom_metadata) {
            stream_info.metadata[key] = value;
            metadata_found = true;
        }
    }
    
    return metadata_found;
}

bool ProResParser::extract_color_metadata(const std::vector<ProResAtom>& atoms, 
                                         VideoRepairEngine::StreamInfo& stream_info) const {
    bool metadata_found = false;
    
    for (const auto& atom : atoms) {
        if (atom.color_metadata.has_color_correction) {
            stream_info.metadata["has_color_correction"] = "true";
            stream_info.metadata["lift_rgb"] = std::to_string(atom.color_metadata.lift_r) + "," +
                                             std::to_string(atom.color_metadata.lift_g) + "," +
                                             std::to_string(atom.color_metadata.lift_b);
            stream_info.metadata["gamma_rgb"] = std::to_string(atom.color_metadata.gamma_r) + "," +
                                              std::to_string(atom.color_metadata.gamma_g) + "," +
                                              std::to_string(atom.color_metadata.gamma_b);
            stream_info.metadata["gain_rgb"] = std::to_string(atom.color_metadata.gain_r) + "," +
                                             std::to_string(atom.color_metadata.gain_g) + "," +
                                             std::to_string(atom.color_metadata.gain_b);
            stream_info.metadata["saturation"] = std::to_string(atom.color_metadata.saturation);
            stream_info.metadata["contrast"] = std::to_string(atom.color_metadata.contrast);
            stream_info.metadata["brightness"] = std::to_string(atom.color_metadata.brightness);
            
            if (!atom.color_metadata.lut_name.empty()) {
                stream_info.metadata["lut_name"] = atom.color_metadata.lut_name;
            }
            
            metadata_found = true;
        }
    }
    
    return metadata_found;
}

std::vector<std::pair<int64_t, int64_t>> ProResParser::find_corrupted_prores_frames(AVFormatContext* format_ctx) const {
    std::vector<std::pair<int64_t, int64_t>> corrupted_ranges;
    
    if (!format_ctx) return corrupted_ranges;
    
    // Find video stream
    int video_stream_index = -1;
    for (unsigned int i = 0; i < format_ctx->nb_streams; i++) {
        if (format_ctx->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO) {
            video_stream_index = i;
            break;
        }
    }
    
    if (video_stream_index == -1) return corrupted_ranges;
    
    AVPacket packet;
    av_init_packet(&packet);
    
    int64_t frame_number = 0;
    int64_t corruption_start = -1;
    int frames_analyzed = 0;
    
    // Read packets and analyze for corruption
    while (av_read_frame(format_ctx, &packet) >= 0 && frames_analyzed < m_config.max_frames_to_analyze) {
        if (packet.stream_index == video_stream_index) {
            bool frame_corrupted = !can_repair_prores_frame(packet.data, packet.size);
            
            if (frame_corrupted && corruption_start == -1) {
                corruption_start = frame_number;
            } else if (!frame_corrupted && corruption_start != -1) {
                corrupted_ranges.emplace_back(corruption_start, frame_number - 1);
                corruption_start = -1;
            }
            
            frame_number++;
            frames_analyzed++;
        }
        av_packet_unref(&packet);
    }
    
    // Handle case where corruption extends to end of file
    if (corruption_start != -1) {
        corrupted_ranges.emplace_back(corruption_start, frame_number - 1);
    }
    
    return corrupted_ranges;
}

bool ProResParser::can_repair_prores_frame(const uint8_t* frame_data, size_t data_size) const {
    if (!frame_data || data_size < 32) return false;
    
    // Basic signature validation
    if (!validate_prores_signature(frame_data, data_size)) {
        return false;
    }
    
    // Parse and validate frame header
    ProResFrameHeader header;
    if (!parse_prores_frame_header(frame_data, data_size, header)) {
        return false;
    }
    
    // Additional integrity checks
    return !detect_header_corruption(header) && 
           validate_prores_frame_integrity(frame_data, data_size);
}

bool ProResParser::supports_alpha_channel(ProResVariant variant) const {
    return variant == ProResVariant::PRORES_4444 || 
           variant == ProResVariant::PRORES_4444_XQ;
}

bool ProResParser::supports_12bit_depth(ProResVariant variant) const {
    return variant == ProResVariant::PRORES_4444_XQ || 
           variant == ProResVariant::RAW || 
           variant == ProResVariant::RAW_HQ;
}

std::string ProResParser::get_recommended_encoder_settings(ProResVariant variant) const {
    std::stringstream settings;
    
    switch (variant) {
        case ProResVariant::PROXY:
            settings << "-c:v prores_ks -profile:v 0 -vendor ap10 -bits_per_mb 200";
            break;
        case ProResVariant::LT:
            settings << "-c:v prores_ks -profile:v 1 -vendor ap10 -bits_per_mb 360";
            break;
        case ProResVariant::STANDARD:
            settings << "-c:v prores_ks -profile:v 2 -vendor ap10 -bits_per_mb 540";
            break;
        case ProResVariant::HQ:
            settings << "-c:v prores_ks -profile:v 3 -vendor ap10 -bits_per_mb 720";
            break;
        case ProResVariant::PRORES_4444:
            settings << "-c:v prores_ks -profile:v 4 -vendor ap4h -bits_per_mb 990";
            break;
        case ProResVariant::PRORES_4444_XQ:
            settings << "-c:v prores_ks -profile:v 5 -vendor ap4x -bits_per_mb 1485";
            break;
        default:
            settings << "-c:v prores_ks -profile:v 2 -vendor ap10";
            break;
    }
    
    return settings.str();
}

std::string ProResParser::variant_to_string(ProResVariant variant) {
    auto it = VARIANT_NAMES.find(variant);
    return (it != VARIANT_NAMES.end()) ? it->second : "Unknown ProRes Variant";
}

std::string ProResParser::get_prores_fourcc(ProResVariant variant) {
    switch (variant) {
        case ProResVariant::PROXY: return "apco";
        case ProResVariant::LT: return "apcs";
        case ProResVariant::STANDARD: return "apcn";
        case ProResVariant::HQ: return "apch";
        case ProResVariant::PRORES_4444: return "ap4h";
        case ProResVariant::PRORES_4444_XQ: return "ap4x";
        case ProResVariant::RAW: return "aprn";
        case ProResVariant::RAW_HQ: return "aprh";
        default: return "unkn";
    }
}

size_t ProResParser::get_expected_bitrate(ProResVariant variant, int width, int height, double framerate) {
    auto it = VARIANT_BITRATES.find(variant);
    if (it == VARIANT_BITRATES.end()) return 0;
    
    // Base bitrate for 1920x1080@25fps
    size_t base_bitrate = it->second;
    
    // Scale based on resolution and frame rate
    double resolution_factor = (static_cast<double>(width * height)) / (1920.0 * 1080.0);
    double framerate_factor = framerate / 25.0;
    
    return static_cast<size_t>(base_bitrate * resolution_factor * framerate_factor);
}

// Private implementation methods
bool ProResParser::validate_prores_signature(const uint8_t* data, size_t size) const {
    if (!data || size < 8) return false;
    
    // Check for ProRes frame header signature 'icpf'
    return (size >= 8 && std::memcmp(data + 4, "icpf", 4) == 0);
}

ProResParser::ProResVariant ProResParser::identify_variant_from_fourcc(const uint8_t fourcc[4]) const {
    if (!fourcc) return ProResVariant::UNKNOWN;
    
    uint32_t fourcc_val = (fourcc[0] << 24) | (fourcc[1] << 16) | (fourcc[2] << 8) | fourcc[3];
    
    auto it = FOURCC_TO_VARIANT.find(fourcc_val);
    return (it != FOURCC_TO_VARIANT.end()) ? it->second : ProResVariant::UNKNOWN;
}

bool ProResParser::parse_atom_hierarchy(const uint8_t* data, size_t size, std::vector<ProResAtom>& atoms) const {
    // Implementation for parsing QuickTime-style atoms in ProRes metadata
    // This is a simplified version - full implementation would be more complex
    if (!data || size < 8) return false;
    
    size_t offset = 0;
    while (offset + 8 <= size) {
        ProResAtom atom;
        
        // Read atom size (4 bytes, big-endian)
        atom.size = (data[offset] << 24) | (data[offset+1] << 16) | 
                   (data[offset+2] << 8) | data[offset+3];
        offset += 4;
        
        // Read atom type (4 bytes)
        atom.type = (data[offset] << 24) | (data[offset+1] << 16) | 
                   (data[offset+2] << 8) | data[offset+3];
        offset += 4;
        
        // Validate atom size
        if (atom.size < 8 || offset + atom.size - 8 > size) {
            break;
        }
        
        // Copy atom data
        atom.data.resize(atom.size - 8);
        std::memcpy(atom.data.data(), data + offset, atom.size - 8);
        
        atoms.push_back(atom);
        offset += atom.size - 8;
    }
    
    return !atoms.empty();
}

bool ProResParser::analyze_prores_macroblock_structure(const uint8_t* frame_data, size_t data_size) const {
    // Simplified macroblock structure analysis
    if (!frame_data || data_size < 64) return false;
    
    ProResFrameHeader header;
    if (!parse_prores_frame_header(frame_data, data_size, header)) {
        return false;
    }
    
    // ProRes uses different macroblock sizes based on chroma format
    int mb_size = (header.chroma_format == 3) ? 16 : 16; // 16x16 for both 422 and 444
    int mb_width = (header.width + mb_size - 1) / mb_size;
    int mb_height = (header.height + mb_size - 1) / mb_size;
    
    // Basic validation that we have enough data for expected macroblocks
    size_t expected_mb_data = mb_width * mb_height * 64; // Rough estimate
    size_t available_data = data_size - header.header_size;
    
    return available_data >= expected_mb_data / 4; // Allow for compression
}

bool ProResParser::validate_prores_slice_structure(const uint8_t* frame_data, size_t data_size) const {
    // Simplified slice structure validation
    if (!frame_data || data_size < 32) return false;
    
    ProResFrameHeader header;
    if (!parse_prores_frame_header(frame_data, data_size, header)) {
        return false;
    }
    
    // ProRes typically uses slice-based encoding
    // This is a basic validation that slice headers are present and valid
    const uint8_t* slice_data = frame_data + header.header_size;
    size_t slice_data_size = data_size - header.header_size;
    
    // Look for slice header patterns
    return slice_data_size > 16; // Minimum size for slice data
}

bool ProResParser::detect_header_corruption(const ProResFrameHeader& header) const {
    if (!header.is_valid) return true;
    
    // Validate header fields for reasonable values
    if (header.width == 0 || header.height == 0 || 
        header.width > 16384 || header.height > 16384) {
        return true;
    }
    
    if (header.chroma_format != 2 && header.chroma_format != 3) {
        return true;
    }
    
    if (header.framerate_code == 0 || header.framerate_code > 11) {
        return true;
    }
    
    return false;
}

bool ProResParser::detect_payload_corruption(const uint8_t* frame_data, size_t data_size, 
                                            const ProResFrameHeader& header) const {
    if (!frame_data || data_size <= header.header_size) return true;
    
    // Check if payload size matches header specification
    if (header.frame_size != data_size) {
        return true;
    }
    
    // Additional payload validation could be added here
    return false;
}

double ProResParser::calculate_corruption_severity(const std::vector<std::pair<int64_t, int64_t>>& corrupted_ranges, 
                                                  int64_t total_frames) const {
    if (total_frames == 0) return 0.0;
    
    int64_t corrupted_frames = 0;
    for (const auto& range : corrupted_ranges) {
        corrupted_frames += (range.second - range.first + 1);
    }
    
    return static_cast<double>(corrupted_frames) / total_frames;
}