#include "VideoRepair/FormatParsers/BRAWParser.h"
#include "Core/ErrorHandling.h"
#include <algorithm>
#include <iomanip>
#include <sstream>
#include <cstring>
#include <cmath>

extern "C" {
#include <libavutil/timecode.h>
#include <libavutil/dict.h>
}

// Static constant definitions
const std::unordered_map<uint32_t, BRAWParser::BRAWCompressionRatio> BRAWParser::COMPRESSION_RATIO_CODES = {
    {0x33315F42, BRAWCompressionRatio::RATIO_3_1},    // "B_13" (3:1)
    {0x35315F42, BRAWCompressionRatio::RATIO_5_1},    // "B_15" (5:1)
    {0x38315F42, BRAWCompressionRatio::RATIO_8_1},    // "B_18" (8:1)
    {0x32315F42, BRAWCompressionRatio::RATIO_12_1},   // "B_112" (12:1)
    {0x51435F42, BRAWCompressionRatio::RATIO_CONSTANT_QUALITY}, // "B_CQ"
    {0x52425F42, BRAWCompressionRatio::RATIO_CONSTANT_BITRATE}  // "B_BR"
};

const std::unordered_map<BRAWParser::BRAWCompressionRatio, std::string> BRAWParser::COMPRESSION_RATIO_NAMES = {
    {BRAWCompressionRatio::RATIO_3_1, "BRAW 3:1"},
    {BRAWCompressionRatio::RATIO_5_1, "BRAW 5:1"},
    {BRAWCompressionRatio::RATIO_8_1, "BRAW 8:1"},
    {BRAWCompressionRatio::RATIO_12_1, "BRAW 12:1"},
    {BRAWCompressionRatio::RATIO_CONSTANT_QUALITY, "BRAW CQ"},
    {BRAWCompressionRatio::RATIO_CONSTANT_BITRATE, "BRAW CBR"}
};

const std::unordered_map<BRAWParser::BRAWCompressionRatio, size_t> BRAWParser::COMPRESSION_RATIO_BITRATES = {
    {BRAWCompressionRatio::RATIO_3_1, 800000000},     // ~800 Mbps for 4K@24p
    {BRAWCompressionRatio::RATIO_5_1, 480000000},     // ~480 Mbps
    {BRAWCompressionRatio::RATIO_8_1, 300000000},     // ~300 Mbps
    {BRAWCompressionRatio::RATIO_12_1, 200000000},    // ~200 Mbps
    {BRAWCompressionRatio::RATIO_CONSTANT_QUALITY, 400000000}, // Variable
    {BRAWCompressionRatio::RATIO_CONSTANT_BITRATE, 350000000}  // Configurable
};

const std::unordered_map<BRAWParser::ColorScienceVersion, std::string> BRAWParser::COLOR_SCIENCE_NAMES = {
    {ColorScienceVersion::GENERATION_1, "Blackmagic Design Color Science Gen 1"},
    {ColorScienceVersion::GENERATION_2, "Blackmagic Design Color Science Gen 2"},
    {ColorScienceVersion::GENERATION_3, "Blackmagic Design Color Science Gen 3"},
    {ColorScienceVersion::GENERATION_4, "Blackmagic Design Color Science Gen 4"},
    {ColorScienceVersion::GENERATION_5, "Blackmagic Design Color Science Gen 5"}
};

const std::unordered_map<uint8_t, std::string> BRAWParser::GAMMA_CURVE_NAMES = {
    {0x00, "Blackmagic Design Film"},
    {0x01, "Blackmagic Design Video"},
    {0x02, "Blackmagic Design Extended Video"},
    {0x03, "Rec. 709"},
    {0x04, "Rec. 2020"},
    {0x05, "SMPTE ST 2084 (PQ)"},
    {0x06, "Hybrid Log-Gamma (HLG)"},
    {0x07, "Custom LUT"}
};

const std::unordered_map<uint8_t, std::string> BRAWParser::COLOR_SPACE_NAMES = {
    {0x00, "Blackmagic Wide Gamut"},
    {0x01, "DCI-P3"},
    {0x02, "Rec. 709"},
    {0x03, "Rec. 2020"},
    {0x04, "Adobe RGB"},
    {0x05, "P3 D65"},
    {0x06, "Blackmagic Pocket Cinema Gamut"}
};

const std::unordered_map<uint8_t, std::string> BRAWParser::BAYER_PATTERN_NAMES = {
    {0x00, "RGGB"},
    {0x01, "GRBG"},
    {0x02, "GBRG"},
    {0x03, "BGGR"}
};

BRAWParser::BRAWParser() : VideoRepairEngine::FormatParser(VideoRepairEngine::VideoFormat::BLACKMAGIC_RAW) {
    // Initialize parser configuration for BRAW
    m_config.strict_validation = true;
    m_config.extract_all_metadata = true;
    m_config.deep_sensor_analysis = false;
    m_config.extract_embedded_luts = true;
    m_config.corruption_threshold = 0.03;
    m_config.max_frames_to_analyze = 50;
    m_config.blackmagic_sdk_available = false; // Would be detected at runtime
}

bool BRAWParser::can_parse(const std::string& file_path) const {
    // Check file extension first
    if (file_path.size() > 5) {
        std::string extension = file_path.substr(file_path.size() - 5);
        std::transform(extension.begin(), extension.end(), extension.begin(), ::tolower);
        if (extension != ".braw") {
            return false;
        }
    }
    
    AVFormatContext* format_ctx = nullptr;
    
    // Open file and check for BRAW streams
    if (avformat_open_input(&format_ctx, file_path.c_str(), nullptr, nullptr) < 0) {
        return false;
    }
    
    bool can_parse = false;
    if (avformat_find_stream_info(format_ctx, nullptr) >= 0) {
        for (unsigned int i = 0; i < format_ctx->nb_streams; i++) {
            AVStream* stream = format_ctx->streams[i];
            if (stream->codecpar->codec_type == AVMEDIA_TYPE_VIDEO) {
                // Check for BRAW-specific codec tags or format indicators
                if (stream->codecpar->codec_tag == MKTAG('B','R','A','W') ||
                    (stream->codecpar->extradata && stream->codecpar->extradata_size >= 4 &&
                     memcmp(stream->codecpar->extradata, "BRAW", 4) == 0)) {
                    can_parse = true;
                    break;
                }
            }
        }
        
        // Additional check by examining format name
        if (!can_parse && format_ctx->iformat) {
            std::string format_name = format_ctx->iformat->name ? format_ctx->iformat->name : "";
            if (format_name.find("braw") != std::string::npos) {
                can_parse = true;
            }
        }
    }
    
    avformat_close_input(&format_ctx);
    return can_parse;
}

VideoRepairEngine::StreamInfo BRAWParser::parse_stream_info(AVFormatContext* format_ctx, int stream_index) const {
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
    stream_info.detected_format = VideoRepairEngine::VideoFormat::BLACKMAGIC_RAW;
    
    if (codecpar->codec_type == AVMEDIA_TYPE_VIDEO) {
        stream_info.width = codecpar->width;
        stream_info.height = codecpar->height;
        stream_info.frame_rate = stream->avg_frame_rate;
        stream_info.time_base = stream->time_base;
        stream_info.pixel_format = static_cast<AVPixelFormat>(codecpar->format);
        stream_info.color_space = codecpar->color_space;
        stream_info.color_primaries = codecpar->color_primaries;
        stream_info.color_trc = codecpar->color_trc;
        
        // Parse BRAW-specific metadata from extradata
        if (codecpar->extradata && codecpar->extradata_size > 0) {
            BRAWCompressionRatio compression_ratio = detect_compression_ratio(codecpar->extradata, codecpar->extradata_size);
            
            BRAWFrameHeader header;
            if (parse_braw_frame_header(codecpar->extradata, codecpar->extradata_size, header)) {
                stream_info.bit_depth = header.bit_depth;
                
                // Add BRAW-specific metadata
                stream_info.metadata["compression_ratio"] = compression_ratio_to_string(compression_ratio);
                stream_info.metadata["color_science"] = color_science_to_string(header.color_science);
                stream_info.metadata["sensor_width"] = std::to_string(header.sensor_width);
                stream_info.metadata["sensor_height"] = std::to_string(header.sensor_height);
                stream_info.metadata["active_width"] = std::to_string(header.active_width);
                stream_info.metadata["active_height"] = std::to_string(header.active_height);
                stream_info.metadata["windowed_recording"] = header.is_windowed_recording ? "true" : "false";
                stream_info.metadata["white_balance_kelvin"] = std::to_string(header.white_balance_kelvin);
                stream_info.metadata["iso_value"] = std::to_string(header.iso_value);
                stream_info.metadata["exposure_compensation"] = std::to_string(header.exposure_compensation);
                
                // Calculate expected bitrate
                size_t expected_bitrate = get_expected_bitrate(compression_ratio, 
                    stream_info.width, stream_info.height, av_q2d(stream_info.frame_rate));
                stream_info.metadata["expected_bitrate"] = std::to_string(expected_bitrate);
            }
        }
        
        // Extract comprehensive metadata
        std::vector<BRAWMetadataAtom> atoms = parse_braw_metadata(format_ctx);
        extract_camera_settings(atoms, stream_info);
        extract_lens_metadata(atoms, stream_info);
        extract_production_metadata(atoms, stream_info);
        extract_timecode_from_braw(format_ctx, stream_info.timecode);
        
        // Professional format indicators
        stream_info.metadata["requires_blackmagic_sdk"] = requires_blackmagic_sdk(stream_info) ? "true" : "false";
        stream_info.metadata["davinci_resolve_compatible"] = "true";
        stream_info.metadata["supports_embedded_lut"] = "true";
    }
    
    return stream_info;
}

bool BRAWParser::detect_corruption(AVFormatContext* format_ctx, VideoRepairEngine::StreamInfo& stream_info) const {
    if (!format_ctx || stream_info.stream_index < 0) {
        return false;
    }
    
    // Find corrupted frame ranges
    std::vector<std::pair<int64_t, int64_t>> corrupted_ranges = find_corrupted_braw_frames(format_ctx);
    
    if (!corrupted_ranges.empty()) {
        stream_info.has_corruption = true;
        stream_info.corrupted_ranges = corrupted_ranges;
        
        // Calculate corruption percentage
        int64_t total_frames = format_ctx->streams[stream_info.stream_index]->nb_frames;
        if (total_frames > 0) {
            stream_info.corruption_percentage = calculate_corruption_severity_braw(corrupted_ranges, total_frames);
        }
        
        // Add corruption details to metadata
        stream_info.metadata["corruption_detected"] = "true";
        stream_info.metadata["corrupted_frame_ranges"] = std::to_string(corrupted_ranges.size());
        stream_info.metadata["corruption_severity"] = std::to_string(stream_info.corruption_percentage);
        
        return true;
    }
    
    return false;
}

std::vector<VideoRepairEngine::RepairTechnique> BRAWParser::recommend_techniques(
    const VideoRepairEngine::StreamInfo& stream_info) const {
    
    return get_braw_specific_techniques(stream_info);
}

BRAWParser::BRAWCompressionRatio BRAWParser::detect_compression_ratio(const uint8_t* frame_data, size_t data_size) const {
    if (!frame_data || data_size < 16) {
        return BRAWCompressionRatio::UNKNOWN;
    }
    
    // Look for BRAW signature and compression ratio identifier
    if (!validate_braw_signature(frame_data, data_size)) {
        return BRAWCompressionRatio::UNKNOWN;
    }
    
    // BRAW compression ratio is typically stored in the first 16 bytes
    BRAWFrameHeader header;
    if (parse_braw_frame_header(frame_data, data_size, header)) {
        return header.compression_ratio;
    }
    
    return BRAWCompressionRatio::UNKNOWN;
}

bool BRAWParser::parse_braw_frame_header(const uint8_t* frame_data, size_t data_size, BRAWFrameHeader& header) const {
    if (!frame_data || data_size < sizeof(BRAWFrameHeader)) {
        return false;
    }
    
    // Clear header structure
    std::memset(&header, 0, sizeof(header));
    
    // Parse BRAW frame header fields (Little-endian format for BRAW)
    const uint8_t* ptr = frame_data;
    
    // Magic number - should be 'BRAW'
    header.magic_number = ptr[0] | (ptr[1] << 8) | (ptr[2] << 16) | (ptr[3] << 24);
    ptr += 4;
    
    // Frame size (4 bytes)
    header.frame_size = ptr[0] | (ptr[1] << 8) | (ptr[2] << 16) | (ptr[3] << 24);
    ptr += 4;
    
    // Frame number (4 bytes)
    header.frame_number = ptr[0] | (ptr[1] << 8) | (ptr[2] << 16) | (ptr[3] << 24);
    ptr += 4;
    
    // Timestamp (8 bytes)
    header.timestamp_low = ptr[0] | (ptr[1] << 8) | (ptr[2] << 16) | (ptr[3] << 24);
    ptr += 4;
    header.timestamp_high = ptr[0] | (ptr[1] << 8) | (ptr[2] << 16) | (ptr[3] << 24);
    ptr += 4;
    header.full_timestamp = (static_cast<uint64_t>(header.timestamp_high) << 32) | header.timestamp_low;
    
    // Video dimensions
    header.width = ptr[0] | (ptr[1] << 8);
    ptr += 2;
    header.height = ptr[0] | (ptr[1] << 8);
    ptr += 2;
    header.sensor_width = ptr[0] | (ptr[1] << 8);
    ptr += 2;
    header.sensor_height = ptr[0] | (ptr[1] << 8);
    ptr += 2;
    header.active_width = ptr[0] | (ptr[1] << 8);
    ptr += 2;
    header.active_height = ptr[0] | (ptr[1] << 8);
    ptr += 2;
    
    // Color and compression information
    header.bit_depth = *ptr++;
    header.color_format = *ptr++;
    header.compression_ratio = static_cast<BRAWCompressionRatio>(*ptr++);
    header.color_science = static_cast<ColorScienceVersion>(*ptr++);
    header.gamma_curve = *ptr++;
    header.color_space = *ptr++;
    
    // Camera settings (stored as IEEE 754 floats)
    std::memcpy(&header.white_balance_kelvin, ptr, 4);
    ptr += 4;
    std::memcpy(&header.tint_adjustment, ptr, 4);
    ptr += 4;
    std::memcpy(&header.exposure_compensation, ptr, 4);
    ptr += 4;
    std::memcpy(&header.iso_value, ptr, 4);
    ptr += 4;
    
    // Reserved bytes
    std::memcpy(header.reserved, ptr, 32);
    
    // Validate magic number
    if (header.magic_number != MKTAG('B','R','A','W')) {
        header.is_valid = false;
        return false;
    }
    
    // Additional validation
    header.is_valid = (header.width > 0 && header.height > 0 && 
                      header.bit_depth >= 8 && header.bit_depth <= 16 &&
                      header.compression_ratio != BRAWCompressionRatio::UNKNOWN);
    
    // Determine if this is a windowed recording
    header.is_windowed_recording = (header.active_width < header.sensor_width || 
                                   header.active_height < header.sensor_height);
    
    // Calculate quality metrics
    if (header.is_valid) {
        header.quality_factor = calculate_braw_quality_score(header, header.frame_size);
    }
    
    return header.is_valid;
}

std::vector<BRAWParser::BRAWMetadataAtom> BRAWParser::parse_braw_metadata(AVFormatContext* format_ctx) const {
    std::vector<BRAWMetadataAtom> atoms;
    
    if (!format_ctx) return atoms;
    
    // Parse metadata from AVFormatContext
    AVDictionaryEntry* entry = nullptr;
    while ((entry = av_dict_get(format_ctx->metadata, "", entry, AV_DICT_IGNORE_SUFFIX))) {
        BRAWMetadataAtom atom;
        atom.atom_type = 0; // Generic metadata atom
        
        std::string key = entry->key;
        std::string value = entry->value;
        
        // Parse Blackmagic-specific metadata
        if (key.find("camera") != std::string::npos) {
            atom.camera_metadata.camera_model = value;
        } else if (key.find("serial") != std::string::npos) {
            atom.camera_metadata.camera_serial_number = value;
        } else if (key.find("lens") != std::string::npos) {
            atom.camera_metadata.lens_metadata.lens_model = value;
        } else if (key.find("timecode") != std::string::npos) {
            atom.production_metadata.timecode_metadata.timecode_start = value;
        }
        
        atoms.push_back(atom);
    }
    
    return atoms;
}

double BRAWParser::calculate_braw_quality_score(const BRAWFrameHeader& header, size_t actual_frame_size) const {
    if (!header.is_valid || header.compression_ratio == BRAWCompressionRatio::UNKNOWN) {
        return 0.0;
    }
    
    // Get expected frame size based on compression ratio and resolution
    size_t expected_frame_size = estimate_frame_size(header);
    
    if (expected_frame_size == 0) return 0.0;
    
    double size_ratio = static_cast<double>(actual_frame_size) / expected_frame_size;
    
    // Quality score calculation for BRAW
    // Higher compression ratios (12:1) have lower expected quality
    double base_quality = 1.0;
    switch (header.compression_ratio) {
        case BRAWCompressionRatio::RATIO_3_1:
            base_quality = 1.0;
            break;
        case BRAWCompressionRatio::RATIO_5_1:
            base_quality = 0.9;
            break;
        case BRAWCompressionRatio::RATIO_8_1:
            base_quality = 0.8;
            break;
        case BRAWCompressionRatio::RATIO_12_1:
            base_quality = 0.7;
            break;
        default:
            base_quality = 0.85;
            break;
    }
    
    // Adjust based on actual vs expected size
    if (size_ratio > 1.5) {
        return base_quality; // Capped at base quality
    } else if (size_ratio < 0.5) {
        return base_quality * size_ratio * 2.0; // Significantly degraded
    } else {
        return base_quality * (0.5 + size_ratio * 0.5); // Linear scaling
    }
}

bool BRAWParser::validate_braw_frame_integrity(const uint8_t* frame_data, size_t data_size) const {
    if (!frame_data || data_size < 64) return false;
    
    BRAWFrameHeader header;
    if (!parse_braw_frame_header(frame_data, data_size, header)) {
        return false;
    }
    
    // Additional integrity checks specific to BRAW
    return !detect_header_corruption(header) && 
           !detect_sensor_data_corruption(frame_data, data_size, header) &&
           analyze_braw_sensor_pattern(frame_data, data_size, header) &&
           validate_braw_compression_structure(frame_data, data_size);
}

BRAWParser::ColorScienceVersion BRAWParser::detect_color_science_version(const BRAWFrameHeader& header) const {
    return header.color_science;
}

std::string BRAWParser::get_color_science_name(ColorScienceVersion version) const {
    auto it = COLOR_SCIENCE_NAMES.find(version);
    return (it != COLOR_SCIENCE_NAMES.end()) ? it->second : "Unknown Color Science";
}

bool BRAWParser::supports_extended_video_levels(ColorScienceVersion version) const {
    // Generation 4 and 5 support extended video levels
    return version >= ColorScienceVersion::GENERATION_4;
}

bool BRAWParser::is_constant_quality_mode(BRAWCompressionRatio ratio) const {
    return ratio == BRAWCompressionRatio::RATIO_CONSTANT_QUALITY;
}

double BRAWParser::get_expected_compression_efficiency(BRAWCompressionRatio ratio) const {
    return calculate_expected_compression_ratio(ratio);
}

size_t BRAWParser::estimate_frame_size(const BRAWFrameHeader& header) const {
    if (!header.is_valid) return 0;
    
    // Calculate based on sensor data and compression ratio
    size_t sensor_pixels = static_cast<size_t>(header.active_width) * header.active_height;
    size_t bits_per_pixel = header.bit_depth;
    size_t raw_size = (sensor_pixels * bits_per_pixel) / 8;
    
    // Apply compression efficiency
    double compression_efficiency = get_expected_compression_efficiency(header.compression_ratio);
    return static_cast<size_t>(raw_size * compression_efficiency);
}

bool BRAWParser::extract_camera_settings(const std::vector<BRAWMetadataAtom>& atoms, 
                                        VideoRepairEngine::StreamInfo& stream_info) const {
    bool metadata_found = false;
    
    for (const auto& atom : atoms) {
        if (!atom.camera_metadata.camera_model.empty()) {
            stream_info.camera_model = atom.camera_metadata.camera_model;
            stream_info.metadata["camera_model"] = atom.camera_metadata.camera_model;
            metadata_found = true;
        }
        
        if (!atom.camera_metadata.camera_serial_number.empty()) {
            stream_info.metadata["camera_serial"] = atom.camera_metadata.camera_serial_number;
            metadata_found = true;
        }
        
        if (!atom.camera_metadata.firmware_version.empty()) {
            stream_info.metadata["firmware_version"] = atom.camera_metadata.firmware_version;
            metadata_found = true;
        }
        
        // Recording settings
        const auto& rec_settings = atom.camera_metadata.recording_settings;
        if (!rec_settings.recording_format.empty()) {
            stream_info.metadata["recording_format"] = rec_settings.recording_format;
            metadata_found = true;
        }
        
        if (rec_settings.iso_value > 0) {
            stream_info.metadata["iso_value"] = std::to_string(rec_settings.iso_value);
            metadata_found = true;
        }
        
        if (rec_settings.shutter_angle > 0) {
            stream_info.metadata["shutter_angle"] = std::to_string(rec_settings.shutter_angle);
            metadata_found = true;
        }
    }
    
    return metadata_found;
}

bool BRAWParser::extract_lens_metadata(const std::vector<BRAWMetadataAtom>& atoms, 
                                      VideoRepairEngine::StreamInfo& stream_info) const {
    bool metadata_found = false;
    
    for (const auto& atom : atoms) {
        const auto& lens_meta = atom.camera_metadata.lens_metadata;
        
        if (!lens_meta.lens_model.empty()) {
            stream_info.lens_model = lens_meta.lens_model;
            stream_info.metadata["lens_model"] = lens_meta.lens_model;
            metadata_found = true;
        }
        
        if (!lens_meta.lens_serial_number.empty()) {
            stream_info.metadata["lens_serial"] = lens_meta.lens_serial_number;
            metadata_found = true;
        }
        
        if (lens_meta.focal_length_mm > 0) {
            stream_info.metadata["focal_length_mm"] = std::to_string(lens_meta.focal_length_mm);
            metadata_found = true;
        }
        
        if (lens_meta.aperture_f_stop > 0) {
            stream_info.metadata["aperture_f_stop"] = std::to_string(lens_meta.aperture_f_stop);
            metadata_found = true;
        }
        
        if (lens_meta.focus_distance_m > 0) {
            stream_info.metadata["focus_distance_m"] = std::to_string(lens_meta.focus_distance_m);
            metadata_found = true;
        }
    }
    
    return metadata_found;
}

bool BRAWParser::extract_production_metadata(const std::vector<BRAWMetadataAtom>& atoms, 
                                            VideoRepairEngine::StreamInfo& stream_info) const {
    bool metadata_found = false;
    
    for (const auto& atom : atoms) {
        const auto& prod_meta = atom.production_metadata;
        
        if (!prod_meta.project_name.empty()) {
            stream_info.metadata["project_name"] = prod_meta.project_name;
            metadata_found = true;
        }
        
        if (!prod_meta.scene_name.empty()) {
            stream_info.metadata["scene_name"] = prod_meta.scene_name;
            metadata_found = true;
        }
        
        if (!prod_meta.shot_name.empty()) {
            stream_info.metadata["shot_name"] = prod_meta.shot_name;
            metadata_found = true;
        }
        
        if (!prod_meta.director.empty()) {
            stream_info.metadata["director"] = prod_meta.director;
            metadata_found = true;
        }
        
        if (!prod_meta.cinematographer.empty()) {
            stream_info.metadata["cinematographer"] = prod_meta.cinematographer;
            metadata_found = true;
        }
    }
    
    return metadata_found;
}

bool BRAWParser::extract_timecode_from_braw(AVFormatContext* format_ctx, std::string& timecode) const {
    if (!format_ctx) return false;
    
    // Look for timecode in metadata
    AVDictionaryEntry* tc_entry = av_dict_get(format_ctx->metadata, "timecode", nullptr, 0);
    if (tc_entry && tc_entry->value) {
        timecode = tc_entry->value;
        return true;
    }
    
    // BRAW-specific timecode extraction
    tc_entry = av_dict_get(format_ctx->metadata, "blackmagic_timecode", nullptr, 0);
    if (tc_entry && tc_entry->value) {
        timecode = tc_entry->value;
        return true;
    }
    
    return false;
}

std::vector<std::pair<int64_t, int64_t>> BRAWParser::find_corrupted_braw_frames(AVFormatContext* format_ctx) const {
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
            bool frame_corrupted = !can_repair_braw_frame(packet.data, packet.size);
            
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

bool BRAWParser::can_repair_braw_frame(const uint8_t* frame_data, size_t data_size) const {
    if (!frame_data || data_size < 64) return false;
    
    // Basic signature validation
    if (!validate_braw_signature(frame_data, data_size)) {
        return false;
    }
    
    // Parse and validate frame header
    BRAWFrameHeader header;
    if (!parse_braw_frame_header(frame_data, data_size, header)) {
        return false;
    }
    
    // Additional integrity checks
    return !detect_header_corruption(header) && 
           validate_braw_frame_integrity(frame_data, data_size);
}

bool BRAWParser::validate_braw_sensor_data(const uint8_t* frame_data, size_t data_size, 
                                          const BRAWFrameHeader& header) const {
    if (!frame_data || data_size < 64 || !header.is_valid) return false;
    
    // Calculate expected sensor data size
    size_t expected_sensor_data_size = (static_cast<size_t>(header.active_width) * 
                                       header.active_height * header.bit_depth) / 8;
    
    // Account for compression
    double compression_efficiency = get_expected_compression_efficiency(header.compression_ratio);
    size_t expected_compressed_size = static_cast<size_t>(expected_sensor_data_size * compression_efficiency);
    
    // Allow for reasonable variance in compressed size
    return (data_size >= expected_compressed_size * 0.5 && 
            data_size <= expected_compressed_size * 2.0);
}

bool BRAWParser::requires_blackmagic_sdk(const VideoRepairEngine::StreamInfo& stream_info) const {
    // BRAW files benefit from the Blackmagic RAW SDK for optimal processing
    return true;
}

std::string BRAWParser::get_davinci_resolve_settings(const BRAWFrameHeader& header) const {
    std::stringstream settings;
    
    settings << "{\n";
    settings << "  \"format\": \"BRAW\",\n";
    settings << "  \"compression\": \"" << compression_ratio_to_string(header.compression_ratio) << "\",\n";
    settings << "  \"colorScience\": \"" << color_science_to_string(header.color_science) << "\",\n";
    settings << "  \"whiteBalance\": " << header.white_balance_kelvin << ",\n";
    settings << "  \"tint\": " << header.tint_adjustment << ",\n";
    settings << "  \"exposure\": " << header.exposure_compensation << ",\n";
    settings << "  \"iso\": " << header.iso_value << ",\n";
    settings << "  \"bitDepth\": " << static_cast<int>(header.bit_depth) << "\n";
    settings << "}";
    
    return settings.str();
}

bool BRAWParser::extract_embedded_lut(const std::vector<BRAWMetadataAtom>& atoms, 
                                     std::vector<uint8_t>& lut_data) const {
    // Implementation would parse LUT data from BRAW metadata atoms
    // This is simplified - actual implementation would be more complex
    for (const auto& atom : atoms) {
        if (atom.camera_metadata.recording_settings.lut_enabled && !atom.atom_data.empty()) {
            lut_data = atom.atom_data;
            return true;
        }
    }
    return false;
}

std::vector<VideoRepairEngine::RepairTechnique> BRAWParser::get_braw_specific_techniques(
    const VideoRepairEngine::StreamInfo& stream_info) const {
    
    std::vector<VideoRepairEngine::RepairTechnique> techniques;
    
    if (stream_info.has_corruption) {
        // Header-level corruption
        if (stream_info.corruption_percentage > 0.7) {
            techniques.push_back(VideoRepairEngine::RepairTechnique::HEADER_RECONSTRUCTION);
        }
        
        // BRAW-specific repairs
        techniques.push_back(VideoRepairEngine::RepairTechnique::INDEX_REBUILD);
        techniques.push_back(VideoRepairEngine::RepairTechnique::CONTAINER_REMUX);
        
        // Frame-level corruption
        if (stream_info.corruption_percentage < 0.4) {
            techniques.push_back(VideoRepairEngine::RepairTechnique::FRAGMENT_RECOVERY);
        }
        
        // AI-based repairs for RAW content
        if (stream_info.corruption_percentage > 0.05) {
            techniques.push_back(VideoRepairEngine::RepairTechnique::AI_INPAINTING);
            techniques.push_back(VideoRepairEngine::RepairTechnique::SUPER_RESOLUTION);
        }
        
        // Metadata recovery
        techniques.push_back(VideoRepairEngine::RepairTechnique::METADATA_RECOVERY);
    }
    
    return techniques;
}

bool BRAWParser::supports_partial_frame_recovery(BRAWCompressionRatio ratio) const {
    // Lower compression ratios have better partial recovery potential
    return ratio == BRAWCompressionRatio::RATIO_3_1 || 
           ratio == BRAWCompressionRatio::RATIO_5_1;
}

// Static utility methods
std::string BRAWParser::compression_ratio_to_string(BRAWCompressionRatio ratio) {
    auto it = COMPRESSION_RATIO_NAMES.find(ratio);
    return (it != COMPRESSION_RATIO_NAMES.end()) ? it->second : "Unknown BRAW Compression";
}

std::string BRAWParser::color_science_to_string(ColorScienceVersion version) {
    auto it = COLOR_SCIENCE_NAMES.find(version);
    return (it != COLOR_SCIENCE_NAMES.end()) ? it->second : "Unknown Color Science";
}

size_t BRAWParser::get_expected_bitrate(BRAWCompressionRatio ratio, int width, int height, double framerate) {
    auto it = COMPRESSION_RATIO_BITRATES.find(ratio);
    if (it == COMPRESSION_RATIO_BITRATES.end()) return 0;
    
    // Base bitrate for 4096x2160@24fps
    size_t base_bitrate = it->second;
    
    // Scale based on resolution and frame rate
    double resolution_factor = (static_cast<double>(width * height)) / (4096.0 * 2160.0);
    double framerate_factor = framerate / 24.0;
    
    return static_cast<size_t>(base_bitrate * resolution_factor * framerate_factor);
}

bool BRAWParser::is_high_dynamic_range_compatible(ColorScienceVersion version) {
    // Generation 4 and 5 have better HDR support
    return version >= ColorScienceVersion::GENERATION_4;
}

// Private implementation methods
bool BRAWParser::validate_braw_signature(const uint8_t* data, size_t size) const {
    if (!data || size < 4) return false;
    
    // Check for BRAW magic number
    uint32_t magic = data[0] | (data[1] << 8) | (data[2] << 16) | (data[3] << 24);
    return magic == MKTAG('B','R','A','W');
}

BRAWParser::BRAWCompressionRatio BRAWParser::identify_compression_from_header(const BRAWFrameHeader& header) const {
    return header.compression_ratio;
}

bool BRAWParser::parse_blackmagic_atom_hierarchy(const uint8_t* data, size_t size, 
                                                std::vector<BRAWMetadataAtom>& atoms) const {
    // Simplified atom parsing for BRAW metadata
    if (!data || size < 8) return false;
    
    size_t offset = 0;
    while (offset + 8 <= size) {
        BRAWMetadataAtom atom;
        
        // Read atom size (4 bytes, little-endian for BRAW)
        atom.atom_size = data[offset] | (data[offset+1] << 8) | 
                        (data[offset+2] << 16) | (data[offset+3] << 24);
        offset += 4;
        
        // Read atom type (4 bytes)
        atom.atom_type = data[offset] | (data[offset+1] << 8) | 
                        (data[offset+2] << 16) | (data[offset+3] << 24);
        offset += 4;
        
        // Validate atom size
        if (atom.atom_size < 8 || offset + atom.atom_size - 8 > size) {
            break;
        }
        
        // Copy atom data
        atom.atom_data.resize(atom.atom_size - 8);
        std::memcpy(atom.atom_data.data(), data + offset, atom.atom_size - 8);
        
        atoms.push_back(atom);
        offset += atom.atom_size - 8;
    }
    
    return !atoms.empty();
}

bool BRAWParser::analyze_braw_sensor_pattern(const uint8_t* frame_data, size_t data_size, 
                                            const BRAWFrameHeader& header) const {
    if (!frame_data || data_size < 128 || !header.is_valid) return false;
    
    // Basic validation of Bayer pattern structure
    return check_bayer_pattern_integrity(frame_data + 64, data_size - 64, header.color_format);
}

bool BRAWParser::validate_braw_compression_structure(const uint8_t* frame_data, size_t data_size) const {
    // Simplified compression structure validation
    return frame_data && data_size >= 64;
}

bool BRAWParser::check_bayer_pattern_integrity(const uint8_t* sensor_data, size_t data_size, 
                                              uint8_t expected_pattern) const {
    if (!sensor_data || data_size < 16) return false;
    
    // Basic pattern validation - actual implementation would be more sophisticated
    return expected_pattern < 4; // Valid Bayer pattern codes are 0-3
}

bool BRAWParser::detect_header_corruption(const BRAWFrameHeader& header) const {
    if (!header.is_valid) return true;
    
    // Validate header fields for reasonable values
    if (header.width == 0 || header.height == 0 || 
        header.width > 12288 || header.height > 8192) { // Max sensor sizes
        return true;
    }
    
    if (header.bit_depth < 8 || header.bit_depth > 16) {
        return true;
    }
    
    if (header.color_format > 3) { // Invalid Bayer pattern
        return true;
    }
    
    return false;
}

bool BRAWParser::detect_sensor_data_corruption(const uint8_t* frame_data, size_t data_size, 
                                              const BRAWFrameHeader& header) const {
    if (!frame_data || data_size <= 64 || !header.is_valid) return true;
    
    // Check if payload size is reasonable for given resolution and bit depth
    size_t expected_min_size = (static_cast<size_t>(header.active_width) * 
                               header.active_height * header.bit_depth) / (8 * 20); // Account for compression
    size_t expected_max_size = (static_cast<size_t>(header.active_width) * 
                               header.active_height * header.bit_depth) / 4; // Minimum compression
    
    size_t payload_size = data_size - 64; // Subtract header size
    return (payload_size < expected_min_size || payload_size > expected_max_size);
}

double BRAWParser::calculate_corruption_severity_braw(const std::vector<std::pair<int64_t, int64_t>>& corrupted_ranges, 
                                                     int64_t total_frames) const {
    if (total_frames == 0) return 0.0;
    
    int64_t corrupted_frames = 0;
    for (const auto& range : corrupted_ranges) {
        corrupted_frames += (range.second - range.first + 1);
    }
    
    return static_cast<double>(corrupted_frames) / total_frames;
}

double BRAWParser::calculate_expected_compression_ratio(BRAWCompressionRatio ratio) const {
    switch (ratio) {
        case BRAWCompressionRatio::RATIO_3_1: return 1.0 / 3.0;
        case BRAWCompressionRatio::RATIO_5_1: return 1.0 / 5.0;
        case BRAWCompressionRatio::RATIO_8_1: return 1.0 / 8.0;
        case BRAWCompressionRatio::RATIO_12_1: return 1.0 / 12.0;
        case BRAWCompressionRatio::RATIO_CONSTANT_QUALITY: return 1.0 / 6.0; // Variable
        case BRAWCompressionRatio::RATIO_CONSTANT_BITRATE: return 1.0 / 7.0; // Variable
        default: return 1.0 / 5.0;
    }
}