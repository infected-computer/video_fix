#ifndef BRAW_PARSER_H
#define BRAW_PARSER_H

#include "VideoRepair/VideoRepairEngine.h"
#include <memory>
#include <vector>
#include <unordered_map>
#include <cstdint>

extern "C" {
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
}

/**
 * @brief Blackmagic RAW (BRAW) format parser with comprehensive support
 * 
 * Supports all BRAW variants including:
 * - BRAW 3:1, 5:1, 8:1, 12:1 compression ratios
 * - Blackmagic Generation 4/5 Color Science
 * - Full resolution and windowed recordings
 * - Embedded LUTs and color metadata
 * 
 * Features:
 * - Frame-accurate parsing with embedded timecode
 * - Professional camera metadata extraction
 * - Color science and gamma curve preservation
 * - Lens and recording metadata support
 * - DaVinci Resolve integration metadata
 */
class BRAWParser : public VideoRepairEngine::FormatParser {
public:
    // BRAW compression ratios
    enum class BRAWCompressionRatio {
        RATIO_3_1 = 0,    // 3:1 - Highest quality
        RATIO_5_1 = 1,    // 5:1 - High quality
        RATIO_8_1 = 2,    // 8:1 - Medium quality
        RATIO_12_1 = 3,   // 12:1 - Lower quality (proxy)
        RATIO_CONSTANT_QUALITY = 4, // CQ mode
        RATIO_CONSTANT_BITRATE = 5, // CBR mode
        UNKNOWN = 255
    };

    // Blackmagic Color Science versions
    enum class ColorScienceVersion {
        GENERATION_1 = 1,
        GENERATION_2 = 2,
        GENERATION_3 = 3,
        GENERATION_4 = 4,
        GENERATION_5 = 5,
        UNKNOWN = 0
    };

    // BRAW frame header structure
    struct BRAWFrameHeader {
        uint32_t magic_number;        // 'BRAW' magic
        uint32_t frame_size;
        uint32_t frame_number;
        uint32_t timestamp_low;
        uint32_t timestamp_high;
        uint16_t width;
        uint16_t height;
        uint16_t sensor_width;        // Full sensor width
        uint16_t sensor_height;       // Full sensor height
        uint16_t active_width;        // Active recording area
        uint16_t active_height;
        uint8_t bit_depth;            // 10, 12, or 16 bit
        uint8_t color_format;         // RAW Bayer pattern
        BRAWCompressionRatio compression_ratio;
        ColorScienceVersion color_science;
        uint8_t gamma_curve;
        uint8_t color_space;
        float white_balance_kelvin;
        float tint_adjustment;
        float exposure_compensation;
        float iso_value;
        uint8_t reserved[32];
        
        // Derived information
        bool is_valid;
        double quality_factor;
        bool is_windowed_recording;
        uint64_t full_timestamp;
    };

    // BRAW metadata atom structure
    struct BRAWMetadataAtom {
        uint32_t atom_type;
        uint32_t atom_size;
        std::vector<uint8_t> atom_data;
        
        // Camera-specific metadata
        struct CameraMetadata {
            std::string camera_model;           // URSA Mini Pro 12K, etc.
            std::string camera_serial_number;
            std::string firmware_version;
            std::string camera_id;
            std::string camera_notes;
            
            // Recording settings
            struct RecordingSettings {
                std::string recording_format;   // BRAW 3:1, etc.
                std::string project_frame_rate;
                std::string sensor_frame_rate;
                std::string off_speed_frame_rate;
                bool high_frame_rate_mode;
                bool windowed_mode;
                uint16_t crop_left, crop_top, crop_right, crop_bottom;
                
                // Sensor settings
                std::string sensor_gain_mode;  // Extended highlights, etc.
                int iso_value;
                float shutter_angle;
                float shutter_speed_fraction;
                
                // Color settings
                ColorScienceVersion color_science_version;
                std::string gamma_curve;       // Blackmagic Design Film, Video, etc.
                std::string color_space;       // Rec. 2020, DCI-P3, etc.
                float white_balance_kelvin;
                float tint_adjustment;
                float exposure_compensation;
                
                // Post settings embedded in camera
                float lift_r, lift_g, lift_b, lift_y;
                float gamma_r, gamma_g, gamma_b, gamma_y;
                float gain_r, gain_g, gain_b, gain_y;
                float offset_r, offset_g, offset_b, offset_y;
                float contrast;
                float saturation;
                float hue_adjustment;
                
                // LUT settings
                bool lut_enabled;
                std::string lut_name;
                float lut_strength;
            } recording_settings;
            
            // Lens metadata
            struct LensMetadata {
                std::string lens_type;          // Manual, EF, PL, etc.
                std::string lens_model;
                std::string lens_serial_number;
                float focal_length_mm;
                float aperture_f_stop;
                float focus_distance_m;
                float zoom_position;
                bool image_stabilization_enabled;
                bool autofocus_enabled;
                
                // EF/Canon-specific
                std::string ef_lens_name;
                uint16_t ef_lens_id;
                
                // PL/Cinema-specific
                std::string pl_lens_manufacturer;
                std::string pl_lens_model;
            } lens_metadata;
            
        } camera_metadata;
        
        // Production metadata
        struct ProductionMetadata {
            std::string project_name;
            std::string scene_name;
            std::string shot_name;
            std::string take_number;
            std::string director;
            std::string cinematographer;
            std::string operator;
            std::string location;
            std::string production_notes;
            
            // Timecode information
            struct TimecodeMetadata {
                std::string timecode_start;
                std::string timecode_format;   // 23.976p, 24p, 25p, 29.97p, etc.
                std::string sync_source;       // Internal, External, etc.
                bool drop_frame_enabled;
                uint32_t frame_count;
            } timecode_metadata;
            
        } production_metadata;
        
        // Audio metadata (BRAW can contain audio tracks)
        struct AudioMetadata {
            bool has_audio;
            uint8_t channel_count;
            uint32_t sample_rate;
            uint8_t bit_depth;
            std::string audio_format;
            std::vector<std::string> channel_names;
            std::vector<float> audio_levels_db;
            bool phantom_power_enabled;
            
        } audio_metadata;
    };

public:
    explicit BRAWParser();
    ~BRAWParser() override = default;

    // FormatParser interface implementation
    bool can_parse(const std::string& file_path) const override;
    VideoRepairEngine::StreamInfo parse_stream_info(AVFormatContext* format_ctx, int stream_index) const override;
    bool detect_corruption(AVFormatContext* format_ctx, VideoRepairEngine::StreamInfo& stream_info) const override;
    std::vector<VideoRepairEngine::RepairTechnique> recommend_techniques(const VideoRepairEngine::StreamInfo& stream_info) const override;

    // BRAW-specific methods
    BRAWCompressionRatio detect_compression_ratio(const uint8_t* frame_data, size_t data_size) const;
    bool parse_braw_frame_header(const uint8_t* frame_data, size_t data_size, BRAWFrameHeader& header) const;
    std::vector<BRAWMetadataAtom> parse_braw_metadata(AVFormatContext* format_ctx) const;
    
    // Quality assessment
    double calculate_braw_quality_score(const BRAWFrameHeader& header, size_t actual_frame_size) const;
    bool validate_braw_frame_integrity(const uint8_t* frame_data, size_t data_size) const;
    
    // Color science methods
    ColorScienceVersion detect_color_science_version(const BRAWFrameHeader& header) const;
    std::string get_color_science_name(ColorScienceVersion version) const;
    bool supports_extended_video_levels(ColorScienceVersion version) const;
    
    // Compression analysis
    bool is_constant_quality_mode(BRAWCompressionRatio ratio) const;
    double get_expected_compression_efficiency(BRAWCompressionRatio ratio) const;
    size_t estimate_frame_size(const BRAWFrameHeader& header) const;
    
    // Metadata extraction
    bool extract_camera_settings(const std::vector<BRAWMetadataAtom>& atoms, VideoRepairEngine::StreamInfo& stream_info) const;
    bool extract_lens_metadata(const std::vector<BRAWMetadataAtom>& atoms, VideoRepairEngine::StreamInfo& stream_info) const;
    bool extract_production_metadata(const std::vector<BRAWMetadataAtom>& atoms, VideoRepairEngine::StreamInfo& stream_info) const;
    bool extract_timecode_from_braw(AVFormatContext* format_ctx, std::string& timecode) const;
    
    // Frame-level operations
    std::vector<std::pair<int64_t, int64_t>> find_corrupted_braw_frames(AVFormatContext* format_ctx) const;
    bool can_repair_braw_frame(const uint8_t* frame_data, size_t data_size) const;
    bool validate_braw_sensor_data(const uint8_t* frame_data, size_t data_size, const BRAWFrameHeader& header) const;
    
    // Professional workflow features
    bool requires_blackmagic_sdk(const VideoRepairEngine::StreamInfo& stream_info) const;
    std::string get_davinci_resolve_settings(const BRAWFrameHeader& header) const;
    bool extract_embedded_lut(const std::vector<BRAWMetadataAtom>& atoms, std::vector<uint8_t>& lut_data) const;
    
    // Repair recommendations
    std::vector<VideoRepairEngine::RepairTechnique> get_braw_specific_techniques(const VideoRepairEngine::StreamInfo& stream_info) const;
    bool supports_partial_frame_recovery(BRAWCompressionRatio ratio) const;
    
    // Utility methods
    static std::string compression_ratio_to_string(BRAWCompressionRatio ratio);
    static std::string color_science_to_string(ColorScienceVersion version);
    static size_t get_expected_bitrate(BRAWCompressionRatio ratio, int width, int height, double framerate);
    static bool is_high_dynamic_range_compatible(ColorScienceVersion version);

private:
    // Internal parsing methods
    bool validate_braw_signature(const uint8_t* data, size_t size) const;
    BRAWCompressionRatio identify_compression_from_header(const BRAWFrameHeader& header) const;
    bool parse_blackmagic_atom_hierarchy(const uint8_t* data, size_t size, std::vector<BRAWMetadataAtom>& atoms) const;
    
    // Frame structure analysis
    bool analyze_braw_sensor_pattern(const uint8_t* frame_data, size_t data_size, const BRAWFrameHeader& header) const;
    bool validate_braw_compression_structure(const uint8_t* frame_data, size_t data_size) const;
    bool check_bayer_pattern_integrity(const uint8_t* sensor_data, size_t data_size, uint8_t expected_pattern) const;
    
    // Corruption detection helpers
    bool detect_header_corruption(const BRAWFrameHeader& header) const;
    bool detect_sensor_data_corruption(const uint8_t* frame_data, size_t data_size, const BRAWFrameHeader& header) const;
    double calculate_corruption_severity_braw(const std::vector<std::pair<int64_t, int64_t>>& corrupted_ranges, int64_t total_frames) const;
    
    // Professional metadata parsers
    bool parse_camera_settings_atom(const BRAWMetadataAtom& atom, VideoRepairEngine::StreamInfo& stream_info) const;
    bool parse_lens_settings_atom(const BRAWMetadataAtom& atom, VideoRepairEngine::StreamInfo& stream_info) const;
    bool parse_production_atom(const BRAWMetadataAtom& atom, VideoRepairEngine::StreamInfo& stream_info) const;
    bool parse_color_metadata_atom(const BRAWMetadataAtom& atom, VideoRepairEngine::StreamInfo& stream_info) const;
    bool parse_audio_metadata_atom(const BRAWMetadataAtom& atom, VideoRepairEngine::StreamInfo& stream_info) const;
    
    // Quality metrics
    double calculate_expected_compression_ratio(BRAWCompressionRatio ratio) const;
    bool assess_sensor_data_quality(const uint8_t* frame_data, size_t data_size, const BRAWFrameHeader& header) const;
    double calculate_dynamic_range_utilization(const uint8_t* sensor_data, size_t data_size, uint8_t bit_depth) const;
    
    // Constants and lookup tables
    static const std::unordered_map<uint32_t, BRAWCompressionRatio> COMPRESSION_RATIO_CODES;
    static const std::unordered_map<BRAWCompressionRatio, std::string> COMPRESSION_RATIO_NAMES;
    static const std::unordered_map<BRAWCompressionRatio, size_t> COMPRESSION_RATIO_BITRATES;
    static const std::unordered_map<ColorScienceVersion, std::string> COLOR_SCIENCE_NAMES;
    static const std::unordered_map<uint8_t, std::string> GAMMA_CURVE_NAMES;
    static const std::unordered_map<uint8_t, std::string> COLOR_SPACE_NAMES;
    static const std::unordered_map<uint8_t, std::string> BAYER_PATTERN_NAMES;
    
    // Parser configuration
    struct ParserConfig {
        bool strict_validation = true;
        bool extract_all_metadata = true;
        bool deep_sensor_analysis = false;
        bool extract_embedded_luts = true;
        double corruption_threshold = 0.03;
        size_t max_frames_to_analyze = 50;
        bool blackmagic_sdk_available = false;
    } m_config;
};

/**
 * @brief BRAW frame processor for validation and repair
 */
class BRAWFrameProcessor {
public:
    explicit BRAWFrameProcessor(const BRAWParser::BRAWCompressionRatio ratio, 
                               const BRAWParser::ColorScienceVersion color_science);
    
    // Frame validation
    bool validate_frame(const uint8_t* frame_data, size_t data_size) const;
    std::vector<std::string> get_validation_errors(const uint8_t* frame_data, size_t data_size) const;
    
    // Frame repair
    bool can_repair_frame(const uint8_t* frame_data, size_t data_size) const;
    std::vector<uint8_t> repair_frame(const uint8_t* frame_data, size_t data_size) const;
    
    // Sensor data reconstruction
    std::vector<uint8_t> reconstruct_frame_header(const BRAWParser::BRAWFrameHeader& template_header) const;
    bool repair_sensor_data_structure(std::vector<uint8_t>& frame_data) const;
    bool interpolate_corrupted_sensor_pixels(std::vector<uint8_t>& sensor_data, 
                                           const std::vector<std::pair<int, int>>& corrupted_pixels) const;
    
    // Color science validation
    bool validate_color_science_consistency(const uint8_t* frame_data, size_t data_size) const;
    bool repair_color_science_metadata(std::vector<uint8_t>& frame_data) const;

private:
    BRAWParser::BRAWCompressionRatio m_compression_ratio;
    BRAWParser::ColorScienceVersion m_color_science;
    
    // Internal repair methods
    bool repair_frame_header(std::vector<uint8_t>& frame_data) const;
    bool repair_sensor_metadata(std::vector<uint8_t>& frame_data) const;
    bool repair_compression_structure(std::vector<uint8_t>& frame_data) const;
    
    // Validation helpers
    bool validate_header_checksum(const uint8_t* header_data, size_t header_size) const;
    bool validate_sensor_data_structure(const uint8_t* frame_data, size_t data_size) const;
    bool validate_compression_integrity(const uint8_t* frame_data, size_t data_size) const;
    
    // Sensor-specific repairs
    bool repair_bayer_pattern_errors(std::vector<uint8_t>& sensor_data, uint8_t pattern_type) const;
    bool interpolate_missing_sensor_lines(std::vector<uint8_t>& sensor_data, int width, int height, int bit_depth) const;
};

/**
 * @brief BRAW metadata extractor for professional workflows
 */
class BRAWMetadataExtractor {
public:
    struct ComprehensiveMetadata {
        // Camera information
        std::string camera_manufacturer = "Blackmagic Design";
        std::string camera_model;
        std::string camera_serial_number;
        std::string firmware_version;
        std::string camera_id;
        
        // Recording specifications
        std::string recording_format;
        std::string compression_ratio_name;
        BRAWParser::BRAWCompressionRatio compression_ratio;
        BRAWParser::ColorScienceVersion color_science_version;
        std::string color_science_name;
        
        // Sensor information
        uint16_t sensor_width, sensor_height;
        uint16_t active_width, active_height;
        uint8_t bit_depth;
        std::string bayer_pattern;
        bool windowed_recording;
        
        // Color grading settings (embedded from camera)
        struct ColorGradingSettings {
            std::string gamma_curve;
            std::string color_space;
            float white_balance_kelvin;
            float tint_adjustment;
            float exposure_compensation;
            
            // Primary color correction
            float lift_r, lift_g, lift_b, lift_y;
            float gamma_r, gamma_g, gamma_b, gamma_y;
            float gain_r, gain_g, gain_b, gain_y;
            float offset_r, offset_g, offset_b, offset_y;
            float contrast, saturation, hue_adjustment;
            
            // LUT information
            bool lut_applied;
            std::string lut_name;
            float lut_strength;
            std::vector<uint8_t> embedded_lut_data;
        } color_grading;
        
        // Lens information
        struct LensInformation {
            std::string lens_type;
            std::string lens_model;
            std::string lens_serial_number;
            float focal_length_mm;
            float aperture_f_stop;
            float focus_distance_m;
            std::string lens_notes;
        } lens_info;
        
        // Production metadata
        struct ProductionInformation {
            std::string project_name;
            std::string scene_name;
            std::string shot_name;
            std::string take_number;
            std::string director;
            std::string cinematographer;
            std::string camera_operator;
            std::string location;
            std::string notes;
        } production_info;
        
        // Technical metadata
        struct TechnicalInformation {
            std::string project_frame_rate;
            std::string sensor_frame_rate;
            bool high_frame_rate_mode;
            int iso_value;
            float shutter_angle;
            std::string timecode_start;
            std::string timecode_format;
            bool drop_frame_timecode;
        } technical_info;
        
        // Audio metadata (if present)
        struct AudioInformation {
            bool has_audio;
            uint8_t channel_count;
            uint32_t sample_rate;
            uint8_t bit_depth;
            std::vector<std::string> channel_names;
        } audio_info;
        
        // Custom metadata
        std::unordered_map<std::string, std::string> custom_fields;
    };
    
    // Extract comprehensive metadata
    ComprehensiveMetadata extract_comprehensive_metadata(AVFormatContext* format_ctx) const;
    bool validate_metadata_integrity(const ComprehensiveMetadata& metadata) const;
    
    // Specific extractors for different camera models
    bool extract_ursa_mini_metadata(const std::vector<BRAWParser::BRAWMetadataAtom>& atoms, ComprehensiveMetadata& metadata) const;
    bool extract_ursa_mini_pro_metadata(const std::vector<BRAWParser::BRAWMetadataAtom>& atoms, ComprehensiveMetadata& metadata) const;
    bool extract_ursa_broadcast_metadata(const std::vector<BRAWParser::BRAWMetadataAtom>& atoms, ComprehensiveMetadata& metadata) const;
    bool extract_pocket_cinema_metadata(const std::vector<BRAWParser::BRAWMetadataAtom>& atoms, ComprehensiveMetadata& metadata) const;
    
    // DaVinci Resolve integration
    std::string generate_davinci_resolve_xml(const ComprehensiveMetadata& metadata) const;
    std::string generate_davinci_resolve_project_settings(const ComprehensiveMetadata& metadata) const;
    bool export_color_grading_settings(const ComprehensiveMetadata& metadata, const std::string& output_path) const;
    
    // Professional workflow exports
    std::string serialize_to_xml(const ComprehensiveMetadata& metadata) const;
    std::string serialize_to_json(const ComprehensiveMetadata& metadata) const;
    bool export_to_edl(const ComprehensiveMetadata& metadata, const std::string& output_path) const;
    bool export_to_ale(const ComprehensiveMetadata& metadata, const std::string& output_path) const;
    bool export_embedded_lut(const ComprehensiveMetadata& metadata, const std::string& output_path) const;

private:
    // Internal extraction methods
    bool parse_camera_metadata_atom(const uint8_t* atom_data, size_t atom_size, ComprehensiveMetadata& metadata) const;
    bool parse_lens_metadata_atom(const uint8_t* atom_data, size_t atom_size, ComprehensiveMetadata& metadata) const;
    bool parse_color_grading_atom(const uint8_t* atom_data, size_t atom_size, ComprehensiveMetadata& metadata) const;
    bool parse_production_metadata_atom(const uint8_t* atom_data, size_t atom_size, ComprehensiveMetadata& metadata) const;
    bool parse_technical_metadata_atom(const uint8_t* atom_data, size_t atom_size, ComprehensiveMetadata& metadata) const;
    bool parse_audio_metadata_atom(const uint8_t* atom_data, size_t atom_size, ComprehensiveMetadata& metadata) const;
    
    // Validation helpers
    bool validate_camera_model(const std::string& model) const;
    bool validate_color_science_version(BRAWParser::ColorScienceVersion version) const;
    bool validate_compression_settings(BRAWParser::BRAWCompressionRatio ratio, int width, int height) const;
    bool validate_lens_parameters(float focal_length, float aperture, float focus_distance) const;
    
    // Utility methods
    std::string format_timecode(uint32_t frame_count, float frame_rate, bool drop_frame) const;
    std::string kelvin_to_color_temperature_name(float kelvin) const;
};

#endif // BRAW_PARSER_H