#ifndef PRORES_PARSER_H
#define PRORES_PARSER_H

#include "VideoRepair/VideoRepairEngine.h"
#include <memory>
#include <vector>
#include <unordered_map>

extern "C" {
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
}

/**
 * @brief Professional ProRes format parser with comprehensive support
 * 
 * Supports all ProRes variants including:
 * - ProRes 422 Proxy, LT, Standard, HQ
 * - ProRes 4444 and 4444 XQ
 * - ProRes RAW and RAW HQ
 * 
 * Features:
 * - Frame-accurate parsing
 * - Professional metadata extraction
 * - Corruption detection and repair strategies
 * - Timecode and camera metadata support
 * - Color space and gamma curve preservation
 */
class ProResParser : public VideoRepairEngine::FormatParser {
public:
    // ProRes codec identifiers
    enum class ProResVariant {
        PROXY = 0,      // 'apco' - ProRes 422 Proxy
        LT = 1,         // 'apcs' - ProRes 422 LT  
        STANDARD = 2,   // 'apcn' - ProRes 422 Standard
        HQ = 3,         // 'apch' - ProRes 422 HQ
        PRORES_4444 = 4,     // 'ap4h' - ProRes 4444
        PRORES_4444_XQ = 5,  // 'ap4x' - ProRes 4444 XQ
        RAW = 6,        // 'aprn' - ProRes RAW
        RAW_HQ = 7,     // 'aprh' - ProRes RAW HQ
        UNKNOWN = 255
    };
    
    // ProRes frame header structure
    struct ProResFrameHeader {
        uint32_t frame_size;
        uint8_t frame_identifier[4];  // 'icpf'
        uint16_t header_size;
        uint8_t version;
        uint8_t encoder_identifier[4];
        uint16_t width;
        uint16_t height;
        uint8_t chroma_format;        // 2=422, 3=444
        uint8_t interlaced_mode;
        uint8_t aspect_ratio_info;
        uint8_t framerate_code;
        uint8_t color_primaries;
        uint8_t transfer_characteristics;
        uint8_t matrix_coefficients;
        uint8_t source_format;
        uint8_t alpha_channel_type;
        uint8_t reserved[3];
        
        // Derived information
        ProResVariant variant;
        bool is_valid;
        double quality_factor;
        size_t expected_bitrate;
    };
    
    // ProRes container atom structure
    struct ProResAtom {
        uint32_t size;
        uint32_t type;
        std::vector<uint8_t> data;
        
        // Professional metadata atoms
        struct {
            std::string timecode;
            std::string reel_name;
            std::string camera_model;
            std::string camera_serial;
            std::string lens_model;
            std::string lens_serial;
            std::string operator_name;
            std::string director_name;
            std::string production_name;
            std::unordered_map<std::string, std::string> custom_metadata;
        } professional_metadata;
        
        // Color correction metadata
        struct {
            double lift_r, lift_g, lift_b;
            double gamma_r, gamma_g, gamma_b;
            double gain_r, gain_g, gain_b;
            double saturation;
            double contrast;
            double brightness;
            std::string lut_name;
            bool has_color_correction;
        } color_metadata;
    };

public:
    explicit ProResParser();
    ~ProResParser() override = default;

    // FormatParser interface implementation
    bool can_parse(const std::string& file_path) const override;
    VideoRepairEngine::StreamInfo parse_stream_info(AVFormatContext* format_ctx, int stream_index) const override;
    bool detect_corruption(AVFormatContext* format_ctx, VideoRepairEngine::StreamInfo& stream_info) const override;
    std::vector<VideoRepairEngine::RepairTechnique> recommend_techniques(const VideoRepairEngine::StreamInfo& stream_info) const override;

    // ProRes-specific methods
    ProResVariant detect_prores_variant(const uint8_t* frame_data, size_t data_size) const;
    bool parse_prores_frame_header(const uint8_t* frame_data, size_t data_size, ProResFrameHeader& header) const;
    std::vector<ProResAtom> parse_professional_metadata(AVFormatContext* format_ctx) const;
    
    // Quality assessment
    double calculate_prores_quality_score(const ProResFrameHeader& header, size_t actual_frame_size) const;
    bool validate_prores_frame_integrity(const uint8_t* frame_data, size_t data_size) const;
    
    // Repair recommendations
    std::vector<VideoRepairEngine::RepairTechnique> get_prores_specific_techniques(const VideoRepairEngine::StreamInfo& stream_info) const;
    bool requires_professional_tools(const VideoRepairEngine::StreamInfo& stream_info) const;
    
    // Metadata extraction
    bool extract_timecode_from_prores(AVFormatContext* format_ctx, std::string& timecode) const;
    bool extract_camera_metadata(const std::vector<ProResAtom>& atoms, VideoRepairEngine::StreamInfo& stream_info) const;
    bool extract_color_metadata(const std::vector<ProResAtom>& atoms, VideoRepairEngine::StreamInfo& stream_info) const;
    
    // Frame-level operations
    std::vector<std::pair<int64_t, int64_t>> find_corrupted_prores_frames(AVFormatContext* format_ctx) const;
    bool can_repair_prores_frame(const uint8_t* frame_data, size_t data_size) const;
    
    // Professional features
    bool supports_alpha_channel(ProResVariant variant) const;
    bool supports_12bit_depth(ProResVariant variant) const;
    std::string get_recommended_encoder_settings(ProResVariant variant) const;
    
    // Utility methods
    static std::string variant_to_string(ProResVariant variant);
    static std::string get_prores_fourcc(ProResVariant variant);
    static size_t get_expected_bitrate(ProResVariant variant, int width, int height, double framerate);

private:
    // Internal parsing methods
    bool validate_prores_signature(const uint8_t* data, size_t size) const;
    ProResVariant identify_variant_from_fourcc(const uint8_t fourcc[4]) const;
    bool parse_atom_hierarchy(const uint8_t* data, size_t size, std::vector<ProResAtom>& atoms) const;
    
    // Frame structure analysis
    bool analyze_prores_macroblock_structure(const uint8_t* frame_data, size_t data_size) const;
    bool validate_prores_slice_structure(const uint8_t* frame_data, size_t data_size) const;
    
    // Corruption detection helpers
    bool detect_header_corruption(const ProResFrameHeader& header) const;
    bool detect_payload_corruption(const uint8_t* frame_data, size_t data_size, const ProResFrameHeader& header) const;
    double calculate_corruption_severity(const std::vector<std::pair<int64_t, int64_t>>& corrupted_ranges, int64_t total_frames) const;
    
    // Professional metadata parsers
    bool parse_camera_atom(const ProResAtom& atom, VideoRepairEngine::StreamInfo& stream_info) const;
    bool parse_lens_atom(const ProResAtom& atom, VideoRepairEngine::StreamInfo& stream_info) const;
    bool parse_production_atom(const ProResAtom& atom, VideoRepairEngine::StreamInfo& stream_info) const;
    bool parse_color_atom(const ProResAtom& atom, VideoRepairEngine::StreamInfo& stream_info) const;
    
    // Quality metrics
    double calculate_expected_compression_ratio(ProResVariant variant) const;
    bool assess_visual_quality_indicators(const uint8_t* frame_data, size_t data_size) const;
    
    // Constants and lookup tables
    static const std::unordered_map<uint32_t, ProResVariant> FOURCC_TO_VARIANT;
    static const std::unordered_map<ProResVariant, std::string> VARIANT_NAMES;
    static const std::unordered_map<ProResVariant, size_t> VARIANT_BITRATES;
    static const std::unordered_map<uint8_t, std::string> FRAMERATE_CODES;
    static const std::unordered_map<uint8_t, std::string> CHROMA_FORMATS;
    
    // Parser configuration
    struct ParserConfig {
        bool strict_validation = true;
        bool extract_all_metadata = true;
        bool deep_frame_analysis = false;
        double corruption_threshold = 0.05;
        size_t max_frames_to_analyze = 100;
    } m_config;
};

/**
 * @brief ProRes frame validator and repairer
 */
class ProResFrameProcessor {
public:
    explicit ProResFrameProcessor(const ProResParser::ProResVariant variant);
    
    // Frame validation
    bool validate_frame(const uint8_t* frame_data, size_t data_size) const;
    std::vector<std::string> get_validation_errors(const uint8_t* frame_data, size_t data_size) const;
    
    // Frame repair
    bool can_repair_frame(const uint8_t* frame_data, size_t data_size) const;
    std::vector<uint8_t> repair_frame(const uint8_t* frame_data, size_t data_size) const;
    
    // Frame reconstruction
    std::vector<uint8_t> reconstruct_frame_header(const ProResParser::ProResFrameHeader& template_header) const;
    bool repair_macroblock_structure(std::vector<uint8_t>& frame_data) const;
    
private:
    ProResParser::ProResVariant m_variant;
    
    // Internal repair methods
    bool repair_frame_header(std::vector<uint8_t>& frame_data) const;
    bool repair_slice_headers(std::vector<uint8_t>& frame_data) const;
    bool repair_macroblock_data(std::vector<uint8_t>& frame_data) const;
    
    // Validation helpers
    bool validate_header_checksum(const uint8_t* header_data, size_t header_size) const;
    bool validate_slice_structure(const uint8_t* frame_data, size_t data_size) const;
    bool validate_quantization_matrices(const uint8_t* frame_data, size_t data_size) const;
};

/**
 * @brief ProRes metadata extractor for professional workflows
 */
class ProResMetadataExtractor {
public:
    struct ProfessionalMetadata {
        // Production information
        std::string project_name;
        std::string scene_number;
        std::string take_number;
        std::string reel_name;
        std::string tape_name;
        
        // Camera information
        std::string camera_manufacturer;
        std::string camera_model;
        std::string camera_serial_number;
        std::string firmware_version;
        
        // Lens information
        std::string lens_manufacturer;
        std::string lens_model;
        std::string lens_serial_number;
        double focal_length_mm;
        double aperture_f_stop;
        double focus_distance_m;
        
        // Recording settings
        std::string recording_format;
        std::string gamma_curve;
        std::string color_space;
        std::string white_balance_preset;
        int iso_sensitivity;
        double shutter_angle;
        
        // Timecode information
        std::string timecode_start;
        std::string timecode_format;
        std::string sync_source;
        
        // Crew information
        std::string director_of_photography;
        std::string camera_operator;
        std::string focus_puller;
        std::string script_supervisor;
        
        // Post-production metadata
        std::string colorist_notes;
        std::string editorial_notes;
        std::string workflow_notes;
        bool requires_special_handling;
        
        // Custom metadata
        std::unordered_map<std::string, std::string> custom_fields;
    };
    
    // Extract comprehensive metadata
    ProfessionalMetadata extract_professional_metadata(AVFormatContext* format_ctx) const;
    bool validate_metadata_integrity(const ProfessionalMetadata& metadata) const;
    
    // Specific extractors
    bool extract_arri_metadata(const std::vector<ProResParser::ProResAtom>& atoms, ProfessionalMetadata& metadata) const;
    bool extract_red_metadata(const std::vector<ProResParser::ProResAtom>& atoms, ProfessionalMetadata& metadata) const;
    bool extract_blackmagic_metadata(const std::vector<ProResParser::ProResAtom>& atoms, ProfessionalMetadata& metadata) const;
    bool extract_sony_metadata(const std::vector<ProResParser::ProResAtom>& atoms, ProfessionalMetadata& metadata) const;
    bool extract_canon_metadata(const std::vector<ProResParser::ProResAtom>& atoms, ProfessionalMetadata& metadata) const;
    
    // Metadata serialization
    std::string serialize_to_xml(const ProfessionalMetadata& metadata) const;
    std::string serialize_to_json(const ProfessionalMetadata& metadata) const;
    bool export_to_edl(const ProfessionalMetadata& metadata, const std::string& output_path) const;
    bool export_to_ale(const ProfessionalMetadata& metadata, const std::string& output_path) const;

private:
    // Internal extraction methods
    bool parse_production_atom(const uint8_t* atom_data, size_t atom_size, ProfessionalMetadata& metadata) const;
    bool parse_camera_atom(const uint8_t* atom_data, size_t atom_size, ProfessionalMetadata& metadata) const;
    bool parse_lens_atom(const uint8_t* atom_data, size_t atom_size, ProfessionalMetadata& metadata) const;
    bool parse_crew_atom(const uint8_t* atom_data, size_t atom_size, ProfessionalMetadata& metadata) const;
    bool parse_timecode_atom(const uint8_t* atom_data, size_t atom_size, ProfessionalMetadata& metadata) const;
    
    // Validation helpers
    bool validate_timecode_format(const std::string& timecode) const;
    bool validate_camera_serial(const std::string& serial) const;
    bool validate_lens_parameters(double focal_length, double aperture, double focus_distance) const;
};

#endif // PRORES_PARSER_H