#ifndef PROFESSIONAL_FORMATS_H
#define PROFESSIONAL_FORMATS_H

#include "AdvancedVideoRepairEngine.h"
#include <memory>
#include <unordered_map>
#include <vector>

namespace AdvancedVideoRepair {

/**
 * @brief Professional Video Format Support
 * 
 * Comprehensive support for professional video formats with deep
 * understanding of their internal structures:
 * 
 * - Apple ProRes (all variants)
 * - Blackmagic RAW (BRAW)
 * - RED R3D RAW
 * - ARRI ALEXA formats
 * - Sony XAVC/Venice
 * - Canon Cinema RAW
 * - Panasonic P2/MXF
 * - DNxHD/DNxHR
 * - CinemaDNG sequences
 */

enum class ProfessionalFormat {
    // Apple ProRes family
    PRORES_422_PROXY,
    PRORES_422_LT,
    PRORES_422,
    PRORES_422_HQ,
    PRORES_4444,
    PRORES_4444_XQ,
    PRORES_RAW,
    PRORES_RAW_HQ,
    
    // Blackmagic Design
    BLACKMAGIC_RAW_3_1,
    BLACKMAGIC_RAW_5_1,
    BLACKMAGIC_RAW_8_1,
    BLACKMAGIC_RAW_12_1,
    BLACKMAGIC_RAW_Q0,
    BLACKMAGIC_RAW_Q5,
    
    // RED Digital Cinema
    RED_R3D_REDCODE_36,
    RED_R3D_REDCODE_60,
    RED_R3D_REDCODE_120,
    RED_R3D_REDCODE_240,
    RED_HELIUM_8K,
    RED_MONSTRO_8K,
    RED_DRAGON_6K,
    
    // ARRI
    ARRI_ALEXA_ARRIRAW,
    ARRI_ALEXA_PRORES,
    ARRI_ALEXA_MINI_ARRIRAW,
    ARRI_ALEXA_LF_ARRIRAW,
    ARRI_ALEXA_65_ARRIRAW,
    
    // Sony Professional
    SONY_XAVC_INTRA_CBG,
    SONY_XAVC_LONG_GOP,
    SONY_VENICE_RAW,
    SONY_FX9_XAVC,
    SONY_F55_RAW,
    SONY_F65_RAW,
    
    // Canon Cinema
    CANON_CINEMA_RAW_LIGHT,
    CANON_CINEMA_RAW_LT,
    CANON_XF_AVC_INTRA,
    CANON_XF_AVC_LONG_GOP,
    CANON_C300_MXF,
    CANON_C500_4K_RAW,
    
    // Panasonic
    PANASONIC_P2_MXF_AVC_INTRA,
    PANASONIC_P2_MXF_DVCPRO,
    PANASONIC_VARICAM_RAW,
    PANASONIC_GH5_ALL_INTRA,
    
    // Avid
    AVID_DNXHD_36,
    AVID_DNXHD_120,
    AVID_DNXHD_185,
    AVID_DNXHR_LB,
    AVID_DNXHR_SQ,
    AVID_DNXHR_HQ,
    AVID_DNXHR_HQX,
    AVID_DNXHR_444,
    
    // CinemaDNG
    CINEMA_DNG_COMPRESSED,
    CINEMA_DNG_UNCOMPRESSED,
    CINEMA_DNG_LOSSLESS,
    
    // MXF Variants
    MXF_OP1A_SINGLE_TRACK,
    MXF_OP_ATOM_SINGLE_FILE,
    MXF_AS02_VERSIONING,
    MXF_AS11_UK_DPP,
    MXF_AS03_PODCAST,
    
    UNKNOWN_PROFESSIONAL
};

/**
 * @brief Professional Format Metadata Structure
 */
struct ProfessionalMetadata {
    // Technical specifications
    std::string codec_name;
    std::string profile_level;
    int bit_depth = 0;
    std::string color_space;
    std::string color_primaries;
    std::string transfer_characteristics;
    std::string chroma_subsampling;
    
    // Camera/Recording metadata
    std::string camera_manufacturer;
    std::string camera_model;
    std::string firmware_version;
    std::string lens_model;
    std::string recording_format;
    std::string project_name;
    std::string scene_name;
    std::string take_number;
    
    // Professional workflow metadata
    std::string timecode_start;
    std::string timecode_format;
    double frame_rate_exact = 0.0;
    std::string aspect_ratio;
    std::string reel_name;
    std::string operator_name;
    std::string director_name;
    std::string production_company;
    
    // Color grading metadata
    std::string lut_applied;
    std::string color_temperature;
    std::string tint;
    std::string exposure_compensation;
    std::string gamma_curve;
    bool rec2020_container = false;
    bool hdr_metadata_present = false;
    
    // Custom metadata fields
    std::unordered_map<std::string, std::string> custom_fields;
};

/**
 * @brief Professional Format Analyzer and Processor
 */
class ProfessionalFormatProcessor {
public:
    explicit ProfessionalFormatProcessor();
    ~ProfessionalFormatProcessor();
    
    bool initialize();
    void shutdown();
    
    // Format detection and analysis
    ProfessionalFormat detect_professional_format(const std::string& file_path);
    ProfessionalMetadata extract_professional_metadata(const std::string& file_path);
    
    // Format-specific corruption analysis
    CorruptionAnalysis analyze_professional_corruption(
        const std::string& file_path,
        ProfessionalFormat format
    );
    
    // Professional format repair
    bool repair_professional_format(
        const std::string& input_file,
        const std::string& output_file,
        ProfessionalFormat format,
        const RepairStrategy& strategy
    );
    
    // Metadata preservation and restoration
    bool preserve_professional_metadata(
        const std::string& source_file,
        const std::string& target_file,
        ProfessionalFormat format
    );
    
    // Format conversion with quality preservation
    bool convert_professional_format(
        const std::string& input_file,
        const std::string& output_file,
        ProfessionalFormat input_format,
        ProfessionalFormat output_format,
        const RepairStrategy& strategy
    );
    
    // Validation and compliance checking
    bool validate_professional_compliance(
        const std::string& file_path,
        ProfessionalFormat format,
        std::vector<std::string>& compliance_issues
    );

private:
    // Format-specific processors
    class ProResProcessor;
    class BlackmagicRAWProcessor;
    class REDProcessor;
    class ARRIProcessor;
    class SonyProcessor;
    class CanonProcessor;
    class PanasonicProcessor;
    class DNxProcessor;
    class CinemaDNGProcessor;
    class MXFProcessor;
    
    bool m_initialized = false;
    
    std::unique_ptr<ProResProcessor> m_prores_processor;
    std::unique_ptr<BlackmagicRAWProcessor> m_braw_processor;
    std::unique_ptr<REDProcessor> m_red_processor;
    std::unique_ptr<ARRIProcessor> m_arri_processor;
    std::unique_ptr<SonyProcessor> m_sony_processor;
    std::unique_ptr<CanonProcessor> m_canon_processor;
    std::unique_ptr<PanasonicProcessor> m_panasonic_processor;
    std::unique_ptr<DNxProcessor> m_dnx_processor;
    std::unique_ptr<CinemaDNGProcessor> m_cinemadng_processor;
    std::unique_ptr<MXFProcessor> m_mxf_processor;
    
    // Format detection utilities
    ProfessionalFormat detect_from_file_signature(const std::string& file_path);
    ProfessionalFormat detect_from_metadata(const std::string& file_path);
    ProfessionalFormat detect_from_codec_analysis(AVFormatContext* format_ctx);
    
    // Utility methods
    bool is_raw_format(ProfessionalFormat format);
    bool requires_proprietary_sdk(ProfessionalFormat format);
    std::string get_format_description(ProfessionalFormat format);
};

/**
 * @brief Apple ProRes Processor with Deep Understanding
 */
class ProResProcessor {
public:
    explicit ProResProcessor();
    ~ProResProcessor();
    
    bool can_process(const std::string& file_path);
    
    // ProRes-specific analysis
    struct ProResInfo {
        int profile_level;          // 0=Proxy, 1=LT, 2=422, 3=HQ, 4=4444, 5=4444XQ
        int frame_size_bytes;
        bool alpha_channel_present;
        std::string vendor_fourcc;
        int horizontal_resolution;
        int vertical_resolution;
        std::string chroma_format;
        std::string bit_depth_per_component;
        std::string color_primaries;
        std::string transfer_characteristics;
        std::string matrix_coefficients;
    };
    
    ProResInfo analyze_prores_stream(AVCodecContext* codec_ctx);
    
    // ProRes frame-level repair
    bool repair_prores_frame(
        const uint8_t* corrupted_frame_data,
        size_t frame_size,
        uint8_t* repaired_frame_data,
        const ProResInfo& stream_info
    );
    
    // ProRes-specific corruption detection
    std::vector<std::pair<int64_t, int64_t>> detect_prores_corruption(
        const std::string& file_path
    );

private:
    // ProRes frame structure parsing
    struct ProResFrameHeader {
        uint32_t frame_size;
        uint16_t frame_identifier;  // "icpf"
        uint16_t header_size;
        uint8_t version;
        uint8_t encoder_identifier[4];
        uint16_t horizontal_size;
        uint16_t vertical_size;
        uint8_t chroma_format;
        uint8_t interlaced_mode;
        uint8_t aspect_ratio_info;
        uint8_t framerate_code;
        uint8_t color_primaries;
        uint8_t transfer_characteristics;
        uint8_t matrix_coefficients;
        uint8_t source_format;
        uint8_t alpha_channel_type;
        uint8_t reserved[3];
    };
    
    bool parse_prores_frame_header(const uint8_t* data, ProResFrameHeader& header);
    bool validate_prores_frame_structure(const uint8_t* data, size_t size);
    bool repair_prores_header(ProResFrameHeader& header);
    
    // ProRes slice and macroblock handling
    bool parse_prores_slices(const uint8_t* frame_data, const ProResFrameHeader& header);
    bool repair_corrupted_slice(uint8_t* slice_data, size_t slice_size);
};

/**
 * @brief Blackmagic RAW (BRAW) Processor
 */
class BlackmagicRAWProcessor {
public:
    explicit BlackmagicRAWProcessor();
    ~BlackmagicRAWProcessor();
    
    bool initialize();  // May require Blackmagic SDK
    void shutdown();
    
    bool can_process(const std::string& file_path);
    
    // BRAW-specific analysis
    struct BRAWInfo {
        int compression_ratio;      // 3:1, 5:1, 8:1, 12:1, Q0, Q5
        int resolution_width;
        int resolution_height;
        std::string color_science;  // Gen 4, Gen 5, etc.
        float iso_value;
        float white_balance;
        float tint;
        float exposure;
        std::string gamma_curve;
        std::string color_space;
        bool partial_debayer;
        std::string lens_metadata;
        std::string camera_metadata;
    };
    
    BRAWInfo analyze_braw_file(const std::string& file_path);
    
    // BRAW repair operations
    bool repair_braw_metadata(const std::string& file_path);
    bool repair_braw_frame_data(const std::string& file_path, int frame_number);
    
    // BRAW to standard format conversion for repair
    bool decode_braw_frame(
        const std::string& file_path,
        int frame_number,
        cv::Mat& decoded_frame,
        const BRAWInfo& settings
    );

private:
    bool m_sdk_available = false;
    void* m_braw_sdk_handle = nullptr;  // Opaque handle to SDK
    
    // BRAW file structure understanding
    struct BRAWClipInfo {
        uint32_t width;
        uint32_t height;
        uint32_t frame_count;
        double frame_rate;
        std::string clip_name;
        std::string camera_type;
        std::string firmware_version;
    };
    
    bool load_blackmagic_sdk();
    void unload_blackmagic_sdk();
    bool parse_braw_clip_info(const std::string& file_path, BRAWClipInfo& info);
};

/**
 * @brief RED R3D Processor with RED SDK Integration
 */
class REDProcessor {
public:
    explicit REDProcessor();
    ~REDProcessor();
    
    bool initialize();  // Requires RED SDK
    void shutdown();
    
    bool can_process(const std::string& file_path);
    
    // RED-specific analysis
    struct REDInfo {
        int redcode_setting;        // 2:1, 3:1, 5:1, 7:1, 9:1, 12:1, 18:1, 22:1
        std::string camera_type;    // WEAPON, EPIC, SCARLET, etc.
        std::string sensor_type;    // DRAGON, HELIUM, MONSTRO
        int recording_resolution;   // 2K, 4K, 5K, 6K, 8K
        float iso_value;
        float kelvin;
        float tint;
        float exposure;
        std::string lens_mount;
        std::string lens_model;
        std::string project_fps;
        std::string timecode;
        std::vector<std::string> audio_tracks;
    };
    
    REDInfo analyze_red_file(const std::string& file_path);
    
    // RED repair operations
    bool repair_red_metadata(const std::string& file_path);
    bool validate_red_structure(const std::string& file_path);
    
    // RED frame extraction for repair
    bool decode_red_frame(
        const std::string& file_path,
        int frame_number,
        cv::Mat& decoded_frame,
        const REDInfo& decode_settings
    );

private:
    bool m_red_sdk_available = false;
    void* m_red_sdk_handle = nullptr;
    
    struct REDClipInfo {
        uint32_t video_tracks;
        uint32_t audio_tracks;
        uint32_t total_frames;
        double frame_rate;
        std::string reel_name;
        std::string camera_serial;
    };
    
    bool load_red_sdk();
    void unload_red_sdk();
    bool parse_red_clip_structure(const std::string& file_path, REDClipInfo& info);
};

/**
 * @brief ARRI Format Processor (ARRIRAW, ALEXA formats)
 */
class ARRIProcessor {
public:
    explicit ARRIProcessor();
    ~ARRIProcessor();
    
    bool can_process(const std::string& file_path);
    
    // ARRI-specific analysis
    struct ARRIInfo {
        std::string camera_model;   // ALEXA, ALEXA Mini, ALEXA LF, ALEXA 65
        std::string recording_format; // ARRIRAW, ProRes
        int sensor_fps;
        int project_fps;
        float exposure_index;
        std::string color_temperature;
        std::string lens_data;
        std::string user_metadata;
        bool anamorphic_desqueeze;
        std::string look_name;
        std::string cdl_values;
    };
    
    ARRIInfo analyze_arri_file(const std::string& file_path);
    
    // ARRI repair operations
    bool repair_arriraw_structure(const std::string& file_path);
    bool validate_arri_metadata(const std::string& file_path);
    
private:
    // ARRIRAW structure understanding
    bool parse_arriraw_header(const std::string& file_path);
    bool validate_arriraw_frame_structure(const uint8_t* frame_data, size_t size);
};

/**
 * @brief MXF (Material Exchange Format) Advanced Processor
 */
class MXFProcessor {
public:
    explicit MXFProcessor();
    ~MXFProcessor();
    
    bool can_process(const std::string& file_path);
    
    // MXF structure analysis
    struct MXFInfo {
        std::string operational_pattern;  // OP1a, OP-Atom, etc.
        std::vector<std::string> essence_tracks;
        std::string metadata_scheme;
        std::string universal_label;
        std::vector<std::string> descriptive_metadata;
        bool has_timecode_track;
        bool has_closed_captions;
        std::string application_specification;
        std::vector<std::string> klv_metadata;
    };
    
    MXFInfo analyze_mxf_structure(const std::string& file_path);
    
    // MXF repair operations
    bool repair_mxf_partition_pack(const std::string& file_path);
    bool rebuild_mxf_index_table(const std::string& file_path);
    bool repair_mxf_metadata(const std::string& file_path);
    
    // MXF validation against broadcast standards
    bool validate_as11_compliance(const std::string& file_path);
    bool validate_as02_compliance(const std::string& file_path);
    bool validate_dpp_compliance(const std::string& file_path);

private:
    // MXF structure parsing
    struct MXFPartitionPack {
        uint8_t key[16];
        uint64_t length;
        uint16_t major_version;
        uint16_t minor_version;
        uint32_t kag_size;
        uint64_t this_partition;
        uint64_t previous_partition;
        uint64_t footer_partition;
        uint64_t header_byte_count;
        uint64_t index_byte_count;
        uint32_t index_sid;
        uint64_t body_offset;
        uint32_t body_sid;
        uint8_t operational_pattern[16];
        std::vector<uint8_t> essence_containers;
    };
    
    bool parse_mxf_partition_pack(const uint8_t* data, MXFPartitionPack& partition);
    bool validate_mxf_klv_structure(const uint8_t* data, size_t size);
    bool repair_mxf_ul_key(uint8_t* key_data);
};

} // namespace AdvancedVideoRepair

#endif // PROFESSIONAL_FORMATS_H