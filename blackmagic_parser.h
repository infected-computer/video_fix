#pragma once
#include "professional_video_formats.h"

// Blackmagic Design specific structures and constants
#define BRAW_MAGIC      0x42524157  // "BRAW" in big endian
#define BRAW_VERSION_1  0x00000001
#define BRAW_VERSION_2  0x00000002

// CinemaDNG constants
#define TIFF_MAGIC_II   0x4949      // "II" little endian
#define TIFF_MAGIC_MM   0x4D4D      // "MM" big endian
#define TIFF_VERSION    42
#define CINEMA_DNG_VERSION 0x01040000

struct BRAWHeader {
    uint32_t magic;           // 'BRAW'
    uint32_t version;         // Format version
    uint32_t headerSize;      // Header size
    uint32_t frameCount;      // Total frames
    uint32_t width;           // Frame width
    uint32_t height;          // Frame height
    uint32_t bitDepth;        // Bit depth (12, 16)
    float frameRate;          // Frame rate
    uint32_t compression;     // Compression type (3:1, 5:1, 8:1, 12:1)
    uint32_t colorScience;    // Blackmagic color science version
    uint64_t timecode;        // Starting timecode
    uint32_t iso;             // ISO value
    uint32_t whiteBalance;    // White balance in Kelvin
    float tint;               // Tint adjustment
    float saturation;         // Saturation
    float contrast;           // Contrast
    float gamma;              // Gamma
    float lift[3];            // RGB lift
    float gain[3];            // RGB gain
    float offset[3];          // RGB offset
    char camera[64];          // Camera model
    char lens[64];            // Lens info
    char operator[32];        // Operator
    char scene[32];           // Scene
    char shot[16];            // Shot
    char take[8];             // Take
    char reel[16];            // Reel
    uint32_t generation;      // Blackmagic generation
    uint32_t lutApplied;      // LUT applied flag
    char lutName[64];         // Applied LUT name
    uint32_t flags;           // Various flags
    uint8_t reserved[128];    // Reserved space
    uint32_t checksum;        // Header checksum
};

struct BRAWFrameHeader {
    uint32_t frameSize;       // Frame size in bytes
    uint32_t frameNumber;     // Frame number
    uint64_t timestamp;       // Frame timestamp
    uint32_t compressionRatio; // Actual compression ratio
    uint16_t quality;         // Frame quality
    uint16_t flags;           // Frame flags
    uint32_t dataOffset;      // Offset to frame data
    uint32_t metadataOffset;  // Offset to metadata
    uint32_t metadataSize;    // Metadata size
    uint32_t reserved;
};

struct CinemaDNGHeader {
    uint16_t byteOrder;       // Byte order (II or MM)
    uint16_t version;         // TIFF version (42)
    uint32_t ifdOffset;       // IFD offset
    uint32_t dngVersion;      // DNG version
    uint32_t dngBackwardVersion; // DNG backward version
    char uniqueCameraModel[64]; // Unique camera model
    char cameraSerialNumber[32]; // Camera serial
    uint32_t colorMatrix1[9]; // Color matrix 1
    uint32_t colorMatrix2[9]; // Color matrix 2
    uint32_t whiteLevel;      // White level
    uint32_t blackLevel;      // Black level
    uint32_t bayerPattern;    // Bayer pattern
    uint32_t activeArea[4];   // Active area
    uint32_t defaultCropOrigin[2]; // Default crop origin
    uint32_t defaultCropSize[2];   // Default crop size
};

struct BlackmagicColorMetadata {
    uint32_t colorScience;    // Color science version
    float colorTemperature;   // Color temperature
    float tint;               // Tint
    float saturation;         // Saturation
    float contrast;           // Contrast
    float gamma;              // Gamma
    float highlight;          // Highlight
    float shadow;             // Shadow
    float midtone;            // Midtone
    float hue[6];             // Hue adjustments (6 color ranges)
    float saturation_color[6]; // Saturation per color
    float luminance[6];       // Luminance per color
    char lutName[64];         // Applied LUT
    uint32_t lutStrength;     // LUT strength (0-100)
};

class BlackmagicParser : public ProfessionalVideoParser {
private:
    // BRAW specific methods
    bool validateBRAWHeader(const BRAWHeader& header) const;
    uint32_t calculateBRAWChecksum(const BRAWHeader& header) const;
    bool repairBRAWHeader(BRAWHeader& header) const;
    
    // CinemaDNG specific methods
    bool validateCinemaDNGHeader(const CinemaDNGHeader& header) const;
    bool repairCinemaDNGHeader(CinemaDNGHeader& header) const;
    bool parseTIFFTags(const std::vector<uint8_t>& fileData, VideoMetadata& metadata) const;
    
    VideoFrameInfo parseBRAWFrameInfo(const BRAWHeader& header) const;
    VideoFrameInfo parseCinemaDNGFrameInfo(const std::vector<uint8_t>& fileData) const;
    
    VideoMetadata parseBRAWMetadata(const BRAWHeader& header) const;
    VideoMetadata parseCinemaDNGMetadata(const std::vector<uint8_t>& fileData) const;
    
    BlackmagicColorMetadata parseColorMetadata(const std::vector<uint8_t>& fileData) const;
    
    bool isValidBRAWCompression(uint32_t compression) const;
    bool isValidColorScience(uint32_t colorScience) const;
    std::string getCompressionName(uint32_t compression) const;
    std::string getColorScienceName(uint32_t colorScience) const;
    std::string getCameraGenerationName(uint32_t generation) const;
    
    // Frame processing
    std::vector<BRAWFrameHeader> parseBRAWFrameIndex(const std::vector<uint8_t>& fileData) const;
    bool validateBRAWFrame(const std::vector<uint8_t>& frameData) const;
    bool repairBRAWFrameIndex(std::vector<BRAWFrameHeader>& frameIndex, size_t fileSize) const;
    
public:
    bool canParse(const std::vector<uint8_t>& fileData) const override;
    VideoFrameInfo parseFrameInfo(const std::vector<uint8_t>& fileData) const override;
    VideoMetadata parseMetadata(const std::vector<uint8_t>& fileData) const override;
    bool repair(std::vector<uint8_t>& fileData) const override;
    std::string getDescription() const override;
    bool validateStructure(const std::vector<uint8_t>& fileData) const override;
    
    // Blackmagic-specific methods
    bool extractBRAWThumbnail(const std::vector<uint8_t>& fileData, std::vector<uint8_t>& thumbnail) const;
    BlackmagicColorMetadata extractColorGrading(const std::vector<uint8_t>& fileData) const;
    bool convertCinemaDNGToBRAW(const std::vector<uint8_t>& dngData, std::vector<uint8_t>& brawData) const;
    
    // Static utility methods
    static bool isBRAWFile(const std::vector<uint8_t>& fileData);
    static bool isCinemaDNGFile(const std::vector<uint8_t>& fileData);
    static ProfessionalVideoFormat getBlackmagicFormat(const std::vector<uint8_t>& fileData);
    static bool supportsCamera(const std::string& cameraModel);
};