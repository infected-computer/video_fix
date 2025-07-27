#pragma once
#include "professional_video_formats.h"

// ARRI ALEXA specific structures and constants
#define ARRI_MAGIC_ARI  0x41524920  // "ARI " in big endian
#define ARRI_MAGIC_MXF  0x060E2B34  // MXF Universal Label start

struct ARRIHeader {
    uint32_t magic;           // 'ARI ' signature
    uint32_t version;         // Format version
    uint32_t headerSize;      // Header size in bytes
    uint32_t imageWidth;      // Image width in pixels
    uint32_t imageHeight;     // Image height in pixels
    uint32_t activeWidth;     // Active area width
    uint32_t activeHeight;    // Active area height
    uint32_t offsetX;         // Active area X offset
    uint32_t offsetY;         // Active area Y offset
    uint32_t bitDepth;        // Bit depth (12, 16)
    uint32_t colorFormat;     // Color format (Bayer, RGB, etc.)
    uint32_t compression;     // Compression type
    float frameRate;          // Frame rate
    uint64_t timestamp;       // Capture timestamp
    uint32_t iso;             // ISO/ASA sensitivity
    uint32_t shutterAngle;    // Shutter angle in degrees * 100
    uint32_t whiteBalance;    // White balance in Kelvin
    float tint;               // Tint adjustment
    float saturation;         // Saturation adjustment
    float contrast;           // Contrast adjustment
    float gamma;              // Gamma value
    char camera[32];          // Camera model
    char lens[64];            // Lens information
    char operator[32];        // Camera operator
    char scene[32];           // Scene information
    char take[16];            // Take number
    char notes[128];          // Additional notes
    uint32_t lutType;         // LUT type applied
    char lutName[64];         // LUT name
    uint32_t flags;           // Various flags
    uint32_t checksum;        // Header checksum
};

struct ARRIFrameInfo {
    uint64_t offset;          // Frame offset in file
    uint32_t size;            // Frame size in bytes
    uint32_t frameNumber;     // Frame number
    uint64_t timestamp;       // Frame timestamp
    uint16_t quality;         // Frame quality indicator
    uint16_t flags;           // Frame flags
};

struct ARRIColorMetadata {
    float exposureIndex;      // Exposure Index
    float colorTemperature;   // Color temperature
    float tint;               // Tint value
    float saturation;         // Saturation
    float contrast;           // Contrast
    float gamma;              // Gamma
    float redGain;            // Red channel gain
    float greenGain;          // Green channel gain
    float blueGain;           // Blue channel gain
    float lift[3];            // RGB lift values
    float gain[3];            // RGB gain values
    float gamma_rgb[3];       // RGB gamma values
    char primaryColorSpace[32]; // Primary color space
    char transferFunction[32];  // Transfer function
};

class ARRIAlexaParser : public ProfessionalVideoParser {
private:
    bool validateARRIHeader(const ARRIHeader& header) const;
    uint32_t calculateARRIChecksum(const ARRIHeader& header) const;
    bool repairARRIHeader(ARRIHeader& header) const;
    
    VideoFrameInfo parseARRIFrameInfo(const ARRIHeader& header) const;
    VideoMetadata parseARRIMetadata(const ARRIHeader& header) const;
    ARRIColorMetadata parseColorMetadata(const std::vector<uint8_t>& fileData) const;
    
    bool isValidARRIColorFormat(uint32_t colorFormat) const;
    bool isValidARRICompression(uint32_t compression) const;
    std::string getColorFormatName(uint32_t colorFormat) const;
    std::string getCompressionName(uint32_t compression) const;
    std::string getCameraModelName(const std::string& camera) const;
    
    // MXF specific methods
    bool isMXFFile(const std::vector<uint8_t>& fileData) const;
    bool isARRIMXF(const std::vector<uint8_t>& fileData) const;
    bool parseMXFMetadata(const std::vector<uint8_t>& fileData, VideoMetadata& metadata) const;
    bool repairMXFStructure(std::vector<uint8_t>& fileData) const;
    
public:
    bool canParse(const std::vector<uint8_t>& fileData) const override;
    VideoFrameInfo parseFrameInfo(const std::vector<uint8_t>& fileData) const override;
    VideoMetadata parseMetadata(const std::vector<uint8_t>& fileData) const override;
    bool repair(std::vector<uint8_t>& fileData) const override;
    std::string getDescription() const override;
    bool validateStructure(const std::vector<uint8_t>& fileData) const override;
    
    // ARRI-specific methods
    bool extractARRIThumbnail(const std::vector<uint8_t>& fileData, std::vector<uint8_t>& thumbnail) const;
    ARRIColorMetadata extractColorGrading(const std::vector<uint8_t>& fileData) const;
    bool validateARRIFrame(const std::vector<uint8_t>& frameData) const;
    std::vector<ARRIFrameInfo> parseFrameIndex(const std::vector<uint8_t>& fileData) const;
    
    // Static utility methods
    static bool isARRIFile(const std::vector<uint8_t>& fileData);
    static ProfessionalVideoFormat getARRIFormat(const std::vector<uint8_t>& fileData);
    static bool supportsFormat(const std::string& extension);
};