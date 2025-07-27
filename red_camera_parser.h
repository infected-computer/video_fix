#pragma once
#include "professional_video_formats.h"
#include <cstring>

// RED Camera specific structures and constants
#define RED_MAGIC_R3D 0x52454444  // "REDD" in little endian
#define RED_MAGIC_RMD 0x524D4444  // "RMDD" in little endian

struct REDHeader {
    uint32_t magic;           // 'REDD' or 'RMDD'
    uint32_t version;         // Format version
    uint32_t headerSize;      // Size of header in bytes
    uint32_t frameCount;      // Total number of frames
    uint32_t width;           // Frame width
    uint32_t height;          // Frame height
    uint32_t bitDepth;        // Bit depth (12, 14, 16)
    uint32_t frameRate;       // Frame rate * 1000 (24000 = 24fps)
    uint32_t compression;     // Compression type
    uint32_t colorSpace;      // Color space identifier
    uint64_t timecode;        // Starting timecode
    uint32_t iso;             // ISO sensitivity
    uint32_t shutterAngle;    // Shutter angle * 100
    uint32_t whiteBalance;    // White balance in Kelvin
    uint32_t tint;            // Tint adjustment
    char camera[64];          // Camera model
    char lens[64];            // Lens information
    char operator[64];        // Camera operator
    char location[128];       // Shooting location
    char notes[256];          // Production notes
    uint32_t lutApplied;      // LUT application flag
    uint32_t checksum;        // Header checksum
    uint8_t reserved[256];    // Reserved for future use
};

struct REDFrameIndex {
    uint64_t offset;          // Byte offset in file
    uint32_t size;            // Frame size in bytes
    uint32_t timestamp;       // Frame timestamp
    uint16_t flags;           // Frame flags (keyframe, etc.)
    uint16_t reserved;
};

struct REDMetadataChunk {
    uint32_t chunkType;       // Chunk type identifier
    uint32_t chunkSize;       // Chunk size
    uint32_t timestamp;       // Associated timestamp
    uint32_t reserved;
    // Data follows...
};

class REDCameraParser : public ProfessionalVideoParser {
private:
    bool validateREDHeader(const REDHeader& header) const;
    uint32_t calculateHeaderChecksum(const REDHeader& header) const;
    bool repairREDHeader(REDHeader& header) const;
    bool validateFrameIndex(const std::vector<REDFrameIndex>& frameIndex) const;
    bool repairFrameIndex(std::vector<REDFrameIndex>& frameIndex, size_t fileSize) const;
    
    VideoFrameInfo parseREDFrameInfo(const REDHeader& header) const;
    VideoMetadata parseREDMetadata(const REDHeader& header, const std::vector<uint8_t>& fileData) const;
    
    bool isValidREDCompression(uint32_t compression) const;
    bool isValidREDColorSpace(uint32_t colorSpace) const;
    std::string getCompressionName(uint32_t compression) const;
    std::string getColorSpaceName(uint32_t colorSpace) const;
    
public:
    bool canParse(const std::vector<uint8_t>& fileData) const override;
    VideoFrameInfo parseFrameInfo(const std::vector<uint8_t>& fileData) const override;
    VideoMetadata parseMetadata(const std::vector<uint8_t>& fileData) const override;
    bool repair(std::vector<uint8_t>& fileData) const override;
    std::string getDescription() const override;
    bool validateStructure(const std::vector<uint8_t>& fileData) const override;
    
    // RED-specific methods
    bool extractREDThumbnail(const std::vector<uint8_t>& fileData, std::vector<uint8_t>& thumbnail) const;
    bool validateREDFrame(const std::vector<uint8_t>& frameData) const;
    std::vector<REDFrameIndex> parseFrameIndex(const std::vector<uint8_t>& fileData) const;
    
    // Static utility methods
    static bool isREDFile(const std::vector<uint8_t>& fileData);
    static ProfessionalVideoFormat getREDFormat(const std::vector<uint8_t>& fileData);
};