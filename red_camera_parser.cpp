#include "red_camera_parser.h"
#include <algorithm>
#include <crc32.h>

bool REDCameraParser::canParse(const std::vector<uint8_t>& fileData) const {
    return isREDFile(fileData);
}

bool REDCameraParser::isREDFile(const std::vector<uint8_t>& fileData) {
    if (fileData.size() < sizeof(REDHeader)) {
        return false;
    }
    
    const REDHeader* header = reinterpret_cast<const REDHeader*>(fileData.data());
    return (header->magic == RED_MAGIC_R3D || header->magic == RED_MAGIC_RMD);
}

ProfessionalVideoFormat REDCameraParser::getREDFormat(const std::vector<uint8_t>& fileData) {
    if (fileData.size() < sizeof(REDHeader)) {
        return ProfessionalVideoFormat::UNKNOWN;
    }
    
    const REDHeader* header = reinterpret_cast<const REDHeader*>(fileData.data());
    if (header->magic == RED_MAGIC_R3D) {
        return ProfessionalVideoFormat::R3D;
    } else if (header->magic == RED_MAGIC_RMD) {
        return ProfessionalVideoFormat::RMD;
    }
    
    return ProfessionalVideoFormat::UNKNOWN;
}

bool REDCameraParser::validateREDHeader(const REDHeader& header) const {
    if (header.magic != RED_MAGIC_R3D && header.magic != RED_MAGIC_RMD) {
        return false;
    }
    
    if (header.headerSize < sizeof(REDHeader) || header.headerSize > 8192) {
        return false;
    }
    
    if (header.width == 0 || header.height == 0 || 
        header.width > 8192 || header.height > 8192) {
        return false;
    }
    
    if (header.bitDepth != 12 && header.bitDepth != 14 && header.bitDepth != 16) {
        return false;
    }
    
    if (header.frameRate < 1000 || header.frameRate > 240000) { // 1fps to 240fps
        return false;
    }
    
    if (!isValidREDCompression(header.compression)) {
        return false;
    }
    
    if (!isValidREDColorSpace(header.colorSpace)) {
        return false;
    }
    
    return true;
}

uint32_t REDCameraParser::calculateHeaderChecksum(const REDHeader& header) const {
    const uint8_t* data = reinterpret_cast<const uint8_t*>(&header);
    size_t checksumOffset = offsetof(REDHeader, checksum);
    
    uint32_t crc = 0xFFFFFFFF;
    
    for (size_t i = 0; i < checksumOffset; i++) {
        crc = crc32_table[(crc ^ data[i]) & 0xFF] ^ (crc >> 8);
    }
    
    size_t afterChecksum = checksumOffset + sizeof(uint32_t);
    for (size_t i = afterChecksum; i < sizeof(REDHeader); i++) {
        crc = crc32_table[(crc ^ data[i]) & 0xFF] ^ (crc >> 8);
    }
    
    return crc ^ 0xFFFFFFFF;
}

bool REDCameraParser::repairREDHeader(REDHeader& header) const {
    bool repaired = false;
    
    if (header.magic != RED_MAGIC_R3D && header.magic != RED_MAGIC_RMD) {
        header.magic = RED_MAGIC_R3D;
        repaired = true;
    }
    
    if (header.headerSize < sizeof(REDHeader)) {
        header.headerSize = sizeof(REDHeader);
        repaired = true;
    }
    
    if (header.width == 0 || header.height == 0) {
        header.width = 4096;
        header.height = 2160;
        repaired = true;
    }
    
    if (header.bitDepth != 12 && header.bitDepth != 14 && header.bitDepth != 16) {
        header.bitDepth = 12;
        repaired = true;
    }
    
    if (header.frameRate < 1000 || header.frameRate > 240000) {
        header.frameRate = 24000;
        repaired = true;
    }
    
    if (!isValidREDCompression(header.compression)) {
        header.compression = 1; // Default RED compression
        repaired = true;
    }
    
    if (!isValidREDColorSpace(header.colorSpace)) {
        header.colorSpace = 1; // Default RED color space
        repaired = true;
    }
    
    uint32_t correctChecksum = calculateHeaderChecksum(header);
    if (header.checksum != correctChecksum) {
        header.checksum = correctChecksum;
        repaired = true;
    }
    
    return repaired;
}

VideoFrameInfo REDCameraParser::parseFrameInfo(const std::vector<uint8_t>& fileData) const {
    VideoFrameInfo info = {};
    
    if (fileData.size() < sizeof(REDHeader)) {
        return info;
    }
    
    const REDHeader* header = reinterpret_cast<const REDHeader*>(fileData.data());
    return parseREDFrameInfo(*header);
}

VideoFrameInfo REDCameraParser::parseREDFrameInfo(const REDHeader& header) const {
    VideoFrameInfo info = {};
    
    info.width = header.width;
    info.height = header.height;
    info.bitDepth = header.bitDepth;
    info.frameRate = static_cast<float>(header.frameRate) / 1000.0f;
    info.frameCount = header.frameCount;
    info.timecode = header.timecode;
    info.colorSpace = getColorSpaceName(header.colorSpace);
    info.codec = "RED " + getCompressionName(header.compression);
    
    return info;
}

VideoMetadata REDCameraParser::parseMetadata(const std::vector<uint8_t>& fileData) const {
    VideoMetadata metadata = {};
    
    if (fileData.size() < sizeof(REDHeader)) {
        return metadata;
    }
    
    const REDHeader* header = reinterpret_cast<const REDHeader*>(fileData.data());
    return parseREDMetadata(*header, fileData);
}

VideoMetadata REDCameraParser::parseREDMetadata(const REDHeader& header, const std::vector<uint8_t>& fileData) const {
    VideoMetadata metadata = {};
    
    metadata.camera = std::string(header.camera, strnlen(header.camera, 64));
    metadata.lens = std::string(header.lens, strnlen(header.lens, 64));
    metadata.iso = std::to_string(header.iso);
    metadata.shutterSpeed = "1/" + std::to_string(360.0f / (header.shutterAngle / 100.0f));
    metadata.whiteBalance = std::to_string(header.whiteBalance) + "K";
    metadata.recordingFormat = "RED R3D";
    metadata.lutApplied = header.lutApplied ? "Yes" : "No";
    
    metadata.customMetadata["operator"] = std::string(header.operator, strnlen(header.operator, 64));
    metadata.customMetadata["location"] = std::string(header.location, strnlen(header.location, 128));
    metadata.customMetadata["notes"] = std::string(header.notes, strnlen(header.notes, 256));
    metadata.customMetadata["tint"] = std::to_string(header.tint);
    metadata.customMetadata["compression"] = getCompressionName(header.compression);
    metadata.customMetadata["color_space"] = getColorSpaceName(header.colorSpace);
    
    return metadata;
}

bool REDCameraParser::repair(std::vector<uint8_t>& fileData) const {
    if (fileData.size() < sizeof(REDHeader)) {
        return false;
    }
    
    REDHeader* header = reinterpret_cast<REDHeader*>(fileData.data());
    bool headerRepaired = repairREDHeader(*header);
    
    if (fileData.size() > header->headerSize) {
        std::vector<REDFrameIndex> frameIndex = parseFrameIndex(fileData);
        bool indexRepaired = repairFrameIndex(frameIndex, fileData.size());
        
        if (indexRepaired) {
            size_t indexOffset = header->headerSize;
            if (indexOffset + frameIndex.size() * sizeof(REDFrameIndex) <= fileData.size()) {
                std::memcpy(fileData.data() + indexOffset, frameIndex.data(), 
                           frameIndex.size() * sizeof(REDFrameIndex));
            }
        }
        
        return headerRepaired || indexRepaired;
    }
    
    return headerRepaired;
}

bool REDCameraParser::validateStructure(const std::vector<uint8_t>& fileData) const {
    if (fileData.size() < sizeof(REDHeader)) {
        return false;
    }
    
    const REDHeader* header = reinterpret_cast<const REDHeader*>(fileData.data());
    
    if (!validateREDHeader(*header)) {
        return false;
    }
    
    if (fileData.size() < header->headerSize) {
        return false;
    }
    
    std::vector<REDFrameIndex> frameIndex = parseFrameIndex(fileData);
    return validateFrameIndex(frameIndex);
}

std::string REDCameraParser::getDescription() const {
    return "RED Digital Cinema Camera Format Parser";
}

bool REDCameraParser::isValidREDCompression(uint32_t compression) const {
    return compression >= 1 && compression <= 8; // RED compression levels 1-8
}

bool REDCameraParser::isValidREDColorSpace(uint32_t colorSpace) const {
    return colorSpace >= 1 && colorSpace <= 4; // RED color spaces
}

std::string REDCameraParser::getCompressionName(uint32_t compression) const {
    switch (compression) {
        case 1: return "REDCODE RAW 2:1";
        case 2: return "REDCODE RAW 3:1";
        case 3: return "REDCODE RAW 4:1";
        case 4: return "REDCODE RAW 5:1";
        case 5: return "REDCODE RAW 6:1";
        case 6: return "REDCODE RAW 7:1";
        case 7: return "REDCODE RAW 8:1";
        case 8: return "REDCODE RAW 12:1";
        default: return "Unknown";
    }
}

std::string REDCameraParser::getColorSpaceName(uint32_t colorSpace) const {
    switch (colorSpace) {
        case 1: return "REDcolor";
        case 2: return "REDcolor2";
        case 3: return "REDcolor3";
        case 4: return "REDcolor4";
        default: return "Unknown";
    }
}

std::vector<REDFrameIndex> REDCameraParser::parseFrameIndex(const std::vector<uint8_t>& fileData) const {
    std::vector<REDFrameIndex> frameIndex;
    
    if (fileData.size() < sizeof(REDHeader)) {
        return frameIndex;
    }
    
    const REDHeader* header = reinterpret_cast<const REDHeader*>(fileData.data());
    size_t indexOffset = header->headerSize;
    size_t indexSize = header->frameCount * sizeof(REDFrameIndex);
    
    if (indexOffset + indexSize > fileData.size()) {
        return frameIndex;
    }
    
    const REDFrameIndex* indexData = reinterpret_cast<const REDFrameIndex*>(fileData.data() + indexOffset);
    frameIndex.assign(indexData, indexData + header->frameCount);
    
    return frameIndex;
}

bool REDCameraParser::validateFrameIndex(const std::vector<REDFrameIndex>& frameIndex) const {
    if (frameIndex.empty()) {
        return false;
    }
    
    uint64_t previousOffset = 0;
    for (const auto& frame : frameIndex) {
        if (frame.offset <= previousOffset) {
            return false;
        }
        
        if (frame.size == 0 || frame.size > 100 * 1024 * 1024) { // Max 100MB per frame
            return false;
        }
        
        previousOffset = frame.offset;
    }
    
    return true;
}

bool REDCameraParser::repairFrameIndex(std::vector<REDFrameIndex>& frameIndex, size_t fileSize) const {
    if (frameIndex.empty()) {
        return false;
    }
    
    bool repaired = false;
    uint64_t currentOffset = sizeof(REDHeader) + frameIndex.size() * sizeof(REDFrameIndex);
    
    for (size_t i = 0; i < frameIndex.size(); i++) {
        if (frameIndex[i].offset < currentOffset || frameIndex[i].offset >= fileSize) {
            frameIndex[i].offset = currentOffset;
            repaired = true;
        }
        
        if (frameIndex[i].size == 0) {
            frameIndex[i].size = 1024 * 1024; // Default 1MB frame size
            repaired = true;
        }
        
        if (frameIndex[i].offset + frameIndex[i].size > fileSize) {
            frameIndex[i].size = fileSize - frameIndex[i].offset;
            repaired = true;
        }
        
        currentOffset = frameIndex[i].offset + frameIndex[i].size;
    }
    
    return repaired;
}

bool REDCameraParser::extractREDThumbnail(const std::vector<uint8_t>& fileData, std::vector<uint8_t>& thumbnail) const {
    // RED thumbnail extraction would require more detailed format knowledge
    // This is a placeholder implementation
    return false;
}

bool REDCameraParser::validateREDFrame(const std::vector<uint8_t>& frameData) const {
    // RED frame validation would require more detailed format knowledge
    // This is a placeholder implementation
    return frameData.size() > 0;
}