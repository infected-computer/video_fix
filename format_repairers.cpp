#include "format_repairers.h"
#include <algorithm>
#include <cstring>

// JPEG Repairer Implementation
bool JPEGRepairer::canRepair(const std::vector<uint8_t>& fileData) const {
    if (fileData.size() < 3) return false;
    return fileData[0] == 0xFF && fileData[1] == 0xD8;
}

bool JPEGRepairer::repair(std::vector<uint8_t>& fileData) const {
    bool repaired = false;
    
    if (repairJPEGHeader(fileData)) repaired = true;
    if (addMissingEOI(fileData)) repaired = true;
    
    return repaired;
}

bool JPEGRepairer::repairJPEGHeader(std::vector<uint8_t>& data) const {
    if (data.size() < 3) return false;
    
    if (data[0] != 0xFF || data[1] != 0xD8) {
        data.insert(data.begin(), {0xFF, 0xD8, 0xFF});
        return true;
    }
    
    if (data[2] != 0xFF) {
        data[2] = 0xFF;
        return true;
    }
    
    return false;
}

bool JPEGRepairer::addMissingEOI(std::vector<uint8_t>& data) const {
    if (data.size() < 2) return false;
    
    if (data[data.size()-2] != 0xFF || data[data.size()-1] != 0xD9) {
        data.push_back(0xFF);
        data.push_back(0xD9);
        return true;
    }
    
    return false;
}

bool JPEGRepairer::validateJPEGSegments(const std::vector<uint8_t>& data) const {
    size_t pos = 2;
    while (pos < data.size() - 1) {
        if (data[pos] != 0xFF) return false;
        
        uint8_t marker = data[pos + 1];
        if (marker == 0xD9) break;
        
        if (marker >= 0xD0 && marker <= 0xD7) {
            pos += 2;
            continue;
        }
        
        if (pos + 3 >= data.size()) return false;
        uint16_t length = (data[pos + 2] << 8) | data[pos + 3];
        pos += 2 + length;
    }
    
    return true;
}

// PNG Repairer Implementation
bool PNGRepairer::canRepair(const std::vector<uint8_t>& fileData) const {
    if (fileData.size() < 8) return false;
    
    const uint8_t pngSig[] = {0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A};
    return std::equal(std::begin(pngSig), std::end(pngSig), fileData.begin());
}

bool PNGRepairer::repair(std::vector<uint8_t>& fileData) const {
    bool repaired = false;
    
    if (repairPNGHeader(fileData)) repaired = true;
    if (repairChunkCRCs(fileData)) repaired = true;
    
    return repaired;
}

bool PNGRepairer::repairPNGHeader(std::vector<uint8_t>& data) const {
    if (data.size() < 8) return false;
    
    const uint8_t correctSig[] = {0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A};
    bool modified = false;
    
    for (int i = 0; i < 8; i++) {
        if (data[i] != correctSig[i]) {
            data[i] = correctSig[i];
            modified = true;
        }
    }
    
    return modified;
}

uint32_t PNGRepairer::calculateCRC(const std::vector<uint8_t>& data, size_t start, size_t length) const {
    static uint32_t crcTable[256];
    static bool tableInit = false;
    
    if (!tableInit) {
        for (uint32_t i = 0; i < 256; i++) {
            uint32_t c = i;
            for (int k = 0; k < 8; k++) {
                if (c & 1) c = 0xEDB88320 ^ (c >> 1);
                else c = c >> 1;
            }
            crcTable[i] = c;
        }
        tableInit = true;
    }
    
    uint32_t crc = 0xFFFFFFFF;
    for (size_t i = start; i < start + length && i < data.size(); i++) {
        crc = crcTable[(crc ^ data[i]) & 0xFF] ^ (crc >> 8);
    }
    return crc ^ 0xFFFFFFFF;
}

bool PNGRepairer::repairChunkCRCs(std::vector<uint8_t>& data) const {
    if (data.size() < 33) return false;
    
    bool repaired = false;
    size_t pos = 8;
    
    while (pos + 12 <= data.size()) {
        uint32_t chunkLength = (data[pos] << 24) | (data[pos+1] << 16) | 
                              (data[pos+2] << 8) | data[pos+3];
        
        if (pos + 12 + chunkLength > data.size()) break;
        
        uint32_t calculatedCRC = calculateCRC(data, pos + 4, 4 + chunkLength);
        uint32_t storedCRC = (data[pos + 8 + chunkLength] << 24) |
                            (data[pos + 9 + chunkLength] << 16) |
                            (data[pos + 10 + chunkLength] << 8) |
                            data[pos + 11 + chunkLength];
        
        if (calculatedCRC != storedCRC) {
            data[pos + 8 + chunkLength] = (calculatedCRC >> 24) & 0xFF;
            data[pos + 9 + chunkLength] = (calculatedCRC >> 16) & 0xFF;
            data[pos + 10 + chunkLength] = (calculatedCRC >> 8) & 0xFF;
            data[pos + 11 + chunkLength] = calculatedCRC & 0xFF;
            repaired = true;
        }
        
        pos += 12 + chunkLength;
        
        if (std::string(data.begin() + pos - chunkLength - 8, 
                       data.begin() + pos - chunkLength - 4) == "IEND") {
            break;
        }
    }
    
    return repaired;
}

// PDF Repairer Implementation
bool PDFRepairer::canRepair(const std::vector<uint8_t>& fileData) const {
    if (fileData.size() < 4) return false;
    return fileData[0] == '%' && fileData[1] == 'P' && 
           fileData[2] == 'D' && fileData[3] == 'F';
}

bool PDFRepairer::repair(std::vector<uint8_t>& fileData) const {
    bool repaired = false;
    
    if (repairPDFHeader(fileData)) repaired = true;
    if (addMissingEOF(fileData)) repaired = true;
    
    return repaired;
}

bool PDFRepairer::repairPDFHeader(std::vector<uint8_t>& data) const {
    if (data.size() < 8) return false;
    
    std::string header = "%PDF-1.4";
    if (std::string(data.begin(), data.begin() + 8) != header) {
        std::copy(header.begin(), header.end(), data.begin());
        return true;
    }
    
    return false;
}

bool PDFRepairer::addMissingEOF(std::vector<uint8_t>& data) const {
    std::string eof = "%%EOF";
    std::string endData(data.end() - std::min(data.size(), static_cast<size_t>(10)), data.end());
    
    if (endData.find("%%EOF") == std::string::npos) {
        data.insert(data.end(), eof.begin(), eof.end());
        data.push_back('\n');
        return true;
    }
    
    return false;
}

bool PDFRepairer::repairXrefTable(std::vector<uint8_t>& data) const {
    return false;
}

// ZIP Repairer Implementation
bool ZIPRepairer::canRepair(const std::vector<uint8_t>& fileData) const {
    if (fileData.size() < 4) return false;
    return fileData[0] == 0x50 && fileData[1] == 0x4B;
}

bool ZIPRepairer::repair(std::vector<uint8_t>& fileData) const {
    bool repaired = false;
    
    if (repairZIPHeader(fileData)) repaired = true;
    
    return repaired;
}

bool ZIPRepairer::repairZIPHeader(std::vector<uint8_t>& data) const {
    if (data.size() < 4) return false;
    
    if (data[0] != 0x50 || data[1] != 0x4B) {
        data[0] = 0x50;
        data[1] = 0x4B;
        data[2] = 0x03;
        data[3] = 0x04;
        return true;
    }
    
    return false;
}

bool ZIPRepairer::repairCentralDirectory(std::vector<uint8_t>& data) const {
    return false;
}

bool ZIPRepairer::validateZIPStructure(const std::vector<uint8_t>& data) const {
    return true;
}

// MP4 Repairer Implementation
bool MP4Repairer::canRepair(const std::vector<uint8_t>& fileData) const {
    if (fileData.size() < 8) return false;
    return fileData[4] == 'f' && fileData[5] == 't' && 
           fileData[6] == 'y' && fileData[7] == 'p';
}

bool MP4Repairer::repair(std::vector<uint8_t>& fileData) const {
    bool repaired = false;
    
    if (repairMP4Header(fileData)) repaired = true;
    if (repairAtomSizes(fileData)) repaired = true;
    
    return repaired;
}

bool MP4Repairer::repairMP4Header(std::vector<uint8_t>& data) const {
    if (data.size() < 8) return false;
    
    if (data[4] != 'f' || data[5] != 't' || data[6] != 'y' || data[7] != 'p') {
        data[4] = 'f';
        data[5] = 't';
        data[6] = 'y';
        data[7] = 'p';
        return true;
    }
    
    return false;
}

bool MP4Repairer::validateAtomStructure(const std::vector<uint8_t>& data) const {
    return true;
}

bool MP4Repairer::repairAtomSizes(std::vector<uint8_t>& data) const {
    return false;
}