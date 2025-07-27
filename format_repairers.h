#pragma once
#include "file_format_detector.h"
#include <vector>
#include <cstdint>

class JPEGRepairer : public FormatRepairer {
public:
    bool canRepair(const std::vector<uint8_t>& fileData) const override;
    bool repair(std::vector<uint8_t>& fileData) const override;
    std::string getDescription() const override { return "JPEG Image Repairer"; }
    
private:
    bool repairJPEGHeader(std::vector<uint8_t>& data) const;
    bool addMissingEOI(std::vector<uint8_t>& data) const;
    bool validateJPEGSegments(const std::vector<uint8_t>& data) const;
};

class PNGRepairer : public FormatRepairer {
public:
    bool canRepair(const std::vector<uint8_t>& fileData) const override;
    bool repair(std::vector<uint8_t>& fileData) const override;
    std::string getDescription() const override { return "PNG Image Repairer"; }
    
private:
    bool repairPNGHeader(std::vector<uint8_t>& data) const;
    bool validateCRC(const std::vector<uint8_t>& data, size_t chunkStart) const;
    uint32_t calculateCRC(const std::vector<uint8_t>& data, size_t start, size_t length) const;
    bool repairChunkCRCs(std::vector<uint8_t>& data) const;
};

class PDFRepairer : public FormatRepairer {
public:
    bool canRepair(const std::vector<uint8_t>& fileData) const override;
    bool repair(std::vector<uint8_t>& fileData) const override;
    std::string getDescription() const override { return "PDF Document Repairer"; }
    
private:
    bool repairPDFHeader(std::vector<uint8_t>& data) const;
    bool addMissingEOF(std::vector<uint8_t>& data) const;
    bool repairXrefTable(std::vector<uint8_t>& data) const;
};

class ZIPRepairer : public FormatRepairer {
public:
    bool canRepair(const std::vector<uint8_t>& fileData) const override;
    bool repair(std::vector<uint8_t>& fileData) const override;
    std::string getDescription() const override { return "ZIP Archive Repairer"; }
    
private:
    bool repairZIPHeader(std::vector<uint8_t>& data) const;
    bool repairCentralDirectory(std::vector<uint8_t>& data) const;
    bool validateZIPStructure(const std::vector<uint8_t>& data) const;
};

class MP4Repairer : public FormatRepairer {
public:
    bool canRepair(const std::vector<uint8_t>& fileData) const override;
    bool repair(std::vector<uint8_t>& fileData) const override;
    std::string getDescription() const override { return "MP4 Video Repairer"; }
    
private:
    bool repairMP4Header(std::vector<uint8_t>& data) const;
    bool validateAtomStructure(const std::vector<uint8_t>& data) const;
    bool repairAtomSizes(std::vector<uint8_t>& data) const;
};