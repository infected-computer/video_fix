#include "file_format_detector.h"
#include "format_repairers.h"
#include <fstream>
#include <iostream>
#include <algorithm>

FileFormatDetector::FileFormatDetector() {
    initializeSignatures();
    
    repairers[FileFormat::JPEG] = std::make_unique<JPEGRepairer>();
    repairers[FileFormat::PNG] = std::make_unique<PNGRepairer>();
    repairers[FileFormat::PDF] = std::make_unique<PDFRepairer>();
    repairers[FileFormat::ZIP] = std::make_unique<ZIPRepairer>();
    repairers[FileFormat::MP4] = std::make_unique<MP4Repairer>();
}

FileFormatDetector::~FileFormatDetector() = default;

void FileFormatDetector::initializeSignatures() {
    signatures = {
        // Image formats
        {{0xFF, 0xD8, 0xFF}, 0, FileFormat::JPEG, ".jpg", "JPEG Image"},
        {{0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A}, 0, FileFormat::PNG, ".png", "PNG Image"},
        {{0x47, 0x49, 0x46, 0x38, 0x37, 0x61}, 0, FileFormat::GIF, ".gif", "GIF Image (87a)"},
        {{0x47, 0x49, 0x46, 0x38, 0x39, 0x61}, 0, FileFormat::GIF, ".gif", "GIF Image (89a)"},
        {{0x42, 0x4D}, 0, FileFormat::BMP, ".bmp", "BMP Image"},
        {{0x49, 0x49, 0x2A, 0x00}, 0, FileFormat::TIFF, ".tiff", "TIFF Image (Little Endian)"},
        {{0x4D, 0x4D, 0x00, 0x2A}, 0, FileFormat::TIFF, ".tiff", "TIFF Image (Big Endian)"},
        
        // Document formats
        {{0x25, 0x50, 0x44, 0x46}, 0, FileFormat::PDF, ".pdf", "PDF Document"},
        {{0xD0, 0xCF, 0x11, 0xE0, 0xA1, 0xB1, 0x1A, 0xE1}, 0, FileFormat::DOC, ".doc", "Microsoft Office Document"},
        {{0x50, 0x4B, 0x03, 0x04}, 0, FileFormat::DOCX, ".docx", "Office Open XML Document"},
        {{0x50, 0x4B, 0x07, 0x08}, 0, FileFormat::DOCX, ".docx", "Office Open XML Document (alt)"},
        
        // Archive formats
        {{0x50, 0x4B, 0x03, 0x04}, 0, FileFormat::ZIP, ".zip", "ZIP Archive"},
        {{0x52, 0x61, 0x72, 0x21, 0x1A, 0x07, 0x00}, 0, FileFormat::RAR, ".rar", "RAR Archive (v1.5+)"},
        {{0x52, 0x61, 0x72, 0x21, 0x1A, 0x07, 0x01, 0x00}, 0, FileFormat::RAR, ".rar", "RAR Archive (v5.0+)"},
        
        // Video formats
        {{0x66, 0x74, 0x79, 0x70}, 4, FileFormat::MP4, ".mp4", "MP4 Video"},
        {{0x52, 0x49, 0x46, 0x46}, 0, FileFormat::AVI, ".avi", "AVI Video"},
        {{0x1A, 0x45, 0xDF, 0xA3}, 0, FileFormat::MKV, ".mkv", "Matroska Video"},
        
        // Audio formats
        {{0x49, 0x44, 0x33}, 0, FileFormat::MP3, ".mp3", "MP3 Audio (ID3v2)"},
        {{0xFF, 0xFB}, 0, FileFormat::MP3, ".mp3", "MP3 Audio (MPEG-1 Layer 3)"},
        {{0xFF, 0xF3}, 0, FileFormat::MP3, ".mp3", "MP3 Audio (MPEG-1 Layer 3)"},
        {{0xFF, 0xF2}, 0, FileFormat::MP3, ".mp3", "MP3 Audio (MPEG-1 Layer 3)"},
        {{0x52, 0x49, 0x46, 0x46}, 0, FileFormat::WAV, ".wav", "WAV Audio"},
        {{0x66, 0x4C, 0x61, 0x43}, 0, FileFormat::FLAC, ".flac", "FLAC Audio"}
    };
}

bool FileFormatDetector::matchSignature(const std::vector<uint8_t>& fileData, const FormatSignature& sig) const {
    if (fileData.size() < sig.offset + sig.signature.size()) {
        return false;
    }
    
    return std::equal(sig.signature.begin(), sig.signature.end(), 
                     fileData.begin() + sig.offset);
}

FileFormat FileFormatDetector::detectFormat(const std::string& filePath) const {
    std::ifstream file(filePath, std::ios::binary);
    if (!file.is_open()) {
        return FileFormat::UNKNOWN;
    }
    
    file.seekg(0, std::ios::end);
    size_t fileSize = file.tellg();
    file.seekg(0, std::ios::beg);
    
    size_t readSize = std::min(fileSize, static_cast<size_t>(1024));
    std::vector<uint8_t> buffer(readSize);
    file.read(reinterpret_cast<char*>(buffer.data()), readSize);
    
    return detectFormat(buffer);
}

FileFormat FileFormatDetector::detectFormat(const std::vector<uint8_t>& fileData) const {
    for (const auto& sig : signatures) {
        if (matchSignature(fileData, sig)) {
            if (sig.format == FileFormat::ZIP || sig.format == FileFormat::DOCX) {
                if (fileData.size() > 30) {
                    std::string content(fileData.begin(), fileData.begin() + 30);
                    if (content.find("word/") != std::string::npos ||
                        content.find("xl/") != std::string::npos ||
                        content.find("ppt/") != std::string::npos) {
                        if (content.find("word/") != std::string::npos) return FileFormat::DOCX;
                        if (content.find("xl/") != std::string::npos) return FileFormat::XLSX;
                        if (content.find("ppt/") != std::string::npos) return FileFormat::PPTX;
                    }
                }
                return FileFormat::ZIP;
            }
            return sig.format;
        }
    }
    return FileFormat::UNKNOWN;
}

bool FileFormatDetector::repairFile(const std::string& filePath, const std::string& outputPath) const {
    std::ifstream file(filePath, std::ios::binary);
    if (!file.is_open()) {
        return false;
    }
    
    file.seekg(0, std::ios::end);
    size_t fileSize = file.tellg();
    file.seekg(0, std::ios::beg);
    
    std::vector<uint8_t> fileData(fileSize);
    file.read(reinterpret_cast<char*>(fileData.data()), fileSize);
    file.close();
    
    FileFormat format = detectFormat(fileData);
    if (format == FileFormat::UNKNOWN) {
        return false;
    }
    
    auto it = repairers.find(format);
    if (it == repairers.end()) {
        return false;
    }
    
    if (!it->second->canRepair(fileData)) {
        return false;
    }
    
    std::vector<uint8_t> repairedData = fileData;
    bool repairSuccess = it->second->repair(repairedData);
    
    if (repairSuccess) {
        std::string outPath = outputPath.empty() ? filePath + "_repaired" : outputPath;
        std::ofstream outFile(outPath, std::ios::binary);
        if (outFile.is_open()) {
            outFile.write(reinterpret_cast<const char*>(repairedData.data()), repairedData.size());
            outFile.close();
            return true;
        }
    }
    
    return false;
}

bool FileFormatDetector::validateFile(const std::string& filePath) const {
    FileFormat format = detectFormat(filePath);
    return format != FileFormat::UNKNOWN;
}

std::string FileFormatDetector::getFormatDescription(FileFormat format) const {
    for (const auto& sig : signatures) {
        if (sig.format == format) {
            return sig.description;
        }
    }
    return "Unknown Format";
}

std::string FileFormatDetector::getExpectedExtension(FileFormat format) const {
    for (const auto& sig : signatures) {
        if (sig.format == format) {
            return sig.extension;
        }
    }
    return "";
}