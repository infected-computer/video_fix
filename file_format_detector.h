#pragma once
#include <vector>
#include <string>
#include <unordered_map>
#include <memory>
#include <fstream>

enum class FileFormat {
    UNKNOWN,
    JPEG,
    PNG,
    GIF,
    BMP,
    TIFF,
    PDF,
    ZIP,
    RAR,
    MP4,
    AVI,
    MKV,
    MP3,
    WAV,
    FLAC,
    DOC,
    DOCX,
    XLS,
    XLSX,
    PPT,
    PPTX
};

struct FormatSignature {
    std::vector<uint8_t> signature;
    size_t offset;
    FileFormat format;
    std::string extension;
    std::string description;
};

class FileFormatDetector {
private:
    std::vector<FormatSignature> signatures;
    std::unordered_map<FileFormat, std::unique_ptr<class FormatRepairer>> repairers;
    
    void initializeSignatures();
    bool matchSignature(const std::vector<uint8_t>& fileData, const FormatSignature& sig) const;
    
public:
    FileFormatDetector();
    ~FileFormatDetector();
    
    FileFormat detectFormat(const std::string& filePath) const;
    FileFormat detectFormat(const std::vector<uint8_t>& fileData) const;
    
    bool repairFile(const std::string& filePath, const std::string& outputPath = "") const;
    bool validateFile(const std::string& filePath) const;
    
    std::string getFormatDescription(FileFormat format) const;
    std::string getExpectedExtension(FileFormat format) const;
};

class FormatRepairer {
public:
    virtual ~FormatRepairer() = default;
    virtual bool canRepair(const std::vector<uint8_t>& fileData) const = 0;
    virtual bool repair(std::vector<uint8_t>& fileData) const = 0;
    virtual std::string getDescription() const = 0;
};