#ifndef VIDEO_FORMAT_DETECTOR_H
#define VIDEO_FORMAT_DETECTOR_H

#include <string>
#include <vector>
#include <map>
#include <cstdint>
#include <memory>

namespace PhoenixDRS {

struct VideoSignature {
    std::vector<uint8_t> signature;
    size_t offset;
    std::string format_name;
    std::string container;
    std::string description;
    bool is_professional;
    int repair_priority;
    std::vector<std::string> common_codecs;
    std::vector<std::string> audio_codecs;
    std::string max_resolution;
};

struct FileAnalysisResult {
    bool is_valid;
    std::string format_name;
    std::string container;
    std::string detected_codec;
    std::string file_extension;
    size_t file_size;
    bool is_professional;
    int repair_priority;
    std::string validation_message;
    std::vector<std::string> supported_codecs;
    std::vector<std::string> supported_audio_codecs;
    std::string max_resolution;
    double confidence_score;
    bool header_intact;
    bool metadata_present;
};

class VideoFormatDetector {
private:
    std::map<std::string, VideoSignature> format_signatures;
    std::map<std::string, std::vector<uint8_t>> codec_signatures;
    
    void initializeSignatures();
    void initializeCodecSignatures();
    
    bool checkSignature(const std::vector<uint8_t>& data, 
                       const VideoSignature& signature) const;
    
    std::string detectCodec(const std::vector<uint8_t>& data) const;
    
    double calculateConfidence(const std::vector<uint8_t>& data, 
                             const VideoSignature& signature) const;

public:
    VideoFormatDetector();
    ~VideoFormatDetector() = default;
    
    // Core detection functions
    FileAnalysisResult analyzeFile(const std::string& file_path) const;
    FileAnalysisResult analyzeBuffer(const std::vector<uint8_t>& data, 
                                   const std::string& filename = "") const;
    
    // Format validation
    bool validateFileFormat(const std::string& file_path) const;
    bool isFormatSupported(const std::string& extension) const;
    bool isProfessionalFormat(const std::string& extension) const;
    
    // Information retrieval
    std::vector<std::string> getSupportedExtensions() const;
    std::vector<std::string> getProfessionalFormats() const;
    VideoSignature getFormatInfo(const std::string& extension) const;
    
    // Statistics
    size_t getTotalSupportedFormats() const;
    size_t getProfessionalFormatCount() const;
    std::map<std::string, int> getFormatStatistics() const;
    
    // Advanced detection
    std::vector<FileAnalysisResult> detectMultipleFormats(
        const std::vector<uint8_t>& data) const;
    
    bool isFileCorrupted(const std::string& file_path) const;
    std::vector<std::string> suggestRepairMethods(
        const FileAnalysisResult& analysis) const;
};

// Utility functions
std::vector<uint8_t> readFileHeader(const std::string& file_path, 
                                   size_t bytes_to_read = 4096);
bool compareBytes(const std::vector<uint8_t>& data, 
                 const std::vector<uint8_t>& pattern, 
                 size_t offset = 0);
std::string bytesToHex(const std::vector<uint8_t>& data, 
                      size_t max_bytes = 16);

} // namespace PhoenixDRS

#endif // VIDEO_FORMAT_DETECTOR_H