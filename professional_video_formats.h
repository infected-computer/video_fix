#pragma once
#include "file_format_detector.h"
#include <cstdint>
#include <map>
#include <string>

enum class ProfessionalVideoFormat {
    UNKNOWN,
    // RED Digital Cinema
    R3D,           // RED RAW 3D
    RMD,           // RED Metadata
    // ARRI ALEXA
    ARI,           // ARRI RAW
    MXF_ARRI,      // ARRI MXF
    // Blackmagic Design
    BRAW,          // Blackmagic RAW
    CINEMA_DNG,    // CinemaDNG RAW
    // Sony Professional
    XAVC,          // Sony XAVC
    MXF_SONY,      // Sony MXF
    F55_RAW,       // Sony F55 RAW
    F65_RAW,       // Sony F65 RAW
    // Canon Cinema
    CRM,           // Canon RAW Material
    C4K,           // Canon 4K
    XF_AVC,        // Canon XF-AVC
    // Panasonic
    P2_MXF,        // Panasonic P2 MXF
    AVCHD_PRO,     // AVCHD Professional
    AVC_INTRA,     // AVC-Intra
    // Additional Professional Formats
    PRORES_RAW,    // Apple ProRes RAW
    PRORES_4444,   // Apple ProRes 4444
    PRORES_422,    // Apple ProRes 422
    DNX_HD,        // Avid DNxHD
    DNX_HR,        // Avid DNxHR
    CINEFORM,      // GoPro CineForm
    // Atomos/External Recorders
    ATOMOS_MOV,    // Atomos ProRes MOV
    NINJA_RAW      // Atomos Ninja RAW
};

struct ProfessionalVideoSignature {
    std::vector<uint8_t> signature;
    size_t offset;
    ProfessionalVideoFormat format;
    std::string extension;
    std::string description;
    std::string manufacturer;
    uint32_t headerSize;
    bool requiresSecondaryValidation;
};

struct VideoFrameInfo {
    uint32_t width;
    uint32_t height;
    uint32_t bitDepth;
    float frameRate;
    std::string colorSpace;
    std::string codec;
    uint64_t frameCount;
    uint64_t timecode;
};

struct VideoMetadata {
    std::string camera;
    std::string lens;
    std::string shootingMode;
    std::string iso;
    std::string shutterSpeed;
    std::string aperture;
    std::string whiteBalance;
    std::string recordingFormat;
    std::string lutApplied;
    std::map<std::string, std::string> customMetadata;
};

class ProfessionalVideoDetector {
private:
    std::vector<ProfessionalVideoSignature> signatures;
    std::map<ProfessionalVideoFormat, std::unique_ptr<class ProfessionalVideoParser>> parsers;
    
    void initializeProfessionalSignatures();
    bool validateREDFormat(const std::vector<uint8_t>& data) const;
    bool validateARRIFormat(const std::vector<uint8_t>& data) const;
    bool validateBlackmagicFormat(const std::vector<uint8_t>& data) const;
    bool validateSonyFormat(const std::vector<uint8_t>& data) const;
    bool validateCanonFormat(const std::vector<uint8_t>& data) const;
    bool validatePanasonicFormat(const std::vector<uint8_t>& data) const;
    
public:
    ProfessionalVideoDetector();
    ~ProfessionalVideoDetector();
    
    ProfessionalVideoFormat detectProfessionalFormat(const std::string& filePath) const;
    ProfessionalVideoFormat detectProfessionalFormat(const std::vector<uint8_t>& fileData) const;
    
    VideoFrameInfo extractFrameInfo(const std::string& filePath) const;
    VideoMetadata extractMetadata(const std::string& filePath) const;
    
    bool repairProfessionalVideo(const std::string& filePath, const std::string& outputPath = "") const;
    bool validateProfessionalVideo(const std::string& filePath) const;
    
    std::string getFormatDescription(ProfessionalVideoFormat format) const;
    std::string getManufacturer(ProfessionalVideoFormat format) const;
    std::string getExpectedExtension(ProfessionalVideoFormat format) const;
    
    std::vector<std::string> getSupportedFormats() const;
    bool isRAWFormat(ProfessionalVideoFormat format) const;
    bool requiresProprietarySoftware(ProfessionalVideoFormat format) const;
};

class ProfessionalVideoParser {
public:
    virtual ~ProfessionalVideoParser() = default;
    virtual bool canParse(const std::vector<uint8_t>& fileData) const = 0;
    virtual VideoFrameInfo parseFrameInfo(const std::vector<uint8_t>& fileData) const = 0;
    virtual VideoMetadata parseMetadata(const std::vector<uint8_t>& fileData) const = 0;
    virtual bool repair(std::vector<uint8_t>& fileData) const = 0;
    virtual std::string getDescription() const = 0;
    virtual bool validateStructure(const std::vector<uint8_t>& fileData) const = 0;
};