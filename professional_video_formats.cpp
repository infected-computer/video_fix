#include "professional_video_formats.h"
#include "red_camera_parser.h"
#include "arri_alexa_parser.h"
#include "blackmagic_parser.h"
#include <algorithm>
#include <fstream>

ProfessionalVideoDetector::ProfessionalVideoDetector() {
    initializeProfessionalSignatures();
    
    // Initialize parsers
    parsers[ProfessionalVideoFormat::R3D] = std::make_unique<REDCameraParser>();
    parsers[ProfessionalVideoFormat::RMD] = std::make_unique<REDCameraParser>();
    parsers[ProfessionalVideoFormat::ARI] = std::make_unique<ARRIAlexaParser>();
    parsers[ProfessionalVideoFormat::MXF_ARRI] = std::make_unique<ARRIAlexaParser>();
    parsers[ProfessionalVideoFormat::BRAW] = std::make_unique<BlackmagicParser>();
    parsers[ProfessionalVideoFormat::CINEMA_DNG] = std::make_unique<BlackmagicParser>();
}

ProfessionalVideoDetector::~ProfessionalVideoDetector() = default;

void ProfessionalVideoDetector::initializeProfessionalSignatures() {
    signatures = {
        // RED Digital Cinema
        {{0x52, 0x45, 0x44, 0x44}, 0, ProfessionalVideoFormat::R3D, ".r3d", "RED R3D File", "RED Digital Cinema", 1024, true},
        {{0x52, 0x4D, 0x44, 0x44}, 0, ProfessionalVideoFormat::RMD, ".rmd", "RED Metadata File", "RED Digital Cinema", 512, false},
        
        // ARRI ALEXA
        {{0x41, 0x52, 0x49, 0x20}, 0, ProfessionalVideoFormat::ARI, ".ari", "ARRI RAW File", "ARRI", 2048, true},
        {{0x06, 0x0E, 0x2B, 0x34}, 0, ProfessionalVideoFormat::MXF_ARRI, ".mxf", "ARRI MXF File", "ARRI", 1024, true},
        
        // Blackmagic Design
        {{0x42, 0x52, 0x41, 0x57}, 0, ProfessionalVideoFormat::BRAW, ".braw", "Blackmagic RAW", "Blackmagic Design", 1024, true},
        {{0x49, 0x49, 0x2A, 0x00}, 0, ProfessionalVideoFormat::CINEMA_DNG, ".dng", "CinemaDNG RAW", "Blackmagic Design", 512, true},
        {{0x4D, 0x4D, 0x00, 0x2A}, 0, ProfessionalVideoFormat::CINEMA_DNG, ".dng", "CinemaDNG RAW (BE)", "Blackmagic Design", 512, true},
        
        // Sony Professional
        {{0x58, 0x41, 0x56, 0x43}, 4, ProfessionalVideoFormat::XAVC, ".mp4", "Sony XAVC", "Sony", 1024, true},
        {{0x06, 0x0E, 0x2B, 0x34, 0x02, 0x05, 0x01, 0x01}, 0, ProfessionalVideoFormat::MXF_SONY, ".mxf", "Sony MXF", "Sony", 1024, true},
        
        // Canon Cinema
        {{0x43, 0x52, 0x4D, 0x00}, 0, ProfessionalVideoFormat::CRM, ".crm", "Canon RAW Material", "Canon", 2048, true},
        {{0x43, 0x34, 0x4B, 0x00}, 0, ProfessionalVideoFormat::C4K, ".c4k", "Canon 4K", "Canon", 1024, false},
        
        // Panasonic
        {{0x50, 0x32, 0x4D, 0x58}, 0, ProfessionalVideoFormat::P2_MXF, ".mxf", "Panasonic P2 MXF", "Panasonic", 1024, true},
        {{0x41, 0x56, 0x43, 0x48}, 0, ProfessionalVideoFormat::AVCHD_PRO, ".mts", "AVCHD Professional", "Panasonic", 512, false},
        
        // Apple ProRes
        {{0x69, 0x63, 0x70, 0x66}, 4, ProfessionalVideoFormat::PRORES_RAW, ".mov", "ProRes RAW", "Apple", 1024, true},
        {{0x61, 0x70, 0x63, 0x68}, 4, ProfessionalVideoFormat::PRORES_4444, ".mov", "ProRes 4444", "Apple", 512, false},
        {{0x61, 0x70, 0x63, 0x73}, 4, ProfessionalVideoFormat::PRORES_422, ".mov", "ProRes 422", "Apple", 512, false},
        
        // Avid DNxHD/HR
        {{0x44, 0x4E, 0x58, 0x48}, 0, ProfessionalVideoFormat::DNX_HD, ".dnxhd", "Avid DNxHD", "Avid", 512, false},
        {{0x44, 0x4E, 0x58, 0x52}, 0, ProfessionalVideoFormat::DNX_HR, ".dnxhr", "Avid DNxHR", "Avid", 512, false},
        
        // GoPro CineForm
        {{0x43, 0x46, 0x48, 0x44}, 0, ProfessionalVideoFormat::CINEFORM, ".avi", "GoPro CineForm", "GoPro", 512, false},
        
        // Atomos
        {{0x71, 0x74, 0x20, 0x20}, 4, ProfessionalVideoFormat::ATOMOS_MOV, ".mov", "Atomos ProRes", "Atomos", 512, false},
        {{0x4E, 0x49, 0x4E, 0x4A}, 0, ProfessionalVideoFormat::NINJA_RAW, ".raw", "Ninja RAW", "Atomos", 1024, true}
    };
}

ProfessionalVideoFormat ProfessionalVideoDetector::detectProfessionalFormat(const std::string& filePath) const {
    std::ifstream file(filePath, std::ios::binary);
    if (!file.is_open()) {
        return ProfessionalVideoFormat::UNKNOWN;
    }
    
    file.seekg(0, std::ios::end);
    size_t fileSize = file.tellg();
    file.seekg(0, std::ios::beg);
    
    size_t readSize = std::min(fileSize, static_cast<size_t>(4096));
    std::vector<uint8_t> buffer(readSize);
    file.read(reinterpret_cast<char*>(buffer.data()), readSize);
    
    return detectProfessionalFormat(buffer);
}

ProfessionalVideoFormat ProfessionalVideoDetector::detectProfessionalFormat(const std::vector<uint8_t>& fileData) const {
    for (const auto& sig : signatures) {
        if (fileData.size() >= sig.offset + sig.signature.size()) {
            if (std::equal(sig.signature.begin(), sig.signature.end(), 
                          fileData.begin() + sig.offset)) {
                
                if (sig.requiresSecondaryValidation) {
                    switch (sig.format) {
                        case ProfessionalVideoFormat::R3D:
                        case ProfessionalVideoFormat::RMD:
                            if (validateREDFormat(fileData)) return sig.format;
                            break;
                        case ProfessionalVideoFormat::ARI:
                        case ProfessionalVideoFormat::MXF_ARRI:
                            if (validateARRIFormat(fileData)) return sig.format;
                            break;
                        case ProfessionalVideoFormat::BRAW:
                        case ProfessionalVideoFormat::CINEMA_DNG:
                            if (validateBlackmagicFormat(fileData)) return sig.format;
                            break;
                        case ProfessionalVideoFormat::XAVC:
                        case ProfessionalVideoFormat::MXF_SONY:
                            if (validateSonyFormat(fileData)) return sig.format;
                            break;
                        case ProfessionalVideoFormat::CRM:
                        case ProfessionalVideoFormat::C4K:
                            if (validateCanonFormat(fileData)) return sig.format;
                            break;
                        case ProfessionalVideoFormat::P2_MXF:
                            if (validatePanasonicFormat(fileData)) return sig.format;
                            break;
                        default:
                            return sig.format;
                    }
                } else {
                    return sig.format;
                }
            }
        }
    }
    
    return ProfessionalVideoFormat::UNKNOWN;
}

bool ProfessionalVideoDetector::validateREDFormat(const std::vector<uint8_t>& data) const {
    return REDCameraParser::isREDFile(data);
}

bool ProfessionalVideoDetector::validateARRIFormat(const std::vector<uint8_t>& data) const {
    return ARRIAlexaParser::isARRIFile(data);
}

bool ProfessionalVideoDetector::validateBlackmagicFormat(const std::vector<uint8_t>& data) const {
    return BlackmagicParser::isBRAWFile(data) || BlackmagicParser::isCinemaDNGFile(data);
}

bool ProfessionalVideoDetector::validateSonyFormat(const std::vector<uint8_t>& data) const {
    // Sony format validation would require more detailed analysis
    return data.size() > 8;
}

bool ProfessionalVideoDetector::validateCanonFormat(const std::vector<uint8_t>& data) const {
    // Canon format validation would require more detailed analysis
    return data.size() > 8;
}

bool ProfessionalVideoDetector::validatePanasonicFormat(const std::vector<uint8_t>& data) const {
    // Panasonic format validation would require more detailed analysis
    return data.size() > 8;
}

VideoFrameInfo ProfessionalVideoDetector::extractFrameInfo(const std::string& filePath) const {
    ProfessionalVideoFormat format = detectProfessionalFormat(filePath);
    
    auto it = parsers.find(format);
    if (it != parsers.end()) {
        std::ifstream file(filePath, std::ios::binary);
        if (file.is_open()) {
            file.seekg(0, std::ios::end);
            size_t fileSize = file.tellg();
            file.seekg(0, std::ios::beg);
            
            std::vector<uint8_t> fileData(fileSize);
            file.read(reinterpret_cast<char*>(fileData.data()), fileSize);
            
            return it->second->parseFrameInfo(fileData);
        }
    }
    
    return VideoFrameInfo{};
}

VideoMetadata ProfessionalVideoDetector::extractMetadata(const std::string& filePath) const {
    ProfessionalVideoFormat format = detectProfessionalFormat(filePath);
    
    auto it = parsers.find(format);
    if (it != parsers.end()) {
        std::ifstream file(filePath, std::ios::binary);
        if (file.is_open()) {
            file.seekg(0, std::ios::end);
            size_t fileSize = file.tellg();
            file.seekg(0, std::ios::beg);
            
            std::vector<uint8_t> fileData(fileSize);
            file.read(reinterpret_cast<char*>(fileData.data()), fileSize);
            
            return it->second->parseMetadata(fileData);
        }
    }
    
    return VideoMetadata{};
}

bool ProfessionalVideoDetector::repairProfessionalVideo(const std::string& filePath, const std::string& outputPath) const {
    ProfessionalVideoFormat format = detectProfessionalFormat(filePath);
    
    auto it = parsers.find(format);
    if (it == parsers.end()) {
        return false;
    }
    
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
    
    bool repairSuccess = it->second->repair(fileData);
    
    if (repairSuccess) {
        std::string outPath = outputPath.empty() ? filePath + "_repaired" : outputPath;
        std::ofstream outFile(outPath, std::ios::binary);
        if (outFile.is_open()) {
            outFile.write(reinterpret_cast<const char*>(fileData.data()), fileData.size());
            outFile.close();
            return true;
        }
    }
    
    return false;
}

bool ProfessionalVideoDetector::validateProfessionalVideo(const std::string& filePath) const {
    ProfessionalVideoFormat format = detectProfessionalFormat(filePath);
    return format != ProfessionalVideoFormat::UNKNOWN;
}

std::string ProfessionalVideoDetector::getFormatDescription(ProfessionalVideoFormat format) const {
    for (const auto& sig : signatures) {
        if (sig.format == format) {
            return sig.description;
        }
    }
    return "Unknown Professional Format";
}

std::string ProfessionalVideoDetector::getManufacturer(ProfessionalVideoFormat format) const {
    for (const auto& sig : signatures) {
        if (sig.format == format) {
            return sig.manufacturer;
        }
    }
    return "Unknown";
}

std::string ProfessionalVideoDetector::getExpectedExtension(ProfessionalVideoFormat format) const {
    for (const auto& sig : signatures) {
        if (sig.format == format) {
            return sig.extension;
        }
    }
    return "";
}

std::vector<std::string> ProfessionalVideoDetector::getSupportedFormats() const {
    std::vector<std::string> formats;
    for (const auto& sig : signatures) {
        formats.push_back(sig.description + " (" + sig.extension + ")");
    }
    return formats;
}

bool ProfessionalVideoDetector::isRAWFormat(ProfessionalVideoFormat format) const {
    switch (format) {
        case ProfessionalVideoFormat::R3D:
        case ProfessionalVideoFormat::ARI:
        case ProfessionalVideoFormat::BRAW:
        case ProfessionalVideoFormat::CINEMA_DNG:
        case ProfessionalVideoFormat::CRM:
        case ProfessionalVideoFormat::PRORES_RAW:
        case ProfessionalVideoFormat::NINJA_RAW:
            return true;
        default:
            return false;
    }
}

bool ProfessionalVideoDetector::requiresProprietarySoftware(ProfessionalVideoFormat format) const {
    switch (format) {
        case ProfessionalVideoFormat::R3D:
        case ProfessionalVideoFormat::RMD:
        case ProfessionalVideoFormat::ARI:
        case ProfessionalVideoFormat::BRAW:
        case ProfessionalVideoFormat::CRM:
            return true;
        default:
            return false;
    }
}