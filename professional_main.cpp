#include "professional_video_formats.h"
#include <iostream>
#include <string>
#include <chrono>
#include <iomanip>

void printProfessionalUsage() {
    std::cout << "Professional Video Format Repair Tool\n";
    std::cout << "Usage: professional_video_repair [options] <file_path>\n\n";
    std::cout << "Options:\n";
    std::cout << "  -d, --detect     Detect professional video format\n";
    std::cout << "  -r, --repair     Repair professional video file (default)\n";
    std::cout << "  -v, --validate   Validate professional video format\n";
    std::cout << "  -m, --metadata   Extract detailed metadata\n";
    std::cout << "  -f, --frameinfo  Extract frame information\n";
    std::cout << "  -l, --list       List supported formats\n";
    std::cout << "  -o <output>      Specify output file path\n";
    std::cout << "  -h, --help       Show this help\n\n";
    std::cout << "Supported Professional Formats:\n";
    std::cout << "  RED Digital Cinema: R3D, RMD\n";
    std::cout << "  ARRI ALEXA: ARI, MXF\n";
    std::cout << "  Blackmagic: BRAW, CinemaDNG\n";
    std::cout << "  Sony: XAVC, MXF\n";
    std::cout << "  Canon: CRM, C4K, XF-AVC\n";
    std::cout << "  Panasonic: P2 MXF, AVCHD-Pro\n";
    std::cout << "  Apple: ProRes RAW, ProRes 4444/422\n";
    std::cout << "  Avid: DNxHD, DNxHR\n";
    std::cout << "  Others: CineForm, Atomos formats\n";
}

std::string professionalFormatToString(ProfessionalVideoFormat format) {
    switch (format) {
        case ProfessionalVideoFormat::R3D: return "RED R3D";
        case ProfessionalVideoFormat::RMD: return "RED RMD";
        case ProfessionalVideoFormat::ARI: return "ARRI RAW";
        case ProfessionalVideoFormat::MXF_ARRI: return "ARRI MXF";
        case ProfessionalVideoFormat::BRAW: return "Blackmagic RAW";
        case ProfessionalVideoFormat::CINEMA_DNG: return "CinemaDNG";
        case ProfessionalVideoFormat::XAVC: return "Sony XAVC";
        case ProfessionalVideoFormat::MXF_SONY: return "Sony MXF";
        case ProfessionalVideoFormat::F55_RAW: return "Sony F55 RAW";
        case ProfessionalVideoFormat::F65_RAW: return "Sony F65 RAW";
        case ProfessionalVideoFormat::CRM: return "Canon RAW Material";
        case ProfessionalVideoFormat::C4K: return "Canon 4K";
        case ProfessionalVideoFormat::XF_AVC: return "Canon XF-AVC";
        case ProfessionalVideoFormat::P2_MXF: return "Panasonic P2 MXF";
        case ProfessionalVideoFormat::AVCHD_PRO: return "AVCHD Professional";
        case ProfessionalVideoFormat::AVC_INTRA: return "AVC-Intra";
        case ProfessionalVideoFormat::PRORES_RAW: return "ProRes RAW";
        case ProfessionalVideoFormat::PRORES_4444: return "ProRes 4444";
        case ProfessionalVideoFormat::PRORES_422: return "ProRes 422";
        case ProfessionalVideoFormat::DNX_HD: return "Avid DNxHD";
        case ProfessionalVideoFormat::DNX_HR: return "Avid DNxHR";
        case ProfessionalVideoFormat::CINEFORM: return "GoPro CineForm";
        case ProfessionalVideoFormat::ATOMOS_MOV: return "Atomos ProRes";
        case ProfessionalVideoFormat::NINJA_RAW: return "Ninja RAW";
        default: return "Unknown";
    }
}

void printFrameInfo(const VideoFrameInfo& info) {
    std::cout << "\n=== Frame Information ===\n";
    std::cout << "Resolution: " << info.width << "x" << info.height << "\n";
    std::cout << "Bit Depth: " << info.bitDepth << " bits\n";
    std::cout << "Frame Rate: " << std::fixed << std::setprecision(3) << info.frameRate << " fps\n";
    std::cout << "Frame Count: " << info.frameCount << "\n";
    std::cout << "Color Space: " << info.colorSpace << "\n";
    std::cout << "Codec: " << info.codec << "\n";
    
    if (info.timecode > 0) {
        uint32_t hours = (info.timecode >> 24) & 0xFF;
        uint32_t minutes = (info.timecode >> 16) & 0xFF;
        uint32_t seconds = (info.timecode >> 8) & 0xFF;
        uint32_t frames = info.timecode & 0xFF;
        std::cout << "Timecode: " << std::setfill('0') << std::setw(2) << hours << ":"
                  << std::setw(2) << minutes << ":" << std::setw(2) << seconds << ":"
                  << std::setw(2) << frames << "\n";
    }
}

void printMetadata(const VideoMetadata& metadata) {
    std::cout << "\n=== Camera Metadata ===\n";
    if (!metadata.camera.empty()) std::cout << "Camera: " << metadata.camera << "\n";
    if (!metadata.lens.empty()) std::cout << "Lens: " << metadata.lens << "\n";
    if (!metadata.iso.empty()) std::cout << "ISO: " << metadata.iso << "\n";
    if (!metadata.shutterSpeed.empty()) std::cout << "Shutter Speed: " << metadata.shutterSpeed << "\n";
    if (!metadata.aperture.empty()) std::cout << "Aperture: " << metadata.aperture << "\n";
    if (!metadata.whiteBalance.empty()) std::cout << "White Balance: " << metadata.whiteBalance << "\n";
    if (!metadata.recordingFormat.empty()) std::cout << "Recording Format: " << metadata.recordingFormat << "\n";
    if (!metadata.lutApplied.empty()) std::cout << "LUT Applied: " << metadata.lutApplied << "\n";
    
    if (!metadata.customMetadata.empty()) {
        std::cout << "\n=== Additional Metadata ===\n";
        for (const auto& pair : metadata.customMetadata) {
            if (!pair.second.empty()) {
                std::cout << pair.first << ": " << pair.second << "\n";
            }
        }
    }
}

void listSupportedFormats() {
    ProfessionalVideoDetector detector;
    auto formats = detector.getSupportedFormats();
    
    std::cout << "\n=== Supported Professional Video Formats ===\n";
    for (const auto& format : formats) {
        std::cout << "  • " << format << "\n";
    }
    std::cout << "\nTotal: " << formats.size() << " formats supported\n";
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        printProfessionalUsage();
        return 1;
    }
    
    std::string filePath;
    std::string outputPath;
    bool detectOnly = false;
    bool validateOnly = false;
    bool extractMetadata = false;
    bool extractFrameInfo = false;
    bool listFormats = false;
    bool repair = true;
    
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        
        if (arg == "-h" || arg == "--help") {
            printProfessionalUsage();
            return 0;
        } else if (arg == "-d" || arg == "--detect") {
            detectOnly = true;
            repair = false;
        } else if (arg == "-v" || arg == "--validate") {
            validateOnly = true;
            repair = false;
        } else if (arg == "-m" || arg == "--metadata") {
            extractMetadata = true;
            repair = false;
        } else if (arg == "-f" || arg == "--frameinfo") {
            extractFrameInfo = true;
            repair = false;
        } else if (arg == "-l" || arg == "--list") {
            listFormats = true;
            repair = false;
        } else if (arg == "-r" || arg == "--repair") {
            repair = true;
        } else if (arg == "-o" && i + 1 < argc) {
            outputPath = argv[++i];
        } else if (arg[0] != '-') {
            filePath = arg;
        }
    }
    
    if (listFormats) {
        listSupportedFormats();
        return 0;
    }
    
    if (filePath.empty()) {
        std::cerr << "Error: No file path specified\n";
        return 1;
    }
    
    auto start = std::chrono::high_resolution_clock::now();
    
    ProfessionalVideoDetector detector;
    
    if (detectOnly) {
        ProfessionalVideoFormat format = detector.detectProfessionalFormat(filePath);
        std::cout << "Detected format: " << professionalFormatToString(format) << "\n";
        std::cout << "Description: " << detector.getFormatDescription(format) << "\n";
        std::cout << "Manufacturer: " << detector.getManufacturer(format) << "\n";
        std::cout << "Expected extension: " << detector.getExpectedExtension(format) << "\n";
        std::cout << "Is RAW format: " << (detector.isRAWFormat(format) ? "Yes" : "No") << "\n";
        std::cout << "Requires proprietary software: " << (detector.requiresProprietarySoftware(format) ? "Yes" : "No") << "\n";
        
    } else if (validateOnly) {
        bool isValid = detector.validateProfessionalVideo(filePath);
        std::cout << "File is " << (isValid ? "valid professional video format" : "invalid or unknown format") << "\n";
        return isValid ? 0 : 1;
        
    } else if (extractFrameInfo) {
        ProfessionalVideoFormat format = detector.detectProfessionalFormat(filePath);
        std::cout << "Format: " << professionalFormatToString(format) << "\n";
        
        if (format != ProfessionalVideoFormat::UNKNOWN) {
            VideoFrameInfo frameInfo = detector.extractFrameInfo(filePath);
            printFrameInfo(frameInfo);
        } else {
            std::cout << "Cannot extract frame info: Unknown format\n";
        }
        
    } else if (extractMetadata) {
        ProfessionalVideoFormat format = detector.detectProfessionalFormat(filePath);
        std::cout << "Format: " << professionalFormatToString(format) << "\n";
        
        if (format != ProfessionalVideoFormat::UNKNOWN) {
            VideoFrameInfo frameInfo = detector.extractFrameInfo(filePath);
            VideoMetadata metadata = detector.extractMetadata(filePath);
            
            printFrameInfo(frameInfo);
            printMetadata(metadata);
        } else {
            std::cout << "Cannot extract metadata: Unknown format\n";
        }
        
    } else if (repair) {
        ProfessionalVideoFormat format = detector.detectProfessionalFormat(filePath);
        std::cout << "Detected format: " << professionalFormatToString(format) << "\n";
        
        if (format == ProfessionalVideoFormat::UNKNOWN) {
            std::cerr << "Cannot repair: Unknown professional video format\n";
            return 1;
        }
        
        std::cout << "Attempting to repair professional video file...\n";
        bool success = detector.repairProfessionalVideo(filePath, outputPath);
        
        if (success) {
            std::cout << "Professional video file repaired successfully!\n";
            if (!outputPath.empty()) {
                std::cout << "Repaired file saved as: " << outputPath << "\n";
            } else {
                std::cout << "Repaired file saved as: " << filePath << "_repaired\n";
            }
        } else {
            std::cout << "Professional video repair failed or no repairs needed\n";
        }
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "\nOperation completed in " << duration.count() << "ms\n";
    
    return 0;
}