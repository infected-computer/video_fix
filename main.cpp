#include "file_format_detector.h"
#include <iostream>
#include <string>
#include <chrono>

void printUsage() {
    std::cout << "Usage: file_repair [options] <file_path>\n";
    std::cout << "Options:\n";
    std::cout << "  -d, --detect    Detect file format only\n";
    std::cout << "  -r, --repair    Repair file (default)\n";
    std::cout << "  -v, --validate  Validate file format\n";
    std::cout << "  -o <output>     Specify output file path\n";
    std::cout << "  -h, --help      Show this help\n";
}

std::string formatToString(FileFormat format) {
    switch (format) {
        case FileFormat::JPEG: return "JPEG";
        case FileFormat::PNG: return "PNG";
        case FileFormat::GIF: return "GIF";
        case FileFormat::BMP: return "BMP";
        case FileFormat::TIFF: return "TIFF";
        case FileFormat::PDF: return "PDF";
        case FileFormat::ZIP: return "ZIP";
        case FileFormat::RAR: return "RAR";
        case FileFormat::MP4: return "MP4";
        case FileFormat::AVI: return "AVI";
        case FileFormat::MKV: return "MKV";
        case FileFormat::MP3: return "MP3";
        case FileFormat::WAV: return "WAV";
        case FileFormat::FLAC: return "FLAC";
        case FileFormat::DOC: return "DOC";
        case FileFormat::DOCX: return "DOCX";
        case FileFormat::XLS: return "XLS";
        case FileFormat::XLSX: return "XLSX";
        case FileFormat::PPT: return "PPT";
        case FileFormat::PPTX: return "PPTX";
        default: return "UNKNOWN";
    }
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        printUsage();
        return 1;
    }
    
    std::string filePath;
    std::string outputPath;
    bool detectOnly = false;
    bool validateOnly = false;
    bool repair = true;
    
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        
        if (arg == "-h" || arg == "--help") {
            printUsage();
            return 0;
        } else if (arg == "-d" || arg == "--detect") {
            detectOnly = true;
            repair = false;
        } else if (arg == "-v" || arg == "--validate") {
            validateOnly = true;
            repair = false;
        } else if (arg == "-r" || arg == "--repair") {
            repair = true;
        } else if (arg == "-o" && i + 1 < argc) {
            outputPath = argv[++i];
        } else if (arg[0] != '-') {
            filePath = arg;
        }
    }
    
    if (filePath.empty()) {
        std::cerr << "Error: No file path specified\n";
        return 1;
    }
    
    auto start = std::chrono::high_resolution_clock::now();
    
    FileFormatDetector detector;
    
    if (detectOnly) {
        FileFormat format = detector.detectFormat(filePath);
        std::cout << "Detected format: " << formatToString(format) << "\n";
        std::cout << "Description: " << detector.getFormatDescription(format) << "\n";
        std::cout << "Expected extension: " << detector.getExpectedExtension(format) << "\n";
    } else if (validateOnly) {
        bool isValid = detector.validateFile(filePath);
        std::cout << "File is " << (isValid ? "valid" : "invalid or unknown format") << "\n";
        return isValid ? 0 : 1;
    } else if (repair) {
        FileFormat format = detector.detectFormat(filePath);
        std::cout << "Detected format: " << formatToString(format) << "\n";
        
        if (format == FileFormat::UNKNOWN) {
            std::cerr << "Cannot repair: Unknown file format\n";
            return 1;
        }
        
        std::cout << "Attempting to repair file...\n";
        bool success = detector.repairFile(filePath, outputPath);
        
        if (success) {
            std::cout << "File repaired successfully!\n";
            if (!outputPath.empty()) {
                std::cout << "Repaired file saved as: " << outputPath << "\n";
            } else {
                std::cout << "Repaired file saved as: " << filePath << "_repaired\n";
            }
        } else {
            std::cout << "File repair failed or no repairs needed\n";
        }
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "Operation completed in " << duration.count() << "ms\n";
    
    return 0;
}