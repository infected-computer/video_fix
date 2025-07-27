#include "AdvancedVideoRepair/AdvancedVideoRepairEngine.h"
#include <iostream>
#include <string>
#include <chrono>

using namespace AdvancedVideoRepair;

/**
 * @brief Demo application for Advanced Video Repair Engine
 * 
 * This demonstrates the sophisticated video repair capabilities:
 * - Deep corruption analysis
 * - Multi-strategy repair approaches
 * - Quality assessment and validation
 */

void print_usage(const char* program_name) {
    std::cout << "Advanced Video Repair Engine Demo\n";
    std::cout << "=================================\n\n";
    std::cout << "Usage: " << program_name << " <input_file> <output_file> [options]\n\n";
    std::cout << "Options:\n";
    std::cout << "  --fast           Use fast repair mode (lower quality)\n";
    std::cout << "  --high-quality   Use high quality mode (slower)\n";
    std::cout << "  --gpu            Enable GPU acceleration\n";
    std::cout << "  --threads N      Use N threads (default: auto)\n";
    std::cout << "  --analyze-only   Only analyze corruption, don't repair\n";
    std::cout << "\nExamples:\n";
    std::cout << "  " << program_name << " corrupted.mp4 repaired.mp4\n";
    std::cout << "  " << program_name << " damaged.avi fixed.avi --high-quality --gpu\n";
    std::cout << "  " << program_name << " broken.mkv analyze --analyze-only\n";
}

void print_corruption_analysis(const CorruptionAnalysis& analysis) {
    std::cout << "\n=== Corruption Analysis Results ===\n";
    std::cout << "Overall corruption: " << analysis.overall_corruption_percentage << "%\n";
    std::cout << "Is repairable: " << (analysis.is_repairable ? "YES" : "NO") << "\n";
    std::cout << "Detected issues: " << analysis.detected_issues.size() << "\n";
    
    for (const auto& issue : analysis.detected_issues) {
        std::cout << "  - ";
        switch (issue) {
            case CorruptionType::CONTAINER_STRUCTURE:
                std::cout << "Container structure corruption";
                break;
            case CorruptionType::BITSTREAM_ERRORS:
                std::cout << "Bitstream errors";
                break;
            case CorruptionType::MISSING_FRAMES:
                std::cout << "Missing frames";
                break;
            case CorruptionType::SYNC_LOSS:
                std::cout << "Audio/video sync loss";
                break;
            case CorruptionType::INDEX_CORRUPTION:
                std::cout << "Index corruption";
                break;
            case CorruptionType::HEADER_DAMAGE:
                std::cout << "Header damage";
                break;
            case CorruptionType::INCOMPLETE_FRAMES:
                std::cout << "Incomplete frames";
                break;
            case CorruptionType::TEMPORAL_ARTIFACTS:
                std::cout << "Temporal artifacts";
                break;
        }
        std::cout << "\n";
    }
    
    if (analysis.corrupted_frame_numbers.size() > 0) {
        std::cout << "Corrupted frames: " << analysis.corrupted_frame_numbers.size() << "\n";
        if (analysis.corrupted_frame_numbers.size() <= 10) {
            std::cout << "Frame numbers: ";
            for (size_t i = 0; i < analysis.corrupted_frame_numbers.size(); i++) {
                std::cout << analysis.corrupted_frame_numbers[i];
                if (i < analysis.corrupted_frame_numbers.size() - 1) std::cout << ", ";
            }
            std::cout << "\n";
        } else {
            std::cout << "First 10 corrupted frames: ";
            for (int i = 0; i < 10; i++) {
                std::cout << analysis.corrupted_frame_numbers[i];
                if (i < 9) std::cout << ", ";
            }
            std::cout << "...\n";
        }
    }
    
    std::cout << "\nDetailed report:\n" << analysis.detailed_report << "\n";
    
    // Container-specific issues
    if (analysis.container_issues.missing_moov_atom ||
        analysis.container_issues.corrupted_mdat_atom ||
        analysis.container_issues.invalid_chunk_offsets) {
        
        std::cout << "\nContainer Issues:\n";
        if (analysis.container_issues.missing_moov_atom) {
            std::cout << "  - Missing moov atom (metadata)\n";
        }
        if (analysis.container_issues.corrupted_mdat_atom) {
            std::cout << "  - Corrupted mdat atom (media data)\n";
        }
        if (analysis.container_issues.invalid_chunk_offsets) {
            std::cout << "  - Invalid chunk offsets\n";
        }
        if (!analysis.container_issues.missing_required_boxes.empty()) {
            std::cout << "  - Missing required boxes: ";
            for (size_t i = 0; i < analysis.container_issues.missing_required_boxes.size(); i++) {
                std::cout << analysis.container_issues.missing_required_boxes[i];
                if (i < analysis.container_issues.missing_required_boxes.size() - 1) std::cout << ", ";
            }
            std::cout << "\n";
        }
    }
    
    // Bitstream-specific issues
    if (analysis.bitstream_issues.corrupted_macroblocks > 0 ||
        analysis.bitstream_issues.missing_reference_frames > 0 ||
        analysis.bitstream_issues.corrupted_sps_pps) {
        
        std::cout << "\nBitstream Issues:\n";
        if (analysis.bitstream_issues.corrupted_macroblocks > 0) {
            std::cout << "  - Corrupted macroblocks: " << analysis.bitstream_issues.corrupted_macroblocks << "\n";
        }
        if (analysis.bitstream_issues.missing_reference_frames > 0) {
            std::cout << "  - Missing reference frames: " << analysis.bitstream_issues.missing_reference_frames << "\n";
        }
        if (analysis.bitstream_issues.corrupted_sps_pps) {
            std::cout << "  - Corrupted SPS/PPS headers\n";
        }
    }
}

void print_repair_result(const AdvancedRepairResult& result) {
    std::cout << "\n=== Repair Results ===\n";
    std::cout << "Success: " << (result.success ? "YES" : "NO") << "\n";
    
    if (!result.success) {
        std::cout << "Error: " << result.error_message << "\n";
        return;
    }
    
    std::cout << "Processing time: " << result.processing_time.count() << " ms\n";
    std::cout << "Frames reconstructed: " << result.frames_reconstructed << "\n";
    std::cout << "Frames interpolated: " << result.frames_interpolated << "\n";
    std::cout << "Bytes repaired: " << result.bytes_repaired << "\n";
    
    // Quality metrics
    std::cout << "\nQuality Improvements:\n";
    std::cout << "  PSNR improvement: " << result.quality_metrics.psnr_improvement << " dB\n";
    std::cout << "  SSIM improvement: " << result.quality_metrics.ssim_improvement << "\n";
    std::cout << "  Temporal consistency: " << result.quality_metrics.temporal_consistency_score << "\n";
    std::cout << "  Artifact reduction: " << result.quality_metrics.artifact_reduction_percentage << "%\n";
    
    // Repairs performed
    if (!result.repairs_performed.empty()) {
        std::cout << "\nRepairs performed:\n";
        for (const auto& repair : result.repairs_performed) {
            std::cout << "  - " << repair << "\n";
        }
    }
    
    // Warnings
    if (!result.warnings.empty()) {
        std::cout << "\nWarnings:\n";
        for (const auto& warning : result.warnings) {
            std::cout << "  ! " << warning << "\n";
        }
    }
    
    // Validation
    std::cout << "\nValidation:\n";
    std::cout << "  Output playable: " << (result.output_playable ? "YES" : "NO") << "\n";
    std::cout << "  Audio sync maintained: " << (result.audio_sync_maintained ? "YES" : "NO") << "\n";
    std::cout << "  Validation report: " << result.validation_report << "\n";
}

int main(int argc, char* argv[]) {
    std::cout << "Advanced Video Repair Engine Demo\n";
    std::cout << "==================================\n";
    std::cout << "Sophisticated video repair with temporal processing\n\n";
    
    if (argc < 3) {
        print_usage(argv[0]);
        return 1;
    }
    
    std::string input_file = argv[1];
    std::string output_file = argv[2];
    
    // Parse command line options
    RepairStrategy strategy;
    bool analyze_only = false;
    bool fast_mode = false;
    bool high_quality = false;
    
    for (int i = 3; i < argc; i++) {
        std::string arg = argv[i];
        
        if (arg == "--fast") {
            fast_mode = true;
        } else if (arg == "--high-quality") {
            high_quality = true;
        } else if (arg == "--gpu") {
            strategy.use_gpu_acceleration = true;
        } else if (arg == "--threads" && i + 1 < argc) {
            strategy.thread_count = std::stoi(argv[++i]);
        } else if (arg == "--analyze-only") {
            analyze_only = true;
        }
    }
    
    // Configure strategy based on options
    if (fast_mode) {
        strategy.use_temporal_analysis = false;
        strategy.enable_motion_compensation = false;
        strategy.enable_post_processing = false;
        strategy.max_interpolation_distance = 2;
        strategy.error_concealment_strength = 0.5;
        std::cout << "Using FAST repair mode\n";
    } else if (high_quality) {
        strategy.use_temporal_analysis = true;
        strategy.enable_motion_compensation = true;
        strategy.enable_post_processing = true;
        strategy.max_interpolation_distance = 8;
        strategy.error_concealment_strength = 0.9;
        std::cout << "Using HIGH QUALITY repair mode\n";
    } else {
        std::cout << "Using BALANCED repair mode\n";
    }
    
    // Initialize repair engine
    AdvancedVideoRepairEngine engine;
    
    std::cout << "Initializing repair engine...\n";
    if (!engine.initialize()) {
        std::cerr << "Failed to initialize repair engine!\n";
        return 1;
    }
    
    // Setup progress callback
    engine.set_progress_callback([](double progress, const std::string& status) {
        std::cout << "\r[" << std::fixed << std::setprecision(1) << (progress * 100) 
                  << "%] " << status << std::flush;
    });
    
    try {
        // Phase 1: Analyze corruption
        std::cout << "Analyzing file: " << input_file << "\n";
        CorruptionAnalysis analysis = engine.analyze_corruption(input_file);
        print_corruption_analysis(analysis);
        
        if (analyze_only) {
            std::cout << "\nAnalysis complete (analyze-only mode).\n";
            engine.shutdown();
            return 0;
        }
        
        if (!analysis.is_repairable) {
            std::cout << "\nFile is too severely corrupted for repair.\n";
            engine.shutdown();
            return 1;
        }
        
        // Phase 2: Perform repair
        std::cout << "\n\nStarting repair process...\n";
        auto start_time = std::chrono::high_resolution_clock::now();
        
        AdvancedRepairResult result = engine.repair_video_file(input_file, output_file, strategy);
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto total_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        
        std::cout << "\n"; // New line after progress
        print_repair_result(result);
        
        std::cout << "\nTotal execution time: " << total_time.count() << " ms\n";
        
        if (result.success) {
            std::cout << "\n✓ Repair completed successfully!\n";
            std::cout << "Output file: " << output_file << "\n";
        } else {
            std::cout << "\n✗ Repair failed.\n";
            return 1;
        }
        
    } catch (const std::exception& e) {
        std::cerr << "\nError: " << e.what() << "\n";
        engine.shutdown();
        return 1;
    }
    
    engine.shutdown();
    return 0;
}