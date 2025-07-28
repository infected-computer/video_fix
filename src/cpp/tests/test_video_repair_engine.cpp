#include <gtest/gtest.h>
#include <fstream>
#include <vector>
#include <cstring>
#include "AdvancedVideoRepair/AdvancedVideoRepairEngine.h"
#include "AdvancedVideoRepair/FFmpegUtils.h"

using namespace AdvancedVideoRepair;

class VideoRepairEngineTest : public ::testing::Test {
protected:
    void SetUp() override {
        engine = std::make_unique<AdvancedVideoRepairEngine>();
        ASSERT_TRUE(engine->initialize()) << "Failed to initialize video repair engine";
        
        // Create test data directory if it doesn't exist
        std::filesystem::create_directories("test_data");
        std::filesystem::create_directories("test_output");
    }
    
    void TearDown() override {
        if (engine) {
            engine->shutdown();
        }
        
        // Clean up test output files
        std::filesystem::remove_all("test_output");
    }
    
    // Helper function to create a minimal valid MP4 file for testing
    void create_test_mp4(const std::string& filename, bool corrupt_header = false) {
        std::ofstream file(filename, std::ios::binary);
        
        if (!corrupt_header) {
            // Write minimal valid MP4 header (ftyp box)
            const char ftyp_box[] = {
                0x00, 0x00, 0x00, 0x20,  // box size (32 bytes)
                'f', 't', 'y', 'p',       // box type 'ftyp'
                'i', 's', 'o', 'm',       // major brand 'isom'
                0x00, 0x00, 0x02, 0x00,   // minor version
                'i', 's', 'o', 'm',       // compatible brand 'isom'
                'm', 'p', '4', '1',       // compatible brand 'mp41'
                'a', 'v', 'c', '1',       // compatible brand 'avc1'
                'd', 'a', 's', 'h'        // compatible brand 'dash'
            };
            file.write(ftyp_box, sizeof(ftyp_box));
            
            // Write minimal mdat box
            const char mdat_header[] = {
                0x00, 0x00, 0x00, 0x08,   // box size (8 bytes)
                'm', 'd', 'a', 't'        // box type 'mdat'
            };
            file.write(mdat_header, sizeof(mdat_header));
        } else {
            // Write corrupted header
            const char corrupt_header_data[] = {
                0xFF, 0xFF, 0xFF, 0xFF,   // Invalid box size
                'X', 'X', 'X', 'X',       // Invalid box type
                0x00, 0x00, 0x00, 0x00,
                0x00, 0x00, 0x00, 0x00
            };
            file.write(corrupt_header_data, sizeof(corrupt_header_data));
        }
    }
    
    std::unique_ptr<AdvancedVideoRepairEngine> engine;
};

// Basic initialization tests
TEST_F(VideoRepairEngineTest, InitializationSuccess) {
    EXPECT_TRUE(engine->is_initialized());
}

TEST_F(VideoRepairEngineTest, DoubleInitializationSafe) {
    EXPECT_TRUE(engine->initialize());  // Should return true even if already initialized
    EXPECT_TRUE(engine->is_initialized());
}

// File format detection tests
TEST_F(VideoRepairEngineTest, DetectMP4Format) {
    create_test_mp4("test_data/valid.mp4");
    
    ContainerFormat detected = engine->detect_container_format("test_data/valid.mp4");
    EXPECT_EQ(detected, ContainerFormat::MP4_ISOBMFF);
}

TEST_F(VideoRepairEngineTest, DetectUnknownFormat) {
    // Create file with unknown format
    std::ofstream file("test_data/unknown.bin", std::ios::binary);
    const char unknown_data[] = {0x12, 0x34, 0x56, 0x78, 0x9A, 0xBC, 0xDE, 0xF0};
    file.write(unknown_data, sizeof(unknown_data));
    file.close();
    
    ContainerFormat detected = engine->detect_container_format("test_data/unknown.bin");
    EXPECT_EQ(detected, ContainerFormat::UNKNOWN);
}

TEST_F(VideoRepairEngineTest, DetectNonExistentFile) {
    ContainerFormat detected = engine->detect_container_format("test_data/nonexistent.mp4");
    EXPECT_EQ(detected, ContainerFormat::UNKNOWN);
}

// Header validation tests
TEST_F(VideoRepairEngineTest, ValidateValidHeader) {
    create_test_mp4("test_data/valid_header.mp4");
    
    // Use reflection or create a public wrapper method for testing private methods
    // For now, test indirectly through analyze_corruption
    CorruptionAnalysis analysis = engine->analyze_corruption("test_data/valid_header.mp4");
    
    // Valid file should not have header damage
    bool has_header_damage = false;
    for (const auto& issue : analysis.detected_issues) {
        if (issue == CorruptionType::HEADER_DAMAGE) {
            has_header_damage = true;
            break;
        }
    }
    EXPECT_FALSE(has_header_damage);
}

TEST_F(VideoRepairEngineTest, DetectCorruptedHeader) {
    create_test_mp4("test_data/corrupt_header.mp4", true);
    
    CorruptionAnalysis analysis = engine->analyze_corruption("test_data/corrupt_header.mp4");
    
    // Corrupted file should have header damage
    bool has_header_damage = false;
    for (const auto& issue : analysis.detected_issues) {
        if (issue == CorruptionType::HEADER_DAMAGE) {
            has_header_damage = true;
            break;
        }
    }
    EXPECT_TRUE(has_header_damage);
}

// Corruption analysis tests
TEST_F(VideoRepairEngineTest, AnalyzeValidFile) {
    create_test_mp4("test_data/analysis_valid.mp4");
    
    CorruptionAnalysis analysis = engine->analyze_corruption("test_data/analysis_valid.mp4");
    
    EXPECT_TRUE(analysis.is_repairable);
    EXPECT_LT(analysis.overall_corruption_percentage, 50.0);
    EXPECT_FALSE(analysis.detailed_report.empty());
}

TEST_F(VideoRepairEngineTest, AnalyzeCorruptedFile) {
    create_test_mp4("test_data/analysis_corrupt.mp4", true);
    
    CorruptionAnalysis analysis = engine->analyze_corruption("test_data/analysis_corrupt.mp4");
    
    EXPECT_FALSE(analysis.detected_issues.empty());
    EXPECT_FALSE(analysis.detailed_report.empty());
    EXPECT_GT(analysis.overall_corruption_percentage, 0.0);
}

TEST_F(VideoRepairEngineTest, AnalyzeNonExistentFile) {
    CorruptionAnalysis analysis = engine->analyze_corruption("test_data/nonexistent.mp4");
    
    EXPECT_FALSE(analysis.is_repairable);
    EXPECT_FALSE(analysis.detailed_report.empty());
}

// Video codec detection tests
TEST_F(VideoRepairEngineTest, DetectCodecFromNonExistentFile) {
    VideoCodec codec = engine->detect_video_codec("test_data/nonexistent.mp4");
    EXPECT_EQ(codec, VideoCodec::UNKNOWN_CODEC);
}

// Repair functionality tests
TEST_F(VideoRepairEngineTest, RepairNonExistentFile) {
    RepairStrategy strategy;
    AdvancedRepairResult result = engine->repair_video_file(
        "test_data/nonexistent.mp4", 
        "test_output/repaired.mp4", 
        strategy
    );
    
    EXPECT_FALSE(result.success);
    EXPECT_FALSE(result.error_message.empty());
}

TEST_F(VideoRepairEngineTest, RepairWithValidInput) {
    create_test_mp4("test_data/repair_input.mp4");
    
    RepairStrategy strategy;
    strategy.preserve_original_quality = true;
    strategy.enable_motion_compensation = false;  // Disable complex features for basic test
    
    AdvancedRepairResult result = engine->repair_video_file(
        "test_data/repair_input.mp4", 
        "test_output/repaired_basic.mp4", 
        strategy
    );
    
    // Basic file might not be repairable, but should not crash
    EXPECT_FALSE(result.input_file.empty());
    EXPECT_FALSE(result.output_file.empty());
    EXPECT_GE(result.processing_time.count(), 0);
}

// Configuration tests
TEST_F(VideoRepairEngineTest, SetThreadCount) {
    engine->set_thread_count(4);
    // No direct way to verify, but should not crash
    EXPECT_TRUE(engine->is_initialized());
}

TEST_F(VideoRepairEngineTest, EnableGPUAcceleration) {
    engine->enable_gpu_acceleration(true);
    // No direct way to verify, but should not crash
    EXPECT_TRUE(engine->is_initialized());
}

TEST_F(VideoRepairEngineTest, SetMemoryLimit) {
    engine->set_memory_limit_mb(2048);
    // No direct way to verify, but should not crash
    EXPECT_TRUE(engine->is_initialized());
}

TEST_F(VideoRepairEngineTest, SetLogLevel) {
    engine->set_log_level(1);
    // No direct way to verify, but should not crash
    EXPECT_TRUE(engine->is_initialized());
}

// Progress callback tests
TEST_F(VideoRepairEngineTest, ProgressCallbackSetup) {
    bool callback_called = false;
    std::string last_status;
    double last_progress = -1.0;
    
    engine->set_progress_callback([&](double progress, const std::string& status) {
        callback_called = true;
        last_progress = progress;
        last_status = status;
    });
    
    create_test_mp4("test_data/progress_test.mp4");
    RepairStrategy strategy;
    
    AdvancedRepairResult result = engine->repair_video_file(
        "test_data/progress_test.mp4", 
        "test_output/progress_output.mp4", 
        strategy
    );
    
    // Callback should have been called at least once during processing
    EXPECT_TRUE(callback_called);
    EXPECT_GE(last_progress, 0.0);
    EXPECT_LE(last_progress, 1.0);
    EXPECT_FALSE(last_status.empty());
}

// Edge case tests
TEST_F(VideoRepairEngineTest, EmptyFilePaths) {
    RepairStrategy strategy;
    AdvancedRepairResult result = engine->repair_video_file("", "", strategy);
    
    EXPECT_FALSE(result.success);
    EXPECT_FALSE(result.error_message.empty());
}

TEST_F(VideoRepairEngineTest, AnalyzeEmptyFilePath) {
    CorruptionAnalysis analysis = engine->analyze_corruption("");
    
    EXPECT_FALSE(analysis.is_repairable);
    EXPECT_FALSE(analysis.detailed_report.empty());
}

// Shutdown and cleanup tests
TEST_F(VideoRepairEngineTest, ShutdownAndReinitialize) {
    engine->shutdown();
    EXPECT_FALSE(engine->is_initialized());
    
    EXPECT_TRUE(engine->initialize());
    EXPECT_TRUE(engine->is_initialized());
}

TEST_F(VideoRepairEngineTest, MultipleShutdownsSafe) {
    engine->shutdown();
    engine->shutdown();  // Should be safe to call multiple times
    EXPECT_FALSE(engine->is_initialized());
}