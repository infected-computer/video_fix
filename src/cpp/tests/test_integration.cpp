#include <gtest/gtest.h>
#include <thread>
#include <future>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <vector>
#include <random>
#include "AdvancedVideoRepair/AdvancedVideoRepairEngine.h"
#include "AdvancedVideoRepair/ThreadSafeFrameBuffer.h"

using namespace AdvancedVideoRepair;

class IntegrationTest : public ::testing::Test {
protected:
    void SetUp() override {
        engine = std::make_unique<AdvancedVideoRepairEngine>();
        ASSERT_TRUE(engine->initialize());
        
        std::filesystem::create_directories("test_data");
        std::filesystem::create_directories("test_output");
        
        createTestFiles();
    }
    
    void TearDown() override {
        if (engine) {
            engine->shutdown();
        }
        std::filesystem::remove_all("test_output");
    }
    
    void createTestFiles() {
        // Create a larger MP4 file (simulated)
        create_large_mp4("test_data/large_test.mp4", 10 * 1024);  // 10KB simulated large file
        
        // Create multiple test files for parallel processing
        for (int i = 0; i < 5; i++) {
            std::string filename = "test_data/parallel_test_" + std::to_string(i) + ".mp4";
            create_test_mp4_with_corruption(filename, i % 3 == 0);  // Every 3rd file is corrupted
        }
        
        // Create corrupted large file
        create_corrupted_large_file("test_data/corrupted_large.mp4", 8 * 1024);
    }
    
    void create_large_mp4(const std::string& filename, size_t size_bytes) {
        std::ofstream file(filename, std::ios::binary);
        
        // Write ftyp box
        write_mp4_ftyp_box(file);
        
        // Write large mdat box with random data
        const size_t header_size = 32 + 16;  // ftyp + moov headers
        const size_t mdat_data_size = size_bytes - header_size - 8;  // subtract mdat header
        
        write_mp4_mdat_box(file, mdat_data_size);
        
        // Write moov box at the end
        write_mp4_moov_box(file);
        
        file.close();
    }
    
    void create_test_mp4_with_corruption(const std::string& filename, bool corrupt) {
        std::ofstream file(filename, std::ios::binary);
        
        if (!corrupt) {
            // Valid file
            write_mp4_ftyp_box(file);
            write_mp4_moov_box(file);
            write_mp4_mdat_box(file, 1024);
        } else {
            // Corrupted file
            write_corrupted_mp4_header(file);
            write_mp4_mdat_box(file, 512);
        }
        
        file.close();
    }
    
    void create_corrupted_large_file(const std::string& filename, size_t size_bytes) {
        std::ofstream file(filename, std::ios::binary);
        
        // Write valid start
        write_mp4_ftyp_box(file);
        
        // Write corrupted data in the middle
        std::vector<char> corrupt_data(size_bytes - 32, 0xFF);
        file.write(corrupt_data.data(), corrupt_data.size());
        
        file.close();
    }
    
    void write_mp4_ftyp_box(std::ofstream& file) {
        const char ftyp_box[] = {
            0x00, 0x00, 0x00, 0x20,  // size
            'f', 't', 'y', 'p',      // type
            'i', 's', 'o', 'm',      // major brand
            0x00, 0x00, 0x02, 0x00,  // minor version
            'i', 's', 'o', 'm',      // compatible brands
            'm', 'p', '4', '1',
            'a', 'v', 'c', '1',
            'd', 'a', 's', 'h'
        };
        file.write(ftyp_box, sizeof(ftyp_box));
    }
    
    void write_mp4_moov_box(std::ofstream& file) {
        const char moov_box[] = {
            0x00, 0x00, 0x00, 0x10,  // size
            'm', 'o', 'o', 'v',      // type
            0x00, 0x00, 0x00, 0x08,  // mvhd size
            'm', 'v', 'h', 'd'       // mvhd type
        };
        file.write(moov_box, sizeof(moov_box));
    }
    
    void write_mp4_mdat_box(std::ofstream& file, size_t data_size) {
        // Write mdat header
        uint32_t box_size = static_cast<uint32_t>(data_size + 8);
        uint32_t box_size_be = ((box_size & 0xFF) << 24) | 
                               (((box_size >> 8) & 0xFF) << 16) | 
                               (((box_size >> 16) & 0xFF) << 8) | 
                               ((box_size >> 24) & 0xFF);
        
        file.write(reinterpret_cast<const char*>(&box_size_be), 4);
        file.write("mdat", 4);
        
        // Write data
        std::vector<char> data(data_size, 0x42);  // Fill with 'B'
        file.write(data.data(), data.size());
    }
    
    void write_corrupted_mp4_header(std::ofstream& file) {
        const char corrupt_header[] = {
            0xFF, 0xFF, 0xFF, 0xFF,  // invalid size
            'X', 'X', 'X', 'X',      // invalid type
            0x00, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x00
        };
        file.write(corrupt_header, sizeof(corrupt_header));
    }
    
    std::unique_ptr<AdvancedVideoRepairEngine> engine;
};

// Large file processing tests
TEST_F(IntegrationTest, ProcessLargeFile) {
    RepairStrategy strategy;
    strategy.preserve_original_quality = true;
    
    auto start_time = std::chrono::steady_clock::now();
    
    AdvancedRepairResult result = engine->repair_video_file(
        "test_data/large_test.mp4",
        "test_output/large_repaired.mp4",
        strategy
    );
    
    auto end_time = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    
    // Test should complete within reasonable time (less than 30 seconds for test)
    EXPECT_LT(duration.count(), 30000);
    
    // Should not crash
    EXPECT_FALSE(result.input_file.empty());
    EXPECT_FALSE(result.output_file.empty());
    
    // Processing time should be recorded
    EXPECT_GE(result.processing_time.count(), 0);
}

TEST_F(IntegrationTest, AnalyzeLargeCorruptedFile) {
    auto start_time = std::chrono::steady_clock::now();
    
    CorruptionAnalysis analysis = engine->analyze_corruption("test_data/corrupted_large.mp4");
    
    auto end_time = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    
    // Should complete analysis within reasonable time
    EXPECT_LT(duration.count(), 10000);  // Less than 10 seconds
    
    // Should detect corruption
    EXPECT_FALSE(analysis.detected_issues.empty());
    EXPECT_GT(analysis.overall_corruption_percentage, 0.0);
    EXPECT_FALSE(analysis.detailed_report.empty());
}

// Parallel processing tests
TEST_F(IntegrationTest, ParallelFileProcessing) {
    const int num_files = 5;
    std::vector<std::future<AdvancedRepairResult>> futures;
    
    // Create multiple engines for parallel processing
    std::vector<std::unique_ptr<AdvancedVideoRepairEngine>> engines;
    for (int i = 0; i < num_files; i++) {
        engines.push_back(std::make_unique<AdvancedVideoRepairEngine>());
        ASSERT_TRUE(engines[i]->initialize());
    }
    
    auto start_time = std::chrono::steady_clock::now();
    
    // Launch parallel repair tasks
    for (int i = 0; i < num_files; i++) {
        std::string input_file = "test_data/parallel_test_" + std::to_string(i) + ".mp4";
        std::string output_file = "test_output/parallel_output_" + std::to_string(i) + ".mp4";
        
        futures.push_back(std::async(std::launch::async, [&, i, input_file, output_file]() {
            RepairStrategy strategy;
            return engines[i]->repair_video_file(input_file, output_file, strategy);
        }));
    }
    
    // Wait for all tasks to complete
    std::vector<AdvancedRepairResult> results;
    for (auto& future : futures) {
        results.push_back(future.get());
    }
    
    auto end_time = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    
    // All tasks should complete
    EXPECT_EQ(results.size(), num_files);
    
    // Should be faster than sequential processing (rough estimate)
    EXPECT_LT(duration.count(), 60000);  // Less than 1 minute total
    
    // All results should have proper input/output file names
    for (size_t i = 0; i < results.size(); i++) {
        EXPECT_FALSE(results[i].input_file.empty());
        EXPECT_FALSE(results[i].output_file.empty());
        EXPECT_GE(results[i].processing_time.count(), 0);
    }
    
    // Cleanup engines
    for (auto& engine : engines) {
        engine->shutdown();
    }
}

TEST_F(IntegrationTest, ParallelAnalysis) {
    const int num_files = 5;
    std::vector<std::future<CorruptionAnalysis>> futures;
    
    auto start_time = std::chrono::steady_clock::now();
    
    // Launch parallel analysis tasks
    for (int i = 0; i < num_files; i++) {
        std::string input_file = "test_data/parallel_test_" + std::to_string(i) + ".mp4";
        
        futures.push_back(std::async(std::launch::async, [this, input_file]() {
            return engine->analyze_corruption(input_file);
        }));
    }
    
    // Wait for all analyses to complete
    std::vector<CorruptionAnalysis> analyses;
    for (auto& future : futures) {
        analyses.push_back(future.get());
    }
    
    auto end_time = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    
    // All analyses should complete
    EXPECT_EQ(analyses.size(), num_files);
    
    // Should complete within reasonable time
    EXPECT_LT(duration.count(), 30000);  // Less than 30 seconds
    
    // All analyses should have valid reports
    for (const auto& analysis : analyses) {
        EXPECT_FALSE(analysis.detailed_report.empty());
        EXPECT_GE(analysis.overall_corruption_percentage, 0.0);
        EXPECT_LE(analysis.overall_corruption_percentage, 100.0);
    }
}

// Thread-safe frame buffer integration test
TEST_F(IntegrationTest, ConcurrentFrameBufferAccess) {
    const size_t buffer_capacity = 100;
    const size_t num_producers = 3;
    const size_t num_consumers = 2;
    const size_t frames_per_producer = 20;
    
    VideoRepair::ThreadSafeFrameBuffer frame_buffer(buffer_capacity);
    
    // Create test frames
    cv::Mat test_frame = cv::Mat::zeros(320, 240, CV_8UC3);
    test_frame.setTo(cv::Scalar(100, 150, 200));
    
    std::atomic<int> total_produced(0);
    std::atomic<int> total_consumed(0);
    std::vector<std::future<void>> producer_futures;
    std::vector<std::future<void>> consumer_futures;
    
    auto start_time = std::chrono::steady_clock::now();
    
    // Launch producer threads
    for (size_t p = 0; p < num_producers; p++) {
        producer_futures.push_back(std::async(std::launch::async, [&, p]() {
            for (size_t i = 0; i < frames_per_producer; i++) {
                cv::Mat frame = test_frame.clone();
                
                // Add unique marker to each frame
                cv::putText(frame, std::to_string(p * 1000 + i), cv::Point(10, 30), 
                           cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255, 255, 255), 2);
                
                if (frame_buffer.push_frame(frame)) {
                    total_produced.fetch_add(1);
                }
                
                // Small delay to simulate real processing
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
        }));
    }
    
    // Launch consumer threads
    for (size_t c = 0; c < num_consumers; c++) {
        consumer_futures.push_back(std::async(std::launch::async, [&]() {
            size_t local_consumed = 0;
            const size_t max_iterations = 1000;  // Prevent infinite loop
            
            for (size_t iter = 0; iter < max_iterations && local_consumed < frames_per_producer; iter++) {
                if (frame_buffer.size() > 0) {
                    cv::Mat frame = frame_buffer.get_frame(0);
                    if (!frame.empty()) {
                        local_consumed++;
                        total_consumed.fetch_add(1);
                    }
                }
                
                std::this_thread::sleep_for(std::chrono::milliseconds(2));
            }
        }));
    }
    
    // Wait for all producers to finish
    for (auto& future : producer_futures) {
        future.wait();
    }
    
    // Wait a bit for consumers to catch up
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    
    // Stop consumers (they should finish naturally)
    for (auto& future : consumer_futures) {
        future.wait();
    }
    
    auto end_time = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    
    // Verify results
    EXPECT_GT(total_produced.load(), 0);
    EXPECT_GT(total_consumed.load(), 0);
    EXPECT_LE(total_produced.load(), static_cast<int>(num_producers * frames_per_producer));
    
    // Should complete within reasonable time
    EXPECT_LT(duration.count(), 10000);  // Less than 10 seconds
    
    // Buffer should be in valid state
    EXPECT_LE(frame_buffer.size(), buffer_capacity);
    
    // Get statistics
    auto stats = frame_buffer.get_stats();
    EXPECT_EQ(stats.frame_count, frame_buffer.size());
    EXPECT_GT(stats.read_operations, 0u);
    EXPECT_GT(stats.write_operations, 0u);
}

// Memory stress test
TEST_F(IntegrationTest, MemoryStressTest) {
    const size_t num_iterations = 50;
    const size_t buffer_size = 1000;
    
    // Monitor memory usage pattern
    std::vector<size_t> memory_usage;
    
    for (size_t i = 0; i < num_iterations; i++) {
        VideoRepair::ThreadSafeFrameBuffer buffer(buffer_size);
        
        // Fill buffer with frames
        cv::Mat test_frame = cv::Mat::ones(640, 480, CV_8UC3) * (i % 255);
        
        for (size_t j = 0; j < buffer_size && j < 100; j++) {  // Limit to prevent excessive memory use
            buffer.push_frame(test_frame);
        }
        
        auto stats = buffer.get_stats();
        memory_usage.push_back(stats.total_bytes);
        
        // Clear buffer
        buffer.clear();
        
        // Verify cleanup
        EXPECT_EQ(buffer.size(), 0u);
        
        if (i % 10 == 0) {
            // Periodic progress check
            EXPECT_LT(i, num_iterations);
        }
    }
    
    // Should complete all iterations
    EXPECT_EQ(memory_usage.size(), num_iterations);
    
    // Memory usage should be reasonable
    for (size_t usage : memory_usage) {
        EXPECT_LT(usage, 500 * 1024 * 1024);  // Less than 500MB per buffer
    }
}

// Performance benchmark test
TEST_F(IntegrationTest, PerformanceBenchmark) {
    const int num_test_files = 3;
    std::vector<std::chrono::milliseconds> processing_times;
    
    for (int i = 0; i < num_test_files; i++) {
        std::string input_file = "test_data/parallel_test_" + std::to_string(i) + ".mp4";
        std::string output_file = "test_output/benchmark_" + std::to_string(i) + ".mp4";
        
        RepairStrategy strategy;
        strategy.enable_motion_compensation = (i % 2 == 0);  // Alternate settings
        
        auto start_time = std::chrono::steady_clock::now();
        
        AdvancedRepairResult result = engine->repair_video_file(input_file, output_file, strategy);
        
        auto end_time = std::chrono::steady_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        
        processing_times.push_back(duration);
        
        // Verify result
        EXPECT_FALSE(result.input_file.empty());
        EXPECT_GE(result.processing_time.count(), 0);
    }
    
    // Calculate average processing time
    auto total_time = std::chrono::milliseconds(0);
    for (const auto& time : processing_times) {
        total_time += time;
    }
    
    auto average_time = total_time / num_test_files;
    
    // Performance should be reasonable (less than 10 seconds average for test files)
    EXPECT_LT(average_time.count(), 10000);
    
    // No processing time should be extremely long
    for (const auto& time : processing_times) {
        EXPECT_LT(time.count(), 30000);  // Less than 30 seconds per file
    }
}

// Error handling under stress
TEST_F(IntegrationTest, ErrorHandlingStressTest) {
    const int num_invalid_operations = 20;
    int handled_errors = 0;
    
    for (int i = 0; i < num_invalid_operations; i++) {
        try {
            // Try various invalid operations
            switch (i % 4) {
                case 0:
                    // Invalid file path
                    engine->analyze_corruption("nonexistent_file_" + std::to_string(i) + ".mp4");
                    break;
                    
                case 1:
                    // Empty file path
                    engine->analyze_corruption("");
                    break;
                    
                case 2:
                    // Invalid repair attempt
                    {
                        RepairStrategy strategy;
                        engine->repair_video_file("invalid.mp4", "", strategy);
                    }
                    break;
                    
                case 3:
                    // Format detection on invalid file
                    engine->detect_container_format("test_data/corrupted_large.mp4");
                    break;
            }
            
            handled_errors++;
            
        } catch (const std::exception& e) {
            // Exceptions are acceptable for invalid operations
            handled_errors++;
        }
    }
    
    // All operations should be handled (either succeed or fail gracefully)
    EXPECT_EQ(handled_errors, num_invalid_operations);
    
    // Engine should still be functional after stress test
    EXPECT_TRUE(engine->is_initialized());
}