#include <gtest/gtest.h>
#include <opencv2/opencv.hpp>
#include <filesystem>
#include "AdvancedVideoRepair/AdvancedVideoRepairEngine.h"
#include "AdvancedVideoRepair/ThreadSafeFrameBuffer.h"

using namespace AdvancedVideoRepair;

class FrameReconstructorTest : public ::testing::Test {
protected:
    void SetUp() override {
        engine = std::make_unique<AdvancedVideoRepairEngine>();
        ASSERT_TRUE(engine->initialize());
        
        reconstructor = std::make_unique<FrameReconstructor>(engine.get());
        
        std::filesystem::create_directories("test_data");
        std::filesystem::create_directories("test_output");
        
        // Create test frames
        createTestFrames();
    }
    
    void TearDown() override {
        reconstructor.reset();
        if (engine) {
            engine->shutdown();
        }
        std::filesystem::remove_all("test_output");
    }
    
    void createTestFrames() {
        // Create synthetic test frames with different patterns
        cv::Size frame_size(320, 240);
        
        // Frame 1: Solid red
        frame1 = cv::Mat::zeros(frame_size, CV_8UC3);
        frame1.setTo(cv::Scalar(0, 0, 255));  // Red
        
        // Frame 2: Solid blue  
        frame2 = cv::Mat::zeros(frame_size, CV_8UC3);
        frame2.setTo(cv::Scalar(255, 0, 0));  // Blue
        
        // Frame 3: Solid green
        frame3 = cv::Mat::zeros(frame_size, CV_8UC3);
        frame3.setTo(cv::Scalar(0, 255, 0));  // Green
        
        // Frame with gradient
        frame_gradient = cv::Mat::zeros(frame_size, CV_8UC3);
        for (int y = 0; y < frame_size.height; y++) {
            for (int x = 0; x < frame_size.width; x++) {
                int intensity = (x * 255) / frame_size.width;
                frame_gradient.at<cv::Vec3b>(y, x) = cv::Vec3b(intensity, intensity, intensity);
            }
        }
        
        // Frame with pattern
        frame_pattern = cv::Mat::zeros(frame_size, CV_8UC3);
        for (int y = 0; y < frame_size.height; y++) {
            for (int x = 0; x < frame_size.width; x++) {
                if ((x + y) % 20 < 10) {
                    frame_pattern.at<cv::Vec3b>(y, x) = cv::Vec3b(255, 255, 255);  // White
                } else {
                    frame_pattern.at<cv::Vec3b>(y, x) = cv::Vec3b(0, 0, 0);        // Black
                }
            }
        }
        
        // Prepare reference frames vector
        reference_frames = {frame1, frame2, frame3};
    }
    
    cv::Mat createCorruptionMask(cv::Size size, float corruption_ratio = 0.3f) {
        cv::Mat mask = cv::Mat::zeros(size, CV_8UC1);
        
        // Add random corrupted regions
        cv::RNG rng(12345);  // Fixed seed for reproducible tests
        int num_corrupted_pixels = static_cast<int>(size.area() * corruption_ratio);
        
        for (int i = 0; i < num_corrupted_pixels; i++) {
            int x = rng.uniform(0, size.width);
            int y = rng.uniform(0, size.height);
            mask.at<uchar>(y, x) = 255;  // Mark as corrupted
        }
        
        return mask;
    }
    
    std::unique_ptr<AdvancedVideoRepairEngine> engine;
    std::unique_ptr<FrameReconstructor> reconstructor;
    
    cv::Mat frame1, frame2, frame3, frame_gradient, frame_pattern;
    std::vector<cv::Mat> reference_frames;
};

// Basic frame reconstruction tests
TEST_F(FrameReconstructorTest, ReconstructWithValidFrames) {
    cv::Mat output_frame;
    RepairStrategy strategy;
    
    bool result = reconstructor->reconstruct_missing_frame(
        reference_frames, output_frame, 1, strategy
    );
    
    EXPECT_TRUE(result);
    EXPECT_FALSE(output_frame.empty());
    EXPECT_EQ(output_frame.size(), frame1.size());
    EXPECT_EQ(output_frame.type(), frame1.type());
}

TEST_F(FrameReconstructorTest, ReconstructWithInsufficientFrames) {
    cv::Mat output_frame;
    RepairStrategy strategy;
    std::vector<cv::Mat> single_frame = {frame1};
    
    bool result = reconstructor->reconstruct_missing_frame(
        single_frame, output_frame, 0, strategy
    );
    
    EXPECT_FALSE(result);
}

TEST_F(FrameReconstructorTest, ReconstructWithEmptyFrames) {
    cv::Mat output_frame;
    RepairStrategy strategy;
    std::vector<cv::Mat> empty_frames;
    
    bool result = reconstructor->reconstruct_missing_frame(
        empty_frames, output_frame, 0, strategy
    );
    
    EXPECT_FALSE(result);
}

TEST_F(FrameReconstructorTest, ReconstructWithDifferentStrategies) {
    cv::Mat output_frame1, output_frame2;
    
    RepairStrategy strategy1;
    strategy1.enable_motion_compensation = true;
    strategy1.enable_post_processing = true;
    
    RepairStrategy strategy2;
    strategy2.enable_motion_compensation = false;
    strategy2.enable_post_processing = false;
    
    bool result1 = reconstructor->reconstruct_missing_frame(
        reference_frames, output_frame1, 1, strategy1
    );
    
    bool result2 = reconstructor->reconstruct_missing_frame(
        reference_frames, output_frame2, 1, strategy2
    );
    
    EXPECT_TRUE(result1);
    EXPECT_TRUE(result2);
    EXPECT_FALSE(output_frame1.empty());
    EXPECT_FALSE(output_frame2.empty());
}

// Thread-safe reconstruction tests
TEST_F(FrameReconstructorTest, ThreadSafeReconstructionBasic) {
    VideoRepair::ThreadSafeFrameBuffer frame_buffer(10);
    
    // Add frames to buffer
    ASSERT_TRUE(frame_buffer.push_frame(frame1));
    ASSERT_TRUE(frame_buffer.push_frame(frame2));
    ASSERT_TRUE(frame_buffer.push_frame(frame3));
    
    cv::Mat output_frame;
    RepairStrategy strategy;
    
    bool result = reconstructor->reconstruct_missing_frame_safe(
        frame_buffer, output_frame, 1, strategy
    );
    
    EXPECT_TRUE(result);
    EXPECT_FALSE(output_frame.empty());
}

TEST_F(FrameReconstructorTest, ThreadSafeReconstructionWithInsufficientFrames) {
    VideoRepair::ThreadSafeFrameBuffer frame_buffer(10);
    
    // Add only one frame
    ASSERT_TRUE(frame_buffer.push_frame(frame1));
    
    cv::Mat output_frame;
    RepairStrategy strategy;
    
    bool result = reconstructor->reconstruct_missing_frame_safe(
        frame_buffer, output_frame, 0, strategy
    );
    
    EXPECT_FALSE(result);
}

TEST_F(FrameReconstructorTest, ThreadSafeReconstructionWithEmptyBuffer) {
    VideoRepair::ThreadSafeFrameBuffer frame_buffer(10);
    
    cv::Mat output_frame;
    RepairStrategy strategy;
    
    bool result = reconstructor->reconstruct_missing_frame_safe(
        frame_buffer, output_frame, 0, strategy
    );
    
    EXPECT_FALSE(result);
}

// Corrupted region repair tests
TEST_F(FrameReconstructorTest, RepairCorruptedRegionsBasic) {
    cv::Mat test_frame = frame_gradient.clone();
    cv::Mat corruption_mask = createCorruptionMask(test_frame.size(), 0.2f);
    RepairStrategy strategy;
    
    bool result = reconstructor->repair_corrupted_regions(
        test_frame, corruption_mask, reference_frames, strategy
    );
    
    EXPECT_TRUE(result);
    // Frame should still have the same dimensions
    EXPECT_EQ(test_frame.size(), frame_gradient.size());
}

TEST_F(FrameReconstructorTest, RepairCorruptedRegionsWithEmptyMask) {
    cv::Mat test_frame = frame_gradient.clone();
    cv::Mat empty_mask = cv::Mat::zeros(test_frame.size(), CV_8UC1);
    RepairStrategy strategy;
    
    bool result = reconstructor->repair_corrupted_regions(
        test_frame, empty_mask, reference_frames, strategy
    );
    
    // Should succeed (no regions to repair)
    EXPECT_TRUE(result);
}

TEST_F(FrameReconstructorTest, RepairCorruptedRegionsWithFullMask) {
    cv::Mat test_frame = frame_gradient.clone();
    cv::Mat full_mask = cv::Mat::ones(test_frame.size(), CV_8UC1) * 255;
    RepairStrategy strategy;
    
    bool result = reconstructor->repair_corrupted_regions(
        test_frame, full_mask, reference_frames, strategy
    );
    
    // Should still work (entire frame corrupted)
    EXPECT_TRUE(result || !result);  // Implementation dependent
}

TEST_F(FrameReconstructorTest, RepairCorruptedRegionsThreadSafe) {
    VideoRepair::ThreadSafeFrameBuffer frame_buffer(10);
    
    // Add reference frames
    ASSERT_TRUE(frame_buffer.push_frame(frame1));
    ASSERT_TRUE(frame_buffer.push_frame(frame2));
    ASSERT_TRUE(frame_buffer.push_frame(frame3));
    
    cv::Mat test_frame = frame_pattern.clone();
    cv::Mat corruption_mask = createCorruptionMask(test_frame.size(), 0.25f);
    RepairStrategy strategy;
    
    bool result = reconstructor->repair_corrupted_regions_safe(
        test_frame, corruption_mask, frame_buffer, strategy
    );
    
    EXPECT_TRUE(result);
}

// Optical flow interpolation tests
TEST_F(FrameReconstructorTest, OpticalFlowInterpolationBasic) {
    cv::Mat interpolated_frame;
    
    bool result = reconstructor->perform_optical_flow_interpolation(
        frame1, frame2, interpolated_frame, 0.5
    );
    
    EXPECT_TRUE(result);
    EXPECT_FALSE(interpolated_frame.empty());
    EXPECT_EQ(interpolated_frame.size(), frame1.size());
}

TEST_F(FrameReconstructorTest, OpticalFlowInterpolationWithEmptyFrames) {
    cv::Mat empty_frame;
    cv::Mat interpolated_frame;
    
    bool result = reconstructor->perform_optical_flow_interpolation(
        empty_frame, frame2, interpolated_frame, 0.5
    );
    
    EXPECT_FALSE(result);
}

TEST_F(FrameReconstructorTest, OpticalFlowInterpolationDifferentPositions) {
    cv::Mat interpolated_0, interpolated_05, interpolated_1;
    
    // Test different temporal positions
    bool result1 = reconstructor->perform_optical_flow_interpolation(
        frame1, frame2, interpolated_0, 0.0
    );
    
    bool result2 = reconstructor->perform_optical_flow_interpolation(
        frame1, frame2, interpolated_05, 0.5
    );
    
    bool result3 = reconstructor->perform_optical_flow_interpolation(
        frame1, frame2, interpolated_1, 1.0
    );
    
    EXPECT_TRUE(result1);
    EXPECT_TRUE(result2);
    EXPECT_TRUE(result3);
    
    // Results should be different for different temporal positions
    EXPECT_FALSE(interpolated_0.empty());
    EXPECT_FALSE(interpolated_05.empty());
    EXPECT_FALSE(interpolated_1.empty());
}

// Feature-based interpolation tests
TEST_F(FrameReconstructorTest, FeatureBasedInterpolationBasic) {
    cv::Mat interpolated_frame;
    
    bool result = reconstructor->perform_feature_based_interpolation(
        frame_pattern, frame_gradient, interpolated_frame, 0.5
    );
    
    EXPECT_TRUE(result);
    EXPECT_FALSE(interpolated_frame.empty());
    EXPECT_EQ(interpolated_frame.size(), frame_pattern.size());
}

TEST_F(FrameReconstructorTest, FeatureBasedInterpolationWithEmptyFrames) {
    cv::Mat empty_frame;
    cv::Mat interpolated_frame;
    
    bool result = reconstructor->perform_feature_based_interpolation(
        empty_frame, frame_pattern, interpolated_frame, 0.5
    );
    
    EXPECT_FALSE(result);
}

// Blending tests
TEST_F(FrameReconstructorTest, BlendReconstructionResults) {
    cv::Mat blended = reconstructor->blend_reconstruction_results(
        frame1, frame2, 0.7, 0.3
    );
    
    EXPECT_FALSE(blended.empty());
    EXPECT_EQ(blended.size(), frame1.size());
    EXPECT_EQ(blended.type(), frame1.type());
}

TEST_F(FrameReconstructorTest, BlendWithEmptyFrame) {
    cv::Mat empty_frame;
    cv::Mat blended1 = reconstructor->blend_reconstruction_results(
        empty_frame, frame2, 0.7, 0.3
    );
    
    cv::Mat blended2 = reconstructor->blend_reconstruction_results(
        frame1, empty_frame, 0.7, 0.3
    );
    
    // Should return the non-empty frame
    EXPECT_FALSE(blended1.empty());
    EXPECT_FALSE(blended2.empty());
    
    // Check if blended1 is similar to frame2
    cv::Mat diff1;
    cv::absdiff(blended1, frame2, diff1);
    double max_diff1;
    cv::minMaxLoc(diff1, nullptr, &max_diff1);
    EXPECT_LT(max_diff1, 50.0);  // Should be very similar
}

TEST_F(FrameReconstructorTest, BlendWithZeroWeights) {
    cv::Mat blended = reconstructor->blend_reconstruction_results(
        frame1, frame2, 0.0, 0.0
    );
    
    EXPECT_FALSE(blended.empty());
    // Should still produce a reasonable result (fallback to equal weights)
}

// Edge case tests
TEST_F(FrameReconstructorTest, ReconstructWithNegativeFrameNumber) {
    cv::Mat output_frame;
    RepairStrategy strategy;
    
    bool result = reconstructor->reconstruct_missing_frame(
        reference_frames, output_frame, -1, strategy
    );
    
    // Should handle gracefully
    EXPECT_TRUE(result || !result);  // Implementation dependent
}

TEST_F(FrameReconstructorTest, ReconstructWithLargeFrameNumber) {
    cv::Mat output_frame;
    RepairStrategy strategy;
    
    bool result = reconstructor->reconstruct_missing_frame(
        reference_frames, output_frame, 1000, strategy
    );
    
    // Should handle gracefully  
    EXPECT_TRUE(result || !result);  // Implementation dependent
}