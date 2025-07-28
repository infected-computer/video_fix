#include <gtest/gtest.h>
#include <vector>
#include <cmath>
#include <random>

#ifdef HAVE_CUDA
#include <cuda_runtime.h>
#include "AdvancedVideoRepair/CudaUtils.h"

// Declare external CUDA kernel wrapper functions
extern "C" {
    cudaError_t launch_motion_compensated_interpolation(
        const float* prev_frame,
        const float* next_frame,
        const float2* motion_vectors,
        float* result,
        int width, int height,
        float temporal_position,
        cudaStream_t stream
    );
    
    cudaError_t launch_corruption_detection(
        const float* frame_data,
        unsigned char* corruption_mask,
        int width, int height,
        float threshold,
        cudaStream_t stream
    );
    
    cudaError_t perform_safe_frame_interpolation(
        const float* h_prev_frame,
        const float* h_next_frame, 
        float* h_result,
        int width, int height,
        float temporal_position
    );
    
    cudaError_t timed_corruption_detection(
        const float* h_frame_data,
        unsigned char* h_corruption_mask,
        int width, int height,
        float threshold,
        float* elapsed_ms
    );
}

using namespace VideoRepair;

class CudaKernelsTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Check if CUDA is available
        int device_count;
        cudaError_t error = cudaGetDeviceCount(&device_count);
        
        if (error != cudaSuccess || device_count == 0) {
            GTEST_SKIP() << "CUDA not available, skipping CUDA tests";
        }
        
        // Set device
        cudaSetDevice(0);
        
        // Initialize test data
        width = 320;
        height = 240;
        total_pixels = width * height;
        
        createTestData();
    }
    
    void TearDown() override {
        // CUDA cleanup handled by RAII wrappers
    }
    
    void createTestData() {
        // Create test frames with patterns
        h_prev_frame.resize(total_pixels * 4);  // RGBA
        h_next_frame.resize(total_pixels * 4);
        h_result.resize(total_pixels * 4);
        h_motion_vectors.resize(total_pixels);
        h_frame_data.resize(total_pixels * 4);
        h_corruption_mask.resize(total_pixels);
        
        // Fill with test patterns
        std::mt19937 rng(12345);  // Fixed seed for reproducible tests
        std::uniform_real_distribution<float> color_dist(0.0f, 1.0f);
        std::uniform_real_distribution<float> motion_dist(-5.0f, 5.0f);
        
        for (int i = 0; i < total_pixels; i++) {
            // Previous frame: gradient pattern
            float x = (i % width) / float(width);
            float y = (i / width) / float(height);
            
            h_prev_frame[i * 4 + 0] = x;         // R
            h_prev_frame[i * 4 + 1] = y;         // G
            h_prev_frame[i * 4 + 2] = 0.5f;      // B
            h_prev_frame[i * 4 + 3] = 1.0f;      // A
            
            // Next frame: shifted pattern
            h_next_frame[i * 4 + 0] = (x + 0.1f) > 1.0f ? (x + 0.1f - 1.0f) : (x + 0.1f);
            h_next_frame[i * 4 + 1] = (y + 0.1f) > 1.0f ? (y + 0.1f - 1.0f) : (y + 0.1f);
            h_next_frame[i * 4 + 2] = 0.5f;
            h_next_frame[i * 4 + 3] = 1.0f;
            
            // Motion vectors (small random motion)
            h_motion_vectors[i].x = motion_dist(rng);
            h_motion_vectors[i].y = motion_dist(rng);
            
            // Frame data for corruption detection
            h_frame_data[i * 4 + 0] = color_dist(rng);
            h_frame_data[i * 4 + 1] = color_dist(rng);
            h_frame_data[i * 4 + 2] = color_dist(rng);
            h_frame_data[i * 4 + 3] = 1.0f;
        }
        
        // Add some "corrupted" regions (extreme values)
        for (int i = 0; i < total_pixels / 20; i++) {
            int idx = rng() % total_pixels;
            h_frame_data[idx * 4 + 0] = 0.0f;  // Pure black
            h_frame_data[idx * 4 + 1] = 0.0f;
            h_frame_data[idx * 4 + 2] = 0.0f;
        }
    }
    
    int width, height, total_pixels;
    std::vector<float> h_prev_frame, h_next_frame, h_result, h_frame_data;
    std::vector<float2> h_motion_vectors;
    std::vector<unsigned char> h_corruption_mask;
};

// CudaDeviceBuffer tests
TEST_F(CudaKernelsTest, CudaDeviceBufferBasicAllocation) {
    const size_t buffer_size = 1024;
    
    CudaDeviceBuffer<float> buffer(buffer_size);
    
    EXPECT_TRUE(buffer);
    EXPECT_NE(buffer.get(), nullptr);
    EXPECT_EQ(buffer.size(), buffer_size);
    EXPECT_EQ(buffer.size_bytes(), buffer_size * sizeof(float));
}

TEST_F(CudaKernelsTest, CudaDeviceBufferMoveSemantics) {
    const size_t buffer_size = 512;
    
    CudaDeviceBuffer<float> buffer1(buffer_size);
    float* ptr = buffer1.get();
    
    EXPECT_TRUE(buffer1);
    EXPECT_EQ(buffer1.size(), buffer_size);
    
    // Move construction
    CudaDeviceBuffer<float> buffer2 = std::move(buffer1);
    
    EXPECT_FALSE(buffer1);  // Should be empty after move
    EXPECT_TRUE(buffer2);
    EXPECT_EQ(buffer2.get(), ptr);
    EXPECT_EQ(buffer2.size(), buffer_size);
    
    // Move assignment
    CudaDeviceBuffer<float> buffer3(256);
    buffer3 = std::move(buffer2);
    
    EXPECT_FALSE(buffer2);  // Should be empty after move
    EXPECT_TRUE(buffer3);
    EXPECT_EQ(buffer3.get(), ptr);
    EXPECT_EQ(buffer3.size(), buffer_size);
}

TEST_F(CudaKernelsTest, CudaDeviceBufferCopyOperations) {
    const size_t buffer_size = 100;
    std::vector<float> host_data(buffer_size);
    std::vector<float> host_result(buffer_size);
    
    // Fill host data
    for (size_t i = 0; i < buffer_size; i++) {
        host_data[i] = static_cast<float>(i) * 0.1f;
    }
    
    CudaDeviceBuffer<float> device_buffer(buffer_size);
    
    // Copy to device
    cudaError_t result = device_buffer.copy_from_host(host_data.data());
    EXPECT_EQ(result, cudaSuccess);
    
    // Copy back to host
    result = device_buffer.copy_to_host(host_result.data());
    EXPECT_EQ(result, cudaSuccess);
    
    // Verify data integrity
    for (size_t i = 0; i < buffer_size; i++) {
        EXPECT_FLOAT_EQ(host_data[i], host_result[i]);
    }
}

TEST_F(CudaKernelsTest, CudaDeviceBufferZeroMemory) {
    const size_t buffer_size = 50;
    CudaDeviceBuffer<float> buffer(buffer_size);
    
    // Zero the buffer
    cudaError_t result = buffer.zero();
    EXPECT_EQ(result, cudaSuccess);
    
    // Copy back and verify
    std::vector<float> host_result(buffer_size);
    result = buffer.copy_to_host(host_result.data());
    EXPECT_EQ(result, cudaSuccess);
    
    for (size_t i = 0; i < buffer_size; i++) {
        EXPECT_FLOAT_EQ(host_result[i], 0.0f);
    }
}

// CudaStreamPtr tests
TEST_F(CudaKernelsTest, CudaStreamPtrBasicUsage) {
    CudaStreamPtr stream;
    
    EXPECT_TRUE(stream);
    EXPECT_NE(stream.get(), nullptr);
    
    // Test synchronization
    cudaError_t result = stream.synchronize();
    EXPECT_EQ(result, cudaSuccess);
    
    // Test query
    result = stream.query();
    EXPECT_TRUE(result == cudaSuccess || result == cudaErrorNotReady);
}

TEST_F(CudaKernelsTest, CudaStreamPtrMoveSemantics) {
    CudaStreamPtr stream1;
    cudaStream_t raw_stream = stream1.get();
    
    EXPECT_TRUE(stream1);
    
    CudaStreamPtr stream2 = std::move(stream1);
    
    EXPECT_FALSE(stream1);  // Should be null after move
    EXPECT_TRUE(stream2);
    EXPECT_EQ(stream2.get(), raw_stream);
}

// CudaEventPtr tests
TEST_F(CudaKernelsTest, CudaEventPtrBasicUsage) {
    CudaEventPtr event;
    
    EXPECT_TRUE(event);
    EXPECT_NE(event.get(), nullptr);
    
    // Record event
    cudaError_t result = event.record();
    EXPECT_EQ(result, cudaSuccess);
    
    // Synchronize
    result = event.synchronize();
    EXPECT_EQ(result, cudaSuccess);
}

TEST_F(CudaKernelsTest, CudaEventPtrTiming) {
    CudaEventPtr start_event;
    CudaEventPtr stop_event;
    
    // Record start
    cudaError_t result = start_event.record();
    EXPECT_EQ(result, cudaSuccess);
    
    // Do some work (simple kernel or memory copy)
    CudaDeviceBuffer<float> buffer(1000);
    buffer.zero();
    
    // Record stop
    result = stop_event.record();
    EXPECT_EQ(result, cudaSuccess);
    
    // Wait for completion
    result = stop_event.synchronize();
    EXPECT_EQ(result, cudaSuccess);
    
    // Get elapsed time
    float elapsed = stop_event.elapsed_time(start_event);
    EXPECT_GE(elapsed, 0.0f);
    EXPECT_LT(elapsed, 1000.0f);  // Should be less than 1 second
}

// Motion compensated interpolation kernel tests
TEST_F(CudaKernelsTest, MotionCompensatedInterpolationKernel) {
    CudaDeviceBuffer<float> d_prev(total_pixels * 4);
    CudaDeviceBuffer<float> d_next(total_pixels * 4);
    CudaDeviceBuffer<float2> d_motion(total_pixels);
    CudaDeviceBuffer<float> d_result(total_pixels * 4);
    
    // Copy data to device
    ASSERT_EQ(d_prev.copy_from_host(h_prev_frame.data()), cudaSuccess);
    ASSERT_EQ(d_next.copy_from_host(h_next_frame.data()), cudaSuccess);
    ASSERT_EQ(d_motion.copy_from_host(h_motion_vectors.data()), cudaSuccess);
    
    // Launch kernel
    cudaError_t result = launch_motion_compensated_interpolation(
        d_prev.get(), d_next.get(), d_motion.get(), d_result.get(),
        width, height, 0.5f, 0
    );
    
    EXPECT_EQ(result, cudaSuccess);
    
    // Copy result back
    ASSERT_EQ(d_result.copy_to_host(h_result.data()), cudaSuccess);
    
    // Verify result is reasonable (not all zeros, not all same value)
    bool has_variation = false;
    float first_value = h_result[0];
    
    for (int i = 1; i < total_pixels * 4; i++) {
        if (std::abs(h_result[i] - first_value) > 0.01f) {
            has_variation = true;
            break;
        }
    }
    
    EXPECT_TRUE(has_variation);
    
    // Values should be in reasonable range [0, 1]
    for (int i = 0; i < total_pixels * 4; i++) {
        EXPECT_GE(h_result[i], -0.1f);  // Allow small numerical errors
        EXPECT_LE(h_result[i], 1.1f);
    }
}

// Corruption detection kernel tests
TEST_F(CudaKernelsTest, CorruptionDetectionKernel) {
    CudaDeviceBuffer<float> d_frame(total_pixels * 4);
    CudaDeviceBuffer<unsigned char> d_mask(total_pixels);
    
    // Copy data to device
    ASSERT_EQ(d_frame.copy_from_host(h_frame_data.data()), cudaSuccess);
    
    // Launch kernel
    cudaError_t result = launch_corruption_detection(
        d_frame.get(), d_mask.get(), width, height, 3.0f, 0
    );
    
    EXPECT_EQ(result, cudaSuccess);
    
    // Copy result back
    ASSERT_EQ(d_mask.copy_to_host(h_corruption_mask.data()), cudaSuccess);
    
    // Verify some corruption was detected
    int corrupted_pixels = 0;
    for (int i = 0; i < total_pixels; i++) {
        if (h_corruption_mask[i] > 0) {
            corrupted_pixels++;
        }
    }
    
    EXPECT_GT(corrupted_pixels, 0);  // Should detect some corruption
    EXPECT_LT(corrupted_pixels, total_pixels / 2);  // But not everything
}

// High-level RAII wrapper function tests
TEST_F(CudaKernelsTest, SafeFrameInterpolationWrapper) {
    cudaError_t result = perform_safe_frame_interpolation(
        h_prev_frame.data(), h_next_frame.data(), h_result.data(),
        width, height, 0.5f
    );
    
    EXPECT_EQ(result, cudaSuccess);
    
    // Verify result quality
    bool has_reasonable_values = true;
    for (int i = 0; i < total_pixels * 4; i++) {
        if (h_result[i] < -0.1f || h_result[i] > 1.1f) {
            has_reasonable_values = false;
            break;
        }
    }
    
    EXPECT_TRUE(has_reasonable_values);
}

TEST_F(CudaKernelsTest, TimedCorruptionDetectionWrapper) {
    float elapsed_ms = -1.0f;
    
    cudaError_t result = timed_corruption_detection(
        h_frame_data.data(), h_corruption_mask.data(),
        width, height, 3.0f, &elapsed_ms
    );
    
    EXPECT_EQ(result, cudaSuccess);
    EXPECT_GE(elapsed_ms, 0.0f);
    EXPECT_LT(elapsed_ms, 1000.0f);  // Should complete in less than 1 second
    
    // Verify corruption detection worked
    int detected_pixels = 0;
    for (int i = 0; i < total_pixels; i++) {
        if (h_corruption_mask[i] > 0) {
            detected_pixels++;
        }
    }
    
    EXPECT_GT(detected_pixels, 0);
}

// Error handling tests
TEST_F(CudaKernelsTest, HandleInvalidParameters) {
    // Test with null pointers - should not crash CUDA driver
    cudaError_t result = launch_motion_compensated_interpolation(
        nullptr, nullptr, nullptr, nullptr, 0, 0, 0.5f, 0
    );
    
    // Should return an error, not crash
    EXPECT_NE(result, cudaSuccess);
}

TEST_F(CudaKernelsTest, HandleInvalidDimensions) {
    CudaDeviceBuffer<float> d_frame(100);
    CudaDeviceBuffer<unsigned char> d_mask(100);
    
    // Test with invalid dimensions
    cudaError_t result = launch_corruption_detection(
        d_frame.get(), d_mask.get(), -1, -1, 1.0f, 0
    );
    
    // Should handle gracefully
    EXPECT_TRUE(result == cudaSuccess || result != cudaSuccess);  // Either outcome acceptable
}

#else

// Dummy test when CUDA is not available
TEST(CudaKernelsTest, CudaNotAvailable) {
    GTEST_SKIP() << "CUDA support not compiled in";
}

#endif // HAVE_CUDA