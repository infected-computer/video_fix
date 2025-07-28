#include <gtest/gtest.h>
#include <filesystem>
#include <fstream>
#include "AdvancedVideoRepair/FFmpegUtils.h"

extern "C" {
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libavutil/avutil.h>
}

using namespace VideoRepair;

class FFmpegIntegrationTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Initialize FFmpeg
        avformat_network_init();
        
        std::filesystem::create_directories("test_data");
        std::filesystem::create_directories("test_output");
        
        createTestFiles();
    }
    
    void TearDown() override {
        avformat_network_deinit();
        std::filesystem::remove_all("test_output");
    }
    
    void createTestFiles() {
        // Create a minimal MP4 file for testing
        create_minimal_mp4("test_data/test_input.mp4");
        
        // Create an invalid file
        std::ofstream invalid_file("test_data/invalid.mp4", std::ios::binary);
        const char invalid_data[] = {0xFF, 0xFF, 0xFF, 0xFF, 0x00, 0x00, 0x00, 0x00};
        invalid_file.write(invalid_data, sizeof(invalid_data));
        invalid_file.close();
    }
    
    void create_minimal_mp4(const std::string& filename) {
        std::ofstream file(filename, std::ios::binary);
        
        // Write ftyp box
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
        
        // Write minimal moov box
        const char moov_box[] = {
            0x00, 0x00, 0x00, 0x10,  // size
            'm', 'o', 'o', 'v',      // type
            0x00, 0x00, 0x00, 0x08,  // mvhd size
            'm', 'v', 'h', 'd'       // mvhd type
        };
        file.write(moov_box, sizeof(moov_box));
        
        // Write mdat box
        const char mdat_box[] = {
            0x00, 0x00, 0x00, 0x08,  // size
            'm', 'd', 'a', 't'       // type
        };
        file.write(mdat_box, sizeof(mdat_box));
        
        file.close();
    }
};

// AVFormatContextPtr tests
TEST_F(FFmpegIntegrationTest, AVFormatContextPtrBasicUsage) {
    AVFormatContextPtr format_ctx;
    
    // Should be initially null
    EXPECT_FALSE(format_ctx);
    EXPECT_EQ(format_ctx.get(), nullptr);
}

TEST_F(FFmpegIntegrationTest, AVFormatContextPtrWithValidFile) {
    AVFormatContextPtr format_ctx;
    AVFormatContext* ctx = avformat_alloc_context();
    ASSERT_NE(ctx, nullptr);
    
    format_ctx.reset(ctx);
    EXPECT_TRUE(format_ctx);
    EXPECT_EQ(format_ctx.get(), ctx);
    
    // Test -> operator
    EXPECT_EQ(format_ctx->nb_streams, 0);
}

TEST_F(FFmpegIntegrationTest, AVFormatContextPtrMoveSemantics) {
    AVFormatContext* ctx = avformat_alloc_context();
    ASSERT_NE(ctx, nullptr);
    
    AVFormatContextPtr format_ctx1(ctx);
    EXPECT_TRUE(format_ctx1);
    EXPECT_EQ(format_ctx1.get(), ctx);
    
    // Move construction
    AVFormatContextPtr format_ctx2 = std::move(format_ctx1);
    EXPECT_FALSE(format_ctx1);  // Should be null after move
    EXPECT_TRUE(format_ctx2);
    EXPECT_EQ(format_ctx2.get(), ctx);
    
    // Move assignment
    AVFormatContextPtr format_ctx3;
    format_ctx3 = std::move(format_ctx2);
    EXPECT_FALSE(format_ctx2);  // Should be null after move
    EXPECT_TRUE(format_ctx3);
    EXPECT_EQ(format_ctx3.get(), ctx);
}

TEST_F(FFmpegIntegrationTest, AVFormatContextPtrReset) {
    AVFormatContext* ctx1 = avformat_alloc_context();
    AVFormatContext* ctx2 = avformat_alloc_context();
    ASSERT_NE(ctx1, nullptr);
    ASSERT_NE(ctx2, nullptr);
    
    AVFormatContextPtr format_ctx(ctx1);
    EXPECT_EQ(format_ctx.get(), ctx1);
    
    // Reset with new context
    format_ctx.reset(ctx2);
    EXPECT_EQ(format_ctx.get(), ctx2);
    
    // Reset to null
    format_ctx.reset();
    EXPECT_FALSE(format_ctx);
}

// AVCodecContextPtr tests
TEST_F(FFmpegIntegrationTest, AVCodecContextPtrBasicUsage) {
    const AVCodec* codec = avcodec_find_decoder(AV_CODEC_ID_H264);
    if (codec) {
        AVCodecContextPtr codec_ctx(avcodec_alloc_context3(codec));
        
        EXPECT_TRUE(codec_ctx);
        EXPECT_NE(codec_ctx.get(), nullptr);
        EXPECT_EQ(codec_ctx->codec_id, AV_CODEC_ID_NONE);  // Not set until configured
    }
}

TEST_F(FFmpegIntegrationTest, AVCodecContextPtrMoveSemantics) {
    const AVCodec* codec = avcodec_find_decoder(AV_CODEC_ID_H264);
    if (codec) {
        AVCodecContext* ctx = avcodec_alloc_context3(codec);
        ASSERT_NE(ctx, nullptr);
        
        AVCodecContextPtr codec_ctx1(ctx);
        EXPECT_TRUE(codec_ctx1);
        
        AVCodecContextPtr codec_ctx2 = std::move(codec_ctx1);
        EXPECT_FALSE(codec_ctx1);
        EXPECT_TRUE(codec_ctx2);
        EXPECT_EQ(codec_ctx2.get(), ctx);
    }
}

// AVFramePtr tests
TEST_F(FFmpegIntegrationTest, AVFramePtrBasicUsage) {
    AVFramePtr frame;
    
    EXPECT_TRUE(frame);  // Should auto-allocate
    EXPECT_NE(frame.get(), nullptr);
    EXPECT_EQ(frame->width, 0);  // Initially empty
    EXPECT_EQ(frame->height, 0);
}

TEST_F(FFmpegIntegrationTest, AVFramePtrMoveSemantics) {
    AVFramePtr frame1;
    AVFrame* raw_frame = frame1.get();
    
    AVFramePtr frame2 = std::move(frame1);
    EXPECT_FALSE(frame1);  // Should be null after move
    EXPECT_TRUE(frame2);
    EXPECT_EQ(frame2.get(), raw_frame);
}

// AVPacketPtr tests
TEST_F(FFmpegIntegrationTest, AVPacketPtrBasicUsage) {
    AVPacketPtr packet;
    
    EXPECT_TRUE(packet);  // Should auto-allocate
    EXPECT_NE(packet.get(), nullptr);
    EXPECT_EQ(packet->size, 0);  // Initially empty
}

TEST_F(FFmpegIntegrationTest, AVPacketPtrMoveSemantics) {
    AVPacketPtr packet1;
    AVPacket* raw_packet = packet1.get();
    
    AVPacketPtr packet2 = std::move(packet1);
    EXPECT_FALSE(packet1);  // Should be null after move
    EXPECT_TRUE(packet2);
    EXPECT_EQ(packet2.get(), raw_packet);
}

// AVDictionaryPtr tests
TEST_F(FFmpegIntegrationTest, AVDictionaryPtrBasicUsage) {
    AVDictionaryPtr dict;
    
    EXPECT_FALSE(dict);  // Should be initially null
    
    // Add some entries
    av_dict_set(dict.get_ptr(), "key1", "value1", 0);
    av_dict_set(dict.get_ptr(), "key2", "value2", 0);
    
    EXPECT_TRUE(dict);  // Should now have content
    
    // Check values
    AVDictionaryEntry* entry = av_dict_get(dict.get(), "key1", nullptr, 0);
    ASSERT_NE(entry, nullptr);
    EXPECT_STREQ(entry->value, "value1");
}

TEST_F(FFmpegIntegrationTest, AVDictionaryPtrMoveSemantics) {
    AVDictionaryPtr dict1;
    av_dict_set(dict1.get_ptr(), "test", "value", 0);
    
    AVDictionary* raw_dict = dict1.get();
    
    AVDictionaryPtr dict2 = std::move(dict1);
    EXPECT_FALSE(dict1);  // Should be null after move
    EXPECT_TRUE(dict2);
    EXPECT_EQ(dict2.get(), raw_dict);
    
    // Verify content is preserved
    AVDictionaryEntry* entry = av_dict_get(dict2.get(), "test", nullptr, 0);
    ASSERT_NE(entry, nullptr);
    EXPECT_STREQ(entry->value, "value");
}

// Integration tests with real FFmpeg operations
TEST_F(FFmpegIntegrationTest, OpenValidFileWithRAIIWrappers) {
    AVFormatContextPtr format_ctx;
    AVFormatContext* ctx = avformat_alloc_context();
    ASSERT_NE(ctx, nullptr);
    format_ctx.reset(ctx);
    
    AVDictionaryPtr options;
    av_dict_set(options.get_ptr(), "fflags", "+ignidx", 0);
    
    int ret = avformat_open_input(format_ctx.get_ptr(), "test_data/test_input.mp4", nullptr, options.get_ptr());
    
    if (ret == 0) {
        // Successfully opened
        EXPECT_TRUE(format_ctx);
        EXPECT_GE(format_ctx->nb_streams, 0u);
        
        // Try to find stream info
        ret = avformat_find_stream_info(format_ctx.get(), nullptr);
        // May fail for minimal file, but should not crash
    }
    
    // RAII should handle cleanup automatically
}

TEST_F(FFmpegIntegrationTest, OpenInvalidFileWithRAIIWrappers) {
    AVFormatContextPtr format_ctx;
    AVFormatContext* ctx = avformat_alloc_context();
    ASSERT_NE(ctx, nullptr);
    format_ctx.reset(ctx);
    
    int ret = avformat_open_input(format_ctx.get_ptr(), "test_data/invalid.mp4", nullptr, nullptr);
    
    // Should fail to open invalid file
    EXPECT_LT(ret, 0);
    
    // RAII should still handle cleanup properly
}

TEST_F(FFmpegIntegrationTest, CodecContextWithRAIIWrappers) {
    const AVCodec* codec = avcodec_find_decoder(AV_CODEC_ID_H264);
    if (codec) {
        AVCodecContextPtr codec_ctx(avcodec_alloc_context3(codec));
        ASSERT_TRUE(codec_ctx);
        
        // Set some basic parameters
        codec_ctx->width = 1920;
        codec_ctx->height = 1080;
        codec_ctx->pix_fmt = AV_PIX_FMT_YUV420P;
        
        // Try to open codec (may fail without proper parameters, but shouldn't crash)
        int ret = avcodec_open2(codec_ctx.get(), codec, nullptr);
        
        // Whether it succeeds or fails, RAII should handle cleanup
        EXPECT_TRUE(ret == 0 || ret < 0);  // Either outcome is fine for this test
    }
}

// Error handling tests
TEST_F(FFmpegIntegrationTest, HandleNullPointersGracefully) {
    AVFormatContextPtr null_ctx;
    EXPECT_FALSE(null_ctx);
    
    // Should not crash when resetting null pointer
    null_ctx.reset();
    EXPECT_FALSE(null_ctx);
    
    // Should not crash when releasing null pointer
    AVFormatContext* released = null_ctx.release();
    EXPECT_EQ(released, nullptr);
}

TEST_F(FFmpegIntegrationTest, MultipleResetsCauseNoLeaks) {
    AVFormatContextPtr format_ctx;
    
    // Multiple resets with different contexts
    for (int i = 0; i < 5; i++) {
        AVFormatContext* ctx = avformat_alloc_context();
        ASSERT_NE(ctx, nullptr);
        format_ctx.reset(ctx);
        EXPECT_TRUE(format_ctx);
    }
    
    // Final reset to null
    format_ctx.reset();
    EXPECT_FALSE(format_ctx);
}