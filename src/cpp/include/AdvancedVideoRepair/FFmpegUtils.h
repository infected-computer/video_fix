#pragma once

#include <memory>
extern "C" {
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libswscale/swscale.h>
#include <libavutil/avutil.h>
}

namespace VideoRepair {

/**
 * @brief RAII wrapper for AVFormatContext
 * 
 * Automatically manages the lifecycle of AVFormatContext pointers,
 * ensuring proper cleanup and preventing memory leaks.
 */
class AVFormatContextPtr {
private:
    AVFormatContext* ctx_;

public:
    AVFormatContextPtr() : ctx_(nullptr) {}
    
    explicit AVFormatContextPtr(AVFormatContext* ctx) : ctx_(ctx) {}
    
    // Destructor ensures proper cleanup
    ~AVFormatContextPtr() {
        if (ctx_) {
            avformat_close_input(&ctx_);
        }
    }
    
    // Move constructor
    AVFormatContextPtr(AVFormatContextPtr&& other) noexcept : ctx_(other.ctx_) {
        other.ctx_ = nullptr;
    }
    
    // Move assignment
    AVFormatContextPtr& operator=(AVFormatContextPtr&& other) noexcept {
        if (this != &other) {
            if (ctx_) {
                avformat_close_input(&ctx_);
            }
            ctx_ = other.ctx_;
            other.ctx_ = nullptr;
        }
        return *this;
    }
    
    // Delete copy constructor and assignment
    AVFormatContextPtr(const AVFormatContextPtr&) = delete;
    AVFormatContextPtr& operator=(const AVFormatContextPtr&) = delete;
    
    // Access operators
    AVFormatContext* get() const { return ctx_; }
    AVFormatContext* operator->() const { return ctx_; }
    AVFormatContext& operator*() const { return *ctx_; }
    
    // Conversion to bool for null checks
    explicit operator bool() const { return ctx_ != nullptr; }
    
    // Release ownership
    AVFormatContext* release() {
        AVFormatContext* temp = ctx_;
        ctx_ = nullptr;
        return temp;
    }
    
    // Reset with new context
    void reset(AVFormatContext* new_ctx = nullptr) {
        if (ctx_) {
            avformat_close_input(&ctx_);
        }
        ctx_ = new_ctx;
    }
    
    // Get pointer to pointer (for functions that modify the pointer)
    AVFormatContext** get_ptr() { return &ctx_; }
};

/**
 * @brief RAII wrapper for AVCodecContext
 */
class AVCodecContextPtr {
private:
    AVCodecContext* ctx_;

public:
    AVCodecContextPtr() : ctx_(nullptr) {}
    
    explicit AVCodecContextPtr(AVCodecContext* ctx) : ctx_(ctx) {}
    
    ~AVCodecContextPtr() {
        if (ctx_) {
            avcodec_free_context(&ctx_);
        }
    }
    
    // Move semantics
    AVCodecContextPtr(AVCodecContextPtr&& other) noexcept : ctx_(other.ctx_) {
        other.ctx_ = nullptr;
    }
    
    AVCodecContextPtr& operator=(AVCodecContextPtr&& other) noexcept {
        if (this != &other) {
            if (ctx_) {
                avcodec_free_context(&ctx_);
            }
            ctx_ = other.ctx_;
            other.ctx_ = nullptr;
        }
        return *this;
    }
    
    // Delete copy operations
    AVCodecContextPtr(const AVCodecContextPtr&) = delete;
    AVCodecContextPtr& operator=(const AVCodecContextPtr&) = delete;
    
    // Access operations
    AVCodecContext* get() const { return ctx_; }
    AVCodecContext* operator->() const { return ctx_; }
    AVCodecContext& operator*() const { return *ctx_; }
    explicit operator bool() const { return ctx_ != nullptr; }
    
    AVCodecContext* release() {
        AVCodecContext* temp = ctx_;
        ctx_ = nullptr;
        return temp;
    }
    
    void reset(AVCodecContext* new_ctx = nullptr) {
        if (ctx_) {
            avcodec_free_context(&ctx_);
        }
        ctx_ = new_ctx;
    }
    
    AVCodecContext** get_ptr() { return &ctx_; }
};

/**
 * @brief RAII wrapper for AVFrame
 */
class AVFramePtr {
private:
    AVFrame* frame_;

public:
    AVFramePtr() : frame_(av_frame_alloc()) {}
    
    explicit AVFramePtr(AVFrame* frame) : frame_(frame) {}
    
    ~AVFramePtr() {
        if (frame_) {
            av_frame_free(&frame_);
        }
    }
    
    // Move semantics
    AVFramePtr(AVFramePtr&& other) noexcept : frame_(other.frame_) {
        other.frame_ = nullptr;
    }
    
    AVFramePtr& operator=(AVFramePtr&& other) noexcept {
        if (this != &other) {
            if (frame_) {
                av_frame_free(&frame_);
            }
            frame_ = other.frame_;
            other.frame_ = nullptr;
        }
        return *this;
    }
    
    // Delete copy operations
    AVFramePtr(const AVFramePtr&) = delete;
    AVFramePtr& operator=(const AVFramePtr&) = delete;
    
    // Access operations
    AVFrame* get() const { return frame_; }
    AVFrame* operator->() const { return frame_; }
    AVFrame& operator*() const { return *frame_; }
    explicit operator bool() const { return frame_ != nullptr; }
    
    AVFrame* release() {
        AVFrame* temp = frame_;
        frame_ = nullptr;
        return temp;
    }
    
    void reset(AVFrame* new_frame = nullptr) {
        if (frame_) {
            av_frame_free(&frame_);
        }
        frame_ = new_frame;
    }
};

/**
 * @brief RAII wrapper for AVPacket
 */
class AVPacketPtr {
private:
    AVPacket* packet_;

public:
    AVPacketPtr() : packet_(av_packet_alloc()) {}
    
    explicit AVPacketPtr(AVPacket* packet) : packet_(packet) {}
    
    ~AVPacketPtr() {
        if (packet_) {
            av_packet_free(&packet_);
        }
    }
    
    // Move semantics
    AVPacketPtr(AVPacketPtr&& other) noexcept : packet_(other.packet_) {
        other.packet_ = nullptr;
    }
    
    AVPacketPtr& operator=(AVPacketPtr&& other) noexcept {
        if (this != &other) {
            if (packet_) {
                av_packet_free(&packet_);
            }
            packet_ = other.packet_;
            other.packet_ = nullptr;
        }
        return *this;
    }
    
    // Delete copy operations
    AVPacketPtr(const AVPacketPtr&) = delete;
    AVPacketPtr& operator=(const AVPacketPtr&) = delete;
    
    // Access operations
    AVPacket* get() const { return packet_; }
    AVPacket* operator->() const { return packet_; }
    AVPacket& operator*() const { return *packet_; }
    explicit operator bool() const { return packet_ != nullptr; }
    
    AVPacket* release() {
        AVPacket* temp = packet_;
        packet_ = nullptr;
        return temp;
    }
    
    void reset(AVPacket* new_packet = nullptr) {
        if (packet_) {
            av_packet_free(&packet_);
        }
        packet_ = new_packet;
    }
};

/**
 * @brief RAII wrapper for SwsContext
 */
class SwsContextPtr {
private:
    SwsContext* ctx_;

public:
    SwsContextPtr() : ctx_(nullptr) {}
    
    explicit SwsContextPtr(SwsContext* ctx) : ctx_(ctx) {}
    
    ~SwsContextPtr() {
        if (ctx_) {
            sws_freeContext(ctx_);
        }
    }
    
    // Move semantics
    SwsContextPtr(SwsContextPtr&& other) noexcept : ctx_(other.ctx_) {
        other.ctx_ = nullptr;
    }
    
    SwsContextPtr& operator=(SwsContextPtr&& other) noexcept {
        if (this != &other) {
            if (ctx_) {
                sws_freeContext(ctx_);
            }
            ctx_ = other.ctx_;
            other.ctx_ = nullptr;
        }
        return *this;
    }
    
    // Delete copy operations
    SwsContextPtr(const SwsContextPtr&) = delete;
    SwsContextPtr& operator=(const SwsContextPtr&) = delete;
    
    // Access operations
    SwsContext* get() const { return ctx_; }
    explicit operator bool() const { return ctx_ != nullptr; }
    
    SwsContext* release() {
        SwsContext* temp = ctx_;
        ctx_ = nullptr;
        return temp;
    }
    
    void reset(SwsContext* new_ctx = nullptr) {
        if (ctx_) {
            sws_freeContext(ctx_);
        }
        ctx_ = new_ctx;
    }
};

/**
 * @brief RAII wrapper for AVDictionary
 */
class AVDictionaryPtr {
private:
    AVDictionary* dict_;

public:
    AVDictionaryPtr() : dict_(nullptr) {}
    
    explicit AVDictionaryPtr(AVDictionary* dict) : dict_(dict) {}
    
    ~AVDictionaryPtr() {
        if (dict_) {
            av_dict_free(&dict_);
        }
    }
    
    // Move semantics
    AVDictionaryPtr(AVDictionaryPtr&& other) noexcept : dict_(other.dict_) {
        other.dict_ = nullptr;
    }
    
    AVDictionaryPtr& operator=(AVDictionaryPtr&& other) noexcept {
        if (this != &other) {
            if (dict_) {
                av_dict_free(&dict_);
            }
            dict_ = other.dict_;
            other.dict_ = nullptr;
        }
        return *this;
    }
    
    // Delete copy operations
    AVDictionaryPtr(const AVDictionaryPtr&) = delete;
    AVDictionaryPtr& operator=(const AVDictionaryPtr&) = delete;
    
    // Access operations
    AVDictionary* get() const { return dict_; }
    explicit operator bool() const { return dict_ != nullptr; }
    
    AVDictionary* release() {
        AVDictionary* temp = dict_;
        dict_ = nullptr;
        return temp;
    }
    
    void reset(AVDictionary* new_dict = nullptr) {
        if (dict_) {
            av_dict_free(&dict_);
        }
        dict_ = new_dict;
    }
    
    AVDictionary** get_ptr() { return &dict_; }
};

} // namespace VideoRepair