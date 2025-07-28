#pragma once

#include <vector>
#include <shared_mutex>
#include <mutex>
#include <atomic>
#include <condition_variable>
#include <opencv2/opencv.hpp>

namespace VideoRepair {

/**
 * @brief Thread-safe frame buffer for concurrent video processing
 * 
 * This class provides thread-safe access to a collection of video frames,
 * supporting concurrent read operations while ensuring data integrity
 * during write operations.
 */
class ThreadSafeFrameBuffer {
private:
    mutable std::shared_mutex mutex_;
    std::vector<cv::Mat> frames_;
    std::atomic<size_t> frame_count_{0};
    std::atomic<bool> finalized_{false};
    
    // Statistics for monitoring
    std::atomic<size_t> read_operations_{0};
    std::atomic<size_t> write_operations_{0};
    
    // Memory management
    size_t max_frames_;
    size_t total_memory_bytes_{0};
    size_t max_memory_bytes_;

public:
    /**
     * @brief Constructor with optional capacity limits
     * 
     * @param max_frames Maximum number of frames to store (0 = unlimited)
     * @param max_memory_mb Maximum memory usage in megabytes (0 = unlimited)
     */
    explicit ThreadSafeFrameBuffer(size_t max_frames = 0, size_t max_memory_mb = 0)
        : max_frames_(max_frames)
        , max_memory_bytes_(max_memory_mb * 1024 * 1024) {
        if (max_frames_ > 0) {
            frames_.reserve(max_frames_);
        }
    }
    
    /**
     * @brief Add a frame to the buffer (thread-safe)
     * 
     * @param frame Frame to add (will be cloned if not moved)
     * @return true if frame was added successfully
     */
    bool push_frame(cv::Mat frame) {
        if (finalized_.load()) {
            return false; // Cannot add frames to finalized buffer
        }
        
        std::unique_lock lock(mutex_);
        
        // Check capacity limits
        if (max_frames_ > 0 && frames_.size() >= max_frames_) {
            return false; // Buffer full
        }
        
        size_t frame_bytes = frame.total() * frame.elemSize();
        if (max_memory_bytes_ > 0 && total_memory_bytes_ + frame_bytes > max_memory_bytes_) {
            return false; // Memory limit exceeded
        }
        
        // Add frame (move if possible, otherwise clone)
        frames_.push_back(frame.empty() ? frame : frame.clone());
        total_memory_bytes_ += frame_bytes;
        frame_count_.store(frames_.size());
        write_operations_.fetch_add(1);
        
        return true;
    }
    
    /**
     * @brief Get a frame by index (thread-safe, returns a copy)
     * 
     * @param idx Frame index
     * @return Cloned frame at index, empty Mat if index is invalid
     */
    cv::Mat get_frame(size_t idx) const {
        std::shared_lock lock(mutex_);
        
        if (idx >= frames_.size()) {
            return cv::Mat(); // Return empty Mat for invalid index
        }
        
        read_operations_.fetch_add(1);
        return frames_[idx].clone(); // Always return a clone for safety
    }
    
    /**
     * @brief Get a frame by index without cloning (thread-safe, const reference)
     * 
     * WARNING: The returned reference is only valid while the shared_lock is held.
     * Use this method carefully and only for read-only operations.
     * 
     * @param idx Frame index
     * @param lock Shared lock that will be locked during access
     * @return Const reference to frame, or empty Mat if invalid
     */
    const cv::Mat& get_frame_ref(size_t idx, std::shared_lock<std::shared_mutex>& lock) const {
        static const cv::Mat empty_mat;
        
        lock = std::shared_lock(mutex_);
        
        if (idx >= frames_.size()) {
            return empty_mat;
        }
        
        read_operations_.fetch_add(1);
        return frames_[idx];
    }
    
    /**
     * @brief Get multiple frames as copies (thread-safe)
     * 
     * @param start_idx Starting index
     * @param count Number of frames to retrieve
     * @return Vector of cloned frames
     */
    std::vector<cv::Mat> get_frames(size_t start_idx, size_t count) const {
        std::shared_lock lock(mutex_);
        
        std::vector<cv::Mat> result;
        size_t end_idx = std::min(start_idx + count, frames_.size());
        
        if (start_idx >= frames_.size()) {
            return result; // Return empty vector
        }
        
        result.reserve(end_idx - start_idx);
        for (size_t i = start_idx; i < end_idx; ++i) {
            result.push_back(frames_[i].clone());
        }
        
        read_operations_.fetch_add(1);
        return result;
    }
    
    /**
     * @brief Update a frame at specific index (thread-safe)
     * 
     * @param idx Frame index
     * @param frame New frame data
     * @return true if update was successful
     */
    bool update_frame(size_t idx, const cv::Mat& frame) {
        if (finalized_.load()) {
            return false;
        }
        
        std::unique_lock lock(mutex_);
        
        if (idx >= frames_.size()) {
            return false;
        }
        
        // Update memory usage tracking
        size_t old_bytes = frames_[idx].total() * frames_[idx].elemSize();
        size_t new_bytes = frame.total() * frame.elemSize();
        
        if (max_memory_bytes_ > 0) {
            if (total_memory_bytes_ - old_bytes + new_bytes > max_memory_bytes_) {
                return false; // Would exceed memory limit
            }
        }
        
        frames_[idx] = frame.clone();
        total_memory_bytes_ = total_memory_bytes_ - old_bytes + new_bytes;
        write_operations_.fetch_add(1);
        
        return true;
    }
    
    /**
     * @brief Get the number of frames in the buffer
     */
    size_t size() const {
        return frame_count_.load();
    }
    
    /**
     * @brief Check if buffer is empty
     */
    bool empty() const {
        return frame_count_.load() == 0;
    }
    
    /**
     * @brief Clear all frames from the buffer
     */
    void clear() {
        std::unique_lock lock(mutex_);
        frames_.clear();
        frame_count_.store(0);
        total_memory_bytes_ = 0;
        finalized_.store(false);
    }
    
    /**
     * @brief Reserve capacity for frames (thread-safe)
     */
    void reserve(size_t capacity) {
        std::unique_lock lock(mutex_);
        frames_.reserve(capacity);
    }
    
    /**
     * @brief Finalize the buffer (no more frames can be added)
     */
    void finalize() {
        finalized_.store(true);
    }
    
    /**
     * @brief Check if buffer is finalized
     */
    bool is_finalized() const {
        return finalized_.load();
    }
    
    /**
     * @brief Get memory usage statistics
     */
    struct MemoryStats {
        size_t total_bytes;
        size_t frame_count;
        size_t read_operations;
        size_t write_operations;
        bool is_finalized;
    };
    
    MemoryStats get_stats() const {
        return MemoryStats{
            total_memory_bytes_,
            frame_count_.load(),
            read_operations_.load(),
            write_operations_.load(),
            finalized_.load()
        };
    }
    
    /**
     * @brief Get frame at index with bounds checking (thread-safe)
     * Throws std::out_of_range if index is invalid
     */
    cv::Mat at(size_t idx) const {
        std::shared_lock lock(mutex_);
        
        if (idx >= frames_.size()) {
            throw std::out_of_range("Frame index " + std::to_string(idx) + 
                                  " is out of range (size: " + std::to_string(frames_.size()) + ")");
        }
        
        read_operations_.fetch_add(1);
        return frames_[idx].clone();
    }
    
    /**
     * @brief Apply a function to all frames (thread-safe)
     * 
     * @param func Function to apply to each frame (receives const cv::Mat&)
     */
    template<typename Func>
    void for_each(Func&& func) const {
        std::shared_lock lock(mutex_);
        
        for (const auto& frame : frames_) {
            func(frame);
        }
        
        read_operations_.fetch_add(1);
    }
    
    /**
     * @brief Apply a function to a range of frames (thread-safe)
     * 
     * @param start_idx Starting index
     * @param end_idx Ending index (exclusive)
     * @param func Function to apply to each frame
     */
    template<typename Func>
    void for_each_range(size_t start_idx, size_t end_idx, Func&& func) const {
        std::shared_lock lock(mutex_);
        
        size_t actual_end = std::min(end_idx, frames_.size());
        for (size_t i = start_idx; i < actual_end; ++i) {
            func(i, frames_[i]);
        }
        
        read_operations_.fetch_add(1);
    }
};

/**
 * @brief RAII lock guard for reading multiple frames safely
 * 
 * Ensures the frame buffer remains locked while accessing multiple frames
 */
class FrameBufferReadGuard {
private:
    std::shared_lock<std::shared_mutex> lock_;
    const ThreadSafeFrameBuffer& buffer_;

public:
    explicit FrameBufferReadGuard(const ThreadSafeFrameBuffer& buffer)
        : buffer_(buffer), lock_(buffer.mutex_) {}
    
    // Access frames without additional locking
    const cv::Mat& operator[](size_t idx) const {
        static const cv::Mat empty_mat;
        if (idx >= buffer_.frames_.size()) {
            return empty_mat;
        }
        return buffer_.frames_[idx];
    }
    
    size_t size() const {
        return buffer_.frames_.size();
    }
};

} // namespace VideoRepair