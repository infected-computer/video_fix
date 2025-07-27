#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <pybind11/numpy.h>
#include <pybind11/functional.h>
#include <pybind11/chrono.h>
#include <pybind11/complex.h>

#include "VideoRepair/VideoRepairEngine.h"
#include "Core/MemoryManager.h"
#include "Core/ErrorHandling.h"

#include <memory>
#include <vector>
#include <unordered_map>
#include <functional>
#include <chrono>

namespace py = pybind11;

/**
 * @brief Pybind11 bindings for VideoRepairEngine
 * 
 * This file provides comprehensive Python bindings for the C++ VideoRepairEngine
 * enabling seamless integration between Python orchestration and C++ processing
 */

// Custom type caster for cv::Mat to numpy array
namespace pybind11 { namespace detail {
    template <> struct type_caster<cv::Mat> {
    public:
        PYBIND11_TYPE_CASTER(cv::Mat, _("numpy.ndarray"));

        bool load(handle src, bool) {
            if (!isinstance<array>(src))
                return false;
            
            array buf = reinterpret_borrow<array>(src);
            
            if (py::array_t<uint8_t>::check_(buf)) {
                auto arr = buf.cast<py::array_t<uint8_t, py::array::c_style | py::array::forcecast>>();
                auto info = arr.request();
                
                int ndims = info.ndim;
                if (ndims == 2) {
                    value = cv::Mat(info.shape[0], info.shape[1], CV_8UC1, info.ptr);
                } else if (ndims == 3 && info.shape[2] == 3) {
                    value = cv::Mat(info.shape[0], info.shape[1], CV_8UC3, info.ptr);
                } else if (ndims == 3 && info.shape[2] == 4) {
                    value = cv::Mat(info.shape[0], info.shape[1], CV_8UC4, info.ptr);
                } else {
                    return false;
                }
                return true;
            }
            return false;
        }

        static handle cast(const cv::Mat& m, return_value_policy, handle defval) {
            std::string format = py::format_descriptor<uint8_t>::format();
            size_t itemsize = sizeof(uint8_t);
            
            std::vector<size_t> shape, strides;
            
            if (m.channels() == 1) {
                shape = {static_cast<size_t>(m.rows), static_cast<size_t>(m.cols)};
                strides = {sizeof(uint8_t) * m.cols, sizeof(uint8_t)};
            } else {
                shape = {static_cast<size_t>(m.rows), static_cast<size_t>(m.cols), static_cast<size_t>(m.channels())};
                strides = {sizeof(uint8_t) * m.cols * m.channels(), sizeof(uint8_t) * m.channels(), sizeof(uint8_t)};
            }
            
            return array(buffer_info(
                m.data,
                itemsize,
                format,
                shape.size(),
                shape,
                strides
            )).release();
        }
    };
}} // namespace pybind11::detail

/**
 * @brief Wrapper class for VideoRepairEngine to handle Python-specific needs
 */
class PyVideoRepairEngine {
public:
    PyVideoRepairEngine() : m_engine(std::make_unique<VideoRepairEngine>()) {}
    
    bool initialize() {
        return m_engine->initialize();
    }
    
    void shutdown() {
        m_engine->shutdown();
    }
    
    bool is_initialized() const {
        return m_engine->is_initialized();
    }
    
    // GPU capabilities wrapper
    py::dict get_gpu_capabilities() const {
        auto caps = m_engine->get_gpu_capabilities();
        py::dict result;
        
        result["cuda_available"] = caps.cuda_available;
        result["opencv_cuda_available"] = caps.opencv_cuda_available;
        result["tensorrt_available"] = caps.tensorrt_available;
        result["cudnn_available"] = caps.cudnn_available;
        result["device_count"] = caps.device_count;
        result["total_memory"] = caps.total_memory;
        result["free_memory"] = caps.free_memory;
        result["compute_capability_major"] = caps.compute_capability_major;
        result["compute_capability_minor"] = caps.compute_capability_minor;
        result["device_name"] = caps.device_name;
        result["available_devices"] = caps.available_devices;
        
        return result;
    }
    
    bool set_gpu_device(int device_id) {
        return m_engine->set_gpu_device(device_id);
    }
    
    int get_current_gpu_device() const {
        return m_engine->get_current_gpu_device();
    }
    
    // Format support
    bool is_format_supported(const std::string& file_path) const {
        return m_engine->is_format_supported(file_path);
    }
    
    std::string detect_video_format(const std::string& file_path) const {
        auto format = m_engine->detect_video_format(file_path);
        return VideoRepairUtils::format_to_string(format);
    }
    
    std::vector<std::string> get_supported_formats() const {
        auto formats = m_engine->get_supported_formats();
        std::vector<std::string> result;
        result.reserve(formats.size());
        
        for (const auto& format : formats) {
            result.push_back(VideoRepairUtils::format_to_string(format));
        }
        
        return result;
    }
    
    // File analysis wrapper
    py::dict analyze_file(const std::string& file_path) const {
        auto stream_info = m_engine->analyze_file(file_path);
        py::dict result;
        
        result["stream_index"] = stream_info.stream_index;
        result["media_type"] = static_cast<int>(stream_info.media_type);
        result["codec_id"] = static_cast<int>(stream_info.codec_id);
        result["detected_format"] = VideoRepairUtils::format_to_string(stream_info.detected_format);
        
        // Video information
        result["width"] = stream_info.width;
        result["height"] = stream_info.height;
        result["frame_rate"] = py::make_tuple(stream_info.frame_rate.num, stream_info.frame_rate.den);
        result["time_base"] = py::make_tuple(stream_info.time_base.num, stream_info.time_base.den);
        result["pixel_format"] = static_cast<int>(stream_info.pixel_format);
        result["bit_depth"] = stream_info.bit_depth;
        
        // Audio information
        result["sample_rate"] = stream_info.sample_rate;
        result["channels"] = stream_info.channels;
        result["sample_format"] = static_cast<int>(stream_info.sample_format);
        
        // Professional metadata
        result["timecode"] = stream_info.timecode;
        result["camera_model"] = stream_info.camera_model;
        result["lens_model"] = stream_info.lens_model;
        
        // Convert metadata map
        py::dict metadata;
        for (const auto& pair : stream_info.metadata) {
            metadata[pair.first] = pair.second;
        }
        result["metadata"] = metadata;
        
        // Corruption information
        result["has_corruption"] = stream_info.has_corruption;
        result["corruption_percentage"] = stream_info.corruption_percentage;
        
        py::list corrupted_ranges;
        for (const auto& range : stream_info.corrupted_ranges) {
            corrupted_ranges.append(py::make_tuple(range.first, range.second));
        }
        result["corrupted_ranges"] = corrupted_ranges;
        
        // Quality metrics
        result["psnr"] = stream_info.psnr;
        result["ssim"] = stream_info.ssim;
        result["vmaf"] = stream_info.vmaf;
        
        return result;
    }
    
    // Repair recommendations wrapper
    py::list recommend_repair_techniques(const py::dict& stream_info_dict) const {
        // Convert Python dict back to StreamInfo
        VideoRepairEngine::StreamInfo stream_info;
        
        if (stream_info_dict.contains("width"))
            stream_info.width = stream_info_dict["width"].cast<int>();
        if (stream_info_dict.contains("height"))
            stream_info.height = stream_info_dict["height"].cast<int>();
        if (stream_info_dict.contains("has_corruption"))
            stream_info.has_corruption = stream_info_dict["has_corruption"].cast<bool>();
        if (stream_info_dict.contains("corruption_percentage"))
            stream_info.corruption_percentage = stream_info_dict["corruption_percentage"].cast<double>();
        if (stream_info_dict.contains("psnr"))
            stream_info.psnr = stream_info_dict["psnr"].cast<double>();
        
        auto techniques = m_engine->recommend_repair_techniques(stream_info);
        py::list result;
        
        for (const auto& technique : techniques) {
            result.append(VideoRepairUtils::technique_to_string(technique));
        }
        
        return result;
    }
    
    double estimate_repair_time(const py::dict& stream_info_dict, const py::list& techniques_list) const {
        // Convert inputs
        VideoRepairEngine::StreamInfo stream_info;
        // ... populate stream_info from dict (abbreviated for brevity)
        
        std::vector<VideoRepairEngine::RepairTechnique> techniques;
        // ... convert techniques list (abbreviated for brevity)
        
        return m_engine->estimate_repair_time(stream_info, techniques);
    }
    
    // Repair operations wrapper
    std::string start_repair_async(const py::dict& parameters_dict) {
        VideoRepairEngine::RepairParameters params = convert_repair_parameters(parameters_dict);
        return m_engine->start_repair_async(params);
    }
    
    py::dict repair_file_sync(const py::dict& parameters_dict) {
        VideoRepairEngine::RepairParameters params = convert_repair_parameters(parameters_dict);
        auto result = m_engine->repair_file_sync(params);
        return convert_repair_result(result);
    }
    
    // Session management wrappers
    std::string get_repair_status(const std::string& session_id) const {
        auto status = m_engine->get_repair_status(session_id);
        return VideoRepairUtils::status_to_string(status);
    }
    
    double get_repair_progress(const std::string& session_id) const {
        return m_engine->get_repair_progress(session_id);
    }
    
    bool cancel_repair(const std::string& session_id) {
        return m_engine->cancel_repair(session_id);
    }
    
    py::dict get_repair_result(const std::string& session_id) const {
        auto result = m_engine->get_repair_result(session_id);
        return convert_repair_result(result);
    }
    
    void cleanup_session(const std::string& session_id) {
        m_engine->cleanup_session(session_id);
    }
    
    // Batch processing wrapper
    std::string start_batch_repair(const py::dict& batch_job_dict) {
        VideoRepairEngine::BatchJob batch_job;
        
        if (batch_job_dict.contains("job_id"))
            batch_job.job_id = batch_job_dict["job_id"].cast<std::string>();
        if (batch_job_dict.contains("input_files"))
            batch_job.input_files = batch_job_dict["input_files"].cast<std::vector<std::string>>();
        if (batch_job_dict.contains("output_directory"))
            batch_job.output_directory = batch_job_dict["output_directory"].cast<std::string>();
        if (batch_job_dict.contains("base_parameters"))
            batch_job.base_parameters = convert_repair_parameters(batch_job_dict["base_parameters"].cast<py::dict>());
        
        return m_engine->start_batch_repair(batch_job);
    }
    
    py::list get_batch_results(const std::string& job_id) const {
        auto results = m_engine->get_batch_results(job_id);
        py::list py_results;
        
        for (const auto& result : results) {
            py_results.append(convert_repair_result(result));
        }
        
        return py_results;
    }
    
    // Configuration methods
    void set_memory_limit(size_t limit_mb) {
        m_engine->set_memory_limit(limit_mb);
    }
    
    void set_thread_count(int thread_count) {
        m_engine->set_thread_count(thread_count);
    }
    
    void enable_debug_output(bool enable) {
        m_engine->enable_debug_output(enable);
    }
    
    void set_log_level(int level) {
        m_engine->set_log_level(level);
    }
    
    // Performance monitoring wrapper
    py::dict get_performance_metrics() const {
        auto metrics = m_engine->get_performance_metrics();
        py::dict result;
        
        result["cpu_usage_percent"] = metrics.cpu_usage_percent;
        result["gpu_usage_percent"] = metrics.gpu_usage_percent;
        result["memory_usage_mb"] = metrics.memory_usage_mb;
        result["gpu_memory_usage_mb"] = metrics.gpu_memory_usage_mb;
        result["active_sessions"] = metrics.active_sessions;
        result["average_processing_fps"] = metrics.average_processing_fps;
        
        return result;
    }
    
    void reset_performance_counters() {
        m_engine->reset_performance_counters();
    }

private:
    std::unique_ptr<VideoRepairEngine> m_engine;
    
    // Helper methods for parameter conversion
    VideoRepairEngine::RepairParameters convert_repair_parameters(const py::dict& params_dict) {
        VideoRepairEngine::RepairParameters params;
        
        if (params_dict.contains("input_file"))
            params.input_file = params_dict["input_file"].cast<std::string>();
        if (params_dict.contains("output_file"))
            params.output_file = params_dict["output_file"].cast<std::string>();
        if (params_dict.contains("reference_file"))
            params.reference_file = params_dict["reference_file"].cast<std::string>();
        
        // Convert techniques list
        if (params_dict.contains("techniques")) {
            auto techniques_list = params_dict["techniques"].cast<py::list>();
            for (const auto& tech_name : techniques_list) {
                std::string tech_str = tech_name.cast<std::string>();
                params.techniques.push_back(string_to_technique(tech_str));
            }
        }
        
        if (params_dict.contains("use_gpu"))
            params.use_gpu = params_dict["use_gpu"].cast<bool>();
        if (params_dict.contains("gpu_device_id"))
            params.gpu_device_id = params_dict["gpu_device_id"].cast<int>();
        if (params_dict.contains("enable_ai_processing"))
            params.enable_ai_processing = params_dict["enable_ai_processing"].cast<bool>();
        if (params_dict.contains("preserve_original_quality"))
            params.preserve_original_quality = params_dict["preserve_original_quality"].cast<bool>();
        
        if (params_dict.contains("max_threads"))
            params.max_threads = params_dict["max_threads"].cast<int>();
        if (params_dict.contains("memory_limit_mb"))
            params.memory_limit_mb = params_dict["memory_limit_mb"].cast<size_t>();
        if (params_dict.contains("processing_queue_size"))
            params.processing_queue_size = params_dict["processing_queue_size"].cast<int>();
        if (params_dict.contains("enable_hardware_decoding"))
            params.enable_hardware_decoding = params_dict["enable_hardware_decoding"].cast<bool>();
        if (params_dict.contains("enable_hardware_encoding"))
            params.enable_hardware_encoding = params_dict["enable_hardware_encoding"].cast<bool>();
        
        if (params_dict.contains("maintain_bit_depth"))
            params.maintain_bit_depth = params_dict["maintain_bit_depth"].cast<bool>();
        if (params_dict.contains("maintain_color_space"))
            params.maintain_color_space = params_dict["maintain_color_space"].cast<bool>();
        if (params_dict.contains("maintain_frame_rate"))
            params.maintain_frame_rate = params_dict["maintain_frame_rate"].cast<bool>();
        if (params_dict.contains("quality_factor"))
            params.quality_factor = params_dict["quality_factor"].cast<double>();
        
        if (params_dict.contains("ai_strength"))
            params.ai_strength = params_dict["ai_strength"].cast<double>();
        if (params_dict.contains("mark_ai_regions"))
            params.mark_ai_regions = params_dict["mark_ai_regions"].cast<bool>();
        if (params_dict.contains("ai_model_path"))
            params.ai_model_path = params_dict["ai_model_path"].cast<std::string>();
        
        // Set up callbacks if provided
        if (params_dict.contains("progress_callback")) {
            auto py_callback = params_dict["progress_callback"].cast<py::function>();
            params.progress_callback = [py_callback](double progress, const std::string& status) {
                py_callback(progress, status);
            };
        }
        
        if (params_dict.contains("log_callback")) {
            auto py_callback = params_dict["log_callback"].cast<py::function>();
            params.log_callback = [py_callback](const std::string& message) {
                py_callback(message);
            };
        }
        
        return params;
    }
    
    py::dict convert_repair_result(const VideoRepairEngine::RepairResult& result) {
        py::dict py_result;
        
        py_result["success"] = result.success;
        py_result["final_status"] = VideoRepairUtils::status_to_string(result.final_status);
        py_result["error_message"] = result.error_message;
        
        // Processing statistics
        py_result["processing_time_seconds"] = result.processing_time_seconds;
        py_result["gpu_utilization_average"] = result.gpu_utilization_average;
        py_result["memory_peak_usage_mb"] = result.memory_peak_usage_mb;
        py_result["frames_processed"] = result.frames_processed;
        py_result["frames_repaired"] = result.frames_repaired;
        
        // Quality metrics
        py::dict quality_metrics;
        quality_metrics["psnr_before"] = result.quality_metrics.psnr_before;
        quality_metrics["psnr_after"] = result.quality_metrics.psnr_after;
        quality_metrics["ssim_before"] = result.quality_metrics.ssim_before;
        quality_metrics["ssim_after"] = result.quality_metrics.ssim_after;
        quality_metrics["vmaf_before"] = result.quality_metrics.vmaf_before;
        quality_metrics["vmaf_after"] = result.quality_metrics.vmaf_after;
        py_result["quality_metrics"] = quality_metrics;
        
        // Techniques applied
        py::list techniques_applied;
        for (const auto& technique : result.techniques_applied) {
            techniques_applied.append(VideoRepairUtils::technique_to_string(technique));
        }
        py_result["techniques_applied"] = techniques_applied;
        
        // Repair details
        py::dict repair_details;
        for (const auto& pair : result.repair_details) {
            repair_details[pair.first] = pair.second;
        }
        py_result["repair_details"] = repair_details;
        
        // AI results
        py::dict ai_results;
        ai_results["ai_processing_used"] = result.ai_results.ai_processing_used;
        ai_results["frames_ai_processed"] = result.ai_results.frames_ai_processed;
        ai_results["ai_confidence_average"] = result.ai_results.ai_confidence_average;
        
        py::list ai_regions;
        for (const auto& region : result.ai_results.ai_processed_regions) {
            py::dict region_dict;
            region_dict["x"] = region.x;
            region_dict["y"] = region.y;
            region_dict["width"] = region.width;
            region_dict["height"] = region.height;
            ai_regions.append(region_dict);
        }
        ai_results["ai_processed_regions"] = ai_regions;
        py_result["ai_results"] = ai_results;
        
        // Output information
        py_result["output_file_path"] = result.output_file_path;
        py_result["output_file_size"] = result.output_file_size;
        
        // Warnings and recommendations
        py_result["warnings"] = result.warnings;
        py_result["recommendations"] = result.recommendations;
        
        return py_result;
    }
    
    VideoRepairEngine::RepairTechnique string_to_technique(const std::string& technique_str) {
        if (technique_str == "header_reconstruction") return VideoRepairEngine::RepairTechnique::HEADER_RECONSTRUCTION;
        if (technique_str == "index_rebuild") return VideoRepairEngine::RepairTechnique::INDEX_REBUILD;
        if (technique_str == "fragment_recovery") return VideoRepairEngine::RepairTechnique::FRAGMENT_RECOVERY;
        if (technique_str == "container_remux") return VideoRepairEngine::RepairTechnique::CONTAINER_REMUX;
        if (technique_str == "frame_interpolation") return VideoRepairEngine::RepairTechnique::FRAME_INTERPOLATION;
        if (technique_str == "ai_inpainting") return VideoRepairEngine::RepairTechnique::AI_INPAINTING;
        if (technique_str == "super_resolution") return VideoRepairEngine::RepairTechnique::SUPER_RESOLUTION;
        if (technique_str == "denoising") return VideoRepairEngine::RepairTechnique::DENOISING;
        if (technique_str == "metadata_recovery") return VideoRepairEngine::RepairTechnique::METADATA_RECOVERY;
        
        throw std::invalid_argument("Unknown repair technique: " + technique_str);
    }
};

/**
 * @brief Utility class for shared memory operations between Python and C++
 */
class SharedMemoryManager {
public:
    SharedMemoryManager() = default;
    
    // Create shared buffer for large data transfers
    py::array_t<uint8_t> create_shared_buffer(size_t size) {
        auto buffer = py::array_t<uint8_t>(size);
        auto info = buffer.request();
        
        m_shared_buffers[reinterpret_cast<uintptr_t>(info.ptr)] = buffer;
        
        return buffer;
    }
    
    // Release shared buffer
    void release_shared_buffer(py::array_t<uint8_t> buffer) {
        auto info = buffer.request();
        auto ptr_key = reinterpret_cast<uintptr_t>(info.ptr);
        
        auto it = m_shared_buffers.find(ptr_key);
        if (it != m_shared_buffers.end()) {
            m_shared_buffers.erase(it);
        }
    }
    
    // Get buffer statistics
    py::dict get_buffer_stats() const {
        py::dict stats;
        stats["active_buffers"] = m_shared_buffers.size();
        
        size_t total_size = 0;
        for (const auto& pair : m_shared_buffers) {
            auto info = pair.second.request();
            total_size += info.size * info.itemsize;
        }
        stats["total_memory_mb"] = total_size / (1024.0 * 1024.0);
        
        return stats;
    }
    
private:
    std::unordered_map<uintptr_t, py::array_t<uint8_t>> m_shared_buffers;
};

/**
 * @brief Performance profiler for Python-C++ interface
 */
class InterfaceProfiler {
public:
    struct ProfileEntry {
        std::string function_name;
        std::chrono::high_resolution_clock::time_point start_time;
        std::chrono::high_resolution_clock::time_point end_time;
        size_t data_size_bytes;
    };
    
    void start_profile(const std::string& function_name, size_t data_size = 0) {
        std::lock_guard<std::mutex> lock(m_mutex);
        
        ProfileEntry entry;
        entry.function_name = function_name;
        entry.start_time = std::chrono::high_resolution_clock::now();
        entry.data_size_bytes = data_size;
        
        m_active_profiles[function_name] = entry;
    }
    
    void end_profile(const std::string& function_name) {
        std::lock_guard<std::mutex> lock(m_mutex);
        
        auto it = m_active_profiles.find(function_name);
        if (it != m_active_profiles.end()) {
            it->second.end_time = std::chrono::high_resolution_clock::now();
            m_completed_profiles.push_back(it->second);
            m_active_profiles.erase(it);
        }
    }
    
    py::dict get_profile_statistics() const {
        std::lock_guard<std::mutex> lock(m_mutex);
        
        py::dict stats;
        std::unordered_map<std::string, std::vector<double>> function_times;
        std::unordered_map<std::string, std::vector<double>> function_throughputs;
        
        for (const auto& entry : m_completed_profiles) {
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
                entry.end_time - entry.start_time).count() / 1000.0; // Convert to milliseconds
            
            function_times[entry.function_name].push_back(duration);
            
            if (entry.data_size_bytes > 0 && duration > 0) {
                double throughput = (entry.data_size_bytes / (1024.0 * 1024.0)) / (duration / 1000.0); // MB/s
                function_throughputs[entry.function_name].push_back(throughput);
            }
        }
        
        py::dict function_stats;
        for (const auto& pair : function_times) {
            const auto& times = pair.second;
            if (times.empty()) continue;
            
            py::dict func_stats;
            func_stats["call_count"] = times.size();
            func_stats["total_time_ms"] = std::accumulate(times.begin(), times.end(), 0.0);
            func_stats["average_time_ms"] = func_stats["total_time_ms"].cast<double>() / times.size();
            func_stats["min_time_ms"] = *std::min_element(times.begin(), times.end());
            func_stats["max_time_ms"] = *std::max_element(times.begin(), times.end());
            
            if (function_throughputs.count(pair.first)) {
                const auto& throughputs = function_throughputs[pair.first];
                func_stats["average_throughput_mbps"] = std::accumulate(throughputs.begin(), throughputs.end(), 0.0) / throughputs.size();
            }
            
            function_stats[pair.first] = func_stats;
        }
        
        stats["functions"] = function_stats;
        stats["total_profiles"] = m_completed_profiles.size();
        stats["active_profiles"] = m_active_profiles.size();
        
        return stats;
    }
    
    void clear_statistics() {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_completed_profiles.clear();
        m_active_profiles.clear();
    }
    
private:
    mutable std::mutex m_mutex;
    std::vector<ProfileEntry> m_completed_profiles;
    std::unordered_map<std::string, ProfileEntry> m_active_profiles;
};

// Global instances
static SharedMemoryManager g_memory_manager;
static InterfaceProfiler g_profiler;

/**
 * @brief RAII profiler for automatic function profiling
 */
class ScopedProfiler {
public:
    ScopedProfiler(const std::string& function_name, size_t data_size = 0) 
        : m_function_name(function_name) {
        g_profiler.start_profile(m_function_name, data_size);
    }
    
    ~ScopedProfiler() {
        g_profiler.end_profile(m_function_name);
    }
    
private:
    std::string m_function_name;
};

#define PROFILE_FUNCTION(name, size) ScopedProfiler _prof(name, size)

/**
 * @brief Enhanced PyVideoRepairEngine with profiling
 */
class ProfiledPyVideoRepairEngine : public PyVideoRepairEngine {
public:
    py::dict analyze_file(const std::string& file_path) const override {
        PROFILE_FUNCTION("analyze_file", 0);
        return PyVideoRepairEngine::analyze_file(file_path);
    }
    
    std::string start_repair_async(const py::dict& parameters_dict) override {
        PROFILE_FUNCTION("start_repair_async", 0);
        return PyVideoRepairEngine::start_repair_async(parameters_dict);
    }
    
    py::dict repair_file_sync(const py::dict& parameters_dict) override {
        PROFILE_FUNCTION("repair_file_sync", 0);
        return PyVideoRepairEngine::repair_file_sync(parameters_dict);
    }
};

// Module definition
PYBIND11_MODULE(video_repair_engine, m) {
    m.doc() = "PhoenixDRS Video Repair Engine - Python bindings for high-performance video repair";
    
    // Bind main engine class
    py::class_<PyVideoRepairEngine>(m, "VideoRepairEngine")
        .def(py::init<>())
        .def("initialize", &PyVideoRepairEngine::initialize,
             "Initialize the video repair engine")
        .def("shutdown", &PyVideoRepairEngine::shutdown,
             "Shutdown the video repair engine")
        .def("is_initialized", &PyVideoRepairEngine::is_initialized,
             "Check if engine is initialized")
        
        // GPU management
        .def("get_gpu_capabilities", &PyVideoRepairEngine::get_gpu_capabilities,
             "Get GPU capabilities and information")
        .def("set_gpu_device", &PyVideoRepairEngine::set_gpu_device,
             "Set active GPU device", py::arg("device_id"))
        .def("get_current_gpu_device", &PyVideoRepairEngine::get_current_gpu_device,
             "Get current GPU device ID")
        
        // Format support
        .def("is_format_supported", &PyVideoRepairEngine::is_format_supported,
             "Check if video format is supported", py::arg("file_path"))
        .def("detect_video_format", &PyVideoRepairEngine::detect_video_format,
             "Detect video format of file", py::arg("file_path"))
        .def("get_supported_formats", &PyVideoRepairEngine::get_supported_formats,
             "Get list of supported video formats")
        
        // File analysis
        .def("analyze_file", &PyVideoRepairEngine::analyze_file,
             "Analyze video file and return detailed information", py::arg("file_path"))
        .def("recommend_repair_techniques", &PyVideoRepairEngine::recommend_repair_techniques,
             "Recommend repair techniques based on analysis", py::arg("stream_info"))
        .def("estimate_repair_time", &PyVideoRepairEngine::estimate_repair_time,
             "Estimate repair time for given techniques", py::arg("stream_info"), py::arg("techniques"))
        
        // Repair operations
        .def("start_repair_async", &PyVideoRepairEngine::start_repair_async,
             "Start asynchronous repair operation", py::arg("parameters"))
        .def("repair_file_sync", &PyVideoRepairEngine::repair_file_sync,
             "Perform synchronous repair operation", py::arg("parameters"))
        
        // Session management
        .def("get_repair_status", &PyVideoRepairEngine::get_repair_status,
             "Get repair session status", py::arg("session_id"))
        .def("get_repair_progress", &PyVideoRepairEngine::get_repair_progress,
             "Get repair session progress", py::arg("session_id"))
        .def("cancel_repair", &PyVideoRepairEngine::cancel_repair,
             "Cancel repair session", py::arg("session_id"))
        .def("get_repair_result", &PyVideoRepairEngine::get_repair_result,
             "Get repair session result", py::arg("session_id"))
        .def("cleanup_session", &PyVideoRepairEngine::cleanup_session,
             "Clean up completed session", py::arg("session_id"))
        
        // Batch processing
        .def("start_batch_repair", &PyVideoRepairEngine::start_batch_repair,
             "Start batch repair operation", py::arg("batch_job"))
        .def("get_batch_results", &PyVideoRepairEngine::get_batch_results,
             "Get batch repair results", py::arg("job_id"))
        
        // Configuration
        .def("set_memory_limit", &PyVideoRepairEngine::set_memory_limit,
             "Set memory limit in MB", py::arg("limit_mb"))
        .def("set_thread_count", &PyVideoRepairEngine::set_thread_count,
             "Set number of worker threads", py::arg("thread_count"))
        .def("enable_debug_output", &PyVideoRepairEngine::enable_debug_output,
             "Enable/disable debug output", py::arg("enable"))
        .def("set_log_level", &PyVideoRepairEngine::set_log_level,
             "Set logging level", py::arg("level"))
        
        // Performance monitoring
        .def("get_performance_metrics", &PyVideoRepairEngine::get_performance_metrics,
             "Get current performance metrics")
        .def("reset_performance_counters", &PyVideoRepairEngine::reset_performance_counters,
             "Reset performance counters");
    
    // Bind profiled version
    py::class_<ProfiledPyVideoRepairEngine, PyVideoRepairEngine>(m, "ProfiledVideoRepairEngine")
        .def(py::init<>());
    
    // Bind SharedMemoryManager
    py::class_<SharedMemoryManager>(m, "SharedMemoryManager")
        .def("create_shared_buffer", &SharedMemoryManager::create_shared_buffer,
             "Create shared memory buffer", py::arg("size"))
        .def("release_shared_buffer", &SharedMemoryManager::release_shared_buffer,
             "Release shared memory buffer", py::arg("buffer"))
        .def("get_buffer_stats", &SharedMemoryManager::get_buffer_stats,
             "Get buffer statistics");
    
    // Bind InterfaceProfiler
    py::class_<InterfaceProfiler>(m, "InterfaceProfiler")
        .def("get_profile_statistics", &InterfaceProfiler::get_profile_statistics,
             "Get profiling statistics")
        .def("clear_statistics", &InterfaceProfiler::clear_statistics,
             "Clear profiling statistics");
    
    // Global instances
    m.def("get_memory_manager", []() -> SharedMemoryManager& { return g_memory_manager; },
          py::return_value_policy::reference,
          "Get global shared memory manager");
    
    m.def("get_profiler", []() -> InterfaceProfiler& { return g_profiler; },
          py::return_value_policy::reference,
          "Get global interface profiler");
    
    // Utility functions
    m.def("format_to_string", &VideoRepairUtils::format_to_string,
          "Convert video format enum to string", py::arg("format"));
    m.def("technique_to_string", &VideoRepairUtils::technique_to_string,
          "Convert repair technique enum to string", py::arg("technique"));
    m.def("status_to_string", &VideoRepairUtils::status_to_string,
          "Convert repair status enum to string", py::arg("status"));
    m.def("is_professional_format", &VideoRepairUtils::is_professional_format,
          "Check if format is professional", py::arg("format"));
    m.def("requires_reference_file", &VideoRepairUtils::requires_reference_file,
          "Check if technique requires reference file", py::arg("technique"));
    
    // Constants
    m.attr("__version__") = "2.0.0";
    m.attr("__author__") = "PhoenixDRS Team";
    
    // Exception handling
    py::register_exception<VideoRepairException>(m, "VideoRepairException");
    
    // Module initialization
    m.def("initialize_logging", [](int level) {
        // Initialize logging for the C++ side
        // This would set up proper logging integration
    }, "Initialize logging system", py::arg("level") = 2);
    
    m.def("get_build_info", []() {
        py::dict info;
        info["build_date"] = __DATE__;
        info["build_time"] = __TIME__;
        info["compiler"] = 
#ifdef __GNUC__
            "GCC " + std::to_string(__GNUC__) + "." + std::to_string(__GNUC_MINOR__);
#elif defined(_MSC_VER)
            "MSVC " + std::to_string(_MSC_VER);
#elif defined(__clang__)
            "Clang " + std::to_string(__clang_major__) + "." + std::to_string(__clang_minor__);
#else
            "Unknown";
#endif
        
        info["ffmpeg_version"] = av_version_info();
        info["opencv_version"] = CV_VERSION;
        
#ifdef CUDA_VERSION
        info["cuda_version"] = CUDA_VERSION;
#endif
        
        return info;
    }, "Get build information");
}