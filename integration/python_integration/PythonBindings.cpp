/*
 * PhoenixDRS Professional - Python Bindings Implementation
 * מימוש Python Bindings - PhoenixDRS מקצועי
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <pybind11/chrono.h>

#include "PythonBridge.h"
#include "../cpp_gui/include/DiskImager.h"
#include "../cpp_gui/include/FileCarver.h"
#include "../cpp_gui/include/VideoRebuilder.h"
#include "../cpp_gui/include/ForensicLogger.h"
#include "../cpp_gui/include/PerformanceMonitor.h"
#include "../cpp_gui/include/MLClassifier.h"
#include "../cpp_gui/include/AIAnalyzer.h"

namespace py = pybind11;
using namespace PhoenixDRS;
using namespace PhoenixDRS::Integration;

PYBIND11_MODULE(phoenixdrs_cpp, m) {
    m.doc() = R"pbdoc(
        PhoenixDRS Professional - C++ Forensics Engine Python Bindings
        מנוע פורנזיקה C++ עם Python Bindings - PhoenixDRS מקצועי
        
        This module provides high-performance C++ implementations of forensic
        analysis tools with Python bindings for easy integration.
    )pbdoc";

    // Error handling
    py::register_exception<PhoenixDRS::Core::PhoenixException>(m, "PhoenixException");
    
    py::enum_<PhoenixDRS::Core::ErrorCode>(m, "ErrorCode")
        .value("Success", PhoenixDRS::Core::ErrorCode::Success)
        .value("UnknownError", PhoenixDRS::Core::ErrorCode::UnknownError)
        .value("InvalidParameter", PhoenixDRS::Core::ErrorCode::InvalidParameter)
        .value("FileNotFound", PhoenixDRS::Core::ErrorCode::FileNotFound)
        .value("AccessDenied", PhoenixDRS::Core::ErrorCode::AccessDenied)
        .value("OutOfMemory", PhoenixDRS::Core::ErrorCode::OutOfMemory)
        .export_values();

    // DiskImager Python wrapper
    py::class_<PythonDiskImager::ImageResult>(m, "ImageResult")
        .def(py::init<>())
        .def_readwrite("success", &PythonDiskImager::ImageResult::success)
        .def_readwrite("error_message", &PythonDiskImager::ImageResult::errorMessage)
        .def_readwrite("image_path", &PythonDiskImager::ImageResult::imagePath)
        .def_readwrite("total_bytes", &PythonDiskImager::ImageResult::totalBytes)
        .def_readwrite("total_sectors", &PythonDiskImager::ImageResult::totalSectors)
        .def_readwrite("bad_sector_count", &PythonDiskImager::ImageResult::badSectorCount)
        .def_readwrite("md5_hash", &PythonDiskImager::ImageResult::md5Hash)
        .def_readwrite("sha256_hash", &PythonDiskImager::ImageResult::sha256Hash)
        .def_readwrite("elapsed_seconds", &PythonDiskImager::ImageResult::elapsedSeconds);

    py::class_<PythonDiskImager>(m, "DiskImager")
        .def(py::init<>())
        .def("create_image", &PythonDiskImager::createImage,
             "Create disk image from source device",
             py::arg("source_path"), 
             py::arg("destination_path"),
             py::arg("sector_size") = 512,
             py::arg("progress_callback") = py::none(),
             py::arg("error_callback") = py::none())
        .def("verify_image", &PythonDiskImager::verifyImage,
             "Verify integrity of disk image",
             py::arg("image_path"),
             py::arg("progress_callback") = py::none())
        .def("is_running", &PythonDiskImager::isRunning)
        .def("cancel", &PythonDiskImager::cancel)
        .def("set_max_retries", &PythonDiskImager::setMaxRetries)
        .def("set_retry_delay", &PythonDiskImager::setRetryDelay)
        .def("set_compression_enabled", &PythonDiskImager::setCompressionEnabled)
        .def("set_encryption_enabled", &PythonDiskImager::setEncryptionEnabled);

    // FileCarver Python wrapper
    py::class_<PythonFileCarver::CarveResult>(m, "CarveResult")
        .def(py::init<>())
        .def_readwrite("success", &PythonFileCarver::CarveResult::success)
        .def_readwrite("error_message", &PythonFileCarver::CarveResult::errorMessage)
        .def_readwrite("carved_files", &PythonFileCarver::CarveResult::carvedFiles)
        .def_readwrite("total_files_found", &PythonFileCarver::CarveResult::totalFilesFound)
        .def_readwrite("valid_files", &PythonFileCarver::CarveResult::validFiles)
        .def_readwrite("total_bytes_recovered", &PythonFileCarver::CarveResult::totalBytesRecovered)
        .def_readwrite("elapsed_seconds", &PythonFileCarver::CarveResult::elapsedSeconds);

    py::class_<PythonFileCarver>(m, "FileCarver")
        .def(py::init<>())
        .def("carve_files", &PythonFileCarver::carveFiles,
             "Carve files from disk image",
             py::arg("image_path"),
             py::arg("output_directory"),
             py::arg("signatures_path") = "",
             py::arg("progress_callback") = py::none(),
             py::arg("file_found_callback") = py::none())
        .def("carve_files_parallel", &PythonFileCarver::carveFilesParallel,
             "Carve files using parallel processing",
             py::arg("image_path"),
             py::arg("output_directory"),
             py::arg("max_workers") = 0,
             py::arg("signatures_path") = "",
             py::arg("progress_callback") = py::none(),
             py::arg("file_found_callback") = py::none())
        .def("is_running", &PythonFileCarver::isRunning)
        .def("cancel", &PythonFileCarver::cancel)
        .def("set_chunk_size", &PythonFileCarver::setChunkSize)
        .def("set_min_file_size", &PythonFileCarver::setMinFileSize)
        .def("set_max_file_size", &PythonFileCarver::setMaxFileSize)
        .def("set_file_type_filter", &PythonFileCarver::setFileTypeFilter);

    // VideoRebuilder Python wrapper  
    py::class_<PythonVideoRebuilder::RebuildResult>(m, "RebuildResult")
        .def(py::init<>())
        .def_readwrite("success", &PythonVideoRebuilder::RebuildResult::success)
        .def_readwrite("error_message", &PythonVideoRebuilder::RebuildResult::errorMessage)
        .def_readwrite("rebuilt_videos", &PythonVideoRebuilder::RebuildResult::rebuiltVideos)
        .def_readwrite("total_videos_found", &PythonVideoRebuilder::RebuildResult::totalVideosFound)
        .def_readwrite("successful_rebuilds", &PythonVideoRebuilder::RebuildResult::successfulRebuilds)
        .def_readwrite("total_bytes_processed", &PythonVideoRebuilder::RebuildResult::totalBytesProcessed)
        .def_readwrite("elapsed_seconds", &PythonVideoRebuilder::RebuildResult::elapsedSeconds);

    py::class_<PythonVideoRebuilder>(m, "VideoRebuilder")
        .def(py::init<>())
        .def("rebuild_videos", &PythonVideoRebuilder::rebuildVideos,
             "Rebuild video files from disk image",
             py::arg("image_path"),
             py::arg("output_directory"),
             py::arg("video_format") = "mov",
             py::arg("progress_callback") = py::none(),
             py::arg("video_found_callback") = py::none())
        .def("is_running", &PythonVideoRebuilder::isRunning)
        .def("cancel", &PythonVideoRebuilder::cancel)
        .def("set_max_video_size", &PythonVideoRebuilder::setMaxVideoSize)
        .def("set_min_video_size", &PythonVideoRebuilder::setMinVideoSize)
        .def("set_quality_threshold", &PythonVideoRebuilder::setQualityThreshold);

    // Unified API
    py::class_<PythonForensicsAPI::GlobalStatistics>(m, "GlobalStatistics")
        .def(py::init<>())
        .def_readwrite("active_operations", &PythonForensicsAPI::GlobalStatistics::activeOperations)
        .def_readwrite("total_bytes_processed", &PythonForensicsAPI::GlobalStatistics::totalBytesProcessed)
        .def_readwrite("total_images_created", &PythonForensicsAPI::GlobalStatistics::totalImagesCreated)
        .def_readwrite("total_files_carved", &PythonForensicsAPI::GlobalStatistics::totalFilesCarved)
        .def_readwrite("total_videos_rebuilt", &PythonForensicsAPI::GlobalStatistics::totalVideosRebuilt)
        .def_readwrite("total_elapsed_time", &PythonForensicsAPI::GlobalStatistics::totalElapsedTime);

    py::class_<PythonForensicsAPI>(m, "ForensicsAPI")
        .def_static("instance", &PythonForensicsAPI::instance, 
                   py::return_value_policy::reference_internal)
        .def("initialize", &PythonForensicsAPI::initialize)
        .def("shutdown", &PythonForensicsAPI::shutdown)
        .def("create_disk_imager", &PythonForensicsAPI::createDiskImager)
        .def("create_file_carver", &PythonForensicsAPI::createFileCarver)
        .def("create_video_rebuilder", &PythonForensicsAPI::createVideoRebuilder)
        .def("set_temp_directory", &PythonForensicsAPI::setTempDirectory)
        .def("get_temp_directory", &PythonForensicsAPI::getTempDirectory)
        .def("set_log_level", &PythonForensicsAPI::setLogLevel)
        .def("get_log_level", &PythonForensicsAPI::getLogLevel)
        .def("enable_performance_logging", &PythonForensicsAPI::enablePerformanceLogging)
        .def("is_performance_logging_enabled", &PythonForensicsAPI::isPerformanceLoggingEnabled)
        .def("get_system_info", &PythonForensicsAPI::getSystemInfo)
        .def("get_memory_info", &PythonForensicsAPI::getMemoryInfo)
        .def("get_disk_info", &PythonForensicsAPI::getDiskInfo)
        .def("get_last_error", &PythonForensicsAPI::getLastError)
        .def("clear_last_error", &PythonForensicsAPI::clearLastError)
        .def("get_global_statistics", &PythonForensicsAPI::getGlobalStatistics)
        .def("reset_global_statistics", &PythonForensicsAPI::resetGlobalStatistics);

    // SharedMemoryManager for high-performance data exchange
    py::class_<SharedMemoryManager>(m, "SharedMemoryManager")
        .def_static("instance", &SharedMemoryManager::instance,
                   py::return_value_policy::reference_internal)
        .def("create_shared_buffer", &SharedMemoryManager::createSharedBuffer)
        .def("attach_to_shared_buffer", &SharedMemoryManager::attachToSharedBuffer)
        .def("detach_from_shared_buffer", &SharedMemoryManager::detachFromSharedBuffer)
        .def("delete_shared_buffer", &SharedMemoryManager::deleteSharedBuffer)
        .def("get_buffer_size", &SharedMemoryManager::getBufferSize)
        .def("lock_buffer", &SharedMemoryManager::lockBuffer)
        .def("unlock_buffer", &SharedMemoryManager::unlockBuffer)
        .def("get_active_buffers", &SharedMemoryManager::getActiveBuffers)
        .def("get_total_memory_used", &SharedMemoryManager::getTotalMemoryUsed);

    // Advanced ML/AI classes for Python integration
    py::enum_<MLModelType>(m, "MLModelType")
        .value("FILE_TYPE_CLASSIFIER", MLModelType::FILE_TYPE_CLASSIFIER)
        .value("MALWARE_DETECTOR", MLModelType::MALWARE_DETECTOR)
        .value("FACE_RECOGNIZER", MLModelType::FACE_RECOGNIZER)
        .value("OBJECT_DETECTOR", MLModelType::OBJECT_DETECTOR)
        .value("TEXT_ANALYZER", MLModelType::TEXT_ANALYZER)
        .value("CONTENT_MODERATOR", MLModelType::CONTENT_MODERATOR)
        .value("DEEPFAKE_DETECTOR", MLModelType::DEEPFAKE_DETECTOR)
        .export_values();

    py::class_<MLClassificationResult>(m, "MLClassificationResult")
        .def(py::init<>())
        .def_readwrite("file_name", &MLClassificationResult::fileName)
        .def_readwrite("file_path", &MLClassificationResult::filePath)
        .def_readwrite("file_size", &MLClassificationResult::fileSize)
        .def_readwrite("detected_type", &MLClassificationResult::detectedType)
        .def_readwrite("primary_category", &MLClassificationResult::primaryCategory)
        .def_readwrite("confidence", &MLClassificationResult::confidence)
        .def_readwrite("risk_score", &MLClassificationResult::riskScore)
        .def_readwrite("is_suspicious", &MLClassificationResult::isSuspicious)
        .def_readwrite("is_encrypted", &MLClassificationResult::isEncrypted)
        .def_readwrite("has_hidden_data", &MLClassificationResult::hasHiddenData);

    // Utility functions
    m.def("get_version", []() {
        return "PhoenixDRS Professional 2.0.0";
    });
    
    m.def("get_build_info", []() {
        py::dict info;
        info["version"] = "2.0.0";
        info["build_date"] = __DATE__ " " __TIME__;
        info["compiler"] = 
#ifdef _MSC_VER
            "MSVC " + std::to_string(_MSC_VER);
#elif defined(__GNUC__)
            "GCC " + std::to_string(__GNUC__) + "." + std::to_string(__GNUC_MINOR__);
#elif defined(__clang__)
            "Clang " + std::to_string(__clang_major__) + "." + std::to_string(__clang_minor__);
#else
            "Unknown";
#endif
        info["platform"] = 
#ifdef _WIN32
            "Windows";
#elif defined(__linux__)
            "Linux";
#elif defined(__APPLE__)
            "macOS";
#else
            "Unknown";
#endif
        return info;
    });

    // Initialize logging on module import
    m.def("init_logging", [](const std::string& log_file, const std::string& level) {
        ForensicLogger::instance()->initialize(QString::fromStdString(log_file));
        return true;
    }, py::arg("log_file") = "", py::arg("level") = "INFO");

    // Performance monitoring
    m.def("start_performance_monitoring", []() {
        // Implementation would start performance monitoring
        return true;
    });
    
    m.def("stop_performance_monitoring", []() {
        // Implementation would stop performance monitoring
        return true;
    });
    
    // Memory management helpers
    m.def("force_garbage_collection", []() {
        // Force C++ cleanup and Python GC
        PyGC_Collect();
        return true;
    });

    // Feature availability checks
    m.def("has_gpu_support", []() {
#ifdef ENABLE_CUDA
        return true;
#else
        return false;
#endif
    });
    
    m.def("has_opencv_support", []() {
#ifdef ENABLE_OPENCV
        return true;
#else
        return false;
#endif
    });
    
    m.def("has_openssl_support", []() {
#ifdef ENABLE_OPENSSL
        return true;
#else
        return false;
#endif
    });

    // Module cleanup on exit
    m.add_object("_cleanup", py::capsule([]() {
        PythonForensicsAPI::instance().shutdown();
    }));

#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif
}

// Helper macro for version string
#define MACRO_STRINGIFY(x) #x