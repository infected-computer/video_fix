#include "performance_optimizer.h"
#include "file_format_detector.h"
#include <iostream>
#include <filesystem>
#include <queue>
#include <condition_variable>

// Memory Mapped File Implementation
MemoryMappedFile::MemoryMappedFile(const std::string& filePath) 
    : mappedData(nullptr), fileSize(0) {
    
#ifdef _WIN32
    fileHandle = CreateFileA(filePath.c_str(), GENERIC_READ, FILE_SHARE_READ, 
                            nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    
    if (fileHandle == INVALID_HANDLE_VALUE) {
        return;
    }
    
    LARGE_INTEGER size;
    if (!GetFileSizeEx(fileHandle, &size)) {
        CloseHandle(fileHandle);
        fileHandle = INVALID_HANDLE_VALUE;
        return;
    }
    
    fileSize = static_cast<size_t>(size.QuadPart);
    
    mappingHandle = CreateFileMappingA(fileHandle, nullptr, PAGE_READONLY, 0, 0, nullptr);
    if (mappingHandle == nullptr) {
        CloseHandle(fileHandle);
        fileHandle = INVALID_HANDLE_VALUE;
        return;
    }
    
    mappedData = MapViewOfFile(mappingHandle, FILE_MAP_READ, 0, 0, 0);
    if (mappedData == nullptr) {
        CloseHandle(mappingHandle);
        CloseHandle(fileHandle);
        fileHandle = INVALID_HANDLE_VALUE;
        mappingHandle = nullptr;
    }
    
#else
    fileDescriptor = open(filePath.c_str(), O_RDONLY);
    if (fileDescriptor == -1) {
        return;
    }
    
    struct stat sb;
    if (fstat(fileDescriptor, &sb) == -1) {
        close(fileDescriptor);
        fileDescriptor = -1;
        return;
    }
    
    fileSize = sb.st_size;
    
    mappedData = mmap(nullptr, fileSize, PROT_READ, MAP_PRIVATE, fileDescriptor, 0);
    if (mappedData == MAP_FAILED) {
        close(fileDescriptor);
        fileDescriptor = -1;
        mappedData = nullptr;
    }
#endif
}

MemoryMappedFile::~MemoryMappedFile() {
#ifdef _WIN32
    if (mappedData) {
        UnmapViewOfFile(mappedData);
    }
    if (mappingHandle) {
        CloseHandle(mappingHandle);
    }
    if (fileHandle != INVALID_HANDLE_VALUE) {
        CloseHandle(fileHandle);
    }
#else
    if (mappedData && mappedData != MAP_FAILED) {
        munmap(mappedData, fileSize);
    }
    if (fileDescriptor != -1) {
        close(fileDescriptor);
    }
#endif
}

const uint8_t* MemoryMappedFile::getData() const {
    return static_cast<const uint8_t*>(mappedData);
}

size_t MemoryMappedFile::getSize() const {
    return fileSize;
}

bool MemoryMappedFile::isValid() const {
    return mappedData != nullptr;
}

// Thread Pool Implementation
ThreadPool::ThreadPool(size_t numThreads) : stop(false) {
    for (size_t i = 0; i < numThreads; ++i) {
        workers.emplace_back([this] {
            for (;;) {
                std::function<void()> task;
                
                {
                    std::unique_lock<std::mutex> lock(this->queueMutex);
                    this->condition.wait(lock, [this] { return this->stop || !this->tasks.empty(); });
                    
                    if (this->stop && this->tasks.empty()) {
                        return;
                    }
                    
                    task = std::move(this->tasks.front());
                    this->tasks.pop();
                }
                
                task();
            }
        });
    }
}

ThreadPool::~ThreadPool() {
    {
        std::unique_lock<std::mutex> lock(queueMutex);
        stop = true;
    }
    
    condition.notify_all();
    
    for (std::thread &worker : workers) {
        worker.join();
    }
}

template<class F, class... Args>
auto ThreadPool::enqueue(F&& f, Args&&... args) 
    -> std::future<typename std::result_of<F(Args...)>::type> {
    
    using return_type = typename std::result_of<F(Args...)>::type;
    
    auto task = std::make_shared<std::packaged_task<return_type()>>(
        std::bind(std::forward<F>(f), std::forward<Args>(args)...)
    );
    
    std::future<return_type> res = task->get_future();
    
    {
        std::unique_lock<std::mutex> lock(queueMutex);
        
        if (stop) {
            throw std::runtime_error("enqueue on stopped ThreadPool");
        }
        
        tasks.emplace([task](){ (*task)(); });
    }
    
    condition.notify_one();
    return res;
}

void ThreadPool::waitForAll() {
    std::unique_lock<std::mutex> lock(queueMutex);
    condition.wait(lock, [this] { return tasks.empty(); });
}

// Parallel Format Detector Implementation
ParallelFormatDetector::ParallelFormatDetector() : threadPool() {}

std::vector<std::pair<std::string, FileFormat>> 
ParallelFormatDetector::detectMultipleFiles(const std::vector<std::string>& filePaths) const {
    
    std::vector<std::future<std::pair<std::string, FileFormat>>> futures;
    FileFormatDetector detector;
    
    for (const auto& filePath : filePaths) {
        auto future = threadPool.enqueue([&detector, filePath]() {
            MemoryMappedFile mmFile(filePath);
            FileFormat format = FileFormat::UNKNOWN;
            
            if (mmFile.isValid()) {
                std::vector<uint8_t> data(mmFile.getData(), 
                                        mmFile.getData() + std::min(mmFile.getSize(), static_cast<size_t>(1024)));
                format = detector.detectFormat(data);
            }
            
            return std::make_pair(filePath, format);
        });
        
        futures.push_back(std::move(future));
    }
    
    std::vector<std::pair<std::string, FileFormat>> results;
    for (auto& future : futures) {
        results.push_back(future.get());
    }
    
    return results;
}

bool ParallelFormatDetector::repairMultipleFiles(const std::vector<std::string>& filePaths,
                                                const std::string& outputDir) const {
    
    std::vector<std::future<bool>> futures;
    FileFormatDetector detector;
    
    for (const auto& filePath : filePaths) {
        auto future = threadPool.enqueue([&detector, filePath, outputDir]() {
            std::string outputPath;
            if (!outputDir.empty()) {
                std::filesystem::path inPath(filePath);
                std::filesystem::path outPath = std::filesystem::path(outputDir) / (inPath.filename().string() + "_repaired");
                outputPath = outPath.string();
            }
            
            return detector.repairFile(filePath, outputPath);
        });
        
        futures.push_back(std::move(future));
    }
    
    bool allSuccess = true;
    for (auto& future : futures) {
        if (!future.get()) {
            allSuccess = false;
        }
    }
    
    return allSuccess;
}