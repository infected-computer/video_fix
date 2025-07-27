#pragma once
#include <vector>
#include <memory>
#include <thread>
#include <future>
#include <functional>

#ifdef _WIN32
#include <windows.h>
#include <memoryapi.h>
#else
#include <sys/mman.h>
#include <unistd.h>
#include <fcntl.h>
#endif

class MemoryMappedFile {
private:
    void* mappedData;
    size_t fileSize;
    
#ifdef _WIN32
    HANDLE fileHandle;
    HANDLE mappingHandle;
#else
    int fileDescriptor;
#endif

public:
    MemoryMappedFile(const std::string& filePath);
    ~MemoryMappedFile();
    
    const uint8_t* getData() const;
    size_t getSize() const;
    bool isValid() const;
};

class ThreadPool {
private:
    std::vector<std::thread> workers;
    std::queue<std::function<void()>> tasks;
    std::mutex queueMutex;
    std::condition_variable condition;
    bool stop;
    
public:
    ThreadPool(size_t numThreads = std::thread::hardware_concurrency());
    ~ThreadPool();
    
    template<class F, class... Args>
    auto enqueue(F&& f, Args&&... args) 
        -> std::future<typename std::result_of<F(Args...)>::type>;
        
    void waitForAll();
};

class ParallelFormatDetector {
private:
    ThreadPool threadPool;
    
public:
    ParallelFormatDetector();
    
    std::vector<std::pair<std::string, FileFormat>> 
    detectMultipleFiles(const std::vector<std::string>& filePaths) const;
    
    bool repairMultipleFiles(const std::vector<std::string>& filePaths,
                           const std::string& outputDir = "") const;
};