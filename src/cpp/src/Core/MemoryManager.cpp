/*
 * PhoenixDRS Professional - Enterprise Memory Management Implementation
 * מימוש ניהול זיכרון ארגוני - PhoenixDRS מקצועי
 */

#include "../include/Core/MemoryManager.h"
#include "../include/Core/ErrorHandling.h"
#include "../include/ForensicLogger.h"

#include <QtCore/QThread>
#include <QtCore/QDateTime>
#include <QtCore/QTimer>
#include <QtCore/QCoreApplication>
#include <QtCore/QMutexLocker>
#include <QtCore/QJsonObject>
#include <QtCore/QJsonDocument>

#include <algorithm>
#include <chrono>
#include <cstring>

#ifdef _WIN32
#include <windows.h>
#include <psapi.h>
#else
#include <sys/resource.h>
#include <unistd.h>
#endif

namespace PhoenixDRS {
namespace Core {

// MemoryPool Implementation
class MemoryPoolPrivate
{
public:
    MemoryPoolPrivate(size_t blockSize, size_t poolSize)
        : blockSize(blockSize)
        , poolSize(poolSize)
        , totalBlocks(poolSize / blockSize)
        , allocatedBlocks(0)
        , peakAllocatedBlocks(0)
    {
        // Align block size to pointer boundary
        this->blockSize = alignSize(blockSize);
        
        // Allocate the memory pool
        pool = static_cast<char*>(std::aligned_alloc(alignof(std::max_align_t), this->poolSize));
        if (!pool) {
            throw PhoenixException(ErrorCode::OutOfMemory, 
                                 QString("Failed to allocate memory pool of size %1 bytes").arg(poolSize),
                                 "MemoryPool");
        }
        
        // Initialize free list
        initializeFreeList();
        
        // Start memory monitoring
        setupMonitoring();
    }
    
    ~MemoryPoolPrivate()
    {
        if (monitoring) {
            monitoring->stop();
        }
        
        if (pool) {
            std::free(pool);
        }
    }
    
    void* allocate()
    {
        QMutexLocker locker(&mutex);
        
        if (freeList.empty()) {
            // Try to defragment first
            if (!defragmentPool()) {
                throw PhoenixException(ErrorCode::OutOfMemory,
                                     "Memory pool exhausted",
                                     "MemoryPool::allocate");
            }
        }
        
        void* block = freeList.back();
        freeList.pop_back();
        
        allocatedBlocks++;
        peakAllocatedBlocks = std::max(peakAllocatedBlocks, allocatedBlocks);
        
        // Track allocation
        AllocationInfo info;
        info.address = block;
        info.size = blockSize;
        info.timestamp = std::chrono::steady_clock::now();
        info.threadId = reinterpret_cast<qint64>(QThread::currentThreadId());
        
        allocations[block] = info;
        
        return block;
    }
    
    void deallocate(void* ptr)
    {
        if (!ptr || !isValidPointer(ptr)) {
            return;
        }
        
        QMutexLocker locker(&mutex);
        
        auto it = allocations.find(ptr);
        if (it != allocations.end()) {
            // Clear the memory for security
            std::memset(ptr, 0, blockSize);
            
            freeList.push_back(ptr);
            allocatedBlocks--;
            
            allocations.erase(it);
        }
    }
    
    MemoryPool::Statistics getStatistics() const
    {
        QMutexLocker locker(&mutex);
        
        MemoryPool::Statistics stats;
        stats.blockSize = blockSize;
        stats.totalBlocks = totalBlocks;
        stats.allocatedBlocks = allocatedBlocks;
        stats.freeBlocks = freeList.size();
        stats.peakAllocatedBlocks = peakAllocatedBlocks;
        stats.fragmentationRatio = calculateFragmentation();
        stats.totalAllocations = totalAllocations;
        stats.totalDeallocations = totalDeallocations;
        
        return stats;
    }
    
private:
    size_t alignSize(size_t size)
    {
        const size_t alignment = alignof(std::max_align_t);
        return (size + alignment - 1) & ~(alignment - 1);
    }
    
    void initializeFreeList()
    {
        for (size_t i = 0; i < totalBlocks; ++i) {
            freeList.push_back(pool + i * blockSize);
        }
    }
    
    bool isValidPointer(void* ptr)
    {
        char* charPtr = static_cast<char*>(ptr);
        return charPtr >= pool && charPtr < pool + poolSize;
    }
    
    bool defragmentPool()
    {
        // Simple defragmentation - in a real implementation this would be more sophisticated
        return false; // For now, just return failure
    }
    
    double calculateFragmentation() const
    {
        if (totalBlocks == 0) return 0.0;
        
        // Simple fragmentation calculation
        size_t usedBlocks = allocatedBlocks;
        size_t totalUsableBlocks = totalBlocks;
        
        if (totalUsableBlocks == 0) return 0.0;
        
        return static_cast<double>(usedBlocks) / totalUsableBlocks;
    }
    
    void setupMonitoring()
    {
        monitoring = std::make_unique<QTimer>();
        QObject::connect(monitoring.get(), &QTimer::timeout, [this]() {
            monitorMemoryUsage();
        });
        monitoring->start(5000); // Monitor every 5 seconds
    }
    
    void monitorMemoryUsage()
    {
        QMutexLocker locker(&mutex);
        
        double fragmentationRatio = calculateFragmentation();
        if (fragmentationRatio > 0.9) { // 90% fragmentation threshold
            if (ForensicLogger::instance()) {
                ForensicLogger::instance()->logWarning(
                    QString("High memory fragmentation detected: %1%")
                    .arg(fragmentationRatio * 100, 0, 'f', 1));
            }
        }
        
        if (allocatedBlocks > totalBlocks * 0.95) { // 95% usage threshold
            if (ForensicLogger::instance()) {
                ForensicLogger::instance()->logWarning(
                    QString("Memory pool near exhaustion: %1/%2 blocks used")
                    .arg(allocatedBlocks).arg(totalBlocks));
            }
        }
    }
    
    // Memory pool data
    char* pool = nullptr;
    size_t blockSize;
    size_t poolSize;
    size_t totalBlocks;
    size_t allocatedBlocks;
    size_t peakAllocatedBlocks;
    
    std::vector<void*> freeList;
    
    // Allocation tracking
    struct AllocationInfo {
        void* address;
        size_t size;
        std::chrono::steady_clock::time_point timestamp;
        qint64 threadId;
    };
    
    std::unordered_map<void*, AllocationInfo> allocations;
    
    // Statistics
    mutable size_t totalAllocations = 0;
    mutable size_t totalDeallocations = 0;
    
    // Thread safety
    mutable QMutex mutex;
    
    // Monitoring
    std::unique_ptr<QTimer> monitoring;
};

MemoryPool::MemoryPool(size_t blockSize, size_t poolSize)
    : d(std::make_unique<MemoryPoolPrivate>(blockSize, poolSize))
{
}

MemoryPool::~MemoryPool() = default;

void* MemoryPool::allocate()
{
    return d->allocate();
}

void MemoryPool::deallocate(void* ptr)
{
    d->deallocate(ptr);
}

MemoryPool::Statistics MemoryPool::getStatistics() const
{
    return d->getStatistics();
}

// SmartBuffer Implementation
SmartBuffer::SmartBuffer(size_t size, MemoryPool* pool)
    : m_size(size)
    , m_pool(pool)
    , m_data(nullptr)
    , m_isPoolAllocated(false)
{
    allocate();
}

SmartBuffer::SmartBuffer(SmartBuffer&& other) noexcept
    : m_size(other.m_size)
    , m_pool(other.m_pool)
    , m_data(other.m_data)
    , m_isPoolAllocated(other.m_isPoolAllocated)
{
    other.m_data = nullptr;
    other.m_size = 0;
    other.m_isPoolAllocated = false;
}

SmartBuffer& SmartBuffer::operator=(SmartBuffer&& other) noexcept
{
    if (this != &other) {
        deallocate();
        
        m_size = other.m_size;
        m_pool = other.m_pool;
        m_data = other.m_data;
        m_isPoolAllocated = other.m_isPoolAllocated;
        
        other.m_data = nullptr;
        other.m_size = 0;
        other.m_isPoolAllocated = false;
    }
    return *this;
}

SmartBuffer::~SmartBuffer()
{
    deallocate();
}

void SmartBuffer::resize(size_t newSize)
{
    if (newSize == m_size) {
        return;
    }
    
    if (newSize == 0) {
        deallocate();
        return;
    }
    
    // For simplicity, always reallocate
    deallocate();
    m_size = newSize;
    allocate();
}

void SmartBuffer::clear()
{
    if (m_data) {
        std::memset(m_data, 0, m_size);
    }
}

void SmartBuffer::allocate()
{
    if (m_size == 0) {
        return;
    }
    
    if (m_pool) {
        try {
            m_data = m_pool->allocate();
            m_isPoolAllocated = true;
            return;
        } catch (const PhoenixException&) {
            // Pool allocation failed, fall back to system allocation
        }
    }
    
    m_data = std::malloc(m_size);
    if (!m_data) {
        throw PhoenixException(ErrorCode::OutOfMemory,
                             QString("Failed to allocate %1 bytes").arg(m_size),
                             "SmartBuffer");
    }
    m_isPoolAllocated = false;
}

void SmartBuffer::deallocate()
{
    if (m_data) {
        if (m_isPoolAllocated && m_pool) {
            m_pool->deallocate(m_data);
        } else {
            std::free(m_data);
        }
        m_data = nullptr;
    }
    m_size = 0;
    m_isPoolAllocated = false;
}

// MemoryManager Implementation
class MemoryManagerPrivate
{
public:
    MemoryManagerPrivate()
        : totalSystemMemory(0)
        , availableSystemMemory(0)
        , processMemoryUsage(0)
        , peakProcessMemoryUsage(0)
        , memoryPressureThreshold(0.85)
        , autoCleanupEnabled(true)
        , monitoringEnabled(true)
    {
        setupSystemInfo();
        setupMonitoring();
        createDefaultPools();
    }
    
    ~MemoryManagerPrivate()
    {
        if (monitoringTimer) {
            monitoringTimer->stop();
        }
        cleanupTimer.reset();
    }
    
    void setupSystemInfo()
    {
#ifdef _WIN32
        MEMORYSTATUSEX memStatus;
        memStatus.dwLength = sizeof(memStatus);
        if (GlobalMemoryStatusEx(&memStatus)) {
            totalSystemMemory = memStatus.ullTotalPhys;
            availableSystemMemory = memStatus.ullAvailPhys;
        }
#else
        long pages = sysconf(_SC_PHYS_PAGES);
        long pageSize = sysconf(_SC_PAGE_SIZE);
        if (pages > 0 && pageSize > 0) {
            totalSystemMemory = static_cast<size_t>(pages) * pageSize;
        }
        
        long availablePages = sysconf(_SC_AVPHYS_PAGES);
        if (availablePages > 0) {
            availableSystemMemory = static_cast<size_t>(availablePages) * pageSize;
        }
#endif
    }
    
    void setupMonitoring()
    {
        if (!monitoringEnabled) return;
        
        monitoringTimer = std::make_unique<QTimer>();
        QObject::connect(monitoringTimer.get(), &QTimer::timeout, [this]() {
            updateMemoryStatistics();
            checkMemoryPressure();
        });
        monitoringTimer->start(1000); // Monitor every second
        
        if (autoCleanupEnabled) {
            cleanupTimer = std::make_unique<QTimer>();
            QObject::connect(cleanupTimer.get(), &QTimer::timeout, [this]() {
                performAutoCleanup();
            });
            cleanupTimer->start(30000); // Cleanup every 30 seconds
        }
    }
    
    void createDefaultPools()
    {
        // Create pools for common allocation sizes
        memoryPools[64] = std::make_unique<MemoryPool>(64, 1024 * 1024); // 1MB pool for 64-byte blocks
        memoryPools[256] = std::make_unique<MemoryPool>(256, 2 * 1024 * 1024); // 2MB pool for 256-byte blocks
        memoryPools[1024] = std::make_unique<MemoryPool>(1024, 4 * 1024 * 1024); // 4MB pool for 1KB blocks
        memoryPools[4096] = std::make_unique<MemoryPool>(4096, 8 * 1024 * 1024); // 8MB pool for 4KB blocks
    }
    
    void updateMemoryStatistics()
    {
#ifdef _WIN32
        PROCESS_MEMORY_COUNTERS_EX pmc;
        if (GetProcessMemoryInfo(GetCurrentProcess(), 
                               reinterpret_cast<PROCESS_MEMORY_COUNTERS*>(&pmc), 
                               sizeof(pmc))) {
            processMemoryUsage = pmc.WorkingSetSize;
            peakProcessMemoryUsage = std::max(peakProcessMemoryUsage, processMemoryUsage);
        }
        
        MEMORYSTATUSEX memStatus;
        memStatus.dwLength = sizeof(memStatus);
        if (GlobalMemoryStatusEx(&memStatus)) {
            availableSystemMemory = memStatus.ullAvailPhys;
        }
#else
        struct rusage usage;
        if (getrusage(RUSAGE_SELF, &usage) == 0) {
            // On Linux, ru_maxrss is in KB
            processMemoryUsage = usage.ru_maxrss * 1024;
            peakProcessMemoryUsage = std::max(peakProcessMemoryUsage, processMemoryUsage);
        }
#endif
    }
    
    void checkMemoryPressure()
    {
        double memoryPressure = static_cast<double>(processMemoryUsage) / totalSystemMemory;
        
        if (memoryPressure > memoryPressureThreshold) {
            if (ForensicLogger::instance()) {
                ForensicLogger::instance()->logWarning(
                    QString("High memory pressure detected: %1% of system memory in use")
                    .arg(memoryPressure * 100, 0, 'f', 1));
            }
            
            if (autoCleanupEnabled) {
                performAutoCleanup();
            }
        }
    }
    
    void performAutoCleanup()
    {
        // Force garbage collection in all pools
        for (auto& [size, pool] : memoryPools) {
            auto stats = pool->getStatistics();
            if (stats.fragmentationRatio > 0.7) { // 70% fragmentation threshold
                // In a real implementation, we might defragment or recreate the pool
            }
        }
        
        // Clean up expired allocations
        cleanupExpiredAllocations();
    }
    
    void cleanupExpiredAllocations()
    {
        auto now = std::chrono::steady_clock::now();
        QMutexLocker locker(&allocationMutex);
        
        for (auto it = allocations.begin(); it != allocations.end();) {
            auto elapsed = std::chrono::duration_cast<std::chrono::minutes>(now - it->second.timestamp);
            if (elapsed.count() > 60) { // Clean up allocations older than 1 hour
                it = allocations.erase(it);
            } else {
                ++it;
            }
        }
    }
    
    // System memory info
    size_t totalSystemMemory;
    size_t availableSystemMemory;
    size_t processMemoryUsage;
    size_t peakProcessMemoryUsage;
    
    // Configuration
    double memoryPressureThreshold;
    bool autoCleanupEnabled;
    bool monitoringEnabled;
    
    // Memory pools
    std::unordered_map<size_t, std::unique_ptr<MemoryPool>> memoryPools;
    
    // Allocation tracking
    struct AllocationRecord {
        size_t size;
        std::chrono::steady_clock::time_point timestamp;
        QString context;
    };
    
    std::unordered_map<void*, AllocationRecord> allocations;
    QMutex allocationMutex;
    
    // Monitoring
    std::unique_ptr<QTimer> monitoringTimer;
    std::unique_ptr<QTimer> cleanupTimer;
};

MemoryManager& MemoryManager::instance()
{
    static MemoryManager instance;
    return instance;
}

MemoryManager::MemoryManager()
    : d(std::make_unique<MemoryManagerPrivate>())
{
}

MemoryManager::~MemoryManager() = default;

MemoryPool* MemoryManager::getPool(size_t blockSize)
{
    // Find the best fitting pool
    auto it = d->memoryPools.lower_bound(blockSize);
    if (it != d->memoryPools.end()) {
        return it->second.get();
    }
    return nullptr;
}

std::unique_ptr<SmartBuffer> MemoryManager::allocateBuffer(size_t size, const QString& context)
{
    MemoryPool* pool = getPool(size);
    auto buffer = std::make_unique<SmartBuffer>(size, pool);
    
    // Track the allocation
    QMutexLocker locker(&d->allocationMutex);
    MemoryManagerPrivate::AllocationRecord record;
    record.size = size;
    record.timestamp = std::chrono::steady_clock::now();
    record.context = context;
    d->allocations[buffer->data()] = record;
    
    return buffer;
}

MemoryManager::SystemInfo MemoryManager::getSystemInfo() const
{
    SystemInfo info;
    info.totalSystemMemory = d->totalSystemMemory;
    info.availableSystemMemory = d->availableSystemMemory;
    info.processMemoryUsage = d->processMemoryUsage;
    info.peakProcessMemoryUsage = d->peakProcessMemoryUsage;
    
    return info;
}

QJsonObject MemoryManager::getDetailedStatistics() const
{
    QJsonObject stats;
    
    // System info
    QJsonObject systemInfo;
    systemInfo["total_system_memory"] = static_cast<qint64>(d->totalSystemMemory);
    systemInfo["available_system_memory"] = static_cast<qint64>(d->availableSystemMemory);
    systemInfo["process_memory_usage"] = static_cast<qint64>(d->processMemoryUsage);
    systemInfo["peak_process_memory_usage"] = static_cast<qint64>(d->peakProcessMemoryUsage);
    stats["system"] = systemInfo;
    
    // Pool statistics
    QJsonObject poolStats;
    for (const auto& [size, pool] : d->memoryPools) {
        auto poolStat = pool->getStatistics();
        QJsonObject poolInfo;
        poolInfo["block_size"] = static_cast<qint64>(poolStat.blockSize);
        poolInfo["total_blocks"] = static_cast<qint64>(poolStat.totalBlocks);
        poolInfo["allocated_blocks"] = static_cast<qint64>(poolStat.allocatedBlocks);
        poolInfo["free_blocks"] = static_cast<qint64>(poolStat.freeBlocks);
        poolInfo["peak_allocated_blocks"] = static_cast<qint64>(poolStat.peakAllocatedBlocks);
        poolInfo["fragmentation_ratio"] = poolStat.fragmentationRatio;
        
        poolStats[QString::number(size)] = poolInfo;
    }
    stats["pools"] = poolStats;
    
    // Active allocations
    QMutexLocker locker(&d->allocationMutex);
    stats["active_allocations"] = static_cast<qint64>(d->allocations.size());
    
    return stats;
}

void MemoryManager::setMemoryPressureThreshold(double threshold)
{
    d->memoryPressureThreshold = std::clamp(threshold, 0.0, 1.0);
}

void MemoryManager::setAutoCleanupEnabled(bool enabled)
{
    d->autoCleanupEnabled = enabled;
    if (enabled && !d->cleanupTimer) {
        d->cleanupTimer = std::make_unique<QTimer>();
        QObject::connect(d->cleanupTimer.get(), &QTimer::timeout, [this]() {
            d->performAutoCleanup();
        });
        d->cleanupTimer->start(30000);
    } else if (!enabled && d->cleanupTimer) {
        d->cleanupTimer->stop();
        d->cleanupTimer.reset();
    }
}

void MemoryManager::forceCleanup()
{
    d->performAutoCleanup();
}

} // namespace Core
} // namespace PhoenixDRS