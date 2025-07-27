/*
 * PhoenixDRS Professional - Enterprise Memory Management and Resource Pool
 * ניהול זיכרון ברמה תעשייתית ומאגר משאבים - PhoenixDRS מקצועי
 */

#pragma once

#include "ErrorHandling.h"
#include <QtCore/QObject>
#include <QtCore/QMutex>
#include <QtCore/QReadWriteLock>
#include <QtCore/QTimer>
#include <QtCore/QThread>
#include <QtCore/QElapsedTimer>

#include <memory>
#include <atomic>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <queue>
#include <stack>
#include <functional>
#include <chrono>
#include <type_traits>

namespace PhoenixDRS {
namespace Core {

// Memory allocation strategies
enum class AllocationStrategy {
    Standard,           // Standard system allocator
    Pool,              // Memory pool allocation
    Stack,             // Stack-based allocation
    Ring,              // Ring buffer allocation
    Buddy,             // Buddy system allocation
    Slab,              // Slab allocation
    NUMA,              // NUMA-aware allocation
    Huge,              // Huge page allocation
    Lock_Free,         // Lock-free allocation
    Debug              // Debug allocation with tracking
};

// Memory alignment requirements
enum class MemoryAlignment {
    Default = 0,       // Default alignment
    Byte_1 = 1,        // 1-byte alignment
    Byte_2 = 2,        // 2-byte alignment  
    Byte_4 = 4,        // 4-byte alignment
    Byte_8 = 8,        // 8-byte alignment
    Byte_16 = 16,      // 16-byte alignment (SSE)
    Byte_32 = 32,      // 32-byte alignment (AVX)
    Byte_64 = 64,      // 64-byte alignment (cache line)
    Page_4K = 4096,    // 4KB page alignment
    Page_2M = 2097152, // 2MB huge page alignment
    Page_1G = 1073741824 // 1GB huge page alignment
};

// Memory access patterns for optimization
enum class AccessPattern {
    Sequential,        // Sequential access
    Random,           // Random access  
    Temporal,         // High temporal locality
    Spatial,          // High spatial locality
    Write_Once,       // Write once, read many
    Streaming,        // Streaming data
    Sparse,           // Sparse access
    Hot_Cold          // Mixed hot/cold data
};

// Forward declarations
class MemoryPool;
class StackAllocator;
class RingBuffer;
class ResourceTracker;
template<typename T> class ObjectPool;

/*
 * Base allocator interface
 */
class PHOENIXDRS_EXPORT IAllocator
{
public:
    virtual ~IAllocator() = default;
    
    virtual void* allocate(size_t size, MemoryAlignment alignment = MemoryAlignment::Default) = 0;
    virtual void deallocate(void* ptr) noexcept = 0;
    virtual bool owns(void* ptr) const noexcept = 0;
    virtual size_t getAllocatedSize(void* ptr) const noexcept = 0;
    virtual size_t getTotalAllocated() const noexcept = 0;
    virtual size_t getTotalFree() const noexcept = 0;
    virtual void defragment() = 0;
    virtual AllocationStrategy getStrategy() const noexcept = 0;
};

/*
 * RAII wrapper for automatic resource management
 */
template<typename T, typename Deleter = std::default_delete<T>>
class UniqueResource
{
public:
    UniqueResource() noexcept : m_resource(nullptr) {}
    
    explicit UniqueResource(T* resource) noexcept 
        : m_resource(resource) {}
    
    UniqueResource(T* resource, Deleter deleter) noexcept
        : m_resource(resource), m_deleter(std::move(deleter)) {}
    
    ~UniqueResource() noexcept {
        reset();
    }
    
    // Non-copyable
    UniqueResource(const UniqueResource&) = delete;
    UniqueResource& operator=(const UniqueResource&) = delete;
    
    // Movable
    UniqueResource(UniqueResource&& other) noexcept
        : m_resource(other.m_resource), m_deleter(std::move(other.m_deleter)) {
        other.m_resource = nullptr;
    }
    
    UniqueResource& operator=(UniqueResource&& other) noexcept {
        if (this != &other) {
            reset();
            m_resource = other.m_resource;
            m_deleter = std::move(other.m_deleter);
            other.m_resource = nullptr;
        }
        return *this;
    }
    
    // Access
    T* get() const noexcept { return m_resource; }
    T* operator->() const noexcept { return m_resource; }
    T& operator*() const noexcept { return *m_resource; }
    explicit operator bool() const noexcept { return m_resource != nullptr; }
    
    // Release ownership
    T* release() noexcept {
        T* resource = m_resource;
        m_resource = nullptr;
        return resource;
    }
    
    // Reset resource
    void reset(T* resource = nullptr) noexcept {
        if (m_resource) {
            m_deleter(m_resource);
        }
        m_resource = resource;
    }

private:
    T* m_resource;
    Deleter m_deleter;
};

/*
 * Memory pool for efficient allocation of same-sized objects
 */
class PHOENIXDRS_EXPORT MemoryPool : public IAllocator
{
public:
    MemoryPool(size_t objectSize, size_t blockSize = 1024, 
               MemoryAlignment alignment = MemoryAlignment::Default);
    ~MemoryPool() override;
    
    // IAllocator interface
    void* allocate(size_t size, MemoryAlignment alignment = MemoryAlignment::Default) override;
    void deallocate(void* ptr) noexcept override;
    bool owns(void* ptr) const noexcept override;
    size_t getAllocatedSize(void* ptr) const noexcept override;
    size_t getTotalAllocated() const noexcept override;
    size_t getTotalFree() const noexcept override;
    void defragment() override;
    AllocationStrategy getStrategy() const noexcept override { return AllocationStrategy::Pool; }
    
    // Pool-specific methods
    size_t getObjectSize() const noexcept { return m_objectSize; }
    size_t getBlockSize() const noexcept { return m_blockSize; }
    size_t getNumBlocks() const noexcept;
    size_t getNumFreeObjects() const noexcept;
    size_t getNumAllocatedObjects() const noexcept;
    
    // Statistics
    struct Statistics {
        size_t totalAllocations = 0;
        size_t totalDeallocations = 0;
        size_t currentAllocations = 0;
        size_t peakAllocations = 0;
        size_t blocksAllocated = 0;
        size_t bytesAllocated = 0;
        std::chrono::nanoseconds totalAllocationTime{0};
        std::chrono::nanoseconds averageAllocationTime{0};
    };
    
    Statistics getStatistics() const;
    void resetStatistics();

private:
    struct Block;
    struct FreeNode;
    
    void allocateNewBlock();
    Block* findBlock(void* ptr) const noexcept;
    
    const size_t m_objectSize;
    const size_t m_blockSize;
    const MemoryAlignment m_alignment;
    
    std::vector<std::unique_ptr<Block>> m_blocks;
    std::stack<FreeNode*> m_freeList;
    
    mutable QReadWriteLock m_lock;
    Statistics m_statistics;
};

/*
 * Stack allocator for LIFO memory allocation
 */
class PHOENIXDRS_EXPORT StackAllocator : public IAllocator
{
public:
    explicit StackAllocator(size_t stackSize);
    ~StackAllocator() override;
    
    // IAllocator interface
    void* allocate(size_t size, MemoryAlignment alignment = MemoryAlignment::Default) override;
    void deallocate(void* ptr) noexcept override;
    bool owns(void* ptr) const noexcept override;
    size_t getAllocatedSize(void* ptr) const noexcept override;
    size_t getTotalAllocated() const noexcept override;
    size_t getTotalFree() const noexcept override;
    void defragment() override {}
    AllocationStrategy getStrategy() const noexcept override { return AllocationStrategy::Stack; }
    
    // Stack-specific methods
    class Marker {
    public:
        explicit Marker(size_t position) : m_position(position) {}
        size_t position() const { return m_position; }
    private:
        size_t m_position;
    };
    
    Marker getMarker() const;
    void freeToMarker(const Marker& marker);
    void clear();
    
    size_t getStackSize() const noexcept { return m_stackSize; }
    size_t getUsedSize() const noexcept { return m_top; }
    size_t getFreeSize() const noexcept { return m_stackSize - m_top; }

private:
    const size_t m_stackSize;
    std::unique_ptr<uint8_t[]> m_memory;
    size_t m_top;
    
    mutable QMutex m_mutex;
};

/*
 * Object pool for recycling objects of a specific type
 */
template<typename T>
class ObjectPool
{
public:
    explicit ObjectPool(size_t initialSize = 16, size_t maxSize = 1024)
        : m_maxSize(maxSize) {
        m_pool.reserve(initialSize);
        for (size_t i = 0; i < initialSize; ++i) {
            m_pool.emplace_back(std::make_unique<T>());
        }
    }
    
    ~ObjectPool() = default;
    
    // Get object from pool
    UniqueResource<T> acquire() {
        QMutexLocker locker(&m_mutex);
        
        if (m_pool.empty()) {
            if (m_totalCreated < m_maxSize) {
                ++m_totalCreated;
                return UniqueResource<T>(new T(), [this](T* obj) { release(obj); });
            } else {
                throw PhoenixException::create(ErrorCode::ResourceExhausted,
                                             "Object pool maximum size reached");
            }
        }
        
        auto obj = m_pool.back().release();
        m_pool.pop_back();
        ++m_acquired;
        
        return UniqueResource<T>(obj, [this](T* obj) { release(obj); });
    }
    
    // Get current pool statistics
    struct Statistics {
        size_t poolSize = 0;
        size_t acquired = 0;
        size_t totalCreated = 0;
        size_t maxSize = 0;
    };
    
    Statistics getStatistics() const {
        QMutexLocker locker(&m_mutex);
        return {m_pool.size(), m_acquired, m_totalCreated, m_maxSize};
    }

private:
    void release(T* obj) {
        QMutexLocker locker(&m_mutex);
        
        // Reset object to default state if possible
        if constexpr (std::is_default_constructible_v<T>) {
            *obj = T{};
        }
        
        m_pool.emplace_back(obj);
        --m_acquired;
    }
    
    std::vector<std::unique_ptr<T>> m_pool;
    size_t m_maxSize;
    size_t m_acquired = 0;
    size_t m_totalCreated = 0;
    mutable QMutex m_mutex;
};

/*
 * Resource tracker for debugging memory leaks
 */
class PHOENIXDRS_EXPORT ResourceTracker : public QObject
{
    Q_OBJECT
    
public:
    static ResourceTracker& instance();
    
    // Resource tracking
    void trackAllocation(void* ptr, size_t size, const QString& file, int line, const QString& function);
    void trackDeallocation(void* ptr);
    void trackObjectConstruction(void* ptr, const QString& typeName, const QString& file, int line);
    void trackObjectDestruction(void* ptr);
    
    // Leak detection
    struct LeakInfo {
        void* address;
        size_t size;
        QString typeName;
        QString file;
        int line;
        QString function;
        std::chrono::steady_clock::time_point timestamp;
        QStringList stackTrace;
    };
    
    std::vector<LeakInfo> getMemoryLeaks() const;
    std::vector<LeakInfo> getObjectLeaks() const;
    
    // Statistics
    struct Statistics {
        size_t totalAllocations = 0;
        size_t totalDeallocations = 0;
        size_t currentAllocations = 0;
        size_t peakAllocations = 0;
        size_t totalBytesAllocated = 0;
        size_t totalBytesFreed = 0;  
        size_t currentBytesAllocated = 0;
        size_t peakBytesAllocated = 0;
        size_t objectsConstructed = 0;
        size_t objectsDestroyed = 0;
        size_t currentObjects = 0;
    };
    
    Statistics getStatistics() const;
    void resetStatistics();
    
    // Configuration
    void setTrackingEnabled(bool enabled);
    bool isTrackingEnabled() const;
    void setStackTraceEnabled(bool enabled);
    bool isStackTraceEnabled() const;
    
    // Reporting
    void dumpLeaks(const QString& filePath) const;
    void generateReport(const QString& filePath) const;

signals:
    void memoryLeakDetected(const LeakInfo& leak);
    void objectLeakDetected(const LeakInfo& leak);

private:
    ResourceTracker();
    ~ResourceTracker();
    
    std::unordered_map<void*, LeakInfo> m_allocations;
    std::unordered_map<void*, LeakInfo> m_objects;
    Statistics m_statistics;
    bool m_trackingEnabled;
    bool m_stackTraceEnabled;
    mutable QReadWriteLock m_lock;
};

/*
 * Memory manager coordinating all allocation strategies
 */
class PHOENIXDRS_EXPORT MemoryManager : public QObject
{
    Q_OBJECT
    
public:
    static MemoryManager& instance();
    
    // Allocator registration
    void registerAllocator(const QString& name, std::unique_ptr<IAllocator> allocator);
    void unregisterAllocator(const QString& name);
    IAllocator* getAllocator(const QString& name) const;
    IAllocator* getDefaultAllocator() const;
    void setDefaultAllocator(const QString& name);
    
    // High-level allocation interface
    void* allocate(size_t size, 
                   MemoryAlignment alignment = MemoryAlignment::Default,
                   AllocationStrategy strategy = AllocationStrategy::Standard,
                   AccessPattern pattern = AccessPattern::Random);
    
    void deallocate(void* ptr) noexcept;
    
    // Typed allocation
    template<typename T, typename... Args>
    T* construct(Args&&... args) {
        void* memory = allocate(sizeof(T), 
                               static_cast<MemoryAlignment>(alignof(T)));
        try {
            T* object = new(memory) T(std::forward<Args>(args)...);
            if (m_trackingEnabled) {
                ResourceTracker::instance().trackObjectConstruction(
                    object, typeid(T).name(), __FILE__, __LINE__);
            }
            return object;
        } catch (...) {
            deallocate(memory);
            throw;
        }
    }
    
    template<typename T>
    void destroy(T* object) noexcept {
        if (object) {
            if (m_trackingEnabled) {
                ResourceTracker::instance().trackObjectDestruction(object);
            }
            object->~T();
            deallocate(object);
        }
    }
    
    // Memory pools
    std::shared_ptr<MemoryPool> createMemoryPool(size_t objectSize, 
                                                  size_t blockSize = 1024,
                                                  MemoryAlignment alignment = MemoryAlignment::Default);
    
    std::shared_ptr<StackAllocator> createStackAllocator(size_t stackSize);
    
    template<typename T>
    std::shared_ptr<ObjectPool<T>> createObjectPool(size_t initialSize = 16, 
                                                     size_t maxSize = 1024) {
        return std::make_shared<ObjectPool<T>>(initialSize, maxSize);
    }
    
    // Memory monitoring
    struct GlobalStatistics {
        size_t totalSystemMemory = 0;
        size_t availableSystemMemory = 0;
        size_t processMemoryUsage = 0;
        size_t phoenixMemoryUsage = 0;
        size_t peakMemoryUsage = 0;
        double fragmentationRatio = 0.0;
        size_t numberOfAllocators = 0;
        size_t activeAllocations = 0;
    };
    
    GlobalStatistics getGlobalStatistics() const;
    void runGarbageCollection();
    void defragmentAll();
    
    // Configuration
    void setTrackingEnabled(bool enabled);
    bool isTrackingEnabled() const { return m_trackingEnabled; }
    
    void setLowMemoryThreshold(size_t threshold);
    size_t getLowMemoryThreshold() const { return m_lowMemoryThreshold; }
    
    // Memory pressure handling
    using LowMemoryHandler = std::function<void(size_t availableMemory)>;
    void registerLowMemoryHandler(LowMemoryHandler handler);
    
    // Debugging and profiling
    void enableAllocationProfiling(bool enable);
    void dumpAllocationProfile(const QString& filePath) const;
    
    // NUMA support
    void enableNUMAOptimization(bool enable);
    bool isNUMAOptimizationEnabled() const;
    int getCurrentNUMANode() const;

signals:
    void lowMemoryWarning(size_t availableMemory);
    void outOfMemoryError();
    void memoryLeakDetected(void* address, size_t size);

private slots:
    void checkMemoryPressure();
    void performPeriodicMaintenance();

private:
    MemoryManager();
    ~MemoryManager();
    
    void initializeSystemInfo();
    void setupPeriodicTasks();
    
    std::unordered_map<QString, std::unique_ptr<IAllocator>> m_allocators;
    QString m_defaultAllocatorName;
    
    std::vector<LowMemoryHandler> m_lowMemoryHandlers;
    bool m_trackingEnabled;
    size_t m_lowMemoryThreshold;
    bool m_profilingEnabled;
    bool m_numaOptimization;
    
    QTimer* m_memoryPressureTimer;
    QTimer* m_maintenanceTimer;
    
    mutable QReadWriteLock m_allocatorsLock;
    mutable QMutex m_handlersLock;
    
    GlobalStatistics m_globalStats;
};

// Memory allocation macros with tracking
#ifdef PHOENIX_DEBUG_MEMORY
    #define PHOENIX_NEW(type, ...) \
        PhoenixDRS::Core::MemoryManager::instance().construct<type>(__VA_ARGS__)
    
    #define PHOENIX_DELETE(ptr) \
        PhoenixDRS::Core::MemoryManager::instance().destroy(ptr)
    
    #define PHOENIX_MALLOC(size) \
        PhoenixDRS::Core::MemoryManager::instance().allocate(size)
    
    #define PHOENIX_FREE(ptr) \
        PhoenixDRS::Core::MemoryManager::instance().deallocate(ptr)
#else
    #define PHOENIX_NEW(type, ...) new type(__VA_ARGS__)
    #define PHOENIX_DELETE(ptr) delete ptr
    #define PHOENIX_MALLOC(size) std::malloc(size)
    #define PHOENIX_FREE(ptr) std::free(ptr)
#endif

// RAII helper macros
#define PHOENIX_SCOPED_ALLOCATOR(name, allocator) \
    auto name##_guard = PhoenixDRS::Core::UniqueResource<PhoenixDRS::Core::IAllocator>(&allocator, \
        [](PhoenixDRS::Core::IAllocator*){});

#define PHOENIX_SCOPED_MEMORY(ptr, size) \
    auto ptr##_guard = PhoenixDRS::Core::UniqueResource<void>(ptr, \
        [](void* p) { PhoenixDRS::Core::MemoryManager::instance().deallocate(p); });

} // namespace Core
} // namespace PhoenixDRS