#ifndef HARDWARE_ACCELERATOR_H
#define HARDWARE_ACCELERATOR_H

#include <QObject>
#include <QMutex>
#include <QVariant>
#include <QThread>
#include <memory>
#include <vector>

/**
 * @brief Hardware acceleration types supported by the system
 */
enum class AccelerationType {
    CPU_SIMD,           // SIMD instructions (SSE, AVX, NEON)
    GPU_CUDA,           // NVIDIA CUDA
    GPU_OPENCL,         // OpenCL (cross-platform)
    GPU_VULKAN,         // Vulkan compute
    GPU_METAL,          // Apple Metal (macOS/iOS)
    GPU_DIRECTCOMPUTE,  // DirectCompute (Windows)
    FPGA,               // FPGA acceleration
    ASIC,               // Application-specific integrated circuits
    TPU,                // Tensor Processing Units
    QUANTUM,            // Quantum computing simulators
    CUSTOM              // Custom hardware
};

/**
 * @brief Computation types that can be accelerated
 */
enum class ComputationType {
    MATRIX_OPERATIONS,      // Linear algebra operations
    SIGNAL_PROCESSING,      // DSP operations
    IMAGE_PROCESSING,       // Image manipulation
    VIDEO_PROCESSING,       // Video encoding/decoding
    CRYPTOGRAPHIC,          // Cryptographic operations
    MACHINE_LEARNING,       // AI/ML computations
    SEARCH_ALGORITHMS,      // Pattern matching, sorting
    COMPRESSION,            // Data compression/decompression
    HASH_COMPUTATION,       // Hash function calculations
    FOURIER_TRANSFORM,      // FFT operations
    CUSTOM_KERNEL          // User-defined kernels
};

/**
 * @brief Hardware device information
 */
struct HardwareDevice {
    QString id;
    QString name;
    AccelerationType type;
    QString vendor;
    QString version;
    qint64 memoryMB;
    int computeUnits;
    double clockSpeedMHz;
    QStringList supportedFeatures;
    QVariantMap capabilities;
    bool isAvailable;
    bool isInitialized;
    double utilization;
    
    HardwareDevice() : memoryMB(0), computeUnits(0), clockSpeedMHz(0.0), 
                      isAvailable(false), isInitialized(false), utilization(0.0) {}
};

/**
 * @brief Acceleration task definition
 */
struct AccelerationTask {
    QString id;
    ComputationType type;
    AccelerationType preferredAcceleration;
    QByteArray inputData;
    QVariantMap parameters;
    std::function<void(const QByteArray&)> callback;
    std::function<void(const QString&)> errorCallback;
    int priority;
    QDateTime submittedAt;
    QDateTime startedAt;
    QDateTime completedAt;
    QString assignedDevice;
    
    AccelerationTask() : priority(0), submittedAt(QDateTime::currentDateTime()) {}
};

/**
 * @brief Performance benchmark results
 */
struct BenchmarkResult {
    AccelerationType type;
    ComputationType computation;
    QString deviceId;
    double executionTimeMs;
    double throughputMBps;
    double powerWatts;
    double efficiency; // operations per watt
    QVariantMap detailedMetrics;
};

/**
 * @brief CUDA acceleration manager
 */
class CUDAAccelerator : public QObject {
    Q_OBJECT
    
public:
    explicit CUDAAccelerator(QObject* parent = nullptr);
    ~CUDAAccelerator();
    
    // Device management
    bool initialize();
    void shutdown();
    QList<HardwareDevice> getAvailableDevices() const;
    bool selectDevice(const QString& deviceId);
    QString getCurrentDevice() const;
    
    // Memory management
    bool allocateDeviceMemory(const QString& bufferId, qint64 sizeBytes);
    bool copyToDevice(const QString& bufferId, const QByteArray& data);
    QByteArray copyFromDevice(const QString& bufferId, qint64 sizeBytes = -1);
    void freeDeviceMemory(const QString& bufferId);
    qint64 getAvailableMemory() const;
    
    // Kernel execution
    bool loadKernel(const QString& kernelName, const QString& source);
    bool loadKernelFromFile(const QString& kernelName, const QString& filePath);
    QByteArray executeKernel(const QString& kernelName, const QVariantMap& parameters);
    void executeKernelAsync(const QString& kernelName, const QVariantMap& parameters,
                           std::function<void(const QByteArray&)> callback);
    
    // Built-in operations
    QByteArray matrixMultiply(const QByteArray& matrixA, const QByteArray& matrixB, 
                             int rowsA, int colsA, int colsB);
    QByteArray parallelSort(const QByteArray& data, const QString& dataType);
    QByteArray parallelSearch(const QByteArray& data, const QByteArray& pattern);
    QByteArray imageFilter(const QByteArray& imageData, const QString& filterType,
                          int width, int height, const QVariantMap& parameters = {});
    
    // Performance monitoring
    double getDeviceUtilization() const;
    QVariantMap getPerformanceMetrics() const;
    
signals:
    void deviceInitialized(const QString& deviceId);
    void kernelExecutionCompleted(const QString& kernelName, bool success);
    void memoryAllocationChanged(qint64 totalAllocated, qint64 available);
    void cudaError(const QString& error);
    
private:
    struct CUDAContext;
    std::unique_ptr<CUDAContext> m_context;
    
    void detectDevices();
    bool initializeDevice(int deviceIndex);
    void cleanupDevice();
};

/**
 * @brief OpenCL acceleration manager
 */
class OpenCLAccelerator : public QObject {
    Q_OBJECT
    
public:
    explicit OpenCLAccelerator(QObject* parent = nullptr);
    ~OpenCLAccelerator();
    
    // Platform and device management
    bool initialize();
    void shutdown();
    QStringList getAvailablePlatforms() const;
    QList<HardwareDevice> getDevicesForPlatform(const QString& platform) const;
    bool selectDevice(const QString& platformId, const QString& deviceId);
    
    // Context and queue management
    bool createContext();
    bool createCommandQueue();
    void synchronize();
    
    // Buffer management
    bool createBuffer(const QString& bufferId, qint64 sizeBytes, bool readOnly = false);
    bool writeBuffer(const QString& bufferId, const QByteArray& data);
    QByteArray readBuffer(const QString& bufferId, qint64 sizeBytes = -1);
    void releaseBuffer(const QString& bufferId);
    
    // Kernel management
    bool loadProgram(const QString& programName, const QString& source);
    bool buildProgram(const QString& programName, const QString& options = QString());
    bool createKernel(const QString& kernelName, const QString& programName);
    
    // Kernel execution
    QByteArray executeKernel(const QString& kernelName, const QVariantList& arguments,
                            const QList<qint64>& globalWorkSize,
                            const QList<qint64>& localWorkSize = {});
    void executeKernelAsync(const QString& kernelName, const QVariantList& arguments,
                           const QList<qint64>& globalWorkSize,
                           const QList<qint64>& localWorkSize,
                           std::function<void(const QByteArray&)> callback);
    
    // Built-in operations
    QByteArray vectorAdd(const QByteArray& vectorA, const QByteArray& vectorB);
    QByteArray matrixTranspose(const QByteArray& matrix, int rows, int cols);
    QByteArray convolution2D(const QByteArray& image, const QByteArray& kernel,
                            int imageWidth, int imageHeight, int kernelSize);
    
    // Performance monitoring
    QVariantMap getDeviceInfo() const;
    double getKernelExecutionTime(const QString& kernelName) const;
    
signals:
    void platformsDetected(const QStringList& platforms);
    void deviceSelected(const QString& platformId, const QString& deviceId);
    void programBuilt(const QString& programName, bool success);
    void kernelCreated(const QString& kernelName, bool success);
    void openclError(const QString& error);
    
private:
    struct OpenCLContext;
    std::unique_ptr<OpenCLContext> m_context;
    
    void detectPlatformsAndDevices();
    bool setupContext();
    void cleanupResources();
};

/**
 * @brief SIMD instruction manager for CPU acceleration
 */
class SIMDAccelerator : public QObject {
    Q_OBJECT
    
public:
    enum class SIMDType {
        SSE2,       // 128-bit SIMD (x86)
        SSE3,       // SSE3 extensions
        SSE4_1,     // SSE 4.1
        SSE4_2,     // SSE 4.2
        AVX,        // 256-bit Advanced Vector Extensions
        AVX2,       // AVX2 with integer operations
        AVX512,     // 512-bit AVX
        NEON,       // ARM NEON
        ALTIVEC,    // PowerPC AltiVec
        MSA         // MIPS SIMD Architecture
    };
    
    explicit SIMDAccelerator(QObject* parent = nullptr);
    ~SIMDAccelerator();
    
    // SIMD capability detection
    bool initialize();
    QList<SIMDType> getSupportedInstructions() const;
    bool isInstructionSupported(SIMDType type) const;
    QString getCPUInfo() const;
    
    // Vector operations
    QByteArray vectorAdd(const QByteArray& vectorA, const QByteArray& vectorB, 
                        const QString& dataType = "float32");
    QByteArray vectorMultiply(const QByteArray& vectorA, const QByteArray& vectorB,
                             const QString& dataType = "float32");
    QByteArray vectorDotProduct(const QByteArray& vectorA, const QByteArray& vectorB,
                               const QString& dataType = "float32");
    
    // Array operations
    QByteArray arraySum(const QByteArray& array, const QString& dataType = "float32");
    QByteArray arrayMinMax(const QByteArray& array, const QString& dataType = "float32");
    QByteArray arraySort(const QByteArray& array, const QString& dataType = "float32");
    
    // String/pattern operations
    QList<qint64> parallelPatternSearch(const QByteArray& data, const QByteArray& pattern);
    QByteArray parallelStringCompare(const QByteArray& stringA, const QByteArray& stringB);
    QByteArray parallelStringTransform(const QByteArray& input, const QString& transformation);
    
    // Image processing
    QByteArray imageBlur(const QByteArray& imageData, int width, int height, double sigma);
    QByteArray imageConvolve(const QByteArray& imageData, const QByteArray& kernel,
                            int width, int height, int kernelSize);
    QByteArray imageResize(const QByteArray& imageData, int currentWidth, int currentHeight,
                          int newWidth, int newHeight);
    
    // Performance monitoring
    double getInstructionThroughput(SIMDType type) const;
    QVariantMap getBenchmarkResults() const;
    void runBenchmarks();
    
signals:
    void simdInitialized(const QList<SIMDType>& supportedTypes);
    void benchmarkCompleted(SIMDType type, const QVariantMap& results);
    void simdError(const QString& error);
    
private:
    struct SIMDContext;
    std::unique_ptr<SIMDContext> m_context;
    
    void detectCPUFeatures();
    void initializeInstructionSets();
    QByteArray executeOperation(const QString& operation, const QVariantList& operands);
};

/**
 * @brief Main hardware accelerator managing all acceleration types
 */
class HardwareAccelerator : public QObject {
    Q_OBJECT
    
public:
    explicit HardwareAccelerator(QObject* parent = nullptr);
    ~HardwareAccelerator();
    
    // Initialization and discovery
    bool initialize();
    void shutdown();
    QList<HardwareDevice> getAllDevices() const;
    QList<HardwareDevice> getDevicesByType(AccelerationType type) const;
    bool isAccelerationAvailable(AccelerationType type) const;
    
    // Device selection and management
    bool selectBestDevice(ComputationType computation);
    bool selectDevice(const QString& deviceId);
    QString getCurrentDevice() const;
    HardwareDevice getDeviceInfo(const QString& deviceId) const;
    
    // Task submission and execution
    QString submitTask(const AccelerationTask& task);
    bool cancelTask(const QString& taskId);
    QVariantMap getTaskStatus(const QString& taskId) const;
    QStringList getActiveTasks() const;
    
    // Direct acceleration operations
    QByteArray accelerateComputation(ComputationType type, const QByteArray& data,
                                    const QVariantMap& parameters = {},
                                    AccelerationType preferredType = AccelerationType::CPU_SIMD);
    
    // Batch operations
    QList<QByteArray> accelerateBatch(ComputationType type, const QList<QByteArray>& dataList,
                                     const QVariantMap& parameters = {});
    void accelerateBatchAsync(ComputationType type, const QList<QByteArray>& dataList,
                             const QVariantMap& parameters,
                             std::function<void(const QList<QByteArray>&)> callback);
    
    // Performance benchmarking
    BenchmarkResult benchmarkDevice(const QString& deviceId, ComputationType computation);
    QList<BenchmarkResult> benchmarkAllDevices(ComputationType computation);
    void runComprehensiveBenchmark();
    
    // Configuration and optimization
    void setPreferredAccelerationType(ComputationType computation, AccelerationType type);
    AccelerationType getPreferredAccelerationType(ComputationType computation) const;
    void enableLoadBalancing(bool enable);
    void setDevicePriority(const QString& deviceId, int priority);
    
    // Resource monitoring
    QVariantMap getSystemPerformanceMetrics() const;
    QVariantMap getDeviceUtilization() const;
    double getPowerConsumption() const;
    void resetPerformanceCounters();
    
    // Component access
    CUDAAccelerator* getCUDAAccelerator() const;
    OpenCLAccelerator* getOpenCLAccelerator() const;
    SIMDAccelerator* getSIMDAccelerator() const;
    
signals:
    void acceleratorInitialized();
    void deviceDiscovered(const HardwareDevice& device);
    void taskSubmitted(const QString& taskId);
    void taskCompleted(const QString& taskId, const QByteArray& result);
    void taskFailed(const QString& taskId, const QString& error);
    void benchmarkCompleted(const QString& deviceId, const BenchmarkResult& result);
    void performanceAlert(const QString& deviceId, const QString& message);
    void accelerationError(const QString& error);
    
private slots:
    void onCudaTaskCompleted(const QString& taskId, const QByteArray& result);
    void onOpenCLTaskCompleted(const QString& taskId, const QByteArray& result);
    void onSIMDTaskCompleted(const QString& taskId, const QByteArray& result);
    void onDeviceUtilizationChanged(const QString& deviceId, double utilization);
    
private:
    // Acceleration components
    std::unique_ptr<CUDAAccelerator> m_cudaAccelerator;
    std::unique_ptr<OpenCLAccelerator> m_openclAccelerator;
    std::unique_ptr<SIMDAccelerator> m_simdAccelerator;
    
    // Device and task management
    mutable QMutex m_devicesMutex;
    QList<HardwareDevice> m_devices;
    QString m_currentDevice;
    
    mutable QMutex m_tasksMutex;
    QMap<QString, AccelerationTask> m_activeTasks;
    QMap<QString, AccelerationTask> m_completedTasks;
    QAtomicInt m_taskCounter;
    
    // Configuration
    QMap<ComputationType, AccelerationType> m_preferredAccelerators;
    QMap<QString, int> m_devicePriorities;
    bool m_loadBalancingEnabled;
    bool m_initialized;
    
    // Performance monitoring
    mutable QMutex m_metricsMutex;
    QMap<QString, BenchmarkResult> m_benchmarkResults;
    QVariantMap m_performanceMetrics;
    QElapsedTimer m_performanceTimer;
    
    // Helper methods
    void discoverDevices();
    void initializeAccelerators();
    QString selectOptimalDevice(ComputationType computation, AccelerationType preferredType);
    void updateDeviceUtilization();
    void processTaskQueue();
    QString generateTaskId();
    void routeTaskToAccelerator(const AccelerationTask& task);
    double calculateDeviceScore(const HardwareDevice& device, ComputationType computation) const;
    void loadAcceleratorConfiguration();
    void saveAcceleratorConfiguration() const;
};

#endif // HARDWARE_ACCELERATOR_H