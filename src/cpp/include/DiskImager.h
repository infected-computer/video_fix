/*
 * PhoenixDRS Professional - High-Performance Disk Imaging Engine
 * מנוע הדמיית דיסק בביצועים גבוהים - PhoenixDRS מקצועי
 * 
 * Advanced forensic disk imaging with DD, E01, AFF4 support
 * הדמיית דיסק פורנזית מתקדמת עם תמיכה ב-DD, E01, AFF4
 */

#pragma once

#include "Common.h"
#include <QObject>
#include <QThread>
#include <QMutex>
#include <QWaitCondition>
#include <QTimer>
#include <QCryptographicHash>
#include <QFile>
#include <QDateTime>
#include <QElapsedTimer>
#include <QFileInfo>
#include <QStorageInfo>

#include <memory>
#include <atomic>
#include <array>
#include <vector>
#include <unordered_map>
#include <fstream>

#ifdef Q_OS_WIN
#include <Windows.h>
#include <winioctl.h>
#include <setupapi.h>
#include <devguid.h>
#pragma comment(lib, "setupapi.lib")
#endif

#ifdef Q_OS_LINUX
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/ioctl.h>
#include <linux/fs.h>
#include <linux/hdreg.h>
#include <fcntl.h>
#include <unistd.h>
#include <libudev.h>
#endif

namespace PhoenixDRS {

// Image format types
enum class ImageFormat {
    RAW_DD,           // Raw DD format
    ENCASE_E01,       // EnCase Expert Witness format
    AFF4,             // Advanced Forensic Format v4
    VMDK,             // VMware Virtual Disk
    VHD,              // Microsoft Virtual Hard Disk
    QEMU_QCOW2        // QEMU Copy-On-Write v2
};

// Compression levels
enum class CompressionLevel {
    None = 0,
    Fast = 1,
    Normal = 6,
    Maximum = 9
};

// Hash algorithms
enum class HashAlgorithm {
    MD5,
    SHA1,
    SHA256,
    SHA512,
    CRC32
};

// Device information structure
struct DeviceInfo {
    QString devicePath;
    QString model;
    QString serialNumber;
    QString firmware;
    qint64 totalSize;
    qint64 sectorSize;
    qint64 totalSectors;
    QString busType;
    bool isRemovable;
    bool isReadOnly;
    QString vendor;
    QDateTime detectionTime;
    
    // Performance characteristics
    qint64 maxTransferRate;
    qint64 averageSeekTime;
    
    // Health information
    QString smartStatus;
    int temperature;
    int powerOnHours;
    
    DeviceInfo() : totalSize(0), sectorSize(512), totalSectors(0), 
                   isRemovable(false), isReadOnly(false), 
                   maxTransferRate(0), averageSeekTime(0),
                   temperature(-1), powerOnHours(0) {}
};

// Imaging parameters
struct ImagingParameters {
    QString sourceDevice;
    QString destinationPath;
    ImageFormat format;
    CompressionLevel compression;
    std::vector<HashAlgorithm> hashAlgorithms;
    
    // Advanced settings
    qint64 blockSize;
    int maxRetries;
    int retryDelay;
    bool verifyAfterImaging;
    bool skipBadSectors;
    bool zerofillBadSectors;
    
    // Forensic metadata
    QString caseName;
    QString examiner;
    QString evidence;
    QString notes;
    QDateTime acquisitionDate;
    
    ImagingParameters() : format(ImageFormat::RAW_DD), 
                         compression(CompressionLevel::None),
                         blockSize(1024 * 1024), // 1MB default
                         maxRetries(3),
                         retryDelay(100),
                         verifyAfterImaging(true),
                         skipBadSectors(false),
                         zerofillBadSectors(true) {}
};

// Progress information
struct ImagingProgress {
    qint64 bytesProcessed;
    qint64 totalBytes;
    qint64 currentTransferRate;
    qint64 averageTransferRate;
    QTime estimatedTimeRemaining;
    QTime elapsedTime;
    int badSectorsFound;
    int retriesPerformed;
    QString currentOperation;
    
    ImagingProgress() : bytesProcessed(0), totalBytes(0),
                       currentTransferRate(0), averageTransferRate(0),
                       badSectorsFound(0), retriesPerformed(0) {}
};

// Hash result structure
struct HashResult {
    HashAlgorithm algorithm;
    QByteArray hash;
    QString hexString;
    
    HashResult(HashAlgorithm alg = HashAlgorithm::SHA256) : algorithm(alg) {}
};

// Forward declarations
class DiskReader;
class ImageWriter;
class HashCalculator;

/*
 * Main disk imaging class
 * מחלקה ראשית להדמיית דיסק
 */
class PHOENIXDRS_EXPORT DiskImager : public QObject
{
    Q_OBJECT

public:
    explicit DiskImager(QObject* parent = nullptr);
    ~DiskImager() override;

    // Device discovery and information
    static std::vector<DeviceInfo> discoverDevices();
    static DeviceInfo getDeviceInfo(const QString& devicePath);
    static bool isDeviceAccessible(const QString& devicePath);
    
    // Main imaging operations
    bool startImaging(const ImagingParameters& params);
    void pauseImaging();
    void resumeImaging();
    void cancelImaging();
    
    // Status and progress
    bool isRunning() const { return m_isRunning.load(); }
    bool isPaused() const { return m_isPaused.load(); }
    ImagingProgress getProgress() const;
    
    // Configuration
    void setBlockSize(qint64 size);
    void setMaxRetries(int retries);
    void setRetryDelay(int milliseconds);
    void setCompressionLevel(CompressionLevel level);
    
    // Results and verification
    std::vector<HashResult> getCalculatedHashes() const;
    bool verifyImage(const QString& imagePath, const std::vector<HashResult>& expectedHashes);
    
    // Performance monitoring
    struct PerformanceStats {
        qint64 totalBytesRead;
        qint64 totalBytesWritten;
        qint64 peakTransferRate;
        qint64 averageTransferRate;
        int totalRetries;
        int badSectorsSkipped;
        QTime totalTime;
        
        PerformanceStats() : totalBytesRead(0), totalBytesWritten(0),
                           peakTransferRate(0), averageTransferRate(0),
                           totalRetries(0), badSectorsSkipped(0) {}
    };
    
    PerformanceStats getPerformanceStats() const { return m_stats; }

public slots:
    void startImagingAsync(const ImagingParameters& params);

signals:
    void imagingStarted(const QString& sourceDevice, const QString& destinationPath);
    void progressUpdated(const ImagingProgress& progress);
    void imagingCompleted(bool success, const QString& message);
    void imagingPaused();
    void imagingResumed();
    void imagingCancelled();
    void errorOccurred(const QString& error);
    void badSectorDetected(qint64 sectorNumber, qint64 byteOffset);
    void retryAttempt(int attempt, int maxAttempts);
    void hashCalculated(HashAlgorithm algorithm, const QString& hexHash);

private slots:
    void updateProgress();
    void handleWorkerFinished();
    void handleWorkerError(const QString& error);

private:
    // Internal classes
    class ImagingWorker;
    friend class ImagingWorker;
    
    // Core functionality
    bool initializeSource(const QString& devicePath);
    bool initializeDestination(const QString& destinationPath, ImageFormat format);
    bool performImaging();
    void cleanup();
    
    // Platform-specific device access
#ifdef Q_OS_WIN
    HANDLE openWindowsDevice(const QString& devicePath);
    bool getWindowsDeviceInfo(HANDLE device, DeviceInfo& info);
    static std::vector<DeviceInfo> enumerateWindowsDevices();
#endif

#ifdef Q_OS_LINUX
    int openLinuxDevice(const QString& devicePath);
    bool getLinuxDeviceInfo(int fd, DeviceInfo& info);
    static std::vector<DeviceInfo> enumerateLinuxDevices();
#endif

    // Hash calculation
    void initializeHashCalculators();
    void updateHashCalculators(const QByteArray& data);
    void finalizeHashCalculators();
    
    // Error handling and recovery
    bool handleReadError(qint64 offset, qint64 size);
    bool retryOperation(std::function<bool()> operation, int maxRetries);
    
    // Performance optimization
    void optimizeBufferSizes();
    void adjustTransferParameters();
    
    // Member variables
    std::atomic<bool> m_isRunning{false};
    std::atomic<bool> m_isPaused{false};
    std::atomic<bool> m_shouldCancel{false};
    
    ImagingParameters m_parameters;
    ImagingProgress m_progress;
    PerformanceStats m_stats;
    
    // Threading
    std::unique_ptr<ImagingWorker> m_worker;
    QThread* m_workerThread;
    QTimer* m_progressTimer;
    QMutex m_progressMutex;
    QWaitCondition m_pauseCondition;
    
    // I/O handles
#ifdef Q_OS_WIN
    HANDLE m_sourceHandle;
#else
    int m_sourceHandle;
#endif
    
    std::unique_ptr<QFile> m_destinationFile;
    std::unique_ptr<ImageWriter> m_imageWriter;
    
    // Hash calculation
    std::vector<std::unique_ptr<HashCalculator>> m_hashCalculators;
    std::vector<HashResult> m_calculatedHashes;
    
    // Buffer management
    static constexpr size_t DEFAULT_BUFFER_SIZE = 1024 * 1024; // 1MB
    static constexpr size_t MAX_BUFFER_SIZE = 64 * 1024 * 1024; // 64MB
    std::vector<char> m_readBuffer;
    std::vector<char> m_writeBuffer;
    
    // Performance monitoring
    QElapsedTimer m_operationTimer;
    std::array<qint64, 10> m_recentTransferRates{};
    size_t m_transferRateIndex{0};
    
    // Constants
    static constexpr int PROGRESS_UPDATE_INTERVAL = 100; // milliseconds
    static constexpr int TRANSFER_RATE_SAMPLES = 10;
    static constexpr qint64 MIN_BLOCK_SIZE = 512;
    static constexpr qint64 MAX_BLOCK_SIZE = 64 * 1024 * 1024;
};

/*
 * Specialized image writers for different formats
 */
class PHOENIXDRS_EXPORT ImageWriter
{
public:
    virtual ~ImageWriter() = default;
    
    virtual bool initialize(const QString& path, const ImagingParameters& params) = 0;
    virtual bool writeBlock(const QByteArray& data, qint64 offset) = 0;
    virtual bool finalize() = 0;
    virtual qint64 getBytesWritten() const = 0;
    virtual QString getError() const = 0;
    
    static std::unique_ptr<ImageWriter> createWriter(ImageFormat format);
};

/*
 * Raw DD format writer
 */
class PHOENIXDRS_EXPORT RawImageWriter : public ImageWriter
{
public:
    bool initialize(const QString& path, const ImagingParameters& params) override;
    bool writeBlock(const QByteArray& data, qint64 offset) override;
    bool finalize() override;
    qint64 getBytesWritten() const override { return m_bytesWritten; }
    QString getError() const override { return m_lastError; }

private:
    std::unique_ptr<QFile> m_file;
    qint64 m_bytesWritten{0};
    QString m_lastError;
};

/*
 * EnCase E01 format writer
 */
class PHOENIXDRS_EXPORT E01ImageWriter : public ImageWriter
{
public:
    bool initialize(const QString& path, const ImagingParameters& params) override;
    bool writeBlock(const QByteArray& data, qint64 offset) override;
    bool finalize() override;
    qint64 getBytesWritten() const override { return m_bytesWritten; }
    QString getError() const override { return m_lastError; }

private:
    struct E01Header;
    struct E01Section;
    
    bool writeHeader();
    bool writeDataSection(const QByteArray& data);
    bool writeHashSection();
    bool writeDoneSection();
    
    std::unique_ptr<QFile> m_file;
    qint64 m_bytesWritten{0};
    QString m_lastError;
    ImagingParameters m_params;
    
    // E01 specific data
    std::vector<qint64> m_chunkOffsets;
    QCryptographicHash m_md5Hash{QCryptographicHash::Md5};
    QCryptographicHash m_sha1Hash{QCryptographicHash::Sha1};
};

/*
 * Multi-threaded hash calculator
 */
class PHOENIXDRS_EXPORT HashCalculator : public QObject
{
    Q_OBJECT

public:
    explicit HashCalculator(HashAlgorithm algorithm, QObject* parent = nullptr);
    ~HashCalculator() override;
    
    void addData(const QByteArray& data);
    HashResult getResult();
    void reset();
    
    HashAlgorithm getAlgorithm() const { return m_algorithm; }

private:
    HashAlgorithm m_algorithm;
    std::unique_ptr<QCryptographicHash> m_hash;
    QMutex m_mutex;
};

} // namespace PhoenixDRS