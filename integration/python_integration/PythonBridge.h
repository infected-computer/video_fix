/*
 * PhoenixDRS Professional - Python-C++ Integration Bridge
 * גשר אינטגרציה Python-C++ - PhoenixDRS מקצועי
 */

#pragma once

#include "../cpp_gui/include/Common.h"
#include "../cpp_gui/include/Core/ErrorHandling.h"
#include "../cpp_gui/include/DiskImager.h"
#include "../cpp_gui/include/FileCarver.h"
#include "../cpp_gui/include/VideoRebuilder.h"
#include "../cpp_gui/include/ForensicLogger.h"

#include <QtCore/QObject>
#include <QtCore/QString>
#include <QtCore/QVariant>
#include <QtCore/QJsonObject>
#include <QtCore/QThread>
#include <QtCore/QMutex>

#include <Python.h>
#include <memory>
#include <functional>
#include <unordered_map>

namespace PhoenixDRS {
namespace Integration {

/*
 * Python callback wrapper for C++ progress notifications
 */
class PHOENIXDRS_EXPORT PythonCallback
{
public:
    explicit PythonCallback(PyObject* callback);
    ~PythonCallback();
    
    // Call Python function with various argument types
    bool call();
    bool call(int arg);
    bool call(double arg);
    bool call(const QString& arg);
    bool call(int arg1, const QString& arg2);
    bool call(double progress, const QString& message);
    
    bool isValid() const { return m_callback != nullptr; }

private:
    PyObject* m_callback;
    mutable QMutex m_mutex;
};

/*
 * Python-aware wrapper for DiskImager
 */
class PHOENIXDRS_EXPORT PythonDiskImager : public QObject
{
    Q_OBJECT
    
public:
    explicit PythonDiskImager(QObject* parent = nullptr);
    ~PythonDiskImager() override;
    
    // Python-friendly interface
    struct ImageResult {
        bool success = false;
        QString errorMessage;
        QString imagePath;
        qint64 totalBytes = 0;
        qint64 totalSectors = 0;
        int badSectorCount = 0;
        QString md5Hash;
        QString sha256Hash;
        double elapsedSeconds = 0.0;
    };
    
    ImageResult createImage(const QString& sourcePath, 
                           const QString& destinationPath,
                           int sectorSize = 512,
                           PyObject* progressCallback = nullptr,
                           PyObject* errorCallback = nullptr);
    
    ImageResult verifyImage(const QString& imagePath,
                           PyObject* progressCallback = nullptr);
    
    // Status queries
    bool isRunning() const;
    void cancel();
    
    // Configuration
    void setMaxRetries(int retries);
    void setRetryDelay(double seconds);
    void setCompressionEnabled(bool enabled);
    void setEncryptionEnabled(bool enabled, const QString& password = QString());

signals:
    void progressChanged(double percentage, const QString& message);
    void errorOccurred(const QString& error);
    void imageCreated(const QString& imagePath);

private slots:
    void onProgressChanged(int percentage);
    void onStatusChanged(const QString& status);
    void onImageComplete(bool success, const QString& message);

private:
    std::unique_ptr<DiskImager> m_imager;
    std::unique_ptr<PythonCallback> m_progressCallback;
    std::unique_ptr<PythonCallback> m_errorCallback;
    QThread* m_workerThread;
    bool m_isRunning;
    mutable QMutex m_mutex;
};

/*
 * Python-aware wrapper for FileCarver
 */
class PHOENIXDRS_EXPORT PythonFileCarver : public QObject
{
    Q_OBJECT
    
public:
    explicit PythonFileCarver(QObject* parent = nullptr);
    ~PythonFileCarver() override;
    
    struct CarveResult {
        bool success = false;
        QString errorMessage;
        QStringList carvedFiles;
        int totalFilesFound = 0;
        int validFiles = 0;
        qint64 totalBytesRecovered = 0;
        double elapsedSeconds = 0.0;
        QJsonObject statistics;
    };
    
    CarveResult carveFiles(const QString& imagePath,
                          const QString& outputDirectory,
                          const QString& signaturesPath = QString(),
                          PyObject* progressCallback = nullptr,
                          PyObject* fileFoundCallback = nullptr);
    
    CarveResult carveFilesParallel(const QString& imagePath,
                                  const QString& outputDirectory,
                                  int maxWorkers = 0,
                                  const QString& signaturesPath = QString(),
                                  PyObject* progressCallback = nullptr,
                                  PyObject* fileFoundCallback = nullptr);
    
    // Status queries
    bool isRunning() const;
    void cancel();
    
    // Configuration
    void setChunkSize(int chunkSize);
    void setMinFileSize(int minSize);
    void setMaxFileSize(qint64 maxSize);
    void setFileTypeFilter(const QStringList& extensions);

signals:
    void progressChanged(double percentage, const QString& message);
    void fileFound(const QString& filePath, const QString& fileType);
    void carvingComplete(int filesFound);

private slots:
    void onProgressChanged(int percentage);
    void onFileCarved(const QString& filePath, const QString& fileType);
    void onCarvingComplete(bool success, const QString& message);

private:
    std::unique_ptr<FileCarver> m_carver;
    std::unique_ptr<PythonCallback> m_progressCallback;
    std::unique_ptr<PythonCallback> m_fileFoundCallback;
    QThread* m_workerThread;
    bool m_isRunning;
    mutable QMutex m_mutex;
};

/*
 * Python-aware wrapper for VideoRebuilder
 */
class PHOENIXDRS_EXPORT PythonVideoRebuilder : public QObject
{
    Q_OBJECT
    
public:
    explicit PythonVideoRebuilder(QObject* parent = nullptr);
    ~PythonVideoRebuilder() override;
    
    struct RebuildResult {
        bool success = false;
        QString errorMessage;
        QStringList rebuiltVideos;
        int totalVideosFound = 0;
        int successfulRebuilds = 0;
        qint64 totalBytesProcessed = 0;
        double elapsedSeconds = 0.0;
    };
    
    RebuildResult rebuildVideos(const QString& imagePath,
                               const QString& outputDirectory,
                               const QString& videoFormat = "mov",
                               PyObject* progressCallback = nullptr,
                               PyObject* videoFoundCallback = nullptr);
    
    // Status queries
    bool isRunning() const;
    void cancel();
    
    // Configuration
    void setMaxVideoSize(qint64 maxSize);
    void setMinVideoSize(qint64 minSize);
    void setQualityThreshold(double threshold);

signals:
    void progressChanged(double percentage, const QString& message);
    void videoFound(const QString& videoPath, qint64 size);
    void rebuildComplete(int videosRebuilt);

private slots:
    void onProgressChanged(int percentage);
    void onVideoRebuilt(const QString& videoPath);
    void onRebuildComplete(bool success, const QString& message);

private:
    std::unique_ptr<VideoRebuilder> m_rebuilder;
    std::unique_ptr<PythonCallback> m_progressCallback;
    std::unique_ptr<PythonCallback> m_videoFoundCallback;
    QThread* m_workerThread;
    bool m_isRunning;
    mutable QMutex m_mutex;
};

/*
 * Unified Python API facade
 */
class PHOENIXDRS_EXPORT PythonForensicsAPI : public QObject
{
    Q_OBJECT
    
public:
    static PythonForensicsAPI& instance();
    
    // Initialize/cleanup
    bool initialize();
    void shutdown();
    
    // Factory methods for Python wrappers
    std::shared_ptr<PythonDiskImager> createDiskImager();
    std::shared_ptr<PythonFileCarver> createFileCarver();
    std::shared_ptr<PythonVideoRebuilder> createVideoRebuilder();
    
    // Global configuration
    void setTempDirectory(const QString& path);
    QString getTempDirectory() const;
    
    void setLogLevel(const QString& level);
    QString getLogLevel() const;
    
    void enablePerformanceLogging(bool enabled);
    bool isPerformanceLoggingEnabled() const;
    
    // System information
    QJsonObject getSystemInfo() const;
    QJsonObject getMemoryInfo() const;
    QJsonObject getDiskInfo() const;
    
    // Error handling
    QString getLastError() const;
    void clearLastError();
    
    // Statistics
    struct GlobalStatistics {
        int activeOperations = 0;
        qint64 totalBytesProcessed = 0;
        int totalImagesCreated = 0;
        int totalFilesCarved = 0;
        int totalVideosRebuilt = 0;
        double totalElapsedTime = 0.0;
    };
    
    GlobalStatistics getGlobalStatistics() const;
    void resetGlobalStatistics();

signals:
    void operationStarted(const QString& operationType);
    void operationCompleted(const QString& operationType, bool success);
    void globalError(const QString& error);

private:
    PythonForensicsAPI();
    ~PythonForensicsAPI();
    
    void updateStatistics(const QString& operation, bool success, double elapsedTime);
    
    QString m_tempDirectory;
    QString m_logLevel;
    bool m_performanceLogging;
    QString m_lastError;
    GlobalStatistics m_statistics;
    
    std::vector<std::shared_ptr<PythonDiskImager>> m_diskImagers;
    std::vector<std::shared_ptr<PythonFileCarver>> m_fileCarvers;
    std::vector<std::shared_ptr<PythonVideoRebuilder>> m_videoRebuilders;
    
    mutable QMutex m_mutex;
};

/*
 * Shared memory manager for high-performance data exchange
 */
class PHOENIXDRS_EXPORT SharedMemoryManager : public QObject
{
    Q_OBJECT
    
public:
    static SharedMemoryManager& instance();
    
    // Shared memory operations
    QString createSharedBuffer(const QString& name, qint64 size);
    bool attachToSharedBuffer(const QString& name);
    void detachFromSharedBuffer(const QString& name);
    void deleteSharedBuffer(const QString& name);
    
    // Data access
    void* getBufferPointer(const QString& name);
    qint64 getBufferSize(const QString& name) const;
    bool writeToBuffer(const QString& name, const void* data, qint64 size, qint64 offset = 0);
    bool readFromBuffer(const QString& name, void* data, qint64 size, qint64 offset = 0);
    
    // Synchronization
    bool lockBuffer(const QString& name, int timeoutMs = -1);
    void unlockBuffer(const QString& name);
    
    // Status
    QStringList getActiveBuffers() const;
    qint64 getTotalMemoryUsed() const;

private:
    SharedMemoryManager();
    ~SharedMemoryManager();
    
    struct BufferInfo {
        void* pointer = nullptr;
        qint64 size = 0;
        QString systemName;
        QMutex mutex;
        bool locked = false;
    };
    
    std::unordered_map<QString, std::unique_ptr<BufferInfo>> m_buffers;
    mutable QMutex m_mutex;
};

// Utility functions for Python integration
PHOENIXDRS_EXPORT QString qtStringFromPython(PyObject* obj);
PHOENIXDRS_EXPORT PyObject* qtStringToPython(const QString& str);
PHOENIXDRS_EXPORT QJsonObject jsonFromPython(PyObject* obj);
PHOENIXDRS_EXPORT PyObject* jsonToPython(const QJsonObject& json);
PHOENIXDRS_EXPORT QStringList stringListFromPython(PyObject* obj);
PHOENIXDRS_EXPORT PyObject* stringListToPython(const QStringList& list);

// Exception handling
PHOENIXDRS_EXPORT void setPythonException(const PhoenixDRS::Core::PhoenixException& exception);
PHOENIXDRS_EXPORT bool handlePythonException();

} // namespace Integration
} // namespace PhoenixDRS