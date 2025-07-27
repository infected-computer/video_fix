#pragma once
/*
 * PhoenixDRS Professional - Worker Thread
 * מערכת שחזור מידע מקצועית - Thread עובד
 * 
 * High-performance background processing
 */

#include "Common.h"
#include <QThread>
#include <QMutex>
#include <QWaitCondition>
#include <QQueue>
#include <QElapsedTimer>
#include <QAtomicInt>
#include <QAtomicPointer>

namespace PhoenixDRS {

class DiskImager;
class FileCarver;
class RaidReconstructor;
class VideoRebuilder;
class ForensicLogger;

struct OperationParameters {
    OperationType type;
    QVariantMap parameters;
    QString operationId;
    int priority;
    
    OperationParameters() : type(OperationType::Unknown), priority(0) {}
};

class WorkerThread : public QThread
{
    Q_OBJECT

public:
    enum Priority {
        LowPriority = 0,
        NormalPriority = 1,
        HighPriority = 2,
        CriticalPriority = 3
    };

    explicit WorkerThread(QObject *parent = nullptr);
    ~WorkerThread() override;

    // Operation management
    QString submitOperation(OperationType type, const QVariantMap &parameters, 
                          Priority priority = NormalPriority);
    void cancelOperation(const QString &operationId);
    void cancelAllOperations();
    void pauseOperations();
    void resumeOperations();
    
    // State queries
    bool isOperationRunning() const;
    bool hasQueuedOperations() const;
    int getQueuedOperationCount() const;
    QString getCurrentOperationId() const;
    OperationType getCurrentOperationType() const;
    double getCurrentProgress() const;
    
    // Performance control
    void setMaxWorkers(int maxWorkers);
    int getMaxWorkers() const;
    void setPriority(QThread::Priority priority);
    void setProcessingMode(ProcessingMode mode);
    ProcessingMode getProcessingMode() const;
    
    // Statistics
    qint64 getTotalBytesProcessed() const;
    int getTotalOperationsCompleted() const;
    int getTotalOperationsFailed() const;
    double getAverageOperationTime() const;
    
    // Memory management
    void setMemoryLimit(qint64 limitBytes);
    qint64 getMemoryUsage() const;
    void optimizeMemoryUsage();
    
    // Resource monitoring
    void enableResourceMonitoring(bool enabled);
    bool isResourceMonitoringEnabled() const;

signals:
    // Operation lifecycle
    void operationStarted(OperationType type, const QString &operationId);
    void progressUpdated(int percentage, const QString &message, const QString &operationId);
    void operationCompleted(const OperationResult &result);
    void operationCancelled(const QString &operationId);
    void operationFailed(const QString &operationId, const QString &error);
    
    // Logging
    void logMessage(LogLevel level, const QString &message);
    
    // Resource monitoring
    void memoryUsageChanged(qint64 usage);
    void cpuUsageChanged(double percentage);
    void diskIoRateChanged(qint64 bytesPerSecond);
    
    // Warnings
    void memoryLimitReached();
    void operationStalled(const QString &operationId);
    void resourceConstraint(const QString &constraint);

protected:
    void run() override;

private slots:
    void onProgressTimer();
    void onResourceMonitorTimer();
    void onMemoryCleanupTimer();

private:
    // Core processing methods
    void processOperationQueue();
    bool executeOperation(const OperationParameters &params);
    void updateProgress(int percentage, const QString &message = QString());
    void completeOperation(const OperationResult &result);
    void failOperation(const QString &error);
    
    // Specific operation handlers
    bool executeDiskImaging(const QVariantMap &params);
    bool executeFileCarving(const QVariantMap &params);
    bool executeVideoRebuilding(const QVariantMap &params);
    bool executeRaidReconstruction(const QVariantMap &params);
    bool executeValidation(const QVariantMap &params);
    
    // Resource management
    bool checkResourceLimits();
    void adjustProcessingStrategy();
    void cleanupMemory();
    void monitorSystemResources();
    
    // Performance optimization
    void optimizeForOperation(OperationType type);
    void adjustThreadPriority();
    void manageWorkerThreads();
    
    // Error handling
    void handleOperationError(const std::exception &e);
    void handleSystemError(const QString &operation, int errorCode);
    void reportResourceIssue(const QString &issue);
    
    // State management
    void setState(const QString &state);
    void updateStatistics(const OperationResult &result);
    void logOperationMetrics(const OperationParameters &params, const OperationResult &result);

private:
    // Core components
    std::unique_ptr<DiskImager> m_diskImager;
    std::unique_ptr<FileCarver> m_fileCarver;
    std::unique_ptr<RaidReconstructor> m_raidReconstructor;
    std::unique_ptr<VideoRebuilder> m_videoRebuilder;
    std::unique_ptr<ForensicLogger> m_forensicLogger;
    
    // Operation queue management
    QQueue<OperationParameters> m_operationQueue;
    mutable QMutex m_queueMutex;
    QWaitCondition m_queueNotEmpty;
    QWaitCondition m_operationComplete;
    
    // Current operation state
    QAtomicPointer<OperationParameters> m_currentOperation;
    QAtomicInt m_currentProgress;
    QString m_currentOperationId;
    QString m_currentStatusMessage;
    QElapsedTimer m_operationTimer;
    
    // Threading control
    QAtomicInt m_shouldStop;
    QAtomicInt m_isPaused;
    QAtomicInt m_isProcessing;
    ProcessingMode m_processingMode;
    int m_maxWorkers;
    
    // Performance monitoring
    QTimer *m_progressTimer;
    QTimer *m_resourceMonitorTimer;
    QTimer *m_memoryCleanupTimer;
    bool m_resourceMonitoringEnabled;
    
    // Resource limits
    qint64 m_memoryLimit;
    qint64 m_currentMemoryUsage;
    double m_cpuUsageThreshold;
    qint64 m_diskIoThreshold;
    
    // Statistics
    QAtomicInt m_totalOperationsCompleted;
    QAtomicInt m_totalOperationsFailed;
    std::atomic<qint64> m_totalBytesProcessed;
    std::atomic<qint64> m_totalProcessingTime; // microseconds
    
    // Performance metrics
    QList<double> m_operationTimes;
    QList<qint64> m_memoryUsageHistory;
    QList<double> m_cpuUsageHistory;
    static const int MaxHistorySize = 100;
    
    // Error handling
    int m_consecutiveErrors;
    QDateTime m_lastErrorTime;
    static const int MaxConsecutiveErrors = 5;
    static const int ErrorCooldownMs = 30000; // 30 seconds
    
    // Thread synchronization
    mutable QMutex m_stateMutex;
    mutable QMutex m_statisticsMutex;
    mutable QMutex m_resourceMutex;
    
    // Operation ID generation
    static QAtomicInt s_operationIdCounter;
    
    // Constants
    static const int ProgressUpdateIntervalMs = 250;
    static const int ResourceMonitorIntervalMs = 1000;
    static const int MemoryCleanupIntervalMs = 60000; // 1 minute
    static const qint64 DefaultMemoryLimitMB = 1024; // 1GB
    static const double DefaultCpuThreshold = 90.0;
    static const qint64 DefaultDiskIoThresholdMBps = 100;
};

// Helper class for automatic operation timing
class OperationTimer {
public:
    explicit OperationTimer(const QString &operationName);
    ~OperationTimer();
    
    qint64 elapsed() const;
    void logProgress(const QString &stage) const;
    
private:
    QString m_operationName;
    QElapsedTimer m_timer;
};

// Helper class for memory management
class MemoryGuard {
public:
    explicit MemoryGuard(qint64 expectedUsage, qint64 limit);
    ~MemoryGuard();
    
    bool canProceed() const;
    void updateUsage(qint64 currentUsage);
    
private:
    qint64 m_expectedUsage;
    qint64 m_limit;
    qint64 m_initialUsage;
    bool m_canProceed;
};

// Thread pool for parallel operations
class WorkerThreadPool {
public:
    explicit WorkerThreadPool(int maxThreads = 4);
    ~WorkerThreadPool();
    
    void setMaxThreads(int maxThreads);
    int getMaxThreads() const;
    int getActiveThreads() const;
    
    QString submitOperation(OperationType type, const QVariantMap &parameters,
                          WorkerThread::Priority priority = WorkerThread::NormalPriority);
    void cancelOperation(const QString &operationId);
    void cancelAllOperations();
    
    void pauseAll();
    void resumeAll();
    void waitForAll(int timeoutMs = -1);
    
    // Statistics
    int getTotalQueuedOperations() const;
    qint64 getTotalBytesProcessed() const;
    double getAverageOperationTime() const;

signals:
    void operationCompleted(const OperationResult &result);
    void operationFailed(const QString &operationId, const QString &error);
    void allOperationsCompleted();

private:
    void redistributeOperations();
    WorkerThread* getLeastBusyThread();
    void cleanupFinishedThreads();
    
private:
    QList<WorkerThread*> m_threads;
    QHash<QString, WorkerThread*> m_operationToThread;
    int m_maxThreads;
    mutable QMutex m_poolMutex;
    
    QTimer *m_cleanupTimer;
};

} // namespace PhoenixDRS