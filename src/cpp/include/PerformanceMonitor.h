#ifndef PERFORMANCEMONITOR_H
#define PERFORMANCEMONITOR_H

#include <QObject>
#include <QTimer>
#include <QDateTime>
#include <QJsonObject>
#include <QMutex>
#include <memory>
#include <atomic>
#include <vector>

#ifdef Q_OS_WIN
#include <windows.h>
#include <psapi.h>
#include <pdh.h>
#pragma comment(lib, "pdh.lib")
#pragma comment(lib, "psapi.lib")
#elif defined(Q_OS_LINUX)
#include <sys/sysinfo.h>
#include <sys/times.h>
#include <sys/vtimes.h>
#include <unistd.h>
#elif defined(Q_OS_MACOS)
#include <mach/mach.h>
#include <sys/sysctl.h>
#endif

// Enhanced performance metrics structure
struct PerformanceMetrics {
    double cpuUsagePercent = 0.0;
    double memoryUsagePercent = 0.0;
    qint64 memoryUsedBytes = 0;
    qint64 memoryTotalBytes = 0;
    double diskUsagePercent = 0.0;
    qint64 diskReadRate = 0;
    qint64 diskWriteRate = 0;
    double processCpuPercent = 0.0;
    qint64 processMemoryBytes = 0;
    int processThreadCount = 0;
    QDateTime timestamp;
    
    PerformanceMetrics() {
        timestamp = QDateTime::currentDateTime();
    }
};

// Performance alert structure
struct PerformanceAlert {
    enum AlertType {
        CPU_HIGH,
        MEMORY_HIGH, 
        DISK_HIGH,
        PROCESS_HIGH,
        CUSTOM
    };
    
    QDateTime timestamp;
    AlertType type;
    QString message;
    double value;
    double threshold;
    QString severity;
    
    PerformanceAlert() : type(CUSTOM), value(0.0), threshold(0.0), severity("INFO") {
        timestamp = QDateTime::currentDateTime();
    }
};

// Enhanced performance monitor with real-time metrics and alerting
class PerformanceMonitor : public QObject
{
    Q_OBJECT

public:
    explicit PerformanceMonitor(QObject *parent = nullptr, int intervalMs = 1000);
    ~PerformanceMonitor();

    // Basic control
    void start();
    void stop();
    void pause();
    void resume();
    bool isRunning() const { return m_isRunning.load(); }
    
    // Configuration
    void setUpdateInterval(int milliseconds);
    int getUpdateInterval() const { return m_updateInterval; }
    void setAlertThresholds(double cpuThreshold, double memoryThreshold, double diskThreshold);
    
    // Current metrics
    PerformanceMetrics getCurrentMetrics() const;
    PerformanceMetrics getAverageMetrics(int seconds = 60) const;
    
    // Historical data
    std::vector<PerformanceMetrics> getHistoricalData(int minutes = 60) const;
    std::vector<PerformanceAlert> getRecentAlerts(int minutes = 60) const;
    
    // Statistics
    struct PerformanceStatistics {
        double avgCpuUsage = 0.0;
        double peakCpuUsage = 0.0;
        double avgMemoryUsage = 0.0;
        double peakMemoryUsage = 0.0;
        qint64 peakMemoryBytes = 0;
        QDateTime monitoringStart;
        qint64 totalSamples = 0;
        int alertsGenerated = 0;
    };
    
    PerformanceStatistics getStatistics() const;
    void resetStatistics();
    
    // Export functionality
    bool exportMetricsToCSV(const QString& filePath, int minutes = 60) const;
    QJsonObject generatePerformanceReport() const;
    
    // System information
    static QJsonObject getSystemInfo();
    static qint64 getTotalMemory();
    static int getProcessorCount();

signals:
    // Enhanced signals
    void performanceUpdate(double cpuLoad, double memoryUsedMB);
    void metricsUpdated(const PerformanceMetrics& metrics);
    void alertGenerated(const PerformanceAlert& alert);
    void monitoringStarted();
    void monitoringStopped();

private slots:
    void captureMetrics();
    void checkAlerts(const PerformanceMetrics& metrics);

private:
    // Core functionality
    void init();
    double getCurrentCpuLoad();
    double getCurrentMemoryUsageMB();
    bool collectSystemMetrics(PerformanceMetrics& metrics);
    bool collectProcessMetrics(PerformanceMetrics& metrics);
    
#ifdef Q_OS_WIN
    bool initializeWindowsCounters();
    void cleanupWindowsCounters();
    bool collectWindowsMetrics(PerformanceMetrics& metrics);
#endif

#ifdef Q_OS_LINUX
    bool collectLinuxMetrics(PerformanceMetrics& metrics);
    bool readProcStat(PerformanceMetrics& metrics);
    bool readMemInfo(PerformanceMetrics& metrics);
#endif

#ifdef Q_OS_MACOS
    bool collectMacOSMetrics(PerformanceMetrics& metrics);
#endif

    // Alert processing
    void generateAlert(PerformanceAlert::AlertType type, const QString& message,
                      double value, double threshold, const QString& severity);
    
    // Data management
    void addMetricsToHistory(const PerformanceMetrics& metrics);
    void updateStatistics(const PerformanceMetrics& metrics);
    void cleanupOldData();
    
    // Member variables
    std::atomic<bool> m_isRunning{false};
    std::atomic<bool> m_isPaused{false};
    
    QTimer m_timer;
    int m_updateInterval;
    
    // Current metrics and history
    PerformanceMetrics m_currentMetrics;
    std::vector<PerformanceMetrics> m_metricsHistory;
    std::vector<PerformanceAlert> m_alertHistory;
    
    // Alert thresholds
    double m_cpuThreshold = 80.0;    // 80%
    double m_memoryThreshold = 85.0; // 85%
    double m_diskThreshold = 90.0;   // 90%
    
    // Statistics
    mutable PerformanceStatistics m_statistics;
    QDateTime m_monitoringStartTime;
    
    // Thread safety
    mutable QMutex m_metricsMutex;
    mutable QMutex m_historyMutex;
    
    // Platform-specific data
#ifdef Q_OS_WIN
    ULARGE_INTEGER m_lastCpu, m_lastSysCpu, m_lastUserCpu;
    int m_numProcessors;
    PDH_HQUERY m_queryHandle;
    PDH_HCOUNTER m_cpuCounter;
    PDH_HCOUNTER m_memoryCounter;
    bool m_countersInitialized;
#elif defined(Q_OS_LINUX)
    clock_t m_lastCpu, m_lastSysCpu, m_lastUserCpu;
    int m_numProcessors;
    qint64 m_lastCpuTotal;
    qint64 m_lastCpuIdle;
#elif defined(Q_OS_MACOS)
    uint64_t m_totalTime;
    uint64_t m_prevTotalTime;
    uint64_t m_userTime;
    uint64_t m_prevUserTime;
    uint64_t m_systemTime;
    uint64_t m_prevSystemTime;
    mach_port_t m_hostPort;
#endif

    // Constants
    static constexpr int MAX_HISTORY_MINUTES = 1440; // 24 hours
    static constexpr int MAX_ALERT_HISTORY = 1000;
};

#endif // PERFORMANCEMONITOR_H
