/*
 * PhoenixDRS Professional - Advanced Memory Forensics and Live RAM Analysis
 * מנוע פורנזיקה מתקדם לזיכרון וניתוח RAM חי - PhoenixDRS מקצועי
 * 
 * State-of-the-art memory analysis for live systems and memory dumps
 * ניתוח זיכרון מתקדם למערכות חיות ודמפים של זיכרון
 */

#pragma once

#include "Common.h"
#include <QObject>
#include <QThread>
#include <QMutex>
#include <QTimer>
#include <QJsonObject>
#include <QJsonArray>
#include <QDateTime>
#include <QNetworkAccessManager>
#include <QProcess>

#include <memory>
#include <atomic>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <queue>

// Memory analysis libraries
#ifdef ENABLE_VOLATILITY
#include <volatility/volatility.h>
#include <volatility/framework.h>
#endif

// Windows memory analysis
#ifdef Q_OS_WIN
#include <windows.h>
#include <winternl.h>
#include <psapi.h>
#include <tlhelp32.h>
#include <dbghelp.h>
#pragma comment(lib, "psapi.lib")
#pragma comment(lib, "dbghelp.lib")
#endif

// Linux memory analysis
#ifdef Q_OS_LINUX
#include <sys/ptrace.h>
#include <sys/wait.h>
#include <sys/user.h>
#include <unistd.h>
#include <fcntl.h>
#include <elf.h>
#endif

// macOS memory analysis
#ifdef Q_OS_MACOS
#include <mach/mach.h>
#include <mach/mach_vm.h>
#include <mach/vm_map.h>
#include <mach/task.h>
#endif

namespace PhoenixDRS {

// Memory region types
enum class MemoryRegionType {
    UNKNOWN = 0,
    CODE_SECTION,          // Executable code
    DATA_SECTION,          // Initialized data
    BSS_SECTION,           // Uninitialized data
    HEAP_MEMORY,           // Dynamic allocation
    STACK_MEMORY,          // Function call stack
    SHARED_LIBRARY,        // Dynamic libraries
    MEMORY_MAPPED_FILE,    // mmap regions
    DEVICE_MEMORY,         // Hardware memory
    KERNEL_MEMORY,         // Kernel space
    USER_MEMORY,           // User space
    VIDEO_MEMORY,          // Graphics memory
    NETWORK_BUFFER,        // Network buffers
    FILESYSTEM_CACHE,      // File system cache
    ENCRYPTED_REGION,      // Encrypted memory
    COMPRESSED_REGION,     // Compressed memory
    SWAP_MEMORY,           // Paged out memory
    HYPERVISOR_MEMORY,     // Virtualization
    SECURE_ENCLAVE         // Hardware security
};

// Memory artifact types
enum class MemoryArtifactType {
    PROCESS_LIST,          // Running processes
    THREAD_LIST,           // Active threads
    MODULE_LIST,           // Loaded modules/DLLs
    HANDLE_TABLE,          // Open handles
    REGISTRY_KEYS,         // Registry artifacts
    NETWORK_CONNECTIONS,   // Network sockets
    FILE_HANDLES,          // Open files
    MUTEX_OBJECTS,         // Synchronization objects
    ENVIRONMENT_VARS,      // Environment variables
    COMMAND_LINE_ARGS,     // Process arguments
    LOADED_DRIVERS,        // Kernel drivers
    INTERRUPT_HANDLERS,    // IRQ handlers
    SYSTEM_CALLS,          // Syscall table
    ROOTKIT_ARTIFACTS,     // Rootkit indicators
    MALWARE_SIGNATURES,    // Malware patterns
    ENCRYPTION_KEYS,       // Cryptographic keys
    PASSWORD_HASHES,       // Cached passwords
    BROWSER_ARTIFACTS,     // Web browser data
    EMAIL_ARTIFACTS,       // Email client data
    INSTANT_MESSAGING,     // Chat applications
    CRYPTOCURRENCY_WALLETS, // Crypto wallets
    FORENSIC_TIMELINE,     // Timeline events
    MEMORY_STRINGS,        // Extracted strings
    BINARY_SIGNATURES,     // Binary patterns
    STEGANOGRAPHIC_DATA,   // Hidden data
    ANTI_FORENSICS,        // Evasion techniques
    HYPERVISOR_ARTIFACTS,  // VM detection
    CONTAINER_ARTIFACTS,   // Docker/containers
    CLOUD_ARTIFACTS        // Cloud services
};

// Memory analysis result
struct MemoryAnalysisResult {
    QString analysisId;                    // Unique analysis ID
    QDateTime analysisTime;               // Analysis timestamp
    QString memorySource;                 // Source (dump file/live system)
    qint64 memorySize;                   // Total memory size
    QString operatingSystem;             // Detected OS
    QString osVersion;                   // OS version
    QString architecture;                // CPU architecture (x64, ARM, etc.)
    
    // Process information
    QJsonArray processList;              // All processes
    QJsonArray suspiciousProcesses;      // Flagged processes
    QJsonArray hiddenProcesses;          // Hidden/rootkit processes
    QJsonArray terminatedProcesses;      // Recently terminated
    
    // Network analysis
    QJsonArray networkConnections;       // Active connections
    QJsonArray suspiciousConnections;    // Malicious connections
    QJsonArray dnsQueries;              // DNS resolution history
    QJsonArray httpTraffic;             // HTTP communications
    
    // File system artifacts
    QJsonArray openFiles;               // Currently open files
    QJsonArray recentFiles;             // Recently accessed files
    QJsonArray deletedFiles;            // Recovered deleted files
    QJsonArray encryptedFiles;          // Encrypted file artifacts
    
    // Registry artifacts (Windows)
    QJsonArray registryKeys;            // Important registry keys
    QJsonArray autoStartEntries;        // Persistence mechanisms
    QJsonArray installedSoftware;       // Installed applications
    QJsonArray userAccounts;            // User account information
    
    // Security artifacts
    QJsonArray passwordHashes;          // Extracted password hashes
    QJsonArray encryptionKeys;          // Found crypto keys
    QJsonArray certificates;            // Digital certificates
    QJsonArray securityTokens;          // Authentication tokens
    
    // Malware analysis
    QJsonArray malwareIndicators;       // Malware signatures
    QJsonArray rootkitArtifacts;        // Rootkit evidence
    QJsonArray packedExecutables;       // Packed/obfuscated code
    QJsonArray injectedCode;            // Code injection
    
    // Browser forensics
    QJsonArray browserHistory;          // Web browsing history
    QJsonArray downloadHistory;         // Downloaded files
    QJsonArray cookies;                 // Browser cookies
    QJsonArray savedPasswords;          // Saved credentials
    QJsonArray bookmarks;               // Browser bookmarks
    
    // Communication artifacts
    QJsonArray emailArtifacts;          // Email evidence
    QJsonArray chatMessages;            // Instant messages
    QJsonArray voipCalls;               // Voice/video calls
    QJsonArray socialMediaActivity;     // Social media evidence
    
    // Cryptocurrency evidence
    QJsonArray cryptoWallets;           // Cryptocurrency wallets
    QJsonArray cryptoTransactions;      // Transaction history
    QJsonArray miningActivity;          // Crypto mining evidence
    
    // Timeline reconstruction
    QJsonArray timelineEvents;          // Chronological events
    QJsonArray userActivity;            // User behavior timeline
    QJsonArray systemEvents;            // System activity timeline
    
    // Advanced analysis
    QJsonArray memoryStrings;           // Extracted strings
    QJsonArray binaryPatterns;          // Binary signatures
    QJsonArray anomalousRegions;        // Suspicious memory areas
    QJsonArray dataCarving;             // Carved file fragments
    
    // Forensic metadata
    double analysisConfidence;          // Overall confidence (0.0-1.0)
    QString evidenceIntegrity;          // Evidence integrity status
    QJsonObject chainOfCustody;         // Chain of custody info
    std::vector<QString> forensicTags;  // Evidence classification
    
    MemoryAnalysisResult() : memorySize(0), analysisConfidence(0.0) {
        analysisId = QUuid::createUuid().toString(QUuid::WithoutBraces);
        analysisTime = QDateTime::currentDateTime();
    }
};

// Live monitoring parameters
struct LiveMonitoringParams {
    bool enableRealTimeAnalysis;        // Real-time analysis
    bool enableBehaviorAnalysis;        // Behavioral analysis
    bool enableThreatDetection;         // Active threat detection
    bool enableNetworkMonitoring;       // Network activity monitoring
    bool enableFileSystemWatching;      // File system monitoring
    bool enableRegistryMonitoring;      // Registry change monitoring
    bool enableProcessMonitoring;       // Process activity monitoring
    bool enableMemoryScanning;          // Continuous memory scanning
    
    // Alerting thresholds
    int suspiciousProcessThreshold;     // Max suspicious processes
    int networkConnectionThreshold;     // Max network connections
    int fileSystemChangeThreshold;      // Max file changes per minute
    double memoryUsageThreshold;        // Memory usage alert (%)
    double cpuUsageThreshold;          // CPU usage alert (%)
    
    // Monitoring intervals
    int processCheckIntervalMs;         // Process check frequency
    int networkCheckIntervalMs;         // Network check frequency
    int memoryCheckIntervalMs;          // Memory check frequency
    int fileSystemCheckIntervalMs;      // File system check frequency
    
    // Data retention
    int maxEventHistory;                // Maximum events to keep
    int maxTimelineEvents;              // Maximum timeline entries
    bool persistToDatabase;             // Save to persistent storage
    QString databasePath;               // Database file path
    
    LiveMonitoringParams() : enableRealTimeAnalysis(true), enableBehaviorAnalysis(true),
                           enableThreatDetection(true), enableNetworkMonitoring(true),
                           enableFileSystemWatching(true), enableRegistryMonitoring(true),
                           enableProcessMonitoring(true), enableMemoryScanning(true),
                           suspiciousProcessThreshold(5), networkConnectionThreshold(100),
                           fileSystemChangeThreshold(50), memoryUsageThreshold(85.0),
                           cpuUsageThreshold(90.0), processCheckIntervalMs(1000),
                           networkCheckIntervalMs(5000), memoryCheckIntervalMs(2000),
                           fileSystemCheckIntervalMs(3000), maxEventHistory(100000),
                           maxTimelineEvents(50000), persistToDatabase(true) {}
};

// Memory dump parameters
struct MemoryDumpParams {
    QString outputPath;                 // Dump file output path
    bool includeBitmap;                // Include bitmap of allocated pages
    bool includeDrivers;               // Include driver memory
    bool includeUserSpace;             // Include user space memory
    bool includeKernelSpace;           // Include kernel space memory
    bool includeHardwareRegs;          // Include hardware registers
    bool compressDump;                 // Compress dump file
    bool encryptDump;                  // Encrypt dump file
    QString encryptionPassword;        // Dump encryption password
    bool verifyIntegrity;              // Calculate checksums
    bool includeMetadata;              // Include system metadata
    
    // Selective dumping
    std::vector<qint64> targetProcessIds; // Specific processes to dump
    std::vector<QString> targetProcessNames; // Process names to dump
    qint64 memoryRangeStart;           // Start address for partial dump
    qint64 memoryRangeEnd;             // End address for partial dump
    bool dumpVolatileOnly;             // Only volatile memory
    bool dumpNonPagedPool;             // Include non-paged pool
    bool dumpPagedPool;                // Include paged pool
    
    MemoryDumpParams() : includeBitmap(true), includeDrivers(true),
                        includeUserSpace(true), includeKernelSpace(true),
                        includeHardwareRegs(false), compressDump(true),
                        encryptDump(false), verifyIntegrity(true),
                        includeMetadata(true), memoryRangeStart(0),
                        memoryRangeEnd(0), dumpVolatileOnly(false),
                        dumpNonPagedPool(true), dumpPagedPool(true) {}
};

// Forward declarations
class VolatilityEngine;
class ProcessAnalyzer;
class NetworkAnalyzer;
class RegistryAnalyzer;
class MalwareDetector;
class CryptoScanner;
class TimelineReconstructor;
class LiveMonitor;

/*
 * Advanced memory forensics engine
 * מנוע פורנזיקה מתקדם לזיכרון
 */
class PHOENIXDRS_EXPORT MemoryForensicsEngine : public QObject
{
    Q_OBJECT

public:
    explicit MemoryForensicsEngine(QObject* parent = nullptr);
    ~MemoryForensicsEngine() override;

    // Initialization
    bool initialize();
    void shutdown();
    bool isInitialized() const { return m_isInitialized; }
    
    // Memory dump analysis
    bool analyzeMemoryDump(const QString& dumpFilePath);
    bool analyzeLiveMemory();
    MemoryAnalysisResult getAnalysisResult(const QString& analysisId) const;
    std::vector<MemoryAnalysisResult> getAllAnalysisResults() const;
    
    // Live memory operations
    bool createMemoryDump(const MemoryDumpParams& params);
    bool startLiveMonitoring(const LiveMonitoringParams& params);
    void stopLiveMonitoring();
    bool isMonitoringActive() const { return m_isMonitoring.load(); }
    
    // Process analysis
    struct ProcessInfo {
        qint64 processId;
        QString processName;
        QString executablePath;
        qint64 parentProcessId;
        QDateTime creationTime;
        qint64 memoryUsage;
        double cpuUsage;
        QString commandLine;
        QString userName;
        bool isHidden;
        bool isSuspicious;
        bool isSystem;
        QString architecture;
        QJsonArray loadedModules;
        QJsonArray openHandles;
        QJsonArray networkConnections;
        QString integrityLevel;
        bool hasDebugPrivileges;
        bool isPackedExecutable;
        QString malwareFamily;
        double suspiciousScore;
        
        ProcessInfo() : processId(0), parentProcessId(0), memoryUsage(0),
                       cpuUsage(0.0), isHidden(false), isSuspicious(false),
                       isSystem(false), hasDebugPrivileges(false),
                       isPackedExecutable(false), suspiciousScore(0.0) {}
    };
    
    std::vector<ProcessInfo> getRunningProcesses() const;
    std::vector<ProcessInfo> getHiddenProcesses() const;
    std::vector<ProcessInfo> getSuspiciousProcesses() const;
    ProcessInfo getProcessInfo(qint64 processId) const;
    bool terminateProcess(qint64 processId, bool force = false);
    bool suspendProcess(qint64 processId);
    bool resumeProcess(qint64 processId);
    
    // Memory region analysis
    struct MemoryRegion {
        qint64 baseAddress;
        qint64 size;
        MemoryRegionType type;
        QString protection;
        QString regionState;
        qint64 processId;
        QString moduleName;
        bool isExecutable;
        bool isWritable;
        bool isReadable;
        bool hasAnomalies;
        double entropy;
        QByteArray signature;
        QString description;
        
        MemoryRegion() : baseAddress(0), size(0), type(MemoryRegionType::UNKNOWN),
                        processId(0), isExecutable(false), isWritable(false),
                        isReadable(false), hasAnomalies(false), entropy(0.0) {}
    };
    
    std::vector<MemoryRegion> getMemoryRegions(qint64 processId = 0) const;
    std::vector<MemoryRegion> getAnomalousRegions() const;
    QByteArray readMemoryRegion(qint64 baseAddress, qint64 size, qint64 processId = 0) const;
    bool writeMemoryRegion(qint64 baseAddress, const QByteArray& data, qint64 processId = 0);
    
    // Network forensics
    struct NetworkConnection {
        QString protocol;
        QString localAddress;
        int localPort;
        QString remoteAddress;
        int remotePort;
        QString state;
        qint64 processId;
        QString processName;
        QDateTime establishedTime;
        qint64 bytesReceived;
        qint64 bytesSent;
        bool isSuspicious;
        QString geoLocation;
        QString reputation;
        bool isEncrypted;
        QString tlsVersion;
        
        NetworkConnection() : localPort(0), remotePort(0), processId(0),
                             bytesReceived(0), bytesSent(0), isSuspicious(false),
                             isEncrypted(false) {}
    };
    
    std::vector<NetworkConnection> getNetworkConnections() const;
    std::vector<NetworkConnection> getSuspiciousConnections() const;
    std::vector<QString> getDnsHistory() const;
    bool blockNetworkConnection(const QString& remoteAddress, int port);
    
    // Registry analysis (Windows)
    struct RegistryArtifact {
        QString keyPath;
        QString valueName;
        QString valueData;
        QString valueType;
        QDateTime lastModified;
        bool isAutoStart;
        bool isSuspicious;
        QString category;
        QString description;
        double riskScore;
        
        RegistryArtifact() : isAutoStart(false), isSuspicious(false), riskScore(0.0) {}
    };
    
    std::vector<RegistryArtifact> getRegistryArtifacts() const;
    std::vector<RegistryArtifact> getAutoStartEntries() const;
    std::vector<RegistryArtifact> getSuspiciousRegistry() const;
    
    // String and pattern extraction
    struct ExtractedString {
        QString content;
        qint64 memoryAddress;
        qint64 processId;
        QString encoding;
        QString category;
        bool isPotentiallyMalicious;
        double relevanceScore;
        
        ExtractedString() : memoryAddress(0), processId(0),
                           isPotentiallyMalicious(false), relevanceScore(0.0) {}
    };
    
    std::vector<ExtractedString> extractStrings(int minLength = 4, const QString& encoding = "utf-8") const;
    std::vector<ExtractedString> searchMemoryPatterns(const QByteArray& pattern) const;
    std::vector<ExtractedString> extractCredentials() const;
    std::vector<ExtractedString> extractCryptographicKeys() const;
    std::vector<ExtractedString> extractNetworkArtifacts() const;
    
    // Malware detection
    struct MalwareIndicator {
        QString indicatorType;
        QString indicatorValue;
        QString malwareFamily;
        QString description;
        double confidence;
        qint64 memoryAddress;
        qint64 processId;
        QString detectionMethod;
        QDateTime detectionTime;
        
        MalwareIndicator() : confidence(0.0), memoryAddress(0), processId(0) {}
    };
    
    std::vector<MalwareIndicator> scanForMalware() const;
    std::vector<MalwareIndicator> detectRootkits() const;
    std::vector<MalwareIndicator> detectCodeInjection() const;
    std::vector<MalwareIndicator> detectPackedExecutables() const;
    bool quarantineMalware(qint64 processId);
    
    // Timeline reconstruction
    struct TimelineEvent {
        QDateTime timestamp;
        QString eventType;
        QString description;
        qint64 processId;
        QString processName;
        QString userName;
        QJsonObject metadata;
        QString evidenceSource;
        double reliability;
        
        TimelineEvent() : processId(0), reliability(0.0) {}
    };
    
    std::vector<TimelineEvent> reconstructTimeline() const;
    std::vector<TimelineEvent> getUserActivityTimeline() const;
    std::vector<TimelineEvent> getNetworkActivityTimeline() const;
    std::vector<TimelineEvent> getFileSystemTimeline() const;
    
    // Advanced analysis
    bool detectAntiForensics() const;
    bool detectHypervisorPresence() const;
    bool detectDebuggingActivity() const;
    std::vector<QString> identifyEncryptedRegions() const;
    std::vector<QString> recoverDeletedFiles() const;
    QJsonObject performBehavioralAnalysis() const;
    
    // Configuration
    void setLiveMonitoringParams(const LiveMonitoringParams& params);
    LiveMonitoringParams getLiveMonitoringParams() const { return m_liveParams; }
    void setMemoryDumpParams(const MemoryDumpParams& params);
    MemoryDumpParams getMemoryDumpParams() const { return m_dumpParams; }
    
    // Export and reporting
    bool exportAnalysisReport(const QString& filePath, const QString& format = "json");
    bool exportTimelineReport(const QString& filePath);
    bool exportMemoryMap(const QString& filePath);
    bool exportProcessTree(const QString& filePath);
    QJsonObject generateForensicReport() const;
    
    // System requirements check
    static bool checkSystemRequirements();
    static bool hasAdministratorPrivileges();
    static bool isDebuggerPresent();
    static QString getSystemArchitecture();
    static qint64 getTotalPhysicalMemory();
    static qint64 getAvailablePhysicalMemory();

signals:
    void analysisStarted(const QString& analysisId);
    void analysisProgress(const QString& analysisId, int progressPercent);
    void analysisCompleted(const QString& analysisId, bool success);
    void memoryDumpCreated(const QString& dumpPath, qint64 dumpSize);
    void liveMonitoringStarted();
    void liveMonitoringStopped();
    void processDetected(const ProcessInfo& process);
    void suspiciousProcessDetected(const ProcessInfo& process);
    void hiddenProcessDetected(const ProcessInfo& process);
    void malwareDetected(const MalwareIndicator& indicator);
    void networkConnectionDetected(const NetworkConnection& connection);
    void suspiciousNetworkActivity(const NetworkConnection& connection);
    void registryChangeDetected(const RegistryArtifact& artifact);
    void timelineEventDetected(const TimelineEvent& event);
    void antiForensicsDetected(const QString& technique);
    void errorOccurred(const QString& error);

private:
    // Core functionality
    bool initializeVolatilityFramework();
    bool loadMemoryAnalysisPlugins();
    void cleanupResources();
    
    // Memory dump processing
    bool validateMemoryDump(const QString& dumpPath);
    QString detectDumpFormat(const QString& dumpPath);
    bool parseMemoryDump(const QString& dumpPath);
    
    // Live memory access
    bool attachToLiveSystem();
    bool detachFromLiveSystem();
    bool acquireSystemPrivileges();
    
    // Analysis implementation
    void analyzeProcessList();
    void analyzeNetworkConnections();
    void analyzeRegistryArtifacts();
    void analyzeLoadedModules();
    void analyzeMemoryRegions();
    void extractArtifacts();
    void performMalwareScanning();
    void reconstructActivityTimeline();
    
    // Platform-specific implementations
#ifdef Q_OS_WIN
    bool initializeWindowsAnalysis();
    std::vector<ProcessInfo> getWindowsProcesses() const;
    std::vector<NetworkConnection> getWindowsConnections() const;
    std::vector<RegistryArtifact> getWindowsRegistry() const;
    bool createWindowsMemoryDump(const QString& outputPath);
#endif

#ifdef Q_OS_LINUX
    bool initializeLinuxAnalysis();
    std::vector<ProcessInfo> getLinuxProcesses() const;
    std::vector<NetworkConnection> getLinuxConnections() const;
    bool createLinuxMemoryDump(const QString& outputPath);
#endif

#ifdef Q_OS_MACOS
    bool initializeMacOSAnalysis();
    std::vector<ProcessInfo> getMacOSProcesses() const;
    std::vector<NetworkConnection> getMacOSConnections() const;
    bool createMacOSMemoryDump(const QString& outputPath);
#endif

    // Member variables
    std::atomic<bool> m_isInitialized{false};
    std::atomic<bool> m_isAnalyzing{false};
    std::atomic<bool> m_isMonitoring{false};
    std::atomic<bool> m_shouldCancel{false};
    
    LiveMonitoringParams m_liveParams;
    MemoryDumpParams m_dumpParams;
    
    // Analysis components
    std::unique_ptr<VolatilityEngine> m_volatilityEngine;
    std::unique_ptr<ProcessAnalyzer> m_processAnalyzer;
    std::unique_ptr<NetworkAnalyzer> m_networkAnalyzer;
    std::unique_ptr<RegistryAnalyzer> m_registryAnalyzer;
    std::unique_ptr<MalwareDetector> m_malwareDetector;
    std::unique_ptr<CryptoScanner> m_cryptoScanner;
    std::unique_ptr<TimelineReconstructor> m_timelineReconstructor;
    std::unique_ptr<LiveMonitor> m_liveMonitor;
    
    // Data storage
    std::unordered_map<QString, MemoryAnalysisResult> m_analysisResults;
    std::vector<ProcessInfo> m_processCache;
    std::vector<NetworkConnection> m_networkCache;
    std::vector<RegistryArtifact> m_registryCache;
    std::vector<TimelineEvent> m_timelineCache;
    std::vector<MalwareIndicator> m_malwareCache;
    
    // Thread safety
    mutable QMutex m_analysisMutex;
    mutable QMutex m_cacheMutex;
    
    // System state
    QString m_currentAnalysisId;
    QString m_currentDumpPath;
    bool m_hasSystemPrivileges{false};
    QString m_systemArchitecture;
    qint64 m_totalPhysicalMemory{0};
    
    // Constants
    static constexpr int MAX_ANALYSIS_RESULTS = 100;
    static constexpr int MAX_CACHE_ENTRIES = 50000;
    static constexpr int DEFAULT_STRING_MIN_LENGTH = 4;
    static constexpr double SUSPICIOUS_ENTROPY_THRESHOLD = 7.5;
};

} // namespace PhoenixDRS