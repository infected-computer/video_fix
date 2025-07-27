/*
 * PhoenixDRS Professional - Advanced Memory Forensics Engine Implementation
 * מימוש מנוע פורנזיקה מתקדם לזיכרון - PhoenixDRS מקצועי
 */

#include "../include/MemoryForensicsEngine.h"
#include "../include/Core/ErrorHandling.h"
#include "../include/Core/MemoryManager.h"
#include "../include/ForensicLogger.h"

#include <QtCore/QCoreApplication>
#include <QtCore/QStandardPaths>
#include <QtCore/QDir>
#include <QtCore/QFileInfo>
#include <QtCore/QMutexLocker>
#include <QtCore/QTimer>
#include <QtCore/QJsonDocument>
#include <QtCore/QJsonObject>
#include <QtCore/QJsonArray>
#include <QtCore/QCryptographicHash>
#include <QtCore/QRegularExpression>

#include <algorithm>
#include <chrono>
#include <iomanip>
#include <sstream>

#ifdef Q_OS_WIN
#include <windows.h>
#include <psapi.h>
#include <tlhelp32.h>
#include <winternl.h>
#include <dbghelp.h>
#endif

#ifdef Q_OS_LINUX
#include <sys/ptrace.h>
#include <sys/wait.h>
#include <sys/user.h>
#include <unistd.h>
#include <dirent.h>
#include <proc/readproc.h>
#endif

namespace PhoenixDRS {
namespace Forensics {

// Memory Forensics Worker Class
class MemoryForensicsEngine::MemoryAnalysisWorker : public QObject
{
    Q_OBJECT

public:
    explicit MemoryAnalysisWorker(MemoryForensicsEngine* parent, const MemoryAnalysisParameters& params)
        : QObject(nullptr), m_engine(parent), m_params(params) {}

public slots:
    void performAnalysis();

signals:
    void finished();
    void error(const QString& message);
    void progressUpdate(int percentage);
    void processFound(const ProcessInfo& process);
    void stringFound(const MemoryString& string);
    void artifactFound(const MemoryArtifact& artifact);

private:
    MemoryForensicsEngine* m_engine;
    MemoryAnalysisParameters m_params;
};

// MemoryForensicsEngine Implementation
MemoryForensicsEngine& MemoryForensicsEngine::instance()
{
    static MemoryForensicsEngine instance;
    return instance;
}

MemoryForensicsEngine::MemoryForensicsEngine(QObject* parent)
    : QObject(parent)
    , m_isRunning(false)
    , m_currentProgress(0)
    , m_totalProcesses(0)
    , m_analyzedProcesses(0)
    , m_foundArtifacts(0)
    , m_workerThread(nullptr)
{
    setupSignatures();
    setupAnalysisProfiles();
    
    // Initialize performance monitoring
    m_performanceTimer = std::make_unique<QTimer>();
    connect(m_performanceTimer.get(), &QTimer::timeout, this, &MemoryForensicsEngine::updatePerformanceMetrics);
    m_performanceTimer->start(1000); // Update every second
    
    ForensicLogger::instance()->logInfo("MemoryForensicsEngine initialized");
}

MemoryForensicsEngine::~MemoryForensicsEngine()
{
    if (m_isRunning.load()) {
        cancelAnalysis();
    }
    cleanup();
}

bool MemoryForensicsEngine::analyzeMemoryDump(const QString& dumpFilePath, const QString& outputDirectory, const MemoryAnalysisParameters& params)
{
    if (m_isRunning.load()) {
        emit error("Memory analysis already in progress");
        return false;
    }

    // Validate input parameters
    if (!validateAnalysisParameters(dumpFilePath, outputDirectory, params)) {
        return false;
    }

    // Setup analysis environment
    if (!setupAnalysisEnvironment(outputDirectory)) {
        return false;
    }

    // Start analysis in separate thread
    m_workerThread = QThread::create([this, dumpFilePath, outputDirectory, params]() {
        performMemoryDumpAnalysis(dumpFilePath, outputDirectory, params);
    });

    connect(m_workerThread, &QThread::finished, this, &MemoryForensicsEngine::onAnalysisFinished);
    
    m_isRunning = true;
    m_workerThread->start();
    
    emit analysisStarted();
    ForensicLogger::instance()->logInfo(QString("Memory dump analysis started: %1").arg(dumpFilePath));
    
    return true;
}

bool MemoryForensicsEngine::analyzeLiveMemory(const MemoryAnalysisParameters& params)
{
    if (m_isRunning.load()) {
        emit error("Memory analysis already in progress");
        return false;
    }

#ifdef Q_OS_WIN
    return performWindowsLiveAnalysis(params);
#elif defined(Q_OS_LINUX)
    return performLinuxLiveAnalysis(params);
#else
    emit error("Live memory analysis not supported on this platform");
    return false;
#endif
}

bool MemoryForensicsEngine::extractProcessMemory(qint64 processId, const QString& outputDirectory)
{
    try {
#ifdef Q_OS_WIN
        return extractWindowsProcessMemory(processId, outputDirectory);
#elif defined(Q_OS_LINUX)
        return extractLinuxProcessMemory(processId, outputDirectory);
#else
        throw PhoenixDRS::Core::PhoenixException(
            PhoenixDRS::Core::ErrorCode::NotImplemented,
            "Process memory extraction not supported on this platform",
            "MemoryForensicsEngine::extractProcessMemory"
        );
#endif
    } catch (const PhoenixDRS::Core::PhoenixException& e) {
        emit error(e.message());
        return false;
    }
}

QList<ProcessInfo> MemoryForensicsEngine::getRunningProcesses()
{
    QList<ProcessInfo> processes;
    
    try {
#ifdef Q_OS_WIN
        processes = getWindowsProcesses();
#elif defined(Q_OS_LINUX)
        processes = getLinuxProcesses();
#endif
        
        // Sort by PID
        std::sort(processes.begin(), processes.end(), 
                 [](const ProcessInfo& a, const ProcessInfo& b) {
                     return a.pid < b.pid;
                 });
        
    } catch (const PhoenixDRS::Core::PhoenixException& e) {
        ForensicLogger::instance()->logError(e.message());
    }
    
    return processes;
}

QList<MemoryString> MemoryForensicsEngine::extractStrings(const QString& memorySource, const StringExtractionOptions& options)
{
    QList<MemoryString> strings;
    
    try {
        if (QFileInfo::exists(memorySource)) {
            // File-based string extraction
            strings = extractStringsFromFile(memorySource, options);
        } else {
            // Live memory string extraction
            qint64 pid = memorySource.toLongLong();
            if (pid > 0) {
                strings = extractStringsFromProcess(pid, options);
            }
        }
        
        // Filter and sort results
        filterStrings(strings, options);
        
    } catch (const PhoenixDRS::Core::PhoenixException& e) {
        ForensicLogger::instance()->logError(e.message());
    }
    
    return strings;
}

QList<NetworkConnection> MemoryForensicsEngine::getNetworkConnections()
{
    QList<NetworkConnection> connections;
    
    try {
#ifdef Q_OS_WIN
        connections = getWindowsNetworkConnections();
#elif defined(Q_OS_LINUX)
        connections = getLinuxNetworkConnections();
#endif
    } catch (const PhoenixDRS::Core::PhoenixException& e) {
        ForensicLogger::instance()->logError(e.message());
    }
    
    return connections;
}

QJsonObject MemoryForensicsEngine::getAnalysisReport() const
{
    QJsonObject report;
    
    // Basic information
    report["analysis_id"] = m_currentAnalysisId;
    report["timestamp"] = QDateTime::currentDateTime().toString(Qt::ISODate);
    report["status"] = m_isRunning.load() ? "running" : "completed";
    report["progress"] = m_currentProgress.load();
    
    // Statistics
    QJsonObject stats;
    stats["total_processes"] = m_totalProcesses.load();
    stats["analyzed_processes"] = m_analyzedProcesses.load();
    stats["found_artifacts"] = m_foundArtifacts.load();
    stats["analysis_duration"] = m_analysisTimer.elapsed();
    report["statistics"] = stats;
    
    // Performance metrics
    QJsonObject performance;
    performance["memory_usage_mb"] = m_currentMemoryUsage / (1024 * 1024);
    performance["cpu_usage_percent"] = m_currentCpuUsage;
    performance["disk_io_mb_per_sec"] = m_currentDiskIo / (1024 * 1024);
    report["performance"] = performance;
    
    // Findings summary
    QJsonArray findings;
    for (const auto& artifact : m_foundArtifacts) {
        QJsonObject artifactObj;
        artifactObj["type"] = static_cast<int>(artifact.type);
        artifactObj["description"] = artifact.description;
        artifactObj["confidence"] = artifact.confidence;
        artifactObj["location"] = artifact.location;
        findings.append(artifactObj);
    }
    report["findings"] = findings;
    
    return report;
}

void MemoryForensicsEngine::cancelAnalysis()
{
    if (!m_isRunning.load()) {
        return;
    }
    
    m_shouldCancel = true;
    
    if (m_workerThread && m_workerThread->isRunning()) {
        m_workerThread->requestInterruption();
        if (!m_workerThread->wait(5000)) {
            m_workerThread->terminate();
            m_workerThread->wait();
        }
    }
    
    m_isRunning = false;
    emit analysisCancelled();
    
    ForensicLogger::instance()->logInfo("Memory analysis cancelled");
}

// Private Implementation Methods

bool MemoryForensicsEngine::validateAnalysisParameters(const QString& dumpFilePath, const QString& outputDirectory, const MemoryAnalysisParameters& params)
{
    // Validate dump file
    QFileInfo dumpFile(dumpFilePath);
    if (!dumpFile.exists()) {
        emit error(QString("Memory dump file not found: %1").arg(dumpFilePath));
        return false;
    }
    
    if (!dumpFile.isReadable()) {
        emit error(QString("Cannot read memory dump file: %1").arg(dumpFilePath));
        return false;
    }
    
    // Validate output directory
    QDir outputDir(outputDirectory);
    if (!outputDir.exists()) {
        if (!outputDir.mkpath(".")) {
            emit error(QString("Cannot create output directory: %1").arg(outputDirectory));
            return false;
        }
    }
    
    // Validate memory profile
    if (params.memoryProfile.isEmpty()) {
        emit error("Memory profile cannot be empty");
        return false;
    }
    
    return true;
}

bool MemoryForensicsEngine::setupAnalysisEnvironment(const QString& outputDirectory)
{
    try {
        // Create analysis subdirectories
        QDir outputDir(outputDirectory);
        
        QStringList subdirs = {"processes", "strings", "network", "registry", "files", "malware", "timeline"};
        for (const QString& subdir : subdirs) {
            if (!outputDir.mkpath(subdir)) {
                throw PhoenixDRS::Core::PhoenixException(
                    PhoenixDRS::Core::ErrorCode::FileAccessError,
                    QString("Failed to create analysis directory: %1").arg(subdir),
                    "MemoryForensicsEngine::setupAnalysisEnvironment"
                );
            }
        }
        
        // Initialize analysis session
        m_currentAnalysisId = QUuid::createUuid().toString();
        m_analysisStartTime = QDateTime::currentDateTime();
        m_analysisTimer.start();
        
        // Reset counters
        m_currentProgress = 0;
        m_totalProcesses = 0;
        m_analyzedProcesses = 0;
        m_foundArtifacts = 0;
        m_shouldCancel = false;
        
        return true;
        
    } catch (const PhoenixDRS::Core::PhoenixException& e) {
        emit error(e.message());
        return false;
    }
}

void MemoryForensicsEngine::performMemoryDumpAnalysis(const QString& dumpFilePath, const QString& outputDirectory, const MemoryAnalysisParameters& params)
{
    try {
        emit progressUpdate(5);
        
        // Load memory dump
        if (!loadMemoryDump(dumpFilePath)) {
            return;
        }
        
        emit progressUpdate(10);
        
        // Extract process list
        if (params.analyzeProcesses) {
            extractProcessList(outputDirectory);
            emit progressUpdate(25);
        }
        
        // Extract strings
        if (params.extractStrings) {
            performStringExtraction(outputDirectory, params.stringOptions);
            emit progressUpdate(40);
        }
        
        // Analyze network connections
        if (params.analyzeNetwork) {
            analyzeNetworkArtifacts(outputDirectory);
            emit progressUpdate(55);
        }
        
        // Registry analysis (Windows only)
        if (params.analyzeRegistry && isWindowsDump()) {
            analyzeRegistryHives(outputDirectory);
            emit progressUpdate(70);
        }
        
        // Malware detection
        if (params.detectMalware) {
            performMalwareDetection(outputDirectory);
            emit progressUpdate(85);
        }
        
        // Generate timeline
        if (params.generateTimeline) {
            generateTimeline(outputDirectory);
            emit progressUpdate(95);
        }
        
        // Finalize analysis
        finalizeAnalysis(outputDirectory);
        emit progressUpdate(100);
        
        emit analysisCompleted();
        
    } catch (const PhoenixDRS::Core::PhoenixException& e) {
        emit error(e.message());
    } catch (const std::exception& e) {
        emit error(QString("System error: %1").arg(e.what()));
    }
}

#ifdef Q_OS_WIN
bool MemoryForensicsEngine::performWindowsLiveAnalysis(const MemoryAnalysisParameters& params)
{
    try {
        // Enable debug privileges
        if (!enableDebugPrivileges()) {
            throw PhoenixDRS::Core::PhoenixException(
                PhoenixDRS::Core::ErrorCode::AccessDenied,
                "Failed to enable debug privileges. Run as administrator.",
                "MemoryForensicsEngine::performWindowsLiveAnalysis"
            );
        }
        
        // Get running processes
        auto processes = getWindowsProcesses();
        m_totalProcesses = processes.size();
        
        // Analyze each process
        for (const auto& process : processes) {
            if (m_shouldCancel.load()) {
                break;
            }
            
            analyzeWindowsProcess(process, params);
            m_analyzedProcesses++;
            
            int progress = (m_analyzedProcesses * 100) / m_totalProcesses;
            emit progressUpdate(progress);
        }
        
        emit analysisCompleted();
        return true;
        
    } catch (const PhoenixDRS::Core::PhoenixException& e) {
        emit error(e.message());
        return false;
    }
}

bool MemoryForensicsEngine::enableDebugPrivileges()
{
    HANDLE token;
    if (!OpenProcessToken(GetCurrentProcess(), TOKEN_ADJUST_PRIVILEGES | TOKEN_QUERY, &token)) {
        return false;
    }
    
    TOKEN_PRIVILEGES privileges;
    privileges.PrivilegeCount = 1;
    if (!LookupPrivilegeValue(nullptr, SE_DEBUG_NAME, &privileges.Privileges[0].Luid)) {
        CloseHandle(token);
        return false;
    }
    
    privileges.Privileges[0].Attributes = SE_PRIVILEGE_ENABLED;
    
    bool result = AdjustTokenPrivileges(token, FALSE, &privileges, sizeof(privileges), nullptr, nullptr);
    CloseHandle(token);
    
    return result && GetLastError() == ERROR_SUCCESS;
}

QList<ProcessInfo> MemoryForensicsEngine::getWindowsProcesses()
{
    QList<ProcessInfo> processes;
    
    HANDLE snapshot = CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0);
    if (snapshot == INVALID_HANDLE_VALUE) {
        return processes;
    }
    
    PROCESSENTRY32W entry;
    entry.dwSize = sizeof(entry);
    
    if (Process32FirstW(snapshot, &entry)) {
        do {
            ProcessInfo info;
            info.pid = entry.th32ProcessID;
            info.parentPid = entry.th32ParentProcessID;
            info.name = QString::fromWCharArray(entry.szExeFile);
            info.threadCount = entry.cntThreads;
            
            // Get additional process information
            HANDLE process = OpenProcess(PROCESS_QUERY_INFORMATION | PROCESS_VM_READ, FALSE, entry.th32ProcessID);
            if (process) {
                // Get memory usage
                PROCESS_MEMORY_COUNTERS_EX memInfo;
                if (GetProcessMemoryInfo(process, (PROCESS_MEMORY_COUNTERS*)&memInfo, sizeof(memInfo))) {
                    info.memoryUsage = memInfo.WorkingSetSize;
                }
                
                // Get process path
                WCHAR path[MAX_PATH];
                DWORD pathLen = MAX_PATH;
                if (QueryFullProcessImageNameW(process, 0, path, &pathLen)) {
                    info.path = QString::fromWCharArray(path);
                }
                
                CloseHandle(process);
            }
            
            processes.append(info);
            
        } while (Process32NextW(snapshot, &entry));
    }
    
    CloseHandle(snapshot);
    return processes;
}
#endif

#ifdef Q_OS_LINUX
bool MemoryForensicsEngine::performLinuxLiveAnalysis(const MemoryAnalysisParameters& params)
{
    try {
        // Get running processes from /proc
        auto processes = getLinuxProcesses();
        m_totalProcesses = processes.size();
        
        // Analyze each process
        for (const auto& process : processes) {
            if (m_shouldCancel.load()) {
                break;
            }
            
            analyzeLinuxProcess(process, params);
            m_analyzedProcesses++;
            
            int progress = (m_analyzedProcesses * 100) / m_totalProcesses;
            emit progressUpdate(progress);
        }
        
        emit analysisCompleted();
        return true;
        
    } catch (const PhoenixDRS::Core::PhoenixException& e) {
        emit error(e.message());
        return false;
    }
}

QList<ProcessInfo> MemoryForensicsEngine::getLinuxProcesses()
{
    QList<ProcessInfo> processes;
    
    DIR* procDir = opendir("/proc");
    if (!procDir) {
        return processes;
    }
    
    struct dirent* entry;
    while ((entry = readdir(procDir)) != nullptr) {
        if (entry->d_type != DT_DIR) {
            continue;
        }
        
        // Check if directory name is numeric (PID)
        char* endptr;
        long pid = strtol(entry->d_name, &endptr, 10);
        if (*endptr != '\0' || pid <= 0) {
            continue;
        }
        
        ProcessInfo info;
        info.pid = pid;
        
        // Read process information from /proc/[pid]/stat
        QString statPath = QString("/proc/%1/stat").arg(pid);
        QFile statFile(statPath);
        if (statFile.open(QIODevice::ReadOnly)) {
            QString statData = statFile.readAll();
            QStringList statFields = statData.split(' ');
            
            if (statFields.size() >= 4) {
                info.name = statFields[1].mid(1, statFields[1].length() - 2); // Remove parentheses
                info.parentPid = statFields[3].toLongLong();
            }
        }
        
        // Read memory information from /proc/[pid]/status
        QString statusPath = QString("/proc/%1/status").arg(pid);
        QFile statusFile(statusPath);
        if (statusFile.open(QIODevice::ReadOnly)) {
            QTextStream stream(&statusFile);
            while (!stream.atEnd()) {
                QString line = stream.readLine();
                if (line.startsWith("VmRSS:")) {
                    QStringList parts = line.split(QRegularExpression("\\s+"));
                    if (parts.size() >= 2) {
                        info.memoryUsage = parts[1].toLongLong() * 1024; // Convert KB to bytes
                    }
                    break;
                }
            }
        }
        
        processes.append(info);
    }
    
    closedir(procDir);
    return processes;
}
#endif

void MemoryForensicsEngine::setupSignatures()
{
    // Common malware signatures
    m_malwareSignatures = {
        {"PE_HEADER", QByteArray("\x4D\x5A", 2)},              // MZ header
        {"ELF_HEADER", QByteArray("\x7F\x45\x4C\x46", 4)},     // ELF header
        {"MACH_O_32", QByteArray("\xFE\xED\xFA\xCE", 4)},      // Mach-O 32-bit
        {"MACH_O_64", QByteArray("\xFE\xED\xFA\xCF", 4)},      // Mach-O 64-bit
    };
    
    // Suspicious strings
    m_suspiciousStrings = {
        "cmd.exe", "powershell.exe", "wscript.exe", "cscript.exe",
        "regsvr32.exe", "rundll32.exe", "schtasks.exe", "net.exe",
        "wget", "curl", "nc.exe", "netcat", "telnet",
        "password", "passwd", "credential", "token", "secret"
    };
}

void MemoryForensicsEngine::setupAnalysisProfiles()
{
    // Windows profiles
    m_analysisProfiles["Win10x64"] = {
        .kernelBase = 0xFFFFF80000000000ULL,
        .pageSize = 4096,
        .pointerSize = 8,
        .systemCallTable = "nt!KiServiceTable"
    };
    
    m_analysisProfiles["Win10x86"] = {
        .kernelBase = 0x80000000ULL,
        .pageSize = 4096,
        .pointerSize = 4,
        .systemCallTable = "nt!KiServiceTable"
    };
    
    // Linux profiles
    m_analysisProfiles["Linux64"] = {
        .kernelBase = 0xFFFFFFFF80000000ULL,
        .pageSize = 4096,
        .pointerSize = 8,
        .systemCallTable = "sys_call_table"
    };
}

void MemoryForensicsEngine::updatePerformanceMetrics()
{
    // Update memory usage
    m_currentMemoryUsage = PhoenixDRS::Core::MemoryManager::instance().getSystemInfo().processMemoryUsage;
    
    // Update CPU usage (simplified)
    static auto lastTime = std::chrono::steady_clock::now();
    auto currentTime = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(currentTime - lastTime).count();
    
    if (elapsed > 0) {
        m_currentCpuUsage = std::min(100.0, (m_analyzedProcesses.load() * 1000.0) / elapsed);
    }
    
    lastTime = currentTime;
}

void MemoryForensicsEngine::onAnalysisFinished()
{
    m_isRunning = false;
    
    if (m_workerThread) {
        m_workerThread->deleteLater();
        m_workerThread = nullptr;
    }
    
    ForensicLogger::instance()->logInfo("Memory analysis completed");
}

void MemoryForensicsEngine::cleanup()
{
    if (m_performanceTimer) {
        m_performanceTimer->stop();
    }
    
    // Clear analysis data
    m_foundArtifacts.clear();
    m_analysisCache.clear();
    
    ForensicLogger::instance()->logInfo("MemoryForensicsEngine cleaned up");
}

} // namespace Forensics
} // namespace PhoenixDRS

#include "MemoryForensicsEngine.moc"