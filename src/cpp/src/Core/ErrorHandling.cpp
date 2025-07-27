/*
 * PhoenixDRS Professional - Enterprise Error Handling Implementation
 * מימוש ניהול שגיאות ארגוני - PhoenixDRS מקצועי
 */

#include "../include/Core/ErrorHandling.h"
#include "../include/ForensicLogger.h"

#include <QtCore/QCoreApplication>
#include <QtCore/QDateTime>
#include <QtCore/QStandardPaths>
#include <QtCore/QDir>
#include <QtCore/QJsonDocument>
#include <QtCore/QJsonObject>
#include <QtCore/QFileInfo>
#include <QtCore/QMutexLocker>

#include <memory>
#include <chrono>

namespace PhoenixDRS {
namespace Core {

// PhoenixException Implementation
PhoenixException::PhoenixException(ErrorCode code, const QString& message, const QString& context)
    : m_errorCode(code)
    , m_message(message)
    , m_context(context)
    , m_timestamp(QDateTime::currentDateTime())
    , m_threadId(reinterpret_cast<qint64>(QThread::currentThreadId()))
{
    constructDetailedMessage();
}

PhoenixException::PhoenixException(const PhoenixException& other)
    : std::exception(other)
    , m_errorCode(other.m_errorCode)
    , m_message(other.m_message)
    , m_context(other.m_context)
    , m_timestamp(other.m_timestamp)
    , m_threadId(other.m_threadId)
    , m_detailedMessage(other.m_detailedMessage)
{
}

PhoenixException& PhoenixException::operator=(const PhoenixException& other)
{
    if (this != &other) {
        std::exception::operator=(other);
        m_errorCode = other.m_errorCode;
        m_message = other.m_message;
        m_context = other.m_context;
        m_timestamp = other.m_timestamp;
        m_threadId = other.m_threadId;
        m_detailedMessage = other.m_detailedMessage;
    }
    return *this;
}

const char* PhoenixException::what() const noexcept
{
    return m_detailedMessage.toUtf8().constData();
}

QString PhoenixException::toString() const
{
    QJsonObject errorJson;
    errorJson["error_code"] = static_cast<int>(m_errorCode);
    errorJson["error_name"] = errorCodeToString(m_errorCode);
    errorJson["message"] = m_message;
    errorJson["context"] = m_context;
    errorJson["timestamp"] = m_timestamp.toString(Qt::ISODate);
    errorJson["thread_id"] = m_threadId;
    
    return QJsonDocument(errorJson).toJson(QJsonDocument::Compact);
}

void PhoenixException::constructDetailedMessage()
{
    m_detailedMessage = QString("[%1] %2: %3")
        .arg(errorCodeToString(m_errorCode))
        .arg(m_context.isEmpty() ? "PhoenixDRS" : m_context)
        .arg(m_message);
}

QString PhoenixException::errorCodeToString(ErrorCode code)
{
    switch (code) {
        case ErrorCode::Success: return "SUCCESS";
        case ErrorCode::UnknownError: return "UNKNOWN_ERROR";
        case ErrorCode::InvalidParameter: return "INVALID_PARAMETER";
        case ErrorCode::FileNotFound: return "FILE_NOT_FOUND";
        case ErrorCode::FileAccessError: return "FILE_ACCESS_ERROR";
        case ErrorCode::AccessDenied: return "ACCESS_DENIED";
        case ErrorCode::OutOfMemory: return "OUT_OF_MEMORY";
        case ErrorCode::DiskFull: return "DISK_FULL";
        case ErrorCode::NetworkError: return "NETWORK_ERROR";
        case ErrorCode::DatabaseError: return "DATABASE_ERROR";
        case ErrorCode::ConfigurationError: return "CONFIGURATION_ERROR";
        case ErrorCode::InitializationError: return "INITIALIZATION_ERROR";
        case ErrorCode::OperationCancelled: return "OPERATION_CANCELLED";
        case ErrorCode::TimeoutError: return "TIMEOUT_ERROR";
        case ErrorCode::ValidationError: return "VALIDATION_ERROR";
        case ErrorCode::LicenseError: return "LICENSE_ERROR";
        case ErrorCode::SystemError: return "SYSTEM_ERROR";
        case ErrorCode::InternalError: return "INTERNAL_ERROR";
        case ErrorCode::NotImplemented: return "NOT_IMPLEMENTED";
        case ErrorCode::VersionMismatch: return "VERSION_MISMATCH";
        case ErrorCode::CorruptedData: return "CORRUPTED_DATA";
        default: return "UNKNOWN";
    }
}

// ErrorManager Implementation
class ErrorManagerPrivate
{
public:
    ErrorManagerPrivate()
        : maxErrorHistory(1000)
        , errorReportingEnabled(true)
        , crashDumpEnabled(true)
    {
        crashDumpDirectory = QStandardPaths::writableLocation(QStandardPaths::AppDataLocation) + "/crashdumps";
        QDir().mkpath(crashDumpDirectory);
    }

    struct ErrorRecord {
        ErrorCode errorCode;
        QString message;
        QString context;
        QDateTime timestamp;
        qint64 threadId;
        QString stackTrace;
        int occurrenceCount = 1;
    };

    QList<ErrorRecord> errorHistory;
    QMutex historyMutex;
    
    int maxErrorHistory;
    bool errorReportingEnabled;
    bool crashDumpEnabled;
    QString crashDumpDirectory;
    
    std::function<void(const PhoenixException&)> customErrorHandler;
    std::function<void(const QString&)> crashHandler;
    
    // Error statistics
    QMap<ErrorCode, int> errorFrequency;
    QDateTime lastErrorTime;
    int totalErrors = 0;
};

ErrorManager& ErrorManager::instance()
{
    static ErrorManager instance;
    return instance;
}

ErrorManager::ErrorManager()
    : d(std::make_unique<ErrorManagerPrivate>())
{
    setupSignalHandlers();
}

ErrorManager::~ErrorManager() = default;

void ErrorManager::reportError(const PhoenixException& exception)
{
    QMutexLocker locker(&d->historyMutex);
    
    // Update statistics
    d->totalErrors++;
    d->errorFrequency[exception.errorCode()]++;
    d->lastErrorTime = QDateTime::currentDateTime();
    
    // Check if this error already exists in recent history
    bool foundDuplicate = false;
    for (auto& record : d->errorHistory) {
        if (record.errorCode == exception.errorCode() && 
            record.message == exception.message() &&
            record.context == exception.context()) {
            record.occurrenceCount++;
            record.timestamp = exception.timestamp();
            foundDuplicate = true;
            break;
        }
    }
    
    if (!foundDuplicate) {
        ErrorManagerPrivate::ErrorRecord record;
        record.errorCode = exception.errorCode();
        record.message = exception.message();
        record.context = exception.context();
        record.timestamp = exception.timestamp();
        record.threadId = exception.threadId();
        record.stackTrace = captureStackTrace();
        
        d->errorHistory.append(record);
        
        // Maintain maximum history size
        while (d->errorHistory.size() > d->maxErrorHistory) {
            d->errorHistory.removeFirst();
        }
    }
    
    // Log the error
    if (ForensicLogger::instance()) {
        ForensicLogger::instance()->logError(exception.toString());
    }
    
    // Call custom error handler if set
    if (d->customErrorHandler) {
        try {
            d->customErrorHandler(exception);
        } catch (...) {
            // Prevent recursive error handling
        }
    }
    
    // For critical errors, consider generating crash dump
    if (isCriticalError(exception.errorCode()) && d->crashDumpEnabled) {
        generateCrashDump(exception.toString());
    }
}

void ErrorManager::setCustomErrorHandler(std::function<void(const PhoenixException&)> handler)
{
    d->customErrorHandler = std::move(handler);
}

void ErrorManager::setCrashHandler(std::function<void(const QString&)> handler)
{
    d->crashHandler = std::move(handler);
}

QList<ErrorManager::ErrorSummary> ErrorManager::getErrorHistory() const
{
    QMutexLocker locker(&d->historyMutex);
    QList<ErrorSummary> summaries;
    
    for (const auto& record : d->errorHistory) {
        ErrorSummary summary;
        summary.errorCode = record.errorCode;
        summary.message = record.message;
        summary.context = record.context;
        summary.timestamp = record.timestamp;
        summary.occurrenceCount = record.occurrenceCount;
        summaries.append(summary);
    }
    
    return summaries;
}

QJsonObject ErrorManager::getErrorStatistics() const
{
    QMutexLocker locker(&d->historyMutex);
    
    QJsonObject stats;
    stats["total_errors"] = d->totalErrors;
    stats["unique_errors"] = d->errorHistory.size();
    stats["last_error_time"] = d->lastErrorTime.toString(Qt::ISODate);
    
    QJsonObject frequency;
    for (auto it = d->errorFrequency.begin(); it != d->errorFrequency.end(); ++it) {
        frequency[PhoenixException::errorCodeToString(it.key())] = it.value();
    }
    stats["error_frequency"] = frequency;
    
    // Most common errors
    QJsonArray topErrors;
    auto sortedErrors = d->errorFrequency.toStdMap();
    for (auto it = sortedErrors.rbegin(); it != sortedErrors.rend() && topErrors.size() < 5; ++it) {
        QJsonObject errorInfo;
        errorInfo["error_code"] = PhoenixException::errorCodeToString(it->first);
        errorInfo["count"] = it->second;
        topErrors.append(errorInfo);
    }
    stats["top_errors"] = topErrors;
    
    return stats;
}

void ErrorManager::clearErrorHistory()
{
    QMutexLocker locker(&d->historyMutex);
    d->errorHistory.clear();
    d->errorFrequency.clear();
    d->totalErrors = 0;
}

void ErrorManager::setMaxErrorHistory(int maxHistory)
{
    QMutexLocker locker(&d->historyMutex);
    d->maxErrorHistory = maxHistory;
    
    while (d->errorHistory.size() > d->maxErrorHistory) {
        d->errorHistory.removeFirst();
    }
}

void ErrorManager::setErrorReportingEnabled(bool enabled)
{
    d->errorReportingEnabled = enabled;
}

void ErrorManager::setCrashDumpEnabled(bool enabled)
{
    d->crashDumpEnabled = enabled;
}

void ErrorManager::setCrashDumpDirectory(const QString& directory)
{
    d->crashDumpDirectory = directory;
    QDir().mkpath(directory);
}

bool ErrorManager::isCriticalError(ErrorCode code)
{
    switch (code) {
        case ErrorCode::OutOfMemory:
        case ErrorCode::SystemError:
        case ErrorCode::InternalError:
        case ErrorCode::CorruptedData:
            return true;
        default:
            return false;
    }
}

QString ErrorManager::captureStackTrace()
{
    // Platform-specific stack trace capture would go here
    // For now, return basic information
    QString stackTrace = QString("Thread: %1, Time: %2")
        .arg(reinterpret_cast<qint64>(QThread::currentThreadId()))
        .arg(QDateTime::currentDateTime().toString(Qt::ISODate));
    
    return stackTrace;
}

void ErrorManager::generateCrashDump(const QString& errorInfo)
{
    if (!d->crashDumpEnabled) {
        return;
    }
    
    QString fileName = QString("crashdump_%1.json")
        .arg(QDateTime::currentDateTime().toString("yyyyMMdd_hhmmss"));
    
    QString filePath = QDir(d->crashDumpDirectory).filePath(fileName);
    
    QJsonObject crashInfo;
    crashInfo["timestamp"] = QDateTime::currentDateTime().toString(Qt::ISODate);
    crashInfo["application"] = QCoreApplication::applicationName();
    crashInfo["version"] = QCoreApplication::applicationVersion();
    crashInfo["error_info"] = errorInfo;
    crashInfo["system_info"] = getSystemInfo();
    crashInfo["error_history"] = QJsonArray(); // Could include recent errors
    
    QFile file(filePath);
    if (file.open(QIODevice::WriteOnly)) {
        file.write(QJsonDocument(crashInfo).toJson());
        file.close();
    }
    
    if (ForensicLogger::instance()) {
        ForensicLogger::instance()->logCritical(QString("Crash dump generated: %1").arg(filePath));
    }
}

QJsonObject ErrorManager::getSystemInfo()
{
    QJsonObject sysInfo;
    sysInfo["os"] = QSysInfo::prettyProductName();
    sysInfo["architecture"] = QSysInfo::currentCpuArchitecture();
    sysInfo["kernel"] = QSysInfo::kernelType() + " " + QSysInfo::kernelVersion();
    sysInfo["qt_version"] = QT_VERSION_STR;
    
    return sysInfo;
}

void ErrorManager::setupSignalHandlers()
{
    // Platform-specific signal handling setup would go here
    // This would handle SIGSEGV, SIGABRT, etc. on Unix systems
    // and structured exception handling on Windows
}

// Utility functions
void handleError(ErrorCode code, const QString& message, const QString& context)
{
    PhoenixException exception(code, message, context);
    ErrorManager::instance().reportError(exception);
    throw exception;
}

void logError(ErrorCode code, const QString& message, const QString& context)
{
    PhoenixException exception(code, message, context);
    ErrorManager::instance().reportError(exception);
    // Don't throw, just log
}

void assertCondition(bool condition, const QString& message, const QString& context)
{
    if (!condition) {
        handleError(ErrorCode::ValidationError, 
                   QString("Assertion failed: %1").arg(message), 
                   context);
    }
}

void validateParameter(bool condition, const QString& paramName, const QString& context)
{
    if (!condition) {
        handleError(ErrorCode::InvalidParameter,
                   QString("Invalid parameter: %1").arg(paramName),
                   context);
    }
}

} // namespace Core
} // namespace PhoenixDRS