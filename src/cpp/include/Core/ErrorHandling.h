/*
 * PhoenixDRS Professional - Enterprise Error Handling and Exception Management
 * ניהול שגיאות וחריגות ברמה תעשייתית - PhoenixDRS מקצועי
 */

#pragma once

#include <QtCore/QObject>
#include <QtCore/QString>
#include <QtCore/QDateTime>
#include <QtCore/QJsonObject>
#include <QtCore/QDebug>
#include <QtCore/QLoggingCategory>

#include <exception>
#include <stdexcept>
#include <memory>
#include <string>
#include <vector>
#include <unordered_map>
#include <functional>

Q_DECLARE_LOGGING_CATEGORY(phoenixCore)
Q_DECLARE_LOGGING_CATEGORY(phoenixForensics)
Q_DECLARE_LOGGING_CATEGORY(phoenixNetwork)
Q_DECLARE_LOGGING_CATEGORY(phoenixCrypto)
Q_DECLARE_LOGGING_CATEGORY(phoenixML)

namespace PhoenixDRS {
namespace Core {

// Error severity levels
enum class ErrorSeverity {
    Debug = 0,      // Debug information
    Info = 1,       // Informational messages
    Warning = 2,    // Warning conditions
    Error = 3,      // Error conditions
    Critical = 4,   // Critical conditions
    Fatal = 5       // Fatal errors - application must terminate
};

// Error categories for better classification
enum class ErrorCategory {
    Unknown = 0,
    System,             // System-level errors (OS, hardware)
    Network,            // Network-related errors
    FileSystem,         // File system operations
    Database,           // Database operations
    Memory,             // Memory allocation/management
    Threading,          // Multi-threading issues
    Security,           // Security/authentication failures
    Cryptography,       // Cryptographic operations
    Forensics,          // Forensic analysis operations
    UserInterface,      // UI-related errors
    Configuration,      // Configuration/settings errors
    Plugin,             // Plugin loading/execution
    License,            // Licensing/activation errors
    Hardware,           // Hardware-specific errors
    Performance,        // Performance-related issues
    Validation,         // Input validation errors
    Business,           // Business logic errors
    External            // External service errors
};

// Error codes for programmatic handling
enum class ErrorCode {
    Success = 0,
    
    // Generic errors (1000-1999)
    UnknownError = 1000,
    InvalidParameter = 1001,
    NullPointer = 1002,
    OutOfMemory = 1003,
    NotImplemented = 1004,
    NotSupported = 1005,
    Timeout = 1006,
    Cancelled = 1007,
    
    // File system errors (2000-2999)
    FileNotFound = 2000,
    AccessDenied = 2001,
    FileCorrupted = 2002,
    DiskFull = 2003,
    InvalidPath = 2004,
    FileInUse = 2005,
    DirectoryNotEmpty = 2006,
    
    // Network errors (3000-3999)
    NetworkUnavailable = 3000,
    ConnectionFailed = 3001,
    ConnectionTimeout = 3002,
    HostNotFound = 3003,
    InvalidUrl = 3004,
    SslError = 3005,
    AuthenticationFailed = 3006,
    
    // Database errors (4000-4999)
    DatabaseConnectionFailed = 4000,
    DatabaseQueryFailed = 4001,
    DatabaseTransactionFailed = 4002,
    DatabaseCorrupted = 4003,
    DatabaseVersionMismatch = 4004,
    
    // Cryptography errors (5000-5999)
    CryptoInitFailed = 5000,
    InvalidKey = 5001,
    DecryptionFailed = 5002,
    EncryptionFailed = 5003,
    HashCalculationFailed = 5004,
    CertificateInvalid = 5005,
    
    // Forensics errors (6000-6999)
    InvalidForensicImage = 6000,
    UnsupportedFormat = 6001,
    AnalysisFailed = 6002,
    EvidenceCorrupted = 6003,
    ChainOfCustodyBroken = 6004,
    
    // Hardware errors (7000-7999)
    HardwareNotFound = 7000,
    HardwareInitFailed = 7001,
    DeviceError = 7002,
    DriverError = 7003,
    
    // License errors (8000-8999)
    LicenseInvalid = 8000,
    LicenseExpired = 8001,
    LicenseNotFound = 8002,
    ActivationFailed = 8003,
    
    // Performance errors (9000-9999)
    PerformanceThresholdExceeded = 9000,
    ResourceExhausted = 9001,
    DeadlockDetected = 9002
};

// Forward declarations
class ErrorContext;
class ErrorReporter;

/*
 * Base exception class for PhoenixDRS
 */
class PHOENIXDRS_EXPORT PhoenixException : public std::exception
{
public:
    PhoenixException(ErrorCode code, 
                    ErrorCategory category, 
                    ErrorSeverity severity,
                    const QString& message,
                    const QString& file = QString(),
                    int line = 0,
                    const QString& function = QString());
    
    PhoenixException(const PhoenixException& other) noexcept;
    PhoenixException& operator=(const PhoenixException& other) noexcept;
    
    virtual ~PhoenixException() noexcept override = default;
    
    // std::exception interface
    const char* what() const noexcept override;
    
    // PhoenixDRS specific interface
    ErrorCode errorCode() const noexcept { return m_errorCode; }
    ErrorCategory category() const noexcept { return m_category; }
    ErrorSeverity severity() const noexcept { return m_severity; }
    const QString& message() const noexcept { return m_message; }
    const QString& file() const noexcept { return m_file; }
    int line() const noexcept { return m_line; }
    const QString& function() const noexcept { return m_function; }
    QDateTime timestamp() const noexcept { return m_timestamp; }
    
    // Additional context
    void addContext(const QString& key, const QVariant& value);
    QVariant getContext(const QString& key) const;
    QJsonObject getAllContext() const;
    
    // Stack trace support
    void captureStackTrace();
    QStringList stackTrace() const { return m_stackTrace; }
    
    // Exception chaining
    void setCause(std::shared_ptr<PhoenixException> cause);
    std::shared_ptr<PhoenixException> getCause() const { return m_cause; }
    
    // Utility methods
    QString toString() const;
    QJsonObject toJson() const;
    
    // Static factory methods
    static PhoenixException create(ErrorCode code, const QString& message);
    static PhoenixException systemError(const QString& message, int systemErrorCode = 0);
    static PhoenixException networkError(const QString& message, int networkErrorCode = 0);
    static PhoenixException fileSystemError(const QString& message, const QString& filePath = QString());
    static PhoenixException cryptographicError(const QString& message);
    static PhoenixException forensicsError(const QString& message);

private:
    ErrorCode m_errorCode;
    ErrorCategory m_category;
    ErrorSeverity m_severity;
    QString m_message;
    QString m_file;
    int m_line;
    QString m_function;
    QDateTime m_timestamp;
    QJsonObject m_context;
    QStringList m_stackTrace;
    std::shared_ptr<PhoenixException> m_cause;
    mutable std::string m_whatString; // Cache for what() method
};

/*
 * Result template for operations that can fail
 */
template<typename T>
class Result
{
public:
    // Success constructor
    Result(T&& value) noexcept 
        : m_hasValue(true), m_value(std::forward<T>(value)) {}
    
    Result(const T& value) 
        : m_hasValue(true), m_value(value) {}
    
    // Error constructor
    Result(const PhoenixException& error) 
        : m_hasValue(false), m_error(std::make_shared<PhoenixException>(error)) {}
    
    Result(PhoenixException&& error) 
        : m_hasValue(false), m_error(std::make_shared<PhoenixException>(std::move(error))) {}
    
    // Copy/Move constructors
    Result(const Result& other) = default;
    Result(Result&& other) noexcept = default;
    Result& operator=(const Result& other) = default;
    Result& operator=(Result&& other) noexcept = default;
    
    // Status checking
    bool isSuccess() const noexcept { return m_hasValue; }
    bool isError() const noexcept { return !m_hasValue; }
    explicit operator bool() const noexcept { return isSuccess(); }
    
    // Value access (throws if error)
    const T& value() const {
        if (!m_hasValue) {
            throw *m_error;
        }
        return m_value;
    }
    
    T& value() {
        if (!m_hasValue) {
            throw *m_error;
        }
        return m_value;
    }
    
    // Safe value access
    const T& valueOr(const T& defaultValue) const noexcept {
        return m_hasValue ? m_value : defaultValue;
    }
    
    // Error access
    const PhoenixException& error() const {
        if (m_hasValue) {
            throw PhoenixException::create(ErrorCode::InvalidParameter, 
                                          "Attempting to access error from successful result");
        }
        return *m_error;
    }
    
    // Functional operations
    template<typename F>
    auto map(F&& func) const -> Result<decltype(func(m_value))> {
        if (m_hasValue) {
            try {
                return Result<decltype(func(m_value))>(func(m_value));
            } catch (const PhoenixException& e) {
                return Result<decltype(func(m_value))>(e);
            } catch (const std::exception& e) {
                return Result<decltype(func(m_value))>(
                    PhoenixException::create(ErrorCode::UnknownError, QString::fromStdString(e.what())));
            }
        }
        return Result<decltype(func(m_value))>(*m_error);
    }
    
    template<typename F>
    Result<T> flatMap(F&& func) const {
        if (m_hasValue) {
            try {
                return func(m_value);
            } catch (const PhoenixException& e) {
                return Result<T>(e);
            } catch (const std::exception& e) {
                return Result<T>(PhoenixException::create(ErrorCode::UnknownError, 
                                                         QString::fromStdString(e.what())));
            }
        }
        return *this;
    }

private:
    bool m_hasValue;
    T m_value;
    std::shared_ptr<PhoenixException> m_error;
};

// Specialization for void results
template<>
class Result<void>
{
public:
    // Success constructor
    Result() noexcept : m_hasError(false) {}
    
    // Error constructor
    Result(const PhoenixException& error) 
        : m_hasError(true), m_error(std::make_shared<PhoenixException>(error)) {}
    
    Result(PhoenixException&& error) 
        : m_hasError(true), m_error(std::make_shared<PhoenixException>(std::move(error))) {}
    
    // Status checking
    bool isSuccess() const noexcept { return !m_hasError; }
    bool isError() const noexcept { return m_hasError; }
    explicit operator bool() const noexcept { return isSuccess(); }
    
    // Error access
    const PhoenixException& error() const {
        if (!m_hasError) {
            throw PhoenixException::create(ErrorCode::InvalidParameter, 
                                          "Attempting to access error from successful result");
        }
        return *m_error;
    }
    
    // Throw if error
    void throwIfError() const {
        if (m_hasError) {
            throw *m_error;
        }
    }

private:
    bool m_hasError;
    std::shared_ptr<PhoenixException> m_error;
};

/*
 * Error context for collecting additional debugging information
 */
class PHOENIXDRS_EXPORT ErrorContext
{
public:
    ErrorContext();
    ~ErrorContext();
    
    // Add context information
    void addContext(const QString& key, const QVariant& value);
    void addSystemInfo();
    void addPerformanceInfo();
    void addMemoryInfo();
    void addThreadInfo();
    
    // Get context
    QJsonObject getContext() const;
    void clear();
    
    // Stack trace utilities
    static QStringList captureStackTrace(int maxFrames = 50);
    static QString symbolFromAddress(void* address);

private:
    QJsonObject m_context;
    mutable QMutex m_mutex;
};

/*
 * Global error reporting and handling
 */
class PHOENIXDRS_EXPORT ErrorReporter : public QObject
{
    Q_OBJECT
    
public:
    static ErrorReporter& instance();
    
    // Error reporting
    void reportError(const PhoenixException& exception);
    void reportCriticalError(const PhoenixException& exception);
    void reportWarning(const QString& message, ErrorCategory category = ErrorCategory::Unknown);
    
    // Error handlers
    using ErrorHandler = std::function<void(const PhoenixException&)>;
    void registerErrorHandler(ErrorCategory category, ErrorHandler handler);
    void registerGlobalErrorHandler(ErrorHandler handler);
    
    // Error statistics
    struct ErrorStatistics {
        int totalErrors = 0;
        int criticalErrors = 0;
        int warningCount = 0;
        QDateTime lastError;
        std::unordered_map<ErrorCategory, int> errorsByCategory;
        std::unordered_map<ErrorCode, int> errorsByCode;
    };
    
    ErrorStatistics getStatistics() const;
    void resetStatistics();
    
    // Error persistence
    void enableErrorPersistence(const QString& logFilePath);
    void disableErrorPersistence();
    
    // Configuration
    void setMinimumSeverity(ErrorSeverity severity);
    ErrorSeverity getMinimumSeverity() const;
    
    void enableStackTraces(bool enable);
    bool isStackTracesEnabled() const;

signals:
    void errorOccurred(const PhoenixException& exception);
    void criticalErrorOccurred(const PhoenixException& exception);
    void warningOccurred(const QString& message, ErrorCategory category);

private:
    ErrorReporter();
    ~ErrorReporter();
    
    void writeToLog(const PhoenixException& exception);
    void notifyHandlers(const PhoenixException& exception);
    
    std::unordered_map<ErrorCategory, std::vector<ErrorHandler>> m_errorHandlers;
    std::vector<ErrorHandler> m_globalHandlers;
    ErrorStatistics m_statistics;
    QString m_logFilePath;
    bool m_persistenceEnabled;
    ErrorSeverity m_minimumSeverity;
    bool m_stackTracesEnabled;
    mutable QMutex m_mutex;
};

// Utility macros for error handling
#define PHOENIX_THROW(code, message) \
    throw PhoenixException(code, PhoenixDRS::Core::ErrorCategory::Unknown, \
                          PhoenixDRS::Core::ErrorSeverity::Error, message, \
                          __FILE__, __LINE__, Q_FUNC_INFO)

#define PHOENIX_THROW_IF(condition, code, message) \
    do { if (condition) PHOENIX_THROW(code, message); } while(0)

#define PHOENIX_ASSERT(condition, message) \
    PHOENIX_THROW_IF(!(condition), PhoenixDRS::Core::ErrorCode::InvalidParameter, message)

#define PHOENIX_REQUIRE_NON_NULL(ptr, name) \
    PHOENIX_THROW_IF(ptr == nullptr, PhoenixDRS::Core::ErrorCode::NullPointer, \
                    QString("Required parameter '%1' is null").arg(name))

// Try-catch helpers
#define PHOENIX_TRY_RESULT(operation) \
    [&]() -> decltype(operation) { \
        try { \
            return operation; \
        } catch (const PhoenixException& e) { \
            return decltype(operation)(e); \
        } catch (const std::exception& e) { \
            return decltype(operation)(PhoenixException::create( \
                PhoenixDRS::Core::ErrorCode::UnknownError, \
                QString::fromStdString(e.what()))); \
        } \
    }()

// Logging integration
#define qCDebugPhoenix(category) qCDebug(category)
#define qCInfoPhoenix(category) qCInfo(category)  
#define qCWarningPhoenix(category) qCWarning(category)
#define qCCriticalPhoenix(category) qCCritical(category)

// String conversion utilities
QString errorCodeToString(ErrorCode code);
QString errorCategoryToString(ErrorCategory category);
QString errorSeverityToString(ErrorSeverity severity);

ErrorCode stringToErrorCode(const QString& str);
ErrorCategory stringToErrorCategory(const QString& str);
ErrorSeverity stringToErrorSeverity(const QString& str);

} // namespace Core
} // namespace PhoenixDRS