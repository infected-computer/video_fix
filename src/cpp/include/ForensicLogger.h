#ifndef FORENSICLOGGER_H
#define FORENSICLOGGER_H

#include <QObject>
#include <QString>
#include <QFile>
#include <QTextStream>
#include <QMutex>
#include <QDateTime>
#include <QJsonObject>
#include <QJsonArray>
#include <QTimer>
#include <QUuid>
#include <QCryptographicHash>
#include "Common.h"

#include <memory>
#include <atomic>
#include <queue>

// Enhanced forensic logger with chain of custody and integrity verification
class ForensicLogger : public QObject
{
    Q_OBJECT
    Q_DISABLE_COPY(ForensicLogger)

public:
    // Enhanced log levels for forensic operations
    enum LogLevel {
        LevelTrace = 0,    // Detailed trace information
        LevelDebug = 1,    // Debug information
        LevelInfo = 2,     // General information  
        LevelWarn = 3,     // Warnings
        LevelError = 4,    // Errors
        LevelCritical = 5, // Critical errors
        LevelEvidence = 6, // Forensic evidence trail (highest priority)
        LevelAudit = 7     // Audit trail entries
    };
    Q_ENUM(LogLevel)

    // Log entry structure for enhanced logging
    struct LogEntry {
        QUuid entryId;
        QDateTime timestamp;
        LogLevel level;
        QString category;
        QString operation;
        QString message;
        QString examiner;
        QString caseContext;
        QJsonObject metadata;
        QString checksum;
        
        LogEntry() : level(LevelInfo) {
            entryId = QUuid::createUuid();
            timestamp = QDateTime::currentDateTimeUtc();
        }
    };

    // Chain of custody entry
    struct CustodyEntry {
        QUuid entryId;
        QDateTime timestamp;
        QString action;
        QString examiner;
        QString evidence;
        QString description;
        QJsonObject parameters;
        QString hash;
        QString signature;
        
        CustodyEntry() {
            entryId = QUuid::createUuid();
            timestamp = QDateTime::currentDateTimeUtc();
        }
    };

    // Get the singleton instance
    static ForensicLogger* instance();

    // Initialization and session management
    bool initialize(const QString& logFilePath, LogLevel level = LevelInfo);
    void startSession(const QString& examiner, const QString& caseName = QString());
    void endSession();
    void shutdown();

    // Enhanced logging methods
    void log(const QString& message, LogLevel level);
    void logWithMetadata(const QString& category, const QString& operation, 
                        const QString& message, LogLevel level, 
                        const QJsonObject& metadata = QJsonObject());

    // Chain of custody methods
    void recordCustody(const QString& action, const QString& evidence, 
                      const QString& description, const QJsonObject& parameters = QJsonObject());
    void recordEvidenceAcquisition(const QString& evidence, const QString& source, 
                                  const QString& method, const QJsonObject& details);
    void recordEvidenceTransfer(const QString& evidence, const QString& from, 
                               const QString& to, const QString& reason);

    // Configuration methods
    void setLogLevel(LogLevel level) { m_logLevel = level; }
    LogLevel getLogLevel() const { return m_logLevel; }
    void setExaminer(const QString& examiner) { m_currentExaminer = examiner; }
    QString getExaminer() const { return m_currentExaminer; }
    void setCaseContext(const QString& caseName) { m_currentCase = caseName; }
    QString getCaseContext() const { return m_currentCase; }

    // Output configuration
    void enableConsoleOutput(bool enable) { m_consoleOutput = enable; }
    void enableJsonOutput(bool enable) { m_jsonOutput = enable; }
    void setAutoFlush(bool enable);

    // Integrity and security
    QString calculateSessionHash() const;
    bool verifyLogIntegrity() const;
    bool exportAuditTrail(const QString& filePath);

    // Query methods
    std::vector<LogEntry> getLogEntries(LogLevel minLevel = LevelInfo) const;
    std::vector<CustodyEntry> getCustodyChain() const;

    // Statistics
    struct LogStatistics {
        int totalEntries = 0;
        int entriesByLevel[8] = {0}; // One for each LogLevel
        QDateTime sessionStart;
        QDateTime lastEntry;
        int custodyEntries = 0;
        qint64 logFileSize = 0;
    };
    
    LogStatistics getStatistics() const;

    // Get the current log file path
    QString getLogFilePath() const { return m_logFile.fileName(); }

    // Convenience methods for backward compatibility
    void logInfo(const QString& message) { log(message, LevelInfo); }
    void logWarning(const QString& message) { log(message, LevelWarn); }
    void logError(const QString& message) { log(message, LevelError); }
    void logDebug(const QString& message) { log(message, LevelDebug); }
    void logCritical(const QString& message) { log(message, LevelCritical); }

signals:
    // Enhanced signals
    void newMessage(const QString& formattedMessage, LogLevel level);
    void logEntryAdded(const LogEntry& entry);
    void custodyEntryAdded(const CustodyEntry& entry);
    void sessionStarted(const QString& sessionId);
    void sessionEnded(const QString& sessionId);
    void integrityViolation(const QString& details);

private slots:
    void flushLogs();

private:
    // Private constructor for singleton pattern
    ForensicLogger(QObject *parent = nullptr);
    ~ForensicLogger();

    // Core logging implementation
    void writeLogEntry(const LogEntry& entry);
    void writeCustodyEntry(const CustodyEntry& entry);
    void writeToFile(const QString& text);
    void writeToConsole(const QString& text);
    void writeToJson(const QJsonObject& jsonEntry);

    // Utility methods
    QString formatLogEntry(const LogEntry& entry) const;
    QString formatCustodyEntry(const CustodyEntry& entry) const;
    QString logLevelToString(LogLevel level) const;
    QString calculateEntryChecksum(const LogEntry& entry) const;
    QString calculateCustodyChecksum(const CustodyEntry& entry) const;
    QJsonObject getSystemEnvironment() const;

    // Static instance
    static ForensicLogger* m_instance;
    static QMutex m_mutex;

    // File handles
    QFile m_logFile;
    QFile m_jsonFile;
    QTextStream m_logStream;
    QTextStream m_jsonStream;

    // Configuration
    LogLevel m_logLevel;
    bool m_consoleOutput;
    bool m_jsonOutput;
    bool m_autoFlush;

    // Session context
    QString m_sessionId;
    QString m_currentExaminer;
    QString m_currentCase;
    QDateTime m_sessionStart;

    // Buffering and threading
    std::queue<LogEntry> m_logBuffer;
    std::queue<CustodyEntry> m_custodyBuffer;
    QTimer* m_flushTimer;
    std::atomic<bool> m_shouldFlush{false};

    // Integrity monitoring
    QCryptographicHash m_sessionHasher{QCryptographicHash::Sha256};
    std::atomic<int> m_logSequence{0};

    // Statistics
    mutable LogStatistics m_statistics;
    std::vector<LogEntry> m_logEntries;
    std::vector<CustodyEntry> m_custodyEntries;
};

// Global helper functions for easy logging (enhanced)
void logDebug(const QString& message);
void logInfo(const QString& message);
void logWarn(const QString& message);
void logError(const QString& message);
void logCritical(const QString& message);
void logEvidence(const QString& caseNumber, const QString& evidenceId, const QString& description);
void logAudit(const QString& category, const QString& operation, const QString& message);

#endif // FORENSICLOGGER_H
