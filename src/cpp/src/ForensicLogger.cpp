#include "include/ForensicLogger.h"
#include <QDir>
#include <QCoreApplication>

// Initialize static members
ForensicLogger* ForensicLogger::m_instance = nullptr;
QMutex ForensicLogger::m_mutex;

ForensicLogger* ForensicLogger::instance()
{
    QMutexLocker locker(&m_mutex);
    if (!m_instance) {
        m_instance = new ForensicLogger(QCoreApplication::instance());
    }
    return m_instance;
}

ForensicLogger::ForensicLogger(QObject *parent)
    : QObject(parent), m_logLevel(LevelInfo)
{
}

ForensicLogger::~ForensicLogger()
{
    if (m_logFile.isOpen()) {
        m_logStream.flush();
        m_logFile.close();
    }
}

bool ForensicLogger::initialize(const QString& logFilePath, LogLevel level)
{
    QMutexLocker locker(&m_mutex);
    if (m_logFile.isOpen()) {
        return true; // Already initialized
    }

    m_logLevel = level;
    m_logFile.setFileName(logFilePath);

    QDir logDir = QFileInfo(logFilePath).absoluteDir();
    if (!logDir.exists()) {
        if (!logDir.mkpath(".")) {
            // Cannot use log() here as it's not ready
            qCritical("Failed to create log directory: %s", qPrintable(logDir.path()));
            return false;
        }
    }

    if (!m_logFile.open(QIODevice::WriteOnly | QIODevice::Append | QIODevice::Text)) {
        qCritical("Failed to open log file: %s", qPrintable(m_logFile.errorString()));
        return false;
    }

    m_logStream.setDevice(&m_logFile);
    m_logStream.setEncoding(QStringConverter::Utf8);
    return true;
}

void ForensicLogger::log(const QString& message, LogLevel level)
{
    if (level < m_logLevel) {
        return;
    }

    QMutexLocker locker(&m_mutex);
    if (!m_logFile.isOpen()) {
        // Log to console if file is not available
        qWarning("Log file not initialized. Message: %s", qPrintable(message));
        return;
    }

    QString levelStr;
    switch (level) {
        case LevelDebug:    levelStr = "DEBUG"; break;
        case LevelInfo:     levelStr = "INFO"; break;
        case LevelWarn:     levelStr = "WARN"; break;
        case LevelError:    levelStr = "ERROR"; break;
        case LevelCritical: levelStr = "CRITICAL"; break;
        case LevelEvidence: levelStr = "EVIDENCE"; break;
    }

    QString formattedMessage = QString("%1 | %2 | %3")
        .arg(QDateTime::currentDateTime().toString(Qt::ISODate))
        .arg(levelStr, -8) // Pad to 8 chars
        .arg(message);

    m_logStream << formattedMessage << Qt::endl;
    m_logStream.flush(); // Ensure logs are written immediately

    emit newMessage(formattedMessage, level);
}

void ForensicLogger::setLogLevel(LogLevel level)
{
    QMutexLocker locker(&m_mutex);
    m_logLevel = level;
}

ForensicLogger::LogLevel ForensicLogger::getLogLevel() const
{
    return m_logLevel;
}

QString ForensicLogger::getLogFilePath() const
{
    return m_logFile.fileName();
}

// Global helper function implementations
void logDebug(const QString& message)
{
    ForensicLogger::instance()->log(message, ForensicLogger::LevelDebug);
}

void logInfo(const QString& message)
{
    ForensicLogger::instance()->log(message, ForensicLogger::LevelInfo);
}

void logWarn(const QString& message)
{
    ForensicLogger::instance()->log(message, ForensicLogger::LevelWarn);
}

void logError(const QString& message)
{
    ForensicLogger::instance()->log(message, ForensicLogger::LevelError);
}

void logCritical(const QString& message)
{
    ForensicLogger::instance()->log(message, ForensicLogger::LevelCritical);
}

void logEvidence(const QString& caseNumber, const QString& evidenceId, const QString& description)
{
    QString message = QString("Case: %1 | Evidence: %2 | Details: %3")
        .arg(caseNumber, evidenceId, description);
    ForensicLogger::instance()->log(message, ForensicLogger::LevelEvidence);
}
