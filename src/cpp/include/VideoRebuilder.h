#ifndef VIDEOREBUILDER_H
#define VIDEOREBUILDER_H

#include <QObject>
#include <QString>
#include <atomic>
#include "Common.h"

// Enum for supported video formats
enum class VideoFormat {
    MP4_MOV,
    AVI,
    Unknown
};

// High-level video file rebuilder
class VideoRebuilder : public QObject
{
    Q_OBJECT

public:
    explicit VideoRebuilder(QObject *parent = nullptr);

    void cancel();

public slots:
    void rebuild(const QString& corruptedFilePath, const QString& outputFilePath, VideoFormat format);

signals:
    void progressChanged(int percentage);
    void statusChanged(TaskStatus status);
    void rebuildFinished(bool success, const QString& message);

private:
    bool processMp4(const QString& inPath, const QString& outPath);
    bool processAvi(const QString& inPath, const QString& outPath);

    void setStatus(TaskStatus status);

    TaskStatus m_status;
    int m_progress;
    std::atomic<bool> m_isCancelled;
};

#endif // VIDEOREBUILDER_H
