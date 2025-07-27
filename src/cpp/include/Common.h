#ifndef COMMON_H
#define COMMON_H

#include <QString>
#include <QVector>

// Enum for the status of a long-running task
enum class TaskStatus {
    Idle,
    InProgress,
    Completed,
    Cancelled,
    Error
};

// Enum for RAID types
enum class RaidType {
    Unknown,
    RAID0,
    RAID1,
    RAID5,
    RAID6,
    JBOD
};

// Configuration for a RAID array
struct RaidConfiguration {
    RaidType type;
    qint64 stripeSize; // in bytes
    QVector<QString> disks; // paths to disk images
    // For RAID5/6, parity position and rotation might be needed
};

// Struct for a detected file signature
struct FileSignature {
    QString extension;
    QString description;
    QByteArray hexSignature;
    qint64 offset = 0;
};


#endif // COMMON_H
