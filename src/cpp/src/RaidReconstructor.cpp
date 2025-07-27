#include "include/RaidReconstructor.h"
#include "include/ForensicLogger.h"
#include <QFile>
#include <QVector>
#include <algorithm> // For std::min

RaidReconstructor::RaidReconstructor(QObject *parent)
    : QObject(parent)
    , m_status(TaskStatus::Idle)
    , m_progress(0)
    , m_isCancelled(false)
{
}

void RaidReconstructor::setConfiguration(const RaidConfiguration& config)
{
    m_config = config;
}

void RaidReconstructor::cancel()
{
    if (m_status == TaskStatus::InProgress) {
        m_isCancelled.store(true);
    }
}

void RaidReconstructor::reconstruct(const QString& outputImagePath)
{
    if (m_status == TaskStatus::InProgress) {
        logWarn("RAID reconstruction is already in progress.");
        return;
    }

    if (m_config.diskPaths.isEmpty() || m_config.type == RaidType::Unknown || m_config.stripeSize == 0) {
        logError("RAID configuration is invalid.");
        emit reconstructionFinished(false, "Invalid RAID configuration.");
        return;
    }

    m_outputImagePath = outputImagePath;
    m_isCancelled = false;
    m_progress = 0;
    setStatus(TaskStatus::InProgress);
    emit progressChanged(0);

    logInfo(QString("Starting RAID %1 reconstruction.").arg(raidTypeToString(m_config.type)));
    logInfo(QString("Stripe size: %1 KB").arg(m_config.stripeSize / 1024));
    logInfo(QString("Output image: %1").arg(outputImagePath));

    // This should be run in a worker thread in a real application
    try {
        bool success = performReconstruction();
        if (success) {
            logInfo("RAID reconstruction completed successfully.");
            setStatus(TaskStatus::Completed);
            emit reconstructionFinished(true, "Success");
        } else {
            if (m_isCancelled) {
                logWarn("RAID reconstruction was cancelled.");
                setStatus(TaskStatus::Cancelled);
                emit reconstructionFinished(false, "Cancelled by user.");
            } else {
                logError("RAID reconstruction failed.");
                setStatus(TaskStatus::Error);
                emit reconstructionFinished(false, "Reconstruction failed.");
            }
        }
    } catch (const std::exception& e) {
        logCritical(QString("A critical error occurred during RAID reconstruction: %1").arg(e.what()));
        setStatus(TaskStatus::Error);
        emit reconstructionFinished(false, e.what());
    }
}

bool RaidReconstructor::performReconstruction()
{
    QFile outputFile(m_outputImagePath);
    if (!outputFile.open(QIODevice::WriteOnly)) {
        logError(QString("Cannot open output file: %1").arg(outputFile.errorString()));
        return false;
    }

    QVector<QFile*> sourceFiles;
    for (const QString& diskPath : m_config.diskPaths) {
        QFile* file = new QFile(diskPath);
        if (!file->open(QIODevice::ReadOnly)) {
            logError(QString("Failed to open disk image: %1").arg(diskPath));
            qDeleteAll(sourceFiles);
            return false;
        }
        sourceFiles.append(file);
    }

    bool result = false;
    switch(m_config.type) {
        case RaidType::RAID0:
            result = reconstructRaid0(outputFile, sourceFiles);
            break;
        case RaidType::RAID5:
            result = reconstructRaid5(outputFile, sourceFiles);
            break;
        default:
            logError(QString("RAID type '%1' is not yet supported.").arg(raidTypeToString(m_config.type)));
            result = false;
            break;
    }

    qDeleteAll(sourceFiles);
    outputFile.close();
    return result;
}

bool RaidReconstructor::reconstructRaid0(QFile& outputFile, QVector<QFile*>& sourceFiles)
{
    if (sourceFiles.size() < 2) {
        logError("RAID 0 requires at least 2 disks.");
        return false;
    }

    qint64 totalSize = 0;
    for(auto file : sourceFiles) totalSize += file->size();
    if (totalSize == 0) return true;

    qint64 bytesWritten = 0;
    int currentDisk = 0;
    QByteArray buffer(m_config.stripeSize, 0);

    while (bytesWritten < totalSize && !m_isCancelled) {
        qint64 bytesToRead = std::min((qint64)buffer.size(), sourceFiles[currentDisk]->size() - sourceFiles[currentDisk]->pos());
        if(bytesToRead <= 0) break;

        qint64 bytesRead = sourceFiles[currentDisk]->read(buffer.data(), bytesToRead);
        if (bytesRead <= 0) break;

        outputFile.write(buffer.data(), bytesRead);
        bytesWritten += bytesRead;

        currentDisk = (currentDisk + 1) % sourceFiles.size();

        int newProgress = static_cast<int>((bytesWritten * 100) / totalSize);
        if (newProgress != m_progress) {
            m_progress = newProgress;
            emit progressChanged(m_progress);
        }
    }
    return !m_isCancelled;
}

bool RaidReconstructor::reconstructRaid5(QFile& outputFile, QVector<QFile*>& sourceFiles)
{
    int numDisks = sourceFiles.size();
    if (numDisks < 3) {
        logError("RAID 5 requires at least 3 disks.");
        return false;
    }

    qint64 singleDiskSize = sourceFiles[0]->size();
    qint64 totalDataSize = singleDiskSize * (numDisks - 1);
    qint64 bytesWritten = 0;
    qint64 stripeNum = 0;

    QVector<QByteArray> stripeBuffers(numDisks);
    for(int i=0; i<numDisks; ++i) stripeBuffers[i].resize(m_config.stripeSize);
    QByteArray parityBuffer(m_config.stripeSize, 0);

    while(bytesWritten < totalDataSize && !m_isCancelled)
    {
        int parityDisk = (numDisks - 1) - (stripeNum % numDisks); // Simple Left-Symmetric layout

        // Read a full stripe
        for(int i=0; i<numDisks; ++i) {
            if(sourceFiles[i]->read(stripeBuffers[i].data(), m_config.stripeSize) != m_config.stripeSize) {
                // Reached end of a file prematurely
                goto end_loop;
            }
        }

        // Write data blocks, reconstructing if necessary
        for(int i=0; i<numDisks; ++i) {
            if(i == parityDisk) continue; // Skip parity block

            if(i == m_config.missingDiskIndex) {
                // Reconstruct missing data block
                QVector<QByteArray> availableBlocks;
                for(int j=0; j<numDisks; ++j) {
                    if(j != i) { // Exclude the block we are reconstructing
                        availableBlocks.append(stripeBuffers[j]);
                    }
                }
                xorBlocks(availableBlocks, parityBuffer);
                outputFile.write(parityBuffer);
            } else {
                outputFile.write(stripeBuffers[i]);
            }
            bytesWritten += m_config.stripeSize;
        }

        stripeNum++;
        int newProgress = static_cast<int>((bytesWritten * 100) / totalDataSize);
        if (newProgress != m_progress) {
            m_progress = newProgress;
            emit progressChanged(m_progress);
        }
    }

end_loop:
    return !m_isCancelled;
}

void RaidReconstructor::xorBlocks(const QVector<QByteArray>& sources, QByteArray& dest)
{
    if (sources.isEmpty()) return;
    
    dest = sources[0];
    for (int i = 1; i < sources.size(); ++i) {
        for (int j = 0; j < dest.size(); ++j) {
            dest[j] = dest[j] ^ sources[i][j];
        }
    }
}


void RaidReconstructor::setStatus(TaskStatus status)
{
    if (m_status != status) {
        m_status = status;
        emit statusChanged(status);
    }
}

QString RaidReconstructor::raidTypeToString(RaidType type)
{
    switch(type) {
        case RaidType::RAID0: return "RAID 0";
        case RaidType::RAID1: return "RAID 1";
        case RaidType::RAID5: return "RAID 5";
        case RaidType::RAID6: return "RAID 6";
        default: return "Unknown";
    }
}
