#include "include/VideoRebuilder.h"
#include "include/ForensicLogger.h"
#include <QFile>
#include <QDataStream>
#include <QMap>

// --- MP4/MOV Structures ---
struct BoxHeader {
    quint32 size;
    char type[5];

    static BoxHeader read(QDataStream& stream) {
        BoxHeader header;
        stream.readRawData(reinterpret_cast<char*>(&header.size), sizeof(quint32));
        stream.readRawData(header.type, 4);
        header.size = qFromBigEndian(header.size);
        header.type[4] = '\0';
        return header;
    }
};

// --- AVI (RIFF) Structures ---
struct RiffHeader {
    char riff[5]; // "RIFF"
    quint32 size;
    char format[5]; // "AVI "
};

struct ListHeader {
    char list[5]; // "LIST"
    quint32 size;
    char type[5]; // "hdrl", "movi"
};

struct ChunkHeader {
    char id[5];
    quint32 size;
};


VideoRebuilder::VideoRebuilder(QObject *parent)
    : QObject(parent)
    , m_status(TaskStatus::Ready)
    , m_progress(0)
    , m_isCancelled(false)
{
}

void VideoRebuilder::rebuild(const QString& corruptedFilePath, const QString& outputFilePath, VideoFormat format)
{
    if (m_status == TaskStatus::Running) {
        ForensicLogger::instance()->logWarning("Video rebuild is already in progress.");
        return;
    }

    m_isCancelled = false;
    m_progress = 0;
    setStatus(TaskStatus::Running);
    emit progressChanged(0);

    ForensicLogger::instance()->logInfo(QString("Starting video rebuild for: %1").arg(corruptedFilePath));

    try {
        bool success = false;
        switch(format) {
            case VideoFormat::MP4_MOV:
                success = processMp4(corruptedFilePath, outputFilePath);
                break;
            case VideoFormat::AVI:
                success = processAvi(corruptedFilePath, outputFilePath);
                break;
            default:
                ForensicLogger::instance()->logError("Unsupported video format for rebuild.");
                success = false;
                break;
        }

        if (success) {
            ForensicLogger::instance()->logInfo("Video rebuild completed successfully.");
            setStatus(TaskStatus::Completed);
            emit rebuildFinished(true, "Success");
        } else {
            if (m_isCancelled) {
                ForensicLogger::instance()->logWarning("Video rebuild was cancelled.");
                setStatus(TaskStatus::Cancelled);
                emit rebuildFinished(false, "Cancelled by user.");
            } else {
                ForensicLogger::instance()->logError("Video rebuild failed. The file may be unsupported or severely corrupted.");
                setStatus(TaskStatus::Failed);
                emit rebuildFinished(false, "Rebuild failed.");
            }
        }
    } catch (const std::exception& e) {
        ForensicLogger::instance()->logError(QString("A critical error occurred during video rebuild: %1").arg(e.what()));
        setStatus(TaskStatus::Failed);
        emit rebuildFinished(false, e.what());
    }
}

void VideoRebuilder::cancel()
{
    if (m_status == TaskStatus::Running) {
        m_isCancelled.store(true);
    }
}

bool VideoRebuilder::processMp4(const QString& inPath, const QString& outPath)
{
    QFile inFile(inPath);
    if (!inFile.open(QIODevice::ReadOnly)) {
        ForensicLogger::instance()->logError(QString("Cannot open input file: %1").arg(inFile.errorString()));
        return false;
    }

    emit progressChanged(10);

    // Find and store all top-level atoms
    QMap<QString, QPair<qint64, qint64>> atoms; // type -> {offset, size}
    qint64 currentOffset = 0;
    qint64 totalSize = inFile.size();
    
    while(currentOffset < totalSize && !m_isCancelled.load()) {
        inFile.seek(currentOffset);
        QDataStream stream(&inFile);
        BoxHeader header = BoxHeader::read(stream);
        if (header.size == 0 || header.size > totalSize) break;
        
        atoms[header.type] = {currentOffset, header.size};
        currentOffset += header.size;
        
        // Update progress for atom discovery
        int progress = 10 + static_cast<int>((currentOffset * 30) / totalSize);
        emit progressChanged(progress);
    }

    if (m_isCancelled.load()) return false;

    // A playable MP4 needs ftyp, moov, and mdat in that order.
    if (!atoms.contains("ftyp") || !atoms.contains("moov") || !atoms.contains("mdat")) {
        ForensicLogger::instance()->logError("Essential MP4 atoms (ftyp, moov, mdat) not found.");
        return false;
    }

    emit progressChanged(50);

    QFile outFile(outPath);
    if (!outFile.open(QIODevice::WriteOnly)) {
        ForensicLogger::instance()->logError(QString("Cannot open output file: %1").arg(outFile.errorString()));
        return false;
    }

    // Write atoms in the correct order
    QStringList orderedAtoms = {"ftyp", "moov", "mdat"};
    qint64 totalBytesToWrite = 0;
    for(const QString& type : orderedAtoms) {
        if (atoms.contains(type)) {
            totalBytesToWrite += atoms[type].second;
        }
    }
    
    qint64 bytesWritten = 0;
    for(const QString& type : orderedAtoms) {
        if(m_isCancelled.load()) break;
        if (!atoms.contains(type)) continue;
        
        auto atom = atoms[type];
        inFile.seek(atom.first);
        
        // Write in chunks to show progress
        qint64 remaining = atom.second;
        const qint64 chunkSize = 64 * 1024; // 64KB chunks
        
        while (remaining > 0 && !m_isCancelled.load()) {
            qint64 toRead = qMin(chunkSize, remaining);
            QByteArray data = inFile.read(toRead);
            if (data.size() != toRead) break;
            
            qint64 written = outFile.write(data);
            if (written != data.size()) break;
            
            remaining -= written;
            bytesWritten += written;
            
            // Update progress
            int progress = 50 + static_cast<int>((bytesWritten * 45) / totalBytesToWrite);
            emit progressChanged(progress);
        }
    }

    outFile.close();
    inFile.close();
    
    emit progressChanged(100);
    
    return !m_isCancelled.load();
}

bool VideoRebuilder::processAvi(const QString& inPath, const QString& outPath)
{
    QFile inFile(inPath);
    if (!inFile.open(QIODevice::ReadOnly)) {
        ForensicLogger::instance()->logError(QString("Cannot open input AVI file: %1").arg(inFile.errorString()));
        return false;
    }

    emit progressChanged(10);

    QFile outFile(outPath);
    if (!outFile.open(QIODevice::WriteOnly)) {
        ForensicLogger::instance()->logError(QString("Cannot open output AVI file: %1").arg(outFile.errorString()));
        return false;
    }

    qint64 totalSize = inFile.size();
    QDataStream inStream(&inFile);
    inStream.setByteOrder(QDataStream::LittleEndian);

    // Read RIFF header
    RiffHeader riffHeader;
    inStream.readRawData(riffHeader.riff, 4);
    riffHeader.riff[4] = '\0';
    inStream >> riffHeader.size;
    inStream.readRawData(riffHeader.format, 4);
    riffHeader.format[4] = '\0';

    if (QString(riffHeader.riff) != "RIFF" || QString(riffHeader.format) != "AVI ") {
        ForensicLogger::instance()->logError("Invalid AVI file format");
        return false;
    }

    emit progressChanged(20);

    // Write RIFF header to output
    QDataStream outStream(&outFile);
    outStream.setByteOrder(QDataStream::LittleEndian);
    outStream.writeRawData(riffHeader.riff, 4);
    outStream << riffHeader.size;
    outStream.writeRawData(riffHeader.format, 4);

    // Process chunks
    qint64 bytesProcessed = 12; // RIFF header size
    QMap<QString, QByteArray> chunks;
    
    while (bytesProcessed < totalSize && !m_isCancelled.load()) {
        if (inFile.pos() >= totalSize - 8) break;
        
        ChunkHeader chunkHeader;
        inStream.readRawData(chunkHeader.id, 4);
        chunkHeader.id[4] = '\0';
        inStream >> chunkHeader.size;
        
        if (chunkHeader.size == 0 || chunkHeader.size > totalSize) break;
        
        QString chunkId(chunkHeader.id);
        QByteArray chunkData = inFile.read(chunkHeader.size);
        
        if (chunkId == "LIST") {
            // Handle LIST chunks specially
            if (chunkData.size() >= 4) {
                QString listType = QString::fromLatin1(chunkData.left(4));
                if (listType == "hdrl" || listType == "movi") {
                    chunks[chunkId + "_" + listType] = chunkData;
                }
            }
        } else {
            chunks[chunkId] = chunkData;
        }
        
        bytesProcessed += 8 + chunkHeader.size;
        
        // Align to word boundary
        if (chunkHeader.size % 2 != 0) {
            inFile.read(1);
            bytesProcessed++;
        }
        
        // Update progress
        int progress = 20 + static_cast<int>((bytesProcessed * 60) / totalSize);
        emit progressChanged(progress);
    }

    if (m_isCancelled.load()) {
        outFile.close();
        return false;
    }

    emit progressChanged(80);

    // Write essential chunks in correct order
    QStringList essentialChunks = {"LIST_hdrl", "LIST_movi", "idx1"};
    
    for (const QString& chunkKey : essentialChunks) {
        if (chunks.contains(chunkKey) && !m_isCancelled.load()) {
            QByteArray chunkData = chunks[chunkKey];
            
            // Write chunk header
            if (chunkKey.startsWith("LIST_")) {
                outStream.writeRawData("LIST", 4);
                outStream << static_cast<quint32>(chunkData.size());
            } else {
                outStream.writeRawData(chunkKey.toLatin1().constData(), 4);
                outStream << static_cast<quint32>(chunkData.size());
            }
            
            // Write chunk data
            outFile.write(chunkData);
            
            // Align to word boundary
            if (chunkData.size() % 2 != 0) {
                outFile.write("\0", 1);
            }
        }
    }

    outFile.close();
    inFile.close();
    
    emit progressChanged(100);
    
    ForensicLogger::instance()->logInfo("AVI rebuild completed");
    return !m_isCancelled.load();
}


void VideoRebuilder::setStatus(TaskStatus status)
{
    if (m_status != status) {
        m_status = status;
        emit statusChanged(status);
    }
}
