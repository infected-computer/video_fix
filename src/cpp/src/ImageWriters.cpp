/*
 * PhoenixDRS Professional - Image Format Writers Implementation
 * מימוש כותבי פורמטי תמונה מקצועיים
 */

#include "DiskImager.h"
#include "ForensicLogger.h"
#include <QDataStream>
#include <QCryptographicHash>
#include <QUuid>
#include <QFileInfo>
#include <QtEndian>
#include <QDateTime>
#include <QTextStream>

#include <zlib.h>
#include <cstring>
#include <algorithm>

namespace PhoenixDRS {

/*
 * ImageWriter Factory Method
 */
std::unique_ptr<ImageWriter> ImageWriter::createWriter(ImageFormat format)
{
    switch (format) {
        case ImageFormat::RAW_DD:
            return std::make_unique<RawImageWriter>();
        case ImageFormat::ENCASE_E01:
            return std::make_unique<E01ImageWriter>();
        default:
            return nullptr;
    }
}

/*
 * Raw DD Image Writer Implementation
 */
bool RawImageWriter::initialize(const QString& path, const ImagingParameters& params)
{
    m_file = std::make_unique<QFile>(path);
    m_bytesWritten = 0;
    m_lastError.clear();
    
    if (!m_file->open(QIODevice::WriteOnly)) {
        m_lastError = m_file->errorString();
        return false;
    }
    
    // Pre-allocate file space if compression is not used
    if (params.compression == CompressionLevel::None) {
        DeviceInfo sourceInfo = DiskImager::getDeviceInfo(params.sourceDevice);
        if (sourceInfo.totalSize > 0) {
            if (!m_file->resize(sourceInfo.totalSize)) {
                m_lastError = QObject::tr("Failed to pre-allocate file space");
                return false;
            }
        }
    }
    
    ForensicLogger::instance().info("raw_writer_init", "image_writer", 
                                   QStringLiteral("Raw DD writer initialized for: %1").arg(path));
    
    return true;
}

bool RawImageWriter::writeBlock(const QByteArray& data, qint64 offset)
{
    if (!m_file || !m_file->isOpen()) {
        m_lastError = QObject::tr("File not open");
        return false;
    }
    
    // Seek to offset
    if (!m_file->seek(offset)) {
        m_lastError = QObject::tr("Failed to seek to offset %1").arg(offset);
        return false;
    }
    
    // Write data
    qint64 bytesWritten = m_file->write(data);
    if (bytesWritten != data.size()) {
        m_lastError = QObject::tr("Write failed: %1").arg(m_file->errorString());
        return false;
    }
    
    m_bytesWritten += bytesWritten;
    
    // Flush every 64MB for better performance monitoring
    if (m_bytesWritten % (64 * 1024 * 1024) == 0) {
        m_file->flush();
    }
    
    return true;
}

bool RawImageWriter::finalize()
{
    if (!m_file) {
        return true;
    }
    
    bool success = m_file->flush();
    m_file->close();
    
    if (success) {
        ForensicLogger::instance().info("raw_writer_finalized", "image_writer",
                                       QStringLiteral("Raw DD image finalized. Bytes written: %1")
                                       .arg(m_bytesWritten));
    }
    
    return success;
}

/*
 * EnCase E01 Image Writer Implementation
 */

// E01 format structures
struct E01ImageWriter::E01Header {
    char signature[8];      // "EVF\x09\x0D\x0A\xFF\x00"
    uint8_t fields_start;   // 0x01
    uint16_t fields_segment; // Little-endian
    uint16_t fields_end;    // 0x00
    
    E01Header() {
        memcpy(signature, "EVF\x09\x0D\x0A\xFF\x00", 8);
        fields_start = 0x01;
        fields_segment = qToLittleEndian<uint16_t>(1);
        fields_end = qToLittleEndian<uint16_t>(0);
    }
};

struct E01ImageWriter::E01Section {
    char type[16];          // Section type (null-padded)
    uint64_t next_offset;   // Offset to next section
    uint64_t size;          // Size of this section
    uint8_t padding[40];    // Padding
    uint32_t checksum;      // Adler-32 checksum
    
    E01Section() {
        memset(this, 0, sizeof(E01Section));
    }
};

bool E01ImageWriter::initialize(const QString& path, const ImagingParameters& params)
{
    m_file = std::make_unique<QFile>(path);
    m_bytesWritten = 0;
    m_lastError.clear();
    m_params = params;
    m_chunkOffsets.clear();
    
    if (!m_file->open(QIODevice::WriteOnly)) {
        m_lastError = m_file->errorString();
        return false;
    }
    
    // Write E01 header
    if (!writeHeader()) {
        return false;
    }
    
    ForensicLogger::instance().info("e01_writer_init", "image_writer",
                                   QStringLiteral("E01 writer initialized for: %1").arg(path));
    
    return true;
}

bool E01ImageWriter::writeHeader()
{
    // Write file signature
    E01Header header;
    if (m_file->write(reinterpret_cast<const char*>(&header), sizeof(header)) != sizeof(header)) {
        m_lastError = QObject::tr("Failed to write E01 header");
        return false;
    }
    
    // Write header section
    E01Section headerSection;
    memcpy(headerSection.type, "header", 6);
    headerSection.size = qToLittleEndian<uint64_t>(sizeof(E01Section));
    headerSection.next_offset = qToLittleEndian<uint64_t>(sizeof(E01Header) + sizeof(E01Section));
    
    // Calculate checksum (simplified)
    headerSection.checksum = qToLittleEndian<uint32_t>(
        adler32(0, reinterpret_cast<const Bytef*>(&headerSection), sizeof(E01Section) - 4));
    
    if (m_file->write(reinterpret_cast<const char*>(&headerSection), sizeof(headerSection)) != sizeof(headerSection)) {
        m_lastError = QObject::tr("Failed to write E01 header section");
        return false;
    }
    
    // Write case information section
    QByteArray caseInfo;
    QDataStream stream(&caseInfo, QIODevice::WriteOnly);
    stream.setByteOrder(QDataStream::LittleEndian);
    
    // Case information in E01 format
    QString caseData = QString("c\tn\t%1\n").arg(m_params.caseName.isEmpty() ? "Unknown Case" : m_params.caseName);
    caseData += QString("a\tn\t%1\n").arg(m_params.examiner.isEmpty() ? "PhoenixDRS" : m_params.examiner);
    caseData += QString("e\tn\t%1\n").arg(m_params.evidence.isEmpty() ? "Evidence" : m_params.evidence);
    caseData += QString("n\tn\t%1\n").arg(m_params.notes.isEmpty() ? "Created by PhoenixDRS Professional" : m_params.notes);
    caseData += QString("m\tn\t%1\n").arg(QDateTime::currentDateTime().toString(Qt::ISODate));
    caseData += QString("u\tn\tPhoenixDRS Professional\n");
    caseData += QString("p\tn\t%1\n").arg(qHash(caseData)); // Simple hash for now
    
    QByteArray utf8Data = caseData.toUtf8();
    stream.writeRawData(utf8Data.constData(), utf8Data.size());
    
    // Pad to 16-byte boundary
    int padding = (16 - (caseInfo.size() % 16)) % 16;
    for (int i = 0; i < padding; ++i) {
        stream << static_cast<uint8_t>(0);
    }
    
    // Write case section header
    E01Section caseSection;
    memcpy(caseSection.type, "header2", 7);
    caseSection.size = qToLittleEndian<uint64_t>(sizeof(E01Section) + caseInfo.size());
    caseSection.next_offset = qToLittleEndian<uint64_t>(m_file->pos() + sizeof(E01Section) + caseInfo.size());
    caseSection.checksum = qToLittleEndian<uint32_t>(
        adler32(0, reinterpret_cast<const Bytef*>(caseInfo.constData()), caseInfo.size()));
    
    if (m_file->write(reinterpret_cast<const char*>(&caseSection), sizeof(caseSection)) != sizeof(caseSection)) {
        m_lastError = QObject::tr("Failed to write case section header");
        return false;
    }
    
    if (m_file->write(caseInfo) != caseInfo.size()) {
        m_lastError = QObject::tr("Failed to write case information");
        return false;
    }
    
    m_bytesWritten = m_file->pos();
    return true;
}

bool E01ImageWriter::writeBlock(const QByteArray& data, qint64 offset)
{
    if (!m_file || !m_file->isOpen()) {
        m_lastError = QObject::tr("File not open");
        return false;
    }
    
    return writeDataSection(data);
}

bool E01ImageWriter::writeDataSection(const QByteArray& data)
{
    // Record chunk offset
    m_chunkOffsets.push_back(m_file->pos());
    
    // Compress data if needed
    QByteArray compressedData;
    bool isCompressed = (m_params.compression != CompressionLevel::None);
    
    if (isCompressed) {
        uLongf compressedSize = compressBound(data.size());
        compressedData.resize(compressedSize);
        
        int result = compress2(
            reinterpret_cast<Bytef*>(compressedData.data()),
            &compressedSize,
            reinterpret_cast<const Bytef*>(data.constData()),
            data.size(),
            static_cast<int>(m_params.compression)
        );
        
        if (result != Z_OK) {
            m_lastError = QObject::tr("Compression failed");
            return false;
        }
        
        compressedData.resize(compressedSize);
    }
    
    const QByteArray& dataToWrite = isCompressed ? compressedData : data;
    
    // Create data section
    E01Section dataSection;
    memcpy(dataSection.type, "data", 4);
    dataSection.size = qToLittleEndian<uint64_t>(sizeof(E01Section) + dataToWrite.size() + 4); // +4 for chunk size
    dataSection.next_offset = qToLittleEndian<uint64_t>(m_file->pos() + sizeof(E01Section) + dataToWrite.size() + 4);
    dataSection.checksum = qToLittleEndian<uint32_t>(
        adler32(0, reinterpret_cast<const Bytef*>(dataToWrite.constData()), dataToWrite.size()));
    
    // Write section header
    if (m_file->write(reinterpret_cast<const char*>(&dataSection), sizeof(dataSection)) != sizeof(dataSection)) {
        m_lastError = QObject::tr("Failed to write data section header");
        return false;
    }
    
    // Write original chunk size
    uint32_t originalSize = qToLittleEndian<uint32_t>(static_cast<uint32_t>(data.size()));
    if (m_file->write(reinterpret_cast<const char*>(&originalSize), sizeof(originalSize)) != sizeof(originalSize)) {
        m_lastError = QObject::tr("Failed to write chunk size");
        return false;
    }
    
    // Write compressed/uncompressed data
    if (m_file->write(dataToWrite) != dataToWrite.size()) {
        m_lastError = QObject::tr("Failed to write chunk data");
        return false;
    }
    
    // Update hash calculations
    m_md5Hash.addData(data);
    m_sha1Hash.addData(data);
    
    m_bytesWritten += sizeof(dataSection) + sizeof(originalSize) + dataToWrite.size();
    
    return true;
}

bool E01ImageWriter::finalize()
{
    if (!m_file) {
        return true;
    }
    
    // Write hash section
    if (!writeHashSection()) {
        return false;
    }
    
    // Write done section
    if (!writeDoneSection()) {
        return false;
    }
    
    bool success = m_file->flush();
    m_file->close();
    
    if (success) {
        ForensicLogger::instance().info("e01_writer_finalized", "image_writer",
                                       QStringLiteral("E01 image finalized. Bytes written: %1, Chunks: %2")
                                       .arg(m_bytesWritten)
                                       .arg(m_chunkOffsets.size()));
    }
    
    return success;
}

bool E01ImageWriter::writeHashSection()
{
    // Calculate final hashes
    QByteArray md5Result = m_md5Hash.result();
    QByteArray sha1Result = m_sha1Hash.result();
    
    // Create hash data
    QByteArray hashData;
    QDataStream stream(&hashData, QIODevice::WriteOnly);
    stream.setByteOrder(QDataStream::LittleEndian);
    
    // Write MD5 hash
    hashData.append("MD5:");
    hashData.append(md5Result.toHex());
    hashData.append('\n');
    
    // Write SHA1 hash
    hashData.append("SHA1:");
    hashData.append(sha1Result.toHex());
    hashData.append('\n');
    
    // Pad to 16-byte boundary
    int padding = (16 - (hashData.size() % 16)) % 16;
    for (int i = 0; i < padding; ++i) {
        hashData.append('\0');
    }
    
    // Write hash section
    E01Section hashSection;
    memcpy(hashSection.type, "hash", 4);
    hashSection.size = qToLittleEndian<uint64_t>(sizeof(E01Section) + hashData.size());
    hashSection.next_offset = qToLittleEndian<uint64_t>(m_file->pos() + sizeof(E01Section) + hashData.size());
    hashSection.checksum = qToLittleEndian<uint32_t>(
        adler32(0, reinterpret_cast<const Bytef*>(hashData.constData()), hashData.size()));
    
    if (m_file->write(reinterpret_cast<const char*>(&hashSection), sizeof(hashSection)) != sizeof(hashSection)) {
        m_lastError = QObject::tr("Failed to write hash section header");
        return false;
    }
    
    if (m_file->write(hashData) != hashData.size()) {
        m_lastError = QObject::tr("Failed to write hash data");
        return false;
    }
    
    return true;
}

bool E01ImageWriter::writeDoneSection()
{
    E01Section doneSection;
    memcpy(doneSection.type, "done", 4);
    doneSection.size = qToLittleEndian<uint64_t>(sizeof(E01Section));
    doneSection.next_offset = qToLittleEndian<uint64_t>(0); // No next section
    doneSection.checksum = qToLittleEndian<uint32_t>(
        adler32(0, reinterpret_cast<const Bytef*>(&doneSection), sizeof(E01Section) - 4));
    
    if (m_file->write(reinterpret_cast<const char*>(&doneSection), sizeof(doneSection)) != sizeof(doneSection)) {
        m_lastError = QObject::tr("Failed to write done section");
        return false;
    }
    
    return true;
}

/*
 * Hash Calculator Implementation
 */
HashCalculator::HashCalculator(HashAlgorithm algorithm, QObject* parent)
    : QObject(parent), m_algorithm(algorithm)
{
    QCryptographicHash::Algorithm qtAlgorithm;
    
    switch (algorithm) {
        case HashAlgorithm::MD5:
            qtAlgorithm = QCryptographicHash::Md5;
            break;
        case HashAlgorithm::SHA1:
            qtAlgorithm = QCryptographicHash::Sha1;
            break;
        case HashAlgorithm::SHA256:
            qtAlgorithm = QCryptographicHash::Sha256;
            break;
        case HashAlgorithm::SHA512:
            qtAlgorithm = QCryptographicHash::Sha512;
            break;
        default:
            qtAlgorithm = QCryptographicHash::Sha256;
            break;
    }
    
    m_hash = std::make_unique<QCryptographicHash>(qtAlgorithm);
}

HashCalculator::~HashCalculator() = default;

void HashCalculator::addData(const QByteArray& data)
{
    QMutexLocker locker(&m_mutex);
    m_hash->addData(data);
}

HashResult HashCalculator::getResult()
{
    QMutexLocker locker(&m_mutex);
    
    HashResult result(m_algorithm);
    result.hash = m_hash->result();
    result.hexString = result.hash.toHex().toUpper();
    
    return result;
}

void HashCalculator::reset()
{
    QMutexLocker locker(&m_mutex);
    m_hash->reset();
}

} // namespace PhoenixDRS