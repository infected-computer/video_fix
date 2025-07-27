/*
 * PhoenixDRS Professional - High-Performance Disk Imaging Engine Implementation
 * מימוש מנוע הדמיית דיסק בביצועים גבוהים
 */

#include "DiskImager.h"
#include "ForensicLogger.h"
#include <QApplication>
#include <QMessageBox>
#include <QDir>
#include <QStandardPaths>
#include <QTextStream>
#include <QJsonDocument>
#include <QJsonObject>
#include <QJsonArray>
#include <QtConcurrent>
#include <QMutexLocker>

#include <chrono>
#include <thread>
#include <algorithm>
#include <numeric>
#include <iomanip>
#include <sstream>

namespace PhoenixDRS {

/*
 * Internal worker class for imaging operations
 */
class DiskImager::ImagingWorker : public QObject
{
    Q_OBJECT

public:
    explicit ImagingWorker(DiskImager* parent, const ImagingParameters& params)
        : QObject(nullptr), m_imager(parent), m_params(params) {}

public slots:
    void performImaging();

signals:
    void finished();
    void error(const QString& message);
    void progressUpdate();

private:
    DiskImager* m_imager;
    ImagingParameters m_params;
};

/*
 * DiskImager Constructor
 */
DiskImager::DiskImager(QObject* parent)
    : QObject(parent)
    , m_workerThread(nullptr)
    , m_progressTimer(new QTimer(this))
#ifdef Q_OS_WIN
    , m_sourceHandle(INVALID_HANDLE_VALUE)
#else
    , m_sourceHandle(-1)
#endif
{
    // Setup progress timer
    m_progressTimer->setInterval(PROGRESS_UPDATE_INTERVAL);
    connect(m_progressTimer, &QTimer::timeout, this, &DiskImager::updateProgress);
    
    // Initialize buffers
    m_readBuffer.resize(DEFAULT_BUFFER_SIZE);
    m_writeBuffer.resize(DEFAULT_BUFFER_SIZE);
    
    // Register meta types for signal/slot system
    qRegisterMetaType<ImagingProgress>("ImagingProgress");
    qRegisterMetaType<HashAlgorithm>("HashAlgorithm");
    qRegisterMetaType<ImagingParameters>("ImagingParameters");
    
    PERF_LOG("DiskImager initialized");
}

/*
 * DiskImager Destructor
 */
DiskImager::~DiskImager()
{
    if (m_isRunning.load()) {
        cancelImaging();
    }
    cleanup();
    
    PERF_LOG("DiskImager destroyed");
}

/*
 * Device Discovery - Static method
 */
std::vector<DeviceInfo> DiskImager::discoverDevices()
{
    PERF_TIMER_START(discoverDevices);
    
#ifdef Q_OS_WIN
    auto devices = enumerateWindowsDevices();
#else
    auto devices = enumerateLinuxDevices();
#endif
    
    // Sort by device path for consistent ordering
    std::sort(devices.begin(), devices.end(), 
              [](const DeviceInfo& a, const DeviceInfo& b) {
                  return a.devicePath < b.devicePath;
              });
    
    PERF_TIMER_END(discoverDevices);
    ForensicLogger::instance().audit("device_discovery", "disk_imager", 
                                    QStringLiteral("Found %1 devices").arg(devices.size()));
    
    return devices;
}

/*
 * Get Device Information - Static method
 */
DeviceInfo DiskImager::getDeviceInfo(const QString& devicePath)
{
    DeviceInfo info;
    info.devicePath = devicePath;
    info.detectionTime = QDateTime::currentDateTime();
    
#ifdef Q_OS_WIN
    HANDLE handle = openWindowsDevice(devicePath);
    if (handle != INVALID_HANDLE_VALUE) {
        getWindowsDeviceInfo(handle, info);
        CloseHandle(handle);
    }
#else
    int fd = openLinuxDevice(devicePath);
    if (fd >= 0) {
        getLinuxDeviceInfo(fd, info);
        close(fd);
    }
#endif
    
    return info;
}

/*
 * Check Device Accessibility - Static method
 */
bool DiskImager::isDeviceAccessible(const QString& devicePath)
{
#ifdef Q_OS_WIN
    HANDLE handle = openWindowsDevice(devicePath);
    if (handle != INVALID_HANDLE_VALUE) {
        CloseHandle(handle);
        return true;
    }
#else
    int fd = openLinuxDevice(devicePath);
    if (fd >= 0) {
        close(fd);
        return true;
    }
#endif
    return false;
}

/*
 * Start Imaging Operation
 */
bool DiskImager::startImaging(const ImagingParameters& params)
{
    if (m_isRunning.load()) {
        return false; // Already running
    }
    
    // Validate parameters
    if (params.sourceDevice.isEmpty() || params.destinationPath.isEmpty()) {
        emit errorOccurred(tr("Source device and destination path must be specified"));
        return false;
    }
    
    if (!isDeviceAccessible(params.sourceDevice)) {
        emit errorOccurred(tr("Cannot access source device: %1").arg(params.sourceDevice));
        return false;
    }
    
    // Check available disk space
    QFileInfo destInfo(params.destinationPath);
    QStorageInfo storage(destInfo.absolutePath());
    DeviceInfo sourceInfo = getDeviceInfo(params.sourceDevice);
    
    if (storage.bytesAvailable() < sourceInfo.totalSize) {
        emit errorOccurred(tr("Insufficient disk space. Required: %1 GB, Available: %2 GB")
                          .arg(sourceInfo.totalSize / (1024*1024*1024))
                          .arg(storage.bytesAvailable() / (1024*1024*1024)));
        return false;
    }
    
    // Store parameters
    m_parameters = params;
    m_progress = ImagingProgress();
    m_stats = PerformanceStats();
    
    // Initialize progress
    m_progress.totalBytes = sourceInfo.totalSize;
    m_progress.currentOperation = tr("Initializing...");
    
    // Create worker thread
    m_workerThread = new QThread(this);
    m_worker = std::make_unique<ImagingWorker>(this, params);
    m_worker->moveToThread(m_workerThread);
    
    // Connect worker signals
    connect(m_workerThread, &QThread::started, m_worker.get(), &ImagingWorker::performImaging);
    connect(m_worker.get(), &ImagingWorker::finished, this, &DiskImager::handleWorkerFinished);
    connect(m_worker.get(), &ImagingWorker::error, this, &DiskImager::handleWorkerError);
    connect(m_worker.get(), &ImagingWorker::progressUpdate, this, &DiskImager::updateProgress);
    
    // Set flags and start
    m_isRunning.store(true);
    m_shouldCancel.store(false);
    m_isPaused.store(false);
    
    m_operationTimer.start();
    m_progressTimer->start();
    m_workerThread->start();
    
    emit imagingStarted(params.sourceDevice, params.destinationPath);
    ForensicLogger::instance().audit("imaging_started", "disk_imager", 
                                    QStringLiteral("Source: %1, Destination: %2")
                                    .arg(params.sourceDevice, params.destinationPath));
    
    return true;
}

/*
 * Start Imaging Asynchronously (Slot)
 */
void DiskImager::startImagingAsync(const ImagingParameters& params)
{
    startImaging(params);
}

/*
 * Pause Imaging
 */
void DiskImager::pauseImaging()
{
    if (m_isRunning.load() && !m_isPaused.load()) {
        m_isPaused.store(true);
        m_progressTimer->stop();
        emit imagingPaused();
        
        ForensicLogger::instance().audit("imaging_paused", "disk_imager", "Imaging operation paused");
    }
}

/*
 * Resume Imaging
 */
void DiskImager::resumeImaging()
{
    if (m_isRunning.load() && m_isPaused.load()) {
        m_isPaused.store(false);
        m_progressTimer->start();
        m_pauseCondition.wakeAll();
        emit imagingResumed();
        
        ForensicLogger::instance().audit("imaging_resumed", "disk_imager", "Imaging operation resumed");
    }
}

/*
 * Cancel Imaging
 */
void DiskImager::cancelImaging()
{
    if (m_isRunning.load()) {
        m_shouldCancel.store(true);
        
        if (m_isPaused.load()) {
            resumeImaging(); // Wake up if paused
        }
        
        if (m_workerThread && m_workerThread->isRunning()) {
            m_workerThread->wait(5000); // Wait up to 5 seconds
        }
        
        cleanup();
        emit imagingCancelled();
        
        ForensicLogger::instance().audit("imaging_cancelled", "disk_imager", "Imaging operation cancelled");
    }
}

/*
 * Get Current Progress
 */
ImagingProgress DiskImager::getProgress() const
{
    QMutexLocker locker(&m_progressMutex);
    return m_progress;
}

/*
 * Update Progress (Private Slot)
 */
void DiskImager::updateProgress()
{
    if (!m_isRunning.load()) {
        return;
    }
    
    QMutexLocker locker(&m_progressMutex);
    
    // Calculate transfer rates
    qint64 elapsed = m_operationTimer.elapsed();
    if (elapsed > 0) {
        m_progress.averageTransferRate = (m_progress.bytesProcessed * 1000) / elapsed;
        
        // Calculate current transfer rate from recent samples
        qint64 recentSum = std::accumulate(m_recentTransferRates.begin(), 
                                         m_recentTransferRates.end(), 0LL);
        m_progress.currentTransferRate = recentSum / TRANSFER_RATE_SAMPLES;
    }
    
    // Calculate elapsed and estimated time
    m_progress.elapsedTime = QTime(0, 0).addMSecs(elapsed);
    
    if (m_progress.currentTransferRate > 0) {
        qint64 remainingBytes = m_progress.totalBytes - m_progress.bytesProcessed;
        qint64 remainingSeconds = remainingBytes / m_progress.currentTransferRate;
        m_progress.estimatedTimeRemaining = QTime(0, 0).addSecs(remainingSeconds);
    }
    
    emit progressUpdated(m_progress);
}

/*
 * Handle Worker Finished
 */
void DiskImager::handleWorkerFinished()
{
    m_progressTimer->stop();
    
    // Final progress update
    updateProgress();
    
    // Finalize hash calculations
    finalizeHashCalculators();
    
    // Log completion
    qint64 totalTime = m_operationTimer.elapsed();
    QString message = tr("Imaging completed successfully in %1 seconds. "
                        "Average speed: %2 MB/s")
                     .arg(totalTime / 1000.0, 0, 'f', 1)
                     .arg(m_progress.averageTransferRate / (1024*1024), 0, 'f', 2);
    
    ForensicLogger::instance().audit("imaging_completed", "disk_imager", message);
    
    cleanup();
    emit imagingCompleted(true, message);
}

/*
 * Handle Worker Error
 */
void DiskImager::handleWorkerError(const QString& error)
{
    m_progressTimer->stop();
    
    ForensicLogger::instance().error("imaging_error", "disk_imager", error);
    
    cleanup();
    emit imagingCompleted(false, error);
    emit errorOccurred(error);
}

/*
 * Initialize Source Device
 */
bool DiskImager::initializeSource(const QString& devicePath)
{
#ifdef Q_OS_WIN
    m_sourceHandle = openWindowsDevice(devicePath);
    return m_sourceHandle != INVALID_HANDLE_VALUE;
#else
    m_sourceHandle = openLinuxDevice(devicePath);
    return m_sourceHandle >= 0;
#endif
}

/*
 * Initialize Destination
 */
bool DiskImager::initializeDestination(const QString& destinationPath, ImageFormat format)
{
    m_imageWriter = ImageWriter::createWriter(format);
    if (!m_imageWriter) {
        return false;
    }
    
    return m_imageWriter->initialize(destinationPath, m_parameters);
}

/*
 * Initialize Hash Calculators
 */
void DiskImager::initializeHashCalculators()
{
    m_hashCalculators.clear();
    m_calculatedHashes.clear();
    
    for (HashAlgorithm algorithm : m_parameters.hashAlgorithms) {
        auto calculator = std::make_unique<HashCalculator>(algorithm, this);
        m_hashCalculators.push_back(std::move(calculator));
    }
}

/*
 * Update Hash Calculators
 */
void DiskImager::updateHashCalculators(const QByteArray& data)
{
    for (auto& calculator : m_hashCalculators) {
        calculator->addData(data);
    }
}

/*
 * Finalize Hash Calculators
 */
void DiskImager::finalizeHashCalculators()
{
    m_calculatedHashes.clear();
    
    for (auto& calculator : m_hashCalculators) {
        HashResult result = calculator->getResult();
        m_calculatedHashes.push_back(result);
        
        emit hashCalculated(result.algorithm, result.hexString);
        
        ForensicLogger::instance().audit("hash_calculated", "disk_imager",
                                        QStringLiteral("Algorithm: %1, Hash: %2")
                                        .arg(static_cast<int>(result.algorithm))
                                        .arg(result.hexString));
    }
}

/*
 * Get Calculated Hashes
 */
std::vector<HashResult> DiskImager::getCalculatedHashes() const
{
    return m_calculatedHashes;
}

/*
 * Cleanup Resources
 */
void DiskImager::cleanup()
{
    m_isRunning.store(false);
    m_isPaused.store(false);
    m_shouldCancel.store(false);
    
    if (m_workerThread) {
        m_workerThread->quit();
        m_workerThread->wait();
        m_workerThread->deleteLater();
        m_workerThread = nullptr;
    }
    
    m_worker.reset();
    
#ifdef Q_OS_WIN
    if (m_sourceHandle != INVALID_HANDLE_VALUE) {
        CloseHandle(m_sourceHandle);
        m_sourceHandle = INVALID_HANDLE_VALUE;
    }
#else
    if (m_sourceHandle >= 0) {
        close(m_sourceHandle);
        m_sourceHandle = -1;
    }
#endif
    
    m_destinationFile.reset();
    m_imageWriter.reset();
    m_hashCalculators.clear();
}

#ifdef Q_OS_WIN
/*
 * Windows-specific device operations
 */
HANDLE DiskImager::openWindowsDevice(const QString& devicePath)
{
    QString winPath = devicePath;
    if (!winPath.startsWith("\\\\.\\")) {
        winPath = "\\\\.\\" + devicePath;
    }
    
    HANDLE handle = CreateFileW(
        reinterpret_cast<const wchar_t*>(winPath.utf16()),
        GENERIC_READ,
        FILE_SHARE_READ | FILE_SHARE_WRITE,
        nullptr,
        OPEN_EXISTING,
        FILE_FLAG_NO_BUFFERING | FILE_FLAG_SEQUENTIAL_SCAN,
        nullptr
    );
    
    return handle;
}

bool DiskImager::getWindowsDeviceInfo(HANDLE device, DeviceInfo& info)
{
    // Get disk geometry
    DISK_GEOMETRY_EX diskGeometry;
    DWORD bytesReturned;
    
    if (DeviceIoControl(device, IOCTL_DISK_GET_DRIVE_GEOMETRY_EX,
                       nullptr, 0, &diskGeometry, sizeof(diskGeometry),
                       &bytesReturned, nullptr)) {
        info.totalSize = diskGeometry.DiskSize.QuadPart;
        info.sectorSize = diskGeometry.Geometry.BytesPerSector;
        info.totalSectors = info.totalSize / info.sectorSize;
    }
    
    // Get device properties
    STORAGE_PROPERTY_QUERY query = {};
    query.PropertyId = StorageDeviceProperty;
    query.QueryType = PropertyStandardQuery;
    
    STORAGE_DEVICE_DESCRIPTOR descriptor = {};
    if (DeviceIoControl(device, IOCTL_STORAGE_QUERY_PROPERTY,
                       &query, sizeof(query), &descriptor, sizeof(descriptor),
                       &bytesReturned, nullptr)) {
        
        if (descriptor.VendorIdOffset > 0) {
            char* buffer = reinterpret_cast<char*>(&descriptor);
            info.vendor = QString::fromLatin1(buffer + descriptor.VendorIdOffset).trimmed();
        }
        
        if (descriptor.ProductIdOffset > 0) {
            char* buffer = reinterpret_cast<char*>(&descriptor);
            info.model = QString::fromLatin1(buffer + descriptor.ProductIdOffset).trimmed();
        }
        
        if (descriptor.SerialNumberOffset > 0) {
            char* buffer = reinterpret_cast<char*>(&descriptor);
            info.serialNumber = QString::fromLatin1(buffer + descriptor.SerialNumberOffset).trimmed();
        }
        
        info.isRemovable = (descriptor.RemovableMedia == TRUE);
    }
    
    return true;
}

std::vector<DeviceInfo> DiskImager::enumerateWindowsDevices()
{
    std::vector<DeviceInfo> devices;
    
    // Enumerate disk drives
    for (int i = 0; i < 64; ++i) {
        QString devicePath = QString("\\\\.\\PhysicalDrive%1").arg(i);
        
        HANDLE handle = openWindowsDevice(devicePath);
        if (handle != INVALID_HANDLE_VALUE) {
            DeviceInfo info;
            info.devicePath = devicePath;
            info.detectionTime = QDateTime::currentDateTime();
            
            if (getWindowsDeviceInfo(handle, info)) {
                devices.push_back(info);
            }
            
            CloseHandle(handle);
        }
    }
    
    return devices;
}
#endif

#ifdef Q_OS_LINUX
/*
 * Linux-specific device operations
 */
int DiskImager::openLinuxDevice(const QString& devicePath)
{
    return open(devicePath.toLocal8Bit().constData(), O_RDONLY | O_DIRECT);
}

bool DiskImager::getLinuxDeviceInfo(int fd, DeviceInfo& info)
{
    // Get device size
    if (ioctl(fd, BLKGETSIZE64, &info.totalSize) == 0) {
        // Get sector size
        int sectorSize = 512;
        if (ioctl(fd, BLKSSZGET, &sectorSize) == 0) {
            info.sectorSize = sectorSize;
            info.totalSectors = info.totalSize / info.sectorSize;
        }
    }
    
    // Try to get device model from sysfs
    QString sysfsPath = QString("/sys/block/%1/device/model")
                       .arg(QFileInfo(info.devicePath).baseName());
    
    QFile modelFile(sysfsPath);
    if (modelFile.open(QIODevice::ReadOnly)) {
        info.model = QString::fromLocal8Bit(modelFile.readAll()).trimmed();
    }
    
    return true;
}

std::vector<DeviceInfo> DiskImager::enumerateLinuxDevices()
{
    std::vector<DeviceInfo> devices;
    
    QDir sysBlock("/sys/block");
    QStringList blockDevices = sysBlock.entryList(QDir::Dirs | QDir::NoDotAndDotDot);
    
    for (const QString& device : blockDevices) {
        if (device.startsWith("sd") || device.startsWith("hd") || 
            device.startsWith("nvme") || device.startsWith("mmcblk")) {
            
            QString devicePath = "/dev/" + device;
            int fd = openLinuxDevice(devicePath);
            
            if (fd >= 0) {
                DeviceInfo info;
                info.devicePath = devicePath;
                info.detectionTime = QDateTime::currentDateTime();
                
                if (getLinuxDeviceInfo(fd, info)) {
                    devices.push_back(info);
                }
                
                close(fd);
            }
        }
    }
    
    return devices;
}
#endif

/*
 * Imaging Worker Implementation
 */
void DiskImager::ImagingWorker::performImaging()
{
    try {
        // Initialize source
        if (!m_imager->initializeSource(m_params.sourceDevice)) {
            emit error(QObject::tr("Failed to open source device: %1").arg(m_params.sourceDevice));
            return;
        }
        
        // Initialize destination
        if (!m_imager->initializeDestination(m_params.destinationPath, m_params.format)) {
            emit error(QObject::tr("Failed to create destination image: %1").arg(m_params.destinationPath));
            return;
        }
        
        // Initialize hash calculators
        m_imager->initializeHashCalculators();
        
        // Main imaging loop
        const qint64 blockSize = m_params.blockSize;
        qint64 totalBytesRead = 0;
        
        DeviceInfo sourceInfo = DiskImager::getDeviceInfo(m_params.sourceDevice);
        const qint64 totalSize = sourceInfo.totalSize;
        
        std::vector<char> buffer(blockSize);
        auto startTime = std::chrono::high_resolution_clock::now();
        
        while (totalBytesRead < totalSize && !m_imager->m_shouldCancel.load()) {
            // Handle pause
            if (m_imager->m_isPaused.load()) {
                QMutexLocker locker(&m_imager->m_progressMutex);
                m_imager->m_pauseCondition.wait(&m_imager->m_progressMutex);
            }
            
            qint64 bytesToRead = std::min(blockSize, totalSize - totalBytesRead);
            
#ifdef Q_OS_WIN
            DWORD bytesRead = 0;
            BOOL success = ReadFile(m_imager->m_sourceHandle, buffer.data(), 
                                  static_cast<DWORD>(bytesToRead), &bytesRead, nullptr);
            
            if (!success || bytesRead == 0) {
                if (!m_imager->handleReadError(totalBytesRead, bytesToRead)) {
                    emit error(QObject::tr("Read error at offset %1").arg(totalBytesRead));
                    return;
                }
                continue;
            }
#else
            ssize_t bytesRead = read(m_imager->m_sourceHandle, buffer.data(), bytesToRead);
            
            if (bytesRead <= 0) {
                if (!m_imager->handleReadError(totalBytesRead, bytesToRead)) {
                    emit error(QObject::tr("Read error at offset %1").arg(totalBytesRead));
                    return;
                }
                continue;
            }
#endif
            
            // Update hash calculators
            QByteArray data(buffer.data(), bytesRead);
            m_imager->updateHashCalculators(data);
            
            // Write to destination
            if (!m_imager->m_imageWriter->writeBlock(data, totalBytesRead)) {
                emit error(QObject::tr("Write error: %1").arg(m_imager->m_imageWriter->getError()));
                return;
            }
            
            totalBytesRead += bytesRead;
            
            // Update progress
            {
                QMutexLocker locker(&m_imager->m_progressMutex);
                m_imager->m_progress.bytesProcessed = totalBytesRead;
                m_imager->m_progress.currentOperation = QObject::tr("Imaging...");
                
                // Calculate current transfer rate
                auto currentTime = std::chrono::high_resolution_clock::now();
                auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(currentTime - startTime);
                
                if (elapsed.count() > 0) {
                    qint64 rate = (totalBytesRead * 1000) / elapsed.count();
                    m_imager->m_recentTransferRates[m_imager->m_transferRateIndex++ % TRANSFER_RATE_SAMPLES] = rate;
                }
            }
            
            emit progressUpdate();
        }
        
        // Finalize image
        if (!m_imager->m_imageWriter->finalize()) {
            emit error(QObject::tr("Failed to finalize image: %1").arg(m_imager->m_imageWriter->getError()));
            return;
        }
        
        emit finished();
        
    } catch (const std::exception& e) {
        emit error(QObject::tr("Unexpected error: %1").arg(e.what()));
    }
}

/*
 * Handle Read Error with Retry Logic
 */
bool DiskImager::handleReadError(qint64 offset, qint64 size)
{
    for (int retry = 0; retry < m_parameters.maxRetries; ++retry) {
        emit retryAttempt(retry + 1, m_parameters.maxRetries);
        
        // Wait before retry
        if (m_parameters.retryDelay > 0) {
            std::this_thread::sleep_for(std::chrono::milliseconds(m_parameters.retryDelay));
        }
        
        // Attempt to seek and read again
#ifdef Q_OS_WIN
        LARGE_INTEGER pos;
        pos.QuadPart = offset;
        if (SetFilePointerEx(m_sourceHandle, pos, nullptr, FILE_BEGIN)) {
            return true; // Retry the read
        }
#else
        if (lseek(m_sourceHandle, offset, SEEK_SET) == offset) {
            return true; // Retry the read
        }
#endif
    }
    
    // If we get here, all retries failed
    emit badSectorDetected(offset / m_parameters.blockSize, offset);
    
    if (m_parameters.skipBadSectors) {
        if (m_parameters.zerofillBadSectors) {
            // Write zeros for bad sector
            QByteArray zeroData(size, 0);
            updateHashCalculators(zeroData);
            m_imageWriter->writeBlock(zeroData, offset);
        }
        
        m_stats.badSectorsSkipped++;
        return true; // Continue with next block
    }
    
    return false; // Stop imaging
}

} // namespace PhoenixDRS

#include "DiskImager.moc"