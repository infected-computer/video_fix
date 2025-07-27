/*
 * PhoenixDRS Professional - Advanced File Carving Engine Implementation
 * מימוש מנוע חיתוך קבצים מתקדם
 */

#include "FileCarver.h"
#include "ForensicLogger.h"
#include <QApplication>
#include <QMessageBox>
#include <QStandardPaths>
#include <QCryptographicHash>
#include <QtConcurrent>
#include <QMutexLocker>
#include <QFileInfo>
#include <QTextStream>

#include <chrono>
#include <thread>
#include <algorithm>
#include <numeric>
#include <bitset>
#include <cstring>

// SIMD intrinsics
#ifdef _MSC_VER
#include <intrin.h>
#else
#include <cpuid.h>
#endif

namespace PhoenixDRS {

/*
 * Internal worker class for carving operations
 */
class FileCarver::CarvingWorker : public QObject
{
    Q_OBJECT

public:
    explicit CarvingWorker(FileCarver* parent, const CarvingParameters& params)
        : QObject(nullptr), m_carver(parent), m_params(params) {}

public slots:
    void performCarving();

signals:
    void finished();
    void error(const QString& message);
    void progressUpdate();
    void fileFound(const CarvedFile& file);

private:
    FileCarver* m_carver;
    CarvingParameters m_params;
    
    bool processChunk(qint64 chunkOffset, qint64 chunkSize);
    void searchSignaturesInChunk(const QByteArray& chunkData, qint64 baseOffset);
};

/*
 * FileCarver Constructor
 */
FileCarver::FileCarver(QObject* parent)
    : QObject(parent)
    , m_workerThread(nullptr)
    , m_progressTimer(new QTimer(this))
    , m_signatureDb(std::make_unique<SignatureDatabase>())
    , m_patternMatcher(std::make_unique<PatternMatcher>())
    , m_fileValidator(std::make_unique<FileValidator>())
    , m_fragmentReconstructor(std::make_unique<FragmentReconstructor>())
{
    // Setup progress timer
    m_progressTimer->setInterval(PROGRESS_UPDATE_INTERVAL);
    connect(m_progressTimer, &QTimer::timeout, this, &FileCarver::updateProgress);
    
    // Register meta types
    qRegisterMetaType<CarvingProgress>("CarvingProgress");
    qRegisterMetaType<CarvedFile>("CarvedFile");
    qRegisterMetaType<CarvingParameters>("CarvingParameters");
    
    // Detect SIMD support
    setupSIMDAcceleration();
    
    // Load default signatures
    loadDefaultSignatures();
    
    PERF_LOG("FileCarver initialized with SIMD support - SSE2: %s, AVX2: %s", 
             m_hasSSE2 ? "Yes" : "No", m_hasAVX2 ? "Yes" : "No");
}

/*
 * FileCarver Destructor
 */
FileCarver::~FileCarver()
{
    if (m_isRunning.load()) {
        cancelCarving();
    }
    cleanupCarving();
    
    PERF_LOG("FileCarver destroyed");
}

/*
 * Setup SIMD Acceleration
 */
void FileCarver::setupSIMDAcceleration()
{
#ifdef _MSC_VER
    int cpuInfo[4];
    __cpuid(cpuInfo, 1);
    m_hasSSE2 = (cpuInfo[3] & (1 << 26)) != 0;
    
    __cpuid(cpuInfo, 7);
    m_hasAVX2 = (cpuInfo[1] & (1 << 5)) != 0;
#else
    unsigned int eax, ebx, ecx, edx;
    
    // Check SSE2
    if (__get_cpuid(1, &eax, &ebx, &ecx, &edx)) {
        m_hasSSE2 = (edx & (1 << 26)) != 0;
    }
    
    // Check AVX2
    if (__get_cpuid_max(0, nullptr) >= 7) {
        __get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx);
        m_hasAVX2 = (ebx & (1 << 5)) != 0;
    }
#endif
    
    // Configure pattern matcher
    m_patternMatcher->setUseSSE2(m_hasSSE2);
    m_patternMatcher->setUseAVX2(m_hasAVX2);
}

/*
 * Load Default Signatures
 */
bool FileCarver::loadDefaultSignatures()
{
    m_signatureDb->loadDefaultSignatures();
    
    // Load signatures into pattern matcher
    auto signatures = m_signatureDb->getAllSignatures();
    m_patternMatcher->clear();
    
    int patternIndex = 0;
    for (const auto& sig : signatures) {
        if (!sig.headerSignature.isEmpty()) {
            m_patternMatcher->addPattern(sig.headerSignature, patternIndex++);
            m_signatures[sig.name] = sig;
        }
    }
    
    m_patternMatcher->buildAutomaton();
    
    ForensicLogger::instance().info("signatures_loaded", "file_carver",
                                   QStringLiteral("Loaded %1 signatures").arg(signatures.size()));
    
    return !signatures.empty();
}

/*
 * Load Signatures Database
 */
bool FileCarver::loadSignaturesDatabase(const QString& path)
{
    if (!m_signatureDb->loadFromFile(path)) {
        return false;
    }
    
    // Reload signatures into pattern matcher
    return loadDefaultSignatures();
}

/*
 * Start Carving Operation
 */
bool FileCarver::startCarving(const CarvingParameters& params)
{
    if (m_isRunning.load()) {
        return false; // Already running
    }
    
    // Validate parameters
    if (params.sourceImagePath.isEmpty() || params.outputDirectory.isEmpty()) {
        emit errorOccurred(tr("Source image and output directory must be specified"));
        return false;
    }
    
    if (!QFile::exists(params.sourceImagePath)) {
        emit errorOccurred(tr("Source image file does not exist: %1").arg(params.sourceImagePath));
        return false;
    }
    
    // Create output directory
    QDir outputDir;
    if (!outputDir.mkpath(params.outputDirectory)) {
        emit errorOccurred(tr("Cannot create output directory: %1").arg(params.outputDirectory));
        return false;
    }
    
    // Store parameters
    m_parameters = params;
    m_progress = CarvingProgress();
    m_statistics = CarvingStatistics();
    m_carvedFiles.clear();
    m_processedOffsets.clear();
    
    // Initialize source file
    m_sourceFile = std::make_unique<QFile>(params.sourceImagePath);
    if (!m_sourceFile->open(QIODevice::ReadOnly)) {
        emit errorOccurred(tr("Cannot open source image: %1").arg(m_sourceFile->errorString()));
        return false;
    }
    
    // Setup progress
    m_progress.totalBytes = m_sourceFile->size();
    m_progress.totalChunks = (m_progress.totalBytes + params.chunkSize - 1) / params.chunkSize;
    m_progress.currentOperation = tr("Initializing carving...");
    
    // Optimize processing parameters
    optimizeProcessingParameters();
    
    // Create worker thread
    m_workerThread = new QThread(this);
    m_worker = std::make_unique<CarvingWorker>(this, params);
    m_worker->moveToThread(m_workerThread);
    
    // Connect worker signals
    connect(m_workerThread, &QThread::started, m_worker.get(), &CarvingWorker::performCarving);
    connect(m_worker.get(), &CarvingWorker::finished, this, &FileCarver::handleWorkerFinished);
    connect(m_worker.get(), &CarvingWorker::error, this, &FileCarver::handleWorkerError);
    connect(m_worker.get(), &CarvingWorker::progressUpdate, this, &FileCarver::updateProgress);
    connect(m_worker.get(), &CarvingWorker::fileFound, this, &FileCarver::fileFound);
    
    // Set flags and start
    m_isRunning.store(true);
    m_shouldCancel.store(false);
    m_isPaused.store(false);
    
    m_operationTimer.start();
    m_progressTimer->start();
    m_workerThread->start();
    
    emit carvingStarted(params.sourceImagePath, params.outputDirectory);
    ForensicLogger::instance().audit("carving_started", "file_carver",
                                    QStringLiteral("Source: %1, Output: %2")
                                    .arg(params.sourceImagePath, params.outputDirectory));
    
    return true;
}

/*
 * Start Carving Asynchronously (Slot)
 */
void FileCarver::startCarvingAsync(const CarvingParameters& params)
{
    startCarving(params);
}

/*
 * Pause Carving
 */
void FileCarver::pauseCarving()
{
    if (m_isRunning.load() && !m_isPaused.load()) {
        m_isPaused.store(true);
        m_progressTimer->stop();
        emit carvingPaused();
        
        ForensicLogger::instance().audit("carving_paused", "file_carver", "Carving operation paused");
    }
}

/*
 * Resume Carving
 */
void FileCarver::resumeCarving()
{
    if (m_isRunning.load() && m_isPaused.load()) {
        m_isPaused.store(false);
        m_progressTimer->start();
        m_pauseCondition.wakeAll();
        emit carvingResumed();
        
        ForensicLogger::instance().audit("carving_resumed", "file_carver", "Carving operation resumed");
    }
}

/*
 * Cancel Carving
 */
void FileCarver::cancelCarving()
{
    if (m_isRunning.load()) {
        m_shouldCancel.store(true);
        
        if (m_isPaused.load()) {
            resumeCarving(); // Wake up if paused
        }
        
        if (m_workerThread && m_workerThread->isRunning()) {
            m_workerThread->wait(5000); // Wait up to 5 seconds
        }
        
        cleanupCarving();
        emit carvingCancelled();
        
        ForensicLogger::instance().audit("carving_cancelled", "file_carver", "Carving operation cancelled");
    }
}

/*
 * Get Current Progress
 */
CarvingProgress FileCarver::getProgress() const
{
    QMutexLocker locker(&m_progressMutex);
    return m_progress;
}

/*
 * Update Progress (Private Slot)
 */
void FileCarver::updateProgress()
{
    if (!m_isRunning.load()) {
        return;
    }
    
    QMutexLocker locker(&m_progressMutex);
    
    // Calculate processing rates
    qint64 elapsed = m_operationTimer.elapsed();
    if (elapsed > 0) {
        m_progress.averageRate = (m_progress.bytesProcessed * 1000) / elapsed;
        
        // Calculate current rate from recent samples
        qint64 recentSum = std::accumulate(m_recentRates.begin(), m_recentRates.end(), 0LL);
        m_progress.processingRate = recentSum / m_recentRates.size();
    }
    
    // Calculate elapsed and estimated time
    m_progress.elapsedTime = QTime(0, 0).addMSecs(elapsed);
    
    if (m_progress.processingRate > 0) {
        qint64 remainingBytes = m_progress.totalBytes - m_progress.bytesProcessed;
        qint64 remainingSeconds = remainingBytes / m_progress.processingRate;
        m_progress.estimatedTimeRemaining = QTime(0, 0).addSecs(remainingSeconds);
    }
    
    emit progressUpdated(m_progress);
}

/*
 * Handle Worker Finished
 */
void FileCarver::handleWorkerFinished()
{
    m_progressTimer->stop();
    updateProgress();
    
    // Calculate final statistics
    qint64 totalTime = m_operationTimer.elapsed();
    m_statistics.totalProcessingTime = QTime(0, 0).addMSecs(totalTime);
    m_statistics.averageProcessingRate = m_progress.averageRate;
    
    // Calculate file size statistics
    if (!m_carvedFiles.empty()) {
        std::vector<qint64> sizes;
        for (const auto& file : m_carvedFiles) {
            sizes.push_back(file.fileSize);
            m_statistics.fileTypeDistribution[file.fileType]++;
        }
        
        std::sort(sizes.begin(), sizes.end());
        m_statistics.smallestFileSize = sizes.front();
        m_statistics.largestFileSize = sizes.back();
        m_statistics.averageFileSize = std::accumulate(sizes.begin(), sizes.end(), 0LL) / sizes.size();
    }
    
    QString message = tr("Carving completed successfully. Found %1 files in %2 seconds")
                     .arg(m_carvedFiles.size())
                     .arg(totalTime / 1000.0, 0, 'f', 1);
    
    ForensicLogger::instance().audit("carving_completed", "file_carver", message);
    
    cleanupCarving();
    emit carvingCompleted(true, message);
}

/*
 * Handle Worker Error
 */
void FileCarver::handleWorkerError(const QString& error)
{
    m_progressTimer->stop();
    
    ForensicLogger::instance().error("carving_error", "file_carver", error);
    
    cleanupCarving();
    emit carvingCompleted(false, error);
    emit errorOccurred(error);
}

/*
 * Optimize Processing Parameters
 */
void FileCarver::optimizeProcessingParameters()
{
    // Determine optimal worker thread count
    if (m_parameters.workerThreads <= 0) {
        m_parameters.workerThreads = std::max(1, QThread::idealThreadCount());
    }
    
    // Optimize chunk size based on available memory and file size
    qint64 availableMemory = 2LL * 1024 * 1024 * 1024; // Assume 2GB available
    qint64 optimalChunkSize = availableMemory / (m_parameters.workerThreads * 4);
    
    if (optimalChunkSize < m_parameters.chunkSize) {
        m_parameters.chunkSize = std::max(DEFAULT_CHUNK_SIZE / 4, optimalChunkSize);
    }
    
    // Ensure chunk size is reasonable
    m_parameters.chunkSize = std::min(m_parameters.chunkSize, 
                                     std::max(DEFAULT_CHUNK_SIZE, m_progress.totalBytes / 100));
    
    // Setup processing buffer
    m_processingBuffer.resize(m_parameters.chunkSize + m_parameters.overlapSize);
    
    PERF_LOG("Optimized parameters - Workers: %d, Chunk size: %lld MB", 
             m_parameters.workerThreads, m_parameters.chunkSize / (1024*1024));
}

/*
 * Cleanup Carving Resources
 */
void FileCarver::cleanupCarving()
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
    m_sourceFile.reset();
    m_processingBuffer.clear();
}

/*
 * Get Carved Files
 */
std::vector<CarvedFile> FileCarver::getCarvedFiles() const
{
    return m_carvedFiles;
}

/*
 * Get Carved Files by Type
 */
std::vector<CarvedFile> FileCarver::getCarvedFilesByType(const QString& fileType) const
{
    std::vector<CarvedFile> filtered;
    std::copy_if(m_carvedFiles.begin(), m_carvedFiles.end(), 
                 std::back_inserter(filtered),
                 [&fileType](const CarvedFile& file) {
                     return file.fileType == fileType;
                 });
    return filtered;
}

/*
 * Recover File
 */
bool FileCarver::recoverFile(qint64 startOffset, const FileSignature& signature)
{
    if (!m_sourceFile || !m_sourceFile->isOpen()) {
        return false;
    }
    
    // Seek to start position
    if (!m_sourceFile->seek(startOffset)) {
        return false;
    }
    
    // Determine file size
    qint64 fileSize = signature.maxFileSize;
    qint64 endOffset = startOffset + fileSize;
    
    // Look for footer if signature has one
    if (!signature.footerSignature.isEmpty()) {
        QByteArray searchData = m_sourceFile->read(fileSize);
        if (searchData.isEmpty()) {
            return false;
        }
        
        int footerPos = searchData.lastIndexOf(signature.footerSignature);
        if (footerPos >= 0) {
            fileSize = footerPos + signature.footerSignature.size();
            endOffset = startOffset + fileSize;
        }
    }
    
    // Read file data
    if (!m_sourceFile->seek(startOffset)) {
        return false;
    }
    
    QByteArray fileData = m_sourceFile->read(fileSize);
    if (fileData.isEmpty()) {
        return false;
    }
    
    // Validate file if required
    if (m_parameters.validateFiles) {
        if (!m_fileValidator->validateFileContent(fileData, signature)) {
            return false;
        }
    }
    
    // Generate output filename
    QString fileName = QString("%1_%2.%3")
                      .arg(signature.name)
                      .arg(startOffset, 0, 16)
                      .arg(signature.extension);
    
    QString outputPath = QDir(m_parameters.outputDirectory).filePath(fileName);
    
    // Write recovered file
    QFile outputFile(outputPath);
    if (!outputFile.open(QIODevice::WriteOnly)) {
        return false;
    }
    
    if (outputFile.write(fileData) != fileData.size()) {
        return false;
    }
    
    outputFile.close();
    
    // Create carved file record
    CarvedFile carvedFile;
    carvedFile.originalPath = outputPath;
    carvedFile.fileName = fileName;
    carvedFile.fileType = signature.name;
    carvedFile.extension = signature.extension;
    carvedFile.startOffset = startOffset;
    carvedFile.endOffset = endOffset;
    carvedFile.fileSize = fileData.size();
    carvedFile.recoveryTime = QDateTime::currentDateTime();
    carvedFile.isComplete = true;
    carvedFile.isFragmented = false;
    carvedFile.confidenceScore = m_fileValidator->calculateConfidenceScore(fileData, signature);
    
    // Calculate hashes if required
    if (m_parameters.calculateHashes) {
        QCryptographicHash md5(QCryptographicHash::Md5);
        QCryptographicHash sha256(QCryptographicHash::Sha256);
        
        md5.addData(fileData);
        sha256.addData(fileData);
        
        carvedFile.md5Hash = md5.result().toHex().toUpper();
        carvedFile.sha256Hash = sha256.result().toHex().toUpper();
    }
    
    // Extract metadata
    carvedFile.metadata = m_fileValidator->extractMetadata(outputPath, signature);
    
    // Validate carved file
    if (m_parameters.validateFiles) {
        bool isValid = m_fileValidator->validateFile(outputPath, signature);
        carvedFile.validationStatus = isValid ? "Valid" : "Invalid";
        emit fileValidated(carvedFile, isValid);
    }
    
    // Add to results
    m_carvedFiles.push_back(carvedFile);
    
    // Update statistics
    {
        QMutexLocker locker(&m_progressMutex);
        m_progress.filesRecovered++;
        if (m_parameters.validateFiles) {
            m_progress.filesValidated++;
        }
    }
    
    emit fileRecovered(carvedFile);
    
    ForensicLogger::instance().info("file_recovered", "file_carver",
                                   QStringLiteral("Type: %1, Size: %2, Offset: 0x%3")
                                   .arg(signature.name)
                                   .arg(fileData.size())
                                   .arg(startOffset, 0, 16));
    
    return true;
}

/*
 * Calculate File Hash
 */
QString FileCarver::calculateFileHash(const QString& filePath, const QString& algorithm)
{
    QFile file(filePath);
    if (!file.open(QIODevice::ReadOnly)) {
        return QString();
    }
    
    QCryptographicHash::Algorithm alg;
    if (algorithm.toUpper() == "MD5") {
        alg = QCryptographicHash::Md5;
    } else if (algorithm.toUpper() == "SHA1") {
        alg = QCryptographicHash::Sha1;
    } else if (algorithm.toUpper() == "SHA256") {
        alg = QCryptographicHash::Sha256;
    } else {
        return QString();
    }
    
    QCryptographicHash hash(alg);
    
    const qint64 bufferSize = 64 * 1024;
    while (!file.atEnd()) {
        QByteArray data = file.read(bufferSize);
        hash.addData(data);
    }
    
    return hash.result().toHex().toUpper();
}

/*
 * Carving Worker Implementation
 */
void FileCarver::CarvingWorker::performCarving()
{
    try {
        auto startTime = std::chrono::high_resolution_clock::now();
        
        // Process chunks
        qint64 currentOffset = 0;
        qint64 chunkNumber = 0;
        
        while (currentOffset < m_carver->m_progress.totalBytes && !m_carver->m_shouldCancel.load()) {
            // Handle pause
            if (m_carver->m_isPaused.load()) {
                QMutexLocker locker(&m_carver->m_progressMutex);
                m_carver->m_pauseCondition.wait(&m_carver->m_progressMutex);
            }
            
            qint64 chunkSize = std::min(m_params.chunkSize, 
                                       m_carver->m_progress.totalBytes - currentOffset);
            
            if (!processChunk(currentOffset, chunkSize)) {
                emit error(QObject::tr("Failed to process chunk at offset %1").arg(currentOffset));
                return;
            }
            
            currentOffset += chunkSize;
            chunkNumber++;
            
            // Update progress
            {
                QMutexLocker locker(&m_carver->m_progressMutex);
                m_carver->m_progress.bytesProcessed = currentOffset;
                m_carver->m_progress.currentChunk = chunkNumber;
                m_carver->m_progress.currentOperation = QObject::tr("Processing chunk %1 of %2...")
                                                       .arg(chunkNumber).arg(m_carver->m_progress.totalChunks);
                
                // Calculate current processing rate
                auto currentTime = std::chrono::high_resolution_clock::now();
                auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(currentTime - startTime);
                
                if (elapsed.count() > 0) {
                    qint64 rate = (currentOffset * 1000) / elapsed.count();
                    m_carver->m_recentRates[m_carver->m_rateIndex++ % m_carver->m_recentRates.size()] = rate;
                }
            }
            
            emit progressUpdate();
            emit m_carver->chunkProcessed(chunkNumber, m_carver->m_progress.totalChunks);
        }
        
        emit finished();
        
    } catch (const std::exception& e) {
        emit error(QObject::tr("Unexpected error: %1").arg(e.what()));
    }
}

/*
 * Process Single Chunk
 */
bool FileCarver::CarvingWorker::processChunk(qint64 chunkOffset, qint64 chunkSize)
{
    // Seek to chunk position
    if (!m_carver->m_sourceFile->seek(chunkOffset)) {
        return false;
    }
    
    // Read chunk data with overlap
    qint64 readSize = chunkSize;
    if (chunkOffset + chunkSize < m_carver->m_progress.totalBytes) {
        readSize += m_params.overlapSize;
    }
    
    QByteArray chunkData = m_carver->m_sourceFile->read(readSize);
    if (chunkData.isEmpty()) {
        return false;
    }
    
    // Search for signatures in this chunk
    searchSignaturesInChunk(chunkData, chunkOffset);
    
    return true;
}

/*
 * Search Signatures in Chunk
 */
void FileCarver::CarvingWorker::searchSignaturesInChunk(const QByteArray& chunkData, qint64 baseOffset)
{
    // Use pattern matcher to find all signature matches
    auto matches = m_carver->m_patternMatcher->findAllPatterns(chunkData, baseOffset);
    
    for (const auto& match : matches) {
        // Skip if we've already processed this offset
        if (m_carver->m_processedOffsets.count(match.offset)) {
            continue;
        }
        
        m_carver->m_processedOffsets.insert(match.offset);
        
        // Find corresponding signature
        QString signatureName;
        for (const auto& [name, signature] : m_carver->m_signatures) {
            if (signature.headerSignature == match.pattern) {
                signatureName = name;
                break;
            }
        }
        
        if (signatureName.isEmpty()) {
            continue;
        }
        
        const FileSignature& signature = m_carver->m_signatures[signatureName];
        
        // Check file type filters
        if (!m_params.fileTypes.isEmpty() && !m_params.fileTypes.contains(signature.name)) {
            continue;
        }
        
        if (m_params.excludeTypes.contains(signature.name)) {
            continue;
        }
        
        // Create preliminary carved file record
        CarvedFile foundFile;
        foundFile.fileType = signature.name;
        foundFile.extension = signature.extension;
        foundFile.startOffset = match.offset;
        foundFile.recoveryTime = QDateTime::currentDateTime();
        
        // Update progress
        {
            QMutexLocker locker(&m_carver->m_progressMutex);
            m_carver->m_progress.filesFound++;
        }
        
        emit fileFound(foundFile);
        
        // Attempt to recover the file
        if (m_carver->recoverFile(match.offset, signature)) {
            // File recovered successfully
        } else if (m_params.recoverFragmented && signature.supportsFragmentation) {
            // Attempt fragmented recovery
            auto fragments = m_carver->m_fragmentReconstructor->findFragments(
                m_params.sourceImagePath, signature, match.offset, signature.maxFileSize);
            
            if (fragments.size() > 1) {
                QString fragmentedFileName = QString("%1_%2_fragmented.%3")
                                           .arg(signature.name)
                                           .arg(match.offset, 0, 16)
                                           .arg(signature.extension);
                
                QString fragmentedPath = QDir(m_params.outputDirectory).filePath(fragmentedFileName);
                
                if (m_carver->m_fragmentReconstructor->reconstructFile(
                    fragments, m_params.sourceImagePath, fragmentedPath, signature)) {
                    
                    // Create fragmented file record
                    CarvedFile fragmentedFile = foundFile;
                    fragmentedFile.originalPath = fragmentedPath;
                    fragmentedFile.fileName = fragmentedFileName;
                    fragmentedFile.isFragmented = true;
                    fragmentedFile.endOffset = fragments.back() + signature.maxFileSize; // Approximate
                    
                    QFileInfo info(fragmentedPath);
                    fragmentedFile.fileSize = info.size();
                    
                    if (m_params.calculateHashes) {
                        fragmentedFile.md5Hash = m_carver->calculateFileHash(fragmentedPath, "MD5");
                        fragmentedFile.sha256Hash = m_carver->calculateFileHash(fragmentedPath, "SHA256");
                    }
                    
                    m_carver->m_carvedFiles.push_back(fragmentedFile);
                    
                    {
                        QMutexLocker locker(&m_carver->m_progressMutex);
                        m_carver->m_progress.filesRecovered++;
                        m_carver->m_statistics.totalFragmentedFiles++;
                    }
                    
                    emit m_carver->fileRecovered(fragmentedFile);
                }
            }
        }
    }
}

} // namespace PhoenixDRS

#include "FileCarver.moc"