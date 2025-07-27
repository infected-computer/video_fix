/*
 * PhoenixDRS Professional - Advanced Machine Learning File Classification Engine Implementation
 * מימוש מנוע סיווג קבצים מתקדם עם למידת מכונה - PhoenixDRS מקצועי
 * 
 * High-performance ML inference engine for forensic file analysis
 * מנוע הסקה ML בביצועים גבוהים לניתוח קבצים פורנזי
 */

#include "include/MLClassifier.h"
#include "include/ForensicLogger.h"
#include <QApplication>
#include <QDir>
#include <QFileInfo>
#include <QMimeDatabase>
#include <QMimeType>
#include <QStandardPaths>
#include <QJsonDocument>
#include <QDebug>
#include <QThread>
#include <QImageReader>
#include <QBuffer>

#include <algorithm>
#include <cmath>
#include <random>

namespace PhoenixDRS {

// Internal worker class for threaded ML processing
class MLClassifier::ClassificationWorker : public QObject
{
    Q_OBJECT

public:
    ClassificationWorker(MLClassifier* parent) : m_classifier(parent) {}

    void processFiles(const QStringList& filePaths, const MLProcessingParameters& params) {
        m_filePaths = filePaths;
        m_parameters = params;
        m_shouldCancel = false;
        
        emit started();
        
        for (int i = 0; i < m_filePaths.size() && !m_shouldCancel; ++i) {
            const QString& filePath = m_filePaths[i];
            
            try {
                MLClassificationResult result = m_classifier->classifyFile(filePath, m_parameters.enabledModels);
                emit fileProcessed(result);
                
                if (result.isSuspicious) {
                    emit suspiciousFileDetected(result);
                }
                
            } catch (const std::exception& e) {
                emit errorOccurred(QString("Error processing %1: %2").arg(filePath, e.what()));
            }
            
            // Update progress
            int progress = static_cast<int>((i + 1) * 100 / m_filePaths.size());
            emit progressUpdated(progress);
        }
        
        emit finished(!m_shouldCancel);
    }

    void cancel() { m_shouldCancel = true; }

signals:
    void started();
    void fileProcessed(const MLClassificationResult& result);
    void suspiciousFileDetected(const MLClassificationResult& result);
    void progressUpdated(int progress);
    void errorOccurred(const QString& error);
    void finished(bool success);

private:
    MLClassifier* m_classifier;
    QStringList m_filePaths;
    MLProcessingParameters m_parameters;
    std::atomic<bool> m_shouldCancel{false};
};

MLClassifier::MLClassifier(QObject* parent)
    : QObject(parent)
    , m_workerThread(nullptr)
    , m_progressTimer(new QTimer(this))
{
    m_progressTimer->setInterval(PROGRESS_UPDATE_INTERVAL);
    connect(m_progressTimer, &QTimer::timeout, this, &MLClassifier::updateProgress);
    
    // Initialize default processing parameters
    m_parameters.enabledModels = {
        MLModelType::FILE_TYPE_CLASSIFIER,
        MLModelType::MALWARE_DETECTOR,
        MLModelType::CONTENT_MODERATOR
    };
}

MLClassifier::~MLClassifier()
{
    shutdown();
}

bool MLClassifier::initialize()
{
    if (m_isInitialized) {
        return true;
    }

    ForensicLogger::instance()->logInfo("Initializing ML Classification Engine...");

    try {
        // Initialize ML frameworks
        if (!initializeMLFrameworks()) {
            ForensicLogger::instance()->logError("Failed to initialize ML frameworks");
            return false;
        }

        // Load default models
        if (!loadDefaultModels()) {
            ForensicLogger::instance()->logWarning("Some default models failed to load");
        }

        // Initialize GPU acceleration if enabled
        if (m_parameters.useGPUAcceleration) {
            if (!initializeGPUAcceleration()) {
                ForensicLogger::instance()->logWarning("GPU acceleration initialization failed, falling back to CPU");
                m_parameters.useGPUAcceleration = false;
            }
        }

        m_isInitialized = true;
        ForensicLogger::instance()->logInfo("ML Classification Engine initialized successfully");
        
        return true;

    } catch (const std::exception& e) {
        ForensicLogger::instance()->logError(QString("ML initialization failed: %1").arg(e.what()));
        return false;
    }
}

void MLClassifier::shutdown()
{
    if (!m_isInitialized) {
        return;
    }

    ForensicLogger::instance()->logInfo("Shutting down ML Classification Engine...");

    // Cancel any running operations
    if (m_isRunning) {
        cancelClassification();
    }

    // Cleanup worker thread
    if (m_workerThread) {
        m_workerThread->quit();
        m_workerThread->wait(5000);
        delete m_workerThread;
        m_workerThread = nullptr;
    }

    // Cleanup ML frameworks
    cleanupMLFrameworks();
    cleanupGPUAcceleration();

    m_loadedModels.clear();
    m_results.clear();
    m_faceDatabase.clear();

    m_isInitialized = false;
    ForensicLogger::instance()->logInfo("ML Classification Engine shutdown complete");
}

bool MLClassifier::loadModel(const QString& modelPath, MLModelType modelType)
{
    if (!m_isInitialized) {
        ForensicLogger::instance()->logError("ML Classifier not initialized");
        return false;
    }

    QFileInfo modelFile(modelPath);
    if (!modelFile.exists() || !modelFile.isReadable()) {
        ForensicLogger::instance()->logError(QString("Model file not accessible: %1").arg(modelPath));
        return false;
    }

    try {
        MLModelInfo modelInfo;
        modelInfo.modelId = QUuid::createUuid().toString();
        modelInfo.modelName = modelFile.baseName();
        modelInfo.modelType = modelType;
        modelInfo.modelPath = modelPath;
        modelInfo.modelSizeBytes = modelFile.size();
        modelInfo.lastUpdated = modelFile.lastModified();

        // Attempt to load the model based on file extension
        QString extension = modelFile.suffix().toLower();
        bool loaded = false;

#ifdef ENABLE_TENSORFLOW
        if (extension == "tflite") {
            loaded = loadTensorFlowLiteModel(modelPath, modelInfo);
        }
#endif

#ifdef ENABLE_ONNX
        if (extension == "onnx") {
            loaded = loadONNXModel(modelPath, modelInfo);
        }
#endif

        if (loaded) {
            m_loadedModels[modelType] = modelInfo;
            emit modelLoaded(modelInfo);
            ForensicLogger::instance()->logInfo(QString("Successfully loaded model: %1").arg(modelInfo.modelName));
            return true;
        } else {
            emit modelLoadFailed(modelPath, "Unsupported model format or loading error");
            return false;
        }

    } catch (const std::exception& e) {
        QString error = QString("Failed to load model %1: %2").arg(modelPath, e.what());
        ForensicLogger::instance()->logError(error);
        emit modelLoadFailed(modelPath, error);
        return false;
    }
}

bool MLClassifier::loadModelsFromDirectory(const QString& modelsDirectory)
{
    QDir dir(modelsDirectory);
    if (!dir.exists()) {
        ForensicLogger::instance()->logError(QString("Models directory does not exist: %1").arg(modelsDirectory));
        return false;
    }

    QStringList filters;
    filters << "*.tflite" << "*.onnx" << "*.pb" << "*.h5";
    
    QFileInfoList modelFiles = dir.entryInfoList(filters, QDir::Files | QDir::Readable);
    
    int loadedCount = 0;
    for (const QFileInfo& fileInfo : modelFiles) {
        // Determine model type from filename or metadata
        MLModelType modelType = determineModelTypeFromFile(fileInfo.fileName());
        
        if (loadModel(fileInfo.absoluteFilePath(), modelType)) {
            loadedCount++;
        }
    }

    ForensicLogger::instance()->logInfo(QString("Loaded %1 out of %2 models from directory")
                                       .arg(loadedCount).arg(modelFiles.size()));
    
    return loadedCount > 0;
}

MLClassificationResult MLClassifier::classifyFile(const QString& filePath, const std::vector<MLModelType>& models)
{
    MLClassificationResult result;
    result.fileName = QFileInfo(filePath).fileName();
    result.filePath = filePath;
    result.fileSize = QFileInfo(filePath).size();

    if (!m_isInitialized) {
        result.primaryCategory = "ERROR";
        result.metadata["error"] = "ML Classifier not initialized";
        return result;
    }

    try {
        // Basic file analysis
        QMimeDatabase mimeDb;
        QMimeType mimeType = mimeDb.mimeTypeForFile(filePath);
        result.detectedType = mimeType.name();

        // Extract basic features
        result.features = extractFileFeatures(filePath);
        
        // Determine which models to use
        std::vector<MLModelType> modelsToUse = models.empty() ? m_parameters.enabledModels : models;

        // Run inference with each enabled model
        for (MLModelType modelType : modelsToUse) {
            if (m_loadedModels.find(modelType) != m_loadedModels.end()) {
                MLClassificationResult modelResult = runInference(filePath, modelType);
                
                // Merge results
                if (modelResult.confidence > result.confidence) {
                    result.confidence = modelResult.confidence;
                    result.primaryCategory = modelResult.primaryCategory;
                }
                
                // Accumulate secondary categories
                for (const QString& category : modelResult.secondaryCategories) {
                    if (std::find(result.secondaryCategories.begin(), 
                                 result.secondaryCategories.end(), category) == result.secondaryCategories.end()) {
                        result.secondaryCategories.push_back(category);
                    }
                }
                
                // Merge predictions
                QJsonObject modelPreds = modelResult.predictions;
                for (auto it = modelPreds.begin(); it != modelPreds.end(); ++it) {
                    result.predictions[it.key()] = it.value();
                }
            }
        }

        // Calculate risk assessment
        result.riskScore = calculateRiskScore(result);
        result.riskLevel = determineRiskLevel(result.riskScore);
        result.riskFactors = identifyRiskFactors(result);

        // Determine confidence level
        if (result.confidence >= 0.95) result.confidenceLevel = ConfidenceLevel::CERTAIN;
        else if (result.confidence >= 0.80) result.confidenceLevel = ConfidenceLevel::VERY_HIGH;
        else if (result.confidence >= 0.60) result.confidenceLevel = ConfidenceLevel::HIGH;
        else if (result.confidence >= 0.40) result.confidenceLevel = ConfidenceLevel::MEDIUM;
        else if (result.confidence >= 0.20) result.confidenceLevel = ConfidenceLevel::LOW;
        else result.confidenceLevel = ConfidenceLevel::VERY_LOW;

        // Forensic relevance assessment
        result.forensicRelevance = calculateForensicRelevance(result);
        result.priorityLevel = determinePriorityLevel(result.forensicRelevance, result.riskScore);

        // Flag suspicious files
        result.isSuspicious = (result.riskScore > 0.7) || 
                             (result.confidenceLevel <= ConfidenceLevel::LOW && result.riskScore > 0.3);
        
        // Check for encryption
        result.isEncrypted = detectEncryption(filePath);
        
        // Check for hidden data (steganography)
        result.hasHiddenData = detectSteganography(filePath);
        
        // Determine if manual review is needed
        result.requiresManualReview = result.isSuspicious || 
                                     result.confidenceLevel <= ConfidenceLevel::LOW ||
                                     result.hasHiddenData;

        // Update statistics
        updateStatistics(result);

    } catch (const std::exception& e) {
        result.primaryCategory = "ERROR";
        result.metadata["error"] = e.what();
        ForensicLogger::instance()->logError(QString("Classification error for %1: %2").arg(filePath, e.what()));
    }

    return result;
}

bool MLClassifier::startClassification(const QStringList& filePaths, const MLProcessingParameters& params)
{
    if (!m_isInitialized) {
        ForensicLogger::instance()->logError("ML Classifier not initialized");
        return false;
    }

    if (m_isRunning) {
        ForensicLogger::instance()->logWarning("Classification is already running");
        return false;
    }

    m_parameters = params;
    m_progress.totalFiles = filePaths.size();
    m_progress.filesProcessed = 0;
    m_shouldCancel = false;
    m_isRunning = true;

    // Create worker thread
    if (m_workerThread) {
        m_workerThread->quit();
        m_workerThread->wait();
        delete m_workerThread;
    }

    m_workerThread = new QThread(this);
    m_worker = std::make_unique<ClassificationWorker>(this);
    m_worker->moveToThread(m_workerThread);

    // Connect signals
    connect(m_workerThread, &QThread::started, 
            [this, filePaths, params]() { m_worker->processFiles(filePaths, params); });
    connect(m_worker.get(), &ClassificationWorker::fileProcessed, 
            this, &MLClassifier::fileClassified);
    connect(m_worker.get(), &ClassificationWorker::suspiciousFileDetected, 
            this, &MLClassifier::suspiciousFileDetected);
    connect(m_worker.get(), &ClassificationWorker::finished, 
            this, &MLClassifier::handleWorkerFinished);
    connect(m_worker.get(), &ClassificationWorker::errorOccurred, 
            this, &MLClassifier::handleWorkerError);

    // Start processing
    m_operationTimer.start();
    m_progressTimer->start();
    m_workerThread->start();

    emit classificationStarted(filePaths.size());
    ForensicLogger::instance()->logInfo(QString("Started classification of %1 files").arg(filePaths.size()));

    return true;
}

void MLClassifier::cancelClassification()
{
    if (!m_isRunning) {
        return;
    }

    m_shouldCancel = true;
    
    if (m_worker) {
        m_worker->cancel();
    }

    ForensicLogger::instance()->logInfo("Classification cancellation requested");
}

MLClassificationResult MLClassifier::runInference(const QString& filePath, MLModelType modelType)
{
    MLClassificationResult result;
    result.fileName = QFileInfo(filePath).fileName();
    result.filePath = filePath;

    try {
        // Load file for analysis
        QFile file(filePath);
        if (!file.open(QIODevice::ReadOnly)) {
            throw std::runtime_error("Cannot open file for analysis");
        }

        // Different inference paths based on model type
        switch (modelType) {
            case MLModelType::FILE_TYPE_CLASSIFIER:
                result = runFileTypeInference(filePath);
                break;
                
            case MLModelType::MALWARE_DETECTOR:
                result = runMalwareInference(filePath);
                break;
                
            case MLModelType::STEGANOGRAPHY_DETECTOR:
                result = runSteganographyInference(filePath);
                break;
                
            case MLModelType::FACE_RECOGNIZER:
                result = runFaceRecognitionInference(filePath);
                break;
                
            case MLModelType::OBJECT_DETECTOR:
                result = runObjectDetectionInference(filePath);
                break;
                
            case MLModelType::TEXT_ANALYZER:
                result = runTextAnalysisInference(filePath);
                break;
                
            case MLModelType::CONTENT_MODERATOR:
                result = runContentModerationInference(filePath);
                break;
                
            case MLModelType::DEEPFAKE_DETECTOR:
                result = runDeepfakeInference(filePath);
                break;
                
            default:
                result.primaryCategory = "UNKNOWN";
                result.confidence = 0.0;
                break;
        }

        result.modelVersion = m_loadedModels[modelType].modelVersion;

    } catch (const std::exception& e) {
        result.primaryCategory = "ERROR";
        result.confidence = 0.0;
        result.metadata["inference_error"] = e.what();
    }

    return result;
}

QJsonObject MLClassifier::extractFileFeatures(const QString& filePath)
{
    QJsonObject features;

    try {
        QFileInfo fileInfo(filePath);
        features["file_size"] = fileInfo.size();
        features["file_extension"] = fileInfo.suffix().toLower();
        features["creation_time"] = fileInfo.birthTime().toString(Qt::ISODate);
        features["modification_time"] = fileInfo.lastModified().toString(Qt::ISODate);

        // MIME type detection
        QMimeDatabase mimeDb;
        QMimeType mimeType = mimeDb.mimeTypeForFile(filePath);
        features["mime_type"] = mimeType.name();
        features["mime_comment"] = mimeType.comment();

        // File header analysis (first 512 bytes)
        QFile file(filePath);
        if (file.open(QIODevice::ReadOnly)) {
            QByteArray header = file.read(512);
            features["header_size"] = header.size();
            
            // Calculate header entropy
            features["header_entropy"] = calculateEntropy(header);
            
            // Detect common file signatures
            features["file_signature"] = detectFileSignature(header);
            
            file.close();
        }

        // For image files, extract additional features
        if (mimeType.name().startsWith("image/")) {
            QJsonObject imageFeatures = extractImageFeatures(filePath);
            features["image_features"] = imageFeatures;
        }

        // For text files, extract text features
        if (mimeType.name().startsWith("text/")) {
            QString textContent = readTextFile(filePath, 10000); // First 10KB
            QJsonObject textFeatures = extractTextFeatures(textContent);
            features["text_features"] = textFeatures;
        }

    } catch (const std::exception& e) {
        features["extraction_error"] = e.what();
    }

    return features;
}

double MLClassifier::calculateRiskScore(const MLClassificationResult& result)
{
    double riskScore = 0.0;

    // Base risk from confidence (low confidence = higher risk)
    riskScore += (1.0 - result.confidence) * 0.3;

    // Risk from file type
    QString primaryCat = result.primaryCategory.toLower();
    if (primaryCat.contains("executable") || primaryCat.contains("script")) {
        riskScore += 0.4;
    } else if (primaryCat.contains("archive") || primaryCat.contains("compressed")) {
        riskScore += 0.2;
    }

    // Risk from file size anomalies
    if (result.fileSize == 0) {
        riskScore += 0.3; // Empty files are suspicious
    } else if (result.fileSize > 1024 * 1024 * 1024) { // > 1GB
        riskScore += 0.1; // Very large files
    }

    // Risk from metadata
    if (result.metadata.contains("header_entropy")) {
        double entropy = result.metadata["header_entropy"].toDouble();
        if (entropy > 7.5) { // High entropy suggests encryption/compression
            riskScore += 0.2;
        }
    }

    // Risk from detection flags
    if (result.isEncrypted) riskScore += 0.3;
    if (result.hasHiddenData) riskScore += 0.4;

    return qMin(1.0, riskScore);
}

bool MLClassifier::initializeMLFrameworks()
{
    bool success = true;

    try {
#ifdef ENABLE_TENSORFLOW
        // Initialize TensorFlow Lite
        m_tfResolver = std::make_unique<tflite::ops::builtin::BuiltinOpResolver>();
        ForensicLogger::instance()->logInfo("TensorFlow Lite initialized");
#endif

#ifdef ENABLE_OPENCV
        // Initialize OpenCV DNN module
        // Load default face cascade classifier
        QString faceCascadePath = QStandardPaths::locate(QStandardPaths::DataLocation, 
                                                        "haarcascade_frontalface_alt.xml");
        if (!faceCascadePath.isEmpty()) {
            m_faceClassifier.load(faceCascadePath.toStdString());
        }
        ForensicLogger::instance()->logInfo("OpenCV initialized");
#endif

#ifdef ENABLE_ONNX
        // Initialize ONNX Runtime
        m_onnxEnv = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "PhoenixDRS");
        m_onnxSessionOptions = std::make_unique<Ort::SessionOptions>();
        m_onnxSessionOptions->SetIntraOpNumThreads(QThread::idealThreadCount());
        ForensicLogger::instance()->logInfo("ONNX Runtime initialized");
#endif

    } catch (const std::exception& e) {
        ForensicLogger::instance()->logError(QString("ML framework initialization error: %1").arg(e.what()));
        success = false;
    }

    return success;
}

void MLClassifier::cleanupMLFrameworks()
{
#ifdef ENABLE_TENSORFLOW
    m_tfInterpreter.reset();
    m_tfModel.reset();
    m_tfResolver.reset();
#endif

#ifdef ENABLE_ONNX
    m_onnxSession.reset();
    m_onnxSessionOptions.reset();
    m_onnxEnv.reset();
#endif
}

bool MLClassifier::loadDefaultModels()
{
    // Try to load models from standard locations
    QStringList modelPaths = {
        QStandardPaths::writableLocation(QStandardPaths::DataLocation) + "/models",
        QApplication::applicationDirPath() + "/models",
        ":/models" // Resource path
    };

    bool anyLoaded = false;
    for (const QString& path : modelPaths) {
        if (QDir(path).exists()) {
            if (loadModelsFromDirectory(path)) {
                anyLoaded = true;
            }
        }
    }

    return anyLoaded;
}

void MLClassifier::updateProgress()
{
    if (!m_isRunning) {
        m_progressTimer->stop();
        return;
    }

    // Calculate processing rate
    qint64 elapsed = m_operationTimer.elapsed();
    if (elapsed > 0 && m_progress.filesProcessed > 0) {
        m_progress.processingRate = static_cast<qint64>((m_progress.filesProcessed * 1000.0) / elapsed);
        
        // Estimate time remaining
        if (m_progress.processingRate > 0) {
            qint64 remaining = (m_progress.totalFiles - m_progress.filesProcessed) / m_progress.processingRate;
            m_progress.estimatedTimeRemaining = QTime::fromMSecsSinceStartOfDay(remaining * 1000);
        }
    }

    emit progressUpdated(m_progress);
}

void MLClassifier::handleWorkerFinished()
{
    m_isRunning = false;
    m_progressTimer->stop();

    emit classificationCompleted(true, QString("Processed %1 files").arg(m_progress.filesProcessed));
    ForensicLogger::instance()->logInfo(QString("Classification completed. Processed %1 files in %2ms")
                                       .arg(m_progress.filesProcessed)
                                       .arg(m_operationTimer.elapsed()));
}

void MLClassifier::handleWorkerError(const QString& error)
{
    ForensicLogger::instance()->logError(QString("Worker error: %1").arg(error));
    emit errorOccurred(error);
}

MLModelType MLClassifier::determineModelTypeFromFile(const QString& fileName)
{
    QString name = fileName.toLower();
    
    if (name.contains("malware") || name.contains("virus")) {
        return MLModelType::MALWARE_DETECTOR;
    } else if (name.contains("face") || name.contains("facial")) {
        return MLModelType::FACE_RECOGNIZER;
    } else if (name.contains("object") || name.contains("yolo")) {
        return MLModelType::OBJECT_DETECTOR;
    } else if (name.contains("text") || name.contains("nlp")) {
        return MLModelType::TEXT_ANALYZER;
    } else if (name.contains("content") || name.contains("moderation")) {
        return MLModelType::CONTENT_MODERATOR;
    } else if (name.contains("deepfake") || name.contains("fake")) {
        return MLModelType::DEEPFAKE_DETECTOR;
    } else if (name.contains("stego") || name.contains("hidden")) {
        return MLModelType::STEGANOGRAPHY_DETECTOR;
    } else {
        return MLModelType::FILE_TYPE_CLASSIFIER;
    }
}

// Placeholder implementations for specific inference methods
MLClassificationResult MLClassifier::runFileTypeInference(const QString& filePath)
{
    MLClassificationResult result;
    result.fileName = QFileInfo(filePath).fileName();
    result.filePath = filePath;
    
    // Basic file type classification based on extension and content
    QMimeDatabase mimeDb;
    QMimeType mimeType = mimeDb.mimeTypeForFile(filePath);
    
    result.primaryCategory = mimeType.name();
    result.confidence = 0.8; // Reasonable confidence for MIME detection
    result.detectedType = mimeType.name();
    
    return result;
}

MLClassificationResult MLClassifier::runMalwareInference(const QString& filePath)
{
    MLClassificationResult result;
    result.fileName = QFileInfo(filePath).fileName();
    result.filePath = filePath;
    
    // Placeholder malware detection logic
    result.primaryCategory = "CLEAN";
    result.confidence = 0.9;
    result.riskScore = 0.1;
    
    return result;
}

double MLClassifier::calculateEntropy(const QByteArray& data)
{
    if (data.isEmpty()) return 0.0;
    
    // Calculate Shannon entropy
    std::array<int, 256> freq = {};
    for (unsigned char byte : data) {
        freq[byte]++;
    }
    
    double entropy = 0.0;
    double size = data.size();
    
    for (int count : freq) {
        if (count > 0) {
            double p = count / size;
            entropy -= p * std::log2(p);
        }
    }
    
    return entropy;
}

} // namespace PhoenixDRS

#include "MLClassifier.moc"