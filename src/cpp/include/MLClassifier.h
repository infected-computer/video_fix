/*
 * PhoenixDRS Professional - Advanced Machine Learning File Classification Engine
 * מנוע סיווג קבצים מתקדם עם למידת מכונה - PhoenixDRS מקצועי
 * 
 * State-of-the-art AI-powered file classification and forensic analysis
 * סיווג קבצים וניתוח פורנזי מתקדם המונע על ידי בינה מלאכותית
 */

#pragma once

#include "Common.h"
#include <QObject>
#include <QThread>
#include <QMutex>
#include <QTimer>
#include <QJsonObject>
#include <QJsonArray>
#include <QDateTime>
#include <QImage>
#include <QPixmap>

#include <memory>
#include <atomic>
#include <vector>
#include <unordered_map>
#include <array>
#include <functional>

// TensorFlow Lite C++ API for inference
#ifdef ENABLE_TENSORFLOW
#include <tensorflow/lite/interpreter.h>
#include <tensorflow/lite/kernels/register.h>
#include <tensorflow/lite/model.h>
#include <tensorflow/lite/optional_debug_tools.h>
#include <tensorflow/lite/delegates/gpu/delegate.h>
#endif

// OpenCV for image processing and feature extraction
#ifdef ENABLE_OPENCV
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/face.hpp>
#include <opencv2/dnn.hpp>
#endif

// ONNX Runtime for cross-platform ML inference
#ifdef ENABLE_ONNX
#include <onnxruntime_cxx_api.h>
#endif

namespace PhoenixDRS {

// ML model types
enum class MLModelType {
    FILE_TYPE_CLASSIFIER,       // General file type classification
    MALWARE_DETECTOR,          // Malware detection
    STEGANOGRAPHY_DETECTOR,    // Hidden data detection  
    FACE_RECOGNIZER,           // Face recognition in images/videos
    OBJECT_DETECTOR,           // Object detection in images
    TEXT_ANALYZER,             // Text content analysis
    BEHAVIORAL_ANALYZER,       // Behavioral pattern analysis
    SIMILARITY_MATCHER,        // File similarity matching
    CONTENT_MODERATOR,         // Inappropriate content detection
    DEEPFAKE_DETECTOR,         // Deepfake video/image detection
    ENCRYPTION_DETECTOR,       // Encrypted content detection
    CUSTOM_MODEL              // User-defined custom models
};

// Classification confidence levels
enum class ConfidenceLevel {
    VERY_LOW = 0,    // 0-20%
    LOW = 1,         // 20-40%
    MEDIUM = 2,      // 40-60%
    HIGH = 3,        // 60-80%
    VERY_HIGH = 4,   // 80-95%
    CERTAIN = 5      // 95-100%
};

// ML classification result
struct MLClassificationResult {
    QString fileName;                    // Original file name
    QString filePath;                   // Full file path
    qint64 fileSize;                   // File size in bytes
    QString detectedType;              // Detected file type
    QString primaryCategory;           // Primary classification category
    std::vector<QString> secondaryCategories; // Secondary categories
    double confidence;                 // Classification confidence (0.0-1.0)
    ConfidenceLevel confidenceLevel;   // Confidence level enum
    
    // Detailed analysis
    QJsonObject features;              // Extracted features
    QJsonObject metadata;              // File metadata
    QJsonObject predictions;           // All model predictions
    
    // Risk assessment
    double riskScore;                  // Risk score (0.0-1.0)
    QString riskLevel;                 // "LOW", "MEDIUM", "HIGH", "CRITICAL"
    std::vector<QString> riskFactors;  // Identified risk factors
    
    // Forensic relevance
    double forensicRelevance;          // Forensic importance (0.0-1.0)
    QString priorityLevel;             // "LOW", "NORMAL", "HIGH", "URGENT"
    std::vector<QString> forensicTags; // Forensic classification tags
    
    // Additional insights
    bool isSuspicious;                 // Flagged as suspicious
    bool isEncrypted;                  // Contains encryption
    bool hasHiddenData;                // Possible steganography
    bool requiresManualReview;         // Needs human examination
    
    QDateTime analysisTime;            // When analysis was performed
    QString modelVersion;              // ML model version used
    
    MLClassificationResult() : fileSize(0), confidence(0.0), confidenceLevel(ConfidenceLevel::VERY_LOW),
                              riskScore(0.0), riskLevel("LOW"), forensicRelevance(0.0), 
                              priorityLevel("NORMAL"), isSuspicious(false), isEncrypted(false),
                              hasHiddenData(false), requiresManualReview(false) {
        analysisTime = QDateTime::currentDateTime();
    }
};

// ML model information
struct MLModelInfo {
    QString modelId;                   // Unique model identifier
    QString modelName;                 // Human-readable name
    QString modelVersion;              // Model version
    MLModelType modelType;             // Model type
    QString modelPath;                 // Path to model file
    QString description;               // Model description
    QStringList supportedFormats;     // Supported file formats
    double accuracy;                   // Model accuracy (0.0-1.0)
    QDateTime lastUpdated;             // Last model update
    QJsonObject configuration;         // Model configuration
    bool isEnabled;                    // Whether model is active
    bool requiresGPU;                  // GPU acceleration required
    qint64 modelSizeBytes;            // Model file size
    
    MLModelInfo() : modelType(MLModelType::FILE_TYPE_CLASSIFIER), accuracy(0.0),
                   isEnabled(true), requiresGPU(false), modelSizeBytes(0) {
        lastUpdated = QDateTime::currentDateTime();
    }
};

// ML processing parameters
struct MLProcessingParameters {
    std::vector<MLModelType> enabledModels;    // Models to use
    bool useGPUAcceleration;                   // Enable GPU acceleration
    bool parallelProcessing;                   // Enable parallel processing
    int workerThreads;                         // Number of worker threads
    double confidenceThreshold;               // Minimum confidence threshold
    bool deepAnalysis;                         // Enable deep analysis
    bool extractThumbnails;                    // Extract image thumbnails
    bool analyzeMetadata;                      // Analyze file metadata
    
    // Advanced options
    int maxImageSize;                          // Maximum image size for processing
    int maxVideoLength;                        // Maximum video length (seconds)
    bool enableFaceRecognition;                // Enable face recognition
    bool enableObjectDetection;                // Enable object detection
    bool enableTextAnalysis;                   // Enable text content analysis
    bool enableBehavioralAnalysis;             // Enable behavioral analysis
    
    MLProcessingParameters() : useGPUAcceleration(true), parallelProcessing(true),
                              workerThreads(0), confidenceThreshold(0.5), deepAnalysis(true),
                              extractThumbnails(true), analyzeMetadata(true),
                              maxImageSize(4096), maxVideoLength(300),
                              enableFaceRecognition(true), enableObjectDetection(true),
                              enableTextAnalysis(true), enableBehavioralAnalysis(true) {}
};

// ML processing progress
struct MLProcessingProgress {
    qint64 filesProcessed;             // Files processed so far
    qint64 totalFiles;                 // Total files to process
    qint64 currentFileSize;            // Current file size being processed
    QString currentFileName;           // Current file being processed
    QString currentOperation;          // Current ML operation
    MLModelType currentModel;          // Currently running model
    QTime elapsedTime;                // Elapsed processing time
    QTime estimatedTimeRemaining;     // Estimated time remaining
    qint64 processingRate;            // Files per second
    
    // Analysis results
    qint64 suspiciousFilesFound;       // Number of suspicious files
    qint64 encryptedFilesFound;        // Number of encrypted files
    qint64 hiddenDataFound;            // Files with hidden data
    qint64 facesDetected;              // Total faces detected
    qint64 objectsDetected;            // Total objects detected
    
    MLProcessingProgress() : filesProcessed(0), totalFiles(0), currentFileSize(0),
                           currentModel(MLModelType::FILE_TYPE_CLASSIFIER), processingRate(0),
                           suspiciousFilesFound(0), encryptedFilesFound(0), hiddenDataFound(0),
                           facesDetected(0), objectsDetected(0) {}
};

// Forward declarations
class MLInferenceEngine;
class FeatureExtractor;
class ModelManager;
class GPUAccelerator;

/*
 * Main ML classification engine
 * מנוע סיווג ML ראשי
 */
class PHOENIXDRS_EXPORT MLClassifier : public QObject
{
    Q_OBJECT

public:
    explicit MLClassifier(QObject* parent = nullptr);
    ~MLClassifier() override;

    // Initialization and configuration
    bool initialize();
    void shutdown();
    bool isInitialized() const { return m_isInitialized; }
    
    // Model management
    bool loadModel(const QString& modelPath, MLModelType modelType);
    bool loadModelsFromDirectory(const QString& modelsDirectory);
    void unloadModel(MLModelType modelType);
    std::vector<MLModelInfo> getLoadedModels() const;
    bool downloadModel(const QString& modelId, const QString& downloadUrl);
    
    // Main processing operations
    bool startClassification(const QStringList& filePaths, const MLProcessingParameters& params);
    void pauseClassification();
    void resumeClassification();
    void cancelClassification();
    
    // Status and progress
    bool isRunning() const { return m_isRunning.load(); }
    bool isPaused() const { return m_isPaused.load(); }
    MLProcessingProgress getProgress() const;
    
    // Single file analysis
    MLClassificationResult classifyFile(const QString& filePath, 
                                       const std::vector<MLModelType>& models = {});
    MLClassificationResult analyzeImage(const QString& imagePath);
    MLClassificationResult analyzeVideo(const QString& videoPath);
    MLClassificationResult analyzeDocument(const QString& documentPath);
    MLClassificationResult analyzeExecutable(const QString& executablePath);
    
    // Batch operations
    std::vector<MLClassificationResult> classifyFiles(const QStringList& filePaths);
    std::vector<MLClassificationResult> getResults() const;
    std::vector<MLClassificationResult> getSuspiciousFiles() const;
    std::vector<MLClassificationResult> getHighRiskFiles() const;
    
    // Advanced analysis
    std::vector<MLClassificationResult> findSimilarFiles(const QString& referencePath, 
                                                        double similarityThreshold = 0.8);
    std::vector<MLClassificationResult> detectDuplicates(double threshold = 0.95);
    std::vector<MLClassificationResult> findAnomalies();
    
    // Face recognition
    struct FaceRecognitionResult {
        QString fileName;
        QRect boundingBox;
        double confidence;
        QString personId;
        QString personName;
        QJsonObject features;
        QImage faceImage;
    };
    
    std::vector<FaceRecognitionResult> detectFaces(const QString& imagePath);
    bool trainFaceRecognition(const QStringList& trainingImages, const QStringList& personNames);
    std::vector<FaceRecognitionResult> searchFaces(const QString& queryFacePath);
    
    // Object detection
    struct ObjectDetectionResult {
        QString fileName;
        QString objectClass;
        QRect boundingBox;
        double confidence;
        QJsonObject properties;
    };
    
    std::vector<ObjectDetectionResult> detectObjects(const QString& imagePath);
    std::vector<ObjectDetectionResult> detectWeapons(const QString& imagePath);
    std::vector<ObjectDetectionResult> detectVehicles(const QString& imagePath);
    
    // Text analysis
    struct TextAnalysisResult {
        QString fileName;
        QString extractedText;
        QString language;
        QStringList keywords;
        QStringList entities;
        double sentimentScore;
        QString sentiment; // "positive", "negative", "neutral"
        bool containsPII;  // Personal Identifiable Information
        QStringList piiTypes;
        bool isInappropriate;
        QJsonObject metadata;
    };
    
    TextAnalysisResult analyzeText(const QString& filePath);
    std::vector<TextAnalysisResult> searchTextContent(const QString& query);
    
    // Malware detection
    struct MalwareAnalysisResult {
        QString fileName;
        bool isMalware;
        QString malwareFamily;
        QString malwareType;
        double riskScore;
        QStringList indicators;
        QStringList behaviors;
        QJsonObject staticAnalysis;
        bool requiresSandbox;
    };
    
    MalwareAnalysisResult analyzeMalware(const QString& filePath);
    std::vector<MalwareAnalysisResult> scanForMalware(const QStringList& filePaths);
    
    // Configuration
    void setProcessingParameters(const MLProcessingParameters& params);
    MLProcessingParameters getProcessingParameters() const { return m_parameters; }
    void setConfidenceThreshold(double threshold);
    double getConfidenceThreshold() const { return m_parameters.confidenceThreshold; }
    void enableGPUAcceleration(bool enable);
    bool isGPUAccelerationEnabled() const { return m_parameters.useGPUAcceleration; }
    
    // Statistics and reporting
    struct MLStatistics {
        qint64 totalFilesProcessed;
        qint64 totalProcessingTime;     // milliseconds
        double averageProcessingTime;   // milliseconds per file
        qint64 suspiciousFilesDetected;
        qint64 malwareDetected;
        qint64 encryptedFilesDetected;
        qint64 facesDetected;
        qint64 objectsDetected;
        std::unordered_map<QString, int> fileTypeDistribution;
        std::unordered_map<QString, int> riskLevelDistribution;
        QDateTime lastProcessingSession;
        
        MLStatistics() : totalFilesProcessed(0), totalProcessingTime(0),
                        averageProcessingTime(0.0), suspiciousFilesDetected(0),
                        malwareDetected(0), encryptedFilesDetected(0),
                        facesDetected(0), objectsDetected(0) {}
    };
    
    MLStatistics getStatistics() const;
    void resetStatistics();
    
    // Export and reporting
    bool exportResults(const QString& filePath, const QString& format = "json");
    bool exportSuspiciousFiles(const QString& filePath);
    QJsonObject generateAnalysisReport() const;
    bool exportFaceDatabase(const QString& filePath);
    bool importFaceDatabase(const QString& filePath);

public slots:
    void startClassificationAsync(const QStringList& filePaths, const MLProcessingParameters& params);

signals:
    void classificationStarted(int totalFiles);
    void progressUpdated(const MLProcessingProgress& progress);
    void fileClassified(const MLClassificationResult& result);
    void suspiciousFileDetected(const MLClassificationResult& result);
    void faceDetected(const FaceRecognitionResult& result);
    void objectDetected(const ObjectDetectionResult& result);
    void malwareDetected(const MalwareAnalysisResult& result);
    void classificationCompleted(bool success, const QString& message);
    void classificationPaused();
    void classificationResumed();
    void classificationCancelled();
    void errorOccurred(const QString& error);
    void modelLoaded(const MLModelInfo& modelInfo);
    void modelLoadFailed(const QString& modelPath, const QString& error);

private slots:
    void updateProgress();
    void handleWorkerFinished();
    void handleWorkerError(const QString& error);

private:
    // Internal worker class
    class ClassificationWorker;
    friend class ClassificationWorker;
    
    // Core functionality
    bool initializeMLFrameworks();
    void cleanupMLFrameworks();
    bool loadDefaultModels();
    
    // Feature extraction
    QJsonObject extractFileFeatures(const QString& filePath);
    QJsonObject extractImageFeatures(const QImage& image);
    QJsonObject extractTextFeatures(const QString& text);
    QJsonObject extractBinaryFeatures(const QByteArray& data);
    
    // Model inference
    MLClassificationResult runInference(const QString& filePath, MLModelType modelType);
    std::vector<double> preprocessImage(const QImage& image, const QSize& inputSize);
    std::vector<double> preprocessText(const QString& text, int maxLength);
    
    // Risk assessment
    double calculateRiskScore(const MLClassificationResult& result);
    QString determineRiskLevel(double riskScore);
    std::vector<QString> identifyRiskFactors(const MLClassificationResult& result);
    
    // Similarity analysis
    double calculateFileSimilarity(const QString& file1, const QString& file2);
    QByteArray calculatePerceptualHash(const QImage& image);
    
    // GPU acceleration
    bool initializeGPUAcceleration();
    void cleanupGPUAcceleration();
    
    // Member variables
    std::atomic<bool> m_isInitialized{false};
    std::atomic<bool> m_isRunning{false};
    std::atomic<bool> m_isPaused{false};
    std::atomic<bool> m_shouldCancel{false};
    
    MLProcessingParameters m_parameters;
    MLProcessingProgress m_progress;
    MLStatistics m_statistics;
    
    // Threading
    std::unique_ptr<ClassificationWorker> m_worker;
    QThread* m_workerThread;
    QTimer* m_progressTimer;
    QMutex m_progressMutex;
    QWaitCondition m_pauseCondition;
    
    // Components
    std::unique_ptr<MLInferenceEngine> m_inferenceEngine;
    std::unique_ptr<FeatureExtractor> m_featureExtractor;
    std::unique_ptr<ModelManager> m_modelManager;
    std::unique_ptr<GPUAccelerator> m_gpuAccelerator;
    
    // Data structures
    std::vector<MLClassificationResult> m_results;
    std::unordered_map<MLModelType, MLModelInfo> m_loadedModels;
    std::unordered_map<QString, QByteArray> m_faceDatabase;
    
    // Performance monitoring
    QElapsedTimer m_operationTimer;
    std::array<qint64, 10> m_recentRates{};
    size_t m_rateIndex{0};
    
    // ML framework contexts
#ifdef ENABLE_TENSORFLOW
    std::unique_ptr<tflite::FlatBufferModel> m_tfModel;
    std::unique_ptr<tflite::Interpreter> m_tfInterpreter;
    std::unique_ptr<tflite::ops::builtin::BuiltinOpResolver> m_tfResolver;
#endif

#ifdef ENABLE_OPENCV
    cv::dnn::Net m_cvModel;
    cv::CascadeClassifier m_faceClassifier;
    cv::Ptr<cv::face::FaceRecognizer> m_faceRecognizer;
#endif

#ifdef ENABLE_ONNX
    std::unique_ptr<Ort::Env> m_onnxEnv;
    std::unique_ptr<Ort::Session> m_onnxSession;
    std::unique_ptr<Ort::SessionOptions> m_onnxSessionOptions;
#endif

    // Constants
    static constexpr int PROGRESS_UPDATE_INTERVAL = 200; // milliseconds
    static constexpr int MAX_RESULTS_HISTORY = 100000;
    static constexpr double DEFAULT_CONFIDENCE_THRESHOLD = 0.5;
    static constexpr int DEFAULT_IMAGE_SIZE = 224; // Standard CNN input size
};

} // namespace PhoenixDRS