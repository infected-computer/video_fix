/*
 * PhoenixDRS Professional - Advanced Distributed Computing Engine Implementation
 * מימוש מנוע מחשוב מבוזר מתקדם - PhoenixDRS מקצועי
 */

#include "../include/DistributedComputingEngine.h"
#include "../include/Core/ErrorHandling.h"
#include "../include/Core/MemoryManager.h"
#include "../include/ForensicLogger.h"

#include <QtCore/QCoreApplication>
#include <QtCore/QStandardPaths>
#include <QtCore/QDir>
#include <QtCore/QFileInfo>
#include <QtCore/QMutexLocker>
#include <QtCore/QTimer>
#include <QtCore/QJsonDocument>
#include <QtCore/QJsonObject>
#include <QtCore/QJsonArray>
#include <QtCore/QCryptographicHash>
#include <QtCore/QRegularExpression>
#include <QtCore/QUuid>
#include <QtNetwork/QTcpServer>
#include <QtNetwork/QTcpSocket>
#include <QtNetwork/QUdpSocket>
#include <QtNetwork/QHostAddress>
#include <QtNetwork/QNetworkInterface>

#include <algorithm>
#include <chrono>
#include <iomanip>
#include <sstream>

namespace PhoenixDRS {
namespace Forensics {

// Distributed Job Worker Class
class DistributedComputingEngine::JobWorker : public QObject
{
    Q_OBJECT

public:
    explicit JobWorker(DistributedComputingEngine* parent, const DistributedJob& job)
        : QObject(nullptr), m_engine(parent), m_job(job) {}

public slots:
    void executeJob();

signals:
    void finished();
    void error(const QString& message);
    void progressUpdate(int percentage);
    void resultReady(const JobResult& result);

private:
    DistributedComputingEngine* m_engine;
    DistributedJob m_job;
};

// Cluster Node Manager Class
class DistributedComputingEngine::ClusterNodeManager : public QObject
{
    Q_OBJECT

public:
    explicit ClusterNodeManager(DistributedComputingEngine* parent)
        : QObject(nullptr), m_engine(parent) {}

    void startListening(quint16 port);
    void connectToNode(const QString& address, quint16 port);
    void broadcastJob(const DistributedJob& job);
    void collectResults();

public slots:
    void handleNewConnection();
    void handleNodeMessage();
    void handleNodeDisconnection();

signals:
    void nodeConnected(const ClusterNode& node);
    void nodeDisconnected(const QString& nodeId);
    void messageReceived(const ClusterMessage& message);

private:
    DistributedComputingEngine* m_engine;
    QTcpServer* m_server = nullptr;
    QList<QTcpSocket*> m_connectedNodes;
    QMap<QString, ClusterNode> m_nodes;
    QMutex m_nodesMutex;
};

// DistributedComputingEngine Implementation
DistributedComputingEngine& DistributedComputingEngine::instance()
{
    static DistributedComputingEngine instance;
    return instance;
}

DistributedComputingEngine::DistributedComputingEngine(QObject* parent)
    : QObject(parent)
    , m_isRunning(false)
    , m_isClusterLeader(false)
    , m_currentProgress(0)
    , m_totalJobs(0)
    , m_completedJobs(0)
    , m_activeNodes(0)
    , m_nodeManager(new ClusterNodeManager(this))
{
    setupJobTypes();
    setupLoadBalancing();
    initializeClusterConfiguration();
    
    // Initialize performance monitoring
    m_performanceTimer = std::make_unique<QTimer>();
    connect(m_performanceTimer.get(), &QTimer::timeout, this, &DistributedComputingEngine::updatePerformanceMetrics);
    m_performanceTimer->start(1000); // Update every second
    
    // Initialize heartbeat system
    m_heartbeatTimer = std::make_unique<QTimer>();
    connect(m_heartbeatTimer.get(), &QTimer::timeout, this, &DistributedComputingEngine::sendHeartbeat);
    m_heartbeatTimer->start(5000); // Heartbeat every 5 seconds
    
    ForensicLogger::instance()->logInfo("DistributedComputingEngine initialized");
}

DistributedComputingEngine::~DistributedComputingEngine()
{
    if (m_isRunning.load()) {
        shutdown();
    }
    cleanup();
}

bool DistributedComputingEngine::createCluster(const ClusterConfiguration& config)
{
    try {
        if (m_isRunning.load()) {
            emit error("Cluster already running");
            return false;
        }
        
        m_clusterConfig = config;
        m_nodeId = QUuid::createUuid().toString();
        m_isClusterLeader = true;
        
        // Start listening for node connections
        m_nodeManager->startListening(config.listenPort);
        
        // Initialize job queue and scheduler
        initializeJobScheduler();
        
        m_isRunning = true;
        emit clusterCreated(m_nodeId);
        
        ForensicLogger::instance()->logInfo(QString("Cluster created with ID: %1").arg(m_nodeId));
        return true;
        
    } catch (const PhoenixDRS::Core::PhoenixException& e) {
        emit error(e.message());
        return false;
    }
}

bool DistributedComputingEngine::joinCluster(const QString& leaderAddress, quint16 leaderPort, const QString& authToken)
{
    try {
        if (m_isRunning.load()) {
            emit error("Already part of a cluster");
            return false;
        }
        
        m_nodeId = QUuid::createUuid().toString();
        m_isClusterLeader = false;
        
        // Connect to cluster leader
        m_nodeManager->connectToNode(leaderAddress, leaderPort);
        
        // Send join request
        ClusterMessage joinMessage;
        joinMessage.type = MessageType::JoinRequest;
        joinMessage.senderId = m_nodeId;
        joinMessage.data["auth_token"] = authToken;
        joinMessage.data["node_capabilities"] = getNodeCapabilities();
        
        sendMessage(joinMessage);
        
        m_isRunning = true;
        emit clusterJoined(m_nodeId);
        
        ForensicLogger::instance()->logInfo(QString("Joined cluster as node: %1").arg(m_nodeId));
        return true;
        
    } catch (const PhoenixDRS::Core::PhoenixException& e) {
        emit error(e.message());
        return false;
    }
}

QString DistributedComputingEngine::submitJob(const DistributedJob& job)
{
    try {
        if (!m_isRunning.load()) {
            throw PhoenixDRS::Core::PhoenixException(
                PhoenixDRS::Core::ErrorCode::InitializationError,
                "Distributed computing engine not running",
                "DistributedComputingEngine::submitJob"
            );
        }
        
        // Generate unique job ID
        QString jobId = QUuid::createUuid().toString();
        
        // Create job with metadata
        DistributedJob jobWithId = job;
        jobWithId.jobId = jobId;
        jobWithId.submissionTime = QDateTime::currentDateTime();
        jobWithId.status = JobStatus::Queued;
        jobWithId.submitterId = m_nodeId;
        
        // Add to job queue
        QMutexLocker locker(&m_jobQueueMutex);
        m_jobQueue.enqueue(jobWithId);
        m_jobs[jobId] = jobWithId;
        m_totalJobs++;
        
        emit jobSubmitted(jobId);
        
        // Schedule job execution if we're the leader
        if (m_isClusterLeader) {
            scheduleNextJob();
        }
        
        ForensicLogger::instance()->logInfo(QString("Job submitted: %1").arg(jobId));
        return jobId;
        
    } catch (const PhoenixDRS::Core::PhoenixException& e) {
        ForensicLogger::instance()->logError(e.message());
        return QString();
    }
}

JobResult DistributedComputingEngine::getJobResult(const QString& jobId)
{
    QMutexLocker locker(&m_jobQueueMutex);
    
    if (m_jobResults.contains(jobId)) {
        return m_jobResults[jobId];
    }
    
    // Return empty result if not found
    JobResult result;
    result.jobId = jobId;
    result.status = JobStatus::NotFound;
    return result;
}

QList<ClusterNode> DistributedComputingEngine::getClusterNodes() const
{
    QMutexLocker locker(&m_nodesMutex);
    return m_clusterNodes.values();
}

ClusterStatistics DistributedComputingEngine::getClusterStatistics() const
{
    ClusterStatistics stats;
    
    QMutexLocker jobLocker(&m_jobQueueMutex);
    QMutexLocker nodeLocker(&m_nodesMutex);
    
    stats.totalNodes = m_clusterNodes.size();
    stats.activeNodes = m_activeNodes.load();
    stats.totalJobs = m_totalJobs.load();
    stats.completedJobs = m_completedJobs.load();
    stats.queuedJobs = m_jobQueue.size();
    stats.runningJobs = countRunningJobs();
    
    // Calculate performance metrics
    stats.averageJobDuration = calculateAverageJobDuration();
    stats.clusterUtilization = calculateClusterUtilization();
    stats.throughput = calculateThroughput();
    
    return stats;
}

bool DistributedComputingEngine::cancelJob(const QString& jobId)
{
    try {
        QMutexLocker locker(&m_jobQueueMutex);
        
        if (!m_jobs.contains(jobId)) {
            return false;
        }
        
        DistributedJob& job = m_jobs[jobId];
        
        if (job.status == JobStatus::Running) {
            // Send cancellation message to executing node
            ClusterMessage cancelMessage;
            cancelMessage.type = MessageType::JobCancellation;
            cancelMessage.data["job_id"] = jobId;
            
            sendMessageToNode(job.executingNodeId, cancelMessage);
        }
        
        job.status = JobStatus::Cancelled;
        emit jobCancelled(jobId);
        
        ForensicLogger::instance()->logInfo(QString("Job cancelled: %1").arg(jobId));
        return true;
        
    } catch (const PhoenixDRS::Core::PhoenixException& e) {
        ForensicLogger::instance()->logError(e.message());
        return false;
    }
}

void DistributedComputingEngine::shutdown()
{
    if (!m_isRunning.load()) {
        return;
    }
    
    try {
        // Cancel all running jobs
        QMutexLocker locker(&m_jobQueueMutex);
        for (auto& job : m_jobs) {
            if (job.status == JobStatus::Running) {
                job.status = JobStatus::Cancelled;
            }
        }
        
        // Notify cluster nodes about shutdown
        if (m_isClusterLeader) {
            ClusterMessage shutdownMessage;
            shutdownMessage.type = MessageType::ClusterShutdown;
            shutdownMessage.senderId = m_nodeId;
            
            broadcastMessage(shutdownMessage);
        }
        
        m_isRunning = false;
        
        // Stop timers
        if (m_performanceTimer) {
            m_performanceTimer->stop();
        }
        if (m_heartbeatTimer) {
            m_heartbeatTimer->stop();
        }
        
        emit clusterShutdown();
        ForensicLogger::instance()->logInfo("Distributed computing engine shutdown");
        
    } catch (const PhoenixDRS::Core::PhoenixException& e) {
        ForensicLogger::instance()->logError(e.message());
    }
}

// Private Implementation Methods

void DistributedComputingEngine::setupJobTypes()
{
    // Initialize supported job types
    m_supportedJobTypes = {
        {JobType::FileCarving, "Advanced file carving with signature detection"},
        {JobType::DiskImaging, "High-performance disk imaging operations"},
        {JobType::VideoRebuilding, "Video file reconstruction and repair"},
        {JobType::MemoryAnalysis, "Memory dump analysis and forensics"},
        {JobType::NetworkAnalysis, "Network traffic and packet analysis"},
        {JobType::CryptographicAnalysis, "Cryptographic algorithm detection and analysis"},
        {JobType::BlockchainAnalysis, "Cryptocurrency and blockchain forensics"},
        {JobType::CustomScript, "User-defined custom analysis scripts"}
    };
}

void DistributedComputingEngine::setupLoadBalancing()
{
    // Initialize load balancing strategies
    m_loadBalancingStrategy = LoadBalancingStrategy::RoundRobin;
    m_currentRoundRobinIndex = 0;
}

void DistributedComputingEngine::initializeClusterConfiguration()
{
    // Default cluster configuration
    m_clusterConfig.maxNodes = 50;
    m_clusterConfig.heartbeatInterval = 5000; // 5 seconds
    m_clusterConfig.jobTimeout = 3600000; // 1 hour
    m_clusterConfig.enableEncryption = true;
    m_clusterConfig.enableAuthentication = true;
    m_clusterConfig.listenPort = 8080;
}

void DistributedComputingEngine::initializeJobScheduler()
{
    // Initialize job scheduling system
    m_jobScheduler = std::make_unique<QTimer>();
    connect(m_jobScheduler.get(), &QTimer::timeout, this, &DistributedComputingEngine::scheduleNextJob);
    m_jobScheduler->start(1000); // Check for new jobs every second
}

void DistributedComputingEngine::scheduleNextJob()
{
    if (!m_isClusterLeader || m_jobQueue.isEmpty()) {
        return;
    }
    
    // Find available node
    ClusterNode* availableNode = findAvailableNode();
    if (!availableNode) {
        return; // No available nodes
    }
    
    // Get next job from queue
    QMutexLocker locker(&m_jobQueueMutex);
    if (m_jobQueue.isEmpty()) {
        return;
    }
    
    DistributedJob job = m_jobQueue.dequeue();
    job.status = JobStatus::Running;
    job.startTime = QDateTime::currentDateTime();
    job.executingNodeId = availableNode->nodeId;
    
    m_jobs[job.jobId] = job;
    
    // Send job to node
    ClusterMessage jobMessage;
    jobMessage.type = MessageType::JobAssignment;
    jobMessage.data["job"] = jobToJson(job);
    
    sendMessageToNode(availableNode->nodeId, jobMessage);
    
    // Update node status
    availableNode->currentJobs++;
    availableNode->lastActivity = QDateTime::currentDateTime();
    
    emit jobStarted(job.jobId);
    ForensicLogger::instance()->logInfo(QString("Job %1 assigned to node %2").arg(job.jobId, availableNode->nodeId));
}

ClusterNode* DistributedComputingEngine::findAvailableNode()
{
    QMutexLocker locker(&m_nodesMutex);
    
    // Find node with capacity based on load balancing strategy
    switch (m_loadBalancingStrategy) {
        case LoadBalancingStrategy::RoundRobin:
            return findNodeRoundRobin();
        case LoadBalancingStrategy::LeastLoaded:
            return findLeastLoadedNode();
        case LoadBalancingStrategy::CapabilityBased:
            return findBestCapabilityNode();
        default:
            return findNodeRoundRobin();
    }
}

ClusterNode* DistributedComputingEngine::findNodeRoundRobin()
{
    auto nodes = m_clusterNodes.values();
    if (nodes.isEmpty()) {
        return nullptr;
    }
    
    // Simple round-robin selection
    for (int i = 0; i < nodes.size(); ++i) {
        int index = (m_currentRoundRobinIndex + i) % nodes.size();
        ClusterNode* node = &nodes[index];
        
        if (node->isAvailable && node->currentJobs < node->maxConcurrentJobs) {
            m_currentRoundRobinIndex = (index + 1) % nodes.size();
            return node;
        }
    }
    
    return nullptr;
}

ClusterNode* DistributedComputingEngine::findLeastLoadedNode()
{
    ClusterNode* leastLoaded = nullptr;
    double minLoad = std::numeric_limits<double>::max();
    
    for (auto& node : m_clusterNodes) {
        if (node.isAvailable && node.currentJobs < node.maxConcurrentJobs) {
            double load = static_cast<double>(node.currentJobs) / node.maxConcurrentJobs;
            if (load < minLoad) {
                minLoad = load;
                leastLoaded = &node;
            }
        }
    }
    
    return leastLoaded;
}

ClusterNode* DistributedComputingEngine::findBestCapabilityNode()
{
    // This would implement capability-based selection
    // For now, fall back to least loaded
    return findLeastLoadedNode();
}

void DistributedComputingEngine::sendMessage(const ClusterMessage& message)
{
    // Serialize and send message
    QJsonObject messageJson;
    messageJson["type"] = static_cast<int>(message.type);
    messageJson["sender_id"] = message.senderId;
    messageJson["recipient_id"] = message.recipientId;
    messageJson["timestamp"] = message.timestamp.toString(Qt::ISODate);
    messageJson["data"] = message.data;
    
    QJsonDocument doc(messageJson);
    QByteArray messageData = doc.toJson(QJsonDocument::Compact);
    
    // Send to appropriate recipient(s)
    if (message.recipientId.isEmpty()) {
        broadcastMessage(message);
    } else {
        sendMessageToNode(message.recipientId, message);
    }
}

void DistributedComputingEngine::broadcastMessage(const ClusterMessage& message)
{
    QMutexLocker locker(&m_nodesMutex);
    
    for (const auto& node : m_clusterNodes) {
        sendMessageToNode(node.nodeId, message);
    }
}

void DistributedComputingEngine::sendMessageToNode(const QString& nodeId, const ClusterMessage& message)
{
    // Implementation would send message to specific node
    // This could use TCP sockets, message queues, or other communication mechanisms
    
    ForensicLogger::instance()->logDebug(QString("Sending message to node %1").arg(nodeId));
}

QJsonObject DistributedComputingEngine::jobToJson(const DistributedJob& job)
{
    QJsonObject jobJson;
    jobJson["job_id"] = job.jobId;
    jobJson["type"] = static_cast<int>(job.type);
    jobJson["priority"] = static_cast<int>(job.priority);
    jobJson["input_data"] = QString::fromUtf8(job.inputData.toBase64());
    jobJson["parameters"] = job.parameters;
    jobJson["submission_time"] = job.submissionTime.toString(Qt::ISODate);
    jobJson["timeout"] = job.timeoutSeconds;
    
    return jobJson;
}

QJsonObject DistributedComputingEngine::getNodeCapabilities() const
{
    QJsonObject capabilities;
    
    // System information
    auto memoryInfo = PhoenixDRS::Core::MemoryManager::instance().getSystemInfo();
    capabilities["total_memory"] = static_cast<qint64>(memoryInfo.totalSystemMemory);
    capabilities["available_memory"] = static_cast<qint64>(memoryInfo.availableSystemMemory);
    
    // CPU information
    capabilities["cpu_cores"] = QThread::idealThreadCount();
    
    // Supported job types
    QJsonArray supportedTypes;
    for (const auto& [type, description] : m_supportedJobTypes) {
        supportedTypes.append(static_cast<int>(type));
    }
    capabilities["supported_job_types"] = supportedTypes;
    
    // Node version and capabilities
    capabilities["node_version"] = "2.0.0";
    capabilities["max_concurrent_jobs"] = 4;
    
    return capabilities;
}

void DistributedComputingEngine::updatePerformanceMetrics()
{
    // Update cluster performance metrics
    m_clusterStats.activeNodes = m_activeNodes.load();
    m_clusterStats.totalJobs = m_totalJobs.load();
    m_clusterStats.completedJobs = m_completedJobs.load();
    
    // Calculate throughput (jobs per minute)
    static auto lastUpdate = std::chrono::steady_clock::now();
    static qint64 lastCompletedJobs = 0;
    
    auto now = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::minutes>(now - lastUpdate).count();
    
    if (elapsed > 0) {
        qint64 currentCompleted = m_completedJobs.load();
        m_clusterStats.throughput = (currentCompleted - lastCompletedJobs) / elapsed;
        lastCompletedJobs = currentCompleted;
        lastUpdate = now;
    }
}

void DistributedComputingEngine::sendHeartbeat()
{
    if (!m_isRunning.load()) {
        return;
    }
    
    ClusterMessage heartbeat;
    heartbeat.type = MessageType::Heartbeat;
    heartbeat.senderId = m_nodeId;
    heartbeat.timestamp = QDateTime::currentDateTime();
    heartbeat.data["status"] = "alive";
    heartbeat.data["load"] = getCurrentNodeLoad();
    
    if (m_isClusterLeader) {
        // Leader sends heartbeat to all nodes
        broadcastMessage(heartbeat);
    } else {
        // Worker nodes send heartbeat to leader
        sendMessage(heartbeat);
    }
}

double DistributedComputingEngine::getCurrentNodeLoad() const
{
    // Calculate current node load (simplified)
    auto memoryInfo = PhoenixDRS::Core::MemoryManager::instance().getSystemInfo();
    return static_cast<double>(memoryInfo.processMemoryUsage) / memoryInfo.totalSystemMemory;
}

void DistributedComputingEngine::cleanup()
{
    if (m_performanceTimer) {
        m_performanceTimer->stop();
    }
    
    if (m_heartbeatTimer) {
        m_heartbeatTimer->stop();
    }
    
    if (m_jobScheduler) {
        m_jobScheduler->stop();
    }
    
    // Clear all data structures
    m_jobQueue.clear();
    m_jobs.clear();
    m_jobResults.clear();
    m_clusterNodes.clear();
    
    ForensicLogger::instance()->logInfo("DistributedComputingEngine cleaned up");
}

int DistributedComputingEngine::countRunningJobs() const
{
    int count = 0;
    for (const auto& job : m_jobs) {
        if (job.status == JobStatus::Running) {
            count++;
        }
    }
    return count;
}

double DistributedComputingEngine::calculateAverageJobDuration() const
{
    qint64 totalDuration = 0;
    int completedCount = 0;
    
    for (const auto& job : m_jobs) {
        if (job.status == JobStatus::Completed && job.startTime.isValid() && job.endTime.isValid()) {
            totalDuration += job.startTime.msecsTo(job.endTime);
            completedCount++;
        }
    }
    
    return completedCount > 0 ? static_cast<double>(totalDuration) / completedCount : 0.0;
}

double DistributedComputingEngine::calculateClusterUtilization() const
{
    if (m_clusterNodes.isEmpty()) {
        return 0.0;
    }
    
    int totalCapacity = 0;
    int usedCapacity = 0;
    
    for (const auto& node : m_clusterNodes) {
        totalCapacity += node.maxConcurrentJobs;
        usedCapacity += node.currentJobs;
    }
    
    return totalCapacity > 0 ? static_cast<double>(usedCapacity) / totalCapacity : 0.0;
}

double DistributedComputingEngine::calculateThroughput() const
{
    return m_clusterStats.throughput;
}

} // namespace Forensics
} // namespace PhoenixDRS

#include "DistributedComputingEngine.moc"