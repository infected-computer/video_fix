/*
 * PhoenixDRS Professional - Advanced Network Forensics Engine Implementation
 * מימוש מנוע פורנזיקה מתקדם לרשתות - PhoenixDRS מקצועי
 */

#include "../include/NetworkForensicsEngine.h"
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
#include <QtNetwork/QHostAddress>
#include <QtNetwork/QNetworkInterface>

#include <algorithm>
#include <chrono>
#include <iomanip>
#include <sstream>

// Network analysis includes
#ifdef Q_OS_WIN
#include <winsock2.h>
#include <ws2tcpip.h>
#include <iphlpapi.h>
#pragma comment(lib, "ws2_32.lib")
#pragma comment(lib, "iphlpapi.lib")
#endif

#ifdef Q_OS_LINUX
#include <sys/socket.h>
#include <netinet/in.h>
#include <netinet/ip.h>
#include <netinet/tcp.h>
#include <netinet/udp.h>
#include <arpa/inet.h>
#include <pcap/pcap.h>
#endif

namespace PhoenixDRS {
namespace Forensics {

// Network Analysis Worker Class
class NetworkForensicsEngine::NetworkAnalysisWorker : public QObject
{
    Q_OBJECT

public:
    explicit NetworkAnalysisWorker(NetworkForensicsEngine* parent, const NetworkAnalysisParameters& params)
        : QObject(nullptr), m_engine(parent), m_params(params) {}

public slots:
    void performAnalysis();

signals:
    void finished();
    void error(const QString& message);
    void progressUpdate(int percentage);
    void packetAnalyzed(const PacketInfo& packet);
    void connectionFound(const NetworkConnection& connection);
    void protocolDetected(const ProtocolInfo& protocol);
    void anomalyDetected(const NetworkAnomaly& anomaly);

private:
    NetworkForensicsEngine* m_engine;
    NetworkAnalysisParameters m_params;
};

// NetworkForensicsEngine Implementation
NetworkForensicsEngine& NetworkForensicsEngine::instance()
{
    static NetworkForensicsEngine instance;
    return instance;
}

NetworkForensicsEngine::NetworkForensicsEngine(QObject* parent)
    : QObject(parent)
    , m_isRunning(false)
    , m_currentProgress(0)
    , m_totalPackets(0)
    , m_analyzedPackets(0)
    , m_detectedConnections(0)
    , m_workerThread(nullptr)
{
    setupProtocolSignatures();
    setupMaliciousPatterns();
    initializeNetworkInterfaces();
    
    // Initialize performance monitoring
    m_performanceTimer = std::make_unique<QTimer>();
    connect(m_performanceTimer.get(), &QTimer::timeout, this, &NetworkForensicsEngine::updatePerformanceMetrics);
    m_performanceTimer->start(1000); // Update every second
    
    ForensicLogger::instance()->logInfo("NetworkForensicsEngine initialized");
}

NetworkForensicsEngine::~NetworkForensicsEngine()
{
    if (m_isRunning.load()) {
        cancelAnalysis();
    }
    cleanup();
}

bool NetworkForensicsEngine::analyzePcapFile(const QString& pcapFilePath, const QString& outputDirectory, const NetworkAnalysisParameters& params)
{
    if (m_isRunning.load()) {
        emit error("Network analysis already in progress");
        return false;
    }

    // Validate input parameters
    if (!validateAnalysisParameters(pcapFilePath, outputDirectory, params)) {
        return false;
    }

    // Setup analysis environment
    if (!setupAnalysisEnvironment(outputDirectory)) {
        return false;
    }

    // Start analysis in separate thread
    m_workerThread = QThread::create([this, pcapFilePath, outputDirectory, params]() {
        performPcapAnalysis(pcapFilePath, outputDirectory, params);
    });

    connect(m_workerThread, &QThread::finished, this, &NetworkForensicsEngine::onAnalysisFinished);
    
    m_isRunning = true;
    m_workerThread->start();
    
    emit analysisStarted();
    ForensicLogger::instance()->logInfo(QString("PCAP analysis started: %1").arg(pcapFilePath));
    
    return true;
}

bool NetworkForensicsEngine::captureLiveTraffic(const QString& interface, const QString& outputDirectory, const NetworkCaptureParameters& params)
{
    if (m_isRunning.load()) {
        emit error("Network capture already in progress");
        return false;
    }

    try {
        // Validate interface
        if (!isValidNetworkInterface(interface)) {
            throw PhoenixDRS::Core::PhoenixException(
                PhoenixDRS::Core::ErrorCode::InvalidParameter,
                QString("Invalid network interface: %1").arg(interface),
                "NetworkForensicsEngine::captureLiveTraffic"
            );
        }

        // Setup capture environment
        if (!setupCaptureEnvironment(outputDirectory)) {
            return false;
        }

        // Start live capture
        return startLiveCapture(interface, outputDirectory, params);

    } catch (const PhoenixDRS::Core::PhoenixException& e) {
        emit error(e.message());
        return false;
    }
}

QList<NetworkConnection> NetworkForensicsEngine::reconstructTcpStreams(const QString& pcapFilePath)
{
    QList<NetworkConnection> connections;
    
    try {
        // Open PCAP file
        auto packets = loadPacketsFromPcap(pcapFilePath);
        
        // Group packets by connection
        std::unordered_map<QString, QList<PacketInfo>> connectionMap;
        
        for (const auto& packet : packets) {
            if (packet.protocol == "TCP") {
                QString connectionId = generateConnectionId(packet);
                connectionMap[connectionId].append(packet);
            }
        }
        
        // Reconstruct each TCP stream
        for (const auto& [connectionId, packets] : connectionMap) {
            NetworkConnection connection = reconstructTcpConnection(packets);
            if (connection.isValid()) {
                connections.append(connection);
            }
        }
        
        // Sort by timestamp
        std::sort(connections.begin(), connections.end(),
                 [](const NetworkConnection& a, const NetworkConnection& b) {
                     return a.startTime < b.startTime;
                 });
        
    } catch (const PhoenixDRS::Core::PhoenixException& e) {
        ForensicLogger::instance()->logError(e.message());
    }
    
    return connections;
}

QList<ExtractedFile> NetworkForensicsEngine::extractFilesFromTraffic(const QString& pcapFilePath, const QString& outputDirectory)
{
    QList<ExtractedFile> extractedFiles;
    
    try {
        // Reconstruct TCP streams first
        auto connections = reconstructTcpStreams(pcapFilePath);
        
        for (const auto& connection : connections) {
            // Analyze HTTP traffic for file transfers
            if (connection.applicationProtocol == "HTTP" || connection.applicationProtocol == "HTTPS") {
                auto httpFiles = extractHttpFiles(connection, outputDirectory);
                extractedFiles.append(httpFiles);
            }
            
            // Analyze FTP traffic
            else if (connection.applicationProtocol == "FTP") {
                auto ftpFiles = extractFtpFiles(connection, outputDirectory);
                extractedFiles.append(ftpFiles);
            }
            
            // Analyze SMTP traffic for email attachments
            else if (connection.applicationProtocol == "SMTP") {
                auto emailFiles = extractEmailAttachments(connection, outputDirectory);
                extractedFiles.append(emailFiles);
            }
        }
        
    } catch (const PhoenixDRS::Core::PhoenixException& e) {
        ForensicLogger::instance()->logError(e.message());
    }
    
    return extractedFiles;
}

QList<NetworkAnomaly> NetworkForensicsEngine::detectNetworkAnomalies(const QString& pcapFilePath)
{
    QList<NetworkAnomaly> anomalies;
    
    try {
        auto packets = loadPacketsFromPcap(pcapFilePath);
        
        // Detect various types of anomalies
        anomalies.append(detectPortScans(packets));
        anomalies.append(detectDdosAttacks(packets));
        anomalies.append(detectMaliciousTraffic(packets));
        anomalies.append(detectUnusualProtocols(packets));
        anomalies.append(detectDataExfiltration(packets));
        
        // Sort by severity
        std::sort(anomalies.begin(), anomalies.end(),
                 [](const NetworkAnomaly& a, const NetworkAnomaly& b) {
                     return a.severity > b.severity;
                 });
        
    } catch (const PhoenixDRS::Core::PhoenixException& e) {
        ForensicLogger::instance()->logError(e.message());
    }
    
    return anomalies;
}

QJsonObject NetworkForensicsEngine::generateTrafficStatistics(const QString& pcapFilePath)
{
    QJsonObject statistics;
    
    try {
        auto packets = loadPacketsFromPcap(pcapFilePath);
        
        // Basic statistics
        statistics["total_packets"] = static_cast<qint64>(packets.size());
        statistics["analysis_timestamp"] = QDateTime::currentDateTime().toString(Qt::ISODate);
        
        // Protocol distribution
        QJsonObject protocolStats;
        std::unordered_map<QString, int> protocolCounts;
        
        for (const auto& packet : packets) {
            protocolCounts[packet.protocol]++;
        }
        
        for (const auto& [protocol, count] : protocolCounts) {
            protocolStats[protocol] = count;
        }
        statistics["protocol_distribution"] = protocolStats;
        
        // Traffic volume statistics
        QJsonObject volumeStats;
        qint64 totalBytes = 0;
        qint64 inboundBytes = 0;
        qint64 outboundBytes = 0;
        
        for (const auto& packet : packets) {
            totalBytes += packet.size;
            if (packet.direction == PacketDirection::Inbound) {
                inboundBytes += packet.size;
            } else {
                outboundBytes += packet.size;
            }
        }
        
        volumeStats["total_bytes"] = totalBytes;
        volumeStats["inbound_bytes"] = inboundBytes;
        volumeStats["outbound_bytes"] = outboundBytes;
        statistics["traffic_volume"] = volumeStats;
        
        // Top talkers
        QJsonArray topTalkers;
        std::unordered_map<QString, qint64> hostTraffic;
        
        for (const auto& packet : packets) {
            QString sourceKey = QString("%1:%2").arg(packet.sourceIp).arg(packet.sourcePort);
            QString destKey = QString("%1:%2").arg(packet.destinationIp).arg(packet.destinationPort);
            
            hostTraffic[sourceKey] += packet.size;
            hostTraffic[destKey] += packet.size;
        }
        
        // Sort and get top 10
        std::vector<std::pair<QString, qint64>> sortedHosts(hostTraffic.begin(), hostTraffic.end());
        std::sort(sortedHosts.begin(), sortedHosts.end(),
                 [](const std::pair<QString, qint64>& a, const std::pair<QString, qint64>& b) {
                     return a.second > b.second;
                 });
        
        for (size_t i = 0; i < std::min(sortedHosts.size(), size_t(10)); ++i) {
            QJsonObject talker;
            talker["host"] = sortedHosts[i].first;
            talker["bytes"] = sortedHosts[i].second;
            topTalkers.append(talker);
        }
        statistics["top_talkers"] = topTalkers;
        
    } catch (const PhoenixDRS::Core::PhoenixException& e) {
        ForensicLogger::instance()->logError(e.message());
    }
    
    return statistics;
}

QList<QString> NetworkForensicsEngine::getAvailableNetworkInterfaces()
{
    QList<QString> interfaces;
    
    try {
        auto qInterfaces = QNetworkInterface::allInterfaces();
        
        for (const auto& interface : qInterfaces) {
            if (interface.flags() & QNetworkInterface::IsUp &&
                interface.flags() & QNetworkInterface::IsRunning &&
                !(interface.flags() & QNetworkInterface::IsLoopBack)) {
                
                interfaces.append(interface.name());
            }
        }
        
    } catch (const std::exception& e) {
        ForensicLogger::instance()->logError(QString("Error getting network interfaces: %1").arg(e.what()));
    }
    
    return interfaces;
}

void NetworkForensicsEngine::cancelAnalysis()
{
    if (!m_isRunning.load()) {
        return;
    }
    
    m_shouldCancel = true;
    
    if (m_workerThread && m_workerThread->isRunning()) {
        m_workerThread->requestInterruption();
        if (!m_workerThread->wait(5000)) {
            m_workerThread->terminate();
            m_workerThread->wait();
        }
    }
    
    m_isRunning = false;
    emit analysisCancelled();
    
    ForensicLogger::instance()->logInfo("Network analysis cancelled");
}

// Private Implementation Methods

bool NetworkForensicsEngine::validateAnalysisParameters(const QString& pcapFilePath, const QString& outputDirectory, const NetworkAnalysisParameters& params)
{
    // Validate PCAP file
    QFileInfo pcapFile(pcapFilePath);
    if (!pcapFile.exists()) {
        emit error(QString("PCAP file not found: %1").arg(pcapFilePath));
        return false;
    }
    
    if (!pcapFile.isReadable()) {
        emit error(QString("Cannot read PCAP file: %1").arg(pcapFilePath));
        return false;
    }
    
    // Validate output directory
    QDir outputDir(outputDirectory);
    if (!outputDir.exists()) {
        if (!outputDir.mkpath(".")) {
            emit error(QString("Cannot create output directory: %1").arg(outputDirectory));
            return false;
        }
    }
    
    return true;
}

bool NetworkForensicsEngine::setupAnalysisEnvironment(const QString& outputDirectory)
{
    try {
        // Create analysis subdirectories
        QDir outputDir(outputDirectory);
        
        QStringList subdirs = {"connections", "extracted_files", "protocols", "anomalies", "statistics", "timeline"};
        for (const QString& subdir : subdirs) {
            if (!outputDir.mkpath(subdir)) {
                throw PhoenixDRS::Core::PhoenixException(
                    PhoenixDRS::Core::ErrorCode::FileAccessError,
                    QString("Failed to create analysis directory: %1").arg(subdir),
                    "NetworkForensicsEngine::setupAnalysisEnvironment"
                );
            }
        }
        
        // Initialize analysis session
        m_currentAnalysisId = QUuid::createUuid().toString();
        m_analysisStartTime = QDateTime::currentDateTime();
        m_analysisTimer.start();
        
        // Reset counters
        m_currentProgress = 0;
        m_totalPackets = 0;
        m_analyzedPackets = 0;
        m_detectedConnections = 0;
        m_shouldCancel = false;
        
        return true;
        
    } catch (const PhoenixDRS::Core::PhoenixException& e) {
        emit error(e.message());
        return false;
    }
}

void NetworkForensicsEngine::performPcapAnalysis(const QString& pcapFilePath, const QString& outputDirectory, const NetworkAnalysisParameters& params)
{
    try {
        emit progressUpdate(5);
        
        // Load and parse PCAP file
        auto packets = loadPacketsFromPcap(pcapFilePath);
        m_totalPackets = packets.size();
        emit progressUpdate(15);
        
        // Analyze packet headers
        if (params.analyzeHeaders) {
            analyzePacketHeaders(packets, outputDirectory);
            emit progressUpdate(30);
        }
        
        // Reconstruct network connections
        if (params.reconstructConnections) {
            auto connections = reconstructAllConnections(packets);
            saveConnectionsToFile(connections, outputDirectory);
            emit progressUpdate(50);
        }
        
        // Extract files from traffic
        if (params.extractFiles) {
            auto extractedFiles = extractFilesFromPackets(packets, outputDirectory);
            saveExtractedFilesInfo(extractedFiles, outputDirectory);
            emit progressUpdate(70);
        }
        
        // Detect anomalies and threats
        if (params.detectAnomalies) {
            auto anomalies = detectAnomaliesInPackets(packets);
            saveAnomaliesToFile(anomalies, outputDirectory);
            emit progressUpdate(85);
        }
        
        // Generate comprehensive statistics
        auto statistics = generateComprehensiveStatistics(packets);
        saveStatisticsToFile(statistics, outputDirectory);
        emit progressUpdate(95);
        
        // Finalize analysis
        finalizeNetworkAnalysis(outputDirectory);
        emit progressUpdate(100);
        
        emit analysisCompleted();
        
    } catch (const PhoenixDRS::Core::PhoenixException& e) {
        emit error(e.message());
    } catch (const std::exception& e) {
        emit error(QString("System error: %1").arg(e.what()));
    }
}

void NetworkForensicsEngine::setupProtocolSignatures()
{
    // HTTP signatures
    m_protocolSignatures["HTTP"] = {
        QByteArray("GET "), QByteArray("POST "), QByteArray("PUT "), QByteArray("DELETE "),
        QByteArray("HTTP/1.0"), QByteArray("HTTP/1.1"), QByteArray("HTTP/2.0")
    };
    
    // HTTPS/TLS signatures
    m_protocolSignatures["TLS"] = {
        QByteArray("\x16\x03\x01"), QByteArray("\x16\x03\x02"), QByteArray("\x16\x03\x03")
    };
    
    // FTP signatures
    m_protocolSignatures["FTP"] = {
        QByteArray("220 "), QByteArray("USER "), QByteArray("PASS "), QByteArray("RETR "), QByteArray("STOR ")
    };
    
    // SMTP signatures
    m_protocolSignatures["SMTP"] = {
        QByteArray("220 "), QByteArray("HELO "), QByteArray("EHLO "), QByteArray("MAIL FROM:"), QByteArray("RCPT TO:")
    };
    
    // DNS signatures
    m_protocolSignatures["DNS"] = {
        QByteArray("\x00\x01\x00\x00"), QByteArray("\x00\x01\x00\x01")
    };
}

void NetworkForensicsEngine::setupMaliciousPatterns()
{
    // Common malware communication patterns
    m_maliciousPatterns = {
        {"BOTNET_C2", QByteArray("cmd=")},
        {"SQL_INJECTION", QByteArray("UNION SELECT")},
        {"XSS_ATTACK", QByteArray("<script>")},
        {"SHELLCODE", QByteArray("\x90\x90\x90\x90")},
        {"METASPLOIT", QByteArray("msf_")},
    };
    
    // Suspicious domains and IPs
    m_suspiciousDomains = {
        "tempfile.xyz", "3322.org", "no-ip.com", "duckdns.org",
        "bit.ly", "tinyurl.com", "t.co", "goo.gl"
    };
}

void NetworkForensicsEngine::initializeNetworkInterfaces()
{
    auto interfaces = getAvailableNetworkInterfaces();
    
    for (const QString& interface : interfaces) {
        ForensicLogger::instance()->logInfo(QString("Available network interface: %1").arg(interface));
    }
}

void NetworkForensicsEngine::updatePerformanceMetrics()
{
    // Update memory usage
    m_currentMemoryUsage = PhoenixDRS::Core::MemoryManager::instance().getSystemInfo().processMemoryUsage;
    
    // Update packet processing rate
    static qint64 lastAnalyzedPackets = 0;
    qint64 currentAnalyzed = m_analyzedPackets.load();
    m_packetsPerSecond = currentAnalyzed - lastAnalyzedPackets;
    lastAnalyzedPackets = currentAnalyzed;
    
    // Update bandwidth utilization (simplified)
    m_currentBandwidth = m_packetsPerSecond * 1500; // Assume average packet size of 1500 bytes
}

void NetworkForensicsEngine::onAnalysisFinished()
{
    m_isRunning = false;
    
    if (m_workerThread) {
        m_workerThread->deleteLater();
        m_workerThread = nullptr;
    }
    
    ForensicLogger::instance()->logInfo("Network analysis completed");
}

void NetworkForensicsEngine::cleanup()
{
    if (m_performanceTimer) {
        m_performanceTimer->stop();
    }
    
    // Clear analysis data
    m_detectedConnections.clear();
    m_extractedFiles.clear();
    m_detectedAnomalies.clear();
    
    ForensicLogger::instance()->logInfo("NetworkForensicsEngine cleaned up");
}

QList<PacketInfo> NetworkForensicsEngine::loadPacketsFromPcap(const QString& pcapFilePath)
{
    QList<PacketInfo> packets;
    
    // This is a simplified implementation
    // In a real implementation, you would use libpcap or similar library
    
    QFile pcapFile(pcapFilePath);
    if (!pcapFile.open(QIODevice::ReadOnly)) {
        throw PhoenixDRS::Core::PhoenixException(
            PhoenixDRS::Core::ErrorCode::FileAccessError,
            QString("Cannot open PCAP file: %1").arg(pcapFilePath),
            "NetworkForensicsEngine::loadPacketsFromPcap"
        );
    }
    
    // Parse PCAP file format
    // This would involve reading the PCAP global header and packet records
    // For now, we'll create dummy data for demonstration
    
    return packets;
}

QString NetworkForensicsEngine::generateConnectionId(const PacketInfo& packet)
{
    // Generate unique connection identifier
    QStringList components = {
        packet.sourceIp, QString::number(packet.sourcePort),
        packet.destinationIp, QString::number(packet.destinationPort),
        packet.protocol
    };
    
    return components.join(":");
}

} // namespace Forensics
} // namespace PhoenixDRS

#include "NetworkForensicsEngine.moc"