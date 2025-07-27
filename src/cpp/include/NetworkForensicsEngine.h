/*
 * PhoenixDRS Professional - Advanced Network Forensics and Packet Reconstruction Engine
 * מנוע פורנזיקה מתקדם לרשתות ושחזור מנות - PhoenixDRS מקצועי
 * 
 * Next-generation network traffic analysis and communication reconstruction
 * ניתוח תעבורת רשת מהדור הבא ושחזור תקשורת
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
#include <QNetworkAccessManager>
#include <QHostAddress>
#include <QTcpSocket>
#include <QUdpSocket>

#include <memory>
#include <atomic>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <queue>
#include <map>

// Network analysis libraries
#ifdef ENABLE_PCAP
#include <pcap.h>
#include <pcap/pcap.h>
#endif

#ifdef ENABLE_WIRESHARK
#include <wireshark/epan/epan.h>
#include <wireshark/epan/dissectors/packet.h>
#endif

// Deep packet inspection
#ifdef ENABLE_NDPI
#include <ndpi/ndpi_api.h>
#include <ndpi/ndpi_typedefs.h>
#endif

// TLS/SSL analysis
#ifdef ENABLE_OPENSSL
#include <openssl/ssl.h>
#include <openssl/x509.h>
#include <openssl/x509v3.h>
#endif

// Network protocol headers
#ifdef ENABLE_NETINET
#include <netinet/ip.h>
#include <netinet/tcp.h>
#include <netinet/udp.h>
#include <netinet/ip_icmp.h>
#include <netinet/ether.h>
#include <arpa/inet.h>
#endif

namespace PhoenixDRS {

// Network protocol types
enum class NetworkProtocol {
    UNKNOWN = 0,
    
    // Layer 2 protocols
    ETHERNET,
    WIFI_80211,
    BLUETOOTH,
    ZIGBEE,
    LORA,
    
    // Layer 3 protocols  
    IPv4,
    IPv6,
    ICMP,
    ICMPv6,
    ARP,
    RARP,
    MPLS,
    
    // Layer 4 protocols
    TCP,
    UDP,
    SCTP,
    DCCP,
    
    // Application protocols
    HTTP,
    HTTPS,
    FTP,
    FTPS,
    SFTP,
    SSH,
    TELNET,
    SMTP,
    SMTPS,
    POP3,
    POP3S,
    IMAP,
    IMAPS,
    DNS,
    DHCP,
    SNMP,
    NTP,
    LDAP,
    LDAPS,
    KERBEROS,
    RADIUS,
    
    // Messaging protocols
    XMPP,
    IRC,
    SIP,
    RTP,
    RTCP,
    SKYPE,
    WHATSAPP,
    TELEGRAM,
    SIGNAL,
    DISCORD,
    SLACK,
    
    // File sharing
    BITTORRENT,
    GNUTELLA,
    EMULE,
    KAZAA,
    
    // VPN/Tunneling
    OPENVPN,
    IPSEC,
    L2TP,
    PPTP,
    WIREGUARD,
    TOR,
    I2P,
    
    // Cryptocurrency
    BITCOIN,
    ETHEREUM,
    MONERO,
    ZCASH,
    
    // Malware C&C
    MALWARE_HTTP,
    MALWARE_DNS,
    MALWARE_CUSTOM,
    RAT_PROTOCOL,
    BOTNET_PROTOCOL,
    
    // Industrial/IoT
    MODBUS,
    PROFINET,
    BACNET,
    MQTT,
    COAP,
    ZIGBEE_CLUSTER,
    
    // Streaming/Media
    RTMP,
    RTSP,
    HLS,
    MPEG_DASH,
    WEBRTC,
    
    // Cloud services
    AWS_API,
    AZURE_API,
    GCP_API,
    DROPBOX_API,
    ONEDRIVE_API,
    GDRIVE_API
};

// Traffic analysis types
enum class TrafficAnalysisType {
    PROTOCOL_IDENTIFICATION,    // Identify protocols used
    FLOW_RECONSTRUCTION,       // Reconstruct communication flows
    CONTENT_EXTRACTION,        // Extract transmitted content
    METADATA_ANALYSIS,         // Analyze communication metadata
    BEHAVIORAL_ANALYSIS,       // Identify communication patterns
    THREAT_DETECTION,          // Detect malicious traffic
    GEOLOCATION_ANALYSIS,      // Geographic traffic analysis
    TIMING_ANALYSIS,           // Communication timing patterns
    ENCRYPTION_ANALYSIS,       // Analyze encrypted communications
    STEGANOGRAPHY_DETECTION,   // Hidden data in network traffic
    ANOMALY_DETECTION,         // Detect unusual patterns
    CORRELATION_ANALYSIS,      // Cross-protocol correlation
    ATTRIBUTION_ANALYSIS,      // Link traffic to entities
    FORENSIC_TIMELINE,         // Create communication timeline
    EVIDENCE_EXTRACTION        // Extract forensic evidence
};

// Network forensics result
struct NetworkForensicsResult {
    QString analysisId;                    // Unique analysis identifier
    QDateTime analysisTime;               // Analysis timestamp
    QString dataSource;                   // Source file/interface
    qint64 totalPackets;                 // Total packets analyzed
    qint64 totalBytes;                   // Total bytes analyzed
    QDateTime captureStart;              // First packet timestamp
    QDateTime captureEnd;                // Last packet timestamp
    
    // Protocol distribution
    QJsonObject protocolDistribution;    // Protocol usage statistics
    QJsonArray detectedProtocols;        // All detected protocols
    QJsonArray encryptedProtocols;       // Encrypted communications
    QJsonArray suspiciousProtocols;      // Potentially malicious protocols
    
    // Communication flows
    QJsonArray tcpFlows;                 // TCP communication flows
    QJsonArray udpFlows;                 // UDP communication flows
    QJsonArray dnsQueries;               // DNS resolution requests
    QJsonArray httpRequests;             // HTTP/HTTPS requests
    QJsonArray ftpSessions;              // FTP file transfers
    QJsonArray emailSessions;            // Email communications
    
    // Extracted content
    QJsonArray extractedFiles;           // Files reconstructed from traffic
    QJsonArray extractedImages;          // Images from HTTP/email
    QJsonArray extractedDocuments;       // Documents transferred
    QJsonArray extractedCredentials;     // Captured usernames/passwords
    QJsonArray extractedCertificates;    // SSL/TLS certificates
    
    // Geolocation data
    QJsonArray geolocatedConnections;    // Geographic connection data
    QJsonObject trafficByCountry;        // Traffic distribution by country
    QJsonArray suspiciousGeolocations;   // Connections to suspicious locations
    
    // Threat intelligence
    QJsonArray maliciousIPs;             // Known malicious IP addresses
    QJsonArray suspiciousDomains;        // Potentially malicious domains
    QJsonArray malwareSignatures;        // Network-based malware signatures
    QJsonArray botnetIndicators;         // Botnet communication indicators
    QJsonArray c2Communications;         // Command & control traffic
    
    // Communication patterns
    QJsonObject communicationGraph;      // Communication relationship graph
    QJsonArray periodicCommunications;   // Regular communication patterns
    QJsonArray anomalousPatterns;        // Unusual communication patterns
    QJsonArray dataExfiltration;         // Potential data exfiltration
    
    // Forensic evidence
    QJsonArray evidencePackets;          // Key forensic packets
    QJsonArray reconstructedSessions;    // Reconstructed communication sessions
    QJsonArray digitalFingerprints;      // Digital fingerprints extracted
    QJsonArray timelineEvents;           // Communication timeline
    
    // Privacy analysis
    QJsonArray personalData;             // Personal information detected
    QJsonArray financialData;            // Financial information
    QJsonArray healthcareData;           // Medical information
    QJsonArray corporateData;            // Corporate/business data
    
    // Advanced analysis
    QJsonObject trafficEntropy;          // Traffic randomness analysis
    QJsonArray covertChannels;           // Hidden communication channels
    QJsonArray tunnelledProtocols;       // Protocols within other protocols
    QJsonArray steganographicData;       // Hidden data in traffic
    
    // Quality metrics
    double analysisConfidence;           // Overall confidence (0.0-1.0)
    QString integrityStatus;             // Data integrity assessment
    qint64 corruptedPackets;            // Number of corrupted packets
    qint64 duplicatePackets;            // Number of duplicate packets
    
    NetworkForensicsResult() : totalPackets(0), totalBytes(0), analysisConfidence(0.0),
                              corruptedPackets(0), duplicatePackets(0) {
        analysisId = QUuid::createUuid().toString(QUuid::WithoutBraces);
        analysisTime = QDateTime::currentDateTime();
    }
};

// Live capture parameters
struct LiveCaptureParams {
    QString interfaceName;               // Network interface to capture
    QString captureFilter;               // BPF capture filter
    int maxPacketsToCapture;            // Maximum packets to capture (0 = unlimited)
    int maxCaptureTimeSeconds;          // Maximum capture time (0 = unlimited)
    qint64 maxCaptureSizeMB;            // Maximum capture size in MB
    bool enablePromiscuousMode;         // Promiscuous mode capture
    bool enableTimestamping;            // High-precision timestamping
    bool enableDeepInspection;          // Deep packet inspection
    bool enableContentExtraction;       // Extract file content
    bool enableThreatDetection;         // Real-time threat detection
    bool saveToFile;                    // Save capture to file
    QString outputFilePath;             // Output file path
    QString outputFormat;               // "pcap", "pcapng", "custom"
    
    // Advanced options
    int snapLength;                     // Packet snap length
    int bufferSize;                     // Capture buffer size
    int captureTimeout;                 // Capture timeout in ms
    bool enableHardwareTimestamping;    // Hardware timestamping if available
    bool enablePacketMmap;              // Memory-mapped packet capture
    bool enableZeroCopy;                // Zero-copy packet capture
    
    LiveCaptureParams() : maxPacketsToCapture(0), maxCaptureTimeSeconds(0),
                         maxCaptureSizeMB(0), enablePromiscuousMode(true),
                         enableTimestamping(true), enableDeepInspection(true),
                         enableContentExtraction(true), enableThreatDetection(true),
                         saveToFile(true), outputFormat("pcapng"), snapLength(65535),
                         bufferSize(1024*1024), captureTimeout(1000),
                         enableHardwareTimestamping(false), enablePacketMmap(false),
                         enableZeroCopy(false) {}
};

// Packet reconstruction parameters
struct PacketReconstructionParams {
    bool reconstructTcpStreams;          // Reconstruct TCP streams
    bool reconstructUdpFlows;            // Reconstruct UDP flows
    bool reconstructFiles;               // Reconstruct transmitted files
    bool reconstructEmails;              // Reconstruct email messages
    bool reconstructWebPages;            // Reconstruct web pages
    bool reconstructVoipCalls;           // Reconstruct VoIP communications
    bool reconstructChatSessions;        // Reconstruct chat conversations
    bool reconstructDnsQueries;          // Reconstruct DNS queries
    
    // Content extraction
    bool extractImages;                  // Extract image files
    bool extractDocuments;               // Extract document files
    bool extractExecutables;             // Extract executable files
    bool extractCompressedFiles;         // Extract archives
    bool extractCertificates;            // Extract digital certificates
    bool extractCredentials;             // Extract login credentials
    
    // Filtering options
    QStringList includeProtocols;        // Only reconstruct these protocols
    QStringList excludeProtocols;        // Skip these protocols
    QStringList includeIpRanges;         // Only include these IP ranges
    QStringList excludeIpRanges;         // Exclude these IP ranges
    int minFileSize;                     // Minimum file size to extract
    int maxFileSize;                     // Maximum file size to extract
    
    // Output options
    QString outputDirectory;             // Directory for extracted content
    bool organizeByProtocol;            // Organize output by protocol
    bool organizeByTimestamp;           // Organize output by timestamp
    bool includeMetadata;               // Include metadata files
    bool compressOutput;                // Compress extracted files
    
    PacketReconstructionParams() : reconstructTcpStreams(true), reconstructUdpFlows(true),
                                  reconstructFiles(true), reconstructEmails(true),
                                  reconstructWebPages(true), reconstructVoipCalls(true),
                                  reconstructChatSessions(true), reconstructDnsQueries(true),
                                  extractImages(true), extractDocuments(true),
                                  extractExecutables(true), extractCompressedFiles(true),
                                  extractCertificates(true), extractCredentials(true),
                                  minFileSize(0), maxFileSize(100*1024*1024),
                                  organizeByProtocol(true), organizeByTimestamp(false),
                                  includeMetadata(true), compressOutput(false) {}
};

// Forward declarations
class PacketCaptureEngine;
class ProtocolAnalyzer;
class FlowReconstructor;
class ContentExtractor;
class ThreatDetector;
class GeolocationAnalyzer;
class CommunicationAnalyzer;
class NetworkTimeline;

/*
 * Advanced network forensics engine
 * מנוע פורנזיקה מתקדם לרשתות
 */
class PHOENIXDRS_EXPORT NetworkForensicsEngine : public QObject
{
    Q_OBJECT

public:
    explicit NetworkForensicsEngine(QObject* parent = nullptr);
    ~NetworkForensicsEngine() override;

    // Initialization
    bool initialize();
    void shutdown();
    bool isInitialized() const { return m_isInitialized; }
    
    // Packet capture file analysis
    bool analyzePacketCapture(const QString& pcapFilePath);
    bool analyzeLiveTraffic(const LiveCaptureParams& params);
    NetworkForensicsResult getAnalysisResult(const QString& analysisId) const;
    std::vector<NetworkForensicsResult> getAllAnalysisResults() const;
    
    // Live capture operations
    bool startLiveCapture(const LiveCaptureParams& params);
    void stopLiveCapture();
    bool isCapturing() const { return m_isCapturing.load(); }
    std::vector<QString> getAvailableInterfaces() const;
    
    // Packet reconstruction
    bool reconstructPackets(const QString& pcapFilePath, const PacketReconstructionParams& params);
    QString getReconstructionOutputPath() const { return m_reconstructionOutputPath; }
    
    // Protocol analysis
    struct ProtocolStatistics {
        NetworkProtocol protocol;
        QString protocolName;
        qint64 packetCount;
        qint64 byteCount;
        double percentage;
        QDateTime firstSeen;
        QDateTime lastSeen;
        std::vector<QString> endpoints;
        bool isEncrypted;
        bool isSuspicious;
        
        ProtocolStatistics() : protocol(NetworkProtocol::UNKNOWN), packetCount(0),
                              byteCount(0), percentage(0.0), isEncrypted(false),
                              isSuspicious(false) {}
    };
    
    std::vector<ProtocolStatistics> getProtocolStatistics() const;
    std::vector<ProtocolStatistics> getEncryptedProtocols() const;
    std::vector<ProtocolStatistics> getSuspiciousProtocols() const;
    
    // Communication flow analysis
    struct CommunicationFlow {
        QString flowId;
        NetworkProtocol protocol;
        QString sourceAddress;
        int sourcePort;
        QString destinationAddress;
        int destinationPort;
        QDateTime startTime;
        QDateTime endTime;
        qint64 packetsAtoB;
        qint64 packetsBtoA;
        qint64 bytesAtoB;
        qint64 bytesBtoA;
        QString flowState;
        bool isComplete;
        bool hasAnomalies;
        QString geoLocationSource;
        QString geoLocationDestination;
        QString applicationProtocol;
        bool isEncrypted;
        QString encryptionMethod;
        QJsonObject extractedContent;
        
        CommunicationFlow() : sourcePort(0), destinationPort(0), packetsAtoB(0),
                             packetsBtoA(0), bytesAtoB(0), bytesBtoA(0),
                             isComplete(false), hasAnomalies(false), isEncrypted(false) {}
    };
    
    std::vector<CommunicationFlow> getCommunicationFlows() const;
    std::vector<CommunicationFlow> getAnomalousFlows() const;
    std::vector<CommunicationFlow> getEncryptedFlows() const;
    CommunicationFlow getFlowById(const QString& flowId) const;
    
    // Content extraction
    struct ExtractedFile {
        QString fileName;
        QString filePath;
        QString fileType;
        qint64 fileSize;
        QString md5Hash;
        QString sha256Hash;
        QDateTime extractionTime;
        QString sourceProtocol;
        QString sourceFlow;
        QString sourceAddress;
        QString destinationAddress;
        bool isComplete;
        bool isSuspicious;
        QString suspiciousReason;
        
        ExtractedFile() : fileSize(0), isComplete(false), isSuspicious(false) {}
    };
    
    std::vector<ExtractedFile> getExtractedFiles() const;
    std::vector<ExtractedFile> getSuspiciousFiles() const;
    bool exportExtractedFile(const QString& fileId, const QString& outputPath);
    
    // Credential extraction
    struct ExtractedCredential {
        QString protocol;
        QString service;
        QString username;
        QString password;
        QString hash;
        QString hashType;
        QString sourceAddress;
        QString destinationAddress;
        QDateTime captureTime;
        QString extractionMethod;
        bool isHashed;
        bool isEncrypted;
        
        ExtractedCredential() : isHashed(false), isEncrypted(false) {}
    };
    
    std::vector<ExtractedCredential> getExtractedCredentials() const;
    bool exportCredentialList(const QString& outputPath, const QString& format = "json");
    
    // Geolocation analysis
    struct GeoLocationInfo {
        QString ipAddress;
        QString country;
        QString region;
        QString city;
        QString organization;
        QString isp;
        double latitude;
        double longitude;
        QString timeZone;
        bool isVpn;
        bool isTor;
        bool isProxy;
        bool isMalicious;
        QString threatLevel;
        QStringList threatCategories;
        
        GeoLocationInfo() : latitude(0.0), longitude(0.0), isVpn(false),
                           isTor(false), isProxy(false), isMalicious(false) {}
    };
    
    std::vector<GeoLocationInfo> getGeoLocationData() const;
    std::vector<GeoLocationInfo> getSuspiciousLocations() const;
    QJsonObject generateGeoLocationMap() const;
    
    // Threat detection
    struct NetworkThreat {
        QString threatId;
        QString threatType;
        QString threatName;
        QString description;
        QString severity;
        double confidence;
        QDateTime detectionTime;
        QString sourceAddress;
        QString destinationAddress;
        NetworkProtocol protocol;
        QJsonArray indicators;
        QJsonArray affectedFlows;
        QString mitigationAdvice;
        
        NetworkThreat() : confidence(0.0), protocol(NetworkProtocol::UNKNOWN) {}
    };
    
    std::vector<NetworkThreat> getDetectedThreats() const;
    std::vector<NetworkThreat> getCriticalThreats() const;
    bool blockThreat(const QString& threatId);
    
    // Communication timeline
    struct TimelineEvent {
        QDateTime timestamp;
        QString eventType;
        QString description;
        QString sourceAddress;
        QString destinationAddress;
        NetworkProtocol protocol;
        qint64 dataSize;
        QString direction;
        QJsonObject metadata;
        QString forensicSignificance;
        
        TimelineEvent() : protocol(NetworkProtocol::UNKNOWN), dataSize(0) {}
    };
    
    std::vector<TimelineEvent> getCommunicationTimeline() const;
    std::vector<TimelineEvent> getTimelineByProtocol(NetworkProtocol protocol) const;
    std::vector<TimelineEvent> getTimelineByAddress(const QString& address) const;
    
    // Advanced analysis
    struct TrafficPattern {
        QString patternId;
        QString patternName;
        QString description;
        QString patternType;
        double confidence;
        QDateTime firstObserved;
        QDateTime lastObserved;
        int occurrenceCount;
        std::vector<QString> involvedAddresses;
        std::vector<NetworkProtocol> involvedProtocols;
        bool isSuspicious;
        QString suspiciousReason;
        
        TrafficPattern() : confidence(0.0), occurrenceCount(0), isSuspicious(false) {}
    };
    
    std::vector<TrafficPattern> identifyTrafficPatterns() const;
    std::vector<TrafficPattern> getSuspiciousPatterns() const;
    bool correlateWithExternalThreatIntel(const QString& threatIntelSource);
    
    // Steganography detection
    struct SteganographyEvidence {
        QString evidenceId;
        QString detectionMethod;
        QString hiddenDataType;
        qint64 estimatedDataSize;
        QString carrierProtocol;
        QString sourceAddress;
        QString destinationAddress;
        double confidence;
        QByteArray extractedData;
        QString description;
        
        SteganographyEvidence() : estimatedDataSize(0), confidence(0.0) {}
    };
    
    std::vector<SteganographyEvidence> detectSteganography() const;
    bool extractHiddenData(const QString& evidenceId, const QString& outputPath);
    
    // Configuration
    void setLiveCaptureParams(const LiveCaptureParams& params);
    LiveCaptureParams getLiveCaptureParams() const { return m_liveParams; }
    void setReconstructionParams(const PacketReconstructionParams& params);
    PacketReconstructionParams getReconstructionParams() const { return m_reconstructionParams; }
    
    // Filter management
    void addPacketFilter(const QString& bpfFilter);
    void removePacketFilter(const QString& bpfFilter);
    std::vector<QString> getActiveFilters() const;
    bool validateBpfFilter(const QString& filter);
    
    // Export and reporting
    bool exportAnalysisReport(const QString& filePath, const QString& format = "json");
    bool exportPacketData(const QString& filePath, const QString& format = "pcap");
    bool exportCommunicationFlows(const QString& filePath);
    bool exportTimelineReport(const QString& filePath);
    bool exportThreatIntelligence(const QString& filePath);
    QJsonObject generateForensicReport() const;
    
    // Integration with external tools
    bool integrateWithVirusTotal(const QString& apiKey);
    bool integrateWithShodanIO(const QString& apiKey);
    bool integrateWithAbuseIPDB(const QString& apiKey);
    bool queryThreatIntelligence(const QString& indicator, const QString& service);
    
    // Performance monitoring
    struct CaptureStatistics {
        qint64 packetsReceived;
        qint64 packetsDropped;
        qint64 packetsIfDropped;
        qint64 bytesReceived;
        double packetsPerSecond;
        double bytesPerSecond;
        double cpuUsage;
        double memoryUsage;
        QDateTime captureStartTime;
        qint64 captureElapsedSeconds;
        
        CaptureStatistics() : packetsReceived(0), packetsDropped(0),
                             packetsIfDropped(0), bytesReceived(0),
                             packetsPerSecond(0.0), bytesPerSecond(0.0),
                             cpuUsage(0.0), memoryUsage(0.0), captureElapsedSeconds(0) {}
    };
    
    CaptureStatistics getCaptureStatistics() const;
    void resetStatistics();

signals:
    void analysisStarted(const QString& analysisId);
    void analysisProgress(const QString& analysisId, int progressPercent);
    void analysisCompleted(const QString& analysisId, bool success);
    void liveCaptureStar ted();
    void liveCaptureProgress(qint64 packetsReceived, qint64 bytesReceived);
    void liveCaptureStopped();
    void packetReceived(const QByteArray& packetData, qint64 timestamp);
    void protocolDetected(const ProtocolStatistics& protocol);
    void communicationFlowDetected(const CommunicationFlow& flow);
    void fileExtracted(const ExtractedFile& file);
    void credentialExtracted(const ExtractedCredential& credential);
    void threatDetected(const NetworkThreat& threat);
    void suspiciousPatternDetected(const TrafficPattern& pattern);
    void steganographyDetected(const SteganographyEvidence& evidence);
    void timelineEventDetected(const TimelineEvent& event);
    void geoLocationAnalyzed(const GeoLocationInfo& location);
    void errorOccurred(const QString& error);
    void captureBufferFull();
    void packetDropped(qint64 droppedCount);

private:
    // Core functionality
    bool initializePacketCapture();
    bool initializeProtocolAnalyzers();
    bool loadThreatIntelligence();
    void cleanupResources();
    
    // Packet processing
    void processPacket(const QByteArray& packetData, qint64 timestamp);
    NetworkProtocol identifyProtocol(const QByteArray& packetData);
    void updateFlowStatistics(const QByteArray& packetData);
    void extractPacketContent(const QByteArray& packetData);
    
    // Flow reconstruction
    void reconstructTcpFlow(const QString& flowId);
    void reconstructUdpFlow(const QString& flowId);
    bool extractFileFromFlow(const QString& flowId);
    bool extractCredentialsFromFlow(const QString& flowId);
    
    // Analysis implementation
    void performProtocolAnalysis();
    void performFlowAnalysis();
    void performContentExtraction();
    void performThreatDetection();
    void performGeolocationAnalysis();
    void performPatternAnalysis();
    void performSteganographyDetection();
    void buildCommunicationTimeline();
    
    // Threat intelligence
    bool queryVirusTotal(const QString& hash);
    bool queryShodan(const QString& ipAddress);
    bool queryAbuseIPDB(const QString& ipAddress);
    void updateThreatDatabase();
    
    // Member variables
    std::atomic<bool> m_isInitialized{false};
    std::atomic<bool> m_isAnalyzing{false};
    std::atomic<bool> m_isCapturing{false};
    std::atomic<bool> m_shouldCancel{false};
    
    LiveCaptureParams m_liveParams;
    PacketReconstructionParams m_reconstructionParams;
    CaptureStatistics m_captureStats;
    
    // Analysis components
    std::unique_ptr<PacketCaptureEngine> m_captureEngine;
    std::unique_ptr<ProtocolAnalyzer> m_protocolAnalyzer;
    std::unique_ptr<FlowReconstructor> m_flowReconstructor;
    std::unique_ptr<ContentExtractor> m_contentExtractor;
    std::unique_ptr<ThreatDetector> m_threatDetector;
    std::unique_ptr<GeolocationAnalyzer> m_geoAnalyzer;
    std::unique_ptr<CommunicationAnalyzer> m_commAnalyzer;
    std::unique_ptr<NetworkTimeline> m_timeline;
    
    // Data storage
    std::unordered_map<QString, NetworkForensicsResult> m_analysisResults;
    std::vector<ProtocolStatistics> m_protocolStats;
    std::vector<CommunicationFlow> m_communicationFlows;
    std::vector<ExtractedFile> m_extractedFiles;
    std::vector<ExtractedCredential> m_extractedCredentials;
    std::vector<NetworkThreat> m_detectedThreats;
    std::vector<GeoLocationInfo> m_geoLocationData;
    std::vector<TrafficPattern> m_trafficPatterns;
    std::vector<SteganographyEvidence> m_steganographyEvidence;
    std::vector<TimelineEvent> m_timelineEvents;
    
    // Capture state
    QString m_currentAnalysisId;
    QString m_currentCaptureFile;
    QString m_reconstructionOutputPath;
    std::vector<QString> m_activeFilters;
    std::vector<QString> m_availableInterfaces;
    
    // External API keys
    QString m_virusTotalApiKey;
    QString m_shodanApiKey;
    QString m_abuseIPDBApiKey;
    
    // Thread safety
    mutable QMutex m_analysisMutex;
    mutable QMutex m_captureMutex;
    mutable QMutex m_dataMutex;
    
    // Packet capture handle
#ifdef ENABLE_PCAP
    pcap_t* m_pcapHandle{nullptr};
#endif
    
    // Constants
    static constexpr int MAX_ANALYSIS_RESULTS = 100;
    static constexpr int MAX_FLOW_CACHE = 100000;
    static constexpr int MAX_PACKET_BUFFER = 1000000;
    static constexpr int DEFAULT_SNAP_LENGTH = 65535;
    static constexpr int CAPTURE_TIMEOUT_MS = 1000;
};

} // namespace PhoenixDRS