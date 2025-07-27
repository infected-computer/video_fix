#ifndef PLUGIN_MANAGER_H
#define PLUGIN_MANAGER_H

#include <QObject>
#include <QMap>
#include <QVariant>
#include <QJsonObject>
#include <QLibrary>
#include <QDir>
#include <QMutex>
#include <QFuture>
#include <memory>
#include <functional>

// Forward declarations
class IPlugin;
class PluginMetadata;
class PluginRegistry;
class PluginSandbox;

/**
 * @brief Plugin interface types for different system components
 */
enum class PluginType {
    Analysis,           // File analysis plugins
    Export,            // Data export format plugins
    Signature,         // File signature detection plugins
    Storage,           // Storage backend plugins
    AI,                // AI/ML model plugins
    Codec,             // Media codec plugins
    Network,           // Network protocol plugins
    Forensic,          // Specialized forensic tools
    Visualization,     // Data visualization plugins
    Custom             // User-defined plugins
};

/**
 * @brief Plugin execution context and capabilities
 */
struct PluginCapabilities {
    bool requiresGPU = false;
    bool requiresNetwork = false;
    bool requiresFileSystem = false;
    bool requiresHardwareAccess = false;
    bool isThreadSafe = true;
    bool supportsHotReload = false;
    QStringList dependencies;
    QStringList conflictsWith;
    int maxInstances = 1;
    int memoryLimitMB = 512;
    int executionTimeoutSec = 300;
};

/**
 * @brief Base interface for all plugins
 */
class IPlugin {
public:
    virtual ~IPlugin() = default;
    
    // Plugin identification
    virtual QString getName() const = 0;
    virtual QString getVersion() const = 0;
    virtual QString getAuthor() const = 0;
    virtual QString getDescription() const = 0;
    virtual PluginType getType() const = 0;
    virtual PluginCapabilities getCapabilities() const = 0;
    
    // Plugin lifecycle
    virtual bool initialize(const QJsonObject& config = {}) = 0;
    virtual void shutdown() = 0;
    virtual bool isInitialized() const = 0;
    
    // Configuration
    virtual QJsonObject getDefaultConfig() const { return {}; }
    virtual bool validateConfig(const QJsonObject& config) const { Q_UNUSED(config); return true; }
    virtual void updateConfig(const QJsonObject& config) { Q_UNUSED(config); }
    
    // Plugin operation
    virtual QVariant execute(const QString& operation, const QVariantMap& parameters = {}) = 0;
    virtual bool supportsOperation(const QString& operation) const = 0;
    virtual QStringList getSupportedOperations() const = 0;
    
    // Status and monitoring
    virtual QString getStatus() const { return "Unknown"; }
    virtual QVariantMap getMetrics() const { return {}; }
    virtual QString getLastError() const { return m_lastError; }
    
protected:
    mutable QString m_lastError;
    
    void setError(const QString& error) const {
        m_lastError = error;
    }
};

/**
 * @brief Plugin metadata container
 */
class PluginMetadata {
public:
    PluginMetadata() = default;
    PluginMetadata(const QJsonObject& json);
    
    QString id;
    QString name;
    QString version;
    QString author;
    QString description;
    PluginType type;
    PluginCapabilities capabilities;
    QString filePath;
    QString pythonModule;
    QJsonObject configuration;
    QDateTime loadTime;
    QDateTime lastUsed;
    bool isActive = false;
    bool isLoaded = false;
    int usageCount = 0;
    
    QJsonObject toJson() const;
    static PluginMetadata fromJson(const QJsonObject& json);
    
    bool isValid() const;
    QString getUniqueId() const;
};

/**
 * @brief Plugin registry for metadata and dependency management
 */
class PluginRegistry : public QObject {
    Q_OBJECT
    
public:
    explicit PluginRegistry(QObject* parent = nullptr);
    ~PluginRegistry();
    
    // Registry management
    void registerPlugin(const PluginMetadata& metadata);
    void unregisterPlugin(const QString& pluginId);
    bool isRegistered(const QString& pluginId) const;
    
    // Plugin discovery
    QList<PluginMetadata> getAllPlugins() const;
    QList<PluginMetadata> getPluginsByType(PluginType type) const;
    PluginMetadata getPlugin(const QString& pluginId) const;
    QStringList getPluginIds() const;
    
    // Dependency resolution
    QStringList resolveDependencies(const QString& pluginId) const;
    bool checkConflicts(const QString& pluginId) const;
    QStringList getConflicts(const QString& pluginId) const;
    
    // Persistence
    bool saveRegistry(const QString& filePath = QString()) const;
    bool loadRegistry(const QString& filePath = QString());
    
signals:
    void pluginRegistered(const QString& pluginId);
    void pluginUnregistered(const QString& pluginId);
    void registryChanged();
    
private:
    mutable QMutex m_mutex;
    QMap<QString, PluginMetadata> m_plugins;
    QString m_registryPath;
    
    QString getDefaultRegistryPath() const;
};

/**
 * @brief Plugin sandbox for secure execution
 */
class PluginSandbox : public QObject {
    Q_OBJECT
    
public:
    explicit PluginSandbox(const PluginCapabilities& capabilities, QObject* parent = nullptr);
    ~PluginSandbox();
    
    // Sandbox configuration
    void setMemoryLimit(int limitMB);
    void setExecutionTimeout(int timeoutSec);
    void setFileSystemAccess(const QStringList& allowedPaths);
    void setNetworkAccess(bool allowed);
    
    // Secure execution
    QVariant executeInSandbox(std::function<QVariant()> operation);
    bool isWithinLimits() const;
    
    // Resource monitoring
    int getCurrentMemoryUsage() const;
    int getExecutionTime() const;
    QStringList getAccessedFiles() const;
    QStringList getNetworkConnections() const;
    
signals:
    void resourceLimitExceeded(const QString& resource);
    void securityViolation(const QString& violation);
    
private:
    PluginCapabilities m_capabilities;
    int m_memoryLimitMB;
    int m_executionTimeoutSec;
    QStringList m_allowedPaths;
    bool m_networkAllowed;
    
    // Monitoring data
    mutable QMutex m_monitorMutex;
    int m_currentMemoryMB;
    int m_executionTimeSec;
    QStringList m_accessedFiles;
    QStringList m_networkConnections;
    
    void startMonitoring();
    void stopMonitoring();
    bool checkResourceLimits();
};

/**
 * @brief Main plugin manager for loading, managing, and orchestrating plugins
 */
class PluginManager : public QObject {
    Q_OBJECT
    
public:
    static PluginManager* instance();
    
    // Plugin discovery and loading
    void scanForPlugins(const QString& directory);
    void scanForPythonPlugins(const QString& directory);
    bool loadPlugin(const QString& pluginId);
    bool unloadPlugin(const QString& pluginId);
    bool reloadPlugin(const QString& pluginId);
    
    // Plugin access
    IPlugin* getPlugin(const QString& pluginId);
    QList<IPlugin*> getPluginsByType(PluginType type);
    bool isPluginLoaded(const QString& pluginId) const;
    QStringList getLoadedPlugins() const;
    
    // Plugin execution
    QVariant executePlugin(const QString& pluginId, const QString& operation, 
                          const QVariantMap& parameters = {});
    QFuture<QVariant> executePluginAsync(const QString& pluginId, const QString& operation,
                                        const QVariantMap& parameters = {});
    
    // Configuration management
    void setPluginConfig(const QString& pluginId, const QJsonObject& config);
    QJsonObject getPluginConfig(const QString& pluginId) const;
    void savePluginConfigs();
    void loadPluginConfigs();
    
    // Hot reload support
    void enableHotReload(bool enabled = true);
    void watchPluginDirectories();
    
    // Plugin marketplace integration
    void updatePluginMarketplace();
    QList<PluginMetadata> getAvailablePlugins() const;
    bool installPlugin(const QString& pluginId, const QString& source = QString());
    bool updatePlugin(const QString& pluginId);
    bool removePlugin(const QString& pluginId);
    
    // System integration
    PluginRegistry* getRegistry() const { return m_registry.get(); }
    void setPluginDirectory(const QString& directory);
    QString getPluginDirectory() const;
    
    // Statistics and monitoring
    QVariantMap getPluginStatistics(const QString& pluginId) const;
    QVariantMap getSystemStatistics() const;
    void resetStatistics();
    
signals:
    void pluginLoaded(const QString& pluginId);
    void pluginUnloaded(const QString& pluginId);
    void pluginError(const QString& pluginId, const QString& error);
    void pluginExecutionStarted(const QString& pluginId, const QString& operation);
    void pluginExecutionFinished(const QString& pluginId, const QString& operation, bool success);
    
private slots:
    void onFileChanged(const QString& path);
    void onDirectoryChanged(const QString& path);
    
private:
    explicit PluginManager(QObject* parent = nullptr);
    ~PluginManager();
    
    static PluginManager* m_instance;
    static QMutex m_instanceMutex;
    
    // Core components
    std::unique_ptr<PluginRegistry> m_registry;
    mutable QMutex m_pluginsMutex;
    QMap<QString, std::shared_ptr<IPlugin>> m_loadedPlugins;
    QMap<QString, std::unique_ptr<QLibrary>> m_pluginLibraries;
    QMap<QString, std::unique_ptr<PluginSandbox>> m_sandboxes;
    
    // Configuration
    QString m_pluginDirectory;
    QString m_pythonPluginDirectory;
    QString m_configDirectory;
    QMap<QString, QJsonObject> m_pluginConfigs;
    
    // Hot reload
    bool m_hotReloadEnabled;
    std::unique_ptr<QFileSystemWatcher> m_fileWatcher;
    
    // Statistics
    mutable QMutex m_statsMutex;
    QMap<QString, QVariantMap> m_pluginStats;
    QVariantMap m_systemStats;
    
    // Helper methods
    bool loadCppPlugin(const PluginMetadata& metadata);
    bool loadPythonPlugin(const PluginMetadata& metadata);
    void unloadCppPlugin(const QString& pluginId);
    void unloadPythonPlugin(const QString& pluginId);
    
    PluginMetadata scanPluginFile(const QString& filePath);
    PluginMetadata scanPythonPlugin(const QString& filePath);
    
    void updatePluginStatistics(const QString& pluginId, const QString& operation, 
                               int executionTimeMs, bool success);
    
    QString getDefaultPluginDirectory() const;
    QString getDefaultConfigDirectory() const;
    
    // Dependency management
    bool resolveDependencies(const QString& pluginId);
    void checkAndUnloadConflicts(const QString& pluginId);
};

// Plugin creation macros for C++ plugins
#define PHOENIXDRS_PLUGIN_INTERFACE_IID "org.phoenixdrs.IPlugin"
Q_DECLARE_INTERFACE(IPlugin, PHOENIXDRS_PLUGIN_INTERFACE_IID)

#define PHOENIXDRS_PLUGIN_METADATA(className, jsonData) \
    class className##Factory : public QObject { \
        Q_OBJECT \
        Q_PLUGIN_METADATA(IID PHOENIXDRS_PLUGIN_INTERFACE_IID META_DATA jsonData) \
        Q_INTERFACES(IPlugin) \
    public: \
        IPlugin* create() override { return new className; } \
    };

// Convenience macros
#define PHOENIXDRS_DECLARE_PLUGIN(className) \
    PHOENIXDRS_PLUGIN_METADATA(className, \
        "{ \"name\": \"" #className "\", \"type\": \"custom\" }")

#endif // PLUGIN_MANAGER_H