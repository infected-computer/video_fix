#include "Core/PluginManager.h"
#include "Core/ConfigurationManager.h"
#include "Core/ErrorHandling.h"
#include "Core/MemoryManager.h"

#include <QCoreApplication>
#include <QDir>
#include <QJsonDocument>
#include <QJsonObject>
#include <QJsonArray>
#include <QLibrary>
#include <QPluginLoader>
#include <QFileSystemWatcher>
#include <QTimer>
#include <QThread>
#include <QProcess>
#include <QStandardPaths>
#include <QLoggingCategory>
#include <QtConcurrent>

Q_LOGGING_CATEGORY(pluginManager, "phoenixdrs.plugin.manager")

// Static members
PluginManager* PluginManager::m_instance = nullptr;
QMutex PluginManager::m_instanceMutex;

//==============================================================================
// PluginMetadata Implementation
//==============================================================================

PluginMetadata::PluginMetadata(const QJsonObject& json) {
    id = json["id"].toString();
    name = json["name"].toString();
    version = json["version"].toString();
    author = json["author"].toString();
    description = json["description"].toString();
    
    QString typeStr = json["type"].toString();
    if (typeStr == "analysis") type = PluginType::Analysis;
    else if (typeStr == "export") type = PluginType::Export;
    else if (typeStr == "signature") type = PluginType::Signature;
    else if (typeStr == "storage") type = PluginType::Storage;
    else if (typeStr == "ai") type = PluginType::AI;
    else if (typeStr == "codec") type = PluginType::Codec;
    else if (typeStr == "network") type = PluginType::Network;
    else if (typeStr == "forensic") type = PluginType::Forensic;
    else if (typeStr == "visualization") type = PluginType::Visualization;
    else type = PluginType::Custom;
    
    filePath = json["filePath"].toString();
    pythonModule = json["pythonModule"].toString();
    configuration = json["configuration"].toObject();
    
    QJsonObject capsObj = json["capabilities"].toObject();
    capabilities.requiresGPU = capsObj["requiresGPU"].toBool();
    capabilities.requiresNetwork = capsObj["requiresNetwork"].toBool();
    capabilities.requiresFileSystem = capsObj["requiresFileSystem"].toBool();
    capabilities.requiresHardwareAccess = capsObj["requiresHardwareAccess"].toBool();
    capabilities.isThreadSafe = capsObj["isThreadSafe"].toBool(true);
    capabilities.supportsHotReload = capsObj["supportsHotReload"].toBool();
    capabilities.maxInstances = capsObj["maxInstances"].toInt(1);
    capabilities.memoryLimitMB = capsObj["memoryLimitMB"].toInt(512);
    capabilities.executionTimeoutSec = capsObj["executionTimeoutSec"].toInt(300);
    
    QJsonArray depArray = capsObj["dependencies"].toArray();
    for (const auto& dep : depArray) {
        capabilities.dependencies.append(dep.toString());
    }
    
    QJsonArray conflictArray = capsObj["conflictsWith"].toArray();
    for (const auto& conflict : conflictArray) {
        capabilities.conflictsWith.append(conflict.toString());
    }
    
    loadTime = QDateTime::fromString(json["loadTime"].toString(), Qt::ISODate);
    lastUsed = QDateTime::fromString(json["lastUsed"].toString(), Qt::ISODate);
    isActive = json["isActive"].toBool();
    isLoaded = json["isLoaded"].toBool();
    usageCount = json["usageCount"].toInt();
}

QJsonObject PluginMetadata::toJson() const {
    QJsonObject json;
    json["id"] = id;
    json["name"] = name;
    json["version"] = version;
    json["author"] = author;
    json["description"] = description;
    
    QString typeStr;
    switch (type) {
        case PluginType::Analysis: typeStr = "analysis"; break;
        case PluginType::Export: typeStr = "export"; break;
        case PluginType::Signature: typeStr = "signature"; break;
        case PluginType::Storage: typeStr = "storage"; break;
        case PluginType::AI: typeStr = "ai"; break;
        case PluginType::Codec: typeStr = "codec"; break;
        case PluginType::Network: typeStr = "network"; break;
        case PluginType::Forensic: typeStr = "forensic"; break;
        case PluginType::Visualization: typeStr = "visualization"; break;
        default: typeStr = "custom"; break;
    }
    json["type"] = typeStr;
    
    json["filePath"] = filePath;
    json["pythonModule"] = pythonModule;
    json["configuration"] = configuration;
    
    QJsonObject capsObj;
    capsObj["requiresGPU"] = capabilities.requiresGPU;
    capsObj["requiresNetwork"] = capabilities.requiresNetwork;
    capsObj["requiresFileSystem"] = capabilities.requiresFileSystem;
    capsObj["requiresHardwareAccess"] = capabilities.requiresHardwareAccess;
    capsObj["isThreadSafe"] = capabilities.isThreadSafe;
    capsObj["supportsHotReload"] = capabilities.supportsHotReload;
    capsObj["maxInstances"] = capabilities.maxInstances;
    capsObj["memoryLimitMB"] = capabilities.memoryLimitMB;
    capsObj["executionTimeoutSec"] = capabilities.executionTimeoutSec;
    
    QJsonArray depArray;
    for (const QString& dep : capabilities.dependencies) {
        depArray.append(dep);
    }
    capsObj["dependencies"] = depArray;
    
    QJsonArray conflictArray;
    for (const QString& conflict : capabilities.conflictsWith) {
        conflictArray.append(conflict);
    }
    capsObj["conflictsWith"] = conflictArray;
    
    json["capabilities"] = capsObj;
    json["loadTime"] = loadTime.toString(Qt::ISODate);
    json["lastUsed"] = lastUsed.toString(Qt::ISODate);
    json["isActive"] = isActive;
    json["isLoaded"] = isLoaded;
    json["usageCount"] = usageCount;
    
    return json;
}

PluginMetadata PluginMetadata::fromJson(const QJsonObject& json) {
    return PluginMetadata(json);
}

bool PluginMetadata::isValid() const {
    return !id.isEmpty() && !name.isEmpty() && !version.isEmpty() && 
           (!filePath.isEmpty() || !pythonModule.isEmpty());
}

QString PluginMetadata::getUniqueId() const {
    return QString("%1_%2").arg(id, version);
}

//==============================================================================
// PluginRegistry Implementation
//==============================================================================

PluginRegistry::PluginRegistry(QObject* parent)
    : QObject(parent), m_registryPath(getDefaultRegistryPath()) {
    loadRegistry();
}

PluginRegistry::~PluginRegistry() {
    saveRegistry();
}

void PluginRegistry::registerPlugin(const PluginMetadata& metadata) {
    if (!metadata.isValid()) {
        qCWarning(pluginManager) << "Cannot register invalid plugin:" << metadata.id;
        return;
    }
    
    QMutexLocker locker(&m_mutex);
    QString uniqueId = metadata.getUniqueId();
    m_plugins[uniqueId] = metadata;
    
    emit pluginRegistered(uniqueId);
    emit registryChanged();
    
    qCInfo(pluginManager) << "Plugin registered:" << uniqueId;
}

void PluginRegistry::unregisterPlugin(const QString& pluginId) {
    QMutexLocker locker(&m_mutex);
    if (m_plugins.remove(pluginId) > 0) {
        emit pluginUnregistered(pluginId);
        emit registryChanged();
        qCInfo(pluginManager) << "Plugin unregistered:" << pluginId;
    }
}

bool PluginRegistry::isRegistered(const QString& pluginId) const {
    QMutexLocker locker(&m_mutex);
    return m_plugins.contains(pluginId);
}

QList<PluginMetadata> PluginRegistry::getAllPlugins() const {
    QMutexLocker locker(&m_mutex);
    return m_plugins.values();
}

QList<PluginMetadata> PluginRegistry::getPluginsByType(PluginType type) const {
    QMutexLocker locker(&m_mutex);
    QList<PluginMetadata> result;
    
    for (const auto& plugin : m_plugins.values()) {
        if (plugin.type == type) {
            result.append(plugin);
        }
    }
    
    return result;
}

PluginMetadata PluginRegistry::getPlugin(const QString& pluginId) const {
    QMutexLocker locker(&m_mutex);
    return m_plugins.value(pluginId);
}

QStringList PluginRegistry::getPluginIds() const {
    QMutexLocker locker(&m_mutex);
    return m_plugins.keys();
}

QStringList PluginRegistry::resolveDependencies(const QString& pluginId) const {
    QMutexLocker locker(&m_mutex);
    QStringList resolved;
    QStringList toResolve;
    
    if (!m_plugins.contains(pluginId)) {
        return resolved;
    }
    
    toResolve.append(pluginId);
    
    while (!toResolve.isEmpty()) {
        QString current = toResolve.takeFirst();
        if (resolved.contains(current)) {
            continue;
        }
        
        const PluginMetadata& metadata = m_plugins[current];
        resolved.append(current);
        
        for (const QString& dep : metadata.capabilities.dependencies) {
            if (!resolved.contains(dep) && !toResolve.contains(dep)) {
                if (m_plugins.contains(dep)) {
                    toResolve.append(dep);
                } else {
                    qCWarning(pluginManager) << "Missing dependency:" << dep << "for plugin:" << current;
                }
            }
        }
    }
    
    return resolved;
}

bool PluginRegistry::checkConflicts(const QString& pluginId) const {
    return !getConflicts(pluginId).isEmpty();
}

QStringList PluginRegistry::getConflicts(const QString& pluginId) const {
    QMutexLocker locker(&m_mutex);
    QStringList conflicts;
    
    if (!m_plugins.contains(pluginId)) {
        return conflicts;
    }
    
    const PluginMetadata& metadata = m_plugins[pluginId];
    
    for (const QString& conflict : metadata.capabilities.conflictsWith) {
        if (m_plugins.contains(conflict) && m_plugins[conflict].isActive) {
            conflicts.append(conflict);
        }
    }
    
    return conflicts;
}

bool PluginRegistry::saveRegistry(const QString& filePath) const {
    QString path = filePath.isEmpty() ? m_registryPath : filePath;
    
    QJsonObject rootObj;
    QJsonArray pluginArray;
    
    QMutexLocker locker(&m_mutex);
    for (const auto& plugin : m_plugins.values()) {
        pluginArray.append(plugin.toJson());
    }
    
    rootObj["plugins"] = pluginArray;
    rootObj["version"] = "2.0";
    rootObj["timestamp"] = QDateTime::currentDateTime().toString(Qt::ISODate);
    
    QJsonDocument doc(rootObj);
    
    QFile file(path);
    if (!file.open(QIODevice::WriteOnly)) {
        qCWarning(pluginManager) << "Cannot open registry file for writing:" << path;
        return false;
    }
    
    file.write(doc.toJson());
    qCInfo(pluginManager) << "Plugin registry saved to:" << path;
    return true;
}

bool PluginRegistry::loadRegistry(const QString& filePath) {
    QString path = filePath.isEmpty() ? m_registryPath : filePath;
    
    QFile file(path);
    if (!file.exists()) {
        qCInfo(pluginManager) << "Registry file does not exist, creating new:" << path;
        return true; // Not an error for first run
    }
    
    if (!file.open(QIODevice::ReadOnly)) {
        qCWarning(pluginManager) << "Cannot open registry file for reading:" << path;
        return false;
    }
    
    QJsonParseError error;
    QJsonDocument doc = QJsonDocument::fromJson(file.readAll(), &error);
    
    if (error.error != QJsonParseError::NoError) {
        qCWarning(pluginManager) << "JSON parse error in registry:" << error.errorString();
        return false;
    }
    
    QJsonObject rootObj = doc.object();
    QJsonArray pluginArray = rootObj["plugins"].toArray();
    
    QMutexLocker locker(&m_mutex);
    m_plugins.clear();
    
    for (const auto& value : pluginArray) {
        PluginMetadata metadata = PluginMetadata::fromJson(value.toObject());
        if (metadata.isValid()) {
            m_plugins[metadata.getUniqueId()] = metadata;
        }
    }
    
    qCInfo(pluginManager) << "Loaded" << m_plugins.size() << "plugins from registry:" << path;
    return true;
}

QString PluginRegistry::getDefaultRegistryPath() const {
    QString dataDir = QStandardPaths::writableLocation(QStandardPaths::AppDataLocation);
    return QDir(dataDir).absoluteFilePath("plugin_registry.json");
}

//==============================================================================
// PluginSandbox Implementation
//==============================================================================

PluginSandbox::PluginSandbox(const PluginCapabilities& capabilities, QObject* parent)
    : QObject(parent), m_capabilities(capabilities),
      m_memoryLimitMB(capabilities.memoryLimitMB),
      m_executionTimeoutSec(capabilities.executionTimeoutSec),
      m_networkAllowed(capabilities.requiresNetwork),
      m_currentMemoryMB(0), m_executionTimeSec(0) {
}

PluginSandbox::~PluginSandbox() {
    stopMonitoring();
}

void PluginSandbox::setMemoryLimit(int limitMB) {
    m_memoryLimitMB = limitMB;
}

void PluginSandbox::setExecutionTimeout(int timeoutSec) {
    m_executionTimeoutSec = timeoutSec;
}

void PluginSandbox::setFileSystemAccess(const QStringList& allowedPaths) {
    m_allowedPaths = allowedPaths;
}

void PluginSandbox::setNetworkAccess(bool allowed) {
    m_networkAllowed = allowed;
}

QVariant PluginSandbox::executeInSandbox(std::function<QVariant()> operation) {
    startMonitoring();
    
    try {
        QVariant result = operation();
        stopMonitoring();
        return result;
    } catch (const std::exception& e) {
        stopMonitoring();
        emit securityViolation(QString("Exception during execution: %1").arg(e.what()));
        return QVariant();
    }
}

bool PluginSandbox::isWithinLimits() const {
    return checkResourceLimits();
}

int PluginSandbox::getCurrentMemoryUsage() const {
    QMutexLocker locker(&m_monitorMutex);
    return m_currentMemoryMB;
}

int PluginSandbox::getExecutionTime() const {
    QMutexLocker locker(&m_monitorMutex);
    return m_executionTimeSec;
}

QStringList PluginSandbox::getAccessedFiles() const {
    QMutexLocker locker(&m_monitorMutex);
    return m_accessedFiles;
}

QStringList PluginSandbox::getNetworkConnections() const {
    QMutexLocker locker(&m_monitorMutex);
    return m_networkConnections;
}

void PluginSandbox::startMonitoring() {
    QMutexLocker locker(&m_monitorMutex);
    m_currentMemoryMB = 0;
    m_executionTimeSec = 0;
    m_accessedFiles.clear();
    m_networkConnections.clear();
}

void PluginSandbox::stopMonitoring() {
    // Implementation would stop resource monitoring
}

bool PluginSandbox::checkResourceLimits() {
    QMutexLocker locker(&m_monitorMutex);
    
    if (m_currentMemoryMB > m_memoryLimitMB) {
        emit resourceLimitExceeded("memory");
        return false;
    }
    
    if (m_executionTimeSec > m_executionTimeoutSec) {
        emit resourceLimitExceeded("execution_time");
        return false;
    }
    
    return true;
}

//==============================================================================
// PluginManager Implementation
//==============================================================================

PluginManager* PluginManager::instance() {
    QMutexLocker locker(&m_instanceMutex);
    if (!m_instance) {
        m_instance = new PluginManager();
    }
    return m_instance;
}

PluginManager::PluginManager(QObject* parent)
    : QObject(parent), m_registry(std::make_unique<PluginRegistry>(this)),
      m_pluginDirectory(getDefaultPluginDirectory()),
      m_pythonPluginDirectory(m_pluginDirectory + "/python"),
      m_configDirectory(getDefaultConfigDirectory()),
      m_hotReloadEnabled(false),
      m_fileWatcher(std::make_unique<QFileSystemWatcher>(this)) {
    
    // Create directories if they don't exist
    QDir().mkpath(m_pluginDirectory);
    QDir().mkpath(m_pythonPluginDirectory);
    QDir().mkpath(m_configDirectory);
    
    // Connect file watcher
    connect(m_fileWatcher.get(), &QFileSystemWatcher::fileChanged,
            this, &PluginManager::onFileChanged);
    connect(m_fileWatcher.get(), &QFileSystemWatcher::directoryChanged,
            this, &PluginManager::onDirectoryChanged);
    
    // Load plugin configurations
    loadPluginConfigs();
    
    qCInfo(pluginManager) << "PluginManager initialized";
    qCInfo(pluginManager) << "Plugin directory:" << m_pluginDirectory;
    qCInfo(pluginManager) << "Python plugin directory:" << m_pythonPluginDirectory;
    qCInfo(pluginManager) << "Config directory:" << m_configDirectory;
}

PluginManager::~PluginManager() {
    // Unload all plugins
    QStringList loadedPlugins = getLoadedPlugins();
    for (const QString& pluginId : loadedPlugins) {
        unloadPlugin(pluginId);
    }
    
    savePluginConfigs();
    qCInfo(pluginManager) << "PluginManager destroyed";
}

void PluginManager::scanForPlugins(const QString& directory) {
    QString scanDir = directory.isEmpty() ? m_pluginDirectory : directory;
    QDir dir(scanDir);
    
    if (!dir.exists()) {
        qCWarning(pluginManager) << "Plugin directory does not exist:" << scanDir;
        return;
    }
    
    qCInfo(pluginManager) << "Scanning for C++ plugins in:" << scanDir;
    
    // Scan for shared libraries
    QStringList filters;
#ifdef Q_OS_WIN
    filters << "*.dll";
#elif defined(Q_OS_MAC)
    filters << "*.dylib";
#else
    filters << "*.so";
#endif
    
    QFileInfoList files = dir.entryInfoList(filters, QDir::Files);
    for (const QFileInfo& fileInfo : files) {
        PluginMetadata metadata = scanPluginFile(fileInfo.absoluteFilePath());
        if (metadata.isValid()) {
            m_registry->registerPlugin(metadata);
        }
    }
    
    // Recursively scan subdirectories
    QFileInfoList subdirs = dir.entryInfoList(QDir::Dirs | QDir::NoDotAndDotDot);
    for (const QFileInfo& subdir : subdirs) {
        scanForPlugins(subdir.absoluteFilePath());
    }
}

void PluginManager::scanForPythonPlugins(const QString& directory) {
    QString scanDir = directory.isEmpty() ? m_pythonPluginDirectory : directory;
    QDir dir(scanDir);
    
    if (!dir.exists()) {
        qCWarning(pluginManager) << "Python plugin directory does not exist:" << scanDir;
        return;
    }
    
    qCInfo(pluginManager) << "Scanning for Python plugins in:" << scanDir;
    
    QFileInfoList files = dir.entryInfoList(QStringList() << "*.py", QDir::Files);
    for (const QFileInfo& fileInfo : files) {
        PluginMetadata metadata = scanPythonPlugin(fileInfo.absoluteFilePath());
        if (metadata.isValid()) {
            m_registry->registerPlugin(metadata);
        }
    }
    
    // Also scan for plugin packages (directories with __init__.py)
    QFileInfoList subdirs = dir.entryInfoList(QDir::Dirs | QDir::NoDotAndDotDot);
    for (const QFileInfo& subdir : subdirs) {
        QString initFile = QDir(subdir.absoluteFilePath()).absoluteFilePath("__init__.py");
        if (QFile::exists(initFile)) {
            PluginMetadata metadata = scanPythonPlugin(initFile);
            if (metadata.isValid()) {
                metadata.pythonModule = subdir.baseName();
                m_registry->registerPlugin(metadata);
            }
        }
    }
}

bool PluginManager::loadPlugin(const QString& pluginId) {
    qCInfo(pluginManager) << "Loading plugin:" << pluginId;
    
    if (isPluginLoaded(pluginId)) {
        qCWarning(pluginManager) << "Plugin already loaded:" << pluginId;
        return true;
    }
    
    PluginMetadata metadata = m_registry->getPlugin(pluginId);
    if (!metadata.isValid()) {
        qCWarning(pluginManager) << "Plugin not found in registry:" << pluginId;
        return false;
    }
    
    // Check dependencies
    if (!resolveDependencies(pluginId)) {
        qCWarning(pluginManager) << "Failed to resolve dependencies for plugin:" << pluginId;
        return false;
    }
    
    // Check conflicts
    checkAndUnloadConflicts(pluginId);
    
    // Load the plugin
    bool success = false;
    if (!metadata.pythonModule.isEmpty()) {
        success = loadPythonPlugin(metadata);
    } else {
        success = loadCppPlugin(metadata);
    }
    
    if (success) {
        metadata.isLoaded = true;
        metadata.isActive = true;
        metadata.loadTime = QDateTime::currentDateTime();
        m_registry->registerPlugin(metadata); // Update metadata
        
        emit pluginLoaded(pluginId);
        qCInfo(pluginManager) << "Plugin loaded successfully:" << pluginId;
    } else {
        emit pluginError(pluginId, "Failed to load plugin");
        qCWarning(pluginManager) << "Failed to load plugin:" << pluginId;
    }
    
    return success;
}

bool PluginManager::unloadPlugin(const QString& pluginId) {
    qCInfo(pluginManager) << "Unloading plugin:" << pluginId;
    
    if (!isPluginLoaded(pluginId)) {
        qCWarning(pluginManager) << "Plugin not loaded:" << pluginId;
        return true;
    }
    
    PluginMetadata metadata = m_registry->getPlugin(pluginId);
    
    // Unload based on type
    if (!metadata.pythonModule.isEmpty()) {
        unloadPythonPlugin(pluginId);
    } else {
        unloadCppPlugin(pluginId);
    }
    
    // Update metadata
    metadata.isLoaded = false;
    metadata.isActive = false;
    m_registry->registerPlugin(metadata);
    
    emit pluginUnloaded(pluginId);
    qCInfo(pluginManager) << "Plugin unloaded:" << pluginId;
    
    return true;
}

bool PluginManager::reloadPlugin(const QString& pluginId) {
    qCInfo(pluginManager) << "Reloading plugin:" << pluginId;
    
    if (isPluginLoaded(pluginId)) {
        if (!unloadPlugin(pluginId)) {
            return false;
        }
    }
    
    return loadPlugin(pluginId);
}

IPlugin* PluginManager::getPlugin(const QString& pluginId) {
    QMutexLocker locker(&m_pluginsMutex);
    auto it = m_loadedPlugins.find(pluginId);
    return (it != m_loadedPlugins.end()) ? it->second.get() : nullptr;
}

QList<IPlugin*> PluginManager::getPluginsByType(PluginType type) {
    QList<IPlugin*> result;
    QMutexLocker locker(&m_pluginsMutex);
    
    for (const auto& pair : m_loadedPlugins) {
        if (pair.second->getType() == type) {
            result.append(pair.second.get());
        }
    }
    
    return result;
}

bool PluginManager::isPluginLoaded(const QString& pluginId) const {
    QMutexLocker locker(&m_pluginsMutex);
    return m_loadedPlugins.contains(pluginId);
}

QStringList PluginManager::getLoadedPlugins() const {
    QMutexLocker locker(&m_pluginsMutex);
    return m_loadedPlugins.keys();
}

QVariant PluginManager::executePlugin(const QString& pluginId, const QString& operation, 
                                     const QVariantMap& parameters) {
    emit pluginExecutionStarted(pluginId, operation);
    
    IPlugin* plugin = getPlugin(pluginId);
    if (!plugin) {
        emit pluginError(pluginId, "Plugin not found or not loaded");
        emit pluginExecutionFinished(pluginId, operation, false);
        return QVariant();
    }
    
    if (!plugin->supportsOperation(operation)) {
        emit pluginError(pluginId, QString("Operation not supported: %1").arg(operation));
        emit pluginExecutionFinished(pluginId, operation, false);
        return QVariant();
    }
    
    QElapsedTimer timer;
    timer.start();
    
    try {
        QVariant result = plugin->execute(operation, parameters);
        
        int executionTime = timer.elapsed();
        updatePluginStatistics(pluginId, operation, executionTime, true);
        
        emit pluginExecutionFinished(pluginId, operation, true);
        return result;
        
    } catch (const std::exception& e) {
        int executionTime = timer.elapsed();
        updatePluginStatistics(pluginId, operation, executionTime, false);
        
        emit pluginError(pluginId, QString("Execution error: %1").arg(e.what()));
        emit pluginExecutionFinished(pluginId, operation, false);
        return QVariant();
    }
}

QFuture<QVariant> PluginManager::executePluginAsync(const QString& pluginId, 
                                                   const QString& operation,
                                                   const QVariantMap& parameters) {
    return QtConcurrent::run([this, pluginId, operation, parameters]() {
        return executePlugin(pluginId, operation, parameters);
    });
}

// Implementation continues with remaining methods...
// Due to length constraints, I'll continue with the essential helper methods

QString PluginManager::getDefaultPluginDirectory() const {
    return QDir(QCoreApplication::applicationDirPath()).absoluteFilePath("plugins");
}

QString PluginManager::getDefaultConfigDirectory() const {
    QString dataDir = QStandardPaths::writableLocation(QStandardPaths::AppDataLocation);
    return QDir(dataDir).absoluteFilePath("plugin_configs");
}

void PluginManager::onFileChanged(const QString& path) {
    if (!m_hotReloadEnabled) return;
    
    qCInfo(pluginManager) << "Plugin file changed:" << path;
    
    // Find plugin by file path and reload
    QList<PluginMetadata> plugins = m_registry->getAllPlugins();
    for (const PluginMetadata& metadata : plugins) {
        if (metadata.filePath == path && metadata.capabilities.supportsHotReload) {
            QTimer::singleShot(100, [this, metadata]() {
                reloadPlugin(metadata.getUniqueId());
            });
            break;
        }
    }
}

void PluginManager::onDirectoryChanged(const QString& path) {
    if (!m_hotReloadEnabled) return;
    
    qCInfo(pluginManager) << "Plugin directory changed:" << path;
    
    // Rescan directory for new plugins
    QTimer::singleShot(500, [this, path]() {
        if (path.contains("python")) {
            scanForPythonPlugins(path);
        } else {
            scanForPlugins(path);
        }
    });
}

bool PluginManager::loadCppPlugin(const PluginMetadata& metadata) {
    // Implementation for loading C++ plugins via QPluginLoader
    QPluginLoader loader(metadata.filePath);
    QObject* pluginObj = loader.instance();
    
    if (!pluginObj) {
        qCWarning(pluginManager) << "Failed to load C++ plugin:" << loader.errorString();
        return false;
    }
    
    IPlugin* plugin = qobject_cast<IPlugin*>(pluginObj);
    if (!plugin) {
        qCWarning(pluginManager) << "Plugin does not implement IPlugin interface";
        return false;
    }
    
    // Initialize plugin
    QJsonObject config = getPluginConfig(metadata.getUniqueId());
    if (!plugin->initialize(config)) {
        qCWarning(pluginManager) << "Plugin initialization failed";
        return false;
    }
    
    // Store plugin
    QMutexLocker locker(&m_pluginsMutex);
    m_loadedPlugins[metadata.getUniqueId()] = std::shared_ptr<IPlugin>(plugin);
    m_pluginLibraries[metadata.getUniqueId()] = std::make_unique<QLibrary>(metadata.filePath);
    
    // Create sandbox if needed
    m_sandboxes[metadata.getUniqueId()] = std::make_unique<PluginSandbox>(metadata.capabilities, this);
    
    return true;
}

bool PluginManager::loadPythonPlugin(const PluginMetadata& metadata) {
    // Implementation for loading Python plugins would go here
    // This would involve Python C API or embedded Python interpreter
    qCWarning(pluginManager) << "Python plugin loading not yet implemented";
    return false;
}

void PluginManager::unloadCppPlugin(const QString& pluginId) {
    QMutexLocker locker(&m_pluginsMutex);
    
    auto pluginIt = m_loadedPlugins.find(pluginId);
    if (pluginIt != m_loadedPlugins.end()) {
        pluginIt->second->shutdown();
        m_loadedPlugins.erase(pluginIt);
    }
    
    m_pluginLibraries.erase(pluginId);
    m_sandboxes.erase(pluginId);
}

void PluginManager::unloadPythonPlugin(const QString& pluginId) {
    QMutexLocker locker(&m_pluginsMutex);
    m_loadedPlugins.erase(pluginId);
    m_sandboxes.erase(pluginId);
}

PluginMetadata PluginManager::scanPluginFile(const QString& filePath) {
    // Implementation would scan plugin metadata from file
    PluginMetadata metadata;
    metadata.filePath = filePath;
    metadata.id = QFileInfo(filePath).baseName();
    metadata.name = metadata.id;
    metadata.version = "1.0.0";
    metadata.type = PluginType::Custom;
    return metadata;
}

PluginMetadata PluginManager::scanPythonPlugin(const QString& filePath) {
    // Implementation would scan Python plugin metadata
    PluginMetadata metadata;
    metadata.filePath = filePath;
    metadata.pythonModule = QFileInfo(filePath).baseName();
    metadata.id = metadata.pythonModule;
    metadata.name = metadata.id;
    metadata.version = "1.0.0";
    metadata.type = PluginType::Custom;
    return metadata;
}

void PluginManager::updatePluginStatistics(const QString& pluginId, const QString& operation, 
                                          int executionTimeMs, bool success) {
    QMutexLocker locker(&m_statsMutex);
    
    QVariantMap& stats = m_pluginStats[pluginId];
    stats["totalExecutions"] = stats["totalExecutions"].toInt() + 1;
    stats["totalExecutionTime"] = stats["totalExecutionTime"].toInt() + executionTimeMs;
    
    if (success) {
        stats["successfulExecutions"] = stats["successfulExecutions"].toInt() + 1;
    } else {
        stats["failedExecutions"] = stats["failedExecutions"].toInt() + 1;
    }
    
    stats["lastOperation"] = operation;
    stats["lastExecutionTime"] = executionTimeMs;
    stats["lastExecution"] = QDateTime::currentDateTime().toString(Qt::ISODate);
}

bool PluginManager::resolveDependencies(const QString& pluginId) {
    QStringList dependencies = m_registry->resolveDependencies(pluginId);
    
    for (const QString& dep : dependencies) {
        if (dep != pluginId && !isPluginLoaded(dep)) {
            if (!loadPlugin(dep)) {
                return false;
            }
        }
    }
    
    return true;
}

void PluginManager::checkAndUnloadConflicts(const QString& pluginId) {
    QStringList conflicts = m_registry->getConflicts(pluginId);
    
    for (const QString& conflict : conflicts) {
        if (isPluginLoaded(conflict)) {
            qCInfo(pluginManager) << "Unloading conflicting plugin:" << conflict;
            unloadPlugin(conflict);
        }
    }
}

void PluginManager::setPluginConfig(const QString& pluginId, const QJsonObject& config) {
    m_pluginConfigs[pluginId] = config;
    
    // Update running plugin if loaded
    IPlugin* plugin = getPlugin(pluginId);
    if (plugin) {
        plugin->updateConfig(config);
    }
}

QJsonObject PluginManager::getPluginConfig(const QString& pluginId) const {
    return m_pluginConfigs.value(pluginId);
}

void PluginManager::savePluginConfigs() {
    QString configFile = QDir(m_configDirectory).absoluteFilePath("plugin_configs.json");
    
    QJsonObject rootObj;
    for (auto it = m_pluginConfigs.begin(); it != m_pluginConfigs.end(); ++it) {
        rootObj[it.key()] = it.value();
    }
    
    QJsonDocument doc(rootObj);
    
    QFile file(configFile);
    if (file.open(QIODevice::WriteOnly)) {
        file.write(doc.toJson());
        qCInfo(pluginManager) << "Plugin configurations saved";
    }
}

void PluginManager::loadPluginConfigs() {
    QString configFile = QDir(m_configDirectory).absoluteFilePath("plugin_configs.json");
    
    QFile file(configFile);
    if (!file.exists() || !file.open(QIODevice::ReadOnly)) {
        return;
    }
    
    QJsonParseError error;
    QJsonDocument doc = QJsonDocument::fromJson(file.readAll(), &error);
    
    if (error.error == QJsonParseError::NoError) {
        QJsonObject rootObj = doc.object();
        for (auto it = rootObj.begin(); it != rootObj.end(); ++it) {
            m_pluginConfigs[it.key()] = it.value().toObject();
        }
        qCInfo(pluginManager) << "Plugin configurations loaded";
    }
}