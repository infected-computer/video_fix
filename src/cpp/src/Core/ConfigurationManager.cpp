/*
 * PhoenixDRS Professional - Enterprise Configuration Management Implementation
 * מימוש ניהול הגדרות ארגוני - PhoenixDRS מקצועי
 */

#include "../include/Core/ConfigurationManager.h"
#include "../include/Core/ErrorHandling.h"
#include "../include/ForensicLogger.h"

#include <QtCore/QCoreApplication>
#include <QtCore/QStandardPaths>
#include <QtCore/QDir>
#include <QtCore/QFileInfo>
#include <QtCore/QJsonDocument>
#include <QtCore/QJsonObject>
#include <QtCore/QJsonArray>
#include <QtCore/QMutexLocker>
#include <QtCore/QTimer>
#include <QtCore/QFileSystemWatcher>
#include <QtCore/QCryptographicHash>
#include <QtCore/QSettings>

#include <algorithm>

namespace PhoenixDRS {
namespace Core {

// ConfigurationValue Implementation
ConfigurationValue::ConfigurationValue()
    : m_type(Type::Null)
{
}

ConfigurationValue::ConfigurationValue(bool value)
    : m_value(value), m_type(Type::Bool)
{
}

ConfigurationValue::ConfigurationValue(int value)
    : m_value(value), m_type(Type::Int)
{
}

ConfigurationValue::ConfigurationValue(double value)
    : m_value(value), m_type(Type::Double)
{
}

ConfigurationValue::ConfigurationValue(const QString& value)
    : m_value(value), m_type(Type::String)
{
}

ConfigurationValue::ConfigurationValue(const QStringList& value)
    : m_value(value), m_type(Type::StringList)
{
}

ConfigurationValue::ConfigurationValue(const QJsonObject& value)
    : m_value(value), m_type(Type::Object)
{
}

ConfigurationValue::ConfigurationValue(const QJsonArray& value)
    : m_value(value), m_type(Type::Array)
{
}

bool ConfigurationValue::toBool() const
{
    switch (m_type) {
        case Type::Bool: return m_value.toBool();
        case Type::Int: return m_value.toInt() != 0;
        case Type::Double: return m_value.toDouble() != 0.0;
        case Type::String: return !m_value.toString().isEmpty();
        default: return false;
    }
}

int ConfigurationValue::toInt() const
{
    switch (m_type) {
        case Type::Bool: return m_value.toBool() ? 1 : 0;
        case Type::Int: return m_value.toInt();
        case Type::Double: return static_cast<int>(m_value.toDouble());
        case Type::String: return m_value.toString().toInt();
        default: return 0;
    }
}

double ConfigurationValue::toDouble() const
{
    switch (m_type) {
        case Type::Bool: return m_value.toBool() ? 1.0 : 0.0;
        case Type::Int: return static_cast<double>(m_value.toInt());
        case Type::Double: return m_value.toDouble();
        case Type::String: return m_value.toString().toDouble();
        default: return 0.0;
    }
}

QString ConfigurationValue::toString() const
{
    switch (m_type) {
        case Type::Bool: return m_value.toBool() ? "true" : "false";
        case Type::Int: return QString::number(m_value.toInt());
        case Type::Double: return QString::number(m_value.toDouble());
        case Type::String: return m_value.toString();
        case Type::StringList: return m_value.toStringList().join(", ");
        default: return QString();
    }
}

QStringList ConfigurationValue::toStringList() const
{
    if (m_type == Type::StringList) {
        return m_value.toStringList();
    } else if (m_type == Type::String) {
        return QStringList() << m_value.toString();
    }
    return QStringList();
}

QJsonObject ConfigurationValue::toJsonObject() const
{
    if (m_type == Type::Object) {
        return m_value.toJsonObject();
    }
    return QJsonObject();
}

QJsonArray ConfigurationValue::toJsonArray() const
{
    if (m_type == Type::Array) {
        return m_value.toJsonArray();
    }
    return QJsonArray();
}

QJsonValue ConfigurationValue::toJsonValue() const
{
    switch (m_type) {
        case Type::Bool: return m_value.toBool();
        case Type::Int: return m_value.toInt();
        case Type::Double: return m_value.toDouble();
        case Type::String: return m_value.toString();
        case Type::StringList: return QJsonArray::fromStringList(m_value.toStringList());
        case Type::Object: return m_value.toJsonObject();
        case Type::Array: return m_value.toJsonArray();
        default: return QJsonValue();
    }
}

// ConfigurationSection Implementation
class ConfigurationSectionPrivate
{
public:
    ConfigurationSectionPrivate(const QString& name) : name(name) {}
    
    QString name;
    QMap<QString, ConfigurationValue> values;
    QMap<QString, std::unique_ptr<ConfigurationSection>> subsections;
    mutable QMutex mutex;
};

ConfigurationSection::ConfigurationSection(const QString& name)
    : d(std::make_unique<ConfigurationSectionPrivate>(name))
{
}

ConfigurationSection::~ConfigurationSection() = default;

void ConfigurationSection::setValue(const QString& key, const ConfigurationValue& value)
{
    QMutexLocker locker(&d->mutex);
    d->values[key] = value;
}

ConfigurationValue ConfigurationSection::getValue(const QString& key, const ConfigurationValue& defaultValue) const
{
    QMutexLocker locker(&d->mutex);
    return d->values.value(key, defaultValue);
}

bool ConfigurationSection::hasValue(const QString& key) const
{
    QMutexLocker locker(&d->mutex);
    return d->values.contains(key);
}

void ConfigurationSection::removeValue(const QString& key)
{
    QMutexLocker locker(&d->mutex);
    d->values.remove(key);
}

QStringList ConfigurationSection::getKeys() const
{
    QMutexLocker locker(&d->mutex);
    return d->values.keys();
}

ConfigurationSection* ConfigurationSection::getSection(const QString& name)
{
    QMutexLocker locker(&d->mutex);
    
    auto it = d->subsections.find(name);
    if (it == d->subsections.end()) {
        d->subsections[name] = std::make_unique<ConfigurationSection>(name);
    }
    
    return d->subsections[name].get();
}

const ConfigurationSection* ConfigurationSection::getSection(const QString& name) const
{
    QMutexLocker locker(&d->mutex);
    
    auto it = d->subsections.find(name);
    if (it != d->subsections.end()) {
        return it->second.get();
    }
    
    return nullptr;
}

QStringList ConfigurationSection::getSectionNames() const
{
    QMutexLocker locker(&d->mutex);
    QStringList names;
    for (const auto& [name, section] : d->subsections) {
        names.append(name);
    }
    return names;
}

void ConfigurationSection::removeSection(const QString& name)
{
    QMutexLocker locker(&d->mutex);
    d->subsections.erase(name);
}

void ConfigurationSection::clear()
{
    QMutexLocker locker(&d->mutex);
    d->values.clear();
    d->subsections.clear();
}

QJsonObject ConfigurationSection::toJson() const
{
    QMutexLocker locker(&d->mutex);
    
    QJsonObject obj;
    
    // Add values
    for (auto it = d->values.begin(); it != d->values.end(); ++it) {
        obj[it.key()] = it.value().toJsonValue();
    }
    
    // Add subsections
    for (const auto& [name, section] : d->subsections) {
        obj[name] = section->toJson();
    }
    
    return obj;
}

void ConfigurationSection::fromJson(const QJsonObject& obj)
{
    QMutexLocker locker(&d->mutex);
    
    for (auto it = obj.begin(); it != obj.end(); ++it) {
        const QString& key = it.key();
        const QJsonValue& value = it.value();
        
        if (value.isObject()) {
            // It's a subsection
            auto section = std::make_unique<ConfigurationSection>(key);
            section->fromJson(value.toObject());
            d->subsections[key] = std::move(section);
        } else {
            // It's a value
            ConfigurationValue configValue;
            
            if (value.isBool()) {
                configValue = ConfigurationValue(value.toBool());
            } else if (value.isDouble()) {
                configValue = ConfigurationValue(value.toDouble());
            } else if (value.isString()) {
                configValue = ConfigurationValue(value.toString());
            } else if (value.isArray()) {
                configValue = ConfigurationValue(value.toArray());
            }
            
            d->values[key] = configValue;
        }
    }
}

QString ConfigurationSection::getName() const
{
    return d->name;
}

// ConfigurationManager Implementation
class ConfigurationManagerPrivate
{
public:
    ConfigurationManagerPrivate()
        : rootSection("root")
        , autoSaveEnabled(true)
        , encryptionEnabled(false)
        , validationEnabled(true)
        , isDirty(false)
    {
        setupDefaultConfiguration();
        setupAutoSave();
        setupFileWatching();
    }
    
    ~ConfigurationManagerPrivate()
    {
        if (autoSaveTimer) {
            autoSaveTimer->stop();
        }
    }
    
    void setupDefaultConfiguration()
    {
        // Application settings
        auto* appSection = rootSection.getSection("application");
        appSection->setValue("name", ConfigurationValue("PhoenixDRS Professional"));
        appSection->setValue("version", ConfigurationValue("2.0.0"));
        appSection->setValue("organization", ConfigurationValue("PhoenixDRS"));
        
        // Forensics settings
        auto* forensicsSection = rootSection.getSection("forensics");
        forensicsSection->setValue("default_temp_directory", 
                                 ConfigurationValue(QStandardPaths::writableLocation(QStandardPaths::TempLocation)));
        forensicsSection->setValue("max_parallel_operations", ConfigurationValue(4));
        forensicsSection->setValue("enable_checksum_verification", ConfigurationValue(true));
        forensicsSection->setValue("enable_compression", ConfigurationValue(false));
        
        // Disk imaging settings
        auto* imagingSection = forensicsSection->getSection("disk_imaging");
        imagingSection->setValue("default_sector_size", ConfigurationValue(512));
        imagingSection->setValue("enable_bad_sector_retry", ConfigurationValue(true));
        imagingSection->setValue("max_retry_count", ConfigurationValue(3));
        imagingSection->setValue("retry_delay_ms", ConfigurationValue(1000));
        
        // File carving settings
        auto* carvingSection = forensicsSection->getSection("file_carving");
        carvingSection->setValue("chunk_size_mb", ConfigurationValue(64));
        carvingSection->setValue("enable_parallel_carving", ConfigurationValue(true));
        carvingSection->setValue("max_file_size_mb", ConfigurationValue(100));
        carvingSection->setValue("min_file_size_bytes", ConfigurationValue(64));
        
        // Video rebuilding settings
        auto* videoSection = forensicsSection->getSection("video_rebuilding");
        videoSection->setValue("quality_threshold", ConfigurationValue(0.7));
        videoSection->setValue("enable_metadata_recovery", ConfigurationValue(true));
        videoSection->setValue("max_video_size_mb", ConfigurationValue(1024));
        
        // Security settings
        auto* securitySection = rootSection.getSection("security");
        securitySection->setValue("enable_audit_logging", ConfigurationValue(true));
        securitySection->setValue("hash_algorithm", ConfigurationValue("SHA256"));
        securitySection->setValue("enable_data_encryption", ConfigurationValue(false));
        
        // Performance settings
        auto* perfSection = rootSection.getSection("performance");
        perfSection->setValue("memory_pool_size_mb", ConfigurationValue(512));
        perfSection->setValue("enable_performance_monitoring", ConfigurationValue(true));
        perfSection->setValue("max_memory_usage_mb", ConfigurationValue(2048));
        perfSection->setValue("enable_auto_cleanup", ConfigurationValue(true));
        
        // UI settings
        auto* uiSection = rootSection.getSection("ui");
        uiSection->setValue("theme", ConfigurationValue("dark"));
        uiSection->setValue("language", ConfigurationValue("en"));
        uiSection->setValue("enable_progress_notifications", ConfigurationValue(true));
        uiSection->setValue("auto_save_interval_minutes", ConfigurationValue(5));
        
        // Logging settings
        auto* loggingSection = rootSection.getSection("logging");
        loggingSection->setValue("log_level", ConfigurationValue("INFO"));
        loggingSection->setValue("enable_file_logging", ConfigurationValue(true));
        loggingSection->setValue("enable_console_logging", ConfigurationValue(true));
        loggingSection->setValue("max_log_file_size_mb", ConfigurationValue(50));
        loggingSection->setValue("max_log_files", ConfigurationValue(10));
    }
    
    void setupAutoSave()
    {
        if (autoSaveEnabled) {
            autoSaveTimer = std::make_unique<QTimer>();
            QObject::connect(autoSaveTimer.get(), &QTimer::timeout, [this]() {
                if (isDirty && !configFile.isEmpty()) {
                    saveToFile(configFile);
                }
            });
            autoSaveTimer->start(300000); // Auto-save every 5 minutes
        }
    }
    
    void setupFileWatching()
    {
        fileWatcher = std::make_unique<QFileSystemWatcher>();
        QObject::connect(fileWatcher.get(), &QFileSystemWatcher::fileChanged, 
                        [this](const QString& path) {
            if (path == configFile) {
                reloadFromFile();
            }
        });
    }
    
    bool saveToFile(const QString& filePath)
    {
        try {
            QFileInfo fileInfo(filePath);
            QDir().mkpath(fileInfo.absoluteDir().path());
            
            QJsonObject rootObj = rootSection.toJson();
            QJsonDocument doc(rootObj);
            
            QByteArray data = doc.toJson(QJsonDocument::Indented);
            
            if (encryptionEnabled) {
                data = encryptData(data);
            }
            
            QFile file(filePath);
            if (!file.open(QIODevice::WriteOnly)) {
                throw PhoenixException(ErrorCode::FileAccessError,
                                     QString("Cannot open configuration file for writing: %1").arg(filePath),
                                     "ConfigurationManager");
            }
            
            qint64 written = file.write(data);
            if (written != data.size()) {
                throw PhoenixException(ErrorCode::FileAccessError,
                                     "Failed to write complete configuration data",
                                     "ConfigurationManager");
            }
            
            file.close();
            isDirty = false;
            
            // Update file watcher
            if (!fileWatcher->files().contains(filePath)) {
                fileWatcher->addPath(filePath);
            }
            
            if (ForensicLogger::instance()) {
                ForensicLogger::instance()->logInfo(
                    QString("Configuration saved to: %1").arg(filePath));
            }
            
            return true;
            
        } catch (const PhoenixException& e) {
            if (ForensicLogger::instance()) {
                ForensicLogger::instance()->logError(
                    QString("Failed to save configuration: %1").arg(e.message()));
            }
            return false;
        }
    }
    
    bool loadFromFile(const QString& filePath)
    {
        try {
            QFile file(filePath);
            if (!file.exists()) {
                return false; // Not an error, just no config file yet
            }
            
            if (!file.open(QIODevice::ReadOnly)) {
                throw PhoenixException(ErrorCode::FileAccessError,
                                     QString("Cannot open configuration file for reading: %1").arg(filePath),
                                     "ConfigurationManager");
            }
            
            QByteArray data = file.readAll();
            file.close();
            
            if (encryptionEnabled) {
                data = decryptData(data);
            }
            
            QJsonParseError error;
            QJsonDocument doc = QJsonDocument::fromJson(data, &error);
            if (doc.isNull()) {
                throw PhoenixException(ErrorCode::ConfigurationError,
                                     QString("Invalid JSON in configuration file: %1").arg(error.errorString()),
                                     "ConfigurationManager");
            }
            
            rootSection.fromJson(doc.object());
            
            if (validationEnabled && !validateConfiguration()) {
                throw PhoenixException(ErrorCode::ValidationError,
                                     "Configuration validation failed",
                                     "ConfigurationManager");
            }
            
            isDirty = false;
            
            // Update file watcher
            if (!fileWatcher->files().contains(filePath)) {
                fileWatcher->addPath(filePath);
            }
            
            if (ForensicLogger::instance()) {
                ForensicLogger::instance()->logInfo(
                    QString("Configuration loaded from: %1").arg(filePath));
            }
            
            return true;
            
        } catch (const PhoenixException& e) {
            if (ForensicLogger::instance()) {
                ForensicLogger::instance()->logError(
                    QString("Failed to load configuration: %1").arg(e.message()));
            }
            return false;
        }
    }
    
    void reloadFromFile()
    {
        if (!configFile.isEmpty()) {
            loadFromFile(configFile);
        }
    }
    
    bool validateConfiguration()
    {
        // Basic validation rules
        validationErrors.clear();
        
        // Check required sections exist
        QStringList requiredSections = {"application", "forensics", "security", "performance"};
        for (const QString& section : requiredSections) {
            if (!rootSection.getSection(section)) {
                validationErrors.append(QString("Missing required section: %1").arg(section));
            }
        }
        
        // Validate specific values
        auto* perfSection = rootSection.getSection("performance");
        if (perfSection) {
            int maxMemory = perfSection->getValue("max_memory_usage_mb", ConfigurationValue(2048)).toInt();
            if (maxMemory < 256) {
                validationErrors.append("max_memory_usage_mb must be at least 256 MB");
            }
        }
        
        auto* forensicsSection = rootSection.getSection("forensics");
        if (forensicsSection) {
            int maxParallel = forensicsSection->getValue("max_parallel_operations", ConfigurationValue(4)).toInt();
            if (maxParallel < 1 || maxParallel > 32) {
                validationErrors.append("max_parallel_operations must be between 1 and 32");
            }
        }
        
        return validationErrors.isEmpty();
    }
    
    QByteArray encryptData(const QByteArray& data)
    {
        // Simple XOR encryption for demonstration - in production use proper encryption
        QByteArray encrypted = data;
        const char key = 0x42;
        for (int i = 0; i < encrypted.size(); ++i) {
            encrypted[i] = encrypted[i] ^ key;
        }
        return encrypted.toBase64();
    }
    
    QByteArray decryptData(const QByteArray& encryptedData)
    {
        // Reverse of encryptData
        QByteArray data = QByteArray::fromBase64(encryptedData);
        const char key = 0x42;
        for (int i = 0; i < data.size(); ++i) {
            data[i] = data[i] ^ key;
        }
        return data;
    }
    
    ConfigurationSection rootSection;
    QString configFile;
    
    bool autoSaveEnabled;
    bool encryptionEnabled;
    bool validationEnabled;
    bool isDirty;
    
    std::unique_ptr<QTimer> autoSaveTimer;
    std::unique_ptr<QFileSystemWatcher> fileWatcher;
    
    QStringList validationErrors;
    mutable QMutex configMutex;
};

ConfigurationManager& ConfigurationManager::instance()
{
    static ConfigurationManager instance;
    return instance;
}

ConfigurationManager::ConfigurationManager()
    : d(std::make_unique<ConfigurationManagerPrivate>())
{
}

ConfigurationManager::~ConfigurationManager() = default;

bool ConfigurationManager::loadFromFile(const QString& filePath)
{
    QMutexLocker locker(&d->configMutex);
    d->configFile = filePath;
    return d->loadFromFile(filePath);
}

bool ConfigurationManager::saveToFile(const QString& filePath)
{
    QMutexLocker locker(&d->configMutex);
    if (!filePath.isEmpty()) {
        d->configFile = filePath;
    }
    return d->saveToFile(d->configFile);
}

bool ConfigurationManager::save()
{
    return saveToFile(QString());
}

void ConfigurationManager::setValue(const QString& path, const ConfigurationValue& value)
{
    QMutexLocker locker(&d->configMutex);
    
    QStringList pathParts = path.split('.', Qt::SkipEmptyParts);
    if (pathParts.isEmpty()) {
        return;
    }
    
    ConfigurationSection* section = &d->rootSection;
    
    // Navigate to the correct section
    for (int i = 0; i < pathParts.size() - 1; ++i) {
        section = section->getSection(pathParts[i]);
    }
    
    // Set the value
    section->setValue(pathParts.last(), value);
    d->isDirty = true;
}

ConfigurationValue ConfigurationManager::getValue(const QString& path, const ConfigurationValue& defaultValue) const
{
    QMutexLocker locker(&d->configMutex);
    
    QStringList pathParts = path.split('.', Qt::SkipEmptyParts);
    if (pathParts.isEmpty()) {
        return defaultValue;
    }
    
    const ConfigurationSection* section = &d->rootSection;
    
    // Navigate to the correct section
    for (int i = 0; i < pathParts.size() - 1; ++i) {
        section = section->getSection(pathParts[i]);
        if (!section) {
            return defaultValue;
        }
    }
    
    return section->getValue(pathParts.last(), defaultValue);
}

ConfigurationSection* ConfigurationManager::getSection(const QString& path)
{
    QMutexLocker locker(&d->configMutex);
    
    if (path.isEmpty()) {
        return &d->rootSection;
    }
    
    QStringList pathParts = path.split('.', Qt::SkipEmptyParts);
    ConfigurationSection* section = &d->rootSection;
    
    for (const QString& part : pathParts) {
        section = section->getSection(part);
    }
    
    return section;
}

const ConfigurationSection* ConfigurationManager::getSection(const QString& path) const
{
    QMutexLocker locker(&d->configMutex);
    
    if (path.isEmpty()) {
        return &d->rootSection;
    }
    
    QStringList pathParts = path.split('.', Qt::SkipEmptyParts);
    const ConfigurationSection* section = &d->rootSection;
    
    for (const QString& part : pathParts) {
        section = section->getSection(part);
        if (!section) {
            return nullptr;
        }
    }
    
    return section;
}

void ConfigurationManager::setAutoSaveEnabled(bool enabled)
{
    QMutexLocker locker(&d->configMutex);
    d->autoSaveEnabled = enabled;
    d->setupAutoSave();
}

void ConfigurationManager::setEncryptionEnabled(bool enabled)
{
    QMutexLocker locker(&d->configMutex);
    d->encryptionEnabled = enabled;
}

void ConfigurationManager::setValidationEnabled(bool enabled)
{
    QMutexLocker locker(&d->configMutex);
    d->validationEnabled = enabled;
}

bool ConfigurationManager::validate() const
{
    QMutexLocker locker(&d->configMutex);
    return d->validateConfiguration();
}

QStringList ConfigurationManager::getValidationErrors() const
{
    QMutexLocker locker(&d->configMutex);
    return d->validationErrors;
}

void ConfigurationManager::resetToDefaults()
{
    QMutexLocker locker(&d->configMutex);
    d->rootSection.clear();
    d->setupDefaultConfiguration();
    d->isDirty = true;
}

QString ConfigurationManager::getDefaultConfigPath()
{
    QString appDataPath = QStandardPaths::writableLocation(QStandardPaths::AppDataLocation);
    return QDir(appDataPath).filePath("config.json");
}

QJsonObject ConfigurationManager::exportToJson() const
{
    QMutexLocker locker(&d->configMutex);
    return d->rootSection.toJson();
}

void ConfigurationManager::importFromJson(const QJsonObject& json)
{
    QMutexLocker locker(&d->configMutex);
    d->rootSection.fromJson(json);
    d->isDirty = true;
}

} // namespace Core
} // namespace PhoenixDRS