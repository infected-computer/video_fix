/*
 * PhoenixDRS Professional - Enterprise Configuration Management System
 * מערכת ניהול תצורה ברמה תעשייתית - PhoenixDRS מקצועי
 */

#pragma once

#include "ErrorHandling.h"
#include <QtCore/QObject>
#include <QtCore/QSettings>
#include <QtCore/QJsonObject>
#include <QtCore/QJsonDocument>
#include <QtCore/QVariant>
#include <QtCore/QStringList>
#include <QtCore/QMutex>
#include <QtCore/QReadWriteLock>
#include <QtCore/QFileSystemWatcher>
#include <QtCore/QTimer>
#include <QtCore/QDir>

#include <memory>
#include <functional>
#include <unordered_map>
#include <unordered_set>
#include <type_traits>

namespace PhoenixDRS {
namespace Core {

// Configuration value types
enum class ConfigType {
    Invalid,
    Bool,
    Int,
    UInt,
    Int64,
    UInt64,
    Double,
    String,
    StringList,
    ByteArray,
    Date,
    DateTime,
    Time,
    Url,
    Json,
    Variant
};

// Configuration scopes
enum class ConfigScope {
    System,        // System-wide configuration
    Application,   // Application-level configuration
    User,          // User-specific configuration
    Session,       // Session-specific configuration
    Temporary,     // Temporary configuration (not persisted)
    Remote,        // Remote configuration (from server)
    Override       // Override configuration (highest priority)
};

// Configuration storage backends
enum class StorageBackend {
    Registry,      // Windows Registry
    IniFile,       // INI file format
    JsonFile,      // JSON file format
    XmlFile,       // XML file format
    Database,      // SQL database
    Memory,        // In-memory only
    Remote,        // Remote server
    Encrypted      // Encrypted storage
};

// Configuration validation rules
enum class ValidationRule {
    None,
    Range,         // Numeric range validation
    Length,        // String length validation
    Regex,         // Regular expression validation
    Enum,          // Enumerated values validation
    File,          // File existence validation
    Directory,     // Directory existence validation
    Network,       // Network address validation
    Custom         // Custom validation function
};

// Forward declarations
class ConfigValue;
class ConfigValidator;
class ConfigurationProvider;
class ConfigurationWatcher;

/*
 * Configuration value with type safety and validation
 */
class PHOENIXDRS_EXPORT ConfigValue
{
public:
    ConfigValue();
    ConfigValue(const QVariant& value);
    ConfigValue(const ConfigValue& other);
    ConfigValue(ConfigValue&& other) noexcept;
    ~ConfigValue();
    
    ConfigValue& operator=(const ConfigValue& other);
    ConfigValue& operator=(ConfigValue&& other) noexcept;
    
    // Type checking
    ConfigType type() const;
    bool isValid() const;
    bool isNull() const;
    
    // Value access with type safety
    template<typename T>
    T value() const {
        if constexpr (std::is_same_v<T, bool>) {
            return m_value.toBool();
        } else if constexpr (std::is_same_v<T, int>) {
            return m_value.toInt();
        } else if constexpr (std::is_same_v<T, uint>) {
            return m_value.toUInt();
        } else if constexpr (std::is_same_v<T, qint64>) {
            return m_value.toLongLong();
        } else if constexpr (std::is_same_v<T, quint64>) {
            return m_value.toULongLong();
        } else if constexpr (std::is_same_v<T, double>) {
            return m_value.toDouble();
        } else if constexpr (std::is_same_v<T, QString>) {
            return m_value.toString();
        } else if constexpr (std::is_same_v<T, QStringList>) {
            return m_value.toStringList();
        } else if constexpr (std::is_same_v<T, QByteArray>) {
            return m_value.toByteArray();
        } else if constexpr (std::is_same_v<T, QDateTime>) {
            return m_value.toDateTime();
        } else if constexpr (std::is_same_v<T, QUrl>) {
            return m_value.toUrl();
        } else if constexpr (std::is_same_v<T, QJsonObject>) {
            return QJsonDocument::fromVariant(m_value).object();
        } else {
            return m_value.value<T>();
        }
    }
    
    template<typename T>
    T valueOr(const T& defaultValue) const {
        if (!isValid()) {
            return defaultValue;
        }
        try {
            return value<T>();
        } catch (...) {
            return defaultValue;
        }
    }
    
    // Convenient accessors
    bool toBool() const { return value<bool>(); }
    int toInt() const { return value<int>(); }
    uint toUInt() const { return value<uint>(); }
    qint64 toLongLong() const { return value<qint64>(); }
    quint64 toULongLong() const { return value<quint64>(); }
    double toDouble() const { return value<double>(); }
    QString toString() const { return value<QString>(); }
    QStringList toStringList() const { return value<QStringList>(); }
    QByteArray toByteArray() const { return value<QByteArray>(); }
    QDateTime toDateTime() const { return value<QDateTime>(); }
    QUrl toUrl() const { return value<QUrl>(); }
    QJsonObject toJsonObject() const { return value<QJsonObject>(); }
    
    // Validation
    void setValidator(std::shared_ptr<ConfigValidator> validator);
    std::shared_ptr<ConfigValidator> getValidator() const;
    Result<void> validate() const;
    
    // Metadata
    void setDescription(const QString& description);
    QString getDescription() const;
    void setDefaultValue(const QVariant& defaultValue);
    QVariant getDefaultValue() const;
    void setScope(ConfigScope scope);
    ConfigScope getScope() const;
    void setReadOnly(bool readOnly);
    bool isReadOnly() const;
    void setEncrypted(bool encrypted);
    bool isEncrypted() const;
    
    // Serialization
    QJsonObject toJson() const;
    static ConfigValue fromJson(const QJsonObject& json);
    QVariant toVariant() const { return m_value; }
    static ConfigValue fromVariant(const QVariant& variant);
    
    // Operators
    bool operator==(const ConfigValue& other) const;
    bool operator!=(const ConfigValue& other) const;

private:
    QVariant m_value;
    std::shared_ptr<ConfigValidator> m_validator;
    QString m_description;
    QVariant m_defaultValue;
    ConfigScope m_scope;
    bool m_readOnly;
    bool m_encrypted;
};

/*
 * Configuration validator interface
 */
class PHOENIXDRS_EXPORT ConfigValidator
{
public:
    virtual ~ConfigValidator() = default;
    virtual Result<void> validate(const ConfigValue& value) const = 0;
    virtual QString getDescription() const = 0;
    virtual ValidationRule getRule() const = 0;
};

// Concrete validator implementations
class PHOENIXDRS_EXPORT RangeValidator : public ConfigValidator
{
public:
    RangeValidator(const QVariant& min, const QVariant& max);
    Result<void> validate(const ConfigValue& value) const override;
    QString getDescription() const override;
    ValidationRule getRule() const override { return ValidationRule::Range; }

private:
    QVariant m_min, m_max;
};

class PHOENIXDRS_EXPORT LengthValidator : public ConfigValidator
{
public:
    LengthValidator(int minLength, int maxLength);
    Result<void> validate(const ConfigValue& value) const override;
    QString getDescription() const override;
    ValidationRule getRule() const override { return ValidationRule::Length; }

private:
    int m_minLength, m_maxLength;
};

class PHOENIXDRS_EXPORT RegexValidator : public ConfigValidator
{
public:
    explicit RegexValidator(const QString& pattern);
    Result<void> validate(const ConfigValue& value) const override;
    QString getDescription() const override;
    ValidationRule getRule() const override { return ValidationRule::Regex; }

private:
    QRegularExpression m_regex;
};

class PHOENIXDRS_EXPORT EnumValidator : public ConfigValidator
{
public:
    explicit EnumValidator(const QStringList& validValues);
    Result<void> validate(const ConfigValue& value) const override;
    QString getDescription() const override;
    ValidationRule getRule() const override { return ValidationRule::Enum; }

private:
    QStringList m_validValues;
};

/*
 * Configuration provider interface for different storage backends
 */
class PHOENIXDRS_EXPORT ConfigurationProvider
{
public:
    virtual ~ConfigurationProvider() = default;
    
    virtual Result<void> initialize() = 0;
    virtual Result<void> shutdown() = 0;
    
    virtual Result<ConfigValue> getValue(const QString& key) const = 0;
    virtual Result<void> setValue(const QString& key, const ConfigValue& value) = 0;
    virtual Result<void> removeValue(const QString& key) = 0;
    virtual Result<bool> containsKey(const QString& key) const = 0;
    virtual Result<QStringList> getAllKeys() const = 0;
    virtual Result<QStringList> getChildKeys(const QString& prefix) const = 0;
    
    virtual Result<void> sync() = 0;
    virtual StorageBackend getBackendType() const = 0;
    virtual QString getLocation() const = 0;
    
    virtual Result<void> backup(const QString& backupPath) = 0;
    virtual Result<void> restore(const QString& backupPath) = 0;
    
    virtual Result<void> exportConfiguration(const QString& filePath, const QString& format) const = 0;
    virtual Result<void> importConfiguration(const QString& filePath, const QString& format) = 0;
};

// Concrete provider implementations
class PHOENIXDRS_EXPORT IniFileProvider : public ConfigurationProvider
{
public:
    explicit IniFileProvider(const QString& filePath);
    ~IniFileProvider() override;
    
    Result<void> initialize() override;
    Result<void> shutdown() override;
    
    Result<ConfigValue> getValue(const QString& key) const override;
    Result<void> setValue(const QString& key, const ConfigValue& value) override;
    Result<void> removeValue(const QString& key) override;
    Result<bool> containsKey(const QString& key) const override;
    Result<QStringList> getAllKeys() const override;
    Result<QStringList> getChildKeys(const QString& prefix) const override;
    
    Result<void> sync() override;
    StorageBackend getBackendType() const override { return StorageBackend::IniFile; }
    QString getLocation() const override { return m_filePath; }
    
    Result<void> backup(const QString& backupPath) override;
    Result<void> restore(const QString& backupPath) override;
    
    Result<void> exportConfiguration(const QString& filePath, const QString& format) const override;
    Result<void> importConfiguration(const QString& filePath, const QString& format) override;

private:
    QString m_filePath;
    std::unique_ptr<QSettings> m_settings;
    mutable QReadWriteLock m_lock;
};

class PHOENIXDRS_EXPORT JsonFileProvider : public ConfigurationProvider
{
public:
    explicit JsonFileProvider(const QString& filePath);
    ~JsonFileProvider() override;
    
    Result<void> initialize() override;
    Result<void> shutdown() override;
    
    Result<ConfigValue> getValue(const QString& key) const override;
    Result<void> setValue(const QString& key, const ConfigValue& value) override;
    Result<void> removeValue(const QString& key) override;
    Result<bool> containsKey(const QString& key) const override;
    Result<QStringList> getAllKeys() const override;
    Result<QStringList> getChildKeys(const QString& prefix) const override;
    
    Result<void> sync() override;
    StorageBackend getBackendType() const override { return StorageBackend::JsonFile; }
    QString getLocation() const override { return m_filePath; }
    
    Result<void> backup(const QString& backupPath) override;
    Result<void> restore(const QString& backupPath) override;
    
    Result<void> exportConfiguration(const QString& filePath, const QString& format) const override;
    Result<void> importConfiguration(const QString& filePath, const QString& format) override;

private:
    Result<void> loadFromFile();
    Result<void> saveToFile() const;
    QJsonValue getNestedValue(const QJsonObject& obj, const QStringList& keyParts) const;
    void setNestedValue(QJsonObject& obj, const QStringList& keyParts, const QJsonValue& value);
    void removeNestedValue(QJsonObject& obj, const QStringList& keyParts);
    void collectKeys(const QJsonObject& obj, const QString& prefix, QStringList& keys) const;
    
    QString m_filePath;
    QJsonObject m_config;
    mutable QReadWriteLock m_lock;
    bool m_modified;
};

/*
 * Configuration watcher for monitoring configuration changes
 */
class PHOENIXDRS_EXPORT ConfigurationWatcher : public QObject
{
    Q_OBJECT
    
public:
    explicit ConfigurationWatcher(QObject* parent = nullptr);
    ~ConfigurationWatcher() override;
    
    void watchKey(const QString& key);
    void unwatchKey(const QString& key);
    void watchProvider(ConfigurationProvider* provider);
    void unwatchProvider(ConfigurationProvider* provider);
    
    QStringList getWatchedKeys() const;
    std::vector<ConfigurationProvider*> getWatchedProviders() const;

signals:
    void configurationChanged(const QString& key, const ConfigValue& oldValue, const ConfigValue& newValue);
    void providerChanged(ConfigurationProvider* provider);
    void configurationFileChanged(const QString& filePath);

private slots:
    void onFileChanged(const QString& path);
    void onDirectoryChanged(const QString& path);

private:
    QFileSystemWatcher* m_fileWatcher;
    std::unordered_set<QString> m_watchedKeys;
    std::unordered_set<ConfigurationProvider*> m_watchedProviders;
    std::unordered_map<QString, ConfigValue> m_lastValues;
    mutable QMutex m_mutex;
};

/*
 * Main configuration manager
 */
class PHOENIXDRS_EXPORT ConfigurationManager : public QObject
{
    Q_OBJECT
    
public:
    static ConfigurationManager& instance();
    
    // Provider management
    void registerProvider(const QString& name, std::unique_ptr<ConfigurationProvider> provider, ConfigScope scope);
    void unregisterProvider(const QString& name);
    ConfigurationProvider* getProvider(const QString& name) const;
    ConfigurationProvider* getProviderForScope(ConfigScope scope) const;
    QStringList getProviderNames() const;
    
    // Configuration access
    template<typename T>
    T getValue(const QString& key, const T& defaultValue = T{}, ConfigScope scope = ConfigScope::Application) const {
        auto result = getConfigValue(key, scope);
        if (result.isSuccess()) {
            return result.value().valueOr(defaultValue);
        }
        return defaultValue;
    }
    
    template<typename T>
    Result<void> setValue(const QString& key, const T& value, ConfigScope scope = ConfigScope::Application) {
        return setConfigValue(key, ConfigValue(QVariant::fromValue(value)), scope);
    }
    
    Result<ConfigValue> getConfigValue(const QString& key, ConfigScope scope = ConfigScope::Application) const;
    Result<void> setConfigValue(const QString& key, const ConfigValue& value, ConfigScope scope = ConfigScope::Application);
    Result<void> removeValue(const QString& key, ConfigScope scope = ConfigScope::Application);
    Result<bool> containsKey(const QString& key, ConfigScope scope = ConfigScope::Application) const;
    Result<QStringList> getAllKeys(ConfigScope scope = ConfigScope::Application) const;
    Result<QStringList> getChildKeys(const QString& prefix, ConfigScope scope = ConfigScope::Application) const;
    
    // Configuration schema management
    struct ConfigSchema {
        QString key;
        ConfigType type;
        QVariant defaultValue;
        QString description;
        std::shared_ptr<ConfigValidator> validator;
        ConfigScope scope;
        bool required;
        bool readOnly;
        bool encrypted;
    };
    
    void registerConfigSchema(const ConfigSchema& schema);
    void unregisterConfigSchema(const QString& key);
    Result<ConfigSchema> getConfigSchema(const QString& key) const;
    QStringList getRegisteredKeys() const;
    
    // Validation
    Result<void> validateConfiguration() const;
    Result<void> validateKey(const QString& key) const;
    std::vector<QString> getValidationErrors() const;
    
    // Synchronization and persistence
    Result<void> sync();
    Result<void> syncProvider(const QString& providerName);
    void enableAutoSync(bool enable, int intervalMs = 30000);
    bool isAutoSyncEnabled() const;
    
    // Import/Export
    Result<void> exportConfiguration(const QString& filePath, const QString& format = "json", ConfigScope scope = ConfigScope::Application) const;
    Result<void> importConfiguration(const QString& filePath, const QString& format = "json", ConfigScope scope = ConfigScope::Application);
    Result<void> mergeConfiguration(const QString& filePath, const QString& format = "json", ConfigScope scope = ConfigScope::Application);
    
    // Backup and restore
    Result<void> createBackup(const QString& backupPath) const;
    Result<void> restoreFromBackup(const QString& backupPath);
    Result<QStringList> listBackups(const QString& backupDirectory) const;
    Result<void> cleanupOldBackups(const QString& backupDirectory, int maxBackups = 10) const;
    
    // Configuration watching
    void enableConfigurationWatching(bool enable);
    bool isConfigurationWatchingEnabled() const;
    void watchKey(const QString& key);
    void unwatchKey(const QString& key);
    
    // Environment and command line integration
    void loadFromEnvironment(const QString& prefix = "PHOENIX_");
    void loadFromCommandLine(const QStringList& arguments);
    void setEnvironmentOverrides(const QStringList& keys);
    
    // Configuration profiles
    void createProfile(const QString& profileName);
    void deleteProfile(const QString& profileName);
    void switchToProfile(const QString& profileName);
    QString getCurrentProfile() const;
    QStringList getAvailableProfiles() const;
    
    // Migration support
    struct MigrationRule {
        QString oldKey;
        QString newKey;
        std::function<QVariant(const QVariant&)> transformer;
        QString description;
    };
    
    void registerMigrationRule(const MigrationRule& rule);
    Result<void> migrateConfiguration(int fromVersion, int toVersion);
    void setConfigurationVersion(int version);
    int getConfigurationVersion() const;
    
    // Security
    void setEncryptionKey(const QByteArray& key);
    bool isEncryptionEnabled() const;
    void enableConfigurationEncryption(bool enable);
    
    // Statistics and monitoring
    struct Statistics {
        size_t totalKeys = 0;
        size_t providersCount = 0;
        size_t schemasCount = 0;
        size_t validationErrors = 0;
        QDateTime lastSync;
        QDateTime lastBackup;
        std::unordered_map<ConfigScope, size_t> keysByScope;
        std::unordered_map<ConfigType, size_t> keysByType;
    };
    
    Statistics getStatistics() const;
    void resetStatistics();

signals:
    void configurationChanged(const QString& key, const ConfigValue& oldValue, const ConfigValue& newValue);
    void providerRegistered(const QString& name, ConfigScope scope);
    void providerUnregistered(const QString& name);
    void schemaRegistered(const QString& key);
    void schemaUnregistered(const QString& key);
    void validationFailed(const QString& key, const QString& error);
    void syncCompleted(bool success);
    void backupCreated(const QString& backupPath);
    void configurationMigrated(int fromVersion, int toVersion);
    void profileSwitched(const QString& profileName);

private slots:
    void onAutoSync();
    void onConfigurationChanged(const QString& key, const ConfigValue& oldValue, const ConfigValue& newValue);

private:
    ConfigurationManager();
    ~ConfigurationManager();
    
    void initializeDefaultProviders();
    void setupAutoSync();
    ConfigurationProvider* selectProvider(ConfigScope scope) const;
    Result<void> validateAgainstSchema(const QString& key, const ConfigValue& value) const;
    
    std::unordered_map<QString, std::unique_ptr<ConfigurationProvider>> m_providers;
    std::unordered_map<ConfigScope, QString> m_scopeToProvider;
    std::unordered_map<QString, ConfigSchema> m_schemas;
    std::vector<MigrationRule> m_migrationRules;
    
    std::unique_ptr<ConfigurationWatcher> m_watcher;
    QTimer* m_autoSyncTimer;
    
    QString m_currentProfile;
    QByteArray m_encryptionKey;
    bool m_encryptionEnabled;
    bool m_autoSyncEnabled;
    bool m_watchingEnabled;
    int m_configurationVersion;
    
    Statistics m_statistics;
    std::vector<QString> m_validationErrors;
    
    mutable QReadWriteLock m_providersLock;
    mutable QReadWriteLock m_schemasLock;
    mutable QMutex m_statisticsLock;
};

// Convenience macros for configuration access
#define PHOENIX_CONFIG(key, defaultValue) \
    PhoenixDRS::Core::ConfigurationManager::instance().getValue(key, defaultValue)

#define PHOENIX_SET_CONFIG(key, value) \
    PhoenixDRS::Core::ConfigurationManager::instance().setValue(key, value)

#define PHOENIX_CONFIG_SCOPED(key, defaultValue, scope) \
    PhoenixDRS::Core::ConfigurationManager::instance().getValue(key, defaultValue, scope)

#define PHOENIX_SET_CONFIG_SCOPED(key, value, scope) \
    PhoenixDRS::Core::ConfigurationManager::instance().setValue(key, value, scope)

// Configuration schema registration helper
#define PHOENIX_REGISTER_CONFIG(key, type, defaultValue, description, validator, scope, required, readOnly) \
    do { \
        PhoenixDRS::Core::ConfigurationManager::ConfigSchema schema; \
        schema.key = key; \
        schema.type = type; \
        schema.defaultValue = defaultValue; \
        schema.description = description; \
        schema.validator = validator; \
        schema.scope = scope; \
        schema.required = required; \
        schema.readOnly = readOnly; \
        schema.encrypted = false; \
        PhoenixDRS::Core::ConfigurationManager::instance().registerConfigSchema(schema); \
    } while(0)

} // namespace Core
} // namespace PhoenixDRS