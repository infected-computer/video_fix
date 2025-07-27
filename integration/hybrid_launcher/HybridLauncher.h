/*
 * PhoenixDRS Professional - Hybrid Python-C++ Launcher
 * משגר היברידי Python-C++ - PhoenixDRS מקצועי
 */

#pragma once

#include "../cpp_gui/include/Common.h"
#include "../cpp_gui/include/Core/ErrorHandling.h"

#include <QtCore/QObject>
#include <QtCore/QString>
#include <QtCore/QStringList>
#include <QtCore/QProcess>
#include <QtCore/QDir>
#include <QtCore/QCoreApplication>
#include <QtWidgets/QApplication>

#include <memory>
#include <Python.h>

namespace PhoenixDRS {
namespace Launcher {

// Launch modes
enum class LaunchMode {
    Unknown,
    PythonCLI,          // Pure Python command line interface
    CppGUI,             // Pure C++ GUI application
    HybridMode,         // Integrated Python CLI + C++ GUI
    PythonWithCppLibs,  // Python script using C++ libraries
    CppWithPythonPlugins // C++ app with Python plugins
};

// Command line argument structure
struct LaunchArguments {
    LaunchMode mode = LaunchMode::Unknown;
    QString command;
    QStringList arguments;
    QString workingDirectory;
    QString configFile;
    QString logFile;
    QString tempDirectory;
    bool verbose = false;
    bool debugMode = false;
    bool enableProfiling = false;
    QString pythonPath;
    QString cppLibPath;
};

/*
 * Python environment manager
 */
class PHOENIXDRS_EXPORT PythonEnvironment
{
public:
    PythonEnvironment();
    ~PythonEnvironment();
    
    bool initialize(const QString& pythonPath = QString());
    void shutdown();
    bool isInitialized() const { return m_initialized; }
    
    // Python path management
    bool addToPath(const QString& path);
    bool addModulePath(const QString& path);
    QStringList getPythonPath() const;
    
    // Module loading
    bool importModule(const QString& moduleName);
    bool loadScript(const QString& scriptPath);
    
    // Execution
    PhoenixDRS::Core::Result<int> executeScript(const QString& scriptPath, const QStringList& arguments);
    PhoenixDRS::Core::Result<int> executeCommand(const QString& command);
    PhoenixDRS::Core::Result<PyObject*> evaluateExpression(const QString& expression);
    
    // Error handling
    QString getLastPythonError() const;
    void clearPythonError();
    
    // Environment info
    QString getPythonVersion() const;
    QString getPythonExecutable() const;
    QStringList getInstalledPackages() const;

private:
    bool m_initialized;
    QString m_pythonPath;
    QString m_lastError;
    wchar_t* m_programName;
};

/*
 * Unified launcher that handles both Python and C++ execution
 */
class PHOENIXDRS_EXPORT HybridLauncher : public QObject
{
    Q_OBJECT
    
public:
    explicit HybridLauncher(QObject* parent = nullptr);
    ~HybridLauncher() override;
    
    // Initialization
    bool initialize(const LaunchArguments& args);
    void shutdown();
    
    // Launch methods
    PhoenixDRS::Core::Result<int> launch();
    PhoenixDRS::Core::Result<int> launchPythonCLI();
    PhoenixDRS::Core::Result<int> launchCppGUI();
    PhoenixDRS::Core::Result<int> launchHybridMode();
    
    // Argument parsing
    static LaunchArguments parseArguments(const QStringList& args);
    static void printUsage();
    static void printVersion();
    
    // Environment setup
    bool setupEnvironment();
    bool validateDependencies();
    bool checkPythonEnvironment();
    bool checkCppLibraries();
    
    // Configuration
    void setConfigFile(const QString& configFile);
    QString getConfigFile() const { return m_configFile; }
    
    void setLogFile(const QString& logFile);
    QString getLogFile() const { return m_logFile; }
    
    // Status
    LaunchMode getCurrentMode() const { return m_args.mode; }
    bool isRunning() const { return m_isRunning; }
    
    // Inter-process communication
    bool startIPCServer();
    void stopIPCServer();
    bool sendIPCMessage(const QString& message);

signals:
    void processStarted(const QString& processType);
    void processFinished(const QString& processType, int exitCode);
    void errorOccurred(const QString& error);
    void statusChanged(const QString& status);
    void ipcMessageReceived(const QString& message);

private slots:
    void onPythonProcessFinished(int exitCode, QProcess::ExitStatus exitStatus);
    void onPythonProcessError(QProcess::ProcessError error);
    void onIPCConnectionReceived();

private:
    // Argument parsing helpers
    static LaunchMode determineLaunchMode(const QStringList& args);
    static bool isGuiCommand(const QStringList& args);
    static bool isPythonCommand(const QStringList& args);
    
    // Environment setup
    bool setupPythonEnvironment();
    bool setupCppEnvironment();
    bool setupHybridEnvironment();
    
    // Process management
    bool startPythonProcess(const QStringList& arguments);
    bool startCppProcess(const QStringList& arguments);
    
    // Resource management
    void cleanupResources();
    
    // Configuration loading
    bool loadConfiguration();
    void applyConfiguration();
    
    LaunchArguments m_args;
    QString m_configFile;
    QString m_logFile;
    bool m_isRunning;
    
    std::unique_ptr<PythonEnvironment> m_pythonEnv;
    std::unique_ptr<QProcess> m_pythonProcess;
    std::unique_ptr<QApplication> m_qtApp;
    
    // IPC components
    class IPCServer* m_ipcServer;
    
    // Paths and directories
    QString m_applicationDir;
    QString m_pythonScriptsDir;
    QString m_cppLibrariesDir;
    QString m_tempDir;
};

/*
 * IPC server for communication between Python and C++ components
 */
class PHOENIXDRS_EXPORT IPCServer : public QObject
{
    Q_OBJECT
    
public:
    explicit IPCServer(QObject* parent = nullptr);
    ~IPCServer() override;
    
    bool startServer(const QString& serverName = "PhoenixDRS_IPC");
    void stopServer();
    bool isRunning() const;
    
    QString getServerName() const { return m_serverName; }
    int getConnectedClients() const;
    
    // Message broadcasting
    void broadcastMessage(const QString& message);
    void sendMessageToClient(const QString& clientId, const QString& message);

signals:
    void clientConnected(const QString& clientId);
    void clientDisconnected(const QString& clientId);
    void messageReceived(const QString& clientId, const QString& message);
    void serverError(const QString& error);

private slots:
    void onNewConnection();
    void onClientDisconnected();
    void onMessageReceived();

private:
    class QLocalServer* m_server;
    QString m_serverName;
    std::unordered_map<QString, class QLocalSocket*> m_clients;
};

/*
 * Configuration manager for hybrid launcher
 */
class PHOENIXDRS_EXPORT LauncherConfiguration
{
public:
    struct Config {
        // Python configuration
        QString pythonExecutable;
        QString pythonHome;
        QStringList pythonPath;
        QString pythonScriptsDirectory;
        
        // C++ configuration
        QString cppExecutable;
        QString cppLibrariesDirectory;
        QString qtPluginsDirectory;
        
        // Hybrid configuration
        bool enableIPC = true;
        QString ipcServerName;
        int ipcTimeout = 30000;
        
        // Logging configuration
        QString logDirectory;
        QString logLevel;
        bool enableConsoleLogging = true;
        bool enableFileLogging = true;
        
        // Performance configuration
        bool enableProfiling = false;
        QString profilingOutputDirectory;
        int maxMemoryUsageMB = -1; // -1 = unlimited
        
        // Security configuration
        bool enableSandboxMode = false;
        QStringList allowedDirectories;
        QStringList blockedCommands;
    };
    
    LauncherConfiguration();
    ~LauncherConfiguration();
    
    bool loadFromFile(const QString& configFile);
    bool saveToFile(const QString& configFile) const;
    
    bool loadFromCommandLine(const QStringList& args);
    bool loadFromEnvironment();
    
    const Config& getConfig() const { return m_config; }
    void setConfig(const Config& config) { m_config = config; }
    
    // Configuration validation
    bool validate() const;
    QStringList getValidationErrors() const;
    
    // Default configuration
    static Config getDefaultConfig();
    static QString getDefaultConfigPath();

private:
    Config m_config;
    mutable QStringList m_validationErrors;
};

// Utility functions
PHOENIXDRS_EXPORT QString findPythonExecutable();
PHOENIXDRS_EXPORT QString findPhoenixDRSInstallation();
PHOENIXDRS_EXPORT QStringList getAvailablePythonVersions();
PHOENIXDRS_EXPORT bool isValidPythonEnvironment(const QString& pythonPath);
PHOENIXDRS_EXPORT bool isValidCppEnvironment(const QString& cppPath);
PHOENIXDRS_EXPORT QString getSystemTempDirectory();
PHOENIXDRS_EXPORT QString getUserConfigDirectory();

} // namespace Launcher
} // namespace PhoenixDRS