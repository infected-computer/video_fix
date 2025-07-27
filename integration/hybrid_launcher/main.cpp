/*
 * PhoenixDRS Professional - Unified Hybrid Launcher Main Entry Point
 * נקודת כניסה ראשית למשגר היברידי מאוחד - PhoenixDRS מקצועי
 */

#include "HybridLauncher.h"
#include "../cpp_gui/include/Core/ErrorHandling.h"
#include "../cpp_gui/include/ForensicLogger.h"

#include <QtCore/QCoreApplication>
#include <QtCore/QCommandLineParser>
#include <QtCore/QDir>
#include <QtCore/QStandardPaths>
#include <QtCore/QProcess>
#include <QtCore/QDebug>

#include <iostream>
#include <memory>

using namespace PhoenixDRS;
using namespace PhoenixDRS::Launcher;

// Global application information
constexpr const char* APPLICATION_NAME = "PhoenixDRS Professional";
constexpr const char* APPLICATION_VERSION = "2.0.0";
constexpr const char* ORGANIZATION_NAME = "PhoenixDRS";
constexpr const char* ORGANIZATION_DOMAIN = "phoenixdrs.com";

// Signal handlers for graceful shutdown
volatile sig_atomic_t g_shouldExit = 0;

#ifdef _WIN32
#include <windows.h>
BOOL WINAPI consoleHandler(DWORD signal) {
    if (signal == CTRL_C_EVENT || signal == CTRL_BREAK_EVENT) {
        g_shouldExit = 1;
        return TRUE;
    }
    return FALSE;
}
#else
#include <signal.h>
void signalHandler(int signal) {
    if (signal == SIGINT || signal == SIGTERM) {
        g_shouldExit = 1;
    }
}
#endif

void setupSignalHandlers() {
#ifdef _WIN32
    SetConsoleCtrlHandler(consoleHandler, TRUE);
#else
    signal(SIGINT, signalHandler);
    signal(SIGTERM, signalHandler);
#endif
}

void printBanner() {
    std::cout << R"(
╔═══════════════════════════════════════════════════════════════════════════════╗
║                          PhoenixDRS Professional v2.0.0                      ║
║                     Advanced Digital Forensics Suite                         ║
║                    מערכת פורנזיקה דיגיטלית מתקדמת                          ║
╠═══════════════════════════════════════════════════════════════════════════════╣
║ Hybrid Python-C++ Architecture | Enterprise-Grade Performance               ║
║ Memory Forensics | Network Analysis | Blockchain Investigation               ║
║ Machine Learning | AI Content Analysis | Distributed Computing              ║
╚═══════════════════════════════════════════════════════════════════════════════╝
)" << std::endl;
}

void printUsage() {
    std::cout << R"(
PhoenixDRS Professional - Unified Command Line Interface

USAGE:
    phoenixdrs [OPTIONS] <COMMAND> [COMMAND_ARGS...]

GLOBAL OPTIONS:
    -h, --help              Show this help message
    -v, --version           Show version information
    --config <file>         Use custom configuration file
    --log-file <file>       Log output to file
    --log-level <level>     Set log level (DEBUG, INFO, WARN, ERROR)
    --temp-dir <dir>        Set temporary directory
    --python-path <path>    Set Python executable path
    --cpp-lib-path <path>   Set C++ libraries path
    --verbose               Enable verbose output
    --debug                 Enable debug mode
    --profile               Enable performance profiling
    --no-gui                Force command-line mode
    --gui-only              Force GUI-only mode

FORENSIC COMMANDS:
    gui                     Launch graphical user interface (default)
    
    image create            Create disk image
        --source <device>       Source device or file
        --dest <file>           Destination image file
        --format <fmt>          Image format (dd, e01, aff4)
        --compression <level>   Compression level (0-9)
        --verify                Verify image after creation
        --progress              Show progress bar
    
    image verify            Verify disk image integrity
        --image <file>          Image file to verify
        --hash-file <file>      Expected hash file
    
    carve                   Carve files from disk image
        --image <file>          Disk image file
        --output <dir>          Output directory
        --signatures <file>     Signature database file
        --parallel              Enable parallel processing
        --workers <count>       Number of worker threads
        --min-size <bytes>      Minimum file size
        --max-size <bytes>      Maximum file size
        --types <list>          File types to carve (comma-separated)
    
    rebuild-video           Rebuild corrupted video files
        --image <file>          Disk image file
        --output <dir>          Output directory
        --format <fmt>          Video format (mov, mp4, avi)
        --quality <threshold>   Quality threshold (0.0-1.0)
    
    analyze-memory          Analyze memory dump
        --dump <file>           Memory dump file
        --profile <name>        Memory profile
        --output <dir>          Analysis output directory
        --extract-strings       Extract strings from memory
        --find-processes        Find hidden processes
        --network-connections   Analyze network connections
    
    analyze-network         Analyze network capture
        --pcap <file>           PCAP capture file
        --output <dir>          Analysis output directory
        --reconstruct-flows     Reconstruct TCP flows
        --extract-files         Extract transmitted files
        --detect-threats        Detect malicious traffic
    
    analyze-blockchain      Analyze blockchain data
        --data <file>           Blockchain data file
        --currency <type>       Cryptocurrency type
        --output <dir>          Analysis output directory
        --trace-flows           Trace money flows
        --detect-mixing         Detect mixing services
    
    ml-classify             Machine learning file classification
        --input <path>          Input file or directory
        --output <dir>          Classification results directory
        --models <list>         ML models to use
        --confidence <threshold> Minimum confidence threshold
    
    crypto-analyze          Cryptographic analysis
        --input <file>          File to analyze
        --output <dir>          Analysis results directory
        --attempt-crack         Attempt to crack encryption
        --dictionary <file>     Password dictionary file
        --gpu-acceleration      Use GPU acceleration

CASE MANAGEMENT:
    case create             Create new forensic case
        --name <name>           Case name
        --number <number>       Case number
        --examiner <name>       Examiner name
        --description <text>    Case description
        --evidence-dir <dir>    Evidence storage directory
    
    case open               Open existing case
        --case-file <file>      Case file path
    
    case export             Export case data
        --case-file <file>      Case file path
        --output <dir>          Export directory
        --format <fmt>          Export format (xml, json, pdf)

SYSTEM COMMANDS:
    system info             Show system information
    system benchmark        Run performance benchmark
    system check            Check system requirements
    system update           Update PhoenixDRS components
    
    config show             Show current configuration
    config set              Set configuration value
        --key <key>             Configuration key
        --value <value>         Configuration value
    
    license show            Show license information
    license activate        Activate license
        --key <license-key>     License key
    
    plugin list             List available plugins
    plugin install          Install plugin
        --plugin <name>         Plugin name or file
    plugin remove           Remove plugin
        --plugin <name>         Plugin name

DISTRIBUTED COMPUTING:
    cluster create          Create compute cluster
        --nodes <list>          Cluster node addresses
        --config <file>         Cluster configuration file
    
    cluster join            Join existing cluster
        --address <addr>        Cluster address
        --token <token>         Authentication token
    
    job submit              Submit distributed job
        --task <type>           Task type
        --input <path>          Input data path
        --output <path>         Output data path
        --priority <level>      Job priority (1-10)

EXAMPLES:
    # Launch GUI
    phoenixdrs gui
    
    # Create disk image with verification
    phoenixdrs image create --source /dev/sdb --dest evidence.e01 --format e01 --verify
    
    # Carve files with parallel processing
    phoenixdrs carve --image evidence.e01 --output carved_files --parallel --workers 8
    
    # Rebuild Canon MOV videos
    phoenixdrs rebuild-video --image evidence.e01 --output videos --format mov
    
    # Analyze memory dump
    phoenixdrs analyze-memory --dump memory.dmp --profile Win10x64 --output memory_analysis
    
    # ML-based file classification
    phoenixdrs ml-classify --input suspicious_files/ --output classification_results
    
    # Create forensic case
    phoenixdrs case create --name "Case001" --examiner "John Doe" --evidence-dir /cases/case001
    
    # Run distributed file carving
    phoenixdrs job submit --task carve --input large_image.dd --output results/ --priority 8

For more information, visit: https://phoenixdrs.com/docs
Support: support@phoenixdrs.com
)" << std::endl;
}

void printVersion() {
    std::cout << APPLICATION_NAME << " version " << APPLICATION_VERSION << std::endl;
    std::cout << "Built with Qt " << QT_VERSION_STR << " on " << __DATE__ << " " << __TIME__ << std::endl;
    std::cout << "Copyright (C) 2024 PhoenixDRS Team. All rights reserved." << std::endl;
    
    // Print feature availability
    std::cout << "\nFeatures:" << std::endl;
    std::cout << "  Python Integration: Yes" << std::endl;
    std::cout << "  C++ High Performance: Yes" << std::endl;
    std::cout << "  GUI Interface: Yes" << std::endl;
    std::cout << "  Distributed Computing: Yes" << std::endl;
    std::cout << "  Machine Learning: Yes" << std::endl;
    std::cout << "  Blockchain Analysis: Yes" << std::endl;
    std::cout << "  Memory Forensics: Yes" << std::endl;
    std::cout << "  Network Forensics: Yes" << std::endl;
    
#ifdef ENABLE_OPENSSL
    std::cout << "  OpenSSL Support: Yes" << std::endl;
#else
    std::cout << "  OpenSSL Support: No" << std::endl;
#endif

#ifdef ENABLE_OPENCV
    std::cout << "  OpenCV Support: Yes" << std::endl;
#else
    std::cout << "  OpenCV Support: No" << std::endl;
#endif

#ifdef ENABLE_CUDA
    std::cout << "  CUDA Support: Yes" << std::endl;
#else
    std::cout << "  CUDA Support: No" << std::endl;
#endif
}

bool checkPrerequisites() {
    // Check if running as administrator/root for certain operations
    bool hasElevatedPrivileges = false;
#ifdef _WIN32
    // Windows: Check if running as administrator
    BOOL isElevated = FALSE;
    HANDLE token = nullptr;
    if (OpenProcessToken(GetCurrentProcess(), TOKEN_QUERY, &token)) {
        TOKEN_ELEVATION elevation;
        DWORD size = sizeof(TOKEN_ELEVATION);
        if (GetTokenInformation(token, TokenElevation, &elevation, sizeof(elevation), &size)) {
            isElevated = elevation.TokenIsElevated;
        }
        CloseHandle(token);
    }
    hasElevatedPrivileges = isElevated;
#else
    // Unix-like: Check if running as root
    hasElevatedPrivileges = (geteuid() == 0);
#endif

    if (!hasElevatedPrivileges) {
        std::cout << "Warning: Running without elevated privileges. Some operations may be limited." << std::endl;
        std::cout << "For full functionality, run as administrator (Windows) or root (Unix)." << std::endl;
    }

    // Check available disk space
    QString tempDir = QStandardPaths::writableLocation(QStandardPaths::TempLocation);
    QDir temp(tempDir);
    // Basic check - more sophisticated checks would be in the implementation
    if (!temp.exists()) {
        std::cerr << "Error: Cannot access temporary directory: " << tempDir.toStdString() << std::endl;
        return false;
    }

    return true;
}

int main(int argc, char *argv[])
{
    // Setup signal handlers for graceful shutdown
    setupSignalHandlers();
    
    // Set application information
    QCoreApplication::setApplicationName(APPLICATION_NAME);
    QCoreApplication::setApplicationVersion(APPLICATION_VERSION);
    QCoreApplication::setOrganizationName(ORGANIZATION_NAME);
    QCoreApplication::setOrganizationDomain(ORGANIZATION_DOMAIN);
    
    try {
        // Parse initial arguments to determine if we need GUI
        QStringList arguments;
        for (int i = 1; i < argc; ++i) {
            arguments << QString::fromLocal8Bit(argv[i]);
        }
        
        // Show help or version if requested
        if (arguments.contains("-h") || arguments.contains("--help")) {
            printBanner();
            printUsage();
            return 0;
        }
        
        if (arguments.contains("-v") || arguments.contains("--version")) {
            printVersion();
            return 0;
        }
        
        // Check if we should show banner
        bool shouldShowBanner = !arguments.contains("--no-banner") && 
                               !arguments.contains("--quiet") &&
                               (arguments.isEmpty() || arguments.first() != "gui");
        
        if (shouldShowBanner) {
            printBanner();
        }
        
        // Check system prerequisites
        if (!checkPrerequisites()) {
            return 1;
        }
        
        // Create appropriate QApplication type
        std::unique_ptr<QCoreApplication> app;
        LaunchArguments launchArgs = HybridLauncher::parseArguments(arguments);
        
        if (launchArgs.mode == LaunchMode::CppGUI || 
            launchArgs.mode == LaunchMode::HybridMode ||
            (arguments.isEmpty() && !arguments.contains("--no-gui"))) {
            // GUI mode - create QApplication
            app = std::make_unique<QApplication>(argc, argv);
        } else {
            // CLI mode - create QCoreApplication
            app = std::make_unique<QCoreApplication>(argc, argv);
        }
        
        // Initialize logging early
        QString logFile = launchArgs.logFile;
        if (logFile.isEmpty()) {
            QString logDir = QStandardPaths::writableLocation(QStandardPaths::AppDataLocation);
            QDir().mkpath(logDir);
            logFile = QDir(logDir).filePath("phoenixdrs.log");
        }
        
        ForensicLogger::instance()->initialize(logFile);
        ForensicLogger::instance()->logInfo("PhoenixDRS Professional starting...");
        
        // Create and initialize hybrid launcher
        HybridLauncher launcher;
        if (!launcher.initialize(launchArgs)) {
            std::cerr << "Error: Failed to initialize PhoenixDRS launcher" << std::endl;
            ForensicLogger::instance()->logError("Failed to initialize launcher");
            return 1;
        }
        
        // Set up cleanup on exit
        QObject::connect(app.get(), &QCoreApplication::aboutToQuit, [&launcher]() {
            launcher.shutdown();
            ForensicLogger::instance()->logInfo("PhoenixDRS Professional shutting down...");
        });
        
        // Launch the application
        auto result = launcher.launch();
        if (!result.isSuccess()) {
            std::cerr << "Error: " << result.error().message().toStdString() << std::endl;
            ForensicLogger::instance()->logError(result.error().message());
            return 1;
        }
        
        // If this is a CLI command, return immediately
        if (launchArgs.mode == LaunchMode::PythonCLI) {
            return result.value();
        }
        
        // Otherwise, enter Qt event loop
        int exitCode = app->exec();
        
        ForensicLogger::instance()->logInfo(QString("PhoenixDRS Professional exited with code %1").arg(exitCode));
        return exitCode;
        
    } catch (const PhoenixDRS::Core::PhoenixException& e) {
        std::cerr << "PhoenixDRS Error: " << e.message().toStdString() << std::endl;
        if (ForensicLogger::instance()) {
            ForensicLogger::instance()->logError(e.message());
        }
        return static_cast<int>(e.errorCode());
        
    } catch (const std::exception& e) {
        std::cerr << "System Error: " << e.what() << std::endl;
        if (ForensicLogger::instance()) {
            ForensicLogger::instance()->logError(QString("System error: %1").arg(e.what()));
        }
        return 1;
        
    } catch (...) {
        std::cerr << "Unknown Error: An unexpected error occurred" << std::endl;
        if (ForensicLogger::instance()) {
            ForensicLogger::instance()->logError("Unknown error occurred");
        }
        return 1;
    }
}