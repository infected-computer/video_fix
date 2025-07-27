#ifndef MAIN_APPLICATION_H
#define MAIN_APPLICATION_H

#include <QApplication>
#include <QMainWindow>
#include <QSystemTrayIcon>
#include <QMenu>
#include <QAction>
#include <QTimer>
#include <QSettings>
#include <QTranslator>
#include <QSplashScreen>
#include <memory>

// Forward declarations
class MainDashboard;
class WorkflowManager;
class PluginManager;
class CoreEngine;
class PreferencesDialog;
class AboutDialog;
class LogViewerDialog;
class PerformanceMonitor;

/**
 * @brief Main application class for PhoenixDRS GUI
 */
class MainApplication : public QApplication {
    Q_OBJECT

public:
    explicit MainApplication(int &argc, char **argv);
    ~MainApplication();

    // Application lifecycle
    bool initialize();
    void shutdown();
    
    // Global access
    static MainApplication* instance();
    
    // Configuration management
    QSettings* getSettings() const { return m_settings.get(); }
    void saveSettings();
    void loadSettings();
    
    // Language/Localization
    void setLanguage(const QString& languageCode);
    QString getCurrentLanguage() const;
    QStringList getAvailableLanguages() const;
    
    // Theme management
    void setTheme(const QString& themeName);
    QString getCurrentTheme() const;
    QStringList getAvailableThemes() const;
    
    // Component access
    MainDashboard* getMainDashboard() const { return m_mainDashboard.get(); }
    WorkflowManager* getWorkflowManager() const;
    PluginManager* getPluginManager() const;
    CoreEngine* getCoreEngine() const;
    
    // System tray
    void setupSystemTray();
    void showSystemTrayMessage(const QString& title, const QString& message);
    
    // Error handling
    void handleCriticalError(const QString& error);
    void showErrorDialog(const QString& title, const QString& message);
    
signals:
    void applicationInitialized();
    void applicationShuttingDown();
    void themeChanged(const QString& themeName);
    void languageChanged(const QString& languageCode);
    
private slots:
    void onSystemTrayActivated(QSystemTrayIcon::ActivationReason reason);
    void onAboutToQuit();
    void onUpdateCheck();
    void showAboutDialog();
    void showPreferences();
    void showLogViewer();
    void toggleMainWindow();
    
private:
    static MainApplication* s_instance;
    
    // Core components
    std::unique_ptr<MainDashboard> m_mainDashboard;
    std::unique_ptr<QSettings> m_settings;
    std::unique_ptr<QTranslator> m_translator;
    std::unique_ptr<QSplashScreen> m_splashScreen;
    
    // System tray
    std::unique_ptr<QSystemTrayIcon> m_systemTray;
    std::unique_ptr<QMenu> m_trayMenu;
    
    // Dialogs
    std::unique_ptr<PreferencesDialog> m_preferencesDialog;
    std::unique_ptr<AboutDialog> m_aboutDialog;
    std::unique_ptr<LogViewerDialog> m_logViewerDialog;
    
    // Timers
    std::unique_ptr<QTimer> m_updateTimer;
    std::unique_ptr<QTimer> m_performanceTimer;
    
    // Application state
    bool m_initialized;
    QString m_currentTheme;
    QString m_currentLanguage;
    
    // Helper methods
    void setupTranslations();
    void setupThemes();
    void createSplashScreen();
    void createSystemTrayMenu();
    void initializeComponents();
    void connectSignals();
    void loadApplicationStyle(const QString& themeName);
    QString getDefaultSettingsPath() const;
};

/**
 * @brief Main dashboard window
 */
class MainDashboard : public QMainWindow {
    Q_OBJECT

public:
    explicit MainDashboard(QWidget* parent = nullptr);
    ~MainDashboard();

    // Window management
    void showDashboard();
    void hideDashboard();
    bool isDashboardVisible() const;
    
    // UI components access
    class CentralWidget* getCentralWidget() const;
    class StatusBar* getStatusBar() const;
    class MenuBar* getMenuBar() const;
    class ToolBar* getToolBar() const;
    
    // View management
    void showView(const QString& viewName);
    QString getCurrentView() const;
    QStringList getAvailableViews() const;
    
    // Progress and notifications
    void showProgress(const QString& operation, int percentage);
    void hideProgress();
    void showNotification(const QString& message, int timeoutMs = 5000);
    
    // Window state
    void saveWindowState();
    void restoreWindowState();
    
protected:
    void closeEvent(QCloseEvent* event) override;
    void changeEvent(QEvent* event) override;
    void showEvent(QShowEvent* event) override;
    void hideEvent(QHideEvent* event) override;
    
signals:
    void dashboardShown();
    void dashboardHidden();
    void viewChanged(const QString& viewName);
    void operationRequested(const QString& operation, const QVariantMap& parameters);
    
private slots:
    void onNewCase();
    void onOpenCase();
    void onSaveCase();
    void onExportResults();
    void onImportEvidence();
    void onStartAnalysis();
    void onStopAnalysis();
    void onViewLogs();
    void onShowPreferences();
    void onShowAbout();
    void onViewChanged();
    void onOperationProgress(const QString& operation, int progress);
    void onOperationCompleted(const QString& operation, bool success);
    void onSystemStatusChanged();
    
private:
    // UI Components
    class CentralWidget* m_centralWidget;
    class CustomMenuBar* m_menuBar;
    class CustomToolBar* m_toolBar;
    class CustomStatusBar* m_statusBar;
    class SidePanel* m_sidePanel;
    class NotificationWidget* m_notificationWidget;
    class ProgressWidget* m_progressWidget;
    
    // View management
    QStringList m_availableViews;
    QString m_currentView;
    QMap<QString, QWidget*> m_viewWidgets;
    
    // Window state
    bool m_isInitialized;
    QTimer* m_statusUpdateTimer;
    
    void setupUI();
    void createMenuBar();
    void createToolBar();
    void createStatusBar();
    void createCentralWidget();
    void createSidePanel();
    void setupViews();
    void connectUISignals();
    void updateUI();
    void updateSystemStatus();
};

/**
 * @brief Central widget containing main content areas
 */
class CentralWidget : public QWidget {
    Q_OBJECT

public:
    explicit CentralWidget(QWidget* parent = nullptr);
    ~CentralWidget();

    // View management
    void addView(const QString& name, QWidget* widget);
    void removeView(const QString& name);
    void showView(const QString& name);
    void hideView(const QString& name);
    QString getCurrentView() const;
    
    // Layout management
    void setSplitterSizes(const QList<int>& sizes);
    QList<int> getSplitterSizes() const;
    void setSplitterOrientation(Qt::Orientation orientation);
    
    // Content areas
    class WorkspaceArea* getWorkspaceArea() const;
    class PropertiesPanel* getPropertiesPanel() const;
    class OutputPanel* getOutputPanel() const;
    
signals:
    void viewChanged(const QString& viewName);
    void splitterMoved();
    
private slots:
    void onViewStackChanged(int index);
    void onSplitterMoved();
    
private:
    class QStackedWidget* m_viewStack;
    class QSplitter* m_mainSplitter;
    class QSplitter* m_rightSplitter;
    
    class WorkspaceArea* m_workspaceArea;
    class PropertiesPanel* m_propertiesPanel;
    class OutputPanel* m_outputPanel;
    
    QMap<QString, QWidget*> m_views;
    QString m_currentView;
    
    void setupLayout();
    void createWorkspaceArea();
    void createPropertiesPanel();
    void createOutputPanel();
};

/**
 * @brief Workspace area for main content
 */
class WorkspaceArea : public QWidget {
    Q_OBJECT

public:
    explicit WorkspaceArea(QWidget* parent = nullptr);
    ~WorkspaceArea();

    // Content management
    void setContent(QWidget* content);
    QWidget* getContent() const;
    void clearContent();
    
    // Tab management
    void addTab(const QString& title, QWidget* content);
    void removeTab(int index);
    void setCurrentTab(int index);
    int getCurrentTab() const;
    int getTabCount() const;
    
    // View modes
    void setViewMode(const QString& mode);
    QString getViewMode() const;
    QStringList getAvailableViewModes() const;
    
signals:
    void contentChanged();
    void tabChanged(int index);
    void viewModeChanged(const QString& mode);
    
private slots:
    void onTabChanged(int index);
    void onTabCloseRequested(int index);
    
private:
    class QTabWidget* m_tabWidget;
    class QStackedWidget* m_contentStack;
    class QVBoxLayout* m_layout;
    
    QString m_currentViewMode;
    QStringList m_viewModes;
    
    void setupLayout();
    void updateViewMode();
};

/**
 * @brief Properties panel for displaying item details
 */
class PropertiesPanel : public QWidget {
    Q_OBJECT

public:
    explicit PropertiesPanel(QWidget* parent = nullptr);
    ~PropertiesPanel();

    // Property display
    void setProperties(const QVariantMap& properties);
    void addProperty(const QString& key, const QVariant& value);
    void removeProperty(const QString& key);
    void clearProperties();
    
    // Grouping
    void setPropertyGroup(const QString& group, const QVariantMap& properties);
    void addPropertyToGroup(const QString& group, const QString& key, const QVariant& value);
    QStringList getPropertyGroups() const;
    
    // Display modes
    void setDisplayMode(const QString& mode);
    QString getDisplayMode() const;
    
signals:
    void propertyChanged(const QString& key, const QVariant& value);
    void propertySelectionChanged(const QString& key);
    
private slots:
    void onPropertyChanged();
    void onPropertySelected();
    
private:
    class QTreeWidget* m_propertyTree;
    class QVBoxLayout* m_layout;
    class QComboBox* m_displayModeCombo;
    
    QVariantMap m_properties;
    QMap<QString, QVariantMap> m_propertyGroups;
    QString m_displayMode;
    
    void setupUI();
    void updatePropertyDisplay();
    void addPropertyItem(QTreeWidgetItem* parent, const QString& key, const QVariant& value);
};

/**
 * @brief Output panel for logs, results, and messages
 */
class OutputPanel : public QWidget {
    Q_OBJECT

public:
    explicit OutputPanel(QWidget* parent = nullptr);
    ~OutputPanel();

    // Tab management
    void addOutputTab(const QString& name, QWidget* content);
    void removeOutputTab(const QString& name);
    void showOutputTab(const QString& name);
    QString getCurrentOutputTab() const;
    
    // Built-in output types
    void addLogMessage(const QString& message, const QString& level = "INFO");
    void addResultItem(const QVariantMap& result);
    void addErrorMessage(const QString& error);
    void clearOutput(const QString& type = QString());
    
    // Output export
    void exportOutput(const QString& type, const QString& filePath);
    QString getOutputAsText(const QString& type) const;
    
signals:
    void outputTabChanged(const QString& name);
    void outputCleared(const QString& type);
    
private slots:
    void onOutputTabChanged(int index);
    void onClearOutput();
    void onExportOutput();
    
private:
    class QTabWidget* m_outputTabs;
    class QTextEdit* m_logOutput;
    class QTreeWidget* m_resultOutput;
    class QTextEdit* m_errorOutput;
    
    QMap<QString, QWidget*> m_outputWidgets;
    
    void setupUI();
    void createLogOutput();
    void createResultOutput();
    void createErrorOutput();
};

/**
 * @brief Side panel for navigation and quick access
 */
class SidePanel : public QWidget {
    Q_OBJECT

public:
    explicit SidePanel(QWidget* parent = nullptr);
    ~SidePanel();

    // Panel management
    void addPanel(const QString& name, QWidget* panel);
    void removePanel(const QString& name);
    void showPanel(const QString& name);
    void hidePanel(const QString& name);
    QString getActivePanel() const;
    
    // Built-in panels
    class CaseExplorer* getCaseExplorer() const;
    class EvidenceExplorer* getEvidenceExplorer() const;
    class PluginExplorer* getPluginExplorer() const;
    class TaskMonitor* getTaskMonitor() const;
    
    // Panel state
    void setPanelCollapsed(bool collapsed);
    bool isPanelCollapsed() const;
    
signals:
    void panelChanged(const QString& name);
    void panelCollapsedChanged(bool collapsed);
    
private slots:
    void onPanelChanged();
    void onCollapseToggled();
    
private:
    class QStackedWidget* m_panelStack;
    class QListWidget* m_panelList;
    class QSplitter* m_splitter;
    class QPushButton* m_collapseButton;
    
    // Built-in panels
    class CaseExplorer* m_caseExplorer;
    class EvidenceExplorer* m_evidenceExplorer;
    class PluginExplorer* m_pluginExplorer;
    class TaskMonitor* m_taskMonitor;
    
    QMap<QString, QWidget*> m_panels;
    QString m_activePanel;
    bool m_collapsed;
    
    void setupUI();
    void createBuiltinPanels();
};

// Utility classes for UI components
class NotificationWidget : public QWidget {
    Q_OBJECT

public:
    explicit NotificationWidget(QWidget* parent = nullptr);
    void showNotification(const QString& message, int timeoutMs = 5000);
    void hideNotification();

private slots:
    void onTimeout();

private:
    class QLabel* m_messageLabel;
    class QTimer* m_timer;
    class QPropertyAnimation* m_animation;

    void setupUI();
    void animateShow();
    void animateHide();
};

class ProgressWidget : public QWidget {
    Q_OBJECT

public:
    explicit ProgressWidget(QWidget* parent = nullptr);
    void showProgress(const QString& operation, int percentage);
    void hideProgress();
    void setProgressText(const QString& text);
    void setProgressValue(int value);

private:
    class QProgressBar* m_progressBar;
    class QLabel* m_operationLabel;
    class QPushButton* m_cancelButton;

    void setupUI();
};

#endif // MAIN_APPLICATION_H