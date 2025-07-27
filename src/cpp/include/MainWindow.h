#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include "DiskImager.h"
#include "FileCarver.h"

#include "CaseManager.h"

namespace Ui {
class MainWindow;
}

class PerformanceMonitor;

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

protected:
    void closeEvent(QCloseEvent *event) override;

private slots:
    // Case Actions
    void newCase();
    void openCase();
    void onCaseOpened(CaseData* caseData);
    void onCaseClosed();

    // Menu Actions
    void showAboutDialog();
    void openSettingsDialog();


    // UI Actions
    void browseSourceDevice();
    void browseDestImage();
    void startDiskImaging();
    void cancelDiskImaging();

    // File Carver Actions
    void browseCarverSource();
    void browseCarverDest();
    void startFileCarving();
    void cancelFileCarving();

    // FileCarver Signals
    void onCarvingProgress(const PhoenixDRS::CarvingProgress& progress);
    void onFileCarved(const PhoenixDRS::CarvedFile& file);
    void onCarvingFinished(bool success, const QString& message);

    // DiskImager Signals
    void onImagingProgress(const PhoenixDRS::ImagingProgress& progress);
    void onImagingFinished(bool success, const QString& message);

    // Performance
    void updatePerformanceMetrics(double cpuLoad, double memoryUsedMB);

private:
    void initApplication();
    void setupConnections();
    void setControlsEnabled(bool enabled);
    void setupCarverResultsTable();

    Ui::MainWindow *ui;
    PerformanceMonitor* m_perfMonitor;
    
    // Core components
    PhoenixDRS::DiskImager* m_diskImager;
    QThread* m_imagerThread;
    PhoenixDRS::FileCarver* m_fileCarver;
    QThread* m_carverThread;
    // RaidReconstructor is not thread-safe in this simple version
    // PhoenixDRS::RaidReconstructor* m_raidReconstructor;
    // QThread* m_raidThread;
    // VideoRebuilder is not thread-safe in this simple version
};

#endif // MAINWINDOW_H
