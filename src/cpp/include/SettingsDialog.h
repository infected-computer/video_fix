#ifndef SETTINGSDIALOG_H
#define SETTINGSDIALOG_H

#include <QDialog>
#include <QSettings>

namespace Ui {
class SettingsDialog;
}

// Manages application settings through a dialog window.
class SettingsDialog : public QDialog
{
    Q_OBJECT

public:
    explicit SettingsDialog(QWidget *parent = nullptr);
    ~SettingsDialog();

    // Static method to get a global QSettings instance
    static QSettings* getSettings();

    // Static getters for common settings
    static int maxWorkerThreads();
    static int chunkSizeKB();
    static bool useDirectIO();
    static QString tempDirectory();

protected:
    void changeEvent(QEvent *e);

private slots:
    void on_buttonBox_accepted();
    void on_browseTempDirButton_clicked();
    void on_resetButton_clicked();

private:
    void loadSettings();
    void saveSettings();
    void setDefaults();

    Ui::SettingsDialog *ui;
    static QSettings* m_settings;
};

#endif // SETTINGSDIALOG_H
