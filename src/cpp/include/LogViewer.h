#ifndef LOGVIEWER_H
#define LOGVIEWER_H

#include <QWidget>
#include <QPlainTextEdit>
#include "include/ForensicLogger.h"

// A simple widget to display log messages in real-time.
class LogViewer : public QWidget
{
    Q_OBJECT

public:
    explicit LogViewer(QWidget *parent = nullptr);

public slots:
    // Receives new messages from the ForensicLogger
    void onNewMessage(const QString& formattedMessage, ForensicLogger::LogLevel level);

private:
    void setupUi();
    QPlainTextEdit* m_logOutput;
};

#endif // LOGVIEWER_H
