#include "include/LogViewer.h"
#include <QVBoxLayout>
#include <QApplication>
#include <QPalette>

LogViewer::LogViewer(QWidget *parent) : QWidget(parent)
{
    setupUi();

    // Connect to the logger's signal
    connect(ForensicLogger::instance(), &ForensicLogger::newMessage,
            this, &LogViewer::onNewMessage, Qt::QueuedConnection);
}

void LogViewer::setupUi()
{
    auto* layout = new QVBoxLayout(this);
    layout->setContentsMargins(0, 0, 0, 0);

    m_logOutput = new QPlainTextEdit(this);
    m_logOutput->setReadOnly(true);
    m_logOutput->setLineWrapMode(QPlainTextEdit::NoWrap);

    QFont font("Monospace");
    font.setStyleHint(QFont::TypeWriter);
    m_logOutput->setFont(font);

    // Set a dark theme for the log viewer
    QPalette p = m_logOutput->palette();
    p.setColor(QPalette::Base, QColor(30, 30, 30));
    p.setColor(QPalette::Text, QColor(220, 220, 220));
    m_logOutput->setPalette(p);

    layout->addWidget(m_logOutput);
    setLayout(layout);
}

void LogViewer::onNewMessage(const QString& formattedMessage, ForensicLogger::LogLevel level)
{
    QColor color;
    switch (level) {
        case ForensicLogger::LevelDebug:
            color = Qt::gray;
            break;
        case ForensicLogger::LevelInfo:
            color = Qt::white;
            break;
        case ForensicLogger::LevelWarn:
            color = Qt::yellow;
            break;
        case ForensicLogger::LevelError:
        case ForensicLogger::LevelCritical:
            color = Qt::red;
            break;
        case ForensicLogger::LevelEvidence:
            color = Qt::cyan;
            break;
    }

    // The message is already formatted, but we can add color
    // This requires appending HTML
    QString htmlMessage = QString("<font color='%1'>%2</font>")
        .arg(color.name())
        .arg(Qt::toHtmlEscaped(formattedMessage));

    m_logOutput->appendHtml(htmlMessage);
}
