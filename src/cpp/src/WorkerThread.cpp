#include "include/WorkerThread.h"
#include "include/ForensicLogger.h"

WorkerThread::WorkerThread(QObject *parent)
    : QThread(parent)
{
}

WorkerThread::~WorkerThread()
{
    // Ensure the thread is stopped before destruction
    quit();
    wait();
}

void WorkerThread::start(std::function<void()> task)
{
    if (isRunning()) {
        logWarn("WorkerThread::start called while thread is already running.");
        return;
    }

    m_task = std::move(task);
    QThread::start();
}

void WorkerThread::run()
{
    if (!m_task) {
        logError("WorkerThread started without a task.");
        return;
    }

    try {
        // Execute the task
        m_task();
    } catch (const std::exception& e) {
        logCritical(QString("Unhandled exception in worker thread: %1").arg(e.what()));
    } catch (...) {
        logCritical("Unhandled unknown exception in worker thread.");
    }

    // Clear the task once it's done
    m_task = nullptr;
}
