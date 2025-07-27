#include "include/CaseManager.h"
#include "include/ForensicLogger.h"
#include <QDir>
#include <QJsonDocument>
#include <QJsonObject>
#include <QFile>

// --- CaseData Implementation ---

CaseData::CaseData(QObject* parent) : QObject(parent) {}

QString CaseData::getLogsDirectory() const {
    return QDir(caseDirectory).filePath("Logs");
}

QString CaseData::getImagingOutputDirectory() const {
    return QDir(caseDirectory).filePath("ImagingOutput");
}

QString CaseData::getCarvingOutputDirectory() const {
    return QDir(caseDirectory).filePath("CarvingOutput");
}

bool CaseData::save(const QString& path) {
    QJsonObject caseJson;
    caseJson["caseName"] = caseName;
    caseJson["caseNumber"] = caseNumber;
    caseJson["examiner"] = examiner;
    caseJson["description"] = description;

    QFile saveFile(path);
    if (!saveFile.open(QIODevice::WriteOnly)) {
        logError(QString("Couldn't save case file to %1").arg(path));
        return false;
    }
    saveFile.write(QJsonDocument(caseJson).toJson());
    return true;
}

CaseData* CaseData::load(const QString& path) {
    QFile loadFile(path);
    if (!loadFile.open(QIODevice::ReadOnly)) {
        logError(QString("Couldn't open case file from %1").arg(path));
        return nullptr;
    }

    QJsonDocument doc = QJsonDocument::fromJson(loadFile.readAll());
    if (!doc.isObject()) return nullptr;

    QJsonObject json = doc.object();
    CaseData* data = new CaseData();
    data->caseName = json["caseName"].toString();
    data->caseNumber = json["caseNumber"].toString();
    data->examiner = json["examiner"].toString();
    data->description = json["description"].toString();
    data->caseDirectory = QFileInfo(path).absolutePath();
    
    return data;
}


// --- CaseManager Implementation ---

CaseManager::CaseManager(QObject *parent) : QObject(parent), m_currentCase(nullptr)
{
}

CaseData* CaseManager::currentCase() const {
    return m_currentCase;
}

bool CaseManager::isCaseOpen() const {
    return m_currentCase != nullptr;
}

bool CaseManager::newCase(const QString& rootDir, const QString& caseName, const QString& examiner)
{
    if (isCaseOpen()) {
        closeCase();
    }

    QDir root(rootDir);
    QString caseDirName = QString("%1_%2").arg(QDateTime::currentDateTime().toString("yyyyMMdd"), caseName);
    QString casePath = root.filePath(caseDirName);
    
    if (!QDir().mkpath(casePath)) {
        logError(QString("Failed to create case directory: %1").arg(casePath));
        return false;
    }

    m_currentCase = new CaseData(this);
    m_currentCase->caseName = caseName;
    m_currentCase->examiner = examiner;
    m_currentCase->caseDirectory = casePath;

    // Create subdirectories
    QDir().mkpath(m_currentCase->getLogsDirectory());
    QDir().mkpath(m_currentCase->getImagingOutputDirectory());
    QDir().mkpath(m_currentCase->getCarvingOutputDirectory());

    QString caseFilePath = QDir(casePath).filePath("case.json");
    if (!m_currentCase->save(caseFilePath)) {
        closeCase();
        return false;
    }

    logInfo(QString("New case '%1' created at %2").arg(caseName, casePath));
    emit caseOpened(m_currentCase);
    return true;
}

bool CaseManager::openCase(const QString& caseFilePath)
{
    if (isCaseOpen()) {
        closeCase();
    }

    CaseData* data = CaseData::load(caseFilePath);
    if (data) {
        m_currentCase = data;
        m_currentCase->setParent(this);
        logInfo(QString("Opened case '%1'").arg(m_currentCase->caseName));
        emit caseOpened(m_currentCase);
        return true;
    }
    return false;
}

bool CaseManager::closeCase()
{
    if (m_currentCase) {
        logInfo(QString("Closing case '%1'").arg(m_currentCase->caseName));
        delete m_currentCase;
        m_currentCase = nullptr;
        emit caseClosed();
    }
    return true;
}
