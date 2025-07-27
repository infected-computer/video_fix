#ifndef CASEMANAGER_H
#define CASEMANAGER_H

#include <QObject>
#include <QString>
#include <QJsonObject>

// Represents the data for a single case
class CaseData : public QObject {
    Q_OBJECT
public:
    explicit CaseData(QObject* parent = nullptr);

    QString caseName;
    QString caseNumber;
    QString examiner;
    QString description;
    QString caseDirectory;

    QString getLogsDirectory() const;
    QString getImagingOutputDirectory() const;
    QString getCarvingOutputDirectory() const;

    bool save(const QString& path);
    static CaseData* load(const QString& path);
};

// Manages the lifecycle of cases
class CaseManager : public QObject
{
    Q_OBJECT

public:
    explicit CaseManager(QObject *parent = nullptr);

    CaseData* currentCase() const;
    bool isCaseOpen() const;

public slots:
    bool newCase(const QString& rootDir, const QString& caseName, const QString& examiner);
    bool openCase(const QString& caseFilePath);
    bool closeCase();

signals:
    void caseOpened(CaseData* caseData);
    void caseClosed();

private:
    CaseData* m_currentCase;
};

#endif // CASEMANAGER_H
