#ifndef RAIDRECONSTRUCTOR_H
#define RAIDRECONSTRUCTOR_H

#include <QObject>
#include <QVector>
#include <QStringList>
#include <atomic>
#include "Common.h" // Assuming Common.h has TaskStatus and RaidType

// Use the advanced structures from the existing header
namespace PhoenixDRS {

// Forward-declare the full-featured classes
class RaidReconstructor;

} // namespace PhoenixDRS

// For simplicity in connecting to the UI, we can use a simplified version
// or alias the complex one. Let's stick to a simpler interface for now
// that can be backed by the complex implementation.

// Defines the parity layout for RAID 5/6
enum class RaidParityLayout {
    LeftAsymmetric,
    LeftSymmetric,
    RightAsymmetric,
    RightSymmetric
};

// Extends RaidConfiguration for more complex RAID levels
struct RaidConfig {
    RaidType type = RaidType::Unknown;
    qint64 stripeSize = 64 * 1024; // 64KB default
    QStringList diskPaths;
    QVector<int> diskOrder;
    RaidParityLayout parityLayout = RaidParityLayout::LeftSymmetric;
    int missingDiskIndex = -1; // Index of the failed/missing disk
};

// Reconstructs various RAID levels from disk images
class RaidReconstructor : public QObject
{
    Q_OBJECT

public:
    explicit RaidReconstructor(QObject *parent = nullptr);

    void setConfiguration(const RaidConfig& config);
    void cancel();

public slots:
    void reconstruct(const QString& outputImagePath);

signals:
    void progressChanged(int percentage);
    void statusChanged(TaskStatus status);
    void reconstructionFinished(bool success, const QString& message);

private:
    bool performReconstruction();
    bool reconstructRaid0(QFile& outputFile, QVector<QFile*>& sourceFiles);
    bool reconstructRaid5(QFile& outputFile, QVector<QFile*>& sourceFiles);
    void xorBlocks(const QVector<QByteArray>& sources, QByteArray& dest);

    void setStatus(TaskStatus status);
    QString raidTypeToString(RaidType type);

    RaidConfig m_config;
    QString m_outputImagePath;
    
    TaskStatus m_status;
    int m_progress;
    std::atomic<bool> m_isCancelled;
};

#endif // RAIDRECONSTRUCTOR_H
