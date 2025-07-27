/*
 * PhoenixDRS Professional - Signature Database Implementation
 * מימוש מסד נתוני חתימות מקצועי
 */

#include "FileCarver.h"
#include "ForensicLogger.h"
#include <QJsonDocument>
#include <QJsonObject>
#include <QJsonArray>
#include <QFile>
#include <QTextStream>
#include <QMutexLocker>

namespace PhoenixDRS {

/*
 * SignatureDatabase Constructor
 */
SignatureDatabase::SignatureDatabase()
{
    // Constructor implementation
}

/*
 * SignatureDatabase Destructor
 */
SignatureDatabase::~SignatureDatabase() = default;

/*
 * Load from File
 */
bool SignatureDatabase::loadFromFile(const QString& path)
{
    QFile file(path);
    if (!file.open(QIODevice::ReadOnly)) {
        return false;
    }
    
    QJsonDocument doc = QJsonDocument::fromJson(file.readAll());
    return loadFromJson(doc);
}

/*
 * Save to File
 */
bool SignatureDatabase::saveToFile(const QString& path)
{
    QFile file(path);
    if (!file.open(QIODevice::WriteOnly)) {
        return false;
    }
    
    QJsonDocument doc = saveToJson();
    return file.write(doc.toJson()) > 0;
}

/*
 * Load from JSON
 */
bool SignatureDatabase::loadFromJson(const QJsonDocument& doc)
{
    QMutexLocker locker(&m_mutex);
    
    if (!doc.isObject()) {
        return false;
    }
    
    QJsonObject root = doc.object();
    QJsonArray signatures = root["signatures"].toArray();
    
    m_signatures.clear();
    
    for (const QJsonValue& value : signatures) {
        QJsonObject sigObj = value.toObject();
        
        FileSignature signature;
        signature.name = sigObj["name"].toString();
        signature.extension = sigObj["extension"].toString();
        signature.headerSignature = QByteArray::fromHex(sigObj["header"].toString().toUtf8());
        signature.footerSignature = QByteArray::fromHex(sigObj["footer"].toString().toUtf8());
        signature.maxFileSize = sigObj["maxSize"].toVariant().toLongLong();
        signature.minFileSize = sigObj["minSize"].toVariant().toLongLong();
        signature.headerOffset = sigObj["headerOffset"].toInt();
        signature.footerOffset = sigObj["footerOffset"].toInt();
        signature.mimeType = sigObj["mimeType"].toString();
        signature.category = sigObj["category"].toString();
        signature.priority = sigObj["priority"].toDouble();
        signature.requiresFooter = sigObj["requiresFooter"].toBool();
        signature.supportsFragmentation = sigObj["supportsFragmentation"].toBool();
        
        m_signatures[signature.name] = signature;
    }
    
    return true;
}

/*
 * Save to JSON
 */
QJsonDocument SignatureDatabase::saveToJson() const
{
    QMutexLocker locker(&m_mutex);
    
    QJsonObject root;
    QJsonArray signatures;
    
    for (const auto& [name, signature] : m_signatures) {
        QJsonObject sigObj;
        sigObj["name"] = signature.name;
        sigObj["extension"] = signature.extension;
        sigObj["header"] = QString::fromUtf8(signature.headerSignature.toHex());
        sigObj["footer"] = QString::fromUtf8(signature.footerSignature.toHex());
        sigObj["maxSize"] = QJsonValue::fromVariant(signature.maxFileSize);
        sigObj["minSize"] = QJsonValue::fromVariant(signature.minFileSize);
        sigObj["headerOffset"] = signature.headerOffset;
        sigObj["footerOffset"] = signature.footerOffset;
        sigObj["mimeType"] = signature.mimeType;
        sigObj["category"] = signature.category;
        sigObj["priority"] = signature.priority;
        sigObj["requiresFooter"] = signature.requiresFooter;
        sigObj["supportsFragmentation"] = signature.supportsFragmentation;
        
        signatures.append(sigObj);
    }
    
    root["signatures"] = signatures;
    root["version"] = "1.0";
    root["generator"] = "PhoenixDRS Professional";
    
    return QJsonDocument(root);
}

/*
 * Add Signature
 */
void SignatureDatabase::addSignature(const FileSignature& signature)
{
    QMutexLocker locker(&m_mutex);
    m_signatures[signature.name] = signature;
}

/*
 * Remove Signature
 */
void SignatureDatabase::removeSignature(const QString& name)
{
    QMutexLocker locker(&m_mutex);
    m_signatures.erase(name);
}

/*
 * Get Signature
 */
FileSignature SignatureDatabase::getSignature(const QString& name) const
{
    QMutexLocker locker(&m_mutex);
    auto it = m_signatures.find(name);
    return (it != m_signatures.end()) ? it->second : FileSignature();
}

/*
 * Get All Signatures
 */
std::vector<FileSignature> SignatureDatabase::getAllSignatures() const
{
    QMutexLocker locker(&m_mutex);
    std::vector<FileSignature> result;
    result.reserve(m_signatures.size());
    
    for (const auto& [name, signature] : m_signatures) {
        result.push_back(signature);
    }
    
    return result;
}

/*
 * Get Signatures by Category
 */
std::vector<FileSignature> SignatureDatabase::getSignaturesByCategory(const QString& category) const
{
    QMutexLocker locker(&m_mutex);
    std::vector<FileSignature> result;
    
    for (const auto& [name, signature] : m_signatures) {
        if (signature.category == category) {
            result.push_back(signature);
        }
    }
    
    return result;
}

/*
 * Clear Database
 */
void SignatureDatabase::clear()
{
    QMutexLocker locker(&m_mutex);
    m_signatures.clear();
}

/*
 * Get Size
 */
size_t SignatureDatabase::size() const
{
    QMutexLocker locker(&m_mutex);
    return m_signatures.size();
}

/*
 * Check if Empty
 */
bool SignatureDatabase::isEmpty() const
{
    QMutexLocker locker(&m_mutex);
    return m_signatures.empty();
}

/*
 * Load Default Signatures
 */
void SignatureDatabase::loadDefaultSignatures()
{
    clear();
    
    loadImageSignatures();
    loadVideoSignatures();
    loadAudioSignatures();
    loadDocumentSignatures();
    loadArchiveSignatures();
    loadExecutableSignatures();
    
    ForensicLogger::instance().info("default_signatures_loaded", "signature_database",
                                   QStringLiteral("Loaded %1 default signatures").arg(size()));
}

/*
 * Create Image Signature Helper
 */
FileSignature SignatureDatabase::createImageSignature(const QString& name, const QString& ext,
                                                    const QByteArray& header, const QByteArray& footer,
                                                    qint64 maxSize)
{
    FileSignature sig;
    sig.name = name;
    sig.extension = ext;
    sig.headerSignature = header;
    sig.footerSignature = footer;
    sig.maxFileSize = maxSize;
    sig.minFileSize = header.size();
    sig.category = "Image";
    sig.priority = 0.8;
    sig.requiresFooter = !footer.isEmpty();
    sig.supportsFragmentation = false;
    sig.mimeType = QString("image/%1").arg(ext.toLower());
    
    return sig;
}

/*
 * Create Video Signature Helper
 */
FileSignature SignatureDatabase::createVideoSignature(const QString& name, const QString& ext,
                                                    const QByteArray& header, const QByteArray& footer,
                                                    qint64 maxSize)
{
    FileSignature sig;
    sig.name = name;
    sig.extension = ext;
    sig.headerSignature = header;
    sig.footerSignature = footer;
    sig.maxFileSize = maxSize;
    sig.minFileSize = header.size();
    sig.category = "Video";
    sig.priority = 0.9;
    sig.requiresFooter = !footer.isEmpty();
    sig.supportsFragmentation = true;
    sig.mimeType = QString("video/%1").arg(ext.toLower());
    
    return sig;
}

/*
 * Load Image Signatures
 */
void SignatureDatabase::loadImageSignatures()
{
    // JPEG
    addSignature(createImageSignature("JPEG Image", "jpg", 
                                    QByteArray::fromHex("FFD8FF"), 
                                    QByteArray::fromHex("FFD9")));
    
    // PNG
    addSignature(createImageSignature("PNG Image", "png",
                                    QByteArray::fromHex("89504E470D0A1A0A")));
    
    // GIF87a
    addSignature(createImageSignature("GIF87a Image", "gif",
                                    QByteArray::fromHex("474946383761")));
    
    // GIF89a
    addSignature(createImageSignature("GIF89a Image", "gif",
                                    QByteArray::fromHex("474946383961")));
    
    // BMP
    addSignature(createImageSignature("BMP Image", "bmp",
                                    QByteArray::fromHex("424D")));
    
    // TIFF (Intel)
    addSignature(createImageSignature("TIFF Image (Intel)", "tif",
                                    QByteArray::fromHex("49492A00")));
    
    // TIFF (Motorola)
    addSignature(createImageSignature("TIFF Image (Motorola)", "tif",
                                    QByteArray::fromHex("4D4D002A")));
    
    // WEBP
    addSignature(createImageSignature("WebP Image", "webp",
                                    QByteArray::fromHex("52494646") + QByteArray(4, 0) + QByteArray::fromHex("57454250")));
    
    // Adobe Photoshop
    addSignature(createImageSignature("Adobe Photoshop", "psd",
                                    QByteArray::fromHex("38425053")));
    
    // Canon RAW (CR2)
    addSignature(createImageSignature("Canon RAW", "cr2",
                                    QByteArray::fromHex("49492A00") + QByteArray(6, 0) + QByteArray::fromHex("435232")));
    
    // Nikon RAW (NEF)
    addSignature(createImageSignature("Nikon RAW", "nef",
                                    QByteArray::fromHex("4D4D002A")));
    
    // Sony RAW (ARW)
    addSignature(createImageSignature("Sony RAW", "arw",
                                    QByteArray::fromHex("49492A00")));
    
    // HEIF/HEIC
    addSignature(createImageSignature("HEIF Image", "heic",
                                    QByteArray(4, 0) + QByteArray::fromHex("6674797068656963")));
}

/*
 * Load Video Signatures
 */
void SignatureDatabase::loadVideoSignatures()
{
    // MP4/MOV
    addSignature(createVideoSignature("MP4 Video", "mp4",
                                    QByteArray(4, 0) + QByteArray::fromHex("66747970")));
    
    // Canon MOV
    addSignature(createVideoSignature("Canon MOV", "mov",
                                    QByteArray(4, 0) + QByteArray::fromHex("6674797071742020")));
    
    // AVI
    addSignature(createVideoSignature("AVI Video", "avi",
                                    QByteArray::fromHex("52494646") + QByteArray(4, 0) + QByteArray::fromHex("41564920")));
    
    // WMV/ASF
    addSignature(createVideoSignature("Windows Media Video", "wmv",
                                    QByteArray::fromHex("3026B2758E66CF11A6D900AA0062CE6C")));
    
    // FLV
    addSignature(createVideoSignature("Flash Video", "flv",
                                    QByteArray::fromHex("464C5601")));
    
    // MKV/WebM
    addSignature(createVideoSignature("Matroska Video", "mkv",
                                    QByteArray::fromHex("1A45DFA3")));
    
    // 3GP
    addSignature(createVideoSignature("3GP Video", "3gp",
                                    QByteArray(4, 0) + QByteArray::fromHex("6674797033677035")));
    
    // MPEG
    addSignature(createVideoSignature("MPEG Video", "mpg",
                                    QByteArray::fromHex("000001B3")));
    
    // M4V
    addSignature(createVideoSignature("iTunes Video", "m4v",
                                    QByteArray(4, 0) + QByteArray::fromHex("6674797069736F6D")));
    
    // VOB
    addSignature(createVideoSignature("DVD Video", "vob",
                                    QByteArray::fromHex("000001BA")));
}

/*
 * Load Audio Signatures
 */
void SignatureDatabase::loadAudioSignatures()
{
    // MP3
    addSignature(createImageSignature("MP3 Audio", "mp3",
                                    QByteArray::fromHex("494433"), QByteArray(), 50 * 1024 * 1024));
    
    // MP3 (alternative)
    addSignature(createImageSignature("MP3 Audio (Alt)", "mp3",
                                    QByteArray::fromHex("FFFB"), QByteArray(), 50 * 1024 * 1024));
    
    // WAV
    addSignature(createImageSignature("WAV Audio", "wav",
                                    QByteArray::fromHex("52494646") + QByteArray(4, 0) + QByteArray::fromHex("57415645")));
    
    // FLAC
    addSignature(createImageSignature("FLAC Audio", "flac",
                                    QByteArray::fromHex("664C6143")));
    
    // OGG
    addSignature(createImageSignature("OGG Audio", "ogg",
                                    QByteArray::fromHex("4F676753")));
    
    // M4A
    addSignature(createImageSignature("M4A Audio", "m4a",
                                    QByteArray(4, 0) + QByteArray::fromHex("6674797069736F6D")));
    
    // AAC
    addSignature(createImageSignature("AAC Audio", "aac",
                                    QByteArray::fromHex("FFF1")));
    
    // WMA
    addSignature(createImageSignature("Windows Media Audio", "wma",
                                    QByteArray::fromHex("3026B2758E66CF11A6D900AA0062CE6C")));
}

/*
 * Load Document Signatures
 */
void SignatureDatabase::loadDocumentSignatures()
{
    // PDF
    addSignature(createImageSignature("PDF Document", "pdf",
                                    QByteArray::fromHex("255044462D"), 
                                    QByteArray::fromHex("0A2525454F46"),
                                    500 * 1024 * 1024));
    
    // Microsoft Office (DOCX, XLSX, PPTX)
    addSignature(createImageSignature("MS Office 2007+", "docx",
                                    QByteArray::fromHex("504B0304"), QByteArray(), 100 * 1024 * 1024));
    
    // Microsoft Word 97-2003
    addSignature(createImageSignature("MS Word 97-2003", "doc",
                                    QByteArray::fromHex("D0CF11E0A1B11AE1"), QByteArray(), 50 * 1024 * 1024));
    
    // RTF
    addSignature(createImageSignature("Rich Text Format", "rtf",
                                    QByteArray::fromHex("7B5C727466")));
    
    // Plain Text
    addSignature(createImageSignature("Text Document", "txt",
                                    QByteArray(), QByteArray(), 10 * 1024 * 1024));
    
    // HTML
    addSignature(createImageSignature("HTML Document", "html",
                                    QByteArray::fromHex("3C21444F43545950")));
    
    // XML
    addSignature(createImageSignature("XML Document", "xml",
                                    QByteArray::fromHex("3C3F786D6C")));
    
    // PostScript
    addSignature(createImageSignature("PostScript", "ps",
                                    QByteArray::fromHex("25215053")));
}

/*
 * Load Archive Signatures
 */
void SignatureDatabase::loadArchiveSignatures()
{
    // ZIP
    addSignature(createImageSignature("ZIP Archive", "zip",
                                    QByteArray::fromHex("504B0304"), 
                                    QByteArray::fromHex("504B0506"),
                                    2LL * 1024 * 1024 * 1024));
    
    // RAR
    addSignature(createImageSignature("RAR Archive", "rar",
                                    QByteArray::fromHex("526172211A0700")));
    
    // 7-Zip
    addSignature(createImageSignature("7-Zip Archive", "7z",
                                    QByteArray::fromHex("377ABCAF271C")));
    
    // GZIP
    addSignature(createImageSignature("GZIP Archive", "gz",
                                    QByteArray::fromHex("1F8B")));
    
    // TAR
    addSignature(createImageSignature("TAR Archive", "tar",
                                    QByteArray::fromHex("7573746172003030"), QByteArray(), 
                                    1LL * 1024 * 1024 * 1024));
    
    // CAB
    addSignature(createImageSignature("Cabinet Archive", "cab",
                                    QByteArray::fromHex("4D534346")));
    
    // ISO
    addSignature(createImageSignature("ISO Image", "iso",
                                    QByteArray::fromHex("4344303031"), QByteArray(),
                                    4LL * 1024 * 1024 * 1024));
}

/*
 * Load Executable Signatures
 */
void SignatureDatabase::loadExecutableSignatures()
{
    // Windows PE
    addSignature(createImageSignature("Windows Executable", "exe",
                                    QByteArray::fromHex("4D5A"), QByteArray(), 100 * 1024 * 1024));
    
    // Windows DLL
    addSignature(createImageSignature("Windows DLL", "dll",
                                    QByteArray::fromHex("4D5A"), QByteArray(), 50 * 1024 * 1024));
    
    // Linux ELF
    addSignature(createImageSignature("Linux Executable", "elf",
                                    QByteArray::fromHex("7F454C46")));
    
    // macOS Mach-O (32-bit)
    addSignature(createImageSignature("macOS Executable (32-bit)", "macho",
                                    QByteArray::fromHex("FEEDFACE")));
    
    // macOS Mach-O (64-bit)
    addSignature(createImageSignature("macOS Executable (64-bit)", "macho",
                                    QByteArray::fromHex("FEEDFACF")));
    
    // Java Class
    addSignature(createImageSignature("Java Class", "class",
                                    QByteArray::fromHex("CAFEBABE")));
    
    // Android APK (ZIP-based)
    addSignature(createImageSignature("Android APK", "apk",
                                    QByteArray::fromHex("504B0304"), QByteArray(), 100 * 1024 * 1024));
}

} // namespace PhoenixDRS