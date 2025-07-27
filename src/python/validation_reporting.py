"""
PhoenixDRS - Validation and Reporting Module
מודול אימות תוצאות ויצירת דוחות מקצועיים
"""

import os
import hashlib
import json
import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from pathlib import Path
import mimetypes


@dataclass
class FileValidationResult:
    """תוצאת אימות קובץ"""
    file_path: str
    file_size: int
    is_valid: bool
    mime_type: Optional[str]
    md5_hash: Optional[str]
    validation_errors: List[str]
    validation_timestamp: float


@dataclass
class RecoveryStatistics:
    """סטטיסטיקות שחזור"""
    total_files_found: int
    valid_files: int
    corrupted_files: int
    total_size_recovered: int
    file_types: Dict[str, int]
    recovery_timestamp: float
    processing_time: float


class FileValidator:
    """מאמת שלמות קבצים"""
    
    def __init__(self):
        self.validation_results: List[FileValidationResult] = []
    
    def validate_file(self, file_path: str) -> FileValidationResult:
        """אימות קובץ יחיד"""
        validation_errors = []
        is_valid = True
        md5_hash = None
        mime_type = None
        
        try:
            # בדיקת קיום הקובץ
            if not os.path.exists(file_path):
                validation_errors.append("הקובץ לא קיים")
                is_valid = False
                file_size = 0
            else:
                file_size = os.path.getsize(file_path)
                
                # בדיקת גודל קובץ
                if file_size == 0:
                    validation_errors.append("קובץ ריק")
                    is_valid = False
                
                # זיהוי סוג קובץ
                mime_type, _ = mimetypes.guess_type(file_path)
                
                # חישוב hash
                md5_hash = self._calculate_md5(file_path)
                
                # אימות ספציפי לסוג קובץ
                file_specific_errors = self._validate_file_type_specific(file_path, mime_type)
                validation_errors.extend(file_specific_errors)
                
                if file_specific_errors:
                    is_valid = False
        
        except Exception as e:
            validation_errors.append(f"שגיאה באימות: {str(e)}")
            is_valid = False
            file_size = 0
        
        result = FileValidationResult(
            file_path=file_path,
            file_size=file_size,
            is_valid=is_valid,
            mime_type=mime_type,
            md5_hash=md5_hash,
            validation_errors=validation_errors,
            validation_timestamp=time.time()
        )
        
        self.validation_results.append(result)
        return result
    
    def validate_directory(self, directory_path: str) -> List[FileValidationResult]:
        """אימות כל הקבצים בתיקייה"""
        results = []
        
        for root, dirs, files in os.walk(directory_path):
            for file in files:
                file_path = os.path.join(root, file)
                result = self.validate_file(file_path)
                results.append(result)
        
        return results
    
    def _calculate_md5(self, file_path: str) -> str:
        """חישוב MD5 hash"""
        hash_md5 = hashlib.md5()
        try:
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
        except:
            return None
    
    def _validate_file_type_specific(self, file_path: str, mime_type: Optional[str]) -> List[str]:
        """אימות ספציפי לסוג קובץ"""
        errors = []
        
        try:
            with open(file_path, 'rb') as f:
                header = f.read(16)
            
            # אימות תמונות JPEG
            if mime_type and 'jpeg' in mime_type.lower():
                if not header.startswith(b'\xff\xd8\xff'):
                    errors.append("header JPEG לא תקין")
                
                # בדיקת footer
                with open(file_path, 'rb') as f:
                    f.seek(-2, 2)
                    footer = f.read(2)
                    if footer != b'\xff\xd9':
                        errors.append("footer JPEG לא תקין")
            
            # אימות תמונות PNG
            elif mime_type and 'png' in mime_type.lower():
                png_signature = b'\x89PNG\r\n\x1a\n'
                if not header.startswith(png_signature):
                    errors.append("header PNG לא תקין")
            
            # אימות PDF
            elif mime_type and 'pdf' in mime_type.lower():
                if not header.startswith(b'%PDF-'):
                    errors.append("header PDF לא תקין")
            
            # אימות ZIP
            elif file_path.lower().endswith('.zip'):
                if not header.startswith(b'PK\x03\x04'):
                    errors.append("header ZIP לא תקין")
        
        except Exception as e:
            errors.append(f"שגיאה באימות ספציפי: {str(e)}")
        
        return errors


class StatisticsCollector:
    """אוסף סטטיסטיקות שחזור"""
    
    def __init__(self):
        self.start_time = time.time()
    
    def collect_statistics(self, validation_results: List[FileValidationResult]) -> RecoveryStatistics:
        """איסוף סטטיסטיקות מתוצאות האימות"""
        total_files = len(validation_results)
        valid_files = sum(1 for r in validation_results if r.is_valid)
        corrupted_files = total_files - valid_files
        total_size = sum(r.file_size for r in validation_results)
        
        # ספירת סוגי קבצים
        file_types = {}
        for result in validation_results:
            if result.mime_type:
                file_type = result.mime_type.split('/')[0]  # image, video, application, etc.
            else:
                file_type = Path(result.file_path).suffix.lower() or 'unknown'
            
            file_types[file_type] = file_types.get(file_type, 0) + 1
        
        processing_time = time.time() - self.start_time
        
        return RecoveryStatistics(
            total_files_found=total_files,
            valid_files=valid_files,
            corrupted_files=corrupted_files,
            total_size_recovered=total_size,
            file_types=file_types,
            recovery_timestamp=time.time(),
            processing_time=processing_time
        )


class ReportGenerator:
    """יוצר דוחות מקצועיים"""
    
    def __init__(self):
        pass
    
    def generate_comprehensive_report(self, validation_results: List[FileValidationResult], 
                                    statistics: RecoveryStatistics, 
                                    output_path: str, 
                                    case_info: Optional[Dict[str, Any]] = None):
        """יצירת דוח מקיף"""
        
        # יצירת דוח HTML
        html_report = self._generate_html_report(validation_results, statistics, case_info)
        html_path = output_path.replace('.txt', '.html')
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_report)
        
        # יצירת דוח טקסט
        text_report = self._generate_text_report(validation_results, statistics, case_info)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(text_report)
        
        # יצירת דוח JSON
        json_report = self._generate_json_report(validation_results, statistics, case_info)
        json_path = output_path.replace('.txt', '.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(json_report, f, ensure_ascii=False, indent=2)
        
        print(f"דוחות נוצרו:")
        print(f"  טקסט: {output_path}")
        print(f"  HTML: {html_path}")
        print(f"  JSON: {json_path}")
    
    def _generate_text_report(self, validation_results: List[FileValidationResult], 
                            statistics: RecoveryStatistics, 
                            case_info: Optional[Dict[str, Any]]) -> str:
        """יצירת דוח טקסט"""
        report = []
        report.append("=" * 80)
        report.append("PhoenixDRS - דוח שחזור מידע מקצועי")
        report.append("=" * 80)
        report.append("")
        
        # מידע על המקרה
        if case_info:
            report.append("מידע כללי:")
            for key, value in case_info.items():
                report.append(f"  {key}: {value}")
            report.append("")
        
        # סטטיסטיקות כלליות
        report.append("סטטיסטיקות כלליות:")
        report.append(f"  סה\"כ קבצים שנמצאו: {statistics.total_files_found:,}")
        report.append(f"  קבצים תקינים: {statistics.valid_files:,}")
        report.append(f"  קבצים פגומים: {statistics.corrupted_files:,}")
        report.append(f"  אחוז הצלחה: {(statistics.valid_files/statistics.total_files_found*100):.1f}%")
        report.append(f"  סה\"כ נפח שוחזר: {self._format_size(statistics.total_size_recovered)}")
        report.append(f"  זמן עיבוד: {statistics.processing_time:.1f} שניות")
        report.append("")
        
        # פילוח לפי סוגי קבצים
        report.append("פילוח לפי סוגי קבצים:")
        for file_type, count in sorted(statistics.file_types.items()):
            report.append(f"  {file_type}: {count:,} קבצים")
        report.append("")
        
        # רשימת קבצים פגומים
        corrupted_files = [r for r in validation_results if not r.is_valid]
        if corrupted_files:
            report.append("קבצים פגומים:")
            for result in corrupted_files[:50]:  # מוגבל ל-50 הראשונים
                report.append(f"  {result.file_path}")
                for error in result.validation_errors:
                    report.append(f"    - {error}")
            
            if len(corrupted_files) > 50:
                report.append(f"  ... ועוד {len(corrupted_files) - 50} קבצים")
            report.append("")
        
        # חתימה
        report.append("-" * 80)
        report.append(f"דוח נוצר ב-{time.strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("PhoenixDRS - Phoenix Data Recovery Suite")
        
        return "\n".join(report)
    
    def _generate_html_report(self, validation_results: List[FileValidationResult], 
                            statistics: RecoveryStatistics, 
                            case_info: Optional[Dict[str, Any]]) -> str:
        """יצירת דוח HTML"""
        html = f"""
<!DOCTYPE html>
<html dir="rtl" lang="he">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>PhoenixDRS - דוח שחזור מידע</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        .header {{ text-align: center; border-bottom: 2px solid #333; padding-bottom: 20px; margin-bottom: 30px; }}
        .stats {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin-bottom: 30px; }}
        .stat-card {{ background: #f8f9fa; padding: 15px; border-radius: 5px; text-align: center; }}
        .stat-number {{ font-size: 2em; font-weight: bold; color: #007bff; }}
        .file-types {{ margin-bottom: 30px; }}
        .progress-bar {{ background: #e9ecef; border-radius: 10px; overflow: hidden; height: 20px; }}
        .progress-fill {{ background: #28a745; height: 100%; transition: width 0.3s ease; }}
        .error-list {{ max-height: 400px; overflow-y: auto; background: #f8f9fa; padding: 15px; border-radius: 5px; }}
        .footer {{ text-align: center; margin-top: 30px; padding-top: 20px; border-top: 1px solid #dee2e6; color: #6c757d; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>PhoenixDRS - דוח שחזור מידע מקצועי</h1>
            <p>נוצר ב-{time.strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
        
        <div class="stats">
            <div class="stat-card">
                <div class="stat-number">{statistics.total_files_found:,}</div>
                <div>סה"כ קבצים</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">{statistics.valid_files:,}</div>
                <div>קבצים תקינים</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">{statistics.corrupted_files:,}</div>
                <div>קבצים פגומים</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">{(statistics.valid_files/statistics.total_files_found*100):.1f}%</div>
                <div>אחוז הצלחה</div>
            </div>
        </div>
        
        <div class="progress-bar">
            <div class="progress-fill" style="width: {(statistics.valid_files/statistics.total_files_found*100):.1f}%"></div>
        </div>
        
        <div class="file-types">
            <h3>פילוח לפי סוגי קבצים</h3>
            <ul>
        """
        
        for file_type, count in sorted(statistics.file_types.items()):
            html += f"<li>{file_type}: {count:,} קבצים</li>"
        
        html += """
            </ul>
        </div>
        """
        
        # קבצים פגומים
        corrupted_files = [r for r in validation_results if not r.is_valid]
        if corrupted_files:
            html += """
        <div>
            <h3>קבצים פגומים</h3>
            <div class="error-list">
            """
            for result in corrupted_files[:100]:  # מוגבל ל-100
                html += f"<p><strong>{result.file_path}</strong><br>"
                for error in result.validation_errors:
                    html += f"• {error}<br>"
                html += "</p>"
            html += "</div></div>"
        
        html += f"""
        <div class="footer">
            <p>PhoenixDRS - Phoenix Data Recovery Suite</p>
            <p>נפח כולל שוחזר: {self._format_size(statistics.total_size_recovered)}</p>
        </div>
    </div>
</body>
</html>
        """
        
        return html
    
    def _generate_json_report(self, validation_results: List[FileValidationResult], 
                            statistics: RecoveryStatistics, 
                            case_info: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """יצירת דוח JSON"""
        return {
            "report_info": {
                "generated_at": time.strftime('%Y-%m-%d %H:%M:%S'),
                "generator": "PhoenixDRS",
                "version": "1.0"
            },
            "case_info": case_info or {},
            "statistics": asdict(statistics),
            "validation_results": [asdict(result) for result in validation_results]
        }
    
    def _format_size(self, size_bytes: int) -> str:
        """עיצוב גודל קובץ"""
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.1f} PB"


def validate_and_report(directory_path: str, output_report_path: str, 
                       case_info: Optional[Dict[str, Any]] = None):
    """פונקציה מרכזית לאימות ויצירת דוח"""
    print(f"מתחיל אימות קבצים ב-{directory_path}")
    
    # אימות קבצים
    validator = FileValidator()
    validation_results = validator.validate_directory(directory_path)
    
    # איסוף סטטיסטיקות
    stats_collector = StatisticsCollector()
    statistics = stats_collector.collect_statistics(validation_results)
    
    # יצירת דוח
    report_generator = ReportGenerator()
    report_generator.generate_comprehensive_report(
        validation_results, statistics, output_report_path, case_info
    )
    
    print(f"אימות הושלם: {statistics.valid_files}/{statistics.total_files_found} קבצים תקינים")
    return validation_results, statistics


if __name__ == "__main__":
    # דוגמה לשימוש
    # validate_and_report("carved_files", "recovery_report.txt")
    print("מודול האימות והדיווח מוכן לשימוש")