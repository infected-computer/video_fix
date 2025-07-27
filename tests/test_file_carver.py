"""
PhoenixDRS File Carver Module Tests
בדיקות עבור מודול חיתוך הקבצים
"""

import pytest
import json
import os
from pathlib import Path
from unittest.mock import Mock, patch, mock_open
import mmap

from file_carver import (
    FileSignature, CarvedFile, SignatureDatabase, 
    AhoCorasickSearcher, FileCarver
)


class TestFileSignature:
    """בדיקות עבור FileSignature"""
    
    def test_file_signature_creation(self):
        """בדיקת יצירת חתימת קובץ"""
        sig = FileSignature(
            name="Test JPEG",
            extension=".jpg", 
            header="FFD8FF",
            footer="FFD9",
            max_size=1024000,
            footer_search_offset=512,
            description="Test description"
        )
        
        assert sig.name == "Test JPEG"
        assert sig.extension == ".jpg"
        assert sig.header == "FFD8FF"
        assert sig.footer == "FFD9"
        assert sig.max_size == 1024000
        assert sig.footer_search_offset == 512
        assert sig.description == "Test description"
    
    def test_header_bytes_conversion(self):
        """בדיקת המרת header לבתים"""
        sig = FileSignature(
            name="JPEG", extension=".jpg", header="FFD8FF", 
            footer="", max_size=1024, footer_search_offset=0
        )
        
        expected = bytes.fromhex("FFD8FF")
        assert sig.header_bytes() == expected
    
    def test_footer_bytes_conversion(self):
        """בדיקת המרת footer לבתים"""
        sig = FileSignature(
            name="JPEG", extension=".jpg", header="FFD8FF",
            footer="FFD9", max_size=1024, footer_search_offset=0
        )
        
        expected = bytes.fromhex("FFD9")
        assert sig.footer_bytes() == expected
    
    def test_empty_footer_bytes(self):
        """בדיקת footer ריק"""
        sig = FileSignature(
            name="JPEG", extension=".jpg", header="FFD8FF",
            footer="", max_size=1024, footer_search_offset=0
        )
        
        assert sig.footer_bytes() == b""
    
    def test_invalid_hex_header(self):
        """בדיקת header לא תקין"""
        sig = FileSignature(
            name="JPEG", extension=".jpg", header="INVALID",
            footer="", max_size=1024, footer_search_offset=0
        )
        
        with pytest.raises(ValueError):
            sig.header_bytes()


class TestCarvedFile:
    """בדיקות עבור CarvedFile"""
    
    def test_carved_file_creation(self):
        """בדיקת יצירת קובץ חתוך"""
        sig = FileSignature(
            name="JPEG", extension=".jpg", header="FFD8FF",
            footer="FFD9", max_size=1024, footer_search_offset=0
        )
        
        carved = CarvedFile(
            signature=sig,
            start_offset=1024,
            end_offset=2048,
            size=1024,
            output_path="/path/to/file.jpg",
            is_complete=True
        )
        
        assert carved.signature == sig
        assert carved.start_offset == 1024
        assert carved.end_offset == 2048
        assert carved.size == 1024
        assert carved.output_path == "/path/to/file.jpg"
        assert carved.is_complete is True


class TestSignatureDatabase:
    """בדיקות עבור SignatureDatabase"""
    
    def test_load_signatures_success(self, sample_signatures_db):
        """בדיקת טעינת חתימות מוצלחת"""
        db = SignatureDatabase(str(sample_signatures_db))
        
        assert len(db.signatures) == 3
        assert db.signatures[0].name == "JPEG Image"
        assert db.signatures[1].name == "PNG Image"
        assert db.signatures[2].name == "PDF Document"
    
    def test_load_signatures_file_not_found(self, temp_dir):
        """בדיקת טעינת חתימות - קובץ לא נמצא"""
        non_existent_file = temp_dir / "non_existent.json"
        
        with patch('builtins.print') as mock_print:
            db = SignatureDatabase(str(non_existent_file))
            assert len(db.signatures) == 0
            mock_print.assert_called()
    
    def test_load_signatures_invalid_json(self, temp_dir):
        """בדיקת טעינת חתימות - JSON לא תקין"""
        invalid_json_file = temp_dir / "invalid.json"
        invalid_json_file.write_text("invalid json content")
        
        with patch('builtins.print') as mock_print:
            db = SignatureDatabase(str(invalid_json_file))
            assert len(db.signatures) == 0
            mock_print.assert_called()
    
    def test_get_all_headers(self, sample_signatures_db):
        """בדיקת קבלת כל ה-headers"""
        db = SignatureDatabase(str(sample_signatures_db))
        headers = db.get_all_headers()
        
        assert len(headers) == 3
        assert bytes.fromhex("FFD8FF") in headers
        assert bytes.fromhex("89504E470D0A1A0A") in headers
        assert bytes.fromhex("255044462D") in headers
        
        # בדיקת מיפוי נכון
        jpeg_sig = headers[bytes.fromhex("FFD8FF")]
        assert jpeg_sig.name == "JPEG Image"


class TestAhoCorasickSearcher:
    """בדיקות עבור AhoCorasickSearcher"""
    
    def test_searcher_creation(self):
        """בדיקת יצירת מחפש"""
        patterns = [b"test", b"pattern", b"search"]
        searcher = AhoCorasickSearcher(patterns)
        
        assert searcher.patterns == patterns
        assert len(searcher.pattern_map) == 3
    
    def test_single_pattern_search(self):
        """בדיקת חיפוש דפוס יחיד"""
        patterns = [b"test"]
        searcher = AhoCorasickSearcher(patterns)
        data = b"this is a test string"
        
        results = list(searcher.search(data))
        assert len(results) == 1
        assert results[0] == (10, b"test")
    
    def test_multiple_patterns_search(self):
        """בדיקת חיפוש מרובה דפוסים"""
        patterns = [b"test", b"string"]
        searcher = AhoCorasickSearcher(patterns)
        data = b"this is a test string"
        
        results = list(searcher.search(data))
        assert len(results) == 2
        
        # מיון לפי מיקום
        results.sort(key=lambda x: x[0])
        assert results[0] == (10, b"test")
        assert results[1] == (15, b"string")
    
    def test_overlapping_patterns(self):
        """בדיקת דפוסים חופפים"""
        patterns = [b"abc", b"bcd"]
        searcher = AhoCorasickSearcher(patterns)
        data = b"abcd"
        
        results = list(searcher.search(data))
        assert len(results) == 2
        
        results.sort(key=lambda x: x[0])
        assert results[0] == (0, b"abc")
        assert results[1] == (1, b"bcd")
    
    def test_no_matches(self):
        """בדיקת אי מציאת תוצאות"""
        patterns = [b"notfound"]
        searcher = AhoCorasickSearcher(patterns)
        data = b"this string has no matches"
        
        results = list(searcher.search(data))
        assert len(results) == 0


class TestFileCarver:
    """בדיקות עבור FileCarver"""
    
    def test_carver_initialization(self):
        """בדיקת אתחול המחתך"""
        carver = FileCarver(chunk_size=2048)
        
        assert carver.chunk_size == 2048
        assert carver.carved_files == []
    
    def test_carver_default_chunk_size(self):
        """בדיקת גודל chunk ברירת מחדל"""
        carver = FileCarver()
        
        assert carver.chunk_size == 1024 * 1024  # 1MB
    
    @pytest.mark.integration
    def test_carve_basic_functionality(self, sample_disk_image, sample_signatures_db, temp_dir):
        """בדיקת פונקציונליות חיתוך בסיסית"""
        output_dir = temp_dir / "carved_output"
        
        carver = FileCarver(chunk_size=1024)  # Small chunks for testing
        carved_files = carver.carve(
            str(sample_disk_image),
            str(sample_signatures_db), 
            str(output_dir)
        )
        
        # בדיקת תוצאות
        assert len(carved_files) >= 3  # JPEG, PNG, PDF
        assert output_dir.exists()
        
        # בדיקת יצירת דוח
        report_file = output_dir / "carving_report.txt"
        assert report_file.exists()
        
        # בדיקת קיום קبצים שנחתכו
        carved_file_paths = [Path(cf.output_path) for cf in carved_files]
        for path in carved_file_paths:
            assert path.exists()
            assert path.stat().st_size > 0
    
    @pytest.mark.integration 
    def test_carve_with_nonexistent_image(self, sample_signatures_db, temp_dir):
        """בדיקת חיתוך עם תמונה לא קיימת"""
        non_existent = temp_dir / "non_existent.dd"
        output_dir = temp_dir / "carved_output"
        
        carver = FileCarver()
        
        with pytest.raises(FileNotFoundError):
            carver.carve(str(non_existent), str(sample_signatures_db), str(output_dir))
    
    @pytest.mark.integration
    def test_carve_with_invalid_signatures(self, sample_disk_image, temp_dir):
        """בדיקת חיתוך עם חתימות לא תקינות"""
        invalid_db = temp_dir / "invalid.json"
        invalid_db.write_text("invalid json")
        output_dir = temp_dir / "carved_output"
        
        carver = FileCarver()
        carved_files = carver.carve(
            str(sample_disk_image),
            str(invalid_db),
            str(output_dir)
        )
        
        # צריך לקבל רשימה ריקה כי לא נטענו חתימות
        assert len(carved_files) == 0
    
    def test_carve_single_file_with_footer(self, temp_dir):
        """בדיקת חיתוך קובץ יחיד עם footer"""
        # יצירת נתונים מדומים
        test_data = b'\x00' * 1000 + bytes.fromhex('FFD8FF') + b'jpeg data' + bytes.fromhex('FFD9') + b'\x00' * 1000
        
        carver = FileCarver()
        
        # Mock של mmap
        mock_mmap = Mock()
        mock_mmap.__len__ = Mock(return_value=len(test_data))
        mock_mmap.__getitem__ = Mock(side_effect=lambda key: test_data[key])
        mock_mmap.find = Mock(return_value=test_data.find(bytes.fromhex('FFD9')))
        
        sig = FileSignature(
            name="JPEG", extension=".jpg", header="FFD8FF",
            footer="FFD9", max_size=10000, footer_search_offset=100
        )
        
        with patch('builtins.open', mock_open()):
            carved_file = carver._carve_single_file(
                mock_mmap, 1003, sig, str(temp_dir), 1  # 1003 is position of FFD8FF
            )
        
        assert carved_file is not None
        assert carved_file.is_complete is True
        assert carved_file.start_offset == 1003
        assert carved_file.signature == sig
    
    def test_carve_single_file_without_footer(self, temp_dir):
        """בדיקת חיתוך קובץ יחיד ללא footer"""
        test_data = b'\x00' * 1000 + bytes.fromhex('FFD8FF') + b'jpeg data without footer' + b'\x00' * 1000
        
        carver = FileCarver()
        
        mock_mmap = Mock()
        mock_mmap.__len__ = Mock(return_value=len(test_data))
        mock_mmap.__getitem__ = Mock(side_effect=lambda key: test_data[key])
        mock_mmap.find = Mock(return_value=-1)  # Footer not found
        
        sig = FileSignature(
            name="JPEG", extension=".jpg", header="FFD8FF",
            footer="FFD9", max_size=5000, footer_search_offset=100
        )
        
        with patch('builtins.open', mock_open()):
            carved_file = carver._carve_single_file(
                mock_mmap, 1003, sig, str(temp_dir), 1
            )
        
        assert carved_file is not None
        assert carved_file.is_complete is False
        assert carved_file.size == 5000  # Should use max_size
    
    def test_carve_single_file_no_footer_signature(self, temp_dir):
        """בדיקת חיתוך קובץ עם חתימה ללא footer"""
        test_data = b'\x00' * 1000 + bytes.fromhex('504B0304') + b'zip data' + b'\x00' * 1000
        
        carver = FileCarver()
        
        mock_mmap = Mock()
        mock_mmap.__len__ = Mock(return_value=len(test_data))
        mock_mmap.__getitem__ = Mock(side_effect=lambda key: test_data[key])
        
        sig = FileSignature(
            name="ZIP", extension=".zip", header="504B0304",
            footer="", max_size=3000, footer_search_offset=0  # No footer
        )
        
        with patch('builtins.open', mock_open()):
            carved_file = carver._carve_single_file(
                mock_mmap, 1003, sig, str(temp_dir), 1
            )
        
        assert carved_file is not None
        assert carved_file.is_complete is True  # Should be complete without footer
        assert carved_file.size == 3000
    
    def test_generate_report(self, temp_dir):
        """בדיקת יצירת דוח"""
        carver = FileCarver()
        
        # הוספת קבצים מדומים
        sig1 = FileSignature("JPEG", ".jpg", "FFD8FF", "FFD9", 1024, 100)
        sig2 = FileSignature("PNG", ".png", "89504E47", "49454E44", 2048, 200)
        
        carver.carved_files = [
            CarvedFile(sig1, 0, 1000, 1000, "file1.jpg", True),
            CarvedFile(sig1, 2000, 3000, 1000, "file2.jpg", False),
            CarvedFile(sig2, 4000, 6000, 2000, "file3.png", True)
        ]
        
        carver._generate_report(str(temp_dir))
        
        report_path = temp_dir / "carving_report.txt"
        assert report_path.exists()
        
        report_content = report_path.read_text(encoding='utf-8')
        assert "3" in report_content  # Total files
        assert "2" in report_content  # Complete files  
        assert "1" in report_content  # Partial files
        assert "JPEG: 2" in report_content
        assert "PNG: 1" in report_content


@pytest.mark.performance
class TestFileCarverPerformance:
    """בדיקות ביצועים עבור FileCarver"""
    
    def test_large_file_carving_performance(self, temp_dir, benchmark):
        """בדיקת ביצועים על קובץ גדול"""
        # יצירת קובץ גדול מדומה (10MB)
        large_image = temp_dir / "large_test.dd"
        with open(large_image, 'wb') as f:
            # יצירת נתונים עם כמה headers מפוזרים
            for i in range(100):
                f.write(b'\x00' * 102400)  # 100KB padding
                if i % 10 == 0:  # כל 10 חלקים הוסף header
                    f.write(bytes.fromhex('FFD8FF'))
                    f.write(b'jpeg data' * 100)
                    f.write(bytes.fromhex('FFD9'))
        
        # יצירת DB חתימות פשוט
        simple_db = temp_dir / "simple.json"
        with open(simple_db, 'w') as f:
            json.dump({
                "signatures": [{
                    "name": "JPEG", "extension": ".jpg", "header": "FFD8FF",
                    "footer": "FFD9", "max_size": 50000, "footer_search_offset": 1000
                }]
            }, f)
        
        output_dir = temp_dir / "performance_output"
        carver = FileCarver(chunk_size=1024*1024)  # 1MB chunks
        
        def carve_large_file():
            return carver.carve(str(large_image), str(simple_db), str(output_dir))
        
        result = benchmark(carve_large_file)
        
        # בדיקת תוצאות
        assert len(result) >= 10  # Should find at least 10 JPEG files
        assert output_dir.exists()


@pytest.mark.slow
class TestFileCarverIntegration:
    """בדיקות אינטגרציה מקיפות"""
    
    def test_full_workflow_with_multiple_file_types(self, temp_dir):
        """בדיקת זרימת עבודה מלאה עם סוגי קבצים מרובים"""
        # יצירת תמונת דיסק מורכבת
        complex_image = temp_dir / "complex_test.dd"
        with open(complex_image, 'wb') as f:
            # JPEG files
            for i in range(3):
                f.write(b'\x00' * 512)
                f.write(bytes.fromhex('FFD8FF'))
                f.write(f'jpeg content {i}'.encode() * 50)
                f.write(bytes.fromhex('FFD9'))
            
            # PNG files
            for i in range(2):
                f.write(b'\x00' * 1024)
                f.write(bytes.fromhex('89504E470D0A1A0A'))
                f.write(f'png content {i}'.encode() * 100)
                f.write(bytes.fromhex('49454E44AE426082'))
            
            # PDF files
            f.write(b'\x00' * 2048)
            f.write(bytes.fromhex('255044462D'))
            f.write(b'pdf content' * 200)
            f.write(bytes.fromhex('0A2525454F46'))
        
        # יצירת DB מקיף
        comprehensive_db = temp_dir / "comprehensive.json"
        with open(comprehensive_db, 'w', encoding='utf-8') as f:
            json.dump({
                "signatures": [
                    {
                        "name": "JPEG Image", "extension": ".jpg", "header": "FFD8FF",
                        "footer": "FFD9", "max_size": 10485760, "footer_search_offset": 1024
                    },
                    {
                        "name": "PNG Image", "extension": ".png", "header": "89504E470D0A1A0A", 
                        "footer": "49454E44AE426082", "max_size": 52428800, "footer_search_offset": 2048
                    },
                    {
                        "name": "PDF Document", "extension": ".pdf", "header": "255044462D",
                        "footer": "0A2525454F46", "max_size": 104857600, "footer_search_offset": 4096
                    }
                ]
            }, f, ensure_ascii=False, indent=2)
        
        output_dir = temp_dir / "comprehensive_output"
        carver = FileCarver()
        
        carved_files = carver.carve(
            str(complex_image), str(comprehensive_db), str(output_dir)
        )
        
        # בדיקת תוצאות מקיפה
        assert len(carved_files) == 6  # 3 JPEG + 2 PNG + 1 PDF
        
        # בדיקת סוגי קבצים
        jpeg_count = sum(1 for cf in carved_files if cf.signature.name == "JPEG Image")
        png_count = sum(1 for cf in carved_files if cf.signature.name == "PNG Image")
        pdf_count = sum(1 for cf in carved_files if cf.signature.name == "PDF Document")
        
        assert jpeg_count == 3
        assert png_count == 2
        assert pdf_count == 1
        
        # בדיקת שכל הקבצים נוצרו בפועל
        for carved_file in carved_files:
            assert Path(carved_file.output_path).exists()
            assert Path(carved_file.output_path).stat().st_size > 0
        
        # בדיקת דוח
        report_path = output_dir / "carving_report.txt"
        assert report_path.exists()
        
        report_content = report_path.read_text(encoding='utf-8')
        assert "6" in report_content  # Total files
        assert "JPEG Image: 3" in report_content
        assert "PNG Image: 2" in report_content  
        assert "PDF Document: 1" in report_content