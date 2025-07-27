"""
PhoenixDRS Test Configuration and Fixtures
תצורה ו-fixtures עבור בדיקות PhoenixDRS
"""

import pytest
import tempfile
import os
from pathlib import Path
from typing import Generator
import json


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """יצירת תיקייה זמנית עבור בדיקות"""
    with tempfile.TemporaryDirectory() as temp_path:
        yield Path(temp_path)


@pytest.fixture
def sample_disk_image(temp_dir: Path) -> Path:
    """יצירת תמונת דיסק מדומה עבור בדיקות"""
    image_path = temp_dir / "test_disk.dd"
    
    # יצירת תמונת דיסק מדומה עם נתונים מדומים
    with open(image_path, 'wb') as f:
        # JPEG header
        f.write(bytes.fromhex('FFD8FF'))
        f.write(b'fake jpeg data' * 100)
        f.write(bytes.fromhex('FFD9'))  # JPEG footer
        
        # Some padding
        f.write(b'\x00' * 1000)
        
        # PNG header
        f.write(bytes.fromhex('89504E470D0A1A0A'))
        f.write(b'fake png data' * 200)
        f.write(bytes.fromhex('49454E44AE426082'))  # PNG footer
        
        # More padding
        f.write(b'\x00' * 2000)
        
        # PDF header
        f.write(bytes.fromhex('255044462D'))
        f.write(b'fake pdf content' * 50)
        f.write(bytes.fromhex('0A2525454F46'))  # PDF footer
        
    return image_path


@pytest.fixture
def sample_signatures_db(temp_dir: Path) -> Path:
    """יצירת מסד נתוני חתימות מדומה"""
    db_path = temp_dir / "test_signatures.json"
    
    signatures = {
        "signatures": [
            {
                "name": "JPEG Image",
                "extension": ".jpg",
                "header": "FFD8FF",
                "footer": "FFD9",
                "max_size": 10485760,
                "footer_search_offset": 1024,
                "description": "תמונת JPEG לבדיקה"
            },
            {
                "name": "PNG Image", 
                "extension": ".png",
                "header": "89504E470D0A1A0A",
                "footer": "49454E44AE426082",
                "max_size": 52428800,
                "footer_search_offset": 2048,
                "description": "תמונת PNG לבדיקה"
            },
            {
                "name": "PDF Document",
                "extension": ".pdf", 
                "header": "255044462D",
                "footer": "0A2525454F46",
                "max_size": 104857600,
                "footer_search_offset": 4096,
                "description": "מסמך PDF לבדיקה"
            }
        ]
    }
    
    with open(db_path, 'w', encoding='utf-8') as f:
        json.dump(signatures, f, ensure_ascii=False, indent=2)
    
    return db_path


@pytest.fixture
def carved_files_dir(temp_dir: Path) -> Path:
    """יצירת תיקייה עם קבצים שנחתכו לבדיקת אימות"""
    carved_dir = temp_dir / "carved_files"
    carved_dir.mkdir()
    
    # יצירת קבצים מדומים
    (carved_dir / "000001_JPEG_Image.jpg").write_bytes(
        bytes.fromhex('FFD8FF') + b'fake jpeg' + bytes.fromhex('FFD9')
    )
    (carved_dir / "000002_PNG_Image.png").write_bytes(
        bytes.fromhex('89504E470D0A1A0A') + b'fake png' + bytes.fromhex('49454E44AE426082')
    )
    (carved_dir / "000003_PDF_Document.pdf").write_bytes(
        bytes.fromhex('255044462D') + b'fake pdf' + bytes.fromhex('0A2525454F46')
    )
    
    return carved_dir


@pytest.fixture
def mock_bad_sectors(temp_dir: Path) -> Path:
    """יצירת קובץ סקטורים פגומים מדומה"""
    bad_sectors_file = temp_dir / "test_disk.dd.bad_sectors"
    
    with open(bad_sectors_file, 'w') as f:
        f.write("# Bad sectors file for test_disk.dd\n")
        f.write("# Format: sector_number\n")
        f.write("1024\n")
        f.write("2048\n") 
        f.write("4096\n")
    
    return bad_sectors_file


@pytest.fixture(scope="session")
def large_test_data():
    """נתונים גדולים לבדיקות ביצועים (נטען פעם אחת לכל session)"""
    return b'A' * (10 * 1024 * 1024)  # 10MB of data


class TestDataFactory:
    """Factory לייצור נתוני בדיקה"""
    
    @staticmethod
    def create_corrupt_file(size: int = 1024) -> bytes:
        """יצירת קובץ פגום"""
        return b'\x00' * size
    
    @staticmethod
    def create_valid_jpeg(size: int = 1024) -> bytes:
        """יצירת JPEG תקין"""
        header = bytes.fromhex('FFD8FF')
        footer = bytes.fromhex('FFD9')
        body = b'fake jpeg data' * (size // 15)
        return header + body[:size-6] + footer
    
    @staticmethod
    def create_fragmented_file(total_size: int, fragment_size: int) -> list[bytes]:
        """יצירת קובץ מפוצל לחלקים"""
        fragments = []
        for i in range(0, total_size, fragment_size):
            end = min(i + fragment_size, total_size)
            fragments.append(b'X' * (end - i))
        return fragments


@pytest.fixture
def test_factory():
    """גישה ל-TestDataFactory"""
    return TestDataFactory


# Custom markers for better test organization
pytest_plugins = []