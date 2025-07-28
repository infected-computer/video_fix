"""
PhoenixDRS Test Configuration and Fixtures
תצורה ו-fixtures עבור בדיקות PhoenixDRS
"""

import pytest
import tempfile
import os
import sys
import asyncio
import logging
from pathlib import Path
from typing import Generator, Dict, Any
import json

# Add src directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Configure logging for tests
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Disable verbose logging from external libraries during tests
logging.getLogger('transformers').setLevel(logging.WARNING)
logging.getLogger('torch').setLevel(logging.WARNING)
logging.getLogger('PIL').setLevel(logging.WARNING)


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """יצירת תיקייה זמנית עבור בדיקות"""
    with tempfile.TemporaryDirectory() as temp_path:
        yield Path(temp_path)


@pytest.fixture
def temp_directory() -> Generator[Path, None, None]:
    """Alternative name for temp_dir to match integration tests"""
    temp_dir = Path(tempfile.mkdtemp(prefix="phoenixdrs_test_"))
    try:
        yield temp_dir
    finally:
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)


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


# Video test fixtures for integration tests
@pytest.fixture
def sample_video_files(temp_directory: Path) -> Dict[str, Path]:
    """Create sample video files for testing."""
    files = {}
    
    # Create MP4 file with basic structure
    mp4_content = (
        b"\x00\x00\x00\x20ftypisom\x00\x00\x02\x00isomiso2avc1mp41"  # ftyp box
        b"\x00\x00\x00\x08free"  # free box
        b"\x00\x00\x01\x00mdat"  # mdat box start
        + b"\x00" * 200  # dummy video data
    )
    
    mp4_path = temp_directory / "sample.mp4"
    mp4_path.write_bytes(mp4_content)
    files["mp4"] = mp4_path
    
    return files


@pytest.fixture
def corrupted_video_files(temp_directory: Path) -> Dict[str, Path]:
    """Create corrupted video files for testing repair functionality."""
    files = {}
    
    # Corrupted header
    corrupted_header = (
        b"\xFF\xFF\xFF\xFFftypisom\x00\x00\x02\x00isomiso2avc1mp41"  # corrupted ftyp
        b"\x00\x00\x00\x08free"
        b"\x00\x00\x01\x00mdat"
        + b"\x00" * 200
    )
    
    header_corrupted = temp_directory / "header_corrupted.mp4"
    header_corrupted.write_bytes(corrupted_header)
    files["header_corrupted"] = header_corrupted
    
    return files


@pytest.fixture
def repair_config_basic():
    """Basic repair configuration for testing."""
    try:
        from python.video_repair_orchestrator import RepairConfiguration, RepairTechnique
        
        return RepairConfiguration(
            use_gpu=False,
            enable_ai_processing=False,
            max_cpu_threads=2,
            techniques=[RepairTechnique.HEADER_RECONSTRUCTION, RepairTechnique.INDEX_REBUILD],
            quality_factor=0.8
        )
    except ImportError:
        # Return mock config if module not available
        return {
            "use_gpu": False,
            "enable_ai_processing": False,
            "max_cpu_threads": 2,
            "quality_factor": 0.8
        }


# Test markers
def pytest_configure(config):
    """Configure custom pytest markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "gpu: marks tests as requiring GPU"
    )
    config.addinivalue_line(
        "markers", "ai: marks tests as requiring AI models"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "benchmark: marks tests as performance benchmarks"
    )


# Custom markers for better test organization
pytest_plugins = []