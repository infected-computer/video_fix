"""
PhoenixDRS Filesystem Parser Tests
בדיקות עבור מודול פרסור מערכות הקבצים
"""

import pytest
import struct
from unittest.mock import Mock, patch, mock_open
from datetime import datetime
from pathlib import Path

from filesystem_parser import (
    FilesystemType, FileEntry, FilesystemInfo,
    FilesystemParser, NTFSParser, EXT4Parser, FilesystemDetector
)


class TestFileEntry:
    """בדיקות עבור FileEntry"""
    
    def test_file_entry_creation(self):
        """בדיקת יצירת entry של קובץ"""
        entry = FileEntry(
            name="test.txt",
            path="/path/to/test.txt",
            size=1024,
            created_time=datetime(2023, 1, 1, 12, 0, 0),
            modified_time=datetime(2023, 1, 2, 12, 0, 0),
            accessed_time=datetime(2023, 1, 3, 12, 0, 0),
            is_directory=False, 
            is_deleted=False,
            inode=12345,
            cluster_chain=[1, 2, 3],
            attributes={"readonly": True}
        )
        
        assert entry.name == "test.txt"
        assert entry.path == "/path/to/test.txt"
        assert entry.size == 1024
        assert entry.is_directory is False
        assert entry.is_deleted is False
        assert entry.inode == 12345
        assert entry.cluster_chain == [1, 2, 3]
        assert entry.attributes["readonly"] is True


class TestFilesystemInfo:
    """בדיקות עבור FilesystemInfo"""
    
    def test_filesystem_info_creation(self):
        """בדיקת יצירת מידע מערכת קבצים"""
        info = FilesystemInfo(
            filesystem_type=FilesystemType.NTFS,
            volume_label="TestVolume",
            total_size=1000000000,
            used_size=500000000,
            cluster_size=4096,
            root_directory_cluster=5,
            filesystem_created=datetime(2023, 1, 1),
            last_mount_time=datetime(2023, 1, 10),
            mount_count=5,
            errors_found=["Minor corruption"]
        )
        
        assert info.filesystem_type == FilesystemType.NTFS
        assert info.volume_label == "TestVolume"
        assert info.total_size == 1000000000
        assert info.used_size == 500000000
        assert info.cluster_size == 4096
        assert info.root_directory_cluster == 5
        assert info.mount_count == 5
        assert len(info.errors_found) == 1


class MockFilesystemParser(FilesystemParser):
    """Mock parser לבדיקות"""
    
    def __init__(self, image_path: str):
        super().__init__(image_path)
        self.detected = False
    
    def detect_filesystem(self) -> bool:
        return self.detected
    
    def parse_filesystem_info(self) -> FilesystemInfo:
        return FilesystemInfo(
            filesystem_type=FilesystemType.UNKNOWN,
            volume_label="Mock",
            total_size=1000,
            used_size=500,
            cluster_size=512,
            root_directory_cluster=0,
            filesystem_created=None,
            last_mount_time=None,
            mount_count=0,
            errors_found=[]
        )
    
    def enumerate_files(self):
        yield FileEntry(
            name="mock_file.txt", path="/mock_file.txt", size=100,
            created_time=None, modified_time=None, accessed_time=None,
            is_directory=False, is_deleted=False, inode=1,
            cluster_chain=[1], attributes={}
        )
    
    def recover_deleted_files(self):
        return iter([])


class TestFilesystemParser:
    """בדיקות עבור FilesystemParser base class"""
    
    def test_context_manager(self, temp_dir):
        """בדיקת context manager"""
        test_file = temp_dir / "test.img"
        test_file.write_bytes(b"test data" * 1000)
        
        parser = MockFilesystemParser(str(test_file))
        
        # בדיקת שהקבצים לא פתוחים לפני __enter__
        assert parser._image_file is None
        assert parser._mmapped_file is None
        
        with parser as p:
            # בדיקת שהקבצים פתוחים
            assert p._image_file is not None
            assert p._mmapped_file is not None
            
            # בדיקת קריאת נתונים
            data = p.read_sector(0, 512)
            assert len(data) == 512
            assert data.startswith(b"test data")
        
        # בדיקת שהקבצים נסגרו
        assert parser._image_file is None or parser._image_file.closed
    
    def test_read_sector(self, temp_dir):
        """בדיקת קריאת סקטור"""
        test_file = temp_dir / "test.img"
        test_data = b"A" * 512 + b"B" * 512 + b"C" * 512
        test_file.write_bytes(test_data)
        
        parser = MockFilesystemParser(str(test_file))
        
        with parser as p:
            # קריאת סקטור ראשון
            sector0 = p.read_sector(0, 512)
            assert sector0 == b"A" * 512
            
            # קריאת סקטור שני
            sector1 = p.read_sector(1, 512)
            assert sector1 == b"B" * 512
            
            # קריאת סקטור שלישי
            sector2 = p.read_sector(2, 512)
            assert sector2 == b"C" * 512
    
    def test_read_clusters(self, temp_dir):
        """בדיקת קריאת clusters"""
        test_file = temp_dir / "test.img"
        cluster_size = 1024
        test_data = (b"1" * cluster_size + 
                    b"2" * cluster_size + 
                    b"3" * cluster_size + 
                    b"4" * cluster_size)
        test_file.write_bytes(test_data)
        
        parser = MockFilesystemParser(str(test_file))
        
        with parser as p:
            # קריאת cluster יחיד
            cluster_data = p.read_clusters([1], cluster_size)
            assert cluster_data == b"2" * cluster_size
            
            # קריאת מספר clusters
            clusters_data = p.read_clusters([0, 2], cluster_size)
            expected = b"1" * cluster_size + b"3" * cluster_size
            assert clusters_data == expected
    
    def test_read_sector_without_context(self, temp_dir):
        """בדיקת שגיאה בקריאת סקטור ללא context manager"""
        test_file = temp_dir / "test.img"
        test_file.write_bytes(b"test")
        
        parser = MockFilesystemParser(str(test_file))
        
        with pytest.raises(RuntimeError, match="Parser not initialized"):
            parser.read_sector(0)


class TestNTFSParser:
    """בדיקות עבור NTFSParser"""
    
    def create_ntfs_boot_sector(self) -> bytes:
        """יצירת boot sector מדומה של NTFS"""
        boot_sector = bytearray(512)
        
        # Jump instruction
        boot_sector[0:3] = b'\xEB\x52\x90'
        
        # OEM ID
        boot_sector[3:11] = b'NTFS    '
        
        # Bytes per sector (512)
        boot_sector[0x0B:0x0D] = struct.pack('<H', 512)
        
        # Sectors per cluster (8 = 4KB clusters)
        boot_sector[0x0D] = 8
        
        # Total sectors (1,000,000)
        boot_sector[0x28:0x30] = struct.pack('<Q', 1000000)
        
        # MFT cluster location (1000)
        boot_sector[0x30:0x38] = struct.pack('<Q', 1000)
        
        # MFT Mirror cluster location (500000)
        boot_sector[0x38:0x40] = struct.pack('<Q', 500000)
        
        # Volume serial (0x12345678)
        boot_sector[0x48:0x4C] = struct.pack('<L', 0x12345678)
        
        return bytes(boot_sector)
    
    def test_detect_ntfs_valid(self, temp_dir):
        """בדיקת זיהוי NTFS תקין"""
        test_file = temp_dir / "ntfs.img"
        
        # יצירת תמונה עם boot sector תקין
        boot_sector = self.create_ntfs_boot_sector()
        test_file.write_bytes(boot_sector + b'\x00' * 10000)
        
        parser = NTFSParser(str(test_file))
        
        with parser as p:
            assert p.detect_filesystem() is True
            assert p.boot_sector is not None
            assert p.boot_sector == boot_sector
    
    def test_detect_ntfs_invalid(self, temp_dir):
        """בדיקת זיהוי NTFS לא תקין"""
        test_file = temp_dir / "not_ntfs.img"
        
        # יצירת boot sector לא תקין
        invalid_boot = b"INVALID_BOOT_SECTOR" + b'\x00' * 494
        test_file.write_bytes(invalid_boot)
        
        parser = NTFSParser(str(test_file))
        
        with parser as p:
            assert p.detect_filesystem() is False
            assert p.boot_sector is None
    
    def test_parse_filesystem_info(self, temp_dir):
        """בדיקת פרסור מידע NTFS"""
        test_file = temp_dir / "ntfs.img"
        boot_sector = self.create_ntfs_boot_sector()
        test_file.write_bytes(boot_sector + b'\x00' * 100000)
        
        parser = NTFSParser(str(test_file))
        
        with parser as p:
            p.detect_filesystem()
            fs_info = p.parse_filesystem_info()
            
            assert fs_info.filesystem_type == FilesystemType.NTFS
            assert fs_info.total_size == 1000000 * 512  # total_sectors * bytes_per_sector
            assert fs_info.cluster_size == 512 * 8  # 4KB clusters
            assert fs_info.root_directory_cluster == 5
            assert p.mft_location == 1000 * 4096  # MFT cluster * cluster_size
    
    def test_parse_filesystem_info_not_detected(self, temp_dir):
        """בדיקת שגיאה בפרסור ללא זיהוי"""
        test_file = temp_dir / "test.img"
        test_file.write_bytes(b'\x00' * 1000)
        
        parser = NTFSParser(str(test_file))
        
        with parser as p:
            with pytest.raises(RuntimeError, match="NTFS not detected"):
                p.parse_filesystem_info()
    
    def test_enumerate_files_manual(self, temp_dir):
        """בדיקת מניית קבצים באופן ידני"""
        test_file = temp_dir / "ntfs.img"
        boot_sector = self.create_ntfs_boot_sector()
        
        # יצירת MFT record מדומה
        mft_record = bytearray(1024)
        mft_record[0:4] = b'FILE'  # MFT signature
        mft_record[0x16:0x18] = struct.pack('<H', 0x01)  # Flags (file in use)
        mft_record[0x14:0x16] = struct.pack('<H', 0x30)  # First attribute offset
        
        # הוספת $DATA attribute מדומה (type 0x80)
        mft_record[0x30:0x34] = struct.pack('<L', 0x80)  # Attribute type
        mft_record[0x34:0x38] = struct.pack('<L', 24)    # Attribute length
        mft_record[0x38:0x40] = struct.pack('<Q', 1024)  # File size
        
        # סיום attributes
        mft_record[0x48:0x4C] = struct.pack('<L', 0xFFFFFFFF)
        
        # בניית התמונה
        image_data = boot_sector + b'\x00' * (1000 * 4096 - 512) + bytes(mft_record)
        test_file.write_bytes(image_data)
        
        parser = NTFSParser(str(test_file))
        parser.use_tsk = False  # כפה שימוש בפרסור ידני
        
        with parser as p:
            p.detect_filesystem()
            p.parse_filesystem_info()
            
            files = list(p.enumerate_files())
            assert len(files) >= 1
            
            # בדיקת הקובץ הראשון
            first_file = files[0]
            assert first_file.name.startswith("MFT_Record_")
            assert first_file.size == 1024
            assert first_file.is_deleted is False
            assert first_file.inode == 0
    
    @patch('filesystem_parser.TSK_AVAILABLE', True)
    @patch('filesystem_parser.pytsk3')
    def test_enumerate_files_tsk(self, mock_pytsk3, temp_dir):
        """בדיקת מניית קבצים עם TSK"""
        test_file = temp_dir / "ntfs.img"
        boot_sector = self.create_ntfs_boot_sector()
        test_file.write_bytes(boot_sector + b'\x00' * 100000)
        
        # Mock TSK objects
        mock_img = Mock()
        mock_fs = Mock()
        mock_entry = Mock()
        mock_meta = Mock()
        mock_name = Mock()
        
        # הגדרת mock values
        mock_name.name = b"test_file.txt"
        mock_name.flags = 0  # Not deleted
        mock_meta.size = 1024
        mock_meta.type = 1  # Regular file
        mock_meta.addr = 123  # Inode
        mock_meta.crtime = 1640995200  # Unix timestamp
        mock_meta.mtime = 1640995200
        mock_meta.atime = 1640995200
        
        mock_entry.info.name = mock_name
        mock_entry.info.meta = mock_meta
        
        mock_dir = [mock_entry]
        mock_fs.open_dir.return_value = mock_dir
        
        mock_pytsk3.Img_Info.return_value = mock_img
        mock_pytsk3.FS_Info.return_value = mock_fs
        mock_pytsk3.TSK_FS_META_TYPE_DIR = 2
        mock_pytsk3.TSK_FS_NAME_FLAG_UNALLOC = 1
        
        parser = NTFSParser(str(test_file))
        parser.use_tsk = True
        
        with parser as p:
            p.detect_filesystem()
            p.parse_filesystem_info()
            
            files = list(p.enumerate_files())
            assert len(files) >= 1
            
            first_file = files[0]
            assert first_file.name == "test_file.txt"
            assert first_file.size == 1024
            assert first_file.is_deleted is False
            assert first_file.inode == 123
    
    def test_recover_deleted_files(self, temp_dir):
        """בדיקת שחזור קבצים מחוקים"""
        test_file = temp_dir / "ntfs.img"
        boot_sector = self.create_ntfs_boot_sector()
        
        # יצירת MFT record של קובץ מחוק
        mft_record = bytearray(1024)
        mft_record[0:4] = b'FILE'
        mft_record[0x16:0x18] = struct.pack('<H', 0x00)  # Flags (deleted)
        mft_record[0x14:0x16] = struct.pack('<H', 0x30)
        mft_record[0x30:0x34] = struct.pack('<L', 0x80)
        mft_record[0x34:0x38] = struct.pack('<L', 24)
        mft_record[0x38:0x40] = struct.pack('<Q', 2048)
        mft_record[0x48:0x4C] = struct.pack('<L', 0xFFFFFFFF)
        
        image_data = boot_sector + b'\x00' * (1000 * 4096 - 512) + bytes(mft_record)
        test_file.write_bytes(image_data)
        
        parser = NTFSParser(str(test_file))
        parser.use_tsk = False
        
        with parser as p:
            p.detect_filesystem()
            p.parse_filesystem_info()
            
            deleted_files = list(p.recover_deleted_files())
            assert len(deleted_files) >= 1
            
            deleted_file = deleted_files[0]
            assert deleted_file.is_deleted is True
            assert deleted_file.size == 2048


class TestEXT4Parser:
    """בדיקות עבור EXT4Parser"""
    
    def create_ext4_superblock(self) -> bytes:
        """יצירת superblock מדומה של EXT4"""
        superblock = bytearray(1024)
        
        # Block count (offset 0x04)
        superblock[0x04:0x08] = struct.pack('<L', 100000)
        
        # Log block size (offset 0x18) - 0 means 1024 bytes
        superblock[0x18:0x1C] = struct.pack('<L', 1)  # 2048 byte blocks
        
        # Magic number (offset 0x38)
        superblock[0x38:0x3A] = struct.pack('<H', 0xEF53)
        
        # Volume label (offset 0x78)
        label = b"TestEXT4Volume\x00"
        superblock[0x78:0x78 + len(label)] = label
        
        return bytes(superblock)
    
    def test_detect_ext4_valid(self, temp_dir):
        """בדיקת זיהוי EXT4 תקין"""
        test_file = temp_dir / "ext4.img"
        
        # יצירת תמונה עם superblock תקין
        superblock = self.create_ext4_superblock()
        image_data = b'\x00' * 1024 + superblock + b'\x00' * 10000
        test_file.write_bytes(image_data)
        
        parser = EXT4Parser(str(test_file))
        
        with parser as p:
            assert p.detect_filesystem() is True
            assert p.superblock is not None
    
    def test_detect_ext4_invalid(self, temp_dir):
        """בדיקת זיהוי EXT4 לא תקין"""
        test_file = temp_dir / "not_ext4.img"
        
        # יצירת superblock לא תקין (magic number שגוי)
        invalid_superblock = bytearray(1024)
        invalid_superblock[0x38:0x3A] = struct.pack('<H', 0x1234)  # Wrong magic
        
        image_data = b'\x00' * 1024 + bytes(invalid_superblock)
        test_file.write_bytes(image_data)
        
        parser = EXT4Parser(str(test_file))
        
        with parser as p:
            assert p.detect_filesystem() is False
            assert p.superblock is None
    
    def test_parse_filesystem_info(self, temp_dir):
        """בדיקת פרסור מידע EXT4"""
        test_file = temp_dir / "ext4.img"
        superblock = self.create_ext4_superblock()
        image_data = b'\x00' * 1024 + superblock + b'\x00' * 100000
        test_file.write_bytes(image_data)
        
        parser = EXT4Parser(str(test_file))
        
        with parser as p:
            p.detect_filesystem()
            fs_info = p.parse_filesystem_info()
            
            assert fs_info.filesystem_type == FilesystemType.EXT4
            assert fs_info.volume_label == "TestEXT4Volume"
            assert fs_info.total_size == 100000 * 2048  # blocks * block_size
            assert fs_info.cluster_size == 2048  # 2KB blocks
            assert fs_info.root_directory_cluster == 2
    
    def test_enumerate_files_basic(self, temp_dir):
        """בדיקת מניית קבצים בסיסית ב-EXT4"""
        test_file = temp_dir / "ext4.img"
        superblock = self.create_ext4_superblock()
        image_data = b'\x00' * 1024 + superblock + b'\x00' * 100000
        test_file.write_bytes(image_data)
        
        parser = EXT4Parser(str(test_file))
        parser.use_tsk = False  # Force manual parsing
        
        with parser as p:
            p.detect_filesystem()
            p.parse_filesystem_info()
            
            files = list(p.enumerate_files())
            
            # בדיקה שיש לפחות root directory
            assert len(files) >= 1
            root_file = files[0]
            assert root_file.name == "root"
            assert root_file.path == "/"
            assert root_file.is_directory is True
            assert root_file.inode == 2


class TestFilesystemDetector:
    """בדיקות עבור FilesystemDetector"""
    
    def test_detect_ntfs(self, temp_dir):
        """בדיקת זיהוי NTFS אוטומטי"""
        test_file = temp_dir / "ntfs_disk.img"
        
        # יצירת boot sector של NTFS
        boot_sector = bytearray(512)
        boot_sector[0:3] = b'\xEB\x52\x90'
        boot_sector[3:11] = b'NTFS    '
        boot_sector[0x0B:0x0D] = struct.pack('<H', 512)
        boot_sector[0x0D] = 8
        boot_sector[0x28:0x30] = struct.pack('<Q', 1000)
        boot_sector[0x30:0x38] = struct.pack('<Q', 100)
        boot_sector[0x38:0x40] = struct.pack('<Q', 500)
        
        test_file.write_bytes(bytes(boot_sector) + b'\x00' * 10000)
        
        fs_type = FilesystemDetector.detect_filesystem_type(str(test_file))
        assert fs_type == FilesystemType.NTFS
    
    def test_detect_ext4(self, temp_dir):
        """בדיקת זיהוי EXT4 אוטומטי"""
        test_file = temp_dir / "ext4_disk.img"
        
        # יצירת superblock של EXT4
        superblock = bytearray(1024)
        superblock[0x04:0x08] = struct.pack('<L', 10000)
        superblock[0x18:0x1C] = struct.pack('<L', 0)
        superblock[0x38:0x3A] = struct.pack('<H', 0xEF53)
        
        image_data = b'\x00' * 1024 + bytes(superblock) + b'\x00' * 10000
        test_file.write_bytes(image_data)
        
        fs_type = FilesystemDetector.detect_filesystem_type(str(test_file))
        assert fs_type == FilesystemType.EXT4
    
    def test_detect_unknown(self, temp_dir):
        """בדיקת זיהוי מערכת קבצים לא ידועה"""
        test_file = temp_dir / "unknown_disk.img"
        test_file.write_bytes(b"UNKNOWN_FILESYSTEM_DATA" * 1000)
        
        fs_type = FilesystemDetector.detect_filesystem_type(str(test_file))
        assert fs_type == FilesystemType.UNKNOWN
    
    def test_get_parser_ntfs(self, temp_dir):
        """בדיקת קבלת פרסר NTFS"""
        test_file = temp_dir / "ntfs_disk.img"
        
        # יצירת boot sector של NTFS
        boot_sector = bytearray(512)
        boot_sector[0:3] = b'\xEB\x52\x90'
        boot_sector[3:11] = b'NTFS    '
        boot_sector[0x0B:0x0D] = struct.pack('<H', 512)
        boot_sector[0x0D] = 8
        boot_sector[0x28:0x30] = struct.pack('<Q', 1000)
        boot_sector[0x30:0x38] = struct.pack('<Q', 100)
        
        test_file.write_bytes(bytes(boot_sector) + b'\x00' * 10000)
        
        parser = FilesystemDetector.get_parser(str(test_file))
        assert parser is not None
        assert isinstance(parser, NTFSParser)
    
    def test_get_parser_ext4(self, temp_dir):
        """בדיקת קבלת פרסר EXT4"""
        test_file = temp_dir / "ext4_disk.img"
        
        superblock = bytearray(1024)
        superblock[0x38:0x3A] = struct.pack('<H', 0xEF53)
        
        image_data = b'\x00' * 1024 + bytes(superblock)
        test_file.write_bytes(image_data)
        
        parser = FilesystemDetector.get_parser(str(test_file))
        assert parser is not None
        assert isinstance(parser, EXT4Parser)
    
    def test_get_parser_unknown(self, temp_dir):
        """בדיקת קבלת None עבור מערכת קבצים לא ידועה"""
        test_file = temp_dir / "unknown_disk.img"
        test_file.write_bytes(b"UNKNOWN" * 1000)
        
        parser = FilesystemDetector.get_parser(str(test_file))
        assert parser is None


@pytest.mark.integration
class TestFilesystemParserIntegration:
    """בדיקות אינטגרציה עבור פרסרי מערכות הקבצים"""
    
    def test_full_ntfs_workflow(self, temp_dir):
        """בדיקת זרימת עבודה מלאה עם NTFS"""
        test_file = temp_dir / "full_ntfs.img"
        
        # יצירת תמונת NTFS מורכבת
        boot_sector = bytearray(512)
        boot_sector[0:3] = b'\xEB\x52\x90'
        boot_sector[3:11] = b'NTFS    '
        boot_sector[0x0B:0x0D] = struct.pack('<H', 512)
        boot_sector[0x0D] = 8
        boot_sector[0x28:0x30] = struct.pack('<Q', 100000)
        boot_sector[0x30:0x38] = struct.pack('<Q', 1000)
        boot_sector[0x38:0x40] = struct.pack('<Q', 50000)
        
        # יצירת MFT records מרובים
        mft_records = bytearray(10240)  # 10 records
        for i in range(10):
            offset = i * 1024
            mft_records[offset:offset + 4] = b'FILE'
            mft_records[offset + 0x16:offset + 0x18] = struct.pack('<H', 0x01)
            mft_records[offset + 0x14:offset + 0x16] = struct.pack('<H', 0x30)
            mft_records[offset + 0x30:offset + 0x34] = struct.pack('<L', 0x80)
            mft_records[offset + 0x34:offset + 0x38] = struct.pack('<L', 24)
            mft_records[offset + 0x38:offset + 0x40] = struct.pack('<Q', 1024 * (i + 1))
            mft_records[offset + 0x48:offset + 0x4C] = struct.pack('<L', 0xFFFFFFFF)
        
        # בניית התמונה המלאה
        mft_offset = 1000 * 4096  # MFT location
        image_size = mft_offset + len(mft_records)
        image_data = bytearray(image_size)
        image_data[0:512] = boot_sector
        image_data[mft_offset:mft_offset + len(mft_records)] = mft_records
        
        test_file.write_bytes(bytes(image_data))
        
        # זרימת עבודה מלאה
        # 1. זיהוי אוטומטי
        fs_type = FilesystemDetector.detect_filesystem_type(str(test_file))
        assert fs_type == FilesystemType.NTFS
        
        # 2. קבלת פרסר
        parser = FilesystemDetector.get_parser(str(test_file))
        assert isinstance(parser, NTFSParser)
        
        # 3. ניתוח מלא
        with parser as p:
            # זיהוי
            assert p.detect_filesystem() is True
            
            # פרסור מידע
            fs_info = p.parse_filesystem_info()
            assert fs_info.filesystem_type == FilesystemType.NTFS
            assert fs_info.total_size == 100000 * 512
            assert fs_info.cluster_size == 4096
            
            # מניית קבצים
            p.use_tsk = False  # Force manual parsing
            files = list(p.enumerate_files())
            assert len(files) >= 10
            
            # בדיקת קבצים ספציפיים
            for i, file_entry in enumerate(files[:10]):
                assert file_entry.name.startswith("MFT_Record_")
                assert file_entry.size == 1024 * (i + 1)
                assert file_entry.is_deleted is False
                assert file_entry.inode == i
    
    def test_filesystem_comparison(self, temp_dir):
        """בדיקת השוואה בין מערכות קבצים שונות"""
        # יצירת תמונות של מערכות קבצים שונות
        ntfs_file = temp_dir / "compare_ntfs.img"
        ext4_file = temp_dir / "compare_ext4.img"
        
        # NTFS
        ntfs_boot = bytearray(512)
        ntfs_boot[3:11] = b'NTFS    '
        ntfs_boot[0x0B:0x0D] = struct.pack('<H', 512)
        ntfs_boot[0x0D] = 8
        ntfs_boot[0x28:0x30] = struct.pack('<Q', 50000)
        ntfs_file.write_bytes(bytes(ntfs_boot) + b'\x00' * 100000)
        
        # EXT4
        ext4_super = bytearray(1024)
        ext4_super[0x38:0x3A] = struct.pack('<H', 0xEF53)
        ext4_super[0x04:0x08] = struct.pack('<L', 25000)
        ext4_super[0x18:0x1C] = struct.pack('<L', 1)
        ext4_data = b'\x00' * 1024 + bytes(ext4_super) + b'\x00' * 100000
        ext4_file.write_bytes(ext4_data)
        
        # השוואת הזיהוי
        ntfs_type = FilesystemDetector.detect_filesystem_type(str(ntfs_file))
        ext4_type = FilesystemDetector.detect_filesystem_type(str(ext4_file))
        
        assert ntfs_type == FilesystemType.NTFS
        assert ext4_type == FilesystemType.EXT4
        assert ntfs_type != ext4_type
        
        # השוואת הפרסרים
        ntfs_parser = FilesystemDetector.get_parser(str(ntfs_file))
        ext4_parser = FilesystemDetector.get_parser(str(ext4_file))
        
        assert isinstance(ntfs_parser, NTFSParser)
        assert isinstance(ext4_parser, EXT4Parser)
        assert type(ntfs_parser) != type(ext4_parser)
        
        # השוואת מידע מערכות הקבצים
        with ntfs_parser as np, ext4_parser as ep:
            np.detect_filesystem()
            ep.detect_filesystem()
            
            ntfs_info = np.parse_filesystem_info()
            ext4_info = ep.parse_filesystem_info()
            
            assert ntfs_info.filesystem_type != ext4_info.filesystem_type
            assert ntfs_info.total_size == 50000 * 512
            assert ext4_info.total_size == 25000 * 2048
            assert ntfs_info.cluster_size == 4096
            assert ext4_info.cluster_size == 2048