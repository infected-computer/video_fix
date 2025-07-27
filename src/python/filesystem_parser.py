"""
PhoenixDRS - Filesystem Parser Module
מודול פרסור מערכות קבצים מתקדם

This module provides unified interface for parsing different filesystem types,
enabling recovery of file system structures and metadata even from damaged drives.
"""

import struct
import os
import mmap
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Iterator
from datetime import datetime
from enum import Enum

try:
    import pytsk3
    TSK_AVAILABLE = True
except ImportError:
    TSK_AVAILABLE = False
    print("Warning: pytsk3 not available. Some filesystem parsers will be limited.")


class FilesystemType(Enum):
    """סוגי מערכות קבצים נתמכות"""
    NTFS = "ntfs"
    APFS = "apfs"
    EXT4 = "ext4"
    EXFAT = "exfat"
    FAT32 = "fat32"
    HFS_PLUS = "hfsplus"
    UNKNOWN = "unknown"


@dataclass
class FileEntry:
    """ייצוג קובץ במערכת הקבצים"""
    name: str
    path: str
    size: int
    created_time: Optional[datetime]
    modified_time: Optional[datetime]
    accessed_time: Optional[datetime]
    is_directory: bool
    is_deleted: bool
    inode: Optional[int]
    cluster_chain: List[int]
    attributes: Dict[str, any]


@dataclass
class FilesystemInfo:
    """מידע כללי על מערכת הקבצים"""
    filesystem_type: FilesystemType
    volume_label: str
    total_size: int
    used_size: int
    cluster_size: int
    root_directory_cluster: int
    filesystem_created: Optional[datetime]
    last_mount_time: Optional[datetime]
    mount_count: int
    errors_found: List[str]


class FilesystemParser(ABC):
    """מחלקת בסיס לפרסרי מערכות קבצים"""
    
    def __init__(self, image_path: str):
        self.image_path = image_path
        self.filesystem_info: Optional[FilesystemInfo] = None
        self.files: List[FileEntry] = []
        self._image_file = None
        self._mmapped_file = None
    
    def __enter__(self):
        """Context manager entry"""
        self._image_file = open(self.image_path, 'rb')
        self._mmapped_file = mmap.mmap(
            self._image_file.fileno(), 0, access=mmap.ACCESS_READ
        )
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        if self._mmapped_file:
            self._mmapped_file.close()
        if self._image_file:
            self._image_file.close()
    
    @abstractmethod
    def detect_filesystem(self) -> bool:
        """זיהוי סוג מערכת הקבצים"""
        pass
    
    @abstractmethod
    def parse_filesystem_info(self) -> FilesystemInfo:
        """ניתוח מידע כללי על מערכת הקבצים"""
        pass
    
    @abstractmethod
    def enumerate_files(self) -> Iterator[FileEntry]:
        """מניית כל הקבצים במערכת הקבצים"""
        pass
    
    @abstractmethod
    def recover_deleted_files(self) -> Iterator[FileEntry]:
        """שחזור קבצים מחוקים"""
        pass
    
    def read_sector(self, sector_number: int, sector_size: int = 512) -> bytes:
        """קריאת סקטור מהתמונה"""
        if not self._mmapped_file:
            raise RuntimeError("Parser not initialized. Use with context manager.")
        
        offset = sector_number * sector_size
        return self._mmapped_file[offset:offset + sector_size]
    
    def read_clusters(self, cluster_numbers: List[int], cluster_size: int) -> bytes:
        """קריאת רצף של clusters"""
        data = b""
        for cluster_num in cluster_numbers:
            offset = cluster_num * cluster_size
            data += self._mmapped_file[offset:offset + cluster_size]
        return data


class NTFSParser(FilesystemParser):
    """פרסר מערכת קבצים NTFS"""
    
    def __init__(self, image_path: str):
        super().__init__(image_path)
        self.boot_sector = None
        self.mft_location = None
        self.cluster_size = None
        self.use_tsk = TSK_AVAILABLE
    
    def detect_filesystem(self) -> bool:
        """זיהוי NTFS על בסיס boot sector"""
        try:
            boot_sector = self.read_sector(0)
            
            # בדיקת חתימת NTFS
            if boot_sector[3:11] == b'NTFS    ':
                self.boot_sector = boot_sector
                return True
                
            # בדיקת חתימה חלופית
            if boot_sector[0:3] == b'\xEB\x52\x90' and b'NTFS' in boot_sector:
                self.boot_sector = boot_sector
                return True
                
        except Exception as e:
            print(f"Error detecting NTFS: {e}")
        
        return False
    
    def parse_filesystem_info(self) -> FilesystemInfo:
        """ניתוח מידע NTFS מה-boot sector"""
        if not self.boot_sector:
            raise RuntimeError("NTFS not detected. Call detect_filesystem() first.")
        
        # פרסור Boot Sector של NTFS
        # Offset 0x0B-0x0C: Bytes per sector
        bytes_per_sector = struct.unpack('<H', self.boot_sector[0x0B:0x0D])[0]
        
        # Offset 0x0D: Sectors per cluster
        sectors_per_cluster = self.boot_sector[0x0D]
        self.cluster_size = bytes_per_sector * sectors_per_cluster
        
        # Offset 0x28-0x2F: Total sectors
        total_sectors = struct.unpack('<Q', self.boot_sector[0x28:0x30])[0]
        total_size = total_sectors * bytes_per_sector
        
        # Offset 0x30-0x37: MFT cluster location
        mft_cluster = struct.unpack('<Q', self.boot_sector[0x30:0x38])[0]
        self.mft_location = mft_cluster * self.cluster_size
        
        # Offset 0x38-0x3F: MFT Mirror cluster location
        mft_mirror_cluster = struct.unpack('<Q', self.boot_sector[0x38:0x40])[0]
        
        # Offset 0x47-0x4A: Volume serial number
        volume_serial = struct.unpack('<L', self.boot_sector[0x48:0x4C])[0]
        
        self.filesystem_info = FilesystemInfo(
            filesystem_type=FilesystemType.NTFS,
            volume_label="",  # נקבע מה-MFT
            total_size=total_size,
            used_size=0,  # יחושב מה-MFT
            cluster_size=self.cluster_size,
            root_directory_cluster=5,  # MFT record 5 הוא root directory
            filesystem_created=None,  # נקבע מה-MFT
            last_mount_time=None,
            mount_count=0,
            errors_found=[]
        )
        
        return self.filesystem_info
    
    def enumerate_files(self) -> Iterator[FileEntry]:
        """מניית קבצים מתוך MFT"""
        if self.use_tsk and TSK_AVAILABLE:
            yield from self._enumerate_files_tsk()
        else:
            yield from self._enumerate_files_manual()
    
    def _enumerate_files_tsk(self) -> Iterator[FileEntry]:
        """מניית קבצים באמצעות The Sleuth Kit"""
        try:
            img = pytsk3.Img_Info(self.image_path)
            fs = pytsk3.FS_Info(img)
            
            def process_directory(directory, path=""):
                for entry in directory:
                    if entry.info.name and entry.info.name.name:
                        name = entry.info.name.name.decode('utf-8', errors='ignore')
                        
                        if name in ['.', '..']:
                            continue
                        
                        full_path = f"{path}/{name}" if path else name
                        
                        # קבלת מידע על הקובץ
                        file_entry = FileEntry(
                            name=name,
                            path=full_path,
                            size=entry.info.meta.size if entry.info.meta else 0,
                            created_time=self._convert_tsk_time(entry.info.meta.crtime) if entry.info.meta else None,
                            modified_time=self._convert_tsk_time(entry.info.meta.mtime) if entry.info.meta else None,
                            accessed_time=self._convert_tsk_time(entry.info.meta.atime) if entry.info.meta else None,
                            is_directory=entry.info.meta.type == pytsk3.TSK_FS_META_TYPE_DIR if entry.info.meta else False,
                            is_deleted=(entry.info.name.flags & pytsk3.TSK_FS_NAME_FLAG_UNALLOC) != 0,
                            inode=entry.info.meta.addr if entry.info.meta else None,
                            cluster_chain=[],  # TSK לא מספק מידע על clusters בקלות
                            attributes={}
                        )
                        
                        yield file_entry
                        
                        # אם זו תיקייה, עבור על התוכן שלה
                        if (entry.info.meta and 
                            entry.info.meta.type == pytsk3.TSK_FS_META_TYPE_DIR and
                            not file_entry.is_deleted):
                            try:
                                sub_directory = entry.as_directory()
                                yield from process_directory(sub_directory, full_path)
                            except:
                                pass  # תיקיות פגומות
            
            root_dir = fs.open_dir(path="/")
            yield from process_directory(root_dir)
            
        except Exception as e:
            print(f"Error using TSK for file enumeration: {e}")
            # Fallback to manual parsing
            yield from self._enumerate_files_manual()
    
    def _enumerate_files_manual(self) -> Iterator[FileEntry]:
        """מניית קבצים באופן ידני מה-MFT"""
        if not self.mft_location:
            return
        
        try:
            # קריאת MFT records
            mft_record_size = 1024  # גודל standard
            record_number = 0
            
            while record_number < 100000:  # הגבלה למניעת לולאה אינסופית
                offset = self.mft_location + (record_number * mft_record_size)
                
                try:
                    record_data = self._mmapped_file[offset:offset + mft_record_size]
                    
                    # בדיקת חתימת MFT record
                    if record_data[:4] != b'FILE':
                        record_number += 1
                        continue
                    
                    # פרסור בסיסי של MFT record
                    file_entry = self._parse_mft_record(record_data, record_number)
                    if file_entry:
                        yield file_entry
                        
                except Exception as e:
                    # Record פגום - המשך לבא
                    pass
                
                record_number += 1
                
        except Exception as e:
            print(f"Error in manual MFT parsing: {e}")
    
    def _parse_mft_record(self, record_data: bytes, record_number: int) -> Optional[FileEntry]:
        """פרסור MFT record יחיד"""
        try:
            # קריאת header של MFT record
            if len(record_data) < 48:
                return None
            
            # Offset 0x16-0x17: Flags
            flags = struct.unpack('<H', record_data[0x16:0x18])[0]
            is_directory = (flags & 0x02) != 0
            is_deleted = (flags & 0x01) == 0
            
            # Offset 0x14-0x15: First attribute offset
            first_attr_offset = struct.unpack('<H', record_data[0x14:0x16])[0]
            
            # איתור attributes
            filename = f"MFT_Record_{record_number}"
            file_size = 0
            created_time = None
            modified_time = None
            accessed_time = None
            
            # פרסור בסיסי של attributes
            attr_offset = first_attr_offset
            while attr_offset < len(record_data) - 4:
                attr_type = struct.unpack('<L', record_data[attr_offset:attr_offset + 4])[0]
                
                if attr_type == 0xFFFFFFFF:  # סוף attributes
                    break
                
                if attr_type == 0x30:  # $FILE_NAME attribute
                    # קריאת שם הקובץ (פרסור מפושט)
                    try:
                        attr_length = struct.unpack('<L', record_data[attr_offset + 4:attr_offset + 8])[0]
                        if attr_length > 0 and attr_offset + attr_length <= len(record_data):
                            # ניסיון לחלץ שם קובץ (פרסור בסיסי)
                            name_data = record_data[attr_offset + 24:attr_offset + attr_length]
                            if len(name_data) > 64:  # $FILE_NAME header
                                name_length = name_data[64]
                                if name_length > 0 and len(name_data) > 66 + name_length * 2:
                                    filename = name_data[66:66 + name_length * 2].decode('utf-16le', errors='ignore')
                    except:
                        pass
                
                elif attr_type == 0x80:  # $DATA attribute
                    # קריאת גודל קובץ
                    try:
                        attr_length = struct.unpack('<L', record_data[attr_offset + 4:attr_offset + 8])[0]
                        if attr_length >= 16:
                            file_size = struct.unpack('<Q', record_data[attr_offset + 8:attr_offset + 16])[0]
                    except:
                        pass
                
                # מעבר ל-attribute הבא
                try:
                    attr_length = struct.unpack('<L', record_data[attr_offset + 4:attr_offset + 8])[0]
                    if attr_length == 0:
                        break
                    attr_offset += attr_length
                except:
                    break
            
            return FileEntry(
                name=filename,
                path=f"/MFT_{record_number}/{filename}",
                size=file_size,
                created_time=created_time,
                modified_time=modified_time,
                accessed_time=accessed_time,
                is_directory=is_directory,
                is_deleted=is_deleted,
                inode=record_number,
                cluster_chain=[],
                attributes={"mft_record": record_number}
            )
            
        except Exception as e:
            return None
    
    def _convert_tsk_time(self, tsk_time) -> Optional[datetime]:
        """המרת זמן TSK לDateTime"""
        try:
            if tsk_time and tsk_time > 0:
                return datetime.fromtimestamp(tsk_time)
        except:
            pass
        return None
    
    def recover_deleted_files(self) -> Iterator[FileEntry]:
        """שחזור קבצים מחוקים מ-NTFS"""
        for file_entry in self.enumerate_files():
            if file_entry.is_deleted:
                yield file_entry


class EXT4Parser(FilesystemParser):
    """פרסר מערכת קבצים EXT4"""
    
    def __init__(self, image_path: str):
        super().__init__(image_path)
        self.superblock = None
        self.block_size = None
    
    def detect_filesystem(self) -> bool:
        """זיהוי EXT4 על בסיס superblock"""
        try:
            # EXT4 superblock נמצא ב-offset 1024
            superblock_data = self._mmapped_file[1024:1024 + 1024]
            
            # בדיקת magic number (offset 0x38 בsuperblock)
            magic = struct.unpack('<H', superblock_data[0x38:0x3A])[0]
            if magic == 0xEF53:  # EXT magic number
                self.superblock = superblock_data
                return True
                
        except Exception as e:
            print(f"Error detecting EXT4: {e}")
        
        return False
    
    def parse_filesystem_info(self) -> FilesystemInfo:
        """ניתוח מידע EXT4"""
        if not self.superblock:
            raise RuntimeError("EXT4 not detected. Call detect_filesystem() first.")
        
        # פרסור superblock
        block_count = struct.unpack('<L', self.superblock[0x04:0x08])[0]
        log_block_size = struct.unpack('<L', self.superblock[0x18:0x1C])[0]
        self.block_size = 1024 << log_block_size
        
        total_size = block_count * self.block_size
        
        # Volume label (offset 0x78)
        volume_label = self.superblock[0x78:0x88].rstrip(b'\x00').decode('utf-8', errors='ignore')
        
        self.filesystem_info = FilesystemInfo(
            filesystem_type=FilesystemType.EXT4,
            volume_label=volume_label,
            total_size=total_size,
            used_size=0,  # יחושב מה-group descriptors
            cluster_size=self.block_size,
            root_directory_cluster=2,  # Root inode = 2
            filesystem_created=None,
            last_mount_time=None,
            mount_count=0,
            errors_found=[]
        )
        
        return self.filesystem_info
    
    def enumerate_files(self) -> Iterator[FileEntry]:
        """מניית קבצים ב-EXT4 (מימוש בסיסי)"""
        # מימוש בסיסי - ייטב בעתיד
        if self.use_tsk and TSK_AVAILABLE:
            yield from self._enumerate_files_tsk()
        else:
            # Manual EXT4 parsing is very complex
            # This is a placeholder for basic functionality
            yield FileEntry(
                name="root",
                path="/",
                size=0,
                created_time=None,
                modified_time=None,
                accessed_time=None,
                is_directory=True,
                is_deleted=False,
                inode=2,
                cluster_chain=[],
                attributes={"filesystem": "ext4"}
            )
    
    def _enumerate_files_tsk(self) -> Iterator[FileEntry]:
        """מניית קבצים באמצעות TSK"""
        try:
            img = pytsk3.Img_Info(self.image_path)
            fs = pytsk3.FS_Info(img)
            
            # Similar to NTFS implementation but for EXT4
            # Implementation would be similar to NTFSParser._enumerate_files_tsk()
            root_dir = fs.open_dir(path="/")
            # ... implementation details
            
        except Exception as e:
            print(f"Error using TSK for EXT4: {e}")
    
    def recover_deleted_files(self) -> Iterator[FileEntry]:
        """שחזור קבצים מחוקים מ-EXT4"""
        # EXT4 deleted file recovery is complex
        # Placeholder implementation
        return iter([])


class FilesystemDetector:
    """גלאי מערכות קבצים אוטומטי"""
    
    @staticmethod
    def detect_filesystem_type(image_path: str) -> FilesystemType:
        """זיהוי אוטומטי של סוג מערכת הקבצים"""
        
        # רשימת פרסרים לבדיקה
        parsers = [
            (NTFSParser, FilesystemType.NTFS),
            (EXT4Parser, FilesystemType.EXT4),
        ]
        
        for parser_class, fs_type in parsers:
            try:
                with parser_class(image_path) as parser:
                    if parser.detect_filesystem():
                        return fs_type
            except Exception as e:
                print(f"Error testing {fs_type.value}: {e}")
                continue
        
        return FilesystemType.UNKNOWN
    
    @staticmethod
    def get_parser(image_path: str) -> Optional[FilesystemParser]:
        """קבלת פרסר מתאים למערכת הקבצים"""
        fs_type = FilesystemDetector.detect_filesystem_type(image_path)
        
        if fs_type == FilesystemType.NTFS:
            return NTFSParser(image_path)
        elif fs_type == FilesystemType.EXT4:
            return EXT4Parser(image_path)
        # Add more filesystem parsers here
        
        return None


# Example usage and testing
if __name__ == "__main__":
    # דוגמה לשימוש
    image_path = "test_disk.dd"
    
    # זיהוי אוטומטי של מערכת הקבצים
    fs_type = FilesystemDetector.detect_filesystem_type(image_path)
    print(f"Detected filesystem: {fs_type.value}")
    
    # קבלת פרסר מתאים
    parser = FilesystemDetector.get_parser(image_path)
    if parser:
        with parser as fs_parser:
            # ניתוח מידע כללי
            fs_info = fs_parser.parse_filesystem_info()
            print(f"Filesystem: {fs_info.filesystem_type.value}")
            print(f"Total size: {fs_info.total_size:,} bytes")
            print(f"Cluster size: {fs_info.cluster_size} bytes")
            
            # מניית קבצים
            print("\nEnumerating files:")
            file_count = 0
            for file_entry in fs_parser.enumerate_files():
                print(f"  {file_entry.path} ({file_entry.size:,} bytes)")
                file_count += 1
                if file_count >= 10:  # הגבלה לדוגמה
                    print("  ...")
                    break
    else:
        print("Unsupported filesystem type")