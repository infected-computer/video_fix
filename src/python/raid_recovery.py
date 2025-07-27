"""
PhoenixDRS - Advanced RAID Recovery Engine
מנוע שחזור RAID מתקדם עבור PhoenixDRS

Professional-grade RAID reconstruction capabilities supporting RAID 0, 1, 5, 6, and 10.
Includes automatic parameter detection and intelligent reconstruction algorithms.
"""

import os
import struct
import hashlib
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Iterator, Union
from dataclasses import dataclass
from enum import Enum
import itertools
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, as_completed

from logging_config import get_logger, LoggedOperation, PerformanceMonitor


class RAIDLevel(Enum):
    """רמות RAID נתמכות"""
    RAID0 = "raid0"
    RAID1 = "raid1" 
    RAID5 = "raid5"
    RAID6 = "raid6"
    RAID10 = "raid10"
    UNKNOWN = "unknown"


class RAIDStatus(Enum):
    """סטטוס מערך RAID"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    FAILED = "failed"
    REBUILDING = "rebuilding"
    UNKNOWN = "unknown"


@dataclass
class RAIDDisk:
    """דיסק במערך RAID"""
    path: str
    index: int
    size: int
    is_available: bool
    health_status: str
    serial_number: Optional[str] = None
    model: Optional[str] = None
    bad_sectors: List[int] = None
    
    def __post_init__(self):
        if self.bad_sectors is None:
            self.bad_sectors = []


@dataclass
class RAIDConfiguration:
    """תצורת מערך RAID"""
    level: RAIDLevel
    disk_count: int
    stripe_size: int  # בבתים
    chunk_size: int   # בבתים  
    total_size: int
    usable_size: int
    parity_disks: int
    disks: List[RAIDDisk]
    status: RAIDStatus
    metadata: Dict[str, any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class ReconstructionResult:
    """תוצאות שחזור RAID"""
    success: bool
    output_path: str
    original_size: int
    recovered_size: int
    sectors_recovered: int
    sectors_failed: int
    reconstruction_time: float
    confidence_score: float  # 0.0-1.0
    errors: List[str] = None
    
    def __post_init__(self):
        if self.errors is None:
            self.errors = []


class RAIDSignatureDetector:
    """גלאי חתימות RAID למערכות שונות"""
    
    # חתימות מערכות RAID שונות
    RAID_SIGNATURES = {
        # Intel Matrix RAID
        b"Intel Raid ISM Cfg Sig. ": "intel_matrix",
        b"IRSTOR": "intel_rst",
        
        # Promise RAID
        b"Promise Technology": "promise",
        
        # Adaptec RAID
        b"Adaptec": "adaptec",
        
        # 3ware RAID
        b"3ware": "3ware",
        
        # Linux Software RAID (mdadm)
        b"\xfc\x4e\x2b\xa9": "linux_mdadm_09",
        b"\xa9\x2b\x4e\xfc": "linux_mdadm_1x",
        
        # Windows Dynamic Disks
        b"PRIVHEAD": "windows_dynamic",
        
        # FreeBSD RAID
        b"GEOM::RAID": "freebsd_graid",
        
        # ZFS RAID
        b"\x0c\xb1\xba\x00": "zfs_raid",
    }
    
    @classmethod
    def detect_raid_signature(cls, disk_path: str) -> Optional[Tuple[str, int]]:
        """זיהוי חתימת RAID בדיסק"""
        logger = get_logger()
        
        try:
            with open(disk_path, 'rb') as f:
                # בדיקת תחילת הדיסק
                header = f.read(65536)  # 64KB
                
                for signature, raid_type in cls.RAID_SIGNATURES.items():
                    if signature in header:
                        offset = header.find(signature)
                        logger.debug(f"Found RAID signature", 
                                   signature=raid_type, offset=offset, disk=disk_path)
                        return raid_type, offset
                
                # בדיקת סוף הדיסק
                f.seek(-65536, 2)  # 64KB מהסוף
                footer = f.read(65536)
                
                for signature, raid_type in cls.RAID_SIGNATURES.items():
                    if signature in footer:
                        file_size = f.seek(0, 2)
                        offset = file_size - 65536 + footer.find(signature)
                        logger.debug(f"Found RAID signature at end", 
                                   signature=raid_type, offset=offset, disk=disk_path)
                        return raid_type, offset
                        
        except Exception as e:
            logger.error(f"Error detecting RAID signature in {disk_path}", exception=e)
        
        return None


class RAIDParameterDetector:
    """גלאי פרמטרים אוטומטי למערכי RAID"""
    
    def __init__(self, disks: List[str]):
        self.disks = disks
        self.logger = get_logger()
    
    def detect_stripe_size(self, max_stripe_size: int = 1024*1024) -> List[int]:
        """זיהוי גדלי stripe אפשריים"""
        possible_sizes = []
        
        # גדלי stripe נפוצים (בבתים)
        common_sizes = [
            4096, 8192, 16384, 32768, 65536,
            128*1024, 256*1024, 512*1024, 1024*1024
        ]
        
        for stripe_size in common_sizes:
            if stripe_size <= max_stripe_size:
                if self._test_stripe_size(stripe_size):
                    possible_sizes.append(stripe_size)
        
        return possible_sizes
    
    def _test_stripe_size(self, stripe_size: int) -> bool:
        """בדיקת גודל stripe על ידי חיפוש patterns"""
        try:
            # קריאת דגימות מכל דיסק
            samples = []
            for disk_path in self.disks[:4]:  # מגביל ל-4 דיסקים לבדיקה
                try:
                    with open(disk_path, 'rb') as f:
                        f.seek(stripe_size * 10)  # מתחיל מאזור נתונים
                        sample = f.read(stripe_size * 2)
                        samples.append(sample)
                except:
                    continue
            
            if len(samples) < 2:
                return False
            
            # חיפוש patterns חוזרים שמצביעים על striping
            pattern_matches = 0
            for i in range(0, min(len(samples[0]), stripe_size), 512):
                chunk = samples[0][i:i+512]
                if len(chunk) == 512:
                    for other_sample in samples[1:]:
                        if i < len(other_sample) and chunk == other_sample[i:i+512]:
                            pattern_matches += 1
                            break
            
            # יחס התאמות גבוה מצביע על striping
            total_chunks = stripe_size // 512
            match_ratio = pattern_matches / total_chunks if total_chunks > 0 else 0
            
            return match_ratio > 0.1  # 10% התאמות
            
        except Exception as e:
            self.logger.debug(f"Error testing stripe size {stripe_size}", exception=e)
            return False
    
    def detect_raid_level(self, disks: List[str]) -> RAIDLevel:
        """זיהוי רמת RAID על בסיס מספר דיסקים ו-patterns"""
        disk_count = len(disks)
        
        if disk_count < 2:
            return RAIDLevel.UNKNOWN
        
        # בדיקת RAID 1 - דיסקים זהים
        if disk_count == 2:
            if self._test_raid1_mirror(disks[0], disks[1]):
                return RAIDLevel.RAID1
        
        # בדיקת RAID 0 - striping ללא redundancy
        if self._test_raid0_striping(disks):
            return RAIDLevel.RAID0
        
        # בדיקת RAID 5 - parity distributed
        if disk_count >= 3 and self._test_raid5_parity(disks):
            return RAIDLevel.RAID5
        
        # בדיקת RAID 6 - dual parity
        if disk_count >= 4 and self._test_raid6_dual_parity(disks):
            return RAIDLevel.RAID6
        
        # בדיקת RAID 10 - striped mirrors
        if disk_count >= 4 and disk_count % 2 == 0:
            if self._test_raid10_striped_mirror(disks):
                return RAIDLevel.RAID10
        
        return RAIDLevel.UNKNOWN
    
    def _test_raid1_mirror(self, disk1: str, disk2: str) -> bool:
        """בדיקת RAID 1 - השוואת דיסקים"""
        try:
            with open(disk1, 'rb') as f1, open(disk2, 'rb') as f2:
                # השוואת כמה chunks
                for offset in [0, 1024*1024, 10*1024*1024]:
                    f1.seek(offset)
                    f2.seek(offset)
                    
                    chunk1 = f1.read(4096)
                    chunk2 = f2.read(4096)
                    
                    if len(chunk1) == len(chunk2) == 4096:
                        if chunk1 == chunk2:
                            return True
            
            return False
            
        except Exception:
            return False
    
    def _test_raid0_striping(self, disks: List[str]) -> bool:
        """בדיקת RAID 0 - striping patterns"""
        if len(disks) < 2:
            return False
        
        try:
            # בדיקת חלוקת נתונים בין דיסקים
            stripe_sizes = [64*1024, 128*1024, 256*1024]
            
            for stripe_size in stripe_sizes:
                if self._verify_striping_pattern(disks, stripe_size):
                    return True
            
            return False
            
        except Exception:
            return False
    
    def _test_raid5_parity(self, disks: List[str]) -> bool:
        """בדיקת RAID 5 - distributed parity"""
        if len(disks) < 3:
            return False
        
        try:
            # בדיקת parity calculation
            stripe_size = 64*1024
            
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = []
                for disk_path in disks:
                    future = executor.submit(self._read_disk_stripe, disk_path, 0, stripe_size)
                    futures.append(future)
                
                stripes = []
                for future in as_completed(futures):
                    try:
                        stripe_data = future.result()
                        stripes.append(stripe_data)
                    except:
                        continue
                
                if len(stripes) >= 3:
                    return self._verify_raid5_parity(stripes)
            
            return False
            
        except Exception:
            return False
    
    def _test_raid6_dual_parity(self, disks: List[str]) -> bool:
        """בדיקת RAID 6 - dual parity"""
        # מימוש מפושט - בפרקטיקה זה מורכב יותר
        return len(disks) >= 4 and self._test_raid5_parity(disks)
    
    def _test_raid10_striped_mirror(self, disks: List[str]) -> bool:
        """בדיקת RAID 10 - striped mirrors"""
        if len(disks) % 2 != 0:
            return False
        
        try:
            # בדיקת זוגות מראה
            for i in range(0, len(disks), 2):
                if i + 1 < len(disks):
                    if not self._test_raid1_mirror(disks[i], disks[i + 1]):
                        return False
            
            return True
            
        except Exception:
            return False
    
    def _verify_striping_pattern(self, disks: List[str], stripe_size: int) -> bool:
        """אימות pattern של striping"""
        try:
            # קריאת stripe מכל דיסק
            stripes = []
            for disk_path in disks:
                stripe = self._read_disk_stripe(disk_path, stripe_size, stripe_size)
                if stripe:
                    stripes.append(stripe)
            
            if len(stripes) != len(disks):
                return False
            
            # בדיקת שכל stripe שונה (אין duplication)
            for i, stripe1 in enumerate(stripes):
                for j, stripe2 in enumerate(stripes):
                    if i != j and stripe1 == stripe2:
                        return False  # דיסקים זהים = לא RAID 0
            
            return True
            
        except Exception:
            return False
    
    def _verify_raid5_parity(self, stripes: List[bytes]) -> bool:
        """אימות parity של RAID 5"""
        if len(stripes) < 3:
            return False
        
        try:
            # נניח שה-parity הוא ה-stripe האחרון
            data_stripes = stripes[:-1]
            parity_stripe = stripes[-1]
            
            # חישוב XOR של data stripes
            calculated_parity = bytearray(len(data_stripes[0]))
            
            for stripe in data_stripes:
                for i in range(min(len(calculated_parity), len(stripe))):
                    calculated_parity[i] ^= stripe[i]
            
            # השוואה ל-parity stripe
            parity_matches = 0
            total_bytes = min(len(calculated_parity), len(parity_stripe))
            
            for i in range(total_bytes):
                if calculated_parity[i] == parity_stripe[i]:
                    parity_matches += 1
            
            # אם יותר מ-80% מ-parity נכון, זה כנראה RAID 5
            match_ratio = parity_matches / total_bytes if total_bytes > 0 else 0
            return match_ratio > 0.8
            
        except Exception:
            return False
    
    def _read_disk_stripe(self, disk_path: str, offset: int, size: int) -> Optional[bytes]:
        """קריאת stripe מדיסק"""
        try:
            with open(disk_path, 'rb') as f:
                f.seek(offset)
                return f.read(size)
        except Exception:
            return None


class RAIDReconstructor:
    """מנוע שחזור RAID ראשי"""
    
    def __init__(self, max_workers: int = 4):
        self.max_workers = max_workers
        self.logger = get_logger()
    
    def auto_detect_raid(self, disk_paths: List[str]) -> Optional[RAIDConfiguration]:
        """זיהוי אוטומטי של תצורת RAID"""
        with LoggedOperation("raid_auto_detection", target=f"{len(disk_paths)} disks") as op:
            self.logger.info(f"Starting RAID auto-detection", disk_count=len(disk_paths))
            
            # יצירת אובייקטי דיסק
            disks = []
            for i, disk_path in enumerate(disk_paths):
                if os.path.exists(disk_path):
                    size = os.path.getsize(disk_path)
                    disk = RAIDDisk(
                        path=disk_path,
                        index=i,
                        size=size,
                        is_available=True,
                        health_status="unknown"
                    )
                    disks.append(disk)
                else:
                    self.logger.warning(f"Disk not found: {disk_path}")
            
            if len(disks) < 2:
                self.logger.error("Insufficient disks for RAID detection")
                return None
            
            # זיהוי חתימות RAID
            detected_signatures = []
            for disk in disks:
                signature = RAIDSignatureDetector.detect_raid_signature(disk.path)
                if signature:
                    detected_signatures.append(signature)
            
            # זיהוי פרמטרים
            detector = RAIDParameterDetector([disk.path for disk in disks])
            
            # זיהוי רמת RAID
            raid_level = detector.detect_raid_level([disk.path for disk in disks])
            self.logger.info(f"Detected RAID level: {raid_level.value}")
            
            # זיהוי גדלי stripe
            possible_stripe_sizes = detector.detect_stripe_size()
            stripe_size = possible_stripe_sizes[0] if possible_stripe_sizes else 65536
            
            # חישוב גדלים
            total_size = sum(disk.size for disk in disks)
            usable_size = self._calculate_usable_size(raid_level, disks, total_size)
            parity_disks = self._get_parity_disk_count(raid_level)
            
            config = RAIDConfiguration(
                level=raid_level,
                disk_count=len(disks),
                stripe_size=stripe_size,
                chunk_size=stripe_size // len(disks) if len(disks) > 0 else stripe_size,
                total_size=total_size,
                usable_size=usable_size,
                parity_disks=parity_disks,
                disks=disks,
                status=RAIDStatus.UNKNOWN,
                metadata={
                    "detected_signatures": detected_signatures,
                    "possible_stripe_sizes": possible_stripe_sizes
                }
            )
            
            self.logger.info("RAID detection completed", 
                           level=raid_level.value, 
                           stripe_size=stripe_size,
                           usable_size=usable_size)
            
            return config
    
    def reconstruct_raid(self, config: RAIDConfiguration, output_path: str) -> ReconstructionResult:
        """שחזור מערך RAID לקובץ יחיד"""
        
        with LoggedOperation("raid_reconstruction", target=output_path) as op:
            start_time = __import__('time').time()
            
            self.logger.info("Starting RAID reconstruction", 
                           level=config.level.value,
                           output_path=output_path,
                           disk_count=config.disk_count)
            
            try:
                if config.level == RAIDLevel.RAID0:
                    result = self._reconstruct_raid0(config, output_path)
                elif config.level == RAIDLevel.RAID1:
                    result = self._reconstruct_raid1(config, output_path)
                elif config.level == RAIDLevel.RAID5:
                    result = self._reconstruct_raid5(config, output_path)
                elif config.level == RAIDLevel.RAID6:
                    result = self._reconstruct_raid6(config, output_path)
                elif config.level == RAIDLevel.RAID10:
                    result = self._reconstruct_raid10(config, output_path)
                else:
                    raise ValueError(f"Unsupported RAID level: {config.level}")
                
                end_time = __import__('time').time()
                result.reconstruction_time = end_time - start_time
                
                self.logger.info("RAID reconstruction completed",
                               success=result.success,
                               recovered_size=result.recovered_size,
                               time_taken=result.reconstruction_time)
                
                return result
                
            except Exception as e:
                self.logger.error("RAID reconstruction failed", exception=e)
                
                end_time = __import__('time').time()
                return ReconstructionResult(
                    success=False,
                    output_path=output_path,
                    original_size=config.usable_size,
                    recovered_size=0,
                    sectors_recovered=0,
                    sectors_failed=0,
                    reconstruction_time=end_time - start_time,
                    confidence_score=0.0,
                    errors=[str(e)]
                )
    
    def _reconstruct_raid0(self, config: RAIDConfiguration, output_path: str) -> ReconstructionResult:
        """שחזור RAID 0 - Striping"""
        self.logger.info("Reconstructing RAID 0")
        
        available_disks = [disk for disk in config.disks if disk.is_available]
        if len(available_disks) != config.disk_count:
            raise ValueError(f"RAID 0 requires all {config.disk_count} disks")
        
        sectors_recovered = 0
        sectors_failed = 0
        
        with open(output_path, 'wb') as output_file:
            stripe_index = 0
            
            while True:
                stripe_written = False
                
                # קריאת stripe מכל דיסק
                for disk in available_disks:
                    try:
                        with open(disk.path, 'rb') as disk_file:
                            offset = stripe_index * config.stripe_size
                            disk_file.seek(offset)
                            stripe_data = disk_file.read(config.stripe_size)
                            
                            if not stripe_data:
                                break  # הגענו לסוף
                            
                            output_file.write(stripe_data)
                            sectors_recovered += len(stripe_data) // 512
                            stripe_written = True
                            
                    except Exception as e:
                        self.logger.warning(f"Error reading stripe {stripe_index} from {disk.path}", 
                                          exception=e)
                        sectors_failed += config.stripe_size // 512
                        # כתיבת אפסים במקום הנתונים החסרים
                        output_file.write(b'\x00' * config.stripe_size)
                
                if not stripe_written:
                    break
                
                stripe_index += 1
                
                if stripe_index % 1000 == 0:
                    self.logger.debug(f"Processed {stripe_index} stripes")
        
        recovered_size = os.path.getsize(output_path)
        confidence_score = sectors_recovered / (sectors_recovered + sectors_failed) if (sectors_recovered + sectors_failed) > 0 else 0.0
        
        return ReconstructionResult(
            success=True,
            output_path=output_path,
            original_size=config.usable_size,
            recovered_size=recovered_size,
            sectors_recovered=sectors_recovered,
            sectors_failed=sectors_failed,
            reconstruction_time=0.0,  # יחושב ברמה גבוהה יותר
            confidence_score=confidence_score,
            errors=[]
        )
    
    def _reconstruct_raid1(self, config: RAIDConfiguration, output_path: str) -> ReconstructionResult:
        """שחזור RAID 1 - Mirroring"""
        self.logger.info("Reconstructing RAID 1")
        
        available_disks = [disk for disk in config.disks if disk.is_available]
        if len(available_disks) == 0:
            raise ValueError("No available disks for RAID 1 reconstruction")
        
        # בחירת הדיסק הטוב ביותר (עם הכי מעט bad sectors)
        best_disk = min(available_disks, key=lambda d: len(d.bad_sectors))
        self.logger.info(f"Using disk {best_disk.path} as primary source")
        
        # העתקת הדיסק הטוב ביותר
        sectors_recovered = 0
        sectors_failed = 0
        
        with open(best_disk.path, 'rb') as source_file, open(output_path, 'wb') as output_file:
            chunk_size = 1024 * 1024  # 1MB chunks
            
            while True:
                chunk = source_file.read(chunk_size)
                if not chunk:
                    break
                
                output_file.write(chunk)
                sectors_recovered += len(chunk) // 512
        
        recovered_size = os.path.getsize(output_path)
        
        return ReconstructionResult(
            success=True,
            output_path=output_path,
            original_size=best_disk.size,
            recovered_size=recovered_size,
            sectors_recovered=sectors_recovered,
            sectors_failed=sectors_failed,
            reconstruction_time=0.0,
            confidence_score=1.0,  # RAID 1 שלם
            errors=[]
        )
    
    def _reconstruct_raid5(self, config: RAIDConfiguration, output_path: str) -> ReconstructionResult:
        """שחזור RAID 5 - Distributed Parity"""
        self.logger.info("Reconstructing RAID 5")
        
        available_disks = [disk for disk in config.disks if disk.is_available]
        missing_disks = config.disk_count - len(available_disks)
        
        if missing_disks > 1:
            raise ValueError(f"RAID 5 can only recover from 1 missing disk, {missing_disks} missing")
        
        sectors_recovered = 0
        sectors_failed = 0
        
        with open(output_path, 'wb') as output_file:
            stripe_index = 0
            
            while True:
                try:
                    # חישוב איזה דיסק מכיל parity עבור stripe זה
                    parity_disk_index = stripe_index % config.disk_count
                    
                    # קריאת stripes מכל הדיסקים הזמינים
                    stripe_data = {}
                    data_available = False
                    
                    for disk in available_disks:
                        offset = stripe_index * config.stripe_size
                        
                        try:
                            with open(disk.path, 'rb') as disk_file:
                                disk_file.seek(offset)
                                data = disk_file.read(config.stripe_size)
                                
                                if data:
                                    stripe_data[disk.index] = data
                                    data_available = True
                                    
                        except Exception as e:
                            self.logger.debug(f"Error reading stripe {stripe_index} from disk {disk.index}")
                    
                    if not data_available:
                        break  # הגענו לסוף
                    
                    # שחזור הנתונים (אם יש דיסק חסר)
                    if missing_disks == 1:
                        reconstructed_stripe = self._reconstruct_raid5_stripe(
                            stripe_data, config.disk_count, parity_disk_index, config.stripe_size
                        )
                    else:
                        # כל הדיסקים זמינים - פשוט מחבר את הנתונים
                        reconstructed_stripe = b""
                        for i in range(config.disk_count):
                            if i != parity_disk_index and i in stripe_data:
                                reconstructed_stripe += stripe_data[i]
                    
                    if reconstructed_stripe:
                        output_file.write(reconstructed_stripe)
                        sectors_recovered += len(reconstructed_stripe) // 512
                    else:
                        sectors_failed += config.stripe_size // 512
                    
                    stripe_index += 1
                    
                    if stripe_index % 1000 == 0:
                        self.logger.debug(f"Processed {stripe_index} RAID 5 stripes")
                
                except Exception as e:
                    self.logger.warning(f"Error processing RAID 5 stripe {stripe_index}", exception=e)
                    sectors_failed += config.stripe_size // 512
                    stripe_index += 1
        
        recovered_size = os.path.getsize(output_path)
        confidence_score = sectors_recovered / (sectors_recovered + sectors_failed) if (sectors_recovered + sectors_failed) > 0 else 0.0
        
        return ReconstructionResult(
            success=True,
            output_path=output_path,
            original_size=config.usable_size,
            recovered_size=recovered_size,
            sectors_recovered=sectors_recovered,
            sectors_failed=sectors_failed,
            reconstruction_time=0.0,
            confidence_score=confidence_score,
            errors=[]
        )
    
    def _reconstruct_raid5_stripe(self, stripe_data: Dict[int, bytes], 
                                 disk_count: int, parity_disk_index: int, 
                                 stripe_size: int) -> bytes:
        """שחזור stripe יחיד של RAID 5"""
        
        # מציאת הדיסק החסר
        available_disks = set(stripe_data.keys())
        all_disks = set(range(disk_count))
        missing_disks = all_disks - available_disks
        
        if len(missing_disks) > 1:
            return b""  # לא יכול לשחזר יותר מדיסק אחד
        
        if len(missing_disks) == 0:
            # כל הדיסקים זמינים - פשוט מחזיר את הנתונים (ללא parity)
            result = b""
            for i in range(disk_count):
                if i != parity_disk_index and i in stripe_data:
                    result += stripe_data[i]
            return result
        
        # יש דיסק חסר אחד - משחזר באמצעות XOR
        missing_disk = missing_disks.pop()
        
        # חישוב XOR של כל הדיסקים הזמינים
        xor_result = bytearray(stripe_size)
        
        for disk_index in available_disks:
            data = stripe_data[disk_index]
            for i in range(min(len(xor_result), len(data))):
                xor_result[i] ^= data[i]
        
        # אם הדיסק החסר הוא parity, מחזיר את הנתונים
        if missing_disk == parity_disk_index:
            result = b""
            for i in range(disk_count):
                if i != parity_disk_index and i in stripe_data:
                    result += stripe_data[i]
            return result
        
        # אם הדיסק החסר הוא data disk, השחזור הוא תוצאת ה-XOR
        result = b""
        for i in range(disk_count):
            if i == missing_disk:
                result += bytes(xor_result)
            elif i != parity_disk_index and i in stripe_data:
                result += stripe_data[i]
        
        return result
    
    def _reconstruct_raid6(self, config: RAIDConfiguration, output_path: str) -> ReconstructionResult:
        """שחזור RAID 6 - Dual Parity (מימוש מפושט)"""
        self.logger.info("Reconstructing RAID 6 (simplified implementation)")
        
        # כרגע מימוש מפושט שמתבסס על RAID 5
        # במציאות RAID 6 מורכב הרבה יותר עם Reed-Solomon codes
        
        available_disks = [disk for disk in config.disks if disk.is_available]
        missing_disks = config.disk_count - len(available_disks)
        
        if missing_disks > 2:
            raise ValueError(f"RAID 6 can only recover from 2 missing disks, {missing_disks} missing")
        
        # אם יש רק דיסק חסר אחד או פחות, משתמש באלגוריתם של RAID 5
        if missing_disks <= 1:
            return self._reconstruct_raid5(config, output_path)
        
        # עבור 2 דיסקים חסרים - מימוש מפושט
        self.logger.warning("RAID 6 dual-disk recovery not fully implemented - using fallback")
        
        return ReconstructionResult(
            success=False,
            output_path=output_path,
            original_size=config.usable_size,
            recovered_size=0,
            sectors_recovered=0,
            sectors_failed=0,
            reconstruction_time=0.0,
            confidence_score=0.0,
            errors=["RAID 6 dual-disk recovery not implemented"]
        )
    
    def _reconstruct_raid10(self, config: RAIDConfiguration, output_path: str) -> ReconstructionResult:
        """שחזור RAID 10 - Striped Mirrors"""
        self.logger.info("Reconstructing RAID 10")
        
        if config.disk_count % 2 != 0:
            raise ValueError("RAID 10 requires even number of disks")
        
        available_disks = [disk for disk in config.disks if disk.is_available]
        
        # יצירת זוגות מראה
        mirror_pairs = []
        for i in range(0, config.disk_count, 2):
            pair_disks = [disk for disk in available_disks if disk.index in [i, i+1]]
            if pair_disks:
                mirror_pairs.append(pair_disks)
        
        sectors_recovered = 0
        sectors_failed = 0
        
        with open(output_path, 'wb') as output_file:
            stripe_index = 0
            
            while True:
                stripe_written = False
                
                # קריאה מכל זוג מראה
                for pair in mirror_pairs:
                    if not pair:
                        continue
                    
                    # בחירת הדיסק הטוב ביותר מהזוג
                    best_disk = min(pair, key=lambda d: len(d.bad_sectors))
                    
                    try:
                        with open(best_disk.path, 'rb') as disk_file:
                            offset = stripe_index * config.stripe_size
                            disk_file.seek(offset)
                            stripe_data = disk_file.read(config.stripe_size)
                            
                            if not stripe_data:
                                break
                            
                            output_file.write(stripe_data)
                            sectors_recovered += len(stripe_data) // 512
                            stripe_written = True
                            
                    except Exception as e:
                        self.logger.warning(f"Error reading RAID 10 stripe {stripe_index} from {best_disk.path}")
                        sectors_failed += config.stripe_size // 512
                        output_file.write(b'\x00' * config.stripe_size)
                
                if not stripe_written:
                    break
                
                stripe_index += 1
                
                if stripe_index % 1000 == 0:
                    self.logger.debug(f"Processed {stripe_index} RAID 10 stripes")
        
        recovered_size = os.path.getsize(output_path)
        confidence_score = sectors_recovered / (sectors_recovered + sectors_failed) if (sectors_recovered + sectors_failed) > 0 else 0.0
        
        return ReconstructionResult(
            success=True,
            output_path=output_path,
            original_size=config.usable_size,
            recovered_size=recovered_size,
            sectors_recovered=sectors_recovered,
            sectors_failed=sectors_failed,
            reconstruction_time=0.0,
            confidence_score=confidence_score,
            errors=[]
        )
    
    def _calculate_usable_size(self, raid_level: RAIDLevel, disks: List[RAIDDisk], total_size: int) -> int:
        """חישוב נפח שמיש במערך RAID"""
        if not disks:
            return 0
        
        min_disk_size = min(disk.size for disk in disks)
        disk_count = len(disks)
        
        if raid_level == RAIDLevel.RAID0:
            return min_disk_size * disk_count
        elif raid_level == RAIDLevel.RAID1:
            return min_disk_size  # רק דיסק אחד של נתונים
        elif raid_level == RAIDLevel.RAID5:
            return min_disk_size * (disk_count - 1)  # דיסק אחד לparity
        elif raid_level == RAIDLevel.RAID6:
            return min_disk_size * (disk_count - 2)  # שני דיסקים לparity
        elif raid_level == RAIDLevel.RAID10:
            return min_disk_size * (disk_count // 2)  # מחצית מהדיסקים לנתונים
        else:
            return min_disk_size * disk_count  # הערכה
    
    def _get_parity_disk_count(self, raid_level: RAIDLevel) -> int:
        """קבלת מספר דיסקי parity"""
        if raid_level in [RAIDLevel.RAID0, RAIDLevel.RAID1, RAIDLevel.RAID10]:
            return 0
        elif raid_level == RAIDLevel.RAID5:
            return 1
        elif raid_level == RAIDLevel.RAID6:
            return 2
        else:
            return 0


# Example usage and CLI integration
def main():
    """דוגמה לשימוש במנוע RAID"""
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python raid_recovery.py <disk1> <disk2> [disk3...] <output>")
        return 1
    
    disk_paths = sys.argv[1:-1]
    output_path = sys.argv[-1]
    
    print(f"PhoenixDRS RAID Recovery")
    print(f"Input disks: {disk_paths}")
    print(f"Output: {output_path}")
    
    # יצירת מנוע שחזור
    reconstructor = RAIDReconstructor()
    
    # זיהוי אוטומטי
    config = reconstructor.auto_detect_raid(disk_paths)
    if not config:
        print("Failed to detect RAID configuration")
        return 1
    
    print(f"Detected: {config.level.value.upper()}")
    print(f"Stripe size: {config.stripe_size:,} bytes")
    print(f"Usable size: {config.usable_size:,} bytes")
    
    # שחזור
    result = reconstructor.reconstruct_raid(config, output_path) 
    
    if result.success:
        print(f"✓ Reconstruction successful!")
        print(f"  Recovered: {result.recovered_size:,} bytes")
        print(f"  Time: {result.reconstruction_time:.2f} seconds")
        print(f"  Confidence: {result.confidence_score:.1%}")
    else:
        print(f"✗ Reconstruction failed:")
        for error in result.errors:
            print(f"  - {error}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())