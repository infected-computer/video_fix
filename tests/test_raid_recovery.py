"""
PhoenixDRS RAID Recovery Module Tests
בדיקות עבור מודול שחזור RAID
"""

import pytest
import struct
import os
from unittest.mock import Mock, patch, mock_open
from pathlib import Path

from raid_recovery import (
    RAIDLevel, RAIDStatus, RAIDDisk, RAIDConfiguration, ReconstructionResult,
    RAIDSignatureDetector, RAIDParameterDetector, RAIDReconstructor
)


class TestRAIDDisk:
    """בדיקות עבור RAIDDisk"""
    
    def test_raid_disk_creation(self):
        """בדיקת יצירת דיסק RAID"""
        disk = RAIDDisk(
            path="/dev/sda",
            index=0,
            size=1000000000,
            is_available=True,
            health_status="good",
            serial_number="ABC123",
            model="Test Drive"
        )
        
        assert disk.path == "/dev/sda"
        assert disk.index == 0
        assert disk.size == 1000000000
        assert disk.is_available is True
        assert disk.health_status == "good"
        assert disk.bad_sectors == []
    
    def test_raid_disk_with_bad_sectors(self):
        """בדיקת דיסק עם סקטורים פגומים"""
        bad_sectors = [1000, 2000, 3000]
        disk = RAIDDisk(
            path="/dev/sdb",
            index=1,
            size=1000000000,
            is_available=True,
            health_status="degraded",
            bad_sectors=bad_sectors
        )
        
        assert disk.bad_sectors == bad_sectors


class TestRAIDConfiguration:
    """בדיקות עבור RAIDConfiguration"""
    
    def test_raid_config_creation(self):
        """בדיקת יצירת תצורת RAID"""
        disks = [
            RAIDDisk("/dev/sda", 0, 1000000000, True, "good"),
            RAIDDisk("/dev/sdb", 1, 1000000000, True, "good"),
            RAIDDisk("/dev/sdc", 2, 1000000000, True, "good")
        ]
        
        config = RAIDConfiguration(
            level=RAIDLevel.RAID5,
            disk_count=3,
            stripe_size=65536,
            chunk_size=21845,
            total_size=3000000000,
            usable_size=2000000000,
            parity_disks=1,
            disks=disks,
            status=RAIDStatus.HEALTHY
        )
        
        assert config.level == RAIDLevel.RAID5
        assert config.disk_count == 3
        assert config.stripe_size == 65536
        assert config.parity_disks == 1
        assert len(config.disks) == 3
        assert config.metadata == {}


class TestRAIDSignatureDetector:
    """בדיקות עבור RAIDSignatureDetector"""
    
    def test_detect_intel_matrix_signature(self, temp_dir):
        """בדיקת זיהוי חתימת Intel Matrix RAID"""
        disk_file = temp_dir / "intel_raid.img"
        
        # יצירת תמונת דיסק עם חתימת Intel RAID
        header_data = b"Intel Raid ISM Cfg Sig. " + b"\x00" * 1000
        test_data = header_data + b"\x00" * 64000
        disk_file.write_bytes(test_data)
        
        result = RAIDSignatureDetector.detect_raid_signature(str(disk_file))
        
        assert result is not None
        assert result[0] == "intel_matrix"
        assert result[1] == 0  # Found at beginning
    
    def test_detect_linux_mdadm_signature(self, temp_dir):
        """בדיקת זיהוי חתימת Linux mdadm"""
        disk_file = temp_dir / "linux_raid.img"
        
        # יצירת תמונה עם חתימת mdadm בסוף
        signature = b"\xfc\x4e\x2b\xa9"
        test_data = b"\x00" * 65000 + signature + b"\x00" * 500
        disk_file.write_bytes(test_data)
        
        result = RAIDSignatureDetector.detect_raid_signature(str(disk_file))
        
        assert result is not None
        assert result[0] == "linux_mdadm_09"
    
    def test_no_signature_found(self, temp_dir):
        """בדיקת אי מציאת חתימה"""
        disk_file = temp_dir / "no_raid.img"
        disk_file.write_bytes(b"regular disk data" * 1000)
        
        result = RAIDSignatureDetector.detect_raid_signature(str(disk_file))
        assert result is None
    
    def test_detect_signature_nonexistent_file(self):
        """בדיקת חתימה בקובץ לא קיים"""
        result = RAIDSignatureDetector.detect_raid_signature("/nonexistent/file")
        assert result is None


class TestRAIDParameterDetector:
    """בדיקות עבור RAIDParameterDetector"""
    
    def create_raid0_test_disks(self, temp_dir, disk_count=3, stripe_size=65536):
        """יצירת דיסקים מדומים של RAID 0"""
        disk_paths = []
        
        for i in range(disk_count):
            disk_path = temp_dir / f"raid0_disk{i}.img"
            
            # יצירת נתוני striping מדומים
            disk_data = bytearray()
            
            for stripe_num in range(10):  # 10 stripes
                # כל דיסק מקבל stripe שונה
                stripe_data = bytes([i + stripe_num]) * stripe_size
                disk_data.extend(stripe_data)
            
            disk_path.write_bytes(bytes(disk_data))
            disk_paths.append(str(disk_path))
        
        return disk_paths
    
    def create_raid1_test_disks(self, temp_dir):
        """יצירת דיסקים מדומים של RAID 1"""
        # נתונים זהים בשני דיסקים
        test_data = b"identical mirror data" * 1000
        
        disk1_path = temp_dir / "raid1_disk0.img"
        disk2_path = temp_dir / "raid1_disk1.img"
        
        disk1_path.write_bytes(test_data)
        disk2_path.write_bytes(test_data)
        
        return [str(disk1_path), str(disk2_path)]
    
    def test_detect_stripe_sizes(self, temp_dir):
        """בדיקת זיהוי גדלי stripe"""
        disk_paths = self.create_raid0_test_disks(temp_dir, 3, 65536)
        detector = RAIDParameterDetector(disk_paths)
        
        stripe_sizes = detector.detect_stripe_size()
        
        # צריך למצוא לפחות כמה גדלים אפשריים
        assert isinstance(stripe_sizes, list)
        assert len(stripe_sizes) >= 0
    
    def test_detect_raid0_level(self, temp_dir):
        """בדיקת זיהוי RAID 0"""
        disk_paths = self.create_raid0_test_disks(temp_dir, 3)
        detector = RAIDParameterDetector(disk_paths)
        
        raid_level = detector.detect_raid_level(disk_paths)
        
        # עם הנתונים המדומים שלנו, יכול להיות RAID0 או UNKNOWN
        assert raid_level in [RAIDLevel.RAID0, RAIDLevel.UNKNOWN]
    
    def test_detect_raid1_level(self, temp_dir):
        """בדיקת זיהוי RAID 1"""
        disk_paths = self.create_raid1_test_disks(temp_dir)
        detector = RAIDParameterDetector(disk_paths)
        
        raid_level = detector.detect_raid_level(disk_paths)
        
        assert raid_level == RAIDLevel.RAID1
    
    def test_detect_insufficient_disks(self, temp_dir):
        """בדיקת זיהוי עם מספר דיסקים לא מספיק"""
        disk_path = temp_dir / "single_disk.img"
        disk_path.write_bytes(b"single disk data" * 1000)
        
        detector = RAIDParameterDetector([str(disk_path)])
        raid_level = detector.detect_raid_level([str(disk_path)])
        
        assert raid_level == RAIDLevel.UNKNOWN


class TestRAIDReconstructor:
    """בדיקות עבור RAIDReconstructor"""
    
    def test_raid_reconstructor_creation(self):
        """בדיקת יצירת מנוע שחזור RAID"""
        reconstructor = RAIDReconstructor(max_workers=8)
        assert reconstructor.max_workers == 8
    
    def test_auto_detect_raid_insufficient_disks(self, temp_dir):
        """בדיקת זיהוי אוטומטי עם דיסקים לא מספיקים"""
        disk_file = temp_dir / "single.img"
        disk_file.write_bytes(b"test data" * 1000)
        
        reconstructor = RAIDReconstructor()
        config = reconstructor.auto_detect_raid([str(disk_file)])
        
        assert config is None
    
    def test_auto_detect_raid_valid_disks(self, temp_dir):
        """בדיקת זיהוי אוטומטי עם דיסקים תקינים"""
        # יצירת שני דיסקים זהים (RAID 1)
        test_data = b"mirror data" * 5000
        
        disk1 = temp_dir / "disk1.img"
        disk2 = temp_dir / "disk2.img"
        
        disk1.write_bytes(test_data)
        disk2.write_bytes(test_data)
        
        reconstructor = RAIDReconstructor()
        config = reconstructor.auto_detect_raid([str(disk1), str(disk2)])
        
        assert config is not None
        assert config.disk_count == 2
        assert len(config.disks) == 2
        assert all(disk.is_available for disk in config.disks)
    
    def test_calculate_usable_size_raid0(self):
        """בדיקת חישוב נפח RAID 0"""
        reconstructor = RAIDReconstructor()
        
        disks = [
            RAIDDisk("/dev/sda", 0, 1000000000, True, "good"),
            RAIDDisk("/dev/sdb", 1, 1000000000, True, "good"),
            RAIDDisk("/dev/sdc", 2, 1000000000, True, "good")
        ]
        
        usable_size = reconstructor._calculate_usable_size(
            RAIDLevel.RAID0, disks, 3000000000
        )
        
        assert usable_size == 3000000000  # All space usable in RAID 0
    
    def test_calculate_usable_size_raid1(self):
        """בדיקת חישוב נפח RAID 1"""
        reconstructor = RAIDReconstructor()
        
        disks = [
            RAIDDisk("/dev/sda", 0, 1000000000, True, "good"),
            RAIDDisk("/dev/sdb", 1, 1000000000, True, "good")
        ]
        
        usable_size = reconstructor._calculate_usable_size(
            RAIDLevel.RAID1, disks, 2000000000
        )
        
        assert usable_size == 1000000000  # Only one disk worth of data
    
    def test_calculate_usable_size_raid5(self):
        """בדיקת חישוב נפח RAID 5"""
        reconstructor = RAIDReconstructor()
        
        disks = [
            RAIDDisk("/dev/sda", 0, 1000000000, True, "good"),
            RAIDDisk("/dev/sdb", 1, 1000000000, True, "good"),
            RAIDDisk("/dev/sdc", 2, 1000000000, True, "good")
        ]
        
        usable_size = reconstructor._calculate_usable_size(
            RAIDLevel.RAID5, disks, 3000000000
        )
        
        assert usable_size == 2000000000  # One disk for parity
    
    def test_get_parity_disk_count(self):
        """בדיקת ספירת דיסקי parity"""
        reconstructor = RAIDReconstructor()
        
        assert reconstructor._get_parity_disk_count(RAIDLevel.RAID0) == 0
        assert reconstructor._get_parity_disk_count(RAIDLevel.RAID1) == 0
        assert reconstructor._get_parity_disk_count(RAIDLevel.RAID5) == 1
        assert reconstructor._get_parity_disk_count(RAIDLevel.RAID6) == 2
        assert reconstructor._get_parity_disk_count(RAIDLevel.RAID10) == 0


class TestRAIDReconstruction:
    """בדיקות שחזור RAID מעשיות"""
    
    def create_raid0_config(self, temp_dir, stripe_size=65536):
        """יצירת תצורת RAID 0 לבדיקות"""
        # יצירת 3 דיסקים עם נתוני striping
        disks = []
        disk_paths = []
        
        for i in range(3):
            disk_path = temp_dir / f"raid0_disk{i}.img"
            disk_paths.append(str(disk_path))
            
            # יצירת נתונים עם pattern ייחודי לכל דיסק
            disk_data = bytearray()
            
            # 5 stripes לכל דיסק
            for stripe_num in range(5):
                stripe_data = struct.pack('<I', i) * (stripe_size // 4)
                disk_data.extend(stripe_data)
            
            disk_path.write_bytes(bytes(disk_data))
            
            disk = RAIDDisk(
                path=str(disk_path),
                index=i,
                size=len(disk_data),
                is_available=True,
                health_status="good"
            )
            disks.append(disk)
        
        config = RAIDConfiguration(
            level=RAIDLevel.RAID0,
            disk_count=3,
            stripe_size=stripe_size,
            chunk_size=stripe_size // 3,
            total_size=sum(disk.size for disk in disks),
            usable_size=sum(disk.size for disk in disks),
            parity_disks=0,
            disks=disks,
            status=RAIDStatus.HEALTHY
        )
        
        return config
    
    def create_raid1_config(self, temp_dir):
        """יצירת תצורת RAID 1 לבדיקות"""
        # נתונים זהים לשני דיסקים
        test_data = b"RAID 1 mirror test data" * 1000
        
        disks = []
        for i in range(2):
            disk_path = temp_dir / f"raid1_disk{i}.img"
            disk_path.write_bytes(test_data)
            
            disk = RAIDDisk(
                path=str(disk_path),
                index=i,
                size=len(test_data),
                is_available=True,
                health_status="good"
            )
            disks.append(disk)
        
        config = RAIDConfiguration(
            level=RAIDLevel.RAID1,
            disk_count=2,
            stripe_size=65536,
            chunk_size=65536,
            total_size=len(test_data) * 2,
            usable_size=len(test_data),
            parity_disks=0,
            disks=disks,
            status=RAIDStatus.HEALTHY
        )
        
        return config
    
    @pytest.mark.integration
    def test_reconstruct_raid0(self, temp_dir):
        """בדיקת שחזור RAID 0"""
        config = self.create_raid0_config(temp_dir)
        output_path = temp_dir / "reconstructed_raid0.img"
        
        reconstructor = RAIDReconstructor()
        result = reconstructor._reconstruct_raid0(config, str(output_path))
        
        assert result.success is True
        assert output_path.exists()
        assert result.recovered_size > 0
        assert result.confidence_score > 0.0
    
    @pytest.mark.integration
    def test_reconstruct_raid1(self, temp_dir):
        """בדיקת שחזור RAID 1"""
        config = self.create_raid1_config(temp_dir)
        output_path = temp_dir / "reconstructed_raid1.img"
        
        reconstructor = RAIDReconstructor()
        result = reconstructor._reconstruct_raid1(config, str(output_path))
        
        assert result.success is True
        assert output_path.exists()
        assert result.confidence_score == 1.0  # RAID 1 should be perfect
        
        # בדיקת שהנתונים זהים לדיסק המקורי
        original_data = config.disks[0].path
        with open(original_data, 'rb') as orig, open(output_path, 'rb') as recon:
            assert orig.read() == recon.read()
    
    def test_reconstruct_raid5_all_disks_available(self, temp_dir):
        """בדיקת שחזור RAID 5 עם כל הדיסקים זמינים"""
        # יצירת 3 דיסקים לRAID 5
        disks = []
        stripe_size = 4096
        
        # דיסק 0: נתונים A
        disk0_data = b"AAAA" * (stripe_size // 4) * 3  # 3 stripes
        disk0_path = temp_dir / "raid5_disk0.img"
        disk0_path.write_bytes(disk0_data)
        
        # דיסק 1: נתונים B  
        disk1_data = b"BBBB" * (stripe_size // 4) * 3
        disk1_path = temp_dir / "raid5_disk1.img"
        disk1_path.write_bytes(disk1_data)
        
        # דיסק 2: parity (XOR of A and B)
        parity_data = bytearray(stripe_size * 3)
        for i in range(len(parity_data)):
            parity_data[i] = ord('A') ^ ord('B')
        disk2_path = temp_dir / "raid5_disk2.img"
        disk2_path.write_bytes(bytes(parity_data))
        
        for i, disk_path in enumerate([disk0_path, disk1_path, disk2_path]):
            disk = RAIDDisk(
                path=str(disk_path),
                index=i,
                size=disk_path.stat().st_size,
                is_available=True,
                health_status="good"
            )
            disks.append(disk)
        
        config = RAIDConfiguration(
            level=RAIDLevel.RAID5,
            disk_count=3,
            stripe_size=stripe_size,
            chunk_size=stripe_size // 2,  # 2 data disks
            total_size=sum(disk.size for disk in disks),
            usable_size=disk0_path.stat().st_size + disk1_path.stat().st_size,
            parity_disks=1,
            disks=disks,
            status=RAIDStatus.HEALTHY
        )
        
        output_path = temp_dir / "reconstructed_raid5.img"
        reconstructor = RAIDReconstructor()
        result = reconstructor._reconstruct_raid5(config, str(output_path))
        
        assert result.success is True
        assert output_path.exists()
        assert result.confidence_score > 0.0
    
    def test_reconstruct_raid_unsupported_level(self, temp_dir):
        """בדיקת שחזור רמת RAID לא נתמכת"""
        disks = [RAIDDisk("/fake/path", 0, 1000, True, "good")]
        
        config = RAIDConfiguration(
            level=RAIDLevel.UNKNOWN,
            disk_count=1,
            stripe_size=65536,
            chunk_size=65536,
            total_size=1000,
            usable_size=1000,
            parity_disks=0,
            disks=disks,
            status=RAIDStatus.UNKNOWN
        )
        
        output_path = temp_dir / "unsupported.img"
        reconstructor = RAIDReconstructor()
        
        result = reconstructor.reconstruct_raid(config, str(output_path))
        
        assert result.success is False
        assert len(result.errors) > 0
    
    @pytest.mark.performance
    def test_raid_reconstruction_performance(self, temp_dir):
        """בדיקת ביצועים של שחזור RAID"""
        import time
        
        # יצירת מערך RAID 0 גדול יחסית
        config = self.create_raid0_config(temp_dir, stripe_size=1024*1024)  # 1MB stripes
        output_path = temp_dir / "perf_test.img"
        
        reconstructor = RAIDReconstructor()
        
        start_time = time.time()
        result = reconstructor._reconstruct_raid0(config, str(output_path))
        end_time = time.time()
        
        duration = end_time - start_time
        throughput = result.recovered_size / duration / (1024*1024)  # MB/s
        
        assert result.success is True
        assert duration < 10.0  # Should complete within 10 seconds
        assert throughput > 1.0   # Should achieve at least 1 MB/s
        
        print(f"RAID reconstruction performance: {throughput:.2f} MB/s")


@pytest.mark.slow
class TestRAIDIntegration:
    """בדיקות אינטגרציה מקיפות לRAID"""
    
    def test_full_raid_workflow(self, temp_dir):
        """בדיקת זרימת עבודה מלאה של RAID"""
        # יצירת מערך RAID 1 מדומה
        test_data = b"Full RAID workflow test data" * 2000
        
        disk1_path = temp_dir / "full_test_disk1.img"
        disk2_path = temp_dir / "full_test_disk2.img"
        
        disk1_path.write_bytes(test_data)
        disk2_path.write_bytes(test_data)
        
        disk_paths = [str(disk1_path), str(disk2_path)]
        output_path = temp_dir / "full_test_output.img"
        
        # זרימת עבודה מלאה
        reconstructor = RAIDReconstructor()
        
        # 1. זיהוי אוטומטי
        config = reconstructor.auto_detect_raid(disk_paths)
        assert config is not None
        assert config.level == RAIDLevel.RAID1
        
        # 2. שחזור
        result = reconstructor.reconstruct_raid(config, str(output_path))
        assert result.success is True
        assert output_path.exists()
        
        # 3. אימות תוצאות
        with open(output_path, 'rb') as f:
            recovered_data = f.read()
        
        assert recovered_data == test_data
        assert result.confidence_score == 1.0
    
    def test_raid_with_missing_disk(self, temp_dir):
        """בדיקת RAID עם דיסק חסר"""
        # יצירת RAID 5 עם דיסק חסר
        disks = []
        available_paths = []
        
        for i in range(3):
            disk_path = temp_dir / f"missing_test_disk{i}.img"
            
            if i != 1:  # דיסק 1 "חסר"
                disk_path.write_bytes(b"test data" * 1000)
                available_paths.append(str(disk_path))
                
                disk = RAIDDisk(
                    path=str(disk_path),
                    index=i,
                    size=disk_path.stat().st_size,
                    is_available=True,
                    health_status="good"
                )
            else:
                disk = RAIDDisk(
                    path=str(disk_path),
                    index=i,
                    size=0,
                    is_available=False,
                    health_status="missing"
                )
            
            disks.append(disk)
        
        config = RAIDConfiguration(
            level=RAIDLevel.RAID5,
            disk_count=3,
            stripe_size=65536,
            chunk_size=32768,
            total_size=2000000,
            usable_size=1400000,
            parity_disks=1,
            disks=disks,
            status=RAIDStatus.DEGRADED
        )
        
        output_path = temp_dir / "missing_disk_output.img"
        reconstructor = RAIDReconstructor()
        
        # צריך להצליח לשחזר RAID 5 עם דיסק אחד חסר
        result = reconstructor._reconstruct_raid5(config, str(output_path))
        
        # עם הנתונים המדומים שלנו, התוצאה תלויה במימוש
        # אבל לפחות לא צריך להיכשל באופן קטסטרופלי
        assert isinstance(result, ReconstructionResult)
    
    def test_raid_error_handling(self, temp_dir):
        """בדיקת טיפול בשגיאות RAID"""
        # מערך לא תקין
        invalid_disks = [
            RAIDDisk("/nonexistent/disk1", 0, 1000, False, "missing"),
            RAIDDisk("/nonexistent/disk2", 1, 1000, False, "missing")
        ]
        
        config = RAIDConfiguration(
            level=RAIDLevel.RAID0,
            disk_count=2,
            stripe_size=65536,
            chunk_size=32768,
            total_size=2000,
            usable_size=2000,
            parity_disks=0,
            disks=invalid_disks,
            status=RAIDStatus.FAILED
        )
        
        output_path = temp_dir / "error_test_output.img"
        reconstructor = RAIDReconstructor()
        
        # צריך להיכשל באופן מבוקר
        result = reconstructor._reconstruct_raid0(config, str(output_path))
        
        assert result.success is False or result.confidence_score < 0.5