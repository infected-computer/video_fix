
"""
PhoenixDRS - Disk Imaging Module
מודול הדמיה מתקדם לשחזור מידע מקצועי, מותאם לשימוש עם GUI.
"""

import os
import time
import struct
import hashlib
from typing import Dict, List, Optional, Tuple, Callable, Any
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ImageMetadata:
    """מטא-דטה של תמונת דיסק"""
    source_device: str
    destination_file: str
    sector_size: int
    total_sectors: int
    created_timestamp: float
    md5_hash: Optional[str] = None
    sha256_hash: Optional[str] = None
    bad_sectors: List[int] = None
    
    def __post_init__(self):
        if self.bad_sectors is None:
            self.bad_sectors = []


class BadSectorMap:
    """מפת סקטורים פגומים"""
    
    def __init__(self, map_file_path: str):
        self.map_file_path = map_file_path
        self.bad_sectors: Dict[int, int] = {}  # sector_number: retry_count
        
    def add_bad_sector(self, sector_num: int, retry_count: int = 1):
        """הוספת סקטור פגום למפה"""
        self.bad_sectors[sector_num] = retry_count
        self._save_to_file()
        
    def _save_to_file(self):
        """שמירת מפת הסקטורים הפגומים לקובץ"""
        with open(self.map_file_path, 'w') as f:
            f.write("# PhoenixDRS Bad Sector Map\n")
            f.write("# Sector_Number,Retry_Count\n")
            for sector, retries in self.bad_sectors.items():
                f.write(f"{sector},{retries}\n")


class ProgressTracker:
    """מעקב התקדמות הדמיה עם תמיכה ב-Callback"""
    
    def __init__(self, total_sectors: int, update_callback: Optional[Callable[[Dict[str, Any]], None]] = None):
        self.total_sectors = total_sectors
        self.processed_sectors = 0
        self.start_time = time.time()
        self.last_update_time = 0
        self.update_callback = update_callback
        
    def update(self, sectors_processed: int):
        """עדכון התקדמות"""
        self.processed_sectors += sectors_processed
        current_time = time.time()
        
        # עדכון כל שנייה או אם הסתיים
        if current_time - self.last_update_time >= 1.0 or self.processed_sectors == self.total_sectors:
            elapsed = current_time - self.start_time
            # Handle division by zero if total_sectors is 0
            percentage = (self.processed_sectors / self.total_sectors) * 100 if self.total_sectors > 0 else 100
            # MB/s
            speed = (self.processed_sectors * 512 / 1024 / 1024) / elapsed if elapsed > 0 else 0
            
            status_payload = {
                "progress": percentage,
                "processed_sectors": self.processed_sectors,
                "total_sectors": self.total_sectors,
                "speed_mbs": speed,
                "elapsed_time": elapsed
            }

            if self.update_callback:
                self.update_callback(status_payload)
            else:
                self._print_progress(status_payload)
            
            self.last_update_time = current_time
            
    def _print_progress(self, status: dict):
        """הדפסת התקדמות לקונסול"""
        print(f"\rProgress: {status['progress']:.1f}% | "
              f"Sectors: {status['processed_sectors']:,}/{status['total_sectors']:,} | "
              f"Speed: {status['speed_mbs']:.2f} MB/s", end='', flush=True)


class DiskImager:
    """מנוע הדמיה ראשי"""
    
    def __init__(self, max_retries: int = 3, retry_delay: float = 0.1):
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.log_callback: Optional[Callable[[str], None]] = None
        
    def _log(self, message: str):
        if self.log_callback:
            self.log_callback(message)
        else:
            print(message)

    def create_image(self, source_device: str, destination_file: str, 
                     sector_size: int = 512, 
                     progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
                     log_callback: Optional[Callable[[str], None]] = None) -> Optional[ImageMetadata]:
        """
        יצירת תמונת דיסק sector-by-sector עם תמיכה ב-callbacks.
        
        Args:
            source_device: התקן המקור
            destination_file: קובץ היעד
            sector_size: גודל סקטור
            progress_callback: פונקציה לקבלת עדכוני התקדמות
            log_callback: פונקציה לרישום הודעות
            
        Returns:
            ImageMetadata או None אם נכשל
        """
        self.log_callback = log_callback
        self._log(f"Starting imaging: {source_device} -> {destination_file}")
        
        bad_sector_map = BadSectorMap(f"{destination_file}.bad_sectors")
        
        try:
            total_size = self._get_device_size(source_device)
            if total_size == 0:
                self._log("Error: Source device size is 0 or could not be determined.")
                return None
            total_sectors = total_size // sector_size
        except Exception as e:
            self._log(f"Error getting device size: {e}")
            return None

        self._log(f"Device size: {total_sectors:,} sectors ({sector_size} bytes each)")
        
        progress = ProgressTracker(total_sectors, progress_callback)
        
        md5_hash = hashlib.md5()
        sha256_hash = hashlib.sha256()
        
        try:
            with open(source_device, 'rb') as source, \
                 open(destination_file, 'wb') as dest:
                
                for sector_num in range(total_sectors):
                    sector_data = self._read_sector_with_retry(
                        source, sector_num, sector_size, bad_sector_map
                    )
                    
                    dest.write(sector_data)
                    
                    md5_hash.update(sector_data)
                    sha256_hash.update(sector_data)
                    
                    progress.update(1)
            
            self._log("\nImaging completed successfully!")
            
            metadata = ImageMetadata(
                source_device=source_device,
                destination_file=destination_file,
                sector_size=sector_size,
                total_sectors=total_sectors,
                created_timestamp=time.time(),
                md5_hash=md5_hash.hexdigest(),
                sha256_hash=sha256_hash.hexdigest(),
                bad_sectors=list(bad_sector_map.bad_sectors.keys())
            )
            
            self._save_metadata(metadata, f"{destination_file}.metadata")
            
            return metadata
        except (IOError, OSError) as e:
            self._log(f"\nFATAL ERROR during imaging: {e}")
            return None

    
    def _get_device_size(self, device_path: str) -> int:
        """קבלת גודל התקן בבתים"""
        # The fcntl method is Linux-specific and will fail on Windows.
        # The os.path.getsize method is more portable for regular files.
        if os.path.isfile(device_path):
            return os.path.getsize(device_path)
        
        # For block devices on Windows, this is more complex.
        # A simple seek/tell is a fallback for file-based images.
        try:
            with open(device_path, 'rb') as f:
                f.seek(0, 2)
                return f.tell()
        except Exception as e:
            self._log(f"Could not determine size of {device_path}: {e}")
            raise

    def _read_sector_with_retry(self, source_file, sector_num: int, 
                              sector_size: int, bad_sector_map: BadSectorMap) -> bytes:
        """קריאת סקטור עם ניסיונות חוזרים"""
        offset = sector_num * sector_size
        
        for attempt in range(self.max_retries):
            try:
                source_file.seek(offset)
                data = source_file.read(sector_size)
                
                if len(data) == sector_size:
                    return data
                else:
                    # Partial sector at end of file - pad with zeros
                    return data.ljust(sector_size, b'\x00')
                    
            except IOError as e:
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                    continue
                else:
                    bad_sector_map.add_bad_sector(sector_num, attempt + 1)
                    self._log(f"\nError reading sector {sector_num}: {e}. Marking as bad.")
                    return b'\x00' * sector_size
        
        return b'\x00' * sector_size
    
    def _save_metadata(self, metadata: ImageMetadata, metadata_file: str):
        """שמירת מטא-דטה לקובץ"""
        self._log(f"Saving metadata to {metadata_file}")
        with open(metadata_file, 'w') as f:
            f.write("# PhoenixDRS Image Metadata\n")
            f.write(f"source_device={metadata.source_device}\n")
            f.write(f"destination_file={metadata.destination_file}\n")
            f.write(f"sector_size={metadata.sector_size}\n")
            f.write(f"total_sectors={metadata.total_sectors}\n")
            f.write(f"created_timestamp={metadata.created_timestamp}\n")
            f.write(f"md5_hash={metadata.md5_hash}\n")
            f.write(f"sha256_hash={metadata.sha256_hash}\n")
            f.write(f"bad_sectors_count={len(metadata.bad_sectors)}\n")


if __name__ == "__main__":
    # Example usage
    imager = DiskImager()
    
    # Create a dummy file to image
    DUMMY_FILE_NAME = "dummy_source_file.bin"
    DUMMY_FILE_SIZE = 10 * 1024 * 1024 # 10 MB
    with open(DUMMY_FILE_NAME, 'wb') as f:
        f.write(os.urandom(DUMMY_FILE_SIZE))

    print(f"Created a dummy file '{DUMMY_FILE_NAME}' for imaging demo.")
    
    def console_progress(status: dict):
        print(f"\rProgress: {status['progress']:.1f}% | "
              f"Speed: {status['speed_mbs']:.2f} MB/s", end='', flush=True)

    def console_log(message: str):
        # Clear the progress line before printing a log message
        print(f"\r{' ' * 80}\r{message}")

    metadata = imager.create_image(
        DUMMY_FILE_NAME, 
        "dummy_image.img",
        progress_callback=console_progress,
        log_callback=console_log
    )
    
    if metadata:
        print("\n--- Imaging Complete ---")
        print(f"  MD5: {metadata.md5_hash}")
        print(f"  SHA256: {metadata.sha256_hash}")
        print(f"  Bad Sectors: {len(metadata.bad_sectors)}")
        print("------------------------")
    else:
        print("\n--- Imaging Failed ---")

    # Clean up dummy file
    os.remove(DUMMY_FILE_NAME)
