"""
PhoenixDRS - Video Rebuilder Module
מודול שחזור וידאו מתקדם לקבצי Canon MOV/MP4 מפוצלים
"""

import os
import struct
from typing import Dict, List, Optional, Tuple, NamedTuple
from pathlib import Path
import mmap


class AtomInfo(NamedTuple):
    """מידע על אטום QuickTime/MP4"""
    atom_type: str
    size: int
    offset: int
    data: bytes


class MOOVMetadata(NamedTuple):
    """מטא-דטה מאטום MOOV"""
    expected_mdat_size: int
    duration: Optional[int] = None
    track_count: Optional[int] = None
    creation_time: Optional[int] = None


class VideoRebuilder:
    """מנוע שחזור וידאו ראשי"""
    
    def __init__(self):
        self.carved_atoms: List[AtomInfo] = []
        self.moov_atoms: List[AtomInfo] = []
        self.mdat_atoms: List[AtomInfo] = []
    
    def rebuild_canon_mov(self, image_path: str, output_dir: str) -> List[str]:
        """
        שחזור קבצי Canon MOV מתמונת דיסק
        
        Args:
            image_path: נתיב לתמונת הדיסק
            output_dir: תיקיית פלט
            
        Returns:
            רשימת קבצי וידאו שנבנו מחדש
        """
        print(f"מתחיל שחזור וידאו Canon MOV: {image_path}")
        
        # יצירת תיקיית פלט
        os.makedirs(output_dir, exist_ok=True)
        atoms_dir = os.path.join(output_dir, "atoms")
        os.makedirs(atoms_dir, exist_ok=True)
        
        # שלב 1: חיתוך אטומים
        print("שלב 1: חיתוך אטומי MOOV ו-MDAT...")
        self.carve_qt_atoms(image_path, atoms_dir)
        
        # שלב 2: ניתוח אטומי MOOV
        print("שלב 2: ניתוח אטומי MOOV...")
        moov_metadata = self._analyze_moov_atoms(atoms_dir)
        
        # שלב 3: התאמה ובנייה מחדש
        print("שלב 3: התאמה ובנייה מחדש...")
        rebuilt_videos = self._match_and_rebuild(atoms_dir, output_dir, moov_metadata)
        
        print(f"שחזור הושלם! נבנו {len(rebuilt_videos)} קבצי וידאו")
        return rebuilt_videos
    
    def carve_qt_atoms(self, image_path: str, atoms_dir: str):
        """חיתוך אטומי QuickTime/MP4"""
        print("מחפש אטומי MOOV ו-MDAT...")
        
        # חתימות אטומים
        moov_signature = b'moov'
        mdat_signature = b'mdat'
        
        with open(image_path, 'rb') as f:
            with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mmapped_file:
                file_size = len(mmapped_file)
                
                # חיפוש אטומים
                self._search_atoms(mmapped_file, moov_signature, 'moov', atoms_dir)
                self._search_atoms(mmapped_file, mdat_signature, 'mdat', atoms_dir)
        
        print(f"נמצאו {len(self.moov_atoms)} אטומי MOOV ו-{len(self.mdat_atoms)} אטומי MDAT")
    
    def _search_atoms(self, mmapped_file: mmap.mmap, signature: bytes, 
                     atom_type: str, atoms_dir: str):
        """חיפוש אטומים ספציפיים"""
        file_size = len(mmapped_file)
        search_pos = 0
        atom_index = 0
        
        while search_pos < file_size - 8:
            # חיפוש חתימת האטום
            pos = mmapped_file.find(signature, search_pos)
            if pos == -1:
                break
            
            # בדיקה שזה באמת תחילת אטום (4 בתים לפני החתימה = גודל)
            if pos >= 4:
                atom_start = pos - 4
                try:
                    # קריאת גודל האטום
                    size_bytes = mmapped_file[atom_start:atom_start + 4]
                    atom_size = struct.unpack('>I', size_bytes)[0]
                    
                    # בדיקת תקינות הגודל
                    if 8 <= atom_size <= file_size - atom_start:
                        # חילוץ האטום
                        atom_data = mmapped_file[atom_start:atom_start + atom_size]
                        
                        # שמירת האטום לקובץ
                        filename = f"{atom_type}_{atom_index:06d}_offset_{atom_start:08X}_size_{atom_size}.atom"
                        atom_path = os.path.join(atoms_dir, filename)
                        
                        with open(atom_path, 'wb') as atom_file:
                            atom_file.write(atom_data)
                        
                        # יצירת מידע על האטום
                        atom_info = AtomInfo(
                            atom_type=atom_type,
                            size=atom_size,
                            offset=atom_start,
                            data=atom_data
                        )
                        
                        if atom_type == 'moov':
                            self.moov_atoms.append(atom_info)
                        elif atom_type == 'mdat':
                            self.mdat_atoms.append(atom_info)
                        
                        atom_index += 1
                        search_pos = atom_start + atom_size
                    else:
                        search_pos = pos + 1
                        
                except (struct.error, IndexError):
                    search_pos = pos + 1
            else:
                search_pos = pos + 1
    
    def parse_moov_for_mdat_size(self, moov_data: bytes) -> Optional[MOOVMetadata]:
        """
        ניתוח אטום MOOV לקבלת גודל MDAT הצפוי
        """
        try:
            # ניתוח בסיסי של מבנה האטום
            if len(moov_data) < 8:
                return None
            
            # קריאת גודל האטום וסוג
            atom_size = struct.unpack('>I', moov_data[0:4])[0]
            atom_type = moov_data[4:8].decode('ascii', errors='ignore')
            
            if atom_type != 'moov':
                return None
            
            # חיפוש אטומי משנה חשובים
            expected_mdat_size = self._calculate_mdat_size_from_moov(moov_data)
            duration = self._extract_duration_from_moov(moov_data)
            track_count = self._count_tracks_in_moov(moov_data)
            creation_time = self._extract_creation_time_from_moov(moov_data)
            
            return MOOVMetadata(
                expected_mdat_size=expected_mdat_size,
                duration=duration,
                track_count=track_count,
                creation_time=creation_time
            )
            
        except Exception as e:
            print(f"שגיאה בניתוח MOOV: {e}")
            return None
    
    def _calculate_mdat_size_from_moov(self, moov_data: bytes) -> int:
        """חישוב גודל MDAT הצפוי מתוך MOOV"""
        total_size = 0
        
        # חיפוש אטום STSZ (Sample Size)
        stsz_pos = moov_data.find(b'stsz')
        if stsz_pos != -1 and stsz_pos >= 4:
            try:
                # קריאת מידע מאטום STSZ
                stsz_start = stsz_pos - 4
                stsz_size = struct.unpack('>I', moov_data[stsz_start:stsz_start + 4])[0]
                
                if stsz_size >= 20:  # גודל מינימלי לאטום STSZ
                    # דילוג על header (8 bytes) + version/flags (4 bytes)
                    data_start = stsz_start + 12
                    
                    # קריאת גודל דגימה ומספר דגימות
                    sample_size = struct.unpack('>I', moov_data[data_start:data_start + 4])[0]
                    sample_count = struct.unpack('>I', moov_data[data_start + 4:data_start + 8])[0]
                    
                    if sample_size == 0:
                        # גדלים משתנים - קריאת טבלת גדלים
                        table_start = data_start + 8
                        for i in range(min(sample_count, (stsz_size - 20) // 4)):
                            if table_start + 4 <= len(moov_data):
                                size = struct.unpack('>I', moov_data[table_start:table_start + 4])[0]
                                total_size += size
                                table_start += 4
                    else:
                        # גודל קבוע לכל הדגימות
                        total_size = sample_size * sample_count
                        
            except (struct.error, IndexError):
                pass
        
        # אם לא נמצא STSZ, ניסיון חיפוש אטומי STCO/CO64 (Chunk Offset)
        if total_size == 0:
            total_size = self._estimate_size_from_chunks(moov_data)
        
        # הוספת מרווח בטיחות של 10%
        return int(total_size * 1.1) if total_size > 0 else 1024 * 1024  # 1MB default
    
    def _estimate_size_from_chunks(self, moov_data: bytes) -> int:
        """הערכת גודל על בסיס chunk offsets"""
        # זוהי הערכה פשוטה - בפועל צריך ניתוח מורכב יותר
        return 10 * 1024 * 1024  # 10MB הערכה בסיסית
    
    def _extract_duration_from_moov(self, moov_data: bytes) -> Optional[int]:
        """חילוץ משך הוידאו מאטום MOOV"""
        mvhd_pos = moov_data.find(b'mvhd')
        if mvhd_pos != -1 and mvhd_pos >= 4:
            try:
                mvhd_start = mvhd_pos - 4
                # דילוג על header + version/flags
                data_start = mvhd_start + 12
                if data_start + 16 <= len(moov_data):
                    # קריאת duration (bytes 12-15 באטום mvhd)
                    duration = struct.unpack('>I', moov_data[data_start + 12:data_start + 16])[0]
                    return duration
            except (struct.error, IndexError):
                pass
        return None
    
    def _count_tracks_in_moov(self, moov_data: bytes) -> Optional[int]:
        """ספירת מספר tracks באטום MOOV"""
        track_count = moov_data.count(b'trak')
        return track_count if track_count > 0 else None
    
    def _extract_creation_time_from_moov(self, moov_data: bytes) -> Optional[int]:
        """חילוץ זמן יצירה מאטום MOOV"""
        mvhd_pos = moov_data.find(b'mvhd')
        if mvhd_pos != -1 and mvhd_pos >= 4:
            try:
                mvhd_start = mvhd_pos - 4
                data_start = mvhd_start + 12
                if data_start + 8 <= len(moov_data):
                    creation_time = struct.unpack('>I', moov_data[data_start + 4:data_start + 8])[0]
                    return creation_time
            except (struct.error, IndexError):
                pass
        return None
    
    def _analyze_moov_atoms(self, atoms_dir: str) -> Dict[str, MOOVMetadata]:
        """ניתוח כל אטומי MOOV"""
        moov_metadata = {}
        
        for moov_file in Path(atoms_dir).glob("moov_*.atom"):
            try:
                with open(moov_file, 'rb') as f:
                    moov_data = f.read()
                
                metadata = self.parse_moov_for_mdat_size(moov_data)
                if metadata:
                    moov_metadata[moov_file.name] = metadata
                    print(f"MOOV {moov_file.name}: צפוי MDAT בגודל {metadata.expected_mdat_size:,} bytes")
                    
            except Exception as e:
                print(f"שגיאה בניתוח {moov_file}: {e}")
        
        return moov_metadata
    
    def _match_and_rebuild(self, atoms_dir: str, output_dir: str, 
                          moov_metadata: Dict[str, MOOVMetadata]) -> List[str]:
        """התאמה ובנייה מחדש של קבצי וידאו"""
        rebuilt_videos = []
        
        # רשימת קבצי MDAT
        mdat_files = list(Path(atoms_dir).glob("mdat_*.atom"))
        
        for moov_file_name, metadata in moov_metadata.items():
            moov_path = os.path.join(atoms_dir, moov_file_name)
            expected_size = metadata.expected_mdat_size
            
            # חיפוש MDAT מתאים
            best_match = None
            best_size_diff = float('inf')
            
            for mdat_file in mdat_files:
                mdat_size = mdat_file.stat().st_size - 8  # מינוס header של האטום
                size_diff = abs(mdat_size - expected_size)
                
                # התאמה עם סובלנות של 20%
                if size_diff < expected_size * 0.2 and size_diff < best_size_diff:
                    best_match = mdat_file
                    best_size_diff = size_diff
            
            if best_match:
                # בנייה מחדש של קובץ הוידאו
                video_filename = f"rebuilt_{moov_file_name.replace('.atom', '')}.mov"
                video_path = os.path.join(output_dir, video_filename)
                
                success = self._build_mov_file(moov_path, str(best_match), video_path)
                if success:
                    rebuilt_videos.append(video_path)
                    print(f"נבנה בהצלחה: {video_filename}")
                    
                    # הסרת MDAT שנוצל מהרשימה
                    mdat_files.remove(best_match)
                else:
                    print(f"כישלון בבנייה: {video_filename}")
            else:
                print(f"לא נמצא MDAT מתאים עבור {moov_file_name}")
        
        return rebuilt_videos
    
    def _build_mov_file(self, moov_path: str, mdat_path: str, output_path: str) -> bool:
        """בנייה מחדש של קובץ MOV"""
        try:
            with open(output_path, 'wb') as output_file:
                # כתיבת FTYP atom (File Type)
                ftyp_atom = self._create_ftyp_atom()
                output_file.write(ftyp_atom)
                
                # כתיבת MOOV atom
                with open(moov_path, 'rb') as moov_file:
                    moov_data = moov_file.read()
                    output_file.write(moov_data)
                
                # כתיבת MDAT atom
                with open(mdat_path, 'rb') as mdat_file:
                    mdat_data = mdat_file.read()
                    output_file.write(mdat_data)
            
            return True
            
        except Exception as e:
            print(f"שגיאה בבנייה מחדש: {e}")
            return False
    
    def _create_ftyp_atom(self) -> bytes:
        """יצירת אטום FTYP סטנדרטי לקבצי Canon MOV"""
        # FTYP atom for Canon MOV
        ftyp_data = b'qt  '  # major brand
        ftyp_data += struct.pack('>I', 0)  # minor version
        ftyp_data += b'qt  '  # compatible brands
        
        # אטום שלם עם header
        atom_size = 8 + len(ftyp_data)
        ftyp_atom = struct.pack('>I', atom_size) + b'ftyp' + ftyp_data
        
        return ftyp_atom


if __name__ == "__main__":
    # דוגמה לשימוש
    rebuilder = VideoRebuilder()
    # rebuilt_videos = rebuilder.rebuild_canon_mov("disk_image.dd", "rebuilt_videos")
    print("מודול שחזור הוידאו מוכן לשימוש")