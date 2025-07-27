"""
PhoenixDRS - File Carver Module
מודול חיתוך קבצים מתקדם על בסיס חתימות
"""

import json
import os
import re
from typing import Dict, List, Optional, Tuple, Generator
from dataclasses import dataclass
from pathlib import Path
import mmap

"""
PhoenixDRS - File Carver Module (GUI Enabled)
מודול חיתוך קבצים מתקדם, מותאם לעבודה עם ממשק גרפי.
"""

import json
import os
import re
from typing import Dict, List, Optional, Tuple, Generator, Callable, Any
from dataclasses import dataclass
from pathlib import Path
import mmap

# --- Data Classes ---
@dataclass
class FileSignature:
    """הגדרת חתימת קובץ"""
    name: str
    extension: str
    header: str  # hex string
    footer: Optional[str] = None  # hex string
    max_size: int = 0
    footer_search_offset: int = 0
    description: str = ""
    
    def header_bytes(self) -> bytes:
        return bytes.fromhex(self.header.replace(" ", ""))
    
    def footer_bytes(self) -> Optional[bytes]:
        if self.footer:
            return bytes.fromhex(self.footer.replace(" ", ""))
        return None

@dataclass
class CarvedFile:
    """קובץ שנחתך"""
    signature_name: str
    extension: str
    start_offset: int
    end_offset: int
    size: int
    output_path: str
    is_complete: bool

# --- Core Classes ---
class SignatureDatabase:
    """טוען ומנהל את מסד נתוני החתימות מקובץ JSON."""
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.signatures: List[FileSignature] = []
        self.load_signatures()
    
    def load_signatures(self):
        try:
            with open(self.db_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            for sig_data in data.get('signatures', []):
                self.signatures.append(FileSignature(**sig_data))
        except (IOError, json.JSONDecodeError) as e:
            print(f"Error loading signature database: {e}")
            self.signatures = []

class FileCarver:
    """מנוע חיתוך קבצים ראשי, מותאם ל-GUI."""
    
    def __init__(self, chunk_size: int = 1024 * 1024): # 1MB chunks
        self.chunk_size = chunk_size
        self.log_callback: Optional[Callable[[str], None]] = None

    def _log(self, message: str):
        if self.log_callback:
            self.log_callback(message)
        else:
            print(message)

    def carve(
        self, 
        image_path: str, 
        output_dir: str,
        selected_signatures: List[FileSignature],
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        file_found_callback: Optional[Callable[[CarvedFile], None]] = None,
        log_callback: Optional[Callable[[str], None]] = None
    ):
        """
        חיתוך קבצים מתמונת דיסק עם תמיכה ב-callbacks.
        """
        self.log_callback = log_callback
        self._log(f"Starting file carving operation on {image_path}")

        if not selected_signatures:
            self._log("Error: No file signatures selected. Aborting.")
            return

        os.makedirs(output_dir, exist_ok=True)
        self._log(f"Output directory: {output_dir}")

        headers_map = {sig.header_bytes(): sig for sig in selected_signatures}
        header_patterns = list(headers_map.keys())
        max_header_len = max(len(h) for h in header_patterns) if header_patterns else 0

        try:
            with open(image_path, 'rb') as f:
                with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm_file:
                    file_size = len(mm_file)
                    self._log(f"Image size: {file_size / (1024*1024):.2f} MB")

                    total_chunks = (file_size + self.chunk_size - 1) // self.chunk_size
                    found_files_count = 0

                    for i in range(total_chunks):
                        chunk_start = i * self.chunk_size
                        chunk_end = min(chunk_start + self.chunk_size + max_header_len, file_size)
                        chunk_data = mm_file[chunk_start:chunk_end]

                        if progress_callback:
                            progress_callback({
                                'progress': (i + 1) * 100 / total_chunks,
                                'status': f"Scanning chunk {i+1}/{total_chunks}..."
                            })

                        # Simple find for each pattern in the chunk
                        for header, sig in headers_map.items():
                            current_pos = 0
                            while True:
                                header_pos = chunk_data.find(header, current_pos)
                                if header_pos == -1:
                                    break
                                
                                abs_start_offset = chunk_start + header_pos
                                carved_file = self._carve_single_file(
                                    mm_file, abs_start_offset, sig, output_dir, found_files_count
                                )
                                if carved_file:
                                    found_files_count += 1
                                    if file_found_callback:
                                        file_found_callback(carved_file)
                                
                                current_pos = header_pos + 1

            self._log(f"File carving completed. Found {found_files_count} files.")

        except Exception as e:
            self._log(f"An error occurred during carving: {e}")

    def _carve_single_file(
        self, mm_file: mmap.mmap, start_offset: int, 
        signature: FileSignature, output_dir: str, file_index: int
    ) -> Optional[CarvedFile]:
        try:
            file_size = len(mm_file)
            end_offset = -1
            is_complete = False

            footer_bytes = signature.footer_bytes()
            if footer_bytes:
                search_start = start_offset + signature.footer_search_offset
                search_end = min(start_offset + signature.max_size, file_size)
                footer_pos = mm_file.find(footer_bytes, search_start, search_end)
                if footer_pos != -1:
                    end_offset = footer_pos + len(footer_bytes)
                    is_complete = True
            
            if end_offset == -1:
                end_offset = min(start_offset + signature.max_size, file_size)
                is_complete = (footer_bytes is None) # Complete only if no footer was expected

            carved_size = end_offset - start_offset
            if carved_size <= 0:
                return None

            filename = f"{file_index:06d}_{signature.name.replace(' ', '_')}{signature.extension}"
            output_path = os.path.join(output_dir, filename)

            with open(output_path, 'wb') as out_f:
                out_f.write(mm_file[start_offset:end_offset])

            return CarvedFile(
                signature_name=signature.name,
                extension=signature.extension,
                start_offset=start_offset,
                end_offset=end_offset,
                size=carved_size,
                output_path=output_path,
                is_complete=is_complete
            )

        except Exception as e:
            self._log(f"Error carving file at offset {start_offset}: {e}")
            return None
