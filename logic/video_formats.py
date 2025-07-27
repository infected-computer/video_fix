"""
PhoenixDRS - Video Format Support
תמיכה מקיפה בפורמטי וידאו
"""

import os
from typing import Dict, List, Tuple, Optional

class VideoFormatManager:
    """מנהל פורמטי וידאו מקיף"""
    
    def __init__(self):
        self.formats = self._initialize_formats()
        self.codec_signatures = self._initialize_codec_signatures()
        self.professional_formats = self._initialize_professional_formats()
    
    def _initialize_formats(self) -> Dict[str, Dict]:
        """אתחול פורמטי וידאו"""
        return {
            # Container formats - MP4 family
            '.mp4': {
                'name': 'MPEG-4 Part 14',
                'container': 'MP4',
                'common_codecs': ['H.264', 'H.265', 'AV1', 'VP9'],
                'audio_codecs': ['AAC', 'MP3', 'AC3'],
                'signature': b'ftyp',
                'signature_offset': 4,
                'description': 'Most common video format',
                'max_resolution': '8K',
                'professional': True
            },
            '.m4v': {
                'name': 'iTunes Video',
                'container': 'MP4',
                'common_codecs': ['H.264', 'H.265'],
                'audio_codecs': ['AAC'],
                'signature': b'ftyp',
                'signature_offset': 4,
                'description': 'Apple iTunes video format',
                'max_resolution': '8K',
                'professional': False
            },
            '.3gp': {
                'name': '3GPP',
                'container': '3GPP',
                'common_codecs': ['H.263', 'H.264', 'MPEG-4'],
                'audio_codecs': ['AMR', 'AAC'],
                'signature': b'ftyp3gp',
                'signature_offset': 4,
                'description': 'Mobile video format',
                'max_resolution': '720p',
                'professional': False
            },
            '.3g2': {
                'name': '3GPP2',
                'container': '3GPP2',
                'common_codecs': ['H.263', 'H.264'],
                'audio_codecs': ['AMR', 'AAC'],
                'signature': b'ftyp3g2',
                'signature_offset': 4,
                'description': 'CDMA mobile video format',
                'max_resolution': '720p',
                'professional': False
            },
            
            # Apple formats
            '.mov': {
                'name': 'QuickTime Movie',
                'container': 'QuickTime',
                'common_codecs': ['H.264', 'H.265', 'ProRes', 'DNxHD'],
                'audio_codecs': ['AAC', 'PCM', 'MP3'],
                'signature': b'ftyp',
                'signature_offset': 4,
                'description': 'Apple QuickTime format',
                'max_resolution': '8K',
                'professional': True
            },
            '.qt': {
                'name': 'QuickTime',
                'container': 'QuickTime',
                'common_codecs': ['H.264', 'ProRes'],
                'audio_codecs': ['AAC', 'PCM'],
                'signature': b'moov',
                'signature_offset': 4,
                'description': 'Legacy QuickTime format',
                'max_resolution': '4K',
                'professional': True
            },
            
            # Microsoft formats
            '.avi': {
                'name': 'Audio Video Interleave',
                'container': 'AVI',
                'common_codecs': ['DivX', 'Xvid', 'H.264', 'MJPEG'],
                'audio_codecs': ['MP3', 'AC3', 'PCM'],
                'signature': b'RIFF',
                'signature_offset': 0,
                'description': 'Microsoft AVI format',
                'max_resolution': '4K',
                'professional': False
            },
            '.wmv': {
                'name': 'Windows Media Video',
                'container': 'ASF',
                'common_codecs': ['WMV1', 'WMV2', 'WMV3', 'VC-1'],
                'audio_codecs': ['WMA', 'MP3'],
                'signature': b'\x30\x26\xb2\x75\x8e\x66\xcf\x11',
                'signature_offset': 0,
                'description': 'Microsoft Windows Media format',
                'max_resolution': '1080p',
                'professional': False
            },
            '.asf': {
                'name': 'Advanced Systems Format',
                'container': 'ASF',
                'common_codecs': ['WMV', 'VC-1'],
                'audio_codecs': ['WMA'],
                'signature': b'\x30\x26\xb2\x75\x8e\x66\xcf\x11',
                'signature_offset': 0,
                'description': 'Microsoft streaming format',
                'max_resolution': '1080p',
                'professional': False
            },
            
            # Matroska family
            '.mkv': {
                'name': 'Matroska Video',
                'container': 'Matroska',
                'common_codecs': ['H.264', 'H.265', 'VP9', 'AV1', 'Xvid'],
                'audio_codecs': ['AAC', 'MP3', 'FLAC', 'DTS', 'AC3'],
                'signature': b'\x1a\x45\xdf\xa3',
                'signature_offset': 0,
                'description': 'Open source container format',
                'max_resolution': '8K',
                'professional': True
            },
            '.webm': {
                'name': 'WebM',
                'container': 'WebM',
                'common_codecs': ['VP8', 'VP9', 'AV1'],
                'audio_codecs': ['Vorbis', 'Opus'],
                'signature': b'\x1a\x45\xdf\xa3',
                'signature_offset': 0,
                'description': 'Google web video format',
                'max_resolution': '8K',
                'professional': False
            },
            
            # Adobe formats
            '.flv': {
                'name': 'Flash Video',
                'container': 'FLV',
                'common_codecs': ['H.263', 'H.264', 'VP6'],
                'audio_codecs': ['MP3', 'AAC', 'Speex'],
                'signature': b'FLV',
                'signature_offset': 0,
                'description': 'Adobe Flash video format',
                'max_resolution': '1080p',
                'professional': False
            },
            '.f4v': {
                'name': 'Flash MP4',
                'container': 'MP4',
                'common_codecs': ['H.264'],
                'audio_codecs': ['AAC'],
                'signature': b'ftyp',
                'signature_offset': 4,
                'description': 'Adobe Flash MP4 format',
                'max_resolution': '1080p',
                'professional': False
            },
            
            # Professional broadcast formats
            '.mxf': {
                'name': 'Material Exchange Format',
                'container': 'MXF',
                'common_codecs': ['DNxHD', 'DNxHR', 'ProRes', 'AVC-Intra'],
                'audio_codecs': ['PCM', 'AAC'],
                'signature': b'\x06\x0e\x2b\x34\x02\x05\x01\x01',
                'signature_offset': 0,
                'description': 'Professional broadcast format',
                'max_resolution': '8K',
                'professional': True
            },
            '.gxf': {
                'name': 'General eXchange Format',
                'container': 'GXF',
                'common_codecs': ['DV25', 'DV50', 'MPEG-2'],
                'audio_codecs': ['PCM'],
                'signature': b'\x00\x00\x00\x01\xbc',
                'signature_offset': 0,
                'description': 'Professional video format',
                'max_resolution': '1080p',
                'professional': True
            },
            
            # MPEG formats
            '.mpg': {
                'name': 'MPEG-1/2',
                'container': 'MPEG-PS',
                'common_codecs': ['MPEG-1', 'MPEG-2'],
                'audio_codecs': ['MP2', 'MP3', 'AC3'],
                'signature': b'\x00\x00\x01\xb3',
                'signature_offset': 0,
                'description': 'MPEG video format',
                'max_resolution': '1080i',
                'professional': True
            },
            '.mpeg': {
                'name': 'MPEG Video',
                'container': 'MPEG-PS',
                'common_codecs': ['MPEG-1', 'MPEG-2'],
                'audio_codecs': ['MP2', 'AC3'],
                'signature': b'\x00\x00\x01\xb3',
                'signature_offset': 0,
                'description': 'MPEG video format',
                'max_resolution': '1080i',
                'professional': True
            },
            '.m2v': {
                'name': 'MPEG-2 Video',
                'container': 'MPEG-ES',
                'common_codecs': ['MPEG-2'],
                'audio_codecs': [],
                'signature': b'\x00\x00\x01\xb3',
                'signature_offset': 0,
                'description': 'MPEG-2 video elementary stream',
                'max_resolution': '1080i',
                'professional': True
            },
            '.ts': {
                'name': 'MPEG Transport Stream',
                'container': 'MPEG-TS',
                'common_codecs': ['H.264', 'H.265', 'MPEG-2'],
                'audio_codecs': ['AAC', 'MP3', 'AC3'],
                'signature': b'\x47',
                'signature_offset': 0,
                'description': 'Broadcasting transport stream',
                'max_resolution': '8K',
                'professional': True
            },
            '.m2ts': {
                'name': 'Blu-ray Transport Stream',
                'container': 'MPEG-TS',
                'common_codecs': ['H.264', 'H.265', 'VC-1'],
                'audio_codecs': ['DTS', 'TrueHD', 'AC3'],
                'signature': b'\x47',
                'signature_offset': 0,
                'description': 'Blu-ray disc format',
                'max_resolution': '4K',
                'professional': True
            },
            '.mts': {
                'name': 'AVCHD Video',
                'container': 'MPEG-TS',
                'common_codecs': ['H.264'],
                'audio_codecs': ['AC3', 'DTS'],
                'signature': b'\x47',
                'signature_offset': 0,
                'description': 'AVCHD camcorder format',
                'max_resolution': '1080p',
                'professional': False
            },
            
            # Raw and uncompressed formats
            '.dv': {
                'name': 'Digital Video',
                'container': 'DV',
                'common_codecs': ['DV'],
                'audio_codecs': ['PCM'],
                'signature': b'\x1f\x07\x00\x3f',
                'signature_offset': 0,
                'description': 'Digital Video format',
                'max_resolution': '720p',
                'professional': True
            },
            '.hdv': {
                'name': 'High Definition Video',
                'container': 'MPEG-TS',
                'common_codecs': ['MPEG-2'],
                'audio_codecs': ['MPEG-1 Layer II'],
                'signature': b'\x47',
                'signature_offset': 0,
                'description': 'High Definition DV format',
                'max_resolution': '1080i',
                'professional': True
            },
            
            # Modern streaming formats
            '.y4m': {
                'name': 'YUV4MPEG2',
                'container': 'Y4M',
                'common_codecs': ['Raw YUV'],
                'audio_codecs': [],
                'signature': b'YUV4MPEG2',
                'signature_offset': 0,
                'description': 'Uncompressed YUV format',
                'max_resolution': '8K',
                'professional': True
            },
            '.yuv': {
                'name': 'Raw YUV',
                'container': 'Raw',
                'common_codecs': ['Raw YUV'],
                'audio_codecs': [],
                'signature': None,
                'signature_offset': 0,
                'description': 'Raw YUV video data',
                'max_resolution': '8K',
                'professional': True
            },
            
            # Camera-specific formats
            '.r3d': {
                'name': 'RED Raw',
                'container': 'R3D',
                'common_codecs': ['REDCODE'],
                'audio_codecs': ['PCM'],
                'signature': b'RED1',
                'signature_offset': 0,
                'description': 'RED Digital Cinema format',
                'max_resolution': '8K',
                'professional': True
            },
            '.braw': {
                'name': 'Blackmagic RAW',
                'container': 'BRAW',
                'common_codecs': ['BRAW'],
                'audio_codecs': ['PCM'],
                'signature': b'BRAW',
                'signature_offset': 0,
                'description': 'Blackmagic raw format',
                'max_resolution': '8K',
                'professional': True
            },
            '.arw': {
                'name': 'Sony RAW',
                'container': 'ARW',
                'common_codecs': ['Sony Raw'],
                'audio_codecs': [],
                'signature': b'II*\x00',
                'signature_offset': 0,
                'description': 'Sony Alpha raw format',
                'max_resolution': '8K',
                'professional': True
            },
            
            # VR and 360 formats
            '.vr': {
                'name': 'Virtual Reality Video',
                'container': 'MP4',
                'common_codecs': ['H.264', 'H.265'],
                'audio_codecs': ['AAC', 'Spatial Audio'],
                'signature': b'ftyp',
                'signature_offset': 4,
                'description': '360-degree VR video',
                'max_resolution': '8K',
                'professional': True
            },
            
            # Lossless formats
            '.ffv1': {
                'name': 'FFV1',
                'container': 'MKV',
                'common_codecs': ['FFV1'],
                'audio_codecs': ['FLAC', 'PCM'],
                'signature': b'\x1a\x45\xdf\xa3',
                'signature_offset': 0,
                'description': 'Lossless video codec',
                'max_resolution': '8K',
                'professional': True
            },
            
            # Legacy formats
            '.rm': {
                'name': 'RealMedia',
                'container': 'RM',
                'common_codecs': ['RV30', 'RV40'],
                'audio_codecs': ['RealAudio'],
                'signature': b'.RMF',
                'signature_offset': 0,
                'description': 'RealNetworks format',
                'max_resolution': '720p',
                'professional': False
            },
            '.rmvb': {
                'name': 'RealMedia Variable Bitrate',
                'container': 'RMVB',
                'common_codecs': ['RV30', 'RV40'],
                'audio_codecs': ['RealAudio'],
                'signature': b'.RMF',
                'signature_offset': 0,
                'description': 'Variable bitrate RealMedia',
                'max_resolution': '720p',
                'professional': False
            },
            '.ogv': {
                'name': 'Ogg Video',
                'container': 'Ogg',
                'common_codecs': ['Theora', 'VP8'],
                'audio_codecs': ['Vorbis'],
                'signature': b'OggS',
                'signature_offset': 0,
                'description': 'Open source Ogg format',
                'max_resolution': '1080p',
                'professional': False
            }
        }
    
    def _initialize_codec_signatures(self) -> Dict[str, bytes]:
        """חתימות של קודקים נפוצים"""
        return {
            'H.264': b'\x00\x00\x00\x01\x67',
            'H.265': b'\x00\x00\x00\x01\x40',
            'VP9': b'\x83\x42\x83',
            'AV1': b'\x81\x8c\x00',
            'MPEG-4': b'\x00\x00\x01\xb0',
            'DivX': b'DivX',
            'Xvid': b'Xvid',
            'ProRes': b'icpf',
            'DNxHD': b'AVdn'
        }
    
    def _initialize_professional_formats(self) -> List[str]:
        """פורמטים מקצועיים"""
        return [
            '.mov', '.mxf', '.r3d', '.braw', '.mkv', '.mp4', 
            '.m2ts', '.ts', '.dv', '.hdv', '.y4m', '.yuv',
            '.ffv1', '.gxf', '.mpg', '.mpeg', '.m2v'
        ]
    
    def get_format_info(self, file_path: str) -> Optional[Dict]:
        """קבלת מידע על פורמט הקובץ"""
        if not os.path.exists(file_path):
            return None
        
        file_ext = os.path.splitext(file_path)[1].lower()
        
        # חיפוש בפורמטים הידועים
        if file_ext in self.formats:
            format_info = self.formats[file_ext].copy()
            format_info['extension'] = file_ext
            format_info['file_size'] = os.path.getsize(file_path)
            
            # ניסיון זיהוי עמוק יותר
            detected_codec = self.detect_codec(file_path)
            if detected_codec:
                format_info['detected_codec'] = detected_codec
            
            return format_info
        
        # ניסיון זיהוי לפי תוכן הקובץ
        return self.detect_format_by_content(file_path)
    
    def detect_codec(self, file_path: str) -> Optional[str]:
        """זיהוי קודק לפי תוכן הקובץ"""
        try:
            with open(file_path, 'rb') as f:
                header = f.read(1024)  # קריאת 1KB ראשונים
                
                for codec_name, signature in self.codec_signatures.items():
                    if signature in header:
                        return codec_name
                        
        except Exception:
            pass
        
        return None
    
    def detect_format_by_content(self, file_path: str) -> Optional[Dict]:
        """זיהוי פורמט לפי תוכן הקובץ"""
        try:
            with open(file_path, 'rb') as f:
                header = f.read(32)
                
                for ext, format_info in self.formats.items():
                    signature = format_info.get('signature')
                    if signature:
                        offset = format_info.get('signature_offset', 0)
                        if len(header) > offset + len(signature):
                            if header[offset:offset + len(signature)] == signature:
                                detected_info = format_info.copy()
                                detected_info['extension'] = ext
                                detected_info['file_size'] = os.path.getsize(file_path)
                                detected_info['detected_by_content'] = True
                                return detected_info
                                
        except Exception:
            pass
        
        return None
    
    def get_supported_extensions(self) -> List[str]:
        """קבלת רשימת כל הסיומות הנתמכות"""
        return sorted(list(self.formats.keys()))
    
    def get_professional_formats(self) -> List[str]:
        """קבלת רשימת פורמטים מקצועיים"""
        return self.professional_formats
    
    def is_professional_format(self, file_path: str) -> bool:
        """בדיקה האם הפורמט מקצועי"""
        ext = os.path.splitext(file_path)[1].lower()
        return ext in self.professional_formats
    
    def get_format_statistics(self) -> Dict:
        """סטטיסטיקות על הפורמטים הנתמכים"""
        total_formats = len(self.formats)
        professional_count = len(self.professional_formats)
        
        containers = set()
        codecs = set()
        
        for format_info in self.formats.values():
            containers.add(format_info['container'])
            codecs.update(format_info['common_codecs'])
        
        return {
            'total_formats': total_formats,
            'professional_formats': professional_count,
            'consumer_formats': total_formats - professional_count,
            'unique_containers': len(containers),
            'unique_codecs': len(codecs),
            'containers': sorted(list(containers)),
            'codecs': sorted(list(codecs))
        }
    
    def validate_file_format(self, file_path: str) -> Tuple[bool, str]:
        """אימות פורמט הקובץ"""
        if not os.path.exists(file_path):
            return False, "File does not exist"
        
        if os.path.getsize(file_path) == 0:
            return False, "File is empty"
        
        format_info = self.get_format_info(file_path)
        if not format_info:
            return False, "Unsupported or unrecognized format"
        
        # בדיקת שלמות בסיסית
        try:
            with open(file_path, 'rb') as f:
                signature = format_info.get('signature')
                if signature:
                    offset = format_info.get('signature_offset', 0)
                    f.seek(offset)
                    file_signature = f.read(len(signature))
                    
                    if file_signature != signature:
                        return False, "File signature mismatch - possibly corrupted"
                        
        except Exception as e:
            return False, f"Error reading file: {str(e)}"
        
        return True, "Valid format"
    
    def get_repair_priority(self, file_path: str) -> int:
        """קבלת עדיפות תיקון (1-10, גבוה יותר = עדיפות גבוהה יותר)"""
        format_info = self.get_format_info(file_path)
        if not format_info:
            return 1
        
        # פורמטים מקצועיים מקבלים עדיפות גבוהה
        if format_info.get('professional', False):
            return 8
        
        # פורמטים נפוצים
        ext = format_info.get('extension', '')
        if ext in ['.mp4', '.mov', '.mkv']:
            return 7
        elif ext in ['.avi', '.wmv', '.flv']:
            return 5
        else:
            return 3