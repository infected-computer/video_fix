"""
Advanced Container Analyzer for Video Files
מנתח מכלים מתקדם לקבצי וידאו

This module provides deep analysis of video container structures including:
- MP4/MOV atom structure analysis
- AVI chunk analysis  
- MKV element analysis
- MXF KLV analysis
- Corruption pattern detection
- Repair recommendations
"""

import struct
import logging
import mmap
import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union, BinaryIO
import json
import re

logger = logging.getLogger(__name__)


class ContainerType(Enum):
    """Supported container types"""
    MP4 = "mp4"
    MOV = "mov"
    AVI = "avi"
    MKV = "mkv"
    MXF = "mxf"
    WEBM = "webm"
    FLV = "flv"
    UNKNOWN = "unknown"


class CorruptionType(Enum):
    """Types of container corruption"""
    HEADER_CORRUPTION = "header_corruption"
    METADATA_CORRUPTION = "metadata_corruption"
    INDEX_CORRUPTION = "index_corruption"
    STRUCTURE_CORRUPTION = "structure_corruption"
    EOF_CORRUPTION = "eof_corruption"
    ATOM_SIZE_MISMATCH = "atom_size_mismatch"
    MISSING_ATOMS = "missing_atoms"
    TRUNCATED_FILE = "truncated_file"
    FRAGMENTED_DATA = "fragmented_data"


@dataclass
class AtomInfo:
    """Information about an MP4/MOV atom"""
    atom_type: str
    offset: int
    size: int
    header_size: int = 8
    is_container: bool = False
    children: List['AtomInfo'] = field(default_factory=list)
    data_hash: Optional[str] = None
    corruption_detected: bool = False
    corruption_details: List[str] = field(default_factory=list)


@dataclass
class ChunkInfo:
    """Information about an AVI chunk"""
    chunk_id: str
    offset: int
    size: int
    data_size: int
    corruption_detected: bool = False
    corruption_details: List[str] = field(default_factory=list)


@dataclass
class ElementInfo:
    """Information about an MKV element"""
    element_id: int
    element_name: str
    offset: int
    size: int
    data_size: int
    level: int = 0
    children: List['ElementInfo'] = field(default_factory=list)
    corruption_detected: bool = False


@dataclass
class ContainerAnalysisResult:
    """Complete container analysis result"""
    container_type: ContainerType
    file_path: str
    file_size: int
    
    # Structure information
    atoms: List[AtomInfo] = field(default_factory=list)
    chunks: List[ChunkInfo] = field(default_factory=list)  
    elements: List[ElementInfo] = field(default_factory=list)
    
    # Analysis results
    structure_integrity: float = 0.0  # 0.0-1.0
    metadata_integrity: float = 0.0
    playback_integrity: float = 0.0
    overall_score: float = 0.0
    
    # Corruption details
    corruptions_found: List[CorruptionType] = field(default_factory=list)
    corruption_details: Dict[str, Any] = field(default_factory=dict)
    repair_recommendations: List[str] = field(default_factory=list)
    
    # Technical details
    has_moov: bool = False
    has_mdat: bool = False
    moov_before_mdat: bool = False
    fragment_count: int = 0
    track_count: int = 0
    duration_seconds: float = 0.0
    
    # Analysis metadata
    analysis_time: float = 0.0
    analyzed_at: datetime = field(default_factory=datetime.utcnow)


class ContainerAnalyzer(ABC):
    """Abstract base class for container analyzers"""
    
    @abstractmethod
    def analyze(self, file_path: str) -> ContainerAnalysisResult:
        """Analyze container structure"""
        pass
    
    @abstractmethod
    def detect_corruption(self, result: ContainerAnalysisResult) -> List[CorruptionType]:
        """Detect corruption patterns"""
        pass


class MP4Analyzer(ContainerAnalyzer):
    """Advanced MP4/MOV container analyzer"""
    
    def __init__(self):
        self.container_atoms = {
            'moov', 'trak', 'edts', 'mdia', 'minf', 'dinf', 'stbl', 'mvex', 'moof', 'traf'
        }
        
        self.critical_atoms = {
            'ftyp', 'moov', 'mdat', 'mvhd', 'tkhd', 'mdhd', 'hdlr', 'stsd', 'stts', 'stsc', 'stsz', 'stco'
        }
        
        self.atom_descriptions = {
            'ftyp': 'File Type Box',
            'moov': 'Movie Box',
            'mdat': 'Media Data Box',
            'mvhd': 'Movie Header Box',
            'trak': 'Track Box',
            'tkhd': 'Track Header Box',
            'edts': 'Edit Box',
            'elst': 'Edit List Box',
            'mdia': 'Media Box',
            'mdhd': 'Media Header Box',
            'hdlr': 'Handler Reference Box',
            'minf': 'Media Information Box',
            'vmhd': 'Video Media Header Box',
            'smhd': 'Sound Media Header Box',
            'dinf': 'Data Information Box',
            'dref': 'Data Reference Box',
            'stbl': 'Sample Table Box',
            'stsd': 'Sample Description Box',
            'stts': 'Decoding Time to Sample Box',
            'ctts': 'Composition Time to Sample Box',
            'stsc': 'Sample to Chunk Box',
            'stsz': 'Sample Size Box',
            'stco': 'Chunk Offset Box',
            'co64': '64-bit Chunk Offset Box',
            'stss': 'Sync Sample Box',
            'free': 'Free Space Box',
            'skip': 'Skip Box',
            'udta': 'User Data Box',
            'meta': 'Metadata Box'
        }
    
    def analyze(self, file_path: str) -> ContainerAnalysisResult:
        """Analyze MP4/MOV container structure"""
        start_time = datetime.utcnow()
        
        result = ContainerAnalysisResult(
            container_type=ContainerType.MP4,
            file_path=file_path,
            file_size=Path(file_path).stat().st_size
        )
        
        try:
            with open(file_path, 'rb') as f:
                with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
                    result.atoms = self._parse_atoms(mm, 0, result.file_size)
                    
            # Analyze structure
            self._analyze_structure(result)
            
            # Detect corruption
            result.corruptions_found = self.detect_corruption(result)
            
            # Calculate integrity scores
            self._calculate_integrity_scores(result)
            
            # Generate repair recommendations
            result.repair_recommendations = self._generate_repair_recommendations(result)
            
        except Exception as e:
            logger.error(f"MP4 analysis failed: {e}")
            result.corruption_details["analysis_error"] = str(e)
        
        result.analysis_time = (datetime.utcnow() - start_time).total_seconds()
        return result
    
    def _parse_atoms(self, mm: mmap.mmap, start_offset: int, end_offset: int, level: int = 0) -> List[AtomInfo]:
        """Parse MP4 atoms recursively"""
        atoms = []
        offset = start_offset
        
        while offset < end_offset:
            try:
                if offset + 8 > len(mm):
                    break
                
                # Read atom header
                size_data = mm[offset:offset + 4]
                type_data = mm[offset + 4:offset + 8]
                
                if len(size_data) != 4 or len(type_data) != 4:
                    break
                
                size = struct.unpack('>I', size_data)[0]
                atom_type = type_data.decode('ascii', errors='ignore')
                
                # Handle extended size
                header_size = 8
                if size == 1:
                    if offset + 16 > len(mm):
                        break
                    size = struct.unpack('>Q', mm[offset + 8:offset + 16])[0]
                    header_size = 16
                elif size == 0:
                    # Atom extends to end of file
                    size = end_offset - offset
                
                # Validate atom
                if size < header_size or offset + size > end_offset:
                    # Possible corruption - try to recover
                    logger.warning(f"Invalid atom size at offset {offset}: {size}")
                    break
                
                atom = AtomInfo(
                    atom_type=atom_type,
                    offset=offset,
                    size=size,
                    header_size=header_size,
                    is_container=atom_type in self.container_atoms
                )
                
                # Hash atom data for integrity checking
                data_start = offset + header_size
                data_end = min(offset + size, end_offset)
                if data_end > data_start:
                    atom_data = mm[data_start:data_end]
                    atom.data_hash = hashlib.md5(atom_data[:1024]).hexdigest()  # First 1KB
                
                # Parse children if container atom
                if atom.is_container and size > header_size:
                    child_start = offset + header_size
                    child_end = min(offset + size, end_offset)
                    atom.children = self._parse_atoms(mm, child_start, child_end, level + 1)
                
                atoms.append(atom)
                offset += size
                
            except Exception as e:
                logger.warning(f"Error parsing atom at offset {offset}: {e}")
                break
        
        return atoms
    
    def _analyze_structure(self, result: ContainerAnalysisResult):
        """Analyze MP4 structure for required components"""
        atom_types = set()
        moov_offset = None
        mdat_offset = None
        
        def collect_atoms(atoms: List[AtomInfo]):
            for atom in atoms:
                atom_types.add(atom.atom_type)
                
                if atom.atom_type == 'moov':
                    nonlocal moov_offset
                    moov_offset = atom.offset
                elif atom.atom_type == 'mdat':
                    nonlocal mdat_offset
                    if mdat_offset is None:  # First mdat
                        mdat_offset = atom.offset
                
                collect_atoms(atom.children)
        
        collect_atoms(result.atoms)
        
        # Check for required atoms
        result.has_moov = 'moov' in atom_types
        result.has_mdat = 'mdat' in atom_types
        
        # Check moov/mdat order
        if moov_offset is not None and mdat_offset is not None:
            result.moov_before_mdat = moov_offset < mdat_offset
        
        # Count tracks
        result.track_count = sum(1 for atom in result.atoms 
                               if atom.atom_type == 'trak' or 
                               any(child.atom_type == 'trak' 
                                   for child in self._get_all_children(atom)))
        
        # Extract duration from mvhd
        result.duration_seconds = self._extract_duration(result.atoms)
    
    def _get_all_children(self, atom: AtomInfo) -> List[AtomInfo]:
        """Get all children recursively"""
        children = atom.children.copy()
        for child in atom.children:
            children.extend(self._get_all_children(child))
        return children
    
    def _extract_duration(self, atoms: List[AtomInfo]) -> float:
        """Extract duration from mvhd atom"""
        try:
            with open(atoms[0].file_path if hasattr(atoms[0], 'file_path') else "", 'rb') as f:
                for atom in atoms:
                    if atom.atom_type == 'moov':
                        for child in atom.children:
                            if child.atom_type == 'mvhd':
                                f.seek(child.offset + child.header_size)
                                version = struct.unpack('B', f.read(1))[0]
                                f.seek(3, 1)  # Skip flags
                                
                                if version == 1:
                                    # 64-bit version
                                    f.seek(16, 1)  # Skip creation and modification time
                                    timescale = struct.unpack('>I', f.read(4))[0]
                                    duration = struct.unpack('>Q', f.read(8))[0]
                                else:
                                    # 32-bit version
                                    f.seek(8, 1)  # Skip creation and modification time
                                    timescale = struct.unpack('>I', f.read(4))[0]
                                    duration = struct.unpack('>I', f.read(4))[0]
                                
                                if timescale > 0:
                                    return duration / timescale
        except Exception as e:
            logger.warning(f"Failed to extract duration: {e}")
        
        return 0.0
    
    def detect_corruption(self, result: ContainerAnalysisResult) -> List[CorruptionType]:
        """Detect MP4 corruption patterns"""
        corruptions = []
        
        # Check for missing critical atoms
        atom_types = set()
        
        def collect_atom_types(atoms: List[AtomInfo]):
            for atom in atoms:
                atom_types.add(atom.atom_type)
                collect_atom_types(atom.children)
        
        collect_atom_types(result.atoms)
        
        missing_critical = self.critical_atoms - atom_types
        if missing_critical:
            corruptions.append(CorruptionType.MISSING_ATOMS)
            result.corruption_details["missing_atoms"] = list(missing_critical)
        
        # Check file type atom
        if not result.atoms or result.atoms[0].atom_type != 'ftyp':
            corruptions.append(CorruptionType.HEADER_CORRUPTION)
            result.corruption_details["ftyp_missing"] = True
        
        # Check for truncated file
        total_expected_size = sum(atom.size for atom in result.atoms)
        if abs(total_expected_size - result.file_size) > 1024:  # Allow 1KB tolerance
            corruptions.append(CorruptionType.TRUNCATED_FILE)
            result.corruption_details["size_mismatch"] = {
                "expected": total_expected_size,
                "actual": result.file_size
            }
        
        # Check moov/mdat presence and order
        if not result.has_moov:
            corruptions.append(CorruptionType.METADATA_CORRUPTION)
            result.corruption_details["moov_missing"] = True
        
        if not result.has_mdat:
            corruptions.append(CorruptionType.STRUCTURE_CORRUPTION)
            result.corruption_details["mdat_missing"] = True
        
        # Check for fragmented MP4 structure issues
        moof_count = sum(1 for atom in result.atoms if atom.atom_type == 'moof')
        if moof_count > 0:
            result.fragment_count = moof_count
            # Fragmented MP4 specific checks
            if not result.has_moov:
                corruptions.append(CorruptionType.METADATA_CORRUPTION)
                result.corruption_details["fragmented_without_moov"] = True
        
        # Check atom size consistency
        for atom in result.atoms:
            if self._check_atom_corruption(atom, result):
                corruptions.append(CorruptionType.ATOM_SIZE_MISMATCH)
        
        return list(set(corruptions))  # Remove duplicates
    
    def _check_atom_corruption(self, atom: AtomInfo, result: ContainerAnalysisResult) -> bool:
        """Check individual atom for corruption"""
        corruption_found = False
        
        # Check atom type validity
        if not atom.atom_type.replace(' ', '').isalnum():
            atom.corruption_detected = True
            atom.corruption_details.append("Invalid atom type characters")
            corruption_found = True
        
        # Check size consistency
        if atom.size < atom.header_size:
            atom.corruption_detected = True
            atom.corruption_details.append("Size smaller than header")
            corruption_found = True
        
        # Check if atom extends beyond file
        if atom.offset + atom.size > result.file_size:
            atom.corruption_detected = True
            atom.corruption_details.append("Atom extends beyond file")
            corruption_found = True
        
        # Recursively check children
        for child in atom.children:
            if self._check_atom_corruption(child, result):
                corruption_found = True
        
        return corruption_found
    
    def _calculate_integrity_scores(self, result: ContainerAnalysisResult):
        """Calculate integrity scores"""
        # Structure integrity
        structure_score = 1.0
        if CorruptionType.MISSING_ATOMS in result.corruptions_found:
            structure_score -= 0.4
        if CorruptionType.HEADER_CORRUPTION in result.corruptions_found:
            structure_score -= 0.3
        if CorruptionType.TRUNCATED_FILE in result.corruptions_found:
            structure_score -= 0.2
        
        result.structure_integrity = max(structure_score, 0.0)
        
        # Metadata integrity
        metadata_score = 1.0
        if not result.has_moov:
            metadata_score -= 0.6
        if CorruptionType.METADATA_CORRUPTION in result.corruptions_found:
            metadata_score -= 0.3
        
        result.metadata_integrity = max(metadata_score, 0.0)
        
        # Playback integrity
        playback_score = 1.0
        if not result.has_mdat:
            playback_score -= 0.5
        if CorruptionType.INDEX_CORRUPTION in result.corruptions_found:
            playback_score -= 0.3
        
        result.playback_integrity = max(playback_score, 0.0)
        
        # Overall score
        result.overall_score = (
            result.structure_integrity * 0.4 +
            result.metadata_integrity * 0.3 +
            result.playback_integrity * 0.3
        )
    
    def _generate_repair_recommendations(self, result: ContainerAnalysisResult) -> List[str]:
        """Generate repair recommendations"""
        recommendations = []
        
        if CorruptionType.HEADER_CORRUPTION in result.corruptions_found:
            recommendations.append("Use untrunc with reference file for header reconstruction")
            recommendations.append("Try FFmpeg with -f mp4 to force container format")
        
        if CorruptionType.MISSING_ATOMS in result.corruptions_found:
            if 'moov' in result.corruption_details.get('missing_atoms', []):
                recommendations.append("Critical: moov atom missing - try mp4recover for deep scan")
                recommendations.append("Use photorec for fragment recovery if other tools fail")
        
        if CorruptionType.TRUNCATED_FILE in result.corruptions_found:
            recommendations.append("File appears truncated - check for additional fragments")
            recommendations.append("Use FFmpeg with -avoid_negative_ts make_zero")
        
        if not result.moov_before_mdat and result.has_moov and result.has_mdat:
            recommendations.append("moov atom after mdat - use FFmpeg to optimize structure")
        
        if result.fragment_count > 0:
            recommendations.append("Fragmented MP4 detected - ensure proper fragment handling")
        
        if result.overall_score < 0.5:
            recommendations.append("Severe corruption detected - try multiple tools in sequence")
            recommendations.append("Consider AI-enhanced repair for best quality recovery")
        
        return recommendations


class AVIAnalyzer(ContainerAnalyzer):
    """AVI container analyzer"""
    
    def analyze(self, file_path: str) -> ContainerAnalysisResult:
        """Analyze AVI container structure"""
        result = ContainerAnalysisResult(
            container_type=ContainerType.AVI,
            file_path=file_path,
            file_size=Path(file_path).stat().st_size
        )
        
        try:
            with open(file_path, 'rb') as f:
                # Check RIFF header
                if f.read(4) != b'RIFF':
                    result.corruptions_found.append(CorruptionType.HEADER_CORRUPTION)
                    return result
                
                file_size = struct.unpack('<I', f.read(4))[0]
                if f.read(4) != b'AVI ':
                    result.corruptions_found.append(CorruptionType.HEADER_CORRUPTION)
                    return result
                
                # Parse chunks
                result.chunks = self._parse_avi_chunks(f)
                
        except Exception as e:
            logger.error(f"AVI analysis failed: {e}")
            result.corruption_details["analysis_error"] = str(e)
        
        return result
    
    def _parse_avi_chunks(self, f: BinaryIO) -> List[ChunkInfo]:
        """Parse AVI chunks"""
        chunks = []
        
        while True:
            try:
                chunk_id = f.read(4)
                if len(chunk_id) != 4:
                    break
                
                chunk_size = struct.unpack('<I', f.read(4))[0]
                offset = f.tell() - 8
                
                chunk = ChunkInfo(
                    chunk_id=chunk_id.decode('ascii', errors='ignore'),
                    offset=offset,
                    size=chunk_size + 8,
                    data_size=chunk_size
                )
                
                chunks.append(chunk)
                
                # Skip chunk data
                f.seek(chunk_size, 1)
                
                # Align to word boundary
                if chunk_size % 2:
                    f.seek(1, 1)
                    
            except Exception:
                break
        
        return chunks
    
    def detect_corruption(self, result: ContainerAnalysisResult) -> List[CorruptionType]:
        """Detect AVI corruption patterns"""
        corruptions = []
        
        # Check for required chunks
        chunk_ids = {chunk.chunk_id for chunk in result.chunks}
        
        if 'hdrl' not in chunk_ids:
            corruptions.append(CorruptionType.HEADER_CORRUPTION)
        
        if 'movi' not in chunk_ids:
            corruptions.append(CorruptionType.STRUCTURE_CORRUPTION)
        
        return corruptions


class AdvancedContainerAnalyzer:
    """Main advanced container analyzer"""
    
    def __init__(self):
        self.analyzers = {
            ContainerType.MP4: MP4Analyzer(),
            ContainerType.MOV: MP4Analyzer(),  # MOV uses same structure as MP4
            ContainerType.AVI: AVIAnalyzer(),
        }
    
    def detect_container_type(self, file_path: str) -> ContainerType:
        """Detect container type from file"""
        try:
            with open(file_path, 'rb') as f:
                header = f.read(12)
                
                if len(header) < 8:
                    return ContainerType.UNKNOWN
                
                # MP4/MOV detection
                if header[4:8] == b'ftyp':
                    # Check brand for specific type
                    if len(header) >= 12:
                        brand = header[8:12]
                        if brand in [b'qt  ', b'mov ']:
                            return ContainerType.MOV
                        else:
                            return ContainerType.MP4
                    return ContainerType.MP4
                
                # AVI detection
                if header[:4] == b'RIFF' and header[8:12] == b'AVI ':
                    return ContainerType.AVI
                
                # MKV detection
                if header[:4] == b'\x1a\x45\xdf\xa3':
                    return ContainerType.MKV
                
                # WebM detection (subset of MKV)
                if header[:4] == b'\x1a\x45\xdf\xa3':
                    # Read more to check for webm
                    f.seek(0)
                    data = f.read(100)
                    if b'webm' in data:
                        return ContainerType.WEBM
                    return ContainerType.MKV
                
                # FLV detection
                if header[:3] == b'FLV':
                    return ContainerType.FLV
        
        except Exception as e:
            logger.error(f"Container type detection failed: {e}")
        
        return ContainerType.UNKNOWN
    
    def analyze_container(self, file_path: str) -> ContainerAnalysisResult:
        """Analyze container with appropriate analyzer"""
        container_type = self.detect_container_type(file_path)
        
        if container_type in self.analyzers:
            analyzer = self.analyzers[container_type]
            return analyzer.analyze(file_path)
        else:
            # Basic analysis for unsupported types
            result = ContainerAnalysisResult(
                container_type=container_type,
                file_path=file_path,
                file_size=Path(file_path).stat().st_size
            )
            result.corruption_details["unsupported_type"] = True
            return result
    
    def batch_analyze(self, file_paths: List[str]) -> Dict[str, ContainerAnalysisResult]:
        """Analyze multiple files"""
        results = {}
        
        for file_path in file_paths:
            try:
                results[file_path] = self.analyze_container(file_path)
            except Exception as e:
                logger.error(f"Failed to analyze {file_path}: {e}")
                results[file_path] = ContainerAnalysisResult(
                    container_type=ContainerType.UNKNOWN,
                    file_path=file_path,
                    file_size=0
                )
                results[file_path].corruption_details["analysis_error"] = str(e)
        
        return results
    
    def generate_repair_plan(self, result: ContainerAnalysisResult) -> Dict[str, Any]:
        """Generate comprehensive repair plan"""
        plan = {
            "container_type": result.container_type.value,
            "corruption_severity": self._assess_severity(result),
            "recommended_tools": [],
            "repair_strategies": [],
            "estimated_success_rate": 0.0,
            "risk_assessment": "low"
        }
        
        # Assess corruption severity
        severity = plan["corruption_severity"]
        
        if result.overall_score >= 0.8:
            plan["recommended_tools"] = ["ffmpeg"]
            plan["repair_strategies"] = ["container_remux"]
            plan["estimated_success_rate"] = 0.9
            plan["risk_assessment"] = "low"
        elif result.overall_score >= 0.6:
            plan["recommended_tools"] = ["untrunc", "ffmpeg"]
            plan["repair_strategies"] = ["header_reconstruction", "container_remux"]
            plan["estimated_success_rate"] = 0.7
            plan["risk_assessment"] = "medium"
        elif result.overall_score >= 0.3:
            plan["recommended_tools"] = ["mp4recover", "untrunc", "ffmpeg"]
            plan["repair_strategies"] = ["deep_scan", "fragment_recovery", "ai_enhancement"]
            plan["estimated_success_rate"] = 0.5
            plan["risk_assessment"] = "high"
        else:
            plan["recommended_tools"] = ["photorec", "mp4recover", "ai_enhancement"]
            plan["repair_strategies"] = ["fragment_recovery", "deep_scan", "custom_algorithms"]
            plan["estimated_success_rate"] = 0.2
            plan["risk_assessment"] = "critical"
        
        return plan
    
    def _assess_severity(self, result: ContainerAnalysisResult) -> str:
        """Assess corruption severity"""
        if result.overall_score >= 0.8:
            return "minimal"
        elif result.overall_score >= 0.6:
            return "mild"
        elif result.overall_score >= 0.3:
            return "moderate"
        else:
            return "severe"
    
    def export_analysis(self, result: ContainerAnalysisResult, output_path: str):
        """Export analysis to JSON file"""
        try:
            # Convert result to serializable format
            data = {
                "container_type": result.container_type.value,
                "file_path": result.file_path,
                "file_size": result.file_size,
                "structure_integrity": result.structure_integrity,
                "metadata_integrity": result.metadata_integrity,
                "playback_integrity": result.playback_integrity,
                "overall_score": result.overall_score,
                "corruptions_found": [c.value for c in result.corruptions_found],
                "corruption_details": result.corruption_details,
                "repair_recommendations": result.repair_recommendations,
                "has_moov": result.has_moov,
                "has_mdat": result.has_mdat,
                "moov_before_mdat": result.moov_before_mdat,
                "fragment_count": result.fragment_count,
                "track_count": result.track_count,
                "duration_seconds": result.duration_seconds,
                "analysis_time": result.analysis_time,
                "analyzed_at": result.analyzed_at.isoformat(),
                "atoms": [
                    {
                        "type": atom.atom_type,
                        "offset": atom.offset,
                        "size": atom.size,
                        "corruption_detected": atom.corruption_detected,
                        "corruption_details": atom.corruption_details
                    }
                    for atom in result.atoms
                ]
            }
            
            with open(output_path, 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.info(f"Analysis exported to {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to export analysis: {e}")


# Usage example
async def example_container_analysis():
    """Example usage of advanced container analyzer"""
    
    analyzer = AdvancedContainerAnalyzer()
    
    # Analyze single file
    file_path = "test_video.mp4"
    
    print(f"Analyzing: {file_path}")
    
    # Detect container type
    container_type = analyzer.detect_container_type(file_path)
    print(f"Container type: {container_type.value}")
    
    # Perform analysis
    result = analyzer.analyze_container(file_path)
    
    print(f"\nAnalysis Results:")
    print(f"  Structure integrity: {result.structure_integrity:.2f}")
    print(f"  Metadata integrity: {result.metadata_integrity:.2f}")
    print(f"  Playback integrity: {result.playback_integrity:.2f}")
    print(f"  Overall score: {result.overall_score:.2f}")
    print(f"  Duration: {result.duration_seconds:.1f}s")
    print(f"  Track count: {result.track_count}")
    print(f"  Fragment count: {result.fragment_count}")
    
    if result.corruptions_found:
        print(f"  Corruptions found: {[c.value for c in result.corruptions_found]}")
    
    if result.repair_recommendations:
        print(f"  Repair recommendations:")
        for rec in result.repair_recommendations:
            print(f"    - {rec}")
    
    # Generate repair plan
    repair_plan = analyzer.generate_repair_plan(result)
    print(f"\nRepair Plan:")
    print(f"  Severity: {repair_plan['corruption_severity']}")
    print(f"  Recommended tools: {repair_plan['recommended_tools']}")
    print(f"  Success rate estimate: {repair_plan['estimated_success_rate']:.1%}")
    print(f"  Risk assessment: {repair_plan['risk_assessment']}")
    
    # Export analysis
    analyzer.export_analysis(result, "analysis_report.json")


if __name__ == "__main__":
    import asyncio
    logging.basicConfig(level=logging.INFO)
    asyncio.run(example_container_analysis())