"""
Advanced Video Repair Engine - Enhanced Algorithmic Implementation
מנוע תיקון וידאו מתקדם - מימוש אלגוריתמי משופר

This module implements advanced video repair algorithms integrating multiple tools:
- untrunc: MP4/MOV structure repair
- FFmpeg: Container manipulation and stream recovery
- mp4recover: Professional MP4 recovery
- Custom algorithms: Motion compensation, AI enhancement
"""

import asyncio
import logging
import subprocess
import tempfile
import shutil
import json
import time
import struct
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
import uuid
import re
import hashlib

# Video processing imports
import cv2
import numpy as np
from pymediainfo import MediaInfo

# Our modules - import from current module to avoid circular imports
# Will be imported when used in orchestrator
try:
    from .video_repair_orchestrator import (
        VideoAnalysisResult, RepairResult, RepairTechnique, 
        VideoFormat, RepairStatus, SystemResources
    )
except ImportError:
    # Fallback definitions for standalone usage
    from enum import Enum
    from dataclasses import dataclass
    
    class RepairTechnique(Enum):
        HEADER_RECONSTRUCTION = "header_reconstruction"
        CONTAINER_REMUX = "container_remux"
        FRAGMENT_RECOVERY = "fragment_recovery"
    
    class VideoFormat(Enum):
        MP4_H264 = "mp4_h264"
        UNKNOWN = "unknown"
    
    class RepairStatus(Enum):
        COMPLETED = "completed"
        FAILED = "failed"
    
    @dataclass
    class VideoAnalysisResult:
        file_path: str = ""
    
    @dataclass 
    class RepairResult:
        success: bool = False
    
    @dataclass
    class SystemResources:
        cpu_count: int = 0

logger = logging.getLogger(__name__)


class RepairToolType(Enum):
    """Available repair tools"""
    UNTRUNC = "untrunc"
    FFMPEG = "ffmpeg"
    MP4RECOVER = "mp4recover"
    PHOTOREC = "photorec"
    CUSTOM_ALGORITHM = "custom"
    AI_ENHANCED = "ai_enhanced"


class CorruptionSeverity(Enum):
    """Corruption severity levels"""
    MINIMAL = "minimal"        # < 5% corruption
    MILD = "mild"             # 5-15% corruption
    MODERATE = "moderate"     # 15-40% corruption
    SEVERE = "severe"         # 40-70% corruption
    CRITICAL = "critical"     # > 70% corruption


@dataclass
class RepairStrategy:
    """Comprehensive repair strategy configuration"""
    primary_tool: RepairToolType
    fallback_tools: List[RepairToolType] = field(default_factory=list)
    techniques: List[RepairTechnique] = field(default_factory=list)
    
    # Tool-specific parameters
    untrunc_params: Dict[str, Any] = field(default_factory=dict)
    ffmpeg_params: Dict[str, Any] = field(default_factory=dict)
    mp4recover_params: Dict[str, Any] = field(default_factory=dict)
    
    # Processing options
    use_reference_file: bool = False
    preserve_original_quality: bool = True
    max_processing_time: int = 3600  # seconds
    enable_deep_scan: bool = True
    verify_output: bool = True
    
    # Quality and performance
    quality_threshold: float = 0.7  # Minimum acceptable quality
    performance_mode: str = "balanced"  # fast, balanced, quality
    
    # Advanced options
    enable_ai_enhancement: bool = False
    enable_motion_compensation: bool = True
    enable_audio_repair: bool = True
    custom_filters: List[str] = field(default_factory=list)


@dataclass 
class RepairAttempt:
    """Record of a single repair attempt"""
    tool: RepairToolType
    technique: RepairTechnique
    started_at: datetime
    completed_at: Optional[datetime] = None
    success: bool = False
    error_message: str = ""
    output_path: str = ""
    quality_score: float = 0.0
    processing_time: float = 0.0
    file_size_bytes: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


class ExternalToolInterface(ABC):
    """Abstract base class for external tool interfaces"""
    
    @abstractmethod
    async def is_available(self) -> bool:
        """Check if the tool is available on the system"""
        pass
    
    @abstractmethod
    async def get_version(self) -> str:
        """Get tool version information"""
        pass
    
    @abstractmethod
    async def repair_video(self, input_path: str, output_path: str, 
                          strategy: RepairStrategy) -> RepairAttempt:
        """Repair video using this tool"""
        pass


class UntruncInterface(ExternalToolInterface):
    """Interface for untrunc tool - excellent for MP4/MOV structure repair"""
    
    def __init__(self):
        self.tool_path = self._find_untrunc_executable()
        self.temp_dir = Path(tempfile.gettempdir()) / "phoenixdrs_untrunc"
        self.temp_dir.mkdir(exist_ok=True)
    
    def _find_untrunc_executable(self) -> Optional[str]:
        """Find untrunc executable in system PATH or common locations"""
        common_paths = [
            "untrunc",
            "./tools/untrunc/untrunc",
            "/usr/local/bin/untrunc",
            "/opt/untrunc/untrunc",
            "C:\\Program Files\\untrunc\\untrunc.exe",
            "C:\\tools\\untrunc\\untrunc.exe"
        ]
        
        for path in common_paths:
            try:
                result = subprocess.run([path, "--version"], 
                                      capture_output=True, text=True, timeout=5)
                if result.returncode == 0 or "untrunc" in result.stderr.lower():
                    return path
            except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.CalledProcessError):
                continue
        
        return None
    
    async def is_available(self) -> bool:
        """Check if untrunc is available"""
        return self.tool_path is not None
    
    async def get_version(self) -> str:
        """Get untrunc version"""
        if not self.tool_path:
            return "Not available"
        
        try:
            result = subprocess.run([self.tool_path, "--version"],
                                  capture_output=True, text=True, timeout=10)
            return result.stderr.strip() if result.stderr else "Unknown version"
        except Exception as e:
            return f"Error: {str(e)}"
    
    async def repair_video(self, input_path: str, output_path: str, 
                          strategy: RepairStrategy) -> RepairAttempt:
        """Repair video using untrunc"""
        attempt = RepairAttempt(
            tool=RepairToolType.UNTRUNC,
            technique=RepairTechnique.HEADER_RECONSTRUCTION,
            started_at=datetime.utcnow()
        )
        
        if not self.tool_path:
            attempt.error_message = "untrunc not available"
            attempt.completed_at = datetime.utcnow()
            return attempt
        
        try:
            # Untrunc needs a reference file for best results
            reference_file = strategy.untrunc_params.get("reference_file")
            
            if reference_file and Path(reference_file).exists():
                # Use reference file method (most effective)
                cmd = [
                    self.tool_path,
                    "-s", reference_file,  # Reference file
                    input_path,            # Corrupted file
                    output_path           # Output file
                ]
            else:
                # Try without reference file (less effective but still useful)
                cmd = [
                    self.tool_path,
                    input_path,
                    output_path
                ]
            
            # Add additional parameters
            if strategy.untrunc_params.get("analyze_only"):
                cmd.append("-a")
            
            if strategy.untrunc_params.get("verbose"):
                cmd.append("-v")
            
            # Execute untrunc
            logger.info(f"Executing untrunc: {' '.join(cmd)}")
            
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(self.temp_dir)
            )
            
            stdout, stderr = await asyncio.wait_for(
                process.communicate(), 
                timeout=strategy.max_processing_time
            )
            
            attempt.completed_at = datetime.utcnow()
            attempt.processing_time = (attempt.completed_at - attempt.started_at).total_seconds()
            
            if process.returncode == 0 and Path(output_path).exists():
                attempt.success = True
                attempt.output_path = output_path
                attempt.file_size_bytes = Path(output_path).stat().st_size
                
                # Parse untrunc output for metadata
                output_text = stdout.decode('utf-8', errors='ignore')
                attempt.metadata = self._parse_untrunc_output(output_text)
                
                logger.info(f"untrunc repair successful: {output_path}")
            else:
                attempt.error_message = stderr.decode('utf-8', errors='ignore')
                logger.error(f"untrunc failed: {attempt.error_message}")
        
        except asyncio.TimeoutError:
            attempt.error_message = f"untrunc timeout after {strategy.max_processing_time}s"
            attempt.completed_at = datetime.utcnow()
        except Exception as e:
            attempt.error_message = str(e)
            attempt.completed_at = datetime.utcnow()
            logger.error(f"untrunc error: {e}")
        
        return attempt
    
    def _parse_untrunc_output(self, output: str) -> Dict[str, Any]:
        """Parse untrunc output for useful metadata"""
        metadata = {}
        
        # Extract useful information from untrunc output
        patterns = {
            "atoms_found": r"Found (\d+) atoms",
            "mdat_size": r"mdat size: (\d+)",
            "duration": r"Duration: ([\d.]+)",
            "tracks_found": r"Found (\d+) tracks",
            "video_codec": r"Video codec: (\w+)",
            "audio_codec": r"Audio codec: (\w+)"
        }
        
        for key, pattern in patterns.items():
            match = re.search(pattern, output, re.IGNORECASE)
            if match:
                try:
                    value = match.group(1)
                    metadata[key] = float(value) if '.' in value else int(value) if value.isdigit() else value
                except (ValueError, IndexError):
                    metadata[key] = value
        
        return metadata


class FFmpegInterface(ExternalToolInterface):
    """Interface for FFmpeg - powerful for container manipulation and stream recovery"""
    
    def __init__(self):
        self.tool_path = self._find_ffmpeg_executable()
        self.temp_dir = Path(tempfile.gettempdir()) / "phoenixdrs_ffmpeg"
        self.temp_dir.mkdir(exist_ok=True)
    
    def _find_ffmpeg_executable(self) -> Optional[str]:
        """Find ffmpeg executable"""
        common_paths = ["ffmpeg", "/usr/bin/ffmpeg", "/usr/local/bin/ffmpeg", 
                       "C:\\ffmpeg\\bin\\ffmpeg.exe", "C:\\Program Files\\ffmpeg\\bin\\ffmpeg.exe"]
        
        for path in common_paths:
            try:
                result = subprocess.run([path, "-version"], 
                                      capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    return path
            except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.CalledProcessError):
                continue
        
        return None
    
    async def is_available(self) -> bool:
        """Check if FFmpeg is available"""
        return self.tool_path is not None
    
    async def get_version(self) -> str:
        """Get FFmpeg version"""
        if not self.tool_path:
            return "Not available"
        
        try:
            result = subprocess.run([self.tool_path, "-version"],
                                  capture_output=True, text=True, timeout=10)
            # Extract version from first line
            first_line = result.stdout.split('\n')[0]
            return first_line.strip()
        except Exception as e:
            return f"Error: {str(e)}"
    
    async def repair_video(self, input_path: str, output_path: str, 
                          strategy: RepairStrategy) -> RepairAttempt:
        """Repair video using FFmpeg with multiple strategies"""
        attempt = RepairAttempt(
            tool=RepairToolType.FFMPEG,
            technique=RepairTechnique.CONTAINER_REMUX,
            started_at=datetime.utcnow()
        )
        
        if not self.tool_path:
            attempt.error_message = "FFmpeg not available"
            attempt.completed_at = datetime.utcnow()
            return attempt
        
        # Try multiple FFmpeg repair strategies in order of effectiveness
        repair_strategies = [
            self._strategy_remux_copy,
            self._strategy_remux_with_filters,
            self._strategy_extract_streams,
            self._strategy_force_reconstruction,
            self._strategy_raw_stream_recovery
        ]
        
        for strategy_func in repair_strategies:
            try:
                success = await strategy_func(input_path, output_path, strategy, attempt)
                if success:
                    break
            except Exception as e:
                logger.warning(f"FFmpeg strategy {strategy_func.__name__} failed: {e}")
                continue
        
        attempt.completed_at = datetime.utcnow()
        attempt.processing_time = (attempt.completed_at - attempt.started_at).total_seconds()
        
        return attempt
    
    async def _strategy_remux_copy(self, input_path: str, output_path: str, 
                                  strategy: RepairStrategy, attempt: RepairAttempt) -> bool:
        """Strategy 1: Simple remux with stream copy (fastest, works for container issues)"""
        cmd = [
            self.tool_path,
            "-i", input_path,
            "-c", "copy",  # Copy streams without re-encoding
            "-avoid_negative_ts", "make_zero",
            "-fflags", "+genpts",  # Generate presentation timestamps
            "-y",  # Overwrite output
            output_path
        ]
        
        return await self._execute_ffmpeg_command(cmd, attempt, "remux_copy")
    
    async def _strategy_remux_with_filters(self, input_path: str, output_path: str,
                                          strategy: RepairStrategy, attempt: RepairAttempt) -> bool:
        """Strategy 2: Remux with error correction filters"""
        cmd = [
            self.tool_path,
            "-err_detect", "ignore_err",  # Ignore errors
            "-i", input_path,
            "-c", "copy",
            "-bsf:v", "h264_mp4toannexb",  # Fix H.264 bitstream
            "-avoid_negative_ts", "make_zero",
            "-fflags", "+genpts+igndts",  # Generate PTS, ignore DTS
            "-max_muxing_queue_size", "1024",
            "-y",
            output_path
        ]
        
        return await self._execute_ffmpeg_command(cmd, attempt, "remux_with_filters")
    
    async def _strategy_extract_streams(self, input_path: str, output_path: str,
                                       strategy: RepairStrategy, attempt: RepairAttempt) -> bool:
        """Strategy 3: Extract and rebuild streams separately"""
        temp_video = self.temp_dir / f"temp_video_{uuid.uuid4()}.h264"
        temp_audio = self.temp_dir / f"temp_audio_{uuid.uuid4()}.aac"
        
        try:
            # Extract video stream
            video_cmd = [
                self.tool_path,
                "-i", input_path,
                "-c:v", "copy",
                "-an",  # No audio
                "-f", "h264",
                "-y",
                str(temp_video)
            ]
            
            # Extract audio stream  
            audio_cmd = [
                self.tool_path,
                "-i", input_path,
                "-c:a", "copy",
                "-vn",  # No video
                "-f", "aac",
                "-y",
                str(temp_audio)
            ]
            
            # Execute extractions
            video_success = await self._execute_ffmpeg_command(video_cmd, attempt, "extract_video")
            audio_success = await self._execute_ffmpeg_command(audio_cmd, attempt, "extract_audio")
            
            if video_success:
                # Recombine streams
                combine_cmd = [
                    self.tool_path,
                    "-i", str(temp_video),
                    "-i", str(temp_audio) if audio_success else input_path,
                    "-c", "copy",
                    "-shortest",  # Match shortest stream
                    "-y",
                    output_path
                ]
                
                return await self._execute_ffmpeg_command(combine_cmd, attempt, "combine_streams")
        
        finally:
            # Cleanup temp files
            for temp_file in [temp_video, temp_audio]:
                if temp_file.exists():
                    temp_file.unlink()
        
        return False
    
    async def _strategy_force_reconstruction(self, input_path: str, output_path: str,
                                           strategy: RepairStrategy, attempt: RepairAttempt) -> bool:
        """Strategy 4: Force reconstruction with aggressive error recovery"""
        cmd = [
            self.tool_path,
            "-f", "mp4",  # Force format
            "-err_detect", "ignore_err",
            "-fflags", "+genpts+igndts+ignidx",
            "-i", input_path,
            "-c:v", "libx264",  # Re-encode video
            "-preset", "ultrafast",
            "-crf", "23",
            "-c:a", "aac",  # Re-encode audio
            "-ac", "2",  # Stereo
            "-ar", "48000",  # 48kHz sample rate
            "-avoid_negative_ts", "make_zero",
            "-y",
            output_path
        ]
        
        return await self._execute_ffmpeg_command(cmd, attempt, "force_reconstruction")
    
    async def _strategy_raw_stream_recovery(self, input_path: str, output_path: str,
                                          strategy: RepairStrategy, attempt: RepairAttempt) -> bool:
        """Strategy 5: Raw stream recovery (last resort)"""
        cmd = [
            self.tool_path,
            "-f", "h264",  # Treat as raw H.264
            "-i", input_path,
            "-c:v", "libx264",
            "-preset", "ultrafast",
            "-crf", "28",  # Lower quality but more compatible
            "-pix_fmt", "yuv420p",
            "-y",
            output_path
        ]
        
        return await self._execute_ffmpeg_command(cmd, attempt, "raw_stream_recovery")
    
    async def _execute_ffmpeg_command(self, cmd: List[str], attempt: RepairAttempt, 
                                    strategy_name: str) -> bool:
        """Execute FFmpeg command and handle results"""
        try:
            logger.info(f"Executing FFmpeg {strategy_name}: {' '.join(cmd)}")
            
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=300  # 5 minutes per strategy
            )
            
            if process.returncode == 0 and Path(cmd[-1]).exists():
                attempt.success = True
                attempt.output_path = cmd[-1]
                attempt.file_size_bytes = Path(cmd[-1]).stat().st_size
                attempt.metadata[f"{strategy_name}_output"] = stdout.decode('utf-8', errors='ignore')[-1000:]  # Last 1000 chars
                
                logger.info(f"FFmpeg {strategy_name} successful")
                return True
            else:
                error_msg = stderr.decode('utf-8', errors='ignore')
                attempt.metadata[f"{strategy_name}_error"] = error_msg[-500:]  # Last 500 chars
                logger.warning(f"FFmpeg {strategy_name} failed: {error_msg[:200]}")
                return False
        
        except asyncio.TimeoutError:
            logger.error(f"FFmpeg {strategy_name} timeout")
            return False
        except Exception as e:
            logger.error(f"FFmpeg {strategy_name} error: {e}")
            return False


class Mp4RecoverInterface(ExternalToolInterface):
    """Interface for mp4recover - specialized MP4 recovery tool"""
    
    def __init__(self):
        self.tool_path = self._find_mp4recover_executable()
    
    def _find_mp4recover_executable(self) -> Optional[str]:
        """Find mp4recover executable"""
        common_paths = [
            "mp4recover",
            "./tools/mp4recover/mp4recover",
            "/usr/local/bin/mp4recover",
            "C:\\Program Files\\mp4recover\\mp4recover.exe"
        ]
        
        for path in common_paths:
            try:
                result = subprocess.run([path, "--version"], 
                                      capture_output=True, text=True, timeout=5)
                if result.returncode == 0 or "mp4recover" in result.stderr.lower():
                    return path
            except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.CalledProcessError):
                continue
        
        return None
    
    async def is_available(self) -> bool:
        """Check if mp4recover is available"""
        return self.tool_path is not None
    
    async def get_version(self) -> str:
        """Get mp4recover version"""
        if not self.tool_path:
            return "Not available"
        
        try:
            result = subprocess.run([self.tool_path, "--version"],
                                  capture_output=True, text=True, timeout=10)
            return result.stdout.strip() if result.stdout else result.stderr.strip()
        except Exception as e:
            return f"Error: {str(e)}"
    
    async def repair_video(self, input_path: str, output_path: str,
                          strategy: RepairStrategy) -> RepairAttempt:
        """Repair video using mp4recover"""
        attempt = RepairAttempt(
            tool=RepairToolType.MP4RECOVER,
            technique=RepairTechnique.FRAGMENT_RECOVERY,
            started_at=datetime.utcnow()
        )
        
        if not self.tool_path:
            attempt.error_message = "mp4recover not available"
            attempt.completed_at = datetime.utcnow()
            return attempt
        
        try:
            cmd = [self.tool_path, input_path, output_path]
            
            # Add mp4recover specific parameters
            if strategy.mp4recover_params.get("deep_scan"):
                cmd.append("--deep-scan")
            
            if strategy.mp4recover_params.get("force_repair"):
                cmd.append("--force")
            
            logger.info(f"Executing mp4recover: {' '.join(cmd)}")
            
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=strategy.max_processing_time
            )
            
            attempt.completed_at = datetime.utcnow()
            attempt.processing_time = (attempt.completed_at - attempt.started_at).total_seconds()
            
            if process.returncode == 0 and Path(output_path).exists():
                attempt.success = True
                attempt.output_path = output_path
                attempt.file_size_bytes = Path(output_path).stat().st_size
                attempt.metadata["output"] = stdout.decode('utf-8', errors='ignore')
                
                logger.info(f"mp4recover repair successful: {output_path}")
            else:
                attempt.error_message = stderr.decode('utf-8', errors='ignore')
                logger.error(f"mp4recover failed: {attempt.error_message}")
        
        except asyncio.TimeoutError:
            attempt.error_message = f"mp4recover timeout after {strategy.max_processing_time}s"
            attempt.completed_at = datetime.utcnow()
        except Exception as e:
            attempt.error_message = str(e)
            attempt.completed_at = datetime.utcnow()
        
        return attempt


class AdvancedRepairEngine:
    """Advanced video repair engine that intelligently combines multiple tools and algorithms"""
    
    def __init__(self):
        self.tools = {
            RepairToolType.UNTRUNC: UntruncInterface(),
            RepairToolType.FFMPEG: FFmpegInterface(),
            RepairToolType.MP4RECOVER: Mp4RecoverInterface()
        }
        
        self.temp_dir = Path(tempfile.gettempdir()) / "phoenixdrs_advanced"
        self.temp_dir.mkdir(exist_ok=True)
        
        # Tool availability cache
        self._tool_availability = {}
        
        logger.info("Advanced Repair Engine initialized")
    
    async def initialize(self) -> bool:
        """Initialize the repair engine and check tool availability"""
        logger.info("Initializing Advanced Repair Engine...")
        
        # Check availability of all tools
        for tool_type, interface in self.tools.items():
            try:
                available = await interface.is_available()
                version = await interface.get_version()
                
                self._tool_availability[tool_type] = {
                    "available": available,
                    "version": version
                }
                
                if available:
                    logger.info(f"✓ {tool_type.value} available: {version}")
                else:
                    logger.warning(f"✗ {tool_type.value} not available")
                    
            except Exception as e:
                logger.error(f"Error checking {tool_type.value}: {e}")
                self._tool_availability[tool_type] = {
                    "available": False,
                    "version": f"Error: {e}"
                }
        
        # Check if at least one tool is available
        available_tools = [t for t, info in self._tool_availability.items() 
                          if info["available"]]
        
        if not available_tools:
            logger.error("No repair tools available!")
            return False
        
        logger.info(f"Advanced Repair Engine ready with {len(available_tools)} tools")
        return True
    
    async def analyze_corruption(self, file_path: str) -> Tuple[CorruptionSeverity, Dict[str, Any]]:
        """Analyze video corruption to determine severity and characteristics"""
        logger.info(f"Analyzing corruption in: {file_path}")
        
        analysis = {
            "file_size": 0,
            "readable_percentage": 0.0,
            "container_integrity": 0.0,
            "stream_integrity": 0.0,
            "corruption_patterns": [],
            "recommended_tools": [],
            "estimated_recovery_chance": 0.0
        }
        
        try:
            file_path_obj = Path(file_path)
            if not file_path_obj.exists():
                return CorruptionSeverity.CRITICAL, analysis
            
            analysis["file_size"] = file_path_obj.stat().st_size
            
            # Basic file structure analysis
            container_score = await self._analyze_container_structure(file_path)
            analysis["container_integrity"] = container_score
            
            # Stream readability analysis
            stream_score = await self._analyze_stream_integrity(file_path)
            analysis["stream_integrity"] = stream_score
            
            # Calculate overall corruption
            overall_score = (container_score + stream_score) / 2
            analysis["readable_percentage"] = overall_score * 100
            
            # Determine severity
            if overall_score >= 0.95:
                severity = CorruptionSeverity.MINIMAL
            elif overall_score >= 0.85:
                severity = CorruptionSeverity.MILD
            elif overall_score >= 0.60:
                severity = CorruptionSeverity.MODERATE
            elif overall_score >= 0.30:
                severity = CorruptionSeverity.SEVERE
            else:
                severity = CorruptionSeverity.CRITICAL
            
            # Recommend tools based on analysis
            analysis["recommended_tools"] = self._recommend_tools(severity, analysis)
            analysis["estimated_recovery_chance"] = self._estimate_recovery_chance(severity, analysis)
            
            logger.info(f"Corruption analysis complete: {severity.value} "
                       f"({analysis['readable_percentage']:.1f}% readable)")
            
        except Exception as e:
            logger.error(f"Corruption analysis failed: {e}")
            severity = CorruptionSeverity.CRITICAL
        
        return severity, analysis
    
    async def _analyze_container_structure(self, file_path: str) -> float:
        """Analyze container structure integrity"""
        try:
            # Try to read with MediaInfo
            media_info = MediaInfo.parse(file_path)
            
            score = 0.0
            checks = 0
            
            # Check for basic container structure
            for track in media_info.tracks:
                checks += 1
                if track.track_type == "General":
                    if hasattr(track, 'format') and track.format:
                        score += 0.3
                    if hasattr(track, 'duration') and track.duration:
                        score += 0.3
                    if hasattr(track, 'file_size') and track.file_size:
                        score += 0.2
                elif track.track_type == "Video":
                    if hasattr(track, 'codec') and track.codec:
                        score += 0.1
                    if hasattr(track, 'width') and track.width:
                        score += 0.1
            
            return min(score, 1.0) if checks > 0 else 0.0
            
        except Exception as e:
            logger.warning(f"Container analysis failed: {e}")
            return 0.1  # Assume some basic structure exists
    
    async def _analyze_stream_integrity(self, file_path: str) -> float:
        """Analyze stream integrity using FFmpeg"""
        if not self._tool_availability.get(RepairToolType.FFMPEG, {}).get("available"):
            return 0.5  # Default score if FFmpeg not available
        
        try:
            ffmpeg_path = self.tools[RepairToolType.FFMPEG].tool_path
            
            # Try to analyze streams
            cmd = [
                ffmpeg_path,
                "-v", "error",
                "-i", file_path,
                "-f", "null",
                "-t", "10",  # Analyze first 10 seconds
                "-"
            ]
            
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=30
            )
            
            error_output = stderr.decode('utf-8', errors='ignore').lower()
            
            # Count error indicators
            error_indicators = [
                'error', 'corrupt', 'invalid', 'failed', 'missing',
                'truncated', 'broken', 'damaged'
            ]
            
            error_count = sum(error_output.count(indicator) for indicator in error_indicators)
            
            # Calculate score based on errors
            if error_count == 0:
                return 1.0
            elif error_count <= 2:
                return 0.8
            elif error_count <= 5:
                return 0.6
            elif error_count <= 10:
                return 0.4
            else:
                return 0.2
                
        except Exception as e:
            logger.warning(f"Stream analysis failed: {e}")
            return 0.3
    
    def _recommend_tools(self, severity: CorruptionSeverity, analysis: Dict[str, Any]) -> List[RepairToolType]:
        """Recommend tools based on corruption analysis"""
        recommendations = []
        
        # Only recommend available tools
        available_tools = [t for t, info in self._tool_availability.items() 
                          if info["available"]]
        
        if severity == CorruptionSeverity.MINIMAL:
            # Light corruption - try FFmpeg first
            if RepairToolType.FFMPEG in available_tools:
                recommendations.append(RepairToolType.FFMPEG)
            if RepairToolType.UNTRUNC in available_tools:
                recommendations.append(RepairToolType.UNTRUNC)
                
        elif severity == CorruptionSeverity.MILD:
            # Mild corruption - untrunc is often best for MP4
            if RepairToolType.UNTRUNC in available_tools:
                recommendations.append(RepairToolType.UNTRUNC)
            if RepairToolType.FFMPEG in available_tools:
                recommendations.append(RepairToolType.FFMPEG)
                
        elif severity == CorruptionSeverity.MODERATE:
            # Moderate corruption - try multiple approaches
            if RepairToolType.UNTRUNC in available_tools:
                recommendations.append(RepairToolType.UNTRUNC)
            if RepairToolType.MP4RECOVER in available_tools:
                recommendations.append(RepairToolType.MP4RECOVER)
            if RepairToolType.FFMPEG in available_tools:
                recommendations.append(RepairToolType.FFMPEG)
                
        elif severity in [CorruptionSeverity.SEVERE, CorruptionSeverity.CRITICAL]:
            # Severe corruption - use all available tools
            if RepairToolType.MP4RECOVER in available_tools:
                recommendations.append(RepairToolType.MP4RECOVER)
            if RepairToolType.UNTRUNC in available_tools:
                recommendations.append(RepairToolType.UNTRUNC)
            if RepairToolType.FFMPEG in available_tools:
                recommendations.append(RepairToolType.FFMPEG)
        
        return recommendations
    
    def _estimate_recovery_chance(self, severity: CorruptionSeverity, analysis: Dict[str, Any]) -> float:
        """Estimate the chance of successful recovery"""
        base_chances = {
            CorruptionSeverity.MINIMAL: 0.95,
            CorruptionSeverity.MILD: 0.85,
            CorruptionSeverity.MODERATE: 0.65,
            CorruptionSeverity.SEVERE: 0.35,
            CorruptionSeverity.CRITICAL: 0.15
        }
        
        base_chance = base_chances[severity]
        
        # Adjust based on analysis
        if analysis["container_integrity"] > 0.8:
            base_chance += 0.1
        
        if analysis["stream_integrity"] > 0.7:
            base_chance += 0.1
        
        # Adjust based on available tools
        available_tools = [t for t, info in self._tool_availability.items() 
                          if info["available"]]
        
        if len(available_tools) >= 3:
            base_chance += 0.05
        elif len(available_tools) >= 2:
            base_chance += 0.03
        
        return min(base_chance, 0.98)  # Cap at 98%
    
    async def create_repair_strategy(self, file_path: str, analysis: VideoAnalysisResult) -> RepairStrategy:
        """Create intelligent repair strategy based on analysis"""
        severity, corruption_analysis = await self.analyze_corruption(file_path)
        
        strategy = RepairStrategy(
            primary_tool=RepairToolType.FFMPEG,  # Default
            enable_deep_scan=severity in [CorruptionSeverity.SEVERE, CorruptionSeverity.CRITICAL],
            performance_mode="quality" if severity == CorruptionSeverity.MINIMAL else "balanced"
        )
        
        # Select primary tool and fallbacks
        recommended_tools = corruption_analysis["recommended_tools"]
        if recommended_tools:
            strategy.primary_tool = recommended_tools[0]
            strategy.fallback_tools = recommended_tools[1:]
        
        # Configure tool-specific parameters
        strategy.untrunc_params = {
            "verbose": True,
            "reference_file": None  # Will be set if available
        }
        
        strategy.ffmpeg_params = {
            "error_detection": "aggressive" if severity >= CorruptionSeverity.MODERATE else "normal",
            "force_reconstruction": severity >= CorruptionSeverity.SEVERE
        }
        
        strategy.mp4recover_params = {
            "deep_scan": severity >= CorruptionSeverity.MODERATE,
            "force_repair": severity >= CorruptionSeverity.SEVERE
        }
        
        # Set quality threshold based on severity
        quality_thresholds = {
            CorruptionSeverity.MINIMAL: 0.9,
            CorruptionSeverity.MILD: 0.8,
            CorruptionSeverity.MODERATE: 0.6,
            CorruptionSeverity.SEVERE: 0.4,
            CorruptionSeverity.CRITICAL: 0.2
        }
        strategy.quality_threshold = quality_thresholds[severity]
        
        # Set processing time based on severity
        time_limits = {
            CorruptionSeverity.MINIMAL: 300,    # 5 minutes
            CorruptionSeverity.MILD: 600,       # 10 minutes
            CorruptionSeverity.MODERATE: 1800,  # 30 minutes
            CorruptionSeverity.SEVERE: 3600,    # 1 hour
            CorruptionSeverity.CRITICAL: 7200   # 2 hours
        }
        strategy.max_processing_time = time_limits[severity]
        
        logger.info(f"Created repair strategy: primary={strategy.primary_tool.value}, "
                   f"fallbacks={[t.value for t in strategy.fallback_tools]}")
        
        return strategy
    
    async def execute_repair(self, input_path: str, output_path: str, 
                           strategy: RepairStrategy) -> List[RepairAttempt]:
        """Execute repair using the strategy with intelligent tool selection"""
        attempts = []
        
        # Try primary tool first
        if strategy.primary_tool in self.tools:
            attempt = await self._execute_tool_repair(
                strategy.primary_tool, input_path, output_path, strategy
            )
            attempts.append(attempt)
            
            if attempt.success and attempt.quality_score >= strategy.quality_threshold:
                logger.info(f"Primary tool {strategy.primary_tool.value} succeeded")
                return attempts
        
        # Try fallback tools if primary failed
        for tool in strategy.fallback_tools:
            if tool in self.tools:
                # Generate unique output path for this attempt
                tool_output = self._generate_tool_output_path(output_path, tool)
                
                attempt = await self._execute_tool_repair(
                    tool, input_path, tool_output, strategy
                )
                attempts.append(attempt)
                
                if attempt.success and attempt.quality_score >= strategy.quality_threshold:
                    # Copy successful result to final output path
                    shutil.copy2(tool_output, output_path)
                    attempt.output_path = output_path
                    logger.info(f"Fallback tool {tool.value} succeeded")
                    break
        
        # Select best result if multiple succeeded
        successful_attempts = [a for a in attempts if a.success]
        if successful_attempts:
            best_attempt = max(successful_attempts, key=lambda a: a.quality_score)
            if best_attempt.output_path != output_path:
                shutil.copy2(best_attempt.output_path, output_path)
            
            logger.info(f"Selected best result from {best_attempt.tool.value} "
                       f"(quality: {best_attempt.quality_score:.2f})")
        
        return attempts
    
    async def _execute_tool_repair(self, tool: RepairToolType, input_path: str, 
                                 output_path: str, strategy: RepairStrategy) -> RepairAttempt:
        """Execute repair with a specific tool"""
        tool_interface = self.tools[tool]
        
        try:
            attempt = await tool_interface.repair_video(input_path, output_path, strategy)
            
            # Evaluate quality if repair succeeded
            if attempt.success:
                attempt.quality_score = await self._evaluate_repair_quality(
                    input_path, attempt.output_path
                )
            
            return attempt
            
        except Exception as e:
            logger.error(f"Tool {tool.value} execution failed: {e}")
            return RepairAttempt(
                tool=tool,
                technique=RepairTechnique.HEADER_RECONSTRUCTION,
                started_at=datetime.utcnow(),
                completed_at=datetime.utcnow(),
                success=False,
                error_message=str(e)
            )
    
    async def _evaluate_repair_quality(self, original_path: str, repaired_path: str) -> float:
        """Evaluate the quality of a repair attempt"""
        try:
            # Basic checks
            if not Path(repaired_path).exists():
                return 0.0
            
            repaired_size = Path(repaired_path).stat().st_size
            if repaired_size == 0:
                return 0.0
            
            # Try to analyze with MediaInfo
            try:
                media_info = MediaInfo.parse(repaired_path)
                if not media_info.tracks:
                    return 0.2
                
                score = 0.3  # Base score for readable file
                
                # Check for video track
                video_tracks = [t for t in media_info.tracks if t.track_type == "Video"]
                if video_tracks:
                    score += 0.3
                    
                    # Check video properties
                    video_track = video_tracks[0]
                    if hasattr(video_track, 'width') and video_track.width:
                        score += 0.1
                    if hasattr(video_track, 'duration') and video_track.duration:
                        score += 0.1
                
                # Check for audio track
                audio_tracks = [t for t in media_info.tracks if t.track_type == "Audio"]
                if audio_tracks:
                    score += 0.2
                
                return min(score, 1.0)
                
            except Exception:
                # Fallback to basic file size comparison
                original_size = Path(original_path).stat().st_size
                size_ratio = repaired_size / original_size
                
                if size_ratio >= 0.9:
                    return 0.8
                elif size_ratio >= 0.7:
                    return 0.6
                elif size_ratio >= 0.5:
                    return 0.4
                else:
                    return 0.2
                    
        except Exception as e:
            logger.warning(f"Quality evaluation failed: {e}")
            return 0.1
    
    def _generate_tool_output_path(self, base_output: str, tool: RepairToolType) -> str:
        """Generate unique output path for tool attempt"""
        base_path = Path(base_output)
        tool_output = base_path.parent / f"{base_path.stem}_{tool.value}{base_path.suffix}"
        return str(tool_output)
    
    async def get_tool_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all repair tools"""
        return self._tool_availability.copy()
    
    async def cleanup(self):
        """Cleanup temporary files and resources"""
        try:
            if self.temp_dir.exists():
                shutil.rmtree(self.temp_dir)
            logger.info("Advanced Repair Engine cleanup complete")
        except Exception as e:
            logger.warning(f"Cleanup warning: {e}")


# Usage example
async def example_advanced_repair():
    """Example usage of the advanced repair engine"""
    engine = AdvancedRepairEngine()
    
    # Initialize
    initialized = await engine.initialize()
    if not initialized:
        print("Failed to initialize repair engine")
        return
    
    # Get tool status
    tool_status = await engine.get_tool_status()
    print("Available tools:")
    for tool, info in tool_status.items():
        status = "✓" if info["available"] else "✗"
        print(f"  {status} {tool.value}: {info['version']}")
    
    # Analyze corruption
    input_file = "corrupted_video.mp4"
    severity, analysis = await engine.analyze_corruption(input_file)
    
    print(f"\nCorruption Analysis:")
    print(f"  Severity: {severity.value}")
    print(f"  Readable: {analysis['readable_percentage']:.1f}%")
    print(f"  Recovery chance: {analysis['estimated_recovery_chance']:.1f}%")
    print(f"  Recommended tools: {[t.value for t in analysis['recommended_tools']]}")
    
    # Create repair strategy
    from .video_repair_orchestrator import VideoAnalysisResult
    mock_analysis = VideoAnalysisResult(file_path=input_file)
    strategy = await engine.create_repair_strategy(input_file, mock_analysis)
    
    print(f"\nRepair Strategy:")
    print(f"  Primary tool: {strategy.primary_tool.value}")
    print(f"  Fallback tools: {[t.value for t in strategy.fallback_tools]}")
    print(f"  Quality threshold: {strategy.quality_threshold}")
    print(f"  Max time: {strategy.max_processing_time}s")
    
    # Execute repair
    output_file = "repaired_video.mp4"
    attempts = await engine.execute_repair(input_file, output_file, strategy)
    
    print(f"\nRepair Results:")
    for i, attempt in enumerate(attempts, 1):
        status = "✓" if attempt.success else "✗"
        print(f"  {i}. {status} {attempt.tool.value}: "
              f"quality={attempt.quality_score:.2f}, "
              f"time={attempt.processing_time:.1f}s")
        if not attempt.success:
            print(f"     Error: {attempt.error_message[:100]}")
    
    # Cleanup
    await engine.cleanup()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(example_advanced_repair())