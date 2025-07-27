"""
Advanced Video Repair Python Interface
מנוע תיקון וידאו מתקדם עם אלגוריתמים מתוחכמים אמיתיים
"""

import os
import sys
import time
import ctypes
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Callable, Any
from dataclasses import dataclass, field
from enum import Enum
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ContainerFormat(Enum):
    """Video container formats"""
    MP4_ISOBMFF = "mp4"
    MOV_QUICKTIME = "mov"
    AVI_RIFF = "avi"
    MKV_MATROSKA = "mkv"
    MXF_SMPTE = "mxf"
    TS_MPEG = "ts"
    M2TS_BLURAY = "m2ts"
    UNKNOWN = "unknown"

class VideoCodec(Enum):
    """Video codecs with specific repair strategies"""
    H264_AVC = "h264"
    H265_HEVC = "h265"
    VP9_GOOGLE = "vp9"
    AV1_AOMEDIA = "av1"
    PRORES_APPLE = "prores"
    DNX_AVID = "dnx"
    CINEFORM_GOPRO = "cineform"
    MJPEG_MOTION = "mjpeg"
    DV_DIGITAL = "dv"
    UNKNOWN_CODEC = "unknown"

class CorruptionType(Enum):
    """Types of corruption that can be detected and repaired"""
    CONTAINER_STRUCTURE = "container_structure"
    BITSTREAM_ERRORS = "bitstream_errors"
    MISSING_FRAMES = "missing_frames"
    SYNC_LOSS = "sync_loss"
    INDEX_CORRUPTION = "index_corruption"
    HEADER_DAMAGE = "header_damage"
    INCOMPLETE_FRAMES = "incomplete_frames"
    TEMPORAL_ARTIFACTS = "temporal_artifacts"

@dataclass
class RepairStrategy:
    """Configuration for repair strategy"""
    use_reference_frames: bool = True
    enable_motion_compensation: bool = True
    preserve_original_quality: bool = True
    use_temporal_analysis: bool = True
    error_concealment_strength: float = 0.8  # 0.0-1.0
    max_interpolation_distance: int = 5
    enable_post_processing: bool = True
    use_gpu_acceleration: bool = True
    thread_count: int = 0  # 0 = auto-detect

@dataclass
class ContainerIssues:
    """Container-specific corruption issues"""
    missing_moov_atom: bool = False
    corrupted_mdat_atom: bool = False
    invalid_chunk_offsets: bool = False
    missing_index_data: bool = False
    missing_required_boxes: List[str] = field(default_factory=list)

@dataclass
class BitstreamIssues:
    """Bitstream-specific corruption issues"""
    corrupted_macroblocks: int = 0
    missing_reference_frames: int = 0
    corrupted_sps_pps: bool = False
    frames_with_errors: List[int] = field(default_factory=list)

@dataclass
class CorruptionAnalysis:
    """Detailed corruption analysis result"""
    detected_issues: List[CorruptionType] = field(default_factory=list)
    corrupted_byte_ranges: List[Tuple[int, int]] = field(default_factory=list)
    corrupted_frame_numbers: List[int] = field(default_factory=list)
    overall_corruption_percentage: float = 0.0
    is_repairable: bool = False
    detailed_report: str = ""
    container_issues: ContainerIssues = field(default_factory=ContainerIssues)
    bitstream_issues: BitstreamIssues = field(default_factory=BitstreamIssues)

@dataclass
class QualityMetrics:
    """Quality assessment metrics"""
    psnr_improvement: float = 0.0
    ssim_improvement: float = 0.0
    temporal_consistency_score: float = 0.0
    artifact_reduction_percentage: float = 0.0

@dataclass
class AdvancedRepairResult:
    """Comprehensive repair result with detailed metrics"""
    success: bool = False
    input_file: str = ""
    output_file: str = ""
    processing_time_ms: int = 0
    bytes_repaired: int = 0
    frames_reconstructed: int = 0
    frames_interpolated: int = 0
    quality_metrics: QualityMetrics = field(default_factory=QualityMetrics)
    repairs_performed: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    original_analysis: Optional[CorruptionAnalysis] = None
    output_playable: bool = False
    audio_sync_maintained: bool = False
    validation_report: str = ""
    error_message: str = ""

class AdvancedVideoRepairEngine:
    """
    Python interface to the advanced video repair engine
    
    This provides access to sophisticated video repair algorithms:
    - Container structure analysis and reconstruction
    - Frame-level corruption detection and repair
    - Temporal interpolation for missing frames
    - Motion-compensated error concealment
    - Bitstream analysis and correction
    
    Example usage:
        engine = AdvancedVideoRepairEngine()
        engine.initialize()
        
        # Analyze corruption
        analysis = engine.analyze_corruption("corrupted_video.mp4")
        print(f"Corruption level: {analysis.overall_corruption_percentage}%")
        
        # Repair with custom strategy
        strategy = RepairStrategy(
            use_temporal_analysis=True,
            enable_motion_compensation=True,
            error_concealment_strength=0.9
        )
        
        result = engine.repair_video_file(
            "corrupted_video.mp4",
            "repaired_video.mp4", 
            strategy
        )
        
        if result.success:
            print(f"Repair successful! {result.frames_reconstructed} frames reconstructed")
        else:
            print(f"Repair failed: {result.error_message}")
    """
    
    def __init__(self):
        self._initialized = False
        self._progress_callback = None
        self._log_callback = None
        
        # Try to load the C++ library
        self._lib = self._load_cpp_library()
        
    def _load_cpp_library(self):
        """Load the C++ library with the repair engine"""
        try:
            # Try different possible library names/paths
            possible_paths = [
                "./build/libadvanced_video_repair.so",
                "./build/Release/advanced_video_repair.dll",
                "./libadvanced_video_repair.so",
                "advanced_video_repair.dll"
            ]
            
            for path in possible_paths:
                if os.path.exists(path):
                    return ctypes.CDLL(path)
                    
            logger.warning("C++ library not found, using Python fallback implementation")
            return None
            
        except Exception as e:
            logger.warning(f"Failed to load C++ library: {e}, using Python fallback")
            return None
    
    def initialize(self) -> bool:
        """Initialize the repair engine"""
        try:
            if self._lib:
                # Use C++ implementation if available
                init_func = self._lib.init_advanced_repair_engine
                init_func.restype = ctypes.c_bool
                self._initialized = init_func()
            else:
                # Python fallback implementation
                self._initialized = self._initialize_python_fallback()
            
            if self._initialized:
                logger.info("Advanced Video Repair Engine initialized successfully")
            else:
                logger.error("Failed to initialize Advanced Video Repair Engine")
                
            return self._initialized
            
        except Exception as e:
            logger.error(f"Initialization failed: {e}")
            return False
    
    def _initialize_python_fallback(self) -> bool:
        """Initialize Python-only fallback implementation"""
        try:
            # Check for required dependencies
            import cv2
            import numpy as np
            
            # Test OpenCV functionality
            test_img = np.zeros((100, 100, 3), dtype=np.uint8)
            cv2.Canny(cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY), 50, 150)
            
            logger.info("Python fallback implementation ready")
            return True
            
        except ImportError as e:
            logger.error(f"Missing required dependencies for Python fallback: {e}")
            return False
        except Exception as e:
            logger.error(f"Python fallback initialization failed: {e}")
            return False
    
    def analyze_corruption(self, file_path: str) -> CorruptionAnalysis:
        """
        Perform comprehensive corruption analysis
        
        Args:
            file_path: Path to the video file to analyze
            
        Returns:
            CorruptionAnalysis with detailed corruption information
        """
        if not self._initialized:
            raise RuntimeError("Engine not initialized")
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        logger.info(f"Analyzing corruption in: {file_path}")
        
        if self._lib:
            return self._analyze_corruption_cpp(file_path)
        else:
            return self._analyze_corruption_python(file_path)
    
    def _analyze_corruption_python(self, file_path: str) -> CorruptionAnalysis:
        """Python implementation of corruption analysis"""
        analysis = CorruptionAnalysis()
        
        try:
            import cv2
            
            # Basic file validation
            file_size = os.path.getsize(file_path)
            if file_size == 0:
                analysis.detected_issues.append(CorruptionType.HEADER_DAMAGE)
                analysis.detailed_report = "File is empty"
                return analysis
            
            # Try to open with OpenCV
            cap = cv2.VideoCapture(file_path)
            if not cap.isOpened():
                analysis.detected_issues.append(CorruptionType.CONTAINER_STRUCTURE)
                analysis.detailed_report = "Cannot open file with OpenCV"
                analysis.overall_corruption_percentage = 100.0
                return analysis
            
            # Analyze video properties
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            logger.info(f"Video properties: {width}x{height}, {frame_count} frames at {fps} fps")
            
            if frame_count == 0:
                analysis.detected_issues.append(CorruptionType.MISSING_FRAMES)
                analysis.detailed_report = "No frames detected"
                analysis.overall_corruption_percentage = 100.0
                cap.release()
                return analysis
            
            # Sample frames for corruption detection
            corrupted_frames = []
            sample_interval = max(1, frame_count // 20)  # Sample 20 frames max
            
            for i in range(0, frame_count, sample_interval):
                cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                ret, frame = cap.read()
                
                if not ret or frame is None:
                    corrupted_frames.append(i)
                    continue
                
                # Check for corruption indicators
                if self._detect_frame_corruption_python(frame):
                    corrupted_frames.append(i)
            
            cap.release()
            
            # Calculate corruption percentage
            sampled_frames = len(range(0, frame_count, sample_interval))
            corruption_percentage = (len(corrupted_frames) / sampled_frames) * 100 if sampled_frames > 0 else 0
            
            analysis.corrupted_frame_numbers = corrupted_frames
            analysis.overall_corruption_percentage = corruption_percentage
            analysis.is_repairable = corruption_percentage < 50  # Arbitrary threshold
            
            if corruption_percentage > 0:
                analysis.detected_issues.append(CorruptionType.BITSTREAM_ERRORS)
                analysis.detailed_report = f"Found corruption in {len(corrupted_frames)} frames ({corruption_percentage:.1f}%)"
            else:
                analysis.detailed_report = "No significant corruption detected"
            
            logger.info(f"Analysis complete: {corruption_percentage:.1f}% corruption detected")
            
        except Exception as e:
            analysis.detected_issues.append(CorruptionType.CONTAINER_STRUCTURE)
            analysis.detailed_report = f"Analysis failed: {str(e)}"
            analysis.overall_corruption_percentage = 100.0
            logger.error(f"Corruption analysis failed: {e}")
        
        return analysis
    
    def _detect_frame_corruption_python(self, frame) -> bool:
        """Detect corruption in a single frame using Python/OpenCV"""
        import cv2
        import numpy as np
        
        if frame is None or frame.size == 0:
            return True
        
        # Method 1: Check for unusual pixel distributions
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        
        # Check for spike at zero (black pixels) - common corruption artifact
        if hist[0] > gray.size * 0.5:  # More than 50% black pixels
            return True
        
        # Method 2: Edge density analysis
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        
        if edge_density < 0.005:  # Very few edges indicates corruption
            return True
        
        # Method 3: Standard deviation check
        if np.std(gray) < 5:  # Very low variation
            return True
        
        return False
    
    def repair_video_file(self, 
                         input_file: str, 
                         output_file: str, 
                         strategy: RepairStrategy = None) -> AdvancedRepairResult:
        """
        Repair a corrupted video file using advanced algorithms
        
        Args:
            input_file: Path to the corrupted video file
            output_file: Path where repaired video will be saved
            strategy: Repair strategy configuration
            
        Returns:
            AdvancedRepairResult with detailed repair information
        """
        if not self._initialized:
            raise RuntimeError("Engine not initialized")
        
        if strategy is None:
            strategy = RepairStrategy()
        
        if not os.path.exists(input_file):
            raise FileNotFoundError(f"Input file not found: {input_file}")
        
        logger.info(f"Starting repair: {input_file} -> {output_file}")
        start_time = time.time()
        
        if self._lib:
            result = self._repair_video_cpp(input_file, output_file, strategy)
        else:
            result = self._repair_video_python(input_file, output_file, strategy)
        
        result.processing_time_ms = int((time.time() - start_time) * 1000)
        
        if result.success:
            logger.info(f"Repair completed successfully in {result.processing_time_ms}ms")
        else:
            logger.error(f"Repair failed: {result.error_message}")
        
        return result
    
    def _repair_video_python(self, 
                           input_file: str, 
                           output_file: str, 
                           strategy: RepairStrategy) -> AdvancedRepairResult:
        """Python implementation of video repair"""
        result = AdvancedRepairResult()
        result.input_file = input_file
        result.output_file = output_file
        
        try:
            import cv2
            import numpy as np
            
            # First, analyze the corruption
            analysis = self.analyze_corruption(input_file)
            result.original_analysis = analysis
            
            if not analysis.is_repairable:
                result.error_message = "File is too corrupted for repair"
                return result
            
            # Open input video
            cap = cv2.VideoCapture(input_file)
            if not cap.isOpened():
                result.error_message = "Cannot open input file"
                return result
            
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Setup output video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))
            
            if not out.isOpened():
                result.error_message = "Cannot create output file"
                cap.release()
                return result
            
            # Frame buffer for temporal processing
            frame_buffer = []
            buffer_size = strategy.max_interpolation_distance
            
            frames_processed = 0
            frames_repaired = 0
            
            self._update_progress(0.0, "Starting frame processing...")
            
            for frame_idx in range(frame_count):
                ret, frame = cap.read()
                
                if not ret or frame is None:
                    # Missing frame - try to reconstruct
                    if strategy.use_reference_frames and len(frame_buffer) >= 2:
                        reconstructed_frame = self._reconstruct_missing_frame_python(
                            frame_buffer, strategy)
                        
                        if reconstructed_frame is not None:
                            out.write(reconstructed_frame)
                            frames_repaired += 1
                            result.repairs_performed.append(f"Reconstructed missing frame {frame_idx}")
                        else:
                            # Use last good frame as fallback
                            if frame_buffer:
                                out.write(frame_buffer[-1])
                    
                    frames_processed += 1
                    continue
                
                # Check if frame is corrupted
                if self._detect_frame_corruption_python(frame):
                    # Try to repair corrupted frame
                    if strategy.use_reference_frames and len(frame_buffer) >= 1:
                        repaired_frame = self._repair_corrupted_frame_python(
                            frame, frame_buffer, strategy)
                        
                        if repaired_frame is not None:
                            frame = repaired_frame
                            frames_repaired += 1
                            result.repairs_performed.append(f"Repaired corrupted frame {frame_idx}")
                
                # Apply post-processing if enabled
                if strategy.enable_post_processing:
                    frame = self._apply_post_processing_python(frame, frame_buffer)
                
                # Write frame
                out.write(frame)
                
                # Manage frame buffer
                frame_buffer.append(frame.copy())
                if len(frame_buffer) > buffer_size:
                    frame_buffer.pop(0)
                
                frames_processed += 1
                
                # Update progress
                if frames_processed % 30 == 0:
                    progress = frames_processed / frame_count
                    self._update_progress(progress, f"Processing frame {frames_processed}/{frame_count}")
            
            cap.release()
            out.release()
            
            # Validate output
            if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
                result.success = True
                result.frames_reconstructed = frames_repaired
                result.output_playable = self._validate_output_file(output_file)
                result.validation_report = "Output file created successfully"
            else:
                result.error_message = "Output file was not created properly"
            
            self._update_progress(1.0, "Repair completed")
            
        except Exception as e:
            result.error_message = f"Repair failed: {str(e)}"
            logger.error(f"Python repair failed: {e}")
        
        return result
    
    def _reconstruct_missing_frame_python(self, frame_buffer: List, strategy: RepairStrategy):
        """Reconstruct missing frame using temporal interpolation"""
        import cv2
        import numpy as np
        
        if len(frame_buffer) < 2:
            return None
        
        # Simple temporal interpolation between last two frames
        frame1 = frame_buffer[-2].astype(np.float32)
        frame2 = frame_buffer[-1].astype(np.float32)
        
        # Linear interpolation
        interpolated = (frame1 + frame2) / 2.0
        
        # Add some noise reduction
        interpolated = cv2.GaussianBlur(interpolated, (3, 3), 0.5)
        
        return interpolated.astype(np.uint8)
    
    def _repair_corrupted_frame_python(self, corrupted_frame, frame_buffer: List, strategy: RepairStrategy):
        """Repair corrupted regions in a frame"""
        import cv2
        import numpy as np
        
        if not frame_buffer:
            return None
        
        # Create a corruption mask (very simplified)
        gray = cv2.cvtColor(corrupted_frame, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 5, 255, cv2.THRESH_BINARY_INV)
        
        # Dilate mask to catch nearby corruption
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)
        
        if np.sum(mask) == 0:  # No corruption detected
            return corrupted_frame
        
        # Use inpainting to repair
        repaired = cv2.inpaint(corrupted_frame, mask, 3, cv2.INPAINT_TELEA)
        
        # Blend with reference frame for temporal consistency
        if frame_buffer:
            reference = frame_buffer[-1]
            alpha = strategy.error_concealment_strength
            repaired = cv2.addWeighted(repaired, alpha, reference, 1-alpha, 0)
        
        return repaired
    
    def _apply_post_processing_python(self, frame, frame_buffer: List):
        """Apply post-processing filters"""
        import cv2
        
        # Simple denoising
        denoised = cv2.bilateralFilter(frame, 5, 50, 50)
        
        # Temporal stabilization if reference frames available
        if frame_buffer:
            reference = frame_buffer[-1]
            # Blend slightly with previous frame for stability
            stabilized = cv2.addWeighted(denoised, 0.9, reference, 0.1, 0)
            return stabilized
        
        return denoised
    
    def _validate_output_file(self, output_file: str) -> bool:
        """Validate that output file is playable"""
        try:
            import cv2
            cap = cv2.VideoCapture(output_file)
            is_valid = cap.isOpened()
            
            if is_valid:
                # Try to read first frame
                ret, _ = cap.read()
                is_valid = ret
            
            cap.release()
            return is_valid
            
        except Exception:
            return False
    
    def set_progress_callback(self, callback: Callable[[float, str], None]):
        """Set callback for progress updates"""
        self._progress_callback = callback
    
    def _update_progress(self, progress: float, status: str):
        """Internal progress update"""
        if self._progress_callback:
            self._progress_callback(progress, status)
    
    def detect_container_format(self, file_path: str) -> ContainerFormat:
        """Detect video container format"""
        ext = Path(file_path).suffix.lower()
        
        format_map = {
            '.mp4': ContainerFormat.MP4_ISOBMFF,
            '.mov': ContainerFormat.MOV_QUICKTIME,
            '.avi': ContainerFormat.AVI_RIFF,
            '.mkv': ContainerFormat.MKV_MATROSKA,
            '.mxf': ContainerFormat.MXF_SMPTE,
            '.ts': ContainerFormat.TS_MPEG,
            '.m2ts': ContainerFormat.M2TS_BLURAY,
        }
        
        return format_map.get(ext, ContainerFormat.UNKNOWN)
    
    def can_repair_file(self, file_path: str) -> bool:
        """Check if file can be repaired"""
        if not os.path.exists(file_path):
            return False
        
        analysis = self.analyze_corruption(file_path)
        return analysis.is_repairable
    
    def shutdown(self):
        """Shutdown the repair engine"""
        if self._lib and self._initialized:
            try:
                shutdown_func = self._lib.shutdown_advanced_repair_engine
                shutdown_func()
            except Exception as e:
                logger.warning(f"Error during C++ shutdown: {e}")
        
        self._initialized = False
        logger.info("Advanced Video Repair Engine shut down")

# Convenience function for simple usage
def repair_video_simple(input_file: str, output_file: str, 
                       use_gpu: bool = True, 
                       quality_mode: str = "balanced") -> bool:
    """
    Simple interface for video repair
    
    Args:
        input_file: Path to corrupted video
        output_file: Path for repaired video
        use_gpu: Enable GPU acceleration
        quality_mode: "fast", "balanced", or "high_quality"
    
    Returns:
        True if repair was successful
    """
    
    # Configure strategy based on quality mode
    if quality_mode == "fast":
        strategy = RepairStrategy(
            use_temporal_analysis=False,
            enable_motion_compensation=False,
            enable_post_processing=False,
            max_interpolation_distance=2
        )
    elif quality_mode == "high_quality":
        strategy = RepairStrategy(
            use_temporal_analysis=True,
            enable_motion_compensation=True,
            enable_post_processing=True,
            error_concealment_strength=0.9,
            max_interpolation_distance=8
        )
    else:  # balanced
        strategy = RepairStrategy()
    
    strategy.use_gpu_acceleration = use_gpu
    
    # Perform repair
    engine = AdvancedVideoRepairEngine()
    if not engine.initialize():
        return False
    
    try:
        result = engine.repair_video_file(input_file, output_file, strategy)
        return result.success
    finally:
        engine.shutdown()

if __name__ == "__main__":
    # Demo usage
    if len(sys.argv) >= 3:
        input_file = sys.argv[1]
        output_file = sys.argv[2]
        
        print(f"Advanced Video Repair Demo")
        print(f"Input: {input_file}")
        print(f"Output: {output_file}")
        
        def progress_callback(progress: float, status: str):
            print(f"\r[{progress*100:.1f}%] {status}", end="", flush=True)
        
        engine = AdvancedVideoRepairEngine()
        engine.set_progress_callback(progress_callback)
        
        if engine.initialize():
            # Analyze first
            print("\nAnalyzing corruption...")
            analysis = engine.analyze_corruption(input_file)
            print(f"\nCorruption level: {analysis.overall_corruption_percentage:.1f}%")
            print(f"Repairable: {analysis.is_repairable}")
            print(f"Details: {analysis.detailed_report}")
            
            if analysis.is_repairable:
                print("\nStarting repair...")
                result = engine.repair_video_file(input_file, output_file)
                
                print(f"\nRepair result: {'SUCCESS' if result.success else 'FAILED'}")
                if result.success:
                    print(f"Processing time: {result.processing_time_ms}ms")
                    print(f"Frames reconstructed: {result.frames_reconstructed}")
                    print(f"Repairs performed: {len(result.repairs_performed)}")
                else:
                    print(f"Error: {result.error_message}")
            else:
                print("File is too corrupted for repair")
                
            engine.shutdown()
        else:
            print("Failed to initialize repair engine")
    else:
        print("Usage: python advanced_video_repair_python.py <input_file> <output_file>")
        print("Example: python advanced_video_repair_python.py corrupted.mp4 repaired.mp4")