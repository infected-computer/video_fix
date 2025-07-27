#!/usr/bin/env python3
"""
PhoenixDRS Professional - Unified Python-C++ Interface
ממשק מאוחד Python-C++ - PhoenixDRS מקצועי

This module provides a unified interface that combines Python flexibility
with C++ performance for digital forensics operations.
"""

import sys
import os
import argparse
import time
import json
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
import traceback

# Try to import the C++ bindings
try:
    import phoenixdrs_cpp as cpp_backend
    CPP_AVAILABLE = True
    print("✓ C++ backend loaded successfully")
except ImportError as e:
    CPP_AVAILABLE = False
    print(f"✗ C++ backend not available: {e}")
    print("  Falling back to pure Python implementation")

# Import Python modules
try:
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src', 'python'))
    
    from disk_imager import DiskImager as PythonDiskImager
    from file_carver import FileCarver as PythonFileCarver
    from video_rebuilder import VideoRebuilder as PythonVideoRebuilder
    from logging_config import setup_logging
    PYTHON_MODULES_AVAILABLE = True
except ImportError as e:
    PYTHON_MODULES_AVAILABLE = False
    print(f"✗ Python modules not available: {e}")

@dataclass
class OperationResult:
    """Unified result structure for all operations"""
    success: bool
    message: str
    data: Optional[Dict[str, Any]] = None
    error_code: Optional[str] = None
    elapsed_time: float = 0.0
    files_processed: int = 0
    bytes_processed: int = 0

class UnifiedForensicsAPI:
    """
    Unified API that automatically chooses between C++ and Python implementations
    based on availability and performance requirements.
    """
    
    def __init__(self, prefer_cpp: bool = True, enable_logging: bool = True):
        self.prefer_cpp = prefer_cpp and CPP_AVAILABLE
        self.enable_logging = enable_logging
        self.logger = None
        
        # Initialize logging
        if enable_logging:
            self.setup_logging()
        
        # Initialize backends
        self.cpp_api = None
        self.python_modules = {}
        
        if CPP_AVAILABLE:
            try:
                self.cpp_api = cpp_backend.ForensicsAPI.instance()
                self.cpp_api.initialize()
                if self.logger:
                    self.logger.info("C++ backend initialized successfully")
            except Exception as e:
                if self.logger:
                    self.logger.error(f"Failed to initialize C++ backend: {e}")
                self.cpp_api = None
        
        if PYTHON_MODULES_AVAILABLE:
            try:
                self.python_modules = {
                    'disk_imager': PythonDiskImager,
                    'file_carver': PythonFileCarver,
                    'video_rebuilder': PythonVideoRebuilder
                }
                if self.logger:
                    self.logger.info("Python modules initialized successfully")
            except Exception as e:
                if self.logger:
                    self.logger.error(f"Failed to initialize Python modules: {e}")
    
    def setup_logging(self):
        """Setup logging configuration"""
        log_dir = Path.home() / '.phoenixdrs' / 'logs'
        log_dir.mkdir(parents=True, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / 'phoenixdrs_unified.log'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger('PhoenixDRS.Unified')
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get comprehensive system information"""
        info = {
            'python_version': sys.version,
            'platform': sys.platform,
            'cpp_backend_available': CPP_AVAILABLE,
            'python_modules_available': PYTHON_MODULES_AVAILABLE,
            'preferred_backend': 'C++' if self.prefer_cpp else 'Python'
        }
        
        if CPP_AVAILABLE and self.cpp_api:
            try:
                cpp_info = self.cpp_api.get_system_info()
                info['cpp_system_info'] = cpp_info
                info['memory_info'] = self.cpp_api.get_memory_info()
                info['build_info'] = cpp_backend.get_build_info()
            except Exception as e:
                info['cpp_error'] = str(e)
        
        return info
    
    def create_disk_image(self, 
                         source_path: str, 
                         destination_path: str,
                         sector_size: int = 512,
                         verify: bool = True,
                         compression: bool = False,
                         progress_callback: Optional[Callable] = None) -> OperationResult:
        """
        Create disk image using the best available backend
        """
        start_time = time.time()
        
        if self.logger:
            self.logger.info(f"Creating disk image: {source_path} -> {destination_path}")
        
        try:
            # Try C++ backend first if preferred and available
            if self.prefer_cpp and self.cpp_api:
                try:
                    imager = self.cpp_api.create_disk_imager()
                    if compression:
                        imager.set_compression_enabled(True)
                    
                    # Convert Python callback to C++ compatible format
                    cpp_callback = None
                    if progress_callback:
                        def cpp_progress_wrapper(percentage, message):
                            try:
                                progress_callback(percentage, message)
                            except Exception as e:
                                if self.logger:
                                    self.logger.error(f"Progress callback error: {e}")
                        cpp_callback = cpp_progress_wrapper
                    
                    result = imager.create_image(source_path, destination_path, 
                                               sector_size, cpp_callback)
                    
                    elapsed_time = time.time() - start_time
                    
                    if result.success:
                        if verify:
                            if self.logger:
                                self.logger.info("Verifying created image...")
                            verify_result = imager.verify_image(destination_path)
                            if not verify_result.success:
                                return OperationResult(
                                    success=False,
                                    message=f"Image verification failed: {verify_result.error_message}",
                                    elapsed_time=elapsed_time
                                )
                        
                        return OperationResult(
                            success=True,
                            message="Disk image created successfully",
                            data={
                                'image_path': result.image_path,
                                'total_bytes': result.total_bytes,
                                'total_sectors': result.total_sectors,
                                'bad_sector_count': result.bad_sector_count,
                                'md5_hash': result.md5_hash,
                                'sha256_hash': result.sha256_hash,
                                'backend': 'C++'
                            },
                            elapsed_time=elapsed_time,
                            bytes_processed=result.total_bytes
                        )
                    else:
                        return OperationResult(
                            success=False,
                            message=result.error_message,
                            elapsed_time=elapsed_time
                        )
                        
                except Exception as e:
                    if self.logger:
                        self.logger.warning(f"C++ backend failed, falling back to Python: {e}")
            
            # Fall back to Python implementation
            if PYTHON_MODULES_AVAILABLE:
                try:
                    imager = PythonDiskImager()
                    metadata = imager.create_image(source_path, destination_path, sector_size)
                    
                    elapsed_time = time.time() - start_time
                    
                    return OperationResult(
                        success=True,
                        message="Disk image created successfully",
                        data={
                            'image_path': destination_path,
                            'total_bytes': metadata.total_sectors * sector_size,
                            'total_sectors': metadata.total_sectors,
                            'bad_sector_count': len(metadata.bad_sectors),
                            'md5_hash': metadata.md5_hash,
                            'sha256_hash': metadata.sha256_hash,
                            'backend': 'Python'
                        },
                        elapsed_time=elapsed_time,
                        bytes_processed=metadata.total_sectors * sector_size
                    )
                    
                except Exception as e:
                    return OperationResult(
                        success=False,
                        message=f"Python backend failed: {str(e)}",
                        elapsed_time=time.time() - start_time
                    )
            
            # No backend available
            return OperationResult(
                success=False,
                message="No suitable backend available for disk imaging",
                elapsed_time=time.time() - start_time
            )
            
        except Exception as e:
            return OperationResult(
                success=False,
                message=f"Unexpected error: {str(e)}",
                error_code="UNEXPECTED_ERROR",
                elapsed_time=time.time() - start_time
            )
    
    def carve_files(self,
                   image_path: str,
                   output_directory: str,
                   signatures_path: Optional[str] = None,
                   parallel: bool = True,
                   max_workers: int = 0,
                   file_types: Optional[List[str]] = None,
                   progress_callback: Optional[Callable] = None) -> OperationResult:
        """
        Carve files from disk image using the best available backend
        """
        start_time = time.time()
        
        if self.logger:
            self.logger.info(f"Carving files from: {image_path}")
        
        try:
            # Try C++ backend first if preferred and available
            if self.prefer_cpp and self.cpp_api:
                try:
                    carver = self.cpp_api.create_file_carver()
                    
                    if file_types:
                        carver.set_file_type_filter(file_types)
                    
                    # Convert Python callback to C++ compatible format
                    cpp_progress_callback = None
                    cpp_file_found_callback = None
                    
                    if progress_callback:
                        def cpp_progress_wrapper(percentage, message):
                            try:
                                progress_callback(percentage, message)
                            except Exception as e:
                                if self.logger:
                                    self.logger.error(f"Progress callback error: {e}")
                        cpp_progress_callback = cpp_progress_wrapper
                    
                    if parallel and max_workers > 0:
                        result = carver.carve_files_parallel(
                            image_path, output_directory, max_workers,
                            signatures_path or "", cpp_progress_callback, cpp_file_found_callback
                        )
                    else:
                        result = carver.carve_files(
                            image_path, output_directory, signatures_path or "",
                            cpp_progress_callback, cpp_file_found_callback
                        )
                    
                    elapsed_time = time.time() - start_time
                    
                    if result.success:
                        return OperationResult(
                            success=True,
                            message="File carving completed successfully",
                            data={
                                'carved_files': result.carved_files,
                                'total_files_found': result.total_files_found,
                                'valid_files': result.valid_files,
                                'output_directory': output_directory,
                                'backend': 'C++'
                            },
                            elapsed_time=elapsed_time,
                            files_processed=result.total_files_found,
                            bytes_processed=result.total_bytes_recovered
                        )
                    else:
                        return OperationResult(
                            success=False,
                            message=result.error_message,
                            elapsed_time=elapsed_time
                        )
                        
                except Exception as e:
                    if self.logger:
                        self.logger.warning(f"C++ backend failed, falling back to Python: {e}")
            
            # Fall back to Python implementation
            if PYTHON_MODULES_AVAILABLE:
                try:
                    carver = PythonFileCarver()
                    
                    if parallel:
                        carved_files = carver.carve_parallel(image_path, signatures_path or 'signatures.json', output_directory)
                    else:
                        carved_files = carver.carve(image_path, signatures_path or 'signatures.json', output_directory)
                    
                    elapsed_time = time.time() - start_time
                    
                    return OperationResult(
                        success=True,
                        message="File carving completed successfully",
                        data={
                            'carved_files': carved_files,
                            'total_files_found': len(carved_files),
                            'output_directory': output_directory,
                            'backend': 'Python'
                        },
                        elapsed_time=elapsed_time,
                        files_processed=len(carved_files)
                    )
                    
                except Exception as e:
                    return OperationResult(
                        success=False,
                        message=f"Python backend failed: {str(e)}",
                        elapsed_time=time.time() - start_time
                    )
            
            # No backend available
            return OperationResult(
                success=False,
                message="No suitable backend available for file carving",
                elapsed_time=time.time() - start_time
            )
            
        except Exception as e:
            return OperationResult(
                success=False,
                message=f"Unexpected error: {str(e)}",
                error_code="UNEXPECTED_ERROR",
                elapsed_time=time.time() - start_time
            )
    
    def rebuild_videos(self,
                      image_path: str,
                      output_directory: str,
                      video_format: str = "mov",
                      quality_threshold: float = 0.7,
                      progress_callback: Optional[Callable] = None) -> OperationResult:
        """
        Rebuild video files using the best available backend
        """
        start_time = time.time()
        
        if self.logger:
            self.logger.info(f"Rebuilding {video_format} videos from: {image_path}")
        
        try:
            # Try C++ backend first if preferred and available
            if self.prefer_cpp and self.cpp_api:
                try:
                    rebuilder = self.cpp_api.create_video_rebuilder()
                    rebuilder.set_quality_threshold(quality_threshold)
                    
                    # Convert Python callback to C++ compatible format
                    cpp_progress_callback = None
                    cpp_video_found_callback = None
                    
                    if progress_callback:
                        def cpp_progress_wrapper(percentage, message):
                            try:
                                progress_callback(percentage, message)
                            except Exception as e:
                                if self.logger:
                                    self.logger.error(f"Progress callback error: {e}")
                        cpp_progress_callback = cpp_progress_wrapper
                    
                    result = rebuilder.rebuild_videos(
                        image_path, output_directory, video_format,
                        cpp_progress_callback, cpp_video_found_callback
                    )
                    
                    elapsed_time = time.time() - start_time
                    
                    if result.success:
                        return OperationResult(
                            success=True,
                            message="Video rebuilding completed successfully",
                            data={
                                'rebuilt_videos': result.rebuilt_videos,
                                'total_videos_found': result.total_videos_found,
                                'successful_rebuilds': result.successful_rebuilds,
                                'output_directory': output_directory,
                                'backend': 'C++'
                            },
                            elapsed_time=elapsed_time,
                            files_processed=result.total_videos_found,
                            bytes_processed=result.total_bytes_processed
                        )
                    else:
                        return OperationResult(
                            success=False,
                            message=result.error_message,
                            elapsed_time=elapsed_time
                        )
                        
                except Exception as e:
                    if self.logger:
                        self.logger.warning(f"C++ backend failed, falling back to Python: {e}")
            
            # Fall back to Python implementation
            if PYTHON_MODULES_AVAILABLE:
                try:
                    rebuilder = PythonVideoRebuilder()
                    
                    if video_format.lower() == 'mov':
                        rebuilt_videos = rebuilder.rebuild_canon_mov(image_path, output_directory)
                    else:
                        raise ValueError(f"Unsupported video format: {video_format}")
                    
                    elapsed_time = time.time() - start_time
                    
                    return OperationResult(
                        success=True,
                        message="Video rebuilding completed successfully",
                        data={
                            'rebuilt_videos': rebuilt_videos,
                            'total_videos_found': len(rebuilt_videos),
                            'output_directory': output_directory,
                            'backend': 'Python'
                        },
                        elapsed_time=elapsed_time,
                        files_processed=len(rebuilt_videos)
                    )
                    
                except Exception as e:
                    return OperationResult(
                        success=False,
                        message=f"Python backend failed: {str(e)}",
                        elapsed_time=time.time() - start_time
                    )
            
            # No backend available
            return OperationResult(
                success=False,
                message="No suitable backend available for video rebuilding",
                elapsed_time=time.time() - start_time
            )
            
        except Exception as e:
            return OperationResult(
                success=False,
                message=f"Unexpected error: {str(e)}",
                error_code="UNEXPECTED_ERROR",
                elapsed_time=time.time() - start_time
            )
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics from all backends"""
        stats = {
            'unified_api': {
                'preferred_backend': 'C++' if self.prefer_cpp else 'Python',
                'cpp_available': CPP_AVAILABLE,
                'python_available': PYTHON_MODULES_AVAILABLE
            }
        }
        
        if CPP_AVAILABLE and self.cpp_api:
            try:
                stats['cpp_backend'] = self.cpp_api.get_global_statistics()
            except Exception as e:
                stats['cpp_backend_error'] = str(e)
        
        return stats
    
    def cleanup(self):
        """Cleanup resources"""
        if self.cpp_api:
            try:
                self.cpp_api.shutdown()
            except Exception as e:
                if self.logger:
                    self.logger.error(f"Error shutting down C++ API: {e}")

def main():
    """Main CLI interface"""
    parser = argparse.ArgumentParser(
        description='PhoenixDRS Professional - Unified Python-C++ Forensics Suite',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  phoenixdrs_unified.py system-info
  phoenixdrs_unified.py image --source /dev/sdb --dest evidence.dd --verify
  phoenixdrs_unified.py carve --image evidence.dd --output carved/ --parallel
  phoenixdrs_unified.py rebuild-video --image evidence.dd --output videos/ --format mov
        """
    )
    
    parser.add_argument('--prefer-python', action='store_true',
                       help='Prefer Python backend over C++')
    parser.add_argument('--no-logging', action='store_true',
                       help='Disable logging')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose output')
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # System info command
    subparsers.add_parser('system-info', help='Show system information')
    
    # Image command
    image_parser = subparsers.add_parser('image', help='Create disk image')
    image_parser.add_argument('--source', required=True, help='Source device')
    image_parser.add_argument('--dest', required=True, help='Destination file')
    image_parser.add_argument('--sector-size', type=int, default=512, help='Sector size')
    image_parser.add_argument('--verify', action='store_true', help='Verify image after creation')
    image_parser.add_argument('--compress', action='store_true', help='Enable compression')
    
    # Carve command
    carve_parser = subparsers.add_parser('carve', help='Carve files from image')
    carve_parser.add_argument('--image', required=True, help='Disk image file')
    carve_parser.add_argument('--output', required=True, help='Output directory')
    carve_parser.add_argument('--signatures', help='Signatures file')
    carve_parser.add_argument('--parallel', action='store_true', help='Enable parallel processing')
    carve_parser.add_argument('--workers', type=int, default=0, help='Number of workers')
    carve_parser.add_argument('--types', help='File types to carve (comma-separated)')
    
    # Video rebuild command
    video_parser = subparsers.add_parser('rebuild-video', help='Rebuild video files')
    video_parser.add_argument('--image', required=True, help='Disk image file')
    video_parser.add_argument('--output', required=True, help='Output directory')
    video_parser.add_argument('--format', default='mov', help='Video format')
    video_parser.add_argument('--quality', type=float, default=0.7, help='Quality threshold')
    
    # Statistics command
    subparsers.add_parser('stats', help='Show statistics')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    # Initialize unified API
    api = UnifiedForensicsAPI(
        prefer_cpp=not args.prefer_python,
        enable_logging=not args.no_logging
    )
    
    try:
        if args.command == 'system-info':
            info = api.get_system_info()
            print(json.dumps(info, indent=2, default=str))
            return 0
        
        elif args.command == 'image':
            def progress_callback(percentage, message):
                if args.verbose:
                    print(f"\rProgress: {percentage:.1f}% - {message}", end='', flush=True)
            
            result = api.create_disk_image(
                args.source, args.dest, args.sector_size,
                args.verify, args.compress, progress_callback
            )
            
            if args.verbose:
                print()  # New line after progress bar
            
            if result.success:
                print(f"✓ {result.message}")
                if result.data:
                    print(f"  Backend: {result.data.get('backend', 'Unknown')}")
                    print(f"  Total bytes: {result.data.get('total_bytes', 0):,}")
                    print(f"  Bad sectors: {result.data.get('bad_sector_count', 0)}")
                    print(f"  MD5: {result.data.get('md5_hash', 'N/A')}")
                    print(f"  SHA256: {result.data.get('sha256_hash', 'N/A')}")
                print(f"  Time: {result.elapsed_time:.2f}s")
                return 0
            else:
                print(f"✗ {result.message}")
                return 1
        
        elif args.command == 'carve':
            def progress_callback(percentage, message):
                if args.verbose:
                    print(f"\rProgress: {percentage:.1f}% - {message}", end='', flush=True)
            
            file_types = args.types.split(',') if args.types else None
            
            result = api.carve_files(
                args.image, args.output, args.signatures,
                args.parallel, args.workers, file_types, progress_callback
            )
            
            if args.verbose:
                print()  # New line after progress bar
            
            if result.success:
                print(f"✓ {result.message}")
                if result.data:
                    print(f"  Backend: {result.data.get('backend', 'Unknown')}")
                    print(f"  Files found: {result.data.get('total_files_found', 0)}")
                    print(f"  Valid files: {result.data.get('valid_files', 0)}")
                print(f"  Time: {result.elapsed_time:.2f}s")
                return 0
            else:
                print(f"✗ {result.message}")
                return 1
        
        elif args.command == 'rebuild-video':
            def progress_callback(percentage, message):
                if args.verbose:
                    print(f"\rProgress: {percentage:.1f}% - {message}", end='', flush=True)
            
            result = api.rebuild_videos(
                args.image, args.output, args.format,
                args.quality, progress_callback
            )
            
            if args.verbose:
                print()  # New line after progress bar
            
            if result.success:
                print(f"✓ {result.message}")
                if result.data:
                    print(f"  Backend: {result.data.get('backend', 'Unknown')}")
                    print(f"  Videos found: {result.data.get('total_videos_found', 0)}")
                    print(f"  Successful rebuilds: {result.data.get('successful_rebuilds', 0)}")
                print(f"  Time: {result.elapsed_time:.2f}s")
                return 0
            else:
                print(f"✗ {result.message}")
                return 1
        
        elif args.command == 'stats':
            stats = api.get_statistics()
            print(json.dumps(stats, indent=2, default=str))
            return 0
        
        else:
            parser.print_help()
            return 1
    
    except KeyboardInterrupt:
        print("\n\nOperation cancelled by user")
        return 130
    
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        if args.verbose:
            traceback.print_exc()
        return 1
    
    finally:
        api.cleanup()

if __name__ == "__main__":
    sys.exit(main())